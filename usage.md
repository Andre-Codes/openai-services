# openai-services — Usage Examples (from simple to spicy)

> This doc shows how to use the new `gpt_util` surface—starting with the “hello world” and ramping up to tools, structured outputs, and history handling. All examples are Python.

---

## 0) Install & setup

```bash
pip install -U openai
# Your repo/package with the refactor installed/available
# export environment variables:
export OPENAI_API_KEY=sk-...
# optional: override defaults
export OPENAI_DEFAULT_MODEL=gpt-4o
export OPENAI_TEMPERATURE=0.2
```

```python
from openai_services import get_text, get_structured, chat, call_with_tools
from openai_services.history import RollingTokenHistory
from openai_services.tools import ToolSpec
```

---

## 1) The tiniest possible call (plain text)

```python
res = get_text("Explain weighted averages like I’m busy and hungry.")
print(res["text"])
```

* Returns a dict: `{"text", "usage", "request_id", "model"}`
* **Streaming is off by default**; you get a single, final answer.

---

## 2) Multiple prompts → multiple user turns

```python
res = get_text([
    "You are a helpful explainer.",
    "Now, summarize K-means in 2 lines."
])
print(res["text"])
```

* A list of strings becomes **multiple user turns** in order.

---

## 3) Mixed text + images in one turn

```python
img_url = "https://example.com/plot.png"
res = get_text(["Caption this figure for a report:", img_url])
print(res["text"])
```

* Mixed inputs coalesce into **one `user` message** with text + an image part.

---

## 4) A structured “conversation” object

```python
conversation = {
    "system": "You are terse and precise.",
    "developer": "If the user asks for code, prefer Python.",
    "history": [
        {"role": "user", "content": [{"type":"input_text", "text":"Give me 3 facts about Mars"}]},
        {"role": "assistant", "content": [{"type":"output_text", "text":"• It’s dusty\n• It’s cold\n• It’s photogenic"}]},
    ],
    "user": "Now give me 1 more fact."
}
res = get_text(conversation)
print(res["text"])
```

* System + developer messages are preserved.
* Adjacent content from the same role is merged cleanly.

---

## 5) Guaranteed JSON with Structured Outputs (strict mode)

```python
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "estimated_minutes": {"type": "integer", "minimum": 1}
    },
    "required": ["title", "topics"],
    "additionalProperties": False
}

res = get_structured("Create a study plan about decision trees.", schema=schema)
plan = res["json"]     # already parsed & validated
print(plan["title"], plan["topics"])
```

* If the model adds extra keys or returns non-JSON, you’ll get a **SchemaValidationError** with a helpful pointer.
* Uses OpenAI Structured Outputs under the hood (JSON Schema + strict mode).

---

## 6) Bring-your-own messages (you handle the roles)

```python
msgs = [
    {"role":"system","content":[{"type":"input_text","text":"You are brief."}]},
    {"role":"user","content":[{"type":"input_text","text":"List 3 fast sorting algorithms"}]},
]
res = chat(msgs)
print(res["text"])
```

* `chat()` assumes you already built valid `messages[]`.
* We still trim history if you attach a `HistoryAdapter`.

---

## 7) History control with a rolling window

```python
history = RollingTokenHistory(max_turns=8)  # simple heuristic by turns

res = get_text(
    ["We’ll talk a lot.", "Remember I prefer examples."],
    history=history
)

res = get_text(
    "Now explain Bloom filters with an example.",
    history=history
)

print(res["text"])
```

* The adapter keeps system/developer messages and the last `N` turns.
* Swap in a tokenizer-backed adapter later if you want exact token budgets.

---

## 8) Tool calling (manual execution)

```python
import json
from datetime import datetime
from openai_services import call_with_tools
from openai_services.tools import ToolSpec

def weather_lookup(args):
    # pretend to call a real weather API
    city = args["city"]
    units = args.get("units", "F")
    return {"city": city, "units": units, "temp": 97, "as_of": datetime.utcnow().isoformat() + "Z"}

weather_tool = ToolSpec(
    name="get_weather",
    description="Get current weather by city name.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type":"string"},
            "units": {"type":"string", "enum":["F","C"]}
        },
        "required": ["city"],
        "additionalProperties": False
    }
    # handler omitted: we'll execute manually in this example
)

prompt = "What's the weather in Dallas right now? Return the tool arguments only."
res = call_with_tools(prompt, tools=[weather_tool])

print("Tool calls:", res["tool_calls"])     # → [{"id","name","args":{...}}]
print("Model follow-up (if any):", res["follow_up"])

# MANUAL EXECUTION:
calls = res["tool_calls"]
if calls:
    result = weather_lookup(calls[0]["args"])
    # Pass the result back as a tool message and ask for a final summary:
    tool_msg = {
        "role": "tool",
        "tool_call_id": calls[0]["id"],
        "content": [{"type":"output_text","text": json.dumps(result)}]
    }
    # Build a new conversation and call get_text/chat again:
    final = chat([
        {"role":"user","content":[{"type":"input_text","text": prompt}]},
        tool_msg
    ])
    print("Final:", final["text"])
```

* By default, `call_with_tools` does **not** execute handlers (safe, no side effects).
* You get normalized `tool_calls` (name + parsed JSON args). You decide what to do.

---

## 9) Tool calling with `auto_execute=True` (batteries included)

```python
email_log = []

def send_email(args):
    # Super fake “email”
    email_log.append({"to": args["to"], "subject": args["subject"], "body": args["body"]})
    return {"status":"queued","id": len(email_log)}

email_tool = ToolSpec(
    name="send_email",
    description="Send an email.",
    parameters={
        "type":"object",
        "properties":{
            "to":{"type":"string","format":"email"},
            "subject":{"type":"string"},
            "body":{"type":"string"}
        },
        "required":["to","subject","body"],
        "additionalProperties": False
    },
    handler=send_email   # now we *can* auto-execute
)

res = call_with_tools(
    "Draft and send a short email to alice@example.com about our meeting at 3pm.",
    tools=[email_tool],
    auto_execute=True,       # <- do the side effects
    max_round_trips=1        # model may do one follow-up, then we stop
)

print("Tool calls:", res["tool_calls"])
print("Follow-up:", res["follow_up"])
print("Email log:", email_log)
```

* Great for internal scripts. For prod systems, you might keep manual control.

---

## 10) Structured outputs + tools (combo move)

```python
schema = {
    "type":"object",
    "properties":{
        "city":{"type":"string"},
        "temp":{"type":"number"},
        "units":{"type":"string", "enum":["F","C"]},
        "advice":{"type":"string"}
    },
    "required":["city","temp","units"],
    "additionalProperties": False
}

weather = ToolSpec(
    name="get_weather",
    description="Weather by city.",
    parameters={
        "type":"object",
        "properties":{"city":{"type":"string"},"units":{"type":"string","enum":["F","C"]}},
        "required":["city"], "additionalProperties": False
    },
    handler=lambda args: {"city": args["city"], "units": args.get("units","F"), "temp": 93}
)

# 1) Ask the model to decide city → tool call; 2) execute; 3) ask for structured JSON summary.
step1 = call_with_tools("Get the weather for Austin and give me a short recommendation.", tools=[weather], auto_execute=True)
summary_prompt = {
    "system": "Return structured JSON describing the weather and a one-line advice.",
    "user": step1["follow_up"] or "Summarize the weather."
}
res = get_structured(summary_prompt, schema)
print(res["json"])
```

* The model uses the tool; you then **commit** to strict JSON for downstream code.

---

## 11) Error handling patterns

```python
from openai_services.errors import NormalizationError, SchemaValidationError, ToolCallError, RateLimitError

try:
    res = get_structured("Return only the keys I asked for.", schema={"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":False})
    print(res["json"])
except SchemaValidationError as e:
    print("Schema error at:", e.json_pointer, "Partial:", (e.partial_text or "")[:120])

try:
    # Bad input on purpose
    res = get_text([{"role":"wizard","content":[{"type":"input_text","text":"oops"}]}])
except NormalizationError as e:
    print("Normalization failed:", e)

try:
    # Simulate loud tool args
    raise ToolCallError("Tool arguments not valid JSON at $.coords")
except ToolCallError as e:
    print("Tool error:", e)
```

* Errors include **helpful context** (path, partial text).

---

## 12) Timeouts & retries (tuning knobs)

```python
# bump retries to 6 attempts, backoff base to 0.8s
res = get_text("Be concise: what is MAPE?", retry=(6, 0.8))
print(res["text"])

# model control per-call
res = get_text("Explain XGBoost in one sentence.", model="gpt-4o-mini", temperature=0.0)
print(res["text"])
```

* Defaults come from env/config; you can override per call.

---

## 13) Optional streaming (when you really want it)

> Streaming is **off** by default. If you enable it in your app, you’ll get an event iterator compatible with the tiny stream router shipped in the library.

*Pseudo-usage sketch* (the actual API depends on your app’s integration):

```python
# Pseudocode: when you wire stream=True in your app, use the router to assemble text or JSON:
from openai_services.stream_router import TextStreamAssembler

assembler = TextStreamAssembler()
for ev in stream_events():      # ← your SSE/WebSocket loop
    assembler.on_event(ev)

final_text = assembler.final_text()
print(final_text)
```

* For structured JSON, use the `JsonStreamAssembler` and validate once the stream completes.
* For tools, route via `ToolEventRouter` to get `{type:"tool-call-complete", id, name, args}` events.

---

## 14) Debugging the “what did I actually send?” question

```python
from openai_services.message_normalizer import format_messages

msgs = format_messages(["Summarize this:", "And be kind."])
for m in msgs:
    print(m["role"], m["content"])
```

* Use this to **inspect** the normalized `messages[]` your call will send.
* The normalizer is deterministic: system → developer → turns; adjacent user content is merged.

---

## 15) A tiny end-to-end: vision → tool → structured output

```python
# 1) Caption an image
img = "https://example.com/menu.jpg"
caption = get_text(["Describe the key items on this menu:", img])["text"]

# 2) Price lookup tool (fake)
prices = {"espresso": 3.25, "latte": 4.25}
lookup = ToolSpec(
    name="lookup_price",
    description="Get price by item name.",
    parameters={"type":"object","properties":{"item":{"type":"string"}},"required":["item"],"additionalProperties":False},
    handler=lambda args: {"item": args["item"], "price": prices.get(args["item"].lower(), None)}
)

# Ask model to pick one item and check the price:
step = call_with_tools(f"From this caption: {caption}\nPick one item and look up the price.", tools=[lookup], auto_execute=True)

# 3) Structured output for downstream code
schema = {
    "type":"object",
    "properties":{
        "item":{"type":"string"},
        "price":{"type":["number","null"]},
        "note":{"type":"string"}
    },
    "required":["item","price"],
    "additionalProperties": False
}

summary = get_structured(
    {"system":"Return a single JSON object.", "user": step["follow_up"] or "Summarize price."},
    schema=schema
)

print(summary["json"])
```