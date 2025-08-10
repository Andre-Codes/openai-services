# openai-services — Comprehensive Guide

This is your one-stop manual for the refactored **chat-first** module. It’s designed to be intuitive: pass in nearly anything—strings, lists, images, full conversations—and get a clean result without babysitting the OpenAI SDK.

---

## What this library gives you

* **Simple “get” functions**

  * `get_text(...)` → plain text back (non-streaming by default)
  * `get_structured(...)` → strict JSON back (schema-validated)
* **Low-level `chat(...)`** for advanced, role-tagged message arrays
* **Stateful history** with `ChatHistory` (automatic memory)
* **Stateless trimming** with `RollingTokenHistory` (lightweight budgets)
* **Optional streaming** with sinks for terminal, Jupyter, Streamlit, or custom GUIs
* **Vision (image understanding)** via image inputs to `get_text`/`get_structured`
* **(Separate) Image generation** using the Images API (outside chat\_util)

Everything defaults to safe, predictable behavior:

* **Streaming OFF** unless you ask.
* The default model is resolved dynamically via `DefaultModel()`.
* Structured outputs use a **tool-enforced JSON** pattern so you get real JSON or a clear error.

---

# 1) Setup

```bash
pip install -U openai
# set your key
export OPENAI_API_KEY=sk-...
# optional env tuning
export OPENAI_DEFAULT_MODEL=gpt-4o-mini
export OPENAI_TIMEOUT=30
export OPENAI_MAX_OUTPUT_TOKENS=1024
export OPENAI_TEMPERATURE=0.2
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

In code:

```python
from openai_services import get_text, get_structured, chat
from openai_services.history import ChatHistory, RollingTokenHistory
from openai_services.sinks import terminal_sink, jupyter_sink, streamlit_sink, buffer_sink
from openai_services.models import MODELS, DefaultModel
```

---

# 2) Inputs & Message Normalization (what you can pass)

The “get” functions accept flexible inputs. Internally we normalize them into the Responses API format:

### Plain string

```python
get_text("Summarize K-means in 2 lines.")
```

### List of strings → multiple user turns (in order)

```python
get_text(["First, define entropy.", "Now, explain information gain."])
```

### Mixed text + images → one user turn with multiple parts

```python
img = "https://example.com/plot.png"
get_text(["Caption this figure for a report:", img], model="gpt-4o")
```

You can also pass images as dicts with base64:

```python
image_part = {"image_data": {"b64": base64_png, "mime_type": "image/png"}}
get_text(["What’s in this image?", image_part], model="gpt-4o")
```

### “Conversation” object (system/developer/history + user/images)

```python
conversation = {
  "system": "You are terse.",
  "developer": "Prefer Python examples.",
  "history": [
    {"role":"user","content":[{"type":"input_text","text":"Three facts about Mars"}]},
    {"role":"assistant","content":[{"type":"output_text","text":"Dusty, cold, photogenic"}]},
  ],
  "user": "Give me one more fact."
}
get_text(conversation)
```

### Already role-tagged messages

```python
messages = [
  {"role":"system","content":[{"type":"input_text","text":"You are brief."}]},
  {"role":"user","content":[{"type":"input_text","text":"List 3 fast sorting algorithms."}]},
]
chat(messages)
```

**Deterministic rules** (so you’re never surprised):

* `system` then `developer`, then alternating `user/assistant` turns.
* Adjacent user content is coalesced to avoid token waste.
* Images become `{"type":"input_image", "image_url": "https://..."}` or `{"image_data": {...}}`.
* Unknown roles throw a clear `NormalizationError`.

---

# 3) Text mode (`get_text`)

The simplest path:

```python
res = get_text("Explain cosine similarity in one sentence.")
print(res["text"])
```

Returns:

* `text` (final answer)
* `usage` (token summary if provided)
* `request_id`
* `model`

---

# 4) Vision mode (image understanding)

Any “get” call can be multimodal if you include image parts. Use a **vision-capable model** like `gpt-4o` or `gpt-4o-mini`.

```python
res = get_text(["Describe this image in 2 lines.", "https://example.com/menu.jpg"], model="gpt-4o")
print(res["text"])
```

For structured OCR/parse tasks:

```python
schema = {
  "type":"object",
  "properties":{"items":{"type":"array","items":{"type":"string"}}},
  "required":["items"],
  "additionalProperties": False
}
res = get_structured({"user":"Extract menu items as a list.","images":["https://.../menu.jpg"]}, schema, model="gpt-4o")
print(res["json"]["items"])
```

---

# 5) Structured outputs (`get_structured`)

You pass a **JSON Schema**, you get **validated JSON** back or a neat `SchemaValidationError`.

```python
schema = {
  "type":"object",
  "properties":{
    "title":{"type":"string"},
    "topics":{"type":"array","items":{"type":"string"}},
    "estimated_minutes":{"type":"integer","minimum":1}
  },
  "required":["title","topics"],
  "additionalProperties": False
}
res = get_structured("Create a super short study plan about decision trees.", schema=schema)
print(res["json"])
```

**Under the hood:** since your SDK version doesn’t support `response_format`, we use a tool-enforced JSON pattern so the model returns strict JSON in function-call arguments.

---

# 6) Image generation (creating images)

Image *generation* is separate from chat. Use the OpenAI Images API directly (not part of `chat_util`):

```python
from openai import OpenAI
client = OpenAI()

img = client.images.generate(
  model="gpt-image-1",
  prompt="A cozy reading nook with plants and a cat"
)
# img.data[0].url or b64_json depending on your settings
```

If you want, you can wrap this in your module as `generate_image(prompt, ...)`. It’s intentionally separate from “vision,” which is image understanding inside `get_text/get_structured`.

---

# 7) Streaming

Streaming is **off by default**. When you want it, pass `stream=True` and choose a **sink**:

### Terminal

```python
from openai_services import get_text, terminal_sink

res = get_text("Stream like a typewriter.", stream=True, sink=terminal_sink())
print("\n[done]", res["request_id"])
```

### Jupyter (renders Markdown live)

```python
from openai_services import get_text, jupyter_sink

sink = jupyter_sink()
res = get_text("• Stream bullet points\n• As Markdown\n• Thanks", stream=True, sink=sink)
res["usage"]
```

### Streamlit

```python
import streamlit as st
from openai_services import get_text, streamlit_sink

ph = st.empty()
res = get_text("Explain Bloom filters (live).", stream=True, sink=streamlit_sink(ph))
st.caption(res["usage"])
```

### Custom GUIs (e.g., Shiny via `buffer_sink`)

```python
chunks = []
res = get_text("Stream into a list buffer.", stream=True, sink=buffer_sink(chunks))
final_text = "".join(chunks)
```

What the sinks do:

* `terminal_sink()` writes chunks to stdout as they arrive.
* `jupyter_sink()` updates the same output cell with **Markdown** each chunk.
* `streamlit_sink()` updates a placeholder via `.markdown()`.
* `buffer_sink(list)` appends chunks to your container; you decide how to render.

---

# 8) History & Conversations

You have two complementary ways to manage context. Use them together: **ChatHistory** gives you a growing transcript, and **RollingTokenHistory** trims that transcript before each API call so you don’t blow the context window.

---

## A) **ChatHistory** — stateful, automatic memory (recommended)

Create one `ChatHistory` and pass it to every `get_text` / `get_structured` call. The library will:

1. take your new input,
2. **append a user turn** to `ChatHistory`,
3. call the API,
4. **append the assistant’s reply**,
5. (optionally) chain server-side state (see below).

```python
from openai_services.history import ChatHistory, RollingTokenHistory
from openai_services import get_text

hist = ChatHistory(
    adapter=RollingTokenHistory(max_turns=8),   # trims before each call
    system="Be concise."                        # always prepended
)

get_text("Remember: one-word answers.", conversation=hist)
res = get_text("How would you describe a burger?", conversation=hist)
print(res["text"])  # e.g., "Juicy."
```

### What’s inside `ChatHistory`?

* **Headers**: optional `system` and `developer` messages (sent every time).
* **Turns**: alternating `user`/`assistant` messages that we add for you.
* **Adapter**: a pluggable **trimmer** (see RollingTokenHistory below).
* **Server-side chain state**: `previous_response_id` (optional).

`ChatHistory.messages()` constructs the **current payload** as:

```
[system?, developer?] + adapter.trim(all_turns)
```

We never mutate your headers, only the growing turns list.

### Useful operations

```python
hist.reset()                           # clear all turns + server-side chain id
hist.system = "Be *very* concise."     # update header anytime
# If you need to inspect what will be sent:
for m in hist.messages(): print(m["role"], m["content"][:1])
```

> Tip: Want a brand-new conversational fork? Create another `ChatHistory` and start passing that one instead. No need to copy turns unless you want a true branch; otherwise just start fresh.

---

## B) **RollingTokenHistory** — stateless trimming (used by ChatHistory)

`RollingTokenHistory(max_turns=N)` is a **pure adapter**. It **doesn’t remember** anything by itself; it just takes a message list and returns a trimmed version:

* Keeps **system/developer** headers.
* Keeps the **last N “turns”** of the conversation (a “turn” is one message—user or assistant—so `max_turns=8` roughly means “last \~4 exchanges”).
* Simple, fast, predictable. Great default.

How it works with `ChatHistory`:

* You keep calling `get_text(..., conversation=hist)`.
* Before each API call, we run `adapter.trim(hist._turns)` to cap context size.
* The older turns are dropped from **that request**, but your full transcript still lives inside `hist._turns` if you want it (until you `reset()`).

**Example (advanced / low-level):**

```python
from openai_services import chat
from openai_services.history import RollingTokenHistory

adapter = RollingTokenHistory(max_turns=8)
msgs = [
  {"role":"system","content":[{"type":"input_text","text":"Be concise."}]},
  {"role":"user","content":[{"type":"input_text","text":"Remember: one word."}]},
  {"role":"assistant","content":[{"type":"output_text","text":"Understood."}]},
  {"role":"user","content":[{"type":"input_text","text":"Describe a burger."}]},
]
res = chat(msgs, history=adapter)  # adapter trims prior to the call
print(res["text"])
```

**Tuning:**

* For tighter control, lower `max_turns`.
* For longer conversations, raise it (or replace with a tokenizer-based adapter if you need token-precise budgets later).

---

## C) Server-side chaining (optional, additive to ChatHistory)

OpenAI’s Responses API can maintain short-term state if you pass:

* `store=True` on a call (let OpenAI store the result briefly),
* `previous_response_id=<the previous call’s id>` on the next call.

`ChatHistory` can do this **automatically** when you set:

```python
hist.chain_server_side = True
```

### What happens under the hood

* On each call:

  * we include `store=True` and `previous_response_id` (if known),
  * OpenAI stitches context server-side,
  * we still send your **system/developer** headers (best practice),
  * we still keep and trim your **client-side** transcript (belt + suspenders).

### Why use it?

* **Fewer bytes**: you don’t resend the whole conversation every turn.
* **Resilience**: the API can rely on its own state between immediate turns.

### When *not* to rely solely on it

* If you need full auditability or portability of context, keep **client-side history** as the source of truth (which `ChatHistory` already does). Consider server-side chaining an optimization, not your only memory.

### Exact usage

```python
from openai_services import get_text
from openai_services.history import ChatHistory, RollingTokenHistory

hist = ChatHistory(adapter=RollingTokenHistory(8), system="Be concise.")
hist.chain_server_side = True

# First call stores a response id internally
get_text("Remember one-word.", conversation=hist)

# Next call threads previous_response_id automatically
res = get_text("Describe a burger.", conversation=hist)
print(res["text"])
```

**Notes & pitfalls**

* We **always** resend `system`/`developer` headers (good hygiene).
* If you `reset()` the history, the chain id is cleared too (fresh thread).
* If a call fails mid-stream, the chain id might not update; the next successful call will set it again.

---

# 9) `chat()` — low-level control

`chat(messages, ...)` is your “bring your own messages” escape hatch.

```python
msgs = [
  {"role":"system","content":[{"type":"input_text","text":"You are brief."}]},
  {"role":"user","content":[{"type":"input_text","text":"List 3 prime numbers."}]},
]
res = chat(msgs)
print(res["text"])
```

### Why use `chat()`?

* You already have a structured, role-tagged conversation (from a DB, logs, etc.).
* You want to apply a trimming adapter but **not** maintain a `ChatHistory`.
* You need to experiment with unusual role sequences for research/prototyping.

### `chat()` + conversations

If you also want to keep a transcript using `ChatHistory`, you can pass it **optionally** to enable server-side chaining and (optionally) append the assistant reply:

```python
res = chat(
  msgs,
  conversation=hist,                      # enables store/previous_response_id if hist.chain_server_side=True
  append_assistant_to_conversation=True   # add the model’s reply to hist
)
```

By design, `chat()` does **not** merge `hist` turns into `messages` (to avoid duplicates); it treats `messages` as authoritative.

---

# 10) Tools (optional heads-up)

This doc focuses on the “get” functions and `chat()`. If you need function/tool calling:

```python
from openai_services import call_with_tools
from openai_services.tools import ToolSpec

weather = ToolSpec(
  name="get_weather",
  description="Get current weather by city.",
  parameters={
    "type":"object",
    "properties":{"city":{"type":"string"}, "units":{"type":"string","enum":["F","C"]}},
    "required":["city"], "additionalProperties": False
  }
)

res = call_with_tools("What's the weather in Austin?", tools=[weather])
print(res["tool_calls"])         # normalized calls
print(res["follow_up"])          # model message (if any)
```

You can also pass `auto_execute=True` and provide `handler`s for batteries-included flows.

---

# 11) Error handling

* **NormalizationError** — malformed inputs/roles.
* **SchemaValidationError** — structured outputs aren’t valid JSON/shape.
* **ToolCallError** — tool arguments weren’t valid JSON.
* **RateLimitError** — retries exhausted (429/5xx).

Example:

```python
from openai_services.errors import NormalizationError, SchemaValidationError

try:
    get_structured("Return only key 'a'.", schema={
      "type":"object", "properties":{"a":{"type":"string"}}, "required":["a"], "additionalProperties": False
    })
except SchemaValidationError as e:
    print("Schema fail:", e)
```

---

# 12) Models & configuration

* `DefaultModel()` resolves a sensible default (dynamic registry with caching).
* `MODELS.list()` fetches available model IDs (cached, TTL).
* Override models per call: `get_text(..., model="gpt-4o")`.

Environment knobs:

```env
OPENAI_DEFAULT_MODEL=gpt-4o-mini
OPENAI_TIMEOUT=30
OPENAI_MAX_OUTPUT_TOKENS=1024
OPENAI_TEMPERATURE=0.2
OPENAI_RETRY_MAX=3
OPENAI_RETRY_BACKOFF=0.5
# Optional model discovery filters:
OPENAI_MODELS_AUTO_REFRESH=true
OPENAI_MODELS_INCLUDE=gpt-4o.*
OPENAI_MODELS_EXCLUDE=.*preview,.*deprecated
```

---

## Quick recipes

**Minimal chat with memory**

```python
hist = ChatHistory(system="Be concise.")
get_text("Remember: one-word answers.", conversation=hist)
print(get_text("Describe a burger.", conversation=hist)["text"])
```

**Vision**

```python
get_text(["What’s in this image?", "https://example.com/photo.jpg"], model="gpt-4o")
```

**Structured**

```python
schema = {"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":False}
get_structured("Give a title for this chat.", schema)
```

**Streaming in Jupyter**

```python
res = get_text("• Stream me\n• As Markdown", stream=True, sink=jupyter_sink())
```

**Low-level chat with trimming**

```python
adapter = RollingTokenHistory(max_turns=6)
res = chat([...messages...], history=adapter)
```

---

### Final thoughts

* Use **`ChatHistory`** for human conversations—automatic memory, minimal fuss.
* Use **`RollingTokenHistory`** when you want to manually assemble messages and just need a budget-friendly last-N turns policy.
* Flip **streaming** on only where it adds value; pick a sink that fits your environment.
* Keep **structured outputs** for the parts of your app that feed data into code or DBs—it’ll save you from brittle parsing logic.