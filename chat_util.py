# openai_services/chat_util.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from openai import OpenAI
import json
import time
import random
import warnings

from .config import CONFIG
from .models import DefaultModel
from .errors import (
    NormalizationError, SchemaValidationError, ToolCallError,
    RateLimitError,
)
from .message_normalizer import format_messages
from .history import HistoryAdapter, NoHistory, ChatHistory
from .tools import ToolSpec
from .stream_router import TextStreamAssembler, JsonStreamAssembler

Message = Dict[str, Any]
Sink = Callable[[str], None]  # receives text deltas if streaming

# ──────────────────────────────────────────────────────────────────────────────
# Internal client (per-process singleton-ish)
_client: Optional[OpenAI] = None

def _client_get() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=CONFIG.timeout_seconds, **CONFIG.client_kwargs)
    return _client

# ──────────────────────────────────────────────────────────────────────────────
# Retry wrapper with exp backoff + jitter (429/5xx)
def _with_retries(fn, *, attempts: int, backoff_base: float):
    def run():
        delay = backoff_base
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                msg = str(e).lower()
                status = getattr(e, "status", "")
                if ("rate limit" in msg) or ("429" in msg) or (isinstance(status, str) and status.startswith("5")):
                    if i == attempts - 1:
                        raise RateLimitError(f"Retry attempts exhausted: {e}") from e
                    time.sleep(delay + random.random() * delay)
                    delay *= 2.0
                else:
                    raise
    return run()

# ──────────────────────────────────────────────────────────────────────────────
# Streaming helpers

def _stream_text(
    client: OpenAI,
    *,
    model: str,
    messages: List[Message],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    sink: Optional[Sink],
) -> Dict[str, Any]:
    assembler = TextStreamAssembler(on_delta=sink)
    with client.responses.stream(
        model=model,
        input=messages,
        temperature=temperature if temperature is not None else CONFIG.temperature,
        max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
    ) as stream:
        for event in stream:
            assembler.on_event(_normalize_event(event))
        final = stream.get_final_response()
        if getattr(final, "usage", None):
            assembler.usage = final.usage
        if getattr(final, "id", None):
            assembler.request_id = final.id
    return {
        "text": assembler.final_text(),
        "usage": assembler.usage,
        "request_id": assembler.request_id,
        "model": model,
    }

def _stream_json(
    client: OpenAI,
    *,
    model: str,
    messages: List[Message],
    response_format: Dict[str, Any],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    sink: Optional[Sink],
) -> Dict[str, Any]:
    assembler = JsonStreamAssembler(on_delta=sink)
    with client.responses.stream(
        model=model,
        input=messages,
        response_format=response_format,
        temperature=temperature if temperature is not None else CONFIG.temperature,
        max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
    ) as stream:
        for event in stream:
            assembler.on_event(_normalize_event(event))
        final = stream.get_final_response()
        if getattr(final, "usage", None):
            assembler.usage = final.usage
        if getattr(final, "id", None):
            assembler.request_id = final.id
    text = assembler.final_text()
    try:
        data = json.loads(text or "{}")
    except Exception as e:
        raise SchemaValidationError(f"Streamed JSON invalid: {e}", partial_text=text) from e
    return {
        "json": data,
        "usage": assembler.usage,
        "request_id": assembler.request_id,
        "model": model,
    }

def _normalize_event(event: Any) -> Dict[str, Any]:
    # Start with dict if possible
    if isinstance(event, dict):
        out = dict(event)
    elif hasattr(event, "to_dict"):
        try:
            out = event.to_dict()
        except Exception:
            out = {}
    else:
        out = {}

    # Always set type if available
    out["type"] = out.get("type") or getattr(event, "type", None)

    # Normalize common fields
    for attr in ("delta", "arguments_delta", "call_id", "name", "error", "usage", "request_id"):
        if attr not in out and hasattr(event, attr):
            out[attr] = getattr(event, attr)

    # response may be a Pydantic object → extract id
    resp = out.get("response") or getattr(event, "response", None)
    if resp is not None:
        if isinstance(resp, dict):
            out["response"] = {"id": resp.get("id")}
        else:
            out["response"] = {"id": getattr(resp, "id", None)}

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Conversation History helpers

# helper: make user-turn parts from flexible input
def _as_user_parts(input_obj: Any) -> List[Dict[str, Any]]:
    msgs = format_messages(input_obj)
    # Expect one user message (format_messages ensures that for simple inputs)
    if not msgs or msgs[-1].get("role") != "user":
        # If caller passed a full conversation, keep last user turn’s parts
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m["content"]
        # fallback
        return [{"type":"input_text","text": str(input_obj)}]
    return msgs[-1]["content"]

def _build_messages(input_obj: Any, conversation: Optional[ChatHistory], adapter: Optional[HistoryAdapter]) -> List[Message]:
    if conversation:
        base = conversation.messages()
        parts = _as_user_parts(input_obj)
        return base + [{"role":"user","content": parts}]
    # legacy: stateless call with trimming adapter
    msgs = format_messages(input_obj)
    return (adapter or NoHistory()).trim(msgs)

# ──────────────────────────────────────────────────────────────────────────────
# Public API

def get_text(
    input_obj: Any,
    *,
    model: str | None = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    history: Optional[HistoryAdapter] = None,   # kept for backward-compat (stateless trimming)
    conversation: Optional[ChatHistory] = None, # NEW: stateful memory
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    stream: bool = False,
    sink: Optional[Sink] = None,
) -> Dict[str, Any]:
    model = model or DefaultModel()
    client = _client_get()
    attempts, backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    # build user parts & outgoing messages
    user_parts = _as_user_parts(input_obj)
    messages = (
        conversation.messages() + [{"role": "user", "content": user_parts}]
        if conversation else (history or NoHistory()).trim(format_messages(input_obj))
    )

    # server-side chaining flags
    store_flag = bool(conversation and conversation.chain_server_side)
    prev_id = conversation.previous_response_id() if conversation else None

    if stream:
        def _run():
            assembler = TextStreamAssembler(on_delta=sink)
            with client.responses.stream(
                model=model,
                input=messages,
                store=store_flag,
                previous_response_id=prev_id,
                temperature=temperature if temperature is not None else CONFIG.temperature,
                max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
                tool_choice=tool_choice,
            ) as s:
                for ev in s:
                    assembler.on_event(_normalize_event(ev))
                final = s.get_final_response()
                if getattr(final, "usage", None):
                    assembler.usage = final.usage
                if getattr(final, "id", None):
                    assembler.request_id = final.id
            text = assembler.final_text()
            # auto-append to conversation
            if conversation:
                conversation.add_user_parts(user_parts)
                conversation.add_assistant_text(text)
                conversation.set_previous_response_id(assembler.request_id)
            return {"text": text, "usage": assembler.usage, "request_id": assembler.request_id, "model": model}
        return _with_retries(_run, attempts=attempts, backoff_base=backoff)

    def _do():
        resp = client.responses.create(
            model=model,
            input=messages,
            store=store_flag,
            previous_response_id=prev_id,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
            tool_choice=tool_choice,
        )
        text = resp.output_text or ""
        if conversation:
            conversation.add_user_parts(user_parts)
            conversation.add_assistant_text(text)
            conversation.set_previous_response_id(getattr(resp, "id", None))
        return {
            "text": text,
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }
    return _with_retries(_do, attempts=attempts, backoff_base=backoff)

# ---- get_structured ----
def get_structured(
    input_obj: Any,
    schema: Dict[str, Any],
    *,
    model: str | None = None,
    history: Optional[HistoryAdapter] = None,
    conversation: Optional[ChatHistory] = None,  # NEW
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,
    stream: bool = False,
    sink: Optional[Sink] = None,
) -> Dict[str, Any]:
    model = model or DefaultModel()
    client = _client_get()
    attempts, backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    user_parts = _as_user_parts(input_obj)
    messages = (
        conversation.messages() + [{"role": "user", "content": user_parts}]
        if conversation else (history or NoHistory()).trim(format_messages(input_obj))
    )

    # Until your SDK exposes response_format, use the tool-forced JSON pattern:
    tools = [{
        "type": "function",
        "function": {
            "name": "structured_output",
            "description": "Return a JSON object following the provided schema",
            "parameters": {**schema, "additionalProperties": schema.get("additionalProperties", False)},
        },
    }]
    tool_choice = {"type": "function", "function": {"name": "structured_output"}}

    store_flag = bool(conversation and conversation.chain_server_side)
    prev_id = conversation.previous_response_id() if conversation else None

    if stream:
        def _run():
            assembler = JsonStreamAssembler(on_delta=sink)
            with client.responses.stream(
                model=model,
                input=messages,
                tools=tools,
                tool_choice=tool_choice,
                store=store_flag,
                previous_response_id=prev_id,
                temperature=temperature if temperature is not None else CONFIG.temperature,
                max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
            ) as s:
                for ev in s:
                    assembler.on_event(_normalize_event(ev))
                final = s.get_final_response()
                if getattr(final, "usage", None):
                    assembler.usage = final.usage
                if getattr(final, "id", None):
                    assembler.request_id = final.id
            text = assembler.final_text()
            try:
                data = json.loads(text or "{}")
            except Exception as e:
                raise SchemaValidationError(f"Streamed JSON invalid: {e}", partial_text=text) from e
            if conversation:
                # Best-effort: also append assistant text for transcript readability
                conversation.add_user_parts(user_parts)
                conversation.add_assistant_text(text)
                conversation.set_previous_response_id(assembler.request_id)
            return {"json": data, "usage": assembler.usage, "request_id": assembler.request_id, "model": model}
        return _with_retries(_run, attempts=attempts, backoff_base=backoff)

    def _do():
        resp = client.responses.create(
            model=model,
            input=messages,
            tools=tools,
            tool_choice=tool_choice,
            store=store_flag,
            previous_response_id=prev_id,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
        )
        # Parse strict JSON from function args (prefer parsed if SDK returns it)
        args = None
        if getattr(resp, "tool_calls", None):
            for c in resp.tool_calls:
                if getattr(c, "type", None) == "function":
                    payload = getattr(c.function, "arguments", None)
                    args = payload
                    break
        text = args or resp.output_text or "{}"
        try:
            data = json.loads(text)
        except Exception as e:
            raise SchemaValidationError(f"Model did not return valid JSON text: {e}", partial_text=text) from e

        if conversation:
            conversation.add_user_parts(user_parts)
            conversation.add_assistant_text(text)
            conversation.set_previous_response_id(getattr(resp, "id", None))
        return {"json": data, "usage": getattr(resp, "usage", None), "request_id": getattr(resp, "id", None), "model": model}
    return _with_retries(_do, attempts=attempts, backoff_base=backoff)

def chat(
    messages: List[Message],
    *,
    model: str | None = None,
    history: Optional[HistoryAdapter] = None,
    conversation: Optional[ChatHistory] = None,   # NEW (optional)
    append_assistant_to_conversation: bool = False,  # NEW (default off)
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,
) -> Dict[str, Any]:
    """
    Low-level call with caller-supplied messages.
    - If `conversation` is provided, we *only*:
        • pass server-side chaining flags (store/previous_response_id)
        • optionally append assistant reply to conversation (if enabled)
      We do NOT merge `conversation` turns into `messages` to avoid duplicates.
    """
    if not isinstance(messages, list) or not messages:
        raise NormalizationError("chat(messages=...) requires a non-empty list of message dicts.")

    model = model or DefaultModel()
    adapter = history or NoHistory()
    eff = adapter.trim(messages)

    # server-side chaining if requested
    store_flag = bool(conversation and conversation.chain_server_side)
    prev_id = conversation.previous_response_id() if conversation else None

    client = _client_get()
    attempts, backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    def _do():
        resp = client.responses.create(
            model=model,
            input=eff,
            store=store_flag,
            previous_response_id=prev_id,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
        )
        text = resp.output_text or ""

        # Optional: only append assistant text; do NOT add the user turn here
        if conversation and append_assistant_to_conversation:
            conversation.add_assistant_text(text)
            conversation.set_previous_response_id(getattr(resp, "id", None))

        return {
            "text": text,
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }

    return _with_retries(_do, attempts=attempts, backoff_base=backoff)


def call_with_tools(
    input_or_messages: Any,
    tools: List[ToolSpec],
    *,
    model: str | None = None,
    tool_choice: Union[str, Dict[str, Any]] = "auto",
    auto_execute: bool = False,
    max_round_trips: int = 1,
    history: Optional[HistoryAdapter] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,
) -> Dict[str, Any]:
    """
    Returns tool_calls (parsed) by default. If auto_execute=True and handlers
    are provided, executes tools and asks model for the follow-up (bounded).
    """
    model = model or DefaultModel()
    messages = (
        input_or_messages
        if (isinstance(input_or_messages, list) and input_or_messages and "role" in input_or_messages[0])
        else format_messages(input_or_messages)
    )
    adapter = history or NoHistory()
    messages = adapter.trim(messages)

    tool_defs = [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
    } for t in tools]

    client = _client_get()
    attempts, backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    def _single_pass(msgs: List[Message]) -> Any:
        return client.responses.create(
            model=model,
            input=msgs,
            tools=tool_defs,
            tool_choice=tool_choice,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
        )

    def _parse_tool_calls(resp) -> List[Dict[str, Any]]:
        calls = []
        for c in getattr(resp, "tool_calls", []) or []:
            if getattr(c, "type", None) == "function" or (isinstance(c, dict) and c.get("type") == "function"):
                fn = getattr(c, "function", None) or c.get("function")
                name = getattr(fn, "name", None) if fn is not None else None
                if name is None and isinstance(fn, dict):
                    name = fn.get("name")
                args_str = getattr(fn, "arguments", None) if fn is not None else None
                if args_str is None and isinstance(fn, dict):
                    args_str = fn.get("arguments")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception as e:
                    raise ToolCallError(f"Tool '{name}' arguments not valid JSON: {e}") from e
                calls.append({"id": getattr(c, "id", None) or c.get("id"), "name": name, "args": args})
        return calls

    def _auto_exec_and_follow_up(msgs: List[Message], calls: List[Dict[str, Any]]) -> Tuple[List[Message], Optional[str]]:
        name_map = {t.name: t for t in tools}
        tool_msgs: List[Message] = []
        for call in calls:
            spec = name_map.get(call["name"])
            if not spec or not spec.handler:
                raise ToolCallError(f"No handler provided for tool '{call['name']}' (required for auto_execute).")
            result = spec.handler(call["args"])
            tool_msgs.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": [{"type": "output_text", "text": json.dumps(result)}],
            })
        msgs2 = msgs + tool_msgs
        resp2 = _single_pass(msgs2)
        return msgs2, (resp2.output_text or "")

    def _do():
        resp = _single_pass(messages)
        calls = _parse_tool_calls(resp)
        out: Dict[str, Any] = {
            "tool_calls": calls,
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }
        if not auto_execute or not calls:
            out["follow_up"] = resp.output_text or ""
            return out

        msgs = messages
        follow_up = resp.output_text or ""
        for _ in range(max_round_trips):
            msgs, follow_up = _auto_exec_and_follow_up(msgs, calls)
            respN = _single_pass(msgs)
            calls = _parse_tool_calls(respN)
            if not calls:
                out["follow_up"] = respN.output_text or follow_up
                out["usage"] = getattr(respN, "usage", out["usage"])
                out["request_id"] = getattr(respN, "id", out["request_id"])
                return out
        out["follow_up"] = follow_up
        return out

    return _with_retries(_do, attempts=attempts, backoff_base=backoff)
