from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable, Union, Tuple
from openai import OpenAI
import json
import time
import random

from .config import CONFIG, ServiceConfig
from .models import DefaultModel
from .errors import (
    NormalizationError, SchemaValidationError, ToolCallError,
    RateLimitError,
)
from .message_normalizer import format_messages
from .history import HistoryAdapter, NoHistory
from .tools import ToolSpec

Message = Dict[str, Any]

# ──────────────────────────────────────────────────────────────────────────────
# Internal client (per-process singleton-ish)
_client: Optional[OpenAI] = None

def _client_get() -> OpenAI:
    global _client
    if _client is None:
        # Uses new OpenAI SDK style; Responses API is the recommended path.  # :contentReference[oaicite:3]{index=3}
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
                if ("rate limit" in msg) or ("429" in msg) or ("5" == getattr(e, "status", "")[:1]):
                    if i == attempts - 1:
                        raise RateLimitError(f"Retry attempts exhausted: {e}") from e
                    time.sleep(delay + random.random() * delay)
                    delay *= 2.0
                else:
                    raise
    return run()

# ──────────────────────────────────────────────────────────────────────────────
# Public API

def get_text(
    input_obj: Any,
    *,
    model: str | None = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    history: Optional[HistoryAdapter] = None,
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,  # (attempts, backoff_base)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Returns final text + metadata. Streaming is OFF by default.
    """
    model = model or DefaultModel()
    messages = format_messages(input_obj)
    adapter = history or NoHistory()
    messages = adapter.trim(messages)

    client = _client_get()
    cfg_attempts, cfg_backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    def _do():
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
            tool_choice=tool_choice,
        )
        # Text lives in output_text in Responses; final usage info provided at end.  # :contentReference[oaicite:4]{index=4}
        text = resp.output_text or ""
        return {
            "text": text,
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }

    return _with_retries(_do, attempts=cfg_attempts, backoff_base=cfg_backoff)

def get_structured(
    input_obj: Any,
    schema: Dict[str, Any],
    *,
    model: str | None = None,
    history: Optional[HistoryAdapter] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,
) -> Dict[str, Any]:
    """
    Guaranteed JSON via Structured Outputs (JSON Schema strict mode).
    Raises SchemaValidationError if invalid.  # Structured Outputs guide.  # :contentReference[oaicite:5]{index=5}
    """
    model = model or DefaultModel()
    messages = format_messages(input_obj)
    adapter = history or NoHistory()
    messages = adapter.trim(messages)

    client = _client_get()
    cfg_attempts, cfg_backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    def _do():
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": {**schema, "additionalProperties": schema.get("additionalProperties", False)},
                    "strict": True,
                },
            },
        )
        try:
            data = json.loads(resp.output_text or "{}")
        except Exception as e:
            raise SchemaValidationError(f"Model did not return valid JSON text: {e}", partial_text=resp.output_text) from e
        return {
            "json": data,
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }

    return _with_retries(_do, attempts=cfg_attempts, backoff_base=cfg_backoff)

def chat(
    messages: List[Message],
    *,
    model: str | None = None,
    response_format: Optional[Dict[str, Any]] = None,
    history: Optional[HistoryAdapter] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    retry: Optional[Tuple[int, float]] = None,
) -> Dict[str, Any]:
    """
    For callers who already have role-tagged messages[]. We still trim history
    and call Responses once.  # Recommended: manage context client-side.  # :contentReference[oaicite:6]{index=6}
    """
    if not isinstance(messages, list) or not messages:
        raise NormalizationError("chat(messages=...) requires a non-empty list of message dicts.")

    model = model or DefaultModel()
    adapter = history or NoHistory()
    eff = adapter.trim(messages)

    client = _client_get()
    cfg_attempts, cfg_backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

    def _do():
        resp = client.responses.create(
            model=model,
            input=eff,
            temperature=temperature if temperature is not None else CONFIG.temperature,
            max_output_tokens=max_output_tokens or CONFIG.max_output_tokens,
            response_format=response_format,
        )
        return {
            "text": resp.output_text or "",
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }

    return _with_retries(_do, attempts=cfg_attempts, backoff_base=cfg_backoff)

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
    Function/Tool-calling via Responses guide.  # :contentReference[oaicite:7]{index=7}
    """
    model = model or DefaultModel()
    messages = input_or_messages if (isinstance(input_or_messages, list) and input_or_messages and "role" in input_or_messages[0]) else format_messages(input_or_messages)
    adapter = history or NoHistory()
    messages = adapter.trim(messages)

    # Build tool defs for Responses
    tool_defs = [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
    } for t in tools]

    client = _client_get()
    cfg_attempts, cfg_backoff = retry or (CONFIG.retry_max_attempts, CONFIG.retry_backoff_base)

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
            if c.type == "function":
                # Some SDKs expose as attributes; be defensive:
                name = getattr(c.function, "name", None) or c.function.get("name")
                args_str = getattr(c.function, "arguments", None) or c.function.get("arguments")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except Exception as e:
                    raise ToolCallError(f"Tool '{name}' arguments not valid JSON: {e}") from e
                calls.append({"id": getattr(c, "id", None), "name": name, "args": args})
        return calls

    def _auto_exec_and_follow_up(msgs: List[Message], calls: List[Dict[str, Any]]) -> Tuple[List[Message], Optional[str]]:
        # Execute if handlers exist; append results as tool messages.
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

    # First pass
    def _do():
        resp = _single_pass(messages)
        calls = _parse_tool_calls(resp)
        result: Dict[str, Any] = {
            "tool_calls": calls,
            "usage": getattr(resp, "usage", None),
            "request_id": getattr(resp, "id", None),
            "model": model,
        }
        if not auto_execute or not calls:
            result["follow_up"] = resp.output_text or ""
            return result

        # Auto-exec loop (bounded)
        msgs = messages
        follow_up = resp.output_text or ""
        for _ in range(max_round_trips):
            msgs, follow_up = _auto_exec_and_follow_up(msgs, calls)
            # Re-check if the model made further calls
            respN = _single_pass(msgs)
            calls = _parse_tool_calls(respN)
            if not calls:
                result["follow_up"] = respN.output_text or follow_up
                result["usage"] = getattr(respN, "usage", result["usage"])
                result["request_id"] = getattr(respN, "id", result["request_id"])
                return result
        # If still calling tools, return whatever we have
        result["follow_up"] = follow_up
        return result

    return _with_retries(_do, attempts=cfg_attempts, backoff_base=cfg_backoff)
