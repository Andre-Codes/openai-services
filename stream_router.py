from __future__ import annotations
from typing import Any, Dict, Iterator, Optional
from .errors import StreamAbortedError, SchemaValidationError

"""
Tiny router for Responses streaming events.
When stream=True, gpt_util plugs into one of these modes.

Event taxonomy mirrors Responses API streaming docs:
- response.output_text.delta, response.completed
- response.function_call.arguments.delta, response.refusal.delta, etc.
- response.error
Docs: Streaming responses & event types.  # :contentReference[oaicite:2]{index=2}
"""

class TextStreamAssembler:
    def __init__(self) -> None:
        self._buf: list[str] = []
        self.usage: Optional[Dict[str, Any]] = None
        self.request_id: Optional[str] = None

    def on_event(self, ev: Dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "response.output_text.delta":
            self._buf.append(ev["delta"])
        elif t == "response.completed":
            self.usage = ev.get("usage")
            self.request_id = ev.get("request_id")
        elif t == "response.error":
            raise StreamAbortedError(str(ev.get("error", "stream error")))

    def final_text(self) -> str:
        return "".join(self._buf)

class JsonStreamAssembler(TextStreamAssembler):
    # Assemble text then validate outside using the schema in gpt_util
    pass

class ToolEventRouter:
    """
    Surfaces tool call deltas in a structured way.
    Accumulates per call id so the caller gets complete args.
    """
    def __init__(self) -> None:
        self._calls: Dict[str, Dict[str, Any]] = {}

    def on_event(self, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        t = ev.get("type")
        if t == "response.function_call.arguments.delta":
            cid = ev["call_id"]
            name = ev["name"]
            self._calls.setdefault(cid, {"id": cid, "name": name, "args_text": ""})
            self._calls[cid]["args_text"] += ev["arguments_delta"]
        elif t == "response.function_call.completed":
            cid = ev["call_id"]
            call = self._calls.get(cid)
            if call:
                return {"type": "tool-call-complete", **call}
        elif t == "response.error":
            raise StreamAbortedError(str(ev.get("error", "stream error")))
        return None
