from __future__ import annotations
from typing import Any, Dict, Optional, Callable, Iterable, Iterator, Literal

StreamEvent = Dict[str, Any]
Sink = Callable[[str], None]  # receives text deltas

class TextStreamAssembler:
    """Assembles text from Responses streaming events."""
    def __init__(self, on_delta: Optional[Sink] = None) -> None:
        self._buf: list[str] = []
        self.usage: Optional[Dict[str, Any]] = None
        self.request_id: Optional[str] = None
        self.on_delta = on_delta

    def on_event(self, ev: StreamEvent) -> None:
        t = ev.get("type")
        if t == "response.output_text.delta":
            chunk = ev["delta"]
            self._buf.append(chunk)
            if self.on_delta:
                self.on_delta(chunk)
        elif t == "response.completed":
            self.usage = ev.get("usage")
            self.request_id = ev.get("response", {}).get("id") or ev.get("request_id")
        elif t == "response.error":
            raise RuntimeError(str(ev.get("error", "stream error")))
        # ignore other event types for plain text mode

    def final_text(self) -> str:
        return "".join(self._buf)

class JsonStreamAssembler(TextStreamAssembler):
    """Same as text, but caller will json.loads() at end if using strict JSON outputs."""
    pass

class ToolEventRouter:
    """Collects tool-call args (function calling) from stream."""
    def __init__(self, on_tool_call_complete: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self._calls: Dict[str, Dict[str, Any]] = {}
        self.on_tool_call_complete = on_tool_call_complete

    def on_event(self, ev: StreamEvent) -> None:
        t = ev.get("type")
        if t == "response.function_call.arguments.delta":
            cid = ev["call_id"]; name = ev["name"]
            entry = self._calls.setdefault(cid, {"id": cid, "name": name, "args_text": ""})
            entry["args_text"] += ev["arguments_delta"]
        elif t == "response.function_call.completed":
            cid = ev["call_id"]
            entry = self._calls.get(cid)
            if entry and self.on_tool_call_complete:
                self.on_tool_call_complete({"id": cid, "name": entry["name"], "args_text": entry["args_text"]})
        elif t == "response.error":
            raise RuntimeError(str(ev.get("error", "stream error")))
