from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

Message = Dict[str, Any]

class HistoryAdapter:
    def trim(self, messages: List[Message]) -> List[Message]:
        return messages

class NoHistory(HistoryAdapter):
    def trim(self, messages: List[Message]) -> List[Message]:
        return messages

class RollingTokenHistory(HistoryAdapter):
    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns
    def trim(self, messages: List[Message]) -> List[Message]:
        sys = [m for m in messages if m.get("role") in ("system", "developer")]
        convo = [m for m in messages if m.get("role") not in ("system", "developer")]
        return sys + convo[-self.max_turns:]

@dataclass
class ChatHistory:
    """
    Stateful conversation you can pass to get_text/get_structured each time.
    - Accepts flexible inputs (strings, lists, mixed images) and formats them.
    - Automatically appends assistant replies after each call.
    - Optional server-side chaining with previous_response_id.
    """
    adapter: HistoryAdapter = field(default_factory=RollingTokenHistory)
    system: Optional[str] = None
    developer: Optional[str] = None
    _turns: List[Message] = field(default_factory=list, init=False)
    # server-side chaining:
    chain_server_side: bool = False
    _last_response_id: Optional[str] = field(default=None, init=False)

    def reset(self):
        self._turns.clear()
        self._last_response_id = None

    def _base_headers(self) -> List[Message]:
        headers: List[Message] = []
        if self.system:
            headers.append({"role": "system", "content": [{"type":"input_text","text": self.system}]})
        if self.developer:
            headers.append({"role": "developer", "content": [{"type":"input_text","text": self.developer}]})
        return headers

    def add_user_parts(self, content_parts: List[Dict[str, Any]]):
        self._turns.append({"role": "user", "content": content_parts})

    def add_assistant_text(self, text: str):
        self._turns.append({"role": "assistant", "content": [{"type":"output_text","text": text}]})

    def messages(self) -> List[Message]:
        msgs = self._base_headers() + list(self._turns)
        return self.adapter.trim(msgs)

    # internal hooks used by chat_util.py
    def set_previous_response_id(self, rid: Optional[str]):
        self._last_response_id = rid
    def previous_response_id(self) -> Optional[str]:
        return self._last_response_id if self.chain_server_side else None
