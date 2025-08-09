from __future__ import annotations
from typing import List, Dict, Any, Protocol, Optional

Message = Dict[str, Any]

class HistoryAdapter(Protocol):
    def trim(self, messages: List[Message]) -> List[Message]:
        ...

class NoHistory(HistoryAdapter):
    def trim(self, messages: List[Message]) -> List[Message]:
        return messages

class RollingTokenHistory(HistoryAdapter):
    """
    Token-budgeted adapter (very simple heuristic). In production,
    integrate a tokenizer for exact budgets.
    """
    def __init__(self, max_turns: int = 12) -> None:
        self.max_turns = max_turns

    def trim(self, messages: List[Message]) -> List[Message]:
        # Keep system/developer, then last N turns.
        sys = [m for m in messages if m.get("role") in ("system", "developer")]
        convo = [m for m in messages if m.get("role") not in ("system", "developer")]
        return sys + convo[-self.max_turns:]
