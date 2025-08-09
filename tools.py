from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for args
    handler: Optional[Callable[[Dict[str, Any]], Any]] = None  # optional auto-exec
