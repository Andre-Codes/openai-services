# openai_services/__init__.py
from .chat_util import get_text, get_structured, chat, call_with_tools
from .models import MODELS, DefaultModel, Models
from .history import HistoryAdapter, RollingTokenHistory
from .tools import ToolSpec
from .errors import (
    NormalizationError, SchemaValidationError, ToolCallError,
    RateLimitError, StreamAbortedError
)

__all__ = [
    "get_text", "get_structured", "chat", "call_with_tools",
    "MODELS", "DefaultModel", "Models",
    "HistoryAdapter", "RollingTokenHistory",
    "ToolSpec",
    "NormalizationError", "SchemaValidationError", "ToolCallError",
    "RateLimitError", "StreamAbortedError",
]
