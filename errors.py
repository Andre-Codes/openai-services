from __future__ import annotations

class NormalizationError(ValueError):
    pass

class SchemaValidationError(ValueError):
    def __init__(self, message: str, json_pointer: str | None = None, partial_text: str | None = None):
        super().__init__(message)
        self.json_pointer = json_pointer
        self.partial_text = partial_text

class ToolCallError(RuntimeError):
    pass

class RateLimitError(RuntimeError):
    pass

class StreamAbortedError(RuntimeError):
    pass
