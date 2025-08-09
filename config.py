from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ServiceConfig:
    # Defaults: keep model configurable, not hard-coded fossils.
    model: str = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o")
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT", "60"))
    max_output_tokens: Optional[int] = None  # allow model to decide if None
    temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    retry_max_attempts: int = int(os.getenv("OPENAI_RETRY_MAX", "4"))
    retry_backoff_base: float = float(os.getenv("OPENAI_RETRY_BACKOFF", "0.5"))

    # Optional: pass extra kwargs to the SDK client
    client_kwargs: Dict[str, Any] = field(default_factory=dict)

CONFIG = ServiceConfig()
