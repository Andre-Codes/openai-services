from __future__ import annotations
import os, time, re
from typing import List, Dict, Optional, Iterable
from openai import OpenAI

_FALLBACK_DEFAULT = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
_AUTO_REFRESH = os.getenv("OPENAI_MODELS_AUTO_REFRESH", "false").lower() in ("1","true","yes")
_TTL_SECONDS = int(os.getenv("OPENAI_MODELS_TTL_SECONDS", "3600"))
_INCLUDE = [p for p in os.getenv("OPENAI_MODELS_INCLUDE","").split(",") if p.strip()]
_EXCLUDE = [p for p in os.getenv("OPENAI_MODELS_EXCLUDE","").split(",") if p.strip()]

class _ModelRegistry:
    def __init__(self):
        self._client: Optional[OpenAI] = None
        self._cache_ids: List[str] = []
        self._cache_meta: Dict[str, Dict] = {}  # id -> {"created": int, "owned_by": str}
        self._ts: float = 0.0
        self._default: str = _FALLBACK_DEFAULT

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _expired(self) -> bool:
        return (time.time() - self._ts) > _TTL_SECONDS

    def _apply_filters(self, ids: Iterable[str]) -> List[str]:
        out = list(ids)
        if _INCLUDE:
            out = [m for m in out if any(re.search(p.strip(), m) for p in _INCLUDE)]
        if _EXCLUDE:
            out = [m for m in out if not any(re.search(p.strip(), m) for p in _EXCLUDE)]
        return sorted(set(out))

    def refresh(self, force: bool = False) -> List[str]:
        if not force and self._cache_ids and not self._expired():
            return self._cache_ids

        try:
            resp = self._get_client().models.list()
            items = getattr(resp, "data", resp)  # handle SDK container or plain iterable
            ids, meta = [], {}
            for m in items:
                mid = getattr(m, "id", None) or (isinstance(m, dict) and m.get("id"))
                if not mid:
                    continue
                ids.append(mid)
                meta[mid] = {
                    "created": getattr(m, "created", None) or (isinstance(m, dict) and m.get("created")),
                    "owned_by": getattr(m, "owned_by", None) or (isinstance(m, dict) and m.get("owned_by")),
                }
            ids = self._apply_filters(ids)
            if ids:
                self._cache_ids = ids
                self._cache_meta = meta
                self._ts = time.time()
                if self._default not in self._cache_ids:
                    # Prefer a modern general model if available
                    preferred = next((m for m in self._cache_ids if any(k in m for k in ("gpt-4o", "gpt-4.1", "o3"))), None)
                    if preferred:
                        self._default = preferred
        except Exception:
            if not self._cache_ids:
                self._cache_ids = [_FALLBACK_DEFAULT]
                self._cache_meta = {_FALLBACK_DEFAULT: {"created": None, "owned_by": "fallback"}}
        return self._cache_ids

    def list(self, use_cache: bool = True) -> List[str]:
        if _AUTO_REFRESH:
            return self.refresh(force=not use_cache)
        return self._cache_ids or self.refresh(force=False)

    def list_with_meta(self) -> List[Dict]:
        """Convenience: [{'id', 'created', 'owned_by'}...]"""
        self.list()  # ensure cache
        return [{"id": i, **self._cache_meta.get(i, {})} for i in self._cache_ids]

    def get_default(self) -> str:
        if _AUTO_REFRESH and (self._expired() or self._default not in self._cache_ids):
            self.refresh(force=False)
        return self._default

    def is_available(self, model_id: str) -> bool:
        return model_id in (self._cache_ids or self.list())

MODELS = _ModelRegistry()

class Models:
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

DefaultModel = MODELS.get_default