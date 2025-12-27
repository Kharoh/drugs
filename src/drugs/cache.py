"""Lightweight on-disk cache for API responses with TTL support."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_CACHE_PATH = Path(os.environ.get("DRUGS_CACHE_PATH", "artifacts/cache/api_cache.json"))
DEFAULT_TTL_SECONDS = float(os.environ.get("DRUGS_CACHE_TTL_SECONDS", 24 * 3600))
CACHE_DISABLED = os.environ.get("DRUGS_CACHE_DISABLED", "false").lower() in {"1", "true", "yes"}


class APICache:
    """Simple JSON-backed cache with expiration timestamps.

    The cache is intentionally lightweight and avoids external dependencies.
    Entries are stored as ``{"value": obj, "expires_at": epoch_seconds}``.
    """

    def __init__(self, path: Path = DEFAULT_CACHE_PATH, *, default_ttl: float = DEFAULT_TTL_SECONDS, enabled: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.enabled = enabled and not CACHE_DISABLED
        self._store: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._loaded = True
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._store = data.get("entries", {})  # type: ignore[assignment]
            self._loaded = True
        except Exception:
            # Corrupt cache: start fresh
            self._store = {}
            self._loaded = True

    def _persist(self) -> None:
        if not self.enabled:
            return
        payload = {"entries": self._store}
        try:
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            # Ignore persistence issues to avoid breaking callers
            return

    def clear_expired(self) -> None:
        now = time.time()
        changed = False
        for key, entry in list(self._store.items()):
            exp = entry.get("expires_at", 0)
            if exp and exp < now:
                self._store.pop(key, None)
                changed = True
        if changed:
            self._persist()

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        if not self._loaded:
            self._load()
        entry = self._store.get(key)
        if not entry:
            return None
        exp = entry.get("expires_at")
        if exp and exp < time.time():
            self._store.pop(key, None)
            self._persist()
            return None
        return entry.get("value")

    def set(self, key: str, value: Any, *, ttl: Optional[float] = None) -> None:
        if not self.enabled:
            return
        expires_at = time.time() + (ttl if ttl is not None else self.default_ttl)
        try:
            # ensure value is serializable
            json.dumps(value)
        except TypeError:
            # fallback: attempt to coerce via repr
            value = json.loads(json.dumps(value, default=repr))
        self._store[key] = {"value": value, "expires_at": expires_at}
        self._persist()

    def invalidate(self, key: Optional[str] = None) -> None:
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)
        self._persist()


def get_default_cache() -> Optional[APICache]:
    """Return a process-wide default API cache instance."""
    global _DEFAULT_CACHE
    try:
        return _DEFAULT_CACHE
    except NameError:
        pass

    cache = APICache()
    _DEFAULT_CACHE = cache
    return cache


__all__ = ["APICache", "get_default_cache"]
