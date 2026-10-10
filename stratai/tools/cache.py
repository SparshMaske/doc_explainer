"""Thin TTL cache over diskcache, stored in DATA_DIR/cache. Values must be JSON-compatible."""

from pathlib import Path
from typing import Any

import diskcache

from stratai.core.config import get_settings

DEFAULT_TTL_S = 7 * 24 * 3600

_open_caches: dict[Path, diskcache.Cache] = {}


class Cache:
    def __init__(self, namespace: str, directory: Path | None = None, ttl_s: int = DEFAULT_TTL_S) -> None:
        directory = Path(directory or Path(get_settings().DATA_DIR) / "cache").resolve()
        if directory not in _open_caches:
            _open_caches[directory] = diskcache.Cache(str(directory))
        self._store = _open_caches[directory]
        self.namespace = namespace
        self.ttl_s = ttl_s

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Any | None:
        return self._store.get(self._key(key))

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        self._store.set(self._key(key), value, expire=ttl_s if ttl_s is not None else self.ttl_s)
