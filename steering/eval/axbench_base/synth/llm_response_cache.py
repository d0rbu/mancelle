"""Per-run SQLite cache for LLM response bodies.

Keyed on a caller-provided ``cache_key`` string, which is typically
derived from ``(model, prompt, tag)`` where ``tag`` disambiguates
repeated calls with the same prompt (e.g., QC retry attempts that
need DISTINCT responses).

Usage::

    with SynthResponseCache(output_paths.synth_response_cache()) as cache:
        client = OpenRouterLLMClient(cfg.synth, response_cache=cache)
        ...

The cache is safe to leave open for the lifetime of a stage —
every write is committed immediately so a crash mid-stage still
leaves the cache queryable on re-run.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from types import TracebackType
from typing import Protocol

from steering.eval.axbench_base.profiler import profile_region, record_counter


class ResponseCache(Protocol):
    """Shared cache boundary for OpenRouter-backed generation paths."""

    def get(self, cache_key: str) -> str | None: ...

    def set(self, cache_key: str, response: str) -> None: ...


class SynthResponseCache:
    """Per-run SQLite-backed cache of LLM response bodies.

    Thread-safe: the underlying connection is opened with
    ``check_same_thread=False`` and all access is serialized through
    ``self._lock`` so concurrent judge/worker threads can share one cache.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "  key TEXT PRIMARY KEY,"
                "  response TEXT NOT NULL,"
                "  created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))"
                ")"
            )
            self._conn.commit()

    def get(self, cache_key: str) -> str | None:
        """Return the cached response for ``cache_key``, or ``None`` on miss."""
        with profile_region("cache.synth_response.get"):
            with self._lock:
                row = self._conn.execute(
                    "SELECT response FROM cache WHERE key = ?", (cache_key,),
                ).fetchone()
        if row:
            record_counter("cache.synth_response.hit", 1)
            return row[0]
        record_counter("cache.synth_response.miss", 1)
        return None

    def set(self, cache_key: str, response: str) -> None:
        """Upsert a response. Commits immediately for crash safety."""
        with profile_region("cache.synth_response.set"):
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
                    (cache_key, response),
                )
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SynthResponseCache:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def make_cache_key(*, model: str, prompt: str, tag: str) -> str:
    """Build a deterministic SHA256-based cache key from call identity.

    Three components go into the hash:

    - ``model``: the LLM model identifier (so runs against two different
      models don't share cache entries).
    - ``prompt``: the full user prompt content (so the same model
      answering different questions doesn't collide).
    - ``tag``: a caller-supplied discriminator used to distinguish
      otherwise-identical calls — e.g., QC retry attempts where the
      caller wants a FRESH response rather than a cache hit. Typical
      format: ``f"pos_c{concept_id}_s{seed_idx}_a{attempt_idx}"``.
    """
    payload = f"{model}\n---\n{prompt}\n---\n{tag}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
