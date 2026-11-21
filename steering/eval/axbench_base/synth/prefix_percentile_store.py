"""Disk persistence for Stage 1b prefix-percentile scores.

``PrefixPercentileCache`` already memoises prefix scores, but only for the
lifetime of one process. Stage 1b scores every admissible seed prefix with an
SAE forward before any provider access, so a relaunch repeats the whole pass:
on a 50-concept run that measured ~8m38s of GPU work (of an 11m42s startup)
before the first API call, paid again after every transport failure.

This adds an optional durable backing store. It does not change what a score
is, when one is computed, or the in-memory key -- it only lets a hit survive
process exit.

Correctness stance
------------------
Scores are a deterministic pure function of the existing cache key, which
already pins model id, context window, SAE id and layer, tokenizer, percentile
namespace, feature id, token count and a token digest. Persisting keyed on that
identity therefore returns exactly the float the SAE forward would have
produced, so a warm run is observationally identical to a cold one. If any
identity component changes, the key changes and the old row is simply never
read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from steering.eval.axbench_base.types import AxBenchBaseError

#: Bump when the stored VALUE's meaning changes (scorer semantics, BOS
#: handling, percentile convention). Rows are namespaced by it, so a bump
#: makes previous rows unreadable rather than reinterpreted.
PREFIX_PERCENTILE_STORE_SCHEMA_VERSION = 1

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS prefix_percentiles (
    schema_version INTEGER NOT NULL,
    key            TEXT    NOT NULL,
    percentile     REAL    NOT NULL,
    PRIMARY KEY (schema_version, key)
)
"""


def cache_key_to_text(key: Any) -> str:
    """Render an in-memory cache key as a stable string.

    Both ``PrefixPercentileCacheKey`` and its cluster variant are frozen
    dataclasses, so field order is fixed by declaration and the rendering is
    deterministic. The class name is included so the scalar and cluster keys
    can never collide in one table.
    """

    if not is_dataclass(key) or isinstance(key, type):
        raise AxBenchBaseError(
            "prefix-percentile store expects a dataclass cache key, got "
            f"{type(key).__name__}"
        )
    fields = asdict(key)
    rendered = ",".join(f"{name}={fields[name]!r}" for name in sorted(fields))
    return f"{type(key).__name__}({rendered})"


class PrefixPercentileStore:
    """SQLite-backed durable store for prefix-percentile scores."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.execute(_TABLE_DDL)
            self._conn.commit()
        except sqlite3.DatabaseError as exc:
            # A corrupt store is an anomaly worth surfacing, not something to
            # silently delete -- doing so would hide a filesystem fault behind
            # a mysteriously slow run.
            raise AxBenchBaseError(
                f"Stage 1b prefix-percentile store at {self._path} is unusable: "
                f"{exc}. Delete the file to rebuild it, or investigate the "
                "filesystem."
            ) from exc
        logger.info("Stage 1b prefix-percentile store: {}", self._path)

    def close(self) -> None:
        self._conn.close()

    def get(self, key: Any) -> float | None:
        row = self._conn.execute(
            "SELECT percentile FROM prefix_percentiles "
            "WHERE schema_version = ? AND key = ?",
            (PREFIX_PERCENTILE_STORE_SCHEMA_VERSION, cache_key_to_text(key)),
        ).fetchone()
        return None if row is None else float(row[0])

    def put_many(self, items: list[tuple[Any, float]]) -> None:
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO prefix_percentiles "
            "(schema_version, key, percentile) VALUES (?, ?, ?)",
            [
                (
                    PREFIX_PERCENTILE_STORE_SCHEMA_VERSION,
                    cache_key_to_text(key),
                    float(value),
                )
                for key, value in items
            ],
        )
        self._conn.commit()
