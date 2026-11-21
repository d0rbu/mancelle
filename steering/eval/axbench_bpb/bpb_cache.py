"""Rank-local SQLite cache for BPB scoring. See spec §11.

The cache is a fast resumable lookup keyed by the composite primary key
``(method_name, concept_id, sample_id, coefficient_str, target)``. On
resume, the scoring loop skips any work unit already present — the
parquet is the durable store, and the SQLite is an index for O(1)
``has()`` checks. Task 9.3 adds ``reconcile_cache`` to handle crashes
between parquet writes and SQLite commits.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from steering.eval.axbench_bpb.types import AxBenchBpbError

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bpb_scores (
    method_name TEXT NOT NULL,
    concept_id INTEGER NOT NULL,
    sample_id TEXT NOT NULL,
    coefficient_str TEXT NOT NULL,
    target TEXT NOT NULL,
    bpb REAL NOT NULL,
    n_output_tokens INTEGER NOT NULL,
    n_output_bytes INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    inserted_at INTEGER NOT NULL,
    PRIMARY KEY (method_name, concept_id, sample_id, coefficient_str, target)
);
CREATE INDEX IF NOT EXISTS bpb_scores_concept ON bpb_scores (concept_id);
"""


class BpbCache:
    """SQLite-backed resumable cache for BpbScoreRow entries.

    Usage:
        with BpbCache(path) as cache:
            if not cache.has(method_name=..., concept_id=..., ...):
                bpb_row = score_sample(...)
                cache.put(method_name=..., bpb=bpb_row.bpb_intervened, ...)
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> BpbCache:
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(_SCHEMA_SQL)
        return self

    def __exit__(self, *args) -> None:
        assert self._conn is not None
        self._conn.close()
        self._conn = None

    def has(
        self, *, method_name: str, concept_id: int, sample_id: str,
        coefficient_str: str, target: str,
    ) -> bool:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT 1 FROM bpb_scores "
            "WHERE method_name=? AND concept_id=? AND sample_id=? "
            "AND coefficient_str=? AND target=? LIMIT 1",
            (method_name, concept_id, sample_id, coefficient_str, target),
        )
        return cur.fetchone() is not None

    def put(
        self, *, method_name: str, concept_id: int, sample_id: str,
        coefficient_str: str, target: str, bpb: float,
        n_output_tokens: int, n_output_bytes: int, truncated: bool,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            "INSERT OR REPLACE INTO bpb_scores "
            "(method_name, concept_id, sample_id, coefficient_str, target, "
            "bpb, n_output_tokens, n_output_bytes, truncated, inserted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (method_name, concept_id, sample_id, coefficient_str, target,
             float(bpb), int(n_output_tokens), int(n_output_bytes),
             1 if truncated else 0, int(time.time())),
        )
        self._conn.commit()


def reconcile_cache(*, rank_dir: Path, cache_path: Path) -> None:
    """Bidirectionally sync ``cache_path`` SQLite with ``rank_dir/*.parquet``.

    Invariant after reconcile: the set of primary keys
    ``(method_name, concept_id, sample_id, coefficient_str, target)`` in
    SQLite equals the set present in the rank's parquet shards.

    Failure modes this handles:
    - *Post-parquet crash*: parquet flushed durably but SQLite commit
      didn't land. We reseed SQLite from parquet rows not present in
      the index.
    - *Pre-flush crash*: SQLite committed but the parquet file is missing
      (rare — would require external manipulation since writes are
      atomic). We drop SQLite entries not found in parquet.

    Safe to call on a fresh cache (SQLite empty, parquet absent) — no-op.
    Safe to call on an up-to-date cache — no-op. Idempotent.
    """
    parquet_keys: set[tuple[str, int, str, str, str]] = set()
    parquet_rows_by_key: dict[tuple[str, int, str, str, str], dict] = {}

    if rank_dir.exists():
        for batch_path in sorted(rank_dir.glob("*.parquet")):
            try:
                tbl = pq.read_table(batch_path)
            except pa.ArrowInvalid as exc:
                raise AxBenchBpbError(
                    f"Corrupted parquet shard at {batch_path}: {exc}. "
                    f"Delete this file and re-run; reconcile will reseed "
                    f"the SQLite cache from the remaining shards."
                ) from exc
            except pa.ArrowIOError as exc:
                raise AxBenchBpbError(
                    f"I/O error reading parquet shard at {batch_path}: "
                    f"{exc}. Verify the file is readable and not actively "
                    f"being written."
                ) from exc
            for row in tbl.to_pylist():
                k = (
                    row["method_name"], int(row["concept_id"]),
                    row["sample_id"], row["coefficient_str"], row["target"],
                )
                parquet_keys.add(k)
                parquet_rows_by_key[k] = row

    with BpbCache(cache_path) as cache:
        assert cache._conn is not None
        cur = cache._conn.execute(
            "SELECT method_name, concept_id, sample_id, coefficient_str, target "
            "FROM bpb_scores"
        )
        sqlite_keys: set[tuple[str, int, str, str, str]] = {
            (r[0], int(r[1]), r[2], r[3], r[4]) for r in cur.fetchall()
        }

        # Drop SQLite entries not in parquet
        to_drop = sqlite_keys - parquet_keys
        for k in to_drop:
            cache._conn.execute(
                "DELETE FROM bpb_scores WHERE method_name=? AND concept_id=? "
                "AND sample_id=? AND coefficient_str=? AND target=?",
                k,
            )

        # Reseed SQLite from parquet rows missing from the index
        to_add = parquet_keys - sqlite_keys
        for k in to_add:
            row = parquet_rows_by_key[k]
            cache.put(
                method_name=row["method_name"],
                concept_id=int(row["concept_id"]),
                sample_id=row["sample_id"],
                coefficient_str=row["coefficient_str"],
                target=row["target"],
                bpb=float(row["bpb_intervened"]),
                n_output_tokens=int(row["n_output_tokens"]),
                n_output_bytes=int(row["n_output_bytes"]),
                truncated=bool(row["truncated"]),
            )

        cache._conn.commit()


__all__ = ["BpbCache", "reconcile_cache"]
