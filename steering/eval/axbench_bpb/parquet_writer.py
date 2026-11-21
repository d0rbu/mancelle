"""Parquet writer for axbench-bpb score row outputs.

Uses pyarrow directly for atomic rank-local writes.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from steering.eval.axbench_bpb.types import BpbScoreRow

BPB_SCORE_SCHEMA = pa.schema([
    ("method_name", pa.string()),
    ("concept_id", pa.int32()),
    ("mode", pa.string()),
    ("content_class", pa.string()),
    ("target", pa.string()),
    ("coefficient_str", pa.string()),
    ("sample_id", pa.string()),
    ("bpb_intervened", pa.float64()),
    ("n_output_tokens", pa.int32()),
    ("n_output_bytes", pa.int32()),
    ("truncated", pa.bool_()),
])


def write_bpb_score_rows(rows: Iterable[BpbScoreRow], path: Path) -> None:
    """Atomic-write a batch of BpbScoreRow to parquet with enforced schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(r) for r in rows], schema=BPB_SCORE_SCHEMA)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.rename(path)


__all__ = [
    "BPB_SCORE_SCHEMA",
    "write_bpb_score_rows",
]
