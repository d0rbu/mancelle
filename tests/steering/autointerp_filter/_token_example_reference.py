"""Whole-shard reference loader for row-group token-example tests."""

from pathlib import Path

import pyarrow.parquet as pq


def load_shard_text(parquet_path: Path) -> list[str]:
    """Load every text row without using the production row-group fast path."""
    pf = pq.ParquetFile(parquet_path)
    texts: list[str] = []
    for row_group_index in range(pf.num_row_groups):
        table = pf.read_row_group(row_group_index)
        texts.extend(str(x) for x in table.column("text").to_pylist())
    return texts
