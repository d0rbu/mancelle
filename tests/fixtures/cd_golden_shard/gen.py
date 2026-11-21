"""Generate the golden cd shard fixture used by the backward-compat test.

Produces a deterministic cd shard from a fixed synthetic parquet. Re-running
this script must produce byte-identical output (modulo the timestamp encoded
in the zstd container, which is why the regression test compares the DECODED
contents, not raw bytes).

Usage:
    uv run python tests/fixtures/cd_golden_shard/gen.py
"""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from steering.extract_features import PoolingMode, process_shard
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE

FIXTURE_DIR = Path(__file__).parent
SEED = 20260422


def main():
    t.manual_seed(SEED)
    model = TinyHookedTransformer(d_model=64, vocab_size=256, device="cpu")
    sae = TinySAE(d_model=64, n_features=128, device="cpu")
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "mathematics and computing science",
        "cooking recipes for a rainy day",
    ]
    parquet_path = FIXTURE_DIR / "source.parquet"
    with pq.ParquetWriter(
        str(parquet_path),
        pa.schema([("text", pa.string())]),
    ) as w:
        w.write_table(pa.table({"text": texts}))

    result = process_shard(
        str(parquet_path),
        model,  # ty: ignore[invalid-argument-type]
        sae,  # ty: ignore[invalid-argument-type]
        layer=8,
        output_dir=str(FIXTURE_DIR),
        context_window=64,
        pooling=PoolingMode.max_pool,
        segment_size=4,
        gpu_batch_size=2,
        cpu_batch_size=4,
        context_independent_features=False,
    )
    print(f"Fixture written: {result.output_path}")


if __name__ == "__main__":
    main()
