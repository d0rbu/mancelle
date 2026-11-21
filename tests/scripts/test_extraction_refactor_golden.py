"""X-EQ-1: feature-extraction segmentation metadata must be byte-identical
before and after the segmentation-helper consolidation (Phase A)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from sae_lens import SAE
from transformer_lens import HookedTransformer

from steering.extract_features import PoolingMode, ShardData, process_shard
from steering.feature_io import load_features
from steering.segmentation import SegmentMode
from tests.steering.test_extract_features_integration import TinyHookedTransformer, TinySAE

GOLDEN = Path(__file__).parent / "golden" / "extraction_refactor" / "tiny_shard_meta.json"

# A fixed corpus chosen to exercise multi-segment docs at segment_size=8.
CORPUS = [
    "the quick brown fox jumps over the lazy dog near the river bank today",
    "machine learning models segment text into fixed token windows for pooling",
    "short doc",
    "another medium length document with several words to force two segments here",
]


def _extract_meta(tmp_path: Path) -> dict:
    parquet = tmp_path / "shard_00000.parquet"
    pq.write_table(pa.table({"text": CORPUS}), parquet)
    model = TinyHookedTransformer(d_model=64, vocab_size=1000, device="cpu")
    sae = TinySAE(d_model=64, n_features=256, device="cpu")
    process_shard(
        str(parquet),
        cast(HookedTransformer, model),
        cast(SAE, sae),
        layer=8,
        output_dir=str(tmp_path / "out"),
        context_window=128, pooling=PoolingMode.max_pool, segment_mode=SegmentMode.fixed_tokens,
        segment_size=8,
    )
    shard_path = next((tmp_path / "out").glob("shard_00000_features*"))
    data = ShardData.from_dict(load_features(str(shard_path)))
    return {
        "segment_positions": data.segment_positions.tolist(),
        "segment_char_starts": data.segment_char_starts.tolist(),
        "segment_char_ends": data.segment_char_ends.tolist(),
        "doc_ids": data.doc_ids.tolist(),
        "segment_indices": data.segment_indices.tolist(),
        "doc_token_counts": data.doc_token_counts.tolist(),
    }


def test_extraction_segmentation_metadata_matches_golden(tmp_path):
    meta = _extract_meta(tmp_path)
    assert GOLDEN.exists(), (
        f"Golden missing: {GOLDEN}. Generate it ONCE on pre-refactor code via the "
        f"Task 1 Step 2 command, then commit it."
    )
    assert meta == json.loads(GOLDEN.read_text())
