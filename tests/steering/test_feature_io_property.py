"""Property-based tests for feature_io — slice 01 Phase 1 audit.

Hypothesis: derandomize=True, max_examples=200 per design spec §7.
"""

from pathlib import Path

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from errors import EmptyExtractionSegmentError
from steering.feature_io import load_features, save_features

# ---------------------------------------------------------------------------
# B-4 + B-5: save → load round-trip; post-load IDs ascending within segment
# ---------------------------------------------------------------------------

# Strategy: small CSR shards. Each segment has 0-50 features; 1-8 segments.
# Feature IDs in [0, 1000). We pre-sort ascending so we don't need to depend
# on extract_sparse_gpu's output ordering.
_segment_features = st.lists(
    st.integers(min_value=0, max_value=999),
    min_size=1,
    max_size=50,
    unique=True,
)
_n_segs = st.integers(min_value=1, max_value=8)


def _build_shard(seg_feats: list[list[int]]) -> dict:
    """Build a CSR shard with per-segment sorted-ascending IDs."""
    flat_ids: list[int] = []
    flat_mags: list[float] = []
    offsets: list[int] = [0]
    for ids in seg_feats:
        sorted_ids = sorted(ids)
        flat_ids.extend(sorted_ids)
        flat_mags.extend(float(i + 1) for i in range(len(sorted_ids)))
        offsets.append(offsets[-1] + len(sorted_ids))
    n_segs = len(seg_feats)
    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(flat_ids, dtype=t.int32),
        "feature_mags": t.tensor(flat_mags, dtype=t.float32),
        "doc_ids": t.zeros(n_segs, dtype=t.int32),
        "segment_indices": t.arange(n_segs, dtype=t.int32),
        "segment_positions": t.zeros(n_segs, dtype=t.int32),
        "segment_char_starts": t.zeros(n_segs, dtype=t.int32),
        "segment_char_ends": t.full((n_segs,), 10, dtype=t.int32),
    }


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    n_segs=_n_segs,
    seg_feats_seed=st.lists(_segment_features, min_size=1, max_size=8),
)
def test_save_load_roundtrip(tmp_path: Path, n_segs, seg_feats_seed):
    """save_features → load_features preserves IDs (post-sort) and mags."""
    seg_feats = [seg_feats_seed[i % len(seg_feats_seed)] for i in range(n_segs)]
    shard = _build_shard(seg_feats)
    path = save_features(shard, tmp_path / "shard_features.safetensors.zst")
    loaded = load_features(path)
    # IDs round-trip exactly (input was sorted ascending; on-disk is sorted
    # ascending; load preserves that).
    assert (loaded["feature_ids"] == shard["feature_ids"]).all()
    # Mags round-trip within fp16 precision (we used small ints as values).
    assert t.allclose(loaded["feature_mags"], shard["feature_mags"], atol=1e-3)


@settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)
@given(
    seg_feats_seed=st.lists(_segment_features, min_size=1, max_size=8),
)
def test_post_load_ids_ascending_within_segment(tmp_path: Path, seg_feats_seed):
    """Post-load `feature_ids` MUST be ascending within each segment.

    This pins map suspected-bug #36: the on-disk order is ascending-by-ID
    (delta-encodable); load_features delta-decodes back to that order.
    """
    shard = _build_shard(seg_feats_seed)
    path = save_features(shard, tmp_path / "asc_features.safetensors.zst")
    loaded = load_features(path)
    offsets = loaded["offsets"]
    ids = loaded["feature_ids"]
    n_segs = offsets.numel() - 1
    for i in range(n_segs):
        s = int(offsets[i].item())
        e = int(offsets[i + 1].item())
        if e - s < 2:
            continue
        seg_ids = ids[s:e]
        assert (seg_ids[1:] >= seg_ids[:-1]).all(), (
            f"segment {i} not ascending: {seg_ids.tolist()}"
        )


def test_save_features_rejects_empty_segment(tmp_path: Path):
    """An interior empty segment must be rejected by the base writer (issue #522)."""
    shard = _build_shard([[1, 2], [], [3]])  # middle segment empty -> offsets [0,2,2,3]
    with pytest.raises(EmptyExtractionSegmentError):
        save_features(shard, tmp_path / "x.safetensors.zst")
