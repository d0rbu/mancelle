"""Functional tests for feature_io save/load round-trip — slice 01 Phase 1."""

import torch as t

from steering.feature_io import load_features, save_features


def _make_csr_shard(
    feature_ids_per_seg: list[list[int]],
    feature_mags_per_seg: list[list[float]],
) -> dict:
    """Build a minimal CSR shard from per-segment lists.

    The producer side (`extract_sparse_gpu`) emits per-segment descending-mag
    arrays; we accept that here, then save_features re-sorts ascending by ID
    on disk.
    """
    n_segs = len(feature_ids_per_seg)
    flat_ids: list[int] = []
    flat_mags: list[float] = []
    offsets: list[int] = [0]
    for ids, mags in zip(feature_ids_per_seg, feature_mags_per_seg, strict=True):
        flat_ids.extend(ids)
        flat_mags.extend(mags)
        offsets.append(offsets[-1] + len(ids))
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


def test_save_load_roundtrip_simple(tmp_path):
    """A minimal shard with 2 segments survives save → load."""
    shard = _make_csr_shard(
        feature_ids_per_seg=[[5, 3, 1], [10, 8]],
        feature_mags_per_seg=[[3.0, 2.0, 1.0], [5.0, 4.0]],
    )
    path = save_features(shard, tmp_path / "test_features.safetensors.zst")
    loaded = load_features(path)
    # IDs come back ascending within segment (on-disk order).
    assert list(loaded["feature_ids"][0:3].tolist()) == [1, 3, 5]
    assert list(loaded["feature_ids"][3:5].tolist()) == [8, 10]
    # Mags follow IDs (co-sorted).
    assert loaded["feature_mags"][0:3].tolist() == [1.0, 2.0, 3.0]
    assert loaded["feature_mags"][3:5].tolist() == [4.0, 5.0]


def test_save_load_empty_shard(tmp_path):
    """An empty shard (no features, no segments) round-trips."""
    shard = {
        "offsets": t.tensor([0], dtype=t.int64),
        "feature_ids": t.tensor([], dtype=t.int32),
        "feature_mags": t.tensor([], dtype=t.float32),
        "doc_ids": t.tensor([], dtype=t.int32),
        "segment_indices": t.tensor([], dtype=t.int32),
        "segment_positions": t.tensor([], dtype=t.int32),
        "segment_char_starts": t.tensor([], dtype=t.int32),
        "segment_char_ends": t.tensor([], dtype=t.int32),
    }
    path = save_features(shard, tmp_path / "empty_features.safetensors.zst")
    loaded = load_features(path)
    assert loaded["offsets"].numel() == 1
    assert loaded["feature_ids"].numel() == 0
    assert loaded["feature_mags"].numel() == 0


def test_save_load_default_context_independent_false(tmp_path):
    """Legacy shards without `context_independent_features` metadata default to False."""
    shard = _make_csr_shard(
        feature_ids_per_seg=[[1, 2]],
        feature_mags_per_seg=[[1.0, 2.0]],
    )
    path = save_features(shard, tmp_path / "default_features.safetensors.zst")
    loaded = load_features(path)
    assert loaded["context_independent_features"] is False
