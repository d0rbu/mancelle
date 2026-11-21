"""Unit tests for ignore_bos_features metadata in feature_io."""

import torch as t

from steering.feature_io import (
    MetadataKey,
    _parse_metadata_value,
    load_features,
    save_features,
)


def _minimal_shard_dict(**extra):
    d = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "context_window": 2048,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }
    d.update(extra)
    return d


def test_metadata_key_present():
    assert MetadataKey.IGNORE_BOS_FEATURES == "ignore_bos_features"


def test_parse_metadata_value_bool():
    assert _parse_metadata_value(MetadataKey.IGNORE_BOS_FEATURES, "true") is True
    assert _parse_metadata_value(MetadataKey.IGNORE_BOS_FEATURES, "false") is False
    assert _parse_metadata_value(MetadataKey.IGNORE_BOS_FEATURES, "TRUE") is True


def test_save_load_roundtrip_true(tmp_path):
    p = save_features(_minimal_shard_dict(ignore_bos_features=True), tmp_path / "s")
    loaded = load_features(p)
    assert loaded["ignore_bos_features"] is True


def test_save_load_roundtrip_false(tmp_path):
    p = save_features(_minimal_shard_dict(ignore_bos_features=False), tmp_path / "s")
    loaded = load_features(p)
    assert loaded["ignore_bos_features"] is False


def test_legacy_shard_defaults_false(tmp_path):
    p = save_features(_minimal_shard_dict(), tmp_path / "s")
    loaded = load_features(p)
    assert loaded["ignore_bos_features"] is False
