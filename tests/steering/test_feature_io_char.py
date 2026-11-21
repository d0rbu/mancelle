"""Tests for character-level segment boundary fields in feature I/O."""

import torch as t

from steering.extract_features import ShardData
from steering.feature_io import load_features, save_features


def _make_shard_data() -> ShardData:
    """Create minimal ShardData with char boundary fields."""
    n_segs = 4
    n_feats = 20

    return ShardData(
        offsets=t.arange(n_segs + 1, dtype=t.int64) * 5,
        feature_ids=t.arange(n_feats, dtype=t.int32),
        feature_mags=t.randn(n_feats, dtype=t.float32).abs(),
        doc_ids=t.tensor([0, 0, 1, 1], dtype=t.int32),
        segment_indices=t.tensor([0, 1, 0, 1], dtype=t.int32),
        segment_positions=t.tensor([0, 128, 0, 128], dtype=t.int32),
        doc_token_counts=t.tensor([256, 256], dtype=t.int32),
        n_docs=2,
        n_segments=n_segs,
        layer=12,
        context_window=1024,
        pooling="mean_pool",
        segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
        segment_size=128,
        segment_char_starts=t.tensor([0, 500, 0, 400], dtype=t.int32),
        segment_char_ends=t.tensor([500, 1000, 400, 800], dtype=t.int32),
    )


class TestFeatureIOCharBoundaries:
    def test_roundtrip_with_char_fields(self, tmp_path):
        """Save and load feature file with char boundary fields."""
        shard = _make_shard_data()
        path = save_features(shard.to_dict(), tmp_path / "test_features")
        loaded = load_features(path)

        assert "segment_char_starts" in loaded
        assert "segment_char_ends" in loaded
        assert t.equal(loaded["segment_char_starts"], shard.segment_char_starts)
        assert t.equal(loaded["segment_char_ends"], shard.segment_char_ends)

    def test_shard_data_to_dict_includes_char_fields(self):
        """ShardData.to_dict() includes char fields."""
        shard = _make_shard_data()
        d = shard.to_dict()
        assert "segment_char_starts" in d
        assert "segment_char_ends" in d

    def test_shard_data_from_dict_with_chars(self):
        """ShardData.from_dict handles dicts with char fields."""
        shard = _make_shard_data()
        d = shard.to_dict()
        restored = ShardData.from_dict(d)
        assert t.equal(restored.segment_char_starts, shard.segment_char_starts)
        assert t.equal(restored.segment_char_ends, shard.segment_char_ends)
