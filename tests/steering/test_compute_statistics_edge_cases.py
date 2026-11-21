"""Edge-case and assertion tests for compute_statistics.py.

Targets undertested paths: accumulator expansion, mismatched inputs,
filter_shard_by_percentile edge cases, and input validation.
"""

import pytest
import torch as t

from steering.compute_statistics import (
    accumulate_shard,
    compute_per_feature_thresholds,
    filter_shard_by_percentile,
    init_accumulators,
)


def _make_shard(
    feat_ids: list[int],
    feat_mags: list[float],
    offsets: list[int],
    n_segments: int,
    *,
    n_docs: int = 1,
    doc_token_counts: list[int] | None = None,
) -> dict:
    """Build a minimal shard dict for testing."""
    if doc_token_counts is None:
        doc_token_counts = [1024] * n_docs
    return {
        "feature_ids": t.tensor(feat_ids, dtype=t.int32),
        "feature_mags": t.tensor(feat_mags, dtype=t.float32),
        "offsets": t.tensor(offsets, dtype=t.int64),
        "n_segments": n_segments,
        "n_docs": n_docs,
        "doc_ids": t.zeros(n_segments, dtype=t.int64),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "context_window": 1024,
        "pooling": "mean_pool",
        "layer": 8,
    }


class TestAccumulateShardEdgeCases:
    """Edge cases for accumulate_shard."""

    def test_mismatched_lengths_asserts(self):
        """feat_ids and feat_mags with different lengths should assert."""
        acc = init_accumulators(10)
        feat_ids = t.tensor([0, 1, 2], dtype=t.int32)
        feat_mags = t.tensor([1.0, 2.0], dtype=t.float32)  # wrong length
        with pytest.raises(AssertionError, match="feat_ids length"):
            accumulate_shard(acc, feat_ids, feat_mags)

    def test_auto_expansion_beyond_initial_size(self):
        """Accumulator expands when feature IDs exceed initial size."""
        acc = init_accumulators(5)  # only 5 features
        feat_ids = t.tensor([0, 1, 10], dtype=t.int32)  # ID 10 is beyond initial size
        feat_mags = t.tensor([1.0, 2.0, 3.0], dtype=t.float32)
        accumulate_shard(acc, feat_ids, feat_mags)
        assert len(acc.occurrence) == 11  # expanded to fit ID 10
        assert acc.occurrence[10].item() == 1
        assert acc.mag_sum[10].item() == 3.0

    def test_auto_expansion_preserves_existing_data(self):
        """Expansion doesn't corrupt previously accumulated data."""
        acc = init_accumulators(5)
        # First accumulation: IDs 0-4
        accumulate_shard(
            acc,
            t.tensor([0, 1, 2], dtype=t.int32),
            t.tensor([1.0, 2.0, 3.0], dtype=t.float32),
        )
        assert acc.occurrence[0].item() == 1
        # Second accumulation: ID 10 forces expansion
        accumulate_shard(
            acc,
            t.tensor([0, 10], dtype=t.int32),
            t.tensor([5.0, 7.0], dtype=t.float32),
        )
        assert acc.occurrence[0].item() == 2  # old data preserved
        assert acc.mag_sum[0].item() == 6.0  # 1.0 + 5.0
        assert acc.occurrence[10].item() == 1

    def test_empty_input_is_noop(self):
        """Empty tensors don't modify accumulators."""
        acc = init_accumulators(5)
        accumulate_shard(
            acc,
            t.tensor([], dtype=t.int32),
            t.tensor([], dtype=t.float32),
        )
        assert acc.occurrence.sum().item() == 0

    def test_repeated_feature_id(self):
        """Same feature ID appearing multiple times accumulates correctly."""
        acc = init_accumulators(3)
        feat_ids = t.tensor([1, 1, 1, 1], dtype=t.int32)
        feat_mags = t.tensor([2.0, 4.0, 6.0, 8.0], dtype=t.float32)
        accumulate_shard(acc, feat_ids, feat_mags)
        assert acc.occurrence[1].item() == 4
        assert abs(acc.mag_sum[1].item() - 20.0) < 1e-6
        assert abs(acc.mag_min[1].item() - 2.0) < 1e-6
        assert abs(acc.mag_max[1].item() - 8.0) < 1e-6


class TestFilterShardByPercentileEdgeCases:
    """Edge cases for filter_shard_by_percentile."""

    def test_empty_shard(self):
        """Shard with no features returns unchanged."""
        shard = _make_shard([], [], [0], n_segments=0, n_docs=0, doc_token_counts=[])
        # n_segments=0 means offsets has one element [0], but filter expects n_segments from dict
        shard["n_segments"] = 0
        thresholds = t.tensor([1.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        assert len(result["feature_ids"]) == 0

    def test_all_features_filtered_out(self):
        """All activations below threshold -> empty result with valid offsets."""
        shard = _make_shard(
            feat_ids=[0, 0, 1],
            feat_mags=[0.1, 0.2, 0.3],
            offsets=[0, 2, 3],
            n_segments=2,
        )
        thresholds = t.tensor([10.0, 10.0], dtype=t.float64)  # all above any activation
        result = filter_shard_by_percentile(shard, thresholds)
        assert len(result["feature_ids"]) == 0
        assert len(result["feature_mags"]) == 0
        assert result["offsets"][0] == 0
        assert result["offsets"][-1] == 0
        # Postcondition assertions should hold
        assert result["offsets"][-1] == len(result["feature_ids"])

    def test_feature_ids_beyond_threshold_range_are_dropped(self):
        """Feature IDs beyond threshold tensor length are treated as inf threshold."""
        shard = _make_shard(
            feat_ids=[0, 5],  # ID 5 is beyond thresholds length of 3
            feat_mags=[1.0, 1.0],
            offsets=[0, 1, 2],
            n_segments=2,
        )
        thresholds = t.tensor([0.5, float("inf"), float("inf")], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        # Only feature 0 (mag 1.0 > threshold 0.5) should survive
        assert len(result["feature_ids"]) == 1
        assert result["feature_ids"][0].item() == 0

    def test_partial_filtering_preserves_offsets(self):
        """Filtering some activations produces correct new offsets."""
        shard = _make_shard(
            feat_ids=[0, 1, 2, 0, 1, 2],
            feat_mags=[5.0, 0.1, 3.0, 0.1, 5.0, 0.1],
            offsets=[0, 3, 6],
            n_segments=2,
        )
        # Threshold: only keep mags > 1.0 for all features
        thresholds = t.tensor([1.0, 1.0, 1.0], dtype=t.float64)
        result = filter_shard_by_percentile(shard, thresholds)
        # Segment 0: keeps IDs 0 (5.0>1.0) and 2 (3.0>1.0), drops 1 (0.1)
        # Segment 1: keeps ID 1 (5.0>1.0), drops 0 (0.1) and 2 (0.1)
        assert len(result["feature_ids"]) == 3
        assert result["offsets"].tolist() == [0, 2, 3]


class TestComputePerFeatureThresholdsValidation:
    """Input validation tests for compute_per_feature_thresholds."""

    def test_negative_percentile_asserts(self):
        """Negative percentile should raise AssertionError."""
        feat_ids = t.tensor([0], dtype=t.int32)
        feat_mags = t.tensor([1.0], dtype=t.float32)
        with pytest.raises(AssertionError, match="percentile must be in"):
            compute_per_feature_thresholds(feat_ids, feat_mags, -0.1)

    def test_percentile_above_one_asserts(self):
        """Percentile > 1.0 should raise AssertionError."""
        feat_ids = t.tensor([0], dtype=t.int32)
        feat_mags = t.tensor([1.0], dtype=t.float32)
        with pytest.raises(AssertionError, match="percentile must be in"):
            compute_per_feature_thresholds(feat_ids, feat_mags, 1.5)
