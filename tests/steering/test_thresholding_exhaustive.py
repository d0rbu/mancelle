"""Exhaustive edge-case tests for steering.conditional.thresholding.

Covers gaps not present in test_thresholding.py or test_stats_thresholds.py.
Run with:
    uv run pytest tests/test_thresholding_exhaustive.py -x -v
"""

import math

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    _parse_combined_threshold,
    apply_feature_percentile,
    apply_segment_percentile,
    apply_segment_topk,
    apply_threshold,
    extract_threshold_percentile,
)


# ---------------------------------------------------------------------------
# 1. TestSegmentTopkEdgeCases
# ---------------------------------------------------------------------------
class TestSegmentTopkEdgeCases:
    """Edge cases for apply_segment_topk beyond the basics."""

    def test_k_equals_1_returns_maximum(self):
        """k=1 should return exactly the feature with highest magnitude."""
        feature_ids = t.tensor([10, 20, 30, 40], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.9, 0.5, 0.3], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=1)
        assert len(ids) == 1
        assert ids[0].item() == 20
        assert abs(mags[0].item() - 0.9) < 1e-6

    def test_k_equals_count_returns_all(self):
        """k = len(features) returns all features."""
        feature_ids = t.tensor([1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.4, 0.2, 0.6], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=3)
        assert set(ids.tolist()) == {1, 2, 3}
        assert len(ids) == 3

    def test_tied_magnitudes(self):
        """When multiple features have equal magnitudes, k still returns exactly k items."""
        feature_ids = t.tensor([1, 2, 3, 4, 5], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=3)
        assert len(ids) == 3
        # All returned magnitudes should be 0.5
        for m in mags.tolist():
            assert abs(m - 0.5) < 1e-6

    def test_single_feature(self):
        """Only 1 feature in segment, k=1 returns it."""
        feature_ids = t.tensor([42], dtype=t.int32)
        feature_mags = t.tensor([0.7], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=1)
        assert len(ids) == 1
        assert ids[0].item() == 42
        assert abs(mags[0].item() - 0.7) < 1e-6

    def test_large_k_on_single_feature(self):
        """k=1000, only 1 feature -> returns that 1 feature."""
        feature_ids = t.tensor([99], dtype=t.int32)
        feature_mags = t.tensor([1.5], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=1000)
        assert len(ids) == 1
        assert ids[0].item() == 99

    def test_negative_k_raises(self):
        """k must be a strict positive integer."""
        feature_ids = t.tensor([1, 2], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2], dtype=t.float32)
        with pytest.raises(BeartypeCallHintParamViolation, match="k"):
            apply_segment_topk(feature_ids, feature_mags, k=-1)

    def test_shape_mismatch_raises(self):
        """feature_ids and feature_mags different lengths."""
        feature_ids = t.tensor([1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2], dtype=t.float32)
        with pytest.raises(AssertionError, match="Shape mismatch"):
            apply_segment_topk(feature_ids, feature_mags, k=1)


# ---------------------------------------------------------------------------
# 2. TestSegmentPercentileEdgeCases
# ---------------------------------------------------------------------------
class TestSegmentPercentileEdgeCases:
    """Edge cases for apply_segment_percentile beyond the basics."""

    def test_very_small_percentile_rounds_to_1(self):
        """percentile=0.01, 3 features -> ceil(0.03) = 1."""
        feature_ids = t.tensor([10, 20, 30], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.9, 0.5], dtype=t.float32)
        ids, mags = apply_segment_percentile(feature_ids, feature_mags, percentile=0.01)
        # max(1, ceil(3 * 0.01)) = max(1, 1) = 1
        assert len(ids) == 1
        # Should return the maximum magnitude feature
        assert ids[0].item() == 20

    def test_zero_percentile_raises(self):
        """percentile=0 is outside the typed interval."""
        feature_ids = t.tensor([1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)
        with pytest.raises(BeartypeCallHintParamViolation, match="percentile"):
            apply_segment_percentile(feature_ids, feature_mags, percentile=0.0)

    def test_percentile_above_1_raises(self):
        """percentile=1.5 should raise."""
        feature_ids = t.tensor([1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)
        with pytest.raises(BeartypeCallHintParamViolation, match="percentile"):
            apply_segment_percentile(feature_ids, feature_mags, percentile=1.5)

    def test_single_feature_any_percentile(self):
        """1 feature, percentile=0.5 -> always keeps 1 (ceil(0.5) = 1)."""
        feature_ids = t.tensor([7], dtype=t.int32)
        feature_mags = t.tensor([3.14], dtype=t.float32)
        ids, mags = apply_segment_percentile(feature_ids, feature_mags, percentile=0.5)
        assert len(ids) == 1
        assert ids[0].item() == 7

    def test_exact_percentile_rounding(self):
        """10 features, percentile=0.3 -> ceil(3.0) = 3."""
        feature_ids = t.arange(10, dtype=t.int32)
        feature_mags = t.linspace(0.1, 1.0, 10)
        ids, mags = apply_segment_percentile(feature_ids, feature_mags, percentile=0.3)
        expected_k = math.ceil(10 * 0.3)  # ceil(3.0) = 3
        assert len(ids) == expected_k


# ---------------------------------------------------------------------------
# 3. TestFeaturePercentileEdgeCases
# ---------------------------------------------------------------------------
class TestFeaturePercentileEdgeCases:
    """Edge cases for feature_percentile application."""

    def test_percentile_boundary_value_included(self):
        """Activation exactly at the percentile threshold is KEPT (inclusive >=).

        Audit 2026-05-05 S-07-02 fix: previously excluded with strict ``>``.
        """
        # Set threshold for feature 0 to exactly 0.5
        precomputed = t.tensor([0.5], dtype=t.float64)
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.6, 0.4], dtype=t.float32)
        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, precomputed
        )
        # 0.5 IS >= 0.5, so kept. 0.6 >= 0.5 passes. 0.4 < 0.5 excluded.
        assert len(out_ids) == 2
        kept = sorted(out_mags.tolist())
        assert abs(kept[0] - 0.5) < 1e-6
        assert abs(kept[1] - 0.6) < 1e-6

    def test_percentile_all_same_value(self):
        """All same magnitude -> quantile = that value -> all KEPT under inclusive >="""
        feature_ids = t.tensor([0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.5, 0.5, 0.5], dtype=t.float32)
        thresholds = t.tensor([0.5], dtype=t.float64)
        # quantile of constant values = that value
        assert abs(thresholds[0].item() - 0.5) < 1e-6
        # Apply: all pass since mag >= threshold is true when equal (audit S-07-02 fix)
        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, thresholds
        )
        assert len(out_ids) == 4

    def test_percentile_p0_keeps_at_and_above_min(self):
        """percentile=0 -> threshold is minimum -> values >= min survive (inclusive)."""
        feature_ids = t.tensor([0, 0, 0, 0, 0], dtype=t.int32)
        # Use float64 magnitudes so promotion doesn't introduce drift
        feature_mags = t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float64)
        # threshold = 0.1 (the minimum)
        precomputed = t.tensor([0.1], dtype=t.float64)
        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, precomputed
        )
        # 0.1 IS >= 0.1 (inclusive). 0.2, 0.3, 0.4, 0.5 are also >= 0.1.
        assert len(out_ids) == 5

    def test_feature_with_single_occurrence(self):
        """1 activation, percentile=0.5 -> threshold = that value -> the one activation IS kept (>=)."""
        feature_ids = t.tensor([0], dtype=t.int32)
        feature_mags = t.tensor([0.7], dtype=t.float64)
        thresholds = t.tensor([0.7], dtype=t.float64)
        # quantile([0.7], 0.5) = 0.7
        assert abs(thresholds[0].item() - 0.7) < 1e-6
        # Apply: 0.7 IS >= 0.7 under the inclusive semantic
        out_ids, out_mags = apply_feature_percentile(
            feature_ids, feature_mags, thresholds
        )
        assert len(out_ids) == 1

    def test_many_features_independent_thresholds(self):
        """Verify each feature uses ITS OWN percentile, not global."""
        # Feature 0: magnitudes [0.1, 0.2, 0.3, 0.4, 0.5]
        # Feature 1: magnitudes [1.0, 2.0, 3.0, 4.0, 5.0]
        pct = 0.5
        # Feature 0: quantile([0.1..0.5], 0.5) = 0.3
        expected_0 = t.quantile(
            t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float64), pct
        )
        # Feature 1: quantile([1.0..5.0], 0.5) = 3.0
        expected_1 = t.quantile(
            t.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=t.float64), pct
        )
        thresholds = t.tensor([expected_0.item(), expected_1.item()], dtype=t.float64)
        assert abs(thresholds[0].item() - expected_0.item()) < 1e-6
        assert abs(thresholds[1].item() - expected_1.item()) < 1e-6

        # Apply to a mixed segment: feature 0 with mag 0.35, feature 1 with mag 2.5
        seg_ids = t.tensor([0, 1], dtype=t.int32)
        seg_mags = t.tensor([0.35, 2.5], dtype=t.float32)
        out_ids, out_mags = apply_feature_percentile(seg_ids, seg_mags, thresholds)
        # Feature 0: 0.35 > 0.3 -> passes
        # Feature 1: 2.5 is NOT > 3.0 -> excluded
        assert set(out_ids.tolist()) == {0}


# ---------------------------------------------------------------------------
# 6. TestCombinedThresholdParsing
# ---------------------------------------------------------------------------
class TestCombinedThresholdParsing:
    """Floating-point precision and edge cases for _parse_combined_threshold."""

    def test_floating_point_precision_2048_9(self):
        """Verify 2048.9 - 2048 doesn't produce 0.9000000000001."""
        k, pct = _parse_combined_threshold(2048.9)
        assert k == 2048
        # round(threshold - k, 2) should give exactly 0.9
        assert pct == 0.9

    def test_floating_point_precision_100_01(self):
        """Verify 100.01 -> (100, 0.01)."""
        k, pct = _parse_combined_threshold(100.01)
        assert k == 100
        assert pct == 0.01

    def test_very_large_k(self):
        """10000.5 -> (10000, 0.5)."""
        k, pct = _parse_combined_threshold(10000.5)
        assert k == 10000
        assert abs(pct - 0.5) < 1e-9

    def test_integer_threshold_raises(self):
        """100.0 -> percentile=0.0 -> raises assertion."""
        with pytest.raises(AssertionError, match="percentile must be in"):
            _parse_combined_threshold(100.0)


# ---------------------------------------------------------------------------
# 7. TestExtractThresholdPercentile
# ---------------------------------------------------------------------------
class TestExtractThresholdPercentile:
    """Tests for extract_threshold_percentile."""

    def test_feature_percentile_returns_threshold(self):
        """feature_percentile with threshold=0.7 -> returns 0.7."""
        cfg = ThresholdConfig(ThresholdType.feature_percentile, threshold=0.7)
        result = extract_threshold_percentile(cfg)
        assert result == 0.7

    def test_combined_returns_percentile_part(self):
        """feature_percentile_segment_topk with 2048.9 -> returns 0.9."""
        cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk,
            threshold=2048.9,
        )
        result = extract_threshold_percentile(cfg)
        assert result == 0.9

    def test_segment_topk_raises(self):
        """segment_topk raises NotImplementedError."""
        cfg = ThresholdConfig(ThresholdType.segment_topk, threshold=5)
        with pytest.raises(NotImplementedError):
            extract_threshold_percentile(cfg)

    def test_segment_percentile_raises(self):
        """segment_percentile raises NotImplementedError."""
        cfg = ThresholdConfig(ThresholdType.segment_percentile, threshold=0.5)
        with pytest.raises(NotImplementedError):
            extract_threshold_percentile(cfg)

# ---------------------------------------------------------------------------
# 8. TestThresholdOrderOfOperations
# ---------------------------------------------------------------------------
class TestThresholdOrderOfOperations:
    """Verify thresholding operates on raw magnitudes and order of operations."""

    def test_threshold_before_discretization_equivalence(self):
        """Apply threshold THEN discretize. Verify discretized output only contains
        features that passed the threshold.

        Simulates the pipeline: threshold -> discretize (bucket magnitudes).
        """
        feature_ids = t.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=t.int32)
        feature_mags = t.tensor(
            [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
            dtype=t.float32,
        )

        # Step 1: apply segment_topk threshold (k=4)
        cfg = ThresholdConfig(ThresholdType.segment_topk, threshold=4)
        thresh_ids, thresh_mags = apply_threshold(cfg, feature_ids, feature_mags)

        # Step 2: simulate discretization (bucket into 4 bins)
        bin_edges = t.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        buckets = t.bucketize(thresh_mags, bin_edges)

        # Verify: only top 4 features survived thresholding
        assert len(thresh_ids) == 4
        surviving_set = set(thresh_ids.tolist())
        # Features 4,5,6,7 (mags 0.45, 0.55, 0.65, 0.75) are top 4
        assert surviving_set == {4, 5, 6, 7}
        # Discretized values exist for exactly those that passed
        assert len(buckets) == 4

    def test_threshold_on_raw_magnitudes_not_buckets(self):
        """Verify thresholding operates on raw float magnitudes, not on bucket indices.

        If thresholding incorrectly operated on bucket indices (small ints), the
        results would differ from operating on raw floats.
        """
        feature_ids = t.tensor([0, 0, 0], dtype=t.int32)
        raw_mags = t.tensor([0.3, 0.7, 0.5], dtype=t.float32)

        # Precompute with threshold that creates a boundary at the float level
        precomputed = t.tensor([0.5], dtype=t.float64)  # threshold = 0.5

        # Apply on raw magnitudes
        out_ids, out_mags = apply_feature_percentile(feature_ids, raw_mags, precomputed)
        # 0.3 < 0.5 excluded, 0.5 >= 0.5 included, 0.7 >= 0.5 included
        assert len(out_ids) == 2
        out_mags_sorted = sorted(out_mags.tolist())
        assert abs(out_mags_sorted[0] - 0.5) < 1e-6
        assert abs(out_mags_sorted[1] - 0.7) < 1e-6

        # If we incorrectly used bucket indices (e.g., [1, 3, 2]) with the same
        # threshold of 0.5, all would pass since all bucket indices > 0.5.
        # The test passing confirms we use raw magnitudes.
        bucket_indices = t.tensor([1, 3, 2], dtype=t.float32)
        wrong_ids, wrong_mags = apply_feature_percentile(
            feature_ids, bucket_indices, precomputed
        )
        # All bucket indices (1, 2, 3) are >= 0.5, so all pass
        assert len(wrong_ids) == 3  # different from 2 above, confirming input matters
