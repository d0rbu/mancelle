"""Tests for bucket granularity with post-filter boundary computation.

The old approach used num_buckets=100 to get granularity after filtering.
The new approach keeps num_buckets=10 but computes bucket boundaries in the
post-filter percentile range: e.g. with threshold_percentile=0.9 and
num_buckets=10, boundaries land at p91..p99 instead of p10..p90.
"""

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.discretization import (
    build_boundaries_from_stats,
    discretize_activations,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    build_thresholds_from_stats,
    extract_threshold_percentile,
)
from steering.stats_types import PerFeatureStat


def _make_per_feature_stats(
    n_features: int, n_activations: int = 1000
) -> dict[str, PerFeatureStat]:
    """Generate synthetic per-feature stats with full p1..p99 percentiles.

    Each feature has activations drawn from a uniform(0, 1) distribution,
    so percentiles are approximately p_k ~ k/100.
    """
    stats: dict[str, PerFeatureStat] = {}
    for fid in range(n_features):
        # Simulate uniform(0,1) distribution: p_k ~ k/100
        percentiles = {f"p{p}": round(p / 100.0, 6) for p in range(1, 100)}
        stats[str(fid)] = PerFeatureStat(
            count=n_activations,
            pct_segments_active=50.0,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.29,
            min_mag=0.001,
            max_mag=0.999,
            median_mag=0.5,
            percentiles=percentiles,
        )
    return stats


def _simulate_filtered_activations(
    n_features: int,
    per_feature_stats: dict[str, PerFeatureStat],
    threshold_type: ThresholdType,
    threshold: float,
    n_sae_features: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Simulate activations that survive a percentile filter.

    Generates activations uniformly in (0, 1) per feature, then applies the
    threshold filter to keep only those above the percentile threshold.
    """
    # Generate activations: 500 per feature, uniform(0,1)
    all_ids = []
    all_mags = []
    for fid in range(n_features):
        mags = t.linspace(0.01, 0.99, 500)
        ids = t.full((500,), fid, dtype=t.int32)
        all_ids.append(ids)
        all_mags.append(mags)
    all_ids = t.cat(all_ids)
    all_mags = t.cat(all_mags)

    # Build thresholds and filter
    threshold_cfg = ThresholdConfig(threshold_type, threshold)
    precomputed = build_thresholds_from_stats(
        per_feature_stats,
        n_sae_features,
        threshold_type,
        threshold,
    )
    filtered_ids, filtered_mags = apply_threshold(
        threshold_cfg,
        all_ids,
        all_mags,
        precomputed=precomputed,
    )
    return filtered_ids, filtered_mags


class TestBucketGranularityBug:
    """Documents the old bug: full-range boundaries + percentile filter = no granularity."""

    def test_full_range_10_buckets_p90_all_same_bucket(self):
        """With full-range boundaries, all post-filter activations land in bucket 9."""
        n_features = 5
        n_sae_features = n_features
        stats = _make_per_feature_stats(n_features)

        # Full-range boundaries: no threshold_percentile (default None)
        boundaries = build_boundaries_from_stats(stats, n_sae_features, num_buckets=10)

        filtered_ids, filtered_mags = _simulate_filtered_activations(
            n_features,
            stats,
            ThresholdType.feature_percentile,
            0.9,
            n_sae_features,
        )
        assert len(filtered_ids) > 0, "Filter should keep some activations"

        buckets = discretize_activations(filtered_ids, filtered_mags, boundaries)

        # BUG: every single activation is in bucket 9
        unique_buckets = buckets.unique()
        assert len(unique_buckets) == 1, (
            f"Expected all activations in 1 bucket (the bug), got {len(unique_buckets)}"
        )
        assert unique_buckets[0].item() == 9


class TestPostFilterBoundaries:
    """Tests for post-filter boundary computation logic."""

    def test_boundaries_p90_10_buckets(self):
        """threshold_percentile=0.9, num_buckets=10 -> boundaries at p91..p99."""
        stats = _make_per_feature_stats(3)
        boundaries = build_boundaries_from_stats(
            stats, 3, num_buckets=10, threshold_percentile=0.9
        )
        # With uniform(0,1) stats, p_k = k/100, so:
        # boundary values should be [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        expected = t.tensor([p / 100.0 for p in range(91, 100)], dtype=t.float64)
        for fid in range(3):
            assert t.allclose(boundaries[fid], expected, atol=1e-4)

    def test_boundaries_p70_10_buckets(self):
        """threshold_percentile=0.7, num_buckets=10 -> 9 boundaries spanning p70-p99 range."""
        stats = _make_per_feature_stats(3)
        boundaries = build_boundaries_from_stats(
            stats, 3, num_buckets=10, threshold_percentile=0.7
        )
        # With n_boundaries=9, start=70, end=99:
        # raw_pcts = [70 + 29*(i+1)/9 for i in range(9)]
        # = [73.22, 76.44, 79.67, 82.89, 86.11, 89.33, 92.56, 95.78, 99.0]
        # rounded: [73, 76, 80, 83, 86, 89, 93, 96, 99]
        expected_pcts = [round(70 + 29 * (i + 1) / 9) for i in range(9)]
        expected = t.tensor([p / 100.0 for p in expected_pcts], dtype=t.float64)
        for fid in range(3):
            assert t.allclose(boundaries[fid], expected, atol=1e-4)

    def test_boundaries_none_is_full_range(self):
        """threshold_percentile=None -> original behavior (p10..p90 for 10 buckets)."""
        stats = _make_per_feature_stats(3)
        boundaries = build_boundaries_from_stats(
            stats, 3, num_buckets=10, threshold_percentile=None
        )
        expected = t.tensor([p / 100.0 for p in range(10, 100, 10)], dtype=t.float64)
        for fid in range(3):
            assert t.allclose(boundaries[fid], expected, atol=1e-4)

    def test_invalid_threshold_percentile_raises(self):
        """threshold_percentile outside (0, 1) fails at the typed boundary."""
        stats = _make_per_feature_stats(1)
        with pytest.raises(BeartypeCallHintParamViolation):
            build_boundaries_from_stats(
                stats, 1, num_buckets=10, threshold_percentile=1.0
            )
        with pytest.raises(BeartypeCallHintParamViolation):
            build_boundaries_from_stats(
                stats, 1, num_buckets=10, threshold_percentile=0.0
            )

    def test_too_narrow_range_raises(self):
        """Range too narrow for num_buckets raises AssertionError."""
        stats = _make_per_feature_stats(1)
        # threshold_percentile=0.95, num_buckets=10 → 9 boundaries in 4-point range
        # spacing = 4/9 ≈ 0.44 < 1.0
        with pytest.raises(AssertionError, match="too narrow"):
            build_boundaries_from_stats(
                stats, 1, num_buckets=10, threshold_percentile=0.95
            )


class TestPostFilterDiscretization:
    """End-to-end tests: post-filter boundaries produce granular buckets."""

    def test_10_buckets_p90_post_filter_has_granularity(self):
        """With post-filter boundaries, p90 filter + 10 buckets = multiple distinct buckets."""
        stats = _make_per_feature_stats(5)
        boundaries = build_boundaries_from_stats(
            stats, 5, num_buckets=10, threshold_percentile=0.9
        )

        filtered_ids, filtered_mags = _simulate_filtered_activations(
            5,
            stats,
            ThresholdType.feature_percentile,
            0.9,
            5,
        )
        assert len(filtered_ids) > 0
        buckets = discretize_activations(filtered_ids, filtered_mags, boundaries)

        unique_buckets = buckets.unique()
        assert len(unique_buckets) >= 5, (
            f"Expected >= 5 distinct buckets, got {len(unique_buckets)}: {unique_buckets.tolist()}"
        )

    def test_10_buckets_p90_bucket_distribution_uniform(self):
        """Post-filter activations spread roughly uniformly across buckets."""
        stats = _make_per_feature_stats(3, n_activations=10000)
        boundaries = build_boundaries_from_stats(
            stats, 3, num_buckets=10, threshold_percentile=0.9
        )

        filtered_ids, filtered_mags = _simulate_filtered_activations(
            3,
            stats,
            ThresholdType.feature_percentile,
            0.9,
            3,
        )
        buckets = discretize_activations(filtered_ids, filtered_mags, boundaries)

        counts = t.bincount(buckets, minlength=10)
        nonzero = (counts > 0).sum().item()
        assert nonzero >= 5, f"Expected >= 5 non-empty buckets, got {nonzero}"

    def test_combined_threshold_has_granularity(self):
        """feature_percentile_segment_topk(2048.9) + 10 buckets + post-filter boundaries."""
        stats = _make_per_feature_stats(5)
        boundaries = build_boundaries_from_stats(
            stats, 5, num_buckets=10, threshold_percentile=0.9
        )

        filtered_ids, filtered_mags = _simulate_filtered_activations(
            5,
            stats,
            ThresholdType.feature_percentile_segment_topk,
            2048.9,
            5,
        )
        assert len(filtered_ids) > 0
        buckets = discretize_activations(filtered_ids, filtered_mags, boundaries)

        unique_buckets = buckets.unique()
        assert len(unique_buckets) >= 5


class TestExtractThresholdPercentile:
    """Tests for extract_threshold_percentile helper."""

    def test_feature_percentile(self):
        cfg = ThresholdConfig(ThresholdType.feature_percentile, 0.9)
        assert extract_threshold_percentile(cfg) == 0.9

    def test_combined_threshold(self):
        cfg = ThresholdConfig(ThresholdType.feature_percentile_segment_topk, 2048.9)
        assert extract_threshold_percentile(cfg) == pytest.approx(0.9)

    def test_segment_topk_raises(self):
        cfg = ThresholdConfig(ThresholdType.segment_topk, 10)
        with pytest.raises(NotImplementedError, match="not supported"):
            extract_threshold_percentile(cfg)

    def test_segment_percentile_raises(self):
        cfg = ThresholdConfig(ThresholdType.segment_percentile, 0.5)
        with pytest.raises(NotImplementedError, match="not supported"):
            extract_threshold_percentile(cfg)
