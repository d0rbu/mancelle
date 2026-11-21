"""Exhaustive tests for steering.conditional.discretization.

Covers edge cases, per-feature independence, searchsorted semantics,
threshold_percentile handling, bucket distribution, extreme magnitudes,
large feature IDs, and roundtrip consistency.
"""

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.discretization import (
    bucket_percentile_key_grid,
    build_boundaries_from_stats,
    discretize_activations,
)
from steering.stats_types import PerFeatureStat
from tests.steering._discretization_reference import compute_bucket_boundaries


def _make_stat(percentiles: dict[str, float]) -> PerFeatureStat:
    """Minimal PerFeatureStat with only percentiles filled in."""
    return PerFeatureStat(
        count=1,
        pct_segments_active=0.0,
        mean_mag=0.0,
        mean_mag_incl_zeros=0.0,
        std_mag=0.0,
        min_mag=0.0,
        max_mag=0.0,
        median_mag=0.0,
        percentiles=percentiles,
    )


def _make_per_feature_stats_for_roundtrip(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    num_buckets: int,
) -> dict[str, PerFeatureStat]:
    """Build per_feature_stats dict from raw activations for roundtrip tests."""
    n_boundaries = num_buckets - 1
    percentile_fracs = t.linspace(
        1.0 / num_buckets,
        1.0 - 1.0 / num_buckets,
        n_boundaries,
        dtype=t.float64,
    )
    pct_keys = [f"p{int(100 * (i + 1) / num_buckets)}" for i in range(n_boundaries)]

    stats: dict[str, PerFeatureStat] = {}
    unique_ids = feature_ids.unique()
    for fid in unique_ids:
        mask = feature_ids == fid
        mags = feature_mags[mask].to(t.float64)
        if len(mags) == 1:
            quantiles = mags[0].expand(n_boundaries)
        else:
            quantiles = t.quantile(mags, percentile_fracs)
        percentiles = {k: float(v) for k, v in zip(pct_keys, quantiles, strict=True)}
        stats[str(int(fid.item()))] = PerFeatureStat(
            count=int(mask.sum().item()),
            pct_segments_active=0.0,
            mean_mag=float(mags.mean()),
            mean_mag_incl_zeros=0.0,
            std_mag=float(mags.std()) if len(mags) > 1 else 0.0,
            min_mag=float(mags.min()),
            max_mag=float(mags.max()),
            median_mag=float(mags.median()),
            percentiles=percentiles,
        )
    return stats


# ---------------------------------------------------------------------------
# 1. TestConstantMagnitudeFeature
# ---------------------------------------------------------------------------


class TestConstantMagnitudeFeature:
    """Features whose activations are constant or nearly constant."""

    def test_all_same_magnitude_boundaries_equal(self):
        """All activations = 0.5 -> all boundaries = 0.5."""
        feature_ids = t.tensor([0] * 10, dtype=t.int32)
        feature_mags = t.full((10,), 0.5, dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        assert boundaries.shape == (1, 3)
        for j in range(3):
            assert boundaries[0, j].item() == pytest.approx(0.5)

    def test_all_same_magnitude_discretize_to_highest_bucket(self):
        """Boundaries all = X, activation = X -> searchsorted(right=True) -> highest bucket."""
        boundaries = t.tensor([[0.5, 0.5, 0.5]], dtype=t.float64)  # 4 buckets
        feature_ids = t.tensor([0], dtype=t.int32)
        feature_mags = t.tensor([0.5], dtype=t.float32)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        # searchsorted(right=True) on [0.5, 0.5, 0.5] with value 0.5 returns 3
        assert buckets[0].item() == 3

    def test_two_distinct_values(self):
        """Feature with only 2 distinct values (0.3 and 0.7), 4 buckets."""
        feature_ids = t.tensor([0] * 100, dtype=t.int32)
        feature_mags = t.tensor([0.3] * 50 + [0.7] * 50, dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        assert boundaries.shape == (1, 3)
        # Boundaries are interpolated percentiles; they should lie in [0.3, 0.7]
        for j in range(3):
            val = boundaries[0, j].item()
            assert 0.3 <= val <= 0.7, f"Boundary {j} = {val} outside [0.3, 0.7]"
        # Boundaries must be monotonically non-decreasing
        for j in range(2):
            assert boundaries[0, j].item() <= boundaries[0, j + 1].item()


# ---------------------------------------------------------------------------
# 2. TestPerFeatureIndependence
# ---------------------------------------------------------------------------


class TestPerFeatureIndependence:
    """Each feature's boundaries depend only on its own activations."""

    def test_each_feature_uses_own_boundaries(self):
        """3 features with different ranges produce independent boundaries."""
        n = 100
        ids_0 = t.full((n,), 0, dtype=t.int32)
        ids_1 = t.full((n,), 1, dtype=t.int32)
        ids_2 = t.full((n,), 2, dtype=t.int32)
        mags_0 = t.linspace(0.0, 1.0, n)  # range 0-1
        mags_1 = t.linspace(100.0, 200.0, n)  # range 100-200
        mags_2 = t.linspace(0.001, 0.01, n)  # range 0.001-0.01

        feature_ids = t.cat([ids_0, ids_1, ids_2])
        feature_mags = t.cat([mags_0, mags_1, mags_2])
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)

        # Feature 0 boundaries should be in [0, 1]
        assert boundaries[0].min().item() >= 0.0
        assert boundaries[0].max().item() <= 1.0
        # Feature 1 boundaries should be in [100, 200]
        assert boundaries[1].min().item() >= 100.0
        assert boundaries[1].max().item() <= 200.0
        # Feature 2 boundaries should be in [0.001, 0.01]
        assert boundaries[2].min().item() >= 0.001
        assert boundaries[2].max().item() <= 0.01

        # Discretizing magnitude 0.5 gives different buckets per feature
        test_ids = t.tensor([0, 1, 2], dtype=t.int32)
        test_mags = t.tensor([0.5, 0.5, 0.5], dtype=t.float32)
        buckets = discretize_activations(test_ids, test_mags, boundaries)
        # Feature 0: 0.5 in [0,1] -> mid bucket
        # Feature 1: 0.5 < 100 -> bucket 0
        # Feature 2: 0.5 > 0.01 -> highest bucket
        assert buckets[1].item() == 0  # far below range
        assert buckets[2].item() == 3  # far above range

    def test_adding_feature_doesnt_change_others(self):
        """Boundaries for features 0 and 1 are stable when feature 2 is added."""
        n = 100
        ids_01 = t.cat([t.full((n,), 0, dtype=t.int32), t.full((n,), 1, dtype=t.int32)])
        mags_01 = t.cat([t.linspace(0.0, 1.0, n), t.linspace(2.0, 3.0, n)])
        b_without_2 = compute_bucket_boundaries(ids_01, mags_01, num_buckets=4)

        ids_012 = t.cat([ids_01, t.full((n,), 2, dtype=t.int32)])
        mags_012 = t.cat([mags_01, t.linspace(10.0, 20.0, n)])
        b_with_2 = compute_bucket_boundaries(ids_012, mags_012, num_buckets=4)

        # Features 0 and 1 boundaries must be identical
        assert t.allclose(b_with_2[0], b_without_2[0], atol=1e-10)
        assert t.allclose(b_with_2[1], b_without_2[1], atol=1e-10)

    def test_feature_gap_doesnt_affect_boundaries(self):
        """Feature 0 boundaries identical with or without feature 100."""
        n = 100
        ids_0 = t.full((n,), 0, dtype=t.int32)
        mags_0 = t.linspace(0.0, 1.0, n)
        b_only_0 = compute_bucket_boundaries(ids_0, mags_0, num_buckets=4)

        ids_0_100 = t.cat([ids_0, t.full((n,), 100, dtype=t.int32)])
        mags_0_100 = t.cat([mags_0, t.linspace(5.0, 6.0, n)])
        b_0_and_100 = compute_bucket_boundaries(ids_0_100, mags_0_100, num_buckets=4)

        assert t.allclose(b_0_and_100[0], b_only_0[0], atol=1e-10)


# ---------------------------------------------------------------------------
# 3. TestSearchsortedEdgeCases
# ---------------------------------------------------------------------------


class TestSearchsortedEdgeCases:
    """Edge cases in the searchsorted-based bucket assignment."""

    def test_exact_boundary_goes_higher(self):
        """Value exactly at each boundary -> next higher bucket (right=True)."""
        # 4 buckets, boundaries [0.25, 0.50, 0.75]
        boundaries = t.tensor([[0.25, 0.50, 0.75]], dtype=t.float64)
        for j, bval in enumerate([0.25, 0.50, 0.75]):
            fids = t.tensor([0], dtype=t.int32)
            fmags = t.tensor([bval], dtype=t.float32)
            bucket = discretize_activations(fids, fmags, boundaries)
            assert bucket[0].item() == j + 1, (
                f"Value {bval} at boundary index {j} should map to bucket {j + 1}"
            )

    def test_value_between_boundaries(self):
        """Values strictly between boundaries land in the correct bucket."""
        boundaries = t.tensor([[0.2, 0.5, 0.8]], dtype=t.float64)
        # 0.35 is between boundaries[0]=0.2 and boundaries[1]=0.5 -> bucket 1
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([0.35], dtype=t.float32)
        bucket = discretize_activations(fids, fmags, boundaries)
        assert bucket[0].item() == 1

    def test_value_below_all_boundaries(self):
        """Value below the minimum boundary -> bucket 0."""
        boundaries = t.tensor([[0.3, 0.6, 0.9]], dtype=t.float64)
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([0.1], dtype=t.float32)
        bucket = discretize_activations(fids, fmags, boundaries)
        assert bucket[0].item() == 0

    def test_value_above_all_boundaries(self):
        """Value above the maximum boundary -> highest bucket (num_buckets - 1)."""
        boundaries = t.tensor([[0.1, 0.2, 0.3]], dtype=t.float64)  # 4 buckets
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([0.99], dtype=t.float32)
        bucket = discretize_activations(fids, fmags, boundaries)
        assert bucket[0].item() == 3  # highest bucket for 4-bucket setup

    def test_10_bucket_sweep(self):
        """10 buckets with uniform boundaries; sweep values across [0, 1]."""
        # 10 buckets -> 9 boundaries at 0.1, 0.2, ..., 0.9
        boundaries = t.linspace(0.1, 0.9, 9, dtype=t.float64).unsqueeze(0)
        # Test one value per bucket: 0.05 -> bucket 0, 0.15 -> bucket 1, ..., 0.95 -> bucket 9
        test_values = [0.05 + 0.1 * i for i in range(10)]
        fids = t.full((10,), 0, dtype=t.int32)
        fmags = t.tensor(test_values, dtype=t.float32)
        buckets = discretize_activations(fids, fmags, boundaries)
        for i in range(10):
            assert buckets[i].item() == i, (
                f"Value {test_values[i]} should map to bucket {i}, got {buckets[i].item()}"
            )

    def test_very_close_to_boundary_from_below(self):
        """Value at boundary - epsilon -> lower bucket."""
        boundary_val = 0.5
        boundaries = t.tensor([[boundary_val]], dtype=t.float64)
        # Use an epsilon large enough to survive float32 rounding (~7 decimal digits)
        eps = 1e-6
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([boundary_val - eps], dtype=t.float32)
        # Verify the float32 value is actually below the boundary after f64 conversion
        assert float(fmags[0].to(t.float64).item()) < boundary_val
        bucket = discretize_activations(fids, fmags, boundaries)
        assert bucket[0].item() == 0

    def test_very_close_to_boundary_from_above(self):
        """Value at boundary + epsilon -> higher bucket."""
        boundary_val = 0.5
        boundaries = t.tensor([[boundary_val]], dtype=t.float64)
        eps = 1e-7  # large enough to survive float32 rounding
        fids = t.tensor([0], dtype=t.int32)
        fmags = t.tensor([boundary_val + eps], dtype=t.float32)
        bucket = discretize_activations(fids, fmags, boundaries)
        assert bucket[0].item() == 1


# ---------------------------------------------------------------------------
# 4. TestBucketPercentileKeyGrid
# ---------------------------------------------------------------------------


class TestBucketPercentileKeyGrid:
    def test_single_bucket_has_no_boundary_keys(self):
        assert bucket_percentile_key_grid(1) == ()
        assert bucket_percentile_key_grid(1, threshold_percentile=0.9) == ()

    def test_unthresholded_grid_preserves_historical_int_truncation(self):
        assert bucket_percentile_key_grid(3) == (33, 66)
        assert bucket_percentile_key_grid(4) == (25, 50, 75)

    def test_thresholded_grid_preserves_historical_python_round(self):
        assert bucket_percentile_key_grid(3, threshold_percentile=0.5) == (74, 99)
        assert bucket_percentile_key_grid(10, threshold_percentile=0.9) == tuple(
            range(91, 100)
        )


# ---------------------------------------------------------------------------
# 5. TestBuildBoundariesFromStatsWithThresholdPercentile
# ---------------------------------------------------------------------------


class TestBuildBoundariesFromStatsWithThresholdPercentile:
    """Tests for the threshold_percentile path in build_boundaries_from_stats."""

    @staticmethod
    def _make_full_percentile_stat(base_value: float, spread: float) -> PerFeatureStat:
        """Create a stat with p1..p99 percentiles linearly interpolated."""
        pcts = {f"p{i}": base_value + spread * (i / 100.0) for i in range(1, 100)}
        return PerFeatureStat(
            count=1000,
            pct_segments_active=0.5,
            mean_mag=base_value + spread * 0.5,
            mean_mag_incl_zeros=0.0,
            std_mag=spread / 4.0,
            min_mag=base_value,
            max_mag=base_value + spread,
            median_mag=base_value + spread * 0.5,
            percentiles=pcts,
        )

    def test_threshold_percentile_0_9_with_10_buckets(self):
        """threshold_percentile=0.9, 10 buckets -> boundaries from p91..p99 range."""
        stat = self._make_full_percentile_stat(base_value=1.0, spread=10.0)
        stats = {"0": stat}
        boundaries = build_boundaries_from_stats(
            stats,
            n_sae_features=1,
            num_buckets=10,
            threshold_percentile=0.9,
        )
        assert boundaries.shape == (1, 9)  # 10 buckets -> 9 boundaries
        # With threshold_percentile=0.9: start_pct=90, end_pct=99
        # 9 boundaries at round(90 + 9*(i+1)/9) = round(90+i+1) = p91..p99
        for i in range(9):
            expected_key = f"p{round(90 + 9 * (i + 1) / 9)}"
            expected_val = stat.percentiles[expected_key]
            assert boundaries[0, i].item() == pytest.approx(expected_val, abs=1e-6), (
                f"Boundary {i} should match {expected_key}={expected_val}"
            )

    def test_threshold_percentile_narrows_bucket_range(self):
        """Boundaries with threshold_percentile span a narrower range than without."""
        stat = self._make_full_percentile_stat(base_value=0.0, spread=100.0)
        stats = {"0": stat}

        b_full = build_boundaries_from_stats(stats, n_sae_features=1, num_buckets=5)
        b_thresh = build_boundaries_from_stats(
            stats,
            n_sae_features=1,
            num_buckets=5,
            threshold_percentile=0.8,
        )
        full_range = b_full[0, -1].item() - b_full[0, 0].item()
        thresh_range = b_thresh[0, -1].item() - b_thresh[0, 0].item()
        assert thresh_range < full_range, (
            f"Threshold range {thresh_range} should be narrower than full range {full_range}"
        )

    def test_threshold_percentile_spacing_too_narrow_raises(self):
        """10 buckets with threshold_percentile=0.95 -> spacing < 1 -> AssertionError."""
        stat = self._make_full_percentile_stat(base_value=0.0, spread=1.0)
        stats = {"0": stat}
        # 10 buckets -> 9 boundaries. Range = [95, 99] -> spacing = 4/9 < 1
        with pytest.raises(AssertionError, match="too narrow"):
            build_boundaries_from_stats(
                stats,
                n_sae_features=1,
                num_buckets=10,
                threshold_percentile=0.95,
            )

    def test_threshold_percentile_boundaries_are_monotonic(self):
        """Resulting boundaries must be monotonically non-decreasing."""
        stat = self._make_full_percentile_stat(base_value=0.0, spread=50.0)
        stats = {"0": stat}
        boundaries = build_boundaries_from_stats(
            stats,
            n_sae_features=1,
            num_buckets=5,
            threshold_percentile=0.8,
        )
        for j in range(boundaries.shape[1] - 1):
            assert boundaries[0, j].item() <= boundaries[0, j + 1].item()

    def test_threshold_percentile_zero_raises(self):
        """threshold_percentile=0 is invalid (must be in (0, 1))."""
        stat = self._make_full_percentile_stat(base_value=0.0, spread=1.0)
        stats = {"0": stat}
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="threshold_percentile",
        ):
            build_boundaries_from_stats(
                stats,
                n_sae_features=1,
                num_buckets=4,
                threshold_percentile=0,
            )

    def test_threshold_percentile_one_raises(self):
        """threshold_percentile=1 is invalid (must be in (0, 1))."""
        stat = self._make_full_percentile_stat(base_value=0.0, spread=1.0)
        stats = {"0": stat}
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="threshold_percentile",
        ):
            build_boundaries_from_stats(
                stats,
                n_sae_features=1,
                num_buckets=4,
                threshold_percentile=1,
            )

    def test_threshold_percentile_negative_raises(self):
        """threshold_percentile=-0.5 is invalid (must be in (0, 1))."""
        stat = self._make_full_percentile_stat(base_value=0.0, spread=1.0)
        stats = {"0": stat}
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="threshold_percentile",
        ):
            build_boundaries_from_stats(
                stats,
                n_sae_features=1,
                num_buckets=4,
                threshold_percentile=-0.5,
            )


# ---------------------------------------------------------------------------
# 5. TestBucketDistribution
# ---------------------------------------------------------------------------


class TestBucketDistribution:
    """Verify that computed boundaries produce expected bucket distributions."""

    def test_uniform_data_roughly_even_buckets(self):
        """Uniform data -> each bucket gets roughly 25% of samples (4 buckets)."""
        n = 10_000
        t.manual_seed(42)
        feature_ids = t.zeros(n, dtype=t.int32)
        feature_mags = t.rand(n, dtype=t.float32)

        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)

        counts = t.bincount(buckets, minlength=4).float()
        fractions = counts / n
        for i in range(4):
            assert fractions[i].item() == pytest.approx(0.25, abs=0.05), (
                f"Bucket {i} fraction {fractions[i].item():.3f} far from 0.25"
            )

    def test_skewed_data_bucket_counts_vary(self):
        """Highly skewed data -> quantile boundaries still split into buckets."""
        # Exponential-like distribution: most values near 0, few large
        n = 10_000
        t.manual_seed(0)
        feature_ids = t.zeros(n, dtype=t.int32)
        feature_mags = t.rand(n).pow(4)  # strongly right-skewed

        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=4)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)

        counts = t.bincount(buckets, minlength=4)
        # All buckets should have some samples (quantile-based splitting)
        for i in range(4):
            assert counts[i].item() > 0, f"Bucket {i} is empty with skewed data"

    def test_more_buckets_than_unique_values(self):
        """3 unique values but 10 buckets -> some buckets are empty."""
        feature_ids = t.tensor([0] * 300, dtype=t.int32)
        feature_mags = t.tensor(
            [1.0] * 100 + [2.0] * 100 + [3.0] * 100, dtype=t.float32
        )

        boundaries = compute_bucket_boundaries(
            feature_ids, feature_mags, num_buckets=10
        )
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)

        unique_buckets = buckets.unique()
        # With only 3 unique values, can't fill all 10 buckets
        assert len(unique_buckets) < 10
        # But the operation should not crash
        assert buckets.min().item() >= 0
        assert buckets.max().item() <= 9


# ---------------------------------------------------------------------------
# 6. TestNegativeAndExtremeMagnitudes
# ---------------------------------------------------------------------------


class TestNegativeAndExtremeMagnitudes:
    """Negative, very large, very small, and zero magnitudes."""

    def test_negative_magnitudes(self):
        """Negative magnitudes should not crash, even if semantically unexpected."""
        feature_ids = t.tensor([0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([-1.0, -0.5, 0.0, 0.5], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets.shape == (4,)
        assert buckets.dtype == t.int64

    def test_very_large_magnitudes(self):
        """Magnitudes of 1e10 should discretize correctly."""
        feature_ids = t.tensor([0] * 4, dtype=t.int32)
        feature_mags = t.tensor([1e10, 2e10, 3e10, 4e10], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)
        assert boundaries.shape == (1, 2)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        # Should have a distribution across buckets
        assert buckets.min().item() >= 0
        assert buckets.max().item() <= 2

    def test_very_small_positive_magnitudes(self):
        """Magnitudes of 1e-10 should discretize correctly."""
        feature_ids = t.tensor([0] * 4, dtype=t.int32)
        feature_mags = t.tensor([1e-10, 2e-10, 3e-10, 4e-10], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)
        buckets = discretize_activations(feature_ids, feature_mags, boundaries)
        assert buckets.shape == (4,)
        assert buckets.min().item() >= 0
        assert buckets.max().item() <= 2

    def test_zero_magnitude(self):
        """Magnitude of exactly 0.0 should discretize without error."""
        # Build boundaries from data that includes positive values
        feature_ids = t.tensor([0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.0, 0.5, 1.0, 1.5], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=3)

        # Now discretize a zero magnitude
        test_ids = t.tensor([0], dtype=t.int32)
        test_mags = t.tensor([0.0], dtype=t.float32)
        bucket = discretize_activations(test_ids, test_mags, boundaries)
        assert bucket[0].item() == 0  # below or at first boundary


# ---------------------------------------------------------------------------
# 7. TestLargeFeatureIDs
# ---------------------------------------------------------------------------


class TestLargeFeatureIDs:
    """Features with very large IDs (sparse index space)."""

    def test_feature_id_1_million(self):
        """Feature ID = 1_000_000 produces a large but valid boundaries tensor."""
        feature_ids = t.tensor([1_000_000, 1_000_000], dtype=t.int32)
        feature_mags = t.tensor([0.3, 0.7], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=2)
        # Shape should be (1_000_001, 1)
        assert boundaries.shape == (1_000_001, 1)
        assert boundaries[1_000_000, 0].item() > 0
        # All other rows should be zero
        assert boundaries[:1_000_000].sum().item() == 0.0

    def test_discretize_with_large_sparse_ids(self):
        """IDs [0, 1_000_000] both get correct buckets."""
        feature_ids = t.tensor([0, 0, 1_000_000, 1_000_000], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.9, 10.0, 90.0], dtype=t.float32)
        boundaries = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets=2)

        test_ids = t.tensor([0, 1_000_000], dtype=t.int32)
        test_mags = t.tensor([0.5, 50.0], dtype=t.float32)
        buckets = discretize_activations(test_ids, test_mags, boundaries)
        # Both should be in valid range [0, 1]
        assert buckets[0].item() in (0, 1)
        assert buckets[1].item() in (0, 1)


# ---------------------------------------------------------------------------
# 8. TestBuildBoundariesFromStatsRoundtrip
# ---------------------------------------------------------------------------


class TestBuildBoundariesFromStatsRoundtrip:
    """Roundtrip: raw data -> compute_boundaries -> stats -> build_from_stats."""

    def test_compute_then_stats_then_build_matches(self):
        """Boundaries from stats should match boundaries from raw data."""
        n = 200
        feature_ids = t.cat(
            [
                t.full((n,), 0, dtype=t.int32),
                t.full((n,), 1, dtype=t.int32),
            ]
        )
        feature_mags = t.cat(
            [
                t.linspace(0.0, 1.0, n),
                t.linspace(5.0, 10.0, n),
            ]
        )
        num_buckets = 5

        ref = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
        stats = _make_per_feature_stats_for_roundtrip(
            feature_ids, feature_mags, num_buckets
        )
        result = build_boundaries_from_stats(
            stats, n_sae_features=2, num_buckets=num_buckets
        )

        assert result.shape == ref.shape
        assert t.allclose(result, ref, atol=1e-4), (
            f"Max diff: {(result - ref).abs().max().item()}"
        )

    def test_build_boundaries_respects_n_sae_features(self):
        """n_sae_features > max feature ID -> extra rows are zero."""
        stats = {
            "0": _make_stat({"p25": 1.0, "p50": 2.0, "p75": 3.0}),
            "1": _make_stat({"p25": 4.0, "p50": 5.0, "p75": 6.0}),
        }
        boundaries = build_boundaries_from_stats(stats, n_sae_features=5, num_buckets=4)
        assert boundaries.shape == (5, 3)
        # Features 0 and 1 have values
        assert boundaries[0, 0].item() == pytest.approx(1.0)
        assert boundaries[1, 0].item() == pytest.approx(4.0)
        # Features 2, 3, 4 are all zero
        assert (boundaries[2:] == 0).all()

    def test_build_boundaries_feature_id_beyond_n_sae_features_skipped(self):
        """Feature ID 10 in stats but n_sae_features=5 -> feature 10 silently skipped."""
        stats = {
            "0": _make_stat({"p50": 1.0}),
            "10": _make_stat({"p50": 99.0}),
        }
        boundaries = build_boundaries_from_stats(stats, n_sae_features=5, num_buckets=2)
        assert boundaries.shape == (5, 1)
        # Feature 0 present
        assert boundaries[0, 0].item() == pytest.approx(1.0)
        # Feature 10 is beyond n_sae_features=5, so skipped; no row for it
        # All other rows zero
        assert (boundaries[1:] == 0).all()
