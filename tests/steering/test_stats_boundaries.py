"""Tests for build_boundaries_from_stats in discretization module."""

import pytest
import torch as t

from steering.conditional.discretization import build_boundaries_from_stats
from steering.stats_types import PerFeatureStat
from tests.steering._discretization_reference import compute_bucket_boundaries


def _make_per_feature_stats(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    num_buckets: int,
) -> dict[str, PerFeatureStat]:
    """Helper: build a per_feature_stats dict from raw activations.

    Mimics what compute_statistics writes to feature_statistics.json by
    computing percentiles at the same points used by compute_bucket_boundaries.
    """
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


class TestBuildBoundariesFromStats:
    """Tests for build_boundaries_from_stats."""

    def test_matches_reference_for_10_buckets(self):
        """Boundaries from stats match compute_bucket_boundaries for 10 buckets.

        Creates 3 features with known distributions (uniform 1-10, 0.1-1, 5-50),
        100 samples each. Computes percentiles via torch.quantile, verifies the
        stats-based builder matches the raw-data builder within atol=1e-4.
        """
        num_buckets = 10
        n_samples = 100

        # 3 features with distinct ranges
        mags_0 = t.linspace(1.0, 10.0, n_samples)  # uniform 1-10
        mags_1 = t.linspace(0.1, 1.0, n_samples)  # uniform 0.1-1
        mags_2 = t.linspace(5.0, 50.0, n_samples)  # uniform 5-50

        feature_ids = t.cat(
            [
                t.full((n_samples,), 0, dtype=t.int32),
                t.full((n_samples,), 1, dtype=t.int32),
                t.full((n_samples,), 2, dtype=t.int32),
            ]
        )
        feature_mags = t.cat([mags_0, mags_1, mags_2])

        # Reference: compute from raw data
        ref_boundaries = compute_bucket_boundaries(
            feature_ids, feature_mags, num_buckets
        )

        # Build from stats
        per_feature_stats = _make_per_feature_stats(
            feature_ids, feature_mags, num_buckets
        )
        n_sae_features = 3
        stats_boundaries = build_boundaries_from_stats(
            per_feature_stats,
            n_sae_features,
            num_buckets,
        )

        assert stats_boundaries.shape == ref_boundaries.shape
        assert stats_boundaries.dtype == t.float64
        assert t.allclose(stats_boundaries, ref_boundaries, atol=1e-4), (
            f"Max diff: {(stats_boundaries - ref_boundaries).abs().max().item()}"
        )

    def test_missing_feature_gets_zero_boundaries(self):
        """Feature ID not present in stats dict gets all-zero boundaries."""
        per_feature_stats = {
            "0": PerFeatureStat(
                count=1,
                pct_segments_active=0.0,
                mean_mag=5.0,
                mean_mag_incl_zeros=0.0,
                std_mag=0.0,
                min_mag=5.0,
                max_mag=5.0,
                median_mag=5.0,
                percentiles={"p50": 5.0},
            ),
        }
        n_sae_features = 3  # features 0, 1, 2 -- but only 0 is in stats
        num_buckets = 2  # 1 boundary at p50

        boundaries = build_boundaries_from_stats(
            per_feature_stats,
            n_sae_features,
            num_buckets,
        )

        assert boundaries.shape == (3, 1)
        # Feature 0 should have the value from stats
        assert boundaries[0, 0].item() == pytest.approx(5.0)
        # Features 1 and 2 should be zero
        assert boundaries[1, 0].item() == 0.0
        assert boundaries[2, 0].item() == 0.0

    def test_empty_stats(self):
        """Empty stats dict produces all-zero boundaries."""
        boundaries = build_boundaries_from_stats(
            per_feature_stats={},
            n_sae_features=5,
            num_buckets=4,
        )
        assert boundaries.shape == (5, 3)
        assert (boundaries == 0).all()

    def test_different_bucket_counts(self):
        """Test with 4 and 5 buckets (not just 10)."""
        n_samples = 200
        mags = t.linspace(0.0, 1.0, n_samples)
        feature_ids = t.zeros(n_samples, dtype=t.int32)
        feature_mags = mags

        for num_buckets in (4, 5):
            ref = compute_bucket_boundaries(feature_ids, feature_mags, num_buckets)
            stats = _make_per_feature_stats(feature_ids, feature_mags, num_buckets)
            result = build_boundaries_from_stats(
                stats, n_sae_features=1, num_buckets=num_buckets
            )

            assert result.shape == ref.shape, (
                f"Shape mismatch for {num_buckets} buckets"
            )
            assert t.allclose(result, ref, atol=1e-4), (
                f"Mismatch for {num_buckets} buckets, max diff: "
                f"{(result - ref).abs().max().item()}"
            )

    def test_asserts_on_missing_percentile_key(self):
        """Raises AssertionError when a required percentile key is missing."""
        # 4 buckets need p25, p50, p75 -- provide only p25 and p50
        per_feature_stats = {
            "0": PerFeatureStat(
                count=1,
                pct_segments_active=0.0,
                mean_mag=1.5,
                mean_mag_incl_zeros=0.0,
                std_mag=0.0,
                min_mag=1.0,
                max_mag=2.0,
                median_mag=1.5,
                percentiles={"p25": 1.0, "p50": 2.0},
            ),
        }
        with pytest.raises(
            AssertionError, match="Bucket boundary percentile keys.*not found"
        ):
            build_boundaries_from_stats(
                per_feature_stats, n_sae_features=1, num_buckets=4
            )
