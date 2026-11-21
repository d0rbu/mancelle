"""Tests for build_thresholds_from_stats in thresholding module.

Run with:
    uv run pytest tests/test_stats_thresholds.py -v
"""

import pytest
import torch as t

from steering.conditional.thresholding import (
    ThresholdType,
    build_thresholds_from_stats,
)
from steering.stats_types import PerFeatureStat


def _make_per_feature_stats(
    feature_ids: t.Tensor,
    feature_mags: t.Tensor,
    threshold: float,
) -> dict[str, PerFeatureStat]:
    """Helper: build a per_feature_stats dict from raw activations.

    Mimics what compute_statistics writes to feature_statistics.json by
    computing per-feature percentiles and counts via torch.quantile.
    """
    pct_key = f"p{int(threshold * 100)}"

    stats: dict[str, PerFeatureStat] = {}
    unique_ids = feature_ids.unique()
    for fid in unique_ids:
        mask = feature_ids == fid
        mags = feature_mags[mask].to(t.float64)
        count = int(mask.sum().item())
        pct_val = float(t.quantile(mags, threshold).item())
        stats[str(int(fid.item()))] = PerFeatureStat(
            count=count,
            pct_segments_active=0.0,
            mean_mag=float(mags.mean()),
            mean_mag_incl_zeros=0.0,
            std_mag=float(mags.std()) if len(mags) > 1 else 0.0,
            min_mag=float(mags.min()),
            max_mag=float(mags.max()),
            median_mag=float(mags.median()),
            percentiles={pct_key: pct_val},
        )

    return stats


def _make_percentile_sentinel_stat() -> PerFeatureStat:
    """Return one feature with pN=N so the selected key is observable."""
    return PerFeatureStat(
        count=100,
        pct_segments_active=0.5,
        mean_mag=50.0,
        mean_mag_incl_zeros=25.0,
        std_mag=1.0,
        min_mag=1.0,
        max_mag=99.0,
        median_mag=50.0,
        percentiles={f"p{k}": float(k) for k in range(1, 100)},
    )


class TestFeaturePercentileFromStats:
    """build_thresholds_from_stats with feature_percentile type."""

    def test_feature_percentile_matches_reference(self):
        """Thresholds from stats match the stored per-feature percentiles.

        Creates 3 features with different sample counts (50, 30, 20),
        computes percentiles via torch.quantile, and verifies the stats-based
        builder reads exactly those stored percentile values.
        """
        threshold = 0.7

        mags_0 = t.linspace(0.1, 5.0, 50)
        mags_1 = t.linspace(0.5, 3.0, 30)
        mags_2 = t.linspace(1.0, 10.0, 20)

        feature_ids = t.cat(
            [
                t.full((50,), 0, dtype=t.int32),
                t.full((30,), 1, dtype=t.int32),
                t.full((20,), 2, dtype=t.int32),
            ]
        )
        feature_mags = t.cat([mags_0, mags_1, mags_2])

        # Build from stats
        per_feature_stats = _make_per_feature_stats(
            feature_ids,
            feature_mags,
            threshold,
        )
        n_sae_features = 3
        stats_thresholds = build_thresholds_from_stats(
            per_feature_stats,
            n_sae_features,
            ThresholdType.feature_percentile,
            threshold,
        )

        assert stats_thresholds.shape == (n_sae_features,)
        assert stats_thresholds.dtype == t.float64
        pct_key = f"p{int(threshold * 100)}"
        expected = t.tensor(
            [per_feature_stats[str(fid)].percentiles[pct_key] for fid in range(3)],
            dtype=t.float64,
        )
        assert t.allclose(stats_thresholds[:3], expected, atol=1e-4), (
            f"Max diff: {(stats_thresholds[:3] - expected).abs().max().item()}"
        )

    def test_empty_stats_returns_all_inf(self):
        """Empty per_feature_stats with feature_percentile returns all-inf tensor."""
        result = build_thresholds_from_stats(
            per_feature_stats={},
            n_sae_features=5,
            threshold_type=ThresholdType.feature_percentile,
            threshold=0.5,
        )
        assert isinstance(result, t.Tensor)
        assert result.shape == (5,)
        assert result.dtype == t.float64
        assert (result == float("inf")).all()

    def test_feature_percentile_missing_pct_key_raises(self):
        """Raises AssertionError when a required percentile key is missing."""
        per_feature_stats = {
            "0": PerFeatureStat(
                count=10,
                pct_segments_active=0.0,
                mean_mag=1.5,
                mean_mag_incl_zeros=0.0,
                std_mag=0.0,
                min_mag=1.0,
                max_mag=2.0,
                median_mag=1.5,
                percentiles={"p50": 1.5},
            ),
        }
        # threshold=0.7 needs "p70" but only "p50" is present
        with pytest.raises(AssertionError, match="p70"):
            build_thresholds_from_stats(
                per_feature_stats,
                n_sae_features=1,
                threshold_type=ThresholdType.feature_percentile,
                threshold=0.7,
            )


class TestFeaturePercentileSegmentTopkFromStats:
    """build_thresholds_from_stats with feature_percentile_segment_topk type."""

    def test_matches_feature_percentile_reference(self):
        """Combined thresholds read the same stored percentile values.

        The combined type should produce the same per-feature percentile
        thresholds as feature_percentile (the segment topk is applied at
        runtime, not while building threshold tensors).
        """
        # threshold=5.7 -> k=5, percentile=0.7
        combined_threshold = 5.7
        pct = 0.7

        mags_0 = t.linspace(0.1, 5.0, 50)
        mags_1 = t.linspace(0.5, 3.0, 30)
        mags_2 = t.linspace(1.0, 10.0, 20)

        feature_ids = t.cat(
            [
                t.full((50,), 0, dtype=t.int32),
                t.full((30,), 1, dtype=t.int32),
                t.full((20,), 2, dtype=t.int32),
            ]
        )
        feature_mags = t.cat([mags_0, mags_1, mags_2])

        # Build from stats using combined type
        per_feature_stats = _make_per_feature_stats(
            feature_ids,
            feature_mags,
            pct,
        )
        n_sae_features = 3
        stats_thresholds = build_thresholds_from_stats(
            per_feature_stats,
            n_sae_features,
            ThresholdType.feature_percentile_segment_topk,
            combined_threshold,
        )

        assert stats_thresholds.shape == (n_sae_features,)
        assert stats_thresholds.dtype == t.float64
        pct_key = f"p{int(pct * 100)}"
        expected = t.tensor(
            [per_feature_stats[str(fid)].percentiles[pct_key] for fid in range(3)],
            dtype=t.float64,
        )
        assert t.allclose(stats_thresholds[:3], expected, atol=1e-4), (
            f"Max diff: {(stats_thresholds[:3] - expected).abs().max().item()}"
        )

    def test_missing_pct_key_raises(self):
        """Raises AssertionError when a required percentile key is missing."""
        per_feature_stats = {
            "0": PerFeatureStat(
                count=10,
                pct_segments_active=0.0,
                mean_mag=1.5,
                mean_mag_incl_zeros=0.0,
                std_mag=0.0,
                min_mag=1.0,
                max_mag=2.0,
                median_mag=1.5,
                percentiles={"p50": 1.5},
            ),
        }
        # threshold=5.7 needs "p70" but only "p50" is present
        with pytest.raises(AssertionError, match="p70"):
            build_thresholds_from_stats(
                per_feature_stats,
                n_sae_features=1,
                threshold_type=ThresholdType.feature_percentile_segment_topk,
                threshold=5.7,
            )

    def test_empty_stats_returns_all_inf(self):
        """Empty per_feature_stats returns all-inf tensor."""
        result = build_thresholds_from_stats(
            per_feature_stats={},
            n_sae_features=5,
            threshold_type=ThresholdType.feature_percentile_segment_topk,
            threshold=10.9,
        )
        assert isinstance(result, t.Tensor)
        assert result.shape == (5,)
        assert (result == float("inf")).all()


class TestLegacyPercentileKeyProjection:
    """Pin F2's known-wrong projection until identity migration is authorized."""

    @pytest.mark.parametrize(
        ("threshold_type", "threshold", "legacy_value"),
        [
            (ThresholdType.feature_percentile, 0.29, 28.0),
            (ThresholdType.feature_percentile, 0.57, 56.0),
            (ThresholdType.feature_percentile, 0.58, 57.0),
            (ThresholdType.feature_percentile_segment_topk, 4.29, 28.0),
            (ThresholdType.feature_percentile_segment_topk, 4.57, 56.0),
            (ThresholdType.feature_percentile_segment_topk, 4.58, 57.0),
        ],
    )
    def test_binary_float_hundredths_retain_legacy_values(
        self,
        threshold_type: ThresholdType,
        threshold: float,
        legacy_value: float,
    ):
        actual = build_thresholds_from_stats(
            {"0": _make_percentile_sentinel_stat()},
            n_sae_features=1,
            threshold_type=threshold_type,
            threshold=threshold,
        )

        assert t.equal(actual, t.tensor([legacy_value], dtype=t.float64))


class TestUnsupportedThresholdTypes:
    """build_thresholds_from_stats raises for unsupported types."""

    def test_segment_topk_raises(self):
        with pytest.raises(AssertionError):
            build_thresholds_from_stats(
                per_feature_stats={},
                n_sae_features=1,
                threshold_type=ThresholdType.segment_topk,
                threshold=5,
            )

    def test_segment_percentile_raises(self):
        """segment_percentile is also unsupported."""
        with pytest.raises(AssertionError):
            build_thresholds_from_stats(
                per_feature_stats={},
                n_sae_features=1,
                threshold_type=ThresholdType.segment_percentile,
                threshold=0.5,
            )
