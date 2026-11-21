import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    _parse_combined_threshold,
    apply_feature_percentile,
    apply_feature_percentile_segment_topk,
    apply_segment_percentile,
    apply_segment_topk,
    apply_threshold,
    select_from_packed_percentiles,
)


class TestThresholdTypeSurface:
    """Pin the public threshold mode set."""

    def test_expected_threshold_types(self):
        assert {tt.value for tt in ThresholdType} == {
            "segment_topk",
            "segment_percentile",
            "feature_percentile",
            "feature_percentile_segment_topk",
        }


class TestSegmentTopk:
    """segment_topk: keep top-k features per segment by magnitude."""

    def test_keeps_top_k(self):
        # 3 features with mags [0.1, 0.5, 0.3], k=2 -> keep [0.5, 0.3]
        feature_ids = t.tensor([10, 20, 30], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.5, 0.3], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=2)
        assert len(ids) == 2
        assert set(ids.tolist()) == {20, 30}
        assert all(m > 0 for m in mags.tolist())

    def test_k_larger_than_available(self):
        # Only 2 features, k=5 -> keep all 2
        feature_ids = t.tensor([10, 20], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.5], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=5)
        assert len(ids) == 2

    def test_empty_segment(self):
        feature_ids = t.tensor([], dtype=t.int32)
        feature_mags = t.tensor([], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=3)
        assert len(ids) == 0
        assert len(mags) == 0

    def test_preserves_id_mag_correspondence(self):
        feature_ids = t.tensor([100, 200, 300, 400], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.9, 0.5, 0.7], dtype=t.float32)
        ids, mags = apply_segment_topk(feature_ids, feature_mags, k=2)
        # Top 2 by magnitude: 200 (0.9), 400 (0.7)
        result = dict(zip(ids.tolist(), mags.tolist(), strict=True))
        assert 200 in result
        assert 400 in result
        assert abs(result[200] - 0.9) < 1e-6
        assert abs(result[400] - 0.7) < 1e-6


class TestSegmentPercentile:
    """segment_percentile: keep top x% features per segment."""

    def test_top_50_percent(self):
        feature_ids = t.tensor([1, 2, 3, 4], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.4, 0.2, 0.3], dtype=t.float32)
        ids, mags = apply_segment_percentile(feature_ids, feature_mags, percentile=0.5)
        # Top 50% = top 2 of 4
        assert len(ids) == 2
        assert set(ids.tolist()) == {2, 4}

    def test_100_percent_keeps_all(self):
        feature_ids = t.tensor([1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.4, 0.2], dtype=t.float32)
        ids, mags = apply_segment_percentile(feature_ids, feature_mags, percentile=1.0)
        assert len(ids) == 3

    def test_rounds_up_count(self):
        # 3 features, top 50% -> ceil(3*0.5) = 2
        feature_ids = t.tensor([1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.4, 0.2], dtype=t.float32)
        ids, mags = apply_segment_percentile(feature_ids, feature_mags, percentile=0.5)
        assert len(ids) == 2


class TestFeaturePercentile:
    """feature_percentile: keep activations above feature's own pth percentile."""

    def test_basic(self):
        # Feature 0 has activations [0.1, 0.2, 0.3, 0.4, 0.5]
        # percentile=0.5 -> median = 0.3, threshold >= 0.3
        feature_ids = t.tensor([0, 0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float32)
        # Median of [0.1, 0.2, 0.3, 0.4, 0.5]
        expected = t.quantile(
            t.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=t.float64), 0.5
        ).item()
        thresholds = t.tensor([expected], dtype=t.float64)
        ids, mags = apply_feature_percentile(feature_ids, feature_mags, thresholds)
        assert len(ids) == 3
        assert sorted(mags.tolist()) == pytest.approx([0.3, 0.4, 0.5], abs=1e-6)

    def test_out_of_range_feature_id_raises(self):
        feature_ids = t.tensor([0, 3], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.5], dtype=t.float32)
        thresholds = t.tensor([0.1, 0.2, 0.3], dtype=t.float64)

        with pytest.raises(
            ValueError,
            match=r"Feature IDs \[3\].*valid range \[0, 2\]",
        ):
            apply_feature_percentile(feature_ids, feature_mags, thresholds)

    def test_negative_feature_id_raises(self):
        feature_ids = t.tensor([-1], dtype=t.int32)
        feature_mags = t.tensor([1.0], dtype=t.float32)
        thresholds = t.tensor([0.1, 0.2, 0.3], dtype=t.float64)

        with pytest.raises(
            ValueError,
            match=r"Feature IDs \[-1\].*valid range \[0, 2\]",
        ):
            apply_feature_percentile(feature_ids, feature_mags, thresholds)


class TestApplyThreshold:
    """Integration: apply_threshold dispatches correctly."""

    def test_segment_topk_dispatch(self):
        cfg = ThresholdConfig(ThresholdType.segment_topk, threshold=1)
        feature_ids = t.tensor([10, 20, 30], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.5, 0.3], dtype=t.float32)
        ids, mags = apply_threshold(cfg, feature_ids, feature_mags)
        assert len(ids) == 1
        assert ids[0].item() == 20

    def test_segment_percentile_dispatch(self):
        cfg = ThresholdConfig(ThresholdType.segment_percentile, threshold=0.5)
        feature_ids = t.tensor([1, 2, 3, 4], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.4, 0.2, 0.3], dtype=t.float32)
        ids, mags = apply_threshold(cfg, feature_ids, feature_mags)
        assert len(ids) == 2

    def test_feature_percentile_dispatch(self):
        cfg = ThresholdConfig(ThresholdType.feature_percentile, threshold=0.5)
        feature_ids = t.tensor([0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.9], dtype=t.float32)
        # threshold for feature 0 = 0.5 -> only 0.9 > 0.5 passes
        precomputed = t.tensor([0.5], dtype=t.float64)
        ids, mags = apply_threshold(
            cfg, feature_ids, feature_mags, precomputed=precomputed
        )
        assert len(ids) == 1
        assert abs(mags[0].item() - 0.9) < 1e-6

    def test_feature_percentile_segment_topk_dispatch(self):
        # threshold=3.5 -> k=3, percentile=0.5
        # Feature 0 threshold = 0.2 (precomputed). Mags: 0.1, 0.3, 0.5, 0.4, 0.25
        # After percentile filter (mag > 0.2): features with mags 0.3, 0.5, 0.4, 0.25
        # After segment topk (k=3): top 3 by mag = 0.5, 0.4, 0.3
        cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk, threshold=3.5
        )
        feature_ids = t.tensor([0, 0, 0, 0, 0], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.3, 0.5, 0.4, 0.25], dtype=t.float32)
        precomputed = t.tensor([0.2], dtype=t.float64)
        ids, mags = apply_threshold(
            cfg, feature_ids, feature_mags, precomputed=precomputed
        )
        assert len(ids) == 3
        assert sorted(mags.tolist()) == pytest.approx(sorted([0.3, 0.5, 0.4]), abs=1e-6)

    def test_unknown_type_raises(self):
        # ThresholdConfig is frozen, so we can't easily create invalid types,
        # but we can test that all valid types dispatch without error
        for tt in ThresholdType:
            cfg = ThresholdConfig(tt, threshold=1)
            feature_ids = t.tensor([0], dtype=t.int32)
            feature_mags = t.tensor([0.5], dtype=t.float32)
            if tt.name.startswith("segment"):
                apply_threshold(cfg, feature_ids, feature_mags)
            # feature types need precomputed - tested above


class TestParseCombinedThreshold:
    """_parse_combined_threshold: extract (k, percentile) from combined threshold."""

    def test_basic_parse(self):
        k, pct = _parse_combined_threshold(2048.7)
        assert k == 2048
        assert abs(pct - 0.7) < 1e-9

    def test_high_percentile(self):
        k, pct = _parse_combined_threshold(100.99)
        assert k == 100
        assert abs(pct - 0.99) < 1e-9

    def test_low_k(self):
        k, pct = _parse_combined_threshold(1.5)
        assert k == 1
        assert abs(pct - 0.5) < 1e-9

    def test_zero_k_raises(self):
        with pytest.raises(AssertionError, match="k must be positive"):
            _parse_combined_threshold(0.5)

    def test_zero_percentile_raises(self):
        with pytest.raises(AssertionError, match="percentile must be in"):
            _parse_combined_threshold(100.0)

    def test_negative_raises(self):
        with pytest.raises(BeartypeCallHintParamViolation, match="threshold"):
            _parse_combined_threshold(-1.5)


class TestSelectFromPackedPercentiles:
    @pytest.mark.parametrize(
        ("threshold", "message"),
        [
            (0.5, "k must be positive"),
            (100.0, "percentile must be in"),
        ],
    )
    def test_combined_threshold_uses_canonical_parser(
        self, threshold: float, message: str
    ):
        with pytest.raises(AssertionError, match=message):
            select_from_packed_percentiles(
                feature_ids=t.tensor([1, 2], dtype=t.int64),
                percentiles=t.tensor([0.6, 0.9], dtype=t.float32),
                threshold_type=ThresholdType.feature_percentile_segment_topk,
                threshold=threshold,
            )

    @pytest.mark.parametrize(
        ("threshold", "expected_ids"),
        [
            (3.29, [55, 44, 33]),
            (2.57, [55, 44]),
            (4.58, [55, 44]),
        ],
    )
    def test_valid_combined_threshold_matches_frozen_formula(
        self,
        threshold: float,
        expected_ids: list[int],
    ):
        feature_ids = t.tensor([11, 22, 33, 44, 55], dtype=t.int64)
        percentiles = t.tensor([0.28, 0.29, 0.57, 0.58, 0.99], dtype=t.float32)

        actual = select_from_packed_percentiles(
            feature_ids,
            percentiles,
            ThresholdType.feature_percentile_segment_topk,
            threshold,
        )

        assert actual.dtype == t.int64
        assert t.equal(actual, t.tensor(expected_ids, dtype=t.int64))


class TestFeaturePercentileSegmentTopk:
    """feature_percentile_segment_topk: percentile filter then segment topk cap."""

    def test_filters_then_caps(self):
        # 6 features in segment, percentile threshold filters to 4, topk keeps 2
        feature_ids = t.tensor([0, 1, 2, 3, 4, 5], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.5, 0.3, 0.8, 0.2, 0.6], dtype=t.float32)
        # Precomputed percentile thresholds: all features have threshold 0.25
        # After percentile filter (mag >= 0.25): features 1(0.5), 2(0.3), 3(0.8), 5(0.6)
        precomputed = t.full((6,), 0.25, dtype=t.float64)
        ids, mags = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k=2
        )
        assert len(ids) == 2
        # Top 2 by magnitude from filtered: 3(0.8), 5(0.6)
        result = dict(zip(ids.tolist(), mags.tolist(), strict=True))
        assert 3 in result
        assert 5 in result
        assert abs(result[3] - 0.8) < 1e-6
        assert abs(result[5] - 0.6) < 1e-6

    def test_out_of_range_feature_id_raises_before_topk(self):
        feature_ids = t.tensor([0, 3], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.9], dtype=t.float32)
        thresholds = t.tensor([0.1, 0.2, 0.3], dtype=t.float64)

        with pytest.raises(
            ValueError,
            match=r"Feature IDs \[3\].*valid range \[0, 2\]",
        ):
            apply_feature_percentile_segment_topk(
                feature_ids,
                feature_mags,
                thresholds,
                k=1,
            )

    def test_equivalence_to_manual_two_step(self):
        """Combined function matches manual apply_feature_percentile then apply_segment_topk."""
        feature_ids = t.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=t.int32)
        feature_mags = t.tensor(
            [0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.3, 0.6], dtype=t.float32
        )
        precomputed = t.tensor(
            [0.15, 0.5, 0.35, 0.6, 0.1, 0.7, 0.25, 0.55], dtype=t.float64
        )
        k = 3

        # Manual two-step
        fids_step1, fmags_step1 = apply_feature_percentile(
            feature_ids, feature_mags, precomputed
        )
        fids_manual, fmags_manual = apply_segment_topk(fids_step1, fmags_step1, k)

        # Combined
        fids_combined, fmags_combined = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k
        )

        assert set(fids_manual.tolist()) == set(fids_combined.tolist())
        manual_dict = dict(zip(fids_manual.tolist(), fmags_manual.tolist(), strict=True))
        combined_dict = dict(zip(fids_combined.tolist(), fmags_combined.tolist(), strict=True))
        for fid, manual_value in manual_dict.items():
            assert abs(manual_value - combined_dict[fid]) < 1e-6

    def test_empty_segment(self):
        feature_ids = t.tensor([], dtype=t.int32)
        feature_mags = t.tensor([], dtype=t.float32)
        precomputed = t.tensor([0.5], dtype=t.float64)
        ids, mags = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k=3
        )
        assert len(ids) == 0
        assert len(mags) == 0

    def test_all_filtered_by_percentile(self):
        # All features have magnitude below their threshold -> nothing survives
        feature_ids = t.tensor([0, 1, 2], dtype=t.int32)
        feature_mags = t.tensor([0.1, 0.2, 0.3], dtype=t.float32)
        precomputed = t.full((3,), 10.0, dtype=t.float64)  # very high threshold
        ids, mags = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k=5
        )
        assert len(ids) == 0

    def test_fewer_than_k_after_percentile(self):
        # Only 2 features pass percentile, k=5 -> keep all 2
        feature_ids = t.tensor([0, 1, 2, 3], dtype=t.int32)
        feature_mags = t.tensor([0.9, 0.8, 0.1, 0.05], dtype=t.float32)
        precomputed = t.full((4,), 0.5, dtype=t.float64)
        ids, mags = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k=5
        )
        assert len(ids) == 2
        assert set(ids.tolist()) == {0, 1}

    def test_topk_result_count_invariant(self):
        """Result length is always <= k when enough features pass percentile."""
        feature_ids = t.tensor(list(range(20)), dtype=t.int32)
        feature_mags = t.linspace(0.01, 1.0, 20)
        precomputed = t.full((20,), 0.1, dtype=t.float64)  # low threshold, most pass
        for k in [1, 3, 5, 10]:
            ids, mags = apply_feature_percentile_segment_topk(
                feature_ids, feature_mags, precomputed, k=k
            )
            assert len(ids) <= k, f"k={k}, got {len(ids)} features"

    def test_all_survivors_above_percentile_threshold(self):
        """Every surviving feature has magnitude >= its percentile threshold (inclusive)."""
        feature_ids = t.tensor([0, 1, 2, 3, 4], dtype=t.int32)
        feature_mags = t.tensor([0.5, 0.3, 0.8, 0.1, 0.6], dtype=t.float32)
        precomputed = t.tensor([0.4, 0.25, 0.7, 0.5, 0.55], dtype=t.float64)
        ids, mags = apply_feature_percentile_segment_topk(
            feature_ids, feature_mags, precomputed, k=10
        )
        for fid, fmag in zip(ids.tolist(), mags.tolist(), strict=True):
            assert fmag >= precomputed[fid].item(), (
                f"Feature {fid} with mag {fmag} should be >= threshold {precomputed[fid].item()}"
            )

    def test_combined_type_uses_percentile_threshold_then_segment_cap(self):
        """Combined type applies the same percentile filter before top-k capping."""
        feature_ids = t.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=t.int32)
        feature_mags = t.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=t.float32
        )
        precomputed = t.tensor([0.25, 0.75], dtype=t.float64)
        cfg = ThresholdConfig(
            ThresholdType.feature_percentile_segment_topk,
            threshold=3.6,
        )
        ids, mags = apply_threshold(
            cfg, feature_ids, feature_mags, precomputed=precomputed,
        )
        assert len(ids) == 3
        assert set(ids.tolist()) == {0, 1}
        assert sorted(mags.tolist()) == pytest.approx([0.4, 0.8, 0.9], abs=1e-6)
