"""Tests for steering/conditional/delta.py.

Tests cover:
- DeltaMode enum values and string conversion
- DeltaSamplingWeights enum values and string conversion
- FeatureDeltaConfig creation, defaults, validation
- parse_feature_delta: valid configs, null, missing mode, invalid mode, bounds
"""

from __future__ import annotations

from hashlib import sha256

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
    parse_feature_delta,
    sample_delta_negatives,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingSelection,
    build_percentile_table,
    parse_feature_selection,
    sample_features,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# DeltaMode enum
# ---------------------------------------------------------------------------


class TestDeltaMode:
    def test_prev_segment_value(self) -> None:
        assert DeltaMode.prev_segment.value == "prev_segment"

    def test_cum_max_value(self) -> None:
        assert DeltaMode.cum_max.value == "cum_max"

    def test_from_string(self) -> None:
        assert DeltaMode("prev_segment") is DeltaMode.prev_segment
        assert DeltaMode("cum_max") is DeltaMode.cum_max

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            DeltaMode("invalid")


# ---------------------------------------------------------------------------
# DeltaSamplingWeights enum
# ---------------------------------------------------------------------------


class TestDeltaSamplingWeights:
    def test_absolute_percentile_value(self) -> None:
        assert DeltaSamplingWeights.absolute_percentile.value == "absolute_percentile"

    def test_percentile_delta_value(self) -> None:
        assert DeltaSamplingWeights.percentile_delta.value == "percentile_delta"

    def test_from_string(self) -> None:
        assert DeltaSamplingWeights("absolute_percentile") is DeltaSamplingWeights.absolute_percentile
        assert DeltaSamplingWeights("percentile_delta") is DeltaSamplingWeights.percentile_delta

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            DeltaSamplingWeights("bad")


# ---------------------------------------------------------------------------
# FeatureDeltaConfig
# ---------------------------------------------------------------------------


class TestFeatureDeltaConfig:
    def test_defaults(self) -> None:
        cfg = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        assert cfg.mode is DeltaMode.prev_segment
        assert cfg.neg_delta_ratio == 1.0
        assert type(cfg.neg_delta_ratio) is float

    def test_custom_ratio(self) -> None:
        cfg = FeatureDeltaConfig(mode=DeltaMode.cum_max, neg_delta_ratio=0.5)
        assert cfg.neg_delta_ratio == 0.5

    def test_frozen(self) -> None:
        cfg = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        with pytest.raises(AttributeError):
            cfg.mode = DeltaMode.cum_max  # type: ignore[misc]  # ty: ignore[invalid-assignment]

    def test_rejects_plain_string_mode(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="DeltaMode"):
            FeatureDeltaConfig(mode="prev_segment")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_rejects_bool_neg_delta_ratio(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            FeatureDeltaConfig(  # type: ignore[arg-type]
                mode=DeltaMode.prev_segment,
                neg_delta_ratio=True,
            )

    def test_rejects_nan_neg_delta_ratio(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            FeatureDeltaConfig(
                mode=DeltaMode.prev_segment,
                neg_delta_ratio=float("nan"),
            )


# ---------------------------------------------------------------------------
# parse_feature_delta
# ---------------------------------------------------------------------------


class TestParseFeatureDelta:
    def test_none_returns_none(self) -> None:
        assert parse_feature_delta(None) is None

    @pytest.mark.parametrize("value", [0.3, None])
    def test_unknown_key_raises_even_when_null(self, value) -> None:
        cfg = {
            "mode": "prev_segment",
            "neg_delta_ratioo": value,
        }
        with pytest.raises(ValueError, match="neg_delta_ratioo"):
            parse_feature_delta(cfg)

    def test_complete_allowed_key_set_is_unchanged(self) -> None:
        assert parse_feature_delta(
            {"mode": "cum_max", "neg_delta_ratio": 0.25}
        ) == FeatureDeltaConfig(
            mode=DeltaMode.cum_max,
            neg_delta_ratio=0.25,
        )

    def test_prev_segment(self) -> None:
        cfg = parse_feature_delta({"mode": "prev_segment"})
        assert cfg is not None
        assert cfg.mode is DeltaMode.prev_segment
        assert cfg.neg_delta_ratio == 1.0

    def test_cum_max(self) -> None:
        cfg = parse_feature_delta({"mode": "cum_max"})
        assert cfg is not None
        assert cfg.mode is DeltaMode.cum_max

    def test_custom_neg_delta_ratio(self) -> None:
        cfg = parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": 0.3})
        assert cfg is not None
        assert cfg.neg_delta_ratio == pytest.approx(0.3)

    def test_missing_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            parse_feature_delta({"neg_delta_ratio": 0.5})

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported.*mode"):
            parse_feature_delta({"mode": "invalid"})

    def test_neg_delta_ratio_below_zero_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": -0.1})

    def test_neg_delta_ratio_above_one_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": 1.5})

    def test_neg_delta_ratio_null_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": None})

    def test_neg_delta_ratio_bool_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": True})

    def test_neg_delta_ratio_boundary_zero(self) -> None:
        cfg = parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": 0.0})
        assert cfg is not None
        assert cfg.neg_delta_ratio == 0.0

    def test_neg_delta_ratio_boundary_one(self) -> None:
        cfg = parse_feature_delta({"mode": "prev_segment", "neg_delta_ratio": 1.0})
        assert cfg is not None
        assert cfg.neg_delta_ratio == 1.0


# ---------------------------------------------------------------------------
# SamplingSelection.delta_weights
# ---------------------------------------------------------------------------


class TestSamplingSelectionDeltaWeights:
    def test_default_none(self) -> None:
        sel = SamplingSelection()
        assert sel.delta_weights is None

    def test_set_percentile_delta(self) -> None:
        sel = SamplingSelection(delta_weights=DeltaSamplingWeights.percentile_delta)
        assert sel.delta_weights is DeltaSamplingWeights.percentile_delta

    def test_set_absolute_percentile(self) -> None:
        sel = SamplingSelection(delta_weights=DeltaSamplingWeights.absolute_percentile)
        assert sel.delta_weights is DeltaSamplingWeights.absolute_percentile

    def test_frozen(self) -> None:
        sel = SamplingSelection(delta_weights=DeltaSamplingWeights.percentile_delta)
        with pytest.raises(AttributeError):
            sel.delta_weights = None  # type: ignore[misc]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# parse_feature_selection with delta_weights
# ---------------------------------------------------------------------------


class TestParseFeatureSelectionDeltaWeights:
    def test_sampling_without_delta_weights(self) -> None:
        cfg = {"distribution": "geometric", "mean": 5.0, "std": 5.0}
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.delta_weights is None

    def test_sampling_with_percentile_delta(self) -> None:
        cfg = {"distribution": "geometric", "mean": 5.0, "std": 5.0,
               "delta_weights": "percentile_delta"}
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.delta_weights is DeltaSamplingWeights.percentile_delta

    def test_sampling_with_absolute_percentile(self) -> None:
        cfg = {"distribution": "geometric", "mean": 5.0, "std": 5.0,
               "delta_weights": "absolute_percentile"}
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.delta_weights is DeltaSamplingWeights.absolute_percentile

    def test_sampling_with_invalid_delta_weights(self) -> None:
        cfg = {"distribution": "geometric", "mean": 5.0, "std": 5.0,
               "delta_weights": "bad_value"}
        with pytest.raises(ValueError, match="delta_weights"):
            parse_feature_selection(cfg)

    def test_threshold_rejects_delta_weights(self) -> None:
        cfg = {
            "threshold_type": "segment_topk",
            "threshold": 10.0,
            "delta_weights": None,
        }
        with pytest.raises(
            ValueError,
            match=r"unknown or incompatible.*delta_weights",
        ):
            parse_feature_selection(cfg)


def _make_stat(p_values: list[float]) -> PerFeatureStat:
    """Create a PerFeatureStat with p1..p99 from a list of 99 values."""
    assert len(p_values) == 99
    percentiles = {f"p{i+1}": float(v) for i, v in enumerate(p_values)}
    return PerFeatureStat(
        count=1000, pct_segments_active=0.5,
        mean_mag=float(p_values[49]), mean_mag_incl_zeros=float(p_values[49]) * 0.5,
        std_mag=1.0, min_mag=float(p_values[0]), max_mag=float(p_values[-1]),
        median_mag=float(p_values[49]),
        percentiles=percentiles,
    )


def _make_table(n_features: int = 100) -> t.Tensor:
    """Build a simple percentile table where p_k = k for all features."""
    stats = {
        str(f): _make_stat([float(k) for k in range(1, 100)])
        for f in range(n_features)
    }
    return build_percentile_table(stats, n_features)


def _make_ref_mask(fids: set[int], n: int) -> t.Tensor:
    """Helper: build a ref_mask bool tensor from a set of feature IDs."""
    mask = t.zeros(n, dtype=t.bool)
    for fid in fids:
        mask[fid] = True
    return mask


# ---------------------------------------------------------------------------
# sample_features with override_weights
# ---------------------------------------------------------------------------


class TestSampleFeaturesOverrideWeights:
    def test_override_weights_used_instead_of_percentiles(self) -> None:
        """When override_weights is provided, those weights drive sampling."""
        table = _make_table(10)
        fids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        fmags = t.tensor([50.0, 50.0, 50.0, 50.0, 50.0])
        cfg = SamplingSelection(distribution=Distribution.normal, mean=3.0, std=0.001)

        # Give feature 4 overwhelming weight
        override = t.tensor([0.01, 0.01, 0.01, 0.01, 100.0], dtype=t.float32)

        counts = t.zeros(5, dtype=t.int64)
        for seed in range(200):
            rng = t.Generator().manual_seed(seed)
            ids, _ = sample_features(fids, fmags, table, cfg, rng, override_weights=override)
            for fid in ids.tolist():
                counts[fid] += 1

        # Feature 4 should be selected more often than any other feature
        assert counts[4] == counts.max(), f"Feature 4 not dominant: {counts}"
        assert counts[4] > counts[:4].float().mean() * 1.5, f"Feature 4 not sufficiently dominant: {counts}"

    def test_override_weights_none_uses_percentiles(self) -> None:
        """When override_weights is None, normal percentile-based weighting applies."""
        table = _make_table(10)
        fids = t.tensor([0, 1, 2], dtype=t.int64)
        fmags = t.tensor([50.0, 50.0, 50.0])
        cfg = SamplingSelection(distribution=Distribution.normal, mean=2.0, std=0.001)
        rng = t.Generator().manual_seed(42)

        ids_a, mags_a = sample_features(fids, fmags, table, cfg, rng, override_weights=None)
        rng2 = t.Generator().manual_seed(42)
        ids_b, mags_b = sample_features(fids, fmags, table, cfg, rng2)

        assert t.equal(ids_a, ids_b)
        assert t.equal(mags_a, mags_b)

    def test_override_weights_returns_original_mags(self) -> None:
        """Even with override_weights, returned magnitudes are the originals."""
        table = _make_table(10)
        fids = t.tensor([0, 1, 2], dtype=t.int64)
        fmags = t.tensor([10.0, 20.0, 30.0])
        override = t.tensor([1.0, 1.0, 1.0], dtype=t.float32)
        cfg = SamplingSelection(distribution=Distribution.normal, mean=2.0, std=0.001)
        rng = t.Generator().manual_seed(42)

        ids, mags = sample_features(fids, fmags, table, cfg, rng, override_weights=override)
        # Each returned mag must match the original for that feature
        for fid, mag in zip(ids.tolist(), mags.tolist(), strict=True):
            idx = (fids == fid).nonzero(as_tuple=True)[0][0]
            assert mag == fmags[idx].item()

    def test_override_weights_empty_input(self) -> None:
        """Empty features with override_weights returns empty."""
        table = _make_table(10)
        fids = t.tensor([], dtype=t.int64)
        fmags = t.tensor([])
        override = t.tensor([], dtype=t.float32)
        cfg = SamplingSelection()
        rng = t.Generator().manual_seed(0)

        ids, mags = sample_features(fids, fmags, table, cfg, rng, override_weights=override)
        assert len(ids) == 0
        assert len(mags) == 0

    def test_override_weights_length_mismatch_raises(self) -> None:
        """override_weights must match feature_ids length."""
        table = _make_table(10)
        fids = t.tensor([0, 1, 2], dtype=t.int64)
        fmags = t.tensor([1.0, 2.0, 3.0])
        override = t.tensor([1.0, 2.0], dtype=t.float32)  # wrong length
        cfg = SamplingSelection()
        rng = t.Generator().manual_seed(0)

        with pytest.raises(ValueError, match="override_weights"):
            sample_features(fids, fmags, table, cfg, rng, override_weights=override)


# ---------------------------------------------------------------------------
# sample_delta_negatives
# ---------------------------------------------------------------------------


class TestSampleDeltaNegatives:
    @pytest.mark.parametrize(
        ("surface", "bad_id"),
        [
            ("neg_delta_fids", -1),
            ("neg_delta_fids", 8),
            ("raw_feature_ids", -1),
            ("raw_feature_ids", 8),
            ("selected_positive_ids", -1),
            ("selected_positive_ids", 8),
        ],
    )
    def test_out_of_range_input_raises_before_rng(
        self,
        surface: str,
        bad_id: int,
    ) -> None:
        inputs = {
            "neg_delta_fids": t.tensor([2, 3], dtype=t.int64),
            "raw_feature_ids": t.tensor([0, 4], dtype=t.int64),
            "selected_positive_ids": t.tensor([1], dtype=t.int64),
        }
        if surface == "neg_delta_fids":
            inputs[surface] = t.tensor([bad_id, 3], dtype=t.int64)
        elif surface == "raw_feature_ids":
            inputs[surface] = t.tensor([bad_id, 4], dtype=t.int64)
        else:
            inputs[surface] = t.tensor([bad_id], dtype=t.int64)

        rng = t.Generator().manual_seed(1234)
        initial_rng_state = rng.get_state().clone()
        with pytest.raises(
            ValueError,
            match=rf"{surface}.*{bad_id}.*\[0, 8\)",
        ):
            sample_delta_negatives(
                k=3,
                neg_delta_ratio=0.5,
                neg_delta_fids=inputs["neg_delta_fids"],
                neg_delta_weights=t.tensor([0.8, 0.4], dtype=t.float32),
                raw_feature_ids=inputs["raw_feature_ids"],
                ref_mask=t.zeros(8, dtype=t.bool),
                selected_positive_ids=inputs["selected_positive_ids"],
                n_sae_features=8,
                rng=rng,
            )
        assert t.equal(rng.get_state(), initial_rng_state)

    def test_k_zero_keeps_invalid_ids_as_a_no_op(self) -> None:
        rng = t.Generator().manual_seed(987)
        initial_rng_state = rng.get_state().clone()

        result = sample_delta_negatives(
            k=0,
            neg_delta_ratio=1.0,
            neg_delta_fids=t.tensor([-1, 8], dtype=t.int64),
            neg_delta_weights=t.tensor([0.8, 0.4], dtype=t.float32),
            raw_feature_ids=t.tensor([-1, 8], dtype=t.int64),
            ref_mask=t.zeros(8, dtype=t.bool),
            selected_positive_ids=t.tensor([-1, 8], dtype=t.int64),
            n_sae_features=8,
            rng=rng,
        )

        assert result.dtype == t.int64
        assert result.shape == (0,)
        assert t.equal(rng.get_state(), initial_rng_state)

    def test_valid_boundary_ids_match_frozen_origin_main(self) -> None:
        rng = t.Generator().manual_seed(55)

        actual = sample_delta_negatives(
            k=4,
            neg_delta_ratio=0.5,
            neg_delta_fids=t.tensor([0, 7], dtype=t.int64),
            neg_delta_weights=t.tensor([0.9, 0.4], dtype=t.float32),
            raw_feature_ids=t.tensor([0, 7], dtype=t.int64),
            ref_mask=t.tensor(
                [True, False, False, False, False, False, False, True],
                dtype=t.bool,
            ),
            selected_positive_ids=t.tensor([7], dtype=t.int64),
            n_sae_features=8,
            rng=rng,
        )
        expected = t.tensor([0, 2, 6, 1], dtype=t.int64)

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.device == expected.device
        assert (
            actual.detach().cpu().contiguous().numpy().tobytes()
            == expected.numpy().tobytes()
        )
        assert sha256(
            rng.get_state().cpu().contiguous().numpy().tobytes()
        ).hexdigest() == (
            "64bac25f5d715f4d50ffa94492600d76a4e525e861702492cd0e5a8f77a10f06"
        )

    def test_all_from_deltas_enough_available(self) -> None:
        ndf = t.tensor([100, 101, 102, 103, 104], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3, 0.8, 0.1, 0.6], dtype=t.float32)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=3, neg_delta_ratio=1.0,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.tensor([0, 1, 2], dtype=t.int64),
            ref_mask=_make_ref_mask({0, 1, 2, 100, 101, 102, 103, 104}, 200),
            selected_positive_ids=t.tensor([0], dtype=t.int64),
            n_sae_features=200,
            rng=rng,
        )
        assert len(result) == 3
        neg_set = set(ndf.tolist())
        assert all(fid in neg_set for fid in result.tolist())

    def test_all_from_deltas_not_enough(self) -> None:
        ndf = t.tensor([100, 101], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3], dtype=t.float32)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=5, neg_delta_ratio=1.0,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.tensor([0, 1], dtype=t.int64),
            ref_mask=_make_ref_mask({0, 1, 100, 101}, 200),
            selected_positive_ids=t.tensor([0], dtype=t.int64),
            n_sae_features=200,
            rng=rng,
        )
        assert len(result) == 5
        neg_set = set(ndf.tolist())
        delta_fids = {fid for fid in result.tolist() if fid in neg_set}
        assert delta_fids == {100, 101}
        uniform_fids = set(result.tolist()) - delta_fids
        assert len(uniform_fids) == 3
        excluded = {0, 1, 100, 101}
        assert uniform_fids.isdisjoint(excluded)

    def test_partial_ratio(self) -> None:
        fids_list = list(range(100, 140))
        ndf = t.tensor(fids_list, dtype=t.int64)
        ndw = t.full((40,), 0.5, dtype=t.float32)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=16, neg_delta_ratio=0.5,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.arange(10, dtype=t.int64),
            ref_mask=_make_ref_mask(set(range(10)) | set(range(100, 140)), 500),
            selected_positive_ids=t.tensor([0], dtype=t.int64),
            n_sae_features=500,
            rng=rng,
        )
        assert len(result) == 16
        neg_set = set(fids_list)
        delta_count = sum(1 for fid in result.tolist() if fid in neg_set)
        assert delta_count == 8
        uniform_count = len(result) - delta_count
        assert uniform_count == 8

    def test_partial_ratio_not_enough_deltas(self) -> None:
        ndf = t.tensor([100, 101, 102], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3, 0.8], dtype=t.float32)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=16, neg_delta_ratio=0.5,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.arange(5, dtype=t.int64),
            ref_mask=_make_ref_mask(set(range(5)) | {100, 101, 102}, 500),
            selected_positive_ids=t.tensor([0], dtype=t.int64),
            n_sae_features=500,
            rng=rng,
        )
        assert len(result) == 16
        neg_set = set(ndf.tolist())
        delta_count = sum(1 for fid in result.tolist() if fid in neg_set)
        assert delta_count == 3
        uniform_count = len(result) - delta_count
        assert uniform_count == 13

    def test_zero_ratio_all_from_uniform(self) -> None:
        ndf = t.tensor([100, 101], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3], dtype=t.float32)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=5, neg_delta_ratio=0.0,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.tensor([0], dtype=t.int64),
            ref_mask=_make_ref_mask({0, 100, 101}, 200),
            selected_positive_ids=t.tensor([0], dtype=t.int64),
            n_sae_features=200,
            rng=rng,
        )
        assert len(result) == 5
        neg_set = set(ndf.tolist())
        assert all(fid not in neg_set for fid in result.tolist())

    def test_k_zero_returns_empty(self) -> None:
        result = sample_delta_negatives(
            k=0, neg_delta_ratio=1.0,
            neg_delta_fids=t.tensor([100], dtype=t.int64),
            neg_delta_weights=t.tensor([0.5], dtype=t.float32),
            raw_feature_ids=t.tensor([0], dtype=t.int64),
            ref_mask=_make_ref_mask({0, 100}, 200),
            selected_positive_ids=t.tensor([], dtype=t.int64),
            n_sae_features=200,
            rng=t.Generator().manual_seed(0),
        )
        assert len(result) == 0

    def test_no_overlap_with_positives(self) -> None:
        ndf = t.tensor([5, 10, 15], dtype=t.int64)
        ndw = t.tensor([0.9, 0.5, 0.3], dtype=t.float32)
        selected_pos = t.tensor([5], dtype=t.int64)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=2, neg_delta_ratio=1.0,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.tensor([5, 10, 15], dtype=t.int64),
            ref_mask=_make_ref_mask({5, 10, 15}, 200),
            selected_positive_ids=selected_pos,
            n_sae_features=200,
            rng=rng,
        )
        assert 5 not in result.tolist()

    def test_delta_weights_proportional(self) -> None:
        ndf = t.tensor([100, 101], dtype=t.int64)
        ndw = t.tensor([0.01, 10.0], dtype=t.float32)
        counts = {100: 0, 101: 0}
        for seed in range(500):
            rng = t.Generator().manual_seed(seed)
            result = sample_delta_negatives(
                k=1, neg_delta_ratio=1.0,
                neg_delta_fids=ndf, neg_delta_weights=ndw,
                raw_feature_ids=t.tensor([0], dtype=t.int64),
                ref_mask=_make_ref_mask({0, 100, 101}, 200),
                selected_positive_ids=t.tensor([], dtype=t.int64),
                n_sae_features=200,
                rng=rng,
            )
            if len(result) > 0:
                counts[result[0].item()] += 1  # ty: ignore[invalid-argument-type]
        assert counts[101] > counts[100] * 3, f"High-delta not dominant: {counts}"

    def test_deterministic_with_same_seed(self) -> None:
        ndf = t.tensor([100, 101, 102], dtype=t.int64)
        ndw = t.tensor([0.5, 0.3, 0.8], dtype=t.float32)
        kwargs = {
            "k": 2, "neg_delta_ratio": 1.0,
            "neg_delta_fids": ndf, "neg_delta_weights": ndw,
            "raw_feature_ids": t.tensor([0], dtype=t.int64),
            "ref_mask": _make_ref_mask({0, 100, 101, 102}, 200),
            "selected_positive_ids": t.tensor([], dtype=t.int64),
            "n_sae_features": 200,
        }
        rng1 = t.Generator().manual_seed(42)
        rng2 = t.Generator().manual_seed(42)
        r1 = sample_delta_negatives(**kwargs, rng=rng1)  # ty: ignore[invalid-argument-type]
        r2 = sample_delta_negatives(**kwargs, rng=rng2)  # ty: ignore[invalid-argument-type]
        assert t.equal(r1, r2)

    def test_unique_results(self) -> None:
        fids_list = list(range(100, 120))
        ndf = t.tensor(fids_list, dtype=t.int64)
        ndw = t.full((20,), 0.5, dtype=t.float32)
        rng = t.Generator().manual_seed(42)
        result = sample_delta_negatives(
            k=10, neg_delta_ratio=0.5,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.arange(5, dtype=t.int64),
            ref_mask=_make_ref_mask(set(range(5)) | set(range(100, 120)), 500),
            selected_positive_ids=t.tensor([], dtype=t.int64),
            n_sae_features=500,
            rng=rng,
        )
        assert len(result) == len(set(result.tolist()))
