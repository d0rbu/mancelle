"""Sampling config parsing, refined domains, and range relationships."""

from dataclasses import asdict, replace

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.feature_pipeline import feature_selection_dirname
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
    ThresholdSelection,
    parse_feature_selection,
    parse_negative_selection,
    sample_k,
)


class TestRefinedRangeDomains:
    """Unary bounds belong to types; only the min/max relation stays explicit."""

    @pytest.mark.parametrize("min_k", [-1, True, 1.5, "1"])
    def test_constructor_rejects_invalid_min_k(self, min_k: object) -> None:
        with pytest.raises(BeartypeCallHintParamViolation):
            SamplingSelection(min_k=min_k)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("max_k", [0, -3, True, 1.5, "3"])
    def test_constructor_rejects_invalid_max_k(self, max_k: object) -> None:
        with pytest.raises(BeartypeCallHintParamViolation):
            SamplingSelection(max_k=max_k)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(("min_k", "max_k"), [(5, 5), (10, 3)])
    def test_constructor_retains_range_relation(
        self,
        min_k: int,
        max_k: int,
    ) -> None:
        with pytest.raises(ValueError, match="max_k must be > min_k"):
            SamplingSelection(min_k=min_k, max_k=max_k)

    def test_valid_constructor_range_is_stored_as_phantom_fields(self) -> None:
        selection = SamplingSelection(min_k=2, max_k=10)
        assert selection.min_k == 2
        assert selection.max_k == 10

    def test_dataclass_protocols_preserve_the_refined_values(self) -> None:
        selection = SamplingSelection(min_k=2, max_k=10)
        assert asdict(selection) == {
            "distribution": Distribution.geometric,
            "mean": 5.0,
            "std": 5.0,
            "delta_weights": None,
            "min_k": 2,
            "max_k": 10,
            "mode": selection.mode,
            "gradient": None,
        }
        assert replace(selection, max_k=12) == SamplingSelection(min_k=2, max_k=12)

    def test_explicit_null_std_is_not_treated_as_an_omitted_default(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation):
            SamplingSelection(std=None)  # ty: ignore[invalid-argument-type]

    def test_geometric_mean_override_requires_matching_std(self) -> None:
        with pytest.raises(ValueError, match="std must equal mean"):
            SamplingSelection(mean=6.0)


# ---------------------------------------------------------------------------
# parse_feature_selection
# ---------------------------------------------------------------------------


class TestParseFeatureSelection:
    """parse_feature_selection min_k/max_k handling."""

    def test_min_max_k_passed_through_to_sampling_selection(self):
        """Explicit min_k/max_k end up in the SamplingSelection dataclass."""
        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "min_k": 2,
            "max_k": 12,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.min_k == 2
        assert sel.max_k == 12

    def test_default_max_k_is_none(self):
        """When max_k not specified, it remains None (uncapped)."""
        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.max_k is None

    def test_default_min_k_is_1(self):
        """When min_k not specified, defaults to 1 for positive selection."""
        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.min_k == 1

    def test_explicit_min_max_override_defaults(self):
        """Explicit min_k/max_k override the computed defaults."""
        cfg = {
            "distribution": "normal",
            "mean": 10.0,
            "std": 3.0,
            "min_k": 0,
            "max_k": 50,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        # Would-be default max_k = int(10 + 4*3) = 22, but explicit 50 overrides
        assert sel.min_k == 0
        assert sel.max_k == 50

    def test_threshold_selection_ignores_range(self):
        """ThresholdSelection parsing does not touch min_k/max_k."""
        cfg = {
            "threshold_type": "segment_topk",
            "threshold": 10.0,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, ThresholdSelection)
        assert sel.threshold == 10.0

    def test_distribution_and_mean_std_propagate(self):
        """Distribution, mean, std are correctly stored."""
        cfg = {
            "distribution": "normal",
            "mean": 8.0,
            "std": 2.0,
            "min_k": 3,
            "max_k": 20,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        assert sel.distribution == Distribution.normal
        assert sel.mean == 8.0
        assert sel.std == 2.0

    def test_integer_numeric_inputs_keep_legacy_float_normalization(self) -> None:
        sampling = parse_feature_selection(
            {"distribution": "geometric", "mean": 5, "std": 5}
        )
        threshold = parse_feature_selection(
            {"threshold_type": "segment_topk", "threshold": 10}
        )
        assert isinstance(sampling, SamplingSelection)
        assert isinstance(threshold, ThresholdSelection)
        assert type(sampling.mean) is float
        assert type(sampling.std) is float
        assert type(threshold.threshold) is float
        assert feature_selection_dirname(sampling) == "dist_geometric/mean_5.0/std_5.0"
        assert feature_selection_dirname(threshold) == "t10.0"


# ---------------------------------------------------------------------------
# parse_negative_selection
# ---------------------------------------------------------------------------


class TestParseNegativeSelection:
    """parse_negative_selection min_k/max_k handling."""

    def test_none_returns_none(self):
        """None input produces None output."""
        assert parse_negative_selection(None) is None

    @pytest.mark.parametrize("value", [8, None])
    def test_unknown_key_raises_even_when_null(self, value):
        cfg = {
            "distribution": "geometric",
            "mean": 3.0,
            "max_kk": value,
        }
        with pytest.raises(ValueError, match="max_kk"):
            parse_negative_selection(cfg)

    def test_complete_allowed_key_set_is_unchanged(self):
        cfg = {
            "distribution": "normal",
            "mean": 7.0,
            "std": 2.0,
            "min_k": 1,
            "max_k": 9,
        }
        assert parse_negative_selection(cfg) == NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=7.0,
            std=2.0,
            min_k=1,
            max_k=9,
        )

    def test_default_min_k_is_0_for_negatives(self):
        """Negative selection defaults min_k to 0 (allow empty)."""
        cfg = {
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
        }
        sel = parse_negative_selection(cfg)
        assert isinstance(sel, NegativeSamplingSelection)
        assert sel.min_k == 0

    def test_min_max_k_passed_through(self):
        """Explicit min_k/max_k end up in NegativeSamplingSelection."""
        cfg = {
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
            "min_k": 1,
            "max_k": 8,
        }
        sel = parse_negative_selection(cfg)
        assert isinstance(sel, NegativeSamplingSelection)
        assert sel.min_k == 1
        assert sel.max_k == 8

    def test_default_max_k_is_none(self):
        """When max_k not specified, it remains None (uncapped)."""
        cfg = {
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
        }
        sel = parse_negative_selection(cfg)
        assert isinstance(sel, NegativeSamplingSelection)
        assert sel.max_k is None


# ---------------------------------------------------------------------------
# Roundtrip: parsed config -> sample_k
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Parsed config produces correct sample_k calls."""

    def test_positive_selection_roundtrip(self):
        """Parsing a config and sampling from it honours the range."""
        import torch

        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "min_k": 3,
            "max_k": 9,
        }
        sel = parse_feature_selection(cfg)
        assert isinstance(sel, SamplingSelection)
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(sel.distribution, sel.mean, sel.std, rng,
                       min_k=sel.min_k, max_k=sel.max_k)
            for _ in range(2000)
        ]
        assert all(3 <= k < 9 for k in results), (
            f"Out of range: min={min(results)}, max={max(results)}"
        )
        # Should see some variety
        assert len(set(results)) > 1

    def test_negative_selection_roundtrip(self):
        """Parsed negative config respects its own range."""
        import torch

        cfg = {
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
            "min_k": 0,
            "max_k": 6,
        }
        sel = parse_negative_selection(cfg)
        assert isinstance(sel, NegativeSamplingSelection)
        rng = torch.Generator()
        rng.manual_seed(42)
        results = [
            sample_k(sel.distribution, sel.mean, sel.std, rng,
                       min_k=sel.min_k, max_k=sel.max_k)
            for _ in range(2000)
        ]
        assert all(0 <= k < 6 for k in results), (
            f"Out of range: min={min(results)}, max={max(results)}"
        )
        # min_k=0: should see 0 at least once
        assert 0 in results, "min_k=0 never produced 0 in 2000 samples"
