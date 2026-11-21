"""Property tests for the feature-selection configuration value domains."""

from __future__ import annotations

import math
from typing import Any, assert_type, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.delta import DeltaMode, FeatureDeltaConfig, parse_feature_delta
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
    ThresholdSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.thresholding import ThresholdType
from steering.refined_types import (
    FiniteNumber,
    NonNegativeFiniteNumber,
    NonNegativeInt,
    PositiveFiniteNumber,
    PositiveInt,
    UnitInterval,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=16),
)
OPTIONAL_SCALARS = st.one_of(st.none(), SCALARS)


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_positive_finite_number(value: object) -> bool:
    return _is_finite_number(value) and value > 0  # ty: ignore[unsupported-operator]


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sampling_unary_domains_hold(
    mean: object,
    std: object,
    min_k: object,
    max_k: object,
) -> bool:
    return (
        _is_finite_number(mean)
        and _is_positive_finite_number(std)
        and _is_non_negative_int(min_k)
        and (max_k is None or _is_positive_int(max_k))
    )


def _construct_sampling(
    *,
    distribution: Distribution,
    mean: object,
    std: object,
    min_k: object,
    max_k: object,
) -> SamplingSelection:
    return SamplingSelection(
        distribution=distribution,
        mean=cast(Any, mean),
        std=cast(Any, std),
        min_k=cast(Any, min_k),
        max_k=cast(Any, max_k),
    )


def _construct_negative(
    *,
    distribution: Distribution,
    mean: object,
    std: object,
    min_k: object,
    max_k: object,
) -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=distribution,
        mean=cast(Any, mean),
        std=cast(Any, std),
        min_k=cast(Any, min_k),
        max_k=cast(Any, max_k),
    )


@PROPERTY_SETTINGS
@given(mean=SCALARS, std=SCALARS, min_k=SCALARS, max_k=OPTIONAL_SCALARS)
def test_normal_sampling_constructor_matches_independent_domain_oracle(
    mean: object,
    std: object,
    min_k: object,
    max_k: object,
) -> None:
    unary_valid = _sampling_unary_domains_hold(mean, std, min_k, max_k)
    if not unary_valid:
        with pytest.raises(BeartypeCallHintParamViolation):
            _construct_sampling(
                distribution=Distribution.normal,
                mean=mean,
                std=std,
                min_k=min_k,
                max_k=max_k,
            )
        with pytest.raises(BeartypeCallHintParamViolation):
            parse_feature_selection(
                {
                    "distribution": "normal",
                    "mean": mean,
                    "std": std,
                    "min_k": min_k,
                    "max_k": max_k,
                }
            )
        return

    relation_valid = max_k is None or max_k > min_k  # ty: ignore[unsupported-operator]
    if not relation_valid:
        with pytest.raises(ValueError, match="max_k must be > min_k"):
            _construct_sampling(
                distribution=Distribution.normal,
                mean=mean,
                std=std,
                min_k=min_k,
                max_k=max_k,
            )
        with pytest.raises(ValueError, match="max_k must be > min_k"):
            parse_feature_selection(
                {
                    "distribution": "normal",
                    "mean": mean,
                    "std": std,
                    "min_k": min_k,
                    "max_k": max_k,
                }
            )
        return

    selection = _construct_sampling(
        distribution=Distribution.normal,
        mean=mean,
        std=std,
        min_k=min_k,
        max_k=max_k,
    )
    assert_type(selection.mean, FiniteNumber)
    assert_type(selection.std, PositiveFiniteNumber)
    assert_type(selection.min_k, NonNegativeInt)
    assert_type(selection.max_k, PositiveInt | None)
    assert isinstance(selection.mean, FiniteNumber)
    assert isinstance(selection.std, PositiveFiniteNumber)
    assert type(selection.mean) is float
    assert type(selection.std) is float
    assert isinstance(selection.min_k, NonNegativeInt)
    assert selection.max_k is None or isinstance(selection.max_k, PositiveInt)
    parsed = parse_feature_selection(
        {
            "distribution": "normal",
            "mean": mean,
            "std": std,
            "min_k": min_k,
            "max_k": max_k,
        }
    )
    assert parsed == selection


@PROPERTY_SETTINGS
@given(mean=SCALARS, std=SCALARS, min_k=SCALARS, max_k=OPTIONAL_SCALARS)
def test_geometric_sampling_constructor_matches_independent_domain_oracle(
    mean: object,
    std: object,
    min_k: object,
    max_k: object,
) -> None:
    unary_valid = _sampling_unary_domains_hold(mean, std, min_k, max_k)
    if not unary_valid:
        with pytest.raises(BeartypeCallHintParamViolation):
            _construct_sampling(
                distribution=Distribution.geometric,
                mean=mean,
                std=std,
                min_k=min_k,
                max_k=max_k,
            )
        with pytest.raises(BeartypeCallHintParamViolation):
            parse_feature_selection(
                {
                    "distribution": "geometric",
                    "mean": mean,
                    "std": std,
                    "min_k": min_k,
                    "max_k": max_k,
                }
            )
        return

    relation_valid = (
        mean > 0  # ty: ignore[unsupported-operator]
        and std == mean
        and (max_k is None or max_k > min_k)  # ty: ignore[unsupported-operator]
    )
    if not relation_valid:
        with pytest.raises(ValueError):
            _construct_sampling(
                distribution=Distribution.geometric,
                mean=mean,
                std=std,
                min_k=min_k,
                max_k=max_k,
            )
        with pytest.raises(ValueError):
            parse_feature_selection(
                {
                    "distribution": "geometric",
                    "mean": mean,
                    "std": std,
                    "min_k": min_k,
                    "max_k": max_k,
                }
            )
        return

    selection = _construct_sampling(
        distribution=Distribution.geometric,
        mean=mean,
        std=std,
        min_k=min_k,
        max_k=max_k,
    )
    assert selection.mean == mean
    assert selection.std == std
    assert type(selection.mean) is float
    assert type(selection.std) is float
    parsed = parse_feature_selection(
        {
            "distribution": "geometric",
            "mean": mean,
            "std": std,
            "min_k": min_k,
            "max_k": max_k,
        }
    )
    assert parsed == selection


@PROPERTY_SETTINGS
@given(
    distribution=st.sampled_from(list(Distribution)),
    mean=SCALARS,
    std=SCALARS,
    min_k=SCALARS,
    max_k=OPTIONAL_SCALARS,
)
def test_negative_sampling_constructor_matches_same_shared_contract(
    distribution: Distribution,
    mean: object,
    std: object,
    min_k: object,
    max_k: object,
) -> None:
    unary_valid = _sampling_unary_domains_hold(mean, std, min_k, max_k)
    if not unary_valid:
        with pytest.raises(BeartypeCallHintParamViolation):
            _construct_negative(
                distribution=distribution,
                mean=mean,
                std=std,
                min_k=min_k,
                max_k=max_k,
            )
        with pytest.raises(BeartypeCallHintParamViolation):
            parse_negative_selection(
                {
                    "distribution": distribution.value,
                    "mean": mean,
                    "std": std,
                    "min_k": min_k,
                    "max_k": max_k,
                }
            )
        return

    relation_valid = max_k is None or max_k > min_k  # ty: ignore[unsupported-operator]
    if distribution is Distribution.geometric:
        relation_valid = (
            relation_valid
            and mean > 0  # ty: ignore[unsupported-operator]
            and std == mean
        )
    if not relation_valid:
        with pytest.raises(ValueError):
            _construct_negative(
                distribution=distribution,
                mean=mean,
                std=std,
                min_k=min_k,
                max_k=max_k,
            )
        with pytest.raises(ValueError):
            parse_negative_selection(
                {
                    "distribution": distribution.value,
                    "mean": mean,
                    "std": std,
                    "min_k": min_k,
                    "max_k": max_k,
                }
            )
        return

    selection = _construct_negative(
        distribution=distribution,
        mean=mean,
        std=std,
        min_k=min_k,
        max_k=max_k,
    )
    assert isinstance(selection.mean, FiniteNumber)
    assert isinstance(selection.std, PositiveFiniteNumber)
    assert type(selection.mean) is float
    assert type(selection.std) is float
    assert isinstance(selection.min_k, NonNegativeInt)
    assert selection.max_k is None or isinstance(selection.max_k, PositiveInt)
    parsed = parse_negative_selection(
        {
            "distribution": distribution.value,
            "mean": mean,
            "std": std,
            "min_k": min_k,
            "max_k": max_k,
        }
    )
    assert parsed == selection


@PROPERTY_SETTINGS
@given(threshold=SCALARS)
def test_threshold_selection_matches_non_negative_finite_oracle(
    threshold: object,
) -> None:
    if not (_is_finite_number(threshold) and threshold >= 0):  # ty: ignore[unsupported-operator]
        with pytest.raises(BeartypeCallHintParamViolation):
            ThresholdSelection(
                ThresholdType.feature_percentile,
                cast(Any, threshold),
            )
        with pytest.raises(BeartypeCallHintParamViolation):
            parse_feature_selection(
                {
                    "threshold_type": "feature_percentile",
                    "threshold": threshold,
                }
            )
        return

    selection = ThresholdSelection(
        ThresholdType.feature_percentile,
        cast(Any, threshold),
    )
    assert_type(selection.threshold, NonNegativeFiniteNumber)
    assert isinstance(selection.threshold, NonNegativeFiniteNumber)
    assert type(selection.threshold) is float
    assert selection.threshold == threshold
    assert parse_feature_selection(
        {
            "threshold_type": "feature_percentile",
            "threshold": threshold,
        }
    ) == selection


@PROPERTY_SETTINGS
@given(ratio=SCALARS)
def test_feature_delta_matches_unit_interval_oracle(ratio: object) -> None:
    valid = (
        _is_finite_number(ratio)
        and ratio >= 0  # ty: ignore[unsupported-operator]
        and ratio <= 1  # ty: ignore[unsupported-operator]
    )
    if not valid:
        with pytest.raises(BeartypeCallHintParamViolation):
            FeatureDeltaConfig(
                DeltaMode.prev_segment,
                cast(Any, ratio),
            )
        with pytest.raises(BeartypeCallHintParamViolation):
            parse_feature_delta(
                {
                    "mode": "prev_segment",
                    "neg_delta_ratio": ratio,
                }
            )
        return

    config = FeatureDeltaConfig(DeltaMode.prev_segment, cast(Any, ratio))
    assert_type(config.neg_delta_ratio, UnitInterval)
    assert isinstance(config.neg_delta_ratio, UnitInterval)
    assert type(config.neg_delta_ratio) is float
    assert config.neg_delta_ratio == ratio
    assert parse_feature_delta(
        {
            "mode": "prev_segment",
            "neg_delta_ratio": ratio,
        }
    ) == config


@PROPERTY_SETTINGS
@given(
    mean=st.floats(min_value=1e-6, max_value=100.0, allow_nan=False),
    min_k=st.integers(min_value=0, max_value=100),
    width=st.integers(min_value=1, max_value=100),
)
def test_raw_parsers_roundtrip_valid_refined_domains(
    mean: float,
    min_k: int,
    width: int,
) -> None:
    max_k = min_k + width
    positive = parse_feature_selection(
        {
            "distribution": "geometric",
            "mean": mean,
            "min_k": min_k,
            "max_k": max_k,
        }
    )
    negative = parse_negative_selection(
        {
            "distribution": "normal",
            "mean": mean,
            "std": mean,
            "min_k": min_k,
            "max_k": max_k,
        }
    )
    assert isinstance(positive, SamplingSelection)
    assert isinstance(negative, NegativeSamplingSelection)
    assert isinstance(positive.mean, FiniteNumber)
    assert isinstance(positive.std, PositiveFiniteNumber)
    assert isinstance(positive.min_k, NonNegativeInt)
    assert isinstance(positive.max_k, PositiveInt)
    assert isinstance(negative.mean, FiniteNumber)
    assert isinstance(negative.std, PositiveFiniteNumber)
    assert isinstance(negative.min_k, NonNegativeInt)
    assert isinstance(negative.max_k, PositiveInt)
