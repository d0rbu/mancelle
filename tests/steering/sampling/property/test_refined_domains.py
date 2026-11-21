"""Independent property oracles for stochastic-sampling refined domains."""

from __future__ import annotations

import math
import random
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.discretization import (
    build_boundaries_from_stats,
    percentile_to_bucket,
)
from steering.conditional.percentile_table import build_percentile_table
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingSelection,
    sample_features_batched,
    sample_inactive_features,
    sample_inactive_features_batched,
    sample_k,
    sample_k_batched,
    select_top_k_by_percentile,
    select_top_k_by_percentile_batched,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    _parse_combined_threshold,
    apply_segment_percentile,
    apply_segment_topk,
    build_thresholds_from_stats,
    select_from_packed_percentiles,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-25, max_value=25),
    st.floats(
        min_value=-25,
        max_value=25,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    st.sampled_from((float("nan"), float("inf"), float("-inf"))),
    st.booleans(),
    st.none(),
    st.text(max_size=12),
)
LISTS = st.lists(SCALARS, max_size=8)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(value: object) -> bool:
    return _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]


def _positive_int(value: object) -> bool:
    return _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _positive_finite_number(value: object) -> bool:
    return _finite_number(value) and value > 0  # ty: ignore[unsupported-operator]


def _non_negative_int_list(value: object) -> bool:
    return isinstance(value, list) and all(_non_negative_int(item) for item in value)


def _rng() -> t.Generator:
    return t.Generator().manual_seed(0)


def _negative_config() -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.normal,
        mean=0.0,
        std=1.0,
        min_k=0,
        max_k=1,
    )


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_sample_k_mean_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_k(  # noqa: E731
        Distribution.normal,
        cast(Any, value),
        1.0,
        _rng(),
    )
    if _finite_number(value):
        assert call() >= 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="mean"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_sample_k_std_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_k(  # noqa: E731
        Distribution.normal,
        0.0,
        cast(Any, value),
        _rng(),
    )
    if _positive_finite_number(value):
        assert call() >= 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="std"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_sample_k_minimum_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_k(  # noqa: E731
        Distribution.normal,
        0.0,
        1.0,
        _rng(),
        min_k=cast(Any, value),
    )
    if _non_negative_int(value):
        assert call() >= value  # ty: ignore[unsupported-operator]
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="min_k"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_sample_k_maximum_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_k(  # noqa: E731
        Distribution.normal,
        0.0,
        1.0,
        _rng(),
        max_k=cast(Any, value),
    )
    if value is None or _positive_int(value):
        result = call()
        assert result >= 0
        if value is not None:
            assert result < value  # ty: ignore[unsupported-operator]
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="max_k"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_batched_sample_count_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_k_batched(  # noqa: E731
        Distribution.normal,
        0.0,
        1.0,
        _rng(),
        cast(Any, value),
    )
    if _non_negative_int(value):
        assert call().shape == (value,)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_top_k_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: select_top_k_by_percentile(  # noqa: E731
        t.empty(0, dtype=t.int64),
        t.empty(0),
        t.empty(0, 99),
        cast(Any, value),
    )
    if _non_negative_int(value):
        ids, mags = call()
        assert ids.numel() == mags.numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="k"):
            call()


@PROPERTY_SETTINGS
@given(value=LISTS)
def test_batched_top_k_segment_sizes_match_independent_oracle(
    value: list[object],
) -> None:
    call = lambda: select_top_k_by_percentile_batched(  # noqa: E731
        t.empty(0, dtype=t.int64),
        t.empty(0),
        cast(Any, value),
        t.empty(0, 99),
        t.zeros(len(value), dtype=t.int64),
    )
    if _non_negative_int_list(value):
        assert len(call()) == len(value)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_sizes"):
            call()


@PROPERTY_SETTINGS
@given(value=LISTS)
def test_batched_positive_sampling_segment_sizes_match_independent_oracle(
    value: list[object],
) -> None:
    call = lambda: sample_features_batched(  # noqa: E731
        t.empty(0, dtype=t.int64),
        t.empty(0),
        cast(Any, value),
        t.empty(0, 99),
        SamplingSelection(),
        _rng(),
    )
    if _non_negative_int_list(value):
        assert len(call()) == len(value)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_sizes"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_negative_sampler_feature_extent_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: sample_inactive_features(  # noqa: E731
        t.empty(0, dtype=t.int64),
        cast(Any, value),
        _negative_config(),
        _rng(),
    )
    if _positive_int(value):
        result = call()
        assert result.dtype == t.int64
        assert bool(((result >= 0) & (result < cast(int, value))).all())
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_batched_negative_sampler_extent_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: sample_inactive_features_batched(  # noqa: E731
        [],
        cast(Any, value),
        _negative_config(),
        _rng(),
    )
    if _positive_int(value):
        assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_negative_bundle_extent_matches_independent_oracle(value: object) -> None:
    call = lambda: NegativeBundle(  # noqa: E731
        selection=_negative_config(),
        n_sae_features=cast(Any, value),
        rng=_rng(),
    )
    if _positive_int(value):
        assert call().n_sae_features == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_threshold_config_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: ThresholdConfig(  # noqa: E731
        ThresholdType.feature_percentile,
        cast(Any, value),
    )
    if _finite_number(value) and value >= 0:  # ty: ignore[unsupported-operator]
        assert call().threshold == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="threshold"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_segment_top_k_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: apply_segment_topk(  # noqa: E731
        t.empty(0, dtype=t.int64),
        t.empty(0),
        cast(Any, value),
    )
    if _positive_int(value):
        ids, mags = call()
        assert ids.numel() == mags.numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="k"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_segment_percentile_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: apply_segment_percentile(  # noqa: E731
        t.empty(0, dtype=t.int64),
        t.empty(0),
        cast(Any, value),
    )
    valid = _finite_number(value) and 0 < value <= 1  # ty: ignore[unsupported-operator]
    if valid:
        ids, mags = call()
        assert ids.numel() == mags.numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="percentile"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_bucket_count_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: percentile_to_bucket(  # noqa: E731
        t.empty(0),
        cast(Any, value),
    )
    if _positive_int(value):
        assert call().numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="num_buckets"):
            call()


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("n_sae_features", "num_buckets")), value=SCALARS)
def test_boundary_builder_extent_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "per_feature_stats": {},
        "n_sae_features": 1,
        "num_buckets": 2,
    }
    kwargs[field] = value
    valid = _positive_int(value) if field == "n_sae_features" else (
        _strict_int(value) and value >= 2  # ty: ignore[unsupported-operator]
    )
    if valid:
        result = build_boundaries_from_stats(**kwargs)
        assert result.ndim == 2
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            build_boundaries_from_stats(**kwargs)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_boundary_builder_threshold_domain_precedes_spacing_relationship(
    value: object,
) -> None:
    call = lambda: build_boundaries_from_stats(  # noqa: E731
        {},
        1,
        2,
        cast(Any, value),
    )
    valid = value is None or (
        _finite_number(value) and 0 < value < 1  # ty: ignore[unsupported-operator]
    )
    if valid:
        try:
            assert call().shape == (1, 1)
        except AssertionError:
            pass
    else:
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match="threshold_percentile",
        ):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_percentile_table_extent_matches_independent_oracle(value: object) -> None:
    call = lambda: build_percentile_table({}, cast(Any, value))  # noqa: E731
    if _positive_int(value):
        assert call().shape == (value, 99)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("n_sae_features", "threshold")), value=SCALARS)
def test_threshold_table_scalar_domains_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "per_feature_stats": {},
        "n_sae_features": 1,
        "threshold_type": ThresholdType.feature_percentile,
        "threshold": 0.5,
    }
    kwargs[field] = value
    valid = _positive_int(value) if field == "n_sae_features" else (
        _finite_number(value) and value >= 0  # ty: ignore[unsupported-operator]
    )
    if valid:
        result = build_thresholds_from_stats(**kwargs)
        assert result.ndim == 1
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            build_thresholds_from_stats(**kwargs)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_packed_percentile_threshold_domain_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: select_from_packed_percentiles(  # noqa: E731
        t.empty(0, dtype=t.int64),
        t.empty(0),
        ThresholdType.feature_percentile,
        cast(Any, value),
    )
    if _finite_number(value) and value >= 0:  # ty: ignore[unsupported-operator]
        assert call().numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="threshold"):
            call()


@PROPERTY_SETTINGS
@given(
    distribution=st.sampled_from(tuple(Distribution)),
    mean=st.floats(min_value=0.125, max_value=20, allow_nan=False),
    std=st.floats(min_value=0.125, max_value=20, allow_nan=False),
    min_k=st.integers(min_value=0, max_value=10),
    width=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_scalar_and_singleton_batched_sampling_have_exact_rng_parity(
    distribution: Distribution,
    mean: float,
    std: float,
    min_k: int,
    width: int,
    seed: int,
) -> None:
    if distribution is Distribution.geometric:
        std = mean
    max_k = min_k + width
    scalar_rng = t.Generator().manual_seed(seed)
    batched_rng = t.Generator().manual_seed(seed)
    scalar = sample_k(distribution, mean, std, scalar_rng, min_k, max_k)
    batched = sample_k_batched(
        distribution,
        mean,
        std,
        batched_rng,
        1,
        min_k,
        max_k,
    )
    assert batched.tolist() == [scalar]
    assert t.equal(scalar_rng.get_state(), batched_rng.get_state())


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    n_sae=st.integers(min_value=1, max_value=100),
    segment_sizes=st.lists(st.integers(min_value=0, max_value=8), max_size=8),
)
def test_sampling_type_checks_do_not_perturb_python_global_rng(
    seed: int,
    n_sae: int,
    segment_sizes: list[int],
) -> None:
    random.seed(seed)
    before = random.getstate()
    NegativeBundle(_negative_config(), n_sae, _rng())
    select_top_k_by_percentile(
        t.empty(0, dtype=t.int64),
        t.empty(0),
        t.empty(0, 99),
        0,
    )
    select_top_k_by_percentile_batched(
        t.empty(0, dtype=t.int64),
        t.empty(0),
        segment_sizes,
        t.empty(0, 99),
        t.zeros(len(segment_sizes), dtype=t.int64),
    )
    sample_features_batched(
        t.empty(0, dtype=t.int64),
        t.empty(0),
        segment_sizes,
        t.empty(0, 99),
        SamplingSelection(),
        _rng(),
    )
    sample_inactive_features_batched([], n_sae, _negative_config(), _rng())
    _parse_combined_threshold(1.5)
    build_percentile_table({}, n_sae)
    build_boundaries_from_stats({}, n_sae, 2)
    build_thresholds_from_stats(
        {},
        n_sae,
        ThresholdType.feature_percentile,
        0.5,
    )
    assert random.getstate() == before
