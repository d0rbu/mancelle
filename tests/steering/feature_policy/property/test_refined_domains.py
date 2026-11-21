"""Independent property oracles for feature-policy refined domains."""

from __future__ import annotations

import math
import random
from typing import Any, assert_type, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from errors import FeatureClustersError, FeatureMaskError
from steering.conditional.delta import (
    DeltaMode,
    compute_doc_counterfactual_deltas,
    compute_doc_deltas,
    sample_delta_negatives,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.feature_clusters import (
    FEATURE_CLUSTERS_SEMANTICS,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    resolve_feature_mask,
)
from steering.conditional.freq_blacklist import (
    BlacklistResult,
    FrequencyBlacklistOutcome,
    FrequencyBlacklistStatus,
    FrequencyBounds,
)
from steering.refined_types import (
    LowercaseSha256,
    NonEmptyNonNegativeIntTuple,
    NonNegativeInt,
    NonNegativeIntTuple,
    PositiveInt,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-25, max_value=100),
    st.floats(
        min_value=-25,
        max_value=100,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    st.sampled_from((float("nan"), float("inf"), float("-inf"))),
    st.booleans(),
    st.none(),
    st.text(max_size=70),
)
TUPLES = st.lists(SCALARS, max_size=8).map(tuple)
LISTS = st.lists(SCALARS, max_size=8)
TENSORISH = st.one_of(
    st.just(t.empty(0)),
    st.just(t.empty(0, dtype=t.int64)),
    SCALARS,
)
OPTIONAL_TENSORISH = st.one_of(st.none(), TENSORISH)
COUNTERFACTUAL_REFS = st.lists(
    st.one_of(
        st.none(),
        st.tuples(
            st.just(t.empty(0, dtype=t.int64)),
            st.just(t.empty(0)),
            st.one_of(st.none(), st.just(t.empty(0))),
        ),
        SCALARS,
        st.tuples(SCALARS, SCALARS, SCALARS),
    ),
    max_size=5,
)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(value: object) -> bool:
    return _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]


def _positive_int(value: object) -> bool:
    return _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_freq_min(value: object) -> bool:
    return value is None or (
        _finite_number(value) and 0 <= value < 1  # ty: ignore[unsupported-operator]
    )


def _valid_freq_max(value: object) -> bool:
    return value is None or (
        _finite_number(value) and 0 < value <= 1  # ty: ignore[unsupported-operator]
    )


def _valid_mask_tuple(value: object) -> bool:
    return (
        isinstance(value, tuple) and bool(value) and all(_non_negative_int(item) for item in value)
    )


def _valid_config_ids(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_non_negative_int(item) for item in value)
        and len(value) == len(set(value))
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _valid_tensor_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, t.Tensor) for item in value)


def _valid_optional_tensor_list(value: object) -> bool:
    return isinstance(value, list) and all(
        item is None or isinstance(item, t.Tensor) for item in value
    )


def _valid_counterfactual_refs(value: object) -> bool:
    return isinstance(value, list) and all(
        item is None
        or (
            isinstance(item, tuple)
            and len(item) == 3
            and isinstance(item[0], t.Tensor)
            and isinstance(item[1], t.Tensor)
            and (item[2] is None or isinstance(item[2], t.Tensor))
        )
        for item in value
    )


def _empty_delta_call_kwargs(n_sae_features: int = 32) -> dict[str, Any]:
    return {
        "neg_delta_fids": t.empty(0, dtype=t.int64),
        "neg_delta_weights": t.empty(0),
        "raw_feature_ids": t.empty(0, dtype=t.int64),
        "ref_mask": t.ones(n_sae_features, dtype=t.bool),
        "selected_positive_ids": t.empty(0, dtype=t.int64),
        "n_sae_features": n_sae_features,
        "rng": t.Generator().manual_seed(0),
    }


def test_feature_policy_objects_expose_nominal_refined_fields() -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (3, 1))
    assert_type(mask.ids, NonEmptyNonNegativeIntTuple)

    result = BlacklistResult((), 8, 12, 4, 0.0, 1.0, "test")
    assert_type(result.ids, NonNegativeIntTuple)
    assert_type(result.n_sae_features, PositiveInt)
    assert_type(result.n_segments, NonNegativeInt)
    assert_type(result.n_active_features, NonNegativeInt)

    descriptor = FeatureClusterDescriptor(8, "a" * 64)
    assert_type(descriptor.n_sae_features, PositiveInt)
    assert_type(descriptor.content_sha256, LowercaseSha256)


@PROPERTY_SETTINGS
@given(value=TUPLES)
def test_feature_mask_id_domain_matches_independent_oracle(
    value: tuple[object, ...],
) -> None:
    call = lambda: FeatureMask(  # noqa: E731
        FeatureMaskMode.blacklist,
        cast(Any, value),
    )
    if _valid_mask_tuple(value):
        valid_ids = cast(tuple[int, ...], value)
        assert call().ids == tuple(sorted(set(valid_ids)))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_feature_mask_width_domain_matches_independent_oracle(value: object) -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (0,))
    call = lambda: mask.allowed_universe_mask(cast(Any, value))  # noqa: E731
    if _positive_int(value):
        assert call().shape == (value,)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=LISTS)
def test_feature_mask_config_collection_translation_matches_oracle(
    value: list[object],
) -> None:
    call = lambda: resolve_feature_mask(  # noqa: E731
        {"feature_mask": {"mode": "blacklist", "ids": value}},
        101,
    )
    if _valid_config_ids(value):
        result = call()
        assert result is not None
        assert result.ids == tuple(sorted(cast(list[int], value)))
    else:
        with pytest.raises(FeatureMaskError, match="non-empty unique list"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_frequency_min_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: FrequencyBounds(min=cast(Any, value), max=None)  # noqa: E731
    if _valid_freq_min(value):
        result = call()
        assert result.min is None or type(result.min) is float
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="min"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_frequency_max_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: FrequencyBounds(min=None, max=cast(Any, value))  # noqa: E731
    if _valid_freq_max(value):
        result = call()
        assert result.max is None or type(result.max) is float
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="max"):
            call()


@PROPERTY_SETTINGS
@given(
    lower=st.floats(min_value=0, max_value=0.999, allow_nan=False),
    upper=st.floats(min_value=0.001, max_value=1, allow_nan=False),
)
def test_frequency_bound_order_remains_explicit_relationship(
    lower: float,
    upper: float,
) -> None:
    if lower < upper:
        assert FrequencyBounds(lower, upper).min == lower
    else:
        with pytest.raises(FeatureMaskError, match="strictly less"):
            FrequencyBounds(lower, upper)


@PROPERTY_SETTINGS
@given(value=TUPLES)
def test_blacklist_result_id_domain_matches_independent_oracle(
    value: tuple[object, ...],
) -> None:
    call = lambda: BlacklistResult(  # noqa: E731
        ids=cast(Any, value),
        n_sae_features=101,
        n_segments=0,
        n_active_features=0,
        freq_min=None,
        freq_max=None,
        stats_generated_at="test",
    )
    valid = all(_non_negative_int(item) for item in value)
    if valid:
        valid_ids = cast(tuple[int, ...], value)
        assert call().ids == tuple(sorted(set(valid_ids)))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="ids"):
            call()


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("n_sae_features", "n_segments", "n_active_features")),
    value=SCALARS,
)
def test_blacklist_result_integer_domains_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "ids": (),
        "n_sae_features": 1,
        "n_segments": 0,
        "n_active_features": 0,
        "freq_min": None,
        "freq_max": None,
        "stats_generated_at": "test",
    }
    kwargs[field] = value
    valid = _positive_int(value) if field == "n_sae_features" else _non_negative_int(value)
    if valid:
        assert getattr(BlacklistResult(**kwargs), field) == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            BlacklistResult(**kwargs)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_frequency_outcome_count_domain_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: FrequencyBlacklistOutcome(  # noqa: E731
        FrequencyBlacklistStatus.DISABLED,
        n_blacklisted=cast(Any, value),
    )
    if _non_negative_int(value):
        assert call().n_blacklisted == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_blacklisted"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_cluster_definition_width_domain_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: FeatureClusterDefinition.from_mapping(  # noqa: E731
        {"schema_version": 1, "clusters": [[0, 1]]},
        n_sae_features=cast(Any, value),
    )
    if _positive_int(value):
        if value >= 2:  # ty: ignore[unsupported-operator]
            assert call().clusters == ((0, 1),)
        else:
            with pytest.raises(FeatureClustersError, match="out of range"):
                call()
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_cluster_descriptor_width_domain_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: FeatureClusterDescriptor(  # noqa: E731
        n_sae_features=cast(Any, value),
        content_sha256="a" * 64,
    )
    if _positive_int(value):
        assert call().n_sae_features == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_cluster_descriptor_digest_domain_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: FeatureClusterDescriptor(  # noqa: E731
        n_sae_features=1,
        content_sha256=cast(Any, value),
        semantics=FEATURE_CLUSTERS_SEMANTICS,
    )
    if _valid_sha256(value):
        assert call().content_sha256 == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="content_sha256"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_cluster_resolver_width_domain_precedes_disabled_config(
    value: object,
) -> None:
    call = lambda: resolve_feature_clusters(None, cast(Any, value))  # noqa: E731
    if _positive_int(value):
        assert call() is None
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_document_delta_width_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: compute_doc_deltas(  # noqa: E731
        [],
        [],
        [],
        cast(Any, value),
        DeltaMode.prev_segment,
    )
    if _positive_int(value):
        assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_counterfactual_delta_width_domain_matches_independent_oracle(
    value: object,
) -> None:
    call = lambda: compute_doc_counterfactual_deltas(  # noqa: E731
        [],
        [],
        [],
        [],
        cast(Any, value),
    )
    if _positive_int(value):
        assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae"):
            call()


@PROPERTY_SETTINGS
@given(value=st.lists(TENSORISH, max_size=5))
def test_delta_tensor_list_domain_checks_every_item(value: list[object]) -> None:
    call = lambda: compute_doc_deltas(  # noqa: E731
        [],
        cast(Any, value),
        [],
        8,
        DeltaMode.prev_segment,
    )
    if _valid_tensor_list(value):
        assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="seg_mags"):
            call()


@PROPERTY_SETTINGS
@given(value=st.lists(OPTIONAL_TENSORISH, max_size=5))
def test_delta_optional_tensor_list_checks_every_item(value: list[object]) -> None:
    call = lambda: compute_doc_deltas(  # noqa: E731
        [],
        [],
        cast(Any, value),
        8,
        DeltaMode.prev_segment,
    )
    if _valid_optional_tensor_list(value):
        assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="seg_pcts"):
            call()


@PROPERTY_SETTINGS
@given(value=COUNTERFACTUAL_REFS)
def test_counterfactual_reference_list_checks_every_item(
    value: list[object],
) -> None:
    call = lambda: compute_doc_counterfactual_deltas(  # noqa: E731
        [],
        [],
        [],
        cast(Any, value),
        8,
    )
    if _valid_counterfactual_refs(value):
        if value:
            with pytest.raises(ValueError, match="seg_refs length"):
                call()
        else:
            assert call() == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="seg_refs"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_delta_negative_count_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_delta_negatives(  # noqa: E731
        cast(Any, value),
        0.5,
        **_empty_delta_call_kwargs(),
    )
    if _non_negative_int(value):
        assert call().numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="k"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_delta_negative_ratio_domain_matches_independent_oracle(value: object) -> None:
    call = lambda: sample_delta_negatives(  # noqa: E731
        0,
        cast(Any, value),
        **_empty_delta_call_kwargs(),
    )
    valid = _finite_number(value) and 0 <= value <= 1  # ty: ignore[unsupported-operator]
    if valid:
        assert call().numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="neg_delta_ratio"):
            call()


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_delta_negative_width_domain_matches_independent_oracle(value: object) -> None:
    kwargs = _empty_delta_call_kwargs()
    kwargs["n_sae_features"] = value
    call = lambda: sample_delta_negatives(0, 0.5, **kwargs)  # noqa: E731
    if _positive_int(value):
        assert call().numel() == 0
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
            call()


@PROPERTY_SETTINGS
@given(
    k=st.integers(min_value=0, max_value=20),
    ratio=st.floats(min_value=0, max_value=1, allow_nan=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_delta_sampling_is_deterministic_and_respects_exclusion_contract(
    k: int,
    ratio: float,
    seed: int,
) -> None:
    n_sae = 32
    common: dict[str, Any] = {
        "k": k,
        "neg_delta_ratio": ratio,
        "neg_delta_fids": t.tensor([1, 3, 5, 7], dtype=t.int64),
        "neg_delta_weights": t.tensor([0.1, 0.2, 0.3, 0.4]),
        "raw_feature_ids": t.tensor([0], dtype=t.int64),
        "ref_mask": t.tensor(
            [index in {0, 2} for index in range(n_sae)],
            dtype=t.bool,
        ),
        "selected_positive_ids": t.tensor([4], dtype=t.int64),
        "n_sae_features": n_sae,
    }
    first_rng = t.Generator().manual_seed(seed)
    second_rng = t.Generator().manual_seed(seed)
    first = sample_delta_negatives(**common, rng=first_rng)
    second = sample_delta_negatives(**common, rng=second_rng)
    assert t.equal(first, second)
    assert t.equal(first_rng.get_state(), second_rng.get_state())
    assert first.numel() <= k
    assert len(set(first.tolist())) == first.numel()
    assert not (set(first.tolist()) & {0, 2, 4})
    assert bool(((first >= 0) & (first < n_sae)).all())


@PROPERTY_SETTINGS
@given(seed=st.integers(min_value=-(2**31), max_value=2**31 - 1))
def test_feature_policy_type_checks_do_not_perturb_python_global_rng(
    seed: int,
) -> None:
    random.seed(seed)
    before = random.getstate()
    FeatureMask(FeatureMaskMode.blacklist, (1, 3))
    FrequencyBounds(0.0, 1.0)
    BlacklistResult((), 8, 0, 0, None, None, "test")
    FrequencyBlacklistOutcome(FrequencyBlacklistStatus.DISABLED)
    FeatureClusterDescriptor(8, "a" * 64)
    compute_doc_deltas([], [], [], 8, DeltaMode.prev_segment)
    compute_doc_counterfactual_deltas([], [], [], [], 8)
    sample_delta_negatives(0, 0.5, **_empty_delta_call_kwargs(8))
    splice_counterfactual_manifest_identity(clustered=True)
    assert random.getstate() == before
