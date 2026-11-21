"""Independent property oracles for target-conditioning refined domains."""

from __future__ import annotations

import random
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.feature_config import FeatureConfig
from steering.eval.pooling_op import PoolingOp
from steering.eval.roundtrip_steering import (
    RoundtripEvalConfig,
    parse_roundtrip_eval_config,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    MultiTargetSegments,
    SingleTargetSegment,
    validate_feature_dict,
)
from steering.eval.target_segments import filter_segments_to_target, pool_segments_to_dict
from steering.refined_types import (
    NonNegativeIntMappingItems,
    NonNegativeIntTuple,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-25, max_value=25),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=12),
)
TUPLES = st.lists(SCALARS, max_size=8).map(tuple)
MAPPINGS = st.dictionaries(SCALARS, SCALARS, max_size=6)
MAPPING_LISTS = st.lists(MAPPINGS, max_size=5)


def _strict_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_tuple(value: object, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, tuple)
        and (bool(value) or not nonempty)
        and all(_strict_non_negative_int(item) for item in value)
    )


def _valid_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        _strict_non_negative_int(key) and _strict_non_negative_int(item)
        for key, item in value.items()
    )


def _valid_feature_config_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        _strict_non_negative_int(feature_id)
        and isinstance(score, int)
        and not isinstance(score, bool)
        for feature_id, score in value.items()
    )


def _valid_mapping_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_valid_mapping(item) for item in value)
    )


@PROPERTY_SETTINGS
@given(value=TUPLES)
def test_non_negative_tuple_phantom_matches_independent_oracle(
    value: tuple[object, ...],
) -> None:
    assert isinstance(value, NonNegativeIntTuple) is _valid_tuple(value)


@PROPERTY_SETTINGS
@given(value=MAPPINGS)
def test_mapping_items_phantom_matches_independent_oracle(
    value: dict[object, object],
) -> None:
    items = tuple(value.items())
    assert isinstance(items, NonNegativeIntMappingItems) is _valid_mapping(value)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("feature_ids", "buckets")), value=TUPLES)
def test_context_segment_tuple_domains_match_independent_oracle(
    field: str,
    value: tuple[object, ...],
) -> None:
    kwargs: dict[str, Any] = {
        "feature_ids": (),
        "buckets": (),
        "char_start": 0,
        "char_end": 0,
    }
    kwargs[field] = value
    if _valid_tuple(value):
        ContextFeatureSegment(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            ContextFeatureSegment(**kwargs)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("char_start", "char_end")), value=SCALARS)
def test_context_segment_offset_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "feature_ids": (),
        "buckets": (),
        "char_start": 0,
        "char_end": 0,
    }
    kwargs[field] = value
    if _strict_non_negative_int(value):
        ContextFeatureSegment(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            ContextFeatureSegment(**kwargs)


@PROPERTY_SETTINGS
@given(value=MAPPINGS)
def test_single_target_and_feature_config_mapping_domains(
    value: dict[object, object],
) -> None:
    if _valid_mapping(value):
        expected = cast(dict[int, int], value)
        assert SingleTargetSegment(features=expected).features == expected
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="features"):
            SingleTargetSegment(features=cast(Any, value))

    if _valid_feature_config_mapping(value):
        expected = cast(dict[int, int], value)
        assert FeatureConfig(name="property", features=expected).features == expected
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="features"):
            FeatureConfig(name="property", features=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=MAPPING_LISTS)
def test_multi_target_mapping_collection_matches_independent_oracle(
    value: list[dict[object, object]],
) -> None:
    assignments = t.tensor([0], dtype=t.int64)
    if _valid_mapping_list(value):
        target = MultiTargetSegments(
            segments=cast(Any, value),
            token_assignments=assignments,
        )
        assert target.segments == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="segments"):
            MultiTargetSegments(
                segments=cast(Any, value),
                token_assignments=assignments,
            )


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("n_sae_features", "num_buckets")), value=SCALARS)
def test_feature_dict_extent_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "features": {},
        "n_sae_features": 1,
        "num_buckets": 1,
    }
    kwargs[field] = value
    if _strict_non_negative_int(value) and value > 0:  # ty: ignore[unsupported-operator]
        validate_feature_dict(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            validate_feature_dict(**kwargs)


@PROPERTY_SETTINGS
@given(fid=st.integers(min_value=0, max_value=30), bucket=st.integers(0, 30))
def test_feature_dict_upper_bounds_remain_explicit_relationships(
    fid: int,
    bucket: int,
) -> None:
    if fid >= 16:
        with pytest.raises(ValueError, match="feature id"):
            validate_feature_dict({fid: bucket}, 16, 8)
    elif bucket >= 8:
        with pytest.raises(ValueError, match="bucket"):
            validate_feature_dict({fid: bucket}, 16, 8)
    else:
        validate_feature_dict({fid: bucket}, 16, 8)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("target_char_start", "target_char_end")), value=SCALARS)
def test_target_filter_offset_domains_precede_range_relationships(
    field: str,
    value: object,
) -> None:
    segment = ContextFeatureSegment((), (), 0, 10)
    kwargs: dict[str, Any] = {
        "segments": (segment,),
        "target_char_start": 0,
        "target_char_end": 10,
    }
    kwargs[field] = value
    if _strict_non_negative_int(value):
        try:
            filter_segments_to_target(**kwargs)
        except ValueError:
            pass
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            filter_segments_to_target(**kwargs)


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_roundtrip_chunk_domain_matches_independent_oracle(value: object) -> None:
    valid = _strict_non_negative_int(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        RoundtripEvalConfig(feature_chunk_size=cast(Any, value))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="feature_chunk_size"):
            RoundtripEvalConfig(feature_chunk_size=cast(Any, value))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_roundtrip_parser_wraps_refined_chunk_domain(value: object) -> None:
    cfg = {"roundtrip_steering": {"feature_chunk_size": value}}
    valid = _strict_non_negative_int(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        assert parse_roundtrip_eval_config(cfg).feature_chunk_size == value
    else:
        with pytest.raises(ValueError, match="feature_chunk_size"):
            parse_roundtrip_eval_config(cfg)


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    fid=st.integers(min_value=0, max_value=100),
    bucket=st.integers(min_value=0, max_value=100),
)
def test_refined_collection_checks_do_not_perturb_global_rng(
    seed: int,
    fid: int,
    bucket: int,
) -> None:
    random.seed(seed)
    before = random.getstate()
    segment = ContextFeatureSegment((fid,), (bucket,), 0, 1)
    SingleTargetSegment({fid: bucket})
    MultiTargetSegments([{fid: bucket}], t.tensor([0], dtype=t.int64))
    FeatureConfig(name="property", features={fid: bucket})
    RoundtripEvalConfig(feature_chunk_size=1)
    pool_segments_to_dict((segment,), op=PoolingOp.BUCKET_MAX)
    assert random.getstate() == before
