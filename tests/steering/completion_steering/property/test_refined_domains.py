"""Independent property oracles for completion-steering refined domains."""

from __future__ import annotations

import math
import random
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from scripts.eval.completion_steering_eval import _generate_completions
from steering.conditional.feature_index import SegmentFeatures
from steering.eval.completion_steering_eval import (
    EvalMode,
    RawFeatureSelection,
    ScanResultTensors,
    select_from_tensors,
)
from steering.eval.multiseg_generator import (
    AddedBucketCondition,
    CompletionCondition,
    CompletionSample,
    DocumentCompletion,
    GroundTruthFeatureSignals,
    added_bucket_condition,
    build_multiseg_inputs,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-25, max_value=25),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=12),
)
LISTS = st.lists(SCALARS, max_size=8)
MAPPINGS = st.dictionaries(SCALARS, SCALARS, max_size=6)
SETS = st.sets(SCALARS, max_size=8)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(value: object) -> bool:
    return _strict_int(value) and value >= 0  # ty: ignore[unsupported-operator]


def _positive_int(value: object) -> bool:
    return _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]


def _non_negative_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _non_negative_int_list(value: object) -> bool:
    return isinstance(value, list) and all(_non_negative_int(item) for item in value)


def _non_negative_int_mapping(value: object) -> bool:
    return isinstance(value, dict) and all(
        _non_negative_int(key) and _non_negative_int(item)
        for key, item in value.items()
    )


def _non_negative_int_set(value: object) -> bool:
    return isinstance(value, set) and all(_non_negative_int(item) for item in value)


def _empty_scan() -> ScanResultTensors:
    return ScanResultTensors(
        feature_ids=t.empty(0, dtype=t.int64),
        n_docs_activated=t.empty(0, dtype=t.int64),
        offsets=t.tensor([0], dtype=t.int64),
        doc_idx=t.empty(0, dtype=t.int64),
        max_activation=t.empty(0),
        max_segment_idx=t.empty(0, dtype=t.int64),
        percentile=t.empty(0),
        total_segments=t.empty(0, dtype=t.int64),
    )


def _segment() -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=t.tensor([1], dtype=t.int32),
        feature_mags=t.tensor([2], dtype=t.int64),
        char_start=0,
        char_end=100,
    )


class _Tokenizer:
    def decode(self, token_ids: list[int]) -> str:
        return "x" * len(token_ids)


@PROPERTY_SETTINGS
@given(value=LISTS)
def test_selection_document_list_matches_independent_oracle(value: list[object]) -> None:
    if _non_negative_int_list(value):
        selection = RawFeatureSelection(0, 0, cast(Any, value))
        assert selection.selected_docs == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="selected_docs"):
            RawFeatureSelection(0, 0, cast(Any, value))


@PROPERTY_SETTINGS
@given(value=MAPPINGS)
def test_selection_target_mapping_matches_independent_oracle(
    value: dict[object, object],
) -> None:
    if _non_negative_int_mapping(value):
        selection = RawFeatureSelection(0, 0, [], cast(Any, value))
        assert selection.target_segments == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="target_segments"):
            RawFeatureSelection(0, 0, [], cast(Any, value))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("feature_id", "n_docs_activated")), value=SCALARS)
def test_selection_scalar_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "feature_id": 0,
        "n_docs_activated": 0,
        "selected_docs": [],
    }
    kwargs[field] = value
    if _non_negative_int(value):
        RawFeatureSelection(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            RawFeatureSelection(**kwargs)


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("top_k", "docs_per_feature", "seed")), value=SCALARS)
def test_selection_function_scalar_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "scan_tensors": _empty_scan(),
        "top_k": 0,
        "docs_per_feature": 0,
        "mode": EvalMode.ABLATION,
        "seed": 0,
    }
    kwargs[field] = value
    valid = _strict_int(value) if field == "seed" else _non_negative_int(value)
    if valid:
        assert select_from_tensors(**kwargs) == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            select_from_tensors(**kwargs)


@PROPERTY_SETTINGS
@given(value=SETS)
def test_selection_document_set_matches_independent_oracle(value: set[object]) -> None:
    kwargs = {
        "scan_tensors": _empty_scan(),
        "top_k": 0,
        "docs_per_feature": 0,
        "mode": EvalMode.INSERTION,
        "all_doc_indices": cast(Any, value),
    }
    if _non_negative_int_set(value):
        assert select_from_tensors(**cast(Any, kwargs)) == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="all_doc_indices"):
            select_from_tensors(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_added_bucket_domain_matches_independent_oracle(value: object) -> None:
    if _non_negative_int(value):
        condition = added_bucket_condition(cast(Any, value))
        assert condition == f"added_bucket_{value}"
        assert condition.bucket_index == value
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="bucket_index"):
            added_bucket_condition(cast(Any, value))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("seed", "bucket_index")), value=SCALARS)
def test_completion_sample_scalar_domains_precede_relationships(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "condition": CompletionCondition.REAL,
        "text": "property",
        "seed": 0,
        "bucket_index": None,
    }
    kwargs[field] = value
    valid = (
        _strict_int(value)
        if field == "seed"
        else value is None or _non_negative_int(value)
    )
    if valid:
        try:
            CompletionSample(**kwargs)
        except ValueError:
            pass
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            CompletionSample(**kwargs)


@PROPERTY_SETTINGS
@given(value=MAPPINGS)
def test_ground_truth_mapping_matches_independent_oracle(
    value: dict[object, object],
) -> None:
    kwargs = {
        "feature_signals": cast(Any, value),
        "n_features_active": 0,
        "avg_magnitude": 0.0,
        "max_magnitude": 0,
        "min_magnitude": 0,
        "feature_of_interest_magnitude": None,
    }
    if _non_negative_int_mapping(value):
        GroundTruthFeatureSignals(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="feature_signals"):
            GroundTruthFeatureSignals(**kwargs)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "n_features_active",
            "avg_magnitude",
            "max_magnitude",
            "min_magnitude",
            "feature_of_interest_magnitude",
        )
    ),
    value=SCALARS,
)
def test_ground_truth_scalar_domains_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "feature_signals": {},
        "n_features_active": 0,
        "avg_magnitude": 0.0,
        "max_magnitude": 0,
        "min_magnitude": 0,
        "feature_of_interest_magnitude": None,
    }
    kwargs[field] = value
    if field == "avg_magnitude":
        valid = _non_negative_finite_number(value)
    elif field == "feature_of_interest_magnitude":
        valid = value is None or _non_negative_int(value)
    else:
        valid = _non_negative_int(value)
    if valid:
        GroundTruthFeatureSignals(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            GroundTruthFeatureSignals(**kwargs)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("doc_idx", "feature_id", "target_segment_idx")),
    value=SCALARS,
)
def test_document_completion_indices_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "doc_idx": 0,
        "feature_id": 0,
        "pre_segment_text": "before",
        "ground_truth_segment": "target",
        "target_segment_idx": 0,
    }
    kwargs[field] = value
    if _non_negative_int(value):
        DocumentCompletion(**kwargs)
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            DocumentCompletion(**kwargs)


@PROPERTY_SETTINGS
@given(value=LISTS)
def test_multiseg_prefix_domain_matches_independent_oracle(value: list[object]) -> None:
    kwargs = {
        "segments": [_segment()],
        "prefix_tokens": cast(Any, value),
        "n_gen_tokens": 0,
        "target_segment_idx": 0,
        "tokenizer": _Tokenizer(),
    }
    if _non_negative_int_list(value):
        build_multiseg_inputs(**cast(Any, kwargs))
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match="prefix_tokens"):
            build_multiseg_inputs(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("n_gen_tokens", "target_segment_idx")), value=SCALARS)
def test_multiseg_scalar_domains_precede_index_relationships(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "segments": [_segment()],
        "prefix_tokens": [],
        "n_gen_tokens": 0,
        "target_segment_idx": 0,
        "tokenizer": _Tokenizer(),
    }
    kwargs[field] = value
    if _non_negative_int(value):
        try:
            build_multiseg_inputs(**cast(Any, kwargs))
        except ValueError:
            pass
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            build_multiseg_inputs(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "num_buckets",
            "samples_per_condition",
            "temperature",
            "top_k_sampling",
            "bos_id",
        )
    ),
    value=SCALARS,
)
def test_pipeline_generation_scalars_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {
        "selections": [],
        "target_segment_maps": None,
        "doc_texts": [],
        "feature_index": object(),
        "val_shard_stem": "property",
        "model": object(),
        "tokenizer": object(),
        "mode": EvalMode.ABLATION,
        "boundaries": t.empty(0),
        "feature_cfg": None,
        "precomputed_thresholds": None,
        "device": t.device("cpu"),
        "num_buckets": 1,
        "samples_per_condition": 1,
        "temperature": 0.0,
        "top_k_sampling": 1,
        "bos_id": 0,
    }
    kwargs[field] = value
    if field in {"num_buckets", "samples_per_condition"}:
        valid = _positive_int(value)
    elif field == "top_k_sampling":
        valid = value is None or _positive_int(value)
    elif field == "temperature":
        valid = _non_negative_finite_number(value)
    else:
        valid = _non_negative_int(value)
    if valid:
        assert _generate_completions(**kwargs) == []
    else:
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _generate_completions(**kwargs)


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=-(2**31), max_value=2**31 - 1),
    fid=st.integers(min_value=0, max_value=100),
    bucket=st.integers(min_value=0, max_value=100),
)
def test_completion_collection_checks_do_not_perturb_global_rng(
    seed: int,
    fid: int,
    bucket: int,
) -> None:
    random.seed(seed)
    before = random.getstate()
    selection = RawFeatureSelection(fid, 1, [0], {0: 0})
    sample = CompletionSample(AddedBucketCondition(bucket), "property", seed, bucket)
    signals = GroundTruthFeatureSignals({fid: bucket}, 1, bucket, bucket, bucket, bucket)
    DocumentCompletion(0, fid, "before", "target", 0, [sample], signals)
    select_from_tensors(_empty_scan(), 0, 0, EvalMode.ABLATION, {0}, seed)
    build_multiseg_inputs([_segment()], [0], 0, 0, _Tokenizer())
    _generate_completions(
        [],
        None,
        [],
        object(),
        "property",
        object(),
        object(),
        EvalMode.ABLATION,
        t.empty(0),
        None,
        None,
        t.device("cpu"),
        1,
        1,
        0.0,
        1,
        0,
    )
    assert selection.feature_id == fid
    assert random.getstate() == before
