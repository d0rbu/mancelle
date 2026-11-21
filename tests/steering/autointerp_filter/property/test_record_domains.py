"""Property coverage for autointerp record and calibration domains."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, assert_type, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.autointerp_filter.config import IntruderJudgeConfig
from steering.eval.autointerp_filter.prompt_calibration import (
    PromptCalibrationConfig,
    PromptCalibrationReport,
    build_prompt_calibration_config,
)
from steering.eval.autointerp_filter.token_examples import _DocActivations
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    AutoInterpCacheError,
    AutoInterpConfigError,
    AutoInterpDataError,
    AutoInterpFilterReport,
    IntruderTrial,
    IntruderTrialResult,
    RenderedExample,
    ScoringTarget,
    SourceDocRef,
    TargetKind,
    TargetOutcome,
    TargetScore,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.refined_types import (
    NonEmptyNonNegativeIntTuple,
    NonEmptyStr,
    NonNegativeFiniteNumber,
    NonNegativeFiniteNumberTuple,
    NonNegativeInt,
    NonNegativeIntTuple,
    PositiveInt,
    SingletonNonNegativeIntTuple,
    StrTuple,
    UnitInterval,
)

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)

_VALID_NON_NEGATIVE_INT = st.integers(min_value=0, max_value=10**9)
_VALID_POSITIVE_INT = st.integers(min_value=1, max_value=10**9)
_VALID_NON_EMPTY_STR = st.text(min_size=1, max_size=32)
_VALID_STR = st.text(max_size=32)
_VALID_NON_NEGATIVE_NUMBER = st.one_of(
    st.integers(min_value=0, max_value=10**100),
    st.floats(
        min_value=0.0,
        max_value=1e100,
        allow_nan=False,
        allow_infinity=False,
    ),
)
_VALID_NON_NEGATIVE_FLOAT = st.floats(
    min_value=0.0,
    max_value=1e100,
    allow_nan=False,
    allow_infinity=False,
)
_VALID_UNIT_INTERVAL = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)
_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.none(),
    st.text(),
)
_INVALID_POSITIVE_INT = st.one_of(
    st.booleans(),
    st.integers(max_value=0),
    st.floats(allow_nan=True, allow_infinity=True),
    st.none(),
    st.text(),
)
_INVALID_NON_EMPTY_STR = st.one_of(st.just(""), st.integers(), st.none())
_INVALID_NON_NEGATIVE_NUMBER = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.just(10**1000),
    st.floats(max_value=-1e-300, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
    st.just(float("inf")),
    st.none(),
    st.text(),
)
_INVALID_UNIT_INTERVAL = st.one_of(
    st.booleans(),
    st.floats(max_value=-1e-10, allow_nan=False, allow_infinity=True),
    st.floats(min_value=1.0000001, allow_nan=False, allow_infinity=True),
    st.just(float("nan")),
    st.none(),
    st.text(),
)
_INVALID_OPTIONAL_NON_NEGATIVE_INT = _INVALID_NON_NEGATIVE_INT.filter(
    lambda value: value is not None
)
_INVALID_OPTIONAL_POSITIVE_INT = _INVALID_POSITIVE_INT.filter(lambda value: value is not None)
_INVALID_OPTIONAL_UNIT_INTERVAL = _INVALID_UNIT_INTERVAL.filter(lambda value: value is not None)


def _target(**updates: Any) -> ScoringTarget:
    values: dict[str, Any] = {
        "kind": TargetKind.feature,
        "feature_ids": (3,),
        "target_key": "feature:3",
        "display_name": "feature_3",
    }
    values.update(updates)
    return ScoringTarget(**values)


def _window(**updates: Any) -> TokenWindowRef:
    values: dict[str, Any] = {
        "shard_stem": "shard_00000_features",
        "shard_index": 0,
        "doc_id": 1,
        "source_text_hash": "hash",
        "token_start": 4,
        "token_end": 6,
        "token_ids": (11, 12),
        "token_strings": ("a", "b"),
        "target_feature_ids": (3,),
        "activations": (0.0, 2.0),
        "max_activation": 2.0,
        "quantile": 0,
        "active": True,
    }
    values.update(updates)
    return TokenWindowRef(**values)


def _rendered(**updates: Any) -> RenderedExample:
    values: dict[str, Any] = {
        "text": "a<<b>>",
        "ref": _window(),
        "activating": True,
        "highlighted_token_indices": (1,),
    }
    values.update(updates)
    return RenderedExample(**values)


def _trial(**updates: Any) -> IntruderTrial:
    ref = _window()
    values: dict[str, Any] = {
        "target": _target(),
        "trial_index": 0,
        "positive_refs": (ref,),
        "intruder_ref": ref,
        "intruder_index": 1,
        "quantile": 0,
        "intruder_highlight_indices": (0,),
        "activation_threshold_ratio": 0.3,
        "trial_key": "trial-key",
    }
    values.update(updates)
    return IntruderTrial(**values)


def _result(**updates: Any) -> IntruderTrialResult:
    values: dict[str, Any] = {
        "trial_key": "trial-key",
        "prediction": 1,
        "correct_index": 1,
        "correct": True,
        "response": "[RESPONSE]: 1",
        "cached": False,
        "rendered_examples": (_rendered(),),
        "final_user_prompt": "prompt",
    }
    values.update(updates)
    return IntruderTrialResult(**values)


def _score(**updates: Any) -> TargetScore:
    values: dict[str, Any] = {
        "target_key": "feature:3",
        "display_name": "feature_3",
        "kind": TargetKind.feature,
        "feature_ids": (3,),
        "score": 0.5,
        "n_correct": 1,
        "n_trials": 2,
        "outcome": TargetOutcome.accepted,
        "descriptions": {3: "description"},
    }
    values.update(updates)
    return TargetScore(**values)


def _report(**updates: Any) -> AutoInterpFilterReport:
    values: dict[str, Any] = {
        "n_targets_total": 4,
        "n_targets_scored": 3,
        "n_targets_accepted": 2,
        "n_features_accepted": 2,
        "score_threshold": 0.5,
        "top_k": 2,
        "target_scores_path": "target_scores.jsonl",
        "ranked_yaml_path": "ranked.yaml",
        "accepted_features_path": "accepted.txt",
        "manual_feature_filter_path": "manual.yaml",
        "manual_feature_filter_meta_path": "manual.meta.json",
        "trial_audit_path": "audit.jsonl",
        "trial_audit_yaml_path": "audit.yaml",
    }
    values.update(updates)
    return AutoInterpFilterReport(**values)


@PROPERTY_SETTINGS
@given(
    shard_stem=_VALID_NON_EMPTY_STR,
    shard_index=_VALID_NON_NEGATIVE_INT,
    doc_id=_VALID_NON_NEGATIVE_INT,
    start=_VALID_NON_NEGATIVE_INT,
    span=_VALID_POSITIVE_INT,
)
def test_source_doc_ref_accepts_and_stores_every_scalar_domain(
    shard_stem: str,
    shard_index: int,
    doc_id: int,
    start: int,
    span: int,
) -> None:
    end = start + span
    ref = SourceDocRef(shard_stem, shard_index, doc_id, start, end)
    assert isinstance(ref.shard_stem, NonEmptyStr)
    assert all(
        isinstance(value, NonNegativeInt)
        for value in (
            ref.shard_index,
            ref.doc_id,
            ref.segment_token_start,
            ref.segment_token_end,
        )
    )
    assert ref.stable_key() == (f"{shard_stem}:shard{shard_index}:doc{doc_id}:tok{start}-{end}")


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "shard_stem",
            "shard_index",
            "doc_id",
            "segment_token_start",
            "segment_token_end",
        ]
    ),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
    bad_str=_INVALID_NON_EMPTY_STR,
)
def test_source_doc_ref_rejects_every_invalid_scalar_domain(
    field_name: str,
    bad_int: Any,
    bad_str: Any,
) -> None:
    values: dict[str, Any] = {
        "shard_stem": "shard",
        "shard_index": 0,
        "doc_id": 0,
        "segment_token_start": 0,
        "segment_token_end": 1,
    }
    values[field_name] = bad_str if field_name == "shard_stem" else bad_int
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        SourceDocRef(**values)


@PROPERTY_SETTINGS
@given(start=_VALID_NON_NEGATIVE_INT, offset=_VALID_NON_NEGATIVE_INT)
def test_source_doc_ref_retains_segment_ordering_relationship(
    start: int,
    offset: int,
) -> None:
    with pytest.raises(AutoInterpDataError, match="segment_token_end must exceed"):
        SourceDocRef("shard", 0, 0, start, start - min(start, offset))


@PROPERTY_SETTINGS
@given(
    feature_id=_VALID_NON_NEGATIVE_INT,
    target_key=_VALID_NON_EMPTY_STR,
    display_name=_VALID_NON_EMPTY_STR,
)
def test_scoring_target_accepts_and_stores_every_domain(
    feature_id: int,
    target_key: str,
    display_name: str,
) -> None:
    target = ScoringTarget(
        TargetKind.feature,
        (feature_id,),
        target_key,
        display_name,
    )
    assert isinstance(target.feature_ids, SingletonNonNegativeIntTuple)
    assert isinstance(target.target_key, NonEmptyStr)
    assert isinstance(target.display_name, NonEmptyStr)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(["feature_ids", "target_key", "display_name"]),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
    bad_str=_INVALID_NON_EMPTY_STR,
)
def test_scoring_target_rejects_invalid_domains(
    field_name: str,
    bad_int: Any,
    bad_str: Any,
) -> None:
    bad_value = (bad_int,) if field_name == "feature_ids" else bad_str
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _target(**{field_name: bad_value})


@PROPERTY_SETTINGS
@given(first=_VALID_NON_NEGATIVE_INT, second=_VALID_NON_NEGATIVE_INT)
def test_scoring_target_rejects_empty_and_multi_feature_shapes(
    first: int,
    second: int,
) -> None:
    for feature_ids in ((), (first, second)):
        with pytest.raises(BeartypeCallHintParamViolation, match="feature_ids"):
            _target(feature_ids=feature_ids)


@PROPERTY_SETTINGS
@given(
    token_ids=st.lists(_VALID_NON_NEGATIVE_INT, max_size=8).map(tuple),
    token_strings=st.data(),
    activations=st.data(),
    start=_VALID_NON_NEGATIVE_INT,
    target_feature_ids=st.lists(
        _VALID_NON_NEGATIVE_INT,
        min_size=1,
        max_size=5,
    ).map(tuple),
    max_activation=_VALID_NON_NEGATIVE_FLOAT,
    quantile=st.one_of(st.none(), _VALID_NON_NEGATIVE_INT),
    active=st.booleans(),
)
def test_token_window_accepts_nominal_domains_and_roundtrips_raw_json(
    token_ids: tuple[int, ...],
    token_strings: st.DataObject,
    activations: st.DataObject,
    start: int,
    target_feature_ids: tuple[int, ...],
    max_activation: int | float,
    quantile: int | None,
    active: bool,
) -> None:
    strings = tuple(
        token_strings.draw(st.lists(_VALID_STR, min_size=len(token_ids), max_size=len(token_ids)))
    )
    activation_values = tuple(
        activations.draw(
            st.lists(
                _VALID_NON_NEGATIVE_FLOAT,
                min_size=len(token_ids),
                max_size=len(token_ids),
            )
        )
    )
    ref = _window(
        token_start=start,
        token_end=start + len(token_ids),
        token_ids=token_ids,
        token_strings=strings,
        target_feature_ids=target_feature_ids,
        activations=activation_values,
        max_activation=max_activation,
        quantile=quantile,
        active=active,
    )
    assert isinstance(ref.token_ids, NonNegativeIntTuple)
    assert isinstance(ref.token_strings, StrTuple)
    assert isinstance(ref.target_feature_ids, NonEmptyNonNegativeIntTuple)
    assert isinstance(ref.activations, NonNegativeFiniteNumberTuple)
    assert isinstance(ref.max_activation, NonNegativeFiniteNumber)
    assert ref.quantile is None or isinstance(ref.quantile, NonNegativeInt)
    decoded = TokenWindowRef.from_json(ref.to_json())
    assert decoded == ref
    assert decoded.stable_key() == ref.stable_key()


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "shard_stem",
            "shard_index",
            "doc_id",
            "source_text_hash",
            "token_start",
            "token_end",
            "quantile",
        ]
    ),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
    bad_str=_INVALID_NON_EMPTY_STR,
)
def test_token_window_rejects_invalid_scalar_domains(
    field_name: str,
    bad_int: Any,
    bad_str: Any,
) -> None:
    bad_value = bad_str if field_name in {"shard_stem", "source_text_hash"} else bad_int
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _window(**{field_name: bad_value})


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(["token_ids", "token_strings", "target_feature_ids", "activations"]),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
    bad_number=_INVALID_NON_NEGATIVE_NUMBER,
)
def test_token_window_rejects_invalid_collection_domains(
    field_name: str,
    bad_int: Any,
    bad_number: Any,
) -> None:
    if field_name == "token_strings":
        bad_value: Any = (1, "b")
    elif field_name == "activations":
        bad_value = (0.0, bad_number)
    else:
        bad_value = (bad_int, 1)
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _window(**{field_name: bad_value})


@PROPERTY_SETTINGS
@given(
    relation=st.sampled_from(["strings", "activations", "reverse", "span"]),
    extra=_VALID_STR,
)
def test_token_window_retains_only_cross_field_relationship_checks(
    relation: str,
    extra: str,
) -> None:
    updates: dict[str, Any]
    if relation == "strings":
        updates = {"token_strings": ("a", "b", extra)}
    elif relation == "activations":
        updates = {"activations": (0.0, 1.0, 2.0)}
    elif relation == "reverse":
        updates = {"token_start": 7, "token_end": 6}
    else:
        updates = {"token_end": 7}
    with pytest.raises(AutoInterpDataError):
        _window(**updates)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "shard_stem",
            "shard_index",
            "doc_id",
            "source_text_hash",
            "token_start",
            "token_ids",
            "target_feature_ids",
            "activations",
            "max_activation",
            "quantile",
        ]
    )
)
def test_token_window_raw_decoder_translates_invalid_domains(
    field_name: str,
) -> None:
    payload = _window().to_json()
    invalid: dict[str, Any] = {
        "shard_stem": "",
        "shard_index": -1,
        "doc_id": -1,
        "source_text_hash": "",
        "token_start": -1,
        "token_ids": [-1, 12],
        "target_feature_ids": [],
        "activations": [0.0, float("nan")],
        "max_activation": float("inf"),
        "quantile": -1,
    }
    payload[field_name] = invalid[field_name]
    with pytest.raises(AutoInterpCacheError, match="invalid data"):
        TokenWindowRef.from_json(payload)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "shard_index",
            "doc_id",
            "token_start",
            "token_end",
            "token_ids",
            "target_feature_ids",
            "activations",
            "max_activation",
            "quantile",
        ]
    ),
    boolean=st.booleans(),
)
def test_token_window_raw_decoder_does_not_coerce_booleans_into_numbers(
    field_name: str,
    boolean: bool,
) -> None:
    payload = _window().to_json()
    payload[field_name] = (
        [boolean]
        if field_name
        in {
            "token_ids",
            "target_feature_ids",
            "activations",
        }
        else boolean
    )
    with pytest.raises(AutoInterpCacheError, match="invalid data"):
        TokenWindowRef.from_json(payload)


def test_token_window_raw_decoder_requires_an_exact_boolean_active_flag() -> None:
    payload = _window().to_json()
    payload["active"] = 1

    with pytest.raises(AutoInterpCacheError, match="invalid data"):
        TokenWindowRef.from_json(payload)


@PROPERTY_SETTINGS
@given(n_positive=_VALID_NON_NEGATIVE_INT, n_non_activating=_VALID_NON_NEGATIVE_INT)
def test_token_example_set_accepts_exact_record_tuples_and_roundtrips(
    n_positive: int,
    n_non_activating: int,
) -> None:
    positives = tuple(_window(doc_id=index) for index in range(n_positive % 5))
    non_activating = tuple(
        _window(doc_id=100 + index, active=False) for index in range(n_non_activating % 5)
    )
    examples = TokenExampleSet(positives, non_activating)
    assert TokenExampleSet.from_json(examples.to_json()) == examples


@PROPERTY_SETTINGS
@given(field_name=st.sampled_from(["positives", "non_activating"]))
def test_token_example_set_rejects_mutable_or_wrong_record_collections(
    field_name: str,
) -> None:
    for bad_value in ([_window()], (object(),)):
        with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
            TokenExampleSet(
                **cast(
                    Any,
                    {
                        field_name: bad_value,
                        "positives" if field_name == "non_activating" else "non_activating": (),
                    },
                )
            )


@PROPERTY_SETTINGS
@given(text=_VALID_NON_EMPTY_STR, index=_VALID_NON_NEGATIVE_INT, active=st.booleans())
def test_rendered_example_accepts_and_stores_domains(
    text: str,
    index: int,
    active: bool,
) -> None:
    rendered = _rendered(
        text=text,
        activating=active,
        highlighted_token_indices=(index,),
    )
    assert isinstance(rendered.text, NonEmptyStr)
    assert isinstance(rendered.highlighted_token_indices, NonNegativeIntTuple)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(["text", "highlighted_token_indices", "activating"]),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
)
def test_rendered_example_rejects_invalid_domains(
    field_name: str,
    bad_int: Any,
) -> None:
    bad_value: Any
    if field_name == "text":
        bad_value = ""
    elif field_name == "highlighted_token_indices":
        bad_value = (bad_int,)
    else:
        bad_value = 1
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _rendered(**{field_name: bad_value})


@PROPERTY_SETTINGS
@given(
    trial_index=_VALID_NON_NEGATIVE_INT,
    intruder_index=_VALID_NON_NEGATIVE_INT,
    quantile=st.one_of(st.none(), _VALID_NON_NEGATIVE_INT),
    threshold=_VALID_UNIT_INTERVAL,
    trial_key=_VALID_NON_EMPTY_STR,
)
def test_intruder_trial_accepts_and_stores_domains(
    trial_index: int,
    intruder_index: int,
    quantile: int | None,
    threshold: float,
    trial_key: str,
) -> None:
    trial = _trial(
        trial_index=trial_index,
        intruder_index=intruder_index,
        quantile=quantile,
        activation_threshold_ratio=threshold,
        trial_key=trial_key,
    )
    assert isinstance(trial.trial_index, NonNegativeInt)
    assert isinstance(trial.intruder_index, NonNegativeInt)
    assert isinstance(trial.activation_threshold_ratio, UnitInterval)
    assert isinstance(trial.trial_key, NonEmptyStr)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "trial_index",
            "intruder_index",
            "quantile",
            "intruder_highlight_indices",
            "activation_threshold_ratio",
            "trial_key",
        ]
    ),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
    bad_threshold=_INVALID_UNIT_INTERVAL,
)
def test_intruder_trial_rejects_invalid_domains(
    field_name: str,
    bad_int: Any,
    bad_threshold: Any,
) -> None:
    if field_name == "trial_key":
        bad_value: Any = ""
    elif field_name == "activation_threshold_ratio":
        bad_value = bad_threshold
    elif field_name == "intruder_highlight_indices":
        bad_value = (bad_int,)
    else:
        bad_value = bad_int
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _trial(**{field_name: bad_value})


@PROPERTY_SETTINGS
@given(
    prediction=st.one_of(st.none(), _VALID_NON_NEGATIVE_INT),
    correct_index=_VALID_NON_NEGATIVE_INT,
    correct=st.booleans(),
    cached=st.booleans(),
)
def test_intruder_result_accepts_and_stores_domains(
    prediction: int | None,
    correct_index: int,
    correct: bool,
    cached: bool,
) -> None:
    result = _result(
        prediction=prediction,
        correct_index=correct_index,
        correct=correct,
        cached=cached,
    )
    assert result.prediction is None or isinstance(result.prediction, NonNegativeInt)
    assert isinstance(result.correct_index, NonNegativeInt)
    assert result.correct is correct
    assert result.cached is cached


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(["trial_key", "prediction", "correct_index", "correct", "cached"]),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
)
def test_intruder_result_rejects_invalid_domains(field_name: str, bad_int: Any) -> None:
    bad_value: Any
    if field_name == "trial_key":
        bad_value = ""
    elif field_name in {"correct", "cached"}:
        bad_value = 1
    else:
        bad_value = bad_int
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _result(**{field_name: bad_value})


@PROPERTY_SETTINGS
@given(
    n_trials=_VALID_NON_NEGATIVE_INT,
    data=st.data(),
    score=st.one_of(st.none(), _VALID_UNIT_INTERVAL),
)
def test_target_score_accepts_nominal_domains_and_valid_count_relations(
    n_trials: int,
    data: st.DataObject,
    score: float | None,
) -> None:
    n_correct = data.draw(st.integers(min_value=0, max_value=n_trials))
    record = _score(n_trials=n_trials, n_correct=n_correct, score=score)
    assert isinstance(record.feature_ids, NonEmptyNonNegativeIntTuple)
    assert record.score is None or isinstance(record.score, UnitInterval)
    assert isinstance(record.n_correct, NonNegativeInt)
    assert isinstance(record.n_trials, NonNegativeInt)
    assert isinstance(record.schema_version, PositiveInt)
    assert asdict(record)["schema_version"] == SCHEMA_VERSION


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "target_key",
            "display_name",
            "feature_ids",
            "score",
            "n_correct",
            "n_trials",
            "schema_version",
        ]
    ),
    bad_int=_INVALID_NON_NEGATIVE_INT,
    bad_positive=_INVALID_POSITIVE_INT,
    bad_score=_INVALID_OPTIONAL_UNIT_INTERVAL,
)
def test_target_score_rejects_invalid_domains(
    field_name: str,
    bad_int: Any,
    bad_positive: Any,
    bad_score: Any,
) -> None:
    invalid: dict[str, Any] = {
        "target_key": "",
        "display_name": "",
        "feature_ids": (bad_int,),
        "score": bad_score,
        "n_correct": bad_int,
        "n_trials": bad_int,
        "schema_version": bad_positive,
    }
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _score(**{field_name: invalid[field_name]})


@PROPERTY_SETTINGS
@given(n_trials=_VALID_NON_NEGATIVE_INT, excess=_VALID_POSITIVE_INT)
def test_target_score_retains_count_relationship_check(
    n_trials: int,
    excess: int,
) -> None:
    with pytest.raises(AutoInterpDataError, match="n_correct.*exceeds"):
        _score(n_correct=n_trials + excess, n_trials=n_trials)


@PROPERTY_SETTINGS
@given(total=_VALID_NON_NEGATIVE_INT, data=st.data())
def test_filter_report_accepts_nominal_domains_and_ordered_counts(
    total: int,
    data: st.DataObject,
) -> None:
    scored = data.draw(st.integers(min_value=0, max_value=total))
    accepted = data.draw(st.integers(min_value=0, max_value=scored))
    report = _report(
        n_targets_total=total,
        n_targets_scored=scored,
        n_targets_accepted=accepted,
    )
    assert isinstance(report.n_targets_total, NonNegativeInt)
    assert isinstance(report.n_targets_scored, NonNegativeInt)
    assert isinstance(report.n_targets_accepted, NonNegativeInt)
    assert isinstance(report.n_features_accepted, NonNegativeInt)
    assert isinstance(report.score_threshold, UnitInterval)
    assert isinstance(report.top_k, PositiveInt)


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "n_targets_total",
            "n_targets_scored",
            "n_targets_accepted",
            "n_features_accepted",
            "score_threshold",
            "top_k",
            "target_scores_path",
            "schema_version",
        ]
    ),
    bad_int=_INVALID_NON_NEGATIVE_INT,
    bad_positive=_INVALID_OPTIONAL_POSITIVE_INT,
    bad_score=_INVALID_OPTIONAL_UNIT_INTERVAL,
)
def test_filter_report_rejects_invalid_domains(
    field_name: str,
    bad_int: Any,
    bad_positive: Any,
    bad_score: Any,
) -> None:
    invalid: dict[str, Any] = {
        "n_targets_total": bad_int,
        "n_targets_scored": bad_int,
        "n_targets_accepted": bad_int,
        "n_features_accepted": bad_int,
        "score_threshold": bad_score,
        "top_k": bad_positive,
        "target_scores_path": "",
        "schema_version": bad_positive,
    }
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _report(**{field_name: invalid[field_name]})


@PROPERTY_SETTINGS
@given(total=_VALID_NON_NEGATIVE_INT, excess=_VALID_POSITIVE_INT)
def test_filter_report_retains_count_relationship_checks(total: int, excess: int) -> None:
    with pytest.raises(AutoInterpDataError, match="scored exceeds"):
        _report(n_targets_total=total, n_targets_scored=total + excess)
    with pytest.raises(AutoInterpDataError, match="accepted exceeds"):
        _report(
            n_targets_total=total + excess,
            n_targets_scored=total,
            n_targets_accepted=total + excess,
        )


@PROPERTY_SETTINGS
@given(
    max_trials=st.one_of(st.none(), _VALID_POSITIVE_INT),
    progress=_VALID_NON_NEGATIVE_INT,
    checkpoint=st.one_of(st.none(), _VALID_NON_NEGATIVE_INT),
    fail=st.booleans(),
)
def test_prompt_calibration_config_accepts_and_stores_domains(
    max_trials: int | None,
    progress: int,
    checkpoint: int | None,
    fail: bool,
) -> None:
    cfg = PromptCalibrationConfig(
        "audit.yaml",
        "output.yaml",
        max_trials=max_trials,
        progress_interval=progress,
        checkpoint_interval=checkpoint,
        fail_on_parse_error=fail,
    )
    assert isinstance(cfg.trial_audit_path, NonEmptyStr)
    assert cfg.max_trials is None or isinstance(cfg.max_trials, PositiveInt)
    assert isinstance(cfg.progress_interval, NonNegativeInt)
    assert cfg.checkpoint_interval is None or isinstance(
        cfg.checkpoint_interval,
        NonNegativeInt,
    )


@PROPERTY_SETTINGS
@given(
    field_name=st.sampled_from(
        [
            "trial_audit_path",
            "output_path",
            "json_output_path",
            "max_trials",
            "fail_on_parse_error",
            "progress_interval",
            "checkpoint_interval",
        ]
    ),
    bad_int=_INVALID_OPTIONAL_NON_NEGATIVE_INT,
    bad_positive=_INVALID_OPTIONAL_POSITIVE_INT,
)
def test_prompt_calibration_config_rejects_invalid_domains(
    field_name: str,
    bad_int: Any,
    bad_positive: Any,
) -> None:
    invalid: dict[str, Any] = {
        "trial_audit_path": "",
        "output_path": "",
        "json_output_path": "",
        "max_trials": bad_positive,
        "fail_on_parse_error": 1,
        "progress_interval": bad_int,
        "checkpoint_interval": bad_int,
    }
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        values: dict[str, Any] = {
            "trial_audit_path": "audit.yaml",
            "output_path": "output.yaml",
        }
        values[field_name] = invalid[field_name]
        PromptCalibrationConfig(**values)


@PROPERTY_SETTINGS
@given(n_trials=_VALID_NON_NEGATIVE_INT, data=st.data())
def test_prompt_calibration_report_accepts_nominal_domains_and_relations(
    n_trials: int,
    data: st.DataObject,
) -> None:
    total = data.draw(st.integers(min_value=n_trials, max_value=n_trials + 1000))
    n_correct = data.draw(st.integers(min_value=0, max_value=n_trials))
    n_errors = data.draw(st.integers(min_value=0, max_value=n_trials))
    report = PromptCalibrationReport(
        "audit.yaml",
        "output.yaml",
        None,
        "model",
        "variant",
        "few-shot",
        "v1",
        False,
        n_trials,
        total,
        n_correct,
        n_errors,
        0.0 if n_trials == 0 else n_correct / n_trials,
        n_trials == total,
    )
    assert isinstance(report.n_trials, NonNegativeInt)
    assert isinstance(report.n_trials_total, NonNegativeInt)
    assert isinstance(report.accuracy, UnitInterval)


@PROPERTY_SETTINGS
@given(n_trials=_VALID_NON_NEGATIVE_INT, excess=_VALID_POSITIVE_INT)
def test_prompt_calibration_report_retains_count_relationship_checks(
    n_trials: int,
    excess: int,
) -> None:
    values: dict[str, Any] = {
        "source_trial_audit_path": "audit.yaml",
        "output_path": "output.yaml",
        "json_output_path": None,
        "model": "model",
        "prompt_variant": "variant",
        "few_shot_set": "few-shot",
        "prompt_version": "v1",
        "cot": False,
        "n_trials": n_trials,
        "n_trials_total": n_trials,
        "n_correct": 0,
        "n_parse_errors": 0,
        "accuracy": 0.0,
        "complete": True,
    }
    with pytest.raises(AutoInterpConfigError, match="n_trials exceeds"):
        PromptCalibrationReport(**cast(Any, {**values, "n_trials": n_trials + excess}))
    with pytest.raises(AutoInterpConfigError, match="counts exceed"):
        PromptCalibrationReport(**cast(Any, {**values, "n_correct": n_trials + excess}))


@PROPERTY_SETTINGS
@given(
    token_ids=st.lists(_VALID_NON_NEGATIVE_INT, min_size=1, max_size=8).map(tuple),
    feature_ids=st.lists(
        _VALID_NON_NEGATIVE_INT,
        min_size=1,
        max_size=8,
    ).map(tuple),
)
def test_doc_activations_accepts_and_stores_nominal_domains(
    token_ids: tuple[int, ...],
    feature_ids: tuple[int, ...],
) -> None:
    record = _DocActivations(
        source_text_hash="hash",
        token_offset=0,
        token_ids=token_ids,
        token_strings=tuple("token" for _ in token_ids),
        sae_feature_ids=feature_ids,
        sae_acts=t.zeros((len(token_ids), len(feature_ids))),
    )
    assert isinstance(record.source_text_hash, NonEmptyStr)
    assert isinstance(record.token_offset, NonNegativeInt)
    assert isinstance(record.token_ids, NonNegativeIntTuple)
    assert isinstance(record.token_strings, StrTuple)
    assert isinstance(record.sae_feature_ids, NonEmptyNonNegativeIntTuple)


@PROPERTY_SETTINGS
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_record_constructors_do_not_consume_global_rng(seed: int) -> None:
    random.seed(seed)
    expected = random.getstate()
    ref = _window()
    target = _target()
    rendered = _rendered(ref=ref)
    TokenExampleSet((ref,), ())
    IntruderTrial(target, 0, (ref,), ref, 1, 0, (0,), 0.3, "trial")
    IntruderTrialResult("trial", 1, 1, True, "response", False, rendered_examples=(rendered,))
    _score()
    _report()
    PromptCalibrationConfig("audit.yaml", "output.yaml")
    assert random.getstate() == expected


def test_record_fields_are_statically_nominal() -> None:
    ref = _window()
    target = _target()
    score = _score()
    report = _report()
    cfg = PromptCalibrationConfig("audit.yaml", "output.yaml")
    assert_type(ref.shard_stem, NonEmptyStr)
    assert_type(ref.token_ids, NonNegativeIntTuple)
    assert_type(ref.activations, NonNegativeFiniteNumberTuple)
    assert_type(target.feature_ids, SingletonNonNegativeIntTuple)
    assert_type(score.score, UnitInterval | None)
    assert_type(score.n_trials, NonNegativeInt)
    assert_type(report.top_k, PositiveInt | None)
    assert_type(cfg.max_trials, PositiveInt | None)


def test_raw_prompt_config_builder_translates_beartype_failures() -> None:
    with pytest.raises(AutoInterpConfigError, match="invalid fields"):
        build_prompt_calibration_config(
            {
                "trial_audit_path": "audit.yaml",
                "output_path": "output.yaml",
                "progress_interval": -1,
            }
        )
    with pytest.raises(AutoInterpConfigError, match="invalid fields"):
        build_prompt_calibration_config(
            {
                "trial_audit_path": "audit.yaml",
                "output_path": "output.yaml",
                "judge": {"max_concurrent": 0},
            }
        )


def test_default_prompt_config_judge_matches_explicit_default() -> None:
    implicit = PromptCalibrationConfig("audit.yaml", "output.yaml")
    explicit = PromptCalibrationConfig(
        "audit.yaml",
        "output.yaml",
        judge=IntruderJudgeConfig(),
    )
    assert implicit == explicit
