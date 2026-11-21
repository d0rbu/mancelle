"""Property contracts for typed BPB-Adv attack and population records."""

from __future__ import annotations

import random
from typing import Any, assert_type

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import (
    BpbAttackPrefix,
    BpbAttackSample,
    BpbAttackWorkSpec,
    bpb_attack_sample_from_synth_row,
    core_prefix_id,
    pair_samples_into_prefixes,
    prepare_bpb_attack_rows,
)
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    EvaluationPairKey,
    EvaluationPopulation,
    population_from_rows,
)
from steering.refined_types import NonEmptyStr, NonNegativeInt

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)

_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)
_INVALID_NON_EMPTY_STR = st.one_of(
    st.just(""),
    st.integers(),
    st.booleans(),
    st.none(),
)
_INVALID_TOKEN = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)
_TOKEN_LIST = st.lists(st.integers(min_value=0, max_value=100_000), max_size=12)
_NON_EMPTY_TOKEN_LIST = st.lists(
    st.integers(min_value=0, max_value=100_000),
    min_size=1,
    max_size=12,
)


def _work_spec() -> BpbAttackWorkSpec:
    return BpbAttackWorkSpec(
        method_name="diffmean_decoder_only",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
        coefficient_str="1.0",
        baseline_family="diffmean",
    )


def _sample(
    *,
    content_class: str = "concept_rich",
    prefix_tokens: list[int] | None = None,
    output_tokens: list[int] | None = None,
) -> BpbAttackSample:
    fields: dict[str, Any] = {
        "sample_id": "sample-id",
        "concept_id": 0,
        "mode": SteeringMode.POSITIVE,
        "content_class": content_class,
        "prefix_text": "",
        "prefix_hash": "prefix-hash",
        "prefix_tokens": [0] if prefix_tokens is None else prefix_tokens,
        "output_tokens": [] if output_tokens is None else output_tokens,
    }
    return BpbAttackSample(**fields)


def _prefix(
    *,
    prefix_tokens: list[int] | None = None,
    rich_ids: list[str] | None = None,
    avoid_ids: list[str] | None = None,
    rich_outputs: list[list[int]] | None = None,
    avoid_outputs: list[list[int]] | None = None,
) -> BpbAttackPrefix:
    return BpbAttackPrefix(
        core_prefix_id="core-prefix",
        concept_id=0,
        mode=SteeringMode.POSITIVE,
        prefix_text="",
        prefix_hash="prefix-hash",
        prefix_tokens=[0] if prefix_tokens is None else prefix_tokens,
        concept_rich_sample_ids=["rich-id"] if rich_ids is None else rich_ids,
        concept_avoiding_sample_ids=["avoid-id"] if avoid_ids is None else avoid_ids,
        concept_rich_output_tokens=[[]] if rich_outputs is None else rich_outputs,
        concept_avoiding_output_tokens=[[]] if avoid_outputs is None else avoid_outputs,
    )


def _cfg() -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[AdditiveDirectionConfig(magnitudes=(0.0,))],
    )


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, _text: str) -> list[int]:
        return [1]


@PROPERTY_SETTINGS
@given(
    concept_id=st.integers(min_value=0),
    mode=st.sampled_from(tuple(SteeringMode)),
    prefix_tokens=_NON_EMPTY_TOKEN_LIST,
    output_tokens=_TOKEN_LIST,
)
def test_attack_sample_normalizes_valid_scalar_domains_nominally(
    concept_id: int,
    mode: SteeringMode,
    prefix_tokens: list[int],
    output_tokens: list[int],
) -> None:
    sample = BpbAttackSample(
        sample_id="sample-id",
        concept_id=concept_id,
        mode=mode,
        content_class="concept_rich",
        prefix_text="",
        prefix_hash="prefix-hash",
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
    )

    assert_type(sample.sample_id, NonEmptyStr)
    assert_type(sample.concept_id, NonNegativeInt)
    assert sample.concept_id == concept_id
    assert type(sample.concept_id) is int
    assert sample.mode is mode
    assert sample.prefix_tokens is prefix_tokens
    assert sample.output_tokens is output_tokens


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("concept_id", "sample_id", "prefix_hash")),
    invalid_non_negative=_INVALID_NON_NEGATIVE_INT,
    invalid_string=_INVALID_NON_EMPTY_STR,
)
def test_attack_sample_rejects_every_invalid_nominal_scalar(
    field: str,
    invalid_non_negative: object,
    invalid_string: object,
) -> None:
    fields: dict[str, Any] = {
        "sample_id": "sample-id",
        "concept_id": 0,
        "mode": SteeringMode.POSITIVE,
        "content_class": "concept_rich",
        "prefix_text": "",
        "prefix_hash": "prefix-hash",
        "prefix_tokens": [0],
        "output_tokens": [],
    }
    fields[field] = invalid_non_negative if field == "concept_id" else invalid_string
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        BpbAttackSample(**fields)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("prefix_tokens", "output_tokens")),
    prefix=_TOKEN_LIST,
    invalid=_INVALID_TOKEN,
    suffix=_TOKEN_LIST,
)
def test_attack_sample_checks_every_token_collection_item(
    field: str,
    prefix: list[int],
    invalid: object,
    suffix: list[int],
) -> None:
    fields: dict[str, Any] = {
        "sample_id": "sample-id",
        "concept_id": 0,
        "mode": SteeringMode.POSITIVE,
        "content_class": "concept_rich",
        "prefix_text": "",
        "prefix_hash": "prefix-hash",
        "prefix_tokens": [0],
        "output_tokens": [],
    }
    fields[field] = [*prefix, invalid, *suffix]
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        BpbAttackSample(**fields)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("concept_id", "content_class")),
    invalid_non_negative=_INVALID_NON_NEGATIVE_INT,
    invalid_content_class=st.one_of(
        st.text().filter(lambda value: value not in {"concept_rich", "concept_avoiding"}),
        st.integers(),
        st.booleans(),
        st.none(),
    ),
)
def test_synth_row_ingress_translates_invalid_record_fields_contextually(
    field: str,
    invalid_non_negative: object,
    invalid_content_class: object,
) -> None:
    row: dict[str, Any] = {
        "concept_id": 0,
        "prefix_text": "prefix",
        "output_text": "output",
        "content_class": "concept_rich",
    }
    row[field] = (
        invalid_non_negative if field == "concept_id" else invalid_content_class
    )
    with pytest.raises(AxBenchBpbAdvError, match=field):
        bpb_attack_sample_from_synth_row(
            row,
            mode=SteeringMode.POSITIVE,
            tokenizer=_Tokenizer(),
        )


@PROPERTY_SETTINGS
@given(value=_INVALID_NON_NEGATIVE_INT)
def test_core_prefix_id_rejects_every_invalid_concept_id(value: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="concept_id"):
        core_prefix_id(
            concept_id=value,  # ty: ignore[invalid-argument-type]
            mode=SteeringMode.POSITIVE,
            prefix_text="prefix",
        )


@PROPERTY_SETTINGS
@given(
    rich_outputs=st.lists(_TOKEN_LIST, min_size=1, max_size=8),
    avoid_outputs=st.lists(_TOKEN_LIST, min_size=1, max_size=8),
)
def test_attack_prefix_accepts_every_valid_nested_token_shape(
    rich_outputs: list[list[int]],
    avoid_outputs: list[list[int]],
) -> None:
    rich_ids = [f"rich-{index}" for index in range(len(rich_outputs))]
    avoid_ids = [f"avoid-{index}" for index in range(len(avoid_outputs))]
    prefix = _prefix(
        rich_ids=rich_ids,
        avoid_ids=avoid_ids,
        rich_outputs=rich_outputs,
        avoid_outputs=avoid_outputs,
    )

    assert_type(prefix.concept_id, NonNegativeInt)
    assert_type(prefix.core_prefix_id, NonEmptyStr)
    assert prefix.concept_rich_output_tokens is rich_outputs
    assert prefix.concept_avoiding_output_tokens is avoid_outputs


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("concept_rich_output_tokens", "concept_avoiding_output_tokens")),
    before=st.lists(_TOKEN_LIST, max_size=6),
    token_prefix=_TOKEN_LIST,
    invalid=_INVALID_TOKEN,
    token_suffix=_TOKEN_LIST,
    after=st.lists(_TOKEN_LIST, max_size=6),
)
def test_attack_prefix_checks_every_nested_token_item(
    field: str,
    before: list[list[int]],
    token_prefix: list[int],
    invalid: object,
    token_suffix: list[int],
    after: list[list[int]],
) -> None:
    invalid_outputs: list[Any] = [
        *before,
        [*token_prefix, invalid, *token_suffix],
        *after,
    ]
    fields: dict[str, Any] = {
        "prefix_tokens": [0],
        "rich_ids": [f"rich-{index}" for index in range(len(invalid_outputs))],
        "avoid_ids": ["avoid-id"],
        "rich_outputs": invalid_outputs,
        "avoid_outputs": [[]],
    }
    if field == "concept_avoiding_output_tokens":
        fields["rich_ids"] = ["rich-id"]
        fields["rich_outputs"] = [[]]
        fields["avoid_ids"] = [f"avoid-{index}" for index in range(len(invalid_outputs))]
        fields["avoid_outputs"] = invalid_outputs
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        _prefix(**fields)


@PROPERTY_SETTINGS
@given(
    side=st.sampled_from(("rich", "avoiding")),
    n_ids=st.integers(min_value=1, max_value=8),
    n_outputs=st.integers(min_value=1, max_value=8),
)
def test_attack_prefix_keeps_aligned_length_relationship_explicit(
    side: str,
    n_ids: int,
    n_outputs: int,
) -> None:
    if n_ids == n_outputs:
        return
    kwargs: dict[str, Any] = {}
    if side == "rich":
        kwargs["rich_ids"] = [f"rich-{index}" for index in range(n_ids)]
        kwargs["rich_outputs"] = [[] for _ in range(n_outputs)]
    else:
        kwargs["avoid_ids"] = [f"avoid-{index}" for index in range(n_ids)]
        kwargs["avoid_outputs"] = [[] for _ in range(n_outputs)]
    with pytest.raises(AxBenchBpbAdvError, match="equal lengths"):
        _prefix(**kwargs)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "method_name",
            "target_model",
            "coefficient_str",
            "baseline_family",
        )
    ),
    value=_INVALID_NON_EMPTY_STR,
)
def test_attack_work_spec_rejects_every_invalid_name_field(
    field: str,
    value: object,
) -> None:
    fields: dict[str, Any] = {
        "method_name": "method",
        "mode": SteeringMode.POSITIVE,
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "baseline_family": "diffmean",
    }
    fields[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        BpbAttackWorkSpec(**fields)


@PROPERTY_SETTINGS
@given(
    collection=st.sampled_from(("samples", "prefixes", "work_specs")),
    before=st.integers(min_value=0, max_value=30),
    after=st.integers(min_value=0, max_value=30),
)
def test_attack_sequence_boundaries_check_every_member(
    collection: str,
    before: int,
    after: int,
) -> None:
    invalid = object()
    if collection == "samples":
        values: list[Any] = [_sample()] * before + [invalid] + [_sample()] * after
        with pytest.raises(BeartypeCallHintParamViolation, match="samples"):
            pair_samples_into_prefixes(values)
        return
    if collection == "prefixes":
        prefixes: list[Any] = [_prefix()] * before + [invalid] + [_prefix()] * after
        with pytest.raises(BeartypeCallHintParamViolation, match="prefixes"):
            prepare_bpb_attack_rows(
                prefixes=prefixes,
                work_specs=[_work_spec()],
                cfg=_cfg(),
            )
        return
    work_specs: list[Any] = [_work_spec()] * before + [invalid] + [_work_spec()] * after
    with pytest.raises(BeartypeCallHintParamViolation, match="work_specs"):
        prepare_bpb_attack_rows(
            prefixes=[_prefix()],
            work_specs=work_specs,
            cfg=_cfg(),
        )


def test_attack_record_boundaries_do_not_consume_global_random_state() -> None:
    rich_outputs = [[index, index + 1] for index in range(100)]
    avoid_outputs = [[index + 1000] for index in range(100)]
    rich = _sample(content_class="concept_rich")
    avoid = _sample(content_class="concept_avoiding")
    state = random.getstate()
    _prefix(
        rich_ids=[f"rich-{index}" for index in range(100)],
        avoid_ids=[f"avoid-{index}" for index in range(100)],
        rich_outputs=rich_outputs,
        avoid_outputs=avoid_outputs,
    )
    pair_samples_into_prefixes([rich, avoid])
    assert random.getstate() == state


@PROPERTY_SETTINGS
@given(
    concept_id=st.integers(min_value=0),
    mode=st.sampled_from(tuple(SteeringMode)),
    core_prefix_id=st.text(min_size=1),
    rich_digest=st.text(min_size=1),
    avoid_digest=st.text(min_size=1),
)
def test_evaluation_pair_key_normalizes_valid_domains_nominally(
    concept_id: int,
    mode: SteeringMode,
    core_prefix_id: str,
    rich_digest: str,
    avoid_digest: str,
) -> None:
    key = EvaluationPairKey(
        concept_id=concept_id,
        mode=mode,
        core_prefix_id=core_prefix_id,
        concept_rich_output_digest=rich_digest,
        concept_avoiding_output_digest=avoid_digest,
    )
    assert_type(key.concept_id, NonNegativeInt)
    assert_type(key.core_prefix_id, NonEmptyStr)
    assert key.concept_id == concept_id
    assert key.mode is mode


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "concept_id",
            "core_prefix_id",
            "concept_rich_output_digest",
            "concept_avoiding_output_digest",
        )
    ),
    invalid_non_negative=_INVALID_NON_NEGATIVE_INT,
    invalid_string=_INVALID_NON_EMPTY_STR,
)
def test_evaluation_pair_key_rejects_every_invalid_scalar(
    field: str,
    invalid_non_negative: object,
    invalid_string: object,
) -> None:
    fields: dict[str, Any] = {
        "concept_id": 0,
        "mode": SteeringMode.POSITIVE,
        "core_prefix_id": "prefix",
        "concept_rich_output_digest": "rich",
        "concept_avoiding_output_digest": "avoid",
    }
    fields[field] = invalid_non_negative if field == "concept_id" else invalid_string
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        EvaluationPairKey(**fields)


@PROPERTY_SETTINGS
@given(
    concept_id=st.integers(min_value=0),
    mode=st.sampled_from(tuple(SteeringMode)),
    core_prefix_id=st.text(min_size=1),
    rich_digest=st.text(min_size=1),
    avoid_digest=st.text(min_size=1),
)
def test_population_from_rows_builds_the_exact_typed_group(
    concept_id: int,
    mode: SteeringMode,
    core_prefix_id: str,
    rich_digest: str,
    avoid_digest: str,
) -> None:
    population = population_from_rows(
        [
            {
                "concept_id": concept_id,
                "mode": mode.value,
                "core_prefix_id": core_prefix_id,
                "concept_rich_output_digest": rich_digest,
                "concept_avoiding_output_digest": avoid_digest,
            }
        ]
    )
    (key,) = population.pair_keys
    assert key.concept_id == concept_id
    assert key.mode is mode
    assert population.prefix_ids_by_group == {
        (key.concept_id, key.mode): frozenset({key.core_prefix_id})
    }


def test_evaluation_population_keeps_cross_field_group_relation_explicit() -> None:
    key = EvaluationPairKey(
        concept_id=0,
        mode=SteeringMode.POSITIVE,
        core_prefix_id="prefix",
        concept_rich_output_digest="rich",
        concept_avoiding_output_digest="avoid",
    )
    with pytest.raises(AxBenchBpbAdvError, match="exactly match"):
        EvaluationPopulation(
            pair_keys=frozenset({key}),
            prefix_ids_by_group={
                (key.concept_id, key.mode): frozenset({NonEmptyStr.parse("other")})
            },
        )


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        (
            "concept_id",
            "mode",
            "core_prefix_id",
            "concept_rich_output_digest",
            "concept_avoiding_output_digest",
        )
    ),
    invalid_non_negative=_INVALID_NON_NEGATIVE_INT,
    invalid_string=_INVALID_NON_EMPTY_STR,
)
def test_population_row_ingress_translates_invalid_fields_contextually(
    field: str,
    invalid_non_negative: object,
    invalid_string: object,
) -> None:
    row: dict[str, Any] = {
        "concept_id": 0,
        "mode": SteeringMode.POSITIVE.value,
        "core_prefix_id": "prefix",
        "concept_rich_output_digest": "rich",
        "concept_avoiding_output_digest": "avoid",
    }
    if field == "concept_id":
        row[field] = invalid_non_negative
    elif field == "mode":
        row[field] = "not-a-mode"
    else:
        row[field] = invalid_string
    with pytest.raises(AxBenchBpbAdvError, match="invalid.*row key"):
        population_from_rows([row])
