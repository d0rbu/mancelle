"""Property contracts for AxBench-Adv refined runtime domains."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.fitness import evaluate_fitness, validate_fitness_config
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample
from steering.eval.axbench_adv.pipeline import AxBenchAdvRunConfig
from steering.eval.axbench_adv.prompt_surgery import (
    QWEN_CONTROL_TOKENS,
    QWEN_TERMINAL_MARKER,
    canonicalize_suffix,
    insert_suffix,
    split_terminal_marker,
)
from steering.eval.axbench_adv.seeding import canonical_factor, row_seed
from steering.eval.axbench_adv.self_conditioning import _BoundSelfConditioningCell
from steering.eval.axbench_adv.source_contract import SourceExpectations, SourcePaths
from steering.eval.axbench_adv.worker_client import (
    AxBenchAdvJudgeClient,
    AxBenchAdvWorkerClient,
    WorkerExample,
)
from steering.eval.axbench_base.config import LLMClientConfig

PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True, deadline=None)
SCALARS = st.one_of(
    st.integers(min_value=-10_000, max_value=10_000),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
    st.booleans(),
    st.none(),
    st.text(max_size=24),
)
HASH_VALUES = st.one_of(
    SCALARS,
    st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
)
PROMPT = f"prompt{QWEN_TERMINAL_MARKER}"
_MAX_SIGNED_63_BIT = (1 << 63) - 1


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_utf8(value: str) -> bool:
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


def _expectations(**overrides: Any) -> SourceExpectations:
    values: dict[str, Any] = {
        "max_concepts": 2,
        "repository_git_commit": "1" * 40,
        "dataset_revision": "2" * 40,
        "target_model": "model",
        "seed": 42,
        "temperature": 1.0,
        "output_length": 128,
        "package_lock_sha256": "3" * 64,
        "vendored_axbench_upstream_commit": "4" * 40,
        "project_repository_git_commit": "5" * 40,
        "project_package_lock_sha256": "6" * 64,
        "config_name_prefix": "qwen2_5_3b",
        "layer": 28,
    }
    values.update(overrides)
    return SourceExpectations(**values)


def _run_config(**overrides: Any) -> AxBenchAdvRunConfig:
    values: dict[str, Any] = {
        "source_paths": SourcePaths(
            Path("/source/prepared"),
            Path("/source/main.parquet"),
            Path("/source/aux.parquet"),
            Path("/source/evaluated.jsonl"),
            Path("/source/project.parquet"),
            Path("/source/meta_000001.json"),
            Path("/source/statistics.safetensors"),
        ),
        "source_expectations": _expectations(),
        "source_hashes": {f"source-{index}": "a" * 64 for index in range(100)},
        "output_dir": Path("/tmp/axbench-adv-property-output"),
        "judge_cache": Path("/tmp/axbench-adv-property-judge.sqlite"),
        "reflection_cache": Path("/tmp/axbench-adv-property-reflection.sqlite"),
        "reflection": LLMClientConfig(model="test/reflection"),
        "factor_overrides": {},
    }
    values.update(overrides)
    return AxBenchAdvRunConfig(**values)


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("max_concepts", "output_length", "layer")),
    value=SCALARS,
)
def test_source_positive_integer_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        assert getattr(_expectations(**{field: value}), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            _expectations(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("seed", "temperature", "target_model", "config_name_prefix")),
    value=SCALARS,
)
def test_source_mixed_refined_fields_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    valid = {
        "seed": _strict_int(value) and value >= 0,  # ty: ignore[unsupported-operator]
        "temperature": _finite_number(value) and value >= 0,  # ty: ignore[unsupported-operator]
        "target_model": isinstance(value, str) and bool(value.strip()),
        "config_name_prefix": isinstance(value, str) and bool(value.strip()),
    }[field]
    if valid:
        assert getattr(_expectations(**{field: value}), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            _expectations(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("package_lock_sha256", "project_package_lock_sha256")),
    value=HASH_VALUES,
)
def test_source_hash_fields_match_exact_lowercase_sha256_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )
    if valid:
        assert getattr(_expectations(**{field: value}), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            _expectations(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("master_seed", "gepa_seed", "reflection_max_retries")),
    value=SCALARS,
)
def test_run_config_non_negative_integer_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = (
        _strict_int(value)
        and value >= 0  # ty: ignore[unsupported-operator]
        and (field == "reflection_max_retries" or value <= _MAX_SIGNED_63_BIT)  # ty: ignore[unsupported-operator]
    )
    if valid:
        assert getattr(_run_config(**{field: value}), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            _run_config(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(("reflection_minibatch_size", "worker_timeout_seconds")),
    value=SCALARS,
)
def test_run_config_positive_fields_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    valid = {
        "reflection_minibatch_size": (
            _strict_int(value)
            and value > 0  # ty: ignore[unsupported-operator]
            and value <= 10  # ty: ignore[unsupported-operator]
        ),
        "worker_timeout_seconds": _finite_number(value) and value > 0,  # ty: ignore[unsupported-operator]
    }[field]
    if valid:
        assert getattr(_run_config(**{field: value}), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            _run_config(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("device", "api_base_url", "seed_suffix")), value=SCALARS)
def test_run_config_nonblank_fields_match_independent_oracle(
    field: str,
    value: object,
) -> None:
    valid = isinstance(value, str) and bool(value.strip())
    if valid:
        assert getattr(_run_config(**{field: value}), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            _run_config(**{field: cast(Any, value)})


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_run_config_boolean_field_excludes_integer_lookalikes(value: object) -> None:
    if isinstance(value, bool):
        assert _run_config(cache_only=value).cache_only is value
    else:
        with pytest.raises(AxBenchAdvError, match="cache_only"):
            _run_config(cache_only=cast(Any, value))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("input_id", "user_message")), value=SCALARS)
def test_optimization_example_domains_match_independent_oracles(
    field: str,
    value: object,
) -> None:
    valid = {
        "input_id": _strict_int(value) and value >= 0,  # ty: ignore[unsupported-operator]
        "user_message": isinstance(value, str) and bool(value),
    }[field]
    kwargs = {"input_id": 0, "user_message": "message", field: value}
    if valid:
        assert getattr(OptimizationExample(**kwargs), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            OptimizationExample(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("input_id", "input", "steered_input")), value=SCALARS)
def test_worker_example_domains_match_independent_oracles(field: str, value: object) -> None:
    valid = {
        "input_id": _strict_int(value) and value >= 0,  # ty: ignore[unsupported-operator]
        "input": isinstance(value, str),
        "steered_input": isinstance(value, str),
    }[field]
    kwargs = {"input_id": 0, "input": PROMPT, "steered_input": PROMPT, field: value}
    if valid:
        assert getattr(WorkerExample(**kwargs), field) == value
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            WorkerExample(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("min_output_chars", "fluency_floor")), value=SCALARS)
def test_fitness_thresholds_match_independent_oracles(field: str, value: object) -> None:
    valid = {
        "min_output_chars": _strict_int(value) and value > 0,  # ty: ignore[unsupported-operator]
        "fluency_floor": (
            _finite_number(value)
            and value >= 0  # ty: ignore[unsupported-operator]
            and value <= 2  # ty: ignore[unsupported-operator]
        ),
    }[field]
    kwargs = {"min_output_chars": 8, "fluency_floor": 1.0, field: value}
    if valid:
        validate_fitness_config(**cast(Any, kwargs))
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            validate_fitness_config(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("concept_rating", "fluency_rating", "output")), value=SCALARS)
def test_fitness_inputs_match_independent_oracles(field: str, value: object) -> None:
    valid = {
        "concept_rating": (
            _finite_number(value)
            and value >= 0  # ty: ignore[unsupported-operator]
            and value <= 2  # ty: ignore[unsupported-operator]
        ),
        "fluency_rating": (
            _finite_number(value)
            and value >= 0  # ty: ignore[unsupported-operator]
            and value <= 2  # ty: ignore[unsupported-operator]
        ),
        "output": isinstance(value, str),
    }[field]
    kwargs = {
        "concept_rating": 1.0,
        "fluency_rating": 1.0,
        "output": "substantive output",
        field: value,
    }
    if valid:
        evaluate_fitness(**cast(Any, kwargs))
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            evaluate_fitness(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_suffix_domain_and_semantics_match_independent_oracle(value: object) -> None:
    valid = (
        isinstance(value, str)
        and "\x00" not in value
        and all(token not in value for token in QWEN_CONTROL_TOKENS)
        and _valid_utf8(value)
    )
    if valid:
        assert canonicalize_suffix(value) == value.strip()
    else:
        with pytest.raises(AxBenchAdvError, match="suffix"):
            canonicalize_suffix(cast(Any, value))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_prompt_token_limit_matches_positive_integer_oracle(value: object) -> None:
    valid = _strict_int(value) and value > 0  # ty: ignore[unsupported-operator]
    if valid:
        assert (
            insert_suffix(
                PROMPT,
                "",
                token_count=lambda _text: 0,
                max_prompt_tokens=cast(Any, value),
            )
            == PROMPT
        )
    else:
        with pytest.raises(AxBenchAdvError, match="max_prompt_tokens"):
            insert_suffix(
                PROMPT,
                "",
                token_count=lambda _text: 0,
                max_prompt_tokens=cast(Any, value),
            )


@PROPERTY_SETTINGS
@given(value=st.one_of(SCALARS, st.just("")))
def test_terminal_prompt_rejects_every_nonempty_string_domain_violation(value: object) -> None:
    if isinstance(value, str) and value:
        return
    with pytest.raises(AxBenchAdvError, match="prompt"):
        split_terminal_marker(cast(Any, value))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("master_seed", "concept_id", "input_id")), value=SCALARS)
def test_row_seed_integer_fields_match_independent_oracle(field: str, value: object) -> None:
    valid = (
        _strict_int(value)
        and value >= 0  # ty: ignore[unsupported-operator]
        and value <= _MAX_SIGNED_63_BIT  # ty: ignore[unsupported-operator]
    )
    kwargs = {
        "master_seed": 0,
        "method": "DiffMean",
        "concept_id": 0,
        "input_id": 0,
        "factor": 1.0,
        "unattacked_prompt": "prompt",
        field: value,
    }
    if valid:
        assert isinstance(row_seed(**cast(Any, kwargs)), int)
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            row_seed(**cast(Any, kwargs))


@PROPERTY_SETTINGS
@given(value=SCALARS)
def test_canonical_factor_matches_finite_number_oracle(value: object) -> None:
    if _finite_number(value):
        numeric = float(cast(Any, value))
        expected = (0.0 if numeric == 0.0 else numeric).hex()
        assert canonical_factor(cast(Any, value)) == expected
    else:
        with pytest.raises(AxBenchAdvError, match="factor"):
            canonical_factor(cast(Any, value))


class _PropertyProcess:
    def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert operation == "evaluate"
        return {
            "concept_rating": 0.25,
            "fluency_rating": 1.75,
            "generation": f"generation-{payload['input_id']}",
            "input_id": payload["input_id"],
            "judge_rationale": "rationale",
            "paid_judge_cache_misses": 0,
        }


def _property_worker(input_ids: list[int]) -> AxBenchAdvWorkerClient:
    client = object.__new__(AxBenchAdvWorkerClient)
    client._closed = False
    client._method = "LoRA"
    client._concept_id = 0
    client._factor = 1.0
    client._master_seed = 0
    client._examples = {input_id: WorkerExample(input_id, PROMPT, PROMPT) for input_id in input_ids}
    client._process = cast(Any, _PropertyProcess())
    return client


@PROPERTY_SETTINGS
@given(values=st.lists(SCALARS, max_size=12), shape=st.sampled_from((list, tuple)))
def test_input_id_sequence_boundary_revalidates_every_member(
    values: list[object],
    shape: type[list] | type[tuple],
) -> None:
    sequence = shape(values)
    valid = (
        bool(values)
        and all(_strict_int(value) and value >= 0 for value in values)  # ty: ignore[unsupported-operator]
        and len(values) == len(set(cast(Any, values)))
    )
    if valid:
        typed_values = cast(list[int], values)
        state = random.getstate()
        result = _property_worker(typed_values).evaluate("", cast(Any, sequence))
        assert [entry.generation for entry in result] == [
            f"generation-{input_id}" for input_id in typed_values
        ]
        assert random.getstate() == state
    else:
        with pytest.raises(AxBenchAdvError, match="input_ids"):
            _property_worker([]).evaluate("", cast(Any, sequence))

    cell = object.__new__(_BoundSelfConditioningCell)
    cell._closed = True
    if valid:
        with pytest.raises(AxBenchAdvError, match="not active"):
            cell.evaluate("", cast(Any, sequence))
    else:
        with pytest.raises(AxBenchAdvError, match="input_ids"):
            cell.evaluate("", cast(Any, sequence))


@PROPERTY_SETTINGS
@given(field=st.sampled_from(("generation", "input_id")), value=SCALARS)
def test_judge_consumer_domains_match_independent_oracles(field: str, value: object) -> None:
    valid = {
        "generation": isinstance(value, str),
        "input_id": _strict_int(value) and value >= 0,  # ty: ignore[unsupported-operator]
    }[field]
    client = object.__new__(AxBenchAdvJudgeClient)
    client._closed = True
    kwargs = {"generation": "output", "input_id": 0, field: value}
    if valid:
        with pytest.raises(AxBenchAdvError, match="closed"):
            client.evaluate(**cast(Any, kwargs))
    else:
        with pytest.raises(AxBenchAdvError, match=field):
            client.evaluate(**cast(Any, kwargs))


def test_refined_boundaries_do_not_consume_global_random_state() -> None:
    state = random.getstate()
    _expectations()
    _run_config()
    OptimizationExample(0, "message")
    WorkerExample(0, PROMPT, PROMPT)
    validate_fitness_config(8, 1.0)
    evaluate_fitness(concept_rating=1.0, fluency_rating=2.0, output="substantive")
    canonicalize_suffix("suffix")
    split_terminal_marker(PROMPT)
    insert_suffix(PROMPT, "", token_count=lambda _text: 0, max_prompt_tokens=32)
    canonical_factor(1.0)
    row_seed(
        master_seed=0,
        method="DiffMean",
        concept_id=0,
        input_id=0,
        factor=1.0,
        unattacked_prompt="prompt",
    )
    assert random.getstate() == state
