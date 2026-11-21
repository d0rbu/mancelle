"""Build and judge sparse AxBench-Adv frames for the selected six or seven methods."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nanochat.atomic_io import atomic_write_text
from scripts.eval.axbench_upstream.run_evaluation import (
    DEFAULT_API_BASE_URL,
    EVALUATE_CONFIG,
    EVALUATE_SCRIPT,
    VENDOR_ROOT,
)
from steering.eval.axbench_adv.artifacts import logical_table_digest, write_validated_parquet
from steering.eval.axbench_adv.best_point import load_steering_entries
from steering.eval.axbench_adv.counter_resume import _RATING
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.execution_scope import validate_concept_ids
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.method_grid import ALL_METHOD_FACTORS
from steering.eval.axbench_adv.types import METHODS, NATIVE_METHOD, validate_method_subset
from steering.eval.axbench_adv.worker_client import REPO_ROOT, _FramedProcess

_BASE_COLUMNS = (
    "dataset_name",
    "concept_id",
    "input_concept",
    "input_id",
    "factor",
    "original_prompt",
)
_RESULT_FIELDS = {
    "factor",
    "fluency_completions",
    "fluency_ratings",
    "lm_judge_rating",
    "raw_aggregated_ratings",
    "raw_fluency_ratings",
    "raw_relevance_concept_ratings",
    "raw_relevance_instruction_ratings",
    "relevance_concept_completions",
    "relevance_concept_ratings",
    "relevance_instruction_completions",
    "relevance_instruction_ratings",
}
_EVALUATION_FIELDS = {
    "LMJudgeEvaluator": "raw_aggregated_ratings",
    "LMJudgeEvaluator_relevance_concept_ratings": "raw_relevance_concept_ratings",
    "LMJudgeEvaluator_relevance_concept_completions": "relevance_concept_completions",
    "LMJudgeEvaluator_relevance_instruction_ratings": "raw_relevance_instruction_ratings",
    "LMJudgeEvaluator_relevance_instruction_completions": "relevance_instruction_completions",
    "LMJudgeEvaluator_fluency_ratings": "raw_fluency_ratings",
    "LMJudgeEvaluator_fluency_completions": "fluency_completions",
}
_CALL_ACCOUNTING_FILE = "judge_call_accounting.json"


@dataclass(frozen=True)
class FinalJudgeResult:
    """Published final-judge path and exact vendored evaluator accounting."""

    steering_jsonl: Path
    actual_evaluator_calls: int
    paid_judge_cache_misses: int


def _generation_column(method: str) -> str:
    return f"{method}_steered_generation"


def _point_schema(methods: tuple[str, ...] = METHODS) -> pa.Schema:
    return pa.schema(
        [
            ("dataset_name", pa.string(), False),
            ("concept_id", pa.int64(), False),
            ("input_concept", pa.string(), False),
            ("input_id", pa.int64(), False),
            ("factor", pa.float64(), False),
            ("original_prompt", pa.string(), False),
            *((_generation_column(method), pa.string()) for method in methods),
        ]
    )


POINT_FRAME_SCHEMA = _point_schema()


def _factor_map(
    best_factors: Mapping[str, float],
    *,
    method_factors: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, float]:
    if method_factors is None:
        methods = validate_method_subset(best_factors)
        grids: Mapping[str, Sequence[float]] = {
            method: (1.0,) if method == NATIVE_METHOD else ALL_METHOD_FACTORS[method]
            for method in methods
        }
    else:
        methods = tuple(method_factors)
        if not methods or set(best_factors) != set(methods):
            raise AxBenchAdvError("best factors must cover the exact nonempty method-factor mapping")
        grids = method_factors
    parsed: dict[str, float] = {}
    for method in methods:
        value = best_factors[method]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxBenchAdvError(f"best factor for {method} must be numeric")
        factor = float(value)
        if not math.isfinite(factor):
            raise AxBenchAdvError(f"best factor for {method} must be finite")
        allowed = grids[method]
        if factor not in tuple(float(candidate) for candidate in allowed):
            raise AxBenchAdvError(f"best factor for {method} is outside its declared grid")
        parsed[method] = factor
    return parsed


def _emission_map(
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
    *,
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
    methods: tuple[str, ...] = METHODS,
) -> dict[tuple[str, int, int], RawEvaluation]:
    if type(max_concepts) is not int or max_concepts <= 0:
        raise AxBenchAdvError("max_concepts must be a positive integer")
    concepts = _concept_ids(max_concepts, concept_ids)
    expected = {
        (method, concept_id, input_id)
        for method in methods
        for concept_id in concepts
        for input_id in range(10)
    }
    if set(emissions) != expected:
        missing = sorted(expected - set(emissions))
        extra = sorted(set(emissions) - expected)
        raise AxBenchAdvError(
            f"point emission coverage differs; missing={missing[:5]}, extra={extra[:5]}"
        )
    materialized = dict(emissions)
    for key, raw in materialized.items():
        if not isinstance(raw, RawEvaluation) or not isinstance(raw.generation, str):
            raise AxBenchAdvError(f"point emission {key} is not a RawEvaluation")
    return materialized


def _concept_ids(max_concepts: int, concept_ids: Sequence[int] | None) -> tuple[int, ...]:
    selected = tuple(range(max_concepts)) if concept_ids is None else tuple(concept_ids)
    return validate_concept_ids(selected, max_concepts=max_concepts)


def _base_inputs(
    frame: pd.DataFrame,
    *,
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    required = {"dataset_name", "concept_id", "input_concept", "input_id", "original_prompt"}
    if not required.issubset(frame.columns):
        raise AxBenchAdvError(
            f"source input frame is missing {sorted(required - set(frame.columns))}"
        )
    selected = frame.loc[
        frame["factor"].eq(1.0),
        ["dataset_name", "concept_id", "input_concept", "input_id", "original_prompt"],
    ].copy()
    selected = (
        selected.drop_duplicates()
        .sort_values(["concept_id", "input_id"], kind="stable")
        .reset_index(drop=True)
    )
    concepts = _concept_ids(max_concepts, concept_ids)
    selected = selected.loc[selected["concept_id"].isin(concepts)].reset_index(drop=True)
    expected_keys = [(concept_id, input_id) for concept_id in concepts for input_id in range(10)]
    observed_keys = list(zip(selected["concept_id"], selected["input_id"], strict=True))
    if observed_keys != expected_keys:
        raise AxBenchAdvError("source inputs do not have the exact concept/input product")
    if selected.isna().any().any():
        raise AxBenchAdvError("source inputs contain null values")
    return selected


def validate_point_frame(
    table: pa.Table,
    *,
    best_factors: Mapping[str, float],
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
    method_factors: Mapping[str, Sequence[float]] | None = None,
) -> None:
    """Require the sparse union-factor frame consumed by the standard judge."""

    factors = _factor_map(best_factors, method_factors=method_factors)
    if type(max_concepts) is not int or max_concepts <= 0:
        raise AxBenchAdvError("max_concepts must be a positive integer")
    if not isinstance(table, pa.Table) or not table.schema.equals(_point_schema(tuple(factors))):
        raise AxBenchAdvError("table does not conform to POINT_FRAME_SCHEMA")
    frame = table.to_pandas()
    concepts = _concept_ids(max_concepts, concept_ids)
    union = tuple(sorted(set(factors.values())))
    expected_keys = [
        (concept_id, input_id, factor)
        for concept_id in concepts
        for input_id in range(10)
        for factor in union
    ]
    observed_keys = list(zip(frame["concept_id"], frame["input_id"], frame["factor"], strict=True))
    if observed_keys != expected_keys:
        raise AxBenchAdvError("point frame keys are not the exact canonical union-factor product")
    if frame.loc[:, list(_BASE_COLUMNS)].isna().any().any():
        raise AxBenchAdvError("point frame base columns contain null values")
    invariant = frame.groupby(["concept_id", "input_id"], sort=False, dropna=False)
    for column in ("dataset_name", "input_concept", "original_prompt"):
        if (invariant[column].nunique(dropna=False) != 1).any():
            raise AxBenchAdvError(f"point frame {column} varies across factors")
    for method, factor in factors.items():
        declared = frame["factor"].eq(factor)
        values = frame[_generation_column(method)]
        if values.loc[declared].isna().any() or values.loc[~declared].notna().any():
            raise AxBenchAdvError(f"point frame generation availability differs for {method}")


def build_point_frame(
    *,
    source_frame: pd.DataFrame,
    best_factors: Mapping[str, float],
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
    method_factors: Mapping[str, Sequence[float]] | None = None,
) -> pa.Table:
    """Build one all-method point frame without historical generations."""

    factors = _factor_map(best_factors, method_factors=method_factors)
    concepts = _concept_ids(max_concepts, concept_ids)
    materialized = _emission_map(
        emissions,
        max_concepts=max_concepts,
        concept_ids=concepts,
        methods=tuple(factors),
    )
    inputs = _base_inputs(
        source_frame,
        max_concepts=max_concepts,
        concept_ids=concepts,
    )
    rows: list[dict[str, Any]] = []
    for base in inputs.to_dict(orient="records"):
        concept_id = int(base["concept_id"])
        input_id = int(base["input_id"])
        for factor in sorted(set(factors.values())):
            row = {**base, "factor": factor}
            for method in factors:
                row[_generation_column(method)] = (
                    materialized[(method, concept_id, input_id)].generation
                    if factors[method] == factor
                    else None
                )
            rows.append(row)
    table = pa.Table.from_pylist(rows, schema=_point_schema(tuple(factors)))
    validate_point_frame(
        table,
        best_factors=factors,
        max_concepts=max_concepts,
        concept_ids=concepts,
        method_factors=method_factors,
    )
    return table


def write_point_frame(
    table: pa.Table,
    path: Path,
    *,
    best_factors: Mapping[str, float],
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
    method_factors: Mapping[str, Sequence[float]] | None = None,
) -> str:
    """Atomically publish and reread one final-judge input frame."""

    return write_validated_parquet(
        table,
        Path(path),
        validator=lambda candidate: validate_point_frame(
            candidate,
            best_factors=best_factors,
            max_concepts=max_concepts,
            concept_ids=concept_ids,
            method_factors=method_factors,
        ),
    )


def _final_config(path: Path, methods: tuple[str, ...] = METHODS) -> None:
    loaded = yaml.safe_load(EVALUATE_CONFIG.read_text(encoding="utf-8"))
    evaluate = loaded.get("evaluate") if isinstance(loaded, dict) else None
    if not isinstance(evaluate, dict):
        raise AxBenchAdvError("vendored evaluation config has no evaluate mapping")
    if evaluate.get("lm_model") != "openai/gpt-5.6-luna":
        raise AxBenchAdvError("vendored evaluation config judge model drifted")
    evaluate["models"] = list(methods)
    atomic_write_text(yaml.safe_dump(loaded, sort_keys=False), path)


def _command(
    *,
    config_path: Path,
    input_path: Path,
    output_dir: Path,
    cache_path: Path,
    cache_only: bool,
    api_base_url: str,
) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        str(VENDOR_ROOT),
        "--frozen",
        "python",
        str(EVALUATE_SCRIPT),
        "--config",
        str(config_path),
        "--mode",
        "steering",
        "--overwrite_inference_dump_dir",
        str(input_path.parent),
        "--overwrite_evaluate_dump_dir",
        str(output_dir),
        "--judge_cache",
        str(cache_path),
        "--cache_only",
        str(cache_only).lower(),
        "--api_base_url",
        api_base_url,
    ]


def _read_call_accounting(path: Path, *, expected_calls: int) -> tuple[int, int]:
    if not path.is_file():
        raise AxBenchAdvError("vendored final judge did not write call accounting")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AxBenchAdvError("final judge call accounting is malformed") from exc
    fields = {"total_calls", "total_cache_hits"}
    if not isinstance(document, dict) or set(document) != fields:
        raise AxBenchAdvError("final judge call accounting fields differ from the contract")
    calls = document["total_calls"]
    hits = document["total_cache_hits"]
    if type(calls) is not int or type(hits) is not int:
        raise AxBenchAdvError("final judge call accounting values must be integers")
    if calls < 0 or not 0 <= hits <= calls:
        raise AxBenchAdvError("final judge call accounting values are inconsistent")
    if calls != expected_calls:
        raise AxBenchAdvError(
            f"final judge made {calls} evaluator calls; expected exactly {expected_calls}"
        )
    return calls, calls - hits


def _ratings(values: object, *, field: str, length: int) -> list[float | None]:
    if not isinstance(values, list) or len(values) != length:
        raise AxBenchAdvError(f"final judge field {field} has wrong length")
    parsed: list[float | None] = []
    for value in values:
        if value is None:
            parsed.append(None)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxBenchAdvError(f"final judge field {field} contains a nonnumeric rating")
        else:
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 2.0:
                raise AxBenchAdvError(f"final judge field {field} contains a rating outside [0, 2]")
            parsed.append(numeric)
    return parsed


def _completions(values: object, *, field: str, length: int) -> list[str | None]:
    if not isinstance(values, list) or len(values) != length:
        raise AxBenchAdvError(f"final judge field {field} has wrong length")
    parsed: list[str | None] = []
    for value in values:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise AxBenchAdvError(f"final judge field {field} contains an invalid completion")
        parsed.append(value)
    return parsed


def _rationale(raw: RawEvaluation) -> tuple[str, str]:
    try:
        parsed = json.loads(raw.judge_rationale)
    except json.JSONDecodeError as exc:
        raise AxBenchAdvError("worker judge rationale is not JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"concept", "fluency"}:
        raise AxBenchAdvError("worker judge rationale fields differ from the worker contract")
    if not all(isinstance(parsed[field], str) and parsed[field].strip() for field in parsed):
        raise AxBenchAdvError("worker judge rationale completions must be nonblank strings")
    return parsed["concept"], parsed["fluency"]


def validate_rating_completion(rating: float, completion: str) -> None:
    """Bind a saved numeric grade to the vendor's one-rating response format."""
    matches = _RATING.findall(completion)
    if len(matches) != 1 or completion.count("Rating:") != 1 or float(matches[0]) != rating:
        raise AxBenchAdvError("judge rating differs from its completion")


def _expected_columns(methods: tuple[str, ...] = METHODS) -> tuple[str, ...]:
    return tuple(
        [*_point_schema(methods).names]
        + [f"{method}_{suffix}" for method in methods for suffix in _EVALUATION_FIELDS]
    )


def _validate_evaluated_frame(
    *,
    input_frame: pd.DataFrame,
    evaluated: pd.DataFrame,
    expected_values: Mapping[str, list[object]],
    methods: tuple[str, ...] = METHODS,
) -> None:
    schema = _point_schema(methods)
    expected = set(_expected_columns(methods))
    observed = set(evaluated.columns)
    if not evaluated.columns.is_unique or observed != expected:
        raise AxBenchAdvError(
            "final judged Parquet columns differ from the contract; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if tuple(evaluated.columns[: len(schema.names)]) != tuple(schema.names):
        raise AxBenchAdvError("final judged Parquet must preserve point columns as its prefix")
    if not evaluated.loc[:, schema.names].equals(input_frame):
        raise AxBenchAdvError("final judged Parquet changed its point-frame prefix")
    for column, expected_column in expected_values.items():
        observed_column = evaluated[column].tolist()
        if len(observed_column) != len(expected_column) or any(
            (not pd.isna(observed) if expected is None else observed != expected)
            for observed, expected in zip(observed_column, expected_column, strict=True)
        ):
            raise AxBenchAdvError(f"final judged Parquet {column} differs from its JSONL values")


def validate_final_judge_outputs(
    *,
    input_table: pa.Table,
    output_dir: Path,
    best_factors: Mapping[str, float],
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
    method_factors: Mapping[str, Sequence[float]] | None = None,
) -> None:
    """Validate every raw rating and prove inner concept/fluency parity."""

    factors = _factor_map(best_factors, method_factors=method_factors)
    methods = tuple(factors)
    concepts = _concept_ids(max_concepts, concept_ids)
    materialized = _emission_map(
        emissions,
        max_concepts=max_concepts,
        concept_ids=concepts,
        methods=tuple(factors),
    )
    validate_point_frame(
        input_table,
        best_factors=factors,
        max_concepts=max_concepts,
        concept_ids=concepts,
        method_factors=method_factors,
    )
    input_frame = input_table.to_pandas()
    jsonl_path = Path(output_dir) / "steering.jsonl"
    parquet_path = Path(output_dir) / "steering_data.parquet"
    if not jsonl_path.is_file() or not parquet_path.is_file():
        raise AxBenchAdvError("vendored final judge did not write both required outputs")
    entries = load_steering_entries(jsonl_path, expected_concept_ids=concepts)
    expected_values = {
        f"{method}_{suffix}": [] for method in methods for suffix in _EVALUATION_FIELDS
    }

    for entry in entries:
        results = entry["results"]
        evaluator = results.get("LMJudgeEvaluator") if isinstance(results, dict) else None
        if not isinstance(evaluator, dict) or tuple(evaluator) != methods:
            raise AxBenchAdvError("final judge method coverage or order differs")
        concept_id = int(entry["concept_id"])
        concept_frame = input_frame.loc[input_frame["concept_id"].eq(concept_id)]
        row_count = len(concept_frame)
        for method in methods:
            block = evaluator[method]
            if not isinstance(block, dict) or set(block) != _RESULT_FIELDS:
                raise AxBenchAdvError(f"final judge result fields differ for {method}")
            factor = block["factor"]
            if (
                not isinstance(factor, list) or len(factor) != 1
                or type(factor[0]) not in (int, float) or factor[0] != factors[method]
            ):
                raise AxBenchAdvError(f"final judge factor differs for {method}")
            for suffix, field in _EVALUATION_FIELDS.items():
                values = block[field]
                if isinstance(values, list):
                    expected_values[f"{method}_{suffix}"].extend(values)
            declared = concept_frame["factor"].eq(factors[method]).tolist()
            concept = _ratings(
                block["raw_relevance_concept_ratings"],
                field=f"{method}.raw_relevance_concept_ratings",
                length=row_count,
            )
            instruction = _ratings(
                block["raw_relevance_instruction_ratings"],
                field=f"{method}.raw_relevance_instruction_ratings",
                length=row_count,
            )
            fluency = _ratings(
                block["raw_fluency_ratings"],
                field=f"{method}.raw_fluency_ratings",
                length=row_count,
            )
            concept_text = _completions(
                block["relevance_concept_completions"],
                field=f"{method}.relevance_concept_completions",
                length=row_count,
            )
            instruction_text = _completions(
                block["relevance_instruction_completions"],
                field=f"{method}.relevance_instruction_completions",
                length=row_count,
            )
            fluency_text = _completions(
                block["fluency_completions"],
                field=f"{method}.fluency_completions",
                length=row_count,
            )
            aggregate = _ratings(
                block["raw_aggregated_ratings"],
                field=f"{method}.raw_aggregated_ratings",
                length=row_count,
            )
            for position, is_declared in enumerate(declared):
                values = (
                    concept[position],
                    instruction[position],
                    fluency[position],
                    aggregate[position],
                )
                texts = (concept_text[position], instruction_text[position], fluency_text[position])
                if not is_declared:
                    if any(value is not None for value in (*values, *texts)):
                        raise AxBenchAdvError(f"final judge populated an off-grid {method} row")
                    continue
                input_id = int(concept_frame.iloc[position]["input_id"])
                raw = materialized[(method, concept_id, input_id)]
                if concept_frame.iloc[position][_generation_column(method)] != raw.generation:
                    raise AxBenchAdvError(
                        f"final judge input generation differs from worker for "
                        f"{method}/c{concept_id}/i{input_id}"
                    )
                expected_concept_text, expected_fluency_text = _rationale(raw)
                if (
                    concept[position] != raw.concept_rating
                    or fluency[position] != raw.fluency_rating
                ):
                    raise AxBenchAdvError(
                        f"final judge concept/fluency rating differs from worker for "
                        f"{method}/c{concept_id}/i{input_id}"
                    )
                if (
                    concept_text[position] != expected_concept_text
                    or fluency_text[position] != expected_fluency_text
                ):
                    raise AxBenchAdvError(
                        f"final judge concept/fluency completion differs from worker for "
                        f"{method}/c{concept_id}/i{input_id}"
                    )
                instruction_rating = instruction[position]
                instruction_completion = instruction_text[position]
                if instruction_rating is None or instruction_completion is None:
                    raise AxBenchAdvError("instruction judging is absent from the final pass")
                for rating, completion in (
                    (raw.concept_rating, expected_concept_text),
                    (instruction_rating, instruction_completion),
                    (raw.fluency_rating, expected_fluency_text),
                ):
                    validate_rating_completion(rating, completion)
                scores = (
                    2.0 - raw.concept_rating
                    if concept_frame.iloc[position]["dataset_name"] == "AlpacaEvalSuppress"
                    else raw.concept_rating,
                    instruction_rating,
                    raw.fluency_rating,
                )
                expected_aggregate = 0.0 if 0.0 in scores else 3.0 / sum(1.0 / x for x in scores)
                aggregate_rating = aggregate[position]
                if aggregate_rating is None or not math.isclose(
                    aggregate_rating, expected_aggregate, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AxBenchAdvError("final judge aggregate differs from its component ratings")
            summaries = {
                "lm_judge_rating": aggregate,
                "relevance_concept_ratings": concept,
                "relevance_instruction_ratings": instruction,
                "fluency_ratings": fluency,
            }
            for field, raw_values in summaries.items():
                summary = _ratings(block[field], field=f"{method}.{field}", length=1)[0]
                present = [
                    value
                    for value, is_declared in zip(raw_values, declared, strict=True)
                    if is_declared
                ]
                if summary is None or any(value is None for value in present):
                    raise AxBenchAdvError(f"final judge summary {method}.{field} is incomplete")
                present_values = [value for value in present if value is not None]
                expected = sum(present_values) / len(present_values)
                if not math.isclose(summary, expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise AxBenchAdvError(f"final judge summary {method}.{field} is inconsistent")

    evaluated = pd.read_parquet(parquet_path)
    _validate_evaluated_frame(
        input_frame=input_frame,
        evaluated=evaluated,
        expected_values=expected_values,
        methods=methods,
    )


def run_final_judge(
    *,
    input_table: pa.Table,
    input_path: Path,
    output_dir: Path,
    cache_path: Path,
    best_factors: Mapping[str, float],
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
    max_concepts: int,
    concept_ids: Sequence[int] | None = None,
    cache_only: bool = False,
    api_base_url: str = DEFAULT_API_BASE_URL,
    method_factors: Mapping[str, Sequence[float]] | None = None,
    _process: _FramedProcess | None = None,
) -> FinalJudgeResult:
    """Run the unchanged standard evaluator, then atomically publish its outputs."""

    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    cache_path = Path(cache_path).resolve()
    api_base_url = api_base_url.strip().rstrip("/")
    if type(cache_only) is not bool:
        raise AxBenchAdvError("cache_only must be a boolean")
    if not api_base_url:
        raise AxBenchAdvError("api_base_url must be nonblank")
    if not input_path.is_file():
        raise AxBenchAdvError(f"final judge input is missing: {input_path}")
    if logical_table_digest(pq.read_table(input_path)) != logical_table_digest(input_table):
        raise AxBenchAdvError("final judge input bytes do not reread to the supplied point table")
    if not cache_path.is_file():
        raise AxBenchAdvError(f"final judge cache is missing: {cache_path}")
    if output_dir.exists():
        raise AxBenchAdvError(f"final judge output directory already exists: {output_dir}")
    if not cache_only and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise AxBenchAdvError("OPENROUTER_API_KEY is required for live final judging")
    concepts = _concept_ids(max_concepts, concept_ids)
    validate_point_frame(
        input_table,
        best_factors=best_factors,
        max_concepts=max_concepts,
        concept_ids=concepts,
        method_factors=method_factors,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="axbench-adv-final-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        config_path = staging / "evaluate.yaml"
        judged_dir = staging / "judged"
        _final_config(
            config_path,
            tuple(_factor_map(best_factors, method_factors=method_factors)),
        )
        environment = os.environ.copy()
        if "AXBENCH_ADV_PILOT_BUDGET_FILE" in environment:
            # The final judge runs from the vendored cwd, unlike the worker.
            # Its opt-in root-owned HTTP fence must resolve in that process too.
            environment["PYTHONPATH"] = os.pathsep.join((
                str(VENDOR_ROOT.parents[1]),
                *filter(None, environment.get("PYTHONPATH", "").split(os.pathsep)),
            ))
        if _process is None:
            subprocess.run(
                _command(
                    config_path=config_path,
                    input_path=input_path,
                    output_dir=judged_dir,
                    cache_path=cache_path,
                    cache_only=cache_only,
                    api_base_url=api_base_url,
                ),
                cwd=VENDOR_ROOT,
                env=environment,
                check=True,
            )
        else:
            result = _process.request(
                "evaluate",
                {
                    "config_path": str(config_path),
                    "input_dir": str(input_path.parent),
                    "output_dir": str(judged_dir),
                    "cache_path": str(cache_path),
                    "cache_only": cache_only,
                    "api_base_url": api_base_url,
                },
            )
            if result:
                raise AxBenchAdvError("final worker evaluation response must be empty")
        accounting_path = judged_dir / _CALL_ACCOUNTING_FILE
        actual_calls, paid_misses = _read_call_accounting(
            accounting_path,
            expected_calls=len(best_factors) * len(concepts) * 10 * 3,
        )
        validate_final_judge_outputs(
            input_table=input_table,
            output_dir=judged_dir,
            best_factors=best_factors,
            emissions=emissions,
            max_concepts=max_concepts,
            concept_ids=concepts,
            method_factors=method_factors,
        )
        json_text = (judged_dir / "steering.jsonl").read_text(encoding="utf-8")
        accounting_text = accounting_path.read_text(encoding="utf-8")
        evaluated_digest = logical_table_digest(pq.read_table(judged_dir / "steering_data.parquet"))
        manifest = {
            "schema_version": 1,
            "best_factors": dict(_factor_map(best_factors, method_factors=method_factors)),
            "call_accounting": {
                "actual_evaluator_calls": actual_calls,
                "paid_judge_cache_misses": paid_misses,
            },
            "files": {
                _CALL_ACCOUNTING_FILE: {
                    "sha256": hashlib.sha256(accounting_text.encode("utf-8")).hexdigest(),
                },
                "steering.jsonl": {
                    "sha256": hashlib.sha256(json_text.encode("utf-8")).hexdigest(),
                },
                "steering_data.parquet": {"logical_sha256": evaluated_digest},
            },
            "input_logical_sha256": logical_table_digest(input_table),
        }
        atomic_write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            judged_dir / "manifest.json",
        )
        os.replace(judged_dir, output_dir)
    return FinalJudgeResult(
        steering_jsonl=output_dir / "steering.jsonl",
        actual_evaluator_calls=actual_calls,
        paid_judge_cache_misses=paid_misses,
    )


class PersistentFinalJudge:
    """Reuse the vendored evaluator process across sequential final points."""

    def __init__(self, output_dir: Path) -> None:
        self._stderr_path = Path(output_dir).resolve() / "logs" / "final-worker.log"
        self._process: _FramedProcess | None = None
        self._closed = False

    def __call__(self, **kwargs: Any) -> FinalJudgeResult:
        if self._closed:
            raise AxBenchAdvError("persistent final judge is closed")
        if self._process is None:
            environment = os.environ.copy()
            if "AXBENCH_ADV_PILOT_BUDGET_FILE" in environment:
                environment["PYTHONPATH"] = os.pathsep.join((
                    str(REPO_ROOT),
                    *filter(None, environment.get("PYTHONPATH", "").split(os.pathsep)),
                ))
            self._process = _FramedProcess(
                [
                    "uv", "run", "--project", str(VENDOR_ROOT), "--frozen", "python",
                    "-m", "scripts.eval.axbench_adv.vendored.final_bootstrap",
                ],
                cwd=REPO_ROOT,
                timeout_seconds=None,
                stderr_path=self._stderr_path,
                env=environment,
            )
        return run_final_judge(**kwargs, _process=self._process)

    def __enter__(self) -> PersistentFinalJudge:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is not None:
            self._process.__exit__(exc_type, exc_value, traceback)


__all__ = [
    "FinalJudgeResult",
    "PersistentFinalJudge",
    "POINT_FRAME_SCHEMA",
    "build_point_frame",
    "run_final_judge",
    "validate_final_judge_outputs",
    "validate_point_frame",
    "write_point_frame",
]


def judge_generated_point(
    *,
    source_frame: pd.DataFrame,
    point_dir: Path,
    best_factors: Mapping[str, float],
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
    max_concepts: int,
    concept_ids: Sequence[int],
    cache_path: Path,
    cache_only: bool,
    api_base_url: str,
    method_factors: Mapping[str, Sequence[float]] | None = None,
    judge: Callable[..., FinalJudgeResult] = run_final_judge,
) -> FinalJudgeResult:
    """Publish and judge a GEPA or counter point through the identical schema.

    Pressure coordinates live in the parent directory, never in judge prompts
    or provider cache keys. This also publishes the common regenerated zero.
    """
    input_path = point_dir / "input" / "steering_data.parquet"
    table = build_point_frame(
        source_frame=source_frame,
        best_factors=best_factors,
        emissions=emissions,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
        method_factors=method_factors,
    )
    write_point_frame(
        table,
        input_path,
        best_factors=best_factors,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
        method_factors=method_factors,
    )
    return judge(
        input_table=table,
        input_path=input_path,
        output_dir=point_dir / "judged",
        cache_path=cache_path,
        best_factors=best_factors,
        emissions=emissions,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
        cache_only=cache_only,
        api_base_url=api_base_url,
        method_factors=method_factors,
    )
