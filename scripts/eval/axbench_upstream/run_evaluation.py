"""Run the vendored AxBench judge on one validated merged steering Parquet."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from scripts.eval.axbench_upstream.merge_methods import (
    METHOD_FACTORS,
    read_canonical_merged_frame,
    require_fresh_output_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "vendor" / "axbench"
EVALUATE_SCRIPT = VENDOR_ROOT / "axbench" / "scripts" / "evaluate.py"
EVALUATE_CONFIG = VENDOR_ROOT / "axbench" / "sweep" / "nano_gpt_pretrain_steer" / "evaluate.yaml"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
EXPECTED_METHODS = tuple(METHOD_FACTORS)

_SUMMARY_RATING_FIELDS = (
    "lm_judge_rating",
    "relevance_concept_ratings",
    "relevance_instruction_ratings",
    "fluency_ratings",
)
_RAW_RATING_FIELDS = (
    "raw_relevance_concept_ratings",
    "raw_relevance_instruction_ratings",
    "raw_fluency_ratings",
    "raw_aggregated_ratings",
)
_COMPLETION_FIELDS = (
    "relevance_concept_completions",
    "relevance_instruction_completions",
    "fluency_completions",
)
_PARQUET_SUFFIXES = (
    "LMJudgeEvaluator",
    "LMJudgeEvaluator_relevance_concept_ratings",
    "LMJudgeEvaluator_relevance_concept_completions",
    "LMJudgeEvaluator_relevance_instruction_ratings",
    "LMJudgeEvaluator_relevance_instruction_completions",
    "LMJudgeEvaluator_fluency_ratings",
    "LMJudgeEvaluator_fluency_completions",
)


def _evaluation_columns() -> tuple[str, ...]:
    return tuple(
        f"{method}_{suffix}" for method in EXPECTED_METHODS for suffix in _PARQUET_SUFFIXES
    )


def _require_evaluation_config(path: Path = EVALUATE_CONFIG) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"AxBench evaluation config not found: {path}")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("evaluate"), Mapping):
        raise ValueError(f"config is missing an 'evaluate' mapping: {path}")
    config = loaded["evaluate"]
    expected = {
        "models": list(EXPECTED_METHODS),
        "steering_evaluators": ["LMJudgeEvaluator"],
        "lm_model": "openai/gpt-5.6-luna",
        "defense": None,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"evaluation config {field!r} must be exactly {value!r}: {path}")


def _read_merged_input(path: Path) -> pd.DataFrame:
    """The vendored judge's merged-frame preflight: the shared reader, evaluator wording."""
    return read_canonical_merged_frame(path, surface="evaluation")


def _require_cache_path(path: Path, *, cache_only: bool) -> None:
    if path.exists() and not path.is_file():
        raise ValueError(f"judge cache path is not a file: {path}")
    if cache_only and not path.is_file():
        raise FileNotFoundError(f"--cache-only requires an existing judge cache: {path}")


def _require_disjoint_output_and_cache(output_dir: Path, cache_path: Path) -> None:
    if (
        output_dir == cache_path
        or cache_path.is_relative_to(output_dir)
        or output_dir.is_relative_to(cache_path)
    ):
        raise ValueError("judge cache and evaluation output directory must be path-disjoint")


def _as_list(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"evaluation JSONL field {field!r} must be a list")
    return value


def _require_finite_ratings(values: list[Any], *, field: str) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"evaluation JSONL field {field!r} contains a nonnumeric rating")
        if not math.isfinite(float(value)):
            raise ValueError(f"evaluation JSONL field {field!r} contains a nonfinite rating")


def _require_raw_coverage(
    values: list[Any],
    *,
    field: str,
    declared: list[bool],
    completions: bool,
) -> None:
    if len(values) != len(declared):
        raise ValueError(f"evaluation JSONL field {field} has wrong length")
    for position, (value, is_declared) in enumerate(zip(values, declared, strict=True)):
        if is_declared and value is None:
            raise ValueError(
                f"evaluation JSONL field {field} is null on-grid at position {position}"
            )
        if not is_declared and value is not None:
            raise ValueError(
                f"evaluation JSONL field {field} is non-null off-grid at position {position}"
            )
    present = [value for value, is_declared in zip(values, declared, strict=True) if is_declared]
    if completions:
        if not all(isinstance(value, str) and value.strip() for value in present):
            raise ValueError(f"evaluation JSONL field {field} contains an invalid completion")
    else:
        _require_finite_ratings(present, field=field)


def _validate_jsonl(path: Path, *, input_frame: pd.DataFrame) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"AxBench did not write evaluation JSONL: {path}")
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    max_concepts = int(input_frame["concept_id"].nunique())
    concept_ids = [entry.get("concept_id") for entry in entries if isinstance(entry, Mapping)]
    if concept_ids != list(range(max_concepts)) or len(entries) != max_concepts:
        raise ValueError(
            "evaluation JSONL must contain each concept exactly once in ascending order"
        )

    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"concept_id", "results"}:
            raise ValueError("evaluation JSONL entries must contain only concept_id and results")
        results = entry["results"]
        if not isinstance(results, Mapping) or set(results) != {"LMJudgeEvaluator"}:
            raise ValueError("evaluation JSONL must contain exactly LMJudgeEvaluator results")
        methods = results["LMJudgeEvaluator"]
        if not isinstance(methods, Mapping) or tuple(methods) != EXPECTED_METHODS:
            raise ValueError("evaluation JSONL method coverage or order does not match the config")

        concept_frame = input_frame.loc[input_frame["concept_id"].eq(entry["concept_id"])]
        for method, expected_factors in METHOD_FACTORS.items():
            result = methods[method]
            if not isinstance(result, Mapping):
                raise ValueError(f"evaluation JSONL result for {method} must be a mapping")
            factors = _as_list(result.get("factor"), field=f"{method}.factor")
            if tuple(float(factor) for factor in factors) != expected_factors:
                raise ValueError(f"evaluation JSONL factors do not match {method}")
            for field in _SUMMARY_RATING_FIELDS:
                ratings = _as_list(result.get(field), field=f"{method}.{field}")
                if len(ratings) != len(expected_factors):
                    raise ValueError(f"evaluation JSONL field {method}.{field} has wrong length")
                _require_finite_ratings(ratings, field=f"{method}.{field}")
            declared = concept_frame["factor"].isin(expected_factors).tolist()
            for field in _RAW_RATING_FIELDS:
                values = _as_list(result.get(field), field=f"{method}.{field}")
                _require_raw_coverage(
                    values,
                    field=f"{method}.{field}",
                    declared=declared,
                    completions=False,
                )
            for field in _COMPLETION_FIELDS:
                values = _as_list(result.get(field), field=f"{method}.{field}")
                _require_raw_coverage(
                    values,
                    field=f"{method}.{field}",
                    declared=declared,
                    completions=True,
                )


def _validate_parquet(input_frame: pd.DataFrame, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"AxBench did not write evaluated Parquet: {path}")
    evaluated = pd.read_parquet(path)
    if not evaluated.columns.is_unique:
        raise ValueError("evaluated Parquet has duplicate columns")

    expected_columns = (*input_frame.columns, *_evaluation_columns())
    missing = sorted(set(expected_columns) - set(evaluated.columns))
    extra = sorted(set(evaluated.columns) - set(expected_columns))
    if missing or extra:
        raise ValueError(
            f"evaluated Parquet columns do not match; missing={missing}, extra={extra}"
        )
    if tuple(evaluated.columns[: len(input_frame.columns)]) != tuple(input_frame.columns):
        raise ValueError("evaluated Parquet must preserve the input columns as its prefix")
    if not evaluated.loc[:, input_frame.columns].equals(input_frame):
        raise ValueError("evaluated Parquet changed input rows, values, dtypes, or order")

    for method, factors in METHOD_FACTORS.items():
        declared = evaluated["factor"].isin(factors)
        for suffix in _PARQUET_SUFFIXES:
            column = f"{method}_{suffix}"
            if evaluated.loc[declared, column].isna().any():
                raise ValueError(f"evaluated Parquet column {column!r} is null on-grid")
            if evaluated.loc[~declared, column].notna().any():
                raise ValueError(f"evaluated Parquet column {column!r} must be null off-grid")

        rating_columns = (
            f"{method}_LMJudgeEvaluator",
            f"{method}_LMJudgeEvaluator_relevance_concept_ratings",
            f"{method}_LMJudgeEvaluator_relevance_instruction_ratings",
            f"{method}_LMJudgeEvaluator_fluency_ratings",
        )
        for column in rating_columns:
            values = evaluated.loc[declared, column]
            if not pd.api.types.is_numeric_dtype(values.dtype):
                raise ValueError(f"evaluated Parquet column {column!r} must be numeric")
            if not values.map(math.isfinite).all():
                raise ValueError(f"evaluated Parquet column {column!r} must be finite on-grid")

        completion_columns = (
            f"{method}_LMJudgeEvaluator_relevance_concept_completions",
            f"{method}_LMJudgeEvaluator_relevance_instruction_completions",
            f"{method}_LMJudgeEvaluator_fluency_completions",
        )
        for column in completion_columns:
            values = evaluated.loc[declared, column]
            if not values.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError(
                    f"evaluated Parquet column {column!r} must contain nonblank completions on-grid"
                )


def validate_evaluation_outputs(input_frame: pd.DataFrame, output_dir: Path) -> None:
    """Check exact concept, method, factor, rating, and completion coverage."""

    _validate_jsonl(output_dir / "steering.jsonl", input_frame=input_frame)
    _validate_parquet(input_frame, output_dir / "steering_data.parquet")


def build_command(
    *,
    input_path: Path,
    output_dir: Path,
    cache_path: Path,
    cache_only: bool,
    api_base_url: str,
) -> list[str]:
    """Build the isolated vendored-environment invocation."""

    return [
        "uv",
        "run",
        "--project",
        str(VENDOR_ROOT),
        "--frozen",
        "python",
        str(EVALUATE_SCRIPT),
        "--config",
        str(EVALUATE_CONFIG),
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


def run_evaluation(
    *,
    input_path: Path,
    output_dir: Path,
    cache_path: Path,
    cache_only: bool = False,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> None:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    api_base_url = api_base_url.strip().rstrip("/")
    if not api_base_url:
        raise ValueError("--api-base-url must be nonblank")

    _require_evaluation_config()
    input_frame = _read_merged_input(input_path)
    _require_disjoint_output_and_cache(output_dir, cache_path)
    require_fresh_output_dir(output_dir, what="evaluation")
    _require_cache_path(cache_path, cache_only=cache_only)
    if not cache_only and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("OPENROUTER_API_KEY is required unless --cache-only is set")

    command = build_command(
        input_path=input_path,
        output_dir=output_dir,
        cache_path=cache_path,
        cache_only=cache_only,
        api_base_url=api_base_url,
    )
    logger.info("Running vendored AxBench evaluation: {}", " ".join(command))
    subprocess.run(command, cwd=VENDOR_ROOT, env=os.environ.copy(), check=True)
    validate_evaluation_outputs(input_frame, output_dir)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_evaluation(
        input_path=args.input,
        output_dir=args.output_dir,
        cache_path=args.cache,
        cache_only=args.cache_only,
        api_base_url=args.api_base_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
