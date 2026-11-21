"""Judge the selected baseline-plus-project merged Parquet with vendored AxBench.

Unlike :mod:`run_evaluation` (all ten methods) and :mod:`run_baseline_evaluation`
(the four vendored baselines alone), this judges six methods: the four
vendored baselines (DiffMean, LinearProbe, PromptSteering, LoRA) plus the
project's own SelfConditioning and SelfConditioningMasked -- for a base model
(e.g. Pythia, SmolLM3) with no encoder-decoder baseline analogs.
The explicit NL option adds ``NLSteering`` without changing that default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from scripts.eval.axbench_upstream.merge_baseline_methods import (
    BASELINE_MAX_BLANK_GENERATION_FRACTION,
)
from scripts.eval.axbench_upstream.merge_baseline_project_methods import (
    BASELINE_PROJECT_METHODS,
    BASELINE_PROJECT_NL_METHODS,
    require_baseline_project_methods,
)
from scripts.eval.axbench_upstream.merge_methods import (
    SUPPORTED_METHOD_FACTORS,
    read_canonical_merged_frame,
    require_fresh_output_dir,
)
from scripts.eval.axbench_upstream.run_evaluation import (
    _PARQUET_SUFFIXES,
    _as_list,
    _require_cache_path,
    _require_disjoint_output_and_cache,
    _require_finite_ratings,
    _require_raw_coverage,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "vendor" / "axbench"
EVALUATE_SCRIPT = VENDOR_ROOT / "axbench" / "scripts" / "evaluate.py"
EVALUATE_CONFIG = (
    VENDOR_ROOT
    / "axbench"
    / "sweep"
    / "nano_gpt_pretrain_steer"
    / "baseline_project_evaluate.yaml"
)
NL_EVALUATE_CONFIG = EVALUATE_CONFIG.with_name("baseline_project_nl_evaluate.yaml")
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
EXPECTED_METHODS = BASELINE_PROJECT_METHODS

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


def _method_factors(methods: Sequence[str]) -> dict[str, tuple[float, ...]]:
    return {method: SUPPORTED_METHOD_FACTORS[method] for method in methods}


def _evaluation_columns(
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> tuple[str, ...]:
    return tuple(
        f"{method}_{suffix}" for method in methods for suffix in _PARQUET_SUFFIXES
    )


def _require_evaluation_config(
    path: Path = EVALUATE_CONFIG,
    *,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> None:
    methods = require_baseline_project_methods(methods)
    if not path.is_file():
        raise FileNotFoundError(f"AxBench baseline+project evaluation config not found: {path}")
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, Mapping) or not isinstance(loaded.get("evaluate"), Mapping):
        raise ValueError(f"config is missing an 'evaluate' mapping: {path}")
    config = loaded["evaluate"]
    expected = {
        "models": list(methods),
        "steering_evaluators": ["LMJudgeEvaluator"],
        "lm_model": "openai/gpt-5.6-luna",
        "defense": None,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"evaluation config {field!r} must be exactly {value!r}: {path}")


def _read_merged_input(
    path: Path,
    *,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> pd.DataFrame:
    """Preflight the selected baseline-plus-project frame with the shared reader."""
    return read_canonical_merged_frame(
        path,
        surface="evaluation",
        methods=require_baseline_project_methods(methods),
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )


def _validate_jsonl(
    path: Path,
    *,
    input_frame: pd.DataFrame,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> None:
    selected_methods = require_baseline_project_methods(methods)
    method_factors = _method_factors(selected_methods)
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
        result_methods = results["LMJudgeEvaluator"]
        if not isinstance(result_methods, Mapping) or tuple(result_methods) != selected_methods:
            raise ValueError("evaluation JSONL method coverage or order does not match the config")

        concept_frame = input_frame.loc[input_frame["concept_id"].eq(entry["concept_id"])]
        for method, expected_factors in method_factors.items():
            result = result_methods[method]
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
                    values, field=f"{method}.{field}", declared=declared, completions=False
                )
            for field in _COMPLETION_FIELDS:
                values = _as_list(result.get(field), field=f"{method}.{field}")
                _require_raw_coverage(
                    values, field=f"{method}.{field}", declared=declared, completions=True
                )


def _validate_parquet(
    input_frame: pd.DataFrame,
    path: Path,
    *,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> None:
    methods = require_baseline_project_methods(methods)
    method_factors = _method_factors(methods)
    if not path.is_file():
        raise FileNotFoundError(f"AxBench did not write evaluated Parquet: {path}")
    evaluated = pd.read_parquet(path)
    if not evaluated.columns.is_unique:
        raise ValueError("evaluated Parquet has duplicate columns")

    expected_columns = (*input_frame.columns, *_evaluation_columns(methods))
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

    for method, factors in method_factors.items():
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


def validate_evaluation_outputs(
    input_frame: pd.DataFrame,
    output_dir: Path,
    *,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> None:
    """Check exact concept, method, factor, rating, and completion coverage."""

    methods = require_baseline_project_methods(methods)
    _validate_jsonl(output_dir / "steering.jsonl", input_frame=input_frame, methods=methods)
    _validate_parquet(input_frame, output_dir / "steering_data.parquet", methods=methods)


def build_command(
    *,
    input_path: Path,
    output_dir: Path,
    cache_path: Path,
    cache_only: bool,
    api_base_url: str,
    methods: tuple[str, ...] = EXPECTED_METHODS,
) -> list[str]:
    """Build the isolated vendored invocation against the selected exact config."""

    methods = require_baseline_project_methods(methods)
    config_path = (
        NL_EVALUATE_CONFIG if methods == BASELINE_PROJECT_NL_METHODS else EVALUATE_CONFIG
    )
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


def run_evaluation(
    *,
    input_path: Path,
    output_dir: Path,
    cache_path: Path,
    cache_only: bool = False,
    api_base_url: str = DEFAULT_API_BASE_URL,
    include_nl_steering: bool = False,
) -> None:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    cache_path = Path(cache_path).expanduser().resolve()
    api_base_url = api_base_url.strip().rstrip("/")
    if not api_base_url:
        raise ValueError("--api-base-url must be nonblank")

    if type(include_nl_steering) is not bool:
        raise TypeError("include_nl_steering must be a boolean")
    methods = BASELINE_PROJECT_NL_METHODS if include_nl_steering else EXPECTED_METHODS
    config_path = NL_EVALUATE_CONFIG if include_nl_steering else EVALUATE_CONFIG
    _require_evaluation_config(config_path, methods=methods)
    input_frame = _read_merged_input(input_path, methods=methods)
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
        methods=methods,
    )
    logger.info("Running vendored AxBench baseline+project evaluation: {}", " ".join(command))
    subprocess.run(command, cwd=VENDOR_ROOT, env=os.environ.copy(), check=True)
    validate_evaluation_outputs(input_frame, output_dir, methods=methods)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument(
        "--include-nl-steering",
        action="store_true",
        help="judge the explicit seven-method product including NLSteering",
    )
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
        include_nl_steering=args.include_nl_steering,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
