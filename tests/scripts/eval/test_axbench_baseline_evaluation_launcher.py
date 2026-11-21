"""Focused tests for the baseline-only (four vendored methods) AxBench judge launcher.

The narrower sibling of ``test_axbench_evaluation_launcher.py``, covering
:mod:`scripts.eval.axbench_upstream.run_baseline_evaluation` -- used for a
base model (e.g. Pythia) with no project/encoder-decoder method comparison.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from scripts.eval.axbench_upstream import run_baseline_evaluation as launcher
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, INPUT_COLUMNS

_PERPLEXITY_METHODS = frozenset({"PromptSteering", "LoRA"})
_RATING_SUFFIXES = (
    "LMJudgeEvaluator",
    "LMJudgeEvaluator_relevance_concept_ratings",
    "LMJudgeEvaluator_relevance_instruction_ratings",
    "LMJudgeEvaluator_fluency_ratings",
)
_COMPLETION_SUFFIXES = (
    "LMJudgeEvaluator_relevance_concept_completions",
    "LMJudgeEvaluator_relevance_instruction_completions",
    "LMJudgeEvaluator_fluency_completions",
)
EXPECTED_METHODS = ("DiffMean", "LinearProbe", "PromptSteering", "LoRA")


def _metric_column(method: str) -> str:
    suffix = "perplexity" if method in _PERPLEXITY_METHODS else "strength"
    return f"{method}_{suffix}"


def _merged_frame(*, max_concepts: int = 1) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for concept_id in range(max_concepts):
        for input_id in range(10):
            for factor in FULL_FACTORS:
                row: dict[str, object] = {
                    "dataset_name": "AlpacaEval",
                    "concept_id": concept_id,
                    "input_concept": f"concept-{concept_id}",
                    "input_id": input_id,
                    "factor": factor,
                    "original_prompt": f"original prompt {concept_id}/{input_id}",
                    "steered_input": f"steered input {concept_id}/{input_id}",
                    "input": f"plain input {concept_id}/{input_id}",
                    "suppress_original": f"suppress original {concept_id}/{input_id}",
                    "suppress_rewrite": f"suppress rewrite {concept_id}/{input_id}",
                    "steered_prompt": f"steered prompt {concept_id}/{input_id}",
                    "defense": "none",
                    "sae_link": f"https://example.test/sae/{concept_id}",
                    "sae_id": 100 + concept_id,
                }
                for method in EXPECTED_METHODS:
                    method_factors = (1.0,) if method in _PERPLEXITY_METHODS else FULL_FACTORS
                    declared = factor in method_factors
                    row[f"{method}_steered_generation"] = (
                        f"{method} healthy generated answer {concept_id}/{input_id}/{factor}"
                        if declared
                        else None
                    )
                    row[_metric_column(method)] = (
                        (2.0 if method in _PERPLEXITY_METHODS else factor + 1.0)
                        if declared
                        else math.nan
                    )
                rows.append(row)

    frame = pd.DataFrame(rows)
    for column in ("concept_id", "input_id", "sae_id"):
        frame[column] = frame[column].astype("int64")
    frame["factor"] = frame["factor"].astype("float64")
    for method in EXPECTED_METHODS:
        frame[_metric_column(method)] = frame[_metric_column(method)].astype("float64")
    method_columns = [
        column
        for method in EXPECTED_METHODS
        for column in (f"{method}_steered_generation", _metric_column(method))
    ]
    return frame.loc[:, [*INPUT_COLUMNS, *method_columns]]


def _json_result(method: str) -> dict[str, list[Any]]:
    method_factors = (1.0,) if method in _PERPLEXITY_METHODS else FULL_FACTORS
    factors = list(method_factors)
    declared = [factor in method_factors for _input_id in range(10) for factor in FULL_FACTORS]
    raw_ratings: list[object] = [1.0 if include else None for include in declared]
    completions: list[object] = ["Rating: 1" if include else None for include in declared]
    return {
        "lm_judge_rating": [1.0] * len(factors),
        "relevance_concept_ratings": [1.0] * len(factors),
        "relevance_instruction_ratings": [1.0] * len(factors),
        "fluency_ratings": [1.0] * len(factors),
        "factor": factors,
        "raw_relevance_concept_ratings": raw_ratings.copy(),
        "raw_relevance_instruction_ratings": raw_ratings.copy(),
        "raw_fluency_ratings": raw_ratings.copy(),
        "raw_aggregated_ratings": raw_ratings.copy(),
        "relevance_concept_completions": completions.copy(),
        "relevance_instruction_completions": completions.copy(),
        "fluency_completions": completions.copy(),
    }


def _write_evaluation_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluated = frame.copy()
    for method in EXPECTED_METHODS:
        method_factors = (1.0,) if method in _PERPLEXITY_METHODS else FULL_FACTORS
        declared = evaluated["factor"].isin(method_factors)
        for suffix in _RATING_SUFFIXES:
            evaluated[f"{method}_{suffix}"] = declared.map(lambda value: 1.0 if value else math.nan)
        for suffix in _COMPLETION_SUFFIXES:
            evaluated[f"{method}_{suffix}"] = declared.map(
                lambda value: "Rating: 1" if value else None
            )
    evaluated.to_parquet(output_dir / "steering_data.parquet", index=False)

    entries = []
    for concept_id in range(int(frame["concept_id"].nunique())):
        entries.append(
            {
                "concept_id": concept_id,
                "results": {
                    "LMJudgeEvaluator": {
                        method: _json_result(method) for method in EXPECTED_METHODS
                    }
                },
            }
        )
    (output_dir / "steering.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries)
    )


def _write_input(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    input_path = tmp_path / "merged" / "steering_data.parquet"
    input_path.parent.mkdir(parents=True)
    frame = _merged_frame()
    frame.to_parquet(input_path, index=False)
    return input_path, pd.read_parquet(input_path)


def test_baseline_evaluation_config_declares_only_the_four_methods() -> None:
    config = yaml.safe_load(launcher.EVALUATE_CONFIG.read_text())["evaluate"]
    assert launcher.EXPECTED_METHODS == EXPECTED_METHODS
    assert config["models"] == list(EXPECTED_METHODS)
    assert config["steering_evaluators"] == ["LMJudgeEvaluator"]
    assert config["lm_model"] == "openai/gpt-5.6-luna"
    assert config["defense"] is None


def test_baseline_evaluation_config_rejects_the_ten_method_list(tmp_path: Path) -> None:
    config = yaml.safe_load(launcher.EVALUATE_CONFIG.read_text())
    config["evaluate"]["models"] = [
        "DiffMean",
        "LinearProbe",
        "PromptSteering",
        "LoRA",
        "EncDecDiffMean",
        "EncDecLinearProbe",
        "EncDecPromptSteering",
        "EncDecLoRA",
        "SelfConditioning",
        "SelfConditioningMasked",
    ]
    path = tmp_path / "baseline_evaluate.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="models"):
        launcher._require_evaluation_config(path)


def test_build_command_points_at_the_baseline_config() -> None:
    input_path = Path("/tmp/merged/steering_data.parquet")
    command = launcher.build_command(
        input_path=input_path,
        output_dir=Path("/tmp/evaluated"),
        cache_path=Path("/tmp/judge-cache.sqlite3"),
        cache_only=True,
        api_base_url="http://127.0.0.1:8042/v1",
    )
    assert str(launcher.EVALUATE_CONFIG) in command
    assert "baseline_evaluate.yaml" in command[command.index("--config") + 1]


def test_cache_only_invokes_vendor_then_validates_baseline_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, frame = _write_input(tmp_path)
    output_dir = tmp_path / "evaluated"
    cache = tmp_path / "judge-cache.sqlite3"
    cache.write_bytes(b"ordinary existing cache")
    calls: list[tuple[list[str], Path, bool]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> None:
        assert "OPENROUTER_API_KEY" not in env
        calls.append((command, cwd, check))
        _write_evaluation_outputs(frame, output_dir)

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.run_evaluation(
        input_path=input_path,
        output_dir=output_dir,
        cache_path=cache,
        cache_only=True,
        api_base_url="http://127.0.0.1:8042/v1/",
    )

    assert len(calls) == 1
    command, cwd, check = calls[0]
    assert cwd == launcher.VENDOR_ROOT
    assert check is True
    assert command[-1] == "http://127.0.0.1:8042/v1"


def test_live_mode_requires_key_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, _ = _write_input(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        launcher.run_evaluation(
            input_path=input_path,
            output_dir=tmp_path / "evaluated",
            cache_path=tmp_path / "judge-cache.sqlite3",
        )


def test_preflight_rejects_a_ten_method_shaped_frame(tmp_path: Path) -> None:
    """A merged frame carrying encdec/project columns is not a baseline-only frame."""

    frame = _merged_frame()
    frame["SelfConditioning_steered_generation"] = "unexpected column"
    input_path = tmp_path / "merged" / "steering_data.parquet"
    input_path.parent.mkdir(parents=True)
    frame.to_parquet(input_path, index=False)
    with pytest.raises(ValueError, match="unknown method result columns"):
        launcher._read_merged_input(input_path)


def test_preflight_accepts_the_baseline_blank_bound_not_the_stricter_default(
    tmp_path: Path,
) -> None:
    """LinearProbe et al. carry real, occasional blank generations even in a
    healthy merge -- the shared reader's own strict 0.01 default has already
    been raised project-wide to BASELINE_MAX_BLANK_GENERATION_FRACTION (0.02)
    for baseline methods (see merge_baseline_methods.py); the four-method
    reader must use that same raised bound, not silently fall back to 0.01."""

    frame = _merged_frame()  # max_concepts=1 -> 10 inputs * 12 factors = 120
    # declared LinearProbe rows (FULL_FACTORS applies to every non-perplexity
    # method). 2/120 = 0.0167: above the shared default (0.01), at/under the
    # baseline-ratified bound (0.02).
    blanked = frame.index[:2]
    frame.loc[blanked, "LinearProbe_steered_generation"] = ""
    input_path = tmp_path / "merged" / "steering_data.parquet"
    input_path.parent.mkdir(parents=True)
    frame.to_parquet(input_path, index=False)

    result = launcher._read_merged_input(input_path)

    assert len(result) == len(frame)
