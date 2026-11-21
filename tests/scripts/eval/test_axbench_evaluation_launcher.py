"""Focused tests for the thin AxBench evaluation launcher."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from scripts.eval.axbench_upstream import run_evaluation as launcher
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    METHOD_FACTORS,
)

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
EXPECTED_METHODS = (
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
)


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
                for method, method_factors in METHOD_FACTORS.items():
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
    for method in METHOD_FACTORS:
        frame[_metric_column(method)] = frame[_metric_column(method)].astype("float64")
    method_columns = [
        column
        for method in METHOD_FACTORS
        for column in (f"{method}_steered_generation", _metric_column(method))
    ]
    return frame.loc[:, [*INPUT_COLUMNS, *method_columns]]


def _json_result(method: str) -> dict[str, list[Any]]:
    factors = list(METHOD_FACTORS[method])
    declared = [
        factor in METHOD_FACTORS[method] for _input_id in range(10) for factor in FULL_FACTORS
    ]
    raw_ratings: list[object] = [1.0 if include else None for include in declared]
    completions: list[object] = ["Rating: [[1]]" if include else None for include in declared]
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
    for method, factors in METHOD_FACTORS.items():
        declared = evaluated["factor"].isin(factors)
        for suffix in _RATING_SUFFIXES:
            evaluated[f"{method}_{suffix}"] = declared.map(lambda value: 1.0 if value else math.nan)
        for suffix in _COMPLETION_SUFFIXES:
            evaluated[f"{method}_{suffix}"] = declared.map(
                lambda value: "Rating: [[1]]" if value else None
            )
    evaluated.to_parquet(output_dir / "steering_data.parquet", index=False)

    entries = []
    for concept_id in range(int(frame["concept_id"].nunique())):
        entries.append(
            {
                "concept_id": concept_id,
                "results": {
                    "LMJudgeEvaluator": {method: _json_result(method) for method in METHOD_FACTORS}
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


def _forbid_external_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("external work must not start")

    monkeypatch.setattr(launcher, "require_fresh_output_dir", forbidden)
    monkeypatch.setattr(launcher, "_require_cache_path", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)


def test_evaluation_config_declares_only_the_ten_methods_and_judge() -> None:
    config = yaml.safe_load(launcher.EVALUATE_CONFIG.read_text())["evaluate"]
    assert launcher.EXPECTED_METHODS == EXPECTED_METHODS
    assert tuple(METHOD_FACTORS) == EXPECTED_METHODS
    assert len(launcher._evaluation_columns()) == 70
    assert config["models"] == list(EXPECTED_METHODS)
    assert config["steering_evaluators"] == ["LMJudgeEvaluator"]
    assert config["lm_model"] == "openai/gpt-5.6-luna"
    assert config["num_of_workers"] == 8
    assert config["defense"] is None
    assert config["report_to"] == []
    assert config["run_winrate"] is False


def test_evaluation_config_rejects_the_old_six_method_list(tmp_path: Path) -> None:
    config = yaml.safe_load(launcher.EVALUATE_CONFIG.read_text())
    config["evaluate"]["models"] = [
        "DiffMean",
        "LinearProbe",
        "PromptSteering",
        "LoRA",
        "SelfConditioning",
        "SelfConditioningMasked",
    ]
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="models"):
        launcher._require_evaluation_config(path)


def test_evaluation_config_rejects_judge_model_drift(tmp_path: Path) -> None:
    config = yaml.safe_load(launcher.EVALUATE_CONFIG.read_text())
    config["evaluate"]["lm_model"] = "provider/other-model"
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="lm_model"):
        launcher._require_evaluation_config(path)


def test_build_command_uses_the_isolated_vendored_environment(tmp_path: Path) -> None:
    input_path = (tmp_path / "merged" / "steering_data.parquet").resolve()
    output_dir = (tmp_path / "evaluated").resolve()
    cache = (tmp_path / "judge-cache.sqlite3").resolve()

    command = launcher.build_command(
        input_path=input_path,
        output_dir=output_dir,
        cache_path=cache,
        cache_only=True,
        api_base_url="http://127.0.0.1:8042/v1",
    )

    assert command == [
        "uv",
        "run",
        "--project",
        str(launcher.VENDOR_ROOT),
        "--frozen",
        "python",
        str(launcher.EVALUATE_SCRIPT),
        "--config",
        str(launcher.EVALUATE_CONFIG),
        "--mode",
        "steering",
        "--overwrite_inference_dump_dir",
        str(input_path.parent),
        "--overwrite_evaluate_dump_dir",
        str(output_dir),
        "--judge_cache",
        str(cache),
        "--cache_only",
        "true",
        "--api_base_url",
        "http://127.0.0.1:8042/v1",
    ]


def test_cache_only_invokes_vendor_then_validates_outputs(
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


def test_nonempty_output_is_rejected_even_in_cache_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, _ = _write_input(tmp_path)
    output_dir = tmp_path / "evaluated"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale")
    cache = tmp_path / "judge-cache.sqlite3"
    cache.write_bytes(b"cache")
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(FileExistsError, match="must be empty"):
        launcher.run_evaluation(
            input_path=input_path,
            output_dir=output_dir,
            cache_path=cache,
            cache_only=True,
        )


def test_preflight_rejects_wrong_name_and_invalid_method_mask(tmp_path: Path) -> None:
    wrong_name = tmp_path / "merged.parquet"
    _merged_frame().to_parquet(wrong_name, index=False)
    with pytest.raises(ValueError, match="named steering_data.parquet"):
        launcher._read_merged_input(wrong_name)

    input_path, frame = _write_input(tmp_path / "invalid")
    frame.loc[frame["factor"].eq(0.0).idxmax(), "LoRA_perplexity"] = 2.0
    frame.to_parquet(input_path, index=False)
    with pytest.raises(ValueError, match="must be null outside factor 1.0"):
        launcher._read_merged_input(input_path)


@pytest.mark.parametrize("row_order", ["reversed", "interleaved"])
def test_preflight_rejects_noncanonical_rows_before_external_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    row_order: str,
) -> None:
    frame = _merged_frame(max_concepts=2)
    if row_order == "reversed":
        frame = frame.iloc[::-1].reset_index(drop=True)
    else:
        frame = frame.sort_values(["input_id", "factor", "concept_id"], kind="stable").reset_index(
            drop=True
        )
    input_path = tmp_path / "merged" / "steering_data.parquet"
    input_path.parent.mkdir()
    frame.to_parquet(input_path, index=False)
    output_dir = tmp_path / "evaluated"
    cache_path = tmp_path / "judge-cache.sqlite3"
    _forbid_external_work(monkeypatch)

    with pytest.raises(ValueError, match="canonical.*concept_id, input_id, factor.*order"):
        launcher.run_evaluation(
            input_path=input_path,
            output_dir=output_dir,
            cache_path=cache_path,
        )

    assert not output_dir.exists()
    assert not cache_path.exists()


@pytest.mark.parametrize(
    ("output_parts", "cache_parts"),
    [
        (("evaluated",), ("evaluated",)),
        (("evaluated",), ("evaluated", "steering_data.parquet")),
        (("judge-cache.sqlite3", "evaluated"), ("judge-cache.sqlite3",)),
    ],
)
def test_preflight_rejects_absent_output_cache_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_parts: tuple[str, ...],
    cache_parts: tuple[str, ...],
) -> None:
    input_path, _ = _write_input(tmp_path)
    output_dir = tmp_path.joinpath(*output_parts)
    cache_path = tmp_path.joinpath(*cache_parts)
    _forbid_external_work(monkeypatch)

    with pytest.raises(ValueError, match="path-disjoint"):
        launcher.run_evaluation(
            input_path=input_path,
            output_dir=output_dir,
            cache_path=cache_path,
        )

    assert not output_dir.exists()
    assert not cache_path.exists()


def test_postflight_rejects_missing_and_off_grid_judge_results(tmp_path: Path) -> None:
    frame = _merged_frame()
    output_dir = tmp_path / "evaluated"
    _write_evaluation_outputs(frame, output_dir)
    launcher.validate_evaluation_outputs(frame, output_dir)

    evaluated = pd.read_parquet(output_dir / "steering_data.parquet")
    off_grid = evaluated["factor"].ne(1.0)
    evaluated.loc[off_grid.idxmax(), "LoRA_LMJudgeEvaluator"] = 1.0
    evaluated.to_parquet(output_dir / "steering_data.parquet", index=False)
    with pytest.raises(ValueError, match="must be null off-grid"):
        launcher.validate_evaluation_outputs(frame, output_dir)


def test_postflight_rejects_wrong_json_method_cardinality(tmp_path: Path) -> None:
    frame = _merged_frame()
    output_dir = tmp_path / "evaluated"
    _write_evaluation_outputs(frame, output_dir)
    path = output_dir / "steering.jsonl"
    entry = json.loads(path.read_text())
    entry["results"]["LMJudgeEvaluator"]["LoRA"]["raw_aggregated_ratings"].append(1.0)
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="LoRA.raw_aggregated_ratings.*wrong length"):
        launcher.validate_evaluation_outputs(frame, output_dir)


def test_postflight_rejects_sparse_json_values_shifted_off_grid(tmp_path: Path) -> None:
    frame = _merged_frame()
    output_dir = tmp_path / "evaluated"
    _write_evaluation_outputs(frame, output_dir)
    path = output_dir / "steering.jsonl"
    entry = json.loads(path.read_text())
    ratings = entry["results"]["LMJudgeEvaluator"]["LoRA"]["raw_aggregated_ratings"]
    factor_zero_position = 0
    factor_one_position = list(FULL_FACTORS).index(1.0)
    ratings[factor_zero_position], ratings[factor_one_position] = (
        ratings[factor_one_position],
        ratings[factor_zero_position],
    )
    path.write_text(json.dumps(entry) + "\n")

    with pytest.raises(ValueError, match="LoRA.raw_aggregated_ratings.*non-null off-grid"):
        launcher.validate_evaluation_outputs(frame, output_dir)
