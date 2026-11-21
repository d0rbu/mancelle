"""The native judge must reject corrupt metrics and changed benchmark inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from steering.eval.axbench_upstream.native_judge import (
    FIELDS,
    METHOD,
    SUMMARY,
    validate_native_judged,
)


def fixture_output(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "dataset_name": ["AlpacaEval"] * 10,
            "concept_id": [0] * 10,
            "input_id": list(range(10)),
            "input_concept": ["flowers"] * 10,
            "original_prompt": ["Say hi"] * 10,
            "factor": [1.0] * 10,
            f"{METHOD}_steered_generation": ["Hello, flower."] * 10,
        }
    )
    result = {"factor": [1.0]}
    evaluated = frame.copy()
    for suffix, field in FIELDS.items():
        values = ["Rating: [[1]]"] * 10 if field.endswith("completions") else [1.0] * 10
        result[field] = values
        evaluated[f"{METHOD}_{suffix}"] = values
    for summary in SUMMARY:
        result[summary] = [1.0]
    evaluated.to_parquet(path / "steering_data.parquet", index=False)
    (path / "steering.jsonl").write_text(
        json.dumps({"concept_id": 0, "results": {"LMJudgeEvaluator": {METHOD: result}}}) + "\n"
    )
    return frame


def test_valid_native_judge(tmp_path):
    frame = fixture_output(tmp_path)
    assert validate_native_judged(frame, tmp_path) == {
        "concept": 1.0,
        "instruction": 1.0,
        "fluency": 1.0,
        "aggregate": 1.0,
    }


@pytest.mark.parametrize(
    "mutation",
    ["prompt", "missing_row", "wrong_method", "nan", "raw_disagreement", "aggregate", "summary",
     "columns", "concept_id", "factor", "rating_count", "blank_completion"],
)
def test_reject_corrupt_judge(tmp_path, mutation):
    frame = fixture_output(tmp_path)
    parquet = tmp_path / "steering_data.parquet"
    jsonl = tmp_path / "steering.jsonl"
    evaluated = pd.read_parquet(parquet)
    entry = json.loads(jsonl.read_text())
    result = entry["results"]["LMJudgeEvaluator"][METHOD]
    if mutation == "prompt":
        evaluated.loc[0, "original_prompt"] = "Changed benchmark question"
    elif mutation == "missing_row":
        evaluated = evaluated.iloc[:-1]
    elif mutation == "wrong_method":
        entry["results"]["LMJudgeEvaluator"]["SelfConditioning"] = entry["results"][
            "LMJudgeEvaluator"
        ].pop(METHOD)
    elif mutation == "nan":
        result["raw_fluency_ratings"][0] = float("nan")
    elif mutation == "raw_disagreement":
        result["raw_fluency_ratings"][0] = 0.0
    elif mutation == "aggregate":
        result["raw_aggregated_ratings"] = [2.0] * 10
        result["lm_judge_rating"] = [2.0]
        evaluated[f"{METHOD}_LMJudgeEvaluator"] = [2.0] * 10
    elif mutation == "summary":
        result["fluency_ratings"] = [0.0]
    elif mutation == "columns":
        evaluated["unexpected"] = 1
    elif mutation == "concept_id":
        entry["concept_id"] = 1
    elif mutation == "factor":
        result["factor"] = [2.0]
    elif mutation == "rating_count":
        result["raw_fluency_ratings"].pop()
    elif mutation == "blank_completion":
        result["fluency_completions"][0] = ""
    evaluated.to_parquet(parquet, index=False)
    jsonl.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError):
        validate_native_judged(frame, tmp_path)
