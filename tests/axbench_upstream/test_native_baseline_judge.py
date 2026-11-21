"""Standalone native baseline judge validates method/factor schema exactly."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from steering.eval.axbench_upstream import native_baseline_judge as judge
from steering.eval.axbench_upstream.native_baselines import METHOD_FACTORS


def _write_judged(tmp_path, *, method: str = "DiffMean") -> pd.DataFrame:
    factors = METHOD_FACTORS[method]
    frame = (
        pd.DataFrame(
            [
                {
                    "dataset_name": "AlpacaEval",
                    "concept_id": 0,
                    "input_id": input_id,
                    "input_concept": "flowers",
                    "original_prompt": f"Question {input_id}?",
                    "factor": factor,
                    f"{method}_steered_generation": "A fluent flower response.",
                }
                for input_id in range(10)
                for factor in factors
            ]
        )
        .sort_values(["concept_id", "input_id", "factor"], kind="stable")
        .reset_index(drop=True)
    )
    result = {"factor": list(factors)}
    evaluated = frame.copy()
    for suffix, field in judge.FIELDS.items():
        values = (
            ["Rating: [[1]]"] * len(frame) if field.endswith("completions") else [1.0] * len(frame)
        )
        result[field] = values
        evaluated[f"{method}_{suffix}"] = values
    for summary in judge.SUMMARY:
        result[summary] = [1.0] * len(factors)
    evaluated.to_parquet(tmp_path / "steering_data.parquet", index=False)
    (tmp_path / "steering.jsonl").write_text(
        json.dumps({"concept_id": 0, "results": {"LMJudgeEvaluator": {method: result}}}) + "\n"
    )
    return frame


def test_multifactor_native_judged_schema(tmp_path) -> None:
    frame = _write_judged(tmp_path)
    scores = judge.validate_native_baseline_judged(frame, tmp_path, method="DiffMean")
    assert set(scores["lm_judge_rating"]) == {
        float(factor).hex() for factor in METHOD_FACTORS["DiffMean"]
    }
    assert set(scores) == set(judge.SUMMARY)


@pytest.mark.parametrize("mutation", ["method", "factor", "prompt", "aggregate"])
def test_native_judged_schema_rejects_identity_and_arithmetic_drift(tmp_path, mutation) -> None:
    frame = _write_judged(tmp_path)
    parquet = tmp_path / "steering_data.parquet"
    jsonl = tmp_path / "steering.jsonl"
    evaluated = pd.read_parquet(parquet)
    entry = json.loads(jsonl.read_text())
    result = entry["results"]["LMJudgeEvaluator"]["DiffMean"]
    if mutation == "method":
        entry["results"]["LMJudgeEvaluator"]["LoRA"] = entry["results"]["LMJudgeEvaluator"].pop(
            "DiffMean"
        )
    elif mutation == "factor":
        result["factor"] = [1.0]
    elif mutation == "prompt":
        evaluated.loc[0, "original_prompt"] = "changed"
    else:
        result["raw_aggregated_ratings"][0] = 2.0
        evaluated.loc[0, "DiffMean_LMJudgeEvaluator"] = 2.0
    evaluated.to_parquet(parquet, index=False)
    jsonl.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError):
        judge.validate_native_baseline_judged(frame, tmp_path, method="DiffMean")
