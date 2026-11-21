"""Independent score and identity checks for the EncDec shared judge interfaces."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from steering.eval.axbench_adv.best_point import BestPoint, select_best_points
from steering.eval.axbench_adv.encdec_sources import METHOD_FACTORS_ENCDEC
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import (
    build_point_frame,
    validate_final_judge_outputs,
    validate_point_frame,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from tests.axbench_adv.unit.test_final_judge import _outputs, _raw, _source

FACTORS = {
    "EncDecPromptSteering": 1.0,
    "EncDecDiffMean": 2.0,
    "EncDecLinearProbe": 3.0,
    "EncDecLoRA": 1.0,
}


@pytest.fixture
def encdec_point(tmp_path: Path):
    emissions = {
        (method, 1, input_id): _raw(method, 1, input_id)
        for method in ENCDEC_BASELINE_METHODS
        for input_id in range(10)
    }
    args = {
        "best_factors": FACTORS,
        "emissions": emissions,
        "max_concepts": 2,
        "concept_ids": (1,),
        "method_factors": METHOD_FACTORS_ENCDEC,
    }
    table = build_point_frame(source_frame=_source(), **args)
    output = tmp_path / "judged"
    _outputs(output, table, emissions, factors=FACTORS, concept_ids=(1,))
    validate_final_judge_outputs(input_table=table, output_dir=output, **args)
    return table, output, args


def test_encdec_factor_selection_matches_literal_macro_oracle() -> None:
    # EncDec winners must come from the EncDec blocks. The decoy unconditioned
    # blocks favor a different factor and must never affect their selection.
    entries = []
    for concept_id in range(2):
        blocks = {}
        for method, grid in METHOD_FACTORS_ENCDEC.items():
            concept = [0.0] * len(grid)
            fluency = [2.0] * len(grid)
            winner = grid.index(FACTORS[method])
            concept[winner] = 1.0 if concept_id == 0 else 2.0
            if method == "EncDecDiffMean":
                # An equally good later factor checks the declared-grid tie rule.
                concept[grid.index(3.0)] = concept[winner]
            blocks[method] = {
                "factor": list(grid),
                "relevance_concept_ratings": concept,
                "fluency_ratings": fluency,
            }
        blocks["DiffMean"] = {
            "factor": [6.0],
            "relevance_concept_ratings": [2.0],
            "fluency_ratings": [2.0],
        }
        entries.append({"concept_id": concept_id, "results": {"LMJudgeEvaluator": blocks}})
    points = select_best_points(
        entries, ENCDEC_BASELINE_METHODS, method_factors=METHOD_FACTORS_ENCDEC
    )
    assert points == {
        method: BestPoint(method, factor, 1.5, 2.0, 12.0 / 7.0)
        for method, factor in FACTORS.items()
    }
    with pytest.raises(AxBenchAdvError, match="unknown attacked method"):
        select_best_points(entries, ENCDEC_BASELINE_METHODS)


def test_encdec_selected_factors_survive_sorted_json_roundtrip(encdec_point) -> None:
    table, output, args = encdec_point
    reordered = json.loads(json.dumps(FACTORS, sort_keys=True))
    assert tuple(reordered) != ENCDEC_BASELINE_METHODS
    rebuilt = build_point_frame(source_frame=_source(), **(args | {"best_factors": reordered}))
    assert rebuilt.equals(table)
    validate_final_judge_outputs(
        input_table=rebuilt, output_dir=output, **(args | {"best_factors": reordered})
    )


def test_encdec_factor_frame_does_not_open_generic_method_grid(encdec_point) -> None:
    table, _, args = encdec_point
    assert table.num_rows == 30
    with pytest.raises(AxBenchAdvError):
        validate_point_frame(
            table, best_factors=FACTORS,
            max_concepts=args["max_concepts"], concept_ids=args["concept_ids"],
        )


@pytest.mark.parametrize("change", ["missing", "extra", "empty", "boolean", "off_grid"])
def test_encdec_factor_mapping_rejects_invalid_selection(encdec_point, change: str) -> None:
    table, _, args = encdec_point
    factors = dict(FACTORS)
    grid = METHOD_FACTORS_ENCDEC
    if change == "missing":
        factors.pop("EncDecLoRA")
    elif change == "extra":
        factors["DiffMean"] = 1.0
    elif change == "empty":
        factors, grid = {}, {}
    elif change == "boolean":
        factors["EncDecLoRA"] = True
    else:
        factors["EncDecLoRA"] = 2.0
    with pytest.raises(AxBenchAdvError):
        validate_point_frame(
            table,
            best_factors=factors,
            max_concepts=args["max_concepts"],
            concept_ids=args["concept_ids"],
            method_factors=grid,
        )


@pytest.mark.parametrize("mutation", [
    "aggregate", "boolean_factor", "boolean_concept", "float_concept",
    "duplicate_concept", "nonfinite_results",
])
def test_encdec_judge_rejects_consistent_but_invalid_artifacts(encdec_point, mutation: str) -> None:
    table, output, args = encdec_point
    path = output / "steering.jsonl"
    entry = json.loads(path.read_text())
    block = entry["results"]["LMJudgeEvaluator"]["EncDecPromptSteering"]
    if mutation == "aggregate":
        # A correlated writer defect can alter JSONL, parquet and its mean
        # together. Agreement between files is insufficient to prove the score.
        block["raw_aggregated_ratings"][0] = 0.0
        present = [value for value in block["raw_aggregated_ratings"] if value is not None]
        block["lm_judge_rating"] = [sum(present) / len(present)]
        evaluated = pd.read_parquet(output / "steering_data.parquet")
        evaluated.loc[0, "EncDecPromptSteering_LMJudgeEvaluator"] = 0.0
        evaluated.to_parquet(output / "steering_data.parquet", index=False)
    elif mutation == "boolean_factor":
        block["factor"] = [True]
    elif mutation == "boolean_concept":
        entry["concept_id"] = True
    elif mutation == "float_concept":
        entry["concept_id"] = 1.0
    elif mutation == "nonfinite_results":
        entry["results"]["unexpected"] = float("nan")
    text = json.dumps(entry)
    if mutation == "duplicate_concept":
        text = text.replace('"concept_id": 1', '"concept_id": 0, "concept_id": 1', 1)
    path.write_text(text + "\n")
    with pytest.raises(AxBenchAdvError):
        validate_final_judge_outputs(input_table=table, output_dir=output, **args)


@pytest.mark.parametrize("dataset,concept,aggregate", [
    ("AlpacaEval", 0.0, 0.0),
    ("AlpacaEval", 0.5, 45.0 / 52.0),
    ("AlpacaEval", 2.0, 90.0 / 59.0),
    ("AlpacaEvalSuppress", 0.0, 90.0 / 59.0),
    ("AlpacaEvalSuppress", 0.5, 45.0 / 32.0),
    ("AlpacaEvalSuppress", 2.0, 0.0),
])
def test_final_judge_aggregate_matches_literal_arithmetic(
    tmp_path: Path, dataset: str, concept: float, aggregate: float
) -> None:
    # Keep the oracle literal: for C=.5, I=1.25, F=1.5 the normal
    # harmonic mean is 45/52, and suppression substitutes C'=1.5 =>45/32.
    source = _source(max_concepts=1)
    source["dataset_name"] = dataset
    rationale = json.dumps({
        "concept": f"concept Rating: [[{concept}]]",
        "fluency": "fluency Rating: [[1.5]]",
    })
    emissions = {
        (method, 0, input_id): RawEvaluation(
            f"{method}/{input_id}", concept, 1.5, rationale, 0
        )
        for method in ENCDEC_BASELINE_METHODS
        for input_id in range(10)
    }
    table = build_point_frame(
        source_frame=source, best_factors=FACTORS, emissions=emissions,
        max_concepts=1, method_factors=METHOD_FACTORS_ENCDEC,
    )
    output = tmp_path / "judged"
    _outputs(output, table, emissions, factors=FACTORS, max_concepts=1)
    path = output / "steering.jsonl"
    entry = json.loads(path.read_text())
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    for method, factor in FACTORS.items():
        block = entry["results"]["LMJudgeEvaluator"][method]
        block["raw_aggregated_ratings"] = [
            None if value is None else aggregate for value in block["raw_aggregated_ratings"]
        ]
        block["lm_judge_rating"] = [aggregate]
        evaluated.loc[evaluated["factor"].eq(factor), f"{method}_LMJudgeEvaluator"] = aggregate
    path.write_text(json.dumps(entry) + "\n")
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    validate_final_judge_outputs(
        input_table=table, output_dir=output, best_factors=FACTORS, emissions=emissions,
        max_concepts=1, method_factors=METHOD_FACTORS_ENCDEC,
    )


def test_instruction_rating_cannot_disagree_with_its_retained_completion(encdec_point) -> None:
    table, output, args = encdec_point
    path = output / "steering.jsonl"
    entry = json.loads(path.read_text())
    block = entry["results"]["LMJudgeEvaluator"]["EncDecPromptSteering"]
    assert block["relevance_instruction_completions"][0].endswith("Rating: [[1.25]]")
    block["raw_relevance_instruction_ratings"][0] = 0.75
    # With C=.5, I=.75 and F=1.5 the correctly recomputed aggregate is .75.
    block["raw_aggregated_ratings"][0] = 0.75
    for summary, raw in (
        ("relevance_instruction_ratings", "raw_relevance_instruction_ratings"),
        ("lm_judge_rating", "raw_aggregated_ratings"),
    ):
        present = [value for value in block[raw] if value is not None]
        block[summary] = [sum(present) / len(present)]
    path.write_text(json.dumps(entry) + "\n")
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    evaluated.loc[0, "EncDecPromptSteering_LMJudgeEvaluator_relevance_instruction_ratings"] = 0.75
    evaluated.loc[0, "EncDecPromptSteering_LMJudgeEvaluator"] = 0.75
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    with pytest.raises(AxBenchAdvError, match="judge rating differs from its completion"):
        validate_final_judge_outputs(input_table=table, output_dir=output, **args)


@pytest.mark.parametrize("axis", ["concept", "instruction", "fluency"])
@pytest.mark.parametrize("frame,valid", [
    pytest.param("Reason. Rating: [[{rating:+.2f}]]", True, id="signed_decimal"),
    pytest.param("Reason. Rating: [[{rating:.2e}]]", True, id="exponent"),
    pytest.param("Reason. Rating:\n[[  {rating}  ]]", True, id="whitespace"),
    pytest.param("There is no numeric rating.", False, id="missing"),
    pytest.param("Rating: [[{rating}]] Rating: [[{rating}]]", False, id="duplicate"),
    pytest.param("Rating: [[{rating}]] Rating: invalid", False, id="extra_marker"),
    pytest.param("notRating: [[{rating}]]", False, id="word_prefix"),
    pytest.param("Rating: [[NaN]]", False, id="nonfinite"),
    pytest.param("Rating: [[1e999]]", False, id="overflow"),
    pytest.param("Rating: [[2]]", False, id="wrong_number"),
])
def test_each_final_axis_checks_the_vendor_rating_frame(
    encdec_point, axis: str, frame: str, valid: bool
) -> None:
    table, output, args = encdec_point
    path = output / "steering.jsonl"
    entry = json.loads(path.read_text())
    method = "EncDecPromptSteering"
    block = entry["results"]["LMJudgeEvaluator"][method]
    prefix = {"concept": "relevance_concept", "instruction": "relevance_instruction", "fluency": "fluency"}[axis]
    completion = frame.format(rating=block[f"raw_{prefix}_ratings"][0])
    block[f"{prefix}_completions"][0] = completion
    path.write_text(json.dumps(entry) + "\n")
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    evaluated.loc[0, f"{method}_LMJudgeEvaluator_{prefix}_completions"] = completion
    evaluated.to_parquet(output / "steering_data.parquet", index=False)
    if axis != "instruction":
        # Keep inner/final parity intact so rejection must come from parsing
        # the completion, not from detecting a changed rationale string.
        key = (method, 1, 0)
        raw = args["emissions"][key]
        rationale = json.loads(raw.judge_rationale)
        rationale[axis] = completion
        args["emissions"][key] = replace(raw, judge_rationale=json.dumps(rationale))
    if valid:
        validate_final_judge_outputs(input_table=table, output_dir=output, **args)
    else:
        with pytest.raises(AxBenchAdvError, match="judge rating differs from its completion"):
            validate_final_judge_outputs(input_table=table, output_dir=output, **args)
