"""Exact predecessor-oracle tests for two-way best-factor selection."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest

from steering.eval.axbench_adv.best_point import (
    BestPoint,
    concept_fluency_point,
    load_steering_entries,
    per_concept_ratings,
    point_at_factor,
    select_best_point,
    select_best_points,
    write_best_points,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS, ADV_METHOD_FACTORS
from steering.eval.axbench_adv.types import METHODS
from steering.eval.axbench_base.aggregation import hmean


def _block(
    factors: tuple[float, ...], concept: Sequence[object], fluency: Sequence[object]
) -> dict[str, object]:
    return {
        "factor": list(factors),
        "relevance_concept_ratings": list(concept),
        "fluency_ratings": list(fluency),
    }


def _methods(*, concept: float = 1.0, fluency: float = 1.0) -> dict[str, dict[str, object]]:
    return {
        method: _block(
            factors,
            [concept] * len(factors),
            [fluency] * len(factors),
        )
        for method, factors in ADV_METHOD_FACTORS.items()
    }


def _write(path: Path, rows: list[dict[str, dict[str, object]]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"concept_id": concept_id, "results": {"LMJudgeEvaluator": methods}}) + "\n"
            for concept_id, methods in enumerate(rows)
        ),
        encoding="utf-8",
    )


def test_predecessor_argmax_tie_and_fractional_macro_oracles(tmp_path: Path) -> None:
    rows = [_methods(concept=1.0), _methods(concept=2.0), _methods(concept=2.0)]
    one = ADV_FULL_FACTORS.index(1.0)
    one_five = ADV_FULL_FACTORS.index(1.5)
    for methods in rows:
        block = methods["DiffMean"]
        block["relevance_concept_ratings"][one] = 2.0  # ty: ignore[invalid-assignment]
        block["fluency_ratings"][one] = 2.0  # ty: ignore[invalid-assignment]
        block["relevance_concept_ratings"][one_five] = 2.0  # ty: ignore[invalid-assignment]
        block["fluency_ratings"][one_five] = 2.0  # ty: ignore[invalid-assignment]
    path = tmp_path / "steering.jsonl"
    _write(path, rows)

    best = select_best_point(path, "DiffMean")
    assert best == BestPoint("DiffMean", 1.0, 2.0, 2.0, 2.0)

    fractional_rows = [_methods(concept=value, fluency=2.0) for value in (1.0, 2.0, 2.0)]
    _write(path, fractional_rows)
    best = select_best_point(path, "DiffMean")
    expected_concept = 5.0 / 3.0
    assert best.factor == ADV_FULL_FACTORS[0]
    assert best.concept_score == pytest.approx(expected_concept)
    assert best.hm2 == pytest.approx(hmean(expected_concept, 2.0))
    assert hmean(0.4, 1.6) == pytest.approx(0.64)


def test_all_six_best_points_match_literal_synthetic_oracle(tmp_path: Path) -> None:
    factors = {
        "DiffMean": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
        "LinearProbe": (
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ),
        "LoRA": (1.0,),
        "PromptSteering": (1.0,),
        "SelfConditioning": ADV_FULL_FACTORS,
        "SelfConditioningMasked": ADV_FULL_FACTORS,
    }
    targets = {
        "DiffMean": 1.0,
        "LinearProbe": 3.0,
        "LoRA": 1.0,
        "PromptSteering": 1.0,
        "SelfConditioning": 6.0,
        "SelfConditioningMasked": 5.0,
    }
    scores = {
        "DiffMean": ((1.0, 2.0), (2.0, 2.0)),
        "LinearProbe": ((1.0, 1.0), (1.0, 2.0)),
        "LoRA": ((0.5, 2.0), (1.5, 2.0)),
        "PromptSteering": ((2.0, 0.5), (2.0, 1.5)),
        "SelfConditioning": ((1.5, 0.5), (1.5, 1.5)),
        "SelfConditioningMasked": ((1.0, 1.5), (2.0, 1.5)),
    }
    rows: list[dict[str, dict[str, object]]] = []
    for concept_id in range(2):
        methods: dict[str, dict[str, object]] = {}
        for method, grid in factors.items():
            concept = [0.0] * len(grid)
            fluency = [0.0] * len(grid)
            index = grid.index(targets[method])
            concept[index], fluency[index] = scores[method][concept_id]
            methods[method] = _block(grid, concept, fluency)
        rows.append(methods)
    path = tmp_path / "steering.jsonl"
    _write(path, rows)

    assert select_best_points(path) == {
        "DiffMean": BestPoint("DiffMean", 1.0, 1.5, 2.0, 1.7142857142857142),
        "LinearProbe": BestPoint("LinearProbe", 3.0, 1.0, 1.5, 1.2),
        "LoRA": BestPoint("LoRA", 1.0, 1.0, 2.0, 1.3333333333333333),
        "PromptSteering": BestPoint("PromptSteering", 1.0, 2.0, 1.0, 1.3333333333333333),
        "SelfConditioning": BestPoint("SelfConditioning", 6.0, 1.5, 1.0, 1.2),
        "SelfConditioningMasked": BestPoint(
            "SelfConditioningMasked", 5.0, 1.5, 1.5, 1.5
        ),
    }


def test_ratified_c2_best_points_when_exact_source_is_available() -> None:
    raw_path = os.environ.get("AXBENCH_ADV_C2_STEERING_JSONL")
    if raw_path is None:
        pytest.skip("set AXBENCH_ADV_C2_STEERING_JSONL to run the exact c2 oracle")
    path = Path(raw_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "3dce8f1383ce3ff034dbad4bebbbc36fa4e640ee7684bbd2d69979233cd44ba9"
    )
    legacy = {
        "DiffMean": BestPoint("DiffMean", 1.0, 0.55, 1.15, 0.7441176470588234),
        "LinearProbe": BestPoint("LinearProbe", 0.6, 0.4, 1.9, 0.6608695652173914),
        "LoRA": BestPoint("LoRA", 1.0, 1.05, 1.75, 1.3125000000000002),
        "PromptSteering": BestPoint("PromptSteering", 1.0, 1.0, 1.4, 1.1666666666666667),
        "SelfConditioning": BestPoint(
            "SelfConditioning",
            5.0,
            1.4,
            0.7,
            0.9333333333333333,
        ),
    }
    assert {
        method: select_best_point(path, method) for method in legacy
    } == legacy


def test_zero_bottom_single_factor_macro_and_explicit_override(tmp_path: Path) -> None:
    methods0 = _methods(concept=1.0, fluency=0.0)
    methods1 = _methods(concept=2.0, fluency=0.0)
    methods0["DiffMean"]["fluency_ratings"][0] = 1.0  # ty: ignore[invalid-assignment]
    methods1["DiffMean"]["fluency_ratings"][0] = 1.0  # ty: ignore[invalid-assignment]
    path = tmp_path / "steering.jsonl"
    _write(path, [methods0, methods1])

    assert select_best_point(path, "DiffMean").factor == 0.0
    assert select_best_point(path, "LoRA").factor == 1.0
    concept, fluency = per_concept_ratings(path, "DiffMean", 0.0)
    assert concept == [1.0, 2.0]
    assert fluency == [1.0, 1.0]
    assert concept_fluency_point(path, "DiffMean", 0.0) == (1.5, 1.0)
    assert point_at_factor(path, "DiffMean", 0.0) == BestPoint(
        "DiffMean", 0.0, 1.5, 1.0, hmean(1.5, 1.0)
    )


def test_select_and_write_exactly_the_six_methods(tmp_path: Path) -> None:
    path = tmp_path / "steering.jsonl"
    _write(path, [_methods(), _methods()])
    points = select_best_points(load_steering_entries(path))
    assert tuple(points) == METHODS
    json_path, csv_path = write_best_points(points, tmp_path / "selected")
    assert set(json.loads(json_path.read_text())) == set(METHODS)
    with csv_path.open() as handle:
        assert {row["method"] for row in csv.DictReader(handle)} == set(METHODS)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "not found"),
        ("bad_json", "invalid JSON"),
        ("duplicate", "invalid JSON"),
        ("nonfinite", "invalid JSON"),
        ("bad_shape", "contain only"),
        ("empty", "is empty"),
        ("order", "exactly once"),
        ("bool_id", "exactly once"),
    ],
)
def test_load_rejects_malformed_jsonl(tmp_path: Path, mutation: str, match: str) -> None:
    path = tmp_path / "steering.jsonl"
    if mutation == "missing":
        pass
    elif mutation == "bad_json":
        path.write_text("{")
    elif mutation == "duplicate":
        path.write_text('{"concept_id": 99, "concept_id": 0, "results": {}}\n')
    elif mutation == "nonfinite":
        path.write_text('{"concept_id": 0, "results": {"score": NaN}}\n')
    elif mutation == "bad_shape":
        path.write_text('{"concept_id": 0}\n')
    elif mutation == "empty":
        path.write_text("\n")
    elif mutation == "order":
        path.write_text(json.dumps({"concept_id": 1, "results": {}}) + "\n")
    else:
        path.write_text(json.dumps({"concept_id": False, "results": {}}) + "\n")
    with pytest.raises(AxBenchAdvError, match=match):
        load_steering_entries(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unknown", "unknown attacked method"),
        ("off_grid", "not on DiffMean"),
        ("no_results", "no results mapping"),
        ("no_evaluator", "no LMJudgeEvaluator"),
        ("no_method", "no 'DiffMean' judge block"),
        ("factor_not_list", "factor must be a list"),
        ("length", "mismatched lengths"),
        ("factor_text", "nonnumeric value"),
        ("factor_nan", "nonfinite value"),
        ("factor_duplicate", "exactly once"),
        ("request_factor_bool", "must be numeric"),
        ("request_factor_nan", "must be finite"),
        ("concept_order", "exactly once"),
        ("concept_missing", "carry a concept_id"),
        ("rating_text", "nonnumeric rating"),
        ("rating_range", r"outside \[0, 2\]"),
    ],
)
def test_selection_rejects_malformed_blocks(mutation: str, match: str) -> None:
    entries: list[dict[str, object]] = [
        {"concept_id": 0, "results": {"LMJudgeEvaluator": _methods()}}
    ]
    method = "DiffMean"
    factor = 1.0
    block = entries[0]["results"]["LMJudgeEvaluator"][method]  # type: ignore[index]
    if mutation == "unknown":
        method = "Unknown"
    elif mutation == "off_grid":
        factor = 99.0
    elif mutation == "no_results":
        entries[0]["results"] = None
    elif mutation == "no_evaluator":
        entries[0]["results"] = {}
    elif mutation == "no_method":
        del entries[0]["results"]["LMJudgeEvaluator"][method]  # type: ignore[index]
    elif mutation == "factor_not_list":
        block["factor"] = "bad"
    elif mutation == "length":
        block["fluency_ratings"] = []
    elif mutation == "factor_text":
        block["factor"][0] = "bad"  # ty: ignore[invalid-assignment]
    elif mutation == "factor_nan":
        block["factor"][0] = float("nan")  # ty: ignore[invalid-assignment]
    elif mutation == "factor_duplicate":
        block["factor"][0] = 1.0  # ty: ignore[invalid-assignment]
    elif mutation == "request_factor_bool":
        factor = True
    elif mutation == "request_factor_nan":
        factor = float("nan")
    elif mutation == "concept_order":
        entries[0]["concept_id"] = 1
    elif mutation == "concept_missing":
        del entries[0]["concept_id"]
    elif mutation == "rating_text":
        block["relevance_concept_ratings"][5] = "bad"  # ty: ignore[invalid-assignment]
    else:
        block["fluency_ratings"][5] = 3.0  # ty: ignore[invalid-assignment]
    with pytest.raises(AxBenchAdvError, match=match):
        concept_fluency_point(entries, method, factor)


def test_selection_rejects_empty_or_duplicate_requests_and_empty_write(tmp_path: Path) -> None:
    entries = [{"concept_id": 0, "results": {"LMJudgeEvaluator": _methods()}}]
    with pytest.raises(AxBenchAdvError, match="unknown attacked method"):
        select_best_point(entries, "Unknown")
    with pytest.raises(AxBenchAdvError, match="entries must not be empty"):
        per_concept_ratings([], "LoRA", 1.0)
    with pytest.raises(AxBenchAdvError, match="nonempty and unique"):
        select_best_points(entries, methods=[])
    with pytest.raises(AxBenchAdvError, match="nonempty and unique"):
        select_best_points(entries, methods=["LoRA", "LoRA"])
    with pytest.raises(AxBenchAdvError, match="must not be empty"):
        write_best_points({}, tmp_path)
