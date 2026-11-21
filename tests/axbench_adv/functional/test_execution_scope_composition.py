"""Independent population and numerical oracles for execution-scope composition."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.eval.axbench_adv.merge_slices import merge_slices
from steering.eval.axbench_adv.aggregation import build_aggregate
from steering.eval.axbench_adv.best_point import per_concept_ratings
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import build_point_frame, validate_final_judge_outputs
from steering.eval.axbench_adv.pipeline import run_pipeline
from steering.eval.axbench_adv.types import ALL_METHODS, PLOT_EFFORT_LEVELS
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import (
    FakeFinalJudge,
    FakeOptimizer,
    FakeReflection,
    FakeSelfConditioningRuntime,
    FakeWorker,
    _config,
    _patch_source,
)
from tests.axbench_adv.unit.test_final_judge import FACTORS, _outputs, _raw, _source


@pytest.mark.parametrize("concept_ids", [(-1,), (True,), (1.0,), (1, 1), (3, 1)])
def test_rating_reader_rejects_invalid_expected_identity_even_when_values_compare_equal(
    concept_ids: tuple,
) -> None:
    entries = [
        {"concept_id": int(concept), "results": {"LMJudgeEvaluator": {"DiffMean": {
            "factor": [1.0], "relevance_concept_ratings": [1.0], "fluency_ratings": [1.0],
        }}}}
        for concept in concept_ids
    ]
    with pytest.raises(AxBenchAdvError, match="expected_concept_ids"):
        per_concept_ratings(entries, "DiffMean", 1.0, expected_concept_ids=concept_ids)


@pytest.mark.parametrize("mask", range(1, 1 << len(ALL_METHODS)))
def test_every_method_subset_preserves_sparse_grid_and_exact_concept_statistics(
    tmp_path: Path, mask: int,
) -> None:
    methods = tuple(method for index, method in enumerate(ALL_METHODS) if mask & (1 << index))
    all_factors = {**FACTORS, "NLSteering": 2.0}
    factors = {method: all_factors[method] for method in methods}
    concepts = (1, 3)
    emissions = {
        (method, concept, input_id): _raw(method, concept, input_id)
        for method in methods for concept in concepts for input_id in range(10)
    }
    table = build_point_frame(
        source_frame=_source(5), best_factors=factors, emissions=emissions,
        max_concepts=5, concept_ids=concepts,
    )
    assert table.schema.names[-len(methods):] == [f"{method}_steered_generation" for method in methods]
    rows = table.to_pylist()
    assert len(rows) == len(concepts) * 10 * len(set(factors.values()))
    for row in rows:
        assert row["concept_id"] in concepts
        for method in methods:
            expected = (
                emissions[(method, row["concept_id"], row["input_id"])].generation
                if row["factor"] == factors[method] else None
            )
            assert row[f"{method}_steered_generation"] == expected
    judged = tmp_path / "judged"
    _outputs(judged, table, emissions, factors=factors, concept_ids=concepts)
    validate_final_judge_outputs(
        input_table=table, output_dir=judged, best_factors=factors,
        emissions=emissions, max_concepts=5, concept_ids=concepts,
    )
    aggregate = build_aggregate(
        best_factors=factors,
        judged_points={(method, level): judged / "steering.jsonl" for method in methods for level in PLOT_EFFORT_LEVELS},
        expected_n_concepts=2, expected_concept_ids=concepts,
        output_path=tmp_path / "aggregate.parquet",
    )
    assert aggregate.num_rows == len(methods) * 5
    for row in aggregate.to_pylist():
        assert row["n_concepts"] == 2
        assert row["concept_score"] == 0.75
        assert row["concept_score_sem"] == 0.25
        assert row["fluency_score"] == pytest.approx(1.525)
        assert row["fluency_score_sem"] == 0.0


@pytest.mark.parametrize("methods", [("DiffMean",), ("SelfConditioning",), ("DiffMean", "SelfConditioning")])
def test_completed_concept_slices_merge_to_exact_unsliced_pipeline_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, methods: tuple[str, ...],
) -> None:
    monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(reversed(methods)))
    roots = []
    for label, lo, hi in (("full", 0, 2), ("low", 0, 1), ("high", 1, 2)):
        output = tmp_path / label
        _patch_source(monkeypatch, output)
        monkeypatch.setenv("AXBENCH_ADV_CONCEPT_LO", str(lo))
        monkeypatch.setenv("AXBENCH_ADV_CONCEPT_HI", str(hi))
        FakeWorker.instances = []
        FakeSelfConditioningRuntime.instances = []
        manifest = run_pipeline(
            _config(output), reflection_lm=FakeReflection(),
            worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
            self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
            optimizer=FakeOptimizer(), final_judge=FakeFinalJudge(),
        )
        assert json.loads(manifest.read_text())["methods"] == list(methods)
        assert bool(FakeWorker.instances) == ("DiffMean" in methods)
        assert bool(FakeSelfConditioningRuntime.instances) == ("SelfConditioning" in methods)
        roots.append(output)
    merged = merge_slices(roots[1:], tmp_path / "merged")
    assert pq.read_table(merged.parent / "aggregate.parquet").equals(
        pq.read_table(roots[0] / "aggregate.parquet")
    )
