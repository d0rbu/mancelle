"""GCG pressure boundaries that must remain disjoint from legacy campaigns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

from scripts.eval.axbench_adv.merge_slices import _validate_search_receipts
from scripts.eval.axbench_adv.pressures import (
    COUNTER_PRESSURES,
    LEGACY_PRESSURES,
    Pressure,
    parse_pressures,
)
from steering.eval.axbench_adv import pipeline as pipeline_module
from steering.eval.axbench_adv.aggregation import (
    GCG_AGGREGATE_SCHEMA,
    build_gcg_aggregate,
    validate_gcg_aggregate,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample, RawEvaluation
from steering.eval.axbench_adv.plotting import method_trajectory, plot_concept_vs_fluency
from steering.eval.axbench_adv.types import PLOT_EFFORT_LEVELS
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_histograms import generate_histogram_reports
from steering.eval.axbench_upstream.data import ConceptTrainingPairs


def _judged(method: str, factor: float, token_count: int) -> list[dict]:
    return [
        {
            "concept_id": concept_id,
            "results": {
                "LMJudgeEvaluator": {
                    method: {
                        "factor": [factor],
                        "relevance_concept_ratings": [1.5 - 0.1 * token_count],
                        "fluency_ratings": [1.0 + 0.1 * concept_id],
                        "relevance_instruction_ratings": [1.25 + 0.1 * concept_id],
                    }
                }
            },
        }
        for concept_id in range(2)
    ]


def test_pressure_families_freeze_issue1673_and_keep_gcg_disjoint() -> None:
    assert LEGACY_PRESSURES == (
        Pressure.GEPA,
        Pressure.LINEAR_PROBE_COUNTER,
        Pressure.DIFFMEAN_COUNTER,
    )
    assert COUNTER_PRESSURES == (
        Pressure.LINEAR_PROBE_COUNTER,
        Pressure.DIFFMEAN_COUNTER,
    )
    assert Pressure.GCG not in LEGACY_PRESSURES
    assert Pressure.GCG not in COUNTER_PRESSURES
    assert parse_pressures(["gcg", "gepa"], environ={}) == (
        Pressure.GEPA,
        Pressure.GCG,
    )


def test_gcg_aggregate_has_integer_token_axis_and_three_judge_metrics(
    tmp_path: Path,
) -> None:
    methods = {"DiffMean": 1.0, "LinearProbe": 1.0}
    counts = (0, 2)
    table = build_gcg_aggregate(
        best_factors=methods,
        judged_points={
            (method, count): _judged(method, factor, count)
            for method, factor in methods.items()
            for count in counts
        },
        token_counts=counts,
        concept_ids=(0, 1),
        output_path=tmp_path / "gcg" / "aggregate.parquet",
    )
    assert table.schema.equals(GCG_AGGREGATE_SCHEMA)
    assert table["token_count"].type == pa.int32()
    first = table.to_pylist()[0]
    assert first["instruction_score"] == pytest.approx(1.3)
    assert first["instruction_score_sem"] == pytest.approx(0.05)
    assert [point.proposal_count for point in method_trajectory(table, "DiffMean")] == [0, 2]
    paths = plot_concept_vs_fluency(
        table,
        tmp_path / "gcg" / "plots",
        pressure=Pressure.GCG,
    )
    assert all(path.is_file() for path in paths)


def test_gcg_aggregate_rejects_missing_zero_and_instruction_mutations(
    tmp_path: Path,
) -> None:
    with pytest.raises(AxBenchAdvError, match="start at zero"):
        build_gcg_aggregate(
            best_factors={"DiffMean": 1.0},
            judged_points={("DiffMean", 2): _judged("DiffMean", 1.0, 2)},
            token_counts=(2,),
            concept_ids=(0, 1),
            output_path=tmp_path / "bad.parquet",
        )
    rows = build_gcg_aggregate(
        best_factors={"DiffMean": 1.0},
        judged_points={("DiffMean", 0): _judged("DiffMean", 1.0, 0)},
        token_counts=(0,),
        concept_ids=(0, 1),
        output_path=tmp_path / "good.parquet",
    ).to_pylist()
    rows[0]["instruction_score"] = 3.0
    with pytest.raises(AxBenchAdvError, match="instruction_score"):
        validate_gcg_aggregate(pa.Table.from_pylist(rows, schema=GCG_AGGREGATE_SCHEMA))


def _raw(input_id: int, label: str) -> RawEvaluation:
    return RawEvaluation(
        generation=f"{label}-{input_id}",
        concept_rating=1.0,
        fluency_rating=1.5,
        judge_rationale="{}",
        paid_judge_cache_misses=0,
    )


def test_execute_cell_reuses_control_and_accepts_only_positive_gcg_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_execute(**kwargs: object) -> dict[int, list[tuple[int, RawEvaluation]]]:
        seen.update(kwargs)
        bound = kwargs["config"]
        assert isinstance(bound, AxBenchAdvGcgConfig)
        assert bound.search.master_seed == 17
        return {2: [(input_id, _raw(input_id, "gcg")) for input_id in range(10)]}

    class Runtime:
        def evaluate(self, suffix: str, input_ids: Sequence[int]) -> tuple[RawEvaluation, ...]:
            assert suffix == ""
            return tuple(_raw(input_id, "control") for input_id in input_ids)

        def evaluate_counter(
            self, _counter: object, _magnitude: float, _input_ids: Sequence[int]
        ) -> tuple[RawEvaluation, ...]:
            raise AssertionError("GCG must not enter the additive-counter runtime")

    monkeypatch.setattr(pipeline_module, "execute_gcg_cell", fake_execute)
    gcg = AxBenchAdvGcgConfig(search=replace(FasterGcgConfig(), optimized_token_counts=(0, 2)))
    config = SimpleNamespace(
        pressures=(Pressure.GCG,),
        gcg=gcg,
        master_seed=17,
    )
    emissions = {level: {} for level in PLOT_EFFORT_LEVELS}
    gcg_emissions = {2: {}}
    totals = {
        "control_generation_calls": 0,
        "control_paid_judge_cache_misses": 0,
    }
    training = ConceptTrainingPairs(
        concept_id=0,
        concept="topic",
        genre="genre",
        positive=(("positive prompt", "positive response"),),
        negative=(("negative prompt", "negative response"),),
    )
    examples = [OptimizationExample(input_id=i, user_message=f"eval {i}") for i in range(10)]
    pipeline_module._execute_cell(
        config=config,  # ty: ignore[invalid-argument-type]
        runtime=Runtime(),  # ty: ignore[invalid-argument-type]
        reflection_lm=None,
        optimizer=lambda **_kwargs: None,  # ty: ignore[invalid-argument-type]
        output_dir=tmp_path,
        method="DiffMean",
        concept_id=0,
        concept="topic",
        factor=1.0,
        optimization_examples=examples,
        emissions=emissions,
        totals=totals,
        counter_emissions={},
        gcg_emissions=gcg_emissions,
        directions={},
        training_pairs=training,
    )
    assert len(emissions[0]) == 10
    assert len(gcg_emissions[2]) == 10
    assert totals["control_generation_calls"] == 10
    assert seen["training_pairs"] is training
    assert seen["evaluation_prompts"] == tuple(example.user_message for example in examples)


def _write_histogram_point(directory: Path, *, shift: float) -> None:
    rows: list[dict[str, object]] = []
    entries = []
    for concept_id in range(2):
        concept_scores: list[float] = []
        fluency_scores: list[float] = []
        instruction_scores: list[float] = []
        for input_id in range(10):
            concept = 1.0 + shift
            fluency = 1.5
            instruction = 1.25
            concept_scores.append(concept)
            fluency_scores.append(fluency)
            instruction_scores.append(instruction)
            rows.append(
                {
                    "concept_id": concept_id,
                    "input_id": input_id,
                    "dataset_name": "fixture",
                    "input_concept": f"concept {concept_id}",
                    "original_prompt": f"prompt {concept_id}/{input_id}",
                    "factor": 1.0,
                    "DiffMean_LMJudgeEvaluator_relevance_concept_ratings": concept,
                    "DiffMean_LMJudgeEvaluator_fluency_ratings": fluency,
                    "DiffMean_LMJudgeEvaluator_relevance_instruction_ratings": instruction,
                }
            )
        entries.append(
            {
                "concept_id": concept_id,
                "results": {
                    "LMJudgeEvaluator": {
                        "DiffMean": {
                            "factor": [1.0],
                            "relevance_concept_ratings": [sum(concept_scores) / 10],
                            "fluency_ratings": [sum(fluency_scores) / 10],
                            "relevance_instruction_ratings": [sum(instruction_scores) / 10],
                            "raw_relevance_concept_ratings": concept_scores,
                            "raw_fluency_ratings": fluency_scores,
                            "raw_relevance_instruction_ratings": instruction_scores,
                        }
                    }
                },
            }
        )
    directory.mkdir(parents=True)
    (directory / "steering.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    pd.DataFrame(rows).to_parquet(directory / "steering_data.parquet", index=False)


def test_histogram_explicit_gcg_pressure_uses_k_grid_and_shared_zero(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_histogram_point(root / "points" / "p0" / "judged", shift=0.0)
    _write_histogram_point(root / "gcg" / "points" / "k2" / "judged", shift=0.1)
    manifest = {
        "schema_version": 2,
        "methods": ["DiffMean"],
        "max_concepts": 2,
        "concept_ids": [0, 1],
        "best_factors": {"DiffMean": 1.0},
        "pressures": ["gcg"],
        "gcg": {"search": {"optimized_token_counts": [0, 2]}},
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = generate_histogram_reports(
        input_dir=root,
        output_dir=tmp_path / "histograms",
        pipeline="adv",
        pressure="gcg",
        method_factors={},
    )
    metadata = json.loads(result.read_text(encoding="utf-8"))
    assert metadata["pressure"] == "gcg"
    assert set(metadata["reports"]) == {"k0", "k2"}
    assert (result.parent / "k0" / "score_histograms.png").is_file()
    assert (result.parent / "k2" / "score_histograms.png").is_file()


def test_merge_authenticates_gcg_grid_config_factor_training_and_zero(tmp_path: Path) -> None:
    root = tmp_path / "slice"
    identity = "a" * 64
    effective = json.loads(
        json.dumps(
            AxBenchAdvGcgConfig(
                device_batch_size=1,
                search=replace(FasterGcgConfig(), optimized_token_counts=(0, 2)),
            ).effective_identity()
        )
    )
    files: dict[str, str] = {}
    for token_count in (0, 2):
        document = {
            "schema_version": 1,
            "objective": "response_margin",
            "effective_config": effective,
            "training_pair_identity": identity,
            "method": "DiffMean",
            "concept_id": 0,
            "concept": "topic",
            "factor": 1.0,
            "token_count": token_count,
            "suffix_token_ids": [] if token_count == 0 else [3, 4],
            "suffix": "" if token_count == 0 else "suffix",
            "diagnostics": (
                {"control": True, "accepted_steps": 0, "visited_sequences": 0}
                if token_count == 0
                else {"control": False, "accepted_steps": 1, "visited_sequences": 2}
            ),
        }
        if token_count > 0:
            document.update(
                {
                    "model_tokenizer_identity": {
                        "concept_id": 0,
                        "factor": 1.0,
                        "layer": 1,
                        "method": "DiffMean",
                        "model_name": "fixture/model",
                        "model_revision": "revision",
                        "rank_sentinel_id": 0,
                        "tokenizer_class": "FixtureTokenizer",
                        "vocab_size": 128,
                    },
                    "layouts_sha256": "b" * 64,
                    "direction_calibration": None,
                    "objective_value": 1.0,
                }
            )
        relative = Path("gcg") / "DiffMean" / "c0" / f"k{token_count}" / "search.json"
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(document), encoding="utf-8")
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "max_concepts": 1,
        "master_seed": 0,
        "gcg": effective,
        "gcg_training_source": {
            "train_data": {
                "path": "/source/train_data.parquet",
                "sha256": "c" * 64,
                "byte_length": 10,
            },
            "metadata": {
                "path": "/source/metadata.jsonl",
                "sha256": "d" * 64,
                "byte_length": 10,
            },
            "pair_sha256": {"0": identity},
        },
    }
    required = _validate_search_receipts(
        pressure=Pressure.GCG,
        root=root,
        manifest=manifest,
        files=files,
        methods=("DiffMean",),
        concept_ids=(0,),
        factors={"DiffMean": 1.0},
        token_counts=(0, 2),
    )
    assert len(required) == 2
    bad = dict(manifest)
    bad["gcg_training_source"] = {
        **manifest["gcg_training_source"],
        "pair_sha256": {"0": "e" * 64},
    }
    with pytest.raises(AxBenchAdvError, match="receipt identity differs"):
        _validate_search_receipts(
            pressure=Pressure.GCG,
            root=root,
            manifest=bad,
            files=files,
            methods=("DiffMean",),
            concept_ids=(0,),
            factors={"DiffMean": 1.0},
            token_counts=(0, 2),
        )
