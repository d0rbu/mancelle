"""Whole-pipeline fake slice for the exact 6x2x(0,1,2,4,8) contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nanochat.atomic_io import atomic_write_text
from steering.eval.axbench_adv import pipeline as pipeline_module
from steering.eval.axbench_adv.aggregation import validate_aggregate
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import FinalJudgeResult
from steering.eval.axbench_adv.fitness import evaluate_fitness
from steering.eval.axbench_adv.gepa_optimization import (
    CallAccounting,
    RawEvaluation,
    ScoredEvaluation,
    SuffixOptimizationResult,
)
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS, ADV_METHOD_FACTORS
from steering.eval.axbench_adv.pipeline import AxBenchAdvRunConfig, run_pipeline
from steering.eval.axbench_adv.source_contract import (
    SourceExpectations,
    SourcePaths,
    ValidatedSources,
)
from steering.eval.axbench_adv.types import (
    METHODS,
    PLOT_EFFORT_LEVELS,
    PROPOSAL_COUNTS,
    QWEN_METHODS,
)
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_upstream.methods import SELF_CONDITIONING_MASKED_POLICY

BEST_FACTORS = {
    "DiffMean": 1.0,
    "LinearProbe": 3.0,
    "LoRA": 1.0,
    "PromptSteering": 1.0,
    "SelfConditioning": 6.0,
    "SelfConditioningMasked": 5.0,
}


def _frame(*, aux: bool) -> pd.DataFrame:
    factors = (1.0,) if aux else ADV_FULL_FACTORS
    return pd.DataFrame(
        [
            {
                "dataset_name": "AlpacaEval",
                "concept_id": concept_id,
                "input_concept": f"concept-{concept_id}",
                "input_id": input_id,
                "factor": float(factor),
                "original_prompt": f"original {concept_id}/{input_id}",
                "input": f"plain prompt {concept_id}/{input_id}",
                "steered_input": f"steered prompt {concept_id}/{input_id}",
            }
            for concept_id in range(2)
            for input_id in range(10)
            for factor in factors
        ]
    )


def _steering_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for concept_id in range(2):
        methods: dict[str, dict[str, list[float]]] = {}
        for method in METHODS:
            factors = [float(value) for value in ADV_METHOD_FACTORS[method]]
            concept = [0.5] * len(factors)
            fluency = [0.5] * len(factors)
            best = factors.index(BEST_FACTORS[method])
            concept[best] = 1.5
            fluency[best] = 1.75
            methods[method] = {
                "factor": factors,
                "relevance_concept_ratings": concept,
                "fluency_ratings": fluency,
            }
        entries.append({"concept_id": concept_id, "results": {"LMJudgeEvaluator": methods}})
    return entries


def _validated(output_dir: Path) -> ValidatedSources:
    paths = SourcePaths(
        prepared_dir=Path("/source/prepared"),
        main_parquet=Path("/source/main/inference/steering_data.parquet"),
        aux_parquet=Path("/source/aux/inference/steering_data.parquet"),
        evaluated_jsonl=Path("/source/evaluated/steering.jsonl"),
        project_parquet=Path("/source/project/inference/steering_data.parquet"),
        target_meta=Path("/source/checkpoint/meta_054600.json"),
        feature_statistics=Path("/source/feature_statistics.safetensors"),
    )
    expectations = SourceExpectations(
        max_concepts=2,
        repository_git_commit="1" * 40,
        dataset_revision="2" * 40,
        target_model="fake/Qwen",
        seed=42,
        temperature=1.0,
        output_length=128,
        package_lock_sha256="3" * 64,
        vendored_axbench_upstream_commit="4" * 40,
        project_repository_git_commit="a" * 40,
        project_package_lock_sha256="b" * 64,
        config_name_prefix="qwen2_5_3b",
        layer=28,
    )
    return ValidatedSources(
        paths=paths,
        expectations=expectations,
        output_dir=output_dir.resolve(),
        hashes={
            "main_parquet": "5" * 64,
            "aux_parquet": "6" * 64,
            "evaluated_jsonl": "7" * 64,
            "project_parquet": "8" * 64,
            "project_manifest": "9" * 64,
            "target_receipt": "a" * 64,
        },
        main_frame=_frame(aux=False),
        aux_frame=_frame(aux=True),
        project_frame=_frame(aux=False),
        steering_entries=_steering_entries(),
        project_manifest={
            "package_lock_sha256": "b" * 64,
            "repository_git_commit": "1" * 40,
            "target": {"model_tag": "fake", "step": 54_600},
            "methods": {
                "SelfConditioningMasked": {
                    "policy": SELF_CONDITIONING_MASKED_POLICY,
                }
            },
        },
        method_setup=object(),  # ty: ignore[invalid-argument-type]
        training_texts={
            concept_id: SimpleNamespace(concept=f"concept-{concept_id}")
            for concept_id in range(2)
        },  # ty: ignore[invalid-argument-type]
        runtime_hashes={
            "target_meta": "b" * 64,
            "target_model": "c" * 64,
            "feature_statistics": "d" * 64,
        },
        runtime_byte_lengths={
            "target_meta": 1,
            "target_model": 2,
            "feature_statistics": 3,
        },
        extraction_batch_size=8,
        target_meta={
            "model_config": {"n_embd": 2},
            "remote_run_fingerprint": "e" * 64,
            "step": 54_600,
            "user_config": {"model_tag": "fake"},
        },
    )


def _config(output_dir: Path) -> AxBenchAdvRunConfig:
    validated = _validated(output_dir)
    return AxBenchAdvRunConfig(
        source_paths=validated.paths,
        source_expectations=validated.expectations,
        source_hashes=validated.hashes,
        output_dir=output_dir,
        judge_cache=output_dir.parent / "judge.sqlite",
        reflection_cache=output_dir.parent / "reflection.sqlite",
        reflection=LLMClientConfig(provider="openrouter", model="fake/reflection"),
        worker_max_prompt_tokens=2048,
    )


def _raw(method: str, concept_id: int, input_id: int, suffix: str) -> RawEvaluation:
    concept = 1.5 if not suffix else max(0.0, 1.0 - 0.1 * int(suffix.removeprefix("suffix-p")))
    fluency = 1.75
    return RawEvaluation(
        generation=f"{method} c{concept_id} i{input_id} suffix={suffix or '<empty>'}",
        concept_rating=concept,
        fluency_rating=fluency,
        judge_rationale=json.dumps(
            {
                "concept": f"concept Rating: [[{concept}]]",
                "fluency": "fluency Rating: [[1.75]]",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        paid_judge_cache_misses=0,
    )


class FakeWorker:
    instances: list[FakeWorker] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.method = kwargs["init_payload"]["method"]
        self.concept_id = kwargs["init_payload"]["concept_id"]
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.closed = False
        self.instances.append(self)

    def evaluate(self, suffix: str, input_ids: list[int]) -> tuple[RawEvaluation, ...]:
        self.calls.append((suffix, tuple(input_ids)))
        return tuple(_raw(self.method, self.concept_id, input_id, suffix) for input_id in input_ids)

    def __enter__(self) -> FakeWorker:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.closed = True


class FakeSelfConditioningCell:
    def __init__(self, method: str, concept_id: int) -> None:
        self.method = method
        self.concept_id = concept_id
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.closed = False

    def evaluate(self, suffix: str, input_ids: list[int]) -> tuple[RawEvaluation, ...]:
        self.calls.append((suffix, tuple(input_ids)))
        return tuple(
            _raw(self.method, self.concept_id, input_id, suffix) for input_id in input_ids
        )

    def __enter__(self) -> FakeSelfConditioningCell:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.closed = True


class FakeSelfConditioningRuntime:
    instances: list[FakeSelfConditioningRuntime] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.cells: list[FakeSelfConditioningCell] = []
        self.closed = False
        self.instances.append(self)

    def bind_cell(
        self, *, method: str, concept_id: int, **_kwargs: Any
    ) -> FakeSelfConditioningCell:
        cell = FakeSelfConditioningCell(method, concept_id)
        self.cells.append(cell)
        return cell

    def __enter__(self) -> FakeSelfConditioningRuntime:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.closed = True


class FakeReflection:
    @property
    def paid_cache_misses(self) -> int:
        return 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        del prompt
        raise AssertionError("the injected optimizer must own reflection in this slice")


class FakeOptimizer:
    calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> SuffixOptimizationResult:
        self.calls.append(kwargs)
        proposal_count = kwargs["proposal_count"]
        suffix = f"suffix-p{proposal_count}"
        kwargs["run_dir"].mkdir(parents=True)
        evaluations: list[ScoredEvaluation] = []
        for example in kwargs["examples"]:
            raw = kwargs["runtime"](suffix, example.input_id)
            evaluations.append(
                ScoredEvaluation(
                    example=example,
                    raw=raw,
                    fitness=evaluate_fitness(
                        concept_rating=raw.concept_rating,
                        fluency_rating=raw.fluency_rating,
                        output=raw.generation,
                    ),
                )
            )
        fitness = sum(item.fitness.score for item in evaluations) / len(evaluations)
        return SuffixOptimizationResult(
            suffix=suffix,
            best_fitness=fitness,
            evaluations=tuple(evaluations),
            accounting=CallAccounting(
                gepa_nominal_metric_calls=proposal_count,
                actual_evaluator_calls=10,
                actual_generation_calls=10,
                paid_judge_cache_misses=0,
                paid_reflection_cache_misses=0,
            ),
        )


class FakeFinalJudge:
    calls: list[dict[str, Any]] = []

    def __init__(self, fail_level: int | None = None) -> None:
        self.fail_level = fail_level

    def __call__(self, **kwargs: Any) -> FinalJudgeResult:
        self.calls.append(kwargs)
        output_dir = Path(kwargs["output_dir"])
        level = int(output_dir.parent.name.removeprefix("p"))
        if level == self.fail_level:
            raise RuntimeError("injected final-judge failure")
        output_dir.mkdir(parents=True)
        entries: list[dict[str, Any]] = []
        evaluated = kwargs["input_table"].to_pandas()
        evaluation_fields = {
            "LMJudgeEvaluator": "raw_aggregated_ratings",
            "LMJudgeEvaluator_relevance_concept_ratings": "raw_relevance_concept_ratings",
            "LMJudgeEvaluator_relevance_concept_completions": "relevance_concept_completions",
            "LMJudgeEvaluator_relevance_instruction_ratings": (
                "raw_relevance_instruction_ratings"
            ),
            "LMJudgeEvaluator_relevance_instruction_completions": (
                "relevance_instruction_completions"
            ),
            "LMJudgeEvaluator_fluency_ratings": "raw_fluency_ratings",
            "LMJudgeEvaluator_fluency_completions": "fluency_completions",
        }
        evaluated_values = {
            f"{method}_{suffix}": []
            for method in kwargs["best_factors"]
            for suffix in evaluation_fields
        }
        concept_ids = kwargs.get("concept_ids", range(kwargs["max_concepts"]))
        for concept_id in concept_ids:
            concept_frame = evaluated.loc[evaluated["concept_id"].eq(concept_id)]
            methods: dict[str, dict[str, list[Any]]] = {}
            for method in kwargs["best_factors"]:
                declared = concept_frame["factor"].eq(kwargs["best_factors"][method]).tolist()
                raw_concept: list[float | None] = []
                raw_instruction: list[float | None] = []
                raw_fluency: list[float | None] = []
                raw_aggregate: list[float | None] = []
                concept_completions: list[str | None] = []
                instruction_completions: list[str | None] = []
                fluency_completions: list[str | None] = []
                for position, selected in enumerate(declared):
                    if not selected:
                        raw_concept.append(None)
                        raw_instruction.append(None)
                        raw_fluency.append(None)
                        raw_aggregate.append(None)
                        concept_completions.append(None)
                        instruction_completions.append(None)
                        fluency_completions.append(None)
                        continue
                    input_id = int(concept_frame.iloc[position]["input_id"])
                    raw = kwargs["emissions"][(method, concept_id, input_id)]
                    rationale = json.loads(raw.judge_rationale)
                    instruction = 1.25
                    raw_concept.append(raw.concept_rating)
                    raw_instruction.append(instruction)
                    raw_fluency.append(raw.fluency_rating)
                    raw_aggregate.append(
                        (raw.concept_rating + instruction + raw.fluency_rating) / 3
                    )
                    concept_completions.append(rationale["concept"])
                    instruction_completions.append("instruction Rating: [[1.25]]")
                    fluency_completions.append(rationale["fluency"])
                selected_concept = [value for value in raw_concept if value is not None]
                selected_instruction = [
                    value for value in raw_instruction if value is not None
                ]
                selected_fluency = [value for value in raw_fluency if value is not None]
                selected_aggregate = [value for value in raw_aggregate if value is not None]
                block: dict[str, list[Any]] = {
                    "factor": [kwargs["best_factors"][method]],
                    "lm_judge_rating": [sum(selected_aggregate) / len(selected_aggregate)],
                    "relevance_concept_ratings": [
                        sum(selected_concept) / len(selected_concept)
                    ],
                    "relevance_instruction_ratings": [
                        sum(selected_instruction) / len(selected_instruction)
                    ],
                    "fluency_ratings": [sum(selected_fluency) / len(selected_fluency)],
                    "raw_aggregated_ratings": raw_aggregate,
                    "raw_relevance_concept_ratings": raw_concept,
                    "raw_relevance_instruction_ratings": raw_instruction,
                    "raw_fluency_ratings": raw_fluency,
                    "relevance_concept_completions": concept_completions,
                    "relevance_instruction_completions": instruction_completions,
                    "fluency_completions": fluency_completions,
                }
                methods[method] = block
                for suffix, field in evaluation_fields.items():
                    evaluated_values[f"{method}_{suffix}"].extend(block[field])
            entries.append({"concept_id": concept_id, "results": {"LMJudgeEvaluator": methods}})
        steering = output_dir / "steering.jsonl"
        atomic_write_text(
            "".join(json.dumps(entry) + "\n" for entry in entries),
            steering,
        )
        for column, values in evaluated_values.items():
            evaluated[column] = values
        evaluated.to_parquet(output_dir / "steering_data.parquet", index=False)
        atomic_write_text('{"schema_version":1}\n', output_dir / "manifest.json")
        return FinalJudgeResult(
            steering_jsonl=steering,
            actual_evaluator_calls=len(kwargs["best_factors"]) * len(concept_ids) * 10 * 3,
            paid_judge_cache_misses=level + 1,
        )


def _patch_source(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
) -> None:
    validated = _validated(output_dir)
    monkeypatch.setattr(
        "steering.eval.axbench_adv.pipeline.validate_source_contract",
        lambda **_kwargs: validated,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_adv.pipeline.revalidate_source_hashes",
        lambda _validated: None,
    )

    def write_source(_validated: ValidatedSources, path: Path) -> Path:
        document: dict[str, object] = {
            "schema_version": 2,
            "hashes": dict(_validated.hashes),
            "runtime_hashes": dict(_validated.runtime_hashes),
            "runtime_byte_lengths": dict(_validated.runtime_byte_lengths),
            "project_identity": {
                "package_lock_sha256": _validated.project_manifest["package_lock_sha256"],
                "repository_git_commit": _validated.project_manifest[
                    "repository_git_commit"
                ],
                "target_model_tag": _validated.project_manifest["target"]["model_tag"],
                "target_step": _validated.project_manifest["target"]["step"],
                "self_conditioning_masked_policy": _validated.project_manifest["methods"][
                    "SelfConditioningMasked"
                ]["policy"],
            },
        }
        if _validated.nl_attachment is not None:
            document["nl_source_attachment"] = {
                **dict(_validated.nl_attachment.identity),
                "paths": {},
                "runtime_byte_lengths": dict(
                    _validated.nl_attachment.runtime_byte_lengths
                ),
            }
        atomic_write_text(json.dumps(document, sort_keys=True) + "\n", path)
        return path

    monkeypatch.setattr("steering.eval.axbench_adv.pipeline.write_source_contract", write_source)


def test_full_fake_pipeline_exact_grid_lifecycle_artifacts_and_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    _patch_source(monkeypatch, output)
    FakeWorker.instances = []
    FakeSelfConditioningRuntime.instances = []
    optimizer = FakeOptimizer()
    judge = FakeFinalJudge()
    config = replace(
        _config(output),
        reflection=LLMClientConfig(model="test/reflection-model"),
        seed_suffix="custom attack suffix",
        reflection_minibatch_size=7,
    )
    manifest_path = run_pipeline(
        config,
        reflection_lm=FakeReflection(),
        worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=optimizer,
        final_judge=judge,
    )

    assert manifest_path == output.resolve() / "manifest.json"
    assert len(FakeWorker.instances) == len(QWEN_METHODS) * 2
    assert all(worker.closed for worker in FakeWorker.instances)
    assert all(worker.calls[0] == ("", tuple(range(10))) for worker in FakeWorker.instances)
    assert len(FakeSelfConditioningRuntime.instances) == 1
    self_runtime = FakeSelfConditioningRuntime.instances[0]
    assert self_runtime.closed
    assert len(self_runtime.cells) == 4
    assert [cell.method for cell in self_runtime.cells] == [
        "SelfConditioning",
        "SelfConditioning",
        "SelfConditioningMasked",
        "SelfConditioningMasked",
    ]
    assert all(cell.closed for cell in self_runtime.cells)
    assert all(cell.calls[0] == ("", tuple(range(10))) for cell in self_runtime.cells)
    assert [call["proposal_count"] for call in optimizer.calls] == list(PROPOSAL_COUNTS) * 12
    assert {call["gepa_seed"] for call in optimizer.calls} == {42}
    assert len({call["run_dir"] for call in optimizer.calls}) == 48
    assert len(judge.calls) == len(PLOT_EFFORT_LEVELS)
    assert [Path(call["output_dir"]).parent.name for call in judge.calls] == [
        f"p{level}" for level in PLOT_EFFORT_LEVELS
    ]
    assert len(list((output / "suffixes").glob("*/c*/p*/manifest.json"))) == 48

    aggregate = pq.read_table(output / "aggregate.parquet")
    validate_aggregate(aggregate)
    assert aggregate.num_rows == len(METHODS) * len(PLOT_EFFORT_LEVELS)
    assert len(list((output / "plots").glob("*.png"))) == 7
    manifest = json.loads(manifest_path.read_text())
    assert manifest["best_factors"] == BEST_FACTORS
    assert manifest["worker_max_prompt_tokens"] == 2048
    assert manifest["attack_config"] == {
        "reflection": {"provider": "openrouter", "model": "test/reflection-model"},
        "seed_suffix": "custom attack suffix",
        "reflection_minibatch_size": 7,
    }
    assert {call["seed_suffix"] for call in optimizer.calls} == {"custom attack suffix"}
    assert {call["reflection_minibatch_size"] for call in optimizer.calls} == {7}
    assert "natural_language" not in manifest
    assert manifest["methods"] == [
        "DiffMean", "LinearProbe", "LoRA", "PromptSteering",
        "SelfConditioning", "SelfConditioningMasked",
    ]
    assert manifest["source_project"] == {
        "package_lock_sha256": "b" * 64,
        "repository_git_commit": "1" * 40,
        "self_conditioning_masked_policy": SELF_CONDITIONING_MASKED_POLICY,
    }
    assert set(manifest["implementation"]) == {
        "package_lock_sha256",
        "repository_git_commit",
    }
    assert manifest["call_accounting"] == {
        "actual_evaluator_calls": 480,
        "actual_generation_calls": 480,
        "control_generation_calls": 120,
        "control_paid_judge_cache_misses": 0,
        "final_actual_evaluator_calls": 1800,
        "final_paid_judge_cache_misses": 20,
        "gepa_nominal_metric_calls": 180,
        "paid_judge_cache_misses": 0,
        "paid_reflection_cache_misses": 0,
    }
    assert "aggregate.parquet" in manifest["files"]
    assert "gepa" not in {part for path in manifest["files"] for part in Path(path).parts}


def test_concept_slice_runs_only_selected_ids_and_records_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    _patch_source(monkeypatch, output)
    monkeypatch.setenv("AXBENCH_ADV_CONCEPT_LO", "1")
    monkeypatch.setenv("AXBENCH_ADV_CONCEPT_HI", "2")
    FakeWorker.instances = []
    FakeSelfConditioningRuntime.instances = []
    FakeOptimizer.calls = []

    manifest_path = run_pipeline(
        _config(output),
        reflection_lm=FakeReflection(),
        worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=FakeFinalJudge(),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["concept_ids"] == [1]
    assert {worker.concept_id for worker in FakeWorker.instances} == {1}
    runtime = FakeSelfConditioningRuntime.instances[0]
    assert {cell.concept_id for cell in runtime.cells} == {1}
    assert len(FakeOptimizer.calls) == len(METHODS) * len(PROPOSAL_COUNTS)
    aggregate = pq.read_table(output / "aggregate.parquet")
    assert set(aggregate.column("n_concepts").to_pylist()) == {1}


def test_factor_override_for_excluded_method_does_not_expand_execution(tmp_path: Path) -> None:
    points = pipeline_module._best_points(
        _validated(tmp_path / "run"), {"LoRA": 1.0}, methods=("DiffMean",),
    )
    assert tuple(points) == ("DiffMean",)
    assert points["DiffMean"].factor == BEST_FACTORS["DiffMean"]


def test_pipeline_failure_never_publishes_top_level_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed"
    _patch_source(monkeypatch, output)
    with pytest.raises(RuntimeError, match="injected final-judge failure"):
        run_pipeline(
            _config(output),
            reflection_lm=FakeReflection(),
            worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
            self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
            optimizer=FakeOptimizer(),
            final_judge=FakeFinalJudge(fail_level=2),
        )
    assert not (output / "manifest.json").exists()


def test_pipeline_helpers_reject_invalid_cells_and_emission_sets(tmp_path: Path) -> None:
    validated = _validated(tmp_path / "run")
    missing_row = validated.main_frame.index[
        validated.main_frame["concept_id"].eq(1)
        & validated.main_frame["input_id"].eq(9)
        & validated.main_frame["factor"].eq(1.0)
    ][0]
    malformed = replace(validated, main_frame=validated.main_frame.drop(index=missing_row))
    with pytest.raises(AxBenchAdvError, match="source cell"):
        pipeline_module._cell_rows(malformed, method="DiffMean", concept_id=1, factor=1.0)

    invalid_concept = validated.main_frame.copy()
    invalid_concept.loc[invalid_concept["concept_id"].eq(0), "input_concept"] = ""
    with pytest.raises(AxBenchAdvError, match="invalid concept text"):
        pipeline_module._cell_rows(
            replace(validated, main_frame=invalid_concept),
            method="DiffMean",
            concept_id=0,
            factor=1.0,
        )

    raw = _raw("DiffMean", 0, 0, "")
    for evaluations, match in (
        ([(input_id, raw) for input_id in range(9)], "ten inputs"),
        ([(input_id + 1, raw) for input_id in range(10)], "input ids"),
        ([(input_id, raw) for input_id in range(9)] + [(8, raw)], "input ids"),
    ):
        with pytest.raises(AxBenchAdvError, match=match):
            pipeline_module._store_evaluations(
                {},
                method="DiffMean",
                concept_id=0,
                evaluations=evaluations,
            )

    target = {("DiffMean", 0, 0): raw}
    with pytest.raises(AxBenchAdvError, match="duplicate"):
        pipeline_module._store_evaluations(
            target,
            method="DiffMean",
            concept_id=0,
            evaluations=[(input_id, raw) for input_id in range(10)],
        )
    with pytest.raises(AxBenchAdvError, match="not a RawEvaluation"):
        pipeline_module._store_evaluations(
            {},
            method="DiffMean",
            concept_id=0,
            evaluations=[(input_id, "bad") for input_id in range(10)],  # ty: ignore[invalid-argument-type]
        )

    selected = pipeline_module._best_points(validated, {"DiffMean": 0.8})
    assert selected["DiffMean"].factor == 0.8


def test_run_pipeline_rejects_nonfresh_output_before_source_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_called = False

    def unexpected_source(**_kwargs: Any) -> ValidatedSources:
        nonlocal source_called
        source_called = True
        raise AssertionError("source validation must not run")

    monkeypatch.setattr(pipeline_module, "validate_source_contract", unexpected_source)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pipeline_module._fresh_output(empty) == empty.resolve()
    output_file = tmp_path / "file"
    output_file.touch()
    with pytest.raises(AxBenchAdvError, match="must be fresh"):
        run_pipeline(_config(output_file), reflection_lm=FakeReflection())

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "artifact").touch()
    with pytest.raises(AxBenchAdvError, match="must be fresh"):
        run_pipeline(_config(nonempty), reflection_lm=FakeReflection())
    assert source_called is False


def test_default_reflection_cache_path_is_wired_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    _patch_source(monkeypatch, output)
    reflection = FakeReflection()
    seen: dict[str, object] = {}

    def fake_build(**kwargs: Any) -> FakeReflection:
        seen["build"] = kwargs
        return reflection

    def fake_execute(**kwargs: Any) -> Path:
        seen["execute"] = kwargs
        return output / "manifest.json"

    monkeypatch.setattr(pipeline_module, "build_openrouter_reflection_lm", fake_build)
    monkeypatch.setattr(pipeline_module, "_execute", fake_execute)
    result = run_pipeline(_config(output))
    assert result == output / "manifest.json"
    assert seen["execute"]["reflection_lm"] is reflection  # ty: ignore[not-subscriptable]
    assert seen["build"]["max_retries"] == 3  # ty: ignore[not-subscriptable]
