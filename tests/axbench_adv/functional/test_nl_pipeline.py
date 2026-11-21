"""Seven-method pipeline coverage without model/provider substitution semantics."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.eval.axbench_adv.merge_slices import _load_run_slice
from steering.eval.axbench_adv import pipeline, source_contract
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.source_contract import (
    NLAttachmentExpectations,
    NLAttachmentPaths,
    NLSourceAttachment,
)
from steering.eval.axbench_adv.types import ALL_METHODS, METHODS
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import (
    BEST_FACTORS,
    FakeFinalJudge,
    FakeOptimizer,
    FakeReflection,
    FakeSelfConditioningRuntime,
    FakeWorker,
    _config,
    _patch_source,
    _validated,
)
from tests.axbench_adv.unit.test_nl_source_contract import _nl_fixture


def test_optional_nl_pipeline_runs_all_cells_and_publishes_all_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    _patch_source(monkeypatch, output)
    validated = _validated(output)
    entries = copy.deepcopy(validated.steering_entries)
    for entry in entries:
        block = entry["results"]["LMJudgeEvaluator"]
        block["NLSteering"] = copy.deepcopy(block["SelfConditioning"])
    project_manifest = copy.deepcopy(validated.project_manifest)
    project_manifest["methods"]["NLSteering"] = {"fixture_identity": "NL retrieval"}
    validated = replace(
        validated, expectations=replace(validated.expectations, include_nl_steering=True),
        steering_entries=entries, project_manifest=project_manifest,
        nl_target_scores={0: t.tensor([-1.0, 1.0]), 1: t.tensor([1.0, -1.0])},
    )
    monkeypatch.setattr(pipeline, "validate_source_contract", lambda **_kwargs: validated)
    FakeWorker.instances = []
    FakeSelfConditioningRuntime.instances = []
    FakeOptimizer.calls = []
    FakeFinalJudge.calls = []
    config = replace(_config(output), source_expectations=validated.expectations)
    manifest_path = pipeline.run_pipeline(
        config, reflection_lm=FakeReflection(),
        worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=FakeFinalJudge(),
    )
    document = json.loads(manifest_path.read_text())
    assert document["methods"] == list(ALL_METHODS)
    assert document["best_factors"] == {**BEST_FACTORS, "NLSteering": 6.0}
    assert document["natural_language"] == project_manifest["methods"]["NLSteering"]
    assert len(FakeWorker.instances) == 8
    runtime = FakeSelfConditioningRuntime.instances[0]
    assert [cell.method for cell in runtime.cells] == [
        method for method in ("SelfConditioning", "SelfConditioningMasked", "NLSteering") for _ in range(2)
    ]
    assert runtime.kwargs["nl_target_scores"] is validated.nl_target_scores
    assert runtime.closed and all(cell.closed for cell in runtime.cells)
    assert len(FakeOptimizer.calls) == 7 * 2 * 4
    assert document["call_accounting"]["control_generation_calls"] == 7 * 2 * 10
    assert document["call_accounting"]["actual_generation_calls"] == 7 * 2 * 4 * 10
    assert document["call_accounting"]["final_actual_evaluator_calls"] == 7 * 2 * 10 * 3 * 5
    aggregate = pq.read_table(output / "aggregate.parquet")
    assert aggregate.num_rows == 7 * 5
    assert set(aggregate.column("method").to_pylist()) == set(ALL_METHODS)
    assert len(list((output / "plots").glob("*.png"))) == 8
    for level in (0, 1, 2, 4, 8):
        frame = pq.read_table(output / "points" / f"p{level}" / "input" / "steering_data.parquet")
        assert frame.schema.names[-7:] == [f"{method}_steered_generation" for method in ALL_METHODS]
        for method in METHODS:
            generations = frame.column(f"{method}_steered_generation").to_pylist()
            assert all(value is None or value.startswith(method + " ") for value in generations)


def test_method_filter_runs_only_nl_and_records_effective_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    _patch_source(monkeypatch, output)
    validated = _validated(output)
    entries = copy.deepcopy(validated.steering_entries)
    for entry in entries:
        block = entry["results"]["LMJudgeEvaluator"]
        block["NLSteering"] = copy.deepcopy(block["SelfConditioning"])
    project_manifest = copy.deepcopy(validated.project_manifest)
    project_manifest["methods"]["NLSteering"] = {"fixture_identity": "NL retrieval"}
    validated = replace(
        validated,
        expectations=replace(validated.expectations, include_nl_steering=True),
        steering_entries=entries,
        project_manifest=project_manifest,
        nl_target_scores={0: t.tensor([-1.0, 1.0]), 1: t.tensor([1.0, -1.0])},
    )
    monkeypatch.setattr(pipeline, "validate_source_contract", lambda **_kwargs: validated)
    monkeypatch.setenv("AXBENCH_ADV_METHODS", "NLSteering")
    FakeWorker.instances = []
    FakeSelfConditioningRuntime.instances = []
    FakeOptimizer.calls = []
    FakeFinalJudge.calls = []

    manifest_path = pipeline.run_pipeline(
        replace(_config(output), source_expectations=validated.expectations),
        reflection_lm=FakeReflection(),
        worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=FakeFinalJudge(),
    )

    document = json.loads(manifest_path.read_text())
    assert document["methods"] == ["NLSteering"]
    assert document["best_factors"] == {"NLSteering": 6.0}
    assert FakeWorker.instances == []
    runtime = FakeSelfConditioningRuntime.instances[0]
    assert [cell.method for cell in runtime.cells] == ["NLSteering", "NLSteering"]
    assert len(FakeOptimizer.calls) == 2 * 4
    aggregate = pq.read_table(output / "aggregate.parquet")
    assert aggregate.num_rows == 5
    assert set(aggregate.column("method").to_pylist()) == {"NLSteering"}
    assert len(list((output / "plots").glob("*.png"))) == 2


def test_separate_nl_attachment_dispatches_without_changing_base_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    _patch_source(monkeypatch, output)
    validated = _validated(output)
    entries = copy.deepcopy(validated.steering_entries)
    for entry in entries:
        block = entry["results"]["LMJudgeEvaluator"]
        block["NLSteering"] = copy.deepcopy(block["SelfConditioning"])
    attachment_paths = NLAttachmentPaths(
        *(tmp_path / name for name in (
            "project.parquet", "project.json", "evaluated.jsonl",
        ))
    )
    attachment_expectations = NLAttachmentExpectations()
    attachment_hashes = dict.fromkeys(attachment_paths.hashed_files(), "3" * 64)
    attachment = NLSourceAttachment(
        paths=attachment_paths,
        expectations=attachment_expectations,
        hashes=attachment_hashes,
        runtime_hashes={},
        runtime_byte_lengths={},
        selected_frame=validated.project_frame,
        steering_entries=entries,
        target_scores={0: t.tensor([-1.0, 1.0]), 1: t.tensor([1.0, -1.0])},
        identity={"schema_version": 1, "fixture": "separate-c100-source"},
    )
    attached = replace(
        validated,
        nl_attachment=attachment,
        nl_target_scores=attachment.target_scores,
    )
    monkeypatch.setattr(pipeline, "validate_source_contract", lambda **_kwargs: attached)
    FakeWorker.instances = []
    FakeSelfConditioningRuntime.instances = []
    FakeOptimizer.calls = []
    FakeFinalJudge.calls = []
    config = replace(
        _config(output),
        nl_attachment_paths=attachment_paths,
        nl_attachment_expectations=attachment_expectations,
    )

    manifest_path = pipeline.run_pipeline(
        config,
        reflection_lm=FakeReflection(),
        worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
        self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
        optimizer=FakeOptimizer(),
        final_judge=FakeFinalJudge(),
    )

    document = json.loads(manifest_path.read_text())
    assert attached.expectations.include_nl_steering is False
    assert document["methods"] == list(ALL_METHODS)
    assert document["best_factors"]["NLSteering"] == 6.0
    assert document["nl_source_attachment"] == attachment.identity
    assert "natural_language" not in document
    runtime = FakeSelfConditioningRuntime.instances[0]
    assert runtime.kwargs["nl_target_scores"] is attachment.target_scores


@pytest.mark.parametrize("enabled", [False, True])
def test_nl_factor_override_requires_explicit_enablement(tmp_path: Path, enabled: bool) -> None:
    config = _config(tmp_path / "run")
    expectations = replace(config.source_expectations, include_nl_steering=enabled)
    if enabled:
        configured = replace(config, source_expectations=expectations, factor_overrides={"NLSteering": 3.0})
        assert configured.factor_overrides == {"NLSteering": 3.0}
    else:
        with pytest.raises(AxBenchAdvError, match="unknown methods.*NLSteering"):
            replace(config, source_expectations=expectations, factor_overrides={"NLSteering": 3.0})


@pytest.mark.parametrize("mutate_input", [False, True])
def test_real_nl_source_contract_reaches_pipeline_and_final_input_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate_input: bool,
) -> None:
    case = _nl_fixture(tmp_path / "source", monkeypatch)
    output = tmp_path / "run"
    config = replace(
        _config(output), source_paths=case.paths, source_hashes=case.hashes,
        source_expectations=case.expectations,
    )
    original_scorer = source_contract.build_retrieval_target_scores
    computed: list[dict[int, t.Tensor]] = []

    def score(**kwargs: Any) -> dict[int, t.Tensor]:
        result = original_scorer(**kwargs)
        computed.append(result)
        return result

    monkeypatch.setattr(source_contract, "build_retrieval_target_scores", score)
    FakeSelfConditioningRuntime.instances = []
    optimizer = FakeOptimizer()

    def optimize(**kwargs: Any):
        result = optimizer(**kwargs)
        if mutate_input:
            case.description_embeddings.write_bytes(b"changed after source validation")
        return result

    def run() -> Path:
        return pipeline.run_pipeline(
            config, reflection_lm=FakeReflection(),
            worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
            self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
            optimizer=optimize, final_judge=FakeFinalJudge(),
        )

    if mutate_input:
        with pytest.raises(AxBenchAdvError, match="source hash mismatch"):
            run()
        assert not (output / "manifest.json").exists()
    else:
        manifest = json.loads(run().read_text())
        source = json.loads((output / "source_contract.json").read_text())
        assert manifest["natural_language"] == source["natural_language"]
        assert manifest["methods"] == list(ALL_METHODS)
        merged_source = _load_run_slice(output)
        assert merged_source.methods == ALL_METHODS
    assert len(computed) == 1
    runtime = FakeSelfConditioningRuntime.instances[-1]
    forwarded = runtime.kwargs["nl_target_scores"]
    assert set(forwarded) == set(computed[0])
    for concept_id, target in computed[0].items():
        assert forwarded[concept_id].numpy().tobytes() == target.numpy().tobytes()
    assert set(computed[0]) == {0, 1}
    assert all(scores.shape == (5,) for scores in computed[0].values())
    assert runtime.closed
