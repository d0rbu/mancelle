"""Pipeline-to-merger coverage for concept-sliced attached NL runs."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch as t

from scripts.eval.axbench_adv.merge_slices import merge_slices
from steering.eval.axbench_adv import pipeline
from steering.eval.axbench_adv.source_contract import (
    NLAttachmentExpectations,
    NLAttachmentPaths,
    NLSourceAttachment,
)
from tests.axbench_adv.functional.test_phase5_pipeline_vertical_slice import (
    FakeFinalJudge,
    FakeOptimizer,
    FakeReflection,
    FakeSelfConditioningRuntime,
    FakeWorker,
    _config,
    _patch_source,
    _validated,
)


@pytest.mark.parametrize("methods", [("DiffMean",), ("DiffMean", "NLSteering")])
def test_attached_source_diffmean_concept_slices_merge_from_pipeline_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple[str, ...],
) -> None:
    source_root = tmp_path / "source"
    _patch_source(monkeypatch, source_root)
    validated = _validated(source_root)
    entries = copy.deepcopy(validated.steering_entries)
    for entry in entries:
        evaluator = entry["results"]["LMJudgeEvaluator"]
        evaluator["NLSteering"] = copy.deepcopy(evaluator["SelfConditioning"])
    attachment_paths = NLAttachmentPaths(
        tmp_path / "project.parquet",
        tmp_path / "project.json",
        tmp_path / "evaluated.jsonl",
    )
    attachment_expectations = NLAttachmentExpectations()
    attachment = NLSourceAttachment(
        paths=attachment_paths,
        expectations=attachment_expectations,
        hashes=dict.fromkeys(attachment_paths.hashed_files(), "3" * 64),
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
    monkeypatch.setattr(
        pipeline,
        "validate_source_contract",
        lambda **kwargs: replace(attached, output_dir=kwargs["output_dir"]),
    )
    monkeypatch.setenv("AXBENCH_ADV_METHODS", ",".join(methods))

    roots: list[Path] = []
    for label, lo, hi in (("low", 0, 1), ("high", 1, 2)):
        output = tmp_path / label
        monkeypatch.setenv("AXBENCH_ADV_CONCEPT_LO", str(lo))
        monkeypatch.setenv("AXBENCH_ADV_CONCEPT_HI", str(hi))
        config = replace(
            _config(output),
            nl_attachment_paths=attachment_paths,
            nl_attachment_expectations=attachment_expectations,
        )
        manifest = pipeline.run_pipeline(
            config,
            reflection_lm=FakeReflection(),
            worker_factory=FakeWorker,  # ty: ignore[invalid-argument-type]
            self_conditioning_factory=FakeSelfConditioningRuntime,  # ty: ignore[invalid-argument-type]
            optimizer=FakeOptimizer(),
            final_judge=FakeFinalJudge(),
        )
        document = json.loads(manifest.read_text())
        assert document["methods"] == list(methods)
        assert document["nl_source_attachment"] == attachment.identity
        roots.append(manifest.parent)

    merged = merge_slices(roots, tmp_path / "merged")
    document = json.loads(merged.read_text())
    assert document["methods"] == list(methods)
    assert document["concept_ids"] == [0, 1]
    aggregate = pq.read_table(merged.parent / "aggregate.parquet")
    assert aggregate.num_rows == len(methods) * 5
    assert set(aggregate.column("method").to_pylist()) == set(methods)
