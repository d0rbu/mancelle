"""Opt-in real checkpoint/SAE soft suffix qualification; the judge is local and fake.

The existing weight-free loading path avoids unused TransformerLens weights;
the nnterp Gemma decoder, SAE, checkpoint, and scientific inputs are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch as t
from safetensors.torch import load_file

import nanochat.flash_attention as attention
from nanochat.common import COMPUTE_DTYPE
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_project_gcg_activation_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.gcg_config import GcgObjective
from steering.eval.axbench_adv.gcg_driver import _training_rows
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample
from steering.eval.axbench_adv.self_conditioning import (
    SelfConditioningMethodRuntime,
    _BoundSelfConditioningCell,
)
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.soft_prompt_driver import execute_soft_prompt_cell
from steering.eval.axbench_upstream.data import select_training_pairs, select_training_texts
from steering.eval.axbench_upstream.methods import load_method_setup
from tests.axbench_adv.integration.test_self_conditioning_cuda import _FakeJudge

pytestmark = pytest.mark.slow


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture
def deterministic_project_cuda(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    t.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        t.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.usefixtures("deterministic_project_cuda")
def test_real_project_soft_objectives_adam_and_retained_generation(tmp_path: Path) -> None:
    source = os.environ.get("AXBENCH_ADV_GCG_REAL_SOURCE")
    if source is None:
        pytest.skip("set AXBENCH_ADV_GCG_REAL_SOURCE for cached real project qualification")
    paths = json.loads(Path(source).read_text())
    if "target_meta_path" not in paths:
        pytest.skip("the source JSON must include project checkpoint and direction paths")
    assert t.cuda.is_available(), "explicit real project qualification requires CUDA"
    prepared_dir = Path(paths["prepared_dir"])
    meta_path = Path(paths["target_meta_path"])
    metadata = json.loads(meta_path.read_text())
    checkpoint = meta_path.with_name(meta_path.name.replace("meta_", "model_")).with_suffix(".pt")
    setup = load_method_setup(meta_path, Path(paths["feature_statistics_path"]))
    fit_root = Path(paths["project_direction_root"])
    fit_manifest = json.loads((fit_root / "fit_manifest.json").read_text())
    fit_concepts = {record["concept_id"] for record in fit_manifest["artifacts"]}
    canonical = select_training_pairs(prepared_dir, max_concepts=max(fit_concepts) + 1)
    # Authenticate the entire fitted concept population, then run concept zero.
    fit = preflight_project_direction_fit(
        fit_root,
        ProjectCounterExpectations(
            target_model_sha256=_sha256(checkpoint),
            target_metadata_sha256=_sha256(meta_path),
            concept_texts={
                concept_id: canonical[concept_id].concept for concept_id in fit_concepts
            },
            decoder_width=metadata["model_config"]["n_embd"],
            layer=paths["project_layer"],
            fit_seed=42,
        ),
    )
    direction = load_project_gcg_activation_direction(fit, concept_id=0)
    training = replace(
        canonical[0], positive=canonical[0].positive[:2], negative=canonical[0].negative[:2]
    )
    main = pd.read_parquet(Path(paths["main_dump_dir"]) / "inference/steering_data.parquet")
    main = main.loc[main["concept_id"].eq(0)].copy()
    examples = [
        OptimizationExample(
            input_id=index,
            user_message=str(main.loc[main["input_id"].eq(index), "original_prompt"].iloc[0]),
        )
        for index in range(10)
    ]
    rows, _identity = _training_rows(
        training,
        master_seed=42,
        evaluation_prompts=[example.user_message for example in examples],
    )
    evidence: dict[str, Any] = {
        "target_model_sha256": _sha256(checkpoint),
        "target_metadata_sha256": _sha256(meta_path),
        "fit_manifest_sha256": _sha256(fit_root / "fit_manifest.json"),
        "fit_concepts": len(fit_concepts),
        "compute_dtype": str(COMPUTE_DTYPE),
        "attention_impl": str(attention._active_impl()),
        "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": t.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": t.get_float32_matmul_precision(),
        "torch_version": t.__version__,
        "cuda_version": t.version.cuda,
        "weight_free_extraction": True,
        "native_extraction": False,
        "gpu": t.cuda.get_device_name(),
        "cells": [],
    }
    t.cuda.reset_peak_memory_stats()
    with SelfConditioningMethodRuntime(
        setup=setup,
        training=select_training_texts(prepared_dir, max_concepts=1),
        main_frame=main,
        max_concepts=1,
        max_new_tokens=8,
        master_seed=42,
        device="cuda:0",
        extraction_batch_size=1,
        weight_free_extraction=True,
        judge_factory=cast(Any, _FakeJudge),
        judge_cache=tmp_path / "unused.sqlite3",
        judge_model="test/no-provider",
        api_base_url="http://127.0.0.1:1",
        cache_only=True,
        timeout_seconds=120,
        stderr_dir=tmp_path / "judge-logs",
        gcg_activation_directions={0: direction},
        nl_target_scores={
            0: t.linspace(
                -1,
                1,
                setup.n_sae_features if setup.feature_vocab is None else setup.feature_vocab.size,
            )
        },
    ) as runtime:
        for method in ("SelfConditioning", "SelfConditioningMasked", "NLSteering"):
            with runtime.bind_cell(
                method=method,
                concept_id=0,
                concept=training.concept,
                factor=1.0,
                examples=examples,
            ) as bound_cell:
                cell = cast(_BoundSelfConditioningCell, bound_cell)
                before = cell.evaluate("", [0])
                for objective in (GcgObjective.RESPONSE_MARGIN, GcgObjective.ACTIVATION_L2):
                    prepared = cell.prepare_soft_prompt(
                        objective=objective.value,
                        alpha=0.01,
                        rows=rows,
                        suffix_token_count=1,
                        device_batch_size=1,
                        separator="\n\n",
                    )
                    embeddings = prepared.embedding_weight[list(prepared.seed_token_ids)]
                    scores = cell.soft_score(embeddings, [1, 0])
                    assert scores == cell.gcg_score(
                        [prepared.seed_token_ids], [1, 0], candidate_eval_device_batch_size=1
                    )
                    gradients = cell.soft_gradient(embeddings, [1, 0])
                    expected = cell.gcg_gradient(prepared.seed_token_ids, [1, 0])
                    for actual, reference in zip(gradients, expected, strict=True):
                        assert t.equal(actual, reference)
                        assert bool(t.isfinite(actual).all()) and bool(actual.ne(0).any())
                    changed = embeddings + t.linspace(-0.15, 0.15, embeddings.numel()).view_as(
                        embeddings
                    )
                    assert cell.soft_score(changed, [1, 0]) != scores
                    output_dir = tmp_path / method / objective.value
                    totals = {
                        "actual_generation_calls": 0,
                        "actual_evaluator_calls": 0,
                        "paid_judge_cache_misses": 0,
                    }
                    result = execute_soft_prompt_cell(
                        config=AxBenchAdvSoftPromptConfig(
                            objective=objective,
                            device_batch_size=1,
                            optimized_token_counts=(0, 1),
                            max_steps=2,
                            universal_minibatch_size=2,
                        ),
                        runtime=cell,
                        master_seed=42,
                        method=method,
                        concept_id=0,
                        concept=training.concept,
                        factor=1.0,
                        training_pairs=training,
                        evaluation_prompts=[example.user_message for example in examples],
                        evaluation_input_ids=[0],
                        output_dir=output_dir,
                        totals=totals,
                    )
                    cell_dir = output_dir / "soft_prompt" / method / "c0" / "k1"
                    retained = load_file(cell_dir / "embeddings.safetensors")["embeddings"]
                    receipt = json.loads((cell_dir / "search.json").read_text())
                    assert receipt["diagnostics"]["updates"] == 2
                    assert cell.evaluate_soft(retained, [0]) == tuple(
                        raw for _index, raw in result[1]
                    )
                    assert totals == {
                        "actual_generation_calls": 1,
                        "actual_evaluator_calls": 1,
                        "paid_judge_cache_misses": 0,
                    }
                    assert cell.evaluate("", [0]) == before
                    evidence["cells"].append(
                        {
                            "method": method,
                            "objective": objective.value,
                            "scores": scores,
                            "tensor_sha256": receipt["tensor"]["sha256"],
                            "objective_value": receipt["objective_value"],
                            "diagnostics": receipt["diagnostics"],
                            "generation": result[1][0][1].generation,
                        }
                    )
    evidence["peak_allocated_bytes"] = t.cuda.max_memory_allocated()
    evidence["peak_reserved_bytes"] = t.cuda.max_memory_reserved()
    (tmp_path / "real-project-soft-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
