"""Opt-in cached real Gemma/SAE/project-checkpoint GCG qualification.

AXBENCH_ADV_GCG_REAL_SOURCE additionally supplies target_meta_path,
feature_statistics_path, project_direction_root, and project_layer. The normal
runtime loads all concept-zero training texts; search uses two ordered pairs.
Only the judge is replaced, so this test never calls an external provider.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch as t

from nanochat.common import COMPUTE_DTYPE
from nanochat.tokenizer import SPECIAL_TOKENS
from steering.eval.axbench_adv import gcg_driver
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_project_gcg_activation_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective
from steering.eval.axbench_adv.gcg_driver import _training_rows, execute_gcg_cell
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample
from steering.eval.axbench_adv.self_conditioning import (
    SelfConditioningMethodRuntime,
    _BoundSelfConditioningCell,
)
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_upstream.data import (
    select_training_pairs,
    select_training_texts,
)
from steering.eval.axbench_upstream.methods import load_method_setup
from tests.axbench_adv.gcg_factor_reference import capture_search_progress, tensor_digest
from tests.axbench_adv.gcg_pool_reference import original_project_allowed_token_ids
from tests.axbench_adv.integration.gcg_project_reference import verify_project_l2_numerics
from tests.axbench_adv.integration.test_self_conditioning_cuda import _FakeJudge

pytestmark = pytest.mark.slow


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def test_real_project_gcg_objectives_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = os.environ.get("AXBENCH_ADV_GCG_REAL_SOURCE")
    if source is None:
        pytest.skip("set AXBENCH_ADV_GCG_REAL_SOURCE for real checkpoint qualification")
    paths = json.loads(Path(source).read_text())
    if "target_meta_path" not in paths:
        pytest.skip("the source JSON must include the project checkpoint/fit paths")
    assert t.cuda.is_available(), "explicit GCG qualification requires CUDA"
    prepared_dir = Path(paths["prepared_dir"])
    meta_path = Path(paths["target_meta_path"])
    setup = load_method_setup(meta_path, Path(paths["feature_statistics_path"]))
    meta = json.loads(meta_path.read_text())
    checkpoint_path = meta_path.with_name(meta_path.name.replace("meta_", "model_")).with_suffix(
        ".pt"
    )
    all_pairs = select_training_pairs(prepared_dir, max_concepts=1)[0]
    training = replace(all_pairs, positive=all_pairs.positive[:2], negative=all_pairs.negative[:2])
    rows, _identity = _training_rows(training, master_seed=42, evaluation_prompts=())
    fit = preflight_project_direction_fit(
        Path(paths["project_direction_root"]),
        ProjectCounterExpectations(
            target_model_sha256=_sha256(checkpoint_path),
            target_metadata_sha256=_sha256(meta_path),
            concept_texts={0: all_pairs.concept},
            decoder_width=meta["model_config"]["n_embd"],
            layer=paths["project_layer"],
            fit_seed=42,
        ),
    )
    direction = load_project_gcg_activation_direction(fit, concept_id=0)
    main = pd.read_parquet(Path(paths["main_dump_dir"]) / "inference/steering_data.parquet")
    main = main.loc[main["concept_id"].eq(0)].copy()
    examples = [
        OptimizationExample(
            input_id=input_id,
            user_message=str(main.loc[main["input_id"].eq(input_id), "original_prompt"].iloc[0]),
        )
        for input_id in range(10)
    ]
    progress = capture_search_progress(gcg_driver, monkeypatch)
    with (
        verify_project_l2_numerics(),
        SelfConditioningMethodRuntime(
            setup=setup,
            training=select_training_texts(prepared_dir, max_concepts=1),
            main_frame=main,
            max_concepts=1,
            max_new_tokens=8,
            master_seed=42,
            device="cuda:0",
            extraction_batch_size=4,
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
                    -1.0,
                    1.0,
                    setup.n_sae_features
                    if setup.feature_vocab is None
                    else setup.feature_vocab.size,
                )
            },
        ) as runtime,
    ):
        for method, factor in product(
            ("SelfConditioning", "SelfConditioningMasked", "NLSteering"), (0.0, 0.6, 1.0, 3.0)
        ):
            with runtime.bind_cell(
                method=method,
                concept_id=0,
                concept=all_pairs.concept,
                factor=factor,
                examples=examples,
            ) as cell:
                assert isinstance(cell, _BoundSelfConditioningCell)
                progress_start = len(progress)
                before = cell.evaluate("", [0])
                attacked_before = cell.evaluate("!", [0])
                trace = {
                    "method": method,
                    "factor": factor,
                    "cells": [],
                    "clean_generation": [asdict(row) for row in before],
                    "attacked_generation": [asdict(row) for row in attacked_before],
                }
                for objective in ("response_margin", "activation_l2"):
                    for k in (1, 32):
                        # Cached weights must remain usable by normal backward
                        # after inference-mode preparation, including CUDA transfer.
                        inference = objective == "response_margin" and k == 32
                        with t.inference_mode(inference):
                            prepared = cell.prepare_gcg(
                                objective=objective,
                                alpha=0.01,
                                rows=rows,
                                suffix_token_count=k,
                                device_batch_size=1,
                                separator="\n\n",
                            )
                            assert t.is_inference_mode_enabled() is inference
                        assert prepared.member_count == 2
                        assert len(prepared.seed_token_ids) == k
                        assert prepared.allowed_token_ids == original_project_allowed_token_ids(
                            runtime._target_tokenizer, SPECIAL_TOKENS
                        )
                        [filtered] = cell.gcg_filter([prepared.seed_token_ids])
                        assert filtered.valid, filtered.error
                        assert filtered.text == " ".join(["!"] * k)
                        one = cell.gcg_score(
                            [prepared.seed_token_ids],
                            [0, 1],
                            candidate_eval_device_batch_size=1,
                        )
                        two = cell.gcg_score(
                            [prepared.seed_token_ids],
                            [0, 1],
                            candidate_eval_device_batch_size=2,
                        )
                        # Zero-offset BF16 cancellation defeats a fixed relative-loss tolerance.
                        # Both batches still pass the exact independent score oracle.
                        if (
                            factor != 0.0
                            or objective != "activation_l2"
                            or COMPUTE_DTYPE != t.bfloat16
                        ):
                            t.testing.assert_close(
                                t.tensor(one), t.tensor(two), rtol=5e-3, atol=5e-3
                            )
                        gradients = cell.gcg_gradient(prepared.seed_token_ids, [1, 0])
                        assert len(gradients) == 2
                        for gradient in gradients:
                            assert gradient.shape == (k, prepared.embedding_weight.shape[1])
                            assert gradient.dtype is t.float32
                            assert bool(t.isfinite(gradient).all())
                            assert bool(gradient.ne(0).any())
                        repeated = cell.gcg_gradient(prepared.seed_token_ids, [0])[0]
                        t.testing.assert_close(repeated, gradients[1], rtol=1e-4, atol=1e-6)
                        trace["cells"].append(
                            {
                                "objective": objective,
                                "k": k,
                                "scores": one,
                                "batched_scores": two,
                                "gradients": [tensor_digest(g) for g in gradients],
                                "clean": [
                                    None if row.clean_l2 is None else tensor_digest(row.clean_l2)
                                    for pair in cell._gcg_pairs
                                    for row in (pair.positive, pair.negative)
                                ],
                            }
                        )
                        assert (prepared.direction_calibration is None) == (
                            objective == "response_margin"
                        )
                        if prepared.direction_calibration is not None:
                            assert prepared.direction_calibration["factor"] == factor
                            assert (
                                prepared.direction_calibration["resolved_scale"]
                                == direction.max_activation * factor
                            )
                totals = {
                    "actual_generation_calls": 0,
                    "actual_evaluator_calls": 0,
                    "paid_judge_cache_misses": 0,
                }
                result = execute_gcg_cell(
                    config=AxBenchAdvGcgConfig(
                        objective=GcgObjective.ACTIVATION_L2,
                        device_batch_size=1,
                        search=replace(
                            FasterGcgConfig(),
                            optimized_token_counts=(1,),
                            max_steps=2,
                            candidate_batch_size=4,
                            top_k=32,
                            universal_minibatch_size=2,
                            candidate_eval_device_batch_size=1,
                        ),
                    ),
                    runtime=cell,
                    method=method,
                    concept_id=0,
                    concept=all_pairs.concept,
                    factor=factor,
                    training_pairs=training,
                    evaluation_prompts=[examples[0].user_message],
                    evaluation_input_ids=[0],
                    output_dir=tmp_path / method / str(factor),
                    totals=totals,
                )
                assert isinstance(result[1][0][1].generation, str)
                assert totals["actual_generation_calls"] == 1
                assert totals["paid_judge_cache_misses"] == 0
                assert cell.evaluate("", [0]) == before
                attacked_after = cell.evaluate("!", [0])
                assert attacked_after == attacked_before
                [attacked] = attacked_after
                assert isinstance(attacked.generation, str)
                assert attacked.paid_judge_cache_misses == 0
                trace["generations"] = [asdict(raw) for _input_id, raw in result[1]]
                trace["search_progress"] = progress[progress_start:]
                assert any(event["event"] == "step_done" for event in trace["search_progress"])
                (tmp_path / method / str(factor) / "trace.json").write_text(
                    json.dumps(trace, sort_keys=True)
                )
