"""Opt-in real SmolLM3/artifact/IPC qualification without provider calls.

AXBENCH_ADV_GCG_REAL_SOURCE points to a JSON object with ``prepared_dir``,
``main_dump_dir``, and ``aux_dump_dir`` for the audited SmolLM3 c50 bundle.
The test uses two ordered training pairs, both objectives, and k=1/32, followed
by a bounded real optimizer search and normal sampled generation. Only the
judge is replaced. Campaign quality remains a separate judged experiment.
Exact repeated-gradient checks explicitly select deterministic CUDA algorithms;
the production worker retains its normal backend defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import torch as t

from steering.eval.axbench_adv import gcg_driver, worker_client
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective
from steering.eval.axbench_adv.gcg_driver import _training_rows, execute_gcg_cell
from steering.eval.axbench_adv.method_grid import ADV_METHOD_FACTORS
from steering.eval.axbench_adv.worker_client import AxBenchAdvWorkerClient, WorkerExample
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_upstream.data import select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    prompt_wrapper_from_manifest,
)
from tests.axbench_adv.gcg_factor_reference import capture_search_progress, tensor_digest
from tests.scripts.eval.test_axbench_judge_cache import JudgeCache

pytestmark = pytest.mark.slow


@pytest.fixture
def deterministic_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = worker_client.build_worker_command

    def command() -> list[str]:
        prefix = original()
        python_index = prefix.index("python")
        bootstrap = (
            "import torch as t; t.use_deterministic_algorithms(True); "
            "from scripts.eval.axbench_adv.vendored import worker; "
            "from tests.axbench_adv.integration.gcg_worker_reference import verify_gcg_numerics; "
            f"verify_gcg_numerics(capture_path={str(tmp_path / 'worker-captures.jsonl')!r}); "
            "worker.WorkerRuntime._judge = lambda self, generation: (0.0, 2.0, '{}', 0); "
            "worker.main()"
        )
        return prefix[: python_index + 1] + ["-c", bootstrap]

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(worker_client, "build_worker_command", command)


@pytest.mark.parametrize(
    ("method", "factor"),
    [
        (method, factor)
        for method in ("PromptSteering", "DiffMean", "LinearProbe", "LoRA")
        for factor in (0.0, 0.6, 1.0, 3.0)
        if factor in ADV_METHOD_FACTORS[method]
    ],
)
def test_real_hf_gcg_objectives_and_ipc(
    method: str,
    factor: float,
    tmp_path: Path,
    deterministic_worker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = os.environ.get("AXBENCH_ADV_GCG_REAL_SOURCE")
    if source is None:
        pytest.skip("set AXBENCH_ADV_GCG_REAL_SOURCE for the real SmolLM3 qualification")
    assert t.cuda.is_available(), "explicit GCG qualification requires CUDA"
    paths = json.loads(Path(source).read_text())
    prepared_dir = Path(paths["prepared_dir"])
    main_dir, aux_dir = Path(paths["main_dump_dir"]), Path(paths["aux_dump_dir"])
    wrapper = prompt_wrapper_from_manifest(main_dir / "run_manifest.json")
    assert wrapper.model.value == "HuggingFaceTB/SmolLM3-3B-checkpoints"
    assert wrapper.revision == "it-soup-APO"
    all_pairs = select_training_pairs(prepared_dir, max_concepts=1)[0]
    training = replace(all_pairs, positive=all_pairs.positive[:2], negative=all_pairs.negative[:2])
    rows, _identity = _training_rows(training, master_seed=42, evaluation_prompts=())
    main = pd.read_parquet(main_dir / "inference/steering_data.parquet")
    aux = pd.read_parquet(aux_dir / "inference/steering_data.parquet")
    main_row = main.loc[main["concept_id"].eq(0) & main["input_id"].eq(0)].iloc[0]
    aux_row = aux.loc[aux["concept_id"].eq(0) & aux["input_id"].eq(0)].iloc[0]
    with JudgeCache(tmp_path / "judge.sqlite3"):
        pass
    payload = {
        "api_base_url": "http://127.0.0.1:1",
        "cache_only": True,
        "aux_dump_dir": str(aux_dir),
        "main_dump_dir": str(main_dir),
        "prepared_dir": str(prepared_dir),
        "concept": all_pairs.concept,
        "concept_id": 0,
        "device": "cuda:0",
        "factor": factor,
        "intervene_on_prompt": True,
        "intervention_type": "addition",
        "judge_cache": str(tmp_path / "judge.sqlite3"),
        "judge_model": "openai/gpt-5.6-luna",
        "layer": 28,
        "max_concepts": 50,
        "method": method,
        "model_name": wrapper.model.value,
        "model_revision": wrapper.revision,
        "output_length": 8,
        "steering_layers": [28],
        "temperature": 1.0,
        "use_bf16": True,
        "worker_max_prompt_tokens": 1024,
    }
    examples = [
        WorkerExample(input_id=0, input=main_row["input"], steered_input=aux_row["steered_input"])
    ]
    progress = capture_search_progress(gcg_driver, monkeypatch)
    with AxBenchAdvWorkerClient(
        init_payload=payload,
        examples=examples,
        master_seed=42,
        timeout_seconds=300,
        stderr_path=tmp_path / "worker.log",
    ) as worker:
        clean_before = worker.evaluate("", [0])
        attacked_before = worker.evaluate("!", [0])
        trace = {
            "method": method,
            "factor": factor,
            "cells": [],
            "generations": [],
            "clean_generation": [asdict(row) for row in clean_before],
            "attacked_generation": [asdict(row) for row in attacked_before],
        }
        for objective in ("response_margin", "activation_l2"):
            for k in (1, 32):
                prepared = worker.prepare_gcg(
                    objective=objective,
                    alpha=0.01,
                    rows=rows,
                    suffix_token_count=k,
                    device_batch_size=2,
                    separator="\n\n",
                )
                assert prepared.member_count == 2
                assert len(prepared.seed_token_ids) == k
                [filtered] = worker.gcg_filter([prepared.seed_token_ids])
                assert filtered.valid, filtered.error
                assert filtered.text == " ".join(["!"] * k)
                single = worker.gcg_score(
                    [prepared.seed_token_ids],
                    [0, 1],
                    candidate_eval_device_batch_size=1,
                )
                batched = worker.gcg_score(
                    [prepared.seed_token_ids],
                    [0, 1],
                    candidate_eval_device_batch_size=2,
                )
                # Zero-offset BF16 cancellation defeats a fixed relative-loss tolerance.
                # Both batches still pass the exact independent score oracle.
                if factor != 0.0 or objective != "activation_l2":
                    t.testing.assert_close(
                        t.tensor(single),
                        t.tensor(batched),
                        # Real BF16 LoRA varied by 0.9%; the same loaded model in
                        # FP32 varied by <1e-6 across all four objective/k cells.
                        rtol=1e-2 if method == "LoRA" else 5e-3,
                        atol=5e-3,
                        msg=f"{method}/{objective}/k{k}: {single=} {batched=}",
                    )
                gradients = worker.gcg_gradient(prepared.seed_token_ids, [1, 0])
                assert len(gradients) == 2
                for gradient in gradients:
                    assert gradient.shape == (k, prepared.embedding_weight.shape[1])
                    assert gradient.dtype is t.float32
                    assert bool(t.isfinite(gradient).all())
                    assert bool(gradient.ne(0).any())
                repeated = worker.gcg_gradient(prepared.seed_token_ids, [1, 0])
                assert all(t.equal(a, b) for a, b in zip(repeated, gradients, strict=True))
                trace["cells"].append(
                    {
                        "objective": objective,
                        "k": k,
                        "scores": single,
                        "batched_scores": batched,
                        "gradients": [tensor_digest(g) for g in gradients],
                    }
                )
                assert (prepared.direction_calibration is None) == (objective == "response_margin")
                if prepared.direction_calibration is not None:
                    assert prepared.direction_calibration["factor"] == factor
                    assert (
                        prepared.direction_calibration["resolved_scale"]
                        == prepared.direction_calibration["max_activation"] * factor
                    )
            totals = {
                "actual_generation_calls": 0,
                "actual_evaluator_calls": 0,
                "paid_judge_cache_misses": 0,
            }
            result = execute_gcg_cell(
                config=AxBenchAdvGcgConfig(
                    objective=GcgObjective(objective),
                    device_batch_size=2,
                    search=replace(
                        FasterGcgConfig(),
                        optimized_token_counts=(1,),
                        max_steps=3,
                        candidate_batch_size=4,
                        top_k=32,
                        universal_minibatch_size=2,
                    ),
                ),
                runtime=worker,
                method=method,
                concept_id=0,
                concept=all_pairs.concept,
                factor=factor,
                training_pairs=training,
                evaluation_prompts=[str(main_row["original_prompt"])],
                evaluation_input_ids=[0],
                output_dir=tmp_path / objective,
                totals=totals,
            )
            assert list(result) == [1]
            assert len(result[1]) == 1
            assert isinstance(result[1][0][1].generation, str)
            assert totals["actual_generation_calls"] == 1
            assert totals["paid_judge_cache_misses"] == 0
            assert worker.evaluate("", [0]) == clean_before
            assert worker.evaluate("!", [0]) == attacked_before
            trace["generations"].append([asdict(raw) for _input_id, raw in result[1]])
            assert any(event["event"] == "step_done" for event in progress)
            trace["search_progress"] = progress
            (tmp_path / "trace.json").write_text(json.dumps(trace, sort_keys=True))
