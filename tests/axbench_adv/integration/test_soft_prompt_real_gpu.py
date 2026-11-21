"""Opt-in real SmolLM3 fits/IPC/Adam/retained-generation gate, without providers.

Use the existing AXBENCH_ADV_GCG_REAL_SOURCE JSON containing prepared_dir,
main_dump_dir, and aux_dump_dir. This bounds qualification to two training
pairs, k=1, two Adam updates, and one final row per objective and method.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch as t
from safetensors.torch import load_file

from steering.eval.axbench_adv import worker_client
from steering.eval.axbench_adv.gcg_config import GcgObjective
from steering.eval.axbench_adv.gcg_driver import _training_rows
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.soft_prompt_driver import execute_soft_prompt_cell
from steering.eval.axbench_adv.worker_client import AxBenchAdvWorkerClient, WorkerExample
from steering.eval.axbench_upstream.data import select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    prompt_wrapper_from_manifest,
)
from tests.axbench_adv.gcg_factor_reference import tensor_digest
from tests.scripts.eval.test_axbench_judge_cache import JudgeCache

pytestmark = pytest.mark.slow


@pytest.fixture
def deterministic_soft_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = worker_client.build_worker_command
    (tmp_path / "soft_prompt_real_worker.py").write_text(
        "import torch as t\n"
        "from scripts.eval.axbench_adv.vendored import worker\n"
        "t.use_deterministic_algorithms(True)\n"
        "worker.WorkerRuntime._judge = lambda self, generation: (0.0, 2.0, '{}', 0)\n"
        "serve = worker.serve\n"
    )

    def command() -> list[str]:
        prefix = original()
        python_index = prefix.index("python")
        bootstrap = (
            "from scripts.eval.axbench_adv.vendored.bootstrap import serve_module; "
            "raise SystemExit(serve_module('soft_prompt_real_worker'))"
        )
        return prefix[: python_index + 1] + ["-c", bootstrap]

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            (str(tmp_path), str(worker_client.VENDOR_ROOT), str(worker_client.REPO_ROOT))
        ),
    )
    monkeypatch.setattr(worker_client, "build_worker_command", command)


@pytest.mark.parametrize("method", ["PromptSteering", "DiffMean", "LinearProbe", "LoRA"])
def test_real_hf_soft_prompt_parity_adam_and_retained_generation(
    tmp_path: Path, method: str, deterministic_soft_worker: None
) -> None:
    source = os.environ.get("AXBENCH_ADV_GCG_REAL_SOURCE")
    if source is None:
        pytest.skip("set AXBENCH_ADV_GCG_REAL_SOURCE for the real SmolLM3 soft-prompt gate")
    assert t.cuda.is_available(), "explicit real soft-prompt qualification requires CUDA"
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
        "factor": 1.0,
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
    examples = [WorkerExample(0, main_row["input"], aux_row["steered_input"])]
    trace: dict[str, Any] = {"method": method, "factor": 1.0, "cells": []}
    with AxBenchAdvWorkerClient(
        init_payload=payload,
        examples=examples,
        master_seed=42,
        timeout_seconds=600,
        stderr_path=tmp_path / "worker.log",
    ) as runtime:
        clean, textual = runtime.evaluate("", [0]), runtime.evaluate("!", [0])
        for objective in (GcgObjective.RESPONSE_MARGIN, GcgObjective.ACTIVATION_L2):
            ordinary = runtime.prepare_gcg(
                objective=objective.value,
                alpha=0.01,
                rows=rows,
                suffix_token_count=1,
                device_batch_size=2,
            )
            expected_scores = runtime.gcg_score(
                [ordinary.seed_token_ids], [1, 0], candidate_eval_device_batch_size=2
            )
            expected_gradients = runtime.gcg_gradient(ordinary.seed_token_ids, [1, 0])
            prepared = runtime.prepare_soft_prompt(
                objective=objective.value,
                alpha=0.01,
                rows=rows,
                suffix_token_count=1,
                device_batch_size=2,
            )
            seed = prepared.embedding_weight[list(prepared.seed_token_ids)].clone()
            assert runtime.soft_score(seed, [1, 0]) == expected_scores
            gradients = runtime.soft_gradient(seed, [1, 0])
            assert all(
                t.equal(actual, expected)
                for actual, expected in zip(gradients, expected_gradients, strict=True)
            )
            assert runtime.evaluate_soft(seed, [0]) == textual
            totals = {
                "actual_generation_calls": 0,
                "actual_evaluator_calls": 0,
                "paid_judge_cache_misses": 0,
            }
            output = tmp_path / objective.value
            generated = execute_soft_prompt_cell(
                config=AxBenchAdvSoftPromptConfig(
                    objective=objective,
                    device_batch_size=2,
                    optimized_token_counts=(0, 1),
                    max_steps=2,
                    universal_minibatch_size=2,
                ),
                runtime=runtime,
                master_seed=42,
                method=method,
                concept_id=0,
                concept=all_pairs.concept,
                factor=1.0,
                training_pairs=training,
                evaluation_prompts=[str(main_row["original_prompt"])],
                evaluation_input_ids=[0],
                output_dir=output,
                totals=totals,
            )
            cell = output / "soft_prompt" / method / "c0/k1"
            retained = load_file(cell / "embeddings.safetensors")["embeddings"]
            receipt = json.loads((cell / "search.json").read_text())
            assert retained.dtype is t.float32 and retained.shape == seed.shape
            assert receipt["diagnostics"]["updates"] == 2
            assert receipt["diagnostics"]["full_population_scores"] == 3
            assert math.fsum(runtime.soft_score(retained, [0, 1])) / 2 == receipt["objective_value"]
            assert receipt["objective_value"] <= math.fsum(expected_scores) / 2
            assert [raw for _input_id, raw in generated[1]] == list(
                runtime.evaluate_soft(retained, [0])
            )
            assert totals == {
                "actual_generation_calls": 1,
                "actual_evaluator_calls": 1,
                "paid_judge_cache_misses": 0,
            }
            assert runtime.evaluate("", [0]) == clean
            assert runtime.evaluate("!", [0]) == textual
            trace["cells"].append(
                {
                    "objective": objective.value,
                    "initial_scores": expected_scores,
                    "gradients": [tensor_digest(value) for value in gradients],
                    "retained": tensor_digest(retained),
                    "receipt": receipt,
                    "generation": [asdict(raw) for _input_id, raw in generated[1]],
                }
            )
            (tmp_path / "trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True))
