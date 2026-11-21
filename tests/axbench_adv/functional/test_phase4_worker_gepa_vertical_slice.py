"""Real GEPA over the real framed root client and a deterministic fake child."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_adv import worker_client
from steering.eval.axbench_adv.gepa_optimization import (
    CallAccounting,
    OptimizationExample,
    optimize_suffix,
)
from steering.eval.axbench_adv.prompt_surgery import QWEN_TERMINAL_MARKER
from steering.eval.axbench_adv.worker_client import AxBenchAdvWorkerClient, WorkerExample

CHILD_MODULE = "tests.axbench_adv.fixtures.gepa_worker_child"


class Reflection:
    paid_cache_misses = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        assert isinstance(prompt, str)
        self.paid_cache_misses += 1
        return "```better```"


def test_real_gepa_real_client_persistent_fake_worker_improves_and_hits_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        worker_client,
        "build_worker_command",
        lambda: [sys.executable, "-m", CHILD_MODULE],
    )
    prompt = f"request {{input_id}}{QWEN_TERMINAL_MARKER}"
    worker_examples = [
        WorkerExample(input_id, prompt.format(input_id=input_id), "unused") for input_id in range(4)
    ]
    optimization_examples = [
        OptimizationExample(example.input_id, f"request {example.input_id}")
        for example in worker_examples
    ]
    with AxBenchAdvWorkerClient(
        init_payload={"method": "LoRA", "concept_id": 0, "factor": 1.0},
        examples=worker_examples,
        master_seed=123,
        stderr_path=tmp_path / "worker.log",
    ) as client:
        result = optimize_suffix(
            examples=optimization_examples,
            runtime=lambda suffix, input_id: client.evaluate(suffix, [input_id])[0],
            reflection_lm=Reflection(),
            objective="Avoid the target concept while remaining useful.",
            run_dir=tmp_path / "gepa",
            proposal_count=1,
            gepa_seed=17,
            seed_suffix="seed",
            reflection_minibatch_size=3,
        )
    assert result.suffix == "better"
    assert result.best_fitness == 0.5
    assert result.accounting == CallAccounting(14, 8, 8, 16, 1)
    assert [evaluation.raw.generation for evaluation in result.evaluations] == [
        "A useful substantive answer that avoids the target."
    ] * 4
