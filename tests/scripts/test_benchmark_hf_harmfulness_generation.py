"""Contracts for the repository-HF harmfulness generation benchmark."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch as t

from scripts.benchmark import benchmark_hf_harmfulness_generation as benchmark


class _RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def generate_batch(self, prompts: list[str], *, seed: int, **_kwargs: Any) -> list[str]:
        self.calls.append((prompts, seed))
        return [f"{seed + row}:{prompt}" for row, prompt in enumerate(prompts)]


def test_outer_batching_seeds_each_batch_from_its_absolute_start() -> None:
    engine = _RecordingEngine()
    prompts = [f"prompt-{index}" for index in range(5)]

    outputs = benchmark._generate_outer_batched(
        cast(Any, engine),
        prompts,
        batch_size=2,
        max_new_tokens=8,
        temperature=0.7,
        top_k=50,
        seed=42,
    )

    assert engine.calls == [
        (["prompt-0", "prompt-1"], 42),
        (["prompt-2", "prompt-3"], 44),
        (["prompt-4"], 46),
    ]
    assert outputs == [
        "42:prompt-0",
        "43:prompt-1",
        "44:prompt-2",
        "45:prompt-3",
        "46:prompt-4",
    ]


def test_batch_scaling_report_uses_one_fixed_prompt_pool() -> None:
    engine = _RecordingEngine()
    prompts = [f"prompt-{index}" for index in range(4)]

    report = benchmark._run_batch_scaling(
        engine=cast(Any, engine),
        prompts=prompts,
        batch_sizes=[2, 4],
        common={
            "max_new_tokens": 8,
            "temperature": 0.7,
            "top_k": 50,
            "seed": 42,
        },
        device=t.device("cpu"),
        iterations=2,
        checkpoint_identity={"format": "repository_hf_sft"},
    )

    assert report["mode"] == "batch_scaling"
    assert report["prompt_count"] == 4
    assert report["batch_sizes"] == [2, 4]
    assert report["measurements"]["2"]["repeatable"] is True
    assert report["measurements"]["4"]["repeatable"] is True
    assert report["measurements"]["2"]["peak_allocated_mib"] is None


@pytest.mark.parametrize("batch_sizes", [[2, 2], [5]])
def test_batch_scaling_rejects_ambiguous_sizes(batch_sizes: list[int]) -> None:
    with pytest.raises(ValueError, match="batch-sizes"):
        benchmark._run_batch_scaling(
            engine=cast(Any, _RecordingEngine()),
            prompts=["a", "b", "c", "d"],
            batch_sizes=batch_sizes,
            common={
                "max_new_tokens": 8,
                "temperature": 0.7,
                "top_k": 50,
                "seed": 42,
            },
            device=t.device("cpu"),
            iterations=1,
            checkpoint_identity={"format": "repository_hf_sft"},
        )


def test_profile_and_batch_scaling_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        benchmark._parse_args(
            [
                "--checkpoint-dir",
                "/tmp/checkpoint",
                "--batch-sizes",
                "8",
                "32",
                "--profile-dir",
                "/tmp/profile",
            ]
        )
