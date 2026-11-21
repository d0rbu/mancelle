"""Coupled multivariate oracle for actual best-iterate Adam retention."""

from __future__ import annotations

import hashlib
import math
import random
from typing import cast

import numpy as np
import pytest
import torch as t

from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.soft_prompt_driver import SoftPromptRuntime, optimize_soft_prompt
from steering.eval.axbench_adv.worker_client import GcgPrepared


class _CoupledPopulation:
    """Seven quadratic losses couple every coordinate of three suffix positions."""

    def __init__(self) -> None:
        generator = np.random.default_rng(193)
        matrices = generator.standard_normal((7, 9, 9))
        self.hessians = np.transpose(matrices, (0, 2, 1)) @ matrices + np.eye(9)
        self.targets = generator.standard_normal((7, 9)) * 0.4
        self.iterates: list[t.Tensor] = []
        self.minibatches: list[list[int]] = []

    def soft_score(self, embeddings: t.Tensor, indices: list[int]) -> list[float]:
        self.iterates.append(embeddings.clone())
        vector = embeddings.numpy().reshape(-1).astype(np.float64)
        differences = vector - self.targets
        return [float(0.5 * differences[i] @ self.hessians[i] @ differences[i]) for i in indices]

    def soft_gradient(self, embeddings: t.Tensor, indices: list[int]) -> list[t.Tensor]:
        self.minibatches.append(list(indices))
        vector = embeddings.numpy().reshape(-1).astype(np.float64)
        return [
            t.from_numpy(
                (self.hessians[i] @ (vector - self.targets[i]))
                .astype(np.float32)
                .reshape(embeddings.shape)
            )
            for i in indices
        ]


@pytest.mark.parametrize(
    ("updates", "learning_rate", "expected_best_step"), [(13, 1.0, 12), (64, 0.01, 64)]
)
def test_every_adam_iterate_matches_coupled_numpy_oracle_and_retains_actual_full_score_best(
    updates: int, learning_rate: float, expected_best_step: int
) -> None:
    runtime = _CoupledPopulation()
    prepared = GcgPrepared(
        embedding_weight=t.tensor([[0.0, 0.0, 0.0], [0.75, -0.5, 0.2]]),
        allowed_token_ids=(1,),
        seed_token_ids=(1, 1, 1),
        member_count=7,
        identity={
            "embedding_layout": [
                {"name": "input_embeddings", "width": 3, "dtype": "torch.float32"}
            ],
            "generation_conditioning": "ordinary_prompt",
        },
        layouts_sha256="0" * 64,
        direction_calibration=None,
    )
    result = optimize_soft_prompt(
        runtime=cast(SoftPromptRuntime, runtime),
        prepared=prepared,
        config=AxBenchAdvSoftPromptConfig(
            max_steps=updates, learning_rate=learning_rate, universal_minibatch_size=3
        ),
        master_seed=813,
        group_key="independent-cross-coupled",
    )
    position = np.tile(prepared.embedding_weight[1].numpy().astype(np.float64), 3)
    moment = np.zeros_like(position)
    squared_moment = np.zeros_like(position)
    expected_iterates = [position.copy()]
    expected_batches = []
    for step in range(updates):
        epoch, offset = divmod(step, math.ceil(7 / 3))
        order = list(range(7))
        seed = hashlib.sha256(
            f"independent-cross-coupled|seed813|epoch{epoch}".encode()
        ).hexdigest()
        random.Random(seed).shuffle(order)
        indices = order[offset * 3 : (offset + 1) * 3]
        expected_batches.append(indices)
        derivatives = [runtime.hessians[i] @ (position - runtime.targets[i]) for i in indices]
        gradient = np.mean(derivatives, axis=0)
        moment = 0.9 * moment + 0.1 * gradient
        squared_moment = 0.999 * squared_moment + 0.001 * gradient**2
        first_unbiased = moment / (1 - 0.9 ** (step + 1))
        second_unbiased = squared_moment / (1 - 0.999 ** (step + 1))
        position = position - learning_rate * first_unbiased / (np.sqrt(second_unbiased) + 1e-8)
        expected_iterates.append(position.copy())

    actual = np.stack([value.numpy().reshape(-1) for value in runtime.iterates])
    expected = np.stack(expected_iterates)
    # The production master and gradients are FP32; this independent oracle is FP64.
    np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=8e-7)
    assert runtime.minibatches == expected_batches
    assert list(map(len, expected_batches[:3])) == [3, 3, 1]
    losses = [
        math.fsum(
            0.5 * (value - runtime.targets[i]) @ runtime.hessians[i] @ (value - runtime.targets[i])
            for i in range(7)
        )
        / 7
        for value in expected
    ]
    best = min(range(len(losses)), key=losses.__getitem__)
    assert best == expected_best_step
    assert result.best_step == best
    assert result.updates == updates
    assert t.equal(result.embeddings, runtime.iterates[best])
    assert result.objective == pytest.approx(losses[best], abs=1e-6, rel=2e-6)
