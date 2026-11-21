"""Committed SFT loop-position and no-post-final-prefetch regression tests."""

from __future__ import annotations

import pytest

from steering._common.lr_schedules import sft_flat_decay
from steering.sft.iteration import resolve_sft_loop_position


@pytest.mark.parametrize("num_iterations", [1, 2, 7])
def test_explicit_horizon_executes_exactly_requested_steps(num_iterations: int) -> None:
    completed = 0
    learning_rates: list[float] = []
    pulls = 0
    while True:
        position = resolve_sft_loop_position(
            completed_steps=completed,
            num_iterations=num_iterations,
            committed_consumed=0,
            dataset_size=10,
        )
        if position.last_step:
            break
        learning_rates.append(sft_flat_decay(position.progress))
        pulls += 3
        completed += 1
    assert completed == num_iterations
    assert pulls == num_iterations * 3
    assert learning_rates[0] == 1.0
    assert all(rate > 0 for rate in learning_rates)


def test_dataset_progress_uses_only_committed_consumption() -> None:
    produced = 9
    committed = 0
    before = resolve_sft_loop_position(
        completed_steps=0,
        num_iterations=-1,
        committed_consumed=committed,
        dataset_size=9,
    )
    assert before.progress == 0.0
    assert not before.last_step
    committed = produced
    after = resolve_sft_loop_position(
        completed_steps=1,
        num_iterations=-1,
        committed_consumed=committed,
        dataset_size=9,
    )
    assert after.progress == 1.0
    assert after.last_step


@pytest.mark.parametrize("bad", [0, -2, -10])
def test_invalid_iteration_horizons_fail(bad: int) -> None:
    with pytest.raises(ValueError, match="num_iterations"):
        resolve_sft_loop_position(
            completed_steps=0,
            num_iterations=bad,
            committed_consumed=0,
            dataset_size=1,
        )
