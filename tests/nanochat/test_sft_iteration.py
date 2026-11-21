"""Unit tests for committed SFT loop position and save boundaries."""

from __future__ import annotations

import pytest

from steering.sft.iteration import resolve_sft_loop_position, should_save_checkpoint


@pytest.mark.parametrize("completed", [0, 1, 4, 5, 9])
def test_explicit_horizon_uses_completed_optimizer_steps(completed: int) -> None:
    position = resolve_sft_loop_position(
        completed_steps=completed,
        num_iterations=5,
        committed_consumed=999,
        dataset_size=1000,
    )
    assert position.last_step is (completed >= 5)
    assert position.progress == pytest.approx(min(1.0, completed / 5))


@pytest.mark.parametrize("committed", [0, 1, 7, 8, 20])
def test_dataset_horizon_uses_committed_conversations(committed: int) -> None:
    position = resolve_sft_loop_position(
        completed_steps=999,
        num_iterations=-1,
        committed_consumed=committed,
        dataset_size=8,
    )
    assert position.last_step is (committed >= 8)
    assert position.progress == pytest.approx(min(1.0, committed / 8))


@pytest.mark.parametrize("bad", [0, -2, -99])
def test_invalid_iteration_horizon_fails(bad: int) -> None:
    with pytest.raises(ValueError, match="num_iterations"):
        resolve_sft_loop_position(
            completed_steps=0,
            num_iterations=bad,
            committed_consumed=0,
            dataset_size=1,
        )


def test_invalid_committed_state_fails() -> None:
    with pytest.raises(ValueError, match="completed_steps"):
        resolve_sft_loop_position(
            completed_steps=-1,
            num_iterations=1,
            committed_consumed=0,
            dataset_size=1,
        )
    with pytest.raises(ValueError, match="committed_consumed"):
        resolve_sft_loop_position(
            completed_steps=0,
            num_iterations=-1,
            committed_consumed=-1,
            dataset_size=1,
        )
    with pytest.raises(ValueError, match="dataset_size"):
        resolve_sft_loop_position(
            completed_steps=0,
            num_iterations=-1,
            committed_consumed=0,
            dataset_size=0,
        )


@pytest.mark.parametrize(
    "is_master,is_dry_run,is_last_step,step,expected",
    [
        (False, False, True, 1, False),
        (True, True, True, 1, False),
        (True, False, True, 1, True),
        (True, False, False, 0, False),
        (True, False, False, 10, True),
    ],
)
def test_checkpoint_boundaries(
    is_master: bool,
    is_dry_run: bool,
    is_last_step: bool,
    step: int,
    expected: bool,
) -> None:
    assert (
        should_save_checkpoint(
            is_master=is_master,
            is_dry_run=is_dry_run,
            is_last_step=is_last_step,
            step=step,
            save_every=10,
        )
        is expected
    )
