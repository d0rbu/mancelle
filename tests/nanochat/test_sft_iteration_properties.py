"""Property tests for committed SFT loop-position invariants."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from steering.sft.iteration import resolve_sft_loop_position


@given(
    completed=st.integers(min_value=0, max_value=10_000_000),
    horizon=st.integers(min_value=1, max_value=1_000_000),
)
def test_explicit_progress_is_clamped_monotone_boundary(completed: int, horizon: int) -> None:
    current = resolve_sft_loop_position(
        completed_steps=completed,
        num_iterations=horizon,
        committed_consumed=0,
        dataset_size=1,
    )
    following = resolve_sft_loop_position(
        completed_steps=completed + 1,
        num_iterations=horizon,
        committed_consumed=0,
        dataset_size=1,
    )
    assert 0.0 <= current.progress <= following.progress <= 1.0
    assert current.last_step is (completed >= horizon)


@given(
    committed=st.integers(min_value=0, max_value=10_000_000),
    dataset_size=st.integers(min_value=1, max_value=1_000_000),
)
def test_dataset_progress_is_clamped_monotone_boundary(
    committed: int,
    dataset_size: int,
) -> None:
    current = resolve_sft_loop_position(
        completed_steps=0,
        num_iterations=-1,
        committed_consumed=committed,
        dataset_size=dataset_size,
    )
    following = resolve_sft_loop_position(
        completed_steps=0,
        num_iterations=-1,
        committed_consumed=committed + 1,
        dataset_size=dataset_size,
    )
    assert 0.0 <= current.progress <= following.progress <= 1.0
    assert current.last_step is (committed >= dataset_size)
