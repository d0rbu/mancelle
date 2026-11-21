"""Property-based tests for ``tasks.common.TaskMixture``.

Contracts (per Phase 0 map §3.3):
- Builds full ``(task_idx, local_idx)`` index map; no lazy eval.
- ``len(mixture) == sum(len(t) for t in tasks)``.
- Every ``(task_idx, local_idx)`` pair appears exactly once.
- Shuffle is deterministic via ``random.Random(42)``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from tasks.common import Task, TaskMixture


class _TinyTask(Task):
    """Length-only stub task for testing TaskMixture wiring."""

    def __init__(self, length: int) -> None:
        super().__init__()
        self._length = length

    def num_examples(self) -> int:
        return self._length

    def get_example(self, index: int) -> tuple[str, int]:
        # Encode position so tests can verify routing.
        return ("tiny", index)


@given(
    lengths=st.lists(st.integers(min_value=0, max_value=20), min_size=1, max_size=8),
)
@settings(derandomize=True, max_examples=200)
def test_len_equals_sum_of_subtask_lens(lengths):
    tasks = [_TinyTask(n) for n in lengths]
    mix = TaskMixture(tasks)
    assert len(mix) == sum(lengths)


@given(
    lengths=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5),
)
@settings(derandomize=True, max_examples=200)
def test_index_map_covers_every_pair_exactly_once(lengths):
    tasks = [_TinyTask(n) for n in lengths]
    mix = TaskMixture(tasks)
    # Every (task_idx, local_idx) where task_idx in range(len(tasks))
    # and local_idx in range(lengths[task_idx]) must appear.
    expected = set()
    for ti, n in enumerate(lengths):
        for li in range(n):
            expected.add((ti, li))
    actual = set(mix.index_map)
    assert actual == expected
    # No duplicates.
    assert len(mix.index_map) == len(expected)


@given(
    lengths=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=5),
)
@settings(derandomize=True, max_examples=200)
def test_index_map_is_deterministic(lengths):
    """Two TaskMixture builds with identical inputs produce identical
    shuffled index_maps (Random(42) seed)."""
    tasks_a = [_TinyTask(n) for n in lengths]
    tasks_b = [_TinyTask(n) for n in lengths]
    mix_a = TaskMixture(tasks_a)
    mix_b = TaskMixture(tasks_b)
    assert mix_a.index_map == mix_b.index_map


def test_get_example_routes_through_index_map():
    """``mixture[i]`` returns ``tasks[task_idx][local_idx]`` per index_map[i]."""
    tasks = [_TinyTask(3), _TinyTask(2)]
    mix = TaskMixture(tasks)
    # The shuffled order is fixed by Random(42); we just verify each
    # call routes to the expected (task_idx, local_idx).
    for i in range(len(mix)):
        task_idx, local_idx = mix.index_map[i]
        assert mix[i] == ("tiny", local_idx)


def test_oversampling_via_repeated_task_in_list():
    """Passing the same task twice oversamples it 2x."""
    t = _TinyTask(3)
    mix = TaskMixture([t, t])
    assert len(mix) == 6
    # Each (task_idx, local_idx) appears exactly once.
    seen = set(mix.index_map)
    assert seen == {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_zero_length_subtasks_are_skipped():
    mix = TaskMixture([_TinyTask(0), _TinyTask(2)])
    assert len(mix) == 2
    actual = set(mix.index_map)
    assert actual == {(1, 0), (1, 1)}
