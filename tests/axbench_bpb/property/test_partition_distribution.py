"""Hypothesis: hash-based partitioning is exhaustive + approximately uniform."""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_bpb.bpb_scoring import is_mine
from steering.eval.axbench_bpb.types import WorkUnit


@given(
    world_size=st.sampled_from([1, 2, 4, 8, 16]),
    n_units=st.integers(min_value=100, max_value=2000),
)
def test_partitioning_exhaustive(world_size, n_units):
    """Every unit belongs to exactly one rank; the union covers all units."""
    units = [
        WorkUnit(f"m{i % 3}", i, f"sid{i}", f"c{i % 7}")
        for i in range(n_units)
    ]
    covered = 0
    for wu in units:
        owners = [r for r in range(world_size) if is_mine(wu, rank=r, world_size=world_size)]
        assert len(owners) == 1, f"{wu}: owned by {len(owners)} ranks at ws={world_size}"
        covered += 1
    assert covered == n_units


@given(
    world_size=st.sampled_from([2, 4, 8, 16]),
    n_units=st.integers(min_value=500, max_value=2000),
)
def test_partitioning_approximately_uniform(world_size, n_units):
    """SHA256 hash distribution is loosely balanced across ranks.

    Tolerance: 40% of expected (loose to avoid flakes when n_units/world_size is small).
    """
    units = [
        WorkUnit(f"m{i % 5}", i, f"sid{i}", f"c{i % 7}")
        for i in range(n_units)
    ]
    counts = [0] * world_size
    for wu in units:
        for r in range(world_size):
            if is_mine(wu, rank=r, world_size=world_size):
                counts[r] += 1
                break
    expected = n_units / world_size
    for c in counts:
        assert abs(c - expected) < 0.4 * expected, (
            f"rank count {c} outside 40% of expected {expected} at ws={world_size}"
        )
