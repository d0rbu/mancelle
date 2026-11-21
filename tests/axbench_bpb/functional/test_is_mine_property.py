"""Hypothesis property tests for the is_mine DDP partition function.

Invariants:
1. Disjointness: every WorkUnit is owned by EXACTLY one rank in
   [0, world_size).
2. Determinism: is_mine returns the same value across calls for the
   same (unit, rank, world_size).
3. World-size invariance: shrinking or growing world_size redistributes
   units but every unit still has exactly one owner.
"""
from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb.bpb_scoring import is_mine
from steering.eval.axbench_bpb.types import WorkUnit


def _work_unit_strategy() -> st.SearchStrategy[WorkUnit]:
    """Generate diverse WorkUnits across the schema."""
    return st.builds(
        WorkUnit,
        method_name=st.text(
            alphabet=string.ascii_lowercase + "_", min_size=1, max_size=20,
        ),
        concept_id=st.integers(min_value=0, max_value=10000),
        sample_id=st.text(
            alphabet=string.ascii_lowercase + string.digits,
            min_size=1, max_size=20,
        ),
        coefficient_str=st.one_of(
            st.just("None"),
            st.integers(min_value=-1000, max_value=1000).map(str),
            st.floats(
                min_value=-1.0, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            ).map(str),
        ),
    )


@given(unit=_work_unit_strategy(), world_size=st.integers(min_value=1, max_value=16))
@settings(max_examples=200)
def test_is_mine_assigns_to_exactly_one_rank(unit: WorkUnit, world_size: int) -> None:
    """Sum of is_mine(unit, rank, world_size) over rank in [0, world_size) == 1."""
    owners = sum(
        is_mine(unit, rank=r, world_size=world_size) for r in range(world_size)
    )
    assert owners == 1, (
        f"WorkUnit {unit} owned by {owners} ranks (expected 1) at world_size={world_size}"
    )


@given(unit=_work_unit_strategy(), world_size=st.integers(min_value=1, max_value=16))
@settings(max_examples=100)
def test_is_mine_is_deterministic(unit: WorkUnit, world_size: int) -> None:
    """is_mine returns the same value for the same input across calls."""
    rank = 0  # arbitrary
    a = is_mine(unit, rank=rank, world_size=world_size)
    b = is_mine(unit, rank=rank, world_size=world_size)
    assert a == b


@given(
    units=st.lists(
        _work_unit_strategy(),
        min_size=10, max_size=50,
        unique_by=lambda u: (u.method_name, u.concept_id, u.sample_id, u.coefficient_str),
    ),
)
@settings(max_examples=20)
def test_is_mine_partition_covers_all_units_for_world_size_4(units: list[WorkUnit]) -> None:
    """For world_size=4, every unit is owned by exactly one rank."""
    world_size = 4
    for unit in units:
        owners = [
            r for r in range(world_size)
            if is_mine(unit, rank=r, world_size=world_size)
        ]
        assert len(owners) == 1, (
            f"Unit {unit} has {len(owners)} owners (expected 1) at world_size=4"
        )
