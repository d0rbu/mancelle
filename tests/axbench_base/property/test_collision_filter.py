"""Property tests for ``encoder_input.collision_filter``.

The collision filter resolves fids that appear in BOTH partitions
after a modifier has run. The resolution direction depends on the
method's ``mode``:

* ``mode == "positive"``: the ``active`` partition wins — colliding
  fids are removed from ``inactive``.
* ``mode == "negative"``: the ``inactive`` partition wins — colliding
  fids are removed from ``active``.

Three invariants must hold for all (active, inactive, mode) triples:

1. After filtering, the intersection of key-sets is empty —
   every fid appears in at most one partition.
2. Positive mode: every fid in the filtered ``active`` was in the
   original ``active`` (never invented), and every fid that was ONLY
   in the original ``inactive`` (not in ``active``) still appears in
   the filtered ``inactive`` with its original bucket.
3. Negative mode: mirror of (2).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from steering.eval.axbench_base.encoder_input import collision_filter
from steering.eval.axbench_base.types import SteeringMode

_partition_strategy = st.dictionaries(
    keys=st.integers(min_value=0, max_value=1000),
    values=st.integers(min_value=0, max_value=15),
    max_size=20,
)


# ---------------------------------------------------------------------------
# Invariant 1: filtered partitions have disjoint key-sets
# ---------------------------------------------------------------------------


@given(active=_partition_strategy, inactive=_partition_strategy)
def test_collision_filter_no_duplicates_across_partitions_positive(
    active: dict[int, int],
    inactive: dict[int, int],
) -> None:
    new_active, new_inactive = collision_filter(
        active, inactive, mode=SteeringMode.POSITIVE,
    )
    assert set(new_active.keys()) & set(new_inactive.keys()) == set()


@given(active=_partition_strategy, inactive=_partition_strategy)
def test_collision_filter_no_duplicates_across_partitions_negative(
    active: dict[int, int],
    inactive: dict[int, int],
) -> None:
    new_active, new_inactive = collision_filter(
        active, inactive, mode=SteeringMode.NEGATIVE,
    )
    assert set(new_active.keys()) & set(new_inactive.keys()) == set()


# ---------------------------------------------------------------------------
# Invariant 2: positive mode preserves active's bucket assignments
# ---------------------------------------------------------------------------


@given(active=_partition_strategy, inactive=_partition_strategy)
def test_collision_filter_positive_active_untouched(
    active: dict[int, int],
    inactive: dict[int, int],
) -> None:
    """Positive mode returns ``active`` unchanged (only ``inactive`` loses colliders)."""
    new_active, _ = collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    assert new_active == active


@given(active=_partition_strategy, inactive=_partition_strategy)
def test_collision_filter_positive_inactive_loses_only_colliders(
    active: dict[int, int],
    inactive: dict[int, int],
) -> None:
    """Positive mode: every fid in ``inactive`` that was NOT in ``active``
    still appears in the filtered ``inactive`` with its original bucket."""
    _, new_inactive = collision_filter(active, inactive, mode=SteeringMode.POSITIVE)
    for fid, bucket in inactive.items():
        if fid not in active:
            assert new_inactive.get(fid) == bucket


# ---------------------------------------------------------------------------
# Invariant 3: negative mode preserves inactive's bucket assignments
# ---------------------------------------------------------------------------


@given(active=_partition_strategy, inactive=_partition_strategy)
def test_collision_filter_negative_inactive_untouched(
    active: dict[int, int],
    inactive: dict[int, int],
) -> None:
    """Negative mode returns ``inactive`` unchanged (only ``active`` loses colliders)."""
    _, new_inactive = collision_filter(active, inactive, mode=SteeringMode.NEGATIVE)
    assert new_inactive == inactive


@given(active=_partition_strategy, inactive=_partition_strategy)
def test_collision_filter_negative_active_loses_only_colliders(
    active: dict[int, int],
    inactive: dict[int, int],
) -> None:
    """Negative mode: every fid in ``active`` that was NOT in ``inactive``
    still appears in the filtered ``active`` with its original bucket."""
    new_active, _ = collision_filter(active, inactive, mode=SteeringMode.NEGATIVE)
    for fid, bucket in active.items():
        if fid not in inactive:
            assert new_active.get(fid) == bucket
