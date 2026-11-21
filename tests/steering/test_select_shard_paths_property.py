"""Property-based tests for select_shard_paths — slice 01 Phase 1 audit.

Pins map suspected-bug #14 + map §7 coverage gap "Rank-aware partition no
duplication". The function's contract: every shard in [start_shard, num_shards)
is processed by exactly one rank; no cross-rank duplication.

Hypothesis: deterministic with 500 examples per invariant.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.parquet_io import select_shard_paths as _select_shard_paths
from steering.refined_types import NonNegativeInt, PositiveInt

_n_shards_total = st.integers(min_value=0, max_value=64)
_world_size = st.integers(min_value=1, max_value=16)
_start_shard = st.integers(min_value=0, max_value=64)


def _make_paths(n: int) -> list[str]:
    return [f"shard_{i:05d}.parquet" for i in range(n)]


def select_shard_paths(
    parquet_paths: list[str],
    num_shards: int,
    start_shard: int,
    rank: int,
    world_size: int,
) -> list[str]:
    """Refine generated valid scalars before exercising the production seam."""
    return _select_shard_paths(
        parquet_paths,
        NonNegativeInt.parse(num_shards),
        NonNegativeInt.parse(start_shard),
        NonNegativeInt.parse(rank),
        PositiveInt.parse(world_size),
    )


@settings(derandomize=True, max_examples=500)
@given(
    n_total=_n_shards_total,
    world_size=_world_size,
    start_shard=_start_shard,
    extra=st.integers(min_value=0, max_value=64),
)
def test_partition_disjoint_and_exhaustive(
    n_total: int, world_size: int, start_shard: int, extra: int
):
    """Per the function contract: every shard in [start_shard, num_shards)
    appears in exactly one rank's selection; no shard appears twice."""
    paths = _make_paths(n_total)
    num_shards = start_shard + extra  # may exceed n_total; function clamps
    expected_window_end = min(num_shards, n_total)
    expected_window = paths[start_shard:expected_window_end]

    union: list[str] = []
    for rank in range(world_size):
        sel = select_shard_paths(paths, num_shards, start_shard, rank, world_size)
        union.extend(sel)

    # Exhaustive: every path in the window is assigned to some rank.
    assert sorted(union) == sorted(expected_window)
    # Disjoint: no path appears twice.
    assert len(union) == len(set(union))


@settings(derandomize=True, max_examples=500)
@given(
    n_total=_n_shards_total,
    world_size=_world_size,
    start_shard=_start_shard,
    extra=st.integers(min_value=0, max_value=64),
    rank_offset=st.integers(min_value=0, max_value=16),
)
def test_rank_out_of_range_raises(
    n_total: int,
    world_size: int,
    start_shard: int,
    extra: int,
    rank_offset: int,
):
    """Every non-negative rank at or above world_size must raise ValueError."""
    paths = _make_paths(n_total)
    num_shards = start_shard + extra
    rank = world_size + rank_offset
    with pytest.raises(ValueError):
        select_shard_paths(paths, num_shards, start_shard, rank, world_size)
