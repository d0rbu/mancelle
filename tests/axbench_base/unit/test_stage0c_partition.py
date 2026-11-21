"""Tests for Stage 0c partition primitives (PR-4)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.stage0c_partition import (
    is_my_candidate as _typed_is_my_candidate,
)
from steering.eval.axbench_base.stage0c_partition import (
    my_candidate_block as _typed_my_candidate_block,
)
from steering.refined_types import NonNegativeInt, PositiveInt


def is_my_candidate(idx: int, *, rank: int, world_size: int) -> bool:
    return _typed_is_my_candidate(
        NonNegativeInt.parse(idx),
        rank=NonNegativeInt.parse(rank),
        world_size=PositiveInt.parse(world_size),
    )


def my_candidate_block(
    block_idx: int,
    B: int,
    *,
    rank: int,
    world_size: int,
    n_candidates: int | None = None,
) -> Iterator[int]:
    return _typed_my_candidate_block(
        NonNegativeInt.parse(block_idx),
        PositiveInt.parse(B),
        rank=NonNegativeInt.parse(rank),
        world_size=PositiveInt.parse(world_size),
        n_candidates=(
            None if n_candidates is None else NonNegativeInt.parse(n_candidates)
        ),
    )


@pytest.mark.parametrize("rank,world_size,expected", [
    (0, 1, list(range(20))),
    (0, 2, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]),
    (1, 2, [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]),
    (0, 4, [0, 4, 8, 12, 16]),
    (1, 4, [1, 5, 9, 13, 17]),
    (2, 4, [2, 6, 10, 14, 18]),
    (3, 4, [3, 7, 11, 15, 19]),
])
def test_is_my_candidate_modulo_partition(rank, world_size, expected):
    actual = [i for i in range(20) if is_my_candidate(i, rank=rank, world_size=world_size)]
    assert actual == expected


def test_is_my_candidate_full_coverage_disjoint_at_all_world_sizes():
    """For each world_size in {1..16}, every candidate index is owned by
    EXACTLY one rank — total cover, no duplicates."""
    for ws in [1, 2, 3, 4, 5, 8, 16]:
        owned: list[int] = []
        for i in range(100):
            owners = [r for r in range(ws) if is_my_candidate(i, rank=r, world_size=ws)]
            assert len(owners) == 1, (
                f"At ws={ws}, idx={i}: expected exactly 1 owner, got {len(owners)}: {owners}"
            )
            owned.append(owners[0])
        assert sorted(owned) == sorted(owned), "(trivially)"  # placeholder no-op
        # Rank ids in valid range
        assert all(0 <= o < ws for o in owned), f"At ws={ws}: invalid rank in {owned}"


def test_is_my_candidate_invalid_rank_raises():
    with pytest.raises(ValueError, match="rank .* out of range"):
        is_my_candidate(0, rank=2, world_size=2)


def test_my_candidate_block_layout():
    """B=4, W=2: block 0 covers [0..7]; rank 0 takes [0,2,4,6]; rank 1
    takes [1,3,5,7]."""
    assert list(my_candidate_block(0, 4, rank=0, world_size=2)) == [0, 2, 4, 6]
    assert list(my_candidate_block(0, 4, rank=1, world_size=2)) == [1, 3, 5, 7]
    # Block 1 covers [8..15]
    assert list(my_candidate_block(1, 4, rank=0, world_size=2)) == [8, 10, 12, 14]
    assert list(my_candidate_block(1, 4, rank=1, world_size=2)) == [9, 11, 13, 15]


def test_my_candidate_block_layout_world_size_4():
    """B=2, W=4: block 0 covers [0..7]; rank 0 takes [0,4]; rank 1 takes
    [1,5]; etc."""
    for r in range(4):
        expected = [r, r + 4]
        assert list(my_candidate_block(0, 2, rank=r, world_size=4)) == expected


def test_my_candidate_block_clamps_to_pool_size():
    """block ranges past pool end are clipped at n_candidates."""
    # n_candidates=10, B=4, W=2 → block 1 normally yields [8,10,12,14] for rank 0
    # but with n_candidates=10, only [8] is yielded
    block = list(my_candidate_block(1, 4, rank=0, world_size=2, n_candidates=10))
    assert block == [8]


def test_my_candidate_block_complete_cover_across_blocks():
    """Iterating blocks 0..N until exhaustion covers every owned index
    (no gaps, no duplicates) for that rank."""
    n_candidates = 30
    B = 4
    for ws in [1, 2, 3, 4]:
        for r in range(ws):
            owned_via_blocks: list[int] = []
            block_idx = 0
            while True:
                this_block = list(my_candidate_block(
                    block_idx, B, rank=r, world_size=ws, n_candidates=n_candidates,
                ))
                if not this_block:
                    break
                owned_via_blocks.extend(this_block)
                block_idx += 1
                if block_idx > 100:  # sanity
                    raise AssertionError("infinite loop")
            owned_via_modulo = [
                i for i in range(n_candidates)
                if is_my_candidate(i, rank=r, world_size=ws)
            ]
            assert sorted(owned_via_blocks) == owned_via_modulo, (
                f"At ws={ws}, rank={r}: block iteration mismatch with modulo"
            )


def test_my_candidate_block_invalid_args():
    with pytest.raises(ValueError, match="invalid"):
        list(my_candidate_block(0, 4, rank=2, world_size=2))


_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)
_INVALID_POSITIVE_INT = st.one_of(
    st.integers(max_value=0),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)
_INVALID_OPTIONAL_NON_NEGATIVE_INT = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
)


@settings(max_examples=500, derandomize=True)
@given(
    invalid=st.one_of(
        st.tuples(st.just("candidate"), st.just("idx"), _INVALID_NON_NEGATIVE_INT),
        st.tuples(st.just("candidate"), st.just("rank"), _INVALID_NON_NEGATIVE_INT),
        st.tuples(st.just("candidate"), st.just("world_size"), _INVALID_POSITIVE_INT),
        st.tuples(st.just("block"), st.just("block_idx"), _INVALID_NON_NEGATIVE_INT),
        st.tuples(st.just("block"), st.just("B"), _INVALID_POSITIVE_INT),
        st.tuples(st.just("block"), st.just("rank"), _INVALID_NON_NEGATIVE_INT),
        st.tuples(st.just("block"), st.just("world_size"), _INVALID_POSITIVE_INT),
        st.tuples(
            st.just("block"),
            st.just("n_candidates"),
            _INVALID_OPTIONAL_NON_NEGATIVE_INT,
        ),
    )
)
def test_partition_primitives_reject_every_invalid_scalar(
    invalid: tuple[str, str, Any],
) -> None:
    function_name, field, value = invalid
    if function_name == "candidate":
        kwargs: dict[str, Any] = {
            "idx": NonNegativeInt.parse(3),
            "rank": NonNegativeInt.parse(1),
            "world_size": PositiveInt.parse(2),
        }
        kwargs[field] = value
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            _typed_is_my_candidate(**kwargs)
        return

    kwargs = {
        "block_idx": NonNegativeInt.parse(1),
        "B": PositiveInt.parse(4),
        "rank": NonNegativeInt.parse(1),
        "world_size": PositiveInt.parse(2),
        "n_candidates": NonNegativeInt.parse(30),
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        list(_typed_my_candidate_block(**kwargs))
