"""Dependency-light token and turn segmentation primitives."""

import bisect
import itertools
from collections.abc import Sequence
from enum import StrEnum

from beartype import beartype

from steering.refined_types import (
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
)


class SegmentMode(StrEnum):
    """How to segment a token sequence for feature extraction."""

    fixed_tokens = "fixed_tokens"  # Fixed-size non-overlapping token windows
    # Fixed-size windows that never cross logical-turn boundaries.
    turn_delimited_fixed_tokens = "turn_delimited_fixed_tokens"


@beartype
def _segment_fixed(
    n: int,
    segment_size: PositiveInt,
    token_offset: NonNegativeIntLike = 0,
) -> list[tuple[int, int]]:
    """Split into fixed-size non-overlapping windows.

    Args:
        n: Total number of tokens.
        segment_size: Number of tokens per segment.
        token_offset: Start segmentation at this token index.

    Raises:
        BeartypeCallHintParamViolation: An argument violates its annotated
            contract, including a non-positive ``segment_size`` or negative
            ``token_offset``.
    """
    boundaries = list(range(token_offset, n, segment_size)) + [n]
    return list(itertools.pairwise(boundaries))


@beartype
def segment_turn_delimited(
    turn_starts: Sequence[int],
    n: PositiveIntLike,
    segment_size: PositiveIntLike,
) -> list[tuple[int, int]]:
    """Chunk each logical turn independently into fixed-size token windows."""
    if not turn_starts:
        raise ValueError("turn_starts must be non-empty")
    if turn_starts[0] != 0:
        raise ValueError("turn_starts[0] must be 0")
    if any(right <= left for left, right in itertools.pairwise(turn_starts)):
        raise ValueError("turn_starts must be strictly increasing")
    if any(start >= n for start in turn_starts):
        raise ValueError("turn_starts values must be less than n")

    size = PositiveInt.parse(segment_size)
    turn_ends = [*turn_starts[1:], n]
    return [
        (start, min(start + size, turn_end))
        for turn_start, turn_end in zip(turn_starts, turn_ends, strict=True)
        for start in range(turn_start, turn_end, size)
    ]


@beartype
def turn_token_starts(
    char_offsets: Sequence[int],
    turn_char_starts: Sequence[int],
    *,
    bos_char_len: NonNegativeIntLike,
) -> list[int]:
    """Map BOS-free character turn boundaries to BOS-inclusive token starts.

    Turn zero absorbs the BOS token and starts at token zero. For every later
    retained turn boundary ``B_j``, the first token whose character start is at
    or after ``B_j`` begins the turn, so a token straddling the boundary remains
    in the previous turn. Consequently, every window produced from these starts
    for turn ``j`` has a character start in ``[B_j, B_{j+1})``, where ``B_0 = 0``
    and ``B_j = turn_char_starts[j] + bos_char_len`` for ``j >= 1``.

    The returned list can be shorter than ``turn_char_starts`` when trailing
    boundaries fall beyond the kept tokens; callers must not zip it positionally
    against untruncated per-turn data.
    """
    if not char_offsets:
        raise ValueError("char_offsets must be non-empty")
    if char_offsets[0] != 0:
        raise ValueError("char_offsets[0] must be 0")
    if any(right < left for left, right in itertools.pairwise(char_offsets)):
        raise ValueError("char_offsets must be non-decreasing")
    if not turn_char_starts:
        raise ValueError("turn_char_starts must be non-empty")
    if turn_char_starts[0] != 0:
        raise ValueError("turn_char_starts[0] must be 0")
    if any(right <= left for left, right in itertools.pairwise(turn_char_starts)):
        raise ValueError("turn_char_starts must be strictly increasing")

    n = len(char_offsets) - 1
    result = [0]
    for turn_index, turn_char_start in enumerate(turn_char_starts[1:], start=1):
        boundary = turn_char_start + bos_char_len
        token_start = bisect.bisect_left(char_offsets, boundary)
        if token_start >= n:
            break
        if token_start <= result[-1]:
            raise ValueError(f"turn {turn_index} received zero tokens")
        result.append(token_start)
    return result


__all__ = ["_segment_fixed", "segment_turn_delimited", "turn_token_starts"]
