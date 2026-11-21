"""Feature-prediction group ids for steered SFT conversations.

``seg_doc_id`` is the FP *prediction-group* id: consecutive encoder segments may
predict each other only inside one group.  A group boundary sits at every packed
conversation boundary and, for a steered conversation, ALSO at its first
assistant-turn segment — the steered response's features are deliberately
perturbed, so the head must never learn to predict them from the unperturbed
user turn.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from itertools import pairwise

from beartype import beartype


@beartype
def steered_prediction_groups(
    seg_char_starts: Sequence[int],
    turn_src_starts: Sequence[int],
    *,
    base_group: int,
) -> list[int]:
    """Return one prediction-group id per segment, in generator append order.

    ``seg_char_starts`` are the segments' source-coordinate starts in the order
    the generator appends them; ``turn_src_starts`` are the source-coordinate
    starts of the conversation's logical turns (a steered conversation has
    exactly two: one user turn then one assistant turn).  Segments of the user
    turn keep ``base_group``; from the first assistant-turn segment on they
    carry ``base_group + 1``.

    Raises:
        RuntimeError: ``turn_src_starts`` does not start at 0 or is not
            strictly increasing, a segment maps outside the two turns, or the
            segments' turn indices are not non-decreasing.  All are packing
            invariants the generator establishes, not operator input.
    """
    if not turn_src_starts or turn_src_starts[0] != 0:
        raise RuntimeError(f"turn_src_starts must start at 0, got {tuple(turn_src_starts)}")
    if any(right <= left for left, right in pairwise(turn_src_starts)):
        raise RuntimeError(
            f"turn_src_starts must be strictly increasing, got {tuple(turn_src_starts)}"
        )
    groups: list[int] = []
    previous_turn = 0
    for char_start in seg_char_starts:
        turn_index = bisect_right(turn_src_starts, char_start) - 1
        if turn_index not in (0, 1) or turn_index < previous_turn:
            raise RuntimeError(
                "steered SFT segments must map to turn 0 then turn 1 in order; "
                f"segment char_start={char_start} maps to turn {turn_index} after "
                f"turn {previous_turn} (turn starts {tuple(turn_src_starts)}, "
                f"segment starts {tuple(seg_char_starts)})"
            )
        previous_turn = turn_index
        groups.append(base_group + turn_index)
    return groups


__all__ = ["steered_prediction_groups"]
