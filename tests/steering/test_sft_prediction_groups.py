"""Unit contracts for ``steered_prediction_groups``.

The helper is the whole of the steered FP prediction-group rule: segments of a
steered conversation's user turn keep the conversation's group, and from the
first assistant-turn segment on they get the next one. Everything the SFT
generator does with steered rows reduces to this function plus the append
order, so the invariant violations are pinned here rather than through a batch.
"""

from __future__ import annotations

import pytest

from steering.conditional.encoder_decoder.sft_prediction_groups import (
    steered_prediction_groups,
)

# A steered conversation renders as one user turn then one assistant turn, so
# there are always exactly two turn starts and the first one is source char 0.
TURNS = (0, 40)


class TestGroupAssignment:
    def test_every_segment_inside_the_user_turn_stays_in_the_base_group(self) -> None:
        assert steered_prediction_groups([0, 12, 39], TURNS, base_group=0) == [0, 0, 0]

    def test_the_group_increments_at_the_first_assistant_turn_segment(self) -> None:
        """The split lands on the segment that STARTS the assistant turn."""
        assert steered_prediction_groups([0, 20, 40, 55], TURNS, base_group=0) == [0, 0, 1, 1]

    def test_a_segment_starting_exactly_at_the_turn_boundary_is_assistant(self) -> None:
        """``char_start == turn_src_starts[1]`` belongs to turn 1, not turn 0.

        The boundary is the assistant turn's first character, so a half-open
        reading of the user turn is the only correct one: an off-by-one here
        would leave the perturbed response predictable from the user turn.
        """
        assert steered_prediction_groups([0, 40], TURNS, base_group=0) == [0, 1]
        assert steered_prediction_groups([0, 39], TURNS, base_group=0) == [0, 0]

    def test_the_last_segment_may_be_the_only_assistant_segment(self) -> None:
        """The truncation case: the extra target-only segment opens the group."""
        assert steered_prediction_groups([0, 18, 40], TURNS, base_group=0) == [0, 0, 1]

    def test_all_assistant_segments_share_the_incremented_group(self) -> None:
        """Group ids need only separate groups; ``base_group`` may go unused."""
        assert steered_prediction_groups([40, 52], TURNS, base_group=3) == [4, 4]

    def test_no_segments_returns_no_groups(self) -> None:
        assert steered_prediction_groups([], TURNS, base_group=7) == []


class TestBaseGroupOffset:
    @pytest.mark.parametrize("base_group", [0, 1, 5, 41])
    def test_the_base_group_offsets_every_returned_id(self, base_group: int) -> None:
        """A steered conversation packed after others starts at ITS group id."""
        groups = steered_prediction_groups([0, 20, 40, 60], TURNS, base_group=base_group)
        assert groups == [base_group, base_group, base_group + 1, base_group + 1]

    def test_the_offset_is_not_folded_into_the_turn_index(self) -> None:
        """Mutation killer: ``base_group + turn``, never ``base_group * turn``.

        With ``base_group=1`` the two forms agree on turn 1, so the user-turn
        segments are what discriminate them.
        """
        assert steered_prediction_groups([0, 40], TURNS, base_group=1) == [1, 2]


class TestInvariantViolations:
    def test_a_third_turn_raises_naming_the_segment_and_the_turn(self) -> None:
        """Three turns cannot happen for a validated steered row, so it is loud."""
        with pytest.raises(
            RuntimeError,
            match=r"segment char_start=90 maps to turn 2 after turn 1",
        ):
            steered_prediction_groups([0, 40, 90], (0, 40, 80), base_group=0)

    def test_a_decreasing_turn_sequence_raises(self) -> None:
        """Segments arrive in append order; going back to turn 0 is corruption."""
        with pytest.raises(
            RuntimeError,
            match=r"segment char_start=10 maps to turn 0 after turn 1",
        ):
            steered_prediction_groups([0, 40, 10], TURNS, base_group=0)

    def test_the_message_carries_both_full_sequences(self) -> None:
        """The raise must be debuggable without re-running the generator."""
        with pytest.raises(
            RuntimeError,
            match=r"turn starts \(0, 40, 80\).*segment starts \(0, 90\)",
        ):
            steered_prediction_groups([0, 90], (0, 40, 80), base_group=0)

    def test_a_segment_before_the_first_turn_start_raises(self) -> None:
        """``turn_src_starts[0]`` is 0 by construction; anything earlier is a bug."""
        with pytest.raises(RuntimeError, match=r"maps to turn -1 after turn 0"):
            steered_prediction_groups([-1], TURNS, base_group=0)


class TestTurnStartValidation:
    """The turn-start sequence is validated like every sibling primitive.

    ``segment_turn_delimited`` and ``turn_token_starts``
    (``steering/segmentation_primitives.py``) and ``turn_char_starts``
    (``steering/sft/turn_spans.py``) all reject a non-zero first element and a
    non-strictly-increasing sequence. Accepting them here would be worse than
    a wrong answer: a degenerate ``(0, 0)`` maps EVERY segment to turn 1, so the
    whole conversation lands in ``base_group + 1``, no split happens, and
    ``steered_without_assistant_segment`` is never incremented — a silent no-op
    instead of a loud failure. ``RuntimeError`` (not the siblings' ``ValueError``)
    keeps the generator's ``conditioned SFT example N: …`` recovery in scope.
    """

    def test_an_empty_turn_start_sequence_raises(self) -> None:
        with pytest.raises(RuntimeError, match=r"turn_src_starts must start at 0, got \(\)"):
            steered_prediction_groups([0], (), base_group=0)

    def test_a_first_turn_start_that_is_not_zero_raises(self) -> None:
        with pytest.raises(RuntimeError, match=r"turn_src_starts must start at 0, got \(5, 40\)"):
            steered_prediction_groups([5], (5, 40), base_group=0)

    def test_a_repeated_turn_start_raises_instead_of_collapsing_the_split(self) -> None:
        """``(0, 0)`` would put every segment in ``base_group + 1``, silently."""
        with pytest.raises(
            RuntimeError,
            match=r"turn_src_starts must be strictly increasing, got \(0, 0\)",
        ):
            steered_prediction_groups([0, 20, 40], (0, 0), base_group=0)

    def test_a_decreasing_turn_start_sequence_raises(self) -> None:
        with pytest.raises(
            RuntimeError,
            match=r"turn_src_starts must be strictly increasing, got \(0, 40, 20\)",
        ):
            steered_prediction_groups([0, 40], (0, 40, 20), base_group=0)

    def test_the_valid_sequence_the_generator_builds_is_accepted(self) -> None:
        """Non-vacuity: the two guards reject only degenerate inputs."""
        assert steered_prediction_groups([0, 40], TURNS, base_group=0) == [0, 1]
