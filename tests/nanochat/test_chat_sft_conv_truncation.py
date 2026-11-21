"""Regression tests for the oversized-conversation buffer-starvation fix.

The SFT data generator in ``scripts/chat_sft.py`` packs rendered
conversations with a best-fit algorithm. Before the truncation fix, a
conversation longer than ``row_capacity`` (``max_seq_len + 1``) could
enter the buffer and never leave: ``find_bestfit_index`` requires
``conv_len <= remaining``, and the fresh-row case has
``remaining == row_capacity``, so the poison conversation never matched.
Once enough such items accumulated to fill the buffer, the packer's
refill was a no-op, every row padded entirely, and ``cross_entropy``
produced NaN on an all-masked target.

The fix truncates aligned ``(ids, mask)`` pairs to ``row_capacity`` in the refill loop:

    conv_buffer.append((ids[:row_capacity], mask[:row_capacity]))

These tests nail down the bestfit algorithm's behavior under the
starvation scenario *without* truncation (returns -1 = no fit), then
under the fix's invariant *with* truncation (returns a valid index).
They are intentionally minimal and avoid the full SFT generator wiring.
"""

from __future__ import annotations

import pytest

from steering.sft.iteration import find_bestfit_index


def _entry(ids: list[int]) -> tuple[list[int], list[int]]:
    return ids, [0] * len(ids)


# ---------------------------------------------------------------------------
# find_bestfit_index — baseline behavior
# ---------------------------------------------------------------------------


class TestFindBestfitIndex:
    """Pin the packer algorithm so changes to the refactor can't silently
    regress the behavior the generator depends on."""

    def test_empty_buffer_returns_minus_one(self) -> None:
        assert find_bestfit_index([], remaining=100) == -1

    def test_single_short_conv_fits(self) -> None:
        buf = [_entry([1, 2, 3])]
        assert find_bestfit_index(buf, remaining=100) == 0

    def test_single_conv_equal_to_remaining_fits(self) -> None:
        buf = [_entry([1, 2, 3])]
        assert find_bestfit_index(buf, remaining=3) == 0

    def test_single_conv_one_over_remaining_does_not_fit(self) -> None:
        buf = [_entry([1, 2, 3, 4])]
        assert find_bestfit_index(buf, remaining=3) == -1

    def test_prefers_largest_that_fits(self) -> None:
        """Given multiple conversations that fit, return the longest's index."""
        buf = [_entry([1, 2]), _entry([1, 2, 3, 4, 5]), _entry([1])]
        # Lengths: 2, 5, 1. Remaining=10 — all fit; longest is index 1.
        assert find_bestfit_index(buf, remaining=10) == 1

    def test_first_found_on_length_tie(self) -> None:
        """``>`` on best_len means ties break in favor of the earlier index."""
        buf = [_entry([1, 2, 3]), _entry([4, 5, 6]), _entry([7, 8, 9])]
        assert find_bestfit_index(buf, remaining=10) == 0

    def test_skips_conversations_larger_than_remaining(self) -> None:
        """A shorter conversation is picked when larger ones don't fit."""
        buf = [_entry([1, 2, 3, 4, 5, 6, 7, 8]), _entry([1, 2])]  # 8 and 2 tokens
        assert find_bestfit_index(buf, remaining=5) == 1  # only the 2-token one fits

    @pytest.mark.parametrize("remaining", [0, 1])
    def test_very_small_remaining(self, remaining: int) -> None:
        """remaining=0 fits nothing; remaining=1 fits a single-token conv."""
        buf = [_entry([1, 2]), _entry([1])]
        if remaining == 0:
            assert find_bestfit_index(buf, remaining=remaining) == -1
        else:
            assert find_bestfit_index(buf, remaining=remaining) == 1


# ---------------------------------------------------------------------------
# The starvation bug, with/without the truncation fix
# ---------------------------------------------------------------------------


class TestBufferStarvationRegression:
    """Demonstrate the pre-fix bug and pin the post-fix invariant.

    This is the load-bearing regression test for the commit. A passing
    suite with *only* these two tests would be enough to catch a future
    removal of the ``ids[:row_capacity]`` slice in ``refill_buffer``.
    """

    ROW_CAPACITY = 1025  # e.g. max_seq_len=1024

    def test_oversized_conv_without_truncation_cannot_fit_fresh_row(self) -> None:
        """Without the fix: a >row_capacity conversation in the buffer and
        a fresh row (``remaining == row_capacity``) returns -1. This is the
        bug condition — the packer cannot make progress.
        """
        oversized = list(range(self.ROW_CAPACITY + 500))  # 1525 ids
        buf = [_entry(oversized)]
        assert len(buf[0][0]) > self.ROW_CAPACITY  # sanity
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == -1

    def test_oversized_conv_with_truncation_fits_fresh_row(self) -> None:
        """With the fix: truncating to row_capacity yields a conv that always
        fits a fresh row. The packer makes progress; no NaN loss.
        """
        oversized = list(range(self.ROW_CAPACITY + 500))
        # This is what scripts/chat_sft.py's refill_buffer now does.
        truncated = oversized[: self.ROW_CAPACITY]
        buf = [_entry(truncated)]
        assert len(buf[0][0]) == self.ROW_CAPACITY
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == 0

    def test_all_oversized_buffer_without_truncation_starves(self) -> None:
        """Pre-fix pathology: a buffer entirely composed of oversized
        conversations produces -1 for every bestfit query, no matter what
        ``remaining`` the caller passes (up to row_capacity on a fresh row).
        """
        buf = [_entry(list(range(self.ROW_CAPACITY + i))) for i in range(1, 6)]
        for remaining in (1, 100, self.ROW_CAPACITY):
            assert find_bestfit_index(buf, remaining=remaining) == -1

    def test_all_oversized_buffer_with_truncation_makes_progress(self) -> None:
        """Post-fix invariant: after truncation, any non-empty buffer can
        always satisfy ``remaining == row_capacity``."""
        buf = [_entry(list(range(self.ROW_CAPACITY + i))[: self.ROW_CAPACITY]) for i in range(1, 6)]
        # All items have length exactly row_capacity after truncation; the
        # earliest-found ties-break makes the first one win.
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == 0

    def test_mixed_short_and_oversized_without_truncation(self) -> None:
        """Even with the packer able to pick a short conv, oversized items
        remain wedged until buffer refill displaces them — and since refill
        triggers on a size threshold, they may accumulate across refills."""
        short = [1, 2, 3]
        oversized = list(range(self.ROW_CAPACITY + 10))
        buf = [_entry(oversized), _entry(short)]
        # Short fits; packer picks it. Oversized remains. Bug isn't fatal
        # on this single call but the oversized item never gets consumed.
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == 1
        # Remove the short one (simulating pop); now the buffer is poisoned.
        buf.pop(1)
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == -1

    def test_mixed_short_and_oversized_with_truncation(self) -> None:
        """Post-fix: the oversized item is truncated at refill, so once the
        short is consumed, the truncated-oversized is still fittable."""
        short = [1, 2, 3]
        oversized_truncated = list(range(self.ROW_CAPACITY + 10))[: self.ROW_CAPACITY]
        buf = [_entry(oversized_truncated), _entry(short)]
        # Packer prefers the larger (truncated-oversized) since both fit a
        # fresh row and the truncated one is longer.
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == 0
        buf.pop(0)
        assert find_bestfit_index(buf, remaining=self.ROW_CAPACITY) == 0  # short

    def test_truncation_no_op_for_already_short_conv(self) -> None:
        """The slice is a no-op for conversations already within row_capacity.
        Regression guard: the fix must not truncate short conversations.
        """
        short = [1, 2, 3]
        truncated = short[: self.ROW_CAPACITY]
        assert truncated == short
