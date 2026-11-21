"""Tests for token-to-segment assignment via character overlap."""

import torch as t

from steering.segmentation import assign_tokens_to_segments


class TestAssignTokensToSegments:
    def test_exact_match(self):
        """Tokens and segments align perfectly."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([0, 3]),
            token_char_ends=t.tensor([3, 6]),
            seg_char_starts=t.tensor([0, 3]),
            seg_char_ends=t.tensor([3, 6]),
        )
        assert result.tolist() == [0, 1]

    def test_user_example(self):
        """The example from the feature request.

        Extraction: "h" "ell" "o" " w" "or" "ld" -> segs "hell" "o w" "orld"
          seg boundaries: [0,4), [4,7), [7,11)
        Training: "hel" "lo w" "orld"
          token boundaries: [0,3), [3,7), [7,11)
        """
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([0, 3, 7]),
            token_char_ends=t.tensor([3, 7, 11]),
            seg_char_starts=t.tensor([0, 4, 7]),
            seg_char_ends=t.tensor([4, 7, 11]),
        )
        # "hel" [0,3): seg0 [0,4) overlap=3, seg1 overlap=0 -> seg 0
        # "lo w" [3,7): seg0 [3,4) overlap=1, seg1 [4,7) overlap=3 -> seg 1
        # "orld" [7,11): seg2 [7,11) overlap=4 -> seg 2
        assert result.tolist() == [0, 1, 2]

    def test_token_spans_two_segments_equally(self):
        """Token equally split between two segments -> picks earlier one."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([2]),
            token_char_ends=t.tensor([6]),
            seg_char_starts=t.tensor([0, 4]),
            seg_char_ends=t.tensor([4, 8]),
        )
        # overlap with seg0: 2, seg1: 2 -> tie -> argmax picks first
        assert result.tolist() == [0]

    def test_single_segment(self):
        """All tokens map to the only segment."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([0, 5, 10]),
            token_char_ends=t.tensor([5, 10, 15]),
            seg_char_starts=t.tensor([0]),
            seg_char_ends=t.tensor([15]),
        )
        assert result.tolist() == [0, 0, 0]

    def test_monotonic_output(self):
        """Output should be non-decreasing (segments are sequential)."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([0, 2, 5, 8, 12]),
            token_char_ends=t.tensor([2, 5, 8, 12, 15]),
            seg_char_starts=t.tensor([0, 6, 11]),
            seg_char_ends=t.tensor([6, 11, 15]),
        )
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    def test_single_token(self):
        """Single token maps to correct segment."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([5]),
            token_char_ends=t.tensor([8]),
            seg_char_starts=t.tensor([0, 4, 8]),
            seg_char_ends=t.tensor([4, 8, 12]),
        )
        # Token [5,8) overlaps seg1 [4,8) by 3, seg0 by 0, seg2 by 0
        assert result.tolist() == [1]

    def test_many_tokens_few_segments(self):
        """Many small tokens across few large segments."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([0, 1, 2, 3, 4, 5]),
            token_char_ends=t.tensor([1, 2, 3, 4, 5, 6]),
            seg_char_starts=t.tensor([0, 3]),
            seg_char_ends=t.tensor([3, 6]),
        )
        assert result.tolist() == [0, 0, 0, 1, 1, 1]

    def test_token_partially_outside_all_segments(self):
        """Token extends beyond last segment."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([8]),
            token_char_ends=t.tensor([12]),
            seg_char_starts=t.tensor([0, 5]),
            seg_char_ends=t.tensor([5, 10]),
        )
        # Token [8,12) overlaps seg1 [5,10) by 2
        assert result.tolist() == [1]

    def test_output_dtype(self):
        """Output should be int64 tensor."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([0, 3]),
            token_char_ends=t.tensor([3, 6]),
            seg_char_starts=t.tensor([0, 3]),
            seg_char_ends=t.tensor([3, 6]),
        )
        assert isinstance(result, t.Tensor)
        assert result.dtype == t.int64

    def test_empty_tokens(self):
        """No tokens should return empty tensor."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([], dtype=t.int64),
            token_char_ends=t.tensor([], dtype=t.int64),
            seg_char_starts=t.tensor([0, 5]),
            seg_char_ends=t.tensor([5, 10]),
        )
        assert len(result) == 0

    def test_offset_segments(self):
        """Segments that don't start at 0 (e.g., packed documents)."""
        result = assign_tokens_to_segments(
            token_char_starts=t.tensor([100, 105, 110]),
            token_char_ends=t.tensor([105, 110, 115]),
            seg_char_starts=t.tensor([100, 108]),
            seg_char_ends=t.tensor([108, 115]),
        )
        # [100,105) -> seg0 overlap=5; [105,110) -> seg0 overlap=3, seg1 overlap=2 -> seg0;
        # [110,115) -> seg1 overlap=5
        assert result.tolist() == [0, 0, 1]
