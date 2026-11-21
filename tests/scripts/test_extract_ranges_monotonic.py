"""Unit tests for _extract_ranges_from_monotonic in precompute_token_boundaries.

Verifies that the vectorized contiguous-run extraction produces correct
tiling boundaries and rejects empty segments.
"""

from __future__ import annotations

import pytest
import torch as t

from scripts.preprocess.precompute_token_boundaries import _extract_ranges_from_monotonic


class TestExtractRangesMonotonic:
    """Test _extract_ranges_from_monotonic correctness."""

    def test_contiguous_segments(self):
        starts, ends = _extract_ranges_from_monotonic(t.tensor([0, 0, 0, 1, 1, 2, 2, 2]), 3, 8)
        assert starts.tolist() == [0, 3, 5]
        assert ends.tolist() == [3, 5, 8]
        assert starts.dtype == t.int32

    def test_single_segment(self):
        starts, ends = _extract_ranges_from_monotonic(t.tensor([0, 0, 0, 0]), 1, 4)
        assert starts.tolist() == [0]
        assert ends.tolist() == [4]

    def test_single_token(self):
        starts, ends = _extract_ranges_from_monotonic(t.tensor([0]), 1, 1)
        assert starts.tolist() == [0]
        assert ends.tolist() == [1]

    def test_many_segments(self):
        """16 segments, one token each."""
        starts, ends = _extract_ranges_from_monotonic(t.arange(16), 16, 16)
        assert starts.tolist() == list(range(16))
        assert ends.tolist() == list(range(1, 17))

    def test_uneven_runs(self):
        starts, ends = _extract_ranges_from_monotonic(t.tensor([0, 1, 1, 1, 2, 3, 3, 3, 3, 3]), 4, 10)
        assert starts.tolist() == [0, 1, 4, 5]
        assert ends.tolist() == [1, 4, 5, 10]

    def test_tiling_invariant(self):
        """Output must tile: end[i] == start[i+1], start[0] == 0, end[-1] == n_tokens."""
        starts, ends = _extract_ranges_from_monotonic(t.tensor([0, 0, 1, 1, 2, 2, 2, 3]), 4, 8)
        assert starts[0] == 0
        assert ends[-1] == 8
        for i in range(3):
            assert ends[i] == starts[i + 1]


class TestExtractRangesRejectsEmptySegments:
    """Verify that empty segments (zero tokens) raise ValueError."""

    def test_empty_tokens_raises(self):
        with pytest.raises(ValueError, match="Cannot extract ranges from 0 tokens"):
            _extract_ranges_from_monotonic(t.tensor([]), 3, 0)

    def test_segment_id_gap_raises(self):
        """Segment 1 has no tokens — should raise."""
        with pytest.raises(ValueError, match="Segments with no tokens"):
            _extract_ranges_from_monotonic(t.tensor([0, 0, 2, 2, 2]), 3, 5)

    def test_first_segment_empty_raises(self):
        with pytest.raises(ValueError, match="Segments with no tokens"):
            _extract_ranges_from_monotonic(t.tensor([1, 1, 2, 2]), 3, 4)

    def test_trailing_empty_segments_raises(self):
        with pytest.raises(ValueError, match="Segments with no tokens"):
            _extract_ranges_from_monotonic(t.tensor([0, 0, 1, 1]), 5, 4)

    def test_all_tokens_in_last_segment_raises(self):
        with pytest.raises(ValueError, match="Segments with no tokens"):
            _extract_ranges_from_monotonic(t.tensor([2, 2, 2, 2]), 3, 4)
