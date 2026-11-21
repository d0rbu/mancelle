"""Tests for the shared encdec test helpers themselves.

Pins the contract of helpers used across many test files so a change here
that breaks downstream tests fails first at this single file.
"""

import pytest

from tests.encdec_test_helpers import _make_test_shard


class TestMakeTestShardVariableSegs:
    """_make_test_shard with segs_per_row: list[int] produces variable n_seg."""

    def test_int_default_backward_compat(self):
        """segs_per_row as int still produces uniform segment counts."""
        shard = _make_test_shard(n_rows=3, segs_per_row=4)
        # row_seg_offsets is CSR-style: [0, 4, 8, 12] for 3 rows × 4 segs
        assert shard.row_seg_offsets.tolist() == [0, 4, 8, 12]
        # Each row has exactly 4 segments
        for r in range(3):
            n_seg_r = shard.row_seg_offsets[r + 1] - shard.row_seg_offsets[r]
            assert n_seg_r == 4

    def test_list_produces_heterogeneous_row_seg_counts(self):
        """segs_per_row as list[int] produces per-row variable segment counts."""
        shard = _make_test_shard(n_rows=3, segs_per_row=[2, 5, 3])
        # CSR offsets: [0, 2, 7, 10]
        assert shard.row_seg_offsets.tolist() == [0, 2, 7, 10]
        # Per-row segment counts match the input list
        seg_counts = [
            shard.row_seg_offsets[r + 1] - shard.row_seg_offsets[r]
            for r in range(3)
        ]
        assert seg_counts == [2, 5, 3]

    def test_list_tiling_respects_token_invariant(self):
        """Each row's segments tile [0, T) exactly: first starts at 0, last ends at T."""
        shard = _make_test_shard(n_rows=3, segs_per_row=[2, 5, 3], context_len=64)
        T = 64
        row_seg_offsets = shard.row_seg_offsets.tolist()
        for r in range(3):
            start_idx = row_seg_offsets[r]
            end_idx = row_seg_offsets[r + 1]
            # First segment of this row starts at 0
            assert shard.seg_token_start[start_idx].item() == 0
            # Last segment of this row ends at T
            assert shard.seg_token_end[end_idx - 1].item() == T
            # Segments are contiguous (no gaps)
            for s in range(start_idx, end_idx - 1):
                assert shard.seg_token_end[s].item() == shard.seg_token_start[s + 1].item()

    def test_list_length_mismatch_with_n_rows_raises(self):
        """segs_per_row list length must equal n_rows."""
        with pytest.raises(AssertionError, match="segs_per_row"):
            _make_test_shard(n_rows=3, segs_per_row=[2, 5])  # only 2 items
