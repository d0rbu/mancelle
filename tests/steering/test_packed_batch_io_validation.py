"""Slice 05 audit — ``_build_batch_index`` tile invariants.

Pin the segment-tile invariants enforced by ``_build_batch_index``:
- first segment per row starts at token 0
- last segment per row ends at T
- consecutive same-row segments are contiguous (``tok_lo[i+1] == tok_hi[i]``)
- ``seg_doc_id`` <= 255 (uint8 max — already enforced by ``_row_to_csr``)
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    _build_batch_index,
    reconstruct_batch,
)


def _make_simple_shard(n_rows=2, T=8, n_seg_per_row=2) -> PackedRowShard:
    """Build a valid packed row shard with uniform tile structure for tests."""
    seg_size = T // n_seg_per_row
    total_segs = n_rows * n_seg_per_row
    tokens = t.zeros(n_rows, T + 1, dtype=t.uint16)
    row_seg_offsets = t.tensor(
        [i * n_seg_per_row for i in range(n_rows + 1)], dtype=t.int32,
    )
    seg_token_start = t.tensor(
        [i * seg_size for r in range(n_rows) for i in range(n_seg_per_row)],
        dtype=t.uint16,
    )
    seg_token_end = t.tensor(
        [(i + 1) * seg_size for r in range(n_rows) for i in range(n_seg_per_row)],
        dtype=t.uint16,
    )
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)
    seg_feat_offsets = t.tensor(
        list(range(total_segs + 1)), dtype=t.int32,
    )
    feature_ids = t.zeros(total_segs, dtype=t.uint16)
    feature_pcts = t.zeros(total_segs, dtype=t.uint8)
    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


class TestValidTiling:
    def test_simple_shard_builds(self):
        shard = _make_simple_shard()
        idx = _build_batch_index(shard, [0, 1])
        assert idx.B == 2
        assert idx.T == 8
        assert idx.batch_n_seg >= 2
        assert idx.seg_map.shape == (2, 8)
        # All seg_map values should be in [0, batch_n_seg)
        assert (idx.seg_map >= 0).all()
        assert (idx.seg_map < idx.batch_n_seg).all()


class TestInvalidTilingDetected:
    def test_first_segment_not_zero_raises(self):
        shard = _make_simple_shard()
        # Mutate: first segment of row 0 starts at 1, not 0.
        bad = replace(
            shard,
            seg_token_start=t.tensor([1, 4, 0, 4], dtype=t.uint16),
        )
        with pytest.raises(ValueError, match="first segment"):
            _build_batch_index(bad, [0, 1])

    def test_last_segment_not_T_raises(self):
        T = 8
        shard = _make_simple_shard(T=T)
        # Mutate: last segment of row 0 ends at T-1.
        bad = replace(
            shard,
            seg_token_end=t.tensor([4, T - 1, 4, T], dtype=t.uint16),
        )
        with pytest.raises(ValueError, match="last segment"):
            _build_batch_index(bad, [0, 1])

    def test_gap_between_consecutive_segments_raises(self):
        T = 8
        shard = _make_simple_shard(T=T)
        # Mutate: row 0 has seg ending at 3, next seg starting at 4 (gap of 1).
        bad = replace(
            shard,
            seg_token_start=t.tensor([0, 4, 0, 4], dtype=t.uint16),
            seg_token_end=t.tensor([3, T, 4, T], dtype=t.uint16),  # gap from 3→4
        )
        with pytest.raises(ValueError, match="seg .+ ends at"):
            _build_batch_index(bad, [0, 1])


class TestSegMapShape:
    @settings(
        derandomize=True, max_examples=200,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        n_rows=st.integers(min_value=1, max_value=4),
        T=st.integers(min_value=4, max_value=16),
        n_seg_per_row=st.integers(min_value=1, max_value=4),
    )
    def test_seg_map_shape_matches_BT(self, n_rows, T, n_seg_per_row):
        # Skip when T is not divisible by n_seg_per_row (invalid layout).
        if T % n_seg_per_row != 0:
            return
        shard = _make_simple_shard(n_rows=n_rows, T=T, n_seg_per_row=n_seg_per_row)
        row_indices = list(range(n_rows))
        idx = _build_batch_index(shard, row_indices)
        assert idx.seg_map.shape == (n_rows, T)


class TestSegDocIdRange:
    """``_row_to_csr`` asserts ``max_doc_id <= 255``; ``_build_batch_index``
    cast ``seg_doc_id`` from uint8 to int64 — values must round-trip without
    overflow.
    """

    def test_seg_doc_id_round_trip(self):
        shard = _make_simple_shard()
        # Patch seg_doc_id to a known value pattern.
        custom = t.tensor([0, 1, 2, 3], dtype=t.uint8)
        shard2 = replace(shard, seg_doc_id=custom)
        idx = _build_batch_index(shard2, [0, 1])
        # idx.seg_doc_id is (B, batch_n_seg) int64 with -1 padding.
        # row 0: [0, 1, ...]; row 1: [2, 3, ...]
        assert idx.seg_doc_id[0, 0].item() == 0
        assert idx.seg_doc_id[0, 1].item() == 1
        assert idx.seg_doc_id[1, 0].item() == 2
        assert idx.seg_doc_id[1, 1].item() == 3


class TestReconstructBatchEnd2End:
    """Sanity that the full path runs against the simple shard fixture.
    Detailed semantics live in ``tests/steering/test_packed_batch_io.py``.
    """

    def test_smoke(self):
        shard = _make_simple_shard()
        # Empty selection shard (no positives, no negatives)
        sel = SelectionShard(
            sel_offsets=t.zeros(shard.total_segments + 1, dtype=t.int32),
            sel_fids=t.tensor([], dtype=t.uint16),
            sel_buckets=t.tensor([], dtype=t.uint8),
            sel_pcts=t.tensor([], dtype=t.uint8),
            neg_offsets=t.zeros(shard.total_segments + 1, dtype=t.int32),
            neg_fids=t.tensor([], dtype=t.uint16),
        )
        batch = reconstruct_batch(shard, [0, 1], sel, effective_buckets=2)
        assert batch.inputs.shape == (2, 8)
        assert batch.targets.shape == (2, 8)
        assert batch.seg_map.shape == (2, 8)
