"""Characterization tests for static padding parameters in packed_batch_io.

Verifies that ``pad_n_seg`` and ``pad_F`` produce deterministic, compile-friendly
shapes in both ``reconstruct_batch`` and ``reconstruct_batch_with_selection``.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.packed_batch_io import (
    SelectionShard,
    reconstruct_batch,
)
from tests.encdec_test_helpers import _make_identity_selection, _make_test_shard

# ---------------------------------------------------------------------------
# Local helpers for SelectionShard construction
# ---------------------------------------------------------------------------


def _make_selection_shard(
    total_segs: int,
    sel_per_seg: int = 3,
    neg_per_seg: int = 2,
    n_features: int = 200,
    seed: int = 0,
) -> SelectionShard:
    """Build a synthetic SelectionShard aligned with ``total_segs`` segments."""
    t.manual_seed(seed)
    total_sel = total_segs * sel_per_seg
    total_neg = total_segs * neg_per_seg

    sel_offsets = t.arange(0, (total_segs + 1) * sel_per_seg, sel_per_seg, dtype=t.int32)
    sel_fids = t.randint(0, n_features, (total_sel,), dtype=t.uint16)
    sel_buckets = t.randint(1, 8, (total_sel,), dtype=t.uint8)
    sel_pcts = t.randint(1, 255, (total_sel,), dtype=t.uint8)
    neg_offsets = t.arange(0, (total_segs + 1) * neg_per_seg, neg_per_seg, dtype=t.int32)
    neg_fids = t.randint(0, n_features, (total_neg,), dtype=t.uint16)

    return SelectionShard(
        sel_offsets=sel_offsets,
        sel_fids=sel_fids,
        sel_buckets=sel_buckets,
        sel_pcts=sel_pcts,
        neg_offsets=neg_offsets,
        neg_fids=neg_fids,
    )


# ---------------------------------------------------------------------------
# reconstruct_batch padding tests
# ---------------------------------------------------------------------------


class TestReconstructBatchStaticPadding:
    """pad_n_seg and pad_F produce correct, deterministic shapes."""

    def test_pad_n_seg_expands_segment_dim(self):
        """enc_fids.shape[1] == pad_n_seg when pad_n_seg is set."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        # natural batch_n_seg is 3; pad to 8
        rb = reconstruct_batch(shard, [0, 1, 2], _make_identity_selection(shard), 10, pad_n_seg=8)
        assert rb.enc_fids.shape[1] == 8
        assert rb.enc_mags.shape[1] == 8
        assert rb.enc_mask.shape[1] == 8
        assert rb.enc_pcts.shape[1] == 8
        assert rb.seg_doc_id.shape[1] == 8

    def test_pad_F_expands_feature_dim(self):
        """enc_fids.shape[2] == pad_F when pad_F is set."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        # natural batch_f_max is 20; pad to 64
        rb = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10, pad_F=64)
        assert rb.enc_fids.shape[2] == 64
        assert rb.enc_mags.shape[2] == 64
        assert rb.enc_mask.shape[2] == 64
        assert rb.enc_pcts.shape[2] == 64

    def test_pad_n_seg_and_pad_F_together(self):
        """Both pad_n_seg and pad_F set simultaneously produce correct shapes."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        rb = reconstruct_batch(shard, [0, 1, 2, 3], _make_identity_selection(shard), 10, pad_n_seg=10, pad_F=50)
        assert rb.enc_fids.shape == (4, 10, 50)

    def test_padded_positions_are_zero_and_unmasked(self):
        """Padding slots have enc_mask=False, enc_fids=0, enc_pcts=0."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        # pad segment dim well beyond actual
        pad_n_seg = 10
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10, pad_n_seg=pad_n_seg)
        actual_n_seg = 3  # segs_per_row from _make_test_shard default

        # Positions [actual_n_seg:pad_n_seg] in the segment dim are padding
        pad_mask = rb.enc_mask[0, actual_n_seg:]
        pad_fids = rb.enc_fids[0, actual_n_seg:]
        pad_pcts = rb.enc_pcts[0, actual_n_seg:]

        assert pad_mask.any().item() is False, "Padding segment positions must be unmasked"
        assert pad_fids.eq(0).all().item(), "Padding segment fids must be zero"
        assert pad_pcts.eq(0.0).all().item(), "Padding segment pcts must be zero"

    def test_feature_dim_padding_is_zero_and_unmasked(self):
        """Extra feature slots from pad_F are zeros and unmasked."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        pad_F = 50
        rb = reconstruct_batch(shard, [0], _make_identity_selection(shard), 10, pad_F=pad_F)
        actual_f = 20

        # Positions [actual_f:pad_F] are padding in the feature dim
        pad_mask = rb.enc_mask[0, :, actual_f:]
        pad_fids = rb.enc_fids[0, :, actual_f:]
        pad_pcts = rb.enc_pcts[0, :, actual_f:]

        assert pad_mask.any().item() is False
        assert pad_fids.eq(0).all().item()
        assert pad_pcts.eq(0.0).all().item()

    def test_pad_n_seg_too_small_raises(self):
        """AssertionError when pad_n_seg < actual batch_n_seg."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        with pytest.raises(AssertionError, match="pad_n_seg"):
            reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10, pad_n_seg=2)

    def test_pad_F_too_small_raises(self):
        """AssertionError when pad_F < actual batch_f_max."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        with pytest.raises(AssertionError, match="pad_F"):
            reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10, pad_F=5)

    def test_shapes_identical_across_different_row_subsets(self):
        """Same pad values produce same output shapes regardless of rows chosen."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        pad_n_seg, pad_F = 6, 40

        rb_a = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10, pad_n_seg=pad_n_seg, pad_F=pad_F)
        rb_b = reconstruct_batch(shard, [3, 5, 7], _make_identity_selection(shard), 10, pad_n_seg=pad_n_seg, pad_F=pad_F)

        # Batch size differs; remaining dims must match
        assert rb_a.enc_fids.shape[1:] == rb_b.enc_fids.shape[1:]
        assert rb_a.enc_fids.shape[1] == pad_n_seg
        assert rb_a.enc_fids.shape[2] == pad_F

    def test_pad_values_preserve_real_data(self):
        """Padded batch contains the same feature data as unpadded in valid positions."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        row_indices = [0, 1, 2]

        rb_natural = reconstruct_batch(shard, row_indices, _make_identity_selection(shard), 10)
        rb_padded = reconstruct_batch(shard, row_indices, _make_identity_selection(shard), 10, pad_n_seg=10, pad_F=50)

        natural_n_seg = rb_natural.enc_fids.shape[1]
        natural_f = rb_natural.enc_fids.shape[2]

        # The valid sub-block must be identical
        assert t.equal(
            rb_natural.enc_fids,
            rb_padded.enc_fids[:, :natural_n_seg, :natural_f],
        ), "enc_fids valid region must match unpadded"
        assert t.equal(
            rb_natural.enc_mask,
            rb_padded.enc_mask[:, :natural_n_seg, :natural_f],
        ), "enc_mask valid region must match unpadded"
        assert t.allclose(
            rb_natural.enc_pcts,
            rb_padded.enc_pcts[:, :natural_n_seg, :natural_f],
        ), "enc_pcts valid region must match unpadded"

    def test_pad_n_seg_exact_equal_is_valid(self):
        """pad_n_seg == actual batch_n_seg is valid (no error, no extra padding)."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        rb = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10, pad_n_seg=3)
        assert rb.enc_fids.shape[1] == 3

    def test_pad_F_exact_equal_is_valid(self):
        """pad_F == actual batch_f_max is valid (no error, no extra padding)."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        rb = reconstruct_batch(shard, [0, 1], _make_identity_selection(shard), 10, pad_F=20)
        assert rb.enc_fids.shape[2] == 20


# ---------------------------------------------------------------------------
# reconstruct_batch_with_selection padding tests
# ---------------------------------------------------------------------------


class TestReconstructBatchWithSelectionStaticPadding:
    """pad_n_seg and pad_F work correctly with SelectionShard."""

    # _make_test_shard: n_rows=8, segs_per_row=3, feats_per_seg=20
    # total_segs for a shard with n_rows=8, segs_per_row=3 is 24

    def test_pad_n_seg_with_selection(self):
        """Segment padding is correct with SelectionShard."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard(total_segs=24, sel_per_seg=3, neg_per_seg=2)

        rb = reconstruct_batch(shard, [0, 1], sel, 10, pad_n_seg=8)
        assert rb.enc_fids.shape[1] == 8
        assert rb.enc_mags.shape[1] == 8
        assert rb.enc_mask.shape[1] == 8

    def test_pad_F_with_selection(self):
        """Feature padding is correct with SelectionShard."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        # sel_per_seg + neg_per_seg = 3 + 2 = 5 combined features per seg
        sel = _make_selection_shard(total_segs=24, sel_per_seg=3, neg_per_seg=2)

        rb = reconstruct_batch(shard, [0, 1], sel, 10, pad_F=32)
        assert rb.enc_fids.shape[2] == 32

    def test_pad_both_with_selection(self):
        """Both pad_n_seg and pad_F work simultaneously with SelectionShard."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard(total_segs=24, sel_per_seg=3, neg_per_seg=2)

        rb = reconstruct_batch(
            shard, [0, 1, 2], sel, 10, pad_n_seg=10, pad_F=32,
        )
        assert rb.enc_fids.shape == (3, 10, 32)
        assert rb.enc_mags.shape == (3, 10, 32)
        assert rb.enc_mask.shape == (3, 10, 32)
        assert rb.enc_pcts.shape == (3, 10, 32)

    def test_negative_features_placed_correctly_with_padding(self):
        """Negative features appear after positives even with padding set."""
        n_rows = 8
        segs_per_row = 1
        sel_per_seg = 2
        neg_per_seg = 2

        shard = _make_test_shard(n_rows=n_rows, segs_per_row=segs_per_row, feats_per_seg=5)
        total_segs = n_rows * segs_per_row  # 8

        sel = _make_selection_shard(
            total_segs=total_segs, sel_per_seg=sel_per_seg, neg_per_seg=neg_per_seg, seed=7,
        )
        # Fix known values so we can assert exact placement
        sel.sel_buckets[:] = 3
        effective_buckets = 10

        rb = reconstruct_batch(
            shard, [0], sel, effective_buckets, pad_n_seg=4, pad_F=16,
        )

        # Row 0, segment 0: positives at slots 0..sel_per_seg-1, negatives at slots sel_per_seg..
        seg0 = rb.enc_mags[0, 0]  # [pad_F]
        # First sel_per_seg entries should be 3 (positive bucket)
        for i in range(sel_per_seg):
            assert seg0[i].item() == 3, f"slot {i} should be positive bucket 3"
        # Next neg_per_seg entries should be effective_buckets
        for i in range(neg_per_seg):
            assert seg0[sel_per_seg + i].item() == effective_buckets, (
                f"slot {sel_per_seg + i} should be negative bucket {effective_buckets}"
            )
        # Remaining padding slots should be 0
        combined = sel_per_seg + neg_per_seg
        remaining = rb.enc_mags[0, 0, combined:]
        assert remaining.eq(0).all().item(), "Padding feature slots must be zero"

    def test_padded_selection_preserves_data(self):
        """Valid positions in padded selection batch match unpadded batch."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard(total_segs=24, sel_per_seg=3, neg_per_seg=2)
        row_indices = [0, 1, 2]
        effective_buckets = 10

        rb_natural = reconstruct_batch(shard, row_indices, sel, effective_buckets)
        rb_padded = reconstruct_batch(
            shard, row_indices, sel, effective_buckets, pad_n_seg=10, pad_F=32,
        )

        natural_n_seg = rb_natural.enc_fids.shape[1]
        natural_f = rb_natural.enc_fids.shape[2]

        assert t.equal(
            rb_natural.enc_fids,
            rb_padded.enc_fids[:, :natural_n_seg, :natural_f],
        ), "enc_fids valid region must match unpadded"
        assert t.equal(
            rb_natural.enc_mags,
            rb_padded.enc_mags[:, :natural_n_seg, :natural_f],
        ), "enc_mags valid region must match unpadded"
        assert t.equal(
            rb_natural.enc_mask,
            rb_padded.enc_mask[:, :natural_n_seg, :natural_f],
        ), "enc_mask valid region must match unpadded"

    def test_pad_n_seg_too_small_raises_with_selection(self):
        """AssertionError when pad_n_seg < actual batch_n_seg (selection path)."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard(total_segs=24, sel_per_seg=3, neg_per_seg=2)
        with pytest.raises(AssertionError, match="pad_n_seg"):
            reconstruct_batch(shard, [0, 1], sel, 10, pad_n_seg=2)

    def test_pad_F_too_small_raises_with_selection(self):
        """AssertionError when pad_F < combined (sel + neg) features per seg."""
        shard = _make_test_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        # sel_per_seg=3, neg_per_seg=2 -> combined=5; pad_F=3 is too small
        sel = _make_selection_shard(total_segs=24, sel_per_seg=3, neg_per_seg=2)
        with pytest.raises(AssertionError, match="pad_F"):
            reconstruct_batch(shard, [0, 1], sel, 10, pad_F=3)
