"""Edge-case and assertion tests for encoder_decoder/dataloader.py.

Tests _pad_segments under boundary conditions not covered by the existing test
suites.
"""

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    _pad_segments,
    _stack_batch_parts,
)
from steering.packed_batch_io import ReconstructedBatch


def _make_part(n_seg: int, n_feat: int = 2, T: int = 8) -> ReconstructedBatch:
    """Build a minimal ReconstructedBatch with a given n_seg dim."""
    return ReconstructedBatch(
        inputs=t.zeros((1, T), dtype=t.int64),
        targets=t.zeros((1, T), dtype=t.int64),
        enc_fids=t.zeros((1, n_seg, n_feat), dtype=t.int64),
        enc_mags=t.zeros((1, n_seg, n_feat), dtype=t.int64),
        enc_mask=t.zeros((1, n_seg, n_feat), dtype=t.bool),
        enc_pcts=t.zeros((1, n_seg, n_feat), dtype=t.float32),
        seg_map=t.zeros((1, T), dtype=t.int64),
        seg_doc_id=t.full((1, n_seg), -1, dtype=t.int64),
    )


class TestPadSegmentsEdgeCases:
    """Edge cases for _pad_segments."""

    def test_mismatched_list_lengths_asserts(self):
        """Mismatched fids and buckets list lengths should assert."""
        fids = [t.tensor([1, 2], dtype=t.int64), t.tensor([3], dtype=t.int64)]
        buckets = [t.tensor([1, 1], dtype=t.int64)]  # one fewer
        pcts = [t.tensor([0.5, 0.5], dtype=t.float32), t.tensor([0.5], dtype=t.float32)]
        seg_map = t.zeros(10, dtype=t.int64)
        with pytest.raises(AssertionError, match="per_seg_buckets length"):
            _pad_segments(fids, buckets, pcts, seg_map)

    def test_mismatched_pcts_length_asserts(self):
        """Mismatched fids and pcts list lengths should also assert.

        Without this assert, the per-segment loop's zip() would silently
        truncate and produce corrupted padded output.
        """
        fids = [t.tensor([1, 2], dtype=t.int64), t.tensor([3], dtype=t.int64)]
        buckets = [t.tensor([1, 1], dtype=t.int64), t.tensor([1], dtype=t.int64)]
        pcts = [t.tensor([0.5, 0.5], dtype=t.float32)]  # one fewer
        seg_map = t.zeros(10, dtype=t.int64)
        with pytest.raises(AssertionError, match="per_seg_pcts length"):
            _pad_segments(fids, buckets, pcts, seg_map)

    def test_empty_lists_produce_minimal_output(self):
        """Empty feature lists produce tensors with shape (1, 1)."""
        fids: list[t.Tensor] = []
        buckets: list[t.Tensor] = []
        pcts: list[t.Tensor] = []
        seg_map = t.zeros(10, dtype=t.int64)
        feature_ids, magnitudes, percentiles, mask, out_map = _pad_segments(
            fids, buckets, pcts, seg_map
        )
        assert feature_ids.shape == (1, 1)  # n_seg_total=max(1,0)=1, f_max=max(1,0)=1
        assert not mask.any()

    def test_all_empty_segments(self):
        """All segments have zero features -> mask is all False."""
        fids = [t.tensor([], dtype=t.int64)] * 3
        buckets = [t.tensor([], dtype=t.int64)] * 3
        pcts = [t.tensor([], dtype=t.float32)] * 3
        seg_map = t.zeros(10, dtype=t.int64)
        feature_ids, magnitudes, percentiles, mask, _ = _pad_segments(
            fids, buckets, pcts, seg_map
        )
        assert feature_ids.shape[0] == 3
        assert not mask.any()

    def test_variable_length_segments_padded_correctly(self):
        """Segments with different feature counts are padded to F_max."""
        fids = [
            t.tensor([10, 20, 30], dtype=t.int64),
            t.tensor([40], dtype=t.int64),
            t.tensor([50, 60], dtype=t.int64),
        ]
        buckets = [
            t.tensor([1, 1, 1], dtype=t.int64),
            t.tensor([2], dtype=t.int64),
            t.tensor([3, 3], dtype=t.int64),
        ]
        pcts = [
            t.tensor([0.5, 0.5, 0.5], dtype=t.float32),
            t.tensor([0.3], dtype=t.float32),
            t.tensor([0.7, 0.7], dtype=t.float32),
        ]
        seg_map = t.zeros(10, dtype=t.int64)
        feature_ids, magnitudes, percentiles, mask, _ = _pad_segments(
            fids, buckets, pcts, seg_map
        )
        assert feature_ids.shape == (3, 3)  # F_max = 3
        assert mask[0].tolist() == [True, True, True]
        assert mask[1].tolist() == [True, False, False]
        assert mask[2].tolist() == [True, True, False]
        # Verify padding is zeros
        assert feature_ids[1, 1] == 0
        assert feature_ids[1, 2] == 0


class TestStackBatchPartsFpTargets:
    """Pin the n_seg-padding behavior of fp_targets stacking.

    _stack_batch_parts must pad each per-part fp_targets tensor to the batch-level
    max n_seg before concatenation. Without this, a precomputed packed-row
    dataloader crashes at epoch boundaries when consecutive rows happen to have
    different n_seg values (e.g., 33 vs 34), since torch.cat requires matching
    sizes in all but the cat dim.
    """

    def test_uniform_n_seg_unchanged(self):
        """When all fp_parts share the same n_seg, output is the simple cat
        and shape is coherent with enc_mask."""
        n_sae = 7
        fp_a = t.full((1, 4, n_sae), 1.0)
        fp_b = t.full((1, 4, n_sae), 2.0)
        parts = [_make_part(n_seg=4), _make_part(n_seg=4)]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        enc_mask = result[4]
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.shape == (2, 4, n_sae)
        assert fp_targets.shape[:2] == enc_mask.shape[:2], (
            f"fp_targets.shape[:2]={fp_targets.shape[:2]} != "
            f"enc_mask.shape[:2]={enc_mask.shape[:2]}"
        )
        assert t.equal(fp_targets[0], fp_a[0])
        assert t.equal(fp_targets[1], fp_b[0])

    def test_variable_n_seg_padded(self):
        """fp_parts with different n_seg are padded to the batch max, with
        shape coherent with enc_mask."""
        n_sae = 5
        fp_a = t.full((1, 3, n_sae), 7.0)
        fp_b = t.full((1, 5, n_sae), 9.0)
        parts = [_make_part(n_seg=3), _make_part(n_seg=5)]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        enc_mask = result[4]
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.shape == (2, 5, n_sae)
        assert fp_targets.shape[:2] == enc_mask.shape[:2], (
            f"fp_targets.shape[:2]={fp_targets.shape[:2]} != "
            f"enc_mask.shape[:2]={enc_mask.shape[:2]}"
        )
        # Slot 0: original 3 rows preserved, last 2 rows zero-padded.
        assert t.equal(fp_targets[0, :3], fp_a[0])
        assert (fp_targets[0, 3:] == 0).all()
        # Slot 1: full 5 rows preserved.
        assert t.equal(fp_targets[1], fp_b[0])

    def test_all_none_returns_none(self):
        """No fp_parts -> fp_targets stays None."""
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        result = _stack_batch_parts(
            parts, [None, None], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        assert result[8] is None

    def test_mixed_none_asserts(self):
        """Mixed None/Tensor fp_parts is rejected: silently dropping None entries
        would produce fp_targets.shape[0] < B and misalign downstream. The
        contract is all-or-nothing per call."""
        n_sae = 3
        fp_a = t.full((1, 4, n_sae), 1.0)
        parts = [_make_part(n_seg=4), _make_part(n_seg=2)]
        with pytest.raises(AssertionError, match="mixed None/Tensor"):
            _stack_batch_parts(
                parts, [fp_a, None], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_fast_path_shape_matches_enc(self):
        """Fast path (pad_n_seg + pad_F set): fp_targets shape matches enc_mask."""
        n_sae = 5
        pad_n_seg = 6
        pad_F = 3
        fp_a = t.full((1, pad_n_seg, n_sae), 1.0)
        fp_b = t.full((1, pad_n_seg, n_sae), 2.0)
        parts = [
            _make_part(n_seg=pad_n_seg, n_feat=pad_F),
            _make_part(n_seg=pad_n_seg, n_feat=pad_F),
        ]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu",
            pad_n_seg=pad_n_seg, pad_F=pad_F,
        )
        enc_mask = result[4]
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.shape == (2, pad_n_seg, n_sae)
        assert fp_targets.shape[:2] == enc_mask.shape[:2]

    def test_fast_path_n_seg_drift_asserts(self):
        """Fast path: parts[0].enc_fids.shape[1] must equal pad_n_seg.
        Catches a future regression where upstream reconstruction drifts."""
        pad_n_seg = 4
        pad_F = 3
        # Parts have n_seg=3, but caller claims pad_n_seg=4 — drift
        parts = [_make_part(n_seg=3, n_feat=pad_F), _make_part(n_seg=3, n_feat=pad_F)]
        fp_a = t.zeros((1, 3, 5))
        fp_b = t.zeros((1, 3, 5))
        with pytest.raises(AssertionError, match=r"enc_fids\.shape\[1\]=3 != pad_n_seg=4"):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu",
                pad_n_seg=pad_n_seg, pad_F=pad_F,
            )

    def test_fast_path_F_drift_asserts(self):
        """Fast path: parts[0].enc_fids.shape[2] must equal pad_F.
        Catches a future regression where upstream reconstruction drifts."""
        pad_n_seg = 3
        pad_F = 5
        # Parts have F=2, but caller claims pad_F=5 — drift
        parts = [_make_part(n_seg=pad_n_seg, n_feat=2), _make_part(n_seg=pad_n_seg, n_feat=2)]
        fp_a = t.zeros((1, pad_n_seg, 5))
        fp_b = t.zeros((1, pad_n_seg, 5))
        with pytest.raises(AssertionError, match=r"enc_fids\.shape\[2\]=2 != pad_F=5"):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu",
                pad_n_seg=pad_n_seg, pad_F=pad_F,
            )

    def test_fast_path_equivalent_to_old_alloc_scatter(self):
        """Fast-path t.cat result equals an explicit allocate+scatter pattern.

        Builds the expected output via the old slow-path pattern (zero buffer
        + per-row copy) and asserts the fast-path output matches. If the
        fast-path cat silently dropped a row, transposed dims, or had any
        other defect, this test fails.
        """
        n_sae = 4
        pad_n_seg = 3
        pad_F = 2
        fp_a = t.tensor([[[1.0, 2.0, 3.0, 4.0]] * pad_n_seg])  # (1, 3, 4)
        fp_b = t.tensor([[[5.0, 6.0, 7.0, 8.0]] * pad_n_seg])  # (1, 3, 4)
        parts = [
            _make_part(n_seg=pad_n_seg, n_feat=pad_F),
            _make_part(n_seg=pad_n_seg, n_feat=pad_F),
        ]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu",
            pad_n_seg=pad_n_seg, pad_F=pad_F,
        )
        fp_targets = result[8]
        # Build expected via the OLD allocate+scatter pattern.
        expected = t.zeros(2, pad_n_seg, n_sae)
        for i, fp in enumerate([fp_a, fp_b]):
            expected[i, : fp.shape[1]] = fp[0]
        assert fp_targets is not None
        assert t.equal(fp_targets, expected)

    def test_slow_path_shape_matches_enc(self):
        """Slow path: fp_targets and enc_mask share max_ns even when fp parts agree."""
        n_sae = 4
        fp_a = t.full((1, 3, n_sae), 1.0)
        fp_b = t.full((1, 3, n_sae), 2.0)
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        enc_mask = result[4]
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.shape[1] == enc_mask.shape[1] == 3

    def test_mismatched_n_sae_asserts(self):
        """Two parts with different n_sae fails the per-part assertion."""
        fp_a = t.zeros((1, 3, 5))
        fp_b = t.zeros((1, 3, 7))  # different n_sae
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        with pytest.raises(AssertionError, match=r"fp_parts\[1\]\.shape\[2\]=7 != n_sae=5"):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_mismatched_dtype_asserts(self):
        """Two parts with different dtype fails the per-part assertion."""
        fp_a = t.zeros((1, 3, 5), dtype=t.float32)
        fp_b = t.zeros((1, 3, 5), dtype=t.float64)
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        with pytest.raises(AssertionError, match=r"fp_parts\[1\]\.dtype="):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_mismatched_batch_dim_asserts(self):
        """fp_part with shape[0] != 1 is rejected (parts are per-row, B=1)."""
        fp_a = t.zeros((1, 3, 5))
        fp_b = t.zeros((2, 3, 5))  # shape[0] != 1
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        with pytest.raises(AssertionError, match=r"fp_parts\[1\]\.shape\[0\]=2 \(expected 1"):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_parts_device_mismatch_with_device_param_asserts(self):
        """When the `device` parameter is 'cuda' but parts are on CPU, the
        function rejects the call. Production callers always satisfy this
        invariant; the assert catches a future regression that decouples them.
        """
        # parts on cpu, but caller asks for cuda
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        with pytest.raises(AssertionError, match=r"parts\[0\]\.inputs\.device\.type"):
            _stack_batch_parts(
                parts, [None, None], B=2, device="cuda",
                pad_n_seg=None, pad_F=None,
            )

    def test_parts_device_mismatch_with_device_param_asserts_cpu_vs_meta(self):
        """CPU-runnable: parts on cpu but device='meta' is rejected at entry.

        Mirrors test_parts_device_mismatch_with_device_param_asserts but uses
        torch's meta device so the entry-level device contract is exercised
        on CPU-only CI as well as CUDA-equipped boxes.
        """
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]  # cpu by default
        with pytest.raises(AssertionError, match=r"parts\[0\]\.inputs\.device\.type"):
            _stack_batch_parts(
                parts, [None, None], B=2, device="meta",
                pad_n_seg=None, pad_F=None,
            )

    def test_mismatched_device_asserts_cpu_vs_meta(self):
        """CPU-runnable: fp_parts on cpu + meta are rejected.

        Uses torch's meta device (no storage but distinct device.type) so the
        device-mismatch contract can be exercised on CPU-only CI.
        """
        fp_a = t.zeros((1, 3, 5), device="cpu")
        fp_b = t.zeros((1, 3, 5), device="meta")
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        with pytest.raises(AssertionError, match=r"fp_parts\[1\]\.device="):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
    def test_mismatched_device_asserts(self):
        """fp_parts on different devices (cpu + cuda) are rejected."""
        fp_a = t.zeros((1, 3, 5), device="cpu")
        fp_b = t.zeros((1, 3, 5), device="cuda")
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        with pytest.raises(AssertionError, match=r"fp_parts\[1\]\.device="):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_fp_part_n_seg_mismatch_with_part_asserts(self):
        """fp.shape[1] must equal parts[i].enc_fids.shape[1]; mismatch raises."""
        fp_a = t.zeros((1, 4, 5))  # n_seg=4
        fp_b = t.zeros((1, 3, 5))  # n_seg=3 but part has n_seg=5
        parts = [_make_part(n_seg=4), _make_part(n_seg=5)]
        with pytest.raises(AssertionError, match="must match enc_fids n_seg"):
            _stack_batch_parts(
                parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_len_parts_neq_B_asserts(self):
        """B mismatch with len(parts) is rejected at the entry assertion."""
        parts = [_make_part(n_seg=3)]  # only 1 part
        with pytest.raises(AssertionError, match=r"len\(parts\)"):
            _stack_batch_parts(
                parts, [None], B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_len_fp_parts_neq_B_asserts(self):
        """B mismatch with len(fp_parts) is rejected at the entry assertion."""
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        fp_parts: list[t.Tensor | None] = [None]  # length 1, but B=2
        with pytest.raises(AssertionError, match=r"len\(fp_parts\)"):
            _stack_batch_parts(
                parts, fp_parts, B=2, device="cpu", pad_n_seg=None, pad_F=None
            )

    def test_dtypes_preserved_float32(self):
        """Input float32 yields output float32 (no silent cast)."""
        fp_a = t.zeros((1, 3, 5), dtype=t.float32)
        fp_b = t.zeros((1, 3, 5), dtype=t.float32)
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.dtype == t.float32

    def test_dtypes_preserved_float64(self):
        """Input float64 yields output float64 (no silent cast)."""
        fp_a = t.zeros((1, 3, 5), dtype=t.float64)
        fp_b = t.zeros((1, 3, 5), dtype=t.float64)
        parts = [_make_part(n_seg=3), _make_part(n_seg=3)]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.dtype == t.float64

    def test_n_seg_one_degenerate(self):
        """All parts have n_seg=1; output shape (B, 1, n_sae); no crash."""
        n_sae = 3
        fp_a = t.full((1, 1, n_sae), 1.0)
        fp_b = t.full((1, 1, n_sae), 2.0)
        parts = [_make_part(n_seg=1), _make_part(n_seg=1)]
        result = _stack_batch_parts(
            parts, [fp_a, fp_b], B=2, device="cpu", pad_n_seg=None, pad_F=None
        )
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.shape == (2, 1, n_sae)
        assert t.equal(fp_targets[0, 0], fp_a[0, 0])
        assert t.equal(fp_targets[1, 0], fp_b[0, 0])

    def test_B_one(self):
        """Single-part batch (B=1) works without degenerate-loop bugs."""
        n_sae = 4
        fp = t.full((1, 3, n_sae), 5.0)
        parts = [_make_part(n_seg=3)]
        result = _stack_batch_parts(
            parts, [fp], B=1, device="cpu", pad_n_seg=None, pad_F=None
        )
        enc_mask = result[4]
        fp_targets = result[8]
        assert fp_targets is not None
        assert fp_targets.shape == (1, 3, n_sae)
        assert fp_targets.shape[:2] == enc_mask.shape[:2]
        assert t.equal(fp_targets[0], fp[0])
