"""Extreme-scrutiny tests for MAX_POOL training loss + inference pooling.

MAX_POOL is the new default for FeaturePredictionConfig.loss_mode and the
recommended pool_mode for self-conditioned generation. This file exists
to surface any subtle bugs before training real models with it.

Coverage targets:
1. Property-level invariants (permutation invariance, single-position
   equivalence, dtype/shape preservation).
2. Cross-impl equivalence with a Python reference at MANY random configs.
3. Gradient flow correctness (through max element only).
4. Edge cases (empty segments, all-padding, single source, NaN guard).
5. Numerics under bf16 / fp32 paths.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    _max_pool_loss,
    compute_feature_pred_loss,
    pool_fp_logits,
)

# ---------------------------------------------------------------------------
# Reference: pure Python max-pool loss (mirrors test_feature_prediction.py)
# ---------------------------------------------------------------------------


def _reference_max_pool_loss(logits, targets, seg_map, seg_doc_id, loss_fn, n_sae):
    B, T, _ = logits.shape
    n_seg = targets.shape[1]
    total_loss = 0.0
    total_target_tokens = 0
    for b in range(B):
        for s in range(n_seg - 1):
            d_curr = seg_doc_id[b, s].item()
            d_next = seg_doc_id[b, s + 1].item()
            if d_curr != d_next or d_curr < 0:
                continue
            src = (seg_map[b] == s).nonzero(as_tuple=True)[0]
            tgt_count = (seg_map[b] == s + 1).sum().item()
            if len(src) == 0 or tgt_count == 0:
                continue
            total_target_tokens += tgt_count
    if total_target_tokens == 0:
        return 0.0
    for b in range(B):
        for s in range(n_seg - 1):
            d_curr = seg_doc_id[b, s].item()
            d_next = seg_doc_id[b, s + 1].item()
            if d_curr != d_next or d_curr < 0:
                continue
            src = (seg_map[b] == s).nonzero(as_tuple=True)[0]
            tgt_count = (seg_map[b] == s + 1).sum().item()
            if len(src) == 0 or tgt_count == 0:
                continue
            target = targets[b, s + 1]
            seg_logits = logits[b, src]
            max_logits = seg_logits.amax(dim=0)
            if loss_fn == "bce":
                seg_loss = F.binary_cross_entropy_with_logits(
                    max_logits, target, reduction="mean"
                ).item()
            else:
                seg_loss = F.mse_loss(
                    torch.sigmoid(max_logits), target, reduction="mean"
                ).item()
            total_loss += seg_loss * (tgt_count / total_target_tokens)
    return total_loss


# ---------------------------------------------------------------------------
# pool_fp_logits MAX_POOL — invariants
# ---------------------------------------------------------------------------


class TestPoolFpLogitsMaxPoolInvariants:
    def test_permutation_invariance_over_time(self):
        torch.manual_seed(0)
        x = torch.randn(2, 7, 5)
        a = pool_fp_logits(x, FeaturePredLossMode.MAX_POOL)
        # Random permutation of the time dim
        perm = torch.randperm(7)
        b = pool_fp_logits(x[:, perm, :], FeaturePredLossMode.MAX_POOL)
        assert torch.equal(a, b)

    def test_single_position_equals_value(self):
        x = torch.randn(3, 1, 8)
        out = pool_fp_logits(x, FeaturePredLossMode.MAX_POOL)
        assert torch.equal(out, x[:, 0, :])

    def test_idempotent_for_repeated_position(self):
        x = torch.randn(1, 1, 4)
        x_rep = x.repeat(1, 5, 1)
        out = pool_fp_logits(x_rep, FeaturePredLossMode.MAX_POOL)
        assert torch.allclose(out, x[:, 0, :])

    def test_dtype_preserved(self):
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            x = torch.randn(1, 4, 3).to(dtype)
            out = pool_fp_logits(x, FeaturePredLossMode.MAX_POOL)
            assert out.dtype == dtype

    def test_max_pool_dominates_mean_per_feature(self):
        torch.manual_seed(7)
        x = torch.randn(2, 5, 6)
        a = pool_fp_logits(x, FeaturePredLossMode.MAX_POOL)
        b = pool_fp_logits(x, FeaturePredLossMode.FULL_SEGMENT)
        assert (a >= b - 1e-6).all()

    def test_max_pool_dominates_last_token_per_feature(self):
        torch.manual_seed(7)
        x = torch.randn(2, 5, 6)
        a = pool_fp_logits(x, FeaturePredLossMode.MAX_POOL)
        b = pool_fp_logits(x, FeaturePredLossMode.LAST_TOKEN)
        assert (a >= b - 1e-6).all()


# ---------------------------------------------------------------------------
# _max_pool_loss — equivalence with reference at random configs
# ---------------------------------------------------------------------------


class TestMaxPoolLossReferenceEquivalence:
    @pytest.mark.parametrize("seed", list(range(8)))
    def test_random_single_doc_matches_reference(self, seed):
        torch.manual_seed(seed)
        B, T, n_sae, n_seg = 2, 12, 5, 4
        logits = torch.randn(B, T, n_sae)
        targets = torch.rand(B, n_seg, n_sae)
        # Random monotonic seg_map per batch row
        seg_map = torch.zeros(B, T, dtype=torch.long)
        for b in range(B):
            cuts = sorted(
                torch.randperm(T - 1)[: n_seg - 1].tolist()
            )
            cur = 0
            for s, cut in enumerate(cuts):
                seg_map[b, cur:cut + 1] = s
                cur = cut + 1
            seg_map[b, cur:] = n_seg - 1
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        a = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        b = _reference_max_pool_loss(
            logits, targets, seg_map, seg_doc_id, "bce", n_sae,
        )
        assert abs(a.item() - b) < 1e-4

    @pytest.mark.parametrize("seed", list(range(4)))
    def test_random_multi_doc_matches_reference(self, seed):
        torch.manual_seed(seed * 17)
        B, T, n_sae, n_seg = 2, 16, 4, 4
        logits = torch.randn(B, T, n_sae)
        targets = torch.rand(B, n_seg, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        for b in range(B):
            seg_map[b, :4] = 0
            seg_map[b, 4:8] = 1
            seg_map[b, 8:12] = 2
            seg_map[b, 12:] = 3
        # Two docs per row: (0, 1) in doc 0, (2, 3) in doc 1
        seg_doc_id = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
        for fn, fn_name in (
            (FeaturePredLossFn.BCE, "bce"),
            (FeaturePredLossFn.MSE, "mse"),
        ):
            a = _max_pool_loss(logits, seg_map, targets, seg_doc_id, fn)
            ref = _reference_max_pool_loss(
                logits, targets, seg_map, seg_doc_id, fn_name, n_sae,
            )
            assert abs(a.item() - ref) < 1e-4


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------


class TestMaxPoolGradient:
    def test_gradient_flows_only_through_max(self):
        """If two positions clearly differ, only the max one gets gradient."""
        torch.manual_seed(0)
        B, n_seg = 1, 2
        # Position 1 is the clear max for both features
        logits = torch.tensor(
            [[[1.0, -1.0],
              [3.0, 2.0],   # max
              [0.0, 0.0],
              [0.0, 0.0]]],
            requires_grad=True,
        )
        seg_map = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        loss.backward()
        # Position 1 (the max) gets gradient
        assert logits.grad[0, 1].abs().sum() > 0  # ty: ignore[not-subscriptable]
        # Position 0 (lost the max for both features) gets zero gradient
        assert logits.grad[0, 0].abs().sum() == 0  # ty: ignore[not-subscriptable]

    def test_no_gradient_for_non_source_positions(self):
        """Tokens in segment n_seg - 1 (the last segment) cannot predict
        a next segment, so they should receive zero gradient."""
        torch.manual_seed(0)
        B, T, n_sae, n_seg = 1, 4, 2, 2
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        loss.backward()
        # Last-segment tokens (positions 2, 3) never source any prediction
        assert torch.allclose(
            logits.grad[0, 2:], torch.zeros_like(logits.grad[0, 2:]),  # ty: ignore[not-subscriptable]
        )

    def test_gradient_split_when_different_features_have_different_argmax(self):
        """Per-feature max can land on different time positions. Each
        winning position receives gradient ONLY for the feature it won."""
        torch.manual_seed(0)
        B, n_seg = 1, 2
        logits = torch.tensor(
            [[[10.0, 0.0],   # pos 0 wins feature 0
              [0.0, 10.0],   # pos 1 wins feature 1
              [0.0, 0.0]]],   # not source
            requires_grad=True,
        )
        seg_map = torch.tensor([[0, 0, 1]], dtype=torch.long)
        targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        loss.backward()
        # pos 0 gets gradient on feature 0 only
        assert logits.grad[0, 0, 0].abs() > 0  # ty: ignore[not-subscriptable]
        assert logits.grad[0, 0, 1].abs() < 1e-7  # ty: ignore[not-subscriptable]
        # pos 1 gets gradient on feature 1 only
        assert logits.grad[0, 1, 0].abs() < 1e-7  # ty: ignore[not-subscriptable]
        assert logits.grad[0, 1, 1].abs() > 0  # ty: ignore[not-subscriptable]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMaxPoolEdgeCases:
    def test_single_segment_no_loss(self):
        """With only 1 segment there's no s+1 to predict; loss should be 0."""
        B, T, n_sae, n_seg = 1, 4, 3, 1
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        assert loss.item() == 0.0

    def test_padded_doc_id_excluded(self):
        """seg_doc_id == -1 marks padding; padded segments must not
        contribute to loss."""
        B, T, n_sae, n_seg = 1, 6, 2, 3
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long)
        targets = torch.rand(B, n_seg, n_sae)
        # seg 0 → seg 1 valid (doc 0); seg 2 is padding
        seg_doc_id = torch.tensor([[0, 0, -1]])
        loss = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        # Only seg 0 contributes; weight = next_seg_len / total = 2 / 2 = 1
        # So loss equals BCE(max_logits_seg0, targets[0, 1]).mean()
        max0 = logits[0, :2].amax(dim=0)
        ref = F.binary_cross_entropy_with_logits(
            max0, targets[0, 1], reduction="mean",
        ).item()
        assert abs(loss.item() - ref) < 1e-5

    def test_all_negative_logits(self):
        """All-negative logits should still produce a finite loss."""
        torch.manual_seed(0)
        B, T, n_sae, n_seg = 1, 4, 3, 2
        logits = -torch.rand(B, T, n_sae) * 50  # large negative
        seg_map = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        targets = torch.full((B, n_seg, n_sae), 0.5)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_ties_broken_deterministically(self):
        """Exact ties in logits should not introduce nondeterminism in the
        loss (the value is well-defined; only the gradient distribution
        across tied elements may differ)."""
        torch.manual_seed(0)
        B, T, n_sae, n_seg = 1, 3, 2, 2
        # All positions have IDENTICAL logits → max is well-defined
        logits = torch.full((B, T, n_sae), 0.7, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 1]], dtype=torch.long)
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        # Run twice; values must match
        loss_a = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        loss_b = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        assert torch.equal(loss_a, loss_b)


# ---------------------------------------------------------------------------
# Numerics: bf16 vs fp32
# ---------------------------------------------------------------------------


class TestMaxPoolNumerics:
    def test_bf16_matches_fp32_within_tolerance(self):
        torch.manual_seed(123)
        B, T, n_sae, n_seg = 1, 8, 4, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2 :] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss_fp32 = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE,
        )
        loss_bf16 = _max_pool_loss(
            logits.to(torch.bfloat16),
            seg_map, targets.to(torch.bfloat16), seg_doc_id,
            FeaturePredLossFn.BCE,
        )
        # bf16 has ~7-bit mantissa; tolerate a couple percent.
        assert abs(loss_fp32.item() - float(loss_bf16)) < 0.05

    def test_dispatch_through_compute_feature_pred_loss(self):
        torch.manual_seed(0)
        B, T, n_sae, n_seg = 1, 6, 3, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 3:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        a = _max_pool_loss(
            logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.MSE,
        )
        b = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.MAX_POOL, FeaturePredLossFn.MSE,
        )
        torch.testing.assert_close(a, b)
