"""Unit tests for steering/conditional/encoder_decoder/feature_prediction.py.

Tests the core loss functions in isolation with synthetic inputs:
- _compute_per_token_loss: per-token BCE/MSE
- compute_feature_pred_loss: all three loss modes (full_segment, last_token, best_token)
- Cross-document prediction prevention via seg_doc_id
- Gradient flow through each loss mode
- Edge cases: empty segments, single segment, all-padding
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
    _best_token_loss,
    _compute_per_token_loss,
    _full_segment_loss,
    _last_token_loss,
    _max_pool_loss,
    compute_feature_pred_loss,
    resolve_layer_index,
)

# ---------------------------------------------------------------------------
# resolve_layer_index
# ---------------------------------------------------------------------------


class TestResolveLayerIndex:
    """Resolve Python-style (negative) layer indices to absolute indices."""

    def test_positive_index(self):
        assert resolve_layer_index(0, 10) == 0
        assert resolve_layer_index(5, 10) == 5

    def test_negative_index(self):
        assert resolve_layer_index(-1, 10) == 9
        assert resolve_layer_index(-3, 10) == 7

    def test_out_of_range_raises(self):
        with pytest.raises(AssertionError):
            resolve_layer_index(10, 10)
        with pytest.raises(AssertionError):
            resolve_layer_index(-11, 10)

    def test_single_layer(self):
        """n_layer=1: only index 0 or -1 is valid."""
        assert resolve_layer_index(0, 1) == 0
        assert resolve_layer_index(-1, 1) == 0


# ---------------------------------------------------------------------------
# _compute_per_token_loss
# ---------------------------------------------------------------------------


class TestComputePerTokenLoss:
    """Per-token feature prediction loss (mean over feature dim)."""

    def test_bce_known_value(self):
        """BCE with logits=0.0 and targets=0.5 gives -0.5*log(0.5) - 0.5*log(0.5) = ln(2)."""
        logits = torch.zeros(1, 4)  # sigmoid(0) = 0.5
        targets = torch.full((1, 4), 0.5)
        loss = _compute_per_token_loss(logits, targets, FeaturePredLossFn.BCE)
        expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(
            dim=-1
        )
        torch.testing.assert_close(loss, expected)

    def test_mse_known_value(self):
        """MSE: sigmoid(0)=0.5, target=1.0 → loss = (0.5)^2 = 0.25 per feature."""
        logits = torch.zeros(1, 4)
        targets = torch.ones(1, 4)
        loss = _compute_per_token_loss(logits, targets, FeaturePredLossFn.MSE)
        # sigmoid(0)=0.5, mse = (0.5 - 1.0)^2 = 0.25
        torch.testing.assert_close(loss, torch.tensor([0.25]))

    def test_output_shape(self):
        """Output should be (N,) — one scalar per token."""
        N, n_sae = 5, 32
        logits = torch.randn(N, n_sae)
        targets = torch.rand(N, n_sae)
        loss = _compute_per_token_loss(logits, targets, FeaturePredLossFn.BCE)
        assert loss.shape == (N,)

    def test_perfect_prediction_bce(self):
        """Perfect prediction (logits large positive, targets=1) should give near-zero loss."""
        logits = torch.full((1, 10), 10.0)
        targets = torch.ones(1, 10)
        loss = _compute_per_token_loss(logits, targets, FeaturePredLossFn.BCE)
        assert loss.item() < 1e-3

    def test_perfect_prediction_mse(self):
        """Sigmoid(large positive) ≈ 1.0, target=1.0 → MSE ≈ 0."""
        logits = torch.full((1, 10), 10.0)
        targets = torch.ones(1, 10)
        loss = _compute_per_token_loss(logits, targets, FeaturePredLossFn.MSE)
        assert loss.item() < 1e-3


# ---------------------------------------------------------------------------
# compute_feature_pred_loss — full_segment mode
# ---------------------------------------------------------------------------


class TestFullSegmentLoss:
    """_full_segment_loss: mean over all source tokens globally."""

    def _make_simple_inputs(self, B=1, T=8, n_seg=2, n_sae=4):
        """Two equal segments, same document, known targets."""
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2 :] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        return logits, seg_map, targets, seg_doc_id

    def test_returns_scalar(self):
        """Loss should be a scalar tensor."""
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.dim() == 0

    def test_loss_is_finite(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert torch.isfinite(loss)

    def test_only_source_segments_contribute(self):
        """Last segment (seg n_seg-1) cannot predict a next segment, so it contributes nothing.

        With 2 segments: only segment 0 tokens are source tokens (predicting segment 1).
        """
        B, T, n_sae = 1, 8, 4
        n_seg = 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)

        # Manually compute: only seg 0 tokens (positions 0-3) contribute
        per_token = _compute_per_token_loss(
            logits.reshape(-1, n_sae),
            targets[:, 1:2].expand(B, T, n_sae).reshape(-1, n_sae),
            FeaturePredLossFn.BCE,
        ).reshape(B, T)
        expected = per_token[0, :4].mean()
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        """Gradients propagate back through full_segment loss to logits."""
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        loss.backward()
        assert logits.grad is not None
        # Source tokens (segment 0, positions 0-3) should have non-zero grad
        assert logits.grad[0, :4].abs().sum() > 0
        # Non-source tokens (segment 1, positions 4-7) should have zero grad
        torch.testing.assert_close(
            logits.grad[0, 4:],
            torch.zeros_like(logits.grad[0, 4:]),
        )

    def test_cross_document_masking(self):
        """Segments from different documents should not predict each other."""
        B, T, n_sae, n_seg = 1, 8, 4, 2
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        # Different documents for each segment → no valid source
        seg_doc_id = torch.tensor([[0, 1]])

        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        # With no valid source segments, loss should be 0
        assert loss.item() == 0.0

    def test_single_segment_gives_zero_loss(self):
        """Single segment has no next segment to predict → loss is 0."""
        B, T, n_sae = 1, 4, 4
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        targets = torch.rand(B, 1, n_sae)
        seg_doc_id = torch.zeros(B, 1, dtype=torch.long)

        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# compute_feature_pred_loss — last_token mode
# ---------------------------------------------------------------------------


class TestLastTokenLoss:
    """_last_token_loss: only last token per source segment, weighted by next seg length."""

    def _make_simple_inputs(self, B=1, T=8, n_seg=2, n_sae=4):
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2 :] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        return logits, seg_map, targets, seg_doc_id

    def test_returns_scalar(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.dim() == 0

    def test_loss_is_finite(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert torch.isfinite(loss)

    def test_only_last_token_has_grad(self):
        """Only the last token of segment 0 (position 3) should receive gradient."""
        B, T, n_sae = 1, 8, 4
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.zeros(B, 2, dtype=torch.long)

        loss = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        loss.backward()
        grad = logits.grad
        assert grad is not None

        # Position 3 is last token of segment 0 (source)
        assert grad[0, 3].abs().sum() > 0
        # Positions 0-2 are not last tokens → zero grad
        torch.testing.assert_close(
            grad[0, :3],
            torch.zeros_like(grad[0, :3]),
        )
        # Positions 4-7 are in segment 1 (non-source) → zero grad
        torch.testing.assert_close(
            grad[0, 4:],
            torch.zeros_like(grad[0, 4:]),
        )

    def test_cross_document_masking(self):
        """Different documents → no valid predictions → zero loss."""
        B, T, n_sae = 1, 8, 4
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.tensor([[0, 1]])

        loss = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.item() == 0.0

    def test_weighted_by_next_segment_length(self):
        """Loss is weighted by next segment's token count / total target tokens."""
        B, T, n_sae = 1, 12, 4
        n_seg = 3
        # Unequal segments: 4, 6, 2 tokens
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2]])
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        # Manually set logits so we know exact per-token losses
        logits = torch.zeros(B, T, n_sae)  # sigmoid(0)=0.5

        loss = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_single_segment_gives_zero_loss(self):
        """Single segment has no next segment to predict."""
        B, T, n_sae = 1, 4, 4
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        targets = torch.rand(B, 1, n_sae)
        seg_doc_id = torch.zeros(B, 1, dtype=torch.long)

        loss = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# compute_feature_pred_loss — best_token mode
# ---------------------------------------------------------------------------


class TestBestTokenLoss:
    """_best_token_loss: min-loss token per source segment, weighted by next seg length."""

    def _make_simple_inputs(self, B=1, T=8, n_seg=2, n_sae=4):
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2 :] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        return logits, seg_map, targets, seg_doc_id

    def test_returns_scalar(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.dim() == 0

    def test_loss_is_finite(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert torch.isfinite(loss)

    def test_best_token_leq_full_segment(self):
        """Best-token loss <= full-segment loss (min is always <= mean)."""
        torch.manual_seed(42)
        B, T, n_sae, n_seg = 1, 8, 4, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        loss_best = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        loss_full = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss_best.item() <= loss_full.item() + 1e-6

    def test_gradient_flows_to_min_token_only(self):
        """Gradient should flow primarily through the minimum-loss token."""
        torch.manual_seed(42)
        B, T, n_sae = 1, 8, 4
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.zeros(B, 2, dtype=torch.long)

        loss = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        loss.backward()
        grad = logits.grad
        assert grad is not None

        # Non-source positions (segment 1) have zero gradient
        torch.testing.assert_close(
            grad[0, 4:],
            torch.zeros_like(grad[0, 4:]),
        )
        # At least one source position has non-zero gradient
        source_grads = grad[0, :4]
        assert source_grads.abs().sum() > 0

        # Exactly one token in the source segment should have non-zero grad
        # (the one with minimum loss)
        per_token_grad_norm = source_grads.abs().sum(dim=-1)
        n_nonzero = (per_token_grad_norm > 1e-8).sum().item()
        assert n_nonzero == 1, f"Expected exactly 1 token with grad, got {n_nonzero}"

    def test_cross_document_masking(self):
        """Different documents → zero loss."""
        B, T, n_sae = 1, 8, 4
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.tensor([[0, 1]])

        loss = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.item() == 0.0

    def test_single_token_segment_equals_last_token(self):
        """With 1-token source segments, best_token = last_token = that one token."""
        B, T, n_sae = 1, 4, 4
        seg_map = torch.tensor([[0, 1, 2, 3]])
        targets = torch.rand(B, 4, n_sae)
        seg_doc_id = torch.zeros(B, 4, dtype=torch.long)

        # Use same logits for fair comparison
        logits = torch.randn(B, T, n_sae)
        loss_best = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        loss_last = _last_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        torch.testing.assert_close(loss_best, loss_last, atol=1e-5, rtol=1e-5)

    def test_single_segment_gives_zero_loss(self):
        B, T, n_sae = 1, 4, 4
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        targets = torch.rand(B, 1, n_sae)
        seg_doc_id = torch.zeros(B, 1, dtype=torch.long)

        loss = _best_token_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# compute_feature_pred_loss — dispatcher
# ---------------------------------------------------------------------------


class TestComputeFeaturePredLoss:
    """Top-level dispatch function routes to correct loss mode."""

    @pytest.mark.parametrize("loss_mode", list(FeaturePredLossMode))
    @pytest.mark.parametrize("loss_fn", list(FeaturePredLossFn))
    def test_all_mode_fn_combinations_produce_finite_loss(self, loss_mode, loss_fn):
        """Every (mode, fn) combination produces a finite scalar."""
        B, T, n_sae, n_seg = 2, 8, 4, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        loss = compute_feature_pred_loss(logits, seg_map, targets, seg_doc_id, loss_mode, loss_fn)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("loss_mode", list(FeaturePredLossMode))
    def test_gradient_flows_for_all_modes(self, loss_mode):
        """Gradient reaches logits for all loss modes."""
        B, T, n_sae, n_seg = 1, 8, 4, 2
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id, loss_mode, FeaturePredLossFn.BCE
        )
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


class TestTargetOnlySegments:
    """SFT prediction-target-only segments have targets but no decoder tokens."""

    def _make_target_only_inputs(
        self,
        *,
        same_doc: bool = True,
        nonzero_target: bool = True,
    ):
        B, T, n_seg, n_sae = 1, 4, 3, 5
        logits = torch.zeros(B, T, n_sae, requires_grad=True)
        seg_map = torch.ones(B, T, dtype=torch.long)
        targets = torch.zeros(B, n_seg, n_sae)
        if nonzero_target:
            targets[0, 2, 0] = 1.0
        seg_doc_id = torch.tensor([[-1, 0, 0 if same_doc else 1]])
        return logits, seg_map, targets, seg_doc_id

    @pytest.mark.parametrize("loss_mode", list(FeaturePredLossMode))
    def test_real_no_token_target_segment_contributes_loss(self, loss_mode):
        """A same-doc target-only segment must not be dropped as empty."""
        logits, seg_map, targets, seg_doc_id = self._make_target_only_inputs()

        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id, loss_mode, FeaturePredLossFn.BCE
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    @pytest.mark.parametrize("loss_mode", list(FeaturePredLossMode))
    def test_cross_doc_target_only_segment_is_still_masked(self, loss_mode):
        """No-token target support must not reopen cross-document leakage."""
        logits, seg_map, targets, seg_doc_id = self._make_target_only_inputs(same_doc=False)

        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id, loss_mode, FeaturePredLossFn.BCE
        )
        assert loss.item() == 0.0

        loss.backward()
        assert logits.grad is not None
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))

    @pytest.mark.parametrize("loss_mode", list(FeaturePredLossMode))
    def test_zero_feature_target_only_segment_trains_absence(self, loss_mode):
        """An explicit same-doc target-only segment can supervise all-zero features."""
        logits, seg_map, targets, seg_doc_id = self._make_target_only_inputs(nonzero_target=False)

        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id, loss_mode, FeaturePredLossFn.BCE
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

        loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Multi-segment and batched scenarios
# ---------------------------------------------------------------------------


class TestMultiSegmentScenarios:
    """Tests with 3+ segments and batched inputs."""

    def test_three_segments_same_doc(self):
        """Three segments from same doc: seg 0 → seg 1, seg 1 → seg 2."""
        B, T, n_sae, n_seg = 1, 12, 4, 3
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        for mode in FeaturePredLossMode:
            loss = compute_feature_pred_loss(
                logits, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            assert torch.isfinite(loss)
            assert loss.item() > 0

    def test_three_segments_doc_boundary_in_middle(self):
        """Segments [doc0, doc0, doc1]: seg 0→1 valid, seg 1→2 cross-doc (invalid)."""
        B, T, n_sae, n_seg = 1, 12, 4, 3
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.tensor([[0, 0, 1]])  # doc boundary between seg 1 and 2

        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        loss.backward()
        grad = logits.grad
        assert grad is not None

        # Only segment 0 is a valid source (predicting seg 1, same doc)
        assert grad[0, :4].abs().sum() > 0
        # Segment 1 is not a valid source (seg 2 is different doc)
        torch.testing.assert_close(
            grad[0, 4:8],
            torch.zeros_like(grad[0, 4:8]),
        )
        # Segment 2 is last segment, never a source
        torch.testing.assert_close(
            grad[0, 8:],
            torch.zeros_like(grad[0, 8:]),
        )

    def test_batched_independent(self):
        """Different batch elements have independent losses."""
        B, T, n_sae, n_seg = 2, 8, 4, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        # Batch 0: same doc → non-zero loss
        # Batch 1: different doc → zero loss contribution from those segments
        seg_doc_id = torch.tensor([[0, 0], [0, 1]])

        loss = _full_segment_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        assert torch.isfinite(loss)
        assert loss.item() > 0  # batch 0 contributes

    def test_unequal_segment_sizes(self):
        """Variable segment sizes within a batch element."""
        B, T, n_sae, n_seg = 1, 10, 4, 3
        # Segments: [3, 5, 2] tokens
        seg_map = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 2, 2]])
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        for mode in FeaturePredLossMode:
            logits = torch.randn(B, T, n_sae, requires_grad=True)
            loss = compute_feature_pred_loss(
                logits, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            assert torch.isfinite(loss)
            loss.backward()
            grad = logits.grad
            assert grad is not None
            # Last segment (seg 2) tokens should have zero grad in all modes
            torch.testing.assert_close(
                grad[0, 8:],
                torch.zeros_like(grad[0, 8:]),
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Degenerate inputs that could cause NaN, inf, or crashes."""

    def test_all_segments_different_docs(self):
        """No consecutive same-doc segments → zero loss for all modes."""
        B, T, n_sae, n_seg = 1, 12, 4, 3
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.tensor([[0, 1, 2]])  # all different

        for mode in FeaturePredLossMode:
            loss = compute_feature_pred_loss(
                logits, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            assert loss.item() == 0.0

    def test_padded_segments_negative_doc_id(self):
        """Padded segments with seg_doc_id=-1 should not form valid pairs."""
        B, T, n_sae, n_seg = 1, 12, 4, 4
        seg_map = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]])
        targets = torch.rand(B, n_seg, n_sae)
        # Segments 0,1 real (doc 0), segments 2,3 are padding
        seg_doc_id = torch.tensor([[0, 0, -1, -1]])

        for mode in FeaturePredLossMode:
            logits = torch.randn(B, T, n_sae)
            loss = compute_feature_pred_loss(
                logits, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            assert torch.isfinite(loss)

    def test_large_n_sae_features(self):
        """Reasonably large n_sae doesn't cause issues."""
        B, T, n_sae, n_seg = 1, 8, 1024, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        loss = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        assert torch.isfinite(loss)

    def test_two_segments_identical_logits(self):
        """Identical logits across tokens — no numerical issues from ties."""
        B, T, n_sae = 1, 8, 4
        logits = torch.zeros(B, T, n_sae)  # all identical
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.zeros(B, 2, dtype=torch.long)

        for mode in FeaturePredLossMode:
            loss = compute_feature_pred_loss(
                logits, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            assert torch.isfinite(loss)

    def test_mse_loss_fn_with_all_modes(self):
        """MSE loss fn works with all modes (since sigmoid is applied internally)."""
        B, T, n_sae, n_seg = 1, 8, 4, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        for mode in FeaturePredLossMode:
            loss = compute_feature_pred_loss(
                logits, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.MSE
            )
            assert torch.isfinite(loss)
            assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Numerical property tests
# ---------------------------------------------------------------------------


class TestNumericalProperties:
    """Tests for expected mathematical properties of the loss functions."""

    def test_loss_decreases_with_better_prediction(self):
        """Loss should decrease when logits are pushed toward correct targets."""
        B, T, n_sae, n_seg = 1, 8, 4, 2
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        # Target: all 1.0 for next segment
        targets = torch.zeros(B, n_seg, n_sae)
        targets[:, 1, :] = 1.0
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

        # Bad prediction (negative logits → sigmoid < 0.5 → far from 1.0)
        logits_bad = torch.full((B, T, n_sae), -5.0)
        # Good prediction (large positive → sigmoid ≈ 1.0)
        logits_good = torch.full((B, T, n_sae), 5.0)

        for mode in FeaturePredLossMode:
            loss_bad = compute_feature_pred_loss(
                logits_bad, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            loss_good = compute_feature_pred_loss(
                logits_good, seg_map, targets, seg_doc_id, mode, FeaturePredLossFn.BCE
            )
            assert loss_good.item() < loss_bad.item(), (
                f"Mode {mode}: good prediction should have lower loss"
            )

    def test_loss_nonnegative(self):
        """Loss should always be non-negative."""
        torch.manual_seed(0)
        for _ in range(5):
            B, T, n_sae, n_seg = 2, 16, 8, 4
            logits = torch.randn(B, T, n_sae)
            seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // (T // n_seg)
            seg_map = seg_map.clamp(max=n_seg - 1)
            targets = torch.rand(B, n_seg, n_sae)
            seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

            for mode in FeaturePredLossMode:
                for fn in FeaturePredLossFn:
                    loss = compute_feature_pred_loss(logits, seg_map, targets, seg_doc_id, mode, fn)
                    assert loss.item() >= -1e-7, f"Negative loss for {mode}/{fn}"

    def test_best_token_leq_full_segment_statistically(self):
        """Over many random inputs, best_token loss <= full_segment loss."""
        torch.manual_seed(42)
        violations = 0
        n_trials = 20
        for _ in range(n_trials):
            B, T, n_sae = 1, 16, 8
            n_seg = 4
            logits = torch.randn(B, T, n_sae)
            seg_map = torch.arange(T).unsqueeze(0) // (T // n_seg)
            seg_map = seg_map.clamp(max=n_seg - 1)
            targets = torch.rand(B, n_seg, n_sae)
            seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

            loss_best = _best_token_loss(
                logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE
            )
            loss_full = _full_segment_loss(
                logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE
            )
            if loss_best.item() > loss_full.item() + 1e-5:
                violations += 1
        assert violations == 0, f"best_token > full_segment in {violations}/{n_trials} trials"


# ---------------------------------------------------------------------------
# compute_feature_pred_loss — max_pool mode
# ---------------------------------------------------------------------------


class TestMaxPoolLoss:
    """``_max_pool_loss``: per-feature max of logits across segment positions,
    then BCE/MSE against the next segment's target."""

    def _make_simple_inputs(self, B=1, T=8, n_seg=2, n_sae=4):
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, T // 2 :] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        return logits, seg_map, targets, seg_doc_id

    def test_returns_scalar(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _max_pool_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossFn.BCE,
        )
        assert loss.dim() == 0

    def test_loss_is_finite(self):
        logits, seg_map, targets, seg_doc_id = self._make_simple_inputs()
        loss = _max_pool_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossFn.BCE,
        )
        assert torch.isfinite(loss)

    def test_max_pool_uses_per_feature_max(self):
        """Hand-build: with 2-token segment, the max-pooled logit per
        feature equals the larger of the two logit rows. The loss must
        equal BCE(sigmoid(max_logit), target) for that one (source)
        segment."""
        B, n_seg = 1, 2
        # Segment 0 spans positions 0..1, segment 1 spans positions 2..3
        seg_map = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        # Logits: position 0 = [-2, 1, 0]; position 1 = [3, -1, 0.5]
        # → max per feature for seg 0 = [3, 1, 0.5]
        logits = torch.tensor(
            [[[-2.0, 1.0, 0.0], [3.0, -1.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )
        targets = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]])
        # Manually compute expected loss
        max_logit_seg0 = torch.tensor([3.0, 1.0, 0.5])
        target_seg1 = targets[0, 1]  # [0.5, 0.5, 0.5]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            max_logit_seg0,
            target_seg1,
            reduction="mean",
        )
        # Aggregation: weight by next-seg length / total target tokens.
        # Only seg 0 is a valid source. next_seg_lengths[0] = len(seg 1) = 2.
        # total = 2. weight = 1.0. So expected = bce.
        expected = bce
        loss = _max_pool_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossFn.BCE,
        )
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_gradient_flows_through_max_element(self):
        """scatter_reduce(amax) is autograd-aware. Gradients should flow
        through the max element only."""
        B, n_seg = 1, 2
        logits = torch.tensor(
            [
                [
                    [1.0, 0.0],  # seg 0 pos 0
                    [3.0, 2.0],  # seg 0 pos 1 — both are max for seg 0
                    [0.0, 0.0],  # seg 1 (not source)
                    [0.0, 0.0],
                ]
            ],
            requires_grad=True,
        )
        seg_map = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        targets = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        loss = _max_pool_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossFn.BCE,
        )
        loss.backward()
        grad = logits.grad
        assert grad is not None
        # Position 1 was max for both features → grad must be non-zero
        assert grad[0, 1].abs().sum() > 0
        # Position 0 lost the max → grad must be exactly zero
        torch.testing.assert_close(
            grad[0, 0],
            torch.zeros_like(grad[0, 0]),
        )

    def test_cross_document_masking(self):
        """Different docs → no valid source → loss = 0."""
        B, T, n_sae, n_seg = 1, 4, 2, 2
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.tensor([[0, 1]])  # different docs
        loss = _max_pool_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossFn.BCE,
        )
        assert loss.item() == 0.0

    def test_max_pool_dispatches_through_compute_feature_pred_loss(self):
        from steering.conditional.encoder_decoder.feature_prediction import (
            compute_feature_pred_loss,
        )

        B, T, n_sae, n_seg = 1, 8, 4, 2
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[:, 4:] = 1
        targets = torch.rand(B, n_seg, n_sae)
        seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
        a = _max_pool_loss(logits, seg_map, targets, seg_doc_id, FeaturePredLossFn.BCE)
        b = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.MAX_POOL,
            FeaturePredLossFn.BCE,
        )
        torch.testing.assert_close(a, b)
