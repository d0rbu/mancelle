import pytest
import torch as t
import torch.nn.functional as F

from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    compute_feature_pred_loss,
    resolve_layer_index,
)
from tests.feature_prediction_helpers import dense_targets_from_sparse


class TestFeaturePredictionConfig:
    def test_defaults(self):
        cfg = FeaturePredictionConfig()
        assert cfg.weight == 0.1
        assert cfg.layer == -1
        # Default is MAX_POOL: per-feature max across segment positions, then
        # BCE/MSE against the next segment's target. Symmetric across the
        # segment, runs identically at training and inference, and avoids
        # last-position bias. BEST_TOKEN is impossible at inference (needs
        # future targets); LAST_TOKEN and FULL_SEGMENT remain as opt-ins.
        assert cfg.loss_mode == FeaturePredLossMode.MAX_POOL
        assert cfg.loss_fn == FeaturePredLossFn.BCE

    def test_enum_values(self):
        assert FeaturePredLossMode.FULL_SEGMENT == "full_segment"
        assert FeaturePredLossMode.LAST_TOKEN == "last_token"
        assert FeaturePredLossMode.MAX_POOL == "max_pool"
        assert FeaturePredLossMode.BEST_TOKEN == "best_token"
        assert FeaturePredLossFn.BCE == "bce"
        assert FeaturePredLossFn.MSE == "mse"

    def test_custom_values(self):
        cfg = FeaturePredictionConfig(
            weight=0.5,
            layer=-2,
            loss_mode=FeaturePredLossMode.LAST_TOKEN,
            loss_fn=FeaturePredLossFn.MSE,
        )
        assert cfg.weight == 0.5
        assert cfg.layer == -2


class TestResolveLayerIndex:
    def test_negative_one(self):
        assert resolve_layer_index(-1, 12) == 11

    def test_negative_two(self):
        assert resolve_layer_index(-2, 12) == 10

    def test_zero(self):
        assert resolve_layer_index(0, 12) == 0

    def test_positive(self):
        assert resolve_layer_index(5, 12) == 5

    def test_last_valid_positive(self):
        assert resolve_layer_index(11, 12) == 11

    def test_out_of_range_positive_raises(self):
        with pytest.raises(AssertionError):
            resolve_layer_index(12, 12)

    def test_out_of_range_negative_raises(self):
        with pytest.raises(AssertionError):
            resolve_layer_index(-13, 12)

    def test_negative_n_layer_raises(self):
        # n_layer must be positive
        assert resolve_layer_index(0, 1) == 0


# ---------------------------------------------------------------------------
# Reference implementations (pure Python, for correctness testing)
# ---------------------------------------------------------------------------


def _reference_full_segment_loss(logits, targets, seg_map, seg_doc_id, loss_fn, n_sae):
    """Pure Python reference: full_segment loss computation.

    Only predicts across segments within the same document.
    seg_doc_id: (B, n_seg) -- document index for each segment.
    """
    B, T, _ = logits.shape
    n_seg = targets.shape[1]
    total_loss = 0.0
    n_tokens = 0
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):  # s is source, s+1 is target
            # Skip cross-document pairs
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            # Skip if either segment has no tokens
            src_count = (seg_ids == s).sum().item()
            tgt_count = (seg_ids == s + 1).sum().item()
            if src_count == 0 or tgt_count == 0:
                continue
            target = targets[b, s + 1]  # dense target for next segment
            for pos in range(T):
                if seg_ids[pos] == s:
                    tok_logits = logits[b, pos]
                    if loss_fn == "bce":
                        tok_loss = F.binary_cross_entropy_with_logits(
                            tok_logits, target, reduction="mean"
                        )
                    else:
                        tok_loss = F.mse_loss(t.sigmoid(tok_logits), target, reduction="mean")
                    total_loss += tok_loss.item()
                    n_tokens += 1
    if n_tokens == 0:
        return 0.0
    return total_loss / n_tokens


def _reference_last_token_loss(logits, targets, seg_map, seg_doc_id, loss_fn, n_sae):
    """Pure Python reference: last_token loss computation.

    Only predicts across segments within the same document.
    """
    B, T, _ = logits.shape
    n_seg = targets.shape[1]
    total_loss = 0.0
    total_target_tokens = 0

    # First pass: compute total_target_tokens (only for valid same-doc target segments)
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            src_count = (seg_ids == s).sum().item()
            tgt_count = (seg_ids == s + 1).sum().item()
            if src_count == 0 or tgt_count == 0:
                continue
            total_target_tokens += tgt_count

    if total_target_tokens == 0:
        return 0.0

    # Second pass: compute weighted loss
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):  # source segments
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            source_positions = (seg_ids == s).nonzero(as_tuple=True)[0]
            if len(source_positions) == 0:
                continue
            next_seg_len = (seg_ids == s + 1).sum().item()
            if next_seg_len == 0:
                continue
            last_pos = source_positions[-1].item()
            target = targets[b, s + 1]
            tok_logits = logits[b, last_pos]
            if loss_fn == "bce":
                tok_loss = F.binary_cross_entropy_with_logits(tok_logits, target, reduction="mean")
            else:
                tok_loss = F.mse_loss(t.sigmoid(tok_logits), target, reduction="mean")
            weight = next_seg_len / total_target_tokens
            total_loss += tok_loss.item() * weight

    return total_loss


def _reference_best_token_loss(logits, targets, seg_map, seg_doc_id, loss_fn, n_sae):
    """Pure Python reference: best_token loss computation.

    Only predicts across segments within the same document.
    """
    B, T, _ = logits.shape
    n_seg = targets.shape[1]
    total_loss = 0.0
    total_target_tokens = 0

    # First pass: compute total_target_tokens
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            src_count = (seg_ids == s).sum().item()
            tgt_count = (seg_ids == s + 1).sum().item()
            if src_count == 0 or tgt_count == 0:
                continue
            total_target_tokens += tgt_count

    if total_target_tokens == 0:
        return 0.0

    # Second pass: compute best-token weighted loss
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            source_positions = (seg_ids == s).nonzero(as_tuple=True)[0]
            if len(source_positions) == 0:
                continue
            next_seg_len = (seg_ids == s + 1).sum().item()
            if next_seg_len == 0:
                continue
            target = targets[b, s + 1]
            best_loss = float("inf")
            for pos in source_positions:
                tok_logits = logits[b, pos]
                if loss_fn == "bce":
                    tok_loss = F.binary_cross_entropy_with_logits(
                        tok_logits, target, reduction="mean"
                    ).item()
                else:
                    tok_loss = F.mse_loss(t.sigmoid(tok_logits), target, reduction="mean").item()
                if tok_loss < best_loss:
                    best_loss = tok_loss
            weight = next_seg_len / total_target_tokens
            total_loss += best_loss * weight

    return total_loss


def _reference_max_pool_loss(logits, targets, seg_map, seg_doc_id, loss_fn, n_sae):
    """Pure Python reference: max_pool loss computation.

    For each (source, source+1) segment pair within the same document, take
    the per-feature max of source-segment logits and compute BCE/MSE
    against ``targets[s + 1]``. Aggregate weighted by next-segment length.
    """
    B, T, _ = logits.shape
    n_seg = targets.shape[1]
    total_loss = 0.0
    total_target_tokens = 0

    # First pass: compute total_target_tokens
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            if seg_doc_id[b, s].item() < 0:
                continue
            src_count = (seg_ids == s).sum().item()
            tgt_count = (seg_ids == s + 1).sum().item()
            if src_count == 0 or tgt_count == 0:
                continue
            total_target_tokens += tgt_count

    if total_target_tokens == 0:
        return 0.0

    # Second pass: compute max-pool weighted loss
    for b in range(B):
        seg_ids = seg_map[b]
        for s in range(n_seg - 1):
            if seg_doc_id[b, s].item() != seg_doc_id[b, s + 1].item():
                continue
            if seg_doc_id[b, s].item() < 0:
                continue
            source_positions = (seg_ids == s).nonzero(as_tuple=True)[0]
            if len(source_positions) == 0:
                continue
            next_seg_len = (seg_ids == s + 1).sum().item()
            if next_seg_len == 0:
                continue
            target = targets[b, s + 1]
            seg_logits = logits[b, source_positions]  # (n_src_tok, n_sae)
            max_logits = seg_logits.amax(dim=0)  # (n_sae,)
            if loss_fn == "bce":
                seg_loss = F.binary_cross_entropy_with_logits(
                    max_logits, target, reduction="mean"
                ).item()
            else:
                seg_loss = F.mse_loss(t.sigmoid(max_logits), target, reduction="mean").item()
            weight = next_seg_len / total_target_tokens
            total_loss += seg_loss * weight

    return total_loss


# ---------------------------------------------------------------------------
# Vectorized loss tests
# ---------------------------------------------------------------------------


class TestComputeFeaturePredLoss:
    """Test the vectorized loss computation against pure-Python reference implementations."""

    @pytest.fixture
    def simple_setup(self):
        """4 segments, 16 tokens, 8 SAE features. Single document (all same doc)."""
        B, T, n_sae = 1, 16, 8
        t.manual_seed(42)
        logits = t.randn(B, T, n_sae, requires_grad=True)
        # seg_map: 4 tokens per segment, 4 segments
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
        # All segments from same document
        seg_doc_id = t.zeros(1, 4, dtype=t.long)
        # Features for each segment
        enc_fids = t.tensor([[[0, 1], [2, 3], [4, 5], [6, 7]]])  # (1, 4, 2)
        enc_percentiles = t.tensor([[[0.3, 0.7], [0.5, 0.9], [0.1, 0.4], [0.8, 0.2]]])
        enc_mask = t.ones(1, 4, 2, dtype=t.bool)
        targets = dense_targets_from_sparse(
            enc_fids.view(4, 2), enc_percentiles.view(4, 2), enc_mask.view(4, 2), n_sae
        ).unsqueeze(0)  # (1, 4, n_sae)
        return logits, seg_map, targets, seg_doc_id, n_sae

    def test_full_segment_bce_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        ref = _reference_full_segment_loss(
            logits.detach(), targets, seg_map, seg_doc_id, "bce", n_sae
        )
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_last_token_bce_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.LAST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        ref = _reference_last_token_loss(
            logits.detach(), targets, seg_map, seg_doc_id, "bce", n_sae
        )
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_best_token_bce_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        ref = _reference_best_token_loss(
            logits.detach(), targets, seg_map, seg_doc_id, "bce", n_sae
        )
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_max_pool_bce_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.MAX_POOL,
            FeaturePredLossFn.BCE,
        )
        ref = _reference_max_pool_loss(logits.detach(), targets, seg_map, seg_doc_id, "bce", n_sae)
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_max_pool_mse_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.MAX_POOL,
            FeaturePredLossFn.MSE,
        )
        ref = _reference_max_pool_loss(logits.detach(), targets, seg_map, seg_doc_id, "mse", n_sae)
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_full_segment_mse_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.MSE,
        )
        ref = _reference_full_segment_loss(
            logits.detach(), targets, seg_map, seg_doc_id, "mse", n_sae
        )
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_last_token_mse_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.LAST_TOKEN,
            FeaturePredLossFn.MSE,
        )
        ref = _reference_last_token_loss(
            logits.detach(), targets, seg_map, seg_doc_id, "mse", n_sae
        )
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_best_token_mse_matches_reference(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.MSE,
        )
        ref = _reference_best_token_loss(
            logits.detach(), targets, seg_map, seg_doc_id, "mse", n_sae
        )
        assert abs(result.item() - ref) < 1e-4, f"got {result.item()}, expected {ref}"

    def test_gradient_flows(self, simple_setup):
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        result.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_best_token_gradient_sparse(self, simple_setup):
        """best_token should only produce gradients at the min-loss token per valid source segment."""
        logits, seg_map, targets, seg_doc_id, n_sae = simple_setup
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        result.backward()
        # For each valid source segment (0, 1, 2 -- seg 3 has no next), only 1 out of 4 tokens should have gradient
        for s in range(3):  # 3 valid source segments (0, 1, 2)
            seg_positions = (seg_map[0] == s).nonzero(as_tuple=True)[0]
            grads = logits.grad[0, seg_positions]  # (4, n_sae)
            nonzero_tokens = (grads.abs().sum(dim=-1) > 1e-8).sum().item()
            assert nonzero_tokens == 1, (
                f"Segment {s}: expected 1 gradient token, got {nonzero_tokens}"
            )


class TestEdgeCases:
    def test_single_segment_returns_zero(self):
        """Only one segment -> no source-target pairs -> zero loss."""
        logits = t.randn(1, 8, 4)
        seg_map = t.zeros(1, 8, dtype=t.long)
        targets = t.zeros(1, 1, 4)
        seg_doc_id = t.zeros(1, 1, dtype=t.long)
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        assert result.item() == 0.0

    def test_two_segments(self):
        """Two segments: segment 0 predicts segment 1, segment 1 has no target."""
        logits = t.randn(1, 8, 4)
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = t.rand(1, 2, 4)
        seg_doc_id = t.zeros(1, 2, dtype=t.long)
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        assert result.item() > 0.0  # Should have valid loss

    def test_uneven_segments(self):
        """Segments of different sizes."""
        logits = t.randn(1, 10, 4, requires_grad=True)
        seg_map = t.tensor([[0, 0, 1, 1, 1, 1, 1, 2, 2, 2]])
        targets = t.rand(1, 3, 4)
        seg_doc_id = t.zeros(1, 3, dtype=t.long)

        for mode in FeaturePredLossMode:
            result = compute_feature_pred_loss(
                logits,
                seg_map,
                targets,
                seg_doc_id,
                mode,
                FeaturePredLossFn.BCE,
            )
            assert result.item() > 0.0
            assert not t.isnan(result)

    def test_batch_dimension(self):
        """Multiple sequences in a batch."""
        B = 3
        logits = t.randn(B, 8, 4)
        seg_map = t.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 3],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],  # single segment -- no loss
            ]
        )
        targets = t.rand(B, 4, 4)
        seg_doc_id = t.zeros(B, 4, dtype=t.long)
        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        assert result.item() > 0.0

    def test_last_token_weighting_correctness(self):
        """Verify that last_token weighting matches expected proportions."""
        # 3 segments: [4 tokens, 4 tokens, 2 tokens]
        # Segment 0 -> predicts seg 1 (len 4), segment 1 -> predicts seg 2 (len 2)
        # total_target_tokens = 4 + 2 = 6
        # weight_0 = 4/6, weight_1 = 2/6
        logits = t.zeros(1, 10, 4)  # all zeros -> predictable loss
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2]])
        targets = t.ones(1, 3, 4) * 0.5  # uniform targets
        seg_doc_id = t.zeros(1, 3, dtype=t.long)

        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.LAST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        # Both source segments have same logits (0) and targets (0.5)
        # so per-segment loss is identical. Weights are 4/6 and 2/6, sum = 1.
        # Expected: bce_with_logits(0, 0.5) = ln(2) - 0 = 0.6931...
        expected_per_token = F.binary_cross_entropy_with_logits(
            t.zeros(4), t.ones(4) * 0.5, reduction="mean"
        ).item()
        # Weighted sum: expected_per_token * (4/6 + 2/6) = expected_per_token
        assert abs(result.item() - expected_per_token) < 1e-4


# ---------------------------------------------------------------------------
# Cross-document masking tests
# ---------------------------------------------------------------------------


class TestCrossDocumentMasking:
    """Tests that cross-document segment boundaries are respected."""

    def test_cross_doc_boundary_excluded_full_segment(self):
        """Two documents packed into one row -- last seg of doc A should NOT predict first seg of doc B."""
        B, T, n_sae = 1, 12, 4
        t.manual_seed(42)
        logits = t.randn(B, T, n_sae, requires_grad=True)
        # Doc A: segments 0,1 (4 tokens each), Doc B: segment 2 (4 tokens)
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        seg_doc_id = t.tensor([[0, 0, 1]])  # seg 0,1 = doc 0; seg 2 = doc 1
        targets = t.rand(B, 3, n_sae)

        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        # Only seg 0 -> seg 1 is valid (same doc). Seg 1 -> seg 2 crosses docs.
        # So only tokens in segment 0 (positions 0-3) should have loss.
        result.backward()
        # Segment 0 tokens should have gradient (valid source)
        assert logits.grad[0, :4].abs().sum() > 0  # ty: ignore[not-subscriptable]
        # Segment 1 tokens should have NO gradient (cross-doc target)
        assert logits.grad[0, 4:8].abs().sum() < 1e-8  # ty: ignore[not-subscriptable]
        # Segment 2 tokens should have NO gradient (no next segment)
        assert logits.grad[0, 8:].abs().sum() < 1e-8  # ty: ignore[not-subscriptable]

    def test_cross_doc_excluded_last_token(self):
        """last_token mode: only last token of valid same-doc source segments."""
        B, T, n_sae = 1, 12, 4
        logits = t.randn(B, T, n_sae, requires_grad=True)
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        seg_doc_id = t.tensor([[0, 0, 1]])
        targets = t.rand(B, 3, n_sae)

        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.LAST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        result.backward()
        # Only position 3 (last token of seg 0, valid source) should have gradient
        grad_norms = logits.grad[0].abs().sum(dim=-1)  # ty: ignore[not-subscriptable]
        assert grad_norms[3] > 1e-8
        assert (grad_norms[:3].abs() < 1e-8).all()  # not last token
        assert (grad_norms[4:].abs() < 1e-8).all()  # seg 1,2 not valid

    def test_cross_doc_excluded_best_token(self):
        """best_token mode: min-loss token only from valid same-doc source segments."""
        B, T, n_sae = 1, 12, 4
        logits = t.randn(B, T, n_sae, requires_grad=True)
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        seg_doc_id = t.tensor([[0, 0, 1]])
        targets = t.rand(B, 3, n_sae)

        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        result.backward()
        grad_norms = logits.grad[0].abs().sum(dim=-1)  # ty: ignore[not-subscriptable]
        # Exactly 1 token in segment 0 should have gradient
        assert (grad_norms[:4] > 1e-8).sum() == 1
        # No gradient in segments 1 or 2
        assert (grad_norms[4:] < 1e-8).all()

    def test_multiple_docs_multiple_valid_pairs(self):
        """Two docs, each with 2+ segments -- both contribute valid pairs."""
        B, T, n_sae = 1, 16, 4
        logits = t.randn(B, T, n_sae, requires_grad=True)
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
        seg_doc_id = t.tensor([[0, 0, 1, 1]])  # doc 0: seg 0,1; doc 1: seg 2,3
        targets = t.rand(B, 4, n_sae)

        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
        )
        result.backward()
        grad_norms = logits.grad[0].abs().sum(dim=-1)  # ty: ignore[not-subscriptable]
        # Seg 0 -> seg 1 valid, seg 1 -> seg 2 cross-doc, seg 2 -> seg 3 valid
        assert (grad_norms[:4] > 1e-8).all()  # seg 0: valid source
        assert (grad_norms[4:8] < 1e-8).all()  # seg 1: cross-doc target
        assert (grad_norms[8:12] > 1e-8).all()  # seg 2: valid source
        assert (grad_norms[12:] < 1e-8).all()  # seg 3: no next segment

    def test_all_single_segment_docs_zero_loss(self):
        """Each document has only one segment -> no valid pairs -> zero loss."""
        B, T, n_sae = 1, 12, 4
        logits = t.randn(B, T, n_sae)
        seg_map = t.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]])
        seg_doc_id = t.tensor([[0, 1, 2]])  # each seg is a different doc
        targets = t.rand(B, 3, n_sae)

        for mode in FeaturePredLossMode:
            result = compute_feature_pred_loss(
                logits,
                seg_map,
                targets,
                seg_doc_id,
                mode,
                FeaturePredLossFn.BCE,
            )
            assert result.item() == 0.0, f"Expected 0 loss for {mode}, got {result.item()}"

    def test_weighting_respects_doc_boundaries(self):
        """Verify total_target_tokens only counts targets from valid same-doc pairs."""
        B, T, n_sae = 1, 10, 4
        logits = t.zeros(B, T, n_sae)
        # Doc 0: seg 0 (2 tok), seg 1 (3 tok) | Doc 1: seg 2 (2 tok), seg 3 (3 tok)
        seg_map = t.tensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 3]])
        seg_doc_id = t.tensor([[0, 0, 1, 1]])
        targets = t.ones(B, 4, n_sae) * 0.5

        result = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.LAST_TOKEN,
            FeaturePredLossFn.BCE,
        )
        # Valid pairs: seg 0->1 (next_len=3), seg 2->3 (next_len=3)
        # total_target_tokens = 3 + 3 = 6
        # Both have same loss (logits=0, target=0.5) and same weight (3/6=0.5)
        expected = F.binary_cross_entropy_with_logits(
            t.zeros(n_sae), t.ones(n_sae) * 0.5, reduction="mean"
        ).item()
        # Weighted: expected * 0.5 + expected * 0.5 = expected
        assert abs(result.item() - expected) < 1e-4

    def test_cross_doc_matches_reference(self):
        """Vectorized implementation matches reference for multi-doc scenario."""
        B, T, n_sae = 2, 12, 4
        t.manual_seed(123)
        logits = t.randn(B, T, n_sae)
        seg_map = t.tensor(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            ]
        )
        seg_doc_id = t.tensor(
            [
                [0, 0, 1, 1],  # doc 0: seg 0,1; doc 1: seg 2,3
                [0, 1, -1, -1],  # doc 0: seg 0 only; doc 1: seg 1 only
            ]
        )
        targets = t.rand(B, 4, n_sae)

        for mode in FeaturePredLossMode:
            result = compute_feature_pred_loss(
                logits,
                seg_map,
                targets,
                seg_doc_id,
                mode,
                FeaturePredLossFn.BCE,
            )
            if mode == FeaturePredLossMode.FULL_SEGMENT:
                ref = _reference_full_segment_loss(
                    logits, targets, seg_map, seg_doc_id, "bce", n_sae
                )
            elif mode == FeaturePredLossMode.LAST_TOKEN:
                ref = _reference_last_token_loss(logits, targets, seg_map, seg_doc_id, "bce", n_sae)
            elif mode == FeaturePredLossMode.MAX_POOL:
                ref = _reference_max_pool_loss(logits, targets, seg_map, seg_doc_id, "bce", n_sae)
            else:
                ref = _reference_best_token_loss(logits, targets, seg_map, seg_doc_id, "bce", n_sae)
            assert abs(result.item() - ref) < 1e-4, f"{mode}: got {result.item()}, expected {ref}"
