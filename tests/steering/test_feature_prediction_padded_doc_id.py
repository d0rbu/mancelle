"""Regression test for seg_doc_id=-1 collision creating phantom same-doc pairs.

Bug: In feature_prediction.py, the valid_source_seg gate only checks
  (seg_doc_id[:, :-1] == seg_doc_id[:, 1:]) & seg_has_tokens[:, :-1] & seg_has_tokens[:, 1:]

Two adjacent padded slots have seg_doc_id = -1, so they spuriously match as
"same document". Today this is only prevented by seg_has_tokens (padded slots
have no tokens in normal operation). But when the dataloader's `seg_map.clamp(max=pad_n_seg-1)`
maps real tokens into a padded slot, `seg_has_tokens` flips True for the padded
slot and the cross-document loss contribution leaks through.

Fix: add explicit `seg_doc_id[:, :-1] >= 0` guard.
"""

from __future__ import annotations

import torch

from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    compute_feature_pred_loss,
)

# ---------------------------------------------------------------------------
# Direct mask check
# ---------------------------------------------------------------------------


class TestValidSourceMaskPaddedSlots:
    """Inspect the internal valid_source_seg logic by observing loss output."""

    def _make_collision_batch(self):
        """Craft a batch where positions [3, 4] in seg_doc_id are both -1
        (padded) AND both have tokens assigned to them via seg_map.

        This simulates the failure mode where `seg_map.clamp` pushes real
        tokens into padded slots, making seg_has_tokens=True for those slots.
        """
        B, T, n_sae, n_seg = 1, 12, 4, 5
        torch.manual_seed(2026)
        # seg_map assigns real tokens to EVERY segment index including padded slots 3 and 4
        seg_map = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4]])
        # Segments 0,1,2 are real (doc 0); segments 3,4 are padded (-1)
        seg_doc_id = torch.tensor([[0, 0, 0, -1, -1]])
        targets = torch.rand(B, n_seg, n_sae)
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        return logits, seg_map, targets, seg_doc_id, n_sae

    def test_full_segment_no_cross_pair_through_padded_slots(self):
        """Loss must NOT treat segments 3,4 (both seg_doc_id=-1) as same-doc pair.

        If the gate is buggy, segment 3's tokens (positions 6, 7) will receive
        gradient from the non-existent 3->4 prediction. With the fix, they
        should have zero gradient.
        """
        logits, seg_map, targets, seg_doc_id, n_sae = self._make_collision_batch()
        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT, FeaturePredLossFn.BCE,
        )
        loss.backward()
        grads = logits.grad[0]
        # Positions 6, 7 belong to segment 3 (padded, seg_doc_id=-1)
        # With the fix, they should have zero gradient because segment 3
        # is NOT a valid source (its own seg_doc_id is -1, or equivalently,
        # the `seg_doc_id[:, :-1] >= 0` check filters it out).
        grad_seg3 = grads[6:8].abs().sum()
        assert grad_seg3.item() < 1e-8, (
            f"Segment 3 (padded, seg_doc_id=-1) should NOT be a valid source, "
            f"but got grad sum {grad_seg3.item()}. Buggy gate matched -1 == -1."
        )

    def test_last_token_no_cross_pair_through_padded_slots(self):
        """LAST_TOKEN mode: padded seg 3 should not contribute via its last token."""
        logits, seg_map, targets, seg_doc_id, n_sae = self._make_collision_batch()
        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.LAST_TOKEN, FeaturePredLossFn.BCE,
        )
        loss.backward()
        grads = logits.grad[0]
        # Position 7 is the last token of segment 3; should have zero grad.
        grad_last_of_seg3 = grads[7].abs().sum()
        assert grad_last_of_seg3.item() < 1e-8, (
            f"Last token of padded segment 3 should have zero grad, got "
            f"{grad_last_of_seg3.item()}"
        )

    def test_best_token_no_cross_pair_through_padded_slots(self):
        """BEST_TOKEN mode: padded seg 3 should not contribute any min-loss token."""
        logits, seg_map, targets, seg_doc_id, n_sae = self._make_collision_batch()
        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN, FeaturePredLossFn.BCE,
        )
        loss.backward()
        grads = logits.grad[0]
        # Neither position 6 nor 7 (segment 3 tokens) should receive gradient.
        grad_seg3 = grads[6:8].abs().sum()
        assert grad_seg3.item() < 1e-8, (
            f"Best-token over padded segment 3 should be masked out, got "
            f"grad sum {grad_seg3.item()}"
        )

    def test_adjacent_padded_slots_at_position_3_4(self):
        """Direct check using the task-specified scenario: positions 3, 4 of
        seg_doc_id are both -1; the same_doc gate at index 3 must be False.

        We infer the gate's decision by constructing a batch where:
        - Segments 3, 4 are both padded (seg_doc_id=-1) AND have tokens assigned
        - Segments 0, 1 (real doc 0) provide baseline loss
        - Any loss contribution beyond the 0->1 pair indicates the padded-slot
          collision was not filtered.
        """
        B, T, n_sae = 1, 10, 4
        torch.manual_seed(0)
        # seg_map: 2 tokens each for segments 0, 1, 2, 3, 4
        # positions: 0,1 → seg 0; 2,3 → seg 1; 4,5 → seg 2; 6,7 → seg 3; 8,9 → seg 4
        seg_map = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])
        # seg_doc_id: only seg 0, 1 form a real pair (doc 0). Segs 2, 3, 4 are padded.
        # Note: segs 2, 3 are both -1 → buggy code treats as same-doc
        # Note: segs 3, 4 are both -1 → buggy code treats as same-doc
        seg_doc_id = torch.tensor([[0, 0, -1, -1, -1]])
        targets = torch.rand(B, 5, n_sae)

        # --- Baseline: make seg_doc_ids so there's NO possible padded collision
        seg_doc_id_safe = torch.tensor([[0, 0, -1, -2, -3]])  # all pad values unique

        logits1 = torch.randn(B, T, n_sae, requires_grad=True)
        logits2 = logits1.detach().clone().requires_grad_(True)

        loss_collision = compute_feature_pred_loss(
            logits1, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT, FeaturePredLossFn.BCE,
        )
        loss_safe = compute_feature_pred_loss(
            logits2, seg_map, targets, seg_doc_id_safe,
            FeaturePredLossMode.FULL_SEGMENT, FeaturePredLossFn.BCE,
        )
        # The two losses MUST be equal after the fix. Before the fix, the
        # collision case includes the phantom 2->3 and 3->4 pairs while
        # the safe case excludes them all.
        assert torch.allclose(loss_collision, loss_safe, atol=1e-6), (
            f"Padded-slot collision leaks loss: collision={loss_collision.item()}, "
            f"safe={loss_safe.item()}. The `seg_doc_id[:, :-1] >= 0` guard is missing."
        )


# ---------------------------------------------------------------------------
# Sanity: fix preserves correct behavior for real pairs
# ---------------------------------------------------------------------------


class TestFixDoesNotBreakRealPairs:
    """The `seg_doc_id >= 0` guard must NOT affect legitimate same-doc pairs."""

    def test_real_same_doc_pair_still_produces_loss(self):
        B, T, n_sae = 1, 8, 4
        torch.manual_seed(1)
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.tensor([[0, 0]])  # both real doc 0

        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT, FeaturePredLossFn.BCE,
        )
        loss.backward()
        # Segment 0 tokens (positions 0-3) should receive gradient
        assert logits.grad[0, :4].abs().sum() > 0  # ty: ignore[not-subscriptable]

    def test_different_real_docs_still_excluded(self):
        """Real but different docs (e.g., doc 0 and doc 1) still excluded."""
        B, T, n_sae = 1, 8, 4
        torch.manual_seed(2)
        logits = torch.randn(B, T, n_sae, requires_grad=True)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        targets = torch.rand(B, 2, n_sae)
        seg_doc_id = torch.tensor([[0, 1]])  # different real docs

        loss = compute_feature_pred_loss(
            logits, seg_map, targets, seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT, FeaturePredLossFn.BCE,
        )
        assert loss.item() == 0.0, (
            f"Different docs should produce zero loss, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Direct helper: introspect valid_source_seg behavior via loss isolation
# ---------------------------------------------------------------------------


class TestValidSourceMaskDirect:
    """Test the gate's behavior at specific indices via loss decomposition."""

    def test_position_3_4_both_padded_not_paired(self):
        """When seg_doc_id[3] == seg_doc_id[4] == -1 (both padded), and both
        have assigned tokens, the gate at position 3 MUST be False.

        We detect this by comparing:
        - Batch A: seg_doc_id = [0, 0, 0, -1, -1] (collision case at pos 3)
        - Batch B: seg_doc_id = [0, 0, 0, -1, -2] (no collision possible)
        If the gate is correct, both losses are identical.
        """
        B, T, n_sae = 1, 10, 4
        torch.manual_seed(7)
        seg_map = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])
        targets = torch.rand(B, 5, n_sae)

        for mode in FeaturePredLossMode:
            logits_a = torch.randn(B, T, n_sae, requires_grad=True)
            logits_b = logits_a.detach().clone().requires_grad_(True)

            seg_doc_id_collision = torch.tensor([[0, 0, 0, -1, -1]])
            seg_doc_id_safe = torch.tensor([[0, 0, 0, -1, -7]])

            loss_collision = compute_feature_pred_loss(
                logits_a, seg_map, targets, seg_doc_id_collision,
                mode, FeaturePredLossFn.BCE,
            )
            loss_safe = compute_feature_pred_loss(
                logits_b, seg_map, targets, seg_doc_id_safe,
                mode, FeaturePredLossFn.BCE,
            )
            assert torch.allclose(loss_collision, loss_safe, atol=1e-6), (
                f"Mode {mode}: collision case loss={loss_collision.item()} "
                f"diverges from safe={loss_safe.item()}; padded gate is buggy."
            )
