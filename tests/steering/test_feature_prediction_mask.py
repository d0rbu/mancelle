"""Tests for feature-mask gating of FP targets and loss.

Verifies:
- Test A: zero-loss AND zero-gradient for masked dims (blacklist + whitelist)
- Test B: I5 — no mask ⇒ byte-identical loss, targets, and gradients vs pre-change

All four loss modes × both loss functions are covered.
"""

from __future__ import annotations

import pytest
import torch

from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction import compute_feature_pred_loss
from steering.conditional.feature_mask import FeatureMask
from tests.feature_prediction_helpers import dense_targets_from_sparse

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_setup():
    """B=1, T=16, n_sae=8, 4 segments of 4 tokens each, single document.

    Mirrors the fixture used in test_feature_prediction.py for consistency.
    """
    B, T, n_sae = 1, 16, 8
    torch.manual_seed(42)
    logits = torch.randn(B, T, n_sae, requires_grad=True)
    # 4 equal segments, 4 tokens each
    seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])
    seg_doc_id = torch.zeros(1, 4, dtype=torch.long)
    # Dense targets for 4 segments
    enc_fids = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])  # (4, 2)
    enc_percentiles = torch.tensor([[0.3, 0.7], [0.5, 0.9], [0.1, 0.4], [0.8, 0.2]])
    enc_mask = torch.ones(4, 2, dtype=torch.bool)
    targets = dense_targets_from_sparse(enc_fids, enc_percentiles, enc_mask, n_sae).unsqueeze(0)
    return logits, seg_map, targets, seg_doc_id, n_sae


# ---------------------------------------------------------------------------
# compute_feature_pred_loss mask gating: Test A + Test B
# ---------------------------------------------------------------------------


ALL_MODES = list(FeaturePredLossMode)
ALL_FNS = list(FeaturePredLossFn)


def _compute_loss_and_grad(
    logits_data: torch.Tensor,
    seg_map: torch.Tensor,
    targets: torch.Tensor,
    seg_doc_id: torch.Tensor,
    loss_mode: FeaturePredLossMode,
    loss_fn: FeaturePredLossFn,
    feature_mask: FeatureMask | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, logits.grad) — allocates fresh logits with requires_grad."""
    logits = logits_data.detach().requires_grad_(True)
    loss = compute_feature_pred_loss(
        logits,
        seg_map,
        targets,
        seg_doc_id,
        loss_mode,
        loss_fn,
        feature_mask=feature_mask,
    )
    loss.backward()
    return loss.detach(), logits.grad.detach()  # ty: ignore[unresolved-attribute]


class TestMaskZeroLossAndGrad:
    """Test A: masked dims contribute zero loss AND zero gradient."""

    @pytest.mark.parametrize("loss_mode", ALL_MODES)
    @pytest.mark.parametrize("loss_fn", ALL_FNS)
    def test_blacklist_zero_grad_on_masked_dim(self, simple_setup, loss_mode, loss_fn):
        """Perturbing only a blacklisted logit column leaves loss unchanged
        and yields zero gradient on that column.

        BCE correctness: binary_cross_entropy_with_logits(0, 0) = log(2) ≠ 0.
        Zeroing the target is NOT sufficient — masked dims must be excluded
        from the loss mean (numerator + denominator).
        """
        logits_data, seg_map, targets, seg_doc_id, n_sae = simple_setup
        # Blacklist feature dim 3 (the middle of n_sae=8)
        D = 3
        mask = FeatureMask(mode="blacklist", ids=(D,))

        loss_base, grad_base = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=mask,
        )

        # Perturb only column D in logits and recompute
        logits_perturbed = logits_data.clone()
        logits_perturbed[..., D] += 10.0  # large perturbation
        loss_pert, grad_pert = _compute_loss_and_grad(
            logits_perturbed,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=mask,
        )

        # Loss must be unchanged when only a masked dim is perturbed
        assert torch.allclose(loss_base, loss_pert, atol=1e-5), (
            f"[{loss_mode}/{loss_fn}] Loss changed after perturbing masked dim {D}: "
            f"{loss_base.item()} vs {loss_pert.item()}"
        )

        # Gradient on column D must be exactly zero
        assert grad_base[..., D].abs().max().item() == 0.0, (
            f"[{loss_mode}/{loss_fn}] grad[..., {D}] non-zero for blacklisted dim: "
            f"max={grad_base[..., D].abs().max().item()}"
        )

        # Gradients on non-masked dims must be non-zero (sanity check)
        non_masked_grad = grad_base.clone()
        non_masked_grad[..., D] = 0
        assert non_masked_grad.abs().sum().item() > 0, (
            f"[{loss_mode}/{loss_fn}] No gradient on any non-masked dim"
        )

    @pytest.mark.parametrize("loss_mode", ALL_MODES)
    @pytest.mark.parametrize("loss_fn", ALL_FNS)
    def test_whitelist_zero_grad_on_non_whitelisted_dim(self, simple_setup, loss_mode, loss_fn):
        """Only whitelisted dims contribute loss and gradient.

        Non-whitelisted dim D must: have zero gradient AND perturbing it leaves
        loss unchanged.
        """
        logits_data, seg_map, targets, seg_doc_id, n_sae = simple_setup
        # Whitelist only dims 0, 1, 2 — exclude dim 5, 6, 7
        D_excluded = 6
        mask = FeatureMask(mode="whitelist", ids=(0, 1, 2))

        loss_base, grad_base = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=mask,
        )

        logits_perturbed = logits_data.clone()
        logits_perturbed[..., D_excluded] += 10.0
        loss_pert, _ = _compute_loss_and_grad(
            logits_perturbed,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=mask,
        )

        assert torch.allclose(loss_base, loss_pert, atol=1e-5), (
            f"[{loss_mode}/{loss_fn}] Loss changed after perturbing non-whitelisted dim {D_excluded}"
        )
        assert grad_base[..., D_excluded].abs().max().item() == 0.0, (
            f"[{loss_mode}/{loss_fn}] grad[..., {D_excluded}] non-zero for non-whitelisted dim"
        )

    @pytest.mark.parametrize("loss_fn", ALL_FNS)
    def test_bce_zero_zero_masked_dim_is_not_log2(self, simple_setup, loss_fn):
        """Explicit BCE correctness: without masking, BCE(0,0)=log2≠0.

        Perturbing ONLY the logit of a masked dim (while all other logits/targets
        are fixed) must NOT change the loss when the mask is active.
        This verifies the dim is truly excluded from the reduction.
        """
        logits_data, seg_map, targets, seg_doc_id, n_sae = simple_setup
        # Use zero logits so masked dim also contributes log2 under BCE if included
        logits_zero = torch.zeros_like(logits_data)
        # Targets: all zeros
        targets_zero = torch.zeros_like(targets)

        D = 3
        mask = FeatureMask(mode="blacklist", ids=(D,))

        loss_base, grad_base = _compute_loss_and_grad(
            logits_zero,
            seg_map,
            targets_zero,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            loss_fn,
            feature_mask=mask,
        )

        # Perturb dim D: if it were included, this would change the loss.
        logits_perturbed = logits_zero.clone()
        logits_perturbed[..., D] = 100.0  # very different from 0
        loss_pert, grad_pert = _compute_loss_and_grad(
            logits_perturbed,
            seg_map,
            targets_zero,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            loss_fn,
            feature_mask=mask,
        )

        # Loss must be unchanged (dim D excluded from reduction entirely)
        assert torch.allclose(loss_base, loss_pert, atol=1e-5), (
            f"[{loss_fn}] Loss changed after perturbing masked dim {D}: "
            f"{loss_base.item()} vs {loss_pert.item()}"
        )

        # Grad on dim D must be zero with mask (for BCE: sigmoid(0)-0=0.5 without mask)
        assert grad_base[..., D].abs().max().item() == 0.0, (
            f"[{loss_fn}] Masked dim {D} has non-zero gradient"
        )
        assert grad_pert[..., D].abs().max().item() == 0.0


class TestI5NoMaskByteIdentical:
    """Test B: feature_mask=None produces byte-identical results."""

    @pytest.mark.parametrize("loss_mode", ALL_MODES)
    @pytest.mark.parametrize("loss_fn", ALL_FNS)
    def test_none_mask_loss_identical(self, simple_setup, loss_mode, loss_fn):
        """compute_feature_pred_loss with feature_mask=None == without the param."""
        logits_data, seg_map, targets, seg_doc_id, n_sae = simple_setup

        loss_pre, grad_pre = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
        )
        loss_none, grad_none = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=None,
        )

        # Exact bitwise identity (no RNG, no branching when None)
        assert loss_pre.item() == loss_none.item(), (
            f"[{loss_mode}/{loss_fn}] Loss differs with feature_mask=None: "
            f"{loss_pre.item()} vs {loss_none.item()}"
        )
        (
            torch.testing.assert_close(grad_pre, grad_none, atol=0, rtol=0),
            (f"[{loss_mode}/{loss_fn}] Gradients differ with feature_mask=None"),
        )

    @pytest.mark.parametrize("loss_mode", ALL_MODES)
    @pytest.mark.parametrize("loss_fn", ALL_FNS)
    def test_none_mask_grad_identical(self, simple_setup, loss_mode, loss_fn):
        """Gradient tensors are bit-identical when feature_mask=None."""
        logits_data, seg_map, targets, seg_doc_id, n_sae = simple_setup

        _, grad_pre = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
        )
        _, grad_none = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=None,
        )
        torch.testing.assert_close(grad_pre, grad_none, atol=0, rtol=0)


class TestBestTokenSelectionWithMask:
    """BEST_TOKEN-specific: per-token loss for token selection must also use allowed dims only."""

    def test_best_token_selection_not_skewed_by_masked_dim(self):
        """Token selection in BEST_TOKEN must not depend on masked dims.

        We construct a case where the "best" token changes if the masked dim
        is included in selection. With masking, the correct token (best on
        allowed dims only) should be selected.

        Setup (n_sae=4, blacklist dim 3):
          targets[seg1] = [1, 1, 1, 0]  (dim 3 target=0)
          Token 0 logits = [10, 10, 10, -10]:  good on dims 0-2 (BCE≈0), bad on dim 3
          Token 1 logits = [-10, -10, -10, 10]: bad on dims 0-2 (BCE≈10), good on dim 3

        Without mask: token 1 "wins" because BCE(10, 0) + BCE(-10, 1)*3 < BCE(-10,1)*3 + BCE(10,1)*... actually:
          Token 0 unmasked loss: mean of [BCE(10,1), BCE(10,1), BCE(10,1), BCE(-10,0)]
            = mean of [≈0, ≈0, ≈0, ≈0] = ≈0 (both good predictions)

        Simpler: use target all-ones:
          Token 0: logits [10,10,10,-10] vs target [1,1,1,1]
            dims 0-2: BCE(10,1)≈0 (good); dim 3: BCE(-10,1)≈10 (bad)
          Token 1: logits [-10,-10,-10,10] vs target [1,1,1,1]
            dims 0-2: BCE(-10,1)≈10 (bad); dim 3: BCE(10,1)≈0 (good)
          Without mask: both have same mean (3*10+0)/4 = 7.5 → tie broken by scatter_reduce
          With mask (dim 3 excluded): token 0 loss = (3*BCE(10,1))/3 ≈ 0; token 1 = 10
          → token 0 selected with mask active.
        """
        B, T, n_sae = 1, 8, 4
        # 2 segments: tokens 0-3 = seg 0 (source), tokens 4-7 = seg 1 (target)
        seg_map = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        seg_doc_id = torch.zeros(1, 2, dtype=torch.long)
        # Targets for segment 1: all ones so high positive logit = low BCE loss
        targets = torch.zeros(1, 2, n_sae)
        targets[0, 1, :] = 1.0  # target = 1 for all dims

        # Blacklist dim 3
        # Token 0: logits [10,10,10,-10] — very good on dims 0-2, bad on dim 3
        # Token 1: logits [-10,-10,-10,10] — bad on dims 0-2, good on dim 3
        logits_data = torch.zeros(B, T, n_sae)
        logits_data[0, 0, 0] = 10.0
        logits_data[0, 0, 1] = 10.0
        logits_data[0, 0, 2] = 10.0
        logits_data[0, 0, 3] = -10.0  # bad on masked dim
        logits_data[0, 1, 0] = -10.0
        logits_data[0, 1, 1] = -10.0
        logits_data[0, 1, 2] = -10.0
        logits_data[0, 1, 3] = 10.0  # good on masked dim (but masked)

        mask = FeatureMask(mode="blacklist", ids=(3,))

        # With masking (only dims 0-2 count):
        # token 0 loss on allowed dims: BCE(10,1)*3/3 ≈ 0 (best)
        # token 1 loss on allowed dims: BCE(-10,1)*3/3 ≈ 10 (worst)
        # → token 0 must be selected
        loss_masked, grad_masked = _compute_loss_and_grad(
            logits_data,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.BEST_TOKEN,
            FeaturePredLossFn.BCE,
            feature_mask=mask,
        )

        # Token 0 should be selected (best on allowed dims) → has gradient on dims 0-2
        assert grad_masked[0, 0, :3].abs().sum().item() > 0, (
            "Token 0 should be selected: expected grad on allowed dims 0-2"
        )
        # Token 1 should NOT be selected → no gradient
        assert grad_masked[0, 1].abs().sum().item() == 0.0, (
            "Token 1 should not be selected when dim 3 is masked"
        )
        # Masked dim 3 has zero grad regardless of which token was selected
        assert grad_masked[..., 3].abs().max().item() == 0.0, "Masked dim 3 must have zero gradient"
        # Source segment tokens 2-3 are also never selected
        assert grad_masked[0, 2].abs().sum().item() == 0.0
        assert grad_masked[0, 3].abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# H1: fp16/bf16 overflow + denominator accuracy tests
# ---------------------------------------------------------------------------


def _make_large_sae_inputs(
    n_sae: int,
    dtype: torch.dtype,
    *,
    blacklist_ids: tuple[int, ...] | None = None,
    whitelist_ids: tuple[int, ...] | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FeatureMask]:
    """B=1, T=4, n_sae=n_sae, 2 segments of 2 tokens, single document.

    Targets are non-zero (uniform 0.5) so BCE loss is well above 0 when the
    allowed universe is non-empty.  Uses CPU tensors so no CUDA needed.
    """
    torch.manual_seed(seed)
    B, T = 1, 4
    logits = torch.randn(B, T, n_sae, dtype=dtype)
    seg_map = torch.tensor([[0, 0, 1, 1]])
    seg_doc_id = torch.zeros(1, 2, dtype=torch.long)
    # Targets: non-zero so loss > 0 on any allowed dim
    targets = torch.full((B, 2, n_sae), 0.5, dtype=torch.float32)

    if blacklist_ids is not None:
        mask = FeatureMask(mode="blacklist", ids=blacklist_ids)
    elif whitelist_ids is not None:
        mask = FeatureMask(mode="whitelist", ids=whitelist_ids)
    else:
        raise ValueError("Must supply blacklist_ids or whitelist_ids")

    return logits, seg_map, targets, seg_doc_id, mask


class TestFP16LargeNAllowedNoOverflow:
    """H1: fp16 allowed.sum() overflow → silent zero loss regression.

    With n_sae >= 65504 and fp16 logits, the old code cast allowed to fp16
    and called allowed.sum() → overflowed to inf → loss = 0.0 and grad = 0.
    The fix must compute allowed and n_allowed in float32 regardless of
    logits.dtype.
    """

    def test_fp_loss_n_allowed_no_overflow_fp16_large_sae_bce(self):
        """fp16 logits + n_sae=65536, blacklist=(7,) → loss finite & >0 (BCE)."""
        n_sae = 65536
        logits, seg_map, targets, seg_doc_id, mask = _make_large_sae_inputs(
            n_sae,
            dtype=torch.float16,
            blacklist_ids=(7,),
        )
        loss = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
            feature_mask=mask,
        )
        assert torch.isfinite(loss), (
            f"fp16 BCE loss is not finite: {loss.item()} (overflow bug not fixed?)"
        )
        assert loss.item() > 0.0, (
            "fp16 BCE loss is 0.0 (overflow bug: allowed.sum()→inf → loss=0; not fixed?)"
        )

    def test_fp_loss_n_allowed_no_overflow_fp16_large_sae_mse(self):
        """fp16 logits + n_sae=65536, blacklist=(7,) → loss finite & >0 (MSE)."""
        n_sae = 65536
        logits, seg_map, targets, seg_doc_id, mask = _make_large_sae_inputs(
            n_sae,
            dtype=torch.float16,
            blacklist_ids=(7,),
        )
        loss = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.MSE,
            feature_mask=mask,
        )
        assert torch.isfinite(loss), (
            f"fp16 MSE loss is not finite: {loss.item()} (overflow bug not fixed?)"
        )
        assert loss.item() > 0.0, (
            "fp16 MSE loss is 0.0 (overflow bug: allowed.sum()→inf → loss=0; not fixed?)"
        )


class TestBF16DenominatorAccuracy:
    """H1: bf16 denominator accuracy vs fp32 reference.

    bf16 has fp32-level accumulation in PyTorch sum but the value held in
    bf16 before summing has limited mantissa (7 bits). For small counts the
    integer representation is exact; for counts ~1000 there can be rounding.
    After the fix (always fp32 denominator), the difference from an fp32
    reference run must be < 1% and the zero-grad invariant must still hold.
    """

    def test_fp_loss_bf16_n_allowed_accurate(self):
        """bf16 logits, whitelist of 1000 dims out of 4096 → close to fp32 reference.

        Asserts:
        - |loss_bf16 - loss_fp32| / loss_fp32 < 1e-2  (denominator accuracy)
        - perturbing ONLY a non-whitelisted logit dim yields EXACTLY 0.0 loss delta
          (zero-grad invariant still holds after the fp32 fix)
        """
        n_sae = 4096
        whitelist_ids = tuple(range(1000))  # exactly 1000 allowed dims

        # fp32 reference
        logits_fp32, seg_map, targets, seg_doc_id, mask = _make_large_sae_inputs(
            n_sae,
            dtype=torch.float32,
            whitelist_ids=whitelist_ids,
            seed=1,
        )
        loss_fp32 = compute_feature_pred_loss(
            logits_fp32,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
            feature_mask=mask,
        )

        # bf16 version — same logits cast to bf16
        logits_bf16 = logits_fp32.to(torch.bfloat16)
        loss_bf16 = compute_feature_pred_loss(
            logits_bf16,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
            feature_mask=mask,
        )

        assert torch.isfinite(loss_bf16), f"bf16 loss not finite: {loss_bf16.item()}"
        rel_err = abs(loss_bf16.item() - loss_fp32.item()) / (loss_fp32.item() + 1e-12)
        assert rel_err < 1e-2, (
            f"bf16 loss deviates too much from fp32 reference: "
            f"bf16={loss_bf16.item():.6f}, fp32={loss_fp32.item():.6f}, "
            f"rel_err={rel_err:.4f} (should be <1%)"
        )

        # Zero-grad invariant: perturbing a masked dim must yield exactly 0.0 delta
        # Masked dim: any dim NOT in whitelist_ids, e.g. dim 1000
        D_masked = 1000
        logits_perturbed = logits_bf16.clone()
        logits_perturbed[..., D_masked] += 5.0
        loss_pert = compute_feature_pred_loss(
            logits_perturbed,
            seg_map,
            targets,
            seg_doc_id,
            FeaturePredLossMode.FULL_SEGMENT,
            FeaturePredLossFn.BCE,
            feature_mask=mask,
        )
        assert loss_pert.item() == loss_bf16.item(), (
            f"bf16: perturbing masked dim {D_masked} changed loss: "
            f"base={loss_bf16.item()}, pert={loss_pert.item()} "
            "(zero-grad invariant violated)"
        )


class TestAllMaskedIsZeroNotNaN:
    """H1 (all-masked edge): allowed.sum()==0 → clamp(min=1) → loss==0.0,
    not NaN, not inf.  Covers all 4 modes × both loss fns.
    """

    ALL_MODES = list(FeaturePredLossMode)
    ALL_FNS = list(FeaturePredLossFn)

    @pytest.mark.parametrize("loss_mode", ALL_MODES)
    @pytest.mark.parametrize("loss_fn", ALL_FNS)
    def test_fp_loss_all_masked_is_zero_not_nan(self, loss_mode, loss_fn):
        """FeatureMask blacklist of ALL dims → loss==0.0, not NaN, not inf."""
        n_sae = 8
        # Blacklist every dim → allowed.sum()==0
        all_ids = tuple(range(n_sae))
        mask = FeatureMask(mode="blacklist", ids=all_ids)

        B, T = 1, 4
        torch.manual_seed(7)
        logits = torch.randn(B, T, n_sae)
        seg_map = torch.tensor([[0, 0, 1, 1]])
        seg_doc_id = torch.zeros(1, 2, dtype=torch.long)
        targets = torch.rand(B, 2, n_sae)

        loss = compute_feature_pred_loss(
            logits,
            seg_map,
            targets,
            seg_doc_id,
            loss_mode,
            loss_fn,
            feature_mask=mask,
        )

        assert not torch.isnan(loss), (
            f"[{loss_mode}/{loss_fn}] all-masked loss is NaN (div-by-zero?)"
        )
        assert not torch.isinf(loss), f"[{loss_mode}/{loss_fn}] all-masked loss is inf"
        assert loss.item() == 0.0, (
            f"[{loss_mode}/{loss_fn}] all-masked loss should be 0.0, got {loss.item()}"
        )
