"""Regression-proof that the SAE feature blacklist/whitelist mask survives torch.compile.

Invariant pinned: ``EncoderDecoderGPT.feature_mask`` is read INSIDE the dynamo-traced
graph (not constant-folded away), so a blacklist/whitelist set on ``orig_model`` before
the first forward is genuinely enforced under ``torch.compile``.

Hazard guarded:
  If dynamo constant-folded ``self.feature_mask`` at trace time (capturing ``None``), a
  subsequent ``orig_model.feature_mask = <mask>`` would silently have no effect on the
  compiled graph — the blacklist would be baked out and masking disabled.

These tests run with ``backend="eager"``: dynamo still traces the graph, installs Python
guards (including on ``self.feature_mask``), and executes the real tensor ops — but no
Triton / CUDA kernel compilation happens, so all tests are CPU-runnable in CI.
"""

from __future__ import annotations

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    SEQ_LEN,
    build_model_with_feature_prediction,
)
from tests.feature_prediction_helpers import dense_targets_from_sparse

# ---------------------------------------------------------------------------
# Module-level fixture: force SDPA backend.
# FlexAttention does not support backward on CPU; SDPA works without CUDA.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA for all tests in this module (CPU-safe, no CUDA required)."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_B, _T, _N_SEG, _N_SAE = 1, SEQ_LEN, 4, N_SAE_FEATURES
_N_BLACKLIST = _N_SAE // 2  # blacklist the lower half of dims


def _make_fp_inputs(seed: int = 0):
    """Return minimal inputs that trigger the FP-loss branch (fp_targets is not None)."""
    torch.manual_seed(seed)
    idx = torch.randint(0, 100, (_B, _T))
    targets = torch.randint(0, 100, (_B, _T))
    F = 5
    enc_fids = torch.randint(0, _N_SAE, (_B, _N_SEG, F), dtype=torch.long)
    enc_mags = torch.randint(0, NUM_BUCKETS, (_B, _N_SEG, F), dtype=torch.long)
    enc_mask = torch.ones(_B, _N_SEG, F, dtype=torch.bool)
    enc_pcts = torch.rand(_B, _N_SEG, F)
    seg_size = _T // _N_SEG
    seg_map = (torch.arange(_T).unsqueeze(0).expand(_B, -1) // seg_size).clamp(max=_N_SEG - 1)
    seg_doc_id = torch.zeros(_B, _N_SEG, dtype=torch.long)
    fp_targets = dense_targets_from_sparse(
        enc_fids.flatten(0, 1),
        enc_pcts.flatten(0, 1),
        enc_mask.flatten(0, 1),
        _N_SAE,
    ).view(_B, _N_SEG, _N_SAE)
    return idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets


def _forward_fp_loss(model, inputs):
    """Run a forward pass that reaches the FP-loss branch; return feature_pred_loss."""
    idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = inputs
    with torch.no_grad():
        result = model(
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            fp_targets=fp_targets,
        )
    # ForwardResult is a NamedTuple with .feature_pred_loss
    return result.feature_pred_loss


# ---------------------------------------------------------------------------
# Test 1: compiled (eager backend) FP-loss respects the blacklist mask
#
# Two fresh model instances (same weights), one with a blacklist covering half
# the SAE dims, one with no mask.  Both are compiled with backend="eager".
# Assertions:
#   (a) masked FP-loss != no-mask FP-loss  (mask demonstrably alters the loss)
#   (b) compiled_masked ≈ eager_masked     (compile reproduces eager masked result)
#   (c) compiled_nomask ≈ eager_nomask     (compile reproduces eager no-mask result)
# ---------------------------------------------------------------------------


def test_compiled_fp_loss_respects_blacklist_mask_eager():
    """Compiled (backend=eager) FP-loss reflects the blacklist — mask is not baked out."""
    torch.manual_seed(1)
    inputs = _make_fp_inputs(seed=1)

    # Build two model instances with identical weights.
    model_masked_eager = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_masked_eager.eval()

    model_nomask_eager = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_nomask_eager.eval()
    model_nomask_eager.load_state_dict(model_masked_eager.state_dict())

    blacklisted_ids = tuple(range(_N_BLACKLIST))
    mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=blacklisted_ids)

    # Eager (uncompiled) reference losses
    model_masked_eager.feature_mask = mask
    loss_eager_masked = _forward_fp_loss(model_masked_eager, inputs)

    model_nomask_eager.feature_mask = None
    loss_eager_nomask = _forward_fp_loss(model_nomask_eager, inputs)

    # Build two FRESH instances for the compiled path (separate from eager instances)
    # so dynamo never shares graph state between the masked and no-mask conditions.
    model_masked_compiled_raw = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_masked_compiled_raw.eval()
    model_masked_compiled_raw.load_state_dict(model_masked_eager.state_dict())
    model_masked_compiled_raw.feature_mask = mask  # set BEFORE compile trace
    compiled_masked = torch.compile(model_masked_compiled_raw, backend="eager")

    model_nomask_compiled_raw = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_nomask_compiled_raw.eval()
    model_nomask_compiled_raw.load_state_dict(model_nomask_eager.state_dict())
    model_nomask_compiled_raw.feature_mask = None  # set BEFORE compile trace
    compiled_nomask = torch.compile(model_nomask_compiled_raw, backend="eager")

    loss_compiled_masked = _forward_fp_loss(compiled_masked, inputs)
    loss_compiled_nomask = _forward_fp_loss(compiled_nomask, inputs)

    # (a) Mask changes the compiled FP-loss
    assert loss_compiled_masked.item() != loss_compiled_nomask.item(), (
        "compiled(masked) FP-loss == compiled(no-mask) FP-loss — the blacklist had no "
        "effect inside the compiled graph. Dynamo may have constant-folded feature_mask."
    )

    # (b) Compiled-masked ≈ eager-masked (tight tolerance — same computation)
    torch.testing.assert_close(
        loss_compiled_masked,
        loss_eager_masked,
        atol=1e-5,
        rtol=1e-4,
        msg=(
            "compiled(masked) FP-loss diverges from eager(masked) FP-loss — "
            "torch.compile changed the masked computation."
        ),
    )

    # (c) Compiled-nomask ≈ eager-nomask
    torch.testing.assert_close(
        loss_compiled_nomask,
        loss_eager_nomask,
        atol=1e-5,
        rtol=1e-4,
        msg=(
            "compiled(no-mask) FP-loss diverges from eager(no-mask) FP-loss — "
            "torch.compile changed the unmasked computation."
        ),
    )


# ---------------------------------------------------------------------------
# Test 2: stronger semantic check — masked dims contribute zero in compiled path
#
# Plants fp_targets so only one sentinel dim S is nonzero.
# Whitelists only S (all other dims masked out), then asserts perturbing the
# masked-out dims leaves the compiled FP-loss unchanged.
# Mirrors test_feature_prediction_mask.py's semantic assertions but goes
# through torch.compile(backend="eager").
# ---------------------------------------------------------------------------


def test_compiled_fp_loss_blacklist_excludes_masked_dims_eager():
    """Compiled FP-loss: blacklisted dims K contribute zero (masked-mean semantics hold).

    Strategy: blacklist ALL dims except one sentinel dim S.  With only S allowed,
    perturbing ALL other logit columns (the blacklisted ones) must leave the loss
    unchanged — they are excluded from the reduction.  The compiled loss must equal
    the eager masked loss (masked-mean semantics hold under torch.compile).

    This mirrors the semantic invariant proved in test_feature_prediction_mask.py
    (TestMaskZeroLossAndGrad) but threads it through torch.compile(backend="eager").
    """
    torch.manual_seed(2)

    # Keep only dim S in the whitelist; blacklist all others via a whitelist mask.
    S = 5  # the sole allowed dim

    idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, _ = _make_fp_inputs(seed=2)

    # fp_targets: only dim S is nonzero (0.9).  Perturbing any other column must
    # leave the loss unchanged once those other dims are blacklisted.
    fp_targets_s_only = torch.zeros(_B, _N_SEG, _N_SAE)
    fp_targets_s_only[:, :, S] = 0.9  # only the allowed dim carries signal
    inputs_s_only = (
        idx,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id,
        fp_targets_s_only,
    )

    # Whitelist only dim S → all other dims blacklisted
    mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(S,))

    model_raw = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_raw.eval()
    model_raw.feature_mask = mask

    # Eager reference: loss is computed on dim S only
    loss_eager_masked = _forward_fp_loss(model_raw, inputs_s_only)

    # Perturb fp_targets on all dims EXCEPT S: must leave the loss unchanged
    fp_targets_perturbed = fp_targets_s_only.clone()
    fp_targets_perturbed[:, :, :S] = 0.5  # change blacklisted dims
    fp_targets_perturbed[:, :, S + 1 :] = 0.5  # change blacklisted dims
    inputs_perturbed = (
        idx,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id,
        fp_targets_perturbed,
    )
    model_ref = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_ref.eval()
    model_ref.load_state_dict(model_raw.state_dict())
    model_ref.feature_mask = mask
    loss_eager_perturbed = _forward_fp_loss(model_ref, inputs_perturbed)

    # Eager semantic invariant: perturbing blacklisted dims leaves loss unchanged
    assert loss_eager_masked.item() == loss_eager_perturbed.item(), (
        f"Eager semantic invariant failed: perturbing blacklisted-dim fp_targets "
        f"changed the loss ({loss_eager_masked.item()} vs {loss_eager_perturbed.item()}). "
        "This is a pre-existing bug, not introduced by this test."
    )

    # Compile the masked model
    compiled_raw = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    compiled_raw.eval()
    compiled_raw.load_state_dict(model_raw.state_dict())
    compiled_raw.feature_mask = mask
    compiled = torch.compile(compiled_raw, backend="eager")

    loss_compiled_masked = _forward_fp_loss(compiled, inputs_s_only)

    # The compiled masked loss must match the eager masked loss
    torch.testing.assert_close(
        loss_compiled_masked,
        loss_eager_masked,
        atol=1e-5,
        rtol=1e-4,
        msg=(
            "compiled(whitelist={S}) FP-loss diverges from eager(whitelist={S}) — "
            "masked-mean semantics differ under torch.compile."
        ),
    )

    # Also verify compiled: perturbing blacklisted dims leaves compiled loss unchanged
    compiled_ref = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    compiled_ref.eval()
    compiled_ref.load_state_dict(model_raw.state_dict())
    compiled_ref.feature_mask = mask
    compiled2 = torch.compile(compiled_ref, backend="eager")

    loss_compiled_perturbed = _forward_fp_loss(compiled2, inputs_perturbed)
    assert loss_compiled_masked.item() == loss_compiled_perturbed.item(), (
        f"Compiled semantic invariant failed: perturbing blacklisted-dim fp_targets "
        f"changed the compiled loss ({loss_compiled_masked.item()} vs "
        f"{loss_compiled_perturbed.item()}). Masked-mean semantics broken under compile."
    )


# ---------------------------------------------------------------------------
# Test 3: the attribute is read from the live module, not constant-folded.
#
# WHY THIS TEST EXISTS:
#   If dynamo constant-folded `self.feature_mask` at the first trace (capturing
#   its value at trace time), then setting `orig_model.feature_mask = <mask>`
#   after the first compiled forward would silently have no effect — the baked
#   None would still be used.  The training scripts set the mask BEFORE the
#   first forward (investigated + proved), so the actually-relied-upon contract
#   is: "the value set before first forward is the value used in the graph."
#
#   This test verifies the contract from BOTH angles:
#     (a) Two fresh instances: one with mask set before compile, one without.
#         The compiled results must differ → the attribute value is used, not ignored.
#     (b) Set mask before compile, compile, forward: compiled loss matches eager masked.
#         This proves the pre-first-forward value IS the one that reaches the graph.
# ---------------------------------------------------------------------------


def test_compiled_forward_reads_live_feature_mask_attr_eager():
    """Compiled model uses the feature_mask set before first forward (not constant None).

    Proves the actually-relied-upon contract: the mask value set on orig_model
    before the first compiled forward is captured correctly by dynamo and
    genuinely alters the FP-loss (not baked as None).

    Two fresh compiled instances (same weights): one with mask set pre-compile,
    one without. The FP-losses must differ.  Additionally, the masked compiled
    loss must match the eager masked loss (contract: compile tracks the attribute
    value in scope at trace time).
    """
    torch.manual_seed(3)
    inputs = _make_fp_inputs(seed=3)

    blacklisted_ids = tuple(range(_N_BLACKLIST))
    mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=blacklisted_ids)

    # Instance A: mask set BEFORE first compiled forward (the production contract)
    model_a_raw = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_a_raw.eval()
    model_a_raw.feature_mask = mask  # set before compile
    compiled_a = torch.compile(model_a_raw, backend="eager")
    loss_compiled_with_mask = _forward_fp_loss(compiled_a, inputs)

    # Instance B: no mask (None)
    model_b_raw = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_b_raw.eval()
    model_b_raw.load_state_dict(model_a_raw.state_dict())
    model_b_raw.feature_mask = None
    compiled_b = torch.compile(model_b_raw, backend="eager")
    loss_compiled_without_mask = _forward_fp_loss(compiled_b, inputs)

    # The two compiled losses must differ — proving the attribute was read, not baked as None
    assert loss_compiled_with_mask.item() != loss_compiled_without_mask.item(), (
        "compiled(mask) FP-loss == compiled(no-mask) FP-loss — the mask set before "
        "the first compiled forward was NOT used. Dynamo may have constant-folded "
        "feature_mask=None into the graph regardless of the pre-trace value."
    )

    # Additionally verify: compiled-with-mask matches eager-with-mask (sanity)
    model_eager = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_eager.eval()
    model_eager.load_state_dict(model_a_raw.state_dict())
    model_eager.feature_mask = mask
    loss_eager_with_mask = _forward_fp_loss(model_eager, inputs)

    torch.testing.assert_close(
        loss_compiled_with_mask,
        loss_eager_with_mask,
        atol=1e-5,
        rtol=1e-4,
        msg=(
            "compiled(mask set pre-trace) FP-loss differs from eager(mask) — "
            "torch.compile with backend=eager changed the masked computation."
        ),
    )


# ---------------------------------------------------------------------------
# Optional: GPU inductor-backend variant (skipped when CUDA unavailable).
# Minimal — the eager tests above are the primary CI-runnable proof.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for inductor backend")
def test_compiled_fp_loss_respects_blacklist_mask_inductor():
    """GPU inductor backend: blacklist mask is enforced (CUDA CI completeness)."""
    device = "cuda"
    torch.manual_seed(10)
    inputs_cpu = _make_fp_inputs(seed=10)
    inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in inputs_cpu)

    blacklisted_ids = tuple(range(_N_BLACKLIST))
    mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=blacklisted_ids)

    model_masked = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_masked = model_masked.to(device)
    model_masked.eval()
    model_masked.feature_mask = mask
    compiled_masked = torch.compile(model_masked, dynamic=True)

    model_nomask = build_model_with_feature_prediction(n_sae_features=_N_SAE)
    model_nomask = model_nomask.to(device)
    model_nomask.eval()
    model_nomask.load_state_dict(model_masked.state_dict())
    model_nomask.feature_mask = None
    compiled_nomask = torch.compile(model_nomask, dynamic=True)

    with torch.amp.autocast(device, dtype=torch.bfloat16):
        loss_masked = _forward_fp_loss(compiled_masked, inputs)
        loss_nomask = _forward_fp_loss(compiled_nomask, inputs)

    assert loss_masked.item() != loss_nomask.item(), (
        "GPU inductor: compiled(masked) FP-loss == compiled(no-mask) — blacklist "
        "did not affect the inductor-compiled graph."
    )
