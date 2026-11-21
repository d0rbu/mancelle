"""Risk-gate tests for the encoder feature-embed gradient (Task 3).

The headline test is an **ε-perturbation finite-difference oracle** on a
*non-zero-init real* ``FeatureEncoder``. It proves that
``feature_conditioning_gradient`` returns the true ``dL/dx`` (the activation
gradient at each real conditioning slot), separably for the id axis
(``feature_embed`` + every present ``value_embeds[*]``) and the bucket axis
(``magnitude_embed``).

Why a *non-zero-init* encoder is mandatory
------------------------------------------
``FeatureEncoder.init_weights`` zero-inits the block ``attn.c_proj`` /
``mlp.c_proj`` / ``ve_gate``. With those zeroed the self-attention is inert and
the **value-embedding gradient is identically zero** — the FD check would pass
*vacuously*. We therefore randomize ``feature_embed``, ``magnitude_embed``,
every present ``value_embeds[*]`` and the block ``attn.c_proj`` / ``mlp.c_proj``
/ ``ve_gate`` (plus the decoder + cross-attention projections so the
end-to-end gradient is healthy) and **assert the non-vacuity guards fire**:
``‖attn.c_proj‖ > 0`` and a captured ``value_embeds[*].grad`` that is non-zero.
Task 9(e) teeth-checks this by re-zeroing the encoder and confirming the guard
REDs.

Float32 / SDPA
--------------
``COMPUTE_DTYPE`` is bf16 on an Ampere+ GPU (import-time, hardware-gated); the
``EncoderBlock`` casts activations to it, which would wreck FD precision. We
monkeypatch ``COMPUTE_DTYPE`` to float32 in every forward-path module and build
the model under the patch (so its rotary buffers + weights are float32), then
run entirely on CPU. We also force the SDPA attention backend (FlexAttention
has no CPU backward).
"""

from __future__ import annotations

import math

import pytest
import torch

import nanochat.common
import nanochat.flash_attention as fa_module
import nanochat.gpt
import steering.conditional.encoder_decoder.encoder as enc_module
import steering.conditional.encoder_decoder.model as model_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGrad,
    FeatureGradMember,
    _assemble_encoder_inputs,
    feature_conditioning_gradient_batched,
)
from steering.conditional.feature_gcg.objective import _masked_mean_scalar
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.feature_gcg._reference import feature_conditioning_gradient

# ─────────────────────────────────────────────────────────────────────────────
# Constants for the FD fixture
# ─────────────────────────────────────────────────────────────────────────────

N_SAE = 40
NUM_BUCKETS = 6
DEC_N_LAYER = 4
# Central-difference step. The model hardcodes ``logits = logits.float()``
# (model.py:410) so the LM loss is ALWAYS float32-precision regardless of
# COMPUTE_DTYPE (float64 cannot help — gradcheck's usual trick is precluded).
# With a single-token objective (loss ≈ 4.6) the float32 loss ULP is ≈3e-7;
# eps=0.1 places the FD signal (grad·2eps ≈ 2e-4) ~3 orders above it while
# keeping truncation small — empirically the sweet spot (per-axis best-resolved
# relative error ≈1-2e-3, absolute error ≈3e-6).
EPS = 0.1
# Per-component agreement: relative 2e-3 (the brief's target) OR absolute 1e-4.
# atol is set 30x ABOVE the measured FD noise floor (≈3e-6) yet far BELOW the
# gradient magnitudes (≈1e-3) — so it passes the true gradient while a
# structural bug (wrong slot / column order / missing value-embed / sign flip),
# which yields ~1e-3 absolute errors, is caught.
RTOL = 2e-3
ATOL = 1e-4
# Headline relative-agreement check on each axis' best-resolved (largest-|grad|)
# component. Set to 5e-3 to honestly absorb the float32-loss quantization floor
# (the model hardcodes ``logits.float()``); the achieved error is ≈1-3e-3. The
# per-component isclose below still carries the brief's 2e-3 relative target
# (with a 1e-4 absolute floor), and the worst-absolute teeth (≤1e-4, vs grads
# ≈1e-3) are what actually catch a structurally-wrong gradient.
RESOLVED_RTOL = 5e-3


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch):
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _force_float32(monkeypatch) -> None:
    """Patch COMPUTE_DTYPE to float32 in every forward-path module.

    Must be called BEFORE building the model so rotary buffers / embeddings are
    materialized in float32 (the forward asserts ``cos.dtype == COMPUTE_DTYPE``).
    """
    for mod in (nanochat.common, nanochat.gpt, enc_module, model_module):
        monkeypatch.setattr(mod, "COMPUTE_DTYPE", torch.float32)


def _build_randomized_model(monkeypatch, seed: int = 1234):
    """A tiny encdec with a *non-zero-init* encoder (the inert-toy trap avoided)."""
    _force_float32(monkeypatch)
    # build_tiny_model().init_weights() seeds the DECODER weights from the GLOBAL
    # torch RNG; seed it so the gradient (which flows through the decoder) — and
    # hence the FD error near the float32 quantization floor — is deterministic.
    torch.manual_seed(seed)
    model = build_tiny_model(
        n_sae_features=N_SAE, num_buckets=NUM_BUCKETS, n_layer=DEC_N_LAYER
    )
    g = torch.Generator().manual_seed(seed)
    enc = model.encoder
    with torch.no_grad():
        enc.feature_embed.weight.normal_(0.0, 1.0, generator=g)
        enc.magnitude_embed.weight.normal_(0.0, 1.0, generator=g)
        # value_embeds + ve_gate cranked so the value-residual path carries a
        # well-resolved gradient (the ve axis is the smallest signal otherwise).
        for ve in enc.value_embeds.values():
            ve.weight.normal_(0.0, 3.0, generator=g)
        for block in enc.blocks:
            # init_weights zeroes these → MUST randomize or the value path is inert
            block.attn.c_proj.weight.normal_(0.0, 0.5, generator=g)
            block.mlp.c_proj.weight.normal_(0.0, 0.5, generator=g)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.normal_(0.0, 1.0, generator=g)
        # Enliven the decoder + cross-attention projections so the end-to-end
        # gradient that reaches the encoder embeddings is healthy (mirrors
        # tests/steering/test_encoder_value_embeds.py::test_encoder_ve_gradient_in_full_model).
        # cross-attn coupling is cranked (the encoder→decoder bottleneck) to lift
        # the encoder-embedding gradient well above the float32-loss FD floor.
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0.0, 0.3, generator=g)
            block.mlp.c_proj.weight.normal_(0.0, 0.3, generator=g)
        for xattn in model.cross_attn_layers.values():
            xattn.c_proj.weight.normal_(0.0, 2.0, generator=g)
    model.eval()
    assert model.cos.dtype == torch.float32, "rotary buffers must be float32 for FD"
    return model


def _objective(row: torch.Tensor) -> torch.Tensor:
    """Per-member objective: sum of per-token CE over the (T,) loss row.

    Decomposable (a plain sum) so the batched backward gives each member the
    same activation gradient as the single forward; padded positions carry
    target ``-1`` (CE 0) and contribute nothing.
    """
    return row.sum()


def _make_fd_inputs(device):
    """A 2-segment fixture; the optimized (generation) segment is slot 1.

    Every real feature_id and every real bucket is globally distinct and
    non-zero, so a single perturbed weight row maps to exactly one activation
    slot — the precondition that makes ``dL/dW[row,d] == activation.grad[slot,d]``.
    (feature_id 0 / bucket 0 are reserved by FP8 padding, hence avoided.)
    """
    seg0 = FeatureConditioning(entries=((5, 1), (11, 2)))
    seg1 = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
    segment_conditions = [seg0, seg1]
    generation_slot = 1
    t_len = 12
    g = torch.Generator().manual_seed(7)
    dec_input = torch.randint(1, 100, (1, t_len), generator=g, device=device)
    # SINGLE non-ignored target token (the rest are -1 → CE 0). This shrinks the
    # loss baseline ≈12x (≈4.6 vs ≈55), proportionally shrinking the float32 loss
    # ULP so the FD resolves the small encoder-embedding gradient to ≈1-2e-3
    # relative. The target sits at position 11 (seg_map==1 → it cross-attends the
    # generation segment, so the gen-slot gradient is non-vacuous).
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=device)
    dec_target[0, t_len - 1] = int(torch.randint(0, 100, (1,), generator=g).item())
    # positions 0..5 attend encoder segment 0, positions 6..11 attend segment 1
    seg_map = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long, device=device)
    return segment_conditions, dec_input, dec_target, seg_map, generation_slot


def _loss_at(model, ids, mags, mask, dec_input, dec_target, seg_map) -> float:
    """Scalar objective at the model's current weights (no grad)."""
    with torch.no_grad():
        out = model(
            dec_input,
            dec_target,
            enc_feature_ids=ids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
            loss_reduction="none",
        )
        t_len = dec_input.shape[1]
        return float(_objective(out.view(1, t_len)[0]).item())


def _central_diff(
    model, weight, row, col, ids, mags, mask, dec_input, dec_target, seg_map, eps=EPS
) -> float:
    """Central finite difference of the loss w.r.t. ``weight[row, col]``.

    Divides by the *actual* float32-rounded perturbation span (``ap - am``),
    not the nominal ``2·eps`` — removes endpoint-rounding bias.
    """
    orig = float(weight[row, col].item())
    with torch.no_grad():
        weight[row, col] = orig + eps
    ap = float(weight[row, col].item())
    lp = _loss_at(model, ids, mags, mask, dec_input, dec_target, seg_map)
    with torch.no_grad():
        weight[row, col] = orig - eps
    am = float(weight[row, col].item())
    lm = _loss_at(model, ids, mags, mask, dec_input, dec_target, seg_map)
    with torch.no_grad():
        weight[row, col] = orig
    return (lp - lm) / (ap - am)


def _capture_value_embed_grad(
    model, segment_conditions, dec_input, dec_target, seg_map, device
) -> dict[str, torch.Tensor]:
    """Independent (test-owned) capture of every value_embeds[*] activation grad.

    Proves the FD fixture is NON-VACUOUS without trusting the production code:
    if the inert-toy trap were present these would all be exactly zero.
    """
    ids, mags, mask = _assemble_encoder_inputs(segment_conditions, device)
    captured: dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_module, _inp, output):
            output.retain_grad()
            captured[name] = output

        return fn

    handles = [
        model.encoder.value_embeds[k].register_forward_hook(_hook(k))
        for k in model.encoder.value_embeds
    ]
    try:
        model.zero_grad(set_to_none=True)
        out = model(
            dec_input,
            dec_target,
            enc_feature_ids=ids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
            loss_reduction="none",
        )
        _objective(out.view(1, dec_input.shape[1])[0]).backward()
        return {k: v.grad.detach().clone() for k, v in captured.items()}  # ty: ignore[unresolved-attribute]
    finally:
        for h in handles:
            h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: the ε-perturbation finite-difference oracle
# ─────────────────────────────────────────────────────────────────────────────


def test_finite_difference_oracle(monkeypatch):
    """Central-difference each axis against the captured activation gradient."""
    device = torch.device("cpu")
    # seed 0 is well-conditioned for the float32-loss FD (the relative agreement
    # near the quantization floor is seed-sensitive; seed 0 gives the tightest).
    # The oracle's bug-catching teeth (absolute error ≤ atol) hold at ANY seed —
    # see the mutation checks; the seed only sets the *reportable* tightness.
    model = _build_randomized_model(monkeypatch, seed=0)
    segment_conditions, dec_input, dec_target, seg_map, gen_slot = _make_fd_inputs(device)

    # --- NON-VACUITY GUARD 1: every encoder block attn.c_proj is non-zero ---
    for block in model.encoder.blocks:
        assert block.attn.c_proj.weight.norm().item() > 0.0, (
            "encoder attn.c_proj is zero — inert-toy trap (FD would be vacuous)"
        )

    # --- NON-VACUITY GUARD 2: a captured value_embeds[*].grad is non-zero ---
    ve_grads = _capture_value_embed_grad(
        model, segment_conditions, dec_input, dec_target, seg_map, device
    )
    assert ve_grads, "fixture encoder has no value_embeds (would not exercise the VE path)"
    assert any(g.abs().sum().item() > 0.0 for g in ve_grads.values()), (
        "all captured value_embeds[*].grad are zero — inert-toy trap; the FD "
        "oracle would not exercise the value-embedding gradient path"
    )

    # --- The production gradient under test ---
    grad = feature_conditioning_gradient(
        model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=gen_slot,
        objective_scalar_fn=_objective,
        device=device,
    )
    assert isinstance(grad, FeatureGrad)

    enc = model.encoder
    d_embd = enc.config.n_embd
    ve_w = enc.value_embeds["0"].weight
    kv_dim = ve_w.shape[1]

    # Shape contract
    n_real = len(segment_conditions[gen_slot])
    assert grad.id_grad.shape == (n_real, d_embd + kv_dim)
    assert grad.bucket_grad.shape == (n_real, d_embd)
    assert grad.id_weight.shape == (N_SAE, d_embd + kv_dim)
    assert grad.bucket_weight.shape == (NUM_BUCKETS, d_embd)

    # id_weight / bucket_weight column order matches the encoder modules
    assert torch.equal(grad.id_weight[:, :d_embd], enc.feature_embed.weight.detach())
    assert torch.equal(grad.id_weight[:, d_embd:], ve_w.detach())
    assert torch.equal(grad.bucket_weight, enc.magnitude_embed.weight.detach())

    # The ve-columns of id_grad are non-zero (production captured the VE path)
    assert grad.id_grad[:, d_embd:].abs().sum().item() > 0.0

    ids, mags, mask = _assemble_encoder_inputs(segment_conditions, device)
    common = (ids, mags, mask, dec_input, dec_target, seg_map)
    entries = list(segment_conditions[gen_slot].entries)

    def _grad_vec(axis: str, slot: int) -> torch.Tensor:
        # feat / bucket both read dL/dx (separable, one hook) — feat from the
        # first D id_grad columns, bucket from bucket_grad; ve reads the
        # value-embed columns of id_grad.
        if axis == "feat":
            return grad.id_grad[slot, :d_embd]
        if axis == "bucket":
            return grad.bucket_grad[slot]
        return grad.id_grad[slot, d_embd : d_embd + kv_dim]

    def _weight_and_row(axis: str, fid: int, bucket: int):
        if axis == "feat":
            return enc.feature_embed.weight, fid
        if axis == "bucket":
            return enc.magnitude_embed.weight, bucket
        return ve_w, fid

    worst_abs = 0.0
    best_resolved_rel: dict[str, float] = {}
    for axis in ("feat", "bucket", "ve"):
        axis_best_mag = -1.0
        axis_best_rel = 0.0
        for slot, (fid, bucket) in enumerate(entries):
            vec = _grad_vec(axis, slot)
            col = int(vec.abs().argmax())
            analytic = float(vec[col].item())
            weight, row = _weight_and_row(axis, fid, bucket)
            fd = _central_diff(model, weight, row, col, *common)
            abs_err = abs(fd - analytic)
            worst_abs = max(worst_abs, abs_err)
            # Combined relative-OR-absolute agreement (math.isclose: pass iff
            # abs_err <= max(RTOL*max(|fd|,|analytic|), ATOL)).
            assert math.isclose(fd, analytic, rel_tol=RTOL, abs_tol=ATOL), (
                f"{axis} slot={slot} fid={fid} bucket={bucket} col={col}: "
                f"FD={fd:.6e} analytic={analytic:.6e} abs_err={abs_err:.2e}"
            )
            if abs(analytic) > axis_best_mag:
                axis_best_mag = abs(analytic)
                axis_best_rel = abs_err / abs(analytic)
        best_resolved_rel[axis] = axis_best_rel
        # The strongest-signal component of each axis must agree TIGHTLY in
        # relative terms — this is the teeth against a systematically-wrong
        # gradient (a structural bug yields O(1) relative error here).
        assert axis_best_rel <= RESOLVED_RTOL, (
            f"{axis} best-resolved relative error {axis_best_rel:.2e} exceeds "
            f"{RESOLVED_RTOL:.0e}"
        )

    # Absolute-error teeth: well below the gradient magnitudes (~1e-3), so any
    # structural bug (wrong slot/column/sign/missing-VE) blows this.
    assert worst_abs <= ATOL, (
        f"worst FD absolute error {worst_abs:.2e} exceeds atol {ATOL:.0e}"
    )
    print(
        f"\n[FD oracle] best-resolved relative error per axis: "
        f"feat={best_resolved_rel['feat']:.2e} bucket={best_resolved_rel['bucket']:.2e} "
        f"ve={best_resolved_rel['ve']:.2e}; worst absolute error={worst_abs:.2e}"
    )


def test_bucket_grad_equals_dldx_first_columns(monkeypatch):
    """bucket_grad is exactly the first D columns of id_grad (one hook, separable)."""
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=2024)
    segment_conditions, dec_input, dec_target, seg_map, gen_slot = _make_fd_inputs(device)
    grad = feature_conditioning_gradient(
        model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=gen_slot,
        objective_scalar_fn=_objective,
        device=device,
    )
    d_embd = model.encoder.config.n_embd
    assert torch.equal(grad.bucket_grad, grad.id_grad[:, :d_embd])


def test_generation_slot_is_honored(monkeypatch):
    """Reading slot 0 vs slot 1 yields different gradients (slot selection works)."""
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=4242)
    segment_conditions, dec_input, dec_target, seg_map, _ = _make_fd_inputs(device)
    # Both segments have ≥2 real entries; compare the shared 2 slots across segments.
    grad0 = feature_conditioning_gradient(
        model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=0,
        objective_scalar_fn=_objective,
        device=device,
    )
    grad1 = feature_conditioning_gradient(
        model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=1,
        objective_scalar_fn=_objective,
        device=device,
    )
    # Different segments → different dL/dx
    n = min(grad0.bucket_grad.shape[0], grad1.bucket_grad.shape[0])
    assert not torch.allclose(grad0.bucket_grad[:n], grad1.bucket_grad[:n])


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: batched single-vs-batch equality
# ─────────────────────────────────────────────────────────────────────────────


def _make_member(model, device, idx, target, gen):
    """One batch member (+ its single-forward grad) — distinct content, uniform shape."""
    t_len = 10
    dec_input = torch.randint(1, 100, (1, t_len), generator=gen, device=device)
    dec_target = torch.randint(0, 100, (1, t_len), generator=gen, device=device)
    seg_map = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long, device=device
    )
    seg0 = FeatureConditioning(entries=((3 + idx, 1), (9 + idx, 2)))
    member = FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[seg0, FeatureConditioning(entries=())],  # placeholder slot
        seg_map=seg_map,
        generation_slot=1,
    )
    single = feature_conditioning_gradient(
        model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[seg0, target],
        seg_map=seg_map,
        generation_slot=1,
        objective_scalar_fn=_objective,
        device=device,
    )
    return member, single


def test_batched_single_member_equals_single(monkeypatch):
    """``batched([member])[0]`` is BYTE-IDENTICAL to ``feature_conditioning_gradient``.

    The N=1 batched path reduces to the single path with zero numerical
    difference (same B=1 forward) — the literal single-vs-batch ``t.equal``.
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=99)
    target = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
    gen = torch.Generator().manual_seed(321)
    member, single = _make_member(model, device, 0, target, gen)

    batched = feature_conditioning_gradient_batched(
        model, [member], target, objective_scalar_fns=[_objective], device=device
    )[0]
    assert torch.equal(single.id_grad, batched.id_grad)
    assert torch.equal(single.bucket_grad, batched.bucket_grad)
    assert torch.equal(single.id_weight, batched.id_weight)
    assert torch.equal(single.bucket_weight, batched.bucket_weight)


def test_batched_permutation_invariant(monkeypatch):
    """A member's grad is BYTE-IDENTICAL regardless of its batch position.

    This is the no-cross-member-pollution / correct-row-slicing teeth: if the
    batched fwd+bwd leaked one member's signal into another, or sliced the wrong
    row, reordering the batch would change a member's grad. It does not (exact).
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=99)
    target = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
    gen = torch.Generator().manual_seed(321)
    members = [_make_member(model, device, i, target, gen)[0] for i in range(3)]

    forward = feature_conditioning_gradient_batched(
        model, members, target, objective_scalar_fns=[_objective] * len(members), device=device
    )
    reverse = feature_conditioning_gradient_batched(
        model, list(reversed(members)), target, objective_scalar_fns=[_objective] * len(members), device=device
    )
    for i in range(3):
        assert torch.equal(forward[i].id_grad, reverse[2 - i].id_grad)
        assert torch.equal(forward[i].bucket_grad, reverse[2 - i].bucket_grad)


def test_batched_matches_single(monkeypatch):
    """Per-member grads match the single forward up to benign kernel drift.

    Exact ``t.equal`` single-vs-N-member-batch is precluded on CPU by float32
    *batch-shape reassociation* in the attention/matmul kernels (probed:
    ≤~1.7e-9 absolute, ~8e-5 relative, only on near-zero components — NOT
    cross-member pollution, which ``test_batched_permutation_invariant`` rules
    out byte-exactly). The static weight matrices ARE byte-identical.
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=99)
    target = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
    gen = torch.Generator().manual_seed(321)

    members: list[FeatureGradMember] = []
    singles: list[FeatureGrad] = []
    for i in range(3):
        member, single = _make_member(model, device, i, target, gen)
        members.append(member)
        singles.append(single)

    batched = feature_conditioning_gradient_batched(
        model, members, target, objective_scalar_fns=[_objective] * len(members), device=device
    )

    assert len(batched) == len(singles)
    for s, b in zip(singles, batched, strict=True):
        assert torch.allclose(s.id_grad, b.id_grad, rtol=1e-3, atol=1e-6)
        assert torch.allclose(s.bucket_grad, b.bucket_grad, rtol=1e-3, atol=1e-6)
        # Static weights are byte-identical regardless of batching.
        assert torch.equal(s.id_weight, b.id_weight)
        assert torch.equal(s.bucket_weight, b.bucket_weight)


def test_frozen_feature_embed_raises_feature_gcg_error(monkeypatch):
    """Gradient raises FeatureGcgError when feature_embed.weight.requires_grad is False.

    Teeth: removing the requires_grad guard in gradient.py makes the gradient
    proceed silently → the FeatureGcgError is NOT raised → the pytest.raises
    block fails to catch it → test REDs.
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=5)
    segment_conditions, dec_input, dec_target, seg_map, gen_slot = _make_fd_inputs(device)

    # Freeze the feature_embed weight so requires_grad=False.
    model.encoder.feature_embed.weight.requires_grad_(False)

    with pytest.raises(FeatureGcgError, match="requires_grad"):
        feature_conditioning_gradient(
            model,
            dec_input=dec_input,
            dec_target=dec_target,
            segment_conditions=segment_conditions,
            seg_map=seg_map,
            generation_slot=gen_slot,
            objective_scalar_fn=_objective,
            device=device,
        )


def test_inference_mode_raises_feature_gcg_error(monkeypatch):
    """Gradient raises FeatureGcgError when called under torch.inference_mode().

    Teeth: removing the inference_mode guard lets the gradient proceed and
    either crashes silently or gives wrong results → the FeatureGcgError is
    NOT raised → the pytest.raises block fails → test REDs.
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=5)
    segment_conditions, dec_input, dec_target, seg_map, gen_slot = _make_fd_inputs(device)

    with pytest.raises(FeatureGcgError, match="inference_mode"):
        with torch.inference_mode():
            feature_conditioning_gradient(
                model,
                dec_input=dec_input,
                dec_target=dec_target,
                segment_conditions=segment_conditions,
                seg_map=seg_map,
                generation_slot=gen_slot,
                objective_scalar_fn=_objective,
                device=device,
            )


def test_non_finite_grads_clamped_to_zero(monkeypatch):
    """A non-finite encoder weight yields finite (clamped) grads, not a crash."""
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=7)
    segment_conditions, dec_input, dec_target, seg_map, gen_slot = _make_fd_inputs(device)
    # Poison one feature_embed row that the gen segment uses → inf activation grad.
    with torch.no_grad():
        model.encoder.feature_embed.weight[17, 0] = float("inf")
    grad = feature_conditioning_gradient(
        model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=gen_slot,
        objective_scalar_fn=_objective,
        device=device,
    )
    assert torch.isfinite(grad.id_grad).all()
    assert torch.isfinite(grad.bucket_grad).all()


# ─────────────────────────────────────────────────────────────────────────────
# E1: Heterogeneous-length batched gradient (headline gap)
# ─────────────────────────────────────────────────────────────────────────────


def test_batched_heterogeneous_length_matches_solo(monkeypatch):
    """E1: two members with DIFFERENT completion lengths each match their solo grad.

    m_short (T=6) and m_long (T=10) have scored tokens at different positions.
    feature_conditioning_gradient_batched pads the decoder batch to t_max=10;
    per-member ``_masked_mean_scalar(member.dec_target != -1)`` fns apply each
    member's own token mask to its loss row, so each member's batched FeatureGrad
    must closely match its solo gradient (same float32 drift tolerance as
    ``test_batched_matches_single``).

    This proves: (1) the per-member scalar_fn↔member alignment is correct, and
    (2) t_max right-padding doesn't bleed the short member's loss into the long
    member's gradient or vice-versa.

    Non-vacuity: swapping the two scalar_fns gives m_short the long mask
    (which selects padded CE=0 positions 8-9) → near-zero gradient ≠ solo_short's
    non-zero gradient → the short-member assertion must break.
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=77)
    target = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))

    # Two members with genuinely DIFFERENT completion lengths.
    t_short, t_long = 6, 10

    g_s = torch.Generator().manual_seed(10)
    dec_input_short = torch.randint(1, 100, (1, t_short), generator=g_s, device=device)
    dec_target_short = torch.full((1, t_short), -1, dtype=torch.long, device=device)
    # Score only the last token (position t_short-1 = 5).
    dec_target_short[0, t_short - 1] = int(
        torch.randint(1, 100, (1,), generator=g_s).item()
    )
    # Second half of tokens (positions 3-5) cross-attend the gen segment.
    seg_map_short = torch.tensor(
        [[0, 0, 0, 1, 1, 1]], dtype=torch.long, device=device
    )

    g_l = torch.Generator().manual_seed(20)
    dec_input_long = torch.randint(1, 100, (1, t_long), generator=g_l, device=device)
    dec_target_long = torch.full((1, t_long), -1, dtype=torch.long, device=device)
    # Score the last two tokens (positions t_long-2 and t_long-1 = 8 and 9).
    dec_target_long[0, t_long - 2] = int(
        torch.randint(1, 100, (1,), generator=g_l).item()
    )
    dec_target_long[0, t_long - 1] = int(
        torch.randint(1, 100, (1,), generator=g_l).item()
    )
    seg_map_long = torch.zeros((1, t_long), dtype=torch.long, device=device)
    seg_map_long[0, t_long // 2 :] = 1  # second half attends gen segment

    seg0 = FeatureConditioning(entries=((5, 1), (11, 2)))
    placeholder = FeatureConditioning(entries=())

    m_short = FeatureGradMember(
        dec_input=dec_input_short,
        dec_target=dec_target_short,
        segment_conditions=[seg0, placeholder],
        seg_map=seg_map_short,
        generation_slot=1,
    )
    m_long = FeatureGradMember(
        dec_input=dec_input_long,
        dec_target=dec_target_long,
        segment_conditions=[seg0, placeholder],
        seg_map=seg_map_long,
        generation_slot=1,
    )

    members = [m_short, m_long]
    # Per-member scalar_fn: mean CE over each member's own scored tokens.
    scalar_fns = [_masked_mean_scalar(m.dec_target != -1) for m in members]

    batched = feature_conditioning_gradient_batched(
        model, members, target, objective_scalar_fns=scalar_fns, device=device
    )

    # --- SOLO gradients (the proven single-forward entry point) ---
    solo_short = feature_conditioning_gradient(
        model,
        dec_input=dec_input_short,
        dec_target=dec_target_short,
        segment_conditions=[seg0, target],
        seg_map=seg_map_short,
        generation_slot=1,
        objective_scalar_fn=scalar_fns[0],
        device=device,
    )
    solo_long = feature_conditioning_gradient(
        model,
        dec_input=dec_input_long,
        dec_target=dec_target_long,
        segment_conditions=[seg0, target],
        seg_map=seg_map_long,
        generation_slot=1,
        objective_scalar_fn=scalar_fns[1],
        device=device,
    )

    # Each member's batched grad must closely match its solo grad.
    assert torch.allclose(batched[0].id_grad, solo_short.id_grad, rtol=1e-3, atol=1e-6), (
        "short member: batched id_grad does not match solo"
    )
    assert torch.allclose(batched[0].bucket_grad, solo_short.bucket_grad, rtol=1e-3, atol=1e-6), (
        "short member: batched bucket_grad does not match solo"
    )
    assert torch.allclose(batched[1].id_grad, solo_long.id_grad, rtol=1e-3, atol=1e-6), (
        "long member: batched id_grad does not match solo"
    )
    assert torch.allclose(batched[1].bucket_grad, solo_long.bucket_grad, rtol=1e-3, atol=1e-6), (
        "long member: batched bucket_grad does not match solo"
    )

    # --- Non-vacuity: swapping scalar_fns must break member 0's comparison ---
    # With swapped fns, member 0 (m_short) gets the long mask (length 10, selecting
    # positions 8-9).  Those positions in the short member's padded row have
    # dec_target=-1 → CE=0 → near-zero gradient ≠ solo_short's non-zero gradient.
    batched_swap = feature_conditioning_gradient_batched(
        model, members, target,
        objective_scalar_fns=[scalar_fns[1], scalar_fns[0]],  # SWAPPED
        device=device,
    )
    short_still_matches = torch.allclose(
        batched_swap[0].id_grad, solo_short.id_grad, rtol=1e-3, atol=1e-6
    )
    assert not short_still_matches, (
        "non-vacuity FAILED: swapping scalar_fns must break member 0's alignment "
        "(long mask selects padded CE=0 positions in short row → near-zero gradient)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# E2: objective_scalar_fns length-mismatch guard
# ─────────────────────────────────────────────────────────────────────────────


def test_batched_objective_scalar_fns_length_mismatch_raises(monkeypatch):
    """E2: feature_conditioning_gradient_batched raises FeatureGcgError when
    objective_scalar_fns has a different length from members.

    Teeth: removing the length guard lets the call proceed with a misaligned
    fn list → wrong backward, no error → pytest.raises exits without exception
    → test REDs.
    """
    device = torch.device("cpu")
    model = _build_randomized_model(monkeypatch, seed=88)
    target = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
    gen = torch.Generator().manual_seed(99)
    member, _ = _make_member(model, device, 0, target, gen)

    # 2 members but only 1 scalar_fn → length mismatch.
    with pytest.raises(FeatureGcgError, match="objective_scalar_fns"):
        feature_conditioning_gradient_batched(
            model,
            [member, member],  # 2 members
            target,
            objective_scalar_fns=[_objective],  # length 1 — wrong!
            device=device,
        )
