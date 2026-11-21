"""Risk-gate tests for the feature_gcg objectives (Task 4).

Two objectives, two kinds of teeth:

* **Step 1a — ``lm_loss_on_masked_tokens``:** the forward-only ``losses`` matches
  a hand-computed masked-CE reference (``Σ CE over masked tokens / n_masked``); it
  leaves no parameter ``.grad`` (forward-only); ``gradients`` returns finite
  :class:`FeatureGrad`.

* **Step 1b — ``class_margin_objective`` (the spec's #1 adversary, §7.2):** the
  ``g_steer`` sign trap. The headline pins ``g_steer == −primary_margin_y`` against
  the INDEPENDENT canonical ``signed_class_margin`` in BOTH modes (a sign flip or a
  double swap both collapse to ``+primary_margin_y`` and RED it); a finite-
  difference oracle pins ``g_steer_grad`` against ``grad(g_steer)``; a byte-exact
  check pins ``g_steer_grad == grad_desired − grad_contrast``; and a positive↔
  negative swap is shown to negate both the value and the gradient.

The FD oracle reuses the Task-3 non-zero-init float32 encoder fixture
(``_build_randomized_model``) — the value-residual path must be live or the gen-slot
gradient is vacuous. ``COMPUTE_DTYPE`` is patched to float32 and SDPA is forced
(FlexAttention has no CPU backward), exactly as in the gradient oracle.
"""

from __future__ import annotations

import math

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGrad,
    FeatureGradMember,
    _assemble_encoder_inputs,
)
from steering.conditional.feature_gcg.objective import (
    ClassMarginItem,
    ClassMarginMember,
    FeatureLmItem,
    FeatureObjective,
    class_margin_objective,
    lm_loss_on_masked_tokens,
)
from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_base.types import SteeringMode
from tests.steering.feature_gcg._reference import feature_conditioning_gradient
from tests.steering.feature_gcg.functional.test_feature_gradient import (
    ATOL,
    EPS,
    RESOLVED_RTOL,
    RTOL,
    _build_randomized_model,
)

_CPU = torch.device("cpu")
# A bag whose feature_ids (17/23/29) are globally distinct from the context
# segment's (5/11) — so a perturbed feature_embed row maps to exactly one gen-slot
# activation slot (the FD precondition), mirroring the Task-3 oracle.
_BAG = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
_CONTEXT = FeatureConditioning(entries=((5, 1), (11, 2)))


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch):
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Toy completion / member builders + independent (test-owned) references
# ─────────────────────────────────────────────────────────────────────────────


def _completion(gen, *, t_len: int = 12, n_scored: int = 3, gen_slot: int = 1):
    """One teacher-forced completion (a placeholder-bag ``FeatureGradMember``).

    The optimized (generation) segment is ``gen_slot``; ``seg_map`` routes the
    second half of the sequence — where the ``n_scored`` tail target tokens live —
    to it, so the scored tokens cross-attend the bag and the gen-slot gradient is
    non-vacuous. Every scored target is real; the rest are ``-1`` (CE ignored).
    """
    dec_input = torch.randint(1, 100, (1, t_len), generator=gen, device=_CPU)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=_CPU)
    for pos in range(t_len - n_scored, t_len):
        dec_target[0, pos] = int(torch.randint(0, 100, (1,), generator=gen).item())
    half = t_len // 2
    seg_map = torch.tensor(
        [[0] * half + [1] * (t_len - half)], dtype=torch.long, device=_CPU
    )
    placeholder = FeatureConditioning(entries=())
    conditions = [_CONTEXT, placeholder] if gen_slot == 1 else [placeholder, _CONTEXT]
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=conditions,
        seg_map=seg_map,
        generation_slot=gen_slot,
    )


def _make_member(seed: int, *, n_scored: int = 3) -> ClassMarginMember:
    """A (rich, avoid) pair with distinct decoder content ⇒ a non-zero margin."""
    gen = torch.Generator().manual_seed(seed)
    rich = _completion(gen, n_scored=n_scored)
    avoid = _completion(gen, n_scored=n_scored)  # advances ``gen`` → different tokens
    return ClassMarginMember(rich=rich, avoid=avoid)


def _member_with_margin(
    model, *, min_abs_margin: float = 0.008, n_scored: int = 1, max_seeds: int = 200
) -> ClassMarginMember:
    """Deterministically pick the first member whose ``_BAG`` margin is non-vacuous.

    The randomized toy's bag conditioning is a small perturbation on the baseline
    CE, so a fixed seed can land on a near-zero margin (which would hide the sign).
    Searching a fixed seed range keeps the fixture reproducible while guaranteeing a
    margin comfortably above the ``approx`` tolerance the sign pin uses. The margin
    magnitude ``|avoid − rich|`` is mode-independent, so one search serves both modes.
    """
    for seed in range(max_seeds):
        member = _make_member(seed, n_scored=n_scored)
        loss_rich = _measure_loss(model, member.rich, _BAG)
        loss_avoid = _measure_loss(model, member.avoid, _BAG)
        if abs(loss_avoid - loss_rich) >= min_abs_margin:
            return member
    raise AssertionError(
        f"no toy member reached |margin| >= {min_abs_margin} within {max_seeds} seeds"
    )


def _scored_scalar(dec_target: torch.Tensor):
    """Independent mean-CE-over-scored-tokens reduction (mirrors the objective)."""
    sel = dec_target.reshape(-1) != -1
    n = int(sel.sum().item())

    def _fn(row: torch.Tensor) -> torch.Tensor:
        return row.reshape(-1)[sel].sum() / n

    return _fn


def _measure_loss(model, completion: FeatureGradMember, bag: FeatureConditioning) -> float:
    """Test-owned masked-mean CE for one completion + bag (no_grad is fine here)."""
    conditions = list(completion.segment_conditions)
    conditions[completion.generation_slot] = bag
    ids, mags, mask = _assemble_encoder_inputs(conditions, _CPU)
    with torch.no_grad():
        out = model(
            completion.dec_input,
            completion.dec_target,
            enc_feature_ids=ids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=completion.seg_map,
            loss_reduction="none",
        )
    return float(_scored_scalar(completion.dec_target)(out).item())


def _direct_grad(model, completion: FeatureGradMember, bag: FeatureConditioning) -> FeatureGrad:
    """Test-owned single-completion gradient (the proven Task-3 entry point)."""
    conditions = list(completion.segment_conditions)
    conditions[completion.generation_slot] = bag
    return feature_conditioning_gradient(
        model,
        dec_input=completion.dec_input,
        dec_target=completion.dec_target,
        segment_conditions=conditions,
        seg_map=completion.seg_map,
        generation_slot=completion.generation_slot,
        objective_scalar_fn=_scored_scalar(completion.dec_target),
        device=_CPU,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1a — lm_loss_on_masked_tokens
# ─────────────────────────────────────────────────────────────────────────────


def test_lm_loss_matches_hand_computed_masked_ce(monkeypatch):
    """``losses`` == Σ(CE over the masked tokens) / n_masked (independent ref)."""
    model = _build_randomized_model(monkeypatch, seed=11)
    gen = torch.Generator().manual_seed(5)
    member = _completion(gen, n_scored=4)
    obj = lm_loss_on_masked_tokens(model, device=_CPU)
    assert isinstance(obj, FeatureObjective)

    got = obj.losses([FeatureLmItem(member=member, bag=_BAG)])[0]
    ref = _measure_loss(model, member, _BAG)  # independent masked mean
    assert got == pytest.approx(ref, rel=1e-6, abs=1e-7)
    assert math.isfinite(got) and got > 0.0


def test_lm_loss_is_forward_only_no_param_grad(monkeypatch):
    """A fresh model retains NO parameter ``.grad`` after ``losses`` (no backward)."""
    model = _build_randomized_model(monkeypatch, seed=12)
    model.zero_grad(set_to_none=True)
    gen = torch.Generator().manual_seed(6)
    member = _completion(gen)
    obj = lm_loss_on_masked_tokens(model, device=_CPU)
    obj.losses([FeatureLmItem(member=member, bag=_BAG)])
    assert all(p.grad is None for p in model.parameters())


def test_lm_loss_gradients_returns_finite_featuregrad(monkeypatch):
    """``gradients`` returns finite, correctly-shaped ``FeatureGrad`` per item."""
    model = _build_randomized_model(monkeypatch, seed=13)
    gen = torch.Generator().manual_seed(7)
    member = _completion(gen)
    obj = lm_loss_on_masked_tokens(model, device=_CPU)
    items = [FeatureLmItem(member=member, bag=_BAG)]
    grads = obj.gradients(items, 0)
    assert len(grads) == 1
    grad = grads[0]
    assert isinstance(grad, FeatureGrad)
    n_real = len(_BAG)
    d_embd = model.encoder.config.n_embd
    assert grad.id_grad.shape[0] == n_real
    assert grad.bucket_grad.shape == (n_real, d_embd)
    assert torch.isfinite(grad.id_grad).all()
    assert torch.isfinite(grad.bucket_grad).all()


def test_lm_loss_gradients_differing_bags_raises(monkeypatch):
    """E3: differing-bag digests in the gradient batch raises FeatureGcgError.

    ``_LmLossObjective.gradients`` requires all items to carry the SAME target bag
    (run_gcg's invariant: one shared bag per gradient call).  Passing two items with
    different bags triggers the digest guard.

    Teeth: removing the guard lets the call proceed using only the first item's bag
    for ALL members → silently wrong gradients, no FeatureGcgError → pytest.raises
    exits without exception → test REDs.
    """
    model = _build_randomized_model(monkeypatch, seed=30)
    g0 = torch.Generator().manual_seed(90)
    g1 = torch.Generator().manual_seed(91)
    m0, m1 = _completion(g0), _completion(g1)

    bag_a = FeatureConditioning(entries=((17, 3), (23, 4), (29, 5)))
    bag_b = FeatureConditioning(entries=((6, 2), (14, 3), (31, 5)))  # distinct bag
    assert bag_a.digest() != bag_b.digest(), "fixture bags must have different digests"

    obj = lm_loss_on_masked_tokens(model, device=_CPU)
    with pytest.raises(FeatureGcgError, match="differing bags"):
        obj.gradients(
            [FeatureLmItem(member=m0, bag=bag_a), FeatureLmItem(member=m1, bag=bag_b)],
            0,
        )


def test_lm_loss_gradients_group_shared_bag_preserves_order(monkeypatch):
    """Two items sharing one bag are scored in ONE batched call, order preserved."""
    model = _build_randomized_model(monkeypatch, seed=14)
    g0 = torch.Generator().manual_seed(80)
    g1 = torch.Generator().manual_seed(81)
    m0, m1 = _completion(g0), _completion(g1)
    # Both items share ``_BAG`` (the run_gcg gradient-path invariant) but the masks
    # are equal length (same t_len) so the shared fixed mask applies to both.
    obj = lm_loss_on_masked_tokens(model, device=_CPU)
    grads = obj.gradients(
        [FeatureLmItem(member=m0, bag=_BAG), FeatureLmItem(member=m1, bag=_BAG)], 0
    )
    assert len(grads) == 2
    # member 0's grouped grad equals its standalone single call (byte-identical:
    # B=1 reduces to the single path, per Task-3).
    solo0 = _direct_grad(model, m0, _BAG)
    # the lm mask == dec_target!=-1 here, so _direct_grad's scalar matches exactly
    assert torch.equal(grads[0].id_grad, solo0.id_grad)
    assert torch.equal(grads[0].bucket_grad, solo0.bucket_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — class_margin_objective: the g_steer sign trap (spec §7.2)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
def test_class_margin_g_steer_equals_neg_primary_margin(monkeypatch, mode):
    """THE sign pin: ``g_steer == −primary_margin_y`` in BOTH modes.

    ``primary_margin_y`` is the INDEPENDENT canonical ``signed_class_margin`` over
    test-measured rich/avoid losses. A sign flip (``loss_contrast − loss_desired``)
    or a double swap both equal ``+primary_margin_y`` → fail this (margin ≠ 0).
    """
    model = _build_randomized_model(monkeypatch, seed=21)
    member = _member_with_margin(model)
    item = ClassMarginItem(member=member, bag=_BAG)
    obj = class_margin_objective(model, mode=mode, device=_CPU)

    g_steer = obj.losses([item])[0]
    loss_rich = _measure_loss(model, member.rich, _BAG)
    loss_avoid = _measure_loss(model, member.avoid, _BAG)
    primary_margin_y = signed_class_margin(mode=mode, rich=loss_rich, avoid=loss_avoid)

    assert abs(primary_margin_y) > 1e-3, "vacuous fixture: margin ≈ 0 hides the sign"
    assert g_steer == pytest.approx(-primary_margin_y, rel=1e-5, abs=1e-6)


def test_class_margin_minimizing_g_steer_maximizes_margin(monkeypatch):
    """The optimizer minimizes ``g_steer`` ⇒ it selects the MAX-margin bag.

    Over candidate bags: the objective's best (min ``g_steer``) is exactly the
    independent max-``primary_margin_y`` bag, and ``g_steer == −margin`` per
    candidate — so lowering ``g_steer`` raises the (independently scored) steering
    margin. Non-vacuous: the bag genuinely moves both the desired loss and margin.
    """
    model = _build_randomized_model(monkeypatch, seed=22)
    member = _member_with_margin(model)
    mode = SteeringMode.POSITIVE  # desired = rich
    obj = class_margin_objective(model, mode=mode, device=_CPU)

    # Diverse feature SETS (not just buckets) ⇒ genuinely different conditioning
    # directions ⇒ a real spread of margins.
    candidates = [
        FeatureConditioning(entries=e)
        for e in (
            ((17, 3), (23, 4), (29, 5)),
            ((17, 1), (23, 1), (29, 1)),
            ((6, 2), (14, 3), (31, 5)),
            ((8, 5), (19, 2), (35, 4)),
            ((3, 1), (27, 5), (38, 2)),
            ((2, 4), (12, 2), (33, 5)),
        )
    ]
    g_steers = [obj.losses([ClassMarginItem(member=member, bag=b)])[0] for b in candidates]
    margins = [
        signed_class_margin(
            mode=mode,
            rich=_measure_loss(model, member.rich, b),
            avoid=_measure_loss(model, member.avoid, b),
        )
        for b in candidates
    ]
    desired_losses = [_measure_loss(model, member.rich, b) for b in candidates]

    # Non-vacuity: the candidate bags genuinely span a range of margins / desired
    # losses (otherwise the ranking coincidence below would be meaningless).
    assert max(margins) - min(margins) > 1e-3
    assert max(desired_losses) - min(desired_losses) > 1e-4

    best_by_g = min(range(len(candidates)), key=lambda i: g_steers[i])
    best_by_margin = max(range(len(candidates)), key=lambda i: margins[i])
    assert best_by_g == best_by_margin  # min g_steer == max steering margin

    for g, m in zip(g_steers, margins, strict=True):
        assert g == pytest.approx(-m, rel=1e-5, abs=1e-6)


def test_class_margin_gradient_equals_desired_minus_contrast(monkeypatch):
    """``g_steer_grad`` is BYTE-EXACTLY ``grad_desired − grad_contrast`` (positive).

    The strongest sign teeth on the gradient: ``contrast − desired`` (flip),
    ``desired`` alone, or a double swap all break ``torch.equal`` here.
    """
    model = _build_randomized_model(monkeypatch, seed=23)
    member = _make_member(seed=33)
    item = ClassMarginItem(member=member, bag=_BAG)
    obj = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU)

    got = obj.gradients([item], 0)[0]
    grad_desired = _direct_grad(model, member.rich, _BAG)  # positive ⇒ desired=rich
    grad_contrast = _direct_grad(model, member.avoid, _BAG)  # contrast=avoid

    assert torch.equal(got.id_grad, grad_desired.id_grad - grad_contrast.id_grad)
    assert torch.equal(got.bucket_grad, grad_desired.bucket_grad - grad_contrast.bucket_grad)
    # id_weight / bucket_weight are the SHARED read-only matrices (desired's).
    assert torch.equal(got.id_weight, grad_desired.id_weight)
    assert torch.equal(got.bucket_weight, grad_desired.bucket_weight)
    # The grad is non-trivial (rich ≠ avoid ⇒ the difference is not all-zero).
    assert got.id_grad.abs().sum().item() > 0.0


def test_class_margin_mode_swap_negates_value_and_gradient(monkeypatch):
    """positive↔negative swaps desired/contrast ⇒ negates g_steer AND its grad."""
    model = _build_randomized_model(monkeypatch, seed=24)
    member = _member_with_margin(model)
    item = ClassMarginItem(member=member, bag=_BAG)
    pos = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU)
    neg = class_margin_objective(model, mode=SteeringMode.NEGATIVE, device=_CPU)

    g_pos = pos.losses([item])[0]
    g_neg = neg.losses([item])[0]
    assert abs(g_pos) > 1e-3
    assert g_neg == pytest.approx(-g_pos, rel=1e-5, abs=1e-6)

    grad_pos = pos.gradients([item], 0)[0]
    grad_neg = neg.gradients([item], 0)[0]
    # avoid − rich == −(rich − avoid) exactly (IEEE negation is symmetric).
    assert torch.equal(grad_neg.id_grad, -grad_pos.id_grad)
    assert torch.equal(grad_neg.bucket_grad, -grad_pos.bucket_grad)


def test_class_margin_sign_flip_and_double_swap_are_rejected(monkeypatch):
    """Permanence teeth: the identity REJECTS both adversary signs.

    Documents — inside the suite, so a regression cannot pass — that the correct
    ``g_steer == −primary_margin_y`` is distinguishable from the sign-flip
    (``loss_contrast − loss_desired``) and the double-swap, which both equal
    ``+primary_margin_y``. (The matching production sign flip is teeth-checked to
    RED ``test_class_margin_g_steer_equals_neg_primary_margin`` during review.)
    """
    model = _build_randomized_model(monkeypatch, seed=25)
    member = _member_with_margin(model)
    item = ClassMarginItem(member=member, bag=_BAG)
    mode = SteeringMode.POSITIVE
    obj = class_margin_objective(model, mode=mode, device=_CPU)

    g_steer = obj.losses([item])[0]
    loss_rich = _measure_loss(model, member.rich, _BAG)
    loss_avoid = _measure_loss(model, member.avoid, _BAG)
    primary_margin_y = signed_class_margin(mode=mode, rich=loss_rich, avoid=loss_avoid)
    assert abs(primary_margin_y) > 1e-3

    # Correct sign passes.
    assert g_steer == pytest.approx(-primary_margin_y, rel=1e-5, abs=1e-6)
    # Sign flip: compute loss_contrast − loss_desired instead of loss_desired − loss_contrast.
    # Gives +primary_margin_y, which fails the identity.
    sign_flipped = loss_avoid - loss_rich
    assert not math.isclose(sign_flipped, -primary_margin_y, rel_tol=1e-5, abs_tol=1e-6)
    # Double swap: flip sign AND swap desired↔contrast (use loss_avoid as desired, loss_rich
    # as contrast).  Concretely: −signed_class_margin(mode, rich=loss_avoid, avoid=loss_rich)
    # = −(margin_sign * (loss_rich − loss_avoid)) = loss_avoid − loss_rich = +primary_margin_y.
    # Distinct from sign_flipped in construction; also fails the identity.
    double_swapped = -signed_class_margin(mode=mode, rich=loss_avoid, avoid=loss_rich)
    assert not math.isclose(double_swapped, -primary_margin_y, rel_tol=1e-5, abs_tol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — the finite-difference oracle on g_steer_grad
# ─────────────────────────────────────────────────────────────────────────────


def _g_steer_at(obj: FeatureObjective, item: ClassMarginItem) -> float:
    return obj.losses([item])[0]


def _central_diff_g_steer(model, weight, row, col, obj, item) -> float:
    """Central finite difference of ``g_steer`` w.r.t. ``weight[row, col]``.

    Divides by the actual float32-rounded perturbation span (removes endpoint bias),
    mirroring the Task-3 oracle.
    """
    orig = float(weight[row, col].item())
    with torch.no_grad():
        weight[row, col] = orig + EPS
    ap = float(weight[row, col].item())
    lp = _g_steer_at(obj, item)
    with torch.no_grad():
        weight[row, col] = orig - EPS
    am = float(weight[row, col].item())
    lm = _g_steer_at(obj, item)
    with torch.no_grad():
        weight[row, col] = orig
    return (lp - lm) / (ap - am)


def test_class_margin_g_steer_grad_matches_finite_difference(monkeypatch):
    """FD oracle: ``g_steer_grad`` (feat axis = dL/dx) == central-diff ``g_steer``.

    Perturbs the gen-slot ``feature_embed`` rows (shared by both completions) and
    central-differences ``g_steer`` (the objective's own ``losses``); compares to
    ``g_steer_grad.id_grad[:, :D]`` per the SAME float32-loss tolerances the Task-3
    oracle uses. A sign flip would invert every FD (caught by the per-component
    ``isclose`` AND the best-resolved relative teeth).
    """
    model = _build_randomized_model(monkeypatch, seed=0)
    member = _make_member(seed=7, n_scored=1)  # single scored token ⇒ Task-3-tight FD
    item = ClassMarginItem(member=member, bag=_BAG)
    obj = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU)

    grad = obj.gradients([item], 0)[0]
    enc = model.encoder
    d_embd = enc.config.n_embd
    feat_w = enc.feature_embed.weight
    entries = list(_BAG.entries)

    worst_abs = 0.0
    best_resolved_rel = 0.0
    best_mag = -1.0
    for slot, (fid, _bucket) in enumerate(entries):
        vec = grad.id_grad[slot, :d_embd]  # dL/dx on the feat axis
        col = int(vec.abs().argmax())
        analytic = float(vec[col].item())
        fd = _central_diff_g_steer(model, feat_w, fid, col, obj, item)
        abs_err = abs(fd - analytic)
        worst_abs = max(worst_abs, abs_err)
        assert math.isclose(fd, analytic, rel_tol=RTOL, abs_tol=ATOL), (
            f"slot={slot} fid={fid} col={col}: FD={fd:.6e} analytic={analytic:.6e} "
            f"abs_err={abs_err:.2e}"
        )
        if abs(analytic) > best_mag:
            best_mag = abs(analytic)
            best_resolved_rel = abs_err / abs(analytic)

    assert best_resolved_rel <= RESOLVED_RTOL, (
        f"best-resolved relative error {best_resolved_rel:.2e} exceeds {RESOLVED_RTOL:.0e}"
    )
    assert worst_abs <= ATOL, f"worst FD absolute error {worst_abs:.2e} exceeds {ATOL:.0e}"
    print(
        f"\n[g_steer FD oracle] best-resolved relative error={best_resolved_rel:.2e}; "
        f"worst absolute error={worst_abs:.2e}"
    )
