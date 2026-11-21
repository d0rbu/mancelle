"""CPU exactness tests for the batched objective (the SOLE production codepath).

These tests prove — on the deterministic, non-zero-init **float32 CPU** tiny model
(``_build_randomized_model``) — that the **batched production** objective is
MATHEMATICALLY EQUIVALENT to the **per-item REFERENCE**
(``tests.steering.feature_gcg._reference``, the verbatim relocation of the old per-item
production helpers), bit-for-bit at the structural level and to within float32
BLAS-blocking noise at the scalar level.  This is the "exactness" half of the
correctness gate; the bf16/GPU *deployment tolerance* is derived separately in
``tests/steering/feature_gcg/integration/test_perf_batching.py``.

The batched path is now the only shipping codepath: ``lm_loss_on_masked_tokens`` /
``class_margin_objective`` always batch.  The per-item path remains ONLY in
``tests.steering.feature_gcg._reference`` (``reference_masked_loss`` / ``reference_g_steer``
/ ``reference_g_steer_grad``) as this oracle.  A teeth-test below pins that the old
scaffold ``batched`` flag is gone (the factories reject it).

Why CPU/float32 is the exactness oracle
---------------------------------------
The batched path replaces ``N`` separate forwards with ONE padded ``(B=N, t_max)``
forward.  With right-padding (pad ``target = -1`` / ``seg_map = 0``), causal
decoder attention, and per-row encoder conditioning, member ``b``'s real-position
logits depend ONLY on member ``b``'s own real inputs — never on the padding nor on
sibling rows (padding lives at LATER indices ``[t_len_b, t_max)`` which causal
attention never reaches backward into).  So in EXACT arithmetic
``batched[b] == reference[b]`` bit-for-bit.

In float32, the B=1 path IS bit-identical to the per-item forward (same kernel
shape), but a B>1 batch reassociates the float32 matmul accumulations (the leading
batch·T dimension changes the kernel tiling), so ``batched[b]`` drifts from the
per-item ``reference[b]`` by a float32-ULP-scale rounding (≤ ~1.5e-5 on the toy).
We therefore assert the SCALAR equivalence to a tight near-exact
``_CPU_REASSOC_ATOL`` (the same reassociation that, in GPU bf16, grows to the ~5e-2
deployment atol), and pin the EXACT (bit-for-bit) structural guarantee — that no
sibling row or pad leaks into a member's real positions — confound-free via
fixed-batch-shape SIBLING-INVARIANCE ``torch.equal`` tests.

The class-margin **sign trap** (spec §7.2) is re-pinned in the batched path: the
batched ``losses`` must still return ``g_steer = loss_desired − loss_contrast``
(ONE explicit negation) and the batched ``gradients`` must still return
``grad_desired − grad_contrast`` — a flip or double-swap REDs these tests.
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
    FeatureGradMember,
    _assemble_decoder_batch,
    _assemble_encoder_batch,
    _batched_masked_losses,
    _ForwardSpec,
    feature_conditioning_gradient_batched,
    substitute_bag,
)
from steering.conditional.feature_gcg.objective import (
    ClassMarginItem,
    ClassMarginMember,
    FeatureLmItem,
    _masked_mean_scalar,
    class_margin_objective,
    lm_loss_on_masked_tokens,
)
from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_base.types import SteeringMode
from tests.steering.feature_gcg._equivalence import feature_grads_equal
from tests.steering.feature_gcg._reference import (
    feature_conditioning_gradient,
    reference_g_steer,
    reference_g_steer_grad,
    reference_masked_loss,
)
from tests.steering.feature_gcg.functional.test_feature_gradient import _build_randomized_model
from tests.steering.feature_gcg.functional.test_feature_objective import (
    _BAG,
    _completion,
    _direct_grad,
    _make_member,
    _measure_loss,
    _member_with_margin,
)

_CPU = torch.device("cpu")

# CPU/float32 near-exact tolerance.  The batched (B>1) forward is NOT bit-identical
# to the per-item (B=1) reference even on CPU/float32: changing the leading batch·T
# dimension makes the matmul kernels reassociate their float32 accumulations, so a
# member's loss drifts by a float32-ULP-scale rounding.  That drift is dominated by
# the RELATIVE term (it scales with the loss magnitude / logit spread): empirically
# ~3e-5 relative on a CE of ~4.6 (abs ~1.4e-4), and ~1.4e-5 absolute on a small
# g_steer DIFFERENCE.  We therefore bound it with BOTH a relative tol (1e-3, ~30x
# the observed relative drift) and a small absolute floor (1e-4, for the near-zero
# g_steer values) — pytest.approx passes within EITHER.  The math is identical: a
# real batching bug (a sibling/pad leaking into a member's real positions, a wrong
# mask, or a sign flip) moves a value by O(1%+) relative / O(0.1-1) absolute, well
# ABOVE both floors, so the teeth stay sharp.  The B=1 path IS bit-identical (see
# the byte-identity test), and the EXACT structural guarantee is pinned by the
# sibling-invariance tests.  (The GPU bf16 deployment atol is ~5e-2, derived in
# tests/steering/feature_gcg/integration/test_perf_batching.py.)
_CPU_REASSOC_RTOL = 1e-3
_CPU_REASSOC_ATOL = 1e-4
# Per-token loss ROWS (O(1-10) values) reassociate a touch more than the reduced
# scalar; 1e-3 absolute bounds it with margin yet stays ~100-1000x below a
# contamination bug (which corrupts a real-position logit by O(0.1-1)).
_CPU_ROW_ATOL = 1e-3
# Gradient equivalence atol.  The class-margin batched gradient is one fwd+bwd over
# 2N specs vs the reference's 2N single backwards; on CPU/float32 (deterministic
# index_add) the only delta is the same matmul reassociation, ~1e-4 on O(0.01-1)
# grads.  1e-3 bounds it; the EXACT desired−contrast sign is pinned separately by
# torch.equal / allclose against the per-item references.
_CPU_GRAD_ATOL = 1e-3


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Local builders
# ─────────────────────────────────────────────────────────────────────────────


def _spec_and_fn(
    member: FeatureGradMember, bag: FeatureConditioning
) -> tuple[_ForwardSpec, object]:
    """A fully-resolved ``_ForwardSpec`` (bag substituted) + its masked-mean fn."""
    spec = _ForwardSpec(
        dec_input=member.dec_input,
        dec_target=member.dec_target,
        segment_conditions=substitute_bag(member, bag),
        seg_map=member.seg_map,
        generation_slot=member.generation_slot,
    )
    fn = _masked_mean_scalar(member.dec_target != -1)
    return spec, fn


def _per_item_loss(model: object, member: FeatureGradMember, bag: FeatureConditioning) -> float:
    """The per-item REFERENCE forward (``reference_masked_loss``) for one (member, bag)."""
    return reference_masked_loss(
        model,
        dec_input=member.dec_input,
        dec_target=member.dec_target,
        segment_conditions=substitute_bag(member, bag),
        seg_map=member.seg_map,
        scalar_fn=_masked_mean_scalar(member.dec_target != -1),
        device=_CPU,
    )


# ─────────────────────────────────────────────────────────────────────────────
# _batched_masked_losses — the shared helper (gradient.py)
# ─────────────────────────────────────────────────────────────────────────────


def test_batched_masked_losses_single_item_is_byte_identical_to_reference(monkeypatch):
    """B=1 batched loss == the per-item ``reference_masked_loss`` exactly (no padding ⇒ same forward)."""
    model = _build_randomized_model(monkeypatch, seed=11)
    gen = torch.Generator().manual_seed(5)
    member = _completion(gen, t_len=12, n_scored=4)
    spec, fn = _spec_and_fn(member, _BAG)

    got = _batched_masked_losses(model, [spec], [fn], _CPU)  # ty: ignore[invalid-argument-type]
    ref = _per_item_loss(model, member, _BAG)

    assert len(got) == 1
    # B=1 assembles the identical (1, T) decoder + (1, n_seg, F) encoder tensors the
    # single-item reference builds, so the forward — and thus the scalar — is identical.
    assert got[0] == ref


def _batched_rows(model: object, specs: list[_ForwardSpec]) -> torch.Tensor:
    """Reproduce ``_batched_masked_losses``'s forward and return ``rows`` (B, t_max)."""
    dec_input, dec_target, seg_map, t_max = _assemble_decoder_batch(specs, _CPU)
    ids, mags, mask, _ = _assemble_encoder_batch(specs, _CPU)
    batch = dec_input.shape[0]
    with torch.no_grad():
        out = model(
            dec_input,
            dec_target,
            enc_feature_ids=ids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
            loss_reduction="none",
        )  # ty: ignore[call-non-callable]
    return out.view(batch, t_max)


def test_batched_masked_losses_multi_differing_length_matches_reference(monkeypatch):
    """N members of DIFFERING decoder length: each batched loss == its per-item reference.

    Also pins the INTERMEDIATE claim that the padded batch's per-row REAL positions
    (``rows[b, :t_len_b]``) reproduce the unpadded single forward's loss row.  This
    holds bit-for-bit in EXACT arithmetic (right-padding + causal + per-row
    encoder); on CPU/float32 the two forwards reassociate at the float32-ULP scale
    because the batch·T matmul shape differs (the same effect the scalar atol
    absorbs), so the row check uses a tight ``_CPU_ROW_ATOL`` that a real
    contamination bug (O(0.1-1) per logit) would still blow past.  The EXACT
    (bit-for-bit) form of the structural guarantee is pinned, confound-free, by the
    sibling-invariance tests below.
    """
    model = _build_randomized_model(monkeypatch, seed=12)
    t_lens = [8, 14, 11]
    members = [
        _completion(torch.Generator().manual_seed(100 + i), t_len=t, n_scored=3)
        for i, t in enumerate(t_lens)
    ]
    specs, fns = zip(*[_spec_and_fn(m, _BAG) for m in members], strict=True)
    specs, fns = list(specs), list(fns)

    got = _batched_masked_losses(model, specs, fns, _CPU)
    refs = [_per_item_loss(model, m, _BAG) for m in members]

    assert len(got) == len(members)
    for b, (g, r) in enumerate(zip(got, refs, strict=True)):
        assert g == pytest.approx(r, rel=_CPU_REASSOC_RTOL, abs=_CPU_REASSOC_ATOL), (
            f"member {b} (t_len={t_lens[b]}): batched={g!r} reference={r!r}"
        )

    # Intermediate claim: the padded batch's real-position rows reproduce the single
    # forwards (to within float32 reassociation — see the docstring).
    rows = _batched_rows(model, specs)
    for b, member in enumerate(members):
        t_len = member.dec_input.shape[1]
        single_spec, _fn = _spec_and_fn(member, _BAG)
        single_row = _batched_rows(model, [single_spec])[0]
        diff = (rows[b, :t_len] - single_row).abs().max().item()
        assert diff <= _CPU_ROW_ATOL, (
            f"member {b}: padded-batch real positions diverge from the single "
            f"forward by {diff:.2e} (> {_CPU_ROW_ATOL}); a contamination bug?"
        )


def test_batched_rows_are_invariant_to_a_siblings_bag(monkeypatch):
    """EXACT per-row-encoder guarantee: a member's row is BIT-IDENTICAL regardless of
    a same-shaped sibling's bag.

    Holding the batch SHAPE fixed (same B, same t_max) removes the float32
    reassociation confound, so this is a true ``torch.equal`` proof that per-row
    encoder conditioning does not let one row's bag leak into another row's real
    positions.
    """
    model = _build_randomized_model(monkeypatch, seed=31)
    target = _completion(torch.Generator().manual_seed(1001), t_len=12, n_scored=3)
    sibling = _completion(torch.Generator().manual_seed(1002), t_len=12, n_scored=3)

    target_spec, _ = _spec_and_fn(target, _BAG)
    bag_a = FeatureConditioning(entries=((17, 3), (23, 4)))
    bag_b = FeatureConditioning(entries=((6, 1), (14, 2), (31, 5)))  # different bag, same length-1 row
    sib_a, _ = _spec_and_fn(sibling, bag_a)
    sib_b, _ = _spec_and_fn(sibling, bag_b)

    rows_a = _batched_rows(model, [target_spec, sib_a])
    rows_b = _batched_rows(model, [target_spec, sib_b])
    assert torch.equal(rows_a[0], rows_b[0]), (
        "target row changed when only the SIBLING's bag changed (per-row encoder leak)"
    )
    # And the sibling row genuinely DID move (otherwise the test is vacuous).
    assert not torch.equal(rows_a[1], rows_b[1])


def test_batched_rows_are_invariant_to_a_siblings_tokens(monkeypatch):
    """EXACT causal/right-pad guarantee: a member's row is BIT-IDENTICAL regardless of
    a same-shaped sibling's decoder tokens.

    Same fixed-shape isolation as the bag test — proves no cross-row leakage through
    the decoder (causal attention + per-row seg_map), confound-free.
    """
    model = _build_randomized_model(monkeypatch, seed=32)
    target = _completion(torch.Generator().manual_seed(2001), t_len=12, n_scored=3)
    sib_a = _completion(torch.Generator().manual_seed(2002), t_len=12, n_scored=3)
    sib_b = _completion(torch.Generator().manual_seed(2003), t_len=12, n_scored=3)

    target_spec, _ = _spec_and_fn(target, _BAG)
    spec_a, _ = _spec_and_fn(sib_a, _BAG)
    spec_b, _ = _spec_and_fn(sib_b, _BAG)

    rows_a = _batched_rows(model, [target_spec, spec_a])
    rows_b = _batched_rows(model, [target_spec, spec_b])
    assert torch.equal(rows_a[0], rows_b[0]), (
        "target row changed when only the SIBLING's tokens changed (causal leak)"
    )
    assert not torch.equal(rows_a[1], rows_b[1])


def test_batched_masked_losses_differing_bag_sizes_match_reference(monkeypatch):
    """Items carrying DIFFERENT bags (sizes 3 / 1 / 0-empty) each match the reference.

    ``_objectives_of_candidates`` packs items spanning multiple candidate bags into
    ONE ``objective_batch`` call, so the batched losses MUST be correct under mixed
    per-row encoder conditioning — not just a single shared bag.
    """
    model = _build_randomized_model(monkeypatch, seed=13)
    members = [
        _completion(torch.Generator().manual_seed(200 + i), t_len=12, n_scored=3)
        for i in range(3)
    ]
    bags = [
        FeatureConditioning(entries=((17, 3), (23, 4), (29, 5))),  # size 3
        FeatureConditioning(entries=((8, 2),)),                     # size 1
        FeatureConditioning(entries=()),                            # empty bag
    ]
    specs, fns = zip(*[_spec_and_fn(m, b) for m, b in zip(members, bags, strict=True)], strict=True)

    got = _batched_masked_losses(model, list(specs), list(fns), _CPU)
    refs = [_per_item_loss(model, m, b) for m, b in zip(members, bags, strict=True)]

    for b, (g, r) in enumerate(zip(got, refs, strict=True)):
        assert g == pytest.approx(r, rel=_CPU_REASSOC_RTOL, abs=_CPU_REASSOC_ATOL), (
            f"item {b} (bag size {len(bags[b])}): batched={g!r} reference={r!r}"
        )


def test_batched_masked_losses_empty_bag_single_item(monkeypatch):
    """A lone item conditioned on the EMPTY bag matches its per-item reference exactly."""
    model = _build_randomized_model(monkeypatch, seed=14)
    member = _completion(torch.Generator().manual_seed(7), t_len=10, n_scored=2)
    empty = FeatureConditioning(entries=())
    spec, fn = _spec_and_fn(member, empty)

    got = _batched_masked_losses(model, [spec], [fn], _CPU)  # ty: ignore[invalid-argument-type]
    ref = _per_item_loss(model, member, empty)
    assert got[0] == ref


def test_batched_masked_losses_raises_on_empty_specs(monkeypatch):
    """An empty spec list is a loud FeatureGcgError (mirrors the gradient core)."""
    model = _build_randomized_model(monkeypatch, seed=15)
    with pytest.raises(FeatureGcgError, match="no members"):
        _batched_masked_losses(model, [], [], _CPU)


def test_batched_masked_losses_raises_on_fn_length_mismatch(monkeypatch):
    """A scalar-fn / spec count mismatch is a loud FeatureGcgError."""
    model = _build_randomized_model(monkeypatch, seed=16)
    member = _completion(torch.Generator().manual_seed(8), t_len=10, n_scored=2)
    spec, fn = _spec_and_fn(member, _BAG)
    with pytest.raises(FeatureGcgError, match="objective_scalar_fns length"):
        _batched_masked_losses(model, [spec, spec], [fn], _CPU)  # ty: ignore[invalid-argument-type]


# ─────────────────────────────────────────────────────────────────────────────
# _LmLossObjective — batched losses (the 84%-of-runtime hotspot, sole prod path)
# ─────────────────────────────────────────────────────────────────────────────


def test_lm_factory_rejects_batched_kwarg(monkeypatch):
    """Batched is the SOLE production path: the scaffold ``batched`` flag is GONE.

    Passing the removed kwarg is a loud ``TypeError`` (no silent per-item fallback
    survives in production). Teeth: re-introducing a ``batched=...`` parameter on the
    factory would make this ``pytest.raises`` exit without an exception → RED.
    """
    model = _build_randomized_model(monkeypatch, seed=17)
    with pytest.raises(TypeError):
        lm_loss_on_masked_tokens(model, device=_CPU, batched=True)  # ty: ignore[unknown-argument]


def test_lm_objective_losses_match_per_item_reference_differing_lengths(monkeypatch):
    """Production (batched) ``losses`` == the per-item reference for differing-length members."""
    model = _build_randomized_model(monkeypatch, seed=18)
    members = [
        _completion(torch.Generator().manual_seed(400 + i), t_len=t, n_scored=3)
        for i, t in enumerate([9, 13, 7, 12])
    ]
    items = [FeatureLmItem(member=m, bag=_BAG) for m in members]

    production = lm_loss_on_masked_tokens(model, device=_CPU).losses(items)
    reference = [_per_item_loss(model, m, _BAG) for m in members]

    assert len(production) == len(reference)
    for b, (g, r) in enumerate(zip(production, reference, strict=True)):
        assert g == pytest.approx(r, rel=_CPU_REASSOC_RTOL, abs=_CPU_REASSOC_ATOL), f"item {b}: {g!r} vs {r!r}"


def test_lm_objective_losses_match_per_item_reference_mixed_bags(monkeypatch):
    """Production ``losses`` match the per-item reference when items carry DIFFERENT bags."""
    model = _build_randomized_model(monkeypatch, seed=19)
    members = [_completion(torch.Generator().manual_seed(500 + i), t_len=11) for i in range(3)]
    bags = [
        FeatureConditioning(entries=((17, 3), (23, 4))),
        FeatureConditioning(entries=()),
        FeatureConditioning(entries=((6, 2), (14, 1), (31, 5))),
    ]
    items = [FeatureLmItem(member=m, bag=b) for m, b in zip(members, bags, strict=True)]

    production = lm_loss_on_masked_tokens(model, device=_CPU).losses(items)
    reference = [_per_item_loss(model, m, b) for m, b in zip(members, bags, strict=True)]
    for b, (g, r) in enumerate(zip(production, reference, strict=True)):
        assert g == pytest.approx(r, rel=_CPU_REASSOC_RTOL, abs=_CPU_REASSOC_ATOL), f"item {b}: {g!r} vs {r!r}"


def test_lm_objective_empty_items_returns_empty(monkeypatch):
    """``losses([])`` returns ``[]`` in the production path (matches the reference loop)."""
    model = _build_randomized_model(monkeypatch, seed=20)
    assert lm_loss_on_masked_tokens(model, device=_CPU).losses([]) == []


def test_lm_objective_is_forward_only_no_param_grad(monkeypatch):
    """Production ``losses`` populates no parameter ``.grad`` (issue #510 — no no_grad)."""
    model = _build_randomized_model(monkeypatch, seed=21)
    model.zero_grad(set_to_none=True)
    members = [_completion(torch.Generator().manual_seed(600 + i), t_len=10) for i in range(2)]
    items = [FeatureLmItem(member=m, bag=_BAG) for m in members]
    lm_loss_on_masked_tokens(model, device=_CPU).losses(items)
    assert all(p.grad is None for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# _ClassMarginObjective — batched losses (the §7.2 sign trap, preserved)
# ─────────────────────────────────────────────────────────────────────────────


def test_cm_factory_rejects_batched_kwarg(monkeypatch):
    """Batched is the SOLE production path: the class-margin ``batched`` flag is GONE.

    Teeth: re-adding a ``batched=...`` parameter would let this ``pytest.raises``
    exit without a ``TypeError`` → RED.
    """
    model = _build_randomized_model(monkeypatch, seed=22)
    with pytest.raises(TypeError):
        class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU, batched=True)  # ty: ignore[unknown-argument]


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
def test_cm_objective_losses_match_per_item_reference(monkeypatch, mode):
    """Production (batched) g_steer == the per-item ``reference_g_steer`` in BOTH modes.

    Members of differing length; the reference applies the SAME single
    ``loss_desired − loss_contrast`` negation, so a flip/double-swap in either path
    would break this equivalence.
    """
    model = _build_randomized_model(monkeypatch, seed=23)
    members = [
        ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(700 + i), t_len=t, n_scored=3),
            avoid=_completion(torch.Generator().manual_seed(800 + i), t_len=t + 2, n_scored=3),
        )
        for i, t in enumerate([8, 12, 10])
    ]
    items = [ClassMarginItem(member=m, bag=_BAG) for m in members]

    production = class_margin_objective(model, mode=mode, device=_CPU).losses(items)
    reference = [reference_g_steer(model, m, _BAG, mode=mode, device=_CPU) for m in members]
    assert len(production) == len(reference)
    for b, (g, r) in enumerate(zip(production, reference, strict=True)):
        assert g == pytest.approx(r, rel=_CPU_REASSOC_RTOL, abs=_CPU_REASSOC_ATOL), f"item {b}: {g!r} vs {r!r}"


def test_cm_objective_losses_preserve_negation_sign(monkeypatch):
    """THE sign pin (batched production): g_steer == −primary_margin_y (single negation, not flipped)."""
    model = _build_randomized_model(monkeypatch, seed=24)
    # The margin-search helper avoids a vacuous (≈0) margin hiding the sign.
    member = _member_with_margin(model)
    item = ClassMarginItem(member=member, bag=_BAG)
    mode = SteeringMode.POSITIVE

    g_steer = class_margin_objective(model, mode=mode, device=_CPU).losses([item])[0]
    loss_rich = _measure_loss(model, member.rich, _BAG)
    loss_avoid = _measure_loss(model, member.avoid, _BAG)
    primary_margin_y = signed_class_margin(mode=mode, rich=loss_rich, avoid=loss_avoid)

    assert abs(primary_margin_y) > 1e-3, "vacuous fixture: margin ≈ 0 hides the sign"
    # Correct single-negation sign passes; a flip / double-swap (== +primary_margin_y) REDs.
    assert g_steer == pytest.approx(-primary_margin_y, rel=1e-5, abs=1e-6)
    assert not math.isclose(g_steer, +primary_margin_y, rel_tol=1e-5, abs_tol=1e-6)


def test_cm_objective_losses_empty_items_returns_empty(monkeypatch):
    """``losses([])`` returns ``[]`` in the class-margin production path."""
    model = _build_randomized_model(monkeypatch, seed=25)
    obj = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU)
    assert obj.losses([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# _ClassMarginObjective — batched gradients (sign trap on the gradient)
# ─────────────────────────────────────────────────────────────────────────────


def test_cm_objective_gradients_match_per_item_reference_single(monkeypatch):
    """Production (batched) gradients == the per-item ``reference_g_steer_grad`` (single item).

    A single same-length item batches the (desired, contrast) pair into ONE B=2
    fwd+bwd; the reference subtracts two B=1 backwards.  On CPU/float32 these agree
    to a tight near-exact atol (BLAS-blocking only); the EXACT desired−contrast sign
    is pinned by ``test_cm_objective_gradient_equals_desired_minus_contrast``.
    """
    model = _build_randomized_model(monkeypatch, seed=26)
    member = _make_member(seed=41)
    item = ClassMarginItem(member=member, bag=_BAG)

    production = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU).gradients([item], 0)
    reference = reference_g_steer_grad(model, member, _BAG, mode=SteeringMode.POSITIVE, device=_CPU)
    feature_grads_equal(production[0], reference, exact=False, atol=_CPU_GRAD_ATOL)
    # Non-triviality floor: a vacuous member (rich≈avoid ⇒ grad_desired−grad_contrast
    # ≈ 0) would let the equivalence pass on 0≈0.  Pin the g_steer gradient non-zero.
    assert production[0].id_grad.abs().sum().item() > 0.0, "vacuous (zero) id_grad"
    assert production[0].bucket_grad.abs().sum().item() > 0.0, "vacuous (zero) bucket_grad"


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
def test_cm_objective_gradients_match_per_item_reference(monkeypatch, mode):
    """Production (batched) gradients == the per-item reference (CPU float32, near-exact)."""
    model = _build_randomized_model(monkeypatch, seed=27)
    members = [
        ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(900 + i), t_len=t, n_scored=2),
            avoid=_completion(torch.Generator().manual_seed(950 + i), t_len=t + 1, n_scored=2),
        )
        for i, t in enumerate([9, 11])
    ]
    items = [ClassMarginItem(member=m, bag=_BAG) for m in members]

    production = class_margin_objective(model, mode=mode, device=_CPU).gradients(items, 0)
    reference = [reference_g_steer_grad(model, m, _BAG, mode=mode, device=_CPU) for m in members]
    assert len(production) == len(reference) == len(members)
    for b, (gb, gp) in enumerate(zip(production, reference, strict=True)):
        # CPU float32 + deterministic index_add ⇒ batched and per-item agree to a
        # tight near-exact atol (BLAS-blocking only).
        feature_grads_equal(gb, gp, exact=False, atol=_CPU_GRAD_ATOL)
        # Non-triviality floor (esp. the NEGATIVE-mode parametrization): without it
        # a vacuous rich≈avoid member would pass the equivalence on 0≈0.
        assert gb.id_grad.abs().sum().item() > 0.0, f"item {b}: vacuous (zero) id_grad"
        assert gb.bucket_grad.abs().sum().item() > 0.0, f"item {b}: vacuous (zero) bucket_grad"


def test_cm_objective_gradient_equals_desired_minus_contrast(monkeypatch):
    """Production g_steer_grad is ``grad_desired − grad_contrast`` (positive mode).

    The strongest sign teeth on the BATCHED gradient: a flip (contrast − desired),
    desired-alone, or a double swap all shift the gradient by O(1) — far above the
    ``allclose`` tolerance. The grad fields use ``allclose`` (rtol=1e-3) rather than
    ``torch.equal`` because the batched backward reassociates its float32
    accumulation relative to the two B=1 ``_direct_grad`` references (the same
    precedent as ``test_feature_gradient.py::test_batched_matches_single``); the
    SHARED read-only weight matrices remain ``torch.equal``.
    """
    model = _build_randomized_model(monkeypatch, seed=28)
    member = _make_member(seed=33)
    item = ClassMarginItem(member=member, bag=_BAG)

    got = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU).gradients([item], 0)[0]
    grad_desired = _direct_grad(model, member.rich, _BAG)   # positive ⇒ desired = rich
    grad_contrast = _direct_grad(model, member.avoid, _BAG)  # contrast = avoid

    assert torch.allclose(
        got.id_grad, grad_desired.id_grad - grad_contrast.id_grad, rtol=1e-3, atol=1e-6
    )
    assert torch.allclose(
        got.bucket_grad,
        grad_desired.bucket_grad - grad_contrast.bucket_grad,
        rtol=1e-3,
        atol=1e-6,
    )
    # id_weight / bucket_weight are the SHARED read-only matrices (desired's).
    assert torch.equal(got.id_weight, grad_desired.id_weight)
    assert torch.equal(got.bucket_weight, grad_desired.bucket_weight)
    assert got.id_grad.abs().sum().item() > 0.0  # non-trivial (rich ≠ avoid)


def test_cm_objective_gradients_empty_items_returns_empty(monkeypatch):
    """``gradients([], step)`` returns ``[]`` in the class-margin production path."""
    model = _build_randomized_model(monkeypatch, seed=29)
    obj = class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU)
    assert obj.gradients([], 0) == []


# ─────────────────────────────────────────────────────────────────────────────
# T7 review coverage gaps (B1 segment-count padding, B3 CM forward-only, B4 LM
# gradients empty-guard)
# ─────────────────────────────────────────────────────────────────────────────


def _member_with_segments(
    seed: int, *, n_seg: int, gen_slot: int, t_len: int = 12, n_scored: int = 3
) -> FeatureGradMember:
    """A teacher-forced member with EXACTLY ``n_seg`` segments, gen slot ``gen_slot``.

    The scored (tail) tokens route to ``gen_slot`` (so the gen-slot gradient is
    non-vacuous); the non-scored head routes to a DISTINCT context slot when one
    exists.  The ``gen_slot`` bag is an empty placeholder (substituted by the
    shared target before scoring); every other slot carries a distinct, non-empty
    context bag (feature-ids < ``N_SAE=40``, buckets < ``NUM_BUCKETS=6``).
    """
    gen = torch.Generator().manual_seed(seed)
    dec_input = torch.randint(1, 100, (1, t_len), generator=gen, device=_CPU)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=_CPU)
    for pos in range(t_len - n_scored, t_len):
        dec_target[0, pos] = int(torch.randint(0, 100, (1,), generator=gen).item())
    seg_map = torch.zeros((1, t_len), dtype=torch.long, device=_CPU)
    if n_seg > 1:
        head_ctx = 1 if gen_slot == 0 else 0  # some non-gen slot
        seg_map[0, : t_len - n_scored] = head_ctx
    seg_map[0, t_len - n_scored :] = gen_slot
    segment_conditions = [
        FeatureConditioning(entries=())
        if s == gen_slot
        else FeatureConditioning(entries=((2 + s, 1), (6 + s, 2)))
        for s in range(n_seg)
    ]
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=gen_slot,
    )


def test_batched_differing_segment_counts_match_per_item_reference(monkeypatch):
    """B1: members of DIFFERING segment counts (1/2/3) — incl. a ``generation_slot=0``
    member — each match their per-item loss AND gradient reference.

    Every other batched-objective test uses exactly TWO segments + ``generation_slot
    =1``, so the cross-member segment-count padding branch in
    ``_assemble_encoder_batch`` (a member with fewer segments than a sibling → zero
    + ``mask=False`` padded segments) and the ``generation_slot=0`` read were NEVER
    exercised on the batched path.  Here the 3-segment member forces ``n_seg_max=3``,
    so the 1- and 2-segment members get padded segments; the per-item reference (one
    B=1 forward each, no padding) is the oracle.  A padded-segment leak or a wrong
    gen-slot read moves a member's loss/grad far above the float32 atol.
    """
    model = _build_randomized_model(monkeypatch, seed=71)
    target = FeatureConditioning(entries=((17, 3), (23, 4)))
    members = [
        _member_with_segments(seed=300, n_seg=1, gen_slot=0),
        _member_with_segments(seed=301, n_seg=2, gen_slot=0),
        _member_with_segments(seed=302, n_seg=3, gen_slot=2),
    ]
    # Sanity: the members genuinely span differing segment counts (the branch under
    # test) and include a generation_slot=0 read.
    assert sorted(len(m.segment_conditions) for m in members) == [1, 2, 3]
    assert any(m.generation_slot == 0 for m in members)

    scalar_fns = [_masked_mean_scalar(m.dec_target != -1) for m in members]
    specs = [
        _ForwardSpec(
            dec_input=m.dec_input,
            dec_target=m.dec_target,
            segment_conditions=substitute_bag(m, target),
            seg_map=m.seg_map,
            generation_slot=m.generation_slot,
        )
        for m in members
    ]

    # --- Loss: each member's batched loss == its per-item reference ---
    batched_losses = _batched_masked_losses(model, specs, scalar_fns, _CPU)
    ref_losses = [
        reference_masked_loss(
            model,
            dec_input=m.dec_input,
            dec_target=m.dec_target,
            segment_conditions=substitute_bag(m, target),
            seg_map=m.seg_map,
            scalar_fn=fn,
            device=_CPU,
        )
        for m, fn in zip(members, scalar_fns, strict=True)
    ]
    assert len(batched_losses) == len(ref_losses) == len(members)
    for b, (g, r) in enumerate(zip(batched_losses, ref_losses, strict=True)):
        assert g == pytest.approx(r, rel=_CPU_REASSOC_RTOL, abs=_CPU_REASSOC_ATOL), (
            f"member {b} (n_seg={len(members[b].segment_conditions)}, "
            f"gen_slot={members[b].generation_slot}): batched={g!r} reference={r!r}"
        )

    # --- Gradient: each member's batched grad == its per-item reference ---
    batched_grads = feature_conditioning_gradient_batched(
        model, members, target, objective_scalar_fns=scalar_fns, device=_CPU
    )
    ref_grads = [
        feature_conditioning_gradient(
            model,
            dec_input=m.dec_input,
            dec_target=m.dec_target,
            segment_conditions=substitute_bag(m, target),
            seg_map=m.seg_map,
            generation_slot=m.generation_slot,
            objective_scalar_fn=fn,
            device=_CPU,
        )
        for m, fn in zip(members, scalar_fns, strict=True)
    ]
    assert len(batched_grads) == len(ref_grads) == len(members)
    for b, (gb, gp) in enumerate(zip(batched_grads, ref_grads, strict=True)):
        feature_grads_equal(gb, gp, exact=False, atol=_CPU_GRAD_ATOL)
        # Non-vacuity: the padded-segment branch must still yield a real gen-slot grad.
        assert gb.id_grad.abs().sum().item() > 0.0, f"member {b}: vacuous (zero) id_grad"
        assert gb.bucket_grad.abs().sum().item() > 0.0, f"member {b}: vacuous (zero) bucket_grad"


def test_cm_objective_is_forward_only_no_param_grad(monkeypatch):
    """B3: class-margin ``losses`` populates no parameter ``.grad`` (issue #510).

    Mirrors ``test_lm_objective_is_forward_only_no_param_grad`` for the class-margin
    path, whose ``losses`` interleaves 2N (desired, contrast) specs via
    ``_desired_contrast_specs`` — so the extra interleave machinery must ALSO never
    backpropagate (no ``no_grad`` wrap per #510; it simply never calls backward).
    """
    model = _build_randomized_model(monkeypatch, seed=210)
    model.zero_grad(set_to_none=True)
    members = [
        ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(610 + i), t_len=10),
            avoid=_completion(torch.Generator().manual_seed(660 + i), t_len=12),
        )
        for i in range(2)
    ]
    items = [ClassMarginItem(member=m, bag=_BAG) for m in members]
    class_margin_objective(model, mode=SteeringMode.POSITIVE, device=_CPU).losses(items)
    assert all(p.grad is None for p in model.parameters())


def test_lm_objective_gradients_empty_items_returns_empty(monkeypatch):
    """B4: ``gradients([], step)`` returns ``[]`` in the LM production path.

    Pins the untested ``if not items: return []`` guard in
    ``_LmLossObjective.gradients`` (the CM ``gradients([])`` and both ``losses([])``
    guards were already tested).  Teeth: removing the guard makes ``items[0].bag``
    IndexError on the empty list → this no longer returns ``[]``.
    """
    model = _build_randomized_model(monkeypatch, seed=31)
    assert lm_loss_on_masked_tokens(model, device=_CPU).gradients([], 0) == []


# ─────────────────────────────────────────────────────────────────────────────
# #1245: gradient-path sub-batching (max_grad_batch)
# ─────────────────────────────────────────────────────────────────────────────
#
# The gradient path was the one unbounded batch in the optimizer: a single
# fwd+bwd over all 2N (desired, contrast) specs, retaining every layer's
# activations for the backward. `max_forward_batch` never reached it, which is
# why cutting it 8->4 moved peak GPU memory 23.19 -> 23.24 GiB -- the wrong dial.
#
# Slicing is licensed by the same independence the loss path relies on: members
# are right-padded with causal attention and a per-row encoder, and the backed-off
# objective is a SUM over members, so d(sum_j obj_j)/d(member_i) == d(obj_i)/
# d(member_i). Member i's gradient cannot depend on who shared its slice. These
# pin that: if it ever does, the slice size changes the answer.


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
def test_gradients_are_invariant_to_max_grad_batch(monkeypatch, mode):
    """Same members, every slice size -> the same per-member gradients.

    Sizes span exact divisors, non-divisors (a ragged tail), one-at-a-time, and
    a size exceeding the member count (one whole batch, the pre-#1245 shape).
    """
    model = _build_randomized_model(monkeypatch, seed=27)
    members = [
        ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(900 + i), t_len=t, n_scored=2),
            avoid=_completion(torch.Generator().manual_seed(950 + i), t_len=t + 1, n_scored=2),
        )
        for i, t in enumerate([9, 11, 8, 10, 12])
    ]
    items = [ClassMarginItem(member=m, bag=_BAG) for m in members]

    # 10 specs (2 per item). 32 exceeds them all: ONE batch, the old behaviour.
    baseline = class_margin_objective(
        model, mode=mode, device=_CPU, max_grad_batch=32
    ).gradients(items, 0)
    assert len(baseline) == len(members)
    for b, grad in enumerate(baseline):
        assert grad.id_grad.abs().sum().item() > 0.0, f"item {b}: vacuous (zero) id_grad"
        assert grad.bucket_grad.abs().sum().item() > 0.0, f"item {b}: vacuous bucket_grad"

    for size in (1, 2, 3, 4, 7):
        sliced = class_margin_objective(
            model, mode=mode, device=_CPU, max_grad_batch=size
        ).gradients(items, 0)
        assert len(sliced) == len(baseline)
        for got, want in zip(sliced, baseline, strict=True):
            feature_grads_equal(got, want, exact=False, atol=_CPU_GRAD_ATOL)


def test_every_member_is_returned_exactly_once_in_order(monkeypatch):
    """A ragged tail must not drop or reorder members.

    7 specs at slice 4 -> [4, 3]. An off-by-one in the slice loop would silently
    return a short list, or pair member i's gradient with member j.
    """
    model = _build_randomized_model(monkeypatch, seed=27)
    members = [
        ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(700 + i), t_len=9 + i, n_scored=2),
            avoid=_completion(torch.Generator().manual_seed(750 + i), t_len=10 + i, n_scored=2),
        )
        for i in range(7)
    ]
    items = [ClassMarginItem(member=m, bag=_BAG) for m in members]
    per_member = [
        class_margin_objective(model, device=_CPU, mode=SteeringMode.POSITIVE).gradients(
            [item], 0
        )[0]
        for item in items
    ]
    sliced = class_margin_objective(
        model, device=_CPU, mode=SteeringMode.POSITIVE, max_grad_batch=4
    ).gradients(items, 0)

    assert len(sliced) == 7
    for got, want in zip(sliced, per_member, strict=True):
        feature_grads_equal(got, want, exact=False, atol=_CPU_GRAD_ATOL)
    # Members differ from one another, so a mispairing cannot pass the above.
    assert not torch.allclose(
        per_member[0].id_grad, per_member[-1].id_grad, atol=_CPU_GRAD_ATOL
    ), "vacuous fixture: members are indistinguishable, so ordering is untestable"


def test_max_grad_batch_below_one_raises(monkeypatch):
    """A non-positive bound is a programming error, not a silent whole-batch."""
    model = _build_randomized_model(monkeypatch, seed=27)
    item = ClassMarginItem(
        member=ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(1), t_len=9, n_scored=2),
            avoid=_completion(torch.Generator().manual_seed(2), t_len=10, n_scored=2),
        ),
        bag=_BAG,
    )
    with pytest.raises(FeatureGcgError, match="max_grad_batch must be >= 1"):
        class_margin_objective(
            model, device=_CPU, mode=SteeringMode.POSITIVE, max_grad_batch=0
        ).gradients([item], 0)


def test_slice_count_scales_with_max_grad_batch(monkeypatch):
    """The memory claim, which invariance alone cannot make.

    `test_gradients_are_invariant_to_max_grad_batch` would pass just as happily
    if the bound were ignored and every call ran as ONE batch -- identical
    results are exactly what a no-op produces. Peak memory scales with the
    members per fwd+bwd, so the bound is only real if the number of forwards
    actually rises as the slice shrinks. That is what this counts.
    """
    model = _build_randomized_model(monkeypatch, seed=27)
    members = [
        ClassMarginMember(
            rich=_completion(torch.Generator().manual_seed(300 + i), t_len=9, n_scored=2),
            avoid=_completion(torch.Generator().manual_seed(350 + i), t_len=10, n_scored=2),
        )
        for i in range(5)
    ]
    items = [ClassMarginItem(member=m, bag=_BAG) for m in members]

    forwards: list[int] = []
    real_forward = model.forward

    def _counting_forward(*args, **kwargs):
        forwards.append(1)
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", _counting_forward)

    # 5 items -> 10 (desired, contrast) specs -> ceil(10 / size) forwards.
    for size, expected in ((32, 1), (10, 1), (5, 2), (4, 3), (3, 4), (1, 10)):
        forwards.clear()
        class_margin_objective(
            model, device=_CPU, mode=SteeringMode.POSITIVE, max_grad_batch=size
        ).gradients(items, 0)
        assert len(forwards) == expected, (
            f"max_grad_batch={size} ran {len(forwards)} fwd+bwd passes, expected "
            f"{expected}; the bound is not reaching the gradient path, so peak "
            "memory still scales with the caller's member count (#1245)"
        )
