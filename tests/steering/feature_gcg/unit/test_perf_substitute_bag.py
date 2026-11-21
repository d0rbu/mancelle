"""Unit tests for the shared ``substitute_bag`` helper (opt #14).

Verifies:
  (a) Hypothesis property: ``substitute_bag(member, bag)`` matches the inline
      reference — same length; the original object at every non-generation slot;
      ``bag`` at ``generation_slot``; exactly one slot differs from the input.
  (b) Hypothesis property: out-of-range ``generation_slot`` raises
      ``FeatureGcgError`` with the correct message.
  (c) Tiny-model differential: ``feature_conditioning_gradient_batched`` (which
      calls ``substitute_bag`` internally after the refactor) produces a
      ``FeatureGrad`` that is ``feature_grads_equal(exact=True)`` to the
      unmodified ``feature_conditioning_gradient`` (singular path, which takes
      pre-substituted ``segment_conditions`` and is unaffected by this refactor).
  (d) @slow real-model differential: same oracle on real d12_l12 (GPU required,
      written but not run; auto-marked ``@slow`` via ``real_model`` conftest fixture).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from hypothesis import HealthCheck, given, settings

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGradMember,
    feature_conditioning_gradient_batched,
    substitute_bag,
)
from steering.conditional.feature_gcg.objective import _masked_mean_scalar
from tests.steering.feature_gcg._equivalence import (
    bags,
    differential,
    members,
)
from tests.steering.feature_gcg._reference import feature_conditioning_gradient
from tests.steering.feature_gcg.functional.test_feature_gradient import (
    _build_randomized_model,
    _objective,
)

_CPU = torch.device("cpu")

# Documented GPU atomic-backward tolerance for the real-model differential (D1).
# The N=1 batched path and the singular path build the IDENTICAL (1, T) kernel
# shape, so in exact arithmetic id_grad/bucket_grad are bit-identical; the only
# delta is the GPU encoder-embedding backward's atomic ``index_add`` (not bitwise
# reproducible across two separate backwards).  ``torch.equal`` across two backwards
# therefore FLAKES; ``allclose`` at this atol absorbs the atomic noise (empirically
# ~1e-6) while a real substitution bug (wrong slot/object/off-by-one) shifts the
# grad by O(0.1-1) — far above it.  The SHARED static weights are still compared
# with ``torch.equal`` (one cached object, byte-identical).
_GPU_ATOMIC_GRAD_ATOL = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Inline reference (pinned copy of the pre-refactor logic — the oracle)
# ─────────────────────────────────────────────────────────────────────────────


def _inline_substitute(
    member: FeatureGradMember, bag: FeatureConditioning
) -> list[FeatureConditioning]:
    """Pinned inline reference — the pre-refactor gradient.py logic.

    This is the exact code that lived in ``feature_conditioning_gradient_batched``
    before the substitute_bag refactor.  ``substitute_bag`` must produce an equal
    result for every valid input.
    """
    conditions = list(member.segment_conditions)
    if not 0 <= member.generation_slot < len(conditions):
        raise FeatureGcgError(
            f"generation_slot {member.generation_slot} out of range "
            f"[0, {len(conditions)})"
        )
    conditions[member.generation_slot] = bag
    return conditions


# ─────────────────────────────────────────────────────────────────────────────
# (a) Hypothesis property: substitute_bag matches the inline reference
# ─────────────────────────────────────────────────────────────────────────────


@given(members(None, _CPU), bags())
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_substitute_bag_happy_path(
    member_list: list[FeatureGradMember], bag: FeatureConditioning
) -> None:
    """Property: substitute_bag(member, bag) equals the inline reference.

    Checks:
    * Same length as the original ``segment_conditions``.
    * Non-generation slots hold the SAME object (identity, not equality).
    * The generation slot holds ``bag`` (identity check).
    * Intermediate: exactly ONE slot differs from the original conditions.
    """
    for member in member_list:
        result = substitute_bag(member, bag)
        reference = _inline_substitute(member, bag)

        # Same length
        assert len(result) == len(reference), (
            f"length mismatch: got {len(result)}, expected {len(reference)}"
        )
        assert len(result) == len(member.segment_conditions), (
            "result length differs from original segment_conditions length"
        )

        # Generation slot holds the substituted bag (identity)
        assert result[member.generation_slot] is bag, (
            "generation_slot does not hold the substituted bag (identity check failed)"
        )

        # Every non-generation slot holds the ORIGINAL object (identity, not copy)
        for i, orig in enumerate(member.segment_conditions):
            if i != member.generation_slot:
                assert result[i] is orig, (
                    f"slot {i} (non-generation) does not hold the original object: "
                    f"expected identity, got a different object"
                )

        # Intermediate assertion: EXACTLY one slot differs from the input
        n_differing = sum(
            1 for i, r in enumerate(result)
            if r is not member.segment_conditions[i]
        )
        assert n_differing == 1, (
            f"expected exactly 1 differing slot, got {n_differing}; "
            f"generation_slot={member.generation_slot}, n_segs={len(result)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# (b) Hypothesis property: out-of-range generation_slot raises FeatureGcgError
# ─────────────────────────────────────────────────────────────────────────────


@given(bags())
@settings(max_examples=20, deadline=None)
def test_substitute_bag_out_of_range_negative(bag: FeatureConditioning) -> None:
    """Out-of-range (negative) generation_slot raises FeatureGcgError."""
    member = FeatureGradMember(
        dec_input=torch.zeros((1, 4), dtype=torch.long),
        dec_target=torch.zeros((1, 4), dtype=torch.long),
        segment_conditions=[
            FeatureConditioning(entries=()),
            FeatureConditioning(entries=()),
        ],
        seg_map=torch.zeros((1, 4), dtype=torch.long),
        generation_slot=-1,  # out of range [0, 2)
    )
    with pytest.raises(FeatureGcgError, match=r"generation_slot -1 out of range \[0, 2\)"):
        substitute_bag(member, bag)


@given(bags())
@settings(max_examples=20, deadline=None)
def test_substitute_bag_out_of_range_too_large(bag: FeatureConditioning) -> None:
    """Out-of-range (>= n_segs) generation_slot raises FeatureGcgError."""
    member = FeatureGradMember(
        dec_input=torch.zeros((1, 4), dtype=torch.long),
        dec_target=torch.zeros((1, 4), dtype=torch.long),
        segment_conditions=[
            FeatureConditioning(entries=()),
            FeatureConditioning(entries=()),
        ],
        seg_map=torch.zeros((1, 4), dtype=torch.long),
        generation_slot=5,  # out of range [0, 2)
    )
    with pytest.raises(FeatureGcgError, match=r"generation_slot 5 out of range \[0, 2\)"):
        substitute_bag(member, bag)


# ─────────────────────────────────────────────────────────────────────────────
# (c) Tiny-model differential: batched path (with substitute_bag) == singular
# ─────────────────────────────────────────────────────────────────────────────


def test_batched_substitute_bag_byte_identical_to_singular(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """feature_conditioning_gradient_batched (via substitute_bag) is byte-identical to
    the unmodified singular feature_conditioning_gradient for N=1.

    Oracle: the singular path takes pre-substituted ``segment_conditions`` and is
    unaffected by opt #14 — it is the byte-identical reference.  The N=1 batched
    path reduces to the same B=1 forward+backward, so torch.equal holds.

    This test detects any regression introduced by the substitute_bag refactor: if
    the substitution logic breaks (wrong slot, wrong object, off-by-one), the
    batched result diverges from the singular reference.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=42)
    device = _CPU
    gen = torch.Generator().manual_seed(42)
    t_len = 10

    seg0 = FeatureConditioning(entries=((3, 1), (9, 2)))
    target = FeatureConditioning(entries=((17, 3), (23, 4)))

    dec_input = torch.randint(1, 100, (1, t_len), generator=gen, device=device)
    dec_target = torch.randint(0, 100, (1, t_len), generator=gen, device=device)
    seg_map = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.long, device=device
    )
    generation_slot = 1

    # Member has a placeholder at generation_slot (substitute_bag will replace it)
    member = FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[seg0, FeatureConditioning(entries=())],
        seg_map=seg_map,
        generation_slot=generation_slot,
    )

    def reference_fn() -> Any:
        """Singular path: pre-substituted conditions, unaffected by this refactor."""
        return feature_conditioning_gradient(
            model,
            dec_input=dec_input,
            dec_target=dec_target,
            segment_conditions=[seg0, target],
            seg_map=seg_map,
            generation_slot=generation_slot,
            objective_scalar_fn=_objective,
            device=device,
        )

    def candidate_fn() -> Any:
        """Batched path: uses substitute_bag internally after the refactor."""
        grads = feature_conditioning_gradient_batched(
            model,
            [member],
            target,
            objective_scalar_fns=[_objective],
            device=device,
        )
        return grads[0]

    differential(reference_fn, candidate_fn, tensors_eq=torch.equal)


@given(members(None, _CPU), bags())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_batched_substitute_bag_hypothesis(
    monkeypatch: pytest.MonkeyPatch,
    member_list: list[FeatureGradMember],
    bag: FeatureConditioning,
) -> None:
    """Hypothesis: batched[0] == singular for varied member/bag draws on tiny model.

    Extends the deterministic test to randomized members and bags.  The singular
    path is the oracle: pre-substitute ``bag`` at ``generation_slot`` and call
    ``feature_conditioning_gradient`` directly.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=99)

    # Take just the first member to keep the test fast (batch=1 is sufficient)
    member = member_list[0]
    gen_slot = member.generation_slot
    substituted = list(member.segment_conditions)
    substituted[gen_slot] = bag

    def reference_fn() -> Any:
        return feature_conditioning_gradient(
            model,
            dec_input=member.dec_input,
            dec_target=member.dec_target,
            segment_conditions=substituted,
            seg_map=member.seg_map,
            generation_slot=gen_slot,
            objective_scalar_fn=_objective,
            device=_CPU,
        )

    def candidate_fn() -> Any:
        grads = feature_conditioning_gradient_batched(
            model,
            [member],
            bag,
            objective_scalar_fns=[_objective],
            device=_CPU,
        )
        return grads[0]

    differential(reference_fn, candidate_fn, tensors_eq=torch.equal)


# ─────────────────────────────────────────────────────────────────────────────
# (d) @slow real-model differential (GPU required, written but not run)
# ─────────────────────────────────────────────────────────────────────────────


def test_substitute_bag_real_model_differential(real_model: Any) -> None:
    """[opt #14] Batched path equals singular on real d12_l12 (D1: flake-free).

    Oracle: ``feature_conditioning_gradient`` (singular, unaffected by opt #14)
    with pre-substituted ``segment_conditions``.  The N=1 batched path that calls
    ``substitute_bag`` internally must produce the same ``FeatureGrad``.

    D1 fix: ``id_grad`` / ``bucket_grad`` come from TWO independent real-model
    backwards whose encoder-embedding ``index_add`` is atomic (not bitwise
    reproducible on GPU), so ``torch.equal`` across them FLAKES — compare with
    ``allclose`` at the documented ``_GPU_ATOMIC_GRAD_ATOL`` instead.  The SHARED
    static weights (``id_weight`` / ``bucket_weight``) ARE one cached object, so
    those stay ``torch.equal`` (and the comparison is non-vacuous: a real
    substitution bug moves ``id_grad`` by O(0.1-1), far above the atol).

    Auto-marked @slow by the conftest hook (requests ``real_model`` fixture,
    re-exported via ``tests/steering/feature_gcg/unit/conftest.py``).  Enable via::

        export FEATURE_GCG_D12_CKPT=/path/to/d12_encdec_checkpoint
        pytest -m slow tests/steering/feature_gcg/unit/test_perf_substitute_bag.py -v
    """
    device = next(real_model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(42)
    t_len = 16
    dec_input = torch.randint(1, 1000, (1, t_len), generator=gen, device=device)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=device)
    dec_target[0, t_len - 1] = 50  # single non-ignored target
    seg_map = torch.zeros((1, t_len), dtype=torch.long, device=device)
    seg_map[0, t_len // 2 :] = 1  # second half → generation slot

    seg0 = FeatureConditioning(entries=((5, 1), (11, 2)))
    bag = FeatureConditioning(entries=((17, 1),))
    generation_slot = 1

    member = FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[seg0, FeatureConditioning(entries=())],
        seg_map=seg_map,
        generation_slot=generation_slot,
    )

    objective_fn = _masked_mean_scalar(dec_target[0] != -1)

    reference = feature_conditioning_gradient(
        real_model,
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[seg0, bag],
        seg_map=seg_map,
        generation_slot=generation_slot,
        objective_scalar_fn=objective_fn,
        device=device,
    )
    candidate = feature_conditioning_gradient_batched(
        real_model,
        [member],
        bag,
        objective_scalar_fns=[objective_fn],
        device=device,
    )[0]

    # id_grad / bucket_grad: two independent GPU backwards → atomic index_add
    # non-determinism → allclose (shape-guarded), NOT torch.equal.
    assert candidate.id_grad.shape == reference.id_grad.shape
    assert candidate.bucket_grad.shape == reference.bucket_grad.shape
    assert torch.allclose(
        candidate.id_grad, reference.id_grad, atol=_GPU_ATOMIC_GRAD_ATOL, rtol=0.0
    ), f"id_grad max|Δ|={(candidate.id_grad - reference.id_grad).abs().max().item():.3e}"
    assert torch.allclose(
        candidate.bucket_grad, reference.bucket_grad, atol=_GPU_ATOMIC_GRAD_ATOL, rtol=0.0
    ), f"bucket_grad max|Δ|={(candidate.bucket_grad - reference.bucket_grad).abs().max().item():.3e}"
    # id_weight / bucket_weight: the SAME shared cached object → byte-identical.
    assert torch.equal(candidate.id_weight, reference.id_weight)
    assert torch.equal(candidate.bucket_weight, reference.bucket_weight)
    # Non-vacuity floor: the substitution path yields a real (non-zero) gen-slot grad.
    assert candidate.id_grad.abs().sum().item() > 0.0
    assert candidate.bucket_grad.abs().sum().item() > 0.0
