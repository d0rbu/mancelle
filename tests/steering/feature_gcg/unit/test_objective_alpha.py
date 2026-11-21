"""``objective_alpha`` — the class-margin weighting on the CONTRAST term.

The search minimizes ``g_steer = loss_desired − α·loss_contrast``, i.e. it
MAXIMIZES ``α·unsteered − steered``.  α = 1 is the plain margin, which is why
the default must be exactly recoverable: every artifact on disk was derived
under it, and `_stamp_objective_alpha` strips it from identity on that basis.

What each test here has teeth against:

  * α = 1 equivalence — if α were applied to the DESIRED term instead, or the
    default drifted off 1.0, the plain-margin pins RED.
  * the α·contrast algebra — pinned against SEPARATELY measured per-completion
    losses, so applying α to the wrong term, or dropping it on one of the two
    codepaths (value / gradient), RED.  The value and gradient are checked
    independently precisely because they are two separate implementations of
    the same formula.
  * non-vacuity — every algebra pin asserts the contrast term it scales is not
    ≈0 first; without that, "α changed nothing" would pass for the wrong reason.
"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.config import (
    DEFAULT_OBJECTIVE_ALPHA,
    FeatureGcgConfig,
)
from steering.conditional.feature_gcg.objective import (
    ClassMarginItem,
    ClassMarginMember,
    class_margin_objective,
)
from steering.eval.axbench_base.types import SteeringMode
from tests.steering.feature_gcg._reference import reference_g_steer_grad
from tests.steering.feature_gcg.functional.test_feature_gradient import (
    _build_randomized_model,
)
from tests.steering.feature_gcg.functional.test_feature_objective import (
    _BAG,
    _measure_loss,
    _member_with_margin,
)

_CPU = torch.device("cpu")

# Same CPU/float32 reassociation budget the sibling batching pins use: a batched
# (B>1) forward is not bit-identical to the per-item reference.
_RTOL = 1e-3
_ATOL = 1e-4

# α values exercised by the algebra pins. 1.0 is the default (plain margin), 0.0
# is the degenerate "ignore the unsteered class" end, and the interior values
# are the regime the sweep actually plots.
_ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


def _items(model) -> tuple[list[ClassMarginItem], ClassMarginMember]:
    """One item whose rich/avoid losses differ enough to expose the sign."""
    member = _member_with_margin(model)
    return [ClassMarginItem(member=member, bag=_BAG)], member


def test_default_objective_alpha_is_one() -> None:
    """The default IS the plain margin — the premise identity strip-default rests on.

    Teeth: any drift of this constant off 1.0 silently redefines every bag on
    disk (they were derived under the plain margin but carry no alpha stamp).
    """
    assert DEFAULT_OBJECTIVE_ALPHA == 1.0
    assert FeatureGcgConfig().objective_alpha == 1.0


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
def test_alpha_one_matches_the_unweighted_margin(monkeypatch, mode) -> None:
    """Explicit α=1 == the default == ``loss_desired − loss_contrast``, both modes.

    Teeth: applying α to the desired term instead would still pass at α=1 here,
    which is why ``test_alpha_scales_the_contrast_term_only`` exists — this pin
    covers only the "default is untouched" half.
    """
    model = _build_randomized_model(monkeypatch, seed=41)
    items, _member = _items(model)

    default = class_margin_objective(model, mode=mode, device=_CPU).losses(items)
    explicit = class_margin_objective(
        model, mode=mode, device=_CPU, objective_alpha=1.0
    ).losses(items)

    assert default == explicit


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
@pytest.mark.parametrize("alpha", _ALPHAS)
def test_alpha_scales_the_contrast_term_only(monkeypatch, mode, alpha) -> None:
    """``g(α) == loss_desired − α·loss_contrast`` against separately measured losses.

    desired/contrast are assigned from rich/avoid by ``mode``, so this also pins
    that α rides the CONTRAST side in both modes rather than the rich side in
    one of them.

    Teeth: α on the desired term, or α dropped entirely, breaks every non-1.0
    case. Vacuity is excluded by requiring a non-trivial contrast loss — with
    ``loss_contrast ≈ 0`` all α agree and the pin would prove nothing.
    """
    model = _build_randomized_model(monkeypatch, seed=42)
    items, member = _items(model)

    loss_rich = _measure_loss(model, member.rich, _BAG)
    loss_avoid = _measure_loss(model, member.avoid, _BAG)
    desired, contrast = (
        (loss_rich, loss_avoid) if mode is SteeringMode.POSITIVE else (loss_avoid, loss_rich)
    )
    assert abs(contrast) > 1e-3, "vacuous fixture: contrast ≈ 0 makes every alpha agree"

    produced = class_margin_objective(
        model, mode=mode, device=_CPU, objective_alpha=alpha
    ).losses(items)[0]

    assert produced == pytest.approx(desired - alpha * contrast, rel=_RTOL, abs=_ATOL)


def test_alpha_zero_reduces_to_the_desired_loss(monkeypatch) -> None:
    """α=0 drops the unsteered class entirely: ``g(0) == loss_desired``.

    This is the endpoint the sweep exists to reach — "care only about the
    steered loss" — so it is pinned as a value, not just as a limit.
    """
    model = _build_randomized_model(monkeypatch, seed=43)
    items, member = _items(model)
    loss_rich = _measure_loss(model, member.rich, _BAG)
    loss_avoid = _measure_loss(model, member.avoid, _BAG)
    assert abs(loss_avoid) > 1e-3, "vacuous fixture: contrast ≈ 0 hides the drop"

    produced = class_margin_objective(
        model, mode=SteeringMode.POSITIVE, device=_CPU, objective_alpha=0.0
    ).losses(items)[0]

    assert produced == pytest.approx(loss_rich, rel=_RTOL, abs=_ATOL)
    # ... and is genuinely different from the plain margin it must not equal.
    assert not math.isclose(produced, loss_rich - loss_avoid, rel_tol=1e-5, abs_tol=1e-6)


def test_g_decreases_strictly_as_alpha_rises(monkeypatch) -> None:
    """``g(α)`` is strictly DECREASING in α whenever the contrast loss is positive.

    The direction claim the knob is documented by: raising α pays more for
    inflating the unsteered loss, so it pushes ``g`` down. A sign error on the
    α term reverses this ordering across the whole sweep.
    """
    model = _build_randomized_model(monkeypatch, seed=44)
    items, member = _items(model)
    contrast = _measure_loss(model, member.avoid, _BAG)
    assert contrast > 1e-3, "vacuous fixture: non-positive contrast breaks the ordering"

    def g(a: float) -> float:
        return class_margin_objective(
            model, mode=SteeringMode.POSITIVE, device=_CPU, objective_alpha=a
        ).losses(items)[0]

    values = [g(a) for a in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert all(hi < lo for lo, hi in itertools.pairwise(values)), values


@pytest.mark.parametrize("mode", [SteeringMode.POSITIVE, SteeringMode.NEGATIVE])
@pytest.mark.parametrize("alpha", _ALPHAS)
def test_alpha_scales_the_contrast_gradient(monkeypatch, mode, alpha) -> None:
    """``grad g(α) == grad_desired − α·grad_contrast``, element-wise.

    The gradient is a SEPARATE implementation of the same formula, so it gets
    its own pin: α applied to the value but not the gradient would leave the
    search optimizing one objective while reporting another, and every
    value-side test above would still pass.

    At α=1 the production gradient must equal the frozen per-item reference; at
    every other α it must MOVE OFF it. The exact α algebra is pinned separately
    by ``test_alpha_gradient_equals_desired_minus_alpha_contrast``; this pin is
    the "α actually reaches ``gradients()``" half, in both modes.
    """
    model = _build_randomized_model(monkeypatch, seed=45)
    items, member = _items(model)

    reference = reference_g_steer_grad(model, member, _BAG, mode=mode, device=_CPU)
    produced = class_margin_objective(
        model, mode=mode, device=_CPU, objective_alpha=alpha
    ).gradients(items, step=0)[0]

    if alpha == DEFAULT_OBJECTIVE_ALPHA:
        assert torch.allclose(produced.id_grad, reference.id_grad, rtol=_RTOL, atol=_ATOL)
        return
    # Non-vacuity: the fixture's contrast gradient must be big enough that a
    # different alpha is detectable at all, else "moved off" proves nothing.
    plain = class_margin_objective(model, mode=mode, device=_CPU).gradients(items, step=0)[0]
    contrast_id = (
        class_margin_objective(model, mode=mode, device=_CPU, objective_alpha=0.0)
        .gradients(items, step=0)[0]
        .id_grad
        - plain.id_grad
    )
    assert torch.linalg.vector_norm(contrast_id) > 1e-4, (
        "vacuous fixture: contrast gradient ≈ 0 makes every alpha agree"
    )
    assert not torch.allclose(
        produced.id_grad, reference.id_grad, rtol=1e-5, atol=1e-6
    ), f"alpha={alpha} left the gradient unchanged — alpha is not reaching gradients()"


def test_alpha_gradient_equals_desired_minus_alpha_contrast(monkeypatch) -> None:
    """Exact algebra on the gradient, from the two one-sided gradients.

    ``grad(α)`` is compared against ``grad(0) − α·(grad(0) − grad(1))``:
    ``grad(0) == grad_desired`` (contrast dropped) and
    ``grad(0) − grad(1) == grad_contrast``, so this reconstructs the formula
    from measurements that never assume it.
    """
    model = _build_randomized_model(monkeypatch, seed=46)
    items, _member = _items(model)
    mode = SteeringMode.POSITIVE

    def grad(a: float):
        return class_margin_objective(
            model, mode=mode, device=_CPU, objective_alpha=a
        ).gradients(items, step=0)[0]

    grad_desired = grad(0.0)
    grad_contrast_id = grad_desired.id_grad - grad(1.0).id_grad
    assert torch.linalg.vector_norm(grad_contrast_id) > 1e-4, (
        "vacuous fixture: contrast gradient ≈ 0 makes every alpha agree"
    )

    for alpha in (0.25, 0.5, 2.0):
        expected = grad_desired.id_grad - alpha * grad_contrast_id
        assert torch.allclose(grad(alpha).id_grad, expected, rtol=_RTOL, atol=_ATOL), (
            f"alpha={alpha} gradient does not match desired − alpha*contrast"
        )


@pytest.mark.parametrize("bad_alpha", [-0.5, -1.0, float("inf"), float("nan")])
def test_alpha_domain_is_enforced(bad_alpha) -> None:
    """Negative / non-finite α are rejected by the field's declared domain.

    A NEGATIVE α would REWARD lowering the contrast loss, inverting the class
    margin into a nonsense objective; a non-finite one would poison every
    value and gradient. Pinned to ``BeartypeCallHintParamViolation`` rather
    than bare ``Exception`` so a domain widened to ``float`` still REDs here.
    """
    with pytest.raises(BeartypeCallHintParamViolation):
        FeatureGcgConfig(objective_alpha=bad_alpha)


def test_empty_items_short_circuit_under_alpha(monkeypatch) -> None:
    """The empty-batch fast paths stay empty for any α (no alpha-shaped crash)."""
    model = _build_randomized_model(monkeypatch, seed=47)
    obj = class_margin_objective(
        model, mode=SteeringMode.POSITIVE, device=_CPU, objective_alpha=0.25
    )
    assert obj.losses([]) == []
    assert obj.gradients([], step=0) == []
