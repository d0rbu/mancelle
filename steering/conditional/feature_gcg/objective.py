"""Pluggable GCG objectives for feature-space conditioning optimization.

This module supplies the two :class:`FeatureObjective` implementations the
feature_gcg optimizer ranks candidate feature-id / magnitude-bucket edits with.
Both mirror ``run_gcg``'s ``objective_batch`` (forward-only :meth:`losses`) and
``gradient_provider_batch`` (:meth:`gradients`, returning the Task-3
:class:`FeatureGrad`) seams.

``lm_loss_on_masked_tokens`` (the default)
------------------------------------------
Minimizes the byte/length-normalized cross-entropy of ONE teacher-forced
completion over its scored tokens (``dec_target != -1``) — the single-sequence
language-model objective (spec §4).

``class_margin_objective`` (the spec's TOP ADVERSARY, §7.2 — pulled into Phase 1)
--------------------------------------------------------------------------------
Each member is a ``(prefix, rich, avoid)`` pair scored under ONE shared
conditioning bag. ``desired``/``contrast`` are mapped from ``rich``/``avoid`` by
mode (positive → desired=rich, contrast=avoid; negative swapped), reusing — and
kept equivalent to, by the sign/FD test rather than a call — the canonical
``model_optimize._desired_contrast`` rule. The optimizer **minimizes**::

    g_steer       = loss_desired − loss_contrast    # == −primary_margin_y
    g_steer_grad  = grad_desired − grad_contrast     # element-wise FeatureGrad diff

``g_steer`` is the **single explicit negation** of the existing adversarial
``objective_value`` (``loss_contrast − loss_desired``, `optimize.py:91-111`).
Minimizing ``g_steer`` therefore MAXIMIZES the signed class margin
``primary_margin_y = loss_contrast − loss_desired`` (i.e. STRENGTHENS steering).

**The sign trap (spec §7.2 — the #1 adversary):** there is exactly ONE explicit
negation. Do NOT *also* swap desired/contrast (double negation rebuilds the
adversary), and do NOT negate only the scalar or only the gradient (ranking and
the accept-gate would then disagree). The gradient mirrors the value's single
negation: the desired and contrast completions (each conditioning the SHARED bag
at its own ``generation_slot``) are scored in ONE batched fwd+bwd and their
per-completion :class:`FeatureGrad`s subtracted element-wise. The sign/FD test in
``tests/steering/feature_gcg/functional/test_feature_objective.py`` REDs on any sign flip
or double swap.

Batched-only production
-----------------------
:meth:`losses` and :meth:`gradients` are batched: one padded forward (and, for
gradients, one backward) over every ``(member, bag)`` work-unit — the sole
production codepath.  The per-item reference they are proven equivalent to
(``reference_masked_loss`` / ``reference_g_steer`` / ``reference_g_steer_grad``)
lives in ``tests.steering.feature_gcg._reference``; the equivalence + sign teeth are in
``tests/steering/feature_gcg/{unit,integration}/test_perf_batching.py``.

Shared mechanics
----------------
Both objectives reduce a member's per-token loss row with a **mean masked CE**
(``Σ CE over masked tokens / n_masked``). That reduction is SUM-DECOMPOSABLE (the
normalizer ``n_masked`` is a per-member constant), honoring the Task-3 batched
contract: the batched backward sums the per-member scalars for ONE backward, so
each member receives its own gradient. Per issue #510 the loss-only forward is
**never** wrapped in ``torch.no_grad()``; :meth:`losses` simply never
backpropagates, so no parameter ``.grad`` is populated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

from steering.conditional.feature_gcg.config import (
    DEFAULT_MAX_FORWARD_BATCH,
    DEFAULT_MAX_GRAD_BATCH,
    DEFAULT_OBJECTIVE_ALPHA,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGrad,
    FeatureGradMember,
    ObjectiveScalarFn,
    _batched_masked_losses,
    _compute_feature_grads_batched,
    _ForwardSpec,
    feature_conditioning_gradient_batched,
    substitute_bag,
)
from steering.eval.axbench_base.types import SteeringMode


@runtime_checkable
class FeatureObjective(Protocol):
    """The objective surface ``run_gcg`` consumes (forward losses + gradients)."""

    def losses(self, items: Sequence[Any]) -> list[float]:
        """Forward-only objective value per item (→ ``run_gcg.objective_batch``)."""
        ...

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        """Per-item ``FeatureGrad`` (→ ``run_gcg.gradient_provider_batch``)."""
        ...


@dataclass
class FeatureLmItem:
    """One ``(member, bag)`` work-unit for :func:`lm_loss_on_masked_tokens`.

    ``member`` is a single teacher-forced completion (a :class:`FeatureGradMember`
    whose ``segment_conditions[generation_slot]`` is an ignored placeholder); the
    ``bag`` is the shared conditioning substituted at that slot before scoring.
    """

    member: FeatureGradMember
    bag: FeatureConditioning


@dataclass
class ClassMarginMember:
    """A ``(prefix, rich, avoid)`` pair — the two teacher-forced completions.

    ``rich`` and ``avoid`` are each a :class:`FeatureGradMember` (their
    ``generation_slot`` placeholder is replaced by the shared bag). The
    desired/contrast roles are assigned per mode at scoring time, NOT here.
    """

    rich: FeatureGradMember
    avoid: FeatureGradMember


@dataclass
class ClassMarginItem:
    """One ``(member, bag)`` work-unit for :func:`class_margin_objective`."""

    member: ClassMarginMember
    bag: FeatureConditioning


def _desired_contrast[Completion](
    mode: SteeringMode, rich: Completion, avoid: Completion
) -> tuple[Completion, Completion]:
    """Map ``(rich, avoid)`` → ``(desired, contrast)`` by mode.

    Mirrors (independently — kept equivalent by the sign/FD test, not a call) the
    canonical ``model_optimize._desired_contrast``: positive → desired=rich,
    contrast=avoid; negative flips the pair. This is the SAME ordering the
    canonical ``signed_class_margin`` sign rule is built on, so
    ``g_steer = loss_desired − loss_contrast`` equals ``−signed_class_margin``.
    """
    if mode == SteeringMode.POSITIVE:
        return rich, avoid
    if mode == SteeringMode.NEGATIVE:
        return avoid, rich
    raise FeatureGcgError(f"unknown steering mode {mode!r}")


def _masked_mean_scalar(mask: Tensor) -> ObjectiveScalarFn:
    """Build a SUM-DECOMPOSABLE ``row → mean CE over masked tokens`` reduction.

    ``mask`` is a ``(T,)`` boolean selecting which decoder positions count toward
    the loss. The returned closure reduces a member's per-token loss row to
    ``row[:T][mask].sum() / n_masked``. ``n_masked`` is a per-member constant, so
    summing the scalar across members (the Task-3 batched-backward contract)
    decomposes cleanly per member.

    A loss row may be right-padded LONGER than the mask (the batched decoder pads
    to the group ``t_max``; padded targets are ``-1`` ⇒ CE 0 and live beyond the
    member's real positions, which are left-aligned at index 0); it must not be
    shorter.
    """
    mask_bool = mask.reshape(-1).to(torch.bool)
    n_masked = int(mask_bool.sum().item())
    if n_masked == 0:
        raise FeatureGcgError("objective token mask selects zero tokens to score")

    def _scalar(row: Tensor) -> Tensor:
        flat = row.reshape(-1)
        mask_len = mask_bool.shape[0]
        if flat.shape[0] < mask_len:
            raise FeatureGcgError(
                f"loss row length {flat.shape[0]} is shorter than the token mask "
                f"length {mask_len}"
            )
        selected = flat[:mask_len][mask_bool.to(flat.device)]
        return selected.sum() / n_masked

    return _scalar


class _LmLossObjective:
    """``lm_loss_on_masked_tokens`` implementation — see the factory.

    Each member's scored tokens are its own ``dec_target != -1`` positions, so
    members may carry completions of differing length (the universal multi-member case).
    """

    def __init__(
        self,
        model: Any,
        *,
        device: torch.device | str | int | None,
        max_forward_batch: int = DEFAULT_MAX_FORWARD_BATCH,
        max_grad_batch: int = DEFAULT_MAX_GRAD_BATCH,
    ) -> None:
        self._model = model
        self._device = device
        self._max_forward_batch = max_forward_batch
        self._max_grad_batch = max_grad_batch

    def losses(self, items: Sequence[Any]) -> list[float]:
        """ONE padded forward over all (member, bag) items → per-item masked-mean CE.

        Each item becomes a fully-resolved ``_ForwardSpec`` (its candidate bag
        substituted at the member's ``generation_slot``) with the member's OWN
        ``dec_target != -1`` token mask, so items carrying different bags and/or
        differing completion lengths — as ``run_gcg``'s candidate batch does —
        score correctly under one per-row forward.  The per-item reference this
        batches is ``tests.steering.feature_gcg._reference.reference_masked_loss``.
        """
        if not items:
            return []
        was_training = bool(getattr(self._model, "training", False))
        self._model.eval()
        try:
            return self._losses_batched(items)
        finally:
            self._model.train(was_training)

    def _losses_batched(self, items: Sequence[Any]) -> list[float]:
        specs = [
            _ForwardSpec(
                dec_input=item.member.dec_input,
                dec_target=item.member.dec_target,
                segment_conditions=substitute_bag(item.member, item.bag),
                seg_map=item.member.seg_map,
                generation_slot=item.member.generation_slot,
            )
            for item in items
        ]
        scalar_fns = [_masked_mean_scalar(item.member.dec_target != -1) for item in items]
        return _batched_masked_losses(
            self._model,
            specs,
            scalar_fns,
            self._device,
            max_forward_batch=self._max_forward_batch,
        )

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        # run_gcg's gradient path passes items that all share ONE target bag
        # (core.py builds them from a single `target`): one batched fwd+bwd with a
        # per-member token mask (each member's own dec_target != -1), so
        # completions of differing length score correctly.
        if not items:
            return []
        target = items[0].bag
        if any(item.bag.digest() != target.digest() for item in items[1:]):
            raise FeatureGcgError(
                "LM gradient batch received items with differing bags; run_gcg "
                "passes one shared target bag per gradient call."
            )
        members = [item.member for item in items]
        scalar_fns = [
            _masked_mean_scalar(item.member.dec_target != -1) for item in items
        ]
        return feature_conditioning_gradient_batched(
            self._model,
            members,
            target,
            objective_scalar_fns=scalar_fns,
            device=self._device,
            max_grad_batch=self._max_grad_batch,
        )


class _ClassMarginObjective:
    """``class_margin_objective`` implementation — see the factory (spec §7.2)."""

    def __init__(
        self,
        model: Any,
        *,
        mode: SteeringMode,
        device: torch.device | str | int | None,
        max_forward_batch: int = DEFAULT_MAX_FORWARD_BATCH,
        max_grad_batch: int = DEFAULT_MAX_GRAD_BATCH,
        objective_alpha: float = DEFAULT_OBJECTIVE_ALPHA,
    ) -> None:
        self._model = model
        self._mode = mode
        self._device = device
        self._max_forward_batch = max_forward_batch
        self._max_grad_batch = max_grad_batch
        self._objective_alpha = float(objective_alpha)

    def losses(self, items: Sequence[Any]) -> list[float]:
        """ONE padded forward over all 2N (desired, contrast) completions → g_steer.

        ``g_steer = loss_desired − α·loss_contrast`` is recovered per item from the
        interleaved ``(desired_i, contrast_i)`` flat results — the SINGLE explicit
        §7.2 negation (do NOT flip or double-swap).  At the default ``α = 1`` this
        is the plain margin and the per-item reference it batches is
        ``tests.steering.feature_gcg._reference.reference_g_steer``.
        """
        if not items:
            return []
        was_training = bool(getattr(self._model, "training", False))
        self._model.eval()
        try:
            return self._losses_batched(items)
        finally:
            self._model.train(was_training)

    def _losses_batched(self, items: Sequence[Any]) -> list[float]:
        specs, scalar_fns = self._desired_contrast_specs(items)
        flat = _batched_masked_losses(
            self._model,
            specs,
            scalar_fns,
            self._device,
            max_forward_batch=self._max_forward_batch,
        )
        alpha = self._objective_alpha
        return [flat[2 * i] - alpha * flat[2 * i + 1] for i in range(len(items))]

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        """ONE fwd+bwd over all 2N (desired, contrast) completions → g_steer_grad.

        Pairs the interleaved per-spec :class:`FeatureGrad`s back into
        ``grad_desired − α·grad_contrast`` per item — the SAME element-wise SINGLE
        §7.2 negation as the value, carrying the SAME ``α``, and reusing the
        desired call's shared read-only ``id_weight`` / ``bucket_weight`` (never
        mutated).  At the default ``α = 1`` the per-item reference this batches is
        ``tests.steering.feature_gcg._reference.reference_g_steer_grad``.
        """
        if not items:
            return []
        alpha = self._objective_alpha
        specs, scalar_fns = self._desired_contrast_specs(items)
        flat = _compute_feature_grads_batched(
            self._model,
            specs,
            scalar_fns,
            self._device,
            max_grad_batch=self._max_grad_batch,
        )
        results: list[FeatureGrad] = []
        for i in range(len(items)):
            grad_desired = flat[2 * i]
            grad_contrast = flat[2 * i + 1]
            results.append(
                FeatureGrad(
                    id_grad=grad_desired.id_grad - alpha * grad_contrast.id_grad,
                    bucket_grad=grad_desired.bucket_grad - alpha * grad_contrast.bucket_grad,
                    id_weight=grad_desired.id_weight,
                    bucket_weight=grad_desired.bucket_weight,
                )
            )
        return results

    def _desired_contrast_specs(
        self, items: Sequence[Any]
    ) -> tuple[list[_ForwardSpec], list[ObjectiveScalarFn]]:
        """Interleaved ``(desired_i, contrast_i)`` specs + masked-mean fns for a 2N batch.

        Each item contributes its ``desired`` then its ``contrast`` completion (mode
        mapped via :func:`_desired_contrast`), the shared candidate ``item.bag``
        substituted into each at its ``generation_slot``.  The strict
        ``(desired, contrast)`` order is what :meth:`losses` / :meth:`gradients`
        rely on to re-pair the flat results.
        """
        specs: list[_ForwardSpec] = []
        scalar_fns: list[ObjectiveScalarFn] = []
        for item in items:
            desired, contrast = _desired_contrast(
                self._mode, item.member.rich, item.member.avoid
            )
            for completion in (desired, contrast):
                specs.append(
                    _ForwardSpec(
                        dec_input=completion.dec_input,
                        dec_target=completion.dec_target,
                        segment_conditions=substitute_bag(completion, item.bag),
                        seg_map=completion.seg_map,
                        generation_slot=completion.generation_slot,
                    )
                )
                scalar_fns.append(_masked_mean_scalar(completion.dec_target != -1))
        return specs, scalar_fns


def lm_loss_on_masked_tokens(
    model: Any,
    *,
    device: torch.device | str | int | None = None,
    max_forward_batch: int = DEFAULT_MAX_FORWARD_BATCH,
    max_grad_batch: int = DEFAULT_MAX_GRAD_BATCH,
) -> FeatureObjective:
    """Default objective: minimize mean masked CE of one completion (see module).

    Each item's scored tokens are its member's ``dec_target != -1`` positions
    (length/byte-normalized = divided by the number of scored tokens), so members
    may carry completions of differing length. Items are :class:`FeatureLmItem`
    ``(member, bag)`` pairs.

    ``losses`` runs ONE padded forward over all items (the candidate-evaluation
    hotspot — ~84% of the lm_loss optimize on the real model); ``gradients`` runs
    ONE batched fwd+bwd.  This **batched path is the sole production codepath**.
    The per-item reference it is proven equivalent to (CPU/float32 near-exact in
    ``tests/steering/feature_gcg/unit/test_perf_batching.py``; real-model bf16 within a
    derived atol in ``tests/steering/feature_gcg/integration/test_perf_batching.py``) lives
    in tests as ``tests.steering.feature_gcg._reference.reference_masked_loss``.

    ``max_forward_batch`` bounds the members per sub-batched loss forward
    (default :data:`DEFAULT_MAX_FORWARD_BATCH`; see ``FeatureGcgConfig.max_forward_batch``).
    ``max_grad_batch`` is its GRADIENT-path counterpart (default
    :data:`DEFAULT_MAX_GRAD_BATCH`): that path retains activations for the
    backward, so it is the heavier of the two and needs its own, smaller bound.
    """
    return _LmLossObjective(
        model,
        device=device,
        max_forward_batch=max_forward_batch,
        max_grad_batch=max_grad_batch,
    )


def class_margin_objective(
    model: Any,
    *,
    mode: SteeringMode,
    device: torch.device | str | int | None = None,
    max_forward_batch: int = DEFAULT_MAX_FORWARD_BATCH,
    max_grad_batch: int = DEFAULT_MAX_GRAD_BATCH,
    objective_alpha: float = DEFAULT_OBJECTIVE_ALPHA,
) -> FeatureObjective:
    """Class-margin objective: minimize ``g_steer`` (STRENGTHEN steering); §7.2.

    Items are :class:`ClassMarginItem` ``(member, bag)`` pairs whose member carries
    the ``rich``/``avoid`` completions. ``losses`` returns
    ``g_steer = loss_desired − α·loss_contrast`` and ``gradients`` returns the
    element-wise ``grad_desired − α·grad_contrast`` :class:`FeatureGrad`, with
    desired/contrast assigned from rich/avoid by ``mode``.

    ``objective_alpha`` (α, default :data:`DEFAULT_OBJECTIVE_ALPHA` = 1.0) weights
    the CONTRAST term only: minimizing ``desired − α·contrast`` is maximizing
    ``α·unsteered − steered``. α = 1 is the plain margin, and every value and
    gradient below is then arithmetically identical to the pre-α form, so the
    references named below remain exact at the default. α < 1 discounts the
    reward for inflating the unsteered loss, steering the search toward lowering
    ``loss_desired`` instead of degrading the model off-target.

    Both ``losses`` and ``gradients`` batch all 2N (desired + contrast) forwards
    into ONE call each, preserving the single ``desired − contrast`` negation.
    This **batched path is the sole production codepath**.  The per-item reference
    it is proven equivalent to (CPU/float32 near-exact + bf16 derived atol in
    ``tests/steering/feature_gcg/{unit,integration}/test_perf_batching.py``) lives in tests
    as ``tests.steering.feature_gcg._reference.reference_g_steer`` /
    ``reference_g_steer_grad``.

    ``max_forward_batch`` bounds the members per sub-batched loss forward
    (default :data:`DEFAULT_MAX_FORWARD_BATCH`; see ``FeatureGcgConfig.max_forward_batch``).
    ``max_grad_batch`` is its GRADIENT-path counterpart (default
    :data:`DEFAULT_MAX_GRAD_BATCH`): that path retains activations for the
    backward, so it is the heavier of the two and needs its own, smaller bound.
    """
    return _ClassMarginObjective(
        model,
        mode=mode,
        device=device,
        max_forward_batch=max_forward_batch,
        max_grad_batch=max_grad_batch,
        objective_alpha=objective_alpha,
    )


__all__ = [
    "ClassMarginItem",
    "ClassMarginMember",
    "FeatureLmItem",
    "FeatureObjective",
    "class_margin_objective",
    "lm_loss_on_masked_tokens",
]
