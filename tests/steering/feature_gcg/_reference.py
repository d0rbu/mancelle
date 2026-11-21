"""Per-item OBJECTIVE reference implementations — the equivalence oracle.

These are the ORIGINAL per-item objective helpers, relocated verbatim from
``steering.conditional.feature_gcg.objective`` when the **batched** objective
became the sole production codepath (feature_gcg Phase-1 Task-6 KEEP decision).
They are no longer shipped; they live here as the equivalence ORACLE the batched
production path is checked against:

* ``reference_masked_loss``   — the old ``objective._masked_loss`` (one loss-only
  forward per completion + bag), verbatim.
* ``reference_g_steer``       — the old ``_ClassMarginObjective._g_steer`` logic
  (two per-item ``reference_masked_loss`` forwards, ``loss_desired −
  loss_contrast``), lifted to a free function ``(model, member, bag, *, mode,
  device)``.
* ``reference_g_steer_grad``  — the old ``_ClassMarginObjective._g_steer_grad``
  logic (two single-member ``feature_conditioning_gradient`` backwards,
  ``grad_desired − grad_contrast``), lifted the same way.

They import the production primitives that remain in the shipping code
(``_assemble_encoder_inputs``, ``substitute_bag``, ``_masked_mean_scalar``,
``_desired_contrast``, and the batched gradient core). The single-member
``feature_conditioning_gradient`` oracle also lives here, so the reference
exercises the same underlying model math as production — the only difference
under test is the per-item loop vs the one padded batch. The
equivalence is proven in ``tests/steering/feature_gcg/unit/test_perf_batching.py``
(CPU/float32 near-exact) and ``tests/steering/feature_gcg/integration/test_perf_batching.py``
(real-model bf16 derived atol).

The §7.2 sign rule is preserved here exactly as in the old production code:
``reference_g_steer`` / ``reference_g_steer_grad`` apply the SINGLE explicit
``desired − contrast`` negation (do NOT also swap or double-negate).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGrad,
    ObjectiveScalarFn,
    _assemble_encoder_inputs,
    _compute_feature_grads_batched,
    _ForwardSpec,
    substitute_bag,
)
from steering.conditional.feature_gcg.objective import (
    ClassMarginMember,
    _desired_contrast,
    _masked_mean_scalar,
)
from steering.eval.axbench_base.types import SteeringMode


def feature_conditioning_gradient(
    model: Any,
    *,
    dec_input: Tensor,
    dec_target: Tensor,
    segment_conditions: Sequence[FeatureConditioning],
    seg_map: Tensor,
    generation_slot: int,
    objective_scalar_fn: ObjectiveScalarFn,
    device: torch.device | str | int | None,
) -> FeatureGrad:
    """Reference the production batch core with an exact one-member batch."""
    spec = _ForwardSpec(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=generation_slot,
    )
    return _compute_feature_grads_batched(model, [spec], [objective_scalar_fn], device)[0]


def reference_masked_loss(
    model: Any,
    *,
    dec_input: Tensor,
    dec_target: Tensor,
    segment_conditions: Sequence[FeatureConditioning],
    seg_map: Tensor,
    scalar_fn: ObjectiveScalarFn,
    device: torch.device | str | int | None,
) -> float:
    """Loss-only forward → scalar masked-mean CE for one completion + bag.

    Verbatim of the old production ``objective._masked_loss``: runs the conditional
    forward with ``loss_reduction="none"`` over the RAW ``(ids, mags, mask)``
    assembled from ``segment_conditions`` (never a cached ``encoder_out``) and
    reduces with ``scalar_fn``.  Per issue #510 the forward is NOT wrapped in
    ``torch.no_grad()``; the caller never backpropagates it, so no parameter
    ``.grad`` is populated and the value is read via ``.item()``.
    """
    ids, mags, mask = _assemble_encoder_inputs(segment_conditions, device)
    out = model(
        dec_input.to(device),
        dec_target.to(device),
        enc_feature_ids=ids,
        enc_magnitudes=mags,
        enc_mask=mask,
        seg_map=seg_map.to(device),
        loss_reduction="none",
    )
    return float(scalar_fn(out).item())


def reference_g_steer(
    model: Any,
    member: ClassMarginMember,
    bag: FeatureConditioning,
    *,
    mode: SteeringMode,
    device: torch.device | str | int | None,
) -> float:
    """Per-item ``g_steer = loss_desired − loss_contrast`` for one (member, bag).

    Verbatim of the old ``_ClassMarginObjective._g_steer``: maps ``(rich, avoid)``
    → ``(desired, contrast)`` by ``mode`` and scores each completion under the
    SHARED ``bag`` substituted at its ``generation_slot`` via two
    ``reference_masked_loss`` forwards.

    The §7.2 single explicit negation: ``loss_desired − loss_contrast`` ==
    ``−primary_margin_y``.  Do NOT also swap desired/contrast (double negation
    rebuilds the adversary).
    """
    desired, contrast = _desired_contrast(mode, member.rich, member.avoid)
    loss_desired = reference_masked_loss(
        model,
        dec_input=desired.dec_input,
        dec_target=desired.dec_target,
        segment_conditions=substitute_bag(desired, bag),
        seg_map=desired.seg_map,
        scalar_fn=_masked_mean_scalar(desired.dec_target != -1),
        device=device,
    )
    loss_contrast = reference_masked_loss(
        model,
        dec_input=contrast.dec_input,
        dec_target=contrast.dec_target,
        segment_conditions=substitute_bag(contrast, bag),
        seg_map=contrast.seg_map,
        scalar_fn=_masked_mean_scalar(contrast.dec_target != -1),
        device=device,
    )
    return loss_desired - loss_contrast


def reference_g_steer_grad(
    model: Any,
    member: ClassMarginMember,
    bag: FeatureConditioning,
    *,
    mode: SteeringMode,
    device: torch.device | str | int | None,
) -> FeatureGrad:
    """Per-item ``g_steer_grad = grad_desired − grad_contrast`` for one (member, bag).

    Verbatim of the old ``_ClassMarginObjective._g_steer_grad``: two same-bag
    single-member ``feature_conditioning_gradient`` backwards (desired and
    contrast, each conditioning the SHARED bag at its ``generation_slot``)
    subtracted element-wise — mirroring the value's SINGLE negation.
    ``id_weight`` / ``bucket_weight`` are SHARED read-only across the two
    same-model calls; the desired call's are reused (never mutated).
    """
    desired, contrast = _desired_contrast(mode, member.rich, member.avoid)
    grad_desired = feature_conditioning_gradient(
        model,
        dec_input=desired.dec_input,
        dec_target=desired.dec_target,
        segment_conditions=substitute_bag(desired, bag),
        seg_map=desired.seg_map,
        generation_slot=desired.generation_slot,
        objective_scalar_fn=_masked_mean_scalar(desired.dec_target != -1),
        device=device,
    )
    grad_contrast = feature_conditioning_gradient(
        model,
        dec_input=contrast.dec_input,
        dec_target=contrast.dec_target,
        segment_conditions=substitute_bag(contrast, bag),
        seg_map=contrast.seg_map,
        generation_slot=contrast.generation_slot,
        objective_scalar_fn=_masked_mean_scalar(contrast.dec_target != -1),
        device=device,
    )
    return FeatureGrad(
        id_grad=grad_desired.id_grad - grad_contrast.id_grad,
        bucket_grad=grad_desired.bucket_grad - grad_contrast.bucket_grad,
        id_weight=grad_desired.id_weight,
        bucket_weight=grad_desired.bucket_weight,
    )


__all__ = [
    "reference_g_steer",
    "reference_g_steer_grad",
    "reference_masked_loss",
]
