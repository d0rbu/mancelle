"""The encoder feature-embed gradient — the feature_gcg Phase 1 risk gate.

This module computes ``dL/dx`` (the gradient of a pluggable LM objective with
respect to the encoder's per-segment conditioning *activations*) for a
conditional :class:`EncoderDecoderGPT`. It is the first-order signal GCG uses to
rank candidate feature-id / magnitude-bucket edits.

The math (verified against the model surface, see the plan's "Design basis")
--------------------------------------------------------------------------
The encoder embeds one conditioning segment as::

    x = feature_embed(ids) + magnitude_embed(mags)          # (·, n_embd)

and, on every ``has_ve`` layer, adds a value-residual driven by
``value_embeds[str(i)](ids)``. Because ``x`` is the *sum* of the feature- and
magnitude-embedding outputs, ``dL/d(feature_embed_out) == dL/d(magnitude_embed_out)
== dL/dx``. **One forward hook on ``feature_embed`` therefore yields the gradient
for BOTH axes** (the "separable, one hook suffices" property):

* **bucket axis** ``dL/dx`` paired with ``magnitude_embed.weight`` — a candidate
  bucket ``b`` scores ``dL/dx · magnitude_embed.weight[b]``.
* **id axis** ``dL/dx`` *concatenated with* every present ``value_embeds[*]``
  activation gradient, paired column-for-column with
  ``concat(feature_embed.weight, value_embeds[*].weight)`` — a candidate
  feature-id ``f`` scores the dot product over BOTH the additive-embedding and
  the value-residual paths, since ``f`` indexes rows in ``feature_embed`` AND in
  every ``value_embeds[*]``.

We therefore hook ``feature_embed`` (for ``dL/dx``) and **each present**
``value_embeds[str(i)]`` (a SPARSE ``ModuleDict`` — only ``has_ve`` layers),
``retain_grad`` the activation outputs, run the model with **raw**
``(enc_feature_ids, enc_magnitudes, enc_mask)`` (never a cached ``encoder_out``,
which would skip the encoder and the hooks would never fire), reduce the loss to
a scalar via ``objective_scalar_fn`` and backpropagate. The gradient is read at
the caller-supplied ``generation_slot``'s real slots.

Mirrors :func:`steering.eval.axbench_bpb_adv.surrogate.teacher_forced_bpb_gradient`
(which hooks the *decoder* ``wte`` / value-embeds); the *encoder*
``feature_embed`` / ``value_embeds`` are hooked by no existing path — this is
genuinely new model-math, hence the finite-difference oracle in
``tests/steering/feature_gcg/functional/test_feature_gradient.py``.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from beartype import beartype
from loguru import logger
from torch import Tensor

from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.conditional.feature_gcg.config import (
    DEFAULT_MAX_FORWARD_BATCH,
    DEFAULT_MAX_GRAD_BATCH,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.refined_types import PositiveIntLike

# A reduction from one member's per-token loss row (shape ``(T,)``) to a scalar.
ObjectiveScalarFn = Callable[[Tensor], Tensor]

# ─────────────────────────────────────────────────────────────────────────────
# Static-weight cache (opt #12)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _StaticWeights:
    """Cached static weight matrices for one encoder instance.

    These are valid for the entire lifetime of a GCG optimize: the encoder is
    never updated (GCG does forward/backward but no optimizer step), so
    ``feature_embed.weight``, ``value_embeds[*].weight``, and
    ``magnitude_embed.weight`` are constant across all gradient calls.

    Attributes
    ----------
    id_weight:
        ``(V, D + sum_kv)`` on a compacted encoder and
        ``(n_sae_features, D + sum_kv)`` otherwise —
        ``concat(feature_embed.weight, value_embeds[*].weight)`` in ascending
        integer layer key order. Compact IDs index its rows; callers translate
        original IDs before lookup.
    bucket_weight:
        ``(num_buckets, D)`` — ``magnitude_embed.weight``.
    ve_keys:
        The ``value_embeds`` key tuple used to build ``id_weight``, kept for the
        cheap invariant guard on cache hits.
    version:
        A cheap identity/version probe — ``(id, data_ptr, shape)`` per static
        weight tensor (see :func:`_weight_version`) — captured when the entry was
        built.  Recomputed on each cache hit so an in-place REPLACEMENT of a weight
        tensor (a new ``Parameter`` / a ``.data`` storage swap) is detected and the
        entry refreshed rather than serving stale weights.  Defaults to ``()`` only
        for hand-built entries in tests; every real entry carries a real version.
    """

    id_weight: Tensor
    bucket_weight: Tensor
    ve_keys: tuple[str, ...]
    version: tuple[Any, ...] = ()


# Module-level cache: encoder module → _StaticWeights.
# WeakKeyDictionary auto-evicts the entry when the encoder module is GC'd —
# avoids id()-reuse and memory-leak pitfalls of a plain dict.
#
# SCOPE: this cache is valid only WITHIN one optimize.  During a GCG optimize the
# encoder is static (forward/backward, no optimizer step — a confirmed
# invariant), so the cached id_weight/bucket_weight stay valid for that optimize.
#
# Across optimizes the SAME encoder object's weights may change.  Two detection
# layers guard staleness:
#   * a cheap version probe (:func:`_weight_version`: id + data_ptr + shape per
#     static weight) recomputed on every cache hit — detects an in-place
#     REPLACEMENT of a weight tensor (a new ``Parameter`` / a ``.data`` storage
#     swap, e.g. swapping in a different SAE) and refreshes the entry; and
#   * the structural guard in :func:`_static_weights` (ve_keys + column count)
#     which raises on a structurally-inconsistent cache hit.
# NEITHER can detect an in-place VALUE mutation on the SAME storage (an optimizer
# step's ``param.add_`` / a ``param.data.copy_`` checkpoint reload keep the same
# id + data_ptr + shape), so the optimizer entry point still calls
# :func:`clear_static_weight_cache` at its start to repopulate from current weights.
_encoder_weight_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()


def clear_static_weight_cache() -> None:
    """Drop cached static weights around a ``run_gcg`` cache-populating lifetime.

    The static-weight cache (:data:`_encoder_weight_cache`) is scoped to one
    optimize: within an optimize the encoder is never updated, so the cache is
    valid; ACROSS optimizes the same encoder object's weights may have been
    mutated in-place (checkpoint swap / fine-tuning step), which the structural
    guard in :func:`_static_weights` cannot detect.  Clearing at entry forces the
    next gradient call to rebuild from current weights.  If execution reaches
    ``run_gcg``—the only region that can populate this private cache—the
    optimizer's ``finally`` clears again so detached device tensors cannot
    outlive that region; earlier setup failures receive only the entry clear
    because those paths cannot populate the cache.

    Cleanup relies on :class:`weakref.WeakKeyDictionary.clear`'s practical
    no-throw invariant.  If cleanup did raise, standard ``finally`` semantics
    would mask a primary exception.
    """
    _encoder_weight_cache.clear()


def _weight_version(encoder: Any, ve_keys: Sequence[str]) -> tuple[Any, ...]:
    """Cheap identity/version probe of the encoder's static weight tensors.

    Returns ``(id, data_ptr, shape)`` for ``feature_embed.weight``,
    ``magnitude_embed.weight``, and each ``value_embeds[k].weight`` (in ``ve_keys``
    order).  Used to detect an in-place REPLACEMENT of a static weight tensor (a
    new ``Parameter`` / a ``.data`` storage swap — e.g. swapping in a different
    SAE) so :func:`_static_weights` can refresh a cache entry the structural guard
    cannot see (same keys + same column count).

    This probe is CHEAP (no ``torch.cat``, no tensor reads — only ``id`` /
    ``data_ptr`` / ``shape``), so recomputing it on every cache hit preserves the
    per-optimize reuse the cache exists for.  It does NOT detect an in-place VALUE
    mutation on the SAME storage (``param.add_`` / ``param.data.copy_`` keep id +
    data_ptr + shape constant) — callers mutating weights in place between gradient
    calls must call :func:`clear_static_weight_cache` (the universal optimizer
    does).
    """

    def _probe(w: Tensor) -> tuple[Any, ...]:
        return (id(w), w.data_ptr(), tuple(w.shape))

    return (
        _probe(encoder.feature_embed.weight),
        _probe(encoder.magnitude_embed.weight),
        tuple(_probe(encoder.value_embeds[k].weight) for k in ve_keys),
    )


def _static_weights(encoder: Any, ve_keys: Sequence[str]) -> _StaticWeights:
    """Return the static weight matrices for ``encoder``, computing once and caching.

    On a cache hit a cheap STRUCTURAL invariant guard (matching ``ve_keys`` and
    ``id_weight`` column count) RAISES :class:`FeatureGcgError` on a stale/wrong
    entry — a ``raise`` rather than a bare ``assert`` so it holds even under
    ``python -O`` (which strips asserts), never silently serving wrong weights.  A
    cheap version probe (:func:`_weight_version`) is then compared: if a static
    weight tensor was REPLACED in place (new ``Parameter`` / storage swap) the
    entry is REFRESHED (rebuilt) rather than served stale.  Neither check can
    detect an in-place VALUE mutation on the same storage, which is why the cache
    is scoped to one optimize via :func:`clear_static_weight_cache`.

    On a cache miss (or a version-detected refresh) the matrices are built via
    ``torch.cat`` and stored.  The encoder's weights are ``detach()``'d so the
    cache tensors carry no grad_fn.

    On a compacted encoder ``id_weight`` has ``V`` rows indexed by compact ID.
    Callers translate each original candidate ID to its compact row before
    indexing it (``FeatureConditioningEditSpace._id_weight_rows``, guarded by a
    fail-loud OOV range check), so this builder needs no compaction guard.
    """
    cached = _encoder_weight_cache.get(encoder)
    if cached is not None:
        # Structural invariant guard — RAISE (not a bare `assert`, which `python
        # -O` strips) so a structurally-wrong/stale cache entry fails loudly under
        # all run modes.  Cheap: no torch.cat, just scalar arithmetic.
        if cached.ve_keys != tuple(ve_keys):
            raise FeatureGcgError(
                f"_static_weights: ve_keys mismatch on cache hit — "
                f"cached={cached.ve_keys!r}, current={tuple(ve_keys)!r}"
            )
        expected_id_cols = int(encoder.feature_embed.weight.shape[1]) + sum(
            int(encoder.value_embeds[k].weight.shape[1]) for k in ve_keys
        )
        if cached.id_weight.shape[1] != expected_id_cols:
            raise FeatureGcgError(
                f"_static_weights: id_weight column count mismatch on cache hit — "
                f"cached={cached.id_weight.shape[1]}, expected={expected_id_cols}"
            )
        # Version check (C2): if every static weight tensor is still the SAME
        # object/storage/shape, the cached id_weight/bucket_weight are valid → hit.
        # Otherwise a weight tensor was replaced in place → fall through + refresh.
        if cached.version == _weight_version(encoder, ve_keys):
            return cached

    # Cache miss (or version-detected refresh): build and store.
    feat_w = encoder.feature_embed.weight.detach()
    ve_weights = [encoder.value_embeds[key].weight.detach() for key in ve_keys]
    id_weight = torch.cat([feat_w, *ve_weights], dim=1)
    bucket_weight = encoder.magnitude_embed.weight.detach()
    result = _StaticWeights(
        id_weight=id_weight,
        bucket_weight=bucket_weight,
        ve_keys=tuple(ve_keys),
        version=_weight_version(encoder, ve_keys),
    )
    _encoder_weight_cache[encoder] = result
    return result


@dataclass
class FeatureGrad:
    """First-order edit signal for one conditioning segment's real slots.

    Attributes
    ----------
    id_grad:
        ``(n_real, D + sum_kv)`` — ``dL/dx`` (the first ``D = n_embd`` columns)
        concatenated with each present ``value_embeds[*]`` activation gradient.
        Column order matches :attr:`id_weight`. A candidate feature-id ``f``'s
        first-order score at a slot is ``id_grad[slot] · id_weight[f]``.
    bucket_grad:
        ``(n_real, D)`` — the same ``dL/dx`` (separable), for the magnitude axis.
        A candidate bucket ``b``'s score is ``bucket_grad[slot] · bucket_weight[b]``.
    id_weight:
        ``(n_sae_features, D + sum_kv)`` —
        ``concat(feature_embed.weight, value_embeds[*].weight)`` in the SAME
        column order as :attr:`id_grad` (value-embeds enumerated by ascending
        integer layer key).
    bucket_weight:
        ``(num_buckets, D)`` — ``magnitude_embed.weight``.
    """

    id_grad: Tensor
    bucket_grad: Tensor
    id_weight: Tensor
    bucket_weight: Tensor


@dataclass
class FeatureGradMember:
    """One teacher-forced member of a universal (N-member) gradient batch.

    The shared bag being optimized is substituted into
    ``segment_conditions[generation_slot]`` by
    :func:`feature_conditioning_gradient_batched`; the value carried here at that
    slot is an ignored placeholder.
    """

    dec_input: Tensor  # (1, T) token ids
    dec_target: Tensor  # (1, T) target ids (-1 = ignored)
    segment_conditions: Sequence[FeatureConditioning]  # per-segment bags
    seg_map: Tensor  # (1, T) decoder-position → segment index
    generation_slot: int  # slot to substitute the shared bag into AND read the grad


def substitute_bag(
    member: FeatureGradMember, bag: FeatureConditioning
) -> list[FeatureConditioning]:
    """Return ``member.segment_conditions`` with ``bag`` at ``generation_slot``.

    Bounds-checks ``generation_slot`` and raises :class:`FeatureGcgError` if it
    is out of range.  This is the shared helper used by both
    ``feature_conditioning_gradient_batched`` (which substitutes the target bag
    per-member before each forward) and the :class:`FeatureObjective`
    implementations in ``objective.py`` (which substitute the candidate bag
    before every loss / gradient evaluation).

    Parameters
    ----------
    member:
        The ``FeatureGradMember`` whose ``segment_conditions`` and
        ``generation_slot`` are used.
    bag:
        The :class:`FeatureConditioning` to place at ``generation_slot``.

    Returns
    -------
    list[FeatureConditioning]
        A fresh list (shallow copy of ``member.segment_conditions``) with ``bag``
        at index ``generation_slot``.  All other slots hold the ORIGINAL objects
        (no copies).
    """
    conditions = list(member.segment_conditions)
    slot = member.generation_slot
    if not 0 <= slot < len(conditions):
        raise FeatureGcgError(f"generation_slot {slot} out of range [0, {len(conditions)})")
    conditions[slot] = bag
    return conditions


@dataclass
class _ForwardSpec:
    """A fully-resolved per-member forward (the shared bag already substituted)."""

    dec_input: Tensor
    dec_target: Tensor
    segment_conditions: Sequence[FeatureConditioning]
    seg_map: Tensor
    generation_slot: int


if TYPE_CHECKING:
    _ForwardSpecSequence = Sequence[_ForwardSpec]
    _ObjectiveScalarFnSequence = Sequence[ObjectiveScalarFn]
    _FloatList = list[float]
else:
    _ForwardSpecSequence = Sequence
    _ObjectiveScalarFnSequence = Sequence
    _FloatList = list


def _value_embed_keys(model: Any) -> list[str]:
    """Present ``value_embeds`` keys, ascending by integer layer index.

    ``value_embeds`` is a SPARSE ``ModuleDict`` — only ``has_ve`` layers are
    present — so we enumerate ``.keys()`` rather than assuming all layers. The
    sorted order fixes the column layout of :attr:`FeatureGrad.id_grad` /
    :attr:`FeatureGrad.id_weight`.
    """
    keys = [str(k) for k in model.encoder.value_embeds]
    return sorted(keys, key=lambda k: int(k))


def _assemble_encoder_inputs(
    segment_conditions: Sequence[FeatureConditioning],
    device: torch.device | str | int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Assemble raw ``(ids, mags, mask)`` of shape ``(1, n_seg, F)`` for one member.

    Delegates to the shared
    :func:`steering.conditional.encoder_decoder.eval_adapter.build_segmented_feature_encoder_inputs`
    (each bag bridged to its canonical ``{fid: bucket}`` dict), so the feature-GCG
    gradient path and the eval adapter assemble encoder inputs **identically** —
    byte-identical ``ids``/``mags``/``mask`` with the same ``F = max(n_real, 1)``
    right-padding rule. ``FeatureConditioning.entries`` are canonically sorted, so
    the per-segment feature-id order matches the gradient's slot↔entry alignment.
    The encoder pads FP8 ×16 internally — no padding here.
    """
    if len(segment_conditions) == 0:
        raise FeatureGcgError("segment_conditions must be non-empty")
    return build_segmented_feature_encoder_inputs(
        [dict(cond.entries) for cond in segment_conditions], device
    )


def _assemble_encoder_batch(
    specs: Sequence[_ForwardSpec], device: torch.device | str | int | None
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Stack per-member encoder inputs, right-padding to common ``(n_seg, F)``."""
    per_member = [_assemble_encoder_inputs(s.segment_conditions, device) for s in specs]
    n_seg_max = max(ids.shape[1] for ids, _, _ in per_member)
    f_max = max(ids.shape[2] for ids, _, _ in per_member)
    batch = len(specs)

    ids = torch.zeros((batch, n_seg_max, f_max), dtype=torch.long, device=device)
    mags = torch.zeros((batch, n_seg_max, f_max), dtype=torch.long, device=device)
    mask = torch.zeros((batch, n_seg_max, f_max), dtype=torch.bool, device=device)
    for b, (mid, mmag, mmask) in enumerate(per_member):
        n_seg, f_dim = mid.shape[1], mid.shape[2]
        ids[b, :n_seg, :f_dim] = mid[0]
        mags[b, :n_seg, :f_dim] = mmag[0]
        mask[b, :n_seg, :f_dim] = mmask[0]
    return ids, mags, mask, n_seg_max


def _assemble_decoder_batch(
    specs: Sequence[_ForwardSpec], device: torch.device | str | int | None
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Stack per-member decoder inputs, right-padding to common ``T``.

    Padded positions carry ``target = -1`` (CE ignored) and ``seg_map = 0`` (the
    reserved empty slot); right-padding + causal attention keeps every real
    position bit-identical to the unpadded single forward.
    """
    batch = len(specs)
    t_max = max(s.dec_input.shape[1] for s in specs)

    dec_input = torch.zeros((batch, t_max), dtype=torch.long, device=device)
    dec_target = torch.full((batch, t_max), -1, dtype=torch.long, device=device)
    seg_map = torch.zeros((batch, t_max), dtype=torch.long, device=device)
    for b, spec in enumerate(specs):
        t_len = spec.dec_input.shape[1]
        dec_input[b, :t_len] = spec.dec_input.to(device).reshape(-1)[:t_len]
        dec_target[b, :t_len] = spec.dec_target.to(device).reshape(-1)[:t_len]
        seg_map[b, :t_len] = spec.seg_map.to(device).reshape(-1)[:t_len]
    return dec_input, dec_target, seg_map, t_max


def _sanitize_grad(grad: Tensor) -> Tensor:
    """Clamp non-finite gradient entries to 0 with a logged error.

    Mirrors ``surrogate.py``: a non-finite gradient with an otherwise usable
    forward is recoverable (the GCG ranking degrades gracefully) but must NOT be
    silent — a non-finite gradient signals an upstream numerical blowup, so it is
    logged at error level.
    """
    n_non_finite = int((~torch.isfinite(grad)).sum().item())
    if n_non_finite:
        logger.error(
            f"feature gradient: clamped {n_non_finite} non-finite "
            "gradient entries to 0; the GCG search continues, but this signals an "
            "upstream numerical blowup."
        )
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_feature_grads_batched(
    model: Any,
    specs: Sequence[_ForwardSpec],
    objective_scalar_fns: Sequence[ObjectiveScalarFn],
    device: torch.device | str | int | None,
    *,
    max_grad_batch: PositiveIntLike = DEFAULT_MAX_GRAD_BATCH,
) -> list[FeatureGrad]:
    """Per-member ``FeatureGrad`` over ``specs``, in sub-batched fwd+bwd slices.

    Members are independent (right-padded decoder + causal attention, per-row
    encoder), so ``d(Σ_j objective_j)/d(member_i activation) == d(objective_i)/
    d(member_i activation)`` — each member's gradient equals its single-forward
    value. ``objective_scalar_fns[b]`` reduces member ``b``'s loss row, so each
    member may carry its own token mask (e.g. completions of differing length).

    That same independence is what makes slicing free: member ``i``'s gradient
    never depends on which other members shared its batch, so processing the
    members in ``max_grad_batch`` slices returns exactly what one big batch
    would, up to the compute dtype's batch-shape kernel rounding (the identical
    caveat :func:`_batched_masked_losses` documents for the loss path).

    Slicing is not optional book-keeping — it is the memory bound. Unlike the
    loss path, this one cannot run under ``torch.no_grad()``: the backward needs
    the retained activations. #1245: a single unsliced fwd+bwd over all 2N
    (desired, contrast) specs OOM'd a 23.51 GiB card in
    ``cross_attention.forward``, ~9s into the derive, and was invisible to
    ``max_forward_batch`` (which bounds only the loss path) -- three attempts at
    cutting the forward knobs moved peak 23.19 -> 23.42 GiB, i.e. the wrong dial.
    """
    if len(specs) == 0:
        raise FeatureGcgError("no members provided to the gradient computation")
    if len(objective_scalar_fns) != len(specs):
        raise FeatureGcgError(
            f"objective_scalar_fns length {len(objective_scalar_fns)} != "
            f"number of members {len(specs)}"
        )
    chunk_size = int(max_grad_batch)
    if chunk_size < 1:
        raise FeatureGcgError(
            f"max_grad_batch must be >= 1, got {max_grad_batch!r}; it bounds the "
            "members per sub-batched fwd+bwd"
        )

    grads: list[FeatureGrad] = []
    for start in range(0, len(specs), chunk_size):
        grads.extend(
            _compute_feature_grads_one_batch(
                model,
                specs[start : start + chunk_size],
                objective_scalar_fns[start : start + chunk_size],
                device,
            )
        )
    return grads


def _compute_feature_grads_one_batch(
    model: Any,
    specs: Sequence[_ForwardSpec],
    objective_scalar_fns: Sequence[ObjectiveScalarFn],
    device: torch.device | str | int | None,
) -> list[FeatureGrad]:
    """ONE padded fwd+bwd over ``specs``; per-member ``FeatureGrad``.

    One slice of :func:`_compute_feature_grads_batched`. Every activation this
    retains is released when it returns, so peak scales with the slice, not with
    the caller's member count.
    """
    encoder = model.encoder
    if torch.is_inference_mode_enabled():
        raise FeatureGcgError(
            "feature gradients cannot run under torch.inference_mode(); "
            "the gradient path needs autograd enabled."
        )
    if not encoder.feature_embed.weight.requires_grad:
        raise FeatureGcgError(
            "encoder.feature_embed.weight.requires_grad is False; the feature-GCG "
            "gradient needs a grad-enabled model (do not load it frozen)."
        )
    ve_keys = _value_embed_keys(model)

    dec_input, dec_target, seg_map, _ = _assemble_decoder_batch(specs, device)
    ids, mags, mask, n_seg_max = _assemble_encoder_batch(specs, device)
    batch, t_max = dec_input.shape

    captured: dict[str, Tensor] = {}

    def _capture(name: str):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, Tensor):
                raise FeatureGcgError(
                    f"encoder embedding hook {name!r} returned non-Tensor output "
                    f"{type(output).__name__}"
                )
            output.retain_grad()
            captured[name] = output

        return _hook

    hooks = [encoder.feature_embed.register_forward_hook(_capture("feat"))]
    for key in ve_keys:
        hooks.append(encoder.value_embeds[key].register_forward_hook(_capture(f"ve:{key}")))

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        # RAW (ids, mags, mask) — NEVER a cached encoder_out (that skips the
        # encoder and the hooks would never fire).
        out = model(
            dec_input,
            dec_target,
            enc_feature_ids=ids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
            loss_reduction="none",
        )
        rows = out.view(batch, t_max)
        # Per-member objective, summed → decomposable backward.
        scalar = torch.stack([objective_scalar_fns[b](rows[b]) for b in range(batch)]).sum()
        # Clear any stale parameter grads so repeated calls don't accumulate.
        model.zero_grad(set_to_none=True)
        scalar.backward()

        return _extract_feature_grads(
            specs=specs,
            encoder=encoder,
            ve_keys=ve_keys,
            captured=captured,
            batch=batch,
            n_seg_max=n_seg_max,
        )
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)


def _extract_feature_grads(
    *,
    specs: Sequence[_ForwardSpec],
    encoder: Any,
    ve_keys: Sequence[str],
    captured: dict[str, Tensor],
    batch: int,
    n_seg_max: int,
) -> list[FeatureGrad]:
    """Read the captured activation grads → per-member ``FeatureGrad``."""
    d_embd = int(encoder.config.n_embd)

    feat_act = captured.get("feat")
    if feat_act is None or feat_act.grad is None:
        raise FeatureGcgError(
            "failed to capture the encoder feature_embed activation gradient "
            "(the encoder may have been skipped via a cached encoder_out)"
        )
    f_pad = feat_act.shape[1]
    feat_grad = feat_act.grad.detach().reshape(batch, n_seg_max, f_pad, d_embd)

    ve_grads: dict[str, Tensor] = {}
    for key in ve_keys:
        ve_act = captured.get(f"ve:{key}")
        if ve_act is None or ve_act.grad is None:
            raise FeatureGcgError(
                f"failed to capture the encoder value_embeds[{key!r}] activation gradient"
            )
        kv_dim = ve_act.shape[-1]
        ve_grads[key] = ve_act.grad.detach().reshape(batch, n_seg_max, f_pad, kv_dim)

    # Static weight matrices — cached for the caller-managed optimization scope
    # (GCG never updates the encoder; recomputing on every grad call is pure
    # waste). The universal optimizer clears this cache on both entry and exit.
    _sw = _static_weights(encoder, ve_keys)
    id_weight = _sw.id_weight
    bucket_weight = _sw.bucket_weight

    results: list[FeatureGrad] = []
    for i, spec in enumerate(specs):
        gen_slot = spec.generation_slot
        n_real = len(spec.segment_conditions[gen_slot])

        dldx = feat_grad[i, gen_slot, :n_real, :]  # (n_real, D) == dL/dx
        ve_parts = [ve_grads[key][i, gen_slot, :n_real, :] for key in ve_keys]
        id_grad = torch.cat([dldx, *ve_parts], dim=1)  # (n_real, D + sum_kv)
        bucket_grad = dldx.clone()  # (n_real, D), separable

        results.append(
            FeatureGrad(
                id_grad=_sanitize_grad(id_grad).contiguous(),
                bucket_grad=_sanitize_grad(bucket_grad).contiguous(),
                id_weight=id_weight,
                bucket_weight=bucket_weight,
            )
        )
    return results


@beartype
def _batched_masked_losses(
    model: Any,
    specs: _ForwardSpecSequence,
    objective_scalar_fns: _ObjectiveScalarFnSequence,
    device: torch.device | str | int | None,
    *,
    max_forward_batch: PositiveIntLike = DEFAULT_MAX_FORWARD_BATCH,
) -> _FloatList:
    """ONE padded forward over ``specs`` → per-member masked-loss scalar (forward-only).

    The loss-path analogue of :func:`_compute_feature_grads_batched`: it assembles
    the SAME right-padded ``(batch, t_max)`` decoder + ``(batch, n_seg, F)`` encoder
    batch and runs ONE conditional forward (mirroring that function's forward with
    ``loss_reduction="none"``), then reduces member ``b``'s row with
    ``objective_scalar_fns[b]``.  It registers NO hooks and never calls
    ``.backward()`` — it is purely forward.

    Members are independent (right-padded decoder + causal attention + per-row
    encoder), so each member's masked loss equals its single-forward
    (the per-item reference ``tests.steering.feature_gcg._reference.reference_masked_loss``)
    value up to the compute dtype's batch-shape kernel rounding: bit-identical in
    exact arithmetic, and within a documented ``~5e-2`` atol under the real model's
    GPU bf16 (see ``tests/steering/feature_gcg/integration/test_perf_batching.py`` for the
    derivation).  ``objective_scalar_fns[b]`` carries member ``b``'s own
    (unpadded) token mask, so members of differing length score correctly: each
    mask truncates ``rows[b]`` to that member's real positions, ignoring the
    right-pad.

    Per issue #510 the forward is **not** wrapped in ``torch.no_grad()`` — exactly
    like ``reference_masked_loss``, it simply never backpropagates, so no parameter
    ``.grad`` is populated.  The ``model.eval()`` / ``model.train()`` toggle is the
    CALLER's responsibility (the objective's ``losses`` method), matching the
    per-item path this batches.
    """
    if len(specs) == 0:
        raise FeatureGcgError("no members provided to the batched loss computation")
    if len(objective_scalar_fns) != len(specs):
        raise FeatureGcgError(
            f"objective_scalar_fns length {len(objective_scalar_fns)} != "
            f"number of members {len(specs)}"
        )
    # RAW (ids, mags, mask) — NEVER a cached encoder_out (mirrors the gradient
    # core's forward exactly, minus the autograd hooks + backward).
    #
    # ``torch.no_grad()``: this is the LOSS path — it never backpropagates, so the
    # autograd graph is pure dead weight. WITHOUT it a single batched forward
    # retains every layer's activations (~22GB for a real encdec member batch) and
    # OOMs a 24GB card. Grad-disabled forward math is bit-identical to the
    # grad-enabled forward, so the values (and the reference-match tolerance) are
    # unchanged; the gradient path (``_compute_feature_grads_batched``) is SEPARATE.
    #
    # Sub-batch chunking: each member's loss is independent (it conditions on its
    # own encoder input and its scalar_fn masks to its own real positions), so a
    # caller's large batch — e.g. the GCG per-step candidate eval,
    # ``len(candidates) x len(members)`` items — is processed in ``max_forward_batch``
    # slices whose fp32 decoder logits fit a 24GB card (default
    # :data:`DEFAULT_MAX_FORWARD_BATCH`; configurable via ``FeatureGcgConfig.max_forward_batch``
    # threaded through the objective). Results are identical to the single-batch path
    # up to bf16 batch-shape rounding regardless of the slice size.
    results: list[float] = []
    with torch.no_grad():
        for start in range(0, len(specs), max_forward_batch):
            chunk_specs = specs[start : start + max_forward_batch]
            chunk_fns = objective_scalar_fns[start : start + max_forward_batch]
            dec_input, dec_target, seg_map, t_max = _assemble_decoder_batch(chunk_specs, device)
            ids, mags, mask, _ = _assemble_encoder_batch(chunk_specs, device)
            b = dec_input.shape[0]
            out = model(
                dec_input,
                dec_target,
                enc_feature_ids=ids,
                enc_magnitudes=mags,
                enc_mask=mask,
                seg_map=seg_map,
                loss_reduction="none",
            )
            rows = out.view(b, t_max)
            results.extend(float(chunk_fns[i](rows[i]).item()) for i in range(b))
    return results


def feature_conditioning_gradient_batched(
    model: Any,
    members: Sequence[FeatureGradMember],
    target: FeatureConditioning,
    *,
    objective_scalar_fns: Sequence[ObjectiveScalarFn],
    device: torch.device | str | int | None,
    max_grad_batch: PositiveIntLike = DEFAULT_MAX_GRAD_BATCH,
) -> list[FeatureGrad]:
    """Per-member ``FeatureGrad`` for the SHARED ``target`` bag, sub-batched fwd+bwd.

    ``target`` is substituted into each member's
    ``segment_conditions[generation_slot]`` (the optimizer optimizes ONE bag over
    the whole group); the result for member ``i`` equals
    the equivalent one-member batched computation for that member alone.
    ``objective_scalar_fns[i]`` reduces member ``i``'s loss row (one per member,
    so members may carry different token masks).

    Notes
    -----
    A caller that mutates the encoder's weights in place (same storage) between
    gradient calls, outside an ``optimize_*`` entry point, must call
    :func:`clear_static_weight_cache` to avoid stale ``id_weight`` /
    ``bucket_weight``.
    """
    if len(objective_scalar_fns) != len(members):
        raise FeatureGcgError(
            f"objective_scalar_fns length {len(objective_scalar_fns)} != "
            f"number of members {len(members)}"
        )
    specs: list[_ForwardSpec] = []
    for member in members:
        conditions = substitute_bag(member, target)
        specs.append(
            _ForwardSpec(
                dec_input=member.dec_input,
                dec_target=member.dec_target,
                segment_conditions=conditions,
                seg_map=member.seg_map,
                generation_slot=member.generation_slot,
            )
        )
    return _compute_feature_grads_batched(
        model, specs, objective_scalar_fns, device, max_grad_batch=max_grad_batch
    )
