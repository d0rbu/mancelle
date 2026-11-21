"""Real-model differential tests for performance optimizations (Tasks 3-6).

Layer B (@slow + skip-not-fail, GPU required)
----------------------------------------------
All tests in this module request the ``real_model`` fixture
(``tests/steering/feature_gcg/_equivalence.py``), which is:

* auto-marked ``slow`` by the conftest hook — individual tests need no explicit
  ``@pytest.mark.slow`` annotation.
* skip-not-fail when ``FEATURE_GCG_D12_CKPT`` is absent or the path does not
  exist.

Enable via::

    export FEATURE_GCG_D12_CKPT=/path/to/d12_encdec_checkpoint
    pytest -m slow tests/steering/feature_gcg/integration/test_perf_real_model.py -v

Task 3: id_weight/bucket_weight cache (opt #12)
------------------------------------------------
The static-weight cache (``_static_weights``) must produce tensors that are
BYTE-IDENTICAL (``torch.equal``) to the inline ``torch.cat(feat_w, *ve_ws)``
path on the real d12_l8 / d12_l12 checkpoint.  This is a structural guarantee:
no floating-point accumulation is involved (the cached tensor IS the original
detached weight tensor), so exact equality is the correct predicate — not
``allclose``.

Why the full-FeatureGrad tests share ONE backward
--------------------------------------------------
This opt changes ONLY the ``id_weight`` / ``bucket_weight`` fields (the
``id_grad`` / ``bucket_grad`` come from the backward pass, which the opt does not
touch).  Comparing two *separate* backward passes would conflate the opt with
CUDA backward non-determinism (the encoder embedding backward uses atomic
``index_add``, which is not bitwise-reproducible on GPU).  So the
full-FeatureGrad tests run the real forward/backward ONCE (cached path) and
build the pre-opt reference by re-pairing the SAME ``id_grad`` / ``bucket_grad``
with the INLINE-recomputed weights — isolating the opt (the weights) from
backward non-determinism while still exercising the real d12 model end-to-end.

DO NOT RUN: this file is written as part of Task 3 but requires the GPU and
checkpoint to execute.  The controller runs it on the GPU after Task 3 is
committed.  See the task brief at ``.superpowers/sdd/task-3-brief.md``.
"""

from __future__ import annotations

import weakref
from typing import Any
from unittest.mock import patch

import pytest
import torch

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg import gradient as grad_module
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGrad,
    _compute_feature_grads_batched,
    _ForwardSpec,
    _static_weights,
    _StaticWeights,
    _value_embed_keys,
)
from steering.conditional.feature_gcg.objective import _masked_mean_scalar
from tests.steering.feature_gcg._equivalence import (
    differential,
    feature_grads_equal,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force SDPA (FlexAttention has no CPU backward) — matches the sibling modules.

    Without this (D2), on a box with ``FEATURE_GCG_D12_CKPT`` set but no CUDA the
    ``real_model`` fixture loads on CPU and the gradient tests below hit the
    FlexAttention CPU backward (no CPU impl) → raise instead of running the
    byte-identity check.  Mirrors ``integration/test_perf_batching.py``.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _inline_id_weight(encoder: Any, ve_keys: list[str]) -> torch.Tensor:
    """Reference: the pre-cache inline computation."""
    feat_w = encoder.feature_embed.weight.detach()
    ve_weights = [encoder.value_embeds[k].weight.detach() for k in ve_keys]
    return torch.cat([feat_w, *ve_weights], dim=1)


def _make_real_spec(model: Any, device: torch.device) -> tuple[_ForwardSpec, Any]:
    """Build a minimal _ForwardSpec + ObjectiveScalarFn closure for the real model.

    A single non-ignored target at the last position (which cross-attends the
    generation segment, slot 1) yields a non-vacuous gen-slot gradient — the same
    design as the FD oracle.  Bucket indices are derived from the model's
    ``num_buckets`` (``magnitude_embed.weight.shape[0]``) so they are always in
    range: d12_l12 has ``num_buckets == 2`` (the minimum), and an out-of-range
    bucket triggers a CUDA device-side assert in the magnitude-embed gather.
    """
    t_len = 16
    gen = torch.Generator(device=device).manual_seed(42)
    dec_input = torch.randint(1, 1000, (1, t_len), generator=gen, device=device)
    dec_target = torch.full((1, t_len), -1, dtype=torch.long, device=device)
    dec_target[0, t_len - 1] = 50  # single non-ignored target
    seg_map = torch.zeros((1, t_len), dtype=torch.long, device=device)
    seg_map[0, t_len // 2 :] = 1  # second half → generation slot

    # Bucket indices MUST satisfy 0 <= b < num_buckets or the magnitude-embed
    # gather device-side-asserts.  num_buckets can be as small as 2 (d12_l12),
    # so derive the top valid bucket from the model instead of hardcoding it.
    num_buckets = int(model.encoder.magnitude_embed.weight.shape[0])
    hi = min(1, num_buckets - 1)
    assert 0 <= hi < num_buckets, (
        f"derived bucket {hi} out of range for num_buckets={num_buckets}"
    )
    seg0 = FeatureConditioning(entries=((5, 0), (11, hi)))
    seg1 = FeatureConditioning(entries=((17, hi),))
    spec = _ForwardSpec(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=[seg0, seg1],
        seg_map=seg_map,
        generation_slot=1,
    )
    # _masked_mean_scalar is a FACTORY: call it with the (T,) scored-token mask to
    # get the ObjectiveScalarFn closure that _compute_feature_grads_batched invokes
    # as objective_scalar_fns[b](rows[b]).
    objective_fn = _masked_mean_scalar(dec_target[0] != -1)
    return spec, objective_fn


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 real-model tests (auto-marked @slow via real_model fixture)
# ─────────────────────────────────────────────────────────────────────────────


def test_cached_id_weight_byte_identical_real(real_model: Any) -> None:
    """[Task 3] Cached id_weight is byte-identical to inline on the real d12 model.

    This is STRUCTURAL byte-identity: the cache stores the exact detached weight
    tensor (no floating-point arithmetic), so torch.equal is the correct
    predicate.  Any drift would indicate a code bug, not numerical accumulation.
    """
    encoder = real_model.encoder
    ve_keys = _value_embed_keys(real_model)

    reference = _inline_id_weight(encoder, ve_keys)

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        cached = _static_weights(encoder, ve_keys).id_weight

    assert torch.equal(cached, reference), (
        "cached id_weight differs from inline on the real d12 model"
    )
    assert cached.shape == reference.shape, (
        f"id_weight shape mismatch: cached={cached.shape}, ref={reference.shape}"
    )
    assert cached.dtype == reference.dtype, (
        f"id_weight dtype mismatch: cached={cached.dtype}, ref={reference.dtype}"
    )
    assert cached.device == reference.device, (
        f"id_weight device mismatch: cached={cached.device}, ref={reference.device}"
    )


def test_cached_bucket_weight_byte_identical_real(real_model: Any) -> None:
    """[Task 3] Cached bucket_weight is byte-identical to inline on the real d12 model."""
    encoder = real_model.encoder
    ve_keys = _value_embed_keys(real_model)

    reference = encoder.magnitude_embed.weight.detach()

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        cached_sw = _static_weights(encoder, ve_keys)

    assert torch.equal(cached_sw.bucket_weight, reference), (
        "cached bucket_weight differs from inline on the real d12 model"
    )


def test_cached_path_feature_grad_byte_identical_real(real_model: Any) -> None:
    """[Task 3] Full FeatureGrad from the cached path is byte-identical (exact=True).

    Runs the real forward/backward ONCE (cached path) and builds the pre-opt
    reference by re-pairing the SAME id_grad / bucket_grad with the INLINE
    weights — isolating the opt (cached weights) from CUDA backward
    non-determinism.  See the module docstring.
    """
    device = next(real_model.parameters()).device
    encoder = real_model.encoder
    ve_keys = _value_embed_keys(real_model)
    spec, obj_fn = _make_real_spec(real_model, device)

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        cached_grads = _compute_feature_grads_batched(
            real_model, [spec], [obj_fn], device
        )

    assert len(cached_grads) == 1
    cached_grad = cached_grads[0]

    # Pre-opt reference: identical grads from THIS backward, paired with the
    # inline-recomputed weights (what _extract_feature_grads built pre-cache).
    reference_grad = FeatureGrad(
        id_grad=cached_grad.id_grad,
        bucket_grad=cached_grad.bucket_grad,
        id_weight=_inline_id_weight(encoder, ve_keys),
        bucket_weight=encoder.magnitude_embed.weight.detach(),
    )
    feature_grads_equal(cached_grad, reference_grad, exact=True)


def test_cached_path_differential_real(real_model: Any) -> None:
    """[Task 3] differential() confirms byte-identity for list[FeatureGrad] on real model.

    Same single-backward isolation as the test above, driven through the shared
    ``differential`` harness so all four FeatureGrad tensor fields are compared
    (id_grad / bucket_grad shared ⇒ trivially equal; id_weight / bucket_weight
    cached-vs-inline ⇒ the opt under test).
    """
    device = next(real_model.parameters()).device
    encoder = real_model.encoder
    ve_keys = _value_embed_keys(real_model)
    spec, obj_fn = _make_real_spec(real_model, device)

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        cached_grads = _compute_feature_grads_batched(
            real_model, [spec], [obj_fn], device
        )

    inline_idw = _inline_id_weight(encoder, ve_keys)
    inline_bw = encoder.magnitude_embed.weight.detach()
    reference_grads = [
        FeatureGrad(
            id_grad=g.id_grad,
            bucket_grad=g.bucket_grad,
            id_weight=inline_idw,
            bucket_weight=inline_bw,
        )
        for g in cached_grads
    ]

    differential(
        lambda: reference_grads,
        lambda: cached_grads,
        tensors_eq=torch.equal,
    )
