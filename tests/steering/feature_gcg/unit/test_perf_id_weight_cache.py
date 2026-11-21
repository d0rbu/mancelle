"""Unit tests for the id_weight/bucket_weight static-weight cache (opt #12).

Verifies:
  (a) the cached accessor returns tensors ``torch.equal`` to inline ``torch.cat``/
      ``detach``;
  (b) the underlying concat is computed EXACTLY ONCE across repeated gradient
      calls in one optimize (cache-hit path is cheap on subsequent calls);
  (c) Hypothesis property: cached == inline for randomized tiny models;
  (d) intermediate-state assertions: shape/dtype/device of the cached tensors
      match the reference;
  (e) full ``FeatureGrad`` differential (TDD Step 5): the cached path is
      byte-identical (``feature_grads_equal(exact=True)``) to the pre-cache
      inline computation.

The @slow real-model test lives in
``tests/steering/feature_gcg/integration/test_perf_real_model.py``.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg import gradient as grad_module
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import (
    FeatureGradMember,
    _compute_feature_grads_batched,
    _ForwardSpec,
    _static_weights,
    _StaticWeights,
    _value_embed_keys,
    _weight_version,
)
from steering.conditional.feature_gcg.objective import lm_loss_on_masked_tokens
from steering.conditional.feature_gcg.optimize import optimize_feature_conditioning_universal
from tests.steering.feature_gcg._equivalence import (
    differential,
    feature_grads_equal,
    tiny_model,  # noqa: F401 (fixture import for conftest discovery)
)
from tests.steering.feature_gcg.functional.test_feature_gradient import (
    _build_randomized_model,
    _make_fd_inputs,
    _objective,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_CPU = torch.device("cpu")


def _inline_id_weight(encoder: Any, ve_keys: Sequence[str]) -> torch.Tensor:
    """Reference: inline computation matching the pre-cache code path."""
    feat_w = encoder.feature_embed.weight.detach()
    ve_weights = [encoder.value_embeds[key].weight.detach() for key in ve_keys]
    return torch.cat([feat_w, *ve_weights], dim=1)


def _make_one_spec_and_fn(model: Any) -> tuple[Any, Any]:
    """Return a single _ForwardSpec + objective_fn for _compute_feature_grads_batched."""
    segment_conditions, dec_input, dec_target, seg_map, gen_slot = _make_fd_inputs(_CPU)
    spec = _ForwardSpec(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=gen_slot,
    )
    return spec, _objective


# ─────────────────────────────────────────────────────────────────────────────
# (a) & (d): cached accessor == inline; shape/dtype/device assertions
# ─────────────────────────────────────────────────────────────────────────────


def test_cached_id_weight_equals_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """_static_weights().id_weight is byte-identical to inline torch.cat."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    reference = _inline_id_weight(encoder, ve_keys)
    cached = _static_weights(encoder, ve_keys).id_weight

    assert torch.equal(cached, reference), (
        "cached id_weight is not byte-identical to inline torch.cat"
    )


def test_cached_bucket_weight_equals_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """_static_weights().bucket_weight is byte-identical to magnitude_embed.weight.detach()."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    reference = encoder.magnitude_embed.weight.detach()
    cached = _static_weights(encoder, ve_keys).bucket_weight

    assert torch.equal(cached, reference), (
        "cached bucket_weight is not byte-identical to magnitude_embed.weight.detach()"
    )


def test_cached_id_weight_intermediate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """(d) Intermediate-state assertions: shape/dtype/device match reference."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    reference = _inline_id_weight(encoder, ve_keys)
    cached_sw = _static_weights(encoder, ve_keys)
    cached = cached_sw.id_weight

    assert cached.shape == reference.shape, (
        f"id_weight shape mismatch: cached={cached.shape}, ref={reference.shape}"
    )
    assert cached.dtype == reference.dtype, (
        f"id_weight dtype mismatch: cached={cached.dtype}, ref={reference.dtype}"
    )
    assert cached.device == reference.device, (
        f"id_weight device mismatch: cached={cached.device}, ref={reference.device}"
    )

    bucket_ref = encoder.magnitude_embed.weight.detach()
    cached_bw = cached_sw.bucket_weight
    assert cached_bw.shape == bucket_ref.shape, (
        f"bucket_weight shape mismatch: cached={cached_bw.shape}, ref={bucket_ref.shape}"
    )
    assert cached_bw.dtype == bucket_ref.dtype, (
        f"bucket_weight dtype mismatch: cached={cached_bw.dtype}, ref={bucket_ref.dtype}"
    )
    assert cached_bw.device == bucket_ref.device, (
        f"bucket_weight device mismatch: cached={cached_bw.device}, ref={bucket_ref.device}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (b) Cache computed EXACTLY ONCE across repeated gradient calls
# ─────────────────────────────────────────────────────────────────────────────


def test_cache_computed_once_across_grad_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The id_weight concat is computed exactly ONCE across N gradient calls.

    We spy by counting cache MISSES: each miss triggers one torch.cat (the
    concat that builds id_weight).  On cache hits the cat is skipped entirely.
    For the same encoder, N calls → exactly 1 miss regardless of N.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch)

    # Reset the cache so prior test state doesn't affect the count.
    # We temporarily swap in a fresh WeakKeyDictionary for the duration of this test.
    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    miss_count: list[int] = [0]
    original_fn = grad_module._static_weights

    def _counting_static_weights(enc: Any, keys: Sequence[str]) -> Any:
        is_miss = enc not in grad_module._encoder_weight_cache
        result = original_fn(enc, keys)
        if is_miss:
            miss_count[0] += 1
        return result

    spec, obj_fn = _make_one_spec_and_fn(model)

    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache), patch.object(
        grad_module, "_static_weights", side_effect=_counting_static_weights
    ):
        n_calls = 4
        for _ in range(n_calls):
            _compute_feature_grads_batched(model, [spec], [obj_fn], _CPU)

    assert miss_count[0] == 1, (
        f"Expected exactly 1 cache miss across {n_calls} gradient calls, "
        f"got {miss_count[0]} — the concat is being recomputed unnecessarily"
    )


def test_cache_key_is_encoder_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct encoder instances each get their own cache entry."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model_a = _build_randomized_model(monkeypatch, seed=1)
    model_b = _build_randomized_model(monkeypatch, seed=2)
    enc_a = model_a.encoder
    enc_b = model_b.encoder
    ve_keys_a = _value_embed_keys(model_a)
    ve_keys_b = _value_embed_keys(model_b)

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        _static_weights(enc_a, ve_keys_a)
        _static_weights(enc_b, ve_keys_b)
        # After two distinct encoders, the cache should have 2 entries.
        assert len(fresh_cache) == 2, (
            f"Expected 2 cache entries for 2 distinct encoders, got {len(fresh_cache)}"
        )
        # A third call with enc_a should be a hit (still 2 entries).
        _static_weights(enc_a, ve_keys_a)
        assert len(fresh_cache) == 2, (
            "A repeated call with the same encoder should NOT add a new cache entry"
        )


# ─────────────────────────────────────────────────────────────────────────────
# (c) Hypothesis property test: cached == inline across randomized tiny models
# ─────────────────────────────────────────────────────────────────────────────


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_cached_weights_hypothesis_tiny_models(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    """Hypothesis: for any random seed, cached weights are byte-identical to inline.

    Covers varying random initializations to prove the cache is not just correct
    for one specific model configuration.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=seed)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        sw = _static_weights(encoder, ve_keys)

    # id_weight: byte-identical to inline cat
    reference_id = _inline_id_weight(encoder, ve_keys)
    assert torch.equal(sw.id_weight, reference_id), (
        f"seed={seed}: cached id_weight differs from inline at first element "
        f"{(sw.id_weight - reference_id).abs().max().item():.3e}"
    )

    # bucket_weight: byte-identical to magnitude_embed.weight.detach()
    reference_bw = encoder.magnitude_embed.weight.detach()
    assert torch.equal(sw.bucket_weight, reference_bw), (
        f"seed={seed}: cached bucket_weight differs from inline"
    )


# ─────────────────────────────────────────────────────────────────────────────
# (e) Full FeatureGrad differential: cached path byte-identical (TDD Step 5)
# ─────────────────────────────────────────────────────────────────────────────


def test_differential_cached_path_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached path produces byte-identical FeatureGrad to a reference that uses
    a fresh-cache-cleared call.

    We compare two consecutive calls with the same model/spec:
      - reference: first call (cache miss → computes id_weight)
      - candidate:  second call (cache hit → reuses id_weight)
    Both should produce byte-identical FeatureGrad (exact=True).
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=42)
    spec, obj_fn = _make_one_spec_and_fn(model)

    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )

    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        # 1st call: cache miss — id_weight computed via torch.cat
        ref_grads = _compute_feature_grads_batched(model, [spec], [obj_fn], _CPU)
        # 2nd call: cache hit — id_weight reused from WeakKeyDictionary
        cand_grads = _compute_feature_grads_batched(model, [spec], [obj_fn], _CPU)

    assert len(ref_grads) == 1 and len(cand_grads) == 1
    feature_grads_equal(ref_grads[0], cand_grads[0], exact=True)


def test_differential_via_equivalence_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the shared ``differential`` harness (from _equivalence.py) to compare
    the reference (no cache) vs candidate (with cache) for a list of FeatureGrads.

    Both functions are run under a fresh cache so the first-call baseline is
    always a cache miss (byte-identical to the pre-opt inline path).
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=7)
    spec, obj_fn = _make_one_spec_and_fn(model)

    def reference_fn() -> list:
        fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
            weakref.WeakKeyDictionary()
        )
        with patch.object(grad_module, "_encoder_weight_cache", fresh):
            return _compute_feature_grads_batched(model, [spec], [obj_fn], _CPU)

    def candidate_fn() -> list:
        # Second call under a fresh cache: still a miss on first call, then a hit.
        # The RESULT must be byte-identical to the single-call reference.
        fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
            weakref.WeakKeyDictionary()
        )
        with patch.object(grad_module, "_encoder_weight_cache", fresh):
            _compute_feature_grads_batched(model, [spec], [obj_fn], _CPU)  # prime cache
            return _compute_feature_grads_batched(model, [spec], [obj_fn], _CPU)

    differential(reference_fn, candidate_fn, tensors_eq=torch.equal)


# ─────────────────────────────────────────────────────────────────────────────
# Per-optimize scoping: the cache is refreshed at each optimize entry point so a
# same-encoder object whose weights changed in-place is NOT served stale weights
# ─────────────────────────────────────────────────────────────────────────────


def _attach_geometric_meta(model: Any) -> None:
    """Attach the partition metadata optimize() reads (geometric mean=3, max_k=6)."""
    model._loaded_meta_data = {
        "user_config": {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 3.0,
                "min_k": 0,
                "max_k": 6,
            }
        }
    }


def test_consecutive_optimizes_refresh_cache_after_inplace_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two optimizes on the SAME encoder, mutating feature_embed.weight in-place
    between them, rebuild from the NEW weights and release the cache on exit.

    The structural guard (ve_keys + column count) CANNOT detect an in-place value
    change to same-shaped weights. This test drives the real universal optimizer
    twice, observes the actual matrices served inside each call, and requires the
    module cache to be empty after both successful exits. It REDs if the exit
    release is removed. Because optimize #1's exit already empties the cache, this
    fixture does not isolate the entry clear; that boundary is covered by
    ``test_optimize_loop.py``'s exact call assertions.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=123)
    _attach_geometric_meta(model)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    t_len = 12
    torch.manual_seed(5)
    tokens = torch.randint(1, 100, (1, t_len + 1))
    n_scored = 3
    target_mask = [False] * (t_len - n_scored) + [True] * n_scored
    half = t_len // 2
    seg_map = torch.tensor([[0] * half + [1] * (t_len - half)], dtype=torch.long)
    conditions = [
        FeatureConditioning(entries=((5, 1), (11, 2))),
        FeatureConditioning(entries=()),  # warm start at the optimized slot
    ]
    dec_input = tokens[:, :-1]
    dec_target = tokens[:, 1:].long().clone()
    target = torch.tensor(target_mask, dtype=torch.bool, device=dec_target.device)
    dec_target[0, ~target] = -1
    member = FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=conditions,
        seg_map=seg_map,
        generation_slot=1,
    )
    cfg = FeatureGcgConfig(
        max_steps=2,
        candidate_batch_size=2,
        top_k=4,
        temperature=1.0,
        master_seed=0,
        reg_lambda=0.0,
        min_k_pos=0,
        max_k_pos=6,
    )

    fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
    observed: list[torch.Tensor] = []
    real_static_weights = grad_module._static_weights

    def _observed_static_weights(enc: Any, keys: Sequence[str]) -> _StaticWeights:
        result = real_static_weights(enc, keys)
        observed.append(result.id_weight.detach().clone())
        return result

    with (
        patch.object(grad_module, "_encoder_weight_cache", fresh),
        patch.object(grad_module, "_static_weights", _observed_static_weights),
    ):
        # Optimize #1 — populates the cache from the ORIGINAL (v1) weights.
        optimize_feature_conditioning_universal(
            model,
            members=[member],
            warm_start=conditions[1],
            objective_factory=lambda m: lm_loss_on_masked_tokens(m, device="cpu"),
            config=cfg,
        )
        n_v1_observations = len(observed)
        assert n_v1_observations > 0
        id_weight_v1 = observed[-1]
        assert len(fresh) == 0, "optimize #1 retained static weights after exit"

        # Mutate feature_embed.weight IN PLACE (same object, same shape) → v2.
        with torch.no_grad():
            encoder.feature_embed.weight.add_(7.0)

        # Optimize #2 — must clear + rebuild the cache from the v2 weights.
        optimize_feature_conditioning_universal(
            model,
            members=[member],
            warm_start=conditions[1],
            objective_factory=lambda m: lm_loss_on_masked_tokens(m, device="cpu"),
            config=cfg,
        )
        v2_observations = observed[n_v1_observations:]
        assert v2_observations
        assert len(fresh) == 0, "optimize #2 retained static weights after exit"

    expected_v2 = _inline_id_weight(encoder, ve_keys)  # inline cat of CURRENT weights

    # The mutation must actually have changed id_weight (else the test is vacuous).
    assert not torch.equal(id_weight_v1, expected_v2), (
        "in-place mutation did not change id_weight — test would be vacuous"
    )
    # Every cache observation inside optimize #2 must reflect the NEW weights,
    # never the stale v1 matrix from optimize #1.
    assert all(torch.equal(item, expected_v2) for item in v2_observations), (
        "static-weight cache served STALE id_weight after an in-place weight "
        "mutation between two optimizes"
    )


def test_clear_static_weight_cache_empties_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """clear_static_weight_cache() drops all entries from the module cache."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=55)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
    with patch.object(grad_module, "_encoder_weight_cache", fresh):
        _static_weights(encoder, ve_keys)
        assert len(fresh) == 1, "cache should hold one entry after a miss"
        grad_module.clear_static_weight_cache()
        assert len(fresh) == 0, "clear_static_weight_cache() must empty the cache"


# ─────────────────────────────────────────────────────────────────────────────
# Cache invariant guard: stale/wrong cache raises loudly
# ─────────────────────────────────────────────────────────────────────────────


def test_stale_cache_shape_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached id_weight with wrong column count raises FeatureGcgError loudly.

    This verifies the cheap structural guard in _static_weights fires on a stale
    cache rather than silently serving wrong weights.  The guard is a ``raise``
    (not a bare ``assert``), so it ALSO holds under ``python -O`` (which strips
    asserts) — the whole point of the C1 fix.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    # Plant a poisoned cache entry with wrong column count.
    bad_id_weight = torch.zeros(10, 999, dtype=torch.float32)  # wrong column count
    bad_bw = encoder.magnitude_embed.weight.detach()
    poisoned = _StaticWeights(
        id_weight=bad_id_weight,
        bucket_weight=bad_bw,
        ve_keys=tuple(ve_keys),
    )
    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    fresh_cache[encoder] = poisoned

    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        with pytest.raises(FeatureGcgError, match="id_weight"):
            _static_weights(encoder, ve_keys)


def test_stale_cache_ve_keys_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached entry with wrong ve_keys raises FeatureGcgError loudly (holds under -O)."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    # Plant a cache entry with wrong ve_keys (stale from a different config).
    bad_sw = _StaticWeights(
        id_weight=encoder.feature_embed.weight.detach(),
        bucket_weight=encoder.magnitude_embed.weight.detach(),
        ve_keys=("9999",),  # wrong keys
    )
    fresh_cache: weakref.WeakKeyDictionary[Any, _StaticWeights] = (
        weakref.WeakKeyDictionary()
    )
    fresh_cache[encoder] = bad_sw

    with patch.object(grad_module, "_encoder_weight_cache", fresh_cache):
        with pytest.raises(FeatureGcgError, match="ve_keys"):
            _static_weights(encoder, ve_keys)


# ─────────────────────────────────────────────────────────────────────────────
# C2 (T7 review): the version probe refreshes the cache on an in-place REPLACEMENT
# of a static weight tensor (same shape, new object/storage)
# ─────────────────────────────────────────────────────────────────────────────


def test_cache_refreshes_on_weight_tensor_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing feature_embed.weight with a NEW same-shape Parameter refreshes the
    cache — the next ``_static_weights`` returns the NEW weights, not the stale ones.

    The structural guard (ve_keys + column count) passes (same shape + keys), so
    ONLY the version probe (id + data_ptr + shape) catches the swap.

    Teeth: removing the ``cached.version == _weight_version(...)`` check from
    ``_static_weights`` makes the second call return the STALE cached id_weight →
    ``torch.equal(sw2.id_weight, expected_v2)`` REDs.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=77)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
    with patch.object(grad_module, "_encoder_weight_cache", fresh):
        sw1 = _static_weights(encoder, ve_keys)
        id_weight_v1 = sw1.id_weight.clone()
        version_v1 = sw1.version

        # Replace the feature_embed weight TENSOR (new object + storage, same shape).
        with torch.no_grad():
            new_w = encoder.feature_embed.weight.detach().clone() + 5.0
        encoder.feature_embed.weight = torch.nn.Parameter(new_w)

        # The cheap version probe must have changed (new id/data_ptr).
        assert _weight_version(encoder, ve_keys) != version_v1, (
            "weight-tensor replacement did not change the version probe — vacuous"
        )

        sw2 = _static_weights(encoder, ve_keys)

    expected_v2 = _inline_id_weight(encoder, ve_keys)  # inline cat of the NEW weights
    # The swap must actually have changed id_weight (else the test is vacuous).
    assert not torch.equal(id_weight_v1, expected_v2), (
        "weight replacement did not change id_weight — test would be vacuous"
    )
    # The refreshed entry reflects the NEW weights, not the stale v1.
    assert torch.equal(sw2.id_weight, expected_v2), (
        "static-weight cache served STALE id_weight after the feature_embed weight "
        "tensor was REPLACED in place; the version probe failed to detect the swap"
    )


def test_cache_hit_recomputes_no_concat(monkeypatch: pytest.MonkeyPatch) -> None:
    """The version check on a cache HIT is cheap: it never re-runs ``torch.cat``.

    Spies ``torch.cat`` across N repeated ``_static_weights`` calls on a STATIC
    encoder (no weight mutation): exactly ONE cat (the first miss); subsequent
    cache hits validate the version probe WITHOUT rebuilding — so the C2 version
    check did not regress the per-optimize reuse the cache exists for.
    """
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)
    model = _build_randomized_model(monkeypatch, seed=88)
    encoder = model.encoder
    ve_keys = _value_embed_keys(model)

    cat_calls: list[int] = []
    real_cat = torch.cat

    def _counting_cat(*args: Any, **kwargs: Any) -> torch.Tensor:
        cat_calls.append(1)
        return real_cat(*args, **kwargs)

    fresh: weakref.WeakKeyDictionary[Any, _StaticWeights] = weakref.WeakKeyDictionary()
    with patch.object(grad_module, "_encoder_weight_cache", fresh), patch.object(
        grad_module.torch, "cat", _counting_cat
    ):
        for _ in range(5):
            _static_weights(encoder, ve_keys)

    assert len(cat_calls) == 1, (
        f"expected exactly 1 torch.cat (the miss) across 5 calls, got {len(cat_calls)} "
        "— the version check is rebuilding on cache hits (perf regression)"
    )
