"""Ordering and lifecycle tests for the eager post-cross-attention eval seam."""

from __future__ import annotations

import pytest
import torch as t

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from steering.conditional.encoder_decoder.eval_residual import (
    post_cross_attention_residual_counter,
    scoped_post_cross_attention_callback,
)
from tests.encdec_test_helpers import build_model_with_feature_prediction, build_tiny_model


def _inputs(model, *, length: int = 6):
    idx = t.arange(length, dtype=t.int64).remainder(model.dec_config.vocab_size)[None, :]
    feature_ids = t.tensor([[[0, 1], [2, 3]]], dtype=t.int64)
    magnitudes = t.zeros_like(feature_ids)
    mask = t.ones_like(feature_ids, dtype=t.bool)
    seg_map = t.tensor([[0] * (length // 2) + [1] * (length - length // 2)])
    return idx, feature_ids, magnitudes, mask, seg_map


def _forward(model, *, return_fp_logits: bool = False, fp_logits_only: bool = False):
    idx, feature_ids, magnitudes, mask, seg_map = _inputs(model)
    return model(
        idx,
        None,
        enc_feature_ids=feature_ids,
        enc_magnitudes=magnitudes,
        enc_mask=mask,
        seg_map=seg_map,
        encoder_strict=True,
        return_fp_logits=return_fp_logits,
        fp_logits_only=fp_logits_only,
    )


def _cache(model) -> KVCache:
    config = model.dec_config
    return KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=model.get_device(),
        dtype=COMPUTE_DTYPE,
    )


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_callback_observes_exact_post_cross_attention_order(layer: int) -> None:
    model = build_tiny_model(n_layer=3, n_embd=64, xattn_frequency=1).eval()
    model.init_weights()
    block_outputs: list[t.Tensor] = []
    xattn_outputs: list[t.Tensor] = []
    post_outputs: list[t.Tensor] = []
    block_handle = model._decoder_blocks[layer].register_forward_hook(
        lambda _module, _inputs, output: block_outputs.append(output.detach().clone())
    )
    xattn_handle = model._cross_attention_layer(str(layer)).register_forward_hook(
        lambda _module, _inputs, output: xattn_outputs.append(output.detach().clone())
    )

    def capture(layer_index: int, residual: t.Tensor) -> t.Tensor:
        if layer_index == layer:
            post_outputs.append(residual.detach().clone())
        return residual

    try:
        with scoped_post_cross_attention_callback(model, capture), t.no_grad():
            _forward(model)
    finally:
        block_handle.remove()
        xattn_handle.remove()
    assert len(block_outputs) == len(xattn_outputs) == len(post_outputs) == 1
    t.testing.assert_close(
        post_outputs[0],
        block_outputs[0] + xattn_outputs[0],
        rtol=0,
        atol=0,
    )


def test_callback_site_is_still_ordered_on_a_layer_without_cross_attention() -> None:
    model = build_tiny_model(n_layer=3, n_embd=64, xattn_frequency=2).eval()
    model.init_weights()
    assert model._cross_attention_layer("0") is None
    block_outputs: list[t.Tensor] = []
    post_outputs: list[t.Tensor] = []
    handle = model._decoder_blocks[0].register_forward_hook(
        lambda _module, _inputs, output: block_outputs.append(output.detach().clone())
    )

    def capture(layer_index: int, residual: t.Tensor) -> t.Tensor:
        if layer_index == 0:
            post_outputs.append(residual.detach().clone())
        return residual

    try:
        with scoped_post_cross_attention_callback(model, capture), t.no_grad():
            _forward(model)
    finally:
        handle.remove()
    t.testing.assert_close(post_outputs[0], block_outputs[0], rtol=0, atol=0)


def test_exact_counter_arithmetic_runs_in_actual_model_forward() -> None:
    model = build_tiny_model(n_layer=3, n_embd=64, xattn_frequency=1).eval()
    model.init_weights()
    direction = t.linspace(-2.0, 2.0, model.dec_config.n_embd)
    observed: list[tuple[t.Tensor, t.Tensor]] = []

    def arithmetic(layer_index: int, residual: t.Tensor) -> t.Tensor:
        if layer_index != 1:
            return residual
        shifted = residual - 0.25 * direction.to(residual)
        observed.append((residual.detach().clone(), shifted.detach().clone()))
        return shifted

    with scoped_post_cross_attention_callback(model, arithmetic), t.no_grad():
        _forward(model)
    before, after = observed[0]
    t.testing.assert_close(after, before - 0.25 * direction.to(before), rtol=0, atol=0)


def test_zero_counter_is_literal_no_installation_and_default_state_is_inert() -> None:
    model = build_model_with_feature_prediction(
        n_layer=3,
        n_embd=64,
        xattn_frequency=1,
    ).eval()
    model.init_weights()
    state_keys = tuple(model.state_dict())
    rng_before = t.random.get_rng_state().clone()
    with t.no_grad():
        expected_logits, expected_fp = _forward(model, return_fp_logits=True)
        expected_fp_only = _forward(model, fp_logits_only=True)
    rng_after_expected = t.random.get_rng_state().clone()
    t.random.set_rng_state(rng_before)
    scope = post_cross_attention_residual_counter(
        model,
        layer=1,
        direction=t.ones(model.dec_config.n_embd),
        magnitude=0.0,
    )
    with scope:
        assert model._post_cross_attention_eval_callback is None
        with t.no_grad():
            actual_logits, actual_fp = _forward(model, return_fp_logits=True)
            actual_fp_only = _forward(model, fp_logits_only=True)
    assert model._post_cross_attention_eval_callback is None
    assert tuple(model.state_dict()) == state_keys
    assert t.equal(t.random.get_rng_state(), rng_after_expected)
    assert t.equal(actual_logits, expected_logits)
    assert t.equal(actual_fp, expected_fp)
    assert t.equal(actual_fp_only, expected_fp_only)


def test_default_none_path_matches_a_real_cpu_compiled_forward() -> None:
    model = build_tiny_model(n_layer=1, n_embd=64, xattn_frequency=1).eval()
    model.init_weights()
    with t.no_grad():
        expected = _forward(model)
        compiled = t.compile(model, backend="eager", dynamic=False)
        actual = _forward(compiled)
    assert t.equal(actual, expected)


def test_scope_rejects_compiled_wrapper_and_in_place_compiled_module() -> None:
    def callback(_layer: int, residual: t.Tensor) -> t.Tensor:
        return residual

    wrapped_model = build_tiny_model(n_layer=1, n_embd=64).eval()
    wrapped_model.init_weights()
    compiled = t.compile(wrapped_model, backend="eager", dynamic=False)
    with pytest.raises(RuntimeError, match="exact raw eager model"):
        with scoped_post_cross_attention_callback(compiled, callback):
            pass

    in_place_model = build_tiny_model(n_layer=1, n_embd=64).eval()
    in_place_model.init_weights()
    in_place_model.compile(backend="eager", dynamic=False)
    with pytest.raises(RuntimeError, match="exact raw eager model"):
        with scoped_post_cross_attention_callback(in_place_model, callback):
            pass


def test_active_counter_affects_fp_only_and_preserves_prefill_decode_equivalence() -> None:
    model = build_model_with_feature_prediction(
        n_layer=3,
        n_embd=64,
        xattn_frequency=1,
    ).eval()
    model.init_weights()
    idx, feature_ids, magnitudes, mask, seg_map = _inputs(model)
    direction = t.linspace(-1.0, 1.0, model.dec_config.n_embd)

    def run(
        tokens: t.Tensor,
        assignments: t.Tensor,
        *,
        kv_cache: KVCache | None = None,
        return_fp_logits: bool = False,
        fp_logits_only: bool = False,
    ):
        return model(
            tokens,
            None,
            enc_feature_ids=feature_ids,
            enc_magnitudes=magnitudes,
            enc_mask=mask,
            seg_map=assignments,
            encoder_strict=True,
            kv_cache=kv_cache,
            return_fp_logits=return_fp_logits,
            fp_logits_only=fp_logits_only,
        )

    with t.no_grad():
        default_logits, default_fp = run(idx, seg_map, return_fp_logits=True)
        with post_cross_attention_residual_counter(
            model,
            layer=1,
            direction=direction,
            magnitude=0.25,
        ):
            active_logits, active_fp = run(idx, seg_map, return_fp_logits=True)
            active_fp_only = run(idx, seg_map, fp_logits_only=True)
            cache = _cache(model)
            run(idx[:, :-1], seg_map[:, :-1], kv_cache=cache)
            cached_logits, cached_fp = run(
                idx[:, -1:],
                seg_map[:, -1:],
                kv_cache=cache,
                return_fp_logits=True,
            )

    assert not t.equal(active_logits, default_logits)
    assert not t.equal(active_fp, default_fp)
    assert t.equal(active_fp_only, active_fp)
    t.testing.assert_close(cached_logits, active_logits[:, -1:], rtol=1e-5, atol=1e-6)
    t.testing.assert_close(cached_fp, active_fp[:, -1:], rtol=1e-5, atol=1e-6)
    assert model._post_cross_attention_eval_callback is None


@pytest.mark.parametrize(
    ("layer", "direction", "magnitude", "match"),
    [
        (-1, t.ones(64), 1.0, "nonnegative"),
        (3, t.ones(64), 1.0, "outside"),
        (1, t.ones(63), 1.0, "width"),
        (1, t.tensor([float("nan"), *([0.0] * 63)]), 1.0, "finite"),
        (1, t.ones(64), True, "finite"),
        (1, t.ones(64), float("inf"), "finite"),
        (1, t.ones(64), -1.0, "finite"),
    ],
)
def test_counter_preflight_rejects_invalid_inputs(
    layer: int,
    direction: t.Tensor,
    magnitude: float,
    match: str,
) -> None:
    model = build_tiny_model(n_layer=3, n_embd=64).eval()
    with pytest.raises(ValueError, match=match):
        with post_cross_attention_residual_counter(
            model,
            layer=layer,
            direction=direction,
            magnitude=magnitude,
        ):
            pass


def test_scope_rejects_training_and_nesting_and_cleans_up_exceptions() -> None:
    model = build_tiny_model(n_layer=3, n_embd=64)
    direction = t.ones(64)
    with pytest.raises(RuntimeError, match="eval"):
        with post_cross_attention_residual_counter(
            model, layer=1, direction=direction, magnitude=1.0
        ):
            pass
    model.eval()
    with pytest.raises(RuntimeError, match="already installed"):
        with post_cross_attention_residual_counter(
            model, layer=1, direction=direction, magnitude=1.0
        ):
            with post_cross_attention_residual_counter(
                model, layer=1, direction=direction, magnitude=1.0
            ):
                pass
    assert model._post_cross_attention_eval_callback is None
    with pytest.raises(RuntimeError, match="generation failed"):
        with post_cross_attention_residual_counter(
            model, layer=1, direction=direction, magnitude=1.0
        ):
            raise RuntimeError("generation failed")
    assert model._post_cross_attention_eval_callback is None
    with t.no_grad():
        _forward(model)


def test_counter_preflight_rejects_finite_offset_overflow() -> None:
    model = build_tiny_model(n_layer=3, n_embd=64).eval()
    direction = t.full((64,), float(t.finfo(COMPUTE_DTYPE).max) / 4.0)
    with pytest.raises(ValueError, match="magnitude times direction"):
        with post_cross_attention_residual_counter(
            model,
            layer=1,
            direction=direction,
            magnitude=8.0,
        ):
            pass
