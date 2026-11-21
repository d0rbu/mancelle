"""LM-head slicing must preserve decoder, feature-head, and cache semantics."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.eval_residual import (
    scoped_post_cross_attention_callback,
)
from steering.conditional.feature_vocab import FeatureVocab
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    build_model_with_feature_prediction,
    build_tiny_model,
)
from tests.steering.test_return_fp_logits import _build_inputs


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)


@pytest.mark.parametrize("mode", ["lm", "paired", "fp_only", "paired_compact", "fp_only_compact"])
@pytest.mark.parametrize("keep", [0, 1, 3, 16])
def test_head_slicing_preserves_full_decoder_and_feature_logits(mode: str, keep: int) -> None:
    t.manual_seed(173)
    vocab = FeatureVocab(N_SAE_FEATURES, (1, 3, 7)) if mode.endswith("_compact") else None
    mode = mode.removesuffix("_compact")
    model = (
        build_tiny_model(feature_pred_config=None)
        if mode == "lm"
        else build_model_with_feature_prediction(feature_vocab=vocab)
    ).eval()
    idx, fids, mags, mask, seg_map = _build_inputs(2, 8, model)
    if vocab is not None:
        fids = t.tensor(vocab.original_ids)[fids % vocab.size]
    flags = {"return_fp_logits": mode == "paired", "fp_logits_only": mode == "fp_only"}
    captured: list[t.Tensor] = []

    def capture(_layer: int, hidden: t.Tensor) -> t.Tensor:
        captured.append(hidden.detach().clone())
        return hidden

    with t.no_grad(), scoped_post_cross_attention_callback(model, capture):
        reference = model(idx, None, fids, mags, mask, seg_map, **flags)
        expected_hidden = captured[:]
        captured.clear()
        with patch.object(model.lm_head, "forward", wraps=model.lm_head.forward) as head:
            actual = model(idx, None, fids, mags, mask, seg_map, logits_to_keep=keep, **flags)
    assert len(captured) == len(expected_hidden) == model.dec_config.n_layer
    for hidden, expected in zip(captured, expected_hidden, strict=True):
        assert t.equal(hidden, expected)
    if mode == "fp_only":
        assert head.call_count == 0
        assert actual.shape[-1] == (vocab.size if vocab else N_SAE_FEATURES)
        assert t.equal(actual, reference)
        return
    assert head.call_count == 1
    assert head.call_args.args[0].shape[1] == (min(keep, 8) if keep else 8)
    if mode == "paired":
        actual, actual_fp = actual
        reference, reference_fp = reference
        assert actual_fp.shape[1] == 8
        assert actual_fp.shape[-1] == (vocab.size if vocab else N_SAE_FEATURES)
        assert t.equal(actual_fp, reference_fp)
    expected = reference[:, -keep:] if keep else reference
    t.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("keep", [1, 3])
def test_kept_logits_preserve_input_gradients(keep: int) -> None:
    t.manual_seed(27)
    model = build_tiny_model(feature_pred_config=None).eval()
    idx, fids, mags, mask, seg_map = _build_inputs(2, 8, model)
    root = model.transformer["wte"].weight
    reference = model(idx, None, fids, mags, mask, seg_map)
    [expected] = t.autograd.grad(reference[:, -keep:].square().sum(), root)
    actual = model(idx, None, fids, mags, mask, seg_map, logits_to_keep=keep)
    [gradient] = t.autograd.grad(actual.square().sum(), root)
    assert bool(expected.ne(0).any())
    t.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("keep", [-1, 1.5, True, None, "1"])
def test_invalid_logit_count_fails_before_model_work(keep: Any) -> None:
    model = build_tiny_model(feature_pred_config=None)
    with (
        patch.object(model.encoder, "forward_packed") as encoder,
        pytest.raises(ValueError, match="nonnegative integer"),
    ):
        model(t.zeros(1, 4, dtype=t.long), None, logits_to_keep=keep)
    encoder.assert_not_called()


def test_training_requires_all_logits_and_default_loss_is_unchanged() -> None:
    t.manual_seed(71)
    model = build_tiny_model(feature_pred_config=None)
    idx, fids, mags, mask, seg_map = _build_inputs(2, 8, model)
    targets = idx.roll(-1, dims=1)
    with patch.object(model.encoder, "forward_packed") as encoder:
        with pytest.raises(ValueError, match="requires targets=None"):
            model(idx, targets, fids, mags, mask, seg_map, logits_to_keep=1)
        encoder.assert_not_called()
    expected = model(idx, targets, fids, mags, mask, seg_map)
    actual = model(idx, targets, fids, mags, mask, seg_map, logits_to_keep=0)
    assert t.equal(actual, expected)


def test_reduced_prefill_still_populates_the_entire_kv_cache() -> None:
    t.manual_seed(116)
    model = build_tiny_model(feature_pred_config=None).eval()
    idx, fids, mags, mask, seg_map = _build_inputs(2, 8, model)
    caches = [
        KVCache(
            batch_size=2,
            num_heads=model.dec_config.n_kv_head,
            seq_len=16,
            head_dim=model.dec_config.n_embd // model.dec_config.n_head,
            num_layers=model.dec_config.n_layer,
            device=t.device("cpu"),
            dtype=t.float32,
        )
        for _ in range(2)
    ]
    with t.no_grad():
        packed = model.encoder.forward_packed(fids, mags, mask)
        outputs = [
            model(
                idx,
                None,
                seg_map=seg_map,
                encoder_out=packed,
                kv_cache=cache,
                logits_to_keep=keep,
            )
            for cache, keep in zip(caches, (0, 1), strict=True)
        ]
        t.testing.assert_close(outputs[1], outputs[0][:, -1:], rtol=1e-5, atol=1e-6)
        assert caches[0].get_pos() == caches[1].get_pos() == 8
        assert t.equal(caches[0].k_cache, caches[1].k_cache)
        assert t.equal(caches[0].v_cache, caches[1].v_cache)
        outputs = [
            model(
                idx[:, :1],
                None,
                seg_map=seg_map[:, :1],
                encoder_out=packed,
                kv_cache=cache,
                logits_to_keep=keep,
            )
            for cache, keep in zip(caches, (0, 1), strict=True)
        ]
        assert t.equal(outputs[0], outputs[1])
        assert caches[0].get_pos() == caches[1].get_pos() == 9
        assert t.equal(caches[0].k_cache, caches[1].k_cache)
        assert t.equal(caches[0].v_cache, caches[1].v_cache)
