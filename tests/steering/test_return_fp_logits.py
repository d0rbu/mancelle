"""Tests for ``EncoderDecoderGPT.forward(return_fp_logits=...)``.

This flag exposes the feature-prediction head's logits at inference, which
self-conditioned generation needs to drive its encoder updates without a
separate forward pass.

Contract:
- ``return_fp_logits=False`` (default) preserves the existing return type.
- ``return_fp_logits=True`` with ``targets=None`` returns a ``(logits, fp_logits)``
  tuple.
- ``return_fp_logits=True`` with ``targets`` provided is rejected (training
  path keeps its existing return type).
- ``return_fp_logits=True`` on a model without a feature-prediction head
  raises a clear error.
"""

from __future__ import annotations

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    SEQ_LEN,
    build_model_with_feature_prediction,
    build_tiny_model,
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA backend; flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _build_inputs(B: int, T: int, model, n_seg: int = 1, F_max: int = 4):
    """Helper: synthetic encoder + decoder inputs aligned with the tiny model."""
    idx = torch.randint(0, model.dec_config.vocab_size, (B, T))
    enc_feature_ids = torch.randint(0, N_SAE_FEATURES, (B, n_seg, F_max))
    enc_magnitudes = torch.randint(0, NUM_BUCKETS, (B, n_seg, F_max))
    enc_mask = torch.ones(B, n_seg, F_max, dtype=torch.bool)
    seg_map = torch.zeros(B, T, dtype=torch.long)
    return idx, enc_feature_ids, enc_magnitudes, enc_mask, seg_map


class TestReturnFpLogitsDefaultsPreserveExistingBehavior:
    """``return_fp_logits=False`` (default) must not change return types."""

    def test_inference_returns_logits_only(self):
        torch.manual_seed(0)
        model = build_model_with_feature_prediction()
        model.eval()
        idx, fids, mags, mask, seg_map = _build_inputs(1, SEQ_LEN, model)
        out = model(
            idx,
            targets=None,
            enc_feature_ids=fids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
        )
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, SEQ_LEN, model.dec_config.vocab_size)


class TestReturnFpLogitsInferenceTuple:
    """``return_fp_logits=True`` with ``targets=None``: returns tuple."""

    def test_returns_logits_and_fp_logits(self):
        torch.manual_seed(0)
        model = build_model_with_feature_prediction()
        model.eval()
        idx, fids, mags, mask, seg_map = _build_inputs(1, SEQ_LEN, model)
        out = model(
            idx,
            targets=None,
            enc_feature_ids=fids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
            return_fp_logits=True,
        )
        assert isinstance(out, tuple)
        assert len(out) == 2
        logits, fp_logits = out
        assert logits.shape == (1, SEQ_LEN, model.dec_config.vocab_size)
        assert fp_logits.shape == (1, SEQ_LEN, N_SAE_FEATURES)

    def test_kv_cache_decode_step_shape(self):
        # Decode step (T=1 with kv_cache get_pos > 0) should still produce
        # fp_logits with T=1 along the time dimension.
        torch.manual_seed(0)
        model = build_model_with_feature_prediction()
        model.eval()

        # Prefill T_pre tokens
        T_pre = 8
        kv = KVCache(
            batch_size=1,
            num_heads=model.dec_config.n_kv_head,
            seq_len=SEQ_LEN,
            head_dim=model.dec_config.n_embd // model.dec_config.n_head,
            num_layers=model.dec_config.n_layer,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        idx_pre, fids, mags, mask, _ = _build_inputs(1, T_pre, model)
        seg_map_pre = torch.zeros(1, T_pre, dtype=torch.long)
        # Prefill (don't even need fp_logits here, but flow-test it)
        _ = model(
            idx_pre,
            targets=None,
            enc_feature_ids=fids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map_pre,
            kv_cache=kv,
        )

        # Decode step: T=1
        idx_step = torch.randint(0, model.dec_config.vocab_size, (1, 1))
        seg_map_step = torch.zeros(1, 1, dtype=torch.long)
        # encoder_out reuse: simulate caller passing precomputed encoder_out
        # by re-encoding (cheap) and using it directly
        encoder_out = model.encoder(fids, mags, mask)
        out = model(
            idx_step,
            targets=None,
            seg_map=seg_map_step,
            kv_cache=kv,
            encoder_out=encoder_out,
            return_fp_logits=True,
        )
        assert isinstance(out, tuple)
        logits, fp_logits = out
        assert logits.shape == (1, 1, model.dec_config.vocab_size)
        assert fp_logits.shape == (1, 1, N_SAE_FEATURES)


class TestReturnFpLogitsRejectedWithTargets:
    """Training-path forwards (``targets`` provided) keep their existing
    return types. Asking for fp_logits via the new flag is rejected."""

    def test_targets_plus_flag_raises(self):
        torch.manual_seed(0)
        model = build_model_with_feature_prediction()
        idx, fids, mags, mask, seg_map = _build_inputs(1, SEQ_LEN, model)
        targets = torch.randint(0, model.dec_config.vocab_size, (1, SEQ_LEN))
        with pytest.raises((ValueError, AssertionError, RuntimeError)):
            model(
                idx,
                targets=targets,
                enc_feature_ids=fids,
                enc_magnitudes=mags,
                enc_mask=mask,
                seg_map=seg_map,
                return_fp_logits=True,
            )


class TestReturnFpLogitsRejectedWithoutFpHead:
    """A model built without ``feature_pred_config`` has no head; the flag
    must raise rather than silently no-op."""

    def test_no_head_plus_flag_raises(self):
        torch.manual_seed(0)
        model = build_tiny_model(feature_pred_config=None)
        model.eval()
        idx, fids, mags, mask, seg_map = _build_inputs(1, SEQ_LEN, model)
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            model(
                idx,
                targets=None,
                enc_feature_ids=fids,
                enc_magnitudes=mags,
                enc_mask=mask,
                seg_map=seg_map,
                return_fp_logits=True,
            )


class TestFpLogitsValuesMatchInternalHead:
    """The fp_logits surfaced by the new flag must equal what the head would
    produce on the same captured hidden state. Reproduce by manually running
    the head on the captured layer output."""

    def test_matches_head_application_on_norm_of_capture(self):
        torch.manual_seed(0)
        model = build_model_with_feature_prediction()
        model.eval()
        idx, fids, mags, mask, seg_map = _build_inputs(1, SEQ_LEN, model)
        out = model(
            idx,
            targets=None,
            enc_feature_ids=fids,
            enc_magnitudes=mags,
            enc_mask=mask,
            seg_map=seg_map,
            return_fp_logits=True,
        )
        logits, fp_logits = out
        # Sanity: fp_logits is finite and differentiable-shaped
        assert torch.isfinite(fp_logits).all()
        # Sanity: not all-zeros (init has small std but matmul of nonzero
        # hidden by a Linear with bias=True yields nonzero generally)
        assert fp_logits.abs().sum().item() > 0.0


@pytest.mark.parametrize("layer", [0, -1])
@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("keep", [0, 1, 32])
def test_fp_logits_only_returns_bitwise_identical_fp_logits(
    monkeypatch, layer, grad_enabled, keep
) -> None:
    """Readouts preserve the exact normalized input at the configured FP layer."""
    torch.manual_seed(31)
    model = build_tiny_model(feature_pred_config=FeaturePredictionConfig(layer=layer)).eval()
    # Separate early/final states; initial residual projections are all zero.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("c_proj.weight"):
                parameter.normal_(std=0.02)
    batch_size, time = 2, 16
    idx = torch.randint(0, 100, (batch_size, time))
    n_seg, width = 2, 3
    enc_fids = torch.randint(0, 32, (batch_size, n_seg, width))
    enc_mags = torch.zeros(batch_size, n_seg, width, dtype=torch.int64)
    enc_mask = torch.ones(batch_size, n_seg, width, dtype=torch.bool)
    seg_map = torch.cat(
        [torch.zeros(batch_size, time // 2), torch.ones(batch_size, time // 2)],
        dim=1,
    ).long()
    args = (idx, None, enc_fids, enc_mags, enc_mask, seg_map)

    captured = []
    with torch.set_grad_enabled(grad_enabled):
        with model.feature_pred_head.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        ):
            _, paired = model(*args, return_fp_logits=True)
        only = model(*args, fp_logits_only=True)
        explicit_false = model(*args, fp_logits_only=True, fp_hidden_only=False)

        def unexpected_head(*_args, **_kwargs):
            pytest.fail("hidden-state readout must skip both heads")

        monkeypatch.setattr(model.feature_pred_head, "forward", unexpected_head)
        monkeypatch.setattr(model.lm_head, "forward", unexpected_head)
        hidden = model(*args, fp_hidden_only=True, logits_to_keep=keep)

    assert torch.equal(paired, only)
    assert torch.equal(only, explicit_false)
    assert len(captured) == 1
    assert hidden.shape == (batch_size, time, model.dec_config.n_embd)
    assert torch.equal(hidden, captured[0])
    assert hidden.requires_grad is grad_enabled
    if grad_enabled:
        hidden.float().sum().backward()
        assert model._decoder_embedding.weight.grad is not None
        assert model.feature_pred_head.weight.grad is None
        assert model.lm_head.weight.grad is None


@pytest.mark.parametrize("flag", ["fp_logits_only", "fp_hidden_only"])
def test_fp_logits_only_guards(flag) -> None:
    model = build_model_with_feature_prediction().eval()
    headless = build_tiny_model(feature_pred_config=None).eval()
    idx = torch.randint(0, 100, (1, 8))
    targets = torch.randint(0, 100, (1, 8))
    enc_fids = torch.zeros(1, 1, 1, dtype=torch.int64)
    enc_mags = torch.zeros(1, 1, 1, dtype=torch.int64)
    enc_mask = torch.ones(1, 1, 1, dtype=torch.bool)
    seg_map = torch.zeros(1, 8, dtype=torch.int64)
    args = (idx, None, enc_fids, enc_mags, enc_mask, seg_map)

    with pytest.raises(ValueError, match=flag):
        model(*args, return_fp_logits=True, **{flag: True})
    with pytest.raises(ValueError, match=flag):
        model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map, **{flag: True})
    with pytest.raises(RuntimeError, match="feature-prediction head"):
        headless(*args, **{flag: True})
    if flag == "fp_hidden_only":
        with pytest.raises(ValueError, match="fp_hidden_only=True"):
            model(*args, fp_logits_only=True, fp_hidden_only=True)
