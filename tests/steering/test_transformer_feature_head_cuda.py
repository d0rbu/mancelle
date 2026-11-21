"""Real CUDA decoder/head and packed SFT qualification, with explicit dispatch."""

from __future__ import annotations

import copy

import pytest
import torch as t

from nanochat import flash_attention
from nanochat.engine import KVCache
from nanochat.gpt import Block, GPTConfig
from steering.conditional.encoder_decoder import sm12_fa4
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from tests.steering.test_transformer_feature_head import _head, _rotary

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")]


@pytest.fixture(params=["sdpa", "fa4"])
def backend(request, monkeypatch):
    backend = request.param
    if backend == "fa4" and t.cuda.get_device_capability()[0] < 9:
        pytest.skip("FA4 requires Hopper or newer; Ada is SDPA reference only")
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl(backend))
    monkeypatch.setattr(t._dynamo.config, "cache_size_limit", 32)
    t._dynamo.reset()
    monkeypatch.setattr(flash_attention.flash_attn, "flash_attn_func", flash_attention.flash_attn.flash_attn_func)
    deterministic = t.are_deterministic_algorithms_enabled()
    t.use_deterministic_algorithms(False)
    cached_dispatch = flash_attention.flash_attn.flash_attn_with_kvcache
    sm12_fa4.admit_project_attention()
    if backend == "fa4":
        if t.cuda.get_device_capability()[0] == 12:
            assert sm12_fa4.uses_native_sm12_fa4() and sm12_fa4._native_fa4 is not None
        else:
            assert flash_attention.has_fa4_backend()
            assert flash_attention._use_flex()
    assert flash_attention.flash_attn.flash_attn_with_kvcache is cached_dispatch
    yield backend
    assert flash_attention.flash_attn.flash_attn_with_kvcache is cached_dispatch
    t.use_deterministic_algorithms(deterministic)


def _model(depth):
    with t.device("cuda"):
        model = EncoderDecoderGPT(
            GPTConfig(sequence_len=128, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=256),
            EncoderConfig(n_embd=128, depth=1, n_head=2, n_kv_head=2, head_dim=64),
            CrossAttentionConfig(frequency=1, n_head=2, n_kv_head=2),
            n_sae_features=32, num_buckets=2,
            feature_pred_config=FeaturePredictionConfig(block=FeaturePredictionBlock.TRANSFORMER, head_layers=depth),
        )
        model.init_weights()
    return model


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("depth", [1, 2, 3, 10])
@pytest.mark.parametrize("autocast_enabled", [False, True], ids=["explicit_dtype", "autocast"])
def test_cuda_packed_sft_identity_gradients_update_and_strict_reload(backend, compiled, depth, autocast_enabled):
    t.manual_seed(1719)
    baseline = _model(0)
    cpu_rng, cuda_rng = t.get_rng_state(), t.cuda.get_rng_state()
    t.manual_seed(1719)
    model = _model(depth)
    assert t.equal(cpu_rng, t.get_rng_state()) and t.equal(cuda_rng, t.cuda.get_rng_state())
    baseline.prewarm_decoder_block_masks(2, 128)
    model.prewarm_decoder_block_masks(2, 128)
    batch = {
        "idx": t.randint(0, 128, (2, 128), device="cuda"),
        "targets": t.randint(0, 128, (2, 128), device="cuda"),
        "enc_feature_ids": t.randint(0, 32, (2, 4, 3), device="cuda"),
        "enc_magnitudes": t.randint(0, 2, (2, 4, 3), device="cuda"),
        "enc_mask": t.ones(2, 4, 3, dtype=t.bool, device="cuda"),
        "seg_map": t.arange(4, device="cuda").repeat_interleave(32).expand(2, -1),
        "seg_doc_id": t.zeros(2, 4, dtype=t.long, device="cuda"),
        "fp_targets": t.rand(2, 4, 32, device="cuda"),
        "static_max_seg_size": 32,
    }
    reference = t.compile(baseline, dynamic=False) if compiled else baseline
    candidate = t.compile(model, dynamic=False) if compiled else model
    inference = {**batch, "targets": None}
    with t.no_grad(), t.amp.autocast("cuda", dtype=t.bfloat16, enabled=autocast_enabled):
        lm, fp = candidate(**inference, return_fp_logits=True)
        expected_lm, expected_fp = reference(**inference, return_fp_logits=True)
        assert t.equal(lm, expected_lm) and t.equal(fp, expected_fp)
    optimizer = model.setup_optimizer()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Match SFT: autocast covers each forward, and exits before backward
        # and optimizer updates so cached casts cannot outlive parameter writes.
        with t.amp.autocast("cuda", dtype=t.bfloat16, enabled=autocast_enabled):
            result = candidate(**batch)
        assert t.isfinite(result.loss)
        result.loss.backward()
        if step == 0 and backend == "sdpa":
            with t.amp.autocast("cuda", dtype=t.bfloat16, enabled=autocast_enabled):
                reference_result = reference(**batch)
            assert t.equal(result.loss, reference_result.loss)
            reference_result.loss.backward()
            parameters = dict(model.named_parameters())
            for name, parameter in baseline.named_parameters():
                candidate_gradient = parameters[name].grad
                if parameter.grad is None:
                    assert candidate_gradient is None, name
                else:
                    assert t.equal(parameter.grad, candidate_gradient), name
        for name, parameter in model.feature_pred_head.named_parameters():
            assert parameter.grad is not None and t.isfinite(parameter.grad).all(), name
            if step or "c_proj" in name:
                assert t.count_nonzero(parameter.grad), name
        optimizer.step()
    restored = _model(depth)
    restored.load_state_dict(model.state_dict(), strict=True, assign=True)
    restored.assert_vocab_consistent()
    restored.prewarm_decoder_block_masks(2, 128)
    reloaded = t.compile(restored, dynamic=False) if compiled else restored
    with t.no_grad(), t.amp.autocast("cuda", dtype=t.bfloat16, enabled=autocast_enabled):
        assert t.equal(candidate(**inference, fp_logits_only=True), reloaded(**inference, fp_logits_only=True))
    if backend == "fa4" and t.cuda.get_device_capability()[0] != 12:
        assert flash_attention.has_fa4_backend(), "production requested FA4 but dispatched a fallback"


@pytest.mark.parametrize("heads,kv_heads,length,window", [(4, 2, 129, 31), (13, 13, 129, 64)])
def test_cuda_learned_head_production_heads_and_gqa_match_sdpa(backend, monkeypatch, heads, kv_heads, length, window):
    t.manual_seed(1720)
    # Training retains FP32 parameters with BF16 activations. Keep the learned
    # residual RMS scale independent of width, including the 1664-wide case.
    model_head = _head(1, heads=heads, kv_heads=kv_heads, dim=128, device="cuda", dtype=t.float32)
    with t.no_grad():
        for block in model_head.residual_blocks:
            assert isinstance(block, Block)
            for projection in (block.attn.c_proj, block.mlp.c_proj):
                projection.weight.normal_(0, 0.15 / projection.in_features**0.5)
    oracle = copy.deepcopy(model_head)
    x = t.randn(1, length, heads * 128, device="cuda", dtype=t.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    cos_sin = _rotary(length, 128, start=17, device="cuda", dtype=t.bfloat16)
    actual = model_head(x, cos_sin=cos_sin, window_size=(window, 0))
    with monkeypatch.context() as reference_backend:
        reference_backend.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.SDPA)
        expected = oracle(other, cos_sin=cos_sin, window_size=(window, 0))
    direction = t.randn_like(actual)
    # SFT averages its loss over tokens; use the same reduction so the absolute
    # gradient tolerance is independent of sequence length and output count.
    (actual * direction).mean().backward()
    (expected * direction).mean().backward()
    # Compare the complete BF16 block under matched arithmetic; only the
    # attention kernel differs. CPU tests independently pin the FP64 algebra.
    for observed, target in zip((actual, x.grad, *(p.grad for p in model_head.parameters())),
                               (expected, other.grad, *(p.grad for p in oracle.parameters())), strict=True):
        assert observed is not None and target is not None
        assert t.isfinite(observed).all()
        t.testing.assert_close(observed, target, atol=0.02, rtol=0.02)
        assert (observed.double() - target.double()).norm() / target.double().norm().clamp_min(1e-12) < 0.01


def test_cuda_cached_generation_dispatch_and_learned_head_history(backend, monkeypatch):
    t.manual_seed(1721)
    model = _model(2).eval()
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if ".c_proj.weight" in name:
                parameter.normal_(0, 0.015)
    model.window_sizes = [(7, 0), (31, 0)]
    model.prewarm_decoder_block_masks(1, 128)
    batch = {
        "idx": t.randint(0, 128, (1, 128), device="cuda"), "targets": None,
        "enc_feature_ids": t.tensor([[[1, 3]]], device="cuda"),
        "enc_magnitudes": t.tensor([[[0, 1]]], device="cuda"),
        "enc_mask": t.ones(1, 1, 2, dtype=t.bool, device="cuda"),
        "seg_map": t.zeros(1, 128, dtype=t.long, device="cuda"),
        "static_max_seg_size": 128,
    }
    dispatched = []
    implementation = flash_attention._fa3 if flash_attention.HAS_FA3 and backend != "sdpa" else flash_attention
    name = "flash_attn_with_kvcache" if implementation is flash_attention._fa3 else "_sdpa_attention"
    original = getattr(implementation, name)

    def observed(*args, **kwargs):
        dispatched.append(name)
        return original(*args, **kwargs)

    cache = KVCache(batch_size=1, num_heads=2, seq_len=128, head_dim=128,
                    num_layers=model.num_cache_layers, dtype=t.bfloat16, device="cuda")
    with t.no_grad():
        expected = model(**batch, fp_logits_only=True)
        monkeypatch.setattr(implementation, name, observed)
        pieces = []
        start = 0
        for count in (31, 1, 33, 63):
            chunk = {**batch, "idx": batch["idx"][:, start:start+count], "seg_map": batch["seg_map"][:, start:start+count]}
            pieces.append(model(**chunk, kv_cache=cache, fp_logits_only=True))
            start += count
            assert cache.get_pos() == start
        actual = t.cat(pieces, 1)
    assert dispatched == [name] * (4 * model.num_cache_layers)
    t.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1.0) < 0.01
