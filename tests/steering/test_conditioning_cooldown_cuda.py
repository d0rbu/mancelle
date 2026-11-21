"""Real CUDA empty/mixed conditioning through packed attention and optimizer."""

from __future__ import annotations

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from nanochat.fp8_utils import apply_fp8_training
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.conditioning_cooldown import apply_conditioning_cooldown
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer
from steering.conditional.encoder_decoder.encoder import FeatureEncoder
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from tests.padded_encoder_reference import (
    production_cross_attention_forward,
    production_feature_encoder_forward,
    production_feature_encoder_forward_packed,
)
from tests.steering.selfcond.helpers import make_batch


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("granularity", ["segment", "document"])
@pytest.mark.parametrize("fraction", [0.5, 0.125])
def test_cuda_junction_priority_and_document_atomicity(granularity: str, fraction: float) -> None:
    batch = make_batch([[2, 2], [1, 3]], T=64, F=16, n_sae=64, seed=11)
    mask = batch.enc_mask.cuda()
    documents = batch.seg_doc_id.cuda()
    junctions = t.tensor([[False, False, True, False], [False, True, False, False]], device="cuda")
    before = mask.clone()
    cpu_rng, cuda_rng = t.get_rng_state().clone(), t.cuda.get_rng_state().clone()
    actual = apply_conditioning_cooldown(
        mask,
        documents,
        fraction=fraction,
        granularity=granularity,
        seed=17,
        step=5,
        rank=0,
        micro_step=1,
        splice_mask=junctions,
    )
    assert t.equal(mask, before)
    assert t.equal(t.get_rng_state(), cpu_rng)
    assert t.equal(t.cuda.get_rng_state(), cuda_rng)
    protected = junctions.clone()
    if granularity == "document":
        protected = documents == 1
        for row in range(2):
            for document in range(2):
                decisions = actual[row, documents[row] == document].any(dim=-1)
                assert t.equal(decisions, decisions[:1].expand_as(decisions))
    if fraction == 0.5:
        assert t.equal(actual[protected], before[protected])
    else:
        assert not actual[~protected].any()


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("compile_model", [False, True])
def test_real_cuda_empty_then_conditioned_updates_have_finite_all_parameter_gradients(
    monkeypatch: pytest.MonkeyPatch, fp8: bool, compile_model: bool
) -> None:
    if fp8 and t.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 requires Ada or newer hardware")
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    # Match build_shared_state: Muon's distinct parameter groups legitimately
    # need more specializations than Dynamo's default eight-entry budget.
    monkeypatch.setattr(t._dynamo.config, "cache_size_limit", 64)
    # Explicitly bypass the test-owned CPU oracle patched by conftest. Every
    # encoder and cross-attention call below uses the production CUDA methods.
    monkeypatch.setattr(FeatureEncoder, "forward", production_feature_encoder_forward)
    monkeypatch.setattr(FeatureEncoder, "forward_packed", production_feature_encoder_forward_packed)
    monkeypatch.setattr(CrossAttentionLayer, "forward", production_cross_attention_forward)
    t.manual_seed(331)
    model = EncoderDecoderGPT(
        GPTConfig(
            sequence_len=64,
            vocab_size=128,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            n_embd=256,
            window_pattern="L",
        ),
        EncoderConfig(n_embd=128, depth=1, n_head=2, n_kv_head=2, head_dim=64),
        CrossAttentionConfig(frequency=1, n_head=4, n_kv_head=2),
        n_sae_features=64,
        num_buckets=2,
    ).cuda()
    model.init_weights()
    if fp8:
        converted, _ = apply_fp8_training(model, recipe="tensorwise")
        assert converted > 0
        assert any("Float8" in type(module).__name__ for module in model.encoder.modules())
        assert any(
            "Float8" in type(module).__name__ for module in model.cross_attn_layers.modules()
        )
    optimizer = model.setup_optimizer(
        embedding_lr=1e-3,
        unembedding_lr=1e-3,
        matrix_lr=1e-3,
        scalar_lr=1e-3,
        encoder_embedding_lr=1e-3,
        encoder_matrix_lr=1e-3,
        xattn_matrix_lr=1e-3,
    )
    forward_model = t.compile(model, dynamic=False) if compile_model else model
    batch = make_batch([[2, 2], [1, 3]], T=64, F=16, n_sae=64, seed=11)
    x, y, ids, mags, source_mask, routing, documents = [
        value.cuda()
        for value in (
            batch.inputs,
            batch.targets,
            batch.enc_fids,
            batch.enc_mags,
            batch.enc_mask,
            batch.seg_map,
            batch.seg_doc_id,
        )
    ]
    pristine = source_mask.clone()
    for step, fraction in enumerate((0.0, 0.5, 1.0, 0.0)):
        optimizer.zero_grad(set_to_none=True)
        global_rng = t.cuda.get_rng_state().clone()
        mask = apply_conditioning_cooldown(
            source_mask,
            documents,
            fraction=fraction,
            granularity="segment",
            seed=17,
            step=step,
            rank=0,
            micro_step=0,
        )
        assert t.equal(t.cuda.get_rng_state(), global_rng)
        assert t.equal(source_mask, pristine)
        if fraction == 0.0:
            assert not mask.any()
        elif fraction == 0.5:
            assert mask.any() and not t.equal(mask, source_mask)
        loss = forward_model(x, y, ids, mags, mask, routing, seg_doc_id=documents)
        assert t.isfinite(loss), (fp8, compile_model, step)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, (step, name)
            assert t.isfinite(parameter.grad).all(), (fp8, compile_model, step, name)
            if fraction == 0.0 and name.startswith(("encoder.", "cross_attn_layers.")):
                assert not parameter.grad.count_nonzero(), (step, name)
        optimizer.step()
        assert all(t.isfinite(parameter).all() for parameter in model.parameters())
    t.cuda.synchronize()
