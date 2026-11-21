"""SFT validation must honor autocast at the native FA4 boundary on SM12."""

import math

import pytest
import torch as t
from torch._subclasses.fake_tensor import FakeTensorMode

from nanochat import flash_attention, gpt
from scripts.train.conditional_encdec_sft import (
    SFTState,
    _SFTValidationLoader,
    evaluate_encdec_bpb,
)
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder import sm12_fa4 as adapter
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    pack_encoder_layout,
)

_BATCH = 2
_LENGTH = 8
_LAYERS = 2
_WIDTH = 64
_VOCAB = 64
_SM12 = t.cuda.is_available() and t.cuda.get_device_capability()[0] == 12


@pytest.fixture
def native_dispatch(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(
        flash_attention.flash_attn, "flash_attn_func", adapter._full_prefix_attention
    )
    monkeypatch.setattr(adapter, "_native_fa4", adapter._native_fa4)


def _eval_case(monkeypatch, device, dtype):
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", dtype)
    with t.device("cpu"):
        model = EncoderDecoderGPT(
            gpt.GPTConfig(
                sequence_len=_LENGTH,
                vocab_size=_VOCAB,
                n_layer=_LAYERS,
                n_head=2,
                n_kv_head=1,
                n_embd=_WIDTH,
                window_pattern="L",
            ),
            EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16),
            # Isolate decoder attention; the failure precedes cross-attention.
            CrossAttentionConfig(frequency=_LAYERS + 1),
            n_sae_features=8,
            num_buckets=2,
            feature_pred_config=FeaturePredictionConfig(
                block=FeaturePredictionBlock.TRANSFORMER, head_layers=2
            ),
        )
        model.init_weights()
        # Initialization zeros these branches. Make attention affect the loss.
        with t.no_grad():
            for block in model._decoder_blocks:
                block.attn.c_proj.weight.normal_(std=0.02)
    model.to(device).eval()
    inputs = t.arange(_BATCH * _LENGTH, device=device).reshape(_BATCH, _LENGTH)
    targets = inputs + 1
    feature_ids = t.zeros(_BATCH, 1, 1, device=device, dtype=t.long)
    mask = t.ones_like(feature_ids, dtype=t.bool)
    layout = pack_encoder_layout(mask)
    state = PackedEncoderOutput(
        t.zeros(layout.packed_count, model.enc_dim, device=device, dtype=dtype), layout
    )
    # Supply a valid precomputed encoder result; all decoder projections,
    # value embedding gates, RoPE, Q/K norms and loss arithmetic remain real.
    monkeypatch.setattr(model.encoder, "forward_packed", lambda *_args: state)
    batch = (
        inputs,
        targets,
        feature_ids,
        feature_ids,
        mask,
        t.zeros_like(inputs),
        None,
    )
    loader = _SFTValidationLoader(iter([batch]), SFTState())
    token_bytes = t.ones(_VOCAB, device=device, dtype=t.long)
    return model, loader, token_bytes


def _native_math(q, k, v, *, causal, window_size, deterministic):
    del deterministic
    assert causal and window_size[1] == 0
    query, key, value = (tensor.transpose(1, 2) for tensor in (q, k, v))
    groups = query.shape[1] // key.shape[1]
    key, value = (tensor.repeat_interleave(groups, dim=1) for tensor in (key, value))
    positions = t.arange(query.shape[2], device=q.device)
    distances = positions[:, None] - positions[None, :]
    mask = distances >= 0
    if window_size[0] is not None:
        mask &= distances <= window_size[0]
    scores = (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    probabilities = scores.masked_fill(~mask, -t.inf).softmax(dim=-1)
    return (probabilities @ value).transpose(1, 2), None


@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16])
def test_cuda_autocast_promotes_qk_norm_but_not_projection_on_cpu(monkeypatch, dtype):
    """Use the actual CUDA dispatcher policy without allocating CUDA storage."""
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "is_bf16_supported", lambda: True)
    with FakeTensorMode(), t.no_grad():
        x = t.empty(_BATCH, _LENGTH, _WIDTH, device="cuda", dtype=dtype)
        weight = t.empty(_WIDTH, _WIDTH, device="cuda", dtype=t.float32)
        with t.amp.autocast("cuda", dtype=dtype):
            projected = t.nn.functional.linear(x, weight)
            assert projected.dtype == dtype
            assert gpt.norm(projected).dtype == t.float32
        assert gpt.norm(projected).dtype == dtype


@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16])
def test_sft_bpb_eval_normalizes_real_decoder_inputs_on_cpu(monkeypatch, native_dispatch, dtype):
    model, loader, token_bytes = _eval_case(monkeypatch, "cpu", dtype)
    original_norm = gpt.norm
    # CPU RMSNorm lacks CUDA's FP32 autocast policy (pinned independently above).
    # Reproduce only that operator policy; do not fabricate Q/K/V inputs.
    monkeypatch.setattr(gpt, "norm", lambda x: original_norm(x.float()))
    incoming = []
    received = []

    def cpu_native_route(q, k, v, **kwargs):
        incoming.append((q.dtype, k.dtype, v.dtype))
        return adapter._causal_attention(q, k, v, **kwargs)

    def native(q, k, v, **kwargs):
        received.append((q.dtype, k.dtype, v.dtype))
        assert q.dtype == k.dtype == v.dtype == dtype
        return _native_math(q, k, v, **kwargs)

    # _full_prefix_attention normally sends CPU tensors to the reference
    # callable. Replace that test seam to reach the native adapter on CPU.
    monkeypatch.setattr(flash_attention, "flash_attn_func", cpu_native_route)
    monkeypatch.setattr(adapter, "_native_fa4", native)
    assert adapter.uses_native_sm12_fa4()
    with t.amp.autocast("cpu", dtype=dtype):
        bpb = evaluate_encdec_bpb(model, loader, 1, token_bytes)
    assert math.isfinite(bpb) and bpb > 0
    assert incoming == [(t.float32, t.float32, dtype)] * _LAYERS
    assert received == [(dtype, dtype, dtype)] * _LAYERS


@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16])
@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_native_casts_preserve_values_and_gradients(monkeypatch, dtype, autocast_enabled):
    input_dtypes = (t.float32, t.float32, dtype) if autocast_enabled else (dtype,) * 3
    tensors = [t.randn(1, 2, 4, 8, dtype=kind, requires_grad=True) for kind in input_dtypes]
    received = []

    def native(q, k, v, **_kwargs):
        received.append((q.dtype, k.dtype, v.dtype))
        assert q.dtype == k.dtype == v.dtype == dtype
        for actual, original in zip((q, k, v), tensors, strict=True):
            assert t.equal(actual, original.to(dtype).transpose(1, 2))
            if original.dtype == dtype:
                assert actual.data_ptr() == original.data_ptr()
        return q + 2 * k + 3 * v, None

    monkeypatch.setattr(adapter, "_native_fa4", native)
    with t.amp.autocast("cpu", dtype=dtype, enabled=autocast_enabled):
        result = adapter._causal_attention(*tensors, causal=True)
    assert received == [(dtype,) * 3]
    expected = tensors[0].to(dtype) + 2 * tensors[1].to(dtype) + 3 * tensors[2].to(dtype)
    t.testing.assert_close(result, expected, rtol=0, atol=0)
    gradients = t.autograd.grad(result.float().sum(), tensors)
    for gradient, original, multiplier in zip(gradients, tensors, (1, 2, 3), strict=True):
        t.testing.assert_close(gradient, t.full_like(original, multiplier), rtol=0, atol=0)


@pytest.mark.skipif(not _SM12, reason="native FA4 validation requires SM12 CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_sft_bpb_eval_with_real_sm12_fa4(monkeypatch, native_dispatch, compiled):
    model, loader, token_bytes = _eval_case(monkeypatch, "cuda", t.bfloat16)
    adapter.admit_project_attention()
    assert adapter.uses_native_sm12_fa4()
    if compiled:
        model = t.compile(model)
    with t.amp.autocast("cuda", dtype=t.bfloat16):
        bpb = evaluate_encdec_bpb(model, loader, 1, token_bytes)
    assert math.isfinite(bpb) and bpb > 0
