"""Native SM12 attention contracts and independent CUDA numerical oracles."""

from __future__ import annotations

import builtins
import importlib.util

import pytest
import torch as t
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat import flash_attention, gpt
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder import sm12_fa4 as adapter
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError


@pytest.fixture
def attention_state(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", None)
    monkeypatch.setattr(flash_attention.flash_attn, "flash_attn_func", flash_attention.flash_attn.flash_attn_func)
    deterministic = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        t.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("capability", [None, (8, 9), (9, 0), (10, 0), (12, 0), (12, 1)])
def test_installation_is_sm12_only_idempotent_and_preserves_kv_cache(monkeypatch, attention_state, capability):
    del attention_state
    original = flash_attention.flash_attn.flash_attn_func
    cached = flash_attention.flash_attn.flash_attn_with_kvcache
    canonical = flash_attention.flash_attn_func
    monkeypatch.setattr(t.cuda, "is_available", lambda: capability is not None)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: capability)
    monkeypatch.setattr(adapter, "_native_fa4", object())
    for _ in range(2):
        adapter.admit_project_attention()
        expected = adapter._full_prefix_attention if capability and capability[0] == 12 else original
        assert flash_attention.flash_attn.flash_attn_func is expected
        assert gpt.flash_attn.flash_attn_func is expected
        assert flash_attention.flash_attn.flash_attn_with_kvcache is cached
        assert flash_attention.flash_attn_func is canonical


def test_missing_native_fa4_fails_without_replacing_dispatch(monkeypatch, attention_state):
    del attention_state
    original = flash_attention.flash_attn.flash_attn_func
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(adapter, "_native_fa4", None)
    with pytest.raises(SFTGRPOError, match="requires native FA4/CuTe"):
        adapter.admit_project_attention()
    with pytest.raises(SFTGRPOError, match="requires native FA4/CuTe"):
        adapter._causal_attention(t.empty(0), t.empty(0), t.empty(0))
    assert flash_attention.flash_attn.flash_attn_func is original


def test_mixed_cuda_devices_use_their_own_full_prefix_backend(monkeypatch, attention_state):
    del attention_state
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        t.cuda, "get_device_capability",
        lambda device=None: (12, 0) if device == t.device("cuda:1") else (9, 0),
    )
    monkeypatch.setattr(flash_attention, "HAS_FA3", True)
    monkeypatch.setattr(adapter, "_native_fa4", object())
    adapter.admit_project_attention(device=t.device("cuda:1"))
    assert flash_attention.HAS_FA3
    assert adapter.uses_native_sm12_fa4(t.device("cuda:1"))
    assert not adapter.uses_native_sm12_fa4(t.device("cuda:0"))

    fallback = object()
    def reference(*_args, **_kwargs):
        return fallback

    monkeypatch.setattr(flash_attention, "flash_attn_func", reference)
    q = type("CudaTensor", (), {"is_cuda": True, "device": t.device("cuda:0")})()
    assert adapter._full_prefix_attention(q, q, q, causal=True) is fallback
    monkeypatch.setattr(flash_attention, "HAS_FA4_BACKEND", False)
    monkeypatch.setattr(flash_attention, "has_fa4_backend", lambda: False)
    with pytest.raises(SFTGRPOError, match="available FA4"):
        adapter.admit_project_attention(device=t.device("cuda:0"), require_fa4=True)
    adapter.admit_project_attention(device=t.device("cuda:0"))
    assert not adapter.uses_native_sm12_fa4(t.device("cuda:0"))


def test_selected_sm8_rejects_imported_fa4_backend(monkeypatch, attention_state):
    del attention_state
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (8, 9))
    monkeypatch.setattr(flash_attention, "HAS_FA4_BACKEND", True)
    with pytest.raises(SFTGRPOError, match="imported FA4 backend"):
        adapter.admit_project_attention(device=t.device("cuda:1"))


def test_explicit_fa3_requires_imported_backend_on_sm9(monkeypatch, attention_state):
    del attention_state
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (9, 0))
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.FA3)
    monkeypatch.setattr(flash_attention, "HAS_FA3", False)
    with pytest.raises(SFTGRPOError, match="available backend"):
        adapter.admit_project_attention(device=t.device("cuda:1"))


def test_shared_attention_admission_is_safe_without_visible_cuda(monkeypatch, attention_state):
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    original = flash_attention.flash_attn.flash_attn_func
    cached = flash_attention.flash_attn.flash_attn_with_kvcache
    adapter.admit_project_attention()
    assert flash_attention.flash_attn.flash_attn_func is original
    assert flash_attention.flash_attn.flash_attn_with_kvcache is cached


def _construction_model():
    return EncoderDecoderGPT(
        gpt.GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32),
        EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16),
        CrossAttentionConfig(),
        n_sae_features=8,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(block=FeaturePredictionBlock.TRANSFORMER, head_layers=2),
    )


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("implementation", [None, flash_attention.FlashAttnImpl.FA4])
def test_cpu_meta_model_lifecycle_does_not_require_visible_sm12_dependency(
    monkeypatch, attention_state, device, implementation,
):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(adapter, "_native_fa4", None)
    monkeypatch.setattr(flash_attention, "_override_impl", implementation)
    full_prefix = flash_attention.flash_attn.flash_attn_func
    cached = flash_attention.flash_attn.flash_attn_with_kvcache
    with t.device(device):
        model = _construction_model()
    assert all(parameter.device.type == device for parameter in model.parameters())
    model.to_empty(device="cpu")
    model.init_buffers()
    model.double()
    assert all(parameter.device.type == "cpu" and parameter.dtype == t.float64 for parameter in model.parameters())
    assert flash_attention.flash_attn.flash_attn_func is full_prefix
    assert flash_attention.flash_attn.flash_attn_with_kvcache is cached


@pytest.mark.parametrize("device", [None, t.device("cuda")])
@pytest.mark.parametrize("require_fa4", [False, True])
def test_cuda_and_early_trainer_admission_still_require_sm12_dependency(
    monkeypatch, attention_state, device, require_fa4,
):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(adapter, "_native_fa4", None)
    with pytest.raises(SFTGRPOError, match="requires native FA4/CuTe"):
        adapter.admit_project_attention(device=device, require_fa4=require_fa4)


def test_materialization_admits_actual_decoder_device_and_preserves_recurse_false(monkeypatch, attention_state):
    admitted = []
    original = adapter.admit_project_attention

    def observe(*, device=None, require_fa4=False):
        admitted.append(device)
        return original(device=device, require_fa4=require_fa4)

    monkeypatch.setattr(model_module, "admit_project_attention", observe)
    with t.device("meta"):
        model = _construction_model()
        child_parameters = tuple(model._decoder_blocks.parameters())
        assert model.to_empty(device="cpu", recurse=False) is model
        assert model.resid_lambdas.device.type == "cpu"
        assert model.cos.device.type == "cpu"
        assert all(parameter.device.type == "meta" for parameter in child_parameters)
        assert all(left is right for left, right in zip(child_parameters, model._decoder_blocks.parameters(), strict=True))
        # Ambient meta construction must not hide the materialized CPU device.
        assert model.to_empty(device="cpu") is model
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert model.double() is model
    assert admitted == [t.device("meta"), t.device("meta"), t.device("cpu"), t.device("cpu")]

    # Exercise the post-move CUDA admission without allocating a GPU tensor.
    monkeypatch.setattr(model, "get_device", lambda: t.device("cuda"))
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(adapter, "_native_fa4", None)
    with pytest.raises(SFTGRPOError, match="requires native FA4/CuTe"):
        model.float()
    assert admitted[-1] == t.device("cuda")


def test_adapter_can_import_on_platforms_without_native_cute(monkeypatch):
    original_import = builtins.__import__

    def without_cute(name, *args, **kwargs):
        if name == "flash_attn.cute.interface":
            raise ImportError("CuTe is unavailable on this platform")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cute)
    spec = importlib.util.spec_from_file_location("sm12_without_cute", adapter.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._native_fa4 is None


@pytest.mark.parametrize("implementation", [flash_attention.FlashAttnImpl.SDPA, flash_attention.FlashAttnImpl.FA3])
def test_explicit_reference_override_survives_model_admission(monkeypatch, attention_state, implementation):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(adapter, "_native_fa4", None)
    monkeypatch.setattr(flash_attention, "_override_impl", implementation)
    original = flash_attention.flash_attn.flash_attn_func
    if implementation == flash_attention.FlashAttnImpl.FA3:
        with pytest.raises(SFTGRPOError, match="requires an SM9"):
            adapter.admit_project_attention()
    else:
        adapter.admit_project_attention()
    assert flash_attention.flash_attn.flash_attn_func is original
    assert not adapter.uses_native_sm12_fa4()
    with pytest.raises(SFTGRPOError, match="active attention override"):
        adapter.admit_project_attention(require_fa4=True)


@pytest.mark.parametrize("available", [False, True])
def test_hopper_explicit_fa4_checks_real_backend(monkeypatch, attention_state, available):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (9, 0))
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.FA4)
    monkeypatch.setattr(flash_attention, "_active_impl", "fa4")
    monkeypatch.setattr(flash_attention, "has_fa4_backend", lambda: available)
    if available:
        adapter.admit_project_attention()
    else:
        with pytest.raises(SFTGRPOError, match="available FA4/CuTe backend"):
            adapter.admit_project_attention()


def test_installed_adapter_uses_cpu_reference_and_respects_later_override(monkeypatch, attention_state):
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setattr(adapter, "_native_fa4", lambda *_args, **_kwargs: pytest.fail("CPU/reference reached native FA4"))
    adapter.admit_project_attention()
    assert adapter.uses_native_sm12_fa4()
    calls = []
    monkeypatch.setattr(flash_attention, "flash_attn_func", lambda *_args, **kwargs: calls.append(kwargs))
    tensor = t.empty(1, 1, 2, 64)
    flash_attention.flash_attn.flash_attn_func(tensor, tensor, tensor, causal=True)
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.SDPA)
    assert not adapter.uses_native_sm12_fa4()
    # A CUDA-like argument pins dispatch independently of hardware availability.
    cuda_tensor = type("CudaTensor", (), {"is_cuda": True})()
    flash_attention.flash_attn.flash_attn_func(cuda_tensor, cuda_tensor, cuda_tensor, causal=True)
    assert calls == [{"causal": True, "window_size": (-1, -1)}] * 2


@pytest.mark.parametrize("window", [-1, 0, 5, 1000])
@pytest.mark.parametrize("deterministic", [False, True])
def test_native_layout_window_determinism_and_autograd_are_forwarded(monkeypatch, attention_state, window, deterministic):
    del attention_state
    t.use_deterministic_algorithms(deterministic)
    q, k, v = (t.randn(2, heads, 7, 8, requires_grad=True) for heads in (4, 2, 2))
    calls = []

    def native(q_arg, k_arg, v_arg, **kwargs):
        calls.append(kwargs)
        assert t.equal(q_arg, q.transpose(1, 2))
        assert t.equal(k_arg, k.transpose(1, 2))
        assert t.equal(v_arg, v.transpose(1, 2))
        return q_arg + k_arg.repeat_interleave(2, dim=2) + v_arg.repeat_interleave(2, dim=2), None

    monkeypatch.setattr(adapter, "_native_fa4", native)
    out = adapter._causal_attention(q, k, v, causal=True, window_size=(window, 0))
    assert t.equal(out, q + k.repeat_interleave(2, dim=1) + v.repeat_interleave(2, dim=1))
    assert calls == [{"causal": True, "window_size": (None if window < 0 else window, 0), "deterministic": deterministic}]
    gradients = t.autograd.grad(out.sum(), (q, k, v))
    for actual, scale in zip(gradients, (1, 2, 2), strict=True):
        assert t.equal(actual, t.full_like(actual, scale))


@pytest.mark.parametrize("causal,kv_length", [(False, 7), (True, 8)])
def test_adapter_rejects_non_causal_or_non_prefix_calls(monkeypatch, causal, kv_length):
    monkeypatch.setattr(adapter, "_native_fa4", lambda *_args, **_kwargs: pytest.fail("invalid input reached CuTe"))
    q = t.empty(1, 2, 7, 64)
    k = t.empty(1, 2, kv_length, 64)
    with pytest.raises(SFTGRPOError, match="requires a causal full prefix"):
        adapter._causal_attention(q, k, k, causal=causal)


_SM12 = t.cuda.is_available() and t.cuda.get_device_capability()[0] == 12


def _math_attention(q, k, v, window):
    length = q.shape[2]
    positions = t.arange(length, device=q.device)
    distance = positions[:, None] - positions[None, :]
    mask = distance >= 0
    if window >= 0:
        mask &= distance <= window
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)


@pytest.mark.slow
@pytest.mark.skipif(not _SM12, reason="native CuTe SM12 qualification requires SM12 CUDA")
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16])
@pytest.mark.parametrize("length,width,window,kv_heads", [
    (1, 32, -1, 8), (33, 64, 0, 4), (127, 64, 31, 8),
    (128, 64, -1, 4), (129, 64, -1, 4), (257, 128, 31, 4),
    (257, 128, 1024, 8), (2048, 128, -1, 4), (2048, 128, 1024, 4),
    (193, 128, 31, 13), (2048, 128, -1, 13), (2048, 128, 1024, 13),
    (2048, 128, 2048, 13),
])
def test_sm12_forward_backward_and_replay_against_float64_math(
    attention_state, dtype, length, width, window, kv_heads,
):
    del attention_state
    adapter.admit_project_attention()
    t.use_deterministic_algorithms(False)
    generator = t.Generator(device="cuda").manual_seed(1687)
    query_heads = max(8, kv_heads)
    batch = 2 if length == 129 else 1
    tensors = [t.randn(batch, heads, length, width, device="cuda", dtype=dtype, generator=generator).requires_grad_()
               for heads in (query_heads, kv_heads, kv_heads)]
    oracle = [tensor.detach().double().requires_grad_() for tensor in tensors]
    dout = t.randn(tensors[0].shape, device="cuda", dtype=dtype, generator=generator)
    actual = flash_attention.flash_attn.flash_attn_func(*tensors, causal=True, window_size=(window, 0))
    expected = _math_attention(oracle[0], oracle[1], oracle[2], window)
    gradients = t.autograd.grad(actual, tensors, dout)
    oracle_gradients = t.autograd.grad(expected, oracle, dout.double())
    for result, reference in zip((actual, *gradients), (expected, *oracle_gradients), strict=True):
        t.testing.assert_close(result.double(), reference, atol=0.02, rtol=0.02)
        # Clamp the denominator for mathematically zero gradients (one key).
        # Ordinary nonzero gradients must also have <1% aggregate error.
        assert (result.double() - reference).norm() / reference.norm().clamp_min(1.0) < 0.01
    with t.no_grad():
        rollout = flash_attention.flash_attn.flash_attn_func(
            *(tensor.detach() for tensor in tensors), causal=True, window_size=(window, 0)
        )
    assert t.equal(actual, rollout)
    if window == 0 or length == 1:
        assert t.equal(actual, tensors[2].repeat_interleave(query_heads // kv_heads, dim=1))
        # CuTe's FP32 reduction can leave sub-micro roundoff in the two
        # mathematically zero derivatives; this is separate from BF16 error.
        for gradient in gradients[:2]:
            t.testing.assert_close(gradient, t.zeros_like(gradient), atol=2e-6, rtol=0)


@pytest.mark.slow
@pytest.mark.skipif(not _SM12, reason="native CuTe SM12 qualification requires SM12 CUDA")
@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16])
@pytest.mark.parametrize("query_heads,kv_heads", [(13, 13), (8, 4)])
def test_sm12_fused_qkv_storage_and_contiguous_qk_match_math(attention_state, dtype, query_heads, kv_heads):
    """Match GPT's contiguous normalized Q/K and strided split-projection V."""
    del attention_state
    t.use_deterministic_algorithms(False)
    generator = t.Generator(device="cuda").manual_seed(1687)
    projected = t.randn(2, 129, query_heads + 2 * kv_heads, 128, device="cuda", dtype=dtype,
                        generator=generator).requires_grad_()
    reference_projected = projected.detach().double().requires_grad_()

    def decoder_inputs(storage):
        q, k, v = storage.split((query_heads, kv_heads, kv_heads), dim=2)
        return q.contiguous().transpose(1, 2), k.contiguous().transpose(1, 2), v.transpose(1, 2)

    inputs = decoder_inputs(projected)
    assert inputs[2].stride(2) == (query_heads + 2 * kv_heads) * 128
    actual = adapter._causal_attention(*inputs, causal=True, window_size=(31, 0))
    reference_q, reference_k, reference_v = decoder_inputs(reference_projected)
    expected = _math_attention(reference_q, reference_k, reference_v, 31)
    dout = t.randn(actual.shape, device="cuda", dtype=dtype, generator=generator)
    actual_gradient, = t.autograd.grad(actual, projected, dout)
    expected_gradient, = t.autograd.grad(expected, reference_projected, dout.double())
    for result, reference in ((actual, expected), (actual_gradient, expected_gradient)):
        t.testing.assert_close(result.double(), reference, atol=0.02, rtol=0.02)
        assert (result.double() - reference).norm() / reference.norm().clamp_min(1.0) < 0.01
    with t.no_grad():
        rollout = adapter._causal_attention(*decoder_inputs(projected.detach()), causal=True, window_size=(31, 0))
    assert t.equal(actual, rollout)


@pytest.mark.slow
@pytest.mark.skipif(not _SM12, reason="native CuTe SM12 qualification requires SM12 CUDA")
@pytest.mark.parametrize("window", [-1, 0, 31, 128])
def test_sm12_mask_boundaries_exclude_future_and_expired_values(attention_state, window):
    del attention_state
    adapter.admit_project_attention()
    q = t.zeros(1, 4, 257, 64, device="cuda", dtype=t.bfloat16)
    k = t.zeros(1, 2, 257, 64, device="cuda", dtype=t.bfloat16)
    v = t.arange(257, device="cuda", dtype=t.bfloat16)[None, None, :, None].expand_as(k).contiguous()
    with t.no_grad():
        original = adapter._causal_attention(q, k, v, causal=True, window_size=(window, 0))
        expected = _math_attention(q.double(), k.double(), v.double(), window).to(q.dtype)
        t.testing.assert_close(original, expected, atol=0, rtol=0)
        changed = v.clone()
        changed[:, :, 130:, :] = 1000
        if window >= 0:
            changed[:, :, :max(0, 129 - window), :] = -1000
        modified = adapter._causal_attention(q, k, changed, causal=True, window_size=(window, 0))
    assert t.equal(original[:, :, 129], modified[:, :, 129])


@pytest.mark.slow
@pytest.mark.skipif(not _SM12, reason="native CuTe SM12 qualification requires SM12 CUDA")
def test_sm12_hard_deterministic_backward_is_rejected_by_the_locked_kernel(attention_state):
    del attention_state
    t.use_deterministic_algorithms(True)
    q, k, v = [t.ones(1, 1, 33, 64, device="cuda", dtype=t.bfloat16, requires_grad=True) for _ in range(3)]
    out = adapter._causal_attention(q, k, v, causal=True, window_size=(-1, 0))
    with pytest.raises(AssertionError, match="deterministic backward not supported on SM 12.0"):
        out.sum().backward()
