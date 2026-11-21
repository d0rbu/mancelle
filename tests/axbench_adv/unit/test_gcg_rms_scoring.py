"""RMS scoring fusion preserves eager rounding, inputs, and dispatch contracts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import pytest
import torch as t
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.smollm3.modeling_smollm3 import SmolLM3RMSNorm

from scripts.eval.axbench_adv.vendored import gcg_mlp_scoring as scoring

KINDS = (LlamaRMSNorm, Qwen2RMSNorm, SmolLM3RMSNorm)


@contextmanager
def _context(model: t.nn.Module, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(scoring, "_has_build_prerequisites", lambda: True)
    if scoring.silu_mul_ is None:
        monkeypatch.setattr(scoring, "silu_mul_", lambda *_args: pytest.fail("unexpected MLP"))
    with scoring.fused_scoring_mlps(model):
        yield


def _require_cuda() -> None:
    if (
        not t.cuda.is_available()
        or scoring.rms_postscale is None
        or not scoring._has_build_prerequisites()
    ):
        pytest.skip("requires CUDA, Triton, and compiler/Python headers")


def _assert_bits(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.dtype == expected.dtype == t.bfloat16
    assert t.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    assert t.equal(actual.view(t.int16)[valid], expected.view(t.int16)[valid])


@pytest.mark.parametrize("kind", KINDS)
def test_cpu_fallback_preserves_forward_and_backward(
    kind: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    norm = kind(7)
    x = t.randn(2, 3, 7, requires_grad=True)
    expected = norm(x)
    gradient = t.autograd.grad(expected.sum(), x)[0]
    monkeypatch.setattr(scoring, "rms_postscale", lambda *_args: pytest.fail("unexpected fusion"))
    with _context(norm, monkeypatch):
        assert norm.forward.__func__ is scoring._scoring_norm_forward
        actual = norm(x)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
        t.testing.assert_close(t.autograd.grad(actual.sum(), x)[0], gradient, rtol=0, atol=0)
        with t.no_grad():
            t.testing.assert_close(norm(x), expected, rtol=0, atol=0)
    assert "forward" not in norm.__dict__


@pytest.mark.parametrize("kind", KINDS)
def test_native_keyword_call_signature(kind: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    norm = kind(7)
    x = t.randn(2, 3, 7)
    expected = norm(hidden_states=x)
    with _context(norm, monkeypatch):
        t.testing.assert_close(norm(hidden_states=x), expected, rtol=0, atol=0)
    assert "forward" not in norm.__dict__


@pytest.mark.parametrize(
    "case", ["instance", "class", "compiled", "subclass", "parameter", "tensor", "epsilon"]
)
def test_custom_norms_keep_native_dispatch(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class CustomNorm(SmolLM3RMSNorm):
        pass

    class CustomParameter(t.nn.Parameter):
        pass

    norm: Any = CustomNorm(7) if case == "subclass" else SmolLM3RMSNorm(7)
    if case == "instance":
        norm.forward = norm.forward
    elif case == "class":
        class_forward = SmolLM3RMSNorm.forward
        monkeypatch.setattr(SmolLM3RMSNorm, "forward", lambda self, x: class_forward(self, x))
    elif case == "compiled":
        norm._compiled_call_impl = norm.forward
    elif case == "parameter":
        norm.weight = CustomParameter(norm.weight.detach())
    elif case == "tensor":
        weight = norm.weight.detach()
        del norm.weight
        norm.weight = weight
    elif case == "epsilon":
        norm.variance_epsilon = t.tensor([1e-6], dtype=t.float64)
    original = norm.forward
    x = t.randn(2, 3, 7)
    expected = norm(x)
    if case == "epsilon":
        variance = x.float().pow(2).mean(-1, keepdim=True)
        assert (variance + norm.variance_epsilon).dtype == t.float64
    assert not scoring._eligible_norm(norm)
    with _context(t.nn.Sequential(norm), monkeypatch):
        assert norm.forward == original
        t.testing.assert_close(norm(x), expected, rtol=0, atol=0)
    assert norm.forward == original


@pytest.mark.parametrize(
    "registration",
    [
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
    ],
)
def test_norm_hooks_preserve_original_forward(
    registration: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    norm = SmolLM3RMSNorm(7)
    handle = getattr(norm, registration)(lambda *_args: None)
    try:
        assert not scoring._eligible_norm(norm)
        with _context(t.nn.Sequential(norm), monkeypatch):
            assert "forward" not in norm.__dict__
    finally:
        handle.remove()


def test_nested_context_and_exception_restore_every_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = SmolLM3RMSNorm(7), LlamaRMSNorm(7)
    model = t.nn.Sequential(first, second)
    with pytest.raises(RuntimeError, match="body failure"), _context(model, monkeypatch):
        forward = first.forward
        with scoring.fused_scoring_mlps(model):
            assert first.forward is forward
        raise RuntimeError("body failure")
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__
    original = scoring._eligible_norm

    def fail(module: t.nn.Module) -> bool:
        if module is second:
            raise RuntimeError("entry failure")
        return original(module)

    monkeypatch.setattr(scoring, "_eligible_norm", fail)
    with pytest.raises(RuntimeError, match="entry failure"), _context(model, monkeypatch):
        pytest.fail("entry should fail")
    assert "forward" not in first.__dict__ and "forward" not in second.__dict__


def test_kernel_requires_disabled_autograd_and_cuda() -> None:
    if scoring.rms_postscale is None:
        pytest.skip("requires Triton import")
    x = t.ones(2, 7, dtype=t.bfloat16)
    weight = t.ones(7, dtype=t.bfloat16)
    with t.enable_grad(), pytest.raises(RuntimeError, match="disabled autograd"):
        scoring.rms_postscale(x, weight, 1e-6)
    with t.no_grad(), pytest.raises(ValueError, match="BF16 CUDA operands"):
        scoring.rms_postscale(x, weight, 1e-6)
    with t.no_grad(), pytest.raises(ValueError, match="numeric scalar epsilon"):
        scoring.rms_postscale(x, weight, cast(Any, t.tensor([1e-6])))


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(1,), (3, 7), (2, 3, 127), (0, 2048), (2, 0), (2, 459, 2048)])
def test_cuda_exact_outputs_and_input_preservation(
    kind: Any, shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cuda()
    norm = kind(shape[-1]).cuda().bfloat16()
    x = t.randn(shape, device="cuda", dtype=t.bfloat16)
    kernel = scoring.rms_postscale
    calls = []

    def record(*args: Any) -> t.Tensor:
        calls.append(True)
        assert kernel is not None
        return kernel(*args)

    monkeypatch.setattr(scoring, "rms_postscale", record)
    with t.no_grad():
        norm.weight.copy_(t.randn_like(norm.weight))
        before, weight = x.clone(), norm.weight.clone()
        versions = (x._version, norm.weight._version)
        expected = norm(x)
        with scoring.fused_scoring_mlps(norm):
            actual = norm(hidden_states=x)
        _assert_bits(actual, expected)
        _assert_bits(x, before)
        _assert_bits(norm.weight, weight)
        assert versions == (x._version, norm.weight._version)
        assert actual.is_contiguous()
        if actual.numel():
            assert actual.data_ptr() not in (x.data_ptr(), norm.weight.data_ptr())
    assert len(calls) == 1 and "forward" not in norm.__dict__
    calls.clear()
    if x.numel():
        x.requires_grad_(True)
        expected_gradient = t.autograd.grad(norm(x).float().sum(), x)[0]
        with scoring.fused_scoring_mlps(norm):
            actual_gradient = t.autograd.grad(norm(hidden_states=x).float().sum(), x)[0]
        assert not calls
        _assert_bits(actual_gradient, expected_gradient)


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    [
        "float32",
        "float16",
        "float_weight",
        "input_stride",
        "weight_stride",
        "scalar",
        "weight_shape",
        "autocast",
        "pre_ampere",
        "subclass",
    ],
)
def test_cuda_unsupported_inputs_use_native(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class Subclass(t.Tensor):
        pass

    _require_cuda()
    norm = SmolLM3RMSNorm(7).cuda().bfloat16()
    x = t.randn(2, 3, 7, device="cuda", dtype=t.bfloat16)
    if case in {"float32", "float16"}:
        dtype = t.float32 if case == "float32" else t.float16
        norm, x = norm.to(dtype), x.to(dtype)
    elif case == "float_weight":
        norm.float()
    elif case == "input_stride":
        x = x.transpose(0, 1)
    elif case == "weight_stride":
        norm.weight = t.nn.Parameter(t.randn(14, device="cuda", dtype=t.bfloat16)[::2])
    elif case == "scalar":
        x = x[0, 0, 0]
    elif case == "weight_shape":
        norm.weight = t.nn.Parameter(norm.weight.unsqueeze(0))
    elif case == "pre_ampere":
        monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (7, 5))
    elif case == "subclass":
        x = x.as_subclass(Subclass)
    monkeypatch.setattr(scoring, "rms_postscale", lambda *_args: pytest.fail("unexpected fusion"))
    with t.no_grad(), t.autocast("cuda", dtype=t.float16, enabled=case == "autocast"):
        expected = norm(x)
        with scoring.fused_scoring_mlps(norm):
            actual = norm(x)
    assert type(actual) is type(expected)
    t.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.slow
def test_cross_device_weight_preserves_native_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    norm = SmolLM3RMSNorm(7).bfloat16()
    x = t.randn(2, 7, device="cuda", dtype=t.bfloat16)
    monkeypatch.setattr(scoring, "rms_postscale", lambda *_args: pytest.fail("unexpected fusion"))
    with t.no_grad():
        with pytest.raises(RuntimeError) as expected:
            norm(x)
        with scoring.fused_scoring_mlps(norm), pytest.raises(RuntimeError) as actual:
            norm(x)
    assert actual.value.args == expected.value.args
    assert "forward" not in norm.__dict__


@pytest.mark.slow
@pytest.mark.parametrize("eps", [-1.0, 0.0, 1e-6, 0.25, 1, float("inf"), float("nan")])
def test_inference_mode_extremes_and_aliased_operands(eps: float) -> None:
    _require_cuda()
    norm = SmolLM3RMSNorm(13, eps=eps).cuda().bfloat16()
    with t.inference_mode():
        values = t.tensor(
            [
                0.0,
                -0.0,
                2.0**-133,
                -(2.0**-133),
                1e-38,
                -1e-38,
                1e38,
                -1e38,
                float("inf"),
                -float("inf"),
                float("nan"),
                1.0,
                -1.0,
            ],
            device="cuda",
            dtype=t.bfloat16,
        )
        x = values.repeat(13, 1).transpose(0, 1).contiguous()
        norm.weight.copy_(values)
        expected = norm(x)
        with scoring.fused_scoring_mlps(norm):
            actual = norm(x)
        _assert_bits(actual, expected)
        assert actual.is_inference()
        # The low-level helper must not overwrite its inputs even when they alias.
        norm.weight.fill_(1.25)
        alias = norm.weight.view(1, -1)
        expected = norm(alias)
        before = norm.weight.clone()
        with scoring.fused_scoring_mlps(norm):
            actual = norm(alias)
        _assert_bits(actual, expected)
        _assert_bits(norm.weight, before)


@pytest.mark.slow
@pytest.mark.parametrize(
    "case", ["dtype", "weight_dtype", "shape", "strides", "weight_strides", "device", "scalar"]
)
def test_kernel_rejects_incompatible_operands(case: str) -> None:
    _require_cuda()
    assert scoring.rms_postscale is not None
    x = t.ones(2, 7, device="cuda", dtype=t.bfloat16)
    weight = t.ones(7, device="cuda", dtype=t.bfloat16)
    if case == "dtype":
        x = x.float()
    elif case == "weight_dtype":
        weight = weight.float()
    elif case == "shape":
        weight = weight[:6]
    elif case == "strides":
        x = x.t()
        weight = weight[:2]
    elif case == "weight_strides":
        weight = t.ones(14, device="cuda", dtype=t.bfloat16)[::2]
    elif case == "device":
        weight = weight.cpu()
    elif case == "scalar":
        x = x[0, 0]
    with t.no_grad(), pytest.raises(ValueError, match="BF16 CUDA operands"):
        scoring.rms_postscale(x, weight, 1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("hidden_dtype", [t.float32, t.bfloat16])
def test_postscale_matches_every_bf16_weight_pair(hidden_dtype: t.dtype) -> None:
    _require_cuda()
    assert scoring.rms_postscale is not None
    kernel = cast(Any, scoring.rms_postscale).__globals__["_rms_postscale_kernel"]
    values = t.arange(65536, dtype=t.int32).to(t.int16).view(t.bfloat16).cuda()
    inverse = t.ones(128, 1, device="cuda")
    with t.no_grad():
        for begin in range(0, 65536, 128):
            hidden = (
                values[begin : begin + 128]
                .to(hidden_dtype)
                .unsqueeze(1)
                .expand(-1, 65536)
                .contiguous()
            )
            expected = hidden.bfloat16() * values
            actual = t.empty_like(expected)
            kernel[((hidden.numel() + 1023) // 1024,)](
                hidden,
                inverse,
                values,
                actual,
                hidden.numel(),
                width=65536,
                block=1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
            both_nan = expected.isnan() & actual.isnan()
            assert bool(((expected.view(t.int16) == actual.view(t.int16)) | both_nan).all()), begin


@pytest.mark.slow
def test_square_matches_every_bf16_input_and_preserves_guards() -> None:
    _require_cuda()
    assert scoring.rms_postscale is not None
    kernel = cast(Any, scoring.rms_postscale).__globals__["_bf16_square_kernel"]
    bits = t.arange(65536, dtype=t.int32).to(t.int16).cuda()
    # Offset both pointers and end partway through the final block. The extra
    # storage detects writes outside the output and mutations of any input bit.
    storage = t.cat((bits[:1], bits, bits[:3], bits[:17]))
    hidden = storage[1:65540].view(t.bfloat16)
    before = storage.clone()
    output_storage = t.full((hidden.numel() + 18,), 123.0, device="cuda")
    actual = output_storage[1 : hidden.numel() + 1]
    expected = hidden.float().pow(2)
    kernel[((hidden.numel() + 1023) // 1024,)](
        hidden, actual, hidden.numel(), block=1024, num_warps=4, enable_fp_fusion=False
    )
    assert t.equal(actual.isnan(), expected.isnan())
    valid = ~expected.isnan()
    assert t.equal(actual.view(t.int32)[valid], expected.view(t.int32)[valid])
    assert t.equal(storage, before)
    assert output_storage[0].item() == 123.0
    assert bool((output_storage[hidden.numel() + 1 :] == 123.0).all())


@pytest.mark.slow
def test_unaligned_input_and_weight_preserve_native_norm() -> None:
    _require_cuda()
    norm = SmolLM3RMSNorm(513).cuda().bfloat16()
    x = t.randn(3 * 513 + 1, device="cuda", dtype=t.bfloat16)[1:].view(3, 513)
    weight = t.randn(514, device="cuda", dtype=t.bfloat16)[1:]
    norm.weight = t.nn.Parameter(weight)
    assert x.is_contiguous() and x.storage_offset() == 1
    assert weight.is_contiguous() and weight.storage_offset() == 1
    with t.no_grad():
        before, original_weight = x.clone(), weight.clone()
        expected = norm(x)
        with scoring.fused_scoring_mlps(norm):
            actual = norm(hidden_states=x)
        _assert_bits(actual, expected)
        _assert_bits(x, before)
        _assert_bits(weight, original_weight)


@pytest.mark.slow
def test_private_square_buffer_reuse_on_nondefault_stream() -> None:
    _require_cuda()
    norm = SmolLM3RMSNorm(2048).cuda().bfloat16()
    stream = t.cuda.Stream()
    stream.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(stream), t.no_grad():
        x = t.randn(7, 2048, device="cuda", dtype=t.bfloat16)
        expected = norm(x)
        with scoring.fused_scoring_mlps(norm):
            outputs = [norm(hidden_states=x) for _ in range(8)]
    stream.synchronize()
    for actual in outputs:
        _assert_bits(actual, expected)


@pytest.mark.slow
def test_kernel_offsets_beyond_signed_int32() -> None:
    _require_cuda()
    assert scoring.rms_postscale is not None
    width = 1024
    count = (1 << 31) + width
    free, _total = t.cuda.mem_get_info()
    if free < count * 6 + (2 << 30):
        pytest.skip("requires 14 GiB free for BF16 input and FP32 squares")
    norm = SmolLM3RMSNorm(width).cuda().bfloat16()
    with t.no_grad():
        norm.weight.copy_(t.linspace(-2, 2, width, device="cuda", dtype=t.bfloat16))
        hidden = t.ones(count // width, width, device="cuda", dtype=t.bfloat16)
        indices = t.tensor(
            [0, 1023, (1 << 30) - 1, (1 << 31) - 1, 1 << 31, count - 1], device="cuda"
        )
        hidden.view(-1)[indices] = t.arange(indices.numel(), device="cuda", dtype=t.bfloat16) - 2
        rows, columns = indices // width, indices % width
        before = hidden[rows].clone()
        expected = norm(before)[t.arange(indices.numel(), device="cuda"), columns]
        actual = scoring.rms_postscale(hidden, norm.weight, norm.variance_epsilon)
        _assert_bits(actual.view(-1)[indices], expected)
        _assert_bits(hidden[rows], before)
