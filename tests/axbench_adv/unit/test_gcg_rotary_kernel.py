"""Exact rotary arithmetic, layout, fallback and native-attention checks."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import pytest
import torch as t
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.cache_utils import CacheLayerMixin, DynamicCache
from transformers.models.smollm3.modeling_smollm3 import apply_rotary_pos_emb

from scripts.eval.axbench_adv.vendored import gcg_mlp_scoring as scoring
from tests.axbench_adv.unit.test_gcg_rotary_scoring import KINDS, _attention, _inputs, _plain_native

if find_spec("triton") is not None:
    from scripts.eval.axbench_adv.vendored import gcg_rotary_kernel as rotary
else:
    rotary = None


pytestmark = pytest.mark.slow


def _require_cuda() -> None:
    if not t.cuda.is_available() or rotary is None or not scoring._has_build_prerequisites():
        pytest.skip("requires CUDA, Triton, and compiler/Python headers")
    if t.version.hip is not None or t.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires NVIDIA Ampere or later")


def _assert_bits(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.dtype == expected.dtype == t.bfloat16
    assert actual.shape == expected.shape and actual.stride() == expected.stride()
    assert t.equal(actual.isnan(), expected.isnan())
    finite_or_infinite = ~expected.isnan()
    assert t.equal(
        actual.view(t.int16)[finite_or_infinite], expected.view(t.int16)[finite_or_infinite]
    )


def _operands(
    batch: int = 2,
    sequence: int = 7,
    query_heads: int = 4,
    key_heads: int = 2,
    width: int = 64,
    cos_batch: int = 1,
    offset: int = 0,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    values = []
    for heads in (query_heads, key_heads):
        flat = t.randn(offset + batch * sequence * heads * width, device="cuda", dtype=t.bfloat16)
        values.append(flat[offset:].view(batch, sequence, heads, width).transpose(1, 2))
    angles = t.randn(cos_batch, sequence, width, device="cuda", dtype=t.float32)
    return values[0], values[1], angles.cos().to(t.bfloat16), angles.sin().to(t.bfloat16)


def _check(values: tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]) -> tuple[t.Tensor, t.Tensor]:
    assert rotary is not None
    before = [value.clone() for value in values]
    with t.no_grad():
        expected = apply_rotary_pos_emb(*values)
        actual = rotary.rotary_pair(*values)
    assert actual is not None
    for old, new in zip(expected, actual, strict=True):
        _assert_bits(new, old)
        assert not new.requires_grad
        assert all(
            new.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
            for value in values
        )
    assert actual[0].untyped_storage().data_ptr() != actual[1].untyped_storage().data_ptr()
    for value, snapshot in zip(values, before, strict=True):
        _assert_bits(value, snapshot)
    return actual


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 2, 2, 2, 1),
        (3, 19, 3, 2, 6, 3),
        (2, 7, 4, 2, 64, 1),
        (2, 459, 16, 4, 128, 1),
        (8, 459, 16, 4, 128, 1),
        (2, 443, 16, 4, 128, 2),
        (3, 19, 3, 2, 96, 1),
        (2, 5, 8, 2, 258, 2),
    ],
)
@pytest.mark.parametrize("offset", [0, 7])
def test_random_values_layout_and_storage_offsets(shape: tuple[int, ...], offset: int) -> None:
    _require_cuda()
    q, k, cosine, sine = _operands(*shape, offset=offset)
    # Exercise independent coefficient pointers as well as projection pointers.
    cosine = t.cat((cosine.new_zeros(offset), cosine.flatten()))[offset:].view_as(cosine)
    sine_offset = offset + 1
    sine = t.cat((sine.new_zeros(sine_offset), sine.flatten()))[sine_offset:].view_as(sine)
    assert cosine.storage_offset() == offset and sine.storage_offset() == sine_offset
    _check((q, k, cosine, sine))


@pytest.mark.parametrize(
    "cosine,sine",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, -1.0),
        (0.70703125, 0.70703125),
        (0.5, -0.5),
        (2.0**-133, -(2.0**-133)),
        (float("inf"), 1.0),
        (1.0, float("nan")),
    ],
)
def test_every_bf16_encoding_preserves_bits_and_nan_classification(
    cosine: float, sine: float
) -> None:
    _require_cuda()
    values = t.arange(65536, device="cuda", dtype=t.int32).to(t.int16).view(t.bfloat16)
    q = values.reshape(2, 32, 8, 128).transpose(1, 2)
    k = values.flip(0).reshape(2, 32, 8, 128).transpose(1, 2)
    cos = t.full((1, 32, 128), cosine, device="cuda", dtype=t.bfloat16)
    sin = t.full_like(cos, sine)
    _check((q, k, cos, sin))


def test_independent_random_bf16_bit_patterns() -> None:
    _require_cuda()
    shapes = [(2, 17, 8, 128), (2, 17, 2, 128), (2, 17, 128), (2, 17, 128)]
    values = [
        t.randint(-32768, 32768, shape, device="cuda", dtype=t.int16).view(t.bfloat16)
        for shape in shapes
    ]
    values[:2] = [value.transpose(1, 2) for value in values[:2]]
    _check((values[0], values[1], values[2], values[3]))


def test_inference_mode_aliases_and_nondefault_stream() -> None:
    _require_cuda()
    original_stream = t.cuda.current_stream()
    stream = t.cuda.Stream()
    with t.cuda.stream(stream), t.inference_mode():
        q, _, _, _ = _operands()
        flat = q.transpose(1, 2).view(-1)
        cos = flat[: 7 * 64].view(1, 7, 64)
        sin = flat[7 * 64 : 2 * 7 * 64].view(1, 7, 64)
        actual = _check((q, q, cos, sin))
        assert all(value.is_inference() for value in actual)
        assert t.cuda.current_stream() == stream
    assert t.cuda.current_stream() == original_stream
    stream.synchronize()


@pytest.mark.parametrize(
    "case",
    [
        "float32",
        "float16",
        "cpu_coefficient",
        "q_stride",
        "k_stride",
        "cos_stride",
        "sin_stride",
        "odd_width",
        "singleton_batch",
        "singleton_head",
        "singleton_sequence",
        "empty",
        "q_rank",
        "k_rank",
        "cos_rank",
        "sin_shape",
        "batch_mismatch",
        "sequence_mismatch",
        "cos_batch",
        "cos_width",
        "negated_q",
        "negated_k",
        "negated_cos",
        "negated_sin",
        "subclass",
        "sparse",
        "autocast",
        "grad",
        "pre_ampere",
        "rocm",
        "dim_bool",
        "dim_two",
    ],
)
def test_unsupported_operands_request_native_fallback(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Subclass(t.Tensor):
        pass

    _require_cuda()
    assert rotary is not None
    values = list(_operands())
    q, k, cos, sin = values
    dim = 1
    if case in {"float32", "float16"}:
        values = [value.to(t.float32 if case == "float32" else t.float16) for value in values]
    elif case == "cpu_coefficient":
        values[2] = cos.cpu()
    elif case in {"q_stride", "k_stride"}:
        i = 0 if case == "q_stride" else 1
        values[i] = values[i].contiguous()
    elif case in {"cos_stride", "sin_stride"}:
        i = 2 if case == "cos_stride" else 3
        values[i] = t.empty(1, 7, 128, device="cuda", dtype=t.bfloat16)[..., ::2]
    elif case == "odd_width":
        values = list(_operands(width=3))
    elif case == "singleton_batch":
        values = list(_operands(batch=1))
    elif case == "singleton_head":
        values = list(_operands(key_heads=1))
    elif case == "singleton_sequence":
        values = list(_operands(sequence=1))
    elif case == "empty":
        values = list(_operands(sequence=0))
    elif case in {"q_rank", "k_rank", "cos_rank"}:
        i = {"q_rank": 0, "k_rank": 1, "cos_rank": 2}[case]
        values[i] = values[i][0]
    elif case == "sin_shape":
        values[3] = sin[..., :32]
    elif case == "batch_mismatch":
        values[1] = _operands(batch=3)[1]
    elif case == "sequence_mismatch":
        values[1] = k[:, :, :3]
    elif case == "cos_batch":
        values[2:] = [value.repeat(3, 1, 1) for value in values[2:]]
    elif case == "cos_width":
        values[2:] = [value[..., :32].contiguous() for value in values[2:]]
    elif case.startswith("negated_"):
        i = {"negated_q": 0, "negated_k": 1, "negated_cos": 2, "negated_sin": 3}[case]
        values[i] = values[i]._neg_view()
        assert values[i].is_neg()
    elif case == "subclass":
        values[0] = q.as_subclass(Subclass)
    elif case == "sparse":
        values[0] = q.to_sparse()
    elif case == "pre_ampere":
        monkeypatch.setattr(t.cuda, "get_device_capability", lambda _device: (7, 5))
    elif case == "rocm":
        monkeypatch.setattr(t.version, "hip", "test")
    elif case == "dim_bool":
        dim = True
    elif case == "dim_two":
        dim = 2
    monkeypatch.setattr(rotary, "_rotate", lambda *_args: pytest.fail("unexpected kernel"))
    with (
        t.set_grad_enabled(case == "grad"),
        t.autocast("cuda", dtype=t.float16, enabled=case == "autocast"),
    ):
        assert rotary.rotary_pair(*values, unsqueeze_dim=dim) is None
        if case in {
            "q_stride",
            "k_stride",
            "odd_width",
            "singleton_batch",
            "singleton_head",
            "singleton_sequence",
            "empty",
        } or case.startswith("negated_"):
            expected = apply_rotary_pos_emb(*values)
            actual = scoring._scoring_rotary(
                apply_rotary_pos_emb, values[0], values[1], values[2], values[3]
            )
            for old, new in zip(expected, actual, strict=True):
                _assert_bits(new, old)


@pytest.mark.parametrize("mode_type", [TorchFunctionMode, TorchDispatchMode])
def test_torch_modes_keep_native_operators(mode_type: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_cuda()
    assert rotary is not None
    values = _operands()
    calls = []

    class Observe(mode_type):
        def __torch_function__(
            self, func: Any, _types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            calls.append(func)
            return func(*args, **(kwargs or {}))

        def __torch_dispatch__(
            self, func: Any, _types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            calls.append(func)
            return func(*args, **(kwargs or {}))

    monkeypatch.setattr(rotary, "_rotate", lambda *_args: pytest.fail("unexpected kernel"))
    with t.no_grad():
        expected = apply_rotary_pos_emb(*values)
        with Observe():
            actual = scoring._scoring_rotary(apply_rotary_pos_emb, *values)
        assert calls
        for old, new in zip(expected, actual, strict=True):
            _assert_bits(new, old)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_attention_forward_and_backward(
    kind: Any, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_cuda()
    assert rotary is not None
    model = _attention(kind).cuda().bfloat16()
    model.config._attn_implementation = backend
    inputs = _inputs(model, device="cuda", dtype=t.bfloat16)
    original = rotary._rotate
    calls = []

    def counted(*args: Any) -> t.Tensor:
        calls.append(args[0].shape)
        return original(*args)

    monkeypatch.setattr(rotary, "_rotate", counted)
    with t.no_grad():
        expected = model(**inputs)
        with scoring.fused_scoring_mlps(model):
            actual = model(**inputs)
        t.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == (2 if _plain_native(kind) else 0)
    calls.clear()
    x = inputs["hidden_states"]
    expected_gradient = t.autograd.grad(model(**inputs)[0].float().sum(), x)[0]
    with scoring.fused_scoring_mlps(model):
        actual_gradient = t.autograd.grad(model(**inputs)[0].float().sum(), x)[0]
    assert not calls
    _assert_bits(actual_gradient, expected_gradient)
    assert "forward" not in model.__dict__


@pytest.mark.parametrize("kind", KINDS)
def test_native_attention_cache_updates(kind: Any) -> None:
    _require_cuda()
    model = _attention(kind).cuda().bfloat16()
    expected_cache = DynamicCache(config=model.config)
    actual_cache = DynamicCache(config=model.config)
    with t.no_grad():
        for _ in range(2):
            inputs = _inputs(model, device="cuda", dtype=t.bfloat16)
            expected = model(**inputs, past_key_values=expected_cache)
            with scoring.fused_scoring_mlps(model):
                actual = model(**inputs, past_key_values=actual_cache)
            t.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert isinstance(actual_cache.layers[0], CacheLayerMixin)
            assert isinstance(expected_cache.layers[0], CacheLayerMixin)
            t.testing.assert_close(
                actual_cache.layers[0].keys, expected_cache.layers[0].keys, rtol=0, atol=0
            )
            t.testing.assert_close(
                actual_cache.layers[0].values, expected_cache.layers[0].values, rtol=0, atol=0
            )


@pytest.mark.very_slow
def test_offsets_beyond_signed_int32() -> None:
    _require_cuda()
    assert rotary is not None
    if t.cuda.mem_get_info()[0] < 11 * 1024**3:
        pytest.skip("requires 11 GiB free for two buffers exceeding 2**31 BF16 elements")
    # Large heads keep the coefficients small while crossing the flat-offset
    # boundary. Compare complete sampled heads, including the final head.
    batch, sequence, heads, width = 2, 2, 8193, 65536
    with t.no_grad():
        x = (
            t.empty(batch, sequence, heads, width, device="cuda", dtype=t.bfloat16)
            .uniform_(-2, 2)
            .transpose(1, 2)
        )
        assert x.numel() > 2**31
        cos = t.full((1, sequence, width), 0.70703125, device="cuda", dtype=t.bfloat16)
        sin = t.full_like(cos, -0.70703125)
        selected = [0, 4096, 8192]
        sample = x[:, selected].clone()
        expected = apply_rotary_pos_emb(sample, sample, cos, sin)[0]
        output = rotary._rotate(x, cos, sin)
        t.cuda.synchronize()
        t.testing.assert_close(output[:, selected], expected, rtol=0, atol=0)
        t.testing.assert_close(x[:, selected], sample, rtol=0, atol=0)
