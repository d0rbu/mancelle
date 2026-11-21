"""Exact native mean, revision fallback, allocation bounds, and RMS behavior."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t
from transformers.models.smollm3.modeling_smollm3 import SmolLM3RMSNorm

from scripts.eval.axbench_adv.vendored import gcg_mlp_scoring as scoring

kernel = pytest.importorskip("scripts.eval.axbench_adv.vendored.gcg_rms_kernel")
pytestmark = pytest.mark.slow


def _native(hidden: t.Tensor, weight: t.Tensor, epsilon: float) -> t.Tensor:
    norm = SmolLM3RMSNorm(hidden.shape[-1], eps=epsilon)
    norm.weight = t.nn.Parameter(weight, requires_grad=False)
    return norm(hidden)


def exact(actual: t.Tensor, expected: t.Tensor) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    mask = t.isnan(expected)
    assert t.equal(t.isnan(actual), mask)
    dtype = t.int32 if actual.dtype == t.float32 else t.int16
    assert t.equal(actual.view(dtype)[~mask], expected.view(dtype)[~mask])


@pytest.fixture(autouse=True)
def enforce_revision_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    if not t.cuda.is_available() or not scoring._has_build_prerequisites():
        pytest.skip("requires CUDA, Triton, and compiler/Python headers")
    if t.version.hip is not None or t.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires NVIDIA Ampere or later")
    if t.version.git_version not in kernel._MEAN_REVISIONS:

        def forbidden(_hidden: Any) -> Any:
            raise AssertionError("unqualified Torch revision entered custom reduction")

        monkeypatch.setattr(kernel, "_native_order_variance", forbidden)


def check_variance(hidden: t.Tensor) -> None:
    if t.version.git_version in kernel._MEAN_REVISIONS:
        exact(kernel._native_order_variance(hidden), hidden.float().square().mean(-1, keepdim=True))


@pytest.mark.parametrize("rows", [16, 17, 31, 32, 33, 127, 128, 129, 511, 512, 1836, 3664])
@pytest.mark.parametrize("kind", ["normal", "finite_bits", "rounding"])
def test_variance_and_output(rows: int, kind: str) -> None:
    generator = t.Generator(device="cuda").manual_seed(rows + 401)
    if kind == "normal":
        hidden = t.randn((rows, 2048), generator=generator, device="cuda", dtype=t.bfloat16)
    elif kind == "finite_bits":
        # Bound exponents to keep squares finite while exercising 100 binary
        # orders of magnitude, both signs, and every mantissa field.
        bits = t.randint(0, 256, (rows, 2048), generator=generator, device="cuda", dtype=t.int32)
        exponent = t.randint(75, 176, bits.shape, generator=generator, device="cuda", dtype=t.int32)
        bits = (bits & 127) | ((bits & 128) << 8) | (exponent << 7)
        hidden = bits.to(t.int16).view(t.bfloat16)
    else:
        exponent = t.randint(
            -14, 15, (rows, 2048), generator=generator, device="cuda", dtype=t.int32
        )
        hidden = t.ldexp(t.randn(exponent.shape, generator=generator, device="cuda"), exponent).to(
            t.bfloat16
        )
    weight = t.randn((2048,), generator=generator, device="cuda", dtype=t.bfloat16)
    before = hidden.clone()
    weight_before = weight.clone()
    with t.no_grad():
        check_variance(hidden)
        exact(
            kernel.rms_postscale(hidden, weight, 1e-6),
            _native(hidden, weight, 1e-6),
        )
    assert t.equal(before.view(t.int16), hidden.view(t.int16))
    assert t.equal(weight_before.view(t.int16), weight.view(t.int16))


@pytest.mark.parametrize("mode", ["constant_rows", "isolated_values", "mixed_finite"])
def test_all_bf16_encodings(mode: str) -> None:
    values = t.arange(65536, device="cuda", dtype=t.int32).to(t.int16).view(t.bfloat16)
    if mode == "constant_rows":
        hidden = values[:, None].expand(-1, 2048).contiguous()
    elif mode == "isolated_values":
        hidden = t.zeros((65536, 2048), device="cuda", dtype=t.bfloat16)
        indices = t.arange(65536, device="cuda", dtype=t.long)
        hidden[indices, indices % 2048] = values
    else:
        # Keep finite small encodings separate from overflowing and NaN rows.
        bits = t.arange(65536 * 64, device="cuda", dtype=t.int32)
        bits = (bits * 25173 + 13849) & 65535
        exponent = ((bits >> 7) & 255) % 101 + 75
        bits = (bits & 32895) | (exponent << 7)
        hidden = bits.to(t.int16).view(t.bfloat16).reshape(-1, 2048)
    with t.no_grad():
        check_variance(hidden)
        weight = (
            values[:2048].clone()
            if mode == "mixed_finite"
            else t.ones((2048,), device="cuda", dtype=t.bfloat16)
        )
        exact(
            kernel.rms_postscale(hidden, weight, 1e-6),
            _native(hidden, weight, 1e-6),
        )


@pytest.mark.parametrize("offset", [0, 1, 3, 7])
@pytest.mark.parametrize("epsilon", [0, 1e-6, -1e-5, float("inf"), float("nan")])
def test_offsets_epsilon_and_stream(offset: int, epsilon: float) -> None:
    hidden = t.linspace(-1.5, 1.5, 17 * 2048 + offset, device="cuda", dtype=t.bfloat16)[
        offset:
    ].reshape(1, 17, 2048)
    weight = t.linspace(-2.0, 2.0, 2048 + offset, device="cuda", dtype=t.bfloat16)[offset:]
    stream = t.cuda.Stream()
    stream.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(stream), t.inference_mode():
        result = kernel.rms_postscale(hidden, weight, epsilon)
        expected = _native(hidden, weight, epsilon)
    t.cuda.current_stream().wait_stream(stream)
    exact(result, expected)
    assert result.stride() == expected.stride()
    assert result.data_ptr() != hidden.data_ptr()
    assert result.data_ptr() != weight.data_ptr()


@pytest.mark.parametrize(
    "shape", [(0, 2048), (1, 2048), (15, 2048), (16, 1024), (17, 2047), (2, 9, 2049), (2048,)]
)
def test_unsupported_shapes_use_native(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    def forbidden(_hidden: Any) -> Any:
        raise AssertionError("unsupported shape entered specialized reduction")

    monkeypatch.setattr(kernel, "_native_order_variance", forbidden)
    hidden = t.randn(shape, device="cuda", dtype=t.bfloat16)
    weight = t.randn((shape[-1],), device="cuda", dtype=t.bfloat16)
    with t.no_grad():
        exact(
            kernel.rms_postscale(hidden, weight, 1e-6),
            _native(hidden, weight, 1e-6),
        )


@pytest.mark.very_slow
@pytest.mark.parametrize("rows", [2**18 - 1, 2**18, 2**18 + 1])
def test_native_byte_index_boundary(monkeypatch: pytest.MonkeyPatch, rows: int) -> None:
    t.cuda.empty_cache()
    if t.cuda.mem_get_info()[0] < 13 * 2**30:
        pytest.skip("13 GiB free required for real byte-index boundary allocations")
    hidden = t.ones((rows, 2048), device="cuda", dtype=t.bfloat16)
    hidden[0, 0] = -2.0
    hidden[-1, -1] = 3.0
    weight = t.linspace(-1.0, 1.0, 2048, device="cuda", dtype=t.bfloat16)
    original = kernel._native_order_variance
    calls = []

    def checked(value: t.Tensor) -> t.Tensor:
        calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(kernel, "_native_order_variance", checked)
    with t.no_grad():
        actual = kernel.rms_postscale(hidden, weight, 1e-6)
        expected = _native(hidden, weight, 1e-6)
    assert t.equal(actual.view(t.int16), expected.view(t.int16))
    assert calls == (
        [(rows, 2048)] if rows < 2**18 and t.version.git_version in kernel._MEAN_REVISIONS else []
    )


def test_unknown_torch_revision_uses_existing_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(_hidden: Any) -> Any:
        raise AssertionError("unknown revision entered custom reduction")

    monkeypatch.setattr(t.version, "git_version", "unqualified-test-revision")
    monkeypatch.setattr(kernel, "_native_order_variance", forbidden)
    hidden = t.randn((17, 2048), device="cuda", dtype=t.bfloat16)
    weight = t.randn((2048,), device="cuda", dtype=t.bfloat16)
    with t.no_grad():
        exact(kernel.rms_postscale(hidden, weight, 1e-6), _native(hidden, weight, 1e-6))


def test_eligible_shape_preserves_aliased_weights() -> None:
    hidden = t.randn((32, 2048), device="cuda", dtype=t.bfloat16)
    weight = hidden[0]
    before = hidden.clone()
    with t.no_grad():
        exact(kernel.rms_postscale(hidden, weight, 1e-6), _native(hidden, weight, 1e-6))
    assert t.equal(hidden.view(t.int16), before.view(t.int16))
