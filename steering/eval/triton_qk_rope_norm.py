"""Fused decode RoPE and Q/K RMSNorm for measured Ada eval paths."""

from __future__ import annotations

from typing import Any, cast

import torch as t
import triton
import triton.language as tl


@triton.jit
def _rope_rms_norm_kernel(
    source,
    cos,
    sin,
    output,
    source_batch_stride,
    source_head_stride,
    cos_batch_stride,
    sin_batch_stride,
    output_batch_stride,
    output_head_stride,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    block: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, block)
    mask = offsets < half_dim
    source_base = source + batch * source_batch_stride + head * source_head_stride
    first = tl.load(source_base + offsets, mask=mask, other=0.0)
    second = tl.load(source_base + half_dim + offsets, mask=mask, other=0.0)
    cos_values = tl.load(
        cos + batch * cos_batch_stride + offsets,
        mask=mask,
        other=0.0,
    )
    sin_values = tl.load(
        sin + batch * sin_batch_stride + offsets,
        mask=mask,
        other=0.0,
    )
    rotated_first = first * cos_values + second * sin_values
    rotated_second = first * (-sin_values) + second * cos_values
    rotated_first = rotated_first.to(output.dtype.element_ty)
    rotated_second = rotated_second.to(output.dtype.element_ty)
    first_f32 = rotated_first.to(tl.float32)
    second_f32 = rotated_second.to(tl.float32)
    mean_square = tl.sum(
        first_f32 * first_f32 + second_f32 * second_f32,
        axis=0,
    ) / head_dim
    inverse_rms = tl.rsqrt(mean_square + 1.1920928955078125e-7)
    output_base = output + batch * output_batch_stride + head * output_head_stride
    tl.store(output_base + offsets, first_f32 * inverse_rms, mask=mask)
    tl.store(
        output_base + half_dim + offsets,
        second_f32 * inverse_rms,
        mask=mask,
    )


def _validate_source(
    source: t.Tensor,
    *,
    batch_size: int,
    head_dim: int,
    name: str,
) -> None:
    if source.ndim != 4 or source.size(0) != batch_size or source.size(1) != 1:
        raise ValueError(
            f"{name} must have shape (batch, 1, heads, {head_dim}), "
            f"got {tuple(source.shape)}"
        )
    if source.size(-1) != head_dim:
        raise ValueError(f"{name} head width must be {head_dim}, got {source.size(-1)}")
    if not source.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if source.dtype not in (t.bfloat16, t.float16):
        raise TypeError(f"{name} must use BF16/FP16, got {source.dtype}")
    if source.stride(-1) != 1:
        raise ValueError(f"{name} final dimension must be contiguous")


def _launch_rope_rms_norm(
    source: t.Tensor,
    cos: t.Tensor,
    sin: t.Tensor,
) -> t.Tensor:
    batch_size, _width, heads, head_dim = source.shape
    output = t.empty_like(source, memory_format=t.contiguous_format)
    half_dim = head_dim // 2
    block = triton.next_power_of_2(half_dim)
    kernel = cast(Any, _rope_rms_norm_kernel)
    kernel[(batch_size, heads)](
        source,
        cos,
        sin,
        output,
        source.stride(0),
        source.stride(2),
        cos.stride(0),
        sin.stride(0),
        output.stride(0),
        output.stride(2),
        head_dim=head_dim,
        half_dim=half_dim,
        block=block,
        num_warps=1,
    )
    return output


def triton_qk_rope_rms_norm(
    q: t.Tensor,
    k: t.Tensor,
    cos: t.Tensor,
    sin: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Apply decode RoPE and RMSNorm to strided Q/K views."""
    if q.ndim != 4:
        raise ValueError(f"q must have four dimensions, got {q.ndim}")
    batch_size = q.size(0)
    head_dim = q.size(-1)
    if head_dim < 2 or head_dim > 256 or head_dim % 2 != 0:
        raise ValueError(
            f"Q/K head width must be even and in [2, 256], got {head_dim}"
        )
    _validate_source(q, batch_size=batch_size, head_dim=head_dim, name="q")
    _validate_source(k, batch_size=batch_size, head_dim=head_dim, name="k")
    if k.device != q.device or k.dtype != q.dtype:
        raise ValueError("q and k must share a device and dtype")
    expected_rope_shape = (batch_size, 1, 1, head_dim // 2)
    if cos.shape != expected_rope_shape or sin.shape != expected_rope_shape:
        raise ValueError(
            f"cos/sin must have shape {expected_rope_shape}, "
            f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if cos.device != q.device or sin.device != q.device:
        raise ValueError("cos/sin must share the Q/K device")
    if cos.dtype != q.dtype or sin.dtype != q.dtype:
        raise TypeError("cos/sin must share the Q/K dtype")
    if cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("cos/sin final dimensions must be contiguous")
    return (
        _launch_rope_rms_norm(q, cos, sin),
        _launch_rope_rms_norm(k, cos, sin),
    )


__all__ = ["triton_qk_rope_rms_norm"]
