"""Fused greedy-decode residual blending with eager intermediate rounding."""

from __future__ import annotations

from typing import Any, cast

import torch as t
import triton
import triton.language as tl


@triton.jit
def _residual_blend_kernel(
    x,
    x0,
    resid_lambdas,
    x0_lambdas,
    output,
    layer_idx,
    resid_stride,
    x0_stride,
    elements,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    x_values = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    x0_values = tl.load(x0 + offsets, mask=mask, other=0.0)
    # A CUDA FP32 scalar multiplied by a lower-precision activation is cast
    # to that activation dtype before opmath; the product is rounded again.
    resid_scale = tl.load(resid_lambdas + layer_idx * resid_stride)
    resid_scale = resid_scale.to(x.dtype.element_ty).to(tl.float32)
    x0_scale = tl.load(x0_lambdas + layer_idx * x0_stride)
    residual = (resid_scale * x_values).to(x.dtype.element_ty).to(tl.float32)
    initial = x0_scale * x0_values
    tl.store(output + offsets, residual + initial, mask=mask)


def triton_residual_blend(
    x: t.Tensor,
    x0: t.Tensor,
    resid_lambdas: t.Tensor,
    x0_lambdas: t.Tensor,
    layer_idx: int,
    *,
    output_dtype: t.dtype,
) -> t.Tensor:
    """Return the eager scalar blend in fresh BF16/FP16 storage.

    The initial normalized embedding ``x0`` remains FP32 under CUDA autocast.
    ``x`` is FP32 on the first layer and uses the compute dtype thereafter.
    Scalars are read from their live CUDA parameter arrays on every call.
    """
    if not x.is_cuda:
        raise ValueError("residual blend requires CUDA activations")
    if x.dtype not in (t.float32, t.bfloat16, t.float16):
        raise TypeError(f"x must use FP32/BF16/FP16, got {x.dtype}")
    if x0.dtype != t.float32:
        raise TypeError(f"x0 must use FP32, got {x0.dtype}")
    if x.ndim < 1 or x.shape != x0.shape:
        raise ValueError("x and x0 must have matching non-scalar shapes")
    if not x.is_contiguous() or not x0.is_contiguous():
        raise ValueError("residual blend requires contiguous activations")
    if output_dtype not in (t.bfloat16, t.float16):
        raise TypeError(f"output_dtype must be BF16/FP16, got {output_dtype}")
    if not isinstance(layer_idx, int) or isinstance(layer_idx, bool):
        raise TypeError("layer_idx must be an int")
    for name, source in (
        ("x0", x0),
        ("resid_lambdas", resid_lambdas),
        ("x0_lambdas", x0_lambdas),
    ):
        if source.device != x.device:
            raise ValueError(f"{name} must share the activation CUDA device")
    for name, scalars in (("resid_lambdas", resid_lambdas), ("x0_lambdas", x0_lambdas)):
        if scalars.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional parameter array")
        if scalars.dtype != t.float32:
            raise TypeError(f"{name} must use FP32, got {scalars.dtype}")
        if not 0 <= layer_idx < scalars.numel():
            raise IndexError(f"layer_idx {layer_idx} is outside {name}")

    output = t.empty_like(x, dtype=output_dtype)
    elements = x.numel()
    if elements == 0:
        return output
    kernel = cast(Any, _residual_blend_kernel)
    kernel[(triton.cdiv(elements, 256),)](
        x,
        x0,
        resid_lambdas,
        x0_lambdas,
        output,
        layer_idx,
        resid_lambdas.stride(0),
        x0_lambdas.stride(0),
        elements,
        block=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["triton_residual_blend"]
