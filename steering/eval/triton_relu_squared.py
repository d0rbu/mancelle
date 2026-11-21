"""In-place fused ReLU-squared activation for measured Ada eval paths."""

from __future__ import annotations

from typing import Any, cast

import torch as t
import triton
import triton.language as tl


@triton.jit
def _relu_squared_in_place_kernel(x, elements, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    values = tl.load(x + offsets, mask=mask)
    values = tl.maximum(values, 0.0)
    tl.store(x + offsets, values * values, mask=mask)


def triton_relu_squared_(x: t.Tensor) -> t.Tensor:
    """Apply ReLU-squared in place to one contiguous CUDA tensor."""
    if not x.is_cuda:
        raise ValueError("Triton ReLU-squared requires a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("Triton ReLU-squared requires contiguous storage")
    if not x.is_floating_point():
        raise TypeError(f"Triton ReLU-squared requires floating point, got {x.dtype}")
    elements = x.numel()
    if elements == 0:
        return x
    kernel = cast(Any, _relu_squared_in_place_kernel)
    kernel[(triton.cdiv(elements, 256),)](
        x,
        elements,
        block=256,
        num_warps=4,
    )
    return x


__all__ = ["triton_relu_squared_"]
