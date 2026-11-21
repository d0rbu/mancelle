"""Native-rounded BF16 rotary arithmetic for GCG candidate scoring."""

from __future__ import annotations

from typing import Any, cast

import torch as t
import triton
import triton.language as tl
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import is_in_torch_dispatch_mode


@triton.jit
def _rotary_kernel(
    x,
    cos,
    sin,
    output,
    count,
    sequence,
    heads: tl.constexpr,
    width: tl.constexpr,
    broadcast_batch: tl.constexpr,
    block: tl.constexpr,
):
    # Projection storage is [batch, sequence, heads, width], exposed to the
    # attention implementation as a [batch, heads, sequence, width] view.
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < count
    column = index % width
    token = (index // (heads * width)).to(tl.int32)
    coefficient = (token % sequence if broadcast_batch else token).to(tl.int64) * width + column
    rotated_index = index + tl.where(column < width // 2, width // 2, -(width // 2))
    value = tl.load(x + index, mask=valid, other=0).to(tl.float32)
    rotated = tl.load(x + rotated_index, mask=valid, other=0).to(tl.float32)
    rotated = tl.where(column < width // 2, -rotated, rotated)
    cosine = tl.load(cos + coefficient, mask=valid, other=0).to(tl.float32)
    sine = tl.load(sin + coefficient, mask=valid, other=0).to(tl.float32)
    first = (
        tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[value, cosine],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    second = (
        tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[rotated, sine],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    # Eager Torch rounds each product before the addition. Explicit non-FTZ
    # instructions also preserve FP32 subnormals used by BF16 arithmetic.
    result = tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[first, second],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output + index, result, mask=valid)


def _projection_layout(value: t.Tensor) -> bool:
    _, heads, sequence, width = value.shape
    return value.stride() == (sequence * heads * width, width, heads * width, 1)


def _rotate(value: t.Tensor, cos: t.Tensor, sin: t.Tensor) -> t.Tensor:
    output = t.empty_like(value)
    with t.cuda.device(value.device):
        kernel = cast(Any, _rotary_kernel)
        kernel[(triton.cdiv(value.numel(), 1024),)](
            value,
            cos,
            sin,
            output,
            value.numel(),
            value.shape[2],
            heads=value.shape[1],
            width=value.shape[3],
            broadcast_batch=cos.shape[0] == 1,
            block=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def rotary_pair(
    q: t.Tensor,
    k: t.Tensor,
    cos: t.Tensor,
    sin: t.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[t.Tensor, t.Tensor] | None:
    """Return fresh native-layout outputs, or None to request native fallback."""
    values = (q, k, cos, sin)
    if (
        t.is_grad_enabled()
        or t.is_autocast_enabled()
        or t.jit.is_tracing()
        or _get_current_function_mode() is not None
        or is_in_torch_dispatch_mode()
        or t.version.hip is not None
        or type(unsqueeze_dim) is not int
        or unsqueeze_dim != 1
        or any(
            type(value) is not t.Tensor
            or not value.is_cuda
            or value.dtype != t.bfloat16
            or value.layout != t.strided
            or value.is_nested
            or value.is_neg()
            or value.is_conj()
            for value in values
        )
        or any(value.device != q.device for value in values)
        or q.ndim != 4
        or k.ndim != 4
        or cos.ndim != 3
        or sin.shape != cos.shape
        # Singleton dimensions can change TensorIterator's output strides.
        or min(q.shape) <= 1
        or min(k.shape) <= 1
        or q.shape[0] != k.shape[0]
        or q.shape[2:] != k.shape[2:]
        or q.shape[3] % 2
        or cos.shape[0] not in (1, q.shape[0])
        or cos.shape[1:] != q.shape[2:]
        or q.shape[0] * q.shape[2] >= 2**31
        or not cos.is_contiguous()
        or not sin.is_contiguous()
        or not _projection_layout(q)
        or not _projection_layout(k)
        or t.cuda.get_device_capability(q.device)[0] < 8
    ):
        return None
    return _rotate(q, cos, sin), _rotate(k, cos, sin)
