"""BF16 SiLU/multiply fusion for AxBench-Adv candidate scoring."""

from __future__ import annotations

from typing import Any, cast

import torch as t
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul_kernel(gate, up, output, count, block: tl.constexpr):
    # Large batches can contain more than 2**31 intermediate elements.
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < count
    x = tl.load(gate + index, mask=valid, other=0).to(tl.float32)
    y = tl.load(up + index, mask=valid, other=0).to(tl.float32)
    # Triton 3.1's libdevice div_rn enables flush-to-zero. The eager Torch
    # SiLU preserves subnormals, so request division without that modifier.
    activated = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, 1.0 + libdevice.exp(-x)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # Preserve the rounding between eager SiLU and the separate multiply.
    rounded = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(output + index, rounded * y, mask=valid)


def silu_mul_(gate: t.Tensor, up: t.Tensor) -> t.Tensor:
    """Overwrite an exclusively owned gate projection; keep ``up`` unchanged."""
    if t.is_grad_enabled():
        raise RuntimeError("GCG SiLU fusion requires disabled autograd")
    if (
        not gate.is_cuda
        or gate.device != up.device
        or gate.dtype != t.bfloat16
        or up.dtype != t.bfloat16
        or gate.shape != up.shape
        or not gate.is_contiguous()
        or not up.is_contiguous()
    ):
        raise ValueError("GCG SiLU fusion requires matching contiguous BF16 CUDA operands")
    count = gate.numel()
    if count:
        with t.cuda.device(gate.device):
            kernel = cast(Any, _silu_mul_kernel)
            kernel[(triton.cdiv(count, 1024),)](gate, up, gate, count, block=1024, num_warps=4)
    return gate
