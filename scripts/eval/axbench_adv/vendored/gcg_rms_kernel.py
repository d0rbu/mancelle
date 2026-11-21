"""BF16 RMSNorm fusion preserving native reduction and rounding behavior."""

from __future__ import annotations

from typing import Any, cast

import torch as t
import triton
import triton.language as tl

# The native FP32 mean tree is private and changed in later Torch releases.
# Unknown revisions retain the entire previous helper below.
_MEAN_REVISIONS = frozenset(
    {
        "a8d6afb511a69687bbb2b7e88a3cf67917e1697e",  # Torch 2.5.1
        "e2d141dbde55c2a4370fac5165b0561b6af4798b",  # Torch 2.7.1
    }
)


@triton.jit
def _rms_postscale_kernel(
    hidden, inverse, weight, output, count, width: tl.constexpr, block: tl.constexpr
):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < count
    x = tl.load(hidden + index, mask=valid, other=0).to(tl.float32)
    scale = tl.load(inverse + index // width, mask=valid, other=0).to(tl.float32)
    w = tl.load(weight + index % width, mask=valid, other=0).to(tl.float32)
    normalized = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # Eager RMSNorm rounds to BF16 before multiplying by the BF16 weight.
    rounded = normalized.to(tl.bfloat16).to(tl.float32)
    result = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[w, rounded],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output + index, result, mask=valid)


@triton.jit
def _bf16_square_kernel(hidden, output, count, block: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    valid = index < count
    value = tl.load(hidden + index, mask=valid, other=0).to(tl.float32)
    squared = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[value, value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output + index, squared, mask=valid)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _pair_sum(value, block_rows: tl.constexpr, width: tl.constexpr):
    first, second = tl.split(tl.reshape(value, (block_rows, width, 2)))
    return _add_rn(first, second)


@triton.jit
def _square_mean_2048_kernel(hidden, variance, rows, block_rows: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)
    column = tl.arange(0, 128)
    accumulator = tl.full((block_rows, 128), 0, tl.float32)
    # Native Reduce.cuh: 32 lanes, four independent sequential accumulators
    # per lane, 16 vector loads, then (((acc0 + acc1) + acc2) + acc3).
    for step in tl.static_range(16):
        value = tl.load(
            hidden + row[:, None] * 2048 + step * 128 + column[None, :],
            mask=row[:, None] < rows,
            other=0,
        ).to(tl.float32)
        squared = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[value, value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        accumulator = _add_rn(accumulator, squared)
    pairs = tl.reshape(accumulator, (block_rows, 32, 2, 2))
    even, odd = tl.split(pairs)
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    total = _add_rn(_add_rn(_add_rn(a0, a1), a2), a3)
    # These Torch revisions sum warp lanes at distances 1, 2, 4, 8, 16.
    # Lane zero therefore combines adjacent pairs, then pairs of pairs.
    # Explicit splits preserve that order without relying on tl.sum lowering.
    total = _pair_sum(total, block_rows, 16)
    total = _pair_sum(total, block_rows, 8)
    total = _pair_sum(total, block_rows, 4)
    total = _pair_sum(total, block_rows, 2)
    total = _pair_sum(total, block_rows, 1)
    result = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[total, 0.00048828125],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(variance + row, tl.reshape(result, (block_rows,)), mask=row < rows)


def _native_order_variance(hidden: t.Tensor) -> t.Tensor:
    assert t.version.git_version in _MEAN_REVISIONS
    rows = hidden.numel() // 2048
    assert hidden.shape[-1] == 2048 and rows >= 16 and hidden.numel() < 2**29
    variance = t.empty((*hidden.shape[:-1], 1), device=hidden.device, dtype=t.float32)
    with t.cuda.device(hidden.device):
        kernel = cast(Any, _square_mean_2048_kernel)
        kernel[(triton.cdiv(rows, 4),)](
            hidden, variance, rows, 4, num_warps=4, enable_fp_fusion=False
        )
    return variance


def rms_postscale(hidden: t.Tensor, weight: t.Tensor, eps: float) -> t.Tensor:
    """Preserve native reduction arithmetic and operands; specialize one shape."""
    if t.is_grad_enabled():
        raise RuntimeError("GCG RMS fusion requires disabled autograd")
    if type(eps) not in (int, float):
        raise ValueError("GCG RMS fusion requires a numeric scalar epsilon")
    if (
        not hidden.is_cuda
        or hidden.device != weight.device
        or hidden.dtype != t.bfloat16
        or weight.dtype != t.bfloat16
        or hidden.ndim == 0
        or weight.shape != (hidden.shape[-1],)
        or not hidden.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("GCG RMS fusion requires compatible contiguous BF16 CUDA operands")
    count = hidden.numel()
    # For this shape native mean uses one 32-lane warp per row. Keeping the
    # FP32 byte span below 2**31 also avoids TensorIterator splitting.
    if (
        t.version.git_version in _MEAN_REVISIONS
        and hidden.shape[-1] == 2048
        and 16 * 2048 <= count < 2**29
    ):
        variance = _native_order_variance(hidden)
    else:
        squares = t.empty_like(hidden, dtype=t.float32)
        if count:
            with t.cuda.device(hidden.device):
                square_kernel = cast(Any, _bf16_square_kernel)
                square_kernel[(triton.cdiv(count, 1024),)](
                    hidden,
                    squares,
                    count,
                    1024,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        variance = squares.mean(-1, keepdim=True)
        # No later operation reads squares; the allocator preserves stream ordering.
        del squares
    inverse = t.rsqrt(variance + eps)
    output = t.empty_like(hidden)
    if count:
        with t.cuda.device(hidden.device):
            kernel = cast(Any, _rms_postscale_kernel)
            kernel[(triton.cdiv(count, 1024),)](
                hidden,
                inverse,
                weight,
                output,
                count,
                width=hidden.shape[-1],
                block=1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return output
