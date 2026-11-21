"""Byte-preserving compaction for admitted expandable FP8 CUDA caches."""

from __future__ import annotations

from itertools import pairwise
from typing import Any, cast

import torch as t
import triton
import triton.language as tl

_BLOCK_BYTES = 4096


@triton.jit
def _compact_layer_rows_kernel(
    k_bytes,
    v_bytes,
    keep_rows,
    rows,
    row_bytes,
    block_bytes: tl.constexpr,
):
    # The same coordinate remains in the same CTA/lane throughout the loop.
    offsets = tl.program_id(0).to(tl.int64) * block_bytes + tl.arange(0, block_bytes)
    mask = offsets < row_bytes
    for destination in tl.range(
        0,
        rows,
        num_stages=1,
        loop_unroll_factor=1,
        warp_specialize=False,
        disable_licm=True,
    ):
        source = tl.load(keep_rows + destination)
        if source != destination:
            source_offsets = source * row_bytes + offsets
            destination_offsets = destination.to(tl.int64) * row_bytes + offsets
            k = tl.load(k_bytes + source_offsets, mask=mask, other=0)
            v = tl.load(v_bytes + source_offsets, mask=mask, other=0)
            tl.store(k_bytes + destination_offsets, k, mask=mask)
            tl.store(v_bytes + destination_offsets, v, mask=mask)


def _admitted(cache: Any, keep_rows: list[int], *, cache_type: type[Any]) -> bool:
    if type(cache) is not cache_type or not t.is_inference_mode_enabled():
        return False
    dimensions = tuple(
        getattr(cache, name, None)
        for name in ("batch_size", "max_seq_len", "n_heads", "head_dim", "n_layers")
    )
    if any(type(value) is not int or value <= 0 for value in dimensions):
        return False
    if cache.batch_size > 2147483647 or type(keep_rows) is not list or not keep_rows:
        return False
    if any(type(row) is not int for row in keep_rows):
        return False
    if (
        keep_rows[0] < 0
        or keep_rows[-1] >= cache.batch_size
        or any(a >= b for a, b in pairwise(keep_rows))
    ):
        return False
    if len(keep_rows) == cache.batch_size:
        return False  # Preserve even direct-method no-op view replacement.
    if (
        type(getattr(cache, "k_layers", None)) is not list
        or type(getattr(cache, "v_layers", None)) is not list
    ):
        return False
    if len(cache.k_layers) != cache.n_layers or len(cache.v_layers) != cache.n_layers:
        return False
    positions = getattr(cache, "cache_seqlens", None)
    if (
        type(positions) is not t.Tensor
        or not positions.is_cuda
        or positions.device.index is None
        or positions.dtype != t.int32
        or positions.layout != t.strided
        or tuple(positions.shape) != (cache.batch_size,)
        or not positions.is_contiguous()
        or positions.is_neg()
        or positions.is_conj()
    ):
        return False
    device = positions.device
    shape = dimensions[:4]
    row_bytes = cache.max_seq_len * cache.n_heads * cache.head_dim
    if row_bytes > 2147483647 or cache.batch_size * row_bytes > 9223372036854775807:
        return False
    tensors = [positions, *cache.k_layers, *cache.v_layers]
    for tensor in tensors[1:]:
        if (
            type(tensor) is not t.Tensor
            or tensor.device != device
            or tensor.dtype != t.float8_e4m3fn
            or tensor.layout != t.strided
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.is_neg()
            or tensor.is_conj()
        ):
            return False
    # Compare complete backing ranges, including hidden/dropped rows and offsets.
    ranges = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        start = storage.data_ptr()
        ranges.append((start, start + storage.nbytes()))
    ranges.sort()
    if any(left[1] > right[0] for left, right in pairwise(ranges)):
        return False
    return t.cuda.get_device_capability(device) == (8, 9)


def try_compact_rows(cache: Any, keep_rows: list[int], *, cache_type: type[Any]) -> bool:
    """Return whether compaction completed in the admitted CUDA domain.

    Unsupported metadata returns False before mutation, allowing the caller's
    original compaction body to run. Admitted allocation and launch errors
    propagate. Contiguous nonzero offsets retain their original backing storage.
    """
    if not _admitted(cache, keep_rows, cache_type=cache_type):
        return False

    indices = t.tensor(keep_rows, dtype=t.int64, device=cache.cache_seqlens.device)
    row_bytes = cache.max_seq_len * cache.n_heads * cache.head_dim
    kernel = cast(Any, _compact_layer_rows_kernel)
    for layer_idx in range(cache.n_layers):
        k_cache = cache.k_layers[layer_idx]
        v_cache = cache.v_layers[layer_idx]
        kernel[(triton.cdiv(row_bytes, _BLOCK_BYTES),)](
            k_cache.view(t.uint8),
            v_cache.view(t.uint8),
            indices,
            len(keep_rows),
            row_bytes,
            block_bytes=_BLOCK_BYTES,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            enable_reflect_ftz=False,
        )
        cache.k_layers[layer_idx] = k_cache[: len(keep_rows)]
        cache.v_layers[layer_idx] = v_cache[: len(keep_rows)]
    for packed_row, source_row in enumerate(keep_rows):
        if packed_row != source_row:
            cache.cache_seqlens[packed_row].copy_(cache.cache_seqlens[source_row])
    cache.cache_seqlens = cache.cache_seqlens[: len(keep_rows)]
    cache.batch_size = len(keep_rows)
    return True


__all__ = ["try_compact_rows"]
