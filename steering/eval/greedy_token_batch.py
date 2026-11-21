"""Decoder-only answer selection and exact-prompt greedy batching."""

from __future__ import annotations

import sys
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any, cast

import torch as t
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache, use_calculator
from nanochat.flash_attention import HAS_FA3
from nanochat.gpt import Linear, apply_rotary_emb, norm

if sys.platform == "linux":
    from steering.eval.triton_kv_cache_compaction import try_compact_rows
    from steering.eval.triton_qk_rope_norm import triton_qk_rope_rms_norm
    from steering.eval.triton_ragged_attention import (
        triton_ragged_attention_with_kvcache,
    )
    from steering.eval.triton_relu_squared import triton_relu_squared_
    from steering.eval.triton_residual_blend import triton_residual_blend
else:
    try_compact_rows = None
    triton_qk_rope_rms_norm = None
    triton_ragged_attention_with_kvcache = None
    triton_relu_squared_ = None
    triton_residual_blend = None

_HOST_CHECK_INTERVAL = 16
_STAGED_WEIGHT_HEADROOM_BYTES = 2 * 1024**3
# A 3 GiB reserve admitted a longer 4090 cache but left too little transient
# space for its first layer-pair replacement at the measured growth boundary.
_EXPANDABLE_CACHE_HEADROOM_BYTES = 7 * 1024**3 // 2
_EXPANDABLE_GROWTH_HEADROOM_BYTES = 256 * 1024**2
_PADDED_RAGGED_PREFILL_TOKEN_BUDGET = 4_096
_LINEAR_FORWARD_OVERRIDE_LOCK = RLock()
_ACTIVE_STAGED_LINEAR_MODEL_IDS: set[int] = set()


class GreedyBatchCapacityError(RuntimeError):
    """Request a smaller greedy batch before cache growth can exceed its budget."""


def _supports_staged_linear_weights(device: t.device) -> bool:
    """Use staged GEMM operands only on measured-profitable CUDA generations."""
    if device.type != "cuda":
        return False
    major, _minor = t.cuda.get_device_capability(device)
    return major < 9


def _supports_expandable_ragged_cache(device: t.device) -> bool:
    """Use expandable row-compacting caches only on measured pre-Hopper CUDA."""
    return not HAS_FA3 and _supports_staged_linear_weights(device)


def _staged_linear_shadow_bytes(model: Any) -> int:
    """Return persistent compute-dtype shadow storage for a decoder model."""
    return sum(
        module.weight.numel() * t.empty((), dtype=COMPUTE_DTYPE).element_size()
        for module in model.modules()
        if isinstance(module, Linear) and module.weight.dtype != COMPUTE_DTYPE
    )


def _projected_kv_cache_bytes(
    model: Any,
    *,
    batch_size: int,
    max_total_len: int,
    ragged: bool,
    dtype: t.dtype,
) -> int:
    """Project the exact K/V storage allocated by greedy token batching."""
    config = model.config
    cache_rows = batch_size + int(ragged)
    head_dim = config.n_embd // config.n_head
    return (
        2
        * config.n_layer
        * cache_rows
        * max_total_len
        * config.n_kv_head
        * head_dim
        * t.empty((), dtype=dtype).element_size()
    )


def _initial_expandable_ragged_cache_len(
    model: Any,
    *,
    batch_size: int,
    max_prompt_len: int,
    max_total_len: int,
    dtype: t.dtype,
) -> int | None:
    """Choose one safe initial allocation for a later row-compacting growth."""
    device = model.get_device()
    if not _supports_expandable_ragged_cache(device):
        return None
    free_bytes, _total_bytes = t.cuda.mem_get_info(device)
    available_cache_bytes = (
        free_bytes
        - _staged_linear_shadow_bytes(model)
        - _EXPANDABLE_CACHE_HEADROOM_BYTES
    )
    bytes_per_token = _projected_kv_cache_bytes(
        model,
        batch_size=batch_size,
        max_total_len=1,
        ragged=True,
        dtype=dtype,
    )
    initial_len = min(max_total_len, available_cache_bytes // bytes_per_token)
    if initial_len <= max_prompt_len:
        return None
    return int(initial_len)


@contextmanager
def _replace_linear_forwards_with_staged_weights(
    linears_and_weights: list[tuple[Linear, t.Tensor]],
) -> Iterator[None]:
    """Temporarily read GEMM operands from persistent compute-dtype weights."""
    weights_by_parameter_id = {
        id(linear.weight): compute_weight for linear, compute_weight in linears_and_weights
    }
    with _LINEAR_FORWARD_OVERRIDE_LOCK:
        original_forward = Linear.forward

        def staged_forward(linear: Linear, x: t.Tensor) -> t.Tensor:
            try:
                staged_weight = weights_by_parameter_id[id(linear.weight)]
            except KeyError:
                return cast(t.Tensor, original_forward(linear, x))
            staged_bias = linear.bias.to(dtype=x.dtype) if linear.bias is not None else None
            return F.linear(x, staged_weight, staged_bias)

        Linear.forward = staged_forward  # ty: ignore[invalid-assignment]
        try:
            yield
        finally:
            Linear.forward = original_forward


@contextmanager
def _stage_linear_weights_for_greedy_batch(
    model: Any,
    *,
    batch_size: int,
    max_total_len: int,
    ragged: bool,
    dtype: t.dtype,
) -> Iterator[bool]:
    """Stage decoder linears when their shadows and K/V cache fit safely."""
    model_id = id(model)
    with _LINEAR_FORWARD_OVERRIDE_LOCK:
        if model_id in _ACTIVE_STAGED_LINEAR_MODEL_IDS:
            yield True
            return

    device = model.get_device()
    if not _supports_staged_linear_weights(device) or COMPUTE_DTYPE == t.float32:
        yield False
        return

    linears = [module for module in model.modules() if isinstance(module, Linear)]
    shadow_bytes = _staged_linear_shadow_bytes(model)
    projected_cache_bytes = _projected_kv_cache_bytes(
        model,
        batch_size=batch_size,
        max_total_len=max_total_len,
        ragged=ragged,
        dtype=dtype,
    )
    free_bytes, _total_bytes = t.cuda.mem_get_info(device)
    required_bytes = shadow_bytes + projected_cache_bytes + _STAGED_WEIGHT_HEADROOM_BYTES
    if not linears or free_bytes < required_bytes:
        yield False
        return

    # Serialize shadow construction as well as the temporary class override so
    # concurrent greedy batches cannot each reserve a full shadow set.
    with _LINEAR_FORWARD_OVERRIDE_LOCK:
        if model_id in _ACTIVE_STAGED_LINEAR_MODEL_IDS:
            yield True
            return
        linears_and_weights = [
            (linear, linear.weight.detach().to(COMPUTE_DTYPE)) for linear in linears
        ]
        _ACTIVE_STAGED_LINEAR_MODEL_IDS.add(model_id)
        try:
            with _replace_linear_forwards_with_staged_weights(linears_and_weights):
                yield True
        finally:
            _ACTIVE_STAGED_LINEAR_MODEL_IDS.remove(model_id)


@contextmanager
def _stage_linear_weights_for_greedy_batches(
    model: Any,
    *,
    batch_specs: list[tuple[int, int, bool]],
    dtype: t.dtype,
) -> Iterator[bool]:
    """Stage once for several batches using their largest exact K/V projection."""
    if not batch_specs:
        yield False
        return
    batch_size, max_total_len, ragged = max(
        batch_specs,
        key=lambda spec: _projected_kv_cache_bytes(
            model,
            batch_size=spec[0],
            max_total_len=spec[1],
            ragged=spec[2],
            dtype=dtype,
        ),
    )
    with _stage_linear_weights_for_greedy_batch(
        model,
        batch_size=batch_size,
        max_total_len=max_total_len,
        ragged=ragged,
        dtype=dtype,
    ) as staged:
        yield staged


def predict_categorical_token_batch(
    model: Any,
    *,
    prompt_token_ids: list[list[int]],
    answer_token_ids: list[list[int]],
    pad_token_id: int,
) -> list[int]:
    """Return answer-choice indices without projecting logits at other positions."""
    if getattr(model, "is_target_aware_encdec", False):
        raise ValueError("categorical token batching requires a decoder-only model")
    if not prompt_token_ids:
        return []
    if len(prompt_token_ids) != len(answer_token_ids):
        raise ValueError("categorical prompts and answer-token rows must have equal length")
    if any(not prompt for prompt in prompt_token_ids):
        raise ValueError("categorical prompts must be nonempty")
    if any(not choices for choices in answer_token_ids):
        raise ValueError("categorical answer-token rows must be nonempty")

    device = model.get_device()
    max_length = max(len(prompt) for prompt in prompt_token_ids)
    positions = t.tensor(
        [len(prompt) - 1 for prompt in prompt_token_ids],
        dtype=t.long,
        device=device,
    )
    unique_token_ids = sorted({token for choices in answer_token_ids for token in choices})
    selected_token_ids = t.tensor(unique_token_ids, dtype=t.long, device=device)
    batch = t.tensor(
        [prompt + [pad_token_id] * (max_length - len(prompt)) for prompt in prompt_token_ids],
        dtype=t.long,
        device=device,
    )

    with _LINEAR_FORWARD_OVERRIDE_LOCK:
        original_forward = Linear.forward

        def selected_lm_head_forward(linear: Linear, x: t.Tensor) -> t.Tensor:
            if linear is not model.lm_head:
                return cast(t.Tensor, original_forward(linear, x))
            rows = t.arange(x.size(0), device=x.device)
            answer_x = x[rows, positions]
            weight = linear.weight.index_select(0, selected_token_ids).to(x.dtype)
            bias = (
                linear.bias.index_select(0, selected_token_ids).to(x.dtype)
                if linear.bias is not None
                else None
            )
            return F.linear(answer_x, weight, bias).unsqueeze(1)

        Linear.forward = selected_lm_head_forward  # ty: ignore[invalid-assignment]
        try:
            with t.inference_mode():
                logits = model(batch)
        finally:
            Linear.forward = original_forward

    token_to_local = {
        token_id: local_index
        for local_index, token_id in enumerate(unique_token_ids)
    }
    return [
        int(
            logits[row, 0, [token_to_local[token] for token in choices]]
            .argmax()
            .item()
        )
        for row, choices in enumerate(answer_token_ids)
    ]


def _cache_position_device(cache_device: t.device) -> t.device:
    """Keep SDPA-only cache positions off CUDA's synchronization boundary."""
    if cache_device.type == "cuda" and not HAS_FA3:
        return t.device("cpu")
    return cache_device


class _HostPositionKVCache(KVCache):
    """KV cache whose uniform position does not require a device sync."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        num_layers: int,
        device: t.device,
        dtype: t.dtype,
    ) -> None:
        super().__init__(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            num_layers,
            device,
            dtype,
        )
        position_device = _cache_position_device(device)
        if position_device != self.cache_seqlens.device:
            self.cache_seqlens = t.zeros(
                batch_size,
                dtype=t.int32,
                device=position_device,
            )
        self._host_position = 0

    def get_pos(self) -> int:
        return self._host_position

    def reset(self) -> None:
        super().reset()
        self._host_position = 0

    def advance(self, num_tokens: int) -> None:
        self.cache_seqlens += num_tokens
        self._host_position += num_tokens

    def rewind(self, num_tokens: int) -> None:
        self.cache_seqlens -= num_tokens
        self._host_position -= num_tokens

    def prefill(self, other: KVCache) -> None:
        super().prefill(other)
        self._host_position = int(other.get_pos())


class _ExpandableRaggedKVCache:
    """Layerwise K/V storage that grows once after finished rows are packed."""

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        final_seq_len: int,
        head_dim: int,
        num_layers: int,
        device: t.device,
        dtype: t.dtype,
    ) -> None:
        if seq_len > final_seq_len:
            raise ValueError(
                f"initial cache length {seq_len} exceeds final length {final_seq_len}"
            )
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.final_seq_len = final_seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self._initial_row_token_budget = batch_size * seq_len
        shape = (batch_size, seq_len, num_heads, head_dim)
        self.k_layers = [
            t.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.v_layers = [
            t.zeros(shape, device=device, dtype=dtype) for _ in range(num_layers)
        ]
        self.cache_seqlens = t.zeros(batch_size, dtype=t.int32, device=device)

    def get_layer_cache(self, layer_idx: int) -> tuple[t.Tensor, t.Tensor]:
        return self.k_layers[layer_idx], self.v_layers[layer_idx]

    def advance(self, num_tokens: int) -> None:
        self.cache_seqlens += num_tokens

    def compact(self, keep_rows: list[int]) -> None:
        if try_compact_rows is not None and try_compact_rows(
            self, keep_rows, cache_type=_ExpandableRaggedKVCache
        ):
            return
        for layer_idx in range(self.n_layers):
            k_cache = self.k_layers[layer_idx]
            v_cache = self.v_layers[layer_idx]
            for packed_row, source_row in enumerate(keep_rows):
                if packed_row == source_row:
                    continue
                k_cache[packed_row].copy_(k_cache[source_row])
                v_cache[packed_row].copy_(v_cache[source_row])
            self.k_layers[layer_idx] = k_cache[: len(keep_rows)]
            self.v_layers[layer_idx] = v_cache[: len(keep_rows)]
        for packed_row, source_row in enumerate(keep_rows):
            if packed_row != source_row:
                self.cache_seqlens[packed_row].copy_(self.cache_seqlens[source_row])
        self.cache_seqlens = self.cache_seqlens[: len(keep_rows)]
        self.batch_size = len(keep_rows)

    def ensure_capacity(self, cache_end: int) -> None:
        if cache_end <= self.max_seq_len:
            return
        if cache_end > self.final_seq_len:
            raise ValueError(
                f"cache end {cache_end} exceeds final length {self.final_seq_len}"
            )
        if self.batch_size * self.final_seq_len > self._initial_row_token_budget:
            raise GreedyBatchCapacityError(
                "expandable greedy cache needs a smaller batch: "
                f"rows={self.batch_size}, final_len={self.final_seq_len}, "
                f"initial_row_token_budget={self._initial_row_token_budget}"
            )

        old_capacity = self.max_seq_len
        if self.cache_seqlens.device.type == "cuda":
            element_size = self.k_layers[0].element_size()
            new_layer_pair_bytes = (
                2
                * self.batch_size
                * self.final_seq_len
                * self.n_heads
                * self.head_dim
                * element_size
            )
            free_bytes, _total_bytes = t.cuda.mem_get_info(self.cache_seqlens.device)
            if free_bytes < new_layer_pair_bytes + _EXPANDABLE_GROWTH_HEADROOM_BYTES:
                raise GreedyBatchCapacityError(
                    "expandable greedy cache lacks growth headroom: "
                    f"free_bytes={free_bytes}, layer_pair_bytes={new_layer_pair_bytes}"
                )

        grown_layers: list[int] = []
        try:
            for layer_idx in range(self.n_layers):
                old_k = self.k_layers[layer_idx]
                old_v = self.v_layers[layer_idx]
                shape = (
                    self.batch_size,
                    self.final_seq_len,
                    self.n_heads,
                    self.head_dim,
                )
                new_k = t.empty(shape, dtype=old_k.dtype, device=old_k.device)
                new_v = t.empty(shape, dtype=old_v.dtype, device=old_v.device)
                new_k[:, :old_capacity].copy_(old_k)
                new_v[:, :old_capacity].copy_(old_v)
                self.k_layers[layer_idx] = new_k
                self.v_layers[layer_idx] = new_v
                grown_layers.append(layer_idx)
        except t.OutOfMemoryError as error:
            for layer_idx in grown_layers:
                self.k_layers[layer_idx] = self.k_layers[layer_idx][:, :old_capacity]
                self.v_layers[layer_idx] = self.v_layers[layer_idx][:, :old_capacity]
            raise GreedyBatchCapacityError(
                "expandable greedy cache allocation failed; retry with a smaller batch"
            ) from error
        self.max_seq_len = self.final_seq_len


def _compact_kv_cache_rows(
    cache: KVCache | _ExpandableRaggedKVCache,
    keep_rows: list[int],
) -> None:
    """Pack selected batch rows in place without reallocating cache storage."""
    if len(keep_rows) == cache.batch_size:
        return
    if not keep_rows:
        raise ValueError("cannot compact a KV cache to zero rows")
    if keep_rows != sorted(keep_rows) or len(set(keep_rows)) != len(keep_rows):
        raise ValueError("KV cache rows must be unique and increasing")
    if keep_rows[0] < 0 or keep_rows[-1] >= cache.batch_size:
        raise IndexError("KV cache row is out of range")

    if isinstance(cache, _ExpandableRaggedKVCache):
        cache.compact(keep_rows)
        return

    for packed_row, source_row in enumerate(keep_rows):
        if packed_row == source_row:
            continue
        cache.k_cache[:, packed_row].copy_(cache.k_cache[:, source_row])
        cache.v_cache[:, packed_row].copy_(cache.v_cache[:, source_row])
        cache.cache_seqlens[packed_row].copy_(cache.cache_seqlens[source_row])
    cache.k_cache = cache.k_cache[:, : len(keep_rows)]
    cache.v_cache = cache.v_cache[:, : len(keep_rows)]
    cache.cache_seqlens = cache.cache_seqlens[: len(keep_rows)]
    cache.batch_size = len(keep_rows)


def _should_compact_finished_rows(device: t.device) -> bool:
    """Use live-row packing only where smaller decode batches are profitable."""
    if HAS_FA3:
        return False
    if device.type != "cuda":
        return True
    major, _minor = t.cuda.get_device_capability(device)
    return major < 9


def _should_use_triton_residual_blend(
    x0: t.Tensor,
    resid_lambdas: t.Tensor,
    x0_lambdas: t.Tensor,
    *,
    num_layers: int,
) -> bool:
    """Admit the mixed-precision decode blend on measured Ada devices."""
    if (
        triton_residual_blend is None
        or not x0.is_cuda
        or x0.dtype != t.float32
        or not x0.is_contiguous()
        or COMPUTE_DTYPE not in (t.bfloat16, t.float16)
    ):
        return False
    for scales in (resid_lambdas, x0_lambdas):
        if (
            scales.device != x0.device
            or scales.dtype != t.float32
            or scales.ndim != 1
            or scales.numel() != num_layers
        ):
            return False
    return t.cuda.get_device_capability(x0.device) == (8, 9)


def _should_use_triton_qk_rope_norm(
    q: t.Tensor,
    k: t.Tensor,
    cos: t.Tensor,
    sin: t.Tensor,
) -> bool:
    """Fuse decode Q/K RoPE and RMSNorm only on the measured Ada path."""
    if (
        triton_qk_rope_rms_norm is None
        or not q.is_cuda
        or q.ndim != 4
        or k.ndim != 4
        or q.dtype not in (t.bfloat16, t.float16)
        or k.device != q.device
        or cos.device != q.device
        or sin.device != q.device
        or k.dtype != q.dtype
        or cos.dtype != q.dtype
        or sin.dtype != q.dtype
        or q.size(0) != k.size(0)
        or q.size(1) != 1
        or k.size(1) != 1
        or q.size(-1) != k.size(-1)
        or not 2 <= q.size(-1) <= 256
        or q.size(-1) % 2 != 0
        or cos.shape != (q.size(0), 1, 1, q.size(-1) // 2)
        or sin.shape != cos.shape
        or q.stride(-1) != 1
        or k.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
    ):
        return False
    return t.cuda.get_device_capability(q.device) == (8, 9)


def _should_use_triton_ragged_attention(q: t.Tensor, k_cache: t.Tensor) -> bool:
    """Use the measured fused decode kernel only on Ampere/Ada CUDA."""
    if (
        triton_ragged_attention_with_kvcache is None
        or not q.is_cuda
        or k_cache.dtype not in (t.bfloat16, t.float16, t.float8_e4m3fn)
        or not 0 < q.size(-1) <= 256
    ):
        return False
    major, _minor = t.cuda.get_device_capability(q.device)
    return major == 8


def _ragged_sdpa_with_kvcache(
    q: t.Tensor,
    k_cache: t.Tensor,
    v_cache: t.Tensor,
    k: t.Tensor,
    v: t.Tensor,
    cache_positions: t.Tensor,
    batch_indices: t.Tensor,
    attn_mask: t.Tensor,
    cache_end: int,
) -> t.Tensor:
    """Update and attend to a KV cache at one distinct position per row."""
    batch_size, width, _heads, _head_dim = q.shape
    if width != 1:
        raise ValueError(f"ragged SDPA requires width 1, got {width}")
    if cache_end < 1 or cache_end > k_cache.size(1):
        raise ValueError(
            f"ragged SDPA cache end must be in [1, {k_cache.size(1)}], got {cache_end}"
        )
    if cache_positions.shape != (batch_size,):
        raise ValueError(
            f"ragged SDPA cache position shape must be {(batch_size,)}, "
            f"got {cache_positions.shape}"
        )
    if batch_indices.shape != (batch_size,):
        raise ValueError(
            f"ragged SDPA batch index shape must be {(batch_size,)}, got {batch_indices.shape}"
        )
    if attn_mask.shape != (batch_size, cache_end):
        raise ValueError(
            f"ragged SDPA mask shape must be {(batch_size, cache_end)}, got {attn_mask.shape}"
        )

    if _should_use_triton_ragged_attention(q, k_cache):
        assert triton_ragged_attention_with_kvcache is not None
        return triton_ragged_attention_with_kvcache(
            q,
            k_cache,
            v_cache,
            k,
            v,
            cache_positions,
            attn_mask,
            cache_end,
        )

    k_cache = k_cache[:, :cache_end]
    v_cache = v_cache[:, :cache_end]
    k_cache[batch_indices, cache_positions] = k[:, 0].to(k_cache.dtype)
    v_cache[batch_indices, cache_positions] = v[:, 0].to(v_cache.dtype)

    # QK normalization can promote the one-token query. When the cache already
    # uses the kernel compute dtype, cast that query instead of round-tripping
    # the full K/V cache through the promoted dtype and autocast.
    if k_cache.dtype == COMPUTE_DTYPE:
        q_sdpa = q.transpose(1, 2).to(k_cache.dtype)
        k_sdpa = k_cache.transpose(1, 2)
        v_sdpa = v_cache.transpose(1, 2)
    else:
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k_cache.transpose(1, 2).to(q.dtype)
        v_sdpa = v_cache.transpose(1, 2).to(q.dtype)
    output = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        attn_mask=attn_mask[:, None, None, :],
        enable_gqa=q_sdpa.size(1) != k_sdpa.size(1),
    )
    return output.transpose(1, 2)


def _build_ragged_sdpa_inputs(
    cache_seqlens: t.Tensor,
    *,
    cache_end: int,
    window_sizes: list[tuple[int, int]],
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, dict[tuple[int, int], t.Tensor]]:
    """Build shared cache indices and one mask per distinct attention window."""
    cache_positions = cache_seqlens.to(device=device, dtype=t.long)
    batch_size = cache_positions.numel()
    batch_indices = t.arange(batch_size, device=device)
    key_positions = t.arange(cache_end, device=device)
    prefix_mask = key_positions.unsqueeze(0) < (cache_positions + 1).unsqueeze(1)
    masks: dict[tuple[int, int], t.Tensor] = {}
    for window_size in dict.fromkeys(window_sizes):
        window_left, window_right = window_size
        if window_right != 0:
            raise ValueError(f"ragged SDPA requires right window 0, got {window_size}")
        if window_left < 0 or cache_end <= window_left + 1:
            masks[window_size] = prefix_mask
            continue
        first_positions = (cache_positions - window_left).clamp_min(0)
        masks[window_size] = prefix_mask & (
            key_positions.unsqueeze(0) >= first_positions.unsqueeze(1)
        )
    return cache_positions, batch_indices, masks


def _ragged_attention_forward(
    attention: Any,
    x: t.Tensor,
    value_embedding: t.Tensor | None,
    cos_sin: tuple[t.Tensor, t.Tensor],
    window_size: tuple[int, int],
    cache: KVCache | _ExpandableRaggedKVCache,
    cache_end: int,
    cache_positions: t.Tensor,
    batch_indices: t.Tensor | None,
    attn_mask: t.Tensor | None,
) -> t.Tensor:
    """Route one ragged attention step through native FA3 or mirrored SDPA."""
    if HAS_FA3:
        return attention(x, value_embedding, cos_sin, window_size, cache)
    if batch_indices is None or attn_mask is None:
        raise RuntimeError("ragged SDPA inputs were not prepared for the fallback path")

    batch_size, width, _channels = x.size()
    qkv = attention.c_qkv(x)
    q, k, v = qkv.split(
        [attention.q_dim, attention.kv_dim, attention.kv_dim],
        dim=-1,
    )
    q = q.view(batch_size, width, attention.n_head, attention.head_dim)
    k = k.view(batch_size, width, attention.n_kv_head, attention.head_dim)
    v = v.view(batch_size, width, attention.n_kv_head, attention.head_dim)

    if value_embedding is not None:
        value_embedding = value_embedding.view(
            batch_size,
            width,
            attention.n_kv_head,
            attention.head_dim,
        )
        gate = 2 * t.sigmoid(attention.ve_gate(x[..., : attention.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * value_embedding

    cos, sin = cos_sin
    if _should_use_triton_qk_rope_norm(q, k, cos, sin):
        assert triton_qk_rope_rms_norm is not None
        q, k = triton_qk_rope_rms_norm(q, k, cos, sin)
    else:
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
    k_cache, v_cache = cache.get_layer_cache(attention.layer_idx)
    y = _ragged_sdpa_with_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_positions,
        batch_indices,
        attn_mask,
        cache_end,
    )
    if attention.layer_idx == cache.n_layers - 1:
        cache.advance(width)
    return attention.c_proj(y.contiguous().view(batch_size, width, -1))


def _ragged_forward_one(
    model: Any,
    idx: t.Tensor,
    cache: KVCache | _ExpandableRaggedKVCache,
    *,
    cache_end: int,
) -> t.Tensor:
    """Run one decode token with a distinct RoPE/cache position per row."""
    batch_size, width = idx.shape
    if width != 1:
        raise ValueError(f"ragged decode requires width 1, got {width}")
    if isinstance(cache, _ExpandableRaggedKVCache):
        cache.ensure_capacity(cache_end)
    if HAS_FA3:
        positions = cache.cache_seqlens.to(dtype=t.long)
        batch_indices = None
        attention_masks = None
    else:
        positions, batch_indices, attention_masks = _build_ragged_sdpa_inputs(
            cache.cache_seqlens,
            cache_end=cache_end,
            window_sizes=model.window_sizes,
            device=idx.device,
        )
    if positions.shape != (batch_size,):
        raise ValueError(f"cache position shape must be {(batch_size,)}, got {positions.shape}")
    cos = model.cos[0].index_select(0, positions).unsqueeze(1)
    sin = model.sin[0].index_select(0, positions).unsqueeze(1)

    x = norm(model.transformer.wte(idx).to(COMPUTE_DTYPE))
    x0 = x
    use_fused_blend = _should_use_triton_residual_blend(
        x0, model.resid_lambdas, model.x0_lambdas, num_layers=len(model.transformer.h),
    )
    positions_before = cache.cache_seqlens.clone()
    try:
        for layer_idx, block in enumerate(model.transformer.h):
            if use_fused_blend:
                assert triton_residual_blend is not None
                x = triton_residual_blend(
                    x, x0, model.resid_lambdas, model.x0_lambdas, layer_idx,
                    output_dtype=COMPUTE_DTYPE,
                )
            else:
                x = model.resid_lambdas[layer_idx] * x + model.x0_lambdas[layer_idx] * x0
                x = x.to(COMPUTE_DTYPE)
            value_embedding = (
                model.value_embeds[str(layer_idx)](idx).to(x.dtype)
                if str(layer_idx) in model.value_embeds
                else None
            )
            x = x + _ragged_attention_forward(
                block.attn,
                norm(x),
                value_embedding,
                (cos, sin),
                model.window_sizes[layer_idx],
                cache,
                cache_end,
                positions,
                batch_indices,
                (
                    None
                    if attention_masks is None
                    else attention_masks[model.window_sizes[layer_idx]]
                ),
            )
            x = x + _greedy_mlp_forward(block.mlp, norm(x))
    except BaseException:
        cache.cache_seqlens.copy_(positions_before)
        raise
    logits = model.lm_head(norm(x))[..., : model.config.vocab_size].float()
    return 15 * t.tanh(logits / 15)


def _should_use_padded_ragged_prefill(device: t.device) -> bool:
    """Batch right-padded prefills only on the measured Ada/Ampere path."""
    if HAS_FA3 or device.type != "cuda":
        return False
    major, _minor = t.cuda.get_device_capability(device)
    return major == 8


def _should_use_triton_relu_squared(x: t.Tensor) -> bool:
    """Fuse the measured Ada MLP activation only for supported CUDA tensors."""
    if (
        triton_relu_squared_ is None
        or not x.is_cuda
        or not x.is_contiguous()
        or x.dtype not in (t.bfloat16, t.float16)
    ):
        return False
    return t.cuda.get_device_capability(x.device) == (8, 9)


def _greedy_mlp_forward(mlp: Any, x: t.Tensor) -> t.Tensor:
    """Mirror the vendored MLP while fusing its measured Ada activation."""
    hidden = mlp.c_fc(x)
    if not _should_use_triton_relu_squared(hidden):
        return mlp.c_proj(F.relu(hidden).square())
    assert triton_relu_squared_ is not None
    return mlp.c_proj(triton_relu_squared_(hidden))


def _should_use_float8_ragged_cache(device: t.device) -> bool:
    """Store ragged K/V in measured E4M3 format only on Ada CUDA."""
    if (
        HAS_FA3
        or triton_ragged_attention_with_kvcache is None
        or device.type != "cuda"
    ):
        return False
    major, minor = t.cuda.get_device_capability(device)
    return (major, minor) == (8, 9)


def _allocate_ragged_prompt_cache(
    model: Any,
    *,
    batch_size: int,
    max_total_len: int,
    dtype: t.dtype,
    initial_cache_len: int | None,
) -> KVCache | _ExpandableRaggedKVCache:
    device = model.get_device()
    config = model.config
    allocation_len = max_total_len if initial_cache_len is None else initial_cache_len
    cache_kwargs = {
        "num_heads": config.n_kv_head,
        "seq_len": allocation_len,
        "head_dim": config.n_embd // config.n_head,
        "num_layers": config.n_layer,
        "device": device,
        "dtype": dtype,
    }
    try:
        if initial_cache_len is None:
            return KVCache(batch_size=batch_size, **cache_kwargs)
        return _ExpandableRaggedKVCache(
            batch_size=batch_size,
            final_seq_len=max_total_len,
            **cache_kwargs,
        )
    except t.OutOfMemoryError as error:
        if initial_cache_len is None:
            raise
        raise GreedyBatchCapacityError(
            "expandable greedy cache allocation failed; retry with a smaller batch"
        ) from error


def _padded_prefill_attention(
    q: t.Tensor,
    k: t.Tensor,
    v: t.Tensor,
    window_size: tuple[int, int],
) -> t.Tensor:
    """Mirror cache-prefill SDPA while ignoring causal-future right padding."""
    width = q.size(2)
    window_left, window_right = window_size
    if window_right != 0:
        raise ValueError(f"padded prefill requires right window 0, got {window_size}")
    enable_gqa = q.size(1) != k.size(1)
    if window_left < 0 or window_left >= width:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
    positions = t.arange(width, device=q.device)
    query_positions = positions[:, None]
    key_positions = positions[None, :]
    mask = (key_positions <= query_positions) & (
        query_positions - key_positions <= window_left
    )
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        enable_gqa=enable_gqa,
    )


def _prefill_padded_prompt_group(
    model: Any,
    prompt_tensor: t.Tensor,
    prompt_lengths: t.Tensor,
    cache: KVCache | _ExpandableRaggedKVCache,
    *,
    batch_row_start: int,
) -> t.Tensor:
    """Prefill one right-padded group and place its K/V rows in the main cache."""
    batch_size, width = prompt_tensor.shape
    cos_sin = (model.cos[:, :width], model.sin[:, :width])
    x = norm(model.transformer.wte(prompt_tensor).to(COMPUTE_DTYPE))
    x0 = x
    for layer_idx, block in enumerate(model.transformer.h):
        x = model.resid_lambdas[layer_idx] * x + model.x0_lambdas[layer_idx] * x0
        x = x.to(COMPUTE_DTYPE)
        value_embedding = (
            model.value_embeds[str(layer_idx)](prompt_tensor).to(x.dtype)
            if str(layer_idx) in model.value_embeds
            else None
        )
        attention = block.attn
        attention_input = norm(x)
        qkv = attention.c_qkv(attention_input)
        q, k, v = qkv.split(
            [attention.q_dim, attention.kv_dim, attention.kv_dim],
            dim=-1,
        )
        q = q.view(batch_size, width, attention.n_head, attention.head_dim)
        k = k.view(batch_size, width, attention.n_kv_head, attention.head_dim)
        v = v.view(batch_size, width, attention.n_kv_head, attention.head_dim)
        if value_embedding is not None:
            value_embedding = value_embedding.view(
                batch_size,
                width,
                attention.n_kv_head,
                attention.head_dim,
            )
            gate = 2 * t.sigmoid(
                attention.ve_gate(attention_input[..., : attention.ve_gate_channels])
            )
            v = v + gate.unsqueeze(-1) * value_embedding
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        k_cache, v_cache = cache.get_layer_cache(layer_idx)
        cache_rows = slice(batch_row_start, batch_row_start + batch_size)
        k_cache[cache_rows, :width].copy_(k.to(k_cache.dtype))
        v_cache[cache_rows, :width].copy_(v.to(v_cache.dtype))
        y = _padded_prefill_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            model.window_sizes[layer_idx],
        ).transpose(1, 2)
        x = x + attention.c_proj(y.contiguous().view(batch_size, width, -1))
        x = x + _greedy_mlp_forward(block.mlp, norm(x))

    rows = t.arange(batch_size, device=prompt_tensor.device)
    final_x = norm(x)[rows, prompt_lengths - 1]
    logits = model.lm_head(final_x)[..., : model.config.vocab_size].float()
    return 15 * t.tanh(logits / 15)


def _prefill_padded_ragged_prompt_batch(
    model: Any,
    prompt_tokens: list[list[int]],
    *,
    max_total_len: int,
    dtype: t.dtype,
    initial_cache_len: int | None,
) -> tuple[KVCache | _ExpandableRaggedKVCache, t.Tensor]:
    """Prefill mixed prompts in bounded right-padded groups on pre-Hopper CUDA."""
    device = model.get_device()
    cache_dtype = t.float8_e4m3fn if _should_use_float8_ragged_cache(device) else dtype
    batch_cache = _allocate_ragged_prompt_cache(
        model,
        batch_size=len(prompt_tokens),
        max_total_len=max_total_len,
        dtype=cache_dtype,
        initial_cache_len=initial_cache_len,
    )
    first_logits = []
    start = 0
    while start < len(prompt_tokens):
        end = start + 1
        width = len(prompt_tokens[start])
        while end < len(prompt_tokens):
            next_width = max(width, len(prompt_tokens[end]))
            if (
                (end - start + 1) * next_width
                > _PADDED_RAGGED_PREFILL_TOKEN_BUDGET
            ):
                break
            width = next_width
            end += 1
        group = prompt_tokens[start:end]
        prompt_lengths = t.tensor(
            [len(prompt) for prompt in group],
            dtype=t.long,
            device=device,
        )
        prompt_tensor = t.tensor(
            [prompt + [0] * (width - len(prompt)) for prompt in group],
            dtype=t.long,
            device=device,
        )
        first_logits.append(
            _prefill_padded_prompt_group(
                model,
                prompt_tensor,
                prompt_lengths,
                batch_cache,
                batch_row_start=start,
            )
        )
        start = end
    batch_cache.cache_seqlens.copy_(
        t.tensor(
            [len(prompt) for prompt in prompt_tokens],
            dtype=t.int32,
            device=batch_cache.cache_seqlens.device,
        )
    )
    return batch_cache, t.cat(first_logits)


def _prefill_ragged_prompt_batch(
    model: Any,
    prompt_tokens: list[list[int]],
    *,
    max_total_len: int,
    dtype: t.dtype,
    initial_cache_len: int | None = None,
) -> tuple[KVCache | _ExpandableRaggedKVCache, t.Tensor]:
    """Prefill exact-length prompt groups and assemble their K/V rows."""
    device = model.get_device()
    config = model.config
    if _should_use_padded_ragged_prefill(device):
        return _prefill_padded_ragged_prompt_batch(
            model,
            prompt_tokens,
            max_total_len=max_total_len,
            dtype=dtype,
            initial_cache_len=initial_cache_len,
        )
    allocation_len = max_total_len if initial_cache_len is None else initial_cache_len
    batch_cache = _allocate_ragged_prompt_cache(
        model,
        batch_size=len(prompt_tokens),
        max_total_len=max_total_len,
        dtype=dtype,
        initial_cache_len=initial_cache_len,
    )
    grouped_prompts: dict[int, list[tuple[int, list[int]]]] = {}
    for row, prompt in enumerate(prompt_tokens):
        grouped_prompts.setdefault(len(prompt), []).append((row, prompt))

    first_logits: list[t.Tensor | None] = [None] * len(prompt_tokens)
    for prompt_len, group in grouped_prompts.items():
        # Keep each exact-length source cache within the single-source row-token
        # envelope already included in admission. Large duplicate groups split.
        source_batch_size = max(1, allocation_len // prompt_len)
        for start in range(0, len(group), source_batch_size):
            source_rows = group[start : start + source_batch_size]
            source_cache = KVCache(
                batch_size=len(source_rows),
                num_heads=config.n_kv_head,
                seq_len=prompt_len,
                head_dim=config.n_embd // config.n_head,
                num_layers=config.n_layer,
                device=device,
                dtype=dtype,
            )
            prompt_tensor = t.tensor(
                [prompt for _row, prompt in source_rows],
                dtype=t.long,
                device=device,
            )
            logits = model.forward(prompt_tensor, kv_cache=source_cache)[:, -1, :]
            for source_row, (batch_row, _prompt) in enumerate(source_rows):
                first_logits[batch_row] = logits[source_row]
                if isinstance(batch_cache, _ExpandableRaggedKVCache):
                    for layer_idx in range(config.n_layer):
                        source_k, source_v = source_cache.get_layer_cache(layer_idx)
                        batch_k, batch_v = batch_cache.get_layer_cache(layer_idx)
                        batch_k[batch_row, :prompt_len].copy_(
                            source_k[source_row, :prompt_len]
                        )
                        batch_v[batch_row, :prompt_len].copy_(
                            source_v[source_row, :prompt_len]
                        )
                else:
                    batch_cache.k_cache[:, batch_row, :prompt_len].copy_(
                        source_cache.k_cache[:, source_row, :prompt_len]
                    )
                    batch_cache.v_cache[:, batch_row, :prompt_len].copy_(
                        source_cache.v_cache[:, source_row, :prompt_len]
                    )
                batch_cache.cache_seqlens[batch_row] = prompt_len
    if any(logit is None for logit in first_logits):
        raise RuntimeError("ragged prefill did not produce every prompt logit")
    return batch_cache, t.stack(
        [logit for logit in first_logits if logit is not None]
    )


@t.inference_mode()
def generate_greedy_token_batch(
    model: Any,
    tokenizer: Any,
    prompt_tokens: list[list[int]],
    *,
    max_new_tokens: int,
    enable_python_tools: bool = False,
    ragged_cache_initial_len: int | None = None,
) -> list[list[int]] | None:
    """Generate one greedy completion for each tokenized prompt.

    Equal-length prompts use one ordinary batched prefill. Mixed-length prompts
    are partitioned into exact-length groups, then their K/V rows are assembled
    for ragged SDPA decode with one cache/RoPE position per row. Each source
    group is bounded by the single-source row-token storage already covered by
    cache admission, and no padding tokens enter prompt context. SDPA reads only
    a host-bounded view through the last initialized cache position while the
    allocation remains full sized. The function returns completion tokens only,
    excluding terminal assistant-end/BOS tokens. Greedy tokens, terminal checks,
    and finite-logit checks are transferred to the host in short chunks; the
    returned prefix remains identical even when a terminal token appears inside a chunk. When
    ``enable_python_tools`` is true, every step is inspected and calculator
    output tokens are forced with the same state machine as
    :class:`nanochat.engine.Engine`. Finished rows are removed from the active
    batch at host-check boundaries while results remain in original input
    order. Callers may request a shorter initial mixed-prompt cache and handle
    :class:`GreedyBatchCapacityError` by replaying the same prompts in smaller
    batches if live-row compaction cannot make the final cache fit.

    ``None`` requests an exact sequential restart. This happens if a live row
    emits ``<python_start>`` because calculator state and forced output tokens
    remain owned by :class:`nanochat.engine.Engine`.
    """
    if not prompt_tokens:
        return []
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise TypeError("max_new_tokens must be an int")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not isinstance(enable_python_tools, bool):
        raise TypeError("enable_python_tools must be a bool")
    if ragged_cache_initial_len is not None and (
        not isinstance(ragged_cache_initial_len, int)
        or isinstance(ragged_cache_initial_len, bool)
    ):
        raise TypeError("ragged_cache_initial_len must be an int or None")
    if any(not tokens for tokens in prompt_tokens):
        raise ValueError("prompt token sequences must be non-empty")
    if getattr(model, "is_target_aware_encdec", False):
        raise ValueError("greedy token batching requires a decoder-only model")
    if max_new_tokens == 0:
        return [[] for _ in prompt_tokens]

    device = model.get_device()
    config = model.config
    bos = tokenizer.get_bos_token_id()
    assistant_end = tokenizer.encode_special("<assistant_end>")
    python_start = tokenizer.encode_special("<python_start>")
    python_end = tokenizer.encode_special("<python_end>") if enable_python_tools else None
    output_start = tokenizer.encode_special("<output_start>") if enable_python_tools else None
    output_end = tokenizer.encode_special("<output_end>") if enable_python_tools else None
    prompt_lengths = [len(tokens) for tokens in prompt_tokens]
    for prompt_len in prompt_lengths:
        if prompt_len + max_new_tokens > config.sequence_len:
            raise ValueError(
                f"prompt length ({prompt_len}) + max_new_tokens "
                f"({max_new_tokens}) exceeds context window ({config.sequence_len})"
            )
    max_total_len = max(prompt_lengths) + max_new_tokens

    batch_size = len(prompt_tokens)
    dtype = t.bfloat16 if device.type == "cuda" else t.float32
    ragged = len(set(prompt_lengths)) > 1
    if ragged_cache_initial_len is not None:
        if not ragged:
            raise ValueError("an expandable cache requires mixed-length prompts")
        if not max(prompt_lengths) < ragged_cache_initial_len <= max_total_len:
            raise ValueError(
                "ragged_cache_initial_len must be greater than every prompt length "
                f"and at most {max_total_len}, got {ragged_cache_initial_len}"
            )
    cache_allocation_len = (
        max_total_len
        if ragged_cache_initial_len is None
        else ragged_cache_initial_len
    )
    with _stage_linear_weights_for_greedy_batch(
        model,
        batch_size=batch_size,
        max_total_len=cache_allocation_len,
        ragged=ragged,
        dtype=dtype,
    ):
        if ragged:
            cache, logits = _prefill_ragged_prompt_batch(
                model,
                prompt_tokens,
                max_total_len=max_total_len,
                dtype=dtype,
                initial_cache_len=ragged_cache_initial_len,
            )
        else:
            cache = _HostPositionKVCache(
                batch_size=batch_size,
                num_heads=config.n_kv_head,
                seq_len=max_total_len,
                head_dim=config.n_embd // config.n_head,
                num_layers=config.n_layer,
                device=device,
                dtype=dtype,
            )
            idx = t.tensor(prompt_tokens, dtype=t.long, device=device)
            logits = model.forward(idx, kv_cache=cache)[:, -1, :]
        completions: list[list[int]] = [[] for _ in prompt_tokens]
        finished = [False] * batch_size
        active_rows = list(range(batch_size))
        forced_tokens = [deque[int]() for _ in prompt_tokens]
        in_python_block = [False] * batch_size
        python_expr_tokens: list[list[int]] = [[] for _ in prompt_tokens]
        pending_tokens: list[t.Tensor] = []
        pending_finite: list[t.Tensor] = []
        host_check_interval = 1 if enable_python_tools else _HOST_CHECK_INTERVAL
        compact_finished_rows = _should_compact_finished_rows(device)
        ragged_cache_end = max(prompt_lengths)

        for token_index in range(max_new_tokens):
            pending_finite.append(t.isfinite(logits).all(dim=-1))
            next_ids = t.argmax(logits, dim=-1, keepdim=True)
            pending_tokens.append(next_ids[:, 0])
            check_host = (
                len(pending_tokens) == host_check_interval or token_index + 1 == max_new_tokens
            )
            if check_host:
                token_rows = t.stack(pending_tokens, dim=1).tolist()
                finite_rows = t.stack(pending_finite, dim=1).tolist()
                for offset in range(len(pending_tokens)):
                    for active_row, row in enumerate(active_rows):
                        if finished[row]:
                            continue
                        finite = finite_rows[active_row][offset]
                        if not finite:
                            raise FloatingPointError(
                                "greedy token batch received non-finite logits"
                            )
                    for active_row, row in enumerate(active_rows):
                        if finished[row]:
                            continue
                        sampled_token = token_rows[active_row][offset]
                        token = (
                            forced_tokens[row].popleft() if forced_tokens[row] else sampled_token
                        )
                        token_rows[active_row][offset] = token
                        if token in (assistant_end, bos):
                            finished[row] = True
                            continue
                        completions[row].append(token)
                        if token == python_start:
                            if not enable_python_tools:
                                return None
                            in_python_block[row] = True
                            python_expr_tokens[row] = []
                        elif token == python_end and in_python_block[row]:
                            in_python_block[row] = False
                            if python_expr_tokens[row]:
                                expression = tokenizer.decode(python_expr_tokens[row])
                                result = use_calculator(expression)
                                if result is not None:
                                    assert output_start is not None
                                    assert output_end is not None
                                    forced_tokens[row].append(output_start)
                                    forced_tokens[row].extend(tokenizer.encode(str(result)))
                                    forced_tokens[row].append(output_end)
                            python_expr_tokens[row] = []
                        elif in_python_block[row]:
                            python_expr_tokens[row].append(token)
                    if all(finished):
                        break
                if enable_python_tools:
                    next_ids = t.tensor(
                        [row_tokens[-1] for row_tokens in token_rows],
                        dtype=t.long,
                        device=device,
                    ).unsqueeze(1)
                pending_tokens.clear()
                pending_finite.clear()
                keep_active_rows = [
                    active_row for active_row, row in enumerate(active_rows) if not finished[row]
                ]
                # Smaller live batches help pre-Hopper SDPA. Measured Hopper and FA3
                # throughput is better when the original batch width remains fixed.
                if (
                    compact_finished_rows
                    and keep_active_rows
                    and len(keep_active_rows) != len(active_rows)
                ):
                    _compact_kv_cache_rows(cache, keep_active_rows)
                    keep_tensor = t.tensor(
                        keep_active_rows,
                        dtype=t.long,
                        device=next_ids.device,
                    )
                    next_ids = next_ids.index_select(0, keep_tensor)
                    active_rows = [active_rows[index] for index in keep_active_rows]
            if all(finished) or token_index + 1 == max_new_tokens:
                break
            if ragged:
                ragged_cache_end += next_ids.size(1)
                logits = _ragged_forward_one(
                    model,
                    next_ids,
                    cache,
                    cache_end=ragged_cache_end,
                )
            else:
                logits = model.forward(next_ids, kv_cache=cache)
            logits = logits[:, -1, :]

        return completions
