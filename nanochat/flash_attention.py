"""
Unified Flash Attention interface with automatic FlexAttention/FA3/SDPA switching.

Exports `flash_attn` module that matches a consistent API, selecting the best
available backend at import time.

Backend priority for flash_attn_func:
  1. flex_attention (with FLASH backend on Hopper+/Blackwell, Triton otherwise)
  2. FA3 (Hopper sm90 only)
  3. SDPA (universal fallback)

Backend for flash_attn_with_kvcache (no flex_attention equivalent):
  1. FA3 (Hopper sm90 only)
  2. SDPA (universal fallback)

Layout conventions:
  - flash_attn_func: (B, H, S, D) -- matches flex_attention native layout
  - flash_attn_with_kvcache: (B, T, H, D) -- FA3/SDPA inference layout

Usage:
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache) -- NOTE: (B, H, S, D) layout
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache) -- (B, T, H, D) layout, FA3/SDPA only
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""

import importlib.util
import os
import sys
from enum import StrEnum
from functools import partial
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F
from loguru import logger


# =============================================================================
# FlexAttention imports (PyTorch native, available since torch 2.5+)
# =============================================================================
try:
    from torch.nn.attention.flex_attention import (
        create_block_mask as _create_block_mask,
        flex_attention as _flex_attention,
    )

    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


class FlashAttnImpl(StrEnum):
    """Flash attention backend implementation."""

    FA4 = "fa4"  # flex_attention with FLASH backend on Hopper+/Blackwell
    FA3 = "fa3"  # flash-attention-3 via kernels hub
    SDPA = "sdpa"  # PyTorch scaled_dot_product_attention
    AUTO = "auto"  # auto-detect best available


# =============================================================================
# GPU architecture detection
# =============================================================================
_gpu_arch_major: int = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
)


def _check_cute_dsl_available() -> bool:
    """Check if the CuTe DSL flash_attn package is installed.

    The FLASH backend for flex_attention requires flash_attn.cute (from the
    flash-attn-4 package). Without it, BACKEND="FLASH" raises RuntimeError
    at kernel compile time. We check at import time to avoid this.
    """
    try:
        return importlib.util.find_spec("flash_attn.cute") is not None
    except (ImportError, ModuleNotFoundError):
        return False


# FA4 (FLASH CuTe DSL) backend requires:
#   1. flex_attention available (torch 2.5+)
#   2. Hopper (sm90) or Blackwell (sm100+) GPU
#   3. flash_attn.cute installed (from flash-attn-4 package)
# Without all three, flex_attention still works via the Triton backend.
HAS_FA4_BACKEND: bool = (
    HAS_FLEX_ATTENTION and _gpu_arch_major >= 9 and _check_cute_dsl_available()
)

# Warning about missing FA4 is deferred until after FA3 detection (see below)


# =============================================================================
# Detection: Try to load FA3 on Hopper GPUs (for KV cache path)
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel

        fa3_mod = get_kernel("varunneal/flash-attention-3").flash_attn_interface

        # The FA3 backward pass imports `flash_attn_config` as a top-level
        # module (not a relative import).  When loaded via kernels hub, the
        # package directory isn't on sys.path, so torch.compile fails with
        # ModuleNotFoundError during backward tracing.  Fix: add the FA3
        # package directory to sys.path so the import succeeds.
        fa3_dir = os.path.dirname(fa3_mod.__file__)
        if fa3_dir not in sys.path:
            sys.path.insert(0, fa3_dir)

        return fa3_mod
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as e:
        logger.debug(f"Flash Attention 3 not available: {e}")
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3: bool = _fa3 is not None

# Warn about missing FA4 only if FA3 is also unavailable (i.e. Triton fallback)
if HAS_FLEX_ATTENTION and _gpu_arch_major >= 9 and not HAS_FA4_BACKEND:
    if HAS_FA3:
        logger.info(
            "Hopper/Blackwell GPU: flash-attn-4 (CuTe DSL) not installed, "
            "using FA3 via kernels hub instead."
        )
    else:
        logger.error(
            "Hopper/Blackwell GPU detected but neither flash-attn-4 (CuTe DSL) nor FA3 "
            "is available. Falling back to Triton backend, which is significantly slower. "
            "Install with: uv pip install 'flash-attn-4>=4.0.0b0'"
        )


# =============================================================================
# Override for testing: set to FlashAttnImpl value or raw string, or None (auto)
# Can also be set via NANOCHAT_ATTN_IMPL env var (e.g. "fa3", "sdpa")
# =============================================================================
_env_impl = os.environ.get("NANOCHAT_ATTN_IMPL")
_override_impl: FlashAttnImpl | None = FlashAttnImpl(_env_impl) if _env_impl else None
if _override_impl is not None:
    logger.info(f"Attention implementation override from env: {_override_impl}")


def _active_impl() -> FlashAttnImpl:
    """Return the name of the currently active backend: 'fa4', 'fa3', or 'sdpa'.

    Priority:
      1. FA4 with CuTe DSL backend (fastest on Hopper/Blackwell)
      2. FA3 (precompiled Hopper kernels — faster than Triton-compiled flex)
      3. flex_attention with Triton backend (works everywhere, slow to compile)
      4. SDPA (fallback)
    """
    if _override_impl is not None:
        return _override_impl
    if HAS_FA4_BACKEND:
        return FlashAttnImpl.FA4
    if HAS_FA3:
        return FlashAttnImpl.FA3
    if HAS_FLEX_ATTENTION:
        return FlashAttnImpl.FA4
    return FlashAttnImpl.SDPA


def _use_flex() -> bool:
    """Determine whether to use flex_attention for training."""
    if _override_impl == FlashAttnImpl.FA4:
        assert HAS_FLEX_ATTENTION, (
            "Cannot override to FA4: flex_attention not available"
        )
        return True
    if _override_impl in (FlashAttnImpl.FA3, FlashAttnImpl.SDPA):
        return False
    # Auto: use flex only if we have the fast CuTe backend; otherwise prefer FA3
    if HAS_FA4_BACKEND:
        return True
    if HAS_FA3:
        return False
    return HAS_FLEX_ATTENTION  # Triton flex as last resort before SDPA


def _use_fa3() -> bool:
    """Determine whether to use FA3 based on availability and override."""
    if _override_impl == FlashAttnImpl.FA3:
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl in (FlashAttnImpl.FA4, FlashAttnImpl.SDPA):
        return False
    # auto: flex takes priority over FA3 when both are available
    if HAS_FLEX_ATTENTION:
        return False
    return HAS_FA3


# =============================================================================
# Block sizes per GPU architecture
# =============================================================================
def _get_block_sizes() -> tuple[int, int]:
    """Return (BLOCK_M, BLOCK_N) for create_block_mask based on GPU arch."""
    if _gpu_arch_major >= 10:
        # Blackwell (sm100+): larger M block
        return (256, 128)
    if _gpu_arch_major == 9:
        # Hopper (sm90): balanced blocks
        return (128, 128)
    # Default for other GPUs (Ada sm89, etc.)
    return (128, 128)


_BLOCK_M, _BLOCK_N = _get_block_sizes()


# =============================================================================
# mask_mod functions for flex_attention
# =============================================================================
def _causal_mask(
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Causal mask: each query can only attend to keys at the same or earlier position."""
    return q_idx >= kv_idx


def _make_causal_sliding_mask(window_size: int) -> Callable:
    """Create a causal + sliding window mask_mod function.

    Args:
        window_size: Number of past tokens to attend to (the 'left' window).
                     A query at position q attends to keys in [q - window_size, q].
    """

    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)

    return mask_mod


# =============================================================================
# BlockMask cache: avoid recomputing expensive vmap per forward pass
# =============================================================================
# Key: (device_index, B, H_q, H_kv, seq_len, window_left, block_m, block_n)
_block_mask_cache: dict[tuple, object] = {}


def _get_block_mask(
    B: int,
    H_q: int,
    H_kv: int,
    seq_len: int,
    window_left: int,
    device: torch.device,
) -> object:
    """Get or create a cached BlockMask for the given configuration."""
    dev_idx = device.index if device.index is not None else -1
    cache_key = (dev_idx, B, H_q, H_kv, seq_len, window_left, _BLOCK_M, _BLOCK_N)
    mask = _block_mask_cache.get(cache_key)
    if mask is not None:
        return mask

    if window_left < 0:
        mask_mod = _causal_mask
    else:
        mask_mod = _make_causal_sliding_mask(window_left)

    # For GQA, flex_attention needs H to be the number of Q heads -- it handles
    # the GQA broadcast internally via enable_gqa=True
    mask = _create_block_mask(
        mask_mod,
        B=B,
        H=H_q,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=(_BLOCK_M, _BLOCK_N),
    )
    _block_mask_cache[cache_key] = mask
    return mask


# =============================================================================
# Compiled flex_attention call
# =============================================================================
_FLEX_DECODING_QUERY_LENGTH_CUTOFF = 128


def _build_compiled_flex():
    """Build the compiled flex_attention function based on current HAS_FA4_BACKEND."""
    if not HAS_FLEX_ATTENTION:
        return None
    if HAS_FA4_BACKEND:
        return torch.compile(
            partial(_flex_attention, kernel_options={"BACKEND": "FLASH"}),
            dynamic=False,
        )
    return torch.compile(_flex_attention, dynamic=False)


_compiled_flex = _build_compiled_flex()


# =============================================================================
# Lazy FLASH backend validation: detects seqlen_info incompatibility on first use
# =============================================================================
_flash_compat_validated: bool = False


def _disable_flash_backend() -> None:
    """Disable FLASH backend and switch to Triton. Called on seqlen_info error."""
    global HAS_FA4_BACKEND, _compiled_flex
    if not HAS_FA4_BACKEND:
        return
    logger.error(
        f"flash-attn-4 FLASH backend incompatible with PyTorch {torch.__version__}: "
        "score_mod received unexpected 'seqlen_info' kwarg. "
        "Auto-switching to Triton flex_attention backend for this session. "
        "To resolve: update flash-attn-4 or PyTorch to compatible versions."
    )
    HAS_FA4_BACKEND = False
    _compiled_flex = _build_compiled_flex()


def _ensure_flash_compat() -> None:
    """Lazy validation of FLASH backend. Runs once on first flex_attention call.

    Tests a small forward pass with the compiled FLASH backend. If the CuTe DSL
    FLASH backend passes seqlen_info to score_mod and this PyTorch version
    doesn't support it, catches the error and falls back to Triton.
    """
    global _flash_compat_validated
    if _flash_compat_validated or not HAS_FA4_BACKEND:
        return
    _flash_compat_validated = True

    try:
        q = torch.zeros(1, 1, 128, 64, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            _compiled_flex(q, q.clone(), q.clone())
    except TypeError as e:
        if "seqlen_info" in str(e) or "score_mod" in str(e):
            _disable_flash_backend()
        else:
            raise


def has_fa4_backend() -> bool:
    """Check if FLASH backend is available (validates on first call).

    Use this instead of HAS_FA4_BACKEND directly, because the runtime
    validation may disable FLASH after detecting an incompatibility.
    """
    _ensure_flash_compat()
    return HAS_FA4_BACKEND


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: tuple[int, int],
    enable_gqa: bool,
) -> torch.Tensor:
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=enable_gqa
        )

    # Single token generation
    if Tq == 1:
        if 0 <= window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=False, enable_gqa=enable_gqa
        )

    # Need explicit mask for sliding window / chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if 0 <= window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, enable_gqa=enable_gqa
    )


# =============================================================================
# Public API
# =============================================================================
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, H, S, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, H, S, D)
    """
    if _use_flex():
        _ensure_flash_compat()
        assert causal, "flex_attention path only supports causal=True"
        B, H_q, S, D = q.shape
        H_kv = k.shape[1]
        enable_gqa = H_q != H_kv
        window_left = window_size[0]

        # Bare FlexAttention auto-selects FlexDecoding below this cutoff. On
        # PyTorch 2.13 its forward-only Triton lowering reads a transposed fp32
        # query with contiguous strides, silently producing wrong output. FA4
        # and full FlexAttention are correct, so constrain the copy to the
        # affected fallback boundary. See issue #1601.
        if (
            not HAS_FA4_BACKEND
            and S < _FLEX_DECODING_QUERY_LENGTH_CUTOFF
            and q.dtype == torch.float32
        ):
            q = q.contiguous()

        block_mask = _get_block_mask(B, H_q, H_kv, S, window_left, q.device)
        return _compiled_flex(
            q,
            k,
            v,
            block_mask=block_mask,
            enable_gqa=enable_gqa,
        )

    if _use_fa3():
        # FA3 expects (B, T, H, D) -- transpose from (B, H, S, D)
        q_fa3 = q.transpose(1, 2)
        k_fa3 = k.transpose(1, 2)
        v_fa3 = v.transpose(1, 2)
        out = _fa3.flash_attn_func(
            q_fa3, k_fa3, v_fa3, causal=causal, window_size=window_size
        )
        return out.transpose(1, 2)  # back to (B, H, S, D)

    # SDPA fallback: already expects (B, H, S, D)
    enable_gqa = q.size(1) != k.size(1)
    return _sdpa_attention(q, k, v, window_size, enable_gqa)


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    """
    Flash Attention with KV cache for inference.

    flex_attention has no kvcache equivalent -- falls through to FA3 or SDPA.
    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    # FA3 path (no flex_attention kvcache support)
    if HAS_FA3 and _override_impl != FlashAttnImpl.SDPA:
        return _fa3.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=window_size,
        )

    # SDPA fallback: manually manage KV cache.
    # FA3 accepts per-batch-element ``cache_seqlens`` (shape (B,), int32); our
    # fallback only supports uniform positions across the batch. Raise
    # NotImplementedError explicitly when we see heterogeneous values so that
    # callers get a loud failure instead of silent cache corruption for batch
    # elements >=1.
    B, T_new, H, D = q.shape
    if cache_seqlens is not None and B > 1:
        if not bool((cache_seqlens == cache_seqlens[0]).all()):
            raise NotImplementedError(
                "SDPA fallback requires uniform cache_seqlens; got varying "
                f"positions across batch: {cache_seqlens.tolist()}"
            )
    pos = cache_seqlens[0].item()  # safe: uniform across batch (checked above)

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos : pos + T_new, :, :] = k
        v_cache[:, pos : pos + T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    # Cast KV cache slices to match query dtype (model forward casts to
    # COMPUTE_DTYPE but KV cache is allocated with parameter dtype).
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2).to(q.dtype)
    v_sdpa = v_full.transpose(1, 2).to(q.dtype)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
