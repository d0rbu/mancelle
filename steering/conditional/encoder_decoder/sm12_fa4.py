"""Shared admission for project decoder/head attention, including native SM12 FA4.

SM12 supports native causal/local FA4, but rejects the block masks generated
by nanochat's FlexAttention adapter. Install at the shared worker admission
point and model construction; keep KV-cache and packed encoder kernels intact.
"""

from __future__ import annotations

from collections.abc import Callable

import torch as t

from nanochat import flash_attention
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError

try:
    from flash_attn.cute.interface import flash_attn_func as _native_fa4
except ImportError:
    _native_fa4: Callable[..., tuple[t.Tensor, t.Tensor | None]] | None = None


def _causal_attention(
    q: t.Tensor,
    k: t.Tensor,
    v: t.Tensor,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> t.Tensor:
    """Adapt nanochat's B,H,S,D full-prefix calls without making a block mask."""
    if _native_fa4 is None:
        raise SFTGRPOError("SM12 decoder/head attention requires native FA4/CuTe")
    if not causal or q.shape[2] != k.shape[2]:
        raise SFTGRPOError("native FA4 decoder attention requires a causal full prefix")
    # SFT's evaluate_encdec_bpb (and training) runs under CUDA autocast:
    # nanochat.gpt's Q/K RMSNorm returns FP32 while projected V stays BF16.
    # Match FlexAttention's autocast boundary; native CuTe has no such shim
    # and requires matching low-precision inputs, so q.dtype is not the target.
    if t.is_autocast_enabled(q.device.type):
        dtype = t.get_autocast_dtype(q.device.type)
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    out, _lse = _native_fa4(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=True,
        window_size=(None if window_size[0] < 0 else window_size[0], 0),
        deterministic=t.are_deterministic_algorithms_enabled(),
    )
    return out.transpose(1, 2)


def _full_prefix_attention(q, k, v, causal=False, window_size=(-1, -1)):
    # CPU reference forwards and explicit test overrides remain usable even
    # after a CUDA model has installed the process-wide namespace adapter.
    if not q.is_cuda or flash_attention._override_impl in (
        flash_attention.FlashAttnImpl.SDPA, flash_attention.FlashAttnImpl.FA3
    ) or t.cuda.get_device_capability(q.device)[0] != 12:
        return flash_attention.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    return _causal_attention(q, k, v, causal=causal, window_size=window_size)


def uses_native_sm12_fa4(device: t.device | None = None) -> bool:
    return (
        flash_attention.flash_attn.flash_attn_func is _full_prefix_attention
        and flash_attention._override_impl not in (
            flash_attention.FlashAttnImpl.SDPA, flash_attention.FlashAttnImpl.FA3
        )
        and (device is None or device.type == "cuda" and t.cuda.get_device_capability(device)[0] == 12)
    )


def admit_project_attention(
    *, device: t.device | None = None, require_fa4: bool = False
) -> None:
    """Admit CUDA execution without requiring its dependencies for CPU/meta models.

    Trainers omit ``device`` to check the current CUDA rank before construction.
    Model construction and materialization pass their actual device.
    """
    if (device is not None and device.type != "cuda") or not t.cuda.is_available():
        return
    major = (
        t.cuda.get_device_capability(device)[0]
        if device is not None
        else t.cuda.get_device_capability()[0]
    )
    if flash_attention._override_impl in (
        flash_attention.FlashAttnImpl.SDPA, flash_attention.FlashAttnImpl.FA3
    ):
        if require_fa4:
            raise SFTGRPOError("requested FA4 disagrees with the active attention override")
        if flash_attention._override_impl == flash_attention.FlashAttnImpl.FA3 and (
            major != 9 or not flash_attention.HAS_FA3
        ):
            raise SFTGRPOError("FA3 attention requires an SM9 CUDA device and available backend")
        return
    if major < 9 and flash_attention.HAS_FA4_BACKEND:
        raise SFTGRPOError("selected CUDA device cannot use the imported FA4 backend")
    if major == 12:
        if _native_fa4 is None:
            raise SFTGRPOError("SM12 decoder/head attention requires native FA4/CuTe")
        # Replace only the namespace's full-prefix entry; cached attention is unchanged.
        flash_attention.flash_attn.flash_attn_func = _full_prefix_attention
    if (require_fa4 or flash_attention._override_impl == flash_attention.FlashAttnImpl.FA4) and not (
        uses_native_sm12_fa4(device) or flash_attention.has_fa4_backend()
    ):
        raise SFTGRPOError("requested FA4 requires an available FA4/CuTe backend")
