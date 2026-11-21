"""Build a GPT decoder on the meta device for shape-only construction.

The same dim-rounding pattern recurs across training scripts; this
module hosts the canonical implementation so every site shares the
same rule (round ``depth * aspect_ratio`` up to the nearest multiple
of ``head_dim``) without duplicating the math. Current consumers
include ``scripts/train/base_train.py``; the encoder-decoder training
scripts still inline equivalent logic and can be wired through here
in a follow-up cleanup.
"""

from __future__ import annotations

import math

import torch

from nanochat.gpt import GPT, GPTConfig


def resolve_gpt_dimensions(
    *, depth: int, aspect_ratio: int | float, head_dim: int
) -> tuple[int, int]:
    """Resolve a GPT width and head count from the scaling-law inputs.

    ``aspect_ratio`` is a per-layer width ratio and may be fractional when a
    target width is specified for a fixed depth (for example, ``1664 / 27``).
    The resulting model width is always rounded up to a whole ``head_dim`` and
    both returned values are plain integers suitable for ``GPTConfig``.
    """
    base_dim = depth * aspect_ratio
    model_dim = math.ceil(base_dim / head_dim) * head_dim
    return model_dim, model_dim // head_dim


def build_gpt_meta(
    *,
    depth: int,
    aspect_ratio: int | float,
    head_dim: int,
    max_seq_len: int,
    window_pattern: str,
    vocab_size: int,
) -> GPT:
    """Construct a ``GPT`` decoder on the meta device.

    Parameters
    ----------
    depth
        Number of transformer layers (``n_layer``).
    aspect_ratio
        Per-layer hidden-dim multiplier. It may be fractional when a target
        width is specified for a fixed depth. Model width is
        ``depth * aspect_ratio`` rounded up to the next multiple of
        ``head_dim`` (FA3 requires ``head_dim`` divisible by 8, and
        rounding up to a multiple of ``head_dim`` guarantees the final
        ``n_head`` divides ``n_embd`` cleanly).
    head_dim
        Per-head dimension. Used both for rounding and for the
        ``n_head = model_dim // head_dim`` derivation.
    max_seq_len
        Sequence length in tokens (``GPTConfig.sequence_len``).
    window_pattern
        Sliding-window pattern — one of ``"L"`` (no SWA), ``"SSSL"``,
        etc.
    vocab_size
        Tokenizer vocabulary size.

    Returns
    -------
    GPT
        A ``GPT`` model whose tensors all live on the ``meta`` device
        (no storage). The caller is responsible for ``.to_empty(...)``
        + ``.init_weights()`` (or ``.init_buffers()`` if loading a
        checkpoint).

    Notes
    -----
    * ``n_kv_head`` is set equal to ``n_head``. This matches every
      production training script in the repo.
    * This module is the authoritative implementation of the
      dim-rounding rule — keep the math here.
    """
    model_dim, num_heads = resolve_gpt_dimensions(
        depth=depth, aspect_ratio=aspect_ratio, head_dim=head_dim
    )
    config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        return GPT(config)
