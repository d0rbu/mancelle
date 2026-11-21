"""Unified resolver for the ``compile_static`` config knob.

``conditional_encdec_train.py`` and ``conditional_encdec_sft.py`` both
expose a ``compile_static`` config knob.  Historically each script
resolved it to ``torch.compile``'s ``dynamic`` flag using its own
truth table.  This module is the single source of truth for that
resolution, so the two scripts cannot drift.

Both modes agree on:

* ``True`` / ``"true"``  -> ``dynamic=False`` (force static).
* ``False`` / ``"false"`` -> ``dynamic=True`` (force dynamic).

They differ only on the ``"auto"`` branch:

* ``mode="pretrain"``: ``dynamic = not static_shapes_available``.
  Static shapes exist iff selection shards are present OR ``pad_F`` is
  resolvable from ``max_k`` (live dataloader fused path).

* ``mode="sft"``: ``dynamic = None`` (torch.compile auto-dynamic).
  SFT decoder rows are a fixed ``(device_batch_size, max_seq_len)`` shape
  (bestfit-pad packing, ``steering/sft/iteration.py``), while the packed
  encoder / cross-attention tensors that genuinely vary per batch are
  ``mark_dynamic``'d at construction (``packed_encoder.py``).  Auto-dynamic
  therefore compiles the packed extents symbolically from the first trace
  and keeps every decoder dimension a concrete Python int.  That concreteness
  is load-bearing: torch >= 2.13 lowers decoder self-attention through the
  flex-attention FLASH (CuteDSL) backend on Hopper+, whose template cannot
  inline symbolic values ("NYI: score_mod or mask_mod captures a dynamic
  scalar").  ``dynamic=True`` symbolizes every dimension — including the
  decoder causal mask's — and crashes FA4 at the first compiled forward.

Both scripts call ``resolve_compile_dynamic`` directly with the
appropriate ``mode`` keyword — there is no script-private wrapper.
"""

from __future__ import annotations

from typing import Literal


def resolve_compile_dynamic(
    compile_static: bool | str,
    *,
    mode: Literal["pretrain", "sft"],
    static_shapes_available: bool | None = None,
) -> bool | None:
    """Resolve ``compile_static`` to the ``dynamic`` flag for ``torch.compile``.

    Args:
        compile_static: Config value.  Accepts the booleans ``True`` /
            ``False`` and the strings ``"true"`` / ``"false"`` / ``"auto"``.
        mode: Training script: ``"pretrain"`` (encdec pretraining) or
            ``"sft"`` (encdec SFT).
        static_shapes_available: Required when ``mode="pretrain"``.  True
            when static batch shapes are guaranteed (selection shards
            present or live-loader has known ``pad_F``).
    Returns:
        ``True`` for ``dynamic=True``, ``False`` for ``dynamic=False``,
        ``None`` for auto-dynamic (``mode="sft"`` with ``"auto"``: static
        decoder dims, ``mark_dynamic``-driven symbolic packed extents).

    Raises:
        ValueError: ``mode`` is neither ``"pretrain"`` nor ``"sft"``.
        TypeError: missing keyword for the chosen mode.
    """
    # Force-static / force-dynamic branches: same in both modes.
    if compile_static is True or compile_static == "true":
        return False
    if compile_static is False or compile_static == "false":
        return True

    # "auto" branch: mode-dependent.
    if mode == "pretrain":
        if static_shapes_available is None:
            raise TypeError(
                "resolve_compile_dynamic(mode='pretrain', compile_static='auto') "
                "requires keyword argument static_shapes_available."
            )
        return not static_shapes_available
    if mode == "sft":
        return None

    raise ValueError(
        f"Unknown mode {mode!r}. Expected one of {{'pretrain', 'sft'}}."
    )
