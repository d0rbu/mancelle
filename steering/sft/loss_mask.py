"""Shared completion-only loss masking for supervised fine-tuning."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import torch as t
from beartype import beartype


class LossMaskMode(StrEnum):
    """Decoder target-supervision modes shared by both SFT trainers."""

    FULL_SEQUENCE = "full_sequence"
    COMPLETION_ONLY = "completion_only"


@beartype
def parse_loss_mask_mode(raw: object) -> LossMaskMode:
    """Parse the user-facing ``sft_loss_mask`` value or fail loud."""
    valid = sorted(mode.value for mode in LossMaskMode)
    if not isinstance(raw, str):
        raise ValueError(f"sft_loss_mask must be one of {valid}; got {raw!r}")
    try:
        return LossMaskMode(str(raw))
    except ValueError as exc:
        raise ValueError(f"sft_loss_mask must be one of {valid}; got {raw!r}") from exc


@beartype
def render_conversation_with_mask(
    tokenizer: Any,
    conversation: Any,
) -> tuple[list[int], list[int]]:
    """Render one conversation and verify token/mask positional alignment."""
    ids, mask = tokenizer.render_conversation(conversation)
    assert len(ids) == len(mask), (
        "render_conversation returned misaligned ids/mask lengths: "
        f"len(ids)={len(ids)}, len(mask)={len(mask)}"
    )
    return ids, mask


@beartype
def truncate_conv_and_mask(
    ids: list[int],
    mask: list[int],
    row_capacity: int,
) -> tuple[list[int], list[int], int, bool]:
    """Truncate an aligned token/mask pair to one packing-row capacity."""
    if row_capacity < 0:
        raise ValueError(f"row_capacity must be non-negative, got {row_capacity}")
    assert len(ids) == len(mask), (
        f"cannot truncate a misaligned ids/mask pair: len(ids)={len(ids)}, len(mask)={len(mask)}"
    )
    original_len = len(ids)
    if original_len <= row_capacity:
        return ids, mask, original_len, False
    return ids[:row_capacity], mask[:row_capacity], original_len, True


@beartype
def supervised_target_count(mask: Sequence[int]) -> int:
    """Count supervised next-token targets; the leading BOS is not a target."""
    return sum(mask[1:])


@beartype
def apply_completion_only_row_masks(
    targets: t.Tensor,
    row_masks: Sequence[Sequence[int]],
) -> None:
    """Apply conversation masks in-place to already padding-masked targets.

    Target position ``j`` predicts row token ``j + 1``, so the single
    next-token shift lives here. Positions beyond a row mask are left alone;
    the caller's existing padding mask already owns them.
    """
    if targets.ndim != 2:
        raise ValueError(
            f"completion-only targets must have shape (B, T), got {tuple(targets.shape)}"
        )
    if len(row_masks) != targets.shape[0]:
        raise ValueError(
            "completion-only row mask count must match target batch size: "
            f"len(row_masks)={len(row_masks)}, B={targets.shape[0]}"
        )

    target_width = targets.shape[1]
    for row_idx, row_mask in enumerate(row_masks):
        masked_width = min(target_width, max(0, len(row_mask) - 1))
        if masked_width == 0:
            continue
        shifted_mask = t.as_tensor(
            row_mask[1 : masked_width + 1],
            dtype=t.bool,
            device=targets.device,
        )
        targets[row_idx, :masked_width].masked_fill_(~shifted_mask, -1)


@beartype
def assert_min_one_valid_target(targets: t.Tensor) -> None:
    """Reject completion-only rows that would make mean cross-entropy NaN."""
    if targets.ndim != 2:
        raise ValueError(
            f"completion-only targets must have shape (B, T), got {tuple(targets.shape)}"
        )
    valid_by_row = (targets != -1).any(dim=1)
    invalid_rows = (~valid_by_row).nonzero(as_tuple=False)
    if invalid_rows.numel() > 0:
        row_idx = int(invalid_rows[0, 0].item())
        raise ValueError(
            "sft_loss_mask=completion_only produced an all-masked target row "
            f"at index {row_idx}; every emitted row must contain at least one "
            "supervised target"
        )


@beartype
def assert_resume_sft_loss_mask_matches(
    configured_mode: LossMaskMode,
    checkpoint_meta: dict[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Verify loss-mask identity before resuming an SFT checkpoint.

    Historical checkpoints predate ``sft_loss_mask`` and therefore omit the
    metadata key. They are interpreted as ``full_sequence``, the only mode
    that existed before completion-only masking was introduced.
    """
    checkpoint_raw = (
        LossMaskMode.FULL_SEQUENCE.value
        if checkpoint_meta is None
        else checkpoint_meta.get("sft_loss_mask", LossMaskMode.FULL_SEQUENCE.value)
    )
    checkpoint_mode = parse_loss_mask_mode(checkpoint_raw)
    if checkpoint_mode is configured_mode:
        return
    raise ValueError(
        f"sft_loss_mask mismatch on resume from {checkpoint_dir}: "
        f"checkpoint records {checkpoint_mode.value!r}, "
        f"configured {configured_mode.value!r}. Resume with the original "
        "mode or start a fresh SFT run."
    )
