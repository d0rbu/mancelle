"""Pure helpers and core generator for SFT data batching.

These live in ``steering.sft`` (rather than inline in ``scripts/chat_sft.py``)
so tests can import them without triggering ``chat_sft.py``'s top-level
Hydra compose, wandb init, and model-load side effects.

The generator owns only produced-batch bookkeeping. The training loop owns
termination and promotes produced conversation consumption to committed state
only after a successful optimizer step.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch as t
from loguru import logger

from steering.sft.loss_mask import (
    LossMaskMode,
    apply_completion_only_row_masks,
    assert_min_one_valid_target,
    render_conversation_with_mask,
    supervised_target_count,
    truncate_conv_and_mask,
)


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return value


def validate_decoder_sft_dataloader_state(
    state: Mapping[str, Any],
    *,
    ddp_rank: int,
    ddp_world_size: int,
    buffer_size: int,
) -> dict[str, Any]:
    """Validate one rank's exact decoder best-fit resume state."""
    expected_keys = {
        "next_cursor",
        "buffered_cursor_positions",
        "consumed",
        "micro_batch_count",
    }
    actual_keys = set(state)
    if actual_keys != expected_keys:
        raise ValueError(
            "decoder SFT dataloader state keys differ from the restart contract: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    next_cursor = _strict_nonnegative_int(state["next_cursor"], name="next_cursor")
    if next_cursor < ddp_rank or (next_cursor - ddp_rank) % ddp_world_size != 0:
        raise ValueError(
            f"next_cursor={next_cursor} does not belong to rank {ddp_rank} "
            f"under world size {ddp_world_size}"
        )
    consumed = _strict_nonnegative_int(state["consumed"], name="consumed")
    micro_batch_count = _strict_nonnegative_int(
        state["micro_batch_count"], name="micro_batch_count"
    )

    raw_buffered = state["buffered_cursor_positions"]
    if not isinstance(raw_buffered, list):
        raise ValueError("buffered_cursor_positions must be a list")
    if len(raw_buffered) > buffer_size:
        raise ValueError(
            f"buffered state has {len(raw_buffered)} entries, exceeding buffer_size={buffer_size}"
        )
    buffered: list[int] = []
    for raw_position in raw_buffered:
        position = _strict_nonnegative_int(raw_position, name="buffered_cursor_positions entry")
        if position >= next_cursor:
            raise ValueError(f"buffered position {position} must precede next_cursor={next_cursor}")
        if position < ddp_rank or (position - ddp_rank) % ddp_world_size != 0:
            raise ValueError(
                f"buffered position {position} does not belong to rank {ddp_rank} "
                f"under world size {ddp_world_size}"
            )
        if buffered and position <= buffered[-1]:
            raise ValueError("buffered_cursor_positions must be strictly increasing")
        buffered.append(position)

    attempted = (next_cursor - ddp_rank) // ddp_world_size
    if consumed + len(buffered) > attempted:
        raise ValueError(
            "decoder SFT dataloader state accounts for more than the attempted "
            f"conversation: attempted={attempted}, consumed={consumed}, "
            f"buffered={len(buffered)}"
        )
    return {
        "next_cursor": next_cursor,
        "buffered_cursor_positions": buffered,
        "consumed": consumed,
        "micro_batch_count": micro_batch_count,
    }


def should_save_checkpoint(
    *,
    is_master: bool,
    is_dry_run: bool,
    is_last_step: bool,
    step: int,
    save_every: int,
) -> bool:
    """Return True iff chat_sft.py should write a checkpoint this loop.

    Three conditions compose the predicate:
      - ``is_master`` — only rank 0 writes (avoids concurrent overwrites).
      - ``not is_dry_run`` — dry runs never persist anything.
      - ``is_last_step or (save_every > 0 and step > 0 and step % save_every == 0)``
        — always save at the end; optionally save at regular intervals.

    ``step > 0`` skips the degenerate initial save (model-at-init weights
    with no training applied). ``save_every <= 0`` disables intermediate
    saves and preserves the pre-``save_every`` behavior of saving only
    once at the end. When ``is_last_step`` and the interval boundary
    coincide (e.g. step=1000 with save_every=500), the ``or`` ensures a
    single save call, not a double-write.
    """
    if not is_master:
        return False
    if is_dry_run:
        return False
    if is_last_step:
        return True
    return save_every > 0 and step > 0 and step % save_every == 0


@dataclass(frozen=True)
class SFTLoopPosition:
    """Committed loop boundary used for stop, schedule, eval, and save."""

    last_step: bool
    progress: float


def resolve_sft_loop_position(
    *,
    completed_steps: int,
    num_iterations: int,
    committed_consumed: int,
    dataset_size: int,
) -> SFTLoopPosition:
    """Resolve SFT stop/progress exclusively from successfully committed work."""
    if completed_steps < 0:
        raise ValueError(f"completed_steps must be non-negative, got {completed_steps}")
    if committed_consumed < 0:
        raise ValueError(f"committed_consumed must be non-negative, got {committed_consumed}")
    if num_iterations == 0 or num_iterations < -1:
        raise ValueError(
            f"num_iterations must be -1 or a positive optimizer-step count; got {num_iterations}"
        )
    if num_iterations > 0:
        progress = completed_steps / num_iterations
        return SFTLoopPosition(
            last_step=completed_steps >= num_iterations,
            progress=min(1.0, max(0.0, progress)),
        )
    if dataset_size <= 0:
        raise ValueError(
            f"dataset_size must be positive for a dataset-horizon run, got {dataset_size}"
        )
    progress = committed_consumed / dataset_size
    return SFTLoopPosition(
        last_step=committed_consumed >= dataset_size,
        progress=min(1.0, max(0.0, progress)),
    )


@dataclass
class SFTGeneratorProgress:
    """Per-yield snapshot of the SFT generator's bookkeeping state.

    This object deliberately describes produced data only. Stop and LR
    progress are resolved from committed optimizer-step state by
    :func:`resolve_sft_loop_position` in the training loop.
    """

    micro_batch_count: int
    current_epoch: int
    consumed: int
    nonpadding_target_count: int
    supervised_target_count: int
    dataloader_state: dict[str, Any]


def sft_data_generator_bos_bestfit_core(
    *,
    dataset: Any,
    tokenizer: Any,
    max_seq_len: int,
    device_batch_size: int,
    ddp_rank: int,
    ddp_world_size: int,
    device: t.device,
    pin_memory: bool,
    buffer_size: int = 100,
    loss_mask_mode: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
    resume_state: Mapping[str, Any] | None = None,
) -> Iterator[tuple[t.Tensor, t.Tensor, SFTGeneratorProgress]]:
    """BOS-aligned SFT dataloader with bestfit-pad packing, factored as
    a pure generator over explicit arguments (no module globals).

    This is the testable core of
    ``scripts.train.chat_sft.sft_data_generator_bos_bestfit``.

    Contract:
      - Every yield is a complete micro-batch: ``(inputs, targets, progress)``
        where inputs/targets have shape ``(device_batch_size, max_seq_len)``
        and progress carries the updated bookkeeping state.
      - Conversations longer than ``row_capacity = max_seq_len + 1`` are
        truncated at buffer insertion; see ``find_bestfit_index`` for the
        starvation rationale.
      - Padding uses the tokenizer's BOS id; target positions past the
        content length are masked with -1 (ignore_index for cross-entropy).
    ``ddp_world_size`` and ``pin_memory`` come from the caller's compute
    init; ``device``/``pin_memory`` control tensor placement.
    """
    assert dataset is not None, "dataset must be provided"
    dataset_size = len(dataset)
    assert dataset_size > 0, "dataset is empty"
    if ddp_world_size <= 0:
        raise ValueError(f"ddp_world_size must be positive, got {ddp_world_size}")
    if not 0 <= ddp_rank < ddp_world_size:
        raise ValueError(f"ddp_rank must be in [0, {ddp_world_size}), got {ddp_rank}")
    if dataset_size <= ddp_rank:
        raise ValueError(
            f"dataset_size={dataset_size} gives rank {ddp_rank} no striped conversations"
        )
    if buffer_size <= 0:
        raise ValueError(f"buffer_size must be positive, got {buffer_size}")
    row_capacity = max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Local state (generator instance-private).
    conv_buffer: list[tuple[int, list[int], list[int]]] = []
    normalized_resume = (
        None
        if resume_state is None
        else validate_decoder_sft_dataloader_state(
            resume_state,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            buffer_size=buffer_size,
        )
    )
    cursor = ddp_rank if normalized_resume is None else normalized_resume["next_cursor"]
    consumed = 0 if normalized_resume is None else normalized_resume["consumed"]
    it = 0 if normalized_resume is None else normalized_resume["micro_batch_count"]

    def materialize(logical_position: int) -> tuple[list[int], list[int]] | None:
        example_idx = logical_position % dataset_size
        conversation = dataset[example_idx]
        # Deliberately the RAW conversation, unlike the conditioned trainer.
        # This generator has no parquet, no feature store and no
        # original_index mapping, so nothing binds it to the converter's
        # "truncate after the last assistant turn" contract (issue #1465);
        # its callers' datasets are not even required to have a "messages"
        # key. Do not "unify" the two without deciding what non-conditioned
        # SFT should supervise.
        ids, loss_mask = render_conversation_with_mask(tokenizer, conversation)
        ids, loss_mask, _, _ = truncate_conv_and_mask(
            ids,
            loss_mask,
            row_capacity,
        )
        if (
            loss_mask_mode is LossMaskMode.COMPLETION_ONLY
            and supervised_target_count(loss_mask) == 0
        ):
            return None
        return ids, loss_mask

    if normalized_resume is not None:
        for logical_position in normalized_resume["buffered_cursor_positions"]:
            restored = materialize(logical_position)
            if restored is None:
                raise ValueError(
                    "saved decoder SFT buffer position now renders with zero "
                    f"supervised targets: {logical_position}"
                )
            ids, loss_mask = restored
            conv_buffer.append((logical_position, ids, loss_mask))

    def refill_buffer() -> None:
        nonlocal cursor
        consecutive_skips = 0
        while len(conv_buffer) < buffer_size:
            logical_position = cursor
            example_idx = logical_position % dataset_size
            restored = materialize(logical_position)
            cursor += ddp_world_size
            if restored is None:
                consecutive_skips += 1
                skipped = (cursor - ddp_rank) // ddp_world_size - consumed - len(conv_buffer)
                logger.debug(
                    f"Skipping SFT conversation {example_idx} under "
                    "sft_loss_mask=completion_only: 0 supervised target "
                    f"tokens after truncation; skipped={skipped}"
                )
                if consecutive_skips >= dataset_size:
                    raise RuntimeError(
                        "SFT completion_only buffer starvation: scanned all "
                        f"{dataset_size} conversations without finding one "
                        "supervised target after truncation"
                    )
                continue
            consecutive_skips = 0
            ids, loss_mask = restored
            conv_buffer.append((logical_position, ids, loss_mask))

    while True:
        rows: list[list[int]] = []
        row_masks: list[list[int]] = []
        row_lengths: list[int] = []  # content length (excluding padding)
        for _ in range(device_batch_size):
            row: list[int] = []
            row_mask: list[int] = []
            padded = False
            content_len = 0  # defensive init (if loop fills row_capacity exactly)
            while len(row) < row_capacity:
                while len(conv_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - len(row)
                best_idx = _bestfit_index(
                    (len(ids) for _position, ids, _mask in conv_buffer),
                    remaining,
                )
                if best_idx >= 0:
                    _position, conv_ids, conv_mask = conv_buffer.pop(best_idx)
                    row.extend(conv_ids)
                    row_mask.extend(conv_mask)
                    consumed += 1
                else:
                    content_len = len(row)
                    row.extend([bos_token] * remaining)
                    row_mask.extend([0] * remaining)
                    padded = True
                    break
            row_lengths.append(content_len if padded else row_capacity)
            rows.append(row[:row_capacity])
            row_masks.append(row_mask[:row_capacity])

        it += 1
        batch_tensor = t.tensor(rows, dtype=t.long, pin_memory=pin_memory)
        use_non_blocking = pin_memory  # non_blocking transfers only help with pinned src
        inputs = batch_tensor[:, :-1].to(
            device=device, dtype=t.int32, non_blocking=use_non_blocking
        )
        targets = batch_tensor[:, 1:].to(
            device=device, dtype=t.int64, non_blocking=use_non_blocking
        )
        for i, content_len in enumerate(row_lengths):
            if content_len < row_capacity:
                targets[i, content_len - 1 :] = -1
        nonpadding_target_count = sum(max(0, content_len - 1) for content_len in row_lengths)
        if loss_mask_mode is LossMaskMode.COMPLETION_ONLY:
            apply_completion_only_row_masks(targets, row_masks)
            assert_min_one_valid_target(targets)
            final_supervised_target_count = sum(
                sum(row_mask[1:content_len])
                for row_mask, content_len in zip(row_masks, row_lengths, strict=True)
            )
        else:
            final_supervised_target_count = nonpadding_target_count

        dataloader_state = {
            "next_cursor": cursor,
            "buffered_cursor_positions": [position for position, _ids, _mask in conv_buffer],
            "consumed": consumed,
            "micro_batch_count": it,
        }

        progress = SFTGeneratorProgress(
            micro_batch_count=it,
            current_epoch=cursor // dataset_size + 1,
            consumed=consumed,
            nonpadding_target_count=nonpadding_target_count,
            supervised_target_count=final_supervised_target_count,
            dataloader_state=dataloader_state,
        )
        yield inputs, targets, progress


def find_bestfit_index(
    conv_buffer: list[tuple[list[int], list[int]]],
    remaining: int,
) -> int:
    """Return the index of the largest conversation in ``conv_buffer`` that
    fits in ``remaining`` tokens, or ``-1`` if no conversation fits.

    Ties on length are broken by first-found (we use a strict ``>`` when
    checking for a new best, so earlier indices win on equal lengths).

    Background — buffer starvation without truncation
    -----------------------------------------------------
    The SFT packer must always make progress when called with a non-empty
    buffer and ``remaining == row_capacity`` (a fresh row). If a
    conversation is longer than ``row_capacity`` and enters the buffer
    untruncated, it is "poison" — it never satisfies ``conv_len <=
    remaining``, so it sits in the buffer forever. Once enough such
    conversations accumulate to fill the buffer, ``find_bestfit_index``
    returns ``-1`` every call, every row pads entirely, every micro-batch
    has zero unmasked loss positions, and ``cross_entropy`` produces NaN
    on an all-masked target. Truncating rendered conversations to
    ``row_capacity`` before buffer insertion eliminates the poison.
    """
    return _bestfit_index((len(ids) for ids, _mask in conv_buffer), remaining)


def _bestfit_index(lengths: Sequence[int] | Iterator[int], remaining: int) -> int:
    """Return the first index of the largest positive length that fits."""
    best_idx = -1
    best_len = 0
    for i, conv_len in enumerate(lengths):
        if conv_len <= remaining and conv_len > best_len:
            best_idx = i
            best_len = conv_len
    return best_idx
