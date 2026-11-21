"""Exact buffered resume tests for the decoder-only SFT best-fit packer."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.sft.iteration import (
    SFTGeneratorProgress,
    sft_data_generator_bos_bestfit_core,
    validate_decoder_sft_dataloader_state,
)
from steering.sft.loss_mask import LossMaskMode


class _Dataset:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index, "length": self.lengths[index]}


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 1

    def render_conversation(self, conversation: dict[str, int]) -> tuple[list[int], list[int]]:
        length = conversation["length"]
        index = conversation["index"]
        ids = [1] + [2 + ((index * 31 + offset) % 200) for offset in range(length - 1)]
        # Every accepted conversation has at least one completion target. The
        # alternating tail exercises completion-only masking after packing.
        mask = [0] + [int(offset % 2 == 0) for offset in range(length - 1)]
        return ids, mask


def _generator(
    lengths: list[int],
    *,
    mode: LossMaskMode = LossMaskMode.FULL_SEQUENCE,
    rank: int = 0,
    world_size: int = 1,
    resume_state: dict[str, Any] | None = None,
    max_seq_len: int = 12,
    batch_size: int = 2,
    buffer_size: int = 7,
):
    return sft_data_generator_bos_bestfit_core(
        dataset=_Dataset(lengths),
        tokenizer=_Tokenizer(),
        max_seq_len=max_seq_len,
        device_batch_size=batch_size,
        ddp_rank=rank,
        ddp_world_size=world_size,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=buffer_size,
        loss_mask_mode=mode,
        resume_state=resume_state,
    )


def _snapshot(batch: tuple[t.Tensor, t.Tensor, SFTGeneratorProgress]) -> tuple[Any, ...]:
    inputs, targets, progress = batch
    return (
        inputs.clone(),
        targets.clone(),
        progress.micro_batch_count,
        progress.current_epoch,
        progress.consumed,
        progress.nonpadding_target_count,
        progress.supervised_target_count,
        copy.deepcopy(progress.dataloader_state),
    )


@pytest.mark.parametrize("mode", [LossMaskMode.FULL_SEQUENCE, LossMaskMode.COMPLETION_ONLY])
@pytest.mark.parametrize(("rank", "world_size"), [(0, 1), (0, 2), (1, 2)])
def test_resume_reproduces_every_next_tensor_and_intermediate_state(
    mode: LossMaskMode,
    rank: int,
    world_size: int,
) -> None:
    lengths = [3, 11, 5, 8, 17, 4, 9, 6, 13, 7, 10]
    baseline = _generator(lengths, mode=mode, rank=rank, world_size=world_size)
    committed = None
    for _ in range(5):
        committed = next(baseline)[2].dataloader_state
    assert committed is not None
    serialized = json.loads(json.dumps(committed, allow_nan=False))
    expected = [_snapshot(next(baseline)) for _ in range(8)]

    resumed = _generator(
        lengths,
        mode=mode,
        rank=rank,
        world_size=world_size,
        resume_state=serialized,
    )
    actual = [_snapshot(next(resumed)) for _ in range(8)]
    for expected_batch, actual_batch in zip(expected, actual, strict=True):
        assert t.equal(expected_batch[0], actual_batch[0])
        assert t.equal(expected_batch[1], actual_batch[1])
        assert expected_batch[2:] == actual_batch[2:]


@settings(max_examples=100, deadline=None)
@given(
    lengths=st.lists(
        st.integers(min_value=2, max_value=30),
        min_size=4,
        max_size=24,
    ),
    consumed_batches=st.integers(min_value=1, max_value=6),
    mode=st.sampled_from([LossMaskMode.FULL_SEQUENCE, LossMaskMode.COMPLETION_ONLY]),
)
def test_random_length_resume_is_an_exact_state_machine_replay(
    lengths: list[int],
    consumed_batches: int,
    mode: LossMaskMode,
) -> None:
    baseline = _generator(
        lengths,
        mode=mode,
        max_seq_len=10,
        batch_size=1,
        buffer_size=4,
    )
    committed = None
    for _ in range(consumed_batches):
        committed = next(baseline)[2].dataloader_state
    assert committed is not None
    expected = [_snapshot(next(baseline)) for _ in range(3)]
    resumed = _generator(
        lengths,
        mode=mode,
        resume_state=copy.deepcopy(committed),
        max_seq_len=10,
        batch_size=1,
        buffer_size=4,
    )
    actual = [_snapshot(next(resumed)) for _ in range(3)]
    for left, right in zip(expected, actual, strict=True):
        assert t.equal(left[0], right[0])
        assert t.equal(left[1], right[1])
        assert left[2:] == right[2:]


def test_published_state_accounts_for_every_attempt_and_is_independent() -> None:
    generator = _generator([3, 4, 5, 6, 7, 8])
    _inputs, _targets, first = next(generator)
    saved = copy.deepcopy(first.dataloader_state)
    next(generator)
    assert first.dataloader_state == saved
    state = first.dataloader_state
    attempted = state["next_cursor"]
    assert attempted >= state["consumed"] + len(state["buffered_cursor_positions"])
    assert first.current_epoch == state["next_cursor"] // 6 + 1


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda state: state.update(next_cursor=-1), "next_cursor"),
        (lambda state: state.update(consumed=True), "consumed"),
        (lambda state: state.update(micro_batch_count=1.0), "micro_batch_count"),
        (lambda state: state.update(extra=1), "unexpected"),
        (lambda state: state.pop("consumed"), "missing"),
    ],
)
def test_every_scalar_or_schema_drift_fails_closed(mutation, match: str) -> None:
    progress = next(_generator([3, 4, 5, 6, 7, 8]))[2]
    state = copy.deepcopy(progress.dataloader_state)
    mutation(state)
    with pytest.raises((ValueError, KeyError), match=match):
        validate_decoder_sft_dataloader_state(
            state,
            ddp_rank=0,
            ddp_world_size=1,
            buffer_size=7,
        )


def test_buffer_order_ownership_bounds_and_accounting_fail_closed() -> None:
    progress = next(_generator([3, 4, 5, 6, 7, 8, 9, 10], rank=1, world_size=2))[2]

    def validate(state: dict[str, Any]) -> None:
        validate_decoder_sft_dataloader_state(
            state,
            ddp_rank=1,
            ddp_world_size=2,
            buffer_size=7,
        )

    state = copy.deepcopy(progress.dataloader_state)
    assert len(state["buffered_cursor_positions"]) >= 2
    state["buffered_cursor_positions"] = list(reversed(state["buffered_cursor_positions"]))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate(state)

    state = copy.deepcopy(progress.dataloader_state)
    state["buffered_cursor_positions"][0] += 1
    with pytest.raises(ValueError, match="does not belong to rank"):
        validate(state)

    state = copy.deepcopy(progress.dataloader_state)
    state["buffered_cursor_positions"][0] = state["next_cursor"]
    with pytest.raises(ValueError, match="must precede"):
        validate(state)

    state = copy.deepcopy(progress.dataloader_state)
    state["consumed"] += 1
    with pytest.raises(ValueError, match="more than the attempted"):
        validate(state)

    state = copy.deepcopy(progress.dataloader_state)
    state["buffered_cursor_positions"].extend([state["next_cursor"] - 2] * 7)
    with pytest.raises(ValueError, match="exceeding buffer_size"):
        validate(state)


def test_completion_only_reports_both_denominators_exactly() -> None:
    lengths = [3, 7, 4, 11, 5, 8]
    full_inputs, full_targets, full_progress = next(
        _generator(lengths, mode=LossMaskMode.FULL_SEQUENCE)
    )
    completion_inputs, completion_targets, completion_progress = next(
        _generator(lengths, mode=LossMaskMode.COMPLETION_ONLY)
    )
    assert t.equal(full_inputs, completion_inputs)
    assert full_progress.nonpadding_target_count == int((full_targets != -1).sum())
    assert completion_progress.nonpadding_target_count == full_progress.nonpadding_target_count
    assert completion_progress.supervised_target_count == int((completion_targets != -1).sum())
    assert (
        0
        < completion_progress.supervised_target_count
        < (completion_progress.nonpadding_target_count)
    )


def test_saved_buffer_re_rendering_drift_is_rejected() -> None:
    lengths = [3, 4, 5, 6, 7, 8]
    progress = next(_generator(lengths, mode=LossMaskMode.COMPLETION_ONLY))[2]
    state = copy.deepcopy(progress.dataloader_state)

    class _DriftedTokenizer(_Tokenizer):
        def render_conversation(self, conversation: dict[str, int]) -> tuple[list[int], list[int]]:
            ids, mask = super().render_conversation(conversation)
            return ids, [0] * len(mask)

    resumed = sft_data_generator_bos_bestfit_core(
        dataset=_Dataset(lengths),
        tokenizer=_DriftedTokenizer(),
        max_seq_len=12,
        device_batch_size=2,
        ddp_rank=0,
        ddp_world_size=1,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=7,
        loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
        resume_state=state,
    )
    with pytest.raises(ValueError, match="now renders with zero supervised targets"):
        next(resumed)
