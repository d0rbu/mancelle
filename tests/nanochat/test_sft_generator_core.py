"""Produced-state contract tests for the decoder SFT best-fit packer."""

from __future__ import annotations

from dataclasses import fields
from itertools import pairwise

import torch as t

from steering.sft.iteration import (
    SFTGeneratorProgress,
    sft_data_generator_bos_bestfit_core,
)


class _FakeTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def render_conversation(self, conversation):
        length = conversation["length"]
        seed = conversation["seed"]
        return [2 + (seed + index) % 100 for index in range(length)], [0] * length


class _FakeDataset:
    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"length": self.lengths[index], "seed": index}


def _make_generator(
    *,
    lengths: list[int],
    max_seq_len: int = 16,
    device_batch_size: int = 2,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    buffer_size: int = 4,
):
    return sft_data_generator_bos_bestfit_core(
        dataset=_FakeDataset(lengths),
        tokenizer=_FakeTokenizer(),
        max_seq_len=max_seq_len,
        device_batch_size=device_batch_size,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=buffer_size,
    )


def test_yield_shapes_dtypes_and_produced_progress_contract() -> None:
    generator = _make_generator(lengths=[4] * 100, max_seq_len=8)
    inputs, targets, progress = next(generator)
    assert inputs.shape == targets.shape == (2, 8)
    assert inputs.dtype == t.int32
    assert targets.dtype == t.int64
    assert isinstance(progress, SFTGeneratorProgress)
    assert [field.name for field in fields(progress)] == [
        "micro_batch_count",
        "current_epoch",
        "consumed",
        "nonpadding_target_count",
        "supervised_target_count",
        "dataloader_state",
    ]
    assert progress.micro_batch_count == 1
    assert progress.current_epoch >= 1
    assert progress.consumed > 0
    assert progress.nonpadding_target_count == int((targets != -1).sum().item())
    assert progress.supervised_target_count == progress.nonpadding_target_count
    assert progress.dataloader_state["consumed"] == progress.consumed


def test_generator_has_no_optimizer_horizon_and_yields_indefinitely() -> None:
    generator = _make_generator(lengths=[3] * 4, max_seq_len=8, device_batch_size=1)
    snapshots = [next(generator)[2] for _ in range(20)]
    assert [snapshot.micro_batch_count for snapshot in snapshots] == list(range(1, 21))
    assert all(later.consumed >= earlier.consumed for earlier, later in pairwise(snapshots))


def test_oversized_conversations_are_truncated_without_starvation() -> None:
    generator = _make_generator(
        lengths=[20] * 10,
        max_seq_len=8,
        device_batch_size=1,
    )
    for _ in range(3):
        inputs, targets, _progress = next(generator)
        assert (inputs != 0).any()
        assert (targets != -1).any()


def test_padding_targets_are_ignored_after_content() -> None:
    generator = _make_generator(
        lengths=[7],
        max_seq_len=8,
        device_batch_size=1,
        buffer_size=2,
    )
    _inputs, targets, _progress = next(generator)
    assert (targets[0, :6] >= 0).all()
    assert (targets[0, 6:] == -1).all()


def test_fully_packed_row_has_no_padding_mask() -> None:
    generator = _make_generator(
        lengths=[3] * 10,
        max_seq_len=8,
        device_batch_size=1,
    )
    _inputs, targets, _progress = next(generator)
    assert (targets >= 0).all()


def test_ddp_ranks_start_from_distinct_dataset_stripes() -> None:
    rank_zero = _make_generator(
        lengths=[4, 5, 6, 7],
        ddp_rank=0,
        ddp_world_size=2,
        device_batch_size=1,
    )
    rank_one = _make_generator(
        lengths=[4, 5, 6, 7],
        ddp_rank=1,
        ddp_world_size=2,
        device_batch_size=1,
    )
    inputs_zero, _targets, _progress = next(rank_zero)
    inputs_one, _targets, _progress = next(rank_one)
    assert (inputs_zero != 0).sum().item() != (inputs_one != 0).sum().item()


def test_epoch_advances_when_dataset_wraps() -> None:
    generator = _make_generator(
        lengths=[3] * 2,
        max_seq_len=16,
        device_batch_size=1,
        buffer_size=2,
    )
    assert max(next(generator)[2].current_epoch for _ in range(3)) > 1
