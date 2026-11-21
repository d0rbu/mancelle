"""Decoder-only SFT loss-mask operation ordering."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

import steering.sft.iteration as sft_iteration
from steering.sft.loss_mask import LossMaskMode


class _ShortConversationDataset:
    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int) -> dict[str, Any]:
        return {}


class _ShortConversationTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def render_conversation(self, _conversation: dict[str, Any]) -> tuple[list[int], list[int]]:
        return [0, 2, 3, 4, 5, 6, 7], [0, 1, 1, 1, 1, 1, 1]


def test_completion_mask_runs_after_padding_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """The completion helper must receive targets with padding already ignored."""
    padding_start = 6
    helper_entered = False
    real_apply = sft_iteration.apply_completion_only_row_masks

    def spy(targets: t.Tensor, row_masks: list[list[int]]) -> None:
        nonlocal helper_entered
        helper_entered = True
        assert (targets[0, padding_start:] == -1).all()
        real_apply(targets, row_masks)

    monkeypatch.setattr(sft_iteration, "apply_completion_only_row_masks", spy)
    gen = sft_iteration.sft_data_generator_bos_bestfit_core(
        dataset=_ShortConversationDataset(),
        tokenizer=_ShortConversationTokenizer(),
        max_seq_len=8,
        device_batch_size=1,
        ddp_rank=0,
        ddp_world_size=1,
        device=t.device("cpu"),
        pin_memory=False,
        buffer_size=2,
        loss_mask_mode=LossMaskMode.COMPLETION_ONLY,
    )

    next(gen)
    assert helper_entered
