"""Shared input scaffolding for turn-RLVR tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

import steering.eval.chat_self_conditioned_generation as chat_generation
import steering.eval.inference_engine as inference
from steering.conditional.encoder_decoder.sft_grpo import ActionMode
from steering.conditional.encoder_decoder.sft_grpo_training import GRPOPolicyConfig
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.eval.target_conditioning import ContextRoutingMode


def policy() -> GRPOPolicyConfig:
    return GRPOPolicyConfig(
        action_mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        positive_counts=SamplingSelection(mean=1, std=1, min_k=1, max_k=2),
        negative_counts=NegativeSamplingSelection(mean=1, std=1, min_k=1, max_k=2),
        seed=0,
    )


def factory(fixture: Any, device: t.device | None = None):
    wrapper = fixture.engine._model
    resolved_device = t.device("cpu") if device is None else device

    def build(_rollout_index: int):
        return inference.build_engine_self_conditioned(
            fixture.model,
            fixture.tokenizer,
            resolved_device,
            wrapper._self_cond.config,
            context_provider=fixture.provider,
            context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE,
            turn_contract=fixture.contract,
        )

    return build


def script(monkeypatch: pytest.MonkeyPatch, tokens: list[int]) -> None:
    remaining = iter(tokens)

    def sample(_logits: t.Tensor, *_args: Any) -> t.Tensor:
        return t.tensor([[next(remaining)]])

    monkeypatch.setattr(chat_generation, "sample_next_token", sample)
