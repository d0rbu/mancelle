"""The GCG seed boundary precedes the assistant marker, including turn K=1."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as attention
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import eval_adapter
from tests.conftest import MockTokenizer
from tests.steering.test_self_conditioned_wrapper import (
    _build_fp_encdec,
    _make_turn_wrapper,
    _new_kv,
)


@pytest.mark.parametrize("segment_size", [1, 3])
def test_prompt_seed_exposes_s0_before_marker_transition(
    segment_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    with t.random.fork_rng(devices=[]):
        t.manual_seed(7123)
        model = _build_fp_encdec()
    model.eval()
    wrapper, prompt, _segments = _make_turn_wrapper(model, MockTokenizer(), K=segment_size)
    runtime = wrapper._self_cond
    assert runtime is not None
    captured: list[tuple[Any, int, int]] = []
    seed = eval_adapter.seed_self_conditioning_from_prompt

    def record_seed(**kwargs: Any) -> Any:
        result = seed(**kwargs)
        captured.append(
            (result, kwargs["encoder_out"].layout.n_segments, kwargs["kv_cache"].get_pos())
        )
        return result

    monkeypatch.setattr(eval_adapter, "seed_self_conditioning_from_prompt", record_seed)
    cache = _new_kv(model)
    with t.inference_mode(), wrapper.conditioning_context(context_token_ids=[prompt]):
        output = wrapper(t.tensor([prompt]), kv_cache=cache)
    assert output.shape == (1, 1, model.dec_config.vocab_size)
    assert len(captured) == 1
    prompt_seed, real_slots, prefix_position = captured[0]
    assert real_slots == 1
    assert prefix_position == len(prompt) - 1
    assert prompt_seed.source_logits.shape[1] == len(prompt) - 1
    assert not prompt_seed.source_logits.requires_grad
    assert prompt_seed.state.target_slot_idx == real_slots
    assert prompt_seed.state.encoder_out.layout.n_segments == real_slots + 1
    # Only the normal generation path commits the assistant marker. At K=1 it
    # creates S1 immediately, proving that a post-prefill snapshot is too late.
    snapshot = wrapper.self_cond_runtime_snapshot()
    assert cache.get_pos() == len(prompt)
    assert snapshot.state.target_slot_idx == real_slots + (segment_size == 1)
    assert snapshot.state.encoder_out.layout.n_segments == real_slots + 1 + (segment_size == 1)
    assert snapshot.tokens_in_seg == (0 if segment_size == 1 else 1)
    assert prompt_seed.state.encoder_out.layout.n_segments == 2
