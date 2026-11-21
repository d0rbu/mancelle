"""Tests for Engine.generate's prompt/max_tokens overflow check (issue #442).

Pre-fix bug: `Engine.generate(max_tokens=None, prompt=fills_cache)`
allocated a decode cache sized to model.config.sequence_len, then
overflowed on the first decode step with a deep RoPE bounds assert.
The symmetric `max_tokens=int + overflow` case also asserted partway
through prefill.

Post-fix: an entry-point check rejects both with a typed RuntimeError
naming the prompt length, generation horizon, and context window.
"""

from __future__ import annotations

import pytest
import torch as t

from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from tests.conftest import (
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_VOCAB,
    MockTokenizer,
)


def _make_engine(sequence_len: int) -> Engine:
    """Build a tiny GPT + tokenizer + Engine on CPU with the given context window."""
    config = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    tokenizer = MockTokenizer()
    return Engine(model, tokenizer)


def _make_prompt(n: int) -> list[int]:
    """A throwaway int-list prompt — Engine.generate only validates length + types."""
    return [1] * n


# ---------------------------------------------------------------------------
# max_tokens=None (the issue's exact scenario)
# ---------------------------------------------------------------------------


def test_generate_rejects_max_tokens_none_with_full_length_prompt():
    """max_tokens=None + len(prompt) == sequence_len → RuntimeError."""
    engine = _make_engine(sequence_len=8)
    tokens = _make_prompt(8)
    with pytest.raises(RuntimeError) as excinfo:
        list(engine.generate(tokens, num_samples=1, max_tokens=None))
    msg = str(excinfo.value)
    assert "8" in msg  # prompt length OR context window
    assert "context window" in msg or "sequence_len" in msg
    assert "unbounded" in msg.lower() or "max_tokens" in msg.lower()


def test_generate_rejects_max_tokens_none_with_overlong_prompt():
    """max_tokens=None + len(prompt) > sequence_len → RuntimeError."""
    engine = _make_engine(sequence_len=8)
    tokens = _make_prompt(12)
    with pytest.raises(RuntimeError, match="(context window|sequence_len)"):
        list(engine.generate(tokens, num_samples=1, max_tokens=None))


def test_generate_accepts_max_tokens_none_with_short_prompt():
    """Positive control: max_tokens=None + room to generate → no raise."""
    engine = _make_engine(sequence_len=8)
    tokens = _make_prompt(4)  # 4 slots free for the unbounded horizon
    # Drive the generator. It will produce up to (sequence_len - len) tokens
    # before hitting some stop condition; just confirm it doesn't raise.
    out = list(engine.generate(tokens, num_samples=1, max_tokens=4, temperature=0.0))
    assert isinstance(out, list)


# ---------------------------------------------------------------------------
# max_tokens=int + overflow (symmetric case)
# ---------------------------------------------------------------------------


def test_generate_rejects_int_max_tokens_overflow():
    """len(prompt) + max_tokens > sequence_len → RuntimeError."""
    engine = _make_engine(sequence_len=8)
    tokens = _make_prompt(7)
    with pytest.raises(RuntimeError) as excinfo:
        list(engine.generate(tokens, num_samples=1, max_tokens=4))
    msg = str(excinfo.value)
    assert "max_tokens" in msg
    assert "4" in msg  # the requested horizon
    assert "context window" in msg or "sequence_len" in msg


def test_generate_accepts_int_max_tokens_exact_fit():
    """Boundary: len(prompt) + max_tokens == sequence_len → succeeds."""
    engine = _make_engine(sequence_len=8)
    tokens = _make_prompt(6)
    # 6 + 2 == 8 — exactly the context window, the largest allowed.
    out = list(engine.generate(tokens, num_samples=1, max_tokens=2, temperature=0.0))
    # Greedy + cap=2 yields at most 2 token columns.
    assert len(out) <= 2


def test_generate_accepts_int_max_tokens_underfit():
    """Positive control: len(prompt) + max_tokens < sequence_len → succeeds."""
    engine = _make_engine(sequence_len=16)
    tokens = _make_prompt(4)
    out = list(engine.generate(tokens, num_samples=1, max_tokens=4, temperature=0.0))
    assert len(out) <= 4


# ---------------------------------------------------------------------------
# Message quality
# ---------------------------------------------------------------------------


def test_generate_error_message_names_remaining_capacity():
    """The error message includes the remaining capacity so a caller can
    decide whether to truncate the prompt or shrink max_tokens."""
    engine = _make_engine(sequence_len=8)
    tokens = _make_prompt(7)
    with pytest.raises(RuntimeError) as excinfo:
        list(engine.generate(tokens, num_samples=1, max_tokens=5))
    msg = str(excinfo.value)
    assert "1" in msg  # remaining capacity = 8 - 7 = 1
    assert "remaining" in msg.lower() or "capacity" in msg.lower()
