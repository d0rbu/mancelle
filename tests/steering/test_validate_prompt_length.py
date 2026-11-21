"""Boundary tests for ``InferenceEngine._validate_prompt_length``.

Map SB-13-P1-01 / manifest §7:

    The KV cache is allocated to ``len(token_ids) + max_new_tokens``; the
    decoder's RoPE bounds-check uses ``T0 + T <= cos.size(1)``. So a prompt
    of exactly ``sequence_len`` is OK only when ``max_new_tokens == 0``.
    Previously the helper rejected only ``n_tokens > sequence_len``,
    allowing ``n_tokens == sequence_len`` with ``max_new_tokens > 0`` to
    slip through and assert deep inside ``gpt.py``.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.eval.inference_engine import InferenceEngine
from tests.conftest import MockTokenizer, _make_gpt

_DEVICE = t.device("cpu")


@pytest.fixture()
def engine() -> InferenceEngine:
    """A real InferenceEngine wrapping a tiny GPT on CPU.

    The wrapping is required so that ``self._dec_config.sequence_len`` is
    a real number; we do NOT exercise the model itself, only the prompt-
    length validator.
    """
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    return InferenceEngine(model, tok, _DEVICE)


def _seq_len(engine: InferenceEngine) -> int:
    return int(engine._dec_config.sequence_len)


def test_validate_prompt_length_under_limit_no_decode(engine):
    """A prompt strictly shorter than the context window with no decode
    budget is fine."""
    engine._validate_prompt_length(_seq_len(engine) - 1, max_new_tokens=0)


def test_validate_prompt_length_at_limit_no_decode(engine):
    """A prompt exactly equal to the context window with no decode budget
    is fine — the cache holds the prefill, RoPE check is ``<=``."""
    engine._validate_prompt_length(_seq_len(engine), max_new_tokens=0)


def test_validate_prompt_length_over_limit_no_decode_raises(engine):
    """A prompt longer than the context window raises."""
    with pytest.raises(ValueError, match="exceeds context window"):
        engine._validate_prompt_length(_seq_len(engine) + 1, max_new_tokens=0)


def test_validate_prompt_length_at_limit_with_decode_raises(engine):
    """A prompt exactly equal to the context window with any positive
    decode budget MUST raise. Previously this slipped through and asserted
    deep inside ``gpt.py``'s RoPE bounds check (SB-13-P1-01)."""
    with pytest.raises(ValueError, match="exceeds context window"):
        engine._validate_prompt_length(_seq_len(engine), max_new_tokens=1)


def test_validate_prompt_length_under_limit_with_decode(engine):
    """If ``n_tokens + max_new_tokens <= sequence_len``, no raise."""
    engine._validate_prompt_length(_seq_len(engine) - 4, max_new_tokens=4)


def test_validate_prompt_length_under_limit_with_decode_overflow_raises(engine):
    """The new helper sums prompt + decode and rejects sums over the
    window."""
    with pytest.raises(ValueError, match="exceeds context window"):
        engine._validate_prompt_length(_seq_len(engine) - 3, max_new_tokens=4)


def test_validate_prompt_length_default_max_new_tokens_is_zero(engine):
    """For backwards compatibility, the helper's ``max_new_tokens`` default
    is 0 (matches the pre-fix prompt-only validation)."""
    sl = _seq_len(engine)
    # Prompt of exactly sequence_len passes when max_new_tokens defaults to 0.
    engine._validate_prompt_length(sl)
    # Prompt of sequence_len+1 raises even with default 0.
    with pytest.raises(ValueError):
        engine._validate_prompt_length(sl + 1)


def test_validate_prompt_length_zero_prompt_with_decode(engine):
    """A zero-token prompt with the full decode budget is allowed
    (degenerate but valid edge case for the validator)."""
    engine._validate_prompt_length(0, max_new_tokens=_seq_len(engine))


def test_validate_prompt_length_message_includes_both_values(engine):
    """The error message names both the prompt length and the decode budget
    so triage is easy."""
    sl = _seq_len(engine)
    with pytest.raises(ValueError) as excinfo:
        engine._validate_prompt_length(sl, max_new_tokens=8)
    msg = str(excinfo.value)
    assert str(sl) in msg
    assert "8" in msg
