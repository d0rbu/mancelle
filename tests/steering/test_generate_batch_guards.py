"""Guards on ``InferenceEngine.generate_batch``.

Map SB-13-P0-03 (greedy + non-None top_k silently dropped) and
SB-13-P0-04 (stop-token UNION across prompts hides divergence).

Both guards are slice-13 P0s; the fixes raise ``ValueError`` /
``RuntimeError`` instead of silently dropping/UNIONing.
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

    Greedy batched path is gated on decoder-only; tiny GPT is decoder-only,
    so this engine exercises the greedy batched path.
    """
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    return InferenceEngine(model, tok, _DEVICE)


# ---------------------------------------------------------------------------
# A5 / SB-13-P0-03: greedy + non-None top_k must raise.
# ---------------------------------------------------------------------------


def test_generate_batch_greedy_with_top_k_raises(engine):
    """Greedy is mathematically argmax-only; ``top_k`` is silently dropped
    in the greedy batched path. Reject explicitly so the contract drift
    is visible."""
    with pytest.raises(ValueError, match="does not honour top_k"):
        engine.generate_batch(
            ["hi"],
            max_new_tokens=4,
            temperature=0.0,
            top_k=10,
        )


def test_generate_batch_greedy_with_top_k_zero_does_not_raise(engine):
    """``top_k=0`` is the documented "ignore" value in some samplers; the
    guard should reject ANY non-None top_k under greedy because the docs
    are inconsistent. We choose to reject 0 too."""
    with pytest.raises(ValueError, match="does not honour top_k"):
        engine.generate_batch(
            ["hi"],
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
        )


def test_generate_batch_greedy_top_k_none_runs(engine):
    """The standard greedy path (top_k=None) still works."""
    out = engine.generate_batch(
        ["hi", "yo"],
        max_new_tokens=2,
        temperature=0.0,
        top_k=None,
    )
    assert isinstance(out, list)
    assert len(out) == 2


def test_generate_batch_sampling_top_k_runs(engine):
    """Sampling (temperature > 0) sequential path honours top_k as before
    — we did not break that path."""
    out = engine.generate_batch(
        ["hi"],
        max_new_tokens=2,
        temperature=0.7,
        top_k=5,
        seed=11,
    )
    assert isinstance(out, list)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# A6 / SB-13-P0-04: stop-token-set divergence assertion.
# ---------------------------------------------------------------------------


class _DivergentTokenizer(MockTokenizer):
    """A tokenizer that returns a different ``<assistant_end>`` id for each
    call. Simulates the hazard SB-13-P0-04 warns about: a future tokenizer
    or template setup yielding heterogeneous stop sets across prompts in a
    single batch."""

    def __init__(self):
        super().__init__()
        self._call_n = 0

    def encode_special(self, name: str) -> int:
        if name == "<assistant_end>":
            # Vary the id across calls; this should never happen in
            # production but the guard makes the assumption explicit.
            self._call_n += 1
            return 67 + self._call_n  # divergent each call
        return super().encode_special(name)


def test_generate_batch_render_as_chat_divergent_stops_raises():
    """If tokenizer yields different stop-token sets per prompt, raise
    rather than UNIONing silently. The current single-tokenizer-per-engine
    usage never hits this, but the data flow shouldn't make the assumption
    invisible (SB-13-P0-04)."""
    model = _make_gpt(device="cpu")
    tok = _DivergentTokenizer()
    engine = InferenceEngine(model, tok, _DEVICE)
    with pytest.raises(RuntimeError, match="stop_token_ids must be identical"):
        engine.generate_batch(
            ["hi", "yo"],
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            render_as_chat=True,
        )


def test_generate_batch_render_as_chat_same_stops_runs():
    """Sanity: a normal tokenizer that returns identical stop sets across
    prompts goes through unscathed."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    engine = InferenceEngine(model, tok, _DEVICE)
    out = engine.generate_batch(
        ["hi", "yo"],
        max_new_tokens=2,
        temperature=0.0,
        top_k=None,
        render_as_chat=True,
    )
    assert isinstance(out, list)
    assert len(out) == 2


def test_generate_batch_empty_prompts(engine):
    """Sanity: empty prompts list returns empty list (regression for
    existing behaviour after our guards)."""
    out = engine.generate_batch([], max_new_tokens=2, temperature=0.0)
    assert out == []
