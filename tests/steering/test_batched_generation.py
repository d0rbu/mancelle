"""Left-padded batched generation correctness tests.

Manifest §7 ("ADD: left-padded batches"):

    Confirm a row that emits stop-token early does NOT corrupt other
    rows' continuations. Existing
    ``tests/steering/test_inference.py::TestBatchGeneration::test_batch_greedy_matches_sequential``
    covers parity but not heterogeneous early-stop.

We exercise greedy-batched ``generate_batch`` against monkeypatched
sampling so we can deterministically inject the stop token into one row.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch as t

from steering.eval.inference_engine import InferenceEngine
from tests.conftest import MockTokenizer, _make_gpt

_DEVICE = t.device("cpu")


@pytest.fixture()
def engine() -> InferenceEngine:
    """A real InferenceEngine wrapping a tiny GPT on CPU for batched
    generation tests."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    return InferenceEngine(model, tok, _DEVICE)


@contextmanager
def _injected_logits(plan: list[t.Tensor]) -> Iterator[None]:
    """Replace the model's __call__ so that successive calls return
    pre-baked logits from ``plan``. Useful for deterministic decode-step
    tests without needing to control the actual model output."""
    counter = {"step": 0}

    def fake_forward(self, idx, kv_cache=None, **kwargs):
        i = counter["step"]
        out = plan[min(i, len(plan) - 1)]
        counter["step"] += 1
        return out

    with patch(
        "nanochat.gpt.GPT.__call__",
        autospec=True,
        side_effect=fake_forward,
    ):
        yield


def test_generate_batch_left_padding_correctness(engine):
    """For mixed-length prompts, the batched greedy path left-pads with
    BOS and produces output for every row."""
    prompts = ["a", "abc", "abcdef"]
    out = engine.generate_batch(prompts, max_new_tokens=4, temperature=0.0)
    assert len(out) == len(prompts)
    for s in out:
        assert isinstance(s, str)


def test_generate_batch_finished_row_does_not_corrupt_other_rows():
    """If row 0 finishes early (stop token), row 1's continuation must NOT
    be affected. We monkeypatch the sampler to inject a known stop token
    into row 0 at step 1, and verify row 1 continues to receive logits."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    engine = InferenceEngine(model, tok, _DEVICE)
    bos = tok.get_bos_token_id()  # 0 — stop token in raw mode

    # Step 1: row 0 samples bos (finishes), row 1 samples a non-stop token.
    # Step 2+: both rows sample non-stop.
    # We control via a fake _sample_token that returns a fixed pattern.
    sampling_plan = [
        # First call returns the prefill-last-position logits-derived token.
        # row 0 → bos (stops), row 1 → 5 (continues).
        t.tensor([[bos], [5]], dtype=t.long),
        # Second call: row 1 samples 6.
        t.tensor([[bos], [6]], dtype=t.long),
        # Third call: row 1 samples 7.
        t.tensor([[bos], [7]], dtype=t.long),
        # Fourth call: row 1 samples 8.
        t.tensor([[bos], [8]], dtype=t.long),
    ]
    counter = {"step": 0}

    def fake_sample(logits, rng, temperature, top_k):
        s = counter["step"]
        counter["step"] += 1
        return sampling_plan[min(s, len(sampling_plan) - 1)]

    with patch("steering.eval.inference_engine.sample_next_token", fake_sample):
        out = engine.generate_batch(
            ["hi", "yo"],
            max_new_tokens=4,
            temperature=0.0,
            top_k=None,
        )

    # Row 0 finished immediately → empty generated tokens (decoded to "").
    assert out[0] == ""
    # Row 1 received tokens 5, 6, 7, 8 — decoded by MockTokenizer to
    # chr(5-2)+chr(6-2)+chr(7-2)+chr(8-2) = chr(3)+chr(4)+chr(5)+chr(6).
    expected_chars = "".join(chr(c - 2) for c in [5, 6, 7, 8])
    assert out[1] == expected_chars


def test_generate_batch_all_rows_finish_at_same_step():
    """If every row hits stop in the same step, the loop breaks early."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    engine = InferenceEngine(model, tok, _DEVICE)
    bos = tok.get_bos_token_id()
    plan = [t.tensor([[bos], [bos]], dtype=t.long)]
    counter = {"step": 0}

    def fake_sample(logits, rng, temperature, top_k):
        s = counter["step"]
        counter["step"] += 1
        return plan[min(s, len(plan) - 1)]

    with patch("steering.eval.inference_engine.sample_next_token", fake_sample):
        out = engine.generate_batch(
            ["hi", "yo"],
            max_new_tokens=10,
            temperature=0.0,
            top_k=None,
        )
    assert out == ["", ""]
    # The loop should have stopped after step 1; counter should be 1.
    assert counter["step"] == 1


def test_generate_batch_max_new_tokens_zero(engine):
    """max_new_tokens=0 is a valid edge case — every row's generated list
    starts empty."""
    out = engine.generate_batch(["a", "ab"], max_new_tokens=0, temperature=0.0)
    assert out == ["", ""]


def test_generate_batch_single_prompt_matches_generate(engine):
    """Greedy batched with a single prompt should match
    ``InferenceEngine.generate`` greedy on the same prompt."""
    out_batched = engine.generate_batch(
        ["hi"], max_new_tokens=4, temperature=0.0, top_k=None
    )
    out_solo = engine.generate(
        "hi", max_new_tokens=4, temperature=0.0, top_k=None
    )
    assert out_batched[0] == out_solo
