"""NaN/Inf guard tests for the canonical sample-next-token helper (post-#440).

Pre-consolidation: three duplicates of `_sample_token` lived in
nanochat/engine.py, steering/eval/inference_engine.py, and
steering/eval/multiseg_generator.py. Issue #440 consolidates them into
nanochat/sampling.py.

These tests:
1. Verify the canonical helper rejects non-finite logits at every sampler branch.
2. Verify that all general-purpose callers point at the single canonical helper
   (identity check) — the test fails loudly if anyone re-introduces a guarded
   copy. The self-conditioned fixed-row batch has one finite-input-only adapter
   because it validates all active rows together before preserving independent
   row RNGs; that adapter must not duplicate the guard.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch as t

from nanochat import sample_next_token as nanochat_root_sample_next_token
from nanochat.engine import sample_next_token as nanochat_engine_sample_next_token
from nanochat.sampling import _assert_finite_logits, sample_next_token

_SEED = 12345


def _bad_logits_nan() -> t.Tensor:
    return t.tensor([[0.0, math.nan, 1.0, 2.0]], dtype=t.float32)


def _bad_logits_pos_inf() -> t.Tensor:
    return t.tensor([[0.0, 1.0, math.inf, 2.0]], dtype=t.float32)


def _bad_logits_neg_inf() -> t.Tensor:
    return t.tensor([[0.0, 1.0, -math.inf, 2.0]], dtype=t.float32)


def _good_logits() -> t.Tensor:
    return t.tensor([[0.0, 1.0, 2.0, 0.5]], dtype=t.float32)


# ---------------------------------------------------------------------------
# Canonical helper — guard behaviour at every branch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "logits_factory",
    [_bad_logits_nan, _bad_logits_pos_inf, _bad_logits_neg_inf],
    ids=["nan", "pos_inf", "neg_inf"],
)
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("top_k", [None, 2])
def test_sample_next_token_rejects_non_finite(logits_factory, temperature, top_k):
    """All combinations of (bad logit, temperature, top_k) raise FloatingPointError."""
    rng = t.Generator(device="cpu")
    rng.manual_seed(_SEED)
    with pytest.raises(FloatingPointError):
        sample_next_token(
            logits_factory(), rng, temperature=temperature, top_k=top_k
        )


def test_sample_next_token_greedy_accepts_finite():
    out = sample_next_token(_good_logits(), None, temperature=0.0, top_k=None)
    assert out.shape == (1, 1)
    assert int(out.item()) == 2  # argmax over [0,1,2,0.5]


def test_sample_next_token_topk_accepts_finite():
    rng = t.Generator(device="cpu")
    rng.manual_seed(_SEED)
    out = sample_next_token(_good_logits(), rng, temperature=0.7, top_k=2)
    assert out.shape == (1, 1)
    assert 0 <= int(out.item()) < 4


def test_assert_finite_logits_message_lists_first_positions():
    bad = t.tensor([[1.0, math.nan, 2.0, math.inf]], dtype=t.float32)
    with pytest.raises(FloatingPointError) as excinfo:
        _assert_finite_logits(bad)
    msg = str(excinfo.value)
    assert "non-finite logits" in msg
    assert "[0, 1]" in msg or "0, 1" in msg
    assert "[0, 3]" in msg or "0, 3" in msg


# ---------------------------------------------------------------------------
# Identity check — all reference paths point at the SAME canonical function.
# A future divergence (re-introducing a private copy) breaks this test.
# ---------------------------------------------------------------------------


def test_nanochat_engine_re_exports_canonical_sample_next_token():
    assert nanochat_engine_sample_next_token is sample_next_token


def test_nanochat_root_re_exports_canonical_sample_next_token():
    assert nanochat_root_sample_next_token is sample_next_token


def test_steering_inference_engine_imports_canonical():
    """The steering InferenceEngine module imports sample_next_token from
    nanochat.sampling. Locks the dedup: any future copy would re-bind
    the name elsewhere, breaking the `is` check."""
    from steering.eval import inference_engine

    assert inference_engine.sample_next_token is sample_next_token


def test_steering_multiseg_generator_imports_canonical():
    """The steering multiseg_generator module imports sample_next_token
    from nanochat.sampling. Locks the dedup."""
    from steering.eval import multiseg_generator

    assert multiseg_generator.sample_next_token is sample_next_token


def test_no_duplicate_guarded_sampler_helpers_remain_in_steering():
    """Steering does not restore the deleted guarded sampler copies."""
    from steering.eval import multiseg_generator
    from steering.eval.inference_engine import InferenceEngine

    assert not hasattr(InferenceEngine, "_sample_token"), (
        "InferenceEngine._sample_token must be deleted after #440 dedup"
    )
    assert not hasattr(InferenceEngine, "_assert_finite_logits"), (
        "InferenceEngine._assert_finite_logits must be deleted after #440 dedup"
    )
    assert not hasattr(multiseg_generator, "_sample_token"), (
        "multiseg_generator._sample_token must be deleted after #440 dedup"
    )
    assert not hasattr(multiseg_generator, "_assert_finite_logits"), (
        "multiseg_generator._assert_finite_logits must be deleted after #440 dedup"
    )


def test_self_conditioned_finite_row_sampler_does_not_duplicate_guard():
    from steering.eval import inference_engine

    source = inspect.getsource(inference_engine._sample_next_token_from_finite_logits)
    assert "_assert_finite_logits" not in source
    assert "sample_next_token(" not in source
