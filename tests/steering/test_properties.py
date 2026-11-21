"""Property-based tests for slice 13 invariants (post-#440 consolidation).

Invariants locked here:

1. `sample_next_token` raises FloatingPointError for any non-finite logit.
2. `_validate_prompt_length` boundary: n_tokens + max_new_tokens > sequence_len
   raises; <= does not.
3. `sample_next_token` greedy is deterministic regardless of seed (no RNG draw).
4. `sample_next_token` topk + seed is deterministic across calls.
"""

from __future__ import annotations

import math

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from nanochat.sampling import sample_next_token
from steering.eval.inference_engine import InferenceEngine
from tests.conftest import MockTokenizer, _make_gpt

_DEVICE = t.device("cpu")


# ---------------------------------------------------------------------------
# Hypothesis settings: derandomize=True, low max_examples for fast CI.
# ---------------------------------------------------------------------------
_PROFILE = settings(
    max_examples=50,
    derandomize=True,
    deadline=None,
)


def _seq_with_non_finite(
    vocab_size: int, bad_idx: int, value: float
) -> t.Tensor:
    """A (1, V) logit row with the bad value at ``bad_idx``."""
    out = t.zeros(1, vocab_size, dtype=t.float32)
    out[0, bad_idx] = value
    return out


# ---------------------------------------------------------------------------
# Property 1: any non-finite slot in the logits row raises.
# ---------------------------------------------------------------------------


@_PROFILE
@given(
    vocab_size=st.integers(min_value=4, max_value=64),
    bad_idx=st.integers(min_value=0, max_value=63),
    value=st.sampled_from([math.nan, math.inf, -math.inf]),
)
def test_property_sample_next_token_non_finite_raises_top_k(
    vocab_size, bad_idx, value
):
    bad_idx = bad_idx % vocab_size
    logits = _seq_with_non_finite(vocab_size, bad_idx, value)
    rng = t.Generator(device="cpu")
    rng.manual_seed(7)
    with pytest.raises(FloatingPointError):
        sample_next_token(logits, rng, temperature=0.7, top_k=4)


# ---------------------------------------------------------------------------
# Property 2: prompt-length validator boundary.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine_factory_module():
    """Module-scoped tiny engine — only the validator is exercised."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    return InferenceEngine(model, tok, _DEVICE)


@_PROFILE
@given(
    n_tokens=st.integers(min_value=0, max_value=200),
    max_new_tokens=st.integers(min_value=0, max_value=200),
)
def test_property_validate_prompt_length_boundary(
    engine_factory_module, n_tokens, max_new_tokens
):
    """For any non-negative ``(n_tokens, max_new_tokens)``, the validator
    raises iff ``n_tokens + max_new_tokens > sequence_len``."""
    engine = engine_factory_module
    sl = int(engine._dec_config.sequence_len)
    if n_tokens + max_new_tokens > sl:
        with pytest.raises(ValueError, match="exceeds context window"):
            engine._validate_prompt_length(n_tokens, max_new_tokens)
    else:
        # Should not raise.
        engine._validate_prompt_length(n_tokens, max_new_tokens)


# ---------------------------------------------------------------------------
# Property 3: greedy is deterministic regardless of seed.
# ---------------------------------------------------------------------------


@_PROFILE
@given(
    vocab_size=st.integers(min_value=2, max_value=64),
    seed_a=st.integers(min_value=0, max_value=2**31 - 1),
    seed_b=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_sample_next_token_greedy_deterministic_across_seeds(
    vocab_size, seed_a, seed_b
):
    """Greedy returns argmax regardless of the seed."""
    g = t.Generator(device="cpu")
    g.manual_seed(seed_a)
    logits = t.randn(1, vocab_size, generator=g)
    # Two RNGs with different seeds; greedy ignores rng entirely.
    rng_a = t.Generator(device="cpu")
    rng_a.manual_seed(seed_a)
    rng_b = t.Generator(device="cpu")
    rng_b.manual_seed(seed_b)
    out_a = sample_next_token(
        logits, rng_a, temperature=0.0, top_k=None
    )
    out_b = sample_next_token(
        logits, rng_b, temperature=0.0, top_k=None
    )
    assert int(out_a.item()) == int(out_b.item())


# ---------------------------------------------------------------------------
# Property 4: top_k + seeded sampling is deterministic across calls.
# ---------------------------------------------------------------------------


@_PROFILE
@given(
    vocab_size=st.integers(min_value=4, max_value=32),
    top_k=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_property_sample_next_token_topk_deterministic_across_calls(
    vocab_size, top_k, seed
):
    """A given ``(seed, logits, top_k, temperature>0)`` always yields the
    same token. Lock determinism for parity / reproducibility tests."""
    g = t.Generator(device="cpu")
    g.manual_seed(seed)
    logits = t.randn(1, vocab_size, generator=g)
    capped = min(top_k, vocab_size)

    rng_a = t.Generator(device="cpu")
    rng_a.manual_seed(seed)
    rng_b = t.Generator(device="cpu")
    rng_b.manual_seed(seed)
    out_a = sample_next_token(
        logits, rng_a, temperature=1.0, top_k=capped
    )
    out_b = sample_next_token(
        logits, rng_b, temperature=1.0, top_k=capped
    )
    assert int(out_a.item()) == int(out_b.item())


# ---------------------------------------------------------------------------
# Property 5: ``sample_next_token`` output shape contract.
# ---------------------------------------------------------------------------


@_PROFILE
@given(
    batch=st.integers(min_value=1, max_value=8),
    vocab_size=st.integers(min_value=2, max_value=64),
    temperature=st.sampled_from([0.0, 0.5, 1.0]),
    top_k=st.sampled_from([None, 2, 4]),
)
def test_property_sample_token_shape_is_batch_1(
    batch, vocab_size, temperature, top_k
):
    """The sampler always returns a ``(B, 1)`` tensor."""
    g = t.Generator(device="cpu")
    g.manual_seed(13)
    logits = t.randn(batch, vocab_size, generator=g)
    rng = t.Generator(device="cpu")
    rng.manual_seed(13)
    if top_k is not None:
        top_k = min(top_k, vocab_size)
    out = sample_next_token(
        logits, rng, temperature=temperature, top_k=top_k
    )
    assert out.shape == (batch, 1)
    # Tokens must be valid vocab indices.
    assert (out >= 0).all() and (out < vocab_size).all()
