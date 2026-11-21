"""Eval-contract rejection coverage for ``build_wrapper_for_features`` /
``build_engine_for_features`` (steering/eval/inference_engine.py).

These guards encode the encoder-decoder full-context contract:

* Decoder-only models do NOT support feature/target conditioning, so a
  non-empty ``features`` dict or any explicit ``target`` must raise.
* ``features`` and ``target`` are mutually exclusive.
* Encoder-decoder feature eval requires a real ``context_provider``.
* The valid "full-context, no target delta" path (empty features + a real
  provider) returns a ``TargetAwareEncoderDecoderEvalWrapper``.

All tests use the real tiny ``_make_gpt`` / ``_make_encdec`` fixtures and a
real (in-memory) context provider — no mock-only stubs.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.inference_engine import (
    InferenceEngine,
    build_engine_for_features,
    build_wrapper_for_features,
)
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    MultiTargetSegments,
    SingleTargetSegment,
    TargetConditioningMode,
)
from tests.conftest import MockTokenizer, _make_encdec, _make_gpt


class _RealContextProvider:
    """Minimal but real ``ContextProvider``: returns one full-prefix segment.

    This is a genuine implementation of the ``ContextProvider`` protocol
    (``get(token_ids) -> tuple[ContextFeatureSegment, ...] | None``), not a
    bound method on a SimpleNamespace. It conditions every prefix on a single
    feature (id 0, bucket 1) covering the whole decoded prefix span.
    """

    def __init__(self, char_end: int = 16) -> None:
        self._char_end = char_end

    def get(self, token_ids):
        token_ids = [int(tok) for tok in token_ids]
        if not token_ids:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=self._char_end,
            ),
        )


# ---------------------------------------------------------------------------
# Decoder-only rejection paths
# ---------------------------------------------------------------------------


def test_decoder_only_with_nonempty_features_raises() -> None:
    """A decoder-only GPT cannot accept feature conditioning."""
    model = _make_gpt(device="cpu")
    tokenizer = MockTokenizer()
    with pytest.raises(ValueError, match="decoder-only models do not support"):
        build_wrapper_for_features(
            model,
            tokenizer,
            features={0: 1},
            context_provider=None,
        )


def test_decoder_only_with_explicit_target_raises() -> None:
    """A decoder-only GPT cannot accept an explicit target segment."""
    model = _make_gpt(device="cpu")
    tokenizer = MockTokenizer()
    target = SingleTargetSegment(features={0: 1})
    with pytest.raises(ValueError, match="decoder-only models do not support"):
        build_wrapper_for_features(
            model,
            tokenizer,
            target=target,
            context_provider=None,
        )


def test_decoder_only_with_empty_features_passes_through_unchanged() -> None:
    """Decoder-only + empty features returns the model itself (pass-through)."""
    model = _make_gpt(device="cpu")
    tokenizer = MockTokenizer()

    returned_empty = build_wrapper_for_features(
        model,
        tokenizer,
        features={},
        context_provider=None,
    )
    returned_none = build_wrapper_for_features(
        model,
        tokenizer,
        features=None,
        context_provider=None,
    )

    assert returned_empty is model
    assert returned_none is model


def test_decoder_only_engine_wraps_unmodified_model() -> None:
    """The engine builder threads the pass-through decoder unchanged."""
    model = _make_gpt(device="cpu")
    tokenizer = MockTokenizer()

    engine = build_engine_for_features(
        model,
        tokenizer,
        t.device("cpu"),
        features=None,
        context_provider=None,
    )

    assert isinstance(engine, InferenceEngine)
    # The engine must hold the *original* decoder, not a wrapper.
    assert engine._model is model
    assert not isinstance(engine._model, TargetAwareEncoderDecoderEvalWrapper)


# ---------------------------------------------------------------------------
# features XOR target
# ---------------------------------------------------------------------------


def test_both_features_and_target_raises() -> None:
    """Passing both ``features`` and ``target`` is rejected before model checks.

    Uses an encoder-decoder model so the only thing that can trip the guard is
    the mutual-exclusion check itself (not the decoder-only rejection).
    """
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    target = SingleTargetSegment(features={0: 1})
    with pytest.raises(ValueError, match="pass features OR target"):
        build_wrapper_for_features(
            model,
            tokenizer,
            features={0: 1},
            target=target,
            context_provider=_RealContextProvider(),
        )


def test_both_features_and_target_raises_even_for_decoder_only() -> None:
    """The XOR check fires regardless of model type (it is checked first)."""
    model = _make_gpt(device="cpu")
    tokenizer = MockTokenizer()
    target = SingleTargetSegment(features={})
    with pytest.raises(ValueError, match="pass features OR target"):
        build_wrapper_for_features(
            model,
            tokenizer,
            features={},
            target=target,
            context_provider=None,
        )


# ---------------------------------------------------------------------------
# Encoder-decoder requires a context provider
# ---------------------------------------------------------------------------


def test_encdec_without_context_provider_raises() -> None:
    """Encoder-decoder feature eval requires a context_provider."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    with pytest.raises(ValueError, match="requires a context_provider"):
        build_wrapper_for_features(
            model,
            tokenizer,
            features={0: 1},
            context_provider=None,
        )


def test_encdec_empty_features_without_provider_still_raises() -> None:
    """Even the no-delta encoder-decoder path needs a real provider."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    with pytest.raises(ValueError, match="requires a context_provider"):
        build_wrapper_for_features(
            model,
            tokenizer,
            features={},
            context_provider=None,
        )


# ---------------------------------------------------------------------------
# Valid encoder-decoder full-context paths
# ---------------------------------------------------------------------------


def test_encdec_empty_features_with_provider_returns_wrapper() -> None:
    """The valid full-context, no-delta path returns a target-aware wrapper."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    provider = _RealContextProvider()

    wrapper = build_wrapper_for_features(
        model,
        tokenizer,
        features={},
        context_provider=provider,
    )

    assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
    assert wrapper.context_provider is provider
    assert wrapper.model is model
    # Empty features resolve to a SingleTargetSegment with an empty dict.
    assert wrapper._target_segments == [{}]
    # Default routing mode is CURRENT.
    assert wrapper.target_conditioning_mode is TargetConditioningMode.CURRENT


def test_encdec_nonempty_features_with_provider_returns_wrapper() -> None:
    """A non-empty target-delta dict is wrapped as a SingleTargetSegment."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    provider = _RealContextProvider()

    wrapper = build_wrapper_for_features(
        model,
        tokenizer,
        features={0: 1, 2: 3},
        context_provider=provider,
    )

    assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
    assert wrapper._target_segments == [{0: 1, 2: 3}]


def test_encdec_explicit_target_is_threaded_through() -> None:
    """An explicit MultiTargetSegments target is honored (PR-C lockstep path)."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    provider = _RealContextProvider()
    target = MultiTargetSegments(
        segments=[{0: 1}, {1: 2}],
        token_assignments=t.tensor([0, 0, 1], dtype=t.int64),
    )

    wrapper = build_wrapper_for_features(
        model,
        tokenizer,
        target=target,
        context_provider=provider,
    )

    assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
    assert wrapper._target_segments == [{0: 1}, {1: 2}]


def test_encdec_target_mode_next_is_forwarded() -> None:
    """``target_mode`` is forwarded to the wrapper (NEXT routing)."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    provider = _RealContextProvider()

    wrapper = build_wrapper_for_features(
        model,
        tokenizer,
        features={0: 1},
        context_provider=provider,
        target_mode=TargetConditioningMode.NEXT,
    )

    assert wrapper.target_conditioning_mode is TargetConditioningMode.NEXT


def test_build_engine_for_features_wraps_encdec_in_wrapper() -> None:
    """The engine builder produces an InferenceEngine over the wrapper."""
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()
    provider = _RealContextProvider()

    engine = build_engine_for_features(
        model,
        tokenizer,
        t.device("cpu"),
        features={},
        context_provider=provider,
    )

    assert isinstance(engine, InferenceEngine)
    assert isinstance(engine._model, TargetAwareEncoderDecoderEvalWrapper)
    assert engine._model.context_provider is provider
