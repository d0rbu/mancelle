"""End-to-end integration: real on-disk context cache → production provider →
TargetAwareEncoderDecoderEvalWrapper → forward / generate.

Most wrapper tests stub the context provider (a bound ``get`` on a
SimpleNamespace or a hand-rolled class). This test instead writes a REAL
safetensors context cache with the production :func:`save_context_cache`,
loads it back through the production :class:`SafetensorsContextFeatureProvider`,
and drives a real ``_make_encdec`` wrapper. It then proves the behavioral
contract that the whole pipeline exists to deliver: the cached context
features actually SHIFT the decoder logits versus a cache whose context
segment carries no active features.

A second test runs ``InferenceEngine.generate`` over the real on-disk cache to
exercise the cache-file → provider → wrapper → generate path that unit tests
stub out.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch as t

from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.context_cache_io import (
    CacheIdentity,
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.inference_engine import build_engine_for_features
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SingleTargetSegment,
    TargetConditioningMode,
)
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer, _make_encdec

# Tiny-model dimensions (must match tests/conftest._make_encdec).
_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(64)
_NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)
_IDENTITY = CacheIdentity(
    model_id="tiny-test-model",
    sae_id="tiny-test-sae",
    layer=0,
    pooling="mean",
    feature_selection_hash="a" * 64,
)


def _encode(tokenizer: MockTokenizer, text: str) -> list[int]:
    """Encode ``text`` to a flat list of ints (MockTokenizer.encode's str path)."""
    return cast("list[int]", tokenizer.encode(text))


def _prefix_ids(tokenizer: MockTokenizer, text: str) -> list[int]:
    """Build a BOS-prefixed token list for ``text``."""
    return [tokenizer.get_bos_token_id(), *_encode(tokenizer, text)]


def _prefix_char_end(tokenizer: MockTokenizer, prefix_tokens: list[int]) -> int:
    return int(token_to_char_offsets(tokenizer, prefix_tokens)[-1])


def _write_cache(
    cache_dir: Path,
    prefix_tokens: list[int],
    segments: tuple[ContextFeatureSegment, ...],
) -> SafetensorsContextFeatureProvider:
    """Write a one-row context cache and return a loaded provider."""
    entry = ContextCacheEntry.from_sequences(prefix_tokens, segments)
    save_context_cache(
        cache_dir,
        [entry],
        identity=_IDENTITY,
        n_sae_features=_N_SAE_FEATURES,
        num_buckets=_NUM_BUCKETS,
        overwrite=True,
    )
    return SafetensorsContextFeatureProvider(
        cache_dir,
        identity=_IDENTITY,
        expected_n_sae_features=_N_SAE_FEATURES,
        expected_num_buckets=_NUM_BUCKETS,
    )


def test_real_cache_roundtrip_provider_reads_back_segments(tmp_path: Path) -> None:
    """Sanity: the production write→read roundtrip returns the exact row."""
    tokenizer = MockTokenizer()
    prefix_tokens = _prefix_ids(tokenizer, "hello world")
    char_end = _prefix_char_end(tokenizer, prefix_tokens)
    seg = ContextFeatureSegment(
        feature_ids=(5,), buckets=(2,), char_start=0, char_end=char_end,
    )
    provider = _write_cache(tmp_path / "cache", prefix_tokens, (seg,))

    # Real on-disk roundtrip: the provider returns the stored segment, and an
    # unknown key misses.
    got = provider.get(prefix_tokens)
    assert got is not None
    assert len(got) == 1
    assert got[0].feature_ids == (5,)
    assert got[0].buckets == (2,)
    assert provider.get([tokenizer.get_bos_token_id(), 99, 99]) is None
    assert len(provider) == 1


def test_context_features_shift_logits_through_real_cache(tmp_path: Path) -> None:
    """Active cached context features must shift decoder logits vs. a no-op
    (empty-feature) context segment — proven through the real on-disk cache,
    the production provider, and a real wrapper forward.
    """
    t.manual_seed(0)
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()

    prefix_tokens = _prefix_ids(tokenizer, "steer me")
    char_end = _prefix_char_end(tokenizer, prefix_tokens)

    # Two caches over the SAME prefix key: one with an active feature, one with
    # an all-masked (empty-feature) context segment. Both are valid cache rows.
    active_seg = (
        ContextFeatureSegment(
            feature_ids=(5,), buckets=(3,), char_start=0, char_end=char_end,
        ),
    )
    neutral_seg = (
        ContextFeatureSegment(
            feature_ids=(), buckets=(), char_start=0, char_end=char_end,
        ),
    )
    provider_active = _write_cache(tmp_path / "active", prefix_tokens, active_seg)
    provider_neutral = _write_cache(tmp_path / "neutral", prefix_tokens, neutral_seg)

    # Decoder input: the prefix followed by two "assistant" tokens.
    assistant_tokens = _encode(tokenizer, "ok")
    input_token_ids = [*prefix_tokens, *assistant_tokens]
    input_ids = t.tensor([input_token_ids], dtype=t.long)
    # target_mask marks the assistant span (the final 2 positions).
    target_mask = t.zeros(1, len(input_token_ids), dtype=t.long)
    target_mask[0, len(prefix_tokens):] = 1

    def _logits_for(provider: SafetensorsContextFeatureProvider) -> t.Tensor:
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SingleTargetSegment(features={}),
            context_provider=provider,
            target_conditioning_mode=TargetConditioningMode.CURRENT,
        )
        with t.inference_mode():
            with wrapper.conditioning_context(
                target_mask=target_mask,
                context_token_ids=[prefix_tokens],
            ):
                logits = wrapper(input_ids, kv_cache=None)
        return logits

    logits_active = _logits_for(provider_active)
    logits_neutral = _logits_for(provider_neutral)

    assert logits_active.shape == logits_neutral.shape
    assert t.isfinite(logits_active).all()
    # The active cached feature must measurably change the decoder logits over
    # the assistant span. Without the cache feeding the encoder cross-attention,
    # these would be bit-identical.
    assert not t.allclose(logits_active, logits_neutral, atol=1e-5), (
        "Cached context features did not change the decoder logits; the "
        "cache → provider → wrapper → encoder cross-attention path is inert."
    )


def test_no_op_target_delta_is_identity_relative_to_self(tmp_path: Path) -> None:
    """Control: forwarding the SAME cache twice is deterministic.

    This makes the logit-shift assertion above load-bearing — the difference
    there comes from the feature content, not from per-call nondeterminism.
    """
    t.manual_seed(0)
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()

    prefix_tokens = _prefix_ids(tokenizer, "steer me")
    char_end = _prefix_char_end(tokenizer, prefix_tokens)
    seg = (
        ContextFeatureSegment(
            feature_ids=(5,), buckets=(3,), char_start=0, char_end=char_end,
        ),
    )
    provider = _write_cache(tmp_path / "cache", prefix_tokens, seg)

    assistant_tokens = _encode(tokenizer, "ok")
    input_token_ids = [*prefix_tokens, *assistant_tokens]
    input_ids = t.tensor([input_token_ids], dtype=t.long)
    target_mask = t.zeros(1, len(input_token_ids), dtype=t.long)
    target_mask[0, len(prefix_tokens):] = 1

    def _logits() -> t.Tensor:
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SingleTargetSegment(features={}),
            context_provider=provider,
            target_conditioning_mode=TargetConditioningMode.CURRENT,
        )
        with t.inference_mode():
            with wrapper.conditioning_context(
                target_mask=target_mask,
                context_token_ids=[prefix_tokens],
            ):
                return wrapper(input_ids, kv_cache=None)

    assert t.allclose(_logits(), _logits(), atol=1e-6)


def test_generate_runs_through_real_cache_pipeline(tmp_path: Path) -> None:
    """Drive InferenceEngine.generate over a real on-disk cache.

    Exercises the full cache-file → SafetensorsContextFeatureProvider →
    build_engine_for_features → TargetAwareEncoderDecoderEvalWrapper → generate
    path that unit tests stub. The cache row is keyed on the exact token
    sequence ``InferenceEngine.generate`` builds for a raw (non-chat) prompt
    (``encode(prompt, prepend=bos)``) so the prefill's context-token lookup
    hits the cached row.
    """
    model = _make_encdec(device="cpu")
    tokenizer = MockTokenizer()

    query = "hi"
    # Mirror InferenceEngine.generate's non-chat tokenization exactly
    # (encode(query, prepend=bos) == [bos, *encode(query)]); this is the token
    # sequence the prefill keys the context lookup on.
    prompt_tokens = _prefix_ids(tokenizer, query)
    char_end = _prefix_char_end(tokenizer, prompt_tokens)
    seg = (
        ContextFeatureSegment(
            feature_ids=(5,), buckets=(2,), char_start=0, char_end=char_end,
        ),
    )
    provider = _write_cache(tmp_path / "cache", prompt_tokens, seg)

    engine = build_engine_for_features(
        model,
        tokenizer,
        t.device("cpu"),
        features={},
        context_provider=provider,
    )

    # Greedy generation drives the real prefill (which looks up the cache row)
    # plus the decode loop.
    result = engine.generate(query, max_new_tokens=3, temperature=0.0)
    assert isinstance(result, str)
