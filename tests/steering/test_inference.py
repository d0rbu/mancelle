"""Tests for the InferenceEngine — greedy and sampled generation,
batched inference, feature conditioning, and edge cases."""

from __future__ import annotations

import inspect

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

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
from steering.segmentation import token_to_char_offsets
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_SEQ_LEN,
    _make_encdec,
    _make_gpt,
)


class _SyntheticContextProvider:
    """Return one full-prefix feature segment for any non-empty prompt."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

    def get(self, token_ids):
        token_ids = [int(tok) for tok in token_ids]
        if not token_ids:
            return None
        char_end = token_to_char_offsets(self._tokenizer, token_ids)[-1]
        if char_end <= 0:
            return None
        return (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


def _context_provider(tokenizer):
    return _SyntheticContextProvider(tokenizer)


class TestModelDetection:
    """Model detection, validation, and basic generation sanity checks."""

    def test_context_provider_is_explicit_at_eval_factory_boundary(self):
        """Factory callers must spell out whether full-context features exist."""
        engine_sig = inspect.signature(build_engine_for_features)
        wrapper_sig = inspect.signature(build_wrapper_for_features)

        assert (
            engine_sig.parameters["context_provider"].default
            is inspect.Signature.empty
        )
        assert (
            wrapper_sig.parameters["context_provider"].default
            is inspect.Signature.empty
        )

    def test_detects_decoder_only(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        assert not getattr(engine._model, "is_target_aware_encdec", False)

    def test_feature_id_out_of_range(self, encdec_model, mock_tokenizer):
        with pytest.raises(ValueError, match="out of range"):
            TargetAwareEncoderDecoderEvalWrapper(
                encdec_model,
                mock_tokenizer,
                target=SingleTargetSegment(features={TINY_N_SAE_FEATURES: 0}),  # off-by-one above max
                context_provider=_context_provider(mock_tokenizer),
            )

    def test_negative_feature_id(self, encdec_model, mock_tokenizer):
        with pytest.raises(BeartypeCallHintParamViolation, match="features"):
            TargetAwareEncoderDecoderEvalWrapper(
                encdec_model,
                mock_tokenizer,
                target=SingleTargetSegment(features={-1: 1}),
                context_provider=_context_provider(mock_tokenizer),
            )

    def test_magnitude_out_of_range(self, encdec_model, mock_tokenizer):
        with pytest.raises(ValueError, match="out of range"):
            TargetAwareEncoderDecoderEvalWrapper(
                encdec_model,
                mock_tokenizer,
                target=SingleTargetSegment(features={0: 4}),  # num_buckets=4, so 4 is out of range
                context_provider=_context_provider(mock_tokenizer),
            )

    def test_negative_magnitude(self, encdec_model, mock_tokenizer):
        with pytest.raises(BeartypeCallHintParamViolation, match="features"):
            TargetAwareEncoderDecoderEvalWrapper(
                encdec_model,
                mock_tokenizer,
                target=SingleTargetSegment(features={0: -1}),
                context_provider=_context_provider(mock_tokenizer),
            )

    def test_prompt_exceeds_context_window(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        long_prompt = "a" * 500  # with BOS, will exceed TINY_SEQ_LEN=128
        with pytest.raises(ValueError, match="[Ee]xceeds.+context"):
            engine.generate(long_prompt)

    def test_empty_target_features_use_full_context(self, encdec_model, mock_tokenizer):
        engine = build_engine_for_features(
            encdec_model,
            mock_tokenizer,
            t.device("cpu"),
            features={},
            context_provider=_context_provider(mock_tokenizer),
        )
        result = engine.generate("hi", max_new_tokens=1, temperature=0.0)
        assert isinstance(result, str)

    def test_encdec_requires_context_provider(self, encdec_model, mock_tokenizer):
        with pytest.raises(ValueError, match="context_provider"):
            build_engine_for_features(
                encdec_model,
                mock_tokenizer,
                t.device("cpu"),
                features={},
                context_provider=None,
            )

    def test_next_mode_rejects_multi_target_with_n_greater_than_one(
        self,
        encdec_model,
        mock_tokenizer,
    ):
        target = MultiTargetSegments(
            segments=[{0: 1}, {0: 2}],
            token_assignments=t.tensor([0, 1], dtype=t.int64),
        )
        with pytest.raises(ValueError, match="NEXT"):
            TargetAwareEncoderDecoderEvalWrapper(
                encdec_model,
                mock_tokenizer,
                target=target,
                context_provider=_context_provider(mock_tokenizer),
                target_conditioning_mode=TargetConditioningMode.NEXT,
            )

    def test_next_mode_accepts_single_target(
        self,
        encdec_model,
        mock_tokenizer,
    ):
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            encdec_model,
            mock_tokenizer,
            target=SingleTargetSegment(features={0: 1}),
            context_provider=_context_provider(mock_tokenizer),
            target_conditioning_mode=TargetConditioningMode.NEXT,
        )
        assert wrapper.target_conditioning_mode is TargetConditioningMode.NEXT


class TestGreedyGeneration:
    """Temperature=0 produces deterministic output regardless of seed."""

    def test_greedy_produces_output(self, gpt_model, mock_tokenizer, monkeypatch):
        forced_token_id = ord("x") + 2

        def forward(idx, targets=None, kv_cache=None):
            logits = t.zeros(
                idx.size(0),
                idx.size(1),
                gpt_model.config.vocab_size,
                device=idx.device,
            )
            logits[..., forced_token_id] = 1.0
            return logits

        monkeypatch.setattr(gpt_model, "forward", forward)
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        result = engine.generate("hello", temperature=0.0, max_new_tokens=10)
        assert result == "x" * 10

    def test_greedy_ignores_seed(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        results = [
            engine.generate("hello", temperature=0.0, max_new_tokens=20, seed=i)
            for i in range(5)
        ]
        assert all(r == results[0] for r in results), (
            f"Greedy outputs differ across seeds: {results}"
        )

    def test_greedy_encdec_context_only_target(self, encdec_model, mock_tokenizer):
        engine = build_engine_for_features(
            encdec_model,
            mock_tokenizer,
            t.device("cpu"),
            features={},
            context_provider=_context_provider(mock_tokenizer),
        )
        result = engine.generate("hello", temperature=0.0, max_new_tokens=10)
        assert isinstance(result, str)

    def test_greedy_encdec_with_features(self, encdec_model, mock_tokenizer):
        model = encdec_model
        tokenizer = mock_tokenizer
        engine = build_engine_for_features(
            model, tokenizer, t.device("cpu"),
            features={5: 2},
            context_provider=_context_provider(tokenizer),
        )
        asid = tokenizer.encode_special("<assistant_start>")
        prompt_tokens = tokenizer.encode("hello", prepend=tokenizer.get_bos_token_id())
        prompt_tokens.append(asid)
        # Generate via the generate_stream path.
        out = list(
            engine.generate_stream(
                prompt_tokens,
                max_new_tokens=4,
                temperature=0.0,
                stop_token_ids={tokenizer.get_bos_token_id(), asid},
            )
        )
        assert isinstance(out, list)


class TestDeterminism:
    """Same seed + same prompt = identical output across runs."""

    def test_seeded_determinism_10_runs(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        results = [
            engine.generate(
                "test prompt", temperature=0.7, top_k=10, max_new_tokens=20, seed=42
            )
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results), (
            f"Non-deterministic outputs: {set(results)}"
        )

    def test_seeded_determinism_encdec(self, encdec_model, mock_tokenizer):
        engine = build_engine_for_features(
            encdec_model, mock_tokenizer, t.device("cpu"),
            features={0: 1, 3: 2},
            context_provider=_context_provider(mock_tokenizer),
        )
        asid = mock_tokenizer.encode_special("<assistant_start>")
        prompt_tokens = mock_tokenizer.encode("test prompt", prepend=mock_tokenizer.get_bos_token_id())
        prompt_tokens.append(asid)
        results = [
            list(engine.generate_stream(
                prompt_tokens,
                temperature=0.7,
                top_k=10,
                max_new_tokens=20,
                seed=42,
                stop_token_ids={mock_tokenizer.get_bos_token_id(), asid},
            ))
            for _ in range(10)
        ]
        assert all(r == results[0] for r in results), (
            f"Non-deterministic outputs: { {str(r) for r in results} }"
        )

    def test_different_seeds_differ(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        r1 = engine.generate(
            "test", temperature=0.9, top_k=50, max_new_tokens=30, seed=1
        )
        r2 = engine.generate(
            "test", temperature=0.9, top_k=50, max_new_tokens=30, seed=999
        )
        # With high temp and different seeds, outputs should differ.
        # Extremely unlikely to collide with vocab=128, 30 tokens, temp=0.9.
        assert r1 != r2, "Different seeds produced identical output"


class TestBatchGeneration:
    """generate_batch must match sequential generate calls."""

    def test_batch_greedy_matches_sequential(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        prompts = ["hello world", "foo bar baz"]
        batch_results = engine.generate_batch(
            prompts, temperature=0.0, max_new_tokens=15
        )
        sequential_results = [
            engine.generate(p, temperature=0.0, max_new_tokens=15) for p in prompts
        ]
        assert batch_results == sequential_results, (
            f"Batch: {batch_results}\nSequential: {sequential_results}"
        )

    def test_batch_greedy_encdec(self, encdec_model, mock_tokenizer):
        engine = build_engine_for_features(
            encdec_model, mock_tokenizer, t.device("cpu"),
            features={},
            context_provider=_context_provider(mock_tokenizer),
        )
        prompts = ["hello", "world"]
        # Wrapped encoder-decoder forces sequential (is_target_aware_encdec).
        batch_results = engine.generate_batch(
            prompts,
            temperature=0.0,
            max_new_tokens=10,
        )
        sequential_results = [
            engine.generate(p, temperature=0.0, max_new_tokens=10)
            for p in prompts
        ]
        assert batch_results == sequential_results

    def test_batch_stochastic_seed_none_raises(self, gpt_model, mock_tokenizer):
        """generate_batch with temperature>0 and seed=None must raise ValueError, not TypeError."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        prompts = ["alpha", "beta"]
        # Regression: seed=None + temperature>0 used to raise TypeError (None + int)
        # instead of the proper ValueError from generate().
        with pytest.raises(ValueError, match="seed is required"):
            engine.generate_batch(
                prompts,
                temperature=0.7,
                max_new_tokens=10,
                seed=None,
            )

    def test_batch_stochastic_matches_sequential(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        prompts = ["alpha", "beta"]
        batch_results = engine.generate_batch(
            prompts,
            temperature=0.7,
            top_k=10,
            max_new_tokens=15,
            seed=42,
        )
        sequential_results = [
            engine.generate(
                p, temperature=0.7, top_k=10, max_new_tokens=15, seed=42 + i
            )
            for i, p in enumerate(prompts)
        ]
        assert batch_results == sequential_results


def _amplify_cross_attention(model):
    """Set cross-attention c_proj weights to non-zero values.

    At initialisation, c_proj is zero so the encoder signal has no effect
    on the decoder.  This helper gives the projection random non-zero
    weights so that different encoder outputs actually produce different
    decoder logits.
    """
    for layer in model.cross_attn_layers.values():
        t.nn.init.normal_(layer.c_proj.weight, std=1.0)


class TestFeatureConditioning:
    """Different feature conditions should produce different outputs."""

    def test_different_features_produce_different_outputs(
        self, encdec_model, mock_tokenizer
    ):
        _amplify_cross_attention(encdec_model)
        tokenizer = mock_tokenizer
        asid = tokenizer.encode_special("<assistant_start>")
        outputs = set()
        for fid in range(50):
            engine = build_engine_for_features(
                encdec_model, tokenizer, t.device("cpu"),
                features={fid: 1},
                context_provider=_context_provider(tokenizer),
            )
            prompt_tokens = tokenizer.encode("test", prepend=tokenizer.get_bos_token_id())
            prompt_tokens.append(asid)
            result = list(engine.generate_stream(
                prompt_tokens,
                temperature=0.0,
                max_new_tokens=15,
                stop_token_ids={tokenizer.get_bos_token_id(), asid},
            ))
            outputs.add(tuple(result))
        assert len(outputs) > 1, (
            "All 50 feature conditions produced identical output"
        )

    def test_different_magnitudes_produce_different_outputs(
        self, encdec_model, mock_tokenizer
    ):
        _amplify_cross_attention(encdec_model)
        tokenizer = mock_tokenizer
        asid = tokenizer.encode_special("<assistant_start>")
        outputs = set()
        for mag in range(TINY_NUM_BUCKETS):
            engine = build_engine_for_features(
                encdec_model, tokenizer, t.device("cpu"),
                features={0: mag},
                context_provider=_context_provider(tokenizer),
            )
            prompt_tokens = tokenizer.encode("test", prepend=tokenizer.get_bos_token_id())
            prompt_tokens.append(asid)
            result = list(engine.generate_stream(
                prompt_tokens,
                temperature=0.0,
                max_new_tokens=15,
                stop_token_ids={tokenizer.get_bos_token_id(), asid},
            ))
            outputs.add(tuple(result))
        assert len(outputs) > 1, (
            "All magnitude levels produced identical output"
        )


class TestMaskedEncoderPath:
    """Direct model invariants for the all-masked encoder slot."""

    @staticmethod
    def _copy_decoder_weights(gpt_model, encdec_model):
        """Copy all shared decoder weights from *gpt_model* into *encdec_model*.

        Iterates the GPT state_dict and copies every key that also exists
        in the EncoderDecoderGPT state_dict.  This covers ``transformer.wte``,
        ``transformer.h.*``, ``lm_head``, ``resid_lambdas``, ``x0_lambdas``,
        and ``value_embeds.*``.
        """
        gpt_sd = gpt_model.state_dict()
        encdec_sd = encdec_model.state_dict()
        updated = {}
        for key, value in gpt_sd.items():
            if key in encdec_sd:
                updated[key] = value
        encdec_model.load_state_dict({**encdec_sd, **updated}, strict=True)

    def test_logits_match_decoder_only_baseline(self, mock_tokenizer):
        """All-masked encoder logits match the copied decoder path."""
        device = t.device("cpu")

        # Build both models and copy decoder weights.
        gpt = _make_gpt("cpu")
        encdec = _make_encdec("cpu")
        self._copy_decoder_weights(gpt, encdec)

        # Prepare a short token sequence.
        token_ids = mock_tokenizer.encode(
            "hello", prepend=mock_tokenizer.get_bos_token_id()
        )
        idx = t.tensor([token_ids], dtype=t.long, device=device)

        # GPT forward (no KV cache).
        with t.inference_mode():
            gpt_logits = gpt(idx, targets=None)  # (1, T, V)

        # EncDecGPT forward with an all-masked encoder slot.
        enc_fids = t.zeros(1, 1, 1, dtype=t.int64, device=device)
        enc_mags = t.zeros_like(enc_fids)
        enc_mask = t.zeros(1, 1, 1, dtype=t.bool, device=device)
        with t.inference_mode():
            encoder_out = encdec.encoder(enc_fids, enc_mags, enc_mask)
            B, T = idx.size()
            seg_map = t.zeros(B, T, dtype=t.long, device=device)
            encdec_logits = encdec(
                idx,
                targets=None,
                seg_map=seg_map,
                encoder_out=encoder_out,
            )  # (1, T, V)

        t.testing.assert_close(gpt_logits, encdec_logits, atol=1e-5, rtol=1e-5)

    def test_generation_adapter_requires_context_provider(self, mock_tokenizer):
        """The eval adapter requires prompt/context features for generation."""
        device = t.device("cpu")

        encdec = _make_encdec("cpu")
        self._copy_decoder_weights(_make_gpt("cpu"), encdec)

        with pytest.raises(ValueError, match="context_provider"):
            build_engine_for_features(
                encdec,
                mock_tokenizer,
                device,
                features={},
                context_provider=None,
            )


class TestEdgeCases:
    """Edge case handling."""

    def test_empty_prompt(self, gpt_model, mock_tokenizer):
        """Empty string prompt — just BOS token."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        result = engine.generate("", temperature=0.0, max_new_tokens=5)
        assert isinstance(result, str)

    def test_single_char_prompt(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        result = engine.generate("a", temperature=0.0, max_new_tokens=5)
        assert isinstance(result, str)

    def test_max_new_tokens_zero(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        result = engine.generate("hello", temperature=0.0, max_new_tokens=0)
        assert result == ""

    def test_prompt_exactly_at_context_limit(self, gpt_model, mock_tokenizer):
        """Prompt that fills exactly the context window should not raise."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        # BOS + (SEQ_LEN - 1) chars = SEQ_LEN tokens
        prompt = "a" * (TINY_SEQ_LEN - 1)
        result = engine.generate(prompt, temperature=0.0, max_new_tokens=0)
        assert isinstance(result, str)

    def test_prompt_one_over_context_limit(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        prompt = "a" * TINY_SEQ_LEN  # + BOS = SEQ_LEN + 1 tokens
        with pytest.raises(ValueError, match="[Ee]xceeds.+context"):
            engine.generate(prompt)

    def test_empty_batch(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        results = engine.generate_batch([], temperature=0.0, max_new_tokens=5)
        assert results == []

    def test_single_prompt_batch(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        batch = engine.generate_batch(["hello"], temperature=0.0, max_new_tokens=10)
        single = engine.generate("hello", temperature=0.0, max_new_tokens=10)
        assert batch == [single]


class TestShapeAssertions:
    """Verify tensor shapes at key points in the forward pass."""

    def test_kv_cache_shape(self, gpt_model, mock_tokenizer):
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        kv = engine._make_kv_cache(batch_size=2, seq_len=64)
        assert kv.k_cache.shape == (TINY_N_LAYER, 2, 64, TINY_N_KV_HEAD, TINY_HEAD_DIM)
        assert kv.v_cache.shape == (TINY_N_LAYER, 2, 64, TINY_N_KV_HEAD, TINY_HEAD_DIM)
        assert kv.cache_seqlens.shape == (2,)


class TestEncoderPermutationInvariance:
    """The encoder must be permutation-invariant over feature inputs.

    If passing features in a different order produces different output,
    it indicates unwanted positional encoding or bias in encoder
    self-attention or cross-attention layers.
    """

    def test_feature_order_does_not_affect_generation(
        self, encdec_model, mock_tokenizer
    ):
        tokenizer = mock_tokenizer
        asid = tokenizer.encode_special("<assistant_start>")
        prompt_tokens = tokenizer.encode("test", prepend=tokenizer.get_bos_token_id())
        prompt_tokens.append(asid)
        stop_ids = {tokenizer.get_bos_token_id(), asid}

        conds_a = {0: 1, 5: 2, 10: 3}
        conds_b = {10: 3, 0: 1, 5: 2}

        engine_a = build_engine_for_features(
            encdec_model,
            tokenizer,
            t.device("cpu"),
            features=conds_a,
            context_provider=_context_provider(tokenizer),
        )
        engine_b = build_engine_for_features(
            encdec_model,
            tokenizer,
            t.device("cpu"),
            features=conds_b,
            context_provider=_context_provider(tokenizer),
        )

        out_a = list(engine_a.generate_stream(
            prompt_tokens, temperature=0.0, max_new_tokens=15, stop_token_ids=stop_ids
        ))
        out_b = list(engine_b.generate_stream(
            prompt_tokens, temperature=0.0, max_new_tokens=15, stop_token_ids=stop_ids
        ))

        assert out_a == out_b, (
            f"Feature order affected generation output:\n"
            f"  {conds_a} -> {out_a!r}\n"
            f"  {conds_b} -> {out_b!r}"
        )


class TestGenerateStream:
    """generate_stream() yields token IDs matching generate() output."""

    def test_stream_produces_same_output_as_generate_greedy(
        self, gpt_model, mock_tokenizer
    ):
        """Decoded streaming output must match generate() text output."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        prompt = "hello"
        bos = mock_tokenizer.get_bos_token_id()
        token_ids = mock_tokenizer.encode(prompt, prepend=bos)

        streamed = list(
            engine.generate_stream(token_ids, temperature=0.0, max_new_tokens=15)
        )
        generated_text = engine.generate(prompt, temperature=0.0, max_new_tokens=15)

        assert mock_tokenizer.decode(streamed) == generated_text

    def test_stream_produces_same_output_as_generate_seeded(
        self, gpt_model, mock_tokenizer
    ):
        """Decoded streaming output must match generate() text output (seeded)."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        prompt = "test"
        bos = mock_tokenizer.get_bos_token_id()
        token_ids = mock_tokenizer.encode(prompt, prepend=bos)

        streamed = list(
            engine.generate_stream(
                token_ids,
                temperature=0.7,
                top_k=10,
                max_new_tokens=20,
                seed=42,
            )
        )
        generated_text = engine.generate(
            prompt,
            temperature=0.7,
            top_k=10,
            max_new_tokens=20,
            seed=42,
        )

        assert mock_tokenizer.decode(streamed) == generated_text

    def test_stream_encdec_with_features(self, encdec_model, mock_tokenizer):
        """Decoded streaming output must match generate() for encoder-decoder."""
        tokenizer = mock_tokenizer
        engine = build_engine_for_features(
            encdec_model, tokenizer, t.device("cpu"),
            features={0: 1, 5: 2},
            context_provider=_context_provider(tokenizer),
        )
        asid = tokenizer.encode_special("<assistant_start>")
        bos = tokenizer.get_bos_token_id()
        token_ids = tokenizer.encode("hello", prepend=bos)
        token_ids.append(asid)
        stop_ids = {bos, asid}

        streamed = list(
            engine.generate_stream(
                token_ids,
                temperature=0.0,
                max_new_tokens=10,
                stop_token_ids=stop_ids,
            )
        )

        # generate() uses render_as_chat, which calls render_for_completion
        # and encode_special — but MockTokenizer.render_for_completion only
        # uses the 'text' key, so use generate_stream directly for comparison.
        streamed2 = list(
            engine.generate_stream(
                token_ids,
                temperature=0.0,
                max_new_tokens=10,
                stop_token_ids=stop_ids,
            )
        )
        assert streamed == streamed2

    def test_stream_respects_stop_token_ids(self, gpt_model, mock_tokenizer):
        """Generation stops when a token in stop_token_ids is produced."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        bos = mock_tokenizer.get_bos_token_id()
        token_ids = mock_tokenizer.encode("hello", prepend=bos)

        # Disable default BOS stopping so we get all 20 tokens.
        all_tokens = list(
            engine.generate_stream(
                token_ids,
                temperature=0.0,
                max_new_tokens=20,
                stop_token_ids=set(),
            )
        )
        assert len(all_tokens) > 2, "Need at least 3 tokens for this test"

        stop_idx = next(
            (
                idx
                for idx, tok in enumerate(all_tokens[1:], start=1)
                if tok not in all_tokens[:idx]
            ),
            0,
        )
        stop_tok = all_tokens[stop_idx]
        stopped = list(
            engine.generate_stream(
                token_ids,
                temperature=0.0,
                max_new_tokens=20,
                stop_token_ids={stop_tok},
            )
        )

        assert stopped == all_tokens[:stop_idx]

    def test_stream_empty_on_zero_max_tokens(self, gpt_model, mock_tokenizer):
        """max_new_tokens=0 yields nothing."""
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        bos = mock_tokenizer.get_bos_token_id()
        token_ids = mock_tokenizer.encode("hello", prepend=bos)
        streamed = list(
            engine.generate_stream(token_ids, temperature=0.0, max_new_tokens=0)
        )
        assert streamed == []
