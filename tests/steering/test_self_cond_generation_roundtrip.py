"""End-to-end generation tests for self-conditioned eval through the public
``InferenceEngine`` API.

Unlike ``test_self_conditioned_wrapper.py`` (which pokes the wrapper's
internal forward/decode methods directly), these tests drive the *real*
generation entry point — ``build_engine_self_conditioned(...).generate(...)``
and ``.generate_stream(...)`` — exercising the wrapper through the same path
production eval scripts use. The InferenceEngine runs prefill via
``conditioning_context`` then decodes token-by-token; for a self-cond wrapper
that routes through ``_forward_self_cond`` → prefill/decode-step, growing the
encoder state every ``self_cond_segment_size`` (K) decoded tokens.
"""

from __future__ import annotations

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfCondScoreMasking,
    SelfCondState,
)
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_SEQ_LEN,
    TINY_VOCAB,
    MockTokenizer,
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _build_fp_encdec() -> EncoderDecoderGPT:
    """Tiny EncoderDecoderGPT with an fp head and vocab=TINY_VOCAB so the
    MockTokenizer's ids are always in range for the engine generate path."""
    dec_config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.LAST_TOKEN,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        CrossAttentionConfig(frequency=1),
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=TINY_NUM_BUCKETS,
        pad_vocab_size_to=TINY_VOCAB,
        feature_pred_config=fp_config,
    )
    model.init_weights()
    model.eval()
    return model


class _SyntheticContextProvider:
    """Real ``ContextProvider`` returning one full-prefix feature segment."""

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


def _make_self_cond_config(K: int) -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(K),
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        selection=ThresholdSelectionConfig(
            selection=ThresholdSelection(
                threshold_type=ThresholdType.segment_topk,
                threshold=2,
            ),
            precomputed_thresholds=None,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=TINY_NUM_BUCKETS,
        n_sae_features=TINY_N_SAE_FEATURES,
    )


def _build_engine(K=4, model=None, tokenizer=None):
    model = model or _build_fp_encdec()
    tokenizer = tokenizer or MockTokenizer()
    return build_engine_self_conditioned(
        model,
        tokenizer,
        torch.device("cpu"),
        _make_self_cond_config(K),
        context_provider=_SyntheticContextProvider(tokenizer),
    )


def _runtime_state(engine) -> SelfCondState:
    wrapper = engine._model
    assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
    assert wrapper._self_cond is not None
    assert wrapper._self_cond.state is not None
    return wrapper._self_cond.state


def _assert_every_segment_nonempty(state: SelfCondState) -> None:
    layout = state.encoder_out.layout
    counts = torch.bincount(
        layout.segment_indices[layout.valid_mask],
        minlength=layout.n_segments,
    )
    assert bool((counts > 0).all())


class TestGenerateGreedy:
    def test_generate_returns_string_within_budget(self):
        torch.manual_seed(0)
        engine = _build_engine(K=4)
        out = engine.generate("hello world", max_new_tokens=12, temperature=0.0)
        assert isinstance(out, str)
        # Greedy can stop early on a BOS sentinel; at most max_new_tokens chars
        # (MockTokenizer is char-level, BOS/EOS decode to "").
        assert len(out) <= 12

    def test_generate_drives_self_cond_decode_growth(self):
        """A full greedy generation must grow the provider state by exactly
        one slot per K decoded tokens (the engine routes every decode step
        through ``_self_cond_decode_step``).

        We force generation to run the full budget by feeding ``max_new_tokens``
        and counting actual decoded tokens via a forward spy, since greedy may
        emit a stop token early."""
        torch.manual_seed(0)
        K = 4
        model = _build_fp_encdec()
        tok = MockTokenizer()
        engine = _build_engine(K=K, model=model, tokenizer=tok)

        prompt_tokens = tok.encode("hello", prepend=0)
        produced = list(
            engine.generate_stream(
                prompt_tokens,
                max_new_tokens=20,
                temperature=0.0,
            )
        )

        final_state = _runtime_state(engine)
        final_n_seg = final_state.encoder_out.layout.n_segments
        _assert_every_segment_nonempty(final_state)

        n_decoded = len(produced)
        # Prefill seeds S0. A replacement state is built only when another
        # distribution is requested after a completed K-token window, so a
        # final budget token never constructs an unused state.
        n_after_prefill = 2
        expected_appends = max(0, n_decoded - 1) // K
        assert final_n_seg == n_after_prefill + expected_appends, (
            f"decoded {n_decoded} tokens with K={K}: expected "
            f"{n_after_prefill + expected_appends} slots, got {final_n_seg}"
        )

    def test_generate_stream_yields_tokens(self):
        torch.manual_seed(0)
        engine = _build_engine(K=4)
        tok_ids = engine._tokenizer.encode("hi there", prepend=0)
        produced = list(
            engine.generate_stream(
                tok_ids,
                max_new_tokens=10,
                temperature=0.0,
            )
        )
        assert isinstance(produced, list)
        assert 0 <= len(produced) <= 10
        assert all(isinstance(t, int) for t in produced)

    def test_generate_stream_growth_matches_decoded_count(self):
        """Drive generate_stream and count only states that own a distribution."""
        torch.manual_seed(0)
        K = 3
        model = _build_fp_encdec()
        tok = MockTokenizer()
        engine = _build_engine(K=K, model=model, tokenizer=tok)
        tok_ids = tok.encode("abcdef", prepend=0)
        produced = list(
            engine.generate_stream(
                tok_ids,
                max_new_tokens=15,
                temperature=0.0,
            )
        )
        n_decoded = len(produced)
        final_n_seg = _runtime_state(engine).encoder_out.layout.n_segments
        assert final_n_seg == 2 + (max(0, n_decoded - 1) // K)


class TestGenerateSampled:
    def test_sampled_generation_runs(self):
        """Temperature > 0 with a seed runs the same self-cond decode path."""
        torch.manual_seed(0)
        K = 4
        model = _build_fp_encdec()
        tok = MockTokenizer()
        engine = _build_engine(K=K, model=model, tokenizer=tok)
        out = engine.generate(
            "hello",
            max_new_tokens=12,
            temperature=1.0,
            top_k=10,
            seed=123,
        )
        assert isinstance(out, str)
        # At least the prefill slots exist; growth depends on how many tokens
        # were emitted before a stop token.
        assert _runtime_state(engine).encoder_out.layout.n_segments >= 2

    def test_sampled_generation_is_seed_deterministic(self):
        """Two runs with the same seed + fresh engines (rebuilt per prompt, per
        the wrapper lifecycle contract) yield identical output."""
        K = 4
        prompt = "deterministic"

        torch.manual_seed(0)
        engine_a = _build_engine(K=K)
        out_a = engine_a.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=5,
            seed=7,
        )

        torch.manual_seed(0)
        engine_b = _build_engine(K=K)
        out_b = engine_b.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=5,
            seed=7,
        )
        assert out_a == out_b


class TestSegmentSizeBoundary:
    @pytest.mark.parametrize("K", [1, 2, 5])
    def test_growth_law_holds_across_K(self, K):
        """For several K values, no final-budget state is constructed."""
        torch.manual_seed(K)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        engine = _build_engine(K=K, model=model, tokenizer=tok)

        produced = list(
            engine.generate_stream(
                tok.encode("hello", prepend=0),
                max_new_tokens=14,
                temperature=0.0,
            )
        )

        final_n_seg = _runtime_state(engine).encoder_out.layout.n_segments
        assert final_n_seg == 2 + (max(0, len(produced) - 1) // K)


# ---------------------------------------------------------------------------
# Score-masking policy: full engine generation
# ---------------------------------------------------------------------------

_MASKING_POOL_LIMIT = 26


def _build_masking_fp_encdec() -> EncoderDecoderGPT:
    """Tiny model matching the masking contract: MAX_POOL head, two buckets."""
    dec_config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        CrossAttentionConfig(frequency=1),
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=2,
        pad_vocab_size_to=TINY_VOCAB,
        feature_pred_config=fp_config,
    )
    model.init_weights()
    model.eval()
    return model


def _masking_target() -> torch.Tensor:
    half = int(TINY_N_SAE_FEATURES) // 2
    return torch.cat(
        [
            torch.linspace(0.1, 0.9, half),
            torch.linspace(-0.9, -0.1, int(TINY_N_SAE_FEATURES) - half),
        ]
    )


def _make_masking_config(K: int) -> SelfCondConfig:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(K),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=1,
                max_k=4,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=torch.zeros((int(TINY_N_SAE_FEATURES), 99), dtype=torch.float64),
            rng=rng,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=TINY_N_SAE_FEATURES,
        score_policy=SelfCondScoreMasking(
            target_scores=_masking_target(),
            positive_pool_limit=PositiveInt.parse(_MASKING_POOL_LIMIT),
            negative_pool_limit=PositiveInt.parse(_MASKING_POOL_LIMIT),
            negative_selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=0,
                max_k=4,
            ),
        ),
    )


def _build_masking_engine(K: int, model=None, tokenizer=None):
    model = model or _build_masking_fp_encdec()
    tokenizer = tokenizer or MockTokenizer()
    return build_engine_self_conditioned(
        model,
        tokenizer,
        torch.device("cpu"),
        _make_masking_config(K),
        context_provider=_SyntheticContextProvider(tokenizer),
    )


class TestScoreMaskingGeneration:
    def test_masked_generation_grows_state_and_every_slot_is_sign_consistent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real prefill+decode run under score_masking obeys the growth law,
        and every predicted slot -- the prompt-seeded S0 included -- assigns
        only rows admitted by the stable signed DiffMean rank pools."""
        torch.manual_seed(0)
        K = 4
        model = _build_masking_fp_encdec()
        tok = MockTokenizer()
        engine = _build_masking_engine(K, model=model, tokenizer=tok)
        wrapper = engine._model
        assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
        runtime = wrapper._self_cond
        assert runtime is not None
        provider = runtime.provider
        captured: list[dict[int, int]] = []
        original_build = provider._build_segment_inputs

        def capture(segment: dict[int, int]):
            captured.append(dict(segment))
            return original_build(segment)

        monkeypatch.setattr(provider, "_build_segment_inputs", capture)

        produced = list(
            engine.generate_stream(
                tok.encode("hello there", prepend=0),
                max_new_tokens=20,
                temperature=1.0,
                top_k=10,
                seed=11,
            )
        )

        final_state = _runtime_state(engine)
        assert final_state.encoder_out.layout.n_segments == 2 + (
            max(0, len(produced) - 1) // K
        )
        _assert_every_segment_nonempty(final_state)

        target = _masking_target()
        half = int(TINY_N_SAE_FEATURES) // 2
        eligible_positive = set(range(half - _MASKING_POOL_LIMIT, half))
        eligible_negative = set(range(half, half + _MASKING_POOL_LIMIT))
        predicted_slots = [segment for segment in captured if segment]
        assert predicted_slots, "generation must have built at least the seeded S0 slot"
        for segment in predicted_slots:
            for feature_id, bucket in segment.items():
                value = float(target[feature_id])
                if bucket == 0:
                    assert value > 0.0 and feature_id in eligible_positive
                else:
                    assert bucket == 1
                    assert value < 0.0 and feature_id in eligible_negative

    def test_masked_sampled_generation_is_seed_deterministic(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        K = 4
        prompt = "deterministic"

        def run_once() -> tuple[str, list[dict[int, int]], torch.Tensor]:
            torch.manual_seed(0)
            engine = _build_masking_engine(K)
            wrapper = engine._model
            assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
            runtime = wrapper._self_cond
            assert runtime is not None
            provider = runtime.provider
            captured: list[dict[int, int]] = []
            original_build = provider._build_segment_inputs

            def capture(segment: dict[int, int]):
                captured.append(dict(segment))
                return original_build(segment)

            monkeypatch.setattr(provider, "_build_segment_inputs", capture)
            output = engine.generate(
                prompt,
                max_new_tokens=10,
                temperature=0.8,
                top_k=5,
                seed=7,
            )
            selection = provider.cfg.selection
            assert isinstance(selection, SamplingBundle)
            return output, captured, selection.rng.get_state().clone()

        out_a, segments_a, rng_a = run_once()
        out_b, segments_b, rng_b = run_once()
        assert out_a == out_b
        assert segments_a == segments_b
        assert torch.equal(rng_a, rng_b)
