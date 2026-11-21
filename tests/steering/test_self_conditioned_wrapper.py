"""Behavioral tests for the self-conditioned eval-wrapper decode path.

These tests drive the *actual* wrapper forward/decode machinery in
``steering/conditional/encoder_decoder/eval_adapter.py``
(``_forward_self_cond``, ``_self_cond_prefill``, ``_self_cond_decode_step``),
not the ``SelfConditionedFeatureProvider`` in isolation. They cover:

1. Decode-step slot growth: one new encoder slot is appended per
   ``self_cond_segment_size`` (K) decoded tokens, and ``seg_map`` for each
   decode step references the latest slot.
2. Prefill seeding: prompt tokens are conditioned on real per-segment
   features via the context provider, and a first predicted slot is seeded
   from every FP row routed to the final real prompt segment.
3. The ``kv_cache is None`` guard raises.
4. ``build_engine_self_conditioned`` rejects decoder-only models.

The wrapper API changed: the constructor now requires a real
``context_provider`` (the old ``context_conditioning_mode=EMPTY`` knob is
gone). These tests therefore supply a genuine in-memory ``ContextProvider``
implementation — the same kind used by ``test_inference.py`` /
``test_build_wrapper_for_features.py`` — so prefill exercises the real
``context_provider.get(...)`` → encoder → seg_map path.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from nanochat.tokenizer import SPECIAL_TOKENS
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
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    append_packed_encoder_segment,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
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
from steering.eval.chat_generation import empty_assistant_conversation
from steering.eval.inference_engine import InferenceEngine, build_engine_self_conditioned
from steering.eval.multiseg_generator import MultisegInputs, _self_cond_prefill
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SelfConditionedTarget,
    SingleEntryContextProvider,
    SingleTargetSegment,
)
from steering.refined_types import PositiveInt, UnitInterval
from steering.segmentation import SegmentMode, token_to_char_offsets
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans
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
    _make_gpt,
)
from tests.padded_encoder_reference import pack_dense_encoder_output_reference


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA backend; flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _segment_values(packed: PackedEncoderOutput, segment: int) -> torch.Tensor:
    keep = packed.layout.valid_mask & (packed.layout.segment_indices == segment)
    return packed.values[0, keep]


def _all_segments_nonempty(packed: PackedEncoderOutput) -> bool:
    owners = packed.layout.segment_indices[packed.layout.valid_mask]
    counts = torch.bincount(owners, minlength=packed.layout.n_segments)
    return bool((counts > 0).all())


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _build_fp_encdec(
    *,
    loss_mode: FeaturePredLossMode = FeaturePredLossMode.LAST_TOKEN,
    num_buckets: int = TINY_NUM_BUCKETS,
) -> EncoderDecoderGPT:
    """Build a tiny EncoderDecoderGPT *with* a feature-prediction head whose
    vocab size matches ``MockTokenizer`` (TINY_VOCAB=128).

    The shared ``tests.encdec_test_helpers.build_model_with_feature_prediction``
    uses vocab_size=100, which is incompatible with ``MockTokenizer``'s
    ``(ord(c) % 126) + 2`` ids that can exceed 100. We therefore build a model
    with the conftest TINY_* geometry so the engine ``generate`` path (which
    tokenizes text) stays in-vocab.
    """
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
    xattn_config = CrossAttentionConfig(frequency=1)
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=loss_mode,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=num_buckets,
        pad_vocab_size_to=TINY_VOCAB,
        feature_pred_config=fp_config,
    )
    model.init_weights()
    model.eval()
    return model


def test_packed_mask_reuse_scope_clears_after_failure() -> None:
    model = _build_fp_encdec()
    with pytest.raises(RuntimeError, match="probe failure"):
        with model.reuse_single_token_packed_masks():
            with pytest.raises(RuntimeError, match="cannot be nested"):
                with model.reuse_single_token_packed_masks():
                    pass
            raise RuntimeError("probe failure")
    assert model._single_token_mask_cache_enabled is False
    assert model._single_token_mask_cache is None
    model.train()
    with pytest.raises(RuntimeError, match="evaluation mode"):
        with model.reuse_single_token_packed_masks():
            pass


class _SyntheticContextProvider:
    """Real ``ContextProvider``: one full-prefix feature segment per prompt.

    Mirrors ``tests/steering/test_inference.py``'s provider — a genuine
    implementation of the ``get(token_ids) -> tuple[ContextFeatureSegment, ...]
    | None`` protocol that conditions the whole prompt on a single feature
    (id 0, bucket 1). ``char_end`` is the real decoded character count so the
    wrapper's char-range → token alignment is well-formed.
    """

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


class _ThreeSegmentExactContextProvider:
    """Exact-token provider whose final real segment has multiple positions."""

    def __init__(self, tokenizer, token_ids: list[int]) -> None:
        self._tokenizer = tokenizer
        self._token_ids = tuple(token_ids)
        self.lookups: list[tuple[int, ...]] = []

    def get(self, token_ids):
        exact = tuple(int(token_id) for token_id in token_ids)
        self.lookups.append(exact)
        if exact != self._token_ids:
            return None
        offsets = token_to_char_offsets(self._tokenizer, list(exact))
        boundaries = (0, 3, 6, len(exact))
        return tuple(
            ContextFeatureSegment(
                feature_ids=(segment_index,),
                buckets=(segment_index % 2,),
                char_start=int(offsets[start]),
                char_end=int(offsets[end]),
            )
            for segment_index, (start, end) in enumerate(itertools.pairwise(boundaries))
        )


def _make_self_cond_config(K: int = 4) -> SelfCondConfig:
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


def _make_interpolation_config() -> SelfCondConfig:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(TINY_SEQ_LEN),
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
            percentile_table=torch.zeros((TINY_N_SAE_FEATURES, 99), dtype=torch.float64),
            rng=rng,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=TINY_N_SAE_FEATURES,
        score_policy=SelfCondScoreInterpolation(
            target_scores=torch.linspace(-0.8, 0.8, TINY_N_SAE_FEATURES),
            weight=UnitInterval.parse(0.5),
            negative_selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                min_k=0,
                max_k=4,
            ),
        ),
    )


def _make_wrapper(model, tokenizer, K=4):
    return TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SelfConditionedTarget(),
        context_provider=_SyntheticContextProvider(tokenizer),
        self_cond_config=_make_self_cond_config(K),
    )


def _make_turn_wrapper(model, tokenizer, K=4):
    prompt = tokenizer.render_for_completion(empty_assistant_conversation("q"))
    offsets = token_to_char_offsets(tokenizer, prompt)
    marker_pos = len(prompt) - 1
    segments = (
        ContextFeatureSegment(
            feature_ids=(0,),
            buckets=(1,),
            char_start=int(offsets[0]),
            char_end=int(offsets[marker_pos]),
        ),
        ContextFeatureSegment(
            feature_ids=(TINY_N_SAE_FEATURES - 1,),
            buckets=(1,),
            char_start=int(offsets[marker_pos]),
            char_end=int(offsets[marker_pos + 1]),
        ),
    )
    config = replace(
        _make_self_cond_config(K),
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SelfConditionedTarget(),
        context_provider=SingleEntryContextProvider(
            prompt,
            segments,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            token_assignments=tuple(
                sum(offset >= segment.char_start for segment in segments[1:])
                for offset in token_to_char_offsets(tokenizer, prompt)[:-1]
            ),
        ),
        self_cond_config=config,
    )
    return wrapper, prompt, segments


def _turn_wrapper_for(model, tokenizer, prompt, segments, K=8):
    """Build a turn-mode wrapper over an explicit prompt/segment pairing."""

    return TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SelfConditionedTarget(),
        context_provider=SingleEntryContextProvider(
            prompt,
            segments,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            token_assignments=tuple(
                sum(offset >= segment.char_start for segment in segments[1:])
                for offset in token_to_char_offsets(tokenizer, prompt)[:-1]
            ),
        ),
        self_cond_config=replace(
            _make_self_cond_config(K),
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        ),
    )


def _multi_turn_prompt(tokenizer):
    """A four-message completion prompt plus its logical spans and offsets."""

    prompt = tokenizer.render_for_completion(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "known"},
                {"role": "user", "content": "last"},
                {"role": "assistant", "content": ""},
            ]
        }
    )
    spans = legacy_turn_spans(prompt, ChatSpecialTokens.from_tokenizer(tokenizer))
    return prompt, spans, token_to_char_offsets(tokenizer, prompt)


def _new_kv(model, seq_len=TINY_SEQ_LEN):
    return KVCache(
        batch_size=1,
        num_heads=model.dec_config.n_kv_head,
        seq_len=seq_len,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_accepts_self_conditioned_target(self):
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok)
        assert wrapper.is_target_aware_encdec is True
        assert wrapper._self_cond is not None
        # Non-self-cond targets leave the runtime as None.
        plain = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tok,
            target=SingleTargetSegment(features={}),
            context_provider=_SyntheticContextProvider(tok),
        )
        assert plain._self_cond is None

    def test_rejects_self_cond_without_fp_head(self):
        """A SelfConditionedTarget on a model with feature_pred_head=None
        must raise (the wrapper checks model.feature_pred_head)."""
        from tests.conftest import _make_encdec

        torch.manual_seed(0)
        model = _make_encdec(device="cpu")  # no feature_pred_config => no fp head
        assert model.feature_pred_head is None
        tok = MockTokenizer()
        with pytest.raises((ValueError, RuntimeError)):
            _make_wrapper(model, tok)

    def test_rejects_self_cond_target_without_config(self):
        """SelfConditionedTarget requires a self_cond_config."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        with pytest.raises(ValueError, match="self_cond_config"):
            TargetAwareEncoderDecoderEvalWrapper(
                model=model,
                tokenizer=tok,
                target=SelfConditionedTarget(),
                context_provider=_SyntheticContextProvider(tok),
                self_cond_config=None,
            )

    def test_provider_swap_cannot_change_segment_geometry(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper = _make_wrapper(model, tokenizer)
        fixed_runtime = wrapper._self_cond
        assert fixed_runtime is not None
        turn_config = replace(
            fixed_runtime.config,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )
        turn_provider = SelfConditionedFeatureProvider(
            model,
            turn_config,
            torch.device("cpu"),
        )

        with pytest.raises(ValueError, match="incompatible component segment_mode"):
            wrapper.swap_self_cond_provider(turn_provider, turn_config)
        assert wrapper._self_cond is fixed_runtime
        assert fixed_runtime.config.segment_mode is SegmentMode.fixed_tokens


# ---------------------------------------------------------------------------
# Prefill seeding
# ---------------------------------------------------------------------------


class TestPrefillSeeding:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA bf16 model dtype")
    def test_context_encoder_uses_model_explicit_dtype_without_autocast(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Prefill must not override the model's explicit mixed precision."""
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper = _make_wrapper(model, tokenizer)
        prompt = [1, 2, 3]
        original_forward = model.encoder.forward

        def encoder_spy(*args, **kwargs):
            assert not torch.is_autocast_enabled("cpu")
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(model.encoder, "forward", encoder_spy)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            encoder_out, _seg_map = wrapper._build_conditioning(
                torch.tensor([prompt]),
                kv_cache=_new_kv(model),
            )

        assert encoder_out.values.dtype == torch.bfloat16

    def test_empty_context_segment_fails_before_encoder(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        prompt = [1, 2, 3]
        empty_context = (
            ContextFeatureSegment(
                feature_ids=(),
                buckets=(),
                char_start=0,
                char_end=3,
            ),
        )
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=SingleEntryContextProvider(prompt, empty_context),
            self_cond_config=_make_self_cond_config(),
        )

        def forbidden_encoder(*_args, **_kwargs):
            raise AssertionError("empty context must fail before encoder invocation")

        monkeypatch.setattr(model.encoder, "forward", forbidden_encoder)
        with (
            wrapper.conditioning_context(context_token_ids=[prompt]),
            pytest.raises(
                EmptySelfConditioningSegmentError,
                match="real context segment.*0 is empty",
            ),
        ):
            wrapper(torch.tensor([prompt]), kv_cache=_new_kv(model))

    def test_multisegment_prefill_uses_only_nonempty_real_and_predicted_slots(
        self,
    ) -> None:
        model = _build_fp_encdec()
        provider = SelfConditionedFeatureProvider(
            model,
            _make_self_cond_config(),
            torch.device("cpu"),
        )
        prefix = [1, 2, 3]
        inputs = MultisegInputs(
            enc_feature_ids=torch.tensor([[[0]]]),
            enc_magnitudes=torch.tensor([[[1]]]),
            enc_mask=torch.tensor([[[True]]]),
            seg_map=torch.zeros((1, len(prefix)), dtype=torch.long),
        )
        logits, state = _self_cond_prefill(
            model=model,
            provider=provider,
            encoder_inputs=inputs,
            prefix_tokens=prefix,
            kv_cache=_new_kv(model),
            device=torch.device("cpu"),
        )
        assert logits.shape == (1, model.dec_config.vocab_size)
        assert state.encoder_out.layout.n_segments == 2
        assert state.target_slot_idx == 1
        assert _all_segments_nonempty(state.encoder_out)

    def test_multisegment_prefill_rejects_empty_slot_before_encoder(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec()
        provider = SelfConditionedFeatureProvider(
            model,
            _make_self_cond_config(),
            torch.device("cpu"),
        )
        prefix = [1, 2]
        inputs = MultisegInputs(
            enc_feature_ids=torch.tensor([[[0]]]),
            enc_magnitudes=torch.tensor([[[0]]]),
            enc_mask=torch.tensor([[[False]]]),
            seg_map=torch.zeros((1, len(prefix)), dtype=torch.long),
        )

        def forbidden_encoder(*_args, **_kwargs):
            raise AssertionError("empty slot must fail before encoder invocation")

        monkeypatch.setattr(model.encoder, "forward", forbidden_encoder)
        with pytest.raises(EmptySelfConditioningSegmentError, match=r"slots \[0\]"):
            _self_cond_prefill(
                model=model,
                provider=provider,
                encoder_inputs=inputs,
                prefix_tokens=prefix,
                kv_cache=_new_kv(model),
                device=torch.device("cpu"),
            )

    def test_prefill_conditions_on_context_and_seeds_first_slot(self):
        """Self-cond prefill encodes the real context slot and seeds one more.

        Expected slot layout: 1 context slot + 1 seeded slot = 2 slots;
        ``target_slot_idx`` points at the seeded slot (1). Every slot contains
        at least one real feature.
        Output is the token-1 distribution computed by committing only the
        assistant-boundary token under the seeded slot."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok, K=4)
        T_pre = 12
        prompt = list(range(1, T_pre + 1))
        idx = torch.tensor([prompt], dtype=torch.long)
        kv = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            logits = wrapper(idx, kv_cache=kv)

        assert logits.shape == (1, 1, model.dec_config.vocab_size)
        state = wrapper._self_cond.state
        assert state is not None
        assert state.encoder_out.layout.n_segments == 2
        assert state.target_slot_idx == 1
        assert _all_segments_nonempty(state.encoder_out)
        # The buffers are reset, ready for decode.
        assert wrapper._self_cond.fp_buffer == ()
        assert wrapper._self_cond.tokens_in_seg == 0
        # KV cache advanced over the whole prompt.
        assert kv.get_pos() == T_pre

    def test_prefill_short_prompt_still_seeds_a_slot(self):
        """A prompt shorter than K still seeds exactly one predicted slot
        from every FP row routed to its final real source slot."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok, K=8)
        T_pre = 3  # < K
        prompt = list(range(1, T_pre + 1))
        idx = torch.tensor([prompt], dtype=torch.long)
        kv = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            _ = wrapper(idx, kv_cache=kv)
        state = wrapper._self_cond.state
        # Same layout as the longer prompt: 1 context + 1 seeded slot.
        assert state.encoder_out.layout.n_segments == 2
        assert state.target_slot_idx == 1
        assert _all_segments_nonempty(state.encoder_out)

    def test_prefill_seed_matches_provider_seed_from_prompt(self, monkeypatch: pytest.MonkeyPatch):
        """The slot the wrapper seeds during prefill must be the SAME slot the
        provider would produce from ``seed_from_prompt`` on the exact FP rows
        routed to the prompt's final real source slot.

        This verifies the wrapper drives the real provider seeding rather than
        fabricating a slot. We reproduce the wrapper's seeding by capturing the
        prompt fp_logits and replaying ``seed_from_prompt`` against a clean
        provider state built from the same encoder slots."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok, K=4)
        T_pre = 10
        prompt = list(range(1, T_pre + 1))
        idx = torch.tensor([prompt], dtype=torch.long)
        kv = _new_kv(model)

        # Capture the prompt fp_logits + the pre-seed encoder slots that the
        # wrapper hands to seed_from_prompt, by spying on model.forward.
        captured = {}
        orig_forward = model.forward

        def spy(idx_, targets=None, **kw):
            out = orig_forward(idx_, targets, **kw)
            # Prefill calls forward with return_fp_logits=True.
            if kw.get("return_fp_logits"):
                captured["encoder_out"] = kw["encoder_out"]
                captured["fp_logits"] = out[1]
            return out

        monkeypatch.setattr(model, "forward", spy)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            _ = wrapper(idx, kv_cache=kv)

        wrapper_state = wrapper._self_cond.state

        # Reproduce: build a clean SelfCondState from the captured pre-seed
        # slots, then call the provider's seed_from_prompt with the same
        # final-real-source fp_logits the wrapper used.
        from steering.conditional.encoder_decoder.self_conditioned import (
            SelfCondState,
        )

        assert wrapper._self_cond is not None
        provider = wrapper._self_cond.provider
        pre_seed_state = SelfCondState(
            encoder_out=captured["encoder_out"].detach(),
            target_slot_idx=captured["encoder_out"].layout.n_segments - 1,
        )
        fp_for_seed = captured["fp_logits"].detach()
        expected = provider.seed_from_prompt(pre_seed_state, fp_for_seed)

        # The newest (seeded) slot of the wrapper's state must equal the
        # newest slot the provider produces from the same inputs.
        assert torch.allclose(
            _segment_values(
                wrapper_state.encoder_out,
                wrapper_state.encoder_out.layout.n_segments - 1,
            ),
            _segment_values(
                expected.encoder_out,
                expected.encoder_out.layout.n_segments - 1,
            ),
            atol=1e-5,
        )


class TestScoreInterpolationSplitPrefill:
    def test_uses_only_last_real_segment_and_forwards_prompt_exactly_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        model = _build_fp_encdec(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            num_buckets=2,
        )
        tokenizer = MockTokenizer()
        prompt = list(range(1, 13))
        provider = _ThreeSegmentExactContextProvider(tokenizer, prompt)
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=provider,
            self_cond_config=_make_interpolation_config(),
        )
        runtime = wrapper._self_cond
        assert runtime is not None
        captured_step_rows: list[torch.Tensor] = []
        original_seed = runtime.provider.seed_from_prompt

        def seed_spy(state, fp_logits):
            captured_step_rows.append(fp_logits.clone())
            return original_seed(state, fp_logits)

        monkeypatch.setattr(runtime.provider, "seed_from_prompt", seed_spy)
        calls: list[dict[str, Any]] = []
        fabricated_source: list[torch.Tensor] = []
        original_forward = model.forward

        def forward_spy(input_ids, targets=None, **kwargs):
            calls.append(
                {
                    "tokens": input_ids.detach().clone(),
                    "seg_map": kwargs["seg_map"].detach().clone(),
                    "return_fp_logits": bool(kwargs.get("return_fp_logits")),
                }
            )
            output = original_forward(input_ids, targets, **kwargs)
            if kwargs.get("return_fp_logits"):
                logits, fp_logits = output
                seg_map = kwargs["seg_map"]
                source_slot = int(seg_map[0, -1].item())
                source_mask = seg_map[0] == source_slot
                fabricated = torch.full_like(fp_logits, 50.0, dtype=torch.bfloat16)
                source_positions = torch.nonzero(source_mask, as_tuple=False).flatten()
                fabricated[:, source_mask, :] = -10.0
                for offset, position in enumerate(source_positions):
                    fabricated[0, position, offset % TINY_N_SAE_FEATURES] = float(offset + 1)
                fabricated_source.append(fabricated[:, source_mask, :].clone())
                return logits, fabricated
            return torch.full_like(output, 7.0)

        monkeypatch.setattr(model, "forward", forward_spy)
        kv_cache = _new_kv(model)
        input_ids = torch.tensor([prompt], dtype=torch.long)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            logits = wrapper(input_ids, kv_cache=kv_cache)

        assert len(calls) == 2
        assert calls[0]["tokens"].shape == (1, len(prompt) - 1)
        assert calls[1]["tokens"].shape == (1, 1)
        forwarded = torch.cat([calls[0]["tokens"], calls[1]["tokens"]], dim=1)
        assert torch.equal(forwarded, input_ids)
        assert calls[0]["return_fp_logits"] is True
        assert calls[1]["return_fp_logits"] is False
        assert provider.lookups == [tuple(prompt)]
        assert len(captured_step_rows) == 1
        assert torch.equal(captured_step_rows[0], fabricated_source[0])
        assert captured_step_rows[0].shape[1] >= 2
        pooled = captured_step_rows[0].max(dim=1).values.float()
        assert pooled[0, 0].item() == 1.0
        assert pooled[0, 1].item() == 2.0
        assert not bool((captured_step_rows[0] == 50.0).any())
        state = runtime.state
        assert state is not None
        assert int(calls[1]["seg_map"].item()) == state.target_slot_idx
        assert torch.equal(logits, torch.full_like(logits, 7.0))
        assert kv_cache.get_pos() == len(prompt)
        assert runtime.fp_buffer == ()
        assert runtime.tokens_in_seg == 0

    def test_two_token_prompt_pools_its_one_real_source_position(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            num_buckets=2,
        )
        tokenizer = MockTokenizer()
        prompt = [1, 2]
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=_SyntheticContextProvider(tokenizer),
            self_cond_config=_make_interpolation_config(),
        )
        runtime = wrapper._self_cond
        assert runtime is not None
        captured: list[torch.Tensor] = []
        original_seed = runtime.provider.seed_from_prompt

        def seed_spy(state, fp_logits):
            captured.append(fp_logits.detach().clone())
            return original_seed(state, fp_logits)

        monkeypatch.setattr(runtime.provider, "seed_from_prompt", seed_spy)
        kv_cache = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            logits = wrapper(torch.tensor([prompt]), kv_cache=kv_cache)

        assert logits.shape == (1, 1, model.dec_config.vocab_size)
        assert len(captured) == 1
        assert captured[0].shape == (1, 1, TINY_N_SAE_FEATURES)
        assert kv_cache.get_pos() == 2

    def test_last_real_segment_longer_than_legacy_tail_uses_every_routed_row(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            num_buckets=2,
        )
        tokenizer = MockTokenizer()
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=_SyntheticContextProvider(tokenizer),
            self_cond_config=_make_interpolation_config(),
        )
        runtime = wrapper._self_cond
        assert runtime is not None
        source_length = 300
        prompt_length = source_length + 2
        input_ids = torch.zeros((1, prompt_length), dtype=torch.long)
        encoder_out = pack_dense_encoder_output_reference(
            torch.zeros((1, 2, 1, TINY_N_EMBD)),
            torch.ones((1, 2, 1), dtype=torch.bool),
        )
        seg_map = torch.tensor(
            [[0, *([1] * source_length), 1]],
            dtype=torch.long,
        )
        monkeypatch.setattr(
            wrapper,
            "_build_conditioning",
            lambda *_args, **_kwargs: (encoder_out, seg_map),
        )
        captured: list[torch.Tensor] = []

        def seed_spy(state, fp_logits):
            captured.append(fp_logits.detach().clone())
            new_segment = pack_dense_encoder_output_reference(
                torch.zeros((1, 1, 1, TINY_N_EMBD)),
                torch.ones((1, 1, 1), dtype=torch.bool),
            )
            return SelfCondState(
                encoder_out=append_packed_encoder_segment(
                    state.encoder_out,
                    new_segment,
                ),
                target_slot_idx=state.target_slot_idx + 1,
            )

        monkeypatch.setattr(runtime.provider, "seed_from_prompt", seed_spy)
        calls: list[int] = []

        def fake_forward(tokens, _targets=None, **kwargs):
            calls.append(int(tokens.shape[1]))
            logits = torch.zeros((1, tokens.shape[1], TINY_VOCAB))
            if kwargs.get("return_fp_logits"):
                fp_logits = torch.arange(
                    tokens.shape[1] * TINY_N_SAE_FEATURES,
                    dtype=torch.float32,
                ).reshape(1, tokens.shape[1], TINY_N_SAE_FEATURES)
                fp_logits[:, 0, :] = 1_000_000.0
                return logits, fp_logits
            return logits

        monkeypatch.setattr(model, "forward", fake_forward)
        logits = wrapper._self_cond_prefill(
            input_ids,
            runtime,
            object(),
        )

        assert calls == [prompt_length - 1, 1]
        assert logits.shape == (1, 1, TINY_VOCAB)
        assert len(captured) == 1
        assert captured[0].shape == (1, source_length, TINY_N_SAE_FEATURES)
        assert not bool((captured[0] == 1_000_000.0).any())

    def test_single_token_fails_before_decoder_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            num_buckets=2,
        )
        tokenizer = MockTokenizer()
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=_SyntheticContextProvider(tokenizer),
            self_cond_config=_make_interpolation_config(),
        )
        decoder_calls = 0

        def forbidden_forward(*_args, **_kwargs):
            nonlocal decoder_calls
            decoder_calls += 1
            raise AssertionError("decoder forward must not run")

        monkeypatch.setattr(model, "forward", forbidden_forward)
        build_calls = 0

        def fake_build(_input_ids, kv_cache=None):
            nonlocal build_calls
            build_calls += 1
            return (
                pack_dense_encoder_output_reference(
                    torch.zeros((1, 2, 1, TINY_N_EMBD)),
                    torch.ones((1, 2, 1), dtype=torch.bool),
                ),
                torch.tensor([[1]], dtype=torch.long),
            )

        monkeypatch.setattr(wrapper, "_build_conditioning", fake_build)
        prompt = [1]
        with pytest.raises(ValueError, match="at least two prompt tokens"):
            with wrapper.conditioning_context(context_token_ids=[prompt]):
                wrapper(torch.tensor([prompt]), kv_cache=_new_kv(model))
        assert build_calls == 1
        assert decoder_calls == 0


class _BadBosTokenizer(MockTokenizer):
    """Tokenizer whose BOS accessor violates the int contract."""

    def get_bos_token_id(self) -> Any:
        return None


class _BadSpecialTokenizer(MockTokenizer):
    """Tokenizer whose structural special-token lookup violates the int contract."""

    def encode_special(self, name: str) -> Any:
        return None if name == "<python_start>" else super().encode_special(name)


class _CollidingSpecialTokenizer(MockTokenizer):
    """Tokenizer that aliases two distinct chat control tokens onto one id."""

    def encode_special(self, name: str) -> int:
        lookup = "<assistant_start>" if name == "<python_start>" else name
        return super().encode_special(lookup)


class TestTurnModePrefill:
    # kills: hand-copying the chat special-token spellings instead of deriving
    # the structural stop set from nanochat's own SPECIAL_TOKENS list.
    def test_exposes_every_chat_control_token_as_a_generation_stop(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, _prompt, _segments = _make_turn_wrapper(model, tokenizer, K=3)

        assert SPECIAL_TOKENS[0] == "<bos>"
        assert wrapper.generation_stop_token_ids == frozenset(
            {
                tokenizer.get_bos_token_id(),
                *(tokenizer.encode_special(name) for name in SPECIAL_TOKENS[1:]),
            }
        )
        assert len(wrapper.generation_stop_token_ids) == len(SPECIAL_TOKENS)

    def test_rejects_fixed_entry_even_when_character_geometry_matches(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, segments = _make_turn_wrapper(model, tokenizer)
        wrapper.context_provider = SingleEntryContextProvider(prompt, segments)
        with pytest.raises(ValueError, match="same segment mode"):
            wrapper._prepare_turn_self_cond_prompt_context(prompt, segments)

    def test_turn_entry_requires_persisted_ownership(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, segments = _make_turn_wrapper(model, tokenizer)
        wrapper.context_provider = SingleEntryContextProvider(
            prompt,
            segments,
            segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        )
        with pytest.raises(ValueError, match="persisted token ownership"):
            wrapper._prepare_turn_self_cond_prompt_context(prompt, segments)

    # kills: dropping the int check on tokenizer.get_bos_token_id(), the int
    # check on tokenizer.encode_special(name), or the distinctness check that
    # keeps every structural stop separable.
    @pytest.mark.parametrize(
        ("tokenizer_class", "message"),
        [
            (_BadBosTokenizer, r"tokenizer\.get_bos_token_id\(\) to return an int; got None"),
            (
                _BadSpecialTokenizer,
                r"tokenizer\.encode_special\('<python_start>'\) to return an int; got None",
            ),
            (_CollidingSpecialTokenizer, "distinct chat structural token IDs"),
        ],
    )
    def test_rejects_a_tokenizer_that_breaks_the_structural_token_contract(
        self,
        tokenizer_class: type[MockTokenizer],
        message: str,
    ) -> None:
        model = _build_fp_encdec()
        with pytest.raises(ValueError, match=message):
            _make_turn_wrapper(model, tokenizer_class(), K=3)

    def test_marker_real_slot_is_removed_and_marker_fp_row_starts_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=3)
        runtime = wrapper._self_cond
        assert runtime is not None

        seed_input_slot_counts: list[int] = []
        seeded_rows: list[torch.Tensor] = []
        original_seed = runtime.provider.seed_from_prompt

        def seed_spy(state, fp_logits):
            seed_input_slot_counts.append(state.encoder_out.layout.n_segments)
            seeded_rows.append(fp_logits.detach().clone())
            return original_seed(state, fp_logits)

        monkeypatch.setattr(runtime.provider, "seed_from_prompt", seed_spy)
        calls: list[dict[str, Any]] = []
        original_forward = model.forward

        def forward_spy(input_ids, targets=None, **kwargs):
            output = original_forward(input_ids, targets, **kwargs)
            calls.append(
                {
                    "tokens": input_ids.detach().clone(),
                    "seg_map": kwargs["seg_map"].detach().clone(),
                    "return_fp_logits": bool(kwargs.get("return_fp_logits")),
                    "fp_logits": (
                        output[1].detach().clone() if kwargs.get("return_fp_logits") else None
                    ),
                }
            )
            return output

        monkeypatch.setattr(model, "forward", forward_spy)
        kv_cache = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            logits = wrapper(torch.tensor([prompt]), kv_cache=kv_cache)

        assert logits.shape == (1, 1, model.dec_config.vocab_size)
        assert len(calls) == 2
        assert torch.equal(calls[0]["tokens"], torch.tensor([prompt[:-1]]))
        assert torch.equal(calls[1]["tokens"], torch.tensor([[prompt[-1]]]))
        assert calls[0]["return_fp_logits"] is True
        assert calls[1]["return_fp_logits"] is True
        # Exactly the real user slot reaches the seed. The extracted marker
        # feature slot was discarded before packing and can never condition a
        # decoder position.
        assert seed_input_slot_counts == [1]
        assert seeded_rows[0].shape[1] == len(prompt) - 1
        assert int(calls[1]["seg_map"].item()) == 1

        snapshot = wrapper.self_cond_runtime_snapshot()
        assert snapshot.state.encoder_out.layout.n_segments == 2
        assert snapshot.state.target_slot_idx == 1
        assert snapshot.tokens_in_seg == 1
        assert len(snapshot.fp_buffer) == 1
        assert torch.equal(snapshot.fp_buffer[0], calls[1]["fp_logits"])
        assert kv_cache.get_pos() == len(prompt)

    def test_known_history_keeps_each_real_turn_and_seeds_from_final_user_chunk(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        prompt = tokenizer.render_for_completion(
            {
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "known"},
                    {"role": "user", "content": "last"},
                    {"role": "assistant", "content": ""},
                ]
            }
        )
        spans = legacy_turn_spans(
            prompt,
            ChatSpecialTokens.from_tokenizer(tokenizer),
        )
        assert [span.role for span in spans] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        offsets = token_to_char_offsets(tokenizer, prompt)
        context_segments = tuple(
            ContextFeatureSegment(
                feature_ids=(segment_index,),
                buckets=(1,),
                char_start=int(offsets[span.start + 1]),
                char_end=int(offsets[span.end + 1]),
            )
            for segment_index, span in enumerate(spans)
        )
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=model,
            tokenizer=tokenizer,
            target=SelfConditionedTarget(),
            context_provider=SingleEntryContextProvider(
                prompt, context_segments,
                segment_mode=SegmentMode.turn_delimited_fixed_tokens,
                token_assignments=tuple(
                    sum(offset >= segment.char_start for segment in context_segments[1:])
                    for offset in offsets[:-1]
                ),
            ),
            self_cond_config=replace(
                _make_self_cond_config(8),
                segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            ),
        )
        runtime = wrapper._self_cond
        assert runtime is not None
        seed_inputs: list[tuple[int, torch.Tensor]] = []
        original_seed = runtime.provider.seed_from_prompt

        def seed_spy(state, fp_logits):
            seed_inputs.append((state.encoder_out.layout.n_segments, fp_logits.detach().clone()))
            return original_seed(state, fp_logits)

        monkeypatch.setattr(runtime.provider, "seed_from_prompt", seed_spy)
        prefix_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        original_forward = model.forward

        def forward_spy(input_ids, targets=None, **kwargs):
            output = original_forward(input_ids, targets, **kwargs)
            if int(input_ids.shape[1]) > 1:
                assert kwargs.get("return_fp_logits") is True
                prefix_calls.append(
                    (
                        kwargs["seg_map"].detach().clone(),
                        output[1].detach().clone(),
                    )
                )
            return output

        monkeypatch.setattr(model, "forward", forward_spy)
        kv_cache = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            wrapper(torch.tensor([prompt]), kv_cache=kv_cache)

        assert len(seed_inputs) == 1
        seed_slot_count, seeded_rows = seed_inputs[0]
        # Three known real turns are encoded.  The terminal marker-only fourth
        # turn is dropped, then the prediction from the final user chunk appends
        # the first generated-assistant slot.
        assert seed_slot_count == 3
        assert len(prefix_calls) == 1
        prefix_seg_map, prefix_fp_logits = prefix_calls[0]
        final_user_slot = 2
        expected_source = prefix_fp_logits[:, prefix_seg_map[0] == final_user_slot, :]
        assert expected_source.shape[1] == spans[-2].end - spans[-2].start
        assert torch.equal(seeded_rows, expected_source)
        snapshot = wrapper.self_cond_runtime_snapshot()
        assert snapshot.state.encoder_out.layout.n_segments == 4
        assert snapshot.state.target_slot_idx == 3
        assert snapshot.tokens_in_seg == 1
        assert kv_cache.get_pos() == len(prompt)

    def test_k1_marker_publishes_next_state_but_keeps_outgoing_logits(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=1)
        runtime = wrapper._self_cond
        assert runtime is not None
        step_rows: list[torch.Tensor] = []
        original_step = runtime.provider.step

        def step_spy(state, fp_logits):
            step_rows.append(fp_logits.detach().clone())
            return original_step(state, fp_logits)

        monkeypatch.setattr(runtime.provider, "step", step_spy)
        calls: list[tuple[int, torch.Tensor]] = []
        original_forward = model.forward

        def forward_spy(input_ids, targets=None, **kwargs):
            output = original_forward(input_ids, targets, **kwargs)
            calls.append(
                (
                    int(kwargs["seg_map"].flatten()[0].item()),
                    output[0].detach().clone()
                    if kwargs.get("return_fp_logits")
                    else output.detach().clone(),
                )
            )
            return output

        monkeypatch.setattr(model, "forward", forward_spy)
        kv_cache = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            logits = wrapper(torch.tensor([prompt]), kv_cache=kv_cache)

        assert [slot for slot, _output in calls] == [0, 1]
        assert torch.equal(logits, calls[-1][1])
        assert len(step_rows) == 1
        assert step_rows[0].shape[1] == 1
        snapshot = wrapper.self_cond_runtime_snapshot()
        assert snapshot.state.encoder_out.layout.n_segments == 3
        assert snapshot.state.target_slot_idx == 2
        assert snapshot.fp_buffer == ()
        assert snapshot.tokens_in_seg == 0

    def test_rejects_context_without_exact_terminal_marker_segment(
        self,
    ) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, segments = _make_turn_wrapper(model, tokenizer, K=3)
        malformed = (
            segments[0],
            ContextFeatureSegment(
                feature_ids=segments[1].feature_ids,
                buckets=segments[1].buckets,
                char_start=segments[1].char_start - 1,
                char_end=segments[1].char_end,
            ),
        )

        with pytest.raises(ValueError, match="exact marker-only"):
            wrapper._prepare_turn_self_cond_prompt_context(prompt, malformed)

    def test_rejects_prefill_input_ids_that_end_in_a_different_token(self) -> None:
        # kills: forwarding input_ids' final token as the assistant marker.
        # Marker identity is proved on the *context* tokens; nothing else binds
        # the two sequences, so a caller could otherwise commit an arbitrary
        # token under the predicted first-assistant state.
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=3)
        specials = ChatSpecialTokens.from_tokenizer(tokenizer)
        mismatched = [*prompt[:-1], prompt[0]]
        assert mismatched[-1] != specials.assistant_start

        with wrapper.conditioning_context(context_token_ids=[prompt]):
            with pytest.raises(
                ValueError,
                match=(
                    r"prefill requires input_ids to end in the same "
                    r"<assistant_start> marker"
                ),
            ):
                wrapper(torch.tensor([mismatched]), kv_cache=_new_kv(model))

    # kills: dropping EITHER conjunct of the terminal-turn shape check. The
    # fully rendered conversation trips both; the other two shapes each trip
    # exactly one, so neither conjunct can be deleted silently.
    @pytest.mark.parametrize(
        "shape",
        ["fully-rendered", "marker-merged-into-tool-tail", "trailing-user-marker"],
    )
    def test_rejects_a_prompt_that_is_not_a_marker_only_terminal_turn(self, shape: str) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, _prompt, segments = _make_turn_wrapper(model, tokenizer, K=3)
        specials = ChatSpecialTokens.from_tokenizer(tokenizer)
        if shape == "fully-rendered":
            tokens, _mask = tokenizer.render_conversation(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            )
        elif shape == "marker-merged-into-tool-tail":
            # An <output_end> tail re-opens an ASSISTANT span that then merges
            # with the terminal marker's own span. The last token IS the marker,
            # so only the span-start conjunct rejects this prompt.
            tokens = [
                specials.bos,
                specials.user_start,
                7,
                specials.output_start,
                8,
                specials.output_end,
                9,
                specials.assistant_start,
            ]
        else:
            # The final span does start at the last position here, so only the
            # terminal-token conjunct rejects a prompt ending in <user_start>.
            tokens = [
                specials.bos,
                specials.user_start,
                7,
                specials.assistant_start,
                8,
                specials.user_start,
            ]

        with pytest.raises(
            ValueError,
            match="exact rendered conversation ending in one marker-only",
        ):
            wrapper._prepare_turn_self_cond_prompt_context(tokens, segments)

    # kills: reading ``real_segments[-1]`` without first proving the context
    # holds a real segment, which turns a marker-only cache into an IndexError
    # instead of the fail-loud "features must end at <assistant_start>" error.
    def test_rejects_a_context_holding_only_the_marker_segment(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, segments = _make_turn_wrapper(model, tokenizer, K=3)
        marker_start = int(segments[-1].char_start)

        with pytest.raises(
            ValueError,
            match=(
                rf"real prompt features must end exactly where <assistant_start> "
                rf"begins \({marker_start}\); got None"
            ),
        ):
            wrapper._prepare_turn_self_cond_prompt_context(prompt, (segments[-1],))

    # kills: dropping the check that real prompt features stop exactly where
    # <assistant_start> begins, which would leave a routing hole in front of
    # the marker.
    def test_rejects_real_features_that_do_not_reach_the_marker(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        prompt, spans, offsets = _multi_turn_prompt(tokenizer)
        marker_start = int(offsets[-2])
        segments = tuple(
            ContextFeatureSegment(
                feature_ids=(index,),
                buckets=(1,),
                char_start=int(offsets[span.start + 1]),
                char_end=int(offsets[span.end + 1]) - (1 if index == len(spans) - 2 else 0),
            )
            for index, span in enumerate(spans)
        )
        wrapper = _turn_wrapper_for(model, tokenizer, prompt, segments)

        with pytest.raises(
            ValueError,
            match=(
                rf"real prompt features must end exactly where <assistant_start> "
                rf"begins \({marker_start}\); got {marker_start - 1}"
            ),
        ):
            wrapper._prepare_turn_self_cond_prompt_context(prompt, segments)

    # kills: dropping the resolved-specials check, which would raise a cryptic
    # AttributeError deep inside legacy_turn_spans instead of naming the cause.
    def test_prompt_context_requires_resolved_turn_specials(self) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        _turn_wrapper, prompt, segments = _make_turn_wrapper(model, tokenizer, K=3)
        fixed_wrapper = _make_wrapper(model, tokenizer, K=3)

        with pytest.raises(
            RuntimeError,
            match="turn-mode self-conditioned prompt specials are unavailable",
        ):
            fixed_wrapper._prepare_turn_self_cond_prompt_context(prompt, segments)

    def test_rejects_structural_decode_token_before_forward(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=3)
        kv_cache = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            wrapper(torch.tensor([prompt]), kv_cache=kv_cache)

        decoder_calls = 0

        def forbidden_forward(*_args, **_kwargs):
            nonlocal decoder_calls
            decoder_calls += 1
            raise AssertionError("structural token must fail before model forward")

        monkeypatch.setattr(model, "forward", forbidden_forward)
        position = kv_cache.get_pos()
        with pytest.raises(ValueError, match="unexpected structural token"):
            wrapper(
                torch.tensor([[tokenizer.encode_special("<python_start>")]]),
                kv_cache=kv_cache,
            )
        assert decoder_calls == 0
        assert kv_cache.get_pos() == position

    def test_malformed_restored_fp_buffer_rewinds_kv_and_publishes_nothing(
        self,
    ) -> None:
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=2)
        kv_cache = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            wrapper(torch.tensor([prompt]), kv_cache=kv_cache)

        before = wrapper.self_cond_runtime_snapshot()
        assert before.tokens_in_seg == 1
        malformed_row = torch.zeros(
            (1, 1, TINY_N_SAE_FEATURES + 1),
            dtype=before.fp_buffer[0].dtype,
        )
        malformed = replace(before, fp_buffer=(malformed_row,))
        wrapper.restore_self_cond_runtime(malformed)
        start_position = int(kv_cache.get_pos())

        with pytest.raises(RuntimeError, match="Sizes of tensors must match"):
            wrapper(torch.tensor([[10]]), kv_cache=kv_cache)

        after = wrapper.self_cond_runtime_snapshot()
        assert kv_cache.get_pos() == start_position
        assert after.state is malformed.state
        assert after.fp_buffer[0] is malformed_row
        assert after.tokens_in_seg == malformed.tokens_in_seg


class TestTurnModeEngineIntegration:
    @pytest.mark.parametrize("K", [1, 2, 3])
    @pytest.mark.parametrize("ordinary_before_stop", [0, 1, 2, 3, 4])
    def test_structural_stop_boundary_matrix_never_forwards_stop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        K: int,
        ordinary_before_stop: int,
    ) -> None:
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=K)
        engine = InferenceEngine(wrapper, tokenizer, torch.device("cpu"))
        ordinary = tuple(range(10, 10 + ordinary_before_stop))
        structural_stop = tokenizer.encode_special("<assistant_end>")
        scheduled = iter((*ordinary, structural_stop))

        def scheduled_sample(_logits, _rng, _temperature, _top_k):
            return torch.tensor([[next(scheduled)]], dtype=torch.long)

        monkeypatch.setattr(
            "steering.eval.inference_engine.sample_next_token",
            scheduled_sample,
        )
        generated = list(
            engine.generate_stream(
                prompt,
                max_new_tokens=ordinary_before_stop + 1,
                temperature=0.0,
                stop_token_ids=set(),
            )
        )

        assert generated == list(ordinary)
        # Prefill consumes the terminal marker, and each ordinary token is
        # consumed only to obtain the next distribution. The structural stop
        # itself is neither yielded nor forwarded.
        assert engine._model is wrapper
        assert wrapper._self_cond is not None
        snapshot = wrapper.self_cond_runtime_snapshot()
        consumed_inputs = 1 + ordinary_before_stop
        assert snapshot.state.target_slot_idx == 1 + consumed_inputs // K
        assert snapshot.state.encoder_out.layout.n_segments == 2 + consumed_inputs // K
        assert snapshot.tokens_in_seg == consumed_inputs % K
        assert len(snapshot.fp_buffer) == snapshot.tokens_in_seg

    def test_marker_anchored_cadence_runs_end_to_end_without_replay(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tokenizer = MockTokenizer()
        wrapper, prompt, _segments = _make_turn_wrapper(model, tokenizer, K=3)
        engine = InferenceEngine(wrapper, tokenizer, torch.device("cpu"))

        scheduled = iter((10, 11, 12, 13))

        def scheduled_sample(_logits, _rng, _temperature, _top_k):
            return torch.tensor([[next(scheduled)]], dtype=torch.long)

        monkeypatch.setattr(
            "steering.eval.inference_engine.sample_next_token",
            scheduled_sample,
        )
        forwarded: list[tuple[list[int], list[int]]] = []
        original_forward = model.forward

        def forward_spy(input_ids, targets=None, **kwargs):
            forwarded.append(
                (
                    input_ids.flatten().tolist(),
                    kwargs["seg_map"].flatten().tolist(),
                )
            )
            return original_forward(input_ids, targets, **kwargs)

        monkeypatch.setattr(model, "forward", forward_spy)
        generated = list(
            engine.generate_stream(
                prompt,
                max_new_tokens=4,
                temperature=0.0,
                stop_token_ids=set(),
            )
        )

        assert generated == [10, 11, 12, 13]
        # Prefix, marker, and three consumed generated inputs: every token is
        # forwarded once. The final budgeted sample is intentionally unconsumed.
        assert [tokens for tokens, _slots in forwarded] == [
            prompt[:-1],
            [prompt[-1]],
            [10],
            [11],
            [12],
        ]
        assert [slots for _tokens, slots in forwarded[1:]] == [[1], [1], [1], [2]]
        snapshot = wrapper.self_cond_runtime_snapshot()
        assert snapshot.state.target_slot_idx == 2
        assert snapshot.state.encoder_out.layout.n_segments == 3
        assert snapshot.tokens_in_seg == 1
        assert len(snapshot.fp_buffer) == 1


# ---------------------------------------------------------------------------
# Decode-step slot growth
# ---------------------------------------------------------------------------


class TestDecodeSlotGrowth:
    def test_appends_exactly_one_slot_per_K_tokens(self):
        """Driving the wrapper for 2*K decode steps must append exactly two
        slots, one at each K-token boundary."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        K = 4
        wrapper = _make_wrapper(model, tok, K=K)
        T_pre = 8
        prompt = list(range(1, T_pre + 1))
        kv = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            _ = wrapper(torch.tensor([prompt], dtype=torch.long), kv_cache=kv)
        n_after_prefill = wrapper._self_cond.state.encoder_out.layout.n_segments

        slot_counts = []
        for _ in range(2 * K):
            tok_id = torch.randint(0, model.dec_config.vocab_size, (1, 1))
            _ = wrapper(tok_id, kv_cache=kv)
            slot_counts.append(wrapper._self_cond.state.encoder_out.layout.n_segments)

        # The slot count increments by exactly 1 at decode steps K and 2K
        # (0-indexed K-1 and 2K-1), and is otherwise flat.
        expected = [n_after_prefill] * (K - 1) + [n_after_prefill + 1] * K + [n_after_prefill + 2]
        assert slot_counts == expected
        assert wrapper._self_cond.state.encoder_out.layout.n_segments == n_after_prefill + 2

    def test_no_append_before_first_K_boundary(self):
        """Fewer than K decode steps append zero new slots."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        K = 5
        wrapper = _make_wrapper(model, tok, K=K)
        prompt = list(range(1, 9))
        kv = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            _ = wrapper(torch.tensor([prompt], dtype=torch.long), kv_cache=kv)
        n_after_prefill = wrapper._self_cond.state.encoder_out.layout.n_segments
        for _ in range(K - 1):
            wrapper(
                torch.randint(0, model.dec_config.vocab_size, (1, 1)),
                kv_cache=kv,
            )
        assert wrapper._self_cond.state.encoder_out.layout.n_segments == n_after_prefill
        # The K-th token triggers the first append.
        wrapper(
            torch.randint(0, model.dec_config.vocab_size, (1, 1)),
            kv_cache=kv,
        )
        assert wrapper._self_cond.state.encoder_out.layout.n_segments == n_after_prefill + 1

    def test_seg_map_each_decode_step_points_at_latest_slot(self, monkeypatch: pytest.MonkeyPatch):
        """``seg_map`` passed to the model at each decode step must equal the
        provider's ``target_slot_idx`` (the latest slot) at forward time.

        A boundary token is measured speculatively under the outgoing slot,
        then committed at the same KV position under the freshly appended
        incoming slot. Non-boundary tokens have one forward; boundary tokens
        therefore expose the outgoing and incoming slot in that order."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        K = 3
        wrapper = _make_wrapper(model, tok, K=K)
        prompt = list(range(1, 6))
        kv = _new_kv(model)
        with wrapper.conditioning_context(context_token_ids=[prompt]):
            _ = wrapper(torch.tensor([prompt], dtype=torch.long), kv_cache=kv)

        captured_seg_maps: list[int] = []
        orig_forward = model.forward

        def spy(idx_, targets=None, **kw):
            sm = kw.get("seg_map")
            assert sm is not None
            captured_seg_maps.append(int(sm.flatten()[0].item()))
            return orig_forward(idx_, targets, **kw)

        monkeypatch.setattr(model, "forward", spy)
        for step_index in range(2 * K):
            pre_step_idx = wrapper._self_cond.state.target_slot_idx
            calls_before = len(captured_seg_maps)
            wrapper(
                torch.randint(0, model.dec_config.vocab_size, (1, 1)),
                kv_cache=kv,
            )
            step_maps = captured_seg_maps[calls_before:]
            if (step_index + 1) % K:
                assert step_maps == [pre_step_idx]
            else:
                # The speculative FP row belongs to the outgoing state; the
                # same token is then committed under the incoming state so its
                # logits sample token K+1 without a one-token lag.
                assert step_maps == [pre_step_idx, pre_step_idx + 1]

        # Every captured seg_map indexed a valid, existing slot at the time.
        final_n_seg = wrapper._self_cond.state.encoder_out.layout.n_segments
        assert all(0 <= s < final_n_seg for s in captured_seg_maps)
        # The seg_map sequence is non-decreasing and advances exactly at the
        # K-boundaries (latest-slot routing).
        assert captured_seg_maps == sorted(captured_seg_maps)


# ---------------------------------------------------------------------------
# kv_cache guard
# ---------------------------------------------------------------------------


class TestKvCacheGuard:
    def test_self_cond_forward_requires_kv_cache(self):
        """Self-cond forward without a kv_cache raises (generation-only path)."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok)
        prompt = list(range(1, 6))
        idx = torch.tensor([prompt], dtype=torch.long)
        with pytest.raises((RuntimeError, ValueError), match="kv_cache"):
            with wrapper.conditioning_context(context_token_ids=[prompt]):
                _ = wrapper(idx, kv_cache=None)

    def test_self_cond_forward_rejects_targets(self):
        """Self-cond forward is generation-only and rejects loss targets."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok)
        prompt = list(range(1, 6))
        idx = torch.tensor([prompt], dtype=torch.long)
        kv = _new_kv(model)
        targets = torch.zeros_like(idx)
        with pytest.raises(ValueError, match="targets"):
            with wrapper.conditioning_context(context_token_ids=[prompt]):
                _ = wrapper(idx, targets=targets, kv_cache=kv)

    def test_self_cond_forward_rejects_batch_gt_one(self):
        """Self-cond generation supports B=1 only."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        wrapper = _make_wrapper(model, tok)
        prompt = list(range(1, 6))
        idx = torch.tensor([prompt, prompt], dtype=torch.long)  # B=2
        kv = _new_kv(model)
        with pytest.raises(ValueError, match="B=1"):
            with wrapper.conditioning_context(context_token_ids=[prompt, prompt]):
                _ = wrapper(idx, kv_cache=kv)


# ---------------------------------------------------------------------------
# build_engine_self_conditioned contract
# ---------------------------------------------------------------------------


class TestBuildEngineSelfConditioned:
    def test_decoder_only_model_rejected(self):
        """Decoder-only GPT cannot self-condition (no encoder)."""
        torch.manual_seed(0)
        decoder_only = _make_gpt(device="cpu")
        tok = MockTokenizer()
        with pytest.raises(ValueError, match="EncoderDecoderGPT"):
            build_engine_self_conditioned(
                decoder_only,  # ty: ignore[invalid-argument-type]
                tok,
                torch.device("cpu"),
                _make_self_cond_config(),
                context_provider=_SyntheticContextProvider(tok),
            )

    def test_engine_wraps_self_conditioned_target(self):
        """The engine wraps the model in a target-aware self-cond wrapper."""
        torch.manual_seed(0)
        model = _build_fp_encdec()
        tok = MockTokenizer()
        engine = build_engine_self_conditioned(
            model,
            tok,
            torch.device("cpu"),
            _make_self_cond_config(),
            context_provider=_SyntheticContextProvider(tok),
        )
        assert engine._model.is_target_aware_encdec is True
        assert engine._model._self_cond is not None
