"""Public runtime access for the self-conditioned eval wrapper."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    SelfCondRuntimeSnapshot,
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
    SelfCondState,
)
from steering.conditional.feature_mask import FeatureMask
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
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    SelfConditionedTarget,
    SingleEntryContextProvider,
    SingleTargetSegment,
)
from steering.refined_types import PositiveInt, UnitInterval
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    build_tiny_model,
)
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

_CPU = t.device("cpu")
_K = 8
_PROMPT = [2, 3, 4, 5]
_CONTEXT = (
    ContextFeatureSegment(
        feature_ids=(0,),
        buckets=(0,),
        char_start=0,
        char_end=len(_PROMPT),
    ),
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _positive_selection() -> SamplingSelection:
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )


def _negative_selection() -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=0,
        max_k=3,
    )


def _config(
    *,
    rng: t.Generator,
    percentile_table: t.Tensor,
    segment_size: int = _K,
    feature_mask: FeatureMask | None = None,
) -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(segment_size),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            selection=_positive_selection(),
            percentile_table=percentile_table,
            rng=rng,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=N_SAE_FEATURES,
        feature_mask=feature_mask,
    )


def _build_harness(
    config: SelfCondConfig | None = None,
) -> tuple[
    Any,
    MockTokenizer,
    SelfCondConfig,
    TargetAwareEncoderDecoderEvalWrapper,
    KVCache,
]:
    t.manual_seed(0)
    model = build_tiny_model(
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )
    model.eval()
    tokenizer = MockTokenizer()
    rng = t.Generator(device="cpu")
    rng.manual_seed(123)
    percentile_table = t.zeros((N_SAE_FEATURES, 99), dtype=t.float64)
    if config is None:
        config = _config(rng=rng, percentile_table=percentile_table)
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SelfConditionedTarget(),
        context_provider=SingleEntryContextProvider(_PROMPT, _CONTEXT),
        self_cond_config=config,
    )
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.dec_config.n_kv_head,
        seq_len=64,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer,
        device=_CPU,
        dtype=t.float32,
    )
    return model, tokenizer, config, wrapper, kv_cache


def _prefill(
    wrapper: TargetAwareEncoderDecoderEvalWrapper,
    kv_cache: KVCache,
) -> t.Tensor:
    prompt = t.tensor([_PROMPT], dtype=t.long)
    with wrapper.conditioning_context(context_token_ids=[_PROMPT]):
        return wrapper(prompt, kv_cache=kv_cache)


def _decode(
    wrapper: TargetAwareEncoderDecoderEvalWrapper,
    kv_cache: KVCache,
    token: int,
) -> t.Tensor:
    return wrapper(t.tensor([[token]], dtype=t.long), kv_cache=kv_cache)


@pytest.fixture
def prefilled_runtime_harness() -> tuple[
    Any,
    MockTokenizer,
    SelfCondConfig,
    TargetAwareEncoderDecoderEvalWrapper,
    KVCache,
]:
    harness = _build_harness()
    _prefill(harness[3], harness[4])
    return harness


def test_non_self_cond_wrapper_rejects_all_runtime_methods() -> None:
    model, tokenizer, config, self_cond_wrapper, _kv_cache = _build_harness()
    runtime = self_cond_wrapper._self_cond
    assert runtime is not None
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SingleTargetSegment(features={}),
        context_provider=SingleEntryContextProvider(_PROMPT, _CONTEXT),
    )

    with pytest.raises(RuntimeError, match="self-conditioned mode"):
        wrapper.self_cond_runtime_snapshot()
    with pytest.raises(RuntimeError, match="self-conditioned mode"):
        wrapper.restore_self_cond_runtime(None)
    with pytest.raises(RuntimeError, match="self-conditioned mode"):
        wrapper.swap_self_cond_provider(runtime.provider, config)


def test_snapshot_and_restore_reject_before_prefill() -> None:
    _model, _tokenizer, _config, wrapper, _kv_cache = _build_harness()

    with pytest.raises(RuntimeError, match="before prefill"):
        wrapper.self_cond_runtime_snapshot()
    with pytest.raises(RuntimeError, match="before prefill"):
        wrapper.restore_self_cond_runtime(None)


def test_snapshot_after_real_prefill_is_at_boundary() -> None:
    _model, _tokenizer, _config, wrapper, kv_cache = _build_harness()

    _prefill(wrapper, kv_cache)
    snapshot = wrapper.self_cond_runtime_snapshot()
    runtime = wrapper._self_cond
    assert runtime is not None

    assert isinstance(snapshot, SelfCondRuntimeSnapshot)
    assert snapshot.state is runtime.state
    assert snapshot.fp_buffer is runtime.fp_buffer
    assert snapshot.state.target_slot_idx == len(_CONTEXT)
    assert snapshot.fp_buffer == ()
    assert snapshot.tokens_in_seg == 0
    assert kv_cache.get_pos() == len(_PROMPT)


def test_snapshot_mid_window_tracks_two_decoded_tokens() -> None:
    _model, _tokenizer, _config, wrapper, kv_cache = _build_harness()
    _prefill(wrapper, kv_cache)

    _decode(wrapper, kv_cache, 6)
    _decode(wrapper, kv_cache, 7)
    snapshot = wrapper.self_cond_runtime_snapshot()

    assert snapshot.tokens_in_seg == 2
    assert len(snapshot.fp_buffer) == 2


def test_snapshot_zero_encoder_tensors_are_never_mutated_in_place() -> None:
    _model, _tokenizer, _config, wrapper, kv_cache = _build_harness()
    _prefill(wrapper, kv_cache)
    snapshot = wrapper.self_cond_runtime_snapshot()
    encoder_out = snapshot.state.encoder_out
    tensors = (
        encoder_out.values,
        encoder_out.layout.flat_indices,
        encoder_out.layout.valid_mask,
        encoder_out.layout.batch_indices,
        encoder_out.layout.segment_indices,
    )
    pinned = [(tensor.data_ptr(), tensor.clone()) for tensor in tensors]

    for token in range(3 * _K):
        _decode(wrapper, kv_cache, 6 + token)

    for tensor, (pointer, contents) in zip(tensors, pinned, strict=True):
        assert tensor.data_ptr() == pointer
        assert t.equal(tensor, contents)


def test_discard_explicit_segment_is_public_idempotent() -> None:
    _model, _tokenizer, _config, wrapper, _kv_cache = _build_harness()
    runtime = wrapper._self_cond
    assert runtime is not None
    runtime.provider.queue_explicit_segment({2: 0})

    assert runtime.provider.discard_explicit_segment() is None
    assert runtime.provider._queued_explicit is None
    assert runtime.provider.discard_explicit_segment() is None


def test_restore_and_kv_rewind_reproduce_decode_logits_exactly() -> None:
    _model, _tokenizer, _config, wrapper, kv_cache = _build_harness()
    _prefill(wrapper, kv_cache)
    boundary = wrapper.self_cond_runtime_snapshot()
    tokens = (6, 7, 8)

    first_logits = [_decode(wrapper, kv_cache, token).clone() for token in tokens]
    wrapper.restore_self_cond_runtime(boundary)
    kv_cache.rewind(len(tokens))
    second_logits = [_decode(wrapper, kv_cache, token) for token in tokens]

    assert all(
        t.equal(first, second) for first, second in zip(first_logits, second_logits, strict=True)
    )


def test_restore_rejects_invalid_runtime_triples_before_mutation() -> None:
    model, _tokenizer, config, wrapper, kv_cache = _build_harness()
    _prefill(wrapper, kv_cache)
    _decode(wrapper, kv_cache, 6)
    _decode(wrapper, kv_cache, 7)
    snapshot = wrapper.self_cond_runtime_snapshot()
    runtime = wrapper._self_cond
    assert runtime is not None
    original = (runtime.state, runtime.fp_buffer, runtime.tokens_in_seg)

    with pytest.raises(ValueError, match="cardinality"):
        wrapper.restore_self_cond_runtime(replace(snapshot, fp_buffer=snapshot.fp_buffer[:-1]))
    assert (runtime.state, runtime.fp_buffer, runtime.tokens_in_seg) == original

    with pytest.raises(ValueError, match="tokens_in_seg"):
        wrapper.restore_self_cond_runtime(
            replace(snapshot, tokens_in_seg=int(config.self_cond_segment_size))
        )
    assert (runtime.state, runtime.fp_buffer, runtime.tokens_in_seg) == original

    with pytest.raises(ValueError, match="tokens_in_seg"):
        wrapper.restore_self_cond_runtime(replace(snapshot, tokens_in_seg=True))
    assert (runtime.state, runtime.fp_buffer, runtime.tokens_in_seg) == original

    # An invalid snapshot state cannot be manufactured through the public
    # SelfCondState constructor: its invariant fires at construction time.
    empty_encoder_out = pack_dense_encoder_output_reference(
        t.zeros((1, 1, 1, model.encoder.config.n_embd)),
        t.zeros((1, 1, 1), dtype=t.bool),
    )
    with pytest.raises(EmptySelfConditioningSegmentError, match="empty encoder slots"):
        SelfCondState(encoder_out=empty_encoder_out, target_slot_idx=0)


def test_swap_changes_only_provider_and_config_and_decode_continues() -> None:
    model, _tokenizer, config, wrapper, kv_cache = _build_harness()
    _prefill(wrapper, kv_cache)
    runtime = wrapper._self_cond
    assert runtime is not None
    state_before = runtime.state
    assert state_before is not None
    selection = config.selection
    assert isinstance(selection, SamplingBundle)
    new_config = replace(
        config,
        score_policy=SelfCondScoreInterpolation(
            target_scores=t.zeros(N_SAE_FEATURES),
            weight=UnitInterval.parse(0.0),
            negative_selection=_negative_selection(),
        ),
    )
    new_provider = SelfConditionedFeatureProvider(model, new_config, _CPU)

    wrapper.swap_self_cond_provider(new_provider, new_config)
    logits = _decode(wrapper, kv_cache, 6)

    assert runtime.provider is new_provider
    assert runtime.config is new_config
    assert runtime.state is state_before
    assert logits.shape == (1, 1, model.dec_config.vocab_size)


def test_swap_rejection_matrix_is_transactional(
    prefilled_runtime_harness: tuple[
        Any,
        MockTokenizer,
        SelfCondConfig,
        TargetAwareEncoderDecoderEvalWrapper,
        KVCache,
    ],
) -> None:
    model, _tokenizer, config, wrapper, kv_cache = prefilled_runtime_harness
    runtime = wrapper._self_cond
    assert runtime is not None
    assert isinstance(config.selection, SamplingBundle)
    assert config.score_policy is None
    original_provider = runtime.provider
    original_config = runtime.config
    selection = config.selection

    positive_policy_config = replace(
        config,
        selection=SamplingBundle(
            selection=replace(selection.selection, max_k=5),
            percentile_table=selection.percentile_table,
            rng=selection.rng,
        ),
    )
    percentile_table_config = replace(
        config,
        selection=SamplingBundle(
            selection=selection.selection,
            percentile_table=selection.percentile_table.clone(),
            rng=selection.rng,
        ),
    )
    fp_logit_bias_config = replace(
        config,
        fp_logit_bias=t.zeros(N_SAE_FEATURES),
    )
    bucket_producer_config = replace(
        config,
        bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
    )
    pool_mode_config = replace(
        config,
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
    )
    segment_size_config = replace(
        config,
        self_cond_segment_size=PositiveInt.parse(_K + 1),
    )
    selection_rng_config = replace(
        config,
        selection=SamplingBundle(
            selection=selection.selection,
            percentile_table=selection.percentile_table,
            rng=t.Generator(device="cpu"),
        ),
    )

    cases = (
        ("self_cond_segment_size", segment_size_config),
        ("positive_count_policy", positive_policy_config),
        ("percentile_table", percentile_table_config),
        ("selection_rng", selection_rng_config),
        ("fp_logit_bias", fp_logit_bias_config),
        ("bucket_producer", bucket_producer_config),
        ("pool_mode", pool_mode_config),
    )
    for component, new_config in cases:
        new_provider = SelfConditionedFeatureProvider(model, new_config, _CPU)
        with pytest.raises(ValueError, match=component):
            wrapper.swap_self_cond_provider(new_provider, new_config)
        assert runtime.provider is original_provider
        assert runtime.config is original_config

    # Sampling structurally requires FULL_PERCENTILE_RANK, so selection_producer
    # cannot vary alone; bucket_producer is the independently constructible arm.
    # model/n_sae_features/num_buckets shape the model, while feature_mask and
    # cluster_descriptor shape its effective vocabulary, so this fixture cannot
    # vary any of them alone without changing another compatibility component.

    logits = _decode(wrapper, kv_cache, 6)
    assert logits.shape == (1, 1, model.dec_config.vocab_size)


def test_swap_rejects_different_precomputed_thresholds_transactionally() -> None:
    first_thresholds = t.zeros(N_SAE_FEATURES)
    selection = ThresholdSelectionConfig(
        selection=ThresholdSelection(
            threshold_type=ThresholdType.feature_percentile,
            threshold=0.01,
        ),
        precomputed_thresholds=first_thresholds,
    )
    config = SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(_K),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=selection,
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=N_SAE_FEATURES,
    )
    model, _tokenizer, config, wrapper, kv_cache = _build_harness(config)
    _prefill(wrapper, kv_cache)
    runtime = wrapper._self_cond
    assert runtime is not None
    original_provider = runtime.provider
    original_config = runtime.config
    new_config = replace(
        config,
        selection=replace(
            selection,
            precomputed_thresholds=first_thresholds.clone(),
        ),
    )
    new_provider = SelfConditionedFeatureProvider(model, new_config, _CPU)

    with pytest.raises(ValueError, match="precomputed_thresholds"):
        wrapper.swap_self_cond_provider(new_provider, new_config)
    assert runtime.provider is original_provider
    assert runtime.config is original_config

    logits = _decode(wrapper, kv_cache, 6)
    assert logits.shape == (1, 1, model.dec_config.vocab_size)


def test_swap_rejects_provider_config_identity_mismatch() -> None:
    model, _tokenizer, config, wrapper, _kv_cache = _build_harness()
    provider = SelfConditionedFeatureProvider(model, config, _CPU)
    distinct_config = replace(config)

    with pytest.raises(ValueError, match="provider.cfg"):
        wrapper.swap_self_cond_provider(provider, distinct_config)
