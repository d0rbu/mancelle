"""Observation contract for the provider's most recently appended segment."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
import steering.conditional.encoder_decoder.self_conditioned as selfcond_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
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
from steering.refined_types import PositiveInt, UnitInterval
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_model_with_feature_prediction,
    build_tiny_model,
)
from tests.steering._self_conditioned_state import build_test_self_cond_state

_CPU_DEVICE = t.device("cpu")
_DENSE_FEATURES = 6


class _RecordingEncoder(t.nn.Module):
    def __init__(self, delegate: Any) -> None:
        super().__init__()
        self.delegate = delegate
        self.config = delegate.config
        self.feature_vocab = delegate.feature_vocab
        self.calls: list[tuple[t.Tensor, t.Tensor, t.Tensor]] = []

    def forward(
        self,
        fids: t.Tensor,
        mags: t.Tensor,
        mask: t.Tensor,
    ) -> Any:
        self.calls.append((fids.clone(), mags.clone(), mask.clone()))
        return self.delegate(fids, mags, mask)


def _record_encoder(model: Any) -> _RecordingEncoder:
    recording = _RecordingEncoder(model.encoder)
    model.encoder = recording
    return recording


def _segment_from_call(
    call: tuple[t.Tensor, t.Tensor, t.Tensor],
) -> dict[int, int]:
    fids, mags, mask = call
    return {
        int(fid): int(mag)
        for fid, mag in zip(fids[mask].tolist(), mags[mask].tolist(), strict=True)
    }


def _threshold_selection(
    threshold_type: ThresholdType,
    threshold: float | int,
    precomputed: t.Tensor | None = None,
) -> ThresholdSelectionConfig:
    return ThresholdSelectionConfig(
        selection=ThresholdSelection(
            threshold_type=threshold_type,
            threshold=threshold,
        ),
        precomputed_thresholds=precomputed,
    )


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
        max_k=4,
    )


def _dense_model() -> Any:
    return build_tiny_model(
        n_sae_features=_DENSE_FEATURES,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )


def _dense_config(
    score_policy: SelfCondScoreInterpolation | SelfCondScoreMasking,
) -> SelfCondConfig:
    rng = t.Generator(device="cpu")
    rng.manual_seed(123)
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(4),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            selection=_positive_selection(),
            percentile_table=t.zeros((_DENSE_FEATURES, 99), dtype=t.float64),
            rng=rng,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=PositiveInt.parse(_DENSE_FEATURES),
        score_policy=score_policy,
    )


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


@pytest.fixture
def tiny_model_with_fp():
    t.manual_seed(0)
    return build_model_with_feature_prediction()


@pytest.fixture
def basic_config() -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(8),
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        selection=_threshold_selection(ThresholdType.segment_topk, 3),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=NUM_BUCKETS,
        n_sae_features=N_SAE_FEATURES,
    )


def test_last_appended_segment_is_none_before_append(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )

    assert provider.last_appended_segment is None


def test_seed_from_prompt_records_exact_encoded_segment(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    recording = _record_encoder(tiny_model_with_fp)
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    recording.calls.clear()

    new_state = provider.seed_from_prompt(
        state,
        t.randn(1, 32, basic_config.n_sae_features),
    )
    expected = _segment_from_call(recording.calls[-1])

    assert new_state.target_slot_idx == state.target_slot_idx + 1
    assert provider.last_appended_segment == expected


def test_legacy_step_records_exact_encoded_segment(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    recording = _record_encoder(tiny_model_with_fp)
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    recording.calls.clear()

    new_state = provider.step(
        state,
        t.randn(1, basic_config.self_cond_segment_size, basic_config.n_sae_features),
    )
    expected = _segment_from_call(recording.calls[-1])

    assert new_state.encoder_out.layout.n_segments == state.encoder_out.layout.n_segments + 1
    assert provider.last_appended_segment == expected


def test_interpolation_weight_zero_records_drawn_positive_and_negative_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_counts: list[tuple[int, int]] = []

    def fixed_counts(*_args: Any, **_kwargs: Any) -> tuple[t.Tensor, t.Tensor]:
        drawn_counts.append((2, 2))
        return t.tensor([2]), t.tensor([2])

    monkeypatch.setattr(selfcond_module, "sample_action_counts", fixed_counts)
    config = _dense_config(
        SelfCondScoreInterpolation(
            target_scores=t.zeros(_DENSE_FEATURES),
            weight=UnitInterval.parse(0.0),
            negative_selection=_negative_selection(),
        )
    )
    provider = SelfConditionedFeatureProvider(_dense_model(), config, _CPU_DEVICE)
    state = build_test_self_cond_state(provider)
    logits = t.tensor([[[6.0, 5.0, 4.0, -4.0, -5.0, -6.0]]])

    provider.step(state, logits)

    segment = provider.last_appended_segment
    assert segment is not None
    assert segment == {0: 0, 1: 0, 5: 1, 4: 1}
    assert len(segment) == sum(drawn_counts[-1])
    assert sum(bucket == 0 for bucket in segment.values()) == drawn_counts[-1][0]
    assert sum(bucket == 1 for bucket in segment.values()) == drawn_counts[-1][1]


def test_masking_records_only_target_sign_consistent_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drawn_counts: list[tuple[int, int]] = []

    def fixed_counts(*_args: Any, **_kwargs: Any) -> tuple[t.Tensor, t.Tensor]:
        drawn_counts.append((2, 2))
        return t.tensor([2]), t.tensor([2])

    monkeypatch.setattr(selfcond_module, "sample_action_counts", fixed_counts)
    target = t.tensor([0.9, 0.7, 0.1, -0.1, -0.7, -0.9])
    config = _dense_config(
        SelfCondScoreMasking(
            target_scores=target,
            positive_pool_limit=PositiveInt.parse(2),
            negative_pool_limit=PositiveInt.parse(2),
            negative_selection=_negative_selection(),
        )
    )
    provider = SelfConditionedFeatureProvider(_dense_model(), config, _CPU_DEVICE)
    state = build_test_self_cond_state(provider)

    provider.step(state, t.tensor([[[6.0, 5.0, 4.0, -4.0, -5.0, -6.0]]]))

    segment = provider.last_appended_segment
    assert segment is not None
    assert segment == {0: 0, 1: 0, 5: 1, 4: 1}
    assert len(segment) == sum(drawn_counts[-1])
    assert sum(bucket == 0 for bucket in segment.values()) == drawn_counts[-1][0]
    assert sum(bucket == 1 for bucket in segment.values()) == drawn_counts[-1][1]
    for feature_id, bucket in segment.items():
        assert (target[feature_id] > 0).item() if bucket == 0 else (target[feature_id] < 0).item()


def test_failed_append_preserves_last_successful_segment(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    config = SelfCondConfig(
        self_cond_segment_size=basic_config.self_cond_segment_size,
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        selection=_threshold_selection(
            ThresholdType.feature_percentile,
            0.99,
            t.zeros(basic_config.n_sae_features),
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=basic_config.num_buckets,
        n_sae_features=basic_config.n_sae_features,
    )
    provider = SelfConditionedFeatureProvider(tiny_model_with_fp, config, _CPU_DEVICE)
    state = build_test_self_cond_state(provider)
    state = provider.step(
        state,
        t.full((1, 1, config.n_sae_features), 100.0),
    )
    previous = provider.last_appended_segment
    assert previous is not None
    expected = dict(previous)

    with pytest.raises(
        EmptySelfConditioningSegmentError,
        match="fp_logits_tail selected zero features",
    ):
        provider.step(
            state,
            t.full((1, 1, config.n_sae_features), -100.0),
        )

    assert provider.last_appended_segment == expected
    assert provider.last_appended_segment is previous


@pytest.mark.parametrize("dense_interpolation", [False, True])
def test_post_encode_append_failure_preserves_last_successful_segment(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
    dense_interpolation: bool,
) -> None:
    if dense_interpolation:
        config = _dense_config(
            SelfCondScoreInterpolation(
                target_scores=t.zeros(_DENSE_FEATURES),
                weight=UnitInterval.parse(0.0),
                negative_selection=_negative_selection(),
            )
        )
        provider = SelfConditionedFeatureProvider(_dense_model(), config, _CPU_DEVICE)
        logits = t.tensor([[[6.0, 5.0, 4.0, -4.0, -5.0, -6.0]]])
    else:
        config = basic_config
        provider = SelfConditionedFeatureProvider(
            tiny_model_with_fp,
            config,
            _CPU_DEVICE,
        )
        logits = t.randn(1, 1, config.n_sae_features)

    state = build_test_self_cond_state(provider)
    state = provider.step(state, logits)
    previous = provider.last_appended_segment
    assert previous is not None
    expected = dict(previous)

    def fail_append(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("append boom")

    monkeypatch.setattr(
        selfcond_module,
        "append_packed_encoder_segment",
        fail_append,
    )

    with pytest.raises(RuntimeError, match="append boom"):
        provider.step(state, logits)

    assert provider.last_appended_segment == expected
    assert provider.last_appended_segment is previous


def test_last_appended_segment_is_a_fresh_private_copy_each_step(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    recording = _record_encoder(tiny_model_with_fp)
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    state = provider.step(
        state,
        t.randn(1, basic_config.self_cond_segment_size, basic_config.n_sae_features),
    )
    first = provider.last_appended_segment
    assert first is not None
    first[10_000] = 99
    recording.calls.clear()

    new_state = provider.step(
        state,
        t.randn(1, basic_config.self_cond_segment_size, basic_config.n_sae_features),
    )
    second = provider.last_appended_segment

    assert new_state.target_slot_idx == state.target_slot_idx + 1
    assert second == _segment_from_call(recording.calls[-1])
    assert second is not first
    assert second is not None
    assert 10_000 not in second
