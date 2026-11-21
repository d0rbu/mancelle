"""One-shot explicit-segment override contract for self-conditioning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import nanochat.flash_attention as _fa_mod
from errors import FeatureMaskError, FeatureVocabError
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.feature_vocab import FeatureVocab
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


def _threshold_selection() -> ThresholdSelectionConfig:
    return ThresholdSelectionConfig(
        selection=ThresholdSelection(
            threshold_type=ThresholdType.segment_topk,
            threshold=3,
        ),
        precomputed_thresholds=None,
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


def _dense_config() -> SelfCondConfig:
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
        score_policy=SelfCondScoreInterpolation(
            target_scores=t.zeros(_DENSE_FEATURES),
            weight=UnitInterval.parse(0.0),
            negative_selection=_negative_selection(),
        ),
    )


def _cluster_consolidation(
    tmp_path: Path,
    clusters: list[list[int]],
) -> FeatureClusterConsolidation:
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": clusters}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        int(N_SAE_FEATURES),
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, None)


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
        selection=_threshold_selection(),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=NUM_BUCKETS,
        n_sae_features=N_SAE_FEATURES,
    )


def test_empty_explicit_segment_raises_and_normal_selection_still_runs(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)

    with pytest.raises(
        ValueError,
        match="explicit segment requires at least one feature",
    ):
        provider.queue_explicit_segment({})

    provider.step(
        state,
        t.randn(1, basic_config.self_cond_segment_size, basic_config.n_sae_features),
    )
    assert provider._queued_explicit is None
    assert provider.last_appended_segment
    assert provider.last_appended_segment != {}


def test_double_queue_raises_and_preserves_first_override(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    queued = {5: 1, 9: 2}
    recording = _record_encoder(tiny_model_with_fp)
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    recording.calls.clear()
    provider.queue_explicit_segment(queued)

    with pytest.raises(
        RuntimeError,
        match="explicit segment override is already queued",
    ):
        provider.queue_explicit_segment({10: 3})

    provider.step(state, t.zeros(1, 1, basic_config.n_sae_features))
    assert _segment_from_call(recording.calls[-1]) == queued
    assert provider._queued_explicit is None


def test_explicit_segment_validates_feature_and_encoder_bucket_ranges(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    encoder_num_buckets = int(tiny_model_with_fp.encoder.config.num_buckets)

    with pytest.raises(
        ValueError,
        match=rf"explicit segment: feature id {N_SAE_FEATURES} is out of range",
    ):
        provider.queue_explicit_segment({int(N_SAE_FEATURES): 0})
    assert provider._queued_explicit is None

    with pytest.raises(
        ValueError,
        match=rf"explicit segment: bucket {encoder_num_buckets} is out of range",
    ):
        provider.queue_explicit_segment({0: encoder_num_buckets})
    assert provider._queued_explicit is None

    provider.queue_explicit_segment({0: encoder_num_buckets - 1})
    assert provider._queued_explicit == {0: encoder_num_buckets - 1}


def test_explicit_segment_rejects_float_feature_id(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )

    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="parameter features",
    ):
        provider.queue_explicit_segment({3.5: 0})

    assert provider._queued_explicit is None


def test_explicit_segment_rejects_bool_bucket(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )

    with pytest.raises(
        BeartypeCallHintParamViolation,
        match="parameter features",
    ):
        provider.queue_explicit_segment({3: True})

    assert provider._queued_explicit is None


def test_reserved_inactive_bucket_is_accepted_when_config_has_one_active_bucket() -> None:
    model = _dense_model()
    config = _dense_config()
    provider = SelfConditionedFeatureProvider(model, config, _CPU_DEVICE)

    assert int(config.num_buckets) == 1
    assert int(model.encoder.config.num_buckets) == 2
    provider.queue_explicit_segment({0: 1})
    assert provider._queued_explicit == {0: 1}


def test_compact_vocab_rejects_explicit_id_without_an_encoder_row() -> None:
    vocabulary = FeatureVocab(
        n_sae_features=N_SAE_FEATURES,
        original_ids=(2, 7, 11, 19),
    )
    model = build_model_with_feature_prediction(feature_vocab=vocabulary)
    provider = SelfConditionedFeatureProvider(
        model,
        SelfCondConfig(
            self_cond_segment_size=PositiveInt.parse(8),
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
            selection=_threshold_selection(),
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=NUM_BUCKETS,
            n_sae_features=N_SAE_FEATURES,
        ),
        _CPU_DEVICE,
    )

    with pytest.raises(
        FeatureVocabError,
        match="outside the loaded feature vocabulary",
    ):
        provider.queue_explicit_segment({13: 0})

    assert provider._queued_explicit is None


def test_clustered_compact_vocab_rejects_consolidated_away_explicit_id(
    tmp_path: Path,
) -> None:
    representative = 2
    consolidated_away = 3
    vocabulary = FeatureVocab(
        n_sae_features=N_SAE_FEATURES,
        original_ids=(representative, 7, 11, 19),
    )
    consolidation = _cluster_consolidation(
        tmp_path,
        [[representative, consolidated_away]],
    )
    model = build_model_with_feature_prediction(feature_vocab=vocabulary)
    provider = SelfConditionedFeatureProvider(
        model,
        SelfCondConfig(
            self_cond_segment_size=PositiveInt.parse(8),
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
            selection=_threshold_selection(),
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=NUM_BUCKETS,
            n_sae_features=N_SAE_FEATURES,
            cluster_consolidation=consolidation,
        ),
        _CPU_DEVICE,
    )

    # Adjudication: compaction rejects away ids at FeatureVocab before the cluster guard.
    with pytest.raises(
        FeatureVocabError,
        match="outside the loaded feature vocabulary",
    ):
        provider.queue_explicit_segment({consolidated_away: 0})

    assert provider._queued_explicit is None


def test_feature_mask_rejects_masked_explicit_id_with_its_own_error() -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (7,))
    model = build_model_with_feature_prediction()
    provider = SelfConditionedFeatureProvider(
        model,
        SelfCondConfig(
            self_cond_segment_size=PositiveInt.parse(8),
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
            selection=_threshold_selection(),
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=NUM_BUCKETS,
            n_sae_features=N_SAE_FEATURES,
            feature_mask=mask,
        ),
        _CPU_DEVICE,
    )

    with pytest.raises(
        FeatureMaskError,
        match="feature_mask violated: mode=blacklist produced disallowed ids",
    ):
        provider.queue_explicit_segment({7: 0})

    assert provider._queued_explicit is None


def test_step_consumes_exact_explicit_segment(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    queued = {5: 1, 9: 3, 12: 0}
    recording = _record_encoder(tiny_model_with_fp)
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    recording.calls.clear()
    provider.queue_explicit_segment(queued)

    new_state = provider.step(
        state,
        t.randn(1, basic_config.self_cond_segment_size, basic_config.n_sae_features),
    )

    assert new_state.target_slot_idx == state.target_slot_idx + 1
    assert _segment_from_call(recording.calls[-1]) == queued
    assert provider.last_appended_segment == queued
    assert provider.last_appended_segment is not queued
    assert provider._queued_explicit is None


def test_malformed_fp_logits_retains_queued_explicit_segment(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    queued = {5: 1, 9: 3}
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    provider.queue_explicit_segment(queued)

    with pytest.raises(ValueError, match="must have shape"):
        provider.step(state, t.zeros(1, basic_config.n_sae_features))

    assert provider._queued_explicit == queued
    provider.step(state, t.zeros(1, 1, basic_config.n_sae_features))
    assert provider.last_appended_segment == queued
    assert provider._queued_explicit is None


def test_seed_from_prompt_consumes_exact_explicit_segment(
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    queued = {1: 0, 3: 2}
    recording = _record_encoder(tiny_model_with_fp)
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    recording.calls.clear()
    provider.queue_explicit_segment(queued)

    new_state = provider.seed_from_prompt(
        state,
        t.randn(1, 17, basic_config.n_sae_features),
    )

    assert new_state.target_slot_idx == state.target_slot_idx + 1
    assert _segment_from_call(recording.calls[-1]) == queued
    assert provider.last_appended_segment == queued
    assert provider.last_appended_segment is not queued
    assert provider._queued_explicit is None


def test_override_is_one_shot_and_next_step_uses_normal_selection() -> None:
    model = _dense_model()
    recording = _record_encoder(model)
    provider = SelfConditionedFeatureProvider(model, _dense_config(), _CPU_DEVICE)
    state = build_test_self_cond_state(provider)
    recording.calls.clear()
    queued = {3: 0}
    logits = t.tensor([[[6.0, 5.0, 4.0, -4.0, -5.0, -6.0]]])
    provider.queue_explicit_segment(queued)

    state = provider.step(state, logits)
    overridden = _segment_from_call(recording.calls[-1])
    state = provider.step(state, logits)
    selected = _segment_from_call(recording.calls[-1])

    assert state.target_slot_idx == 2
    assert overridden == queued
    assert selected != queued
    assert provider.last_appended_segment == selected
    assert provider._queued_explicit is None


def test_override_does_not_advance_dense_selection_rng() -> None:
    config = _dense_config()
    provider = SelfConditionedFeatureProvider(_dense_model(), config, _CPU_DEVICE)
    state = build_test_self_cond_state(provider)
    logits = t.tensor([[[6.0, 5.0, 4.0, -4.0, -5.0, -6.0]]])
    before_override = config.selection.rng.get_state().clone()
    provider.queue_explicit_segment({3: 0})

    state = provider.step(state, logits)
    after_override = config.selection.rng.get_state().clone()
    assert t.equal(after_override, before_override)

    provider.step(state, logits)
    after_normal_step = config.selection.rng.get_state()
    assert not t.equal(after_normal_step, after_override)


def test_failed_override_append_is_disarmed_and_preserves_observation(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model_with_fp: Any,
    basic_config: SelfCondConfig,
) -> None:
    provider = SelfConditionedFeatureProvider(
        tiny_model_with_fp,
        basic_config,
        _CPU_DEVICE,
    )
    state = build_test_self_cond_state(provider)
    state = provider.step(state, t.randn(1, 1, basic_config.n_sae_features))
    previous = provider.last_appended_segment
    assert previous is not None

    def fail_encode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(provider, "_encode_one_segment", fail_encode)
    provider.queue_explicit_segment({5: 1})

    with pytest.raises(RuntimeError, match="boom"):
        provider.step(state, t.zeros(1, 1, basic_config.n_sae_features))

    assert provider._queued_explicit is None
    assert provider.last_appended_segment is previous
