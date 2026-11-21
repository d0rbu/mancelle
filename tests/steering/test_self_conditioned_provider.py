"""Unit tests for the self-conditioned feature provider.

The provider owns the encoder state during self-conditioned generation:
- Holds the running packed encoder state and target slot index.
- Exposes ``seed_from_prompt`` (a slot from prompt fp_logits at end of
  prefill) and ``step`` (subsequent slots after each K-token decoder span).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.profiler import ProfilerActivity, profile

import nanochat.flash_attention as _fa_mod
from errors import GradientRankingUnsupportedError
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.delta import DeltaSamplingWeights
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    append_packed_encoder_segment,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondConfig,
    SelfConditionedFeatureProvider,
)
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.gradient_overlay import GradientAttributionMode, GradientRankingSpec
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_model_with_feature_prediction,
    build_tiny_model,
)
from tests.steering._self_conditioned_state import build_test_self_cond_state

_CPU_DEVICE = torch.device("cpu")
PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)


def test_self_cond_config_preserves_historical_optional_positional_order() -> None:
    assert [field.name for field in fields(SelfCondConfig)][-5:] == [
        "feature_mask",
        "cluster_consolidation",
        "fp_logit_bias",
        "score_policy",
        "segment_mode",
    ]


def _segment_values(packed: PackedEncoderOutput, segment: int) -> torch.Tensor:
    keep = packed.layout.valid_mask & (packed.layout.segment_indices == segment)
    return packed.values[0, keep]


def _segment_count(packed: PackedEncoderOutput, segment: int) -> int:
    keep = packed.layout.valid_mask & (packed.layout.segment_indices == segment)
    return int(keep.sum().item())


def _threshold_selection(threshold_type, threshold, precomputed=None):
    """Test helper: wrap a (threshold_type, threshold) into the
    ThresholdSelectionConfig + ThresholdSelection nesting that
    SelfCondConfig expects under the new union API."""
    return ThresholdSelectionConfig(
        selection=ThresholdSelection(
            threshold_type=threshold_type, threshold=threshold,
        ),
        precomputed_thresholds=precomputed,
    )


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA backend; flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


@pytest.fixture
def tiny_model_with_fp():
    torch.manual_seed(0)
    return build_model_with_feature_prediction()


@pytest.fixture
def basic_config():
    """A simple selection config: top-3 by predicted percentile."""
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(8),
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        selection=_threshold_selection(ThresholdType.segment_topk, 3),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=NUM_BUCKETS,
        n_sae_features=N_SAE_FEATURES,
    )


def _make_provider(model, cfg, device=_CPU_DEVICE):
    return SelfConditionedFeatureProvider(model=model, cfg=cfg, device=device)


# ---------------------------------------------------------------------------
# Construction contracts
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_config_rejects_best_token_pool(self, basic_config):
        with pytest.raises(ValueError, match="BEST_TOKEN"):
            SelfCondConfig(
                self_cond_segment_size=basic_config.self_cond_segment_size,
                pool_mode=FeaturePredLossMode.BEST_TOKEN,
                selection=basic_config.selection,
                selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
                bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
                num_buckets=basic_config.num_buckets,
                n_sae_features=basic_config.n_sae_features,
            )

    def test_config_rejects_invalid_selection_type(self, basic_config):
        with pytest.raises(BeartypeCallHintParamViolation, match="selection"):
            SelfCondConfig(
                self_cond_segment_size=basic_config.self_cond_segment_size,
                pool_mode=basic_config.pool_mode,
                selection=cast(Any, "not_a_valid_selection"),
                selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
                bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
                num_buckets=basic_config.num_buckets,
                n_sae_features=basic_config.n_sae_features,
            )

    @pytest.mark.parametrize(
        ("field", "value", "other_field", "other_value"),
        [
            (
                "selection_producer",
                FeatureSelectionProducer.FULL_PERCENTILE_RANK.value,
                "bucket_producer",
                FeatureBucketProducer.UNIFORM_PERCENTILE,
            ),
            (
                "bucket_producer",
                FeatureBucketProducer.UNIFORM_PERCENTILE.value,
                "selection_producer",
                FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            ),
        ],
    )
    def test_config_rejects_raw_producer_strings(
        self,
        basic_config,
        field: str,
        value: str,
        other_field: str,
        other_value: object,
    ):
        with pytest.raises(
            BeartypeCallHintParamViolation,
            match=field,
        ):
            kwargs = {
                "self_cond_segment_size": basic_config.self_cond_segment_size,
                "pool_mode": basic_config.pool_mode,
                "selection": basic_config.selection,
                "num_buckets": basic_config.num_buckets,
                "n_sae_features": basic_config.n_sae_features,
            }
            kwargs[field] = cast(Any, value)
            kwargs[other_field] = other_value
            SelfCondConfig(**cast(Any, kwargs))

    @pytest.mark.parametrize("missing_field", ["selection_producer", "bucket_producer"])
    def test_config_requires_both_producers(self, basic_config, missing_field: str):
        kwargs = {
            "self_cond_segment_size": basic_config.self_cond_segment_size,
            "pool_mode": basic_config.pool_mode,
            "selection": basic_config.selection,
            "selection_producer": FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            "bucket_producer": FeatureBucketProducer.UNIFORM_PERCENTILE,
            "num_buckets": basic_config.num_buckets,
            "n_sae_features": basic_config.n_sae_features,
        }
        del kwargs[missing_field]
        with pytest.raises(TypeError, match=missing_field):
            SelfCondConfig(**cast(Any, kwargs))

    @pytest.mark.parametrize(
        ("threshold_type", "threshold"),
        [
            (ThresholdType.segment_topk, 3),
            (ThresholdType.segment_percentile, 0.5),
            (ThresholdType.feature_percentile_segment_topk, 3.9),
        ],
    )
    def test_raw_cross_feature_selection_fails_closed(
        self,
        basic_config,
        threshold_type: ThresholdType,
        threshold: float,
    ):
        with pytest.raises(NotImplementedError, match="RAW_MAGNITUDE.*only.*feature_percentile"):
            SelfCondConfig(
                self_cond_segment_size=basic_config.self_cond_segment_size,
                pool_mode=basic_config.pool_mode,
                selection=_threshold_selection(threshold_type, threshold),
                selection_producer=FeatureSelectionProducer.RAW_MAGNITUDE,
                bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
                num_buckets=basic_config.num_buckets,
                n_sae_features=basic_config.n_sae_features,
            )

    def test_raw_sampling_producer_fails_closed(self, basic_config):
        bundle = SamplingBundle(
            selection=SamplingSelection(mean=2.0, std=2.0),
            percentile_table=torch.zeros(N_SAE_FEATURES, 99),
            rng=torch.Generator().manual_seed(0),
        )
        with pytest.raises(
            NotImplementedError,
            match="sampling requires.*FULL_PERCENTILE_RANK",
        ):
            SelfCondConfig(
                self_cond_segment_size=basic_config.self_cond_segment_size,
                pool_mode=basic_config.pool_mode,
                selection=bundle,
                selection_producer=FeatureSelectionProducer.RAW_MAGNITUDE,
                bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
                num_buckets=basic_config.num_buckets,
                n_sae_features=basic_config.n_sae_features,
            )

    def test_delta_weighted_sampling_fails_before_generation(self, basic_config):
        bundle = SamplingBundle(
            selection=SamplingSelection(
                mean=2.0,
                std=2.0,
                delta_weights=DeltaSamplingWeights.percentile_delta,
            ),
            percentile_table=torch.zeros(N_SAE_FEATURES, 99),
            rng=torch.Generator().manual_seed(0),
        )
        with pytest.raises(NotImplementedError, match="delta_weights"):
            SelfCondConfig(
                self_cond_segment_size=basic_config.self_cond_segment_size,
                pool_mode=basic_config.pool_mode,
                selection=bundle,
                selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
                bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
                num_buckets=basic_config.num_buckets,
                n_sae_features=basic_config.n_sae_features,
            )

    def test_gradient_ranked_sampling_fails_before_generation(self, basic_config):
        bundle = SamplingBundle(
            selection=SamplingSelection(
                mean=2.0,
                std=2.0,
                gradient=GradientRankingSpec(
                    mode=GradientAttributionMode.future_half_split,
                ),
            ),
            percentile_table=torch.zeros(N_SAE_FEATURES, 99),
            rng=torch.Generator().manual_seed(0),
        )
        with pytest.raises(GradientRankingUnsupportedError, match="offline"):
            SelfCondConfig(
                self_cond_segment_size=basic_config.self_cond_segment_size,
                pool_mode=basic_config.pool_mode,
                selection=bundle,
                selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
                bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
                num_buckets=basic_config.num_buckets,
                n_sae_features=basic_config.n_sae_features,
            )

    def test_provider_rejects_model_without_fp_head(self, basic_config):
        model = build_tiny_model(feature_pred_config=None)
        with pytest.raises(RuntimeError, match="feature_pred_head=None"):
            _make_provider(model, basic_config)

    def test_provider_rejects_checkpoint_trained_with_best_token(self, basic_config):
        model = build_tiny_model(
            feature_pred_config=FeaturePredictionConfig(
                loss_mode=FeaturePredLossMode.BEST_TOKEN,
            ),
        )
        with pytest.raises(RuntimeError, match="trained.*BEST_TOKEN"):
            _make_provider(model, basic_config)


@PROPERTY_SETTINGS
@given(
    self_cond_segment_size=st.integers(min_value=1),
    num_buckets=st.integers(min_value=1),
    n_sae_features=st.integers(min_value=1),
)
def test_config_accepts_the_complete_positive_integer_domains(
    self_cond_segment_size: int,
    num_buckets: int,
    n_sae_features: int,
) -> None:
    config = SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(self_cond_segment_size),
        pool_mode=FeaturePredLossMode.LAST_TOKEN,
        selection=_threshold_selection(ThresholdType.segment_topk, 1),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(num_buckets),
        n_sae_features=PositiveInt.parse(n_sae_features),
    )
    assert config.self_cond_segment_size == self_cond_segment_size
    assert config.num_buckets == num_buckets
    assert config.n_sae_features == n_sae_features


@PROPERTY_SETTINGS
@given(
    field=st.sampled_from(
        ("self_cond_segment_size", "num_buckets", "n_sae_features")
    ),
    value=st.one_of(st.integers(max_value=0), st.booleans()),
)
def test_config_rejects_every_non_positive_or_boolean_integer_field(
    field: str,
    value: int,
) -> None:
    kwargs: dict[str, Any] = {
        "self_cond_segment_size": PositiveInt.parse(4),
        "pool_mode": FeaturePredLossMode.LAST_TOKEN,
        "selection": _threshold_selection(ThresholdType.segment_topk, 1),
        "selection_producer": FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        "bucket_producer": FeatureBucketProducer.UNIFORM_PERCENTILE,
        "num_buckets": PositiveInt.parse(4),
        "n_sae_features": PositiveInt.parse(16),
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        SelfCondConfig(**kwargs)

# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep:
    def test_no_empty_segment_builder_or_encoder_bypass(
        self,
        tiny_model_with_fp,
        basic_config,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = _make_provider(tiny_model_with_fp, basic_config)
        assert not hasattr(provider, "_build_empty_segment_inputs")
        with pytest.raises(
            EmptySelfConditioningSegmentError,
            match="require at least one feature",
        ):
            provider._build_segment_inputs({})

        def forbidden_encoder(*_args, **_kwargs):
            raise AssertionError("all-masked input must fail before encoder invocation")

        monkeypatch.setattr(provider._model.encoder, "forward", forbidden_encoder)
        zeros = torch.zeros((1, 1, 1), dtype=torch.long)
        with pytest.raises(
            EmptySelfConditioningSegmentError,
            match="refusing to invoke the encoder for an all-masked",
        ):
            provider._encode_one_segment(
                zeros,
                zeros,
                torch.zeros_like(zeros, dtype=torch.bool),
            )

    def test_rejects_existing_state_with_an_empty_slot(
        self,
        tiny_model_with_fp,
        basic_config,
    ) -> None:
        provider = _make_provider(tiny_model_with_fp, basic_config)
        valid_state = build_test_self_cond_state(provider)
        valid_state.encoder_out.layout.valid_mask.zero_()
        with pytest.raises(
            EmptySelfConditioningSegmentError,
            match=r"SelfCondState containing empty encoder slots \[0\]",
        ):
            provider.step(
                valid_state,
                torch.zeros((1, 1, basic_config.n_sae_features)),
            )

    @pytest.mark.parametrize("entrypoint", ["seed_from_prompt", "step"])
    def test_zero_feature_selection_fails_before_encoding(
        self,
        tiny_model_with_fp,
        basic_config,
        monkeypatch: pytest.MonkeyPatch,
        entrypoint: str,
    ) -> None:
        config = SelfCondConfig(
            self_cond_segment_size=basic_config.self_cond_segment_size,
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
            selection=_threshold_selection(
                ThresholdType.feature_percentile,
                0.99,
                torch.zeros(basic_config.n_sae_features),
            ),
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=basic_config.num_buckets,
            n_sae_features=basic_config.n_sae_features,
        )
        provider = _make_provider(tiny_model_with_fp, config)
        state = build_test_self_cond_state(provider)

        def forbidden_encode(*_args, **_kwargs):
            raise AssertionError("empty selection must fail before encoder invocation")

        monkeypatch.setattr(provider, "_encode_one_segment", forbidden_encode)
        logits = torch.full((1, 1, config.n_sae_features), -100.0)
        with pytest.raises(
            EmptySelfConditioningSegmentError,
            match=rf"{entrypoint.replace('seed_from_prompt', 'prompt_fp_logits').replace('step', 'fp_logits_tail')} selected zero features",
        ):
            getattr(provider, entrypoint)(state, logits)

    def test_extends_n_seg_by_one(self, tiny_model_with_fp, basic_config):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        n_sae = basic_config.n_sae_features
        # Simulate fp_logits: K=8 decoder positions
        fp_logits_tail = torch.randn(1, 8, n_sae)
        new_state = provider.step(state, fp_logits_tail)
        assert (
            new_state.encoder_out.layout.n_segments
            == state.encoder_out.layout.n_segments + 1
        )
        assert new_state.target_slot_idx == state.target_slot_idx + 1

    def test_torch_profiler_sees_every_default_provider_phase(
        self,
        tiny_model_with_fp,
        basic_config,
    ):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        fp_logits_tail = torch.randn(1, 8, basic_config.n_sae_features)
        reference_provider = _make_provider(tiny_model_with_fp, basic_config)
        reference_state = build_test_self_cond_state(reference_provider)
        reference = reference_provider.step(reference_state, fp_logits_tail)

        with profile(activities=[ProfilerActivity.CPU], with_stack=False) as profiler:
            actual = provider.step(state, fp_logits_tail)

        names = {event.name for event in profiler.events()}
        assert actual.target_slot_idx == reference.target_slot_idx
        assert torch.equal(actual.encoder_out.values, reference.encoder_out.values)
        assert torch.equal(
            actual.encoder_out.layout.flat_indices,
            reference.encoder_out.layout.flat_indices,
        )
        assert torch.equal(
            actual.encoder_out.layout.valid_mask,
            reference.encoder_out.layout.valid_mask,
        )
        assert {
            "selfcond/provider/pool_and_prepare",
            "selfcond/provider/select_features",
            "selfcond/provider/assign_buckets",
            "selfcond/provider/materialize_segment",
            "selfcond/provider/encode_segment",
            "selfcond/provider/append_segment",
        } <= names

    def test_does_not_mutate_prior_slots(self, tiny_model_with_fp, basic_config):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider, {1: 0})
        prior_values = state.encoder_out.values.clone()
        prior_slot_0 = _segment_values(state.encoder_out, 0).clone()
        fp_logits_tail = torch.randn(1, 8, basic_config.n_sae_features)
        new_state = provider.step(state, fp_logits_tail)
        # The OLD state should still be unchanged
        assert torch.equal(state.encoder_out.values, prior_values)
        # New state slot 0 is preserved bitwise; append changes coordinates,
        # never the already-encoded values.
        assert torch.equal(_segment_values(new_state.encoder_out, 0), prior_slot_0)

    def test_step_three_times_grows_to_four_slots(
        self, tiny_model_with_fp, basic_config
    ):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        for _ in range(3):
            fp_logits_tail = torch.randn(1, 8, basic_config.n_sae_features)
            state = provider.step(state, fp_logits_tail)
        assert state.encoder_out.layout.n_segments == 4
        assert state.target_slot_idx == 3

    def test_returns_new_object_does_not_mutate_input(
        self, tiny_model_with_fp, basic_config
    ):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        old_n_seg = state.encoder_out.layout.n_segments
        fp_logits_tail = torch.randn(1, 8, basic_config.n_sae_features)
        new_state = provider.step(state, fp_logits_tail)
        # The old state should still exist and be at its original n_seg
        assert state.encoder_out.layout.n_segments == old_n_seg
        assert new_state is not state


class TestStepPoolModes:
    def test_last_token_uses_position_minus_one(
        self, tiny_model_with_fp, basic_config
    ):
        # Build fp_logits where only the last position has nonzero values for
        # feature_id=0; all other features and positions are zero. The
        # selected feature should be 0 because it has the highest sigmoid.
        config = SelfCondConfig(
            self_cond_segment_size=PositiveInt.parse(4),
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
            selection=_threshold_selection(ThresholdType.segment_topk, 1),
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=NUM_BUCKETS,
            n_sae_features=basic_config.n_sae_features,
        )
        provider = _make_provider(tiny_model_with_fp, config)
        state = build_test_self_cond_state(provider)
        fp_logits_tail = torch.full(
            (1, config.self_cond_segment_size, config.n_sae_features), -10.0,
        )
        # Position -1, feature 0 is large positive
        fp_logits_tail[0, -1, 0] = 10.0
        new_state = provider.step(state, fp_logits_tail)
        # Newest segment's mask has exactly one True
        assert _segment_count(
            new_state.encoder_out,
            new_state.encoder_out.layout.n_segments - 1,
        ) == 1

    def test_full_segment_pool(self, tiny_model_with_fp, basic_config):
        config = SelfCondConfig(
            self_cond_segment_size=PositiveInt.parse(4),
            pool_mode=FeaturePredLossMode.FULL_SEGMENT,
            selection=_threshold_selection(ThresholdType.segment_topk, 1),
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=NUM_BUCKETS,
            n_sae_features=basic_config.n_sae_features,
        )
        provider = _make_provider(tiny_model_with_fp, config)
        state = build_test_self_cond_state(provider)
        # Set feature 1 high in all positions, feature 0 high only in pos 0
        fp_logits_tail = torch.full(
            (1, config.self_cond_segment_size, config.n_sae_features), -10.0,
        )
        fp_logits_tail[0, :, 1] = 10.0  # feature 1 high in all positions
        fp_logits_tail[0, 0, 0] = 10.0  # feature 0 high only at pos 0
        new_state = provider.step(state, fp_logits_tail)
        # Top-1 by mean → feature 1 (mean 10.0) over feature 0 (mean ≈ 10/4)
        assert _segment_count(
            new_state.encoder_out,
            new_state.encoder_out.layout.n_segments - 1,
        ) == 1


class TestStepNanGuard:
    def test_nan_in_fp_logits_raises(self, tiny_model_with_fp, basic_config):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        fp_logits_tail = torch.zeros(1, 8, basic_config.n_sae_features)
        fp_logits_tail[0, 3, 7] = float("nan")
        with pytest.raises((ValueError, RuntimeError)):
            provider.step(state, fp_logits_tail)


class TestStepSampling:
    def test_sampling_step_produces_a_segment(self, tiny_model_with_fp):
        sampling = SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=10,
            mode=SamplingMode.weighted,
        )
        rng = torch.Generator(device="cpu")
        rng.manual_seed(7)
        bundle = SamplingBundle(
            selection=sampling,
            percentile_table=torch.zeros(
                N_SAE_FEATURES, 1, dtype=torch.float64,
            ),
            rng=rng,
        )
        cfg = SelfCondConfig(
            self_cond_segment_size=PositiveInt.parse(4),
            pool_mode=FeaturePredLossMode.MAX_POOL,
            selection=bundle,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
            num_buckets=NUM_BUCKETS,
            n_sae_features=N_SAE_FEATURES,
        )
        provider = _make_provider(tiny_model_with_fp, cfg)
        state = build_test_self_cond_state(provider)
        fp_logits = torch.randn(1, 4, N_SAE_FEATURES)

        new_state = provider.step(state, fp_logits)

        assert (
            new_state.encoder_out.layout.n_segments
            == state.encoder_out.layout.n_segments + 1
        )
        assert _segment_count(
            new_state.encoder_out,
            new_state.encoder_out.layout.n_segments - 1,
        ) >= 1


# ---------------------------------------------------------------------------
# seed_from_prompt
# ---------------------------------------------------------------------------


class TestSeedFromPrompt:
    def test_appends_one_slot(self, tiny_model_with_fp, basic_config):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        # n_prompt = 32
        prompt_fp_logits = torch.randn(1, 32, basic_config.n_sae_features)
        new_state = provider.seed_from_prompt(state, prompt_fp_logits)
        assert new_state.encoder_out.layout.n_segments == 2
        assert new_state.target_slot_idx == 1

    def test_uses_last_K_positions(self, tiny_model_with_fp, basic_config):
        # Pool mode is last_token, so the seed should match a step() called
        # with prompt's last K positions only.
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        K = basic_config.self_cond_segment_size
        prompt_fp_logits = torch.randn(1, 32, basic_config.n_sae_features)
        new_state = provider.seed_from_prompt(state, prompt_fp_logits)
        # And via step() on the last K positions of the prompt:
        provider2 = _make_provider(tiny_model_with_fp, basic_config)
        state2 = build_test_self_cond_state(provider2)
        new_state2 = provider2.step(state2, prompt_fp_logits[:, -K:, :])
        # The encoder outputs should match (deterministic given same fp_logits)
        assert torch.allclose(
            _segment_values(new_state.encoder_out, 1),
            _segment_values(new_state2.encoder_out, 1),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Encoder segment isolation
# ---------------------------------------------------------------------------


class TestEncoderSegmentIsolation:
    """Standalone and joint calls are close; append preserves packed owners."""

    def test_streamed_and_joint_encoding_are_numerically_close(
        self,
        tiny_model_with_fp,
    ):
        # Build two segments worth of (fids, mags) by hand.
        torch.manual_seed(7)
        fids = torch.tensor([[[1, 2, 3, 0, 0, 0, 0, 0],
                              [4, 5, 0, 0, 0, 0, 0, 0]]], dtype=torch.long)
        mags = torch.tensor([[[2, 1, 0, 0, 0, 0, 0, 0],
                              [3, 2, 0, 0, 0, 0, 0, 0]]], dtype=torch.long)
        mask = torch.tensor([[[True, True, True, False, False, False, False, False],
                              [True, True, False, False, False, False, False, False]]])

        # Batched
        enc_batched = tiny_model_with_fp.encoder(fids, mags, mask)

        # Streaming
        enc_seg0 = tiny_model_with_fp.encoder(
            fids[:, 0:1, :], mags[:, 0:1, :], mask[:, 0:1, :],
        )
        enc_seg1 = tiny_model_with_fp.encoder(
            fids[:, 1:2, :], mags[:, 1:2, :], mask[:, 1:2, :],
        )
        enc_streamed = append_packed_encoder_segment(enc_seg0, enc_seg1)

        torch.testing.assert_close(
            enc_batched.values,
            enc_streamed.values,
            rtol=1e-5,
            atol=1e-5,
        )
        assert torch.equal(enc_batched.layout.valid_mask, enc_streamed.layout.valid_mask)
        assert torch.equal(enc_batched.layout.flat_indices, enc_streamed.layout.flat_indices)
        assert torch.equal(enc_batched.layout.batch_indices, enc_streamed.layout.batch_indices)
        assert torch.equal(
            enc_batched.layout.segment_indices,
            enc_streamed.layout.segment_indices,
        )


# ---------------------------------------------------------------------------
# Property invariants
# ---------------------------------------------------------------------------


class TestSelfCondPropertyInvariants:
    def test_target_slot_idx_equals_n_seg_minus_1(
        self, tiny_model_with_fp, basic_config
    ):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        for _ in range(5):
            fp_logits_tail = torch.randn(1, 4, basic_config.n_sae_features)
            state = provider.step(state, fp_logits_tail)
            assert state.target_slot_idx == state.encoder_out.layout.n_segments - 1

    def test_n_seg_grows_monotonically(self, tiny_model_with_fp, basic_config):
        provider = _make_provider(tiny_model_with_fp, basic_config)
        state = build_test_self_cond_state(provider)
        prev_n_seg = state.encoder_out.layout.n_segments
        for _ in range(5):
            fp_logits_tail = torch.randn(1, 4, basic_config.n_sae_features)
            state = provider.step(state, fp_logits_tail)
            assert state.encoder_out.layout.n_segments == prev_n_seg + 1
            prev_n_seg = state.encoder_out.layout.n_segments
