"""Exact blend semantics for chat steering policies."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from scripts.chat_tui.steering import (
    SignalSource,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)
from scripts.chat_tui.steering import (
    build_score_policy as _build_score_policy,
)
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
    SelfCondScoreMasking,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.eval.axbench_upstream.methods import self_conditioning_masked_pool_limit
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import build_tiny_model

_DENSE_FEATURES = 8
_CPU_DEVICE = t.device("cpu")


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _negative_selection(
    *,
    mean: float = 2.0,
    min_k: int = 0,
) -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=mean,
        std=mean,
        min_k=min_k,
        max_k=4,
    )


def _positive_selection(*, mean: float = 2.0, min_k: int = 1) -> SamplingSelection:
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=mean,
        std=mean,
        min_k=min_k,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )


def build_score_policy(*args: Any, **kwargs: Any):
    """Inject the shared positive count policy used by these blend tests."""

    kwargs.setdefault("positive_selection", _positive_selection())
    return _build_score_policy(*args, **kwargs)


def _native_rank_policy(scores: t.Tensor, strength: float) -> SelfCondScoreMasking:
    limit = self_conditioning_masked_pool_limit(
        strength,
        n_candidates=int(scores.numel()),
        mean=2.0,
    )
    return SelfCondScoreMasking(
        target_scores=scores,
        negative_selection=_negative_selection(),
        positive_pool_limit=PositiveInt.parse(limit),
        negative_pool_limit=PositiveInt.parse(limit),
    )


def _signal(name: str, scores: t.Tensor) -> SteeringSignal:
    return SteeringSignal(
        name=name,
        target_scores=scores,
        source=SignalSource.AXBENCH_CONCEPT,
        identity=f"signal/{name}",
    )


def _selection(
    method: SteeringMethod,
    *enabled: tuple[SteeringSignal, float],
) -> SteeringSelection:
    return SteeringSelection(method=method, enabled=enabled)


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


def _ranked_rows(provider: SelfConditionedFeatureProvider) -> tuple[t.Tensor, t.Tensor]:
    positive = provider._masking_positive_rows
    negative = provider._masking_negative_rows
    assert positive is not None
    assert negative is not None
    return positive, negative


@pytest.mark.parametrize(
    "factor",
    (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
)
def test_single_concept_interpolation_is_bit_identical(factor: float) -> None:
    scores = t.tensor(
        [-1.0, -0.75, -0.0, 0.0, 0.125, 0.5, 0.875, 1.0],
        dtype=t.float32,
    )

    policy = build_score_policy(
        _selection(SteeringMethod.INTERPOLATION, (_signal("alpha", scores), factor)),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )

    assert isinstance(policy, SelfCondScoreInterpolation)
    assert policy.target_scores.dtype is t.float32
    assert t.equal(policy.target_scores, scores)
    assert float(policy.weight) == factor / 6.0


def test_multi_concept_interpolation_accumulates_in_float64_once() -> None:
    first = t.tensor([0.12345679, 0.33333334, -0.7777778], dtype=t.float32)
    second = t.tensor([0.7654321, -0.22222222, 0.11111111], dtype=t.float32)
    first_strength = 0.2
    second_strength = 5.8
    total = math.fsum((first_strength, second_strength))

    policy = build_score_policy(
        _selection(
            SteeringMethod.INTERPOLATION,
            (_signal("alpha", first), first_strength),
            (_signal("beta", second), second_strength),
        ),
        negative_selection=_negative_selection(),
        head_width=3,
    )
    expected = (
        (first.to(t.float64) * first_strength + second.to(t.float64) * second_strength) / total
    ).to(t.float32)
    naive_float32 = (first * first_strength + second * second_strength) / total

    assert isinstance(policy, SelfCondScoreInterpolation)
    assert t.equal(policy.target_scores, expected)
    assert not t.equal(naive_float32, expected)


def test_interpolation_total_six_canonicalizes_weight_to_one() -> None:
    scores = t.linspace(-1.0, 1.0, _DENSE_FEATURES)
    policy = build_score_policy(
        _selection(
            SteeringMethod.INTERPOLATION,
            (_signal("alpha", scores), 3.0),
            (_signal("beta", scores.flip(0)), 3.0),
        ),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )

    assert isinstance(policy, SelfCondScoreInterpolation)
    assert float(policy.weight) == 1.0


def test_interpolation_over_budget_lists_every_signal_and_total() -> None:
    scores = t.linspace(-1.0, 1.0, _DENSE_FEATURES)
    selection = _selection(
        SteeringMethod.INTERPOLATION,
        (_signal("alpha", scores), 3.0),
        (_signal("beta", scores.flip(0)), 3.000000002),
    )

    with pytest.raises(
        ValueError,
        match=r"alpha.*3\.0.*beta.*3\.000000002.*total.*6\.000000002",
    ):
        build_score_policy(
            selection,
            negative_selection=_negative_selection(),
            head_width=_DENSE_FEATURES,
        )


@pytest.mark.parametrize(
    "selection",
    (
        _selection(
            SteeringMethod.OFF,
            (_signal("ignored", t.ones(_DENSE_FEATURES)), 6.0),
        ),
        _selection(SteeringMethod.INTERPOLATION),
        _selection(
            SteeringMethod.INTERPOLATION,
            (_signal("zero", t.ones(_DENSE_FEATURES)), 0.0),
        ),
    ),
)
def test_off_empty_and_zero_total_use_exact_zero_baseline(
    selection: SteeringSelection,
) -> None:
    policy = build_score_policy(
        selection,
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )

    assert isinstance(policy, SelfCondScoreInterpolation)
    assert t.equal(policy.target_scores, t.zeros(_DENSE_FEATURES, dtype=t.float32))
    assert float(policy.weight) == 0.0


def test_width_mismatch_names_signal_width_and_head_width() -> None:
    selection = _selection(
        SteeringMethod.INTERPOLATION,
        (_signal("narrow", t.zeros(7)), 1.0),
    )

    with pytest.raises(ValueError, match=r"narrow.*width 7.*head width.*8"):
        build_score_policy(
            selection,
            negative_selection=_negative_selection(),
            head_width=_DENSE_FEATURES,
        )


@pytest.mark.parametrize("strength", (0.0, 1.0, 3.5))
def test_restriction_pools_equal_native_stable_rank(strength: float) -> None:
    scores = t.tensor(
        [
            -1.0,
            -0.35,
            -0.0,
            0.0,
            0.35,
            0.5,
            -0.5,
            1.0,
        ],
        dtype=t.float32,
    )
    native_policy = _native_rank_policy(scores, strength)
    synthesized_policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            (_signal("alpha", scores), strength),
        ),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )

    native = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(native_policy), _CPU_DEVICE
    )
    synthesized = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(synthesized_policy), _CPU_DEVICE
    )

    native_positive, native_negative = _ranked_rows(native)
    synthesized_positive, synthesized_negative = _ranked_rows(synthesized)
    assert t.equal(native_positive, synthesized_positive)
    assert t.equal(native_negative, synthesized_negative)
    assert not bool(synthesized_positive[2])
    assert not bool(synthesized_negative[2])
    assert not bool(synthesized_positive[3])
    assert not bool(synthesized_negative[3])


def test_two_concept_restriction_pools_equal_native_pool_intersection() -> None:
    first = t.tensor([-1.0, -0.4, 0.0, 0.1, 0.3, 0.8, -0.7, 1.0])
    second = t.tensor([-0.9, 0.2, 0.0, 0.4, -0.1, 0.6, -0.8, 1.0])
    first_strength = 2.0
    second_strength = 3.0
    first_native = SelfConditionedFeatureProvider(
        _dense_model(),
        _dense_config(
            _native_rank_policy(first, first_strength)
        ),
        _CPU_DEVICE,
    )
    second_native = SelfConditionedFeatureProvider(
        _dense_model(),
        _dense_config(
            _native_rank_policy(second, second_strength)
        ),
        _CPU_DEVICE,
    )
    synthesized_policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            (_signal("alpha", first), first_strength),
            (_signal("beta", second), second_strength),
        ),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )
    synthesized = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(synthesized_policy), _CPU_DEVICE
    )

    first_positive, first_negative = _ranked_rows(first_native)
    second_positive, second_negative = _ranked_rows(second_native)
    synthesized_positive, synthesized_negative = _ranked_rows(synthesized)
    expected_positive = first_positive & second_positive
    expected_negative = first_negative & second_negative
    assert t.equal(synthesized_positive, expected_positive)
    assert t.equal(synthesized_negative, expected_negative)


def test_three_concept_restriction_pools_equal_native_pool_intersection() -> None:
    scores_and_strengths = (
        (t.tensor([-1.0, -0.4, 0.0, 0.1, 0.3, 0.8, -0.7, 1.0]), 1.0),
        (t.tensor([-0.9, 0.2, 0.0, 0.4, -0.1, 0.6, -0.8, 1.0]), 3.0),
        (t.tensor([-0.2, 0.0, 0.0, -0.1, 0.3, 0.4, -0.9, 0.8]), 5.0),
    )
    native_providers = tuple(
        SelfConditionedFeatureProvider(
            _dense_model(),
            _dense_config(_native_rank_policy(scores, strength)),
            _CPU_DEVICE,
        )
        for scores, strength in scores_and_strengths
    )
    synthesized_policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            *(
                (_signal(name, scores), strength)
                for name, (scores, strength) in zip(
                    ("alpha", "beta", "gamma"), scores_and_strengths, strict=True
                )
            ),
        ),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )
    synthesized = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(synthesized_policy), _CPU_DEVICE
    )

    native_rows = tuple(_ranked_rows(provider) for provider in native_providers)
    expected_positive = native_rows[0][0].clone()
    expected_negative = native_rows[0][1].clone()
    for provider in native_providers[1:]:
        provider_positive, provider_negative = _ranked_rows(provider)
        expected_positive &= provider_positive
        expected_negative &= provider_negative
    synthesized_positive, synthesized_negative = _ranked_rows(synthesized)
    assert t.equal(synthesized_positive, expected_positive)
    assert t.equal(synthesized_negative, expected_negative)


def test_strength_zero_in_restriction_blend_contributes_sign_only_pool() -> None:
    sign_only = t.tensor([-1.0, -0.2, 0.0, 0.4, -0.1, 0.7, -0.8, 1.0])
    ranked = t.tensor([-0.9, 0.5, 0.0, 0.6, -0.4, 0.8, -0.7, 1.0])
    other_strength = 3.0
    sign_native = SelfConditionedFeatureProvider(
        _dense_model(),
        _dense_config(
            _native_rank_policy(sign_only, 0.0)
        ),
        _CPU_DEVICE,
    )
    other_native = SelfConditionedFeatureProvider(
        _dense_model(),
        _dense_config(
            _native_rank_policy(ranked, other_strength)
        ),
        _CPU_DEVICE,
    )
    synthesized_policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            (_signal("sign-only", sign_only), 0.0),
            (_signal("other", ranked), other_strength),
        ),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )
    synthesized = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(synthesized_policy), _CPU_DEVICE
    )

    sign_positive, sign_negative = _ranked_rows(sign_native)
    other_positive, other_negative = _ranked_rows(other_native)
    synthesized_positive, synthesized_negative = _ranked_rows(synthesized)
    expected_positive = sign_positive & other_positive
    expected_negative = sign_negative & other_negative
    assert t.equal(synthesized_positive, expected_positive)
    assert t.equal(synthesized_negative, expected_negative)


def test_restriction_disjoint_support_fails_with_both_concept_names() -> None:
    first = t.tensor([1.0, 0.8, 0.0, 0.0, -1.0, -0.8, 0.0, 0.0])
    second = t.tensor([0.0, 0.0, 1.0, 0.8, 0.0, 0.0, -1.0, -0.8])
    selection = _selection(
        SteeringMethod.RESTRICTION,
        (_signal("alpha", first), 1.0),
        (_signal("beta", second), 1.0),
    )

    with pytest.raises(ValueError, match=r"chat TUI restriction blend \[alpha, beta\]"):
        build_score_policy(
            selection,
            negative_selection=_negative_selection(),
            head_width=_DENSE_FEATURES,
        )


def test_restriction_precheck_applies_feature_mask_in_original_id_order() -> None:
    scores = t.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    feature_mask = FeatureMask(FeatureMaskMode.whitelist, (101,))
    original_ids = t.arange(100, 108)

    with pytest.raises(ValueError, match=r"alpha.*positive candidate pool"):
        build_score_policy(
            _selection(
                SteeringMethod.RESTRICTION,
                (_signal("alpha", scores), 1.0),
            ),
            negative_selection=_negative_selection(),
            head_width=_DENSE_FEATURES,
            feature_mask=feature_mask,
            head_row_original_ids=original_ids,
        )


def test_restriction_uses_post_mask_n_stable_rank_and_independent_side_means() -> None:
    scores = t.tensor([1.0, 0.8, 0.7, -0.1, -0.9, -1.0])
    policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            (_signal("compact", scores), 6.0),
        ),
        positive_selection=_positive_selection(mean=0.5),
        negative_selection=_negative_selection(mean=1.5),
        head_width=6,
        feature_mask=FeatureMask(FeatureMaskMode.blacklist, (10, 60)),
        head_row_original_ids=t.tensor([10, 20, 30, 40, 50, 60]),
    )

    assert isinstance(policy, SelfCondScoreMasking)
    # The masked extrema do not consume the one positive or three negative
    # positions. Only two eligible negative rows exist, so both remain.
    assert policy.target_scores.tolist() == [0.0, 1.0, 0.0, -1.0, -1.0, 0.0]
    assert int(policy.positive_pool_limit) == 6
    assert int(policy.negative_pool_limit) == 6


def test_restriction_accepts_an_optional_empty_negative_intersection() -> None:
    policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            (_signal("positive-only", t.linspace(0.1, 0.8, _DENSE_FEATURES)), 0.0),
        ),
        positive_selection=_positive_selection(),
        negative_selection=_negative_selection(min_k=0),
        head_width=_DENSE_FEATURES,
    )

    assert isinstance(policy, SelfCondScoreMasking)
    assert policy.target_scores.tolist() == [1.0] * _DENSE_FEATURES


@pytest.mark.parametrize(
    "original_ids",
    (t.arange(7), t.arange(8).reshape(2, 4)),
)
def test_restriction_rejects_misaligned_head_row_original_ids(
    original_ids: t.Tensor,
) -> None:
    with pytest.raises(ValueError, match="align one-to-one"):
        build_score_policy(
            _selection(
                SteeringMethod.RESTRICTION,
                (_signal("alpha", t.linspace(-1.0, 1.0, _DENSE_FEATURES)), 1.0),
            ),
            negative_selection=_negative_selection(),
            head_width=_DENSE_FEATURES,
            head_row_original_ids=original_ids,
        )


def test_restriction_rejects_a_feature_mask_with_no_eligible_head_rows() -> None:
    with pytest.raises(ValueError, match="no eligible head rows"):
        build_score_policy(
            _selection(
                SteeringMethod.RESTRICTION,
                (_signal("alpha", t.linspace(-1.0, 1.0, _DENSE_FEATURES)), 1.0),
            ),
            negative_selection=_negative_selection(),
            head_width=_DENSE_FEATURES,
            feature_mask=FeatureMask(FeatureMaskMode.whitelist, (999,)),
            head_row_original_ids=t.arange(100, 108),
        )


def test_strength_zero_restriction_equals_native_sign_only_pools() -> None:
    scores = t.tensor([-1.0, -0.2, -0.0, 0.0, 0.1, 0.5, -0.7, 1.0])
    native_policy = _native_rank_policy(scores, 0.0)
    synthesized_policy = build_score_policy(
        _selection(
            SteeringMethod.RESTRICTION,
            (_signal("sign-only", scores), 0.0),
        ),
        negative_selection=_negative_selection(),
        head_width=_DENSE_FEATURES,
    )
    native = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(native_policy), _CPU_DEVICE
    )
    synthesized = SelfConditionedFeatureProvider(
        _dense_model(), _dense_config(synthesized_policy), _CPU_DEVICE
    )

    native_positive, native_negative = _ranked_rows(native)
    synthesized_positive, synthesized_negative = _ranked_rows(synthesized)
    assert t.equal(native_positive, synthesized_positive)
    assert t.equal(native_negative, synthesized_negative)


@pytest.mark.parametrize(
    "scores",
    (
        t.zeros((2, 2), dtype=t.float32),
        t.zeros(2, dtype=t.float64),
        t.tensor([0.0, float("nan")], dtype=t.float32),
        t.tensor([0.0, 1.01], dtype=t.float32),
    ),
)
def test_signal_rejects_invalid_target_scores(scores: t.Tensor) -> None:
    with pytest.raises(ValueError, match="target_scores"):
        _signal("invalid", scores)


@pytest.mark.parametrize("strength", (-0.1, 6.1, float("nan"), float("inf")))
def test_selection_rejects_invalid_strength(strength: float) -> None:
    with pytest.raises(ValueError, match="strength"):
        _selection(
            SteeringMethod.INTERPOLATION,
            (_signal("alpha", t.zeros(_DENSE_FEATURES)), strength),
        )


def test_selection_rejects_duplicate_signal_identity() -> None:
    signal = _signal("alpha", t.zeros(_DENSE_FEATURES))

    with pytest.raises(ValueError, match=r"duplicate.*signal/alpha"):
        _selection(
            SteeringMethod.INTERPOLATION,
            (signal, 1.0),
            (signal, 2.0),
        )
