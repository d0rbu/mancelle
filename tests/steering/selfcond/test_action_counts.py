"""T-U1/T-U2: action-count sampling, seeding, and checkpoint inheritance."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch as t

from errors import SelfCondTrainingError
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    VAL_ROLLOUT_STEP_SENTINEL,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    resolve_selfcond_count_configs,
    sample_action_counts,
    selfcond_step_seed,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
    sample_k_batched,
)


def _config(
    *,
    positive: SamplingSelection | None = None,
    negative: NegativeSamplingSelection | None = None,
) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=positive
        or SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
            min_k=1,
            max_k=9,
            mode=SamplingMode.top_k_percentile,
        ),
        negative_counts=negative
        or NegativeSamplingSelection(
            distribution=Distribution.normal,
            mean=3.0,
            std=1.5,
            min_k=0,
            max_k=7,
        ),
        seed=37,
    )


def test_action_counts_match_sampler_draw_for_draw_and_pin_draw_order() -> None:
    """T-U1/R5: positives consume the RNG first, then negatives."""
    cfg = _config()
    actual_rng = t.Generator().manual_seed(12345)
    replay_rng = t.Generator().manual_seed(12345)

    actual_positive, actual_negative = sample_action_counts(cfg, actual_rng, 64)
    expected_positive = sample_k_batched(
        cfg.positive_counts.distribution,
        cfg.positive_counts.mean,
        cfg.positive_counts.std,
        replay_rng,
        n=64,
        min_k=cfg.positive_counts.min_k,
        max_k=cfg.positive_counts.max_k,
    )
    expected_negative = sample_k_batched(
        cfg.negative_counts.distribution,
        cfg.negative_counts.mean,
        cfg.negative_counts.std,
        replay_rng,
        n=64,
        min_k=cfg.negative_counts.min_k,
        max_k=cfg.negative_counts.max_k,
    )

    assert t.equal(actual_positive, expected_positive)
    assert t.equal(actual_negative, expected_negative)
    assert t.equal(
        t.rand(32, generator=actual_rng),
        t.rand(32, generator=replay_rng),
    )


def test_action_counts_honor_inclusive_min_and_exclusive_max() -> None:
    cfg = _config(
        positive=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=4,
            max_k=7,
        ),
        negative=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=4.0,
            std=4.0,
            min_k=2,
            max_k=5,
        ),
    )
    positive, negative = sample_action_counts(cfg, t.Generator().manual_seed(9), 4096)

    assert int(positive.min()) >= 4
    assert int(positive.max()) < 7
    assert int(negative.min()) >= 2
    assert int(negative.max()) < 5
    assert 4 in positive.tolist()
    assert 2 in negative.tolist()


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            lambda: _config(
                positive=SamplingSelection(max_k=None),
            ),
            id="positive-unbounded",
        ),
        pytest.param(
            lambda: _config(
                negative=NegativeSamplingSelection(max_k=None),
            ),
            id="negative-unbounded",
        ),
    ],
)
def test_invalid_count_contracts_raise(config) -> None:
    with pytest.raises(SelfCondTrainingError):
        config()


def test_best_token_pool_mode_is_rejected_at_config_construction() -> None:
    base = _config()
    with pytest.raises(SelfCondTrainingError, match="best_token"):
        SelfCondTrainConfig(
            pool_mode=FeaturePredLossMode.BEST_TOKEN,
            train_scope=base.train_scope,
            ste_form=base.ste_form,
            positive_counts=base.positive_counts,
            negative_counts=base.negative_counts,
            seed=base.seed,
        )


def test_selfcond_step_seed_formula_and_validation_sentinel_are_golden() -> None:
    assert selfcond_step_seed(7, 3, 11) == 7_000_045_000_083
    assert selfcond_step_seed(7, 3, VAL_ROLLOUT_STEP_SENTINEL) == 7_002_192_483_719
    assert 0 <= selfcond_step_seed(2**63 - 1, 2**31, 2**31) < 2**63


def test_count_inheritance_from_sampling_checkpoint_is_exact() -> None:
    feature_selection = {
        "distribution": "normal",
        "mean": 7.0,
        "std": 2.5,
        "min_k": 2,
        "max_k": 19,
        "mode": "top_k_percentile",
        "delta_weights": None,
    }
    negative_selection = {
        "distribution": "geometric",
        "mean": 4.0,
        "std": 4.0,
        "min_k": 0,
        "max_k": 13,
    }

    positive, negative = resolve_selfcond_count_configs(
        feature_selection=feature_selection,
        negative_selection=negative_selection,
        positive_override=None,
        negative_override=None,
    )

    assert asdict(positive) == {
        "distribution": Distribution.normal,
        "mean": 7.0,
        "std": 2.5,
        "delta_weights": None,
        "min_k": 2,
        "max_k": 19,
        "mode": SamplingMode.top_k_percentile,
        "gradient": None,
    }
    assert asdict(negative) == {
        "distribution": Distribution.geometric,
        "mean": 4.0,
        "std": 4.0,
        "min_k": 0,
        "max_k": 13,
    }


def test_threshold_checkpoint_without_explicit_positive_counts_raises_loudly() -> None:
    """No graceful fallback (owner directive): a threshold-trained checkpoint
    has NO positive count distribution to inherit, so resolving without an
    explicit ``selfcond.positive_counts`` must raise with the remedy — never
    silently substitute a default."""
    with pytest.raises(SelfCondTrainingError, match=r"(?s)threshold-based.*positive_counts"):
        resolve_selfcond_count_configs(
            feature_selection={"threshold_type": "segment_topk", "threshold": 8},
            negative_selection={
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
                "min_k": 0,
                "max_k": 12,
            },
            positive_override=None,
            negative_override=None,
        )


def test_explicit_count_overrides_win() -> None:
    positive, negative = resolve_selfcond_count_configs(
        feature_selection={"threshold_type": "segment_topk", "threshold": 8},
        negative_selection={
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
            "min_k": 0,
            "max_k": 12,
        },
        positive_override={
            "distribution": "geometric",
            "mean": 6.0,
            "std": 6.0,
            "min_k": 3,
            "max_k": 17,
        },
        negative_override={
            "distribution": "normal",
            "mean": 2.0,
            "std": 0.5,
            "min_k": 1,
            "max_k": 5,
        },
    )

    assert (positive.mean, positive.min_k, positive.max_k) == (6.0, 3, 17)
    assert (negative.distribution, negative.mean, negative.max_k) == (
        Distribution.normal,
        2.0,
        5,
    )


def test_missing_checkpoint_negative_selection_raises_even_with_override() -> None:
    with pytest.raises(SelfCondTrainingError, match="negative_selection"):
        resolve_selfcond_count_configs(
            feature_selection={"threshold_type": "segment_topk", "threshold": 8},
            negative_selection=None,
            positive_override=None,
            negative_override={
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
                "min_k": 0,
                "max_k": 12,
            },
        )
