"""Tripwires for every fail-loud guard branch in selfcond_training.

The numerical happy paths of this module are pinned by independent oracles
elsewhere in the suite; this file exists so that no raise-guard — the branches
the fail-loud doctrine depends on — can be silently weakened or dropped
(pre-merge review finding: essentially every guard branch lacked a direct
regression test). Each test constructs the exact malformed input its guard
rejects and demands the typed error.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.conditional.encoder_decoder.selfcond_training as selfcond_training_module
from errors import SelfCondTrainingError
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    MutableConditioning,
    SelectedActions,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    build_head_only_optimizer,
    build_selfcond_state,
    continuation_mask,
    freeze_for_head_only,
    materialize_head_zero_grads,
    resolve_recorded_num_buckets,
    resolve_selfcond_count_configs,
    select_topk_bottoml,
    validate_selfcond_prerequisites,
    write_predicted_slot,
)
from steering.conditional.encoder_decoder.selfcond_training import (
    pool_fp_logits_per_segment as _pool_fp_logits_per_segment,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
)
from steering.refined_types import PositiveInt
from tests.steering.selfcond.helpers import tiny_head_model


def pool_fp_logits_per_segment(
    fp_logits: t.Tensor,
    seg_map: t.Tensor,
    n_seg: int,
    mode: FeaturePredLossMode,
) -> t.Tensor:
    """Pass malformed values through the production beartype boundary."""
    return _pool_fp_logits_per_segment(
        fp_logits,
        seg_map,
        cast(PositiveInt, n_seg),
        mode,
    )


def _sampling(**overrides) -> SamplingSelection:
    values = {
        "distribution": Distribution.geometric,
        "mean": 5.0,
        "std": 5.0,
        "min_k": 1,
        "max_k": 9,
        "mode": SamplingMode.top_k_percentile,
    }
    values.update(overrides)
    return SamplingSelection(**values)  # ty: ignore[invalid-argument-type]


def _negative(**overrides) -> NegativeSamplingSelection:
    values = {
        "distribution": Distribution.normal,
        "mean": 3.0,
        "std": 1.5,
        "min_k": 0,
        "max_k": 7,
    }
    values.update(overrides)
    return NegativeSamplingSelection(**values)


def _cfg(
    positive: SamplingSelection | None = None,
    negative: NegativeSamplingSelection | None = None,
) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=positive or _sampling(),
        negative_counts=negative or _negative(),
        seed=11,
    )


# ---------------------------------------------------------------- count config


@pytest.mark.parametrize(
    ("positive", "negative", "match"),
    [
        (_sampling(max_k=None), None, "positive_counts.max_k must be non-null"),
        (None, _negative(max_k=None), "negative_counts.max_k must be non-null"),
    ],
)
def test_count_config_guards_raise(positive, negative, match) -> None:
    with pytest.raises(SelfCondTrainingError, match=match):
        _cfg(positive=positive, negative=negative)


# ------------------------------------------------- resolve_recorded_num_buckets


def test_recorded_num_buckets_rejects_non_numeric_string_with_typed_error() -> None:
    """int("two") must not escape as a bare ValueError (review finding)."""
    with pytest.raises(SelfCondTrainingError, match="not an integer"):
        resolve_recorded_num_buckets({"user_config": {"num_buckets": "two"}})


def test_recorded_num_buckets_refuses_to_truncate_non_integral_float() -> None:
    """int(2.9) == 2 would silently accept corrupted metadata (review finding)."""
    with pytest.raises(SelfCondTrainingError, match="not integral"):
        resolve_recorded_num_buckets({"user_config": {"num_buckets": 2.9}})


def test_recorded_num_buckets_accepts_integral_spellings_and_top_level_fallback() -> None:
    assert resolve_recorded_num_buckets({"user_config": {"num_buckets": "2"}}) == 2
    assert resolve_recorded_num_buckets({"user_config": {"num_buckets": 2.0}}) == 2
    # _meta_value falls through to the top level when user_config is absent.
    assert resolve_recorded_num_buckets({"num_buckets": 2}) == 2


def test_recorded_num_buckets_rejects_container_types() -> None:
    with pytest.raises(SelfCondTrainingError, match="unusable type"):
        resolve_recorded_num_buckets({"user_config": {"num_buckets": [2]}})


# -------------------------------------------- resolve_selfcond_count_configs


def test_threshold_override_is_rejected_as_positive_counts() -> None:
    with pytest.raises(SelfCondTrainingError, match="sampling distribution"):
        resolve_selfcond_count_configs(
            feature_selection={"threshold_type": "segment_topk", "threshold": 3},
            negative_selection={
                "distribution": "normal",
                "mean": 3.0,
                "std": 1.5,
                "min_k": 0,
                "max_k": 7,
            },
            positive_override={"threshold_type": "segment_topk", "threshold": 3},
            negative_override=None,
        )


def test_inherited_non_sampling_parse_raises_typed_error(monkeypatch) -> None:
    """Defensive branch: a corrupted meta whose feature_selection parses to
    None must produce a typed error naming the problem, not a downstream
    AttributeError (review finding)."""
    monkeypatch.setattr(selfcond_training_module, "parse_feature_selection", lambda raw: None)
    with pytest.raises(SelfCondTrainingError, match="did not parse to a sampling"):
        resolve_selfcond_count_configs(
            feature_selection={"anything": True},
            negative_selection={
                "distribution": "normal",
                "mean": 3.0,
                "std": 1.5,
                "min_k": 0,
                "max_k": 7,
            },
            positive_override=None,
            negative_override=None,
        )


# ------------------------------------------------- pool_fp_logits_per_segment


def test_pooling_guards_raise() -> None:
    good = t.zeros(2, 4, 3)
    seg_map = t.zeros(2, 4, dtype=t.int64)
    with pytest.raises(SelfCondTrainingError, match=r"\(B, T, V\)"):
        pool_fp_logits_per_segment(t.zeros(2, 4), seg_map, 2, FeaturePredLossMode.MAX_POOL)
    with pytest.raises(SelfCondTrainingError, match="seg_map shape"):
        pool_fp_logits_per_segment(
            good, t.zeros(2, 3, dtype=t.int64), 2, FeaturePredLossMode.MAX_POOL
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="n_seg"):
        pool_fp_logits_per_segment(good, seg_map, 0, FeaturePredLossMode.MAX_POOL)
    with pytest.raises(SelfCondTrainingError, match="at least one token"):
        pool_fp_logits_per_segment(
            t.zeros(2, 0, 3), t.zeros(2, 0, dtype=t.int64), 2, FeaturePredLossMode.MAX_POOL
        )


# ------------------------------------------------------------ continuation_mask


def test_continuation_mask_guards_and_empty_slot_axis() -> None:
    with pytest.raises(SelfCondTrainingError, match="rank-2"):
        continuation_mask(t.zeros(4, dtype=t.int64), t.zeros(1, 2, dtype=t.int64))
    with pytest.raises(SelfCondTrainingError, match="batch dimensions"):
        continuation_mask(t.zeros(2, 4, dtype=t.int64), t.zeros(1, 2, dtype=t.int64))
    with pytest.raises(SelfCondTrainingError, match="int64"):
        continuation_mask(t.zeros(1, 4, dtype=t.int32), t.zeros(1, 2, dtype=t.int64))
    with pytest.raises(SelfCondTrainingError, match=r"lie in \[0, 2\)"):
        continuation_mask(t.full((1, 4), 5, dtype=t.int64), t.zeros(1, 2, dtype=t.int64))
    empty = continuation_mask(t.zeros(2, 4, dtype=t.int64), t.zeros(2, 0, dtype=t.int64))
    assert empty.shape == (2, 0) and empty.dtype == t.bool


# ---------------------------------------------------------- build_selfcond_state


def _raw_batch(**overrides) -> SimpleNamespace:
    values = {
        "enc_fids": t.zeros(2, 3, 4, dtype=t.int64),
        "enc_mags": t.zeros(2, 3, 4, dtype=t.int64),
        "enc_mask": t.zeros(2, 3, 4, dtype=t.bool),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_state_guards_raise() -> None:
    with pytest.raises(SelfCondTrainingError, match=r"\(B, n_seg, F\)"):
        build_selfcond_state(_raw_batch(enc_fids=t.zeros(2, 3, dtype=t.int64)), 4)
    with pytest.raises(SelfCondTrainingError, match="identical shapes"):
        build_selfcond_state(_raw_batch(enc_mags=t.zeros(2, 3, 5, dtype=t.int64)), 4)
    with pytest.raises(SelfCondTrainingError, match="at least the batch width"):
        build_selfcond_state(_raw_batch(), 3)
    with pytest.raises(SelfCondTrainingError, match="int64"):
        build_selfcond_state(_raw_batch(enc_mags=t.zeros(2, 3, 4, dtype=t.int32)), 4)
    with pytest.raises(SelfCondTrainingError, match="bool"):
        build_selfcond_state(_raw_batch(enc_mask=t.zeros(2, 3, 4, dtype=t.int64)), 4)


# ---------------------------------------------------------- write_predicted_slot


def _state(width: int = 4) -> MutableConditioning:
    return MutableConditioning(
        fids=t.zeros(2, 3, width, dtype=t.int64),
        mags=t.zeros(2, 3, width, dtype=t.int64),
        mask=t.zeros(2, 3, width, dtype=t.bool),
        sign=t.zeros(2, 3, width, dtype=t.float32),
    )


def _actions(
    pos_width: int = 2,
    neg_width: int = 2,
    *,
    k: list[int] | None = None,
    l: list[int] | None = None,  # noqa: E741 - spec's negative-count symbol
    pos_fill: int = 1,
    neg_fill: int = 2,
) -> SelectedActions:
    return SelectedActions(
        pos_ids=t.full((2, pos_width), pos_fill, dtype=t.int64),
        neg_ids=t.full((2, neg_width), neg_fill, dtype=t.int64),
        k=t.tensor(k if k is not None else [1, 1], dtype=t.int64),
        l=t.tensor(l if l is not None else [1, 1], dtype=t.int64),
    )


LIVE = t.tensor([True, True])


def test_write_predicted_slot_guards_raise() -> None:
    mismatched = _state()
    mismatched.sign = t.zeros(2, 3, 5, dtype=t.float32)
    with pytest.raises(SelfCondTrainingError, match="identical shapes"):
        write_predicted_slot(mismatched, 1, _actions(), LIVE)

    flat = MutableConditioning(
        fids=t.zeros(2, 3, dtype=t.int64),
        mags=t.zeros(2, 3, dtype=t.int64),
        mask=t.zeros(2, 3, dtype=t.bool),
        sign=t.zeros(2, 3, dtype=t.float32),
    )
    with pytest.raises(SelfCondTrainingError, match="rank 3"):
        write_predicted_slot(flat, 1, _actions(), LIVE)

    with pytest.raises(SelfCondTrainingError, match=r"outside \[0, 3\)"):
        write_predicted_slot(_state(), 3, _actions(), LIVE)
    with pytest.raises(SelfCondTrainingError, match="live must be a bool tensor"):
        write_predicted_slot(_state(), 1, _actions(), t.tensor([True, False, True]))
    with pytest.raises(SelfCondTrainingError, match="live must be a bool tensor"):
        write_predicted_slot(_state(), 1, _actions(), t.tensor([1, 0]))
    with pytest.raises(SelfCondTrainingError, match="one value per batch row"):
        write_predicted_slot(_state(), 1, _actions(k=[1, 1, 1]), LIVE)
    bad_rows = SelectedActions(
        pos_ids=t.ones(3, 2, dtype=t.int64),
        neg_ids=t.full((2, 2), 2, dtype=t.int64),
        k=t.tensor([1, 1], dtype=t.int64),
        l=t.tensor([1, 1], dtype=t.int64),
    )
    with pytest.raises(SelfCondTrainingError, match="one row per batch row"):
        write_predicted_slot(_state(), 1, bad_rows, LIVE)
    with pytest.raises(SelfCondTrainingError, match="nonnegative"):
        write_predicted_slot(_state(), 1, _actions(k=[-1, 1]), LIVE)
    with pytest.raises(SelfCondTrainingError, match="exceed selected-ID tensor widths"):
        write_predicted_slot(_state(), 1, _actions(k=[3, 1]), LIVE)
    with pytest.raises(SelfCondTrainingError, match="more columns than F'"):
        write_predicted_slot(_state(width=3), 1, _actions(k=[2, 2], l=[2, 2]), LIVE)
    with pytest.raises(SelfCondTrainingError, match="IDs must be nonnegative"):
        write_predicted_slot(_state(), 1, _actions(pos_fill=-1), LIVE)


def test_write_predicted_slot_ignores_garbage_on_non_live_rows() -> None:
    """Validation is deliberately live-gated: a non-live row may carry
    arbitrary garbage counts/IDs (the rollout draws them for the whole batch
    and discards them), and the write must neither raise nor touch the row."""
    state = _state()
    before = state.fids.clone()
    actions = _actions(k=[1, -7], pos_fill=1)
    actions.neg_ids[1, 0] = -9
    write_predicted_slot(state, 1, actions, t.tensor([True, False]))
    assert t.equal(state.fids[1], before[1])
    assert bool(state.mask[0, 1, 0])


# ----------------------------------------------------------- select_topk_bottoml


def test_select_topk_bottoml_guards_raise_and_empty_batch_is_wellformed() -> None:
    scores = t.rand(2, 5)
    k = t.tensor([1, 2], dtype=t.int64)
    l = t.tensor([1, 1], dtype=t.int64)  # noqa: E741 - spec symbol
    with pytest.raises(SelfCondTrainingError, match=r"\(B, V\)"):
        select_topk_bottoml(t.rand(5), k, l)
    with pytest.raises(SelfCondTrainingError, match="must both have shape"):
        select_topk_bottoml(scores, t.tensor([1], dtype=t.int64), l)
    with pytest.raises(SelfCondTrainingError, match="int64"):
        select_topk_bottoml(scores, k.int(), l)
    with pytest.raises(SelfCondTrainingError, match="nonnegative k/l"):
        select_topk_bottoml(scores, t.tensor([-1, 0], dtype=t.int64), l)
    with pytest.raises(SelfCondTrainingError, match="nonnegative k/l"):
        select_topk_bottoml(
            scores, t.tensor([3, 0], dtype=t.int64), t.tensor([3, 0], dtype=t.int64)
        )

    empty = select_topk_bottoml(t.zeros(0, 5), t.zeros(0, dtype=t.int64), t.zeros(0, dtype=t.int64))
    assert empty.pos_ids.shape[0] == 0 and empty.neg_ids.shape[0] == 0


# --------------------------------------------------- head-only helper guards


def test_headless_model_raises_in_every_head_only_helper() -> None:
    headless = SimpleNamespace(feature_pred_head=None)
    with pytest.raises(SelfCondTrainingError, match="requires a feature-prediction head"):
        freeze_for_head_only(headless)
    with pytest.raises(SelfCondTrainingError, match="requires a feature-prediction head"):
        materialize_head_zero_grads(headless)
    with pytest.raises(SelfCondTrainingError, match="requires a feature-prediction head"):
        build_head_only_optimizer(headless, unembedding_lr=0.004, adam_betas=(0.8, 0.95))


# ------------------------------------------- model-side prerequisite checks


def _valid_meta() -> dict:
    return {
        "user_config": {
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "max_pool",
                "loss_fn": "bce",
            },
            "num_buckets": 2,
            "negative_selection": {
                "distribution": "normal",
                "mean": 3.0,
                "std": 1.5,
                "min_k": 0,
                "max_k": 7,
            },
        }
    }


def test_prerequisites_pass_on_a_valid_model_and_meta() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32)
    validate_selfcond_prerequisites(model, _valid_meta(), _cfg(), 32)


def test_prerequisites_accept_legacy_top_level_feature_prediction_config() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32)
    meta = {
        "feature_prediction_config": {"weight": 0.1, "layer": -1, "loss_mode": "max_pool"},
        "num_buckets": 2,
        "negative_selection": {
            "distribution": "normal",
            "mean": 3.0,
            "std": 1.5,
            "min_k": 0,
            "max_k": 7,
        },
    }
    validate_selfcond_prerequisites(model, meta, _cfg(), 32)


def test_c1_rejects_head_width_disagreeing_with_feature_universe() -> None:
    """Only the metadata side of C1/C3 was pinned before; the loaded MODEL
    disagreeing with the metadata that describes it must also raise
    (review finding)."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32)
    with pytest.raises(SelfCondTrainingError, match="C1.*32 != 16"):
        validate_selfcond_prerequisites(model, _valid_meta(), _cfg(), 16)


def test_c3_rejects_model_num_buckets_disagreeing_with_metadata() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32, num_buckets=3)
    with pytest.raises(SelfCondTrainingError, match="C3.*metadata=2, model=3"):
        validate_selfcond_prerequisites(model, _valid_meta(), _cfg(), 32)


def test_c2_rejects_live_model_feature_mask() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=32)
    model.feature_mask = object()  # ty: ignore[invalid-assignment]
    with pytest.raises(NotImplementedError, match="C2"):
        validate_selfcond_prerequisites(model, _valid_meta(), _cfg(), 32)
