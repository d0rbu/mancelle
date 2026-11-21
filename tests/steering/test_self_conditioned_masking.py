"""Sign-consistent dense-target candidate masking for SelfConditioning."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.conditional.encoder_decoder.self_conditioned as selfcond_module
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
    SelfCondDenseScorePolicy,
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
    rank_masked_candidate_pools,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    SelectedActions,
)
from steering.refined_types import PositiveInt, UnitInterval
from tests.encdec_test_helpers import build_tiny_model
from tests.steering._self_conditioned_state import build_test_self_cond_state

# Bound once at import time so helpers delegate to the real primitive even
# after earlier invocations monkeypatched the module attribute.
_ORIGINAL_SELECT_TOPK_BOTTOML = selfcond_module.select_topk_bottoml


def _positive_selection(*, max_k: int = 4, min_k: int = 1) -> SamplingSelection:
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=min_k,
        max_k=max_k,
        mode=SamplingMode.top_k_percentile,
    )


def _negative_selection(*, max_k: int = 4, min_k: int = 0) -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=min_k,
        max_k=max_k,
    )


def _model(
    *,
    feature_vocab: FeatureVocab | None = None,
    loss_mode: FeaturePredLossMode = FeaturePredLossMode.MAX_POOL,
):
    return build_tiny_model(
        n_sae_features=6,
        num_buckets=2,
        feature_vocab=feature_vocab,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=loss_mode,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )


def _masking(
    diffmean: t.Tensor,
    *,
    positive_pool_limit: int | None = None,
    negative_pool_limit: int | None = None,
    max_k: int = 4,
    negative_min_k: int = 0,
) -> SelfCondScoreMasking:
    head_width = int(diffmean.numel())
    return SelfCondScoreMasking(
        target_scores=diffmean,
        positive_pool_limit=PositiveInt.parse(
            head_width if positive_pool_limit is None else positive_pool_limit
        ),
        negative_pool_limit=PositiveInt.parse(
            head_width if negative_pool_limit is None else negative_pool_limit
        ),
        negative_selection=_negative_selection(max_k=max_k, min_k=negative_min_k),
    )


def _config(
    diffmean: t.Tensor,
    *,
    seed: int = 123,
    max_k: int = 4,
    positive_min_k: int = 1,
    negative_min_k: int = 0,
    positive_pool_limit: int | None = None,
    negative_pool_limit: int | None = None,
    n_sae_features: int = 6,
    feature_mask: FeatureMask | None = None,
    cluster_consolidation: FeatureClusterConsolidation | None = None,
) -> SelfCondConfig:
    generator = t.Generator(device="cpu")
    generator.manual_seed(seed)
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(64),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            selection=_positive_selection(max_k=max_k, min_k=positive_min_k),
            percentile_table=t.zeros((n_sae_features, 99), dtype=t.float64),
            rng=generator,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=PositiveInt.parse(n_sae_features),
        feature_mask=feature_mask,
        cluster_consolidation=cluster_consolidation,
        score_policy=_masking(
            diffmean,
            positive_pool_limit=positive_pool_limit,
            negative_pool_limit=negative_pool_limit,
            max_k=max_k,
            negative_min_k=negative_min_k,
        ),
    )


def _interpolation_config(
    diffmean: t.Tensor,
    weight: float,
    *,
    seed: int = 123,
    max_k: int = 4,
) -> SelfCondConfig:
    generator = t.Generator(device="cpu")
    generator.manual_seed(seed)
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(64),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=SamplingBundle(
            selection=_positive_selection(max_k=max_k),
            percentile_table=t.zeros((6, 99), dtype=t.float64),
            rng=generator,
        ),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(1),
        n_sae_features=PositiveInt.parse(6),
        score_policy=SelfCondScoreInterpolation(
            target_scores=diffmean,
            weight=UnitInterval.parse(weight),
            negative_selection=_negative_selection(max_k=max_k),
        ),
    )


def _run_and_capture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    diffmean: t.Tensor,
    logits: t.Tensor,
    counts: tuple[int, int] = (2, 2),
    model=None,
    config: SelfCondConfig | None = None,
    use_seed_from_prompt: bool = False,
) -> tuple[dict[int, int], list[tuple[t.Tensor, int, int]]]:
    """Run one masked step; return the built segment and every selection call."""

    model = _model() if model is None else model
    config = _config(diffmean) if config is None else config
    provider = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
    state = build_test_self_cond_state(provider)
    observed_segments: list[dict[int, int]] = []
    original_build = provider._build_segment_inputs

    def capture_segment(segment: dict[int, int]):
        observed_segments.append(dict(segment))
        return original_build(segment)

    monkeypatch.setattr(provider, "_build_segment_inputs", capture_segment)
    monkeypatch.setattr(
        selfcond_module,
        "sample_action_counts",
        lambda *_args, **_kwargs: (t.tensor([counts[0]]), t.tensor([counts[1]])),
    )
    observed_calls: list[tuple[t.Tensor, int, int]] = []
    original_select = _ORIGINAL_SELECT_TOPK_BOTTOML

    def capture_select(scores: t.Tensor, k: t.Tensor, l: t.Tensor):  # noqa: E741
        observed_calls.append((scores.clone(), int(k[0].item()), int(l[0].item())))
        return original_select(scores, k, l)

    monkeypatch.setattr(selfcond_module, "select_topk_bottoml", capture_select)
    if use_seed_from_prompt:
        provider.seed_from_prompt(state, logits)
    else:
        provider.step(state, logits)
    return observed_segments[-1], observed_calls


def _expected_ranking_scores(logits: t.Tensor) -> t.Tensor:
    return logits.max(dim=1).values.detach().to(t.float32)


def test_ranked_candidate_pools_apply_mask_before_stable_rank_and_exclude_zero() -> None:
    target = t.tensor([0.9, 0.8, 0.8, 0.1, -0.9, -0.8, -0.8, 0.0])
    keep = t.tensor([False, True, True, True, False, True, True, True])
    positive, negative = rank_masked_candidate_pools(
        target,
        positive_pool_limit=2,
        negative_pool_limit=1,
        keep_rows=keep,
        positive_min_k=1,
        negative_min_k=1,
        owner="test",
    )
    # Masked extrema do not consume quota; equal scores retain head-row order.
    assert positive.nonzero().flatten().tolist() == [1, 2]
    assert negative.nonzero().flatten().tolist() == [5]
    assert not positive[7] and not negative[7]


def test_unrestricted_rank_pools_match_the_legacy_threshold_zero_boundary() -> None:
    target = t.tensor([0.8, 0.0, -0.5, -0.0, 0.2, -0.9])
    keep = t.tensor([True, True, False, True, True, True])
    positive, negative = rank_masked_candidate_pools(
        target,
        positive_pool_limit=int(keep.sum().item()),
        negative_pool_limit=int(keep.sum().item()),
        keep_rows=keep,
        positive_min_k=1,
        negative_min_k=0,
        owner="factor-zero parity",
    )

    assert t.equal(positive, keep & (target > 0.0))
    assert t.equal(negative, keep & (target < 0.0))


def test_ranked_candidate_pools_accept_large_limits_and_optional_empty_side() -> None:
    positive, negative = rank_masked_candidate_pools(
        t.tensor([0.2, 0.1, 0.0]),
        positive_pool_limit=99,
        negative_pool_limit=99,
        keep_rows=None,
        positive_min_k=1,
        negative_min_k=0,
        owner="test",
    )
    assert positive.tolist() == [True, True, False]
    assert not bool(negative.any())


@pytest.mark.parametrize(
    ("target", "positive_min_k", "negative_min_k", "match"),
    [
        (t.zeros(3), 0, 0, "both candidate pools are empty"),
        (t.tensor([-0.2, -0.1, 0.0]), 1, 0, "empty positive"),
        (t.tensor([0.2, 0.1, 0.0]), 0, 1, "empty negative"),
    ],
)
def test_ranked_candidate_pools_reject_unsatisfied_empty_sides(
    target: t.Tensor,
    positive_min_k: int,
    negative_min_k: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        rank_masked_candidate_pools(
            target,
            positive_pool_limit=1,
            negative_pool_limit=1,
            keep_rows=None,
            positive_min_k=positive_min_k,
            negative_min_k=negative_min_k,
            owner="test",
        )


@pytest.mark.parametrize(
    ("target", "exception", "match"),
    [
        (t.tensor([[0.2, -0.2]]), TypeError, "rank-1 real floating"),
        (t.tensor([1, -1]), TypeError, "rank-1 real floating"),
        (t.empty(2, device="meta"), TypeError, "rank-1 real floating"),
        (t.tensor([float("nan"), -0.2]), ValueError, "must be finite"),
        (t.tensor([1.01, -0.2]), ValueError, r"lie in \[-1, 1\]"),
    ],
)
def test_ranked_candidate_pools_reject_invalid_targets(
    target: t.Tensor,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        rank_masked_candidate_pools(
            target,
            positive_pool_limit=1,
            negative_pool_limit=1,
            keep_rows=None,
            positive_min_k=0,
            negative_min_k=0,
            owner="test",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("positive_pool_limit", True, "positive_pool_limit"),
        ("positive_pool_limit", 0, "positive_pool_limit"),
        ("negative_pool_limit", 1.5, "negative_pool_limit"),
        ("positive_min_k", True, "positive_min_k"),
        ("positive_min_k", -1, "positive_min_k"),
        ("negative_min_k", 0.5, "negative_min_k"),
    ],
)
def test_ranked_candidate_pools_reject_invalid_limits_and_count_floors(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs: dict[str, object] = {
        "positive_pool_limit": 1,
        "negative_pool_limit": 1,
        "keep_rows": None,
        "positive_min_k": 0,
        "negative_min_k": 0,
        "owner": "test",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        rank_masked_candidate_pools(t.tensor([0.2, -0.2]), **kwargs)


@pytest.mark.parametrize(
    ("keep_rows", "exception", "match"),
    [
        ([True, True], TypeError, "materialized rank-1 bool"),
        (t.tensor([[True, True]]), TypeError, "materialized rank-1 bool"),
        (t.tensor([1, 1]), TypeError, "materialized rank-1 bool"),
        (t.tensor([True, True]).to_sparse(), TypeError, "materialized rank-1 bool"),
        (t.empty(2, dtype=t.bool, device="meta"), TypeError, "materialized rank-1 bool"),
        (t.tensor([True]), ValueError, "width 1 does not match target width 2"),
    ],
)
def test_ranked_candidate_pools_reject_invalid_keep_rows(
    keep_rows: object,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        rank_masked_candidate_pools(
            t.tensor([0.2, -0.2]),
            positive_pool_limit=1,
            negative_pool_limit=1,
            keep_rows=keep_rows,  # ty: ignore[invalid-argument-type]
            positive_min_k=0,
            negative_min_k=0,
            owner="test",
        )


@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_ranked_candidate_pools_preserve_cuda_device() -> None:
    target = t.tensor([0.4, 0.2, -0.3, -0.1], device="cuda")
    keep = t.tensor([True, False, True, True], device="cuda")
    positive, negative = rank_masked_candidate_pools(
        target,
        positive_pool_limit=2,
        negative_pool_limit=1,
        keep_rows=keep,
        positive_min_k=1,
        negative_min_k=1,
        owner="cuda test",
    )
    assert positive.device.type == negative.device.type == "cuda"
    assert positive.cpu().tolist() == [True, False, False, False]
    assert negative.cpu().tolist() == [False, False, True, False]

    with pytest.raises(ValueError, match="keep_rows device cpu does not match target device cuda"):
        rank_masked_candidate_pools(
            target,
            positive_pool_limit=2,
            negative_pool_limit=1,
            keep_rows=keep.cpu(),
            positive_min_k=1,
            negative_min_k=1,
            owner="cuda test",
        )


def test_provider_constructs_ranked_pools_once_without_advancing_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = selfcond_module.rank_masked_candidate_pools

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(selfcond_module, "rank_masked_candidate_pools", spy)
    config = _config(t.tensor([0.5, 0.4, 0.0, -0.4, -0.5, 0.0]))
    assert isinstance(config.selection, SamplingBundle)
    before = config.selection.rng.get_state().clone()
    provider = SelfConditionedFeatureProvider(_model(), config, t.device("cpu"))
    assert calls == 1
    assert t.equal(config.selection.rng.get_state(), before)
    state = build_test_self_cond_state(provider)
    provider.step(state, t.zeros((1, 1, 6)))
    assert calls == 1


# ---------------------------------------------------------------------------
# Score semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("positive_limit, negative_limit", [(3, 3), (2, 2)])
def test_selection_scores_are_exactly_pooled_logits_per_pool(
    monkeypatch: pytest.MonkeyPatch,
    positive_limit: int,
    negative_limit: int,
) -> None:
    raw = t.tensor(
        [[[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0], [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]]],
        dtype=t.bfloat16,
    )
    diffmean = t.tensor([0.75, 0.5, 0.25, -0.25, -0.5, -0.75])
    _segment, calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=raw,
        config=_config(
            diffmean,
            positive_pool_limit=positive_limit,
            negative_pool_limit=negative_limit,
        ),
    )
    ranking_scores = _expected_ranking_scores(raw)
    pos_rows = t.tensor([True, True, positive_limit == 3, False, False, False])
    neg_rows = t.tensor([False, False, False, negative_limit == 3, True, True])

    assert len(calls) == 2
    pos_scores, pos_k, pos_l = calls[0]
    neg_scores, neg_k, neg_l = calls[1]
    assert t.equal(pos_scores, ranking_scores[:, pos_rows])
    assert t.equal(neg_scores, ranking_scores[:, neg_rows])
    assert (pos_k, pos_l) == (2, 0)
    assert (neg_k, neg_l) == (0, 2)


def test_masking_ranks_distinct_saturated_logits_without_false_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[-20.0, -21.0, 20.0, 21.0, 0.0, 1.0]]])
    diffmean = t.tensor([-0.5, -0.5, 0.5, 0.5, 0.0, 0.0])
    segment, calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(1, 1),
    )
    assert segment == {3: 0, 1: 1}
    assert t.equal(calls[0][0], t.tensor([[20.0, 21.0]]))
    assert t.equal(calls[1][0], t.tensor([[-20.0, -21.0]]))


def test_sign_rule_overrides_extreme_model_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    # Row 1 (d < 0) has the globally highest q; row 0 (d > 0) the globally
    # lowest. Rows 2 and 5 (d == 0) take both extremes and must appear nowhere.
    logits = t.tensor([[[-6.0, 5.0, 6.0, 1.0, -1.0, -7.0]]])
    diffmean = t.tensor([0.75, -0.75, 0.0, 0.5, -0.5, 0.0])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
    )
    assert segment == {3: 0, 0: 0, 1: 1, 4: 1}


def test_zero_diffmean_rows_are_excluded_even_with_negative_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[9.0, -9.0, 0.5, -0.5, 8.0, -8.0]]])
    diffmean = t.tensor([0.0, -0.0, 0.5, -0.5, 0.0, -0.0])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(1, 1),
    )
    assert segment == {2: 0, 3: 1}


def test_rank_limits_keep_the_strongest_diffmean_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[1.0, 2.0, -1.0, -2.0, 0.0, 3.0]]])
    diffmean = t.tensor([0.5, 0.25, -0.5, -0.25, 0.0, 0.75])
    segment, calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(2, 1),
        config=_config(
            diffmean,
            positive_pool_limit=2,
            negative_pool_limit=1,
        ),
    )
    # Positive pool is exactly {0, 5}; negative pool is exactly {2}.
    assert segment == {5: 0, 0: 0, 2: 1}
    assert calls[0][0].shape == (1, 2)
    assert calls[1][0].shape == (1, 1)


def test_tie_scores_resolve_by_head_row_order_within_each_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.zeros((1, 1, 6))
    diffmean = t.tensor([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
    )
    assert segment == {0: 0, 1: 0, 3: 1, 4: 1}


def test_seed_from_prompt_applies_the_masking_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_logits = t.tensor([[[-6.0, 5.0, 6.0, 1.0, -1.0, -7.0]] * 3])
    diffmean = t.tensor([0.75, -0.75, 0.0, 0.5, -0.5, 0.0])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=prompt_logits,
        use_seed_from_prompt=True,
    )
    assert segment == {3: 0, 0: 0, 1: 1, 4: 1}


# ---------------------------------------------------------------------------
# Count clamping
# ---------------------------------------------------------------------------


def test_counts_above_pool_sizes_clamp_and_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logits = t.tensor([[[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]]])
    diffmean = t.tensor([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    segment, calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(3, 2),
    )
    assert segment == {0: 0, 1: 1}
    assert calls[0][1:] == (1, 0)
    assert calls[1][1:] == (0, 1)
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "positive count: configured pool limit=6, actual ranked pool size=1, "
        "sampled count=3, clamped count=1" in message
        for message in messages
    )
    assert any(
        "negative count: configured pool limit=6, actual ranked pool size=1, "
        "sampled count=2, clamped count=1" in message
        for message in messages
    )


def test_counts_within_pool_sizes_do_not_warn(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    logits = t.tensor([[[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]]])
    diffmean = t.tensor([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])
    _segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(2, 2),
    )
    assert not any("clamped count" in record.getMessage() for record in caplog.records)


def test_empty_negative_pool_clamps_to_positive_only_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[1.0, 4.0, 2.0, 3.0, 0.0, -1.0]]])
    diffmean = t.tensor([0.5, 0.25, 0.75, 0.125, 0.5, 0.25])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(1, 2),
    )
    assert segment == {1: 0}


def test_zero_positive_and_negative_counts_fail_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SelfConditionedFeatureProvider(
        _model(),
        _config(t.linspace(0.8, -0.8, 6)),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    monkeypatch.setattr(
        selfcond_module,
        "sample_action_counts",
        lambda *_args, **_kwargs: (t.tensor([0]), t.tensor([0])),
    )

    def forbidden_encode(*_args, **_kwargs):
        raise AssertionError("empty selection must fail before encoder invocation")

    monkeypatch.setattr(provider, "_encode_one_segment", forbidden_encode)
    with pytest.raises(
        EmptySelfConditioningSegmentError,
        match="fp_logits_tail score masking selected zero positive and zero negative",
    ):
        provider.step(state, t.zeros((1, 1, 6)))


# ---------------------------------------------------------------------------
# Compact checkpoints, masks, and clusters
# ---------------------------------------------------------------------------


def test_compact_head_rows_translate_to_original_feature_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab = FeatureVocab(n_sae_features=6, original_ids=(1, 3, 5))
    model = _model(feature_vocab=vocab)
    config = _config(t.tensor([0.75, 0.0, -0.75]), max_k=2, n_sae_features=6)
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=t.tensor([0.75, 0.0, -0.75]),
        logits=t.zeros((1, 1, 3)),
        counts=(1, 1),
        model=model,
        config=config,
    )
    assert segment == {1: 0, 5: 1}


def test_feature_mask_restricts_pools_before_clamping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Feature 0 has the largest diffmean and the highest score, but the
    # blacklist removes it from the positive pool, so the drawn count of two
    # clamps to the single surviving positive feature.
    mask = FeatureMask(FeatureMaskMode.blacklist, (0,))
    logits = t.tensor([[[9.0, 1.0, -1.0, -2.0, 0.0, 0.0]]])
    diffmean = t.tensor([0.75, 0.5, -0.5, -0.25, 0.0, 0.0])
    segment, calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(2, 1),
        config=_config(diffmean, feature_mask=mask),
    )
    assert segment == {1: 0, 3: 1}
    assert calls[0][0].shape == (1, 1)
    assert calls[0][1:] == (1, 0)


def test_compact_clustered_masked_rank_maps_selected_original_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 2], [4, 5]]}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(cluster_path)}}, 6)
    assert resolved is not None
    consolidation = FeatureClusterConsolidation.build(resolved, None)
    vocab = FeatureVocab(n_sae_features=6, original_ids=(0, 1, 3, 4))
    model = _model(feature_vocab=vocab)
    diffmean = t.tensor([0.9, 0.75, -0.5, -0.75])
    config = _config(
        diffmean,
        cluster_consolidation=consolidation,
        feature_mask=FeatureMask(FeatureMaskMode.blacklist, (0,)),
        n_sae_features=6,
        positive_pool_limit=1,
        negative_pool_limit=1,
    )
    observed: list[t.Tensor] = []
    original_assert = FeatureClusterConsolidation.assert_representatives_only

    def spy(self, selected_ids, *, surface):
        observed.append(selected_ids.clone())
        return original_assert(self, selected_ids, surface=surface)

    monkeypatch.setattr(FeatureClusterConsolidation, "assert_representatives_only", spy)
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=t.tensor([[[0.0, 1.0, 0.0, -1.0]]]),
        counts=(1, 1),
        model=model,
        config=config,
    )
    assert segment == {1: 0, 4: 1}
    assert observed and sorted(observed[-1].tolist()) == [1, 4]


# ---------------------------------------------------------------------------
# Differential equality and RNG contracts
# ---------------------------------------------------------------------------


def test_unrestricted_rank_pools_equal_interpolation_at_zero_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # q descends over rows 0..5 while the diffmean signs align with the
    # interpolation pick: its top-2 all have d > 0 and its bottom-2 all have
    # d < 0, so both policies must produce the identical segment.
    logits = t.tensor([[[3.0, 2.0, 1.0, -1.0, -2.0, -3.0]]])
    diffmean = t.tensor([0.5, 0.375, 0.25, -0.25, -0.375, -0.5])
    monkeypatch.setattr(
        selfcond_module,
        "sample_action_counts",
        lambda *_args, **_kwargs: (t.tensor([2]), t.tensor([2])),
    )

    segments: list[dict[int, int]] = []
    for config in (
        _config(diffmean),
        _interpolation_config(diffmean, 0.0),
    ):
        model = _model()
        provider = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
        state = build_test_self_cond_state(provider)
        original_build = provider._build_segment_inputs

        def capture(segment: dict[int, int], build=original_build):
            segments.append(dict(segment))
            return build(segment)

        monkeypatch.setattr(provider, "_build_segment_inputs", capture)
        provider.step(state, logits)

    assert segments[0] == segments[1] == {0: 0, 1: 0, 5: 1, 4: 1}


def test_count_draws_match_interpolation_for_the_same_selection_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[3.0, 2.0, 1.0, -1.0, -2.0, -3.0]]])
    diffmean = t.tensor([0.5, 0.375, 0.25, -0.25, -0.375, -0.5])
    observed: dict[str, tuple[int, int]] = {}

    original_select = _ORIGINAL_SELECT_TOPK_BOTTOML
    for label, config in (
        ("masking", _config(diffmean, seed=917)),
        ("interpolation", _interpolation_config(diffmean, 0.0, seed=917)),
    ):
        model = _model()
        provider = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
        state = build_test_self_cond_state(provider)
        calls: list[tuple[int, int]] = []

        def capture(scores: t.Tensor, k: t.Tensor, l: t.Tensor, sink=calls):  # noqa: E741
            sink.append((int(k[0].item()), int(l[0].item())))
            return original_select(scores, k, l)

        monkeypatch.setattr(selfcond_module, "select_topk_bottoml", capture)
        provider.step(state, logits)
        if label == "masking":
            observed[label] = (calls[0][0], calls[1][1])
        else:
            observed[label] = calls[0]

    assert observed["masking"] == observed["interpolation"]


def test_local_generator_does_not_mutate_global_rng() -> None:
    model = _model()
    provider = SelfConditionedFeatureProvider(
        model,
        _config(t.linspace(0.8, -0.8, 6)),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    before = t.random.get_rng_state().clone()
    provider.step(state, t.zeros((1, 1, 6)))
    after = t.random.get_rng_state()
    assert t.equal(before, after)


# ---------------------------------------------------------------------------
# Value-object and config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "exception", "match"),
    [
        (t.tensor([[0.0, 0.1]]), TypeError, "materialized rank-1 strided"),
        (t.tensor([float("nan")]), ValueError, "finite after float32"),
        (t.tensor([float("inf")]), ValueError, "finite after float32"),
        (t.tensor([-1.01]), ValueError, r"lie in \[-1, 1\]"),
        (t.tensor([1.01]), ValueError, r"lie in \[-1, 1\]"),
    ],
)
def test_masking_target_rejects_invalid_shape_or_values(
    scores: t.Tensor,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        SelfCondScoreMasking(
            target_scores=scores,
            positive_pool_limit=PositiveInt.parse(1),
            negative_pool_limit=PositiveInt.parse(1),
            negative_selection=_negative_selection(),
        )


def test_masking_target_is_detached_contiguous_cpu_snapshot() -> None:
    source = t.linspace(-0.8, 0.8, 8, requires_grad=True)[::2]
    masking = SelfCondScoreMasking(
        target_scores=source,
        positive_pool_limit=PositiveInt.parse(1),
        negative_pool_limit=PositiveInt.parse(1),
        negative_selection=_negative_selection(),
    )
    source.detach().fill_(0.0)
    assert masking.target_scores.device.type == "cpu"
    assert masking.target_scores.dtype == t.float32
    assert masking.target_scores.is_contiguous()
    assert not masking.target_scores.requires_grad
    assert not t.equal(masking.target_scores, source.detach())


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_masking_pool_limit_requires_a_positive_integer(limit: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        SelfCondScoreMasking(
            target_scores=t.zeros(6),
            positive_pool_limit=limit,  # ty: ignore[invalid-argument-type]
            negative_pool_limit=PositiveInt.parse(1),
            negative_selection=_negative_selection(),
        )


def test_masking_rejects_simultaneous_legacy_bias() -> None:
    base = _config(t.linspace(0.8, -0.8, 6))
    with pytest.raises(ValueError, match="cannot combine fp_logit_bias with score_masking"):
        SelfCondConfig(
            self_cond_segment_size=base.self_cond_segment_size,
            pool_mode=base.pool_mode,
            selection=base.selection,
            selection_producer=base.selection_producer,
            bucket_producer=base.bucket_producer,
            num_buckets=base.num_buckets,
            n_sae_features=base.n_sae_features,
            fp_logit_bias=t.zeros(6),
            score_policy=base.score_policy,
        )


def test_score_policy_union_makes_dual_policies_unrepresentable() -> None:
    # One union-typed field replaces the two optional policy fields, so a
    # config carrying both policies is no longer expressible at all.
    field_names = {field.name for field in dataclasses.fields(SelfCondConfig)}
    assert "score_policy" in field_names
    assert "score_interpolation" not in field_names
    assert "score_masking" not in field_names


def test_abstract_dense_policy_base_is_not_constructible() -> None:
    with pytest.raises(TypeError, match="abstract policy contract"):
        SelfCondDenseScorePolicy(
            target_scores=t.zeros(6),
            negative_selection=_negative_selection(),
        )


def test_score_policy_rejects_a_non_policy_value() -> None:
    base = _config(t.linspace(0.8, -0.8, 6))
    with pytest.raises(BeartypeCallHintParamViolation):
        SelfCondConfig(
            self_cond_segment_size=base.self_cond_segment_size,
            pool_mode=base.pool_mode,
            selection=base.selection,
            selection_producer=base.selection_producer,
            bucket_producer=base.bucket_producer,
            num_buckets=base.num_buckets,
            n_sae_features=base.n_sae_features,
            score_policy=object(),  # ty: ignore[invalid-argument-type]
        )


def _config_kwargs(base: SelfCondConfig) -> dict:
    return {
        "self_cond_segment_size": base.self_cond_segment_size,
        "pool_mode": base.pool_mode,
        "selection": base.selection,
        "selection_producer": base.selection_producer,
        "bucket_producer": base.bucket_producer,
        "num_buckets": base.num_buckets,
        "n_sae_features": base.n_sae_features,
        "score_policy": base.score_policy,
    }


def test_masking_requires_max_pool_pooling() -> None:
    kwargs = _config_kwargs(_config(t.zeros(6)))
    kwargs["pool_mode"] = FeaturePredLossMode.LAST_TOKEN
    with pytest.raises(ValueError, match="score_masking requires SelfCondConfig.pool_mode"):
        SelfCondConfig(**kwargs)


def test_masking_requires_top_k_percentile_sampling() -> None:
    base = _config(t.zeros(6))
    kwargs = _config_kwargs(base)
    weighted = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=4,
        mode=SamplingMode.weighted,
    )
    kwargs["selection"] = SamplingBundle(
        selection=weighted,
        percentile_table=t.zeros((6, 99), dtype=t.float64),
        rng=t.Generator(device="cpu"),
    )
    with pytest.raises(ValueError, match="score_masking requires SamplingMode.top_k_percentile"):
        SelfCondConfig(**kwargs)


def test_masking_requires_one_active_bucket() -> None:
    kwargs = _config_kwargs(_config(t.zeros(6)))
    kwargs["num_buckets"] = PositiveInt.parse(2)
    with pytest.raises(ValueError, match="score_masking requires exactly one active bucket"):
        SelfCondConfig(**kwargs)


@pytest.mark.parametrize("side", ["positive", "negative"])
def test_masking_requires_bounded_counts(side: str) -> None:
    base = _config(t.zeros(6))
    kwargs = _config_kwargs(base)
    if side == "positive":
        unbounded = SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=None,
            mode=SamplingMode.top_k_percentile,
        )
        kwargs["selection"] = SamplingBundle(
            selection=unbounded,
            percentile_table=t.zeros((6, 99), dtype=t.float64),
            rng=t.Generator(device="cpu"),
        )
    else:
        kwargs["score_policy"] = SelfCondScoreMasking(
            target_scores=t.zeros(6),
            positive_pool_limit=PositiveInt.parse(6),
            negative_pool_limit=PositiveInt.parse(6),
            negative_selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=3.0,
                std=3.0,
                min_k=0,
                max_k=None,
            ),
        )
    with pytest.raises(ValueError, match=f"score_masking {side} selection must have bounded"):
        SelfCondConfig(**kwargs)


# ---------------------------------------------------------------------------
# Provider construction validation
# ---------------------------------------------------------------------------


def test_provider_rejects_wrong_target_width() -> None:
    with pytest.raises(ValueError, match="score_masking target width"):
        SelfConditionedFeatureProvider(
            _model(),
            _config(t.tensor([0.5, -0.5])),
            t.device("cpu"),
        )


def test_provider_rejects_single_bucket_encoder() -> None:
    model = build_tiny_model(
        n_sae_features=6,
        num_buckets=1,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )
    with pytest.raises(ValueError, match="score_masking requires an encoder with exactly two"):
        SelfConditionedFeatureProvider(
            model,
            _config(t.linspace(0.8, -0.8, 6)),
            t.device("cpu"),
        )


def test_provider_rejects_non_max_pool_trained_head() -> None:
    with pytest.raises(RuntimeError, match="score_masking requires a checkpoint trained"):
        SelfConditionedFeatureProvider(
            _model(loss_mode=FeaturePredLossMode.LAST_TOKEN),
            _config(t.linspace(0.8, -0.8, 6)),
            t.device("cpu"),
        )


def test_provider_rejects_an_all_zero_target() -> None:
    with pytest.raises(ValueError, match="both candidate pools are empty"):
        SelfConditionedFeatureProvider(
            _model(),
            _config(t.zeros(6)),
            t.device("cpu"),
        )


def test_provider_selects_weak_magnitude_rows_with_rank_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diffmean = t.tensor([0.01, -0.01, 0.005, -0.005, 0.0, 0.0])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=t.tensor([[[0.0, 0.0, 100.0, -100.0, 0.0, 0.0]]]),
        counts=(1, 1),
        config=_config(diffmean, positive_pool_limit=1, negative_pool_limit=1),
    )
    assert segment == {0: 0, 1: 1}


def test_provider_rejects_mask_that_empties_both_pools() -> None:
    mask = FeatureMask(FeatureMaskMode.blacklist, (0, 1))
    diffmean = t.tensor([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="both candidate pools are empty"):
        SelfConditionedFeatureProvider(
            _model(),
            _config(diffmean, feature_mask=mask),
            t.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Misuse and faulty-dependency defenses
# ---------------------------------------------------------------------------


def test_direct_masking_step_on_an_interpolation_provider_is_rejected() -> None:
    provider = SelfConditionedFeatureProvider(
        _model(),
        _interpolation_config(t.linspace(0.8, -0.8, 6), 0.5),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    with pytest.raises(RuntimeError, match="score-masking provider state is incomplete"):
        provider._step_score_masking(state, t.zeros((1, 1, 6)), owner="fp_logits_tail")


def test_masking_step_rejects_a_desynced_non_sampling_selection() -> None:
    provider = SelfConditionedFeatureProvider(
        _model(),
        _config(t.linspace(0.8, -0.8, 6)),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    object.__setattr__(provider.cfg, "selection", object())
    with pytest.raises(RuntimeError, match="score-masking positive selection is not sampling"):
        provider._step_score_masking(state, t.zeros((1, 1, 6)), owner="fp_logits_tail")


def test_masking_step_refuses_duplicate_rows_from_a_faulty_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SelfConditionedFeatureProvider(
        _model(),
        _config(t.linspace(0.8, -0.8, 6)),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    monkeypatch.setattr(
        selfcond_module,
        "sample_action_counts",
        lambda *_args, **_kwargs: (t.tensor([2]), t.tensor([0])),
    )
    monkeypatch.setattr(
        selfcond_module,
        "select_topk_bottoml",
        lambda *_args, **_kwargs: SelectedActions(
            pos_ids=t.tensor([[0, 0]]),
            neg_ids=t.tensor([[0]]),
            k=t.tensor([2]),
            l=t.tensor([0]),
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate or overlapping feature ids"):
        provider.step(state, t.zeros((1, 1, 6)))


def test_masking_step_refuses_out_of_dictionary_ids_from_a_corrupted_vocab_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab = FeatureVocab(n_sae_features=6, original_ids=(1, 3, 5))
    model = _model(feature_vocab=vocab)
    diffmean = t.tensor([0.5, 0.0, 0.75])
    provider = SelfConditionedFeatureProvider(
        model,
        _config(diffmean, n_sae_features=6),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    provider._compact_candidate_feature_ids = t.tensor([1, 3, 99], dtype=t.int32)
    monkeypatch.setattr(
        selfcond_module,
        "sample_action_counts",
        lambda *_args, **_kwargs: (t.tensor([1]), t.tensor([0])),
    )
    with pytest.raises(RuntimeError, match="outside the dictionary"):
        provider.step(state, t.tensor([[[0.0, 0.0, 5.0]]]))


# ---------------------------------------------------------------------------
# Randomized differential oracle
# ---------------------------------------------------------------------------


def _reference_masked_segment(
    q: list[float],
    diffmean: list[float],
    positive_pool_limit: int,
    negative_pool_limit: int,
    positive_count: int,
    negative_count: int,
    allowed: list[int],
) -> dict[int, int]:
    """Independent pure-Python oracle for both rank stages."""

    positive_pool = sorted(
        (row for row in allowed if diffmean[row] > 0.0),
        key=lambda row: (-diffmean[row], row),
    )[:positive_pool_limit]
    negative_pool = sorted(
        (row for row in allowed if diffmean[row] < 0.0),
        key=lambda row: (diffmean[row], row),
    )[:negative_pool_limit]
    positive_count = min(positive_count, len(positive_pool))
    negative_count = min(negative_count, len(negative_pool))
    top = sorted(positive_pool, key=lambda row: (-q[row], row))[:positive_count]
    bottom = sorted(negative_pool, key=lambda row: (q[row], row))[:negative_count]
    return {**dict.fromkeys(top, 0), **dict.fromkeys(bottom, 1)}


def test_masked_selection_matches_the_pure_python_reference_across_random_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = t.Generator(device="cpu")
    generator.manual_seed(20260816)
    model = _model()
    checked = 0
    for case in range(60):
        logits = (t.rand((1, 2, 6), generator=generator) - 0.5) * 8.0
        diffmean = (t.rand(6, generator=generator) - 0.5) * 1.8
        diffmean[t.rand(6, generator=generator) < 0.25] = 0.0
        positive_pool_limit = int(t.randint(1, 8, (), generator=generator).item())
        negative_pool_limit = int(t.randint(1, 8, (), generator=generator).item())
        counts = (
            int(t.randint(0, 5, (), generator=generator).item()),
            int(t.randint(0, 5, (), generator=generator).item()),
        )
        blacklist: tuple[int, ...] = ()
        if case % 4 == 0:
            blacklist = (int(t.randint(0, 6, (), generator=generator).item()),)
        mask = FeatureMask(FeatureMaskMode.blacklist, blacklist) if blacklist else None
        allowed = [row for row in range(6) if row not in blacklist]

        d_list = diffmean.tolist()
        positive_pool_exists = any(d_list[row] > 0.0 for row in allowed)
        config = _config(
            diffmean,
            feature_mask=mask,
            positive_pool_limit=positive_pool_limit,
            negative_pool_limit=negative_pool_limit,
        )
        if not positive_pool_exists:
            # The helper's positive policy has min_k=1, so an empty positive
            # pool is unsatisfiable regardless of the negative pool's state.
            with pytest.raises(
                ValueError,
                match="both candidate pools are empty|empty positive",
            ):
                SelfConditionedFeatureProvider(model, config, t.device("cpu"))
            checked += 1
            continue

        q_row = (2.0 * t.sigmoid(logits.max(dim=1).values.to(t.float32)) - 1.0)[0].tolist()
        expected = _reference_masked_segment(
            q_row,
            d_list,
            positive_pool_limit,
            negative_pool_limit,
            counts[0],
            counts[1],
            allowed,
        )
        if not expected:
            provider = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
            state = build_test_self_cond_state(provider)
            monkeypatch.setattr(
                selfcond_module,
                "sample_action_counts",
                lambda *_args, counts=counts, **_kwargs: (
                    t.tensor([counts[0]]),
                    t.tensor([counts[1]]),
                ),
            )
            with pytest.raises(EmptySelfConditioningSegmentError):
                provider.step(state, logits)
            checked += 1
            continue

        segment, _calls = _run_and_capture(
            monkeypatch,
            diffmean=diffmean,
            logits=logits,
            counts=counts,
            model=model,
            config=config,
        )
        assert segment == expected, (
            f"case {case}: q={q_row}, d={d_list}, "
            f"limits=({positive_pool_limit}, {negative_pool_limit}), "
            f"counts={counts}, blacklist={blacklist}"
        )
        checked += 1
    assert checked >= 40


def test_interpolation_cardinality_check_survives_the_shared_refactor() -> None:
    # The bounded positive+negative cardinality check is interpolation-only
    # (masking clamps instead); the shared dense-policy refactor moved the
    # block, so pin both that it still fires for interpolation and that the
    # equivalent masked config constructs fine.
    with pytest.raises(ValueError, match="does not fit the eligible feature vocabulary"):
        SelfConditionedFeatureProvider(
            _model(),
            _interpolation_config(t.linspace(0.8, -0.8, 6), 0.5, max_k=5),
            t.device("cpu"),
        )
    SelfConditionedFeatureProvider(
        _model(),
        _config(t.linspace(0.8, -0.8, 6), max_k=5),
        t.device("cpu"),
    )


def test_empty_positive_pool_with_min_one_positive_policy_is_rejected() -> None:
    # Every drawn count would clamp below the checkpoint's min_k=1 floor, and
    # any zero negative draw would then crash mid-generation on an empty
    # segment; construction must refuse instead.
    diffmean = t.tensor([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
    with pytest.raises(ValueError, match="empty positive candidate pool"):
        SelfConditionedFeatureProvider(
            _model(),
            _config(diffmean, positive_min_k=1),
            t.device("cpu"),
        )


def test_empty_negative_pool_with_min_one_negative_policy_is_rejected() -> None:
    diffmean = t.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    with pytest.raises(ValueError, match="empty negative candidate pool"):
        SelfConditionedFeatureProvider(
            _model(),
            _config(diffmean, negative_min_k=1),
            t.device("cpu"),
        )


def test_empty_positive_pool_with_min_zero_policy_yields_negatives_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With a zero-count-capable positive policy an empty positive pool is a
    # legitimate degenerate regime: the clamp yields inactive-only segments.
    diffmean = t.tensor([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
    segment, _calls = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=t.tensor([[[1.0, 2.0, 3.0, -1.0, -2.0, -3.0]]]),
        counts=(2, 1),
        config=_config(diffmean, positive_min_k=0),
    )
    assert segment == {4: 1}


def test_ordinary_clamp_logs_info_and_clamp_to_zero_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Pools of one on each side plus an empty negative regime: the in-pool
    # clamp is the intended ranked-pool behavior (INFO), while a clamp all
    # the way to zero is a degenerate one-sided segment (WARNING).
    logits = t.tensor([[[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]]])
    diffmean = t.tensor([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        logits=logits,
        counts=(3, 2),
    )
    clamp_records = [
        record for record in caplog.records if "clamped count" in record.getMessage()
    ]
    assert [record.levelname for record in clamp_records] == ["INFO", "INFO"]

    caplog.clear()
    positive_only = t.tensor([0.5, 0.25, 0.75, 0.125, 0.5, 0.25])
    _run_and_capture(
        monkeypatch,
        diffmean=positive_only,
        logits=t.tensor([[[1.0, 4.0, 2.0, 3.0, 0.0, -1.0]]]),
        counts=(1, 2),
    )
    zero_records = [
        record
        for record in caplog.records
        if "negative count" in record.getMessage() and "clamped count" in record.getMessage()
    ]
    assert [record.levelname for record in zero_records] == ["WARNING"]
