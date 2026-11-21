"""Dense score interpolation and two-sided SelfConditioning selection."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t

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
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    select_topk_bottoml,
)
from steering.refined_types import PositiveInt, UnitInterval
from tests.encdec_test_helpers import build_tiny_model
from tests.steering._self_conditioned_state import build_test_self_cond_state


def _positive_selection(*, max_k: int = 4) -> SamplingSelection:
    return SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=max_k,
        mode=SamplingMode.top_k_percentile,
    )


def _negative_selection(*, max_k: int = 4) -> NegativeSamplingSelection:
    return NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=0,
        max_k=max_k,
    )


def _model(*, feature_vocab: FeatureVocab | None = None):
    return build_tiny_model(
        n_sae_features=6,
        num_buckets=2,
        feature_vocab=feature_vocab,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )


def _config(
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
    weight: float,
    logits: t.Tensor,
    model=None,
) -> tuple[dict[int, int], t.Tensor]:
    model = _model() if model is None else model
    provider = SelfConditionedFeatureProvider(model, _config(diffmean, weight), t.device("cpu"))
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
        lambda *_args, **_kwargs: (t.tensor([2]), t.tensor([2])),
    )
    observed_scores: list[t.Tensor] = []
    original_select = selfcond_module.select_topk_bottoml

    def capture_select(scores: t.Tensor, k: t.Tensor, l: t.Tensor):  # noqa: E741
        observed_scores.append(scores.clone())
        return original_select(scores, k, l)

    monkeypatch.setattr(selfcond_module, "select_topk_bottoml", capture_select)
    provider.step(state, logits)
    return observed_segments[-1], observed_scores[-1]


@pytest.mark.parametrize(
    ("weight", "oracle"),
    [
        (0.0, lambda z, d: z),
        (1.0 / 6.0, lambda q, d: (5.0 / 6.0) * q + (1.0 / 6.0) * d),
        (0.5, lambda q, d: (q + d) / 2.0),
        (1.0, lambda q, d: d),
    ],
)
def test_exact_dense_interpolation_before_selection(
    monkeypatch: pytest.MonkeyPatch,
    weight: float,
    oracle: Callable[[t.Tensor, t.Tensor], t.Tensor],
) -> None:
    raw = t.tensor(
        [[[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0], [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]]],
        dtype=t.bfloat16,
    )
    diffmean = t.tensor([0.9, 0.7, 0.5, -0.5, -0.7, -0.9])
    _segment, scores = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        weight=weight,
        logits=raw,
    )
    pooled = raw.max(dim=1).values.float()
    q = 2.0 * t.sigmoid(pooled) - 1.0
    expected = (
        oracle(pooled, diffmean.unsqueeze(0))
        if weight == 0.0
        else (
            oracle(q, diffmean.unsqueeze(0))
            if weight == 1.0
            else t.lerp(q, diffmean.unsqueeze(0), weight)
        )
    )

    assert t.equal(scores, expected)


def test_opposite_endpoint_rankings_and_bucket_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[-4.0, -3.0, -2.0, -1.0, 1.0, 2.0]]])
    diffmean = t.tensor([0.9, 0.7, 0.5, -0.5, -0.7, -0.9])

    factor_zero, _scores = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        weight=0.0,
        logits=logits,
    )
    factor_six, _scores = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        weight=1.0,
        logits=logits,
    )

    assert factor_zero == {5: 0, 4: 0, 0: 1, 1: 1}
    assert factor_six == {0: 0, 1: 0, 5: 1, 4: 1}


def test_factor_six_is_independent_of_prompt_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    diffmean = t.tensor([0.9, 0.7, 0.5, -0.5, -0.7, -0.9])
    first, first_scores = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        weight=1.0,
        logits=t.full((1, 3, 6), -100.0),
    )
    second, second_scores = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        weight=1.0,
        logits=t.full((1, 3, 6), 100.0),
    )
    assert first == second
    assert t.equal(first_scores, diffmean.unsqueeze(0))
    assert t.equal(second_scores, diffmean.unsqueeze(0))


def test_weight_zero_ranks_raw_logits_without_sigmoid_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment, scores = _run_and_capture(
        monkeypatch,
        diffmean=t.zeros(6),
        weight=0.0,
        logits=t.tensor([[[-20.0, -21.0, 20.0, 21.0, 0.0, 1.0]]]),
    )
    assert t.equal(scores, t.tensor([[-20.0, -21.0, 20.0, 21.0, 0.0, 1.0]]))
    assert segment == {3: 0, 2: 0, 1: 1, 0: 1}


def test_nonzero_interpolation_remains_exactly_legacy_float32_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([[[-20.0, -21.0, 20.0, 21.0, 0.0, 1.0]]])
    diffmean = t.tensor([0.9, 0.7, 0.5, -0.5, -0.7, -0.9])
    _segment, scores = _run_and_capture(
        monkeypatch,
        diffmean=diffmean,
        weight=0.5,
        logits=logits,
    )
    q = 2.0 * t.sigmoid(logits.max(dim=1).values.float()) - 1.0
    assert t.equal(scores, t.lerp(q, diffmean.unsqueeze(0), 0.5))


def test_raw_ranking_matches_both_legacy_transforms_on_unsaturated_reference() -> None:
    generator = t.Generator().manual_seed(1449)
    separated = t.linspace(-8.0, 8.0, 257)
    logits = t.stack([separated[t.randperm(257, generator=generator)] for _ in range(8)])
    k = t.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=t.int64)
    l = t.tensor([15, 13, 11, 9, 7, 5, 3, 1], dtype=t.int64)  # noqa: E741

    raw = select_topk_bottoml(logits, k, l)
    sigmoid = select_topk_bottoml(t.sigmoid(logits), k, l)
    q = select_topk_bottoml(2.0 * t.sigmoid(logits) - 1.0, k, l)

    for reference in (sigmoid, q):
        assert t.equal(raw.pos_ids, reference.pos_ids)
        assert t.equal(raw.neg_ids, reference.neg_ids)


def test_compact_head_rows_translate_to_original_feature_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vocab = FeatureVocab(n_sae_features=6, original_ids=(1, 3, 5))
    model = _model(feature_vocab=vocab)
    config = _config(t.tensor([0.8, 0.0, -0.8]), 1.0, max_k=2)
    provider = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
    state = build_test_self_cond_state(provider)
    original_build = provider._build_segment_inputs
    observed: list[dict[int, int]] = []

    def capture(segment: dict[int, int]):
        observed.append(dict(segment))
        return original_build(segment)

    monkeypatch.setattr(provider, "_build_segment_inputs", capture)
    monkeypatch.setattr(
        selfcond_module,
        "sample_action_counts",
        lambda *_args, **_kwargs: (t.tensor([1]), t.tensor([1])),
    )
    provider.step(state, t.zeros((1, 1, 3)))

    assert observed[-1] == {1: 0, 5: 1}


def test_zero_positive_and_negative_counts_fail_before_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SelfConditionedFeatureProvider(
        _model(),
        _config(t.linspace(0.8, -0.8, 6), 0.5),
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
        match="fp_logits_tail score interpolation selected zero positive and zero negative",
    ):
        provider.step(state, t.zeros((1, 1, 6)))


def test_local_generator_does_not_mutate_global_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    provider = SelfConditionedFeatureProvider(
        model,
        _config(t.linspace(0.8, -0.8, 6), 1.0),
        t.device("cpu"),
    )
    state = build_test_self_cond_state(provider)
    before = t.random.get_rng_state().clone()
    provider.step(state, t.zeros((1, 1, 6)))
    after = t.random.get_rng_state()
    assert t.equal(before, after)


def test_same_selection_seed_draws_same_bucket_cardinalities_at_every_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    diffmean = t.linspace(0.8, -0.8, 6)
    logits = t.tensor([[[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]]])
    observed_counts: list[tuple[int, int]] = []

    for weight in (0.0, 1.0 / 6.0, 0.5, 1.0):
        provider = SelfConditionedFeatureProvider(
            model,
            _config(diffmean, weight, seed=917),
            t.device("cpu"),
        )
        state = build_test_self_cond_state(provider)
        original_build = provider._build_segment_inputs

        def capture(segment: dict[int, int], build=original_build):
            observed_counts.append(
                (
                    sum(bucket == 0 for bucket in segment.values()),
                    sum(bucket == 1 for bucket in segment.values()),
                )
            )
            return build(segment)

        monkeypatch.setattr(provider, "_build_segment_inputs", capture)
        provider.step(state, logits)

    assert len(observed_counts) == 4
    assert len(set(observed_counts)) == 1


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
def test_diffmean_target_rejects_invalid_shape_or_values(
    scores: t.Tensor,
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        SelfCondScoreInterpolation(
            target_scores=scores,
            weight=UnitInterval.parse(0.5),
            negative_selection=_negative_selection(),
        )


def test_diffmean_target_is_detached_contiguous_cpu_snapshot() -> None:
    source = t.linspace(-0.8, 0.8, 8, requires_grad=True)[::2]
    interpolation = SelfCondScoreInterpolation(
        target_scores=source,
        weight=UnitInterval.parse(0.5),
        negative_selection=_negative_selection(),
    )
    source.detach().fill_(0.0)
    assert interpolation.target_scores.device.type == "cpu"
    assert interpolation.target_scores.dtype == t.float32
    assert interpolation.target_scores.is_contiguous()
    assert not interpolation.target_scores.requires_grad
    assert not t.equal(interpolation.target_scores, source.detach())


def test_interpolation_rejects_simultaneous_legacy_bias() -> None:
    base = _config(t.zeros(6), 0.5)
    with pytest.raises(ValueError, match="cannot combine fp_logit_bias"):
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
