"""Independent mathematical oracles for structured-SFT GRPO actions."""

from __future__ import annotations

import itertools
import math

import pytest
import torch as t
import torch.nn.functional as F

import steering.conditional.encoder_decoder.sft_grpo as sft_grpo_module
from steering.conditional.encoder_decoder.sft_grpo import (
    ActionMode,
    SFTGRPOError,
    SparseActionTrace,
    assert_behavior_replay_equal,
    behavior_checked_replay_ratio,
    bernoulli_current_to_reference_kl,
    conversation_group_advantages,
    masked_mean_token_ce,
    policy_gradient_loss,
    replay_ratio,
    sample_bernoulli_topk_bottomk,
    sample_weighted_without_replacement,
    trace_to_conditioning,
    validate_count_support,
    weighted_ordered_log_likelihood,
)


def _weighted_trace(positive: tuple[int, ...], negative: tuple[int, ...]) -> SparseActionTrace:
    return SparseActionTrace(
        mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        requested_positive=len(positive),
        requested_negative=len(negative),
        positive_rows=positive,
        negative_rows=negative,
        behavior_log_terms=(0.0,),
    )


@t.no_grad()
def _synchronized_weighted_sampler_reference(
    logits: t.Tensor,
    *,
    positive_count: int,
    negative_count: int,
    generator: t.Generator,
) -> SparseActionTrace:
    logits_f32 = logits.float()
    positive_log_weights = F.logsigmoid(logits_f32)
    negative_log_weights = F.logsigmoid(-logits_f32)
    available = t.ones(logits.numel(), dtype=t.bool, device=logits.device)
    positive_rows = []
    negative_rows = []
    log_terms = []
    for log_weights, count, rows in (
        (positive_log_weights, positive_count, positive_rows),
        (negative_log_weights, negative_count, negative_rows),
    ):
        for _ in range(count):
            masked = log_weights.masked_fill(~available, -t.inf)
            denominator = t.logsumexp(masked, dim=0)
            selected = int(
                t.multinomial(
                    t.exp(masked - denominator),
                    1,
                    generator=generator,
                ).item()
            )
            rows.append(selected)
            log_terms.append(float((log_weights[selected] - denominator).item()))
            available[selected] = False
    return SparseActionTrace(
        mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        requested_positive=positive_count,
        requested_negative=negative_count,
        positive_rows=tuple(positive_rows),
        negative_rows=tuple(negative_rows),
        behavior_log_terms=(sum(log_terms),),
    )


def _bernoulli_trace(
    logits: t.Tensor,
    positive: tuple[int, ...],
    negative: tuple[int, ...],
) -> SparseActionTrace:
    terms = t.cat(
        (
            F.logsigmoid(logits.detach().float()[list(positive)]),
            F.logsigmoid(-logits.detach().float()[list(negative)]),
        )
    )
    return SparseActionTrace(
        mode=ActionMode.BERNOULLI_TOPK_BOTTOMK,
        requested_positive=len(positive),
        requested_negative=len(negative),
        positive_rows=positive,
        negative_rows=negative,
        behavior_log_terms=tuple(float(value) for value in terms),
    )


def test_weighted_ordered_action_mass_sums_to_one_exhaustively() -> None:
    logits = t.tensor([-1.2, 0.3, 2.1])
    mass = t.zeros(())
    for positive in range(3):
        for negative in range(3):
            if negative == positive:
                continue
            mass = (
                mass
                + weighted_ordered_log_likelihood(
                    logits,
                    _weighted_trace((positive,), (negative,)),
                ).exp()
            )
    assert float(mass) == pytest.approx(1.0, abs=1e-6)


def test_weighted_likelihood_matches_hand_sequential_reference() -> None:
    logits = t.tensor([-0.7, 0.2, 1.3, 2.0], requires_grad=True)
    trace = _weighted_trace((2, 0), (1,))
    positive = F.logsigmoid(logits)
    negative = F.logsigmoid(-logits)
    expected = (
        positive[2]
        - t.logsumexp(positive, 0)
        + positive[0]
        - t.logsumexp(positive[[0, 1, 3]], 0)
        + negative[1]
        - t.logsumexp(negative[[1, 3]], 0)
    )
    actual = weighted_ordered_log_likelihood(logits, trace)
    assert float(actual.detach()) == pytest.approx(float(expected.detach()), abs=1e-7)
    actual.backward()
    assert logits.grad is not None
    assert bool(t.isfinite(logits.grad).all())


@pytest.mark.parametrize(
    ("positive", "negative", "match"),
    [
        ((1, 1), (), "positive trace"),
        ((4,), (), "positive trace"),
        ((1,), (1,), "negative trace"),
        ((), (2, 2), "negative trace"),
        ((), (-1,), "negative trace"),
    ],
)
def test_weighted_likelihood_validates_sparse_rows_before_device_scoring(
    positive: tuple[int, ...],
    negative: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(SFTGRPOError, match=match):
        weighted_ordered_log_likelihood(
            t.tensor([-0.7, 0.2, 1.3, 2.0]),
            _weighted_trace(positive, negative),
        )


def test_public_weighted_math_still_rejects_nonfinite_logits() -> None:
    with pytest.raises(SFTGRPOError, match="action logits must all be finite"):
        sample_weighted_without_replacement(
            t.tensor([0.0, t.nan]),
            positive_count=1,
            negative_count=0,
            generator=t.Generator().manual_seed(1),
        )
    with pytest.raises(SFTGRPOError, match="action logits must all be finite"):
        weighted_ordered_log_likelihood(
            t.tensor([0.0, t.inf]),
            _weighted_trace((0,), ()),
        )


def test_weighted_sampler_is_disjoint_ordered_and_replays_at_one() -> None:
    logits = t.tensor([-100.0, -2.0, 0.0, 2.0, 100.0])
    generator = t.Generator().manual_seed(91)
    trace = sample_weighted_without_replacement(
        logits,
        positive_count=2,
        negative_count=2,
        generator=generator,
    )
    assert len(trace.positive_rows) == 2
    assert len(trace.negative_rows) == 2
    assert len({*trace.positive_rows, *trace.negative_rows}) == 4
    assert_behavior_replay_equal(logits, trace, atol=2e-6, rtol=2e-6)
    assert replay_ratio(logits, trace) == pytest.approx(1.0, abs=2e-6)


@pytest.mark.parametrize(("positive_count", "negative_count"), [(0, 0), (3, 0), (0, 3), (3, 2)])
@pytest.mark.parametrize("seed", [0, 91, 2**31 - 1])
def test_weighted_sampler_preserves_synchronized_draw_and_log_term_sequence(
    positive_count: int,
    negative_count: int,
    seed: int,
) -> None:
    logits = t.tensor([-100.0, -5.0, -1.0, 0.0, 0.4, 3.0, 100.0])
    expected = _synchronized_weighted_sampler_reference(
        logits,
        positive_count=positive_count,
        negative_count=negative_count,
        generator=t.Generator().manual_seed(seed),
    )
    actual = sample_weighted_without_replacement(
        logits,
        positive_count=positive_count,
        negative_count=negative_count,
        generator=t.Generator().manual_seed(seed),
    )
    assert actual == expected


def test_batched_weighted_log_weights_preserve_rows_traces_likelihoods_and_gradients() -> None:
    logits = t.tensor(
        [
            [-8.0, -1.2, 0.0, 0.4, 2.0, 9.0],
            [3.0, -4.0, 1.1, -0.7, 0.2, 5.0],
        ],
        requires_grad=True,
    )
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    expected_positive = t.stack([F.logsigmoid(row) for row in logits])
    expected_negative = t.stack([F.logsigmoid(-row) for row in logits])
    assert t.equal(positive, expected_positive)
    assert t.equal(negative, expected_negative)

    traces = []
    for row, seed in enumerate((17, 93)):
        expected = sample_weighted_without_replacement(
            logits[row].detach(),
            positive_count=2,
            negative_count=2,
            generator=t.Generator().manual_seed(seed),
        )
        actual = sft_grpo_module._sample_weighted_without_replacement_from_log_weights(
            positive[row].detach(),
            negative[row].detach(),
            positive_count=2,
            negative_count=2,
            generator=t.Generator().manual_seed(seed),
        )
        assert actual == expected
        traces.append(actual)

    batched_terms = sft_grpo_module._weighted_ordered_log_likelihood_batch_from_log_weights(
        logits,
        positive,
        negative,
        tuple(traces),
    )
    batched_total = batched_terms.sum()
    batched_gradient = t.autograd.grad(batched_total, logits)[0]

    rowwise_logits = logits.detach().clone().requires_grad_(True)
    rowwise_total = sum(
        (
            weighted_ordered_log_likelihood(rowwise_logits[row], trace)
            for row, trace in enumerate(traces)
        ),
        start=rowwise_logits.sum() * 0.0,
    )
    rowwise_gradient = t.autograd.grad(rowwise_total, rowwise_logits)[0]
    rowwise_terms = t.stack(
        [weighted_ordered_log_likelihood(logits[row], trace) for row, trace in enumerate(traces)]
    )
    # The batched scorer groups selected and never-selected rows before its
    # reductions, so it is mathematically identical but may differ by a few
    # FP32 ulps from the draw-by-draw oracle.
    t.testing.assert_close(batched_terms, rowwise_terms, atol=2e-6, rtol=1e-7)
    t.testing.assert_close(
        batched_total.detach(),
        rowwise_total.detach(),
        atol=2e-6,
        rtol=1e-7,
    )
    t.testing.assert_close(batched_gradient, rowwise_gradient, atol=2e-6, rtol=1e-6)


def test_rng_v2_batched_weighted_sampler_is_deterministic_disjoint_and_replayable() -> None:
    logits = t.tensor(
        [
            [-3.0, -1.0, 0.0, 0.4, 1.5, 4.0],
            [2.0, -2.0, 0.7, -0.4, 3.0, 0.1],
            [-0.8, 1.2, 2.4, -3.0, 0.5, -0.1],
        ]
    )
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    kwargs = {
        "positive_counts": (2, 1, 3),
        "negative_counts": (1, 2, 0),
    }
    first = sft_grpo_module._sample_weighted_without_replacement_batch_from_log_weights(
        positive,
        negative,
        generator=t.Generator().manual_seed(903),
        **kwargs,
    )
    second = sft_grpo_module._sample_weighted_without_replacement_batch_from_log_weights(
        positive,
        negative,
        generator=t.Generator().manual_seed(903),
        **kwargs,
    )

    assert first == second
    for row, trace in enumerate(first):
        assert len(trace.positive_rows) == kwargs["positive_counts"][row]
        assert len(trace.negative_rows) == kwargs["negative_counts"][row]
        assert len({*trace.positive_rows, *trace.negative_rows}) == trace.selected_count
        assert_behavior_replay_equal(logits[row], trace, atol=2e-6, rtol=2e-6)


def test_rng_v2_batched_weighted_sampler_launches_once_per_batch_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = t.multinomial

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(t, "multinomial", counted)
    logits = t.arange(24, dtype=t.float32).reshape(4, 6) / 7
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    traces = sft_grpo_module._sample_weighted_without_replacement_batch_from_log_weights(
        positive,
        negative,
        positive_counts=(3, 1, 2, 1),
        negative_counts=(1, 2, 0, 1),
        generator=t.Generator().manual_seed(77),
    )

    assert calls == 5
    assert len(traces) == 4


def test_rng_v2_batched_weighted_sampler_handles_inactive_exhausted_rows() -> None:
    logits = t.tensor(
        [
            [-3.0, -1.0, 0.0, 0.4, 1.5, 4.0],
            [2.0, -2.0, 0.7, -0.4, 3.0, 0.1],
        ]
    )
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    traces = sft_grpo_module._sample_weighted_without_replacement_batch_from_log_weights(
        positive,
        negative,
        positive_counts=(6, 0),
        negative_counts=(0, 6),
        generator=t.Generator().manual_seed(501),
    )

    assert set(traces[0].positive_rows) == set(range(6))
    assert set(traces[1].negative_rows) == set(range(6))
    for row, trace in enumerate(traces):
        assert_behavior_replay_equal(logits[row], trace, atol=2e-6, rtol=2e-6)


def test_rng_v2_batched_weighted_sampler_matches_target_marginal_distribution() -> None:
    logits_one = t.tensor([-1.4, -0.2, 0.9, 2.1])
    samples = 20_000
    logits = logits_one.expand(samples, -1)
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    traces = sft_grpo_module._sample_weighted_without_replacement_batch_from_log_weights(
        positive,
        negative,
        positive_counts=(1,) * samples,
        negative_counts=(0,) * samples,
        generator=t.Generator().manual_seed(20260828),
    )
    observed = (
        t.bincount(
            t.tensor([trace.positive_rows[0] for trace in traces]),
            minlength=logits_one.numel(),
        ).float()
        / samples
    )
    expected = t.softmax(F.logsigmoid(logits_one), dim=0)
    t.testing.assert_close(observed, expected, atol=0.012, rtol=0)


def test_batched_weighted_likelihood_handles_inactive_full_vocabulary_rows() -> None:
    logits = t.tensor(
        [
            [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0],
            [2.0, 1.0, 0.5, 0.0, -1.0, -2.0],
        ],
        requires_grad=True,
    )
    traces = (
        _weighted_trace((0, 1, 2, 3, 4), (5,)),
        _weighted_trace((0,), (1, 2, 3, 4, 5)),
    )
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    actual = sft_grpo_module._weighted_ordered_log_likelihood_batch_from_log_weights(
        logits,
        positive,
        negative,
        traces,
    )
    expected = t.stack(
        [weighted_ordered_log_likelihood(logits[row], trace) for row, trace in enumerate(traces)]
    )
    actual_gradient = t.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = t.autograd.grad(expected.sum(), logits)[0]
    t.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-7)
    t.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=1e-6)
    assert bool(t.isfinite(actual_gradient).all())


def test_batched_weighted_likelihood_is_stable_after_extreme_dominant_rows() -> None:
    logits = t.tensor(
        [
            [1000.0, -1000.0, -500.0, 0.0],
            [-1000.0, 1000.0, 500.0, 0.0],
        ],
        requires_grad=True,
    )
    traces = (
        _weighted_trace((0, 3), (1,)),
        _weighted_trace((1, 2), (0,)),
    )
    positive, negative = sft_grpo_module._weighted_log_weights_from_finite_forward(logits)
    actual = sft_grpo_module._weighted_ordered_log_likelihood_batch_from_log_weights(
        logits,
        positive,
        negative,
        traces,
    )
    expected = t.stack(
        [weighted_ordered_log_likelihood(logits[row], trace) for row, trace in enumerate(traces)]
    )
    actual_gradient = t.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = t.autograd.grad(expected.sum(), logits)[0]
    t.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-7)
    t.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=1e-6)
    assert bool(t.isfinite(actual).all())
    assert bool(t.isfinite(actual_gradient).all())


def test_weighted_sampler_updates_identical_masked_log_weight_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sft_grpo_module._draw_log_weighted_row
    unavailable_rows_by_draw = []

    def captured(log_weights, available_log_weights, *, generator):
        unavailable_rows_by_draw.append(
            tuple(int(row) for row in t.isneginf(available_log_weights).nonzero().flatten())
        )
        return original(
            log_weights,
            available_log_weights,
            generator=generator,
        )

    monkeypatch.setattr(sft_grpo_module, "_draw_log_weighted_row", captured)
    trace = sample_weighted_without_replacement(
        t.tensor([-2.0, -0.5, 0.0, 0.7, 2.0]),
        positive_count=2,
        negative_count=2,
        generator=t.Generator().manual_seed(43),
    )

    positive_0, positive_1 = trace.positive_rows
    negative_0, _negative_1 = trace.negative_rows
    assert unavailable_rows_by_draw == [
        (),
        (positive_0,),
        tuple(sorted((positive_0, positive_1))),
        tuple(sorted((positive_0, positive_1, negative_0))),
    ]


def test_behavior_checked_ratio_reuses_one_weighted_likelihood_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = t.tensor([-1.1, -0.2, 0.7, 1.4], requires_grad=True)
    trace = sample_weighted_without_replacement(
        logits.detach(),
        positive_count=2,
        negative_count=1,
        generator=t.Generator().manual_seed(37),
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference = replay_ratio(reference_logits, trace)
    reference.backward()
    assert reference_logits.grad is not None

    original = sft_grpo_module.weighted_ordered_log_likelihood
    calls = 0

    def counted(candidate_logits: t.Tensor, candidate_trace: SparseActionTrace) -> t.Tensor:
        nonlocal calls
        calls += 1
        return original(candidate_logits, candidate_trace)

    monkeypatch.setattr(sft_grpo_module, "weighted_ordered_log_likelihood", counted)
    actual = behavior_checked_replay_ratio(logits, trace, atol=2e-6, rtol=2e-6)
    actual.backward()
    assert calls == 1
    assert actual.detach() == reference.detach()
    assert logits.grad is not None
    t.testing.assert_close(logits.grad, reference_logits.grad, atol=0, rtol=0)


def test_behavior_checked_ratio_fails_before_returning_a_mismatched_action() -> None:
    logits = t.tensor([-0.4, 0.8])
    trace = SparseActionTrace(
        mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        requested_positive=1,
        requested_negative=0,
        positive_rows=(1,),
        negative_rows=(),
        behavior_log_terms=(100.0,),
    )
    with pytest.raises(SFTGRPOError, match="behavior/replay log terms disagree"):
        behavior_checked_replay_ratio(logits, trace, atol=0, rtol=0)


def test_weighted_monte_carlo_matches_exact_small_space() -> None:
    logits = t.tensor([-0.5, 0.7, 1.1])
    traces = [
        _weighted_trace((positive,), (negative,))
        for positive, negative in itertools.permutations(range(3), 2)
    ]
    expected = {
        (trace.positive_rows[0], trace.negative_rows[0]): float(
            weighted_ordered_log_likelihood(logits, trace).exp()
        )
        for trace in traces
    }
    counts = dict.fromkeys(expected, 0)
    generator = t.Generator().manual_seed(123)
    samples = 20_000
    for _ in range(samples):
        sampled = sample_weighted_without_replacement(
            logits,
            positive_count=1,
            negative_count=1,
            generator=generator,
        )
        counts[(sampled.positive_rows[0], sampled.negative_rows[0])] += 1
    for action, probability in expected.items():
        assert counts[action] / samples == pytest.approx(probability, abs=0.012)


def test_bernoulli_ties_use_ascending_original_id_and_shortages_do_not_refill() -> None:
    logits = t.tensor([100.0, 100.0, -100.0, -100.0, -100.0, -100.0])
    original_ids = t.tensor([9, 3, 8, 2, 7, 6], dtype=t.int64)
    trace = sample_bernoulli_topk_bottomk(
        logits,
        positive_count=3,
        negative_count=3,
        generator=t.Generator().manual_seed(2),
        original_feature_ids=original_ids,
    )
    assert trace.positive_rows == (1, 0)
    assert trace.negative_rows == (3, 5, 4)
    assert trace.requested_positive == 3
    assert trace.requested_negative == 3
    assert set(trace.positive_rows).isdisjoint(trace.negative_rows)


def test_bernoulli_probability_point_nine_is_rejected_about_ten_percent() -> None:
    logits = t.tensor([math.log(9.0)])
    generator = t.Generator().manual_seed(2026)
    rejected = 0
    samples = 10_000
    for _ in range(samples):
        trace = sample_bernoulli_topk_bottomk(
            logits,
            positive_count=1,
            negative_count=0,
            generator=generator,
        )
        rejected += int(not trace.positive_rows)
    assert rejected / samples == pytest.approx(0.1, abs=0.012)


@pytest.mark.parametrize(
    ("positive", "negative", "row", "expected"),
    [((0,), (), 0, lambda probability: 1.0 - probability), ((), (1,), 1, lambda p: -p)],
)
def test_bernoulli_self_ratio_has_documented_selected_gradient(
    positive: tuple[int, ...],
    negative: tuple[int, ...],
    row: int,
    expected,
) -> None:
    logits = t.tensor([-0.4, 0.8, 1.7], requires_grad=True)
    trace = _bernoulli_trace(logits, positive, negative)
    ratio = replay_ratio(logits, trace)
    assert float(ratio.detach()) == pytest.approx(1.0, abs=1e-7)
    ratio.backward()
    assert logits.grad is not None
    probability = float(t.sigmoid(logits.detach()[row]))
    assert float(logits.grad[row]) == pytest.approx(expected(probability), abs=1e-6)
    unselected = [index for index in range(3) if index != row]
    assert logits.grad[unselected].tolist() == [0.0] * len(unselected)


def test_empty_bernoulli_action_is_graph_connected_ratio_one_with_zero_gradient() -> None:
    logits = t.tensor([-1.0, 2.0], requires_grad=True)
    trace = _bernoulli_trace(logits, (), ())
    ratio = replay_ratio(logits, trace)
    assert ratio.item() == 1.0
    ratio.backward()
    assert logits.grad is not None
    assert logits.grad.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(("positive", "negative"), [((0,), (2,)), ((), ())])
def test_behavior_checked_ratio_matches_bernoulli_replay(
    positive: tuple[int, ...],
    negative: tuple[int, ...],
) -> None:
    logits = t.tensor([-0.8, 0.1, 1.3], requires_grad=True)
    trace = _bernoulli_trace(logits, positive, negative)
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference = replay_ratio(reference_logits, trace)
    reference.backward()
    actual = behavior_checked_replay_ratio(logits, trace, atol=0, rtol=0)
    actual.backward()
    assert actual.detach() == reference.detach()
    assert logits.grad is not None
    assert reference_logits.grad is not None
    t.testing.assert_close(logits.grad, reference_logits.grad, atol=0, rtol=0)


def test_group_normalization_is_population_local_and_zero_variance_exact() -> None:
    rewards = t.tensor([[1.0, 2.0, 3.0], [7.0, 7.0, 7.0]], requires_grad=True)
    advantages = conversation_group_advantages(rewards)
    expected = t.tensor([[-1.2247449, 0.0, 1.2247449], [0.0, 0.0, 0.0]])
    assert t.allclose(advantages, expected, atol=1e-6, rtol=0)
    assert not advantages.requires_grad
    assert advantages[0].mean().item() == pytest.approx(0.0, abs=1e-7)


def test_on_policy_pg_forward_is_zero_but_gradient_is_nonzero() -> None:
    logits = t.tensor([[0.2, -0.1], [0.2, -0.1]], requires_grad=True)
    traces = [
        _bernoulli_trace(logits[0], (0,), ()),
        _bernoulli_trace(logits[1], (1,), ()),
    ]
    ratios = t.stack(
        [replay_ratio(logits[index], trace) for index, trace in enumerate(traces)]
    ).reshape(1, 2)
    advantages = t.tensor([[-1.0, 1.0]])
    loss = policy_gradient_loss(ratios, advantages)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0


def test_detached_reward_uses_direct_mask_without_a_second_shift() -> None:
    ce = t.tensor([[1.0, 10.0, 3.0, 20.0]], requires_grad=True)
    mean_ce = masked_mean_token_ce(
        ce,
        t.tensor([[True, False, True, False]]),
    )
    assert mean_ce.item() == 2.0
    assert mean_ce.requires_grad


def test_all_feature_kl_matches_torch_distributions() -> None:
    current = t.tensor([-2.0, 0.0, 1.7], requires_grad=True)
    reference = t.tensor([0.5, -0.3, 1.2], requires_grad=True)
    actual = bernoulli_current_to_reference_kl(current, reference)
    expected = t.distributions.kl_divergence(
        t.distributions.Bernoulli(logits=current),
        t.distributions.Bernoulli(logits=reference.detach()),
    ).mean()
    assert float(actual.detach()) == pytest.approx(float(expected.detach()), abs=2e-7)
    actual.backward()
    assert current.grad is not None
    assert reference.grad is None


def test_count_support_and_encoder_boundary_mapping_are_strict() -> None:
    with pytest.raises(SFTGRPOError, match="zero total"):
        validate_count_support(
            positive_min=0,
            positive_max_exclusive=2,
            negative_min=0,
            negative_max_exclusive=2,
            feature_count=4,
        )
    validate_count_support(
        positive_min=1,
        positive_max_exclusive=3,
        negative_min=0,
        negative_max_exclusive=2,
        feature_count=4,
    )
    trace = _weighted_trace((2, 0), (1,))
    action = trace_to_conditioning(
        trace,
        compact_to_original=t.tensor([10, 30, 20], dtype=t.int64),
    )
    assert action.feature_ids == (20, 10, 30)
    assert action.magnitudes == (0, 0, 1)
    assert action.mask == (True, True, True)
    assert (
        trace_to_conditioning(
            trace,
            compact_to_original=(10, 30, 20),
        )
        == action
    )
