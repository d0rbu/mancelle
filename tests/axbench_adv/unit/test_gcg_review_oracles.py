"""Independent numerical oracles for the approved AxBench-Adv objectives.

These examples distinguish the scientific choices (class sign, token weighting,
norm versus squared norm, and reference detachment), rather than deriving the
expected result by calling the implementation under test.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import pytest
import torch as t

from scripts.eval.axbench_adv.gcg_math import (
    activation_l2_pair,
    activation_l2_row,
    response_margin_gradient,
    response_margin_value,
    weighted_bpb,
)
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective
from steering.eval.axbench_adv.gcg_driver import AppendSuffixEditSpace, GcgRuntime
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS
from steering.eval.axbench_adv.worker_client import GcgFilteredCandidate, GcgPrepared
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.optimize import (
    TokenTriggerEditSpace,
    objective_gradient,
    objective_value,
)


@pytest.mark.parametrize("alpha", [0.0, 0.01, 0.5, 1.0, 2.0])
def test_response_margin_sign_and_autograd(alpha: float) -> None:
    # The unequal derivatives expose accidental weighting of the negative class.
    suffix = t.tensor([2.0, -3.0], requires_grad=True)
    negative = 2 * suffix[0] + 7 * suffix[1]
    positive = 11 * suffix[0] - 5 * suffix[1]
    loss = response_margin_value(negative, positive, alpha=alpha)
    assert isinstance(loss, t.Tensor)
    expected = -17.0 - alpha * 37.0
    assert loss.item() == pytest.approx(expected)
    gradient = t.autograd.grad(loss, suffix)[0]
    expected_gradient = t.tensor([2.0 - 11.0 * alpha, 7.0 + 5.0 * alpha])
    t.testing.assert_close(gradient, expected_gradient, rtol=0, atol=1e-6)
    assert t.equal(
        gradient,
        response_margin_gradient(t.tensor([2.0, 7.0]), t.tensor([11.0, -5.0]), alpha=alpha),
    )


@pytest.mark.parametrize("dtype", [t.float32, t.float64, t.bfloat16])
def test_alpha_one_exactly_matches_existing_bpb_objective(dtype: t.dtype) -> None:
    generator = t.Generator().manual_seed(9821)
    positive_gradient = t.randn(7, 13, generator=generator, dtype=dtype)
    negative_gradient = t.randn(7, 13, generator=generator, dtype=dtype)
    assert t.equal(
        response_margin_gradient(negative_gradient, positive_gradient, alpha=1.0),
        objective_gradient(
            grad_loss_desired=positive_gradient,
            grad_loss_contrast=negative_gradient,
        ),
    )
    for positive, negative in [(3.25, 11.5), (1e-9, 1e5), (-8.25, 4.0)]:
        assert response_margin_value(negative, positive, alpha=1.0) == objective_value(
            loss_desired=positive, loss_contrast=negative
        )


@pytest.mark.parametrize("dtype", [t.float32, t.float16, t.bfloat16])
@pytest.mark.parametrize("tensor_ln2", [False, True])
def test_bpb_reduction_exactly_preserves_pre_extraction_formula(
    dtype: t.dtype, tensor_ln2: bool
) -> None:
    # Frozen formula from main 34855b24; output bytes deliberately differ and
    # contain zero, distinguishing byte weighting from token averaging.
    losses = t.tensor([0.75, 8.5, 4.0, 9.125], dtype=dtype, requires_grad=True)
    weights = t.tensor([4, 0, 1, 7])
    old_losses = losses.detach().clone().requires_grad_(True)
    old_weights = weights.to(t.float32)
    old_ln2 = t.log(t.tensor(2.0)) if tensor_ln2 else math.log(2.0)
    old_result = (old_losses.float() * old_weights).sum() / (old_ln2 * old_weights.sum())
    result = weighted_bpb(losses, weights, tensor_ln2=tensor_ln2)
    assert t.equal(result, old_result)
    assert t.equal(t.autograd.grad(result, losses)[0], t.autograd.grad(old_result, old_losses)[0])
    # Independent scalar arithmetic additionally pins the actual normalization.
    assert result.item() == pytest.approx((0.75 * 4 + 4.0 + 9.125 * 7) / (12 * math.log(2)))


def test_activation_l2_uses_per_token_norms_and_positive_offset() -> None:
    clean = t.tensor([[10.0, -7.0], [3.0, 5.0]], requires_grad=True)
    direction = t.tensor([2.0, -1.0])
    attacked = t.tensor([[11.0, -2.0], [-5.0, -2.0]], requires_grad=True)
    # Deltas are (3, 4), (-6, -8), so mean L2 = (5+10)/2 = 7.5.
    # Squared L2 would be 62.5; norm of the mean would be 2.5.
    loss = activation_l2_row(attacked, clean, direction)
    assert loss.item() == 7.5
    grad_attack, grad_clean = t.autograd.grad(loss, (attacked, clean), allow_unused=True)
    t.testing.assert_close(grad_attack, t.tensor([[0.3, 0.4], [-0.3, -0.4]]))
    assert grad_clean is None
    # Precisely cancelling the positive offset must minimize the objective.
    cancelled = (clean.detach() - direction).requires_grad_(True)
    zero = activation_l2_row(cancelled, clean, direction)
    assert zero.item() == 0.0
    assert t.equal(t.autograd.grad(zero, cancelled)[0], t.zeros_like(cancelled))


def test_activation_l2_class_balance_is_independent_of_response_lengths() -> None:
    positive = activation_l2_row(t.tensor([[3.0, 4.0]]), t.zeros(1, 2), t.zeros(2))
    negative = activation_l2_row(
        t.tensor([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]), t.zeros(3, 2), t.zeros(2)
    )
    assert activation_l2_pair(positive, negative).item() == 7.5


@pytest.mark.parametrize("factor", ADV_FULL_FACTORS)
def test_activation_l2_gradient_matches_scalar_finite_differences(factor: float) -> None:
    attacked = t.tensor([[0.25, 1.5, -2.0], [3.0, -0.75, 0.5]], requires_grad=True)
    clean = t.tensor([[0.5, -0.25, 0.75], [0.125, -1.0, 2.0]])
    direction = (2.3 * factor) * t.tensor([0.25, -0.5, 1.0])
    actual = t.autograd.grad(activation_l2_row(attacked, clean, direction), attacked)[0]

    def scalar_oracle(values: list[list[float]]) -> float:
        return sum(
            math.sqrt(
                sum(
                    (value + float(direction[j]) - float(clean[i, j])) ** 2
                    for j, value in enumerate(row)
                )
            )
            for i, row in enumerate(values)
        ) / len(values)

    epsilon = 1e-4
    numeric = t.empty_like(attacked)
    for i in range(attacked.shape[0]):
        for j in range(attacked.shape[1]):
            plus, minus = attacked.tolist(), attacked.tolist()
            plus[i][j] += epsilon
            minus[i][j] -= epsilon
            numeric[i, j] = (scalar_oracle(plus) - scalar_oracle(minus)) / (2 * epsilon)
    t.testing.assert_close(actual, numeric, rtol=1e-6, atol=1e-7)


def test_activation_l2_alpha_has_no_effective_identity() -> None:
    zero = AxBenchAdvGcgConfig(objective=GcgObjective.ACTIVATION_L2, alpha=0.0)
    large = AxBenchAdvGcgConfig(objective=GcgObjective.ACTIVATION_L2, alpha=1000.0)
    assert zero.effective_identity() == large.effective_identity()
    assert "alpha" not in zero.effective_identity()
    assert zero.effective_identity()["direction_factor_policy"] == "selected_method_factor"
    assert "direction_factor_policy" not in AxBenchAdvGcgConfig().effective_identity()
    assert AxBenchAdvGcgConfig().alpha == 0.01
    assert (
        AxBenchAdvGcgConfig(alpha=0.0).effective_identity()
        != AxBenchAdvGcgConfig(alpha=1.0).effective_identity()
    )


class _CanonicalCandidates:
    """All candidates are canonical for a comparison of the two search adapters."""

    def gcg_filter(self, candidates: Sequence[Sequence[int]]) -> tuple[GcgFilteredCandidate, ...]:
        return tuple(GcgFilteredCandidate(tuple(ids), repr(ids), True, None) for ids in candidates)


@pytest.mark.parametrize("k", [1, 2, 32])
@pytest.mark.parametrize("distance_weight", [0.0, 10.0])
def test_new_edit_space_exactly_matches_bpb_ranking_sampling_and_exhaustion(
    k: int, distance_weight: float
) -> None:
    generator = t.Generator().manual_seed(332)
    embedding = t.randn(37, 11, generator=generator)
    gradient = t.randn(k, 11, generator=generator)
    current = list(range(1, k + 1))
    allowed = tuple(range(1, 37))
    cfg = FasterGcgConfig(top_k=4, distance_weight=distance_weight)
    old = TokenTriggerEditSpace(
        tokenizer=SimpleNamespace(bos_token_id=0),
        cfg=cfg,
        embedding_weight=embedding,
        allowed_token_ids=allowed,
    )
    prepared = GcgPrepared(
        embedding_weight=embedding,
        allowed_token_ids=allowed,
        seed_token_ids=tuple(current),
        member_count=3,
        identity={"rank_sentinel_id": 0},
        layouts_sha256="a" * 64,
        direction_calibration=None,
    )
    new = AppendSuffixEditSpace(cast(GcgRuntime, _CanonicalCandidates()), prepared)
    old_proposals = old.rank_candidates(gradient, current, cfg)
    new_proposals = new.rank_candidates(gradient, current, cfg)
    assert old_proposals == new_proposals
    assert 1 in new_proposals  # The first suffix coordinate must remain editable.
    assert 0 not in new_proposals
    assert old.digest(current) == new.digest(current)
    old_visited, new_visited = {old.digest(current)}, {new.digest(current)}
    for step in range(5):
        old_candidates = old.sample_candidates(
            old_proposals,
            current,
            batch_size=20,
            temperature=0.1,
            rng=random.Random(step + 4),
            visited=old_visited,
        )
        new_candidates = new.sample_candidates(
            new_proposals,
            current,
            batch_size=20,
            temperature=0.1,
            rng=random.Random(step + 4),
            visited=new_visited,
        )
        assert new_candidates == old_candidates
        assert new_visited == old_visited
        assert new.full_batch_pool_exhausted(
            new_proposals, current, new_visited
        ) == old.full_batch_pool_exhausted(old_proposals, current, old_visited)
    # Exhaustion must include every coordinate, even the last coordinate at k32.
    for position, replacements in new_proposals.items():
        for replacement in replacements:
            candidate = current.copy()
            candidate[position - 1] = replacement.token_id
            new_visited.add(new.digest(candidate))
            old_visited.add(old.digest(candidate))
    assert new.full_batch_pool_exhausted(new_proposals, current, new_visited)
    assert old.full_batch_pool_exhausted(old_proposals, current, old_visited)
