from __future__ import annotations

import math

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
from steering.eval.axbench_bpb_adv.optimize import objective_gradient, objective_value


def test_response_margin_alpha_one_is_old_bpb_objective_exactly() -> None:
    positive = 2.75
    negative = -0.125
    positive_gradient = t.tensor([[1.0, -3.0]], dtype=t.float32)
    negative_gradient = t.tensor([[4.0, 2.0]], dtype=t.float32)

    assert response_margin_value(negative, positive, alpha=1.0) == objective_value(
        loss_desired=positive,
        loss_contrast=negative,
    )
    assert t.equal(
        response_margin_gradient(negative_gradient, positive_gradient, alpha=1.0),
        objective_gradient(
            grad_loss_desired=positive_gradient,
            grad_loss_contrast=negative_gradient,
        ),
    )


@pytest.mark.parametrize("alpha", [0.0, 0.01])
def test_response_margin_weights_negative_and_positive_slots(alpha: float) -> None:
    assert response_margin_value(7.0, 3.0, alpha=alpha) == 7.0 - alpha * 3.0
    result = response_margin_gradient(
        t.tensor([7.0]),
        t.tensor([3.0]),
        alpha=alpha,
    )
    assert result.item() == pytest.approx(7.0 - alpha * 3.0)


def test_weighted_bpb_matches_hand_formula_and_keeps_gradient() -> None:
    losses = t.tensor([1.0, 3.0], requires_grad=True)
    weights = t.tensor([1.0, 2.0])
    result = weighted_bpb(losses, weights, tensor_ln2=False)
    assert result.item() == pytest.approx(7.0 / (3.0 * math.log(2.0)))
    result.backward()
    assert losses.grad is not None
    assert t.allclose(losses.grad, weights / (weights.sum() * math.log(2.0)))


def test_activation_l2_reduces_norm_then_tokens_and_detaches_clean() -> None:
    attacked = t.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    clean = t.tensor([[1.0, 0.0], [2.0, 4.0]], requires_grad=True)
    direction = t.tensor([1.0, -1.0])
    # Deltas are [1, 1] and [2, -1], with distinct tokenwise L2 norms.
    row = activation_l2_row(attacked, clean, direction)
    assert row.item() == pytest.approx((math.sqrt(2.0) + math.sqrt(5.0)) / 2.0)
    pair = activation_l2_pair(row, t.tensor(2.5))
    assert pair.item() == pytest.approx((row.item() + 2.5) / 2.0)
    pair.backward()
    assert attacked.grad is not None
    assert clean.grad is None


def test_activation_l2_identity_excludes_alpha() -> None:
    first = AxBenchAdvGcgConfig(objective=GcgObjective.ACTIVATION_L2, alpha=0.0)
    second = AxBenchAdvGcgConfig(objective=GcgObjective.ACTIVATION_L2, alpha=17.0)
    assert first.effective_identity() == second.effective_identity()
    assert "alpha" not in first.effective_identity()
