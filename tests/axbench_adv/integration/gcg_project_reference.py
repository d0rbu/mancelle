"""Check project GCG against original projections, guard, and response decoding."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import torch as t

from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_adv.self_conditioning import _BoundSelfConditioningCell
from tests.axbench_adv.gcg_factor_reference import l2_with_expected_offset
from tests.axbench_adv.gcg_gradient_reference import full_sequence_logits
from tests.axbench_adv.gcg_state_reference import (
    preserve_gcg_model_state as reference_model_state,
)


@contextmanager
def verify_project_l2_numerics() -> Iterator[None]:
    original = _BoundSelfConditioningCell._project_batch_values
    calls: list[bool] = []

    def batch(cell: _BoundSelfConditioningCell, items: Any, *, gradient: bool) -> Any:
        if cell._gcg_direction is None:
            for row, _tokens in items:
                assert row.response_byte_weights is not None
                assert row.response_byte_weights.device == cell._runtime._target_model.get_device()
        actual = original(cell, items, gradient=gradient)
        reference_items = [
            (
                replace(
                    row,
                    response_byte_weights=t.tensor(
                        [
                            len(cell._runtime._target_tokenizer.decode([token]).encode("utf-8"))
                            for token in row.response
                        ],
                        dtype=t.float32,
                    ),
                )
                if cell._gcg_direction is None
                else row,
                tokens,
            )
            for row, tokens in items
        ]

        def independent_loss(attacked: t.Tensor, clean: t.Tensor, observed: t.Tensor) -> t.Tensor:
            direction = cell._gcg_direction
            assert direction is not None
            expected = (direction.max_activation * cell._factor) * direction.direction.to(
                attacked.dtype
            )
            return l2_with_expected_offset(attacked, clean, observed, expected=expected)

        with (
            full_sequence_logits(cell._runtime._target_model),
            patch.object(self_conditioning, "preserve_gcg_model_state", reference_model_state),
            patch.object(self_conditioning, "activation_l2_row", independent_loss),
        ):
            reference = original(cell, reference_items, gradient=gradient)
        for (value, derivative), (expected, expected_derivative) in zip(
            actual, reference, strict=True
        ):
            t.testing.assert_close(value, expected, rtol=0, atol=0)
            if derivative is None:
                assert expected_derivative is None
            else:
                assert expected_derivative is not None
                t.testing.assert_close(derivative, expected_derivative, rtol=0, atol=0)
        calls.append(gradient)
        return actual

    with patch.object(_BoundSelfConditioningCell, "_project_batch_values", batch):
        yield
    assert True in calls and False in calls
