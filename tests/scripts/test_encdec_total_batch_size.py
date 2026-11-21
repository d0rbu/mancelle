from __future__ import annotations

import math
from typing import Any

import pytest

from scripts.train.conditional_encdec_train import _resolve_total_batch_size


class _ScalingModel:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.calls = 0

    def num_scaling_params(self) -> dict[str, int]:
        self.calls += 1
        return self.counts


def _model() -> _ScalingModel:
    return _ScalingModel(
        {
            "transformer_matrices": 100,
            "lm_head": 200,
            "wte": 300,
            "value_embeds": 400,
            "scalars": 500,
        }
    )


def test_explicit_total_batch_size_does_not_inspect_model() -> None:
    model = _model()
    assert (
        _resolve_total_batch_size(
            524_288,
            model=model,
            target_param_data_ratio=20,
            d_ref=1_500,
            b_ref=32_768,
        )
        == 524_288
    )
    assert model.calls == 0


def test_auto_total_batch_size_matches_scaling_law() -> None:
    model = _model()
    target_tokens = 20 * 1_500
    expected = 2 ** round(math.log2(32_768 * (target_tokens / 1_500) ** 0.383))
    assert (
        _resolve_total_batch_size(
            -1,
            model=model,
            target_param_data_ratio=20,
            d_ref=1_500,
            b_ref=32_768,
        )
        == expected
    )
    assert model.calls == 1


@pytest.mark.parametrize("value", [True, 0, -2, 1.5, None, "524288"])
def test_invalid_total_batch_size_fails_loudly(value: Any) -> None:
    with pytest.raises(ValueError, match="positive strict int or -1"):
        _resolve_total_batch_size(
            value,
            model=_model(),
            target_param_data_ratio=20,
            d_ref=1_500,
            b_ref=32_768,
        )
