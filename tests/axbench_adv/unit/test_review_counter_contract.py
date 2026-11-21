"""Independent regressions for the project-counter publication contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from nanochat.common import COMPUTE_DTYPE
from steering.conditional.encoder_decoder.eval_residual import (
    post_cross_attention_residual_counter,
    scoped_post_cross_attention_callback,
)


@pytest.mark.parametrize(
    "direction",
    [t.ones(2, 2), t.ones(3, dtype=t.int64), t.empty(0)],
)
def test_counter_scope_rejects_malformed_direction_shapes(direction: t.Tensor) -> None:
    model = SimpleNamespace(
        dec_config=SimpleNamespace(n_layer=2, n_embd=3),
        training=False,
        _post_cross_attention_eval_callback=None,
        get_device=lambda: t.device("cpu"),
    )
    with pytest.raises(ValueError, match="floating vector"):
        with post_cross_attention_residual_counter(
            model,
            layer=1,
            direction=direction,
            magnitude=1.0,
        ):
            pass


def test_counter_callback_rejects_residual_drift_and_still_cleans_up() -> None:
    model = SimpleNamespace(
        dec_config=SimpleNamespace(n_layer=2, n_embd=3),
        training=False,
        _post_cross_attention_eval_callback=None,
        get_device=lambda: t.device("cpu"),
    )
    with post_cross_attention_residual_counter(
        model,
        layer=1,
        direction=t.ones(3),
        magnitude=1.0,
    ):
        callback = model._post_cross_attention_eval_callback
        assert callback is not None
        with pytest.raises(ValueError, match="residual width"):
            callback(1, t.ones(1, 2, dtype=COMPUTE_DTYPE))
        with pytest.raises(RuntimeError, match="device/dtype changed"):
            callback(1, t.ones(1, 3, dtype=t.float64))
    assert model._post_cross_attention_eval_callback is None


def test_scoped_callback_rejects_missing_or_corrupt_slot() -> None:
    def callback(_layer: int, residual: t.Tensor) -> t.Tensor:
        return residual

    missing = SimpleNamespace(training=False)
    with pytest.raises(TypeError, match="EncoderDecoderGPT"):
        with scoped_post_cross_attention_callback(missing, callback):
            pass

    corrupt = SimpleNamespace(
        training=False,
        _post_cross_attention_eval_callback=object(),
    )
    with pytest.raises(RuntimeError, match="corrupt"):
        with scoped_post_cross_attention_callback(corrupt, callback):
            pass
