"""Checkpoint-selection and report-path contract for the steering-eval driver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
from omegaconf import OmegaConf

from nanochat.checkpoint_manager import ModelPhase
from scripts.eval.steering_eval import (
    _load_model_for_steering_eval,
    _resolve_output_dir,
)
from steering.conditional.eval_conditioning import EvalConditioning


def _config(**overrides: object):
    values = {
        "source": "encdec",
        "model_tag": "d26",
        "step": 17,
        "segment_size": None,
        "feature_selection": None,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_threshold_selection_pins_the_loaded_checkpoint_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _config(
        segment_size=128,
        feature_selection={
            "threshold_type": "segment_topk",
            "threshold": 2048.9,
        },
    )
    model = object()
    tokenizer = object()
    meta: dict[str, object] = {}

    with patch(
        "scripts.eval.steering_eval.build_encdec_model",
        return_value=(model, tokenizer, meta),
    ) as build:
        loaded_model, loaded_tokenizer, loaded_meta = _load_model_for_steering_eval(
            cfg, t.device("cpu")
        )

    expected_checkpoint = (
        tmp_path / "conditional_encdec_checkpoints" / "d26" / "s128" / "t2048.9"
    )
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert loaded_meta["checkpoint_dir"] == str(expected_checkpoint)
    build.assert_called_once_with(
        str(expected_checkpoint),
        17,
        t.device("cpu"),
        ModelPhase.EVAL,
        eval_conditioning=EvalConditioning.deferred("steering_eval"),
    )
    assert _resolve_output_dir(None, loaded_meta["checkpoint_dir"]) == (
        expected_checkpoint / "report" / "steering_eval"
    )


def test_null_checkpoint_pin_retains_load_model_auto_discovery() -> None:
    cfg = _config(step=None)
    sentinel = (object(), object(), {"checkpoint_dir": "/selected/leaf"})

    with patch("scripts.eval.steering_eval.load_model", return_value=sentinel) as load:
        assert _load_model_for_steering_eval(cfg, t.device("cpu")) is sentinel

    load.assert_called_once_with(
        source="encdec",
        device=t.device("cpu"),
        phase=ModelPhase.EVAL,
        model_tag="d26",
        step=None,
        eval_conditioning=EvalConditioning.deferred("steering_eval"),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"feature_selection": {"threshold_type": "segment_topk", "threshold": 8.5}},
            "requires segment_size",
        ),
        ({"segment_size": 128}, "without feature_selection is ambiguous"),
    ],
)
def test_partial_encdec_checkpoint_pin_is_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _load_model_for_steering_eval(_config(**overrides), t.device("cpu"))


def test_non_encdec_source_rejects_encdec_checkpoint_pin() -> None:
    with pytest.raises(ValueError, match="require an encdec-rooted source"):
        _load_model_for_steering_eval(
            _config(source="safety_cond", segment_size=128),
            t.device("cpu"),
        )


def test_explicit_output_dir_does_not_require_checkpoint_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom"
    assert _resolve_output_dir(str(output_dir), None) == output_dir


def test_default_output_dir_requires_loaded_checkpoint_metadata() -> None:
    with pytest.raises(ValueError, match="concrete checkpoint_dir"):
        _resolve_output_dir(None, None)
