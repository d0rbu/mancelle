"""Tests for harmfulness-eval startup and config contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.eval import harmfulness_eval as he


class _NullCM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_decoder_only_requires_null_feature_configs_and_cadence() -> None:
    he._validate_harmfulness_generation_config(
        feature_configs_configured=False,
        self_cond_segment_size=None,
        is_encdec=False,
        batch_size=8,
    )
    with pytest.raises(ValueError, match="feature_configs=null"):
        he._validate_harmfulness_generation_config(
            feature_configs_configured=True,
            self_cond_segment_size=256,
            is_encdec=False,
            batch_size=1,
        )
    with pytest.raises(ValueError, match="self_cond_segment_size=null"):
        he._validate_harmfulness_generation_config(
            feature_configs_configured=False,
                self_cond_segment_size=256,
            is_encdec=False,
            batch_size=8,
        )


def test_harmfulness_config_omits_removed_threshold_alias() -> None:
    with initialize_config_dir(config_dir=he._CONFIG_DIR, version_base=None):
        cfg = compose(config_name=he._CONFIG_NAME)

    assert "threshold" not in cfg
    assert cfg.batch_size == 8
    assert cfg.repository_hf_batch_size == 64


@pytest.mark.parametrize(
    ("is_hf", "expected"),
    [(False, 8), (True, 64)],
)
def test_generation_batch_size_has_backend_specific_default(
    is_hf: bool,
    expected: int,
) -> None:
    assert (
        he._resolve_generation_batch_size(
            {"batch_size": 8, "repository_hf_batch_size": 64},
            is_hf=is_hf,
        )
        == expected
    )


def test_repository_hf_batch_size_fallback_matches_config_default() -> None:
    assert he._resolve_generation_batch_size({"batch_size": 8}, is_hf=True) == 64


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("batch_size", True),
        ("batch_size", 0),
        ("repository_hf_batch_size", 1.0),
        ("repository_hf_batch_size", -1),
    ],
)
def test_generation_batch_sizes_are_strict_positive_integers(
    key: str,
    value: object,
) -> None:
    cfg: dict[str, object] = {"batch_size": 8, "repository_hf_batch_size": 64}
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        he._resolve_generation_batch_size(cfg, is_hf=True)


def test_force_added_threshold_alias_fails_loudly(monkeypatch) -> None:
    monkeypatch.setattr(he, "initialize_config_dir", lambda **kw: _NullCM())
    monkeypatch.setattr(he, "compose", lambda **kw: {"threshold": 2048.9})
    monkeypatch.setattr(sys, "argv", ["harmfulness_eval.py"])

    with pytest.raises(ValueError, match="harmfulness_eval.threshold was removed"):
        he.main()


def test_encoder_decoder_rejects_sparse_maps_and_requires_batch_one() -> None:
    arguments: dict[str, Any] = {"feature_configs_configured": False,
        "self_cond_segment_size": None, "is_encdec": True, "batch_size": 1}
    he._validate_harmfulness_generation_config(**arguments)
    with pytest.raises(ValueError, match="hand-supplied feature_configs"):
        he._validate_harmfulness_generation_config(**{**arguments,
            "feature_configs_configured": True})
    invalid_batch: dict[str, Any] = {**arguments, "batch_size": 2}
    with pytest.raises(ValueError, match="batch_size=1"):
        he._validate_harmfulness_generation_config(**invalid_batch)
    with pytest.raises(ValueError, match="fixed-cadence assertion"):
        he._validate_harmfulness_generation_config(**{**arguments, "self_cond_segment_size": True})
    with pytest.raises(ValueError, match="encoder-decoder checkpoint"):
        he._validate_harmfulness_generation_config(**{**arguments,
            "is_encdec": False, "safety_steering_configured": True})


def _selector_cfg(**overrides: object):
    cfg = OmegaConf.load(Path(he._CONFIG_DIR) / f"{he._CONFIG_NAME}.yaml")
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    return cfg


def test_model_selector_contract_accepts_native_hf_and_response_only(tmp_path: Path) -> None:
    hf_dir = tmp_path / "step_000123"
    hf_dir.mkdir()
    assert he._resolve_hf_checkpoint_dir(_selector_cfg(source="safety")) is None
    assert (
        he._resolve_hf_checkpoint_dir(_selector_cfg(hf_checkpoint_dir=str(hf_dir)))
        == hf_dir.resolve()
    )
    assert he._resolve_hf_checkpoint_dir(_selector_cfg(responses_file="responses.yaml")) is None


@pytest.mark.parametrize("feature_configs", [None, []])
def test_hf_accepts_only_empty_feature_config_lists(tmp_path, feature_configs):
    checkpoint = tmp_path / "step_000123"
    checkpoint.mkdir()
    cfg = _selector_cfg(hf_checkpoint_dir=str(checkpoint), feature_configs=feature_configs)
    assert he._resolve_hf_checkpoint_dir(cfg) == checkpoint.resolve()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({}, "model selector is required"),
        (
            {"source": "safety", "hf_checkpoint_dir": "/tmp/export/step_000123"},
            "Exactly one",
        ),
        ({"hf_checkpoint_dir": "/tmp/export/step_000123", "model_tag": "d26"}, "model_tag"),
        ({"hf_checkpoint_dir": "/tmp/export/step_000123", "step": 3}, "step"),
        *[
            ({"hf_checkpoint_dir": "/tmp/export/step_000123", "feature_configs": value}, "feature_configs")
            for value in (False, 0, "", {})
        ],
        ({"hf_checkpoint_dir": "/tmp/export/step_000123", "stats_path": "/tmp/statistics.safetensors"}, "stats_path"),
        ({"hf_checkpoint_dir": "/tmp/export/step_000123", "extraction_tokenizer_id": "test-tokenizer"}, "extraction_tokenizer_id"),
        ({"hf_checkpoint_dir": "/tmp/export/step_000123",
            "self_conditioned.safety_steering": {"targets": "/tmp/targets.safetensors", "strength": 0}},
            "self_conditioned.safety_steering"),
        (
            {
                "hf_checkpoint_dir": "/tmp/export/step_000123",
                "feature_configs": [{"name": "f", "features": {1: 2}}],
            },
            "feature_configs",
        ),
        (
            {"hf_checkpoint_dir": "/tmp/export/step_000123", "feature_mask": {"ids": [1]}},
            "feature_mask",
        ),
        (
            {"hf_checkpoint_dir": "/tmp/export/step_000123", "segment_size": 512},
            "segment_size",
        ),
        (
            {
                "hf_checkpoint_dir": "/tmp/export/step_000123",
                "conditioning.context_cache_dir": "/tmp/cache",
            },
            "context_cache_dir",
        ),
        (
            {
                "hf_checkpoint_dir": "/tmp/export/step_000123",
                "next_segment_conditioning_offset": 1,
            },
            "next_segment_conditioning_offset",
        ),
    ],
)
def test_invalid_hf_selector_configs_fail_before_model_or_judge_construction(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    match: str,
) -> None:
    cfg = _selector_cfg(**overrides)
    judge = MagicMock(side_effect=AssertionError("judge must not be constructed"))
    loader = MagicMock(side_effect=AssertionError("weights must not be loaded"))
    monkeypatch.setattr(he, "initialize_config_dir", lambda **kw: _NullCM())
    monkeypatch.setattr(he, "compose", lambda **kw: cfg)
    monkeypatch.setattr(he, "JudgeClient", judge)
    monkeypatch.setattr(he, "load_repository_hf_checkpoint", loader)
    monkeypatch.setattr(sys, "argv", ["harmfulness_eval.py"])

    with pytest.raises(ValueError, match=match):
        he.main()
    judge.assert_not_called()
    loader.assert_not_called()


def test_hf_output_paths_are_checkpoint_local() -> None:
    responses, results = he.resolve_harmfulness_paths(
        source=None,
        model_tag=None,
        output_file=None,
        responses_file=None,
        hf_checkpoint_dir="/tmp/export/step_000123",
        config_name="unconditional",
    )
    assert responses == (
        "/tmp/export/step_000123/report/harmfulness_results_responses/unconditional.yaml"
    )
    assert results == ("/tmp/export/step_000123/report/harmfulness_results/unconditional.yaml")


def test_explicit_responses_file_derives_result_path_without_model_selector() -> None:
    responses, results = he.resolve_harmfulness_paths(
        source=None,
        model_tag=None,
        output_file=None,
        responses_file="/tmp/responses.yaml",
        config_name="unconditional",
    )
    assert responses == "/tmp/responses.yaml"
    assert results == "/tmp/responses_results.yaml"


def test_invalid_hf_path_fails_before_model_or_judge_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _selector_cfg(hf_checkpoint_dir=str(tmp_path / "mutable-parent"))
    judge = MagicMock(side_effect=AssertionError("judge must not be constructed"))
    loader = MagicMock(side_effect=AssertionError("weights must not be loaded"))
    monkeypatch.setattr(he, "initialize_config_dir", lambda **kw: _NullCM())
    monkeypatch.setattr(he, "compose", lambda **kw: cfg)
    monkeypatch.setattr(he, "JudgeClient", judge)
    monkeypatch.setattr(he, "load_repository_hf_checkpoint", loader)
    monkeypatch.setattr(sys, "argv", ["harmfulness_eval.py"])

    with pytest.raises(ValueError, match="missing or unreadable"):
        he.main()
    judge.assert_not_called()
    loader.assert_not_called()
