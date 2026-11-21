"""Opt-in PromptSteering prompt override in the vendored AxBench (no LM call when set)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[3] / "vendor/axbench/axbench/utils/steering_prompt_override.py"
_spec = importlib.util.spec_from_file_location("steering_prompt_override", _PATH)
assert _spec is not None and _spec.loader is not None
override = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(override)

TEMPLATE = "Talk about %s."


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(override.ENV_FILE, raising=False)
    monkeypatch.delenv(override.ENV_TEMPLATE, raising=False)


def test_default_is_no_override():
    assert override.override_steering_prompts(["cats"], TEMPLATE) is None


def test_template_mode(monkeypatch):
    monkeypatch.setenv(override.ENV_TEMPLATE, "1")
    assert override.override_steering_prompts(["cats", "rain"], TEMPLATE) == ["Talk about cats.", "Talk about rain."]


def test_template_zero_means_off(monkeypatch):
    monkeypatch.setenv(override.ENV_TEMPLATE, "0")
    assert override.override_steering_prompts(["cats"], TEMPLATE) is None


def test_file_mode_keeps_concept_order(monkeypatch, tmp_path):
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps({"rain": "Mention rain.", "cats": "Mention cats.", "unused": "x"}))
    monkeypatch.setenv(override.ENV_FILE, str(f))
    assert override.override_steering_prompts(["cats", "rain"], TEMPLATE) == ["Mention cats.", "Mention rain."]


def test_file_missing_concept_fails(monkeypatch, tmp_path):
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps({"cats": "Mention cats."}))
    monkeypatch.setenv(override.ENV_FILE, str(f))
    with pytest.raises(KeyError, match="1 concept"):
        override.override_steering_prompts(["cats", "rain"], TEMPLATE)


@pytest.mark.parametrize("payload", [["a"], {"cats": ""}, {"cats": 3}])
def test_file_rejects_bad_payload(monkeypatch, tmp_path, payload):
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps(payload))
    monkeypatch.setenv(override.ENV_FILE, str(f))
    with pytest.raises(ValueError):
        override.override_steering_prompts(["cats"], TEMPLATE)


def test_both_switches_fail(monkeypatch, tmp_path):
    f = tmp_path / "prompts.json"
    f.write_text(json.dumps({"cats": "Mention cats."}))
    monkeypatch.setenv(override.ENV_FILE, str(f))
    monkeypatch.setenv(override.ENV_TEMPLATE, "1")
    with pytest.raises(ValueError, match="at most one"):
        override.override_steering_prompts(["cats"], TEMPLATE)


def test_bad_template_value_fails(monkeypatch):
    monkeypatch.setenv(override.ENV_TEMPLATE, "yes")
    with pytest.raises(ValueError):
        override.override_steering_prompts(["cats"], TEMPLATE)
