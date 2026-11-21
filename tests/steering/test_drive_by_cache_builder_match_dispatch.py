"""PR-C drive-by: build_eval_context_cache.py stage dispatch uses match + StrEnum."""

from __future__ import annotations

import inspect

import pytest
from omegaconf import OmegaConf

from steering.eval.cache_builder_stage import CacheBuilderStage


def test_cache_builder_stage_enum_values():
    assert CacheBuilderStage.ENUMERATE.value == "enumerate"
    assert CacheBuilderStage.CONVERT.value == "convert"
    assert CacheBuilderStage.VALIDATE.value == "validate"


def test_cache_builder_stage_invalid_value_raises():
    with pytest.raises(ValueError):
        CacheBuilderStage("not_a_real_stage")


def test_build_eval_context_cache_main_uses_match_dispatch():
    import scripts.eval.build_eval_context_cache as mod
    src = inspect.getsource(mod.main)
    assert "match" in src
    assert "CacheBuilderStage" in src


def test_build_eval_context_cache_main_no_string_eq_dispatch():
    """Regression: the if/elif chain must be gone."""
    import scripts.eval.build_eval_context_cache as mod
    src = inspect.getsource(mod.main)
    # The old form had cfg.stage == "enumerate"; the new match form should not.
    assert 'cfg.stage == "enumerate"' not in src
    assert 'cfg.stage == "convert"' not in src
    assert 'cfg.stage == "validate"' not in src


def test_unknown_stage_raises_value_error(monkeypatch):
    """Regression: invalid stage values continue to raise ValueError (from CacheBuilderStage(...))."""
    import scripts.eval.build_eval_context_cache as mod

    cfg = OmegaConf.create({"stage": "delete_everything"})
    with pytest.raises(ValueError):
        mod.main.__wrapped__(cfg)


def test_each_stage_dispatches_to_correct_function(monkeypatch):
    """Verify each StrEnum value routes to the corresponding function."""
    import scripts.eval.build_eval_context_cache as mod

    calls = []
    monkeypatch.setattr(mod, "enumerate_contexts_from_config", lambda cfg: calls.append("enumerate"))
    monkeypatch.setattr(mod, "build_cache_from_config", lambda cfg: calls.append("convert"))
    monkeypatch.setattr(mod, "validate_cache_from_config", lambda cfg: calls.append("validate"))

    for stage in ("enumerate", "convert", "validate"):
        cfg = OmegaConf.create({"stage": stage})
        mod.main.__wrapped__(cfg)

    assert calls == ["enumerate", "convert", "validate"]
