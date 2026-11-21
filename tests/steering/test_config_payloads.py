"""Tests for validated Hydra config payload conversion."""

from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from steering.config_payloads import resolved_config_dict


def test_resolved_config_dict_resolves_interpolations() -> None:
    cfg = OmegaConf.create({"run": "sample", "nested": {"name": "${run}"}})
    assert isinstance(cfg, DictConfig)

    assert resolved_config_dict(cfg) == {
        "run": "sample",
        "nested": {"name": "sample"},
    }


def test_resolved_config_dict_rejects_non_string_keys() -> None:
    cfg = OmegaConf.create({1: "value"})
    assert isinstance(cfg, DictConfig)

    with pytest.raises(TypeError, match="keys must be strings"):
        resolved_config_dict(cfg)
