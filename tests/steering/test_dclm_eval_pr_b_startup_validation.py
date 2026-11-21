"""Full-context startup validation for dclm_eval."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from steering.eval.target_real_runner import require_shared_context_cache_dir

_DCLM_EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval"


def test_dclm_eval_default_feature_configs_are_empty() -> None:
    with initialize_config_dir(config_dir=str(_DCLM_EVAL_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="dclm_eval", overrides=["source=base"])

    fcs = OmegaConf.to_container(cfg.feature_configs, resolve=True)
    assert fcs == []


def test_full_context_validation_requires_cache_dir() -> None:
    cfg = OmegaConf.create({"conditioning": {"context_cache_dir": None}})
    with pytest.raises(ValueError, match="context_cache_dir"):
        require_shared_context_cache_dir(cfg, eval_name="dclm_eval")


def test_full_context_validation_passes_with_cache_dir() -> None:
    cfg = OmegaConf.create({"conditioning": {"context_cache_dir": "/cache"}})
    require_shared_context_cache_dir(cfg, eval_name="dclm_eval")


def test_dclm_identity_defaults_match_base_eval() -> None:
    with initialize_config_dir(config_dir=str(_DCLM_EVAL_CONFIG_DIR), version_base=None):
        dclm_cfg = compose(config_name="dclm_eval", overrides=["source=base"])
        base_cfg = compose(config_name="base_eval", overrides=["source=base"])

    assert dclm_cfg.conditioning.identity.model_id == base_cfg.conditioning.identity.model_id
    assert dclm_cfg.conditioning.identity.sae_id == base_cfg.conditioning.identity.sae_id
    assert dclm_cfg.conditioning.identity.layer == base_cfg.conditioning.identity.layer
    assert dclm_cfg.conditioning.identity.pooling == base_cfg.conditioning.identity.pooling
    assert dclm_cfg.conditioning.n_sae_features == base_cfg.conditioning.n_sae_features
    assert dclm_cfg.conditioning.num_buckets == base_cfg.conditioning.num_buckets
