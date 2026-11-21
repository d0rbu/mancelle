"""PR-D: progression target_real_configs schema and validation.

Pinned invariants:
  - default ships pooled + lockstep entries.
  - legacy no-context conditioning switches are rejected.
  - name collisions across config blocks raise.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import scripts.train.training_progression as tp_mod
from scripts.train.training_progression import (
    _validate_pr_d_target_real_configs,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval"


# ---------------------------------------------------------------------------
# Fix #1: 3-mode default tests (written first, run to fail before implementation)
# ---------------------------------------------------------------------------


def test_default_target_real_configs_is_3_mode_default():
    """Default training_progression.yaml ships pooled + lockstep configs."""
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="training_progression")
    assert cfg.conditioning.target_real_configs is not None
    names = {trc["name"] for trc in cfg.conditioning.target_real_configs}
    modes = {trc["mode"] for trc in cfg.conditioning.target_real_configs}
    assert names == {"pooled", "lockstep"}
    assert modes == {"pooled", "lockstep"}


def test_default_target_real_configs_have_null_cache_dir_for_auto_resolve():
    """cache_dir is null/missing in the default so the parser auto-resolves
    via default_target_cache_dir per suite at iteration time."""
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name="training_progression")
    for trc in cfg.conditioning.target_real_configs:
        assert "cache_dir" not in trc or trc.get("cache_dir") is None


@pytest.mark.parametrize(
    "config_name",
    [
        "training_progression_d12_baseline",
        "training_progression_d12_baseline_reversed",
        "training_progression_d12_baseline_smoke",
        "training_progression_d12_fp",
        "training_progression_d12_fp10",
    ],
)
def test_static_delta_progression_configs_have_full_context_cache_identity(config_name):
    """Concrete static-delta configs must be runnable full-context configs."""
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name)

    conditioned = list(cfg.conditioning.get("conditioned_configs") or [])
    assert conditioned, f"{config_name} should define static target-delta configs"
    assert cfg.conditioning.context_cache_dir, (
        f"{config_name} must provide conditioning.context_cache_dir"
    )
    assert cfg.conditioning.target_real_configs is None
    assert cfg.conditioning.identity.feature_selection_hash is not None
    assert len(str(cfg.conditioning.identity.feature_selection_hash)) == 64
    assert int(cfg.conditioning.n_sae_features) == 16384
    assert int(cfg.conditioning.num_buckets) == 2


def test_progression_collect_loop_uses_per_suite_cache_dir_resolution():
    """parse_target_real_configs_from_cfg(cfg.conditioning, suite='..._target')
    is what the collect-loop calls — verify the call site uses suite kwarg."""
    src = inspect.getsource(tp_mod)
    assert "parse_target_real_configs_from_cfg" in src
    # Per-suite kwarg pattern: f"{eval_name}_target" or equivalent.
    assert "suite=" in src


def test_validation_passes_when_target_real_empty():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": {},
            "target_real_configs": [],
        }
    })
    _validate_pr_d_target_real_configs(cfg)


def test_validation_rejects_legacy_context_mode():
    cfg = OmegaConf.create({
        "conditioning": {
            "context_mode": "cached",
            "conditioned_configs": {},
            "target_real_configs": [
                {"name": "real_safety", "mode": "lockstep", "cache_dir": "/tmp/x"},
            ],
        }
    })
    with pytest.raises(ValueError, match="context_mode.*no longer supported"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_fails_on_name_collision_with_conditioned_configs():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": {"safety": "out/safety_features.yaml"},
            "target_real_configs": [
                {"name": "safety", "mode": "pooled", "cache_dir": "/tmp/x"},
            ],
        }
    })
    with pytest.raises(ValueError, match="name collision"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_fails_on_name_collision_with_conditioned_config_list():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": [
                {"name": "safety", "features": {1: 1}},
                {"name": "capabilities", "features_file": "out/cap.yaml"},
            ],
            "target_real_configs": [
                {"name": "capabilities", "mode": "pooled", "cache_dir": "/tmp/x"},
            ],
        }
    })
    with pytest.raises(ValueError, match="name collision.*capabilities"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_fails_on_name_collision_with_phase_mapped_conditioned_configs():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": {
                "pretrain": [{"name": "safety", "features": {1: 1}}],
                "capabilities_sft": [{"name": "lockstep", "features": {2: 1}}],
            },
            "target_real_configs": [
                {"name": "lockstep", "mode": "lockstep", "cache_dir": "/tmp/x"},
            ],
        }
    })
    with pytest.raises(ValueError, match="name collision.*lockstep"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_rejects_malformed_conditioned_config_entries():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": [123],
            "target_real_configs": [
                {"name": "real_safety", "mode": "pooled", "cache_dir": "/tmp/x"},
            ],
        }
    })
    with pytest.raises(ValueError, match=r"conditioned_configs\[0\].*mapping"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_fails_on_name_collision_with_unconditioned():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": {},
            "target_real_configs": [
                {"name": "unconditioned", "mode": "pooled", "cache_dir": "/tmp/x"},
            ],
        }
    })
    with pytest.raises(ValueError, match="reserved.*unconditioned"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_fails_on_intra_target_real_name_duplicates():
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": {},
            "target_real_configs": [
                {"name": "real_x", "mode": "lockstep", "cache_dir": "/tmp/a"},
                {"name": "real_x", "mode": "pooled", "cache_dir": "/tmp/b"},
            ],
        }
    })
    with pytest.raises(ValueError, match="duplicate"):
        _validate_pr_d_target_real_configs(cfg)


def test_validation_passes_when_target_real_lockstep_and_pooled_share_base_name_via_suffix():
    """Distinct names like real_safety_lockstep / real_safety_pooled are fine."""
    cfg = OmegaConf.create({
        "conditioning": {
            "conditioned_configs": {},
            "target_real_configs": [
                {"name": "real_safety_lockstep", "mode": "lockstep", "cache_dir": "/tmp/a"},
                {"name": "real_safety_pooled", "mode": "pooled", "cache_dir": "/tmp/b"},
            ],
        }
    })
    _validate_pr_d_target_real_configs(cfg)


def test_validator_rejects_string_entries_in_target_real_configs():
    """Pre-validation also gates malformed types."""
    cfg = OmegaConf.create({
        "conditioning": {
            "target_real_configs": ["foo", "bar"],  # malformed
        }
    })
    with pytest.raises(ValueError, match=r"YAML mapping"):
        _validate_pr_d_target_real_configs(cfg)
