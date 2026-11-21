"""Tests for nanochat.config_validation — shared Hydra config validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanochat.config_validation import validate_sweep_mode

_TRAINING_CFG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")


class TestValidateSweepMode:
    """Tests for the polymorphic sweep mode validator."""

    def test_null_valid(self, hydra_compose):
        """null = run all configs."""
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast")
        validate_sweep_mode(cfg)

    def test_dry_run_valid(self, hydra_compose):
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=dry_run"])
        validate_sweep_mode(cfg)

    def test_index_valid(self, hydra_compose):
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=5"])
        validate_sweep_mode(cfg)

    def test_range_valid(self, hydra_compose):
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=[0,10]"])
        validate_sweep_mode(cfg)

    def test_invalid_string_raises(self, hydra_compose):
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=bogus"])
        with pytest.raises(ValueError, match="mode string must be 'dry_run'"):
            validate_sweep_mode(cfg)

    def test_range_single_element_raises(self, hydra_compose):
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=[5]"])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_sweep_mode(cfg)

    def test_range_three_elements_raises(self, hydra_compose):
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=[1,2,3]"])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_sweep_mode(cfg)

    def test_works_with_both_sweep_configs(self, hydra_compose):
        """Validator works with both sweep YAML files."""
        for name in ("run_encdec_sweep", "run_encdec_sweep_fast"):
            cfg = hydra_compose(_TRAINING_CFG_DIR, name, ["mode=dry_run"])
            validate_sweep_mode(cfg)

    def test_zero_index_valid(self, hydra_compose):
        """Edge case: index 0 is valid (not falsy)."""
        cfg = hydra_compose(_TRAINING_CFG_DIR, "run_encdec_sweep_fast", ["mode=0"])
        validate_sweep_mode(cfg)

    def test_unsupported_type_raises(self):
        """Unsupported mode type (e.g. float) raises ValueError."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"mode": 3.14})
        with pytest.raises(ValueError, match="must be null, 'dry_run'"):
            validate_sweep_mode(cfg)
