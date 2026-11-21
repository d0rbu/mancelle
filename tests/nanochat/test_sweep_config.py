"""Tests for sweep script Hydra config migrations.

Covers both run_encdec_sweep.py and run_encdec_sweep_fast.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nanochat.config_validation import validate_sweep_mode

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "training")


# ---------------------------------------------------------------------------
# run_encdec_sweep.py config
# ---------------------------------------------------------------------------


class TestSweepConfigLoading:
    """Config loading and defaults for run_encdec_sweep."""

    def test_defaults_run_all(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep")
        assert cfg.mode is None  # null = run all
        assert cfg.gpus is None  # null = auto-detect
        assert cfg.dump_json is None

    def test_mode_dry_run(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=dry_run"])
        assert cfg.mode == "dry_run"

    def test_mode_index(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=42"])
        assert cfg.mode == 42

    def test_mode_range(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=[0,50]"])
        assert list(cfg.mode) == [0, 50]

    def test_override_gpus(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["gpus=8"])
        assert cfg.gpus == 8

    def test_override_dump_json(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR, "run_encdec_sweep", ["dump_json=/tmp/configs.json"]
        )
        assert cfg.dump_json == "/tmp/configs.json"

    def test_config_keys_complete(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved, dict)
        keys = set(resolved)
        assert keys == {"mode", "gpus", "dump_json"}


class TestSweepConfigValidation:

    def test_null_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep")
        validate_sweep_mode(cfg)

    def test_dry_run_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=dry_run"])
        validate_sweep_mode(cfg)

    def test_index_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=5"])
        validate_sweep_mode(cfg)

    def test_range_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=[0,10]"])
        validate_sweep_mode(cfg)

    def test_invalid_string_raises(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=bogus"])
        with pytest.raises(ValueError, match="mode string must be 'dry_run'"):
            validate_sweep_mode(cfg)

    def test_range_wrong_length_raises(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep", ["mode=[5]"])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_sweep_mode(cfg)


class TestSweepArgparseRemoved:
    def test_no_argparse_import(self):
        source = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "run_encdec_sweep.py"
        )
        text = source.read_text()
        assert "import argparse" not in text
        assert "ArgumentParser" not in text
        assert "parse_args" not in text


# ---------------------------------------------------------------------------
# run_encdec_sweep_fast.py config
# ---------------------------------------------------------------------------


class TestSweepFastConfigLoading:
    """Config loading and defaults for run_encdec_sweep_fast."""

    def test_defaults_run_all(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast")
        assert cfg.mode is None
        assert cfg.min_idx is None
        assert cfg.max_idx is None
        assert cfg.resume is False
        assert cfg.shuffle is False
        assert cfg.shuffle_seed == 42
        assert cfg.multigpu_config is None
        assert cfg.output_dir is None
        assert cfg.config_name == "conditional_encdec_sweep"

    def test_mode_dry_run(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=dry_run"])
        assert cfg.mode == "dry_run"

    def test_mode_index(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=42"])
        assert cfg.mode == 42

    def test_mode_range(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=[10,20]"])
        assert list(cfg.mode) == [10, 20]

    def test_multigpu_config_object(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR,
            "run_encdec_sweep_fast",
            [
                "+multigpu_config.sweep_rank=2",
                "+multigpu_config.sweep_world_size=4",
                "+multigpu_config.gpu=3",
            ],
        )
        assert cfg.multigpu_config.sweep_rank == 2
        assert cfg.multigpu_config.sweep_world_size == 4
        assert cfg.multigpu_config.gpu == 3

    def test_override_resume_shuffle(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR,
            "run_encdec_sweep_fast",
            ["resume=true", "shuffle=true", "shuffle_seed=123"],
        )
        assert cfg.resume is True
        assert cfg.shuffle is True
        assert cfg.shuffle_seed == 123

    def test_override_config_name(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR,
            "run_encdec_sweep_fast",
            ["config_name=conditional_encdec_sweep_4090"],
        )
        assert cfg.config_name == "conditional_encdec_sweep_4090"

    def test_override_filtering(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR,
            "run_encdec_sweep_fast",
            ["min_idx=10", "max_idx=50"],
        )
        assert cfg.min_idx == 10
        assert cfg.max_idx == 50

    def test_config_keys_complete(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast")
        resolved = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(resolved, dict)
        keys = set(resolved)
        assert keys == {
            "mode",
            "min_idx",
            "max_idx",
            "resume",
            "shuffle",
            "shuffle_seed",
            "multigpu_config",
            "output_dir",
            "config_name",
            "training_overrides",
        }


class TestSweepFastConfigValidation:

    def test_null_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast")
        validate_sweep_mode(cfg)

    def test_dry_run_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=dry_run"])
        validate_sweep_mode(cfg)

    def test_index_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=0"])
        validate_sweep_mode(cfg)

    def test_range_valid(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=[0,10]"])
        validate_sweep_mode(cfg)

    def test_invalid_string_raises(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=bogus"])
        with pytest.raises(ValueError, match="mode string must be 'dry_run'"):
            validate_sweep_mode(cfg)

    def test_range_wrong_length_raises(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "run_encdec_sweep_fast", ["mode=[5]"])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            validate_sweep_mode(cfg)


class TestSweepFastArgparseRemoved:
    def test_no_argparse_import(self):
        source = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "run_encdec_sweep_fast.py"
        )
        text = source.read_text()
        assert "import argparse" not in text
        assert "ArgumentParser" not in text
        assert "parse_args" not in text
