"""Tests for viz/cross_axis_plots.py Hydra config migration."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "viz")


class TestCrossAxisPlotsConfigLoading:
    """Layer 1: Config loads correctly with expected defaults and overrides."""

    def test_loads_with_defaults(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "cross_axis_plots")
        assert cfg.grid_dir == "out/data/grid"
        assert cfg.output_dir == "fig/cross_axis"

    def test_override_grid_dir(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "cross_axis_plots", ["grid_dir=/tmp/g"])
        assert cfg.grid_dir == "/tmp/g"

    def test_override_output_dir(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "cross_axis_plots", ["output_dir=/tmp/o"])
        assert cfg.output_dir == "/tmp/o"

    def test_config_keys_complete(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "cross_axis_plots")
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        keys = set(container)
        assert keys == {"grid_dir", "output_dir"}


class TestCrossAxisPlotsArgparseRemoved:
    def test_no_argparse_import(self):
        source = Path(__file__).resolve().parent.parent.parent / "viz" / "cross_axis_plots.py"
        text = source.read_text()
        assert "import argparse" not in text
        assert "ArgumentParser" not in text
        assert "parse_args" not in text
