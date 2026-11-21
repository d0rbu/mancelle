"""Tests for viz/batch_plot.py Hydra config migration."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "viz")


class TestBatchPlotConfigLoading:
    """Layer 1: Config loads correctly with expected defaults and overrides."""

    def test_loads_with_defaults(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "batch_plot")
        assert cfg.grid_dir == "out/data/grid"
        assert cfg.output_dir == "fig/per_combo"
        assert cfg.dpi == 150

    def test_override_grid_dir(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "batch_plot", ["grid_dir=/tmp/grid"])
        assert cfg.grid_dir == "/tmp/grid"

    def test_override_output_dir(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "batch_plot", ["output_dir=/tmp/out"])
        assert cfg.output_dir == "/tmp/out"

    def test_override_dpi(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "batch_plot", ["dpi=300"])
        assert cfg.dpi == 300

    def test_override_all_fields(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR, "batch_plot", ["grid_dir=/a", "output_dir=/b", "dpi=72"]
        )
        assert cfg.grid_dir == "/a"
        assert cfg.output_dir == "/b"
        assert cfg.dpi == 72

    def test_config_keys_complete(self, hydra_compose):
        """All expected keys present, no extras."""
        cfg = hydra_compose(_CONFIG_DIR, "batch_plot")
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        keys = set(container)
        assert keys == {"grid_dir", "output_dir", "dpi"}


class TestBatchPlotArgparseRemoved:
    """Verify argparse is no longer used in the module."""

    def test_no_argparse_import(self):
        source = Path(__file__).resolve().parent.parent.parent / "viz" / "batch_plot.py"
        text = source.read_text()
        assert "import argparse" not in text
        assert "ArgumentParser" not in text
        assert "parse_args" not in text
