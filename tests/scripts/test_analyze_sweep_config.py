"""Tests for scripts/analyze_sweep.py Hydra config migration."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import MissingMandatoryValue, OmegaConf

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")


class TestAnalyzeSweepConfigLoading:

    def test_requires_results(self, hydra_compose):
        with pytest.raises(MissingMandatoryValue):
            cfg = hydra_compose(_CONFIG_DIR, "analyze_sweep")
            _ = cfg.results

    def test_loads_with_results(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "analyze_sweep", ["results=/tmp/r.jsonl"])
        assert cfg.results == "/tmp/r.jsonl"
        assert cfg.output_dir is None
        assert cfg.expected_total is None

    def test_override_output_dir(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR,
            "analyze_sweep",
            ["results=/tmp/r.jsonl", "output_dir=/tmp/plots"],
        )
        assert cfg.output_dir == "/tmp/plots"

    def test_override_expected_total(self, hydra_compose):
        cfg = hydra_compose(
            _CONFIG_DIR,
            "analyze_sweep",
            ["results=/tmp/r.jsonl", "expected_total=315"],
        )
        assert cfg.expected_total == 315

    def test_config_keys_complete(self, hydra_compose):
        cfg = hydra_compose(_CONFIG_DIR, "analyze_sweep", ["results=/tmp/r.jsonl"])
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        keys = set(container)
        # ``kind`` was added in PR-D for sweep-kind filtering; keep this
        # in lockstep with ``configs/eval/analyze_sweep.yaml``.
        assert keys == {"results", "output_dir", "expected_total", "kind"}


class TestAnalyzeSweepArgparseRemoved:
    def test_no_argparse_import(self):
        source = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "eval"
            / "analyze_sweep.py"
        )
        text = source.read_text()
        assert "import argparse" not in text
        assert "ArgumentParser" not in text
        assert "parse_args" not in text
