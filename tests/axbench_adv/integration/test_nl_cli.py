"""Exercise the actual Hydra CLI before any model or provider work can begin."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.axbench_adv.unit.test_run_config import _raw


@pytest.mark.parametrize("mode", ["resolved", "invalid_boolean"])
def test_nl_flag_crosses_the_real_hydra_command_boundary(tmp_path: Path, mode: str) -> None:
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.create(_raw())
    config.source.expectations.include_nl_steering = False
    config.source.expectations.model_revision = "it-soup-APO"
    config.output_dir = str(tmp_path / "output")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    OmegaConf.save(config, config_dir / "nl_probe.yaml")
    command = [
        sys.executable, "-m", "scripts.eval.axbench_adv.run",
        "--config-path", str(config_dir), "--config-name", "nl_probe",
        "source.expectations.include_nl_steering=" + ("true" if mode == "resolved" else "invalid"),
    ]
    if mode == "resolved":
        command.extend(("--cfg", "job", "--resolve"))
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=90, check=False)
    if mode == "resolved":
        assert result.returncode == 0, result.stdout + result.stderr
        resolved = OmegaConf.create(result.stdout)
        assert resolved.source.expectations.include_nl_steering is True
        assert resolved.source.expectations.model_revision == "it-soup-APO"
    else:
        assert result.returncode != 0
        assert "include_nl_steering" in result.stderr
        assert "violates type hint" in result.stderr
    assert not (tmp_path / "output").exists()
