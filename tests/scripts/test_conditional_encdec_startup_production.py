"""Opt-in production-shaped startup gate for issue #1481.

This test is intentionally skipped unless an operator supplies a real model
configuration, extracted feature tree, and (when needed) text corpus. It never
silently substitutes a tiny fixture for production evidence.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.very_slow, pytest.mark.requires_feature_data]

_REQUIRED_PHASES = {
    "hydra_config",
    "startup_validation",
    "device_ddp_init",
    "attention_backend",
    "tokenizer",
    "feature_source_resolution",
    "statistics_checksums",
    "feature_index_overlays",
    "boundaries_and_feature_identity",
    "model_materialization",
    "packed_selection_artifacts",
    "compile_wrapper",
    "eval_contexts",
    "wandb_setup",
    "config_identity",
    "checkpoint_resume",
    "feature_pipeline_setup",
    "optimizer_setup",
    "effective_data_path",
    "dataloader_build",
    "first_batch_fetch",
    "pre_first_update_control",
}


def test_real_artifact_one_update_emits_complete_startup_trace() -> None:
    """Run one real update when an explicit production artifact gate is enabled."""
    if os.environ.get("ENCDEC_STARTUP_PRODUCTION") != "1":
        pytest.skip(
            "set ENCDEC_STARTUP_PRODUCTION=1 with an explicit real feature tree "
            "to run the production-shaped startup gate"
        )

    feature_dir_raw = os.environ.get("ENCDEC_STARTUP_PRODUCTION_FEATURE_DIR")
    if not feature_dir_raw:
        pytest.fail("ENCDEC_STARTUP_PRODUCTION_FEATURE_DIR is required")
    feature_dir = Path(feature_dir_raw).expanduser()
    if not feature_dir.is_dir():
        pytest.fail(f"production feature_dir does not exist: {feature_dir}")

    repo_root = Path(__file__).resolve().parents[2]
    config_name = os.environ.get(
        "ENCDEC_STARTUP_PRODUCTION_CONFIG_NAME",
        "conditional_encdec_4090",
    )
    num_iterations = os.environ.get("ENCDEC_STARTUP_PRODUCTION_NUM_ITERATIONS", "1")
    command = [
        sys.executable,
        "-m",
        "scripts.train.conditional_encdec_train",
        f"config_name={config_name}",
        f"feature_dir={feature_dir}",
        "run=dummy",
        f"num_iterations={num_iterations}",
        "eval_every=0",
        "core_metric_every=0",
        "sample_every=0",
        "steering_eval_every=0",
    ]
    data_dir = os.environ.get("ENCDEC_STARTUP_PRODUCTION_DATA_DIR")
    if data_dir:
        command.append(f"data_dir={Path(data_dir).expanduser()}")
    command.extend(
        shlex.split(os.environ.get("ENCDEC_STARTUP_PRODUCTION_OVERRIDES", ""))
    )

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    timeout_s = int(os.environ.get("ENCDEC_STARTUP_PRODUCTION_TIMEOUT_S", "3600"))
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        pytest.fail(
            f"production startup gate failed with exit={completed.returncode}\n{tail}"
        )

    lines = completed.stdout.splitlines()
    phase_lines = [
        line.split("encdec_startup_phase ", 1)[1]
        for line in lines
        if "encdec_startup_phase " in line
    ]
    phase_names = {
        field.split("=", 1)[1]
        for line in phase_lines
        if (field := next((part for part in line.split() if part.startswith("phase=")), None))
    }
    assert _REQUIRED_PHASES <= phase_names
    data_path_lines = [
        line.split("encdec_startup_data_path ", 1)[1]
        for line in lines
        if "encdec_startup_data_path " in line
    ]
    assert len(data_path_lines) == 1
    assert "packed_rows=" in data_path_lines[0]
    assert "selection=" in data_path_lines[0]
    assert "resolution_s=" in data_path_lines[0]
    assert "decision_s=" in data_path_lines[0]

    first_step_lines = [
        line.split("encdec_startup_first_step ", 1)[1]
        for line in lines
        if "encdec_startup_first_step " in line
    ]
    assert len(first_step_lines) == 1
    assert "first_step_s=" in first_step_lines[0]
    assert "time_to_first_step_s=" in first_step_lines[0]

    summary_lines = [
        line.split("encdec_startup_summary ", 1)[1]
        for line in lines
        if "encdec_startup_summary " in line
    ]
    assert len(summary_lines) == 1
    assert "status=complete" in summary_lines[0]
    assert "phase_seconds_json=" in summary_lines[0]
