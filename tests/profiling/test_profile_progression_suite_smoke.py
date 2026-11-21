"""Smoke test for scripts.profile_progression run_suite (CPU-only, hermetic)."""

from __future__ import annotations

import csv
import json
import time
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from scripts.profile_progression import run_suite
from steering.profiling.markers import timed_region


def _stub_evaluate_single_checkpoint(**_kwargs):
    with timed_region("evaluate_single_checkpoint"):
        with timed_region("load_progression_model"):
            time.sleep(0.002)
        with timed_region("build_eval_wrapper"):
            time.sleep(0.001)
        for marker in ("run_core_eval", "run_chat_eval_suite", "run_harmfulness_eval"):
            with timed_region(marker):
                time.sleep(0.002)


@pytest.fixture
def suite_cfg(tmp_path):
    return OmegaConf.create(
        {
            "level": "suite",
            "num_iterations": 1,
            "warmup_iterations": 0,
            "out_root": str(tmp_path / "profiling"),
            "preflight_skip": True,
            "progression": {
                "model_family": "decoder",
                "evals": "core,chat,harmfulness",
                "num_seeds": 1,
                "base_seed": 1234,
                "dtype": "float32",
                "step": 0,
                "phase": "pretrain",
                "subdir": "",
                "max_per_task": 20,
                "chat_max_problems": 20,
                "harm_num_harmful": 10,
                "harm_num_benign": 10,
            },
        }
    )


def test_suite_writes_trace_and_summary(suite_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch(
        "scripts.profile_progression.runner.evaluate_single_checkpoint",
        _stub_evaluate_single_checkpoint,
    ):
        run_suite(suite_cfg, run_dir)

    # trace.json exists and is valid JSON
    trace = run_dir / "trace.json"
    assert trace.exists()
    with open(trace) as f:
        trace_data = json.load(f)
    assert isinstance(trace_data, (list, dict))

    # key_averages.txt has content
    ka = run_dir / "key_averages.txt"
    assert ka.exists()
    assert len(ka.read_text()) > 0

    # record_function_summary.csv has rows for the coarse markers
    summary = run_dir / "record_function_summary.csv"
    assert summary.exists()
    with open(summary, newline="") as f:
        rows = list(csv.DictReader(f))
    names = {row["name"] for row in rows}
    expected = {
        "evaluate_single_checkpoint",
        "load_progression_model",
        "build_eval_wrapper",
        "run_core_eval",
        "run_chat_eval_suite",
        "run_harmfulness_eval",
    }
    assert expected.issubset(names), f"missing markers: {expected - names}"

    # Schema: 4 columns including the new device_time_us name
    assert set(rows[0].keys()) == {"name", "self_cpu_time_us", "device_time_us", "count"}


def test_suite_scale_reduction_applied(suite_cfg, tmp_path):
    """Assert that run_suite overrides max_per_task / harm_num_* to Level-2 values."""
    captured: dict = {}

    def capture_cfg(**kwargs):
        # Only capture cfg on the first (warmup) call — both calls pass the same cfg
        captured.setdefault("cfg", kwargs["cfg"])

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch(
        "scripts.profile_progression.runner.evaluate_single_checkpoint",
        side_effect=capture_cfg,
    ):
        run_suite(suite_cfg, run_dir)

    cfg = captured["cfg"]
    assert int(cfg.max_per_task) == 5
    assert int(cfg.chat_max_problems) == 5
    assert int(cfg.harm_num_harmful) == 3
    assert int(cfg.harm_num_benign) == 3
