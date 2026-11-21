"""Smoke test for scripts.profile_progression run_macro (CPU-only, hermetic)."""

from __future__ import annotations

import csv
import time
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from scripts.profile_progression import run_macro
from steering.profiling.markers import timed_region


def _stub_evaluate_single_checkpoint(**_kwargs):
    # Exercise the coarse markers in the same nesting the real code uses.
    with timed_region("evaluate_single_checkpoint"):
        with timed_region("load_progression_model"):
            time.sleep(0.005)
        with timed_region("build_eval_wrapper"):
            time.sleep(0.002)
        for marker in ("run_core_eval", "run_chat_eval_suite", "run_harmfulness_eval"):
            with timed_region(marker):
                time.sleep(0.003)


@pytest.fixture
def macro_cfg(tmp_path):
    cfg = OmegaConf.create(
        {
            "level": "macro",
            "num_iterations": 2,
            "warmup_iterations": 1,
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
            },
        }
    )
    return cfg


def test_macro_writes_csv_with_expected_schema(macro_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch(
        "scripts.profile_progression.runner.evaluate_single_checkpoint",
        _stub_evaluate_single_checkpoint,
    ):
        run_macro(macro_cfg, run_dir)

    csv_path = run_dir / "macro_wall_times.csv"
    assert csv_path.exists()

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows, "macro_wall_times.csv must not be empty"

    # Schema
    expected_columns = {
        "path",
        "suite",
        "stage",
        "iter",
        "warmup",
        "wall_seconds",
        "variant",
        "conditioned_config",
        "seed",
    }
    assert set(rows[0].keys()) == expected_columns

    # Content: 3 iters (1 warmup + 2 timed) × {core, chat, harmfulness} = 9 suite rows;
    # plus 1 load + 1 build per iter × 3 = 6 startup rows; plus 3 outer rows.
    suites = [r for r in rows if r["stage"] == "suite_total"]
    loads = [r for r in rows if r["stage"] == "checkpoint_load"]
    builds = [r for r in rows if r["stage"] == "build_wrapper"]
    assert len(suites) == 9
    assert len(loads) == 3
    assert len(builds) == 3

    # Warmup flag: iter 0 rows flagged warmup=true
    warmup_rows = [r for r in rows if r["warmup"] == "true"]
    assert all(r["iter"] == "0" for r in warmup_rows)


def test_macro_preflight_skip_respected(macro_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch(
        "scripts.profile_progression.runner.evaluate_single_checkpoint",
        _stub_evaluate_single_checkpoint,
    ):
        # preflight_skip=True is already set on the fixture; this test just asserts no raise.
        run_macro(macro_cfg, run_dir)
    assert (run_dir / "macro_wall_times.csv").exists()
