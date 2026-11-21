"""Run AxBench-Adv integration suites in AxBench's locked Python 3.12 environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_vendored_axbench_suites() -> None:
    root = Path(__file__).resolve().parents[3]
    suites = [
        Path(__file__).with_name("m4_inference_override_suite.py"),
        Path(__file__).with_name("lora_release_suite.py"),
        Path(__file__).with_name("phase4_worker_suite.py"),
        Path(__file__).with_name("pilot_budget_sdk_suite.py"),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "vendor/axbench"), str(root)))
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(root / "vendor/axbench"),
            "--frozen",
            "--offline",
            "--with",
            "pytest==9.0.2",
            "python",
            "-m",
            "pytest",
            "--confcutdir",
            str(root / "tests/axbench_adv"),
            *(str(suite) for suite in suites),
            "-q",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
