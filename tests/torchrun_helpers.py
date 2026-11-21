"""Subprocess capture for torchrun tests that must clean up worker sessions."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_torchrun(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Capture a launcher result, allowing torchrun to reap workers on failure."""
    with subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            # Workers have separate sessions. SIGKILL of just the launcher
            # or its process group would bypass torchrun's worker cleanup.
            process.terminate()
            try:
                process.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                # Descendants may still own the captured pipes; reap only
                # the launcher rather than waiting for pipe EOF here.
                process.wait()
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
