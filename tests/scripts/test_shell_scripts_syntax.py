"""Shell script syntax validation for slice 18.

Each ``runs/*.sh`` and ``scripts/*.sh`` plus ``scripts/slurm/*.sbatch`` must
pass ``bash -n``. This catches the classic case of an editor saving a script
with a syntax error (mismatched quote, broken heredoc, missing ``fi``)
that would only manifest when the operator runs it.

Slice 18 P0 fixes (#1, #3) re-edited shell scripts; this test prevents
regression.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SHELL_SCRIPTS = sorted(
    [
        *REPO_ROOT.glob("runs/*.sh"),
        *(REPO_ROOT / "scripts").glob("*.sh"),
        *(REPO_ROOT / "scripts" / "slurm").glob("*.sbatch"),
    ]
)


@pytest.mark.skipif(not shutil.which("bash"), reason="bash not available on this system")
@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_shell_script_passes_bash_n(script: Path) -> None:
    """``bash -n <script>`` must succeed with exit code 0."""
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, (
        f"{script.relative_to(REPO_ROOT)} fails ``bash -n``:\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
