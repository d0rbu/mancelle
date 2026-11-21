from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts/slurm/preflight_nccl_transport.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _fake_slurm_bin(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "scontrol",
        "#!/bin/bash\nprintf 'node-a\\nnode-b\\n'\n",
    )
    _write_executable(
        fake_bin / "getent",
        "#!/bin/bash\nprintf '10.0.0.1 STREAM node-a\\n'\n",
    )
    _write_executable(fake_bin / "timeout", "#!/bin/bash\nexit 0\n")
    _write_executable(
        fake_bin / "srun",
        """#!/bin/bash
if [[ "$*" == *"nvidia-smi --query-gpu"* ]]; then
    exit 0
fi
if [[ "${NCCL_NET_GDR_LEVEL:-}" == "LOC" ]]; then
    exit "${FAKE_HOST_STAGED_EXIT:-0}"
fi
exit "${FAKE_DEFAULT_EXIT:-0}"
""",
    )
    return fake_bin


def _run_wrapper(
    tmp_path: Path,
    *,
    default_exit: int,
    host_staged_exit: int,
) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_slurm_bin(tmp_path)
    result_dir = tmp_path / "result"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SLURM_JOB_NODELIST": "node-[a-b]",
        "SLURM_NNODES": "2",
        "SLURM_JOB_ID": "123",
        "SLURM_SUBMIT_DIR": str(REPO_ROOT),
        "PREFLIGHT_RESULT_DIR": str(result_dir),
        "GPUS_PER_NODE": "2",
        "FAKE_DEFAULT_EXIT": str(default_exit),
        "FAKE_HOST_STAGED_EXIT": str(host_staged_exit),
    }
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize(
    ("default_exit", "host_staged_exit", "expected_status", "expected_line"),
    [
        (0, 1, 0, "unset NCCL_NET_GDR_LEVEL"),
        (1, 0, 0, "export NCCL_NET_GDR_LEVEL=LOC"),
        (1, 1, 1, "# Both default and host-staged NCCL preflights failed."),
    ],
)
def test_wrapper_selects_only_a_passing_transport(
    tmp_path: Path,
    default_exit: int,
    host_staged_exit: int,
    expected_status: int,
    expected_line: str,
) -> None:
    result = _run_wrapper(
        tmp_path,
        default_exit=default_exit,
        host_staged_exit=host_staged_exit,
    )

    assert result.returncode == expected_status, result.stderr
    selection = (tmp_path / "result/transport.env").read_text()
    assert expected_line in selection
