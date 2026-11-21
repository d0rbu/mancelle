"""CPU regression for timeout cleanup of torchrun's separate worker sessions."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.torchrun_helpers import run_torchrun


@pytest.mark.skipif(sys.platform != "linux", reason="Worker liveness uses Linux /proc")
def test_timeout_reaps_torchrun_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = tmp_path / "worker.py"
    worker_pid = tmp_path / "worker.pid"
    worker.write_text(
        "import os\n"
        "import signal\n"
        "import sys\n"
        "from pathlib import Path\n"
        "pending = Path(sys.argv[1] + '.pending')\n"
        "pending.write_text(str(os.getpid()))\n"
        "pending.replace(sys.argv[1])\n"
        "signal.pause()\n"
    )
    launchers = []

    class ReadyPopen(subprocess.Popen):
        first = True

        def communicate(self, input=None, timeout=None):
            if self.first:
                self.first = False
                launchers.append(self)
                # The tested timeout starts after readiness, not during cold
                # Torch imports. Bound only an unexpected startup failure.
                deadline = time.monotonic() + 120
                while not worker_pid.exists():
                    assert self.poll() is None, "torchrun exited before worker readiness"
                    assert time.monotonic() < deadline, "torchrun worker did not become ready"
                    time.sleep(0.02)
                timeout = 0.05
            return super().communicate(input, timeout=timeout)

    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=1", str(worker), str(worker_pid)]
    monkeypatch.setattr(subprocess, "Popen", ReadyPopen)
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            run_torchrun(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
                timeout=120,
            )
        assert error.value.cmd == command and error.value.timeout == 0.05
        assert len(launchers) == 1
        assert not (Path("/proc") / str(launchers[0].pid)).exists()
        assert not (Path("/proc") / worker_pid.read_text()).exists()
    finally:
        # Even a regression to subprocess.run (which kills only the launcher)
        # must not leave this test's orphaned worker alive after the failure.
        if worker_pid.exists():
            try:
                os.kill(int(worker_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in launchers:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
