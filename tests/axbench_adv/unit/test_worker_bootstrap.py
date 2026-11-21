"""Real subprocess coverage for import-time noise outside the framed channel."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.eval.axbench_adv.vendored.protocol import decode_frame

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_bootstrap_redirects_python_and_native_import_noise(tmp_path: Path) -> None:
    (tmp_path / "noisy_dependency.py").write_text(
        "import os\nimport sys\n"
        "print('dependency Python import noise', flush=True)\n"
        "sys.__stdout__.write('dependency original stdout noise\\n')\n"
        "sys.__stdout__.flush()\n"
        "os.write(1, b'dependency native import noise\\n')\n"
    )
    (tmp_path / "noisy_worker.py").write_text(
        "import json\nimport os\nimport struct\nimport noisy_dependency\n"
        "def serve(reader, writer):\n"
        "    print('worker runtime noise', flush=True)\n"
        "    os.write(1, b'worker native runtime noise\\n')\n"
        "    payload = json.dumps({'result': reader.read().decode()}).encode()\n"
        "    writer.write(struct.pack('>I', len(payload)) + payload)\n"
        "    writer.flush()\n"
        "    return 7\n"
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT)))}
    result = subprocess.run(
        [sys.executable, "-c",
         "from scripts.eval.axbench_adv.vendored.bootstrap import serve_module; "
         "raise SystemExit(serve_module('noisy_worker'))"],
        cwd=REPO_ROOT, env=environment, input=b"request", capture_output=True, timeout=10, check=False,
    )
    assert result.returncode == 7
    stream = io.BytesIO(result.stdout)
    assert decode_frame(stream) == {"result": "request"}
    assert stream.read() == b""
    for noise in (b"Python import", b"original stdout", b"native import", b"runtime noise", b"native runtime"):
        assert noise in result.stderr


@pytest.mark.parametrize("body,diagnostic", [
    ("raise ValueError('dependency unavailable')", b"dependency unavailable"),
    ("serve = None", b"must expose callable serve"),
])
def test_bootstrap_failure_keeps_stdout_empty(tmp_path: Path, body: str, diagnostic: bytes) -> None:
    (tmp_path / "broken_worker.py").write_text(f"print('noise before failure')\n{body}\n")
    result = subprocess.run(
        [sys.executable, "-c",
         "from scripts.eval.axbench_adv.vendored.bootstrap import serve_module; "
         "raise SystemExit(serve_module('broken_worker'))"],
        cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join((str(tmp_path), str(REPO_ROOT)))},
        capture_output=True, timeout=10, check=False,
    )
    assert result.returncode != 0
    assert result.stdout == b""
    assert b"noise before failure" in result.stderr
    assert diagnostic in result.stderr
