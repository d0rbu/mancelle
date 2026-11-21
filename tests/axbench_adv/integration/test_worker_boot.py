"""Boot the real Python-3.12 worker command without loading a model or provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from steering.eval.axbench_adv.worker_client import (
    REPO_ROOT,
    AxBenchAdvJudgeClient,
    _FramedProcess,
    build_worker_command,
)


def test_real_vendored_worker_boot_has_protocol_clean_stdout(tmp_path: Path) -> None:
    process = _FramedProcess(
        build_worker_command(),
        cwd=REPO_ROOT,
        timeout_seconds=30.0,
        stderr_path=tmp_path / "worker.log",
    )
    with pytest.raises(AxBenchAdvError, match="init payload fields must be exactly"):
        process.request("init", {})
    log = (tmp_path / "worker.log").read_text(encoding="utf-8")
    assert "ValueError: init payload fields must be exactly" in log


def test_real_native_and_encdec_judge_cache_only_init_rebind_and_close(tmp_path: Path) -> None:
    (tmp_path / "judge.sqlite").touch()
    payload = {
        "api_base_url": "https://openrouter.ai/api/v1",
        "cache_only": True, "concept": "cats", "concept_id": 0, "factor": 1.0,
        "judge_cache": str(tmp_path / "judge.sqlite"),
        "judge_model": "openai/gpt-5.6-luna",
        "method": "NativeDecoderPromptSteering", "runtime_kind": "judge",
    }
    with AxBenchAdvJudgeClient(
        init_payload=payload,
        timeout_seconds=60.0, stderr_path=tmp_path / "judge.log",
    ) as client:
        assert not client._closed
        for method in ENCDEC_BASELINE_METHODS:
            client.rebind(payload | {"method": method})
            assert not client._closed
    assert client._closed
