"""Root worker-client tests, including real subprocess failure modes."""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_adv import worker_client
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.prompt_surgery import QWEN_TERMINAL_MARKER, compose_suffix
from steering.eval.axbench_adv.seeding import row_seed
from steering.eval.axbench_adv.worker_client import (
    AxBenchAdvJudgeClient,
    AxBenchAdvWorkerClient,
    WorkerExample,
    _FramedProcess,
    build_worker_command,
)

CHILD_MODULE = "tests.axbench_adv.fixtures.framed_worker_child"


def _child_command(mode: str, *args: Path) -> list[str]:
    return [sys.executable, "-m", CHILD_MODULE, mode, *(str(arg) for arg in args)]


def _process(tmp_path: Path, mode: str, *args: Path, timeout: float = 2.0) -> _FramedProcess:
    return _FramedProcess(
        _child_command(mode, *args),
        cwd=worker_client.REPO_ROOT,
        timeout_seconds=timeout,
        stderr_path=tmp_path / f"{mode}.log",
    )


@pytest.mark.parametrize("mode", ["success", "partial"])
def test_framed_process_real_success_partial_reads_and_close(tmp_path: Path, mode: str) -> None:
    process = _process(tmp_path, mode)
    assert process.request("ping", {"unicode": "café"}) == {"unicode": "café"}
    process.close()
    assert f"child-mode={mode}" in (tmp_path / f"{mode}.log").read_text()


def test_framed_process_default_temporary_log_and_normal_context_exit(tmp_path: Path) -> None:
    sidecar = tempfile.TemporaryDirectory(dir=tmp_path)
    sidecar_path = Path(sidecar.name)
    (sidecar_path / "gradients.safetensors").write_bytes(b"snapshot")
    process = _FramedProcess(
        _child_command("success"),
        cwd=worker_client.REPO_ROOT,
        timeout_seconds=2.0,
        stderr_path=None,
        cleanup_callback=sidecar.cleanup,
    )
    temporary_root = process.stderr_path.parent
    with process:
        assert process.request("ping", {}) == {}
    assert not temporary_root.exists()
    assert not sidecar_path.exists()
    process.close()


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("eof", "EOFError"),
        ("timeout", "TimeoutError"),
        ("corrupt", "ProtocolError"),
        ("wrong-id", "id mismatch"),
        ("wrong-op", "operation mismatch"),
        ("wrong-version", "version mismatch"),
        ("bad-fields", "fields"),
        ("bad-status", "malformed"),
        ("ignore-term", "TimeoutError"),
    ],
)
def test_framed_process_fails_closed_on_transport_and_response_errors(
    tmp_path: Path, mode: str, match: str
) -> None:
    process = _process(tmp_path, mode, timeout=0.2)
    with pytest.raises(AxBenchAdvError, match=match) as exc_info:
        process.request("ping", {})
    assert process._closed is True
    if mode in {"eof", "timeout", "corrupt"}:
        assert f"child-mode={mode}" in str(exc_info.value)
    with pytest.raises(AxBenchAdvError, match="closed"):
        process.request("again", {})


def test_framed_process_surfaces_valid_worker_error_and_persistent_log(tmp_path: Path) -> None:
    process = _process(tmp_path, "error")
    with pytest.raises(AxBenchAdvError, match="ValueError: deliberate child failure"):
        process.request("ping", {})
    assert "child-mode=error" in (tmp_path / "error.log").read_text()


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("close-nonempty", "result must be empty"),
        ("close-hang", "did not exit"),
        ("close-nonzero", "exited nonzero"),
    ],
)
def test_close_fails_on_nonempty_ack_hung_exit_or_nonzero_exit(
    tmp_path: Path, mode: str, match: str
) -> None:
    process = _process(tmp_path, mode, timeout=0.3)
    assert process.request("ping", {}) == {}
    with pytest.raises(AxBenchAdvError, match=match):
        process.close()


def _process_is_live(pid: int) -> bool:
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    return stat.read_text().split()[2] != "Z"


def test_timeout_kills_the_worker_process_group_not_only_uv_parent(tmp_path: Path) -> None:
    pid_path = tmp_path / "grandchild.pid"
    process = _process(tmp_path, "grandchild", pid_path, timeout=0.4)
    with pytest.raises(AxBenchAdvError, match="TimeoutError"):
        process.request("ping", {})
    assert pid_path.is_file()
    grandchild_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2.0
    while _process_is_live(grandchild_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_is_live(grandchild_pid)


def test_context_manager_does_not_mask_an_active_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="caller failure"):
        with _process(tmp_path, "timeout", timeout=1.0):
            raise RuntimeError("caller failure")


def test_timeout_and_command_validation(tmp_path: Path) -> None:
    for timeout in (True, 0, -1, float("inf"), "1"):
        with pytest.raises(AxBenchAdvError, match="timeout_seconds"):
            _FramedProcess(
                _child_command("success"),
                cwd=worker_client.REPO_ROOT,
                timeout_seconds=timeout,  # ty: ignore[invalid-argument-type]
                stderr_path=tmp_path / f"bad-{timeout}.log",
            )
    assert build_worker_command() == [
        "uv",
        "run",
        "--project",
        str(worker_client.VENDOR_ROOT),
        "--frozen",
        "python",
        "-m",
        worker_client.WORKER_MODULE,
    ]
    with pytest.raises(FileNotFoundError):
        _FramedProcess(
            ["/definitely/not/a/worker"],
            cwd=worker_client.REPO_ROOT,
            timeout_seconds=1.0,
            stderr_path=None,
        )


def test_deadline_guard_stderr_fallback_and_direct_response_validation(tmp_path: Path) -> None:
    process = _process(tmp_path, "success")
    process._deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match="timed out"):
        process._read_exact(process._stdout, 1)
    process._abort()
    process._abort()
    process.stderr_path.unlink()
    assert process._stderr_tail() == "<stderr unavailable>"

    with pytest.raises(worker_client.ProtocolError, match="JSON object"):
        _FramedProcess._validate_response([], request_id=1, operation="op")
    for error in (
        "not-a-dict",
        {"message": "m"},
        {"message": "m", "type": 1},
    ):
        with pytest.raises(worker_client.ProtocolError, match="error response"):
            _FramedProcess._validate_response(
                {
                    "error": error,
                    "id": 1,
                    "op": "op",
                    "status": "error",
                    "version": 1,
                },
                request_id=1,
                operation="op",
            )
    for status, result in (("mystery", {}), ("ok", [])):
        with pytest.raises(worker_client.ProtocolError, match="success response"):
            _FramedProcess._validate_response(
                {
                    "id": 1,
                    "op": "op",
                    "result": result,
                    "status": status,
                    "version": 1,
                },
                request_id=1,
                operation="op",
            )
    for mutation, match in (({"version": True}, "version mismatch"), ({"id": True}, "id mismatch")):
        response = {
            "id": 1,
            "op": "op",
            "result": {},
            "status": "ok",
            "version": 1,
        }
        with pytest.raises(worker_client.ProtocolError, match=match):
            _FramedProcess._validate_response(
                response | mutation,
                request_id=1,
                operation="op",
            )


def test_abort_tolerates_process_disappearing_during_both_signals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class VanishingProcess:
        pid = 123

        @staticmethod
        def poll() -> None:
            return None

        def wait(self, timeout: float) -> int:
            del timeout
            if not hasattr(self, "waited"):
                self.waited = True
                raise worker_client.subprocess.TimeoutExpired("worker", 1)
            return 0

    class ClosedStream:
        def close(self) -> None:
            return None

    process = object.__new__(_FramedProcess)
    process._closed = False
    process._process = VanishingProcess()  # ty: ignore[invalid-assignment]
    process._stdin = ClosedStream()  # ty: ignore[invalid-assignment]
    process._stdout = ClosedStream()  # ty: ignore[invalid-assignment]
    process._temporary = None
    process.stderr_path = tmp_path / "missing.log"
    monkeypatch.setattr(
        worker_client.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    process._abort()
    assert process._closed is True


class FakeFramedProcess:
    instances: list[FakeFramedProcess] = []

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.command = command
        self.kwargs = kwargs
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self.aborted = False
        self.init_payload: dict[str, Any] | None = None
        self.instances.append(self)

    def request(self, operation: str, payload: Any) -> dict[str, Any]:
        copied = dict(payload)
        self.requests.append((operation, copied))
        if operation in {"init", "rebind"}:
            self.init_payload = copied
            return {
                "concept_id": copied["concept_id"],
                "factor": float(copied["factor"]),
                "method": copied["method"],
            }
        if operation == "evaluate":
            return {
                "concept_rating": 0.25,
                "fluency_rating": 1.75,
                "generation": f"generation-{copied['input_id']}",
                "input_id": copied["input_id"],
                "judge_rationale": '{"concept":"c","fluency":"f"}',
                "paid_judge_cache_misses": 2,
            }
        if operation == "close":
            return {}
        raise AssertionError(operation)

    def close(self) -> None:
        self.closed = True

    def _abort(self) -> None:
        self.aborted = True


def _init_payload(method: str) -> dict[str, Any]:
    return {"method": method, "concept_id": 3, "factor": 1.5}


def _judge_payload() -> dict[str, Any]:
    return {
        "api_base_url": "http://localhost:8000/v1",
        "cache_only": True,
        "concept": "concept",
        "concept_id": 3,
        "factor": 6.0,
        "judge_cache": "/tmp/judge.sqlite",
        "judge_model": "fake/judge",
        "method": "SelfConditioning",
        "runtime_kind": "judge",
    }


@pytest.mark.parametrize(
    ("method", "target"),
    [
        ("DiffMean", "input"),
        ("LinearProbe", "input"),
        ("LoRA", "input"),
        ("PromptSteering", "steered_input"),
    ],
)
def test_public_client_applies_exact_prompt_and_seed_contract(
    monkeypatch: pytest.MonkeyPatch, method: str, target: str
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    plain = f"plain{QWEN_TERMINAL_MARKER}"
    steered = f"steered{QWEN_TERMINAL_MARKER}"
    example = WorkerExample(7, plain, steered)
    client = AxBenchAdvWorkerClient(
        init_payload=_init_payload(method),
        examples=[example],
        master_seed=99,
    )
    result = client.evaluate("  coherent suffix  ", [7])
    assert result[0].generation == "generation-7"
    assert result[0].concept_rating == 0.25
    process = FakeFramedProcess.instances[-1]
    operation, payload = process.requests[-1]
    assert operation == "evaluate"
    unattacked = plain if target == "input" else steered
    assert payload[target] == compose_suffix(unattacked, "coherent suffix")
    other = "steered_input" if target == "input" else "input"
    assert payload[other] == getattr(example, other)
    assert payload["row_seed"] == row_seed(
        master_seed=99,
        method=method,
        concept_id=3,
        input_id=7,
        factor=1.5,
        unattacked_prompt=unattacked,
    )
    client.close()
    assert process.closed is True


def test_public_client_empty_suffix_is_identity_and_context_aborts_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    prompt = f"prompt{QWEN_TERMINAL_MARKER}"
    with pytest.raises(RuntimeError, match="body"):
        with AxBenchAdvWorkerClient(
            init_payload=_init_payload("LoRA"),
            examples=[WorkerExample(0, prompt, prompt)],
            master_seed=0,
        ) as client:
            client.evaluate("", [0])
            assert FakeFramedProcess.instances[-1].requests[-1][1]["input"] == prompt
            raise RuntimeError("body")
    assert FakeFramedProcess.instances[-1].aborted is True


def test_public_client_normal_context_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    with AxBenchAdvWorkerClient(
        init_payload=_init_payload("LoRA"),
        examples=[WorkerExample(0, "x", "y")],
        master_seed=0,
    ):
        pass
    assert FakeFramedProcess.instances[-1].closed is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"method": "unknown"}, "unknown worker method"),
        ({"concept_id": True}, "concept_id"),
        ({"factor": True}, "factor must be numeric"),
        ({"factor": math.nan}, "factor must be finite"),
    ],
)
def test_public_client_rejects_bad_identity_before_spawn(
    monkeypatch: pytest.MonkeyPatch, mutation: dict[str, Any], match: str
) -> None:
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    payload = _init_payload("LoRA") | mutation
    with pytest.raises(AxBenchAdvError, match=match):
        AxBenchAdvWorkerClient(
            init_payload=payload,
            examples=[WorkerExample(0, "x", "y")],
            master_seed=0,
        )


@pytest.mark.parametrize("seed", [True, -1, 1 << 63])
def test_public_client_rejects_bad_master_seed(monkeypatch: pytest.MonkeyPatch, seed: Any) -> None:
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    with pytest.raises(AxBenchAdvError, match="master_seed"):
        AxBenchAdvWorkerClient(
            init_payload=_init_payload("LoRA"),
            examples=[WorkerExample(0, "x", "y")],
            master_seed=seed,
        )


@pytest.mark.parametrize(
    ("examples", "match"),
    [
        ([], "must not be empty"),
        ([object()], "WorkerExample"),
        ([WorkerExample(0, "x", "y"), WorkerExample(0, "a", "b")], "duplicate"),
    ],
)
def test_public_client_rejects_bad_examples(
    monkeypatch: pytest.MonkeyPatch, examples: Any, match: str
) -> None:
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    with pytest.raises(AxBenchAdvError, match=match):
        AxBenchAdvWorkerClient(init_payload=_init_payload("LoRA"), examples=examples, master_seed=0)


def test_public_client_rejects_bad_requests_and_closed_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    prompt = f"p{QWEN_TERMINAL_MARKER}"
    client = AxBenchAdvWorkerClient(
        init_payload=_init_payload("LoRA"),
        examples=[WorkerExample(0, prompt, prompt)],
        master_seed=0,
    )
    with pytest.raises(AxBenchAdvError, match="unknown worker input_id"):
        client.evaluate("x", [1])
    client.close()
    client.close()
    with pytest.raises(AxBenchAdvError, match="closed"):
        client.evaluate("x", [0])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"extra": 1}, "fields"),
        ({"input_id": 1}, "input_id mismatch"),
        ({"input_id": True}, "non-negative integer"),
        ({"input_id": -1}, "non-negative integer"),
        ({"generation": 1}, "must be strings"),
        ({"concept_rating": True}, "must be numeric"),
        ({"fluency_rating": float("inf")}, "finite in"),
        ({"paid_judge_cache_misses": True}, "must be in"),
        ({"paid_judge_cache_misses": 3}, "must be in"),
    ],
)
def test_raw_evaluation_parser_fails_closed(mutation: dict[str, Any], match: str) -> None:
    value = {
        "concept_rating": 1.0,
        "fluency_rating": 2.0,
        "generation": "answer",
        "input_id": 0,
        "judge_rationale": "reason",
        "paid_judge_cache_misses": 0,
    }
    if "extra" in mutation:
        value.update(mutation)
    else:
        value |= mutation
    with pytest.raises(AxBenchAdvError, match=match):
        AxBenchAdvWorkerClient._raw_evaluation(value, input_id=0)


def test_constructor_aborts_worker_on_init_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongIdentity(FakeFramedProcess):
        def request(self, operation: str, payload: Any) -> dict[str, Any]:
            result = super().request(operation, payload)
            if operation == "init":
                result["factor"] = 9.0
            return result

    WrongIdentity.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", WrongIdentity)
    with pytest.raises(AxBenchAdvError, match="identity mismatch"):
        AxBenchAdvWorkerClient(
            init_payload=_init_payload("LoRA"),
            examples=[WorkerExample(0, "x", "y")],
            master_seed=0,
        )
    assert WrongIdentity.instances[-1].aborted is True


def test_constructor_rejects_bool_identity_equal_to_numeric_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoolIdentity(FakeFramedProcess):
        def request(self, operation: str, payload: Any) -> dict[str, Any]:
            result = super().request(operation, payload)
            if operation == "init":
                result["concept_id"] = True
            return result

    BoolIdentity.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", BoolIdentity)
    with pytest.raises(AxBenchAdvError, match="identity mismatch"):
        AxBenchAdvWorkerClient(
            init_payload={"method": "LoRA", "concept_id": 1, "factor": 1.0},
            examples=[WorkerExample(0, "x", "y")],
            master_seed=0,
        )
    assert BoolIdentity.instances[-1].aborted is True


def test_judge_client_uses_exact_init_and_generation_only_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    with AxBenchAdvJudgeClient(init_payload=_judge_payload()) as client:
        result = client.evaluate(generation="exact generated bytes café", input_id=7)
        assert result.generation == "generation-7"
    process = FakeFramedProcess.instances[-1]
    assert process.requests == [
        ("init", _judge_payload()),
        ("evaluate", {"generation": "exact generated bytes café", "input_id": 7}),
    ]
    assert process.closed


def test_judge_client_accepts_bounded_project_pair_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    payload = _judge_payload() | {"pair_concurrency": 2}
    with AxBenchAdvJudgeClient(init_payload=payload):
        pass
    assert FakeFramedProcess.instances[-1].requests[0] == ("init", payload)


def test_judge_client_rebinds_identity_without_restarting_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    initial = _judge_payload()
    changed = initial | {"method": "SelfConditioningMasked", "concept_id": 4}
    with AxBenchAdvJudgeClient(init_payload=initial) as client:
        client.rebind(changed)
        assert client.evaluate(generation="text", input_id=0).generation == "generation-0"
    assert len(FakeFramedProcess.instances) == 1
    process = FakeFramedProcess.instances[0]
    assert process.requests == [
        ("init", initial),
        ("rebind", changed),
        ("evaluate", {"generation": "text", "input_id": 0}),
    ]
    assert process.closed


def test_judge_client_rebind_identity_mismatch_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongRebind(FakeFramedProcess):
        def request(self, operation: str, payload: Any) -> dict[str, Any]:
            result = super().request(operation, payload)
            if operation == "rebind":
                result["method"] = "LoRA"
            return result

    WrongRebind.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", WrongRebind)
    client = AxBenchAdvJudgeClient(init_payload=_judge_payload())
    with pytest.raises(AxBenchAdvError, match="rebind identity mismatch"):
        client.rebind(_judge_payload() | {"method": "NLSteering"})
    assert WrongRebind.instances[-1].aborted
    with pytest.raises(AxBenchAdvError, match="closed"):
        client.evaluate(generation="text", input_id=0)


def test_judge_client_rebind_rejects_bad_payload_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    client = AxBenchAdvJudgeClient(init_payload=_judge_payload())
    process = FakeFramedProcess.instances[-1]
    with pytest.raises(AxBenchAdvError, match="factor must be finite"):
        client.rebind(_judge_payload() | {"factor": math.inf})
    assert [operation for operation, _payload in process.requests] == ["init"]
    client.close()


def test_judge_client_rebind_worker_error_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRebind(FakeFramedProcess):
        def request(self, operation: str, payload: Any) -> dict[str, Any]:
            if operation == "rebind":
                raise AxBenchAdvError("worker rejected rebind")
            return super().request(operation, payload)

    FailingRebind.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FailingRebind)
    client = AxBenchAdvJudgeClient(init_payload=_judge_payload())
    with pytest.raises(AxBenchAdvError, match="worker rejected rebind"):
        client.rebind(_judge_payload() | {"method": "NLSteering"})
    assert FailingRebind.instances[-1].aborted


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"extra": 1}, "fields"),
        ({"runtime_kind": "worker"}, "runtime_kind"),
        ({"method": "LoRA"}, "method is not a supported"),
        ({"concept_id": True}, "concept_id"),
        ({"factor": True}, "factor must be numeric"),
        ({"factor": math.inf}, "factor must be finite"),
        ({"pair_concurrency": True}, "pair_concurrency"),
        ({"pair_concurrency": 3}, "pair_concurrency"),
    ],
)
def test_judge_client_rejects_invalid_identity_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, Any],
    match: str,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    with pytest.raises(AxBenchAdvError, match=match):
        AxBenchAdvJudgeClient(init_payload=_judge_payload() | mutation)
    assert not FakeFramedProcess.instances


def test_judge_client_rejects_bad_requests_closed_reuse_and_wrong_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    client = AxBenchAdvJudgeClient(init_payload=_judge_payload())
    for generation, input_id, match in (
        (1, 0, "generation"),
        ("\ud800", 0, "UTF-8"),
        ("ok", True, "input_id"),
        ("ok", -1, "input_id"),
    ):
        with pytest.raises(AxBenchAdvError, match=match):
            client.evaluate(generation=generation, input_id=input_id)  # ty: ignore[invalid-argument-type]
    client.close()
    client.close()
    with pytest.raises(AxBenchAdvError, match="closed"):
        client.evaluate(generation="ok", input_id=0)

    class WrongIdentity(FakeFramedProcess):
        def request(self, operation: str, payload: Any) -> dict[str, Any]:
            result = super().request(operation, payload)
            if operation == "init":
                result["method"] = "LoRA"
            return result

    WrongIdentity.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", WrongIdentity)
    with pytest.raises(AxBenchAdvError, match="identity mismatch"):
        AxBenchAdvJudgeClient(init_payload=_judge_payload())
    assert WrongIdentity.instances[-1].aborted


def test_judge_client_context_aborts_on_active_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeFramedProcess.instances.clear()
    monkeypatch.setattr(worker_client, "_FramedProcess", FakeFramedProcess)
    with pytest.raises(RuntimeError, match="body"):
        with AxBenchAdvJudgeClient(init_payload=_judge_payload()):
            raise RuntimeError("body")
    assert FakeFramedProcess.instances[-1].aborted
