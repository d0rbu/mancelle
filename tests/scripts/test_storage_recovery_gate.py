"""Bounded Slurm storage recovery gate contracts for issue #1569."""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.utils.storage_recovery_gate as gate


def test_recovery_wait_policy_caps_scheduled_sleep_at_ten_minutes() -> None:
    assert gate._MAX_PROBES == 3
    assert (gate._MAX_PROBES - 1) * gate._RETRY_DELAY_S == 600


def _measurement(*, bulk_mb_s: float, small_loop_s: float) -> dict[str, object]:
    return {
        "status": "measurement",
        "checkpoint_file": "/group/checkpoints/model_035000.pt",
        "bulk_offset": 0,
        "bulk_bytes": gate._BULK_READ_BYTES,
        "bulk_elapsed_s": 1.0,
        "bulk_mb_s": bulk_mb_s,
        "small_files": gate._SMALL_FILE_COUNT,
        "small_read_bytes": gate._SMALL_READ_BYTES,
        "small_loop_s": small_loop_s,
    }


@pytest.mark.parametrize(
    ("bulk_mb_s", "small_loop_s", "expected"),
    [
        (100.0, 1.999999, True),
        (99.999999, 1.0, False),
        (1000.0, 2.0, False),
        (1000.0, 2.000001, False),
        (100.0, 0.0, True),
    ],
)
def test_health_threshold_boundaries_are_exact(
    bulk_mb_s: float,
    small_loop_s: float,
    expected: bool,
) -> None:
    assert (
        gate._measurement_is_healthy(_measurement(bulk_mb_s=bulk_mb_s, small_loop_s=small_loop_s))
        is expected
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bulk_mb_s", float("nan")),
        ("bulk_mb_s", float("inf")),
        ("bulk_mb_s", -1.0),
        ("small_loop_s", float("nan")),
        ("small_loop_s", float("inf")),
        ("small_loop_s", -0.1),
    ],
)
def test_nonfinite_or_negative_measurements_are_configuration_errors(
    key: str,
    value: float,
) -> None:
    payload = _measurement(bulk_mb_s=100.0, small_loop_s=1.0)
    payload[key] = value
    with pytest.raises(gate._ProbeConfigError, match="invalid"):
        gate._measurement_is_healthy(payload)


@pytest.mark.parametrize("missing_key", ["bulk_mb_s", "small_loop_s"])
def test_missing_measurement_fields_are_configuration_errors(missing_key: str) -> None:
    payload = _measurement(bulk_mb_s=100.0, small_loop_s=1.0)
    del payload[missing_key]
    with pytest.raises(gate._ProbeConfigError, match="malformed"):
        gate._measurement_is_healthy(payload)


@pytest.mark.parametrize("value", ["100.0", True, None, [], {}])
def test_nonnumeric_measurement_fields_are_rejected(value: object) -> None:
    payload = _measurement(bulk_mb_s=100.0, small_loop_s=1.0)
    payload["bulk_mb_s"] = value
    with pytest.raises(gate._ProbeConfigError, match="malformed"):
        gate._measurement_is_healthy(payload)


def test_worker_payload_uses_the_last_valid_json_object() -> None:
    stdout = "srun: informational preface\nnot json\n"
    stdout += json.dumps({"status": "probe_error", "message": "first"}) + "\n"
    stdout += json.dumps({"status": "measurement", "bulk_mb_s": 123.0}) + "\n"
    assert gate._worker_payload(stdout) == {"status": "measurement", "bulk_mb_s": 123.0}


@pytest.mark.parametrize("stdout", ["", "not-json\n[]\n", '{"message": "no status"}\n'])
def test_worker_payload_rejects_missing_status_objects(stdout: str) -> None:
    assert gate._worker_payload(stdout) is None


def test_compute_probe_uses_timeout_and_allocation_local_srun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_command: list[str] = []
    observed_kwargs: dict[str, object] = {}
    payload = _measurement(bulk_mb_s=222.0, small_loop_s=0.2)

    def fake_run(command, **kwargs):
        observed_command.extend(command)
        observed_kwargs.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    checkpoint = tmp_path / "checkpoint path/model_035000.pt"
    selection = tmp_path / "selection path/sel/hash"
    result = gate._run_compute_probe(checkpoint, selection)

    assert result == payload
    assert observed_command[:8] == [
        "timeout",
        "--signal=TERM",
        f"--kill-after={gate._PROBE_KILL_GRACE_S}s",
        f"{gate._PROBE_TIMEOUT_S}s",
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--overlap",
    ]
    assert observed_command[8] == gate.sys.executable
    assert observed_command[10] == "--worker"
    assert observed_command[observed_command.index("--checkpoint-file") + 1] == os.fspath(
        checkpoint
    )
    assert observed_command[observed_command.index("--selection-dir") + 1] == os.fspath(selection)
    assert observed_kwargs == {
        "check": False,
        "capture_output": True,
        "text": True,
    }


@pytest.mark.parametrize(
    ("returncode", "stdout", "worker_status"),
    [
        (124, "", None),
        (137, "", None),
        (124, json.dumps(_measurement(bulk_mb_s=500.0, small_loop_s=0.1)), "measurement"),
    ],
)
def test_compute_probe_classifies_timeout_as_retryable(
    returncode: int,
    stdout: str,
    worker_status: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            returncode,
            stdout=stdout,
            stderr="Terminated\n",
        ),
    )
    result = gate._run_compute_probe(tmp_path / "model.pt", tmp_path / "sel")
    assert result["status"] == "probe_error"
    assert result["type"] == "Timeout"
    assert "exceeded 60s" in result["message"]
    assert f"worker_status={worker_status}" in result["message"]


def test_compute_probe_rejects_missing_json_without_retryable_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="broken output",
            stderr="srun failed before worker startup",
        ),
    )
    result = gate._run_compute_probe(tmp_path / "model.pt", tmp_path / "sel")
    assert result["status"] == "fatal_error"
    assert "without worker JSON" in result["message"]
    assert "srun failed before worker startup" in result["message"]


def test_compute_probe_rejects_measurement_from_failed_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _measurement(bulk_mb_s=200.0, small_loop_s=0.1)
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            9,
            stdout=json.dumps(payload),
            stderr="launcher reported failure",
        ),
    )
    result = gate._run_compute_probe(tmp_path / "model.pt", tmp_path / "sel")
    assert result["status"] == "fatal_error"
    assert "emitted a measurement" in result["message"]
    assert "launcher reported failure" in result["message"]


def test_compute_probe_classifies_launcher_exec_failure_as_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("srun vanished")),
    )
    result = gate._run_compute_probe(tmp_path / "model.pt", tmp_path / "sel")
    assert result == {
        "status": "fatal_error",
        "type": "OSError",
        "message": "could not launch allocation-local storage probe: srun vanished",
    }


def test_driver_requires_a_slurm_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    with pytest.raises(gate._ProbeConfigError, match="inside a Slurm allocation"):
        gate._require_slurm_allocation()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node001")
    assert gate._require_slurm_allocation() is None


def test_healthy_first_probe_returns_without_sleep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probes: list[int] = []
    sleeps: list[float] = []
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)

    def fake_probe(*args, **kwargs):
        probes.append(1)
        return _measurement(bulk_mb_s=100.0, small_loop_s=1.999)

    monkeypatch.setattr(gate, "_run_compute_probe", fake_probe)
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)

    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 0
    assert probes == [1]
    assert sleeps == []
    captured = capsys.readouterr()
    assert "storage probe attempt=1/3" in captured.out
    assert "storage_gate_status=healthy" in captured.out
    assert "exhausted" not in captured.err


def test_sick_sick_healthy_retries_with_exact_delays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = iter(
        [
            _measurement(bulk_mb_s=99.0, small_loop_s=0.1),
            _measurement(bulk_mb_s=500.0, small_loop_s=2.0),
            _measurement(bulk_mb_s=500.0, small_loop_s=0.1),
        ]
    )
    probes: list[int] = []
    sleeps: list[float] = []
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)

    def fake_probe(*args, **kwargs):
        probes.append(len(probes) + 1)
        return next(results)

    monkeypatch.setattr(gate, "_run_compute_probe", fake_probe)
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)

    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 0
    assert probes == [1, 2, 3]
    assert sleeps == [gate._RETRY_DELAY_S, gate._RETRY_DELAY_S]


def test_all_sick_probes_exhaust_once_then_proceed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    probes: list[int] = []
    sleeps: list[float] = []
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)

    def fake_probe(*args, **kwargs):
        probes.append(len(probes) + 1)
        return _measurement(bulk_mb_s=1.0, small_loop_s=20.0)

    monkeypatch.setattr(gate, "_run_compute_probe", fake_probe)
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)

    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 0
    assert probes == list(range(1, gate._MAX_PROBES + 1))
    assert sleeps == [gate._RETRY_DELAY_S] * (gate._MAX_PROBES - 1)
    captured = capsys.readouterr()
    assert captured.out.count("storage probe attempt=") == gate._MAX_PROBES
    assert captured.err.count("storage_gate_status=unhealthy") == gate._MAX_PROBES - 1
    assert captured.err.count("storage_gate_status=exhausted_proceeding") == 1


def test_fatal_driver_outcome_stops_without_sleep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)
    monkeypatch.setattr(
        gate,
        "_run_compute_probe",
        lambda *args, **kwargs: {"status": "fatal_error", "message": "bad invocation"},
    )
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)
    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 2
    assert sleeps == []


def test_malformed_measurement_is_a_fatal_worker_protocol_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)
    monkeypatch.setattr(
        gate,
        "_run_compute_probe",
        lambda *args, **kwargs: {"status": "measurement", "bulk_mb_s": "fast"},
    )
    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 2
    assert "storage_gate_status=fatal_error" in capsys.readouterr().err


def test_unknown_driver_payload_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)
    monkeypatch.setattr(
        gate,
        "_run_compute_probe",
        lambda *args, **kwargs: {"status": "mystery"},
    )
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)
    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 2
    assert sleeps == []


def test_probe_error_is_retryable_but_still_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []
    monkeypatch.setattr(gate, "_MAX_PROBES", 2)
    monkeypatch.setattr(gate, "_require_slurm_allocation", lambda: None)

    def fake_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"status": "probe_error", "type": "OSError", "message": "NFS stalled"}

    monkeypatch.setattr(gate, "_run_compute_probe", fake_probe)
    monkeypatch.setattr(gate.time, "sleep", sleeps.append)
    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 0
    assert calls == 2
    assert sleeps == [gate._RETRY_DELAY_S]


def test_missing_allocation_returns_status_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gate,
        "_require_slurm_allocation",
        lambda: (_ for _ in ()).throw(gate._ProbeConfigError("no allocation")),
    )
    assert gate._driver_main(tmp_path / "model.pt", tmp_path / "sel") == 2


def test_main_uses_lexical_absolute_paths_without_resolving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[Path, Path]] = []

    def forbidden_resolve(*args, **kwargs):
        raise AssertionError("Path.resolve must not touch the storage path")

    def fake_driver(checkpoint: Path, selection: Path) -> int:
        observed.append((checkpoint, selection))
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    monkeypatch.setattr(gate, "_driver_main", fake_driver)
    assert (
        gate.main(
            [
                "--checkpoint-file",
                "checkpoint path/model_000001.pt",
                "--selection-dir",
                "selection path/sel/hash",
            ]
        )
        == 0
    )
    assert observed == [
        (
            Path(os.path.abspath("checkpoint path/model_000001.pt")),
            Path(os.path.abspath("selection path/sel/hash")),
        )
    ]


def test_worker_main_dispatches_lexical_paths_without_calling_the_driver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[Path, Path]] = []

    def fake_worker(checkpoint: Path, selection: Path) -> int:
        observed.append((checkpoint, selection))
        return 7

    def forbidden_driver(*args: object, **kwargs: object) -> int:
        raise AssertionError("worker mode must not call the retry driver")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate, "_worker_main", fake_worker)
    monkeypatch.setattr(gate, "_driver_main", forbidden_driver)

    assert (
        gate.main(
            [
                "--worker",
                "--checkpoint-file",
                "checkpoint/model.pt",
                "--selection-dir",
                "selection/sel/hash",
            ]
        )
        == 7
    )
    assert observed == [
        (
            Path(os.path.abspath("checkpoint/model.pt")),
            Path(os.path.abspath("selection/sel/hash")),
        )
    ]


def test_selection_inventory_is_direct_child_sorted_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_SMALL_READ_BYTES", 1)
    selection_dir = tmp_path / "sel"
    selection_dir.mkdir()
    expected_names = [f"shard_{index:05d}.safetensors.zst" for index in range(25)]
    for name in reversed(expected_names):
        (selection_dir / name).write_bytes(b"x")
    (selection_dir / "nested").mkdir()
    (selection_dir / "nested/shard_00000.safetensors.zst").write_bytes(b"x")
    (selection_dir / "other_00000.safetensors.zst").write_bytes(b"x")

    selected = gate._selection_probe_files(selection_dir)
    assert [path.name for path in selected] == expected_names[: gate._SMALL_FILE_COUNT]
    assert all(path.parent == selection_dir for path in selected)


@pytest.mark.parametrize("shape", ["missing", "file"])
def test_selection_inventory_rejects_missing_or_non_directory(
    tmp_path: Path,
    shape: str,
) -> None:
    selection_dir = tmp_path / "selection"
    if shape == "file":
        selection_dir.write_bytes(b"not a directory")

    message = "not accessible" if shape == "missing" else "not a directory"
    with pytest.raises(gate._ProbeConfigError, match=message):
        gate._selection_probe_files(selection_dir)


def test_selection_inventory_requires_twenty_files(tmp_path: Path) -> None:
    selection_dir = tmp_path / "sel"
    selection_dir.mkdir()
    for index in range(gate._SMALL_FILE_COUNT - 1):
        (selection_dir / f"shard_{index:05d}.safetensors.zst").write_bytes(b"x")

    with pytest.raises(gate._ProbeConfigError, match="19.*20 are required"):
        gate._selection_probe_files(selection_dir)


@pytest.mark.parametrize("kind", ["directory", "short"])
def test_selection_inventory_rejects_unreadable_probe_shapes(
    kind: str,
    tmp_path: Path,
) -> None:
    selection_dir = tmp_path / "sel"
    selection_dir.mkdir()
    for index in range(gate._SMALL_FILE_COUNT):
        path = selection_dir / f"shard_{index:05d}.safetensors.zst"
        if index == 0 and kind == "directory":
            path.mkdir()
        else:
            size = gate._SMALL_READ_BYTES - 1 if index == 0 else gate._SMALL_READ_BYTES
            path.write_bytes(b"x" * size)

    match = "not a regular file" if kind == "directory" else "at least 65536 are required"
    with pytest.raises(gate._ProbeConfigError, match=match):
        gate._selection_probe_files(selection_dir)


@pytest.mark.parametrize("make_directory", [False, True])
def test_regular_file_validation_rejects_missing_or_directory(
    tmp_path: Path,
    make_directory: bool,
) -> None:
    path = tmp_path / "checkpoint"
    if make_directory:
        path.mkdir()
    message = "not a regular file" if make_directory else "not accessible"
    with pytest.raises(gate._ProbeConfigError, match=message):
        gate._validate_regular_file(path, label="checkpoint file")


def test_read_exact_detects_short_input() -> None:
    with pytest.raises(gate._ProbeConfigError, match="ended before 8 bytes"):
        gate._read_exact(io.BytesIO(b"short"), 8, path=Path("short.bin"))


def test_worker_measurement_reads_real_temporary_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_BULK_READ_BYTES", 4096)
    monkeypatch.setattr(gate, "_SMALL_READ_BYTES", 64)
    monkeypatch.setattr(gate, "_READ_CHUNK_BYTES", 127)
    monkeypatch.setattr(gate.random, "randrange", lambda slots: slots - 1)

    checkpoint = tmp_path / "model_000001.pt"
    checkpoint.write_bytes(bytes(range(256)) * 32)
    selection_dir = tmp_path / "sel"
    selection_dir.mkdir()
    for index in range(gate._SMALL_FILE_COUNT):
        (selection_dir / f"shard_{index:05d}.safetensors.zst").write_bytes(bytes([index]) * 128)

    payload = gate._measure_storage(checkpoint, selection_dir)
    assert payload["status"] == "measurement"
    assert payload["checkpoint_file"] == os.fspath(checkpoint)
    assert payload["bulk_offset"] == 0
    assert payload["bulk_bytes"] == 4096
    assert payload["bulk_elapsed_s"] > 0
    assert payload["bulk_mb_s"] > 0
    assert payload["small_files"] == gate._SMALL_FILE_COUNT
    assert payload["small_read_bytes"] == 64
    assert payload["small_loop_s"] > 0


def test_worker_cli_uses_the_full_production_probe_geometry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_000001.pt"
    with checkpoint.open("wb") as file_obj:
        file_obj.truncate(gate._BULK_READ_BYTES)
    selection_dir = tmp_path / "sel"
    selection_dir.mkdir()
    for index in range(gate._SMALL_FILE_COUNT):
        (selection_dir / f"shard_{index:05d}.safetensors.zst").write_bytes(
            b"x" * gate._SMALL_READ_BYTES
        )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.utils.storage_recovery_gate",
            "--worker",
            "--checkpoint-file",
            os.fspath(checkpoint),
            "--selection-dir",
            os.fspath(selection_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "measurement"
    assert payload["checkpoint_file"] == os.fspath(checkpoint)
    assert payload["bulk_bytes"] == 64 * 1024 * 1024
    assert payload["bulk_elapsed_s"] > 0
    assert payload["bulk_mb_s"] > 0
    assert payload["small_files"] == 20
    assert payload["small_read_bytes"] == 64 * 1024
    assert payload["small_loop_s"] > 0


def test_worker_measurement_rejects_small_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_BULK_READ_BYTES", 1024)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x" * 1023)
    selection_dir = tmp_path / "sel"
    selection_dir.mkdir()

    with pytest.raises(gate._ProbeConfigError, match="at least 1024 are required"):
        gate._measure_storage(checkpoint, selection_dir)


@pytest.mark.parametrize(
    ("outcome", "exit_code", "expected"),
    [
        (
            gate._ProbeConfigError("bad paths"),
            2,
            {
                "status": "fatal_error",
                "type": "_ProbeConfigError",
                "message": "bad paths",
            },
        ),
        (
            OSError("NFS read failed"),
            1,
            {"status": "probe_error", "type": "OSError", "message": "NFS read failed"},
        ),
        (
            RuntimeError("implementation bug"),
            2,
            {
                "status": "fatal_error",
                "type": "RuntimeError",
                "message": "implementation bug",
            },
        ),
        (
            {"status": "measurement", "bulk_mb_s": 321.0, "small_loop_s": 0.125},
            0,
            {"status": "measurement", "bulk_mb_s": 321.0, "small_loop_s": 0.125},
        ),
    ],
    ids=["configuration", "storage_runtime", "unexpected", "measurement"],
)
def test_worker_main_classifies_and_serializes_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outcome: BaseException | dict[str, object],
    exit_code: int,
    expected: dict[str, object],
) -> None:
    def measure(*args: object) -> dict[str, object]:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(gate, "_measure_storage", measure)
    assert gate._worker_main(tmp_path / "model", tmp_path / "sel") == exit_code
    assert json.loads(capsys.readouterr().out) == expected
