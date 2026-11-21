"""Deterministic lifecycle tests for encoder-decoder startup instrumentation."""

from __future__ import annotations

import json

import pytest

from steering.profiling import StartupPhaseRecord, StartupTrace


class FakeClock:
    """A monotonic test clock whose advances are explicit and auditable."""

    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        assert seconds >= 0
        self.value += seconds


def test_phase_records_exact_elapsed_and_cumulative_wall_time() -> None:
    clock = FakeClock()
    lines: list[str] = []
    trace = StartupTrace(clock=clock, emit=lines.append)

    clock.advance(2.0)
    with trace.phase("hydra_config"):
        clock.advance(3.25)
    clock.advance(1.0)
    trace.skip_phase("compile_wrapper", fields={"configured": False})

    assert trace.records == (
        StartupPhaseRecord(
            name="hydra_config",
            status="ok",
            elapsed_s=3.25,
            cumulative_s=5.25,
        ),
        StartupPhaseRecord(
            name="compile_wrapper",
            status="skipped",
            elapsed_s=0.0,
            cumulative_s=6.25,
        ),
    )
    assert trace.phase_elapsed("hydra_config") == 3.25
    assert trace.phase_elapsed("missing") is None
    assert lines == [
        "encdec_startup_phase phase=hydra_config status=ok "
        "elapsed_s=3.250000 cumulative_s=5.250000",
        "encdec_startup_phase phase=compile_wrapper status=skipped "
        "elapsed_s=0.000000 cumulative_s=6.250000 configured=no",
    ]


def test_exception_is_recorded_as_error_and_original_exception_survives() -> None:
    clock = FakeClock()
    lines: list[str] = []
    trace = StartupTrace(clock=clock, emit=lines.append)

    with pytest.raises(ValueError, match="bad artifact"):
        with trace.phase("packed_selection_artifacts"):
            clock.advance(4.0)
            raise ValueError("bad artifact")

    assert trace.records[-1].status == "error"
    assert trace.records[-1].elapsed_s == 4.0
    assert lines[0].startswith(
        "encdec_startup_phase phase=packed_selection_artifacts status=error"
    )


def test_nested_phase_elapsed_is_independent_and_records_are_append_only() -> None:
    clock = FakeClock()
    trace = StartupTrace(clock=clock, emit=lambda _line: None)

    with trace.phase("outer"):
        clock.advance(1.0)
        with trace.phase("inner"):
            clock.advance(2.0)
        clock.advance(3.0)

    assert [record.name for record in trace.records] == ["inner", "outer"]
    assert trace.phase_elapsed("inner") == 2.0
    assert trace.phase_elapsed("outer") == 6.0


def test_first_step_is_invocation_local_and_steady_state_uses_existing_cutoff() -> None:
    clock = FakeClock()
    lines: list[str] = []
    trace = StartupTrace(clock=clock, emit=lines.append)

    clock.advance(7.0)
    trace.observe_step(update_step=400, step_s=12.0, compile_configured=True)
    trace.observe_step(update_step=401, step_s=8.0, compile_configured=True)
    trace.observe_step(update_step=10, step_s=99.0, compile_configured=True)
    trace.observe_step(update_step=11, step_s=10.0, compile_configured=True)
    trace.observe_step(update_step=12, step_s=14.0, compile_configured=True)

    assert trace.first_update_step == 400
    first_line = lines[0]
    assert "update_step=400" in first_line
    assert "time_to_first_step_s=7.000000" in first_line
    assert "first_step_s=12.000000" in first_line
    assert "compile_configured=yes" in first_line

    summary = trace.finish(status="complete")
    assert summary["first_update_step"] == 400
    assert summary["time_to_first_step_s"] == 7.0
    assert summary["first_step_s"] == 12.0
    assert summary["steady_state_steps"] == 3
    assert summary["steady_state_step_s"] == (8.0 + 10.0 + 14.0) / 3
    assert summary["warmup_cutoff_step_gt"] == 10


def test_no_training_step_summary_has_explicit_unavailable_metrics() -> None:
    clock = FakeClock()
    lines: list[str] = []
    trace = StartupTrace(clock=clock, emit=lines.append)
    with trace.phase("first_batch_fetch"):
        clock.advance(2.0)

    summary = trace.finish(status="no_training_step")
    assert summary["status"] == "no_training_step"
    assert summary["first_update_step"] is None
    assert summary["time_to_first_step_s"] is None
    assert summary["first_step_s"] is None
    assert summary["steady_state_step_s"] is None
    assert summary["steady_state_steps"] == 0
    phase_seconds = json.loads(str(summary["phase_seconds_json"]))
    assert phase_seconds == {"first_batch_fetch": 2.0}
    assert lines[-1].startswith("encdec_startup_summary status=no_training_step")


def test_scalar_field_formatting_is_stable_for_machine_parsing() -> None:
    lines: list[str] = []
    trace = StartupTrace(clock=FakeClock(), emit=lines.append)
    trace.event(
        "encdec_startup_data_path",
        integer=3,
        boolean=True,
        false_boolean=False,
        absent=None,
        elapsed=1.25,
        label="two words",
    )

    assert lines == [
        "encdec_startup_data_path integer=3 boolean=yes false_boolean=no "
        "absent=none elapsed=1.250000 label=two_words"
    ]


def test_error_summary_after_first_step_preserves_cold_and_steady_metrics() -> None:
    clock = FakeClock()
    trace = StartupTrace(clock=clock, emit=lambda _line: None)
    trace.observe_step(update_step=20, step_s=5.0, compile_configured=False)
    trace.observe_step(update_step=21, step_s=7.0, compile_configured=False)

    summary = trace.finish(status="error")
    assert summary["status"] == "error"
    assert summary["first_update_step"] == 20
    assert summary["first_step_s"] == 5.0
    assert summary["steady_state_steps"] == 1
    assert summary["steady_state_step_s"] == 7.0


def test_phase_fields_and_duplicate_names_fail_closed() -> None:
    clock = FakeClock()
    lines: list[str] = []
    trace = StartupTrace(clock=clock, emit=lines.append)
    with trace.phase("repeat", fields={"source": "local"}):
        clock.advance(1.0)
    assert lines[0].endswith("source=local")
    with pytest.raises(RuntimeError, match="already recorded"):
        with trace.phase("repeat"):
            pass
    assert trace.phase_elapsed("repeat") == 1.0
    assert len(trace.records) == 1


def test_finish_is_idempotent_and_does_not_emit_duplicate_summary() -> None:
    lines: list[str] = []
    trace = StartupTrace(clock=FakeClock(), emit=lines.append)

    first = trace.finish(status="error")
    second = trace.finish(status="complete")

    assert second == first
    assert [line for line in lines if line.startswith("encdec_startup_summary ")] == [
        lines[-1]
    ]
    assert trace.summary == first


def test_invalid_lifecycle_operations_fail_closed() -> None:
    trace = StartupTrace(clock=FakeClock(), emit=lambda _line: None)
    with pytest.raises(ValueError, match="invalid startup summary status"):
        trace.finish(status="running")

    trace.finish(status="error")
    with pytest.raises(RuntimeError, match="after finish"):
        with trace.phase("late_phase"):
            pass
    with pytest.raises(RuntimeError, match="after finish"):
        trace.skip_phase("late")
    with pytest.raises(RuntimeError, match="after finish"):
        trace.event("late_event")
    with pytest.raises(RuntimeError, match="after finish"):
        trace.observe_step(update_step=0, step_s=1.0, compile_configured=False)


def test_logging_sink_failure_cannot_mask_training_or_summary() -> None:
    def broken_sink(_line: str) -> None:
        raise RuntimeError("sink unavailable")

    clock = FakeClock()
    trace = StartupTrace(clock=clock, emit=broken_sink)
    with trace.phase("tokenizer"):
        clock.advance(1.0)
    trace.observe_step(update_step=0, step_s=2.0, compile_configured=False)
    summary = trace.finish(status="complete")

    assert summary["first_step_s"] == 2.0
    assert trace.records[0].status == "ok"
