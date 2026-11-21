"""Low-overhead startup timing for long-running training entrypoints.

``StartupTrace`` deliberately records only wall-clock boundaries at call-site
seams.  It does not synchronize CUDA, touch distributed state, or inspect
artifacts.  The clock and emitter are injectable so lifecycle and schema tests
can use deterministic values without constructing a model or dataset.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class StartupPhaseRecord:
    """One completed startup phase."""

    name: str
    status: str
    elapsed_s: float
    cumulative_s: float


class StartupTrace:
    """Collect and emit a single invocation's startup timeline.

    The default sink is the repository logger.  A trace is intentionally
    opt-in: callers pass ``None`` when instrumentation is not wanted, and the
    training code retains its existing timing and control flow in that case.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        emit: Callable[[str], object] = logger.info,
        event_prefix: str = "encdec",
    ) -> None:
        if not event_prefix or not event_prefix.replace("_", "").isalnum():
            raise ValueError(
                "event_prefix must be a nonempty alphanumeric/underscore string; "
                f"got {event_prefix!r}"
            )
        self._clock = clock
        self._emit = emit
        self._event_prefix = event_prefix
        self._started_at = clock()
        self._records: list[StartupPhaseRecord] = []
        self._first_update_step: int | None = None
        self._time_to_first_step_s: float | None = None
        self._first_step_s: float | None = None
        self._steady_state_total_s = 0.0
        self._steady_state_steps = 0
        self._finished = False
        self._summary: dict[str, object] | None = None

    @property
    def records(self) -> tuple[StartupPhaseRecord, ...]:
        """Return an immutable snapshot of completed phase records."""
        return tuple(self._records)

    @property
    def first_update_step(self) -> int | None:
        """Return the invocation-local step at which the first update ended."""
        return self._first_update_step

    @property
    def summary(self) -> Mapping[str, object] | None:
        """Return the emitted summary, if :meth:`finish` has run."""
        return dict(self._summary) if self._summary is not None else None

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        fields: Mapping[str, object] | None = None,
    ) -> Iterator[None]:
        """Measure a phase and emit an ``encdec_startup_phase`` record.

        Exceptions are re-raised unchanged after an ``error`` phase record is
        emitted.  The record itself is best-effort and cannot mask the original
        training exception.
        """
        if self._finished:
            raise RuntimeError("cannot add a startup phase after finish()")
        if any(record.name == name for record in self._records):
            raise RuntimeError(f"startup phase already recorded: {name}")
        phase_started_at = self._clock()
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            phase_ended_at = self._clock()
            elapsed_s = phase_ended_at - phase_started_at
            cumulative_s = phase_ended_at - self._started_at
            record = StartupPhaseRecord(
                name=name,
                status=status,
                elapsed_s=elapsed_s,
                cumulative_s=cumulative_s,
            )
            self._records.append(record)
            self._emit_line(
                f"{self._event_prefix}_startup_phase",
                phase=name,
                status=status,
                elapsed_s=elapsed_s,
                cumulative_s=cumulative_s,
                fields=fields,
            )

    def skip_phase(
        self,
        name: str,
        *,
        fields: Mapping[str, object] | None = None,
    ) -> None:
        """Emit a zero-duration ``skipped`` phase without running any work."""
        if self._finished:
            raise RuntimeError("cannot add a skipped phase after finish()")
        if any(record.name == name for record in self._records):
            raise RuntimeError(f"startup phase already recorded: {name}")
        now = self._clock()
        record = StartupPhaseRecord(
            name=name,
            status="skipped",
            elapsed_s=0.0,
            cumulative_s=now - self._started_at,
        )
        self._records.append(record)
        self._emit_line(
            f"{self._event_prefix}_startup_phase",
            phase=name,
            status="skipped",
            elapsed_s=0.0,
            cumulative_s=record.cumulative_s,
            fields=fields,
        )

    def phase_elapsed(self, name: str) -> float | None:
        """Return the most recent elapsed time for ``name``."""
        for record in reversed(self._records):
            if record.name == name:
                return record.elapsed_s
        return None

    def event(self, name: str, **fields: object) -> None:
        """Emit a stable, non-timing startup event."""
        if self._finished:
            raise RuntimeError("cannot add a startup event after finish()")
        self._emit_line(name, fields=fields)

    def observe_step(
        self,
        *,
        update_step: int,
        step_s: float,
        compile_configured: bool,
    ) -> None:
        """Record the first completed update and invocation-local steady state.

        The first update is identified by invocation order, not by ``step``;
        this remains correct when resuming from a nonzero checkpoint.  The
        existing trainer convention defines steady state as numeric steps
        strictly greater than ten, excluding the first observed update even if
        it already has a large step number.
        """
        if self._finished:
            raise RuntimeError("cannot observe a startup step after finish()")
        if self._first_step_s is None:
            self._first_update_step = update_step
            self._first_step_s = float(step_s)
            self._time_to_first_step_s = self.elapsed()
            self.event(
                f"{self._event_prefix}_startup_first_step",
                status="complete",
                update_step=update_step,
                time_to_first_step_s=self._time_to_first_step_s,
                first_step_s=self._first_step_s,
                compile_configured=compile_configured,
            )
            return
        if update_step > 10:
            self._steady_state_total_s += float(step_s)
            self._steady_state_steps += 1

    def finish(self, *, status: str) -> Mapping[str, object]:
        """Emit one idempotent final startup summary and return its fields."""
        if self._summary is not None:
            return dict(self._summary)
        if status not in {"complete", "no_training_step", "error"}:
            raise ValueError(f"invalid startup summary status: {status!r}")
        elapsed_s = self.elapsed()
        steady_state_step_s = (
            self._steady_state_total_s / self._steady_state_steps
            if self._steady_state_steps
            else None
        )
        phase_seconds = {
            record.name: record.elapsed_s for record in self._records
        }
        summary: dict[str, object] = {
            "status": status,
            "elapsed_s": elapsed_s,
            "first_update_step": self._first_update_step,
            "time_to_first_step_s": self._time_to_first_step_s,
            "first_step_s": self._first_step_s,
            "steady_state_step_s": steady_state_step_s,
            "steady_state_steps": self._steady_state_steps,
            "warmup_cutoff_step_gt": 10,
            "phase_seconds_json": json.dumps(
                phase_seconds,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        self._summary = summary
        self._finished = True
        self._emit_line(f"{self._event_prefix}_startup_summary", fields=summary)
        return dict(summary)

    def elapsed(self) -> float:
        """Return wall-clock seconds since trace construction."""
        return self._clock() - self._started_at

    def _emit_line(
        self,
        prefix: str,
        *,
        fields: Mapping[str, object] | None = None,
        **required_fields: object,
    ) -> None:
        parts = [prefix]
        for key, value in required_fields.items():
            parts.append(f"{key}={_format_field(value)}")
        if fields:
            for key, value in fields.items():
                parts.append(f"{key}={_format_field(value)}")
        try:
            self._emit(" ".join(parts))
        except Exception:
            # Instrumentation must never alter training because a logging sink
            # is unavailable or malformed.
            return


def _format_field(value: object) -> str:
    """Format scalar log fields without spaces or unstable repr details."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace(" ", "_")
