"""Free-when-off guard for ``timed_region`` (PR #488 P1 production regression).

The 14 raw ``timed_region`` consolidation call sites enter ``with
record_function(name)`` on EVERY call. ``record_function`` is a C++ dispatch
(~4,500 ns measured) that the production default (no recorder, no profiler) was
paying on the hottest path (the live dataloader). The fix gates ``timed_region``
on ``recorder_active()`` — when nothing is listening, it yields immediately and
NEVER touches ``record_function`` or any timing machinery.

The marker MUST still be emitted under a profiler (so traces keep the region
names) and a recorder must still capture timing — those are the two "listening"
states.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from steering.profiling import recorder_active
from steering.profiling.markers import TimingRecorder, timed_region, use_recorder


@contextmanager
def _spy_record_function(monkeypatch):
    """Replace ``markers.record_function`` with a spy that counts enters."""
    enters: list[str] = []

    @contextmanager
    def fake(name):
        enters.append(name)
        yield

    monkeypatch.setattr("steering.profiling.markers.record_function", fake)
    yield enters


class TestRecordFunctionSkippedWhenOff:
    def test_not_entered_with_no_recorder_no_profiler(self, monkeypatch):
        # The production default: nothing listening ⇒ record_function never entered.
        with _spy_record_function(monkeypatch) as enters:
            for _ in range(5):
                with timed_region("consolidation/x", sync_cuda=False):
                    pass
        assert enters == [], (
            "record_function must NOT be entered when no recorder and no profiler "
            f"is active; was entered {len(enters)} time(s)"
        )

    def test_entered_when_recorder_active(self, monkeypatch):
        rec = TimingRecorder()
        with _spy_record_function(monkeypatch) as enters:
            with use_recorder(rec):
                with timed_region("consolidation/x", sync_cuda=False):
                    pass
        assert enters == ["consolidation/x"], (
            "record_function must be entered when a recorder is active"
        )
        # And the recorder still captured the row.
        assert [r["name"] for r in rec.rows()] == ["consolidation/x"]

    def test_entered_under_torch_profiler(self, monkeypatch):
        # Under a torch profiler (no recorder), the marker MUST still be emitted so
        # the region name survives in the trace.
        with _spy_record_function(monkeypatch) as enters:
            with torch.profiler.profile():
                with timed_region("consolidation/x", sync_cuda=False):
                    pass
        assert "consolidation/x" in enters, (
            "record_function must be emitted under torch.profiler.profile() even "
            "with no recorder, so trace markers are preserved"
        )


class TestRecorderActive:
    def test_false_at_baseline(self):
        assert recorder_active() is False

    def test_true_with_recorder(self):
        rec = TimingRecorder()
        with use_recorder(rec):
            assert recorder_active() is True
        assert recorder_active() is False

    def test_true_under_torch_profiler(self):
        assert recorder_active() is False
        with torch.profiler.profile():
            assert recorder_active() is True
        assert recorder_active() is False
