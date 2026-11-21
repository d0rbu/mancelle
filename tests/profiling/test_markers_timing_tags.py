from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from steering.profiling import (
    TimingRecorder as ExportedTimingRecorder,
)
from steering.profiling import (
    current_timing_tags,
)
from steering.profiling import (
    recorder_active as exported_recorder_active,
)
from steering.profiling import (
    timed_region as exported_timed_region,
)
from steering.profiling import (
    timing_tags as exported_timing_tags,
)
from steering.profiling import (
    use_recorder as exported_use_recorder,
)
from steering.profiling.markers import (
    TimingRecorder,
    recorder_active,
    timed_region,
    timing_tags,
    use_recorder,
)


def test_timing_tags_inherit_to_nested_regions():
    recorder = TimingRecorder()
    with use_recorder(recorder), timing_tags(suite="core", eval_variant="conditioned"):
        with timed_region("outer", sync_cuda=False):
            with timed_region("inner", sync_cuda=False):
                pass

    rows = recorder.rows()
    assert [row["name"] for row in rows] == ["inner", "outer"]
    assert all(row["suite"] == "core" for row in rows)
    assert all(row["eval_variant"] == "conditioned" for row in rows)


def test_timing_tags_nest_and_restore():
    recorder = TimingRecorder()
    with use_recorder(recorder), timing_tags(suite="core", eval_variant="target_real"):
        with timed_region("before", sync_cuda=False):
            pass
        with timing_tags(eval_variant="conditioned", conditioned_config="safe"):
            with timed_region("inside", sync_cuda=False):
                pass
        with timed_region("after", sync_cuda=False):
            pass

    rows = recorder.rows()
    assert rows[0]["eval_variant"] == "target_real"
    assert rows[1]["eval_variant"] == "conditioned"
    assert rows[1]["conditioned_config"] == "safe"
    assert rows[2]["eval_variant"] == "target_real"
    assert "conditioned_config" not in rows[2]


def test_timed_region_extra_overrides_inherited_tags():
    recorder = TimingRecorder()
    with use_recorder(recorder), timing_tags(suite="core", seed=1234):
        extra = {"suite": "chat", "seed": 2234}
        with timed_region(
            "x",
            sync_cuda=False,
            extra=extra,
        ):
            extra["suite"] = "mutated"
            extra["seed"] = 9999

    row = recorder.rows()[0]
    assert row["suite"] == "chat"
    assert row["seed"] == 2234


def test_timed_region_records_and_restores_tags_on_exception():
    recorder = TimingRecorder()

    with pytest.raises(RuntimeError, match="boom"):
        with use_recorder(recorder), timing_tags(suite="core"):
            with timed_region("explodes", sync_cuda=False):
                raise RuntimeError("boom")

    rows = recorder.rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "explodes"
    assert rows[0]["suite"] == "core"
    assert current_timing_tags() == {}


def test_nested_recorders_restore_to_outer_recorder():
    outer = TimingRecorder()
    inner = TimingRecorder()

    with use_recorder(outer):
        with timed_region("outer_before", sync_cuda=False):
            pass
        with use_recorder(inner):
            with timed_region("inner", sync_cuda=False):
                pass
        with timed_region("outer_after", sync_cuda=False):
            pass

    assert [row["name"] for row in outer.rows()] == ["outer_before", "outer_after"]
    assert [row["name"] for row in inner.rows()] == ["inner"]


def test_recorder_activated_inside_entered_region_does_not_capture_outer():
    outer = TimingRecorder()
    inner = TimingRecorder()

    with use_recorder(outer):
        with timed_region("outer", sync_cuda=False):
            with use_recorder(inner):
                with timed_region("inner", sync_cuda=False):
                    pass

    assert [row["name"] for row in outer.rows()] == ["outer"]
    assert [row["name"] for row in inner.rows()] == ["inner"]


def test_cuda_sync_only_when_recorder_active_and_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr("steering.profiling.markers.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "steering.profiling.markers.torch.cuda.synchronize",
        lambda: calls.append("sync"),
    )

    with timed_region("no_recorder"):
        pass
    assert calls == []

    recorder = TimingRecorder()
    with use_recorder(recorder), timed_region("sync_default"):
        pass
    assert calls == ["sync", "sync"]

    calls.clear()
    with use_recorder(recorder), timed_region("sync_disabled", sync_cuda=False):
        pass
    assert calls == []


def test_record_function_skipped_without_recorder_or_profiler(monkeypatch):
    # FREE WHEN OFF: with neither a recorder nor a profiler attached, timed_region
    # must NOT enter record_function at all (the production hot-path pays nothing),
    # yet still yield + propagate exceptions cleanly.
    events = []

    @contextmanager
    def fake_record_function(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    monkeypatch.setattr("steering.profiling.markers.record_function", fake_record_function)

    with timed_region("plain"):
        pass

    with pytest.raises(ValueError, match="bad"):
        with timed_region("raises"):
            raise ValueError("bad")

    assert events == [], "record_function must be skipped when nothing is listening"


def test_record_function_enters_under_profiler_and_unwinds_on_exception(monkeypatch):
    # Under a torch profiler (no recorder) the marker IS emitted so the region name
    # survives in the trace, and it unwinds correctly on exception.
    events = []

    @contextmanager
    def fake_record_function(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    monkeypatch.setattr("steering.profiling.markers.record_function", fake_record_function)

    with torch.profiler.profile():
        with timed_region("plain"):
            pass

        with pytest.raises(ValueError, match="bad"):
            with timed_region("raises"):
                raise ValueError("bad")

    assert events == [
        ("enter", "plain"),
        ("exit", "plain"),
        ("enter", "raises"),
        ("exit", "raises"),
    ]


def test_three_level_precedence_on_one_colliding_key():
    # outer timing_tags -> inner timing_tags -> explicit timed_region(extra=) on
    # the SAME key: the explicit value wins over the inner, which wins over outer.
    recorder = TimingRecorder()
    with use_recorder(recorder), timing_tags(k="outer"):
        with timing_tags(k="inner"):
            with timed_region("x", sync_cuda=False, extra={"k": "explicit"}):
                pass
    assert recorder.rows()[0]["k"] == "explicit"


def test_current_timing_tags_returns_a_copy():
    # The dict from current_timing_tags() is a copy: mutating it must not leak into
    # the active tags.
    with timing_tags(a=1):
        snapshot = current_timing_tags()
        snapshot["a"] = 999
        snapshot["b"] = "leak"
        again = current_timing_tags()
        assert again == {"a": 1}


def test_use_recorder_resets_to_none_on_exception():
    # If an exception escapes the `with use_recorder(...)`, the active recorder is
    # reset back to None (no leak into a sibling scope).
    rec = TimingRecorder()
    assert recorder_active() is False
    with pytest.raises(RuntimeError, match="boom"):
        with use_recorder(rec):
            assert recorder_active() is True
            raise RuntimeError("boom")
    assert recorder_active() is False


def test_profiling_package_exports_marker_api():
    assert ExportedTimingRecorder is TimingRecorder
    assert exported_timed_region is timed_region
    assert exported_timing_tags is timing_tags
    assert exported_use_recorder is use_recorder
    assert exported_recorder_active is recorder_active
