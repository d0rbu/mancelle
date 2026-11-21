"""Robustness tests for the profile-mode summary outputs (PR-1).

These tests cover three correctness properties beyond the basic happy
path in ``test_profiler_first_candidate_breakdown.py``:

1. Unit consistency in ``summary.csv`` — the breakdown rows use
   ``elapsed_s`` (float seconds) so the CSV does not mix milliseconds
   and seconds in the same table.
2. Errored regions do not pollute ``first_duration_seconds`` — when a
   ``profile_region`` block exits via exception, the aggregate must NOT
   record a misleadingly small partial elapsed value.
3. The breakdown always emits all six canonical phase rows — even when
   some of the canonical regions never fired — so downstream tooling
   that expects a fixed schema does not silently miss columns.

It also covers the previously-untested counter CSV-row writing path
(``profiler.py:_write_reports`` counter loop).
"""

from __future__ import annotations

import csv
import time
from contextvars import Context, copy_context
from unittest.mock import MagicMock

import pytest
import yaml

from steering.eval.axbench_base import profiler as profiler_mod
from steering.eval.axbench_base.profiler import AxBenchProfiler, profile_region

# ---------------------------------------------------------------------------
# Unit consistency: breakdown rows use elapsed_s (float seconds).
# ---------------------------------------------------------------------------


def test_breakdown_rows_use_elapsed_s_not_elapsed_ms_in_csv(tmp_path):
    """Summary CSV breakdown rows expose ``elapsed_s`` (seconds, float)
    rather than ``elapsed_ms`` (milliseconds, int). The other CSV rows
    use the ``_s`` convention (``total_s``, ``mean_s``, ``min_s``,
    ``max_s``); mixing units in the same table is a footgun."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        with profile_region("stage0c.load_feature_extraction_pair"):
            time.sleep(0.05)

    summary_csv = profile_dir / "summary.csv"
    with summary_csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    assert fieldnames is not None
    assert "elapsed_s" in fieldnames, (
        f"breakdown column must be ``elapsed_s`` for unit consistency; "
        f"saw fieldnames={fieldnames}"
    )
    assert "elapsed_ms" not in fieldnames, (
        "``elapsed_ms`` mixes milliseconds with the rest of the CSV which "
        "is in seconds; rename to ``elapsed_s`` (float) instead"
    )

    breakdown = [
        r for r in rows
        if r.get("section") == "first_verified_candidate_breakdown"
    ]
    matched = [r for r in breakdown if r["phase"] == "model_load"]
    assert matched, f"missing model_load breakdown row in {breakdown!r}"
    elapsed = float(matched[0]["elapsed_s"])
    assert elapsed >= 0.04, (
        f"elapsed_s={elapsed} must reflect ~50ms sleep within tolerance; "
        f"if this is in milliseconds the value would be ~50.0"
    )
    assert elapsed < 5.0, (
        f"elapsed_s={elapsed} unreasonably large; if the field is still in "
        f"milliseconds the value would be ~50.0 (clearly >5)"
    )


def test_breakdown_rows_use_elapsed_s_in_yaml(tmp_path):
    """The summary YAML payload mirrors the CSV convention: breakdown
    rows expose ``elapsed_s`` (float)."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        with profile_region("stage0c.build_percentile_table"):
            time.sleep(0.02)

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    matched = [r for r in bd if r["phase"] == "percentile_table"]
    assert matched, f"missing percentile_table row in YAML breakdown: {bd!r}"
    row = matched[0]
    assert "elapsed_s" in row, (
        f"YAML breakdown row must expose ``elapsed_s``; got keys={list(row)}"
    )
    assert "elapsed_ms" not in row, (
        "YAML breakdown should not carry the legacy ``elapsed_ms`` key"
    )
    assert row["elapsed_s"] >= 0.01


# ---------------------------------------------------------------------------
# Errored region does not update first_duration_seconds.
# ---------------------------------------------------------------------------


def test_errored_region_does_not_record_first_duration(tmp_path):
    """When a ``profile_region`` block exits via an exception, the partial
    duration must NOT be recorded as the region's first duration. A
    user reading the breakdown should see no row (or an empty
    ``elapsed_s``) rather than a misleadingly small partial timing."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    with profiler:
        with pytest.raises(RuntimeError, match="planned"):
            with profile_region("stage0c.try_load_partial_checkpoint"):
                time.sleep(0.01)
                raise RuntimeError("planned")

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    breakdown_rows = payload["first_verified_candidate_breakdown"]
    matched = [r for r in breakdown_rows
               if r["phase"] == "checkpoint_validate"]
    assert matched, (
        "checkpoint_validate row must always be present (even when its "
        "region only fired with an error)"
    )
    elapsed = matched[0]["elapsed_s"]
    assert elapsed in (None, "", 0, 0.0), (
        f"expected null/empty elapsed_s for an errored region; got {elapsed!r}"
    )


def test_errored_region_followed_by_success_uses_success_duration(tmp_path):
    """If a region first errors and is then re-entered successfully, the
    first-duration aggregate should reflect the SUCCESSFUL run, not the
    abandoned partial timing of the errored attempt."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    with profiler:
        with pytest.raises(RuntimeError):
            with profile_region("stage0c.verify_concept"):
                time.sleep(0.005)
                raise RuntimeError("first attempt fails")
        with profile_region("stage0c.verify_concept"):
            time.sleep(0.05)

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    matched = [r for r in bd if r["phase"] == "first_verify_call"]
    assert matched
    elapsed = float(matched[0]["elapsed_s"])
    assert elapsed >= 0.04, (
        f"first-duration should reflect the ~50ms successful re-entry, "
        f"not the ~5ms errored attempt; got elapsed_s={elapsed}"
    )


def test_profile_region_errored_does_not_record_first_duration(
    tmp_path,
):
    """The canonical profile_region helper excludes errored attempts from the
    first-duration aggregate used by runtime stage breakdowns."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    with profiler:
        with pytest.raises(RuntimeError, match="planned-module-helper"):
            with profile_region("stage0c.examples_qc_filter_one"):
                time.sleep(0.01)
                raise RuntimeError("planned-module-helper")

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    matched = [r for r in bd if r["phase"] == "first_qc_call"]
    assert matched
    assert matched[0]["elapsed_s"] is None, (
        f"module-level profile_region errored region should NOT set "
        f"first_duration_seconds; got {matched[0]['elapsed_s']!r}"
    )


def test_profile_region_propagates_baseexception(tmp_path):
    """``BaseException`` (e.g. ``KeyboardInterrupt`` / ``SystemExit``)
    must mark the region as errored and re-raise — not silently swallow.
    The aggregate must NOT record first_duration_seconds for the
    interrupted region."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    with profiler:
        with pytest.raises(KeyboardInterrupt):
            with profile_region("stage0c.try_load_partial_checkpoint"):
                time.sleep(0.005)
                raise KeyboardInterrupt

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    matched = [r for r in bd if r["phase"] == "checkpoint_validate"]
    assert matched and matched[0]["elapsed_s"] is None, (
        f"interrupted region should not record first_duration_seconds; "
        f"got {matched[0]['elapsed_s'] if matched else 'no row'!r}"
    )


def test_module_level_profile_region_propagates_baseexception(tmp_path):
    """Same BaseException property for the module-level helper."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    with profiler:
        with pytest.raises(SystemExit):
            with profile_region("stage0c.candidate_index_load_shard", shard=0):
                time.sleep(0.005)
                raise SystemExit(1)

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    matched = [r for r in bd if r["phase"] == "first_shard_load"]
    assert matched and matched[0]["elapsed_s"] is None


def test_breakdown_emits_six_rows_when_zero_phases_fired(tmp_path):
    """A profiled session that records ZERO of the canonical phases
    still emits all 6 breakdown rows with ``elapsed_s=None``. Exercises
    the empty-aggregates branch of ``_build_first_candidate_breakdown``
    and confirms downstream readers can rely on a fixed schema even on
    an aborted-early run."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        with profile_region("unrelated.region.name"):
            pass

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    assert len(bd) == 6
    assert all(row["elapsed_s"] is None for row in bd), (
        f"all 6 rows must have elapsed_s=None when no canonical phase "
        f"fired; got {bd!r}"
    )

    with (profile_dir / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    breakdown = [
        r for r in rows
        if r.get("section") == "first_verified_candidate_breakdown"
    ]
    assert len(breakdown) == 6
    assert all(r["elapsed_s"] == "" for r in breakdown), (
        f"all 6 CSV rows must have empty elapsed_s when no canonical "
        f"phase fired; got {breakdown!r}"
    )


def test_errored_region_still_writes_jsonl_event_for_debugging(tmp_path):
    """An errored region MUST still emit a jsonl event so post-mortem
    debugging can see partial timings. Only the breakdown's
    first_duration_seconds is suppressed; the per-event ledger is
    unaffected."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    with profiler:
        with pytest.raises(RuntimeError, match="for-jsonl-test"):
            with profile_region("stage0c.build_percentile_table"):
                time.sleep(0.005)
                raise RuntimeError("for-jsonl-test")

    import json as _json
    events_jsonl = profile_dir / "events.jsonl"
    region_events = []
    for line in events_jsonl.read_text().splitlines():
        ev = _json.loads(line)
        if ev.get("type") == "region":
            region_events.append(ev)
    matched = [
        e for e in region_events
        if e["name"] == "stage0c.build_percentile_table"
    ]
    assert len(matched) == 1, f"expected one region event; got {matched!r}"
    assert matched[0]["errored"] is True
    assert matched[0]["duration_s"] >= 0.0


def test_profile_region_thread_safe_under_concurrent_emit(tmp_path):
    """Concurrent region recording must produce well-formed events without
    lost or torn aggregates. Loguru / the
    profiler's ``_lock`` already serialises writes; this test pins the
    contract."""
    import threading
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})

    barrier = threading.Barrier(4)

    def _worker(context: Context) -> None:
        barrier.wait()

        def _record() -> None:
            for _ in range(50):
                with profile_region("stage0c.build_percentile_table"):
                    pass

        context.run(_record)

    with profiler:
        threads = [
            threading.Thread(target=_worker, args=(copy_context(),))
            for _ in range(4)
        ]
        for tA in threads:
            tA.start()
        for tA in threads:
            tA.join()

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    region_row = next(
        r for r in payload["regions"]
        if r["name"] == "stage0c.build_percentile_table"
    )
    assert region_row["count"] == 200, (
        f"expected 4 threads × 50 = 200 region records; got "
        f"{region_row['count']}"
    )


# ---------------------------------------------------------------------------
# Always emit all 6 breakdown rows (even when some regions never fired).
# ---------------------------------------------------------------------------


_CANONICAL_PHASES = (
    "model_load",
    "percentile_table",
    "checkpoint_validate",
    "first_shard_load",
    "first_qc_call",
    "first_verify_call",
)


def test_breakdown_emits_all_six_phases_when_only_subset_fired(tmp_path):
    """Even when only 2 of the 6 canonical regions fired, the breakdown
    must contain rows for ALL six phases (with empty/null ``elapsed_s``
    on the missing ones) so downstream tooling can rely on a fixed
    schema."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        with profile_region("stage0c.build_percentile_table"):
            pass
        with profile_region(
            "stage0c.verify_concept",
            concept_id=0,
            feature_id=0,
        ):
            pass

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    bd = payload["first_verified_candidate_breakdown"]
    phases = [r["phase"] for r in bd]
    assert phases == list(_CANONICAL_PHASES), (
        f"breakdown must list all 6 canonical phases in canonical order; "
        f"got {phases}"
    )
    fired = {"percentile_table", "first_verify_call"}
    for row in bd:
        if row["phase"] in fired:
            assert isinstance(row["elapsed_s"], (int, float))
            assert row["elapsed_s"] >= 0
        else:
            assert row["elapsed_s"] in (None, "", 0, 0.0), (
                f"phase {row['phase']!r} did not fire — elapsed_s should "
                f"be null/empty; got {row['elapsed_s']!r}"
            )


def test_breakdown_csv_emits_all_six_phases_when_only_subset_fired(tmp_path):
    """Same property as the YAML test, but for the flat CSV consumers."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        with profile_region("stage0c.candidate_index_load_shard", shard=0):
            pass

    with (profile_dir / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    breakdown = [r for r in rows
                 if r.get("section") == "first_verified_candidate_breakdown"]
    assert len(breakdown) == 6, (
        f"breakdown must always emit all 6 rows; got {len(breakdown)}: "
        f"{breakdown}"
    )
    fired_phase = "first_shard_load"
    for row in breakdown:
        if row["phase"] == fired_phase:
            assert row["elapsed_s"] != "", (
                "fired region should have a non-empty elapsed_s"
            )
        else:
            assert row["elapsed_s"] == "", (
                f"phase {row['phase']!r} did not fire — CSV elapsed_s "
                f"should be empty string; got {row['elapsed_s']!r}"
            )


# ---------------------------------------------------------------------------
# Counter CSV-row writing (previously 0% covered).
# ---------------------------------------------------------------------------


def test_record_counter_writes_csv_row_with_aggregate_total(tmp_path):
    """Two ``record_counter`` calls with the same name produce one CSV
    row with ``kind=counter``, ``count=2``, ``total_value=<sum>``, and
    every other column empty (per the ``_blank_row()`` template)."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        profiler.record_counter("stage0c.candidate_segments_considered", 5.0)
        profiler.record_counter("stage0c.candidate_segments_considered", 2.0)

    with (profile_dir / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    counter_rows = [r for r in rows if r["kind"] == "counter"]
    assert len(counter_rows) == 1, (
        f"expected exactly one counter row, got {counter_rows!r}"
    )
    row = counter_rows[0]
    assert row["name"] == "stage0c.candidate_segments_considered"
    assert int(row["count"]) == 2
    assert float(row["total_value"]) == 7.0
    for empty_col in ("total_s", "mean_s", "min_s", "max_s",
                      "rows", "peak_cuda_allocated_bytes",
                      "peak_cuda_reserved_bytes",
                      "section", "phase", "region", "elapsed_s"):
        assert row[empty_col] == "", (
            f"counter row should leave {empty_col!r} blank; got {row[empty_col]!r}"
        )


def test_module_level_record_counter_dispatches_to_active_profiler(tmp_path):
    """The module-level ``record_counter`` (no-op when no profiler is
    active) must dispatch to the active profiler's instance method when
    one is in scope. Covers the ``_CURRENT_PROFILER.get()`` dispatch
    branch in ``record_counter``."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        profiler_mod.record_counter("stage0c.test_module_dispatch", 3.0)
        profiler_mod.record_counter("stage0c.test_module_dispatch", 4.0)

    payload = yaml.safe_load((profile_dir / "summary.yaml").read_text())
    counter = next(
        c for c in payload["counters"]
        if c["name"] == "stage0c.test_module_dispatch"
    )
    assert counter["count"] == 2
    assert counter["total"] == 7.0


def test_module_level_record_counter_is_noop_without_active_profiler():
    """Outside an active profiler context, ``record_counter`` is a
    silent no-op rather than an error."""
    profiler_mod.record_counter("stage0c.no_profiler", 1.0)


# ---------------------------------------------------------------------------
# synchronize_cuda=False asserts the sync helper is NOT called.
# ---------------------------------------------------------------------------


def test_synchronize_cuda_false_does_not_call_sync_helper(tmp_path, monkeypatch):
    """With ``synchronize_cuda=False``, ``profile_region`` should not invoke
    ``_synchronize_cuda_if_available``."""
    sync_mock = MagicMock(name="_synchronize_cuda_if_available")
    monkeypatch.setattr(profiler_mod, "_synchronize_cuda_if_available",
                        sync_mock)

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(
        output_dir=profile_dir, metadata={}, synchronize_cuda=False,
    )
    with profiler:
        with profile_region("test.no_sync"):
            pass

    assert sync_mock.call_count == 0, (
        f"_synchronize_cuda_if_available should NOT be called when "
        f"synchronize_cuda=False; got call_count={sync_mock.call_count}"
    )


def test_synchronize_cuda_true_calls_sync_helper_around_each_region(
    tmp_path, monkeypatch,
):
    """With ``synchronize_cuda=True`` (the default) the sync helper is
    invoked at region entry and exit."""
    sync_mock = MagicMock(name="_synchronize_cuda_if_available")
    monkeypatch.setattr(profiler_mod, "_synchronize_cuda_if_available",
                        sync_mock)

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    profiler = AxBenchProfiler(
        output_dir=profile_dir, metadata={}, synchronize_cuda=True,
    )
    with profiler:
        with profile_region("test.sync"):
            pass

    assert sync_mock.call_count == 2, (
        f"expected 1 region × 2 sync calls (entry + exit) = 2; "
        f"got call_count={sync_mock.call_count}"
    )
