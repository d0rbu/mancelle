"""Exact parser and integration coverage for Linux benchmark accounting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analysis import resource_sampler as rs


def _proc_stat_line(
    *,
    pid: int = 123,
    comm: str = "worker (stage) 1",
    state: str = "S",
    ppid: int = 9,
    minflt: int = 101,
    majflt: int = 303,
    utime: int = 404,
    stime: int = 505,
    starttime: int = 1212,
) -> str:
    # Fields 3..22. Values not asserted by the parser remain distinctive so
    # an accidental one-column shift makes the expected values fail loudly.
    tail = [
        state,
        str(ppid),
        "22",
        "33",
        "44",
        "55",
        "66",
        str(minflt),
        "202",
        str(majflt),
        "4040",
        str(utime),
        str(stime),
        "606",
        "707",
        "808",
        "909",
        "10",
        "11",
        str(starttime),
    ]
    return f"{pid} ({comm}) {' '.join(tail)}\n"


def _smaps_text(
    *,
    rss: int = 100,
    pss: int = 70,
    private_clean: int = 20,
    private_dirty: int = 30,
    swap: int = 11,
    swap_pss: int = 7,
) -> str:
    return (
        "00400000-00452000 r--p 00000000 00:00 0 /bin/test\n"
        f"Rss: {rss} kB\n"
        f"Pss: {pss} kB\n"
        "Pss_Dirty: 31 kB\n"
        "Shared_Clean: 12 kB\n"
        "Shared_Dirty: 13 kB\n"
        f"Private_Clean: {private_clean} kB\n"
        f"Private_Dirty: {private_dirty} kB\n"
        f"Swap: {swap} kB\n"
        f"SwapPss: {swap_pss} kB\n"
    )


def _status_text(*, rss: int = 95, swap: int = 8) -> str:
    return (
        "Name:\tworker\n"
        f"VmRSS:\t{rss} kB\n"
        f"VmSwap:\t{swap} kB\n"
        "Threads:\t3\n"
        "voluntary_ctxt_switches:\t44\n"
        "nonvoluntary_ctxt_switches:\t5\n"
    )


def _io_text(*, read_bytes: int = 4096, write_bytes: int = 8192) -> str:
    return (
        "rchar: 10\n"
        "wchar: 20\n"
        "syscr: 30\n"
        "syscw: 40\n"
        f"read_bytes: {read_bytes}\n"
        f"write_bytes: {write_bytes}\n"
        "cancelled_write_bytes: 0\n"
    )


def _write_fake_process(
    proc_root: Path,
    pid: int,
    *,
    ppid: int = 1,
    rss: int = 100,
    pss: int = 70,
    private_clean: int = 20,
    private_dirty: int = 30,
    swap_pss: int = 7,
    store_root: Path | None = None,
) -> None:
    root = proc_root / str(pid)
    root.mkdir(parents=True)
    (root / "stat").write_text(_proc_stat_line(pid=pid, ppid=ppid))
    (root / "smaps_rollup").write_text(
        _smaps_text(
            rss=rss,
            pss=pss,
            private_clean=private_clean,
            private_dirty=private_dirty,
            swap_pss=swap_pss,
        )
    )
    (root / "status").write_text(_status_text(rss=rss - 5))
    (root / "io").write_text(_io_text())
    mapped = (
        store_root / "stem" / "ids.bin" if store_root is not None else Path("/tmp/other.bin")
    )
    (root / "maps").write_text(
        "1000-3000 rw-s 00000000 00:01 42 " + str(mapped) + " (deleted)\n"
    )
    (root / "cgroup").write_text("0::/bench.slice/run.scope\n")


def _write_fake_host(proc_root: Path) -> None:
    (proc_root / "pressure").mkdir(parents=True)
    (proc_root / "meminfo").write_text(
        "MemTotal: 100000 kB\n"
        "MemAvailable: 60000 kB\n"
        "SwapTotal: 200000 kB\n"
        "SwapFree: 180000 kB\n"
        "Shmem: 1234 kB\n"
        "HugePages_Total: 0\n"
    )
    (proc_root / "vmstat").write_text(
        "pswpin 10\npswpout 20\npgmajfault 30\npgfault 400\n"
    )
    (proc_root / "stat").write_text("cpu  1 2 3 4 5 6 7 8 9 10\n")
    psi = "some avg10=0.10 avg60=0.20 avg300=0.30 total=100\nfull avg10=0.01 avg60=0.02 avg300=0.03 total=10\n"
    (proc_root / "pressure" / "memory").write_text(psi)
    (proc_root / "pressure" / "io").write_text(psi)


def _write_fake_cgroup(cgroup_root: Path) -> None:
    group = cgroup_root / "bench.slice" / "run.scope"
    group.mkdir(parents=True)
    (group / "memory.current").write_text("123456\n")
    (group / "memory.swap.current").write_text("654321\n")
    (group / "memory.max").write_text("max\n")
    (group / "memory.high").write_text("999999\n")
    (group / "memory.events").write_text("low 1\nhigh 2\nmax 3\noom 4\noom_kill 5\n")


def test_parse_kb_mapping_preserves_missing_fields() -> None:
    parsed = rs.parse_kb_mapping("A: 1 kB\nignored: 99 kB\n", ("A", "B"))
    assert parsed == {"A": 1, "B": None}


@pytest.mark.parametrize(
    "raw",
    ["A: 1 bytes\n", "A: one kB\n", "A: 1\n", "A: 1 kB trailing\n"],
)
def test_parse_kb_mapping_rejects_non_kernel_units_and_shapes(raw: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        rs.parse_kb_mapping(raw, ("A",))


def test_parse_smaps_rollup_computes_uss_from_private_components() -> None:
    parsed = rs.parse_smaps_rollup(_smaps_text())
    assert parsed == {
        "Rss_kB": 100,
        "Pss_kB": 70,
        "Pss_Dirty_kB": 31,
        "Shared_Clean_kB": 12,
        "Shared_Dirty_kB": 13,
        "Private_Clean_kB": 20,
        "Private_Dirty_kB": 30,
        "Swap_kB": 11,
        "SwapPss_kB": 7,
        "Uss_kB": 50,
    }


def test_parse_smaps_rollup_does_not_invent_uss_with_missing_private_field() -> None:
    parsed = rs.parse_smaps_rollup("Private_Clean: 3 kB\n")
    assert parsed["Private_Clean_kB"] == 3
    assert parsed["Private_Dirty_kB"] is None
    assert parsed["Uss_kB"] is None


def test_parse_proc_status_uses_rss_only_as_separate_diagnostic() -> None:
    parsed = rs.parse_proc_status(_status_text())
    assert parsed == {
        "VmRSS_kB": 95,
        "VmSwap_kB": 8,
        "Threads": 3,
        "voluntary_ctxt_switches": 44,
        "nonvoluntary_ctxt_switches": 5,
    }


def test_parse_proc_stat_handles_spaces_and_parentheses_in_comm() -> None:
    parsed = rs.parse_proc_stat(_proc_stat_line())
    assert parsed == {
        "pid": 123,
        "comm": "worker (stage) 1",
        "state": "S",
        "ppid": 9,
        "minflt": 101,
        "majflt": 303,
        "utime_ticks": 404,
        "stime_ticks": 505,
        "starttime_ticks": 1212,
    }


@pytest.mark.parametrize(
    "raw",
    ["", "123 no-parentheses", "123 (x)", "x (worker) S 1 2 3"],
)
def test_parse_proc_stat_rejects_malformed_or_truncated_lines(raw: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        rs.parse_proc_stat(raw)


def test_parse_proc_io_preserves_every_kernel_counter() -> None:
    assert rs.parse_proc_io(_io_text()) == {
        "rchar": 10,
        "wchar": 20,
        "syscr": 30,
        "syscw": 40,
        "read_bytes": 4096,
        "write_bytes": 8192,
        "cancelled_write_bytes": 0,
    }


def test_parse_proc_io_marks_missing_fields_none() -> None:
    parsed = rs.parse_proc_io("read_bytes: 7\n")
    assert parsed["read_bytes"] == 7
    assert parsed["write_bytes"] is None
    assert parsed["rchar"] is None


def test_parse_meminfo_distinguishes_kb_and_unitless_values() -> None:
    parsed = rs.parse_meminfo(
        "MemAvailable: 123 kB\nShmem: 45 kB\nHugePages_Total: 7\nBad: 4 pages\n"
    )
    assert parsed == {
        "MemAvailable_kB": 123,
        "Shmem_kB": 45,
        "HugePages_Total_kB": 7,
    }


def test_parse_vmstat_rejects_malformed_lines_instead_of_shifting_columns() -> None:
    assert rs.parse_vmstat("pswpin 1\npswpout 2\n") == {"pswpin": 1, "pswpout": 2}
    with pytest.raises(ValueError, match="malformed vmstat"):
        rs.parse_vmstat("pswpin 1 extra\n")


def test_parse_cpu_stat_accounts_all_present_ticks() -> None:
    parsed = rs.parse_proc_cpu_stat("cpu 1 2 3 4 5 6 7 8 9 10\ncpu0 1 1 1 1\n")
    assert parsed["user_ticks"] == 1
    assert parsed["iowait_ticks"] == 5
    assert parsed["steal_ticks"] == 8
    assert parsed["total_ticks"] == 55


@pytest.mark.parametrize("raw", ["", "intr 1 2\n", "cpu 1 2\n"])
def test_parse_cpu_stat_requires_complete_aggregate_line(raw: str) -> None:
    with pytest.raises((ValueError, IndexError)):
        rs.parse_proc_cpu_stat(raw)


def test_parse_psi_retains_averages_and_cumulative_totals() -> None:
    parsed = rs.parse_psi(
        "some avg10=1.00 avg60=2.00 avg300=3.00 total=123\n"
        "full avg10=0.10 avg60=0.20 avg300=0.30 total=45\n"
    )
    assert parsed == {
        "some_avg10": 1.0,
        "some_avg60": 2.0,
        "some_avg300": 3.0,
        "some_total": 123,
        "full_avg10": 0.1,
        "full_avg60": 0.2,
        "full_avg300": 0.3,
        "full_total": 45,
    }


@pytest.mark.parametrize("raw", ["unknown avg10=1\n", "some bad\n"])
def test_parse_psi_rejects_unknown_or_malformed_fields(raw: str) -> None:
    with pytest.raises(ValueError):
        rs.parse_psi(raw)


def test_parse_cgroup_path_selects_only_unified_v2_entry() -> None:
    raw = "7:cpu,cpuacct:/legacy\n0::/bench.slice/run.scope\n"
    assert rs.parse_cgroup_path(raw) == "/bench.slice/run.scope"
    assert rs.parse_cgroup_path("7:cpu:/legacy\n") is None


def test_parse_flat_int_mapping_rejects_non_pair_lines() -> None:
    assert rs.parse_flat_int_mapping("oom 2\noom_kill 1\n") == {"oom": 2, "oom_kill": 1}
    with pytest.raises(ValueError):
        rs.parse_flat_int_mapping("oom 1 extra\n")


def test_deleted_mapping_parser_filters_store_and_labels_virtual_bytes(tmp_path: Path) -> None:
    store = tmp_path / "store"
    inside = store / "stem a" / "ids.bin"
    text = (
        f"1000-3000 rw-s 00000000 00:01 1 {inside} (deleted)\n"
        "4000-5000 rw-s 00000000 00:01 2 /tmp/unrelated (deleted)\n"
        f"6000-9000 rw-s 00000000 00:01 3 {inside}\n"
    )
    parsed = rs.parse_deleted_mappings(text, store)
    assert parsed == {
        "deleted_mapping_count": 1,
        "deleted_mapped_virtual_bytes": 0x2000,
        "deleted_mapping_paths": [str(inside)],
    }


def test_deleted_mapping_parser_counts_duplicate_process_mappings_separately() -> None:
    text = (
        "1000-2000 r--s 00000000 00:01 1 /tmp/a (deleted)\n"
        "3000-5000 r--s 00000000 00:01 1 /tmp/a (deleted)\n"
    )
    parsed = rs.parse_deleted_mappings(text, None)
    assert parsed["deleted_mapping_count"] == 2
    assert parsed["deleted_mapped_virtual_bytes"] == 0x3000
    assert parsed["deleted_mapping_paths"] == ["/tmp/a"]


def test_visible_store_usage_counts_bytes_without_following_symlinks(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stem = store / "stem"
    stem.mkdir(parents=True)
    (stem / "a.bin").write_bytes(b"1234")
    (stem / "b.bin").write_bytes(b"567890")
    outside = tmp_path / "outside"
    outside.write_bytes(b"x" * 100)
    (store / "outside-link").symlink_to(outside)

    usage = rs.visible_store_usage(store)
    assert usage == {"exists": True, "visible_bytes": 10, "files": 2, "directories": 1}
    assert rs.visible_store_usage(tmp_path / "missing") == {
        "exists": False,
        "visible_bytes": 0,
        "files": 0,
        "directories": 0,
    }


def test_sample_process_reads_every_surface_and_computes_deleted_mapping(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    store = tmp_path / "store"
    store.mkdir()
    _write_fake_process(proc_root, 123, store_root=store)

    sample, errors = rs.sample_process(123, proc_root=proc_root, store_root=store)

    assert errors == []
    assert sample is not None
    assert sample["pid"] == 123
    assert sample["ppid"] == 1
    assert sample["smaps_rollup"]["Pss_kB"] == 70
    assert sample["smaps_rollup"]["Uss_kB"] == 50
    assert sample["status"]["Threads"] == 3
    assert sample["io"]["read_bytes"] == 4096
    assert sample["deleted_mappings"]["deleted_mapping_count"] == 1
    assert sample["deleted_mappings"]["deleted_mapped_virtual_bytes"] == 0x2000


def test_sample_process_returns_none_when_process_disappears(tmp_path: Path) -> None:
    sample, errors = rs.sample_process(999, proc_root=tmp_path / "proc")
    assert sample is None
    assert errors == []


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_sample_process_records_terminal_identity_without_live_surface_probes(
    tmp_path: Path,
    state: str,
) -> None:
    """A reaped-child race must not turn the final sample incomplete.

    The real issue #758 benchmark captured `/usr/bin/time` in state ``Z`` and
    Linux returned EACCES for its `/proc/PID/io`.  Malformed sentinel content
    below proves none of the live-process parsers are invoked for a terminal
    process; this is stronger than merely omitting the files.
    """
    proc_root = tmp_path / "proc"
    root = proc_root / "123"
    root.mkdir(parents=True)
    (root / "stat").write_text(_proc_stat_line(state=state))
    for name in ("smaps_rollup", "status", "io", "maps"):
        (root / name).write_text("must not be parsed\n")

    sample, errors = rs.sample_process(123, proc_root=proc_root)

    assert errors == []
    assert sample is not None
    assert sample["pid"] == 123
    assert sample["state"] == state
    assert sample["smaps_rollup"] is None
    assert sample["status"] is None
    assert sample["io"] is None
    assert sample["deleted_mappings"] is None


def test_sample_process_records_optional_surface_parse_failure(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_fake_process(proc_root, 123)
    (proc_root / "123" / "smaps_rollup").write_text("Pss: bad kB\n")

    sample, errors = rs.sample_process(123, proc_root=proc_root)

    assert sample is not None
    assert sample["smaps_rollup"] is None
    assert len(errors) == 1
    assert "smaps_rollup parse" in errors[0]


def test_sample_resources_aggregates_pss_uss_and_swap_pss_not_status_rss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    store = tmp_path / "store"
    store.mkdir()
    (store / "visible.bin").write_bytes(b"123")
    _write_fake_process(
        proc_root,
        100,
        pss=70,
        private_clean=20,
        private_dirty=30,
        swap_pss=7,
        store_root=store,
    )
    _write_fake_process(
        proc_root,
        101,
        ppid=100,
        pss=80,
        private_clean=25,
        private_dirty=35,
        swap_pss=9,
        store_root=store,
    )
    _write_fake_host(proc_root)
    _write_fake_cgroup(cgroup_root)
    monkeypatch.setattr(rs, "process_tree_pids", lambda root_pid: [root_pid, 101])

    sample = rs.sample_resources(
        100,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
        store_root=store,
        monotonic_ns=111,
        wall_time_ns=222,
    )

    assert sample["schema_version"] == 1
    assert sample["monotonic_ns"] == 111
    assert sample["wall_time_ns"] == 222
    assert sample["aggregate"] == {
        "process_count": 2,
        "Rss_kB_sum": 200,
        "Pss_kB_sum": 150,
        "Uss_kB_sum": 110,
        "Swap_kB_sum": 22,
        "SwapPss_kB_sum": 16,
        "deleted_mapping_count_sum": 2,
    }
    assert sample["host"]["meminfo"]["MemAvailable_kB"] == 60000
    assert sample["host"]["vmstat"]["pswpout"] == 20
    assert sample["host"]["cpu"]["iowait_ticks"] == 5
    assert sample["host"]["memory_psi"]["full_total"] == 10
    assert sample["cgroup"] == {
        "path": "/bench.slice/run.scope",
        "memory_current": 123456,
        "memory_swap_current": 654321,
        "memory_max": "max",
        "memory_high": 999999,
        "memory_events": {"low": 1, "high": 2, "max": 3, "oom": 4, "oom_kill": 5},
    }
    assert sample["store"] is not None
    assert sample["store"]["visible_bytes"] == 3
    assert sample["store"]["filesystem_total_bytes"] > 0
    assert sample["errors"] == []


def test_sample_resources_records_missing_optional_host_and_cgroup_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_fake_process(proc_root, 100)
    monkeypatch.setattr(rs, "process_tree_pids", lambda root_pid: [root_pid])

    sample = rs.sample_resources(
        100,
        proc_root=proc_root,
        cgroup_root=tmp_path / "missing-cgroup",
        monotonic_ns=1,
        wall_time_ns=2,
    )

    assert sample["aggregate"]["Pss_kB_sum"] == 70
    assert sample["host"] == {
        "meminfo": None,
        "vmstat": None,
        "cpu": None,
        "memory_psi": None,
        "io_psi": None,
    }
    assert sample["cgroup"] is None
    # Missing files are represented as unavailable AND recorded. A benchmark
    # missing these surfaces must not later be described as complete.
    assert len(sample["errors"]) == 6
    assert all("unavailable" in error for error in sample["errors"])


def test_sample_resources_records_malformed_cgroup_values_without_stopping_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    _write_fake_process(proc_root, 100)
    _write_fake_host(proc_root)
    _write_fake_cgroup(cgroup_root)
    group = cgroup_root / "bench.slice" / "run.scope"
    (group / "memory.current").write_text("not-an-integer\n")
    (group / "memory.events").write_text("oom not-an-integer\n")
    monkeypatch.setattr(rs, "process_tree_pids", lambda root_pid: [root_pid])

    sample = rs.sample_resources(
        100,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
        monotonic_ns=1,
        wall_time_ns=2,
    )

    assert sample["cgroup"]["memory_current"] is None
    assert sample["cgroup"]["memory_events"] is None
    assert any("memory.current parse" in error for error in sample["errors"])
    assert any("memory.events parse" in error for error in sample["errors"])


def _summary_sample(
    *,
    monotonic_ns: int,
    pss: int,
    uss: int,
    rss: int,
    swap_pss: int,
    available: int,
    shmem: int,
    pswpin: int,
    pswpout: int,
    major: int,
    iowait: int,
    visible: int,
    free: int,
    errors: list[str] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "monotonic_ns": monotonic_ns,
        "aggregate": {
            "process_count": 3,
            "Rss_kB_sum": rss,
            "Pss_kB_sum": pss,
            "Uss_kB_sum": uss,
            "SwapPss_kB_sum": swap_pss,
            "deleted_mapping_count_sum": 2,
        },
        "host": {
            "meminfo": {"MemAvailable_kB": available, "Shmem_kB": shmem},
            "vmstat": {"pswpin": pswpin, "pswpout": pswpout, "pgmajfault": major},
            "cpu": {"iowait_ticks": iowait},
        },
        "store": {"visible_bytes": visible, "filesystem_free_bytes": free},
        "errors": errors or [],
    }


def test_summarize_resource_samples_uses_peaks_minima_and_counter_deltas() -> None:
    first = _summary_sample(
        monotonic_ns=1_000_000_000,
        pss=100,
        uss=80,
        rss=150,
        swap_pss=1,
        available=1000,
        shmem=10,
        pswpin=5,
        pswpout=7,
        major=9,
        iowait=11,
        visible=20,
        free=500,
    )
    second = _summary_sample(
        monotonic_ns=3_500_000_000,
        pss=120,
        uss=70,
        rss=140,
        swap_pss=6,
        available=700,
        shmem=30,
        pswpin=8,
        pswpout=12,
        major=15,
        iowait=21,
        visible=50,
        free=400,
    )

    summary = rs.summarize_resource_samples([first, second])

    assert summary == {
        "schema_version": 1,
        "sample_count": 2,
        "duration_seconds": 2.5,
        "peak_process_count": 3,
        "peak_rss_kB_sum": 150,
        "peak_pss_kB_sum": 120,
        "peak_uss_kB_sum": 80,
        "peak_swap_pss_kB_sum": 6,
        "peak_deleted_mapping_count_sum": 2,
        "minimum_mem_available_kB": 700,
        "peak_shmem_kB": 30,
        "pswpin_delta": 3,
        "pswpout_delta": 5,
        "major_fault_delta": 6,
        "cpu_iowait_ticks_delta": 10,
        "peak_store_visible_bytes": 50,
        "minimum_store_filesystem_free_bytes": 400,
        "errors": [],
        "complete": True,
    }


def test_summarize_resource_samples_deduplicates_errors_and_marks_incomplete() -> None:
    sample = _summary_sample(
        monotonic_ns=1,
        pss=1,
        uss=1,
        rss=1,
        swap_pss=0,
        available=1,
        shmem=1,
        pswpin=1,
        pswpout=1,
        major=1,
        iowait=1,
        visible=1,
        free=1,
        errors=["b", "a", "b"],
    )
    summary = rs.summarize_resource_samples([sample])
    assert summary["errors"] == ["a", "b"]
    assert summary["complete"] is False
    assert summary["pswpout_delta"] is None


def test_summarize_resource_samples_rejects_empty_or_wrong_schema() -> None:
    with pytest.raises(ValueError, match="empty"):
        rs.summarize_resource_samples([])
    with pytest.raises(ValueError, match="schema_version"):
        rs.summarize_resource_samples([{"schema_version": 99}])


def test_load_resource_samples_skips_blank_lines_and_requires_objects(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text('\n{"schema_version":1,"x":2}\n\n{"schema_version":1,"x":3}\n')
    assert rs.load_resource_samples(path) == [
        {"schema_version": 1, "x": 2},
        {"schema_version": 1, "x": 3},
    ]
    path.write_text("[]\n")
    with pytest.raises(ValueError, match="expected a JSON object"):
        rs.load_resource_samples(path)


def test_load_resource_samples_reports_file_and_line_for_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{}\n{"bad"\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        rs.load_resource_samples(path)


@pytest.mark.skipif(not Path("/proc/self/smaps_rollup").is_file(), reason="Linux /proc required")
def test_sample_current_process_reports_consistent_identity_and_memory() -> None:
    sample, errors = rs.sample_process(os.getpid())
    assert errors == []
    assert sample is not None
    assert sample["pid"] == os.getpid()
    assert sample["ppid"] == os.getppid()
    assert sample["smaps_rollup"]["Rss_kB"] > 0
    assert sample["smaps_rollup"]["Pss_kB"] > 0
    assert sample["smaps_rollup"]["Uss_kB"] >= 0
    assert sample["smaps_rollup"]["Pss_kB"] <= sample["smaps_rollup"]["Rss_kB"]


@pytest.mark.skipif(not Path("/proc/self/smaps_rollup").is_file(), reason="Linux /proc required")
def test_sample_process_until_exit_writes_parseable_raw_jsonl(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.06)"],
        start_new_session=True,
    )
    path = tmp_path / "live.jsonl"
    with path.open("w") as output:
        returncode = rs.sample_process_until_exit(
            process,
            output,
            interval_seconds=0.01,
        )

    samples = rs.load_resource_samples(path)
    assert returncode == 0
    assert len(samples) >= 2
    assert all(sample["root_pid"] == process.pid for sample in samples)
    assert all(sample["schema_version"] == 1 for sample in samples)
    summary = rs.summarize_resource_samples(samples)
    assert summary["sample_count"] == len(samples)
    assert summary["errors"] == []
    assert summary["complete"] is True


def test_sample_process_until_exit_rejects_nonpositive_interval(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        with (tmp_path / "unused.jsonl").open("w") as output:
            with pytest.raises(ValueError, match="interval_seconds"):
                rs.sample_process_until_exit(process, output, interval_seconds=0)
    finally:
        process.wait()


def test_raw_jsonl_roundtrip_preserves_integer_precision(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    value = 2**63 - 1
    sample = {"schema_version": 1, "monotonic_ns": value, "errors": []}
    path.write_text(json.dumps(sample, separators=(",", ":")) + "\n")
    loaded = rs.load_resource_samples(path)
    assert loaded == [sample]
    assert isinstance(loaded[0]["monotonic_ns"], int)
