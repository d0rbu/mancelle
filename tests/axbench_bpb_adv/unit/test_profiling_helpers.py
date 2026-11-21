"""Unit tests for the AxBench-BPB-Adv profiling helpers (issue #501).

These tests exercise the small additions in
``scripts/eval/axbench_bpb_adv_eval.py`` — the ``@_profiled`` decorator, the
``_cpu_rss_kb`` helper, and ``_record_memory_counters`` — in isolation, with
and without an active ``AxBenchProfiler``. The "no profiler bound" path must
be a complete no-op so the production code is byte-identical to its
un-instrumented form.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pytest
import torch
import yaml

import scripts.eval.axbench_bpb_adv_eval as cli
from steering.eval.axbench_base.profiler import (
    AxBenchProfiler,
    get_current_profiler,
)

# `resource` is POSIX-only; mirror the production module's pattern so the
# test that calls `resource.getrusage` works on every platform that the
# repo runs on. (Windows would just import None and the test should skip.)
_resource: ModuleType | None
try:
    _resource = importlib.import_module("resource")
except ImportError:
    _resource = None


def test_cpu_rss_kb_returns_int_or_none() -> None:
    """_cpu_rss_kb returns a non-negative int (Linux KB / macOS-normalized KB)
    or None when `resource` is unavailable. On the supported POSIX platforms
    we run on (Linux + macOS), this should always be int."""
    rss = cli._cpu_rss_kb()
    assert rss is None or (isinstance(rss, int) and rss >= 0)


def test_profiled_decorator_is_noop_without_active_profiler() -> None:
    """The decorator must NOT alter return value or side-effect ordering when
    no profiler is bound. This is the byte-identity bar."""
    assert get_current_profiler() is None
    calls: list[str] = []

    @cli._profiled("test.no_profiler.region")
    def my_stage() -> int:
        calls.append("ran")
        return 42

    result = my_stage()
    assert result == 42
    assert calls == ["ran"]
    # Still no profiler bound after the call.
    assert get_current_profiler() is None


def test_profiled_decorator_skips_memory_counter_helpers_without_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict zero-overhead: when no profiler is bound, the decorator must NOT
    call ``_cpu_rss_kb`` or ``_record_memory_counters`` (both incur syscalls
    / Python-level work whose only consumer is the profiler). This is the
    "true byte-identity" bar — without this short-circuit the decorator adds
    ~5µs/stage of syscall work to the production path."""
    assert get_current_profiler() is None
    counter_calls: list[str] = []
    rss_calls: list[None] = []

    monkeypatch.setattr(
        cli,
        "_record_memory_counters",
        lambda stage: counter_calls.append(stage),
    )
    monkeypatch.setattr(cli, "_cpu_rss_kb", lambda: rss_calls.append(None))

    @cli._profiled("axbench_bpb_adv.zero_overhead_test")
    def my_stage() -> int:
        return 99

    result = my_stage()
    assert result == 99
    # Zero overhead: neither helper invoked when no profiler is bound.
    assert counter_calls == []
    assert rss_calls == []


def test_profiled_decorator_records_region_when_active(tmp_path: Path) -> None:
    """When a profiler is bound, the decorator emits one `region` event with
    the configured name, plus the per-stage memory counter events."""
    out_dir = tmp_path / "profile"
    calls: list[int] = []

    @cli._profiled("axbench_bpb_adv.unit_test_region")
    def my_stage(x: int) -> int:
        calls.append(x)
        return x * 2

    with AxBenchProfiler(out_dir, metadata={"test": "true"}):
        assert get_current_profiler() is not None
        result = my_stage(7)

    assert result == 14
    assert calls == [7]

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    region_names = [e["name"] for e in events if e["type"] == "region"]
    assert "axbench_bpb_adv.unit_test_region" in region_names

    # Memory counters emitted under the stage's attr suffix (the part after the
    # first dot). For "axbench_bpb_adv.unit_test_region" that's "unit_test_region".
    counter_names = {e["name"] for e in events if e["type"] == "counter"}
    assert "axbench_bpb_adv.unit_test_region.cpu_rss_kb" in counter_names or (
        cli._cpu_rss_kb() is None
    )


def test_profiled_decorator_preserves_signature_and_kwargs() -> None:
    """`functools.wraps` must preserve the wrapped function's metadata + the
    decorator must pass *args/**kwargs through faithfully."""

    @cli._profiled("test.signature.region")
    def my_stage(a: int, b: int, *, c: int = 0) -> int:
        return a + b + c

    assert my_stage.__name__ == "my_stage"
    assert my_stage(1, 2, c=3) == 6


def test_record_memory_counters_no_profiler_is_noop() -> None:
    """`_record_memory_counters` is a no-op when no profiler is bound — must
    not raise, must not write anything anywhere."""
    assert get_current_profiler() is None
    cli._record_memory_counters("some_stage")
    assert get_current_profiler() is None


def test_record_memory_counters_emits_counters_when_active(tmp_path: Path) -> None:
    """When a profiler is bound, `_record_memory_counters` emits the CPU RSS
    counter (always) and CUDA counters (only when CUDA is available)."""
    out_dir = tmp_path / "profile"

    with AxBenchProfiler(out_dir, metadata={}):
        cli._record_memory_counters("test_stage")

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    counter_names = {e["name"] for e in events if e["type"] == "counter"}

    # The CPU RSS counter is emitted when resource.getrusage is available,
    # which it is on the Linux test platform.
    if cli._cpu_rss_kb() is not None:
        assert "axbench_bpb_adv.test_stage.cpu_rss_kb" in counter_names

    # CUDA counters only when CUDA is available.
    if torch.cuda.is_available():
        assert "axbench_bpb_adv.test_stage.cuda_allocated_bytes" in counter_names
        assert "axbench_bpb_adv.test_stage.cuda_reserved_bytes" in counter_names


def test_profiled_decorator_propagates_exception_and_records_errored_region(
    tmp_path: Path,
) -> None:
    """If the wrapped stage raises, the exception must propagate *and* the
    profiler must record the region with `errored=True`. The summary should
    still be written by `AxBenchProfiler.__exit__` (status="error")."""
    out_dir = tmp_path / "profile"

    class StageError(RuntimeError):
        pass

    @cli._profiled("axbench_bpb_adv.failing_stage")
    def my_failing_stage() -> None:
        raise StageError("kaboom")

    # The exception must propagate through the AxBenchProfiler context-exit
    # so that its __exit__ sees exc_type and writes status="error". Catching
    # inside the with-block would make __exit__ see no exception → status="ok".
    raised = False
    try:
        with AxBenchProfiler(out_dir, metadata={}):
            my_failing_stage()
    except StageError:
        raised = True

    assert raised
    # The region event was still emitted by profile_region's finally block.
    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failing_regions = [
        e for e in events if e["type"] == "region"
        and e["name"] == "axbench_bpb_adv.failing_stage"
    ]
    assert len(failing_regions) == 1
    assert failing_regions[0]["errored"] is True

    # And the summary was written.
    summary = yaml.safe_load((out_dir / "summary.yaml").read_text(encoding="utf-8"))
    assert summary["status"] == "error"


def _collect_profiled_region_names(source: str) -> list[str]:
    """Parse the CLI module's source via ``ast`` to extract every
    ``@_profiled("...")`` region-name argument.

    Quote-agnostic: a regex like ``r'@_profiled\\("..."\\)'`` would miss
    single-quoted variants. ``ast`` walks the parse tree and reads the
    actual string constant, so single/double/triple-quoted forms all match.
    """
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if not isinstance(deco, ast.Call):
                continue
            func = deco.func
            if isinstance(func, ast.Name) and func.id == "_profiled" and deco.args:
                arg = deco.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.append(arg.value)
    return names


def test_profile_region_name_namespacing_unique_per_stage(tmp_path: Path) -> None:
    """All 8 stage decorator names must be unique + namespaced under
    `axbench_bpb_adv.`. Asserting against the production module catches a
    regression where someone copy-pastes a decorator and forgets to change
    its argument."""
    source = Path(cli.__file__).read_text(encoding="utf-8")
    matches = _collect_profiled_region_names(source)
    assert len(matches) == 8, (
        f"expected 8 @_profiled stage decorators in axbench_bpb_adv_eval.py, "
        f"got {len(matches)}: {matches}"
    )
    assert len(set(matches)) == 8, f"stage region names not unique: {matches}"
    expected = {
        "axbench_bpb_adv.validate_source",
        "axbench_bpb_adv.prepare_context_cache",
        "axbench_bpb_adv.prepare_attack_rows",
        "axbench_bpb_adv.build_candidates",
        "axbench_bpb_adv.score",
        "axbench_bpb_adv.score_general_val",
        "axbench_bpb_adv.aggregate",
        "axbench_bpb_adv.plot",
    }
    assert set(matches) == expected, (
        f"stage region names drift: expected {expected}, got {set(matches)}"
    )


# ---------------------------------------------------------------------------
# Helper branch coverage (GAP-2, GAP-3, GAP-4 from adversarial review of #503).
# These pin the rarely-traveled fallback branches in `_cpu_rss_kb`,
# `_record_memory_counters`, and `_profiled`. Without these, mutations like
# "make `_cpu_rss_kb` always return None" or "remove `except Exception` from
# `_record_memory_counters`" would pass the suite silently.
# ---------------------------------------------------------------------------


def test_cpu_rss_kb_returns_none_when_resource_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GAP-2: when the ``resource`` module is unavailable (e.g. Windows),
    ``_cpu_rss_kb`` short-circuits to None.

    The CLI module resolves ``resource`` once at load time and stores the
    handle (or None) in ``cli._resource`` — so we simulate the unavailable
    case by monkey-patching that handle to None.
    """
    monkeypatch.setattr(cli, "_resource", None)
    assert cli._cpu_rss_kb() is None


def test_sum_list_lengths_handles_null_cells() -> None:
    """Defensive contract: pyarrow list-typed columns are ``nullable=True`` by
    default (the BPB-Adv schemas don't override this), so any future writer
    that emits a null cell would crash ``_sum_list_lengths`` via ``len(None)``
    — and the crash would propagate out of the profiled stage region, breaking
    the "profiling must never break the run" invariant the helpers document.

    Treat null cells as length 0 (semantically equivalent to "no tokens here";
    no production writer emits nulls today but the helper should not rely on
    caller discipline that isn't schema-enforced).
    """
    tbl = pa.Table.from_pydict(
        {"tokens": [[1, 2, 3], None, [], [9, 9]]},
        schema=pa.schema([("tokens", pa.list_(pa.int64()))]),
    )
    # 3 + 0 (null) + 0 (empty) + 2 = 5
    assert cli._sum_list_lengths(tbl, "tokens") == 5


def test_sum_list_lengths_handles_empty_table() -> None:
    """Empty table → sum is 0. Avoids a regression that would special-case
    zero-row tables (e.g. by accidentally indexing column.chunks[0])."""
    tbl = pa.Table.from_pydict(
        {"tokens": []},
        schema=pa.schema([("tokens", pa.list_(pa.int64()))]),
    )
    assert cli._sum_list_lengths(tbl, "tokens") == 0


@pytest.mark.skipif(_resource is None, reason="POSIX-only")
def test_cpu_rss_kb_returns_none_when_getrusage_raises_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GAP-2: an OSError from getrusage (e.g. some sandboxed CI environments)
    must also degrade to None, not blow up."""
    assert _resource is not None  # narrowed by the skipif

    def boom(_who: int) -> object:
        raise OSError("simulated rusage failure")

    monkeypatch.setattr(_resource, "getrusage", boom)
    assert cli._cpu_rss_kb() is None


def test_record_memory_counters_swallows_torch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GAP-3: if `torch.cuda.is_available()` (or any CUDA call) raises, the
    helper must NOT propagate — profiling-broke-the-run is a bigger sin than
    a missing CUDA counter. CPU RSS still gets emitted."""

    def boom() -> bool:
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(torch.cuda, "is_available", boom)

    out_dir = tmp_path / "profile"
    with AxBenchProfiler(out_dir, metadata={}):
        cli._record_memory_counters("torch_fault_stage")  # MUST NOT RAISE

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    counter_names = {e["name"] for e in events if e["type"] == "counter"}
    # CPU RSS counter still emitted (it runs before the torch try/except).
    if cli._cpu_rss_kb() is not None:
        assert "axbench_bpb_adv.torch_fault_stage.cpu_rss_kb" in counter_names
    # No CUDA counters because the call raised.
    assert "axbench_bpb_adv.torch_fault_stage.cuda_allocated_bytes" not in counter_names
    assert "axbench_bpb_adv.torch_fault_stage.cuda_reserved_bytes" not in counter_names


def test_profiled_decorator_uses_region_as_stage_attr_when_no_dot(
    tmp_path: Path,
) -> None:
    """GAP-4: when `region_name` has no `.`, the decorator falls back to using
    the whole name as the stage attribute (vs the part-after-first-dot for
    namespaced names). This keeps the helper safe if a developer names a
    region without a namespace prefix."""
    out_dir = tmp_path / "profile"

    @cli._profiled("flat_no_namespace")
    def my_stage() -> int:
        return 1

    with AxBenchProfiler(out_dir, metadata={}):
        assert my_stage() == 1

    events = [
        json.loads(line)
        for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    counter_names = {e["name"] for e in events if e["type"] == "counter"}
    # No-dot fallback uses the whole region as the stage attr → counters
    # land under "axbench_bpb_adv.flat_no_namespace.cpu_rss_kb".
    if cli._cpu_rss_kb() is not None:
        assert "axbench_bpb_adv.flat_no_namespace.cpu_rss_kb" in counter_names
