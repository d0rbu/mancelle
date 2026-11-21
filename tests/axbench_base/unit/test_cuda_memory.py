"""Tests for AxBench CUDA memory cleanup helpers."""

from __future__ import annotations

import gc

from steering.eval.axbench_base import cuda_memory


def test_release_cuda_iteration_memory_empties_cache_when_available(
    monkeypatch,
):
    calls: list[str] = []

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("is_available")
            return True

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

    monkeypatch.setattr(cuda_memory.t, "cuda", _FakeCuda)

    cuda_memory.release_cuda_iteration_memory()

    assert calls == ["is_available", "empty_cache"]


def test_release_cuda_iteration_memory_skips_empty_cache_without_cuda(
    monkeypatch,
):
    calls: list[str] = []

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("is_available")
            return False

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

    monkeypatch.setattr(cuda_memory.t, "cuda", _FakeCuda)

    cuda_memory.release_cuda_iteration_memory()

    assert calls == ["is_available"]


def test_release_cuda_iteration_memory_does_not_collect_garbage(monkeypatch):
    """The regression this change exists to prevent.

    ``gc.collect()`` here measured 65% of Stage 1b's serialized result-handler
    thread (581s of 894s, ``py-spy``): a full generational pass walks a heap
    holding a 4B-parameter model, and it ran once per QC batch on the
    single-threaded handler, so it sat directly on the throughput ceiling.
    Reinstating it at this cadence fails this test.
    """
    collected: list[int] = []

    def _tracking_collect(*args: object, **kwargs: object) -> int:
        collected.append(1)
        return 0

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            return None

    monkeypatch.setattr(gc, "collect", _tracking_collect)
    monkeypatch.setattr(cuda_memory.t, "cuda", _FakeCuda)

    cuda_memory.release_cuda_iteration_memory()

    assert collected == [], (
        "release_cuda_iteration_memory must not run a full gc.collect(); it is "
        "called per QC batch on the serialized handler thread"
    )


def test_module_no_longer_imports_gc():
    """Guard the import as well, so a partial revert is obvious in review."""
    assert not hasattr(cuda_memory, "gc"), (
        "cuda_memory should no longer need the gc module"
    )
