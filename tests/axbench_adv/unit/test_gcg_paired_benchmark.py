"""Check paired timing reports and detection of overwritten numerical results."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch as t

from scripts.benchmark import gcg_paired


def test_paired_measurement_reports_known_durations_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    active = [False]
    calls = []
    monkeypatch.setattr(gcg_paired.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(t.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(t.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(t.cuda, "max_memory_allocated", lambda: 10 if active[0] else 8)

    @contextmanager
    def reference() -> Iterator[None]:
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    def operation() -> t.Tensor:
        calls.append("reference" if active[0] else "candidate")
        clock[0] += 2.0 if active[0] else 1.0
        return t.tensor([3.0])

    value, report = gcg_paired.measure_paired(
        operation,
        3,
        reference=reference,
        numerics=lambda value: value,
        warmups=1,
    )
    assert calls == [
        "reference",
        "candidate",
        "reference",
        "candidate",
        "candidate",
        "reference",
        "reference",
        "candidate",
    ]
    assert not active[0]
    assert t.equal(value, t.tensor([3.0]))
    assert report["seconds"] == [1.0] * 3
    assert report["reference_seconds"] == [2.0] * 3
    assert report["paired_speedups"] == [2.0] * 3
    assert report["median_paired_speedup"] == 2.0
    assert report["faster_pairs"] == report["exact_pairs"] == 3
    assert report["peak_allocated"] == 8
    assert report["reference_peak_allocated"] == 10


def test_paired_measurement_catches_mutation_of_a_shared_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = t.tensor([0.0])
    active = [False]
    monkeypatch.setattr(t.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(t.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(t.cuda, "max_memory_allocated", lambda: 0)

    @contextmanager
    def reference() -> Iterator[None]:
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    def operation() -> t.Tensor:
        shared.fill_(1.0 if active[0] else 2.0)
        return shared

    with pytest.raises(AssertionError):
        gcg_paired.measure_paired(
            operation,
            1,
            reference=reference,
            numerics=lambda value: value,
            warmups=0,
        )
    assert not active[0]
