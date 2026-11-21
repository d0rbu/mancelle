from __future__ import annotations

from scripts.benchmark.benchmark_autointerp_concurrency import run_benchmark


def test_benchmark_reports_parity_and_speedup() -> None:
    result = run_benchmark(n_targets=16, n_trials=4, max_concurrent=8, delay_seconds=0.02)

    assert result["artifact_parity"] is True
    assert result["max_concurrent"] == 8
    assert result["serial_seconds"] > 0 and result["concurrent_seconds"] > 0  # ty: ignore[unsupported-operator]
