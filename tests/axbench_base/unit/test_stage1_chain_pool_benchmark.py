"""Regression coverage for the synthetic Stage 1 scheduler benchmark."""

from scripts.benchmark.benchmark_axbench_stage1_chain_pool import run_benchmark


def test_stage1_chain_pool_benchmark_preserves_work_and_concurrency() -> None:
    report = run_benchmark(
        n_concepts=3,
        seeds_per_concept=4,
        stragglers_per_concept=1,
        retry_depth=3,
        max_concurrent=4,
        delay_s=0.001,
        minimum_speedup=0.01,
    )

    assert report["exact_attempt_schedule"] is True
    assert report["exact_call_count"] is True
    assert report["wave"]["calls"] == report["chain_pool"]["calls"] == 21
    assert report["wave"]["peak_in_flight"] <= 4
    assert report["chain_pool"]["peak_in_flight"] <= 4
