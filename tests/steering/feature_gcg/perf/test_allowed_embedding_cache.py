"""Opt-in production-shape benchmark for the issue-#712 ranking cache.

Run explicitly with::

    uv run pytest -m benchmark -s \
      tests/steering/feature_gcg/perf/test_allowed_embedding_cache.py

The raw and cached alternatives run in separate spawned processes. Holding both
roughly 672 MiB allowed-embedding pools in one process changes memory residency
and can reverse the high-``k`` comparison, which is not how either production
alternative runs. The test requires at least 6 GiB of available RAM. Timing is
informational; exact raw/cached output equality is the correctness gate.
"""

from __future__ import annotations

import gc
import json
import math
import multiprocessing
import statistics
import time
import traceback
from queue import Empty
from typing import Any, Literal

import pytest
import torch as t

from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    rank_replacements_from_gradients,
)

_VOCAB = 32_768
_DIM = 5_376
_ALLOWED_COUNT = 32_734
_MIN_AVAILABLE_BYTES = 6 * 1024**3
_REPETITIONS = 5
_WORKER_TIMEOUT_S = 180.0
_KS = (1, 2, 4, 8, 16, 32)
_MAX_THREADS = 16


def _linux_mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


def _normalized_ranking(result: dict[int, tuple[Any, ...]]) -> tuple[Any, ...]:
    return tuple(
        (
            position,
            tuple(
                (replacement.position, replacement.token_id, replacement.score)
                for replacement in replacements
            ),
        )
        for position, replacements in sorted(result.items())
    )


def _benchmark_worker(
    mode: Literal["raw", "cached"],
    output_queue: Any,
) -> None:
    """Measure one alternative in a fresh process with no co-resident pool."""

    try:
        # Match the single-process production CLI on the issue-report host. The
        # recorded value remains bounded by a caller's lower OMP thread limit.
        t.set_num_threads(min(_MAX_THREADS, t.get_num_threads()))
        generator = t.Generator().manual_seed(712)
        weight = t.randn(_VOCAB, _DIM, dtype=t.float32, generator=generator)
        allowed = tuple(range(1, _ALLOWED_COUNT + 1))
        cache = (
            AllowedEmbeddingCache(
                embedding_weight=weight,
                allowed_token_ids=allowed,
            )
            if mode == "cached"
            else None
        )
        gradients = {k: t.randn(k + 1, _DIM, dtype=t.float32, generator=generator) for k in _KS}

        def rank(k: int):
            return rank_replacements_from_gradients(
                embedding_gradient=gradients[k],
                embedding_weight=weight,
                editable_positions=range(1, k + 1),
                allowed_token_ids=(cache.allowed_token_ids if cache is not None else allowed),
                current_tokens=(0, *range(100, 100 + k)),
                top_k=32,
                distance_weight=10.0,
                allowed_embedding_cache=cache,
            )

        first_started = time.perf_counter()
        first = rank(1)
        first_call_s = time.perf_counter() - first_started
        outputs: dict[int, tuple[Any, ...]] = {1: _normalized_ranking(first)}
        rows: list[dict[str, Any]] = []
        for k in _KS:
            expected = rank(k)
            outputs[k] = _normalized_ranking(expected)
            times: list[float] = []
            for _ in range(_REPETITIONS):
                started = time.perf_counter()
                result = rank(k)
                times.append(time.perf_counter() - started)
                if _normalized_ranking(result) != outputs[k]:
                    raise AssertionError(f"{mode} ranking changed across repetitions for k={k}")
                gc.collect()
            rows.append(
                {
                    "k": k,
                    "median_s": statistics.median(times),
                    "p90_s": _p90(times),
                    "times_s": times,
                }
            )

        output_queue.put(
            {
                "status": "ok",
                "mode": mode,
                "threads": t.get_num_threads(),
                "first_call_s": first_call_s,
                "rows": rows,
                "outputs": outputs,
            }
        )
    except BaseException:  # noqa: BLE001 - relay child failure to pytest parent
        output_queue.put(
            {
                "status": "error",
                "mode": mode,
                "traceback": traceback.format_exc(),
            }
        )


def _run_isolated(mode: Literal["raw", "cached"]) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    output_queue = context.Queue()
    process = context.Process(target=_benchmark_worker, args=(mode, output_queue))
    process.start()
    try:
        try:
            payload = output_queue.get(timeout=_WORKER_TIMEOUT_S)
        except Empty:
            process.terminate()
            process.join(timeout=10.0)
            pytest.fail(f"{mode} production-shape benchmark exceeded {_WORKER_TIMEOUT_S:.0f}s")
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
            pytest.fail(f"{mode} production-shape benchmark did not exit")
        assert process.exitcode == 0, (
            f"{mode} production-shape benchmark exited with {process.exitcode}"
        )
        assert payload["status"] == "ok", payload.get("traceback")
        return payload
    finally:
        output_queue.close()
        output_queue.join_thread()
        process.close()


@pytest.mark.benchmark
def test_production_shape_allowed_embedding_cache_benchmark() -> None:
    available = _linux_mem_available_bytes()
    if available is not None and available < _MIN_AVAILABLE_BYTES:
        pytest.skip(
            "production-shape ranking benchmark requires at least 6 GiB "
            f"available RAM; found {available / 1024**3:.2f} GiB"
        )

    raw = _run_isolated("raw")
    cached = _run_isolated("cached")
    assert raw["threads"] == cached["threads"]
    assert raw["outputs"] == cached["outputs"]

    results: list[dict[str, float | int]] = []
    for raw_row, cached_row in zip(raw["rows"], cached["rows"], strict=True):
        assert raw_row["k"] == cached_row["k"]
        raw_median = float(raw_row["median_s"])
        cached_median = float(cached_row["median_s"])
        results.append(
            {
                "k": int(raw_row["k"]),
                "raw_median_s": raw_median,
                "raw_p90_s": float(raw_row["p90_s"]),
                "cached_median_s": cached_median,
                "cached_p90_s": float(cached_row["p90_s"]),
                "median_speedup": raw_median / cached_median,
            }
        )

    print(
        json.dumps(
            {
                "shape": [_VOCAB, _DIM],
                "allowed_count": _ALLOWED_COUNT,
                "threads": raw["threads"],
                "repetitions": _REPETITIONS,
                "process_isolated": True,
                "first_raw_call_s": raw["first_call_s"],
                "first_cached_call_s": cached["first_call_s"],
                "equal_weight_total_raw_median_s": sum(
                    float(row["raw_median_s"]) for row in results
                ),
                "equal_weight_total_cached_median_s": sum(
                    float(row["cached_median_s"]) for row in results
                ),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
