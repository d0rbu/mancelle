"""Benchmark Stage 1 request chains against the former per-concept waves.

This synthetic benchmark uses the production batching helpers with a
deterministic sleeping ``SynthCompleter``. It performs no API, GPU, model,
dataset, or cache access. The baseline models the removed Stage 1b control
flow: concepts run serially and all active items finish attempt ``k`` before
any item begins attempt ``k + 1``.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from steering.eval.axbench_base.synth.batching import (
    SynthRequest,
    SynthResult,
    complete_synth_request_chains,
    complete_synth_requests,
)


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    elapsed_s: float
    calls: int
    peak_in_flight: int
    chains: int


class _LatencyCompleter:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self.per_chain_in_flight: dict[int, int] = {}
        self.attempts_by_chain: dict[int, list[int]] = {}
        self._lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del cache_key, system_prompt
        chain_text, attempt_text = prompt.split(":", maxsplit=1)
        chain = int(chain_text)
        attempt = int(attempt_text)
        with self._lock:
            self.calls += 1
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.per_chain_in_flight[chain] = self.per_chain_in_flight.get(chain, 0) + 1
            if self.per_chain_in_flight[chain] != 1:
                raise AssertionError(f"chain {chain} issued overlapping attempts")
            self.attempts_by_chain.setdefault(chain, []).append(attempt)
        try:
            threading.Event().wait(self.delay_s)
            return f"response:{prompt}"
        finally:
            with self._lock:
                self.in_flight -= 1
                self.per_chain_in_flight[chain] -= 1


def _request(chain: int, attempt: int, terminal_attempt: int) -> SynthRequest:
    return SynthRequest(
        index=chain,
        prompt=f"{chain}:{attempt}",
        cache_key=f"chain-{chain}-attempt-{attempt}",
        attrs={
            "attempt": attempt,
            "terminal_attempt": terminal_attempt,
        },
    )


def _terminal_attempts(
    *,
    n_concepts: int,
    seeds_per_concept: int,
    stragglers_per_concept: int,
    retry_depth: int,
) -> list[list[int]]:
    return [
        [
            (retry_depth if seed >= seeds_per_concept - stragglers_per_concept else 0)
            for seed in range(seeds_per_concept)
        ]
        for _concept in range(n_concepts)
    ]


def _expected_attempts(terminal_attempts: list[list[int]]) -> dict[int, list[int]]:
    expected: dict[int, list[int]] = {}
    chain = 0
    for concept_attempts in terminal_attempts:
        for terminal_attempt in concept_attempts:
            expected[chain] = list(range(terminal_attempt + 1))
            chain += 1
    return expected


def _run_wave_baseline(
    terminal_attempts: list[list[int]],
    *,
    max_concurrent: int,
    delay_s: float,
) -> tuple[BenchmarkResult, dict[int, list[int]]]:
    client = _LatencyCompleter(delay_s)
    started_at = time.perf_counter()
    chain_offset = 0
    for concept_attempts in terminal_attempts:
        for attempt in range(max(concept_attempts) + 1):
            requests = [
                _request(
                    chain_offset + seed,
                    attempt,
                    terminal_attempt,
                )
                for seed, terminal_attempt in enumerate(concept_attempts)
                if terminal_attempt >= attempt
            ]
            complete_synth_requests(
                client,
                requests,
                max_concurrent=max_concurrent,
                profile_region_name="benchmark.stage1.wave_complete",
                batch_region_name="benchmark.stage1.wave",
            )
        chain_offset += len(concept_attempts)
    elapsed_s = time.perf_counter() - started_at
    return (
        BenchmarkResult(
            mode="per_concept_attempt_waves",
            elapsed_s=elapsed_s,
            calls=client.calls,
            peak_in_flight=client.peak_in_flight,
            chains=sum(len(attempts) for attempts in terminal_attempts),
        ),
        client.attempts_by_chain,
    )


def _run_chain_pool(
    terminal_attempts: list[list[int]],
    *,
    max_concurrent: int,
    delay_s: float,
) -> tuple[BenchmarkResult, dict[int, list[int]]]:
    client = _LatencyCompleter(delay_s)
    flattened = [
        terminal_attempt
        for concept_attempts in terminal_attempts
        for terminal_attempt in concept_attempts
    ]

    def handle_result(
        request: SynthRequest,
        result: SynthResult,
    ) -> SynthRequest | None:
        if result.error is not None:
            raise result.error
        attempt = int(request.attrs["attempt"])
        terminal_attempt = int(request.attrs["terminal_attempt"])
        if attempt == terminal_attempt:
            return None
        return _request(request.index, attempt + 1, terminal_attempt)

    started_at = time.perf_counter()
    complete_synth_request_chains(
        client,
        [_request(chain, 0, terminal_attempt) for chain, terminal_attempt in enumerate(flattened)],
        handle_result=handle_result,
        max_concurrent=max_concurrent,
        profile_region_name="benchmark.stage1.chain_complete",
        pool_region_name="benchmark.stage1.chain_pool",
    )
    elapsed_s = time.perf_counter() - started_at
    return (
        BenchmarkResult(
            mode="stage_wide_request_chains",
            elapsed_s=elapsed_s,
            calls=client.calls,
            peak_in_flight=client.peak_in_flight,
            chains=len(flattened),
        ),
        client.attempts_by_chain,
    )


def run_benchmark(
    *,
    n_concepts: int,
    seeds_per_concept: int,
    stragglers_per_concept: int,
    retry_depth: int,
    max_concurrent: int,
    delay_s: float,
    minimum_speedup: float,
) -> dict[str, Any]:
    terminal_attempts = _terminal_attempts(
        n_concepts=n_concepts,
        seeds_per_concept=seeds_per_concept,
        stragglers_per_concept=stragglers_per_concept,
        retry_depth=retry_depth,
    )
    expected = _expected_attempts(terminal_attempts)
    wave, wave_attempts = _run_wave_baseline(
        terminal_attempts,
        max_concurrent=max_concurrent,
        delay_s=delay_s,
    )
    chains, chain_attempts = _run_chain_pool(
        terminal_attempts,
        max_concurrent=max_concurrent,
        delay_s=delay_s,
    )

    if wave_attempts != expected:
        raise AssertionError("wave baseline changed the exact attempt schedule")
    if chain_attempts != expected:
        raise AssertionError("request-chain pool changed the exact attempt schedule")
    if wave.calls != chains.calls:
        raise AssertionError(f"call-count drift: waves={wave.calls}, chains={chains.calls}")
    if wave.peak_in_flight > max_concurrent:
        raise AssertionError("wave baseline exceeded max_concurrent")
    if chains.peak_in_flight > max_concurrent:
        raise AssertionError("request-chain pool exceeded max_concurrent")

    speedup = wave.elapsed_s / chains.elapsed_s
    if speedup < minimum_speedup:
        raise AssertionError(
            f"request-chain speedup {speedup:.3f}x is below required {minimum_speedup:.3f}x"
        )
    return {
        "config": {
            "n_concepts": n_concepts,
            "seeds_per_concept": seeds_per_concept,
            "stragglers_per_concept": stragglers_per_concept,
            "retry_depth": retry_depth,
            "max_concurrent": max_concurrent,
            "delay_s": delay_s,
            "minimum_speedup": minimum_speedup,
        },
        "wave": asdict(wave),
        "chain_pool": asdict(chains),
        "speedup": speedup,
        "exact_attempt_schedule": True,
        "exact_call_count": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-concepts", type=int, default=8)
    parser.add_argument("--seeds-per-concept", type=int, default=12)
    parser.add_argument("--stragglers-per-concept", type=int, default=2)
    parser.add_argument("--retry-depth", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--delay-s", type=float, default=0.01)
    parser.add_argument("--minimum-speedup", type=float, default=1.2)
    args = parser.parse_args()
    if (
        args.n_concepts < 1
        or args.seeds_per_concept < 1
        or args.max_concurrent < 1
        or args.retry_depth < 0
        or args.delay_s < 0
        or args.minimum_speedup <= 0
    ):
        parser.error("counts/concurrency must be positive and other values nonnegative")
    if not 0 <= args.stragglers_per_concept <= args.seeds_per_concept:
        parser.error("stragglers-per-concept must be within the concept seed count")

    report = run_benchmark(
        n_concepts=args.n_concepts,
        seeds_per_concept=args.seeds_per_concept,
        stragglers_per_concept=args.stragglers_per_concept,
        retry_depth=args.retry_depth,
        max_concurrent=args.max_concurrent,
        delay_s=args.delay_s,
        minimum_speedup=args.minimum_speedup,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
