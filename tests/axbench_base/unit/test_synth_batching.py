"""Unit tests for bounded parallel synth request batching."""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from concurrent.futures import ALL_COMPLETED
from typing import Any

import pytest
from loguru import logger as _loguru_logger

from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_base.synth import batching as _batching
from steering.eval.axbench_base.synth.batching import (
    SynthRequest,
    SynthResult,
    complete_synth_request_chains,
    complete_synth_requests,
)
from steering.eval.axbench_base.types import AxBenchBaseError


class _DelayedClient:
    def __init__(self, delays: dict[str, float] | None = None) -> None:
        self.delays = delays or {}
        self.calls: list[tuple[str, str | None]] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.lock = threading.Lock()

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        del system_prompt
        with self.lock:
            self.calls.append((prompt, cache_key))
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            threading.Event().wait(self.delays.get(prompt, 0.0))
            return f"response:{prompt}"
        finally:
            with self.lock:
                self.in_flight -= 1


class _FailingClient:
    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        del cache_key
        if prompt == "bad":
            raise AxBenchBaseError("synthetic outage")
        return f"ok:{prompt}"


class _ExplodingClient:
    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        del cache_key
        if prompt == "boom":
            raise RuntimeError("unexpected failure")
        return f"ok:{prompt}"


class _SystemRecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        self.calls.append((prompt, cache_key, system_prompt))
        return f"ok:{prompt}"


def _requests(prompts: list[str]) -> list[SynthRequest]:
    return [
        SynthRequest(
            index=i,
            prompt=prompt,
            cache_key=f"key-{i}",
            attrs={"request_id": i},
        )
        for i, prompt in enumerate(prompts)
    ]


def _run(client: Any, requests: list[SynthRequest], max_concurrent: int):
    return complete_synth_requests(
        client,
        requests,
        max_concurrent=max_concurrent,
        profile_region_name="test.synth_complete",
        batch_region_name="test.synth_batch",
    )


def _run_chains(
    client: Any,
    requests: list[SynthRequest],
    *,
    handle_result,
    max_concurrent: int,
    terminal_chain_refill=None,
) -> None:
    complete_synth_request_chains(
        client,
        requests,
        handle_result=handle_result,
        terminal_chain_refill=terminal_chain_refill,
        max_concurrent=max_concurrent,
        profile_region_name="test.synth_chain_complete",
        pool_region_name="test.synth_chain_pool",
    )


def _run_chain_batches(
    client: Any,
    requests: list[SynthRequest],
    *,
    handle_results,
    max_concurrent: int,
    result_batch_size: int,
    terminal_chain_refill=None,
) -> None:
    complete_synth_request_chains(
        client,
        requests,
        handle_results=handle_results,
        result_batch_size=result_batch_size,
        terminal_chain_refill=terminal_chain_refill,
        max_concurrent=max_concurrent,
        profile_region_name="test.synth_chain_complete",
        pool_region_name="test.synth_chain_pool",
    )


def test_complete_synth_requests_preserves_input_order_when_workers_finish_out_of_order():
    client = _DelayedClient({"slow": 0.04, "fast": 0.0, "medium": 0.02})

    results = _run(client, _requests(["slow", "fast", "medium"]), max_concurrent=3)

    assert [result.index for result in results] == [0, 1, 2]
    assert [result.response_text for result in results] == [
        "response:slow",
        "response:fast",
        "response:medium",
    ]


def test_complete_synth_requests_honors_max_concurrent_cap():
    client = _DelayedClient({f"p{i}": 0.02 for i in range(8)})

    _run(client, _requests([f"p{i}" for i in range(8)]), max_concurrent=2)

    assert client.peak_in_flight <= 2
    assert client.peak_in_flight > 1


def test_complete_synth_requests_captures_axbench_errors_per_request():
    results = _run(_FailingClient(), _requests(["good", "bad", "also-good"]), 3)

    assert results[0].response_text == "ok:good"
    assert results[0].error is None
    assert results[1].response_text is None
    assert isinstance(results[1].error, AxBenchBaseError)
    assert results[2].response_text == "ok:also-good"


def test_complete_synth_requests_max_concurrent_one_is_sequential():
    client = _DelayedClient()

    results = _run(client, _requests(["a", "b", "c"]), max_concurrent=1)

    assert [response.response_text for response in results] == [
        "response:a",
        "response:b",
        "response:c",
    ]
    assert client.peak_in_flight == 1


def test_complete_synth_requests_forwards_system_prompt_per_request():
    client = _SystemRecordingClient()
    requests = [
        SynthRequest(
            index=0,
            prompt="with-system",
            cache_key="key-0",
            system_prompt="literal protocol",
        ),
    ]

    results = _run(client, requests, max_concurrent=1)

    assert results[0].response_text == "ok:with-system"
    assert client.calls == [
        ("with-system", "key-0", "literal protocol"),
    ]


def test_complete_synth_requests_propagates_unexpected_exceptions():
    with pytest.raises(RuntimeError, match="unexpected failure"):
        _run(_ExplodingClient(), _requests(["ok", "boom", "later"]), 3)


def test_complete_synth_requests_emits_batch_and_queue_profile_events(tmp_path):
    client = _DelayedClient({"a": 0.01, "b": 0.01})
    profile_dir = tmp_path / "profile"

    with AxBenchProfiler(profile_dir, synchronize_cuda=False):
        _run(client, _requests(["a", "b"]), max_concurrent=2)

    events = [json.loads(line) for line in (profile_dir / "events.jsonl").read_text().splitlines()]
    assert any(
        event["type"] == "region" and event["name"] == "test.synth_batch" for event in events
    )
    assert any(
        event["type"] == "region" and event["name"] == "test.synth_complete" for event in events
    )
    assert any(
        event["type"] == "counter" and event["name"] == "synth.parallel.queue_wait_s"
        for event in events
    )


class _EventChainClient:
    def __init__(self) -> None:
        self.release_slow = threading.Event()
        self.slow_started = threading.Event()
        self.fast_retry_started = threading.Event()
        self.queued_started = threading.Event()
        self.calls: list[str] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.per_chain_in_flight: dict[int, int] = {}
        self.lock = threading.Lock()

    def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
        del cache_key
        chain_index = int(prompt.split("-", 1)[0])
        with self.lock:
            self.calls.append(prompt)
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.per_chain_in_flight[chain_index] = self.per_chain_in_flight.get(chain_index, 0) + 1
            assert self.per_chain_in_flight[chain_index] == 1
        try:
            if prompt == "1-slow-0":
                self.slow_started.set()
                assert self.release_slow.wait(timeout=5)
            elif prompt == "0-fast-1":
                self.fast_retry_started.set()
            elif prompt == "2-queued-0":
                self.queued_started.set()
            return f"response:{prompt}"
        finally:
            with self.lock:
                self.in_flight -= 1
                self.per_chain_in_flight[chain_index] -= 1


def test_request_chains_retry_and_refill_without_waiting_for_straggler():
    client = _EventChainClient()
    initial = [
        SynthRequest(index=0, prompt="0-fast-0", cache_key="0-0"),
        SynthRequest(index=1, prompt="1-slow-0", cache_key="1-0"),
        SynthRequest(index=2, prompt="2-queued-0", cache_key="2-0"),
    ]
    errors: list[BaseException] = []

    def handle_result(
        request: SynthRequest,
        result: SynthResult,
    ) -> SynthRequest | None:
        assert result.error is None
        if request.prompt == "0-fast-0":
            return SynthRequest(
                index=request.index,
                prompt="0-fast-1",
                cache_key="0-1",
            )
        return None

    def run() -> None:
        try:
            _run_chains(
                client,
                initial,
                handle_result=handle_result,
                max_concurrent=2,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    assert client.slow_started.wait(timeout=5)
    assert client.fast_retry_started.wait(timeout=5)
    assert client.queued_started.wait(timeout=5)
    assert client.calls.index("0-fast-1") < client.calls.index("2-queued-0")
    assert not client.release_slow.is_set()
    client.release_slow.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert client.peak_in_flight == 2


def test_request_chains_axbench_error_is_handled_as_lazy_next_attempt():
    attempts: list[int] = []

    def handle_result(
        request: SynthRequest,
        result: SynthResult,
    ) -> SynthRequest | None:
        attempt = int(request.attrs["attempt"])
        attempts.append(attempt)
        if result.error is not None:
            assert attempt == 0
            return SynthRequest(
                index=request.index,
                prompt="good",
                cache_key="key-1",
                attrs={"attempt": 1},
            )
        assert result.response_text == "ok:good"
        return None

    _run_chains(
        _FailingClient(),
        [
            SynthRequest(
                index=0,
                prompt="bad",
                cache_key="key-0",
                attrs={"attempt": 0},
            ),
        ],
        handle_result=handle_result,
        max_concurrent=4,
    )

    assert attempts == [0, 1]


def test_request_chains_propagate_unexpected_worker_exceptions():
    with pytest.raises(RuntimeError, match="unexpected failure"):
        _run_chains(
            _ExplodingClient(),
            _requests(["ok", "boom", "later"]),
            handle_result=lambda _request, _result: None,
            max_concurrent=3,
        )


def test_request_chain_handler_failure_abandons_without_waiting():
    release_blocked = threading.Event()
    blocked_started = threading.Event()

    class _BlockedSiblingClient:
        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            del cache_key
            if prompt == "blocked":
                blocked_started.set()
                assert release_blocked.wait(timeout=5)
            else:
                assert blocked_started.wait(timeout=5)
            return f"response:{prompt}"

    def handle_result(
        request: SynthRequest,
        _result: SynthResult,
    ) -> None:
        if request.prompt == "handler-error":
            raise ValueError("synthetic handler failure")

    failures: list[BaseException] = []
    finished = threading.Event()

    def _invoke() -> None:
        try:
            _run_chains(
                _BlockedSiblingClient(),
                _requests(["handler-error", "blocked"]),
                handle_result=handle_result,
                max_concurrent=2,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    runner = threading.Thread(target=_invoke)
    runner.start()
    try:
        assert finished.wait(timeout=5), "handler failure waited for sibling"
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert str(failures[0]) == "synthetic handler failure"
        assert not release_blocked.is_set()
    finally:
        release_blocked.set()
        runner.join(timeout=5)
        assert not runner.is_alive()


def test_request_chain_handler_runs_only_on_calling_thread():
    calling_thread = threading.get_ident()
    handler_threads: list[int] = []

    def handle_result(
        _request: SynthRequest,
        _result: SynthResult,
    ) -> None:
        handler_threads.append(threading.get_ident())

    _run_chains(
        _DelayedClient(),
        _requests(["a", "b", "c"]),
        handle_result=handle_result,
        max_concurrent=3,
    )

    assert handler_threads == [calling_thread] * 3


def test_request_chains_emit_pool_profile_and_lifecycle_counters(tmp_path):
    profile_dir = tmp_path / "profile"

    with AxBenchProfiler(profile_dir, synchronize_cuda=False):
        _run_chains(
            _DelayedClient(),
            _requests(["a", "b"]),
            handle_result=lambda _request, _result: None,
            max_concurrent=2,
        )

    events = [json.loads(line) for line in (profile_dir / "events.jsonl").read_text().splitlines()]
    assert any(
        event["type"] == "region" and event["name"] == "test.synth_chain_pool" for event in events
    )
    counter_names = {event["name"] for event in events if event["type"] == "counter"}
    assert {
        "synth.chain_pool.chains",
        "synth.chain_pool.requests_submitted",
        "synth.chain_pool.requests_completed",
    } <= counter_names


def test_request_chains_reject_handler_index_change():
    def handle_result(
        request: SynthRequest,
        result: SynthResult,
    ) -> SynthRequest | None:
        del result
        return SynthRequest(
            index=request.index + 1,
            prompt="wrong-chain",
            cache_key="wrong-chain",
        )

    with pytest.raises(AxBenchBaseError, match="changed its stable index"):
        _run_chains(
            _DelayedClient(),
            _requests(["first"]),
            handle_result=handle_result,
            max_concurrent=1,
        )


def test_request_chains_require_unique_canonical_initial_indices():
    requests = [
        SynthRequest(index=1, prompt="one", cache_key="one"),
        SynthRequest(index=0, prompt="zero", cache_key="zero"),
    ]

    with pytest.raises(AxBenchBaseError, match="canonical ascending order"):
        _run_chains(
            _DelayedClient(),
            requests,
            handle_result=lambda _request, _result: None,
            max_concurrent=2,
        )


def test_terminal_chain_refill_appends_fresh_work_to_queue_tail() -> None:
    client = _DelayedClient()
    completed: list[int] = []
    refilled = False

    def handle_result(
        request: SynthRequest,
        result: SynthResult,
    ) -> None:
        assert result.error is None
        completed.append(request.index)

    def refill(request: SynthRequest) -> list[SynthRequest]:
        nonlocal refilled
        if request.index != 0 or refilled:
            return []
        refilled = True
        return [
            SynthRequest(
                index=3,
                prompt="replacement",
                cache_key="replacement",
            )
        ]

    _run_chains(
        client,
        _requests(["initial-0", "initial-1", "initial-2"]),
        handle_result=handle_result,
        terminal_chain_refill=refill,
        max_concurrent=1,
    )

    assert completed == [0, 1, 2, 3]
    assert [prompt for prompt, _key in client.calls] == [
        "initial-0",
        "initial-1",
        "initial-2",
        "replacement",
    ]


def test_terminal_chain_refill_fills_all_vacant_worker_slots() -> None:
    replacements_started = threading.Barrier(3)

    class _ReplacementBarrierClient(_DelayedClient):
        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            if prompt.startswith("replacement-"):
                with self.lock:
                    self.calls.append((prompt, cache_key))
                    self.in_flight += 1
                    self.peak_in_flight = max(
                        self.peak_in_flight,
                        self.in_flight,
                    )
                try:
                    replacements_started.wait(timeout=5)
                    return f"response:{prompt}"
                finally:
                    with self.lock:
                        self.in_flight -= 1
            return super().complete(prompt, cache_key=cache_key)

    client = _ReplacementBarrierClient()

    def refill(request: SynthRequest) -> list[SynthRequest]:
        if request.index != 0:
            return []
        return [
            SynthRequest(
                index=index,
                prompt=f"replacement-{index}",
                cache_key=f"replacement-{index}",
            )
            for index in range(1, 4)
        ]

    _run_chains(
        client,
        _requests(["initial"]),
        handle_result=lambda _request, _result: None,
        terminal_chain_refill=refill,
        max_concurrent=3,
    )

    assert client.peak_in_flight == 3


def test_terminal_chain_refill_failure_abandons_without_waiting() -> None:
    release_blocked = threading.Event()
    blocked_started = threading.Event()

    class _BlockedSiblingClient:
        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
        ) -> str:
            del cache_key
            if prompt == "blocked":
                blocked_started.set()
                assert release_blocked.wait(timeout=5)
            else:
                assert blocked_started.wait(timeout=5)
            return f"response:{prompt}"

    def refill(request: SynthRequest) -> list[SynthRequest]:
        if request.prompt == "refill-error":
            raise ValueError("synthetic refill failure")
        return []

    failures: list[BaseException] = []
    finished = threading.Event()

    def _invoke() -> None:
        try:
            _run_chains(
                _BlockedSiblingClient(),
                _requests(["refill-error", "blocked"]),
                handle_result=lambda _request, _result: None,
                terminal_chain_refill=refill,
                max_concurrent=2,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    runner = threading.Thread(target=_invoke)
    runner.start()
    try:
        assert finished.wait(timeout=5), "refill failure waited for sibling"
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert str(failures[0]) == "synthetic refill failure"
        assert not release_blocked.is_set()
    finally:
        release_blocked.set()
        runner.join(timeout=5)
        assert not runner.is_alive()


@pytest.mark.parametrize(
    "fresh_indices",
    [
        [1],
        [2, 2],
        [3, 2],
    ],
)
def test_terminal_chain_refill_rejects_nonmonotonic_or_duplicate_indices(
    fresh_indices: list[int],
) -> None:
    def refill(_request: SynthRequest) -> list[SynthRequest]:
        return [
            SynthRequest(
                index=index,
                prompt=f"fresh-{index}",
                cache_key=f"fresh-{index}",
            )
            for index in fresh_indices
        ]

    with pytest.raises(
        AxBenchBaseError,
        match="non-canonical indices|unique, strictly increasing",
    ):
        _run_chains(
            _DelayedClient(),
            _requests(["initial-0", "initial-1"]),
            handle_result=lambda _request, _result: None,
            terminal_chain_refill=refill,
            max_concurrent=1,
        )


@pytest.mark.parametrize(
    "fresh_indices",
    [
        [1],
        [2, 2],
        [3, 2],
    ],
)
def test_batch_terminal_refill_rejects_nonmonotonic_or_duplicate_indices(
    fresh_indices: list[int],
) -> None:
    def refill(_request: SynthRequest) -> list[SynthRequest]:
        return [
            SynthRequest(
                index=index,
                prompt=f"fresh-{index}",
                cache_key=f"fresh-{index}",
            )
            for index in fresh_indices
        ]

    with pytest.raises(
        AxBenchBaseError,
        match="unique, strictly increasing",
    ):
        _run_chain_batches(
            _DelayedClient(),
            _requests(["initial-0", "initial-1"]),
            handle_results=lambda pairs: [None] * len(pairs),
            terminal_chain_refill=refill,
            max_concurrent=2,
            result_batch_size=2,
        )


def test_request_chain_timeout_is_per_active_request(monkeypatch):
    release = threading.Event()

    class _StuckClient:
        def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
            del prompt, cache_key
            assert release.wait(timeout=5)
            return "released"

    monkeypatch.setattr(_batching, "_CHAIN_REQUEST_TIMEOUT_SECONDS", 0.05)
    try:
        with pytest.raises(AxBenchBaseError, match="chain timed out"):
            _run_chains(
                _StuckClient(),
                _requests(["stuck"]),
                handle_result=lambda _request, _result: None,
                max_concurrent=1,
            )
    finally:
        release.set()


# ---------- caller-thread result batching ----------


def test_request_chain_empty_input_remains_noop_without_callback() -> None:
    complete_synth_request_chains(
        _DelayedClient(),
        [],
        result_batch_size=0,
        max_concurrent=2,
        profile_region_name="test.empty",
        pool_region_name="test.empty_pool",
    )


def test_request_chain_nonempty_input_requires_exactly_one_handler() -> None:
    with pytest.raises(AxBenchBaseError, match="exactly one"):
        complete_synth_request_chains(
            _DelayedClient(),
            _requests(["a"]),
            max_concurrent=2,
            profile_region_name="test.handlers",
            pool_region_name="test.handlers_pool",
        )
    with pytest.raises(AxBenchBaseError, match="exactly one"):
        complete_synth_request_chains(
            _DelayedClient(),
            _requests(["a"]),
            handle_result=lambda _request, _result: None,
            handle_results=lambda _pairs: [None],
            max_concurrent=2,
            profile_region_name="test.handlers",
            pool_region_name="test.handlers_pool",
        )


@pytest.mark.parametrize("batch_size", [0, -1])
def test_request_chain_batch_handler_requires_positive_cap(batch_size: int) -> None:
    with pytest.raises(AxBenchBaseError, match="result_batch_size must be >= 1"):
        _run_chain_batches(
            _DelayedClient(),
            _requests(["a"]),
            handle_results=lambda _pairs: [None],
            max_concurrent=2,
            result_batch_size=batch_size,
        )


def test_request_chain_batch_callback_is_caller_thread_and_workers_are_provider_only() -> None:
    calling_thread = threading.get_ident()
    provider_threads: list[int] = []
    callback_threads: list[int] = []

    class _ThreadRecordingClient(_DelayedClient):
        def complete(
            self,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            del system_prompt
            provider_threads.append(threading.get_ident())
            return super().complete(prompt, cache_key=cache_key)

    def handle_results(
        pairs: list[tuple[SynthRequest, SynthResult]],
    ) -> list[None]:
        callback_threads.append(threading.get_ident())
        assert all(result.response_text is not None for _request, result in pairs)
        return [None] * len(pairs)

    _run_chain_batches(
        _ThreadRecordingClient({"a": 0.01, "b": 0.01}),
        _requests(["a", "b"]),
        handle_results=handle_results,
        max_concurrent=2,
        result_batch_size=2,
    )

    assert callback_threads
    assert set(callback_threads) == {calling_thread}
    assert provider_threads
    assert all(thread_id != calling_thread for thread_id in provider_threads)


def test_request_chain_ready_results_are_sorted_capped_and_profiled(
    monkeypatch,
) -> None:
    original_wait = _batching.wait

    def _wait_for_all(futures, *, timeout, return_when):
        del timeout, return_when
        return original_wait(futures, timeout=5, return_when=ALL_COMPLETED)

    monkeypatch.setattr(_batching, "wait", _wait_for_all)
    counters: list[tuple[str, float]] = []

    def _record_counter(name, value=1, **_attrs):
        counters.append((name, float(value)))

    monkeypatch.setattr(_batching, "record_counter", _record_counter)
    batches: list[list[int]] = []

    def handle_results(pairs):
        batches.append([request.index for request, _result in pairs])
        return [None] * len(pairs)

    requests = [
        SynthRequest(index=index, prompt=f"p{index}", cache_key=f"k{index}")
        for index in range(5)
    ]
    _run_chain_batches(
        _DelayedClient(),
        requests,
        handle_results=handle_results,
        max_concurrent=5,
        result_batch_size=2,
    )

    assert batches == [[0, 1], [2, 3], [4]]
    by_name: dict[str, list[float]] = {}
    for name, value in counters:
        by_name.setdefault(name, []).append(value)
    assert by_name["synth.chain_pool.ready_results_observed"] == [5.0, 0.0, 0.0]
    assert by_name["synth.chain_pool.batch_results_handled"] == [2.0, 2.0, 1.0]
    assert by_name["synth.chain_pool.completed_results_backlogged"] == [3.0, 0.0, 0.0]
    assert sum(by_name["synth.chain_pool.ready_results_observed"]) == 5.0
    assert sum(by_name["synth.chain_pool.completed_results_backlogged"]) == 3.0


def test_request_chain_deterministic_frontier_waits_for_its_partial_batch() -> None:
    release_slow = threading.Event()
    slow_started = threading.Event()
    callback_called = threading.Event()
    failures: list[BaseException] = []

    class _PartialClient:
        def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
            del cache_key
            if prompt == "slow":
                slow_started.set()
                assert release_slow.wait(timeout=5)
            return prompt

    def handle_results(pairs):
        assert [request.index for request, _result in pairs] == [0, 1]
        callback_called.set()
        return [None] * len(pairs)

    def _invoke() -> None:
        try:
            _run_chain_batches(
                _PartialClient(),
                _requests(["fast", "slow"]),
                handle_results=handle_results,
                max_concurrent=2,
                result_batch_size=8,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    runner = threading.Thread(target=_invoke)
    runner.start()
    try:
        assert slow_started.wait(timeout=5)
        assert not callback_called.wait(timeout=0.05)
        assert not release_slow.is_set()
    finally:
        release_slow.set()
        runner.join(timeout=5)
    assert not runner.is_alive()
    assert failures == []
    assert callback_called.is_set()


@pytest.mark.parametrize(
    "delays",
    [
        {"p0": 0.04, "p1": 0.03, "p2": 0.0, "p3": 0.01},
        {"p0": 0.0, "p1": 0.01, "p2": 0.04, "p3": 0.03},
    ],
)
def test_request_chain_batch_membership_ignores_provider_completion_order(
    delays: dict[str, float],
) -> None:
    batches: list[list[int]] = []

    def handle_results(pairs):
        batches.append([request.index for request, _result in pairs])
        return [None] * len(pairs)

    requests = [
        SynthRequest(index=index, prompt=f"p{index}", cache_key=f"k{index}")
        for index in range(4)
    ]
    _run_chain_batches(
        _DelayedClient(delays),
        requests,
        handle_results=handle_results,
        max_concurrent=4,
        result_batch_size=2,
    )

    assert batches == [[0, 1], [2, 3]]


def test_request_chain_batch_membership_is_stable_above_effective_cap() -> None:
    def observe(
        *,
        max_concurrent: int,
        delays: dict[str, float],
    ) -> list[list[int]]:
        batches: list[list[int]] = []

        def handle_results(pairs):
            batches.append([request.index for request, _result in pairs])
            return [
                (
                    SynthRequest(
                        index=0,
                        prompt="retry-0",
                        cache_key="retry-0",
                    )
                    if request.index == 0 and request.prompt == "p0"
                    else None
                )
                for request, _result in pairs
            ]

        requests = [
            SynthRequest(index=index, prompt=f"p{index}", cache_key=f"k{index}")
            for index in range(6)
        ]
        _run_chain_batches(
            _DelayedClient(delays),
            requests,
            handle_results=handle_results,
            max_concurrent=max_concurrent,
            result_batch_size=2,
        )
        return batches

    capped = observe(
        max_concurrent=2,
        delays={"p0": 0.03, "p1": 0.02, "retry-0": 0.0},
    )
    overprovisioned = observe(
        max_concurrent=4,
        delays={"p0": 0.0, "p1": 0.01, "retry-0": 0.03},
    )

    assert capped == overprovisioned == [[0, 1], [0, 2], [3, 4], [5]]


def test_request_chain_future_completing_during_callback_waits_for_next_batch() -> None:
    release_second = threading.Event()
    second_started = threading.Event()
    second_returning = threading.Event()
    batches: list[list[int]] = []

    class _DuringCallbackClient:
        def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
            del cache_key
            if prompt == "second":
                second_started.set()
                assert release_second.wait(timeout=5)
                second_returning.set()
            return prompt

    def handle_results(pairs):
        indices = [request.index for request, _result in pairs]
        batches.append(indices)
        if indices == [0]:
            assert second_started.wait(timeout=5)
            release_second.set()
            assert second_returning.wait(timeout=5)
            time.sleep(0.02)
        return [None] * len(pairs)

    _run_chain_batches(
        _DuringCallbackClient(),
        _requests(["first", "second"]),
        handle_results=handle_results,
        max_concurrent=2,
        result_batch_size=1,
    )

    assert batches == [[0], [1]]


def test_request_chain_batch_result_validation_precedes_retry_and_refill() -> None:
    refill_calls: list[int] = []
    with pytest.raises(AxBenchBaseError, match="concrete list"):
        _run_chain_batches(
            _DelayedClient(),
            _requests(["a"]),
            handle_results=lambda _pairs: (None,),
            max_concurrent=1,
            result_batch_size=2,
            terminal_chain_refill=lambda request: refill_calls.append(request.index) or [],
        )
    assert refill_calls == []

    with pytest.raises(AxBenchBaseError, match="returned 0 results for 1"):
        _run_chain_batches(
            _DelayedClient(),
            _requests(["a"]),
            handle_results=lambda _pairs: [],
            max_concurrent=1,
            result_batch_size=2,
            terminal_chain_refill=lambda request: refill_calls.append(request.index) or [],
        )
    assert refill_calls == []

    wrong = SynthRequest(index=1, prompt="retry", cache_key="retry")
    with pytest.raises(AxBenchBaseError, match="changed its stable index"):
        _run_chain_batches(
            _DelayedClient(),
            _requests(["a"]),
            handle_results=lambda _pairs: [wrong],
            max_concurrent=1,
            result_batch_size=2,
            terminal_chain_refill=lambda request: refill_calls.append(request.index) or [],
        )
    assert refill_calls == []


def test_request_chain_batch_propagates_unexpected_worker_exception() -> None:
    class _ExplodingClient:
        def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
            del prompt, cache_key
            raise ValueError("unexpected batch worker failure")

    callback_called = False

    def handle_results(_pairs):
        nonlocal callback_called
        callback_called = True
        return []

    with pytest.raises(ValueError, match="unexpected batch worker failure"):
        _run_chain_batches(
            _ExplodingClient(),
            _requests(["a"]),
            handle_results=handle_results,
            max_concurrent=1,
            result_batch_size=2,
        )
    assert not callback_called


def test_request_chain_batch_callback_failure_abandons_without_waiting() -> None:
    release_blocked = threading.Event()
    blocked_started = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []

    class _BlockedSiblingClient:
        def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
            del cache_key
            if prompt == "blocked":
                blocked_started.set()
                assert release_blocked.wait(timeout=5)
            else:
                assert blocked_started.wait(timeout=5)
            return prompt

    def _invoke() -> None:
        try:
            _run_chain_batches(
                _BlockedSiblingClient(),
                _requests(["fail", "blocked"]),
                handle_results=lambda _pairs: (_ for _ in ()).throw(
                    ValueError("batch callback failure")
                ),
                max_concurrent=2,
                result_batch_size=1,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            finished.set()

    runner = threading.Thread(target=_invoke)
    runner.start()
    try:
        assert finished.wait(timeout=5), "batch callback drained blocked sibling"
        assert len(failures) == 1
        assert isinstance(failures[0], ValueError)
        assert not release_blocked.is_set()
    finally:
        release_blocked.set()
        runner.join(timeout=5)


def test_request_chain_timeout_includes_time_spent_in_batch_callback(monkeypatch) -> None:
    release_blocked = threading.Event()
    blocked_started = threading.Event()

    class _QcBlockingClient:
        def complete(self, prompt: str, *, cache_key: str | None = None) -> str:
            del cache_key
            if prompt == "blocked-provider":
                blocked_started.set()
                assert release_blocked.wait(timeout=5)
            return prompt

    def handle_results(pairs):
        assert [request.index for request, _result in pairs] == [0]
        assert blocked_started.wait(timeout=5)
        time.sleep(0.08)
        return [None]

    monkeypatch.setattr(_batching, "_CHAIN_REQUEST_TIMEOUT_SECONDS", 0.05)
    try:
        with pytest.raises(AxBenchBaseError, match="chain timed out"):
            _run_chain_batches(
                _QcBlockingClient(),
                _requests(["fast", "blocked-provider"]),
                handle_results=handle_results,
                max_concurrent=2,
                result_batch_size=1,
            )
    finally:
        release_blocked.set()


def test_request_chain_batch_applies_all_terminals_before_aligned_refills(
    monkeypatch,
) -> None:
    original_wait = _batching.wait

    def _wait_for_all(futures, *, timeout, return_when):
        del timeout, return_when
        return original_wait(futures, timeout=5, return_when=ALL_COMPLETED)

    monkeypatch.setattr(_batching, "wait", _wait_for_all)
    terminal_results_applied = 0
    completed: list[int] = []
    refill_observations: list[tuple[int, int]] = []

    def handle_results(pairs):
        nonlocal terminal_results_applied
        indices = [request.index for request, _result in pairs]
        completed.extend(indices)
        terminal_results_applied += len(indices)
        return [None] * len(pairs)

    def refill(request):
        refill_observations.append((request.index, terminal_results_applied))
        if request.index == 0:
            return [
                SynthRequest(index=2, prompt="fresh-2", cache_key="fresh-2"),
                SynthRequest(index=3, prompt="fresh-3", cache_key="fresh-3"),
            ]
        return []

    _run_chain_batches(
        _DelayedClient(),
        _requests(["initial-0", "initial-1"]),
        handle_results=handle_results,
        max_concurrent=4,
        result_batch_size=4,
        terminal_chain_refill=refill,
    )

    assert completed == [0, 1, 2, 3]
    assert refill_observations[:2] == [(0, 2), (1, 2)]


def test_synth_request_frozen_serialization_is_unchanged() -> None:
    request = SynthRequest(
        index=7,
        prompt="prompt",
        cache_key="cache",
        system_prompt="system",
        attrs={"attempt": 2},
    )
    assert dataclasses.asdict(request) == {
        "index": 7,
        "prompt": "prompt",
        "cache_key": "cache",
        "system_prompt": "system",
        "attrs": {"attempt": 2},
    }


@pytest.mark.benchmark
def test_request_chain_batch_reduces_deterministic_qc_invocations(
    monkeypatch,
) -> None:
    """A no-provider bottleneck scales with batches, not ready results."""
    original_wait = _batching.wait

    def _wait_for_all(futures, *, timeout, return_when):
        del timeout, return_when
        return original_wait(futures, timeout=5, return_when=ALL_COMPLETED)

    monkeypatch.setattr(_batching, "wait", _wait_for_all)
    initial = [
        SynthRequest(
            index=index,
            prompt=f"chain-{index}-attempt-0",
            cache_key=f"chain-{index}-attempt-0",
            attrs={"attempt": 0},
        )
        for index in range(16)
    ]

    scalar_client = _DelayedClient()
    scalar_calls = 0
    scalar_attempts: dict[int, list[int]] = {}

    def _scalar_handler(request, _result):
        nonlocal scalar_calls
        scalar_calls += 1
        time.sleep(0.0005)
        attempt = int(request.attrs["attempt"])
        scalar_attempts.setdefault(request.index, []).append(attempt)
        if attempt == 0:
            return SynthRequest(
                index=request.index,
                prompt=f"chain-{request.index}-attempt-1",
                cache_key=f"chain-{request.index}-attempt-1",
                attrs={"attempt": 1},
            )
        return None

    _run_chains(
        scalar_client,
        initial,
        handle_result=_scalar_handler,
        max_concurrent=8,
    )

    batch_client = _DelayedClient()
    batch_calls = 0
    batch_attempts: dict[int, list[int]] = {}

    def _batch_handler(pairs):
        nonlocal batch_calls
        batch_calls += 1
        time.sleep(0.0005)
        next_requests: list[SynthRequest | None] = []
        for request, _result in pairs:
            attempt = int(request.attrs["attempt"])
            batch_attempts.setdefault(request.index, []).append(attempt)
            next_requests.append(
                SynthRequest(
                    index=request.index,
                    prompt=f"chain-{request.index}-attempt-1",
                    cache_key=f"chain-{request.index}-attempt-1",
                    attrs={"attempt": 1},
                )
                if attempt == 0
                else None
            )
        return next_requests

    _run_chain_batches(
        batch_client,
        initial,
        handle_results=_batch_handler,
        max_concurrent=8,
        result_batch_size=4,
    )

    expected_attempts = {index: [0, 1] for index in range(16)}
    expected_cache_keys = {
        f"chain-{index}-attempt-{attempt}"
        for index in range(16)
        for attempt in (0, 1)
    }
    assert scalar_attempts == expected_attempts
    assert batch_attempts == expected_attempts
    assert scalar_calls == 32
    assert batch_calls == 8
    assert {key for _prompt, key in scalar_client.calls} == expected_cache_keys
    assert {key for _prompt, key in batch_client.calls} == expected_cache_keys
    assert scalar_client.peak_in_flight <= 8
    assert batch_client.peak_in_flight <= 8


# ---------- W3: batch-level timeout ----------


def test_complete_synth_requests_batch_timeout_raises_axbenchbase_error(monkeypatch):
    """When workers exceed _BATCH_TIMEOUT_SECONDS, the batch must raise
    AxBenchBaseError instead of hanging."""
    monkeypatch.setattr(_batching, "_BATCH_TIMEOUT_SECONDS", 0.2)
    # Two requests, each sleeping 5s; with timeout=0.2s, the batch times
    # out long before they complete.
    client = _DelayedClient({"a": 5.0, "b": 5.0})
    reqs = _requests(["a", "b"])
    with pytest.raises(AxBenchBaseError) as exc_info:
        _run(client, reqs, max_concurrent=2)
    msg = str(exc_info.value)
    assert "timed out" in msg.lower() or "timeout" in msg.lower()


def test_complete_synth_requests_batch_timeout_logs_warning(monkeypatch, caplog):
    """The timeout path logs a warning naming completed/total."""
    monkeypatch.setattr(_batching, "_BATCH_TIMEOUT_SECONDS", 0.2)
    # Use loguru's bridge to capture via caplog.
    handler_id = _loguru_logger.add(caplog.handler, level="WARNING")
    try:
        client = _DelayedClient({"a": 5.0, "b": 5.0})
        reqs = _requests(["a", "b"])
        with pytest.raises(AxBenchBaseError):
            _run(client, reqs, max_concurrent=2)
    finally:
        _loguru_logger.remove(handler_id)
    has_warning = any(
        "timed out" in r.getMessage().lower() or "timeout" in r.getMessage().lower()
        for r in caplog.records
    )
    assert has_warning


def test_complete_synth_requests_batch_timeout_emits_counter(monkeypatch):
    """The timeout path emits the synth.batch_timeout_count profiler counter."""
    monkeypatch.setattr(_batching, "_BATCH_TIMEOUT_SECONDS", 0.2)

    # Capture counters by hooking record_counter directly.
    captured: list[tuple[str, float, dict]] = []

    def fake_record_counter(name, value=1, **attrs):
        captured.append((name, float(value), attrs))

    monkeypatch.setattr(_batching, "record_counter", fake_record_counter)

    client = _DelayedClient({"a": 5.0, "b": 5.0})
    reqs = _requests(["a", "b"])
    with pytest.raises(AxBenchBaseError):
        _run(client, reqs, max_concurrent=2)
    timeout_counters = [c for c in captured if c[0] == "synth.batch_timeout_count"]
    assert len(timeout_counters) == 1, (
        f"Expected exactly 1 synth.batch_timeout_count emission; got {len(timeout_counters)} (all counters: {captured})"
    )
