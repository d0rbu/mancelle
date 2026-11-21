"""Bounded parallel LLM request helpers for Stage 0c and Stage 1b/1c."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from loguru import logger

from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import AxBenchBaseError

# Each request may use the transport retry budget (default 3), each attempt
# taking up to (connect_timeout + read_timeout + backoff) ≈ 70s. Per-request worst case:
# 4 × 70s + (1+2+4)s ≈ 287s. Multiply by the caller's worker bound for the
# absolute worst-case wall-clock of a fully serialized batch. We add headroom
# for rate-limiter waits. 1800s (30 min) accommodates a 1500-request batch when
# explicitly limited to 60 RPM while catching sockets that never time out.
_BATCH_TIMEOUT_SECONDS = 1800
_CHAIN_REQUEST_TIMEOUT_SECONDS = _BATCH_TIMEOUT_SECONDS


class SynthCompleter(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class SynthRequest:
    """One synth LLM request with its deterministic output slot."""

    index: int
    prompt: str
    cache_key: str
    system_prompt: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SynthRequestWithPayload(SynthRequest):
    """A request carrying caller-only state without changing legacy requests."""

    payload: Any


@dataclass(frozen=True)
class SynthResult:
    """One synth LLM result, preserving the originating request index."""

    index: int
    response_text: str | None = None
    error: AxBenchBaseError | None = None


@dataclass(frozen=True)
class _ActiveChainRequest:
    request: SynthRequest
    submitted_at: float


SynthChainResultHandler = Callable[
    [SynthRequest, SynthResult],
    SynthRequest | None,
]
SynthChainBatchResultHandler = Callable[
    [list[tuple[SynthRequest, SynthResult]]],
    list[SynthRequest | None],
]
SynthTerminalChainRefill = Callable[[SynthRequest], list[SynthRequest]]


def complete_synth_requests(
    llm_client: SynthCompleter,
    requests: list[SynthRequest],
    *,
    max_concurrent: int,
    profile_region_name: str,
    batch_region_name: str,
) -> list[SynthResult]:
    """Run synth requests with bounded concurrency and stable result order.

    The returned list is sorted by ``SynthRequest.index`` regardless of
    completion order. ``AxBenchBaseError`` from the LLM client is captured in
    the corresponding result so Stage 0c/1b/1c can apply their per-attempt
    retry policies. Other exception types are allowed to propagate because the
    pre-existing serial code did not handle them either.
    """
    if not requests:
        return []

    worker_count = min(max(1, max_concurrent), len(requests))
    record_counter(
        "synth.parallel.requests",
        len(requests),
        profile_region=profile_region_name,
        max_concurrent=max_concurrent,
        worker_count=worker_count,
    )
    with profile_region(
        batch_region_name,
        requests=len(requests),
        max_concurrent=max_concurrent,
        worker_count=worker_count,
    ):
        if worker_count == 1:
            return [
                _run_one(
                    llm_client,
                    request,
                    profile_region_name=profile_region_name,
                    submitted_at=time.perf_counter(),
                )
                for request in requests
            ]

        executor = ThreadPoolExecutor(max_workers=worker_count)
        shutdown_started = False
        futures: dict[Future[SynthResult], SynthRequest] = {}
        try:
            for request in requests:
                submitted_at = time.perf_counter()
                context = copy_context()
                future = cast(
                    Future[SynthResult],
                    executor.submit(
                        context.run,
                        _run_one,
                        llm_client,
                        request,
                        profile_region_name=profile_region_name,
                        submitted_at=submitted_at,
                    ),
                )
                futures[future] = request

            results: list[SynthResult] = []
            try:
                for future in as_completed(futures, timeout=_BATCH_TIMEOUT_SECONDS):
                    try:
                        results.append(future.result())
                    except BaseException:
                        for pending in futures:
                            if pending is not future:
                                pending.cancel()
                        shutdown_started = True
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
            except FuturesTimeoutError as exc:
                n_done = sum(1 for f in futures if f.done())
                n_total = len(futures)
                record_counter(
                    "synth.batch_timeout_count",
                    1,
                    completed=n_done,
                    total=n_total,
                    timeout_seconds=_BATCH_TIMEOUT_SECONDS,
                )
                logger.warning(
                    f"Synth batch timed out after {_BATCH_TIMEOUT_SECONDS}s: "
                    f"{n_done}/{n_total} futures completed; abandoning stuck threads"
                )
                for pending in futures:
                    pending.cancel()
                shutdown_started = True
                executor.shutdown(wait=False, cancel_futures=True)
                raise AxBenchBaseError(
                    f"Synth batch timed out after {_BATCH_TIMEOUT_SECONDS}s "
                    f"({n_done}/{n_total} completed)"
                ) from exc
        finally:
            if not shutdown_started:
                executor.shutdown(wait=True, cancel_futures=False)

    return sorted(results, key=lambda result: result.index)


def complete_synth_request_chains(
    llm_client: SynthCompleter,
    initial_requests: list[SynthRequest],
    *,
    handle_result: SynthChainResultHandler | None = None,
    handle_results: SynthChainBatchResultHandler | None = None,
    result_batch_size: int = 1,
    max_concurrent: int,
    profile_region_name: str,
    pool_region_name: str,
    terminal_chain_refill: SynthTerminalChainRefill | None = None,
) -> None:
    """Run independent, sequential request chains in one bounded worker pool.

    ``initial_requests`` are admitted in their canonical index order. Each
    active chain owns one pool slot until the selected result handler returns
    ``None`` for it. ``handle_result`` preserves the original scalar callback
    path. ``handle_results`` receives a bounded, stable-index-sorted list of
    requests on the calling thread. ``result_batch_size`` caps a deterministic
    frontier of the lowest active stable chain indices; the callback waits for
    that frontier rather than letting provider completion timing choose GPU
    batch membership. Returning another request lazily submits the next step
    for that same chain; it is invalid for the returned request to change the
    chain index. When a handler terminates a chain, an optional
    ``terminal_chain_refill`` callback may append fresh chains to the global
    queue tail. Fresh indices must be unique and strictly increasing across the
    pool. Existing callers omit the batch callback and retain the exact
    fixed-chain behavior. Worker threads only execute :func:`_run_one`; all
    callbacks run on the calling thread, where Stage 1 parsing, GPU QC, and
    mutable state live.
    """
    if not initial_requests:
        return

    if (handle_result is None) == (handle_results is None):
        raise AxBenchBaseError(
            "Synth request chains require exactly one of handle_result or "
            "handle_results"
        )
    if handle_results is not None and result_batch_size < 1:
        raise AxBenchBaseError(
            "Synth request-chain result_batch_size must be >= 1, got "
            f"{result_batch_size}"
        )

    indices = [request.index for request in initial_requests]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        raise AxBenchBaseError(
            "Synth request chains require unique initial indices in "
            f"canonical ascending order, got {indices!r}"
        )

    worker_count = (
        max(1, max_concurrent)
        if terminal_chain_refill is not None
        else min(max(1, max_concurrent), len(initial_requests))
    )
    queued = deque(initial_requests)
    last_chain_index = indices[-1]
    record_counter(
        "synth.chain_pool.chains",
        len(initial_requests),
        profile_region=profile_region_name,
        max_concurrent=max_concurrent,
        worker_count=worker_count,
    )

    with profile_region(
        pool_region_name,
        chains=len(initial_requests),
        max_concurrent=max_concurrent,
        worker_count=worker_count,
    ):
        executor = ThreadPoolExecutor(max_workers=worker_count)
        shutdown_started = False
        active: dict[Future[SynthResult], _ActiveChainRequest] = {}
        observed_ready: set[Future[SynthResult]] = set()
        observed_backlogged: set[Future[SynthResult]] = set()

        def submit(request: SynthRequest) -> None:
            submitted_at = time.perf_counter()
            context = copy_context()
            future = cast(
                Future[SynthResult],
                executor.submit(
                    context.run,
                    _run_one,
                    llm_client,
                    request,
                    profile_region_name=profile_region_name,
                    submitted_at=submitted_at,
                ),
            )
            active[future] = _ActiveChainRequest(
                request=request,
                submitted_at=submitted_at,
            )
            record_counter(
                "synth.chain_pool.requests_submitted",
                1,
                profile_region=profile_region_name,
                active=len(active),
                queued=len(queued),
                **request.attrs,
            )

        def abandon() -> None:
            nonlocal shutdown_started
            for future in active:
                future.cancel()
            shutdown_started = True
            executor.shutdown(wait=False, cancel_futures=True)

        try:
            while queued and len(active) < worker_count:
                submit(queued.popleft())

            while active:
                now = time.perf_counter()
                earliest_deadline = min(
                    state.submitted_at + _CHAIN_REQUEST_TIMEOUT_SECONDS for state in active.values()
                )
                done, _pending = wait(
                    active,
                    timeout=max(0.0, earliest_deadline - now),
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    # Close the race where a future completes exactly as the
                    # deadline expires before declaring it stuck.
                    done = {future for future in active if future.done()}
                if not done:
                    now = time.perf_counter()
                    expired = sorted(
                        (
                            state
                            for future, state in active.items()
                            if (
                                not future.done()
                                and state.submitted_at + _CHAIN_REQUEST_TIMEOUT_SECONDS <= now
                            )
                        ),
                        key=lambda state: state.request.index,
                    )
                    if not expired:
                        continue
                    timed_out = expired[0]
                    record_counter(
                        "synth.chain_request_timeout_count",
                        1,
                        chain_index=timed_out.request.index,
                        active=len(active),
                        queued=len(queued),
                        timeout_seconds=_CHAIN_REQUEST_TIMEOUT_SECONDS,
                        **timed_out.request.attrs,
                    )
                    logger.warning(
                        "Synth request chain timed out after {}s: "
                        "chain_index={}, active={}, queued={}; abandoning "
                        "stuck threads",
                        _CHAIN_REQUEST_TIMEOUT_SECONDS,
                        timed_out.request.index,
                        len(active),
                        len(queued),
                    )
                    abandon()
                    raise AxBenchBaseError(
                        "Synth request chain timed out after "
                        f"{_CHAIN_REQUEST_TIMEOUT_SECONDS}s "
                        f"(chain_index={timed_out.request.index}, "
                        f"active={len(active)}, queued={len(queued)})"
                    )

                if handle_result is not None:
                    # Keep the original scalar path structurally separate from
                    # the batch seam. It remains the reference implementation
                    # for Stage 0c and Stage 1 runs configured with scalar QC.
                    completed = sorted(
                        done,
                        key=lambda future: active[future].request.index,
                    )
                    for future in completed:
                        state = active.pop(future)
                        request = state.request
                        try:
                            result = future.result()
                            next_request = handle_result(request, result)
                        except BaseException:
                            abandon()
                            raise

                        record_counter(
                            "synth.chain_pool.requests_completed",
                            1,
                            profile_region=profile_region_name,
                            active=len(active),
                            queued=len(queued),
                            **request.attrs,
                        )
                        if next_request is not None:
                            if next_request.index != request.index:
                                abandon()
                                raise AxBenchBaseError(
                                    "Synth request chain handler changed its stable "
                                    f"index from {request.index} to "
                                    f"{next_request.index}"
                                )
                            submit(next_request)
                        else:
                            if terminal_chain_refill is not None:
                                fresh_requests = terminal_chain_refill(request)
                                for fresh_request in fresh_requests:
                                    if fresh_request.index <= last_chain_index:
                                        abandon()
                                        raise AxBenchBaseError(
                                            "Synth terminal-chain refill requires "
                                            "unique, strictly increasing indices; "
                                            f"last={last_chain_index}, "
                                            f"fresh={fresh_request.index}"
                                        )
                                    last_chain_index = fresh_request.index
                                    queued.append(fresh_request)
                                    record_counter(
                                        "synth.chain_pool.refill_chains",
                                        1,
                                        profile_region=profile_region_name,
                                        queued=len(queued),
                                        **fresh_request.attrs,
                                    )
                            if queued:
                                submit(queued.popleft())
                else:
                    assert handle_results is not None
                    # Provider completion timing must not choose GPU batch
                    # membership. The next callback owns the lowest stable
                    # chain indices that were active at this scheduling
                    # boundary and waits for that entire frontier. Higher-index
                    # requests continue running in parallel and may become a
                    # completed backlog, but they cannot enter this batch just
                    # because their provider happened to return first.
                    selected = sorted(
                        active,
                        key=lambda future: active[future].request.index,
                    )[:result_batch_size]
                    while any(not future.done() for future in selected):
                        pending_frontier = [
                            future for future in selected if not future.done()
                        ]
                        pending_active = [
                            (future, state)
                            for future, state in active.items()
                            if not future.done()
                        ]
                        if not pending_active:
                            break
                        now = time.perf_counter()
                        earliest_pending_deadline = min(
                            state.submitted_at + _CHAIN_REQUEST_TIMEOUT_SECONDS
                            for _future, state in pending_active
                        )
                        wait(
                            pending_frontier,
                            timeout=max(0.0, earliest_pending_deadline - now),
                            return_when=ALL_COMPLETED,
                        )
                        if all(future.done() for future in selected):
                            break
                        now = time.perf_counter()
                        expired = sorted(
                            (
                                state
                                for future, state in pending_active
                                if (
                                    not future.done()
                                    and state.submitted_at
                                    + _CHAIN_REQUEST_TIMEOUT_SECONDS
                                    <= now
                                )
                            ),
                            key=lambda state: state.request.index,
                        )
                        if not expired:
                            continue
                        timed_out = expired[0]
                        record_counter(
                            "synth.chain_request_timeout_count",
                            1,
                            chain_index=timed_out.request.index,
                            active=len(active),
                            queued=len(queued),
                            timeout_seconds=_CHAIN_REQUEST_TIMEOUT_SECONDS,
                            **timed_out.request.attrs,
                        )
                        logger.warning(
                            "Synth request chain timed out after {}s: "
                            "chain_index={}, active={}, queued={}; abandoning "
                            "stuck threads",
                            _CHAIN_REQUEST_TIMEOUT_SECONDS,
                            timed_out.request.index,
                            len(active),
                            len(queued),
                        )
                        abandon()
                        raise AxBenchBaseError(
                            "Synth request chain timed out after "
                            f"{_CHAIN_REQUEST_TIMEOUT_SECONDS}s "
                            f"(chain_index={timed_out.request.index}, "
                            f"active={len(active)}, queued={len(queued)})"
                        )

                    ready = sorted(
                        (future for future in active if future.done()),
                        key=lambda future: active[future].request.index,
                    )
                    selected_set = set(selected)
                    backlog = [
                        future for future in ready if future not in selected_set
                    ]
                    newly_ready = [
                        future for future in ready if future not in observed_ready
                    ]
                    newly_backlogged = [
                        future
                        for future in backlog
                        if future not in observed_backlogged
                    ]
                    observed_ready.update(newly_ready)
                    observed_backlogged.update(newly_backlogged)
                    record_counter(
                        "synth.chain_pool.ready_results_observed",
                        len(newly_ready),
                        profile_region=profile_region_name,
                        result_batch_size=result_batch_size,
                        ready_now=len(ready),
                    )
                    record_counter(
                        "synth.chain_pool.completed_results_backlogged",
                        len(newly_backlogged),
                        profile_region=profile_region_name,
                        result_batch_size=result_batch_size,
                        backlogged_now=len(backlog),
                    )
                    # Selected futures leave ``active`` below, so they can be
                    # forgotten immediately. Only a still-backlogged future
                    # remains referenced long enough to prevent recounting it
                    # on the next scheduling boundary.
                    observed_ready.difference_update(selected_set)
                    observed_backlogged.difference_update(selected_set)

                    completed_pairs: list[tuple[SynthRequest, SynthResult]] = []
                    for future in selected:
                        state = active.pop(future)
                        try:
                            completed_pairs.append(
                                (state.request, future.result())
                            )
                        except BaseException:
                            abandon()
                            raise

                    try:
                        next_requests = handle_results(completed_pairs)
                    except BaseException:
                        abandon()
                        raise
                    if not isinstance(next_requests, list):
                        abandon()
                        raise AxBenchBaseError(
                            "Synth request-chain batch handler must return a "
                            "concrete list"
                        )
                    if len(next_requests) != len(completed_pairs):
                        abandon()
                        raise AxBenchBaseError(
                            "Synth request-chain batch handler returned "
                            f"{len(next_requests)} results for "
                            f"{len(completed_pairs)} completed requests"
                        )
                    # Validate the whole callback result before submitting any
                    # retry or invoking any terminal refill. A malformed later
                    # entry therefore cannot leave a partially advanced batch.
                    for (request, _result), next_request in zip(
                        completed_pairs,
                        next_requests,
                        strict=True,
                    ):
                        if (
                            next_request is not None
                            and next_request.index != request.index
                        ):
                            abandon()
                            raise AxBenchBaseError(
                                "Synth request chain batch handler changed its "
                                f"stable index from {request.index} to "
                                f"{next_request.index}"
                            )

                    record_counter(
                        "synth.chain_pool.batch_results_handled",
                        len(completed_pairs),
                        profile_region=profile_region_name,
                        result_batch_size=result_batch_size,
                    )
                    for (request, _result), next_request in zip(
                        completed_pairs,
                        next_requests,
                        strict=True,
                    ):
                        record_counter(
                            "synth.chain_pool.requests_completed",
                            1,
                            profile_region=profile_region_name,
                            active=len(active),
                            queued=len(queued),
                            **request.attrs,
                        )
                        if next_request is not None:
                            # A same-chain retry owns one of the slots vacated
                            # by the selected completed results and is admitted
                            # immediately, preserving sequential chain progress.
                            submit(next_request)
                            continue
                        if terminal_chain_refill is None:
                            continue
                        fresh_requests = terminal_chain_refill(request)
                        for fresh_request in fresh_requests:
                            if fresh_request.index <= last_chain_index:
                                abandon()
                                raise AxBenchBaseError(
                                    "Synth terminal-chain refill requires "
                                    "unique, strictly increasing indices; "
                                    f"last={last_chain_index}, "
                                    f"fresh={fresh_request.index}"
                                )
                            last_chain_index = fresh_request.index
                            queued.append(fresh_request)
                            record_counter(
                                "synth.chain_pool.refill_chains",
                                1,
                                profile_region=profile_region_name,
                                queued=len(queued),
                                **fresh_request.attrs,
                            )
                # A terminal callback can release several deficits at once
                # (for example, when the first successful chain lifts a
                # zero-hit admission guard). Fill every newly vacant slot
                # after processing the completed batch; fresh chains still
                # remain behind all older queued work.
                while queued and len(active) < worker_count:
                    submit(queued.popleft())
        except BaseException:
            if not shutdown_started:
                abandon()
            raise
        finally:
            if not shutdown_started:
                executor.shutdown(wait=True, cancel_futures=False)


def _run_one(
    llm_client: SynthCompleter,
    request: SynthRequest,
    *,
    profile_region_name: str,
    submitted_at: float,
) -> SynthResult:
    queue_wait_s = time.perf_counter() - submitted_at
    record_counter(
        "synth.parallel.queue_wait_s",
        queue_wait_s,
        profile_region=profile_region_name,
        **request.attrs,
    )
    record_counter(
        "synth.parallel.inflight_start",
        1,
        profile_region=profile_region_name,
        **request.attrs,
    )
    try:
        with profile_region(profile_region_name, **request.attrs):
            try:
                if request.system_prompt is None:
                    # Preserve the exact historical call signature for every
                    # existing user-only synth request. Besides compatibility,
                    # this lets strict test doubles prove that adding optional
                    # system messages cannot perturb the null/local or Stage 1
                    # paths.
                    response_text = llm_client.complete(
                        request.prompt,
                        cache_key=request.cache_key,
                    )
                else:
                    response_text = llm_client.complete(
                        request.prompt,
                        cache_key=request.cache_key,
                        system_prompt=request.system_prompt,
                    )
            except AxBenchBaseError as exc:
                return SynthResult(index=request.index, error=exc)
            return SynthResult(index=request.index, response_text=response_text)
    finally:
        record_counter(
            "synth.parallel.inflight_finish",
            1,
            profile_region=profile_region_name,
            **request.attrs,
        )
