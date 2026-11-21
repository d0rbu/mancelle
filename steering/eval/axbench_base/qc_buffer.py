"""Buffered LLM-QC for Stage 0c.

The legacy Stage 0c calls ``qc_client.filter([candidate])`` with a list of
length 1 per candidate, losing the QC client's batched fetch and
target-based early termination capabilities. This module batches calls so
a buffer fill processes ``qc_batch_size`` candidates in one ``filter``
invocation, then drains in candidate-index order.

Pipelining (PR-2 Task 12) is added on top of this synchronous
implementation; the synchronous path is exercised by ``cfg.qc_pipeline=False``
runs and is the default fallback if a pipelining failure surfaces.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Sequence
from typing import Any

from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import AxBenchBaseError


class QcBuffer:
    """Pull QC results in candidate-index order, fetched in batches.

    Args:
        qc_client: An object exposing
            ``filter(candidates, target_surviving) -> (survivors, report)``
            (the existing ``ConceptExamplesQC.filter`` API).
        candidates: Ordered list of ``CandidateConcept`` objects with
            ``feature_index`` and ``description`` attributes. The buffer
            pulls indices in increasing order; out-of-order ``next()``
            calls are NOT supported (the outer Stage 0c loop iterates
            candidates in increasing order).
        qc_batch_size: Number of candidates fetched per ``filter`` call.
            Must be >= 1.
        target_surviving: Forwarded to ``filter`` as the upper bound on
            ``survivors``. Use ``remaining_accepts_needed`` from the
            outer loop, or ``cfg.max_concepts`` if examples QC has no
            cap. ``None`` is allowed but disables early-termination.
        start_at: Initial value of the internal fill cursor. The buffer
            will never QC candidates with index ``< start_at``, even if
            ``next(idx)`` is called for one of those indices (it returns
            ``None`` directly). Used by Stage 0c partial-checkpoint
            resume to skip re-QC'ing candidates whose decisions are
            already persisted. Must be ``>= 0`` and ``<= len(candidates)``.
        pipeline: When ``True``, the buffer maintains a single in-flight
            fetch in a background thread. After the buffer drains its
            current cached range, the next batch is fetched in the
            background while the caller processes the current results.
            Subsequent ``next()`` calls block on the in-flight future,
            which is typically already complete by the time verification
            finishes (~10-30s GPU vs ~1-5s API). The caller is
            responsible for invoking :meth:`close` to shut down the
            executor on Stage 0c teardown (or use
            ``with QcBuffer(...) as buf:``).

    Raises:
        AxBenchBaseError: if ``qc_batch_size < 1`` or ``start_at`` is out
            of range.

    Pipelining
    ----------
    When ``pipeline=True``, the buffer maintains a single in-flight fetch
    in a background thread. After the buffer drains its current cached
    range, it returns immediately while the next batch is being fetched;
    the caller proceeds with verification. Subsequent ``next()`` calls
    block on the in-flight future, which is typically already complete
    by the time verification finishes (~10-30s GPU vs ~1-5s API).

    The pipelined path is semantically equivalent to the synchronous
    path: same batches, same target_surviving, same per-candidate scores.
    Only the wall-clock distribution of work changes.

    The caller is responsible for invoking :meth:`close` to shut down
    the executor on Stage 0c teardown (or use ``with QcBuffer(...) as buf:``).
    """

    def __init__(
        self,
        qc_client: Any,
        candidates: Sequence[Any],
        *,
        qc_batch_size: int,
        target_surviving: int | None,
        start_at: int = 0,
        pipeline: bool = False,
    ) -> None:
        if qc_batch_size < 1:
            raise AxBenchBaseError(
                f"QcBuffer: qc_batch_size must be >= 1, got {qc_batch_size}"
            )
        n = len(candidates)
        if start_at < 0 or start_at > n:
            raise AxBenchBaseError(
                f"QcBuffer: start_at must be in [0, {n}], got {start_at}"
            )
        self._qc_client = qc_client
        self._candidates = list(candidates)
        self._qc_batch_size = qc_batch_size
        self._target_surviving = target_surviving
        self._pipeline = pipeline
        self._scores_by_idx: dict[int, Any] = {}
        self._next_to_fill = start_at

        # Pipelining state
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending_future: concurrent.futures.Future[tuple[int, int, Any]] | None = None
        if pipeline:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="qc_buffer_prefetch",
            )
            self._kick_prefetch()

    def _kick_prefetch(self) -> None:
        """Start the next batch fetch in the background. No-op when the
        pool is exhausted or a future is already in flight."""
        if not self._pipeline or self._executor is None:
            return
        if self._pending_future is not None:
            return
        if self._next_to_fill >= len(self._candidates):
            return
        start = self._next_to_fill
        self._pending_future = self._executor.submit(
            self._fetch_batch_payload, start,
        )

    def _fetch_batch_payload(
        self, start: int,
    ) -> tuple[int, int, Any]:
        """Worker function: fetch one batch and return
        ``(start, end, report)`` for the caller to merge into the cache."""
        end = min(start + self._qc_batch_size, len(self._candidates))
        batch = [
            (c.feature_index, c.description)
            for c in self._candidates[start:end]
        ]
        with profile_region(
            "stage0c.qc_buffer.fill",
            start=start,
            end=end,
            target_surviving=self._target_surviving,
        ):
            _, report = self._qc_client.filter(
                batch,
                target_surviving=self._target_surviving,
            )
        return start, end, report

    def _drain_pending(self) -> None:
        """Block on the in-flight future and merge its results. Fires the
        next prefetch."""
        if self._pending_future is None:
            return
        start, end, report = self._pending_future.result()
        score_by_fid = {row.feature_index: row for row in report.per_feature}
        for idx, cand in enumerate(self._candidates[start:end], start=start):
            self._scores_by_idx[idx] = score_by_fid.get(cand.feature_index)
        record_counter(
            "stage0c.qc_buffer.batch_size_actual",
            end - start,
        )
        self._next_to_fill = end
        self._pending_future = None
        # Kick the next prefetch eagerly to keep the pipeline warm.
        self._kick_prefetch()

    def next(self, candidate_index: int) -> Any | None:
        """Return the QCFeatureScore for the given candidate index.

        Side-effect: triggers one or more buffer fills if the index is
        not yet cached. Returns ``None`` if the buffer fill skipped this
        candidate (e.g., early termination via ``target_surviving``, or
        the index is below ``start_at``); the caller must treat ``None``
        as "not visited" and skip verification for that candidate.

        Multiple consecutive calls with the same ``candidate_index``
        return the cached score (idempotent).

        When pipelining is enabled, this method is transparent to the
        caller: same per-candidate scores are returned regardless of
        whether ``pipeline=True`` or ``pipeline=False``.
        """
        if candidate_index in self._scores_by_idx:
            return self._scores_by_idx[candidate_index]
        # Drive fills until the index is covered or the pool is exhausted.
        while (
            self._next_to_fill <= candidate_index
            and self._next_to_fill < len(self._candidates)
        ):
            if self._pipeline:
                self._drain_pending()
            else:
                self._fill_one_batch_sync()
        return self._scores_by_idx.get(candidate_index)

    def _fill_one_batch_sync(self) -> None:
        """Synchronous-mode fill (used when pipeline=False)."""
        start, end, report = self._fetch_batch_payload(self._next_to_fill)
        score_by_fid = {row.feature_index: row for row in report.per_feature}
        for idx, cand in enumerate(self._candidates[start:end], start=start):
            self._scores_by_idx[idx] = score_by_fid.get(cand.feature_index)
        record_counter(
            "stage0c.qc_buffer.batch_size_actual",
            end - start,
        )
        self._next_to_fill = end

    def close(self) -> None:
        """Shut down the prefetch executor. Idempotent."""
        if self._executor is not None:
            # Cancel any pending future and let the worker thread finish.
            if self._pending_future is not None:
                self._pending_future.cancel()
                self._pending_future = None
            self._executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self) -> QcBuffer:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


__all__ = ["QcBuffer"]
