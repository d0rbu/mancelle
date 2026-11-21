"""Tests for the Stage 0c QcBuffer (PR-2 Task 10)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from steering.eval.axbench_base.qc_buffer import QcBuffer
from steering.eval.axbench_base.types import AxBenchBaseError


def _make_score(fid: int, outcome: str = "accepted", score: int | None = 2):
    """Build a QCFeatureScore-like mock."""
    m = MagicMock(
        feature_index=fid,
        description=f"d{fid}",
        outcome=outcome,
        score=score,
        n_examples=5,
        error_message=None,
    )
    return m


def _make_qc_client(side_effects):
    """qc_client.filter returns each side_effect in order; raise on overflow."""
    client = MagicMock()
    client.filter = MagicMock(side_effect=side_effects)
    return client


def _make_candidate(fid: int):
    return MagicMock(feature_index=fid, description=f"d{fid}")


def _require_score(buf: QcBuffer, candidate_index: int):
    score = buf.next(candidate_index)
    assert score is not None
    return score


def test_qc_buffer_invalid_batch_size_raises():
    with pytest.raises(AxBenchBaseError, match="qc_batch_size must be >= 1"):
        QcBuffer(
            qc_client=MagicMock(),
            candidates=[_make_candidate(0)],
            qc_batch_size=0,
            target_surviving=1,
        )
    with pytest.raises(AxBenchBaseError, match="qc_batch_size must be >= 1"):
        QcBuffer(
            qc_client=MagicMock(),
            candidates=[_make_candidate(0)],
            qc_batch_size=-1,
            target_surviving=1,
        )


def test_qc_buffer_single_batch_fills_correctly():
    candidates = [_make_candidate(i) for i in range(4)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(i, f"d{i}") for i in range(4)],  # all survive
            MagicMock(per_feature=[_make_score(i) for i in range(4)]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=4, target_surviving=4)

    for i in range(4):
        score = buf.next(i)
        assert score is not None
        assert score.feature_index == i
        assert score.outcome == "accepted"

    assert qc_client.filter.call_count == 1


def test_qc_buffer_multiple_batches_fill_on_demand():
    candidates = [_make_candidate(i) for i in range(8)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(i, f"d{i}") for i in range(4)],
            MagicMock(per_feature=[_make_score(i) for i in range(4)]),
        ),
        (
            [(i, f"d{i}") for i in range(4, 8)],
            MagicMock(per_feature=[_make_score(i) for i in range(4, 8)]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=4, target_surviving=8)

    # Pull all 8; should fire exactly 2 filter calls.
    for i in range(8):
        score = buf.next(i)
        assert score is not None
        assert score.feature_index == i

    assert qc_client.filter.call_count == 2


def test_qc_buffer_idempotent_repeat_calls():
    """Calling next(N) twice returns the same cached score; no extra filter."""
    candidates = [_make_candidate(i) for i in range(2)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(0, "d0"), (1, "d1")],
            MagicMock(per_feature=[_make_score(0), _make_score(1)]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=2, target_surviving=2)

    a = buf.next(0)
    b = buf.next(0)  # repeat
    assert a is b  # same object, cached
    assert qc_client.filter.call_count == 1


def test_qc_buffer_target_surviving_forwarded_to_filter():
    """target_surviving kwarg passed through to qc_client.filter."""
    candidates = [_make_candidate(i) for i in range(2)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(0, "d0")],
            MagicMock(per_feature=[_make_score(0)]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=2, target_surviving=42)
    buf.next(0)
    args, kwargs = qc_client.filter.call_args
    assert kwargs.get("target_surviving") == 42


def test_qc_buffer_returns_results_in_candidate_index_order():
    """Even if QC client returns survivors out-of-order, buffer keys on
    candidate_index → score retrieval is deterministic."""
    candidates = [_make_candidate(i) for i in range(4)]
    qc_client = _make_qc_client(side_effects=[
        (
            # Out-of-order survivors — but per_feature is what we use
            [(2, "d2"), (0, "d0")],
            MagicMock(per_feature=[
                _make_score(0, outcome="accepted"),
                _make_score(1, outcome="rejected", score=0),
                _make_score(2, outcome="accepted"),
                _make_score(3, outcome="rejected", score=0),
            ]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=4, target_surviving=2)
    assert _require_score(buf, 0).outcome == "accepted"
    assert _require_score(buf, 1).outcome == "rejected"
    assert _require_score(buf, 2).outcome == "accepted"
    assert _require_score(buf, 3).outcome == "rejected"


def test_qc_buffer_returns_none_when_per_feature_missing():
    """If per_feature does not include the expected feature_index, next()
    returns None for that index (graceful degradation)."""
    candidates = [_make_candidate(i) for i in range(3)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(0, "d0")],
            # per_feature missing index 2
            MagicMock(per_feature=[
                _make_score(0),
                _make_score(1),
            ]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=3, target_surviving=1)
    assert buf.next(0) is not None
    assert buf.next(1) is not None
    assert buf.next(2) is None  # missing


def test_qc_buffer_pool_exhaustion_returns_none():
    """next(idx) where idx >= len(candidates) returns None (no fetch)."""
    candidates = [_make_candidate(i) for i in range(2)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(0, "d0")],
            MagicMock(per_feature=[_make_score(0), _make_score(1)]),
        ),
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=2, target_surviving=2)
    buf.next(0)
    buf.next(1)
    assert buf.next(2) is None  # past pool
    assert qc_client.filter.call_count == 1  # no extra fetch


def test_qc_buffer_advances_only_when_needed():
    """Calling next(0) doesn't trigger a fetch for indices > 0."""
    candidates = [_make_candidate(i) for i in range(8)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(i, f"d{i}") for i in range(2)],
            MagicMock(per_feature=[_make_score(i) for i in range(2)]),
        ),
        # Subsequent batches not consumed in this test
    ])
    buf = QcBuffer(qc_client, candidates, qc_batch_size=2, target_surviving=8)
    buf.next(0)
    # next(0) only fills the first batch (indices 0-1)
    assert qc_client.filter.call_count == 1
    assert _require_score(buf, 1).feature_index == 1
    # Still only one fill — index 1 was in the first batch
    assert qc_client.filter.call_count == 1


def test_qc_buffer_start_at_skips_initial_candidates():
    """``start_at`` skips the leading prefix; pre-skip ``next(idx)`` returns None."""
    candidates = [_make_candidate(i) for i in range(8)]
    qc_client = _make_qc_client(side_effects=[
        (
            [(i, f"d{i}") for i in range(4, 6)],
            MagicMock(per_feature=[_make_score(i) for i in range(4, 6)]),
        ),
        (
            [(i, f"d{i}") for i in range(6, 8)],
            MagicMock(per_feature=[_make_score(i) for i in range(6, 8)]),
        ),
    ])
    buf = QcBuffer(
        qc_client, candidates,
        qc_batch_size=2, target_surviving=8, start_at=4,
    )
    # Indices < 4 should never trigger a fetch and should return None.
    assert buf.next(0) is None
    assert buf.next(3) is None
    assert qc_client.filter.call_count == 0
    # Indices >= 4 should be QC'd as normal.
    assert _require_score(buf, 4).feature_index == 4
    assert qc_client.filter.call_count == 1
    assert _require_score(buf, 5).feature_index == 5
    assert qc_client.filter.call_count == 1
    assert _require_score(buf, 7).feature_index == 7
    assert qc_client.filter.call_count == 2


def test_qc_buffer_invalid_start_at_raises():
    candidates = [_make_candidate(i) for i in range(3)]
    with pytest.raises(AxBenchBaseError, match="start_at must be in"):
        QcBuffer(
            qc_client=MagicMock(),
            candidates=candidates,
            qc_batch_size=2,
            target_surviving=3,
            start_at=-1,
        )
    with pytest.raises(AxBenchBaseError, match="start_at must be in"):
        QcBuffer(
            qc_client=MagicMock(),
            candidates=candidates,
            qc_batch_size=2,
            target_surviving=3,
            start_at=4,  # > len(candidates)=3
        )


@pytest.mark.parametrize("pipeline", [False, True])
def test_qc_buffer_results_identical_pipeline_off_vs_on(pipeline):
    """Pipeline=True yields identical scores per candidate as pipeline=False
    for the same input. The pipelining is transparent to caller semantics."""
    candidates = [_make_candidate(i) for i in range(8)]

    # Build a deterministic side_effect list (one batch per call)
    def _side_effects():
        # 4-batch + 4-batch
        return [
            (
                [(i, f"d{i}") for i in range(4)],
                MagicMock(per_feature=[_make_score(i) for i in range(4)]),
            ),
            (
                [(i, f"d{i}") for i in range(4, 8)],
                MagicMock(per_feature=[_make_score(i) for i in range(4, 8)]),
            ),
        ]

    qc_client = _make_qc_client(side_effects=_side_effects())
    buf = QcBuffer(
        qc_client, candidates,
        qc_batch_size=4, target_surviving=8, pipeline=pipeline,
    )
    try:
        results = [_require_score(buf, i).feature_index for i in range(8)]
    finally:
        buf.close()
    assert results == list(range(8))


def test_qc_buffer_pipelined_close_cancels_pending():
    """After close(), no in-flight future leaks; next prefetch is suppressed."""
    candidates = [_make_candidate(i) for i in range(8)]
    qc_client = _make_qc_client(side_effects=[
        ([(i, f"d{i}") for i in range(4)],
         MagicMock(per_feature=[_make_score(i) for i in range(4)])),
        ([(i, f"d{i}") for i in range(4, 8)],
         MagicMock(per_feature=[_make_score(i) for i in range(4, 8)])),
    ])
    buf = QcBuffer(
        qc_client, candidates,
        qc_batch_size=4, target_surviving=8, pipeline=True,
    )
    # First call: prefetch fires immediately. .next(0) drains it.
    buf.next(0)
    # close() should clean up the executor + pending future.
    buf.close()
    assert buf._executor is None
    assert buf._pending_future is None


def test_qc_buffer_context_manager_calls_close():
    """Entering as a context manager auto-closes on exit."""
    candidates = [_make_candidate(i) for i in range(2)]
    qc_client = _make_qc_client(side_effects=[
        ([(0, "d0")],
         MagicMock(per_feature=[_make_score(0), _make_score(1)])),
    ])
    with QcBuffer(
        qc_client, candidates,
        qc_batch_size=2, target_surviving=2, pipeline=True,
    ) as buf:
        score = buf.next(0)
        assert score is not None
    assert buf._executor is None


def test_qc_buffer_pipeline_overlap_smoke():
    """Verify pipelining overlaps fetch with caller work — wall time is
    closer to max(fetch_time, caller_work_time) per batch than to their sum.

    This is a smoke test (not exact perf): with a 0.3s artificial fetch
    latency and 0.2s artificial caller work between candidates, 4 candidates
    in 2 batches of 2 should complete in ~0.7s with pipelining (vs ~1.4s
    sequential). Tolerance: 1.0s upper bound to avoid CI flakiness.
    """
    import time

    n = 4
    candidates = [_make_candidate(i) for i in range(n)]

    fetch_times: list[float] = []

    def slow_filter(batch, *, target_surviving):
        fetch_times.append(time.monotonic())
        time.sleep(0.3)
        return (
            [(fid, desc) for fid, desc in batch],
            MagicMock(per_feature=[
                _make_score(fid) for fid, _ in batch
            ]),
        )

    client = MagicMock()
    client.filter = slow_filter

    start = time.monotonic()
    with QcBuffer(
        client, candidates,
        qc_batch_size=2, target_surviving=n, pipeline=True,
    ) as buf:
        for i in range(n):
            buf.next(i)
            time.sleep(0.2)  # caller work
    elapsed = time.monotonic() - start

    # Sequential lower bound: 2 batches × 0.3s + 4 × 0.2s = 1.4s
    # Pipelined upper bound (overlap): max(0.6, 0.8) ≈ 0.8s plus some
    # scheduling overhead. Generous: must be < 1.2s.
    assert elapsed < 1.2, (
        f"Pipelined wall time {elapsed:.2f}s should be < 1.2s "
        f"(sequential would be ~1.4s)"
    )
