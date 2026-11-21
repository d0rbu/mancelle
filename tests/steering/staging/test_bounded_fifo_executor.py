# tests/steering/staging/test_bounded_fifo_executor.py
import threading
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.refined_types import PositiveInt
from steering.staging._queue import BoundedFifoExecutor

_PROPERTY_SETTINGS = settings(max_examples=500, derandomize=True)
_DEPTH_1 = PositiveInt.parse(1)
_DEPTH_2 = PositiveInt.parse(2)
_DEPTH_3 = PositiveInt.parse(3)
_DEPTH_4 = PositiveInt.parse(4)


@_PROPERTY_SETTINGS
@given(
    st.one_of(
        st.integers(max_value=0),
        st.booleans(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(),
        st.none(),
    )
)
def test_rejects_complete_invalid_depth_domain(bad: Any) -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        BoundedFifoExecutor(depth=bad)


@_PROPERTY_SETTINGS
@given(st.integers(min_value=1, max_value=8))
def test_accepts_positive_depth_domain(value: int) -> None:
    with BoundedFifoExecutor(depth=PositiveInt.parse(value)):
        pass


def test_runs_jobs_and_drains():
    out = []
    with BoundedFifoExecutor(depth=_DEPTH_2) as ex:
        for i in range(5):
            ex.submit(out.append, i)
    assert sorted(out) == [0, 1, 2, 3, 4]


def test_in_flight_never_exceeds_depth():
    gate = threading.Event()
    peak = {"v": 0}
    with BoundedFifoExecutor(depth=_DEPTH_2) as ex:
        def job():
            peak["v"] = max(peak["v"], ex.in_flight)
            gate.wait(1.0)
        for _ in range(6):
            ex.submit(job)
        gate.set()
    assert peak["v"] <= 2


def test_exception_propagates_in_fifo_order():
    with pytest.raises(ValueError, match="boom"):
        with BoundedFifoExecutor(depth=_DEPTH_1) as ex:
            ex.submit(lambda: (_ for _ in ()).throw(ValueError("boom")))
            ex.submit(lambda: 1)


def test_submit_after_exit_raises():
    ex = BoundedFifoExecutor(depth=_DEPTH_1)
    with ex:
        pass
    with pytest.raises(RuntimeError, match="after __exit__"):
        ex.submit(lambda: None)


def test_drain_waits_for_pending_and_stays_open():
    """drain() blocks until all in-flight jobs finish but does NOT close the pool —
    so the caller can drain inside a measured region and keep submitting. (Finding #3.)"""
    out = []
    ex = BoundedFifoExecutor(depth=_DEPTH_3)
    for i in range(5):
        ex.submit(out.append, i)
    ex.drain()
    assert sorted(out) == [0, 1, 2, 3, 4]
    assert ex.in_flight == 0
    ex.submit(out.append, 99)  # still usable after drain (not closed)
    ex.drain()
    assert 99 in out
    ex.__exit__(None, None, None)


def test_drain_surfaces_exception():
    ex = BoundedFifoExecutor(depth=_DEPTH_2)
    ex.submit(lambda: (_ for _ in ()).throw(ValueError("drain-boom")))
    with pytest.raises(ValueError, match="drain-boom"):
        ex.drain()
    ex.__exit__(None, None, None)


def test_exit_propagates_enclosing_exception_over_clean_drain():
    """An exception raised inside the `with` body must propagate — a clean drain must
    not turn it into a swallow. Teeth on `__exit__` returning False."""
    with pytest.raises(ValueError, match="enclosing"):
        with BoundedFifoExecutor(depth=_DEPTH_2) as ex:
            ex.submit(lambda: 1)  # clean job
            raise ValueError("enclosing")


def test_exit_enclosing_exception_wins_over_drain_failure():
    """A drain failure during teardown must NOT mask the enclosing exception that
    triggered teardown (the suppress-and-warn branch)."""
    with pytest.raises(ValueError, match="enclosing"):
        with BoundedFifoExecutor(depth=_DEPTH_2) as ex:
            ex.submit(lambda: (_ for _ in ()).throw(RuntimeError("drain boom")))
            raise ValueError("enclosing")  # must win over the RuntimeError drain failure


def test_exit_first_drain_exception_raised_rest_suppressed():
    """With no enclosing exception, the FIRST drain failure is raised and later ones
    are suppressed (the 'additional drain exception suppressed' branch)."""
    with pytest.raises(RuntimeError, match="first"):
        with BoundedFifoExecutor(depth=_DEPTH_4) as ex:
            ex.submit(lambda: (_ for _ in ()).throw(RuntimeError("first")))
            ex.submit(lambda: (_ for _ in ()).throw(RuntimeError("second")))


def test_get_returns_oldest_result_fifo():
    """get() pops the oldest pending result in submission order (the consumer-driven
    pipeline used by StagedPrefetcher)."""
    ex = BoundedFifoExecutor(depth=_DEPTH_3)
    for i in range(3):
        ex.submit(lambda i=i: i * 10)
    assert ex.get() == 0
    assert ex.get() == 10
    assert ex.get() == 20
    assert ex.in_flight == 0
    ex.__exit__(None, None, None)


def test_get_with_nothing_pending_raises():
    ex = BoundedFifoExecutor(depth=_DEPTH_1)
    with pytest.raises(RuntimeError, match="nothing pending"):
        ex.get()
    ex.__exit__(None, None, None)


def test_get_surfaces_job_exception():
    ex = BoundedFifoExecutor(depth=_DEPTH_1)
    ex.submit(lambda: (_ for _ in ()).throw(ValueError("get boom")))
    with pytest.raises(ValueError, match="get boom"):
        ex.get()
    ex.__exit__(None, None, None)
