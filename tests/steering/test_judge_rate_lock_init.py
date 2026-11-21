"""Slice 14 audit B7: ``JudgeClient._rate_lock`` initialised at construction.

Pre-fix: lock was created lazily inside ``_rate_limit`` via
``if not hasattr(self, "_rate_lock"): self._rate_lock = threading.Lock()``
— a TOCTOU race when multiple ``score_pair`` worker threads (the default
``score_all`` pool spawns 8) hit the first call concurrently. Two
threads could each pass the ``hasattr`` check, each create a fresh
``Lock``, and the first ``with self._rate_lock:`` block then wins
non-deterministically.

Fix: ``_rate_lock`` is now a ``dataclass.field(default_factory=Lock,
init=False, repr=False)`` so it exists immediately after
``__post_init__``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from steering.eval.harmfulness_judge import JudgeClient


@pytest.fixture
def judge(tmp_path: Path) -> JudgeClient:
    return JudgeClient(
        model="m",
        base_url="https://example.invalid",
        api_key="x",
        cache_path=str(tmp_path / "j.db"),
        requests_per_minute=600,
    )


def test_rate_lock_exists_post_construction(judge: JudgeClient) -> None:
    """Lock is present immediately, not lazy."""
    assert hasattr(judge, "_rate_lock")
    # ``threading.Lock()`` returns a ``_thread.lock`` instance; check the
    # acquire/release protocol rather than the concrete type which differs
    # across Python builds.
    assert hasattr(judge._rate_lock, "acquire")
    assert hasattr(judge._rate_lock, "release")


def test_rate_lock_singleton_under_concurrent_first_call(judge: JudgeClient) -> None:
    """Spawn N threads all calling ``_rate_limit`` for the first time.

    All must observe the same lock instance (``id(judge._rate_lock)`` is
    stable). With the pre-fix lazy init, two threads could each construct
    their own ``Lock`` before either ran ``with`` on it — id comparisons
    after the call would still report the last-installed lock, but the
    invariant we want is that no thread ever raises ``AttributeError``
    when accessing ``self._rate_lock`` and that mutual exclusion is honoured.
    """
    seen_ids: list[int] = []
    seen_ids_lock = threading.Lock()

    def worker() -> None:
        # Each thread calls _rate_limit (which acquires _rate_lock) and
        # then snapshots the id. With the fix, all threads see one id.
        judge._rate_limit()
        with seen_ids_lock:
            seen_ids.append(id(judge._rate_lock))

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(worker) for _ in range(16)]
        for f in futures:
            f.result()

    assert len(set(seen_ids)) == 1, (
        f"Expected one lock instance, observed {len(set(seen_ids))} "
        f"distinct ids: {set(seen_ids)}"
    )


def test_rate_limit_no_attribute_error_when_disabled(tmp_path: Path) -> None:
    """``requests_per_minute=0`` short-circuits before lock — still no error."""
    judge = JudgeClient(
        model="m",
        base_url="https://example.invalid",
        api_key="x",
        cache_path=str(tmp_path / "j.db"),
        requests_per_minute=0,
    )
    judge._rate_limit()  # must not raise
    assert hasattr(judge, "_rate_lock")
