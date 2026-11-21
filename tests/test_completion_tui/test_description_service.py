"""Tests for DescriptionService — async wrapper around NeuronpediaClient."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

import scripts.completion_tui.description_service as ds_module
from scripts.completion_tui.description_service import DescriptionService
from steering.eval.neuronpedia_client import (
    NeuronpediaEmptyDescriptionError,
    NeuronpediaNotPublishedError,
)


@pytest.fixture
def mock_client():
    """Return a mock NeuronpediaClient with an empty cache."""
    client = MagicMock()
    client.get_cached.return_value = None
    return client


@pytest.fixture
def service(mock_client, tmp_path):
    """DescriptionService with a mocked NeuronpediaClient."""
    with patch(
        "scripts.completion_tui.description_service.NeuronpediaClient",
        return_value=mock_client,
    ):
        svc = DescriptionService(
            model_id="test-model",
            sae_source="test-sae",
            cache_path=tmp_path / "cache.yaml",
        )
    return svc


# ---------- get_description_sync ----------


def test_get_description_sync_returns_cached_value(service, mock_client):
    """Cached descriptions are returned immediately."""
    service._mem_cache[42] = "fires on greetings"
    assert service.get_description_sync(42) == "fires on greetings"


def test_get_description_sync_returns_from_disk_cache(service, mock_client):
    """Falls back to NeuronpediaClient.get_cached when not in memory."""
    mock_client.get_cached.side_effect = lambda fid: "fires on numbers" if fid == 99 else None
    result = service.get_description_sync(99)
    assert result == "fires on numbers"
    # Should also be promoted to the in-memory cache
    assert service._mem_cache[99] == "fires on numbers"


def test_get_description_sync_returns_none_for_uncached(service):
    """Returns None if the feature is not cached anywhere."""
    assert service.get_description_sync(12345) is None


# ---------- request_description ----------


def test_request_description_calls_callback_sync_for_cached(service):
    """When already cached, callback fires immediately (same call stack)."""
    service._mem_cache[7] = "cached desc"
    results = []
    service.request_description(7, lambda fid, desc: results.append((fid, desc)))
    # Callback was invoked synchronously — result available now
    assert results == [(7, "cached desc")]


def test_request_description_fetches_uncached_and_calls_callback(service, mock_client):
    """Background fetch for uncached feature, callback called with result."""
    mock_client.get_description.return_value = "fetched from API"

    event = threading.Event()
    results = []

    def cb(fid, desc):
        results.append((fid, desc))
        event.set()

    service.request_description(100, cb)
    assert event.wait(timeout=5), "Callback was not called within timeout"
    assert results == [(100, "fetched from API")]
    # Should be in memory cache now
    assert service._mem_cache[100] == "fetched from API"


def test_duplicate_requests_single_fetch(service, mock_client):
    """Multiple requests for the same feature_id produce a single fetch."""
    call_count = 0
    fetch_gate = threading.Event()

    def gated_fetch(fid):
        nonlocal call_count
        call_count += 1
        fetch_gate.wait(timeout=5)
        return "description"

    mock_client.get_description.side_effect = gated_fetch

    events = [threading.Event(), threading.Event()]
    results = []

    def make_cb(idx):
        def cb(fid, desc):
            results.append((idx, fid, desc))
            events[idx].set()
        return cb

    service.request_description(50, make_cb(0))
    service.request_description(50, make_cb(1))
    # Both requests registered — now let the fetch proceed
    fetch_gate.set()

    for e in events:
        assert e.wait(timeout=5), "Callback not called within timeout"

    assert call_count == 1, f"Expected 1 fetch but got {call_count}"
    assert len(results) == 2
    for _, fid, desc in results:
        assert fid == 50
        assert desc == "description"


def test_network_failure_returns_error_message(service, mock_client):
    """On fetch failure, callback receives the error details."""
    mock_client.get_description.side_effect = Exception("network error")

    event = threading.Event()
    results = []

    def cb(fid, desc):
        results.append((fid, desc))
        event.set()

    service.request_description(999, cb)
    assert event.wait(timeout=5), "Callback was not called within timeout"
    fid, desc = results[0]
    assert fid == 999
    assert "error" in desc.lower()
    assert "network error" in desc


@pytest.mark.parametrize(
    "missing_error",
    [
        NeuronpediaNotPublishedError("not published"),
        NeuronpediaEmptyDescriptionError("empty description"),
    ],
)
def test_known_missing_description_is_silent_and_negative_cached(
    service, mock_client, missing_error,
):
    """Expected Neuronpedia misses produce an empty, non-error result once."""
    mock_client.get_description.side_effect = missing_error
    event = threading.Event()
    results = []

    service.request_description(123, lambda fid, desc: (results.append((fid, desc)), event.set()))
    assert event.wait(timeout=5), "Callback was not called within timeout"
    assert results == [(123, "")]
    assert 123 not in service._mem_cache
    assert service.get_description_sync(123) == ""

    second_results = []
    service.request_description(123, lambda fid, desc: second_results.append((fid, desc)))
    assert second_results == [(123, "")]
    assert mock_client.get_description.call_count == 1


def test_thread_safety_concurrent_requests(service, mock_client):
    """Concurrent requests from multiple threads don't corrupt state."""
    mock_client.get_description.side_effect = lambda fid: f"desc-{fid}"

    barrier = threading.Barrier(10, timeout=5)
    results: list[tuple[int, str]] = []
    lock = threading.Lock()
    done = threading.Event()
    expected = 10

    def cb(fid, desc):
        with lock:
            results.append((fid, desc))
            if len(results) == expected:
                done.set()

    def requester(feature_id):
        barrier.wait()
        service.request_description(feature_id, cb)

    threads = [threading.Thread(target=requester, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert done.wait(timeout=5), "Not all callbacks fired"
    assert len(results) == 10
    fids = sorted(fid for fid, _ in results)
    assert fids == list(range(10))
    # Verify cache integrity
    for i in range(10):
        assert service._mem_cache[i] == f"desc-{i}"


# ---------- get_all_cached ----------


def test_get_all_cached_returns_snapshot(service):
    """get_all_cached returns a dict of all cached descriptions."""
    service._mem_cache[1] = "one"
    service._mem_cache[2] = "two"
    cached = service.get_all_cached()
    assert cached == {1: "one", 2: "two"}
    # Should be a copy, not a reference
    cached[3] = "three"
    assert 3 not in service._mem_cache


# ---------- shutdown ----------


def test_shutdown_prevents_new_submissions(service, mock_client):
    """After shutdown, callback fires synchronously with shutdown message."""
    service.shutdown()
    results = []
    service.request_description(1, lambda fid, desc: results.append((fid, desc)))
    # Callback fires synchronously in the shutdown path — no sleep needed
    assert results == [(1, "(service shut down)")]


# ---------- context manager ----------


def test_context_manager_calls_shutdown(mock_client, tmp_path):
    """Exiting the context manager shuts down the executor."""
    with patch(
        "scripts.completion_tui.description_service.NeuronpediaClient",
        return_value=mock_client,
    ):
        with DescriptionService(
            model_id="test-model",
            sae_source="test-sae",
            cache_path=tmp_path / "cache.yaml",
        ) as svc:
            pass
    assert svc._shutdown is True


# ---------- LRU eviction ----------


def test_lru_eviction(service, mock_client, monkeypatch):
    """Oldest entries are evicted when cache exceeds _MAX_MEM_CACHE."""
    monkeypatch.setattr(ds_module, "_MAX_MEM_CACHE", 3)
    for i in range(4):
        service._put_mem_cache(i, f"desc-{i}")
    # Feature 0 should have been evicted
    assert 0 not in service._mem_cache
    assert service._mem_cache[3] == "desc-3"
    assert len(service._mem_cache) == 3


# ---------- Callback exception resilience ----------


# ---------- on_start callback ----------


def test_on_start_fires_before_network_call(service, mock_client):
    """on_start fires when the worker begins, before the network fetch."""
    order = []
    fetch_gate = threading.Event()

    def fake_fetch(fid):
        order.append("fetch")
        fetch_gate.set()
        return "description"

    mock_client.get_description.side_effect = fake_fetch

    done = threading.Event()

    def on_start(fid):
        order.append("start")

    def on_done(fid, desc):
        done.set()

    service.request_description(100, on_done, on_start=on_start)
    assert done.wait(timeout=5)
    assert order == ["start", "fetch"]


def test_on_start_not_called_when_cached(service, mock_client):
    """on_start is skipped when the description is already cached."""
    service._mem_cache[42] = "cached"
    start_called = []
    results = []

    service.request_description(
        42,
        lambda fid, desc: results.append(desc),
        on_start=lambda fid: start_called.append(fid),
    )
    assert results == ["cached"]
    assert start_called == []


def test_on_start_fires_per_feature_not_bulk(service, mock_client):
    """on_start fires individually as each feature's fetch begins."""
    gate = threading.Event()
    start_order = []
    done_count = {"n": 0}
    all_done = threading.Event()

    def slow_fetch(fid):
        # First feature blocks until gate is set
        if fid == 1:
            gate.wait(timeout=5)
        return f"desc-{fid}"

    mock_client.get_description.side_effect = slow_fetch

    def on_start(fid):
        start_order.append(fid)

    def on_done(fid, desc):
        done_count["n"] += 1
        if done_count["n"] == 2:
            all_done.set()

    # Queue two features
    service.request_description(1, on_done, on_start=on_start)
    service.request_description(2, on_done, on_start=on_start)

    # Feature 1 is fetching — its on_start should have fired, but not 2's
    import time
    time.sleep(0.1)  # let worker pick up feature 1
    assert 1 in start_order
    assert 2 not in start_order

    # Let feature 1 complete so feature 2 can start
    gate.set()
    assert all_done.wait(timeout=5)
    assert start_order == [1, 2]


# ---------- Callback exception resilience ----------


def test_failing_callback_does_not_block_others(service, mock_client):
    """A callback that raises does not prevent other callbacks from firing."""
    mock_client.get_description.return_value = "good description"

    results = []

    def bad_cb(fid, desc):
        raise RuntimeError("callback exploded")

    def good_cb(fid, desc):
        results.append((fid, desc))

    # Both callbacks registered for the same feature
    with service._lock:
        service._in_flight[200] = [bad_cb, good_cb]

    # Directly invoke the worker method to test callback isolation
    service._fetch_and_dispatch(200)

    assert results == [(200, "good description")]
