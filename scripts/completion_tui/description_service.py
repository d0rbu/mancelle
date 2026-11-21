"""Async description fetching service wrapping NeuronpediaClient.

Provides non-blocking feature description lookups for the TUI:
- In-memory LRU cache with disk-cache fallback
- Background fetching via ThreadPoolExecutor
- De-duplicated in-flight requests
- Thread-safe callback dispatch
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from steering.eval.neuronpedia_client import (
    NeuronpediaClient,
    NeuronpediaEmptyDescriptionError,
    NeuronpediaNotPublishedError,
)

DescriptionCallback = Callable[[int, str], None]
OnStartCallback = Callable[[int], None]

_MAX_MEM_CACHE = 4096


class DescriptionService:
    """Non-blocking wrapper around NeuronpediaClient for TUI use."""

    def __init__(
        self,
        model_id: str,
        sae_source: str,
        cache_path: Path,
    ) -> None:
        self._client = NeuronpediaClient(model_id, sae_source, cache_path)
        self._mem_cache: OrderedDict[int, str] = OrderedDict()
        # Session-only negative cache.  Missing Neuronpedia entries are not
        # written to the positive/disk cache, but should not be retried on
        # every redraw while the user browses the same TUI session.
        self._unavailable: set[int] = set()
        self._lock = threading.Lock()
        # Maps feature_id → list of pending callbacks for de-duplication
        self._in_flight: dict[int, list[DescriptionCallback]] = {}
        # Maps feature_id → on_start callback (fired when worker begins fetch)
        self._on_start: dict[int, OnStartCallback] = {}
        # Single worker serializes NeuronpediaClient calls (not thread-safe)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._shutdown = False

    def __enter__(self) -> DescriptionService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Synchronous (cache-only) lookup
    # ------------------------------------------------------------------

    def get_description_sync(self, feature_id: int) -> str | None:
        """Return cached description or None. Never hits the network."""
        with self._lock:
            if feature_id in self._unavailable:
                return ""
            if feature_id in self._mem_cache:
                self._mem_cache.move_to_end(feature_id)
                return self._mem_cache[feature_id]
        # Check the client's disk cache (no lock needed — read-only dict lookup)
        disk_val = self._client.get_cached(feature_id)
        if disk_val is not None:
            self._put_mem_cache(feature_id, disk_val)
            return disk_val
        return None

    # ------------------------------------------------------------------
    # Async (background) fetch with callback
    # ------------------------------------------------------------------

    def request_description(
        self,
        feature_id: int,
        callback: DescriptionCallback,
        on_start: OnStartCallback | None = None,
    ) -> None:
        """Request a description. Callback receives (feature_id, description).

        If cached, the callback fires synchronously in the calling thread
        (``on_start`` is NOT called in this case — the result is immediate).

        Otherwise, a background fetch is queued.  When the worker thread
        picks up the task it fires ``on_start(feature_id)`` (if provided)
        before hitting the network, then fires ``callback`` with the result.
        """
        # Fast path: already cached — skip on_start
        cached = self.get_description_sync(feature_id)
        if cached is not None:
            callback(feature_id, cached)
            return

        with self._lock:
            if self._shutdown:
                callback(feature_id, "(service shut down)")
                return

            # Check again under lock (another thread may have finished)
            if feature_id in self._unavailable:
                callback(feature_id, "")
                return
            if feature_id in self._mem_cache:
                desc = self._mem_cache[feature_id]
                callback(feature_id, desc)
                return

            # De-duplicate: if already in-flight, just add our callback
            if feature_id in self._in_flight:
                self._in_flight[feature_id].append(callback)
                return

            self._in_flight[feature_id] = [callback]
            if on_start is not None:
                self._on_start[feature_id] = on_start

        self._executor.submit(self._fetch_and_dispatch, feature_id)

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _fetch_and_dispatch(self, feature_id: int) -> None:
        """Fetch from network, cache, and call all pending callbacks."""
        # Fire on_start before the (potentially slow) network call.
        with self._lock:
            start_cb = self._on_start.pop(feature_id, None)
        if start_cb is not None:
            try:
                start_cb(feature_id)
            except Exception:
                logger.opt(exception=True).warning(
                    f"on_start callback error for feature {feature_id}"
                )

        try:
            desc = self._client.get_description(feature_id)
        except (NeuronpediaNotPublishedError, NeuronpediaEmptyDescriptionError):
            with self._lock:
                self._unavailable.add(feature_id)
            # 404s and empty explanations are expected for features/SAEs
            # that Neuronpedia has not ingested.  They are a normal empty
            # result for the TUI, not an error and not a cacheable description.
            logger.debug("No Neuronpedia description for feature {}", feature_id)
            desc = ""
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Failed to fetch description for feature {feature_id}"
            )
            desc = f"(error: {exc})"

        if desc:
            self._put_mem_cache(feature_id, desc)

        with self._lock:
            callbacks = self._in_flight.pop(feature_id, [])

        for cb in callbacks:
            try:
                cb(feature_id, desc)
            except Exception:
                logger.opt(exception=True).warning(
                    f"Callback error for feature {feature_id}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _put_mem_cache(self, feature_id: int, desc: str) -> None:
        with self._lock:
            self._mem_cache[feature_id] = desc
            self._mem_cache.move_to_end(feature_id)
            if len(self._mem_cache) > _MAX_MEM_CACHE:
                self._mem_cache.popitem(last=False)

    def get_all_cached(self) -> dict[int, str]:
        """Return a snapshot of all in-memory cached descriptions."""
        with self._lock:
            return dict(self._mem_cache)

    def shutdown(self) -> None:
        """Shut down the background executor. Safe to call multiple times."""
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=False, cancel_futures=True)
