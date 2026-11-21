"""Neuronpedia API client with disk-cached feature descriptions.

The :class:`NeuronpediaClient` wraps the Neuronpedia public ``/api/feature``
endpoint with:

* per-request timeout (the upstream :mod:`neuronpedia` library has none,
  so a hung connection used to stall Stage 0a indefinitely);
* retry-with-exponential-backoff and explicit ``Retry-After`` honoring on
  429s;
* a ``ThreadPoolExecutor`` batch path for :meth:`get_descriptions` that
  trades the previous serial ~1.4 req/sec for ~14 req/sec at the
  documented account quota;
* a single, batched ``_save_cache`` call per batch (was: one save per
  fetched feature -> O(N^2) disk work for a 16k scan);
* a namespaced cache file format keyed on ``(model_id, sae_source)`` so
  two different SAE configurations sharing the same ``output_root`` can
  no longer serve each other's descriptions.

A legacy flat-dict cache (no ``(model_id, sae_source)`` metadata) cannot
be verified against the current source, so it is discarded with a warning
and its descriptions re-fetched under the current namespaced format. See
:func:`NeuronpediaClient._load_cache`.
"""

from __future__ import annotations

import fcntl
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml
from loguru import logger

from errors import DataLoadError
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.neuronpedia_config import NeuronpediaConfig

__all__ = [
    "NEURONPEDIA_API_KEY_ENV",
    "NEURONPEDIA_BASE_URL",
    "NeuronpediaAuthError",
    "NeuronpediaClient",
    "NeuronpediaEmptyDescriptionError",
    "NeuronpediaError",
    "NeuronpediaFetchError",
    "NeuronpediaNotPublishedError",
    "NeuronpediaRateLimitError",
    "NeuronpediaSourceMismatchError",
    "fetch_feature_description",
]


NEURONPEDIA_API_KEY_ENV = "NEURONPEDIA_API_KEY"
NEURONPEDIA_BASE_URL = "https://neuronpedia.org/api"


# Security — Wave 3B task 5. The Neuronpedia ``/api/feature`` endpoint
# echoes user-visible error messages in 4xx/5xx response bodies. Those
# bodies can include the API key prefix, the caller-supplied
# ``feature_id``, and server-side diagnostic text. Embedding the body
# into an exception or log record leaks that content into loguru /
# stdlib / cloud log aggregators at ERROR severity on every failed
# fetch. See ``steering/eval/axbench_base/llm_judge_client.py`` for
# the sibling tripwire; the default of ``0`` is enforced by
# ``tests/test_openrouter_body_leak.py``. A future debug build MAY
# flip it to e.g. ``200`` but the default MUST remain ``0``.
_LOG_MAX_BODY_BYTES = 0


class NeuronpediaError(Exception):
    """Base for all Neuronpedia client errors."""


class NeuronpediaAuthError(NeuronpediaError):
    """Raised on missing or rejected API key (401 or unset env var)."""


class NeuronpediaRateLimitError(NeuronpediaError):
    """Raised on a 429 response after the retry budget is exhausted.

    Carries the last server-suggested ``Retry-After`` value so callers
    can surface it for diagnostics.
    """

    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class NeuronpediaFetchError(NeuronpediaError):
    """Raised on any other unsuccessful fetch (4xx other than 401/429,
    5xx after retry exhaustion, malformed JSON, missing description)."""


class NeuronpediaEmptyDescriptionError(NeuronpediaFetchError):
    """Raised when a feature returns HTTP 200 but carries no usable
    description (empty ``explanations`` list or a falsy ``description``).

    Subclasses :class:`NeuronpediaFetchError` so existing
    ``except NeuronpediaFetchError`` handlers (e.g. autointerp, issue #691)
    keep treating it as a non-fatal per-feature drop. It is a *distinct*
    type only so the batch path can count it separately: a source that does
    not host an index may answer 200-with-empty rather than 404, so a *mass*
    empty rate is a source/SAE mismatch signal alongside the 404 rate."""


class NeuronpediaNotPublishedError(RuntimeError):
    """Raised when a Neuronpedia URL returns 404 for a feature that ought
    to exist (e.g. a gemma-scope-2 SAE that Neuronpedia doesn't host yet).
    Callers should treat this as a known-limitation, not a hard bug."""


class NeuronpediaSourceMismatchError(NeuronpediaError):
    """Raised by :meth:`NeuronpediaClient.get_descriptions` when the
    not-published (404) rate exceeds ``max_not_published_rate`` OR the
    total-missing (404 + 200-empty) rate exceeds ``max_missing_rate``.

    A single missing description is a known per-feature limitation (see
    :class:`NeuronpediaNotPublishedError` /
    :class:`NeuronpediaEmptyDescriptionError`), but a *high* rate almost always
    means the configured Neuronpedia ``model_id`` / ``sae_source`` does not
    correspond to the SAE being queried (e.g. a 16k source against a 65k SAE).
    Left unguarded that silently truncates and mislabels the downstream concept
    pool, so load-bearing callers set positive thresholds and fail loud
    (``0.0`` disables)."""


def _require_api_key() -> str:
    """Read :data:`NEURONPEDIA_API_KEY_ENV` from the environment.

    Treats an empty-string value as unset (matches the OPENROUTER pattern
    in :mod:`steering.eval.axbench_base.openrouter`)."""
    key = os.environ.get(NEURONPEDIA_API_KEY_ENV) or ""
    if not key:
        raise NeuronpediaAuthError(
            f"{NEURONPEDIA_API_KEY_ENV} is not set in the environment. "
            f"Set it via `export {NEURONPEDIA_API_KEY_ENV}=np-...` and "
            f"re-run."
        )
    return key


def _parse_description(payload: Any) -> str:
    """Extract the first explanation's description from a Neuronpedia
    feature payload.

    The API returns a JSON object that includes an ``explanations``
    list; each entry has a ``description`` string field. Both
    ``explanations`` empty and ``description`` falsy are treated as a
    fetch error so the caller's "drop on failure" path can skip the
    feature."""
    if not isinstance(payload, dict):
        raise NeuronpediaFetchError(
            f"Unexpected response shape (not a dict): {type(payload).__name__}"
        )
    explanations = payload.get("explanations") or []
    if not explanations:
        raise NeuronpediaEmptyDescriptionError(
            f"No explanations in response payload (keys={sorted(payload)})"
        )
    first = explanations[0]
    if not isinstance(first, dict):
        raise NeuronpediaFetchError(
            f"Explanation entry is not a dict: {type(first).__name__}"
        )
    desc = first.get("description")
    if not desc:
        raise NeuronpediaEmptyDescriptionError(
            f"No description in first explanation (keys={sorted(first)})"
        )
    if not isinstance(desc, str):
        raise NeuronpediaFetchError(
            f"Explanation description has unexpected type "
            f"{type(desc).__name__} (expected str)"
        )
    return desc


def _validate_cache_descriptions(
    descriptions: dict[Any, Any],
    *,
    cache_path: Path,
) -> dict[int, str]:
    """Validate the exact persisted feature-description schema."""
    validated: dict[int, str] = {}
    for feature_id, description in descriptions.items():
        if (
            not isinstance(feature_id, int)
            or isinstance(feature_id, bool)
            or feature_id < 0
        ):
            raise DataLoadError(
                f"Neuronpedia cache at {cache_path} has invalid feature ID "
                f"{feature_id!r}; expected a non-negative integer"
            )
        if not isinstance(description, str) or not description.strip():
            raise DataLoadError(
                f"Neuronpedia cache at {cache_path} has invalid description "
                f"for feature {feature_id}: expected a non-empty string"
            )
        validated[feature_id] = description
    return validated


def _parse_retry_after(value: str | None, *, cap_seconds: float) -> float | None:
    """Parse a ``Retry-After`` header into seconds.

    Per RFC 7231 section 7.1.3 the value can be either an integer-seconds
    value or an HTTP-date. We honor the integer form (the format the
    Neuronpedia API uses) and ignore the date form (the wait would be
    >> our cap anyway, so falling through to exponential backoff is
    safe). Negative values are treated as "no header"; huge values are
    clamped to ``cap_seconds`` so a server bug can't induce a 1-hour
    wait."""
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(seconds) or math.isinf(seconds):
        # NaN comparisons all return False, so a NaN would slip past
        # the negativity check and then crash time.sleep(NaN) with
        # ValueError. Inf would clamp to cap_seconds correctly but
        # signals a server bug worth treating as "no header".
        return None
    if seconds < 0:
        return None
    return min(seconds, cap_seconds)


def _sleep_with_backoff(attempt: int, cfg: NeuronpediaConfig) -> None:
    """Exponential backoff with full jitter. Respects ``backoff_max_seconds``."""
    base = cfg.backoff_base_seconds * (2**attempt)
    base = min(base, cfg.backoff_max_seconds)
    jitter = random.uniform(0, cfg.backoff_base_seconds)
    delay = min(base + jitter, cfg.backoff_max_seconds)
    logger.debug(
        f"Neuronpedia backoff: attempt={attempt + 1} sleeping {delay:.2f}s"
    )
    time.sleep(delay)


def _sleep_for_retry_after(
    retry_after: float | None,
    attempt: int,
    cfg: NeuronpediaConfig,
) -> None:
    """If the server gave a ``Retry-After``, honor it. Otherwise
    fall back to exponential backoff."""
    if retry_after is None:
        _sleep_with_backoff(attempt, cfg)
        return
    delay = min(retry_after, cfg.backoff_max_seconds)
    logger.debug(
        f"Neuronpedia 429: honoring Retry-After={retry_after}s "
        f"(capped to {delay:.2f}s)"
    )
    time.sleep(delay)


def fetch_feature_description(
    model_id: str,
    sae_source: str,
    feature_id: int,
    *,
    config: NeuronpediaConfig | None = None,
) -> str:
    """Fetch a single feature's description from Neuronpedia.

    Includes retry-with-backoff and ``Retry-After`` honoring. See
    :class:`NeuronpediaConfig` for the tunable knobs.

    Raises:
        NeuronpediaAuthError: on missing/invalid API key.
        NeuronpediaRateLimitError: on 429 after retries exhausted.
        NeuronpediaFetchError: on 4xx/5xx after retries, network
            error, or malformed payload.
    """
    cfg = config or NeuronpediaConfig()
    api_key = _require_api_key()
    url = f"{NEURONPEDIA_BASE_URL}/feature/{model_id}/{sae_source}/{feature_id}"
    headers = {
        "X-Api-Key": api_key,
        "Accept-Encoding": "gzip",
    }
    last_retry_after: float | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=cfg.timeout_seconds)
        except requests.RequestException as exc:
            # Catch the broad RequestException so transient wire errors
            # like ChunkedEncodingError / ContentDecodingError / SSLError
            # / ProxyError are retried alongside Timeout / ConnectionError.
            # The narrow alternative (Timeout, ConnectionError) lets a
            # ChunkedEncodingError on a single feature kill the whole
            # 16k-feature batch.
            if attempt < cfg.max_retries:
                _sleep_with_backoff(attempt, cfg)
                continue
            raise NeuronpediaFetchError(
                f"Network error after {attempt + 1} attempts for "
                f"feature {feature_id}: {exc}"
            ) from exc
        status = resp.status_code
        if status is None:
            raise NeuronpediaFetchError(
                f"Neuronpedia returned no HTTP status code for "
                f"feature {feature_id}"
            )
        if status == 200:
            try:
                payload = resp.json()
            except (ValueError, requests.JSONDecodeError) as exc:
                # Defensive scrub: a 200 with malformed JSON is rare
                # but the body is still user-opaque, so log length
                # only (same rule as 4xx/5xx).
                body_len = len(resp.text or "")
                logger.warning(
                    f"Neuronpedia malformed-JSON: status=200 url={url} "
                    f"body_len={body_len}"
                )
                raise NeuronpediaFetchError(
                    f"Malformed JSON for feature {feature_id} "
                    f"(body suppressed for privacy; "
                    f"see logs for status + url + length)"
                ) from exc
            return _parse_description(payload)
        if status == 404:
            # Typed signal that Neuronpedia simply does not host this
            # feature/SAE yet (the expected state for gemma-scope-2
            # releases it hasn't ingested). This MUST be checked before
            # the generic non-retryable-4xx path below so the typed
            # error wins and consumers can skip gracefully. Body is
            # user-opaque server text — log status + URL + length only,
            # consistent with the other 4xx/5xx branches.
            body_len = len(resp.text or "")
            logger.debug(
                f"Neuronpedia 404: status=404 url={url} body_len={body_len}"
            )
            raise NeuronpediaNotPublishedError(
                f"Neuronpedia does not host feature {feature_id} for "
                f"{model_id}/{sae_source}. This is expected for SAE "
                f"releases Neuronpedia hasn't ingested yet (e.g. "
                f"gemma-scope-2). (body suppressed for privacy; see "
                f"logs for status + url + length)"
            )
        if status == 401:
            # 401 body is operator-actionable ("invalid API key") but
            # could echo the key prefix. Log status + URL, no body.
            body_len = len(resp.text or "")
            logger.warning(
                f"Neuronpedia 401: status=401 url={url} body_len={body_len}"
            )
            raise NeuronpediaAuthError(
                f"Neuronpedia returned 401 for feature {feature_id}; "
                f"check that {NEURONPEDIA_API_KEY_ENV} is correct."
            )
        if status == 429:
            last_retry_after = _parse_retry_after(
                resp.headers.get("Retry-After"),
                cap_seconds=cfg.backoff_max_seconds,
            )
            if attempt < cfg.max_retries:
                _sleep_for_retry_after(last_retry_after, attempt, cfg)
                continue
            # Retry budget exhausted on a 429. Status + URL + length
            # only; do NOT embed the body (may echo the API key prefix
            # or the requested feature_id).
            body_len = len(resp.text or "")
            logger.warning(
                f"Neuronpedia 429: status=429 url={url} "
                f"body_len={body_len} retry_after={last_retry_after}"
            )
            raise NeuronpediaRateLimitError(
                f"Neuronpedia rate-limited feature {feature_id} after "
                f"{attempt + 1} attempts (Retry-After={last_retry_after})",
                retry_after_seconds=last_retry_after,
            )
        if 500 <= status < 600:
            if attempt < cfg.max_retries:
                _sleep_with_backoff(attempt, cfg)
                continue
            # Retry budget exhausted on 5xx. Body is server
            # diagnostic text (may include stack traces, internal
            # paths, user-supplied fields). Log length only.
            body_len = len(resp.text or "")
            logger.warning(
                f"Neuronpedia 5xx: status={status} url={url} "
                f"body_len={body_len}"
            )
            raise NeuronpediaFetchError(
                f"Neuronpedia returned {status} for feature {feature_id} "
                f"after {attempt + 1} attempts "
                f"(body suppressed for privacy; "
                f"see logs for status + url + length)"
            )
        # Non-retryable 4xx (other than 401/429 handled above). The
        # body can echo the caller-supplied ``feature_id`` and any
        # server diagnostic text — log status + URL + length only.
        body_len = len(resp.text or "")
        logger.warning(
            f"Neuronpedia 4xx: status={status} url={url} "
            f"body_len={body_len}"
        )
        raise NeuronpediaFetchError(
            f"Neuronpedia returned {status} for feature {feature_id} "
            f"(body suppressed for privacy; "
            f"see logs for status + url + length)"
        )
    raise NeuronpediaFetchError(
        f"Internal: retry loop exited without return for feature {feature_id}"
    )


# ---------------------------------------------------------------------------
# NeuronpediaClient
# ---------------------------------------------------------------------------


class NeuronpediaClient:
    """Caching wrapper around Neuronpedia feature lookups.

    Cache file format (namespaced by identity):

    .. code-block:: yaml

        model_id: gemma-2-2b
        sae_source: 8-gemmascope-res-16k
        descriptions:
          42: "description for feature 42"

    A legacy flat-dict cache (root is ``{int: str}`` only, pre-namespacing) is
    NOT adopted: it is unverifiable against this client's ``(model_id,
    sae_source)`` and could carry descriptions from a different source (the
    exact issue-#657 mislabel risk). ``_load_cache`` discards it with a warning
    and re-fetches, so the on-disk cache is rewritten in the namespaced format.
    """

    def __init__(
        self,
        model_id: str,
        sae_source: str,
        cache_path: Path | str,
        *,
        config: NeuronpediaConfig | None = None,
    ) -> None:
        self._model_id = model_id
        self._sae_source = sae_source
        self._cache_path = Path(cache_path)
        self._config = config or NeuronpediaConfig()
        self._cache: dict[int, str] = {}
        self._cache_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._load_cache()

    def get_description(self, feature_id: int) -> str:
        """Return the description for ``feature_id``, fetching if not cached.

        Singular path: if the feature is missing from cache, fetch and
        persist immediately. The batched :meth:`get_descriptions` path
        defers the save until the whole batch is done.
        """
        cached = self.get_cached(feature_id)
        if cached is not None:
            return cached
        desc = fetch_feature_description(
            self._model_id, self._sae_source, feature_id,
            config=self._config,
        )
        self._save_cache({feature_id: desc})
        persisted = self.get_cached(feature_id)
        if persisted is None:  # pragma: no cover - guarded by transactional commit
            raise DataLoadError(
                f"Neuronpedia cache commit lost feature {feature_id} at "
                f"{self._cache_path}"
            )
        return persisted

    def get_descriptions(
        self,
        feature_ids: list[int],
        *,
        cache_only: bool = False,
        max_not_published_rate: float = 0.0,
        max_missing_rate: float = 0.0,
        max_failed_rate: float = 0.0,
        min_mismatch_features: int = 0,
    ) -> dict[int, str]:
        """Fetch multiple descriptions concurrently.

        Cached entries are returned without an API hit. The remaining
        entries are dispatched via :class:`ThreadPoolExecutor` with up
        to ``config.max_concurrent`` workers in flight. A failure for
        an individual feature is logged at WARNING and the feature is
        omitted from the result dict -- preserving the historical
        ``concept_list.py`` "skip on exception" contract.

        When ``cache_only`` is True, ONLY descriptions already in the
        on-disk cache are returned; uncached features are silently
        omitted and NO network calls are made. This makes a run fully
        reproducible against a frozen description set (the cache) and
        avoids re-querying features Neuronpedia does not publish (which
        are never cached and would otherwise be re-fetched every run).

        Three opt-in fail-loud guards protect callers whose concept selection
        *depends* on the descriptions (AxBench-Base Stage 0a), each raising
        :class:`NeuronpediaSourceMismatchError`:

        - ``max_not_published_rate`` -- the HTTP-404 / not-published rate over
          the requested population (a correct source hosts every index, so this
          baseline is ~0%; catches even a partial wrong source/width).
        - ``max_missing_rate`` -- the total-missing rate (404 plus
          200-with-empty-explanations), catching a source that answers
          200-empty for indices it does not host; a correct source has a low
          genuine-empty baseline, so this threshold is deliberately loose.
        - ``max_failed_rate`` -- the total-DROP rate (404 + empty + *transient*
          5xx/rate-limit after retries). A wholly nonexistent / typo'd source
          answers 5xx (not 404) for every index (see the live-API note), so its
          drops are all transient and slip the two rates above; this backstop
          catches that and a rate-limit storm that would otherwise silently
          truncate the pool. Deliberately loose (a correct, reachable source
          fails ~none).

        Each is gated by ``min_mismatch_features`` (an absolute floor) so a
        small candidate pool (whitelist / fetch-limit) does not trip on a few
        legitimately-missing features or a brief network blip. A tripped guard
        almost always means the configured ``model_id`` / ``sae_source`` does
        not match the SAE (e.g. a 16k source against a 65k SAE) or is not a real
        source. All three rates default to ``0.0`` (disabled), preserving the
        non-fatal drop contract for decorative callers (autointerp, issue #691).
        Transient failures count only toward ``max_failed_rate`` (never the
        other two); ``cache_only`` short-circuits before the guards; successful
        fetches are persisted before the guards run so a threshold bump reuses
        them.

        Authentication is handled apart from the source-mismatch guards so an
        auth problem is never misdiagnosed as a mismatch: when any guard is
        active a MISSING key fails fast up front, and a present-but-rejected key
        (per-request 401) is counted in ``n_auth`` (kept out of ``n_failed``) and
        re-raised as :class:`NeuronpediaAuthError` before the rate guards. When
        NO guard is active (decorative callers) the up-front check is skipped and
        auth errors drop per feature, so a partially-warm cache still returns its
        cached subset without a key (issue #691).
        """
        n_total = len(feature_ids)
        record_counter("neuronpedia.descriptions_requested", n_total)
        with self._cache_lock:
            cached = {
                fid: self._cache[fid]
                for fid in feature_ids
                if fid in self._cache
            }
        # Preserve caller order while issuing at most one wire request per
        # feature. Duplicate IDs otherwise race each other and can return
        # divergent upstream descriptions within one batch.
        to_fetch = list(dict.fromkeys(fid for fid in feature_ids if fid not in cached))
        n_to_fetch = len(to_fetch)
        record_counter("neuronpedia.descriptions_cached", len(cached))
        record_counter("neuronpedia.descriptions_to_fetch", n_to_fetch)
        if cache_only:
            logger.info(
                f"Neuronpedia[cache_only]: {len(cached)}/{n_total} cached; "
                f"skipping {n_to_fetch} uncached (no network)"
            )
            record_counter("neuronpedia.descriptions_cache_only_skipped", n_to_fetch)
            record_counter("neuronpedia.descriptions_not_published", 0)
            record_counter("neuronpedia.descriptions_empty", 0)
            record_counter("neuronpedia.descriptions_missing", 0)
            return cached
        logger.info(
            f"Neuronpedia: {len(cached)}/{n_total} descriptions cached, "
            f"{n_to_fetch} to fetch"
        )
        if n_to_fetch == 0:
            record_counter("neuronpedia.descriptions_fetched", 0)
            record_counter("neuronpedia.descriptions_failed", 0)
            record_counter("neuronpedia.descriptions_not_published", 0)
            record_counter("neuronpedia.descriptions_empty", 0)
            record_counter("neuronpedia.descriptions_missing", 0)
            return cached

        # Fail fast on a MISSING key before the fetch loop, but ONLY for a
        # fail-loud caller (a guard is active). Without a key every request
        # raises NeuronpediaAuthError; for a guarded caller (Stage 0a) surface
        # that up front instead of letting it trip the max_failed_rate guard and
        # misdiagnose auth as a source mismatch. Decorative callers (all rates
        # 0.0, issue #691) skip this and keep the graceful per-request drop --
        # returning whatever subset is cached when the key is absent.
        _guards_active = (
            max_not_published_rate > 0.0
            or max_missing_rate > 0.0
            or max_failed_rate > 0.0
        )
        if _guards_active:
            _require_api_key()

        fetched: dict[int, str] = {}
        n_done = 0
        n_failed = 0
        n_not_published = 0
        n_empty = 0
        n_auth = 0
        progress_step = max(1, n_to_fetch // 10) if n_to_fetch > 100 else 100

        def _one(fid: int) -> tuple[int, str]:
            return fid, fetch_feature_description(
                self._model_id, self._sae_source, fid, config=self._config,
            )

        with profile_region("neuronpedia.batch_fetch", rows=n_to_fetch):
            with ThreadPoolExecutor(
                max_workers=self._config.max_concurrent,
                thread_name_prefix="neuronpedia",
            ) as ex:
                futures = {ex.submit(_one, fid): fid for fid in to_fetch}
                for fut in as_completed(futures):
                    fid = futures[fut]
                    try:
                        _, desc = fut.result()
                    except NeuronpediaNotPublishedError as exc:
                        # A 404 for a feature the source does not host.
                        # ``NeuronpediaNotPublishedError`` is RuntimeError-derived
                        # (intentionally, so the singular ``fetch_feature_description``
                        # path doesn't swallow a 404). In the BATCH path a single
                        # un-ingested gemma-scope-2 feature (the documented expected
                        # state for v2 SAEs) is skipped + counted like any other
                        # per-feature failure, not crashing the whole batch eval.
                        # Counted separately so a caller with a positive
                        # max_not_published_rate can fail loud on a *mass* 404
                        # rate (a source/SAE mismatch) below.
                        n_failed += 1
                        n_not_published += 1
                        logger.warning(
                            f"Neuronpedia: dropping feature {fid} (not published): {exc}"
                        )
                        continue
                    except NeuronpediaEmptyDescriptionError as exc:
                        # HTTP 200 but no usable description. Non-fatal per feature
                        # (like #691), but counted separately: a source that does
                        # not host an index may answer 200-empty rather than 404,
                        # so a *mass* empty rate is a mismatch signal (guarded
                        # below by max_missing_rate) alongside the 404 rate.
                        n_failed += 1
                        n_empty += 1
                        logger.warning(
                            f"Neuronpedia: dropping feature {fid} "
                            f"(empty description): {exc}"
                        )
                        continue
                    except NeuronpediaAuthError as exc:
                        # A rejected/invalid API key 401s on EVERY request. This
                        # is an auth problem, NOT a source/SAE mismatch, so it
                        # must not feed the max_failed_rate guard. Count it apart
                        # (n_auth, kept out of n_failed) so a guarded caller can
                        # fail loud with the real auth cause below, while a
                        # decorative caller (issue #691) still drops gracefully.
                        n_auth += 1
                        logger.warning(
                            f"Neuronpedia: auth error on feature {fid}: {exc}"
                        )
                        continue
                    except NeuronpediaError as exc:
                        # Transient 5xx after retries, rate-limit, malformed
                        # JSON: non-fatal per-feature drops that do NOT signal a
                        # source mismatch.
                        n_failed += 1
                        logger.warning(
                            f"Neuronpedia: dropping feature {fid}: {exc}"
                        )
                        continue
                    fetched[fid] = desc
                    n_done += 1
                    if n_done % progress_step == 0 or n_done == n_to_fetch:
                        logger.info(
                            f"Neuronpedia: {n_done}/{n_to_fetch} fetched"
                        )

        n_missing = n_not_published + n_empty
        record_counter("neuronpedia.descriptions_fetched", len(fetched))
        record_counter("neuronpedia.descriptions_failed", n_failed)
        record_counter("neuronpedia.descriptions_not_published", n_not_published)
        record_counter("neuronpedia.descriptions_empty", n_empty)
        record_counter("neuronpedia.descriptions_missing", n_missing)
        if n_failed:
            # 404 / not-published and 200-empty take the fast path (no retry
            # loop); only the transient failures were actually retried. Report
            # them separately so a source mismatch isn't misread as retry
            # exhaustion.
            n_transient = n_failed - n_missing
            logger.info(
                f"Neuronpedia: {n_failed}/{n_to_fetch} features dropped "
                f"({n_not_published} not-published/404, {n_empty} empty, "
                f"{n_transient} transient after retries)"
            )
        record_counter("neuronpedia.descriptions_auth_failed", n_auth)
        # A rejected/invalid key 401s on every request. For a guarded caller
        # (Stage 0a) fail loud with the REAL auth cause, before the rate guards
        # can misread all-transient failures as a source mismatch. A decorative
        # caller (no guard, issue #691) falls through and returns whatever was
        # cached. n_auth is kept out of n_failed, so it never feeds the rate
        # guards regardless.
        if _guards_active and n_auth > 0:
            raise NeuronpediaAuthError(
                f"{n_auth}/{n_to_fetch} Neuronpedia requests failed "
                f"authentication (HTTP 401) for source "
                f"'{self._model_id}/{self._sae_source}'. The "
                f"{NEURONPEDIA_API_KEY_ENV} key is set but rejected by "
                "Neuronpedia; verify it is valid and current. This is an auth "
                "failure, not a source/SAE mismatch."
            )
        # Persist successful fetches (namespaced by (model_id, sae_source)) BEFORE
        # the source-mismatch guard: a later run that raises the threshold reuses
        # them instead of a cold re-fetch, and a genuine mismatch's partial
        # descriptions live only under their own (wrong) namespace, so they never
        # pollute the corrected run.
        if fetched:
            with profile_region("neuronpedia.save_cache", rows=len(fetched)):
                self._save_cache(fetched)
            try:
                record_counter(
                    "neuronpedia.cache_file_bytes",
                    self._cache_path.stat().st_size,
                )
            except OSError:
                pass
        # Fail loud on a source/SAE mismatch. Three signals, each gated by an
        # absolute floor (``min_mismatch_features``) so a small candidate pool
        # (whitelist / fetch-limit) does not trip on a handful of legitimately
        # missing features or a brief network blip. Checked most-specific first
        # so the raised message names the actual failure mode:
        #  - 404 / not-published rate: a correct source hosts every index
        #    (~0% baseline), so a tight threshold catches even a partial
        #    wrong-source / wrong-width.
        #  - total-missing (404 + 200-empty) rate: a source that answers
        #    200-with-empty for indices it does not host is caught here; a
        #    correct source has a low genuine-empty baseline, so this threshold
        #    is deliberately loose (a mass mismatch is nearly all of them).
        #  - total-failed (404 + empty + transient) rate: a wholly nonexistent /
        #    typo'd source answers 5xx (not 404) for every index, so its drops
        #    are all transient and slip the two rates above; this backstop
        #    catches that and a rate-limit storm. Loose (a reachable, correct
        #    source fails ~none).
        if n_total > 0:
            src = f"{self._model_id}/{self._sae_source}"
            n_transient = n_failed - n_missing
            # (rate, count, message) per signal, most-specific first so the
            # raised error names the actual failure mode. The gating -- enabled
            # (rate > 0), over the absolute floor (count >= min_mismatch_features),
            # and above the rate (count / n_total > rate) -- is written ONCE in
            # the loop so the three signals cannot drift apart. Messages are
            # cheap to build eagerly (at most one is raised).
            guard_signals = (
                (
                    max_not_published_rate,
                    n_not_published,
                    f"{n_not_published}/{n_total} requested Neuronpedia features "
                    f"({n_not_published / n_total:.1%}) returned 404 / "
                    f"not-published from source '{src}', exceeding "
                    f"max_not_published_rate={max_not_published_rate:.2f}. This "
                    "almost always means neuronpedia_model_id / neuronpedia_source "
                    "do not match the SAE being queried (e.g. a 16k source against "
                    "a 65k SAE); verify they correspond to the feature model and "
                    "SAE width.",
                ),
                (
                    max_missing_rate,
                    n_missing,
                    f"{n_missing}/{n_total} requested Neuronpedia features "
                    f"({n_missing / n_total:.1%}) had no usable description "
                    f"({n_not_published} 404 + {n_empty} empty) from source "
                    f"'{src}', exceeding max_missing_rate={max_missing_rate:.2f}. "
                    "A source that answers 200-empty for indices it does not host "
                    "looks like this; verify neuronpedia_model_id / "
                    "neuronpedia_source match the SAE (model + width).",
                ),
                (
                    max_failed_rate,
                    n_failed,
                    f"{n_failed}/{n_total} requested Neuronpedia features "
                    f"({n_failed / n_total:.1%}) could not be fetched "
                    f"({n_not_published} 404 + {n_empty} empty + {n_transient} "
                    f"transient after retries) from source '{src}', exceeding "
                    f"max_failed_rate={max_failed_rate:.2f}. This can mean the "
                    "neuronpedia_source does not name a real source (an unknown "
                    "source answers 5xx, not 404, for every index) OR the "
                    "Neuronpedia API is unreachable / rate-limiting. Verify "
                    "neuronpedia_source matches the SAE (model + width) and that "
                    "the API is reachable; a persistent transient failure is not a "
                    "source mismatch.",
                ),
            )
            for rate, count, message in guard_signals:
                if (
                    rate > 0.0
                    and count >= min_mismatch_features
                    and count / n_total > rate
                ):
                    raise NeuronpediaSourceMismatchError(message)
        with self._cache_lock:
            return {
                fid: self._cache[fid]
                for fid in feature_ids
                if fid in self._cache
            }

    def get_cached(self, feature_id: int) -> str | None:
        """Return a cached description without hitting the network."""
        with self._cache_lock:
            return self._cache.get(feature_id)

    def cache_size(self) -> int:
        """Return the number of identity-validated committed descriptions."""
        with self._cache_lock:
            return len(self._cache)

    # -- cache I/O ----------------------------------------------------------

    def _load_cache(self) -> None:
        self._cache = self._read_cache_file()
        if self._cache:
            logger.debug(
                f"Loaded {len(self._cache)} cached Neuronpedia descriptions from {self._cache_path}"
            )

    def _read_cache_file(self) -> dict[int, str]:
        """Read and identity-validate the complete on-disk cache."""
        if self._cache_path.is_symlink():
            raise DataLoadError(
                f"Refusing symlinked Neuronpedia cache path {self._cache_path}; "
                "use a regular file at the canonical identity-addressed path"
            )
        if not self._cache_path.is_file():
            return {}
        try:
            with open(self._cache_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            raise DataLoadError(
                f"Corrupted Neuronpedia cache at {self._cache_path}: {exc}"
            ) from exc
        except OSError as exc:
            # Race: file existed at is_file() check but became inaccessible
            # before open() (deletion, permission change, etc.).
            raise DataLoadError(
                f"Could not read Neuronpedia cache at "
                f"{self._cache_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise DataLoadError(
                f"Neuronpedia cache at {self._cache_path} has unexpected "
                f"top-level type {type(raw).__name__} (expected dict)"
            )

        if not raw:
            return {}

        if self._is_legacy_cache(raw):
            # A legacy flat cache carries no (model_id, sae_source) metadata, so
            # it cannot be verified against the current source. Adopting it could
            # silently reuse mislabeled descriptions and -- if it covers the
            # candidate set -- let get_descriptions early-return before the
            # source-mismatch guard (issue #657). Discard it (do NOT crash every
            # caller): descriptions are re-fetched under the current source, so
            # the guard runs, and _save_cache rewrites the file namespaced.
            logger.warning(
                f"Ignoring legacy (non-namespaced) Neuronpedia cache at "
                f"{self._cache_path}: it has no (model_id, sae_source) metadata "
                f"and cannot be verified against the current "
                f"(model_id={self._model_id!r}, sae_source={self._sae_source!r}). "
                "Descriptions will be re-fetched under the current source (so the "
                "source-mismatch guard still runs) and a validated, namespaced "
                "cache rewritten. Delete the file to silence this warning."
            )
            return {}

        cache_model = raw.get("model_id")
        cache_source = raw.get("sae_source")
        if cache_model != self._model_id or cache_source != self._sae_source:
            raise DataLoadError(
                f"Neuronpedia cache at {self._cache_path} was built for "
                f"(model_id={cache_model!r}, sae_source={cache_source!r}) but "
                f"client was opened with "
                f"(model_id={self._model_id!r}, sae_source={self._sae_source!r}). "
                f"Either point at a different cache file, delete this one, "
                f"or open the client with matching args."
            )
        if "descriptions" not in raw:
            raise DataLoadError(
                f"Neuronpedia cache at {self._cache_path} is missing required "
                "`descriptions` mapping"
            )
        descriptions = raw["descriptions"]
        if not isinstance(descriptions, dict):
            raise DataLoadError(
                f"Neuronpedia cache at {self._cache_path} has unexpected "
                f"`descriptions` type {type(descriptions).__name__}"
            )
        return _validate_cache_descriptions(
            descriptions,
            cache_path=self._cache_path,
        )

    @staticmethod
    def _is_legacy_cache(raw: dict[Any, Any]) -> bool:
        """A legacy cache is a flat ``{int|str: str}`` dict -- no
        ``model_id``/``sae_source``/``descriptions`` keys.

        The caller already early-returns on an empty dict, so this
        function only sees non-empty inputs. Returns ``False`` if any
        value is non-string (we'd rather fall through to the new-format
        path's structured DataLoadError than silently coerce)."""
        if "model_id" in raw or "sae_source" in raw or "descriptions" in raw:
            return False
        return all(isinstance(v, str) for v in raw.values())

    def _save_cache(self, updates: dict[int, str] | None = None) -> None:
        """Transactionally merge and persist newly fetched descriptions.

        ``self._cache`` contains committed data only. Pending ``updates`` are
        published to memory only after the locked read/merge/write, file fsync,
        atomic replacement, and parent-directory fsync all succeed.
        """
        with self._save_lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._cache_path.with_suffix(self._cache_path.suffix + ".lock")
            with lock_path.open("a+b") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    disk = self._read_cache_file()
                    with self._cache_lock:
                        snapshot = dict(self._cache)
                    pending = snapshot if updates is None else dict(updates)
                    memory_conflicts = {
                        feature_id: {
                            "committed_memory": snapshot[feature_id],
                            "pending": pending[feature_id],
                        }
                        for feature_id in snapshot.keys() & pending.keys()
                        if snapshot[feature_id] != pending[feature_id]
                    }
                    candidate = dict(snapshot)
                    candidate.update(pending)
                    disk_conflicts = {
                        feature_id: {
                            "disk": disk[feature_id],
                            "memory_or_pending": candidate[feature_id],
                        }
                        for feature_id in disk.keys() & candidate.keys()
                        if disk[feature_id] != candidate[feature_id]
                    }
                    conflicts = {**memory_conflicts, **disk_conflicts}
                    if conflicts:
                        # Disk is the last successfully committed shared view.
                        # Never leave rejected pending data visible through the
                        # public in-memory fast path after raising.
                        with self._cache_lock:
                            self._cache = dict(sorted(disk.items()))
                        raise DataLoadError(
                            "Concurrent Neuronpedia cache writers produced "
                            "different descriptions for the same feature IDs "
                            f"at {self._cache_path}: {conflicts}"
                        )
                    candidate.update(disk)
                    candidate = dict(sorted(candidate.items()))
                    candidate = _validate_cache_descriptions(
                        candidate,
                        cache_path=self._cache_path,
                    )
                    doc = {
                        "model_id": self._model_id,
                        "sae_source": self._sae_source,
                        "descriptions": candidate,
                    }
                    tmp_path = self._cache_path.parent / (
                        f".{self._cache_path.name}.tmp.{os.getpid()}."
                        f"{uuid.uuid4().hex}"
                    )
                    try:
                        with open(tmp_path, "x", encoding="utf-8") as f:
                            yaml.safe_dump(
                                doc,
                                f,
                                default_flow_style=False,
                                sort_keys=False,
                                allow_unicode=True,
                            )
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_path, self._cache_path)
                        directory_fd = os.open(
                            self._cache_path.parent,
                            os.O_RDONLY | os.O_DIRECTORY,
                        )
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except BaseException:
                        try:
                            tmp_path.unlink()
                        except OSError:
                            pass
                        raise
                    with self._cache_lock:
                        self._cache = candidate
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
