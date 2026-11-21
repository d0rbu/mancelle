"""Synchronous AxBench-style judge client with SQLite-backed caching
plus Stage 5 ``judge_all`` orchestration.

This module hosts both halves of the Task 33/34 pair:

- :class:`AxBenchJudgeClient` (Task 33): a synchronous OpenRouter wrapper
  with per-run SQLite caching, rate limiting, and parse-retry logic.
- :func:`judge_all` (Task 34): the stage-level orchestrator that iterates
  the Stage 4 generations directory-dataset, scores each row twice
  (concept relevance + fluency), and writes the results to the Stage 5
  judge_scores directory-dataset via
  :class:`~steering.eval.axbench_base.parquet_writer.ParquetAppendWriter`.

Wire protocol: OpenRouter chat-completions API. The client is deliberately
synchronous — judges run at stage boundaries, not in a hot generation loop,
so sync + per-call rate-limit + SQLite cache is the right tradeoff.

Cache schema (``judge_cache`` table):

    CREATE TABLE judge_cache (
        key TEXT PRIMARY KEY,
        score INTEGER,          -- nullable NULL = "we tried and gave up"
        raw_response TEXT NOT NULL,
        attempts INTEGER NOT NULL,
        timestamp REAL NOT NULL
    )

The cache stores the parsed ``score`` directly alongside the ``raw_response``
so re-runs can short-circuit both successful and exhausted calls without
re-parsing. Storing the raw response lets operators audit bad judgements
offline. Each row is keyed on SHA256(model || prompt) via
:func:`_make_cache_key` — parse-retry attempts use their own transient
``_a{n}`` suffixed keys that are NEVER written to the cache (only the
initial canonical key gets the final persisted result).

Retry policy:

- **Parse retries**: controlled by ``cfg.parse_max_retries``. A parse
  failure (``parse_axbench_rating`` → ``None``) triggers up to that many
  additional API calls. After exhaustion, the final row is stored with
  ``score=NULL`` and ``attempts = parse_max_retries + 1``.
- **API transport retries**: fixed budget of 5 extra attempts on 429 / 5xx
  responses or ``requests.RequestException`` transport failures. Each
  backoff is ``min(2**attempt, 60)`` seconds. Transport retries happen
  inside a single parse attempt — they do NOT count against the parse
  retry budget.
- **Non-retryable 4xx**: immediate ``AxBenchBaseError`` raise, no retry.

Stage 5 output schema (:data:`_JUDGE_SCORES_SCHEMA`):

Thirteen columns per judged generation: the five-field generation-key
fields for join-back, the two judge scores (concept relevance +
fluency) with parallel ``_attempts`` / ``_invalid`` markers, the raw
``generation_key`` and the judge model identifier for reproducibility.
Unparseable scores are stored as the ``-1`` sentinel with
``*_invalid = True`` — Stage 6 aggregation is responsible for
filtering/dropping them via the boolean column.

The generations directory-dataset read by :func:`judge_all` has a
tentative schema declared in :func:`_expected_generations_schema`. Task
38 (``generate_all``) owns the canonical emission of this schema; until
Task 38 lands, Stage 5 is test-only and synthesizes generations directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from loguru import logger

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig, JudgeConfig
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA as _GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge_parser import parse_axbench_rating
from steering.eval.axbench_base.openrouter import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    get_openrouter_session,
)
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.templates import (
    _safe_substitute,
    assert_no_unfilled_placeholders,
    load_template,
)
from steering.eval.axbench_base.types import AxBenchBaseError, Concept

__all__ = [
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "AxBenchJudgeClient",
    "RateLimiter",
    "judge_all",
]

# Transport-level retry budget. Applies per parse attempt to 429/5xx and
# requests.RequestException. NOT the parse-retry budget — that lives in
# cfg.parse_max_retries.
_TRANSPORT_MAX_RETRIES = 5

# Backoff cap: exponential 2**n seconds, clamped at this ceiling.
_BACKOFF_CAP_SECONDS = 60.0

_REQUEST_TIMEOUT_SECONDS = 60


def _make_cache_key(model: str, prompt: str) -> str:
    """SHA256 hex of ``"{model}\\n---\\n{prompt}"``.

    Two distinct judges (different models) on the same prompt produce
    different keys, matching the pattern in
    :class:`steering.eval.completion_judge.CompletionJudgeClient`.
    """
    payload = f"{model}\n---\n{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_retryable_judge_body_error(exc: AxBenchBaseError) -> bool:
    """Return whether a malformed 200 response should consume a parse retry."""
    message = str(exc)
    return (
        "OpenRouter returned null content" in message
        or "OpenRouter returned non-string content" in message
        or "malformed OpenRouter response" in message
    )


@dataclass(frozen=True)
class _JudgeCacheEntry:
    """One row from the judge SQLite cache, typed for safe access.

    Returned by :meth:`AxBenchJudgeClient._cache_get` instead of a
    ``dict[str, Any]`` with magic string keys, so field accesses are
    type-checked and IDE-discoverable. ``score`` may be ``None`` — the
    NULL sentinel marking a parse-exhausted judgement that future runs
    should short-circuit.
    """

    score: int | None
    raw_response: str
    attempts: int
    timestamp: float


class RateLimiter:
    """Thread-safe sliding-window requests-per-minute limiter.

    ``requests_per_minute <= 0`` disables throttling. Otherwise every
    :meth:`acquire` reserves one slot in the preceding 60-second window.
    The lock protects the timestamp deque so multiple worker threads
    cannot oversubscribe the limit.
    """

    _WINDOW_SECONDS = 60.0

    def __init__(self, requests_per_minute: int) -> None:
        self._requests_per_minute = requests_per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one request slot is available."""
        if self._requests_per_minute <= 0:
            return

        while True:
            sleep_for = 0.0
            with self._lock:
                now = time.monotonic()
                cutoff = now - self._WINDOW_SECONDS
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self._requests_per_minute:
                    self._calls.append(now)
                    return

                oldest = self._calls[0]
                sleep_for = max(0.0, self._WINDOW_SECONDS - (now - oldest))

            if sleep_for > 0.0:
                time.sleep(sleep_for)


class AxBenchJudgeClient:
    """Sync OpenRouter judge client with SQLite caching and rate limiting.

    Usage::

        with AxBenchJudgeClient(
            cfg, cache_path=output_paths.judge_cache()
        ) as j:
            score, attempts = j.score(prompt)

    The returned ``attempts`` is the number of API calls made on this
    particular invocation (1 on cache hit, up to
    ``cfg.parse_max_retries + 1`` on a cache miss). Tracking attempts
    separately lets the downstream ``judge_all`` orchestration surface
    billing/debug stats without re-deriving them from the cache.
    """

    def __init__(
        self,
        cfg: JudgeConfig,
        *,
        cache_path: Path,
    ) -> None:
        self._cfg = cfg
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise AxBenchBaseError(
                f"{OPENROUTER_API_KEY_ENV} not set in environment",
            )
        self._api_key = api_key
        self._rate_limiter = RateLimiter(cfg.requests_per_minute)
        self._conn_lock = threading.Lock()
        self._cache_key_locks: dict[str, threading.Lock] = {}
        self._cache_key_locks_lock = threading.Lock()

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                str(cache_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS judge_cache ("
                "  key TEXT PRIMARY KEY,"
                "  score INTEGER,"
                "  raw_response TEXT NOT NULL,"
                "  attempts INTEGER NOT NULL,"
                "  timestamp REAL NOT NULL"
                ")",
            )
            self._conn.commit()
        except sqlite3.DatabaseError as exc:
            raise AxBenchBaseError(
                f"Failed to open judge cache at {cache_path}: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the persistent SQLite connection."""
        with self._conn_lock:
            self._conn.close()

    def __enter__(self) -> AxBenchJudgeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Cache helpers (private)
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> _JudgeCacheEntry | None:
        """Return a typed :class:`_JudgeCacheEntry` or ``None`` on miss.

        Distinguishes between "row exists with NULL score" (returns an
        entry with ``score=None``) and "no row at all" (returns ``None``).
        Callers must check ``is None`` on the entry, NOT on
        ``entry.score``.
        """
        with profile_region("cache.judge.get"):
            with self._conn_lock:
                row = self._conn.execute(
                    "SELECT score, raw_response, attempts, timestamp"
                    " FROM judge_cache WHERE key = ?",
                    (key,),
                ).fetchone()
        if row is None:
            record_counter("cache.judge.miss")
            return None
        record_counter("cache.judge.hit")
        return _JudgeCacheEntry(
            score=row[0],  # may be None (NULL sentinel)
            raw_response=row[1],
            attempts=row[2],
            timestamp=row[3],
        )

    def _cache_put(
        self,
        *,
        key: str,
        score: int | None,
        raw_response: str,
        attempts: int,
    ) -> None:
        """Upsert a judge result. Commits immediately for crash safety."""
        with profile_region("cache.judge.set"):
            with self._conn_lock:
                if score is None:
                    self._conn.execute(
                        "INSERT INTO judge_cache"
                        " (key, score, raw_response, attempts, timestamp)"
                        " VALUES (?, ?, ?, ?, ?)"
                        " ON CONFLICT(key) DO UPDATE SET"
                        "   score = excluded.score,"
                        "   raw_response = excluded.raw_response,"
                        "   attempts = excluded.attempts,"
                        "   timestamp = excluded.timestamp"
                        " WHERE judge_cache.score IS NULL",
                        (key, score, raw_response, attempts, time.time()),
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO judge_cache"
                        " (key, score, raw_response, attempts, timestamp)"
                        " VALUES (?, ?, ?, ?, ?)"
                        " ON CONFLICT(key) DO UPDATE SET"
                        "   score = excluded.score,"
                        "   raw_response = excluded.raw_response,"
                        "   attempts = excluded.attempts,"
                        "   timestamp = excluded.timestamp",
                        (key, score, raw_response, attempts, time.time()),
                    )
                self._conn.commit()

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Reserve one HTTP request slot."""
        self._rate_limiter.acquire()

    def _cache_lock_for_key(self, key: str) -> threading.Lock:
        """Return the per-cache-key lock for concurrent miss dedupe."""
        with self._cache_key_locks_lock:
            lock = self._cache_key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._cache_key_locks[key] = lock
            return lock

    # ------------------------------------------------------------------
    # HTTP call (private)
    # ------------------------------------------------------------------

    def _backoff(self, attempt: int) -> None:
        """Sleep ``min(2**attempt, _BACKOFF_CAP_SECONDS)`` seconds."""
        delay = min(float(2**attempt), _BACKOFF_CAP_SECONDS)
        logger.warning(
            f"OpenRouter judge retry attempt {attempt + 1} after {delay}s backoff",
        )
        time.sleep(delay)

    def _call_api(self, prompt: str) -> str:
        """Send one chat-completions request and return the content string.

        Handles transport-level retries (429/5xx, transient network
        errors). Raises ``AxBenchBaseError`` on persistent failure, on
        non-retryable 4xx, or on malformed/empty responses.
        """
        payload = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            try:
                record_counter(
                    "api.openrouter.attempts",
                    client="judge",
                    model=self._cfg.model,
                )
                self._rate_limit()
                with profile_region(
                    "api.openrouter.chat_completion",
                    client="judge",
                    model=self._cfg.model,
                ):
                    resp = get_openrouter_session().post(
                        OPENROUTER_CHAT_COMPLETIONS_URL,
                        json=payload,
                        headers=headers,
                        timeout=(10, 60),
                    )
            except requests.RequestException as exc:
                if attempt >= _TRANSPORT_MAX_RETRIES:
                    raise AxBenchBaseError(
                        f"OpenRouter request failed after {attempt + 1} attempts: {exc}",
                    ) from exc
                self._backoff(attempt)
                attempt += 1
                continue

            status_code = resp.status_code
            if status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    if attempt >= _TRANSPORT_MAX_RETRIES:
                        raise AxBenchBaseError(
                            f"OpenRouter judge returned invalid JSON body after "
                            f"{attempt + 1} attempts: {exc}",
                        ) from exc
                    self._backoff(attempt)
                    attempt += 1
                    continue
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise AxBenchBaseError(
                        f"malformed OpenRouter response: {exc}",
                    ) from exc
                if content is None:
                    raise AxBenchBaseError(
                        "OpenRouter returned null content",
                    )
                if not isinstance(content, str):
                    raise AxBenchBaseError(
                        f"OpenRouter returned non-string content: {type(content).__name__}",
                    )
                return content

            if status_code in (429, 500, 502, 503, 504):
                if attempt >= _TRANSPORT_MAX_RETRIES:
                    raise AxBenchBaseError(
                        f"OpenRouter returned {status_code} after {attempt + 1} attempts",
                    )
                self._backoff(attempt)
                attempt += 1
                continue

            # Non-retryable 4xx (other than 429): raise immediately.
            raise AxBenchBaseError(
                f"OpenRouter returned non-retryable HTTP {status_code}",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _score_cache_miss_locked(
        self,
        *,
        key: str,
        prompt: str,
        retry_body_errors: bool,
    ) -> tuple[int | None, int]:
        """Call the API/parse loop and persist the result.

        The caller must hold the per-key lock and must already have
        double-checked the cache. Keeping this work in one helper lets
        ``score`` and ``score_many`` share the exact same retry/cache
        semantics.
        """
        max_total_attempts = self._cfg.parse_max_retries + 1
        last_raw: str = ""
        for attempt in range(max_total_attempts):
            try:
                raw = self._call_api(prompt)
            except AxBenchBaseError as exc:
                if (
                    not retry_body_errors
                    or not _is_retryable_judge_body_error(exc)
                ):
                    raise
                raw = f"[judge api body error] {exc}"
                logger.warning(
                    f"Judge API body failure "
                    f"(attempt {attempt + 1}/{max_total_attempts}): {exc}",
                )
            last_raw = raw
            parsed = parse_axbench_rating(raw)
            if parsed is not None:
                attempts_made = attempt + 1
                self._cache_put(
                    key=key,
                    score=parsed,
                    raw_response=raw,
                    attempts=attempts_made,
                )
                return parsed, attempts_made
            logger.warning(
                f"Judge parse failure (attempt {attempt + 1}/{max_total_attempts}): "
                f"raw={raw[:200]!r}",
            )

        # All parse attempts exhausted. Persist the NULL sentinel so future
        # runs short-circuit instead of paying the API again.
        self._cache_put(
            key=key,
            score=None,
            raw_response=last_raw,
            attempts=max_total_attempts,
        )
        return None, max_total_attempts

    def _score_one(
        self,
        prompt: str,
        *,
        retry_body_errors: bool = False,
    ) -> tuple[int | None, int]:
        """Thread-safe implementation for one prompt."""
        key = _make_cache_key(self._cfg.model, prompt)
        cached = self._cache_get(key)
        if cached is not None:
            return cached.score, 1

        key_lock = self._cache_lock_for_key(key)
        with key_lock:
            cached = self._cache_get(key)
            if cached is not None:
                return cached.score, 1
            return self._score_cache_miss_locked(
                key=key,
                prompt=prompt,
                retry_body_errors=retry_body_errors,
            )

    def score(self, prompt: str) -> tuple[int | None, int]:
        """Judge a single prompt, returning ``(score_or_None, attempts_made)``.

        Behaviour:

        1. Check the cache under ``_make_cache_key(cfg.model, prompt)``.
           On hit — whether ``score`` is an int or ``NULL`` — return the
           stored value with ``attempts=1``.
        2. On miss, acquire the prompt's cache-key lock, double-check the
           cache, then call the API up to ``cfg.parse_max_retries + 1``
           times. Each call that produces an unparseable response costs
           one parse attempt; a parseable response short-circuits the
           loop and writes the result to the cache under the ORIGINAL key
           (not any ``_a{n}``-suffixed transient key).
        3. If every parse attempt fails, persist a NULL-score row under
           the original key with the LAST raw response, then return
           ``(None, parse_max_retries + 1)``.

        Raises:
            AxBenchBaseError: Transport-level failure (persistent 5xx,
                network error, non-retryable 4xx client error, etc.) from
                any underlying ``_call_api`` invocation. Malformed 200-body
                judge responses consume parse retries, matching
                :meth:`score_many`.
        """
        return self._score_one(prompt, retry_body_errors=True)

    def score_many(self, prompts: Sequence[str]) -> list[tuple[int | None, int]]:
        """Judge prompts concurrently, preserving input order.

        Duplicate prompts are coalesced by cache key inside this call.
        The lowest input index owns the API/cache-miss work; later
        duplicates receive cache-hit semantics ``(score, 1)`` once the
        owner completes. This keeps batch results aligned with repeated
        sequential ``score`` calls while avoiding duplicate in-flight
        OpenRouter requests.
        """
        if not prompts:
            return []

        results: list[tuple[int | None, int] | None] = [None] * len(prompts)
        owner_by_key: dict[str, int] = {}
        duplicate_indices_by_owner: dict[int, list[int]] = {}

        for index, prompt in enumerate(prompts):
            key = _make_cache_key(self._cfg.model, prompt)
            owner_index = owner_by_key.get(key)
            if owner_index is None:
                owner_by_key[key] = index
            else:
                duplicate_indices_by_owner.setdefault(owner_index, []).append(index)

        work = [(index, prompts[index]) for index in sorted(owner_by_key.values())]
        max_workers = min(max(1, self._cfg.max_concurrent), len(work))

        executor = ThreadPoolExecutor(max_workers=max_workers)
        shutdown_started = False
        futures: dict[Future[tuple[int | None, int]], int] = {}
        try:
            for owner_index, prompt in work:
                context = copy_context()
                future = cast(
                    Future[tuple[int | None, int]],
                    executor.submit(
                        context.run,
                        self._score_one,
                        prompt,
                        retry_body_errors=True,
                    ),
                )
                futures[future] = owner_index

            for future in as_completed(futures):
                owner_index = futures[future]
                try:
                    owner_result = future.result()
                except BaseException:
                    for pending in futures:
                        if pending is not future:
                            pending.cancel()
                    shutdown_started = True
                    executor.shutdown(wait=True, cancel_futures=True)
                    raise

                results[owner_index] = owner_result
                for duplicate_index in duplicate_indices_by_owner.get(
                    owner_index,
                    [],
                ):
                    results[duplicate_index] = (owner_result[0], 1)
        finally:
            if not shutdown_started:
                executor.shutdown(wait=True, cancel_futures=False)

        # All futures completed without exception, so every slot must be
        # populated either by an owner result or by duplicate fill-in.
        completed_results: list[tuple[int | None, int]] = []
        for result in results:
            if result is None:
                raise AxBenchBaseError(
                    "internal error: score_many finished with missing result",
                )
            completed_results.append(result)
        return completed_results


# ===========================================================================
# Stage 5 orchestration — ``judge_all``
# ===========================================================================


# Sentinel stored in ``*_score`` columns when the judge failed to return a
# parseable rating. The parallel ``*_invalid`` boolean column is the
# authoritative "row failed to parse" marker; the sentinel keeps the
# ``int32`` column non-nullable for downstream pyarrow consumers.
_INVALID_SCORE_SENTINEL = -1

# Row buffer size before an incremental parquet flush. Small enough that a
# crash mid-Stage-5 doesn't lose much work; large enough that per-flush
# metadata overhead stays negligible.
_JUDGE_WRITE_BATCH_SIZE = 100

_JUDGE_SCORES_SCHEMA = pa.schema(
    [
        ("generation_key", pa.string()),
        ("concept_id", pa.int32()),
        ("method_name", pa.string()),
        ("mode", pa.string()),
        ("eval_seed", pa.int32()),
        ("coefficient", pa.string()),
        ("concept_score", pa.int32()),
        ("concept_attempts", pa.int32()),
        ("concept_invalid", pa.bool_()),
        ("fluency_score", pa.int32()),
        ("fluency_attempts", pa.int32()),
        ("fluency_invalid", pa.bool_()),
        ("judge_model", pa.string()),
    ],
)

_JUDGE_RESUME_MANIFEST_NAME = "_resume_manifest.json"

_STAGE_BATCH_PATTERN = re.compile(r"^batch_(\d+)\.parquet$")


def _judge_resume_manifest_path(judge_dir: Path) -> Path:
    return judge_dir / _JUDGE_RESUME_MANIFEST_NAME


def _judge_resume_fingerprint(
    *,
    judge_model: str,
    concept_template: str,
    fluency_template: str,
    concepts: list[Concept],
) -> str:
    """Fingerprint every input that changes Stage 5 judge prompts."""
    payload = {
        "judge_model": judge_model,
        "concept_template": concept_template,
        "fluency_template": fluency_template,
        "concepts": [
            {
                "concept_id": concept.concept_id,
                "description": concept.description,
            }
            for concept in sorted(concepts, key=lambda item: item.concept_id)
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_judge_resume_manifest(
    judge_dir: Path,
    *,
    judge_model: str,
    prompt_fingerprint: str,
) -> None:
    judge_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "judge_model": judge_model,
        "prompt_fingerprint": prompt_fingerprint,
    }
    _judge_resume_manifest_path(judge_dir).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_judge_resume_manifest(
    judge_dir: Path,
    *,
    judge_model: str,
    prompt_fingerprint: str,
    existing_rows: int,
) -> None:
    """Validate or create the Stage 5 resume manifest.

    The parquet schema intentionally stays focused on score rows. This
    sidecar pins the prompt-affecting inputs that make a skipped
    ``generation_key`` safe to reuse.
    """
    manifest_path = _judge_resume_manifest_path(judge_dir)
    if manifest_path.exists() and existing_rows == 0:
        _write_judge_resume_manifest(
            judge_dir,
            judge_model=judge_model,
            prompt_fingerprint=prompt_fingerprint,
        )
        return
    if not manifest_path.exists():
        if existing_rows:
            logger.warning(
                f"Existing judge scores at {judge_dir} have no "
                f"{_JUDGE_RESUME_MANIFEST_NAME}; treating them as a legacy "
                "resume and writing a manifest for future checks.",
            )
        _write_judge_resume_manifest(
            judge_dir,
            judge_model=judge_model,
            prompt_fingerprint=prompt_fingerprint,
        )
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AxBenchBaseError(
            f"Could not parse Stage 5 resume manifest at {manifest_path}: {exc}"
        ) from exc

    existing_model = manifest.get("judge_model")
    existing_fingerprint = manifest.get("prompt_fingerprint")
    if (
        existing_model != judge_model
        or existing_fingerprint != prompt_fingerprint
    ):
        raise AxBenchBaseError(
            f"Existing judge scores at {judge_dir} have resume manifest "
            f"judge_model={existing_model!r}, "
            f"prompt_fingerprint={existing_fingerprint!r}, but the current "
            f"run has judge_model={judge_model!r}, "
            f"prompt_fingerprint={prompt_fingerprint!r}. Use a new run_tag "
            "or delete the stale judge_scores directory before resuming."
        )


def _expected_generations_schema() -> pa.Schema:
    """Return the canonical Stage 4 generations schema read by Stage 5.

    Task 38 (``generate_all``) owns the canonical schema and this
    helper stays byte-identical to it. The schema has eleven columns —
    the seven that ``judge_all`` dereferences (``generation_key``,
    ``concept_id``, ``method_name``, ``mode``, ``eval_seed``,
    ``coefficient``, ``generation_text``) plus four provenance columns
    (``target_model``, ``prefix_text``, ``token_count``,
    ``stop_reason``) that later debugging stages join back on the
    ``generation_key`` field.

    The round-trip through ``load_existing`` enforces strict equality,
    so any drift between this schema and Stage 4's
    ``GENERATIONS_SCHEMA`` surfaces as a loud
    :class:`AxBenchBaseError` on the next stage-4-to-5 boundary.

    Column types are pinned to what pyarrow's ``Table.from_pylist`` emits
    naturally for Python values ``int``/``float``/``str``: ``int32`` for
    integer fields, ``string`` for the coefficient-serialized-as-string
    (a coefficient can be ``None`` for baselines that take no
    magnitude, or a Python int/float; storing the string representation
    sidesteps nullable-int awkwardness).
    """
    return _GENERATIONS_SCHEMA


def _scan_stage_batch_paths(out_dir: Path) -> list[Path]:
    """Return sorted ``batch_NNNNNN.parquet`` paths under ``out_dir``."""
    if not out_dir.exists():
        return []

    indexed_paths: list[tuple[int, Path]] = []
    for entry in out_dir.iterdir():
        if not entry.is_file():
            continue
        match = _STAGE_BATCH_PATTERN.match(entry.name)
        if match is None:
            continue
        indexed_paths.append((int(match.group(1)), entry))
    indexed_paths.sort(key=lambda item: item[0])
    return [path for _, path in indexed_paths]


def _iter_generation_row_chunks(
    gen_dir: Path,
    *,
    schema: pa.Schema,
    chunk_size: int,
) -> Iterator[list[dict[str, Any]]]:
    """Yield Stage 4 generation rows without materializing the full table."""
    for path in _scan_stage_batch_paths(gen_dir):
        parquet_file = pq.ParquetFile(path)
        file_schema = parquet_file.schema_arrow
        if not file_schema.equals(schema):
            raise AxBenchBaseError(
                f"schema mismatch in {path}: "
                f"expected {schema}, got {file_schema}"
            )

        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            if batch.num_rows == 0:
                continue
            table = pa.Table.from_batches([batch], schema=schema)
            yield table.to_pylist()


def _count_generation_rows_to_judge(
    gen_dir: Path,
    *,
    schema: pa.Schema,
    already_judged: set[str],
    chunk_size: int,
) -> tuple[int, int]:
    """Return ``(total_rows, pending_rows)`` with bounded Python memory."""
    n_total = 0
    n_pending = 0
    for chunk_rows in _iter_generation_row_chunks(
        gen_dir,
        schema=schema,
        chunk_size=chunk_size,
    ):
        n_total += len(chunk_rows)
        n_pending += sum(
            1
            for gen_row in chunk_rows
            if gen_row["generation_key"] not in already_judged
        )
    return n_total, n_pending


def _count_true_column(table: pa.Table, column_name: str) -> int:
    """Count truthy bool values in a pyarrow table column."""
    if table.num_rows == 0:
        return 0
    return sum(bool(value) for value in table.column(column_name).to_pylist())


def _log_invalid_rates(
    cfg: AxBenchBaseConfig,
    *,
    judge_dir: Path,
    n_rows: int,
    n_concept_invalid: int,
    n_fluency_invalid: int,
    n_new_rows: int,
) -> None:
    """Report Stage 5 invalid rates across all on-disk rows for this model."""
    if n_rows <= 0:
        return
    concept_rate = n_concept_invalid / n_rows
    fluency_rate = n_fluency_invalid / n_rows
    combined_rate = max(concept_rate, fluency_rate)
    if combined_rate > cfg.judge.invalid_drop_threshold:
        logger.error(
            f"Stage 5 invalid rate exceeds drop threshold: "
            f"concept={concept_rate:.3f} fluency={fluency_rate:.3f} "
            f"(drop_threshold={cfg.judge.invalid_drop_threshold:.3f}). "
            "Stage 6 aggregation will drop the affected rows.",
        )
    elif combined_rate > cfg.judge.invalid_warning_threshold:
        logger.warning(
            f"Stage 5 invalid rate exceeds warning threshold: "
            f"concept={concept_rate:.3f} fluency={fluency_rate:.3f} "
            f"(warning_threshold="
            f"{cfg.judge.invalid_warning_threshold:.3f}).",
        )
    logger.info(
        f"judge_all wrote {n_new_rows} new rows to {judge_dir} "
        f"({n_rows} total rows for judge_model={cfg.judge.model!r}; "
        f"concept_invalid={n_concept_invalid}, "
        f"fluency_invalid={n_fluency_invalid})",
    )


def judge_all(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept],
    client: AxBenchJudgeClient,
) -> Path:
    """Run Stage 5 judge scoring for every Stage 4 generation row.

    Iterates the generations directory-dataset at
    :func:`paths.generations_dir`, calls the judge client twice per row
    (concept relevance + fluency), and appends the scored rows to the
    Stage 5 directory-dataset at
    :meth:`AxBenchBasePaths.judge_scores_dir` via
    :class:`ParquetAppendWriter`. The output path is a directory containing
    ``batch_NNNNNN.parquet`` files.

    Resume
    ------
    On every call, the writer scans the output directory for existing
    batches via :func:`load_existing` and builds a set of already-judged
    ``generation_key`` values. Any generations row whose key is in that
    set is skipped. Combined with the ``ParquetAppendWriter``'s
    ``batch_NNNNNN.parquet`` cursor, a crash mid-Stage-5 survives cleanly
    — the next call picks up exactly where the previous left off.

    Client lifecycle
    ----------------
    The caller supplies the run-owned :class:`AxBenchJudgeClient`. The client
    is used as a context manager so the SQLite connection is closed on the way
    out.

    Invalid tracking
    ----------------
    A judge call that can't produce a parseable rating returns
    ``(None, attempts)``; we translate that to the
    ``-1`` sentinel in the corresponding ``*_score`` column and set the
    parallel ``*_invalid`` boolean. Stage 6 aggregation is responsible
    for excluding invalid rows. After the run, if the per-column invalid
    rate crosses ``cfg.judge.invalid_warning_threshold`` we log a
    WARNING; if it crosses ``cfg.judge.invalid_drop_threshold`` we log
    an ERROR. Neither raises — Stage 6 owns the decision to drop.

    Parameters
    ----------
    cfg:
        Pipeline config. ``cfg.run_tag`` must be set.
    concepts:
        The concept list from Stage 0a (:func:`load_and_validate_concepts`).
        ``judge_all`` builds a lookup keyed on ``concept_id`` to resolve
        a generation row's concept description into the judge prompt.
        A generation row whose ``concept_id`` is absent from this list
        is a Stage 0 / Stage 4 contract violation and raises
        :class:`AxBenchBaseError`.
    client:
        Run-owned SQLite-backed judge client. The Stage 5 CLI obtains this
        from the runtime state's ``judge_client()`` accessor.

    Returns
    -------
    Path
        The judge_scores directory-dataset root (the return value of
        :func:`paths.judge_scores_dir`).

    Raises
    ------
    AxBenchBaseError
        If ``cfg.run_tag`` is unset, the generations directory does not
        exist, or a generations row points at a ``concept_id`` that is
        not present in the caller-supplied ``concepts`` list.
    AssertionError
        If a built prompt still contains an unfilled ``{{KEY}}``
        placeholder after substitution — a defensive guard against
        template drift.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling judge_all",
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)

    gen_dir = output_paths.generations_dir()
    if not gen_dir.exists():
        raise AxBenchBaseError(
            f"generations directory does not exist: {gen_dir}. Run Stage 4 (generate_all) first.",
        )

    judge_dir = output_paths.judge_scores_dir()
    generations_schema = _expected_generations_schema()

    concept_lookup: dict[int, str] = {
        concept.concept_id: concept.description for concept in concepts
    }

    concept_template = load_template("judge_concept_relevance.txt")
    fluency_template = load_template("judge_fluency.txt")
    prompt_fingerprint = _judge_resume_fingerprint(
        judge_model=cfg.judge.model,
        concept_template=concept_template,
        fluency_template=fluency_template,
        concepts=concepts,
    )

    # Build the set of already-judged generation_keys for resume.
    with profile_region("stage5.load_existing_judge_scores", path=judge_dir):
        existing = load_existing(judge_dir, schema=_JUDGE_SCORES_SCHEMA)
    existing_models = (
        set(existing.column("judge_model").to_pylist())
        if existing.num_rows else set()
    )
    stale_models = existing_models - {cfg.judge.model}
    if stale_models:
        raise AxBenchBaseError(
            f"Existing judge scores at {judge_dir} were produced with "
            f"judge_model(s) {sorted(stale_models)!r}, but the current "
            f"cfg.judge.model is {cfg.judge.model!r}. Use a new run_tag or "
            "delete the existing judge_scores directory before rejudging "
            "with a different model."
        )
    _ensure_judge_resume_manifest(
        judge_dir,
        judge_model=cfg.judge.model,
        prompt_fingerprint=prompt_fingerprint,
        existing_rows=existing.num_rows,
    )
    already_judged: set[str] = set(
        existing.column("generation_key").to_pylist(),
    )
    n_existing_rows = existing.num_rows
    n_existing_concept_invalid = _count_true_column(existing, "concept_invalid")
    n_existing_fluency_invalid = _count_true_column(existing, "fluency_invalid")
    if already_judged:
        logger.info(
            f"judge_all resume: {len(already_judged)} rows already judged "
            f"at {judge_dir}; skipping them",
        )

    chunk_size = max(1, cfg.judge.max_concurrent * 4)
    with profile_region("stage5.count_generation_rows", path=gen_dir):
        n_total_in, n_to_judge = _count_generation_rows_to_judge(
            gen_dir,
            schema=generations_schema,
            already_judged=already_judged,
            chunk_size=chunk_size,
        )

    if n_to_judge == 0:
        judge_dir.mkdir(parents=True, exist_ok=True)
        with client:
            pass
        _log_invalid_rates(
            cfg,
            judge_dir=judge_dir,
            n_rows=n_existing_rows,
            n_concept_invalid=n_existing_concept_invalid,
            n_fluency_invalid=n_existing_fluency_invalid,
            n_new_rows=0,
        )
        logger.info(
            f"[Stage 5] judge_all: no rows to judge "
            f"({len(already_judged)} resume-skips of {n_total_in} total)"
        )
        return judge_dir

    n_rows = 0
    n_concept_invalid = 0
    n_fluency_invalid = 0

    if n_to_judge:
        logger.info(
            f"[Stage 5] judge_all: {n_to_judge} rows to judge "
            f"({len(already_judged)} resume-skips of {n_total_in} total)"
        )
    stage_started_at = time.monotonic()
    last_heartbeat = stage_started_at

    with (
        client as active_client,
        ParquetAppendWriter(
            out_dir=judge_dir,
            schema=_JUDGE_SCORES_SCHEMA,
            batch_size=_JUDGE_WRITE_BATCH_SIZE,
        ) as writer,
    ):
        for generation_chunk_rows in _iter_generation_row_chunks(
            gen_dir,
            schema=generations_schema,
            chunk_size=chunk_size,
        ):
            chunk_rows = [
                gen_row
                for gen_row in generation_chunk_rows
                if gen_row["generation_key"] not in already_judged
            ]
            if not chunk_rows:
                continue

            prompts: list[str] = []

            for gen_row in chunk_rows:
                concept_id = gen_row["concept_id"]
                if concept_id not in concept_lookup:
                    raise AxBenchBaseError(
                        f"generations row references concept_id={concept_id} "
                        f"but that id is not in the supplied concepts list "
                        f"(known ids: {sorted(concept_lookup.keys())}). "
                        "This indicates a Stage 0 / Stage 4 contract "
                        "violation.",
                    )
                concept_description = concept_lookup[concept_id]
                generation_text = gen_row["generation_text"]

                concept_prompt = _safe_substitute(
                    concept_template,
                    CONCEPT=concept_description,
                    SENTENCE=generation_text,
                )
                assert_no_unfilled_placeholders(
                    concept_prompt, label="concept-relevance",
                )

                fluency_prompt = _safe_substitute(
                    fluency_template,
                    SENTENCE=generation_text,
                )
                assert_no_unfilled_placeholders(
                    fluency_prompt, label="fluency",
                )

                prompts.extend([concept_prompt, fluency_prompt])

            with profile_region(
                "stage5.score_prompts",
                rows=len(prompts),
                generation_rows=len(chunk_rows),
            ):
                scored_prompts = active_client.score_many(prompts)
            if len(scored_prompts) != 2 * len(chunk_rows):
                raise AxBenchBaseError(
                    "judge client returned the wrong number of scores: "
                    f"expected {2 * len(chunk_rows)}, got {len(scored_prompts)}",
                )

            for chunk_index, gen_row in enumerate(chunk_rows):
                generation_key = gen_row["generation_key"]
                concept_id = gen_row["concept_id"]
                concept_score, concept_attempts = scored_prompts[2 * chunk_index]
                fluency_score, fluency_attempts = scored_prompts[
                    2 * chunk_index + 1
                ]

                # Promote Optional[int] → int with the -1 sentinel when the
                # judge couldn't parse a rating. The parallel _invalid
                # boolean is the authoritative "row failed" signal; the
                # sentinel just keeps the int32 column non-nullable.
                if concept_score is None:
                    concept_score_out = _INVALID_SCORE_SENTINEL
                    concept_invalid = True
                    n_concept_invalid += 1
                else:
                    concept_score_out = concept_score
                    concept_invalid = False
                if fluency_score is None:
                    fluency_score_out = _INVALID_SCORE_SENTINEL
                    fluency_invalid = True
                    n_fluency_invalid += 1
                else:
                    fluency_score_out = fluency_score
                    fluency_invalid = False

                with profile_region("stage5.writer_append"):
                    writer.append(
                        {
                            "generation_key": generation_key,
                            "concept_id": concept_id,
                            "method_name": gen_row["method_name"],
                            "mode": gen_row["mode"],
                            "eval_seed": gen_row["eval_seed"],
                            "coefficient": gen_row["coefficient"],
                            "concept_score": concept_score_out,
                            "concept_attempts": concept_attempts,
                            "concept_invalid": concept_invalid,
                            "fluency_score": fluency_score_out,
                            "fluency_attempts": fluency_attempts,
                            "fluency_invalid": fluency_invalid,
                            "judge_model": cfg.judge.model,
                        },
                    )
                n_rows += 1

                now = time.monotonic()
                if (now - last_heartbeat) >= 30.0:
                    elapsed = now - stage_started_at
                    pct = 100.0 * n_rows / n_to_judge if n_to_judge else 0.0
                    rate = n_rows / elapsed if elapsed > 0 else 0.0
                    remaining = max(0, n_to_judge - n_rows)
                    eta = remaining / rate if rate > 0 else float("inf")
                    invalid_pct = (
                        100.0 * (n_concept_invalid + n_fluency_invalid)
                        / (2 * n_rows)
                        if n_rows else 0.0
                    )
                    logger.info(
                        f"[Stage 5] judge progress: "
                        f"{n_rows}/{n_to_judge} ({pct:.0f}%), "
                        f"invalid={invalid_pct:.1f}%, "
                        f"rate={rate:.2f} rows/s, "
                        f"elapsed={elapsed:.0f}s, "
                        f"eta≈{eta:.0f}s"
                    )
                    last_heartbeat = now

    # Report post-run invalid rates. We exercise the check whether or
    # not the client was injected — the thresholds live on cfg.judge.
    _log_invalid_rates(
        cfg,
        judge_dir=judge_dir,
        n_rows=n_existing_rows + n_rows,
        n_concept_invalid=n_existing_concept_invalid + n_concept_invalid,
        n_fluency_invalid=n_existing_fluency_invalid + n_fluency_invalid,
        n_new_rows=n_rows,
    )

    return judge_dir
