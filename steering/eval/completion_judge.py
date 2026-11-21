"""LLM judge for completion steering eval — rates coherence and feature alignment."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import matplotlib
import requests as _requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml as _yaml
from loguru import logger

from errors import ParseError


@dataclass(frozen=True)
class JudgementScore:
    coherence: int
    feature_alignment: int


@dataclass(frozen=True)
class CompletionJudgeItem:
    """One typed request in a completion-judging batch."""

    context: str
    completion: str
    feature_description: str
    ground_truth: str
    include_ground_truth: bool = True


_MAX_CONTEXT_CHARS = 16384
_MAX_RETRIES = 3
_MIN_SCORE = 1
_MAX_SCORE = 5

_JUDGE_SYSTEM_PROMPT_WITH_GT = """\
You are an expert evaluator assessing text completions. You will be given:
1. A context (preceding text)
2. A completion (generated text following the context)
3. A feature description (a trait the completion may exhibit)
4. A ground truth (the actual continuation for reference)

Rate the completion on two dimensions, each from 1 (worst) to 5 (best):

- **coherence**: How well does the completion follow from the context as a natural, fluent continuation? (1=incoherent/nonsensical, 5=perfectly fluent and logical)
- **feature_alignment**: How strongly does the completion exhibit the trait described by the feature? (1=not at all, 5=very strongly)

Respond with ONLY two lines in this exact format:
coherence: N
feature_alignment: N

Where N is an integer from 1 to 5. Do not include any other text."""

_JUDGE_SYSTEM_PROMPT_NO_GT = """\
You are an expert evaluator assessing text completions. You will be given:
1. A context (preceding text)
2. A completion (generated text following the context)
3. A feature description (a trait the completion may exhibit)

Rate the completion on two dimensions, each from 1 (worst) to 5 (best):

- **coherence**: How well does the completion follow from the context as a natural, fluent continuation? (1=incoherent/nonsensical, 5=perfectly fluent and logical)
- **feature_alignment**: How strongly does the completion exhibit the trait described by the feature? (1=not at all, 5=very strongly)

Respond with ONLY two lines in this exact format:
coherence: N
feature_alignment: N

Where N is an integer from 1 to 5. Do not include any other text."""

_JUDGE_USER_TEMPLATE = """\
## Context
{context}

## Completion
{completion}

## Feature Description
{feature_description}"""

_JUDGE_GT_SECTION = """

## Ground Truth (for reference)
{ground_truth}"""


def build_judge_prompt(
    context: str,
    completion: str,
    feature_description: str,
    ground_truth: str,
    include_ground_truth: bool = True,
) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for the judge.

    Returns:
        Tuple of (system_prompt, user_prompt) to be sent as separate messages.
    """
    if len(context) > _MAX_CONTEXT_CHARS:
        context = "..." + context[-_MAX_CONTEXT_CHARS:]

    user_section = _JUDGE_USER_TEMPLATE.format(
        context=context,
        completion=completion,
        feature_description=feature_description,
    )
    if include_ground_truth:
        user_section += _JUDGE_GT_SECTION.format(ground_truth=ground_truth)
        system_prompt = _JUDGE_SYSTEM_PROMPT_WITH_GT
    else:
        system_prompt = _JUDGE_SYSTEM_PROMPT_NO_GT

    return system_prompt, user_section


_SCORE_PATTERN = re.compile(
    r"coherence:\s*(\d+)\s*\n\s*feature_alignment:\s*(\d+)",
    re.IGNORECASE,
)


def parse_judge_response(response: str) -> JudgementScore:
    """Parse a judge response string into a JudgementScore.

    Raises:
        ParseError: If the response does not contain the expected score pattern.
    """
    match = _SCORE_PATTERN.search(response)
    if not match:
        raise ParseError(f"Judge response missing score pattern: {response[:200]}")

    coherence = max(_MIN_SCORE, min(_MAX_SCORE, int(match.group(1))))
    feature_alignment = max(_MIN_SCORE, min(_MAX_SCORE, int(match.group(2))))

    return JudgementScore(coherence=coherence, feature_alignment=feature_alignment)


def save_scores_yaml(scores_serializable: dict, scores_path: Path | str) -> None:
    """Atomically save scores dict to YAML via temp file + os.replace."""
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=scores_path.parent, suffix=".yaml.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            _yaml.safe_dump(scores_serializable, f, default_flow_style=False)
        os.replace(tmp_path, scores_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def compute_dose_response(
    scores_by_bucket: dict[int, list[JudgementScore]],
) -> dict[str, list[float]]:
    buckets = sorted(scores_by_bucket.keys())
    means: list[float] = []
    stderrs: list[float] = []
    coherence_means: list[float] = []

    for b in buckets:
        scores = scores_by_bucket[b]
        if not scores:
            means.append(float("nan"))
            stderrs.append(float("nan"))
            coherence_means.append(float("nan"))
            continue

        fa_vals = [s.feature_alignment for s in scores]
        co_vals = [s.coherence for s in scores]
        mean = sum(fa_vals) / len(fa_vals)
        means.append(mean)

        if len(fa_vals) >= 2:
            variance = sum((v - mean) ** 2 for v in fa_vals) / (len(fa_vals) - 1)
            stderrs.append(math.sqrt(variance / len(fa_vals)))
        else:
            stderrs.append(float("nan"))

        coherence_means.append(sum(co_vals) / len(co_vals))

    return {
        "bucket_means": means,
        "bucket_stderrs": stderrs,
        "coherence_means": coherence_means,
    }


def plot_dose_response(
    dose_data: dict[str, list[float]],
    feature_id: int,
    feature_description: str,
    output_path: Path | str,
    real_baseline: float | None = None,
) -> None:
    """Plot bucket index vs feature_alignment score with error bars (insertion mode)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_means = dose_data["bucket_means"]
    bucket_stderrs = dose_data["bucket_stderrs"]
    x = list(range(len(bucket_means)))

    fig, ax = plt.subplots()
    ax.errorbar(
        x,
        bucket_means,
        yerr=bucket_stderrs,
        marker="o",
        capsize=4,
        label="Feature alignment",
    )
    if real_baseline is not None:
        ax.axhline(
            real_baseline,
            linestyle=":",
            color="blue",
            label="Real conditioning baseline",
        )

    ax.set_xlabel("Bucket index")
    ax.set_ylabel("Feature alignment (1–5)")
    ax.set_ylim(1, 5)
    title_desc = feature_description[:60]
    ax.set_title(f"Feature {feature_id}: {title_desc}")
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved dose-response plot to {output_path}")


def plot_ablation_comparison(
    real_mean: float,
    ablated_mean: float,
    feature_id: int,
    feature_description: str,
    output_path: Path | str,
    real_stderr: float = 0.0,
    ablated_stderr: float = 0.0,
) -> None:
    """Bar chart comparing real vs ablated feature alignment (ablation mode)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Real", "Ablated"]
    means = [real_mean, ablated_mean]
    stderrs = [real_stderr, ablated_stderr]

    fig, ax = plt.subplots()
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stderrs, capsize=4, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Feature alignment (1–5)")
    ax.set_ylim(0, 5)
    title_desc = feature_description[:60]
    ax.set_title(f"Feature {feature_id}: {title_desc}")

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved ablation comparison plot to {output_path}")


# ---------------------------------------------------------------------------
# API-backed judge client with caching and rate limiting
# ---------------------------------------------------------------------------


@dataclass
class CompletionJudgeClient:
    """OpenAI-compatible API client for judging completion quality.

    Sends (context, completion, feature_description, ground_truth) tuples
    to a chat-completions endpoint, caches raw responses in SQLite, and
    respects a configurable requests-per-minute rate limit.
    """

    model: str = "openai/gpt-5-nano"
    base_url: str = ""
    api_key: str = ""
    cache_path: str = ""
    requests_per_minute: int = 60
    concurrency: int = 10
    _last_call_time: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not self.api_key:
            self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.cache_path:
            self.cache_path = str(Path.home() / ".cache" / "completion_judge.db")
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()
        self._cache_async_lock: asyncio.Lock = asyncio.Lock()

    def _init_cache(self) -> None:
        """Create the cache table if absent. Uses a fresh, immediately-closed connection."""
        conn = sqlite3.connect(self.cache_path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()
        finally:
            conn.close()

    @contextlib.contextmanager
    def _cache_connection(self) -> Iterator[sqlite3.Connection]:
        """Open a per-call sqlite connection for thread-safe cache access.

        The sqlite3 default ``check_same_thread=True`` only permits connection
        use on the creating thread, which breaks any ``ThreadPoolExecutor``
        dispatch or event-loop-on-a-new-thread pattern. Opening a fresh
        connection per call is the canonical fix (also used by
        ``harmfulness_judge.JudgeClient``).
        """
        conn = sqlite3.connect(self.cache_path)
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Retained for backward compat — no persistent connection to close."""
        return None

    def _cache_key(self, prompt: str) -> str:
        payload = f"{self.model}\n---\n{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        with self._cache_connection() as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _cache_put(self, key: str, value: str) -> None:
        with self._cache_connection() as conn:
            try:
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                    (key, value),
                )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def _rate_limit(self) -> None:
        if self.requests_per_minute <= 0:
            return
        min_interval = 60.0 / self.requests_per_minute
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.monotonic()

    def _call_api(self, messages: list[dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        resp = _requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def judge(
        self,
        context: str,
        completion: str,
        feature_description: str,
        ground_truth: str,
        include_ground_truth: bool = True,
    ) -> JudgementScore:
        """Judge a single completion for coherence and feature alignment.

        Checks cache first.  On miss, calls the API with up to 3 retries.

        Returns:
            JudgementScore on success.

        Raises:
            ParseError: After all retries exhausted without a valid response.
        """
        system_prompt, user_prompt = build_judge_prompt(
            context,
            completion,
            feature_description,
            ground_truth,
            include_ground_truth=include_ground_truth,
        )
        cache_payload = f"{system_prompt}\n---\n{user_prompt}"
        key = self._cache_key(cache_payload)
        cached = self._cache_get(key)
        if cached is not None:
            try:
                return parse_judge_response(cached)
            except ParseError:
                logger.warning(
                    f"Corrupt cached judge response (key={key[:12]}…), re-fetching from API"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limit()
                raw = self._call_api(messages)
                result = parse_judge_response(raw)
                self._cache_put(key, raw)
                return result
            except ParseError as exc:
                logger.warning(
                    f"Invalid judge response (attempt {attempt + 1}/{_MAX_RETRIES}): {exc} | raw={raw[:200]!r}"
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(f"API error (attempt {attempt + 1}/{_MAX_RETRIES}): {exc}")
                last_exc = exc
                time.sleep(backoff)
                backoff *= 2.0

        raise ParseError(f"Failed to judge completion after {_MAX_RETRIES} attempts") from last_exc

    # ------------------------------------------------------------------
    # Async methods
    # ------------------------------------------------------------------

    async def _call_api_async(
        self, session: aiohttp.ClientSession, messages: list[dict[str, str]]
    ) -> str:
        """Async version of _call_api using aiohttp."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    async def judge_async(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        context: str,
        completion: str,
        feature_description: str,
        ground_truth: str,
        include_ground_truth: bool = True,
    ) -> JudgementScore:
        """Async version of judge() -- uses aiohttp and semaphore for concurrency.

        Acquires the semaphore before making the API call. Checks and writes
        cache under ``_cache_async_lock`` to prevent interleaving.

        Raises:
            ParseError: After all retries exhausted without a valid response.
        """
        system_prompt, user_prompt = build_judge_prompt(
            context,
            completion,
            feature_description,
            ground_truth,
            include_ground_truth=include_ground_truth,
        )
        cache_payload = f"{system_prompt}\n---\n{user_prompt}"
        key = self._cache_key(cache_payload)

        async with self._cache_async_lock:
            cached = self._cache_get(key)
        if cached is not None:
            try:
                return parse_judge_response(cached)
            except ParseError:
                logger.warning(
                    f"Corrupt cached judge response (key={key[:12]}…), re-fetching from API"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with semaphore:
                    raw = await self._call_api_async(session, messages)
                result = parse_judge_response(raw)
                async with self._cache_async_lock:
                    self._cache_put(key, raw)
                return result
            except ParseError as exc:
                logger.warning(
                    f"Invalid judge response (attempt {attempt + 1}/{_MAX_RETRIES}): {exc} | raw={raw[:200]!r}"
                )
                last_exc = exc
            except Exception as exc:
                logger.warning(f"Async API error (attempt {attempt + 1}/{_MAX_RETRIES}): {exc}")
                last_exc = exc
                await asyncio.sleep(backoff)
                backoff *= 2.0

        raise ParseError(f"Failed to judge completion after {_MAX_RETRIES} attempts") from last_exc

    async def judge_batch_async(
        self,
        items: list[CompletionJudgeItem],
        concurrency: int | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> list[JudgementScore]:
        """Judge a batch of completions concurrently using asyncio.

        Args:
            items: Typed completion-judging requests.
            concurrency: Max in-flight requests. Defaults to ``self.concurrency``.
            session: Optional aiohttp session. If None, creates and closes one.

        Returns:
            List of JudgementScore, one per item, in input order.

        Raises:
            ParseError: If any single item fails after all retries.
        """
        sem = asyncio.Semaphore(concurrency if concurrency is not None else self.concurrency)
        owns_session = session is None

        if owns_session:
            session = aiohttp.ClientSession()

        try:
            tasks = [
                self.judge_async(
                    session=session,
                    semaphore=sem,
                    context=item.context,
                    completion=item.completion,
                    feature_description=item.feature_description,
                    ground_truth=item.ground_truth,
                    include_ground_truth=item.include_ground_truth,
                )
                for item in items
            ]
            return list(await asyncio.gather(*tasks))
        finally:
            if owns_session:
                await session.close()
