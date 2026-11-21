"""Minimal synchronous LLM client for axbench_base QC/judge steps.

Task 14 (synth client) and Task 33 (async judge client) will add more
capable async clients. This file hosts the minimum-viable synchronous
implementation used by the lazy concept examples QC step and any other
stage-boundary LLM call that happens once per concept (not in a hot
generation loop).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from typing import Any

import requests
from beartype import beartype
from loguru import logger

from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.openrouter import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    get_openrouter_session,
)
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.refined_types import PositiveIntLike

__all__ = [
    "OPENROUTER_API_KEY_ENV",
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "LLMJudgeClient",
    "OpenRouterLLMClient",
]


# Fixed routes only: credentials are never shared or selected by fallback.
_PROVIDER_ROUTES = {
    "openrouter": (OPENROUTER_API_KEY_ENV, "OpenRouter"),
    "openai": ("OPENAI_API_KEY", "OpenAI"),
}


# Security — Wave 3B task 5. OpenRouter's 4xx responses (content filter,
# rate limit, validation) frequently echo the *request* body back to
# the caller. The request contains prompts that include user-supplied
# evaluation text, SAE concept descriptions, and generated completions.
# Echoing that body into an exception message or log record leaks
# sensitive content into loguru / stdlib / cloud log aggregators.
#
# This constant is the opt-in escape hatch for future debugging. When
# set to zero (the default, enforced by ``tests/test_openrouter_body_leak.py``),
# no body bytes are embedded in any log or exception. A future debug
# build MAY flip it to e.g. 200, but the default MUST remain 0.
_LOG_MAX_BODY_BYTES = 0
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Ceiling for one exponential-backoff sleep. Uncapped ``2 ** attempt`` reaches
# hours once a wall-clock retry budget lets the attempt counter run past ~10,
# so the sleep, not the budget, would decide when we give up.
_BACKOFF_CAP_SECONDS = 60.0
# Longest ``Retry-After`` we will honor. The header is provider-controlled;
# an absurd value must not park a worker indefinitely.
_MAX_RETRY_AFTER_SECONDS = 120.0


def _parse_retry_after(resp: Any) -> float | None:
    """Return the ``Retry-After`` delay in seconds, if the provider sent a sane one.

    Only the delta-seconds form is honored. The HTTP-date form is rare from
    JSON APIs and parsing it would add a timezone-dependent failure mode to a
    path whose fallback (exponential backoff) is already correct.
    """
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if seconds != seconds or seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_SECONDS)


def _openrouter_error_code(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, bool):
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str) and code.isdigit():
        return int(code)
    return None


class _SlidingWindowRateLimiter:
    """Thread-safe requests-per-minute limiter for concurrent API clients."""

    _WINDOW_SECONDS = 60.0

    def __init__(self, requests_per_minute: int) -> None:
        self._requests_per_minute = requests_per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, *, client: str, model: str, provider: str = "openrouter") -> None:
        """Block until one request slot is available."""
        if self._requests_per_minute <= 0:
            record_counter(
                f"api.{provider}.rate_limit_bypass",
                1,
                client=client,
                model=model,
            )
            return

        with profile_region(
            f"api.{provider}.rate_limit_wait",
            client=client,
            model=model,
            requests_per_minute=self._requests_per_minute,
        ):
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
                    record_counter(
                        f"api.{provider}.rate_limit_sleeps",
                        1,
                        client=client,
                        model=model,
                    )
                    record_counter(
                        f"api.{provider}.rate_limit_sleep_s",
                        sleep_for,
                        client=client,
                        model=model,
                    )
                    time.sleep(sleep_for)


class LLMJudgeClient:
    """Base class for LLM judge clients. Subclasses implement ``complete``.

    Task 14 and Task 33 will add async subclasses with batching and
    higher-concurrency behaviour. This synchronous base is deliberately
    minimal: it is used for stage-boundary QC that runs once per concept
    rather than in a hot generation loop.
    """

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str:
        raise NotImplementedError


class OpenRouterLLMClient(LLMJudgeClient):
    """Shared sync chat client with fixed OpenRouter and direct OpenAI routes.

    The historical class name and default OpenRouter behavior are retained.
    ``cfg.provider`` selects only the endpoint and its own credential variable;
    both providers use this same session, request shape and retry loop.

    Rate-limits to ``cfg.requests_per_minute`` by sleeping between calls.
    Retries on 429 and 5xx until *both* ``max_retries`` attempts and the
    ``cfg.transport_retry_budget_seconds`` cumulative-backoff budget are
    spent, honoring ``Retry-After`` and otherwise using jittered exponential
    backoff. Raises ``AxBenchBaseError`` on persistent failure or any
    non-retryable HTTP status.

    Optionally takes a ``response_cache`` (typically a
    :class:`steering.eval.axbench_base.synth.llm_response_cache.SynthResponseCache`)
    which, when paired with a per-call ``cache_key``, short-circuits
    identical requests to the stored response body. Caching happens
    ONLY after the successful response-shape validation, so failed
    responses are never persisted.
    """

    # The type of ``response_cache`` is declared as ``Any`` intentionally:
    # ``llm_judge_client.py`` is imported very early in the module graph,
    # and the concrete ``SynthResponseCache`` lives under ``synth/`` which
    # imports back into this module transitively via ``openrouter_client``.
    # Declaring the type as ``Any`` sidesteps the cycle without losing
    # runtime correctness — the method only calls ``.get`` / ``.set``.
    @beartype
    def __init__(
        self,
        cfg: LLMClientConfig,
        max_retries: int = 3,
        response_cache: Any = None,
        max_tokens: PositiveIntLike | None = None,
    ) -> None:
        if not isinstance(cfg.provider, str) or cfg.provider not in _PROVIDER_ROUTES:
            raise AxBenchBaseError(f"unsupported chat provider: {cfg.provider!r}")
        self._provider = cfg.provider
        key_env, self._provider_label = _PROVIDER_ROUTES[self._provider]
        # Resolve the existing OpenRouter constant at construction so its local
        # transport/budget test seam remains intact; snapshot the selected route.
        self._endpoint = (
            OPENROUTER_CHAT_COMPLETIONS_URL if self._provider == "openrouter"
            else "https://api.openai.com/v1/chat/completions"
        )
        self._cfg = cfg
        self._max_retries = max_retries
        self._retry_budget_seconds = float(cfg.transport_retry_budget_seconds)
        self._max_tokens = max_tokens
        self._rate_limiter = _SlidingWindowRateLimiter(cfg.requests_per_minute)
        api_key = os.environ.get(key_env)
        if not api_key:
            raise AxBenchBaseError(f"{key_env} not set in environment")
        self._api_key = api_key
        self._response_cache = response_cache

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant content.

        The retry loop always either returns (on HTTP 200 + well-formed
        response) or raises (on HTTP error, network error, or bad response
        shape), so it has no natural fallthrough.

        If a ``response_cache`` was configured at construction time and
        the caller passes a non-None ``cache_key``, the cache is checked
        first (returning the stored response on hit) and updated with
        the successful response on miss. Failed calls are NEVER cached.
        """
        if cache_key is not None and self._provider == "openai":
            cache_key = f"openai-chat-text-v1:{cache_key}"
        if cache_key is not None and self._response_cache is not None:
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                record_counter(
                    f"api.{self._provider}.cache_hits",
                    1,
                    client="llm",
                    model=self._cfg.model,
                )
                logger.debug(
                    f"LLM cache hit: key_prefix={cache_key[:16]}..."
                )
                return cached
            record_counter(
                f"api.{self._provider}.cache_misses",
                1,
                client="llm",
                model=self._cfg.model,
            )

        payload = {
            "model": self._cfg.model,
            "messages": messages,
        }
        # Preserve the exact historical request body for every existing
        # caller. Stage 0c sets this explicitly because provider defaults can
        # be shorter than its extraction-tokenizer continuation requirement.
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        result = self._complete_payload(payload, cache_key=cache_key)
        assert isinstance(result, str)
        return result

    def complete_response(
        self,
        messages: list[dict[str, str]],
        *,
        sampling: dict[str, float | int],
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        """Return the unmodified JSON response for a separately cached chat call.

        This additive path preserves finish/refusal information and malformed
        semantic responses for the caller's fixed-slot accounting. The legacy
        ``complete`` path keeps its text cache and strict content validation.
        Sampling keys are deliberately closed; no option can replace messages.
        """
        allowed = {"temperature", "top_p", "max_tokens", "seed"}
        if set(sampling) - allowed:
            raise AxBenchBaseError("unsupported structured-chat sampling option")
        namespace = "structured-chat-response-v1" if self._provider == "openrouter" else "openai-chat-response-v1"
        typed_key = None if cache_key is None else f"{namespace}:{cache_key}"
        if typed_key is not None and self._response_cache is not None:
            cached = self._response_cache.get(typed_key)
            if cached is not None:
                try:
                    value = json.loads(cached)
                except (TypeError, ValueError) as exc:
                    raise AxBenchBaseError("invalid structured-chat response cache") from exc
                if not isinstance(value, dict):
                    raise AxBenchBaseError("structured-chat response cache must be an object")
                return value
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            **sampling,
        }
        value = self._complete_payload(payload, cache_key=None, structured=True)
        assert isinstance(value, dict)
        try:
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise AxBenchBaseError("non-JSON structured-chat response") from exc
        if typed_key is not None and self._response_cache is not None:
            self._response_cache.set(typed_key, encoded)
        return value

    def _complete_payload(
        self,
        payload: dict[str, Any],
        *,
        cache_key: str | None,
        structured: bool = False,
    ) -> str | dict[str, Any]:
        """Shared identical-payload HTTP retry loop for both response surfaces."""
        self._rate_limit()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        # Cumulative seconds slept backing off. Every failure that is not a
        # definitive server verdict on this request draws on the wall-clock
        # budget: 429/5xx, an unparseable 200 body, and connection-level
        # errors are all shapes the same upstream congestion takes. Only a
        # non-retryable 4xx bypasses it — that is the server telling us the
        # request itself is wrong, which no amount of waiting fixes.
        backoff_elapsed = 0.0
        while True:
            try:
                record_counter(
                    f"api.{self._provider}.attempts",
                    1,
                    client="llm",
                    model=self._cfg.model,
                )
                with profile_region(
                    f"api.{self._provider}.chat_completion",
                    client="llm",
                    model=self._cfg.model,
                    transport_attempt=attempt,
                ):
                    resp = get_openrouter_session().post(
                        self._endpoint,
                        json=payload,
                        headers=headers,
                        timeout=(10, 60),
                    )
            except requests.RequestException as exc:
                # Connection resets and read timeouts cluster with upstream
                # capacity exhaustion, so they get the wall-clock budget too.
                # A genuinely broken endpoint (DNS, TLS) still fails — just
                # after the budget rather than after 7s, which is the right
                # trade when the alternative is discarding a run that has
                # been generating for over an hour.
                if not self._retry_allowed(attempt, backoff_elapsed):
                    raise AxBenchBaseError(
                        f"{self._provider_label} request failed after {attempt + 1} attempts "
                        f"and {backoff_elapsed:.1f}s of backoff: {exc}"
                    ) from exc
                backoff_elapsed += self._backoff(attempt)
                attempt += 1
                continue
            status_code = resp.status_code
            if status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    record_counter(
                        f"api.{self._provider}.invalid_responses",
                        1,
                        client="llm",
                        model=self._cfg.model,
                        reason="invalid_json",
                    )
                    # An HTTP 200 whose body will not parse is transport
                    # corruption, not a verdict on the request: the provider
                    # holds the connection open against a throttled upstream
                    # and the bytes arrive truncated. Observed in production
                    # as a keepalive preamble ("Expecting value: line 1013
                    # column 1") during the same burst that drives the 429s,
                    # so it draws on the same wall-clock budget.
                    if not self._retry_allowed(attempt, backoff_elapsed):
                        raise AxBenchBaseError(
                            f"{self._provider_label} returned invalid JSON body after "
                            f"{attempt + 1} attempts and "
                            f"{backoff_elapsed:.1f}s of backoff: {exc}"
                        ) from exc
                    backoff_elapsed += self._backoff(attempt)
                    attempt += 1
                    continue
                if structured:
                    embedded_error_code = _openrouter_error_code(data)
                    if embedded_error_code in _RETRYABLE_STATUS_CODES:
                        if not self._retry_allowed(attempt, backoff_elapsed):
                            logger.warning(
                                f"{self._provider_label} embedded retryable error: "
                                f"code={embedded_error_code} url={self._endpoint}"
                            )
                            raise AxBenchBaseError(
                                f"{self._provider_label} returned embedded retryable error "
                                f"{embedded_error_code} after {attempt + 1} attempts "
                                f"and {backoff_elapsed:.1f}s of backoff"
                            )
                        backoff_elapsed += self._backoff(attempt, _parse_retry_after(resp))
                        attempt += 1
                        continue
                    return data if isinstance(data, dict) else {"invalid_response": data}
                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    embedded_error_code = _openrouter_error_code(data)
                    if embedded_error_code in _RETRYABLE_STATUS_CODES:
                        if not self._retry_allowed(attempt, backoff_elapsed):
                            logger.warning(
                                f"{self._provider_label} embedded retryable error: "
                                f"code={embedded_error_code} "
                                f"url={self._endpoint}"
                            )
                            raise AxBenchBaseError(
                                f"{self._provider_label} returned embedded retryable error "
                                f"{embedded_error_code} after {attempt + 1} attempts "
                                f"and {backoff_elapsed:.1f}s of backoff"
                            ) from exc
                        backoff_elapsed += self._backoff(
                            attempt, _parse_retry_after(resp)
                        )
                        attempt += 1
                        continue
                    record_counter(
                        f"api.{self._provider}.invalid_responses",
                        1,
                        client="llm",
                        model=self._cfg.model,
                        reason="shape",
                    )
                    raise AxBenchBaseError(
                        f"Unexpected {self._provider_label} response shape: {data!r}"
                    ) from exc
                if content is None:
                    record_counter(
                        f"api.{self._provider}.invalid_responses",
                        1,
                        client="llm",
                        model=self._cfg.model,
                        reason="null_content",
                    )
                    raise AxBenchBaseError(
                        f"{self._provider_label} returned null content: {data!r}"
                    )
                if not isinstance(content, str):
                    record_counter(
                        f"api.{self._provider}.invalid_responses",
                        1,
                        client="llm",
                        model=self._cfg.model,
                        reason="non_string_content",
                    )
                    raise AxBenchBaseError(
                        f"{self._provider_label} returned non-string content "
                        f"({type(content).__name__}): {data!r}"
                    )
                # Cache the successful response before returning.
                if cache_key is not None and self._response_cache is not None:
                    self._response_cache.set(cache_key, content)
                return content
            if status_code in _RETRYABLE_STATUS_CODES:
                if not self._retry_allowed(attempt, backoff_elapsed):
                    # 5xx body is less sensitive than 4xx (no prompt
                    # echo on OpenRouter quota / validation errors)
                    # but the scrubbing rule is uniform: status + URL
                    # + length, never the body content. See
                    # ``_LOG_MAX_BODY_BYTES`` above.
                    #
                    # loguru uses f-strings / ``{key}`` — positional
                    # ``%s`` is NOT interpolated, so the status and
                    # URL MUST be embedded via f-string here.
                    body_len = len(resp.text or "")
                    logger.warning(
                        f"{self._provider_label} 5xx/429: status={status_code} "
                        f"url={self._endpoint} "
                        f"body_len={body_len}"
                    )
                    raise AxBenchBaseError(
                        f"{self._provider_label} returned {status_code} after "
                        f"{attempt + 1} attempts and "
                        f"{backoff_elapsed:.1f}s of backoff"
                    )
                backoff_elapsed += self._backoff(attempt, _parse_retry_after(resp))
                attempt += 1
                continue
            # Non-retryable 4xx. Scrub the body — it often echoes the
            # prompt (see ``_LOG_MAX_BODY_BYTES`` docstring above).
            body_len = len(resp.text or "")
            logger.warning(
                f"{self._provider_label} 4xx: status={status_code} "
                f"url={self._endpoint} "
                f"body_len={body_len}"
            )
            raise AxBenchBaseError(
                f"{self._provider_label} returned {status_code} "
                f"(body suppressed for privacy; see logs for status + url + length)"
            )

    def _rate_limit(self) -> None:
        self._rate_limiter.acquire(client="llm", model=self._cfg.model, provider=self._provider)

    def _retry_allowed(self, attempt: int, backoff_elapsed: float) -> bool:
        """Return whether one more attempt is permitted for a transient failure.

        Two independent budgets, either of which grants the retry: the
        attempt count (``max_retries``) and the cumulative-backoff wall clock
        (``transport_retry_budget_seconds``). The latter exists because
        provider capacity exhaustion clears on its own schedule; see the
        field's comment in ``config.py``.

        Applies to every failure except a non-retryable 4xx. Scoping it to
        429/5xx alone was not enough in practice: the same congestion also
        surfaces as truncated 200 bodies, which took down a 77-minute Stage 1b
        run that the 429 handling had just saved.
        """
        if attempt < self._max_retries:
            return True
        return backoff_elapsed < self._retry_budget_seconds

    def _backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Sleep before the next attempt and return the seconds actually slept.

        Honors a provider-supplied ``Retry-After``; otherwise capped
        exponential backoff. Jitter is applied so that concurrent workers
        which entered the same upstream outage do not resume in lockstep and
        immediately re-exhaust the capacity they are waiting on.
        """
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(float(2 ** min(attempt, 32)), _BACKOFF_CAP_SECONDS)
            delay *= 0.5 + random.random() * 0.5
        logger.debug(
            f"{self._provider_label} retry attempt {attempt + 1} after {delay:.2f}s backoff"
        )
        record_counter(
            f"api.{self._provider}.retry_sleeps",
            1,
            client="llm",
            model=self._cfg.model,
        )
        record_counter(
            f"api.{self._provider}.retry_sleep_s",
            delay,
            client="llm",
            model=self._cfg.model,
        )
        with profile_region(
            f"api.{self._provider}.retry_sleep",
            client="llm",
            model=self._cfg.model,
            attempt=attempt,
            delay_s=delay,
        ):
            time.sleep(delay)
        return delay
