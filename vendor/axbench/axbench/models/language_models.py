# Modified by nano-gpt-pretrain-steer: add bounded single-layer judge retries,
# an explicit SQLite cache/cache-only mode, and OpenRouter model support.
# See ../../VENDORED.md.

import asyncio
import hashlib
import json
import logging
import math
import pickle
import string
import uuid
from pathlib import Path

import httpx
import openai

from ..evaluators.judge_cache import JudgeCache
from ..utils.constants import PRICING_DOLLAR_PER_1M_TOKEN, UNIT_1M


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)

JUDGE_REQUEST_VERSION = 1
MAX_TOTAL_ATTEMPTS = 4
# Caps ONLY the headerless exponential-backoff fallback below -- never a
# server-sent Retry-After directive (see RETRY_AFTER_WARN_SECONDS below).
MAX_RETRY_DELAY_SECONDS = 60.0
# Issue #1313 owner ruling: a server-sent Retry-After is honored VERBATIM,
# never capped -- the server is the authority on when it will serve us
# again, and silently shortening its instruction is how a client turns one
# 429 into a retry storm. Truncating a long directive would also burn the
# bounded MAX_TOTAL_ATTEMPTS budget against a server that is still
# rate-limiting us. Instead of capping, a long wait is made VISIBLE:
# escalating log bands instead of silent truncation.
RETRY_AFTER_WARN_SECONDS = 60.0
RETRY_AFTER_ERROR_SECONDS = 300.0
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_LUNA_MODEL = "openai/gpt-5.6-luna"


class JudgeCacheOnlyMissError(RuntimeError):
    """Cache-only evaluation requested a judge response that is not cached."""


class EmptyLanguageModelResponseError(RuntimeError):
    """A live language-model response had null or blank content."""


class MalformedLanguageModelResponseError(RuntimeError):
    """A live language-model HTTP response had no ``choices``.

    OpenRouter (and the providers it routes to) can return an HTTP 200 whose
    body is an error object (``{"error": {...}}``) rather than a completion,
    e.g. when the routed provider has a transient failure or no provider is
    available -- the OpenAI SDK does not always surface this as a raised
    exception, since the transport-level status is success. Treating a
    missing ``choices`` key as an unretried ``KeyError`` (the prior
    behavior) let one such response abort an entire multi-hour, paid judge
    run; this is retried the same as any other transient failure instead.
    """


class _RetryableValidationError(RuntimeError):
    def __init__(self, error):
        super().__init__(str(error))
        self.error = error


def is_first_char_punctuation(s):
    if s and s[0] in string.punctuation:
        return True
    return False


def _normalized_api_base_url(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("api_base_url must be a non-empty string")
    return value.strip().rstrip("/")


def _status_code(exc):
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_retryable_transport_error(exc):
    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            httpx.ConnectError,
            httpx.TimeoutException,
        ),
    ):
        return True
    status = _status_code(exc)
    return status in {408, 429} or (status is not None and 500 <= status <= 599)


def _retry_after_seconds(exc, retry_number, model):
    """Delay in seconds before the next judge retry.

    A server-sent ``Retry-After`` header (finite, non-negative) is returned
    VERBATIM -- never capped (issue #1313 owner ruling). A long wait is made
    VISIBLE instead of silently shortened: above RETRY_AFTER_ERROR_SECONDS
    this logs one error; above RETRY_AFTER_WARN_SECONDS (but not above the
    error threshold) it logs one warning instead; at or below
    RETRY_AFTER_WARN_SECONDS, nothing is logged. Each log names the model,
    the exact delay, and that the SERVER requested it, so an operator
    watching a paid run can tell a server-parked wait from our own backoff.

    Without a usable header (missing, malformed, negative, or non-finite),
    falls back to exponential backoff (``2**retry_number``), capped at
    MAX_RETRY_DELAY_SECONDS -- that cap applies ONLY to this fallback.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw_value = headers.get("Retry-After") if headers is not None else None
    if raw_value is not None:
        try:
            delay = float(raw_value)
        except (TypeError, ValueError):
            delay = None
        if delay is not None and math.isfinite(delay) and delay >= 0:
            if delay > RETRY_AFTER_ERROR_SECONDS:
                logger.error(
                    f"{model} judge request received a server Retry-After of {delay}s "
                    f"-- above the {RETRY_AFTER_ERROR_SECONDS}s error threshold. "
                    f"Honoring it verbatim (never capped)."
                )
            elif delay > RETRY_AFTER_WARN_SECONDS:
                logger.warning(
                    f"{model} judge request received a server Retry-After of {delay}s "
                    f"-- above the {RETRY_AFTER_WARN_SECONDS}s warning threshold. "
                    f"Honoring it verbatim (never capped)."
                )
            return delay
    return min(float(2**retry_number), MAX_RETRY_DELAY_SECONDS)


def _has_usable_retry_after(exc):
    """Whether the response supplied a finite non-negative delay."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw_value = headers.get("Retry-After") if headers is not None else None
    if raw_value is None:
        return False
    try:
        delay = float(raw_value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(delay) and delay >= 0


def _jittered_retry_delay(delay, retry_number, request):
    """Add stable request-derived jitter to a fallback backoff delay."""

    request_json = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(f"{retry_number}:{request_json}".encode("utf-8")).digest()
    jitter = int.from_bytes(digest[:8], "big") / 2**64
    return min(delay + jitter, MAX_RETRY_DELAY_SECONDS)


class LanguageModelStats(object):
    """Main class for recording language model usage."""

    def __init__(self, model):
        self.model = model
        self.key = f"{model}-{uuid.uuid4()}"
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.prompt_cache = {}
        self.total_call = 0
        self.total_cache_hit = 0
        self.total_usage_unavailable = 0

    def record(self, api_name, stats, prompt=None, completion=None, cache_hit=False):
        self.total_call += 1
        if cache_hit:
            self.total_cache_hit += 1
            return
        if stats is None:
            self.total_usage_unavailable += 1
            return
        if api_name not in self.completion_tokens:
            self.completion_tokens[api_name] = []
        if api_name not in self.prompt_tokens:
            self.prompt_tokens[api_name] = []
        if api_name not in self.prompt_cache:
            self.prompt_cache[api_name] = []

        completion_tokens = int(stats["completion_tokens"])
        self.completion_tokens[api_name].append(completion_tokens)
        prompt_tokens = int(stats["prompt_tokens"])
        self.prompt_tokens[api_name].append(prompt_tokens)
        logger.debug(
            f"calling {api_name}, input tokens {prompt_tokens}, "
            f"output tokens {completion_tokens}"
        )
        if prompt is not None:
            self.prompt_cache[api_name].append(
                {"prompt": prompt, "completion": completion}
            )

    def get_total_tokens(self, breakdown=True):
        sum_completion_tokens = sum(sum(values) for values in self.completion_tokens.values())
        sum_prompt_tokens = sum(sum(values) for values in self.prompt_tokens.values())
        if breakdown:
            return sum_prompt_tokens, sum_completion_tokens
        return sum_prompt_tokens + sum_completion_tokens

    def reset(self):
        self.completion_tokens = {}
        self.prompt_tokens = {}
        self.prompt_cache = {}
        self.total_usage_unavailable = 0

    def get_total_price(self):
        pricing = PRICING_DOLLAR_PER_1M_TOKEN.get(self.model)
        if pricing is None:
            logger.warning(
                f"No trusted price is configured for {self.model}; cost is unavailable."
            )
            return None
        if self.total_usage_unavailable:
            logger.warning(
                f"Token usage is unavailable for {self.total_usage_unavailable} live "
                f"{self.model} call(s); cost is unavailable."
            )
            return None
        input_tokens, output_tokens = self.get_total_tokens()
        input_price = (input_tokens / UNIT_1M) * pricing["input"]
        output_price = (output_tokens / UNIT_1M) * pricing["output"]
        return input_price + output_price

    def print_report(self):
        logger.warning("=" * 20)
        logger.warning(
            f"Total calls: {self.total_call}, Total cache hits: {self.total_cache_hit}"
        )
        price = self.get_total_price()
        if price is None:
            logger.warning("Total price: unavailable")
        else:
            logger.warning(f"Total price: ${price}")
        logger.warning("=" * 20)

    def get_report(self):
        return {
            "total_calls": self.total_call,
            "total_cache_hits": self.total_cache_hit,
            "total_price": self.get_total_price(),
        }


class LanguageModel(object):
    """Async remote language-model access with optional durable judge cache."""

    def __init__(
        self,
        model,
        client,
        dump_dir=None,
        use_cache=True,
        cache_level="api",
        **kwargs,
    ):
        self.model = model
        if "gpt-4o" not in model and model != OPENROUTER_LUNA_MODEL:
            raise ValueError(f"{model} model class is not supported yet.")
        self.stats = LanguageModelStats(model)
        self.client = client
        if dump_dir:
            cur_save_dir = Path(dump_dir) / "lm_cache"
            cur_save_dir.mkdir(parents=True, exist_ok=True)
            self.dump_dir = cur_save_dir
        self.temperature = float(kwargs.get("temperature", 1.0))
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")

        self.api_base_url = _normalized_api_base_url(
            kwargs.get("api_base_url", DEFAULT_API_BASE_URL)
        )
        self.endpoint = f"{self.api_base_url}/chat/completions"
        cache_only = kwargs.get("cache_only", False)
        if type(cache_only) is not bool:
            raise ValueError("cache_only must be a boolean")
        judge_cache_path = kwargs.get("judge_cache", None)
        if cache_only and judge_cache_path is None:
            raise ValueError("cache_only requires an explicit judge_cache path")
        if cache_only and client is not None:
            raise ValueError("cache_only LanguageModel must not receive a live client")
        self.cache_only = cache_only
        self.judge_cache = (
            None
            if judge_cache_path is None
            else JudgeCache(Path(judge_cache_path), cache_only=cache_only)
        )

        # Preserve upstream's pickle cache only for non-evaluator callers that
        # do not opt into the explicit SQLite judge cache.
        self.cache_dir = None
        self.use_cache = bool(use_cache) and self.judge_cache is None
        self.cache_level = cache_level
        self.cache_in_mem = {}
        self.api_count = {}
        if self.use_cache:
            assert kwargs.get("master_data_dir", None), (
                "master_data_dir is required for cache"
            )
            self.cache_dir = Path(kwargs["master_data_dir"]) / "persist_lm_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if kwargs.get("cache_tag", None):
                self.cache_file = (
                    self.cache_dir / f"{self.model}_{kwargs['cache_tag']}_cache.pkl"
                )
            else:
                self.cache_file = self.cache_dir / f"{self.model}_cache.pkl"
            if self.cache_file.exists():
                with open(self.cache_file, "rb") as handle:
                    self.cache_in_mem = pickle.load(handle)

    def normalize(self, text):
        return text.strip()

    def _get_cache_key(self, prompt, api_count, api_name):
        if self.cache_level == "prompt":
            return f"{prompt}"
        return f"{prompt}_____{api_count}_____{api_name}"

    def _judge_request(self, prompt):
        return {
            "version": JUDGE_REQUEST_VERSION,
            "endpoint": self.endpoint,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

    @staticmethod
    def _validate_completion(completion, validator):
        if completion is None:
            raise EmptyLanguageModelResponseError(
                "language model returned null or blank completion content"
            )
        if not isinstance(completion, str):
            raise TypeError(
                f"language model completion content must be str, got "
                f"{type(completion).__name__}"
            )
        if not completion.strip():
            raise EmptyLanguageModelResponseError(
                "language model returned null or blank completion content"
            )
        if validator is not None:
            validator(completion)

    async def _live_completion(self, request, validator):
        if self.client is None:
            raise RuntimeError("live language-model request has no client")
        last_error = None
        for attempt in range(MAX_TOTAL_ATTEMPTS):
            try:
                raw_completion = await self.client.chat.completions.create(
                    messages=request["messages"],
                    model=self.model,
                    temperature=self.temperature,
                )
                payload = raw_completion.to_dict()
                choices = payload.get("choices")
                if not choices:
                    raise MalformedLanguageModelResponseError(
                        f"{self.model} judge response has no choices; "
                        f"payload error={payload.get('error')!r}"
                    )
                content = choices[0]["message"]["content"]
                try:
                    self._validate_completion(content, validator)
                except ValueError as exc:
                    raise _RetryableValidationError(exc) from exc
                return self.normalize(content), payload.get("usage")
            except _RetryableValidationError as exc:
                last_error = exc
            except EmptyLanguageModelResponseError as exc:
                last_error = exc
            except MalformedLanguageModelResponseError as exc:
                last_error = exc
            except Exception as exc:
                last_error = exc
                retryable = _is_retryable_transport_error(exc)
                if not retryable:
                    raise

            if attempt + 1 == MAX_TOTAL_ATTEMPTS:
                if isinstance(last_error, _RetryableValidationError):
                    raise last_error.error
                raise last_error
            delay = _retry_after_seconds(last_error, attempt, self.model)
            if not _has_usable_retry_after(last_error):
                delay = _jittered_retry_delay(delay, attempt, request)
            logger.warning(
                f"Retrying {self.model} judge request after {type(last_error).__name__} "
                f"(attempt {attempt + 2}/{MAX_TOTAL_ATTEMPTS}, delay={delay}s)"
            )
            await asyncio.sleep(delay)
        raise RuntimeError("unreachable language-model retry state")

    async def _legacy_live_completion(self, request, validator):
        """Preserve upstream non-judge behavior without adding stacked retries."""

        if self.client is None:
            raise RuntimeError("live language-model request has no client")
        raw_completion = await self.client.chat.completions.create(
            messages=request["messages"],
            model=self.model,
            temperature=self.temperature,
        )
        payload = raw_completion.to_dict()
        choices = payload.get("choices")
        if not choices:
            raise MalformedLanguageModelResponseError(
                f"{self.model} response has no choices; payload error={payload.get('error')!r}"
            )
        content = choices[0]["message"]["content"]
        self._validate_completion(content, validator)
        return self.normalize(content), payload.get("usage")

    async def chat_completion(self, client, prompt, api_name, validator=None):
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be str, got {type(prompt).__name__}")

        if self.judge_cache is not None:
            request = self._judge_request(prompt)
            cached = self.judge_cache.get(request)
            if cached is not None:
                self._validate_completion(cached, validator)
                return cached, None, True
            if self.cache_only:
                raise JudgeCacheOnlyMissError(
                    f"cache-only judge miss for model={self.model!r}, api_name={api_name!r}"
                )
            completion, usage = await self._live_completion(request, validator)
            persisted = self.judge_cache.put_if_absent(request, completion)
            self._validate_completion(persisted, validator)
            return persisted, usage, False

        api_count = self.api_count.get(api_name, 0)
        self.api_count[api_name] = api_count + 1
        if self.use_cache:
            cache_key = self._get_cache_key(prompt, api_count, api_name)
            if cache_key in self.cache_in_mem:
                completion = self.cache_in_mem[cache_key]
                self._validate_completion(completion, validator)
                return completion, None, True

        request = self._judge_request(prompt)
        completion, usage = await self._legacy_live_completion(request, validator)
        if self.use_cache:
            self.cache_in_mem[cache_key] = completion
        return completion, usage, False

    async def chat_completions(
        self, api_names, prompts, batch_size=32, validator=None
    ):
        """Handle batched async calls without adding another retry layer."""

        if not isinstance(api_names, list):
            api_names = [api_names] * len(prompts)
        if len(api_names) != len(prompts):
            raise ValueError("api_names and prompts must have equal length")

        all_completions = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_api_names = api_names[i : i + batch_size]
            async_responses = [
                self.chat_completion(
                    self.client,
                    prompt,
                    api_name,
                    validator=validator,
                )
                for prompt, api_name in zip(
                    batch_prompts, batch_api_names
                )
            ]
            raw_completions = await asyncio.gather(*async_responses)
            for j, (completion, usage, cache_hit) in enumerate(raw_completions):
                all_completions.append(completion)
                self.stats.record(
                    batch_api_names[j],
                    usage,
                    prompt=batch_prompts[j],
                    completion=completion,
                    cache_hit=cache_hit,
                )
        return all_completions

    def dump(self):
        with open(self.dump_dir / "tmp_prompt_cache.json", "w") as outfile:
            json.dump(self.stats.prompt_cache, outfile, indent=4)
        with open(self.dump_dir / "cost.jsonl", "a") as handle:
            handle.write(json.dumps({"price": self.stats.get_total_price()}) + "\n")

    def save_cache(self):
        if self.use_cache:
            with open(self.cache_file, "wb") as handle:
                pickle.dump(self.cache_in_mem, handle, protocol=pickle.HIGHEST_PROTOCOL)

    async def close(self):
        """Close the durable cache and underlying HTTP client."""

        try:
            if self.judge_cache is not None:
                self.judge_cache.close()
        finally:
            if self.client is not None:
                await self.client.close()
