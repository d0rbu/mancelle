"""Synchronous OpenRouter client specialized for synth LLM calls.

Stage 0c API verification generation and Stage 1b/1c synthesis send one prompt
per attempt to an OpenRouter-backed chat completion endpoint. The response is
treated as a single completion string for the calling stage's strict parser;
the resulting continuation is then QC-checked against the concept's SAE
activations.

This class is a thin wrapper around :class:`OpenRouterLLMClient`
(from ``llm_judge_client.py``) that adapts a single prompt string to
the underlying client's messages-list API. All retry and rate-limit
logic lives in the underlying client; this wrapper exists so synth
stages can depend on a synth-shaped API without knowing about the
judge-client plumbing.
"""

from __future__ import annotations

from typing import Any

from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient


class OpenRouterSynthClient:
    """Thin wrapper for single-prompt OpenRouter calls.

    Usage::

        client = OpenRouterSynthClient(cfg.synth, max_retries=3)
        response = client.complete("generate a continuation that includes X")

    The ``max_retries`` argument is the HTTP transport retry budget forwarded
    to :class:`OpenRouterLLMClient`. It is independent of the calling stage's
    content/QC proposal budget. The default allows 3 retries after the first
    HTTP attempt.

    A ``response_cache`` may also be forwarded so stages can reuse
    previously-completed LLM calls across mid-stage crashes. Callers
    that want to participate in the cache must pass a ``cache_key``
    per invocation of :meth:`complete`.
    """

    # ``response_cache`` is declared ``Any`` for the same reason as the
    # underlying OpenRouterLLMClient — see its docstring.
    def __init__(
        self,
        cfg: LLMClientConfig,
        max_retries: int = 3,
        response_cache: Any = None,
        max_tokens: int | None = None,
    ) -> None:
        self._client = OpenRouterLLMClient(
            cfg,
            max_retries=max_retries,
            response_cache=response_cache,
            max_tokens=max_tokens,
        )

    def complete(
        self,
        prompt: str,
        *,
        cache_key: str | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Send one prompt, optionally preceded by a system instruction.

        Raises :class:`AxBenchBaseError` on any OpenRouter error, retry
        exhaustion, null/non-string content, or missing API key (the
        latter would have been caught at construction time).

        ``cache_key`` is forwarded to the underlying client's cache (if one was
        configured at construction time). Pass ``None`` to bypass caching.
        The caller owns cache identity for ``system_prompt``; synth stages must
        include it in their semantic fingerprint and cache key.

        Omitting ``system_prompt`` preserves the exact historical one-message
        request body used by Stage 1 and Stage 0c definition inference.
        """
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._client.complete(messages, cache_key=cache_key)
