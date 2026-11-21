"""Deterministic mock OpenRouter / Judge / Synth LLM client.

This is the shared mock used by axbench_base unit and integration tests
that need a stand-in for :class:`AxBenchJudgeClient` /
:class:`OpenRouterLLMClient` /
:class:`steering.eval.axbench_base.synth.openrouter_client.OpenRouterSynthClient`.

The mock is deliberately interface-compatible with all three clients via
duck-typing:

- ``complete(prompt, **kwargs)`` returns a canned string looked up in
  ``response_map`` (exact-match, then prefix-match, then ``None``
  fallback, finally ``KeyError``).
- ``score(prompt)`` and ``score_many(prompts)`` match the corresponding
  :class:`AxBenchJudgeClient` surfaces by wrapping the canned response
  through a 0/1/2-digit parser, returning ``(score_or_None, attempts_made)``.
- Context manager protocol + ``close()`` mirror the real clients' cleanup
  hooks so ``with client:`` callers work unchanged.

``call_count`` and ``recorded_prompts`` are available for assertions in
tests that need to verify invocation order / count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


@dataclass
class MockOpenRouterClient:
    """Deterministic mock of :class:`AxBenchJudgeClient` / :class:`OpenRouterLLMClient`.

    Args:
        response_map: Maps prompt to canned response. Keys are matched in
            this order:

            1. Exact string match.
            2. Prefix match (first key whose prompt starts with it wins —
               iteration order is insertion order, so tests can control
               disambiguation by adding more specific prefixes first).
            3. ``None`` as a default key if present.
            4. Otherwise: ``KeyError`` is raised.
        call_count: Counter incremented on every ``complete`` / ``score``
            call (``score`` calls ``complete`` internally).
        recorded_prompts: Every prompt passed to ``complete`` / ``score``,
            appended in call order.
    """

    response_map: dict[str | None, str] = field(default_factory=dict)
    call_count: int = 0
    recorded_prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str, **_kwargs: Any) -> str:
        """Return the canned response for ``prompt``.

        ``**_kwargs`` is accepted (but ignored) so the mock can be a drop-in
        replacement for both ``LLMJudgeClient.complete(messages, *,
        cache_key=...)`` and the positional ``complete(prompt)`` shape used
        by stage-boundary callers.
        """
        self.call_count += 1
        self.recorded_prompts.append(prompt)
        if prompt in self.response_map:
            return self.response_map[prompt]
        for key, value in self.response_map.items():
            if key is not None and prompt.startswith(key):
                return value
        if None in self.response_map:
            return self.response_map[None]
        raise KeyError(f"No mock response for prompt: {prompt[:80]}...")

    def score(self, prompt: str) -> tuple[int | None, int]:
        """:class:`AxBenchJudgeClient`-compatible score method.

        Returns ``(score, attempts)`` where ``score`` is the first 0/1/2
        digit found in the canned response, or ``None`` if the response
        is the literal string ``"invalid"`` or contains no 0/1/2 digit.
        ``attempts`` is always ``1`` — the mock never retries.
        """
        raw = self.complete(prompt)
        if raw == "invalid":
            return (None, 1)
        for ch in raw:
            if ch in "012":
                return (int(ch), 1)
        return (None, 1)

    def score_many(self, prompts: list[str]) -> list[tuple[int | None, int]]:
        """Score prompts in input order, matching the real batch surface."""
        return [self.score(prompt) for prompt in prompts]

    def close(self) -> None:
        """No-op close for API compatibility."""
        return None

    def __enter__(self) -> MockOpenRouterClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
