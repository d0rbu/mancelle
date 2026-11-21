"""Tunables for :class:`NeuronpediaClient`.

Lives in its own file so that ``concept_list.py`` and other call sites
can import the config without pulling in :mod:`requests` (which the
client module depends on transitively).
"""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from steering.refined_types import (
    NonNegativeIntLike,
    PositiveFiniteNumberLike,
    PositiveIntLike,
)


@beartype
@dataclass(frozen=True)
class NeuronpediaConfig:
    """Tunables for the Neuronpedia HTTP client.

    Defaults target a typical Stage 0a scan (~16k features at the
    documented Neuronpedia rate limit). Override ``max_concurrent`` to
    stay under the operator's account quota; override ``max_retries``
    and ``backoff_*`` to tune resilience to transient 5xx / network
    errors.
    """

    max_concurrent: PositiveIntLike = 10
    """In-flight HTTP request cap during a batch fetch. Neuronpedia is
    rate-limited per account; values above ~20 commonly trigger 429s
    even with the retry/backoff loop. Measured baseline (serial,
    no concurrency) was ~1.4 req/sec; 10-way concurrency takes that
    to ~14 req/sec, ~10x faster than the historical sequential path."""

    max_retries: NonNegativeIntLike = 3
    """Retries per feature *after* the initial attempt. Total attempts
    per feature is ``max_retries + 1``."""

    timeout_seconds: PositiveFiniteNumberLike = 30.0
    """Per-request socket timeout. The library wrapper had no timeout,
    so a hung Neuronpedia frontend stalled Stage 0a indefinitely. 30s
    is generous for the typical sub-second response."""

    backoff_base_seconds: PositiveFiniteNumberLike = 1.0
    """Initial backoff. Each retry doubles the previous wait, capped
    at ``backoff_max_seconds``. Random jitter in ``[0, base)`` is added
    so a thundering-herd retry does not lockstep with peers."""

    backoff_max_seconds: PositiveFiniteNumberLike = 60.0
    """Hard cap on backoff between retries; honored even on 429
    ``Retry-After`` headers to avoid pathological hour-long waits when
    a misbehaving server returns a huge value."""

    requests_per_minute: PositiveIntLike | None = None
    """Optional client-side rate limit. ``None`` means no client-side
    throttle; rely on the server's 429s and the retry loop. Set to a
    positive integer to globally throttle outbound requests (useful
    when sharing a quota across many concurrent clients)."""

    def __post_init__(self) -> None:
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError(
                f"backoff_max_seconds ({self.backoff_max_seconds}) must be "
                f">= backoff_base_seconds ({self.backoff_base_seconds})"
            )
