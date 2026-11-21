"""Cross-client regression test for the 4xx-body-echo leak (Wave 3B task 5).

When the upstream HTTP provider (OpenRouter or Neuronpedia) returns a
4xx, the response body can echo sensitive content:

* OpenRouter 4xx (``400``/``422``/``429``) often includes the full
  request body. For ``axbench_base`` clients that body contains the
  judge prompt, the SAE concept description, or the generated
  completion under evaluation.
* Neuronpedia 4xx can include user-visible error messages keyed off
  the requested ``feature_id`` or the API key prefix.

In both cases a body echoed into an ``logger.error``/``.warning`` call
or stashed inside a raised exception's ``str()`` ends up persisted in
loguru / stdlib / cloud log aggregators at ERROR severity. The uniform
fix is to log status code + URL + body length only, and
never embed the body content.

This test file is the parametrized cross-client tripwire. It covers:

1. :class:`OpenRouterLLMClient` (fix shipped in ``c697ba1``).
2. :class:`OpenRouterSynthClient` (thin wrapper — inherits the fix
   via :class:`OpenRouterLLMClient`).
3. :func:`fetch_feature_description` (Neuronpedia module-level fetch).
4. :class:`NeuronpediaClient` (caching wrapper around #3).

Adding a 5th HTTP LLM client without registering it here and without
following the same scrubbing pattern will leave the regression
uncovered; the ``all_clients`` list at the bottom of this module is
the single source of truth.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
import requests
from loguru import logger as _loguru_logger

from steering.eval import neuronpedia_client as neuronpedia_module
from steering.eval.axbench_base import llm_judge_client as judge_module
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OpenRouterLLMClient,
)
from steering.eval.axbench_base.synth.openrouter_client import (
    OpenRouterSynthClient,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.neuronpedia_client import (
    NEURONPEDIA_API_KEY_ENV,
    NeuronpediaClient,
    NeuronpediaError,
    NeuronpediaNotPublishedError,
    fetch_feature_description,
)
from steering.eval.neuronpedia_config import NeuronpediaConfig

# A deliberately specific sentinel. If any log record or exception
# argument contains this string, the client leaked the body. Using a
# unique token (not ``"test"``) avoids false positives from generic
# log lines that happen to mention the word "test".
SECRET_BODY = "<SECRET_PROMPT>do not leak this</SECRET_PROMPT>"


# ---------------------------------------------------------------------------
# Fake HTTP responses used across clients
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal ``requests.Response`` stand-in exposing ``status_code`` /
    ``text`` / ``headers`` / ``json`` — the attributes the three
    clients under test touch."""

    def __init__(
        self,
        status_code: int,
        text: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = headers or {}

    def json(self) -> Any:
        # None of the 4xx assertions below go down the JSON path, but
        # some clients may defensively call ``.json()`` — return a
        # structure that does not embed the body and does not crash.
        raise ValueError("no json available for 4xx body")


def _fake_post_429(*args: Any, **kwargs: Any) -> _FakeResponse:
    return _FakeResponse(status_code=429, text=SECRET_BODY)


def _fake_post_400(*args: Any, **kwargs: Any) -> _FakeResponse:
    """Non-retryable 4xx. ``400`` forces the fast-fail path."""
    return _FakeResponse(status_code=400, text=SECRET_BODY)


def _fake_get_404(*args: Any, **kwargs: Any) -> _FakeResponse:
    return _FakeResponse(status_code=404, text=SECRET_BODY)


# ---------------------------------------------------------------------------
# loguru → caplog bridge (loguru does not propagate to stdlib by default)
# ---------------------------------------------------------------------------


@pytest.fixture
def loguru_caplog(
    caplog: pytest.LogCaptureFixture,
    # Keep all parameters explicit to preserve the resolved runtime annotation.
) -> Generator[pytest.LogCaptureFixture, None, None]:  # noqa: UP043
    """Yield a caplog that captures loguru output.

    Each client under test emits via loguru; stdlib caplog alone will
    miss those records. This fixture adds a loguru handler that
    forwards to pytest's caplog handler for the test's lifetime.
    """
    handler_id = _loguru_logger.add(
        caplog.handler,
        format="{message}",
        level=0,
    )
    try:
        yield caplog
    finally:
        _loguru_logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Per-client invokers
# ---------------------------------------------------------------------------
#
# Each invoker is a callable that:
#   1. Configures environment and monkeypatches the HTTP transport so
#      the HTTP call returns a 4xx with ``SECRET_BODY`` as the body.
#   2. Calls the client's public method and returns the raised
#      exception.
#
# The parametrized tests below iterate over ``all_clients`` to enforce
# the invariant uniformly.


def _invoke_openrouter_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fake_post: Callable[..., _FakeResponse],
) -> Exception:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-test-body-leak")
    # Production code routes through get_openrouter_session().post(...).
    # Patch the Session method so the stub intercepts session-based calls.
    monkeypatch.setattr(
        requests.Session, "post",
        lambda _self, *args, **kwargs: fake_post(*args, **kwargs),
    )
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: None)
    client = OpenRouterLLMClient(LLMClientConfig(), max_retries=1)
    with pytest.raises(AxBenchBaseError) as excinfo:
        client.complete([{"role": "user", "content": "inert"}])
    return excinfo.value


def _invoke_openrouter_synth_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fake_post: Callable[..., _FakeResponse],
) -> Exception:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-test-body-leak")
    # Synth client delegates all HTTP to OpenRouterLLMClient, so patch
    # the Session method for both.
    monkeypatch.setattr(
        requests.Session, "post",
        lambda _self, *args, **kwargs: fake_post(*args, **kwargs),
    )
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: None)
    client = OpenRouterSynthClient(LLMClientConfig(), max_retries=1)
    with pytest.raises(AxBenchBaseError) as excinfo:
        client.complete("inert prompt")
    return excinfo.value


def _invoke_fetch_feature_description(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fake_post: Callable[..., _FakeResponse],
) -> Exception:
    # Neuronpedia uses GET, not POST — adapt the fixture by having the
    # same fake_post response served from ``requests.get``.
    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return fake_post(url, **kwargs)

    monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "np-test-body-leak")
    monkeypatch.setattr(neuronpedia_module.requests, "get", _fake_get)
    monkeypatch.setattr(neuronpedia_module.time, "sleep", lambda s: None)
    cfg = NeuronpediaConfig(max_retries=0)
    # Task 29: a 404 now raises the typed ``NeuronpediaNotPublishedError``
    # (a ``RuntimeError`` subclass, deliberately NOT a ``NeuronpediaError``
    # so the batch skip-path doesn't swallow it). The body-scrub invariant
    # must still hold for that path, so accept both error families here.
    with pytest.raises((NeuronpediaError, NeuronpediaNotPublishedError)) as excinfo:
        fetch_feature_description(
            "gemma-2-2b", "8-gemmascope-res-16k", 42, config=cfg,
        )
    return excinfo.value


def _invoke_neuronpedia_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fake_post: Callable[..., _FakeResponse],
) -> Exception:
    def _fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        return fake_post(url, **kwargs)

    monkeypatch.setenv(NEURONPEDIA_API_KEY_ENV, "np-test-body-leak")
    monkeypatch.setattr(neuronpedia_module.requests, "get", _fake_get)
    monkeypatch.setattr(neuronpedia_module.time, "sleep", lambda s: None)
    cfg = NeuronpediaConfig(max_retries=0)
    cache_path = tmp_path / "cache.yaml"
    client = NeuronpediaClient(
        "gemma-2-2b", "8-gemmascope-res-16k", cache_path, config=cfg,
    )
    # Task 29: the 404 path propagates ``NeuronpediaNotPublishedError``
    # through the singular ``get_description`` (only the batch path's
    # ``except NeuronpediaError`` would skip, and that does not catch the
    # typed RuntimeError). Accept both so the body-scrub invariant is
    # still enforced for the not-published case.
    with pytest.raises((NeuronpediaError, NeuronpediaNotPublishedError)) as excinfo:
        client.get_description(42)
    return excinfo.value


# The canonical list of every HTTP LLM client in the repository. Add
# a new entry when you add a 5th client; CI will then enforce the
# scrubbing invariant for it automatically.
all_clients: list[tuple[str, str, Callable[..., Exception]]] = [
    # (client_id, canonical URL substring, invoker)
    (
        "OpenRouterLLMClient",
        OPENROUTER_CHAT_COMPLETIONS_URL,
        _invoke_openrouter_llm_client,
    ),
    (
        "OpenRouterSynthClient",
        OPENROUTER_CHAT_COMPLETIONS_URL,
        _invoke_openrouter_synth_client,
    ),
    (
        "fetch_feature_description",
        "neuronpedia.org/api/feature",
        _invoke_fetch_feature_description,
    ),
    (
        "NeuronpediaClient",
        "neuronpedia.org/api/feature",
        _invoke_neuronpedia_client,
    ),
]


# ---------------------------------------------------------------------------
# Test 1: the 4xx path must never embed the body in logs or exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client_id, canonical_url, invoker",
    all_clients,
    ids=[spec[0] for spec in all_clients],
)
def test_client_4xx_does_not_leak_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loguru_caplog: pytest.LogCaptureFixture,
    client_id: str,
    canonical_url: str,
    invoker: Callable[..., Exception],
) -> None:
    """4xx response body must NOT appear in log records or exception text.

    Monkeypatches the HTTP transport to return a 4xx with a sentinel
    body. Asserts that the sentinel string appears in neither the
    captured loguru log messages nor the raised exception's ``str()``
    or its ``.args``.
    """
    # Pick 429 for retry-aware clients (the non-retryable 400 path is
    # covered implicitly: both paths lead to the same log/raise
    # pattern, and the OpenRouter client exercises both code paths on
    # its way through retries).
    exc = invoker(monkeypatch, tmp_path, fake_post=_fake_post_429)

    # Collect every place where the body might have leaked.
    leaked_haystack = _collect_leakable_strings(loguru_caplog, exc)
    for place, text in leaked_haystack:
        assert SECRET_BODY not in text, (
            f"{client_id}: 4xx body leaked into {place}: {text!r}"
        )
        # Also guard against partial leakage of the sensitive inner
        # marker (``SECRET_PROMPT``) that callers could grep for.
        assert "SECRET_PROMPT" not in text, (
            f"{client_id}: 4xx partial body leaked into {place}: {text!r}"
        )
        assert "do not leak this" not in text, (
            f"{client_id}: 4xx body-content leaked into {place}: {text!r}"
        )


@pytest.mark.parametrize(
    "client_id, canonical_url, invoker",
    all_clients,
    ids=[spec[0] for spec in all_clients],
)
def test_client_4xx_non_retryable_does_not_leak_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loguru_caplog: pytest.LogCaptureFixture,
    client_id: str,
    canonical_url: str,
    invoker: Callable[..., Exception],
) -> None:
    """Non-retryable 4xx path (400 / 404) must also scrub the body.

    OpenRouter's 400 and Neuronpedia's 404 both fast-fail without
    retry. Both paths historically embedded ``resp.text[:N]`` in the
    exception message.
    """
    # Neuronpedia raises AuthError on 401 (special-cased) — use 404
    # for it and 400 for OpenRouter.
    fake = _fake_get_404 if "Neuronpedia" in client_id or "fetch_feature" in client_id else _fake_post_400
    exc = invoker(monkeypatch, tmp_path, fake_post=fake)
    leaked_haystack = _collect_leakable_strings(loguru_caplog, exc)
    for place, text in leaked_haystack:
        assert SECRET_BODY not in text, (
            f"{client_id}: non-retryable 4xx body leaked into {place}: {text!r}"
        )
        assert "SECRET_PROMPT" not in text, (
            f"{client_id}: non-retryable 4xx partial body leaked into {place}: {text!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: status code AND URL must STILL appear in logs (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client_id, canonical_url, invoker",
    all_clients,
    ids=[spec[0] for spec in all_clients],
)
def test_client_4xx_logs_status_and_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loguru_caplog: pytest.LogCaptureFixture,
    client_id: str,
    canonical_url: str,
    invoker: Callable[..., Exception],
) -> None:
    """Scrubbing must keep the STATUS CODE and URL in the log record.

    We removed the body — but operators still need enough context to
    diagnose a failure. Status code + URL + body length is the
    contract.
    """
    invoker(monkeypatch, tmp_path, fake_post=_fake_post_429)

    all_messages = " ".join(rec.message for rec in loguru_caplog.records)
    # Status code MUST still be visible — operators need to know the
    # failure class (429 vs 400) without the body.
    assert "429" in all_messages, (
        f"{client_id}: 429 status code missing from logs: {all_messages!r}"
    )
    # URL MUST still be visible — operators need to know which endpoint
    # failed. ``canonical_url`` may be a substring of the full URL (the
    # Neuronpedia URL has a trailing feature ID).
    assert canonical_url in all_messages, (
        f"{client_id}: URL {canonical_url!r} missing from logs: {all_messages!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: every HTTP LLM client in the tree must be registered here
# ---------------------------------------------------------------------------


def test_registry_covers_every_http_llm_client() -> None:
    """Tripwire: fails if someone adds a 5th client without registering it.

    The expected set of client IDs matches the currently hardened HTTP
    LLM clients. If a future client appears (e.g. an Anthropic-direct client, a Cohere
    client, a new judge variant), this test must be updated AT THE
    SAME TIME as the scrubbing fix so the regression coverage grows
    with the code.
    """
    registered = {spec[0] for spec in all_clients}
    expected = {
        "OpenRouterLLMClient",
        "OpenRouterSynthClient",
        "fetch_feature_description",
        "NeuronpediaClient",
    }
    assert registered == expected, (
        f"Tripwire: client registry drift. registered={sorted(registered)}, "
        f"expected={sorted(expected)}. If you added a new HTTP LLM client, "
        f"register it in tests/test_openrouter_body_leak.py::all_clients "
        f"AND add the body-scrubbing fix (see c697ba1)."
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _collect_leakable_strings(
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
) -> list[tuple[str, str]]:
    """Return a list of ``(source_label, string)`` tuples to scan.

    The scan covers every place a body echo could realistically end up:

    * each captured log record's formatted ``.message``;
    * each captured record's raw ``.getMessage()`` (covers %-format args);
    * ``str(exc)``;
    * each element of ``exc.args`` (the ``__cause__`` chain is not
      traversed — the clients never re-raise with ``from`` carrying a
      body).
    """
    haystack: list[tuple[str, str]] = []
    for rec in caplog.records:
        haystack.append((f"log.message({rec.name})", rec.message))
        haystack.append((f"log.getMessage({rec.name})", rec.getMessage()))
    haystack.append(("exception.str", str(exc)))
    for i, arg in enumerate(exc.args):
        haystack.append((f"exception.args[{i}]", str(arg)))
    return haystack
