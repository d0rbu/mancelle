"""Fail-closed, request-recording OpenRouter transport for tests.

The production AxBench clients use a cached thread-local
``requests.Session.post``.  Patching ``requests.post`` therefore does not
intercept their traffic.  This module owns the test-only boundary used by the
shared fixtures and the real-Gemma integration test.

The transport is deliberately not a generic "return some text" mock.  Its
OpenRouter boundary rejects every request that is not classified, before
``HTTPAdapter.send`` can perform DNS, socket creation, or provider I/O. Other
HTTP traffic remains available for the explicitly separate Neuronpedia and
HuggingFace prerequisites.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from steering.eval.axbench_base import openrouter
from steering.eval.axbench_base.judge import AxBenchJudgeClient, _make_cache_key
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.openrouter import OPENROUTER_CHAT_COMPLETIONS_URL
from steering.eval.axbench_base.synth.openrouter_client import OpenRouterSynthClient

OPENROUTER_REQUEST_CLASSES = frozenset(
    {
        "stage0c_definition",
        "stage0c_continuation",
        "stage1_synth_rich",
        "stage1_synth_avoid",
        "stage5_judge",
    }
)


class OfflineOpenRouterError(AssertionError):
    """A request violated the explicit offline mock contract."""


@dataclass(frozen=True)
class RequestMetadata:
    """Test-only metadata carried from a production client call to the wire."""

    request_class: str | None = None
    cache_key: str | None = None
    client: str | None = None


@dataclass
class OpenRouterRequest:
    """Recorded request and response identity at the Session.post boundary."""

    ordinal: int
    url: str
    model: str
    headers: dict[str, str]
    timeout: Any
    payload: dict[str, Any]
    messages: list[dict[str, str]]
    request_class: str | None
    client: str | None
    cache_key: str | None
    payload_hash: str
    prompt_hash: str
    response_hash: str | None = None
    handled: bool = False

    @property
    def message_roles(self) -> tuple[str, ...]:
        return tuple(str(message.get("role", "")) for message in self.messages)

    @property
    def cache_key_prefix(self) -> str:
        return (self.cache_key or "")[:16]


ResponseFactory = Callable[[OpenRouterRequest], str]

_metadata_var: contextvars.ContextVar[RequestMetadata | None] = contextvars.ContextVar(
    "axbench_offline_openrouter_metadata", default=None
)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def classify_cache_key(cache_key: str | None, prompt: str) -> str | None:
    """Classify stable production cache-key tags, with prompt fallbacks."""

    key = cache_key or ""
    if "stage0c_definition" in key:
        return "stage0c_definition"
    if "stage0c_continuation" in key:
        return "stage0c_continuation"
    if "pos_avoid" in key or ("neg_c" in key and "stage0c" not in key):
        return "stage1_synth_avoid"
    if "pos_c" in key:
        return "stage1_synth_rich"
    if "AXBENCH_DEFINITION_BEGIN_" in prompt:
        return "stage0c_definition"
    if "AXBENCH_CONTINUATION_BEGIN_" in prompt:
        return "stage0c_continuation"
    if "AXBENCH_STAGE1B_RICH_BEGIN_" in prompt:
        return "stage1_synth_rich"
    if (
        "AXBENCH_STAGE1B_AVOID_BEGIN_" in prompt
        or "AXBENCH_STAGE1C_AVOID_BEGIN_" in prompt
    ):
        return "stage1_synth_avoid"
    if "Complete the content by adding elements" in prompt:
        return "stage1_synth_rich"
    lowered = prompt.lower()
    if "gracefully transitions away" in lowered:
        return "stage1_synth_avoid"
    if "rating" in lowered and "rating:" in lowered:
        return "stage5_judge"
    return None


def _tripwire_message(boundary: str, *args: Any, **kwargs: Any) -> str:
    return (
        f"OpenRouter offline tripwire reached at {boundary}; refusing all "
        f"network access args={args!r} kwargs_keys={sorted(kwargs)!r}"
    )


class OfflineResponse:
    """Small ``requests.Response``-compatible object used by the clients."""

    def __init__(self, content: str, *, status_code: int = 200) -> None:
        self.status_code = status_code
        self.text = content
        self._content = content

    def json(self) -> dict[str, list[dict[str, dict[str, str]]]]:
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"offline OpenRouter response status={self.status_code}",
                response=self,
            )


class OfflineOpenRouterTransport:
    """Install a fail-closed Session-level OpenRouter mock.

    ``response_factory`` must return the assistant content for every request.
    It is passed a fully recorded request after the request has crossed the
    same ``Session.post`` boundary as production.  Omitting it, or returning
    a response for an unknown request class, raises with diagnostics.
    """

    def __init__(self, response_factory: ResponseFactory | None = None) -> None:
        self.response_factory = response_factory
        self.requests: list[OpenRouterRequest] = []
        self.top_level_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.transport_tripwire_calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._sessions: dict[int, requests.Session] = {}
        self._lock = threading.RLock()
        self._installed = False
        self.post_mock = MagicMock(name="offline_session_post")
        self.top_level_post_mock = MagicMock(
            name="offline_top_level_requests_post",
            side_effect=self._top_level_tripwire,
        )

    def install(self, monkeypatch: pytest.MonkeyPatch) -> OfflineOpenRouterTransport:
        """Clear cached sessions and patch all relevant requests boundaries."""

        self._clear_cached_main_thread_session()
        original_post = requests.sessions.Session.post
        original_request = requests.sessions.Session.request
        original_get = requests.sessions.Session.get
        original_send = requests.sessions.Session.send

        def session_post(
            session: requests.Session,
            url: str,
            *,
            json: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None,
            timeout: Any = None,
            **kwargs: Any,
        ) -> OfflineResponse:
            if not self._is_openrouter_url(url):
                return original_post(
                    session,
                    url,
                    json=json,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                )
            return self._post(session, url, json, headers, timeout, kwargs)

        monkeypatch.setattr(requests.sessions.Session, "post", session_post)
        monkeypatch.setattr(requests, "post", self.top_level_post_mock)

        def request_tripwire(
            session: requests.Session,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not self._is_openrouter_url(self._request_url(args, kwargs)):
                return original_request(session, *args, **kwargs)
            self._transport_tripwire("Session.request", *args, **kwargs)
            return None

        def get_tripwire(
            session: requests.Session,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if not self._is_openrouter_url(self._request_url(args, kwargs)):
                return original_get(session, *args, **kwargs)
            self._transport_tripwire("Session.get", *args, **kwargs)
            return None

        def send_tripwire(
            session: requests.Session,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            request = args[0] if args else kwargs.get("request")
            if not self._is_openrouter_url(getattr(request, "url", None)):
                return original_send(session, *args, **kwargs)
            self._transport_tripwire("Session.send", *args, **kwargs)
            return None

        monkeypatch.setattr(requests.sessions.Session, "request", request_tripwire)
        monkeypatch.setattr(requests.sessions.Session, "get", get_tripwire)
        monkeypatch.setattr(requests.sessions.Session, "send", send_tripwire)
        self._installed = True
        return self

    @staticmethod
    def _is_openrouter_url(url: Any) -> bool:
        return str(url).split("?", 1)[0] == OPENROUTER_CHAT_COMPLETIONS_URL

    @staticmethod
    def _request_url(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if "url" in kwargs:
            return kwargs["url"]
        if len(args) >= 2:
            return args[1]
        if args:
            return args[0]
        return None

    def bind_production_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrap original client methods to carry cache-key metadata to the wire."""
        original_synth_complete = OpenRouterSynthClient.complete

        def synth_complete(
            client: Any,
            prompt: str,
            *,
            cache_key: str | None = None,
            system_prompt: str | None = None,
        ) -> str:
            metadata = RequestMetadata(
                request_class=classify_cache_key(cache_key, prompt),
                cache_key=cache_key,
                client="synth",
            )
            with self.metadata(metadata):
                return original_synth_complete(
                    client,
                    prompt,
                    cache_key=cache_key,
                    system_prompt=system_prompt,
                )

        monkeypatch.setattr(OpenRouterSynthClient, "complete", synth_complete)

        original_llm_complete = OpenRouterLLMClient.complete

        def llm_complete(
            client: Any,
            messages: list[dict[str, str]],
            *,
            cache_key: str | None = None,
        ) -> str:
            current = _metadata_var.get()
            if current is None:
                prompt = str(messages[-1].get("content", "")) if messages else ""
                current = RequestMetadata(
                    request_class=classify_cache_key(cache_key, prompt),
                    cache_key=cache_key,
                    client="llm",
                )
            with self.metadata(current):
                return original_llm_complete(
                    client,
                    messages,
                    cache_key=cache_key,
                )

        monkeypatch.setattr(OpenRouterLLMClient, "complete", llm_complete)

        original_judge_call = AxBenchJudgeClient._call_api

        def judge_call(client: Any, prompt: str) -> str:
            cache_key = _make_cache_key(client._cfg.model, prompt)
            with self.metadata(
                RequestMetadata(
                    request_class="stage5_judge",
                    cache_key=cache_key,
                    client="judge",
                )
            ):
                return original_judge_call(client, prompt)

        monkeypatch.setattr(AxBenchJudgeClient, "_call_api", judge_call)

    @contextlib.contextmanager
    def metadata(self, metadata: RequestMetadata) -> Iterator[None]:
        token = _metadata_var.set(metadata)
        try:
            yield
        finally:
            _metadata_var.reset(token)

    def _post(
        self,
        session: requests.Session,
        url: str,
        payload: dict[str, Any] | None,
        headers: dict[str, str] | None,
        timeout: Any,
        kwargs: dict[str, Any],
    ) -> OfflineResponse:
        if kwargs:
            raise OfflineOpenRouterError(
                _tripwire_message("Session.post unexpected kwargs", kwargs)
            )
        payload = payload or {}
        messages_raw = payload.get("messages")
        messages = messages_raw if isinstance(messages_raw, list) else []
        normalized_messages = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in messages
            if isinstance(message, dict)
        ]
        current = _metadata_var.get()
        prompt = normalized_messages[-1]["content"] if normalized_messages else ""
        request_class = (
            current.request_class if current is not None else classify_cache_key(None, prompt)
        )
        record = OpenRouterRequest(
            ordinal=len(self.requests),
            url=str(url),
            model=str(payload.get("model", "")),
            headers=dict(headers or {}),
            timeout=timeout,
            payload=dict(payload),
            messages=normalized_messages,
            request_class=request_class,
            client=None if current is None else current.client,
            cache_key=None if current is None else current.cache_key,
            payload_hash=_sha256_json(payload),
            prompt_hash=_sha256_json(normalized_messages),
        )
        with self._lock:
            self.requests.append(record)
            self._sessions[id(session)] = session
        if record.request_class not in OPENROUTER_REQUEST_CLASSES:
            raise self._unknown_request(
                record,
                f"unknown request class {record.request_class!r}",
            )
        if self.response_factory is None:
            raise self._unknown_request(record, "no response factory installed")
        content = self.response_factory(record)
        if not isinstance(content, str) or not content:
            raise OfflineOpenRouterError(self._diagnostics(record, f"factory returned {content!r}"))
        record.response_hash = _sha256_text(content)
        record.handled = True
        self.post_mock(session, url, json=payload, headers=headers, timeout=timeout)
        return OfflineResponse(content)

    def _unknown_request(self, record: OpenRouterRequest, reason: str) -> OfflineOpenRouterError:
        return OfflineOpenRouterError(self._diagnostics(record, reason))

    def _diagnostics(self, record: OpenRouterRequest, reason: str) -> str:
        return (
            f"Unhandled OpenRouter request ordinal={record.ordinal} reason={reason}; "
            f"url={record.url!r} model={record.model!r} "
            f"request_class={record.request_class!r} "
            f"cache_key_prefix={record.cache_key_prefix!r} "
            f"message_roles={record.message_roles!r} "
            f"prompt_hash={record.prompt_hash[:16]}"
        )

    def _top_level_tripwire(self, *args: Any, **kwargs: Any) -> None:
        self.top_level_calls.append((args, kwargs))
        raise OfflineOpenRouterError(_tripwire_message("requests.post", *args, **kwargs))

    def _transport_tripwire(self, boundary: str, *args: Any, **kwargs: Any) -> None:
        self.transport_tripwire_calls.append((boundary, args, kwargs))
        raise OfflineOpenRouterError(_tripwire_message(boundary, *args, **kwargs))

    def assert_all_handled(self) -> None:
        unhandled = [request for request in self.requests if not request.handled]
        if unhandled:
            raise OfflineOpenRouterError(
                "OpenRouter offline transport has unhandled recorded requests: "
                + "; ".join(self._diagnostics(request, "not handled") for request in unhandled)
            )

    def close(self) -> None:
        """Close all observed sessions and remove the main cached session."""

        self._clear_cached_main_thread_session()
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
        self._clear_cached_main_thread_session()

    def _clear_cached_main_thread_session(self) -> None:
        cached = getattr(openrouter._thread_local, "session", None)
        if cached is not None:
            cached.close()
            delattr(openrouter._thread_local, "session")

    def __enter__(self) -> OfflineOpenRouterTransport:
        if not self._installed:
            raise RuntimeError("install() must be called before entering transport")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()


__all__ = [
    "OPENROUTER_REQUEST_CLASSES",
    "OfflineOpenRouterError",
    "OfflineOpenRouterTransport",
    "OfflineResponse",
    "OpenRouterRequest",
    "RequestMetadata",
    "classify_cache_key",
]
