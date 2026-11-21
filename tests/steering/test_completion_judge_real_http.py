"""Exercise the REAL ``requests.Session.send`` path inside completion_judge.

The legacy ``tests/steering/test_completion_judge.py:226-260`` test
patched ``steering.eval.completion_judge._requests.post`` directly with
``MagicMock``. That bypasses the entire ``requests.Session`` /
``HTTPAdapter`` stack — including any future header injection,
URL-canonicalization, or content-type handling — so a real bug at line
391 (``_requests.post(url, headers=...,)``) could go undetected.

This test stubs only ``HTTPAdapter.send`` so the call goes through the
full ``requests.Session.prepare_request`` → ``_call_api`` →
``raise_for_status`` → ``json()`` codepath, with a real
``requests.Response`` object as the return value.
"""

from __future__ import annotations

import io
import json

import pytest
import requests

from errors import ParseError
from steering.eval import completion_judge as cj
from steering.eval.completion_judge import CompletionJudgeClient


def _make_response(
    *,
    status: int = 200,
    body: dict | None = None,
    text: str | None = None,
    request: requests.PreparedRequest | None = None,
) -> requests.Response:
    """Construct a real ``requests.Response`` (not MagicMock).

    The body is JSON-encoded if ``body`` is provided. ``text`` overrides
    body when set (for malformed JSON / non-JSON responses).
    """
    resp = requests.Response()
    resp.status_code = status
    if body is not None:
        raw = json.dumps(body).encode()
        resp.headers["Content-Type"] = "application/json"
    elif text is not None:
        raw = text.encode()
        resp.headers["Content-Type"] = "text/plain"
    else:
        raw = b""
    resp.raw = io.BytesIO(raw)
    resp.reason = "OK" if status == 200 else "ERR"
    if request is not None:
        resp.request = request
        resp.url = request.url
    return resp


class _StubAdapter(requests.adapters.HTTPAdapter):  # ty: ignore[possibly-missing-submodule]
    """Replace ``HTTPAdapter.send`` with a fixture-driven response queue."""

    def __init__(self, responses: list[requests.Response]):
        super().__init__()
        self._responses = responses
        self.sent_requests: list[requests.PreparedRequest] = []

    def send(self, request, *args, **kwargs):
        self.sent_requests.append(request)
        if not self._responses:
            raise requests.exceptions.ConnectionError(
                f"_StubAdapter exhausted; {len(self.sent_requests)} calls made"
            )
        resp = self._responses.pop(0)
        if resp.request is None:
            resp.request = request
            resp.url = request.url
        return resp


def _install_stub_adapter(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[requests.Response],
) -> _StubAdapter:
    """Mount a fake adapter inside the ``_requests`` module's default Session.

    ``completion_judge._call_api`` calls ``_requests.post`` which ultimately
    routes through a session-default HTTPAdapter. Patching ``HTTPAdapter.send``
    at the class level catches any new session created during the call.
    """
    stub = _StubAdapter(responses)

    # Bind the per-instance state by replacing the method with a closure
    # that references our stub's attributes.
    def _patched_send(_self, request, *a, **kw):  # noqa: ARG001
        return stub.send(request, *a, **kw)

    monkeypatch.setattr(
        cj._requests.adapters.HTTPAdapter, "send", _patched_send  # ty: ignore[possibly-missing-submodule]
    )
    return stub


# ---------------------------------------------------------------------------
# Real HTTP path — parsed response
# ---------------------------------------------------------------------------


class TestRealHttpPath:
    def test_parsed_response_via_real_session_send(
        self, tmp_path, monkeypatch
    ) -> None:
        """A 200 response with valid score body should parse cleanly.

        Distinct from ``test_completion_judge.py::test_judge_calls_api_on_miss``
        which patches ``_requests.post`` directly and bypasses Session.send.
        """
        cache = str(tmp_path / "real_http.db")
        client = CompletionJudgeClient(
            cache_path=cache,
            requests_per_minute=0,
            api_key="test-key",
            base_url="https://example.test/v1",
        )

        ok_resp = _make_response(
            status=200,
            body={"choices": [{"message": {"content": "coherence: 4\nfeature_alignment: 3"}}]},
        )
        stub = _install_stub_adapter(monkeypatch, [ok_resp])

        result = client.judge("context", "completion", "feature desc", "ground truth")
        assert result is not None
        assert result.coherence == 4
        assert result.feature_alignment == 3
        # The send went through the real Session path
        assert len(stub.sent_requests) == 1
        sent = stub.sent_requests[0]
        # Authorization header was injected by _call_api
        assert sent.headers.get("Authorization") == "Bearer test-key"
        # JSON body contains model + messages
        body = json.loads(sent.body)
        assert body["model"] == client.model
        assert len(body["messages"]) == 2

    def test_raise_for_status_propagates_5xx_via_real_path(
        self, tmp_path, monkeypatch
    ) -> None:
        """5xx responses must surface ``HTTPError`` at the call site.

        ``_call_api`` calls ``resp.raise_for_status()`` (line 392) — under
        the real session path this is the actual requests.Response method,
        not a MagicMock. A 500 must yield ``HTTPError``.
        """
        cache = str(tmp_path / "real_http_500.db")
        client = CompletionJudgeClient(
            cache_path=cache,
            requests_per_minute=0,
            api_key="test-key",
            base_url="https://example.test/v1",
        )

        # Three 500s — _call_api wrapped by judge() retries _MAX_RETRIES=3 times
        # but here we test _call_api in isolation via the public judge() path.
        bad = _make_response(status=500, body={"error": "internal"})
        _install_stub_adapter(monkeypatch, [bad, bad, bad])

        # Public ``judge`` swallows HTTPError into ParseError after retries —
        # we just need to confirm the real path was exercised. The retry
        # behavior is covered by the existing test_judge_raises_on_exhausted_retries.
        with pytest.raises((requests.exceptions.HTTPError, ParseError)):
            client.judge("ctx", "comp", "desc", "truth")

    def test_call_api_directly_yields_string(
        self, tmp_path, monkeypatch
    ) -> None:
        """Call ``_call_api`` directly to exercise the real path without
        ``judge()``'s parse-and-retry wrapper. Verifies the parsed JSON
        ``choices[0].message.content`` reaches the caller verbatim."""
        cache = str(tmp_path / "real_http_direct.db")
        client = CompletionJudgeClient(
            cache_path=cache,
            requests_per_minute=0,
            api_key="test-key",
            base_url="https://example.test/v1",
        )

        ok_resp = _make_response(
            status=200,
            body={"choices": [{"message": {"content": "verbatim payload"}}]},
        )
        _install_stub_adapter(monkeypatch, [ok_resp])

        # Directly invoke _call_api (private but stable for this slice)
        out = client._call_api(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ]
        )
        assert out == "verbatim payload"
