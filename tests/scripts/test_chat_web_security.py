"""Regression tests for scripts.chat.chat_web security P0s (Phase 3 A1).

Covers:
- Auth: `X-API-Key` header gate when `cfg.api_key` is set.
- CORS: `cfg.allow_origins` defaults to an empty list (no `*`).
- Host: `cfg.host` defaults to `127.0.0.1` (no `0.0.0.0`).
- In-flight cap: `cfg.max_in_flight` caps concurrent requests.
- Log-leak: `cfg.log_content=False` (default) does NOT log user content.

Tests use FastAPI's TestClient — no real HTTP server is bound.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import scripts.chat.chat_web as cw

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal tokenizer stub sufficient for chat_completions token-building.

    Produces deterministic token sequences so tests can assert over them.
    """

    _specials: dict[str, int] = {
        "<user_start>": 1001,
        "<user_end>": 1002,
        "<assistant_start>": 1003,
        "<assistant_end>": 1004,
    }

    def get_bos_token_id(self) -> int:
        return 1000

    def encode_special(self, name: str) -> int:
        return self._specials[name]

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 + 10 for c in text]

    def decode(self, ids: list[int]) -> str:
        # Mirror InferenceEngine behavior when assembling assistant text.
        return "".join(chr((i - 10) % 256) for i in ids if i >= 10 and i < 1000)


def _install_fake_worker_pool(monkeypatch, assistant_text: str = "assistant says hi"):
    """Install a worker pool that produces `assistant_text` as the streamed response.

    This avoids loading a real model. The fake pool has one worker whose
    tokenizer+engine emit a fixed token sequence for any request.
    """

    class _FakeEngine:
        def generate(self, tokens, **kwargs):
            # Yield one token per character. token_column is list[int], per-sample.
            for c in assistant_text:
                token = ord(c) % 256 + 10  # same mapping as encode()
                # Mimic Engine.generate signature: (token_column, token_masks)
                yield [token], [None]

    worker = MagicMock()
    worker.gpu_id = 0
    worker.device = "cpu"
    worker.engine = _FakeEngine()
    worker.tokenizer = _FakeTokenizer()
    # generate_stream uses `with worker.autocast_ctx:` -> must be a context manager
    worker.autocast_ctx = _noop_cm()

    queue: asyncio.Queue = asyncio.Queue()

    pool = MagicMock()
    pool.num_gpus = 1
    pool.workers = [worker]
    pool.available_workers = queue

    acquire_count = {"n": 0}
    release_count = {"n": 0}

    async def _acquire():
        acquire_count["n"] += 1
        return worker

    async def _release(_w):
        release_count["n"] += 1

    pool.acquire_worker = _acquire
    pool.release_worker = _release
    pool._acquire_count = acquire_count
    pool._release_count = release_count

    monkeypatch.setattr(cw.app.state, "worker_pool", pool, raising=False)
    return pool


class _noop_cm:
    """Context manager that does nothing — substitute for torch.amp.autocast."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def client():
    """TestClient bound to the chat_web app.

    NOTE: We avoid the TestClient `with` block so the lifespan (which tries to
    load a real model) does NOT run. Endpoints that need `app.state.worker_pool`
    must install a fake via `_install_fake_worker_pool`.
    """
    yield TestClient(cw.app)


@pytest.fixture(autouse=True)
def _reset_cfg_and_semaphore(monkeypatch):
    """Ensure cfg is at a known baseline and the semaphore is full before each test."""
    # Reset api_key / log_content / allow_origins / host to defaults.
    monkeypatch.setattr(cw.cfg, "api_key", None, raising=False)
    monkeypatch.setattr(cw.cfg, "log_content", False, raising=False)
    monkeypatch.setattr(cw.cfg, "allow_origins", [], raising=False)
    # Reset semaphore to its default size (4) to undo any prior leak.
    cw.in_flight_semaphore = asyncio.Semaphore(4)
    yield


# ---------------------------------------------------------------------------
# Host default: 127.0.0.1
# ---------------------------------------------------------------------------


class TestHostDefault:
    def test_default_host_is_localhost(self):
        """P0 fix: chat_web must bind to 127.0.0.1, not 0.0.0.0, by default."""
        # The YAML default must NOT be 0.0.0.0.
        assert cw.cfg.host == "127.0.0.1", (
            f"expected default host 127.0.0.1, got {cw.cfg.host!r}"
        )


# ---------------------------------------------------------------------------
# CORS: no wildcard by default
# ---------------------------------------------------------------------------


class TestCORSDefaults:
    def test_default_allow_origins_is_empty(self):
        """P0 fix: default CORS allow_origins must be empty (no `*`)."""
        # The default allow_origins list is empty.
        assert list(cw.cfg.allow_origins) == []

    def test_cors_middleware_configured_without_wildcard(self):
        """The CORSMiddleware must NOT be configured with allow_origins=['*']."""
        cors_mw = None
        for mw in cw.app.user_middleware:
            # Match by class name to tolerate starlette re-exports.
            if "CORS" in getattr(mw.cls, "__name__", ""):
                cors_mw = mw
                break
        assert cors_mw is not None, "CORSMiddleware not registered"
        # kwargs may be under `kwargs` or as direct options depending on FastAPI
        # version; check both.
        kwargs = getattr(cors_mw, "kwargs", {}) or getattr(cors_mw, "options", {})
        assert kwargs.get("allow_origins") != ["*"], (
            "CORS must not use wildcard allow_origins"
        )
        assert kwargs.get("allow_credentials") is False, (
            "allow_credentials must be False when using header-based auth"
        )


# ---------------------------------------------------------------------------
# Auth: X-API-Key gate
# ---------------------------------------------------------------------------


class TestAuthApiKey:
    def test_chat_completions_without_key_is_401_when_configured(
        self, client, monkeypatch
    ):
        """P0 fix: when cfg.api_key is set, missing header yields 401."""
        monkeypatch.setattr(cw.cfg, "api_key", "secret-token", raising=False)
        # No X-API-Key header sent.
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401, resp.text

    def test_chat_completions_with_wrong_key_is_401(self, client, monkeypatch):
        monkeypatch.setattr(cw.cfg, "api_key", "secret-token", raising=False)
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": "wrong-token"},
        )
        assert resp.status_code == 401, resp.text

    def test_chat_completions_with_correct_key_passes_auth(self, client, monkeypatch):
        """With a correct key the auth gate lets the request through; we don't
        need real generation to succeed — only that 401 is NOT returned."""
        monkeypatch.setattr(cw.cfg, "api_key", "secret-token", raising=False)
        _install_fake_worker_pool(monkeypatch)
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": "secret-token"},
        )
        # Any status other than 401 means the auth gate opened.
        assert resp.status_code != 401, resp.text

    def test_health_requires_api_key_when_configured(self, client, monkeypatch):
        """P0 fix: /health should also be gated when api_key is configured."""
        monkeypatch.setattr(cw.cfg, "api_key", "secret", raising=False)
        resp = client.get("/health")
        assert resp.status_code == 401, resp.text

    def test_stats_requires_api_key_when_configured(self, client, monkeypatch):
        monkeypatch.setattr(cw.cfg, "api_key", "secret", raising=False)
        resp = client.get("/stats")
        assert resp.status_code == 401, resp.text

    def test_health_accessible_when_api_key_unset(self, client, monkeypatch):
        """When cfg.api_key is None, endpoints remain open (backward compatible)."""
        monkeypatch.setattr(cw.cfg, "api_key", None, raising=False)
        resp = client.get("/health")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# In-flight concurrency cap (asyncio.Semaphore)
# ---------------------------------------------------------------------------


class TestInFlightSemaphore:
    def test_semaphore_initialised_from_cfg(self):
        """The semaphore must exist and reflect cfg.max_in_flight."""
        assert isinstance(cw.in_flight_semaphore, asyncio.Semaphore)
        # Default is 4.
        assert cw.cfg.max_in_flight == 4

    def test_semaphore_enforces_cap(self, monkeypatch):
        """When the semaphore is exhausted, a new acquire blocks until release."""

        async def _scenario():
            sem = asyncio.Semaphore(1)
            await sem.acquire()
            # Second acquire must not resolve synchronously.
            acquire_task = asyncio.ensure_future(sem.acquire())
            # Give the loop a chance to schedule.
            await asyncio.sleep(0)
            assert not acquire_task.done(), (
                "acquire must block when semaphore is empty"
            )
            sem.release()
            # Now the blocked acquire should resolve.
            await asyncio.wait_for(acquire_task, timeout=1.0)

        asyncio.run(_scenario())


# ---------------------------------------------------------------------------
# Log-leak: user content must not be logged by default
# ---------------------------------------------------------------------------


class TestLogContentGate:
    def test_default_log_content_is_false(self):
        """P0 fix: cfg.log_content defaults to False (opt-in)."""
        assert cw.cfg.log_content is False

    def test_user_content_not_logged_by_default(self, client, monkeypatch, caplog):
        """P0 fix: user messages containing secrets MUST NOT reach stdout/log."""
        monkeypatch.setattr(cw.cfg, "api_key", None, raising=False)
        monkeypatch.setattr(cw.cfg, "log_content", False, raising=False)
        _install_fake_worker_pool(monkeypatch, assistant_text="ok")

        caplog.set_level(logging.INFO, logger=cw.logger.name)

        sentinel = "SECRET_SENTINEL_sk_live_abc123"
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": sentinel}]},
        )
        # Drain the streaming body.
        _ = resp.text
        # Sentinel must NOT appear in any log record.
        for record in caplog.records:
            assert sentinel not in record.getMessage(), (
                f"Sentinel leaked via log record: {record.getMessage()!r}"
            )

    def test_assistant_content_not_logged_by_default(self, client, monkeypatch, caplog):
        """P0 fix: assistant generations MUST NOT reach logs by default."""
        monkeypatch.setattr(cw.cfg, "api_key", None, raising=False)
        monkeypatch.setattr(cw.cfg, "log_content", False, raising=False)
        assistant_sentinel = "ASSIST_SENTINEL_xyz789"
        _install_fake_worker_pool(monkeypatch, assistant_text=assistant_sentinel)

        caplog.set_level(logging.INFO, logger=cw.logger.name)
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        _ = resp.text
        for record in caplog.records:
            assert assistant_sentinel not in record.getMessage(), (
                f"Assistant sentinel leaked: {record.getMessage()!r}"
            )

    def test_user_content_logged_when_opted_in(self, client, monkeypatch, caplog):
        """When cfg.log_content=True, content DOES appear in logs — opt-in knob works."""
        monkeypatch.setattr(cw.cfg, "api_key", None, raising=False)
        monkeypatch.setattr(cw.cfg, "log_content", True, raising=False)
        _install_fake_worker_pool(monkeypatch, assistant_text="ok")

        caplog.set_level(logging.INFO, logger=cw.logger.name)
        marker = "UNIQUE_LOG_CONTENT_MARKER"
        resp = client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": marker}]},
        )
        _ = resp.text
        messages = [r.getMessage() for r in caplog.records]
        assert any(marker in m for m in messages), (
            f"expected {marker!r} in logs when log_content=True; got {messages}"
        )
