"""Offline tests for the vendored AxBench OpenRouter client seam."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CLIENT = _ROOT / "vendor" / "axbench" / "axbench" / "utils" / "openrouter.py"


def _load_client(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, Any, Any]:
    class FakeAsyncClient:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls.append(kwargs)

    class FakeLimits:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAsyncOpenAI:
        calls: list[dict] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls.append(kwargs)

    fake_httpx: Any = types.ModuleType("httpx")
    fake_httpx.AsyncClient = FakeAsyncClient
    fake_httpx.Limits = FakeLimits
    fake_openai: Any = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    spec = importlib.util.spec_from_file_location("axbench_openrouter_test_client", _CLIENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, FakeAsyncOpenAI, FakeAsyncClient


def test_openrouter_client_requires_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, _ = _load_client(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        module.create_openrouter_client("for PromptSteering synthesis")


def test_openrouter_client_uses_openrouter_key_and_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake_openai, fake_httpx = _load_client(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-secret")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://router.example/v1///")

    module.create_openrouter_client("for PromptSteering synthesis")

    assert len(fake_openai.calls) == 1
    assert {
        key: value
        for key, value in fake_openai.calls[0].items()
        if key != "http_client"
    } == {
        "api_key": "fixture-secret",
        "base_url": "https://router.example/v1",
        "timeout": 60.0,
        "max_retries": 3,
    }
    assert fake_openai.calls[0]["http_client"].kwargs == fake_httpx.calls[0]
    assert fake_httpx.calls[0]["headers"] == {"Connection": "close"}
    assert fake_httpx.calls[0]["limits"].kwargs == {
        "max_keepalive_connections": 100,
        "max_connections": 1000,
    }
