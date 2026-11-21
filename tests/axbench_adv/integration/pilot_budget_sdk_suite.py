"""Real SDK HTTP retry tests in AxBench's isolated Python 3.12 environment."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, AsyncOpenAI

from scripts.eval.axbench_adv.pilot_budget import (
    BUDGET_ENV,
    PilotBudgetExceeded,
    async_budget_hooks,
    budget_summary,
)
from tests.axbench_adv.pilot_budget_helpers import _BODY, _create, server

__all__ = ("server",)


def test_real_sdk_retries_are_separately_fenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server
) -> None:
    endpoint, handler = server
    path = _create(tmp_path / "sdk.sqlite3", endpoint=endpoint)
    monkeypatch.setenv(BUDGET_ENV, str(path))
    handler.replies = [(500, b'{"error":{"message":"retry"}}')] * 2

    async def exercise() -> None:
        async with AsyncOpenAI(
            api_key="offline-fixture",
            base_url=endpoint.rsplit("/chat", 1)[0],
            max_retries=3,
            timeout=5,
            http_client=httpx.AsyncClient(event_hooks=async_budget_hooks()),
        ) as client:
            with pytest.raises(APIConnectionError) as exc:
                await client.chat.completions.create(**_BODY)
            assert isinstance(exc.value.__cause__, PilotBudgetExceeded)

    asyncio.run(exercise())
    assert len(handler.seen) == 2
    assert [json.loads(body) for body in handler.seen] == [_BODY, _BODY]
    assert budget_summary(path)["attempts"] == 2


def test_vendor_factory_installs_hooks_and_success_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server
) -> None:
    endpoint, handler = server
    path = _create(tmp_path / "vendor.sqlite3", endpoint=endpoint)
    monkeypatch.setenv(BUDGET_ENV, str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setenv("OPENROUTER_BASE_URL", endpoint.rsplit("/chat", 1)[0])
    handler.replies = [(200, b'{"id":"fixture","choices":[],"usage":{"cost":0.125}}')]
    module_path = Path(__file__).resolve().parents[3] / "vendor/axbench/axbench/utils/openrouter.py"
    spec = importlib.util.spec_from_file_location("budget_vendor_client", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    async def exercise() -> None:
        async with module.create_openrouter_client("for offline test") as client:
            await client.chat.completions.create(**_BODY)

    asyncio.run(exercise())
    assert [json.loads(body) for body in handler.seen] == [_BODY]
    assert budget_summary(path)["committed_nanos"] == 125_000_000
    assert b"offline-fixture" not in path.read_bytes()
    assert b"exact" not in path.read_bytes()
