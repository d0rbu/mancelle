"""Offline integration of the reflection adapter with the real root client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from steering.eval.axbench_adv.gepa_optimization import build_openrouter_reflection_lm
from steering.eval.axbench_base import llm_judge_client
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_CHAT_COMPLETIONS_URL,
)
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache


class Response:
    status_code = 200
    headers: dict[str, str] = {}
    text = ""

    @staticmethod
    def json() -> dict[str, Any]:
        return {"choices": [{"message": {"content": "```integrated suffix```"}}]}


class Session:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Response:
        self.calls.append({"url": url, **kwargs})
        return Response()


def test_real_openrouter_client_payload_parser_and_cache_only_repeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(OPENROUTER_API_KEY_ENV, "sk-offline-test")
    session = Session()
    monkeypatch.setattr(llm_judge_client, "get_openrouter_session", lambda: session)
    cfg = LLMClientConfig(
        provider="openrouter",
        model="provider/reflection-model",
        requests_per_minute=0,
    )
    prompt = "the exact GEPA reflection prompt"

    with SynthResponseCache(tmp_path / "reflection.sqlite") as cache:
        reflection = build_openrouter_reflection_lm(cfg=cfg, cache=cache, max_retries=0)
        assert reflection(prompt) == "```integrated suffix```"
        assert reflection(prompt) == "```integrated suffix```"
        assert reflection.paid_cache_misses == 1
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == OPENROUTER_CHAT_COMPLETIONS_URL
    assert call["json"] == {
        "model": "provider/reflection-model",
        "messages": [{"role": "user", "content": prompt}],
    }
    assert call["headers"] == {
        "Authorization": "Bearer sk-offline-test",
        "Content-Type": "application/json",
    }
    assert call["timeout"] == (10, 60)
