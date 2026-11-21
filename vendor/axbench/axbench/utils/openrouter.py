"""OpenRouter client construction for AxBench's OpenAI-compatible calls.

Local modification: opt-in AxBench-Adv pilot budget hooks fence each SDK HTTP
attempt against the root runner's shared ledger. Unset preserves normal clients.
"""

from __future__ import annotations

import os
from importlib import import_module

import httpx
from openai import AsyncOpenAI

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def pilot_http_options():
    """Fence both client constructors only when an operator enables the pilot."""
    if "AXBENCH_ADV_PILOT_BUDGET_FILE" not in os.environ:
        return {}
    budget = import_module("scripts.eval.axbench_adv.pilot_budget")
    return {"event_hooks": budget.async_budget_hooks()}


def create_openrouter_client(requirement: str) -> AsyncOpenAI:
    """Create an async OpenRouter client for the requested AxBench operation."""

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"OPENROUTER_API_KEY is required {requirement}")
    base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip()
    if not base_url:
        raise ValueError("OPENROUTER_BASE_URL must be a non-empty URL when set")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=100,
                max_connections=1000,
            ),
            headers={"Connection": "close"},
            **pilot_http_options(),
        ),
        max_retries=3,
    )
