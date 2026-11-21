"""Shared OpenRouter endpoint + API-key constants for axbench_base clients.

Centralizes the endpoint URL and env-var name consumed by both the
sync-client used by concept-examples QC (``llm_judge_client.py``) and the
judge orchestrator (``judge.py``).
"""

from __future__ import annotations

import os
import threading
from importlib import import_module

import requests
from requests.adapters import HTTPAdapter

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


_thread_local = threading.local()


def get_openrouter_session() -> requests.Session:
    """Return a thread-local ``requests.Session`` pre-configured for
    OpenRouter calls.

    Sets ``Connection: close`` to disable urllib3 connection pooling,
    which has been observed to hand off half-closed TCP keepalive
    sockets whose read-side close is undetectable until a per-request
    timeout fires. Each call within the same thread returns the same
    session; calls from different threads return independent sessions
    so concurrent callers don't share a single Session's lock.

    This helper intentionally does NOT set ``socket.setdefaulttimeout``
    or otherwise modify global Python state — every robustness knob is
    scoped to the returned Session.
    """
    cached = getattr(_thread_local, "session", None)
    budget_path = os.environ.get("AXBENCH_ADV_PILOT_BUDGET_FILE")
    if cached is not None and getattr(_thread_local, "budget_path", None) == budget_path:
        return cached
    if cached is not None:
        cached.close()
    s = requests.Session()
    s.headers["Connection"] = "close"
    # Adapter retries are disabled here so our explicit retry loop in
    # llm_judge_client / judge owns the retry policy. pool_maxsize=10
    # mirrors max_concurrent's default.
    if budget_path is not None:
        # Optional operator fence shared with the separate AxBench environment.
        # Ordinary clients retain their original transport and dependencies.
        budget = import_module("scripts.eval.axbench_adv.pilot_budget")
        adapter = budget.PilotBudgetAdapter(budget_path, pool_connections=10, pool_maxsize=10)
    else:
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    _thread_local.session = s
    _thread_local.budget_path = budget_path
    return s
