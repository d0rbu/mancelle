"""W2 unit tests: get_openrouter_session returns a thread-local
requests.Session pre-configured with Connection: close to avoid
urllib3's stale-keepalive failure mode.
"""
from __future__ import annotations

import socket
import threading

import requests


def test_get_openrouter_session_returns_a_requests_session():
    from steering.eval.axbench_base.openrouter import get_openrouter_session
    s = get_openrouter_session()
    assert isinstance(s, requests.Session)


def test_get_openrouter_session_sets_connection_close_header():
    from steering.eval.axbench_base.openrouter import get_openrouter_session
    s = get_openrouter_session()
    assert s.headers.get("Connection") == "close"


def test_get_openrouter_session_is_cached_per_thread():
    from steering.eval.axbench_base.openrouter import get_openrouter_session
    s1 = get_openrouter_session()
    s2 = get_openrouter_session()
    assert s1 is s2, "Calls in the same thread must return the same Session"


def test_get_openrouter_session_is_thread_local():
    from steering.eval.axbench_base.openrouter import get_openrouter_session
    sessions: dict[int, object] = {}

    def runner():
        sessions[threading.get_ident()] = get_openrouter_session()

    t1 = threading.Thread(target=runner)
    t2 = threading.Thread(target=runner)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(sessions) == 2
    s_values = list(sessions.values())
    assert s_values[0] is not s_values[1], (
        "Threads must each get their own Session object"
    )


def test_importing_openrouter_does_not_change_global_socket_timeout():
    """Defends against the rejected `socket.setdefaulttimeout(...)` global
    side effect from WIP commit f8e44593."""
    before = socket.getdefaulttimeout()
    import importlib

    import steering.eval.axbench_base.openrouter as om
    importlib.reload(om)
    after = socket.getdefaulttimeout()
    assert before == after, (
        f"importing openrouter changed socket default timeout: "
        f"{before!r} -> {after!r}"
    )
