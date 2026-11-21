"""Shared offline fixtures that import in both isolated Python environments."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from scripts.eval.axbench_adv.pilot_budget import create_pilot_budget

_MODEL = "fixture/model"
_ENDPOINT = "https://router.invalid/v1/chat/completions"
_BODY = {
    "model": _MODEL,
    "messages": [{"role": "user", "content": "exact π input"}],
    "max_tokens": 17,
}


def _create(path: Path, **kwargs: Any) -> Path:
    options: dict[str, Any] = {
        "limit_usd": "2",
        "request_bound_usd": "1",
        "model": _MODEL,
        "endpoint": _ENDPOINT,
        "max_output_tokens": 128000,
        "pricing_evidence": "offline fixture",
    }
    options.update(kwargs)
    create_pilot_budget(path, **options)
    return path


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        seen: list[bytes] = []
        replies: list[tuple[int, bytes]] = []

        def do_POST(self) -> None:
            self.seen.append(self.rfile.read(int(self.headers["Content-Length"])))
            status, body = self.replies.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("retry-after-ms", "1")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions", Handler
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
