"""Offline acceptance test for the real vendored AxBench evaluation CLI."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from scripts.eval.axbench_upstream import run_baseline_project_evaluation as nl_launcher
from scripts.eval.axbench_upstream import run_evaluation as launcher
from scripts.eval.axbench_upstream.merge_methods import METHOD_FACTORS, SUPPORTED_METHOD_FACTORS
from steering.eval.axbench_histograms import generate_histogram_reports
from tests.scripts.eval.test_axbench_baseline_project_evaluation_launcher import (
    NL_EXPECTED_METHODS,
)
from tests.scripts.eval.test_axbench_baseline_project_evaluation_launcher import (
    _merged_frame as _baseline_project_merged_frame,
)
from tests.scripts.eval.test_axbench_evaluation_launcher import _merged_frame


class _LocalJudgeServer(ThreadingHTTPServer):
    request_queue_size = 128


@pytest.mark.slow
@pytest.mark.parametrize("nl_enabled", [False, True], ids=["ten_method", "seven_method_nl"])
def test_real_vendored_cli_against_local_openai_compatible_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nl_enabled: bool,
) -> None:
    requests: list[tuple[str, dict[str, object], str]] = []
    requests_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, payload: dict[str, object], **headers: str) -> None:
            response = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(response)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            with requests_lock:
                request_index = len(requests)
                fault = (
                    ("rate-limit", "null", "malformed")[request_index]
                    if request_index < 3
                    else "ok"
                )
                requests.append((self.path, payload, fault))

            if fault == "rate-limit":
                self._send_json(
                    429,
                    {
                        "error": {
                            "message": "one deterministic rate limit",
                            "type": "rate_limit_error",
                            "code": "rate_limit",
                        }
                    },
                    **{"Retry-After": "0"},
                )
                return

            prompt = payload["messages"][0]["content"]
            axis_score = (
                0.5
                if "[Concept Start]" in prompt
                else (1.0 if "[Instruction Start]" in prompt else 1.5)
            )
            # Every generated fixture sentence carries concept/input/factor.
            concept_offset = 0.25 if "answer 1/" in prompt else 0.0
            content = {
                "null": None,
                "malformed": "the answer is one",
                "ok": f"Rating: [[{axis_score + concept_offset}]]",
            }[fault]
            self._send_json(
                200,
                {
                    "id": "offline-judge",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": payload["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = _LocalJudgeServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        input_path = tmp_path / "merged" / "steering_data.parquet"
        input_path.parent.mkdir(parents=True)
        if nl_enabled:
            _baseline_project_merged_frame(
                max_concepts=2,
                methods=NL_EXPECTED_METHODS,
            ).to_parquet(input_path, index=False)
        else:
            _merged_frame(max_concepts=2).to_parquet(input_path, index=False)
        live_output_dir = tmp_path / "evaluated-live"
        cache_path = tmp_path / "judge-cache.sqlite3"
        monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")
        api_base_url = f"http://127.0.0.1:{server.server_port}/v1"

        selected_launcher = nl_launcher if nl_enabled else launcher
        selected_launcher.run_evaluation(
            input_path=input_path,
            output_dir=live_output_dir,
            cache_path=cache_path,
            api_base_url=api_base_url,
            **({"include_nl_steering": True} if nl_enabled else {}),
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    method_factors = (
        {method: SUPPORTED_METHOD_FACTORS[method] for method in NL_EXPECTED_METHODS}
        if nl_enabled
        else METHOD_FACTORS
    )
    expected_cache_rows = 2 * 3 * 10 * sum(len(factors) for factors in method_factors.values())
    assert expected_cache_rows == (3_720 if nl_enabled else 4_560)
    assert len(requests) == expected_cache_rows + 3
    assert [fault for _path, _payload, fault in requests].count("rate-limit") == 1
    assert [fault for _path, _payload, fault in requests].count("null") == 1
    assert [fault for _path, _payload, fault in requests].count("malformed") == 1
    assert {path for path, _payload, _fault in requests} == {"/v1/chat/completions"}
    assert {payload["model"] for _path, payload, _fault in requests} == {"openai/gpt-5.6-luna"}
    assert {payload["temperature"] for _path, payload, _fault in requests} == {0.0}
    assert all(
        isinstance(payload["messages"], list) and len(payload["messages"]) == 1
        for _path, payload, _fault in requests
    )

    with sqlite3.connect(cache_path) as connection:
        cache_rows = connection.execute("SELECT COUNT(*) FROM judge_responses").fetchone()[0]
    assert cache_rows == expected_cache_rows
    cache_bytes = cache_path.read_bytes()
    live_frame = pd.read_parquet(live_output_dir / "steering_data.parquet")
    assert live_frame.shape == (240, 77 if nl_enabled else 104)
    live_json = [
        json.loads(line) for line in (live_output_dir / "steering.jsonl").read_text().splitlines()
    ]

    request_count = len(requests)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    cached_output_dir = tmp_path / "evaluated-cache-only"
    selected_launcher.run_evaluation(
        input_path=input_path,
        output_dir=cached_output_dir,
        cache_path=cache_path,
        cache_only=True,
        api_base_url=api_base_url,
        **({"include_nl_steering": True} if nl_enabled else {}),
    )

    assert len(requests) == request_count
    assert cache_path.read_bytes() == cache_bytes
    with sqlite3.connect(cache_path) as connection:
        replay_cache_rows = connection.execute("SELECT COUNT(*) FROM judge_responses").fetchone()[0]
    assert replay_cache_rows == expected_cache_rows
    pd.testing.assert_frame_equal(
        pd.read_parquet(cached_output_dir / "steering_data.parquet"),
        live_frame,
    )
    cached_json = [
        json.loads(line) for line in (cached_output_dir / "steering.jsonl").read_text().splitlines()
    ]
    assert cached_json == live_json

    report = generate_histogram_reports(
        input_dir=cached_output_dir,
        output_dir=tmp_path / "histograms",
        pipeline="upstream",
        method_factors=dict.fromkeys(method_factors, 1.0),
    )
    metadata = json.loads(report.read_text())
    expected_means = {
        "relevance_concept_ratings": 0.625,
        "fluency_ratings": 1.625,
        "relevance_instruction_ratings": 1.125,
    }
    assert len(metadata["reports"]["upstream"]) == 3 * len(method_factors)
    for summary in metadata["reports"]["upstream"]:
        assert summary["mean"] == expected_means[summary["axis"]]
        assert summary["n_concepts"] == 2
    assert len(requests) == request_count
    assert cache_path.read_bytes() == cache_bytes
