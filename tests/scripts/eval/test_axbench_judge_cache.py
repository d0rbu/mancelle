"""Focused tests for the ordinary vendored AxBench judge cache."""

from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vendor"
    / "axbench"
    / "axbench"
    / "evaluators"
    / "judge_cache.py"
)
_SPEC = importlib.util.spec_from_file_location("vendored_axbench_judge_cache", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
JudgeCache = _MODULE.JudgeCache


def _request(**updates: object) -> dict[str, object]:
    request: dict[str, object] = {
        "version": 1,
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Judge café."}],
        "temperature": 0.0,
    }
    request.update(updates)
    return request


def _concurrent_put(
    path: str,
    response: str,
    request: Mapping[str, object],
    start: Any,
    results: Any,
) -> None:
    start.wait()
    with JudgeCache(Path(path)) as cache:
        results.put(cache.put_if_absent(request, response))


def test_canonical_request_identity_and_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    request = _request(endpoint="  https://api.openai.com/v1/chat/completions///  ")
    reordered = {
        "temperature": 0,
        "messages": [{"content": "Judge café.", "role": "user"}],
        "model": "gpt-4o-mini",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "version": 1,
    }

    with JudgeCache(path) as cache:
        assert cache.put_if_absent(request, "answer") == "answer"
        assert cache.get(reordered) == "answer"

    expected_json = (
        '{"endpoint":"https://api.openai.com/v1/chat/completions",'
        '"messages":[{"content":"Judge café.","role":"user"}],'
        '"model":"gpt-4o-mini","temperature":0.0,"version":1}'
    )
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT request_sha256, request_json, response FROM judge_responses"
        ).fetchone()
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'judge_responses'"
        ).fetchone()[0]
        columns = connection.execute("PRAGMA table_info(judge_responses)").fetchall()

    assert row == (hashlib.sha256(expected_json.encode("utf-8")).digest(), expected_json, "answer")
    assert "WITHOUT ROWID" in table_sql.upper()
    assert [(column[1], column[2], column[3], column[5]) for column in columns] == [
        ("request_sha256", "BLOB", 1, 1),
        ("request_json", "TEXT", 1, 0),
        ("response", "TEXT", 1, 0),
    ]


def test_insert_hit_first_writer_wins_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    request = _request()
    with JudgeCache(path) as cache:
        assert cache.get(request) is None
        assert cache.put_if_absent(request, "first") == "first"
        assert cache.put_if_absent(request, "second") == "first"

    with JudgeCache(path) as reopened:
        assert reopened.get(request) == "first"


@pytest.mark.parametrize(
    ("candidate", "error"),
    [
        ({key: value for key, value in _request().items() if key != "model"}, ValueError),
        (_request(extra="unexpected"), ValueError),
        (_request(endpoint=" / "), ValueError),
        (_request(model=" "), ValueError),
        (_request(messages=({"role": "user", "content": "prompt"},)), TypeError),
        (_request(messages=[]), ValueError),
        (_request(temperature=float("nan")), ValueError),
        (_request(version=True), TypeError),
    ],
)
def test_rejects_invalid_requests(
    tmp_path: Path,
    candidate: Mapping[str, object],
    error: type[Exception],
) -> None:
    with JudgeCache(tmp_path / "judge.sqlite") as cache:
        with pytest.raises(error):
            cache.get(candidate)


def test_rejects_invalid_response_digest_and_use_after_close(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    cache = JudgeCache(path)
    with pytest.raises(TypeError, match="response must be a string"):
        cache.put_if_absent(_request(), 3)  # type: ignore[arg-type]
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        cache.get(_request())

    with JudgeCache(path, digest_fn=lambda _value: b"short") as bad_digest:
        with pytest.raises(ValueError, match="exactly 32 bytes"):
            bad_digest.get(_request())


def test_digest_collision_fails_and_rolls_back(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    first = _request(messages=[{"role": "user", "content": "first"}])
    second = _request(messages=[{"role": "user", "content": "second"}])

    def constant_digest(_value: bytes) -> bytes:
        return b"x" * 32

    with JudgeCache(path, digest_fn=constant_digest) as cache:
        assert cache.put_if_absent(first, "first response") == "first response"
        with pytest.raises(ValueError, match="collision"):
            cache.get(second)
        with pytest.raises(ValueError, match="collision"):
            cache.put_if_absent(second, "second response")
        assert cache.get(first) == "first response"


def test_rejects_blob_stored_in_response_text_column(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    request = _request()
    with JudgeCache(path) as cache:
        cache.put_if_absent(request, "valid text")

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE judge_responses SET response = ?",
            (sqlite3.Binary(b"not text"),),
        )

    with JudgeCache(path) as cache:
        with pytest.raises(ValueError, match="stored response is not text"):
            cache.get(request)


def test_two_process_concurrent_puts_return_one_stored_winner(tmp_path: Path) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path):
        pass

    context = multiprocessing.get_context("spawn")
    start = context.Event()
    results = context.Queue()
    request = _request()
    processes = [
        context.Process(
            target=_concurrent_put,
            args=(str(path), response, request, start, results),
        )
        for response in ("from one", "from two")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    try:
        returned = [results.get(timeout=5), results.get(timeout=5)]
    except Empty as exc:  # pragma: no cover - clearer failure than a queue timeout
        pytest.fail(f"concurrent cache worker did not return: {exc}")
    assert returned[0] == returned[1]
    assert returned[0] in {"from one", "from two"}
    with JudgeCache(path) as cache:
        assert cache.get(request) == returned[0]


def test_cache_only_is_read_only_and_never_creates(tmp_path: Path) -> None:
    absent = tmp_path / "absent" / "judge.sqlite"
    with pytest.raises(FileNotFoundError):
        JudgeCache(absent, cache_only=True)
    assert not absent.parent.exists()

    path = tmp_path / "judge.sqlite"
    with JudgeCache(path):
        pass
    before = path.read_bytes()
    with JudgeCache(path, cache_only=True) as cache:
        assert cache.get(_request()) is None
        with pytest.raises(PermissionError, match="read-only"):
            cache.put_if_absent(_request(), "forbidden")
    assert path.read_bytes() == before
    assert not path.with_name(f"{path.name}-journal").exists()
