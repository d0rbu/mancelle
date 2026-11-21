"""Offline cost-fence tests, including real HTTP clients and process contention."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
import requests
from requests.adapters import HTTPAdapter

from scripts.eval.axbench_adv.pilot_budget import (
    BUDGET_ENV,
    PilotBudgetAdapter,
    PilotBudgetExceeded,
    _money,
    async_budget_hooks,
    budget_summary,
    reserve_attempt,
    settle_attempt,
    validate_pilot_budget,
)
from steering.eval.axbench_adv.gepa_optimization import build_openrouter_reflection_lm
from steering.eval.axbench_base import llm_judge_client, openrouter
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from tests.axbench_adv.pilot_budget_helpers import _BODY, _ENDPOINT, _create, server

_NANO = 1_000_000_000
__all__ = ("server",)


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(BUDGET_ENV, raising=False)
    return _create(tmp_path / "pilot.sqlite3")


def _reserve(path: Path, body: object = _BODY, endpoint: str = _ENDPOINT) -> int:
    return reserve_attempt(path, endpoint=endpoint, payload=json.dumps(body).encode())


def _settle(path: Path, attempt: int, cost: object, status: int = 200) -> None:
    settle_attempt(
        path, attempt, status_code=status, response=json.dumps({"usage": {"cost": cost}}).encode()
    )


@pytest.mark.parametrize(
    "value",
    [True, None, [], {}, "", "NaN", "Infinity", "-1", -0.001, "1e999999"],
)
def test_money_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        _money(value)


def test_money_preserves_exact_sqlite_integer_boundary() -> None:
    assert _money("9223372036.854775807") == 2**63 - 1
    for value in ("9223372036.854775808", "9223372036.85477580700000000001"):
        with pytest.raises(ValueError, match="integer range"):
            _money(value)


def test_launch_rejects_missing_budget_configuration(ledger: Path) -> None:
    model = _BODY["model"]
    assert isinstance(model, str)
    with sqlite3.connect(ledger) as connection:
        connection.execute("DELETE FROM budget")
    with pytest.raises(ValueError, match="no singleton budget configuration"):
        validate_pilot_budget(ledger, model=model, endpoint=_ENDPOINT)
    with sqlite3.connect(ledger) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (0,)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("0.0000000001", 1),
        ("0.00000000100000000000000000000000000001", 2),
        ("1.0000000001", _NANO + 1),
        (0.25, _NANO // 4),
    ],
)
def test_money_rounds_up_never_releasing_fractional_nanodollars(
    value: object, expected: int
) -> None:
    assert _money(value) == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"limit_usd": "0"},
        {"request_bound_usd": "0"},
        {"request_bound_usd": "3"},
        {"model": ""},
        {"endpoint": " "},
        {"pricing_evidence": ""},
        {"max_output_tokens": True},
        {"max_output_tokens": 0},
    ],
)
def test_invalid_creation_does_not_create_file(tmp_path: Path, changes: dict[str, object]) -> None:
    path = tmp_path / "bad.sqlite3"
    with pytest.raises(ValueError):
        _create(path, **changes)
    assert not path.exists()


def test_existing_ledger_cannot_be_reset_even_when_empty(ledger: Path, tmp_path: Path) -> None:
    _reserve(ledger)
    before = ledger.read_bytes()
    with pytest.raises(FileExistsError):
        _create(ledger)
    assert ledger.read_bytes() == before
    empty = tmp_path / "empty"
    empty.touch()
    with pytest.raises(FileExistsError):
        _create(empty)


def test_missing_ledger_fails_without_creating_replacement(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        _reserve(missing)
    assert not missing.exists()


def test_relative_creation_fails_without_creating_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        _create(Path("relative.sqlite3"))
    assert not (tmp_path / "relative.sqlite3").exists()


def test_relative_existing_ledger_cannot_split_process_accounting(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ledger.parent)
    with pytest.raises(ValueError, match="absolute"):
        _reserve(Path(ledger.name))
    assert budget_summary(ledger)["attempts"] == 0


def test_exact_limit_and_receipt_refund(ledger: Path) -> None:
    first = _reserve(ledger)
    second = _reserve(ledger)
    with pytest.raises(PilotBudgetExceeded):
        _reserve(ledger)
    _settle(ledger, first, "0.25")
    with pytest.raises(PilotBudgetExceeded):
        _reserve(ledger)
    _settle(ledger, second, "0.75")
    assert _reserve(ledger) == 3
    assert budget_summary(ledger) == {
        "limit_nanos": 2 * _NANO,
        "attempts": 3,
        "known_actual_nanos": _NANO,
        "committed_nanos": 2 * _NANO,
        "halted": 0,
    }


@pytest.mark.parametrize(
    "response",
    [
        b"invalid",
        b"[]",
        b"null",
        b"{}",
        b'{"usage":{}}',
        b'{"usage":{"cost":null}}',
        b'{"usage":{"cost":-1}}',
        b'{"usage":{"cost":"NaN"}}',
        b'{"usage":{"cost":true}}',
        b'{"usage":{"cost":1e999999999999999999999999}}',
        b'{"error":{},"usage":{"cost":0}}',
    ],
)
def test_success_without_valid_cost_halts_and_never_refunds(ledger: Path, response: bytes) -> None:
    attempt = _reserve(ledger)
    with pytest.raises(PilotBudgetExceeded, match="valid cost receipt"):
        settle_attempt(ledger, attempt, status_code=200, response=response)
    assert budget_summary(ledger)["committed_nanos"] == _NANO
    assert budget_summary(ledger)["known_actual_nanos"] == 0
    assert budget_summary(ledger)["halted"] == 1
    with pytest.raises(PilotBudgetExceeded, match="halted"):
        _reserve(ledger)


@pytest.mark.parametrize("status", [201, 302, 400, 429, 500, 503])
def test_non_success_keeps_full_reservation_even_with_zero_cost(ledger: Path, status: int) -> None:
    _settle(ledger, _reserve(ledger), 0, status)
    assert budget_summary(ledger)["committed_nanos"] == _NANO


def test_bound_violation_halts_and_preserves_actual_receipt(ledger: Path) -> None:
    with pytest.raises(PilotBudgetExceeded, match="exceeded"):
        _settle(ledger, _reserve(ledger), "1.1")
    assert budget_summary(ledger)["known_actual_nanos"] == 1_100_000_000
    assert budget_summary(ledger)["halted"] == 1
    with pytest.raises(PilotBudgetExceeded, match="halted"):
        _reserve(ledger)


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        ("1.0000000010000000001", 1_000_000_002),
        ("0.00000000100000000000000000000000000001", 2),
    ],
)
def test_receipt_cost_uses_exact_json_decimal_before_upward_rounding(
    tmp_path: Path, cost: str, expected: int
) -> None:
    path = _create(
        tmp_path / "exact-decimal.sqlite3",
        limit_usd="3",
        request_bound_usd="2",
    )
    attempt = _reserve(path)
    settle_attempt(
        path,
        attempt,
        status_code=200,
        response=f'{{"usage":{{"cost":{cost}}}}}'.encode(),
    )
    assert budget_summary(path)["known_actual_nanos"] == expected


def test_receipts_cannot_be_applied_twice_or_without_attempt(ledger: Path) -> None:
    attempt = _reserve(ledger)
    _settle(ledger, attempt, "0.1")
    with pytest.raises(RuntimeError, match="already settled"):
        _settle(ledger, attempt, 0)
    with pytest.raises(RuntimeError, match="missing"):
        _settle(ledger, 9999, 0)
    assert budget_summary(ledger)["committed_nanos"] == 100_000_000


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"stream": True},
        {"tools": []},
        {"provider": {}},
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user", "content": []}]},
        {"messages": [{"role": "user", "content": "x", "name": "extra"}]},
        {"max_tokens": 128001},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"max_completion_tokens": 128001},
    ],
)
def test_unpriced_request_rejected_before_reserving(
    ledger: Path, change: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        _reserve(ledger, {**_BODY, **change})
    assert budget_summary(ledger)["attempts"] == 0


def test_endpoint_mismatch_rejected_before_reserving(ledger: Path) -> None:
    with pytest.raises(ValueError, match="endpoint"):
        _reserve(ledger, endpoint="https://other.invalid/v1/chat/completions")
    assert budget_summary(ledger)["attempts"] == 0


def test_launch_preflight_requires_exact_priced_identity_and_live_ledger(
    ledger: Path,
) -> None:
    model = str(_BODY["model"])
    validate_pilot_budget(ledger, model=model, endpoint=_ENDPOINT)
    with pytest.raises(ValueError, match="priced ledger"):
        validate_pilot_budget(ledger, model="other", endpoint=_ENDPOINT)
    with pytest.raises(ValueError, match="priced ledger"):
        validate_pilot_budget(ledger, model=model, endpoint="https://other.invalid")
    with pytest.raises(PilotBudgetExceeded, match="exceeded"):
        _settle(ledger, _reserve(ledger), "1.1")
    with pytest.raises(PilotBudgetExceeded, match="halted"):
        validate_pilot_budget(ledger, model=model, endpoint=_ENDPOINT)


def _competing_attempt(path: str) -> bool:
    try:
        _reserve(Path(path))
    except PilotBudgetExceeded:
        return False
    return True


def test_reservations_are_atomic_across_real_processes(tmp_path: Path) -> None:
    path = _create(tmp_path / "concurrent.sqlite3", limit_usd="7")
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        results = list(executor.map(_competing_attempt, [str(path)] * 25))
    assert sum(results) == 7
    assert budget_summary(path)["committed_nanos"] == 7 * _NANO


def test_requests_transport_keeps_payload_exact_and_counts_every_attempt(
    tmp_path: Path, server
) -> None:
    endpoint, handler = server
    path = _create(tmp_path / "sync.sqlite3", endpoint=endpoint)
    handler.replies = [(429, b'{"error":"rate limit"}'), (200, b'{"usage":{"cost":0.25}}')]
    payload = json.dumps(_BODY).encode()
    with requests.Session() as client:
        client.mount("http://", PilotBudgetAdapter(str(path)))
        assert client.post(endpoint, data=payload, timeout=5).status_code == 429
        assert client.post(endpoint, data=payload, timeout=5).status_code == 200
        with pytest.raises(PilotBudgetExceeded):
            client.post(endpoint, data=payload, timeout=5)
    assert handler.seen == [payload, payload]
    assert budget_summary(path)["committed_nanos"] == 1_250_000_000


def test_transport_error_keeps_reservation(ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise requests.Timeout("uncertain delivery")

    monkeypatch.setattr(HTTPAdapter, "send", fail)
    with requests.Session() as client:
        client.mount("https://", PilotBudgetAdapter(str(ledger)))
        with pytest.raises(requests.Timeout):
            client.post(_ENDPOINT, json=_BODY)
    assert budget_summary(ledger)["committed_nanos"] == _NANO


def test_success_without_cost_halts_real_transport_after_one_request(
    tmp_path: Path, server
) -> None:
    endpoint, handler = server
    path = _create(tmp_path / "missing-cost.sqlite3", endpoint=endpoint)
    handler.replies = [(200, b'{"choices":[]}')]
    payload = json.dumps(_BODY).encode()
    with requests.Session() as client:
        client.mount("http://", PilotBudgetAdapter(str(path)))
        with pytest.raises(PilotBudgetExceeded, match="valid cost receipt"):
            client.post(endpoint, data=payload, timeout=5)
    assert handler.seen == [payload]
    assert budget_summary(path)["committed_nanos"] == _NANO
    assert budget_summary(path)["halted"] == 1


def test_cached_root_session_switches_guard_on_and_off(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(openrouter._thread_local, "session", None, raising=False)
    unguarded = openrouter.get_openrouter_session()
    monkeypatch.setenv(BUDGET_ENV, str(ledger))
    guarded = openrouter.get_openrouter_session()
    assert guarded is not unguarded
    assert isinstance(guarded.adapters["https://"], PilotBudgetAdapter)
    assert openrouter.get_openrouter_session() is guarded
    monkeypatch.delenv(BUDGET_ENV)
    normal = openrouter.get_openrouter_session()
    assert normal is not guarded
    assert not isinstance(normal.adapters["https://"], PilotBudgetAdapter)
    assert budget_summary(ledger)["attempts"] == 0


def test_real_reflection_cache_hit_does_not_spend_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server
) -> None:
    endpoint, handler = server
    path = _create(tmp_path / "reflection-budget.sqlite3", endpoint=endpoint)
    monkeypatch.setenv(BUDGET_ENV, str(path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-fixture")
    monkeypatch.setattr(llm_judge_client, "OPENROUTER_CHAT_COMPLETIONS_URL", endpoint)
    handler.replies = [
        (200, b'{"choices":[{"message":{"content":"suffix"}}],"usage":{"cost":0.125}}')
    ]
    cfg = LLMClientConfig(provider="openrouter", model="fixture/model", requests_per_minute=0)
    with SynthResponseCache(tmp_path / "reflection.sqlite3") as cache:
        reflection = build_openrouter_reflection_lm(cfg=cfg, cache=cache, max_retries=0)
        assert reflection("exact reflection prompt") == "suffix"
        assert reflection("exact reflection prompt") == "suffix"
        assert reflection.paid_cache_misses == 1
    assert len(handler.seen) == 1
    assert json.loads(handler.seen[0]) == {
        "model": "fixture/model",
        "messages": [{"role": "user", "content": "exact reflection prompt"}],
    }
    assert budget_summary(path)["committed_nanos"] == 125_000_000


def test_async_transport_enforces_and_settles_without_changing_body(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BUDGET_ENV, str(ledger))
    calls = []

    async def handle(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"usage": {"cost": 0.2}})

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), event_hooks=async_budget_hooks()
        ) as client:
            await client.post(_ENDPOINT, json=_BODY)
            with pytest.raises(ValueError, match="JSON POST"):
                await client.get(_ENDPOINT)

    asyncio.run(exercise())
    assert calls == [_BODY]
    assert budget_summary(ledger)["attempts"] == 1
    assert budget_summary(ledger)["committed_nanos"] == 200_000_000


@pytest.mark.parametrize("payload", [None, [], "not-an-object"])
def test_nonobject_request_never_reserves(ledger: Path, payload: object) -> None:
    with pytest.raises(ValueError):
        _reserve(ledger, body=payload)
    assert budget_summary(ledger)["attempts"] == 0


def test_money_above_sqlite_integer_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer range"):
        _money("1000000000000000000000000000000")


def test_sync_non_post_or_streamed_body_never_reserves(ledger: Path) -> None:
    adapter = PilotBudgetAdapter(str(ledger))
    for request in (
        requests.Request("GET", _ENDPOINT).prepare(),
        requests.Request("POST", _ENDPOINT, data=iter([b"chunk"])).prepare(),
    ):
        with pytest.raises(ValueError, match="JSON POST"):
            adapter.send(request)
    assert budget_summary(ledger)["attempts"] == 0


@pytest.mark.parametrize(
    "response", [b'{"usage":{"cost":1,"cost":0}}', b'{"usage":{"cost":0,"extra":NaN}}']
)
def test_ambiguous_or_nonstandard_json_receipt_halts_without_refund(
    ledger: Path, response: bytes
) -> None:
    attempt = _reserve(ledger)
    with pytest.raises(PilotBudgetExceeded, match="valid cost receipt"):
        settle_attempt(ledger, attempt, status_code=200, response=response)
    assert budget_summary(ledger)["committed_nanos"] == _NANO
    assert budget_summary(ledger)["halted"] == 1


def test_ambiguous_request_never_reaches_reservation(ledger: Path) -> None:
    payload = b'{"model":"other",' + json.dumps(_BODY).encode()[1:]
    with pytest.raises(ValueError, match="duplicate JSON"):
        reserve_attempt(ledger, endpoint=_ENDPOINT, payload=payload)
    assert budget_summary(ledger)["attempts"] == 0
