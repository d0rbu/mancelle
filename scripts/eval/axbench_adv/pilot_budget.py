"""Opt-in pilot spending fence at the actual HTTP-attempt boundary.

Both Python environments share one operator-created local SQLite ledger. Each
attempt reserves a conservatively priced *whole request* before network I/O.
Only a successful response's explicit usage.cost can release that reservation;
timeouts, rejected responses, missing cost, and process death retain it. A 200
response without a valid cost receipt also halts the ledger for inspection. This
does not change model requests, cache identities, provider routes, or key limits.
The operator must verify the per-request bound against current model pricing,
context/output limits, and the supported plain-text request shape before launch.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any

import httpx
import requests
from requests.adapters import HTTPAdapter

BUDGET_ENV = "AXBENCH_ADV_PILOT_BUDGET_FILE"
_ALLOWED_FIELDS = frozenset(
    {"model", "messages", "temperature", "max_tokens", "max_completion_tokens", "stream"}
)


class PilotBudgetExceeded(RuntimeError):
    """No request may be sent without an affordable reservation."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("ambiguous duplicate JSON key in pilot request/receipt")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in pilot request/receipt: {value}")


def _parse_json(payload: bytes) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_object,
        parse_constant=_nonfinite,
        parse_float=Decimal,
    )


def _money(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("pilot money must be a finite nonnegative decimal")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            raise ValueError("pilot money must be a finite nonnegative decimal")
        if number.is_zero():
            return 0
        adjusted = number.adjusted()
        if adjusted > 9:
            raise ValueError("pilot money exceeds the ledger's integer range")
        if adjusted < -9:
            return 1
        numerator, denominator = number.as_integer_ratio()
        nanos = (numerator * 1_000_000_000 + denominator - 1) // denominator
    except DecimalException as exc:
        raise ValueError("pilot money must be a finite nonnegative decimal") from exc
    if nanos > 2**63 - 1:
        raise ValueError("pilot money exceeds the ledger's integer range")
    return int(nanos)


def create_pilot_budget(
    path: Path,
    *,
    limit_usd: str,
    request_bound_usd: str,
    model: str,
    endpoint: str,
    max_output_tokens: int,
    pricing_evidence: str,
) -> None:
    """Create a fresh ledger; never reset a previous pilot's spending."""

    limit, bound = _money(limit_usd), _money(request_bound_usd)
    if not 0 < bound <= limit:
        raise ValueError("pilot requires 0 < request bound <= total limit")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (model, endpoint, pricing_evidence)
    ):
        raise ValueError("pilot model, endpoint, and pricing evidence must be nonblank")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("pilot output bound must be a positive integer")
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("pilot ledger path must be absolute across process working directories")
    # Exclusive creation protects even an empty or partial previous ledger.
    with path.open("xb"):
        pass
    with _connection(path) as connection:
        connection.execute(
            "CREATE TABLE budget (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
            "limit_nanos INTEGER NOT NULL, bound_nanos INTEGER NOT NULL, model TEXT NOT NULL, "
            "endpoint TEXT NOT NULL, max_output_tokens INTEGER NOT NULL, pricing_evidence TEXT NOT NULL, "
            "halted INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute(
            "INSERT INTO budget VALUES (1, ?, ?, ?, ?, ?, ?, 0)",
            (limit, bound, model, endpoint, max_output_tokens, pricing_evidence),
        )
        connection.execute(
            "CREATE TABLE attempts (id INTEGER PRIMARY KEY, reserved_nanos INTEGER NOT NULL, "
            "actual_nanos INTEGER, settled INTEGER NOT NULL DEFAULT 0)"
        )


@contextmanager
def _connection(path: str | Path) -> Iterator[sqlite3.Connection]:
    # Missing paths must not silently create an unrestricted replacement.
    if not Path(path).is_absolute():
        raise ValueError("pilot ledger path must be absolute across process working directories")
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=rw", uri=True, timeout=30)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def reserve_attempt(path: str | Path, *, endpoint: str, payload: bytes) -> int:
    """Atomically reserve before sending; retries each need a new reservation."""

    body = _parse_json(payload)
    if (
        not isinstance(body, dict)
        or set(body) - _ALLOWED_FIELDS
        or body.get("stream", False) is not False
    ):
        raise ValueError("pilot budget supports only non-streaming plain-text completions")
    messages = body.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
            for message in messages
        )
    ):
        raise ValueError("pilot budget requires plain-text role/content messages")
    with _connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        limit, bound, model, expected_endpoint, output_limit, halted = connection.execute(
            "SELECT limit_nanos, bound_nanos, model, endpoint, max_output_tokens, halted FROM budget WHERE singleton=1"
        ).fetchone()
        if halted:
            raise PilotBudgetExceeded("pilot ledger is halted; inspect it before any further calls")
        if body.get("model") != model or endpoint != expected_endpoint:
            raise ValueError("pilot request differs from the priced model/endpoint")
        for key in ("max_tokens", "max_completion_tokens"):
            if key in body and (type(body[key]) is not int or not 0 < body[key] <= output_limit):
                raise ValueError("pilot output request exceeds its priced bound")
        committed = connection.execute(
            "SELECT COALESCE(SUM(COALESCE(actual_nanos, reserved_nanos)), 0) FROM attempts"
        ).fetchone()[0]
        if committed + bound > limit:
            raise PilotBudgetExceeded(
                "pilot budget cannot cover another complete request; no request sent"
            )
        cursor = connection.execute("INSERT INTO attempts(reserved_nanos) VALUES (?)", (bound,))
        attempt_id = cursor.lastrowid
        assert attempt_id is not None
        return attempt_id


def settle_attempt(path: str | Path, attempt_id: int, *, status_code: int, response: bytes) -> None:
    """Release unused funds only for an explicit, valid successful cost receipt."""

    actual = None
    if status_code == 200:
        try:
            body = _parse_json(response)
            if "error" not in body:
                actual = _money(body["usage"]["cost"])
        except (DecimalException, ValueError, TypeError, KeyError):
            pass
    failure = (
        "successful provider response lacked a valid cost receipt; ledger halted"
        if status_code == 200 and actual is None
        else None
    )
    with _connection(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT reserved_nanos, settled FROM attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if row is None or row[1]:
            raise RuntimeError("pilot attempt missing or already settled")
        connection.execute(
            "UPDATE attempts SET actual_nanos=?, settled=1 WHERE id=?", (actual, attempt_id)
        )
        if actual is not None and actual > row[0]:
            failure = "provider cost exceeded the verified request bound; ledger halted"
        if failure is not None:
            connection.execute("UPDATE budget SET halted=1 WHERE singleton=1")
    if failure is not None:
        raise PilotBudgetExceeded(failure)


def budget_summary(path: str | Path) -> dict[str, int]:
    with _connection(path) as connection:
        connection.execute("BEGIN")
        limit, halted = connection.execute(
            "SELECT limit_nanos, halted FROM budget WHERE singleton=1"
        ).fetchone()
        count, actual, committed = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(actual_nanos),0), "
            "COALESCE(SUM(COALESCE(actual_nanos,reserved_nanos)),0) FROM attempts"
        ).fetchone()
    return {
        "limit_nanos": limit,
        "attempts": count,
        "known_actual_nanos": actual,
        "committed_nanos": committed,
        "halted": halted,
    }


def validate_pilot_budget(path: str | Path, *, model: str, endpoint: str) -> None:
    """Bind a launch to the priced model and endpoint before any work starts."""

    with _connection(path) as connection:
        connection.execute("BEGIN")
        row = connection.execute(
            "SELECT model, endpoint, halted FROM budget WHERE singleton=1"
        ).fetchone()
    if row is None:
        raise ValueError("pilot ledger has no singleton budget configuration")
    priced_model, priced_endpoint, halted = row
    if halted:
        raise PilotBudgetExceeded("pilot ledger is halted; inspect it before launch")
    if model != priced_model or endpoint != priced_endpoint:
        raise ValueError("pilot launch model/endpoint differs from its priced ledger")


class PilotBudgetAdapter(HTTPAdapter):
    """The root requests client has no hidden adapter-level retries."""

    def __init__(self, budget_path: str, **kwargs: Any) -> None:
        super().__init__(max_retries=0, **kwargs)
        self.budget_path = budget_path

    def send(
        self, request: requests.PreparedRequest, *args: Any, **kwargs: Any
    ) -> requests.Response:
        if request.method != "POST" or not isinstance(request.body, (str, bytes)):
            raise ValueError("pilot budget requires a JSON POST request")
        payload = request.body.encode("utf-8") if isinstance(request.body, str) else request.body
        attempt = reserve_attempt(self.budget_path, endpoint=str(request.url), payload=payload)
        response = super().send(request, *args, **kwargs)
        settle_attempt(
            self.budget_path, attempt, status_code=response.status_code, response=response.content
        )
        return response


def async_budget_hooks() -> dict[str, list[Any]]:
    """Hook every HTTPX attempt, including OpenAI SDK and judge retries."""

    path = os.environ[BUDGET_ENV]

    async def before(request: httpx.Request) -> None:
        if request.method != "POST":
            raise ValueError("pilot budget requires a JSON POST request")
        request.extensions["axbench_pilot_attempt"] = reserve_attempt(
            path,
            endpoint=str(request.url),
            payload=await request.aread(),
        )

    async def after(response: httpx.Response) -> None:
        attempt = response.request.extensions["axbench_pilot_attempt"]
        settle_attempt(
            path, attempt, status_code=response.status_code, response=await response.aread()
        )

    return {"request": [before], "response": [after]}
