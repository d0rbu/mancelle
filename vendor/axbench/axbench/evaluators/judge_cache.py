"""Small SQLite cache for exact AxBench judge requests."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any

_REQUEST_FIELDS = frozenset({"version", "endpoint", "model", "messages", "temperature"})
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS judge_responses (
    request_sha256 BLOB PRIMARY KEY,
    request_json TEXT NOT NULL,
    response TEXT NOT NULL
) WITHOUT ROWID
"""

DigestFn = Callable[[bytes], bytes]


def _default_digest(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def _canonical_request_json(request: Mapping[str, object]) -> str:
    if not isinstance(request, Mapping):
        raise TypeError("judge request must be a mapping")
    if not all(isinstance(key, str) for key in request):
        raise TypeError("judge request keys must be strings")

    missing = sorted(_REQUEST_FIELDS - set(request))
    extra = sorted(set(request) - _REQUEST_FIELDS)
    if missing or extra:
        raise ValueError(f"judge request fields do not match; missing={missing}, extra={extra}")

    version = request["version"]
    if isinstance(version, bool) or not isinstance(version, (int, str)):
        raise TypeError("judge request version must be an integer or string")
    if (isinstance(version, int) and version < 1) or (
        isinstance(version, str) and not version.strip()
    ):
        raise ValueError("judge request version must be positive or nonblank")

    endpoint = request["endpoint"]
    if not isinstance(endpoint, str):
        raise TypeError("judge request endpoint must be a string")
    normalized_endpoint = endpoint.strip().rstrip("/")
    if not normalized_endpoint:
        raise ValueError("judge request endpoint must be nonblank")

    model = request["model"]
    if not isinstance(model, str):
        raise TypeError("judge request model must be a string")
    if not model.strip():
        raise ValueError("judge request model must be nonblank")

    messages = request["messages"]
    if not isinstance(messages, list):
        raise TypeError("judge request messages must be a list")
    if not messages or not all(isinstance(message, dict) for message in messages):
        raise ValueError("judge request messages must be a nonempty list of objects")

    temperature = request["temperature"]
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("judge request temperature must be numeric")
    normalized_temperature = float(temperature)
    if not math.isfinite(normalized_temperature):
        raise ValueError("judge request temperature must be finite")

    normalized: dict[str, Any] = {
        "version": version,
        "endpoint": normalized_endpoint,
        "model": model,
        "messages": messages,
        "temperature": normalized_temperature,
    }
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class JudgeCache:
    """Context-managed cache with collision-checked, first-writer-wins inserts."""

    def __init__(
        self,
        path: Path,
        cache_only: bool = False,
        *,
        digest_fn: DigestFn | None = None,
    ) -> None:
        self.path = Path(path)
        self.cache_only = cache_only
        self._digest_fn = digest_fn or _default_digest
        self._connection: sqlite3.Connection | None = None

        if cache_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            self._connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                self.path,
                timeout=30.0,
                isolation_level=None,
            )
            self._connection.execute(_CREATE_TABLE)

    def __enter__(self) -> JudgeCache:
        self._require_connection()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("judge cache is closed")
        return self._connection

    def _request_key(self, request: Mapping[str, object]) -> tuple[bytes, str]:
        request_json = _canonical_request_json(request)
        digest = self._digest_fn(request_json.encode("utf-8"))
        if not isinstance(digest, bytes) or len(digest) != hashlib.sha256().digest_size:
            raise ValueError("judge cache digest function must return exactly 32 bytes")
        return digest, request_json

    @staticmethod
    def _checked_response(
        row: tuple[str, str] | None,
        request_json: str,
    ) -> str | None:
        if row is None:
            return None
        stored_request_json, response = row
        if stored_request_json != request_json:
            raise ValueError("judge cache SHA-256 collision")
        if not isinstance(response, str):
            raise ValueError("judge cache stored response is not text")
        return response

    def get(self, request: Mapping[str, object]) -> str | None:
        """Return the cached response, or ``None`` for a cache miss."""

        digest, request_json = self._request_key(request)
        row = self._require_connection().execute(
            "SELECT request_json, response FROM judge_responses WHERE request_sha256 = ?",
            (digest,),
        ).fetchone()
        return self._checked_response(row, request_json)

    def put_if_absent(self, request: Mapping[str, object], response: str) -> str:
        """Store a response if absent and return the existing or inserted value."""

        if self.cache_only:
            raise PermissionError("cache-only judge cache is read-only")
        if not isinstance(response, str):
            raise TypeError("judge response must be a string")

        digest, request_json = self._request_key(request)
        connection = self._require_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO judge_responses(request_sha256, request_json, response)
                VALUES (?, ?, ?)
                """,
                (digest, request_json, response),
            )
            row = connection.execute(
                "SELECT request_json, response FROM judge_responses WHERE request_sha256 = ?",
                (digest,),
            ).fetchone()
            stored = self._checked_response(row, request_json)
            if stored is None:
                raise RuntimeError("judge cache insert completed without a stored row")
            connection.execute("COMMIT")
            return stored
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
