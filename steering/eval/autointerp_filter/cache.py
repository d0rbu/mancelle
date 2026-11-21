"""SQLite caches for token-level intruder-detection autointerp."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    AutoInterpCacheError,
    IntruderTrialResult,
    TokenExampleSet,
)
from steering.refined_types import NonEmptyStrLike


class IntruderTrialCache:
    """Thread-safe SQLite cache keyed by trial identity and target set."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata (  key TEXT PRIMARY KEY,  value TEXT NOT NULL)"
            )
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row[0] != str(SCHEMA_VERSION):
                self._conn.close()
                raise AutoInterpCacheError(f"{path} schema_version={row[0]} != {SCHEMA_VERSION}")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS trial_results ("
                "  trial_key TEXT NOT NULL,"
                "  target_key TEXT NOT NULL,"
                "  judge_key TEXT NOT NULL,"
                "  prediction INTEGER,"
                "  correct_index INTEGER NOT NULL,"
                "  correct INTEGER NOT NULL,"
                "  response TEXT,"
                "  error_message TEXT,"
                "  created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),"
                "  PRIMARY KEY (trial_key, target_key, judge_key)"
                ")"
            )
            self._conn.commit()

    @beartype
    def get(
        self,
        *,
        trial_key: NonEmptyStrLike,
        target_key: NonEmptyStrLike,
        judge_key: NonEmptyStrLike,
    ) -> IntruderTrialResult | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT prediction, correct_index, correct, response, error_message "
                "FROM trial_results WHERE trial_key = ? AND target_key = ? "
                "AND judge_key = ?",
                (trial_key, target_key, judge_key),
            ).fetchone()
        if row is None:
            return None
        if row[4] is not None:
            raise AutoInterpCacheError(
                f"cached intruder trial {trial_key} for target {target_key} "
                f"contains error_message={row[4]!r}"
            )
        if row[0] is None:
            raise AutoInterpCacheError(
                f"cached intruder trial {trial_key} for target {target_key} "
                "is missing a parsed prediction"
            )
        if int(row[2]) not in (0, 1):
            raise AutoInterpCacheError(
                f"cached intruder trial {trial_key} for target {target_key} "
                f"has invalid correct flag {row[2]!r}"
            )
        try:
            return IntruderTrialResult(
                trial_key=trial_key,
                prediction=row[0],
                correct_index=row[1],
                correct=bool(row[2]),
                response=row[3],
                cached=True,
                error_message=row[4],
            )
        except (
            BeartypeCallHintParamViolation,
            OverflowError,
            TypeError,
            ValueError,
        ) as exc:
            raise AutoInterpCacheError(
                f"cached intruder trial {trial_key} for target {target_key} "
                f"contains invalid data: {exc}"
            ) from exc

    @beartype
    def set(
        self,
        *,
        target_key: NonEmptyStrLike,
        judge_key: NonEmptyStrLike,
        result: IntruderTrialResult,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO trial_results ("
                "trial_key, target_key, judge_key, prediction, correct_index, "
                "correct, response, error_message"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.trial_key,
                    target_key,
                    judge_key,
                    result.prediction,
                    result.correct_index,
                    1 if result.correct else 0,
                    result.response,
                    result.error_message,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> IntruderTrialCache:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class TokenExampleCache:
    """Thread-safe SQLite cache for re-forwarded token example sets."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS metadata (  key TEXT PRIMARY KEY,  value TEXT NOT NULL)"
            )
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif row[0] != str(SCHEMA_VERSION):
                self._conn.close()
                raise AutoInterpCacheError(f"{path} schema_version={row[0]} != {SCHEMA_VERSION}")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS token_example_sets ("
                "  namespace TEXT NOT NULL,"
                "  target_key TEXT NOT NULL,"
                "  payload TEXT NOT NULL,"
                "  created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),"
                "  PRIMARY KEY (namespace, target_key)"
                ")"
            )
            self._conn.commit()

    @beartype
    def get(
        self,
        *,
        namespace: NonEmptyStrLike,
        target_key: NonEmptyStrLike,
    ) -> TokenExampleSet | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM token_example_sets WHERE namespace = ? AND target_key = ?",
                (namespace, target_key),
            ).fetchone()
        return self._decode_row(row, target_key=target_key)

    def get_many(
        self,
        keys: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], TokenExampleSet]:
        """Read several independently-namespaced rows under one cache lock."""
        ordered_keys = list(keys)
        if len(ordered_keys) != len(set(ordered_keys)):
            raise AutoInterpCacheError("token example get_many keys contain duplicates")
        rows: dict[tuple[str, str], TokenExampleSet] = {}
        with self._lock:
            for namespace, target_key in ordered_keys:
                row = self._conn.execute(
                    "SELECT payload FROM token_example_sets WHERE namespace = ? AND target_key = ?",
                    (namespace, target_key),
                ).fetchone()
                decoded = self._decode_row(row, target_key=target_key)
                if decoded is not None:
                    rows[(namespace, target_key)] = decoded
        return rows

    @staticmethod
    def _decode_row(
        row: tuple[str] | None,
        *,
        target_key: str,
    ) -> TokenExampleSet | None:
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise AutoInterpCacheError(
                f"token example cache row for {target_key} contains invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise AutoInterpCacheError(
                f"token example cache row for {target_key} is not a JSON object"
            )
        return TokenExampleSet.from_json(payload)

    @beartype
    def set(
        self,
        *,
        namespace: NonEmptyStrLike,
        target_key: NonEmptyStrLike,
        examples: TokenExampleSet,
    ) -> None:
        payload = json.dumps(
            examples.to_json(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO token_example_sets "
                "(namespace, target_key, payload) VALUES (?, ?, ?)",
                (namespace, target_key, payload),
            )
            self._conn.commit()

    def set_many(
        self,
        rows: Iterable[tuple[str, str, TokenExampleSet]],
    ) -> None:
        """Atomically replace a complete deterministic materialization block.

        JSON serialization and duplicate validation happen before the database
        transaction. A SQLite failure rolls the entire block back; callers can
        therefore treat an all-present block as the only valid cache hit state.
        """
        serialized: list[tuple[str, str, str]] = []
        keys: set[tuple[str, str]] = set()
        for namespace, target_key, examples in rows:
            key = (namespace, target_key)
            if key in keys:
                raise AutoInterpCacheError(
                    "token example set_many rows contain duplicate key "
                    f"namespace={namespace!r} target_key={target_key!r}"
                )
            keys.add(key)
            payload = json.dumps(
                examples.to_json(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            serialized.append((namespace, target_key, payload))
        if not serialized:
            raise AutoInterpCacheError("token example set_many requires at least one row")

        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.executemany(
                    "INSERT OR REPLACE INTO token_example_sets "
                    "(namespace, target_key, payload) VALUES (?, ?, ?)",
                    serialized,
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TokenExampleCache:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
