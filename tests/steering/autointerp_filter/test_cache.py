from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.autointerp_filter.cache import IntruderTrialCache, TokenExampleCache
from steering.eval.autointerp_filter.types import (
    SCHEMA_VERSION,
    AutoInterpCacheError,
    IntruderTrialResult,
    TokenExampleSet,
    TokenWindowRef,
)


def _ref(*, token: str = "tok") -> TokenWindowRef:
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=0,
        source_text_hash="hash",
        token_start=0,
        token_end=2,
        token_ids=(1, 2),
        token_strings=(token, " suffix"),
        target_feature_ids=(7,),
        activations=(0.0, 3.0),
        max_activation=3.0,
        quantile=0,
        active=True,
    )


def _examples(*, token: str = "tok") -> TokenExampleSet:
    return TokenExampleSet(
        positives=(_ref(token=token),),
        non_activating=(
            TokenWindowRef(
                shard_stem="shard_00000_features",
                shard_index=0,
                doc_id=1,
                source_text_hash="hash-1",
                token_start=0,
                token_end=2,
                token_ids=(3, 4),
                token_strings=("quiet", " text"),
                target_feature_ids=(7,),
                activations=(0.0, 0.0),
                max_activation=0.0,
                quantile=None,
                active=False,
            ),
        ),
    )


def _result(
    *,
    trial_key: str = "trial",
    prediction: int | None = 1,
    correct_index: int = 1,
    correct: bool = True,
    response: str | None = "[RESPONSE]: 1",
    error_message: str | None = None,
) -> IntruderTrialResult:
    return IntruderTrialResult(
        trial_key=trial_key,
        prediction=prediction,
        correct_index=correct_index,
        correct=correct,
        response=response,
        cached=False,
        error_message=error_message,
    )


def _write_metadata(path: Path, *, schema_version: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
        (schema_version,),
    )
    conn.commit()
    conn.close()


def test_token_example_cache_roundtrips_namespaced_rows_and_replaces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "token_examples.sqlite"

    with TokenExampleCache(path) as cache:
        assert cache.get(namespace="ns-a", target_key="feature:7") is None
        cache.set(
            namespace="ns-a",
            target_key="feature:7",
            examples=_examples(token="cafe"),
        )
        assert cache.get(namespace="ns-a", target_key="feature:7") == _examples(token="cafe")
        assert cache.get(namespace="ns-b", target_key="feature:7") is None
        assert cache.get(namespace="ns-a", target_key="feature:8") is None

        cache.set(
            namespace="ns-a",
            target_key="feature:7",
            examples=_examples(token="naive"),
        )
        assert cache.get(namespace="ns-a", target_key="feature:7") == _examples(token="naive")

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone() == (
        str(SCHEMA_VERSION),
    )
    assert conn.execute("SELECT count(*) FROM token_example_sets").fetchone() == (1,)
    conn.close()


def test_token_example_cache_get_many_preserves_independent_namespaces(
    tmp_path: Path,
) -> None:
    path = tmp_path / "token_examples.sqlite"
    with TokenExampleCache(path) as cache:
        cache.set(namespace="ns-a", target_key="feature:7", examples=_examples(token="a"))
        cache.set(namespace="ns-b", target_key="feature:7", examples=_examples(token="b"))
        cache.set(namespace="ns-a", target_key="feature:8", examples=_examples(token="c"))

        rows = cache.get_many(
            [
                ("ns-a", "feature:8"),
                ("missing", "feature:7"),
                ("ns-b", "feature:7"),
            ]
        )

    assert rows == {
        ("ns-a", "feature:8"): _examples(token="c"),
        ("ns-b", "feature:7"): _examples(token="b"),
    }


def test_token_example_cache_get_many_rejects_duplicate_keys(tmp_path: Path) -> None:
    with TokenExampleCache(tmp_path / "token_examples.sqlite") as cache:
        with pytest.raises(AutoInterpCacheError, match="get_many keys contain duplicates"):
            cache.get_many([("ns", "feature:7"), ("ns", "feature:7")])


def test_token_example_cache_set_many_is_one_atomic_transaction(tmp_path: Path) -> None:
    path = tmp_path / "token_examples.sqlite"
    with TokenExampleCache(path) as cache:
        cache.set(namespace="ns", target_key="feature:7", examples=_examples(token="old"))
        cache.set_many(
            [
                ("ns", "feature:7", _examples(token="new")),
                ("ns", "feature:8", _examples(token="second")),
                ("other", "feature:7", _examples(token="third")),
            ]
        )
        assert cache.get(namespace="ns", target_key="feature:7") == _examples(token="new")
        assert cache.get(namespace="ns", target_key="feature:8") == _examples(token="second")
        assert cache.get(namespace="other", target_key="feature:7") == _examples(token="third")

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT count(*) FROM token_example_sets").fetchone() == (3,)
    conn.close()


def test_token_example_cache_set_many_rejects_empty_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    with TokenExampleCache(tmp_path / "token_examples.sqlite") as cache:
        with pytest.raises(AutoInterpCacheError, match="requires at least one row"):
            cache.set_many([])
        with pytest.raises(AutoInterpCacheError, match="duplicate key"):
            cache.set_many(
                [
                    ("ns", "feature:7", _examples(token="first")),
                    ("ns", "feature:7", _examples(token="second")),
                ]
            )
        assert cache.get(namespace="ns", target_key="feature:7") is None


def test_token_example_cache_set_many_rolls_back_every_row_on_sqlite_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "token_examples.sqlite"
    with TokenExampleCache(path) as cache:
        cache.set(namespace="ns", target_key="feature:7", examples=_examples(token="old"))
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TRIGGER reject_feature_8 BEFORE INSERT ON token_example_sets "
            "WHEN NEW.target_key = 'feature:8' BEGIN "
            "SELECT RAISE(ABORT, 'injected block failure'); END"
        )
        conn.commit()
        conn.close()

        with pytest.raises(sqlite3.IntegrityError, match="injected block failure"):
            cache.set_many(
                [
                    ("ns", "feature:7", _examples(token="must-roll-back")),
                    ("ns", "feature:8", _examples(token="rejected")),
                ]
            )

        assert cache.get(namespace="ns", target_key="feature:7") == _examples(token="old")
        assert cache.get(namespace="ns", target_key="feature:8") is None

        # The explicit rollback also leaves the connection usable immediately.
        cache.set(namespace="ns", target_key="feature:9", examples=_examples(token="after"))
        assert cache.get(namespace="ns", target_key="feature:9") == _examples(token="after")


def test_token_example_cache_set_many_serializes_before_opening_transaction(
    tmp_path: Path,
) -> None:
    class _BrokenExamples:
        def to_json(self) -> dict:
            raise RuntimeError("injected serialization failure")

    path = tmp_path / "token_examples.sqlite"
    with TokenExampleCache(path) as cache:
        cache.set(namespace="ns", target_key="feature:7", examples=_examples(token="old"))
        with pytest.raises(RuntimeError, match="injected serialization failure"):
            cache.set_many(
                [
                    ("ns", "feature:7", _examples(token="not-written")),
                    ("ns", "feature:8", cast(Any, _BrokenExamples())),
                ]
            )
        assert cache.get(namespace="ns", target_key="feature:7") == _examples(token="old")
        assert cache.get(namespace="ns", target_key="feature:8") is None


def test_token_example_cache_rejects_metadata_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "stale_token_examples.sqlite"
    _write_metadata(path, schema_version="1")

    with pytest.raises(AutoInterpCacheError, match="schema_version=1"):
        TokenExampleCache(path)


def test_token_example_cache_rejects_corrupt_payloads(tmp_path: Path) -> None:
    path = tmp_path / "token_examples.sqlite"
    with TokenExampleCache(path) as cache:
        cache.set(namespace="ns", target_key="feature:7", examples=_examples())

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE token_example_sets SET payload = ? WHERE namespace = 'ns'",
        ("{bad json",),
    )
    conn.commit()
    conn.close()
    with TokenExampleCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="invalid JSON"):
            cache.get(namespace="ns", target_key="feature:7")

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE token_example_sets SET payload = ? WHERE namespace = 'ns'",
        ("[]",),
    )
    conn.commit()
    conn.close()
    with TokenExampleCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="not a JSON object"):
            cache.get(namespace="ns", target_key="feature:7")

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE token_example_sets SET payload = ? WHERE namespace = 'ns'",
        ('{"schema_version": 1, "positives": [], "non_activating": []}',),
    )
    conn.commit()
    conn.close()
    with TokenExampleCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="payload schema_version=1"):
            cache.get(namespace="ns", target_key="feature:7")

    bad_window = _examples().to_json()
    del bad_window["positives"][0]["token_ids"]
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE token_example_sets SET payload = ? WHERE namespace = 'ns'",
        (json.dumps(bad_window),),
    )
    conn.commit()
    conn.close()
    with TokenExampleCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="missing fields"):
            cache.get(namespace="ns", target_key="feature:7")


def test_token_example_cache_translates_invalid_record_domains(tmp_path: Path) -> None:
    path = tmp_path / "token_examples.sqlite"
    with TokenExampleCache(path) as cache:
        cache.set(namespace="ns", target_key="feature:7", examples=_examples())

    payload = _examples().to_json()
    payload["positives"][0]["activations"] = [0.0, float("nan")]
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE token_example_sets SET payload = ? WHERE namespace = 'ns'",
        (json.dumps(payload),),
    )
    conn.commit()
    conn.close()

    with TokenExampleCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="invalid data"):
            cache.get(namespace="ns", target_key="feature:7")


def test_cache_direct_keys_are_beartyped_non_empty_strings(tmp_path: Path) -> None:
    with TokenExampleCache(tmp_path / "tokens.sqlite") as cache:
        with pytest.raises(BeartypeCallHintParamViolation, match="namespace"):
            cache.get(namespace="", target_key="feature:7")
        with pytest.raises(BeartypeCallHintParamViolation, match="target_key"):
            cache.set(namespace="ns", target_key="", examples=_examples())

    with IntruderTrialCache(tmp_path / "trials.sqlite") as cache:
        with pytest.raises(BeartypeCallHintParamViolation, match="trial_key"):
            cache.get(trial_key="", target_key="feature:7", judge_key="judge")
        with pytest.raises(BeartypeCallHintParamViolation, match="judge_key"):
            cache.set(target_key="feature:7", judge_key="", result=_result())


def test_intruder_trial_cache_roundtrips_keys_and_replaces(tmp_path: Path) -> None:
    path = tmp_path / "trial_results.sqlite"

    with IntruderTrialCache(path) as cache:
        assert (
            cache.get(
                trial_key="trial",
                target_key="feature:7",
                judge_key="judge-a",
            )
            is None
        )
        cache.set(
            target_key="feature:7",
            judge_key="judge-a",
            result=_result(prediction=1, correct_index=1, response="[RESPONSE]: 1"),
        )
        cache.set(
            target_key="feature:7",
            judge_key="judge-b",
            result=_result(prediction=0, correct_index=0, response="[RESPONSE]: 0"),
        )
        cache.set(
            target_key="feature:8",
            judge_key="judge-a",
            result=_result(prediction=0, correct_index=0, response="[RESPONSE]: 0"),
        )

        assert cache.get(
            trial_key="trial",
            target_key="feature:7",
            judge_key="judge-a",
        ) == IntruderTrialResult(
            trial_key="trial",
            prediction=1,
            correct_index=1,
            correct=True,
            response="[RESPONSE]: 1",
            cached=True,
        )
        assert cache.get(
            trial_key="trial",
            target_key="feature:7",
            judge_key="judge-b",
        ) == IntruderTrialResult(
            trial_key="trial",
            prediction=0,
            correct_index=0,
            correct=True,
            response="[RESPONSE]: 0",
            cached=True,
        )
        assert (
            cache.get(
                trial_key="trial",
                target_key="feature:8",
                judge_key="judge-a",
            )
            is not None
        )

        cache.set(
            target_key="feature:7",
            judge_key="judge-a",
            result=_result(
                prediction=2,
                correct_index=1,
                correct=False,
                response="[RESPONSE]: 2",
            ),
        )
        replaced = cache.get(
            trial_key="trial",
            target_key="feature:7",
            judge_key="judge-a",
        )
        assert replaced == IntruderTrialResult(
            trial_key="trial",
            prediction=2,
            correct_index=1,
            correct=False,
            response="[RESPONSE]: 2",
            cached=True,
        )

    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone() == (
        str(SCHEMA_VERSION),
    )
    assert conn.execute("SELECT count(*) FROM trial_results").fetchone() == (3,)
    conn.close()


def test_intruder_trial_cache_serializes_concurrent_access(tmp_path: Path) -> None:
    path = tmp_path / "trial_results.sqlite"
    n_threads = 12
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    with IntruderTrialCache(path) as cache:

        def worker(index: int) -> None:
            try:
                barrier.wait(timeout=2.0)
                result = _result(
                    trial_key=f"trial-{index}",
                    prediction=index % 3,
                    correct_index=index % 3,
                    response=f"[RESPONSE]: {index % 3}",
                )
                cache.set(target_key="feature:7", judge_key="judge", result=result)
                cached = cache.get(
                    trial_key=f"trial-{index}",
                    target_key="feature:7",
                    judge_key="judge",
                )
                assert cached == IntruderTrialResult(
                    trial_key=f"trial-{index}",
                    prediction=index % 3,
                    correct_index=index % 3,
                    correct=True,
                    response=f"[RESPONSE]: {index % 3}",
                    cached=True,
                )
            except BaseException as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(n_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT count(*) FROM trial_results").fetchone() == (n_threads,)
    conn.close()


def test_intruder_trial_cache_rejects_metadata_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "stale_trial_results.sqlite"
    _write_metadata(path, schema_version="1")

    with pytest.raises(AutoInterpCacheError, match="schema_version=1"):
        IntruderTrialCache(path)


def test_intruder_trial_cache_rejects_poisoned_rows(tmp_path: Path) -> None:
    path = tmp_path / "trial_results.sqlite"
    with IntruderTrialCache(path) as cache:
        cache.set(target_key="feature:7", judge_key="judge", result=_result())
        cache.set(
            target_key="feature:7",
            judge_key="judge",
            result=_result(
                trial_key="parse-error",
                prediction=None,
                correct=False,
                response="not parseable",
                error_message="missing response marker",
            ),
        )
        cache.set(
            target_key="feature:7",
            judge_key="judge",
            result=_result(
                trial_key="missing-prediction",
                prediction=None,
                correct=False,
                response=None,
            ),
        )

    with IntruderTrialCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="error_message"):
            cache.get(
                trial_key="parse-error",
                target_key="feature:7",
                judge_key="judge",
            )
        with pytest.raises(AutoInterpCacheError, match="missing a parsed prediction"):
            cache.get(
                trial_key="missing-prediction",
                target_key="feature:7",
                judge_key="judge",
            )

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE trial_results SET correct = 2 "
        "WHERE trial_key = 'trial' AND target_key = 'feature:7'"
    )
    conn.commit()
    conn.close()
    with IntruderTrialCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="invalid correct flag"):
            cache.get(trial_key="trial", target_key="feature:7", judge_key="judge")

    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE trial_results SET correct = 1, correct_index = -1 "
        "WHERE trial_key = 'trial' AND target_key = 'feature:7'"
    )
    conn.commit()
    conn.close()
    with IntruderTrialCache(path) as cache:
        with pytest.raises(AutoInterpCacheError, match="invalid data"):
            cache.get(trial_key="trial", target_key="feature:7", judge_key="judge")
