"""Strict IO and torn-write coverage for the Stage 1b checkpoint pair."""

from __future__ import annotations

import copy
import inspect
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_rows,
)
from steering.eval.axbench_base.synth import positive_checkpoint
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.types import AxBenchBaseError


def _selected(
    *,
    seed_idx: int = 3,
    candidate_rank: int = 0,
    prefix_text: str = "checkpoint prefix",
) -> dict[str, Any]:
    return {
        "seed_idx": seed_idx,
        "candidate_rank": candidate_rank,
        "prefix_text": prefix_text,
        "seed_percentile": 0.0,
        "stored_clean": True,
    }


def _source(
    *,
    next_candidate_rank: int,
    buffered_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "clean_cursor": next_candidate_rank,
        "dirty_scored_once": False,
        "clean_scored": next_candidate_rank,
        "dirty_scored": 0,
        "headroom_rejected": 0,
        "next_candidate_rank": next_candidate_rank,
        "exhausted": False,
        "buffered_candidates": list(buffered_candidates or []),
    }


def _document(
    pairs: int = 0,
    *,
    fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = [
        {
            **_selected(seed_idx=3 + sequence, candidate_rank=sequence),
            "acceptance_sequence": sequence,
            "natural_avoid": sequence % 2 == 0,
        }
        for sequence in range(pairs)
    ]
    concept_states = [
        {
            "concept_id": 7,
            "feature_id": 42,
            "source": _source(next_candidate_rank=pairs),
            "tracker": {
                "seeds_drawn": pairs,
                "discarded_rich": 0,
                "discarded_avoid": 0,
                "semantic_rich_attempts": pairs,
                "semantic_avoid_attempts": pairs // 2,
                "natural_avoid_acceptances": (pairs + 1) // 2,
                "termination_reason": None,
                "accepted": accepted,
                "discarded": [],
                "outstanding": [],
            },
        }
    ]
    return {
        "schema_version": positive_checkpoint.CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint
        or {
            "checkpoint_algorithm_version": (
                positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
            ),
            "nested": {"target": 2, "enabled": True, "rate": 0.5},
        },
        "highest_requested_target": 72,
        "accepted_pair_count": pairs,
        "next_chain_index": pairs,
        "next_acceptance_sequence": pairs,
        "concept_states": concept_states,
    }


def _v1_document(
    pairs: int = 0,
    *,
    target: int = 72,
) -> dict[str, Any]:
    document = _document(pairs)
    document["schema_version"] = positive_checkpoint.CHECKPOINT_SCHEMA_VERSION_V1
    document.pop("highest_requested_target")
    document["fingerprint"] = {
        "checkpoint_algorithm_version": (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V1
        ),
        "positive_selection_contract": {
            "contract_version": "positive_target_refill_v3",
            "n_positives_per_concept": target,
        },
        "semantic": {"strict": True, "value": 1.0},
    }
    return document


def _ordinary_pair(sequence: int) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix = "checkpoint prefix"
    common = {
        "concept_id": 7,
        "prefix_text": prefix,
        "output_concept": "test concept",
        "n_prefix_nanochat_tokens": 4,
        "qc_feature_id": 42,
        "qc_accepted": True,
    }
    rich = {
        **common,
        "category": "positive",
        "content_class": "concept_rich",
        "output_text": f"rich output {sequence}",
        "gen_source": "openrouter:test/model",
        "gen_seed": 100 + sequence,
        "qc_seed_percentile": 0.0,
        "qc_output_percentile": 80.0,
        "qc_delta": 80.0,
        "qc_attempts": 1,
        "qc_zero_activation_verified": False,
    }
    natural = sequence % 2 == 0
    avoiding = {
        **common,
        "category": "negative",
        "content_class": "concept_avoiding",
        "output_text": f"avoid output {sequence}",
        "gen_source": (
            "corpus:doc_continuation@0/3"
            if natural
            else "openrouter:test/model"
        ),
        "gen_seed": 200 + sequence,
        "qc_seed_percentile": math.nan,
        "qc_output_percentile": math.nan,
        "qc_delta": math.nan,
        "qc_attempts": 0 if natural else 1,
        "qc_zero_activation_verified": True,
    }
    return rich, avoiding


def _checkpoint_rows(pairs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence in range(pairs):
        rich, avoiding = _ordinary_pair(sequence)
        internal = {
            "checkpoint_seed_idx": 3 + sequence,
            "checkpoint_candidate_rank": sequence,
            "checkpoint_acceptance_sequence": sequence,
        }
        rows.extend(({**rich, **internal}, {**avoiding, **internal}))
    return rows


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "stage1b_partial.json", tmp_path / "stage1b_partial.parquet"


def _load_checkpoint(
    *,
    json_path: Path,
    parquet_path: Path,
    expected_fingerprint: dict[str, Any],
    training_schema: pa.Schema,
    current_target: int = 72,
    expected_v1_fingerprint: dict[str, Any] | None = None,
    legacy_fingerprint_factory: Any | None = None,
):
    """Call the strict production loader with an explicit target policy."""

    return positive_checkpoint.load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=expected_fingerprint,
        training_schema=training_schema,
        current_target=current_target,
        legacy_fingerprint_factory=(
            (
                lambda _version: (
                    expected_fingerprint
                    if expected_v1_fingerprint is None
                    else expected_v1_fingerprint
                )
            )
            if legacy_fingerprint_factory is None
            else legacy_fingerprint_factory
        ),
    )


def test_load_checkpoint_requires_explicit_target_policy_inputs() -> None:
    parameters = inspect.signature(positive_checkpoint.load_checkpoint).parameters

    assert parameters["current_target"].default is inspect.Parameter.empty
    assert (
        parameters["legacy_fingerprint_factory"].default
        is inspect.Parameter.empty
    )


def _write_direct_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(
        rows,
        schema=positive_checkpoint.checkpoint_schema(TRAINING_SCHEMA),
    )
    pq.write_table(table, path)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _rows_equal_with_nan(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> bool:
    if len(actual) != len(expected):
        return False
    for actual_row, expected_row in zip(actual, expected, strict=True):
        if actual_row.keys() != expected_row.keys():
            return False
        for key, actual_value in actual_row.items():
            expected_value = expected_row[key]
            if (
                isinstance(actual_value, float)
                and isinstance(expected_value, float)
                and math.isnan(actual_value)
                and math.isnan(expected_value)
            ):
                continue
            if actual_value != expected_value:
                return False
    return True


@pytest.mark.parametrize("pairs", [0, 1, 3])
def test_checkpoint_round_trip_and_exact_schema(tmp_path: Path, pairs: int) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document(pairs)
    rows = _checkpoint_rows(pairs)

    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=rows,
        training_schema=TRAINING_SCHEMA,
    )

    assert json_path.read_bytes() == json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert pq.read_schema(parquet_path).equals(
        positive_checkpoint.checkpoint_schema(TRAINING_SCHEMA),
        check_metadata=True,
    )
    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=document["fingerprint"],
        training_schema=TRAINING_SCHEMA,
    )
    assert loaded is not None
    loaded_document, loaded_rows = loaded
    assert loaded_document == document
    assert len(loaded_rows) == 2 * pairs
    assert _rows_equal_with_nan(
        positive_checkpoint.ordinary_rows(loaded_rows),
        [
            row
            for pair in (
                _ordinary_pair(sequence) for sequence in range(pairs)
            )
            for row in pair
        ],
    )


def test_checkpoint_publication_is_parquet_durable_before_json_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    events: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        positive_checkpoint,
        "atomic_write_parquet",
        lambda _table, path: events.append(("write_parquet", Path(path))),
    )
    monkeypatch.setattr(
        positive_checkpoint,
        "atomic_write_bytes",
        lambda _data, path: events.append(("write_json", Path(path))),
    )
    monkeypatch.setattr(
        positive_checkpoint,
        "_fsync_parent_directory",
        lambda path: events.append(("fsync_parent", path)),
    )

    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=_document(),
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )

    assert events == [
        ("write_parquet", parquet_path),
        ("fsync_parent", parquet_path),
        ("write_json", json_path),
        ("fsync_parent", json_path),
    ]


def test_json_publication_failure_preserves_previous_committed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    committed = _document(1)
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=committed,
        rows=_checkpoint_rows(1),
        training_schema=TRAINING_SCHEMA,
    )

    def fail_json(_data: bytes, _path: Path) -> None:
        raise OSError("synthetic JSON publication failure")

    monkeypatch.setattr(positive_checkpoint, "atomic_write_bytes", fail_json)
    with pytest.raises(OSError, match="JSON publication failure"):
        positive_checkpoint.write_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            document=_document(2),
            rows=_checkpoint_rows(2),
            training_schema=TRAINING_SCHEMA,
        )

    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=committed["fingerprint"],
        training_schema=TRAINING_SCHEMA,
    )
    assert loaded is not None
    assert loaded[0] == committed
    assert len(loaded[1]) == 2


def test_checkpoint_preserves_interpolated_qc_arithmetic_exactly(
    tmp_path: Path,
) -> None:
    """Checkpoint QC fields stay float64 until final-artifact publication."""

    json_path, parquet_path = _paths(tmp_path)
    document = _document(1)
    seed_percentile = 12.345678901234
    output_percentile = 66.659071725697
    delta = output_percentile - seed_percentile
    document["concept_states"][0]["tracker"]["accepted"][0][
        "seed_percentile"
    ] = seed_percentile
    rows = _checkpoint_rows(1)
    rows[0].update(
        qc_seed_percentile=seed_percentile,
        qc_output_percentile=output_percentile,
        qc_delta=delta,
    )

    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=rows,
        training_schema=TRAINING_SCHEMA,
    )
    schema = pq.read_schema(parquet_path)
    for name in (
        "qc_seed_percentile",
        "qc_output_percentile",
        "qc_delta",
    ):
        assert TRAINING_SCHEMA.field(name).type == pa.float32()
        assert schema.field(name).type == pa.float64()

    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=document["fingerprint"],
        training_schema=TRAINING_SCHEMA,
    )
    assert loaded is not None
    ordinary = positive_checkpoint.ordinary_rows(loaded[1])
    rich = ordinary[0]
    assert rich["qc_seed_percentile"] == seed_percentile
    assert rich["qc_output_percentile"] == output_percentile
    assert rich["qc_delta"] == delta
    validate_positive_qc_rows(ordinary, accept_threshold_points=30.0)


def test_orphan_parquet_is_ignored(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    _write_direct_parquet(parquet_path, _checkpoint_rows(1))

    assert _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=_document()["fingerprint"],
        training_schema=TRAINING_SCHEMA,
    ) is None


def test_json_without_parquet_is_committed_data_loss(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document()
    _write_json(json_path, document)

    with pytest.raises(AxBenchBaseError, match="no matching parquet.*data loss"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


@pytest.mark.parametrize("extra_pairs", [1, 3])
def test_parquet_ahead_by_complete_pairs_restores_json_prefix(
    tmp_path: Path,
    extra_pairs: int,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    committed = _document(1)
    ahead = _document(1 + extra_pairs)
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=ahead,
        rows=_checkpoint_rows(1 + extra_pairs),
        training_schema=TRAINING_SCHEMA,
    )
    _write_json(json_path, committed)

    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=committed["fingerprint"],
        training_schema=TRAINING_SCHEMA,
    )
    assert loaded is not None
    assert len(loaded[1]) == 2


def test_parquet_ahead_by_half_pair_fails_loudly(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    committed = _document(1)
    _write_json(json_path, committed)
    _write_direct_parquet(parquet_path, _checkpoint_rows(2)[:-1])

    with pytest.raises(AxBenchBaseError, match="half-pair"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=committed["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


def test_parquet_behind_json_fails_loudly(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    committed = _document(2)
    _write_json(json_path, committed)
    _write_direct_parquet(parquet_path, _checkpoint_rows(1))

    with pytest.raises(AxBenchBaseError, match="data loss"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=committed["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "content_class": "concept_rich"}), "adjacent"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "prefix_text": "wrong"}), "prefix_text"),
        (
            lambda rows: rows.__setitem__(
                1,
                {**rows[1], "checkpoint_acceptance_sequence": 1},
            ),
            "acceptance_sequence",
        ),
        (lambda rows: rows.__setitem__(0, {**rows[0], "checkpoint_seed_idx": 9}), "seed_idx"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "qc_feature_id": 9}), "qc_feature_id"),
        (lambda rows: rows.__setitem__(1, {**rows[1], "gen_source": "openrouter:test/model"}), "natural_avoid"),
    ],
)
def test_committed_pair_corruption_fails(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document(1)
    rows = _checkpoint_rows(1)
    mutate(rows)
    _write_json(json_path, document)
    _write_direct_parquet(parquet_path, rows)

    with pytest.raises(AxBenchBaseError, match=match):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


def test_duplicate_seed_and_rank_are_rejected(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document(2)
    rows = _checkpoint_rows(2)
    for row in rows[2:]:
        row["checkpoint_seed_idx"] = 3
        row["checkpoint_candidate_rank"] = 0
    document["concept_states"][0]["tracker"]["accepted"][1].update(
        seed_idx=3,
        candidate_rank=0,
    )
    _write_json(json_path, document)
    _write_direct_parquet(parquet_path, rows)

    with pytest.raises(AxBenchBaseError, match="duplicate.*seed_idx"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


@pytest.mark.parametrize(
    ("raw_json", "match"),
    [
        ("[]", "must contain an object"),
        ("{", "Malformed"),
        ('{"schema_version":1,"schema_version":1}', "duplicate JSON object key"),
        ('{"schema_version":NaN}', "non-standard JSON"),
    ],
)
def test_malformed_json_is_rejected(
    tmp_path: Path,
    raw_json: str,
    match: str,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    json_path.write_text(raw_json, encoding="utf-8")
    _write_direct_parquet(parquet_path, [])

    with pytest.raises(AxBenchBaseError, match=match):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint={},
            training_schema=TRAINING_SCHEMA,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc.pop("next_chain_index"), "fields mismatch"),
        (lambda doc: doc.update(extra=True), "fields mismatch"),
        (lambda doc: doc.update(schema_version=3), "schema_version"),
        (lambda doc: doc.update(accepted_pair_count=True), "strict int"),
        (
            lambda doc: doc.update(next_acceptance_sequence=1),
            "accepted_pair_count",
        ),
        (
            lambda doc: doc["concept_states"][0]["source"].update(
                dirty_scored_once=1
            ),
            "strict bool",
        ),
        (
            lambda doc: doc["concept_states"][0]["source"].update(
                buffered_candidates=[
                    {**_selected(), "seed_percentile": 1}
                ]
            ),
            "finite strict float",
        ),
        (
            lambda doc: doc["concept_states"][0]["tracker"].update(
                termination_reason="stage_abort"
            ),
            "termination_reason",
        ),
        (
            lambda doc: doc["concept_states"][0]["tracker"].update(
                termination_reason=[]
            ),
            "termination_reason",
        ),
    ],
)
def test_exact_json_shape_and_scalar_types(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(AxBenchBaseError, match=match):
        positive_checkpoint.validate_checkpoint_document(
            document,
            json_path=tmp_path / "stage1b_partial.json",
        )


@pytest.mark.parametrize(
    ("stored", "expected", "field"),
    [
        ({"a": 1}, {"a": 2}, "fingerprint.a"),
        ({"a": 1}, {"a": True}, "fingerprint.a"),
        ({"a": [1]}, {"a": [1, 2]}, "length"),
        ({"a": 1}, {"a": 1, "b": 2}, "keys"),
        ({"a": {"b": "x"}}, {"a": {"b": "y"}}, "fingerprint.a.b"),
    ],
)
def test_fingerprint_reports_first_exact_mismatch(
    tmp_path: Path,
    stored: dict[str, Any],
    expected: dict[str, Any],
    field: str,
) -> None:
    with pytest.raises(AxBenchBaseError, match=field):
        positive_checkpoint.validate_fingerprint(
            stored,
            expected,
            json_path=tmp_path / "stage1b_partial.json",
        )


def test_nonfinite_expected_fingerprint_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="non-finite"):
        positive_checkpoint.validate_fingerprint(
            {"rate": 1.0},
            {"rate": math.inf},
            json_path=tmp_path / "stage1b_partial.json",
        )


def test_checkpoint_schema_rejects_internal_column_collision() -> None:
    with pytest.raises(AxBenchBaseError, match="collides"):
        positive_checkpoint.checkpoint_schema(
            TRAINING_SCHEMA.append(pa.field("checkpoint_seed_idx", pa.int64()))
        )


def test_document_accepts_known_non_null_termination_reason(tmp_path: Path) -> None:
    document = _document()
    document["concept_states"][0]["tracker"]["termination_reason"] = (
        "pool_exhausted"
    )

    assert positive_checkpoint.validate_checkpoint_document(
        document,
        json_path=tmp_path / "stage1b_partial.json",
    ) == document


def test_delete_order_is_json_then_parquet(tmp_path: Path, monkeypatch) -> None:
    json_path, parquet_path = _paths(tmp_path)
    json_path.write_text("{}", encoding="utf-8")
    parquet_path.write_bytes(b"rows")
    calls: list[Path] = []
    original_unlink = Path.unlink

    def recording_unlink(self: Path, *args, **kwargs):
        calls.append(self)
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    positive_checkpoint.delete_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
    )

    assert calls == [json_path, parquet_path]


def test_cleanup_failure_never_attempts_parquet_first(
    tmp_path: Path,
    monkeypatch,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    calls: list[Path] = []

    def failing_unlink(self: Path, *args, **kwargs):
        del args, kwargs
        calls.append(self)
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with pytest.raises(OSError, match="cleanup failure"):
        positive_checkpoint.delete_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
        )
    assert calls == [json_path]


def test_cleanup_directory_sync_failure_leaves_parquet_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    json_path.write_text("{}", encoding="utf-8")
    parquet_path.write_bytes(b"rows")
    unlinks: list[Path] = []
    original_unlink = Path.unlink

    def recording_unlink(self: Path, *args, **kwargs):
        unlinks.append(self)
        return original_unlink(self, *args, **kwargs)

    def fail_directory_sync(_path: Path) -> None:
        raise OSError("synthetic directory fsync failure")

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    monkeypatch.setattr(
        positive_checkpoint,
        "_fsync_parent_directory",
        fail_directory_sync,
    )

    with pytest.raises(OSError, match="directory fsync failure"):
        positive_checkpoint.delete_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
        )

    assert unlinks == [json_path]
    assert not json_path.exists()
    assert parquet_path.read_bytes() == b"rows"


def test_canonical_json_wraps_encoder_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dumps(*_args, **_kwargs):
        raise TypeError("synthetic encoder failure")

    monkeypatch.setattr(positive_checkpoint.json, "dumps", fail_dumps)
    with pytest.raises(AxBenchBaseError, match="canonical-JSON encodable"):
        positive_checkpoint._canonical_json_bytes({"valid": 1})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda doc: doc.update(fingerprint={1: "bad"}),
            "keys must be strings",
        ),
        (
            lambda doc: doc.update(fingerprint={"bad": object()}),
            "non-JSON value",
        ),
        (lambda doc: doc.update(fingerprint=[]), "fingerprint must be an object"),
        (
            lambda doc: doc.update(concept_states={}),
            "concept_states must be a list",
        ),
        (
            lambda doc: doc.update(concept_states=[None]),
            r"concept_states\[0\] must be an object",
        ),
        (
            lambda doc: doc["concept_states"].append(
                copy.deepcopy(doc["concept_states"][0])
            ),
            "duplicate concept_id",
        ),
        (
            lambda doc: doc["concept_states"][0].update(source=None),
            "source must be an object",
        ),
        (
            lambda doc: doc["concept_states"][0]["source"].update(
                buffered_candidates={}
            ),
            "buffered_candidates must be a list",
        ),
        (
            lambda doc: doc["concept_states"][0].update(tracker=None),
            "tracker must be an object",
        ),
        (
            lambda doc: doc["concept_states"][0]["tracker"].update(
                accepted={}
            ),
            "tracker.accepted must be a list",
        ),
        (
            lambda doc: doc["concept_states"][0]["source"].update(
                buffered_candidates=[1]
            ),
            r"buffered_candidates\[0\] must be an object",
        ),
        (
            lambda doc: doc["concept_states"][0]["source"].update(
                buffered_candidates=[{**_selected(), "prefix_text": 3}]
            ),
            "prefix_text must be a string",
        ),
        (
            lambda doc: doc["concept_states"][0]["source"].update(
                buffered_candidates=[
                    {**_selected(), "seed_percentile": 0.25}
                ]
            ),
            r"lie in \[0.5, 99.5\]",
        ),
    ],
)
def test_strict_document_rejects_nested_shape_and_domain_errors(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(AxBenchBaseError, match=match):
        positive_checkpoint.validate_checkpoint_document(
            document,
            json_path=tmp_path / "stage1b_partial.json",
        )


def test_document_accepts_discarded_and_outstanding_descriptors(
    tmp_path: Path,
) -> None:
    document = _document()
    tracker = document["concept_states"][0]["tracker"]
    document["concept_states"][0]["source"] = _source(
        next_candidate_rank=2
    )
    tracker.update(
        seeds_drawn=2,
        discarded_rich=1,
        semantic_rich_attempts=2,
        discarded=[
            {
                **_selected(seed_idx=3, candidate_rank=0),
                "side": "rich",
                "rich_semantic_attempts": 2,
                "avoid_semantic_attempts": 0,
            }
        ],
        outstanding=[
            {
                **_selected(seed_idx=4, candidate_rank=1),
                "chain_order": 1,
            }
        ],
    )
    document["next_chain_index"] = 2

    assert positive_checkpoint.validate_checkpoint_document(
        document,
        json_path=tmp_path / "stage1b_partial.json",
    ) == document

    tracker["discarded"][0]["side"] = "unknown"
    with pytest.raises(AxBenchBaseError, match="side must be 'rich' or 'avoid'"):
        positive_checkpoint.validate_checkpoint_document(
            document,
            json_path=tmp_path / "stage1b_partial.json",
        )

    tracker["discarded"][0]["side"] = []
    with pytest.raises(AxBenchBaseError, match="side must be 'rich' or 'avoid'"):
        positive_checkpoint.validate_checkpoint_document(
            document,
            json_path=tmp_path / "stage1b_partial.json",
        )


def test_document_rejects_acceptance_count_and_sequence_errors(
    tmp_path: Path,
) -> None:
    count_mismatch = _document(1)
    count_mismatch["accepted_pair_count"] = 0
    count_mismatch["next_acceptance_sequence"] = 0
    with pytest.raises(AxBenchBaseError, match="contains 1 accepted"):
        positive_checkpoint.validate_checkpoint_document(
            count_mismatch,
            json_path=tmp_path / "stage1b_partial.json",
        )

    noncontiguous = _document(1)
    noncontiguous["concept_states"][0]["tracker"]["accepted"][0][
        "acceptance_sequence"
    ] = 1
    with pytest.raises(AxBenchBaseError, match="not contiguous"):
        positive_checkpoint.validate_checkpoint_document(
            noncontiguous,
            json_path=tmp_path / "stage1b_partial.json",
        )

    duplicate = _document(2)
    duplicate["concept_states"][0]["tracker"]["accepted"][1][
        "acceptance_sequence"
    ] = 0
    with pytest.raises(AxBenchBaseError, match="duplicate acceptance_sequence"):
        positive_checkpoint.validate_checkpoint_document(
            duplicate,
            json_path=tmp_path / "stage1b_partial.json",
        )


def test_fingerprint_walks_equal_and_mismatched_nested_lists(
    tmp_path: Path,
) -> None:
    value = {"nested": [{"value": 1}, 2]}
    positive_checkpoint.validate_fingerprint(
        value,
        copy.deepcopy(value),
        json_path=tmp_path / "stage1b_partial.json",
    )
    with pytest.raises(AxBenchBaseError, match=r"nested\[1\]"):
        positive_checkpoint.validate_fingerprint(
            value,
            {"nested": [{"value": 1}, 3]},
            json_path=tmp_path / "stage1b_partial.json",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows.pop(), "committed-prefix rows"),
        (lambda rows: rows[0].pop("category"), "fields mismatch"),
        (
            lambda rows: [
                row.update(checkpoint_acceptance_sequence=1) for row in rows
            ],
            "declares acceptance_sequence",
        ),
    ],
)
def test_checkpoint_write_rejects_row_shape_and_sequence_errors(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    rows = _checkpoint_rows(1)
    mutation(rows)
    with pytest.raises(AxBenchBaseError, match=match):
        positive_checkpoint.write_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            document=_document(1),
            rows=rows,
            training_schema=TRAINING_SCHEMA,
        )


def test_checkpoint_write_wraps_arrow_conversion_error(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    rows = _checkpoint_rows(1)
    rows[0]["output_text"] = object()
    with pytest.raises(AxBenchBaseError, match="Could not build.*parquet"):
        positive_checkpoint.write_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            document=_document(1),
            rows=rows,
            training_schema=TRAINING_SCHEMA,
        )


def test_duplicate_candidate_rank_with_distinct_seed_is_rejected(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document(2)
    rows = _checkpoint_rows(2)
    document["concept_states"][0]["tracker"]["accepted"][1][
        "candidate_rank"
    ] = 0
    for row in rows[2:]:
        row["checkpoint_candidate_rank"] = 0
    _write_json(json_path, document)
    _write_direct_parquet(parquet_path, rows)
    with pytest.raises(AxBenchBaseError, match="duplicate.*candidate_rank"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


def test_wrong_or_unreadable_parquet_is_rejected(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document(1)
    _write_json(json_path, document)
    pq.write_table(pa.Table.from_pylist(_checkpoint_rows(1)), parquet_path)
    with pytest.raises(AxBenchBaseError, match="has schema"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )

    parquet_path.write_bytes(b"not parquet")
    with pytest.raises(AxBenchBaseError, match="Could not read"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
        )


def test_multiple_concepts_and_checkpoint_generations_round_trip(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document(2)
    first_state = document["concept_states"][0]
    second_state = copy.deepcopy(first_state)
    first_state["tracker"]["accepted"] = [
        first_state["tracker"]["accepted"][0]
    ]
    first_state["tracker"].update(
        seeds_drawn=1,
        semantic_rich_attempts=1,
        semantic_avoid_attempts=0,
        natural_avoid_acceptances=1,
    )
    first_state["source"] = _source(next_candidate_rank=1)
    second_state.update(concept_id=8, feature_id=43)
    second_descriptor = second_state["tracker"]["accepted"][1]
    second_state["tracker"]["accepted"] = [second_descriptor]
    second_state["tracker"].update(
        seeds_drawn=1,
        semantic_rich_attempts=1,
        semantic_avoid_attempts=1,
        natural_avoid_acceptances=0,
    )
    second_state["source"] = _source(next_candidate_rank=2)
    document["concept_states"] = [first_state, second_state]
    rows = _checkpoint_rows(2)
    for row in rows[2:]:
        row["concept_id"] = 8
        row["qc_feature_id"] = 43

    empty = _document()
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=empty,
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=rows,
        training_schema=TRAINING_SCHEMA,
    )
    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=document["fingerprint"],
        training_schema=TRAINING_SCHEMA,
    )
    assert loaded is not None
    assert loaded[0] == document
    assert len(loaded[1]) == 4


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc.pop("highest_requested_target"), "fields mismatch"),
        (lambda doc: doc.update(extra_target=72), "fields mismatch"),
        (
            lambda doc: doc.update(highest_requested_target=True),
            "positive strict int",
        ),
        (
            lambda doc: doc.update(highest_requested_target=0),
            "positive strict int",
        ),
        (
            lambda doc: doc.update(highest_requested_target=-1),
            "positive strict int",
        ),
        (
            lambda doc: doc.update(highest_requested_target=1),
            "exceeding highest_requested_target",
        ),
    ],
)
def test_schema_v2_highest_requested_target_is_strict_and_consistent(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    document = _document(2)
    mutation(document)

    with pytest.raises(AxBenchBaseError, match=match):
        positive_checkpoint.validate_checkpoint_document(
            document,
            json_path=tmp_path / "stage1b_partial.json",
        )


def test_schema_v2_rejects_target_decrease_before_parquet_access(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document()
    _write_json(json_path, document)

    with pytest.raises(AxBenchBaseError, match="target decrease is unsupported"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=document["fingerprint"],
            training_schema=TRAINING_SCHEMA,
            current_target=71,
        )


@pytest.mark.parametrize(
    ("stored_target", "current_target"),
    [
        (1, 1),
        (1, 15),
        (15, 16),
        (16, 17),
        (17, 71),
        (71, 72),
        (72, 73),
        (73, 100),
        (1, 100),
    ],
)
def test_schema_v2_accepts_same_or_higher_target_matrix(
    tmp_path: Path,
    stored_target: int,
    current_target: int,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document()
    document["highest_requested_target"] = stored_target
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )

    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=document["fingerprint"],
        training_schema=TRAINING_SCHEMA,
        current_target=current_target,
    )

    assert loaded is not None
    assert loaded[0]["highest_requested_target"] == stored_target


@pytest.mark.parametrize("current_target", [72, 100])
def test_schema_v1_target_72_migrates_in_memory_only(
    tmp_path: Path,
    current_target: int,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    v1 = _v1_document(1)
    _write_json(json_path, v1)
    _write_direct_parquet(parquet_path, _checkpoint_rows(1))
    v2_fingerprint = {
        "checkpoint_algorithm_version": (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
        ),
        "semantic": 1,
    }

    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=v2_fingerprint,
        expected_v1_fingerprint=v1["fingerprint"],
        training_schema=TRAINING_SCHEMA,
        current_target=current_target,
    )

    assert loaded is not None
    document, rows = loaded
    assert document["schema_version"] == 2
    assert document["highest_requested_target"] == 72
    assert document["fingerprint"] == v2_fingerprint
    assert json.loads(json_path.read_text(encoding="utf-8")) == v1
    assert _rows_equal_with_nan(rows, _checkpoint_rows(1))


@pytest.mark.parametrize("stored_target", [1, 12, 40, 71, 73, 100])
def test_schema_v1_non_72_target_rejects_with_phase1_batching_guidance(
    tmp_path: Path,
    stored_target: int,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    v1 = _v1_document(target=stored_target)
    _write_json(json_path, v1)

    with pytest.raises(
        AxBenchBaseError,
        match="only target 72 can migrate.*target-shaped BF16 batches.*Phase 1",
    ):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint={"v2": True},
            expected_v1_fingerprint=v1["fingerprint"],
            training_schema=TRAINING_SCHEMA,
            current_target=max(72, stored_target),
        )


def test_schema_v1_target_72_rejects_current_target_below_72(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    v1 = _v1_document()
    _write_json(json_path, v1)

    with pytest.raises(AxBenchBaseError, match="target decrease is unsupported"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint={"v2": True},
            expected_v1_fingerprint=v1["fingerprint"],
            training_schema=TRAINING_SCHEMA,
            current_target=71,
        )


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        (None, "no complete positive_selection_contract"),
        ({}, "invalid stored target"),
        ({"n_positives_per_concept": True}, "invalid stored target"),
        ({"n_positives_per_concept": 0}, "invalid stored target"),
    ],
)
def test_schema_v1_rejects_missing_or_invalid_stored_target(
    tmp_path: Path,
    selection: dict[str, Any] | None,
    match: str,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    v1 = _v1_document()
    if selection is None:
        v1["fingerprint"].pop("positive_selection_contract")
    else:
        v1["fingerprint"]["positive_selection_contract"] = selection
    _write_json(json_path, v1)

    with pytest.raises(AxBenchBaseError, match=match):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint={"v2": True},
            expected_v1_fingerprint=v1["fingerprint"],
            training_schema=TRAINING_SCHEMA,
            current_target=72,
        )


def test_schema_v1_requires_exact_phase1_fingerprint(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    v1 = _v1_document()
    _write_json(json_path, v1)

    expected = copy.deepcopy(v1["fingerprint"])
    expected["semantic"]["strict"] = False
    with pytest.raises(AxBenchBaseError, match="fingerprint.semantic.strict"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint={"v2": True},
            expected_v1_fingerprint=expected,
            training_schema=TRAINING_SCHEMA,
            current_target=72,
        )


def test_checkpoint_writer_refuses_schema_v1(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    with pytest.raises(AxBenchBaseError, match="schema-v1 is read-only"):
        positive_checkpoint.write_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            document=_v1_document(),
            rows=[],
            training_schema=TRAINING_SCHEMA,
        )


def test_fresh_checkpoint_path_never_calls_legacy_factory(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    factory = MagicMock(side_effect=AssertionError("legacy factory called"))
    assert (
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=_document()["fingerprint"],
            training_schema=TRAINING_SCHEMA,
            legacy_fingerprint_factory=factory,
        )
        is None
    )
    factory.assert_not_called()


@pytest.mark.parametrize(
    "legacy_version",
    [
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V2,
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3,
    ],
)
def test_schema_v2_legacy_algorithms_call_factory_once_and_normalize_in_memory(
    tmp_path: Path,
    legacy_version: str,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    current_fingerprint = _document()["fingerprint"]
    legacy_fingerprint = {
        **current_fingerprint,
        "checkpoint_algorithm_version": legacy_version,
        "seed_rows_sha256": "1" * 64,
    }
    document = _document(fingerprint=legacy_fingerprint)
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )
    before_json = json_path.read_bytes()
    before_parquet = parquet_path.read_bytes()
    factory = MagicMock(return_value=legacy_fingerprint)

    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=current_fingerprint,
        training_schema=TRAINING_SCHEMA,
        legacy_fingerprint_factory=factory,
    )

    assert loaded is not None
    assert loaded[0]["fingerprint"] == current_fingerprint
    factory.assert_called_once_with(legacy_version)
    assert json_path.read_bytes() == before_json
    assert parquet_path.read_bytes() == before_parquet


def test_schema_v4_resume_never_calls_legacy_factory(tmp_path: Path) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document()
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )
    factory = MagicMock(side_effect=AssertionError("legacy factory called"))
    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=document["fingerprint"],
        training_schema=TRAINING_SCHEMA,
        legacy_fingerprint_factory=factory,
    )
    assert loaded is not None
    assert loaded[0] == document
    factory.assert_not_called()


@pytest.mark.parametrize(
    "algorithm",
    [None, True, 4, "unknown-stage1b-algorithm"],
)
def test_missing_malformed_or_unknown_algorithm_fails_before_legacy_factory(
    tmp_path: Path,
    algorithm: Any,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    document = _document()
    if algorithm is None:
        document["fingerprint"].pop("checkpoint_algorithm_version")
    else:
        document["fingerprint"]["checkpoint_algorithm_version"] = algorithm
    _write_json(json_path, document)
    factory = MagicMock(side_effect=AssertionError("legacy factory called"))
    with pytest.raises(AxBenchBaseError, match="checkpoint_algorithm_version"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=_document()["fingerprint"],
            training_schema=TRAINING_SCHEMA,
            legacy_fingerprint_factory=factory,
        )
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("document", "current"),
    [
        (
            _document(
                fingerprint={
                    "checkpoint_algorithm_version": (
                        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V1
                    )
                }
            ),
            _document()["fingerprint"],
        ),
        (
            {
                **_v1_document(),
                "fingerprint": {
                    "checkpoint_algorithm_version": (
                        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
                    )
                },
            },
            _document()["fingerprint"],
        ),
    ],
)
def test_schema_algorithm_mismatch_fails_before_legacy_factory(
    tmp_path: Path,
    document: dict[str, Any],
    current: dict[str, Any],
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    _write_json(json_path, document)
    factory = MagicMock(side_effect=AssertionError("legacy factory called"))
    with pytest.raises(AxBenchBaseError, match="incompatible"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint=current,
            training_schema=TRAINING_SCHEMA,
            legacy_fingerprint_factory=factory,
        )
    factory.assert_not_called()


def test_legacy_fingerprint_mismatch_does_not_rewrite_checkpoint_pair(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    legacy = {
        "checkpoint_algorithm_version": (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
        ),
        "seed_rows_sha256": "1" * 64,
    }
    document = _document(fingerprint=legacy)
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )
    before = (json_path.read_bytes(), parquet_path.read_bytes())
    factory = MagicMock(
        return_value={**legacy, "seed_rows_sha256": "2" * 64}
    )
    with pytest.raises(AxBenchBaseError, match="seed_rows_sha256"):
        _load_checkpoint(
            json_path=json_path,
            parquet_path=parquet_path,
            expected_fingerprint={
                **legacy,
                "checkpoint_algorithm_version": (
                    positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
                ),
                "seed_rows_sha256": "3" * 64,
            },
            training_schema=TRAINING_SCHEMA,
            legacy_fingerprint_factory=factory,
        )
    factory.assert_called_once()
    assert (json_path.read_bytes(), parquet_path.read_bytes()) == before


def test_first_post_migration_commit_makes_second_resume_legacy_free(
    tmp_path: Path,
) -> None:
    json_path, parquet_path = _paths(tmp_path)
    legacy = {
        "checkpoint_algorithm_version": (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
        ),
        "seed_rows_sha256": "1" * 64,
    }
    current = {
        "checkpoint_algorithm_version": (
            positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION
        ),
        "seed_rows_sha256": "2" * 64,
    }
    document = _document(fingerprint=legacy)
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=document,
        rows=[],
        training_schema=TRAINING_SCHEMA,
    )
    first_factory = MagicMock(return_value=legacy)
    loaded = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=current,
        training_schema=TRAINING_SCHEMA,
        legacy_fingerprint_factory=first_factory,
    )
    assert loaded is not None
    positive_checkpoint.write_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        document=loaded[0],
        rows=loaded[1],
        training_schema=TRAINING_SCHEMA,
    )
    second_factory = MagicMock(
        side_effect=AssertionError("legacy factory called on v4")
    )
    second = _load_checkpoint(
        json_path=json_path,
        parquet_path=parquet_path,
        expected_fingerprint=current,
        training_schema=TRAINING_SCHEMA,
        legacy_fingerprint_factory=second_factory,
    )
    assert second is not None
    assert second[0]["fingerprint"] == current
    first_factory.assert_called_once_with(
        positive_checkpoint.CHECKPOINT_ALGORITHM_VERSION_V3
    )
    second_factory.assert_not_called()
