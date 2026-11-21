"""Schema and atomic logical-content publication tests."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_adv import artifacts
from steering.eval.axbench_adv.artifacts import (
    logical_table_digest,
    write_suffix_artifacts,
    write_validated_parquet,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.schemas import (
    UNIVERSAL_SUFFIXES_SCHEMA,
    suffix_sha256,
    validate_universal_suffixes,
)


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "method": "DiffMean",
        "concept_id": 0,
        "proposal_count": 1,
        "coefficient": 1.0,
        "suffix": "Avoid the target concept.",
        "suffix_sha256": "039377a9cdf325081c9797f54f7896bd173e05b2a3c94db58d3b11599fb72177",
        "master_seed": 0,
        "best_fitness": 0.75,
        "gepa_nominal_metric_calls": 10,
        "actual_evaluator_calls": 10,
        "actual_generation_calls": 10,
        "paid_judge_cache_misses": 20,
        "paid_reflection_cache_misses": 1,
    }
    row.update(updates)
    return row


def _table(*rows: dict[str, object]) -> pa.Table:
    return pa.Table.from_pylist(list(rows) or [_row()], schema=UNIVERSAL_SUFFIXES_SCHEMA)


def test_suffix_schema_digest_and_publication(tmp_path: Path) -> None:
    table = _table()
    validate_universal_suffixes(table)
    expected_suffix_digest = "039377a9cdf325081c9797f54f7896bd173e05b2a3c94db58d3b11599fb72177"
    expected_table_digest = "76abfc59c6d61c67984e17990703b766434835a56604552ea29fe9c93668ef8c"
    assert suffix_sha256("  Avoid the target concept.  ") == expected_suffix_digest
    assert logical_table_digest(table) == expected_table_digest

    alternates: dict[str, object] = {
        "method": "LinearProbe",
        "concept_id": 1,
        "proposal_count": 2,
        "coefficient": 0.5,
        "suffix": "Different suffix.",
        "suffix_sha256": "1" * 64,
        "master_seed": 1,
        "best_fitness": 0.5,
        "gepa_nominal_metric_calls": 11,
        "actual_evaluator_calls": 11,
        "actual_generation_calls": 11,
        "paid_judge_cache_misses": 21,
        "paid_reflection_cache_misses": 2,
    }
    assert set(alternates) == set(UNIVERSAL_SUFFIXES_SCHEMA.names)
    for field, alternate in alternates.items():
        assert logical_table_digest(_table(_row(**{field: alternate}))) != expected_table_digest

    suffix_path, manifest_path = write_suffix_artifacts(
        suffixes=table,
        output_dir=tmp_path / "cell",
    )
    assert pq.read_table(suffix_path).equals(table)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["files"]["universal_suffixes.parquet"] == {
        "logical_sha256": logical_table_digest(table),
        "rows": 1,
    }
    assert manifest["identity"] == {
        "method": "DiffMean",
        "concept_id": 0,
        "proposal_count": 1,
    }
    assert manifest_path.stat().st_mtime_ns >= suffix_path.stat().st_mtime_ns

    with pytest.raises(AxBenchAdvError, match="already complete"):
        write_suffix_artifacts(suffixes=table, output_dir=tmp_path / "cell")


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"method": "Unknown"}, "unknown"),
        ({"concept_id": -1}, "non-negative"),
        ({"proposal_count": 3}, "one of"),
        ({"coefficient": float("nan")}, "coefficient"),
        ({"coefficient": 7.0}, "factor grid"),
        ({"best_fitness": -0.1}, "best_fitness"),
        ({"suffix": " padded "}, "already be canonical"),
        ({"suffix_sha256": "0" * 64}, "SHA-256"),
        ({"master_seed": -1}, "master_seed"),
        ({"actual_generation_calls": -1}, "actual_generation_calls"),
    ],
)
def test_suffix_semantics_reject_mutations(updates: dict[str, object], match: str) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        validate_universal_suffixes(_table(_row(**updates)))


def test_suffix_semantics_reject_schema_empty_and_duplicate() -> None:
    with pytest.raises(AxBenchAdvError, match="PyArrow"):
        validate_universal_suffixes(None)  # type: ignore[arg-type]
    with pytest.raises(AxBenchAdvError, match="does not conform"):
        validate_universal_suffixes(pa.table({"wrong": [1]}))
    with pytest.raises(AxBenchAdvError, match="must not be empty"):
        validate_universal_suffixes(pa.Table.from_pylist([], schema=UNIVERSAL_SUFFIXES_SCHEMA))
    with pytest.raises(AxBenchAdvError, match="duplicate"):
        validate_universal_suffixes(_table(_row(), _row()))
    with pytest.raises(AxBenchAdvError, match="null values"):
        validate_universal_suffixes(_table(_row(suffix=None)))


def test_suffix_cell_rejects_multiple_rows(tmp_path: Path) -> None:
    with pytest.raises(AxBenchAdvError, match="exactly one row"):
        write_suffix_artifacts(
            suffixes=_table(_row(), _row(concept_id=1)),
            output_dir=tmp_path / "cell",
        )
    assert not (tmp_path / "cell").exists()


def test_reread_digest_mismatch_preserves_old_final_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_path = tmp_path / "suffixes.parquet"
    final_path.write_bytes(b"old-final")
    source = _table()
    reread = _table(_row(master_seed=1))
    monkeypatch.setattr(artifacts.pq, "read_table", lambda _path: reread)
    with pytest.raises(AxBenchAdvError, match="logical digest differs"):
        write_validated_parquet(source, final_path, validator=validate_universal_suffixes)
    assert final_path.read_bytes() == b"old-final"
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_initial_validation_failure_writes_nothing(tmp_path: Path) -> None:
    final_path = tmp_path / "bad.parquet"
    with pytest.raises(AxBenchAdvError, match="does not conform"):
        write_validated_parquet(
            pa.table({"bad": [1]}),
            final_path,
            validator=validate_universal_suffixes,
        )
    assert not final_path.exists()


def test_reread_schema_failure_preserves_old_final_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_path = tmp_path / "suffixes.parquet"
    final_path.write_bytes(b"old-final")
    monkeypatch.setattr(artifacts.pq, "read_table", lambda _path: pa.table({"bad": [1]}))
    with pytest.raises(AxBenchAdvError, match="does not conform"):
        write_validated_parquet(_table(), final_path, validator=validate_universal_suffixes)
    assert final_path.read_bytes() == b"old-final"
    assert list(tmp_path.glob("*.tmp.*")) == []


@pytest.mark.parametrize("operation", ["write", "open", "fsync", "read", "replace"])
def test_atomic_stage_failure_preserves_old_final_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    final_path = tmp_path / "suffixes.parquet"
    final_path.write_bytes(b"old-final")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError(f"injected {operation} failure")

    module, attribute = {
        "write": (artifacts.pq, "write_table"),
        "open": (artifacts.os, "open"),
        "fsync": (artifacts.os, "fsync"),
        "read": (artifacts.pq, "read_table"),
        "replace": (artifacts.os, "replace"),
    }[operation]
    monkeypatch.setattr(module, attribute, fail)
    with pytest.raises(OSError, match="injected"):
        write_validated_parquet(_table(), final_path, validator=validate_universal_suffixes)
    assert final_path.read_bytes() == b"old-final"
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_manifest_failure_leaves_no_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cell"
    original_atomic_write_text = artifacts.atomic_write_text
    monkeypatch.setattr(
        artifacts,
        "atomic_write_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest failure")),
    )
    with pytest.raises(OSError, match="manifest failure"):
        write_suffix_artifacts(suffixes=_table(), output_dir=output)
    assert (output / "universal_suffixes.parquet").is_file()
    assert not (output / "manifest.json").exists()
    monkeypatch.setattr(artifacts, "atomic_write_text", original_atomic_write_text)
    suffix_path, manifest_path = write_suffix_artifacts(suffixes=_table(), output_dir=output)
    assert suffix_path.is_file()
    assert manifest_path.is_file()


def test_incomplete_cell_rejects_unknown_entries(tmp_path: Path) -> None:
    output = tmp_path / "cell"
    output.mkdir()
    (output / "unrecognized.txt").write_text("do not delete", encoding="utf-8")
    with pytest.raises(AxBenchAdvError, match="unknown entries"):
        write_suffix_artifacts(suffixes=_table(), output_dir=output)
    assert (output / "unrecognized.txt").read_text(encoding="utf-8") == "do not delete"
