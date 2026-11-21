"""Resumability pathologies: corrupt JSONL line, duplicate feature_id, schema_version mismatch."""
import json
from pathlib import Path

import pytest

from steering.eval.feature_filter.types import SCHEMA_VERSION


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def test_corrupt_jsonl_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text('{"feature_id": 1}\nthis-is-not-json\n')
    with pytest.raises(json.JSONDecodeError):
        _read_rows(p)


def test_duplicate_feature_id_detected(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text(
        json.dumps({"feature_id": 7, "schema_version": SCHEMA_VERSION}) + "\n" +
        json.dumps({"feature_id": 7, "schema_version": SCHEMA_VERSION}) + "\n"
    )
    rows = _read_rows(p)
    ids = [r["feature_id"] for r in rows]
    assert len(ids) != len(set(ids))


def test_schema_version_mismatch_visible(tmp_path: Path) -> None:
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps({"feature_id": 1, "schema_version": SCHEMA_VERSION + 99}) + "\n")
    rows = _read_rows(p)
    assert rows[0]["schema_version"] != SCHEMA_VERSION
