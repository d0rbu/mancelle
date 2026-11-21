"""Unit tests for steering.profiling.markers.TimingRecorder."""

from __future__ import annotations

import csv

import pytest

from steering.profiling.markers import TimingRecorder


def test_append_preserves_order():
    r = TimingRecorder()
    r.append("a", 100)
    r.append("b", 200)
    r.append("c", 300)
    names = [row["name"] for row in r.rows()]
    assert names == ["a", "b", "c"]


def test_rows_returns_independent_copy():
    r = TimingRecorder()
    r.append("a", 100)
    rows = r.rows()
    rows.append({"name": "evil", "elapsed_ns": 0})
    assert len(r.rows()) == 1
    assert r.rows()[0]["name"] == "a"


def test_tag_rows_since_only_updates_new_rows_with_defaults():
    r = TimingRecorder()
    r.append("before", 1, {"path": "explicit"})
    start_index = len(r)
    r.append("after-a", 2)
    r.append("after-b", 3, {"path": "explicit"})

    next_index = r.tag_rows_since(
        start_index,
        {"path": "default", "iter": 4},
    )

    assert next_index == len(r) == 3
    assert r.rows() == [
        {"name": "before", "elapsed_ns": 1, "path": "explicit"},
        {"name": "after-a", "elapsed_ns": 2, "path": "default", "iter": 4},
        {"name": "after-b", "elapsed_ns": 3, "path": "explicit", "iter": 4},
    ]


@pytest.mark.parametrize("start_index", [-1, 2])
def test_tag_rows_since_rejects_invalid_append_position(start_index: int):
    r = TimingRecorder()
    r.append("only", 1)

    with pytest.raises(IndexError, match="start_index must be"):
        r.tag_rows_since(start_index, {"iter": 0})


def test_to_csv_roundtrip(tmp_path):
    r = TimingRecorder()
    r.append("suite", 1_234_567, extra={"iter": 1, "path": "A"})
    r.append("suite", 2_000_000, extra={"iter": 2, "path": "A"})
    out = tmp_path / "timings.csv"
    r.to_csv(out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["name"] == "suite"
    assert int(rows[0]["elapsed_ns"]) == 1_234_567
    assert rows[0]["iter"] == "1"
    assert rows[1]["path"] == "A"


def test_to_csv_empty_writes_header_only(tmp_path):
    r = TimingRecorder()
    out = tmp_path / "empty.csv"
    r.to_csv(out)
    with open(out) as f:
        content = f.read().strip()
    assert content == "name,elapsed_ns"


def test_elapsed_ns_coerces_to_int():
    r = TimingRecorder()
    r.append("a", 100.7)
    assert r.rows()[0]["elapsed_ns"] == 100


def test_to_csv_heterogeneous_keys_union_in_first_seen_order(tmp_path):
    # The only non-trivial to_csv branch: rows with DIFFERENT extra keys. The
    # header is the first-seen UNION (name, elapsed_ns, a, b) and restval="" fills
    # the cells a row does not have.
    r = TimingRecorder()
    r.append("r0", 1, extra={"a": "A"})
    r.append("r1", 2, extra={"b": "B"})
    r.append("r2", 3)  # neither a nor b
    out = tmp_path / "het.csv"
    r.to_csv(out)

    with open(out, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["name", "elapsed_ns", "a", "b"]
        rows = list(reader)
    assert len(rows) == 3
    # r0 has a, blank b; r1 blank a, has b; r2 blank a and b.
    assert rows[0]["a"] == "A" and rows[0]["b"] == ""
    assert rows[1]["a"] == "" and rows[1]["b"] == "B"
    assert rows[2]["a"] == "" and rows[2]["b"] == ""


def test_to_csv_creates_missing_parent_dirs(tmp_path):
    # to_csv mkdir(parents=True): a 3-level-deep nonexistent path is created.
    r = TimingRecorder()
    r.append("a", 1)
    out = tmp_path / "deep" / "nested" / "dir" / "timings.csv"
    assert not out.parent.exists()
    r.to_csv(out)
    assert out.exists()
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["name"] == "a"


def test_append_coerces_name_and_extra_keys_to_str():
    # append str()-coerces the name AND every extra key (so the CSV columns are
    # stable scalar strings even if a caller passes an int key).
    r = TimingRecorder()
    r.append(123, 5, {42: "v"})
    row = r.rows()[0]
    assert row["name"] == "123"
    assert "42" in row and row["42"] == "v"
    assert 42 not in row


def test_append_falsy_extra_adds_no_columns():
    # Falsy extra ({} or None) must add NO extra columns (the `if extra:` guard).
    r = TimingRecorder()
    r.append("empty_dict", 1, {})
    r.append("none_extra", 2, None)
    for row in r.rows():
        assert set(row.keys()) == {"name", "elapsed_ns"}


def test_rows_dicts_are_independent_copies():
    # Mutating a dict returned by rows() must not corrupt internal state (rows()
    # returns per-row copies, not aliases).
    r = TimingRecorder()
    r.append("a", 1, {"k": "v"})
    snapshot = r.rows()
    snapshot[0]["k"] = "MUTATED"
    snapshot[0]["name"] = "evil"
    fresh = r.rows()
    assert fresh[0]["k"] == "v"
    assert fresh[0]["name"] == "a"
