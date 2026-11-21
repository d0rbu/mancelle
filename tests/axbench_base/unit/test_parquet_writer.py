"""Unit tests for the directory-dataset parquet append writer.

Covers:
  - Single-batch + multi-batch writes, final partial batch
  - Resume semantics: scan existing dir for batch_*.parquet
  - Schema enforcement on every append (missing / extra columns)
  - load_existing: empty dirs, ignoring non-parquet files, schema
    validation against concatenated output
  - Context-manager + close idempotency, append-after-close error
  - Batch-index ordering with zero-padded naming (pins the sort order)
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.types import AxBenchBaseError

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


_SCHEMA = pa.schema(
    [
        ("a", pa.int32()),
        ("b", pa.string()),
    ]
)


def _row(a: int, b: str = "x") -> dict[str, object]:
    return {"a": a, "b": b}


def _count_batch_files(out_dir: Path) -> int:
    return sum(1 for _ in out_dir.glob("batch_*.parquet"))


# ===========================================================================
# 1 - Single small batch becomes one file
# ===========================================================================


def test_single_batch_writes_one_file(tmp_path: Path) -> None:
    """Append 5 rows, close -> exactly one file batch_000000.parquet w/ 5 rows."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=100)
    try:
        for i in range(5):
            writer.append(_row(i))
    finally:
        writer.close()

    files = sorted(out_dir.glob("batch_*.parquet"))
    assert len(files) == 1
    assert files[0].name == "batch_000000.parquet"

    table = pq.read_table(files[0])
    assert table.num_rows == 5
    assert table.schema.equals(_SCHEMA)
    assert table.column("a").to_pylist() == list(range(5))


# ===========================================================================
# 2 - Multi-batch: 250 rows / batch_size 100 -> 100+100+50
# ===========================================================================


def test_multiple_batches_write_multiple_files(tmp_path: Path) -> None:
    """Append 250 rows -> 3 files, last one has exactly 50 rows."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=100)
    try:
        for i in range(250):
            writer.append(_row(i))
    finally:
        writer.close()

    files = sorted(out_dir.glob("batch_*.parquet"))
    assert [f.name for f in files] == [
        "batch_000000.parquet",
        "batch_000001.parquet",
        "batch_000002.parquet",
    ]

    sizes = [pq.read_table(f).num_rows for f in files]
    assert sizes == [100, 100, 50]

    # Values survive concatenation in order.
    combined = pa.concat_tables([pq.read_table(f) for f in files])
    assert combined.column("a").to_pylist() == list(range(250))


# ===========================================================================
# 3 - Resume: existing files push index forward
# ===========================================================================


def test_resume_continues_from_next_batch_index(tmp_path: Path) -> None:
    """Pre-create batch_000000 and batch_000001 -> new writer -> new file is batch_000002."""
    out_dir = tmp_path / "gens"
    out_dir.mkdir()
    for i in range(2):
        table = pa.Table.from_pylist([_row(i)], schema=_SCHEMA)
        pq.write_table(table, out_dir / f"batch_{i:06d}.parquet")

    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(99))
    finally:
        writer.close()

    names = sorted(p.name for p in out_dir.glob("batch_*.parquet"))
    assert names == [
        "batch_000000.parquet",
        "batch_000001.parquet",
        "batch_000002.parquet",
    ]
    new_table = pq.read_table(out_dir / "batch_000002.parquet")
    assert new_table.num_rows == 1
    assert new_table.column("a").to_pylist() == [99]


# ===========================================================================
# 4 - load_existing reads batches in order and returns one table
# ===========================================================================


def test_load_existing_reads_all_batches(tmp_path: Path) -> None:
    """Write 3 batches, load_existing returns one concatenated table."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=3)
    try:
        for i in range(9):
            writer.append(_row(i))
    finally:
        writer.close()

    table = load_existing(out_dir, schema=_SCHEMA)
    assert table.num_rows == 9
    assert table.schema.equals(_SCHEMA)
    assert table.column("a").to_pylist() == list(range(9))


# ===========================================================================
# 5 - load_existing: empty / missing dir returns empty table
# ===========================================================================


def test_load_existing_missing_dir_returns_empty_table(tmp_path: Path) -> None:
    """A non-existent directory yields an empty table w/ the given schema."""
    missing = tmp_path / "nope"
    table = load_existing(missing, schema=_SCHEMA)
    assert table.num_rows == 0
    assert table.schema.equals(_SCHEMA)


def test_load_existing_empty_dir_returns_empty_table(tmp_path: Path) -> None:
    """An existing but empty directory yields an empty table w/ the schema."""
    empty = tmp_path / "gens"
    empty.mkdir()
    table = load_existing(empty, schema=_SCHEMA)
    assert table.num_rows == 0
    assert table.schema.equals(_SCHEMA)


# ===========================================================================
# 6 - load_existing ignores non-parquet files and unrelated parquet files
# ===========================================================================


def test_load_existing_ignores_non_parquet_files(tmp_path: Path) -> None:
    """metadata.json, README.md, and unrelated parquet files must be ignored."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(0))
        writer.append(_row(1))
    finally:
        writer.close()

    (out_dir / "metadata.json").write_text("{}")
    (out_dir / "README.md").write_text("readme")
    (out_dir / "notes.txt").write_text("hi")
    # Parquet-looking file that does NOT match the batch_ pattern must be ignored.
    unrelated = pa.Table.from_pylist([_row(999)], schema=_SCHEMA)
    pq.write_table(unrelated, out_dir / "summary.parquet")

    table = load_existing(out_dir, schema=_SCHEMA)
    assert table.num_rows == 2
    assert table.column("a").to_pylist() == [0, 1]


# ===========================================================================
# 7 - Empty flush is a no-op
# ===========================================================================


def test_empty_flush_is_noop(tmp_path: Path) -> None:
    """flush() with no rows pending writes nothing and does not raise."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.flush()  # no pending rows
        writer.flush()  # idempotent
        assert _count_batch_files(out_dir) == 0
    finally:
        writer.close()
    assert _count_batch_files(out_dir) == 0


# ===========================================================================
# 8 - close() flushes a partial buffer
# ===========================================================================


def test_close_flushes_partial_buffer(tmp_path: Path) -> None:
    """Append 50 rows, close (no explicit flush) -> one file w/ 50 rows."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=100)
    for i in range(50):
        writer.append(_row(i))
    writer.close()

    files = sorted(out_dir.glob("batch_*.parquet"))
    assert len(files) == 1
    assert pq.read_table(files[0]).num_rows == 50


# ===========================================================================
# 9 - Schema enforcement: row missing a column
# ===========================================================================


def test_schema_enforced_missing_column(tmp_path: Path) -> None:
    """A row missing a schema column raises AxBenchBaseError."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        with pytest.raises(AxBenchBaseError, match="missing required columns"):
            writer.append({"a": 1})  # no "b"
    finally:
        writer.close()


# ===========================================================================
# 10 - Schema enforcement: row has an extra column
# ===========================================================================


def test_schema_enforced_extra_column(tmp_path: Path) -> None:
    """A row with an extra column raises AxBenchBaseError."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        with pytest.raises(AxBenchBaseError, match="unexpected columns"):
            writer.append({"a": 1, "b": "x", "c": 9.0})
    finally:
        writer.close()


# ===========================================================================
# 11 - Context manager flushes on exit
# ===========================================================================


def test_context_manager_closes_on_exit(tmp_path: Path) -> None:
    """with ParquetAppendWriter(...) flushes/closes on exit."""
    out_dir = tmp_path / "gens"
    with ParquetAppendWriter(
        out_dir=out_dir, schema=_SCHEMA, batch_size=100
    ) as writer:
        for i in range(7):
            writer.append(_row(i))
    # Post-exit: file exists and contains the buffered rows.
    files = sorted(out_dir.glob("batch_*.parquet"))
    assert len(files) == 1
    assert pq.read_table(files[0]).num_rows == 7

    # And writes after exit fail.
    with pytest.raises(AxBenchBaseError, match="closed"):
        writer.append(_row(100))


# ===========================================================================
# 12 - Double close is idempotent
# ===========================================================================


def test_double_close_idempotent(tmp_path: Path) -> None:
    """Calling close() twice must not raise and must not double-write."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=5)
    for i in range(3):
        writer.append(_row(i))
    writer.close()
    writer.close()  # idempotent
    assert _count_batch_files(out_dir) == 1


# ===========================================================================
# 14 - load_existing raises on schema mismatch
# ===========================================================================


def test_load_existing_with_schema_mismatch_raises(tmp_path: Path) -> None:
    """load_existing with a schema differing from on-disk files must raise."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(0))
        writer.append(_row(1))
    finally:
        writer.close()

    other_schema = pa.schema([("x", pa.float32()), ("y", pa.int64())])
    with pytest.raises(AxBenchBaseError, match="schema mismatch"):
        load_existing(out_dir, schema=other_schema)


def test_load_existing_without_schema_is_permissive(tmp_path: Path) -> None:
    """schema=None (default) skips validation and still concatenates."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(0))
        writer.append(_row(1))
    finally:
        writer.close()
    table = load_existing(out_dir)
    assert table.num_rows == 2
    assert table.schema.equals(_SCHEMA)


def test_load_existing_without_schema_empty_dir_returns_empty(
    tmp_path: Path,
) -> None:
    """schema=None + empty dir yields an empty schemaless table."""
    empty = tmp_path / "nope"
    table = load_existing(empty)
    assert table.num_rows == 0
    assert table.num_columns == 0


# ===========================================================================
# 15 - Resume skips gaps: max index + 1, not gap-fill
# ===========================================================================


def test_batch_index_skipping_in_resume(tmp_path: Path) -> None:
    """Pre-create batch_000000 and batch_000005 -> new writer starts at 000006."""
    out_dir = tmp_path / "gens"
    out_dir.mkdir()
    for i in (0, 5):
        pq.write_table(
            pa.Table.from_pylist([_row(i)], schema=_SCHEMA),
            out_dir / f"batch_{i:06d}.parquet",
        )

    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(42))
    finally:
        writer.close()

    names = sorted(p.name for p in out_dir.glob("batch_*.parquet"))
    assert "batch_000006.parquet" in names
    assert "batch_000001.parquet" not in names  # the gap is NOT filled


# ===========================================================================
# 16 - Appending after close() raises
# ===========================================================================


def test_append_after_close_raises(tmp_path: Path) -> None:
    """Appending after close() must raise AxBenchBaseError."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    writer.append(_row(0))
    writer.close()
    with pytest.raises(AxBenchBaseError, match="closed"):
        writer.append(_row(1))


def test_flush_after_close_raises(tmp_path: Path) -> None:
    """flush() after close() must also raise AxBenchBaseError."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    writer.close()
    with pytest.raises(AxBenchBaseError, match="closed"):
        writer.flush()


# ===========================================================================
# 17 - Zero-padded batch ordering: many files sort correctly
# ===========================================================================


def test_batch_file_ordering_stable(tmp_path: Path) -> None:
    """Verify lexicographic ordering works for 12 batches (0..11)."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=1)
    try:
        for i in range(12):
            writer.append(_row(i))
    finally:
        writer.close()

    files = sorted(out_dir.glob("batch_*.parquet"))
    assert len(files) == 12
    expected = [f"batch_{i:06d}.parquet" for i in range(12)]
    assert [f.name for f in files] == expected

    # And load_existing returns them in batch order (0, 1, 2, ..., 11) -
    # critical since naive lexicographic sort of unpadded names would give
    # 0, 1, 10, 11, 2, 3, ...
    table = load_existing(out_dir, schema=_SCHEMA)
    assert table.column("a").to_pylist() == list(range(12))


# ===========================================================================
# 18 - Mid-batch resume + additional appends continues indices
# ===========================================================================


def test_auto_flush_at_batch_size_boundary(tmp_path: Path) -> None:
    """Exactly batch_size appends trigger an auto-flush, not batch_size+1."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=3)
    try:
        writer.append(_row(0))
        writer.append(_row(1))
        assert _count_batch_files(out_dir) == 0
        writer.append(_row(2))  # auto-flush here
        assert _count_batch_files(out_dir) == 1
    finally:
        writer.close()


# ===========================================================================
# 19 - Files with non-matching-integer names in the batch_* pattern are ignored
# ===========================================================================


def test_load_existing_ignores_malformed_batch_names(tmp_path: Path) -> None:
    """batch_foo.parquet / batch_.parquet should not crash the regex; they
    are skipped and legitimate batch_NNNNNN files still load."""
    out_dir = tmp_path / "gens"
    out_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist([_row(1)], schema=_SCHEMA),
        out_dir / "batch_000000.parquet",
    )
    # Malformed (non-digit) stem component must be ignored, not raise.
    (out_dir / "batch_foo.parquet").write_bytes(b"garbage")

    table = load_existing(out_dir, schema=_SCHEMA)
    assert table.num_rows == 1
    assert table.column("a").to_pylist() == [1]


def test_resume_ignores_malformed_batch_names(tmp_path: Path) -> None:
    """Malformed batch_*.parquet names do not contribute to the next-index
    computation (i.e. they don't cause an int() to blow up at __init__)."""
    out_dir = tmp_path / "gens"
    out_dir.mkdir()
    pq.write_table(
        pa.Table.from_pylist([_row(1)], schema=_SCHEMA),
        out_dir / "batch_000002.parquet",
    )
    (out_dir / "batch_foo.parquet").write_bytes(b"garbage")

    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(7))
    finally:
        writer.close()

    assert (out_dir / "batch_000003.parquet").exists()


# ===========================================================================
# 20 - Wrong Python type for column type bubbles up as ArrowInvalid
# ===========================================================================


def test_wrong_type_bubbles_up_arrow_error(tmp_path: Path) -> None:
    """Type coercion is pyarrow's job: wrong types raise at flush, not append."""
    out_dir = tmp_path / "gens"
    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append({"a": "not an int", "b": "x"})
        # Slice 16 audit (X2 §4.3): narrow from broad-Exception. PyArrow's
        # batch construction surfaces type-coercion failures as either
        # ``pa.ArrowInvalid`` or ``TypeError`` depending on platform /
        # pyarrow version. The contract is "flushing wrong types raises a
        # clear pyarrow-or-Python error" — pin both.
        with pytest.raises((pa.ArrowInvalid, TypeError)):
            writer.flush()
    finally:
        # Close may re-flush; swallow any resulting error to avoid
        # masking the assertion above.
        try:
            writer.close()
        except Exception:
            pass


# ===========================================================================
# 21 - Invalid batch_size rejected at construction
# ===========================================================================


@pytest.mark.parametrize("bad_batch_size", [0, -1, -100])
def test_invalid_batch_size_raises(tmp_path: Path, bad_batch_size: int) -> None:
    """Zero / negative batch_size must raise at construction."""
    with pytest.raises(AxBenchBaseError, match="batch_size must be positive"):
        ParquetAppendWriter(
            out_dir=tmp_path / "gens",
            schema=_SCHEMA,
            batch_size=bad_batch_size,
        )


# ===========================================================================
# 22 - Subdirectories in out_dir are ignored during scan
# ===========================================================================


def test_resume_ignores_subdirectories(tmp_path: Path) -> None:
    """A subdirectory named batch_000000.parquet/ (dir, not file) is skipped."""
    out_dir = tmp_path / "gens"
    out_dir.mkdir()
    # Create a real batch file...
    pq.write_table(
        pa.Table.from_pylist([_row(1)], schema=_SCHEMA),
        out_dir / "batch_000000.parquet",
    )
    # ...and a subdirectory (could be a sidecar, a lock dir, etc.) that
    # happens to match the name pattern. The writer must not try to read it.
    (out_dir / "batch_000042.parquet").mkdir()

    writer = ParquetAppendWriter(out_dir=out_dir, schema=_SCHEMA, batch_size=10)
    try:
        writer.append(_row(7))
    finally:
        writer.close()

    # The new file follows the highest *file* index (0) + 1, not the
    # subdirectory's 42.
    assert (out_dir / "batch_000001.parquet").exists()


def test_immutable_batches_accept_exact_concurrent_replay(tmp_path: Path) -> None:
    out_dir = tmp_path / "gens"
    first = ParquetAppendWriter(
        out_dir=out_dir,
        schema=_SCHEMA,
        immutable_batches=True,
    )
    second = ParquetAppendWriter(
        out_dir=out_dir,
        schema=_SCHEMA,
        immutable_batches=True,
    )
    first.append(_row(7))
    second.append(_row(7))
    first.close()
    original = (out_dir / "batch_000000.parquet").read_bytes()
    second.close()

    assert _count_batch_files(out_dir) == 1
    assert (out_dir / "batch_000000.parquet").read_bytes() == original


def test_immutable_batches_reject_different_concurrent_replay(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "gens"
    first = ParquetAppendWriter(
        out_dir=out_dir,
        schema=_SCHEMA,
        immutable_batches=True,
    )
    second = ParquetAppendWriter(
        out_dir=out_dir,
        schema=_SCHEMA,
        immutable_batches=True,
    )
    first.append(_row(7))
    second.append(_row(8))
    first.close()
    path = out_dir / "batch_000000.parquet"
    original = path.read_bytes()

    with pytest.raises(AxBenchBaseError, match="refusing to overwrite"):
        second.close()

    assert path.read_bytes() == original
    assert pq.read_table(path).column("a").to_pylist() == [7]


def test_immutable_batches_reject_a_staggered_concurrent_append(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "gens"
    validated_indices: tuple[int, ...] = ()
    concurrent = ParquetAppendWriter(
        out_dir=out_dir,
        schema=_SCHEMA,
        immutable_batches=True,
    )
    concurrent.append(_row(7))
    concurrent.close()

    with pytest.raises(AxBenchBaseError, match="changed after resume validation"):
        ParquetAppendWriter(
            out_dir=out_dir,
            schema=_SCHEMA,
            immutable_batches=True,
            expected_existing_indices=validated_indices,
        )

    resumed = ParquetAppendWriter(
        out_dir=out_dir,
        schema=_SCHEMA,
        immutable_batches=True,
        expected_existing_indices=(0,),
    )
    resumed.append(_row(8))
    resumed.close()
    assert _count_batch_files(out_dir) == 2
    assert pq.read_table(
        out_dir / "batch_000001.parquet"
    ).column("a").to_pylist() == [8]


def test_expected_existing_indices_requires_immutable_batches(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="valid only for immutable batches",
    ):
        ParquetAppendWriter(
            out_dir=tmp_path / "gens",
            schema=_SCHEMA,
            immutable_batches=False,
            expected_existing_indices=(),
        )
