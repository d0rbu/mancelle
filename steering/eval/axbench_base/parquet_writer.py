"""Directory-dataset parquet append writer for Stage 4 and Stage 5.

Stage 4 (``generate_all``) produces one generation row per
(concept, method, coefficient, seed) tuple. Stage 5 (``judge_all``)
produces one judge-score row per generation. Both stages run for many
minutes to hours on a remote machine, so we write incrementally and
recover cleanly from crashes.

Layout
------
Each writer owns a directory on disk. Rows are appended to an in-memory
buffer and flushed to a new parquet file once the buffer hits
``batch_size`` entries. File names are zero-padded so that lexicographic
sort matches batch-index order:

    <out_dir>/
        batch_000000.parquet
        batch_000001.parquet
        batch_000002.parquet
        ...

Resume
------
At construction time, the writer scans ``out_dir`` for existing
``batch_NNNNNN.parquet`` files and sets its next-index cursor to
``max(index) + 1``. Non-matching files (``metadata.json``,
``README.md``, ``batch_foo.parquet``) are silently ignored, so callers
are free to drop sidecars next to the batches. Gaps in the existing
index sequence are NOT filled — the writer always appends.

Schema enforcement
------------------
Every row passed to :meth:`ParquetAppendWriter.append` is validated
against the configured schema. A row missing a column or carrying an
extra column raises :class:`AxBenchBaseError` on the spot — the caller's
row-assembly code is the bug source, not the parquet boundary. Type
mismatches (e.g. a string in an ``int32`` column) are detected by
pyarrow at flush time and bubble up unchanged.

Usage
-----
    with ParquetAppendWriter(out_dir=..., schema=..., batch_size=100) as w:
        for row in rows:
            w.append(row)

    # later, read them back:
    table = load_existing(out_dir, schema=...)
"""

from __future__ import annotations

import re
from pathlib import Path
from types import TracebackType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.immutable_io import ImmutableArtifactError, install_bytes_once

_BATCH_PATTERN = re.compile(r"^batch_(\d+)\.parquet$")


def _batch_filename(index: int) -> str:
    """Return the canonical zero-padded batch filename for ``index``."""
    return f"batch_{index:06d}.parquet"


def _scan_batch_indices(out_dir: Path) -> list[int]:
    """Return sorted list of batch indices from ``batch_NNNNNN.parquet`` files.

    Files that do not match the pattern are ignored, not raised on. The
    returned list is sorted ascending. Missing / non-existent ``out_dir``
    returns an empty list.
    """
    if not out_dir.exists():
        return []
    indices: list[int] = []
    for entry in out_dir.iterdir():
        if not entry.is_file():
            continue
        match = _BATCH_PATTERN.match(entry.name)
        if match is None:
            continue
        indices.append(int(match.group(1)))
    indices.sort()
    return indices


class ParquetAppendWriter:
    """Append-only parquet directory-dataset writer with resume support.

    Parameters
    ----------
    out_dir:
        Directory to write ``batch_NNNNNN.parquet`` files under. Created
        if it doesn't exist.
    schema:
        PyArrow schema applied to every row. Field names are used for
        the missing/extra column check; field types are enforced by
        pyarrow at flush time.
    batch_size:
        Buffer size before an automatic flush. Default ``100``.
    immutable_batches:
        When true, install each completed batch without replacement and accept
        an existing batch index only when its complete parquet bytes match.
        Clustered Stage 4 uses this to make concurrent/resumed writes fail
        closed. The default false path retains the legacy writer exactly.
    expected_existing_indices:
        Optional immutable-mode snapshot taken during resume validation. The
        constructor rejects any batch that appeared or disappeared between
        that validation and writer construction instead of advancing past it
        and duplicating its logical rows under a new index.
    """

    def __init__(
        self,
        *,
        out_dir: Path,
        schema: pa.Schema,
        batch_size: int = 100,
        immutable_batches: bool = False,
        expected_existing_indices: tuple[int, ...] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise AxBenchBaseError(
                f"batch_size must be positive, got {batch_size}"
            )
        self._out_dir = out_dir
        self._schema = schema
        self._batch_size = batch_size
        self._immutable_batches = immutable_batches
        self._buffer: list[dict[str, Any]] = []
        self._closed = False
        self._expected_columns = set(schema.names)

        self._out_dir.mkdir(parents=True, exist_ok=True)
        existing = _scan_batch_indices(self._out_dir)
        if expected_existing_indices is not None:
            if not immutable_batches:
                raise AxBenchBaseError(
                    "expected_existing_indices is valid only for immutable batches"
                )
            if tuple(existing) != expected_existing_indices:
                raise AxBenchBaseError(
                    "Clustered Stage 4 generation batches changed after resume "
                    "validation; concurrent generation under one run_tag is "
                    "not resumable. Use a fresh run tag."
                )
        self._next_index = (existing[-1] + 1) if existing else 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, row: dict[str, Any]) -> None:
        """Buffer ``row`` and auto-flush when the buffer hits ``batch_size``.

        Raises
        ------
        AxBenchBaseError
            If the writer is closed, or ``row`` is missing a schema
            column, or carries an unexpected column.
        """
        if self._closed:
            raise AxBenchBaseError("cannot append to a closed ParquetAppendWriter")

        row_keys = set(row.keys())
        missing = self._expected_columns - row_keys
        extra = row_keys - self._expected_columns
        if missing:
            raise AxBenchBaseError(
                f"Row missing required columns: {sorted(missing)}"
            )
        if extra:
            raise AxBenchBaseError(
                f"Row has unexpected columns: {sorted(extra)}"
            )

        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Write the current buffer to a new batch file and clear it.

        A no-op if the buffer is empty. Raises :class:`AxBenchBaseError`
        if the writer is closed.
        """
        if self._closed:
            raise AxBenchBaseError("cannot flush a closed ParquetAppendWriter")
        if not self._buffer:
            return
        n_rows = len(self._buffer)
        out_path = self._out_dir / _batch_filename(self._next_index)
        with profile_region("parquet_writer.flush", rows=n_rows, out_path=out_path):
            table = pa.Table.from_pylist(self._buffer, schema=self._schema)
            if not self._immutable_batches:
                pq.write_table(table, out_path)
            else:
                sink = pa.BufferOutputStream()
                pq.write_table(table, sink)
                try:
                    install_bytes_once(
                        sink.getvalue().to_pybytes(),
                        out_path,
                        surface="clustered Stage 4 parquet batch",
                    )
                except ImmutableArtifactError as exc:
                    raise AxBenchBaseError(str(exc)) from exc
        self._next_index += 1
        self._buffer = []
        record_counter("parquet_writer.rows_flushed", n_rows, out_dir=self._out_dir)

    def close(self) -> None:
        """Flush any remaining rows and mark the writer closed.

        Idempotent: calling ``close()`` more than once is a no-op after
        the first call.
        """
        if self._closed:
            return
        # Flush first, THEN flip the closed flag so the guard in flush()
        # doesn't reject our own final call.
        if self._buffer:
            self.flush()
        self._closed = True

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> ParquetAppendWriter:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        self.close()


def load_existing(
    out_dir: Path,
    schema: pa.Schema | None = None,
) -> pa.Table:
    """Load all ``batch_NNNNNN.parquet`` files under ``out_dir`` in order.

    Parameters
    ----------
    out_dir:
        Directory to scan. A missing or empty directory yields an empty
        table.
    schema:
        Optional schema. If provided:

        * Every on-disk batch is validated against it (raises
          :class:`AxBenchBaseError` on mismatch).
        * An empty / missing directory returns an empty table with
          exactly this schema.

        If ``None``, schemas are not checked and an empty directory
        returns a zero-column empty table.

    Returns
    -------
    pa.Table
        A single table containing all batches concatenated in
        batch-index order.
    """
    indices = _scan_batch_indices(out_dir)
    if not indices:
        if schema is not None:
            return pa.Table.from_pylist([], schema=schema)
        return pa.table({})

    tables: list[pa.Table] = []
    for index in indices:
        path = out_dir / _batch_filename(index)
        with profile_region("parquet_writer.load_batch", path=path):
            table = pq.read_table(path)
            if schema is not None and not table.schema.equals(schema):
                raise AxBenchBaseError(
                    f"schema mismatch in {path}: "
                    f"expected {schema}, got {table.schema}"
                )
            tables.append(table)
    record_counter("parquet_writer.batches_loaded", len(tables), out_dir=out_dir)
    return pa.concat_tables(tables)
