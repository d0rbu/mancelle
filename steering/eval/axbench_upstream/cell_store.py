"""Crash-safe placement-free result storage for AxBench-Upstream cells."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from steering.eval.axbench_upstream.work_grid import CellKey
from steering.immutable_io import ImmutableArtifactError, install_bytes_once

STORE_SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 32
_IDENTITY_NAME = "identity.json"
_TRIPWIRE_NAME = "tripwire.json"
_FIT_ARTIFACTS_NAME = "fit_artifacts.json"
_RANK_PATTERN = re.compile(r"^rank_(\d{5})$")
_BATCH_PATTERN = re.compile(r"^batch_(\d{6})\.parquet$")
_RESULT_SCHEMA = pa.schema(
    [
        pa.field("driver", pa.string(), nullable=False),
        pa.field("method", pa.string(), nullable=False),
        pa.field("source_row_position", pa.int64(), nullable=False),
        pa.field("concept_id", pa.int64(), nullable=False),
        pa.field("input_id", pa.int64(), nullable=False),
        pa.field("factor_hex", pa.string(), nullable=False),
        pa.field("selection_seed", pa.int64(), nullable=False),
        pa.field("token_seed", pa.int64(), nullable=False),
        pa.field("generation", pa.string(), nullable=False),
    ],
    metadata={
        b"format": b"axbench-upstream-cell-results-v1",
        b"ordering": b"append-only; canonical order is recovered from CellKey",
    },
)


class CellStoreError(RuntimeError):
    """A resume store is malformed, stale, incomplete, or conflicting."""


def _canonical_json(payload: Any, *, label: str) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CellStoreError(f"{label} must be strict JSON: {exc}") from exc
    return encoded + b"\n"


def _strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_rank(*, rank: int, world_size: int) -> None:
    if type(world_size) is not int or world_size <= 0:
        raise CellStoreError(
            f"world_size must be a positive int excluding bool, got {world_size!r}"
        )
    if type(rank) is not int or not 0 <= rank < world_size:
        raise CellStoreError(
            f"rank must be an int excluding bool in [0, {world_size}), got {rank!r}"
        )


def _identity_document(
    semantic_identity: Mapping[str, Any],
    cells: Sequence[CellKey],
) -> tuple[bytes, str]:
    grid_bytes = _canonical_json(
        [cell.to_dict() for cell in cells],
        label="logical cell grid",
    )
    bound_identity = {
        "semantic_identity": dict(semantic_identity),
        "logical_cell_count": len(cells),
        "logical_cell_grid_sha256": _sha256_bytes(grid_bytes),
    }
    semantic_bytes = _canonical_json(bound_identity, label="semantic identity")
    fingerprint = _sha256_bytes(semantic_bytes)
    document = {
        "schema_version": STORE_SCHEMA_VERSION,
        "algorithm": "rank-striped-cell-store-v1",
        "semantic_fingerprint": fingerprint,
        **bound_identity,
    }
    return _canonical_json(document, label="resume identity"), fingerprint


def _cell_from_row(row: Mapping[str, Any], *, path: Path, row_index: int) -> CellKey:
    driver = row["driver"]
    if driver not in {"project", "encdec"}:
        raise CellStoreError(f"{path}: row {row_index} has invalid driver {driver!r}")
    integer_fields = (
        "source_row_position",
        "concept_id",
        "input_id",
        "selection_seed",
        "token_seed",
    )
    for field in integer_fields:
        value = row[field]
        if type(value) is not int or value < 0:
            raise CellStoreError(f"{path}: row {row_index} field {field} must be a nonnegative int")
    for field in ("method", "factor_hex"):
        value = row[field]
        if not isinstance(value, str) or not value:
            raise CellStoreError(f"{path}: row {row_index} field {field} must be a nonempty string")
    return CellKey(
        driver=driver,
        method=row["method"],
        source_row_position=row["source_row_position"],
        concept_id=row["concept_id"],
        input_id=row["input_id"],
        factor_hex=row["factor_hex"],
        selection_seed=row["selection_seed"],
        token_seed=row["token_seed"],
    )


def _result_row(cell: CellKey, generation: str) -> dict[str, int | str]:
    return {**cell.to_dict(), "generation": generation}


class CellResultStore:
    """Validate, append, and reconstruct immutable logical-cell results."""

    def __init__(
        self,
        *,
        root: Path,
        cells: Sequence[CellKey],
        tripwire_cells: Sequence[CellKey],
        semantic_identity: Mapping[str, Any],
        rank: int,
        world_size: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        _validate_rank(rank=rank, world_size=world_size)
        if type(batch_size) is not int or batch_size <= 0:
            raise CellStoreError(
                f"batch_size must be a positive int excluding bool, got {batch_size!r}"
            )
        self.root = Path(root).expanduser().resolve()
        self.rank = rank
        self.world_size = world_size
        self._batch_size = batch_size
        self._cells = tuple(cells)
        if not self._cells:
            raise CellStoreError("cell store requires at least one expected cell")
        if len(set(self._cells)) != len(self._cells):
            raise CellStoreError("expected cell grid contains duplicate CellKey values")
        self._expected = frozenset(self._cells)
        self._tripwire_cells = tuple(tripwire_cells)
        if not self._tripwire_cells:
            raise CellStoreError("tripwire cell population must not be empty")
        if len(set(self._tripwire_cells)) != len(self._tripwire_cells):
            raise CellStoreError("tripwire cell population contains duplicate CellKey values")
        unknown_tripwire = [cell for cell in self._tripwire_cells if cell not in self._expected]
        if unknown_tripwire:
            raise CellStoreError(
                f"tripwire population contains out-of-grid cells: {unknown_tripwire[:3]}"
            )
        identity_bytes, self.semantic_fingerprint = _identity_document(
            semantic_identity,
            self._cells,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            install_bytes_once(
                identity_bytes,
                self.root / _IDENTITY_NAME,
                surface="AxBench-Upstream resume identity",
            )
        except ImmutableArtifactError as exc:
            raise CellStoreError(str(exc)) from exc
        self._results = self._scan()
        receipt_exists = (self.root / _TRIPWIRE_NAME).is_file()
        durable_bulk = [
            cell for cell in self._results if cell not in frozenset(self._tripwire_cells)
        ]
        if durable_bulk and not receipt_exists:
            raise CellStoreError(
                "resume store contains durable bulk cells without a tripwire receipt; "
                f"first={durable_bulk[0]}"
            )
        if receipt_exists:
            self.require_tripwire_passed()
            missing_tripwire = [cell for cell in self._tripwire_cells if cell not in self._results]
            if missing_tripwire:
                raise CellStoreError(
                    "tripwire receipt exists but its cell population is incomplete; "
                    f"first={missing_tripwire[0]}"
                )
        self._rank_dir = self.root / f"rank_{rank:05d}"
        self._rank_dir.mkdir(parents=True, exist_ok=True)
        own_indices = self._batch_indices(self._rank_dir)
        self._next_batch_index = own_indices[-1] + 1 if own_indices else 0
        self._buffer: list[dict[str, int | str]] = []
        self._closed = False

    @staticmethod
    def _batch_indices(directory: Path) -> list[int]:
        indices: list[int] = []
        for path in directory.iterdir():
            if path.is_symlink():
                raise CellStoreError(f"resume rank directory contains symlink: {path}")
            if not path.is_file():
                raise CellStoreError(f"resume rank directory contains non-file entry: {path}")
            match = _BATCH_PATTERN.fullmatch(path.name)
            if match is None:
                raise CellStoreError(f"resume rank directory contains unknown file: {path}")
            indices.append(int(match.group(1)))
        indices.sort()
        if indices != list(range(len(indices))):
            raise CellStoreError(
                f"resume rank directory has non-contiguous batch indices: {directory}: {indices}"
            )
        return indices

    def _scan(self) -> dict[CellKey, str]:
        allowed_root_files = {_IDENTITY_NAME, _TRIPWIRE_NAME, _FIT_ARTIFACTS_NAME}
        rank_directories: list[tuple[int, Path]] = []
        for path in self.root.iterdir():
            if path.is_symlink():
                raise CellStoreError(f"resume store contains symlink: {path}")
            if path.is_file():
                if path.name not in allowed_root_files:
                    raise CellStoreError(f"resume store contains unknown file: {path}")
                continue
            if not path.is_dir():
                raise CellStoreError(f"resume store contains unsupported entry: {path}")
            match = _RANK_PATTERN.fullmatch(path.name)
            if match is None:
                raise CellStoreError(f"resume store contains unknown directory: {path}")
            rank_directories.append((int(match.group(1)), path))

        results: dict[CellKey, str] = {}
        for _stored_rank, directory in sorted(rank_directories):
            for batch_index in self._batch_indices(directory):
                path = directory / f"batch_{batch_index:06d}.parquet"
                try:
                    table = pq.read_table(path)
                except Exception as exc:
                    raise CellStoreError(f"could not read resume batch {path}: {exc}") from exc
                if not table.schema.equals(_RESULT_SCHEMA, check_metadata=True):
                    raise CellStoreError(
                        f"resume batch schema mismatch at {path}: "
                        f"expected {_RESULT_SCHEMA}, got {table.schema}"
                    )
                if table.num_rows == 0:
                    raise CellStoreError(f"resume batch must not be empty: {path}")
                for row_index, row in enumerate(table.to_pylist()):
                    cell = _cell_from_row(row, path=path, row_index=row_index)
                    generation = row["generation"]
                    if not isinstance(generation, str):
                        raise CellStoreError(f"{path}: row {row_index} generation must be a string")
                    if cell not in self._expected:
                        raise CellStoreError(
                            f"resume batch contains out-of-grid cell at {path} row {row_index}: "
                            f"{cell}"
                        )
                    previous = results.get(cell)
                    if previous is not None and previous != generation:
                        raise CellStoreError(
                            f"resume batches contain conflicting generations for {cell}: "
                            f"{previous!r} != {generation!r}"
                        )
                    results[cell] = generation
        return results

    @property
    def completed_count(self) -> int:
        return len(self._results)

    def missing(self, cells: Sequence[CellKey] | None = None) -> tuple[CellKey, ...]:
        selected = self._cells if cells is None else tuple(cells)
        unknown = [cell for cell in selected if cell not in self._expected]
        if unknown:
            raise CellStoreError(f"missing() received out-of-grid cells: {unknown[:3]}")
        return tuple(cell for cell in selected if cell not in self._results)

    def append(self, cell: CellKey, generation: str) -> bool:
        if self._closed:
            raise CellStoreError("cannot append to a closed cell store")
        if cell not in self._expected:
            raise CellStoreError(f"cannot append out-of-grid cell: {cell}")
        if not isinstance(generation, str):
            raise CellStoreError(f"generation must be a string, got {type(generation).__name__}")
        previous = self._results.get(cell)
        if previous is not None:
            if previous != generation:
                raise CellStoreError(
                    f"cell already has a conflicting generation: {cell}: "
                    f"{previous!r} != {generation!r}"
                )
            return False
        self._results[cell] = generation
        self._buffer.append(_result_row(cell, generation))
        if len(self._buffer) >= self._batch_size:
            self.flush()
        return True

    def flush(self) -> None:
        if self._closed:
            raise CellStoreError("cannot flush a closed cell store")
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=_RESULT_SCHEMA)
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        path = self._rank_dir / f"batch_{self._next_batch_index:06d}.parquet"
        try:
            installed = install_bytes_once(
                sink.getvalue().to_pybytes(),
                path,
                surface="AxBench-Upstream resume batch",
            )
        except ImmutableArtifactError as exc:
            raise CellStoreError(str(exc)) from exc
        if not installed:
            raise CellStoreError(
                f"resume batch index was already occupied during append: {path}; "
                "do not run concurrent producers with the same output paths"
            )
        self._next_batch_index += 1
        self._buffer = []

    def mark_tripwire_passed(self) -> None:
        missing = [cell for cell in self._tripwire_cells if cell not in self._results]
        if missing:
            raise CellStoreError(
                f"cannot mark tripwire passed with {len(missing)} missing cells: {missing[:3]}"
            )
        # The receipt is a durable claim. Make every result supporting that
        # claim durable before installing it, even when callers did not issue
        # an explicit flush themselves.
        self.flush()
        cell_bytes = _canonical_json(
            [cell.to_dict() for cell in self._tripwire_cells],
            label="tripwire cell population",
        )
        document = {
            "schema_version": STORE_SCHEMA_VERSION,
            "semantic_fingerprint": self.semantic_fingerprint,
            "cell_count": len(self._tripwire_cells),
            "cell_grid_sha256": _sha256_bytes(cell_bytes),
        }
        try:
            install_bytes_once(
                _canonical_json(document, label="tripwire receipt"),
                self.root / _TRIPWIRE_NAME,
                surface="AxBench-Upstream tripwire receipt",
            )
        except ImmutableArtifactError as exc:
            raise CellStoreError(str(exc)) from exc

    def require_tripwire_passed(self) -> None:
        receipt = self.root / _TRIPWIRE_NAME
        if not receipt.is_file():
            raise CellStoreError(f"tripwire receipt is missing: {receipt}")
        cell_bytes = _canonical_json(
            [cell.to_dict() for cell in self._tripwire_cells],
            label="tripwire cell population",
        )
        expected = {
            "schema_version": STORE_SCHEMA_VERSION,
            "semantic_fingerprint": self.semantic_fingerprint,
            "cell_count": len(self._tripwire_cells),
            "cell_grid_sha256": _sha256_bytes(cell_bytes),
        }
        try:
            actual = json.loads(
                receipt.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_json_object_pairs,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise CellStoreError(f"tripwire receipt is unreadable: {receipt}: {exc}") from exc
        if actual != expected:
            raise CellStoreError(
                f"tripwire receipt does not match the current run identity/grid: {receipt}"
            )

    def complete_results(self) -> dict[CellKey, str]:
        missing = self.missing()
        if missing:
            raise CellStoreError(
                f"cell store is incomplete: {len(missing)} of {len(self._cells)} cells missing; "
                f"first={missing[0]}"
            )
        return {cell: self._results[cell] for cell in self._cells}

    def results_for(self, cells: Sequence[CellKey]) -> dict[CellKey, str]:
        selected = tuple(cells)
        missing = self.missing(selected)
        if missing:
            raise CellStoreError(
                f"requested result population is incomplete: {len(missing)} of "
                f"{len(selected)} cells missing; first={missing[0]}"
            )
        return {cell: self._results[cell] for cell in selected}

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True

    def __enter__(self) -> CellResultStore:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


__all__ = (
    "DEFAULT_BATCH_SIZE",
    "STORE_SCHEMA_VERSION",
    "CellResultStore",
    "CellStoreError",
)
