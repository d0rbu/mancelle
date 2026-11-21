"""Validated, atomic publication helpers for AxBench-Adv artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.atomic_io import atomic_write_text
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.schemas import validate_universal_suffixes


def logical_table_digest(table: pa.Table) -> str:
    """Digest schema and logical row content, independent of Parquet encoding."""

    payload = {
        "schema": [[field.name, str(field.type), field.nullable] for field in table.schema],
        "rows": table.to_pylist(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _temporary_path(final_path: Path) -> Path:
    return final_path.parent / (f"{final_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")


def write_validated_parquet(
    table: pa.Table,
    final_path: Path,
    *,
    validator: Callable[[pa.Table], None],
) -> str:
    """Write, reread, validate, compare logical content, then atomically rename."""

    validator(table)
    expected_digest = logical_table_digest(table)
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_path(final_path)
    try:
        pq.write_table(table, temporary_path)
        descriptor = os.open(temporary_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        reread = pq.read_table(temporary_path)
        validator(reread)
        observed_digest = logical_table_digest(reread)
        if observed_digest != expected_digest:
            raise AxBenchAdvError("reread artifact logical digest differs from the source table")
        os.replace(temporary_path, final_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return expected_digest


def write_suffix_artifacts(
    *,
    suffixes: pa.Table,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Publish one suffix cell, deriving its completion identity from the row."""

    output_dir = Path(output_dir)
    if not isinstance(suffixes, pa.Table) or suffixes.num_rows != 1:
        raise AxBenchAdvError("a suffix cell must contain exactly one row")
    validate_universal_suffixes(suffixes)
    if output_dir.exists():
        entries = {entry.name for entry in output_dir.iterdir()}
        unknown = entries - {"universal_suffixes.parquet", "manifest.json"}
        if "manifest.json" in entries:
            raise AxBenchAdvError(f"artifact cell is already complete: {output_dir}")
        if unknown:
            raise AxBenchAdvError(
                f"incomplete artifact cell contains unknown entries: {sorted(unknown)}"
            )
        (output_dir / "universal_suffixes.parquet").unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_path = output_dir / "universal_suffixes.parquet"
    digest = write_validated_parquet(
        suffixes,
        suffix_path,
        validator=validate_universal_suffixes,
    )
    manifest = {
        "schema_version": 1,
        "identity": {
            field: suffixes.column(field)[0].as_py()
            for field in ("method", "concept_id", "proposal_count")
        },
        "files": {
            suffix_path.name: {
                "logical_sha256": digest,
                "rows": suffixes.num_rows,
            }
        },
    }
    manifest_text = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    manifest_path = output_dir / "manifest.json"
    atomic_write_text(manifest_text, manifest_path)
    return suffix_path, manifest_path


__all__ = [
    "logical_table_digest",
    "write_suffix_artifacts",
    "write_validated_parquet",
]
