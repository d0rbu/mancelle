"""Shared metadata helpers for AxBench-Base artifacts."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from steering.eval.axbench_base.types import AxBenchBaseError

EXTRACTION_TOKENIZER_METADATA_KEY = "extraction_tokenizer_id"


def schema_with_extraction_tokenizer_metadata(
    schema: pa.Schema,
    tokenizer_id: str,
) -> pa.Schema:
    """Return ``schema`` tagged with the extraction tokenizer identity."""
    metadata = dict(schema.metadata or {})
    metadata[EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8")] = str(
        tokenizer_id
    ).encode("utf-8")
    return schema.with_metadata(metadata)


def extraction_tokenizer_metadata_from_schema(schema: pa.Schema) -> str | None:
    metadata = schema.metadata or {}
    raw = metadata.get(EXTRACTION_TOKENIZER_METADATA_KEY.encode("utf-8"))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def validate_extraction_tokenizer_metadata(
    path: Path,
    *,
    expected_tokenizer_id: str,
    consumer: str,
    artifact_name: str,
    schema: pa.Schema | None = None,
) -> None:
    """Validate parquet artifact metadata against the current tokenizer."""
    parquet_schema = schema if schema is not None else pq.read_schema(path)
    actual = extraction_tokenizer_metadata_from_schema(parquet_schema)
    if actual is None:
        raise AxBenchBaseError(
            f"{consumer}: {artifact_name} at {path} is missing "
            f"{EXTRACTION_TOKENIZER_METADATA_KEY!r} metadata. Delete or "
            "regenerate the artifact with the current extraction tokenizer."
        )
    if actual != expected_tokenizer_id:
        raise AxBenchBaseError(
            f"{consumer}: {artifact_name} at {path} was created with "
            f"extraction tokenizer {actual!r}, but the current config uses "
            f"{expected_tokenizer_id!r}. Regenerate this artifact with the "
            "current extraction tokenizer."
        )
