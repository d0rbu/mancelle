"""Schema-v2 AxBench seed artifacts and strict runtime validation."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
from safetensors import safe_open
from safetensors.torch import load_file

from nanochat.atomic_io import atomic_safetensors_save
from nanochat.tokenizer import get_tokenizer
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError, SeedRow
from steering.feature_paths import parse_feature_dir
from steering.immutable_io import is_sha256_hex
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth
from steering.segmentation import compute_tokenizer_checksum

SEED_SCHEMA_VERSION = 2
SEED_TOKEN_SEMANTICS = (
    "stored_prefix_and_continuation_are_exact_bos_free_nanochat_source_slices;"
    "one_pipeline_owned_bos_is_added_only_at_model_input"
)
_FRESH_RUN_TAG = (
    "Legacy, incomplete, or incompatible seed artifacts cannot be repaired "
    "safely; use a fresh AxBench run tag and rerun Stage 0b."
)

_SEED_TENSOR_SCHEMA = (
    ("prefix_text_offsets", t.int64),
    ("prefix_text_bytes", t.uint8),
    ("prefix_token_offsets", t.int64),
    ("prefix_content_tokens", t.int32),
    ("continuation_text_offsets", t.int64),
    ("continuation_text_bytes", t.uint8),
    ("continuation_token_offsets", t.int64),
    ("continuation_content_tokens", t.int32),
    ("sae_feature_offsets", t.int64),
    ("sae_feature_ids", t.int32),
    ("sae_feature_mags", t.float32),
    ("source_nanochat_token_count", t.int32),
    ("prefix_source_token_start", t.int32),
    ("prefix_source_token_end", t.int32),
    ("shard_idx", t.int32),
    ("doc_idx", t.int32),
)
_REQUIRED_TENSORS = frozenset(name for name, _dtype in _SEED_TENSOR_SCHEMA)
_EXPECTED_DTYPES = dict(_SEED_TENSOR_SCHEMA)
_STAGE1B_SEED_ROWS_IDENTITY_VERSION = "stage1b_seed_rows_v2"

_REQUIRED_METADATA = frozenset(
    {
        "schema_version",
        "split",
        "seed_fingerprint",
        "token_semantics",
        "nanochat_tokenizer_checksum",
        "extraction_tokenizer_checksum",
        "config_fingerprint",
        "seed_config_json",
        "source_parquet_hashes_json",
        "parameter_hashes_json",
        "runtime_identity_json",
        "extraction_settings_json",
    }
)

_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "seed_fingerprint",
        "token_semantics",
        "candidate_digest",
        "candidate_count",
        "skip_counts",
        "source_parquet_hashes",
        "config_fingerprint",
        "extraction_fingerprint",
        "nanochat_tokenizer_checksum",
        "extraction_tokenizer_checksum",
        "parameter_hashes",
        "runtime_identity",
        "extraction_settings",
        "files",
    }
)


@dataclass(frozen=True)
class SeedPoolExpectation:
    continuation_length: int
    prefix_cap: int
    n_sae_features: int
    nanochat_tokenizer_checksum: str
    config_fingerprint: str


@dataclass(frozen=True)
class SeedPoolSnapshot:
    """Validated seed identity plus the eval rows loaded by that validation."""

    fingerprint: str
    eval_rows: tuple[SeedRow, ...]


@dataclass(frozen=True)
class LoadedSeedSplit:
    """Validated seed rows and their ordered Stage 1b content identity."""

    rows: list[SeedRow]
    rows_sha256: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compute_extraction_fingerprint(
    *,
    extraction_settings: dict[str, Any],
    runtime_identity: dict[str, Any],
    parameter_hashes: dict[str, str],
) -> str:
    """Hash every semantic input to Stage 0b feature extraction."""

    return sha256_bytes(
        canonical_json(
            {
                "settings": extraction_settings,
                "runtime": runtime_identity,
                "parameter_hashes": parameter_hashes,
            }
        ).encode("utf-8")
    )


def compute_seed_fingerprint(
    *,
    candidate_digest: str,
    config_fingerprint: str,
    extraction_fingerprint: str,
    nanochat_tokenizer_checksum: str,
    source_parquet_hashes: dict[str, str],
) -> str:
    """Hash the complete semantic identity of one committed seed pool."""

    return sha256_bytes(
        canonical_json(
            {
                "candidate_digest": candidate_digest,
                "config_fingerprint": config_fingerprint,
                "extraction_fingerprint": extraction_fingerprint,
                "nanochat_tokenizer_checksum": nanochat_tokenizer_checksum,
                "source_parquet_hashes": source_parquet_hashes,
            }
        ).encode("utf-8")
    )




def seed_config_payload(cfg: AxBenchBaseConfig) -> dict[str, Any]:
    parsed = parse_feature_dir(Path(cfg.feature_dir))
    return {
        "feature_dir": str(Path(cfg.feature_dir).resolve()),
        "feature_directory": {
            "segment_size": parsed.segment_size,
            "layer": parsed.layer,
            "activation_type": parsed.activation_type.value,
            "pooling": parsed.pooling.value,
            "width": parsed.width.value,
            "model": parsed.model,
            "dataset_phase": parsed.dataset_phase.value,
            "context_independent_features": parsed.context_independent_features,
            "ignore_bos_features": parsed.ignore_bos_features,
        },
        "validation_shards": [int(value) for value in cfg.validation_shards],
        "seed_prefix_max_nanochat_tokens": cfg.seed_prefix_max_nanochat_tokens,
        "seed_continuation_nanochat_tokens": (
            cfg.seed_continuation_nanochat_tokens
        ),
        "seed_pool_train_ratio": cfg.seed_pool_train_ratio,
        "seed_pool_seed": cfg.seed_pool_seed,
        "extraction_max_tokens": cfg.extraction_max_tokens,
        "feature_extraction_model_id": cfg.feature_extraction_model_id,
        "neuronpedia_model_id": cfg.neuronpedia_model_id,
        "neuronpedia_source": cfg.neuronpedia_source,
        "extraction_tokenizer_id": cfg.extraction_tokenizer_id,
        "seed_extraction_candidate_shard_size": (
            cfg.seed_extraction_candidate_shard_size
        ),
        "seed_extraction_gpu_batch_size": cfg.seed_extraction_gpu_batch_size,
        "seed_extraction_cpu_batch_size": cfg.seed_extraction_cpu_batch_size,
    }


def seed_config_fingerprint(cfg: AxBenchBaseConfig) -> str:
    return sha256_bytes(canonical_json(seed_config_payload(cfg)).encode("utf-8"))


def expectation_from_config(
    cfg: AxBenchBaseConfig,
    tokenizer: Any,
) -> SeedPoolExpectation:
    parsed = parse_feature_dir(Path(cfg.feature_dir))
    return SeedPoolExpectation(
        continuation_length=cfg.seed_continuation_nanochat_tokens,
        prefix_cap=cfg.seed_prefix_max_nanochat_tokens,
        n_sae_features=SAE_WIDTH_TO_NUM_FEATURES[parsed.width],
        nanochat_tokenizer_checksum=compute_tokenizer_checksum(tokenizer),
        config_fingerprint=seed_config_fingerprint(cfg),
    )


def expectation_from_seed_file(
    path: Path,
    tokenizer: Any,
) -> SeedPoolExpectation:
    """Derive row invariants from one file's committed schema-v2 metadata."""

    metadata = _read_metadata(path)
    try:
        config_json = metadata["seed_config_json"]
        config_payload = json.loads(config_json)
        continuation_length = int(
            config_payload["seed_continuation_nanochat_tokens"]
        )
        prefix_cap = int(config_payload["seed_prefix_max_nanochat_tokens"])
        width = SAEWidth(str(config_payload["feature_directory"]["width"]))
        config_fingerprint = str(metadata["config_fingerprint"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(
            f"Cannot derive schema-v2 seed expectations from {path}. {_FRESH_RUN_TAG}"
        ) from exc
    if continuation_length <= 0 or prefix_cap < continuation_length:
        raise AxBenchBaseError(
            f"Seed metadata carries invalid M/cap values. {_FRESH_RUN_TAG}"
        )
    return SeedPoolExpectation(
        continuation_length=continuation_length,
        prefix_cap=prefix_cap,
        n_sae_features=SAE_WIDTH_TO_NUM_FEATURES[width],
        nanochat_tokenizer_checksum=compute_tokenizer_checksum(tokenizer),
        config_fingerprint=config_fingerprint,
    )


def validate_seed_row(
    row: SeedRow,
    *,
    expectation: SeedPoolExpectation,
    tokenizer: Any,
) -> None:
    m = expectation.continuation_length
    if len(row.continuation_content_nanochat_tokens) != m:
        raise AxBenchBaseError(
            f"SeedRow continuation length is {len(row.continuation_content_nanochat_tokens)}, "
            f"expected configured M={m}. {_FRESH_RUN_TAG}"
        )
    expected_start = max(0, row.prefix_source_token_end - expectation.prefix_cap)
    if row.prefix_source_token_start != expected_start:
        raise AxBenchBaseError(
            "SeedRow prefix start does not match the configured cap: "
            f"expected {expected_start}, got {row.prefix_source_token_start}. "
            f"{_FRESH_RUN_TAG}"
        )
    if any(feature_id >= expectation.n_sae_features for feature_id in row.sae_feature_ids):
        raise AxBenchBaseError(
            "SeedRow contains an SAE feature outside the configured dictionary "
            f"width {expectation.n_sae_features}. {_FRESH_RUN_TAG}"
        )
    prefix_decoded = tokenizer.decode(row.prefix_content_nanochat_tokens)
    continuation_decoded = tokenizer.decode(
        row.continuation_content_nanochat_tokens
    )
    if prefix_decoded != row.prefix_text:
        raise AxBenchBaseError(
            "SeedRow prefix tokens do not decode exactly to prefix_text. "
            f"{_FRESH_RUN_TAG}"
        )
    if continuation_decoded != row.continuation_text:
        raise AxBenchBaseError(
            "SeedRow continuation tokens do not decode exactly to "
            f"continuation_text. {_FRESH_RUN_TAG}"
        )


def _append_bytes(parts: list[bytes], offsets: list[int], value: str) -> None:
    encoded = value.encode("utf-8")
    parts.append(encoded)
    offsets.append(offsets[-1] + len(encoded))


def _bytes_tensor(parts: list[bytes]) -> t.Tensor:
    payload = b"".join(parts)
    return (
        t.tensor(list(payload), dtype=t.uint8)
        if payload
        else t.empty(0, dtype=t.uint8)
    )


def seed_rows_to_tensors(rows: list[SeedRow]) -> dict[str, t.Tensor]:
    prefix_text_parts: list[bytes] = []
    prefix_text_offsets = [0]
    continuation_text_parts: list[bytes] = []
    continuation_text_offsets = [0]
    prefix_tokens: list[int] = []
    prefix_token_offsets = [0]
    continuation_tokens: list[int] = []
    continuation_token_offsets = [0]
    feature_ids: list[int] = []
    feature_mags: list[float] = []
    feature_offsets = [0]

    for row in rows:
        _append_bytes(prefix_text_parts, prefix_text_offsets, row.prefix_text)
        _append_bytes(
            continuation_text_parts,
            continuation_text_offsets,
            row.continuation_text,
        )
        prefix_tokens.extend(row.prefix_content_nanochat_tokens)
        prefix_token_offsets.append(len(prefix_tokens))
        continuation_tokens.extend(row.continuation_content_nanochat_tokens)
        continuation_token_offsets.append(len(continuation_tokens))
        feature_ids.extend(row.sae_feature_ids)
        feature_mags.extend(row.sae_feature_mags)
        feature_offsets.append(len(feature_ids))

    return {
        "prefix_text_offsets": t.tensor(prefix_text_offsets, dtype=t.int64),
        "prefix_text_bytes": _bytes_tensor(prefix_text_parts),
        "prefix_token_offsets": t.tensor(prefix_token_offsets, dtype=t.int64),
        "prefix_content_tokens": t.tensor(prefix_tokens, dtype=t.int32),
        "continuation_text_offsets": t.tensor(
            continuation_text_offsets, dtype=t.int64
        ),
        "continuation_text_bytes": _bytes_tensor(continuation_text_parts),
        "continuation_token_offsets": t.tensor(
            continuation_token_offsets, dtype=t.int64
        ),
        "continuation_content_tokens": t.tensor(
            continuation_tokens, dtype=t.int32
        ),
        "sae_feature_offsets": t.tensor(feature_offsets, dtype=t.int64),
        "sae_feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "sae_feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "source_nanochat_token_count": t.tensor(
            [row.source_nanochat_token_count for row in rows], dtype=t.int32
        ),
        "prefix_source_token_start": t.tensor(
            [row.prefix_source_token_start for row in rows], dtype=t.int32
        ),
        "prefix_source_token_end": t.tensor(
            [row.prefix_source_token_end for row in rows], dtype=t.int32
        ),
        "shard_idx": t.tensor([row.shard_idx for row in rows], dtype=t.int32),
        "doc_idx": t.tensor([row.doc_idx for row in rows], dtype=t.int32),
    }


def save_seed_pool(
    path: Path,
    rows: list[SeedRow],
    *,
    metadata: dict[str, str],
) -> None:
    missing_metadata = sorted(_REQUIRED_METADATA - set(metadata))
    if missing_metadata:
        raise AxBenchBaseError(
            f"Cannot write a schema-v2 seed file without metadata fields "
            f"{missing_metadata}. {_FRESH_RUN_TAG}"
        )
    if metadata.get("schema_version") != str(SEED_SCHEMA_VERSION):
        raise AxBenchBaseError("Refusing to write a non-v2 seed file")
    if metadata.get("split") not in {"train", "eval"}:
        raise AxBenchBaseError("Seed metadata split must be 'train' or 'eval'")
    if metadata.get("token_semantics") != SEED_TOKEN_SEMANTICS:
        raise AxBenchBaseError("Refusing to write seed metadata with wrong semantics")
    for key in (
        "seed_fingerprint",
        "nanochat_tokenizer_checksum",
        "extraction_tokenizer_checksum",
        "config_fingerprint",
    ):
        if not is_sha256_hex(metadata.get(key)):
            raise AxBenchBaseError(f"Seed metadata {key} must be a SHA-256 digest")
    for key in (
        "seed_config_json",
        "source_parquet_hashes_json",
        "parameter_hashes_json",
        "runtime_identity_json",
        "extraction_settings_json",
    ):
        try:
            payload = json.loads(metadata[key])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AxBenchBaseError(f"Seed metadata {key} must be canonical JSON") from exc
        if canonical_json(payload) != metadata[key]:
            raise AxBenchBaseError(f"Seed metadata {key} must be canonical JSON")
    if sha256_bytes(metadata["seed_config_json"].encode("utf-8")) != metadata[
        "config_fingerprint"
    ]:
        raise AxBenchBaseError("Seed metadata config payload/fingerprint mismatch")
    atomic_safetensors_save(seed_rows_to_tensors(rows), path, metadata=metadata)


def _validate_offsets(
    name: str,
    offsets: list[int],
    *,
    n_rows: int,
    payload_length: int,
) -> None:
    if len(offsets) != n_rows + 1 or not offsets or offsets[0] != 0:
        raise AxBenchBaseError(f"Malformed seed {name} offsets. {_FRESH_RUN_TAG}")
    if offsets != sorted(offsets) or offsets[-1] != payload_length:
        raise AxBenchBaseError(
            f"Malformed seed {name} offsets/payload length. {_FRESH_RUN_TAG}"
        )


def _read_metadata(path: Path) -> dict[str, str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception as exc:
        raise AxBenchBaseError(
            f"Cannot read seed metadata from {path}: {exc}. {_FRESH_RUN_TAG}"
        ) from exc


def _load_manifest_for_file(path: Path) -> dict[str, Any]:
    manifest_path = path.parent / "seed_pool_manifest.json"
    if not manifest_path.is_file():
        raise AxBenchBaseError(
            f"Seed manifest is missing at {manifest_path}. {_FRESH_RUN_TAG}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AxBenchBaseError(
            f"Seed manifest is unreadable at {manifest_path}. {_FRESH_RUN_TAG}"
        ) from exc
    if manifest.get("schema_version") != SEED_SCHEMA_VERSION:
        raise AxBenchBaseError(
            "Seed manifest has an unsupported schema version. " + _FRESH_RUN_TAG
        )
    missing_manifest_fields = sorted(_REQUIRED_MANIFEST_FIELDS - set(manifest))
    if missing_manifest_fields:
        raise AxBenchBaseError(
            "Seed manifest is missing required fields "
            f"{missing_manifest_fields}. {_FRESH_RUN_TAG}"
        )
    if manifest.get("token_semantics") != SEED_TOKEN_SEMANTICS:
        raise AxBenchBaseError("Seed manifest token semantics mismatch. " + _FRESH_RUN_TAG)
    for field in (
        "seed_fingerprint",
        "candidate_digest",
        "config_fingerprint",
        "extraction_fingerprint",
        "nanochat_tokenizer_checksum",
        "extraction_tokenizer_checksum",
    ):
        if not is_sha256_hex(manifest.get(field)):
            raise AxBenchBaseError(
                f"Seed manifest {field} is not a SHA-256 digest. {_FRESH_RUN_TAG}"
            )
    candidate_count = manifest.get("candidate_count")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count <= 0
    ):
        raise AxBenchBaseError(
            f"Seed manifest candidate count is invalid. {_FRESH_RUN_TAG}"
        )
    source_hashes = manifest.get("source_parquet_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes or any(
        not isinstance(source, str) or not source or not is_sha256_hex(digest)
        for source, digest in source_hashes.items()
    ):
        raise AxBenchBaseError(
            f"Seed manifest source parquet hashes are malformed. {_FRESH_RUN_TAG}"
        )
    parameter_hashes = manifest.get("parameter_hashes")
    if not isinstance(parameter_hashes, dict) or not parameter_hashes or any(
        not isinstance(name, str) or not name or not is_sha256_hex(digest)
        for name, digest in parameter_hashes.items()
    ):
        raise AxBenchBaseError(
            f"Seed manifest parameter hashes are malformed. {_FRESH_RUN_TAG}"
        )
    for field in ("runtime_identity", "extraction_settings"):
        value = manifest.get(field)
        if not isinstance(value, dict) or not value:
            raise AxBenchBaseError(
                f"Seed manifest {field} must be a non-empty mapping. {_FRESH_RUN_TAG}"
            )
    expected_extraction_fingerprint = compute_extraction_fingerprint(
        extraction_settings=manifest["extraction_settings"],
        runtime_identity=manifest["runtime_identity"],
        parameter_hashes=manifest["parameter_hashes"],
    )
    if manifest["extraction_fingerprint"] != expected_extraction_fingerprint:
        raise AxBenchBaseError(
            "Seed manifest extraction fingerprint does not match its semantic "
            f"payload. {_FRESH_RUN_TAG}"
        )
    expected_seed_fingerprint = compute_seed_fingerprint(
        candidate_digest=manifest["candidate_digest"],
        config_fingerprint=manifest["config_fingerprint"],
        extraction_fingerprint=manifest["extraction_fingerprint"],
        nanochat_tokenizer_checksum=manifest["nanochat_tokenizer_checksum"],
        source_parquet_hashes=manifest["source_parquet_hashes"],
    )
    if manifest["seed_fingerprint"] != expected_seed_fingerprint:
        raise AxBenchBaseError(
            "Seed manifest fingerprint does not match its semantic payload. "
            f"{_FRESH_RUN_TAG}"
        )
    files = manifest.get("files")
    expected_names = {"seeds_train.safetensors", "seeds_eval.safetensors"}
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AxBenchBaseError(
            "Seed manifest must commit exactly the train and eval seed files. "
            + _FRESH_RUN_TAG
        )
    for filename, expected_split in (
        ("seeds_train.safetensors", "train"),
        ("seeds_eval.safetensors", "eval"),
    ):
        sibling = path.parent / filename
        entry = files.get(filename)
        if not sibling.is_file() or not isinstance(entry, dict):
            raise AxBenchBaseError(
                f"Seed artifact pair is torn at {sibling}. {_FRESH_RUN_TAG}"
            )
        if entry.get("split") != expected_split:
            raise AxBenchBaseError(
                f"Seed manifest split mismatch for {filename}. {_FRESH_RUN_TAG}"
            )
        row_count = entry.get("row_count")
        if (
            not isinstance(row_count, int)
            or isinstance(row_count, bool)
            or row_count <= 0
        ):
            raise AxBenchBaseError(
                f"Seed manifest row count is invalid for {filename}. {_FRESH_RUN_TAG}"
            )
        if not is_sha256_hex(entry.get("sha256")) or entry.get("sha256") != sha256_file(
            sibling
        ):
            raise AxBenchBaseError(
                f"Seed file hash mismatch for {sibling}. {_FRESH_RUN_TAG}"
            )
    committed_count = sum(int(entry["row_count"]) for entry in files.values())
    if candidate_count != committed_count:
        raise AxBenchBaseError(
            "Seed manifest candidate count does not equal its committed split "
            f"row counts. {_FRESH_RUN_TAG}"
        )
    skip_counts = manifest.get("skip_counts")
    if not isinstance(skip_counts, dict) or any(
        not isinstance(name, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for name, count in skip_counts.items()
    ):
        raise AxBenchBaseError(f"Seed manifest skip counts are malformed. {_FRESH_RUN_TAG}")
    return manifest


def _manifest_file_entry(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    entry = manifest.get("files", {}).get(path.name)
    if not isinstance(entry, dict):
        raise AxBenchBaseError(
            f"Seed manifest does not commit {path.name}. {_FRESH_RUN_TAG}"
        )
    return entry


def _validate_seed_tensor_schema(tensors: Mapping[str, t.Tensor]) -> None:
    actual_names = frozenset(tensors)
    if actual_names != _REQUIRED_TENSORS:
        missing = sorted(_REQUIRED_TENSORS - actual_names)
        extra = sorted(actual_names - _REQUIRED_TENSORS)
        raise AxBenchBaseError(
            f"Seed tensor schema mismatch (missing={missing}, extra={extra}). "
            f"{_FRESH_RUN_TAG}"
        )
    for name, _dtype in _SEED_TENSOR_SCHEMA:
        expected_dtype = _EXPECTED_DTYPES[name]
        tensor = tensors[name]
        if tensor.ndim != 1 or tensor.dtype != expected_dtype:
            raise AxBenchBaseError(
                f"Seed tensor {name} must be 1-D {expected_dtype}, got "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}. {_FRESH_RUN_TAG}"
            )
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise AxBenchBaseError(
                f"Seed tensor {name} must be CPU-contiguous before its buffer "
                f"is consumed, got device={tensor.device}, "
                f"contiguous={tensor.is_contiguous()}. {_FRESH_RUN_TAG}"
            )


def _hash_stage1b_seed_frame(
    digest: Any,
    label: str,
    payload: Any,
) -> None:
    """Hash one v2 seed-identity frame using four streaming updates."""

    label_bytes = label.encode("utf-8")
    digest.update(struct.pack(">Q", len(label_bytes)))
    digest.update(label_bytes)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _stage1b_seed_rows_identity(
    tensors: Mapping[str, t.Tensor],
    *,
    row_count: int,
) -> str:
    """Hash the fixed schema-v2 tensor buffers without materializing rows.

    The tensor names, dtypes, rank, byte order, and frame order are the
    permanent ``stage1b_seed_rows_v2`` contract. Any change to those rules
    requires a new identity version rather than reinterpretation here.
    """

    _validate_seed_tensor_schema(tensors)
    if sys.byteorder != "little":
        raise AxBenchBaseError(
            "Stage 1b seed rows identity v2 requires a little-endian host"
        )
    if row_count < 0:
        raise AxBenchBaseError("Stage 1b seed rows identity has negative row count")
    digest = hashlib.sha256()
    _hash_stage1b_seed_frame(
        digest,
        "identity_version",
        _STAGE1B_SEED_ROWS_IDENTITY_VERSION.encode("utf-8"),
    )
    _hash_stage1b_seed_frame(digest, "row_count", struct.pack(">Q", row_count))
    for name, _dtype in _SEED_TENSOR_SCHEMA:
        payload = memoryview(tensors[name].numpy()).cast("B")
        _hash_stage1b_seed_frame(digest, name, payload)
    return digest.hexdigest()


def _load_seed_pool(
    path: Path,
    *,
    expected_split: str | None = None,
    expected_seed_fingerprint: str | None = None,
    expectation: SeedPoolExpectation | None = None,
    tokenizer: Any | None = None,
    include_rows_identity: bool,
) -> tuple[list[SeedRow], str | None]:
    """Load one committed v2 seed split; legacy/torn artifacts fail loudly."""

    if not path.is_file():
        raise AxBenchBaseError(f"Seed file does not exist: {path}")
    manifest = _load_manifest_for_file(path)
    file_entry = _manifest_file_entry(manifest, path)
    metadata = _read_metadata(path)
    if tokenizer is not None and expectation is None:
        expectation = expectation_from_seed_file(path, tokenizer)
    missing_metadata = sorted(_REQUIRED_METADATA - set(metadata))
    if missing_metadata:
        raise AxBenchBaseError(
            f"Seed metadata is missing {missing_metadata}. {_FRESH_RUN_TAG}"
        )
    if metadata.get("schema_version") != str(SEED_SCHEMA_VERSION):
        raise AxBenchBaseError("Seed file is not schema v2. " + _FRESH_RUN_TAG)
    split = metadata.get("split")
    if split != file_entry.get("split"):
        raise AxBenchBaseError("Seed file/manifest split mismatch. " + _FRESH_RUN_TAG)
    if expected_split is not None and split != expected_split:
        raise AxBenchBaseError(
            f"Seed split mismatch: expected {expected_split!r}, got {split!r}. "
            f"{_FRESH_RUN_TAG}"
        )
    if metadata.get("token_semantics") != SEED_TOKEN_SEMANTICS:
        raise AxBenchBaseError("Seed token semantics mismatch. " + _FRESH_RUN_TAG)
    if manifest.get("token_semantics") != metadata.get("token_semantics"):
        raise AxBenchBaseError("Seed manifest/file semantics mismatch. " + _FRESH_RUN_TAG)
    if metadata.get("seed_fingerprint") != manifest.get("seed_fingerprint"):
        raise AxBenchBaseError("Seed fingerprint mismatch. " + _FRESH_RUN_TAG)
    if (
        expected_seed_fingerprint is not None
        and metadata.get("seed_fingerprint") != expected_seed_fingerprint
    ):
        raise AxBenchBaseError(
            "Seed fingerprint changed during snapshot validation. "
            + _FRESH_RUN_TAG
        )
    if metadata.get("config_fingerprint") != manifest.get("config_fingerprint"):
        raise AxBenchBaseError("Seed manifest/file config mismatch. " + _FRESH_RUN_TAG)
    if metadata.get("nanochat_tokenizer_checksum") != manifest.get(
        "nanochat_tokenizer_checksum"
    ):
        raise AxBenchBaseError(
            "Seed manifest/file nanochat tokenizer mismatch. " + _FRESH_RUN_TAG
        )
    if metadata.get("extraction_tokenizer_checksum") != manifest.get(
        "extraction_tokenizer_checksum"
    ):
        raise AxBenchBaseError(
            "Seed manifest/file extraction tokenizer mismatch. " + _FRESH_RUN_TAG
        )
    parsed_json: dict[str, Any] = {}
    for json_key in (
        "seed_config_json",
        "source_parquet_hashes_json",
        "parameter_hashes_json",
        "runtime_identity_json",
        "extraction_settings_json",
    ):
        try:
            parsed = json.loads(metadata[json_key])
        except (TypeError, json.JSONDecodeError) as exc:
            raise AxBenchBaseError(
                f"Seed metadata field {json_key} is not canonical JSON. "
                f"{_FRESH_RUN_TAG}"
            ) from exc
        if canonical_json(parsed) != metadata[json_key]:
            raise AxBenchBaseError(
                f"Seed metadata field {json_key} is not canonical JSON. "
                f"{_FRESH_RUN_TAG}"
            )
        parsed_json[json_key] = parsed
    if sha256_bytes(metadata["seed_config_json"].encode("utf-8")) != metadata.get(
        "config_fingerprint"
    ):
        raise AxBenchBaseError("Seed config payload/fingerprint mismatch. " + _FRESH_RUN_TAG)
    manifest_payload_pairs = (
        ("source_parquet_hashes_json", "source_parquet_hashes"),
        ("parameter_hashes_json", "parameter_hashes"),
        ("runtime_identity_json", "runtime_identity"),
        ("extraction_settings_json", "extraction_settings"),
    )
    for metadata_key, manifest_key in manifest_payload_pairs:
        if parsed_json[metadata_key] != manifest.get(manifest_key):
            raise AxBenchBaseError(
                f"Seed manifest/file {manifest_key} mismatch. {_FRESH_RUN_TAG}"
            )
    if expectation is not None:
        if tokenizer is None:
            raise AxBenchBaseError("tokenizer is required with a seed expectation")
        if metadata.get("nanochat_tokenizer_checksum") != (
            expectation.nanochat_tokenizer_checksum
        ):
            raise AxBenchBaseError(
                "Seed nanochat tokenizer checksum mismatch. " + _FRESH_RUN_TAG
            )
        if metadata.get("config_fingerprint") != expectation.config_fingerprint:
            raise AxBenchBaseError("Seed config fingerprint mismatch. " + _FRESH_RUN_TAG)

    try:
        tensors = load_file(str(path))
    except Exception as exc:
        raise AxBenchBaseError(f"Cannot load seed tensors: {exc}. {_FRESH_RUN_TAG}") from exc
    _validate_seed_tensor_schema(tensors)
    n_rows = int(tensors["shard_idx"].numel())
    scalar_names = (
        "source_nanochat_token_count",
        "prefix_source_token_start",
        "prefix_source_token_end",
        "doc_idx",
    )
    if any(int(tensors[name].numel()) != n_rows for name in scalar_names):
        raise AxBenchBaseError("Seed row tensor lengths disagree. " + _FRESH_RUN_TAG)
    rows_sha256 = (
        _stage1b_seed_rows_identity(tensors, row_count=n_rows)
        if include_rows_identity
        else None
    )

    prefix_text_offsets = tensors["prefix_text_offsets"].tolist()
    prefix_text_bytes = bytes(tensors["prefix_text_bytes"].tolist())
    continuation_text_offsets = tensors["continuation_text_offsets"].tolist()
    continuation_text_bytes = bytes(tensors["continuation_text_bytes"].tolist())
    prefix_offsets = tensors["prefix_token_offsets"].tolist()
    prefix_tokens = tensors["prefix_content_tokens"].tolist()
    continuation_offsets = tensors["continuation_token_offsets"].tolist()
    continuation_tokens = tensors["continuation_content_tokens"].tolist()
    feature_offsets = tensors["sae_feature_offsets"].tolist()
    feature_ids = tensors["sae_feature_ids"].tolist()
    feature_mags = tensors["sae_feature_mags"].tolist()
    _validate_offsets(
        "prefix_text", prefix_text_offsets, n_rows=n_rows, payload_length=len(prefix_text_bytes)
    )
    _validate_offsets(
        "continuation_text",
        continuation_text_offsets,
        n_rows=n_rows,
        payload_length=len(continuation_text_bytes),
    )
    _validate_offsets(
        "prefix_token", prefix_offsets, n_rows=n_rows, payload_length=len(prefix_tokens)
    )
    _validate_offsets(
        "continuation_token",
        continuation_offsets,
        n_rows=n_rows,
        payload_length=len(continuation_tokens),
    )
    _validate_offsets(
        "sae_feature", feature_offsets, n_rows=n_rows, payload_length=len(feature_ids)
    )
    if len(feature_ids) != len(feature_mags):
        raise AxBenchBaseError("Seed feature tensor lengths disagree. " + _FRESH_RUN_TAG)

    rows: list[SeedRow] = []
    try:
        for index in range(n_rows):
            pt0, pt1 = prefix_text_offsets[index : index + 2]
            ct0, ct1 = continuation_text_offsets[index : index + 2]
            p0, p1 = prefix_offsets[index : index + 2]
            c0, c1 = continuation_offsets[index : index + 2]
            f0, f1 = feature_offsets[index : index + 2]
            row = SeedRow(
                shard_idx=int(tensors["shard_idx"][index].item()),
                doc_idx=int(tensors["doc_idx"][index].item()),
                source_nanochat_token_count=int(
                    tensors["source_nanochat_token_count"][index].item()
                ),
                prefix_source_token_start=int(
                    tensors["prefix_source_token_start"][index].item()
                ),
                prefix_source_token_end=int(
                    tensors["prefix_source_token_end"][index].item()
                ),
                prefix_text=prefix_text_bytes[pt0:pt1].decode("utf-8"),
                prefix_content_nanochat_tokens=[int(value) for value in prefix_tokens[p0:p1]],
                sae_feature_ids=[int(value) for value in feature_ids[f0:f1]],
                sae_feature_mags=[float(value) for value in feature_mags[f0:f1]],
                continuation_text=continuation_text_bytes[ct0:ct1].decode("utf-8"),
                continuation_content_nanochat_tokens=[
                    int(value) for value in continuation_tokens[c0:c1]
                ],
            )
            if expectation is not None:
                assert tokenizer is not None
                validate_seed_row(row, expectation=expectation, tokenizer=tokenizer)
            rows.append(row)
    except (UnicodeDecodeError, IndexError, TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Malformed seed row payload in {path}: {exc}. {_FRESH_RUN_TAG}"
        ) from exc
    if int(file_entry.get("row_count", -1)) != len(rows):
        raise AxBenchBaseError("Seed manifest row count mismatch. " + _FRESH_RUN_TAG)
    return rows, rows_sha256


def load_seed_pool(
    path: Path,
    *,
    expected_split: str | None = None,
    expected_seed_fingerprint: str | None = None,
    expectation: SeedPoolExpectation | None = None,
    tokenizer: Any | None = None,
) -> list[SeedRow]:
    """Load one committed v2 seed split; legacy/torn artifacts fail loudly."""

    rows, _rows_sha256 = _load_seed_pool(
        path,
        expected_split=expected_split,
        expected_seed_fingerprint=expected_seed_fingerprint,
        expectation=expectation,
        tokenizer=tokenizer,
        include_rows_identity=False,
    )
    return rows


def load_seed_pool_with_rows_identity(
    path: Path,
    *,
    expected_split: str | None = None,
    expected_seed_fingerprint: str | None = None,
    expectation: SeedPoolExpectation | None = None,
    tokenizer: Any | None = None,
) -> LoadedSeedSplit:
    """Load rows and hash their validated contiguous tensor representation."""

    rows, rows_sha256 = _load_seed_pool(
        path,
        expected_split=expected_split,
        expected_seed_fingerprint=expected_seed_fingerprint,
        expectation=expectation,
        tokenizer=tokenizer,
        include_rows_identity=True,
    )
    assert rows_sha256 is not None
    return LoadedSeedSplit(rows=rows, rows_sha256=rows_sha256)


def load_seed_pool_snapshot(
    output_paths: paths.AxBenchBasePaths,
) -> SeedPoolSnapshot:
    """Return the fingerprint and eval rows from one validation pass."""

    eval_path = output_paths.seeds_eval()
    if not eval_path.is_file():
        raise AxBenchBaseError(
            f"Committed eval seed pool is missing at {eval_path}. {_FRESH_RUN_TAG}"
        )
    resolved_tokenizer = get_tokenizer()
    manifest = _load_manifest_for_file(eval_path)
    fingerprint = manifest.get("seed_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise AxBenchBaseError("Seed manifest fingerprint is missing. " + _FRESH_RUN_TAG)
    eval_rows: tuple[SeedRow, ...] | None = None
    for split, split_path in (
        ("train", output_paths.seeds_train()),
        ("eval", eval_path),
    ):
        rows = tuple(
            load_seed_pool(
                split_path,
                expected_split=split,
                expected_seed_fingerprint=fingerprint,
                expectation=expectation_from_seed_file(
                    split_path,
                    resolved_tokenizer,
                ),
                tokenizer=resolved_tokenizer,
            )
        )
        if split == "eval":
            eval_rows = rows
    assert eval_rows is not None
    return SeedPoolSnapshot(
        fingerprint=fingerprint,
        eval_rows=eval_rows,
    )


def load_seed_pool_fingerprint(
    output_paths: paths.AxBenchBasePaths,
) -> str:
    """Return the committed v2 fingerprint after validating both split files."""

    return load_seed_pool_snapshot(output_paths).fingerprint


__all__ = [
    "SEED_SCHEMA_VERSION",
    "SEED_TOKEN_SEMANTICS",
    "LoadedSeedSplit",
    "SeedPoolExpectation",
    "SeedPoolSnapshot",
    "canonical_json",
    "compute_extraction_fingerprint",
    "compute_seed_fingerprint",
    "expectation_from_config",
    "expectation_from_seed_file",
    "load_seed_pool",
    "load_seed_pool_with_rows_identity",
    "load_seed_pool_fingerprint",
    "load_seed_pool_snapshot",
    "save_seed_pool",
    "seed_config_fingerprint",
    "seed_config_payload",
    "seed_rows_to_tensors",
    "sha256_bytes",
    "sha256_file",
    "validate_seed_row",
]
