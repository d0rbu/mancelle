"""Certified sparse raw-feature source for remote clustered inline evaluation.

The ordinary schema-v2 token overlay remains bound to the complete raw feature
family.  Remote packed training intentionally stages only the validation raw
shard needed to build its immutable clustered inline-eval cache, so it cannot
authenticate that sparse subset by re-running global overlay discovery.

This module owns the narrow alternative contract: a producer validates the
complete family and publishes a certificate for one exact raw/leaf/parquet
triple; the remote consumer fetches that authoritative certificate, verifies
the three local bodies byte-for-byte, and constructs a one-shard FeatureIndex
without changing normal FeatureIndex semantics.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from nanochat.dataset import list_parquet_files
from steering._common.strict_json import canonical_json
from steering.conditional.feature_index import ArtifactSignature, FeatureIndex
from steering.conditional.splice_counterfactual import derived_feature_fingerprint
from steering.feature_io import (
    COMPRESSED_EXT,
    MetadataKey,
    feature_path,
    glob_features,
    read_shard_metadata,
)
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    read_manifest,
)
from steering.refined_types import LowercaseSha256, NonNegativeInt, PositiveInt
from steering.remote_training import (
    BucketBackend,
    RemoteTrainingConfig,
    RemoteTrainingError,
    _download_metadata_with_retry,
    _dual_digest,
    _join_remote,
    _relative_local_dir,
)
from steering.stats_io import (
    find_statistics_path,
    load_statistics,
    load_tokenizer_checksum,
)

REMOTE_INLINE_EVAL_SOURCE_FILENAME = "remote_inline_eval_source.json"
REMOTE_INLINE_EVAL_SOURCE_SCHEMA_VERSION = 1
REMOTE_INLINE_EVAL_SOURCE_SEMANTICS = "remote_clustered_inline_eval_sparse_source_v1"

_CERTIFICATE_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "stem",
        "derived_feature_fingerprint",
        "tokenizer_checksum",
        "token_overlay",
        "raw_feature",
        "token_boundary",
        "parquet",
        "aggregate_sha256",
    }
)
_TOKEN_OVERLAY_KEYS = frozenset({"schema_version", "kind", "checksum", "base_shard_content_hash"})
_RAW_ARTIFACT_KEYS = frozenset(
    {
        "name",
        "byte_length",
        "content_sha256",
        "file_fingerprint",
        "n_documents",
        "n_segments",
        "segment_size",
    }
)
_ARTIFACT_KEYS = frozenset({"name", "byte_length", "content_sha256"})
_PARQUET_KEYS = _ARTIFACT_KEYS | {"n_documents"}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_sha256(value: Any, *, surface: str) -> LowercaseSha256:
    if not isinstance(value, LowercaseSha256):
        raise RemoteTrainingError(
            f"{surface} must be 64 lowercase hexadecimal characters, got {value!r}"
        )
    return LowercaseSha256.parse(value)


def _strict_nonnegative_int(value: Any, *, surface: str) -> NonNegativeInt:
    if not isinstance(value, NonNegativeInt):
        raise RemoteTrainingError(f"{surface} must be a non-negative strict int, got {value!r}")
    return NonNegativeInt.parse(value)


def _strict_positive_int(value: Any, *, surface: str) -> PositiveInt:
    if not isinstance(value, PositiveInt):
        raise RemoteTrainingError(f"{surface} must be a positive strict int, got {value!r}")
    return PositiveInt.parse(value)


def _strict_direct_name(value: Any, *, suffix: str, surface: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != Path(value).name
        or not value.endswith(suffix)
    ):
        raise RemoteTrainingError(
            f"{surface} must be a direct filename ending {suffix!r}, got {value!r}"
        )
    return value


def _parse_artifact(
    raw: Any,
    *,
    expected_keys: frozenset[str],
    expected_name: str,
    surface: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise RemoteTrainingError(
            f"{surface} must contain exactly {sorted(expected_keys)}; "
            f"got keys={sorted(keys, key=str)}"
        )
    name = _strict_direct_name(
        raw["name"],
        suffix=Path(expected_name).suffix,
        surface=f"{surface}.name",
    )
    if name != expected_name:
        raise RemoteTrainingError(f"{surface}.name must be {expected_name!r}, got {name!r}")
    parsed = {
        "name": name,
        "byte_length": _strict_nonnegative_int(
            raw["byte_length"], surface=f"{surface}.byte_length"
        ),
        "content_sha256": _strict_sha256(
            raw["content_sha256"], surface=f"{surface}.content_sha256"
        ),
    }
    if "file_fingerprint" in expected_keys:
        parsed["file_fingerprint"] = _strict_sha256(
            raw["file_fingerprint"], surface=f"{surface}.file_fingerprint"
        )
    if "n_segments" in expected_keys:
        parsed["n_segments"] = _strict_nonnegative_int(
            raw["n_segments"], surface=f"{surface}.n_segments"
        )
    if "segment_size" in expected_keys:
        parsed["segment_size"] = _strict_positive_int(
            raw["segment_size"], surface=f"{surface}.segment_size"
        )
    if "n_documents" in expected_keys:
        parsed["n_documents"] = _strict_nonnegative_int(
            raw["n_documents"], surface=f"{surface}.n_documents"
        )
    return parsed


def build_remote_inline_eval_certificate(
    *,
    stem: str,
    derived_feature_fingerprint_value: str,
    tokenizer_checksum: str,
    token_overlay_base_shard_content_hash: str,
    raw_feature: Mapping[str, Any],
    token_boundary: Mapping[str, Any],
    parquet: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and self-parse the canonical schema-v1 certificate."""
    if not isinstance(stem, str) or not stem or stem != Path(stem).name:
        raise RemoteTrainingError(
            f"remote inline-eval stem must be a non-empty direct name, got {stem!r}"
        )
    signed = {
        "schema_version": REMOTE_INLINE_EVAL_SOURCE_SCHEMA_VERSION,
        "semantics": REMOTE_INLINE_EVAL_SOURCE_SEMANTICS,
        "stem": stem,
        "derived_feature_fingerprint": derived_feature_fingerprint_value,
        "tokenizer_checksum": tokenizer_checksum,
        "token_overlay": {
            "schema_version": 2,
            "kind": OverlayKind.SEG_BOUNDARY.value,
            "checksum": tokenizer_checksum,
            "base_shard_content_hash": token_overlay_base_shard_content_hash,
        },
        "raw_feature": dict(raw_feature),
        "token_boundary": dict(token_boundary),
        "parquet": dict(parquet),
    }
    certificate = {**signed, "aggregate_sha256": _sha256_json(signed)}
    return parse_remote_inline_eval_certificate(
        certificate,
        surface="built remote inline-eval certificate",
    )


def parse_remote_inline_eval_certificate(
    raw: Any,
    *,
    surface: str,
) -> dict[str, Any]:
    """Strictly parse and authenticate one sparse inline-eval certificate."""
    if not isinstance(raw, Mapping) or set(raw) != _CERTIFICATE_KEYS:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise RemoteTrainingError(
            f"{surface} must contain exactly {sorted(_CERTIFICATE_KEYS)}; "
            f"got keys={sorted(keys, key=str)}"
        )
    schema_version = raw["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != REMOTE_INLINE_EVAL_SOURCE_SCHEMA_VERSION
    ):
        raise RemoteTrainingError(
            f"{surface}.schema_version must be "
            f"{REMOTE_INLINE_EVAL_SOURCE_SCHEMA_VERSION}, got {schema_version!r}"
        )
    if raw["semantics"] != REMOTE_INLINE_EVAL_SOURCE_SEMANTICS:
        raise RemoteTrainingError(
            f"{surface}.semantics must be "
            f"{REMOTE_INLINE_EVAL_SOURCE_SEMANTICS!r}, got {raw['semantics']!r}"
        )
    stem = raw["stem"]
    if not isinstance(stem, str) or not stem or stem != Path(stem).name:
        raise RemoteTrainingError(f"{surface}.stem must be a direct non-empty name")
    derived_fingerprint = _strict_sha256(
        raw["derived_feature_fingerprint"],
        surface=f"{surface}.derived_feature_fingerprint",
    )
    tokenizer_checksum = _strict_sha256(
        raw["tokenizer_checksum"],
        surface=f"{surface}.tokenizer_checksum",
    )
    token_overlay_raw = raw["token_overlay"]
    if not isinstance(token_overlay_raw, Mapping) or set(token_overlay_raw) != (
        _TOKEN_OVERLAY_KEYS
    ):
        keys = set(token_overlay_raw) if isinstance(token_overlay_raw, Mapping) else set()
        raise RemoteTrainingError(
            f"{surface}.token_overlay must contain exactly "
            f"{sorted(_TOKEN_OVERLAY_KEYS)}; got keys={sorted(keys, key=str)}"
        )
    if type(token_overlay_raw["schema_version"]) is not int or (
        token_overlay_raw["schema_version"] != 2
    ):
        raise RemoteTrainingError(f"{surface}.token_overlay.schema_version must be 2")
    if token_overlay_raw["kind"] != OverlayKind.SEG_BOUNDARY.value:
        raise RemoteTrainingError(
            f"{surface}.token_overlay.kind must be {OverlayKind.SEG_BOUNDARY.value!r}"
        )
    overlay_checksum = _strict_sha256(
        token_overlay_raw["checksum"],
        surface=f"{surface}.token_overlay.checksum",
    )
    if overlay_checksum != tokenizer_checksum:
        raise RemoteTrainingError(
            f"{surface} tokenizer checksum disagrees with token-overlay checksum"
        )
    token_overlay = {
        "schema_version": 2,
        "kind": OverlayKind.SEG_BOUNDARY.value,
        "checksum": overlay_checksum,
        "base_shard_content_hash": _strict_sha256(
            token_overlay_raw["base_shard_content_hash"],
            surface=f"{surface}.token_overlay.base_shard_content_hash",
        ),
    }
    raw_feature = _parse_artifact(
        raw["raw_feature"],
        expected_keys=_RAW_ARTIFACT_KEYS,
        expected_name=f"{stem}_features{COMPRESSED_EXT}",
        surface=f"{surface}.raw_feature",
    )
    token_boundary = _parse_artifact(
        raw["token_boundary"],
        expected_keys=_ARTIFACT_KEYS,
        expected_name=f"{stem}{COMPRESSED_EXT}",
        surface=f"{surface}.token_boundary",
    )
    parquet = _parse_artifact(
        raw["parquet"],
        expected_keys=_PARQUET_KEYS,
        expected_name=f"{stem}.parquet",
        surface=f"{surface}.parquet",
    )
    signed = {
        "schema_version": schema_version,
        "semantics": REMOTE_INLINE_EVAL_SOURCE_SEMANTICS,
        "stem": stem,
        "derived_feature_fingerprint": derived_fingerprint,
        "tokenizer_checksum": tokenizer_checksum,
        "token_overlay": token_overlay,
        "raw_feature": raw_feature,
        "token_boundary": token_boundary,
        "parquet": parquet,
    }
    aggregate = _strict_sha256(
        raw["aggregate_sha256"],
        surface=f"{surface}.aggregate_sha256",
    )
    expected_aggregate = _sha256_json(signed)
    if aggregate != expected_aggregate:
        raise RemoteTrainingError(
            f"{surface}.aggregate_sha256 mismatch: certificate has {aggregate}, "
            f"fields hash to {expected_aggregate}"
        )
    return {**signed, "aggregate_sha256": expected_aggregate}


def _direct_regular_file(path: Path, *, surface: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise RemoteTrainingError(f"{surface} is missing or unreadable: {path}") from exc
    if not stat.S_ISREG(mode):
        raise RemoteTrainingError(f"{surface} must be a direct regular file: {path}")


def _matching_parquet(data_dir: str | None, stem: str) -> Path:
    matches = [
        Path(path) for path in list_parquet_files(data_dir=data_dir) if Path(path).stem == stem
    ]
    if len(matches) != 1:
        raise RemoteTrainingError(
            "remote clustered inline evaluation requires exactly one parquet "
            f"named {stem!r}; found {[str(path) for path in matches]}"
        )
    return matches[0]


def _artifact_record(path: Path) -> dict[str, Any]:
    byte_length, content_sha256, _ = _dual_digest(path)
    return {
        "name": path.name,
        "byte_length": byte_length,
        "content_sha256": content_sha256,
    }


def _materialize_authoritative_certificate(
    *,
    backend: BucketBackend,
    remote: RemoteTrainingConfig,
    remote_path: str,
    feature_root: Path,
    local_path: Path,
) -> None:
    """Serialize certificate publication without mutating the feature dir per rank."""
    lock_path = feature_root / ".remote_inline_eval_source.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        downloaded = feature_root / (
            f".{REMOTE_INLINE_EVAL_SOURCE_FILENAME}.remote-discovery."
            f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            _download_metadata_with_retry(
                backend,
                remote_path=remote_path,
                final_path=downloaded,
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
            )
            if local_path.exists():
                _direct_regular_file(
                    local_path,
                    surface="local remote inline-eval certificate",
                )
                if local_path.read_bytes() != downloaded.read_bytes():
                    raise RemoteTrainingError(
                        "local remote inline-eval certificate conflicts with "
                        f"the authoritative remote object: {local_path}"
                    )
            else:
                os.replace(downloaded, local_path)
        finally:
            try:
                downloaded.unlink()
            except OSError:
                pass
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def publish_remote_inline_eval_source(
    *,
    feature_dir: Path,
    data_dir: str | None,
    stem: str,
    fingerprint_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate the complete producer family and atomically write its certificate."""
    feature_dir = Path(feature_dir)
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise FileNotFoundError(
            f"feature_statistics not found in remote inline-eval feature dir {feature_dir}"
        )
    tokenizer_checksum = load_tokenizer_checksum(stats_path)
    if tokenizer_checksum is None:
        raise RemoteTrainingError(
            "remote inline-eval certificate publication requires statistics "
            "with a tokenizer checksum"
        )
    statistics = load_statistics(stats_path)
    raw_path = feature_path(feature_dir, stem)
    parquet_path = _matching_parquet(data_dir, stem)
    all_raw_paths = glob_features(feature_dir)
    if raw_path not in all_raw_paths:
        raise FileNotFoundError(
            f"remote inline-eval raw source is absent from the complete family: {raw_path}"
        )
    cache_dir = fingerprint_cache_dir or feature_dir
    family_fingerprint_before = derived_feature_fingerprint(
        feature_dir,
        cache_dir=cache_dir,
    )
    full_index = FeatureIndex(
        str(feature_dir),
        tokenizer_checksum=tokenizer_checksum,
        required_token_boundary_stems={stem},
    )
    full_index.ensure_resident(stem)
    token_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tokenizer_checksum)
    token_manifest = read_manifest(token_dir)
    if not isinstance(token_manifest, dict):
        raise RemoteTrainingError(
            f"remote inline-eval token manifest is missing or malformed: {token_dir}"
        )
    if (
        token_manifest.get("schema_version") != 2
        or token_manifest.get("kind") != OverlayKind.SEG_BOUNDARY.value
        or token_manifest.get("checksum") != tokenizer_checksum
    ):
        raise RemoteTrainingError(
            "remote inline-eval publication requires the current schema-v2 "
            "token-boundary manifest with the statistics tokenizer checksum"
        )
    base_hash = _strict_sha256(
        token_manifest.get("base_shard_content_hash"),
        surface="token manifest base_shard_content_hash",
    )
    metadata = read_shard_metadata(raw_path)
    try:
        n_documents = int(metadata[MetadataKey.N_DOCS])
        n_segments = int(metadata[MetadataKey.N_SEGMENTS])
        segment_size = int(metadata[MetadataKey.SEGMENT_SIZE])
    except (KeyError, ValueError) as exc:
        raise RemoteTrainingError(
            f"remote inline-eval raw source has malformed metadata: {raw_path}"
        ) from exc
    if segment_size != statistics.metadata.segment_size:
        raise RemoteTrainingError(
            "remote inline-eval raw/statistics segment_size mismatch: "
            f"{segment_size} != {statistics.metadata.segment_size}"
        )
    raw_bytes, raw_sha256, raw_fingerprint = _dual_digest(raw_path)
    token_path = overlay_shard_path(token_dir, stem)
    token_record = _artifact_record(token_path)
    parquet_record = _artifact_record(parquet_path)
    parquet_metadata = pq.read_metadata(parquet_path)
    if "text" not in parquet_metadata.schema.names:
        raise RemoteTrainingError(
            f"remote inline-eval parquet lacks required text column: {parquet_path}"
        )
    if parquet_metadata.num_rows != n_documents:
        raise RemoteTrainingError(
            "remote inline-eval raw/parquet document-count mismatch: "
            f"{n_documents} != {parquet_metadata.num_rows}"
        )
    family_fingerprint_after = derived_feature_fingerprint(
        feature_dir,
        cache_dir=cache_dir,
    )
    if family_fingerprint_after != family_fingerprint_before:
        raise RemoteTrainingError(
            "derived feature family changed while publishing remote inline-eval source"
        )
    final_manifest = read_manifest(token_dir)
    if final_manifest != token_manifest:
        raise RemoteTrainingError(
            "token-boundary manifest changed while publishing remote inline-eval source"
        )
    certificate = build_remote_inline_eval_certificate(
        stem=stem,
        derived_feature_fingerprint_value=family_fingerprint_before,
        tokenizer_checksum=tokenizer_checksum,
        token_overlay_base_shard_content_hash=base_hash,
        raw_feature={
            "name": raw_path.name,
            "byte_length": raw_bytes,
            "content_sha256": raw_sha256,
            "file_fingerprint": raw_fingerprint,
            "n_documents": n_documents,
            "n_segments": n_segments,
            "segment_size": segment_size,
        },
        token_boundary=token_record,
        parquet={
            **parquet_record,
            "n_documents": parquet_metadata.num_rows,
        },
    )
    output_path = feature_dir / REMOTE_INLINE_EVAL_SOURCE_FILENAME
    atomic_json_dump(certificate, output_path)
    reparsed = parse_remote_inline_eval_certificate(
        json.loads(output_path.read_text(encoding="utf-8")),
        surface=str(output_path),
    )
    if reparsed != certificate:
        raise RemoteTrainingError(
            "remote inline-eval certificate changed during atomic serialization"
        )
    return certificate


@dataclass(frozen=True)
class CertifiedRemoteInlineEvalSource:
    """Authenticated local source and its one-shard lookup index."""

    certificate: dict[str, Any]
    feature_index: FeatureIndex
    parquet_path: Path
    signatures: tuple[ArtifactSignature, ...]

    @property
    def stem(self) -> str:
        return cast(str, self.certificate["stem"])

    @property
    def aggregate_sha256(self) -> str:
        return cast(str, self.certificate["aggregate_sha256"])

    def assert_derived_feature_fingerprint(self, expected: str) -> None:
        actual = self.certificate["derived_feature_fingerprint"]
        if actual != expected:
            raise RemoteTrainingError(
                "remote inline-eval certificate derived-feature fingerprint "
                f"mismatch: certificate={actual!r}, verified selections={expected!r}"
            )

    def validate_signatures(self) -> None:
        """Reject source replacement between authentication and cache completion."""
        for signature in self.signatures:
            signature.validate()

    @classmethod
    def discover(
        cls,
        *,
        backend: BucketBackend,
        remote: RemoteTrainingConfig,
        feature_root: Path,
        feature_dir: Path,
        data_dir: str | None,
        tokenizer_checksum: str,
        segment_size: int,
    ) -> CertifiedRemoteInlineEvalSource:
        """Fetch the authoritative certificate and authenticate sparse local bodies."""
        feature_dir = Path(feature_dir)
        relative = _relative_local_dir(
            feature_dir,
            feature_root,
            surface="remote inline-eval feature directory",
        )
        remote_path = _join_remote(
            remote.selection_prefix,
            relative.as_posix(),
            REMOTE_INLINE_EVAL_SOURCE_FILENAME,
        )
        local_path = feature_dir / REMOTE_INLINE_EVAL_SOURCE_FILENAME
        _materialize_authoritative_certificate(
            backend=backend,
            remote=remote,
            remote_path=remote_path,
            feature_root=Path(feature_root),
            local_path=local_path,
        )

        def validate_local() -> CertifiedRemoteInlineEvalSource:
            try:
                raw_certificate = json.loads(local_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RemoteTrainingError(
                    f"remote inline-eval certificate is not valid UTF-8 JSON: {local_path}"
                ) from exc
            certificate = parse_remote_inline_eval_certificate(
                raw_certificate,
                surface=str(local_path),
            )
            if certificate["tokenizer_checksum"] != tokenizer_checksum:
                raise RemoteTrainingError(
                    "remote inline-eval certificate tokenizer checksum mismatch: "
                    f"{certificate['tokenizer_checksum']!r} != {tokenizer_checksum!r}"
                )
            raw_record = cast(dict[str, Any], certificate["raw_feature"])
            if raw_record["segment_size"] != segment_size:
                raise RemoteTrainingError(
                    "remote inline-eval certificate segment_size mismatch: "
                    f"{raw_record['segment_size']} != {segment_size}"
                )
            stem = cast(str, certificate["stem"])
            raw_path = feature_dir / raw_record["name"]
            local_raw_names = {path.name for path in glob_features(feature_dir)}
            if local_raw_names != {raw_record["name"]}:
                raise RemoteTrainingError(
                    "remote inline-eval sparse raw namespace must contain exactly "
                    f"the certified shard; expected={[raw_record['name']]}, "
                    f"found={sorted(local_raw_names)}"
                )
            token_dir = overlay_dir(
                feature_dir,
                OverlayKind.SEG_BOUNDARY,
                tokenizer_checksum,
            )
            token_record = cast(dict[str, Any], certificate["token_boundary"])
            token_path = token_dir / token_record["name"]
            local_token_names = {path.name for path in token_dir.glob(f"*{COMPRESSED_EXT}")}
            if local_token_names != {token_record["name"]}:
                raise RemoteTrainingError(
                    "remote inline-eval sparse token-boundary namespace must "
                    "contain exactly the certified leaf; "
                    f"expected={[token_record['name']]}, "
                    f"found={sorted(local_token_names)}"
                )
            token_manifest = read_manifest(token_dir)
            token_contract = cast(dict[str, Any], certificate["token_overlay"])
            if not isinstance(token_manifest, dict) or any(
                token_manifest.get(key) != token_contract[key]
                for key in (
                    "schema_version",
                    "kind",
                    "checksum",
                    "base_shard_content_hash",
                )
            ):
                raise RemoteTrainingError(
                    "local token-boundary manifest disagrees with the certified "
                    "schema/checksum/full-family base hash"
                )
            parquet_record = cast(dict[str, Any], certificate["parquet"])
            parquet_path = _matching_parquet(data_dir, stem)
            if parquet_path.name != parquet_record["name"]:
                raise RemoteTrainingError(
                    "remote inline-eval parquet name differs from certificate: "
                    f"{parquet_path.name!r} != {parquet_record['name']!r}"
                )
            for label, path, record in (
                ("raw feature", raw_path, raw_record),
                ("token boundary", token_path, token_record),
                ("parquet", parquet_path, parquet_record),
            ):
                _direct_regular_file(path, surface=f"remote inline-eval {label}")
                byte_length, content_sha256, file_fingerprint = _dual_digest(path)
                if (
                    byte_length != record["byte_length"]
                    or content_sha256 != record["content_sha256"]
                ):
                    raise RemoteTrainingError(
                        f"remote inline-eval {label} integrity mismatch for {path}: "
                        f"expected size/sha256 {record['byte_length']}/"
                        f"{record['content_sha256']}, got "
                        f"{byte_length}/{content_sha256}"
                    )
                if label == "raw feature" and (file_fingerprint != raw_record["file_fingerprint"]):
                    raise RemoteTrainingError(
                        "remote inline-eval raw feature name-salted fingerprint "
                        f"mismatch for {path}"
                    )
            metadata = read_shard_metadata(raw_path)
            try:
                actual_n_documents = int(metadata[MetadataKey.N_DOCS])
                actual_n_segments = int(metadata[MetadataKey.N_SEGMENTS])
                actual_segment_size = int(metadata[MetadataKey.SEGMENT_SIZE])
            except (KeyError, ValueError) as exc:
                raise RemoteTrainingError(
                    f"remote inline-eval raw source has malformed metadata: {raw_path}"
                ) from exc
            if (
                actual_n_documents != raw_record["n_documents"]
                or actual_n_documents != parquet_record["n_documents"]
                or actual_n_segments != raw_record["n_segments"]
                or actual_segment_size != raw_record["segment_size"]
            ):
                raise RemoteTrainingError(
                    "remote inline-eval raw metadata disagrees with certificate: "
                    f"documents/segments/size={actual_n_documents}/"
                    f"{actual_n_segments}/{actual_segment_size}, "
                    f"certificate={raw_record['n_documents']}/"
                    f"{raw_record['n_segments']}/"
                    f"{raw_record['segment_size']}"
                )
            parquet_metadata = pq.read_metadata(parquet_path)
            if (
                parquet_metadata.num_rows != parquet_record["n_documents"]
                or "text" not in parquet_metadata.schema.names
            ):
                raise RemoteTrainingError(
                    "remote inline-eval parquet schema/count disagrees with certificate"
                )
            feature_index = FeatureIndex.from_certified_sparse_source(
                feature_dir=feature_dir,
                shard_path=raw_path,
                token_overlay_dir=token_dir,
                tokenizer_checksum=tokenizer_checksum,
                segment_size=segment_size,
                validated_content_fingerprint=certificate["aggregate_sha256"],
            )
            signature_paths = sorted(
                {
                    feature_dir.absolute(),
                    local_path.absolute(),
                    raw_path.absolute(),
                    token_dir.absolute(),
                    (token_dir / "manifest.json").absolute(),
                    token_path.absolute(),
                    parquet_path.parent.absolute(),
                    parquet_path.absolute(),
                },
                key=str,
            )
            signatures = tuple(ArtifactSignature.capture(path) for path in signature_paths)
            feature_index.ensure_resident(stem)
            for signature in signatures:
                signature.validate()
            logger.info(
                "Verified certified sparse remote inline-eval source: "
                f"stem={stem} aggregate={certificate['aggregate_sha256']}"
            )
            return cls(
                certificate=certificate,
                feature_index=feature_index,
                parquet_path=parquet_path,
                signatures=signatures,
            )

        return validate_local()
