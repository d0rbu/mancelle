"""Immutable inventories and bounded transport for remote splice pre-selection."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import torch as t

from nanochat.atomic_io import atomic_write_bytes
from steering._common.strict_json import (
    canonical_json,
    strict_json_object,
    string_keyed_mapping,
)
from steering.conditional.feature_clusters import (
    parse_clustered_selection_shard_inventory,
)
from steering.conditional.feature_index import (
    MaterializedFeatureShard,
    TransportFeatureIndexCatalog,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.splice_counterfactual import (
    PackedProvenanceInfo,
    _file_fingerprint,
    _fingerprint_json,
    _fold_file_digests,
    splice_sidecar_fingerprint,
)
from steering.conditional.splice_resource_admission import (
    validate_source_layout_pair,
)
from steering.feature_io import (
    COMPRESSED_EXT,
    RegularFileIdentity,
)
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    load_overlay_shard,
    read_manifest,
)
from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
    resolve_remote_feature_prefix,
)
from steering.remote_training import (
    REMOTE_PACKED_INVENTORY_FILENAME,
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    RemoteTrainingConfig,
    download_verified_object,
    parse_remote_packed_inventory,
    upload_verified_object,
    verified_local_object_info,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.safetensors_layout import (
    SafetensorsLayoutError,
    parse_semantic_layout,
    read_compressed_safetensors_layout,
    semantic_layout_to_json,
)
from steering.stats_io import find_statistics_path, load_statistics

REMOTE_PRESELECTION_SOURCE_SCHEMA_VERSION = 1
REMOTE_PRESELECTION_TRUSTED_SOURCE_SEMANTICS = (
    "clustered-splice-counterfactual-remote-preselection-trusted-v1"
)
REMOTE_PRESELECTION_SEMANTICS = "clustered-splice-counterfactual-remote-preselection-v1"
REMOTE_PRESELECTION_RECEIPT_SCHEMA_VERSION = 1
REMOTE_UNSPLICED_PRESELECTION_RECEIPT_SCHEMA_VERSION = 2
REMOTE_PRESELECTION_RECEIPTS_DIRNAME = "_remote_preselection_receipts"
REMOTE_UNSPLICED_PRESELECTION_SEMANTICS = "clustered-unspliced-remote-preselection-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RAW_NAME_RE = re.compile(r"^shard_(?P<index>[0-9]{5})_features\.safetensors\.zst$")
_PCT_NAME_RE = re.compile(r"^shard_(?P<index>[0-9]{5})\.safetensors\.zst$")
_SIDECAR_NAME_RE = re.compile(r"^(?P<stem>shard_[0-9]{5})\.json$")
_TRUSTED_SOURCE_TOP_KEYS = frozenset(
    {
        "schema_version",
        "semantics",
        "trust_percentile_family",
        "raw_source",
        "percentile",
        "feature_statistics",
        "sidecars",
        "shards",
        "aggregate_sha256",
    }
)
_REMOTE_CONFIG_KEYS = frozenset({"staging_dir", "max_output_shard_bytes"})


class RemotePreselectionError(RuntimeError):
    """Remote pre-selection configuration or immutable artifact failure."""


def _canonical_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_sha(value: object, *, surface: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RemotePreselectionError(f"{surface} must be 64 lowercase hexadecimal characters")
    return value


def _strict_nonnegative_int(value: object, *, surface: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RemotePreselectionError(
            f"{surface} must be a non-negative strict integer, got {value!r}"
        )
    return value


def _strict_positive_int(value: object, *, surface: str) -> int:
    parsed = _strict_nonnegative_int(value, surface=surface)
    if parsed == 0:
        raise RemotePreselectionError(f"{surface} must be positive")
    return parsed


def _exact_mapping(value: object, expected_keys: Collection[str]) -> dict[str, object] | None:
    parsed = string_keyed_mapping(value)
    if parsed is None or set(parsed) != set(expected_keys):
        return None
    return parsed


def _strict_selection_shard_name(value: object, *, surface: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != Path(value).name
        or value.startswith(".")
        or "/" in value
        or "\\" in value
        or not value.endswith(COMPRESSED_EXT)
    ):
        raise RemotePreselectionError(
            f"{surface} must be a canonical direct {COMPRESSED_EXT} filename"
        )
    return value


def _join_remote(prefix: str | NormalizedRemotePrefix, *parts: str) -> str:
    value = PurePosixPath(str(prefix), *parts).as_posix()
    try:
        return str(NormalizedRemotePrefix.parse(value))
    except (TypeError, ValueError) as exc:
        raise RemotePreselectionError(f"malformed remote path {value!r}") from exc


def _object_to_json(info: RemoteObjectInfo) -> dict[str, object]:
    return {
        "path": info.path,
        "byte_length": info.byte_length,
        "xet_hash": info.xet_hash,
    }


def _parse_object(raw: object, *, surface: str) -> RemoteObjectInfo:
    required = {"path", "byte_length", "xet_hash"}
    parsed = string_keyed_mapping(raw)
    if parsed is None or not required.issubset(parsed):
        raise RemotePreselectionError(f"{surface} remote identity is malformed")
    raw = parsed
    path = raw["path"]
    byte_length = raw["byte_length"]
    xet_hash = raw["xet_hash"]
    if (
        not isinstance(path, str)
        or not path
        or path.startswith("/")
        or "\\" in path
        or PurePosixPath(path).as_posix() != path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise RemotePreselectionError(f"{surface}.path is malformed")
    parsed_size = _strict_positive_int(byte_length, surface=f"{surface}.byte_length")
    return RemoteObjectInfo(
        path=path,
        byte_length=parsed_size,
        xet_hash=_strict_sha(xet_hash, surface=f"{surface}.xet_hash"),
    )


@dataclass(frozen=True)
class RemotePreselectionConfig:
    staging_dir: Path
    max_output_shard_bytes: int

    @classmethod
    def parse(cls, raw: object) -> RemotePreselectionConfig | None:
        if raw is None:
            return None
        parsed = _exact_mapping(raw, _REMOTE_CONFIG_KEYS)
        if parsed is None:
            keys = set(raw) if isinstance(raw, Mapping) else set()
            raise RemotePreselectionError(
                "remote_preselection must contain exactly "
                f"{sorted(_REMOTE_CONFIG_KEYS)}; got keys={sorted(keys, key=str)}"
            )
        raw = parsed
        staging_raw = raw["staging_dir"]
        if not isinstance(staging_raw, str) or not staging_raw.strip():
            raise RemotePreselectionError(
                "remote_preselection.staging_dir must be a non-empty absolute path"
            )
        staging_candidate = Path(staging_raw).expanduser()
        if not staging_candidate.is_absolute():
            raise RemotePreselectionError(
                "remote_preselection.staging_dir must be an absolute path"
            )
        staging = staging_candidate.absolute()
        try:
            info = staging.lstat()
        except OSError as exc:
            raise RemotePreselectionError(
                f"remote_preselection.staging_dir must already exist: {staging}"
            ) from exc
        if not staging.is_absolute() or not stat.S_ISDIR(info.st_mode):
            raise RemotePreselectionError(
                f"remote_preselection.staging_dir must be a direct directory: {staging}"
            )
        return cls(
            staging_dir=staging,
            max_output_shard_bytes=_strict_positive_int(
                raw["max_output_shard_bytes"],
                surface="remote_preselection.max_output_shard_bytes",
            ),
        )

    def assert_disjoint(self, *paths: str | Path | None) -> None:
        staging = self.staging_dir.resolve()
        for raw in paths:
            if raw is None:
                continue
            path = Path(raw).expanduser().absolute().resolve()
            if path == staging or path.is_relative_to(staging) or staging.is_relative_to(path):
                raise RemotePreselectionError(
                    "remote_preselection.staging_dir must be disjoint from "
                    f"selection inputs/caches: staging={staging}, conflict={path}"
                )


@dataclass(frozen=True)
class RemotePreselectionPrefixes:
    raw_source: NormalizedRemotePrefix
    packed_rows: NormalizedRemotePrefix
    selection_output: NormalizedRemotePrefix


@dataclass(frozen=True)
class MaterializedPackedShard:
    path: Path
    identity: RegularFileIdentity
    remote_identity: RemoteObjectInfo


def resolve_remote_preselection_prefixes(
    *,
    remote: RemoteTrainingConfig,
    feature_root: str | Path,
    feature_dir: str | Path,
    packed_row_dir: str | Path,
    selection_checksum: str,
) -> RemotePreselectionPrefixes:
    raw = resolve_remote_feature_prefix(
        feature_dir,
        local_features_root=feature_root,
        remote_features_root=remote.selection_prefix,
    )
    packed = resolve_remote_feature_prefix(
        packed_row_dir,
        local_features_root=feature_root,
        remote_features_root=remote.selection_prefix,
        surface="packed_row_dir",
    )
    return RemotePreselectionPrefixes(
        raw_source=raw,
        packed_rows=packed,
        selection_output=NormalizedRemotePrefix.parse(
            _join_remote(packed, "sel", _strict_sha(selection_checksum, surface="selection hash"))
        ),
    )


def _parse_strict_json(payload: bytes, *, surface: str) -> dict[str, object]:
    try:
        raw = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RemotePreselectionError(f"{surface} is not strict UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise RemotePreselectionError(f"{surface} must be a JSON object")
    return cast(dict[str, object], raw)


def _stable_digest(
    path: Path,
    *,
    fingerprint_name: str | None = None,
) -> tuple[RegularFileIdentity, str, str]:
    identity = RegularFileIdentity.capture(path)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemotePreselectionError("stable remote staging reads require os.O_NOFOLLOW")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    content = hashlib.sha256()
    salted = hashlib.sha256()
    salted.update((fingerprint_name or path.name).encode("utf-8"))
    salted.update(b"\0")
    try:
        if RegularFileIdentity.from_stat(os.fstat(descriptor)) != identity:
            raise RemotePreselectionError(f"staged file changed before hashing: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(16 * 1024 * 1024):
                content.update(block)
                salted.update(block)
        if RegularFileIdentity.from_stat(os.fstat(descriptor)) != identity:
            raise RemotePreselectionError(f"staged file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    if RegularFileIdentity.capture(path) != identity:
        raise RemotePreselectionError(f"staged file path changed while hashing: {path}")
    return identity, content.hexdigest(), salted.hexdigest()


def _download(
    backend: BucketBackend,
    *,
    info: RemoteObjectInfo,
    path: Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> RegularFileIdentity:
    if path.exists() or path.is_symlink():
        raise RemotePreselectionError(f"owned staging path is not empty: {path}")
    download_verified_object(
        backend,
        expected=info,
        final_path=path,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )
    return RegularFileIdentity.capture(path).with_verified_size(
        verified_byte_length=info.byte_length,
        path=path,
    )


def _unlink_owned(path: Path, identity: RegularFileIdentity) -> None:
    try:
        current = RegularFileIdentity.capture(path)
    except OSError as exc:
        raise RemotePreselectionError(f"owned staged file disappeared: {path}") from exc
    if current != identity:
        raise RemotePreselectionError(
            f"refusing to unlink staged path whose identity changed: {path}"
        )
    path.unlink()


def _listing_by_path(
    listing: Sequence[RemoteObjectInfo], *, surface: str
) -> dict[str, RemoteObjectInfo]:
    result: dict[str, RemoteObjectInfo] = {}
    for info in listing:
        if info.path in result:
            raise RemotePreselectionError(f"{surface} contains duplicate path {info.path!r}")
        result[info.path] = info
    return result


def _parse_source_shard(raw: object, *, index: int) -> dict[str, object]:
    surface = f"remote source shards[{index}]"
    expected = {"stem", "raw", "percentile", "counts"}
    parsed_raw = _exact_mapping(raw, expected)
    if parsed_raw is None:
        raise RemotePreselectionError(f"{surface} has unexpected or missing keys")
    raw = parsed_raw
    stem = raw["stem"]
    if not isinstance(stem, str) or re.fullmatch(r"shard_[0-9]{5}", stem) is None:
        raise RemotePreselectionError(f"{surface}.stem is malformed")
    raw_body = raw["raw"]
    pct_body = raw["percentile"]
    raw_keys = {"path", "byte_length", "xet_hash", "layout"}
    pct_keys = {"path", "byte_length", "xet_hash", "layout"}
    parsed_raw_body = _exact_mapping(raw_body, raw_keys)
    if parsed_raw_body is None:
        raise RemotePreselectionError(f"{surface}.raw has unexpected or missing keys")
    raw_body = parsed_raw_body
    parsed_pct_body = _exact_mapping(pct_body, pct_keys)
    if parsed_pct_body is None:
        raise RemotePreselectionError(f"{surface}.percentile has unexpected or missing keys")
    pct_body = parsed_pct_body
    raw_info = _parse_object(raw_body, surface=f"{surface}.raw")
    pct_info = _parse_object(pct_body, surface=f"{surface}.percentile")
    try:
        raw_layout = parse_semantic_layout(raw_body["layout"], surface=f"{surface}.raw.layout")
        pct_layout = parse_semantic_layout(
            pct_body["layout"], surface=f"{surface}.percentile.layout"
        )
    except SafetensorsLayoutError as exc:
        raise RemotePreselectionError(str(exc)) from exc
    if raw_layout.compressed_bytes != raw_info.byte_length:
        raise RemotePreselectionError(f"{surface}.raw layout size mismatch")
    if pct_layout.compressed_bytes != pct_info.byte_length:
        raise RemotePreselectionError(f"{surface}.percentile layout size mismatch")
    try:
        layout_counts = validate_source_layout_pair(raw_layout, pct_layout)
    except SafetensorsLayoutError as exc:
        raise RemotePreselectionError(str(exc)) from exc
    counts = raw["counts"]
    parsed_counts_raw = _exact_mapping(counts, {"documents", "segments", "features"})
    if parsed_counts_raw is None:
        raise RemotePreselectionError(f"{surface}.counts is malformed")
    counts = parsed_counts_raw
    parsed_counts = {
        key: _strict_nonnegative_int(counts[key], surface=f"{surface}.counts.{key}")
        for key in ("documents", "segments", "features")
    }
    if (
        layout_counts.documents != parsed_counts["documents"]
        or layout_counts.segments != parsed_counts["segments"]
        or layout_counts.features != parsed_counts["features"]
    ):
        raise RemotePreselectionError(f"{surface} counts disagree with raw layout")
    parsed: dict[str, object] = {
        "stem": stem,
        "raw": {
            **_object_to_json(raw_info),
            "layout": semantic_layout_to_json(raw_layout),
        },
        "percentile": {
            **_object_to_json(pct_info),
            "layout": semantic_layout_to_json(pct_layout),
        },
        "counts": parsed_counts,
    }
    return parsed


def parse_remote_preselection_source(raw: object, *, surface: str) -> dict[str, object]:
    """Parse the in-memory remote source contract used by streaming selection.

    The trusted path deliberately contains no body digests and is never uploaded.
    Remote bodies remain size/Xet-bound at materialization time; percentile values
    are accepted as an operator-trusted completed family.
    """
    parsed_raw = _exact_mapping(raw, _TRUSTED_SOURCE_TOP_KEYS)
    if parsed_raw is None:
        raise RemotePreselectionError(
            f"{surface} must contain exactly {sorted(_TRUSTED_SOURCE_TOP_KEYS)}"
        )
    raw = parsed_raw
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != REMOTE_PRESELECTION_SOURCE_SCHEMA_VERSION
        or raw["semantics"] != REMOTE_PRESELECTION_TRUSTED_SOURCE_SEMANTICS
        or raw["trust_percentile_family"] is not True
    ):
        raise RemotePreselectionError(f"{surface} schema or semantics is unsupported")
    raw_source = _exact_mapping(
        raw["raw_source"],
        {
            "bucket_id",
            "prefix",
            "listing_sha256",
            "shard_count",
            "base_shard_content_hash",
            "derived_feature_fingerprint",
        },
    )
    if raw_source is None:
        raise RemotePreselectionError(f"{surface}.raw_source is malformed")
    try:
        bucket_id = str(HfBucketId.parse(raw_source["bucket_id"]))
        source_prefix = str(NormalizedRemotePrefix.parse(raw_source["prefix"]))
    except (TypeError, ValueError) as exc:
        raise RemotePreselectionError(f"{surface}.raw_source path is malformed") from exc
    shard_count = _strict_positive_int(
        raw_source["shard_count"], surface=f"{surface}.raw_source.shard_count"
    )
    percentile = _exact_mapping(raw["percentile"], {"prefix", "checksum", "manifest"})
    if percentile is None:
        raise RemotePreselectionError(f"{surface}.percentile is malformed")
    try:
        percentile_prefix = str(NormalizedRemotePrefix.parse(percentile["prefix"]))
    except (TypeError, ValueError) as exc:
        raise RemotePreselectionError(f"{surface}.percentile.prefix is malformed") from exc
    pct_checksum = _strict_sha(percentile["checksum"], surface=f"{surface}.percentile.checksum")
    if percentile_prefix != _join_remote(source_prefix, "pct", pct_checksum[:16]):
        raise RemotePreselectionError(f"{surface}.percentile prefix is noncanonical")
    manifest_info = _parse_object(percentile["manifest"], surface=f"{surface}.percentile.manifest")
    stats_info = _parse_object(raw["feature_statistics"], surface=f"{surface}.feature_statistics")
    sidecars = _exact_mapping(
        raw["sidecars"], {"count", "sidecar_fingerprint", "files"}
    )
    if sidecars is None:
        raise RemotePreselectionError(f"{surface}.sidecars is malformed")
    sidecar_files = sidecars["files"]
    if not isinstance(sidecar_files, list):
        raise RemotePreselectionError(f"{surface}.sidecars.files must be a list")
    parsed_sidecars: list[dict[str, object]] = []
    prior_sidecar: str | None = None
    for index, raw_item in enumerate(sidecar_files):
        item_surface = f"{surface}.sidecars.files[{index}]"
        parsed_item = _exact_mapping(
            raw_item, {"stem", "path", "byte_length", "xet_hash"}
        )
        if parsed_item is None:
            raise RemotePreselectionError(f"{item_surface} is malformed")
        stem = parsed_item["stem"]
        if not isinstance(stem, str) or re.fullmatch(r"shard_[0-9]{5}", stem) is None:
            raise RemotePreselectionError(f"{item_surface}.stem is malformed")
        if prior_sidecar is not None and stem <= prior_sidecar:
            raise RemotePreselectionError(f"{surface}.sidecars must be strictly stem-sorted")
        prior_sidecar = stem
        parsed_sidecars.append(
            {
                "stem": stem,
                **_object_to_json(_parse_object(parsed_item, surface=item_surface)),
            }
        )
    if _strict_nonnegative_int(sidecars["count"], surface=f"{surface}.sidecars.count") != len(
        parsed_sidecars
    ):
        raise RemotePreselectionError(f"{surface}.sidecars count mismatch")
    sidecar_fingerprint = _strict_sha(
        sidecars["sidecar_fingerprint"], surface=f"{surface}.sidecar_fingerprint"
    )
    shards_raw = raw["shards"]
    if not isinstance(shards_raw, list) or len(shards_raw) != shard_count:
        raise RemotePreselectionError(f"{surface}.shards count mismatch")
    shards = [_parse_source_shard(item, index=index) for index, item in enumerate(shards_raw)]
    stems = [cast(str, item["stem"]) for item in shards]
    expected_stems = [f"shard_{index:05d}" for index in range(shard_count)]
    if stems != expected_stems or [cast(str, item["stem"]) for item in parsed_sidecars] != stems:
        raise RemotePreselectionError(f"{surface} source stems are not contiguous and aligned")
    reconstructed_raw_inventory: list[RemoteFeatureShard] = []
    for index, (shard, sidecar) in enumerate(zip(shards, parsed_sidecars, strict=True)):
        stem = cast(str, shard["stem"])
        raw_entry = cast(Mapping[str, object], shard["raw"])
        pct_entry = cast(Mapping[str, object], shard["percentile"])
        expected_raw = _join_remote(source_prefix, f"{stem}_features{COMPRESSED_EXT}")
        expected_pct = _join_remote(percentile_prefix, f"{stem}{COMPRESSED_EXT}")
        expected_sidecar = _join_remote(source_prefix, PROVENANCE_DIRNAME, f"{stem}.json")
        if (
            raw_entry["path"] != expected_raw
            or pct_entry["path"] != expected_pct
            or sidecar["path"] != expected_sidecar
        ):
            raise RemotePreselectionError(f"{surface} contains a noncanonical path for {stem}")
        reconstructed_raw_inventory.append(
            RemoteFeatureShard(
                index=index,
                info=RemoteObjectInfo(
                    path=cast(str, raw_entry["path"]),
                    byte_length=cast(int, raw_entry["byte_length"]),
                    xet_hash=cast(str, raw_entry["xet_hash"]),
                ),
            )
        )
    if raw_source["listing_sha256"] != remote_feature_inventory_digest(reconstructed_raw_inventory):
        raise RemotePreselectionError(f"{surface}.raw_source.listing_sha256 is stale")
    if manifest_info.path != _join_remote(percentile_prefix, MANIFEST_FILENAME):
        raise RemotePreselectionError(f"{surface} percentile manifest path is noncanonical")
    if stats_info.path not in {
        _join_remote(source_prefix, "feature_statistics.safetensors"),
        _join_remote(percentile_prefix, "feature_statistics.safetensors"),
    }:
        raise RemotePreselectionError(f"{surface} statistics path is noncanonical")
    signed = {
        "schema_version": REMOTE_PRESELECTION_SOURCE_SCHEMA_VERSION,
        "semantics": REMOTE_PRESELECTION_TRUSTED_SOURCE_SEMANTICS,
        "trust_percentile_family": True,
        "raw_source": {
            "bucket_id": bucket_id,
            "prefix": source_prefix,
            "listing_sha256": _strict_sha(
                raw_source["listing_sha256"], surface=f"{surface}.raw_source.listing_sha256"
            ),
            "shard_count": shard_count,
            "base_shard_content_hash": _strict_sha(
                raw_source["base_shard_content_hash"],
                surface=f"{surface}.raw_source.base_shard_content_hash",
            ),
            "derived_feature_fingerprint": _strict_sha(
                raw_source["derived_feature_fingerprint"],
                surface=f"{surface}.raw_source.derived_feature_fingerprint",
            ),
        },
        "percentile": {
            "prefix": percentile_prefix,
            "checksum": pct_checksum,
            "manifest": _object_to_json(manifest_info),
        },
        "feature_statistics": _object_to_json(stats_info),
        "sidecars": {
            "count": len(parsed_sidecars),
            "sidecar_fingerprint": sidecar_fingerprint,
            "files": parsed_sidecars,
        },
        "shards": shards,
    }
    expected_aggregate = _sha256_json(signed)
    aggregate = _strict_sha(raw["aggregate_sha256"], surface=f"{surface}.aggregate_sha256")
    if aggregate != expected_aggregate:
        raise RemotePreselectionError(f"{surface}.aggregate_sha256 mismatch")
    return {**signed, "aggregate_sha256": expected_aggregate}


def download_remote_json_object(
    backend: BucketBackend,
    *,
    remote_path: str,
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
    surface: str,
) -> tuple[RemoteObjectInfo, dict[str, object]]:
    """Download and strictly decode one small immutable control object."""
    info = backend.stat(remote_path)
    if info is None:
        raise RemotePreselectionError(f"{surface} is absent: {remote_path}")
    parent = Path(staging_dir).expanduser().absolute()
    if not parent.is_dir():
        raise RemotePreselectionError(f"control staging directory is absent: {parent}")
    with tempfile.TemporaryDirectory(prefix="remote-control-", dir=parent) as tmp:
        path = Path(tmp) / PurePosixPath(remote_path).name
        identity = _download(
            backend,
            info=info,
            path=path,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        _same, digest, _salted = _stable_digest(path)
        payload = path.read_bytes()
        if RegularFileIdentity.capture(path) != identity:
            raise RemotePreselectionError(f"{surface} changed during download")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RemotePreselectionError(f"{surface} digest changed during read")
        parsed = _parse_strict_json(payload, surface=surface)
    return info, parsed


def download_remote_object_bytes(
    backend: BucketBackend,
    *,
    remote_path: str,
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
    surface: str,
) -> tuple[RemoteObjectInfo, bytes]:
    """Download one small immutable control object and return its exact bytes."""
    info = backend.stat(remote_path)
    if info is None:
        raise RemotePreselectionError(f"{surface} is absent: {remote_path}")
    parent = Path(staging_dir).expanduser().absolute()
    if not parent.is_dir():
        raise RemotePreselectionError(f"control staging directory is absent: {parent}")
    with tempfile.TemporaryDirectory(prefix="remote-control-bytes-", dir=parent) as tmp:
        path = Path(tmp) / PurePosixPath(remote_path).name
        identity = _download(
            backend,
            info=info,
            path=path,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        payload = path.read_bytes()
        if RegularFileIdentity.capture(path) != identity:
            raise RemotePreselectionError(f"{surface} changed during download")
        if hashlib.sha256(payload).hexdigest() != _stable_digest(path)[1]:
            raise RemotePreselectionError(f"{surface} digest changed during read")
    return info, payload


def build_trusted_remote_preselection_source(
    backend: BucketBackend,
    *,
    bucket_id: str | HfBucketId,
    prefixes: RemotePreselectionPrefixes,
    percentile_checksum: str,
    feature_dir: str | Path,
    expected_stems: Sequence[str],
    sidecar_fingerprint: str,
    derived_feature_fingerprint: str,
) -> dict[str, object]:
    """Build an ephemeral source contract without certifying percentile bodies.

    The remote listing supplies immutable transport identities. Local compressed
    safetensors headers supply layout/count metadata, while the operator-provided
    percentile family is trusted as-is. Nothing is downloaded, recomputed, or
    published by this helper.
    """
    checksum = _strict_sha(percentile_checksum, surface="remote percentile checksum")
    source_dir = Path(feature_dir).expanduser().absolute()
    expected = list(expected_stems)
    if expected != sorted(set(expected)) or not expected:
        raise RemotePreselectionError("trusted source stems must be sorted and unique")
    try:
        raw_inventory = tuple(
            normalize_remote_feature_inventory(
                backend.list(str(prefixes.raw_source)),
                prefix=prefixes.raw_source,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RemotePreselectionError("remote raw source listing is invalid") from exc
    stems = [f"shard_{entry.index:05d}" for entry in raw_inventory]
    if stems != expected:
        raise RemotePreselectionError(
            f"remote raw source stems differ from packed controls: {stems[:3]}..."
        )
    pct_prefix = NormalizedRemotePrefix.parse(
        _join_remote(prefixes.raw_source, "pct", checksum[:16])
    )
    pct_listing = tuple(backend.list(str(pct_prefix)))
    pct_by_stem: dict[str, RemoteObjectInfo] = {}
    for info in pct_listing:
        try:
            relative = PurePosixPath(info.path).relative_to(PurePosixPath(str(pct_prefix)))
        except ValueError:
            continue
        if len(relative.parts) != 1:
            continue
        match = _PCT_NAME_RE.fullmatch(relative.name)
        if match is not None:
            stem = f"shard_{match.group('index')}"
            if stem in pct_by_stem:
                raise RemotePreselectionError(f"duplicate remote percentile body {stem}")
            pct_by_stem[stem] = info
    if sorted(pct_by_stem) != expected:
        raise RemotePreselectionError("remote percentile bodies do not cover packed controls")

    manifest_info = backend.stat(_join_remote(pct_prefix, MANIFEST_FILENAME))
    if manifest_info is None:
        raise RemotePreselectionError("remote percentile manifest is absent")
    stats_path = find_statistics_path(source_dir)
    if stats_path is None:
        raise RemotePreselectionError(f"local feature statistics are absent: {source_dir}")
    stats_info = backend.stat(_join_remote(prefixes.raw_source, stats_path.name))
    if stats_info is None:
        stats_info = backend.stat(_join_remote(pct_prefix, stats_path.name))
    if stats_info is None:
        raise RemotePreselectionError("remote feature statistics are absent")
    pct_dir = source_dir / "pct" / checksum[:16]
    pct_manifest = read_manifest(pct_dir)
    if (
        pct_manifest is None
        or pct_manifest.get("checksum") != checksum
        or not isinstance(pct_manifest.get("base_shard_content_hash"), str)
    ):
        raise RemotePreselectionError("local percentile manifest is not compatible with trust mode")
    sidecar_listing = tuple(backend.list(_join_remote(prefixes.raw_source, PROVENANCE_DIRNAME)))
    sidecar_by_stem: dict[str, RemoteObjectInfo] = {}
    for info in sidecar_listing:
        name = PurePosixPath(info.path).name
        match = _SIDECAR_NAME_RE.fullmatch(name)
        if match is None or PurePosixPath(info.path).parent != PurePosixPath(
            _join_remote(prefixes.raw_source, PROVENANCE_DIRNAME)
        ):
            continue
        stem = match.group("stem")
        if stem in sidecar_by_stem:
            raise RemotePreselectionError(f"duplicate remote splice sidecar {stem}")
        sidecar_by_stem[stem] = info
    if sorted(sidecar_by_stem) != expected:
        raise RemotePreselectionError("remote splice sidecars do not cover packed controls")

    shard_entries: list[dict[str, object]] = []
    for raw_entry in raw_inventory:
        stem = f"shard_{raw_entry.index:05d}"
        raw_path = source_dir / f"{stem}_features{COMPRESSED_EXT}"
        pct_path = pct_dir / f"{stem}{COMPRESSED_EXT}"
        if not raw_path.is_file() or not pct_path.is_file():
            raise RemotePreselectionError(f"local layout controls are absent for {stem}")
        try:
            raw_layout = read_compressed_safetensors_layout(raw_path)
            pct_layout = read_compressed_safetensors_layout(pct_path)
            counts = validate_source_layout_pair(raw_layout, pct_layout)
        except (OSError, SafetensorsLayoutError) as exc:
            raise RemotePreselectionError(f"local layout controls are invalid for {stem}") from exc
        shard_entries.append(
            {
                "stem": stem,
                "raw": {
                    **_object_to_json(raw_entry.info),
                    "layout": semantic_layout_to_json(raw_layout),
                },
                "percentile": {
                    **_object_to_json(pct_by_stem[stem]),
                    "layout": semantic_layout_to_json(pct_layout),
                },
                "counts": {
                    "documents": counts.documents,
                    "segments": counts.segments,
                    "features": counts.features,
                },
            }
        )

    signed: dict[str, object] = {
        "schema_version": REMOTE_PRESELECTION_SOURCE_SCHEMA_VERSION,
        "semantics": REMOTE_PRESELECTION_TRUSTED_SOURCE_SEMANTICS,
        "trust_percentile_family": True,
        "raw_source": {
            "bucket_id": str(HfBucketId.parse(bucket_id)),
            "prefix": str(prefixes.raw_source),
            "listing_sha256": remote_feature_inventory_digest(raw_inventory),
            "shard_count": len(shard_entries),
            "base_shard_content_hash": pct_manifest["base_shard_content_hash"],
            "derived_feature_fingerprint": _strict_sha(
                derived_feature_fingerprint, surface="derived feature fingerprint"
            ),
        },
        "percentile": {
            "prefix": str(pct_prefix),
            "checksum": checksum,
            "manifest": _object_to_json(manifest_info),
        },
        "feature_statistics": _object_to_json(stats_info),
        "sidecars": {
            "count": len(expected),
            "sidecar_fingerprint": _strict_sha(
                sidecar_fingerprint, surface="splice sidecar fingerprint"
            ),
            "files": [
                {"stem": stem, **_object_to_json(sidecar_by_stem[stem])} for stem in expected
            ],
        },
        "shards": shard_entries,
    }
    return {**signed, "aggregate_sha256": _sha256_json(signed)}


def download_remote_packed_preselection_inventory(
    backend: BucketBackend,
    *,
    prefixes: RemotePreselectionPrefixes,
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> dict[str, object]:
    """Download and strictly parse the schema-v2 packed control inventory."""
    _packed_info, packed_raw = download_remote_json_object(
        backend,
        remote_path=_join_remote(prefixes.packed_rows, REMOTE_PACKED_INVENTORY_FILENAME),
        staging_dir=staging_dir,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
        surface="remote packed inventory",
    )
    try:
        packed = parse_remote_packed_inventory(packed_raw, surface="remote packed inventory")
    except Exception as exc:
        raise RemotePreselectionError("remote packed inventory is invalid") from exc
    if packed["schema_version"] != 2:
        raise RemotePreselectionError(
            "remote pre-selection requires remote_packed_inventory.json schema v2"
        )
    return cast(dict[str, object], packed)


def validate_local_remote_packed_controls(
    *,
    packed_row_dir: str | Path,
    packed_staging_dir: str | Path,
    packed_inventory: Mapping[str, object],
) -> PackedProvenanceInfo:
    """Bind the local packed manifest/provenance controls to inventory v2."""
    packed = parse_remote_packed_inventory(packed_inventory, surface="remote packed inventory")
    if packed["schema_version"] != 2:
        raise RemotePreselectionError("remote pre-selection requires packed schema v2")
    packed_dir = Path(packed_row_dir).expanduser().absolute()
    manifest_path = packed_dir / MANIFEST_FILENAME
    manifest = read_manifest(packed_dir)
    if manifest is None or manifest.get("kind") != OverlayKind.PACKED_ROW.value:
        raise RemotePreselectionError(f"local packed manifest is invalid: {manifest_path}")
    if _file_fingerprint(manifest_path) != packed["packed_manifest_file_fingerprint"]:
        raise RemotePreselectionError(
            "local packed manifest differs from the remote packed inventory"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RemotePreselectionError("local packed manifest metadata is malformed")
    stems_raw = metadata.get("base_shard_stems")
    if (
        not isinstance(stems_raw, list)
        or not stems_raw
        or not all(
            isinstance(stem, str) and re.fullmatch(r"shard_[0-9]{5}", stem) is not None
            for stem in stems_raw
        )
    ):
        raise RemotePreselectionError(
            "local packed manifest base_shard_stems must be a non-empty canonical string list"
        )
    stems = [cast(str, stem) for stem in stems_raw]
    if stems != sorted(set(stems)):
        raise RemotePreselectionError(
            "local packed manifest base_shard_stems must be sorted and unique"
        )

    packed_entries = cast(list[dict[str, object]], packed["shards"])
    names = [cast(str, entry["name"]) for entry in packed_entries]
    if not names or names != sorted(set(names)):
        raise RemotePreselectionError(
            "remote packed inventory names must be non-empty, sorted, and unique"
        )
    prov_dir = provenance_dir(packed_dir)
    try:
        prov_entries = sorted(prov_dir.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise RemotePreselectionError(
            f"cannot enumerate local packed provenance directory: {prov_dir}"
        ) from exc
    if any(
        path.name not in names or not path.is_file() or path.is_symlink() for path in prov_entries
    ):
        raise RemotePreselectionError(
            "local packed provenance directory contains an unknown or non-regular entry"
        )
    prov_paths = sorted(prov_dir.glob(f"*{COMPRESSED_EXT}"), key=lambda path: path.name)
    if [path.name for path in prov_paths] != names:
        raise RemotePreselectionError(
            "local packed provenance does not exactly cover the remote packed inventory"
        )
    if len(prov_paths) != packed["provenance_shard_count"]:
        raise RemotePreselectionError("packed provenance shard count drift")
    for entry, prov_path in zip(packed_entries, prov_paths, strict=True):
        tensors = load_overlay_shard(prov_path)
        if set(tensors) != {
            PROVENANCE_BASE_SHARD_IDX_KEY,
            PROVENANCE_DOC_IDX_KEY,
            PROVENANCE_SEG_ORDINAL_KEY,
        }:
            raise RemotePreselectionError(f"packed provenance tensor keys are invalid: {prov_path}")
        expected_segments = cast(int, entry["total_segments"])
        values = tuple(tensors[key] for key in sorted(tensors))
        if any(
            tensor.dtype is not t.int32 or tensor.ndim != 1 or len(tensor) != expected_segments
            for tensor in values
        ):
            raise RemotePreselectionError(f"packed provenance shape/dtype mismatch: {prov_path}")
        base_idx = tensors[PROVENANCE_BASE_SHARD_IDX_KEY]
        doc_idx = tensors[PROVENANCE_DOC_IDX_KEY]
        seg_ordinal = tensors[PROVENANCE_SEG_ORDINAL_KEY]
        if bool((base_idx < 0).any() or (doc_idx < 0).any() or (seg_ordinal < 0).any()):
            raise RemotePreselectionError(
                f"packed provenance contains negative values: {prov_path}"
            )
        if bool((base_idx >= len(stems)).any()):
            raise RemotePreselectionError(
                f"packed provenance base-shard index is out of range: {prov_path}"
            )
    prov_fingerprint = _fold_file_digests([_file_fingerprint(path) for path in prov_paths])
    if prov_fingerprint != packed["provenance_fingerprint"]:
        raise RemotePreselectionError("local packed provenance fingerprint differs from inventory")
    staging = Path(packed_staging_dir).expanduser().absolute()
    return PackedProvenanceInfo(
        base_shard_stems=stems,
        packed_shard_paths=[staging / name for name in names],
        provenance_shard_paths=prov_paths,
        packed_stem_fingerprint=_fingerprint_json(stems),
        packed_row_fingerprint=cast(str, packed["packed_row_fingerprint"]),
        provenance_fingerprint=prov_fingerprint,
    )


def validate_local_remote_preselection_controls(
    *,
    packed_row_dir: str | Path,
    feature_dir: str | Path,
    packed_staging_dir: str | Path,
    packed_inventory: Mapping[str, object],
    source_inventory: Mapping[str, object],
    expected_segment_size: int,
) -> PackedProvenanceInfo:
    """Bind local control artifacts to both remote body inventories."""
    packed_info = validate_local_remote_packed_controls(
        packed_row_dir=packed_row_dir,
        packed_staging_dir=packed_staging_dir,
        packed_inventory=packed_inventory,
    )
    source = parse_remote_preselection_source(source_inventory, surface="remote source inventory")
    packed_dir = Path(packed_row_dir).expanduser().absolute()
    source_dir = Path(feature_dir).expanduser().absolute()
    manifest = read_manifest(packed_dir)
    assert manifest is not None
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RemotePreselectionError("local packed manifest metadata is malformed")
    stems_raw = metadata.get("base_shard_stems")
    if not isinstance(stems_raw, list) or not all(isinstance(stem, str) for stem in stems_raw):
        raise RemotePreselectionError("local packed manifest has no base_shard_stems")
    stems = packed_info.base_shard_stems
    source_shards = cast(list[dict[str, object]], source["shards"])
    source_stems = [cast(str, entry["stem"]) for entry in source_shards]
    if stems != source_stems:
        raise RemotePreselectionError("packed manifest and source inventory stem domains differ")
    for entry in source_shards:
        raw = cast(Mapping[str, object], entry["raw"])
        layout = parse_semantic_layout(raw["layout"], surface=f"source layout {entry['stem']}")
        segment_size_raw = dict(layout.metadata).get("segment_size")
        if segment_size_raw != str(expected_segment_size):
            raise RemotePreselectionError(
                f"source {entry['stem']} has segment_size={segment_size_raw!r}, "
                f"expected {expected_segment_size}"
            )
    percentile = cast(Mapping[str, object], source["percentile"])
    pct_checksum = percentile["checksum"]
    if metadata.get("pct_checksum") != pct_checksum:
        raise RemotePreselectionError(
            "packed manifest percentile checksum differs from source inventory"
        )
    stats_path = find_statistics_path(source_dir)
    if stats_path is None:
        raise RemotePreselectionError(f"local feature statistics are absent: {source_dir}")
    stats = load_statistics(stats_path)
    statistics_pct_checksum = compute_percentile_checksum(
        build_percentile_table(
            stats.per_feature_stats,
            stats.metadata.n_sae_features,
        )
    )
    if statistics_pct_checksum != pct_checksum:
        raise RemotePreselectionError(
            "local feature statistics produce a different percentile checksum"
        )
    pct_dir = source_dir / "pct" / cast(str, pct_checksum)[:16]
    pct_manifest = read_manifest(pct_dir)
    raw_source = cast(Mapping[str, object], source["raw_source"])
    if (
        pct_manifest is None
        or pct_manifest.get("checksum") != pct_checksum
        or pct_manifest.get("base_shard_content_hash") != raw_source["base_shard_content_hash"]
    ):
        raise RemotePreselectionError(
            "local percentile manifest identity differs from source inventory"
        )

    local_sidecar_hash = splice_sidecar_fingerprint(source_dir)
    sidecars = cast(Mapping[str, object], source["sidecars"])
    if local_sidecar_hash != sidecars["sidecar_fingerprint"]:
        raise RemotePreselectionError("local splice sidecars differ from the source inventory")
    return packed_info


def build_transport_feature_catalog(
    inventory: Mapping[str, object],
    *,
    staging_dir: str | Path,
    segment_size: int,
) -> TransportFeatureIndexCatalog:
    parsed = parse_remote_preselection_source(inventory, surface="transport source inventory")
    root = Path(staging_dir).expanduser().absolute()
    worker = root / f"source-worker-{os.getpid()}"
    raw_dir = worker / "raw"
    percentile_dir = worker / "pct"
    entries = tuple(
        (
            cast(str, item["stem"]),
            str(
                raw_dir
                / PurePosixPath(cast(str, cast(Mapping[str, object], item["raw"])["path"])).name
            ),
            str(percentile_dir / f"{item['stem']}{COMPRESSED_EXT}"),
        )
        for item in cast(list[dict[str, object]], parsed["shards"])
    )
    raw_source = cast(Mapping[str, object], parsed["raw_source"])
    percentile = cast(Mapping[str, object], parsed["percentile"])
    catalog = TransportFeatureIndexCatalog(
        feature_dir=str(raw_dir),
        percentile_overlay_dir=str(percentile_dir),
        entries=entries,
        segment_size=segment_size,
        percentile_checksum=cast(str, percentile["checksum"]),
        validated_content_fingerprint=cast(str, raw_source["derived_feature_fingerprint"]),
    )
    catalog.validate()
    return catalog


class RemoteFeatureShardMaterializer:
    """Worker-local, transient raw/percentile downloader for ``FeatureIndex``."""

    def __init__(
        self,
        *,
        bucket_id: str | HfBucketId,
        inventory: Mapping[str, object],
        catalog: TransportFeatureIndexCatalog,
        retry_timeout_seconds: float,
        retry_interval_seconds: float,
        backend: BucketBackend | None = None,
    ) -> None:
        self.bucket_id = HfBucketId.parse(bucket_id)
        self.inventory = parse_remote_preselection_source(
            inventory, surface="FeatureIndex remote source inventory"
        )
        self.catalog = catalog
        self.catalog.validate()
        self.retry_timeout_seconds = retry_timeout_seconds
        self.retry_interval_seconds = retry_interval_seconds
        self._backend = backend
        self._entries = {
            cast(str, item["stem"]): item
            for item in cast(list[dict[str, object]], self.inventory["shards"])
        }

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_backend"] = None
        return state

    def _resolved_backend(self) -> BucketBackend:
        if self._backend is None:
            self._backend = HfBucketBackend(self.bucket_id)
        return self._backend

    @contextmanager
    def materialize(self, stem: str) -> Iterator[MaterializedFeatureShard]:
        try:
            entry = self._entries[stem]
        except KeyError as exc:
            raise RemotePreselectionError(f"source inventory has no stem {stem!r}") from exc
        paths = {item_stem: (Path(raw), Path(pct)) for item_stem, raw, pct in self.catalog.entries}
        raw_path, pct_path = paths[stem]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        pct_path.parent.mkdir(parents=True, exist_ok=True)
        raw_info = _parse_object(entry["raw"], surface=f"source {stem}.raw")
        pct_info = _parse_object(entry["percentile"], surface=f"source {stem}.percentile")
        raw_identity: RegularFileIdentity | None = None
        pct_identity: RegularFileIdentity | None = None
        try:
            raw_identity = _download(
                self._resolved_backend(),
                info=raw_info,
                path=raw_path,
                timeout_seconds=self.retry_timeout_seconds,
                retry_interval_seconds=self.retry_interval_seconds,
            )
            pct_identity = _download(
                self._resolved_backend(),
                info=pct_info,
                path=pct_path,
                timeout_seconds=self.retry_timeout_seconds,
                retry_interval_seconds=self.retry_interval_seconds,
            )
            yield MaterializedFeatureShard(
                raw_path=raw_path,
                raw_identity=raw_identity,
                percentile_path=pct_path,
                percentile_identity=pct_identity,
            )
        finally:
            if pct_identity is not None:
                _unlink_owned(pct_path, pct_identity)
            if raw_identity is not None:
                _unlink_owned(raw_path, raw_identity)


@contextmanager
def materialize_remote_packed_shard(
    backend: BucketBackend,
    *,
    packed_prefix: str | NormalizedRemotePrefix,
    entry: Mapping[str, object],
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> Iterator[MaterializedPackedShard]:
    """Stage one authenticated packed body and remove it after stable use.

    The parent owns this context for the complete worker lifetime.
    """
    name = entry.get("name")
    if not isinstance(name, str) or name != Path(name).name:
        raise RemotePreselectionError("packed inventory entry name is malformed")
    remote_path = _join_remote(packed_prefix, name)
    live = backend.stat(remote_path)
    if live is None:
        raise RemotePreselectionError(f"remote packed body is absent: {remote_path}")
    if live.byte_length != entry.get("byte_length"):
        raise RemotePreselectionError(f"remote packed body size drift: {remote_path}")
    directory = Path(staging_dir).expanduser().absolute()
    path = directory / name
    identity: RegularFileIdentity | None = None
    try:
        identity = _download(
            backend,
            info=live,
            path=path,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        _same, digest, fingerprint = _stable_digest(path)
        if digest != entry.get("content_sha256") or fingerprint != entry.get("file_fingerprint"):
            raise RemotePreselectionError(
                f"remote packed body differs from inventory: {remote_path}"
            )
        layout = read_compressed_safetensors_layout(path)
        if semantic_layout_to_json(layout) != entry.get("layout"):
            raise RemotePreselectionError(
                f"remote packed body layout differs from inventory: {remote_path}"
            )
        yield MaterializedPackedShard(
            path=path,
            identity=identity,
            remote_identity=live,
        )
        if RegularFileIdentity.capture(path) != identity:
            raise RemotePreselectionError(f"staged packed body changed during selection: {path}")
    finally:
        if identity is not None:
            _unlink_owned(path, identity)


def local_object_info_for_remote_path(
    path: Path, *, remote_path: str
) -> tuple[RemoteObjectInfo, str]:
    """Return publication identity and compressed SHA under one stable fence."""
    info = verified_local_object_info(local_path=path, remote_path=remote_path)
    _identity, content_sha, _fingerprint = _stable_digest(path)
    if RegularFileIdentity.capture(path).size != info.byte_length:
        raise RemotePreselectionError(f"selection body changed while hashing: {path}")
    return info, content_sha


def build_remote_preselection_receipt(
    *,
    selection_checksum: str,
    packed_inventory_aggregate: str,
    source_inventory_aggregate: str,
    packed_row_fingerprint: str,
    shard_name: str,
    expected_packed_segments: int,
    selection_body: RemoteObjectInfo,
    content_sha256: str,
    splice_stats: Mapping[str, object],
    max_features_per_segment: int,
) -> dict[str, object]:
    shard_name = _strict_selection_shard_name(shard_name, surface="receipt shard_name")
    stats = {
        key: _strict_nonnegative_int(splice_stats.get(key), surface=f"receipt stats.{key}")
        for key in (
            "total_packed_segments",
            "total_sidecar_splice_positions_referenced",
            "total_delta_positions_selected",
        )
    }
    signed: dict[str, object] = {
        "schema_version": REMOTE_PRESELECTION_RECEIPT_SCHEMA_VERSION,
        "semantics": REMOTE_PRESELECTION_SEMANTICS,
        "selection_checksum": _strict_sha(selection_checksum, surface="receipt selection_checksum"),
        "packed_inventory_aggregate": _strict_sha(
            packed_inventory_aggregate, surface="receipt packed inventory aggregate"
        ),
        "source_inventory_aggregate": _strict_sha(
            source_inventory_aggregate, surface="receipt source inventory aggregate"
        ),
        "packed_row_fingerprint": _strict_sha(
            packed_row_fingerprint, surface="receipt packed row fingerprint"
        ),
        "shard_name": shard_name,
        "expected_packed_segments": _strict_nonnegative_int(
            expected_packed_segments, surface="receipt expected packed segments"
        ),
        "selection_body": {
            **_object_to_json(selection_body),
            "content_sha256": _strict_sha(
                content_sha256, surface="receipt selection body content sha256"
            ),
        },
        "splice_stats": stats,
        "max_features_per_segment": _strict_nonnegative_int(
            max_features_per_segment, surface="receipt max features per segment"
        ),
    }
    return {**signed, "aggregate_sha256": _sha256_json(signed)}


def parse_remote_preselection_receipt(raw: object, *, surface: str) -> dict[str, object]:
    expected = {
        "schema_version",
        "semantics",
        "selection_checksum",
        "packed_inventory_aggregate",
        "source_inventory_aggregate",
        "packed_row_fingerprint",
        "shard_name",
        "expected_packed_segments",
        "selection_body",
        "splice_stats",
        "max_features_per_segment",
        "aggregate_sha256",
    }
    parsed_raw = _exact_mapping(raw, expected)
    if parsed_raw is None:
        raise RemotePreselectionError(f"{surface} has unexpected or missing keys")
    raw = parsed_raw
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != REMOTE_PRESELECTION_RECEIPT_SCHEMA_VERSION
    ):
        raise RemotePreselectionError(f"{surface} schema version is unsupported")
    if raw["semantics"] != REMOTE_PRESELECTION_SEMANTICS:
        raise RemotePreselectionError(f"{surface} semantics is unsupported")
    body_raw = _exact_mapping(
        raw["selection_body"], {"path", "byte_length", "xet_hash", "content_sha256"}
    )
    if body_raw is None:
        raise RemotePreselectionError(f"{surface}.selection_body is malformed")
    expected_stats = {
        "total_packed_segments",
        "total_sidecar_splice_positions_referenced",
        "total_delta_positions_selected",
    }
    stats_raw = _exact_mapping(raw["splice_stats"], expected_stats)
    if stats_raw is None:
        raise RemotePreselectionError(f"{surface}.splice_stats is malformed")
    expected_packed_segments = _strict_nonnegative_int(
        raw["expected_packed_segments"],
        surface=f"{surface}.expected_packed_segments",
    )
    total_packed_segments = _strict_nonnegative_int(
        stats_raw["total_packed_segments"],
        surface=f"{surface}.splice_stats.total_packed_segments",
    )
    if expected_packed_segments != total_packed_segments:
        raise RemotePreselectionError(
            f"{surface} expected packed segments disagree with splice stats"
        )
    body = _parse_object(body_raw, surface=f"{surface}.selection_body")
    rebuilt = build_remote_preselection_receipt(
        selection_checksum=_strict_sha(
            raw["selection_checksum"], surface=f"{surface}.selection_checksum"
        ),
        packed_inventory_aggregate=_strict_sha(
            raw["packed_inventory_aggregate"],
            surface=f"{surface}.packed_inventory_aggregate",
        ),
        source_inventory_aggregate=_strict_sha(
            raw["source_inventory_aggregate"],
            surface=f"{surface}.source_inventory_aggregate",
        ),
        packed_row_fingerprint=_strict_sha(
            raw["packed_row_fingerprint"],
            surface=f"{surface}.packed_row_fingerprint",
        ),
        shard_name=(raw["shard_name"] if isinstance(raw["shard_name"], str) else ""),
        expected_packed_segments=expected_packed_segments,
        selection_body=body,
        content_sha256=_strict_sha(
            body_raw["content_sha256"],
            surface=f"{surface}.selection_body.content_sha256",
        ),
        splice_stats=stats_raw,
        max_features_per_segment=_strict_nonnegative_int(
            raw["max_features_per_segment"],
            surface=f"{surface}.max_features_per_segment",
        ),
    )
    if dict(raw) != rebuilt:
        raise RemotePreselectionError(f"{surface} aggregate or canonical fields mismatch")
    return rebuilt


def build_remote_unspliced_preselection_receipt(
    *,
    selection_checksum: str,
    packed_inventory_aggregate: str,
    packed_row_fingerprint: str,
    shard_name: str,
    expected_packed_segments: int,
    selection_body: RemoteObjectInfo,
    content_sha256: str,
    max_features_per_segment: int,
    post_selection_rng_sha256: str,
) -> dict[str, object]:
    """Build one canonical clustered ordinary-selection receipt."""
    canonical_name = _strict_selection_shard_name(shard_name, surface="ordinary receipt shard_name")
    canonical_body = _parse_object(
        _object_to_json(selection_body),
        surface="ordinary receipt selection_body",
    )
    signed: dict[str, object] = {
        "schema_version": REMOTE_UNSPLICED_PRESELECTION_RECEIPT_SCHEMA_VERSION,
        "semantics": REMOTE_UNSPLICED_PRESELECTION_SEMANTICS,
        "selection_checksum": _strict_sha(
            selection_checksum, surface="ordinary receipt selection_checksum"
        ),
        "packed_inventory_aggregate": _strict_sha(
            packed_inventory_aggregate,
            surface="ordinary receipt packed inventory aggregate",
        ),
        "packed_row_fingerprint": _strict_sha(
            packed_row_fingerprint,
            surface="ordinary receipt packed row fingerprint",
        ),
        "shard_name": canonical_name,
        "expected_packed_segments": _strict_nonnegative_int(
            expected_packed_segments,
            surface="ordinary receipt expected packed segments",
        ),
        "selection_body": {
            **_object_to_json(canonical_body),
            "content_sha256": _strict_sha(
                content_sha256,
                surface="ordinary receipt selection body content sha256",
            ),
        },
        "max_features_per_segment": _strict_nonnegative_int(
            max_features_per_segment,
            surface="ordinary receipt max features per segment",
        ),
        "post_selection_rng_sha256": _strict_sha(
            post_selection_rng_sha256,
            surface="ordinary receipt post-selection RNG sha256",
        ),
    }
    return {**signed, "aggregate_sha256": _sha256_json(signed)}


def parse_remote_unspliced_preselection_receipt(raw: object, *, surface: str) -> dict[str, object]:
    """Strictly parse and canonically rebuild an ordinary receipt schema v2."""
    expected = {
        "schema_version",
        "semantics",
        "selection_checksum",
        "packed_inventory_aggregate",
        "packed_row_fingerprint",
        "shard_name",
        "expected_packed_segments",
        "selection_body",
        "max_features_per_segment",
        "post_selection_rng_sha256",
        "aggregate_sha256",
    }
    parsed_raw = _exact_mapping(raw, expected)
    if parsed_raw is None:
        raise RemotePreselectionError(f"{surface} has unexpected or missing keys")
    raw = parsed_raw
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != REMOTE_UNSPLICED_PRESELECTION_RECEIPT_SCHEMA_VERSION
    ):
        raise RemotePreselectionError(f"{surface} schema version is unsupported")
    if raw["semantics"] != REMOTE_UNSPLICED_PRESELECTION_SEMANTICS:
        raise RemotePreselectionError(f"{surface} semantics is unsupported")
    body_raw = _exact_mapping(
        raw["selection_body"], {"path", "byte_length", "xet_hash", "content_sha256"}
    )
    if body_raw is None:
        raise RemotePreselectionError(f"{surface}.selection_body is malformed")
    body = _parse_object(body_raw, surface=f"{surface}.selection_body")
    rebuilt = build_remote_unspliced_preselection_receipt(
        selection_checksum=_strict_sha(
            raw["selection_checksum"], surface=f"{surface}.selection_checksum"
        ),
        packed_inventory_aggregate=_strict_sha(
            raw["packed_inventory_aggregate"],
            surface=f"{surface}.packed_inventory_aggregate",
        ),
        packed_row_fingerprint=_strict_sha(
            raw["packed_row_fingerprint"],
            surface=f"{surface}.packed_row_fingerprint",
        ),
        shard_name=_strict_selection_shard_name(raw["shard_name"], surface=f"{surface}.shard_name"),
        expected_packed_segments=_strict_nonnegative_int(
            raw["expected_packed_segments"],
            surface=f"{surface}.expected_packed_segments",
        ),
        selection_body=body,
        content_sha256=_strict_sha(
            body_raw["content_sha256"],
            surface=f"{surface}.selection_body.content_sha256",
        ),
        max_features_per_segment=_strict_nonnegative_int(
            raw["max_features_per_segment"],
            surface=f"{surface}.max_features_per_segment",
        ),
        post_selection_rng_sha256=_strict_sha(
            raw["post_selection_rng_sha256"],
            surface=f"{surface}.post_selection_rng_sha256",
        ),
    )
    if dict(raw) != rebuilt:
        raise RemotePreselectionError(f"{surface} aggregate or canonical fields mismatch")
    return rebuilt


def _selection_integrity_from_parsed_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for name in sorted(receipts):
        receipt = receipts[name]
        if receipt["shard_name"] != name:
            raise RemotePreselectionError(
                f"receipt map key {name!r} disagrees with signed shard name"
            )
        body = cast(Mapping[str, object], receipt["selection_body"])
        entries.append(
            {
                "name": name,
                "byte_length": body["byte_length"],
                "content_sha256": body["content_sha256"],
            }
        )
    signed: dict[str, object] = {"schema_version": 1, "shards": entries}
    try:
        return cast(
            dict[str, object],
            parse_clustered_selection_shard_inventory(
                {**signed, "aggregate_sha256": _sha256_json(signed)},
                surface="remote pre-selection receipts",
            ),
        )
    except Exception as exc:
        raise RemotePreselectionError(
            "remote pre-selection receipts do not form a valid selection inventory"
        ) from exc


def selection_integrity_from_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct the existing selection-body inventory from receipts."""
    parsed = {
        name: parse_remote_preselection_receipt(
            receipt, surface=f"remote pre-selection receipt {name}"
        )
        for name, receipt in receipts.items()
    }
    return _selection_integrity_from_parsed_receipts(parsed)


def selection_integrity_from_unspliced_receipts(
    receipts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct clustered selection integrity from ordinary receipts."""
    parsed = {
        name: parse_remote_unspliced_preselection_receipt(
            receipt, surface=f"remote ordinary pre-selection receipt {name}"
        )
        for name, receipt in receipts.items()
    }
    return _selection_integrity_from_parsed_receipts(parsed)


def _publish_parsed_remote_preselection_shard(
    backend: BucketBackend,
    *,
    selection_path: Path,
    parsed_receipt: Mapping[str, object],
    output_prefix: str | NormalizedRemotePrefix,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> tuple[RemoteObjectInfo, RemoteObjectInfo]:
    """Publish one already parsed body/receipt pair in immutable order."""
    if parsed_receipt["shard_name"] != selection_path.name:
        raise RemotePreselectionError("selection path and receipt shard name differ")
    body_remote_path = _join_remote(output_prefix, selection_path.name)
    signed_body = cast(Mapping[str, object], parsed_receipt["selection_body"])
    if signed_body["path"] != body_remote_path:
        raise RemotePreselectionError("receipt selection-body path is noncanonical")
    body_info = upload_verified_object(
        backend,
        local_path=selection_path,
        remote_path=body_remote_path,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )
    if _object_to_json(body_info) != {
        key: signed_body[key] for key in ("path", "byte_length", "xet_hash")
    }:
        raise RemotePreselectionError("published selection body identity drifted")
    if backend.stat(body_remote_path) != body_info:
        raise RemotePreselectionError("published selection body post-upload identity drifted")
    receipt_bytes = _canonical_bytes(parsed_receipt)
    with tempfile.TemporaryDirectory(
        prefix="remote-selection-receipt-", dir=selection_path.parent
    ) as tmp:
        receipt_path = Path(tmp) / f"{selection_path.name}.json"
        atomic_write_bytes(receipt_bytes, receipt_path)
        receipt_remote_path = _join_remote(
            output_prefix,
            REMOTE_PRESELECTION_RECEIPTS_DIRNAME,
            f"{selection_path.name}.json",
        )
        receipt_info = upload_verified_object(
            backend,
            local_path=receipt_path,
            remote_path=receipt_remote_path,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
            expected_bytes=receipt_bytes,
        )
        if backend.stat(receipt_remote_path) != receipt_info:
            raise RemotePreselectionError("published selection receipt identity drifted")
    return body_info, receipt_info


def publish_remote_preselection_shard(
    backend: BucketBackend,
    *,
    selection_path: Path,
    receipt: Mapping[str, object],
    output_prefix: str | NormalizedRemotePrefix,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> tuple[RemoteObjectInfo, RemoteObjectInfo]:
    """Publish one validated splice body and its schema-v1 receipt."""
    parsed = parse_remote_preselection_receipt(
        receipt, surface=f"receipt for {selection_path.name}"
    )
    return _publish_parsed_remote_preselection_shard(
        backend,
        selection_path=selection_path,
        parsed_receipt=parsed,
        output_prefix=output_prefix,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )


def publish_remote_unspliced_preselection_shard(
    backend: BucketBackend,
    *,
    selection_path: Path,
    receipt: Mapping[str, object],
    output_prefix: str | NormalizedRemotePrefix,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> tuple[RemoteObjectInfo, RemoteObjectInfo]:
    """Publish one validated ordinary body and its schema-v2 receipt."""
    parsed = parse_remote_unspliced_preselection_receipt(
        receipt, surface=f"ordinary receipt for {selection_path.name}"
    )
    return _publish_parsed_remote_preselection_shard(
        backend,
        selection_path=selection_path,
        parsed_receipt=parsed,
        output_prefix=output_prefix,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )


def _classify_remote_preselection_namespace(
    backend: BucketBackend,
    *,
    output_prefix: str | NormalizedRemotePrefix,
    expected_names: Sequence[str],
) -> tuple[
    dict[str, RemoteObjectInfo],
    dict[str, RemoteObjectInfo],
    RemoteObjectInfo | None,
]:
    """Classify direct bodies, receipt objects, and the final manifest."""
    prefix = str(NormalizedRemotePrefix.parse(output_prefix))
    expected_sequence = tuple(expected_names)
    expected = set(expected_sequence)
    if not expected or len(expected) != len(expected_sequence):
        raise RemotePreselectionError("expected selection names must be unique and non-empty")
    for name in expected_sequence:
        _strict_selection_shard_name(name, surface="expected selection name")
    listing = tuple(backend.list(prefix))
    by_path = _listing_by_path(listing, surface="remote selection namespace")
    manifest_path = _join_remote(prefix, MANIFEST_FILENAME)
    manifest_info = by_path.get(manifest_path)
    direct_bodies: dict[str, RemoteObjectInfo] = {}
    receipt_objects: dict[str, RemoteObjectInfo] = {}
    for path in by_path:
        relative = PurePosixPath(path).relative_to(PurePosixPath(prefix)).parts
        if len(relative) == 1 and relative[0] == MANIFEST_FILENAME:
            continue
        if len(relative) == 1 and relative[0].endswith(COMPRESSED_EXT):
            name = _strict_selection_shard_name(relative[0], surface="remote direct selection body")
            direct_bodies[name] = by_path[path]
            continue
        if (
            len(relative) == 2
            and relative[0] == REMOTE_PRESELECTION_RECEIPTS_DIRNAME
            and relative[1].endswith(f"{COMPRESSED_EXT}.json")
        ):
            name = _strict_selection_shard_name(
                relative[1].removesuffix(".json"),
                surface="remote selection receipt name",
            )
            receipt_objects[name] = by_path[path]
            continue
        if (
            len(relative) == 3
            and relative[0] == "_verified_upload_claims"
            and relative[1] in {*expected, MANIFEST_FILENAME}
            and relative[2].endswith(".json")
            and _SHA256_RE.fullmatch(relative[2].removesuffix(".json")) is not None
        ):
            continue
        if (
            len(relative) == 4
            and relative[0] == REMOTE_PRESELECTION_RECEIPTS_DIRNAME
            and relative[1] == "_verified_upload_claims"
            and relative[2].endswith(f"{COMPRESSED_EXT}.json")
            and relative[2].removesuffix(".json") in expected
            and relative[3].endswith(".json")
            and _SHA256_RE.fullmatch(relative[3].removesuffix(".json")) is not None
        ):
            continue
        raise RemotePreselectionError(f"unknown object in remote selection namespace: {path}")
    unexpected_bodies = sorted(set(direct_bodies) - expected)
    if unexpected_bodies:
        raise RemotePreselectionError(
            f"unexpected direct selection bodies: {unexpected_bodies[:5]}"
        )
    extra_receipts = sorted(set(receipt_objects) - expected)
    if extra_receipts:
        raise RemotePreselectionError(
            f"unexpected remote pre-selection receipts: {extra_receipts[:5]}"
        )
    return direct_bodies, receipt_objects, manifest_info


def _validate_common_discovered_receipt(
    *,
    parsed: dict[str, object],
    name: str,
    body_info: RemoteObjectInfo,
    output_prefix: str | NormalizedRemotePrefix,
    expected_run_identity: Mapping[str, object],
) -> None:
    """Validate common signed body identity and exact run-identity fields."""
    signed_body = cast(Mapping[str, object], parsed["selection_body"])
    expected_identity = RemoteObjectInfo(
        path=cast(str, signed_body["path"]),
        byte_length=cast(int, signed_body["byte_length"]),
        xet_hash=cast(str, signed_body["xet_hash"]),
    )
    expected_path = _join_remote(output_prefix, name)
    if parsed["shard_name"] != name or expected_identity.path != expected_path:
        raise RemotePreselectionError(
            f"remote pre-selection receipt contains a noncanonical body path for {name}"
        )
    if expected_identity != body_info:
        raise RemotePreselectionError(f"remote pre-selection receipt body drift for {name}")
    for field, expected in expected_run_identity.items():
        if field not in parsed or parsed[field] != expected:
            raise RemotePreselectionError(
                f"remote pre-selection receipt identity drift for {name}: {field}"
            )


def discover_remote_preselection_receipts(
    backend: BucketBackend,
    *,
    output_prefix: str | NormalizedRemotePrefix,
    expected_names: Sequence[str],
    selection_checksum: str,
    packed_inventory_aggregate: str,
    source_inventory_aggregate: str,
    packed_row_fingerprint: str,
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> tuple[dict[str, dict[str, object]], RemoteObjectInfo | None]:
    """Authenticate existing schema-v1 splice receipts and bodies."""
    direct_bodies, receipt_objects, manifest_info = _classify_remote_preselection_namespace(
        backend,
        output_prefix=output_prefix,
        expected_names=expected_names,
    )
    receipt_root = _join_remote(output_prefix, REMOTE_PRESELECTION_RECEIPTS_DIRNAME)
    receipts: dict[str, dict[str, object]] = {}
    for name in sorted(receipt_objects):
        receipt_path = _join_remote(receipt_root, f"{name}.json")
        body_info = direct_bodies.get(name)
        if body_info is None:
            raise RemotePreselectionError(
                f"receipt exists without its selection body: {receipt_path}"
            )
        _info, raw = download_remote_json_object(
            backend,
            remote_path=receipt_path,
            staging_dir=staging_dir,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
            surface=f"remote pre-selection receipt {name}",
        )
        parsed = parse_remote_preselection_receipt(
            raw, surface=f"remote pre-selection receipt {name}"
        )
        _validate_common_discovered_receipt(
            parsed=parsed,
            name=name,
            body_info=body_info,
            output_prefix=output_prefix,
            expected_run_identity={
                "selection_checksum": selection_checksum,
                "packed_inventory_aggregate": packed_inventory_aggregate,
                "source_inventory_aggregate": source_inventory_aggregate,
                "packed_row_fingerprint": packed_row_fingerprint,
            },
        )
        receipts[name] = parsed
    return receipts, manifest_info


def discover_remote_unspliced_preselection_receipts(
    backend: BucketBackend,
    *,
    output_prefix: str | NormalizedRemotePrefix,
    expected_names: Sequence[str],
    selection_checksum: str,
    packed_inventory_aggregate: str,
    packed_row_fingerprint: str,
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> tuple[
    dict[str, dict[str, object]],
    RemoteObjectInfo | None,
    tuple[str, ...],
]:
    """Authenticate ordinary receipts and enforce its contiguous-prefix state."""
    ordered_names = tuple(expected_names)
    if ordered_names != tuple(sorted(ordered_names)):
        raise RemotePreselectionError("ordinary expected selection names must be strictly sorted")
    direct_bodies, receipt_objects, manifest_info = _classify_remote_preselection_namespace(
        backend,
        output_prefix=output_prefix,
        expected_names=ordered_names,
    )
    receipt_root = _join_remote(output_prefix, REMOTE_PRESELECTION_RECEIPTS_DIRNAME)
    receipts: dict[str, dict[str, object]] = {}
    for name in sorted(receipt_objects):
        receipt_path = _join_remote(receipt_root, f"{name}.json")
        body_info = direct_bodies.get(name)
        if body_info is None:
            raise RemotePreselectionError(
                f"receipt exists without its selection body: {receipt_path}"
            )
        _info, raw = download_remote_json_object(
            backend,
            remote_path=receipt_path,
            staging_dir=staging_dir,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
            surface=f"remote ordinary pre-selection receipt {name}",
        )
        parsed = parse_remote_unspliced_preselection_receipt(
            raw, surface=f"remote ordinary pre-selection receipt {name}"
        )
        _validate_common_discovered_receipt(
            parsed=parsed,
            name=name,
            body_info=body_info,
            output_prefix=output_prefix,
            expected_run_identity={
                "selection_checksum": selection_checksum,
                "packed_inventory_aggregate": packed_inventory_aggregate,
                "packed_row_fingerprint": packed_row_fingerprint,
            },
        )
        receipts[name] = parsed

    receipt_names = tuple(name for name in ordered_names if name in receipts)
    expected_prefix = ordered_names[: len(receipt_names)]
    if receipt_names != expected_prefix:
        raise RemotePreselectionError(
            "ordinary remote pre-selection receipts must form an exact leading prefix"
        )
    body_only = tuple(
        name for name in ordered_names if name in direct_bodies and name not in receipts
    )
    first_missing = (
        ordered_names[len(receipt_names)] if len(receipt_names) < len(ordered_names) else None
    )
    if body_only and body_only != ((first_missing,) if first_missing is not None else ()):
        raise RemotePreselectionError(
            "ordinary remote pre-selection permits a body-only orphan only for the first missing shard"
        )
    return receipts, manifest_info, body_only


def publish_remote_preselection_manifest(
    backend: BucketBackend,
    *,
    manifest_path: Path,
    output_prefix: str | NormalizedRemotePrefix,
    staging_dir: str | Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> RemoteObjectInfo:
    """Publish the deterministic final manifest last and replay exact bytes."""
    payload = manifest_path.read_bytes()
    remote_path = _join_remote(output_prefix, MANIFEST_FILENAME)
    published = upload_verified_object(
        backend,
        local_path=manifest_path,
        remote_path=remote_path,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
        expected_bytes=payload,
    )
    parent = Path(staging_dir).expanduser().absolute()
    with tempfile.TemporaryDirectory(prefix="remote-manifest-replay-", dir=parent) as tmp:
        replay = Path(tmp) / MANIFEST_FILENAME
        _download(
            backend,
            info=published,
            path=replay,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        if replay.read_bytes() != payload:
            raise RemotePreselectionError(
                "published remote selection manifest differs from local bytes"
            )
    return published


__all__ = [
    "MaterializedPackedShard",
    "REMOTE_PRESELECTION_RECEIPTS_DIRNAME",
    "REMOTE_PRESELECTION_SEMANTICS",
    "REMOTE_PRESELECTION_TRUSTED_SOURCE_SEMANTICS",
    "RemoteFeatureShardMaterializer",
    "RemotePreselectionConfig",
    "RemotePreselectionError",
    "RemotePreselectionPrefixes",
    "build_remote_preselection_receipt",
    "build_remote_unspliced_preselection_receipt",
    "build_trusted_remote_preselection_source",
    "build_transport_feature_catalog",
    "discover_remote_preselection_receipts",
    "discover_remote_unspliced_preselection_receipts",
    "download_remote_json_object",
    "download_remote_object_bytes",
    "download_remote_packed_preselection_inventory",
    "local_object_info_for_remote_path",
    "materialize_remote_packed_shard",
    "parse_remote_preselection_receipt",
    "parse_remote_preselection_source",
    "parse_remote_unspliced_preselection_receipt",
    "publish_remote_preselection_manifest",
    "publish_remote_preselection_shard",
    "publish_remote_unspliced_preselection_shard",
    "resolve_remote_preselection_prefixes",
    "selection_integrity_from_receipts",
    "selection_integrity_from_unspliced_receipts",
    "validate_local_remote_packed_controls",
    "validate_local_remote_preselection_controls",
]
