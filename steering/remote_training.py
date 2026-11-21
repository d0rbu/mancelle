"""Verified HF-bucket transport for the opt-in clustered c1024 training run.

This is deliberately not a generic object-store layer. It owns the two
publication protocols required by remote training: signed body discovery for
the rank-striped worker, and complete checkpoint groups published with a marker
written last. It also exposes size/Xet-verified single-object primitives to the
bounded feature-shard splice runner. New canonical uploads are fenced by
immutable content-addressed claims because HF bucket adds do not offer
conditional create. Final packed and selection body publication is owned
exclusively by :mod:`steering.remote_body_prefetch`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, cast

import hf_xet
import torch.distributed as dist
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from huggingface_hub import HfApi
from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from steering._common.strict_json import canonical_json, strict_json_object
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.encoder_decoder.conditioning_cooldown import parse_conditioning_cooldown
from steering.conditional.encoder_decoder.fp_target_source import (
    FP_TARGET_SOURCE_INTEGRITY_KEY,
    validate_fp_target_source_certificate,
    validate_fp_target_source_certificate_remote,
)
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    parse_clustered_selection_shard_inventory,
)
from steering.conditional.splice_counterfactual import (
    PackedProvenanceInfo,
    _file_fingerprint,
    _FileSignature,
    _fingerprint_json,
    _fold_file_digests,
    validate_packed_provenance,
)
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.conditional.splice_resource_admission import validate_packed_layout
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    BaseShardContentHasher,
    glob_overlay_shards,
    load_overlay_shard,
    read_manifest,
)
from steering.refined_types import (
    AtLeastTwoInt,
    AtLeastTwoIntLike,
    LowercaseSha256,
    LowercaseSha256Like,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveFiniteNumber,
    PositiveFiniteNumberLike,
    PositiveInt,
    PositiveIntLike,
)
from steering.remote_types import (
    HfBucketId,
    HfBucketIdLike,
    NormalizedRemotePrefix,
    NormalizedRemotePrefixLike,
)
from steering.safetensors_layout import (
    SafetensorsLayoutError,
    parse_semantic_layout,
    read_compressed_safetensors_layout,
    semantic_layout_to_json,
)

if TYPE_CHECKING:
    _SemanticConfigMapping = Mapping[str, Any]
    _SelectionContracts = Mapping[str, Mapping[str, Any]]
else:
    # Keep beartype deterministic: parameterized containers use randomized
    # sampling, while these functions only own their scalar domains.
    _SemanticConfigMapping = Mapping
    _SelectionContracts = Mapping

REMOTE_LOADER_KEY = "remote_loader"
REMOTE_LOADER_SCHEMA_VERSION = 2
REMOTE_PACKED_INVENTORY_FILENAME = "remote_packed_inventory.json"
REMOTE_PACKED_INVENTORY_SCHEMA_VERSION = 2
REMOTE_PACKED_INVENTORY_SCHEMA_VERSION_V1 = 1
REMOTE_CHECKPOINT_SCHEMA_VERSION = 1
REMOTE_RUN_META_KEY = "remote_run_fingerprint"
REMOTE_SEMANTIC_META_KEY = "remote_semantic_identity"
REMOTE_SEMANTIC_SCHEMA_VERSION = 2
REMOTE_CHECKPOINT_MARKER_RE = re.compile(r"^remote_complete_(?P<step>[0-9]{6,})\.json$")
VERIFIED_UPLOAD_CLAIMS_DIRNAME = "_verified_upload_claims"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_CONFIG_KEYS = frozenset(
    {
        "bucket_id",
        "body_prefetch",
        "selection_prefix",
        "checkpoint_prefix",
        "selection_wait_timeout_seconds",
        "checkpoint_retry_timeout_seconds",
        "retry_interval_seconds",
        "resume_latest",
    }
)
_REMOTE_OPTIONAL_CONFIG_KEYS = frozenset({"async_checkpoint_upload", "train_packed"})
_REMOTE_BODY_PREFETCH_KEYS = frozenset({"lookahead_shards_per_rank", "cache_shards"})
_REMOTE_TRAIN_PACKED_KEYS = frozenset({"trust_existing_cache", "cuda_staging"})
_REMOTE_CUDA_STAGING_KEYS = frozenset({"host_queue_bytes", "pinned_slots"})
_REMOTE_PACKED_INVENTORY_V1_KEYS = frozenset(
    {
        "schema_version",
        "packed_manifest_file_fingerprint",
        "shards",
        "packed_row_fingerprint",
        "provenance_shard_count",
        "provenance_fingerprint",
        "aggregate_sha256",
    }
)
_REMOTE_PACKED_INVENTORY_V2_KEYS = _REMOTE_PACKED_INVENTORY_V1_KEYS | {"base_shard_content_hash"}
_REMOTE_PACKED_SHARD_V1_KEYS = frozenset(
    {
        "name",
        "byte_length",
        "content_sha256",
        "file_fingerprint",
        "total_segments",
    }
)
_REMOTE_PACKED_SHARD_V2_KEYS = _REMOTE_PACKED_SHARD_V1_KEYS | {"layout"}
_LOADER_KEYS = frozenset(
    {
        "schema_version",
        "selection_checksum",
        "selection_inventory_aggregate",
        "packed_row_fingerprint",
        "max_features_per_segment",
        "aggregate_sha256",
    }
)
_CLUSTERED_SPLICE_LOADER_KEYS = _LOADER_KEYS | {CLUSTER_DELTA_SEMANTICS_KEY}
_CHECKPOINT_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "step",
        "ddp_world_size",
        "remote_run_fingerprint",
        "files",
        "aggregate_sha256",
    }
)
_CHECKPOINT_FILE_KEYS = frozenset({"name", "byte_length", "content_sha256", "xet_hash"})


class RemoteTrainingError(RuntimeError):
    """Remote-training configuration, identity, or transfer failure."""


class RemoteMaterializationCancelled(RemoteTrainingError):
    """A caller-requested stop while materializing one remote cache body."""


def _raise_if_remote_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RemoteMaterializationCancelled("remote cache materialization cancelled")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_sha256(value: Any, *, surface: str) -> LowercaseSha256:
    if not isinstance(value, LowercaseSha256):
        raise RemoteTrainingError(
            f"{surface} must be 64 lowercase hexadecimal characters, got {value!r}"
        )
    return LowercaseSha256.parse(value)


def _strict_positive_number(value: Any, *, surface: str) -> PositiveFiniteNumber:
    if not isinstance(value, PositiveFiniteNumber):
        raise RemoteTrainingError(f"{surface} must be a positive number, got {value!r}")
    return PositiveFiniteNumber.parse(float(value))


def _direct_remote_prefix(value: Any, *, surface: str) -> NormalizedRemotePrefix:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RemoteTrainingError(f"{surface} must be a non-empty trimmed string")
    if not isinstance(value, NormalizedRemotePrefix):
        raise RemoteTrainingError(
            f"{surface} must be a normalized relative bucket prefix without traversal, got {value!r}"
        )
    return NormalizedRemotePrefix.parse(value)


@dataclass(frozen=True, init=False)
class RemoteBodyPrefetchConfig:
    lookahead_shards_per_rank: PositiveInt
    cache_shards: AtLeastTwoInt

    @beartype
    def __init__(
        self,
        lookahead_shards_per_rank: PositiveIntLike,
        cache_shards: AtLeastTwoIntLike = 2,
    ) -> None:
        object.__setattr__(
            self,
            "lookahead_shards_per_rank",
            PositiveInt.parse(lookahead_shards_per_rank),
        )
        object.__setattr__(self, "cache_shards", AtLeastTwoInt.parse(cache_shards))

    @classmethod
    def parse(cls, raw: Any) -> RemoteBodyPrefetchConfig:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        if (
            not isinstance(raw, Mapping)
            or "lookahead_shards_per_rank" not in keys
            or not keys.issubset(_REMOTE_BODY_PREFETCH_KEYS)
        ):
            raise RemoteTrainingError(
                "remote_training.body_prefetch must contain "
                f"{sorted(_REMOTE_BODY_PREFETCH_KEYS)}; got keys="
                f"{sorted(keys, key=str)}"
            )
        try:
            return cls(
                lookahead_shards_per_rank=raw["lookahead_shards_per_rank"],
                cache_shards=raw.get("cache_shards", 2),
            )
        except BeartypeCallHintParamViolation as exc:
            invalid_field = (
                "lookahead_shards_per_rank"
                if not isinstance(raw["lookahead_shards_per_rank"], PositiveInt)
                else "cache_shards"
            )
            raise RemoteTrainingError(
                "remote_training.body_prefetch."
                f"{invalid_field} must be a strict int "
                f"{'>= 1' if invalid_field == 'lookahead_shards_per_rank' else '>= 2'}, "
                f"got {raw.get(invalid_field)!r}"
            ) from exc


@dataclass(frozen=True, init=False)
class RemoteCudaStagingConfig:
    host_queue_bytes: PositiveInt
    pinned_slots: PositiveInt

    @beartype
    def __init__(
        self,
        host_queue_bytes: PositiveIntLike,
        pinned_slots: PositiveIntLike,
    ) -> None:
        object.__setattr__(self, "host_queue_bytes", PositiveInt.parse(host_queue_bytes))
        object.__setattr__(self, "pinned_slots", PositiveInt.parse(pinned_slots))

    @classmethod
    def parse(cls, raw: Any) -> RemoteCudaStagingConfig:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        if not isinstance(raw, Mapping) or keys != _REMOTE_CUDA_STAGING_KEYS:
            raise RemoteTrainingError(
                "remote_training.train_packed.cuda_staging must contain exactly "
                f"{sorted(_REMOTE_CUDA_STAGING_KEYS)}; got keys="
                f"{sorted(keys, key=str)}"
            )
        try:
            return cls(
                host_queue_bytes=raw["host_queue_bytes"],
                pinned_slots=raw["pinned_slots"],
            )
        except BeartypeCallHintParamViolation as exc:
            invalid_field = (
                "host_queue_bytes"
                if not isinstance(raw["host_queue_bytes"], PositiveInt)
                else "pinned_slots"
            )
            raise RemoteTrainingError(
                "remote_training.train_packed.cuda_staging."
                f"{invalid_field} must be a positive strict int, "
                f"got {raw[invalid_field]!r}"
            ) from exc


@dataclass(frozen=True, init=False)
class RemoteTrainPackedConfig:
    trust_existing_cache: bool = False
    cuda_staging: RemoteCudaStagingConfig | None = None

    @beartype
    def __init__(
        self,
        trust_existing_cache: bool = False,
        cuda_staging: RemoteCudaStagingConfig | None = None,
    ) -> None:
        object.__setattr__(self, "trust_existing_cache", trust_existing_cache)
        object.__setattr__(self, "cuda_staging", cuda_staging)

    @classmethod
    def parse(cls, raw: Any) -> RemoteTrainPackedConfig:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        if not isinstance(raw, Mapping) or not keys.issubset(_REMOTE_TRAIN_PACKED_KEYS):
            raise RemoteTrainingError(
                "remote_training.train_packed may contain "
                "trust_existing_cache/cuda_staging; got keys="
                f"{sorted(keys, key=str)}"
            )
        trust_existing_cache = raw.get("trust_existing_cache", False)
        if not isinstance(trust_existing_cache, bool):
            raise RemoteTrainingError(
                "remote_training.train_packed.trust_existing_cache must be a "
                f"strict bool, got {trust_existing_cache!r}"
            )
        if trust_existing_cache:
            logger.warning(
                "remote_training.train_packed.trust_existing_cache=true: "
                "completed HF/Xet body downloads are atomically published without "
                "owner-side content rereads, and consumers skip content hashing after "
                "a publication receipt. Preexisting packed and selection cache bodies "
                "also skip content hashing after direct-file, single-link, and signed-"
                "length checks. Trusted mode relies on transport completion and decode; "
                "use trust_existing_cache=false to hash every cache read."
            )
        return cls(
            trust_existing_cache=trust_existing_cache,
            cuda_staging=(
                RemoteCudaStagingConfig.parse(raw["cuda_staging"])
                if "cuda_staging" in raw
                else None
            ),
        )


@dataclass(frozen=True, init=False)
class RemoteTrainingConfig:
    bucket_id: HfBucketId
    body_prefetch: RemoteBodyPrefetchConfig
    selection_prefix: NormalizedRemotePrefix
    checkpoint_prefix: NormalizedRemotePrefix
    selection_wait_timeout_seconds: PositiveFiniteNumber
    checkpoint_retry_timeout_seconds: PositiveFiniteNumber
    retry_interval_seconds: PositiveFiniteNumber
    resume_latest: bool
    async_checkpoint_upload: bool = False
    train_packed: RemoteTrainPackedConfig | None = None

    @beartype
    def __init__(
        self,
        bucket_id: HfBucketIdLike,
        body_prefetch: RemoteBodyPrefetchConfig,
        selection_prefix: NormalizedRemotePrefixLike,
        checkpoint_prefix: NormalizedRemotePrefixLike,
        selection_wait_timeout_seconds: PositiveFiniteNumberLike,
        checkpoint_retry_timeout_seconds: PositiveFiniteNumberLike,
        retry_interval_seconds: PositiveFiniteNumberLike,
        resume_latest: bool,
        async_checkpoint_upload: bool = False,
        train_packed: RemoteTrainPackedConfig | None = None,
    ) -> None:
        object.__setattr__(self, "bucket_id", HfBucketId.parse(bucket_id))
        object.__setattr__(self, "body_prefetch", body_prefetch)
        object.__setattr__(
            self,
            "selection_prefix",
            NormalizedRemotePrefix.parse(selection_prefix),
        )
        object.__setattr__(
            self,
            "checkpoint_prefix",
            NormalizedRemotePrefix.parse(checkpoint_prefix),
        )
        object.__setattr__(
            self,
            "selection_wait_timeout_seconds",
            PositiveFiniteNumber.parse(float(selection_wait_timeout_seconds)),
        )
        object.__setattr__(
            self,
            "checkpoint_retry_timeout_seconds",
            PositiveFiniteNumber.parse(float(checkpoint_retry_timeout_seconds)),
        )
        object.__setattr__(
            self,
            "retry_interval_seconds",
            PositiveFiniteNumber.parse(float(retry_interval_seconds)),
        )
        object.__setattr__(self, "resume_latest", resume_latest)
        object.__setattr__(self, "async_checkpoint_upload", async_checkpoint_upload)
        object.__setattr__(self, "train_packed", train_packed)

    @classmethod
    def parse(cls, raw: Any) -> RemoteTrainingConfig | None:
        if raw is None:
            return None
        raw_keys = set(raw) if isinstance(raw, Mapping) else set()
        if isinstance(raw, Mapping) and "body_prefetch" not in raw_keys:
            raise RemoteTrainingError(
                "remote body materialization now requires "
                "remote_training.body_prefetch.lookahead_shards_per_rank; the "
                "synchronous demand downloader was removed. Add "
                "body_prefetch: {lookahead_shards_per_rank: 2}."
            )
        allowed_keys = _REMOTE_CONFIG_KEYS | _REMOTE_OPTIONAL_CONFIG_KEYS
        if (
            not isinstance(raw, Mapping)
            or not _REMOTE_CONFIG_KEYS.issubset(raw_keys)
            or not raw_keys.issubset(allowed_keys)
        ):
            keys = set(raw) if isinstance(raw, Mapping) else set()
            raise RemoteTrainingError(
                "remote_training must contain exactly "
                f"{sorted(_REMOTE_CONFIG_KEYS)} with optional "
                "async_checkpoint_upload/train_packed; "
                f"got keys={sorted(keys, key=str)}"
            )
        bucket_id = raw["bucket_id"]
        if not isinstance(bucket_id, HfBucketId):
            raise RemoteTrainingError(
                f"remote_training.bucket_id must be an owner/name bucket id, got {bucket_id!r}"
            )
        resume_latest = raw["resume_latest"]
        if not isinstance(resume_latest, bool):
            raise RemoteTrainingError(
                f"remote_training.resume_latest must be a strict bool, got {resume_latest!r}"
            )
        async_checkpoint_upload = raw.get("async_checkpoint_upload", False)
        if not isinstance(async_checkpoint_upload, bool):
            raise RemoteTrainingError(
                "remote_training.async_checkpoint_upload must be a strict bool, "
                f"got {async_checkpoint_upload!r}"
            )
        return cls(
            bucket_id=bucket_id,
            body_prefetch=RemoteBodyPrefetchConfig.parse(raw["body_prefetch"]),
            selection_prefix=_direct_remote_prefix(
                raw["selection_prefix"], surface="remote_training.selection_prefix"
            ),
            checkpoint_prefix=_direct_remote_prefix(
                raw["checkpoint_prefix"], surface="remote_training.checkpoint_prefix"
            ),
            selection_wait_timeout_seconds=_strict_positive_number(
                raw["selection_wait_timeout_seconds"],
                surface="remote_training.selection_wait_timeout_seconds",
            ),
            checkpoint_retry_timeout_seconds=_strict_positive_number(
                raw["checkpoint_retry_timeout_seconds"],
                surface="remote_training.checkpoint_retry_timeout_seconds",
            ),
            retry_interval_seconds=_strict_positive_number(
                raw["retry_interval_seconds"],
                surface="remote_training.retry_interval_seconds",
            ),
            resume_latest=resume_latest,
            async_checkpoint_upload=async_checkpoint_upload,
            train_packed=(
                RemoteTrainPackedConfig.parse(raw["train_packed"])
                if "train_packed" in raw
                else None
            ),
        )


def validate_remote_training_eligibility(
    cfg: Mapping[str, Any], remote: RemoteTrainingConfig
) -> None:
    """Reject every route other than clustered precomputed splice-CF c1024."""
    if cfg.get("use_precomputed_batches") is not True:
        raise RemoteTrainingError(
            "remote_training requires use_precomputed_batches=true; live/fused selection is forbidden"
        )
    if cfg.get("feature_clusters") is None:
        raise RemoteTrainingError("remote_training requires feature_clusters")
    feature_delta = cfg.get("feature_delta")
    if (
        not isinstance(feature_delta, Mapping)
        or feature_delta.get("mode") != "splice_counterfactual"
    ):
        raise RemoteTrainingError(
            "remote_training requires feature_delta.mode=splice_counterfactual"
        )
    if cfg.get("max_seq_len") != 1024:
        raise RemoteTrainingError("remote_training is restricted to max_seq_len=1024")
    model_tag = cfg.get("model_tag")
    if not isinstance(model_tag, str) or not model_tag.strip():
        raise RemoteTrainingError("remote_training requires an explicit non-empty model_tag")
    if remote.resume_latest and cfg.get("resume_from_step", -1) != -1:
        raise RemoteTrainingError(
            "remote_training.resume_latest=true is incompatible with an explicit resume_from_step"
        )
    if (
        remote.train_packed is not None
        and remote.train_packed.cuda_staging is not None
        and cfg.get("label_dropout") != 0.0
    ):
        raise RemoteTrainingError(
            "remote_training.train_packed.cuda_staging requires "
            "label_dropout=0.0 because host lookahead must not advance the "
            "process-global label-dropout RNG"
        )


@beartype
def build_remote_loader_certificate(
    *,
    selection_checksum: LowercaseSha256Like,
    selection_inventory_aggregate: LowercaseSha256Like,
    packed_row_fingerprint: LowercaseSha256Like,
    max_features_per_segment: NonNegativeIntLike,
    cluster_delta_semantics: str | None = None,
) -> dict[str, Any]:
    if cluster_delta_semantics is not None and (
        cluster_delta_semantics != CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
    ):
        raise RemoteTrainingError(
            "cluster_delta_semantics must be the current canonical clustered "
            f"splice value, got {cluster_delta_semantics!r}"
        )
    signed: dict[str, Any] = {
        "schema_version": REMOTE_LOADER_SCHEMA_VERSION,
        "selection_checksum": selection_checksum,
        "selection_inventory_aggregate": selection_inventory_aggregate,
        "packed_row_fingerprint": packed_row_fingerprint,
        "max_features_per_segment": max_features_per_segment,
    }
    if cluster_delta_semantics is not None:
        signed[CLUSTER_DELTA_SEMANTICS_KEY] = cluster_delta_semantics
    return {**signed, "aggregate_sha256": _sha256_json(signed)}


def parse_remote_loader_certificate(raw: Any, *, surface: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RemoteTrainingError(
            f"{surface} remote_loader must be a mapping, got {type(raw).__name__}"
        )
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version != REMOTE_LOADER_SCHEMA_VERSION:
        raise RemoteTrainingError(
            f"{surface} remote_loader.schema_version must be {REMOTE_LOADER_SCHEMA_VERSION}"
        )
    expected_keys = (
        _CLUSTERED_SPLICE_LOADER_KEYS if CLUSTER_DELTA_SEMANTICS_KEY in raw else _LOADER_KEYS
    )
    if set(raw) != expected_keys:
        raise RemoteTrainingError(
            f"{surface} remote_loader schema v{schema_version} must contain "
            f"exactly {sorted(expected_keys)}; got keys="
            f"{sorted(raw, key=str)}"
        )
    selection_checksum = _strict_sha256(
        raw["selection_checksum"], surface=f"{surface} selection_checksum"
    )
    inventory_aggregate = _strict_sha256(
        raw["selection_inventory_aggregate"],
        surface=f"{surface} selection_inventory_aggregate",
    )
    packed_fingerprint = _strict_sha256(
        raw["packed_row_fingerprint"],
        surface=f"{surface} packed_row_fingerprint",
    )
    max_f = _strict_nonnegative_int(
        raw["max_features_per_segment"],
        surface=f"{surface} max_features_per_segment",
    )
    signed: dict[str, Any] = {
        "schema_version": schema_version,
        "selection_checksum": selection_checksum,
        "selection_inventory_aggregate": inventory_aggregate,
        "packed_row_fingerprint": packed_fingerprint,
        "max_features_per_segment": max_f,
    }
    if CLUSTER_DELTA_SEMANTICS_KEY in raw:
        cluster_delta_semantics = raw[CLUSTER_DELTA_SEMANTICS_KEY]
        if cluster_delta_semantics != CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS:
            raise RemoteTrainingError(
                f"{surface} remote_loader.{CLUSTER_DELTA_SEMANTICS_KEY} must "
                "equal the current canonical clustered splice semantics, got "
                f"{cluster_delta_semantics!r}"
            )
        signed[CLUSTER_DELTA_SEMANTICS_KEY] = cluster_delta_semantics
    expected = _sha256_json(signed)
    if raw["aggregate_sha256"] != expected:
        raise RemoteTrainingError(
            f"{surface} remote_loader aggregate mismatch: manifest has "
            f"{raw['aggregate_sha256']!r}, fields hash to {expected!r}"
        )
    return {**signed, "aggregate_sha256": expected}


@dataclass(frozen=True)
class RemoteObjectInfo:
    path: str
    byte_length: int
    xet_hash: str


@dataclass(frozen=True)
class _ResolvedCheckpointGroup:
    layout: str
    aggregate_sha256: str
    objects: Mapping[str, RemoteObjectInfo]


class BucketBackend(Protocol):
    def stat(self, remote_path: str) -> RemoteObjectInfo | None: ...
    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]: ...
    def download(self, remote_path: str, local_path: Path) -> None: ...
    def upload(self, local_path: Path, remote_path: str) -> None: ...


class HfBucketBackend:
    @beartype
    def __init__(self, bucket_id: HfBucketIdLike, *, api: Any = None):
        self.bucket_id = HfBucketId.parse(bucket_id)
        self.api = api or HfApi()

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        # The resolve/HEAD endpoint's Content-Length can describe its absolute
        # Xet redirect rather than the object. Query the exact bucket record.
        matches = list(self.api.get_bucket_paths_info(self.bucket_id, [remote_path]))
        if not matches:
            return None
        if len(matches) != 1:
            raise RemoteTrainingError(
                f"bucket exact-path lookup returned {len(matches)} records for {remote_path!r}"
            )
        metadata = matches[0]
        if metadata.path != remote_path:
            raise RemoteTrainingError(
                "bucket exact-path lookup returned the wrong path: "
                f"requested={remote_path!r}, returned={metadata.path!r}"
            )
        return RemoteObjectInfo(
            path=str(metadata.path),
            byte_length=int(metadata.size),
            xet_hash=str(metadata.xet_hash),
        )

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        out: list[RemoteObjectInfo] = []
        for item in self.api.list_bucket_tree(self.bucket_id, prefix=prefix, recursive=True):
            if getattr(item, "type", None) == "file":
                file_item = cast(Any, item)
                path = str(file_item.path)
                if path != prefix and not path.startswith(f"{prefix}/"):
                    continue
                out.append(
                    RemoteObjectInfo(
                        path=path,
                        byte_length=int(file_item.size),
                        xet_hash=str(file_item.xet_hash),
                    )
                )
        return sorted(out, key=lambda entry: entry.path)

    def download(self, remote_path: str, local_path: Path) -> None:
        self.api.download_bucket_files(
            self.bucket_id,
            [(remote_path, local_path)],
            raise_on_missing_files=True,
        )

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.api.batch_bucket_files(
            self.bucket_id,
            add=[(local_path, remote_path)],
        )


_STABLE_IDENTITY_FIELDS = (
    "dev",
    "ino",
    "mode",
    "size",
    "mtime_ns",
    "ctime_ns",
    "nlink",
)
# Timestamps drift on Lustre as attributes settle, and a single lstat/fstat of
# a recently written file can transiently report a stale (smaller) st_size —
# so on paths whose content is verified by FULL READS (download tmp files,
# staged bodies) these fields are advisory: tolerated with a warning, and
# forcing the strict re-verification pass where one exists. dev/ino/mode/nlink
# drift (a file swap) is always fatal.
_ADVISORY_IDENTITY_FIELDS = frozenset({"size", "mtime_ns", "ctime_ns"})
type _StableIdentity = tuple[int, int, int, int, int, int, int]


def _stat_identity(observed: os.stat_result) -> _StableIdentity:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
        observed.st_nlink,
    )


def _identity_drift(before: _StableIdentity, after: _StableIdentity) -> list[str]:
    """Return the stable-identity fields whose values changed."""
    return [
        name
        for name, expected, observed in zip(
            _STABLE_IDENTITY_FIELDS,
            before,
            after,
            strict=True,
        )
        if expected != observed
    ]


def _check_identity_drift(
    before: _StableIdentity,
    after: _StableIdentity,
    *,
    path: Path,
    message: str,
    allow_advisory_drift: bool,
) -> None:
    drifted = _identity_drift(before, after)
    if not drifted:
        return
    details = ", ".join(
        f"{name}(expected={before[index]}, observed={after[index]})"
        for index, name in enumerate(_STABLE_IDENTITY_FIELDS)
        if name in drifted
    )
    if allow_advisory_drift and set(drifted) <= _ADVISORY_IDENTITY_FIELDS:
        logger.warning(
            f"artifact identity drift tolerated: {path}; "
            f"drifted_fields={','.join(drifted)}; {details}"
        )
        return
    raise RemoteTrainingError(f"{message}: {path}; drifted_fields={','.join(drifted)}; {details}")


def _stable_file_identity(
    path: Path,
    *,
    allow_advisory_drift: bool = False,
    need_sha256: bool = True,
    phase_timings: dict[str, float] | None = None,
) -> tuple[int, int, str, str]:
    """Return size, mode, optional SHA-256, and Xet hash for one stable direct file."""
    for pass_index in range(2):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise RemoteTrainingError(f"required artifact must be a direct regular file: {path}")
        opened_identity = _stat_identity(before)
        if phase_timings is not None:
            phase_timings.setdefault("sha256_seconds", 0.0)
        sha256_started = time.perf_counter() if phase_timings is not None and need_sha256 else 0.0
        digest = hashlib.sha256() if need_sha256 else None
        hashed = 0
        retry_required = False
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            byte_length = int(opened.st_size)
            opened_observed = _stat_identity(opened)
            opened_drift = _identity_drift(opened_identity, opened_observed)
            _check_identity_drift(
                opened_identity,
                opened_observed,
                path=path,
                message="artifact changed before hashing",
                allow_advisory_drift=allow_advisory_drift,
            )
            if opened_drift:
                opened_identity = opened_observed
            if digest is not None:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    hashed += len(chunk)
                if hashed != byte_length:
                    raise RemoteTrainingError(
                        f"artifact byte count changed while hashing {path}: "
                        f"expected={byte_length}, hashed={hashed}"
                    )
            after_open = os.fstat(handle.fileno())
            after_open_identity = _stat_identity(after_open)
            after_open_drift = _identity_drift(opened_identity, after_open_identity)
            _check_identity_drift(
                opened_identity,
                after_open_identity,
                path=path,
                message="artifact changed while hashing",
                allow_advisory_drift=allow_advisory_drift and pass_index == 0,
            )
            retry_required = retry_required or bool(after_open_drift)
        if phase_timings is not None and need_sha256:
            phase_timings["sha256_seconds"] = phase_timings.get("sha256_seconds", 0.0) + (
                time.perf_counter() - sha256_started
            )
        after = path.lstat()
        after_identity = _stat_identity(after)
        after_drift = _identity_drift(opened_identity, after_identity)
        _check_identity_drift(
            opened_identity,
            after_identity,
            path=path,
            message="artifact path changed while hashing",
            allow_advisory_drift=allow_advisory_drift and pass_index == 0,
        )
        retry_required = retry_required or bool(after_drift)
        xet_started = time.perf_counter() if phase_timings is not None else 0.0
        hash_files = cast(Any, hf_xet).hash_files
        xet_result = hash_files([str(path)])[0]
        if phase_timings is not None:
            phase_timings["xet_seconds"] = phase_timings.get("xet_seconds", 0.0) + (
                time.perf_counter() - xet_started
            )
        if int(xet_result.file_size) != byte_length:
            raise RemoteTrainingError(f"Xet hash size disagrees with local file size for {path}")
        final = path.lstat()
        final_identity = _stat_identity(final)
        final_drift = _identity_drift(opened_identity, final_identity)
        _check_identity_drift(
            opened_identity,
            final_identity,
            path=path,
            message="artifact changed while computing its Xet hash",
            allow_advisory_drift=allow_advisory_drift and pass_index == 0,
        )
        retry_required = retry_required or bool(final_drift)
        if retry_required:
            logger.warning(
                f"artifact identity drift requires content re-verification: {path}; reverify_pass=2"
            )
            continue
        sha256 = digest.hexdigest() if digest is not None else ""
        return byte_length, opened_identity[2], sha256, str(xet_result.hash)
    raise AssertionError("stable file verification loop exhausted without returning")


def _verify_file(
    path: Path,
    *,
    byte_length: int,
    content_sha256: str,
    xet_hash: str | None = None,
) -> str:
    actual_size, _, actual_sha, actual_xet = _stable_file_identity(path)
    if actual_size != byte_length or actual_sha != content_sha256:
        raise RemoteTrainingError(
            f"integrity mismatch for {path}: expected size/sha256 "
            f"{byte_length}/{content_sha256}, got {actual_size}/{actual_sha}"
        )
    if xet_hash is not None and actual_xet != xet_hash:
        raise RemoteTrainingError(
            f"Xet identity mismatch for {path}: expected {xet_hash}, got {actual_xet}"
        )
    return actual_xet


def _open_cache_lock(lock_path: Path) -> int:
    """Open one persistent cache lock and prove the path names that exact inode."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemoteTrainingError("remote cache locking requires os.O_NOFOLLOW")
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RemoteTrainingError(f"cannot open remote cache lock {lock_path}: {exc}") from exc
    try:
        try:
            opened = os.fstat(fd)
            linked = lock_path.lstat()
        except OSError as exc:
            raise RemoteTrainingError(
                f"cannot validate remote cache lock {lock_path}: {exc}"
            ) from exc
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise RemoteTrainingError(
                f"remote cache lock must be a direct regular file: {lock_path}"
            )
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise RemoteTrainingError(f"remote cache lock path changed while opening: {lock_path}")
        if opened.st_nlink != 1 or linked.st_nlink != 1:
            raise RemoteTrainingError(
                f"remote cache lock must have exactly one hard link: {lock_path}"
            )
        return fd
    except BaseException:
        os.close(fd)
        raise


def _cache_body_fd_signature(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_mode,
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _cache_body_descriptor(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "link_count": info.st_nlink,
    }


def _cache_body_receipt_identity(descriptor: Mapping[str, int]) -> tuple[int, ...]:
    """Return receipt fields that must agree across shared-filesystem clients."""
    return (
        descriptor["inode"],
        descriptor["mode"],
        descriptor["size"],
        descriptor["link_count"],
    )


def _validate_expected_cache_descriptor(
    value: Mapping[str, Any],
    *,
    surface: str,
) -> dict[str, int]:
    expected_keys = {
        "device",
        "inode",
        "mode",
        "size",
        "mtime_ns",
        "ctime_ns",
        "link_count",
    }
    if set(value) != expected_keys or any(type(value[key]) is not int for key in value):
        raise RemoteTrainingError(f"{surface} descriptor receipt is malformed")
    return {key: cast(int, value[key]) for key in expected_keys}


def _capture_cache_body_descriptor(path: Path) -> dict[str, int]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemoteTrainingError("remote cache body reads require os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RemoteTrainingError(f"cannot capture remote cache body {path}: {exc}") from exc
    try:
        opened = os.fstat(fd)
        linked = path.lstat()
        if not stat.S_ISREG(opened.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise RemoteTrainingError(f"remote cache body must be a direct regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise RemoteTrainingError(
                f"remote cache body path changed while capturing descriptor: {path}"
            )
        if opened.st_nlink != 1 or linked.st_nlink != 1:
            raise RemoteTrainingError(f"remote cache body must have exactly one hard link: {path}")
        return _cache_body_descriptor(opened)
    finally:
        os.close(fd)


def _verify_cache_body_payload(
    data: bytes,
    *,
    path: Path,
    entry: Mapping[str, Any],
) -> None:
    """Verify the exact immutable bytes that will be handed to the decoder."""
    actual_size = len(data)
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_size != entry["byte_length"] or actual_sha256 != entry["content_sha256"]:
        raise RemoteTrainingError(
            f"integrity mismatch for {path}: expected size/sha256 "
            f"{entry['byte_length']}/{entry['content_sha256']}, got "
            f"{actual_size}/{actual_sha256}"
        )
    if "file_fingerprint" not in entry:
        return
    fingerprint = hashlib.sha256()
    fingerprint.update(path.name.encode("utf-8"))
    fingerprint.update(b"\0")
    fingerprint.update(data)
    actual_fingerprint = fingerprint.hexdigest()
    if actual_fingerprint != entry["file_fingerprint"]:
        raise RemoteTrainingError(
            f"remote packed body integrity mismatch for {path}: expected "
            "size/content_sha256/file_fingerprint "
            f"{entry['byte_length']}/{entry['content_sha256']}/"
            f"{entry['file_fingerprint']}, got {actual_size}/{actual_sha256}/"
            f"{actual_fingerprint}"
        )


def _read_cache_body_bytes(
    path: Path,
    *,
    entry: Mapping[str, Any],
    verify_content: bool,
    expected_descriptor: Mapping[str, Any] | None = None,
) -> bytes:
    """Read and optionally hash one finalized body without reopening its path."""
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemoteTrainingError("remote cache body reads require os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise RemoteTrainingError(f"cannot open remote cache body {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        validated_expected = (
            _validate_expected_cache_descriptor(
                expected_descriptor,
                surface=f"remote cache body {path}",
            )
            if expected_descriptor is not None
            else None
        )
        if not stat.S_ISREG(before.st_mode):
            raise RemoteTrainingError(f"remote cache body must be a direct regular file: {path}")
        if before.st_nlink > 1:
            raise RemoteTrainingError(
                f"remote cache body must not have multiple hard links: {path}"
            )
        if before.st_size != entry["byte_length"]:
            raise RemoteTrainingError(
                f"remote cache body integrity mismatch for {path}: expected size "
                f"{entry['byte_length']}, got {before.st_size}"
            )
        before_descriptor = _cache_body_descriptor(before)
        receipt_metadata_drift = False
        if validated_expected is not None and before_descriptor != validated_expected:
            if _cache_body_receipt_identity(before_descriptor) != _cache_body_receipt_identity(
                validated_expected
            ):
                raise RemoteTrainingError(
                    f"fresh remote cache body descriptor does not match its "
                    f"verification receipt: {path}"
                )
            receipt_metadata_drift = True
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read()
        after = os.fstat(fd)
        after_descriptor = _cache_body_descriptor(after)
        if validated_expected is not None and after_descriptor != validated_expected:
            if _cache_body_receipt_identity(after_descriptor) != _cache_body_receipt_identity(
                validated_expected
            ):
                raise RemoteTrainingError(
                    f"fresh remote cache body descriptor changed after its verification receipt: {path}"
                )
            receipt_metadata_drift = True
        if after.st_nlink > 1:
            raise RemoteTrainingError(
                f"remote cache body gained multiple hard links while reading: {path}"
            )
        if after.st_size != entry["byte_length"] or len(data) != entry["byte_length"]:
            raise RemoteTrainingError(
                f"remote cache body integrity mismatch while reading {path}: expected "
                f"{entry['byte_length']}, descriptor={after.st_size}, bytes={len(data)}"
            )
        before_signature = _cache_body_fd_signature(before)
        after_signature = _cache_body_fd_signature(after)
        transitioned = (
            before_signature != after_signature
            or before.st_nlink != 1
            or after.st_nlink != 1
            or receipt_metadata_drift
        )
        if verify_content or transitioned:
            _verify_cache_body_payload(data, path=path, entry=entry)
        if transitioned:
            logger.info(f"remote_training.cache_body_transition_verified path={path}")
        return data
    except RemoteTrainingError:
        raise
    except OSError as exc:
        raise RemoteTrainingError(f"cannot read remote cache body {path}: {exc}") from exc
    finally:
        os.close(fd)


def _check_trusted_cache_body(path: Path, *, byte_length: int) -> None:
    """Cheap, intentionally non-cryptographic checks for an opt-in trusted cache."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RemoteTrainingError(f"trusted cache body must be a direct regular file: {path}")
    if info.st_nlink != 1:
        raise RemoteTrainingError(f"trusted cache body must have exactly one hard link: {path}")
    if info.st_size != byte_length:
        raise RemoteTrainingError(
            f"trusted cache body size mismatch for {path}: expected "
            f"{byte_length}, got {info.st_size}"
        )


_PACKED_DIGEST_CHUNK_BYTES = 16 * 1024 * 1024


def _packed_file_signature(path: Path, *, surface: str) -> _FileSignature:
    try:
        return _FileSignature.capture(path)
    except SpliceProvenanceError as exc:
        raise RemoteTrainingError(f"{surface}: {exc}") from exc


def _packed_file_fingerprint(path: Path, *, surface: str) -> str:
    try:
        return _file_fingerprint(path)
    except SpliceProvenanceError as exc:
        raise RemoteTrainingError(f"{surface}: {exc}") from exc


def _dual_digest(
    path: Path,
    *,
    fingerprint_name: str | None = None,
    base_hasher: BaseShardContentHasher | None = None,
) -> tuple[int, str, str]:
    """Return size, plain SHA-256, and name-salted file fingerprint in one pass."""
    path = Path(path)
    before = _packed_file_signature(path, surface="cannot fingerprint remote packed artifact")
    if not stat.S_ISREG(before.lstat_fields[0]):
        raise RemoteTrainingError(f"remote packed artifact must be a direct regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RemoteTrainingError(f"cannot open remote packed artifact {path}: {exc}") from exc
    content = hashlib.sha256()
    file_fingerprint = hashlib.sha256()
    file_fingerprint.update(
        (fingerprint_name if fingerprint_name is not None else path.name).encode("utf-8")
    )
    file_fingerprint.update(b"\0")
    if base_hasher is not None:
        base_hasher.begin_shard(path.name)
    byte_length = 0
    try:
        opened_fields = _FileSignature._fields(os.fstat(fd))
        if opened_fields != before.stat_fields:
            raise RemoteTrainingError(f"remote packed artifact changed before hashing: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            offset = 0
            while True:
                block = handle.read(_PACKED_DIGEST_CHUNK_BYTES)
                if not block:
                    break
                byte_length += len(block)
                content.update(block)
                file_fingerprint.update(block)
                if base_hasher is not None:
                    base_hasher.update_shard_bytes(block)
                try:
                    os.posix_fadvise(
                        fd,
                        offset,
                        len(block),
                        os.POSIX_FADV_DONTNEED,
                    )
                except (OSError, AttributeError):
                    pass
                offset += len(block)
        if base_hasher is not None:
            base_hasher.end_shard()
        if _FileSignature._fields(os.fstat(fd)) != before.stat_fields:
            raise RemoteTrainingError(f"remote packed artifact changed while hashing: {path}")
    finally:
        os.close(fd)
    after = _packed_file_signature(
        path,
        surface="cannot re-check remote packed artifact after hashing",
    )
    if after != before:
        raise RemoteTrainingError(f"remote packed artifact path changed while hashing: {path}")
    return byte_length, content.hexdigest(), file_fingerprint.hexdigest()


def _strict_nonnegative_int(value: Any, *, surface: str) -> NonNegativeInt:
    if not isinstance(value, NonNegativeInt):
        raise RemoteTrainingError(f"{surface} must be a non-negative strict int, got {value!r}")
    return NonNegativeInt.parse(value)


def parse_remote_packed_inventory(raw: Any, *, surface: str) -> dict[str, Any]:
    """Strictly parse and independently authenticate a schema-v1 or v2 inventory."""
    if not isinstance(raw, Mapping):
        raise RemoteTrainingError(f"{surface} must be a mapping")
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version not in {
        REMOTE_PACKED_INVENTORY_SCHEMA_VERSION_V1,
        REMOTE_PACKED_INVENTORY_SCHEMA_VERSION,
    }:
        raise RemoteTrainingError(
            f"{surface}.schema_version must be one of "
            f"[{REMOTE_PACKED_INVENTORY_SCHEMA_VERSION_V1}, "
            f"{REMOTE_PACKED_INVENTORY_SCHEMA_VERSION}], got {schema_version!r}"
        )
    inventory_keys = (
        _REMOTE_PACKED_INVENTORY_V1_KEYS
        if schema_version == REMOTE_PACKED_INVENTORY_SCHEMA_VERSION_V1
        else _REMOTE_PACKED_INVENTORY_V2_KEYS
    )
    shard_keys = (
        _REMOTE_PACKED_SHARD_V1_KEYS
        if schema_version == REMOTE_PACKED_INVENTORY_SCHEMA_VERSION_V1
        else _REMOTE_PACKED_SHARD_V2_KEYS
    )
    if set(raw) != inventory_keys:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise RemoteTrainingError(
            f"{surface} schema v{schema_version} must contain exactly "
            f"{sorted(inventory_keys)}; "
            f"got keys={sorted(keys, key=str)}"
        )
    manifest_fingerprint = _strict_sha256(
        raw["packed_manifest_file_fingerprint"],
        surface=f"{surface}.packed_manifest_file_fingerprint",
    )
    raw_shards = raw["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise RemoteTrainingError(f"{surface}.shards must be a non-empty list")
    shards: list[dict[str, Any]] = []
    previous_name: str | None = None
    file_fingerprints: list[str] = []
    for index, raw_entry in enumerate(raw_shards):
        entry_surface = f"{surface}.shards[{index}]"
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != shard_keys:
            keys = set(raw_entry) if isinstance(raw_entry, Mapping) else set()
            raise RemoteTrainingError(
                f"{entry_surface} must contain exactly "
                f"{sorted(shard_keys)}; got keys="
                f"{sorted(keys, key=str)}"
            )
        entry_mapping = cast(Mapping[str, Any], raw_entry)
        name = entry_mapping["name"]
        if (
            not isinstance(name, str)
            or not name
            or name != Path(name).name
            or "\\" in name
            or not name.endswith(".safetensors.zst")
        ):
            raise RemoteTrainingError(
                f"{entry_surface}.name must be a direct filename ending "
                f"'.safetensors.zst', got {name!r}"
            )
        if previous_name is not None and name <= previous_name:
            raise RemoteTrainingError(
                f"{surface}.shards names must be unique and strictly name-sorted; "
                f"entry {index} is {name!r} after {previous_name!r}"
            )
        previous_name = name
        content_sha256 = _strict_sha256(
            entry_mapping["content_sha256"],
            surface=f"{entry_surface}.content_sha256",
        )
        file_fingerprint = _strict_sha256(
            entry_mapping["file_fingerprint"],
            surface=f"{entry_surface}.file_fingerprint",
        )
        file_fingerprints.append(file_fingerprint)
        byte_length = _strict_nonnegative_int(
            entry_mapping["byte_length"],
            surface=f"{entry_surface}.byte_length",
        )
        total_segments = _strict_nonnegative_int(
            entry_mapping["total_segments"],
            surface=f"{entry_surface}.total_segments",
        )
        parsed_entry: dict[str, Any] = {
            "name": name,
            "byte_length": byte_length,
            "content_sha256": content_sha256,
            "file_fingerprint": file_fingerprint,
            "total_segments": total_segments,
        }
        if schema_version == REMOTE_PACKED_INVENTORY_SCHEMA_VERSION:
            try:
                layout = parse_semantic_layout(
                    entry_mapping["layout"], surface=f"{entry_surface}.layout"
                )
                layout_total_segments, _layout_total_features = validate_packed_layout(layout)
            except SafetensorsLayoutError as exc:
                raise RemoteTrainingError(str(exc)) from exc
            if layout.compressed_bytes != byte_length:
                raise RemoteTrainingError(
                    f"{entry_surface}.layout.compressed_bytes does not match byte_length"
                )
            if layout_total_segments != total_segments:
                raise RemoteTrainingError(
                    f"{entry_surface}.layout segment count {layout_total_segments} "
                    f"does not match total_segments={total_segments}"
                )
            parsed_entry["layout"] = semantic_layout_to_json(layout)
        shards.append(parsed_entry)
    packed_row_fingerprint_value = _strict_sha256(
        raw["packed_row_fingerprint"],
        surface=f"{surface}.packed_row_fingerprint",
    )
    expected_fold = _fold_file_digests(file_fingerprints)
    if packed_row_fingerprint_value != expected_fold:
        raise RemoteTrainingError(
            f"{surface}.packed_row_fingerprint does not match the name-sorted "
            f"fold of shards[*].file_fingerprint: inventory has "
            f"{packed_row_fingerprint_value}, fields fold to {expected_fold}"
        )
    provenance_shard_count = _strict_nonnegative_int(
        raw["provenance_shard_count"],
        surface=f"{surface}.provenance_shard_count",
    )
    provenance_fingerprint = _strict_sha256(
        raw["provenance_fingerprint"],
        surface=f"{surface}.provenance_fingerprint",
    )
    signed = {
        "schema_version": schema_version,
        "packed_manifest_file_fingerprint": manifest_fingerprint,
        "shards": shards,
        "packed_row_fingerprint": packed_row_fingerprint_value,
        "provenance_shard_count": provenance_shard_count,
        "provenance_fingerprint": provenance_fingerprint,
    }
    if schema_version == REMOTE_PACKED_INVENTORY_SCHEMA_VERSION:
        signed["base_shard_content_hash"] = _strict_sha256(
            raw["base_shard_content_hash"],
            surface=f"{surface}.base_shard_content_hash",
        )
    aggregate_sha256 = _strict_sha256(
        raw["aggregate_sha256"],
        surface=f"{surface}.aggregate_sha256",
    )
    expected_aggregate = _sha256_json(signed)
    if aggregate_sha256 != expected_aggregate:
        raise RemoteTrainingError(
            f"{surface}.aggregate_sha256 mismatch: inventory has "
            f"{aggregate_sha256}, fields hash to {expected_aggregate}"
        )
    return {**signed, "aggregate_sha256": expected_aggregate}


def _fingerprint_files_quiescent(paths: list[Path], *, surface: str) -> str:
    ordered = sorted((Path(path) for path in paths), key=lambda path: path.name)
    if not ordered:
        return hashlib.sha256().hexdigest()
    if len({path.name for path in ordered}) != len(ordered):
        raise RemoteTrainingError(f"{surface} contains duplicate direct filenames")
    before = [
        _packed_file_signature(path, surface=f"cannot fingerprint {surface}") for path in ordered
    ]
    workers = min(len(ordered), os.cpu_count() or 1)
    if workers <= 1:
        digests = [
            _packed_file_fingerprint(path, surface=f"cannot fingerprint {surface}")
            for path in ordered
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            digests = list(
                pool.map(
                    lambda path: _packed_file_fingerprint(
                        path,
                        surface=f"cannot fingerprint {surface}",
                    ),
                    ordered,
                )
            )
    after = [_packed_file_signature(path, surface=f"cannot re-check {surface}") for path in ordered]
    if after != before:
        changed = [
            prior.name for prior, current in zip(before, after, strict=True) if prior != current
        ]
        raise RemoteTrainingError(
            f"{surface} changed while computing its fingerprint: {changed[:5]}; "
            "retry from quiescent immutable artifacts"
        )
    return _fold_file_digests(digests)


def _load_json_object(path: Path, *, surface: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RemoteTrainingError(f"{surface} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise RemoteTrainingError(f"{surface} must be a JSON object: {path}")
    return raw


def _resolve_inventory_selection_manifest(
    packed_row_dir: Path,
    *,
    selection_checksum: str | None,
) -> tuple[Path, dict[str, Any]]:
    sel_root = packed_row_dir / "sel"
    if selection_checksum is not None:
        checksum = _strict_sha256(
            selection_checksum,
            surface="selection_checksum override",
        )
        manifest_path = sel_root / checksum / "manifest.json"
        if not manifest_path.is_file():
            raise RemoteTrainingError(
                "selection_checksum override does not identify a selection "
                f"manifest: {manifest_path}"
            )
        manifest = _load_json_object(
            manifest_path,
            surface="selected remote-loader manifest",
        )
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping) or REMOTE_LOADER_KEY not in metadata:
            raise RemoteTrainingError(
                f"selection_checksum={checksum} manifest has no {REMOTE_LOADER_KEY} "
                f"certificate: {manifest_path}"
            )
        return manifest_path, manifest

    candidates: list[tuple[Path, dict[str, Any]]] = []
    if sel_root.is_dir():
        for manifest_path in sorted(sel_root.glob("*/manifest.json")):
            manifest = _load_json_object(
                manifest_path,
                surface="candidate selection manifest",
            )
            metadata = manifest.get("metadata")
            if isinstance(metadata, Mapping) and REMOTE_LOADER_KEY in metadata:
                candidates.append((manifest_path, manifest))
    if len(candidates) != 1:
        candidate_names = [path.parent.name for path, _manifest in candidates]
        raise RemoteTrainingError(
            "remote packed inventory publication requires exactly one sel/*/"
            f"manifest.json carrying {REMOTE_LOADER_KEY}; found "
            f"{len(candidates)} candidates={candidate_names!r}. Pass an explicit "
            "selection_checksum=<64-hex> override."
        )
    return candidates[0]


def _packed_family_requires_fp_certificate(
    packed_row_dir: Path,
    metadata: Mapping[str, Any],
) -> bool:
    return metadata.get("context_len") == 1024 or "c1024" in packed_row_dir.parts


def packed_body_digest(
    path: Path,
    *,
    base_hasher: BaseShardContentHasher | None = None,
) -> tuple[int, str, str, dict[str, Any]]:
    """One packed body's inventory facts: sizes, both digests, and its layout.

    Shared by the local whole-family builder and the streaming one, so a family
    signed from the bucket cannot drift from the same family signed on disk.
    """
    layout = read_compressed_safetensors_layout(path)
    validate_packed_layout(layout)
    byte_length, content_sha256, file_fingerprint = _dual_digest(path, base_hasher=base_hasher)
    return byte_length, content_sha256, file_fingerprint, semantic_layout_to_json(layout)


def packed_inventory_entry(
    *,
    name: str,
    digest: tuple[int, str, str, dict[str, Any]],
    total_segments: int,
) -> dict[str, Any]:
    """One signed-inventory shard entry, in the one shape both builders emit."""
    byte_length, content_sha256, file_fingerprint, layout = digest
    return {
        "name": name,
        "byte_length": byte_length,
        "content_sha256": content_sha256,
        "file_fingerprint": file_fingerprint,
        "total_segments": total_segments,
        "layout": layout,
    }


def packed_provenance_total_segments(provenance_path: Path) -> int:
    """The inventory's ``total_segments`` for one provenance sidecar."""
    provenance = load_overlay_shard(provenance_path)
    try:
        base_idx = provenance[PROVENANCE_BASE_SHARD_IDX_KEY]
        doc_idx = provenance[PROVENANCE_DOC_IDX_KEY]
        seg_ordinal = provenance[PROVENANCE_SEG_ORDINAL_KEY]
    except KeyError as exc:
        raise RemoteTrainingError(
            f"{provenance_path}: missing provenance tensor key {exc.args[0]!r}"
        ) from exc
    if not (len(base_idx) == len(doc_idx) == len(seg_ordinal)):
        raise RemoteTrainingError(
            f"{provenance_path}: provenance tensor lengths disagree after "
            "full packed-provenance validation"
        )
    return len(base_idx)


def sign_remote_packed_inventory(
    *,
    packed_manifest_file_fingerprint: str,
    shards: list[dict[str, Any]],
    packed_row_fingerprint: str,
    provenance_shard_count: int,
    provenance_fingerprint: str,
    base_shard_content_hash: str,
    surface: str,
) -> dict[str, Any]:
    """Seal the signed envelope both inventory builders publish."""
    signed = {
        "schema_version": REMOTE_PACKED_INVENTORY_SCHEMA_VERSION,
        "packed_manifest_file_fingerprint": packed_manifest_file_fingerprint,
        "shards": shards,
        "packed_row_fingerprint": packed_row_fingerprint,
        "provenance_shard_count": provenance_shard_count,
        "provenance_fingerprint": provenance_fingerprint,
        "base_shard_content_hash": base_shard_content_hash,
    }
    return parse_remote_packed_inventory(
        {**signed, "aggregate_sha256": _sha256_json(signed)},
        surface=surface,
    )


def build_remote_packed_inventory(
    packed_row_dir: str | Path,
    *,
    selection_checksum: str | None = None,
) -> dict[str, Any]:
    """Build and cross-check the signed inventory for one complete packed family."""
    packed_dir = Path(packed_row_dir)
    manifest_path = packed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RemoteTrainingError(
            f"packed-row manifest missing for remote inventory publication: {manifest_path}"
        )
    manifest_signature = _packed_file_signature(
        manifest_path,
        surface="cannot capture packed-row manifest identity",
    )
    packed_info = validate_packed_provenance(
        packed_dir,
        cache_dir=packed_dir,
    )
    manifest = read_manifest(packed_dir)
    if manifest is None:
        raise RemoteTrainingError(
            f"packed-row manifest is invalid for remote inventory publication: {manifest_path}"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RemoteTrainingError(
            f"packed-row manifest metadata must be a mapping: {manifest_path}"
        )

    packed_paths = packed_info.packed_shard_paths
    packed_signatures = [
        _packed_file_signature(path, surface="cannot capture packed-row body identity")
        for path in packed_paths
    ]
    base_hasher = BaseShardContentHasher()
    body_digests = [packed_body_digest(path, base_hasher=base_hasher) for path in packed_paths]
    packed_signatures_after = [
        _packed_file_signature(
            path,
            surface="cannot re-check packed-row body identity",
        )
        for path in packed_paths
    ]
    if packed_signatures_after != packed_signatures:
        changed = [
            prior.name
            for prior, current in zip(
                packed_signatures,
                packed_signatures_after,
                strict=True,
            )
            if prior != current
        ]
        raise RemoteTrainingError(
            "packed-row body changed while building remote inventory: "
            f"{changed[:5]}; retry from quiescent immutable artifacts"
        )

    provenance_by_name = {path.name: path for path in packed_info.provenance_shard_paths}
    total_segments_by_name: dict[str, int] = {}
    for packed_path in packed_paths:
        provenance_path = provenance_by_name.get(packed_path.name)
        if provenance_path is None:
            raise RemoteTrainingError(f"packed provenance shard missing for {packed_path.name}")
        total_segments_by_name[packed_path.name] = packed_provenance_total_segments(provenance_path)

    entries: list[dict[str, Any]] = [
        packed_inventory_entry(
            name=packed_path.name,
            digest=digest,
            total_segments=total_segments_by_name[packed_path.name],
        )
        for packed_path, digest in zip(packed_paths, body_digests, strict=True)
    ]
    packed_fold = _fold_file_digests([entry["file_fingerprint"] for entry in entries])
    if packed_fold != packed_info.packed_row_fingerprint:
        raise RemoteTrainingError(
            "new remote packed inventory file-fingerprint fold disagrees with "
            "validate_packed_provenance: "
            f"{packed_fold} != {packed_info.packed_row_fingerprint}"
        )
    provenance_fingerprint_value = _fingerprint_files_quiescent(
        packed_info.provenance_shard_paths,
        surface="packed provenance",
    )
    if provenance_fingerprint_value != packed_info.provenance_fingerprint or len(
        packed_info.provenance_shard_paths
    ) != len(entries):
        raise RemoteTrainingError(
            "new remote packed inventory provenance identity disagrees with "
            "validate_packed_provenance"
        )

    selection_manifest_path: Path | None = None
    selection_signature: _FileSignature | None = None
    if selection_checksum is not None:
        selection_manifest_path, selection_manifest = _resolve_inventory_selection_manifest(
            packed_dir,
            selection_checksum=selection_checksum,
        )
        selection_signature = _packed_file_signature(
            selection_manifest_path,
            surface="cannot capture selection manifest identity",
        )
        selection_metadata = selection_manifest.get("metadata")
        if not isinstance(selection_metadata, Mapping):
            raise RemoteTrainingError(
                f"selection manifest metadata must be a mapping: {selection_manifest_path}"
            )
        loader = parse_remote_loader_certificate(
            selection_metadata.get(REMOTE_LOADER_KEY),
            surface=f"selection manifest {selection_manifest_path}",
        )
        manifest_checksum = selection_manifest.get("checksum")
        if loader["selection_checksum"] != manifest_checksum:
            raise RemoteTrainingError(
                f"selection manifest {selection_manifest_path} remote_loader selection checksum drift"
            )
        expected_packed_provenance = {
            "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
            "packed_row_fingerprint": packed_info.packed_row_fingerprint,
            "provenance_shard_count": len(packed_info.provenance_shard_paths),
            "provenance_fingerprint": packed_info.provenance_fingerprint,
        }
        if selection_metadata.get("packed_provenance") != expected_packed_provenance:
            raise RemoteTrainingError(
                f"selection manifest {selection_manifest_path} packed_provenance "
                "does not exactly match the validated packed family"
            )
        if loader["packed_row_fingerprint"] != packed_fold:
            raise RemoteTrainingError(
                f"selection manifest {selection_manifest_path} remote_loader "
                "packed-row fingerprint drift"
            )

    has_fp_certificate = FP_TARGET_SOURCE_INTEGRITY_KEY in metadata
    if has_fp_certificate:
        raw_fp_certificate = metadata[FP_TARGET_SOURCE_INTEGRITY_KEY]
        certificate_width = (
            raw_fp_certificate.get("n_sae_features")
            if isinstance(raw_fp_certificate, Mapping)
            else None
        )
        certificate, _digest = validate_fp_target_source_certificate(
            packed_dir,
            n_sae_features=cast(Any, certificate_width),
        )
        expected_ordered_inventory = [
            {"name": entry["name"], "byte_length": entry["byte_length"]} for entry in entries
        ]
        if (
            certificate["ordered_shard_inventory"] != expected_ordered_inventory
            or certificate["packed_row_fingerprint"] != packed_fold
        ):
            raise RemoteTrainingError(
                "packed manifest fp_target_source_integrity certificate "
                "disagrees with the new remote packed inventory"
            )
    elif _packed_family_requires_fp_certificate(packed_dir, metadata):
        raise RemoteTrainingError(
            "the c1024 remote packed family requires a completed "
            f"{FP_TARGET_SOURCE_INTEGRITY_KEY} certificate. Run "
            "scripts.preprocess.validate_fp_target_source before publishing "
            "the remote packed inventory."
        )

    if (
        _packed_file_signature(
            manifest_path,
            surface="cannot re-check packed-row manifest identity",
        )
        != manifest_signature
    ):
        raise RemoteTrainingError(
            "packed-row manifest changed while building remote inventory; "
            "retry after validate_fp_target_source finishes"
        )
    if selection_manifest_path is not None and selection_signature is not None:
        if (
            _packed_file_signature(
                selection_manifest_path,
                surface="cannot re-check selection manifest identity",
            )
            != selection_signature
        ):
            raise RemoteTrainingError(
                "selection manifest changed while building remote inventory; retry "
                "from quiescent immutable artifacts"
            )
    return sign_remote_packed_inventory(
        packed_manifest_file_fingerprint=_packed_file_fingerprint(
            manifest_path,
            surface="cannot fingerprint packed-row manifest",
        ),
        shards=entries,
        packed_row_fingerprint=packed_fold,
        provenance_shard_count=len(packed_info.provenance_shard_paths),
        provenance_fingerprint=provenance_fingerprint_value,
        base_shard_content_hash=base_hasher.hexdigest(),
        surface=f"new remote packed inventory for {packed_dir}",
    )


def publish_remote_packed_inventory(
    packed_row_dir: str | Path,
    *,
    selection_checksum: str | None = None,
) -> dict[str, Any]:
    """Build and atomically publish ``remote_packed_inventory.json`` locally."""
    packed_dir = Path(packed_row_dir)
    inventory = build_remote_packed_inventory(
        packed_dir,
        selection_checksum=selection_checksum,
    )
    atomic_json_dump(
        inventory,
        packed_dir / REMOTE_PACKED_INVENTORY_FILENAME,
    )
    return inventory


def _join_remote(prefix: str, *parts: str) -> str:
    values = [prefix, *parts]
    for value in values:
        _direct_remote_prefix(value, surface="remote object path component")
    return PurePosixPath(*values).as_posix()


def _relative_local_dir(path: Path, root: Path, *, surface: str) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RemoteTrainingError(
            f"{surface} {resolved_path} escapes configured local root {resolved_root}"
        ) from exc


def _verified_download_with_retry(
    backend: BucketBackend,
    *,
    remote_path: str,
    final_path: Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
    expected: RemoteObjectInfo | None = None,
    expected_byte_length: int | None = None,
    expected_content_sha256: str | None = None,
    retry_verification_errors: bool = True,
    cancellation_event: threading.Event | None = None,
    verified_publication_callback: Callable[[os.stat_result], None] | None = None,
    phase_timings: dict[str, float] | None = None,
) -> RemoteObjectInfo:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    final_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if cancellation_event is not None and cancellation_event.is_set():
            raise RemoteTrainingError(f"download cancelled: {remote_path}")
        attempt += 1
        tmp = final_path.parent / (
            f".{final_path.name}.remote-tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        transport_complete = False
        try:
            info = expected or backend.stat(remote_path)
            if info is None:
                raise FileNotFoundError(f"remote object not found: {remote_path}")
            download_started = time.perf_counter() if phase_timings is not None else 0.0
            backend.download(remote_path, tmp)
            if phase_timings is not None:
                phase_timings["download_seconds"] = time.perf_counter() - download_started
            transport_complete = True
            if cancellation_event is not None and cancellation_event.is_set():
                raise RemoteTrainingError(f"download cancelled: {remote_path}")
            verified_inode: _StableIdentity | None = None
            if verified_publication_callback is not None:
                before_verification = tmp.lstat()
                if before_verification.st_nlink != 1:
                    raise RemoteTrainingError(
                        f"downloaded object must be a single-link file: {remote_path}"
                    )
                verified_inode = _stat_identity(before_verification)
            need_sha256 = expected_content_sha256 is not None
            size, _, sha256, xet_hash = _stable_file_identity(
                tmp,
                allow_advisory_drift=True,
                need_sha256=need_sha256,
                phase_timings=phase_timings,
            )
            if verified_inode is not None:
                after_verification = tmp.lstat()
                _check_identity_drift(
                    verified_inode,
                    _stat_identity(after_verification),
                    path=tmp,
                    message=f"downloaded object changed after verification: {remote_path}",
                    allow_advisory_drift=True,
                )
            if size != info.byte_length or xet_hash != info.xet_hash:
                raise RemoteTrainingError(
                    f"downloaded object {remote_path} disagrees with remote size/Xet identity"
                )
            if expected_byte_length is not None and size != expected_byte_length:
                raise RemoteTrainingError(
                    f"downloaded object {remote_path} has size {size}, expected "
                    f"{expected_byte_length} from the committed artifact contract"
                )
            if expected_content_sha256 is not None:
                if sha256 != expected_content_sha256:
                    raise RemoteTrainingError(
                        f"downloaded object {remote_path} has SHA-256 {sha256}, expected "
                        f"{expected_content_sha256} from the committed artifact contract"
                    )
            with tmp.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(tmp, final_path)
            published: os.stat_result | None = None
            if verified_inode is not None:
                published = final_path.lstat()
                _check_identity_drift(
                    verified_inode,
                    _stat_identity(published),
                    path=final_path,
                    message=f"downloaded object changed during publication: {remote_path}",
                    allow_advisory_drift=True,
                )
            directory_fd = os.open(final_path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if verified_publication_callback is not None:
                assert published is not None
                verified_publication_callback(published)
            logger.info(
                f"remote_training.download_verified path={remote_path} "
                f"bytes={size} attempt={attempt}"
            )
            return info
        except BaseException as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            if not isinstance(exc, Exception):
                raise
            if transport_complete and not retry_verification_errors:
                if isinstance(exc, RemoteTrainingError):
                    raise
                raise RemoteTrainingError(
                    f"failed verifying downloaded object {remote_path}: {exc}"
                ) from exc
            if cancellation_event is not None and cancellation_event.is_set():
                raise RemoteTrainingError(f"download cancelled: {remote_path}") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteTrainingError(
                    f"timed out after {timeout_seconds:.1f}s downloading and verifying "
                    f"{remote_path}; attempts={attempt}; last_error={exc!r}"
                ) from exc
            logger.warning(
                f"remote_training.download_retry path={remote_path} attempt={attempt} "
                f"remaining_seconds={remaining:.1f} error={exc!r}"
            )
            retry_delay = min(retry_interval_seconds, remaining)
            if cancellation_event is None:
                time.sleep(retry_delay)
            elif cancellation_event.wait(retry_delay):
                raise RemoteTrainingError(f"download cancelled: {remote_path}") from exc


def _download_with_retry(
    backend: BucketBackend,
    *,
    remote_path: str,
    final_path: Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> None:
    """Download one body to an owner-private path with bounded retries.

    ``final_path`` is the owner's private PID/UUID transport file. The worker
    publishes it atomically only after the bucket client reports completion.
    Transport integrity is delegated to the bucket client; this helper never
    reopens or rereads the downloaded file.
    """
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    final_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        attempt += 1
        try:
            final_path.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise RemoteTrainingError(
                f"cannot prove private remote transport path absent before "
                f"attempt {attempt}: {final_path}: {exc}"
            ) from exc
        else:
            raise RemoteTrainingError(
                f"private remote transport path already exists before "
                f"attempt {attempt}: {final_path}"
            )
        try:
            backend.download(remote_path, final_path)
            logger.info(f"remote_training.download_complete path={remote_path} attempt={attempt}")
            return
        except BaseException as exc:
            try:
                final_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                failure = RemoteTrainingError(
                    "cannot remove the owner-private transport after failed "
                    f"attempt {attempt}: {final_path}: {cleanup_error}"
                )
                failure.add_note(f"the transport attempt also failed: {type(exc).__name__}: {exc}")
                raise failure from cleanup_error
            if not isinstance(exc, Exception):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteTrainingError(
                    f"timed out after {timeout_seconds:.1f}s downloading "
                    f"{remote_path}; attempts={attempt}; last_error={exc!r}"
                ) from exc
            logger.warning(
                f"remote_training.download_retry path={remote_path} attempt={attempt} "
                f"remaining_seconds={remaining:.1f} error={exc!r}"
            )
            time.sleep(min(retry_interval_seconds, remaining))


def _download_metadata_with_retry(
    backend: BucketBackend,
    *,
    remote_path: str,
    final_path: Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> RemoteObjectInfo:
    return _verified_download_with_retry(
        backend,
        remote_path=remote_path,
        final_path=final_path,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )


def _download_checkpoint_with_retry(
    backend: BucketBackend,
    *,
    remote_path: str,
    final_path: Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
    expected: RemoteObjectInfo | None = None,
    expected_byte_length: int | None = None,
    expected_content_sha256: str | None = None,
) -> RemoteObjectInfo:
    return _verified_download_with_retry(
        backend,
        remote_path=remote_path,
        final_path=final_path,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
        expected=expected,
        expected_byte_length=expected_byte_length,
        expected_content_sha256=expected_content_sha256,
    )


def _upload_with_retry(
    backend: BucketBackend,
    *,
    local_path: Path,
    remote_path: str,
    byte_length: int,
    xet_hash: str,
    timeout_seconds: float,
    retry_interval_seconds: float,
    retry_verification_errors: bool = True,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        verification_mismatch = False
        try:
            prior = backend.stat(remote_path)
            if prior is not None:
                if prior.byte_length != byte_length or prior.xet_hash != xet_hash:
                    raise RemoteTrainingError(f"same-path remote object collision at {remote_path}")
                logger.info(
                    f"remote_training.upload_verified path={remote_path} "
                    f"bytes={byte_length} attempt={attempt} idempotent=true"
                )
                return
            backend.upload(local_path, remote_path)
            uploaded = backend.stat(remote_path)
            if uploaded is None:
                raise RemoteTrainingError(
                    f"uploaded object {remote_path} failed size/Xet verification"
                )
            if uploaded.byte_length != byte_length or uploaded.xet_hash != xet_hash:
                verification_mismatch = True
                raise RemoteTrainingError(
                    f"uploaded object {remote_path} failed size/Xet verification"
                )
            logger.info(
                f"remote_training.upload_verified path={remote_path} "
                f"bytes={byte_length} attempt={attempt}"
            )
            return
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, RemoteTrainingError) and "collision" in str(exc):
                raise
            if verification_mismatch and not retry_verification_errors:
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteTrainingError(
                    f"timed out after {timeout_seconds:.1f}s uploading and verifying "
                    f"{remote_path}; attempts={attempt}; last_error={exc!r}"
                ) from exc
            logger.warning(
                f"remote_training.upload_retry path={remote_path} attempt={attempt} "
                f"remaining_seconds={remaining:.1f} error={exc!r}"
            )
            time.sleep(min(retry_interval_seconds, remaining))


def download_verified_object(
    backend: BucketBackend,
    *,
    expected: RemoteObjectInfo,
    final_path: Path,
    timeout_seconds: float,
    retry_interval_seconds: float,
    cancellation_event: threading.Event | None = None,
    verified_publication_callback: Callable[[os.stat_result], None] | None = None,
    phase_timings: dict[str, float] | None = None,
) -> RemoteObjectInfo:
    """Download one snapshotted bucket object and atomically publish verified bytes.

    This intentionally exposes the narrow size/Xet-verified transport already
    used for remote-training metadata and checkpoints.  ``expected`` is the
    sealed listing identity; the helper never refreshes it from a mutable bucket
    listing while retrying. If supplied, ``verified_publication_callback`` receives
    the final pathname stat for the exact inode verified and atomically published.
    """
    return _verified_download_with_retry(
        backend,
        remote_path=expected.path,
        final_path=final_path,
        timeout_seconds=timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
        expected=expected,
        retry_verification_errors=False,
        cancellation_event=cancellation_event,
        verified_publication_callback=verified_publication_callback,
        phase_timings=phase_timings,
    )


def verified_local_object_info(
    *,
    local_path: Path,
    remote_path: str,
) -> RemoteObjectInfo:
    """Return the stable size/Xet identity used by verified publication."""
    byte_length, _, _, xet_hash = _stable_file_identity(local_path)
    return RemoteObjectInfo(
        path=remote_path,
        byte_length=byte_length,
        xet_hash=xet_hash,
    )


@contextmanager
def _bound_verified_upload_source(
    local_path: Path,
    *,
    remote_path: str,
) -> Iterator[tuple[Path, RemoteObjectInfo]]:
    """Yield an fd-bound upload path and its exact size/Xet identity.

    The backend must reopen a path, so passing the mutable caller pathname
    after hashing leaves a claim-to-upload race. Linux ``/proc/self/fd`` keeps
    every retry bound to the one verified inode for the lifetime of the call.
    """
    if not hasattr(os, "O_NOFOLLOW"):
        raise RemoteTrainingError("verified upload requires os.O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(local_path, flags)
    except OSError as exc:
        raise RemoteTrainingError(
            f"verified upload source must be a direct regular file: {local_path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RemoteTrainingError(
                f"verified upload source must be a direct single-link regular file: {local_path}"
            )
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        )
        linked = local_path.lstat()
        if (
            linked.st_dev,
            linked.st_ino,
            linked.st_mode,
            linked.st_size,
            linked.st_mtime_ns,
            linked.st_ctime_ns,
            linked.st_nlink,
        ) != identity:
            raise RemoteTrainingError(
                f"verified upload source path changed while opening: {local_path}"
            )
        bound_path = Path(f"/proc/self/fd/{descriptor}")
        xet_result = cast(Any, hf_xet).hash_files([str(bound_path)])[0]
        after_hash = os.fstat(descriptor)
        if (
            after_hash.st_dev,
            after_hash.st_ino,
            after_hash.st_mode,
            after_hash.st_size,
            after_hash.st_mtime_ns,
            after_hash.st_ctime_ns,
            after_hash.st_nlink,
        ) != identity or int(xet_result.file_size) != opened.st_size:
            raise RemoteTrainingError(f"verified upload source changed while hashing: {local_path}")
        yield (
            bound_path,
            RemoteObjectInfo(
                path=remote_path,
                byte_length=opened.st_size,
                xet_hash=str(xet_result.hash),
            ),
        )
        after_upload = os.fstat(descriptor)
        try:
            linked_after = local_path.lstat()
        except OSError as exc:
            raise RemoteTrainingError(
                f"verified upload source path changed during publication: {local_path}"
            ) from exc
        if (
            after_upload.st_dev,
            after_upload.st_ino,
            after_upload.st_mode,
            after_upload.st_size,
            after_upload.st_mtime_ns,
            after_upload.st_ctime_ns,
            after_upload.st_nlink,
        ) != identity or (
            linked_after.st_dev,
            linked_after.st_ino,
            linked_after.st_mode,
            linked_after.st_size,
            linked_after.st_mtime_ns,
            linked_after.st_ctime_ns,
            linked_after.st_nlink,
        ) != identity:
            raise RemoteTrainingError(
                f"verified upload source path changed during publication: {local_path}"
            )
    finally:
        os.close(descriptor)


def _verified_upload_claim_prefix(remote_path: str) -> str:
    parsed = PurePosixPath(remote_path)
    if (
        not remote_path
        or remote_path.startswith("/")
        or "\\" in remote_path
        or any(part in {"", ".", ".."} for part in remote_path.split("/"))
        or parsed.as_posix() != remote_path
    ):
        raise RemoteTrainingError(f"invalid verified-upload remote path {remote_path!r}")
    return (parsed.parent / VERIFIED_UPLOAD_CLAIMS_DIRNAME / parsed.name).as_posix()


def _verified_upload_claim_bytes(info: RemoteObjectInfo) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "remote_path": info.path,
                "byte_length": info.byte_length,
                "xet_hash": info.xet_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def parse_verified_upload_claim_bytes(
    payload: bytes,
    *,
    expected_path: str,
    expected_xet_hash: str,
) -> RemoteObjectInfo:
    """Parse one canonical immutable-claim body and bind its path/Xet target."""
    try:
        value = json.loads(payload, object_pairs_hook=strict_json_object)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RemoteTrainingError("verified-upload claim is not readable JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "remote_path", "byte_length", "xet_hash"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or not isinstance(value["remote_path"], str)
        or type(value["byte_length"]) is not int
        or value["byte_length"] <= 0
        or not isinstance(value["xet_hash"], str)
        or _SHA256_RE.fullmatch(value["xet_hash"]) is None
    ):
        raise RemoteTrainingError("verified-upload claim schema or identity is malformed")
    claimed = RemoteObjectInfo(
        path=value["remote_path"],
        byte_length=value["byte_length"],
        xet_hash=value["xet_hash"],
    )
    _verified_upload_claim_prefix(claimed.path)
    if payload != _verified_upload_claim_bytes(claimed):
        raise RemoteTrainingError("verified-upload claim is not canonical JSON")
    if claimed.path != expected_path or claimed.xet_hash != expected_xet_hash:
        raise RemoteTrainingError(f"verified-upload claim does not identify {expected_path}")
    return claimed


def validate_local_verified_upload_claim(
    claim_path: Path,
    *,
    expected: RemoteObjectInfo,
) -> None:
    """Authenticate one pulled canonical object against its immutable claim."""
    claim_prefix = _verified_upload_claim_prefix(expected.path)
    claim_remote_path = PurePosixPath(
        claim_prefix,
        f"{expected.xet_hash}.json",
    ).as_posix()
    if (
        claim_path.name != f"{expected.xet_hash}.json"
        or claim_path.parent.name != PurePosixPath(expected.path).name
    ):
        raise RemoteTrainingError(
            f"verified-upload claim path does not identify {expected.path}: {claim_path}"
        )
    with _bound_verified_upload_source(
        claim_path,
        remote_path=claim_remote_path,
    ) as (bound_path, _):
        claimed = parse_verified_upload_claim_bytes(
            bound_path.read_bytes(),
            expected_path=expected.path,
            expected_xet_hash=expected.xet_hash,
        )
        if claimed != expected:
            raise RemoteTrainingError(
                f"verified-upload claim body does not identify {expected.path}"
            )


def _stat_with_retry(
    backend: BucketBackend,
    *,
    remote_path: str,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> RemoteObjectInfo | None:
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            return backend.stat(remote_path)
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteTrainingError(
                    f"timed out after {timeout_seconds:.1f}s reading remote identity "
                    f"for {remote_path}; attempts={attempt}; last_error={exc!r}"
                ) from exc
            logger.warning(
                f"remote_training.stat_retry path={remote_path} attempt={attempt} "
                f"remaining_seconds={remaining:.1f} error={exc!r}"
            )
            time.sleep(min(retry_interval_seconds, remaining))


def _publish_verified_upload_claim(
    backend: BucketBackend,
    *,
    local_info: RemoteObjectInfo,
    timeout_seconds: float,
    retry_interval_seconds: float,
) -> None:
    """Fence one canonical path with an immutable content-addressed claim."""
    claim_bytes = _verified_upload_claim_bytes(local_info)
    claim_prefix = _verified_upload_claim_prefix(local_info.path)
    claim_remote_path = PurePosixPath(claim_prefix, f"{local_info.xet_hash}.json").as_posix()
    descriptor, claim_local_name = tempfile.mkstemp(prefix="verified-upload-claim-")
    claim_local_path = Path(claim_local_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(claim_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with _bound_verified_upload_source(
            claim_local_path,
            remote_path=claim_remote_path,
        ) as (bound_claim_path, claim_info):
            if bound_claim_path.read_bytes() != claim_bytes:
                raise RemoteTrainingError(
                    "verified-upload claim differs from its deterministic bytes"
                )
            _upload_with_retry(
                backend,
                local_path=bound_claim_path,
                remote_path=claim_remote_path,
                byte_length=claim_info.byte_length,
                xet_hash=claim_info.xet_hash,
                timeout_seconds=timeout_seconds,
                retry_interval_seconds=retry_interval_seconds,
                retry_verification_errors=False,
            )
    finally:
        claim_local_path.unlink(missing_ok=True)

    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            observed: set[str] = set()
            prefix_parts = PurePosixPath(claim_prefix).parts
            for info in backend.list(claim_prefix):
                path_parts = PurePosixPath(info.path).parts
                if path_parts[: len(prefix_parts)] != prefix_parts:
                    raise RemoteTrainingError(
                        f"verified-upload claim listing escaped {claim_prefix!r}: {info.path!r}"
                    )
                relative = path_parts[len(prefix_parts) :]
                if (
                    len(relative) != 1
                    or re.fullmatch(r"[0-9a-f]{64}\.json", relative[0]) is None
                    or info.byte_length <= 0
                    or _SHA256_RE.fullmatch(info.xet_hash) is None
                ):
                    raise RemoteTrainingError(
                        f"malformed verified-upload claim under {claim_prefix}: {info!r}"
                    )
                if info.path in observed:
                    raise RemoteTrainingError(
                        f"duplicate verified-upload claim listing entry: {info.path}"
                    )
                observed.add(info.path)
            if claim_remote_path not in observed:
                raise FileNotFoundError(
                    f"verified claim is not visible in its prefix listing: {claim_remote_path}"
                )
            if observed != {claim_remote_path}:
                raise RemoteTrainingError(
                    f"same-path remote object collision at {local_info.path}; "
                    f"immutable_claims={sorted(observed)}"
                )
            return
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, RemoteTrainingError):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RemoteTrainingError(
                    f"timed out after {timeout_seconds:.1f}s verifying immutable "
                    f"publication claim for {local_info.path}; attempts={attempt}; "
                    f"last_error={exc!r}"
                ) from exc
            logger.warning(
                f"remote_training.upload_claim_retry path={local_info.path} "
                f"attempt={attempt} remaining_seconds={remaining:.1f} error={exc!r}"
            )
            time.sleep(min(retry_interval_seconds, remaining))


def upload_verified_object(
    backend: BucketBackend,
    *,
    local_path: Path,
    remote_path: str,
    timeout_seconds: float,
    retry_interval_seconds: float,
    expected_local: RemoteObjectInfo | None = None,
    expected_bytes: bytes | None = None,
) -> RemoteObjectInfo:
    """Idempotently upload one stable direct file and verify its exact identity."""
    with _bound_verified_upload_source(
        local_path,
        remote_path=remote_path,
    ) as (bound_path, local_info):
        if expected_bytes is not None and bound_path.read_bytes() != expected_bytes:
            raise RemoteTrainingError(
                f"local object {local_path} differs from its deterministic producer bytes"
            )
        if expected_local is not None and local_info != expected_local:
            raise RemoteTrainingError(
                f"local object {local_path} changed after publication claim: "
                f"claimed={expected_local!r}, observed={local_info!r}"
            )
        prior = _stat_with_retry(
            backend,
            remote_path=remote_path,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        if prior is not None:
            if prior != local_info:
                raise RemoteTrainingError(f"same-path remote object collision at {remote_path}")
            logger.info(
                f"remote_training.upload_verified path={remote_path} "
                f"bytes={local_info.byte_length} attempt=1 idempotent=true"
            )
            return prior
        _publish_verified_upload_claim(
            backend,
            local_info=local_info,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        _upload_with_retry(
            backend,
            local_path=bound_path,
            remote_path=remote_path,
            byte_length=local_info.byte_length,
            xet_hash=local_info.xet_hash,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
            retry_verification_errors=False,
        )
        verified = _stat_with_retry(
            backend,
            remote_path=remote_path,
            timeout_seconds=timeout_seconds,
            retry_interval_seconds=retry_interval_seconds,
        )
        if (
            verified is None
            or verified.byte_length != local_info.byte_length
            or verified.xet_hash != local_info.xet_hash
        ):
            raise RemoteTrainingError(
                f"uploaded object {remote_path} changed after size/Xet verification"
            )
        return verified


@dataclass
class RemoteSelectionMaterializer:
    local_dir: Path
    remote_dir: str
    inventory: dict[str, dict[str, Any]]
    loader_certificate: dict[str, Any]
    identity_contract: dict[str, Any]
    timeout_seconds: float
    retry_interval_seconds: float
    trust_existing_cache: bool
    preexisting_descriptors: dict[str, dict[str, int]] = field(default_factory=dict)
    _body_prefetcher: Any = field(default=None, repr=False)
    _body_binding: Any = field(default=None, repr=False)

    @classmethod
    def discover(
        cls,
        *,
        backend: BucketBackend,
        remote: RemoteTrainingConfig,
        local_selection_dir: Path,
        feature_root: Path,
        expected_selection_checksum: str,
        packed_row_fingerprint: str,
        packed_stems: set[str],
        label: str,
    ) -> RemoteSelectionMaterializer:
        relative = _relative_local_dir(
            local_selection_dir, feature_root, surface=f"{label} selection cache"
        )
        remote_dir = _join_remote(remote.selection_prefix, relative.as_posix())
        manifest_path = local_selection_dir / "manifest.json"
        downloaded_manifest = manifest_path.parent / (
            f".{manifest_path.name}.remote-discovery.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            _download_metadata_with_retry(
                backend,
                remote_path=_join_remote(remote_dir, "manifest.json"),
                final_path=downloaded_manifest,
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
            )
            try:
                manifest = json.loads(downloaded_manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RemoteTrainingError(
                    f"{label} remote manifest is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(manifest, dict):
                raise RemoteTrainingError(f"{label} remote manifest must be a JSON object")
            if manifest.get("checksum") != expected_selection_checksum:
                raise RemoteTrainingError(
                    f"{label} remote manifest selection checksum mismatch: "
                    f"{manifest.get('checksum')!r} != {expected_selection_checksum!r}"
                )
            metadata = manifest.get("metadata")
            if not isinstance(metadata, Mapping):
                raise RemoteTrainingError(f"{label} remote manifest metadata must be a mapping")
            inventory = parse_clustered_selection_shard_inventory(
                metadata.get(SELECTION_SHARD_INTEGRITY_KEY),
                surface=f"{label} remote manifest",
            )
            loader = parse_remote_loader_certificate(
                metadata.get(REMOTE_LOADER_KEY),
                surface=f"{label} remote manifest",
            )
            if loader["selection_checksum"] != expected_selection_checksum:
                raise RemoteTrainingError(f"{label} remote loader selection checksum drift")
            if loader["selection_inventory_aggregate"] != inventory["aggregate_sha256"]:
                raise RemoteTrainingError(f"{label} remote loader inventory aggregate drift")
            if loader["packed_row_fingerprint"] != packed_row_fingerprint:
                raise RemoteTrainingError(f"{label} remote loader packed-row fingerprint drift")
            splice_metadata = metadata.get("splice_counterfactual")
            if isinstance(splice_metadata, Mapping):
                expected_splice_identity = splice_counterfactual_manifest_identity(clustered=True)
                if any(
                    splice_metadata.get(key) != expected
                    for key, expected in expected_splice_identity.items()
                ):
                    raise RemoteTrainingError(
                        f"{label} remote manifest has stale or missing clustered "
                        "splice counterfactual schema/semantics"
                    )
                manifest_cluster_delta_semantics = splice_metadata[CLUSTER_DELTA_SEMANTICS_KEY]
                if loader.get(CLUSTER_DELTA_SEMANTICS_KEY) != manifest_cluster_delta_semantics:
                    raise RemoteTrainingError(
                        f"{label} remote loader cluster-delta semantics drift"
                    )
            elif CLUSTER_DELTA_SEMANTICS_KEY in loader:
                raise RemoteTrainingError(
                    f"{label} remote loader advertises clustered splice "
                    "semantics without splice_counterfactual manifest metadata"
                )
            entries = cast(list[dict[str, Any]], inventory["shards"])
            entry_stems = {entry["name"].removesuffix(".safetensors.zst") for entry in entries}
            if entry_stems != packed_stems:
                raise RemoteTrainingError(
                    f"{label} remote selection inventory does not exactly cover packed stems; "
                    f"missing={sorted(packed_stems - entry_stems)[:5]}, "
                    f"extra={sorted(entry_stems - packed_stems)[:5]}"
                )
            if manifest_path.exists():
                if (
                    not manifest_path.is_file()
                    or manifest_path.read_bytes() != downloaded_manifest.read_bytes()
                ):
                    raise RemoteTrainingError(
                        f"{label} local manifest conflicts with the verified remote manifest: {manifest_path}"
                    )
                downloaded_manifest.unlink()
            else:
                os.replace(downloaded_manifest, manifest_path)
            materializer = cls(
                local_dir=local_selection_dir,
                remote_dir=remote_dir,
                inventory={entry["name"]: entry for entry in entries},
                loader_certificate=loader,
                identity_contract={
                    "selection_checksum": expected_selection_checksum,
                    "selection_inventory_aggregate": inventory["aggregate_sha256"],
                    "packed_row_fingerprint": loader["packed_row_fingerprint"],
                    "remote_loader_aggregate": loader["aggregate_sha256"],
                    "feature_mask": metadata.get("feature_mask"),
                    "feature_clusters": (
                        {
                            key: metadata["feature_clusters"].get(key)
                            for key in ("descriptor", "canonical", "counts", "status")
                        }
                        if isinstance(metadata.get("feature_clusters"), Mapping)
                        else None
                    ),
                    "packed_rows": metadata.get("packed_rows"),
                    "packed_provenance": metadata.get("packed_provenance"),
                    "splice_counterfactual": metadata.get("splice_counterfactual"),
                },
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
                trust_existing_cache=bool(
                    remote.train_packed and remote.train_packed.trust_existing_cache
                ),
            )
            materializer.validate_cache_namespace()
            return materializer
        finally:
            try:
                downloaded_manifest.unlink()
            except OSError:
                pass

    @property
    def max_features_per_segment(self) -> int:
        return max(1, int(self.loader_certificate["max_features_per_segment"]))

    def _verify_cached_body(self, path: Path, entry: Mapping[str, Any]) -> None:
        if self.trust_existing_cache:
            _check_trusted_cache_body(path, byte_length=entry["byte_length"])
        else:
            _verify_file(
                path,
                byte_length=entry["byte_length"],
                content_sha256=entry["content_sha256"],
            )

    def validate_cache_namespace(self) -> None:
        captured: dict[str, dict[str, int]] = {}
        expected = set(self.inventory)
        for path in self.local_dir.iterdir():
            if path.name.endswith(".safetensors.zst") and path.name not in expected:
                raise RemoteTrainingError(
                    f"unexpected finalized selection body in remote cache: {path}"
                )
            if path.name in expected:
                entry = self.inventory[path.name]
                lock_fd = _open_cache_lock(self.local_dir / f".{path.name}.lock")
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_SH)
                    if path.exists() or path.is_symlink():
                        try:
                            self._verify_cached_body(path, entry)
                        except FileNotFoundError:
                            pass
                        else:
                            captured[path.name] = _capture_cache_body_descriptor(path)
                finally:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lock_fd)
        self.preexisting_descriptors = captured

    def shard_paths(self) -> dict[str, Path]:
        return {
            name.removesuffix(".safetensors.zst"): self.local_dir / name for name in self.inventory
        }

    def read(
        self,
        path: Path,
        *,
        cancel_event: threading.Event | None = None,
        epoch: int = 1,
    ) -> bytes:
        """Read one validated owner-published selection body without transport capability."""
        _raise_if_remote_cancelled(cancel_event)
        path = Path(path)
        if path.parent.resolve() != self.local_dir.resolve() or path.name not in self.inventory:
            raise RemoteTrainingError(f"selection materializer received unexpected path {path}")
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError(
                "remote selection reader is not attached to strict rank-striped "
                "body prefetch; synchronous demand download was removed"
            )
        return cast(
            bytes,
            self._body_prefetcher.read_body(
                binding=self._body_binding,
                path=path,
                entry=self.inventory[path.name],
                trust_existing_cache=self.trust_existing_cache,
                cancel_event=cancel_event,
                epoch=epoch,
            ),
        )

    def attach_body_prefetch(self, prefetcher: Any, binding: Any) -> None:
        if self._body_prefetcher is not None or self._body_binding is not None:
            raise RemoteTrainingError(
                f"remote selection materializer is already attached: {self.local_dir}"
            )
        self._body_prefetcher = prefetcher
        self._body_binding = binding

    def publish_cursor(self, path: Path, *, epoch: int) -> None:
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError("remote selection materializer has no body prefetch worker")
        self._body_prefetcher.publish_cursor(
            binding=self._body_binding,
            epoch=epoch,
            inventory_index=self._body_binding.inventory_index(Path(path).name),
        )

    def mark_loaded(self, path: Path, *, epoch: int) -> None:
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError("remote selection materializer has no body prefetch worker")
        self._body_prefetcher.mark_loaded(
            binding=self._body_binding,
            epoch=epoch,
            inventory_index=self._body_binding.inventory_index(Path(path).name),
        )

    def request_startup_window(self, *, start_index: int = 0) -> None:
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError("remote selection materializer has no body prefetch worker")
        self._body_prefetcher.request_window(
            binding=self._body_binding,
            start_index=start_index,
        )

    def check_worker_errors(self) -> None:
        if self._body_prefetcher is None:
            raise RemoteTrainingError("remote selection materializer has no body prefetch worker")
        self._body_prefetcher.check_worker_errors()


@dataclass
class RemotePackedRowMaterializer:
    """Verified, bounded local cache for one remote train packed-row family."""

    local_dir: Path
    remote_dir: str
    inventory: dict[str, Any]
    cache_shards: int
    base_shard_stems: list[str]
    provenance_shard_paths: list[Path]
    timeout_seconds: float
    retry_interval_seconds: float
    trust_existing_cache: bool
    preexisting_descriptors: dict[str, dict[str, int]] = field(default_factory=dict)
    _body_prefetcher: Any = field(default=None, repr=False)
    _body_binding: Any = field(default=None, repr=False)

    @classmethod
    def discover(
        cls,
        *,
        backend: BucketBackend,
        remote: RemoteTrainingConfig,
        local_packed_dir: Path,
        feature_root: Path,
        cache_shards: int,
        label: str,
    ) -> RemotePackedRowMaterializer:
        if type(cache_shards) is not int or cache_shards < 2:
            raise RemoteTrainingError(
                f"{label} cache_shards must be a strict int >= 2, got {cache_shards!r}"
            )
        local_packed_dir = Path(local_packed_dir)
        local_packed_dir.mkdir(parents=True, exist_ok=True)
        relative = _relative_local_dir(
            local_packed_dir,
            feature_root,
            surface=f"{label} packed-row cache",
        )
        remote_dir = _join_remote(remote.selection_prefix, relative.as_posix())

        manifest_path = local_packed_dir / "manifest.json"
        if not manifest_path.exists():
            _download_metadata_with_retry(
                backend,
                remote_path=_join_remote(remote_dir, "manifest.json"),
                final_path=manifest_path,
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
            )
        elif not manifest_path.is_file():
            raise RemoteTrainingError(
                f"{label} local packed manifest is not a regular file: {manifest_path}"
            )

        inventory_path = local_packed_dir / REMOTE_PACKED_INVENTORY_FILENAME
        downloaded_inventory = local_packed_dir / (
            f".{REMOTE_PACKED_INVENTORY_FILENAME}.remote-discovery."
            f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            _download_metadata_with_retry(
                backend,
                remote_path=_join_remote(
                    remote_dir,
                    REMOTE_PACKED_INVENTORY_FILENAME,
                ),
                final_path=downloaded_inventory,
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
            )
            if inventory_path.exists() or inventory_path.is_symlink():
                if (
                    not inventory_path.is_file()
                    or inventory_path.read_bytes() != downloaded_inventory.read_bytes()
                ):
                    raise RemoteTrainingError(
                        f"{label} local {REMOTE_PACKED_INVENTORY_FILENAME} "
                        "conflicts with the verified remote object: "
                        f"{inventory_path}"
                    )
                downloaded_inventory.unlink()
            else:
                os.replace(downloaded_inventory, inventory_path)

            inventory = parse_remote_packed_inventory(
                _load_json_object(
                    inventory_path,
                    surface=f"{label} remote packed inventory",
                ),
                surface=f"{label} remote packed inventory",
            )
            actual_manifest_fingerprint = _packed_file_fingerprint(
                manifest_path,
                surface=f"{label} cannot fingerprint the local packed manifest",
            )
            expected_manifest_fingerprint = inventory["packed_manifest_file_fingerprint"]
            if actual_manifest_fingerprint != expected_manifest_fingerprint:
                raise RemoteTrainingError(
                    f"{label} packed manifest fingerprint mismatch: local "
                    f"{manifest_path} hashes to {actual_manifest_fingerprint}, "
                    f"inventory requires {expected_manifest_fingerprint}. Re-run "
                    "scripts.preprocess.publish_remote_packed_inventory after "
                    "the final validate_fp_target_source manifest stamp."
                )
            manifest = read_manifest(local_packed_dir)
            if manifest is None:
                raise RemoteTrainingError(
                    f"{label} packed manifest is missing or invalid: {manifest_path}"
                )
            metadata = manifest.get("metadata")
            if not isinstance(metadata, Mapping):
                raise RemoteTrainingError(f"{label} packed manifest metadata must be a mapping")
            raw_stems = metadata.get("base_shard_stems")
            if not isinstance(raw_stems, list) or not all(
                isinstance(stem, str) for stem in raw_stems
            ):
                raise RemoteTrainingError(
                    f"{label} packed manifest metadata.base_shard_stems must be a list of strings"
                )
            entries = cast(list[dict[str, Any]], inventory["shards"])
            recorded_provenance_count = metadata.get("provenance_shards")
            if (
                type(recorded_provenance_count) is not int
                or recorded_provenance_count != inventory["provenance_shard_count"]
                or recorded_provenance_count != len(entries)
            ):
                raise RemoteTrainingError(
                    f"{label} packed provenance count mismatch: manifest="
                    f"{recorded_provenance_count!r}, inventory="
                    f"{inventory['provenance_shard_count']!r}, packed_shards="
                    f"{len(entries)}"
                )
            prov_dir = provenance_dir(local_packed_dir)
            if not prov_dir.is_dir():
                raise RemoteTrainingError(
                    f"{label} requires pre-staged packed provenance at {prov_dir}"
                )
            provenance_paths = glob_overlay_shards(prov_dir)
            expected_names = [entry["name"] for entry in entries]
            provenance_names = [path.name for path in provenance_paths]
            if provenance_names != expected_names:
                missing = sorted(set(expected_names) - set(provenance_names))
                extra = sorted(set(provenance_names) - set(expected_names))
                raise RemoteTrainingError(
                    f"{label} packed provenance names must pair 1:1 with the "
                    f"remote packed inventory; missing={missing[:5]}, "
                    f"extra={extra[:5]}"
                )
            if len(provenance_paths) != recorded_provenance_count:
                raise RemoteTrainingError(
                    f"{label} packed provenance count mismatch: local="
                    f"{len(provenance_paths)}, required="
                    f"{recorded_provenance_count}"
                )
            local_provenance_fingerprint = _fingerprint_files_quiescent(
                provenance_paths,
                surface=f"{label} packed provenance",
            )
            if local_provenance_fingerprint != inventory["provenance_fingerprint"]:
                raise RemoteTrainingError(
                    f"{label} packed provenance fingerprint mismatch: local="
                    f"{local_provenance_fingerprint}, inventory="
                    f"{inventory['provenance_fingerprint']}"
                )
            if FP_TARGET_SOURCE_INTEGRITY_KEY in metadata:
                raw_fp_certificate = metadata[FP_TARGET_SOURCE_INTEGRITY_KEY]
                certificate_width = (
                    raw_fp_certificate.get("n_sae_features")
                    if isinstance(raw_fp_certificate, Mapping)
                    else None
                )
                validate_fp_target_source_certificate_remote(
                    local_packed_dir,
                    n_sae_features=cast(Any, certificate_width),
                    inventory=inventory,
                )
            materializer = cls(
                local_dir=local_packed_dir,
                remote_dir=remote_dir,
                inventory=inventory,
                cache_shards=cache_shards,
                base_shard_stems=[str(stem) for stem in raw_stems],
                provenance_shard_paths=provenance_paths,
                timeout_seconds=remote.selection_wait_timeout_seconds,
                retry_interval_seconds=remote.retry_interval_seconds,
                trust_existing_cache=bool(
                    remote.train_packed and remote.train_packed.trust_existing_cache
                ),
            )
            materializer.validate_cache_namespace()
            return materializer
        finally:
            try:
                downloaded_inventory.unlink()
            except OSError:
                pass

    @property
    def packed_row_fingerprint(self) -> str:
        return cast(str, self.inventory["packed_row_fingerprint"])

    @property
    def total_segments_by_stem(self) -> dict[str, int]:
        return {
            entry["name"].removesuffix(".safetensors.zst"): entry["total_segments"]
            for entry in self.inventory["shards"]
        }

    def shard_paths(self) -> list[Path]:
        return [self.local_dir / entry["name"] for entry in self.inventory["shards"]]

    def stems(self) -> set[str]:
        return set(self.total_segments_by_stem)

    def packed_provenance_info(self) -> PackedProvenanceInfo:
        return PackedProvenanceInfo(
            base_shard_stems=list(self.base_shard_stems),
            packed_shard_paths=self.shard_paths(),
            provenance_shard_paths=list(self.provenance_shard_paths),
            packed_stem_fingerprint=_fingerprint_json(self.base_shard_stems),
            packed_row_fingerprint=self.packed_row_fingerprint,
            provenance_fingerprint=cast(
                str,
                self.inventory["provenance_fingerprint"],
            ),
        )

    def assert_manifest_current(self) -> None:
        manifest_path = self.local_dir / "manifest.json"
        actual = _packed_file_fingerprint(
            manifest_path,
            surface="cannot re-check packed manifest during remote streaming",
        )
        expected = self.inventory["packed_manifest_file_fingerprint"]
        if actual != expected:
            raise RemoteTrainingError(
                "packed manifest changed after remote inventory discovery: "
                f"{manifest_path} hashes to {actual}, inventory requires "
                f"{expected}. Re-run "
                "scripts.preprocess.publish_remote_packed_inventory."
            )

    def _entry(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        if path.parent.resolve() != self.local_dir.resolve():
            raise RemoteTrainingError(
                f"packed materializer received path outside its cache: {path}"
            )
        for entry in self.inventory["shards"]:
            if entry["name"] == path.name:
                return cast(dict[str, Any], entry)
        raise RemoteTrainingError(f"packed materializer received unmanifested path: {path}")

    @staticmethod
    def _verify_body(
        path: Path,
        entry: Mapping[str, Any],
        *,
        fingerprint_name: str | None = None,
    ) -> None:
        size, content_sha256, file_fingerprint = _dual_digest(
            path,
            fingerprint_name=fingerprint_name,
        )
        if (
            size != entry["byte_length"]
            or content_sha256 != entry["content_sha256"]
            or file_fingerprint != entry["file_fingerprint"]
        ):
            raise RemoteTrainingError(
                f"remote packed body integrity mismatch for {path}: expected "
                "size/content_sha256/file_fingerprint "
                f"{entry['byte_length']}/{entry['content_sha256']}/"
                f"{entry['file_fingerprint']}, got {size}/{content_sha256}/"
                f"{file_fingerprint}"
            )

    def _verify_cached_body(self, path: Path, entry: Mapping[str, Any]) -> None:
        if self.trust_existing_cache:
            _check_trusted_cache_body(path, byte_length=entry["byte_length"])
        else:
            self._verify_body(path, entry)

    def validate_cache_namespace(self) -> None:
        # Discovery transports are owned and cleaned by their creating process.
        # A PID from another cluster node cannot be tested for liveness locally.
        captured: dict[str, dict[str, int]] = {}
        expected = {entry["name"]: entry for entry in self.inventory["shards"]}
        for path in self.local_dir.iterdir():
            if path.name.endswith(".safetensors.zst") and path.name not in expected:
                raise RemoteTrainingError(
                    f"unexpected finalized packed body in remote cache: {path}"
                )
            if path.name in expected:
                # Verify under the same per-shard lock every reader/evictor
                # uses. A sibling process may evict between the directory
                # snapshot and this check; absence is a valid cache state, so
                # a vanished body is skipped rather than treated as corruption.
                lock_fd = _open_cache_lock(self.local_dir / f".{path.name}.lock")
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_SH)
                    if path.exists() or path.is_symlink():
                        self._verify_cached_body(path, expected[path.name])
                        captured[path.name] = _capture_cache_body_descriptor(path)
                finally:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    finally:
                        os.close(lock_fd)
        self.preexisting_descriptors = captured

    def read(
        self,
        path: Path,
        *,
        cancel_event: threading.Event | None = None,
        epoch: int = 1,
    ) -> bytes:
        """Read one validated owner-published packed body without transport capability."""
        _raise_if_remote_cancelled(cancel_event)
        path = Path(path)
        entry = self._entry(path)
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError(
                "remote packed reader is not attached to strict rank-striped "
                "body prefetch; synchronous demand download was removed"
            )
        return cast(
            bytes,
            self._body_prefetcher.read_body(
                binding=self._body_binding,
                path=path,
                entry=entry,
                trust_existing_cache=self.trust_existing_cache,
                cancel_event=cancel_event,
                epoch=epoch,
            ),
        )

    def attach_body_prefetch(self, prefetcher: Any, binding: Any) -> None:
        if self._body_prefetcher is not None or self._body_binding is not None:
            raise RemoteTrainingError(
                f"remote packed materializer is already attached: {self.local_dir}"
            )
        self._body_prefetcher = prefetcher
        self._body_binding = binding

    def publish_cursor(self, path: Path, *, epoch: int) -> None:
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError("remote packed materializer has no body prefetch worker")
        self._body_prefetcher.publish_cursor(
            binding=self._body_binding,
            epoch=epoch,
            inventory_index=self._body_binding.inventory_index(Path(path).name),
        )

    def mark_loaded(self, path: Path, *, epoch: int) -> None:
        if self._body_prefetcher is None or self._body_binding is None:
            raise RemoteTrainingError("remote packed materializer has no body prefetch worker")
        self._body_prefetcher.mark_loaded(
            binding=self._body_binding,
            epoch=epoch,
            inventory_index=self._body_binding.inventory_index(Path(path).name),
        )

    def check_worker_errors(self) -> None:
        if self._body_prefetcher is None:
            raise RemoteTrainingError("remote packed materializer has no body prefetch worker")
        self._body_prefetcher.check_worker_errors()

    def validate_loaded_shard(
        self,
        path: Path,
        *,
        total_segments: int,
    ) -> None:
        """Run the deferred packed/provenance semantic gate before consumption."""
        entry = self._entry(Path(path))
        expected_total = entry["total_segments"]
        if total_segments != expected_total:
            raise RemoteTrainingError(
                f"{path}: packed shard total_segments={total_segments} but the "
                f"remote inventory certifies {expected_total}"
            )
        provenance_path = provenance_dir(self.local_dir) / path.name
        provenance = load_overlay_shard(provenance_path)
        try:
            base_idx = provenance[PROVENANCE_BASE_SHARD_IDX_KEY]
            doc_idx = provenance[PROVENANCE_DOC_IDX_KEY]
            seg_ordinal = provenance[PROVENANCE_SEG_ORDINAL_KEY]
        except KeyError as exc:
            raise RemoteTrainingError(
                f"{provenance_path}: missing provenance tensor key {exc.args[0]!r}"
            ) from exc
        if not (len(base_idx) == len(doc_idx) == len(seg_ordinal) == total_segments):
            raise RemoteTrainingError(
                f"{provenance_path}: provenance has lengths {len(base_idx)}, "
                f"{len(doc_idx)}, {len(seg_ordinal)} but packed shard "
                f"{path.name} has {total_segments} segments"
            )
        if bool((base_idx < 0).any() or (doc_idx < 0).any() or (seg_ordinal < 0).any()):
            raise RemoteTrainingError(
                f"{provenance_path}: corrupt packed provenance contains "
                "negative (base_shard_idx, doc_idx, seg_ordinal) fields"
            )
        if bool((base_idx >= len(self.base_shard_stems)).any()):
            bad = int(base_idx[base_idx >= len(self.base_shard_stems)][0].item())
            raise RemoteTrainingError(
                f"{provenance_path}: base_shard_idx {bad} is out of range for "
                "manifest base_shard_stems length "
                f"{len(self.base_shard_stems)}"
            )


@beartype
def _checkpoint_names(
    step: NonNegativeIntLike,
    world_size: PositiveIntLike,
) -> list[str]:
    return [
        f"model_{step:06d}.pt",
        f"meta_{step:06d}.json",
        *(f"optim_{step:06d}_rank{rank}.pt" for rank in range(world_size)),
    ]


@dataclass(frozen=True)
class _CheckpointFileIdentity:
    name: str
    mode: int
    device: int
    inode: int
    byte_length: int
    mtime_ns: int
    ctime_ns: int
    link_count: int

    @classmethod
    def capture(cls, path: Path) -> _CheckpointFileIdentity:
        try:
            info = path.lstat()
        except OSError as exc:
            raise RemoteTrainingError(
                f"cannot capture checkpoint file identity for {path}: {exc}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise RemoteTrainingError(
                f"checkpoint publication requires a direct regular file: {path}"
            )
        if info.st_nlink != 1:
            raise RemoteTrainingError(
                f"checkpoint publication requires exactly one hard link: {path}"
            )
        return cls(
            name=path.name,
            mode=info.st_mode,
            device=info.st_dev,
            inode=info.st_ino,
            byte_length=info.st_size,
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
            link_count=info.st_nlink,
        )


@beartype
def _checkpoint_group_identity(
    checkpoint_dir: Path,
    *,
    step: NonNegativeIntLike,
    world_size: PositiveIntLike,
) -> tuple[_CheckpointFileIdentity, ...]:
    return tuple(
        _CheckpointFileIdentity.capture(checkpoint_dir / name)
        for name in _checkpoint_names(step, world_size)
    )


@beartype
def build_checkpoint_marker(
    checkpoint_dir: Path,
    *,
    step: NonNegativeIntLike,
    world_size: PositiveIntLike,
    remote_run_fingerprint: LowercaseSha256Like,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for name in _checkpoint_names(step, world_size):
        path = checkpoint_dir / name
        size, _, sha256, xet_hash = _stable_file_identity(path)
        entries.append(
            {
                "name": name,
                "byte_length": size,
                "content_sha256": sha256,
                "xet_hash": xet_hash,
            }
        )
    # Recheck the whole group after the first pass. This catches a file that
    # changed while a later group member was being hashed, before a completion
    # marker can describe a cross-file snapshot that never existed locally.
    for entry in entries:
        _verify_file(
            checkpoint_dir / entry["name"],
            byte_length=entry["byte_length"],
            content_sha256=entry["content_sha256"],
            xet_hash=entry["xet_hash"],
        )
    signed = {
        "schema_version": REMOTE_CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "ddp_world_size": world_size,
        "remote_run_fingerprint": remote_run_fingerprint,
        "files": entries,
    }
    return {**signed, "aggregate_sha256": _sha256_json(signed)}


def parse_checkpoint_marker(raw: Any, *, surface: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _CHECKPOINT_MARKER_KEYS:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise RemoteTrainingError(
            f"{surface} marker must contain exactly {sorted(_CHECKPOINT_MARKER_KEYS)}; "
            f"got keys={sorted(keys, key=str)}"
        )
    schema = raw["schema_version"]
    step = raw["step"]
    world_size = raw["ddp_world_size"]
    if type(schema) is not int or schema != REMOTE_CHECKPOINT_SCHEMA_VERSION:
        raise RemoteTrainingError(f"{surface} marker schema_version is unsupported")
    parsed_step = _strict_nonnegative_int(step, surface=f"{surface} marker step")
    if not isinstance(world_size, PositiveInt):
        raise RemoteTrainingError(
            f"{surface} marker DDP world size must be a positive strict int, got {world_size!r}"
        )
    expected_names = _checkpoint_names(parsed_step, PositiveInt.parse(world_size))
    run_fingerprint = _strict_sha256(
        raw["remote_run_fingerprint"], surface=f"{surface} remote_run_fingerprint"
    )
    files = raw["files"]
    if not isinstance(files, list) or len(files) != len(expected_names):
        raise RemoteTrainingError(f"{surface} marker has the wrong required file count")
    parsed: list[dict[str, Any]] = []
    for index, entry_raw in enumerate(files):
        if not isinstance(entry_raw, Mapping) or set(entry_raw) != _CHECKPOINT_FILE_KEYS:
            raise RemoteTrainingError(f"{surface} marker file entry {index} has invalid keys")
        entry = cast(Mapping[str, Any], entry_raw)
        name = entry["name"]
        if not isinstance(name, str) or name != expected_names[index] or name != Path(name).name:
            raise RemoteTrainingError(
                f"{surface} marker filename set/order is invalid at entry {index}: {name!r}"
            )
        size = entry["byte_length"]
        if type(size) is not int or size < 0:
            raise RemoteTrainingError(
                f"{surface} marker byte_length must be a non-negative strict int"
            )
        parsed.append(
            {
                "name": name,
                "byte_length": size,
                "content_sha256": _strict_sha256(
                    entry["content_sha256"], surface=f"{surface} {name} SHA-256"
                ),
                "xet_hash": _strict_sha256(entry["xet_hash"], surface=f"{surface} {name} Xet hash"),
            }
        )
    signed = {
        "schema_version": schema,
        "step": step,
        "ddp_world_size": world_size,
        "remote_run_fingerprint": run_fingerprint,
        "files": parsed,
    }
    expected_aggregate = _sha256_json(signed)
    if raw["aggregate_sha256"] != expected_aggregate:
        raise RemoteTrainingError(f"{surface} marker aggregate mismatch")
    return {**signed, "aggregate_sha256": expected_aggregate}


_NON_SEMANTIC_CONFIG_KEYS = frozenset(
    {
        "remote_training",
        "resume_from_step",
        "feature_dir",
        "feature_root",
        "packed_row_dir",
        "data_dir",
        "original_val_feature_dir",
        "original_val_data_dir",
        "model_tag",
        "run",
        "output_dirname",
        "feature_mask",
        "feature_mask_freq_min",
        "feature_mask_freq_max",
        "feature_clusters",
        "train_node_cache_max_bytes",
        "train_node_cache_reserve_bytes",
        "selection_wait_timeout_seconds",
        "checkpoint_retry_timeout_seconds",
        "retry_interval_seconds",
        "resume_latest",
        "selection_resume",
        "selection_num_workers",
        "selection_source_cache_shards",
        "selection_sidecar_cache_shards",
        "selection_shared_store",
        "selection_shared_store_capacity",
        "selection_packed_spool_dir",
    }
)


def _semantic_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_config(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _NON_SEMANTIC_CONFIG_KEYS
            and str(key) != "path"
            and not str(key).endswith("_path")
            and not str(key).endswith("_dir")
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_config(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RemoteTrainingError(
        f"semantic training config contains unsupported value {type(value).__name__}"
    )


@beartype
def build_remote_run_fingerprint(
    *,
    resolved_training_config: _SemanticConfigMapping,
    world_size: PositiveIntLike,
    selection_contracts: _SelectionContracts,
) -> str:
    if set(selection_contracts) != {"train", "derived_val", "base_val"}:
        raise RemoteTrainingError(
            "remote run fingerprint requires train, derived_val, and base_val selection contracts"
        )
    training_config = dict(resolved_training_config)
    if parse_conditioning_cooldown(training_config.get("conditioning_cooldown")) is None:
        training_config.pop("conditioning_cooldown", None)
    payload = {
        "schema_version": 1,
        "context_length": 1024,
        "ddp_world_size": world_size,
        "training_config": _semantic_config(training_config),
        "selection_contracts": {
            label: _semantic_config(selection_contracts[label])
            for label in ("train", "derived_val", "base_val")
        },
    }
    return _sha256_json(payload)


_OBSERVATION_CONFIG_KEYS_V2 = frozenset(
    {
        "conditioning",
        "eval_every",
        "eval_tokens",
        "core_metric_every",
        "core_metric_max_per_task",
        "sample_every",
        "sample_prompts",
        "steering_eval_every",
        "steering_eval",
        "offload_steering_eval_feature_model",
    }
)
_EXECUTION_CONFIG_KEYS_V2 = frozenset(
    {
        "device_batch_size",
        "save_every",
        "remote_training",
        "resume_from_step",
        "run",
        "model_tag",
        "output_dirname",
        "output_dir",
        "profile_trace",
        "profile_trace_steps",
        "detect_anomaly",
        "verify_features",
        "feature_prefetch",
        "doc_prefetch_depth",
        "train_prefetch_batches",
        "train_shard_lookahead",
        "train_node_cache_dir",
        "train_node_cache_max_bytes",
        "train_node_cache_reserve_bytes",
        "tokenizer_batch_size",
    }
)
_PRODUCER_CONFIG_KEYS_V2 = frozenset(
    {
        "feature_dir",
        "feature_root",
        "packed_row_dir",
        "data_dir",
        "original_val_feature_dir",
        "original_val_data_dir",
        "feature_mask",
        "feature_mask_freq_min",
        "feature_mask_freq_max",
        "feature_clusters",
        "remote_preselection",
        "selection_resume",
        "selection_splice_batch_size",
        "selection_source_cache_shards",
        "selection_sidecar_cache_shards",
        "selection_num_workers",
        "selection_packed_spool_dir",
        "selection_shared_store",
        "selection_shared_store_capacity",
    }
)
_SEMANTIC_CONFIG_KEYS_V2 = frozenset(
    {
        "device_type",
        "attn_impl",
        "fp8",
        "fp8_recipe",
        "depth",
        "aspect_ratio",
        "head_dim",
        "max_seq_len",
        "window_pattern",
        "num_iterations",
        "target_flops",
        "target_param_data_ratio",
        "total_batch_size",
        "embedding_lr",
        "unembedding_lr",
        "weight_decay",
        "matrix_lr",
        "scalar_lr",
        "adam_beta1",
        "adam_beta2",
        "warmup_ratio",
        "warmdown_ratio",
        "final_lr_frac",
        "encoder",
        "cross_attention",
        "pooling",
        "dataset_phase",
        "feature_selection",
        "num_buckets",
        "encoder_embedding_lr",
        "encoder_matrix_lr",
        "xattn_matrix_lr",
        "negative_selection",
        "feature_delta",
        "segment_size",
        "decoder_segment_size",
        "label_dropout",
        "conditioning_cooldown",
        "initialize_encoder_embeddings_from_sae",
        "initialize_encoder_embeddings_cluster_vectors",
        "use_precomputed_batches",
        "feature_prediction",
        "seed",
        "compile",
        "compile_static",
        "sae",
        "next_segment_conditioning_offset",
        "max_train_shards",
    }
)
_CONFIG_CATEGORIES_V2 = {
    "observation": _OBSERVATION_CONFIG_KEYS_V2,
    "execution": _EXECUTION_CONFIG_KEYS_V2,
    "producer": _PRODUCER_CONFIG_KEYS_V2,
    "semantic": _SEMANTIC_CONFIG_KEYS_V2,
}
_DECLARED_NESTED_KEYS_V2: dict[str, frozenset[str]] = {
    "conditioning_cooldown": frozenset(
        {
            "enabled", "start_fraction", "end_fraction", "start_step", "end_step",
            "cooldown_tokens", "granularity",
        }
    ),
    "conditioning": frozenset(
        {
            "context_cache_dir",
            "feature_clusters",
            "feature_delta",
            "feature_mask",
            "feature_selection",
            "identity",
            "n_sae_features",
            "negative_selection",
            "num_buckets",
            "seed",
            "stats_source_sha256",
        }
    ),
    "identity": frozenset(
        {
            "model_id",
            "sae_id",
            "layer",
            "pooling",
            "feature_selection_hash",
        }
    ),
    "steering_eval": frozenset(
        {"n_eval_docs", "samples_per_condition", "temperature", "top_k_sampling", "seed"}
    ),
    "remote_training": _REMOTE_CONFIG_KEYS | _REMOTE_OPTIONAL_CONFIG_KEYS,
    "remote_preselection": frozenset({"staging_dir", "max_output_shard_bytes"}),
    "body_prefetch": _REMOTE_BODY_PREFETCH_KEYS,
    "train_packed": _REMOTE_TRAIN_PACKED_KEYS,
    "cuda_staging": _REMOTE_CUDA_STAGING_KEYS,
    "feature_mask": frozenset({"mode", "ids", "path"}),
    "feature_clusters": frozenset({"path"}),
    "encoder": frozenset({"n_embd", "depth", "n_head", "n_kv_head", "head_dim"}),
    "cross_attention": frozenset({"frequency", "n_head", "n_kv_head"}),
    "feature_selection": frozenset(
        {"mode", "distribution", "mean", "std", "min_k", "max_k", "delta_weights"}
    ),
    "negative_selection": frozenset({"distribution", "mean", "std", "min_k", "max_k"}),
    "feature_delta": frozenset({"mode", "neg_delta_ratio"}),
    "feature_prediction": frozenset({"weight", "layer", "loss_mode", "loss_fn", "block", "head_layers"}),
    "sae": frozenset({"release", "model", "activation_type", "layer", "width", "l0"}),
}


def _strict_json_value_v2(value: Any, *, surface: str) -> Any:
    if isinstance(value, Mapping):
        canonical: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if not isinstance(key, str):
                raise RemoteTrainingError(f"{surface} contains non-string key {key!r}")
            canonical[key] = _strict_json_value_v2(item, surface=f"{surface}.{key}")
        return canonical
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_value_v2(item, surface=f"{surface}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        raise RemoteTrainingError(f"{surface} contains non-finite float {value!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RemoteTrainingError(f"{surface} contains unsupported value {type(value).__name__}")


def _validate_declared_nested_v2(value: Any, *, key: str, surface: str) -> None:
    if value is None:
        return
    if key not in _DECLARED_NESTED_KEYS_V2:
        if isinstance(value, Mapping):
            raise RemoteTrainingError(f"{surface}.{key} contains an unclassified nested mapping")
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                _validate_declared_nested_v2(
                    item,
                    key=key,
                    surface=f"{surface}.{key}[{index}]",
                )
        return
    if not isinstance(value, Mapping):
        raise RemoteTrainingError(f"{surface}.{key} must be a mapping or null")
    allowed = _DECLARED_NESTED_KEYS_V2[key]
    unknown = set(value) - allowed
    if unknown:
        raise RemoteTrainingError(
            f"{surface}.{key} contains unclassified nested keys {sorted(unknown, key=str)}"
        )
    for nested_key, nested_value in value.items():
        if not isinstance(nested_key, str):
            raise RemoteTrainingError(f"{surface}.{key} contains a non-string nested key")
        _validate_declared_nested_v2(
            nested_value,
            key=nested_key,
            surface=f"{surface}.{key}",
        )


def _semantic_config_v2(value: Mapping[str, Any], *, surface: str) -> dict[str, Any]:
    categories_by_key: dict[str, list[str]] = {}
    for category, keys in _CONFIG_CATEGORIES_V2.items():
        for key in keys:
            categories_by_key.setdefault(key, []).append(category)
    duplicate = {key: cats for key, cats in categories_by_key.items() if len(cats) != 1}
    if duplicate:
        raise RemoteTrainingError(f"semantic fingerprint classifier is ambiguous: {duplicate!r}")

    for key in value:
        if not isinstance(key, str):
            raise RemoteTrainingError(f"{surface} contains non-string root key {key!r}")
    canonical: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        categories = categories_by_key.get(key, [])
        if len(categories) != 1:
            raise RemoteTrainingError(
                f"{surface}.{key} is not classified by semantic fingerprint schema v2"
            )
        _validate_declared_nested_v2(item, key=key, surface=surface)
        if key == "conditioning_cooldown" and parse_conditioning_cooldown(item) is None:
            # Disabled schedules preserve the pre-cooldown semantic identity,
            # including callers that pass a raw composed Hydra mapping.
            continue
        canonical_item = _strict_json_value_v2(item, surface=f"{surface}.{key}")
        if categories[0] == "semantic":
            canonical[key] = canonical_item
    if "decoder_segment_size" in value and value["decoder_segment_size"] is not None:
        raise RemoteTrainingError(
            "training_config.decoder_segment_size is an unsupported placeholder "
            "for remote training and must remain null"
        )
    return canonical


@beartype
def build_remote_semantic_fingerprint_v2(
    *,
    resolved_training_config: _SemanticConfigMapping,
    world_size: PositiveIntLike,
    selection_contracts: _SelectionContracts,
) -> str:
    """Build the exhaustive schema-v2 semantic identity without changing run lineage."""
    if set(selection_contracts) != {"train", "derived_val", "base_val"}:
        raise RemoteTrainingError(
            "semantic fingerprint v2 requires train, derived_val, and base_val selection contracts"
        )
    payload = {
        "schema_version": REMOTE_SEMANTIC_SCHEMA_VERSION,
        "context_length": 1024,
        "ddp_world_size": world_size,
        "training_config": _semantic_config_v2(
            resolved_training_config,
            surface="training_config",
        ),
        "selection_contracts": {
            label: _strict_json_value_v2(
                selection_contracts[label],
                surface=f"selection_contracts.{label}",
            )
            for label in ("train", "derived_val", "base_val")
        },
    }
    return _sha256_json(payload)


@beartype
def build_remote_semantic_identity(
    fingerprint: LowercaseSha256Like,
) -> dict[str, Any]:
    return {
        "schema_version": REMOTE_SEMANTIC_SCHEMA_VERSION,
        "fingerprint": fingerprint,
    }


def parse_remote_semantic_identity(raw: Any, *, surface: str) -> str:
    if not isinstance(raw, Mapping) or set(raw) != {"schema_version", "fingerprint"}:
        raise RemoteTrainingError(f"{surface} must contain exactly schema_version and fingerprint")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != REMOTE_SEMANTIC_SCHEMA_VERSION
    ):
        raise RemoteTrainingError(f"{surface} has unsupported schema_version")
    return _strict_sha256(raw["fingerprint"], surface=f"{surface}.fingerprint")


@dataclass
class RemoteCheckpointStore:
    backend: BucketBackend
    remote: RemoteTrainingConfig
    checkpoint_dir: Path
    checkpoint_root: Path
    remote_run_fingerprint: str
    world_size: int
    semantic_fingerprint_v2: str | None = None
    resolved_training_config: Mapping[str, Any] | None = None
    selection_contracts: Mapping[str, Mapping[str, Any]] | None = None
    compatibility_mode: str = "direct_v2"
    _resolved_checkpoint_lineage: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size <= 0:
            raise RemoteTrainingError("checkpoint store world_size must be a positive strict int")
        initial_lineage = _strict_sha256(
            self.remote_run_fingerprint,
            surface="initial checkpoint lineage",
        )
        semantic_inputs = (
            self.semantic_fingerprint_v2,
            self.resolved_training_config,
            self.selection_contracts,
        )
        if any(value is not None for value in semantic_inputs) and not all(
            value is not None for value in semantic_inputs
        ):
            raise RemoteTrainingError(
                "schema-v2 checkpoint compatibility requires semantic_fingerprint_v2, "
                "resolved_training_config, and selection_contracts together"
            )
        if self.semantic_fingerprint_v2 is not None:
            semantic = _strict_sha256(
                self.semantic_fingerprint_v2,
                surface="semantic_fingerprint_v2",
            )
            if initial_lineage != semantic:
                raise RemoteTrainingError(
                    "a schema-v2 checkpoint store must start with its semantic "
                    "fingerprint as the unresolved lineage"
                )
            assert self.resolved_training_config is not None
            assert self.selection_contracts is not None
            recomputed = build_remote_semantic_fingerprint_v2(
                resolved_training_config=self.resolved_training_config,
                world_size=self.world_size,
                selection_contracts=self.selection_contracts,
            )
            if recomputed != semantic:
                raise RemoteTrainingError(
                    "semantic_fingerprint_v2 does not match the supplied "
                    "training config, world size, and selection contracts"
                )

    @property
    def remote_dir(self) -> str:
        relative = _relative_local_dir(
            self.checkpoint_dir,
            self.checkpoint_root,
            surface="checkpoint directory",
        )
        return _join_remote(self.remote.checkpoint_prefix, relative.as_posix())

    def _marker_name(self, step: int) -> str:
        return f"remote_complete_{step:06d}.json"

    @staticmethod
    def _local_marker_matches(
        marker_path: Path,
        marker: Mapping[str, Any],
    ) -> bool:
        if marker_path.is_symlink() or not marker_path.is_file():
            return False
        try:
            raw = json.loads(marker_path.read_text(encoding="utf-8"))
            parsed = parse_checkpoint_marker(
                raw,
                surface=f"local checkpoint marker {marker_path}",
            )
        except (OSError, UnicodeError, json.JSONDecodeError, RemoteTrainingError):
            return False
        return parsed == marker

    def _read_remote_marker(self, remote_path: str) -> dict[str, Any]:
        tmp = self.checkpoint_dir / (
            f".{Path(remote_path).name}.read.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        try:
            _download_checkpoint_with_retry(
                self.backend,
                remote_path=remote_path,
                final_path=tmp,
                timeout_seconds=self.remote.checkpoint_retry_timeout_seconds,
                retry_interval_seconds=self.remote.retry_interval_seconds,
            )
            try:
                raw = json.loads(tmp.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RemoteTrainingError(
                    f"remote checkpoint marker {remote_path} is invalid JSON"
                ) from exc
            return parse_checkpoint_marker(raw, surface=f"remote checkpoint marker {remote_path}")
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _marker_mirror_candidate(
        self,
        remote_path: str,
        *,
        expected: RemoteObjectInfo,
    ) -> dict[str, Any] | None:
        """Parse the local mirror of one committed marker, or report a cache miss.

        A non-None result is proven byte-identical to the remote object named by
        the current listing snapshot (exact size + Xet identity through one pinned
        descriptor), so parsing it is equivalent to downloading and parsing the
        remote object. Every identity shortfall — absent path, non-regular file,
        extra hard links, size or Xet mismatch, mutation while reading — is a
        MISS (returns None), never an error: the caller falls back to the
        verified-download path, which re-derives the same content from the
        authoritative remote object. Errors raised after byte identity is proven
        (invalid JSON, marker-schema violations) are the REMOTE object's own
        errors and use the remote surface string for exact message parity.
        """
        if not hasattr(os, "O_NOFOLLOW"):
            raise RemoteTrainingError("checkpoint marker mirror reads require os.O_NOFOLLOW")
        mirror = self.checkpoint_dir / PurePosixPath(remote_path).name
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(mirror, flags)
        except OSError:
            return None
        try:
            before = os.fstat(fd)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                return None
            if before.st_size != expected.byte_length:
                return None
            with os.fdopen(fd, "rb", closefd=False) as handle:
                data = handle.read()
            after = os.fstat(fd)
            if (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ) != identity or len(data) != expected.byte_length:
                return None
            xet_result = cast(Any, hf_xet).hash_files([f"/proc/self/fd/{fd}"])[0]
            final = os.fstat(fd)
            if (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
                final.st_ctime_ns,
            ) != identity or int(xet_result.file_size) != expected.byte_length:
                return None
            if str(xet_result.hash) != expected.xet_hash:
                return None
        except OSError:
            return None
        finally:
            os.close(fd)
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RemoteTrainingError(
                f"remote checkpoint marker {remote_path} is invalid JSON"
            ) from exc
        return parse_checkpoint_marker(raw, surface=f"remote checkpoint marker {remote_path}")

    def _read_leaf_marker(
        self,
        remote_path: str,
        *,
        expected: RemoteObjectInfo,
    ) -> dict[str, Any]:
        """Read one committed marker for the leaf walk, via mirror or download.

        Cache hits perform zero remote calls. Misses download through the existing
        verified path with the listing identity pinned as ``expected`` (no
        per-marker stat). Downloaded bytes are adopted as the local mirror ONLY
        when no local file exists at the marker path: an existing-but-different
        local marker is an uncommitted/stale artifact whose replacement is owned
        exclusively by publish()/restore() reconciliation (documented ordering:
        replacement only after local data files verify).
        """
        if _SHA256_RE.fullmatch(expected.xet_hash) is None:
            raise RemoteTrainingError(
                "checkpoint marker listing entry has a malformed Xet identity "
                f"(bucket API anomaly): path={remote_path} xet_hash={expected.xet_hash!r}"
            )
        cached = self._marker_mirror_candidate(remote_path, expected=expected)
        if cached is not None:
            return cached
        mirror = self.checkpoint_dir / PurePosixPath(remote_path).name
        tmp = self.checkpoint_dir / (f".{mirror.name}.read.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            _download_checkpoint_with_retry(
                self.backend,
                remote_path=remote_path,
                final_path=tmp,
                timeout_seconds=self.remote.checkpoint_retry_timeout_seconds,
                retry_interval_seconds=self.remote.retry_interval_seconds,
                expected=expected,
            )
            try:
                raw = json.loads(tmp.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RemoteTrainingError(
                    f"remote checkpoint marker {remote_path} is invalid JSON"
                ) from exc
            marker = parse_checkpoint_marker(raw, surface=f"remote checkpoint marker {remote_path}")
            try:
                os.link(tmp, mirror)
            except FileExistsError:
                logger.info(f"remote_training.marker_mirror_stale path={remote_path}")
            except OSError as exc:
                logger.warning(
                    f"remote_training.marker_mirror_adopt_failed path={remote_path} error={exc!r}"
                )
            else:
                logger.info(f"remote_training.marker_mirror_adopted path={remote_path}")
            return marker
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _resolve_committed_group(
        self,
        marker: Mapping[str, Any],
    ) -> _ResolvedCheckpointGroup:
        parsed = parse_checkpoint_marker(marker, surface="committed checkpoint")
        aggregate = parsed["aggregate_sha256"]
        generation_dir = _join_remote(self.remote_dir, "_groups", aggregate)
        generation_paths = {
            entry["name"]: _join_remote(generation_dir, entry["name"]) for entry in parsed["files"]
        }
        generation_infos = {
            name: self.backend.stat(remote_path) for name, remote_path in generation_paths.items()
        }
        if any(info is not None for info in generation_infos.values()):
            layout = "immutable_generation"
            paths = generation_paths
            infos = generation_infos
        else:
            layout = "legacy_flat"
            paths = {
                entry["name"]: _join_remote(self.remote_dir, entry["name"])
                for entry in parsed["files"]
            }
            infos = {name: self.backend.stat(remote_path) for name, remote_path in paths.items()}

        resolved: dict[str, RemoteObjectInfo] = {}
        entries = {entry["name"]: entry for entry in parsed["files"]}
        for name, entry in entries.items():
            remote_path = paths[name]
            info = infos[name]
            if info is None:
                raise RemoteTrainingError(
                    f"marked checkpoint group layout={layout} is missing {remote_path}"
                )
            if (
                info.path != remote_path
                or info.byte_length != entry["byte_length"]
                or info.xet_hash != entry["xet_hash"]
            ):
                raise RemoteTrainingError(
                    f"marked checkpoint object identity drift layout={layout} path={remote_path}"
                )
            resolved[name] = info
        logger.info(
            "remote_training.checkpoint_group_resolved "
            f"layout={layout} aggregate={aggregate} step={parsed['step']}"
        )
        return _ResolvedCheckpointGroup(
            layout=layout,
            aggregate_sha256=aggregate,
            objects=resolved,
        )

    def _assert_remote_group(self, marker: Mapping[str, Any]) -> None:
        self._resolve_committed_group(marker)

    def _assert_group_in_listing(
        self,
        marker: Mapping[str, Any],
        listing: Mapping[str, RemoteObjectInfo],
    ) -> None:
        """Enforce committed-group completeness/drift from one listing snapshot.

        Byte-for-byte the same checks and error messages as
        _resolve_committed_group, with object identity sourced from the walk's
        single recursive listing instead of one exact-path stat per file. The
        listing is already the sole authority for marker discovery, so this adds
        no new trust source.
        """
        parsed = parse_checkpoint_marker(marker, surface="committed checkpoint")
        aggregate = parsed["aggregate_sha256"]
        generation_dir = _join_remote(self.remote_dir, "_groups", aggregate)
        generation_paths = {
            entry["name"]: _join_remote(generation_dir, entry["name"]) for entry in parsed["files"]
        }
        generation_infos = {
            name: listing.get(remote_path) for name, remote_path in generation_paths.items()
        }
        if any(info is not None for info in generation_infos.values()):
            layout = "immutable_generation"
            paths = generation_paths
            infos = generation_infos
        else:
            layout = "legacy_flat"
            paths = {
                entry["name"]: _join_remote(self.remote_dir, entry["name"])
                for entry in parsed["files"]
            }
            infos = {name: listing.get(remote_path) for name, remote_path in paths.items()}
        entries = {entry["name"]: entry for entry in parsed["files"]}
        for name, entry in entries.items():
            remote_path = paths[name]
            info = infos[name]
            if info is None:
                raise RemoteTrainingError(
                    f"marked checkpoint group layout={layout} is missing {remote_path}"
                )
            if (
                info.path != remote_path
                or info.byte_length != entry["byte_length"]
                or info.xet_hash != entry["xet_hash"]
            ):
                raise RemoteTrainingError(
                    f"marked checkpoint object identity drift layout={layout} path={remote_path}"
                )
        logger.info(
            "remote_training.checkpoint_group_resolved "
            f"layout={layout} aggregate={aggregate} step={parsed['step']}"
        )

    def _read_authenticated_metadata(self, marker: Mapping[str, Any]) -> dict[str, Any]:
        meta_entry = next(
            (entry for entry in marker["files"] if str(entry["name"]).startswith("meta_")),
            None,
        )
        if meta_entry is None:
            raise RemoteTrainingError("checkpoint marker has no metadata entry")
        resolved = self._resolve_committed_group(marker)
        remote_info = resolved.objects[meta_entry["name"]]
        remote_path = remote_info.path
        tmp = self.checkpoint_dir / (
            f".{meta_entry['name']}.compat.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            _download_checkpoint_with_retry(
                self.backend,
                remote_path=remote_path,
                final_path=tmp,
                timeout_seconds=self.remote.checkpoint_retry_timeout_seconds,
                retry_interval_seconds=self.remote.retry_interval_seconds,
                expected=remote_info,
                expected_byte_length=meta_entry["byte_length"],
                expected_content_sha256=meta_entry["content_sha256"],
            )
            _verify_file(
                tmp,
                byte_length=meta_entry["byte_length"],
                content_sha256=meta_entry["content_sha256"],
                xet_hash=meta_entry["xet_hash"],
            )
            try:
                raw = json.loads(tmp.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RemoteTrainingError(
                    f"authenticated checkpoint metadata {remote_path} is invalid JSON"
                ) from exc
            if not isinstance(raw, dict):
                raise RemoteTrainingError(
                    f"authenticated checkpoint metadata {remote_path} must be a mapping"
                )
            return raw
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _resolve_checkpoint_lineage(self, marker: Mapping[str, Any]) -> None:
        marker_lineage = str(marker["remote_run_fingerprint"])
        if (
            self._resolved_checkpoint_lineage is not None
            and marker_lineage != self._resolved_checkpoint_lineage
        ):
            raise RemoteTrainingError(
                "checkpoint lineage is immutable after compatibility resolution: "
                f"resolved={self._resolved_checkpoint_lineage}, marker={marker_lineage}"
            )
        expected_semantic = self.semantic_fingerprint_v2 or self.remote_run_fingerprint
        if marker_lineage == expected_semantic:
            if self.semantic_fingerprint_v2 is not None:
                assert self.selection_contracts is not None
                meta = self._read_authenticated_metadata(marker)
                if meta.get(REMOTE_RUN_META_KEY) != marker_lineage:
                    raise RemoteTrainingError(
                        "checkpoint marker/meta lineage disagreement in direct-v2 mode"
                    )
                stored_semantic = parse_remote_semantic_identity(
                    meta.get(REMOTE_SEMANTIC_META_KEY),
                    surface=f"checkpoint metadata {REMOTE_SEMANTIC_META_KEY}",
                )
                if stored_semantic != expected_semantic:
                    raise RemoteTrainingError(
                        "direct-v2 checkpoint semantic metadata does not match "
                        "the configured semantic fingerprint"
                    )
                saved_config_raw = meta.get("user_config")
                if not isinstance(saved_config_raw, Mapping):
                    raise RemoteTrainingError(
                        "direct-v2 compatibility requires checkpoint user_config"
                    )
                saved_config = dict(saved_config_raw)
                saved_config.setdefault(
                    "initialize_encoder_embeddings_cluster_vectors",
                    "mean",
                )
                saved_semantic = build_remote_semantic_fingerprint_v2(
                    resolved_training_config=saved_config,
                    world_size=self.world_size,
                    selection_contracts=self.selection_contracts,
                )
                if saved_semantic != expected_semantic:
                    raise RemoteTrainingError(
                        "direct-v2 checkpoint saved config does not match "
                        "the configured semantic fingerprint"
                    )
            self.remote_run_fingerprint = marker_lineage
            self.compatibility_mode = "direct_v2"
            self._resolved_checkpoint_lineage = marker_lineage
            return
        if self.resolved_training_config is None or self.selection_contracts is None:
            raise RemoteTrainingError(
                "checkpoint remote-run fingerprint differs from semantic fingerprint v2, "
                "but legacy compatibility inputs were not provided"
            )

        meta = self._read_authenticated_metadata(marker)
        if meta.get(REMOTE_RUN_META_KEY) != marker_lineage:
            raise RemoteTrainingError(
                "checkpoint marker/meta lineage disagreement: "
                f"marker={marker_lineage!r}, metadata={meta.get(REMOTE_RUN_META_KEY)!r}"
            )
        saved_config_raw = meta.get("user_config")
        if not isinstance(saved_config_raw, Mapping):
            raise RemoteTrainingError("legacy compatibility requires checkpoint user_config")
        saved_config = dict(saved_config_raw)
        saved_config.setdefault(
            "initialize_encoder_embeddings_cluster_vectors",
            "mean",
        )

        stored_semantic_raw = meta.get(REMOTE_SEMANTIC_META_KEY)
        if stored_semantic_raw is not None:
            stored_semantic = parse_remote_semantic_identity(
                stored_semantic_raw,
                surface=f"checkpoint metadata {REMOTE_SEMANTIC_META_KEY}",
            )
            saved_semantic = build_remote_semantic_fingerprint_v2(
                resolved_training_config=saved_config,
                world_size=self.world_size,
                selection_contracts=self.selection_contracts,
            )
            if stored_semantic != saved_semantic or stored_semantic != expected_semantic:
                raise RemoteTrainingError(
                    "checkpoint semantic fingerprint v2 mismatch: "
                    f"stored={stored_semantic}, saved_config={saved_semantic}, "
                    f"current={expected_semantic}"
                )
        else:
            reconstructed_legacy = build_remote_run_fingerprint(
                resolved_training_config=saved_config,
                world_size=self.world_size,
                selection_contracts=self.selection_contracts,
            )
            if reconstructed_legacy != marker_lineage:
                raise RemoteTrainingError(
                    "legacy checkpoint fingerprint reconstruction mismatch: "
                    f"marker={marker_lineage}, reconstructed={reconstructed_legacy}"
                )
            saved_semantic = build_remote_semantic_fingerprint_v2(
                resolved_training_config=saved_config,
                world_size=self.world_size,
                selection_contracts=self.selection_contracts,
            )
            if saved_semantic != expected_semantic:
                raise RemoteTrainingError(
                    "legacy checkpoint is not semantically compatible with the current run: "
                    f"checkpoint_v2={saved_semantic}, current_v2={expected_semantic}"
                )

        self.remote_run_fingerprint = marker_lineage
        self.compatibility_mode = "legacy_v1_migrated"
        self._resolved_checkpoint_lineage = marker_lineage
        logger.info(
            "remote_training compatibility resolved: "
            f"semantic_fingerprint_v2={expected_semantic} "
            f"checkpoint_lineage={marker_lineage} "
            f"compatibility_mode={self.compatibility_mode}"
        )

    def _assert_local_checkpoint_metadata(self, step: int) -> None:
        meta_path = self.checkpoint_dir / f"meta_{step:06d}.json"
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RemoteTrainingError(
                f"local checkpoint metadata {meta_path} is invalid JSON"
            ) from exc
        if not isinstance(raw, Mapping):
            raise RemoteTrainingError(f"local checkpoint metadata {meta_path} must be a mapping")
        if raw.get(REMOTE_RUN_META_KEY) != self.remote_run_fingerprint:
            raise RemoteTrainingError(
                "local checkpoint metadata lineage does not match the active checkpoint lineage"
            )
        if self.semantic_fingerprint_v2 is None:
            return
        stored_semantic = parse_remote_semantic_identity(
            raw.get(REMOTE_SEMANTIC_META_KEY),
            surface=f"local checkpoint metadata {REMOTE_SEMANTIC_META_KEY}",
        )
        if stored_semantic != self.semantic_fingerprint_v2:
            raise RemoteTrainingError(
                "local checkpoint semantic metadata does not match semantic_fingerprint_v2"
            )
        saved_config_raw = raw.get("user_config")
        if not isinstance(saved_config_raw, Mapping):
            raise RemoteTrainingError("schema-v2 local checkpoint metadata requires user_config")
        saved_config = dict(saved_config_raw)
        saved_config.setdefault(
            "initialize_encoder_embeddings_cluster_vectors",
            "mean",
        )
        assert self.selection_contracts is not None
        saved_semantic = build_remote_semantic_fingerprint_v2(
            resolved_training_config=saved_config,
            world_size=self.world_size,
            selection_contracts=self.selection_contracts,
        )
        if saved_semantic != self.semantic_fingerprint_v2:
            raise RemoteTrainingError(
                "local checkpoint saved config does not match semantic_fingerprint_v2"
            )

    def _assert_local_group_identity(
        self,
        step: int,
        expected: tuple[_CheckpointFileIdentity, ...],
    ) -> None:
        try:
            actual = _checkpoint_group_identity(
                self.checkpoint_dir,
                step=step,
                world_size=self.world_size,
            )
        except RemoteTrainingError as exc:
            raise RemoteTrainingError(
                "checkpoint group changed after asynchronous publication enqueue: "
                f"step={step}; cannot recapture the required direct files: {exc}"
            ) from exc
        if actual != expected:
            changed = [
                expected_entry.name
                for expected_entry, actual_entry in zip(expected, actual, strict=True)
                if expected_entry != actual_entry
            ]
            raise RemoteTrainingError(
                "checkpoint group changed after asynchronous publication enqueue: "
                f"step={step}, files={changed}"
            )

    def publish(
        self,
        step: int,
        *,
        expected_local_group: tuple[_CheckpointFileIdentity, ...] | None = None,
    ) -> dict[str, Any]:
        self.latest_complete_step()
        self._assert_local_checkpoint_metadata(step)
        if expected_local_group is not None:
            self._assert_local_group_identity(step, expected_local_group)
        marker = build_checkpoint_marker(
            self.checkpoint_dir,
            step=step,
            world_size=self.world_size,
            remote_run_fingerprint=self.remote_run_fingerprint,
        )
        if expected_local_group is not None:
            self._assert_local_group_identity(step, expected_local_group)
        marker_name = self._marker_name(step)
        marker_path = self.checkpoint_dir / marker_name
        remote_marker_path = _join_remote(self.remote_dir, marker_name)
        if self.backend.stat(remote_marker_path) is not None:
            prior = self._read_remote_marker(remote_marker_path)
            if prior != marker:
                raise RemoteTrainingError(
                    f"same-step remote checkpoint collision at {remote_marker_path}"
                )
            resolved = self._resolve_committed_group(prior)
            if expected_local_group is not None:
                self._assert_local_group_identity(step, expected_local_group)
            replaced_uncommitted_local_marker = (
                marker_path.exists() or marker_path.is_symlink()
            ) and not self._local_marker_matches(marker_path, prior)
            atomic_json_dump(prior, marker_path)
            logger.info(
                "remote_training.checkpoint_reconciled "
                f"layout={resolved.layout} aggregate={resolved.aggregate_sha256} "
                f"step={step} "
                "replaced_uncommitted_local_marker="
                f"{replaced_uncommitted_local_marker}"
            )
            return marker

        replaced_uncommitted_local_marker = marker_path.exists() or marker_path.is_symlink()
        if replaced_uncommitted_local_marker:
            logger.warning(
                "remote_training replacing uncommitted local checkpoint marker "
                f"step={step} path={marker_path}"
            )
        atomic_json_dump(marker, marker_path)
        generation_dir = _join_remote(
            self.remote_dir,
            "_groups",
            marker["aggregate_sha256"],
        )
        for entry in marker["files"]:
            if expected_local_group is not None:
                self._assert_local_group_identity(step, expected_local_group)
            local_path = self.checkpoint_dir / entry["name"]
            remote_path = _join_remote(generation_dir, entry["name"])
            _upload_with_retry(
                self.backend,
                local_path=local_path,
                remote_path=remote_path,
                byte_length=entry["byte_length"],
                xet_hash=entry["xet_hash"],
                timeout_seconds=self.remote.checkpoint_retry_timeout_seconds,
                retry_interval_seconds=self.remote.retry_interval_seconds,
            )
            if expected_local_group is not None:
                self._assert_local_group_identity(step, expected_local_group)
        marker_size, _, _, marker_xet = _stable_file_identity(marker_path)
        if expected_local_group is not None:
            self._assert_local_group_identity(step, expected_local_group)
        _upload_with_retry(
            self.backend,
            local_path=marker_path,
            remote_path=remote_marker_path,
            byte_length=marker_size,
            xet_hash=marker_xet,
            timeout_seconds=self.remote.checkpoint_retry_timeout_seconds,
            retry_interval_seconds=self.remote.retry_interval_seconds,
        )
        if expected_local_group is not None:
            self._assert_local_group_identity(step, expected_local_group)
        logger.info(f"remote_training.checkpoint_committed step={step} path={remote_marker_path}")
        logger.info(
            "remote_training.checkpoint_publication "
            f"layout=immutable_generation aggregate={marker['aggregate_sha256']} "
            f"step={step} "
            "replaced_uncommitted_local_marker="
            f"{replaced_uncommitted_local_marker}"
        )
        return marker

    def latest_complete_step(self) -> int | None:
        listing: dict[str, RemoteObjectInfo] = {}
        markers: list[tuple[int, str]] = []
        prefix = self.remote_dir + "/"
        for info in self.backend.list(self.remote_dir):
            if not info.path.startswith(prefix):
                raise RemoteTrainingError(f"bucket listing escaped checkpoint prefix: {info.path}")
            prior = listing.get(info.path)
            if prior is not None and prior != info:
                raise RemoteTrainingError(
                    f"bucket listing repeated a path with conflicting identity: {info.path}"
                )
            listing[info.path] = info
            relative = info.path[len(prefix) :]
            if "/" in relative:
                continue
            match = REMOTE_CHECKPOINT_MARKER_RE.fullmatch(relative)
            if match is not None:
                markers.append((int(match.group("step")), info.path))
            elif relative.startswith("remote_complete_"):
                raise RemoteTrainingError(
                    f"malformed checkpoint completion-marker filename: {info.path}"
                )
        if not markers:
            return None
        parsed_by_step: dict[int, dict[str, Any]] = {}
        checkpoint_lineages: set[str] = set()
        for filename_step, path in sorted(markers):
            marker = self._read_leaf_marker(path, expected=listing[path])
            if marker["step"] != filename_step:
                raise RemoteTrainingError(f"marker filename/payload step mismatch at {path}")
            if marker["ddp_world_size"] != self.world_size:
                raise RemoteTrainingError(f"checkpoint world-size drift at {path}")
            if filename_step in parsed_by_step:
                raise RemoteTrainingError(f"duplicate completion marker for step {filename_step}")
            self._assert_group_in_listing(marker, listing)
            checkpoint_lineages.add(marker["remote_run_fingerprint"])
            parsed_by_step[filename_step] = marker
        if len(checkpoint_lineages) != 1:
            raise RemoteTrainingError(
                f"checkpoint leaf contains mixed lineages: {sorted(checkpoint_lineages)}"
            )
        latest_step = max(parsed_by_step)
        self._resolve_checkpoint_lineage(parsed_by_step[latest_step])
        if checkpoint_lineages != {self.remote_run_fingerprint}:
            raise RemoteTrainingError("checkpoint lineage resolution did not cover the full leaf")
        return latest_step

    def restore(self, step: int) -> dict[str, Any]:
        if self.latest_complete_step() is None:
            raise RemoteTrainingError("checkpoint leaf contains no completion markers")
        marker_remote = _join_remote(self.remote_dir, self._marker_name(step))
        marker = self._read_remote_marker(marker_remote)
        if marker["step"] != step:
            raise RemoteTrainingError("checkpoint marker step mismatch")
        if marker["ddp_world_size"] != self.world_size:
            raise RemoteTrainingError("checkpoint marker world-size mismatch")
        self._assert_remote_group(marker)
        self._resolve_checkpoint_lineage(marker)
        if marker["remote_run_fingerprint"] != self.remote_run_fingerprint:
            raise RemoteTrainingError("checkpoint marker remote-run fingerprint mismatch")
        resolved = self._resolve_committed_group(marker)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for entry in marker["files"]:
            remote_info = resolved.objects[entry["name"]]
            final = self.checkpoint_dir / entry["name"]
            if final.exists() or final.is_symlink():
                _verify_file(
                    final,
                    byte_length=entry["byte_length"],
                    content_sha256=entry["content_sha256"],
                    xet_hash=entry["xet_hash"],
                )
                continue
            _download_checkpoint_with_retry(
                self.backend,
                remote_path=remote_info.path,
                final_path=final,
                timeout_seconds=self.remote.checkpoint_retry_timeout_seconds,
                retry_interval_seconds=self.remote.retry_interval_seconds,
                expected=remote_info,
                expected_byte_length=entry["byte_length"],
                expected_content_sha256=entry["content_sha256"],
            )
            _verify_file(
                final,
                byte_length=entry["byte_length"],
                content_sha256=entry["content_sha256"],
                xet_hash=entry["xet_hash"],
            )
        local_marker = self.checkpoint_dir / self._marker_name(step)
        replaced_uncommitted_local_marker = (
            local_marker.exists() or local_marker.is_symlink()
        ) and not self._local_marker_matches(local_marker, marker)
        atomic_json_dump(marker, local_marker)
        logger.info(
            "remote_training.checkpoint_restored "
            f"layout={resolved.layout} aggregate={resolved.aggregate_sha256} "
            f"step={step} "
            "replaced_uncommitted_local_marker="
            f"{replaced_uncommitted_local_marker}"
        )
        return marker


class RemoteCheckpointPublisher:
    """Publish at most one immutable checkpoint group on one owned worker."""

    def __init__(self, store: RemoteCheckpointStore) -> None:
        self.store = store
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="remote-checkpoint-publish",
        )
        self._future: Future[dict[str, Any]] | None = None
        self._pending_step: int | None = None
        self._closed = False

    @property
    def pending_step(self) -> int | None:
        return self._pending_step

    def _publish(
        self,
        step: int,
        expected_local_group: tuple[_CheckpointFileIdentity, ...],
    ) -> dict[str, Any]:
        started = time.monotonic()
        logger.info(f"remote_training.checkpoint_publish_start step={step}")
        try:
            marker = self.store.publish(
                step,
                expected_local_group=expected_local_group,
            )
        except BaseException as exc:
            elapsed = time.monotonic() - started
            logger.error(
                "remote_training.checkpoint_publish_failed "
                f"step={step} elapsed_seconds={elapsed:.3f} error={exc!r}"
            )
            raise
        elapsed = time.monotonic() - started
        logger.info(
            f"remote_training.checkpoint_publish_complete step={step} elapsed_seconds={elapsed:.3f}"
        )
        return marker

    def submit(self, step: int) -> int:
        if self._closed:
            raise RemoteTrainingError("checkpoint publication submitted after publisher close")
        if self._future is not None:
            raise RemoteTrainingError(
                "checkpoint publication capacity is full: "
                f"pending_step={self._pending_step}, submitted_step={step}"
            )
        expected_local_group = _checkpoint_group_identity(
            self.store.checkpoint_dir,
            step=step,
            world_size=self.store.world_size,
        )
        self._future = self._executor.submit(
            self._publish,
            step,
            expected_local_group,
        )
        self._pending_step = step
        logger.info(f"remote_training.checkpoint_enqueue step={step}")
        return step

    def poll(self) -> bool:
        future = self._future
        if future is None:
            return False
        if not future.done():
            return True
        self._future = None
        self._pending_step = None
        future.result()
        return False

    def close(self) -> None:
        if self._closed:
            return
        future = self._future
        self._future = None
        self._pending_step = None
        try:
            if future is not None:
                future.result()
        finally:
            self._executor.shutdown(wait=True)
            self._closed = True


def assert_checkpoint_remote_run(meta_data: Any, expected_fingerprint: str) -> None:
    if not isinstance(meta_data, Mapping):
        raise RemoteTrainingError("checkpoint metadata must be a mapping")
    actual = meta_data.get(REMOTE_RUN_META_KEY)
    if actual != expected_fingerprint:
        raise RemoteTrainingError(
            "checkpoint remote-run contract mismatch: "
            f"checkpoint={actual!r}, configured={expected_fingerprint!r}"
        )


def ddp_collective_result(
    *,
    local_error: BaseException | None,
    world_size: int,
    rank: int,
) -> None:
    """Make every rank observe local-save failures before rank-zero upload."""
    if world_size == 1:
        if local_error is not None:
            raise local_error
        return
    gathered: list[dict[str, Any] | None] = [None] * world_size
    payload = (
        None
        if local_error is None
        else {
            "rank": rank,
            "type": type(local_error).__name__,
            "message": str(local_error),
        }
    )
    dist.all_gather_object(gathered, payload)
    failures = [item for item in gathered if item is not None]
    if failures:
        raise RemoteTrainingError(
            f"checkpoint local-save failure(s): {failures!r}"
        ) from local_error


def ddp_rank0_action(
    *,
    action: Any,
    world_size: int,
    rank: int,
) -> Any:
    """Run an action on rank zero and broadcast its result or exact failure."""
    if world_size == 1:
        return action()
    payload: list[dict[str, Any] | None] = [None]
    result: Any = None
    if rank == 0:
        try:
            result = action()
            payload[0] = {"ok": True, "result": result}
        except BaseException as exc:
            payload[0] = {
                "ok": False,
                "type": type(exc).__name__,
                "message": str(exc),
            }
    dist.broadcast_object_list(payload, src=0)
    outcome = payload[0]
    if not isinstance(outcome, Mapping):
        raise RemoteTrainingError("rank-zero remote action broadcast was malformed")
    if outcome.get("ok") is not True:
        raise RemoteTrainingError(
            f"rank-zero remote action failed: {outcome.get('type')}: {outcome.get('message')}"
        )
    return outcome.get("result")


def ddp_create_checkpoint_publisher(
    *,
    store: RemoteCheckpointStore | None,
    world_size: int,
    rank: int,
) -> RemoteCheckpointPublisher | None:
    """Collectively construct rank zero's publisher or expose its exact failure."""
    publisher: RemoteCheckpointPublisher | None = None

    def _create_rank_zero() -> None:
        nonlocal publisher
        if store is None:
            raise RemoteTrainingError("rank zero is missing its remote checkpoint store")
        publisher = RemoteCheckpointPublisher(store)

    try:
        ddp_rank0_action(
            action=_create_rank_zero,
            world_size=world_size,
            rank=rank,
        )
    except BaseException:
        if publisher is not None:
            publisher.close()
        raise
    return publisher


def ddp_poll_checkpoint_publisher(
    *,
    publisher: RemoteCheckpointPublisher | None,
    world_size: int,
    rank: int,
) -> bool:
    """Collectively poll rank zero's publisher without waiting on its worker."""

    def _poll_rank_zero() -> bool:
        if publisher is None:
            raise RemoteTrainingError("rank zero is missing its checkpoint publisher")
        return publisher.poll()

    pending = ddp_rank0_action(
        action=_poll_rank_zero,
        world_size=world_size,
        rank=rank,
    )
    if not isinstance(pending, bool):
        raise RemoteTrainingError(
            f"checkpoint publisher poll returned a non-bool result: {pending!r}"
        )
    return pending


def ddp_publish_saved_checkpoint(
    *,
    store: RemoteCheckpointStore,
    publisher: RemoteCheckpointPublisher | None,
    async_upload: bool,
    step: int,
    world_size: int,
    rank: int,
) -> bool:
    """Publish one collectively saved group and report whether async work is pending."""
    if not isinstance(async_upload, bool):
        raise RemoteTrainingError(
            f"async checkpoint publication mode must be a strict bool, got {async_upload!r}"
        )
    if not async_upload:
        ddp_rank0_action(
            action=lambda: store.publish(step),
            world_size=world_size,
            rank=rank,
        )
        return False

    def _enqueue_rank_zero() -> int:
        if publisher is None:
            raise RemoteTrainingError("rank zero is missing its checkpoint publisher")
        return publisher.submit(step)

    enqueued_step = ddp_rank0_action(
        action=_enqueue_rank_zero,
        world_size=world_size,
        rank=rank,
    )
    if type(enqueued_step) is not int or enqueued_step != step:
        raise RemoteTrainingError(
            "checkpoint publisher acknowledged the wrong step: "
            f"expected={step}, got={enqueued_step!r}"
        )
    return True


def ddp_drain_checkpoint_publisher(
    *,
    publisher: RemoteCheckpointPublisher | None,
    world_size: int,
    rank: int,
) -> None:
    """Collectively wait using short polls so no collective spans an upload."""
    wait_logged = False
    while ddp_poll_checkpoint_publisher(
        publisher=publisher,
        world_size=world_size,
        rank=rank,
    ):
        if not wait_logged and rank == 0:
            pending_step = publisher.pending_step if publisher is not None else None
            logger.info(f"remote_training.checkpoint_wait step={pending_step}")
            wait_logged = True
        time.sleep(1.0)
