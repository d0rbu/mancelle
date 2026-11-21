"""Shared identity primitives for clustered-evaluation feature contracts.

Provide feature-directory, statistics, and SAE identities; stable file digests;
canonical-JSON hashing; and schema/algorithm version constants used by
checkpoint pinning and the public ``cluster_feature_contract`` facade.

File hashing delegates to descriptor-bound I/O and records a six-field
``(mode, dev, ino, size, mtime_ns, ctime_ns)`` signature for later mutation
checks. Canonical JSON uses sorted string keys, compact separators, finite
floats, and unescaped non-ASCII text. Feature-ID tuples must be non-negative,
sorted, unique, and bounded by the original SAE width.

Checkpoint resolution, SAE loading, and contract assembly live in higher layers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from beartype import beartype

from errors import FeatureClustersError
from steering.feature_paths import FeatureDirectoryComponents
from steering.immutable_io import is_sha256_hex, stream_hash_file_descriptor_bound
from steering.refined_types import (
    LowercaseSha256Like,
    NonBlankStrLike,
    NonEmptyStrLike,
    NonNegativeIntLike,
    NonNegativeIntTupleLike,
    PositiveIntLike,
)
from steering.segmentation import SegmentMode

FEATURE_CONTRACT_SCHEMA_VERSION = 1
MAX_SAFE_SELECTED_FEATURE_WIDTH = 65_536

CONCEPT_PROJECTION_VERSION = "mask_first_min_allowed_representative_v1"
QC_TARGET_REDUCTION_VERSION = "member_token_amax_own_percentile_v1"
PACKED_PERCENTILE_SELECTION_VERSION = "training_uint8_roundtrip_cluster_amax_v1"
FEATURE_DIFFMEAN_PROJECTION_VERSION = "per_completion_uint8_amax_then_mean_v1"
GENERIC_ROOFLINE_SIGNAL_VERSION = "per_document_cluster_signal_v2"

_HASH_CHUNK_SIZE: Literal[8_388_608] = 8_388_608
_STAT_SIGNATURE_ARITY = 6


class ClusterFeatureContractError(FeatureClustersError):
    """A clustered-evaluation semantic or provenance contract failed."""


def _assert_canonical_json_domain(value: Any, *, path: str) -> None:
    """Recursively reject the two collision/format hazards ``json.dumps``
    would otherwise let through: non-``str`` mapping keys (which it silently
    *coerces* by default — ``{1: "x"}`` and ``{"1": "x"}`` would hash
    identically) and non-finite floats (which it renders, absent
    ``allow_nan=False``, as the non-standard ``NaN``/``Infinity``/
    ``-Infinity`` tokens). ``path`` is a JSONPath-like breadcrumb (``$``,
    ``$.key``, ``$[0]``) included in the raised message.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise ClusterFeatureContractError(
                    f"canonical JSON payload at {path} has a non-string key {key!r} "
                    f"({type(key).__name__}); only literal str keys are permitted "
                    "(int/float/bool/None keys silently coerce and collide under "
                    "json.dumps)"
                )
            _assert_canonical_json_domain(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_canonical_json_domain(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ClusterFeatureContractError(
            f"canonical JSON payload at {path} has a non-finite float {value!r}; "
            "NaN/Infinity are not permitted in a collision-safe contract"
        )


def _canonical_json(value: Any) -> str:
    """Collision-safe canonical JSON: sorted keys, compact separators, exact
    non-ASCII passthrough, string-only keys, finite floats only.

    ``-0.0`` policy: negative zero is intentionally **not** normalized to
    ``0.0``. ``json.dumps`` already serializes it deterministically and
    losslessly as the literal ``"-0.0"`` (distinct from ``"0.0"``) on every
    platform — two payloads differing only in float sign-of-zero therefore
    hash *differently*, on purpose, preserving the caller's exact numeric
    input rather than silently discarding a bit of information.
    """
    _assert_canonical_json_domain(value, path="$")
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except TypeError as exc:
        raise ClusterFeatureContractError(
            f"canonical JSON payload contains a value json.dumps cannot serialize: {exc}"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()




@dataclass(frozen=True)
class StableFileDigest:
    """A content digest tied to the file signature observed while hashing.

    ``stat_signature`` is ``(mode, dev, ino, size, mtime_ns, ctime_ns)``;
    mode distinguishes a substituted symlink from a regular file. The signature
    is in-memory provenance for same-process mutation checks through
    ``_assert_digest_stat_unchanged``. Semantic and persisted payloads use
    digest/length fields rather than this filesystem signature.
    """

    path: Path
    sha256: str
    byte_length: int
    stat_signature: tuple[int, int, int, int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ClusterFeatureContractError("file digest path must be a pathlib.Path")
        if not is_sha256_hex(self.sha256):
            raise ClusterFeatureContractError("file SHA-256 must be lowercase hexadecimal")
        if type(self.byte_length) is not int or self.byte_length < 0:
            raise ClusterFeatureContractError("file byte length must be non-negative")
        if (
            not isinstance(self.stat_signature, tuple)
            or len(self.stat_signature) != _STAT_SIGNATURE_ARITY
            or any(type(field) is not int for field in self.stat_signature)
        ):
            raise ClusterFeatureContractError(
                f"file stat signature must be a {_STAT_SIGNATURE_ARITY}-tuple of strict "
                "ints (mode, dev, ino, size, mtime_ns, ctime_ns)"
            )


def _stat_signature(path: Path) -> tuple[int, int, int, int, int, int]:
    """Pathname-based stat tuple for the same-process re-check in
    ``_assert_digest_stat_unchanged`` (no open descriptor exists by then —
    the fd-bound checks in ``stream_hash_file_stable``/
    ``stream_hash_file_descriptor_bound`` do not apply to a later, cold
    re-check of a previously captured digest).

    Uses ``lstat`` (not ``stat``) so a symlink substituted at ``path`` after
    the original hash is itself detected via the mode field, rather than
    transparently followed to its target.
    """
    result = path.lstat()
    return (
        int(result.st_mode),
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def stat_signature_now(path: Path) -> tuple[int, int, int, int, int, int]:
    """Return the shared six-field ``lstat`` signature for ``path`` now."""

    return _stat_signature(path)


_STREAM_HASH_ERROR_MESSAGES: dict[str, str] = {
    "not_a_regular_file": "contract input is not a regular file: {path}",
    "changed_before_open": "contract input changed while it was being hashed: {path}",
    "read_failed": "could not hash contract input {path}",
    "changed_during_read": "contract input changed while it was being hashed: {path}",
    "changed_after_read": "contract input changed while it was being hashed: {path}",
}


def stream_hash_file_stable(path: str | Path) -> StableFileDigest:
    """Stream-hash a regular file and reject mutation during the read.

    Delegates the actual lstat/open/fstat/read/fstat/lstat sequence to
    ``steering.immutable_io.stream_hash_file_descriptor_bound`` (the
    neutral, dependency-free IO primitive — mirrors
    ``steering.conditional.feature_clusters._file_content_identity``'s
    fd-binding) and converts its result into a validated
    ``StableFileDigest``, translating the helper's reason codes into this
    contract's exact wording via a caller-supplied exception factory. This
    keeps ``immutable_io`` free of any dependency on ``steering.eval``.
    """
    resolved = Path(path).expanduser().resolve(strict=True)

    def _make_error(reason: str) -> ClusterFeatureContractError:
        return ClusterFeatureContractError(
            _STREAM_HASH_ERROR_MESSAGES[reason].format(path=resolved)
        )

    streamed = stream_hash_file_descriptor_bound(
        resolved, on_error=_make_error, chunk_size=_HASH_CHUNK_SIZE
    )
    return StableFileDigest(
        path=streamed.path,
        sha256=streamed.sha256,
        byte_length=streamed.byte_length,
        stat_signature=streamed.stat_signature,
    )


def _assert_digest_stat_unchanged(digest: StableFileDigest) -> None:
    try:
        current = _stat_signature(digest.path)
    except OSError as exc:
        raise ClusterFeatureContractError(
            f"pinned contract input disappeared: {digest.path}"
        ) from exc
    if current != digest.stat_signature:
        raise ClusterFeatureContractError(
            f"pinned contract input changed after hashing: {digest.path}"
        )


@beartype
@dataclass(frozen=True)
class FeatureDirectoryIdentity:
    segment_size: PositiveIntLike
    layer: NonNegativeIntLike
    activation_type: NonEmptyStrLike
    pooling: NonEmptyStrLike
    width: NonEmptyStrLike
    model: NonEmptyStrLike
    dataset_phase: NonEmptyStrLike
    context_independent_features: bool
    ignore_bos_features: bool
    segment_mode: NonEmptyStrLike = SegmentMode.fixed_tokens.value

    @classmethod
    def from_components(
        cls,
        components: FeatureDirectoryComponents,
    ) -> FeatureDirectoryIdentity:
        return cls(
            segment_size=components.segment_size,
            layer=components.layer,
            activation_type=components.activation_type.value,
            pooling=components.pooling.value,
            width=components.width.value,
            model=components.model,
            dataset_phase=components.dataset_phase.value,
            context_independent_features=components.context_independent_features,
            ignore_bos_features=components.ignore_bos_features,
            segment_mode=components.segment_mode.value,
        )

    def canonical_dict(self) -> dict[str, Any]:
        result = {
            "segment_size": self.segment_size,
            "layer": self.layer,
            "activation_type": self.activation_type,
            "pooling": self.pooling,
            "width": self.width,
            "model": self.model,
            "dataset_phase": self.dataset_phase,
            "context_independent_features": self.context_independent_features,
            "ignore_bos_features": self.ignore_bos_features,
        }
        if self.segment_mode != SegmentMode.fixed_tokens.value:
            result["segment_mode"] = self.segment_mode
        return result


@beartype
def _validate_feature_id_tuple(
    values: NonNegativeIntTupleLike,
    *,
    n_sae_features: PositiveIntLike,
    field_name: NonBlankStrLike,
) -> None:
    for feature_id in values:
        if feature_id >= n_sae_features:
            raise ClusterFeatureContractError(
                f"{field_name} contains invalid ID {feature_id!r} for width {n_sae_features}"
            )
    for previous, current in itertools.pairwise(values):
        if previous >= current:
            raise ClusterFeatureContractError(f"{field_name} must be a sorted unique tuple")


@beartype
@dataclass(frozen=True)
class FeatureStatisticsIdentity:
    file_sha256: LowercaseSha256Like
    percentile_table_checksum: LowercaseSha256Like
    n_sae_features: PositiveIntLike
    feature_id_domain: NonNegativeIntTupleLike

    def __post_init__(self) -> None:
        _validate_feature_id_tuple(
            self.feature_id_domain,
            n_sae_features=self.n_sae_features,
            field_name="statistics feature-ID domain",
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "percentile_table_checksum": self.percentile_table_checksum,
            "n_sae_features": self.n_sae_features,
            "feature_id_domain": list(self.feature_id_domain),
        }


@beartype
@dataclass(frozen=True)
class SaeIdentity:
    release: NonEmptyStrLike
    sae_id: NonEmptyStrLike
    hook_name: NonEmptyStrLike
    model: NonEmptyStrLike
    layer: NonNegativeIntLike
    activation_type: NonEmptyStrLike
    width: NonEmptyStrLike
    actual_loaded_width: PositiveIntLike

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "release": self.release,
            "sae_id": self.sae_id,
            "hook_name": self.hook_name,
            "model": self.model,
            "layer": self.layer,
            "activation_type": self.activation_type,
            "width": self.width,
            "actual_loaded_width": self.actual_loaded_width,
        }


__all__ = [
    "CONCEPT_PROJECTION_VERSION",
    "FEATURE_CONTRACT_SCHEMA_VERSION",
    "FEATURE_DIFFMEAN_PROJECTION_VERSION",
    "GENERIC_ROOFLINE_SIGNAL_VERSION",
    "MAX_SAFE_SELECTED_FEATURE_WIDTH",
    "PACKED_PERCENTILE_SELECTION_VERSION",
    "QC_TARGET_REDUCTION_VERSION",
    "ClusterFeatureContractError",
    "FeatureDirectoryIdentity",
    "FeatureStatisticsIdentity",
    "SaeIdentity",
    "StableFileDigest",
    "stat_signature_now",
    "stream_hash_file_stable",
]
