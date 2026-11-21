"""Static YAML/JSON feature clusters for selected-feature precomputation.

This module owns only content parsing, canonical identity, mask-derived
representatives, generic aligned-channel max reduction, and artifact guards.
It intentionally has no dependency on delta conditioning, feature selection,
packed rows, model code, or clustering discovery machinery.

All IDs are original SAE feature IDs. Compact encoder-row IDs never enter this
module. Callers must apply the fully resolved :class:`FeatureMask` before
calling :meth:`FeatureClusterConsolidation.consolidate_segment`; the derived
state asserts that contract rather than silently applying a second filter.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch as t
import yaml
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from errors import FeatureClustersError
from steering._common.strict_json import canonical_json
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.refined_types import (
    LowercaseSha256,
    LowercaseSha256Like,
    PositiveInt,
    PositiveIntLike,
)
from steering.removed_features import (
    reject_feature_clustering_config as _reject_legacy_feature_clustering_config,
)
from steering.removed_features import (
    reject_feature_clustering_metadata as _reject_legacy_feature_clustering_metadata,
)

FEATURE_CLUSTERS_SCHEMA_VERSION = 1
# Frozen issue-#714 descriptor identity. Its historical name covers the generic
# aligned-channel reducer and must not be changed for issue #790: clustered
# splice-CF uses the separate conditional identity owned by delta.py, while
# delta-free clustered artifacts retain this exact descriptor/hash.
FEATURE_CLUSTERS_SEMANTICS = "post_mask_post_delta_max_percentile_v1"
FEATURE_CLUSTERS_DESCRIPTOR_KEY = "feature_clusters_descriptor"
FEATURE_CLUSTERS_CANONICAL_KEY = "feature_clusters_canonical"


class ClusteredThresholdSelectionSemantics(StrEnum):
    """Versions the numeric currency of clustered-threshold SFT selection percentiles.

    ``FULL_DISTRIBUTION_RANK`` selects clustered-threshold SFT positives from
    the same full-distribution percentile rank used by clustered sampling and
    the feature-prediction head (``cluster_threshold_percentile_table`` in
    ``scripts/train/conditional_encdec_sft.py``, PR #1321).

    Pre-#1321 checkpoints computed ``selection_percentiles`` by re-ranking
    within the POST-THRESHOLD bucket ``boundaries`` — a boundary-relative
    (tail-relative) rank that double-applies the threshold — and stamped no
    version key at all. That unversioned semantics is a strictly different,
    incompatible numeric currency and is deliberately not representable as a
    member of this enum: there is nothing to translate it to, so a checkpoint
    written under it must never be silently resumed as if it were
    ``FULL_DISTRIBUTION_RANK``.
    """

    FULL_DISTRIBUTION_RANK = "full_distribution_rank_v1"


CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY = "clustered_threshold_selection_semantics"
_SOURCE_KEYS = frozenset({"schema_version", "clusters"})
_CONFIG_KEYS = frozenset({"path"})
_DESCRIPTOR_KEYS = frozenset({"schema_version", "n_sae_features", "content_sha256", "semantics"})
_SELECTION_METADATA_KEYS = frozenset({"descriptor", "canonical", "source_path", "counts", "status"})
_COUNT_KEYS = frozenset(
    {
        "source_clusters",
        "source_members",
        "surviving_cluster_representatives",
        "consolidated_away",
    }
)
SELECTION_SHARD_INTEGRITY_KEY = "selection_shards"
_SELECTION_SHARD_INTEGRITY_SCHEMA_VERSION = 1
_SELECTION_SHARD_INTEGRITY_KEYS = frozenset({"schema_version", "shards", "aggregate_sha256"})
_SELECTION_SHARD_ENTRY_KEYS = frozenset({"name", "byte_length", "content_sha256"})
_SELECTION_SHARD_SUFFIX = ".safetensors.zst"
_SELECTION_SET_VALIDATION_BLOCK_SIZE = 4_096


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise FeatureClustersError(
            f"feature cluster YAML expected a mapping node, got {type(node).__name__}"
        )
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise FeatureClustersError(
                f"feature cluster YAML mapping key {key!r} is not hashable"
            ) from exc
        if duplicate:
            raise FeatureClustersError(
                f"feature cluster YAML contains duplicate mapping key {key!r}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _json_unique_object(pairs: list[tuple[Any, Any]]) -> dict[Any, Any]:
    out: dict[Any, Any] = {}
    for key, value in pairs:
        if key in out:
            raise FeatureClustersError(
                f"feature cluster JSON contains duplicate mapping key {key!r}"
            )
        out[key] = value
    return out


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stat_identity(result: os.stat_result) -> tuple[int, ...]:
    return (
        int(result.st_mode),
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _file_content_identity(path: Path) -> tuple[int, str]:
    """Hash one stable direct regular file without accepting symlink targets."""
    before_result = path.lstat()
    if not stat.S_ISREG(before_result.st_mode):
        raise FeatureClustersError(f"clustered selection shard must be a real regular file: {path}")
    before = _stat_identity(before_result)
    digest = hashlib.sha256()
    byte_length = 0
    with path.open("rb") as handle:
        opened = _stat_identity(os.fstat(handle.fileno()))
        if opened != before:
            raise FeatureClustersError(
                f"clustered selection shard changed before it could be hashed: {path}"
            )
        while chunk := handle.read(1024 * 1024):
            byte_length += len(chunk)
            digest.update(chunk)
        if _stat_identity(os.fstat(handle.fileno())) != opened:
            raise FeatureClustersError(
                f"clustered selection shard changed while it was hashed: {path}"
            )
    if _stat_identity(path.lstat()) != before:
        raise FeatureClustersError(
            f"clustered selection shard path changed while it was hashed: {path}"
        )
    return byte_length, digest.hexdigest()


def build_clustered_selection_shard_integrity(
    shard_paths: list[Path],
) -> dict[str, Any]:
    """Fingerprint the exact ordered compressed clustered-selection artifacts."""
    normalized = sorted((Path(path) for path in shard_paths), key=lambda path: path.name)
    names = [path.name for path in normalized]
    if not normalized:
        raise FeatureClustersError(
            "clustered selection shard content fingerprint requires at least one shard"
        )
    if len(names) != len(set(names)):
        raise FeatureClustersError(
            "clustered selection shard content fingerprint received duplicate filenames"
        )
    entries: list[dict[str, Any]] = []
    for path in normalized:
        if not path.name.endswith(_SELECTION_SHARD_SUFFIX):
            raise FeatureClustersError(
                "clustered selection shard content fingerprint requires "
                f"{_SELECTION_SHARD_SUFFIX} files, got {path.name!r}"
            )
        try:
            byte_length, content_sha256 = _file_content_identity(path)
        except OSError as exc:
            raise FeatureClustersError(
                f"clustered selection shard content fingerprint cannot read {path}"
            ) from exc
        entries.append(
            {
                "name": path.name,
                "byte_length": byte_length,
                "content_sha256": content_sha256,
            }
        )
    signed_payload: dict[str, Any] = {
        "schema_version": _SELECTION_SHARD_INTEGRITY_SCHEMA_VERSION,
        "shards": entries,
    }
    return {
        **signed_payload,
        "aggregate_sha256": _sha256_json(signed_payload),
    }


def extend_clustered_selection_shard_integrity(
    prior: Any,
    shard_path: Path,
    *,
    surface: str,
) -> dict[str, Any]:
    """Add one atomically published shard to a progress fingerprint.

    Only the new shard bytes are hashed. Prior inventories must come from
    this builder or the parser at the resume boundary. Re-adding an identical
    shard is idempotent; changing an already certified shard fails rather than
    silently moving the resume boundary.
    """
    new_integrity = build_clustered_selection_shard_integrity([Path(shard_path)])
    new_entry = new_integrity["shards"][0]
    if prior is None:
        return new_integrity
    by_name = {entry["name"]: dict(entry) for entry in prior["shards"]}
    previous_entry = by_name.get(new_entry["name"])
    if previous_entry is not None and previous_entry != new_entry:
        raise FeatureClustersError(
            f"{surface} refuses to replace the certified identity for "
            f"{new_entry['name']!r}: prior={previous_entry!r}, new={new_entry!r}"
        )
    by_name[new_entry["name"]] = new_entry
    entries = [by_name[name] for name in sorted(by_name)]
    signed_payload: dict[str, Any] = {
        "schema_version": _SELECTION_SHARD_INTEGRITY_SCHEMA_VERSION,
        "shards": entries,
    }
    return {
        **signed_payload,
        "aggregate_sha256": _sha256_json(signed_payload),
    }


def parse_clustered_selection_shard_inventory(
    raw: Any,
    *,
    surface: str,
) -> dict[str, Any]:
    """Parse the strict ordered clustered-selection byte inventory.

    Unlike :func:`validated_clustered_selection_shard_names`, this public
    parser authenticates the manifest structure and aggregate without opening
    the shard bodies.  Remote startup uses it to establish the exact expected
    namespace before any body is materialized; ordinary local consumers keep
    using the body-verifying helper below.
    """
    if not isinstance(raw, Mapping) or set(raw) != _SELECTION_SHARD_INTEGRITY_KEYS:
        keys = set(raw) if isinstance(raw, Mapping) else set()
        raise FeatureClustersError(
            f"{surface} clustered selection shard content fingerprint must contain "
            f"exactly {sorted(_SELECTION_SHARD_INTEGRITY_KEYS)}; got "
            f"keys={sorted(keys, key=str)}"
        )
    schema_version = raw["schema_version"]
    if type(schema_version) is not int or (
        schema_version != _SELECTION_SHARD_INTEGRITY_SCHEMA_VERSION
    ):
        raise FeatureClustersError(
            f"{surface} clustered selection shard content fingerprint "
            f"schema_version must be the strict integer "
            f"{_SELECTION_SHARD_INTEGRITY_SCHEMA_VERSION}, got {schema_version!r}"
        )
    shards = raw["shards"]
    if not isinstance(shards, list) or not shards:
        raise FeatureClustersError(
            f"{surface} clustered selection shard content fingerprint shards must "
            "be a non-empty list"
        )
    parsed_entries: list[dict[str, Any]] = []
    previous_name: str | None = None
    for index, raw_entry in enumerate(shards):
        if not isinstance(raw_entry, Mapping) or (set(raw_entry) != _SELECTION_SHARD_ENTRY_KEYS):
            keys = set(raw_entry) if isinstance(raw_entry, Mapping) else set()
            raise FeatureClustersError(
                f"{surface} clustered selection shard content fingerprint entry "
                f"{index} must contain exactly {sorted(_SELECTION_SHARD_ENTRY_KEYS)}; "
                f"got keys={sorted(keys, key=str)}"
            )
        entry = cast(Mapping[str, Any], raw_entry)
        name = entry["name"]
        if (
            not isinstance(name, str)
            or not name
            or name != Path(name).name
            or not name.endswith(_SELECTION_SHARD_SUFFIX)
        ):
            raise FeatureClustersError(
                f"{surface} clustered selection shard content fingerprint entry "
                f"{index} has invalid direct shard filename {name!r}"
            )
        if previous_name is not None and name <= previous_name:
            raise FeatureClustersError(
                f"{surface} clustered selection shard content fingerprint filenames "
                "must be unique and strictly sorted"
            )
        previous_name = name
        byte_length = entry["byte_length"]
        if type(byte_length) is not int or byte_length < 0:
            raise FeatureClustersError(
                f"{surface} clustered selection shard content fingerprint entry "
                f"{name!r} byte_length must be a non-negative strict int, got "
                f"{byte_length!r}"
            )
        content_sha256 = entry["content_sha256"]
        if not isinstance(content_sha256, LowercaseSha256):
            raise FeatureClustersError(
                f"{surface} clustered selection shard content fingerprint entry "
                f"{name!r} content_sha256 must be 64 lowercase hexadecimal characters"
            )
        parsed_entries.append(
            {
                "name": name,
                "byte_length": byte_length,
                "content_sha256": content_sha256,
            }
        )
    aggregate_sha256 = raw["aggregate_sha256"]
    signed_payload = {
        "schema_version": schema_version,
        "shards": parsed_entries,
    }
    expected_aggregate = _sha256_json(signed_payload)
    if aggregate_sha256 != expected_aggregate:
        raise FeatureClustersError(
            f"{surface} clustered selection shard content fingerprint aggregate "
            f"mismatch: manifest has {aggregate_sha256!r}, entries hash to "
            f"{expected_aggregate!r}"
        )
    return {
        **signed_payload,
        "aggregate_sha256": expected_aggregate,
    }


def assert_clustered_selection_shard_integrity(
    selection_dir: str | Path,
    metadata: Any,
    *,
    surface: str,
) -> None:
    """Verify completed clustered selection bytes against their manifest."""
    validated_clustered_selection_shard_names(
        selection_dir,
        metadata,
        surface=surface,
        allow_untracked=False,
    )


def validated_clustered_selection_shard_names(
    selection_dir: str | Path,
    metadata: Any,
    *,
    surface: str,
    allow_untracked: bool,
) -> frozenset[str]:
    """Verify certified shard bytes and return their direct filenames.

    Completed consumers set ``allow_untracked=False`` and therefore require
    exact equality between the directory and the signed list. An interrupted
    producer sets it to true: files written after the last atomic manifest
    checkpoint are deliberately untrusted and may be deterministically
    overwritten, while every name present in the checkpoint must still match
    byte-for-byte.
    """
    if not isinstance(allow_untracked, bool):
        raise TypeError(f"allow_untracked must be a bool, got {allow_untracked!r}")
    if not isinstance(metadata, Mapping):
        raise FeatureClustersError(
            f"{surface} is missing clustered selection shard content fingerprint metadata"
        )
    expected = parse_clustered_selection_shard_inventory(
        metadata.get(SELECTION_SHARD_INTEGRITY_KEY),
        surface=surface,
    )
    directory = Path(selection_dir)
    expected_names = frozenset(entry["name"] for entry in expected["shards"])
    certified_paths = [directory / entry["name"] for entry in expected["shards"]]
    try:
        actual = build_clustered_selection_shard_integrity(certified_paths)
    except FeatureClustersError as exc:
        raise FeatureClustersError(
            f"{surface} clustered selection shard content fingerprint mismatch: {exc}; "
            "re-run precompute_selected_features"
        ) from exc
    if actual != expected:
        raise FeatureClustersError(
            f"{surface} clustered selection shard content fingerprint mismatch: "
            f"manifest aggregate={expected['aggregate_sha256']!r}, current "
            f"aggregate={actual['aggregate_sha256']!r}; re-run "
            "precompute_selected_features"
        )
    if not allow_untracked:
        actual_names = frozenset(
            path.name for path in directory.glob(f"*{_SELECTION_SHARD_SUFFIX}")
        )
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise FeatureClustersError(
                f"{surface} clustered selection shard content fingerprint mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}; re-run "
                "precompute_selected_features"
            )
    return expected_names


@dataclass(frozen=True)
class FeatureClusterDefinition:
    """Strict canonical schema-v1 cluster content, independent of source path."""

    clusters: tuple[tuple[int, ...], ...]
    schema_version: int = FEATURE_CLUSTERS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != FEATURE_CLUSTERS_SCHEMA_VERSION
        ):
            raise FeatureClustersError(
                "feature cluster schema_version must be the strict integer "
                f"{FEATURE_CLUSTERS_SCHEMA_VERSION}, got {self.schema_version!r}"
            )
        if not isinstance(self.clusters, tuple) or not self.clusters:
            raise FeatureClustersError("feature cluster list must be non-empty")

        canonical_clusters: list[tuple[int, ...]] = []
        union: set[int] = set()
        for cluster_idx, members in enumerate(self.clusters):
            if not isinstance(members, tuple):
                raise FeatureClustersError(
                    f"feature cluster {cluster_idx} must be a list of feature IDs"
                )
            if len(members) < 2:
                raise FeatureClustersError(
                    f"feature cluster {cluster_idx} must contain at least two source IDs"
                )
            for member_idx, feature_id in enumerate(members):
                if type(feature_id) is not int:
                    raise FeatureClustersError(
                        f"feature cluster {cluster_idx} member {member_idx} must be a "
                        f"strict int, got {type(feature_id).__name__} {feature_id!r}"
                    )
                if feature_id < 0:
                    raise FeatureClustersError(
                        f"feature cluster {cluster_idx} ID {feature_id} is negative"
                    )
            if len(set(members)) != len(members):
                raise FeatureClustersError(
                    f"feature cluster {cluster_idx} contains a duplicate member"
                )
            canonical = tuple(sorted(members))
            overlap = union.intersection(canonical)
            if overlap:
                feature_id = min(overlap)
                raise FeatureClustersError(
                    f"feature ID {feature_id} appears in more than one cluster"
                )
            union.update(canonical)
            canonical_clusters.append(canonical)

        object.__setattr__(self, "clusters", tuple(sorted(canonical_clusters)))

    @classmethod
    @beartype
    def from_mapping(
        cls,
        raw: Any,
        *,
        n_sae_features: PositiveIntLike,
    ) -> FeatureClusterDefinition:
        """Validate a decoded source mapping and bind its IDs to an SAE width."""
        if not isinstance(raw, Mapping):
            raise FeatureClustersError(
                f"feature cluster file must contain a top-level mapping, got {type(raw).__name__}"
            )
        keys = set(raw)
        if keys != _SOURCE_KEYS:
            missing = sorted(_SOURCE_KEYS - keys)
            extra = sorted(keys - _SOURCE_KEYS, key=str)
            raise FeatureClustersError(
                "feature cluster file must contain exactly schema_version and "
                f"clusters; missing={missing}, extra={extra}"
            )
        schema_version = raw["schema_version"]
        clusters_raw = raw["clusters"]
        if type(schema_version) is not int or (schema_version != FEATURE_CLUSTERS_SCHEMA_VERSION):
            raise FeatureClustersError(
                "feature cluster schema_version must be the strict integer "
                f"{FEATURE_CLUSTERS_SCHEMA_VERSION}, got {schema_version!r}"
            )
        if not isinstance(clusters_raw, list) or not clusters_raw:
            raise FeatureClustersError("feature cluster 'clusters' must be a non-empty list")
        clusters: list[tuple[int, ...]] = []
        for cluster_idx, members in enumerate(clusters_raw):
            if not isinstance(members, list):
                raise FeatureClustersError(
                    f"feature cluster {cluster_idx} must be a list, got {type(members).__name__}"
                )
            cluster = cast(tuple[int, ...], tuple(members))
            clusters.append(cluster)

        definition = cls(
            clusters=tuple(clusters),
            schema_version=schema_version,
        )
        for cluster in definition.clusters:
            for feature_id in cluster:
                if feature_id >= n_sae_features:
                    raise FeatureClustersError(
                        f"feature cluster ID {feature_id} is out of range [0, {n_sae_features})"
                    )
        return definition

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "clusters": [list(cluster) for cluster in self.clusters],
        }

    def canonical_json(self) -> str:
        # Bare `canonical_json(...)` here is the module-level import from
        # steering._common.strict_json — NOT this method. Do not "fix" it to
        # self.canonical_json(...): that is infinite recursion.
        return canonical_json(self.canonical_dict())

    def content_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def n_members(self) -> int:
        return sum(len(cluster) for cluster in self.clusters)


@dataclass(frozen=True, init=False)
class FeatureClusterDescriptor:
    """Selection/checkpoint identity for canonical cluster semantics."""

    n_sae_features: PositiveInt
    content_sha256: LowercaseSha256
    schema_version: int = FEATURE_CLUSTERS_SCHEMA_VERSION
    semantics: str = FEATURE_CLUSTERS_SEMANTICS

    @beartype
    def __init__(
        self,
        n_sae_features: PositiveIntLike,
        content_sha256: LowercaseSha256Like,
        schema_version: int = FEATURE_CLUSTERS_SCHEMA_VERSION,
        semantics: str = FEATURE_CLUSTERS_SEMANTICS,
    ) -> None:
        if type(schema_version) is not int or (schema_version != FEATURE_CLUSTERS_SCHEMA_VERSION):
            raise FeatureClustersError(
                f"unsupported feature cluster descriptor schema {schema_version!r}"
            )
        if semantics != FEATURE_CLUSTERS_SEMANTICS:
            raise FeatureClustersError(
                f"unsupported feature cluster semantics {semantics!r}; "
                f"expected {FEATURE_CLUSTERS_SEMANTICS!r}"
            )
        object.__setattr__(
            self,
            "n_sae_features",
            PositiveInt.parse(n_sae_features),
        )
        object.__setattr__(
            self,
            "content_sha256",
            LowercaseSha256.parse(content_sha256),
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "semantics", semantics)

    @classmethod
    @beartype
    def from_definition(
        cls,
        definition: FeatureClusterDefinition,
        *,
        n_sae_features: PositiveIntLike,
    ) -> FeatureClusterDescriptor:
        return cls(
            n_sae_features=n_sae_features,
            content_sha256=definition.content_sha256(),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> FeatureClusterDescriptor:
        if not isinstance(raw, Mapping) or set(raw) != _DESCRIPTOR_KEYS:
            keys = set(raw) if isinstance(raw, Mapping) else set()
            raise FeatureClustersError(
                "feature cluster descriptor must contain exactly "
                f"{sorted(_DESCRIPTOR_KEYS)}; got keys={sorted(keys, key=str)}"
            )
        try:
            return cls(
                schema_version=raw["schema_version"],
                n_sae_features=raw["n_sae_features"],
                content_sha256=raw["content_sha256"],
                semantics=raw["semantics"],
            )
        except BeartypeCallHintParamViolation as exc:
            if not isinstance(raw["n_sae_features"], PositiveInt):
                raise FeatureClustersError(
                    "feature cluster descriptor n_sae_features must be a positive "
                    f"strict int, got {raw['n_sae_features']!r}"
                ) from exc
            if not isinstance(raw["content_sha256"], LowercaseSha256):
                raise FeatureClustersError(
                    "feature cluster content_sha256 must be a 64-character lowercase hex string"
                ) from exc
            raise FeatureClustersError(
                "feature cluster descriptor contains an invalid typed field"
            ) from exc

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_sae_features": self.n_sae_features,
            "content_sha256": self.content_sha256,
            "semantics": self.semantics,
        }

    def identity_sha256(self) -> str:
        return _sha256_json(self.canonical_dict())

    def dirname(self) -> str:
        return f"fc_{self.identity_sha256()}"


@dataclass(frozen=True)
class ResolvedFeatureClusters:
    """A source-path provenance record plus path-independent canonical identity."""

    source_path: Path
    definition: FeatureClusterDefinition
    descriptor: FeatureClusterDescriptor

    def __post_init__(self) -> None:
        expected = FeatureClusterDescriptor.from_definition(
            self.definition,
            n_sae_features=self.descriptor.n_sae_features,
        )
        if self.descriptor != expected:
            raise FeatureClustersError(
                "feature cluster descriptor does not match the canonical definition"
            )

    @property
    def n_sae_features(self) -> int:
        return self.descriptor.n_sae_features

    def canonical_dict(self) -> dict[str, Any]:
        return self.definition.canonical_dict()

    def descriptor_dict(self) -> dict[str, Any]:
        return self.descriptor.canonical_dict()


def _decode_feature_cluster_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise FeatureClustersError(f"could not read feature cluster file {path}: {exc}") from exc

    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(text, object_pairs_hook=_json_unique_object)
        if suffix in {".yaml", ".yml"}:
            return yaml.load(text, Loader=_UniqueKeySafeLoader)
    except FeatureClustersError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise FeatureClustersError(f"could not parse feature cluster file {path}: {exc}") from exc
    raise FeatureClustersError(f"feature cluster file {path} must use .yaml, .yml, or .json")


def _feature_clusters_block(
    cfg_block: Mapping[str, Any] | None,
) -> Any | None:
    if cfg_block is None:
        return None
    if not isinstance(cfg_block, Mapping):
        raise TypeError(
            f"feature clusters config must be a mapping, got {type(cfg_block).__name__}"
        )
    block = cfg_block.get("feature_clusters")
    if block is None:
        return None
    if OmegaConf.is_config(block):
        block = OmegaConf.to_container(block, resolve=True)
    return block


@beartype
def resolve_feature_clusters(
    cfg_block: Mapping[str, Any] | None,
    n_sae_features: PositiveIntLike,
) -> ResolvedFeatureClusters | None:
    """Resolve the path-only config and strictly parse a static cluster file."""
    block = _feature_clusters_block(cfg_block)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise FeatureClustersError(
            f"feature_clusters must be a path-only mapping, got {type(block).__name__}"
        )
    keys = set(block)
    if keys != _CONFIG_KEYS:
        raise FeatureClustersError(
            "feature_clusters must contain exactly one key, 'path'; "
            f"got keys={sorted(keys, key=str)}"
        )
    raw_path = block["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise FeatureClustersError("feature_clusters.path must be a non-empty string")
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeatureClustersError(
            f"feature_clusters.path not found or inaccessible: {raw_path}"
        ) from exc
    if not path.is_file():
        raise FeatureClustersError(f"feature_clusters.path is not a regular file: {path}")
    definition = FeatureClusterDefinition.from_mapping(
        _decode_feature_cluster_file(path),
        n_sae_features=n_sae_features,
    )
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=n_sae_features,
    )
    return ResolvedFeatureClusters(
        source_path=path,
        definition=definition,
        descriptor=descriptor,
    )


def canonical_feature_clusters_descriptor(
    resolved: ResolvedFeatureClusters | None,
) -> dict[str, Any] | None:
    """Conditional selection identity. ``None`` remains absent at call sites."""
    return None if resolved is None else resolved.descriptor_dict()


@dataclass(frozen=True)
class FeatureClusterConsolidation:
    """Immutable state derived from static clusters and the fully resolved mask."""

    resolved: ResolvedFeatureClusters
    feature_mask: FeatureMask | None
    rep_of: t.Tensor = field(repr=False, compare=False)
    consolidated_away: t.Tensor = field(repr=False, compare=False)
    allowed_universe: t.Tensor = field(repr=False, compare=False)
    cluster_allowed_universe: t.Tensor = field(repr=False, compare=False)
    is_identity_after_mask: bool
    _effective_negative_feature_mask: FeatureMask | None = field(
        repr=False,
        compare=False,
    )

    @classmethod
    def build(
        cls,
        resolved: ResolvedFeatureClusters,
        feature_mask: FeatureMask | None,
    ) -> FeatureClusterConsolidation:
        n = resolved.n_sae_features
        allowed = (
            feature_mask.allowed_universe_mask(n)
            if feature_mask is not None
            else t.ones(n, dtype=t.bool)
        )
        rep_of = t.arange(n, dtype=t.int64)
        away = t.zeros(n, dtype=t.bool)
        for cluster in resolved.definition.clusters:
            survivors = [feature_id for feature_id in cluster if bool(allowed[feature_id])]
            if not survivors:
                continue
            representative = survivors[0]
            survivor_tensor = t.tensor(survivors, dtype=t.int64)
            rep_of[survivor_tensor] = representative
            if len(survivors) > 1:
                away[survivor_tensor[1:]] = True
        away_ids = t.nonzero(away, as_tuple=False).squeeze(1).tolist()
        away_set: set[int] = set(away_ids)
        if not away_set:
            effective_negative_feature_mask = feature_mask
        elif feature_mask is None:
            effective_negative_feature_mask = FeatureMask(
                FeatureMaskMode.blacklist,
                tuple(away_set),
            )
        elif feature_mask.mode is FeatureMaskMode.blacklist:
            effective_negative_feature_mask = FeatureMask(
                FeatureMaskMode.blacklist,
                tuple(set(feature_mask.ids) | away_set),
            )
        else:
            effective_negative_feature_mask = FeatureMask(
                FeatureMaskMode.whitelist,
                tuple(set(feature_mask.ids) - away_set),
            )
        return cls(
            resolved=resolved,
            feature_mask=feature_mask,
            rep_of=rep_of,
            consolidated_away=away,
            allowed_universe=allowed,
            cluster_allowed_universe=allowed & ~away,
            is_identity_after_mask=not away_set,
            _effective_negative_feature_mask=effective_negative_feature_mask,
        )

    @property
    def n_sae_features(self) -> int:
        return self.resolved.n_sae_features

    def _validate_feature_ids(self, feature_ids: t.Tensor, *, surface: str) -> t.Tensor:
        if feature_ids.ndim != 1:
            raise FeatureClustersError(
                f"{surface} feature IDs must be one-dimensional, got "
                f"shape={tuple(feature_ids.shape)}"
            )
        if (
            feature_ids.dtype is t.bool
            or feature_ids.is_floating_point()
            or feature_ids.is_complex()
        ):
            raise FeatureClustersError(
                f"{surface} feature IDs must use an integer dtype, got {feature_ids.dtype}"
            )
        ids = feature_ids.to(dtype=t.int64)
        if ids.numel() == 0:
            return ids
        invalid = (ids < 0) | (ids >= self.n_sae_features)
        if bool(invalid.any()):
            bad = int(ids[invalid][0].item())
            raise FeatureClustersError(
                f"{surface} feature ID {bad} is out of range [0, {self.n_sae_features})"
            )
        return ids

    def _assert_mask_applied_to_validated_ids(self, ids: t.Tensor) -> None:
        if ids.numel() == 0:
            return
        allowed = self.allowed_universe.to(ids.device)
        if not bool(allowed[ids].all()):
            bad = int(ids[~allowed[ids]][0].item())
            raise FeatureClustersError(
                f"cluster input contains mask-disallowed feature ID {bad}; apply the "
                "fully resolved FeatureMask before cluster consolidation"
            )

    def consolidate_segment(
        self,
        feature_ids: t.Tensor,
        percentiles: t.Tensor,
        *aligned_scores: t.Tensor,
    ) -> tuple[t.Tensor, ...]:
        """Map one masked segment to ascending representatives and max channels."""
        ids = self._validate_feature_ids(feature_ids, surface="cluster input")
        self._assert_mask_applied_to_validated_ids(ids)
        channels: tuple[t.Tensor, ...] = (percentiles, *aligned_scores)
        for channel_idx, channel in enumerate(channels):
            if channel.ndim != 1 or channel.shape[0] != ids.shape[0]:
                raise FeatureClustersError(
                    f"cluster aligned channel {channel_idx} must have shape "
                    f"({ids.shape[0]},), got {tuple(channel.shape)}"
                )
            if channel.device != ids.device:
                raise FeatureClustersError(
                    f"cluster aligned channel {channel_idx} is on {channel.device}, "
                    f"but feature IDs are on {ids.device}"
                )
            if not channel.is_floating_point():
                raise FeatureClustersError(
                    f"cluster aligned channel {channel_idx} must be floating point, "
                    f"got {channel.dtype}"
                )
            if not bool(t.isfinite(channel).all()):
                raise FeatureClustersError(
                    f"cluster aligned channel {channel_idx} contains a non-finite value"
                )
        if percentiles.numel() > 0 and bool(((percentiles < 0.0) | (percentiles > 1.0)).any()):
            bad = float(percentiles[((percentiles < 0.0) | (percentiles > 1.0))][0].item())
            raise FeatureClustersError(
                f"cluster percentiles must be normalized to [0, 1], got {bad}"
            )
        if ids.numel() == 0:
            return (ids, *(channel[:0] for channel in channels))

        reps = self.rep_of.to(ids.device)[ids]
        unique_reps, inverse = t.unique(reps, sorted=True, return_inverse=True)
        reduced: list[t.Tensor] = []
        for channel in channels:
            reduced.append(
                channel.new_zeros(unique_reps.shape[0]).scatter_reduce(
                    0,
                    inverse,
                    channel,
                    reduce="amax",
                    include_self=False,
                )
            )
        return (unique_reps, *reduced)

    def project_allowed_presence(self, presence: t.Tensor) -> t.Tensor:
        """Project mask-allowed original current/reference presence onto reps."""
        if (
            presence.dtype is not t.bool
            or presence.ndim != 1
            or (presence.shape[0] != self.n_sae_features)
        ):
            raise FeatureClustersError(
                "cluster presence must be bool[n_sae_features], got "
                f"dtype={presence.dtype}, shape={tuple(presence.shape)}"
            )
        allowed = self.allowed_universe.to(presence.device)
        active = t.nonzero(presence & allowed, as_tuple=False).squeeze(1)
        projected = t.zeros_like(presence)
        if active.numel() > 0:
            reps = self.rep_of.to(presence.device)[active]
            projected[reps] = True
        return projected

    def assert_representative_presence(
        self,
        presence: t.Tensor,
        *,
        surface: str,
    ) -> None:
        """Require a presence bitmap to contain only mask-allowed representatives.

        Unlike :meth:`project_allowed_presence`, this is a validation-only
        operation: it never repairs raw-ID presence by moving bits to
        representatives. Clustered counterfactual deltas already emit presence
        in representative space, so consumers use this guard before trusting
        that producer contract without allocating a second full-width bitmap.
        """
        if (
            presence.dtype is not t.bool
            or presence.ndim != 1
            or (presence.shape[0] != self.n_sae_features)
        ):
            raise FeatureClustersError(
                f"{surface} must be bool[n_sae_features], got "
                f"dtype={presence.dtype}, shape={tuple(presence.shape)}"
            )
        representative_universe = self.cluster_allowed_universe.to(presence.device)
        invalid = presence & ~representative_universe
        if bool(invalid.any()):
            bad = int(t.nonzero(invalid, as_tuple=False)[0].item())
            raise FeatureClustersError(
                f"{surface} contains feature ID {bad}, which is not a mask-allowed representative"
            )

    def effective_negative_mask(self) -> FeatureMask | None:
        """Return the build-time-cached cluster-aware negative universe mask."""
        return self._effective_negative_feature_mask

    def assert_representatives_only(
        self,
        feature_ids: t.Tensor,
        *,
        surface: str = "clustered selection",
    ) -> None:
        """Bounds-first final artifact invariant for real positive/negative IDs."""
        ids = self._validate_feature_ids(feature_ids, surface=surface)
        if ids.numel() == 0:
            return
        allowed = self.allowed_universe.to(ids.device)
        if not bool(allowed[ids].all()):
            bad = int(ids[~allowed[ids]][0].item())
            raise FeatureClustersError(f"{surface} contains mask-disallowed feature ID {bad}")
        away = self.consolidated_away.to(ids.device)
        if bool(away[ids].any()):
            bad = int(ids[away[ids]][0].item())
            raise FeatureClustersError(f"{surface} contains consolidated-away feature ID {bad}")
        reps = self.rep_of.to(ids.device)
        non_rep = reps[ids] != ids
        if bool(non_rep.any()):
            bad = int(ids[non_rep][0].item())
            raise FeatureClustersError(
                f"{surface} contains non-representative feature ID {bad}; "
                f"it maps to representative {int(reps[bad].item())}"
            )

    def assert_selection_shard(
        self,
        *,
        sel_offsets: t.Tensor,
        sel_fids: t.Tensor,
        sel_buckets: t.Tensor,
        sel_pcts: t.Tensor,
        neg_offsets: t.Tensor,
        neg_fids: t.Tensor,
        expected_segments: int,
        effective_buckets: int,
        surface: str,
    ) -> None:
        """Validate the complete clustered selection-shard storage contract.

        This is intentionally active-cluster-only. Legacy selection artifacts
        keep their existing acceptance behavior, while a clustered manifest's
        ``complete`` marker certifies exact dtypes, CSR shape/monotonicity,
        aligned positive channels, ID/bucket bounds, representative-only IDs,
        and per-segment positive/negative set semantics.
        """
        if type(expected_segments) is not int or expected_segments < 0:
            raise FeatureClustersError(
                f"{surface} expected_segments must be a non-negative strict int, "
                f"got {expected_segments!r}"
            )
        # Positive buckets are stored as uint8, so effective bucket counts up
        # to 256 are lossless (indices 0..255). If negative sampling is active,
        # its sentinel is ``effective_buckets`` but is reconstructed directly
        # into the int64 batch tensor; it is not serialized in sel_buckets.
        if type(effective_buckets) is not int or not (1 <= effective_buckets <= 256):
            raise FeatureClustersError(
                f"{surface} effective_buckets must be a strict int in [1, 256], "
                f"got {effective_buckets!r}"
            )

        tensors = {
            "sel_offsets": (sel_offsets, t.int32),
            "sel_fids": (sel_fids, t.uint16),
            "sel_buckets": (sel_buckets, t.uint8),
            "sel_pcts": (sel_pcts, t.uint8),
            "neg_offsets": (neg_offsets, t.int32),
            "neg_fids": (neg_fids, t.uint16),
        }
        for name, (tensor, expected_dtype) in tensors.items():
            if not isinstance(tensor, t.Tensor):
                raise FeatureClustersError(
                    f"{surface} {name} must be a torch.Tensor, got {type(tensor).__name__}"
                )
            if tensor.ndim != 1:
                raise FeatureClustersError(
                    f"{surface} {name} must be one-dimensional, got shape={tuple(tensor.shape)}"
                )
            if tensor.dtype != expected_dtype:
                raise FeatureClustersError(
                    f"{surface} {name} must have dtype {expected_dtype}, got {tensor.dtype}"
                )
            if tensor.device != sel_offsets.device:
                raise FeatureClustersError(
                    f"{surface} tensors must share one device; sel_offsets is on "
                    f"{sel_offsets.device}, {name} is on {tensor.device}"
                )

        expected_offset_count = expected_segments + 1
        for name, offsets, value_count in (
            ("sel_offsets", sel_offsets, int(sel_fids.numel())),
            ("neg_offsets", neg_offsets, int(neg_fids.numel())),
        ):
            if offsets.numel() != expected_offset_count:
                raise FeatureClustersError(
                    f"{surface} {name} has {offsets.numel()} entries, expected "
                    f"{expected_offset_count} for {expected_segments} segments"
                )
            offsets_i64 = offsets.to(t.int64)
            if int(offsets_i64[0].item()) != 0:
                raise FeatureClustersError(f"{surface} {name} must start at zero")
            if bool((offsets_i64[1:] < offsets_i64[:-1]).any()):
                raise FeatureClustersError(f"{surface} {name} must be monotonically nondecreasing")
            if int(offsets_i64[-1].item()) != value_count:
                raise FeatureClustersError(
                    f"{surface} {name} terminates at "
                    f"{int(offsets_i64[-1].item())}, but its value array has "
                    f"length {value_count}"
                )

        n_positive = int(sel_fids.numel())
        if sel_buckets.numel() != n_positive or sel_pcts.numel() != n_positive:
            raise FeatureClustersError(
                f"{surface} positive channels are misaligned: sel_fids={n_positive}, "
                f"sel_buckets={sel_buckets.numel()}, sel_pcts={sel_pcts.numel()}"
            )
        if sel_buckets.numel() > 0 and bool((sel_buckets.to(t.int64) >= effective_buckets).any()):
            bad = int(sel_buckets[sel_buckets.to(t.int64) >= effective_buckets][0].item())
            raise FeatureClustersError(
                f"{surface} positive bucket {bad} is out of range [0, {effective_buckets})"
            )

        positive_ids = sel_fids.to(t.int64)
        negative_ids = neg_fids.to(t.int64)
        self.assert_representatives_only(
            positive_ids,
            surface=f"{surface} positive selection",
        )
        self.assert_representatives_only(
            negative_ids,
            surface=f"{surface} negative selection",
        )
        if positive_ids.numel() == 0 and negative_ids.numel() == 0:
            return

        positive_offsets = sel_offsets.tolist()
        negative_offsets = neg_offsets.tolist()
        for block_start in range(
            0,
            expected_segments,
            _SELECTION_SET_VALIDATION_BLOCK_SIZE,
        ):
            block_end = min(
                block_start + _SELECTION_SET_VALIDATION_BLOCK_SIZE,
                expected_segments,
            )
            positive_base = positive_offsets[block_start]
            negative_base = negative_offsets[block_start]
            positive_block = positive_ids[positive_base : positive_offsets[block_end]].tolist()
            negative_block = negative_ids[negative_base : negative_offsets[block_end]].tolist()
            # U16 IDs and 4096 local rows give injective keys below 2**28.
            row_tags = np.arange(block_end - block_start, dtype=np.uint32) * np.uint32(65536)
            combined_keys = np.array(positive_block + negative_block, dtype=np.uint32)
            combined_keys[: len(positive_block)] += np.repeat(
                row_tags,
                np.diff(
                    np.asarray(positive_offsets[block_start : block_end + 1], dtype=np.int64)
                ),
            )
            combined_keys[len(positive_block) :] += np.repeat(
                row_tags,
                np.diff(
                    np.asarray(negative_offsets[block_start : block_end + 1], dtype=np.int64)
                ),
            )
            combined_keys.sort()
            if not np.any(combined_keys[1:] == combined_keys[:-1]):
                continue
            for segment_idx in range(block_start, block_end):
                pos_start, pos_end = positive_offsets[segment_idx : segment_idx + 2]
                neg_start, neg_end = negative_offsets[segment_idx : segment_idx + 2]
                positive = positive_block[pos_start - positive_base : pos_end - positive_base]
                negative = negative_block[neg_start - negative_base : neg_end - negative_base]
                if len(positive) != len(set(positive)):
                    raise FeatureClustersError(
                        f"{surface} segment {segment_idx} contains duplicate positive IDs"
                    )
                if len(negative) != len(set(negative)):
                    raise FeatureClustersError(
                        f"{surface} segment {segment_idx} contains duplicate negative IDs"
                    )
                overlap = set(positive).intersection(negative)
                if overlap:
                    raise FeatureClustersError(
                        f"{surface} segment {segment_idx} contains feature ID "
                        f"{min(overlap)} as both positive and negative"
                    )

    def metadata_dict(self) -> dict[str, Any]:
        surviving_representatives = 0
        for cluster in self.resolved.definition.clusters:
            if any(bool(self.allowed_universe[feature_id]) for feature_id in cluster):
                surviving_representatives += 1
        return {
            "descriptor": self.resolved.descriptor_dict(),
            "canonical": self.resolved.canonical_dict(),
            "source_path": str(self.resolved.source_path),
            "counts": {
                "source_clusters": self.resolved.definition.n_clusters,
                "source_members": self.resolved.definition.n_members,
                "surviving_cluster_representatives": surviving_representatives,
                "consolidated_away": int(self.consolidated_away.sum().item()),
            },
        }


def feature_clusters_dirname(resolved: ResolvedFeatureClusters | None) -> str:
    return "" if resolved is None else resolved.descriptor.dirname()


def append_feature_clusters_leaf(
    path: str,
    resolved: ResolvedFeatureClusters | None,
) -> str:
    leaf = feature_clusters_dirname(resolved)
    return os.path.join(path, leaf) if leaf else path


def feature_clusters_checkpoint_metadata(
    resolved: ResolvedFeatureClusters | None,
) -> dict[str, Any]:
    if resolved is None:
        return {}
    return {
        FEATURE_CLUSTERS_DESCRIPTOR_KEY: resolved.descriptor_dict(),
        FEATURE_CLUSTERS_CANONICAL_KEY: resolved.canonical_dict(),
    }


def assert_feature_clusters_selection_metadata(
    configured: FeatureClusterConsolidation | None,
    metadata: Any,
    *,
    surface: str,
    require_complete: bool = True,
) -> None:
    """Validate a selection manifest's cluster contract bidirectionally.

    ``source_path`` is retained and shape-checked as provenance, but is not an
    identity field: equivalent canonical files may live at different paths on
    producer and consumer hosts. Every semantic field, derived count, and the
    completion marker is strict. A disabled consumer rejects an active
    manifest instead of accidentally treating clustered rows as ordinary rows.
    """
    # The supported preselection path accepts the new ``feature_clusters``
    # schema, but it must never reinterpret artifacts from the removed
    # ``feature_clustering`` pipeline as current static clusters.
    _reject_legacy_feature_clustering_metadata(metadata, surface)
    if configured is None:
        reject_feature_clusters_metadata(metadata, surface)
        return
    if not isinstance(metadata, Mapping):
        raise FeatureClustersError(f"{surface} has invalid manifest metadata; expected a mapping")
    raw = metadata.get("feature_clusters")
    if not isinstance(raw, Mapping):
        raise FeatureClustersError(
            f"{surface} is missing feature_clusters metadata required by the "
            "configured static clusters"
        )
    keys = set(raw)
    if keys != _SELECTION_METADATA_KEYS:
        raise FeatureClustersError(
            f"{surface} feature_clusters metadata must contain exactly "
            f"{sorted(_SELECTION_METADATA_KEYS)}; missing="
            f"{sorted(_SELECTION_METADATA_KEYS - keys)}, extra="
            f"{sorted(keys - _SELECTION_METADATA_KEYS, key=str)}"
        )
    source_path = raw["source_path"]
    if not isinstance(source_path, str) or not source_path.strip():
        raise FeatureClustersError(
            f"{surface} feature_clusters.source_path must be a non-empty string"
        )
    status = raw["status"]
    allowed_statuses = (
        {"complete"}
        if require_complete
        else {
            "in_progress",
            "complete",
        }
    )
    if status not in allowed_statuses:
        raise FeatureClustersError(
            f"{surface} feature_clusters.status must be one of "
            f"{sorted(allowed_statuses)}, got {status!r}"
        )

    descriptor = FeatureClusterDescriptor.from_dict(raw["descriptor"])
    definition = FeatureClusterDefinition.from_mapping(
        raw["canonical"],
        n_sae_features=configured.n_sae_features,
    )
    expected_metadata = configured.metadata_dict()
    expected_descriptor = configured.resolved.descriptor
    expected_definition = configured.resolved.definition
    if descriptor != expected_descriptor or definition != expected_definition:
        raise FeatureClustersError(
            f"{surface} feature cluster identity mismatch: manifest "
            f"descriptor/content={descriptor.canonical_dict()!r}/"
            f"{definition.canonical_dict()!r}, configured="
            f"{expected_descriptor.canonical_dict()!r}/"
            f"{expected_definition.canonical_dict()!r}"
        )
    raw_counts = raw["counts"]
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != _COUNT_KEYS:
        keys = set(raw_counts) if isinstance(raw_counts, Mapping) else set()
        raise FeatureClustersError(
            f"{surface} feature_clusters.counts must contain exactly "
            f"{sorted(_COUNT_KEYS)}; got keys={sorted(keys, key=str)}"
        )
    for key in sorted(_COUNT_KEYS):
        value = raw_counts[key]
        if type(value) is not int or value < 0:
            raise FeatureClustersError(
                f"{surface} feature_clusters.counts.{key} must be a "
                f"non-negative strict int, got {value!r}"
            )
    if dict(raw_counts) != expected_metadata["counts"]:
        raise FeatureClustersError(
            f"{surface} feature cluster derived-count mismatch: manifest="
            f"{dict(raw_counts)!r}, configured={expected_metadata['counts']!r}"
        )


def assert_resume_feature_clusters_match(
    configured: ResolvedFeatureClusters | None,
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Require exact bidirectional cluster semantics on training resume."""
    meta = checkpoint_meta or {}
    _reject_legacy_feature_clustering_metadata(
        meta,
        f"checkpoint {checkpoint_dir}",
    )
    checkpoint_descriptor = meta.get(FEATURE_CLUSTERS_DESCRIPTOR_KEY)
    checkpoint_canonical = meta.get(FEATURE_CLUSTERS_CANONICAL_KEY)
    checkpoint_user_config = meta.get("user_config")
    user_config_active = (
        isinstance(checkpoint_user_config, Mapping)
        and checkpoint_user_config.get("feature_clusters") is not None
    )
    checkpoint_active = (
        checkpoint_descriptor is not None or checkpoint_canonical is not None or user_config_active
    )
    if configured is None:
        if checkpoint_active:
            raise FeatureClustersError(
                f"checkpoint {checkpoint_dir} was trained with static feature clusters, "
                "but this run has feature_clusters disabled; configure the exact same "
                "cluster file or start a new run"
            )
        return
    if checkpoint_descriptor is None or checkpoint_canonical is None:
        raise FeatureClustersError(
            f"checkpoint {checkpoint_dir} lacks complete static feature-cluster "
            "metadata required by the configured feature_clusters; start fresh or "
            "resume a checkpoint written by the clustered preselection path"
        )
    try:
        parsed_descriptor = FeatureClusterDescriptor.from_dict(checkpoint_descriptor)
        parsed_definition = FeatureClusterDefinition.from_mapping(
            checkpoint_canonical,
            n_sae_features=parsed_descriptor.n_sae_features,
        )
    except FeatureClustersError as exc:
        raise FeatureClustersError(
            f"checkpoint {checkpoint_dir} has invalid static feature-cluster metadata: {exc}"
        ) from exc
    if parsed_descriptor != configured.descriptor or (parsed_definition != configured.definition):
        raise FeatureClustersError(
            f"feature cluster mismatch on resume from {checkpoint_dir}: checkpoint "
            f"descriptor/content={parsed_descriptor.canonical_dict()!r}/"
            f"{parsed_definition.canonical_dict()!r}, configured="
            f"{configured.descriptor_dict()!r}/{configured.canonical_dict()!r}"
        )


def assert_resume_clustered_threshold_semantics_match(
    meta: Mapping[str, Any] | None,
    *,
    surface: str,
) -> None:
    """Require the resumed checkpoint's clustered-threshold selection semantics to match.

    Callers must gate this on the CURRENT run being clustered + threshold-
    selected (``consolidation is not None and threshold_cfg is not None``);
    sampling and unclustered runs never call it, since neither axis owns
    this guard.

    A checkpoint written before #1321 has no
    :data:`CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY` at all: its
    ``selection_percentiles`` were computed by re-ranking within the
    post-threshold bucket boundaries (a boundary-relative rank that
    double-applies the threshold), which is a different numeric currency than
    :attr:`ClusteredThresholdSelectionSemantics.FULL_DISTRIBUTION_RANK` and
    cannot be reinterpreted as it. A missing key is therefore rejected
    exactly like a mismatched one -- resuming either would silently change
    which features get selected mid-run.
    """
    stored = None if meta is None else meta.get(CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY)
    if stored is None:
        raise FeatureClustersError(
            f"{surface} predates versioned clustered-threshold selection semantics "
            "(pre-#1321 checkpoints computed selection_percentiles from "
            "post-threshold, boundary-relative bucket ranks with no version key). "
            "Resuming would silently change clustered-threshold selection mid-run. "
            "Start a fresh run, or resume this checkpoint under the pre-#1321 commit."
        )
    if stored != ClusteredThresholdSelectionSemantics.FULL_DISTRIBUTION_RANK.value:
        raise FeatureClustersError(
            f"{surface} clustered-threshold selection semantics mismatch: checkpoint "
            f"records {stored!r}, expected "
            f"{ClusteredThresholdSelectionSemantics.FULL_DISTRIBUTION_RANK.value!r}"
        )


def reject_feature_clusters_config(
    cfg: Mapping[str, Any] | None,
    surface: str,
) -> None:
    """Reject legacy or current cluster config at an unsupported surface.

    Supported producer/pretraining paths intentionally use the narrow legacy
    guard directly because they accept ``feature_clusters``. Every unsupported
    consumer should need only this one public guard.
    """
    _reject_legacy_feature_clustering_config(cfg, surface)
    if _feature_clusters_block(cfg) is not None:
        raise FeatureClustersError(
            f"{surface} does not support static feature clusters. feature_clusters is "
            "currently supported only by selected-feature precomputation and "
            "precomputed encoder-decoder pretraining."
        )


def reject_feature_clusters_metadata(meta: Any, surface: str) -> None:
    """Reject legacy or current clustered metadata at an unsupported consumer."""
    _reject_legacy_feature_clustering_metadata(meta, surface)
    if not isinstance(meta, Mapping):
        return
    active = (
        meta.get(FEATURE_CLUSTERS_DESCRIPTOR_KEY) is not None
        or meta.get(FEATURE_CLUSTERS_CANONICAL_KEY) is not None
    )
    user_config = meta.get("user_config")
    if isinstance(user_config, Mapping) and user_config.get("feature_clusters") is not None:
        active = True
    manifest_clusters = meta.get("feature_clusters")
    if manifest_clusters is not None:
        active = True
    if active:
        raise FeatureClustersError(
            f"{surface} cannot consume a checkpoint/artifact trained with static "
            "feature clusters. Cluster semantics are currently supported only by "
            "precomputed encoder-decoder pretraining selections."
        )


def resolve_feature_clusters_from_checkpoint_meta(
    meta: Any,
    *,
    surface: str,
    source_path: str | os.PathLike[str],
) -> ResolvedFeatureClusters | None:
    """Rebuild static feature clusters from complete checkpoint metadata.

    Legacy singular ``feature_clustering`` metadata is rejected first.  The
    descriptor, canonical definition, and active user-config marker are then
    considered together: all absent means clusters are disabled, while any
    partial descriptor/canonical presence raises a surface-named
    :class:`FeatureClustersError`.  Parsed metadata is validated strictly and
    ``source_path`` is retained only as provenance.
    """
    _reject_legacy_feature_clustering_metadata(meta, surface)
    mapping = meta if isinstance(meta, Mapping) else {}
    descriptor_raw = mapping.get(FEATURE_CLUSTERS_DESCRIPTOR_KEY)
    canonical_raw = mapping.get(FEATURE_CLUSTERS_CANONICAL_KEY)
    user_config = mapping.get("user_config")
    user_active = isinstance(user_config, Mapping) and (
        user_config.get("feature_clusters") is not None
    )
    if descriptor_raw is None and canonical_raw is None and not user_active:
        return None
    if descriptor_raw is None or canonical_raw is None:
        raise FeatureClustersError(
            f"{surface} has incomplete static feature-cluster metadata "
            "(descriptor/canonical must both be present); start fresh or use a "
            "checkpoint written by the clustered preselection path"
        )
    try:
        descriptor = FeatureClusterDescriptor.from_dict(descriptor_raw)
        definition = FeatureClusterDefinition.from_mapping(
            canonical_raw,
            n_sae_features=descriptor.n_sae_features,
        )
    except FeatureClustersError as exc:
        raise FeatureClustersError(
            f"{surface} has invalid static feature-cluster metadata: {exc}"
        ) from exc
    return ResolvedFeatureClusters(
        source_path=Path(source_path),
        definition=definition,
        descriptor=descriptor,
    )
