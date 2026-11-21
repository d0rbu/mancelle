"""Safetensors I/O for target-aware eval context feature caches.

The cache stores real, precomputed SAE-derived context features keyed by the
exact nanochat token sequence used during eval.  It intentionally mirrors the
CSR-style sparse conventions used by the feature extraction pipeline while
remaining a separate artifact type: rows are eval contexts, not training
documents.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch as t
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, canonical_dict
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    validate_context_token_assignments,
)
from steering.eval.turn_context_contract import (
    TurnContextContract,
    turn_context_contract_hash,
    turn_context_identity_fields,
    validate_turn_context_contract,
)
from steering.overlay_io import (
    COMPRESSED_EXT,
    load_overlay_shard_bytes,
    save_overlay_shard,
)
from steering.refined_types import PositiveInt, PositiveIntLike
from steering.segmentation import SegmentMode

if TYPE_CHECKING:
    # Type-only: steering.eval.clustered_cache_seed_contract pulls in
    # steering.eval.cluster_checkpoint_pin (torch + torch.distributed) for its
    # checkpoint-pin resolution, which this foundational I/O module has no
    # runtime need for. No cycle either direction (clustered_cache_seed_contract
    # does not import this module); this stays TYPE_CHECKING-only purely to keep
    # that import edge out of this module's real dependency graph.
    from steering.eval.clustered_cache_seed_contract import CheckpointSeedProvenance

LEGACY_SCHEMA_VERSION = 3
SCHEMA_VERSION = 4
TURN_SCHEMA_VERSION = 5
TURN_CONTRACT_KEY = "turn_contract"
TURN_MANIFEST_FIELDS = (
    "source_segment_mode",
    "source_segment_size", "segment_char_coordinate_system", "source_bos_char_len",
)
FORMAT_NAME = "eval_context_safetensors"
MANIFEST_FILENAME = "manifest.json"
SOURCE_SEGMENT_MODE_KEY = "source_segment_mode"
CONTEXT_LEN_DIR_RE = re.compile(r"^c([1-9]\d*)$")
FEATURE_SELECTION_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
HASH_LEAF_RE = re.compile(r"^[0-9a-f]{16}$")
HASH_TRUNCATION_LEN = 16
CACHE_LAYOUT_NAME = "cseq_hash16"
CACHE_LAYOUT_METADATA_KEY = "cache_layout"
CACHE_HASH_LEAF_METADATA_KEY = "cache_hash_leaf"
REMOVED_FEATURES_MANIFEST_KEY = "removed_features"
REMOVED_FEATURES_MARKER = {"feature_clustering": "unsupported"}
CLUSTER_SELECTION_SEMANTICS_KEY = "cluster_selection_semantics"
# D-dedup (PR #739 Track D) bump: v1's "post_mask" prefix described
# steering/eval/context_cache_build.py's pre-delegation clustered branch,
# which scored percentiles AFTER masking, using raw-float percentiles and
# t.multinomial weighted sampling. v2 delegates to the merged A5 primitive
# (steering/conditional/cluster_segment_selection.py::select_clustered_
# segment), which scores from the RAW (pre-mask) ids, uint8-quantizes
# percentiles, and uses Gumbel-max weighted sampling -- a "post_mask" name
# would now be actively misleading, not just stale. The version bump alone
# (independent of the name text) is what matters operationally: it forces
# SafetensorsContextFeatureProvider's semantics check (below) to reject any
# clustered cache built by the OLD (v1) code, since v1 caches hold rows
# selected by the old multinomial/raw-float/adaptive-negative-factor
# algorithm and must not be silently reused as if they were v2 rows.
CLUSTER_SELECTION_SEMANTICS = "pre_mask_representative_max_percentile_v2"
# Identity fields that are part of the cache key's "realm". Changing any of
# these fields means a different cache — caches built for one (model, SAE,
# layer, pooling, feature_selection) combination MUST NOT be loaded for any
# other combination.
_IDENTITY_FIELDS = (
    "model_id",
    "sae_id",
    "layer",
    "pooling",
    "feature_selection_hash",
)

T_CONTEXT_HASH = "context_hash"
T_CONTEXT_TOKEN_OFFSETS = "context_token_offsets"
T_CONTEXT_TOKEN_IDS = "context_token_ids"
T_CONTEXT_SEGMENT_OFFSETS = "context_segment_offsets"
T_SEGMENT_CHAR_STARTS = "segment_char_starts"
T_SEGMENT_CHAR_ENDS = "segment_char_ends"
T_FEATURE_OFFSETS = "feature_offsets"
T_FEATURE_IDS = "feature_ids"
T_BUCKETS = "buckets"


class ContextCacheValidationError(ValueError):
    """Raised when a context cache is malformed or incompatible."""


def clustered_context_cache_metadata(
    *,
    feature_mask: FeatureMask | None,
    feature_clusters: FeatureClusterConsolidation,
    feature_vocab: FeatureVocab | None,
    stats_source_sha256: str,
    checkpoint_seed_provenance: CheckpointSeedProvenance | None = None,
) -> dict:
    """Return path-independent metadata for one clustered prompt cache.

    Every identity object is the existing production object or its established
    checkpoint metadata form. This function deliberately introduces no second
    mask/cluster/vocabulary descriptor schema.

    ``checkpoint_seed_provenance`` is additive and optional (PR #739 Track D,
    D-contract): when a caller resolved a
    ``steering.eval.clustered_cache_seed_contract.CheckpointSeedProvenance``
    for the target checkpoint (its real seed + file provenance), five
    ``checkpoint_``-prefixed provenance keys are stamped alongside the
    existing keys above, none of which share that prefix (no collision).
    Omitting it (the default) is byte-identical to this function's
    pre-Track-D behavior -- both out-of-scope callers
    (``scripts/eval/validate_eval_context_cache.py``,
    ``scripts/train/conditional_encdec_train.py``) never pass it.
    """
    expected_vocab = effective_feature_vocab(
        feature_clusters,
        feature_clusters.n_sae_features,
    )
    if expected_vocab != feature_vocab:
        raise ValueError(
            "clustered context-cache FeatureVocab does not match the resolved "
            "mask/cluster effective vocabulary"
        )
    if not isinstance(stats_source_sha256, str) or not FEATURE_SELECTION_HASH_RE.fullmatch(
        stats_source_sha256
    ):
        raise ValueError(
            "clustered context-cache stats_source_sha256 must be 64 lowercase "
            "hexadecimal characters"
        )
    metadata = {
        "feature_mask_canonical": canonical_dict(feature_mask),
        **feature_clusters_checkpoint_metadata(feature_clusters.resolved),
        FEATURE_VOCAB_META_KEY: (None if feature_vocab is None else feature_vocab.metadata_dict()),
        CLUSTER_SELECTION_SEMANTICS_KEY: CLUSTER_SELECTION_SEMANTICS,
        "feature_delta": None,
        "stats_source_sha256": stats_source_sha256,
    }
    if checkpoint_seed_provenance is not None:
        metadata.update(checkpoint_seed_provenance.metadata_dict())
    return metadata


def context_length_cache_dir(
    cache_root: str | Path,
    model_max_seq_len: int | str,
    *,
    hash_leaf: str | None = None,
) -> Path:
    """Return the canonical context-length/hash-specific cache directory.

    Eval context features are keyed by exact token sequences, and CORE runtime
    truncation depends on the model context length.  Keep separate caches under
    ``c{model_max_seq_len}``.  When ``hash_leaf`` is provided, keep different
    feature-selection identities under ``c{model_max_seq_len}/{hash_leaf}``.

    ``cache_root`` may be the cache root, an already-resolved ``c{N}``
    directory, or an already-resolved ``c{N}/{hash_leaf}`` directory.  Resolved
    inputs must match the requested context length and hash leaf.
    """

    try:
        context_len = int(model_max_seq_len)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "model_max_seq_len must be a positive integer when resolving an "
            f"eval context cache directory; got {model_max_seq_len!r}"
        ) from exc
    if context_len <= 0:
        raise ValueError(
            "model_max_seq_len must be a positive integer when resolving an "
            f"eval context cache directory; got {context_len}"
        )

    normalized_hash_leaf = None
    if hash_leaf is not None:
        normalized_hash_leaf = validate_hash_leaf(hash_leaf)

    root = Path(cache_root)
    context_dir_name = f"c{context_len}"
    root_context_match = CONTEXT_LEN_DIR_RE.fullmatch(root.name)
    if root_context_match:
        if root.name != context_dir_name:
            raise ValueError(
                "context cache layout mismatch: path is scoped to "
                f"{root.name}, expected {context_dir_name}: {root}"
            )
        return root / normalized_hash_leaf if normalized_hash_leaf is not None else root

    parent_context_match = CONTEXT_LEN_DIR_RE.fullmatch(root.parent.name)
    if parent_context_match:
        if root.parent.name != context_dir_name:
            raise ValueError(
                "context cache layout mismatch: path is scoped to "
                f"{root.parent.name}, expected {context_dir_name}: {root}"
            )
        if normalized_hash_leaf is not None and root.name != normalized_hash_leaf:
            raise ValueError(
                "context cache layout mismatch: path hash leaf "
                f"{root.name!r} does not match expected "
                f"{normalized_hash_leaf!r}: {root}"
            )
        return root

    if normalized_hash_leaf is not None:
        return root / context_dir_name / normalized_hash_leaf
    return root / context_dir_name


def validate_feature_selection_hash(
    value: str,
    *,
    field_name: str = "feature_selection_hash",
) -> str:
    """Return ``value`` normalized to lowercase after strict hash validation."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a 64-character hex string")
    normalized = value.lower()
    if not FEATURE_SELECTION_HASH_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a 64-character hex string; got {value!r}")
    return normalized


def feature_selection_hash_leaf(feature_selection_hash: str) -> str:
    """Return the canonical 16-character filesystem leaf for a full hash."""

    return validate_feature_selection_hash(feature_selection_hash)[:HASH_TRUNCATION_LEN]


def validate_hash_leaf(hash_leaf: str) -> str:
    """Return ``hash_leaf`` normalized to lowercase after strict validation."""

    if not isinstance(hash_leaf, str):
        raise TypeError("hash_leaf must be a string")
    normalized = hash_leaf.lower()
    if not HASH_LEAF_RE.fullmatch(normalized):
        raise ValueError(
            f"hash_leaf must be a {HASH_TRUNCATION_LEN}-character hex string; got {hash_leaf!r}"
        )
    return normalized


@dataclass(frozen=True)
class ContextCacheEntry:
    """One eval row, producing geometry, and selected feature segments.

    Transient turn entries retain zero-based token ownership into ``segments``;
    persistent fixed-token cache rows keep their historical character routing.
    """

    token_ids: tuple[int, ...]
    segments: tuple[ContextFeatureSegment, ...]
    segment_mode: SegmentMode = SegmentMode.fixed_tokens
    token_assignments: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        validate_context_token_assignments(self.token_ids, self.segments, self.token_assignments)

    @classmethod
    def from_sequences(
        cls,
        token_ids: Sequence[int],
        segments: Iterable[ContextFeatureSegment],
        *,
        segment_mode: SegmentMode = SegmentMode.fixed_tokens,
        token_assignments: Sequence[int] | None = None,
    ) -> ContextCacheEntry:
        return cls(
            token_ids=tuple(int(tok) for tok in token_ids),
            segments=tuple(segments),
            segment_mode=segment_mode,
            token_assignments=(None if token_assignments is None else tuple(token_assignments)),
        )


def require_identity_field(
    identity_cfg: Mapping[Any, Any],
    field_name: str,
    expected_type: type,
):
    """Read ``identity_cfg[field_name]`` and coerce to ``expected_type``.

    Raises a ``ValueError`` with an operator-actionable message on missing
    or wrongly-typed values. Used when constructing ``CacheIdentity`` from
    Hydra config for shared context caches and target-real caches.
    """
    value = identity_cfg.get(field_name)
    if value is None:
        raise ValueError(
            f"conditioning.identity.{field_name} is required but missing or null. "
            f"Set it in the cfg (expected type: {expected_type.__name__})."
        )
    try:
        return expected_type(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"conditioning.identity.{field_name} must be {expected_type.__name__}, got {value!r}."
        ) from e


@dataclass(frozen=True)
class CacheIdentity:
    """Mandatory identity of the (model, SAE, layer, pooling, feature-selection)
    combination that a context cache was built for.

    Two caches built with different identities MUST NOT share rows — a cache
    key is the SHA-256 of ``(identity, token_ids)``; changing any identity
    field produces a fresh digest and thus a guaranteed cache miss.
    """

    model_id: str
    sae_id: str
    layer: int
    pooling: str
    feature_selection_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("CacheIdentity.model_id must be a non-empty string")
        if not isinstance(self.sae_id, str) or not self.sae_id:
            raise ValueError("CacheIdentity.sae_id must be a non-empty string")
        if not isinstance(self.layer, int) or isinstance(self.layer, bool):
            raise ValueError("CacheIdentity.layer must be int")
        if not isinstance(self.pooling, str) or not self.pooling:
            raise ValueError("CacheIdentity.pooling must be a non-empty string")
        if self.feature_selection_hash is None:
            raise ValueError(
                "CacheIdentity.feature_selection_hash is required for eval "
                "context caches. Set an explicit 64-character hex hash or "
                "provide canonical feature-selection pipeline params so it "
                "can be computed."
            )
        object.__setattr__(
            self,
            "feature_selection_hash",
            validate_feature_selection_hash(
                self.feature_selection_hash,
                field_name="CacheIdentity.feature_selection_hash",
            ),
        )

    def as_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "sae_id": self.sae_id,
            "layer": int(self.layer),
            "pooling": self.pooling,
            "feature_selection_hash": self.feature_selection_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheIdentity:
        missing = [k for k in _IDENTITY_FIELDS if k not in data]
        if missing:
            raise ContextCacheValidationError(f"CacheIdentity missing required fields: {missing}")
        return cls(
            model_id=str(data["model_id"]),
            sae_id=str(data["sae_id"]),
            layer=int(data["layer"]),
            pooling=str(data["pooling"]),
            feature_selection_hash=str(data["feature_selection_hash"]),
        )


def _identity_digest_prefix(identity: CacheIdentity) -> bytes:
    """Canonical bytes for an identity, used to scope the token digest."""

    payload = json.dumps(
        identity.as_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _scoped_row_digest(identity: CacheIdentity, token_ids: Sequence[int]) -> bytes:
    """Raw 32-byte digest for a row: identity-scoped token digest."""

    h = hashlib.sha256()
    h.update(_identity_digest_prefix(identity))
    for token_id in token_ids:
        h.update(int(token_id).to_bytes(8, byteorder="little", signed=True))
    return h.digest()


def _turn_identity_disagreements(
    identity: Mapping[str, Any] | None, contract: TurnContextContract
) -> list[str]:
    """Name the identity fields that disagree, not just the derived hash."""

    expected = {
        **turn_context_identity_fields(contract),
        "feature_selection_hash": turn_context_contract_hash(contract),
    }
    observed = identity if isinstance(identity, Mapping) else {}
    # Key presence is part of the identity: an extra or missing key disagrees.
    return sorted(key for key in set(expected) | set(observed)
                  if key not in expected or key not in observed or observed[key] != expected[key])


@beartype
def save_context_cache(
    cache_dir: str | Path,
    entries: Sequence[ContextCacheEntry],
    *,
    identity: CacheIdentity,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    metadata: dict | None = None,
    shard_size: PositiveIntLike = 1024,
    overwrite: bool = False,
    zstd_level: int | None = None,
    source_segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    turn_contract: TurnContextContract | None = None,
) -> Path:
    """Write a complete eval context cache and return the manifest path.

    ``@beartype`` owns the unary SAE-width, bucket-count, and shard-size
    contracts. The body validates cache layout, entry relationships, and
    artifact identity.

    Raises:
        BeartypeCallHintParamViolation: A scalar argument violates its refined
            type, including a non-positive width, count, or shard size.
    """

    schema_version = SCHEMA_VERSION
    if source_segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        if turn_contract is None:
            raise ValueError("schema-5 turn cache requires a checkpoint contract")
        disagreeing = _turn_identity_disagreements(identity.as_dict(), turn_contract)
        if disagreeing:
            raise ValueError(f"turn cache identity disagrees with contract: {disagreeing}")
        if (n_sae_features, num_buckets) != (turn_contract.lineage.n_sae_features, turn_contract.lineage.num_buckets):
            raise ValueError("turn cache feature width/buckets disagree with contract")
        schema_version = TURN_SCHEMA_VERSION
    elif turn_contract is not None:
        raise ValueError("fixed cache cannot carry a turn contract")
    cache_path = Path(cache_dir)
    metadata_dict = dict(metadata or {})
    if turn_contract is not None:
        required = {**turn_contract.semantic, "model_max_seq_len": turn_contract.model_max_seq_len}
        if turn_contract.consolidation is not None:
            required[CLUSTER_SELECTION_SEMANTICS_KEY] = CLUSTER_SELECTION_SEMANTICS
        for key, value in required.items():
            if key in metadata_dict and metadata_dict[key] != value:
                raise ValueError(f"turn cache metadata {key} disagrees with contract")
        metadata_dict.update(required)
    _stamp_cache_layout_metadata(metadata_dict, identity=identity)
    _validate_cache_path_layout(
        cache_path,
        identity=identity,
        model_max_seq_len=metadata_dict.get("model_max_seq_len"),
        error_type=ValueError,
    )
    manifest_path = cache_path / MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Context cache already exists at {manifest_path}; pass overwrite=True"
        )
    cache_path.mkdir(parents=True, exist_ok=True)

    seen: dict[bytes, tuple[int, ...]] = {}
    for entry in entries:
        _validate_entry(entry, n_sae_features=n_sae_features, num_buckets=num_buckets,
                        source_segment_mode=source_segment_mode)
        digest = _scoped_row_digest(identity, entry.token_ids)
        if digest in seen:
            if seen[digest] == entry.token_ids:
                raise ContextCacheValidationError(
                    "Duplicate context token sequence in cache entries; "
                    "dedupe eval contexts before writing the cache"
                )
            raise ContextCacheValidationError(
                "Two different token sequences produced the same context digest; "
                "refusing to write ambiguous cache"
            )
        seen[digest] = entry.token_ids

    shard_infos = []
    total_segments = 0
    total_features = 0
    for shard_idx, start in enumerate(range(0, len(entries), shard_size)):
        shard_entries = entries[start : start + shard_size]
        shard_name = f"contexts_{shard_idx:05d}{COMPRESSED_EXT}"
        shard_path = cache_path / shard_name
        tensors = _entries_to_tensors(shard_entries, identity=identity)
        save_overlay_shard(
            tensors,
            shard_path,
            metadata={
                "format": FORMAT_NAME,
                "schema_version": str(schema_version),
                "shard_index": str(shard_idx),
            },
            zstd_level=zstd_level,
        )
        shard_segments = int(tensors[T_SEGMENT_CHAR_STARTS].numel())
        shard_features = int(tensors[T_FEATURE_IDS].numel())
        total_segments += shard_segments
        total_features += shard_features
        shard_infos.append(
            {
                "file": shard_name,
                "n_contexts": len(shard_entries),
                "n_segments": shard_segments,
                "n_features": shard_features,
                "sha256": _sha256_file(shard_path),
            }
        )

    manifest = {
        "schema_version": schema_version,
        "format": FORMAT_NAME,
        # Schema 4 retains fixed coordinates; schema 5 requires a complete
        # checkpoint-derived turn contract before any source spans are consumed.
        SOURCE_SEGMENT_MODE_KEY: source_segment_mode.value,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_contexts": len(entries),
        "n_segments": total_segments,
        "n_features": total_features,
        "n_sae_features": int(n_sae_features),
        "num_buckets": int(num_buckets),
        "identity": identity.as_dict(),
        "metadata": metadata_dict,
        REMOVED_FEATURES_MANIFEST_KEY: dict(REMOVED_FEATURES_MARKER),
        "shards": shard_infos,
    }
    if turn_contract is not None:
        manifest[TURN_CONTRACT_KEY] = turn_contract.semantic
        manifest.update({key: turn_contract.semantic[key] for key in TURN_MANIFEST_FIELDS})
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


class SafetensorsContextFeatureProvider:
    """Lookup selected/bucketed context features from a safetensors cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        identity: CacheIdentity,
        expected_n_sae_features: int | None = None,
        expected_num_buckets: int | None = None,
        expected_metadata: dict | None = None,
        expected_feature_vocab: FeatureVocab | None = None,
        expected_source_segment_mode: SegmentMode = SegmentMode.fixed_tokens,
        expected_turn_contract: TurnContextContract | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / MANIFEST_FILENAME
        self.identity = identity
        self.turn_contract = expected_turn_contract
        if expected_turn_contract is not None:
            if expected_feature_vocab is not None and expected_feature_vocab != expected_turn_contract.feature_vocab:
                raise ContextCacheValidationError("explicit feature vocabulary conflicts with turn contract")
            expected_feature_vocab = expected_turn_contract.feature_vocab
        self.expected_feature_vocab = expected_feature_vocab
        self.manifest = load_context_cache_manifest(
            self.cache_dir,
            identity=identity,
            expected_n_sae_features=expected_n_sae_features,
            expected_num_buckets=expected_num_buckets,
            expected_metadata=expected_metadata,
            expected_source_segment_mode=expected_source_segment_mode,
            expected_turn_contract=expected_turn_contract,
        )
        if expected_feature_vocab is not None:
            if int(self.manifest["n_sae_features"]) != expected_feature_vocab.n_sae_features:
                raise ContextCacheValidationError(
                    "Context cache FeatureVocab original SAE width mismatch: "
                    f"cache={self.manifest['n_sae_features']}, "
                    f"expected={expected_feature_vocab.n_sae_features}"
                )
            metadata = self.manifest["metadata"]
            if metadata.get(FEATURE_VOCAB_META_KEY) != (expected_feature_vocab.metadata_dict()):
                raise ContextCacheValidationError(
                    "Context cache feature-vocabulary metadata mismatch: "
                    f"cache={metadata.get(FEATURE_VOCAB_META_KEY)!r}, "
                    f"expected={expected_feature_vocab.metadata_dict()!r}"
                )
            if expected_turn_contract is None and metadata.get(CLUSTER_SELECTION_SEMANTICS_KEY) != (CLUSTER_SELECTION_SEMANTICS):
                raise ContextCacheValidationError(
                    "Context cache is missing the expected clustered selection "
                    f"semantics {CLUSTER_SELECTION_SEMANTICS!r}"
                )
        self._rows: dict[bytes, tuple[tuple[int, ...], tuple[ContextFeatureSegment, ...]]] = {}
        self._load_all_shards()

    def get(self, token_ids: Sequence[int]) -> tuple[ContextFeatureSegment, ...] | None:
        tokens = tuple(int(tok) for tok in token_ids)
        digest = _scoped_row_digest(self.identity, tokens)
        row = self._rows.get(digest)
        if row is None:
            return None
        stored_tokens, segments = row
        if stored_tokens != tokens:
            raise ContextCacheValidationError(
                "Context cache digest matched but token IDs differed; the cache index is corrupt"
            )
        return segments

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def path(self) -> Path:
        return self.manifest_path

    def _load_all_shards(self) -> None:
        for shard in self.manifest["shards"]:
            shard_path = self.cache_dir / shard["file"]
            if not shard_path.is_file():
                raise FileNotFoundError(f"Context cache shard not found: {shard_path}")
            compressed = shard_path.read_bytes()
            digest = hashlib.sha256(compressed).hexdigest()
            if digest != shard.get("sha256"):
                raise ContextCacheValidationError(
                    f"Shard checksum mismatch for {shard_path}: "
                    f"manifest={shard.get('sha256')}, actual={digest}"
                )
            tensors = load_overlay_shard_bytes(compressed)
            if self.expected_feature_vocab is not None:
                _validate_tensors(tensors, source=str(shard_path))
                feature_ids = tensors[T_FEATURE_IDS].to(dtype=t.int64, device="cpu")
                allowed = t.zeros(
                    self.expected_feature_vocab.n_sae_features,
                    dtype=t.bool,
                )
                allowed[self.expected_feature_vocab.new_to_old_tensor()] = True
                invalid = (feature_ids < 0) | (feature_ids >= allowed.numel())
                safe_ids = feature_ids.clamp(min=0, max=allowed.numel() - 1)
                invalid |= ~allowed[safe_ids]
                if bool(invalid.any()):
                    bad = int(feature_ids[invalid][0].item())
                    raise ContextCacheValidationError(
                        f"{shard_path}: cached feature ID {bad} is outside the "
                        "expected effective original-ID vocabulary"
                    )
            entries = _tensors_to_entries(
                tensors,
                identity=self.identity,
                n_sae_features=PositiveInt.parse(
                    int(self.manifest["n_sae_features"])
                ),
                num_buckets=PositiveInt.parse(int(self.manifest["num_buckets"])),
                source=str(shard_path),
            )
            if len(entries) != int(shard["n_contexts"]):
                raise ContextCacheValidationError(
                    f"{shard_path}: manifest says {shard['n_contexts']} contexts, "
                    f"loaded {len(entries)}"
                )
            for entry in entries:
                row_digest = _scoped_row_digest(self.identity, entry.token_ids)
                if row_digest in self._rows:
                    raise ContextCacheValidationError(
                        f"Duplicate context digest in cache: {row_digest.hex()}"
                    )
                self._rows[row_digest] = (entry.token_ids, entry.segments)

        if len(self._rows) != int(self.manifest["n_contexts"]):
            raise ContextCacheValidationError(
                f"Manifest says {self.manifest['n_contexts']} contexts, loaded {len(self._rows)}"
            )


def load_context_cache_manifest(
    cache_dir: str | Path,
    *,
    identity: CacheIdentity | None = None,
    expected_n_sae_features: int | None = None,
    expected_num_buckets: int | None = None,
    expected_metadata: dict | None = None,
    expected_source_segment_mode: SegmentMode = SegmentMode.fixed_tokens,
    expected_turn_contract: TurnContextContract | None = None,
) -> dict:
    """Load and validate a cache manifest."""

    cache_path = Path(cache_dir)
    manifest_path = cache_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Context cache manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ContextCacheValidationError(
            f"Context cache manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ContextCacheValidationError(
            f"Context cache manifest must be a JSON object: {manifest_path}"
        )

    _require_manifest_field(manifest, "schema_version", int)
    _require_manifest_field(manifest, "format", str)
    _require_manifest_field(manifest, "n_contexts", int)
    _require_manifest_field(manifest, "n_segments", int)
    _require_manifest_field(manifest, "n_features", int)
    _require_manifest_field(manifest, "n_sae_features", int)
    _require_manifest_field(manifest, "num_buckets", int)
    _require_manifest_field(manifest, "shards", list)

    expected_schema = SCHEMA_VERSION
    if expected_source_segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        if expected_turn_contract is None:
            raise ContextCacheValidationError("turn cache consumption requires a complete checkpoint-derived contract")
        expected_schema = TURN_SCHEMA_VERSION
    elif expected_turn_contract is not None:
        raise ContextCacheValidationError("fixed consumer cannot use a turn contract")
    if manifest["schema_version"] != expected_schema:
        raise ContextCacheValidationError(
            "Context cache schema has changed: manifest schema_version="
            f"{manifest['schema_version']}, expected {expected_schema}. "
            "This cache does not match the required identity and source "
            "segmentation schema. Re-run build "
            "to regenerate this cache under the new schema."
        )
    _require_manifest_field(manifest, SOURCE_SEGMENT_MODE_KEY, str)
    try:
        source_segment_mode = SegmentMode(manifest[SOURCE_SEGMENT_MODE_KEY])
    except ValueError as exc:
        raise ContextCacheValidationError(
            f"Context cache source_segment_mode is invalid: {manifest[SOURCE_SEGMENT_MODE_KEY]!r}"
        ) from exc
    if source_segment_mode is not expected_source_segment_mode:
        raise ContextCacheValidationError(
            f"Context cache source_segment_mode={source_segment_mode.value!r}, "
            f"expected {expected_source_segment_mode.value!r}. "
            "Rebuild the cache with matching source geometry."
        )
    if expected_turn_contract is not None:
        _require_manifest_field(manifest, "metadata", dict)
        _require_manifest_field(manifest, TURN_CONTRACT_KEY, dict)
        if expected_turn_contract.consolidation is not None:
            if manifest["metadata"].get(CLUSTER_SELECTION_SEMANTICS_KEY) != CLUSTER_SELECTION_SEMANTICS:
                raise ContextCacheValidationError(
                    "turn cache is missing the expected clustered selection "
                    f"semantics {CLUSTER_SELECTION_SEMANTICS!r}"
                )
        try:
            validate_turn_context_contract(manifest[TURN_CONTRACT_KEY], expected_turn_contract)
        except ValueError as exc:
            raise ContextCacheValidationError(str(exc)) from exc
        for key in TURN_MANIFEST_FIELDS:
            if manifest.get(key) != expected_turn_contract.semantic[key]:
                raise ContextCacheValidationError(f"turn cache {key} disagrees with contract")
        disagreeing = _turn_identity_disagreements(manifest.get("identity"), expected_turn_contract)
        if disagreeing:
            raise ContextCacheValidationError(
                f"turn cache identity disagrees with contract: {disagreeing}"
            )
        for key, value in {**expected_turn_contract.semantic,
                           "model_max_seq_len": expected_turn_contract.model_max_seq_len}.items():
            if manifest.get("metadata", {}).get(key) != value:
                raise ContextCacheValidationError(f"turn cache metadata {key} disagrees with contract")
        if (manifest["n_sae_features"], manifest["num_buckets"]) != (
            expected_turn_contract.lineage.n_sae_features, expected_turn_contract.lineage.num_buckets,
        ):
            raise ContextCacheValidationError("turn cache feature width/buckets disagree with contract")
    _require_manifest_field(manifest, REMOVED_FEATURES_MANIFEST_KEY, dict)
    if manifest[REMOVED_FEATURES_MANIFEST_KEY] != REMOVED_FEATURES_MARKER:
        raise ContextCacheValidationError(
            "Context cache manifest has an unexpected removed-features marker; "
            "re-run build to regenerate this cache."
        )
    if "identity" not in manifest or not isinstance(manifest.get("identity"), dict):
        raise ContextCacheValidationError(
            "Context cache schema has changed: manifest is missing the "
            "required 'identity' block (model_id/sae_id/layer/pooling/"
            "feature_selection_hash). This cache pre-dates the SAE-identity "
            "fix; re-run build to regenerate it."
        )
    try:
        manifest_identity = CacheIdentity.from_dict(manifest["identity"])
    except (ValueError, ContextCacheValidationError) as exc:
        raise ContextCacheValidationError(
            f"Context cache identity is malformed: {exc}. Re-run build to regenerate this cache."
        ) from exc
    manifest["identity"] = manifest_identity.as_dict()
    if identity is not None and identity != manifest_identity:
        raise ContextCacheValidationError(
            "Context cache identity mismatch: "
            f"cache={manifest_identity.as_dict()}, expected={identity.as_dict()}"
        )
    if manifest["format"] != FORMAT_NAME:
        raise ContextCacheValidationError(
            f"Unsupported context cache format={manifest['format']!r}; expected {FORMAT_NAME!r}"
        )
    if manifest["n_contexts"] < 0 or manifest["n_segments"] < 0 or manifest["n_features"] < 0:
        raise ContextCacheValidationError("Manifest counts must be non-negative")
    if manifest["n_sae_features"] <= 0:
        raise ContextCacheValidationError("Manifest n_sae_features must be positive")
    if manifest["num_buckets"] <= 0:
        raise ContextCacheValidationError("Manifest num_buckets must be positive")

    if expected_n_sae_features is not None and int(manifest["n_sae_features"]) != int(
        expected_n_sae_features
    ):
        raise ContextCacheValidationError(
            f"Context cache n_sae_features mismatch: "
            f"cache={manifest['n_sae_features']}, expected={expected_n_sae_features}"
        )
    if expected_num_buckets is not None and int(manifest["num_buckets"]) != int(
        expected_num_buckets
    ):
        raise ContextCacheValidationError(
            f"Context cache num_buckets mismatch: "
            f"cache={manifest['num_buckets']}, expected={expected_num_buckets}"
        )

    metadata = manifest.get("metadata", {})
    if metadata is None:
        metadata = {}
        manifest["metadata"] = metadata
    if not isinstance(metadata, dict):
        raise ContextCacheValidationError("Manifest metadata must be a JSON object")
    _validate_manifest_layout_metadata(metadata, identity=manifest_identity)
    _validate_cache_path_layout(
        cache_path,
        identity=manifest_identity,
        model_max_seq_len=metadata.get("model_max_seq_len"),
        error_type=ContextCacheValidationError,
    )
    for key, expected in (expected_metadata or {}).items():
        actual = metadata.get(key)
        if actual != expected:
            raise ContextCacheValidationError(
                f"Context cache metadata mismatch for {key!r}: "
                f"cache={actual!r}, expected={expected!r}"
            )

    for i, shard in enumerate(manifest["shards"]):
        if not isinstance(shard, dict):
            raise ContextCacheValidationError(f"Manifest shard {i} must be an object")
        for key, typ in [
            ("file", str),
            ("n_contexts", int),
            ("n_segments", int),
            ("n_features", int),
            ("sha256", str),
        ]:
            _require_manifest_field(shard, key, typ, prefix=f"shards[{i}]")
        if "/" in shard["file"] or "\\" in shard["file"]:
            raise ContextCacheValidationError(
                f"Manifest shard file must be a basename: {shard['file']!r}"
            )

    return manifest


def _stamp_cache_layout_metadata(metadata: dict, *, identity: CacheIdentity) -> None:
    expected_leaf = feature_selection_hash_leaf(identity.feature_selection_hash)
    existing_layout = metadata.get(CACHE_LAYOUT_METADATA_KEY)
    if existing_layout is not None and existing_layout != CACHE_LAYOUT_NAME:
        raise ValueError(
            f"metadata.{CACHE_LAYOUT_METADATA_KEY} must be {CACHE_LAYOUT_NAME!r}; "
            f"got {existing_layout!r}"
        )
    existing_leaf = metadata.get(CACHE_HASH_LEAF_METADATA_KEY)
    if existing_leaf is not None and validate_hash_leaf(existing_leaf) != expected_leaf:
        raise ValueError(
            f"metadata.{CACHE_HASH_LEAF_METADATA_KEY}={existing_leaf!r} does "
            f"not match identity feature_selection_hash leaf {expected_leaf!r}"
        )
    metadata[CACHE_LAYOUT_METADATA_KEY] = CACHE_LAYOUT_NAME
    metadata[CACHE_HASH_LEAF_METADATA_KEY] = expected_leaf


def _validate_manifest_layout_metadata(metadata: dict, *, identity: CacheIdentity) -> None:
    expected_leaf = feature_selection_hash_leaf(identity.feature_selection_hash)
    layout = metadata.get(CACHE_LAYOUT_METADATA_KEY)
    if layout is not None and layout != CACHE_LAYOUT_NAME:
        raise ContextCacheValidationError(
            f"Context cache manifest metadata {CACHE_LAYOUT_METADATA_KEY!r} "
            f"must be {CACHE_LAYOUT_NAME!r}; got {layout!r}"
        )
    leaf = metadata.get(CACHE_HASH_LEAF_METADATA_KEY)
    if leaf is not None:
        try:
            normalized_leaf = validate_hash_leaf(leaf)
        except (TypeError, ValueError) as exc:
            raise ContextCacheValidationError(
                f"Context cache manifest metadata {CACHE_HASH_LEAF_METADATA_KEY!r} "
                f"is invalid: {leaf!r}"
            ) from exc
        if normalized_leaf != expected_leaf:
            raise ContextCacheValidationError(
                f"Context cache manifest metadata {CACHE_HASH_LEAF_METADATA_KEY!r} "
                f"{normalized_leaf!r} does not match identity hash leaf "
                f"{expected_leaf!r}"
            )


def _validate_cache_path_layout(
    cache_path: Path,
    *,
    identity: CacheIdentity,
    model_max_seq_len,
    error_type: type[Exception],
) -> None:
    """Validate canonical ``.../c{N}/{hash[:16]}`` layout when length is known."""

    expected_leaf = feature_selection_hash_leaf(identity.feature_selection_hash)
    if model_max_seq_len is None:
        if CONTEXT_LEN_DIR_RE.fullmatch(cache_path.name):
            raise error_type(
                "Context cache layout mismatch: identity-scoped caches must "
                f"live under .../{cache_path.name}/{expected_leaf}/"
                f"{MANIFEST_FILENAME}; got {cache_path / MANIFEST_FILENAME}"
            )
        if (
            CONTEXT_LEN_DIR_RE.fullmatch(cache_path.parent.name)
            and cache_path.name != expected_leaf
        ):
            raise error_type(
                "Context cache layout mismatch: path hash leaf "
                f"{cache_path.name!r} does not match identity hash leaf "
                f"{expected_leaf!r}; got {cache_path / MANIFEST_FILENAME}"
            )
        return
    try:
        context_len = int(model_max_seq_len)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"Context cache layout metadata model_max_seq_len must be an "
            f"integer; got {model_max_seq_len!r}"
        ) from exc
    if context_len <= 0:
        raise error_type(
            f"Context cache layout metadata model_max_seq_len must be positive; got {context_len}"
        )
    expected_context_dir = f"c{context_len}"
    if cache_path.parent.name != expected_context_dir or cache_path.name != expected_leaf:
        expected_suffix = f"{expected_context_dir}/{expected_leaf}/{MANIFEST_FILENAME}"
        raise error_type(
            "Context cache layout mismatch: expected manifest under "
            f".../{expected_suffix} for identity hash "
            f"{identity.feature_selection_hash!r}; got {cache_path / MANIFEST_FILENAME}"
        )


def _entries_to_tensors(
    entries: Sequence[ContextCacheEntry],
    *,
    identity: CacheIdentity,
) -> dict[str, t.Tensor]:
    hash_rows: list[list[int]] = []
    token_offsets = [0]
    token_ids: list[int] = []
    segment_offsets = [0]
    char_starts: list[int] = []
    char_ends: list[int] = []
    feature_offsets = [0]
    feature_ids: list[int] = []
    buckets: list[int] = []

    for entry in entries:
        hash_rows.append(list(_scoped_row_digest(identity, entry.token_ids)))
        token_ids.extend(entry.token_ids)
        token_offsets.append(len(token_ids))
        segment_offsets.append(segment_offsets[-1] + len(entry.segments))
        for segment in entry.segments:
            char_starts.append(segment.char_start)
            char_ends.append(segment.char_end)
            feature_ids.extend(segment.feature_ids)
            buckets.extend(segment.buckets)
            feature_offsets.append(len(feature_ids))

    if hash_rows:
        context_hash = t.tensor(hash_rows, dtype=t.uint8)
    else:
        context_hash = t.empty((0, 32), dtype=t.uint8)

    return {
        T_CONTEXT_HASH: context_hash,
        T_CONTEXT_TOKEN_OFFSETS: t.tensor(token_offsets, dtype=t.int64),
        T_CONTEXT_TOKEN_IDS: t.tensor(token_ids, dtype=t.int64),
        T_CONTEXT_SEGMENT_OFFSETS: t.tensor(segment_offsets, dtype=t.int64),
        T_SEGMENT_CHAR_STARTS: t.tensor(char_starts, dtype=t.int32),
        T_SEGMENT_CHAR_ENDS: t.tensor(char_ends, dtype=t.int32),
        T_FEATURE_OFFSETS: t.tensor(feature_offsets, dtype=t.int64),
        T_FEATURE_IDS: t.tensor(feature_ids, dtype=t.int64),
        T_BUCKETS: t.tensor(buckets, dtype=t.int64),
    }


def _tensors_to_entries(
    tensors: dict[str, t.Tensor],
    *,
    identity: CacheIdentity,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    source: str,
) -> list[ContextCacheEntry]:
    _validate_tensors(tensors, source=source)

    context_hash = tensors[T_CONTEXT_HASH].cpu()
    token_offsets = tensors[T_CONTEXT_TOKEN_OFFSETS].cpu().tolist()
    token_ids = tensors[T_CONTEXT_TOKEN_IDS].cpu().tolist()
    segment_offsets = tensors[T_CONTEXT_SEGMENT_OFFSETS].cpu().tolist()
    char_starts = tensors[T_SEGMENT_CHAR_STARTS].cpu().tolist()
    char_ends = tensors[T_SEGMENT_CHAR_ENDS].cpu().tolist()
    feature_offsets = tensors[T_FEATURE_OFFSETS].cpu().tolist()
    feature_ids = tensors[T_FEATURE_IDS].cpu().tolist()
    buckets = tensors[T_BUCKETS].cpu().tolist()

    entries: list[ContextCacheEntry] = []
    for row in range(context_hash.shape[0]):
        tok_start, tok_end = token_offsets[row], token_offsets[row + 1]
        seg_start, seg_end = segment_offsets[row], segment_offsets[row + 1]
        row_tokens = tuple(int(tok) for tok in token_ids[tok_start:tok_end])
        expected_hash = bytes(int(v) for v in context_hash[row].tolist())
        actual_hash = _scoped_row_digest(identity, row_tokens)
        if actual_hash != expected_hash:
            raise ContextCacheValidationError(
                f"{source}: context row {row} hash does not match token_ids"
            )

        segments: list[ContextFeatureSegment] = []
        for seg_idx in range(seg_start, seg_end):
            feat_start, feat_end = feature_offsets[seg_idx], feature_offsets[seg_idx + 1]
            segment_source = f"{source}:context[{row}].segment[{seg_idx - seg_start}]"
            try:
                segment = ContextFeatureSegment(
                    feature_ids=tuple(
                        int(fid) for fid in feature_ids[feat_start:feat_end]
                    ),
                    buckets=tuple(
                        int(bucket) for bucket in buckets[feat_start:feat_end]
                    ),
                    char_start=int(char_starts[seg_idx]),
                    char_end=int(char_ends[seg_idx]),
                )
            except BeartypeCallHintParamViolation as exc:
                raise ContextCacheValidationError(
                    f"{segment_source}: feature ids, buckets, and character "
                    "offsets must be strict non-negative integers"
                ) from exc
            _validate_segment(
                segment,
                n_sae_features=n_sae_features,
                num_buckets=num_buckets,
                source=segment_source,
            )
            segments.append(segment)
        entries.append(ContextCacheEntry(row_tokens, tuple(segments)))

    return entries


def _validate_tensors(tensors: dict[str, t.Tensor], *, source: str) -> None:
    required = [
        T_CONTEXT_HASH,
        T_CONTEXT_TOKEN_OFFSETS,
        T_CONTEXT_TOKEN_IDS,
        T_CONTEXT_SEGMENT_OFFSETS,
        T_SEGMENT_CHAR_STARTS,
        T_SEGMENT_CHAR_ENDS,
        T_FEATURE_OFFSETS,
        T_FEATURE_IDS,
        T_BUCKETS,
    ]
    missing = [key for key in required if key not in tensors]
    if missing:
        raise ContextCacheValidationError(f"{source}: missing tensor(s): {missing}")

    context_hash = tensors[T_CONTEXT_HASH]
    if context_hash.ndim != 2 or context_hash.shape[1] != 32:
        raise ContextCacheValidationError(
            f"{source}: context_hash must have shape (n_contexts, 32)"
        )
    n_contexts = int(context_hash.shape[0])
    _validate_offsets(
        tensors[T_CONTEXT_TOKEN_OFFSETS],
        expected_len=n_contexts + 1,
        terminal=int(tensors[T_CONTEXT_TOKEN_IDS].numel()),
        source=f"{source}:context_token_offsets",
    )
    _validate_offsets(
        tensors[T_CONTEXT_SEGMENT_OFFSETS],
        expected_len=n_contexts + 1,
        terminal=int(tensors[T_SEGMENT_CHAR_STARTS].numel()),
        source=f"{source}:context_segment_offsets",
    )
    n_segments = int(tensors[T_SEGMENT_CHAR_STARTS].numel())
    if int(tensors[T_SEGMENT_CHAR_ENDS].numel()) != n_segments:
        raise ContextCacheValidationError(
            f"{source}: segment_char_starts and segment_char_ends lengths differ"
        )
    _validate_offsets(
        tensors[T_FEATURE_OFFSETS],
        expected_len=n_segments + 1,
        terminal=int(tensors[T_FEATURE_IDS].numel()),
        source=f"{source}:feature_offsets",
    )
    if int(tensors[T_BUCKETS].numel()) != int(tensors[T_FEATURE_IDS].numel()):
        raise ContextCacheValidationError(f"{source}: feature_ids and buckets lengths differ")


def _validate_offsets(
    offsets: t.Tensor,
    *,
    expected_len: int,
    terminal: int,
    source: str,
) -> None:
    if offsets.ndim != 1 or int(offsets.numel()) != expected_len:
        raise ContextCacheValidationError(
            f"{source}: expected length {expected_len}, got shape {tuple(offsets.shape)}"
        )
    if expected_len == 0:
        return
    offsets_cpu = offsets.cpu()
    if int(offsets_cpu[0].item()) != 0:
        raise ContextCacheValidationError(f"{source}: first offset must be 0")
    if int(offsets_cpu[-1].item()) != terminal:
        raise ContextCacheValidationError(
            f"{source}: terminal offset {int(offsets_cpu[-1].item())} != {terminal}"
        )
    if bool((offsets_cpu[1:] < offsets_cpu[:-1]).any().item()):
        raise ContextCacheValidationError(f"{source}: offsets must be monotonic")


def _validate_entry(
    entry: ContextCacheEntry,
    *,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    source_segment_mode: SegmentMode,
) -> None:
    if entry.segment_mode is not source_segment_mode or entry.token_assignments is not None:
        raise ValueError(
            f"persistent {source_segment_mode.value} context caches support only "
            f"{source_segment_mode.value} entries without ownership"
        )
    if not isinstance(entry.token_ids, tuple):
        raise TypeError("ContextCacheEntry.token_ids must be a tuple")
    for token_id in entry.token_ids:
        int(token_id).to_bytes(8, byteorder="little", signed=True)
    for seg_idx, segment in enumerate(entry.segments):
        _validate_segment(
            segment,
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            source=f"entry.segment[{seg_idx}]",
        )


def _validate_segment(
    segment: ContextFeatureSegment,
    *,
    n_sae_features: PositiveInt,
    num_buckets: PositiveInt,
    source: str,
) -> None:
    if len(segment.feature_ids) != len(segment.buckets):
        raise ContextCacheValidationError(
            f"{source}: feature_ids and buckets lengths differ "
            f"({len(segment.feature_ids)} vs {len(segment.buckets)})"
        )
    if segment.char_end < segment.char_start:
        raise ContextCacheValidationError(
            f"{source}: char_end {segment.char_end} < char_start {segment.char_start}"
        )
    if len(set(segment.feature_ids)) != len(segment.feature_ids):
        raise ContextCacheValidationError(f"{source}: duplicate feature ids")
    for fid in segment.feature_ids:
        if fid >= n_sae_features:
            raise ContextCacheValidationError(
                f"{source}: feature id {fid} out of range [0, {n_sae_features})"
            )
    for bucket in segment.buckets:
        if bucket >= num_buckets:
            raise ContextCacheValidationError(
                f"{source}: bucket {bucket} out of range [0, {num_buckets})"
            )


def _require_manifest_field(data: dict, key: str, typ: type, *, prefix: str = "") -> None:
    label = f"{prefix}.{key}" if prefix else key
    if key not in data:
        raise ContextCacheValidationError(f"Manifest missing required field {label}")
    if not isinstance(data[key], typ):
        raise ContextCacheValidationError(
            f"Manifest field {label} must be {typ.__name__}, got {type(data[key]).__name__}"
        )


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.rename(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
