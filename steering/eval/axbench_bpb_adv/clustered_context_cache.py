"""Cluster-safe AxBench-BPB-Adv context-cache construction.

Clustered checkpoints cannot use the ordinary AxBench context conversion:
checkpoint ``FeatureVocab`` still contains mask-allowed rows that were
consolidated away before training selection.  This module therefore uses a
clustered-only, two-phase protocol:

1. DDP owners extract and atomically persist raw per-segment SAE IDs and
   magnitudes. No feature-selection RNG is consumed in this phase.
2. Every rank rebuilds a private CPU feature pipeline and replays every prefix
   in global sorted order. Only the deterministic owner writes (or
   byte-compares) a selected prefix artifact.

The manifest is written last and is the sole completeness marker. The partial
marker pins world size because a resumed partial replay must retain the same
ownership layout; a completed manifest deliberately omits world size. All
artifact digests and parsed facts come from the same in-memory byte snapshot.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Never, cast

import torch as t
import torch.distributed as dist
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from beartype.vale import Is
from loguru import logger
from phantom.sized import NonEmptyStr
from safetensors.torch import load as load_safetensors_bytes

from steering._common.strict_json import canonical_json
from steering.conditional.cluster_segment_selection import (
    ClusteredSegmentSelection,
    select_clustered_segment_from_magnitudes,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    SamplingBundle,
    ThresholdSelectionConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.eval.segment_mode_guard import require_fixed_segment_mode
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.extract_features import extract_document_features
from steering.feature_paths import PoolingMode
from steering.immutable_io import (
    ImmutableArtifactError,
    install_bytes_once,
    is_sha256_hex,
)
from steering.refined_types import (
    LowercaseSha256Like,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
)
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    read_safetensors_metadata_from_bytes,
)
from steering.segmentation import (
    SegmentMode,
    align_char_ranges_to_decoded_prefix,
    token_to_char_offsets,
    truncate_to_training_window,
)
from steering.stats_io import load_statistics_and_checksum_from_bytes

if TYPE_CHECKING:
    _ClusterFeatureContractLike = ClusterFeatureContract
else:
    # Unit-level raw extraction doubles intentionally expose the validated
    # contract surface structurally without constructing its file-backed owner.
    _ClusterFeatureContractLike = Any

CLUSTERED_CONTEXT_CACHE_SCHEMA_VERSION = 1
CLUSTERED_CONTEXT_CACHE_KIND = "axbench_bpb_adv_clustered_context_cache"
RAW_PREFIX_SCHEMA_VERSION = 1
SELECTED_PREFIX_SCHEMA_VERSION = 1
PARTIAL_CACHE_SCHEMA_VERSION = 1
CLUSTERED_CONTEXT_CHAR_ALIGNMENT_VERSION = "source_bos_inclusive_end_marker_reconcile_v2"
_CLUSTERED_CONTEXT_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "axbench_bpb_adv.clustered_context_cache.char_alignment"
)


def _is_prefix_population_entries(value: object) -> bool:
    return isinstance(value, tuple) and all(
        isinstance(entry, tuple)
        and len(entry) == 2
        and all(isinstance(item, str) for item in entry)
        for entry in value
    )


def _is_prefix_population_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(prefix_text, str) and isinstance(prefix_hash, str)
        for prefix_text, prefix_hash in value.items()
    )


def _is_prefix_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_string_key_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def _is_string_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


def _is_string_key_dict_list(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and _is_string_key_mapping(item) for item in value
    )


if TYPE_CHECKING:
    _AnyRecordLike = dict[str, Any]
    _AnyRecordListLike = list[dict[str, Any]]
    _PrefixPopulationEntriesLike = tuple[tuple[str, str], ...]
    _PrefixPopulationMappingLike = Mapping[str, str]
    _PrefixHashLike = str
    _StringMappingLike = Mapping[str, str]
    _StringRecordLike = dict[str, str]
else:
    _AnyRecordLike = Annotated[dict, Is[_is_string_key_mapping]]
    _AnyRecordListLike = Annotated[list, Is[_is_string_key_dict_list]]
    _PrefixPopulationEntriesLike = Annotated[
        tuple,
        Is[_is_prefix_population_entries],
    ]
    _PrefixPopulationMappingLike = Annotated[
        Mapping,
        Is[_is_prefix_population_mapping],
    ]
    _PrefixHashLike = Annotated[str, Is[_is_prefix_hash]]
    _StringMappingLike = Annotated[Mapping, Is[_is_string_mapping]]
    _StringRecordLike = Annotated[dict, Is[_is_string_mapping]]


@dataclass(frozen=True, init=False)
class CanonicalPrefixPopulation:
    """Immutable, validated prefix/hash input for clustered cache construction."""

    entries: tuple[tuple[str, str], ...]

    @beartype
    def __init__(self, entries: _PrefixPopulationEntriesLike) -> None:
        object.__setattr__(self, "entries", entries)
        self.__post_init__()

    def __post_init__(self) -> None:
        previous_prefix: str | None = None
        seen_hashes: dict[str, str] = {}
        for entry in self.entries:
            prefix_text, prefix_hash = entry
            if previous_prefix is not None and prefix_text <= previous_prefix:
                _raise("canonical prefix population prefixes must be strictly increasing")
            if make_prefix_hash(prefix_text) != prefix_hash:
                _raise(f"prefix hash {prefix_hash!r} does not match its prefix text")
            prior = seen_hashes.get(prefix_hash)
            if prior is not None and prior != prefix_text:
                _raise(f"distinct prefix texts collide at prefix hash {prefix_hash}")
            previous_prefix = prefix_text
            seen_hashes[prefix_hash] = prefix_text

    def as_dict(self) -> dict[str, str]:
        """Return a fresh mutable view for the cache builder's local ordering."""

        return dict(self.entries)


@beartype
def canonicalize_prefix_population(
    prefixes: _PrefixPopulationMappingLike,
) -> CanonicalPrefixPopulation:
    """Copy and validate an externally supplied prefix/hash mapping."""

    entries = list(prefixes.items())
    return CanonicalPrefixPopulation(entries=tuple(sorted(entries)))


@dataclass(frozen=True, init=False)
class RawClusteredFeatureSegment:
    """One raw extraction segment, before mask/cluster/selection."""

    feature_ids: t.Tensor
    feature_magnitudes: t.Tensor
    char_start: NonNegativeInt
    char_end: NonNegativeInt

    @beartype
    def __init__(
        self,
        feature_ids: t.Tensor,
        feature_magnitudes: t.Tensor,
        char_start: NonNegativeIntLike,
        char_end: NonNegativeIntLike,
    ) -> None:
        object.__setattr__(self, "feature_ids", feature_ids)
        object.__setattr__(self, "feature_magnitudes", feature_magnitudes)
        object.__setattr__(self, "char_start", NonNegativeInt.parse(char_start))
        object.__setattr__(self, "char_end", NonNegativeInt.parse(char_end))
        if self.char_end < self.char_start:
            _raise(
                "raw clustered segment char_end must be greater than or equal "
                f"to char_start: {self.char_start}:{self.char_end}"
            )


if TYPE_CHECKING:
    _RawClusteredFeatureSegmentTupleLike = tuple[RawClusteredFeatureSegment, ...]
else:
    _RawClusteredFeatureSegmentTupleLike = Annotated[
        tuple,
        Is[
            lambda value: (
                isinstance(value, tuple)
                and all(isinstance(item, RawClusteredFeatureSegment) for item in value)
            )
        ],
    ]


@dataclass(frozen=True, init=False)
class RawClusteredPrefix:
    """Raw extraction evidence for one exact prefix."""

    segments: tuple[RawClusteredFeatureSegment, ...]
    n_tokens: PositiveInt

    @beartype
    def __init__(
        self,
        segments: _RawClusteredFeatureSegmentTupleLike,
        n_tokens: PositiveIntLike,
    ) -> None:
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "n_tokens", PositiveInt.parse(n_tokens))


@dataclass(frozen=True)
class ClusteredReplayState:
    """Private CPU selection state reconstructed from the feature contract."""

    pipeline: FeaturePipelineConfig
    percentile_table: t.Tensor


@dataclass(frozen=True, init=False)
class _MaterializedSegment:
    context: ContextFeatureSegment
    positive_percentiles_uint8: t.Tensor
    n_positive: NonNegativeInt
    n_negative: NonNegativeInt

    @beartype
    def __init__(
        self,
        context: ContextFeatureSegment,
        positive_percentiles_uint8: t.Tensor,
        n_positive: NonNegativeIntLike,
        n_negative: NonNegativeIntLike,
    ) -> None:
        object.__setattr__(self, "context", context)
        object.__setattr__(
            self,
            "positive_percentiles_uint8",
            positive_percentiles_uint8,
        )
        object.__setattr__(self, "n_positive", NonNegativeInt.parse(n_positive))
        object.__setattr__(self, "n_negative", NonNegativeInt.parse(n_negative))


if TYPE_CHECKING:
    _MaterializedSegmentSequenceLike = Sequence[_MaterializedSegment]
else:
    _MaterializedSegmentSequenceLike = Annotated[
        Sequence,
        Is[
            lambda value: (
                isinstance(value, Sequence)
                and all(isinstance(item, _MaterializedSegment) for item in value)
            )
        ],
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _deterministic_safetensors_bytes(
    tensors: dict[str, t.Tensor],
    *,
    metadata: dict[str, str],
) -> bytes:
    """Serialize safetensors with a canonical header order.

    The Rust serializer preserves tensor layout but emits ``__metadata__``
    through a hash map, whose JSON key order varies between processes.  That
    makes otherwise identical DDP owner files differ byte-for-byte.  Data
    offsets are relative to the data section, so canonicalizing and repadding
    only the JSON header is format-preserving and makes the complete file
    deterministic.
    """

    try:
        return deterministic_safetensors_bytes(tensors, metadata=metadata)
    except ValueError as exc:
        _raise(str(exc))


def _raise(message: str) -> Never:
    raise AxBenchBpbAdvError(f"clustered context cache: {message}")


def _install_once(payload: bytes, path: Path, *, surface: str) -> None:
    try:
        install_bytes_once(payload, path, surface=surface)
    except ImmutableArtifactError as exc:
        _raise(str(exc))


@beartype
def _validate_ddp(
    *,
    rank: NonNegativeIntLike,
    world_size: PositiveIntLike,
) -> None:
    """Validate the pre-collective DDP coordinates against the live group.

    Only process-group INITIALIZATION divergence is irreducible: with no group
    there is no channel over which to agree. Bad coordinates *inside* an
    initialized group are not irreducible, because ``dist.get_rank()`` and
    ``dist.get_world_size()`` are authoritative and need no trust in the caller.

    G3c-D2: this previously returned early on ``world_size == 1`` BEFORE asking
    whether a group was initialized, so a rank inside a real multi-rank group
    could claim ``world_size=1``, be accepted as a single-rank builder, skip
    every collective, commit the complete cache and return SUCCESS while its
    peers blocked in the first gather. One rank reporting success while the
    distributed run is broken is the worst available outcome, so the
    authoritative coordinates are now consulted whenever a group exists.
    """

    if rank >= world_size:
        _raise(f"invalid DDP coordinates rank={rank}, world_size={world_size}")
    if dist.is_available() and dist.is_initialized():
        actual_world_size = dist.get_world_size()
        actual_rank = dist.get_rank()
        if (actual_rank, actual_world_size) != (rank, world_size):
            _raise(
                "DDP arguments disagree with the initialized process group: "
                f"arguments=({rank}, {world_size}), "
                f"actual=({actual_rank}, {actual_world_size})"
            )
        return
    if world_size > 1:
        _raise("world_size > 1 requires an initialized torch.distributed process group")


def _status_error_text(exc: BaseException) -> str:
    """Render an exception for a gathered status payload. MUST NEVER RAISE.

    G3c-D3: every gathered phase converts its exception to a status string and
    then joins an unconditional collective. That conversion was
    ``f"{type(exc).__name__}: {exc}"``, and ``__str__`` is arbitrary user code
    that may itself raise -- a corrupted CUDA context is the realistic case. An
    exception escaping the handler skips the collective entirely on that rank,
    so its peers block in a gather it never joins and surface an opaque
    transport error instead of the synchronized failure the design promises.

    The formatting is therefore guarded, and each fallback degrades to
    something that cannot re-enter the offending object: first the type name
    alone, then a constant. ``BaseException`` is deliberate -- a
    ``KeyboardInterrupt`` raised inside ``__str__`` must not escape a handler
    whose only job is to let every rank fail together.
    """

    try:
        return f"{type(exc).__name__}: {exc}"
    except BaseException:
        try:
            return f"{type(exc).__name__}: <exception rendering failed>"
        except BaseException:
            return "<exception rendering failed>"


@beartype
def _all_gather_object(
    value: _AnyRecordLike,
    *,
    world_size: PositiveIntLike,
) -> _AnyRecordListLike:
    if world_size == 1:
        return [value]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, value)
    if any(item is None for item in gathered):
        _raise("distributed status gather returned a missing rank payload")
    return cast(list[dict[str, Any]], gathered)


@beartype
def _broadcast_rank_zero_status(
    local_status: _AnyRecordLike | None,
    *,
    rank: NonNegativeIntLike,
    world_size: PositiveIntLike,
) -> _AnyRecordLike:
    if world_size == 1:
        if local_status is None:
            _raise("rank-zero status was unexpectedly absent")
        return local_status
    payload = [local_status if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    status = payload[0]
    if not isinstance(status, dict):
        _raise("rank-zero status broadcast returned a malformed payload")
    return status


def _raise_gathered_errors(
    statuses: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    failures = [
        {"rank": status.get("rank"), "error": status.get("error")}
        for status in statuses
        if status.get("error") is not None
    ]
    if failures:
        _raise(f"{phase} failed on one or more ranks: {failures}")


def _population_entries(unique_prefixes: Mapping[str, str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    for prefix_text, prefix_hash in unique_prefixes.items():
        if make_prefix_hash(prefix_text) != prefix_hash:
            _raise(f"prefix hash {prefix_hash!r} does not match its prefix text")
        prior = seen_hashes.get(prefix_hash)
        if prior is not None and prior != prefix_text:
            _raise(f"distinct prefix texts collide at prefix hash {prefix_hash}")
        seen_hashes[prefix_hash] = prefix_text
        entries.append(
            {
                "prefix_hash": prefix_hash,
                "prefix_text_sha256": _sha256_text(prefix_text),
            }
        )
    return sorted(entries, key=lambda entry: entry["prefix_hash"])


@beartype
def _expected_extraction_identity(
    *,
    contract: _ClusterFeatureContractLike,
    segment_size: PositiveIntLike,
    extraction_max_tokens: PositiveIntLike,
) -> _AnyRecordLike:
    algorithms = contract.semantic_dict().get("algorithms")
    packed_algorithm = (
        algorithms.get("packed_percentile_selection") if isinstance(algorithms, Mapping) else None
    )
    if not isinstance(packed_algorithm, str) or not packed_algorithm:
        _raise("feature contract is missing the packed selection algorithm version")
    return {
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "checkpoint_model_sha256": contract.checkpoint.model_file.sha256,
        "statistics_file_sha256": contract.statistics.file_sha256,
        "percentile_table_checksum": contract.statistics.percentile_table_checksum,
        "selection_hash": contract.selection.selection_hash,
        "selection_manifest_sha256": contract.selection.selection_manifest_sha256,
        "selection_shards_sha256": contract.selection.selection_shards_sha256,
        "packed_row_fingerprint": contract.selection.packed_row_fingerprint,
        "preselection_seed": contract.selection.preselection_seed,
        "positive_selection": json.loads(contract.selection.positive_selection_json),
        "negative_selection": json.loads(contract.selection.negative_selection_json),
        "num_buckets": contract.selection.num_buckets,
        "effective_positive_buckets": contract.selection.effective_positive_buckets,
        "packed_selection_algorithm": packed_algorithm,
        "char_alignment_algorithm": CLUSTERED_CONTEXT_CHAR_ALIGNMENT_VERSION,
        "segment_size": segment_size,
        "extraction_max_tokens": extraction_max_tokens,
        "sae_layer": contract.sae.layer,
        "sae_hook": contract.sae.hook_name,
        "pooling": contract.feature_directory.pooling,
        "context_independent_features": (contract.feature_directory.context_independent_features),
        "ignore_bos_features": contract.feature_directory.ignore_bos_features,
    }


def _validate_extraction_identity_against_contract(
    identity: Mapping[str, Any],
    *,
    contract: ClusterFeatureContract,
) -> None:
    expected_keys = {
        "feature_contract_fingerprint",
        "checkpoint_model_sha256",
        "statistics_file_sha256",
        "percentile_table_checksum",
        "selection_hash",
        "selection_manifest_sha256",
        "selection_shards_sha256",
        "packed_row_fingerprint",
        "preselection_seed",
        "positive_selection",
        "negative_selection",
        "num_buckets",
        "effective_positive_buckets",
        "packed_selection_algorithm",
        "char_alignment_algorithm",
        "segment_size",
        "extraction_max_tokens",
        "sae_layer",
        "sae_hook",
        "pooling",
        "context_independent_features",
        "ignore_bos_features",
    }
    if set(identity) != expected_keys:
        _raise(
            "clustered extraction identity fields mismatch: "
            f"missing={sorted(expected_keys - set(identity))}, "
            f"extra={sorted(set(identity) - expected_keys)}"
        )
    algorithms = contract.semantic_dict().get("algorithms")
    packed_algorithm = (
        algorithms.get("packed_percentile_selection") if isinstance(algorithms, Mapping) else None
    )
    expected = {
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "checkpoint_model_sha256": contract.checkpoint.model_file.sha256,
        "statistics_file_sha256": contract.statistics.file_sha256,
        "percentile_table_checksum": contract.statistics.percentile_table_checksum,
        "selection_hash": contract.selection.selection_hash,
        "selection_manifest_sha256": contract.selection.selection_manifest_sha256,
        "selection_shards_sha256": contract.selection.selection_shards_sha256,
        "packed_row_fingerprint": contract.selection.packed_row_fingerprint,
        "preselection_seed": contract.selection.preselection_seed,
        "positive_selection": json.loads(contract.selection.positive_selection_json),
        "negative_selection": json.loads(contract.selection.negative_selection_json),
        "num_buckets": contract.selection.num_buckets,
        "effective_positive_buckets": contract.selection.effective_positive_buckets,
        "packed_selection_algorithm": packed_algorithm,
        "char_alignment_algorithm": CLUSTERED_CONTEXT_CHAR_ALIGNMENT_VERSION,
        "segment_size": contract.feature_directory.segment_size,
        "sae_layer": contract.sae.layer,
        "sae_hook": contract.sae.hook_name,
        "pooling": contract.feature_directory.pooling,
        "context_independent_features": (contract.feature_directory.context_independent_features),
        "ignore_bos_features": contract.feature_directory.ignore_bos_features,
    }
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if mismatches:
        _raise(f"clustered extraction identity disagrees with contract: {mismatches}")
    extraction_max_tokens = identity.get("extraction_max_tokens")
    if type(extraction_max_tokens) is not int or extraction_max_tokens <= 0:
        _raise("clustered extraction identity has invalid extraction_max_tokens")


def _validate_runtime_configuration(
    *,
    cfg: Any,
    contract: ClusterFeatureContract,
) -> tuple[int, int]:
    try:
        artifact_segment_mode = SegmentMode(contract.feature_directory.segment_mode)
    except ValueError as exc:
        raise AxBenchBpbAdvError(
            "clustered feature contract has unknown segment mode "
            f"{contract.feature_directory.segment_mode!r}"
        ) from exc
    require_fixed_segment_mode(
        artifact_segment_mode,
        surface="AxBench-BPB-Adv clustered context-cache extraction",
        source="clustered feature contract",
    )
    segment_size = cfg.context_cache.segment_size
    extraction_max_tokens = cfg.extraction_max_tokens
    num_buckets = cfg.context_cache.num_buckets
    extraction_threshold = cfg.extraction_threshold
    if not isinstance(segment_size, PositiveInt):
        _raise("context_cache.segment_size must be a positive integer")
    if not isinstance(extraction_max_tokens, PositiveInt):
        _raise("extraction_max_tokens must be positive")
    if not isinstance(num_buckets, PositiveInt):
        _raise("context_cache.num_buckets must be a positive integer")
    if (
        not isinstance(extraction_threshold, (int, float))
        or isinstance(extraction_threshold, bool)
        or extraction_threshold != 0.0
    ):
        _raise(
            "clustered raw extraction does not support magnitude thresholding; "
            "set extraction_threshold=0.0"
        )
    if segment_size != contract.feature_directory.segment_size:
        _raise(
            "context segment size does not match the validated feature directory: "
            f"cfg={segment_size}, contract={contract.feature_directory.segment_size}"
        )
    if num_buckets != contract.selection.num_buckets:
        _raise(
            "context bucket count does not match checkpoint selection: "
            f"cfg={cfg.context_cache.num_buckets}, "
            f"contract={contract.selection.num_buckets}"
        )
    if contract.selection.num_buckets != 2:
        _raise("clustered context conversion requires checkpoint num_buckets == 2")
    if contract.n_sae_features > 65_536:
        _raise("original SAE width exceeds the safe uint16 training-selection limit")
    return segment_size, extraction_max_tokens


@beartype
def extract_raw_clustered_prefix(
    *,
    prefix_text: str,
    base_model: Any,
    base_sae: Any,
    contract: _ClusterFeatureContractLike,
    segment_size: PositiveIntLike,
    extraction_max_tokens: PositiveIntLike,
) -> RawClusteredPrefix:
    """Extract raw sparse SAE evidence without applying selection or RNG.

    Character bounds emitted by :func:`extract_document_features` live in the
    extraction tokenizer's decoded, BOS-inclusive coordinate space.  Preserve
    that space verbatim (apart from the standard bounded end-marker repair):
    the fixed-context projector consumes target tokens that also include BOS.
    Comparing these bounds to the BOS-free ``prefix_text`` length would clip
    only the final segment while leaving every earlier boundary BOS-inclusive,
    creating a mixed coordinate system that fails at the final target token.
    """

    sae_cfg = getattr(base_sae, "cfg", None)
    loaded_width = getattr(sae_cfg, "d_sae", None)
    if type(loaded_width) is not int or loaded_width != contract.n_sae_features:
        _raise(
            "runtime extraction SAE width does not match the feature contract: "
            f"loaded={loaded_width!r}, contract={contract.n_sae_features}"
        )
    loaded_hook = getattr(sae_cfg, "hook_name", None)
    if not loaded_hook:
        sae_metadata = getattr(sae_cfg, "metadata", None)
        loaded_hook = (
            sae_metadata.get("hook_name")
            if isinstance(sae_metadata, Mapping)
            else getattr(sae_metadata, "hook_name", None)
        )
    if not isinstance(loaded_hook, str) or not loaded_hook or loaded_hook != contract.sae.hook_name:
        _raise(
            "runtime extraction SAE hook does not match the feature contract: "
            f"loaded={loaded_hook!r}, contract={contract.sae.hook_name!r}"
        )
    decoder = getattr(base_sae, "W_dec", None)
    if not isinstance(decoder, t.Tensor) or decoder.ndim != 2:
        _raise("runtime extraction SAE is missing a rank-2 W_dec tensor")
    if int(decoder.shape[0]) != contract.n_sae_features:
        _raise(
            "runtime extraction SAE decoder feature axis does not match the "
            f"original feature width: shape={tuple(decoder.shape)}, "
            f"contract={contract.n_sae_features}"
        )
    try:
        pooling = PoolingMode(contract.feature_directory.pooling)
    except ValueError:
        _raise(f"unsupported contract pooling mode {contract.feature_directory.pooling!r}")
    (
        all_feature_ids,
        all_feature_magnitudes,
        _positions,
        char_starts,
        char_ends,
        n_tokens,
    ) = extract_document_features(
        model=base_model,
        sae=base_sae,
        text=prefix_text,
        layer=contract.sae.layer,
        context_window=extraction_max_tokens,
        pooling=pooling,
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=segment_size,
        hook_name=contract.sae.hook_name,
        context_independent_features=(contract.feature_directory.context_independent_features),
        ignore_bos_features=contract.feature_directory.ignore_bos_features,
    )
    tokenizer = getattr(base_model, "tokenizer", None)
    model_cfg = getattr(base_model, "cfg", None)
    if tokenizer is None or model_cfg is None or not hasattr(model_cfg, "n_ctx"):
        _raise(
            "runtime extraction model is missing tokenizer/cfg.n_ctx required "
            "to validate BOS-inclusive character coordinates"
        )
    try:
        kept_ids, independently_observed_n_tokens = truncate_to_training_window(
            prefix_text,
            tokenizer,
            extraction_max_tokens,
            model_cfg.n_ctx,
        )
        source_offsets = token_to_char_offsets(tokenizer, kept_ids)
    except Exception as exc:
        _raise(
            "could not independently reconstruct extraction-token character "
            f"coordinates for the exact prefix: {exc}"
        )
    if independently_observed_n_tokens != n_tokens:
        _raise(
            "raw extraction token count disagrees with independent tokenizer "
            f"replay: extraction={n_tokens}, replay={independently_observed_n_tokens}"
        )
    if not kept_ids or not source_offsets or source_offsets[-1] <= 0:
        _raise("raw extraction produced no positive-width decoded source context")
    source_decoded_len = int(source_offsets[-1])
    try:
        char_starts, char_ends, kept_indices = align_char_ranges_to_decoded_prefix(
            char_starts=char_starts,
            char_ends=char_ends,
            decoded_len=source_decoded_len,
            counter_name=_CLUSTERED_CONTEXT_ALIGNMENT_COUNTER,
        )
    except (BeartypeCallHintParamViolation, ValueError) as exc:
        _raise(
            "could not reconcile extraction-token character bounds with the "
            f"exact prefix text: {exc}"
        )
    all_feature_ids = [all_feature_ids[index] for index in kept_indices]
    all_feature_magnitudes = [all_feature_magnitudes[index] for index in kept_indices]
    raw_segments: list[RawClusteredFeatureSegment] = []
    previous_end = 0
    for index, (raw_ids, raw_magnitudes, char_start, char_end) in enumerate(
        zip(
            all_feature_ids,
            all_feature_magnitudes,
            char_starts,
            char_ends,
            strict=True,
        )
    ):
        # G3a-D2: reject a non-integral source dtype BEFORE canonicalizing.
        # The cast below is what makes _validate_raw_segment's dtype guard
        # always see int64, so a float extractor result (an upstream regression
        # or a mis-built double) was silently TRUNCATED into immutable evidence
        # -- float32 [1.9] persisted as int64 [1] with no error. Real upstream
        # emits int32 today, so the intended int32 -> int64 widening stays
        # lossless; this only refuses inputs that were never integral.
        if raw_ids.is_floating_point() or raw_ids.dtype is t.bool:
            _raise(
                f"extracted segment {index} feature IDs must be an integer "
                f"tensor, got dtype {raw_ids.dtype}"
            )
        ids = raw_ids.detach().to(device="cpu", dtype=t.int64).contiguous()
        magnitudes = raw_magnitudes.detach().to(device="cpu", dtype=t.float32).contiguous()
        _validate_raw_segment(
            ids,
            magnitudes,
            n_sae_features=contract.n_sae_features,
            surface=f"extracted segment {index}",
        )
        if not char_start <= char_end <= source_decoded_len:
            _raise(
                f"extracted segment {index} has invalid character bounds "
                f"[{char_start}, {char_end}) for BOS-inclusive decoded source "
                f"length {source_decoded_len}"
            )
        # G3a-D1: the WRITER must be at least as strict as the READER.
        # _load_raw_prefix enforces `char_start < previous_end` but extraction
        # validated each interval in isolation, so overlapping bounds such as
        # [(0, 10), (5, 11)] were accepted here, serialized, and installed as
        # IMMUTABLE evidence -- and only then rejected by the loader. Under the
        # frozen protocol the install happens before the verifying load, so the
        # bad bytes are already committed and recovery needs a fresh run tag.
        if char_start < previous_end:
            _raise(
                f"extracted segment {index} starts at {char_start}, before the "
                f"previous segment ended at {previous_end}; character bounds "
                "must be non-overlapping and ordered"
            )
        previous_end = char_end
        raw_segments.append(
            RawClusteredFeatureSegment(
                feature_ids=ids,
                feature_magnitudes=magnitudes,
                char_start=char_start,
                char_end=char_end,
            )
        )
    try:
        return RawClusteredPrefix(segments=tuple(raw_segments), n_tokens=n_tokens)
    except BeartypeCallHintParamViolation:
        _raise(f"raw extraction returned invalid token count {n_tokens!r}")


@beartype
def _validate_raw_segment(
    feature_ids: t.Tensor,
    feature_magnitudes: t.Tensor,
    *,
    n_sae_features: PositiveIntLike,
    surface: NonEmptyStrLike,
) -> None:
    if feature_ids.device.type != "cpu" or feature_magnitudes.device.type != "cpu":
        _raise(f"{surface} must be resident on CPU")
    if feature_ids.ndim != 1 or feature_magnitudes.ndim != 1:
        _raise(f"{surface} IDs and magnitudes must be one-dimensional")
    if feature_ids.shape != feature_magnitudes.shape:
        _raise(f"{surface} IDs and magnitudes are misaligned")
    if feature_ids.dtype is not t.int64:
        _raise(f"{surface} IDs must use int64")
    if feature_magnitudes.dtype is not t.float32:
        _raise(f"{surface} magnitudes must use float32")
    if not bool(t.isfinite(feature_magnitudes).all()):
        _raise(f"{surface} contains non-finite magnitudes")
    if bool((feature_magnitudes <= 0).any()):
        _raise(f"{surface} contains non-positive sparse magnitudes")
    if feature_ids.numel() == 0:
        return
    if bool(((feature_ids < 0) | (feature_ids >= n_sae_features)).any()):
        _raise(f"{surface} contains an out-of-range feature ID")
    if int(t.unique(feature_ids).numel()) != int(feature_ids.numel()):
        _raise(f"{surface} contains duplicate feature IDs")


@beartype
def _raw_metadata(
    *,
    contract: _ClusterFeatureContractLike,
    prefix_hash: _PrefixHashLike,
    prefix_text_sha256: LowercaseSha256Like,
    extraction_identity_sha256: LowercaseSha256Like,
    n_tokens: PositiveIntLike,
) -> _StringRecordLike:
    return {
        "schema_version": str(RAW_PREFIX_SCHEMA_VERSION),
        "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_raw_prefix",
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "prefix_hash": prefix_hash,
        "prefix_text_sha256": prefix_text_sha256,
        "extraction_identity_sha256": extraction_identity_sha256,
        "n_tokens": str(n_tokens),
    }


def _serialize_raw_prefix(
    raw: RawClusteredPrefix,
    *,
    metadata: dict[str, str],
) -> bytes:
    # G3a-D3: bind the metadata token count to the prefix it describes. The
    # loader treats metadata["n_tokens"] as authoritative, so accepting it
    # independently of raw.n_tokens meant a caller mismatch (raw.n_tokens=7,
    # metadata "8") round-tripped to 8 with no error -- disproving total
    # serialize/load fidelity for a helper whose whole job is immutable
    # evidence. The oracle's own caller always passes raw.n_tokens, so this
    # rejects only genuine misuse.
    declared_tokens = metadata.get("n_tokens")
    if declared_tokens != str(raw.n_tokens):
        _raise(
            "raw prefix metadata n_tokens does not describe the prefix being "
            f"serialized: metadata={declared_tokens!r}, prefix={raw.n_tokens}"
        )
    tensors: dict[str, t.Tensor] = {
        "n_segments": t.tensor([len(raw.segments)], dtype=t.int64),
    }
    for index, segment in enumerate(raw.segments):
        tensors[f"seg_{index}_feature_ids"] = segment.feature_ids.contiguous()
        tensors[f"seg_{index}_feature_magnitudes"] = segment.feature_magnitudes.contiguous()
        tensors[f"seg_{index}_char_bounds"] = t.tensor(
            [segment.char_start, segment.char_end],
            dtype=t.int64,
        )
    return _deterministic_safetensors_bytes(tensors, metadata=metadata)


def _metadata_from_bytes(payload: bytes, *, surface: str) -> dict[str, str]:
    """Parse ``__metadata__`` from an in-memory snapshot.

    Deliberately takes bytes, never a ``Path``. Every caller here has already
    read the artifact once; reopening the pathname would issue an independent
    read, so the bytes a caller verified and the metadata it acts on could come
    from different file contents. See G3b-D3.
    """
    try:
        metadata = read_safetensors_metadata_from_bytes(payload)
    except Exception as exc:
        _raise(f"could not read safetensors metadata from {surface}: {exc}")
    if not isinstance(metadata, dict):
        _raise(f"safetensors file {surface} is missing metadata")
    return metadata


@beartype
def _load_raw_prefix_snapshot(
    path: Path,
    *,
    expected_metadata_without_n_tokens: _StringMappingLike,
    n_sae_features: PositiveIntLike,
    expected_sha256: str | None = None,
) -> tuple[RawClusteredPrefix, str]:
    if not path.is_file():
        _raise(f"raw prefix artifact is missing: {path}")
    # G3b-D3: one snapshot for metadata AND tensors. Reading the pathname twice
    # let a replacement between the reads pair THIS metadata -- including the
    # n_tokens returned below -- with a DIFFERENT file's segments, certifying a
    # header that does not describe the tensors actually loaded.
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _raise(f"could not read raw prefix bytes from {path}: {exc}")
    actual_sha256 = _sha256_bytes(payload)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        _raise(f"raw prefix content digest mismatch for {path}")
    metadata = _metadata_from_bytes(payload, surface=str(path))
    expected_keys = set(expected_metadata_without_n_tokens) | {"n_tokens"}
    if set(metadata) != expected_keys:
        _raise(
            f"raw prefix metadata keys mismatch at {path}: "
            f"missing={sorted(expected_keys - set(metadata))}, "
            f"extra={sorted(set(metadata) - expected_keys)}"
        )
    for key, expected in expected_metadata_without_n_tokens.items():
        if metadata.get(key) != expected:
            _raise(
                f"raw prefix metadata mismatch at {path} for {key}: "
                f"expected={expected!r}, actual={metadata.get(key)!r}"
            )
    try:
        n_tokens = int(metadata["n_tokens"])
    except (KeyError, TypeError, ValueError):
        _raise(f"raw prefix {path} has an invalid n_tokens metadata value")
    if n_tokens <= 0:
        _raise(f"raw prefix {path} has non-positive n_tokens")
    try:
        tensors = load_safetensors_bytes(payload)
        n_segments_tensor = tensors["n_segments"]
    except Exception as exc:
        _raise(f"could not load raw prefix tensors from {path}: {exc}")
    if n_segments_tensor.shape != (1,) or n_segments_tensor.dtype is not t.int64:
        _raise(f"raw prefix {path} has malformed n_segments")
    n_segments = int(n_segments_tensor.item())
    if n_segments < 0:
        _raise(f"raw prefix {path} has negative n_segments")
    expected_tensor_keys = {"n_segments"}
    segments: list[RawClusteredFeatureSegment] = []
    previous_end = 0
    for index in range(n_segments):
        ids_key = f"seg_{index}_feature_ids"
        magnitudes_key = f"seg_{index}_feature_magnitudes"
        bounds_key = f"seg_{index}_char_bounds"
        expected_tensor_keys.update({ids_key, magnitudes_key, bounds_key})
        try:
            ids = tensors[ids_key]
            magnitudes = tensors[magnitudes_key]
            bounds = tensors[bounds_key]
        except KeyError as exc:
            _raise(f"raw prefix {path} is missing tensor {exc.args[0]!r}")
        _validate_raw_segment(
            ids,
            magnitudes,
            n_sae_features=n_sae_features,
            surface=f"raw prefix {path} segment {index}",
        )
        if bounds.shape != (2,) or bounds.dtype is not t.int64:
            _raise(f"raw prefix {path} segment {index} has malformed char bounds")
        char_start, char_end = (int(value) for value in bounds.tolist())
        if not 0 <= char_start <= char_end or char_start < previous_end:
            _raise(f"raw prefix {path} segment {index} has invalid or unordered bounds")
        previous_end = char_end
        segments.append(
            RawClusteredFeatureSegment(
                feature_ids=ids,
                feature_magnitudes=magnitudes,
                char_start=char_start,
                char_end=char_end,
            )
        )
    if set(tensors) != expected_tensor_keys:
        _raise(
            f"raw prefix {path} tensor keys mismatch: "
            f"missing={sorted(expected_tensor_keys - set(tensors))}, "
            f"extra={sorted(set(tensors) - expected_tensor_keys)}"
        )
    return (
        RawClusteredPrefix(segments=tuple(segments), n_tokens=n_tokens),
        actual_sha256,
    )


def reconstruct_clustered_replay_state(
    contract: ClusterFeatureContract,
) -> ClusteredReplayState:
    """Rebuild the checkpoint's exact CPU selection pipeline and table."""

    provenance = contract.provenance_dict()
    raw_stats_path = provenance.get("feature_statistics_path")
    if not isinstance(raw_stats_path, str) or not raw_stats_path:
        _raise("feature contract provenance is missing feature_statistics_path")
    try:
        stats_path = Path(raw_stats_path).expanduser().resolve(strict=True)
        payload = stats_path.read_bytes()
    except Exception as exc:
        _raise(f"could not revalidate feature statistics: {exc}")
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != contract.statistics.file_sha256:
        _raise(
            "feature statistics bytes changed while loading clustered selection "
            "replay: "
            f"expected={contract.statistics.file_sha256}, actual={actual_sha256}"
        )
    try:
        statistics, _stored_checksum, _raw_feature_ids = load_statistics_and_checksum_from_bytes(
            payload
        )
        percentile_table = build_percentile_table(
            statistics.per_feature_stats,
            contract.n_sae_features,
        )
    except Exception as exc:
        _raise(f"could not rebuild percentile table from validated statistics: {exc}")
    if statistics.metadata.n_sae_features != contract.n_sae_features:
        _raise("statistics width changed after clustered preflight")
    checksum = compute_percentile_checksum(percentile_table)
    if checksum != contract.statistics.percentile_table_checksum:
        _raise(
            "rebuilt percentile table checksum does not match the feature contract: "
            f"expected={contract.statistics.percentile_table_checksum}, "
            f"actual={checksum}"
        )
    try:
        positive_selection = json.loads(contract.selection.positive_selection_json)
        negative_selection = json.loads(contract.selection.negative_selection_json)
    except json.JSONDecodeError as exc:
        _raise(f"checkpoint selection JSON is malformed: {exc}")
    if not isinstance(positive_selection, dict) or not isinstance(negative_selection, dict):
        _raise("checkpoint positive and negative selections must be JSON objects")
    config_block = {
        "feature_selection": positive_selection,
        "negative_selection": negative_selection,
    }
    try:
        feature_delta = contract.selection.parsed_feature_delta()
        pipeline = build_feature_pipeline_from_cfg_block(
            config_block,
            per_feature_stats=statistics.per_feature_stats,
            n_sae_features=contract.n_sae_features,
            num_buckets=contract.selection.num_buckets,
            seed=contract.selection.preselection_seed,
            ddp_rank=0,
            feature_delta=feature_delta,
            feature_mask=contract.feature_mask,
        )
    except Exception as exc:
        _raise(f"could not reconstruct checkpoint feature pipeline: {exc}")
    if pipeline.feature_mask != contract.feature_mask:
        _raise("reconstructed feature pipeline mask does not match contract")
    if pipeline.effective_buckets != contract.selection.effective_positive_buckets:
        _raise(
            "reconstructed effective bucket count does not match contract: "
            f"pipeline={pipeline.effective_buckets}, "
            f"contract={contract.selection.effective_positive_buckets}"
        )
    _private_pipeline_generator(pipeline)
    return ClusteredReplayState(
        pipeline=pipeline,
        percentile_table=percentile_table,
    )


def _private_pipeline_generator(pipeline: FeaturePipelineConfig) -> t.Generator:
    selection = pipeline.selection_config
    if isinstance(selection, SamplingBundle):
        generator = selection.rng
        if pipeline.negative_bundle is not None and pipeline.negative_bundle.rng is not generator:
            _raise("sampling pipeline did not preserve the shared positive/negative RNG")
        return generator
    if isinstance(selection, ThresholdSelectionConfig):
        if pipeline.negative_bundle is None:
            _raise("clustered threshold pipeline is missing negative selection RNG")
        return pipeline.negative_bundle.rng
    _raise(f"unsupported replay pipeline type {type(selection).__name__}")


def _generator_state_sha256(pipeline: FeaturePipelineConfig) -> str:
    state = _private_pipeline_generator(pipeline).get_state().contiguous().cpu()
    return _sha256_bytes(state.numpy().tobytes())


def select_raw_clustered_segment(
    raw_segment: RawClusteredFeatureSegment,
    *,
    replay_state: ClusteredReplayState,
    contract: ClusterFeatureContract,
) -> ClusteredSegmentSelection:
    """Apply the exact training magnitude -> uint8 -> clustered selection chain."""

    _validate_raw_segment(
        raw_segment.feature_ids,
        raw_segment.feature_magnitudes,
        n_sae_features=contract.n_sae_features,
        surface="clustered replay raw segment",
    )
    return select_clustered_segment_from_magnitudes(
        raw_segment.feature_ids,
        raw_segment.feature_magnitudes,
        percentile_table=replay_state.percentile_table,
        clusters=contract.consolidation,
        pipeline=replay_state.pipeline,
        source="clustered replay raw segment",
    )


def _materialize_segment(
    raw_segment: RawClusteredFeatureSegment,
    *,
    replay_state: ClusteredReplayState,
    contract: ClusterFeatureContract,
) -> _MaterializedSegment:
    selected = select_raw_clustered_segment(
        raw_segment,
        replay_state=replay_state,
        contract=contract,
    )
    positive_ids = selected.positive_ids.to(t.int64)
    negative_ids = selected.negative_ids.to(t.int64)
    feature_ids = t.cat((positive_ids, negative_ids))
    negative_bucket = replay_state.pipeline.effective_buckets
    buckets = t.cat(
        (
            selected.positive_buckets.to(t.int64),
            t.full(
                (negative_ids.numel(),),
                negative_bucket,
                dtype=t.int64,
            ),
        )
    )
    contract.consolidation.assert_representatives_only(
        feature_ids,
        surface="clustered context cache selected segment",
    )
    contract.feature_mask.assert_no_masked(feature_ids)
    if int(t.unique(feature_ids).numel()) != int(feature_ids.numel()):
        _raise("selected context segment contains duplicate feature IDs")
    if bool((buckets < 0).any()) or bool((buckets >= contract.selection.num_buckets).any()):
        _raise("selected context segment contains an out-of-range bucket")
    if selected.positive_buckets.numel() and bool(
        (selected.positive_buckets >= contract.selection.effective_positive_buckets).any()
    ):
        _raise("positive context feature was assigned the reserved negative bucket")
    if negative_ids.numel() and not bool(
        (buckets[positive_ids.numel() :] == negative_bucket).all()
    ):
        _raise("negative context feature was not assigned the reserved final bucket")
    context = ContextFeatureSegment(
        feature_ids=tuple(int(value) for value in feature_ids.tolist()),
        buckets=tuple(int(value) for value in buckets.tolist()),
        char_start=raw_segment.char_start,
        char_end=raw_segment.char_end,
    )
    return _MaterializedSegment(
        context=context,
        positive_percentiles_uint8=selected.positive_percentiles_uint8,
        n_positive=int(positive_ids.numel()),
        n_negative=int(negative_ids.numel()),
    )


_SELECTED_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "feature_contract_fingerprint",
        "prefix_hash",
        "prefix_text_sha256",
        "extraction_identity_sha256",
        "raw_sha256",
        "generator_state_after_sha256",
    }
)


def _validate_selected_metadata(
    metadata: Mapping[str, str],
    *,
    contract: ClusterFeatureContract,
) -> None:
    if set(metadata) != _SELECTED_METADATA_KEYS:
        _raise(
            "selected prefix metadata keys mismatch: "
            f"missing={sorted(_SELECTED_METADATA_KEYS - set(metadata))}, "
            f"extra={sorted(set(metadata) - _SELECTED_METADATA_KEYS)}"
        )
    expected_values = {
        "schema_version": str(SELECTED_PREFIX_SCHEMA_VERSION),
        "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_selected_prefix",
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            _raise(
                f"selected prefix metadata field {key} mismatch: "
                f"expected={expected!r}, actual={metadata.get(key)!r}"
            )
    prefix_hash = metadata.get("prefix_hash")
    if (
        not isinstance(prefix_hash, str)
        or len(prefix_hash) != 16
        or any(char not in "0123456789abcdef" for char in prefix_hash)
    ):
        _raise("selected prefix metadata has an invalid prefix_hash")
    for key in (
        "prefix_text_sha256",
        "extraction_identity_sha256",
        "raw_sha256",
        "generator_state_after_sha256",
    ):
        if not is_sha256_hex(metadata.get(key)):
            _raise(f"selected prefix metadata has an invalid {key}")


@beartype
def _validate_materialized_segments(
    segments: _MaterializedSegmentSequenceLike,
    *,
    contract: ClusterFeatureContract,
) -> None:
    previous_end = 0
    for index, segment in enumerate(segments):
        context = segment.context
        if context.char_end < context.char_start or context.char_start < previous_end:
            _raise(f"selected segment {index} has invalid or unordered char bounds")
        previous_end = context.char_end
        if len(context.feature_ids) != len(context.buckets):
            _raise(f"selected segment {index} IDs and buckets are misaligned")
        ids = t.tensor(context.feature_ids, dtype=t.int64)
        buckets = t.tensor(context.buckets, dtype=t.int64)
        contract.consolidation.assert_representatives_only(
            ids,
            surface=f"selected segment {index}",
        )
        contract.feature_mask.assert_no_masked(ids)
        if int(t.unique(ids).numel()) != int(ids.numel()):
            _raise(f"selected segment {index} contains duplicate feature IDs")
        if buckets.numel() and bool(
            ((buckets < 0) | (buckets >= contract.selection.num_buckets)).any()
        ):
            _raise(f"selected segment {index} has an out-of-range bucket")
        if segment.n_positive > len(context.feature_ids):
            _raise(f"selected segment {index} has an invalid positive count")
        if segment.n_negative != len(context.feature_ids) - segment.n_positive:
            _raise(f"selected segment {index} has an invalid negative count")
        percentiles = segment.positive_percentiles_uint8
        if (
            percentiles.device.type != "cpu"
            or percentiles.ndim != 1
            or percentiles.dtype is not t.uint8
            or percentiles.shape != (segment.n_positive,)
        ):
            _raise(f"selected segment {index} has malformed positive percentiles")
        if segment.n_positive and bool(
            (buckets[: segment.n_positive] >= contract.selection.effective_positive_buckets).any()
        ):
            _raise(f"selected segment {index} has an invalid positive bucket")
        if segment.n_negative and not bool(
            (buckets[segment.n_positive :] == contract.selection.effective_positive_buckets).all()
        ):
            _raise(f"selected segment {index} has an invalid negative bucket")


@beartype
def _serialize_selected_prefix(
    segments: _MaterializedSegmentSequenceLike,
    *,
    metadata: _StringRecordLike,
    contract: ClusterFeatureContract,
) -> bytes:
    _validate_selected_metadata(metadata, contract=contract)
    _validate_materialized_segments(segments, contract=contract)
    tensors: dict[str, t.Tensor] = {
        "n_segments": t.tensor([len(segments)], dtype=t.int64),
    }
    for index, segment in enumerate(segments):
        tensors[f"seg_{index}_feature_ids"] = t.tensor(
            segment.context.feature_ids,
            dtype=t.int64,
        )
        tensors[f"seg_{index}_buckets"] = t.tensor(
            segment.context.buckets,
            dtype=t.int64,
        )
        tensors[f"seg_{index}_char_bounds"] = t.tensor(
            [segment.context.char_start, segment.context.char_end],
            dtype=t.int64,
        )
        tensors[f"seg_{index}_positive_percentiles_uint8"] = (
            segment.positive_percentiles_uint8.contiguous()
        )
        tensors[f"seg_{index}_n_positive"] = t.tensor(
            [segment.n_positive],
            dtype=t.int64,
        )
    return _deterministic_safetensors_bytes(tensors, metadata=metadata)


@beartype
def _selected_metadata(
    *,
    contract: ClusterFeatureContract,
    prefix_hash: _PrefixHashLike,
    prefix_text_sha256: LowercaseSha256Like,
    extraction_identity_sha256: LowercaseSha256Like,
    raw_sha256: LowercaseSha256Like,
    generator_state_after_sha256: LowercaseSha256Like,
) -> _StringRecordLike:
    return {
        "schema_version": str(SELECTED_PREFIX_SCHEMA_VERSION),
        "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_selected_prefix",
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "prefix_hash": prefix_hash,
        "prefix_text_sha256": prefix_text_sha256,
        "extraction_identity_sha256": extraction_identity_sha256,
        "raw_sha256": raw_sha256,
        "generator_state_after_sha256": generator_state_after_sha256,
    }


def _validate_selected_file(
    path: Path,
    *,
    expected_bytes: bytes | None,
    expected_metadata: Mapping[str, str],
    contract: ClusterFeatureContract,
) -> dict[str, int | str]:
    if not path.is_file():
        _raise(f"selected prefix artifact is missing: {path}")
    # G3b-D3: hash ONE snapshot and parse THAT SAME snapshot. This function
    # previously read the pathname three times -- bytes for the digest, then
    # safe_open for metadata, then load_file for tensors -- with no invariant
    # binding them. A replacement between reads made it return the FIRST read's
    # sha256 alongside the THIRD read's structural counts, and G3c records both
    # in the immutable manifest, so the mismatch is only detected after the run
    # tag is committed. Same defect G3b-D2 fixes for statistics, in the same file.
    actual_bytes = path.read_bytes()
    if expected_bytes is not None and actual_bytes != expected_bytes:
        _raise(f"resume selected-prefix byte mismatch at {path}; use a fresh run tag")
    metadata = _metadata_from_bytes(actual_bytes, surface=str(path))
    if metadata != dict(expected_metadata):
        _raise(f"selected prefix metadata mismatch at {path}; use a fresh run tag")
    _validate_selected_metadata(metadata, contract=contract)
    try:
        tensors = load_safetensors_bytes(actual_bytes)
        n_segments_tensor = tensors["n_segments"]
    except Exception as exc:
        _raise(f"could not load selected prefix tensors from {path}: {exc}")
    if n_segments_tensor.shape != (1,) or n_segments_tensor.dtype is not t.int64:
        _raise(f"selected prefix {path} has malformed n_segments")
    n_segments = int(n_segments_tensor.item())
    if n_segments < 0:
        _raise(f"selected prefix {path} has negative n_segments")
    expected_keys = {"n_segments"}
    n_features_total = 0
    n_positive_total = 0
    n_negative_total = 0
    previous_end = 0
    for index in range(n_segments):
        ids_key = f"seg_{index}_feature_ids"
        buckets_key = f"seg_{index}_buckets"
        bounds_key = f"seg_{index}_char_bounds"
        pcts_key = f"seg_{index}_positive_percentiles_uint8"
        n_positive_key = f"seg_{index}_n_positive"
        expected_keys.update({ids_key, buckets_key, bounds_key, pcts_key, n_positive_key})
        try:
            ids = tensors[ids_key]
            buckets = tensors[buckets_key]
            bounds = tensors[bounds_key]
            pcts = tensors[pcts_key]
            n_positive_tensor = tensors[n_positive_key]
        except KeyError as exc:
            _raise(f"selected prefix {path} is missing tensor {exc.args[0]!r}")
        if ids.ndim != 1 or ids.dtype is not t.int64:
            _raise(f"selected prefix {path} segment {index} has malformed IDs")
        if buckets.shape != ids.shape or buckets.dtype is not t.int64:
            _raise(f"selected prefix {path} segment {index} has malformed buckets")
        if n_positive_tensor.shape != (1,) or n_positive_tensor.dtype is not t.int64:
            _raise(f"selected prefix {path} segment {index} has malformed positive count")
        n_positive = int(n_positive_tensor.item())
        if not 0 <= n_positive <= int(ids.numel()):
            _raise(f"selected prefix {path} segment {index} has invalid positive count")
        if pcts.shape != (n_positive,) or pcts.dtype is not t.uint8:
            _raise(f"selected prefix {path} segment {index} has malformed percentiles")
        if bounds.shape != (2,) or bounds.dtype is not t.int64:
            _raise(f"selected prefix {path} segment {index} has malformed char bounds")
        char_start, char_end = (int(value) for value in bounds.tolist())
        if not 0 <= char_start <= char_end or char_start < previous_end:
            _raise(f"selected prefix {path} segment {index} has invalid char bounds")
        previous_end = char_end
        contract.consolidation.assert_representatives_only(
            ids,
            surface=f"selected prefix {path} segment {index}",
        )
        contract.feature_mask.assert_no_masked(ids)
        if int(t.unique(ids).numel()) != int(ids.numel()):
            _raise(f"selected prefix {path} segment {index} contains duplicate IDs")
        if buckets.numel() and bool(
            ((buckets < 0) | (buckets >= contract.selection.num_buckets)).any()
        ):
            _raise(f"selected prefix {path} segment {index} has invalid buckets")
        if n_positive and bool(
            (buckets[:n_positive] >= contract.selection.effective_positive_buckets).any()
        ):
            _raise(f"selected prefix {path} segment {index} positive bucket is invalid")
        if ids.numel() > n_positive and not bool(
            (buckets[n_positive:] == contract.selection.effective_positive_buckets).all()
        ):
            _raise(f"selected prefix {path} segment {index} negative bucket is invalid")
        n_features_total += int(ids.numel())
        n_positive_total += n_positive
        n_negative_total += int(ids.numel()) - n_positive
    if set(tensors) != expected_keys:
        _raise(
            f"selected prefix {path} tensor keys mismatch: "
            f"missing={sorted(expected_keys - set(tensors))}, "
            f"extra={sorted(set(tensors) - expected_keys)}"
        )
    return {
        "selected_sha256": _sha256_bytes(actual_bytes),
        "n_segments": n_segments,
        "n_features_total": n_features_total,
        "n_positive_total": n_positive_total,
        "n_negative_total": n_negative_total,
    }


_PARTIAL_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "status",
        "feature_contract_fingerprint",
        "extraction_identity_sha256",
        "prefix_population_sha256",
        "ddp_world_size",
    }
)


def _validate_partial_marker_document(
    document: Mapping[str, Any],
    *,
    contract: ClusterFeatureContract,
) -> None:
    if set(document) != _PARTIAL_MARKER_KEYS:
        _raise(
            "clustered partial-cache marker fields mismatch: "
            f"missing={sorted(_PARTIAL_MARKER_KEYS - set(document))}, "
            f"extra={sorted(set(document) - _PARTIAL_MARKER_KEYS)}"
        )
    expected_values = {
        "schema_version": PARTIAL_CACHE_SCHEMA_VERSION,
        "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_incomplete",
        "status": "incomplete",
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
    }
    for key, expected in expected_values.items():
        if document.get(key) != expected:
            _raise(
                f"clustered partial-cache marker field {key} mismatch: "
                f"expected={expected!r}, actual={document.get(key)!r}"
            )
    for key in ("extraction_identity_sha256", "prefix_population_sha256"):
        if not is_sha256_hex(document.get(key)):
            _raise(f"clustered partial-cache marker {key} is not a SHA-256 value")
    world_size = document.get("ddp_world_size")
    if type(world_size) is not int or world_size < 1:
        _raise("clustered partial-cache marker has an invalid DDP world size")


@beartype
def _partial_marker_document(
    *,
    contract: ClusterFeatureContract,
    extraction_identity: _AnyRecordLike,
    prefix_population_sha256: LowercaseSha256Like,
    world_size: PositiveIntLike,
) -> _AnyRecordLike:
    _validate_extraction_identity_against_contract(
        extraction_identity,
        contract=contract,
    )
    document = {
        "schema_version": PARTIAL_CACHE_SCHEMA_VERSION,
        "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_incomplete",
        "status": "incomplete",
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "extraction_identity_sha256": _sha256_json(extraction_identity),
        "prefix_population_sha256": prefix_population_sha256,
        "ddp_world_size": world_size,
    }
    _validate_partial_marker_document(document, contract=contract)
    return document


def _read_json_mapping(path: Path, *, surface: str) -> dict[str, Any]:
    """Parse one immutable JSON byte snapshot, never a reopened pathname."""

    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except Exception as exc:
        _raise(f"could not read {surface} {path}: {exc}")
    if not isinstance(value, dict):
        _raise(f"{surface} {path} must contain a JSON object")
    return cast(dict[str, Any], value)


@beartype
def _ensure_partial_marker(
    *,
    paths: AxBenchBpbAdvPaths,
    expected: _AnyRecordLike,
    rank: NonNegativeIntLike,
    world_size: PositiveIntLike,
    contract: ClusterFeatureContract,
) -> None:
    marker_path = paths.clustered_context_cache_partial_marker()
    rank_zero_status: dict[str, Any] | None = None
    if rank == 0:
        try:
            _validate_partial_marker_document(expected, contract=contract)
            if marker_path.exists():
                actual = _read_json_mapping(
                    marker_path,
                    surface="clustered partial-cache marker",
                )
                _validate_partial_marker_document(actual, contract=contract)
                if actual != expected:
                    _raise(
                        "incomplete cache marker does not match this run, "
                        "including its pinned DDP world size; use a fresh run tag"
                    )
            else:
                cache_root = paths.context_cache_dir()
                orphaned = (
                    sorted(
                        str(path.relative_to(cache_root))
                        for path in cache_root.rglob("*")
                        if path.is_file() or path.is_symlink()
                    )
                    if cache_root.exists()
                    else []
                )
                if orphaned:
                    _raise(
                        "clustered cache artifacts exist without the exact "
                        "incomplete marker; orphaned raw or selected files are "
                        "never trusted. Use a fresh run tag. Found: "
                        f"{orphaned[:10]}"
                    )
                _install_once(
                    json.dumps(expected, indent=2).encode("utf-8"),
                    marker_path,
                    surface="clustered context-cache partial marker",
                )
            rank_zero_status = {"error": None}
        except Exception as exc:
            rank_zero_status = {"error": _status_error_text(exc)}
    status = _broadcast_rank_zero_status(
        rank_zero_status,
        rank=rank,
        world_size=world_size,
    )
    if status.get("error") is not None:
        _raise(f"partial-marker setup failed: {status['error']}")

    local_error: str | None = None
    try:
        actual = _read_json_mapping(
            marker_path,
            surface="clustered partial-cache marker",
        )
        _validate_partial_marker_document(actual, contract=contract)
        if actual != expected:
            _raise("ranks observed different clustered partial-cache markers")
    except Exception as exc:
        local_error = _status_error_text(exc)
    statuses = _all_gather_object(
        {"rank": rank, "error": local_error},
        world_size=world_size,
    )
    _raise_gathered_errors(statuses, phase="partial-marker validation")


_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "feature_contract_fingerprint",
        "extraction_identity",
        "prefix_population_sha256",
        "generator_state_sha256",
        "selected_content_sha256",
        "prefixes",
    }
)
_MANIFEST_PREFIX_KEYS = frozenset(
    {
        "prefix_text_sha256",
        "raw_sha256",
        "selected_sha256",
        "n_segments",
        "n_features_total",
        "n_positive_total",
        "n_negative_total",
        "generator_state_after_sha256",
    }
)


def _valid_prefix_hash_slug(value: str) -> str | None:
    """Validate the existing 16-hex prefix-hash filename coordinate."""

    if len(value) != 16 or any(char not in "0123456789abcdef" for char in value):
        return None
    return value


def _validate_manifest_document_structure(
    manifest: Mapping[str, Any],
    *,
    contract: ClusterFeatureContract,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate every path-independent manifest fact.

    The writer calls this before immutable installation and the reader calls it
    before touching any referenced file. This keeps the writer at least as
    strict as its reader for all facts available before publication.
    """

    if set(manifest) != _MANIFEST_KEYS:
        _raise(
            "clustered manifest fields mismatch: "
            f"missing={sorted(_MANIFEST_KEYS - set(manifest))}, "
            f"extra={sorted(set(manifest) - _MANIFEST_KEYS)}"
        )
    if manifest.get("schema_version") != CLUSTERED_CONTEXT_CACHE_SCHEMA_VERSION:
        _raise(f"unsupported clustered context-cache schema {manifest.get('schema_version')!r}")
    if manifest.get("kind") != CLUSTERED_CONTEXT_CACHE_KIND:
        _raise("context-cache manifest is not a clustered BPB-Adv manifest")
    if manifest.get("feature_contract_fingerprint") != contract.feature_contract_fingerprint:
        _raise("context-cache manifest feature contract does not match the target/source")

    extraction_identity_value = manifest.get("extraction_identity")
    if not isinstance(extraction_identity_value, dict):
        _raise("context-cache manifest extraction_identity must be an object")
    extraction_identity = cast(dict[str, Any], extraction_identity_value)
    _validate_extraction_identity_against_contract(
        extraction_identity,
        contract=contract,
    )
    for hash_field in (
        "prefix_population_sha256",
        "generator_state_sha256",
        "selected_content_sha256",
    ):
        if not is_sha256_hex(manifest.get(hash_field)):
            _raise(f"clustered manifest {hash_field} is not a SHA-256 value")

    prefixes_value = manifest.get("prefixes")
    if not isinstance(prefixes_value, dict):
        _raise("clustered manifest prefixes must be an object")
    if not prefixes_value:
        _raise("clustered manifest prefixes must not be empty")

    prefixes: dict[str, dict[str, Any]] = {}
    observed_selected: list[list[str]] = []
    observed_population: list[dict[str, str]] = []
    final_prefix_generator_state: str | None = None
    for raw_prefix_hash in sorted(prefixes_value, key=str):
        if not isinstance(raw_prefix_hash, str) or _valid_prefix_hash_slug(raw_prefix_hash) is None:
            _raise(f"clustered manifest has invalid prefix hash {raw_prefix_hash!r}")
        entry_value = prefixes_value[raw_prefix_hash]
        if not isinstance(entry_value, dict):
            _raise(f"clustered manifest prefix entry {raw_prefix_hash} must be an object")
        entry = cast(dict[str, Any], entry_value)
        if set(entry) != _MANIFEST_PREFIX_KEYS:
            _raise(f"clustered manifest prefix entry {raw_prefix_hash} fields mismatch")
        for hash_field in (
            "prefix_text_sha256",
            "raw_sha256",
            "selected_sha256",
            "generator_state_after_sha256",
        ):
            if not is_sha256_hex(entry.get(hash_field)):
                _raise(f"clustered manifest {raw_prefix_hash}.{hash_field} is invalid")
        for count_field in (
            "n_segments",
            "n_features_total",
            "n_positive_total",
            "n_negative_total",
        ):
            count = entry.get(count_field)
            if type(count) is not int or count < 0:
                _raise(f"clustered manifest {raw_prefix_hash}.{count_field} is invalid")
        if entry["n_positive_total"] + entry["n_negative_total"] != entry["n_features_total"]:
            _raise(f"clustered manifest {raw_prefix_hash} feature counts are inconsistent")
        prefixes[raw_prefix_hash] = entry
        observed_selected.append([raw_prefix_hash, cast(str, entry["selected_sha256"])])
        observed_population.append(
            {
                "prefix_hash": raw_prefix_hash,
                "prefix_text_sha256": cast(str, entry["prefix_text_sha256"]),
            }
        )
        final_prefix_generator_state = cast(
            str,
            entry["generator_state_after_sha256"],
        )

    if _sha256_json(observed_selected) != manifest["selected_content_sha256"]:
        _raise("clustered manifest selected-content aggregate digest mismatch")
    if _sha256_json(observed_population) != manifest["prefix_population_sha256"]:
        _raise("clustered manifest prefix-population digest mismatch")
    if final_prefix_generator_state != manifest["generator_state_sha256"]:
        _raise("clustered manifest final generator state does not match replay entries")
    return extraction_identity, prefixes


def _manifest_document(
    *,
    contract: ClusterFeatureContract,
    extraction_identity: Mapping[str, Any],
    prefix_population_sha256: str,
    generator_state_sha256: str,
    prefix_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_entries = {
        prefix_hash: dict(prefix_entries[prefix_hash]) for prefix_hash in sorted(prefix_entries)
    }
    manifest = {
        "schema_version": CLUSTERED_CONTEXT_CACHE_SCHEMA_VERSION,
        "kind": CLUSTERED_CONTEXT_CACHE_KIND,
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "extraction_identity": dict(extraction_identity),
        "prefix_population_sha256": prefix_population_sha256,
        "generator_state_sha256": generator_state_sha256,
        "selected_content_sha256": _sha256_json(
            [
                [prefix_hash, entry["selected_sha256"]]
                for prefix_hash, entry in canonical_entries.items()
            ]
        ),
        "prefixes": canonical_entries,
    }
    _validate_manifest_document_structure(manifest, contract=contract)
    return manifest


def _validate_manifest_prefix_files(
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    extraction_identity: Mapping[str, Any],
    prefixes: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate each manifest-referenced file from one snapshot per artifact."""

    extraction_identity_sha256 = _sha256_json(extraction_identity)
    for prefix_hash in sorted(prefixes):
        entry = prefixes[prefix_hash]
        raw_path = paths.clustered_context_cache_raw_file(prefix_hash)
        selected_path = paths.context_cache_file(prefix_hash)
        raw_expected_metadata = {
            "schema_version": str(RAW_PREFIX_SCHEMA_VERSION),
            "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_raw_prefix",
            "feature_contract_fingerprint": contract.feature_contract_fingerprint,
            "prefix_hash": prefix_hash,
            "prefix_text_sha256": entry["prefix_text_sha256"],
            "extraction_identity_sha256": extraction_identity_sha256,
        }
        _load_raw_prefix_snapshot(
            raw_path,
            expected_metadata_without_n_tokens=raw_expected_metadata,
            n_sae_features=contract.n_sae_features,
            expected_sha256=cast(str, entry["raw_sha256"]),
        )
        selected_expected_metadata = {
            "schema_version": str(SELECTED_PREFIX_SCHEMA_VERSION),
            "kind": f"{CLUSTERED_CONTEXT_CACHE_KIND}_selected_prefix",
            "feature_contract_fingerprint": contract.feature_contract_fingerprint,
            "prefix_hash": prefix_hash,
            "prefix_text_sha256": entry["prefix_text_sha256"],
            "extraction_identity_sha256": extraction_identity_sha256,
            "raw_sha256": entry["raw_sha256"],
            "generator_state_after_sha256": (entry["generator_state_after_sha256"]),
        }
        structural = _validate_selected_file(
            selected_path,
            expected_bytes=None,
            expected_metadata=selected_expected_metadata,
            contract=contract,
        )
        for field in (
            "selected_sha256",
            "n_segments",
            "n_features_total",
            "n_positive_total",
            "n_negative_total",
        ):
            if structural[field] != entry[field]:
                _raise(f"clustered manifest structural count mismatch for {prefix_hash}.{field}")


def validate_clustered_context_cache_manifest(
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
    expected_extraction_identity: Mapping[str, Any] | None = None,
    expected_prefix_population_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the manifest and each raw/selected snapshot it commits."""

    manifest_path = paths.context_cache_manifest()
    if not manifest_path.is_file():
        _raise(f"clustered context-cache manifest is missing at {manifest_path}")
    manifest = _read_json_mapping(
        manifest_path,
        surface="clustered context-cache manifest",
    )
    extraction_identity, prefixes = _validate_manifest_document_structure(
        manifest,
        contract=contract,
    )
    if expected_extraction_identity is not None and extraction_identity != dict(
        expected_extraction_identity
    ):
        _raise("context-cache extraction identity mismatch; use a fresh run tag")
    if (
        expected_prefix_population_sha256 is not None
        and manifest["prefix_population_sha256"] != expected_prefix_population_sha256
    ):
        _raise("context-cache prefix population mismatch; use a fresh run tag")

    _validate_manifest_prefix_files(
        paths=paths,
        contract=contract,
        extraction_identity=extraction_identity,
        prefixes=prefixes,
    )
    return manifest


def _load_selected_context_segments(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[ContextFeatureSegment, ...]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _raise(f"could not read selected prefix bytes from {path}: {exc}")
    if _sha256_bytes(payload) != expected_sha256:
        _raise(f"clustered prefix content digest mismatch for {path.stem}")
    try:
        tensors = load_safetensors_bytes(payload)
        n_segments = int(tensors["n_segments"].item())
    except Exception as exc:
        _raise(f"could not load selected prefix tensors from {path}: {exc}")
    segments: list[ContextFeatureSegment] = []
    for index in range(n_segments):
        try:
            ids = tensors[f"seg_{index}_feature_ids"].tolist()
            buckets = tensors[f"seg_{index}_buckets"].tolist()
            char_start, char_end = tensors[f"seg_{index}_char_bounds"].tolist()
        except Exception as exc:
            _raise(f"could not load selected prefix segment {index} from {path}: {exc}")
        segments.append(
            ContextFeatureSegment(
                feature_ids=tuple(int(value) for value in ids),
                buckets=tuple(int(value) for value in buckets),
                char_start=int(char_start),
                char_end=int(char_end),
            )
        )
    return tuple(segments)


def load_clustered_context_segments_by_prefix_hash(
    *,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    """Load downstream segments only after full manifest validation."""

    manifest = validate_clustered_context_cache_manifest(
        paths=paths,
        contract=contract,
    )
    prefixes = cast(dict[str, dict[str, Any]], manifest["prefixes"])
    return {
        prefix_hash: _load_selected_context_segments(
            paths.context_cache_file(prefix_hash),
            expected_sha256=cast(str, prefixes[prefix_hash]["selected_sha256"]),
        )
        for prefix_hash in sorted(prefixes)
    }


@beartype
def build_clustered_context_cache(
    *,
    cfg: Any,
    output_paths: AxBenchBpbAdvPaths,
    prefix_population: CanonicalPrefixPopulation,
    base_model: Any,
    base_sae: Any,
    cluster_contract: ClusterFeatureContract,
    ddp_rank: NonNegativeIntLike = 0,
    ddp_world_size: PositiveIntLike = 1,
) -> Path:
    """Build or exactly resume the clustered two-phase BPB context cache."""

    # Irreducible: no collective is safe until these coordinates agree with the
    # initialized process group.
    _validate_ddp(rank=ddp_rank, world_size=ddp_world_size)

    segment_size: int | None = None
    extraction_max_tokens: int | None = None
    extraction_identity: dict[str, Any] | None = None
    extraction_identity_sha256: str | None = None
    local_unique: dict[str, str] = {}
    population_entries: list[dict[str, str]] = []
    prefix_population_sha256: str | None = None
    preflight_error: str | None = None
    try:
        segment_size, extraction_max_tokens = _validate_runtime_configuration(
            cfg=cfg,
            contract=cluster_contract,
        )
        extraction_identity = _expected_extraction_identity(
            contract=cluster_contract,
            segment_size=segment_size,
            extraction_max_tokens=extraction_max_tokens,
        )
        extraction_identity_sha256 = _sha256_json(extraction_identity)
        local_unique = prefix_population.as_dict()
        population_entries = _population_entries(local_unique)
        if not population_entries:
            _raise("source synth artifacts contain no eligible prefixes")
        prefix_population_sha256 = _sha256_json(population_entries)
    except Exception as exc:
        preflight_error = _status_error_text(exc)
    preflight_statuses = _all_gather_object(
        {
            "rank": ddp_rank,
            "error": preflight_error,
            "extraction_identity_sha256": extraction_identity_sha256,
            "prefix_population_sha256": prefix_population_sha256,
            "n_prefixes": len(population_entries),
        },
        world_size=ddp_world_size,
    )
    _raise_gathered_errors(
        preflight_statuses,
        phase="preflight and prefix population resolution",
    )
    extraction_fingerprints = {
        status["extraction_identity_sha256"] for status in preflight_statuses
    }
    population_fingerprints = {status["prefix_population_sha256"] for status in preflight_statuses}
    prefix_counts = {status["n_prefixes"] for status in preflight_statuses}
    if (
        len(extraction_fingerprints) != 1
        or None in extraction_fingerprints
        or len(population_fingerprints) != 1
        or None in population_fingerprints
        or len(prefix_counts) != 1
    ):
        _raise(
            "DDP ranks resolved different extraction identities or prefix "
            "populations: extraction="
            f"{sorted(str(value) for value in extraction_fingerprints)}, "
            "populations="
            f"{sorted(str(value) for value in population_fingerprints)}, "
            f"counts={sorted(prefix_counts)}"
        )
    if (
        segment_size is None
        or extraction_max_tokens is None
        or extraction_identity is None
        or extraction_identity_sha256 is None
        or prefix_population_sha256 is None
    ):
        _raise("clustered cache preflight completed without canonical identities")

    manifest_path = output_paths.context_cache_manifest()
    manifest_error: str | None = None
    manifest_valid = False
    manifest_exists = False
    # G3c-D5: `manifest_exists` is read ONCE and reused for the gathered status.
    #
    # An earlier shape read the path a second time when building that status, so
    # a manifest APPEARING between the two reads produced exists=True with
    # valid=False, caught by a dedicated "ranks disagree about manifest
    # visibility" branch. That branch is deliberately gone, because it bought no
    # correctness:
    #
    #   * Cross-rank disagreement is already fully covered. exists=True implies
    #     validation ran, so a rank is either valid=True or has an error that
    #     _raise_gathered_errors surfaces first. Every disagreement therefore
    #     reaches the `valid` check below.
    #   * A manifest committed concurrently by another builder is resolved
    #     ATOMICALLY at install time, not here: install_bytes_once cannot be
    #     overwritten and accepts an existing artifact only when its bytes are
    #     exact. Identical bytes are a no-op; different bytes raise.
    #
    # A second racy read only narrows a window the atomic layer already closes
    # deterministically, while adding a branch no input can reach.
    try:
        # D1 window 2: this gate deliberately lives inside the next gathered
        # phase. A per-rank config skew must not let one rank raise while a peer
        # enters completed-manifest validation's collective.
        if bool(cfg.context_cache.force_rebuild):
            _raise(
                "clustered context_cache.force_rebuild=true is forbidden because "
                "clustered cache evidence is immutable within a run_tag; use a "
                "fresh run tag"
            )
        manifest_exists = manifest_path.exists()
        if manifest_exists:
            validate_clustered_context_cache_manifest(
                paths=output_paths,
                contract=cluster_contract,
                expected_extraction_identity=extraction_identity,
                expected_prefix_population_sha256=prefix_population_sha256,
            )
            manifest_valid = True
    except Exception as exc:
        manifest_error = _status_error_text(exc)
    manifest_statuses = _all_gather_object(
        {
            "rank": ddp_rank,
            "error": manifest_error,
            "valid": manifest_valid,
            "exists": manifest_exists,
        },
        world_size=ddp_world_size,
    )
    _raise_gathered_errors(
        manifest_statuses,
        phase="completed-manifest validation",
    )
    if any(status["valid"] for status in manifest_statuses):
        if not all(status["valid"] for status in manifest_statuses):
            # Report what each rank actually saw. "ranks disagree" alone does not
            # say WHICH rank or whether the dissenter even saw the file, and this
            # fires on filesystem skew where that distinction is the whole
            # diagnosis.
            observed = [
                {
                    "rank": status["rank"],
                    "exists": status["exists"],
                    "valid": status["valid"],
                }
                for status in manifest_statuses
            ]
            _raise(f"DDP ranks disagree about completed context-cache visibility: {observed}")
        return manifest_path

    partial_document = _partial_marker_document(
        contract=cluster_contract,
        extraction_identity=extraction_identity,
        prefix_population_sha256=prefix_population_sha256,
        world_size=ddp_world_size,
    )
    _ensure_partial_marker(
        paths=output_paths,
        expected=partial_document,
        rank=ddp_rank,
        world_size=ddp_world_size,
        contract=cluster_contract,
    )

    sorted_items = sorted(local_unique.items(), key=lambda item: item[1])
    raw_owned: dict[str, dict[str, str]] = {}
    raw_error: str | None = None
    try:
        for index, (prefix_text, prefix_hash) in enumerate(sorted_items):
            if index % ddp_world_size != ddp_rank:
                continue
            raw_path = output_paths.clustered_context_cache_raw_file(prefix_hash)
            prefix_text_sha256 = _sha256_text(prefix_text)
            expected_without_tokens = _raw_metadata(
                contract=cluster_contract,
                prefix_hash=prefix_hash,
                prefix_text_sha256=prefix_text_sha256,
                extraction_identity_sha256=extraction_identity_sha256,
                n_tokens=1,
            )
            expected_without_tokens.pop("n_tokens")
            if not raw_path.exists():
                raw = extract_raw_clustered_prefix(
                    prefix_text=prefix_text,
                    base_model=base_model,
                    base_sae=base_sae,
                    contract=cluster_contract,
                    segment_size=segment_size,
                    extraction_max_tokens=extraction_max_tokens,
                )
                raw_bytes = _serialize_raw_prefix(
                    raw,
                    metadata=_raw_metadata(
                        contract=cluster_contract,
                        prefix_hash=prefix_hash,
                        prefix_text_sha256=prefix_text_sha256,
                        extraction_identity_sha256=extraction_identity_sha256,
                        n_tokens=raw.n_tokens,
                    ),
                )
                _install_once(
                    raw_bytes,
                    raw_path,
                    surface="clustered context-cache raw prefix",
                )
            _raw, raw_sha256 = _load_raw_prefix_snapshot(
                raw_path,
                expected_metadata_without_n_tokens=expected_without_tokens,
                n_sae_features=cluster_contract.n_sae_features,
            )
            raw_owned[prefix_hash] = {
                "raw_sha256": raw_sha256,
                "prefix_text_sha256": prefix_text_sha256,
            }
    except Exception as exc:
        raw_error = _status_error_text(exc)
    raw_statuses = _all_gather_object(
        {
            "rank": ddp_rank,
            "error": raw_error,
            "owned": raw_owned,
        },
        world_size=ddp_world_size,
    )
    _raise_gathered_errors(raw_statuses, phase="raw extraction")
    all_raw: dict[str, dict[str, str]] = {}
    for status in raw_statuses:
        owned = status.get("owned")
        if not isinstance(owned, dict):
            _raise("raw extraction gather returned malformed ownership")
        for prefix_hash, entry in cast(dict[str, dict[str, str]], owned).items():
            if prefix_hash in all_raw:
                _raise(f"duplicate raw-extraction ownership for {prefix_hash}")
            all_raw[prefix_hash] = entry
    expected_prefix_hashes = {prefix_hash for _prefix_text, prefix_hash in sorted_items}
    if set(all_raw) != expected_prefix_hashes:
        _raise(
            "raw extraction ownership is incomplete: "
            f"missing={sorted(expected_prefix_hashes - set(all_raw))}, "
            f"extra={sorted(set(all_raw) - expected_prefix_hashes)}"
        )

    replay_error: str | None = None
    generator_state_sha256: str | None = None
    replay_content_sha256: str | None = None
    selected_owned: dict[str, dict[str, Any]] = {}
    try:
        replay_state = reconstruct_clustered_replay_state(cluster_contract)
        replay_content: list[list[str]] = []
        # Deliberately replay ALL prefixes on EVERY rank, including an owner's
        # already-written selected files. Skipping any existing artifact would
        # change the shared generator stream for every downstream prefix.
        for index, (prefix_text, prefix_hash) in enumerate(sorted_items):
            raw_path = output_paths.clustered_context_cache_raw_file(prefix_hash)
            expected_without_tokens = _raw_metadata(
                contract=cluster_contract,
                prefix_hash=prefix_hash,
                prefix_text_sha256=_sha256_text(prefix_text),
                extraction_identity_sha256=extraction_identity_sha256,
                n_tokens=1,
            )
            expected_without_tokens.pop("n_tokens")
            raw, raw_sha256 = _load_raw_prefix_snapshot(
                raw_path,
                expected_metadata_without_n_tokens=expected_without_tokens,
                n_sae_features=cluster_contract.n_sae_features,
                expected_sha256=all_raw[prefix_hash]["raw_sha256"],
            )
            selected_segments = [
                _materialize_segment(
                    segment,
                    replay_state=replay_state,
                    contract=cluster_contract,
                )
                for segment in raw.segments
            ]
            state_after = _generator_state_sha256(replay_state.pipeline)
            selected_metadata = _selected_metadata(
                contract=cluster_contract,
                prefix_hash=prefix_hash,
                prefix_text_sha256=_sha256_text(prefix_text),
                extraction_identity_sha256=extraction_identity_sha256,
                raw_sha256=raw_sha256,
                generator_state_after_sha256=state_after,
            )
            selected_bytes = _serialize_selected_prefix(
                selected_segments,
                metadata=selected_metadata,
                contract=cluster_contract,
            )
            selected_sha256 = _sha256_bytes(selected_bytes)
            replay_content.append([prefix_hash, selected_sha256])
            if index % ddp_world_size == ddp_rank:
                selected_path = output_paths.context_cache_file(prefix_hash)
                if not selected_path.exists():
                    _install_once(
                        selected_bytes,
                        selected_path,
                        surface="clustered context-cache selected prefix",
                    )
                structural = _validate_selected_file(
                    selected_path,
                    expected_bytes=selected_bytes,
                    expected_metadata=selected_metadata,
                    contract=cluster_contract,
                )
                if structural["selected_sha256"] != selected_sha256:
                    _raise(f"selected prefix {prefix_hash} changed during owner write")
                selected_owned[prefix_hash] = {
                    "prefix_text_sha256": _sha256_text(prefix_text),
                    "raw_sha256": raw_sha256,
                    "selected_sha256": selected_sha256,
                    "n_segments": structural["n_segments"],
                    "n_features_total": structural["n_features_total"],
                    "n_positive_total": structural["n_positive_total"],
                    "n_negative_total": structural["n_negative_total"],
                    "generator_state_after_sha256": state_after,
                }
        generator_state_sha256 = _generator_state_sha256(replay_state.pipeline)
        replay_content_sha256 = _sha256_json(replay_content)
    except Exception as exc:
        replay_error = _status_error_text(exc)
    replay_statuses = _all_gather_object(
        {
            "rank": ddp_rank,
            "error": replay_error,
            "generator_state_sha256": generator_state_sha256,
            "replay_content_sha256": replay_content_sha256,
            "owned": selected_owned,
        },
        world_size=ddp_world_size,
    )
    _raise_gathered_errors(
        replay_statuses,
        phase="clustered selection replay",
    )
    generator_digests = {status["generator_state_sha256"] for status in replay_statuses}
    replay_digests = {status["replay_content_sha256"] for status in replay_statuses}
    if len(generator_digests) != 1 or None in generator_digests:
        _raise(
            "DDP generator-state disagreement after replay: "
            f"{sorted(str(value) for value in generator_digests)}"
        )
    if len(replay_digests) != 1 or None in replay_digests:
        _raise(
            "DDP selected-content disagreement after replay: "
            f"{sorted(str(value) for value in replay_digests)}"
        )
    final_generator_state = cast(str, next(iter(generator_digests)))
    all_selected: dict[str, dict[str, Any]] = {}
    for status in replay_statuses:
        owned = status.get("owned")
        if not isinstance(owned, dict):
            _raise("selection replay gather returned malformed ownership")
        for prefix_hash, entry in cast(dict[str, dict[str, Any]], owned).items():
            if prefix_hash in all_selected:
                _raise(f"duplicate selected-prefix ownership for {prefix_hash}")
            all_selected[prefix_hash] = entry
    if set(all_selected) != expected_prefix_hashes:
        _raise(
            "selected-prefix ownership is incomplete: "
            f"missing={sorted(expected_prefix_hashes - set(all_selected))}, "
            f"extra={sorted(set(all_selected) - expected_prefix_hashes)}"
        )

    rank_zero_status: dict[str, Any] | None = None
    if ddp_rank == 0:
        try:
            manifest = _manifest_document(
                contract=cluster_contract,
                extraction_identity=extraction_identity,
                prefix_population_sha256=prefix_population_sha256,
                generator_state_sha256=final_generator_state,
                prefix_entries=all_selected,
            )
            # Validate every fact available to the reader before committing the
            # immutable completeness marker. The post-install validation below
            # remains as a visibility and race check.
            _validate_manifest_prefix_files(
                paths=output_paths,
                contract=cluster_contract,
                extraction_identity=extraction_identity,
                prefixes=cast(
                    dict[str, dict[str, Any]],
                    manifest["prefixes"],
                ),
            )
            _install_once(
                json.dumps(manifest, indent=2).encode("utf-8"),
                manifest_path,
                surface="clustered context-cache manifest",
            )
            validate_clustered_context_cache_manifest(
                paths=output_paths,
                contract=cluster_contract,
                expected_extraction_identity=extraction_identity,
                expected_prefix_population_sha256=prefix_population_sha256,
            )
            # The marker is removed only after the durable manifest and every
            # referenced snapshot have passed full validation.
            output_paths.clustered_context_cache_partial_marker().unlink(missing_ok=True)
            rank_zero_status = {"error": None}
        except Exception as exc:
            rank_zero_status = {"error": _status_error_text(exc)}
    commit_status = _broadcast_rank_zero_status(
        rank_zero_status,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )
    if commit_status.get("error") is not None:
        _raise(f"manifest commit failed: {commit_status['error']}")

    final_error: str | None = None
    try:
        validate_clustered_context_cache_manifest(
            paths=output_paths,
            contract=cluster_contract,
            expected_extraction_identity=extraction_identity,
            expected_prefix_population_sha256=prefix_population_sha256,
        )
    except Exception as exc:
        final_error = _status_error_text(exc)
    final_statuses = _all_gather_object(
        {"rank": ddp_rank, "error": final_error},
        world_size=ddp_world_size,
    )
    _raise_gathered_errors(
        final_statuses,
        phase="final manifest visibility",
    )
    logger.info(
        "clustered BPB-Adv context cache committed: prefixes={}, manifest={}",
        len(sorted_items),
        manifest_path,
    )
    return manifest_path


__all__ = [
    "CLUSTERED_CONTEXT_CACHE_KIND",
    "CLUSTERED_CONTEXT_CACHE_SCHEMA_VERSION",
    "ClusteredReplayState",
    "RawClusteredFeatureSegment",
    "RawClusteredPrefix",
    "build_clustered_context_cache",
    "extract_raw_clustered_prefix",
    "load_clustered_context_segments_by_prefix_hash",
    "reconstruct_clustered_replay_state",
    "select_raw_clustered_segment",
    "validate_clustered_context_cache_manifest",
]
