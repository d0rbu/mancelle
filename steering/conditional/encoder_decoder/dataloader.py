"""Encoder-decoder conditional pretraining dataloader.

Provides two data-loading paths:

- **Live**: ``_document_batches_with_features`` reads text parquets, tokenizes
  documents, looks up features from ``FeatureIndex``, and yields each document
  chunk with a JSON-safe, cursor-aligned approximate-resume state.
- **Precomputed**: ``precomputed_data_loader()`` reads packed row shards from
  disk, applies feature selection via ``reconstruct_and_select()``, and yields
  batches for training.

Shared helpers (_build_row_segments, _pad_segments, SegFeatures) are kept
for the precompute scripts and SFT training.
"""

import atexit
import hashlib
import json
import os
import queue
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory as _shared_memory
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import torch as t
from beartype import beartype
from loguru import logger

from errors import FeatureClustersError, GradientRankingUnsupportedError
from nanochat.common import get_dist_info
from nanochat.dataset import DatasetSplit, list_parquet_files
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    parse_clustered_selection_validation,
    parse_completed_splice_validation,
)
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    DeltaMode,
    DeltaSamplingWeights,
    compute_doc_counterfactual_deltas,
    compute_doc_deltas,
    sample_delta_negatives,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.discretization import (
    compute_percentiles,
    discretize_activations,
)
from steering.conditional.encoder_decoder.conditioning_splice import (
    ConditioningSpliceIndex,
    ConditioningSpliceSidecars,
    validate_splice_identity_resume,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.fp_target_source import (
    validate_fp_target_source_certificate,
    validate_fp_target_source_certificate_remote,
)
from steering.conditional.encoder_decoder.sorted_doc_buffer import SortedDocBuffer
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    FeatureClusterDescriptor,
    assert_clustered_selection_shard_integrity,
    assert_feature_clusters_selection_metadata,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import FeatureMask, canonical_dict
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
    append_negative_features,
    negative_sampling_mask,
)
from steering.conditional.feature_vocab import META_KEY as FEATURE_VOCAB_META_KEY
from steering.conditional.percentile_table import uint8_to_percentile
from steering.conditional.sampling import (
    SamplingBundle,
    sample_features_batched,
    sample_inactive_features_batched,
    sample_k,
    sample_k_batched,
)
from steering.conditional.splice_counterfactual import (
    canonical_feature_delta_dict,
    derived_feature_fingerprint,
    packed_row_fingerprint,
    splice_sidecar_fingerprint,
    tensor_content_fingerprint,
    validate_packed_provenance,
)
from steering.conditional.splice_counterfactual import (
    resolve_counterfactual_refs as _resolve_counterfactual_refs,
)
from steering.conditional.splice_counterfactual import (
    validate_counterfactual_delta_results as _validate_counterfactual_delta_results,
)
from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
)
from steering.conditional.splice_selection_certificate import SPLICE_RESUME_METADATA_KEY
from steering.conditional.thresholding import (
    ThresholdConfig,
    apply_threshold,
)
from steering.feature_paths import verify_segment_size
from steering.gradient_overlay import provenance_dir
from steering.immutable_io import is_sha256_hex
from steering.node_shard_cache import NodeLocalShardCache
from steering.overlay_io import COMPRESSED_EXT, glob_overlay_shards, read_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    ReconstructedBatch,
    _BatchIndex,
    _build_batch_index,
    load_packed_row_shard,
    load_packed_row_shard_bytes,
    load_packed_row_shard_safetensors_bytes,
    load_selection_shard,
    load_selection_shard_bytes,
    reconstruct_batch,
)
from steering.profiling.heartbeat import Heartbeat
from steering.refined_types import (
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
)
from steering.remote_training import (
    RemoteMaterializationCancelled,
    RemotePackedRowMaterializer,
    RemoteSelectionMaterializer,
)
from steering.segmentation import assign_tokens_to_segments, token_to_char_offsets

# Default document buffer size for best-fit packing. Used by both the live
# dataloader and the precompute script to ensure identical packing behavior.
DEFAULT_BUFFER_SIZE = 1000

_RNG_SNAPSHOT_KEYS = frozenset({"device_type", "device_index", "state_hex"})
_PIPELINE_RNG_SNAPSHOT_KEYS = frozenset({*_RNG_SNAPSHOT_KEYS, "roles"})
# Fused negative-sampling tuning used by reconstruct_and_select().
_NEG_SMALL_POOL_EXACT_THRESHOLD = 128
_NEG_VEC_MAX_FACTOR = 32
_NEG_VEC_MIN_FACTOR = 4
_NEG_NO_MASK_OVERSAMPLE_FACTOR = 3


def _validate_positive_world_size(ddp_rank: int, ddp_world_size: int) -> None:
    if type(ddp_world_size) is not int or ddp_world_size <= 0:
        raise ValueError(f"ddp_world_size must be a positive int, got {ddp_world_size!r}")
    if type(ddp_rank) is not int or not 0 <= ddp_rank < ddp_world_size:
        raise ValueError(f"ddp_rank must be in [0, {ddp_world_size}), got {ddp_rank!r}")


def _validate_resume_world_size(
    resume_state_dict: Mapping[str, Any] | None,
    ddp_world_size: int,
    *,
    context: str,
) -> None:
    """Reject new-format topology drift; legacy states default to current.

    This guard must run before any striped cursor arithmetic or artifact access.
    A missing ``ddp_world_size`` is the documented legacy-checkpoint default and
    therefore remains resumable without an exactness claim.
    """
    if resume_state_dict is None or "ddp_world_size" not in resume_state_dict:
        return
    saved_world_size = resume_state_dict["ddp_world_size"]
    if type(saved_world_size) is not int or saved_world_size <= 0:
        raise ValueError(
            f"{context}: resume ddp_world_size must be a positive int, got {saved_world_size!r}"
        )
    if saved_world_size != ddp_world_size:
        raise ValueError(
            f"{context}: checkpoint ddp_world_size={saved_world_size} does not "
            f"match current ddp_world_size={ddp_world_size}; row-group/row "
            "striping cannot be resumed under a different topology"
        )


def build_dataloader_state_envelope(
    local_state: Mapping[str, Any],
    *,
    ddp_rank: int,
    ddp_world_size: int,
) -> dict[str, Any]:
    """Gather one JSON-safe loader state per rank for checkpoint metadata.

    Every rank must call this function under the same checkpoint predicate.
    The single-process path performs no collective and works without an
    initialized process group.
    """
    _validate_positive_world_size(ddp_rank, ddp_world_size)
    _validate_resume_world_size(
        local_state,
        ddp_world_size,
        context="dataloader checkpoint save",
    )
    local_copy = dict(local_state)
    if ddp_world_size == 1:
        rank_states: list[Any] = [local_copy]
    else:
        if not t.distributed.is_available() or not t.distributed.is_initialized():
            raise ValueError(
                "dataloader checkpoint save with ddp_world_size>1 requires an "
                "initialized torch.distributed process group"
            )
        actual_world_size = t.distributed.get_world_size()
        actual_rank = t.distributed.get_rank()
        if actual_world_size != ddp_world_size or actual_rank != ddp_rank:
            raise ValueError(
                "dataloader checkpoint save topology disagrees with the initialized "
                f"process group: arguments rank/world={ddp_rank}/{ddp_world_size}, "
                f"process-group rank/world={actual_rank}/{actual_world_size}"
            )
        rank_states = [None] * ddp_world_size
        t.distributed.all_gather_object(rank_states, local_copy)

    validated_rank_states: list[dict[str, Any]] = []
    for rank, state in enumerate(rank_states):
        if not isinstance(state, Mapping):
            raise ValueError(
                "dataloader checkpoint gather returned a non-mapping state for "
                f"rank {rank}: {type(state).__name__}"
            )
        _validate_resume_world_size(
            state,
            ddp_world_size,
            context=f"dataloader checkpoint rank {rank}",
        )
        validated_rank_states.append(dict(state))
    return {
        "ddp_world_size": ddp_world_size,
        "rank_states": validated_rank_states,
    }


def select_dataloader_rank_state(
    saved_state: Mapping[str, Any] | None,
    *,
    ddp_rank: int,
    ddp_world_size: int,
) -> dict[str, Any] | None:
    """Select this rank's local state from a new envelope or legacy flat dict."""
    _validate_positive_world_size(ddp_rank, ddp_world_size)
    if saved_state is None:
        return None
    if not isinstance(saved_state, Mapping):
        raise ValueError(
            f"dataloader_state_dict must be a mapping, got {type(saved_state).__name__}"
        )

    if "rank_states" not in saved_state:
        _validate_resume_world_size(
            saved_state,
            ddp_world_size,
            context="legacy/local dataloader resume",
        )
        return dict(saved_state)

    if "ddp_world_size" not in saved_state:
        raise ValueError("dataloader state envelope has rank_states but no ddp_world_size")
    _validate_resume_world_size(
        saved_state,
        ddp_world_size,
        context="dataloader state envelope",
    )
    rank_states = saved_state["rank_states"]
    if not isinstance(rank_states, list) or len(rank_states) != ddp_world_size:
        raise ValueError(
            "dataloader state envelope rank_states must be a list with one "
            f"entry per rank ({ddp_world_size}), got "
            f"{type(rank_states).__name__} length="
            f"{len(rank_states) if isinstance(rank_states, list) else 'n/a'}"
        )
    local_state = rank_states[ddp_rank]
    if not isinstance(local_state, Mapping):
        raise ValueError(
            f"dataloader state envelope rank {ddp_rank} entry must be a mapping, "
            f"got {type(local_state).__name__}"
        )
    _validate_resume_world_size(
        local_state,
        ddp_world_size,
        context=f"dataloader state envelope rank {ddp_rank}",
    )
    return dict(local_state)


def _rng_state_hex(state: t.Tensor) -> str:
    if state.dtype != t.uint8 or state.ndim != 1:
        raise ValueError(
            "torch RNG state must be a one-dimensional uint8 tensor, got "
            f"dtype={state.dtype}, shape={tuple(state.shape)}"
        )
    return bytes(state.cpu().tolist()).hex()


def _decode_rng_state_hex(value: Any, *, context: str) -> t.Tensor:
    if not isinstance(value, str) or not value or len(value) % 2:
        raise ValueError(f"{context}: state_hex must be a non-empty even-length string")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{context}: state_hex is not valid hexadecimal") from exc
    return t.tensor(list(raw), dtype=t.uint8)


def _canonical_device(device: str | t.device) -> t.device:
    resolved = t.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = t.device("cuda", t.cuda.current_device())
    return resolved


def _encode_rng_snapshot(state: t.Tensor, device: t.device) -> dict[str, Any]:
    return {
        "device_type": device.type,
        "device_index": device.index,
        "state_hex": _rng_state_hex(state),
    }


def _decode_rng_snapshot(
    snapshot: Any,
    expected_device: t.device,
    *,
    context: str,
    pipeline_entry: bool = False,
) -> t.Tensor:
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"{context}: RNG snapshot must be a mapping")
    expected_keys = _PIPELINE_RNG_SNAPSHOT_KEYS if pipeline_entry else _RNG_SNAPSHOT_KEYS
    if frozenset(snapshot) != expected_keys:
        raise ValueError(
            f"{context}: RNG snapshot keys must be {sorted(expected_keys)}, "
            f"got {sorted(str(key) for key in snapshot)}"
        )
    device_type = snapshot["device_type"]
    device_index = snapshot["device_index"]
    if device_type != expected_device.type or device_index != expected_device.index:
        raise ValueError(
            f"{context}: RNG device {device_type}:{device_index} does not match "
            f"current device {expected_device.type}:{expected_device.index}"
        )
    return _decode_rng_state_hex(snapshot["state_hex"], context=context)


def _feature_pipeline_generators(
    pipeline: FeaturePipelineConfig,
) -> list[tuple[list[str], t.Generator]]:
    candidates: list[tuple[str, t.Generator]] = []
    if isinstance(pipeline.selection_config, SamplingBundle):
        candidates.append(("selection", pipeline.selection_config.rng))
    if pipeline.negative_bundle is not None:
        candidates.append(("negative", pipeline.negative_bundle.rng))

    unique: list[tuple[list[str], t.Generator]] = []
    for role, generator in candidates:
        for roles, known_generator in unique:
            if generator is known_generator:
                roles.append(role)
                break
        else:
            unique.append(([role], generator))
    return unique


def _capture_feature_pipeline_rng_state(
    pipeline: FeaturePipelineConfig,
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for roles, generator in _feature_pipeline_generators(pipeline):
        device = _canonical_device(generator.device)
        snapshots.append(
            {
                "roles": list(roles),
                **_encode_rng_snapshot(generator.get_state(), device),
            }
        )
    return snapshots


def _restore_feature_pipeline_rng_state(
    pipeline: FeaturePipelineConfig,
    snapshots: Any,
    *,
    context: str,
) -> None:
    if not isinstance(snapshots, list):
        raise ValueError(f"{context}: feature_pipeline_rng_state must be a list")
    generators = _feature_pipeline_generators(pipeline)
    if len(snapshots) != len(generators):
        raise ValueError(
            f"{context}: feature_pipeline_rng_state has {len(snapshots)} "
            f"generator(s), but the current pipeline has {len(generators)}"
        )
    decoded: list[tuple[t.Generator, t.Tensor]] = []
    for idx, (snapshot, (roles, generator)) in enumerate(zip(snapshots, generators, strict=True)):
        if not isinstance(snapshot, Mapping) or snapshot.get("roles") != roles:
            raise ValueError(
                f"{context}: feature pipeline RNG entry {idx} roles must be "
                f"{roles}, got "
                f"{snapshot.get('roles') if isinstance(snapshot, Mapping) else None}"
            )
        device = _canonical_device(generator.device)
        state = _decode_rng_snapshot(
            snapshot,
            device,
            context=f"{context} feature pipeline RNG entry {idx}",
            pipeline_entry=True,
        )
        decoded.append((generator, state))
    for generator, state in decoded:
        try:
            generator.set_state(state)
        except RuntimeError as exc:
            raise ValueError(f"{context}: invalid feature pipeline RNG state") from exc


def _capture_default_rng_state(device: str | t.device) -> dict[str, Any]:
    resolved = _canonical_device(device)
    if resolved.type == "cpu":
        state = t.get_rng_state()
    elif resolved.type == "cuda":
        state = t.cuda.get_rng_state(resolved)
    else:
        raise ValueError(
            f"label-dropout RNG checkpointing supports cpu and cuda devices, got {resolved.type!r}"
        )
    return _encode_rng_snapshot(state, resolved)


def _restore_default_rng_state(
    snapshot: Any,
    device: str | t.device,
    *,
    context: str,
    apply: bool = True,
) -> None:
    resolved = _canonical_device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"label-dropout RNG checkpointing supports cpu and cuda devices, got {resolved.type!r}"
        )
    state = _decode_rng_snapshot(snapshot, resolved, context=context)
    current_state = t.get_rng_state() if resolved.type == "cpu" else t.cuda.get_rng_state(resolved)
    if state.numel() != current_state.numel():
        raise ValueError(
            f"{context}: invalid default-generator RNG state length "
            f"{state.numel()} (expected {current_state.numel()})"
        )
    if not apply:
        return
    try:
        if resolved.type == "cpu":
            t.set_rng_state(state)
        else:
            t.cuda.set_rng_state(state, resolved)
    except RuntimeError as exc:
        raise ValueError(f"{context}: invalid default-generator RNG state") from exc


class _CanonicalLabelDropoutStream:
    """Replay dropout masks in a checkpointed canonical row grouping."""

    @beartype
    def __init__(
        self,
        snapshot: Any,
        *,
        device: str | t.device,
        draw_batch_size: PositiveIntLike,
        draw_offset: NonNegativeIntLike = 0,
    ) -> None:
        if draw_offset >= draw_batch_size:
            raise ValueError(
                "label_dropout_draw_offset must be a strict int in "
                f"[0, {draw_batch_size}), got {draw_offset!r}"
            )
        self.device = _canonical_device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("canonical label-dropout RNG supports only cpu and cuda devices")
        state = _decode_rng_snapshot(
            snapshot,
            self.device,
            context="canonical label-dropout resume",
        )
        self.generator = t.Generator(device=self.device)
        try:
            self.generator.set_state(state)
        except RuntimeError as exc:
            raise ValueError("canonical label-dropout resume has invalid RNG state") from exc
        self.draw_batch_size = draw_batch_size
        self.offset = draw_offset
        self.group_state = state.clone()
        self.group: t.Tensor | None = None
        self.n_segments: int | None = None

    def _ensure_group(self, n_segments: int) -> None:
        if type(n_segments) is not int or n_segments <= 0:
            raise ValueError("canonical label-dropout segment width must be positive")
        if self.n_segments is None:
            self.n_segments = n_segments
        elif self.n_segments != n_segments:
            raise ValueError(
                "device-batch migration with label_dropout requires a stable "
                "padded segment width; checkpointed canonical draws use "
                f"{self.n_segments}, current batch uses {n_segments}"
            )
        if self.group is None:
            self.group = t.rand(
                self.draw_batch_size,
                n_segments,
                device=self.device,
                generator=self.generator,
            )

    def draw(self, rows: int, n_segments: int) -> tuple[dict[str, Any], int, t.Tensor]:
        if type(rows) is not int or rows <= 0:
            raise ValueError("canonical label-dropout execution batch must be positive")
        replay_state = _encode_rng_snapshot(self.group_state, self.device)
        replay_offset = self.offset
        parts: list[t.Tensor] = []
        remaining = rows
        while remaining:
            self._ensure_group(n_segments)
            assert self.group is not None
            take = min(remaining, self.draw_batch_size - self.offset)
            parts.append(self.group[self.offset : self.offset + take])
            self.offset += take
            remaining -= take
            if self.offset == self.draw_batch_size:
                self.group_state = self.generator.get_state().clone()
                self.group = None
                self.offset = 0
        return replay_state, replay_offset, t.cat(parts, dim=0)


def _max_selected_features_per_segment(
    sel_offsets: t.Tensor,
    neg_offsets: t.Tensor,
) -> int:
    """Return the exact maximum positive-plus-negative CSR row width.

    Keep this computation in tensors: production selection families contain
    well over one hundred million segments, so converting both offset arrays
    to Python lists creates a multi-gigabyte transient object graph and spends
    minutes in a scalar loop during every loader startup.
    """
    selected_counts = sel_offsets[1:] - sel_offsets[:-1]
    negative_counts = neg_offsets[1:] - neg_offsets[:-1]
    if selected_counts.shape != negative_counts.shape:
        raise ValueError(
            "selection positive/negative CSR offsets describe different "
            f"segment counts: {selected_counts.numel()} != "
            f"{negative_counts.numel()}"
        )
    total_counts = selected_counts + negative_counts
    return int(total_counts.max().item()) if total_counts.numel() else 0


def _pin_transfer(tensor: t.Tensor, device: str | t.device) -> t.Tensor:
    """Pin a CPU tensor and async-transfer it to *device*.

    When *device* is ``"cpu"``, returns the tensor unchanged.  Accepts both
    string device names (``"cuda"``, ``"cuda:0"``) and ``torch.device`` objects.
    """
    if str(device) == "cpu":
        return tensor
    return tensor.pin_memory().to(device, non_blocking=True)


def _zero_masked_fp_targets(
    fp_targets: t.Tensor,
    feature_mask: "FeatureMask | None",
    n_sae_features: int,
) -> t.Tensor:
    """Zero masked-feature columns of a dense (..., n_sae_features) FP-target
    tensor. Single source of truth for live/fused parity: BOTH the live encode
    path and build_fp_targets_from_shard call this so a future mechanism/dtype
    change cannot silently diverge them (the bug this consolidates fixed).
    I5: returns fp_targets UNCHANGED (same object/values) when feature_mask is
    None — no allocation, no dtype op — so the no-mask path stays byte-identical.
    """
    if feature_mask is not None:
        allowed = feature_mask.allowed_universe_mask(n_sae_features)
        fp_targets = fp_targets * allowed.to(fp_targets.dtype)
    return fp_targets


def _safe_empty_row_fallback(
    enc_mask: t.Tensor,
    feature_mask: "FeatureMask | None",
) -> None:
    """Guard against fully-empty rows that would produce NaN in cross-attention.

    Called after label-dropout (and any other mask-zeroing) to ensure that
    every row in the batch has at least one valid encoder position — OR is
    genuinely all-False (which cross-attention handles safely via nan_to_num).

    Behaviour (in-place on enc_mask):

    - ``feature_mask is None`` (I5 no-mask path):
        Old behaviour, byte-identical to pre-fix:
        flip ``enc_mask[row,0,0] = True`` for every empty row.
        No assertion is made about the value of enc_fids at that slot — it
        carries whatever fid was placed there during reconstruction/packing
        (historically fid 0 for zero-initialized live buffers and all-fid-0
        shards, but callers must not assume this).

    - ``feature_mask is not None`` (I4-safe path):
        Do NOT flip enc_mask for empty rows.  Leave the row all-False.
        Cross-attention already applies ``nan_to_num`` after SDPA (see
        ``cross_attention.py`` NaN-handling-order block), so an all-False
        row produces finite zero cross-attn output, matching an all-masked
        encoder input.
        Flipping enc_mask at slot (row,0,0) would activate whatever fid is
        there (typically 0 from zero-init), which may be blacklisted / not
        whitelisted → I4 violation.

    I5 statement-by-statement proof (no-mask path):
        1. ``if feature_mask is not None`` → False when feature_mask is None.
        2. We fall through to the else-branch (old guard).
        3. ``row_has_any``, ``row_all_empty``, and the conditional flip are
           identical to the original code at both sites.
        4. No new tensors or RNG calls introduced on the no-mask branch.
        ⇒ enc_fids and enc_mask are EXACTLY as before for all no-mask callers.
    """
    row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
    row_all_empty = ~row_has_any
    if not row_all_empty.any():
        return  # common fast-path: no empty rows, nothing to do
    if feature_mask is not None:
        # I4-safe: leave empty rows all-False.
        # Cross-attention nan_to_num handles all-masked segments correctly.
        return
    # No mask configured: preserve old NaN-guard behaviour (I5 byte-identical).
    enc_mask[row_all_empty, 0, 0] = True


@dataclass
class SegFeatures:
    """Per-segment feature data produced by thresholding + discretization."""

    feature_ids: t.Tensor  # (N,) int64
    buckets: t.Tensor  # (N,) int64
    percentiles: t.Tensor  # (N,) float32
    char_start: int
    char_end: int
    token_start: int | None = None  # precomputed nanochat token start (inclusive)
    token_end: int | None = None  # precomputed nanochat token end (exclusive)
    all_feature_ids: t.Tensor | None = None  # (M,) int64 — all features before selection
    all_feature_pcts: t.Tensor | None = None  # (M,) float32 — percentiles for all features
    prediction_target_only: bool = False  # True = kept past truncation for feature prediction
    is_splice: bool = False  # conditioning source is a splice junction segment


type _LiveDocumentBatch = tuple[
    list[tuple[list[int], list[SegFeatures] | None]],
    dict[str, Any],
]


@dataclass(frozen=True)
class EncDecTrainBatch:
    """Typed unpack of an encoder-decoder train/val batch tuple.

    Provides a structured view over the 10-tuple that the underlying
    dataloaders (``precomputed_data_loader`` /
    ``encoder_decoder_data_loader_with_state``) yield. Consumers call
    :meth:`from_tuple` to get a typed view rather than positional unpacking
    a raw tuple.

    Field meanings (all tensors batched on dim 0):

    inputs        (B, T) int64 — decoder input tokens.
    targets       (B, T) int64 — decoder targets (-1 = ignore_index).
    enc_fids      (B, n_seg, F) int64 — selected encoder feature ids per
                  segment (padded with 0; see enc_mask).
    enc_mags      (B, n_seg, F) int64 — magnitude buckets matching enc_fids.
    enc_mask      (B, n_seg, F) bool — True for real (non-padding) feature
                  slots, False for padding.
    enc_pcts      (B, n_seg, F) float32 — percentile scores in [0, 1) for
                  selected features (or sentinel 1.0 for negative-sample
                  rows; see :func:`apply_negative_sampling`).
    seg_map       (B, T) int64 — per-token segment index in [0, n_seg).
                  Non-decreasing along T.
    seg_doc_id    (B, n_seg) int64 — per-segment FP prediction-group id; -1
                  marks batch-padding slots that should be skipped by FP loss.
                  One group per packed document/conversation, except that a
                  steered SFT conversation opens a second group at its first
                  assistant-turn segment.
    fp_targets    (B, n_seg, n_sae_features) float32 or None — dense
                  feature-prediction targets when ``n_sae_features`` is set,
                  else None.
    state_dict    dict — opaque JSON-safe resume bookkeeping, including
                  cursor/row identities, DDP topology, and batch-affecting RNG
                  snapshots. Consumers SHOULD NOT introspect except for resume.

    The production loaders retain their established 10-tuple interface.
    Typed consumers normalize that interface at their boundary with
    :meth:`from_tuple`.
    """

    inputs: t.Tensor
    targets: t.Tensor
    enc_fids: t.Tensor
    enc_mags: t.Tensor
    enc_mask: t.Tensor
    enc_pcts: t.Tensor
    seg_map: t.Tensor
    seg_doc_id: t.Tensor
    fp_targets: t.Tensor | None
    state_dict: dict

    @classmethod
    def from_tuple(cls, batch: "tuple | list | EncDecTrainBatch") -> "EncDecTrainBatch":
        """Coerce a 10-tuple (or an existing EncDecTrainBatch) into a typed
        view.

        Accepts:

        - ``EncDecTrainBatch`` instance — passthrough.
        - 10-tuple/list ``(inputs, targets, enc_fids, enc_mags, enc_mask,
          enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict)`` — the only
          format both production producers
          (``precomputed_data_loader_with_state``,
          ``encoder_decoder_data_loader_with_state``) yield. ``fp_targets``
          may be ``None`` when ``n_sae_features`` is unset; ``None`` is a
          valid 10-tuple slot, not a separate 9-tuple format.

        Raises ``ValueError`` on any other shape (including 9-tuples — the
        legacy pre-fp_targets format is no longer accepted; see slice-10
        follow-up).
        """
        if isinstance(batch, cls):
            return batch
        if not isinstance(batch, (tuple, list)):
            raise ValueError(
                f"EncDecTrainBatch.from_tuple expected tuple/list/EncDecTrainBatch, "
                f"got {type(batch).__name__}"
            )
        # Cast each element to Any at the boundary: tuple/list element access
        # widens to ``object`` under ``ty``, but the runtime invariants are
        # documented above and exercised by the slice-10 audit tests
        # (test_encdec_train_batch.py).
        seq = cast("list[Any]", list(batch))
        if len(seq) != 10:
            raise ValueError(
                f"EncDecTrainBatch.from_tuple expected a 10-item "
                f"encoder-decoder batch, got {len(seq)} items"
            )
        return cls(
            inputs=seq[0],
            targets=seq[1],
            enc_fids=seq[2],
            enc_mags=seq[3],
            enc_mask=seq[4],
            enc_pcts=seq[5],
            seg_map=seq[6],
            seg_doc_id=seq[7],
            fp_targets=seq[8],
            state_dict=seq[9],
        )


def truncate_to_boundary_coverage(
    text_tokens: list[int],
    seg_features: list[SegFeatures] | None,
    context_len: int,
    segment_size: PositiveInt,
    doc_id: str = "",
) -> list[int]:
    """Truncate document tokens to precomputed boundary coverage.

    When the nanochat tokenizer is more efficient than Gemma (e.g. for
    number-heavy documents), the precomputed token boundaries may not cover
    all tokens up to context_len.  This function detects the gap and either
    truncates (expected small tail gap) or raises (boundaries fundamentally
    wrong).

    Note: this function compares nanochat *token indices* (token_start /
    token_end on SegFeatures), NOT character offsets.  The BOS-decode
    coordinate issue (where char offsets include the 5-char ``<bos>``
    prefix but raw text does not) does not apply here because token
    indices are always relative to the tokenized sequence and already
    account for the BOS token at position 0.

    Returns the (possibly truncated) token list.
    """
    if not seg_features:
        return text_tokens

    # Every segment must have precomputed token boundaries when the overlay
    # is loaded.  A None value means the overlay is missing or corrupt.
    for i, sf in enumerate(seg_features):
        if sf.token_start is None or sf.token_end is None:
            raise ValueError(
                f"doc {doc_id}: segment {i}/{len(seg_features)} has "
                f"token_start={sf.token_start}, token_end={sf.token_end}. "
                f"All segments must have precomputed token boundaries. "
                f"char_start={sf.char_start}, char_end={sf.char_end}"
            )

    # Step 1: truncate to context_len first
    if len(text_tokens) > context_len:
        text_tokens = text_tokens[:context_len]

    doc_token_count = len(text_tokens)
    last_precomputed_end = max(cast(int, sf.token_end) for sf in seg_features)

    # Case (a): fully covered
    if doc_token_count <= last_precomputed_end:
        return text_tokens

    # Gap exists — nanochat can be up to ~37% more efficient than Gemma on
    # extremely number-heavy text (digit pairing: "12345" → 3 nc tokens vs 5).
    # Worst observed across 19.7M docs: 37.4% (shard_00172:40031, 5/19.7M >30%).
    # Use 42% for headroom.
    gap = doc_token_count - last_precomputed_end
    max_gap = int(doc_token_count * 0.42)
    id_prefix = f"doc {doc_id}: " if doc_id else ""

    # Short docs with tiny gaps (1-3 tokens from zero-width byte-fallback
    # truncation at the segment boundary) are fine — just truncate.
    if gap <= 4:
        return text_tokens[:last_precomputed_end]

    # Case (c): gap exceeds tolerance — boundaries are fundamentally wrong
    if gap > max_gap:
        raise ValueError(
            f"{id_prefix}Precomputed token boundaries end at {last_precomputed_end}, "
            f"gap={gap} exceeds max_gap={max_gap} "
            f"(doc_token_count={doc_token_count}, context_len={context_len}, "
            f"segment_size={segment_size}, n_segments={len(seg_features)}). "
            f"This indicates a bug in precomputed boundaries."
        )

    # Case (b): gap within tolerance — truncate
    if gap > max_gap * 0.5:
        logger.error(
            f"{id_prefix}Boundary coverage gap: last_precomputed_end={last_precomputed_end}, "
            f"doc_token_count={doc_token_count}, gap={gap} "
            f"({gap / max_gap:.1%} of max_gap={max_gap}), "
            f"context_len={context_len}, segment_size={segment_size}. Truncating."
        )
    elif gap > max_gap * 0.25:
        logger.warning(
            f"{id_prefix}Boundary coverage gap: last_precomputed_end={last_precomputed_end}, "
            f"doc_token_count={doc_token_count}, gap={gap} "
            f"({gap / max_gap:.1%} of max_gap={max_gap}), "
            f"context_len={context_len}, segment_size={segment_size}. Truncating."
        )

    return text_tokens[:last_precomputed_end]


@dataclass
class _PreparedLiveRowGroup:
    """Phase-1 outputs consumed in the original row-group order."""

    doc_tokens_list: list[list[int]]
    doc_segments_list: list[list]
    doc_filtered_list: list[list]
    doc_decoded_pcts: list[list[t.Tensor | None]]
    rg_all_fids_parts: list[t.Tensor]
    rg_all_mags_parts: list[t.Tensor]
    rg_all_pcts_parts: list[t.Tensor | None]
    rg_seg_sizes: list[int]
    rg_override_weights: list[t.Tensor | None]
    use_sampling: bool
    has_precomputed: bool
    has_delta: bool | None
    shard_expected_cf_seen: int
    shard_scored_cf_seen: int


def _prepare_live_row_group_documents(
    *,
    texts: list,
    tokenizer,
    bos_token: int,
    feature_index: FeatureIndex,
    pipeline: FeaturePipelineConfig,
    segment_size: PositiveInt,
    context_len: int | None,
    shard_stem: str,
    global_doc_offset: int,
    splice_provenance_index: SpliceProvenanceIndex | None,
    shard_expected_cf_seen: int,
    shard_scored_cf_seen: int,
) -> _PreparedLiveRowGroup:
    """Tokenize, look up, and delta-filter documents without sampling."""

    # Per-doc results from phase 1
    doc_tokens_list: list[list[int]] = []
    doc_segments_list: list[list] = []  # raw SegmentFeatures per doc
    doc_filtered_list: list[
        list[
            tuple[
                t.Tensor,
                t.Tensor,
                t.Tensor | None,
                t.Tensor | None,
                t.Tensor | None,
                t.Tensor | None,
            ]
        ]
    ] = []
    doc_decoded_pcts: list[list[t.Tensor | None]] = []  # decoded percentiles per doc

    # Row-group-level accumulators for batched percentile call
    rg_all_fids_parts: list[t.Tensor] = []
    rg_all_mags_parts: list[t.Tensor] = []
    rg_all_pcts_parts: list[t.Tensor | None] = []
    rg_seg_sizes: list[int] = []
    rg_override_weights: list[t.Tensor | None] = []

    use_sampling = isinstance(pipeline.selection_config, SamplingBundle)
    has_precomputed = feature_index.has_precomputed_percentiles
    has_delta: bool | None = None

    with t.profiler.record_function("live/tokenize"):
        all_token_lists = tokenizer.encode(texts, prepend=bos_token)

    for local_idx, text in enumerate(texts):  # noqa: B007
        doc_idx = global_doc_offset + local_idx
        text_tokens = all_token_lists[local_idx]

        with t.profiler.record_function("live/feature_lookup"):
            segments = feature_index.lookup(shard_stem, doc_idx)
        # Re-check after lookup so the shard is loaded and checksum validated
        has_precomputed = feature_index.has_precomputed_percentiles
        # lookup() raises KeyError (naming shard+doc) if the row is absent
        # — corruption. No None to guard against (issue #521).

        # Decode precomputed percentiles for all segments
        decoded_pcts_list: list[t.Tensor | None] = []
        for seg in segments:
            if has_precomputed and seg.feature_percentiles is not None:
                decoded_pcts_list.append(uint8_to_percentile(seg.feature_percentiles))
            else:
                decoded_pcts_list.append(None)

        # Truncate to precomputed boundary coverage when available
        has_boundaries = (
            feature_index.has_precomputed_token_boundaries and segments[0].token_start is not None
        )
        if has_boundaries and context_len is not None:
            # Build lightweight SegFeatures for truncation (only needs
            # char/token boundary fields)
            raw_seg_features = [
                SegFeatures(
                    feature_ids=seg.feature_ids,
                    buckets=t.zeros(len(seg.feature_ids), dtype=t.int64),
                    percentiles=t.zeros(len(seg.feature_ids), dtype=t.float32),
                    char_start=seg.char_start,
                    char_end=seg.char_end,
                    token_start=seg.token_start,
                    token_end=seg.token_end,
                )
                for seg in segments
            ]
            text_tokens = truncate_to_boundary_coverage(
                text_tokens,
                raw_seg_features,
                context_len=context_len,
                segment_size=segment_size,
                doc_id=f"{shard_stem}:{doc_idx}",
            )

        doc_tokens_list.append(text_tokens)
        doc_segments_list.append(segments)
        doc_decoded_pcts.append(decoded_pcts_list)

        # Delta filtering (per-doc, batched within doc).
        # batch_results is None (no delta), or a per-segment list
        # aligned with `segments` where a None slot means "no delta
        # for this segment — apply absolute-percentile conditioning".
        # prev_segment/cum_max never produce None slots;
        # splice_counterfactual produces None only for non-splice segments.
        has_delta = pipeline.delta_bundle is not None

        batch_results: list | None = None
        if has_delta and pipeline.delta_bundle.config.mode is DeltaMode.splice_counterfactual:
            assert splice_provenance_index is not None
            n_sae = pipeline.delta_bundle.percentile_table.shape[0]
            for s_idx, seg in enumerate(segments):
                if len(seg.feature_ids) == 0:
                    raise SpliceProvenanceError(
                        f"{shard_stem} doc {doc_idx} segment {s_idx}: "
                        "zero feature ids under "
                        "feature_delta.mode=splice_counterfactual — "
                        "corrupt feature shard."
                    )
            seg_refs, expected_cf_positions = _resolve_counterfactual_refs(
                splice_provenance_index,
                feature_index,
                shard_stem,
                doc_idx,
                segments,
                has_precomputed,
            )
            batch_results = compute_doc_counterfactual_deltas(
                [seg.feature_ids for seg in segments],
                [seg.feature_mags for seg in segments],
                decoded_pcts_list,
                seg_refs,
                n_sae,
                pipeline.delta_bundle.percentile_table,
            )
            shard_expected_cf_seen += len(expected_cf_positions)
            shard_scored_cf_seen += _validate_counterfactual_delta_results(
                shard_stem=shard_stem,
                doc_idx=doc_idx,
                expected_positions=expected_cf_positions,
                seg_refs=seg_refs,
                batch_results=batch_results,
            )
        elif has_delta:
            assert pipeline.delta_bundle is not None
            n_sae = pipeline.delta_bundle.percentile_table.shape[0]
            batch_results = compute_doc_deltas(
                [seg.feature_ids for seg in segments],
                [seg.feature_mags for seg in segments],
                decoded_pcts_list,
                n_sae,
                pipeline.delta_bundle.config.mode,
                pipeline.delta_bundle.percentile_table,
            )

        filtered_segments: list[
            tuple[
                t.Tensor,
                t.Tensor,
                t.Tensor | None,
                t.Tensor | None,
                t.Tensor | None,
                t.Tensor | None,
            ]
        ] = []
        for s_idx, seg in enumerate(segments):
            r = batch_results[s_idx] if batch_results is not None else None
            if r is None:
                # (4a) Non-delta positive filter: keep only unmasked features.
                f_ids = seg.feature_ids
                f_mags_seg = seg.feature_mags
                f_pcts_seg = decoded_pcts_list[s_idx]
                if pipeline.feature_mask is not None:
                    if f_pcts_seg is not None:
                        f_ids, f_mags_seg, f_pcts_seg = pipeline.feature_mask.apply_to_positive(
                            f_ids, f_mags_seg, f_pcts_seg
                        )
                    else:
                        f_ids, f_mags_seg = pipeline.feature_mask.apply_to_positive(
                            f_ids, f_mags_seg
                        )
                filtered_segments.append((f_ids, f_mags_seg, None, None, None, None))
                if use_sampling:
                    rg_all_fids_parts.append(f_ids)
                    rg_all_mags_parts.append(f_mags_seg)
                    rg_seg_sizes.append(len(f_ids))
                    rg_all_pcts_parts.append(f_pcts_seg)
                    rg_override_weights.append(None)
            else:
                f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask, f_pcts = r
                # (4b) Delta positive filter: keep only unmasked positive features.
                # neg_fids/neg_w/ref_mask are NOT co-indexed with f_fids — delegate
                # to sample_delta_negatives (Task 9) via feature_mask forwarding.
                if pipeline.feature_mask is not None:
                    k = pipeline.feature_mask.keep_mask(f_fids)
                    f_fids = f_fids[k]
                    f_mags = f_mags[k]
                    if d_vals is not None:
                        d_vals = d_vals[k]
                    if f_pcts is not None:
                        f_pcts = f_pcts[k]
                filtered_segments.append((f_fids, f_mags, d_vals, neg_fids, neg_w, ref_mask))
                if use_sampling:
                    rg_all_fids_parts.append(f_fids)
                    rg_all_mags_parts.append(f_mags)
                    rg_seg_sizes.append(len(f_fids))
                    rg_all_pcts_parts.append(f_pcts)
                    ow = None
                    if d_vals is not None and len(d_vals) > 0:
                        dw = pipeline.selection_config.selection.delta_weights
                        if dw is DeltaSamplingWeights.percentile_delta:
                            ow = d_vals
                    rg_override_weights.append(ow)
        doc_filtered_list.append(filtered_segments)

    return _PreparedLiveRowGroup(
        doc_tokens_list=doc_tokens_list,
        doc_segments_list=doc_segments_list,
        doc_filtered_list=doc_filtered_list,
        doc_decoded_pcts=doc_decoded_pcts,
        rg_all_fids_parts=rg_all_fids_parts,
        rg_all_mags_parts=rg_all_mags_parts,
        rg_all_pcts_parts=rg_all_pcts_parts,
        rg_seg_sizes=rg_seg_sizes,
        rg_override_weights=rg_override_weights,
        use_sampling=use_sampling,
        has_precomputed=has_precomputed,
        has_delta=has_delta,
        shard_expected_cf_seen=shard_expected_cf_seen,
        shard_scored_cf_seen=shard_scored_cf_seen,
    )


def _sample_live_row_group_positive_features(
    prepared: _PreparedLiveRowGroup,
    pipeline: FeaturePipelineConfig,
):
    """Run the original single row-group positive-sampling call."""
    use_sampling = prepared.use_sampling
    has_precomputed = prepared.has_precomputed
    rg_all_fids_parts = prepared.rg_all_fids_parts
    rg_all_mags_parts = prepared.rg_all_mags_parts
    rg_all_pcts_parts = prepared.rg_all_pcts_parts
    rg_seg_sizes = prepared.rg_seg_sizes
    rg_override_weights = prepared.rg_override_weights

    if use_sampling:
        selection_config = cast(SamplingBundle, pipeline.selection_config)
        rg_all_fids = t.cat(rg_all_fids_parts) if rg_all_fids_parts else t.tensor([], dtype=t.int64)
        rg_all_mags = (
            t.cat(rg_all_mags_parts) if rg_all_mags_parts else t.tensor([], dtype=t.float32)
        )

        # Build precomputed percentiles tensor if all segments have them
        rg_precomputed_pcts: t.Tensor | None = None
        if has_precomputed and rg_all_pcts_parts and all(p is not None for p in rg_all_pcts_parts):
            rg_precomputed_pcts = t.cat([p for p in rg_all_pcts_parts if p is not None])

        rg_sampled_results = sample_features_batched(
            rg_all_fids,
            rg_all_mags,
            rg_seg_sizes,
            selection_config.percentile_table,
            selection_config.selection,
            selection_config.rng,
            override_weights_list=rg_override_weights,
            precomputed_percentiles=rg_precomputed_pcts,
        )
    return rg_sampled_results if use_sampling else None


def _finalize_live_row_group_documents(
    *,
    texts: list,
    prepared: _PreparedLiveRowGroup,
    rg_sampled_results,
    pipeline: FeaturePipelineConfig,
) -> list[tuple[list[int], list[SegFeatures] | None]]:
    """Discretize and negative-sample documents in original loop order."""
    doc_list: list[tuple[list[int], list[SegFeatures] | None]] = []
    doc_tokens_list = prepared.doc_tokens_list
    doc_segments_list = prepared.doc_segments_list
    doc_filtered_list = prepared.doc_filtered_list
    doc_decoded_pcts = prepared.doc_decoded_pcts
    use_sampling = prepared.use_sampling
    has_delta = prepared.has_delta

    rg_seg_offset = 0  # tracks position in rg_sampled_results

    for local_idx in range(len(texts)):
        text_tokens = doc_tokens_list[local_idx]
        segments = doc_segments_list[local_idx]
        filtered_segments = doc_filtered_list[local_idx]
        decoded_pcts_list = doc_decoded_pcts[local_idx]

        n_doc_segs = len(segments)

        # Get this doc's sampled results from the row-group batch
        if use_sampling:
            sampled_results = rg_sampled_results[rg_seg_offset : rg_seg_offset + n_doc_segs]
            rg_seg_offset += n_doc_segs
        elif isinstance(pipeline.selection_config, ThresholdSelectionConfig):
            threshold_cfg = ThresholdConfig(
                pipeline.selection_config.selection.threshold_type,
                pipeline.selection_config.selection.threshold,
            )
            sampled_results = []
            for filtered_fids, filtered_fmags, _dv, _nd, _nw, _rf in filtered_segments:
                fids, fmags = apply_threshold(
                    threshold_cfg,
                    filtered_fids,
                    filtered_fmags,
                    precomputed=pipeline.selection_config.precomputed_thresholds,
                )
                sampled_results.append((fids, fmags))
        else:
            raise TypeError(
                f"Unsupported selection config: {type(pipeline.selection_config).__name__}"
            )

        # Batched discretization for this doc
        disc_fids_parts = []
        disc_mags_parts = []
        disc_sizes = []
        for fids, fmags in sampled_results:
            disc_fids_parts.append(fids if len(fids) > 0 else t.tensor([], dtype=t.int64))
            disc_mags_parts.append(fmags if len(fmags) > 0 else t.tensor([], dtype=t.float32))
            disc_sizes.append(len(fids))
        all_disc_fids = (
            t.cat(disc_fids_parts)
            if any(s > 0 for s in disc_sizes)
            else t.tensor([], dtype=t.int64)
        )
        all_disc_mags = (
            t.cat(disc_mags_parts)
            if any(s > 0 for s in disc_sizes)
            else t.tensor([], dtype=t.float32)
        )

        if len(all_disc_fids) > 0:
            all_buckets = discretize_activations(all_disc_fids, all_disc_mags, pipeline.boundaries)
            all_pcts = compute_percentiles(all_disc_fids, all_disc_mags, pipeline.boundaries)
        else:
            all_buckets = t.tensor([], dtype=t.int64)
            all_pcts = t.tensor([], dtype=t.float32)

        disc_offset = 0
        per_seg_sampled: list[tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]] = []
        for i, sz in enumerate(disc_sizes):
            fids = disc_fids_parts[i]
            if sz > 0:
                buckets = all_buckets[disc_offset : disc_offset + sz]
                pcts = all_pcts[disc_offset : disc_offset + sz]
            else:
                buckets = t.tensor([], dtype=t.int64)
                pcts = t.tensor([], dtype=t.float32)
            per_seg_sampled.append((fids, disc_mags_parts[i], buckets, pcts))
            disc_offset += sz

        # Negative sampling: batch non-delta segments, keep delta per-segment
        # Pass 1: collect non-delta segments for batched negative sampling
        nondelta_indices: list[int] = []
        nondelta_active_ids: list[t.Tensor] = []
        delta_neg_results: dict[int, t.Tensor] = {}

        if pipeline.negative_bundle is not None:
            neg_mask = negative_sampling_mask(pipeline)
            for seg_i, seg in enumerate(segments):
                _, _, _, seg_neg_delta_fids, seg_neg_delta_weights, seg_ref_mask = (
                    filtered_segments[seg_i]
                )
                if has_delta and seg_neg_delta_fids is not None and seg_ref_mask is not None:
                    # Delta path: per-segment (stateful, needs delta context)
                    assert pipeline.delta_bundle is not None
                    assert seg_neg_delta_weights is not None
                    fids = per_seg_sampled[seg_i][0]
                    neg_k = sample_k(
                        pipeline.negative_bundle.selection.distribution,
                        pipeline.negative_bundle.selection.mean,
                        pipeline.negative_bundle.selection.std,
                        pipeline.negative_bundle.rng,
                        min_k=pipeline.negative_bundle.selection.min_k,
                        max_k=pipeline.negative_bundle.selection.max_k,
                    )
                    active_ids = seg.feature_ids
                    # Clamp to the allowed-universe pool consistent with the
                    # non-delta path. None means full width.
                    _allowed = (
                        neg_mask.allowed_universe_mask(pipeline.negative_bundle.n_sae_features)
                        if neg_mask is not None
                        else None
                    )
                    if _allowed is not None:
                        if len(active_ids) > 0:
                            _active_bits = _allowed.clone()
                            _active_bits[active_ids.long()] = False
                            _allowed_pool = int(_active_bits.sum().item())
                        else:
                            _allowed_pool = int(_allowed.sum().item())
                        neg_k = max(0, min(neg_k, _allowed_pool))
                    else:
                        neg_k = max(
                            0,
                            min(
                                neg_k,
                                pipeline.negative_bundle.n_sae_features - len(active_ids),
                            ),
                        )
                    delta_neg_results[seg_i] = sample_delta_negatives(
                        k=neg_k,
                        neg_delta_ratio=pipeline.delta_bundle.config.neg_delta_ratio,
                        neg_delta_fids=seg_neg_delta_fids,
                        neg_delta_weights=seg_neg_delta_weights,
                        raw_feature_ids=active_ids,
                        ref_mask=seg_ref_mask,
                        selected_positive_ids=fids,
                        n_sae_features=pipeline.negative_bundle.n_sae_features,
                        rng=pipeline.negative_bundle.rng,
                        feature_mask=neg_mask,
                    )
                else:
                    # Non-delta path: collect for batched call.
                    nondelta_indices.append(seg_i)
                    nondelta_active_ids.append(seg.feature_ids)

            # One batched call for all non-delta segments
            if nondelta_indices:
                batched_neg = sample_inactive_features_batched(
                    nondelta_active_ids,
                    pipeline.negative_bundle.n_sae_features,
                    pipeline.negative_bundle.selection,
                    pipeline.negative_bundle.rng,
                    feature_mask=neg_mask,
                )
                nondelta_neg_results = dict(
                    zip(nondelta_indices, batched_neg)  # noqa: B905
                )
            else:
                nondelta_neg_results = {}

        # Pass 2: merge negative results and build SegFeatures
        seg_features: list[SegFeatures] = []
        for seg_i, seg in enumerate(segments):
            fids, fmags, buckets, pcts = per_seg_sampled[seg_i]

            if pipeline.negative_bundle is not None:
                neg_fids = delta_neg_results.get(seg_i)
                if neg_fids is None:
                    neg_fids = nondelta_neg_results.get(seg_i)
                if neg_fids is not None:
                    # Shared with conditioned SFT: one owner for the
                    # "negatives ride in bucket effective_buckets" encoding.
                    fids, buckets, pcts = append_negative_features(
                        fids,
                        buckets,
                        pcts,
                        neg_fids,
                        effective_buckets=pipeline.effective_buckets,
                    )

            # Defense-in-depth (H-assert): assert finalized per-segment
            # ids (positives + negatives combined) contain no masked features.
            # Cheap: boolean membership over fids (tens–hundreds of ids).
            # Strictly mask-gated (I5: no-op when feature_mask is None).
            if pipeline.feature_mask is not None:
                pipeline.feature_mask.assert_no_masked(fids.to(t.int64))

            # Store pre-selection features for fp_targets.
            # Both must be set together (same length) or both None.
            all_pcts_i = decoded_pcts_list[seg_i]
            if all_pcts_i is not None:
                all_fids_i = seg.feature_ids.to(t.int64)
            else:
                all_fids_i = None

            seg_features.append(
                SegFeatures(
                    feature_ids=fids.to(t.int64),
                    buckets=buckets,
                    percentiles=pcts,
                    char_start=seg.char_start,
                    char_end=seg.char_end,
                    token_start=seg.token_start,
                    token_end=seg.token_end,
                    all_feature_ids=all_fids_i,
                    all_feature_pcts=all_pcts_i,
                )
            )

        doc_list.append((text_tokens, seg_features))

    # Postcondition: all sampled results consumed
    if use_sampling:
        assert rg_seg_offset == len(rg_sampled_results), (
            f"rg_seg_offset={rg_seg_offset} != len(rg_sampled_results)="
            f"{len(rg_sampled_results)} — segment count mismatch between "
            f"accumulation and consumption phases"
        )

    return doc_list


def _document_batches_with_features(
    split: DatasetSplit,
    resume_state_dict: dict | None,
    tokenizer_batch_size: int,
    tokenizer,
    feature_index: FeatureIndex,
    pipeline: FeaturePipelineConfig,
    segment_size: PositiveInt,
    data_dir: str | None = None,
    max_shards: int | None = None,
    context_len: int | None = None,
    include_splice_mask: bool = False,
) -> Iterator[_LiveDocumentBatch]:
    """Infinite iterator over tokenized documents with prepared encoder inputs.

    Handles DDP sharding (row groups striped across ranks) and approximate
    resume via pq_idx/rg_idx state dict.

    Args:
        split: Train or val split.
        resume_state_dict: Optional dict with pq_idx/rg_idx/epoch for resume.
        tokenizer_batch_size: Number of documents per yielded batch.
        tokenizer: Training tokenizer instance.
        feature_index: FeatureIndex for looking up segment features.
        pipeline: Feature pipeline config (selection, boundaries, negatives).
        segment_size: Extraction segment size in characters.
        data_dir: Optional override for dataset directory.
        max_shards: Optional limit on number of shards.
        context_len: When set and precomputed token boundaries exist, truncates
            documents to boundary coverage via truncate_to_boundary_coverage().
            Typically set to T+1 (row_capacity in the packing loop).

    Yields:
        ``(doc_list, cursor_state)`` where ``cursor_state`` contains
        ``pq_idx``, ``rg_idx``, ``epoch``, ``ddp_world_size``, and the
        cursor-aligned feature-pipeline RNG snapshot. ``doc_list`` is a list
        of (text_tokens, seg_features) tuples.
        seg_features is a list of SegFeatures per segment, or None if no features exist.
    """
    assert split in DatasetSplit, f"split must be one of {list(DatasetSplit)}, got {split}"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    _validate_positive_world_size(ddp_rank, ddp_world_size)
    _validate_resume_world_size(
        resume_state_dict,
        ddp_world_size,
        context="live dataloader resume",
    )
    if resume_state_dict is not None:
        if "feature_pipeline_rng_state" not in resume_state_dict:
            logger.warning(
                "Live dataloader: legacy resume state has no cursor-aligned "
                "feature_pipeline_rng_state; continuing with the configured seed. "
                "Live resume remains approximate."
            )
        else:
            _restore_feature_pipeline_rng_state(
                pipeline,
                resume_state_dict["feature_pipeline_rng_state"],
                context="live dataloader resume",
            )

    splice_provenance_index: SpliceProvenanceIndex | None = None
    if (
        pipeline.delta_bundle is not None
        and pipeline.delta_bundle.config.mode is DeltaMode.splice_counterfactual
    ):
        cf_feature_dir = feature_index.feature_dir
        if cf_feature_dir is None:
            raise SpliceProvenanceError(
                "feature_delta.mode=splice_counterfactual requires a FeatureIndex "
                "with a feature_dir (a spliced derived directory)."
            )
        splice_provenance_index = (
            ConditioningSpliceSidecars(Path(cf_feature_dir))
            if include_splice_mask
            else SpliceProvenanceIndex(cf_feature_dir)
        )

    cooldown_splice_index = splice_provenance_index if include_splice_mask else None
    splice_identity: dict[str, str] | None = None
    if include_splice_mask:
        source_dir = feature_index.feature_dir
        if source_dir is not None:
            source_dir = Path(source_dir)
            if (source_dir / "_splice_provenance").is_dir() or source_dir.parent.parent.name == "splice":
                if cooldown_splice_index is None:
                    cooldown_splice_index = ConditioningSpliceSidecars(source_dir)
        if cooldown_splice_index is not None:
            assert isinstance(cooldown_splice_index, ConditioningSpliceSidecars)
            splice_identity = {"sidecars": cooldown_splice_index.fingerprint}
        validate_splice_identity_resume(resume_state_dict, splice_identity)

    parquet_paths = list_parquet_files(data_dir)
    assert len(parquet_paths) != 0, "No dataset parquet files found"
    # When max_shards is set, slice BEFORE splitting train/val so the val shard
    # is the last of the limited set (not the global last parquet).
    if max_shards is not None:
        parquet_paths = parquet_paths[:max_shards]
    parquet_paths = parquet_paths[:-1] if split == DatasetSplit.TRAIN else parquet_paths[-1:]
    if max_shards is not None and split == DatasetSplit.TRAIN:
        logger.info(
            f"Limiting training to first {len(parquet_paths)} shards (max_shards={max_shards})"
        )

    bos_token = tokenizer.get_bos_token_id()

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            with t.profiler.record_function("live/open_parquet"):
                pf = pq.ParquetFile(filepath)
            shard_stem = Path(filepath).stem
            resumed_mid_shard = (
                first_pass and resume_state_dict is not None and pq_idx == resume_pq_idx
            )

            if splice_provenance_index is not None:
                n_sidecar = splice_provenance_index.shard_n_docs(shard_stem)
                shard_expected_cf_total = splice_provenance_index.shard_n_splice_positions(
                    shard_stem
                )
                n_rows = pf.metadata.num_rows
                if n_sidecar != n_rows:
                    raise SpliceProvenanceError(
                        f"{shard_stem}: sidecar n_docs={n_sidecar} != parquet "
                        f"rows={n_rows}. The data_dir does not pair with this "
                        f"derived feature_dir — pass data_dir=<feature_dir>/_parquets."
                    )
            else:
                shard_expected_cf_total = None
            if cooldown_splice_index is not None and splice_provenance_index is None:
                if cooldown_splice_index.shard_n_docs(shard_stem) != pf.metadata.num_rows:
                    raise SpliceProvenanceError(
                        f"{shard_stem}: cooldown splice sidecar document count does not "
                        "match parquet rows; use the paired derived data_dir"
                    )
            shard_expected_cf_seen = 0
            shard_scored_cf_seen = 0

            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank

            while rg_idx < pf.num_row_groups:
                # Compute global doc offset for this row group (accounts for all
                # prior row groups, including ones skipped by DDP striping)
                global_doc_offset = sum(pf.metadata.row_group(i).num_rows for i in range(rg_idx))

                with t.profiler.record_function("live/read_parquet_row_group"):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column("text").to_pylist()

                # ============================================================
                # Phase 1: Per-doc tokenization, feature lookup, delta filtering
                # ============================================================
                prepared = _prepare_live_row_group_documents(
                    texts=texts,
                    tokenizer=tokenizer,
                    bos_token=bos_token,
                    feature_index=feature_index,
                    pipeline=pipeline,
                    segment_size=segment_size,
                    context_len=context_len,
                    shard_stem=shard_stem,
                    global_doc_offset=global_doc_offset,
                    splice_provenance_index=splice_provenance_index,
                    shard_expected_cf_seen=shard_expected_cf_seen,
                    shard_scored_cf_seen=shard_scored_cf_seen,
                )
                shard_expected_cf_seen = prepared.shard_expected_cf_seen
                shard_scored_cf_seen = prepared.shard_scored_cf_seen

                # ============================================================
                # Phase 2: Row-group-level batched positive sampling
                # ============================================================
                rg_sampled_results = _sample_live_row_group_positive_features(
                    prepared,
                    pipeline,
                )

                # ============================================================
                # Phase 3 & 4: Per-doc discretization + negative sampling
                # ============================================================
                doc_list = _finalize_live_row_group_documents(
                    texts=texts,
                    prepared=prepared,
                    rg_sampled_results=rg_sampled_results,
                    pipeline=pipeline,
                )
                if cooldown_splice_index is not None:
                    for doc_offset, (_tokens, segments) in enumerate(doc_list):
                        doc_idx = global_doc_offset + doc_offset
                        expected = cooldown_splice_index.doc_n_segments(shard_stem, doc_idx)
                        if segments is None or len(segments) != expected:
                            raise SpliceProvenanceError(
                                f"{shard_stem} document {doc_idx}: cooldown splice "
                                "sidecar segment count does not match features"
                            )
                        junctions = cooldown_splice_index.doc_splices(shard_stem, doc_idx)
                        for ordinal, segment in enumerate(segments):
                            segment.is_splice = ordinal in junctions

                # This snapshot is aligned to the row-group cursor: all feature
                # selection draws for this row group have completed, but no later
                # row group has started. The outer best-fit buffer is not serialized,
                # so restoring it improves continuity without making live resume exact.
                cursor_state = {
                    "pq_idx": pq_idx,
                    "rg_idx": rg_idx,
                    "epoch": epoch,
                    "ddp_world_size": ddp_world_size,
                    "feature_pipeline_rng_state": (_capture_feature_pipeline_rng_state(pipeline)),
                }
                if include_splice_mask:
                    cursor_state["conditioning_splice_identity"] = splice_identity

                # Yield in tokenizer_batch_size chunks
                for i in range(0, len(doc_list), tokenizer_batch_size):
                    yield (
                        doc_list[i : i + tokenizer_batch_size],
                        cursor_state,
                    )

                rg_idx += ddp_world_size
            if splice_provenance_index is not None:
                if shard_expected_cf_seen != shard_scored_cf_seen:
                    raise SpliceProvenanceError(
                        f"{shard_stem}: scored {shard_scored_cf_seen} "
                        f"counterfactual segment(s) but expected "
                        f"{shard_expected_cf_seen} in row groups processed by "
                        f"rank {ddp_rank}."
                    )
                assert shard_expected_cf_total is not None
                if (
                    ddp_world_size == 1
                    and not resumed_mid_shard
                    and shard_scored_cf_seen != shard_expected_cf_total
                ):
                    raise SpliceProvenanceError(
                        f"{shard_stem}: scored {shard_scored_cf_seen} "
                        f"counterfactual segment(s) but the sidecar records "
                        f"{shard_expected_cf_total} splice position(s)."
                    )
                if ddp_world_size == 1 and not resumed_mid_shard:
                    logger.info(
                        "{}: splice_counterfactual scored {}/{} reference "
                        "counterfactual segment(s)",
                        shard_stem,
                        shard_scored_cf_seen,
                        shard_expected_cf_total,
                    )
                else:
                    logger.info(
                        "{}: splice_counterfactual scored {} rank-local "
                        "reference counterfactual segment(s); sidecar shard "
                        "total={} (rank {}, world_size={})",
                        shard_stem,
                        shard_scored_cf_seen,
                        shard_expected_cf_total,
                        ddp_rank,
                        ddp_world_size,
                    )
            pq_idx += 1
        first_pass = False
        epoch += 1


def _build_row_segments(
    row_docs: list[tuple[int, list[SegFeatures] | None, list[int]]],
    T: int,
    tokenizer,
    next_segment_conditioning_offset: bool = False,
    *,
    collect_buckets: bool = True,
    empty_percentile_dtype: t.dtype = t.float32,
) -> tuple[
    list[t.Tensor],
    list[t.Tensor],
    list[t.Tensor],
    t.Tensor,
    t.Tensor,
    list[t.Tensor],
    list[t.Tensor],
]:
    """Build seg_map and collect per-segment feature tensors for a packed row.

    When precomputed token boundaries are available (``token_start``/``token_end``
    on SegFeatures), uses them directly to build the seg_map -- skipping the
    expensive ``token_to_char_offsets`` + ``assign_tokens_to_segments`` + ``cummax``
    path entirely. Falls back to the char-level path when boundaries are missing.

    Returns raw per-segment lists (not padded). Callers that need padded tensors
    can pass the lists to ``_pad_segments``.

    Args:
        row_docs: [(start_pos, seg_features_or_None, doc_tokens_placed)] per packed doc.
        T: Sequence length.
        tokenizer: Training tokenizer (for progressive prefix decode).
        collect_buckets: Whether to collect each segment's discretized magnitude
            buckets. Live collation keeps the default; packed-row preprocessing
            disables it because packed CSR artifacts store percentiles and derive
            buckets only after downstream feature selection.
        empty_percentile_dtype: Dtype for synthesized empty percentile tensors. The
            live path keeps the float32 default; packed-row preprocessing can preserve
            already-encoded uint8 percentile tensors without a decode/re-encode cycle.

    Returns:
        per_seg_fids:      list of (F_i,) int64 tensors
        per_seg_buckets:   list of (F_i,) int64 tensors, or empty when collection
                           is disabled
        per_seg_pcts:      list of (F_i,) percentile tensors (caller dtype preserved)
        seg_map:           (T,) int64
        seg_doc_id:        (n_segs,) int64
        per_seg_all_fids:  list of (M_i,) int64 tensors — all features before selection
        per_seg_all_pcts:  list of (M_i,) percentile tensors — all features
    """
    seg_map = t.zeros(T, dtype=t.int64)
    per_seg_fids: list[t.Tensor] = []
    per_seg_buckets: list[t.Tensor] = []
    per_seg_pcts: list[t.Tensor] = []
    per_seg_all_fids: list[t.Tensor] = []
    per_seg_all_pcts: list[t.Tensor] = []
    per_seg_doc_id: list[int] = []
    global_seg = 0

    for doc_idx, (doc_start, seg_features, doc_tokens) in enumerate(row_docs):
        if doc_idx + 1 < len(row_docs):
            doc_end = min(row_docs[doc_idx + 1][0], T)
        else:
            doc_end = T
        if doc_start >= T:
            break

        n_ext_segs = len(seg_features) if seg_features is not None else 0
        n_doc_tokens = doc_end - doc_start

        if n_ext_segs == 0:
            # No features -> single empty segment
            seg_map[doc_start:doc_end] = global_seg
            per_seg_fids.append(t.tensor([], dtype=t.int64))
            if collect_buckets:
                per_seg_buckets.append(t.tensor([], dtype=t.int64))
            per_seg_pcts.append(t.tensor([], dtype=empty_percentile_dtype))
            per_seg_all_fids.append(t.tensor([], dtype=t.int64))
            per_seg_all_pcts.append(t.tensor([], dtype=empty_percentile_dtype))
            per_seg_doc_id.append(doc_idx)
            global_seg += 1
            continue

        if n_doc_tokens == 0:
            seg_map[doc_start:doc_end] = global_seg
            per_seg_fids.append(t.tensor([], dtype=t.int64))
            if collect_buckets:
                per_seg_buckets.append(t.tensor([], dtype=t.int64))
            per_seg_pcts.append(t.tensor([], dtype=empty_percentile_dtype))
            per_seg_all_fids.append(t.tensor([], dtype=t.int64))
            per_seg_all_pcts.append(t.tensor([], dtype=empty_percentile_dtype))
            per_seg_doc_id.append(doc_idx)
            global_seg += 1
            continue

        assert seg_features is not None
        # Check if precomputed token boundaries are available
        has_precomputed = (
            seg_features[0].token_start is not None and seg_features[0].token_end is not None
        )

        if has_precomputed:
            # Fast path: use precomputed token boundaries directly
            first_token_start = seg_features[0].token_start
            if first_token_start != 0:
                raise ValueError(
                    "precomputed token boundaries for packed document "
                    f"{doc_idx} at doc_start={doc_start} must have first "
                    f"token_start=0, got {first_token_start}"
                )
            n_segs_used = 0
            for s in range(n_ext_segs):
                sf = seg_features[s]
                tok_lo = sf.token_start
                tok_end = sf.token_end
                assert tok_lo is not None and tok_end is not None
                tok_hi = min(tok_end, n_doc_tokens)
                if tok_lo >= n_doc_tokens:
                    break
                seg_map[doc_start + tok_lo : doc_start + tok_hi] = global_seg + s
                n_segs_used = s + 1

            n_segs_used = max(n_segs_used, 1)
        else:
            # Slow path: char-level overlap assignment
            seg_char_starts = t.tensor(
                [sf.char_start for sf in seg_features],
                dtype=t.int64,
            )
            seg_char_ends = t.tensor(
                [sf.char_end for sf in seg_features],
                dtype=t.int64,
            )

            text_tokens = doc_tokens[:n_doc_tokens]  # includes BOS
            char_offsets = token_to_char_offsets(tokenizer, text_tokens)
            token_starts = t.tensor(char_offsets[:-1], dtype=t.int64)
            token_ends = t.tensor(char_offsets[1:], dtype=t.int64)

            assignments = assign_tokens_to_segments(
                token_starts,
                token_ends,
                seg_char_starts,
                seg_char_ends,
            )

            # Enforce monotonicity
            assignments = t.cummax(assignments, dim=0).values
            seg_map[doc_start:doc_end] = global_seg + assignments[:n_doc_tokens]

            n_segs_used = int(assignments.max().item()) + 1 if len(assignments) > 0 else 1
            n_segs_used = max(n_segs_used, 1)

        assert n_segs_used <= n_ext_segs, (
            f"char-level assignment produced segment index {n_segs_used - 1} "
            f"but only {n_ext_segs} extraction segments exist (doc at pos {doc_start})"
        )
        for s in range(n_segs_used):
            # In next-segment mode, decoder segment s consumes features from
            # extraction segment s + 1. The final segment has no next segment,
            # so it receives an explicit empty encoder slot.
            src_s = (s + 1) if next_segment_conditioning_offset else s
            if src_s < n_ext_segs:
                sf = seg_features[src_s]
                per_seg_fids.append(sf.feature_ids)
                if collect_buckets:
                    per_seg_buckets.append(sf.buckets)
                per_seg_pcts.append(sf.percentiles)
                per_seg_all_fids.append(
                    sf.all_feature_ids if sf.all_feature_ids is not None else sf.feature_ids
                )
                per_seg_all_pcts.append(
                    sf.all_feature_pcts if sf.all_feature_pcts is not None else sf.percentiles
                )
            else:
                per_seg_fids.append(t.tensor([], dtype=t.int64))
                if collect_buckets:
                    per_seg_buckets.append(t.tensor([], dtype=t.int64))
                per_seg_pcts.append(t.tensor([], dtype=empty_percentile_dtype))
                per_seg_all_fids.append(t.tensor([], dtype=t.int64))
                per_seg_all_pcts.append(t.tensor([], dtype=empty_percentile_dtype))
            per_seg_doc_id.append(doc_idx)

        global_seg += n_segs_used

    # Postcondition: seg_map values are valid indices
    if global_seg > 0:
        assert seg_map.max() < global_seg, f"seg_map max {seg_map.max()} >= global_seg {global_seg}"
    assert len(per_seg_fids) == global_seg, (
        f"per_seg_fids length {len(per_seg_fids)} != global_seg {global_seg}"
    )
    seg_doc_id = (
        t.tensor(per_seg_doc_id, dtype=t.int64) if per_seg_doc_id else t.tensor([0], dtype=t.int64)
    )
    return (
        per_seg_fids,
        per_seg_buckets,
        per_seg_pcts,
        seg_map,
        seg_doc_id,
        per_seg_all_fids,
        per_seg_all_pcts,
    )


def _pad_segments(
    per_seg_fids: list[t.Tensor],
    per_seg_buckets: list[t.Tensor],
    per_seg_pcts: list[t.Tensor],
    seg_map: t.Tensor,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Pad per-segment features to uniform F_max and return stacked tensors."""
    assert len(per_seg_fids) == len(per_seg_buckets), (
        f"per_seg_fids length {len(per_seg_fids)} != per_seg_buckets length {len(per_seg_buckets)}"
    )
    assert len(per_seg_fids) == len(per_seg_pcts), (
        f"per_seg_fids length {len(per_seg_fids)} != per_seg_pcts length {len(per_seg_pcts)}"
    )
    n_seg_total = max(1, len(per_seg_fids))
    f_max = max((len(f) for f in per_seg_fids), default=0)
    f_max = max(1, f_max)

    feature_ids = t.zeros(n_seg_total, f_max, dtype=t.int64)
    magnitudes = t.zeros(n_seg_total, f_max, dtype=t.int64)
    percentiles = t.zeros(n_seg_total, f_max, dtype=t.float32)
    mask = t.zeros(n_seg_total, f_max, dtype=t.bool)

    for i, (seg_fids, seg_buckets, seg_pcts) in enumerate(
        zip(per_seg_fids, per_seg_buckets, per_seg_pcts)  # noqa: B905
    ):
        n = len(seg_fids)
        if n == 0:
            continue
        feature_ids[i, :n] = seg_fids
        magnitudes[i, :n] = seg_buckets
        percentiles[i, :n] = seg_pcts
        mask[i, :n] = True

    return feature_ids, magnitudes, percentiles, mask, seg_map


def _prefetch_wrapper(inner_gen, prefetch_batches, use_cuda: bool = False):
    """Wrap a generator with a background-thread prefetch buffer.

    Runs the generator in a daemon thread, buffering up to `prefetch_batches`
    complete batches. This hides I/O latency (shard loads) behind GPU compute.

    Cross-stream safety: when *use_cuda* is True, the producer records a
    ``torch.cuda.Event`` on its current stream immediately after each yielded
    item materialises (capturing the non-blocking ``gpu_buffer.copy_`` and
    ``.clone()`` ops). Before re-yielding to its caller, the consumer makes its
    own current stream ``wait_event`` that event. This guarantees the consumer
    observes the fully-written tensor without needing an explicit
    ``torch.cuda.synchronize()``.
    """
    buf = queue.Queue(maxsize=prefetch_batches)
    sentinel = object()
    error_box = [None]

    # The producer issues all of its CUDA work (e.g. the sync unpinned
    # .to(device) copies inside reconstruct_batch) on a dedicated side stream.
    # On the DEFAULT stream each produced batch would block behind the entire
    # enqueued training step (issue #1507: 1.1 ms -> 1189.6 ms/batch under a
    # ~0.5 s enqueued load), capping the producer at ~1 batch/step. The event
    # record/wait pair below provides cross-stream ordering; the consumer
    # additionally record_stream()s each CUDA tensor so the caching allocator
    # does not recycle side-stream blocks still being read by the default
    # stream.
    side_stream = t.cuda.Stream() if use_cuda else None

    def produce_loop():
        for item in inner_gen:
            # Record an event on the producer's current stream AFTER the
            # inner generator has issued gpu_buffer.copy_ + .clone() for
            # this batch. The consumer will wait on this event before
            # reading the tensors. Event is None on CPU paths.
            if use_cuda:
                event = t.cuda.Event(blocking=False)
                event.record()
                buf.put((event, item))
            else:
                buf.put((None, item))

    def producer():
        try:
            if side_stream is not None:
                with t.cuda.stream(side_stream):
                    produce_loop()
            else:
                produce_loop()
        except Exception as e:
            error_box[0] = e
        finally:
            buf.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    while True:
        queued = buf.get()
        if queued is sentinel:
            if error_box[0] is not None:
                raise error_box[0]
            return
        event, item = queued
        if event is not None:
            # Make the consumer's current stream wait on the producer's event
            # so the subsequent reads see the completed copy_/.clone() output.
            # This replaces the previously implicit torch.cuda.synchronize()
            # at the training loop boundary.
            consumer_stream = t.cuda.current_stream()
            event.wait(consumer_stream)
            if side_stream is not None:
                for part in item:
                    if isinstance(part, t.Tensor) and part.is_cuda:
                        part.record_stream(consumer_stream)
        yield item


# ---------------------------------------------------------------------------
# Per-node decode-once packed-shard sharing over POSIX shared memory
# (issue #1507).
#
# Ordinary rank stripes traverse the same shard inventory, so without sharing
# each of a node's local ranks redundantly reads and zstd-decodes the same
# ~1.5 GiB shard — 8x the NFS traffic and CPU per node. Under torchrun
# (LOCAL_RANK / LOCAL_WORLD_SIZE present, local world > 1) exactly one local
# rank per shard — chosen by a stable hash every rank computes identically —
# decodes it and publishes the eight PackedRowShard tensors into one
# shared-memory segment; the other local ranks attach and wrap the buffers
# zero-copy via t.frombuffer. reconstruct_batch only reads the shard, so shared
# pages are safe. Loading is bitwise-identical to the private-decode path.
#
# Publish barrier: payload + JSON header are written first, the 16-byte magic
# last; attachers poll for the magic and warn/direct-decode after
# _SHARED_DECODE_TIMEOUT_S. Segment names hash (lexical absolute path, epoch)
# inside a per-launch namespace (SLURM_JOB_ID / TORCHELASTIC_RUN_ID / parent
# pid), so concurrent jobs and train/val loaders never collide.
#
# Lifetime: eviction only UNLINKS (removes the name) — it never closes the
# mapping. Each wrapped PackedRowShard holds a reference to its segment, so
# the pages live exactly as long as some shard object views them (no
# use-after-free for cross-shard pending batches). The retain window (4) is
# 2x the maximum producer-position skew between local ranks (the prefetch
# queue bounds lookahead to ~2 shards), so a name is never unlinked before a
# lagging sibling attaches; the attach timeout is the fail-loud backstop.
# Python 3.13 track=False keeps the resource tracker from unlinking segments
# owned by sibling ranks at exit; an atexit hook unlinks the owner's
# still-registered names on clean shutdown.
# ---------------------------------------------------------------------------

_SHARED_DECODE_MAGIC = b"NGPTSHARD1\0\0\0\0\0\0"
_SHARED_DECODE_HDR_CAP = 65536
_SHARED_DECODE_RETAIN = 4
_SHARED_DECODE_TIMEOUT_S = 120.0
_SHARED_DECODE_OWNER_TIMEOUT_S = 900.0
_SHARED_DECODE_REAP_TIMEOUT_S = 1.0
_SHARED_DECODE_SLOW_WARN_S = 30.0
_SHARED_DECODE_WORKER_PATH = Path(__file__).with_name("_shared_decode_worker.py")
_SHARD_LOOKAHEAD_POLL_SECONDS = 0.05
_SHARD_LOOKAHEAD_JOIN_SECONDS = 5.0
_SHARD_LOOKAHEAD_WARN_DEPTH = 16
_SHARD_LOOKAHEAD_ERROR_DEPTH = 32
_SHARD_LOOKAHEAD_EXCEPTION_DEPTH = 64
_shared_decode_segments: dict[str, tuple[Any, bool]] = {}  # name -> (SharedMemory, is_owner)
_shared_decode_retain_floor = _SHARED_DECODE_RETAIN
_shared_decode_evict_lock = threading.Lock()


def _shared_decode_retain_for_lookahead(depth: int) -> int:
    """Retain twice the queue-plus-FIFO lead between local-rank producers.

    A producer can lead a sibling by at most ``depth + 3`` shard positions.
    ``2 * (depth + 2)`` strictly exceeds that bound for every non-negative
    depth, while the max preserves the historical depth-zero retain window.
    """
    return max(_SHARED_DECODE_RETAIN, 2 * (depth + 2))


# ---------------------------------------------------------------------------
# Loader storage-health accounting (2026-08-28 hstor013 incidents: slow shard
# I/O surfaced only as consumer fallback warnings and, past the NCCL timeout,
# as watchdog kills — successful-but-slow loads were invisible). Every base
# shard load records its path (owner decode / shared attach / direct) and
# duration; sustained degradation shows up in one periodic summary line
# instead of requiring log archaeology across ranks.
# ---------------------------------------------------------------------------
_LOADER_HEALTH_SUMMARY_EVERY = 50
_loader_health_lock = threading.Lock()
_loader_health_scope_state = threading.local()


def _loader_health_zero() -> dict[str, float]:
    return {
        "base_loads": 0.0,
        "owner_decodes": 0.0,
        "shared_attaches": 0.0,
        "direct_loads": 0.0,
        "fallbacks": 0.0,
        "owner_decode_s_sum": 0.0,
        "owner_decode_s_max": 0.0,
        "attach_wait_s_sum": 0.0,
        "attach_wait_s_max": 0.0,
        "direct_load_s_sum": 0.0,
        "direct_load_s_max": 0.0,
    }


_loader_health = _loader_health_zero()

_LOADER_HEALTH_KINDS = {
    "owner_decode": ("owner_decodes", "owner_decode_s_sum", "owner_decode_s_max"),
    "shared_attach": ("shared_attaches", "attach_wait_s_sum", "attach_wait_s_max"),
    "direct_load": ("direct_loads", "direct_load_s_sum", "direct_load_s_max"),
}
_LOADER_HEALTH_ADDITIVE_KEYS = (
    "base_loads",
    "owner_decodes",
    "shared_attaches",
    "direct_loads",
    "fallbacks",
    "owner_decode_s_sum",
    "attach_wait_s_sum",
    "direct_load_s_sum",
)


def _update_loader_health(
    health: dict[str, float],
    kind: str,
    seconds: float,
    *,
    fallback: bool,
) -> None:
    count_key, sum_key, max_key = _LOADER_HEALTH_KINDS[kind]
    health["base_loads"] += 1
    health[count_key] += 1
    health[sum_key] += seconds
    health[max_key] = max(health[max_key], seconds)
    if fallback:
        health["fallbacks"] += 1


def format_loader_health(
    health: Mapping[str, float],
    *,
    include_max: bool,
) -> str:
    """Format one cumulative/scoped snapshot or one additive gate delta."""
    def duration(count_key: str, sum_key: str, max_key: str) -> str:
        count = health[count_key]
        if count == 0:
            return "-"
        mean = health[sum_key] / count
        return f"{mean:.1f}/{health[max_key]:.1f}s" if include_max else f"{mean:.1f}s"

    statistic_label = "mean/max s" if include_max else "mean s"
    return (
        f"loads={health['base_loads']:.0f} "
        f"owner={health['owner_decodes']:.0f} "
        f"attach={health['shared_attaches']:.0f} "
        f"direct={health['direct_loads']:.0f} "
        f"fallbacks={health['fallbacks']:.0f} | "
        f"{statistic_label}: "
        f"owner_decode={duration('owner_decodes', 'owner_decode_s_sum', 'owner_decode_s_max')} "
        f"attach_wait={duration('shared_attaches', 'attach_wait_s_sum', 'attach_wait_s_max')} "
        f"direct={duration('direct_loads', 'direct_load_s_sum', 'direct_load_s_max')}"
    )


def loader_health_snapshot() -> dict[str, float]:
    """A consistent copy of the loader storage-health counters."""
    with _loader_health_lock:
        return dict(_loader_health)


@contextmanager
def loader_health_scope() -> Iterator[dict[str, float]]:
    """Collect base-shard health recorded synchronously on this thread only."""
    if getattr(_loader_health_scope_state, "health", None) is not None:
        raise RuntimeError("loader health scopes may not be nested on one thread")
    health = _loader_health_zero()
    _loader_health_scope_state.health = health
    try:
        yield health
    finally:
        _loader_health_scope_state.health = None


def loader_health_delta(
    previous: Mapping[str, float],
    current: Mapping[str, float],
) -> dict[str, float]:
    """Return the additive health recorded between two cumulative snapshots."""
    delta = {key: current[key] - previous[key] for key in _LOADER_HEALTH_ADDITIVE_KEYS}
    negative = {key: value for key, value in delta.items() if value < 0}
    if negative:
        raise ValueError(f"loader health counters moved backwards: {negative}")
    return delta


def _loader_health_reset() -> None:
    global _loader_health
    with _loader_health_lock:
        _loader_health = _loader_health_zero()


def _loader_health_record(kind: str, seconds: float, *, fallback: bool = False) -> None:
    with _loader_health_lock:
        _update_loader_health(_loader_health, kind, seconds, fallback=fallback)
        emit_summary = _loader_health["base_loads"] % _LOADER_HEALTH_SUMMARY_EVERY == 0
        if emit_summary:
            summary = f"loader health: {format_loader_health(_loader_health, include_max=True)}"
    scoped_health = getattr(_loader_health_scope_state, "health", None)
    if scoped_health is not None:
        _update_loader_health(scoped_health, kind, seconds, fallback=fallback)
    if emit_summary:
        logger.info(summary)


def _shared_decode_local_topology() -> tuple[int, int] | None:
    """(LOCAL_RANK, LOCAL_WORLD_SIZE) under torchrun; None outside it."""
    local_rank = os.environ.get("LOCAL_RANK")
    local_world = os.environ.get("LOCAL_WORLD_SIZE")
    if local_rank is None or local_world is None:
        return None
    return int(local_rank), int(local_world)


def _shared_decode_run_namespace() -> str:
    """A namespace every local rank of one launch agrees on, unique per job."""
    return (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or str(os.getppid())
    )


def _shared_decode_unlink_registered() -> None:
    for name in list(_shared_decode_segments):
        segment, is_owner = _shared_decode_segments.pop(name)
        if is_owner:
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


atexit.register(_shared_decode_unlink_registered)


def _shared_decode_evict_old_segments(retain: int) -> None:
    # The registry and its retain floor are process-global, but two threads
    # reach here concurrently: the train loader's lookahead producer (a high
    # retain) and a validation loader on the main thread (retain 4), which
    # overlap because eval runs between training steps while the producer keeps
    # going. Without this lock the floor's read-modify-write can interleave as
    # train-reads-4, val-reads-4, train-writes-12, val-writes-4 — silently
    # LOWERING the floor and unlinking names a lagging sibling rank still needs,
    # which is the 120s attach timeout the floor exists to prevent. Holding the
    # lock across the trim also makes next(iter(...))/pop() atomic.
    global _shared_decode_retain_floor
    with _shared_decode_evict_lock:
        _shared_decode_retain_floor = max(_shared_decode_retain_floor, retain)
        _shared_decode_evict_locked()


def _shared_decode_evict_locked() -> None:
    while len(_shared_decode_segments) > _shared_decode_retain_floor:
        name = next(iter(_shared_decode_segments))
        segment, is_owner = _shared_decode_segments.pop(name)
        # Unlink only removes the NAME. The mapping stays alive through the
        # PackedRowShard that references this segment, so tensors from a
        # not-yet-consumed pending batch remain valid.
        if is_owner:
            try:
                segment.unlink()
            except FileNotFoundError:
                pass


def _shared_decode_publish(segment, shard: PackedRowShard) -> None:
    entries: dict[str, dict[str, Any]] = {}
    offset = _SHARED_DECODE_HDR_CAP
    keys = [
        "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
        "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
    ]
    for key in keys:
        tensor = getattr(shard, key).contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        entries[key] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "offset": offset,
            "nbytes": nbytes,
        }
        if nbytes:
            byte_view = tensor.view(t.uint8).view(-1)
            segment.buf[offset : offset + nbytes] = byte_view.numpy()
        offset += (nbytes + 63) // 64 * 64
    header = json.dumps({"tensors": entries}).encode()
    if len(header) > _SHARED_DECODE_HDR_CAP - 32:
        raise RuntimeError("shared decode: header overflow")
    segment.buf[16:24] = len(header).to_bytes(8, "little")
    segment.buf[24 : 24 + len(header)] = header
    segment.buf[0:16] = _SHARED_DECODE_MAGIC  # commit LAST


def _shared_decode_wrap(segment) -> PackedRowShard:
    header_len = int.from_bytes(bytes(segment.buf[16:24]), "little")
    header = json.loads(bytes(segment.buf[24 : 24 + header_len]))
    tensors: dict[str, t.Tensor] = {}
    for key, meta in header["tensors"].items():
        dtype = getattr(t, meta["dtype"])
        raw = segment.buf[meta["offset"] : meta["offset"] + meta["nbytes"]]
        if meta["nbytes"] == 0:
            tensors[key] = t.empty(meta["shape"], dtype=dtype)
        else:
            # frombuffer as uint8 then reinterpret: sidesteps torch's limited
            # frombuffer support for barebones dtypes (uint16).
            flat = t.frombuffer(raw, dtype=t.uint8)
            tensors[key] = flat.view(dtype).view(meta["shape"])
    # backing_segment ties the mapping's lifetime to the shard: pages stay
    # valid for exactly as long as any of these tensors is reachable, even
    # after the segment's name has been unlinked by eviction.
    return PackedRowShard(**tensors, backing_segment=segment)


def _shared_decode_payload_size(shard: PackedRowShard) -> int:
    total = _SHARED_DECODE_HDR_CAP
    for key in (
        "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
        "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
    ):
        tensor = getattr(shard, key)
        total += (tensor.numel() * tensor.element_size() + 63) // 64 * 64
    return total


class _QuietSharedMemory(_shared_memory.SharedMemory):
    """SharedMemory whose destructor tolerates still-exported tensor views.

    At interpreter shutdown a segment can be finalized while t.frombuffer
    views of it are still alive; the stock close() then raises BufferError.
    The pages are reclaimed by process exit either way — swallow the noise.
    """

    def __del__(self):  # pragma: no cover - shutdown-order dependent
        try:
            super().__del__()
        except BufferError:
            pass


def _shared_decode_load_owner_shard(
    shard_path: Path,
    *,
    tee_path: Path | None = None,
) -> PackedRowShard:
    """Bound the owner's packed-body read and zstd decode in a tiny subprocess.

    The deadline covers worker startup after ``Popen`` returns, opening and
    reading the packed path, zstd decode, decoded-byte transfer, and worker
    exit. Python cannot preempt the synchronous ``Popen`` exec handoff itself.
    Safetensors parsing and ``PackedRowShard`` construction stay in this
    process and are intentionally outside the storage-stage deadline.
    """
    start = time.monotonic()
    command = [sys.executable, os.fspath(_SHARED_DECODE_WORKER_PATH), os.fspath(shard_path)]
    if tee_path is not None:
        command.append(os.fspath(tee_path))
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        elapsed = time.monotonic() - start
        raise RuntimeError(
            "shared decode owner failed to launch helper for packed shard "
            f"{shard_path} after {elapsed:.3f}s"
        ) from exc

    remaining = _SHARED_DECODE_OWNER_TIMEOUT_S - (time.monotonic() - start)
    timed_out = remaining <= 0
    if not timed_out:
        try:
            decoded, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True

    if timed_out:
        elapsed = time.monotonic() - start
        cleanup_notes: list[str] = []
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except BaseException as exc:
            cleanup_notes.append(f"helper kill cleanup failed: {exc!r}")

        for pipe_name in ("stdout", "stderr"):
            pipe = getattr(process, pipe_name, None)
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except BaseException as exc:
                    cleanup_notes.append(
                        f"helper {pipe_name} pipe cleanup failed: {exc!r}"
                    )

        helper_reaped = False
        try:
            process.wait(timeout=_SHARED_DECODE_REAP_TIMEOUT_S)
            helper_reaped = True
        except subprocess.TimeoutExpired:
            pass
        except BaseException as exc:
            cleanup_notes.append(f"helper reap cleanup failed: {exc!r}")

        error = RuntimeError(
            f"shared decode owner timed out after {elapsed:.3f}s "
            f"(limit {_SHARED_DECODE_OWNER_TIMEOUT_S:.3f}s) while "
            f"reading/zstd-decoding packed shard {shard_path}; "
            f"helper_pid={process.pid}; "
            f"helper_reaped={'true' if helper_reaped else 'false'}"
        )
        for note in cleanup_notes:
            error.add_note(note)
        raise error

    storage_elapsed = time.monotonic() - start
    if process.returncode != 0:
        stderr_tail = stderr[-4096:].decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "shared decode owner helper failed while reading/zstd-decoding "
            f"packed shard {shard_path} after {storage_elapsed:.3f}s; "
            f"exit_code={process.returncode}; stderr_tail={stderr_tail!r}"
        )

    try:
        return load_packed_row_shard_safetensors_bytes(decoded)
    except Exception as exc:
        exc.add_note(
            "shared decode owner safetensors construction context: "
            f"path={shard_path}, completed_storage_elapsed={storage_elapsed:.3f}s"
        )
        raise


def _shared_decode_load_base_shard(
    shard_path: Path,
    logical_epoch: int,
    local_rank: int,
    local_world: int,
    *,
    retain: int = _SHARED_DECODE_RETAIN,
    node_cache: NodeLocalShardCache | None = None,
) -> PackedRowShard:
    shared_memory_cls = _QuietSharedMemory

    normalized_path = Path(os.path.abspath(os.fspath(shard_path)))
    key = f"{normalized_path}:{logical_epoch}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    owner = int(digest, 16) % local_world
    name = f"ngptshard_{_shared_decode_run_namespace()}_{digest[:16]}"

    if local_rank == owner:
        owner_started = time.monotonic()
        worker_path = normalized_path
        cache_fill = None
        cache_hit = False
        if node_cache is not None:
            cached_path = node_cache.read_path(normalized_path)
            if cached_path is not None:
                worker_path = cached_path
                cache_hit = True
            else:
                cache_fill = node_cache.reserve_fill_tmp(normalized_path)
        try:
            try:
                shard = _shared_decode_load_owner_shard(
                    worker_path,
                    tee_path=cache_fill.tmp_path if cache_fill is not None else None,
                )
            except Exception as exc:
                if not cache_hit:
                    raise
                assert node_cache is not None
                logger.warning(
                    "node shard cache shared-decode hit failed; retrying once from "
                    f"canonical source {normalized_path} after cache path {worker_path}: "
                    f"{exc}"
                )
                node_cache.record_verify_failure()
                node_cache.record_shared_decode_fallback()
                shard = _shared_decode_load_owner_shard(normalized_path)
        except BaseException:
            if cache_fill is not None:
                assert node_cache is not None
                node_cache.discard_fill_tmp(cache_fill)
            raise
        if cache_fill is not None:
            assert node_cache is not None
            node_cache.commit_fill_tmp(cache_fill)
        owner_elapsed = time.monotonic() - owner_started
        if owner_elapsed > _SHARED_DECODE_SLOW_WARN_S:
            logger.warning(
                f"shared decode owner slow: {owner_elapsed:.1f}s decoding "
                f"{shard_path} (warn threshold {_SHARED_DECODE_SLOW_WARN_S:.0f}s; "
                "storage degradation likely)"
            )
        _loader_health_record("owner_decode", owner_elapsed)
        try:  # crash leftovers from a previous attempt in the SAME launch
            stale = shared_memory_cls(name=name, track=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
        segment = shared_memory_cls(
            name=name,
            create=True,
            size=_shared_decode_payload_size(shard),
            track=False,
        )
        _shared_decode_publish(segment, shard)
        del shard
    else:
        wait_started = time.monotonic()
        deadline = wait_started + _SHARED_DECODE_TIMEOUT_S
        segment = None
        while segment is None:
            try:
                candidate = shared_memory_cls(name=name, track=False)
                candidate_buf = candidate.buf
                if (
                    candidate_buf is not None
                    and bytes(candidate_buf[0:16]) == _SHARED_DECODE_MAGIC
                ):
                    segment = candidate
                    continue
                candidate.close()
            except (FileNotFoundError, ValueError):
                # ValueError("cannot mmap an empty file"): the owner has
                # created the segment but not yet ftruncated it to size —
                # attach raced the open->ftruncate window. Not published yet.
                pass
            if time.monotonic() > deadline:
                # A sibling whose traversal legitimately diverges (unequal
                # stripes make per-rank resume cursors differ when
                # rows_per_shard % local participation != 0) or a stalled
                # owner would deadlock a fatal wait. Fall back to the
                # canonical direct decode — byte-identical, locally slower.
                logger.warning(
                    f"shared decode: local rank {owner} did not publish "
                    f"{name} ({shard_path}, epoch {logical_epoch}) within "
                    f"{_SHARED_DECODE_TIMEOUT_S}s; falling back to direct "
                    "decode for this shard."
                )
                direct_started = time.monotonic()
                fallback_shard = _load_packed_row_shard_with_node_cache(
                    shard_path,
                    node_cache,
                )
                if node_cache is not None:
                    node_cache.record_shared_decode_fallback()
                _loader_health_record(
                    "direct_load",
                    time.monotonic() - direct_started,
                    fallback=True,
                )
                return fallback_shard
            time.sleep(0.05)
        attach_waited = time.monotonic() - wait_started
        if attach_waited > _SHARED_DECODE_SLOW_WARN_S:
            logger.warning(
                f"shared decode attach waited {attach_waited:.1f}s for "
                f"{name} ({shard_path}) before the owner published "
                f"(warn threshold {_SHARED_DECODE_SLOW_WARN_S:.0f}s; "
                "storage degradation likely)"
            )
        _loader_health_record("shared_attach", attach_waited)

    _shared_decode_segments[name] = (segment, local_rank == owner)
    _shared_decode_evict_old_segments(retain)
    return _shared_decode_wrap(segment)


def _load_packed_row_shard_with_node_cache(
    shard_path: Path,
    node_cache: NodeLocalShardCache | None,
) -> PackedRowShard:
    if not shard_path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Expected {COMPRESSED_EXT} file, got: {shard_path.name}")
    if node_cache is None:
        return load_packed_row_shard(shard_path)
    compressed = node_cache.read(shard_path)
    if compressed is None:
        source_stat = os.stat(shard_path)
        compressed = shard_path.read_bytes()
        node_cache.offer(shard_path, compressed, source_stat)
    return load_packed_row_shard_bytes(compressed)


def _load_selection_shard_with_node_cache(
    shard_path: Path,
    node_cache: NodeLocalShardCache | None,
) -> Any:
    if not shard_path.name.endswith(COMPRESSED_EXT):
        raise ValueError(f"Expected {COMPRESSED_EXT} file, got: {shard_path.name}")
    if node_cache is None:
        return load_selection_shard(shard_path)
    compressed = node_cache.read(shard_path)
    if compressed is None:
        source_stat = os.stat(shard_path)
        compressed = shard_path.read_bytes()
        node_cache.offer(shard_path, compressed, source_stat)
    return load_selection_shard_bytes(compressed)


@beartype
def resolve_train_shard_lookahead(raw: object) -> NonNegativeInt:
    """Resolve decoded-shard lookahead depth at the config boundary."""
    if raw is None:
        return NonNegativeInt.parse(0)
    return NonNegativeInt.parse(raw)


def _log_shard_lookahead_budget(depth: int, message: str) -> None:
    if depth >= _SHARD_LOOKAHEAD_EXCEPTION_DEPTH:
        logger.exception(message)
    elif depth >= _SHARD_LOOKAHEAD_ERROR_DEPTH:
        logger.error(message)
    elif depth >= _SHARD_LOOKAHEAD_WARN_DEPTH:
        logger.warning(message)
    else:
        logger.info(message)


def _assert_shard_lookahead_supported(
    depth: int,
    packed_materializer: object | None,
    remote_materializer: object | None,
) -> None:
    if depth > 0 and (packed_materializer is not None or remote_materializer is not None):
        raise ValueError(
            "train_shard_lookahead is supported only on the local packed-row path; "
            "remote materializer paths already stage bodies through "
            "remote_training.body_prefetch, so set train_shard_lookahead=0"
        )


@dataclass(frozen=True)
class _ShardVisit:
    shard_path: Path
    epoch: int
    first_row_idx: int
    shard_batch_idx: int
    skip_batches: int


def _precomputed_shard_visits(
    shard_paths: list[Path],
    *,
    start_epoch: int,
    resume_shard: str | None,
    resume_shard_batch_offset: int,
    exact_continuation_shard: str | None,
    exact_continuation_after_row: int,
    exact_continuation_batch_offset: int,
    log_epoch_starts: bool,
) -> Iterator[_ShardVisit]:
    epoch = start_epoch
    while True:  # epoch cycling
        for shard_path in shard_paths:
            # Each branch owns its own outputs; both can match at most once
            # because matching clears the cursor that selected the branch.
            first_row_idx = 0
            shard_batch_idx = 0
            skip_batches = 0
            if exact_continuation_shard is not None:
                if shard_path.name != exact_continuation_shard:
                    continue
                exact_continuation_shard = None
                first_row_idx = exact_continuation_after_row + 1
                shard_batch_idx = exact_continuation_batch_offset
            elif resume_shard is not None:
                if shard_path.name != resume_shard:
                    continue
                resume_shard = None
                skip_batches = resume_shard_batch_offset
            yield _ShardVisit(
                shard_path=shard_path,
                epoch=epoch,
                first_row_idx=first_row_idx,
                shard_batch_idx=shard_batch_idx,
                skip_batches=skip_batches,
            )

        # Preserve the historical unknown-shard fallback for legacy states.
        if resume_shard is not None:
            logger.warning(
                f"PrecomputedBatchLoader: resume shard {resume_shard!r} not found "
                f"in epoch {epoch}. Starting fresh from epoch {epoch + 1}."
            )
            resume_shard = None
        epoch += 1
        if log_epoch_starts:
            logger.info(f"PrecomputedBatchLoader: starting epoch {epoch}")


def _background_lookahead(inner: Iterator[Any], depth: int) -> Generator[Any]:
    if depth <= 0:
        raise ValueError(f"shard lookahead depth must be positive, got {depth!r}")
    ready: queue.Queue[object] = queue.Queue(maxsize=depth)
    stop = threading.Event()
    sentinel = object()

    def put(value: object) -> bool:
        while not stop.is_set():
            try:
                ready.put(value, timeout=_SHARD_LOOKAHEAD_POLL_SECONDS)
            except queue.Full:
                continue
            return True
        return False

    def produce() -> None:
        try:
            for item in inner:
                if not put(item):
                    return
        except BaseException as exc:
            put(exc)
        finally:
            put(sentinel)

    thread = threading.Thread(target=produce, name="packed-shard-lookahead", daemon=True)
    thread.start()
    try:
        while True:
            item = ready.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while True:
            try:
                ready.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=_SHARD_LOOKAHEAD_JOIN_SECONDS)
        if thread.is_alive():
            logger.warning(
                "packed-shard-lookahead thread did not stop within "
                f"{_SHARD_LOOKAHEAD_JOIN_SECONDS:.1f}s; leaving daemon thread to exit"
            )


@beartype
def resolve_train_prefetch_batches(
    raw: object,
    grad_accum_steps: PositiveInt,
) -> NonNegativeInt:
    """Resolve a bounded training-batch FIFO depth at the config boundary.

    ``None`` preserves the historical depth of one optimizer step. An explicit
    non-negative integer overrides it, with zero selecting the synchronous
    loader path.
    """
    if raw is None:
        return NonNegativeInt.parse(grad_accum_steps)
    return NonNegativeInt.parse(raw)


CLOSABLE_BATCH_PREFETCH_JOIN_SECONDS = 5.0


class ClosableBatchPrefetcher:
    """Explicitly closable CPU/device batch producer for remote-body training."""

    @beartype
    def __init__(
        self,
        inner_gen,
        prefetch_batches: PositiveInt,
        *,
        stop_callback,
        failure_check,
    ) -> None:
        self._inner = inner_gen
        self._queue: queue.Queue[object] = queue.Queue(maxsize=prefetch_batches)
        self._sentinel = object()
        self._stop = threading.Event()
        self._stop_callback = stop_callback
        self._failure_check = failure_check
        self._error: BaseException | None = None
        self._closed = False
        self._close_lock = threading.Lock()
        self._stop_callback_called = False
        self._inner_close_attempted = False
        self._thread = threading.Thread(
            target=self._producer,
            name="remote-batch-prefetch",
            daemon=False,
        )
        self._thread.start()

    def _put(self, value: object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(value, timeout=0.05)
            except queue.Full:
                continue
            return True
        return False

    def _producer(self) -> None:
        try:
            for item in self._inner:
                if self._stop.is_set() or not self._put(item):
                    return
        except BaseException as exc:
            self._error = exc
        finally:
            self._put(self._sentinel)

    def __iter__(self) -> "ClosableBatchPrefetcher":
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        self._failure_check()
        while True:
            try:
                queued = self._queue.get(timeout=0.05)
            except queue.Empty:
                self._failure_check()
                if not self._thread.is_alive():
                    if self._error is not None:
                        raise self._error from None
                    raise StopIteration from None
                continue
            if queued is self._sentinel:
                if self._error is not None:
                    raise self._error
                raise StopIteration
            self._failure_check()
            return queued

    def close(self) -> None:
        with self._close_lock:
            if self._closed and not self._thread.is_alive():
                return
            self._closed = True
            self._stop.set()
            failures: list[BaseException] = []
            if not self._stop_callback_called:
                self._stop_callback_called = True
                try:
                    self._stop_callback()
                except BaseException as exc:
                    failures.append(exc)
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._thread.join(timeout=CLOSABLE_BATCH_PREFETCH_JOIN_SECONDS)
            if self._thread.is_alive():
                failures.append(
                    RuntimeError(
                        "remote batch prefetch producer did not stop within "
                        f"{CLOSABLE_BATCH_PREFETCH_JOIN_SECONDS:g} seconds"
                    )
                )
            elif not self._inner_close_attempted:
                self._inner_close_attempted = True
                close = getattr(self._inner, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        failures.append(exc)
            if self._error is not None and not isinstance(
                self._error,
                RemoteMaterializationCancelled,
            ):
                failures.append(self._error)
            if failures:
                primary = failures[0]
                for failure in failures[1:]:
                    primary.add_note(
                        "remote batch prefetch cleanup also failed: "
                        f"{type(failure).__name__}: {failure}"
                    )
                raise primary

    def is_terminal(self) -> bool:
        """Return whether the owned non-daemon producer has definitely exited."""
        return not self._thread.is_alive()


def _assert_pipeline_supports_live_selection(
    pipeline: "FeaturePipelineConfig", context: str
) -> None:
    """Gradient-attribution ranking is offline-only: any loader that performs
    LIVE selection must reject a gradient-configured pipeline up front (the
    central guard in sampling.py would otherwise fire mid-stream)."""
    selection = getattr(pipeline.selection_config, "selection", None)
    if selection is not None and getattr(selection, "gradient", None) is not None:
        raise GradientRankingUnsupportedError(
            f"{context} cannot rank by gradient attribution "
            "(feature_selection.gradient): gradient ranking is offline-only. "
            "Precompute sel/ shards via "
            "scripts.preprocess.precompute_selected_features and train with "
            "use_precomputed_batches=true."
        )


@beartype
def encoder_decoder_data_loader_with_state(
    tokenizer,
    B: int,
    T: int,
    split: DatasetSplit,
    feature_index: FeatureIndex,
    pipeline: FeaturePipelineConfig,
    segment_size: PositiveIntLike = 128,
    decoder_segment_size: int | None = None,
    device: str = "cuda",
    resume_state_dict: dict | None = None,
    tokenizer_batch_size: int = 128,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    data_dir: str | None = None,
    prefetch_batches: int = 0,
    label_dropout: float = 0.0,
    max_shards: int | None = None,
    n_sae_features: int | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    context_len: int | None = None,
    pad_F: int | None = None,
    pad_n_seg: int | None = None,
    doc_prefetch_depth: int = 0,
    next_segment_conditioning_offset: bool = False,
    include_splice_mask: bool = False,
):
    """BOS-bestfit dataloader with encoder-decoder feature conditioning.

    Mirrors tokenizing_distributed_data_loader_with_state_bos_bestfit from
    nanochat/dataloader.py but additionally yields encoder inputs for
    cross-attention conditioning.

    Segment assignment mode:
    - Always char-level overlap assignment (stride mode has been removed).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS (inherited from source documents)
    - 100% utilization (no padding, every position filled)
    - Encoder inputs are segment-aligned with the text
    - Resume is best-effort/approximate, never byte-exact: cursor-aligned RNG
      state is restored, but SortedDocBuffer contents, partially consumed
      row-group chunks, and prefetch queues are not serialized

    Args:
        segment_size: Strict positive extraction segment size. The training
            entry point parses metadata into ``PositiveInt`` before loader
            construction.
        prefetch_batches: When > 0, wraps the generator with a background-thread
            prefetch buffer of the given depth, hiding I/O latency behind GPU compute.
        n_sae_features: When set, builds dense fp_targets from all pre-selection
            features. Passed through to the inner function.
        context_len: When set, passed to _document_batches_with_features for
            boundary truncation. If None, defaults to T+1 (row_capacity).
        doc_prefetch_depth: When > 0, runs document batch production in a
            background thread, buffering up to this many batches ahead.
            Each batch holds ~tokenizer_batch_size documents.

    Yields:
        (inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict)
        - inputs: (B, T) text token IDs
        - targets: (B, T) shifted text tokens
        - enc_fids: (B, n_seg, F_max) feature IDs per segment
        - enc_mags: (B, n_seg, F_max) magnitude bucket indices
        - enc_mask: (B, n_seg, F_max) bool mask
        - enc_pcts: (B, n_seg, F_max) percentile values (float32)
        - seg_map: (B, T) int64 mapping each position to segment index
        - seg_doc_id: (B, n_seg) int64 document index per segment (-1 for padding)
        - fp_targets: (B, n_seg, n_sae_features) or None
        - state_dict: JSON-safe approximate-resume cursor, topology, and RNG state

    Raises:
        BeartypeCallHintParamViolation: An argument violates its annotated
            contract, including a non-positive ``segment_size``.
    """
    parsed_segment_size = PositiveInt.parse(segment_size)
    if fp_target_layout is not None and n_sae_features is not None:
        raise ValueError(
            "encoder_decoder_data_loader_with_state accepts fp_target_layout or "
            "the legacy n_sae_features argument, not both"
        )
    if fp_target_layout is not None and fp_target_layout.is_compact:
        raise ValueError(
            "compact feature-prediction targets require precomputed packed rows. "
            "Run scripts.preprocess.precompute_packed_rows and train with "
            "use_precomputed_batches=true."
        )
    if fp_target_layout is not None and (fp_target_layout.feature_mask != pipeline.feature_mask):
        raise ValueError(
            "live dataloader received an FP target layout derived from a "
            "different FeatureMask than the feature pipeline"
        )

    _assert_pipeline_supports_live_selection(
        pipeline, "encoder_decoder_data_loader_with_state (live dataloader)"
    )
    # Cross-stage segment_size check (Wave 3D task 4).
    fdir = feature_index.feature_dir
    if fdir is not None:
        verify_segment_size(fdir, expected_segment_size=int(parsed_segment_size))

    if prefetch_batches > 0:
        # Thread-based prefetch: runs inner loader in a background thread,
        # buffering prefetch_batches complete batches.
        inner_gen = _encoder_decoder_data_loader_inner(
            tokenizer,
            B,
            T,
            split,
            feature_index,
            pipeline,
            parsed_segment_size,
            decoder_segment_size,
            device,
            resume_state_dict,
            tokenizer_batch_size,
            buffer_size,
            data_dir,
            prefetch_batches,
            label_dropout,
            max_shards=max_shards,
            n_sae_features=n_sae_features,
            fp_target_layout=fp_target_layout,
            context_len=context_len,
            pad_F=pad_F,
            pad_n_seg=pad_n_seg,
            doc_prefetch_depth=doc_prefetch_depth,
            next_segment_conditioning_offset=next_segment_conditioning_offset,
            include_splice_mask=include_splice_mask,
        )
        use_cuda_prefetch = t.device(device).type == "cuda" and t.cuda.is_available()
        yield from _prefetch_wrapper(
            inner_gen,
            prefetch_batches,
            use_cuda=use_cuda_prefetch,
        )
    else:
        gen = _encoder_decoder_data_loader_inner(
            tokenizer,
            B,
            T,
            split,
            feature_index,
            pipeline,
            parsed_segment_size,
            decoder_segment_size,
            device,
            resume_state_dict,
            tokenizer_batch_size,
            buffer_size,
            data_dir,
            prefetch_batches,
            label_dropout,
            max_shards=max_shards,
            n_sae_features=n_sae_features,
            fp_target_layout=fp_target_layout,
            context_len=context_len,
            pad_F=pad_F,
            pad_n_seg=pad_n_seg,
            doc_prefetch_depth=doc_prefetch_depth,
            next_segment_conditioning_offset=next_segment_conditioning_offset,
            include_splice_mask=include_splice_mask,
        )
        yield from gen


@beartype
def _encoder_decoder_data_loader_inner(
    tokenizer,
    B,
    T,
    split,
    feature_index,
    pipeline: FeaturePipelineConfig,
    segment_size: PositiveInt,
    decoder_segment_size,
    device,
    resume_state_dict,
    tokenizer_batch_size,
    buffer_size,
    data_dir,
    prefetch_batches,
    label_dropout=0.0,
    max_shards: int | None = None,
    n_sae_features: int | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    context_len: int | None = None,
    pad_F: int | None = None,
    pad_n_seg: int | None = None,
    doc_prefetch_depth: int = 0,
    next_segment_conditioning_offset: bool = False,
    include_splice_mask: bool = False,
):
    assert B > 0, f"B must be positive, got {B}"
    assert T > 0, f"T must be positive, got {T}"
    assert 0.0 <= label_dropout <= 1.0, f"label_dropout must be in [0, 1], got {label_dropout}"

    assert decoder_segment_size is None, (
        "Stride-based segment assignment has been removed. "
        "Use decoder_segment_size=None (char-level) only."
    )

    row_capacity = T + 1
    effective_context_len = context_len if context_len is not None else row_capacity

    batches = _document_batches_with_features(
        split,
        resume_state_dict,
        tokenizer_batch_size,
        tokenizer,
        feature_index,
        pipeline,
        segment_size,
        data_dir,
        max_shards=max_shards,
        context_len=effective_context_len,
        **({"include_splice_mask": True} if include_splice_mask else {}),
    )

    # Buffer stores (text_tokens, seg_features_or_None) per document
    doc_buffer = SortedDocBuffer()
    pq_idx, rg_idx, epoch = 0, 0, 1
    _ddp, ddp_rank, _ddp_local_rank, ddp_world_size = get_dist_info()
    _validate_positive_world_size(ddp_rank, ddp_world_size)
    document_cursor_state: dict[str, Any] = {
        "pq_idx": pq_idx,
        "rg_idx": rg_idx,
        "epoch": epoch,
        "ddp_world_size": ddp_world_size,
        "feature_pipeline_rng_state": _capture_feature_pipeline_rng_state(pipeline),
    }

    has_resume_label_rng_state = (
        resume_state_dict is not None and "label_dropout_rng_state" in resume_state_dict
    )
    resume_label_rng_state = (
        resume_state_dict["label_dropout_rng_state"] if has_resume_label_rng_state else None
    )
    if has_resume_label_rng_state:
        # Validate eagerly, but mutate the default generator only immediately
        # before the first resumed dropout draw. In particular label_dropout=0
        # remains a complete default-RNG no-op.
        _restore_default_rng_state(
            resume_label_rng_state,
            "cpu",
            context="live dataloader label-dropout resume",
            apply=False,
        )
    elif resume_state_dict is not None and label_dropout > 0.0:
        logger.warning(
            "Live dataloader: legacy resume state has no "
            "label_dropout_rng_state; continuing with the current default "
            "generator. Live resume remains approximate."
        )

    def append_document_batch(item: _LiveDocumentBatch) -> None:
        nonlocal pq_idx, rg_idx, epoch, document_cursor_state
        doc_list, cursor_state = item
        document_cursor_state = dict(cursor_state)
        pq_idx = int(document_cursor_state["pq_idx"])
        rg_idx = int(document_cursor_state["rg_idx"])
        epoch = int(document_cursor_state["epoch"])
        for text_tokens, seg_features in doc_list:
            doc_buffer.append((text_tokens, seg_features))

    if doc_prefetch_depth > 0:
        # Async path: background thread runs the document generator,
        # feeding batches into a bounded queue.  The main thread's
        # refill_buffer() reads from the queue instead of blocking on I/O.
        _doc_queue: queue.Queue = queue.Queue(maxsize=doc_prefetch_depth)
        _doc_sentinel = object()
        _doc_error_box: list[BaseException | None] = [None]

        def _doc_producer():
            try:
                for batch in batches:
                    _doc_queue.put(batch)
            except Exception as e:
                _doc_error_box[0] = e
            finally:
                _doc_queue.put(_doc_sentinel)

        _doc_thread = threading.Thread(target=_doc_producer, daemon=True, name="doc-prefetch")
        _doc_thread.start()

        def refill_buffer():
            item = _doc_queue.get()
            if item is _doc_sentinel:
                if _doc_error_box[0] is not None:
                    raise _doc_error_box[0]
                # Use RuntimeError, not StopIteration — PEP 479 would convert
                # it to RuntimeError anyway since we're inside a generator.
                raise RuntimeError("document generator exhausted")
            append_document_batch(item)
    else:

        def refill_buffer():
            append_document_batch(next(batches))

    use_cuda = t.device(device).type == "cuda"

    # Pre-allocate row buffer for text packing
    row_buffer = t.empty((B, row_capacity), dtype=t.long)

    # Staging area (CPU, optionally pinned) and device buffers for text
    cpu_buffer = t.empty(2 * B * T, dtype=t.long, pin_memory=use_cuda)
    gpu_buffer = t.empty(2 * B * T, dtype=t.long, device=device)

    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        # Track which documents go into each row and their positions
        row_doc_info: list[list] = [[] for _ in range(B)]

        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    with t.profiler.record_function("live/refill_buffer"):
                        refill_buffer()

                remaining = row_capacity - pos

                # Best-fit: O(log n) lookup for largest doc that fits
                result = doc_buffer.pop_best_fit(remaining)

                if result is not None:
                    doc_tokens = result[0]
                    seg_features = result[1]
                    doc_len = len(doc_tokens)
                    row_buffer[row_idx, pos : pos + doc_len] = t.tensor(doc_tokens, dtype=t.long)
                    row_doc_info[row_idx].append((pos, seg_features, doc_tokens))
                    pos += doc_len
                else:
                    # No doc fits: crop the shortest doc to fill remaining space
                    shortest = doc_buffer.pop_shortest()
                    doc_tokens = shortest[0]
                    seg_features = shortest[1]
                    row_buffer[row_idx, pos : pos + remaining] = t.tensor(
                        doc_tokens[:remaining], dtype=t.long
                    )
                    row_doc_info[row_idx].append((pos, seg_features, doc_tokens[:remaining]))
                    pos += remaining

        # Build text inputs/targets
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Build encoder inputs and seg_map for each row
        row_enc_fids: list[t.Tensor] = []
        row_enc_mags: list[t.Tensor] = []
        row_enc_pcts: list[t.Tensor] = []
        row_enc_masks: list[t.Tensor] = []
        row_seg_maps: list[t.Tensor] = []
        row_seg_doc_ids: list[t.Tensor] = []
        row_all_fids: list[list[t.Tensor]] = []
        row_all_pcts: list[list[t.Tensor]] = []

        for row_idx in range(B):
            with t.profiler.record_function("live/build_row_segments"):
                (
                    per_seg_fids,
                    per_seg_bkts,
                    per_seg_pcts,
                    seg_map_row,
                    seg_doc_id_row,
                    per_seg_all_fids,
                    per_seg_all_pcts,
                ) = _build_row_segments(
                    row_doc_info[row_idx],
                    T,
                    tokenizer,
                    next_segment_conditioning_offset=next_segment_conditioning_offset,
                )
                fids, mags, pcts, mask, _ = _pad_segments(
                    per_seg_fids, per_seg_bkts, per_seg_pcts, seg_map_row
                )
            row_all_fids.append(per_seg_all_fids)
            row_all_pcts.append(per_seg_all_pcts)

            row_enc_fids.append(fids)
            row_enc_mags.append(mags)
            row_enc_pcts.append(pcts)
            row_enc_masks.append(mask)
            row_seg_maps.append(seg_map_row)
            row_seg_doc_ids.append(seg_doc_id_row)

        # n_seg = max segments across rows (may vary due to document packing)
        batch_n_seg = max(f.shape[0] for f in row_enc_fids)

        # Pad F_max across all rows in the batch (minimum 1 to avoid 0-dim tensors
        # that crash SDPA -- the padding mask handles masking out the dummy slot).
        batch_f_max = max(1, max(fids.shape[1] for fids in row_enc_fids) if row_enc_fids else 0)

        # Static shape override for torch.compile(dynamic=False).
        #
        # pad_n_seg / pad_F are heuristic capacity bounds (e.g.
        # resolve_live_pad_n_seg_from_training_cfg derives pad_n_seg from a 1.5x
        # guess), NOT invariants: per-batch counts are data-dependent (BOS-bestfit
        # packs a variable number of documents/segments per row). On overflow we
        # RAISE rather than truncate. Truncating drops the segments beyond the
        # bound, and the seg_map clamp below (`seg_map.clamp(max=pad_n_seg-1)`)
        # then silently remaps those segments' decoder tokens onto the last
        # retained segment — conditioning them on a DIFFERENT document's features
        # (mis-conditioning; the cross-attention `assert seg_map.max() < n_seg`
        # makes the clamp load-bearing, so we cannot simply drop it). Failing loud
        # is a STOPGAP: it trades silent training-data corruption for a crash
        # (which hangs other DDP ranks). Graceful overflow handling without
        # mis-conditioning is tracked in issue #641. On real FineWeb-Edu base data
        # this never fires (worst row 12 segs vs bound 14); the risk surface is
        # short-document corpora at segment_size=128.
        if pad_n_seg is not None:
            if batch_n_seg > pad_n_seg:
                raise ValueError(
                    f"Live dataloader: batch_n_seg={batch_n_seg} exceeds "
                    f"pad_n_seg={pad_n_seg}. Refusing to truncate: dropping segments "
                    "would mis-condition their decoder tokens via the seg_map clamp "
                    "(issue #641). Increase pad_n_seg "
                    "(resolve_live_pad_n_seg_from_training_cfg) or handle overflow "
                    "gracefully."
                )
            batch_n_seg = pad_n_seg

        if pad_F is not None:
            if batch_f_max > pad_F:
                raise ValueError(
                    f"Live dataloader: batch_f_max={batch_f_max} exceeds "
                    f"pad_F={pad_F}. Refusing to truncate features: dropping "
                    "conditioning features silently corrupts the batch (issue #641). "
                    "Increase pad_F or handle overflow gracefully."
                )
            batch_f_max = pad_F

        enc_fids = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.int64)
        enc_mags = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.int64)
        enc_pcts = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.float32)
        enc_mask = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.bool)

        for row_idx in range(B):
            n_row_seg = min(row_enc_fids[row_idx].shape[0], batch_n_seg)
            f_row = min(row_enc_fids[row_idx].shape[1], batch_f_max)
            if f_row > 0 and n_row_seg > 0:
                enc_fids[row_idx, :n_row_seg, :f_row] = row_enc_fids[row_idx][:n_row_seg, :f_row]
                enc_mags[row_idx, :n_row_seg, :f_row] = row_enc_mags[row_idx][:n_row_seg, :f_row]
                enc_pcts[row_idx, :n_row_seg, :f_row] = row_enc_pcts[row_idx][:n_row_seg, :f_row]
                enc_mask[row_idx, :n_row_seg, :f_row] = row_enc_masks[row_idx][:n_row_seg, :f_row]

        # Per-segment label dropout: randomly mask entire segments
        label_rng_state: dict[str, Any] | None = None
        if label_dropout > 0.0:
            if resume_label_rng_state is not None:
                _restore_default_rng_state(
                    resume_label_rng_state,
                    enc_mask.device,
                    context="live dataloader label-dropout resume",
                )
                resume_label_rng_state = None
            label_rng_state = _capture_default_rng_state(enc_mask.device)
            drop_mask = t.rand(B, batch_n_seg, device=enc_mask.device) < label_dropout
            enc_mask[drop_mask] = False

        # Prevent fully-empty rows that cause NaN in cross-attention.
        # When feature_mask is set, leaves empty rows all-False (I4-safe, NaN
        # handled by cross-attn nan_to_num).  No-mask path: byte-identical (I5).
        _safe_empty_row_fallback(enc_mask, pipeline.feature_mask)

        seg_map = t.stack(row_seg_maps)
        if pad_n_seg is not None:
            seg_map = seg_map.clamp(max=pad_n_seg - 1)

        seg_doc_id = t.full((B, batch_n_seg), -1, dtype=t.int64)
        for row_idx in range(B):
            n_row_seg = min(row_seg_doc_ids[row_idx].shape[0], batch_n_seg)
            seg_doc_id[row_idx, :n_row_seg] = row_seg_doc_ids[row_idx][:n_row_seg]

        # Build dense fp_targets from all pre-selection features
        fp_targets: t.Tensor | None = None
        if n_sae_features is not None or fp_target_layout is not None:
            # Compact layouts are rejected by the public live-loader boundary.
            live_width = (
                n_sae_features if fp_target_layout is None else fp_target_layout.n_sae_features
            )
            assert live_width is not None
            batch_fp = t.zeros(B, batch_n_seg, live_width, dtype=t.float32)
            for row_idx in range(B):
                for seg_i, (all_fids, all_pcts) in enumerate(
                    zip(  # noqa: B905
                        row_all_fids[row_idx],
                        row_all_pcts[row_idx],
                    )
                ):
                    if seg_i >= batch_n_seg:
                        break
                    if all_pcts is not None and len(all_fids) > 0:
                        batch_fp[row_idx, seg_i, all_fids.long()] = all_pcts
            # Structural parity with build_fp_targets_from_shard: same function,
            # same mechanism — live/fused divergence is now impossible.
            fp_targets = _zero_masked_fp_targets(batch_fp, pipeline.feature_mask, live_width)

        state_dict = dict(document_cursor_state)
        if include_splice_mask:
            splice_rows: list[list[bool]] = []
            for row_idx in range(B):
                ordinals: dict[int, int] = {}
                flags: list[bool] = []
                for doc_id in row_seg_doc_ids[row_idx].tolist():
                    ordinal = ordinals.get(doc_id, 0)
                    ordinals[doc_id] = ordinal + 1
                    segments = row_doc_info[row_idx][doc_id][1]
                    source_ordinal = ordinal + int(next_segment_conditioning_offset)
                    flags.append(
                        segments is not None
                        and source_ordinal < len(segments)
                        and segments[source_ordinal].is_splice
                    )
                splice_rows.append(flags + [False] * (batch_n_seg - len(flags)))
            state_dict["conditioning_splice_mask"] = splice_rows
        if label_rng_state is not None:
            state_dict["label_dropout_rng_state"] = label_rng_state

        # Bulk pin + async HtoD transfer for encoder tensors
        if use_cuda:
            enc_fids = _pin_transfer(enc_fids, device)
            enc_mags = _pin_transfer(enc_mags, device)
            enc_pcts = _pin_transfer(enc_pcts, device)
            enc_mask = _pin_transfer(enc_mask, device)
            seg_map = _pin_transfer(seg_map, device)
            seg_doc_id = _pin_transfer(seg_doc_id, device)
            if fp_targets is not None:
                fp_targets = _pin_transfer(fp_targets, device)

        # Single HtoD transfer for text
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)

        if prefetch_batches > 0:
            # When prefetching, the producer thread will overwrite the
            # pre-allocated buffers while preparing the next batch. Clone
            # the text tensors so the consumer sees stable data.
            yield (
                inputs.clone(),
                targets.clone(),
                enc_fids,
                enc_mags,
                enc_mask,
                enc_pcts,
                seg_map,
                seg_doc_id,
                fp_targets,
                state_dict,
            )
        else:
            yield (
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                enc_pcts,
                seg_map,
                seg_doc_id,
                fp_targets,
                state_dict,
            )


# ---------------------------------------------------------------------------
# Dense feature prediction targets from CSR shard data
# ---------------------------------------------------------------------------


def build_fp_targets_from_shard(
    shard: PackedRowShard,
    idx: _BatchIndex,
    n_sae_features: int | None = None,
    device: str = "cpu",
    *,
    feature_mask: "FeatureMask | None" = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
) -> t.Tensor:
    """Build dense feature prediction targets from a shard's CSR extraction features.

    Scatters ALL extracted feature percentiles (~4000 per segment) into a dense
    [B, n_seg, n_sae_features] tensor. This gives the model the full feature
    distribution to predict, not just the ~15 selected features.

    Args:
        shard: Packed row shard with CSR feature data.
        idx: Pre-built batch index (from _build_batch_index).
        n_sae_features: Total SAE feature vocabulary size.
        device: Target device for the output tensor.
        feature_mask: When set, zeroes out masked feature columns in the returned targets (None ⇒ byte-identical legacy output, invariant I5).

    Returns:
        [B, n_seg, n_sae_features] float32 dense target tensor.
    """
    if fp_target_layout is not None and n_sae_features is not None:
        raise ValueError(
            "build_fp_targets_from_shard accepts fp_target_layout or the legacy "
            "n_sae_features argument, not both"
        )
    if fp_target_layout is None and n_sae_features is None:
        raise ValueError("build_fp_targets_from_shard requires fp_target_layout or n_sae_features")
    if fp_target_layout is not None and feature_mask is not None:
        raise ValueError("feature_mask is carried by fp_target_layout; do not pass it separately")

    source_width = (
        fp_target_layout.n_sae_features
        if fp_target_layout is not None
        else cast(int, n_sae_features)
    )
    total_segs = idx.total_segs
    feat_counts = t.tensor([], dtype=t.long)
    all_fids = t.tensor([], dtype=t.int64)
    all_pcts = t.tensor([], dtype=t.float32)

    # total_segs == 0 is accepted for schema-compatible empty packed rows
    # (defensive/back-compat path); the current producer normally emits at
    # least one empty segment per row. total_feats == 0 is expected when the row
    # has segment slots but no active SAE features to scatter.
    if total_segs > 0:
        feat_offsets_t = shard.seg_feat_offsets.long()
        feat_starts = feat_offsets_t[idx.global_flat]
        feat_ends = feat_offsets_t[idx.global_flat + 1]
        feat_counts = feat_ends - feat_starts
        total_feats = int(feat_counts.sum().item())

        if total_feats > 0:
            cum_fc = t.cat([t.zeros(1, dtype=t.long), feat_counts.cumsum(0)[:-1]])
            gather_idx = t.arange(total_feats) + t.repeat_interleave(feat_starts - cum_fc, feat_counts)

            all_fids = shard.feature_ids[gather_idx].to(t.int64)
            all_pcts = shard.feature_pcts[gather_idx].float() * (1.0 / 255.0)

            if fp_target_layout is not None and fp_target_layout.is_compact:
                assert all_fids.min().item() >= 0, (
                    f"Feature ID {all_fids.min().item()} is negative — corrupt shard?"
                )
            assert all_fids.max().item() < source_width, (
                f"Feature ID {all_fids.max().item()} >= n_sae_features {source_width} — corrupt shard?"
            )

    fp_targets = _build_fp_targets_from_flat_features(
        idx,
        all_fids,
        all_pcts,
        feat_counts,
        n_sae_features,
        feature_mask=feature_mask,
        fp_target_layout=fp_target_layout,
    )

    return _pin_transfer(fp_targets, device)


def _build_fp_targets_from_flat_features(
    idx: _BatchIndex,
    all_fids: t.Tensor,
    all_pcts: t.Tensor,
    segment_sizes: list[int] | t.Tensor,
    n_sae_features: int | None = None,
    *,
    feature_mask: "FeatureMask | None" = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
) -> t.Tensor:
    """Scatter raw CSR percentiles into a full or compact FP target.

    The ``n_sae_features``/``feature_mask`` pair is the frozen legacy test and
    compatibility surface. Production passes ``fp_target_layout``. Full layouts
    execute the literal legacy allocation, indexed assignment, and mask-zeroing
    branch below; compact layouts never allocate an original-width dense tensor.
    """
    if fp_target_layout is not None and n_sae_features is not None:
        raise ValueError(
            "_build_fp_targets_from_flat_features accepts fp_target_layout or "
            "the legacy n_sae_features argument, not both"
        )
    if fp_target_layout is None and n_sae_features is None:
        raise ValueError(
            "_build_fp_targets_from_flat_features requires fp_target_layout or n_sae_features"
        )
    if fp_target_layout is not None and feature_mask is not None:
        raise ValueError("feature_mask is carried by fp_target_layout; do not pass it separately")

    if fp_target_layout is None or not fp_target_layout.is_compact:
        legacy_width = (
            cast(int, n_sae_features)
            if fp_target_layout is None
            else fp_target_layout.n_sae_features
        )
        legacy_mask = feature_mask if fp_target_layout is None else fp_target_layout.feature_mask
        fp_targets = t.zeros(idx.B, idx.batch_n_seg, legacy_width, dtype=t.float32)
    else:
        legacy_width = None
        legacy_mask = None
        fp_targets = t.zeros(
            idx.B,
            idx.batch_n_seg,
            fp_target_layout.output_width,
            dtype=t.float32,
        )

    if idx.total_segs == 0:
        assert int(all_fids.numel()) == 0, (
            f"idx.total_segs=0 but all_fids has {all_fids.numel()} entries"
        )
        assert int(all_pcts.numel()) == 0, (
            f"idx.total_segs=0 but all_pcts has {all_pcts.numel()} entries"
        )
        if isinstance(segment_sizes, t.Tensor):
            assert int(segment_sizes.numel()) == 0, (
                f"idx.total_segs=0 but segment_sizes has {segment_sizes.numel()} entries"
            )
        else:
            assert len(segment_sizes) == 0, (
                f"idx.total_segs=0 but segment_sizes has {len(segment_sizes)} entries"
            )

    if idx.total_segs > 0:
        if isinstance(segment_sizes, t.Tensor):
            sizes_t = segment_sizes.to(dtype=t.long)
        else:
            sizes_t = t.tensor(segment_sizes, dtype=t.long)
        assert sizes_t.numel() == idx.total_segs, (
            f"segment_sizes has {sizes_t.numel()} entries, expected {idx.total_segs}"
        )
        total_feats = int(sizes_t.sum().item())
        assert total_feats == int(all_fids.numel()), (
            f"segment_sizes sum={total_feats} != all_fids.numel()={all_fids.numel()}"
        )
        assert total_feats == int(all_pcts.numel()), (
            f"segment_sizes sum={total_feats} != all_pcts.numel()={all_pcts.numel()}"
        )
        # total_feats == 0 is expected when the batch has segment slots but no
        # active extraction features to scatter into the dense FP target.
        if total_feats > 0:
            all_fids = all_fids.to(t.int64)
            source_width = (
                cast(int, legacy_width)
                if fp_target_layout is None or not fp_target_layout.is_compact
                else fp_target_layout.n_sae_features
            )
            if fp_target_layout is not None and fp_target_layout.is_compact:
                assert all_fids.min().item() >= 0, (
                    f"Feature ID {all_fids.min().item()} is negative — corrupt shard?"
                )
            assert all_fids.max().item() < source_width, (
                f"Feature ID {all_fids.max().item()} >= n_sae_features {source_width} — corrupt shard?"
            )
            flat_seg_slots = idx.b_flat * idx.batch_n_seg + idx.local_flat
            feature_seg_slots = t.repeat_interleave(flat_seg_slots, sizes_t)
            if fp_target_layout is None or not fp_target_layout.is_compact:
                assert legacy_width is not None
                fp_targets.view(-1, legacy_width)[feature_seg_slots, all_fids] = all_pcts.float()
            else:
                original_to_output = fp_target_layout.original_to_output_row
                assert original_to_output is not None
                output_rows = original_to_output[all_fids]
                keep = output_rows >= 0
                kept_slots = feature_seg_slots[keep]
                kept_rows = output_rows[keep]
                kept_pcts = all_pcts[keep].float()
                if fp_target_layout.has_active_clusters:
                    linear = kept_slots * fp_target_layout.output_width + kept_rows
                    fp_targets.view(-1).scatter_reduce_(
                        0,
                        linear,
                        kept_pcts,
                        reduce="amax",
                        include_self=True,
                    )
                else:
                    # Mask-only compaction preserves legacy duplicate semantics
                    # (the final indexed assignment wins) and therefore equals an
                    # exact compact-column slice of the legacy full target.
                    fp_targets.view(-1, fp_target_layout.output_width)[kept_slots, kept_rows] = (
                        kept_pcts
                    )

    if fp_target_layout is None or not fp_target_layout.is_compact:
        assert legacy_width is not None
        fp_targets = _zero_masked_fp_targets(fp_targets, legacy_mask, legacy_width)

    return fp_targets


# ---------------------------------------------------------------------------
# Fused CSR → selected features (no intermediate padded tensor)
# ---------------------------------------------------------------------------


def _build_active_membership(
    all_fids: t.Tensor,
    active_sizes: t.Tensor,
    total_segs: int,
    n_sae_features: int,
) -> t.Tensor:
    """Flat per-segment active-feature membership mask, ``bool[total_segs, n_sae]``.

    ``membership[seg, fid]`` is True iff feature ``fid`` is active in that flat
    segment.  This replaces the padded ``[total_segs, active_width]`` exclusion
    buffer and its ``[total_segs, n_cand, active_width]`` 3-D broadcast collision
    check with an ``O(total_segs * n_sae)`` mask plus an ``O(total_segs * n_cand)``
    advanced-index lookup (the same technique as
    ``sample_inactive_features_batched`` in ``sampling.py``).

    Callers MUST ensure every id in ``all_fids`` lies in ``[0, n_sae_features)``
    (see the corrupt-shard guard in ``reconstruct_and_select``); indexing with an
    out-of-range id would otherwise raise a bare ``IndexError``.
    """
    membership = t.zeros(total_segs, n_sae_features, dtype=t.bool)
    if total_segs > 0 and int(active_sizes.sum().item()) > 0:
        seg_idx = t.repeat_interleave(t.arange(total_segs), active_sizes)
        membership[seg_idx, all_fids.long()] = True
    return membership


def _exact_inactive_negatives(
    active_row: t.Tensor,
    allowed_mask: t.Tensor | None,
    k: int,
    rng: t.Generator,
) -> t.Tensor:
    """Exactly ``min(k, |eligible|)`` distinct inactive ids for one segment.

    ``eligible`` is the allowed universe minus this segment's active features.
    Enumerate-then-permute over the (ascending) eligible set — used both by the
    small-pool exact path and to repair any vectorized-draw shortfall, so
    delivery equals the clamped ``neg_ks`` rather than silently under-delivering
    (the fused-path analogue of ``sample_inactive_features_batched``'s
    per-segment exact fallback).
    """
    if k <= 0:
        return t.zeros(0, dtype=t.int64)
    eligible = ~active_row
    if allowed_mask is not None:
        eligible = eligible & allowed_mask
    eligible_ids = eligible.nonzero(as_tuple=True)[0]
    n_elig = int(eligible_ids.numel())
    if n_elig == 0:
        return t.zeros(0, dtype=t.int64)
    perm = t.randperm(n_elig, generator=rng)[: min(k, n_elig)]
    return eligible_ids[perm].to(t.int64)


def reconstruct_and_select(
    shard: PackedRowShard,
    row_indices: list[int] | range,
    pipeline: FeaturePipelineConfig,
    device: str = "cpu",
    pad_n_seg: int | None = None,
    n_sae_features: int | None = None,
    pad_F: int | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
) -> tuple[ReconstructedBatch, t.Tensor | None]:
    """Fused CSR → selected features, bypassing the ~4000-wide padded intermediate.

    Goes directly from CSR packed row data to selected, discretized features
    with a much smaller F_max (~8-20 instead of ~4000).

    The savings come from skipping:
    1. Allocation of ``(B, n_seg, ~4000)`` padded tensors
    2. Vectorized scatter into those tensors
    3. Immediate unpadding back to flat per-segment arrays

    Instead, features are extracted directly from the CSR arrays and fed
    straight into the sampling/discretization pipeline.

    When *n_sae_features* is provided, also builds dense feature prediction
    targets from ALL extracted features (before selection), returning them
    alongside the batch.

    Args:
        shard: Packed row shard with CSR segment/feature data.
        row_indices: Row indices to include in the batch.
        pipeline: Feature pipeline config (selection, boundaries, negatives).
        device: Target device for output tensors.
        pad_n_seg: Optional static padding for the segment dimension.
        n_sae_features: When set, build dense fp_targets [B, n_seg, n_sae_features]
            from all extracted CSR features. None skips target building.
        pad_F: Optional static padding for the feature dimension. When set,
            output tensors are padded to exactly pad_F features (extra
            positions are zero-filled with mask=False).

    Returns:
        (ReconstructedBatch, fp_targets) where fp_targets is a dense float32 tensor
        of shape [B, n_seg, n_sae_features] with all extracted feature percentiles,
        or None when n_sae_features is not provided.
    """
    if fp_target_layout is not None and n_sae_features is not None:
        raise ValueError(
            "reconstruct_and_select accepts fp_target_layout or the legacy "
            "n_sae_features argument, not both"
        )
    if fp_target_layout is not None and (fp_target_layout.feature_mask != pipeline.feature_mask):
        raise ValueError(
            "reconstruct_and_select received an FP target layout derived from a "
            "different FeatureMask than the feature pipeline"
        )
    if fp_target_layout is not None and fp_target_layout.has_active_clusters:
        raise FeatureClustersError(
            "clustered compact FP targets require precomputed selection shards; "
            "fused live feature selection is unsupported"
        )

    # === Phase 1: Shared index building (tokens, seg_map, seg_doc_id) ===
    idx = _build_batch_index(shard, row_indices, pad_n_seg=pad_n_seg)
    B = idx.B
    batch_n_seg = idx.batch_n_seg
    total_segs = idx.total_segs

    # === Phase 2: Extract features directly from CSR (NO padding) ===
    feat_offsets_t = shard.seg_feat_offsets.long()
    all_fids = t.tensor([], dtype=t.int64)
    all_mags = t.tensor([], dtype=t.float32)
    all_pcts = t.tensor([], dtype=t.float32)
    segment_sizes: list[int] = []
    total_feats = 0

    # total_segs == 0 is accepted for schema-compatible empty packed rows
    # (defensive/back-compat path); the current producer normally emits at
    # least one empty segment per row. total_feats == 0 is expected when those
    # segment slots carry no active SAE features.
    if total_segs > 0:
        feat_starts = feat_offsets_t[idx.global_flat]  # [total_segs]
        feat_ends = feat_offsets_t[idx.global_flat + 1]  # [total_segs]
        feat_counts = feat_ends - feat_starts  # features per segment
        segment_sizes = feat_counts.tolist()
        total_feats = int(feat_counts.sum().item())

        if total_feats > 0:
            # Single vectorized gather of ALL features across ALL segments
            cum_fc = t.cat([t.zeros(1, dtype=t.long), feat_counts.cumsum(0)[:-1]])
            gather_offsets = t.arange(total_feats) - t.repeat_interleave(cum_fc, feat_counts)
            gather_idx = t.repeat_interleave(feat_starts, feat_counts) + gather_offsets

            all_fids = shard.feature_ids[gather_idx].to(t.int64)
            all_pcts = shard.feature_pcts[gather_idx].float() * (1.0 / 255.0)
            all_mags = all_pcts  # percentiles as magnitude proxy

    # === Phase 2b: Build dense feature prediction targets (before selection) ===
    batch_fp_targets: t.Tensor | None = None
    if n_sae_features is not None or fp_target_layout is not None:
        batch_fp_targets = _build_fp_targets_from_flat_features(
            idx,
            all_fids,
            all_pcts,
            segment_sizes,
            n_sae_features,
            feature_mask=(pipeline.feature_mask if fp_target_layout is None else None),
            fp_target_layout=fp_target_layout,
        )

    # === Phase 3 pre-filter: mask ===
    if pipeline.feature_mask is not None and total_feats > 0:
        new_fids, new_mags, new_pcts, new_sizes = [], [], [], []
        off = 0
        for sz in segment_sizes:
            if sz:
                f, m, p = pipeline.feature_mask.apply_to_positive(
                    all_fids[off : off + sz],
                    all_mags[off : off + sz],
                    all_pcts[off : off + sz],
                )
            else:
                f = all_fids[:0]
                m = all_mags[:0]
                p = all_pcts[:0]
            new_fids.append(f)
            new_mags.append(m)
            new_pcts.append(p)
            new_sizes.append(int(f.numel()))
            off += sz
        all_fids = t.cat(new_fids) if new_fids else all_fids[:0]
        all_mags = t.cat(new_mags) if new_mags else all_mags[:0]
        all_pcts = t.cat(new_pcts) if new_pcts else all_pcts[:0]
        segment_sizes = new_sizes
        total_feats = int(all_fids.numel())

    # === Phase 3: Apply sampling/threshold selection ===
    if isinstance(pipeline.selection_config, SamplingBundle):
        sampled = sample_features_batched(
            all_fids,
            all_mags,
            segment_sizes,
            pipeline.selection_config.percentile_table,
            pipeline.selection_config.selection,
            pipeline.selection_config.rng,
            precomputed_percentiles=all_pcts,
        )
    elif isinstance(pipeline.selection_config, ThresholdSelectionConfig):
        threshold_cfg = ThresholdConfig(
            pipeline.selection_config.selection.threshold_type,
            pipeline.selection_config.selection.threshold,
        )
        sampled: list[tuple[t.Tensor, t.Tensor]] = []
        offset = 0
        for sz in segment_sizes:
            if sz > 0:
                sel_fids, sel_mags = apply_threshold(
                    threshold_cfg,
                    all_fids[offset : offset + sz],
                    all_mags[offset : offset + sz],
                    precomputed=pipeline.selection_config.precomputed_thresholds,
                )
                sampled.append((sel_fids, sel_mags))
            else:
                sampled.append((t.tensor([], dtype=t.int64), t.tensor([], dtype=t.float32)))
            offset += sz
    else:
        raise TypeError(f"Unsupported selection config: {type(pipeline.selection_config).__name__}")

    # === Phase 4: Pad selected features into [total_segs, topk_width] + discretize ===
    sel_fids_list = [s[0] for s in sampled]
    sel_mags_list = [s[1] for s in sampled]
    sel_sizes_t = t.tensor([len(s[0]) for s in sampled], dtype=t.long)
    topk_width = max(int(sel_sizes_t.max().item()) if total_segs > 0 else 0, 1)

    # Pad positives into fixed-width [total_segs, topk_width] tensors
    sel_fids = t.zeros(total_segs, topk_width, dtype=t.int64)
    sel_mags = t.zeros(total_segs, topk_width, dtype=t.float32)
    pos_mask = t.zeros(total_segs, topk_width, dtype=t.bool)

    if total_segs > 0 and int(sel_sizes_t.sum().item()) > 0:
        # Vectorized scatter: build flat → padded index mapping
        total_sel = int(sel_sizes_t.sum().item())
        seg_idx_sel = t.repeat_interleave(t.arange(total_segs), sel_sizes_t)  # [total_sel]
        cum_sel = t.cat([t.zeros(1, dtype=t.long), sel_sizes_t.cumsum(0)[:-1]])
        pos_col = t.arange(total_sel) - t.repeat_interleave(cum_sel, sel_sizes_t)  # [total_sel]

        all_sel_fids = t.cat(sel_fids_list)
        all_sel_mags = t.cat(sel_mags_list)

        sel_fids[seg_idx_sel, pos_col] = all_sel_fids
        sel_mags[seg_idx_sel, pos_col] = all_sel_mags
        pos_mask[seg_idx_sel, pos_col] = True
    else:
        all_sel_fids = t.tensor([], dtype=t.int64)
        all_sel_mags = t.tensor([], dtype=t.float32)

    # Batched discretization on the flat concatenated arrays
    if len(all_sel_fids) > 0:
        all_buckets = discretize_activations(all_sel_fids, all_sel_mags, pipeline.boundaries)
        all_out_pcts = compute_percentiles(all_sel_fids, all_sel_mags, pipeline.boundaries)
    else:
        all_buckets = t.tensor([], dtype=t.int64)
        all_out_pcts = t.tensor([], dtype=t.float32)

    # Defense-in-depth (H-assert): assert finalized positive ids contain no masked features.
    # Cheap: boolean membership over all_sel_fids (flat union of per-segment selections).
    # Strictly mask-gated (I5: no-op when feature_mask is None).
    if pipeline.feature_mask is not None:
        pipeline.feature_mask.assert_no_masked(all_sel_fids)

    # Scatter buckets/pcts into padded layout
    sel_buckets = t.zeros(total_segs, topk_width, dtype=t.int64)
    sel_out_pcts = t.zeros(total_segs, topk_width, dtype=t.float32)
    if len(all_buckets) > 0:
        sel_buckets[seg_idx_sel, pos_col] = all_buckets
        sel_out_pcts[seg_idx_sel, pos_col] = all_out_pcts

    # === Phase 5: Vectorized negative sampling ===
    max_neg = 0
    neg_fids = t.zeros(total_segs, 0, dtype=t.int64)
    neg_mask = t.zeros(total_segs, 0, dtype=t.bool)

    if pipeline.negative_bundle is not None and total_segs > 0:
        neg_config = pipeline.negative_bundle
        neg_ks = sample_k_batched(
            neg_config.selection.distribution,
            neg_config.selection.mean,
            neg_config.selection.std,
            neg_config.rng,
            total_segs,
            min_k=neg_config.selection.min_k,
            max_k=neg_config.selection.max_k,
        )
        active_sizes_t = t.tensor(segment_sizes, dtype=t.long)

        # Fail loud on active ids the negative universe cannot represent. An id
        # >= n_sae_features (shard/config SAE-width mismatch, or an id a blacklist
        # keeps because keep_mask treats OOB ids as not-a-member) would otherwise
        # raise a bare IndexError when indexing the allowed mask or the flat
        # membership tensor below. Diagnose it for BOTH masked and no-mask paths.
        if total_feats > 0:
            _amin = int(all_fids.min().item())
            _amax = int(all_fids.max().item())
            if _amin < 0 or _amax >= neg_config.n_sae_features:
                _bad = _amax if _amax >= neg_config.n_sae_features else _amin
                raise ValueError(
                    f"negative sampling: active feature id {_bad} is outside the "
                    f"negative universe [0, {neg_config.n_sae_features}) — "
                    "n_sae_features understates the shard's SAE width (corrupt "
                    "shard or config mismatch)"
                )

        # Clamp neg_ks to the eligible pool (allowed universe minus ALL active
        # ids). `_allowed` (bool[n_sae]) doubles as the per-feature negative gate
        # on the masked path; it is None on the no-mask path (full universe).
        _neg_mask = negative_sampling_mask(pipeline)
        if _neg_mask is not None:
            _allowed = _neg_mask.allowed_universe_mask(neg_config.n_sae_features)
            _univ = int(_allowed.sum().item())
            # Defense-in-depth: every active id must lie within the allowed
            # universe after feature_mask filtering (ids are in range per guard).
            if total_feats > 0 and not bool(_allowed[all_fids.long()].all().item()):
                bad = int(all_fids[~_allowed[all_fids.long()]][0].item())
                raise ValueError(
                    f"negative sampling exclusion id {bad} is outside the allowed "
                    "negative universe after feature_mask filtering"
                )
        else:
            _allowed = None
            _univ = neg_config.n_sae_features
        neg_ks = t.minimum(neg_ks, t.tensor(_univ) - active_sizes_t).clamp(min=0)
        max_neg = int(neg_ks.max().item())

        if max_neg > 0:
            # Flat per-segment active membership: replaces the padded exclusion
            # buffer + 3-D broadcast collision check with an O(n_seg * n_sae) mask.
            # Built only when negatives are actually drawn (all-zero neg_ks skips
            # it) and consumed by every branch below plus the defense-in-depth
            # collision check.
            active_membership = _build_active_membership(
                all_fids, active_sizes_t, total_segs, neg_config.n_sae_features
            )
            if _neg_mask is not None:
                # === Restricted branch: HYBRID fast-vectorized + exact-small-pool ===
                # (`_allowed` is the mask's allowed universe.)
                #
                # Two paths, chosen by |allowed| vs _NEG_SMALL_POOL_EXACT_THRESHOLD:
                #
                # FAST VECTORIZED PATH (|allowed| >= threshold, e.g. blacklist case):
                #   Draw candidates_idx ~ Uniform[0, |allowed|) per (seg, cand),
                #   map to feature IDs via allowed_ids[candidates_idx], dedup +
                #   collision-filter, pack to front.  n_cand uses an adaptive factor
                #   (mirror of sampling.py's MIN_OVERSAMPLE_FACTOR logic) so delivery
                #   is near-certain for any pool comfortably larger than neg_ks.
                #   No per-segment Python loop over a large set; no randperm(_univ).
                #
                # EXACT SMALL-POOL PATH (|allowed| < threshold):
                #   Per-segment Python loop with randperm(len(eligible)) where
                #   len(eligible) <= |allowed| < threshold — always cheap.
                #   Guarantees EXACTLY min(neg_ks[seg], eff_pool[seg]) for all segs.
                #
                # Threshold constant: if |allowed| < this, use exact per-seg path.
                # randperm(< _NEG_SMALL_POOL_EXACT_THRESHOLD) is always fast.
                # Vectorized path oversample factor: adaptive (mirror sampling.py).
                # factor = max(4, |allowed| // max_neg + 1), capped at a sane upper
                # bound so n_cand never exceeds memory budget.

                # Safety: if _univ == 0 the neg_ks clamp above forces max_neg == 0,
                # so this branch is never reached with an empty allowed set.
                assert _univ > 0, (
                    "BUG: max_neg>0 but _univ==0 — neg_ks clamp should have prevented this"
                )
                # This branch is entered iff _neg_mask is not None, so _allowed is
                # the allowed-universe mask (never None here).
                assert _allowed is not None
                # allowed_ids: ascending 1-D LongTensor of allowed feature ids (DDP-stable).
                allowed_ids = _allowed.nonzero(as_tuple=True)[0]

                neg_fids = t.zeros(total_segs, max_neg, dtype=t.int64)
                neg_valid = t.zeros(total_segs, max_neg, dtype=t.bool)

                if _univ < _NEG_SMALL_POOL_EXACT_THRESHOLD:
                    # === EXACT SMALL-POOL PATH ===
                    # Enumerate-permute over the per-segment eligible set (allowed
                    # minus active); randperm arg <= |allowed| < 128 — always cheap.
                    # Guarantees exactly min(neg_ks[seg], eff_pool[seg]) per segment.
                    for _seg in range(total_segs):
                        _k = int(neg_ks[_seg].item())
                        if _k <= 0:
                            continue
                        chosen = _exact_inactive_negatives(
                            active_membership[_seg], _allowed, _k, neg_config.rng
                        )
                        _c = int(chosen.numel())
                        if _c > 0:
                            neg_fids[_seg, :_c] = chosen
                            neg_valid[_seg, :_c] = True
                else:
                    # === FAST VECTORIZED PATH (large |allowed|, e.g. blacklist) ===
                    # No per-segment loop over the full allowed set; no randperm(_univ).
                    # Adaptive n_cand: factor grows with |allowed|/max_neg density to
                    # ensure near-100% delivery in a single vectorized draw.
                    _factor = min(
                        _NEG_VEC_MAX_FACTOR,
                        max(
                            _NEG_VEC_MIN_FACTOR,
                            _univ // max(max_neg, 1) + 1,
                        ),
                    )
                    _n_cand = min(max_neg * _factor, _univ)
                    # Draw indices into allowed_ids (not raw feature IDs), then map.
                    _cand_idx = t.randint(
                        0,
                        _univ,
                        (total_segs, _n_cand),
                        generator=neg_config.rng,
                    )
                    candidates = allowed_ids[_cand_idx]  # [total_segs, n_cand] int64

                    # Dedup: sort, mark adjacent dups, unsort flags
                    sorted_cands, sort_idx = candidates.sort(dim=1)
                    is_dup_sorted = t.zeros(total_segs, _n_cand, dtype=t.bool)
                    is_dup_sorted[:, 1:] = sorted_cands[:, 1:] == sorted_cands[:, :-1]
                    is_dup = t.zeros_like(is_dup_sorted)
                    is_dup.scatter_(1, sort_idx, is_dup_sorted)

                    # Collision with active features via flat membership lookup:
                    # active_membership[seg, candidate] — O(total_segs * n_cand),
                    # no [total_segs, n_cand, active_width] 3-D broadcast.
                    _seg_row = t.arange(total_segs).unsqueeze(1).expand_as(candidates)
                    has_collision = active_membership[_seg_row, candidates]  # [total_segs, n_cand]

                    # All candidates are from allowed_ids by construction (I4 structural).
                    valid_cands = ~is_dup & ~has_collision

                    # Pack valid candidates to front, take first max_neg
                    # Sort by ~valid_cands (False=0 first), stable preserves order
                    _, perm = (~valid_cands).to(t.int8).sort(dim=1, stable=True)
                    neg_fids = candidates.gather(1, perm)[:, :max_neg]
                    neg_valid = valid_cands.gather(1, perm)[:, :max_neg]
                # I4 defense-in-depth: all chosen ids are from allowed_ids by construction;
                # this assert is now structural (no-op in correct code, kept for safety).
                # neg_mask applied below (shared with no-mask path).

            else:
                # === No-mask branch: vectorized randint over the full universe ===
                n_cand = max_neg * _NEG_NO_MASK_OVERSAMPLE_FACTOR
                candidates = t.randint(
                    0,
                    neg_config.n_sae_features,
                    (total_segs, n_cand),
                    generator=neg_config.rng,
                )

                # Dedup: sort, mark adjacent dups, unsort flags
                sorted_cands, sort_idx = candidates.sort(dim=1)
                is_dup_sorted = t.zeros(total_segs, n_cand, dtype=t.bool)
                is_dup_sorted[:, 1:] = sorted_cands[:, 1:] == sorted_cands[:, :-1]
                is_dup = t.zeros_like(is_dup_sorted)
                is_dup.scatter_(1, sort_idx, is_dup_sorted)

                # Collision with active features via flat membership lookup:
                # active_membership[seg, candidate] — O(total_segs * n_cand), no
                # [total_segs, n_cand, active_width] 3-D broadcast.
                _seg_row = t.arange(total_segs).unsqueeze(1).expand_as(candidates)
                has_collision = active_membership[_seg_row, candidates]  # [total_segs, n_cand]

                valid = ~is_dup & ~has_collision

                # Pack valid candidates to front, take first max_neg
                # Sort by ~valid (False=0 first), stable preserves order among equals
                _, perm = (~valid).to(t.int8).sort(dim=1, stable=True)
                neg_fids = candidates.gather(1, perm)[:, :max_neg]
                neg_valid = valid.gather(1, perm)[:, :max_neg]

            # Repair any segment that under-delivered vs its clamped neg_ks. The
            # eligible pool holds >= neg_ks[seg] inactive allowed ids (neg_ks was
            # clamped to it), so exact enumerate-permute guarantees full delivery.
            # The small-pool exact path never under-delivers, so this is a no-op
            # there; it repairs the vectorized draws' tight-pool shortfall.
            #
            # This runs on the no-mask path (_allowed is None) too, by design:
            # the fail-loud-on-under-delivery contract overrides strict no-mask
            # RNG byte-identity with the pre-feature-mask baseline. A no-mask
            # shortfall only arises on tight pools (n_sae_features close to a
            # segment's active count) — absent in production (large n_sae), so
            # the extra randperm draw is confined to synthetic/degenerate cases.
            _within_k = t.arange(max_neg).unsqueeze(0) < neg_ks.unsqueeze(1)
            _short = ((neg_valid & _within_k).sum(dim=1) < neg_ks).nonzero(as_tuple=True)[0]
            for _seg in _short.tolist():
                chosen = _exact_inactive_negatives(
                    active_membership[_seg],
                    _allowed,
                    int(neg_ks[_seg].item()),
                    neg_config.rng,
                )
                neg_fids[_seg, :] = 0
                neg_valid[_seg, :] = False
                _c = int(chosen.numel())
                if _c > 0:
                    neg_fids[_seg, :_c] = chosen
                    neg_valid[_seg, :_c] = True

            neg_mask = neg_valid & _within_k

            # Fail loud: after repair every segment must deliver exactly its
            # clamped neg_ks. A residual shortfall is an invariant violation (the
            # pool was sized to satisfy neg_ks) — never silently ship fewer.
            _delivered = neg_mask.sum(dim=1)
            if bool((_delivered < neg_ks).any().item()):
                _bad = int((_delivered < neg_ks).nonzero(as_tuple=True)[0][0].item())
                raise ValueError(
                    f"negative sampling under-delivered for flat segment {_bad}: "
                    f"{int(_delivered[_bad].item())} valid negatives but "
                    f"neg_ks={int(neg_ks[_bad].item())} (eligible pool exhausted or "
                    "sampler shortfall)"
                )

    # Defense-in-depth: finalized negative ids must be valid feature ids,
    # inactive for their segment, and unique within the segment. The mask
    # assertion below remains strictly mask-gated (I5 no-op when mask is None).
    if max_neg > 0:
        valid_neg_fids = neg_fids[neg_mask]
        if valid_neg_fids.numel() > 0:
            if bool((valid_neg_fids < 0).any().item()):
                bad = int(valid_neg_fids[valid_neg_fids < 0][0].item())
                raise ValueError(f"negative sampling produced invalid negative feature id {bad}")
            if bool((valid_neg_fids >= neg_config.n_sae_features).any().item()):
                bad = int(valid_neg_fids[valid_neg_fids >= neg_config.n_sae_features][0].item())
                raise ValueError(
                    f"negative sampling produced feature id {bad} >= "
                    f"n_sae_features={neg_config.n_sae_features}"
                )

            # Active-collision check via the flat membership mask (no 3-D
            # broadcast): a True at a valid negative slot means an active id was
            # emitted as a negative. Padding slots index membership[:, 0] but are
            # masked out by neg_mask.
            _neg_seg_row = t.arange(total_segs).unsqueeze(1).expand_as(neg_fids)
            negative_active_collision = active_membership[_neg_seg_row, neg_fids] & neg_mask
            if bool(negative_active_collision.any().item()):
                bad_idx = negative_active_collision.nonzero()[0]
                bad_seg = int(bad_idx[0].item())
                bad_neg_col = int(bad_idx[1].item())
                bad_fid = int(neg_fids[bad_seg, bad_neg_col].item())
                raise ValueError(
                    f"negative sampling emitted active feature id {bad_fid} "
                    f"as a negative for flat segment {bad_seg}"
                )

            if max_neg > 1:
                duplicate_negative = (
                    (neg_fids.unsqueeze(2) == neg_fids.unsqueeze(1))
                    & neg_mask.unsqueeze(2)
                    & neg_mask.unsqueeze(1)
                    & ~t.eye(max_neg, dtype=t.bool).unsqueeze(0)
                )
                if bool(duplicate_negative.any().item()):
                    bad_idx = duplicate_negative.nonzero()[0]
                    bad_seg = int(bad_idx[0].item())
                    bad_neg_col = int(bad_idx[1].item())
                    bad_fid = int(neg_fids[bad_seg, bad_neg_col].item())
                    raise ValueError(
                        f"negative sampling emitted duplicate feature id {bad_fid} "
                        f"for flat segment {bad_seg} at negative column {bad_neg_col}"
                    )

        if pipeline.feature_mask is not None:
            pipeline.feature_mask.assert_no_masked(valid_neg_fids)

    # === Phase 6: Bulk assembly (no Python loop) + scatter to (B, n_seg, F) ===
    actual_f = max(1, topk_width + max_neg)

    # When pad_F is provided, use it as the output feature dimension.
    if pad_F is not None and actual_f > pad_F:
        raise ValueError(
            f"Batch feature count {actual_f} exceeds pad_F={pad_F}. "
            "Refusing to truncate features; increase max_k or pad_F."
        )

    output_f = pad_F if pad_F is not None else actual_f

    # Build flat [total_segs, F_new] tensors with bulk copies
    out_fids = t.zeros(total_segs, output_f, dtype=t.int64)
    out_mags = t.zeros(total_segs, output_f, dtype=t.int64)
    out_mask = t.zeros(total_segs, output_f, dtype=t.bool)
    out_pcts = t.zeros(total_segs, output_f, dtype=t.float32)

    if total_segs > 0:
        # Positives: first topk_width columns
        out_fids[:, :topk_width] = sel_fids
        out_mags[:, :topk_width] = sel_buckets
        out_pcts[:, :topk_width] = sel_out_pcts
        out_mask[:, :topk_width] = pos_mask

        # Negatives: next max_neg columns
        if max_neg > 0:
            out_fids[:, topk_width : topk_width + max_neg] = neg_fids
            out_mags[:, topk_width : topk_width + max_neg] = pipeline.effective_buckets
            out_mask[:, topk_width : topk_width + max_neg] = neg_mask
        # Extra columns (topk_width + max_neg .. output_f) stay zero with mask=False

    # Scatter from flat [total_segs, F] → batch [B, n_seg, F]
    enc_fids = t.zeros(B, batch_n_seg, output_f, dtype=t.int64)
    enc_mags = t.zeros(B, batch_n_seg, output_f, dtype=t.int64)
    enc_mask = t.zeros(B, batch_n_seg, output_f, dtype=t.bool)
    enc_pcts = t.zeros(B, batch_n_seg, output_f, dtype=t.float32)

    if total_segs > 0:
        enc_fids[idx.b_flat, idx.local_flat] = out_fids
        enc_mags[idx.b_flat, idx.local_flat] = out_mags
        enc_mask[idx.b_flat, idx.local_flat] = out_mask
        enc_pcts[idx.b_flat, idx.local_flat] = out_pcts

    # === Phase 7: Bulk transfer to device (pin_memory for async) ===
    inputs, targets = idx.inputs, idx.targets
    seg_map, seg_doc_id = idx.seg_map, idx.seg_doc_id
    if device != "cpu":
        inputs = _pin_transfer(inputs, device)
        targets = _pin_transfer(targets, device)
        seg_map = _pin_transfer(seg_map, device)
        seg_doc_id = _pin_transfer(seg_doc_id, device)
        enc_fids = _pin_transfer(enc_fids, device)
        enc_mags = _pin_transfer(enc_mags, device)
        enc_mask = _pin_transfer(enc_mask, device)
        enc_pcts = _pin_transfer(enc_pcts, device)
        if batch_fp_targets is not None:
            batch_fp_targets = _pin_transfer(batch_fp_targets, device)

    result = ReconstructedBatch(
        inputs=inputs,
        targets=targets,
        enc_fids=enc_fids,
        enc_mags=enc_mags,
        enc_mask=enc_mask,
        enc_pcts=enc_pcts,
        seg_map=seg_map,
        seg_doc_id=seg_doc_id,
    )

    return result, batch_fp_targets


# ---------------------------------------------------------------------------
# Cross-shard batch assembly
# ---------------------------------------------------------------------------


def _stack_batch_parts(
    parts: list[ReconstructedBatch],
    fp_parts: list[t.Tensor | None],
    B: int,
    device: str,
    pad_n_seg: int | None,
    pad_F: int | None,
) -> tuple[
    t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor | None
]:
    """Stack individually-reconstructed rows into a single batch.

    Contract:
      - len(parts) == len(fp_parts) == B.
      - Each parts[i] is a per-row ReconstructedBatch with leading B=1.
      - fp_parts is either all-None or all-Tensor. When all-Tensor, each
        fp_parts[i] has shape (1, parts[i].enc_fids.shape[1], n_sae) with
        the same dtype, n_sae, and device across i.
      - Output fp_targets is None iff fp_parts is all-None; otherwise its
        shape is (B, max_ns, n_sae) where max_ns matches enc_fids.shape[1].

    When *pad_n_seg* and *pad_F* are both set, all parts already share
    shape and a simple cat suffices. Otherwise, re-pads to the
    batch-level maximum so fp_targets and enc_* stay shape-coherent.
    """
    assert len(parts) == B, f"len(parts)={len(parts)} != B={B}"
    assert len(fp_parts) == B, f"len(fp_parts)={len(fp_parts)} != B={B}"
    if B > 0:
        # Device coherence: all per-row parts must be on the same device type
        # as the requested output. Production callers route through
        # reconstruct_batch / reconstruct_and_select with device=device, so this
        # is satisfied by construction; the assert catches future regressions
        # where a caller decouples them and produces a silent mixed-device batch.
        expected_device_type = t.device(device).type
        actual_device_type = parts[0].inputs.device.type
        assert actual_device_type == expected_device_type, (
            f"parts[0].inputs.device.type={actual_device_type} != "
            f"device.type={expected_device_type} "
            "(all per-row parts must share device with the requested output)"
        )

    if pad_n_seg is not None and pad_F is not None:
        # Fast path: all parts already share shape — simple cat.
        # Defensive: every part must agree with pad_n_seg/pad_F. Upstream
        # reconstruction honors these by construction; if a future regression
        # drifts, fail loudly here rather than silently violating the
        # compile-traced shape contract.
        if B > 0:
            first_enc = parts[0].enc_fids
            assert first_enc.shape[1] == pad_n_seg, (
                f"parts[0].enc_fids.shape[1]={first_enc.shape[1]} "
                f"!= pad_n_seg={pad_n_seg} (fast-path shape contract)"
            )
            assert first_enc.shape[2] == pad_F, (
                f"parts[0].enc_fids.shape[2]={first_enc.shape[2]} "
                f"!= pad_F={pad_F} (fast-path shape contract)"
            )
        inputs = t.cat([p.inputs for p in parts])
        targets = t.cat([p.targets for p in parts])
        enc_fids = t.cat([p.enc_fids for p in parts])
        enc_mags = t.cat([p.enc_mags for p in parts])
        enc_mask = t.cat([p.enc_mask for p in parts])
        enc_pcts = t.cat([p.enc_pcts for p in parts])
        seg_map = t.cat([p.seg_map for p in parts])
        seg_doc_id = t.cat([p.seg_doc_id for p in parts])
        max_ns = enc_fids.shape[1]  # == pad_n_seg
    else:
        # Slow path: pad to batch-level max for n_seg AND F.
        inputs = t.cat([p.inputs for p in parts])
        targets = t.cat([p.targets for p in parts])
        seg_map = t.cat([p.seg_map for p in parts])
        max_ns = max(p.enc_fids.shape[1] for p in parts)
        max_f = max(p.enc_fids.shape[2] for p in parts)
        enc_fids = t.zeros(B, max_ns, max_f, dtype=t.int64, device=device)
        enc_mags = t.zeros(B, max_ns, max_f, dtype=t.int64, device=device)
        enc_pcts = t.zeros(B, max_ns, max_f, dtype=t.float32, device=device)
        enc_mask = t.zeros(B, max_ns, max_f, dtype=t.bool, device=device)
        seg_doc_id = t.full((B, max_ns), -1, dtype=t.int64, device=device)
        for i, p in enumerate(parts):
            ns, nf = p.enc_fids.shape[1], p.enc_fids.shape[2]
            enc_fids[i, :ns, :nf] = p.enc_fids[0]
            enc_mags[i, :ns, :nf] = p.enc_mags[0]
            enc_mask[i, :ns, :nf] = p.enc_mask[0]
            enc_pcts[i, :ns, :nf] = p.enc_pcts[0]
            seg_doc_id[i, : p.seg_doc_id.shape[1]] = p.seg_doc_id[0]

    fp_targets = None
    if any(fp is not None for fp in fp_parts):
        # Contract: fp_parts is either all-Tensor or all-None. Current
        # callers construct fp_parts via the same n_sae_features flag for
        # every row in pending, so mixed-None is unreachable. Assert
        # defensively: silently dropping None entries would produce
        # fp_targets.shape[0] < B and misalign with the model's batch dim.
        assert all(fp is not None for fp in fp_parts), (
            "_stack_batch_parts: mixed None/Tensor fp_parts is unsupported "
            "(would silently misalign fp_targets batch dim with the model)"
        )
        fp_parts_t = cast(list[t.Tensor], fp_parts)
        first = fp_parts_t[0]
        n_sae = first.shape[2]
        dtype = first.dtype
        for i, fp in enumerate(fp_parts_t):
            assert fp.shape[0] == 1, (
                f"fp_parts[{i}].shape[0]={fp.shape[0]} "
                f"(expected 1, per-row reconstruction with B=1)"
            )
            assert fp.shape[2] == n_sae, f"fp_parts[{i}].shape[2]={fp.shape[2]} != n_sae={n_sae}"
            assert fp.dtype == dtype, f"fp_parts[{i}].dtype={fp.dtype} != {dtype}"
            assert fp.device == first.device, f"fp_parts[{i}].device={fp.device} != {first.device}"
            assert fp.shape[1] == parts[i].enc_fids.shape[1], (
                f"fp_parts[{i}].shape[1]={fp.shape[1]} != "
                f"parts[{i}].enc_fids.shape[1]={parts[i].enc_fids.shape[1]} "
                "(fp_targets n_seg must match enc_fids n_seg for the row)"
            )

        if pad_n_seg is not None and pad_F is not None:
            # Fast path: shapes already uniform — single cat.
            fp_targets = t.cat(fp_parts_t, dim=0)
        else:
            # Slow path: pad each row to batch-level max_ns (shared with enc_*).
            fp_targets = t.zeros(B, max_ns, n_sae, dtype=dtype, device=device)
            for i, fp in enumerate(fp_parts_t):
                fp_targets[i, : fp.shape[1]] = fp[0]

    return inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets


# ---------------------------------------------------------------------------
# Precomputed batch loader
# ---------------------------------------------------------------------------


def _validate_splice_counterfactual_selection_dir(
    *,
    packed_row_dir: Path,
    shard_paths: list[Path],
    selection_dir: Path | None,
    sel_shard_paths: dict[str, Path],
    feature_pipeline: FeaturePipelineConfig,
    splice_feature_dir: Path | None,
    feature_clusters: FeatureClusterConsolidation | None,
    remote_materializer: RemoteSelectionMaterializer | None = None,
    packed_materializer: RemotePackedRowMaterializer | None = None,
    remote_derived_feature_fingerprint: str | None = None,
) -> int | None:
    """Validate splice selection and return its global CSR width when local.

    Returns the validated global positive-plus-negative CSR width, or ``None``
    on the remote-materializer path. A completed splice certificate returns
    its width even when a remote materializer is active.
    """
    assert feature_pipeline.delta_bundle is not None
    configured_delta = canonical_feature_delta_dict(feature_pipeline.delta_bundle.config)
    if selection_dir is None:
        raise ValueError(
            "precomputed_data_loader requires selection_dir for "
            "feature_delta.mode=splice_counterfactual; packed rows cannot run "
            "fused live selection for any feature_delta mode. Run "
            "scripts.preprocess.precompute_selected_features for this packed "
            "row directory."
        )
    if not sel_shard_paths:
        raise ValueError(
            f"splice_counterfactual selection_dir {selection_dir} contains no selection shards."
        )

    manifest = read_manifest(selection_dir)
    if manifest is None:
        raise ValueError(
            f"splice_counterfactual selection_dir {selection_dir} is missing "
            "manifest.json; re-run precompute_selected_features."
        )
    meta = manifest.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError(
            f"splice_counterfactual selection_dir {selection_dir} has invalid "
            "manifest metadata; re-run precompute_selected_features."
        )
    configured_mask = canonical_dict(feature_pipeline.feature_mask)
    manifest_mask = meta.get("feature_mask")
    if manifest_mask != configured_mask:
        raise ValueError(
            f"splice_counterfactual selection manifest feature_mask mismatch "
            f"for {selection_dir}: manifest has {manifest_mask!r}, configured "
            f"{configured_mask!r}. Re-run precompute_selected_features with "
            "the current config."
        )
    assert_feature_clusters_selection_metadata(
        feature_clusters,
        meta,
        surface=f"splice_counterfactual selection manifest {selection_dir}",
    )
    if feature_clusters is not None and remote_materializer is None:
        assert_clustered_selection_shard_integrity(
            selection_dir,
            meta,
            surface=f"splice_counterfactual selection manifest {selection_dir}",
        )
    manifest_delta = meta.get("feature_delta")
    if manifest_delta != configured_delta:
        raise ValueError(
            f"splice_counterfactual selection manifest feature_delta mismatch "
            f"for {selection_dir}: manifest has {manifest_delta!r}, "
            f"configured {configured_delta!r}. Re-run "
            "precompute_selected_features with the current config."
        )

    packed_info = (
        packed_materializer.packed_provenance_info()
        if packed_materializer is not None
        else validate_packed_provenance(
            packed_row_dir,
            cache_dir=packed_row_dir,
        )
    )
    manifest_prov = meta.get("packed_provenance")
    expected_prov = {
        "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
        "packed_row_fingerprint": packed_info.packed_row_fingerprint,
        "provenance_shard_count": len(packed_info.provenance_shard_paths),
        "provenance_fingerprint": packed_info.provenance_fingerprint,
    }
    if manifest_prov != expected_prov:
        raise ValueError(
            f"splice_counterfactual selection manifest packed_provenance "
            f"mismatch for {selection_dir}: manifest has {manifest_prov!r}, "
            f"current packed rows require {expected_prov!r}. Re-run "
            "precompute_selected_features after regenerating packed rows."
        )
    splice_meta = meta.get("splice_counterfactual")
    if not isinstance(splice_meta, dict):
        raise ValueError(
            f"splice_counterfactual selection manifest metadata missing or "
            f"unsupported for {selection_dir}; re-run precompute_selected_features."
        )
    expected_splice_identity = splice_counterfactual_manifest_identity(
        clustered=feature_clusters is not None
    )
    if any(
        splice_meta.get(key) != expected for key, expected in expected_splice_identity.items()
    ) or (feature_clusters is None and CLUSTER_DELTA_SEMANTICS_KEY in splice_meta):
        raise ValueError(
            "splice_counterfactual selection manifest cluster-delta semantics "
            f"mismatch for {selection_dir}: manifest has schema/semantics "
            f"{splice_meta.get('schema_version')!r}/"
            f"{splice_meta.get(CLUSTER_DELTA_SEMANTICS_KEY)!r}, configured path "
            f"requires {expected_splice_identity!r}. Re-run "
            "precompute_selected_features."
        )
    if splice_meta.get("derived_shard_stems") != packed_info.base_shard_stems:
        raise ValueError(
            f"splice_counterfactual selection manifest derived_shard_stems "
            f"mismatch for {selection_dir}; re-run precompute_selected_features."
        )
    manifest_sidecar = splice_meta.get("sidecar_fingerprint")
    if not isinstance(manifest_sidecar, str):
        raise ValueError(
            f"splice_counterfactual selection manifest {selection_dir} lacks a "
            "sidecar_fingerprint; re-run precompute_selected_features."
        )
    if splice_feature_dir is None:
        raise ValueError(
            "precomputed_data_loader requires splice_feature_dir for "
            "feature_delta.mode=splice_counterfactual so it can validate the "
            "selection manifest artifact fingerprints."
        )
    current_sidecar = splice_sidecar_fingerprint(
        splice_feature_dir,
        cache_dir=packed_row_dir,
    )
    if manifest_sidecar != current_sidecar:
        raise ValueError(
            f"splice_counterfactual selection manifest sidecar_fingerprint "
            f"mismatch for {selection_dir}: manifest has {manifest_sidecar!r}, "
            f"current sidecars hash to {current_sidecar!r}. Re-run "
            "precompute_selected_features after regenerating splice sidecars."
        )
    manifest_derived_features = splice_meta.get("derived_feature_fingerprint")
    if remote_derived_feature_fingerprint is not None:
        if manifest_derived_features != remote_derived_feature_fingerprint:
            raise ValueError(
                f"splice_counterfactual selection manifest "
                f"derived_feature_fingerprint mismatch for {selection_dir}: "
                f"manifest has {manifest_derived_features!r}, verified remote "
                "startup manifests require "
                f"{remote_derived_feature_fingerprint!r}. Re-run "
                "precompute_selected_features and republish the remote "
                "selection manifests."
            )
    else:
        current_derived_features = derived_feature_fingerprint(
            splice_feature_dir,
            expected_stems=packed_info.base_shard_stems,
            cache_dir=packed_row_dir,
        )
        if manifest_derived_features != current_derived_features:
            raise ValueError(
                f"splice_counterfactual selection manifest "
                f"derived_feature_fingerprint mismatch for {selection_dir}: "
                f"manifest has {manifest_derived_features!r}, current derived "
                f"features hash to {current_derived_features!r}. Re-run "
                "precompute_selected_features after regenerating derived feature "
                "shards."
            )
    current_boundaries = tensor_content_fingerprint(feature_pipeline.boundaries)
    manifest_boundaries = splice_meta.get("boundary_fingerprint")
    if manifest_boundaries != current_boundaries:
        raise ValueError(
            f"splice_counterfactual selection manifest boundary_fingerprint "
            f"mismatch for {selection_dir}: manifest has {manifest_boundaries!r}, "
            f"current boundaries hash to {current_boundaries!r}. Re-run "
            "precompute_selected_features after changing statistics, buckets, or "
            "threshold settings."
        )

    packed_stems = (
        packed_materializer.stems()
        if packed_materializer is not None
        else {p.stem.split(".")[0] for p in shard_paths}
    )
    sel_stems = set(sel_shard_paths)
    missing = sorted(packed_stems - sel_stems)
    extra = sorted(sel_stems - packed_stems)
    if missing or extra:
        raise ValueError(
            f"splice_counterfactual selection shards at {selection_dir} must "
            f"match packed shard stems exactly; missing={missing[:5]}, "
            f"extra={extra[:5]}."
        )
    completed = None
    if feature_clusters is not None:
        raw_resume = meta.get(SPLICE_RESUME_METADATA_KEY)
        if raw_resume is not None:
            raw_integrity = meta.get(SELECTION_SHARD_INTEGRITY_KEY)
            if not isinstance(raw_integrity, Mapping):
                raise FeatureClustersError(
                    "splice_counterfactual selection manifest "
                    f"{selection_dir} has {SPLICE_RESUME_METADATA_KEY} but no valid "
                    f"{SELECTION_SHARD_INTEGRITY_KEY} metadata"
                )
            completed = parse_completed_splice_validation(
                meta,
                selection_integrity=raw_integrity,
                packed_row_fingerprint=packed_info.packed_row_fingerprint,
                effective_buckets=feature_pipeline.effective_buckets,
                feature_clusters_descriptor=(
                    feature_clusters.resolved.descriptor_dict()
                ),
                expected_stems=sel_stems,
                surface=f"splice_counterfactual selection manifest {selection_dir}",
            )
    if completed is not None:
        logger.info(
            "PrecomputedBatchLoader: clustered splice selection bodies already "
            "validated by the completed producer certificate; skipping redundant "
            f"semantic body scan for {len(sel_stems)} shards"
        )
        return completed.validated_max_features_per_segment
    if remote_materializer is not None:
        # The strict remote inventory + loader certificate establishes the
        # complete namespace without eagerly opening bodies.  Each body runs
        # the same semantic checks immediately after materialization below.
        return None
    packed_by_stem = {p.stem.split(".")[0]: p for p in shard_paths}
    max_features_per_segment = 0
    with Heartbeat(
        f"splice selection body scan ({len(sel_shard_paths)} shards)"
    ) as heartbeat:
        for scanned, (stem, sel_path) in enumerate(sel_shard_paths.items()):
            heartbeat.tick(f"{scanned}/{len(sel_shard_paths)} shards scanned")
            packed = load_packed_row_shard(packed_by_stem[stem])
            sel = load_selection_shard(sel_path)
            width = _max_selected_features_per_segment(
                sel.sel_offsets,
                sel.neg_offsets,
            )
            max_features_per_segment = max(
                max_features_per_segment,
                width,
            )
            if feature_clusters is not None:
                feature_clusters.assert_selection_shard(
                    sel_offsets=sel.sel_offsets,
                    sel_fids=sel.sel_fids,
                    sel_buckets=sel.sel_buckets,
                    sel_pcts=sel.sel_pcts,
                    neg_offsets=sel.neg_offsets,
                    neg_fids=sel.neg_fids,
                    expected_segments=packed.total_segments,
                    effective_buckets=feature_pipeline.effective_buckets,
                    surface=f"splice_counterfactual selection shard {sel_path.name}",
                )
            elif sel.total_segments != packed.total_segments:
                raise ValueError(
                    f"splice_counterfactual selection shard {sel_path.name} has "
                    f"{sel.total_segments} segments but packed shard "
                    f"{packed_by_stem[stem].name} has {packed.total_segments}."
                )
    return max_features_per_segment


def _validate_generic_selection_dir(
    *,
    packed_row_dir: Path,
    shard_paths: list[Path],
    selection_dir: Path,
    sel_shard_paths: dict[str, Path],
    feature_pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation | None,
    remote_materializer: RemoteSelectionMaterializer | None = None,
) -> int | None:
    """Validate generic selection and return its global CSR width when local.

    Returns the validated global positive-plus-negative CSR width, or ``None``
    on the remote-materializer path.
    """
    if not sel_shard_paths:
        raise ValueError(f"selection_dir {selection_dir} contains no selection shards.")
    manifest = read_manifest(selection_dir)
    if manifest is None:
        raise ValueError(
            f"selection_dir {selection_dir} is missing manifest.json; re-run "
            "precompute_selected_features."
        )
    meta = manifest.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError(
            f"selection_dir {selection_dir} has invalid manifest metadata; "
            "re-run precompute_selected_features."
        )
    assert_feature_clusters_selection_metadata(
        feature_clusters,
        meta,
        surface=f"selection manifest {selection_dir}",
    )
    if feature_clusters is not None and remote_materializer is None:
        assert_clustered_selection_shard_integrity(
            selection_dir,
            meta,
            surface=f"selection manifest {selection_dir}",
        )
    configured_delta = None
    if feature_pipeline.delta_bundle is not None:
        configured_delta = canonical_feature_delta_dict(feature_pipeline.delta_bundle.config)
    manifest_delta = meta.get("feature_delta")
    if manifest_delta != configured_delta:
        raise ValueError(
            f"selection manifest feature_delta mismatch for {selection_dir}: "
            f"manifest has {manifest_delta!r}, configured {configured_delta!r}. "
            "Do not reuse selection shards across feature_delta modes; re-run "
            "precompute_selected_features with the current config."
        )
    configured_mask = canonical_dict(feature_pipeline.feature_mask)
    manifest_mask = meta.get("feature_mask")
    if manifest_mask != configured_mask:
        raise ValueError(
            f"selection manifest feature_mask mismatch for {selection_dir}: "
            f"manifest has {manifest_mask!r}, configured {configured_mask!r}. "
            "Re-run precompute_selected_features with the current config."
        )
    manifest_packed = meta.get("packed_rows")
    expected_packed_hash = packed_row_fingerprint(
        packed_row_dir,
        cache_dir=packed_row_dir,
    )
    actual_packed_hash = (
        manifest_packed.get("packed_row_fingerprint") if isinstance(manifest_packed, dict) else None
    )
    if actual_packed_hash != expected_packed_hash:
        raise ValueError(
            f"selection manifest packed row fingerprint mismatch for "
            f"{selection_dir}: manifest has {actual_packed_hash!r}, current "
            f"packed rows hash to {expected_packed_hash!r}. Re-run "
            "precompute_selected_features after regenerating packed rows."
        )
    packed_stems = {p.stem.split(".")[0] for p in shard_paths}
    sel_stems = set(sel_shard_paths)
    missing = sorted(packed_stems - sel_stems)
    extra = sorted(sel_stems - packed_stems)
    if missing or extra:
        raise ValueError(
            f"selection shards at {selection_dir} must match packed shard stems "
            f"exactly; missing={missing[:5]}, extra={extra[:5]}."
        )
    if remote_materializer is not None:
        return None
    if (
        feature_clusters is not None
        and CLUSTERED_SELECTION_VALIDATION_KEY in meta
        and is_sha256_hex(selection_dir.name)
    ):
        authority = parse_clustered_selection_validation(
            meta[CLUSTERED_SELECTION_VALIDATION_KEY],
            selection_checksum=selection_dir.name,
            selection_integrity=meta[SELECTION_SHARD_INTEGRITY_KEY],
            packed_row_fingerprint=expected_packed_hash,
            feature_clusters_descriptor=feature_clusters.resolved.descriptor_dict(),
            feature_delta=configured_delta,
            effective_buckets=feature_pipeline.effective_buckets,
            surface=f"selection manifest {selection_dir}",
        )
        logger.info(
            "PrecomputedBatchLoader: clustered selection bodies already validated "
            "by the schema-v1 certificate; skipping redundant semantic body scan "
            f"for {len(sel_stems)} shards"
        )
        return authority.max_features_per_segment
    packed_by_stem = {p.stem.split(".")[0]: p for p in shard_paths}
    max_features_per_segment = 0
    for stem, sel_path in sel_shard_paths.items():
        packed = load_packed_row_shard(packed_by_stem[stem])
        sel = load_selection_shard(sel_path)
        width = _max_selected_features_per_segment(
            sel.sel_offsets,
            sel.neg_offsets,
        )
        max_features_per_segment = max(
            max_features_per_segment,
            width,
        )
        if feature_clusters is not None:
            feature_clusters.assert_selection_shard(
                sel_offsets=sel.sel_offsets,
                sel_fids=sel.sel_fids,
                sel_buckets=sel.sel_buckets,
                sel_pcts=sel.sel_pcts,
                neg_offsets=sel.neg_offsets,
                neg_fids=sel.neg_fids,
                expected_segments=packed.total_segments,
                effective_buckets=feature_pipeline.effective_buckets,
                surface=f"selection shard {sel_path.name}",
            )
        elif sel.total_segments != packed.total_segments:
            raise ValueError(
                f"selection shard {sel_path.name} has {sel.total_segments} "
                f"segments but packed shard {packed_by_stem[stem].name} has "
                f"{packed.total_segments}."
            )
    return max_features_per_segment


class PrecomputedValidationSetupDriftError(RuntimeError):
    """A retained local validation setup no longer matches its authority."""


def _validation_absolute_path(path: Path) -> Path:
    """Return a lexical absolute path without dereferencing symlinks."""
    return Path(os.path.abspath(path))


_ValidationStatFields = tuple[int, int, int, int, int, int]
_ValidationArtifactSignature = tuple[Path, _ValidationStatFields, _ValidationStatFields]
_ValidationAuthorityFile = tuple[Path, _ValidationArtifactSignature | None, str | None]
_PrecomputedValidationAuthority = tuple[
    tuple[_ValidationAuthorityFile, ...],
    tuple[_ValidationArtifactSignature, ...],
    tuple[_ValidationArtifactSignature, ...],
]


def _validation_stat_fields(
    result: os.stat_result,
) -> _ValidationStatFields:
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_size),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _capture_validation_artifact(
    path: Path,
    *,
    directory: bool = False,
) -> _ValidationArtifactSignature:
    absolute = _validation_absolute_path(path)
    try:
        link_result = absolute.lstat()
        target_result = absolute.stat()
    except OSError as exc:
        raise PrecomputedValidationSetupDriftError(
            f"retained validation artifact is unavailable: {absolute}: {exc}"
        ) from exc
    kind = "directory" if directory else "file"
    kind_matches = (
        stat.S_ISDIR(target_result.st_mode) if directory else stat.S_ISREG(target_result.st_mode)
    )
    if not kind_matches:
        raise PrecomputedValidationSetupDriftError(
            f"retained validation artifact is not a {kind}: {absolute}"
        )
    return (
        absolute,
        _validation_stat_fields(link_result),
        _validation_stat_fields(target_result),
    )


def _capture_validation_authority_file(path: Path) -> _ValidationAuthorityFile:
    """Hash one small authority file, preserving the optional-manifest case."""
    absolute = _validation_absolute_path(path)
    try:
        absolute.lstat()
    except FileNotFoundError:
        return (absolute, None, None)
    except OSError as exc:
        raise PrecomputedValidationSetupDriftError(
            f"retained validation artifact is unavailable: {absolute}: {exc}"
        ) from exc
    before = _capture_validation_artifact(absolute)
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise PrecomputedValidationSetupDriftError(
            f"cannot read retained validation authority {absolute}: {exc}"
        ) from exc
    if _capture_validation_artifact(absolute) != before:
        raise PrecomputedValidationSetupDriftError(
            f"validation authority changed while it was read: {absolute}"
        )
    return (
        absolute,
        before,
        hashlib.sha256(payload).hexdigest(),
    )


def _capture_precomputed_validation_authority(
    *,
    packed_row_dir: Path,
    selection_dir: Path,
    shard_paths: tuple[Path, ...],
    selection_paths: tuple[Path, ...],
    splice_feature_dir: Path | None,
) -> _PrecomputedValidationAuthority:
    """Capture manifests and direct bodies; trust validated lineage bodies."""
    packed_row_dir = _validation_absolute_path(packed_row_dir)
    selection_dir = _validation_absolute_path(selection_dir)
    packed_paths = tuple(_validation_absolute_path(path) for path in shard_paths)
    selected_paths = tuple(
        sorted(
            (_validation_absolute_path(path) for path in selection_paths),
            key=os.fspath,
        )
    )
    inventories = (
        (packed_row_dir, packed_paths),
        (selection_dir, selected_paths),
    )
    for directory, expected_paths in inventories:
        current_paths = tuple(
            _validation_absolute_path(path) for path in glob_overlay_shards(directory)
        )
        if current_paths != expected_paths:
            raise PrecomputedValidationSetupDriftError(
                f"local validation shard inventory changed: {directory}"
            )
    body_paths = sorted(
        {path for _directory, paths in inventories for path in paths},
        key=os.fspath,
    )
    directory_paths: list[Path] = []
    if splice_feature_dir is not None:
        body_paths.extend(
            _validation_absolute_path(path)
            for path in glob_overlay_shards(provenance_dir(packed_row_dir))
        )
        splice_feature_dir = _validation_absolute_path(splice_feature_dir)
        directory_paths.append(splice_feature_dir)
        sidecar_dir = splice_feature_dir / "_splice_provenance"
        if sidecar_dir.exists():
            directory_paths.append(sidecar_dir)
    return (
        tuple(
            _capture_validation_authority_file(path)
            for path in (
                packed_row_dir / "manifest.json",
                selection_dir / "manifest.json",
            )
        ),
        tuple(_capture_validation_artifact(path) for path in body_paths),
        tuple(_capture_validation_artifact(path, directory=True) for path in directory_paths),
    )


@dataclass(frozen=True)
class _PrecomputedValidationSetup:
    request_identity: str
    packed_row_dir: Path
    selection_dir: Path
    splice_feature_dir: Path | None
    shard_paths: tuple[Path, ...]
    selection_paths: tuple[Path, ...]
    pad_n_seg: int | None
    max_seg_size: int | None
    pad_f: int | None
    authority: _PrecomputedValidationAuthority


class PrecomputedValidationSetupCache:
    """Own one immutable local validation setup and no traversal state.

    This object is deliberately process-local, per validation spec, and absent
    from checkpoints. Every call to :func:`precomputed_data_loader` still owns
    a new generator, epoch, pending-row list, RNG replay state, decoded-shard
    cache, and traversal cursor.
    """

    def __init__(self) -> None:
        self._setup: _PrecomputedValidationSetup | None = None

    def _lookup(self, request_identity: str) -> _PrecomputedValidationSetup | None:
        setup = self._setup
        if setup is None:
            return None
        if setup.request_identity != request_identity:
            raise PrecomputedValidationSetupDriftError(
                "a precomputed validation setup cache was reused for a different "
                "artifact or configuration identity"
            )
        current = _capture_precomputed_validation_authority(
            packed_row_dir=setup.packed_row_dir,
            selection_dir=setup.selection_dir,
            shard_paths=setup.shard_paths,
            selection_paths=setup.selection_paths,
            splice_feature_dir=setup.splice_feature_dir,
        )
        if current != setup.authority:
            raise PrecomputedValidationSetupDriftError(
                "retained precomputed validation setup authority changed; "
                "restart from quiescent, immutable artifacts so the complete "
                "canonical validation runs again"
            )
        return setup

    def _publish(self, setup: _PrecomputedValidationSetup) -> None:
        if self._setup is not None:
            raise RuntimeError("precomputed validation setup was published more than once")
        self._setup = setup


def _precomputed_validation_request_identity(
    *,
    packed_row_dir: Path,
    selection_dir: Path,
    splice_feature_dir: Path | None,
    feature_pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation | None,
    effective_buckets: int,
) -> str:
    delta = (
        canonical_feature_delta_dict(feature_pipeline.delta_bundle.config)
        if feature_pipeline.delta_bundle is not None
        else None
    )
    payload = {
        "packed_row_dir": os.fspath(_validation_absolute_path(packed_row_dir)),
        "selection_dir": os.fspath(_validation_absolute_path(selection_dir)),
        "splice_feature_dir": (
            os.fspath(_validation_absolute_path(splice_feature_dir))
            if splice_feature_dir is not None
            else None
        ),
        "feature_mask": canonical_dict(feature_pipeline.feature_mask),
        "feature_delta": delta,
        "feature_clusters": (
            feature_clusters.resolved.descriptor_dict() if feature_clusters is not None else None
        ),
        "loader_effective_buckets": effective_buckets,
        "pipeline_effective_buckets": feature_pipeline.effective_buckets,
        "boundary_fingerprint": (
            tensor_content_fingerprint(feature_pipeline.boundaries) if delta is not None else None
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def precomputed_data_loader(
    packed_row_dir: str | Path,
    B: int,
    feature_pipeline: "FeaturePipelineConfig",
    device: str = "cuda",
    label_dropout: float = 0.0,
    selection_dir: str | Path | None = None,
    effective_buckets: int = 10,
    n_sae_features: int | None = None,
    fp_target_layout: FeaturePredictionTargetLayout | None = None,
    pad_F: int | None = None,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    resume_state_dict: dict | None = None,
    splice_feature_dir: str | Path | None = None,
    feature_clusters: FeatureClusterConsolidation | None = None,
    remote_materializer: RemoteSelectionMaterializer | None = None,
    packed_materializer: RemotePackedRowMaterializer | None = None,
    packed_identity_materializer: RemotePackedRowMaterializer | None = None,
    remote_derived_feature_fingerprint: str | None = None,
    remote_read_cancel_event: threading.Event | None = None,
    train_shard_lookahead: int = 0,
    node_cache: NodeLocalShardCache | None = None,
    validation_setup_cache: PrecomputedValidationSetupCache | None = None,
    include_splice_mask: bool = False,
):
    """Load precomputed packed row shards and yield batches.

    Reads packed row shards from *packed_row_dir*, groups rows into batches of
    size B, and yields 10-tuple batches for training.

    Feature selection is always applied, via one of two paths:
    - **Selection shards** (*selection_dir* provided): loads pre-selected
      features with buckets from ``sel/<hash>/`` shards on disk.
    - **Fused inline** (no *selection_dir*): applies *feature_pipeline*
      selection directly from CSR data via ``reconstruct_and_select()``.

    With DDP (``ddp_world_size > 1``), each rank processes a disjoint subset
    of rows via row-level striping: rank *k* takes rows where
    ``row_idx % ddp_world_size == ddp_rank``.

    A new-format state from this loader is byte-exact when resumed with the
    same DDP topology and unchanged artifact family: ``batch_rows`` rebuilds
    cross-shard/cross-epoch pending rows and the RNG snapshots replay every
    stochastic batch decision. Legacy states without these fields remain
    resumable through the historical shard/offset fallback, with a warning and
    no retroactive exactness claim.

    ``train_shard_lookahead`` banks decoded base/selection shard pairs on a
    bounded background thread. Zero preserves inline loading; positive depths
    are supported only for the local packed-row path.

    ``validation_setup_cache`` is the narrow issue-#1615 local-validation seam.
    It retains immutable discovery and validation results only; every loader
    call still creates a fresh mutable traversal cursor. The trainer supplies
    it only for validation; this low-level boundary rejects resume,
    fused-selection, and materializer paths.
    """
    _validate_positive_world_size(ddp_rank, ddp_world_size)
    if include_splice_mask and validation_setup_cache is not None:
        raise ValueError("conditioning splice masks are training-only; disable validation_setup_cache")
    lookahead = resolve_train_shard_lookahead(train_shard_lookahead)
    _assert_shard_lookahead_supported(lookahead, packed_materializer, remote_materializer)
    if node_cache is not None and (
        packed_materializer is not None or remote_materializer is not None
    ):
        raise ValueError(
            "node-local shard cache is supported only on the local packed-row path; "
            "materializer paths already stage shard bodies through remote caches"
        )
    if validation_setup_cache is not None:
        unsupported_reasons: list[str] = []
        if selection_dir is None:
            unsupported_reasons.append("selection_dir is absent")
        if label_dropout != 0.0:
            unsupported_reasons.append("label_dropout is nonzero")
        if resume_state_dict is not None:
            unsupported_reasons.append("resume_state_dict is present")
        if n_sae_features is not None or fp_target_layout is not None:
            unsupported_reasons.append("feature-prediction targets are active")
        if packed_materializer is not None or packed_identity_materializer is not None:
            unsupported_reasons.append("packed materialization is active")
        if remote_materializer is not None:
            unsupported_reasons.append("selection materialization is active")
        if remote_derived_feature_fingerprint is not None:
            unsupported_reasons.append("remote derived-feature identity is active")
        if remote_read_cancel_event is not None:
            unsupported_reasons.append("remote cancellation is active")
        if lookahead != 0:
            unsupported_reasons.append("shard lookahead is active")
        if unsupported_reasons:
            raise ValueError(
                "validation_setup_cache supports only local, selection-backed, "
                "cursor-fresh validation: " + ", ".join(unsupported_reasons)
            )

    # Per-node decode-once sharing engages only on the local packed path
    # (materializer paths stream bytes and keep their existing behavior) and
    # only under torchrun with more than one local rank — a single process
    # has nobody to share with.
    shared_decode_topology = (
        _shared_decode_local_topology()
        if packed_materializer is None and remote_materializer is None
        else None
    )
    if shared_decode_topology is not None and shared_decode_topology[1] <= 1:
        shared_decode_topology = None
    shared_decode_retain = _shared_decode_retain_for_lookahead(lookahead)
    if lookahead > 0:
        storage_scope = (
            "per-node shared" if shared_decode_topology is not None else "process-private"
        )
        _log_shard_lookahead_budget(
            lookahead,
            f"train_shard_lookahead={lookahead}: banked shard pairs={lookahead}; "
            f"live decoded shards={lookahead + 3} ({storage_scope}); shared-decode "
            f"retain window={shared_decode_retain}; see docs/pipelines/"
            "encoder-decoder-training.md#shard-level-lookahead-train_shard_lookahead",
        )
    _validate_resume_world_size(
        resume_state_dict,
        ddp_world_size,
        context="precomputed dataloader resume",
    )
    if resume_state_dict is not None:
        saved_batch_size = resume_state_dict.get("device_batch_size")
        raw_batch_rows = resume_state_dict.get("batch_rows")
        if raw_batch_rows is not None:
            if not isinstance(raw_batch_rows, list) or not raw_batch_rows:
                raise ValueError(
                    "PrecomputedBatchLoader: exact resume batch_rows must be a non-empty list"
                )
            if saved_batch_size is None:
                if len(raw_batch_rows) != B:
                    raise ValueError(
                        "PrecomputedBatchLoader: device-batch migration requires "
                        "checkpointed device_batch_size"
                    )
                saved_batch_size = len(raw_batch_rows)
            if type(saved_batch_size) is not int or saved_batch_size <= 0:
                raise ValueError(
                    "PrecomputedBatchLoader: saved device_batch_size must be a "
                    f"positive strict int, got {saved_batch_size!r}"
                )
            if len(raw_batch_rows) != saved_batch_size:
                raise ValueError(
                    "PrecomputedBatchLoader: batch_rows length must equal saved "
                    f"device_batch_size={saved_batch_size}, got {len(raw_batch_rows)}"
                )
            if saved_batch_size != B and "ddp_world_size" not in resume_state_dict:
                raise ValueError(
                    "PrecomputedBatchLoader: device-batch migration requires "
                    "checkpointed ddp_world_size topology"
                )
        elif saved_batch_size is not None and saved_batch_size != B:
            raise ValueError(
                "PrecomputedBatchLoader: device-batch migration requires exact "
                "batch_rows; legacy cursor states cannot be safely re-batched"
            )
    if fp_target_layout is not None and n_sae_features is not None:
        raise ValueError(
            "precomputed_data_loader accepts fp_target_layout or the legacy "
            "n_sae_features argument, not both"
        )
    if remote_materializer is not None:
        if selection_dir is None:
            raise ValueError(
                "precomputed_data_loader remote materialization requires selection_dir"
            )
        if Path(selection_dir).resolve() != remote_materializer.local_dir.resolve():
            raise ValueError(
                "precomputed_data_loader selection_dir does not match the remote cache"
            )
        if feature_clusters is None:
            raise ValueError(
                "remote selection materialization is restricted to clustered selections"
            )
    if packed_materializer is not None:
        if remote_materializer is None:
            raise ValueError(
                "precomputed_data_loader packed materialization requires remote "
                "selection materialization"
            )
        if selection_dir is None:
            raise ValueError(
                "precomputed_data_loader packed materialization requires selection_dir"
            )
        if feature_clusters is None:
            raise ValueError("remote packed materialization is restricted to clustered selections")
        if Path(packed_row_dir).resolve() != packed_materializer.local_dir.resolve():
            raise ValueError(
                "precomputed_data_loader packed_row_dir does not match the remote packed cache"
            )
        if remote_derived_feature_fingerprint is None:
            raise ValueError(
                "precomputed_data_loader packed materialization requires the "
                "verified remote derived-feature fingerprint"
            )
    if remote_derived_feature_fingerprint is not None:
        if (
            not isinstance(remote_derived_feature_fingerprint, str)
            or len(remote_derived_feature_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in remote_derived_feature_fingerprint)
        ):
            raise ValueError(
                "remote_derived_feature_fingerprint must be 64 lowercase hexadecimal characters"
            )
        if remote_materializer is None:
            raise ValueError(
                "remote_derived_feature_fingerprint requires a verified remote "
                "selection materializer"
            )
    if fp_target_layout is not None and (
        fp_target_layout.feature_mask != feature_pipeline.feature_mask
    ):
        raise ValueError(
            "precomputed_data_loader received an FP target layout derived from a "
            "different FeatureMask than feature_pipeline.feature_mask"
        )
    if fp_target_layout is not None and (fp_target_layout.feature_clusters != feature_clusters):
        raise ValueError(
            "precomputed_data_loader received an FP target layout derived from a "
            "different FeatureClusterConsolidation"
        )

    target_source_width = (
        fp_target_layout.n_sae_features if fp_target_layout is not None else n_sae_features
    )
    delta_mode = (
        feature_pipeline.delta_bundle.config.mode
        if feature_pipeline.delta_bundle is not None
        else None
    )
    if (
        remote_derived_feature_fingerprint is not None
        and delta_mode is not DeltaMode.splice_counterfactual
    ):
        raise ValueError(
            "remote_derived_feature_fingerprint requires feature_delta.mode=splice_counterfactual"
        )
    cluster_delta_semantics = (
        CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
        if feature_clusters is not None and delta_mode is DeltaMode.splice_counterfactual
        else None
    )
    if feature_clusters is not None:
        if selection_dir is None:
            raise ValueError(
                "precomputed_data_loader requires selection_dir when "
                "feature_clusters is active; clustered semantics exist only in "
                "precomputed selection shards and never fall back to fused selection"
            )
        if feature_pipeline.feature_mask != feature_clusters.feature_mask:
            raise ValueError(
                "precomputed_data_loader received feature_clusters derived from a "
                "different FeatureMask than feature_pipeline.feature_mask"
            )
        if (
            target_source_width is not None
            and target_source_width != feature_clusters.n_sae_features
        ):
            raise ValueError(
                "precomputed_data_loader n_sae_features does not match the static "
                f"cluster descriptor: {target_source_width} != "
                f"{feature_clusters.n_sae_features}"
            )
    if delta_mode is not None and delta_mode is not DeltaMode.splice_counterfactual:
        raise ValueError(
            "precomputed_data_loader does not support feature_delta. "
            "Packed row shards do not currently preserve the live dataloader's "
            "document-history state needed for delta conditioning. Set "
            "feature_delta=null for precomputed packed-row training, or use "
            "the live dataloader with use_precomputed_batches=false."
        )

    packed_row_dir = Path(packed_row_dir)
    selection_dir = Path(selection_dir) if selection_dir is not None else None
    splice_feature_dir_path = Path(splice_feature_dir) if splice_feature_dir is not None else None
    validated_splice_feature_dir = (
        splice_feature_dir_path if delta_mode is DeltaMode.splice_counterfactual else None
    )
    caller_pad_f = pad_F
    identity_materializer = packed_identity_materializer or packed_materializer
    request_identity: str | None = None
    retained_setup: _PrecomputedValidationSetup | None = None
    if validation_setup_cache is not None:
        assert selection_dir is not None
        request_identity = _precomputed_validation_request_identity(
            packed_row_dir=packed_row_dir,
            selection_dir=selection_dir,
            splice_feature_dir=validated_splice_feature_dir,
            feature_pipeline=feature_pipeline,
            feature_clusters=feature_clusters,
            effective_buckets=effective_buckets,
        )
        retained_setup = validation_setup_cache._lookup(request_identity)

    fp_source_certificate_digest: str | None = None
    if retained_setup is not None:
        shard_paths = list(retained_setup.shard_paths)
        sel_shard_paths = {path.stem.split(".")[0]: path for path in retained_setup.selection_paths}
        pad_n_seg = retained_setup.pad_n_seg
        max_seg_size = retained_setup.max_seg_size
        pad_F = retained_setup.pad_f
        logger.info(
            "PrecomputedBatchLoader: reusing retained local validation setup "
            f"for {selection_dir} ({len(shard_paths)} packed, "
            f"{len(sel_shard_paths)} selection shards)"
        )
    else:
        shard_paths = (
            packed_materializer.shard_paths()
            if packed_materializer is not None
            else glob_overlay_shards(packed_row_dir)
        )
        if not shard_paths:
            raise FileNotFoundError(f"No packed row shards found in {packed_row_dir}")

        if (
            packed_materializer is not None
            and packed_identity_materializer is not None
            and packed_materializer.inventory != packed_identity_materializer.inventory
        ):
            raise ValueError(
                "packed_materializer and packed_identity_materializer disagree on "
                "the authenticated packed-row inventory"
            )

        if fp_target_layout is not None and fp_target_layout.is_compact:
            if identity_materializer is not None:
                _, fp_source_certificate_digest = validate_fp_target_source_certificate_remote(
                    packed_row_dir,
                    n_sae_features=fp_target_layout.n_sae_features,
                    inventory=identity_materializer.inventory,
                )
            else:
                _, fp_source_certificate_digest = validate_fp_target_source_certificate(
                    packed_row_dir,
                    n_sae_features=fp_target_layout.n_sae_features,
                )

        sel_shard_paths: dict[str, Path] = {}
        if selection_dir is not None:
            if remote_materializer is not None:
                sel_shard_paths = remote_materializer.shard_paths()
            else:
                for sp in glob_overlay_shards(selection_dir):
                    sel_shard_paths[sp.stem.split(".")[0]] = sp
            if sel_shard_paths:
                logger.info(
                    f"PrecomputedBatchLoader: {len(sel_shard_paths)} selection "
                    f"shards in {selection_dir}"
                )

        authority_before: _PrecomputedValidationAuthority | None = None
        if validation_setup_cache is not None:
            assert selection_dir is not None
            authority_before = _capture_precomputed_validation_authority(
                packed_row_dir=packed_row_dir,
                selection_dir=selection_dir,
                shard_paths=tuple(shard_paths),
                selection_paths=tuple(sel_shard_paths.values()),
                splice_feature_dir=validated_splice_feature_dir,
            )

        # Read manifest for segment/shape metadata (max_n_seg, max_seg_size).
        if identity_materializer is not None:
            identity_materializer.assert_manifest_current()
        manifest = read_manifest(packed_row_dir)
        meta = manifest.get("metadata", {}) if manifest else {}
        pad_n_seg = meta.get("max_n_seg")
        max_seg_size = meta.get("max_seg_size")

        validated_max_f: int | None = None
        if delta_mode is DeltaMode.splice_counterfactual:
            validated_max_f = _validate_splice_counterfactual_selection_dir(
                packed_row_dir=packed_row_dir,
                shard_paths=shard_paths,
                selection_dir=selection_dir,
                sel_shard_paths=sel_shard_paths,
                feature_pipeline=feature_pipeline,
                splice_feature_dir=validated_splice_feature_dir,
                feature_clusters=feature_clusters,
                remote_materializer=remote_materializer,
                packed_materializer=identity_materializer,
                remote_derived_feature_fingerprint=(remote_derived_feature_fingerprint),
            )
        elif selection_dir is not None:
            validated_max_f = _validate_generic_selection_dir(
                packed_row_dir=packed_row_dir,
                shard_paths=shard_paths,
                selection_dir=selection_dir,
                sel_shard_paths=sel_shard_paths,
                feature_pipeline=feature_pipeline,
                feature_clusters=feature_clusters,
                remote_materializer=remote_materializer,
            )

        # pad_F sources (in priority order):
        # 1. Remote authenticated selection inventory
        # 2. Local selection validation (certificate or legacy body scan)
        # 3. Caller-provided pad_F: from config max_k bounds (fused path)
        # 4. None: no static shape padding (base shard F is ~8k, too large to pad to)
        pad_F = None
        if remote_materializer is not None:
            # MUST precede validated width: a splice certificate returns a width
            # even remotely, but remote pad_F remains authenticated-inventory-owned.
            pad_F = remote_materializer.max_features_per_segment
            logger.info(
                f"PrecomputedBatchLoader: max_F={pad_F} from certified remote selection inventory"
            )
        elif validated_max_f is not None and sel_shard_paths:
            pad_F = max(validated_max_f, 1)
            logger.info(f"PrecomputedBatchLoader: max_F={pad_F} from selection validation")
        elif caller_pad_f is not None:
            # Fused path: use caller-provided pad_F from config max_k bounds.
            pad_F = caller_pad_f
            logger.info(f"PrecomputedBatchLoader: using caller pad_F={pad_F} for fused path")
        else:
            # No selection shards and no caller pad_F — disable static shape padding.
            pad_n_seg = None
            max_seg_size = None

        if validation_setup_cache is not None:
            assert authority_before is not None
            assert request_identity is not None
            assert selection_dir is not None
            authority_after = _capture_precomputed_validation_authority(
                packed_row_dir=packed_row_dir,
                selection_dir=selection_dir,
                shard_paths=tuple(shard_paths),
                selection_paths=tuple(sel_shard_paths.values()),
                splice_feature_dir=validated_splice_feature_dir,
            )
            if authority_after != authority_before:
                raise PrecomputedValidationSetupDriftError(
                    "local validation manifests, direct bodies, or lineage "
                    "directories changed during the complete canonical validation; "
                    "retry from quiescent, immutable artifacts"
                )
            validation_setup_cache._publish(
                _PrecomputedValidationSetup(
                    request_identity=request_identity,
                    packed_row_dir=_validation_absolute_path(packed_row_dir),
                    selection_dir=_validation_absolute_path(selection_dir),
                    splice_feature_dir=(
                        _validation_absolute_path(validated_splice_feature_dir)
                        if validated_splice_feature_dir is not None
                        else None
                    ),
                    shard_paths=tuple(_validation_absolute_path(path) for path in shard_paths),
                    selection_paths=tuple(
                        sorted(
                            (_validation_absolute_path(path) for path in sel_shard_paths.values()),
                            key=os.fspath,
                        )
                    ),
                    pad_n_seg=pad_n_seg,
                    max_seg_size=max_seg_size,
                    pad_f=pad_F,
                    authority=authority_after,
                )
            )
            logger.info(
                "PrecomputedBatchLoader: retained complete local validation "
                f"setup for process-lifetime reuse: {selection_dir}"
            )

    if pad_n_seg is not None and pad_F is not None:
        logger.info(
            f"PrecomputedBatchLoader: static shapes enabled — "
            f"max_n_seg={pad_n_seg}, max_F={pad_F}, max_seg_size={max_seg_size}"
        )

    if not sel_shard_paths:
        _assert_pipeline_supports_live_selection(
            feature_pipeline,
            "precomputed_data_loader without selection shards (fused "
            f"reconstruct_and_select over {packed_row_dir})",
        )
        logger.info("PrecomputedBatchLoader: using fused reconstruct_and_select path")

    cooldown_splice_index: ConditioningSpliceIndex | None = None
    if include_splice_mask:
        if splice_feature_dir_path is not None:
            cooldown_splice_index = ConditioningSpliceIndex(
                packed_row_dir,
                splice_feature_dir_path,
                packed_shard_names=[path.name for path in shard_paths],
            )
        validate_splice_identity_resume(
            resume_state_dict,
            cooldown_splice_index.identity if cooldown_splice_index is not None else None,
        )

    logger.info(f"PrecomputedBatchLoader: {len(shard_paths)} base shards in {packed_row_dir}")

    batch_idx = 0
    epoch = 1
    resume_shard: str | None = None
    resume_shard_batch_offset: int = 0

    if resume_state_dict is not None:
        resume_shard = resume_state_dict.get("shard")
        resume_shard_batch_offset = resume_state_dict.get("shard_batch_offset", 0)
        epoch = resume_state_dict.get("epoch", 1)
        batch_idx = resume_state_dict.get("batch_idx", 0)

        compact_fp_resume_keys = {
            FEATURE_VOCAB_META_KEY,
            "feature_prediction_width",
            "fp_target_source_certificate_sha256",
        }
        present_compact_fp_keys = compact_fp_resume_keys.intersection(resume_state_dict)
        if fp_target_layout is not None and fp_target_layout.is_compact:
            if present_compact_fp_keys != compact_fp_resume_keys:
                raise ValueError(
                    "PrecomputedBatchLoader: compact-FP resume metadata is "
                    "missing or partial. expected keys="
                    f"{sorted(compact_fp_resume_keys)}, present="
                    f"{sorted(present_compact_fp_keys)}. Start fresh or restore "
                    "the original complete dataloader state."
                )
            assert fp_target_layout.feature_vocab is not None
            expected_vocab_metadata = fp_target_layout.feature_vocab.metadata_dict()
            if resume_state_dict[FEATURE_VOCAB_META_KEY] != expected_vocab_metadata:
                raise ValueError(
                    "PrecomputedBatchLoader: compact-FP feature vocabulary "
                    "mismatch on resume. state="
                    f"{resume_state_dict[FEATURE_VOCAB_META_KEY]!r}, configured="
                    f"{expected_vocab_metadata!r}"
                )
            if resume_state_dict["feature_prediction_width"] != fp_target_layout.output_width:
                raise ValueError(
                    "PrecomputedBatchLoader: compact-FP output width mismatch "
                    "on resume. state="
                    f"{resume_state_dict['feature_prediction_width']!r}, "
                    f"configured={fp_target_layout.output_width}"
                )
            if (
                resume_state_dict["fp_target_source_certificate_sha256"]
                != fp_source_certificate_digest
            ):
                raise ValueError(
                    "PrecomputedBatchLoader: FP target source certificate "
                    "changed on resume. state="
                    f"{resume_state_dict['fp_target_source_certificate_sha256']!r}, "
                    f"current={fp_source_certificate_digest!r}"
                )
        elif present_compact_fp_keys:
            raise ValueError(
                "PrecomputedBatchLoader: resume state describes compact feature "
                "prediction but the configured loader is full-width or headless: "
                f"present keys={sorted(present_compact_fp_keys)}"
            )

        saved_feature_clusters = resume_state_dict.get("feature_clusters_descriptor")
        expected_feature_clusters = (
            feature_clusters.resolved.descriptor if feature_clusters is not None else None
        )
        parsed_saved_feature_clusters: FeatureClusterDescriptor | None = None
        if saved_feature_clusters is not None:
            try:
                parsed_saved_feature_clusters = FeatureClusterDescriptor.from_dict(
                    saved_feature_clusters
                )
            except FeatureClustersError as exc:
                raise FeatureClustersError(
                    "PrecomputedBatchLoader: invalid feature cluster descriptor "
                    f"in resume state: {exc}"
                ) from exc
        if parsed_saved_feature_clusters != expected_feature_clusters:
            saved_canonical = (
                None
                if parsed_saved_feature_clusters is None
                else parsed_saved_feature_clusters.canonical_dict()
            )
            expected_canonical = (
                None
                if expected_feature_clusters is None
                else expected_feature_clusters.canonical_dict()
            )
            raise ValueError(
                "PrecomputedBatchLoader: feature cluster identity mismatch on "
                f"resume. state={saved_canonical!r}, configured="
                f"{expected_canonical!r}. "
                "Restore the original cluster "
                "configuration and selection shards or start a new run."
            )

        saved_cluster_delta_semantics = resume_state_dict.get(CLUSTER_DELTA_SEMANTICS_KEY)
        if cluster_delta_semantics is None:
            cluster_delta_resume_matches = CLUSTER_DELTA_SEMANTICS_KEY not in resume_state_dict
        else:
            cluster_delta_resume_matches = saved_cluster_delta_semantics == cluster_delta_semantics
        if not cluster_delta_resume_matches:
            raise ValueError(
                "PrecomputedBatchLoader: cluster-delta semantics mismatch on "
                f"resume. state={saved_cluster_delta_semantics!r}, configured="
                f"{cluster_delta_semantics!r}. Restore a checkpoint produced "
                "with the current clustered splice selection semantics or start "
                "a new run."
            )

        # Defensive pad_F resume check (slice-10 map P1-E, manifest §4.15):
        # If the saved state_dict persisted pad_F, ensure it matches the value
        # we just recomputed from the shard set. Drift indicates the shard set
        # changed between save and resume — a static-compile graph traced for
        # the old pad_F would silently desync from the new shape.
        saved_pad_F = resume_state_dict.get("pad_F")
        if saved_pad_F is not None and pad_F is not None and saved_pad_F != pad_F:
            raise ValueError(
                f"PrecomputedBatchLoader: pad_F mismatch on resume. "
                f"resume_state_dict['pad_F']={saved_pad_F}, recomputed pad_F={pad_F}. "
                f"This means the shard set in {packed_row_dir} changed between "
                f"checkpoint save and resume (e.g., a partial shard set was "
                f"loaded, or new shards were added). The static-compile graph "
                f"traced for the old pad_F will desync from the new shape. "
                f"Either (a) restore the original shard set, (b) drop the saved "
                f"pad_F from resume_state_dict to accept the recomputed value, "
                f"or (c) recompile the model with the new pad_F."
            )
        saved_max_seg_size = resume_state_dict.get("max_seg_size")
        if (
            saved_max_seg_size is not None
            and max_seg_size is not None
            and saved_max_seg_size != max_seg_size
        ):
            raise ValueError(
                f"PrecomputedBatchLoader: max_seg_size mismatch on resume. "
                f"resume_state_dict['max_seg_size']={saved_max_seg_size}, "
                f"recomputed max_seg_size={max_seg_size}. See pad_F mismatch "
                f"explanation above (slice-10 map P1-E)."
            )

        logger.info(
            f"PrecomputedBatchLoader: resuming from shard={resume_shard}, "
            f"shard_batch_offset={resume_shard_batch_offset}, epoch={epoch}"
        )

    # A pending row carries both loaded tensors and its stable JSON identity.
    # (epoch, shard_name, base_shard, selection_shard, row_idx)
    PendingRow = tuple[int, str, PackedRowShard, Any, int]
    pending: list[PendingRow] = []
    canonical_dropout_stream: _CanonicalLabelDropoutStream | None = None

    def materialize_pending_batch(
        rows: list[PendingRow],
        shard_batch_idx: int,
        *,
        replay_label_rng_state: Any = None,
    ) -> tuple:
        """Reconstruct one batch without changing the existing fast/slow paths."""
        if packed_materializer is not None:
            packed_materializer.check_worker_errors()
        elif remote_materializer is not None:
            remote_materializer.check_worker_errors()
        feature_rng_state = _capture_feature_pipeline_rng_state(feature_pipeline)

        # Check if all rows come from the same loaded shard (common fast path).
        # Epoch is part of the logical shard identity. A fresh cross-epoch
        # boundary loads the same artifact into a new shard instance and takes
        # the per-row slow path; exact replay must preserve that stochastic
        # reconstruction/draw order even though its cache reuses one object.
        shards_in_batch = {(row_epoch, id(base)) for row_epoch, _, base, _, _ in rows}
        fp_targets = None
        if len(shards_in_batch) == 1:
            _row_epoch, _row_shard, s0, sel0, _row_idx = rows[0]
            row_indices = [row_idx for _, _, _, _, row_idx in rows]
            if sel0 is not None:
                with t.profiler.record_function("precomputed/reconstruct_batch"):
                    rb = reconstruct_batch(
                        s0,
                        row_indices,
                        sel0,
                        effective_buckets,
                        device=device,
                        pad_n_seg=pad_n_seg,
                        pad_F=pad_F,
                    )
                if n_sae_features is not None or fp_target_layout is not None:
                    with t.profiler.record_function("precomputed/build_fp_targets"):
                        idx = _build_batch_index(s0, row_indices, pad_n_seg=pad_n_seg)
                        fp_targets = build_fp_targets_from_shard(
                            s0,
                            idx,
                            n_sae_features,
                            device=device,
                            feature_mask=(
                                feature_pipeline.feature_mask if fp_target_layout is None else None
                            ),
                            fp_target_layout=fp_target_layout,
                        )
            else:
                with t.profiler.record_function("precomputed/reconstruct_and_select"):
                    rb, fp_targets = reconstruct_and_select(
                        s0,
                        row_indices,
                        feature_pipeline,
                        device=device,
                        pad_n_seg=pad_n_seg,
                        n_sae_features=n_sae_features,
                        pad_F=pad_F,
                        fp_target_layout=fp_target_layout,
                    )
            inputs, targets = rb.inputs, rb.targets
            enc_fids, enc_mags = rb.enc_fids, rb.enc_mags
            enc_mask, enc_pcts = rb.enc_mask, rb.enc_pcts
            seg_map, seg_doc_id = rb.seg_map, rb.seg_doc_id
        else:
            # Cross-shard/cross-epoch boundary: reconstruct each row and stack.
            parts: list[ReconstructedBatch] = []
            fp_parts: list[t.Tensor | None] = []
            for _row_epoch, _row_shard, base, selection, row_idx in rows:
                if selection is not None:
                    parts.append(
                        reconstruct_batch(
                            base,
                            [row_idx],
                            selection,
                            effective_buckets,
                            device=device,
                            pad_n_seg=pad_n_seg,
                            pad_F=pad_F,
                        )
                    )
                    if n_sae_features is not None or fp_target_layout is not None:
                        row_batch_idx = _build_batch_index(base, [row_idx], pad_n_seg=pad_n_seg)
                        fp_parts.append(
                            build_fp_targets_from_shard(
                                base,
                                row_batch_idx,
                                n_sae_features,
                                device=device,
                                feature_mask=(
                                    feature_pipeline.feature_mask
                                    if fp_target_layout is None
                                    else None
                                ),
                                fp_target_layout=fp_target_layout,
                            )
                        )
                    else:
                        fp_parts.append(None)
                else:
                    rb_part, fp_part = reconstruct_and_select(
                        base,
                        [row_idx],
                        feature_pipeline,
                        device=device,
                        pad_n_seg=pad_n_seg,
                        n_sae_features=n_sae_features,
                        pad_F=pad_F,
                        fp_target_layout=fp_target_layout,
                    )
                    parts.append(rb_part)
                    fp_parts.append(fp_part)
            (
                inputs,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                enc_pcts,
                seg_map,
                seg_doc_id,
                fp_targets,
            ) = _stack_batch_parts(parts, fp_parts, B, device, pad_n_seg, pad_F)

        if feature_pipeline.feature_mask is not None:
            feature_pipeline.feature_mask.assert_no_masked(enc_fids[enc_mask])
        if feature_clusters is not None:
            feature_clusters.assert_representatives_only(
                enc_fids[enc_mask],
                surface="precomputed training batch",
            )

        label_rng_state: dict[str, Any] | None = None
        label_rng_draw_offset: int | None = None
        if label_dropout > 0.0:
            if canonical_dropout_stream is not None:
                (
                    label_rng_state,
                    label_rng_draw_offset,
                    dropout_random,
                ) = canonical_dropout_stream.draw(B, enc_mask.shape[1])
                drop_mask = dropout_random < label_dropout
            else:
                if replay_label_rng_state is not None:
                    _restore_default_rng_state(
                        replay_label_rng_state,
                        enc_mask.device,
                        context="precomputed dataloader label-dropout resume",
                    )
                label_rng_state = _capture_default_rng_state(enc_mask.device)
                batch_n_seg = enc_mask.shape[1]
                drop_mask = t.rand(B, batch_n_seg, device=enc_mask.device) < label_dropout
            enc_mask[drop_mask] = False
        elif replay_label_rng_state is not None:
            # Present metadata is validated even if the resumed configuration no
            # longer draws dropout, but label_dropout=0 never mutates the RNG.
            _restore_default_rng_state(
                replay_label_rng_state,
                enc_mask.device,
                context="precomputed dataloader label-dropout resume",
                apply=False,
            )

        _safe_empty_row_fallback(enc_mask, feature_pipeline.feature_mask)

        row_epoch, row_shard_name, _base, _selection, _row_idx = rows[-1]
        state_dict: dict[str, Any] = {
            "batch_idx": batch_idx,
            "shard": row_shard_name,
            "shard_batch_offset": shard_batch_idx,
            "epoch": row_epoch,
            "max_seg_size": max_seg_size,
            "pad_F": pad_F,
            "ddp_world_size": ddp_world_size,
            "batch_rows": [
                [pending_epoch, pending_shard, pending_row_idx]
                for (
                    pending_epoch,
                    pending_shard,
                    _pending_base,
                    _pending_selection,
                    pending_row_idx,
                ) in rows
            ],
            "feature_pipeline_rng_state": feature_rng_state,
        }
        if include_splice_mask:
            splice_rows: list[list[bool]] = []
            for _epoch, shard_name, base, _selection, row_idx in rows:
                flags = (
                    cooldown_splice_index.row_flags(shard_name, base, row_idx)
                    if cooldown_splice_index is not None
                    else []
                )
                if len(flags) > enc_mask.shape[1]:
                    raise SpliceProvenanceError("cooldown splice mask exceeds materialized segments")
                splice_rows.append(flags + [False] * (enc_mask.shape[1] - len(flags)))
            state_dict["conditioning_splice_mask"] = splice_rows
            state_dict["conditioning_splice_identity"] = (
                cooldown_splice_index.identity if cooldown_splice_index is not None else None
            )
        if label_rng_state is not None:
            state_dict["label_dropout_rng_state"] = label_rng_state
        if canonical_dropout_stream is not None:
            assert label_rng_draw_offset is not None
            state_dict["label_dropout_draw_batch_size"] = canonical_dropout_stream.draw_batch_size
            state_dict["label_dropout_draw_offset"] = label_rng_draw_offset
        if feature_clusters is not None:
            state_dict["feature_clusters_descriptor"] = feature_clusters.resolved.descriptor_dict()
        if cluster_delta_semantics is not None:
            state_dict[CLUSTER_DELTA_SEMANTICS_KEY] = cluster_delta_semantics
        if fp_target_layout is not None and fp_target_layout.is_compact:
            assert fp_target_layout.feature_vocab is not None
            assert fp_source_certificate_digest is not None
            state_dict[FEATURE_VOCAB_META_KEY] = fp_target_layout.feature_vocab.metadata_dict()
            state_dict["feature_prediction_width"] = fp_target_layout.output_width
            state_dict["fp_target_source_certificate_sha256"] = fp_source_certificate_digest

        return (
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            enc_pcts,
            seg_map,
            seg_doc_id,
            fp_targets,
            state_dict,
        )

    shard_path_by_name = {path.name: path for path in shard_paths}
    shard_index_by_name = {path.name: index for index, path in enumerate(shard_paths)}
    exact_shard_cache: dict[str, tuple[PackedRowShard, Any]] = {}

    def load_base_shard(shard_path: Path, *, logical_epoch: int) -> PackedRowShard:
        if packed_materializer is None:
            if shared_decode_topology is not None:
                with t.profiler.record_function("precomputed/shared_decode_load"):
                    return _shared_decode_load_base_shard(
                        shard_path,
                        logical_epoch,
                        shared_decode_topology[0],
                        shared_decode_topology[1],
                        retain=shared_decode_retain,
                        node_cache=node_cache,
                    )
            direct_started = time.monotonic()
            base = _load_packed_row_shard_with_node_cache(shard_path, node_cache)
            _loader_health_record("direct_load", time.monotonic() - direct_started)
            return base
        packed_materializer.publish_cursor(shard_path, epoch=logical_epoch)
        packed_bytes = (
            packed_materializer.read(shard_path, epoch=logical_epoch)
            if remote_read_cancel_event is None
            else packed_materializer.read(
                shard_path,
                cancel_event=remote_read_cancel_event,
                epoch=logical_epoch,
            )
        )
        try:
            base = load_packed_row_shard_bytes(packed_bytes)
            packed_materializer.validate_loaded_shard(
                shard_path,
                total_segments=base.total_segments,
            )
        except Exception as exc:
            exc.add_note(
                "remote packed-body decode/validation context: "
                f"path={shard_path}, logical_epoch={logical_epoch}"
            )
            raise
        return base

    def read_remote_selection(selection_path: Path, *, logical_epoch: int) -> bytes:
        assert remote_materializer is not None
        if packed_materializer is None:
            remote_materializer.publish_cursor(selection_path, epoch=logical_epoch)
        if remote_read_cancel_event is None:
            return remote_materializer.read(selection_path, epoch=logical_epoch)
        return remote_materializer.read(
            selection_path,
            cancel_event=remote_read_cancel_event,
            epoch=logical_epoch,
        )

    def load_remote_selection_shard(
        selection_path: Path,
        *,
        logical_epoch: int,
        expected_segments: int,
    ) -> Any:
        assert remote_materializer is not None
        assert feature_clusters is not None
        selection_bytes = read_remote_selection(
            selection_path,
            logical_epoch=logical_epoch,
        )
        try:
            selection = load_selection_shard_bytes(selection_bytes)
            feature_clusters.assert_selection_shard(
                sel_offsets=selection.sel_offsets,
                sel_fids=selection.sel_fids,
                sel_buckets=selection.sel_buckets,
                sel_pcts=selection.sel_pcts,
                neg_offsets=selection.neg_offsets,
                neg_fids=selection.neg_fids,
                expected_segments=expected_segments,
                effective_buckets=effective_buckets,
                surface=f"remote selection shard {selection_path.name}",
            )
        except Exception as exc:
            exc.add_note(
                "remote selection-body decode/validation context: "
                f"path={selection_path}, logical_epoch={logical_epoch}"
            )
            raise
        return selection

    def load_exact_shard(shard_name: str, *, logical_epoch: int) -> tuple[PackedRowShard, Any]:
        if shard_name not in shard_path_by_name:
            raise ValueError(
                "PrecomputedBatchLoader: batch_rows references shard "
                f"{shard_name!r}, which is missing from the current inventory"
            )
        cached = exact_shard_cache.get(shard_name)
        if cached is not None:
            shard_path = shard_path_by_name[shard_name]
            stem = shard_path.stem.split(".")[0]
            if packed_materializer is not None:
                packed_materializer.publish_cursor(
                    shard_path,
                    epoch=logical_epoch,
                )
                packed_materializer.mark_loaded(
                    shard_path,
                    epoch=logical_epoch,
                )
            elif remote_materializer is not None and stem in sel_shard_paths:
                selection_path = sel_shard_paths[stem]
                remote_materializer.publish_cursor(
                    selection_path,
                    epoch=logical_epoch,
                )
                remote_materializer.mark_loaded(
                    selection_path,
                    epoch=logical_epoch,
                )
            return cached
        # Exact-resume replay bypasses the per-node shared decode: pending
        # batch_rows can reference DIFFERENT shards on different local ranks
        # (stripes are unequal when rows_per_shard % world != 0, e.g. world
        # 31), so the shared store's identical-traversal invariant does not
        # hold here — a diverged sibling never publishes and the consumer
        # timeout kills the resume. Direct decode runs once per pending shard
        # at restore time and never in steady state.
        if packed_materializer is None:
            base = _load_packed_row_shard_with_node_cache(
                shard_path_by_name[shard_name],
                node_cache,
            )
        else:
            base = load_base_shard(
                shard_path_by_name[shard_name],
                logical_epoch=logical_epoch,
            )
        stem = shard_path_by_name[shard_name].stem.split(".")[0]
        selection = None
        if stem in sel_shard_paths:
            selection_path = sel_shard_paths[stem]
            if remote_materializer is not None:
                selection = load_remote_selection_shard(
                    selection_path,
                    logical_epoch=logical_epoch,
                    expected_segments=base.total_segments,
                )
            else:
                selection = _load_selection_shard_with_node_cache(
                    selection_path,
                    node_cache,
                )
            if packed_materializer is not None:
                packed_materializer.mark_loaded(
                    shard_path_by_name[shard_name],
                    epoch=logical_epoch,
                )
            elif remote_materializer is not None:
                remote_materializer.mark_loaded(
                    selection_path,
                    epoch=logical_epoch,
                )
        exact_shard_cache[shard_name] = (base, selection)
        return base, selection

    def parse_exact_batch_rows(
        raw_rows: Any,
        *,
        expected_batch_size: int,
    ) -> list[PendingRow]:
        if not isinstance(raw_rows, list) or len(raw_rows) != expected_batch_size:
            raise ValueError(
                "PrecomputedBatchLoader: batch_rows must be a list of exactly "
                f"the saved B={expected_batch_size} row references"
            )
        parsed: list[PendingRow] = []
        for ref_idx, raw_ref in enumerate(raw_rows):
            if not isinstance(raw_ref, (list, tuple)) or len(raw_ref) != 3:
                raise ValueError(
                    "PrecomputedBatchLoader: each batch_rows entry must be "
                    f"[epoch, shard_name, row_idx]; entry {ref_idx}={raw_ref!r}"
                )
            ref_epoch, shard_name, row_idx = raw_ref
            if type(ref_epoch) is not int or ref_epoch < 1:
                raise ValueError(
                    f"PrecomputedBatchLoader: batch_rows[{ref_idx}] epoch must "
                    f"be a positive int, got {ref_epoch!r}"
                )
            if not isinstance(shard_name, str):
                raise ValueError(
                    f"PrecomputedBatchLoader: batch_rows[{ref_idx}] shard name "
                    f"must be a string, got {shard_name!r}"
                )
            if type(row_idx) is not int:
                raise ValueError(
                    f"PrecomputedBatchLoader: batch_rows[{ref_idx}] row_idx must "
                    f"be an int, got {row_idx!r}"
                )
            base, selection = load_exact_shard(
                shard_name,
                logical_epoch=ref_epoch,
            )
            if not 0 <= row_idx < base.n_rows:
                raise ValueError(
                    f"PrecomputedBatchLoader: batch_rows[{ref_idx}] row_idx "
                    f"{row_idx} is out of range for {shard_name} with "
                    f"{base.n_rows} rows"
                )
            if row_idx % ddp_world_size != ddp_rank:
                raise ValueError(
                    f"PrecomputedBatchLoader: batch_rows[{ref_idx}] row {row_idx} "
                    f"does not belong to rank {ddp_rank} under world-size "
                    f"striping {ddp_world_size}"
                )
            parsed.append((ref_epoch, shard_name, base, selection, row_idx))

        def next_rank_reference(row: PendingRow) -> tuple[int, str, int]:
            current_epoch, current_name, current_base, _selection, current_idx = row
            next_idx = current_idx + ddp_world_size
            if next_idx < current_base.n_rows:
                return current_epoch, current_name, next_idx
            shard_idx = shard_index_by_name[current_name] + 1
            next_epoch = current_epoch
            # At least one referenced row exists for this rank, so cycling the
            # finite inventory must eventually find another one.
            for _ in range(len(shard_paths) + 1):
                if shard_idx == len(shard_paths):
                    shard_idx = 0
                    next_epoch += 1
                candidate_name = shard_paths[shard_idx].name
                candidate_base, _candidate_selection = load_exact_shard(
                    candidate_name,
                    logical_epoch=next_epoch,
                )
                if ddp_rank < candidate_base.n_rows:
                    return next_epoch, candidate_name, ddp_rank
                shard_idx += 1
            raise ValueError(
                "PrecomputedBatchLoader: no rank-local row exists in the shard "
                "inventory for the configured DDP topology"
            )

        for ref_idx in range(1, len(parsed)):
            expected = next_rank_reference(parsed[ref_idx - 1])
            actual = (
                parsed[ref_idx][0],
                parsed[ref_idx][1],
                parsed[ref_idx][4],
            )
            if actual != expected:
                raise ValueError(
                    "PrecomputedBatchLoader: batch_rows must be ordered, "
                    "consecutive rank-local references; entry "
                    f"{ref_idx} is {actual!r}, expected {expected!r}"
                )

        saved_epoch = resume_state_dict.get("epoch") if resume_state_dict else None
        saved_shard = resume_state_dict.get("shard") if resume_state_dict else None
        last_identity = (parsed[-1][0], parsed[-1][1])
        if last_identity != (saved_epoch, saved_shard):
            raise ValueError(
                "PrecomputedBatchLoader: batch_rows final reference must match "
                f"saved epoch/shard; rows end at {last_identity!r}, state has "
                f"{(saved_epoch, saved_shard)!r}"
            )
        return parsed

    exact_pending: list[PendingRow] | None = None
    exact_continuation_shard: str | None = None
    exact_continuation_after_row = -1
    exact_continuation_batch_offset = 0
    replay_label_rng_state: Any = None
    saved_batch_size = B
    migrating_device_batch = False
    if resume_state_dict is not None:
        if "feature_pipeline_rng_state" in resume_state_dict:
            _restore_feature_pipeline_rng_state(
                feature_pipeline,
                resume_state_dict["feature_pipeline_rng_state"],
                context="precomputed dataloader resume",
            )
        else:
            logger.warning(
                "PrecomputedBatchLoader: resume state has no "
                "feature_pipeline_rng_state; stochastic selection falls back "
                "to the configured seed."
            )

        if "label_dropout_rng_state" in resume_state_dict:
            replay_label_rng_state = resume_state_dict["label_dropout_rng_state"]
            _restore_default_rng_state(
                replay_label_rng_state,
                device,
                context="precomputed dataloader label-dropout resume",
                apply=False,
            )
        elif label_dropout > 0.0:
            logger.warning(
                "PrecomputedBatchLoader: resume state has no "
                "label_dropout_rng_state; label dropout falls back to the "
                "current default generator."
            )

    if resume_state_dict is not None and "batch_rows" in resume_state_dict:
        if type(resume_shard_batch_offset) is not int or resume_shard_batch_offset < 0:
            raise ValueError(
                "PrecomputedBatchLoader: exact resume requires a non-negative "
                "integer shard_batch_offset"
            )
        raw_batch_rows = resume_state_dict["batch_rows"]
        raw_saved_batch_size = resume_state_dict.get("device_batch_size")
        if raw_saved_batch_size is None:
            if not isinstance(raw_batch_rows, list) or not raw_batch_rows:
                raise ValueError(
                    "PrecomputedBatchLoader: exact resume batch_rows must be non-empty"
                )
            saved_batch_size = len(raw_batch_rows)
        elif type(raw_saved_batch_size) is not int or raw_saved_batch_size <= 0:
            raise ValueError(
                "PrecomputedBatchLoader: saved device_batch_size must be a "
                f"positive strict int, got {raw_saved_batch_size!r}"
            )
        else:
            saved_batch_size = raw_saved_batch_size
        exact_pending = parse_exact_batch_rows(
            raw_batch_rows,
            expected_batch_size=saved_batch_size,
        )
        migrating_device_batch = saved_batch_size != B
        last_epoch, last_shard, _base, _selection, last_row = exact_pending[-1]
        epoch = last_epoch
        exact_continuation_shard = last_shard
        exact_continuation_after_row = last_row
        exact_continuation_batch_offset = resume_shard_batch_offset + 1
        resume_shard = None
        canonical_draw_batch_size = resume_state_dict.get(
            "label_dropout_draw_batch_size",
            saved_batch_size,
        )
        canonical_draw_offset = resume_state_dict.get("label_dropout_draw_offset", 0)
        if label_dropout > 0.0 and (
            migrating_device_batch or "label_dropout_draw_batch_size" in resume_state_dict
        ):
            if replay_label_rng_state is None:
                raise ValueError(
                    "PrecomputedBatchLoader: device-batch migration with "
                    "label_dropout requires checkpointed label_dropout_rng_state"
                )
            canonical_dropout_stream = _CanonicalLabelDropoutStream(
                replay_label_rng_state,
                device=device,
                draw_batch_size=canonical_draw_batch_size,
                draw_offset=canonical_draw_offset,
            )
            replay_label_rng_state = None
    elif resume_state_dict is not None:
        saved_batch_value = resume_state_dict.get("device_batch_size")
        if saved_batch_value is not None and saved_batch_value != B:
            raise ValueError(
                "PrecomputedBatchLoader: device-batch migration requires exact "
                "batch_rows; legacy cursor states cannot be safely re-batched"
            )
        logger.warning(
            "PrecomputedBatchLoader: legacy resume state has no batch_rows; "
            "using historical shard/offset replay. Cross-shard pending rows "
            "cannot be reconstructed exactly."
        )

    if exact_pending is not None and not migrating_device_batch:
        yield materialize_pending_batch(
            exact_pending,
            resume_shard_batch_offset,
            replay_label_rng_state=replay_label_rng_state,
        )
        replay_label_rng_state = None
        batch_idx += 1
        # Exact continuation uses its separately preserved next offset above;
        # the legacy skip offset must not participate in any later shard.
        resume_shard_batch_offset = 0
    elif exact_pending is not None:
        pending = list(exact_pending)
        migration_batch_offset = resume_shard_batch_offset
        while len(pending) >= B:
            rows = pending[:B]
            del pending[:B]
            yield materialize_pending_batch(
                rows,
                migration_batch_offset,
                replay_label_rng_state=replay_label_rng_state,
            )
            replay_label_rng_state = None
            batch_idx += 1
            migration_batch_offset += 1
        exact_pending = None
        resume_shard_batch_offset = 0

    plan = _precomputed_shard_visits(
        shard_paths,
        start_epoch=epoch,
        resume_shard=resume_shard,
        resume_shard_batch_offset=resume_shard_batch_offset,
        exact_continuation_shard=exact_continuation_shard,
        exact_continuation_after_row=exact_continuation_after_row,
        exact_continuation_batch_offset=exact_continuation_batch_offset,
        log_epoch_starts=ddp_rank == 0,
    )

    def _visit_loads() -> Generator[tuple[_ShardVisit, PackedRowShard, Any]]:
        for visit in plan:
            with t.profiler.record_function("precomputed/load_shard"):
                base_shard = load_base_shard(visit.shard_path, logical_epoch=visit.epoch)

            sel_shard = None
            base_stem = visit.shard_path.stem.split(".")[0]
            if base_stem in sel_shard_paths:
                with t.profiler.record_function("precomputed/load_selection_shard"):
                    selection_path = sel_shard_paths[base_stem]
                    if remote_materializer is not None:
                        sel_shard = load_remote_selection_shard(
                            selection_path,
                            logical_epoch=visit.epoch,
                            expected_segments=base_shard.total_segments,
                        )
                    else:
                        sel_shard = _load_selection_shard_with_node_cache(
                            selection_path,
                            node_cache,
                        )
                    if packed_materializer is not None:
                        packed_materializer.mark_loaded(visit.shard_path, epoch=visit.epoch)
                    elif remote_materializer is not None:
                        remote_materializer.mark_loaded(
                            selection_path,
                            epoch=visit.epoch,
                        )
            yield visit, base_shard, sel_shard

    loaded_visits = _visit_loads()
    shard_stream: Generator[Any] = (
        _background_lookahead(loaded_visits, lookahead) if lookahead > 0 else loaded_visits
    )
    try:
        for visit, base_shard, sel_shard in shard_stream:
            row_count = base_shard.n_rows
            shard_batch_idx = visit.shard_batch_idx
            for row_idx in range(visit.first_row_idx, row_count):
                if row_idx % ddp_world_size != ddp_rank:
                    continue
                pending.append((visit.epoch, visit.shard_path.name, base_shard, sel_shard, row_idx))
                if len(pending) != B:
                    continue
                if shard_batch_idx < visit.skip_batches:
                    shard_batch_idx += 1
                    pending = []
                    continue
                yield materialize_pending_batch(
                    pending,
                    shard_batch_idx,
                    replay_label_rng_state=replay_label_rng_state,
                )
                replay_label_rng_state = None
                batch_idx += 1
                shard_batch_idx += 1
                pending = []
    finally:
        shard_stream.close()
