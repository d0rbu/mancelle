"""Precompute packed rows for the encoder-decoder training pipeline.

Runs the same document iteration + best-fit packing algorithm as the live
dataloader, but captures per-row data with ALL extraction features (pre-selection)
and writes to compressed CSR shard files.

At training time, the precomputed shards can be loaded and batched at any
batch size, with feature selection applied at runtime.

Usage::

    uv run python -m scripts.preprocess.precompute_packed_rows

    # Override params:
    uv run python -m scripts.preprocess.precompute_packed_rows context_len=1024 max_shards=50

    # Resume (skip shards that already exist):
    uv run python -m scripts.preprocess.precompute_packed_rows resume=true

    # Resume from an explicit output shard, ignoring local file count:
    uv run python -m scripts.preprocess.precompute_packed_rows resume_from_shard=128
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import FrameType, MappingProxyType
from typing import Protocol, cast

import hydra
import pyarrow.parquet as pq
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from errors import PackedRowInvariantError
from nanochat.atomic_io import atomic_json_dump, atomic_write_bytes
from nanochat.dataset import DATA_DIR, list_parquet_files
from nanochat.tokenizer import get_tokenizer
from scripts.preprocess._common import resolve_feature_dir
from scripts.preprocess._packed_rows_profile import PackedRowsProfile, ShardSaveTiming
from steering.conditional.encoder_decoder.dataloader import (
    DEFAULT_BUFFER_SIZE,
    SegFeatures,
    _build_row_segments,
    truncate_to_boundary_coverage,
)
from steering.conditional.encoder_decoder.fp_target_source import (
    PackedShardAttestation,
    attested_inventory,
    certify_fp_target_source,
    certify_fp_target_source_from_attestations,
    validate_fp_target_source_certificate,
    validate_fp_target_source_certificate_remote,
)
from steering.conditional.encoder_decoder.sorted_doc_buffer import SortedDocBuffer
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.feature_io import glob_features, shard_stem_from_path
from steering.feature_paths import SegmentSizeMismatch, parse_feature_dir, verify_segment_size
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.immutable_io import is_sha256_hex
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    OverlayKind,
    _compute_base_hash,
    create_mirror_symlinks,
    discover_overlay,
    glob_overlay_shards,
    overlay_shard_path,
    read_manifest,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    load_packed_row_shard,
    save_packed_row_shard,
)
from steering.packed_rows_offload import (
    OFFLOAD_RECEIPT_DIRNAME,
    PackedRowOffloader,
    read_offload_receipts,
)
from steering.packed_rows_remote import (
    RemotePackedRowsError,
    RemotePackedRowsFeatureLookup,
    SelectedParquet,
    required_packed_rows_staging_capacity_bytes,
)
from steering.refined_types import PositiveInt
from steering.remote_feature_inventory import RemoteShardFamily, resolve_remote_feature_prefix
from steering.remote_training import RemoteObjectInfo, download_verified_object
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth
from steering.segmentation import compute_tokenizer_checksum
from steering.stats_io import find_statistics_path, load_percentile_checksum
from steering.token_boundary_remote import (
    RemoteTokenBoundaryTransport as _RemotePackedRowsTransport,
)
from steering.token_boundary_remote import (
    TokenBoundaryRemoteConfig as _PackedRowsRemoteTransportConfig,
)
from steering.token_boundary_remote import (
    _ensure_staging_directory,
)
from steering.token_boundary_remote import (
    remote_output_lock as _remote_packed_rows_output_lock,
)

_PACKED_ROWS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "extraction" / "precompute_packed_rows.yaml"
)
_REMOTE_PACKING_CHECKPOINT_SCHEMA_VERSION = 1
_REMOTE_PACKING_CHECKPOINT_MAX_BYTES = 4 * 1024**3


def load_packed_rows_config(
    *overrides: Mapping[str, object] | DictConfig,
) -> DictConfig:
    """Load the canonical packed-row schema and merge recognized overrides."""
    base = OmegaConf.load(_PACKED_ROWS_CONFIG_PATH)
    if not isinstance(base, DictConfig):
        raise TypeError(f"packed-row config must be a mapping, got {type(base).__name__}")
    OmegaConf.set_struct(base, True)
    merged = OmegaConf.merge(base, *overrides)
    if not isinstance(merged, DictConfig):
        raise TypeError(f"merged packed-row config must be a mapping, got {type(merged).__name__}")
    return merged


def _compute_shard_max_dims(shard: PackedRowShard) -> tuple[int, int]:
    """Compute max_n_seg and the raw max segment length from a packed-row shard.

    Note: max_F is NOT computed here — base shards store ALL extraction features
    (~8k per segment) which is not the training-time feature dimension. The
    training-time max_F comes from selection shards (post feature selection).

    ``max_seg_size`` is the exact maximum
    ``seg_token_end - seg_token_start``. It is manifest/resume identity, not an
    FP8 allocation extent; cross-attention validates this raw bound and rounds
    its private allocation up to ``FP8_ALIGNMENT``.

    Returns:
        (max_n_seg, max_seg_size) for this shard.
    """
    row_offsets = shard.row_seg_offsets.tolist()

    max_n_seg = 0
    max_seg_size = 0

    for r in range(shard.n_rows):
        seg_start = row_offsets[r]
        seg_end = row_offsets[r + 1]
        n_segs = seg_end - seg_start
        max_n_seg = max(max_n_seg, n_segs)

        for s in range(seg_start, seg_end):
            seg_len = int(shard.seg_token_end[s]) - int(shard.seg_token_start[s])
            max_seg_size = max(max_seg_size, seg_len)

    return max_n_seg, max_seg_size


def _reduce_max_dims_over_shards(shard_paths: list[Path]) -> tuple[int, int]:
    """Reduce (max_n_seg, max_seg_size) over packed-row shard files by loading each and calling
    ``_compute_shard_max_dims``. Shared by the resume legacy fallback (issue #539) and by
    ``backfill_packed_row_shapes``. O(zstd-decompress) per shard — the slow path.
    """
    max_n_seg = 0
    max_seg_size = 0
    for sp in shard_paths:
        mns, mss = _compute_shard_max_dims(load_packed_row_shard(sp))
        max_n_seg = max(max_n_seg, mns)
        max_seg_size = max(max_seg_size, mss)
    return max_n_seg, max_seg_size


def _dims_companion_path(output_dir: Path, shard_idx: int) -> Path:
    """Path of a shard's max-dims companion (issue #539): a sibling of the ``.zst`` shard under a
    ``_dims/`` subdir that is invisible to ``glob_overlay_shards`` (non-recursive + ``.zst``-only).
    Single source of the ``_dims`` dir name + companion-name format (writer + reader share it).
    """
    return output_dir / "_dims" / f"shard_{shard_idx:05d}.json"


def _packed_shard_complete(
    output_dir: Path,
    shard_idx: int,
    *,
    require_dims: bool = False,
    offloaded: Collection[int] = (),
) -> bool:
    """A packed shard is COMPLETE iff BOTH its packed ``.zst`` and its
    ``provenance/`` sidecar exist — they are flushed together in one worker, so
    an incomplete trailing shard (a crash between the two writes, or a dir
    packed before native provenance existed) is NOT complete and resume rewrites
    it. In LOCAL mode the ``_dims/`` companion stays a soft heal-on-resume
    optimization (``_seed_max_dims_from_existing`` decompress-falls-back when it
    is missing) so legacy dirs resume without replaying. REMOTE checkpoint-seek
    resume skips the rewrite that used to heal a torn companion, so remote
    callers pass ``require_dims=True`` (every remote output postdates the
    ``_dims`` writer; there are no legacy remote dirs) — a kill between the
    provenance and ``_dims`` writes then leaves the shard NOT-kept and seek
    rewrites all three artifacts byte-identically. Single source of the
    resume-completeness rule for the producer.

    An OFFLOADED shard is complete without either body: its offload receipt is
    the durable proof that both were published to the bucket and verified there
    before eviction. The ``_dims`` requirement still applies to it, so a lost
    companion simply makes the shard not-kept and the packer re-packs and
    re-publishes it (idempotently, byte-identically) rather than silently
    understating the manifest's padding dims.
    """
    name = f"shard_{shard_idx:05d}"
    if shard_idx not in offloaded and not (
        overlay_shard_path(output_dir, name).exists()
        and overlay_shard_path(provenance_dir(output_dir), name).exists()
    ):
        return False
    return not require_dims or _dims_companion_path(output_dir, shard_idx).is_file()


def _seed_max_dims_from_existing(output_dir: Path, n_kept: int) -> tuple[int, int]:
    """Seed (max_n_seg, max_seg_size) from the first ``n_kept`` shards kept on resume (issue #539).

    Prefers the O(1) per-shard ``_dims/`` companions. A present-but-companionless shard (a dir
    written before #539, or a crash between the shard and its companion) OR a corrupt companion
    falls back to reducing that shard directly (a one-time decompress). Shards **absent on disk**
    are skipped and warned: an explicit ``resume_from_shard=N`` "ignoring local file count" leaves
    the kept prefix off this box, so its max dims are not locally recoverable and the manifest will
    reflect only the new + locally-present shards (it MAY understate) — use ``resume=true`` or
    re-run ``backfill_packed_row_shapes`` once all shards are local for exact dims.
    """
    if n_kept <= 0:
        return 0, 0
    pairs: list[tuple[int, int]] = []
    fallback_paths: list[Path] = []
    n_absent = 0
    for i in range(n_kept):
        # Companion first, body second: an offloaded shard's body is published
        # and gone, so the companion (never evicted, and required for such a
        # shard to count as kept at all) is its only dims authority.
        comp = _dims_companion_path(output_dir, i)
        if comp.is_file():
            try:
                d = json.loads(comp.read_text())
                pairs.append((int(d["max_n_seg"]), int(d["max_seg_size"])))
                continue
            except (ValueError, TypeError, KeyError) as exc:
                # Corrupt companion (external tamper / disk corruption) — degrade to the same
                # reduce-over-shards fallback as a missing one rather than aborting the resume.
                logger.warning(
                    f"Resume: ignoring corrupt _dims companion {comp} ({exc!r}); reducing that "
                    f"shard's dims directly."
                )
        shard_path = overlay_shard_path(output_dir, f"shard_{i:05d}")
        if not shard_path.is_file():
            n_absent += 1
            continue
        fallback_paths.append(shard_path)
    if fallback_paths:
        logger.warning(
            f"Resume: {len(fallback_paths)} existing shard(s) under {output_dir} lack a usable "
            f"_dims companion — reducing their max dims directly (slow). New runs write companions."
        )
        pairs.append(_reduce_max_dims_over_shards(fallback_paths))
    if n_absent:
        logger.warning(
            f"Resume: {n_absent}/{n_kept} prefix shard(s) are NOT present locally (an explicit "
            f"resume_from_shard ignoring local file count) — their max_n_seg/max_seg_size cannot "
            f"be recovered, so the manifest's static-padding dims will reflect only the new + "
            f"locally-present shards and MAY UNDERSTATE. Use resume=true (needs the local shards), "
            f"or re-run scripts.preprocess.backfill_packed_row_shapes once all shards are local."
        )
    if not pairs:
        return 0, 0
    return max(n for n, _ in pairs), max(s for _, s in pairs)


@dataclass(frozen=True)
class _PackedSegFeatures:
    """Packed-row-only segment carrier that keeps stored percentile codes encoded.

    Its attribute surface deliberately matches the subset consumed by
    ``_build_row_segments``. It is not the live dataloader's ``SegFeatures`` and
    never changes that class's float32 percentile contract.
    """

    feature_ids: t.Tensor
    percentiles: t.Tensor  # (N,) uint8, exact precomputed overlay codes
    char_start: int
    char_end: int
    token_start: int | None
    token_end: int | None
    # _build_row_segments shares an attribute-level protocol with live
    # SegFeatures. Packed production never populates these optional fields;
    # checkpoints therefore do not serialize them.
    all_feature_ids: t.Tensor | None = None
    all_feature_pcts: t.Tensor | None = None


def _seg_features_to_packed(seg: SegmentFeatures) -> _PackedSegFeatures:
    """Convert FeatureIndex data without decoding its stored percentile bytes."""
    n = seg.feature_ids.numel()
    codes = (
        seg.feature_percentiles
        if seg.feature_percentiles is not None
        else t.zeros(n, dtype=t.uint8)
    )
    if seg.feature_ids.ndim != 1 or codes.dtype != t.uint8 or codes.ndim != 1 or len(codes) != n:
        raise PackedRowInvariantError(
            "packed feature ids and percentile codes must be aligned rank-1 (uint8 codes): "
            f"features={n}, ids_shape={tuple(seg.feature_ids.shape)}, "
            f"codes_shape={tuple(codes.shape)}, dtype={codes.dtype}"
        )
    return _PackedSegFeatures(
        feature_ids=seg.feature_ids.to(t.int64),
        percentiles=codes,
        char_start=seg.char_start,
        char_end=seg.char_end,
        token_start=seg.token_start,
        token_end=seg.token_end,
    )


def assert_boundaries_covered(
    text_tokens: list[int],
    seg_features: Sequence[SegFeatures | _PackedSegFeatures] | None,
) -> None:
    """Assert precomputed segment boundaries are consistent with the token list.

    Segments may have token_end > n_tokens when the precomputed boundaries
    cover more nanochat tokens than context_len (nanochat can be more efficient
    than Gemma on digit-heavy text). This is fine — _build_row_segments
    clamps tok_hi = min(sf.token_end, n_doc_tokens) for all segments.

    We check that the first segment starts at 0 (BOS-aligned) and that at
    least one segment starts within the token range.
    """
    if not seg_features:
        return

    n_tokens = len(text_tokens)
    if seg_features[0].token_start is None:
        # truncate_to_boundary_coverage already raises on missing boundaries
        # when require_token_boundaries=True. If we get here, the caller
        # explicitly opted out — nothing further to check.
        return

    assert seg_features[0].token_start == 0, (
        f"First segment token_start={seg_features[0].token_start}, expected 0"
    )
    assert any(cast(int, sf.token_start) < n_tokens for sf in seg_features), (
        f"No segment starts within token count {n_tokens}. "
        f"First token_start={seg_features[0].token_start}, "
        f"last token_start={seg_features[-1].token_start}"
    )


class _FeatureLookup(Protocol):
    """Minimal feature-index contract required by raw document iteration."""

    def lookup(self, shard_stem: str, doc_idx: int) -> list[SegmentFeatures]: ...


class _PackedRowsFeatureLookup(_FeatureLookup, Protocol):
    """Complete feature-source contract required by the row packer."""

    @property  # pragma: no cover - protocol declaration
    def segment_size(self) -> int | None: ...

    @property  # pragma: no cover - protocol declaration
    def shard_stems(self) -> list[str]: ...


@dataclass
class _DocumentCursor:
    """Next document to yield from a parquet split.

    ``rows_consumed_in_group`` is the number of documents already yielded
    from ``row_group_idx``. The cursor is canonicalized across row-group and
    file boundaries before each yield reaches the caller.
    """

    parquet_file_idx: int = 0
    row_group_idx: int = 0
    rows_consumed_in_group: int = 0


def _raw_document_iterator(
    parquet_paths: list[str],
    tokenizer,
    bos_token: int,
    feature_index: _FeatureLookup,
    profile: PackedRowsProfile | None = None,
    cursor: _DocumentCursor | None = None,
):
    """Iterate documents in the same order as the training pipeline.

    Yields (text_tokens, packed_seg_features) for each document.
    packed_seg_features keeps ALL extraction feature IDs and percentile bytes
    (always non-empty: a present doc has >= 1 segment; a missing doc raises).

    When ``profile`` is given, per-stage wall-clock (parquet read, batch tokenize,
    feature lookup, segment conversion) and the per-base-shard load cost are
    accumulated. The timers do not change what is yielded (byte-identical).
    """
    active_cursor = cursor if cursor is not None else _DocumentCursor()
    # Capture the start triple before iterating: the yield block below mutates
    # the cursor on every document, so the skip predicates must not read it.
    start_file = active_cursor.parquet_file_idx
    start_rg = active_cursor.row_group_idx
    start_row = active_cursor.rows_consumed_in_group
    if not (0 <= start_file <= len(parquet_paths)):
        raise PackedRowInvariantError(f"parquet cursor file index is out of range: {start_file}")
    for file_idx, filepath in enumerate(parquet_paths[start_file:], start=start_file):
        pf = pq.ParquetFile(filepath)
        shard_stem = Path(filepath).stem
        first_lookup_of_shard = True

        rg_start = start_rg if file_idx == start_file else 0
        global_doc_offset = sum(pf.metadata.row_group(index).num_rows for index in range(rg_start))
        for rg_idx in range(rg_start, pf.num_row_groups):
            n_group_rows = pf.metadata.row_group(rg_idx).num_rows
            row_start = start_row if file_idx == start_file and rg_idx == start_rg else 0
            if not (0 <= row_start <= n_group_rows):
                raise PackedRowInvariantError(
                    "parquet cursor row offset is out of range: "
                    f"file={file_idx}, row_group={rg_idx}, offset={row_start}, "
                    f"rows={n_group_rows}"
                )
            t_pq = time.monotonic()
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()[row_start:]
            if profile is not None:
                profile.stage_totals["parquet"] += time.monotonic() - t_pq

            # Batch-encode all texts in the row group at once.
            # This amortizes Python→Rust FFI overhead and lets tiktoken
            # use internal threading (encode_ordinary_batch).
            t_enc = time.monotonic()
            all_token_lists = tokenizer.encode(texts, prepend=bos_token)
            if profile is not None:
                profile.stage_totals["tokenize"] += time.monotonic() - t_enc

            for local_idx in range(len(texts)):
                row_idx = row_start + local_idx
                doc_idx = global_doc_offset + row_idx
                text_tokens = all_token_lists[local_idx]
                # lookup() raises KeyError (naming shard+doc) for an absent row —
                # corruption (issue #521). A present doc always has >= 1 segment.
                t_lk = time.monotonic()
                segments = feature_index.lookup(shard_stem, doc_idx)
                dt_lk = time.monotonic() - t_lk
                if profile is not None:
                    profile.stage_totals["lookup"] += dt_lk
                    # The first lookup of each shard is the cache miss that loads
                    # (and indexes) the ~GB base shard — attribute it separately.
                    if first_lookup_of_shard:
                        profile.base_shard_loads["count"] += 1
                        profile.base_shard_loads["total_seconds"] += dt_lk
                        first_lookup_of_shard = False
                t_sr = time.monotonic()
                packed_seg_features = [_seg_features_to_packed(seg) for seg in segments]
                if profile is not None:
                    profile.stage_totals["seg_to_raw"] += time.monotonic() - t_sr

                if row_idx + 1 < n_group_rows:
                    active_cursor.parquet_file_idx = file_idx
                    active_cursor.row_group_idx = rg_idx
                    active_cursor.rows_consumed_in_group = row_idx + 1
                elif rg_idx + 1 < pf.num_row_groups:
                    active_cursor.parquet_file_idx = file_idx
                    active_cursor.row_group_idx = rg_idx + 1
                    active_cursor.rows_consumed_in_group = 0
                else:
                    active_cursor.parquet_file_idx = file_idx + 1
                    active_cursor.row_group_idx = 0
                    active_cursor.rows_consumed_in_group = 0
                yield text_tokens, packed_seg_features, shard_stem, doc_idx

            global_doc_offset += n_group_rows


type _PackedDocEntry = tuple[list[int], list[_PackedSegFeatures], tuple[int, int]]


@dataclass(frozen=True)
class _RemotePackingCheckpointContext:
    checkpoint_dir: Path
    identity_digest: str
    split: str
    parquet_inventory: tuple[tuple[Path, int], ...]
    end_position: int
    rows_per_shard: int


@dataclass(frozen=True)
class _LoadedPackingCheckpoint:
    header: dict[str, object]
    buffer_entries: tuple[_PackedDocEntry, ...]
    # Derived by the loader from the cursor + the context's parquet inventory
    # (both frozen by identity_digest), never stored in the file.
    consumed_bodies: int
    data_exhausted: bool


def _checkpoint_path(checkpoint_dir: Path, shard_idx: int) -> Path:
    return checkpoint_dir / f"shard_{shard_idx:05d}.ckpt"


def _flatten_tensors(
    tensors: Sequence[t.Tensor],
    *,
    dtype: t.dtype,
) -> tuple[t.Tensor, t.Tensor]:
    offsets = [0]
    flattened: list[t.Tensor] = []
    for tensor in tensors:
        flat = tensor.detach().cpu().reshape(-1)
        flattened.append(flat)
        offsets.append(offsets[-1] + flat.numel())
    values = t.cat(flattened) if flattened else t.empty(0, dtype=dtype)
    return values.to(dtype), t.tensor(offsets, dtype=t.int64)


def _serialize_checkpoint_buffer(doc_buffer: SortedDocBuffer) -> dict[str, t.Tensor]:
    # Buffer entries were validated where they were built (_seg_features_to_packed
    # enforces aligned rank-1 uint8 percentiles); only the None-sentinel token
    # boundaries need write-side care because -1 encodes absence.
    entries = cast(list[_PackedDocEntry], doc_buffer.snapshot())
    text_tensors = [t.tensor(entry[0], dtype=t.int32) for entry in entries]
    text_tokens, text_offsets = _flatten_tensors(text_tensors, dtype=t.int32)
    provenance = (
        t.tensor([entry[2] for entry in entries], dtype=t.int64)
        if entries
        else t.empty((0, 2), dtype=t.int64)
    )
    segments = [seg for _tokens, doc_segments, _prov in entries for seg in doc_segments]
    doc_seg_offsets = [0]
    for _tokens, doc_segments, _prov in entries:
        doc_seg_offsets.append(doc_seg_offsets[-1] + len(doc_segments))

    for seg in segments:
        if (seg.token_start is None) != (seg.token_end is None):
            raise PackedRowInvariantError(
                "checkpoint token boundaries must be both present or absent"
            )
        if seg.token_start is not None and seg.token_start < 0:
            raise PackedRowInvariantError(
                "checkpoint token boundaries must be non-negative (sentinel -1 is reserved)"
            )

    feature_tensors = [seg.feature_ids for seg in segments]
    if any(int(ids.max()) > t.iinfo(t.uint16).max for ids in feature_tensors if ids.numel()):
        raise PackedRowInvariantError("checkpoint feature id values do not fit uint16")
    feature_ids, feature_offsets = _flatten_tensors(feature_tensors, dtype=t.uint16)
    percentiles, _percentile_offsets = _flatten_tensors(
        [seg.percentiles for seg in segments], dtype=t.uint8
    )

    return {
        "text_tokens": text_tokens,
        "text_offsets": text_offsets,
        "provenance": provenance,
        "doc_seg_offsets": t.tensor(doc_seg_offsets, dtype=t.int64),
        "feature_ids": feature_ids,
        "feature_offsets": feature_offsets,
        "percentiles": percentiles,
        "char_start": t.tensor([seg.char_start for seg in segments], dtype=t.int64),
        "char_end": t.tensor([seg.char_end for seg in segments], dtype=t.int64),
        "token_start": t.tensor(
            [-1 if seg.token_start is None else seg.token_start for seg in segments],
            dtype=t.int64,
        ),
        "token_end": t.tensor(
            [-1 if seg.token_end is None else seg.token_end for seg in segments],
            dtype=t.int64,
        ),
    }


def _deserialize_checkpoint_buffer(payload: dict[str, t.Tensor]) -> tuple[_PackedDocEntry, ...]:
    """Reconstruct buffer entries from digest-verified serializer output.

    The caller verified the file's content digest, so these bytes are exactly
    what `_serialize_checkpoint_buffer` wrote — structural properties hold by
    construction, and any residual inconsistency raises into the caller's
    reject-and-replay funnel.
    """
    tensors = payload
    text_tokens = tensors["text_tokens"]
    text_offsets = tensors["text_offsets"].tolist()
    provenance = tensors["provenance"]
    doc_seg_offsets = tensors["doc_seg_offsets"].tolist()
    feature_ids = tensors["feature_ids"]
    feature_offsets = tensors["feature_offsets"].tolist()
    percentiles = tensors["percentiles"]
    n_docs = provenance.shape[0]
    n_segments = len(feature_offsets) - 1

    segments: list[_PackedSegFeatures] = []
    for seg_idx in range(n_segments):
        lo, hi = feature_offsets[seg_idx : seg_idx + 2]
        token_start = int(tensors["token_start"][seg_idx])
        token_end = int(tensors["token_end"][seg_idx])
        token_absent = token_start == -1 and token_end == -1
        segments.append(
            _PackedSegFeatures(
                feature_ids=feature_ids[lo:hi].to(t.int64),
                percentiles=percentiles[lo:hi].clone(),
                char_start=int(tensors["char_start"][seg_idx]),
                char_end=int(tensors["char_end"][seg_idx]),
                token_start=None if token_absent else token_start,
                token_end=None if token_absent else token_end,
            )
        )

    entries: list[_PackedDocEntry] = []
    for doc_idx in range(n_docs):
        text_lo, text_hi = text_offsets[doc_idx : doc_idx + 2]
        seg_lo, seg_hi = doc_seg_offsets[doc_idx : doc_idx + 2]
        entries.append(
            (
                text_tokens[text_lo:text_hi].tolist(),
                segments[seg_lo:seg_hi],
                (int(provenance[doc_idx, 0]), int(provenance[doc_idx, 1])),
            )
        )
    return tuple(entries)


def _atomic_save_packing_checkpoint(payload: dict[str, object], path: Path) -> None:
    """Atomically write `sha256hex + "\\n" + torch bytes`.

    torch.load does not detect payload corruption (a flipped byte loads
    silently), so the digest is the checkpoint's sole integrity guarantee: the
    loader rejects any file whose bytes differ from what this writer produced.
    """
    buffer = io.BytesIO()
    t.save(payload, buffer)
    raw = buffer.getvalue()
    if len(raw) > _REMOTE_PACKING_CHECKPOINT_MAX_BYTES:
        raise PackedRowInvariantError(f"remote packing checkpoint exceeds 4 GiB: {len(raw)} bytes")
    digest = hashlib.sha256(raw).hexdigest().encode("ascii")
    atomic_write_bytes(digest + b"\n" + raw, path)


def _load_eligible_packing_checkpoint(
    context: _RemotePackingCheckpointContext,
    *,
    n_kept: int,
) -> _LoadedPackingCheckpoint | None:
    path = _checkpoint_path(context.checkpoint_dir, n_kept)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            descriptor_stat = os.fstat(descriptor)
            if not stat.S_ISREG(descriptor_stat.st_mode) or descriptor_stat.st_nlink != 1:
                raise ValueError("checkpoint must be a single-link regular file")
            if not (0 < descriptor_stat.st_size <= _REMOTE_PACKING_CHECKPOINT_MAX_BYTES + 65):
                raise ValueError(
                    "checkpoint serialized size is outside (0, 4 GiB]: "
                    f"{descriptor_stat.st_size} bytes"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as checkpoint_file:
                stored = checkpoint_file.read()
        finally:
            os.close(descriptor)
        # The digest is the sole integrity guarantee (torch.load accepts flipped
        # bytes silently): a match proves the payload is exactly what
        # _atomic_save_packing_checkpoint wrote, so structural properties hold by
        # construction and only bindings to EXTERNAL state are checked below.
        digest, newline, payload_bytes = stored[:64], stored[64:65], stored[65:]
        if newline != b"\n" or hashlib.sha256(payload_bytes).hexdigest().encode("ascii") != digest:
            raise ValueError("checkpoint content digest mismatch")
        raw = t.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
        header = raw["header"]
        expected = {
            "schema_version": _REMOTE_PACKING_CHECKPOINT_SCHEMA_VERSION,
            "identity_digest": context.identity_digest,
            "split": context.split,
            "shard_idx": n_kept,
        }
        for key, value in expected.items():
            if header[key] != value:
                raise ValueError(f"{key} mismatch: expected={value!r}, got={header[key]!r}")
        file_idx = cast(int, header["parquet_file_idx"])
        row_group_idx = cast(int, header["row_group_idx"])
        rows_consumed_in_group = cast(int, header["rows_consumed_in_group"])
        if not (0 <= file_idx <= len(context.parquet_inventory)):
            raise ValueError("parquet_file_idx exceeds split inventory")
        absolute_document_offset = 0
        for prior_path, _position in context.parquet_inventory[:file_idx]:
            absolute_document_offset += pq.ParquetFile(prior_path).metadata.num_rows
        if file_idx == len(context.parquet_inventory):
            if row_group_idx != 0 or rows_consumed_in_group != 0:
                raise ValueError("end-of-split cursor must have zero row-group coordinates")
        else:
            parquet = pq.ParquetFile(context.parquet_inventory[file_idx][0])
            if not (0 <= row_group_idx < parquet.num_row_groups):
                raise ValueError("row_group_idx exceeds parquet metadata")
            absolute_document_offset += sum(
                parquet.metadata.row_group(index).num_rows for index in range(row_group_idx)
            )
            n_group_rows = parquet.metadata.row_group(row_group_idx).num_rows
            if not (0 <= rows_consumed_in_group < n_group_rows):
                raise ValueError("rows_consumed_in_group is not a canonical next-row offset")
            absolute_document_offset += rows_consumed_in_group
        if absolute_document_offset != header["documents_loaded"]:
            raise ValueError(
                "checkpoint document cursor does not match its absolute document position: "
                f"cursor={absolute_document_offset}, "
                f"documents_loaded={header['documents_loaded']!r}"
            )
        total_rows = cast(int, header["total_rows"])
        minimum_rows = (n_kept - 1) * context.rows_per_shard
        maximum_rows = n_kept * context.rows_per_shard
        if not (minimum_rows < total_rows <= maximum_rows):
            raise ValueError(
                f"total_rows is incompatible with shard_idx: {total_rows} not in "
                f"({minimum_rows}, {maximum_rows}]"
            )
        entries = _deserialize_checkpoint_buffer(raw["buffer"])
        if cast(int, header["documents_loaded"]) - cast(int, header["documents_consumed"]) != len(
            entries
        ):
            raise ValueError("document counters do not match serialized buffer")
        data_exhausted = file_idx == len(context.parquet_inventory)
        if total_rows != maximum_rows and not (data_exhausted and not entries):
            raise ValueError(
                "partial total_rows is legal only for a terminal checkpoint: "
                f"got={total_rows}, full={maximum_rows}"
            )
        consumed_bodies = (
            context.parquet_inventory[file_idx][1]
            if file_idx < len(context.parquet_inventory)
            else context.end_position
        )
    except Exception as exc:
        logger.warning(
            f"Remote checkpoint seek rejected for {context.split}: {path}: "
            f"{type(exc).__name__}: {exc}; falling back to replay"
        )
        return None
    return _LoadedPackingCheckpoint(
        header=cast(dict[str, object], header),
        buffer_entries=entries,
        consumed_bodies=consumed_bodies,
        data_exhausted=data_exhausted,
    )


def _load_nearest_eligible_packing_checkpoint(
    context: _RemotePackingCheckpointContext,
    *,
    n_kept: int,
    compress_workers: int,
) -> _LoadedPackingCheckpoint | None:
    """Return the newest eligible checkpoint inside the retained write window."""
    oldest = max(0, n_kept - (compress_workers + 2))
    for checkpoint_idx in range(n_kept, oldest - 1, -1):
        loaded = _load_eligible_packing_checkpoint(context, n_kept=checkpoint_idx)
        if loaded is not None:
            return loaded
    return None


def _write_packing_checkpoint(
    context: _RemotePackingCheckpointContext,
    *,
    shard_idx: int,
    total_rows: int,
    documents_loaded: int,
    documents_consumed: int,
    cursor: _DocumentCursor,
    doc_buffer: SortedDocBuffer,
    compress_workers: int,
) -> None:
    # data_exhausted and consumed_bodies (the earliest remote body a
    # fast-forwarded iterator can still request — resume redownloads at most
    # that one boundary body) are pure functions of the cursor and the
    # identity-digest-frozen parquet inventory; the loader derives them.
    header: dict[str, object] = {
        "schema_version": _REMOTE_PACKING_CHECKPOINT_SCHEMA_VERSION,
        "identity_digest": context.identity_digest,
        "split": context.split,
        "shard_idx": shard_idx,
        "total_rows": total_rows,
        "documents_loaded": documents_loaded,
        "documents_consumed": documents_consumed,
        "parquet_file_idx": cursor.parquet_file_idx,
        "row_group_idx": cursor.row_group_idx,
        "rows_consumed_in_group": cursor.rows_consumed_in_group,
    }
    path = _checkpoint_path(context.checkpoint_dir, shard_idx)
    _atomic_save_packing_checkpoint(
        {"header": header, "buffer": _serialize_checkpoint_buffer(doc_buffer)},
        path,
    )
    expired_idx = shard_idx - (compress_workers + 3)
    if expired_idx >= 0:
        _checkpoint_path(context.checkpoint_dir, expired_idx).unlink(missing_ok=True)


def _pack_row_with_provenance(
    doc_buffer: SortedDocBuffer,
    row_capacity: int,
) -> tuple[t.Tensor, list, list[tuple[int, int]]]:
    """Pack one row of tokens using best-fit-decreasing from a sorted buffer.

    Buffer entries are ``(doc_tokens, seg_features, provenance)``. Provenance
    is the per-document ``(base_shard_idx, doc_idx)`` identity carried alongside
    the placement.

    Positions beyond the final placed doc are ZERO-filled, never left as
    uninitialized memory: the trailing run is excluded from every segment
    (``_row_to_csr`` trims it) but its bytes ARE persisted, and packing must
    be byte-deterministic — resume replay and the determinism/golden tests all
    rely on it. (``t.empty`` here made the final partial row's tail dirty
    allocator memory, which broke bit-reproducibility on Linux.)

    Returns ``(row_tokens [row_capacity], row_doc_info, row_doc_prov)``;
    ``row_doc_info`` entries are ``(pos, seg_features, doc_tokens_placed)``,
    aligned 1:1 with ``row_doc_prov``.
    """
    row_tokens = t.zeros(row_capacity, dtype=t.long)
    row_doc_info: list = []
    row_doc_prov: list[tuple[int, int]] = []
    pos = 0

    while pos < row_capacity and doc_buffer:
        remaining = row_capacity - pos

        popped = doc_buffer.pop_best_fit(remaining)
        if popped is not None:
            entry = cast("_PackedDocEntry", popped)
            placed = entry[0]
        else:
            # No doc fits: crop shortest to fill remaining
            entry = cast("_PackedDocEntry", doc_buffer.pop_shortest())
            placed = entry[0][:remaining]

        row_tokens[pos : pos + len(placed)] = t.tensor(placed, dtype=t.long)
        row_doc_info.append((pos, entry[1], placed))
        row_doc_prov.append(entry[2])
        pos += len(placed)

    return row_tokens, row_doc_info, row_doc_prov


class _RowCSR:
    """CSR data for a single packed row. All fields are tensors."""

    __slots__ = (
        "tokens",
        "seg_token_start",
        "seg_token_end",
        "seg_doc_id",
        "feature_ids",
        "feature_pcts",
        "feat_counts",
    )

    def __init__(
        self,
        tokens: t.Tensor,
        seg_token_start: t.Tensor,
        seg_token_end: t.Tensor,
        seg_doc_id: t.Tensor,
        feature_ids: t.Tensor,
        feature_pcts: t.Tensor,
        feat_counts: t.Tensor,
    ):
        self.tokens = tokens
        self.seg_token_start = seg_token_start
        self.seg_token_end = seg_token_end
        self.seg_doc_id = seg_doc_id
        self.feature_ids = feature_ids
        self.feature_pcts = feature_pcts
        self.feat_counts = feat_counts

    @property
    def n_segs(self) -> int:
        return len(self.seg_token_start)


def _row_to_csr(
    row_tokens: t.Tensor,
    row_doc_info: list,
    T: int,
    tokenizer,
    *,
    profile_detail: dict[str, float] | None = None,
) -> _RowCSR:
    """Convert a packed row to CSR segment/feature data.

    Calls ``_build_row_segments`` to get raw per-segment feature lists and
    seg_map, then concatenates features directly into CSR format — skipping
    the pad-to-matrix-then-extract-with-mask pattern entirely.
    """
    build_segments_started = time.monotonic() if profile_detail is not None else 0.0
    per_seg_fids, _, per_seg_pcts, seg_map_row, seg_doc_id_row, _, _ = _build_row_segments(
        row_doc_info,
        T,
        tokenizer,
        collect_buckets=False,
        empty_percentile_dtype=t.uint8,
    )
    if profile_detail is not None:
        profile_detail["build_segments_seconds"] += time.monotonic() - build_segments_started

    n_segs = len(per_seg_fids)

    # Validate doc_id fits in uint8
    max_doc_id = int(seg_doc_id_row.max().item()) if len(seg_doc_id_row) > 0 else 0
    assert max_doc_id <= 255, (
        f"Too many documents in one row ({max_doc_id + 1} > 256) for uint8 seg_doc_id. "
        f"Consider using uint16 for seg_doc_id."
    )

    boundaries_started = time.monotonic() if profile_detail is not None else 0.0
    # Extract per-segment token boundaries from seg_map.
    # seg_map is monotonically non-decreasing (cummax enforced), so we find
    # contiguous run boundaries by detecting where the value changes.
    if n_segs > 0:
        changes = (seg_map_row[1:] != seg_map_row[:-1]).nonzero(as_tuple=True)[0] + 1
        boundary_positions = t.cat([t.tensor([0]), changes, t.tensor([T])])
        seg_token_start = boundary_positions[:-1].to(t.int32)
        seg_token_end = boundary_positions[1:].to(t.int32)

        # When the last doc in a row is shorter than the remaining space,
        # trailing positions stay zero-filled (see _pack_row_with_provenance).
        # The seg_map for those positions stays at 0, creating an extra run.
        # Trim to actual content.
        if len(seg_token_start) == n_segs + 1:
            # Last run is padding — drop it and adjust last segment's end
            seg_token_start = seg_token_start[:n_segs]
            seg_token_end = seg_token_end[:n_segs]
            seg_token_end[-1] = T
        elif len(seg_token_start) != n_segs:
            raise AssertionError(
                f"Expected {n_segs} contiguous segments in seg_map but found "
                f"{len(seg_token_start)} runs."
            )
    else:
        seg_token_start = t.tensor([], dtype=t.int32)
        seg_token_end = t.tensor([], dtype=t.int32)
    if profile_detail is not None:
        profile_detail["segment_boundaries_seconds"] += time.monotonic() - boundaries_started

    # Direct concatenation — no padding, no mask extraction.
    concat_started = time.monotonic() if profile_detail is not None else 0.0
    operation_started = time.monotonic() if profile_detail is not None else 0.0
    all_active_fids = t.cat(per_seg_fids) if per_seg_fids else t.tensor([], dtype=t.int64)
    if profile_detail is not None:
        profile_detail["feature_ids_concat_seconds"] += time.monotonic() - operation_started
    operation_started = time.monotonic() if profile_detail is not None else 0.0
    raw_pcts = t.cat(per_seg_pcts) if per_seg_pcts else t.tensor([], dtype=t.uint8)
    if profile_detail is not None:
        profile_detail["percentiles_concat_seconds"] += time.monotonic() - operation_started
    if raw_pcts.dtype != t.uint8:
        raise PackedRowInvariantError(
            f"packed row received non-uint8 percentile codes: dtype={raw_pcts.dtype}"
        )
    operation_started = time.monotonic() if profile_detail is not None else 0.0
    feat_counts = t.tensor([len(f) for f in per_seg_fids], dtype=t.int32)
    if profile_detail is not None:
        profile_detail["feature_counts_seconds"] += time.monotonic() - operation_started
        profile_detail["feature_concat_seconds"] += time.monotonic() - concat_started

    finalize_started = time.monotonic() if profile_detail is not None else 0.0
    result = _RowCSR(
        tokens=row_tokens[: T + 1].to(t.uint16),
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id_row.to(t.uint8),
        # O3 (issue #531): store the row's feature_ids at their final uint16 dtype rather than
        # int64. flush_shard already does `cat(...).to(uint16)`, so the assembled-shard bytes
        # are identical (SAE ids < 65536 ⇒ the int64→uint16 round-trip is lossless), but the
        # accumulator holding R rows now uses ~4× less memory (the dominant per-shard tensor:
        # ~109k feats/row × 2 B vs 8 B), cutting the load-vs-accumulate memory pressure.
        feature_ids=all_active_fids.to(t.uint16),
        feature_pcts=raw_pcts,
        feat_counts=feat_counts,
    )
    if profile_detail is not None:
        profile_detail["row_tensor_finalize_seconds"] += time.monotonic() - finalize_started
    return result


def _row_provenance(
    row_csr: _RowCSR,
    row_doc_info: list,
    row_doc_prov: list,
) -> t.Tensor:
    """Per-segment ``(base_shard_idx, doc_idx, seg_ordinal)`` for one packed row.

    ``_build_row_segments`` emits each placed doc's surviving segments as a
    PREFIX (ordinals 0..n-1) in placement order, and ``seg_doc_id`` is the
    within-row doc index — so per-doc segment counts recover the ordinals.
    Returns an ``(n_segs, 3)`` int32 tensor.

    Every present doc has >= 1 feature segment (the FeatureIndex raises for an
    absent doc), so ``n_ext_segs == 0`` is a BUG — this raises rather than
    emit a sentinel/filler triple that would silently hide it.

    A placed doc may legitimately contribute ZERO packed segments: best-fit
    packing can crop a doc down to a single token landing in the T+1-th slot
    (the trailing next-token-prediction target, outside the ``[0, T)`` segment
    range). Such a doc has features but no packed segment in this row, so it
    correctly emits no provenance triple (``count == 0`` is fine; ``count >
    n_ext_segs`` is not — that breaks the prefix-ordinal invariant).
    """
    counts = t.bincount(row_csr.seg_doc_id.to(t.int64), minlength=len(row_doc_info)).tolist()
    # seg_doc_id is the within-row doc index, so every value must be < the
    # number of placed docs; bincount(minlength=...) returns extra bins when a
    # value exceeds that, which would IndexError on row_doc_info[d] below. Turn
    # that structural break into a typed error rather than a bare IndexError.
    if len(counts) > len(row_doc_info):
        raise PackedRowInvariantError(
            f"packing emitted a seg_doc_id {len(counts) - 1} >= placed-doc count "
            f"{len(row_doc_info)} — seg_doc_id must index into the row's docs."
        )
    triples: list[tuple[int, int, int]] = []
    for d, count in enumerate(counts):
        seg_features = row_doc_info[d][1]
        n_ext_segs = len(seg_features) if seg_features else 0
        prov = row_doc_prov[d]
        if prov is None:
            raise PackedRowInvariantError(
                "packing emitted a row without doc provenance — buffer entries "
                "must carry (base_shard_idx, doc_idx) identities."
            )
        base_idx, doc_idx = prov
        if n_ext_segs == 0:
            raise PackedRowInvariantError(
                f"packed doc (base shard idx {base_idx}, doc {doc_idx}) has "
                f"zero feature segments — every placed doc must have features "
                f"(the FeatureIndex raises for an absent doc). This is a "
                f"packing/extraction bug, not a filler case."
            )
        if count > n_ext_segs:
            raise PackedRowInvariantError(
                f"packing emitted {count} segments for a doc with only "
                f"{n_ext_segs} base segments (base shard idx {base_idx}, "
                f"doc {doc_idx}) — the prefix-ordinal invariant is broken."
            )
        triples.extend((base_idx, doc_idx, s) for s in range(count))
    if not triples:
        return t.zeros((0, 3), dtype=t.int32)
    return t.tensor(triples, dtype=t.int32)


def _write_provenance_shard(
    out_path: Path,
    prov_rows: list[t.Tensor],
    *,
    expected_segments: int | None = None,
    zstd_level: int | None = None,
) -> None:
    """Write one provenance shard from per-row ``(n_segs, 3)`` tensors."""
    prov = t.cat(prov_rows) if prov_rows else t.zeros((0, 3), dtype=t.int32)
    if expected_segments is not None and int(prov.shape[0]) != expected_segments:
        raise PackedRowInvariantError(
            f"{out_path.name}: {int(prov.shape[0])} provenance entries for "
            f"{expected_segments} packed segments — packer invariant broken."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_overlay_shard(
        {
            PROVENANCE_BASE_SHARD_IDX_KEY: prov[:, 0].contiguous(),
            PROVENANCE_DOC_IDX_KEY: prov[:, 1].contiguous(),
            PROVENANCE_SEG_ORDINAL_KEY: prov[:, 2].contiguous(),
        },
        out_path,
        zstd_level=zstd_level,
    )


def _assert_base_stems_aligned(globbed_stems: list[str], index_stems: list[str]) -> None:
    """Enforce that the manifest's ``base_shard_stems`` (from ``glob_features``)
    match ``feature_index.shard_stems`` in ORDER.

    Provenance ``base_shard_idx`` is built from
    ``enumerate(feature_index.shard_stems)`` and the manifest records
    ``globbed_stems`` for consumers to index into; a divergence would silently
    mislabel every provenance triple's base shard. Both are the sorted
    base-shard glob — assert it rather than trust the convention.
    """
    if globbed_stems != index_stems:
        raise PackedRowInvariantError(
            "base-shard stem order from glob_features does not match "
            "feature_index.shard_stems; provenance base_shard_idx would point "
            f"at the wrong base shard. glob={globbed_stems!r} index={index_stems!r}"
        )


def _assert_resume_packing_identity_unchanged(
    output_dir: Path,
    *,
    tok_checksum: str,
    buffer_size: int,
    rows_per_shard: int,
) -> None:
    """Refuse replay when payload identity is absent or differs from this run."""
    if not (glob_overlay_shards(output_dir) or glob_overlay_shards(provenance_dir(output_dir))):
        return

    prior_manifest = read_manifest(output_dir)
    metadata = prior_manifest.get("metadata") if prior_manifest is not None else None
    if not isinstance(metadata, dict):
        raise PackedRowInvariantError(
            f"resume over {output_dir}: packed or provenance payloads exist but "
            "manifest.json is missing, unreadable, or lacks a metadata dict; "
            "the resume replay cannot authenticate their packing identity."
        )

    expected = {
        "tok_checksum": tok_checksum,
        "buffer_size": buffer_size,
        "rows_per_shard": rows_per_shard,
    }
    for field, current in expected.items():
        if field not in metadata:
            raise PackedRowInvariantError(
                f"resume over {output_dir}: manifest metadata is missing required "
                f"packing identity field {field!r}; the resume replay cannot "
                "authenticate existing payloads."
            )
        prior = metadata[field]
        if type(prior) is type(current) and prior == current:
            continue
        if field == "tok_checksum" and isinstance(prior, str):
            raise PackedRowInvariantError(
                f"resume over {output_dir}: existing packed rows were built "
                f"with tokenizer {prior[:16]}... but the current tokenizer "
                f"hashes to {tok_checksum[:16]}... — the resume replay cannot "
                f"reproduce the original packing. Delete the dir to rebuild, or "
                f"restore the original tokenizer."
            )
        raise PackedRowInvariantError(
            f"resume over {output_dir}: existing packed rows were built with "
            f"{field}={prior!r}, but the current run uses {field}={current!r}; "
            "the resume replay cannot reproduce the original packing."
        )


def _assert_no_existing_packed_payloads(output_dir: Path) -> None:
    """Refuse a fresh run that would leave or overwrite physical payloads."""
    packed = glob_overlay_shards(output_dir)
    provenance = glob_overlay_shards(provenance_dir(output_dir))
    if not packed and not provenance:
        return
    raise PackedRowInvariantError(
        f"Fresh packed-row output {output_dir} already contains "
        f"{len(packed)} packed and {len(provenance)} provenance shard(s). "
        "Use resume=true, or explicitly move/delete the old output before "
        "starting a fresh run."
    )


def _assemble_and_save(
    shard_idx: int,
    shard_rows: list[_RowCSR],
    shard_prov: list[t.Tensor],
    out_path: Path,
    prov_path: Path,
    zstd_level: int | None,
    dims_path: Path,
) -> tuple[int, int, int, ShardSaveTiming]:
    """Assemble a ``PackedRowShard`` from accumulated per-row CSR, compress+write it, write its
    segment-provenance sidecar, and compute its max dims (issue #531 O1 + PR #528 provenance).

    A PURE function of its inputs — no shared/closure state — so it is safe to run in a worker
    thread of the compression pool. zstd (and the large ``cat``/``stack``) release the GIL, and
    each shard writes its OWN ``shard_{idx}`` files (packed + provenance), so the output bytes are
    identical regardless of which thread runs this or in what order shards finish (byte-identical
    to the serial flush). The provenance shard rides in this same worker (packed first, then its
    mapping) so the two artifacts are written together and can never drift apart. Returns
    ``(shard_idx, max_n_seg, max_seg_size, timing)``; the producer reduces the global maxima from
    these (``max`` is order-independent).
    """
    worker_started = time.monotonic()
    R = len(shard_rows)
    t_assemble = time.monotonic()
    # Build CSR offsets from per-row segment/feature counts
    seg_counts = t.tensor([row.n_segs for row in shard_rows], dtype=t.int32)
    row_seg_offsets = t.zeros(R + 1, dtype=t.int32)
    row_seg_offsets[1:] = seg_counts.cumsum(0)

    all_feat_counts = t.cat([row.feat_counts for row in shard_rows])
    seg_feat_offsets = t.zeros(int(all_feat_counts.shape[0]) + 1, dtype=t.int32)
    seg_feat_offsets[1:] = all_feat_counts.cumsum(0)

    shard = PackedRowShard(
        tokens=t.stack([row.tokens for row in shard_rows]),
        row_seg_offsets=row_seg_offsets,
        seg_token_start=t.cat([row.seg_token_start for row in shard_rows]).to(t.uint16),
        seg_token_end=t.cat([row.seg_token_end for row in shard_rows]).to(t.uint16),
        seg_doc_id=t.cat([row.seg_doc_id for row in shard_rows]),
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=t.cat([row.feature_ids for row in shard_rows]).to(t.uint16),
        feature_pcts=t.cat([row.feature_pcts for row in shard_rows]),
    )
    assemble_seconds = time.monotonic() - t_assemble

    t_save = time.monotonic()
    save_packed_row_shard(shard, out_path, zstd_level=zstd_level)
    save_seconds = time.monotonic() - t_save
    logger.info(
        f"  Shard {shard_idx}: {R} rows, {shard.total_segments} segs, {shard.total_features} feats"
    )

    # Segment provenance rides in the SAME worker as the packed shard (packed
    # first, then its mapping) so the two artifacts can never drift apart; the
    # segment count is cross-checked against the packed shard's total.
    _write_provenance_shard(
        prov_path,
        shard_prov,
        expected_segments=shard.total_segments,
        zstd_level=zstd_level,
    )

    max_n_seg, max_seg_size = _compute_shard_max_dims(shard)
    # Issue #539: persist this shard's max dims atomically beside it, so a later resume can seed
    # the global maxima without re-decompressing shards — keeping the manifest correct (never
    # tail-only or zero). Each worker writes its OWN companion ⇒ no lock, crash-safe.
    atomic_json_dump({"max_n_seg": max_n_seg, "max_seg_size": max_seg_size}, dims_path)
    timing = ShardSaveTiming(
        shard_idx=shard_idx,
        n_rows=R,
        n_segments=shard.total_segments,
        n_features=shard.total_features,
        assemble_seconds=assemble_seconds,
        save_seconds=save_seconds,
        worker_seconds=time.monotonic() - worker_started,
        save_bytes=out_path.stat().st_size,
    )
    return shard_idx, max_n_seg, max_seg_size, timing


def precompute_packed_rows(
    parquet_paths: list[str],
    tokenizer,
    feature_index: _PackedRowsFeatureLookup,
    context_len: int,
    output_dir: Path,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    rows_per_shard: int = 16384,
    resume: bool = False,
    resume_from_shard: int | None = None,
    zstd_level: int | None = None,
    compress_workers: int = 1,
    profile: PackedRowsProfile | None = None,
    remote_checkpoint: _RemotePackingCheckpointContext | None = None,
    offloader: PackedRowOffloader | None = None,
) -> tuple[int, int, int]:
    """Run the packing loop and write packed row shards.

    Args:
        resume_from_shard: Optional explicit output shard index to resume
            from. When set, the replay skip point is
            ``resume_from_shard * rows_per_shard`` rows and local shard-file
            counts are ignored. This assumes every preceding shard
            ``0..resume_from_shard-1`` was full. Existing local shards with
            index >= resume_from_shard may be overwritten with byte-identical
            regenerated data.
        offloader: Optional publish-then-evict sink. When set, every flushed
            shard is queued for verified publication to the bucket and its
            local bodies are removed once published, so a family far larger
            than its filesystem still completes. Already-offloaded shards are
            resume-complete from their receipts alone.

    Returns:
        Tuple of total rows processed (including replay-skipped rows), global
        max segment count, and global max segment size for newly written
        shards.
    """
    if resume_from_shard is not None:
        if isinstance(resume_from_shard, bool) or not isinstance(resume_from_shard, int):
            raise ValueError(
                f"resume_from_shard must be a non-negative int or None, got "
                f"{resume_from_shard!r} ({type(resume_from_shard).__name__})"
            )
        if resume_from_shard < 0:
            raise ValueError(
                f"resume_from_shard must be a non-negative int or None, got {resume_from_shard}"
            )

    T = context_len
    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()

    doc_cursor = _DocumentCursor()
    doc_iter = _raw_document_iterator(
        parquet_paths,
        tokenizer,
        bos_token,
        feature_index,
        profile=profile,
        cursor=doc_cursor,
    )
    doc_buffer = SortedDocBuffer()
    seg_size = PositiveInt.parse(feature_index.segment_size or 128)

    # Provenance identity: index docs by the feature dir's canonical stem order
    # (the same list the packed manifest records as base_shard_stems).
    stem_to_idx = {s: i for i, s in enumerate(feature_index.shard_stems)}
    prov_out = provenance_dir(output_dir)

    # Accumulators for current shard — collect per-row tensors, concatenate at flush.
    shard_rows: list[_RowCSR] = []
    shard_prov: list[t.Tensor] = []

    shard_idx = 0
    total_rows = 0
    global_max_n_seg = 0
    global_max_seg_size = 0
    data_exhausted = False
    documents_loaded = 0
    documents_consumed = 0
    t_start = time.monotonic()

    def _finalize_profile() -> None:
        """Accumulate run totals into the shared profile (idempotent per call; this
        function is invoked at every return point). Accumulates across the train+val
        calls run() makes with one shared profile."""
        if profile is None:
            return
        profile.totals["total_rows"] += total_rows
        profile.totals["wall_seconds"] += time.monotonic() - t_start
        profile.totals["n_shards_written"] = len(profile.per_output_shard)
        wall = profile.totals["wall_seconds"]
        profile.totals["rows_per_sec"] = profile.totals["total_rows"] / wall if wall > 0 else 0.0

    def _record_replay(t0: float, rows_skipped: int) -> None:
        """Record the resume-replay cost (the count-based re-pack of already-written rows).
        ACCUMULATES into the shared profile so the train + val calls run() makes with one
        profile_obj both contribute (issue #540: a plain assignment let val clobber train's
        record)."""
        if profile is not None:
            if loaded_checkpoint is None:
                if remote_checkpoint is None:
                    profile.resume["mode"] = "replay"
                else:
                    cast(dict[str, str], profile.resume["mode"])[resume_profile_key] = "replay"
            profile.resume["replay_seconds"] += time.monotonic() - t0
            profile.resume["rows_replayed"] += rows_skipped

    def _refill_document_buffer(*, profile_append: bool) -> bool:
        """Fill the lookahead buffer, preserving its tail at end-of-input.

        Returns whether at least one buffered document remains packable. Once
        the source iterator is exhausted, later calls only drain the existing
        buffer; they never discard it or probe the exhausted iterator again.
        """
        nonlocal data_exhausted, documents_loaded
        while not data_exhausted and len(doc_buffer) < buffer_size:
            try:
                text_tokens, packed_seg_features, shard_stem, doc_idx = next(doc_iter)
            except StopIteration:
                data_exhausted = True
                break
            t_taa = time.monotonic()
            text_tokens = truncate_to_boundary_coverage(
                text_tokens,
                packed_seg_features,
                T,
                seg_size,
                doc_id=f"{shard_stem}:{doc_idx}",
            )
            assert_boundaries_covered(text_tokens, packed_seg_features)
            doc_buffer.append(
                (
                    text_tokens,
                    packed_seg_features,
                    (stem_to_idx[shard_stem], doc_idx),
                )
            )
            documents_loaded += 1
            if profile_append and profile is not None:
                profile.stage_totals["truncate_assert_append"] += time.monotonic() - t_taa
        return bool(doc_buffer)

    def _record_consumed_documents(row_doc_info: list) -> None:
        nonlocal documents_consumed
        documents_consumed += len(row_doc_info)

    def _assert_all_input_documents_consumed() -> None:
        if documents_loaded == 0:
            raise PackedRowInvariantError(
                "packed-row producer read zero input documents; refusing to "
                "emit an empty packed-row artifact"
            )
        if doc_buffer or documents_consumed != documents_loaded:
            raise PackedRowInvariantError(
                "packed-row producer finished with unconsumed input documents: "
                f"loaded={documents_loaded}, consumed={documents_consumed}, "
                f"buffered={len(doc_buffer)}"
            )

    # Resume: advance the doc stream past already-written rows by re-packing them (the replay).
    # Two entry points share ONE replay loop: an explicit resume_from_shard=N (skip N shards,
    # ignoring local file count), or resume=true counting the CONTIGUOUS shard_00000..M-1 prefix.
    # The contiguous count (not len(glob)) is required because O1's parallel compress finishes
    # shards out of order, so a crash with compress_workers>1 can leave a gap; resuming from the
    # first missing index re-packs the gap shard (and rewrites later shards byte-identically).
    skip_rows = 0
    n_kept = 0
    do_replay = False
    loaded_checkpoint: _LoadedPackingCheckpoint | None = None
    # Offloaded shards are resume-complete without local bodies.
    offloaded = frozenset() if offloader is None else frozenset(offloader.receipts(output_dir))
    resume_profile_key = remote_checkpoint.split if remote_checkpoint is not None else "local"
    if profile is not None and remote_checkpoint is not None:
        if not isinstance(profile.resume["mode"], dict):
            profile.resume["mode"] = {}
        cast(dict[str, str], profile.resume["mode"]).setdefault(resume_profile_key, "none")
    if resume_from_shard is not None:
        n_kept = resume_from_shard
        skip_rows = resume_from_shard * rows_per_shard
        do_replay = True
        logger.info(
            f"Resume: skipping {skip_rows} rows from explicit resume_from_shard={resume_from_shard}"
        )
    elif resume:
        # Count the contiguous prefix of COMPLETE shards (packed + provenance;
        # see _packed_shard_complete). An incomplete trailing shard is NOT
        # counted, so resume rewrites it byte-identically AND (re)writes its
        # provenance via the normal flush. Remote checkpoint-seek additionally
        # requires the _dims companion (see the predicate's docstring).
        while _packed_shard_complete(
            output_dir,
            n_kept,
            require_dims=remote_checkpoint is not None,
            offloaded=offloaded,
        ):
            n_kept += 1
        if n_kept > 0 and remote_checkpoint is not None:
            checkpoint_started = time.monotonic()
            loaded_checkpoint = _load_nearest_eligible_packing_checkpoint(
                remote_checkpoint,
                n_kept=n_kept,
                compress_workers=max(1, compress_workers),
            )
            if loaded_checkpoint is not None:
                header = loaded_checkpoint.header
                checkpoint_shard_idx = cast(int, header["shard_idx"])
                shard_idx = n_kept
                total_rows = cast(int, header["total_rows"])
                documents_loaded = cast(int, header["documents_loaded"])
                documents_consumed = cast(int, header["documents_consumed"])
                data_exhausted = loaded_checkpoint.data_exhausted
                doc_cursor = _DocumentCursor(
                    parquet_file_idx=cast(int, header["parquet_file_idx"]),
                    row_group_idx=cast(int, header["row_group_idx"]),
                    rows_consumed_in_group=cast(int, header["rows_consumed_in_group"]),
                )
                doc_iter = _raw_document_iterator(
                    parquet_paths,
                    tokenizer,
                    bos_token,
                    feature_index,
                    profile=profile,
                    cursor=doc_cursor,
                )
                for entry in loaded_checkpoint.buffer_entries:
                    doc_buffer.append(entry)
                if isinstance(feature_index, RemotePackedRowsFeatureLookup):
                    feature_index.seed_activated_before(loaded_checkpoint.consumed_bodies)
                _smn, _sms = _seed_max_dims_from_existing(output_dir, n_kept)
                global_max_n_seg = max(global_max_n_seg, _smn)
                global_max_seg_size = max(global_max_seg_size, _sms)
                skip_rows = (n_kept - checkpoint_shard_idx) * rows_per_shard
                do_replay = skip_rows > 0
                if profile is not None:
                    cast(dict[str, str], profile.resume["mode"])[resume_profile_key] = (
                        "checkpoint_seek"
                    )
                    profile.resume["checkpoint_seek_seconds"] = profile.resume.get(
                        "checkpoint_seek_seconds", 0.0
                    ) + (time.monotonic() - checkpoint_started)
                logger.info(
                    f"Resume: restored {remote_checkpoint.split} checkpoint at shard "
                    f"{checkpoint_shard_idx} for {n_kept} kept shards; seeking document and "
                    f"remote cursors with {len(doc_buffer)} buffered docs and replaying "
                    f"{skip_rows} row(s)"
                )
        if n_kept > 0 and loaded_checkpoint is None:
            skip_rows = n_kept * rows_per_shard
            do_replay = True
            logger.info(f"Resume: skipping {skip_rows} rows ({n_kept} complete existing shards)")

    if do_replay:
        shard_idx = n_kept
        # Issue #539: seed the global maxima from the kept shards' _dims companions so the manifest
        # reflects the true max over existing+new shards (not the post-resume tail only / zero).
        _smn, _sms = _seed_max_dims_from_existing(output_dir, n_kept)
        global_max_n_seg = max(global_max_n_seg, _smn)
        global_max_seg_size = max(global_max_seg_size, _sms)
        # Consume documents to advance the iterator past the already-precomputed rows.
        rows_skipped = 0
        t_replay = time.monotonic()
        while rows_skipped < skip_rows:
            if not _refill_document_buffer(profile_append=False):
                logger.warning(
                    f"Data exhausted while skipping for resume (skipped {rows_skipped}/{skip_rows})"
                )
                _assert_all_input_documents_consumed()
                _record_replay(t_replay, rows_skipped)
                _finalize_profile()
                return total_rows, global_max_n_seg, global_max_seg_size
            _row_tokens, row_doc_info, _row_doc_prov = _pack_row_with_provenance(
                doc_buffer, row_capacity
            )
            _record_consumed_documents(row_doc_info)
            rows_skipped += 1
            total_rows += 1
        _record_replay(t_replay, rows_skipped)

    # O1 (issue #531): parallelize the per-shard assemble + zstd save across a bounded thread
    # pool. The producer (this thread) runs the order-dependent work strictly serially — buffer
    # refill + the SortedDocBuffer-stateful packing decision; once a shard's rows are fixed,
    # assembling + compressing them is a pure function of those rows, so it runs
    # off-thread (zstd + the big cat/stack release the GIL; threads share the row objects, so
    # NO IPC). Each shard writes its OWN file ⇒ byte-identical regardless of worker count or
    # finish order. A bounded semaphore caps in-flight shards (≈ compress_workers shards of
    # memory). compress_workers <= 1 still overlaps the NEXT shard's packing with the current
    # shard's compression.
    n_compress = max(1, compress_workers)
    pool = ThreadPoolExecutor(max_workers=n_compress)
    inflight = threading.BoundedSemaphore(n_compress)
    futures: list[Future] = []
    queue_wait_by_idx: dict[int, float] = {}
    inflight_count = 0
    inflight_count_lock = threading.Lock()

    def release_inflight(_future: Future) -> None:
        nonlocal inflight_count
        with inflight_count_lock:
            inflight_count -= 1
            assert inflight_count >= 0, "compression in-flight count underflow"
        inflight.release()

    def flush_shard() -> None:
        nonlocal shard_idx, shard_rows, shard_prov, inflight_count
        if not shard_rows:
            return
        out_path = overlay_shard_path(output_dir, f"shard_{shard_idx:05d}")
        # Provenance shard: same shard_{NNNNN} name, under the provenance/ subdir.
        prov_path = overlay_shard_path(prov_out, f"shard_{shard_idx:05d}")
        # Issue #539: each shard's max dims go in a sibling _dims/ companion (same index). The
        # _dims/ dir is created by atomic_json_dump (its parent-mkdir) in the worker, so no
        # producer-side mkdir is needed here.
        dims_path = _dims_companion_path(output_dir, shard_idx)
        rows, prov, idx = shard_rows, shard_prov, shard_idx
        shard_rows = []
        shard_prov = []
        shard_idx += 1
        # Backpressure: block until a compression slot frees up, bounding peak memory.
        queue_wait_started = time.monotonic() if profile is not None else 0.0
        inflight.acquire()
        if profile is not None:
            queue_wait_seconds = time.monotonic() - queue_wait_started
            queue_wait_by_idx[idx] = queue_wait_seconds
            profile.compression["backpressure_seconds"] += queue_wait_seconds
        # Release the permit if submit itself fails (e.g. a MemoryError/thread-spawn failure
        # under the load-vs-accumulate pressure O1 targets), since the done-callback that would
        # release it is never attached — keep acquire/release balanced on every path.
        try:
            fut = pool.submit(
                _assemble_and_save, idx, rows, prov, out_path, prov_path, zstd_level, dims_path
            )
        except BaseException:
            inflight.release()
            raise
        if profile is not None:
            # Register only after the count is visible. If ``fut`` already completed,
            # add_done_callback invokes synchronously here and observes the increment.
            with inflight_count_lock:
                inflight_count += 1
                profile.compression["max_inflight_shards"] = max(
                    profile.compression["max_inflight_shards"], inflight_count
                )
            fut.add_done_callback(release_inflight)
        else:
            fut.add_done_callback(lambda _future: inflight.release())
        futures.append(fut)
        if offloader is not None:
            # Queued against the compression future, so publication starts the
            # moment the bodies exist. ``submit`` blocks once too many shards
            # await publication: that backpressure is what actually bounds
            # local disk, so it is deliberately on the producer thread.
            offloader.submit(output_dir, idx, fut)
        if remote_checkpoint is not None:
            checkpoint_write_started = time.monotonic()
            try:
                _write_packing_checkpoint(
                    remote_checkpoint,
                    shard_idx=shard_idx,
                    total_rows=total_rows,
                    documents_loaded=documents_loaded,
                    documents_consumed=documents_consumed,
                    cursor=doc_cursor,
                    doc_buffer=doc_buffer,
                    compress_workers=n_compress,
                )
            finally:
                if profile is not None:
                    profile.resume["checkpoint_write_seconds"] += (
                        time.monotonic() - checkpoint_write_started
                    )

    try:
        while _refill_document_buffer(profile_append=True):
            # Pack one row (carrying each placed doc's base identity)
            t_pack = time.monotonic()
            row_tokens, row_doc_info, row_doc_prov = _pack_row_with_provenance(
                doc_buffer, row_capacity
            )
            _record_consumed_documents(row_doc_info)
            if profile is not None:
                profile.stage_totals["pack"] += time.monotonic() - t_pack

            # Convert to CSR and accumulate, with the row's segment provenance
            t_csr = time.monotonic()
            row_csr = _row_to_csr(
                row_tokens,
                row_doc_info,
                T,
                tokenizer,
                profile_detail=profile.row_build_detail if profile is not None else None,
            )
            if profile is not None:
                profile.row_build_detail["row_to_csr_seconds"] += time.monotonic() - t_csr
            shard_rows.append(row_csr)
            t_provenance = time.monotonic() if profile is not None else 0.0
            row_provenance = _row_provenance(row_csr, row_doc_info, row_doc_prov)
            shard_prov.append(row_provenance)
            if profile is not None:
                profile.row_build_detail["row_provenance_seconds"] += (
                    time.monotonic() - t_provenance
                )
                # Preserve the established aggregate for downstream dashboards.
                profile.stage_totals["csr_build"] += time.monotonic() - t_csr

            total_rows += 1

            if total_rows % rows_per_shard == 0:
                flush_shard()

            if total_rows % 1000 == 0:
                elapsed = time.monotonic() - t_start
                rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
                logger.info(f"  {total_rows} rows packed ({rows_per_sec:.0f} rows/s)")

        _assert_all_input_documents_consumed()
        flush_shard()

        # Join all compress futures IN ORDER: reduce the global maxima (max is order-
        # independent) and collect per-shard timings. fut.result() re-raises any worker
        # exception (aborting the run loudly).
        for fut in futures:
            _idx, shard_mns, shard_mss, timing = fut.result()
            if profile is not None:
                timing.queue_wait_seconds = queue_wait_by_idx[_idx]
            global_max_n_seg = max(global_max_n_seg, shard_mns)
            global_max_seg_size = max(global_max_seg_size, shard_mss)
            if profile is not None:
                profile.add_output_shard(timing)
        if offloader is not None:
            # Publication finishes before this split's manifest is written, so
            # the manifest counts a family whose bodies are all durably placed.
            offloader.drain()
    finally:
        pool.shutdown()

    elapsed = time.monotonic() - t_start
    logger.info(f"Done: {total_rows} rows in {shard_idx} shards ({elapsed:.0f}s)")
    _finalize_profile()
    return total_rows, global_max_n_seg, global_max_seg_size


def _run_local(cfg: DictConfig) -> int:
    feature_dir = resolve_feature_dir(cfg)
    if not feature_dir.is_dir():
        logger.error(f"Feature directory not found: {feature_dir}")
        return 1

    # Cross-stage segment_size check (Wave 3D task 4).
    verify_segment_size(feature_dir, expected_segment_size=int(cfg.segment_size))

    context_len = int(cfg.context_len)
    n_sae_features = PositiveInt.parse(SAE_WIDTH_TO_NUM_FEATURES[SAEWidth(str(cfg.sae_width))])
    max_shards = cfg.max_shards
    if max_shards is not None:
        max_shards = int(max_shards)
    buffer_size = int(cfg.buffer_size)
    rows_per_shard = int(cfg.rows_per_shard)
    profile = bool(cfg.profile)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer()
    tok_checksum = compute_tokenizer_checksum(tokenizer)
    logger.info(f"Tokenizer checksum: {tok_checksum[:CHECKSUM_PREFIX_LEN]}...")

    # Load percentile checksum
    stats_path = find_statistics_path(feature_dir)
    pct_checksum = load_percentile_checksum(stats_path) if stats_path else None
    if pct_checksum is None:
        logger.error("No percentile checksum found — run precompute_percentiles first")
        return 1
    logger.info(f"Percentile checksum: {pct_checksum[:CHECKSUM_PREFIX_LEN]}...")

    # Create feature index
    feature_index_scan_started = time.monotonic()
    feature_index = FeatureIndex(
        str(feature_dir),
        # O2 (issue #531): prefetch the next base shard (sorted-stem order, == the doc-stream
        # order) in a daemon thread so its ~25 s load+index-build overlaps the current shard's
        # packing instead of stalling the loop. Byte-identical (changes when, not what).
        prefetch=True,
        percentile_checksum=pct_checksum,
        tokenizer_checksum=tok_checksum,
        # O4 (issue #531): the packing CSR uses feature_ids + precomputed percentiles, never
        # feature_mags — skip loading/upcasting/splitting the large mags tensor per shard.
        skip_feature_mags=True,
        profile_loads=profile,
    )
    feature_index_scan_seconds = time.monotonic() - feature_index_scan_started
    logger.info(f"Precomputed percentiles: {feature_index.has_precomputed_percentiles}")
    logger.info(f"Precomputed token boundaries: {feature_index.has_precomputed_token_boundaries}")

    # Resolve output directory
    # Path: pct/<pct_hash>/tok/<tok_hash>/c{context_len}/s{max_shards}/
    pct_dir = Path(feature_dir) / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
    tok_sub = pct_dir / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
    shards_str = f"s{max_shards}" if max_shards is not None else "sall"
    output_dir = tok_sub / f"c{context_len}" / shards_str
    val_output_dir = output_dir.parent / f"val_{output_dir.name}"
    logger.info(f"Output directory: {output_dir}")

    # Get parquet paths
    data_dir = cfg.data_dir
    parquet_paths = list_parquet_files(data_dir)
    # Apply max_shards BEFORE train/val split (same logic as the training dataloader)
    # so that the last shard in the limited set becomes the val shard.
    if max_shards is not None:
        parquet_paths = parquet_paths[:max_shards]
    if len(parquet_paths) < 2:
        logger.error(
            f"Need at least 2 parquet shards for train/val split, got {len(parquet_paths)}. "
            f"Use max_shards >= 2."
        )
        return 1
    train_parquet_paths = parquet_paths[:-1]  # training split (exclude last = val)
    val_parquet_paths = parquet_paths[-1:]  # val split (last shard only)

    zstd_level = cfg.zstd_level
    resume = bool(cfg.resume)
    resume_from_shard = cfg.resume_from_shard
    # O1 (issue #531): parallel shard compress+save. 0 => auto = min(cpu//4, 8); the output is
    # byte-identical across all values (incl. 1, which still overlaps packing with compression).
    compress_workers = int(cfg.compress_workers)
    if compress_workers == 0:
        compress_workers = max(1, min((os.cpu_count() or 4) // 4, 8))
    logger.info(f"Compress workers: {compress_workers}")
    # profile=true runs the full (max_shards-bounded) packing with stage-level instrumentation
    # and writes a JSON sidecar (the sidecar needs real per-output-shard saves, so it does NOT
    # early-return).
    profile_summary_output = cfg.profile_summary_output
    profile_obj = (
        PackedRowsProfile(
            config={
                "context_len": context_len,
                "max_shards": max_shards,
                "buffer_size": buffer_size,
                "rows_per_shard": rows_per_shard,
                "zstd_level": zstd_level,
                "compress_workers": compress_workers,  # the RESOLVED value (0 => auto), not raw cfg
            }
        )
        if profile
        else None
    )
    if profile_obj is not None:
        profile_obj.startup["feature_index_scan_seconds"] = feature_index_scan_seconds

    # Base-shard identity for provenance + the manifest. The stem ORDER must
    # match feature_index.shard_stems — provenance base_shard_idx is built from
    # stem_to_idx = enumerate(feature_index.shard_stems) (inside
    # precompute_packed_rows), and the manifest records THIS list for consumers
    # to index into; a divergence would silently mislabel every triple's base
    # shard. Both are the sorted base-shard glob, but enforce it rather than
    # trust the convention. Hoisted above _manifest_metadata so the closure
    # never reads it before assignment. (base_shard_paths is also reused for the
    # O6 base-hash below — glob is cheap; the hash is what must run once.)
    base_shard_paths = glob_features(feature_dir)
    base_shard_stems = [shard_stem_from_path(p) for p in base_shard_paths]
    _assert_base_stems_aligned(base_shard_stems, list(feature_index.shard_stems))

    def _manifest_metadata(max_n_seg, max_seg_size, packed_dir: Path):
        return {
            "type": "packed_rows",
            "context_len": context_len,
            "max_shards": max_shards,
            "buffer_size": buffer_size,
            "rows_per_shard": rows_per_shard,
            "pct_checksum": pct_checksum,
            "tok_checksum": tok_checksum,
            "max_n_seg": max_n_seg,
            # Raw observed segment maximum. Consumers preserve this exact
            # semantic/resume bound and align only their private allocations.
            "max_seg_size": max_seg_size,
            # Segment provenance (written by this packer in the same pass).
            # base_shard_stems defines the base_shard_idx mapping; consumers
            # require provenance_shards == the packed shard count.
            "base_shard_stems": base_shard_stems,
            "provenance_shards": len(glob_overlay_shards(provenance_dir(packed_dir))),
        }

    # Preflight both splits before either split can write.
    if not resume and resume_from_shard is None:
        _assert_no_existing_packed_payloads(output_dir)
        _assert_no_existing_packed_payloads(val_output_dir)
    else:
        _assert_resume_packing_identity_unchanged(
            output_dir,
            tok_checksum=tok_checksum,
            buffer_size=buffer_size,
            rows_per_shard=rows_per_shard,
        )
    if resume:
        _assert_resume_packing_identity_unchanged(
            val_output_dir,
            tok_checksum=tok_checksum,
            buffer_size=buffer_size,
            rows_per_shard=rows_per_shard,
        )

    # Create output directory (write_manifest will be called after packing).
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Train split ---
    logger.info(f"Processing TRAIN split: {len(train_parquet_paths)} parquet shards → {output_dir}")
    total_rows, g_max_n_seg, g_max_seg_size = precompute_packed_rows(
        parquet_paths=train_parquet_paths,
        tokenizer=tokenizer,
        feature_index=feature_index,
        context_len=context_len,
        output_dir=output_dir,
        buffer_size=buffer_size,
        rows_per_shard=rows_per_shard,
        resume=resume,
        resume_from_shard=resume_from_shard,
        zstd_level=zstd_level,
        compress_workers=compress_workers,
        profile=profile_obj,
    )

    # Write manifest with shape metadata (after packing so max dims are known).
    # Note: max_F is NOT included — base shards store ALL extraction features
    # (~8k per segment). The training-time max_F is determined by selection
    # shards (post feature selection, ~40 per segment).
    #
    # Pass ``OverlayKind.PACKED_ROW`` (label-only enum value; the path was
    # already constructed manually above as pct/<pct>/tok/<tok>/c{T}/s{N}/).
    # Pass ``base_shard_paths`` so the resulting manifest is schema_version=2
    # with a base_shard_content_hash. See consolidation manifest §4.5 + §4.9.
    # O6: compute the base-shard content hash ONCE via the stat-validated #529 cache
    # (warm after step 4 / token boundaries ran) and thread the SAME precomputed hash into
    # BOTH the train and val manifests — instead of write_manifest re-reading + SHA-256ing
    # every base shard twice (~1 TB ×2). Byte-identical: the cache returns the exact streaming
    # hash, and the packed-row shards are unaffected.
    t_mani = time.monotonic()
    # base_shard_paths / base_shard_stems are resolved + invariant-checked once
    # near the top of run() (above _manifest_metadata); reuse them here.
    base_hash_result = _compute_base_hash(base_shard_paths, cache_dir=Path(feature_dir))
    write_manifest(
        output_dir,
        f"{pct_checksum}:{tok_checksum}",
        OverlayKind.PACKED_ROW,
        metadata=_manifest_metadata(g_max_n_seg, g_max_seg_size, output_dir),
        base_shard_paths=base_shard_paths,
        precomputed_base_hash=base_hash_result.hexdigest,
    )
    certify_fp_target_source(
        output_dir,
        n_sae_features=n_sae_features,
    )
    if profile_obj is not None:
        profile_obj.manifest_pre_pass = {
            "seconds": time.monotonic() - t_mani,
            "bytes_read": base_hash_result.bytes_read,  # 0 on a cache hit
            "cache_hit": base_hash_result.cache_hit,
            "n_shards_hashed": len(base_shard_paths),
        }

    # Create mirror symlinks: tok/<tok>/pct/<pct>/c.../s.../ → pct/<pct>/tok/<tok>/c.../s.../
    tok_dir = Path(feature_dir) / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
    pct_sub = tok_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
    mirror_dir = pct_sub / f"c{context_len}" / shards_str
    n_links = create_mirror_symlinks(output_dir, mirror_dir)
    if n_links > 0:
        logger.info(f"Created {n_links} mirror symlinks at {mirror_dir}")

    # --- Val split ---
    # Val packed rows go in a sibling directory: val_s{max_shards}
    # The training script discovers them via: parent / f"val_{train_dir.name}"
    # Naming matches training script discovery: train_dir.parent / f"val_{train_dir.name}"
    logger.info(f"Processing VAL split: {len(val_parquet_paths)} parquet shards → {val_output_dir}")
    val_output_dir.mkdir(parents=True, exist_ok=True)
    val_total_rows, val_max_n_seg, val_max_seg_size = precompute_packed_rows(
        parquet_paths=val_parquet_paths,
        tokenizer=tokenizer,
        feature_index=feature_index,
        context_len=context_len,
        output_dir=val_output_dir,
        buffer_size=buffer_size,
        rows_per_shard=rows_per_shard,
        resume=resume,
        zstd_level=zstd_level,
        compress_workers=compress_workers,
        profile=profile_obj,
    )

    # O6: reuse the SAME precomputed base-shard hash (no second re-hash of the base shards).
    write_manifest(
        val_output_dir,
        f"{pct_checksum}:{tok_checksum}",
        OverlayKind.PACKED_ROW,
        metadata=_manifest_metadata(val_max_n_seg, val_max_seg_size, val_output_dir),
        base_shard_paths=base_shard_paths,
        precomputed_base_hash=base_hash_result.hexdigest,
    )
    certify_fp_target_source(
        val_output_dir,
        n_sae_features=n_sae_features,
    )

    # Mirror symlinks for val
    val_mirror_dir = pct_sub / f"c{context_len}" / f"val_{output_dir.name}"
    n_val_links = create_mirror_symlinks(val_output_dir, val_mirror_dir)
    if n_val_links > 0:
        logger.info(f"Created {n_val_links} val mirror symlinks at {val_mirror_dir}")

    logger.info(
        f"Precomputation complete. Train: {total_rows} rows (max_n_seg={g_max_n_seg}), "
        f"Val: {val_total_rows} rows (max_n_seg={val_max_n_seg})"
    )

    if profile_obj is not None:
        feature_index.close()
        profile_obj.feature_index_loads = feature_index.load_profiles
        # ``or`` preserves the configured null sentinel without passing None to Path.
        sidecar_path = Path(profile_summary_output or "out/traces/packed_rows_profile.json")
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        profile_obj.write(sidecar_path)
        logger.info(f"Profile sidecar → {sidecar_path}")

    return 0


_REMOTE_STATE_SCHEMA_VERSION = 3
_PACKED_PAYLOAD_RE = re.compile(r"^(shard_(\d{5}))\.safetensors\.zst$")


def _remote_enabled(cfg: DictConfig) -> bool:
    """Read only the opt-in bit so disabled subordinate fields stay inert."""
    remote = cfg.get("remote", None)
    if remote is None:
        return False
    if not hasattr(remote, "get"):
        raise ValueError("remote must be a configuration mapping")
    enabled = remote.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"remote.enabled must be a bool, got {enabled!r}")
    return enabled


def _paths_overlap(left: Path, right: Path) -> bool:
    resolved_left = left.resolve()
    resolved_right = right.resolve()
    return (
        resolved_left == resolved_right
        or resolved_left.is_relative_to(resolved_right)
        or resolved_right.is_relative_to(resolved_left)
    )


def _read_required_manifest_path(
    path: Path,
    *,
    kind: OverlayKind,
    checksum: str,
) -> dict:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RemotePackedRowsError(
            f"required {kind.value} manifest is missing or unreadable: {path}"
        ) from exc
    if not isinstance(raw, dict):
        raise RemotePackedRowsError(f"required overlay manifest is not an object: {path}")
    if raw.get("schema_version") != 2:
        raise RemotePackedRowsError(
            f"remote packed rows require schema_version=2 in {path}, "
            f"got {raw.get('schema_version')!r}"
        )
    if raw.get("kind") != kind.value:
        raise RemotePackedRowsError(
            f"remote packed rows expected kind={kind.value!r} in {path}, got {raw.get('kind')!r}"
        )
    if raw.get("checksum") != checksum:
        raise RemotePackedRowsError(
            f"remote packed rows expected exact checksum {checksum!r} in {path}, "
            f"got {raw.get('checksum')!r}"
        )
    base_hash = raw.get("base_shard_content_hash")
    if not is_sha256_hex(base_hash):
        raise RemotePackedRowsError(
            f"remote packed rows require a lowercase SHA-256 base hash in {path}"
        )
    return raw


def _download_required_remote_manifest(
    transport: _RemotePackedRowsTransport,
    *,
    kind: OverlayKind,
    checksum: str,
) -> tuple[dict, RemoteObjectInfo]:
    manifest_path = f"{transport.config.remote_prefix}/manifest.json"
    matches = [info for info in transport.snapshot_listing if info.path == manifest_path]
    if len(matches) != 1:
        raise RemotePackedRowsError(
            f"remote {kind.value} snapshot must contain exactly one direct manifest: "
            f"path={manifest_path!r}, matches={len(matches)}"
        )
    staging_dir = transport.config.staging_dir
    _ensure_staging_directory(staging_dir)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".packed-rows-{kind.value}-manifest.",
        suffix=".json",
        dir=staging_dir,
    )
    os.close(descriptor)
    local_manifest = Path(temporary_name)
    local_manifest.unlink()
    try:
        download_verified_object(
            transport.backend,
            expected=matches[0],
            final_path=local_manifest,
            timeout_seconds=transport.config.retry_timeout_seconds,
            retry_interval_seconds=transport.config.retry_interval_seconds,
        )
        parsed = _read_required_manifest_path(local_manifest, kind=kind, checksum=checksum)
        return parsed, matches[0]
    finally:
        local_manifest.unlink(missing_ok=True)


def _selected_parquets(paths: list[str]) -> tuple[SelectedParquet, ...]:
    selected: list[SelectedParquet] = []
    for raw_path in paths:
        path = Path(raw_path).absolute()
        try:
            n_rows = int(pq.ParquetFile(path).metadata.num_rows)
        except Exception as exc:
            raise RemotePackedRowsError(
                f"cannot read selected parquet metadata from {path}: {exc}"
            ) from exc
        selected.append(
            SelectedParquet(
                stem=path.stem,
                path=path,
                n_rows=n_rows,
            )
        )
    return tuple(selected)


def _overlay_leaf_stems(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise RemotePackedRowsError(f"remote packed-row overlay directory is missing: {directory}")
    stems: list[str] = []
    for path in sorted(directory.glob("*.safetensors.zst")):
        match = _PACKED_PAYLOAD_RE.fullmatch(path.name)
        if match is None:
            raise RemotePackedRowsError(
                f"remote packed-row overlay has a noncanonical direct leaf: {path}"
            )
        # is_file() follows symlinks. Directories and dangling symlinks must not
        # satisfy local token-overlay family coverage.
        if not path.is_file():
            raise RemotePackedRowsError(
                f"remote packed-row overlay leaf is not a regular file: {path}"
            )
        stems.append(match.group(1))
    return stems


def _selected_parquet_inventory_json(
    selected: Sequence[SelectedParquet],
) -> list[dict[str, object]]:
    return [
        {
            "stem": entry.stem,
            "n_rows": entry.n_rows,
        }
        for entry in selected
    ]


def _remote_state_paths(output_dir: Path) -> tuple[Path, Path]:
    prefix = f".{output_dir.name}.remote-packing"
    return output_dir.parent / f"{prefix}-state.json", output_dir.parent / f"{prefix}.lock"


def _remote_checkpoint_dir(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.remote-packing-checkpoint"


def _remote_identity_digest(state: Mapping[str, object]) -> str:
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_ATOMIC_WRITE_DEBRIS_RE = re.compile(r"^.+\.tmp\.\d+\.[0-9a-f]+$")


def _is_direct_regular_atomic_temp(path: Path) -> bool:
    """Whether *path* is a direct regular atomic-write temporary."""
    if _ATOMIC_WRITE_DEBRIS_RE.fullmatch(path.name) is None:
        return False
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _non_debris_entries(directory: Path) -> Iterator[Path]:
    """Directory entries that are not orphaned atomic-write temp files.

    Torn `<name>.tmp.<pid>.<uuid>` files from a killed run must never count
    as payload (they would block fresh classification before the lock-held
    sweep can remove them).
    """
    for path in directory.iterdir():
        if not _is_direct_regular_atomic_temp(path):
            yield path


def _sweep_atomic_write_debris(
    *output_dirs: Path,
    state_path: Path | None = None,
) -> int:
    """Remove orphaned atomic-write temp files from the owned output namespaces.

    Safe only under the exclusive remote output lock: with a single writer,
    every ``<name>.tmp.<pid>.<uuid>`` file is a torn write from a killed run.
    """

    def _sweep_dir(directory: Path, *, name_prefix: str = "") -> int:
        swept = 0
        if not directory.is_dir():
            return swept
        for path in directory.iterdir():
            if not path.name.startswith(name_prefix):
                continue
            if _ATOMIC_WRITE_DEBRIS_RE.fullmatch(path.name) is None:
                continue
            if not stat.S_ISREG(path.lstat().st_mode):
                raise RemotePackedRowsError(
                    f"atomic-temp lookalike is not a direct regular file: {path}"
                )
            path.unlink()
            swept += 1
        return swept

    removed = 0
    for output_dir in output_dirs:
        for directory in (
            output_dir,
            provenance_dir(output_dir),
            output_dir / "_dims",
            _remote_checkpoint_dir(output_dir),
            output_dir / OFFLOAD_RECEIPT_DIRNAME,
        ):
            removed += _sweep_dir(directory)
    if state_path is not None:
        removed += _sweep_dir(state_path.parent, name_prefix=f"{state_path.name}.tmp.")
    if removed:
        logger.warning(f"Swept {removed} orphaned atomic-write temp file(s) from packed output")
    return removed


def _remove_checkpoint_namespace(output_dir: Path) -> None:
    checkpoint_dir = _remote_checkpoint_dir(output_dir)
    if not os.path.lexists(checkpoint_dir):
        return
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise RemotePackedRowsError(
            f"remote packing checkpoint namespace is not a direct directory: {checkpoint_dir}"
        )
    shutil.rmtree(checkpoint_dir)


@dataclass(frozen=True)
class _PackedArtifactIdentity:
    checksum: str
    base_hash: str
    base_shard_stems: tuple[str, ...]
    context_len: int
    max_shards: int | None
    buffer_size: int
    rows_per_shard: int
    n_sae_features: int


def _build_remote_state(
    *,
    remote_config: _PackedRowsRemoteTransportConfig,
    inventory_count: int,
    inventory_sha256: str,
    percentile_inventory_sha256: str,
    percentile_manifest_info: RemoteObjectInfo,
    artifact_identity: _PackedArtifactIdentity,
    segment_size: int,
    percentile_checksum: str,
    tokenizer_checksum: str,
    selected_parquets: Sequence[SelectedParquet],
    zstd_level: int,
) -> dict[str, object]:
    return {
        "schema_version": _REMOTE_STATE_SCHEMA_VERSION,
        "status": "in_progress",
        "remote": {
            "bucket_id": str(remote_config.bucket_id),
            "remote_prefix": str(remote_config.remote_prefix),
            "inventory_count": inventory_count,
            "inventory_sha256": inventory_sha256,
            "percentile_inventory_sha256": percentile_inventory_sha256,
            "percentile_manifest": {
                "path": percentile_manifest_info.path,
                "byte_length": percentile_manifest_info.byte_length,
                "xet_hash": percentile_manifest_info.xet_hash,
            },
        },
        "source": {
            "expected_base_shard_content_hash": artifact_identity.base_hash,
            "ordered_base_shard_stems": list(artifact_identity.base_shard_stems),
            "segment_size": segment_size,
            "percentile_checksum": percentile_checksum,
            "tokenizer_checksum": tokenizer_checksum,
            "selected_parquet_inventory": _selected_parquet_inventory_json(selected_parquets),
        },
        "packing": {
            "context_len": artifact_identity.context_len,
            "max_shards": artifact_identity.max_shards,
            "buffer_size": artifact_identity.buffer_size,
            "rows_per_shard": artifact_identity.rows_per_shard,
            "zstd_level": zstd_level,
            "n_sae_features": artifact_identity.n_sae_features,
        },
    }


def _read_remote_state(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError) as exc:
        raise RemotePackedRowsError(
            f"remote packed-row resume state is unreadable: {path}"
        ) from exc
    if not isinstance(raw, dict):
        raise RemotePackedRowsError(f"remote packed-row resume state must be a JSON object: {path}")
    return raw


def _output_has_payload_or_manifest(output_dir: Path) -> bool:
    dims_dir = output_dir / "_dims"
    provenance_output_dir = provenance_dir(output_dir)
    checkpoint_dir = _remote_checkpoint_dir(output_dir)
    # Offload receipts count as payload: they are the only local trace of a
    # shard whose bodies already live in the bucket, so a "fresh" run must not
    # be allowed to inherit (and then contradict) them.
    receipt_dir = output_dir / OFFLOAD_RECEIPT_DIRNAME
    for directory in (output_dir, provenance_output_dir, dims_dir, checkpoint_dir, receipt_dir):
        if os.path.lexists(directory) and (directory.is_symlink() or not directory.is_dir()):
            return True
    return bool(
        glob_overlay_shards(output_dir)
        or glob_overlay_shards(provenance_output_dir)
        or os.path.lexists(output_dir / "manifest.json")
        or (dims_dir.is_dir() and any(_non_debris_entries(dims_dir)))
        or (checkpoint_dir.is_dir() and any(_non_debris_entries(checkpoint_dir)))
        or (receipt_dir.is_dir() and any(_non_debris_entries(receipt_dir)))
    )


def _contiguous_complete_shard_count(
    output_dir: Path,
    *,
    offloaded: Collection[int] = (),
) -> int:
    # Remote-only caller: requires the _dims companion so checkpoint-seek never
    # keeps a torn shard it would not rewrite (see _packed_shard_complete).
    count = 0
    while _packed_shard_complete(output_dir, count, require_dims=True, offloaded=offloaded):
        count += 1
    return count


def _packed_output_inventory(
    output_dir: Path,
    *,
    offloaded: Collection[int] = (),
) -> int:
    published = frozenset(offloaded)
    packed_paths = glob_overlay_shards(output_dir)
    provenance_paths = glob_overlay_shards(provenance_dir(output_dir))

    def _indices(paths: list[Path], *, surface: str) -> list[int]:
        """The complete family this directory accounts for: local + offloaded."""
        indices: list[int] = []
        for path in paths:
            match = _PACKED_PAYLOAD_RE.fullmatch(path.name)
            if match is None:
                raise RemotePackedRowsError(
                    f"{surface} contains a noncanonical packed-row payload: {path}"
                )
            indices.append(int(match.group(2)))
        # Union, not sum: a kill between an offload receipt and its eviction
        # leaves a shard both receipted and local. That is a disk leak the next
        # run's reconcile pass repairs, not a family-membership disagreement.
        complete = sorted(set(indices) | published)
        if complete != list(range(len(complete))):
            raise RemotePackedRowsError(
                f"{surface} inventory is not the exact contiguous family 0..{len(complete) - 1}: "
                f"observed={indices}, offloaded={sorted(published)}"
            )
        return complete

    packed_indices = _indices(packed_paths, surface=str(output_dir))
    provenance_indices = _indices(
        provenance_paths,
        surface=str(provenance_dir(output_dir)),
    )
    if not packed_indices or packed_indices != provenance_indices:
        raise RemotePackedRowsError(
            "packed/provenance output inventories must be nonempty, paired, and "
            f"contiguous: packed={packed_indices}, provenance={provenance_indices}"
        )
    return len(packed_indices)


def _validate_packed_manifest(
    output_dir: Path,
    *,
    identity: _PackedArtifactIdentity,
    expected_max_n_seg: int | None = None,
    expected_max_seg_size: int | None = None,
    offloaded_receipts: Mapping[int, PackedShardAttestation] = MappingProxyType({}),
) -> dict:
    manifest = _read_required_manifest_path(
        output_dir / "manifest.json",
        kind=OverlayKind.PACKED_ROW,
        checksum=identity.checksum,
    )
    if manifest.get("base_shard_content_hash") != identity.base_hash:
        raise RemotePackedRowsError(
            f"packed-row manifest base hash mismatch in {output_dir / 'manifest.json'}"
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise RemotePackedRowsError(f"packed-row manifest metadata is missing in {output_dir}")
    expected = {
        "type": "packed_rows",
        "context_len": identity.context_len,
        "max_shards": identity.max_shards,
        "buffer_size": identity.buffer_size,
        "rows_per_shard": identity.rows_per_shard,
        "pct_checksum": identity.checksum.split(":", 1)[0],
        "tok_checksum": identity.checksum.split(":", 1)[1],
        "base_shard_stems": list(identity.base_shard_stems),
    }
    for field, value in expected.items():
        if type(metadata.get(field)) is not type(value) or metadata.get(field) != value:
            raise RemotePackedRowsError(
                f"packed-row manifest {output_dir} has incompatible {field}: "
                f"expected={value!r}, got={metadata.get(field)!r}"
            )
    n_payloads = _packed_output_inventory(output_dir, offloaded=offloaded_receipts)
    if metadata.get("provenance_shards") != n_payloads:
        raise RemotePackedRowsError(
            f"packed-row manifest {output_dir} provenance_shards mismatch: "
            f"metadata={metadata.get('provenance_shards')!r}, actual={n_payloads}"
        )
    for field, exact in (
        ("max_n_seg", expected_max_n_seg),
        ("max_seg_size", expected_max_seg_size),
    ):
        observed = metadata.get(field)
        if type(observed) is not int or observed < 0:
            raise RemotePackedRowsError(
                f"packed-row manifest {output_dir} has invalid {field}={observed!r}"
            )
        if exact is not None and observed != exact:
            raise RemotePackedRowsError(
                f"packed-row manifest {output_dir} has {field}={observed}, expected {exact}"
            )
    if offloaded_receipts:
        # The local validator reopens every body to re-derive the certificate's
        # inventory and fingerprint; an offloaded family answers from the
        # receipts that recorded them before eviction instead.
        validate_fp_target_source_certificate_remote(
            output_dir,
            n_sae_features=identity.n_sae_features,
            inventory=attested_inventory(list(offloaded_receipts.values())),
        )
    else:
        validate_fp_target_source_certificate(
            output_dir,
            n_sae_features=identity.n_sae_features,
        )
    return manifest


def _validate_mirror(source_dir: Path, mirror_dir: Path) -> None:
    source_paths = glob_overlay_shards(source_dir)
    mirror_paths = glob_overlay_shards(mirror_dir)
    if [path.name for path in mirror_paths] != [path.name for path in source_paths]:
        raise RemotePackedRowsError(
            f"packed-row mirror inventory mismatch: source={source_dir}, mirror={mirror_dir}"
        )
    for source, link in zip(source_paths, mirror_paths, strict=True):
        if not link.is_symlink() or link.resolve() != source.resolve():
            raise RemotePackedRowsError(
                f"packed-row mirror is not the expected file symlink: {link}"
            )


def _classify_remote_outputs(
    *,
    resume: bool,
    state_path: Path,
    expected_state: dict[str, object],
    output_dir: Path,
    val_output_dir: Path,
    identity: _PackedArtifactIdentity,
    identity_digest: str,
) -> str:
    has_state = os.path.lexists(state_path)
    has_payload = _output_has_payload_or_manifest(output_dir) or _output_has_payload_or_manifest(
        val_output_dir
    )
    if not resume:
        if has_state or has_payload:
            raise RemotePackedRowsError(
                "fresh remote packed-row output already has state or canonical "
                "payloads; use resume=true, or explicitly move/remove the "
                "interrupted output"
            )
        return "fresh"
    if has_state:
        observed_state = _read_remote_state(state_path)
        observed_version = observed_state.get("schema_version")
        if observed_version != _REMOTE_STATE_SCHEMA_VERSION:
            raise RemotePackedRowsError(
                "remote packed-row resume state schema version mismatch: "
                f"state has {observed_version!r}, current version is "
                f"{_REMOTE_STATE_SCHEMA_VERSION}; remove the stale state at {state_path} "
                f"and the partial output directories at {output_dir} and {val_output_dir} "
                "before starting fresh"
            )
        if observed_state != expected_state:
            raise RemotePackedRowsError(
                "remote packed-row resume state does not exactly match the current "
                f"source and packing identity: {state_path}"
            )
        return "state_resume"
    if not has_payload:
        raise RemotePackedRowsError(
            "remote resume=true without an in-progress state is allowed only for "
            "complete manifested outputs; no authenticated output was found"
        )
    for directory in (output_dir, val_output_dir):
        _validate_packed_manifest(
            directory,
            identity=identity,
            offloaded_receipts=read_offload_receipts(
                directory,
                identity_digest=identity_digest,
            ),
        )
    return "completed"


def _run_remote(cfg: DictConfig) -> int:
    feature_dir = resolve_feature_dir(cfg)
    if not feature_dir.is_dir():
        logger.error(f"Logical feature directory not found: {feature_dir}")
        return 1
    segment_size = int(cfg.segment_size)
    parsed_feature_dir = parse_feature_dir(feature_dir)
    if parsed_feature_dir.segment_size != segment_size:
        raise SegmentSizeMismatch(
            "remote packed-row segment_size mismatch before transport: "
            f"cfg={segment_size}, path={parsed_feature_dir.segment_size}"
        )
    configured_width = SAEWidth(str(cfg.sae_width))
    if parsed_feature_dir.width != configured_width:
        raise RemotePackedRowsError(
            "remote packed-row sae_width mismatch before transport: "
            f"cfg={configured_width.value}, path={parsed_feature_dir.width.value}"
        )

    context_len = int(cfg.context_len)
    n_sae_features = PositiveInt.parse(SAE_WIDTH_TO_NUM_FEATURES[parsed_feature_dir.width])
    max_shards_raw = cfg.max_shards
    max_shards = None if max_shards_raw is None else int(max_shards_raw)
    buffer_size = int(cfg.buffer_size)
    rows_per_shard = int(cfg.rows_per_shard)
    zstd_level = int(cfg.zstd_level)
    resume = bool(cfg.resume)
    if cfg.resume_from_shard is not None:
        raise RemotePackedRowsError(
            "remote packed-row mode does not support resume_from_shard; use "
            "resume=true so the contiguous authenticated prefix is derived safely"
        )

    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise RemotePackedRowsError(f"No feature statistics artifact found in {feature_dir}")
    pct_checksum = load_percentile_checksum(stats_path)
    if pct_checksum is None:
        raise RemotePackedRowsError(
            "No percentile checksum found — run precompute_percentiles first"
        )
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer()
    tok_checksum = compute_tokenizer_checksum(tokenizer)
    logger.info(f"Tokenizer checksum: {tok_checksum[:CHECKSUM_PREFIX_LEN]}...")
    logger.info(f"Percentile checksum: {pct_checksum[:CHECKSUM_PREFIX_LEN]}...")

    pct_dir = feature_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
    tok_sub = pct_dir / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
    shards_str = f"s{max_shards}" if max_shards is not None else "sall"
    output_dir = tok_sub / f"c{context_len}" / shards_str
    val_output_dir = output_dir.parent / f"val_{output_dir.name}"
    token_overlay_dir = (
        feature_dir / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN] / OverlayKind.SEG_BOUNDARY.value
    )
    logger.info(f"Output directory: {output_dir}")

    raw_data_dir = cfg.data_dir
    data_dir = Path(DATA_DIR if raw_data_dir is None else str(raw_data_dir)).absolute()
    parquet_paths = list_parquet_files(data_dir)
    if max_shards is not None:
        parquet_paths = parquet_paths[:max_shards]
    if len(parquet_paths) < 2:
        raise RemotePackedRowsError(
            f"Need at least 2 selected parquet shards for train/val, got {len(parquet_paths)}"
        )
    selected = _selected_parquets(parquet_paths)

    remote = cfg.remote
    if remote.staging_dir is None:
        raise ValueError("remote.staging_dir is required when remote.enabled=true")
    staging_dir = Path(str(remote.staging_dir)).absolute()
    offload_root: NormalizedRemotePrefix | None = None
    if bool(remote.offload.enabled):
        if remote.offload.remote_packed_root is None:
            raise ValueError(
                "remote.offload.remote_packed_root is required when remote.offload.enabled=true"
            )
        offload_root = NormalizedRemotePrefix.parse(str(remote.offload.remote_packed_root))
    profile = bool(cfg.profile)
    profile_summary_output = cfg.profile_summary_output
    profile_path = (
        Path(profile_summary_output or "out/traces/packed_rows_profile.json").absolute()
        if profile
        else None
    )
    protected_paths = [feature_dir, data_dir]
    if profile_path is not None:
        protected_paths.append(profile_path)
    if any(_paths_overlap(staging_dir, protected) for protected in protected_paths):
        raise ValueError(
            "remote.staging_dir must be disjoint in both directions from "
            "feature_dir, data_dir, and the profile output"
        )
    local_features_root = Path(str(remote.local_features_root))
    remote_prefix = resolve_remote_feature_prefix(
        feature_dir,
        local_features_root=local_features_root,
        remote_features_root=NormalizedRemotePrefix.parse(str(remote.remote_features_root)),
        surface="feature_dir",
    )
    offload_prefix_by_output_dir: dict[Path, NormalizedRemotePrefix] = {}
    if offload_root is not None:
        # The sealed input family is read-only for this run, and its listing is
        # the transport's family grammar: publishing output anywhere inside it
        # would both violate that rail and corrupt the snapshot a resume takes.
        offload_parts = PurePosixPath(str(offload_root))
        input_parts = PurePosixPath(str(remote_prefix))
        if offload_parts.is_relative_to(input_parts) or input_parts.is_relative_to(offload_parts):
            raise ValueError(
                "remote.offload.remote_packed_root must be disjoint from the sealed "
                f"input family: offload={offload_root}, input={remote_prefix}"
            )
        offload_prefix_by_output_dir = {
            directory.absolute(): resolve_remote_feature_prefix(
                directory,
                local_features_root=local_features_root,
                remote_features_root=offload_root,
                surface="packed-row output directory",
            )
            for directory in (output_dir, val_output_dir)
        }
        logger.info(
            "Offload enabled: packed bodies publish to "
            f"{sorted(str(prefix) for prefix in offload_prefix_by_output_dir.values())} "
            "and are evicted locally once verified"
        )
    expected_raw = remote.expected_shard_count
    if expected_raw is None:
        expected_count = None
    elif isinstance(expected_raw, bool) or not isinstance(expected_raw, int) or expected_raw <= 0:
        raise ValueError(
            f"remote.expected_shard_count must be null or a positive integer, got {expected_raw!r}"
        )
    else:
        expected_count = int(expected_raw)
    if expected_count is None:
        logger.warning(
            "REMOTE PACKED ROWS EXPECTED COUNT IS NULL: the observed complete "
            "family will be sealed, but production must pin "
            "remote.expected_shard_count"
        )
    remote_config = _PackedRowsRemoteTransportConfig(
        bucket_id=HfBucketId.parse(str(remote.bucket_id)),
        remote_prefix=remote_prefix,
        staging_dir=staging_dir,
        expected_shard_count=expected_count,
        retry_timeout_seconds=remote.retry_timeout_seconds,
        retry_interval_seconds=remote.retry_interval_seconds,
    )
    transport = _RemotePackedRowsTransport.snapshot(remote_config)
    base_shard_stems = [
        shard_stem_from_path(Path(entry.info.path)) for entry in transport.inventory
    ]
    selected_stems = [entry.stem for entry in selected]
    if selected_stems != base_shard_stems[: len(selected_stems)]:
        raise RemotePackedRowsError(
            "selected parquet stems must match the same-position prefix of the "
            f"remote family: selected={selected_stems}, "
            f"remote_prefix={base_shard_stems[: len(selected_stems)]}"
        )

    percentile_prefix = NormalizedRemotePrefix.parse(
        f"{remote_prefix}/pct/{pct_checksum[:CHECKSUM_PREFIX_LEN]}"
    )
    percentile_remote_config = _PackedRowsRemoteTransportConfig(
        bucket_id=remote_config.bucket_id,
        remote_prefix=percentile_prefix,
        staging_dir=staging_dir,
        expected_shard_count=len(transport.inventory),
        retry_timeout_seconds=remote_config.retry_timeout_seconds,
        retry_interval_seconds=remote_config.retry_interval_seconds,
    )
    percentile_transport = _RemotePackedRowsTransport.snapshot(
        percentile_remote_config,
        family=RemoteShardFamily.OVERLAY_LEAF,
    )
    required_staging_bytes = required_packed_rows_staging_capacity_bytes(
        transport.inventory,
        percentile_transport.inventory,
    )
    _ensure_staging_directory(staging_dir)
    # Admit BEFORE the manifest download and the state write: failing here on
    # an under-provisioned box leaves nothing behind, so the operator can
    # simply relaunch (a post-state failure would demand resume=true or a
    # manual state removal). The check re-runs adjacent to allocation with the
    # freshest observation inside the dual-transport scope.
    observed_free_bytes = shutil.disk_usage(staging_dir).free
    if observed_free_bytes < required_staging_bytes:
        raise RemotePackedRowsError(
            "remote packed-row staging admission failed before download: "
            f"required={required_staging_bytes:,}, free={observed_free_bytes:,}"
        )

    percentile_manifest, percentile_manifest_info = _download_required_remote_manifest(
        percentile_transport,
        kind=OverlayKind.PERCENTILE,
        checksum=pct_checksum,
    )
    token_manifest = _read_required_manifest_path(
        token_overlay_dir / "manifest.json",
        kind=OverlayKind.SEG_BOUNDARY,
        checksum=tok_checksum,
    )
    expected_base_hash = percentile_manifest["base_shard_content_hash"]
    if token_manifest["base_shard_content_hash"] != expected_base_hash:
        raise RemotePackedRowsError(
            "percentile and token manifests pin different base-shard content hashes"
        )

    # Percentile-family coverage (count + contiguous indices + canonical
    # names) is enforced by snapshot()'s family grammar with
    # expected_shard_count = len(raw inventory); no re-derivation here.
    token_leaf_stems = set(_overlay_leaf_stems(token_overlay_dir))
    remote_stem_set = set(base_shard_stems)
    token_extras = sorted(token_leaf_stems - remote_stem_set)
    if token_extras:
        raise RemotePackedRowsError(
            f"token overlay contains stems outside the remote family: {token_extras}"
        )

    compress_workers = int(cfg.compress_workers)
    if compress_workers == 0:
        compress_workers = max(1, min((os.cpu_count() or 4) // 4, 8))
    logger.info(f"Compress workers: {compress_workers}")
    profile_obj = (
        PackedRowsProfile(
            config={
                "context_len": context_len,
                "max_shards": max_shards,
                "buffer_size": buffer_size,
                "rows_per_shard": rows_per_shard,
                "zstd_level": zstd_level,
                "compress_workers": compress_workers,
                "remote": True,
                "remote_prefix": str(remote_prefix),
            }
        )
        if profile
        else None
    )
    checksum = f"{pct_checksum}:{tok_checksum}"
    artifact_identity = _PackedArtifactIdentity(
        checksum=checksum,
        base_hash=expected_base_hash,
        base_shard_stems=tuple(base_shard_stems),
        context_len=context_len,
        max_shards=max_shards,
        buffer_size=buffer_size,
        rows_per_shard=rows_per_shard,
        n_sae_features=n_sae_features,
    )
    expected_state = _build_remote_state(
        remote_config=remote_config,
        inventory_count=len(transport.inventory),
        inventory_sha256=transport.inventory_sha256,
        percentile_inventory_sha256=percentile_transport.inventory_sha256,
        percentile_manifest_info=percentile_manifest_info,
        artifact_identity=artifact_identity,
        segment_size=segment_size,
        percentile_checksum=pct_checksum,
        tokenizer_checksum=tok_checksum,
        selected_parquets=selected,
        zstd_level=zstd_level,
    )
    identity_digest = _remote_identity_digest(expected_state)
    # Deliberately NOT part of ``expected_state``: offload changes where the
    # produced bodies are stored, never their bytes or the source/packing
    # identity the state pins, so enabling it must not invalidate an existing
    # in-progress resume state.
    offloader = (
        PackedRowOffloader(
            backend=transport.backend,
            remote_prefix_by_output_dir=offload_prefix_by_output_dir,
            identity_digest=identity_digest,
            n_sae_features=n_sae_features,
            retry_timeout_seconds=remote_config.retry_timeout_seconds,
            retry_interval_seconds=remote_config.retry_interval_seconds,
            # Bounds local growth: the producer blocks once this many shards
            # await publication. Matches the checkpoint retention window so a
            # stalled uploader cannot outrun resumable state either.
            max_pending_shards=compress_workers + 2,
        )
        if offload_prefix_by_output_dir
        else None
    )
    position_by_stem = {stem: position for position, stem in enumerate(base_shard_stems)}
    train_positions = tuple(position_by_stem[entry.stem] for entry in selected[:-1])
    val_positions = (position_by_stem[selected[-1].stem],)
    train_checkpoint = _RemotePackingCheckpointContext(
        checkpoint_dir=_remote_checkpoint_dir(output_dir),
        identity_digest=identity_digest,
        split="train",
        parquet_inventory=tuple(
            (entry.path, position)
            for entry, position in zip(selected[:-1], train_positions, strict=True)
        ),
        end_position=val_positions[0],
        rows_per_shard=rows_per_shard,
    )
    val_checkpoint = _RemotePackingCheckpointContext(
        checkpoint_dir=_remote_checkpoint_dir(val_output_dir),
        identity_digest=identity_digest,
        split="val",
        parquet_inventory=((selected[-1].path, val_positions[0]),),
        end_position=len(transport.inventory),
        rows_per_shard=rows_per_shard,
    )
    state_path, lock_path = _remote_state_paths(output_dir)

    # Read-only classification before the lock gives early feedback. The same
    # classification is repeated under the producer lock before mutation.
    _classify_remote_outputs(
        resume=resume,
        state_path=state_path,
        expected_state=expected_state,
        output_dir=output_dir,
        val_output_dir=val_output_dir,
        identity=artifact_identity,
        identity_digest=identity_digest,
    )

    logger.info(
        "Remote packed-row mode streams raw and percentile HF bodies; any local "
        "raw or percentile bodies under the logical feature directory are ignored"
    )
    lookup: RemotePackedRowsFeatureLookup | None = None
    terminal_status = "failed"
    terminal_error: dict[str, str] | None = None
    classification = "unknown"
    total_rows = val_total_rows = 0
    g_max_n_seg = g_max_seg_size = 0
    val_max_n_seg = val_max_seg_size = 0
    kept_train_shards = kept_val_shards = 0
    try:
        with _remote_packed_rows_output_lock(lock_path):
            # A SIGKILL between an atomic write's temp-file creation and its
            # rename orphans `<name>.tmp.<pid>.<uuid>` debris. Under this
            # exclusive lock no other writer exists, so ANY such file in the
            # owned namespaces is orphaned; sweep before classification (debris
            # inside _dims/ would otherwise read as payload and block a fresh
            # run).
            _sweep_atomic_write_debris(
                output_dir,
                val_output_dir,
                state_path=state_path,
            )
            classification = _classify_remote_outputs(
                resume=resume,
                state_path=state_path,
                expected_state=expected_state,
                output_dir=output_dir,
                val_output_dir=val_output_dir,
                identity=artifact_identity,
                identity_digest=identity_digest,
            )
            if classification == "fresh":
                # Post-sweep, a fresh checkpoint namespace is at most an empty
                # directory (anything else classified as payload above); the
                # success path removes it, so no cleanup is needed here.
                atomic_json_dump(expected_state, state_path)
            # Receipts describe the OUTPUT, not this run's config: a family whose
            # bodies already live in the bucket cannot be resumed or revalidated
            # with offload off, so say that instead of failing later and deeper.
            train_receipts = read_offload_receipts(output_dir, identity_digest=identity_digest)
            val_receipts = read_offload_receipts(val_output_dir, identity_digest=identity_digest)
            if offloader is None and (train_receipts or val_receipts):
                raise RemotePackedRowsError(
                    f"packed-row output {output_dir} was produced with "
                    "remote.offload.enabled=true and its bodies live in the bucket; "
                    "re-enable offload to resume or revalidate it"
                )
            kept_train_shards = _contiguous_complete_shard_count(
                output_dir,
                offloaded=train_receipts,
            )
            kept_val_shards = _contiguous_complete_shard_count(
                val_output_dir,
                offloaded=val_receipts,
            )
            if offloader is not None:
                # Reclaim what an earlier window left on disk BEFORE packing
                # resumes: that headroom is what the rest of the run spends.
                offloader.reconcile(output_dir, upto=kept_train_shards)
                offloader.reconcile(val_output_dir, upto=kept_val_shards)
                train_receipts = offloader.receipts(output_dir)
                val_receipts = offloader.receipts(val_output_dir)

            with transport, percentile_transport:
                observed_free_bytes = shutil.disk_usage(staging_dir).free
                if observed_free_bytes < required_staging_bytes:
                    raise RemotePackedRowsError(
                        "remote packed-row joint staging admission failed adjacent to allocation: "
                        f"required={required_staging_bytes:,}, free={observed_free_bytes:,}"
                    )
                if classification == "completed":
                    train_payloads = _packed_output_inventory(output_dir, offloaded=train_receipts)
                    val_payloads = _packed_output_inventory(val_output_dir, offloaded=val_receipts)
                    g_max_n_seg, g_max_seg_size = _seed_max_dims_from_existing(
                        output_dir,
                        train_payloads,
                    )
                    val_max_n_seg, val_max_seg_size = _seed_max_dims_from_existing(
                        val_output_dir,
                        val_payloads,
                    )
                    _validate_packed_manifest(
                        output_dir,
                        identity=artifact_identity,
                        expected_max_n_seg=g_max_n_seg,
                        expected_max_seg_size=g_max_seg_size,
                        offloaded_receipts=train_receipts,
                    )
                    _validate_packed_manifest(
                        val_output_dir,
                        identity=artifact_identity,
                        expected_max_n_seg=val_max_n_seg,
                        expected_max_seg_size=val_max_seg_size,
                        offloaded_receipts=val_receipts,
                    )
                else:
                    packing_resume = classification != "fresh"
                    # A terminal train checkpoint restores inside the train
                    # packer itself (EOF cursor, empty buffer -> the packing
                    # loop exits on its first refill), exactly as the val split
                    # already relies on; its consumed_bodies equals the first
                    # val body, so the transport starts there and a val-side
                    # checkpoint seek downloads nothing extra.
                    train_restore = (
                        _load_nearest_eligible_packing_checkpoint(
                            train_checkpoint,
                            n_kept=kept_train_shards,
                            compress_workers=max(1, compress_workers),
                        )
                        if packing_resume and kept_train_shards > 0
                        else None
                    )
                    start_position = (
                        train_restore.consumed_bodies if train_restore is not None else 0
                    )
                    lookup = RemotePackedRowsFeatureLookup(
                        transport=transport,
                        percentile_transport=percentile_transport,
                        segment_size=segment_size,
                        percentile_checksum=pct_checksum,
                        token_overlay_dir=token_overlay_dir,
                        tokenizer_checksum=tok_checksum,
                        selected_parquets=selected,
                        profile_loads=profile,
                        start_position=start_position,
                    )
                    _assert_base_stems_aligned(base_shard_stems, lookup.shard_stems)
                    logger.info(
                        f"Processing TRAIN split: {len(selected) - 1} parquet shards → {output_dir}"
                    )
                    total_rows, g_max_n_seg, g_max_seg_size = precompute_packed_rows(
                        parquet_paths=[str(entry.path) for entry in selected[:-1]],
                        tokenizer=tokenizer,
                        feature_index=lookup,
                        context_len=context_len,
                        output_dir=output_dir,
                        buffer_size=buffer_size,
                        rows_per_shard=rows_per_shard,
                        resume=packing_resume,
                        zstd_level=zstd_level,
                        compress_workers=compress_workers,
                        profile=profile_obj,
                        remote_checkpoint=train_checkpoint,
                        offloader=offloader,
                    )
                    logger.info(f"Processing VAL split: 1 parquet shard → {val_output_dir}")
                    val_total_rows, val_max_n_seg, val_max_seg_size = precompute_packed_rows(
                        parquet_paths=[str(selected[-1].path)],
                        tokenizer=tokenizer,
                        feature_index=lookup,
                        context_len=context_len,
                        output_dir=val_output_dir,
                        buffer_size=buffer_size,
                        rows_per_shard=rows_per_shard,
                        resume=packing_resume,
                        zstd_level=zstd_level,
                        compress_workers=compress_workers,
                        profile=profile_obj,
                        remote_checkpoint=val_checkpoint,
                        offloader=offloader,
                    )
                    lookup.finish()
                    final_parquet_paths = list_parquet_files(data_dir)
                    if max_shards is not None:
                        final_parquet_paths = final_parquet_paths[:max_shards]
                    final_selected = _selected_parquets(final_parquet_paths)
                    if final_selected != selected:
                        raise RemotePackedRowsError(
                            "selected parquet inventory changed after remote packed-row preflight"
                        )
                # Packing published (and evicted) more shards; the manifest
                # accounting below counts the family, local bodies or not.
                train_receipts = read_offload_receipts(output_dir, identity_digest=identity_digest)
                val_receipts = read_offload_receipts(
                    val_output_dir,
                    identity_digest=identity_digest,
                )
                if (
                    discover_overlay(
                        feature_dir,
                        OverlayKind.SEG_BOUNDARY,
                        tok_checksum,
                        precomputed_base_hash=expected_base_hash,
                    )
                    != token_overlay_dir
                ):
                    raise RemotePackedRowsError(
                        f"token overlay could not be authenticated: {token_overlay_dir}"
                    )
                transport.final_inventory_fence()
                percentile_transport.final_inventory_fence()
                observed_percentile_manifest = percentile_transport.backend.stat(
                    percentile_manifest_info.path
                )
                if observed_percentile_manifest != percentile_manifest_info:
                    raise RemotePackedRowsError(
                        "remote percentile manifest identity changed before packed-row publication"
                    )

                def _manifest_metadata(
                    max_n_seg: int,
                    max_seg_size: int,
                    packed_dir: Path,
                    offloaded: Collection[int],
                ) -> dict[str, object]:
                    return {
                        "type": "packed_rows",
                        "context_len": context_len,
                        "max_shards": max_shards,
                        "buffer_size": buffer_size,
                        "rows_per_shard": rows_per_shard,
                        "pct_checksum": pct_checksum,
                        "tok_checksum": tok_checksum,
                        "max_n_seg": max_n_seg,
                        "max_seg_size": max_seg_size,
                        "base_shard_stems": base_shard_stems,
                        # The family size, counting shards whose bodies were
                        # published and evicted alongside those still local.
                        "provenance_shards": _packed_output_inventory(
                            packed_dir,
                            offloaded=offloaded,
                        ),
                    }

                tok_dir = feature_dir / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
                pct_sub = tok_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
                mirror_dir = pct_sub / f"c{context_len}" / shards_str
                val_mirror_dir = pct_sub / f"c{context_len}" / f"val_{output_dir.name}"
                mirror_pairs = (
                    (output_dir, mirror_dir),
                    (val_output_dir, val_mirror_dir),
                )

                def _certify(
                    packed_dir: Path,
                    receipts: Mapping[int, PackedShardAttestation],
                ) -> None:
                    """Certify FP-target integrity over the whole produced family.

                    The local scan opens every body, which an offloaded family
                    no longer has -- and an empty directory certifies vacuously.
                    Its receipts carry the same per-shard scan, taken while each
                    body was local, so the published certificate stays a true
                    statement about every shard.
                    """
                    if offloader is None:
                        certify_fp_target_source(packed_dir, n_sae_features=n_sae_features)
                        return
                    certify_fp_target_source_from_attestations(
                        packed_dir,
                        n_sae_features=n_sae_features,
                        attestations=list(receipts.values()),
                    )

                if classification != "completed":
                    _packed_output_inventory(output_dir, offloaded=train_receipts)
                    _packed_output_inventory(val_output_dir, offloaded=val_receipts)
                    # Validation is finalized first; the train manifest is the
                    # practical completion marker for the paired artifact.
                    write_manifest(
                        val_output_dir,
                        checksum,
                        OverlayKind.PACKED_ROW,
                        metadata=_manifest_metadata(
                            val_max_n_seg,
                            val_max_seg_size,
                            val_output_dir,
                            val_receipts,
                        ),
                        precomputed_base_hash=expected_base_hash,
                    )
                    _certify(val_output_dir, val_receipts)
                    write_manifest(
                        output_dir,
                        checksum,
                        OverlayKind.PACKED_ROW,
                        metadata=_manifest_metadata(
                            g_max_n_seg,
                            g_max_seg_size,
                            output_dir,
                            train_receipts,
                        ),
                        precomputed_base_hash=expected_base_hash,
                    )
                    _certify(output_dir, train_receipts)

                if offloader is None:
                    # Mirror publication is idempotent and can repair a complete
                    # canonical artifact whose mirrors were removed after commit.
                    for source, mirror in mirror_pairs:
                        create_mirror_symlinks(source, mirror)
                else:
                    # A mirror is a symlink to a local body, so an offloaded
                    # family could only produce dangling links.
                    logger.warning(
                        f"Offload: not publishing mirrors at {mirror_dir} and "
                        f"{val_mirror_dir}; consume the family from the bucket"
                    )

                for directory, max_n_seg, max_seg_size, receipts in (
                    (val_output_dir, val_max_n_seg, val_max_seg_size, val_receipts),
                    (output_dir, g_max_n_seg, g_max_seg_size, train_receipts),
                ):
                    _validate_packed_manifest(
                        directory,
                        identity=artifact_identity,
                        expected_max_n_seg=max_n_seg,
                        expected_max_seg_size=max_seg_size,
                        offloaded_receipts=receipts,
                    )
                if offloader is None:
                    for source, mirror in mirror_pairs:
                        _validate_mirror(source, mirror)
                if os.path.lexists(state_path):
                    if _read_remote_state(state_path) != expected_state:
                        raise RemotePackedRowsError(
                            "remote packed-row state changed before completion removal: "
                            f"{state_path}"
                        )
                    state_path.unlink()
                _remove_checkpoint_namespace(output_dir)
                _remove_checkpoint_namespace(val_output_dir)
                if profile_obj is not None:
                    profile_obj.manifest_pre_pass = {
                        "seconds": 0.0,
                        "bytes_read": 0,
                        "cache_hit": False,
                        "n_shards_hashed": 0,
                    }
            terminal_status = "success"
    except BaseException as exc:
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if offloader is not None:
            offloader.close()
        if profile_obj is not None and profile_path is not None:
            transport_payload = transport.profile_payload()
            percentile_payload = percentile_transport.profile_payload()
            percentile_payload["materialized_bytes"] = percentile_payload["downloaded_bytes"]
            transport_payload["percentile"] = percentile_payload
            transport_payload["joint_admission_required_bytes"] = required_staging_bytes
            transport_payload["joint_admission_observed_free_bytes"] = observed_free_bytes
            if offloader is not None:
                transport_payload["offload"] = offloader.profile_payload()
            if lookup is not None:
                transport_payload.update(lookup.profile_payload())
                profile_obj.feature_index_loads = lookup.feature_index_loads
            transport_payload["materialized_bytes"] = transport_payload["downloaded_bytes"]
            transport_payload.update(
                {
                    "status": terminal_status,
                    "resume_mode": classification,
                    "contiguous_kept_train_shards": kept_train_shards,
                    "contiguous_kept_validation_shards": kept_val_shards,
                    "terminal_error": terminal_error
                    if terminal_error is not None
                    else transport_payload.get("terminal_error"),
                }
            )
            profile_obj.remote_transport = transport_payload
            # A packing exception can occur after one or more worker saves but
            # before the inner profiler finalizer runs. Preserve a useful
            # failure sidecar instead of letting its consistency assertion mask
            # the primary error.
            profile_obj.totals["n_shards_written"] = len(profile_obj.per_output_shard)
            active_exception = sys.exception()
            try:
                profile_obj.write(profile_path)
                logger.info(f"Profile sidecar → {profile_path}")
            except BaseException as profile_error:
                if active_exception is not None:
                    active_exception.add_note(
                        f"remote packed-row profile write also failed: {profile_error!r}"
                    )
                else:
                    raise

    if classification == "completed":
        logger.info(
            "Remote packed-row completed artifact reauthenticated "
            "(train/val manifests + certificates revalidated)"
        )
    else:
        logger.info(
            f"Precomputation complete. Train: {total_rows} rows (max_n_seg={g_max_n_seg}), "
            f"Val: {val_total_rows} rows (max_n_seg={val_max_n_seg}), remote=true"
        )
    return 0


def run(cfg: DictConfig) -> int:
    """Dispatch to the literal local runner or bounded remote raw source."""
    return _run_remote(cfg) if _remote_enabled(cfg) else _run_local(cfg)


@hydra.main(
    config_path="../../configs/extraction",
    config_name="precompute_packed_rows",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    def _handle_sigterm(_signum: int, _frame: FrameType | None) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
