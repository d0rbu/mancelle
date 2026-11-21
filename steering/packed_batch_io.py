"""I/O and reconstruction for precomputed packed row and selection shards.

Packed row shards store pre-packed token sequences with their segment
boundaries and extraction features in a compact CSR format. Individual
rows can be grouped into batches of any size at load time.

Shard layout (for R rows, T+1 tokens per row)::

    Dense:
      tokens           [R, T+1]          uint16

    CSR offsets:
      row_seg_offsets   [R+1]            int32   (segments per row)
      seg_feat_offsets  [total_segs+1]   int32   (features per segment)

    Segment data:
      seg_token_start   [total_segs]     uint16  (start pos in row)
      seg_token_end     [total_segs]     uint16  (end pos in row)
      seg_doc_id        [total_segs]     uint8   (doc index within row)

    Feature data:
      feature_ids       [total_feats]    uint16  (SAE feature indices)
      feature_pcts      [total_feats]    uint8   (precomputed percentiles)
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch as t
import zstandard
from safetensors import SafetensorError, safe_open
from safetensors.torch import load as safetensors_load

from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
)
from steering.conditional.feature_clusters import FeatureClusterDescriptor
from steering.conditional.splice_selection_batching import (
    canonical_splice_selection_batching,
)
from steering.gradient_overlay import (
    canonical_gradient_selection_dict,
    parse_gradient_ranking_spec,
)
from steering.overlay_io import (
    COMPRESSED_EXT,
    load_overlay_shard,
    load_overlay_shard_bytes,
    save_overlay_shard,
)


class _BatchIndex(NamedTuple):
    """Precomputed index arrays for a batch of rows from a packed shard.

    Shared across ``reconstruct_batch`` and the fused
    ``reconstruct_and_select`` in the dataloader.
    """

    B: int
    T: int
    inputs: t.Tensor       # [B, T] int64
    targets: t.Tensor      # [B, T] int64
    batch_n_seg: int
    total_segs: int
    seg_counts: t.Tensor   # [B] int64 — segments per row
    b_flat: t.Tensor       # [total_segs] — batch index per segment
    local_flat: t.Tensor   # [total_segs] — local segment index per segment
    global_flat: t.Tensor  # [total_segs] — global segment index in shard
    seg_map: t.Tensor      # [B, T] int64
    seg_doc_id: t.Tensor   # [B, batch_n_seg] int64


def _build_batch_index(
    shard: PackedRowShard,
    row_indices: list[int] | range,
    pad_n_seg: int | None = None,
) -> _BatchIndex:
    """Build shared index arrays for a batch of rows.

    Extracts tokens, builds segment index arrays, seg_map, and seg_doc_id.
    This is the common first phase of all batch reconstruction functions.
    """
    B = len(row_indices)
    T = shard.context_len

    # --- Extract inputs and targets (CPU) ---
    batch_tokens = shard.tokens[list(row_indices)]  # [B, T+1] uint16
    inputs = batch_tokens[:, :T].to(t.int64)
    targets = batch_tokens[:, 1:].to(t.int64)

    # --- Build flat segment index arrays ---
    ri = t.tensor(list(row_indices), dtype=t.long)
    row_offsets_t = shard.row_seg_offsets.long()
    seg_starts = row_offsets_t[ri]       # [B]
    seg_ends = row_offsets_t[ri + 1]     # [B]
    seg_counts = seg_ends - seg_starts   # [B] segments per row

    batch_n_seg = max(int(seg_counts.max().item()), 1)
    total_segs = int(seg_counts.sum().item())

    b_flat = t.repeat_interleave(t.arange(B), seg_counts)            # [total_segs]
    local_flat = t.cat([t.arange(n) for n in seg_counts.tolist()])   # [total_segs]
    global_flat = t.cat([t.arange(s, e) for s, e in
                         zip(seg_starts.tolist(), seg_ends.tolist(), strict=True)])  # [total_segs]

    # Apply padding override for segment dimension
    if pad_n_seg is not None:
        assert pad_n_seg >= batch_n_seg, f"pad_n_seg={pad_n_seg} < batch_n_seg={batch_n_seg}"
        batch_n_seg = pad_n_seg

    # --- Build seg_map and seg_doc_id ---
    seg_map = t.zeros(B, T, dtype=t.int64)
    seg_doc_id = t.full((B, batch_n_seg), -1, dtype=t.int64)

    if total_segs > 0:
        seg_doc_id[b_flat, local_flat] = shard.seg_doc_id[global_flat].to(t.int64)

        tok_lo = shard.seg_token_start[global_flat].long()             # [total_segs]
        tok_hi = shard.seg_token_end[global_flat].long().clamp(max=T)  # [total_segs]

        # Validate: segments must tile [0, T) exactly within each row.
        # 1. First segment starts at 0
        # 2. Consecutive segments: tok_lo[i+1] == tok_hi[i] (no gaps, no overlap)
        # 3. Last segment ends at T
        if bool((seg_counts > 0).all()):
            row_ends = seg_counts.cumsum(0)
            row_first = row_ends - seg_counts
            row_last = row_ends - 1
        else:
            row_first = t.zeros(B, dtype=t.long)  # flat index of first seg per row
            row_last = t.zeros(B, dtype=t.long)   # flat index of last seg per row
            for b in range(B):
                row_mask = b_flat == b
                indices = row_mask.nonzero(as_tuple=True)[0]
                row_first[b] = indices[0]
                row_last[b] = indices[-1]
        bad_start = tok_lo[row_first] != 0
        if bad_start.any():
            b = int(bad_start.nonzero(as_tuple=True)[0][0].item())
            raise ValueError(
                f"Segment tiling error in batch row {b}: "
                f"first segment starts at {tok_lo[row_first[b]].item()}, expected 0. "
                f"Packed row shards may need regeneration."
            )

        bad_end = tok_hi[row_last] != T
        if bad_end.any():
            b = int(bad_end.nonzero(as_tuple=True)[0][0].item())
            raise ValueError(
                f"Segment tiling error in batch row {b}: "
                f"last segment ends at {tok_hi[row_last[b]].item()}, expected {T}. "
                f"Packed row shards may need regeneration."
            )

        if total_segs > 1:
            same_row = b_flat[1:] == b_flat[:-1]
            gap = (tok_lo[1:] != tok_hi[:-1]) & same_row
            if gap.any():
                flat_idx = int(gap.nonzero(as_tuple=True)[0][0].item())
                row_idx = int(b_flat[flat_idx].item())
                local_seg = int(local_flat[flat_idx].item())
                raise ValueError(
                    f"Segment tiling error in batch row {row_idx}: "
                    f"seg {local_seg} ends at {tok_hi[flat_idx].item()} but "
                    f"seg {local_seg+1} starts at {tok_lo[flat_idx+1].item()} (expected equal). "
                    f"Packed row shards may need regeneration."
                )
        seg_lens = (tok_hi - tok_lo).clamp(min=0)                     # [total_segs]
        total_tokens = int(seg_lens.sum().item())

        if total_tokens > 0:
            b_tok = t.repeat_interleave(b_flat, seg_lens)
            local_tok = t.repeat_interleave(local_flat, seg_lens)
            tok_lo_exp = t.repeat_interleave(tok_lo, seg_lens)
            cum_lens = t.cat([t.zeros(1, dtype=t.long), seg_lens.cumsum(0)[:-1]])
            pos_in_seg = t.arange(total_tokens) - t.repeat_interleave(cum_lens, seg_lens)
            seg_map[b_tok, tok_lo_exp + pos_in_seg] = local_tok

    return _BatchIndex(
        B=B, T=T,
        inputs=inputs, targets=targets,
        batch_n_seg=batch_n_seg, total_segs=total_segs,
        seg_counts=seg_counts,
        b_flat=b_flat, local_flat=local_flat, global_flat=global_flat,
        seg_map=seg_map, seg_doc_id=seg_doc_id,
    )


class ReconstructedBatch(NamedTuple):
    """Batch tensors from packed row reconstruction.

    Return type for ``reconstruct_batch``, matching the training batch
    yield order (minus state_dict).
    """

    inputs: t.Tensor      # (B, T) int64
    targets: t.Tensor     # (B, T) int64
    enc_fids: t.Tensor    # (B, n_seg, F) int64
    enc_mags: t.Tensor    # (B, n_seg, F) int64
    enc_mask: t.Tensor    # (B, n_seg, F) bool
    enc_pcts: t.Tensor    # (B, n_seg, F) float32
    seg_map: t.Tensor     # (B, T) int64
    seg_doc_id: t.Tensor  # (B, n_seg) int64


@dataclass
class PackedRowShard:
    """A shard of precomputed packed rows with CSR segment/feature data."""

    tokens: t.Tensor            # [R, T+1] uint16
    row_seg_offsets: t.Tensor   # [R+1] int32
    seg_token_start: t.Tensor   # [total_segs] uint16
    seg_token_end: t.Tensor     # [total_segs] uint16
    seg_doc_id: t.Tensor        # [total_segs] uint8
    seg_feat_offsets: t.Tensor  # [total_segs+1] int32
    feature_ids: t.Tensor       # [total_feats] uint16
    feature_pcts: t.Tensor      # [total_feats] uint8
    # Optional handle that keeps externally-backed tensor storage alive (e.g.
    # the shared-memory mapping of a node-shared decode). Never serialized.
    backing_segment: object | None = None

    @property
    def n_rows(self) -> int:
        return self.tokens.shape[0]

    @property
    def context_len(self) -> int:
        """T (not T+1 — the stored tokens have T+1 for input/target derivation)."""
        return self.tokens.shape[1] - 1

    @property
    def total_segments(self) -> int:
        return len(self.seg_token_start)

    @property
    def total_features(self) -> int:
        return len(self.feature_ids)


# Tensor keys for safetensors serialization
_TENSOR_KEYS = [
    "tokens", "row_seg_offsets", "seg_token_start", "seg_token_end",
    "seg_doc_id", "seg_feat_offsets", "feature_ids", "feature_pcts",
]


def save_packed_row_shard(
    shard: PackedRowShard, path: Path, *, zstd_level: int | None = None,
) -> Path:
    """Save a packed row shard to compressed safetensors.

    All tensors are stored at their native compact dtypes (uint16, uint8, int32).
    Uses zstd compression via overlay_io.
    """
    tensors = {key: getattr(shard, key) for key in _TENSOR_KEYS}
    return save_overlay_shard(tensors, path, zstd_level=zstd_level)


def _packed_row_shard_from_tensors(
    data: Mapping[str, t.Tensor],
) -> PackedRowShard:
    return PackedRowShard(**{key: data[key] for key in _TENSOR_KEYS})


def load_packed_row_shard_bytes(compressed: bytes) -> PackedRowShard:
    """Load a packed row shard from complete compressed bytes."""
    return _packed_row_shard_from_tensors(load_overlay_shard_bytes(compressed))


def load_packed_row_shard_safetensors_bytes(decoded: bytes) -> PackedRowShard:
    """Load a packed row shard from complete uncompressed safetensors bytes."""
    return _packed_row_shard_from_tensors(safetensors_load(decoded))


def load_packed_row_shard(path: Path) -> PackedRowShard:
    """Load a packed row shard from compressed safetensors."""
    return _packed_row_shard_from_tensors(load_overlay_shard(path))


@dataclass
class PackedRowShardHandle:
    """One eager or disk-backed packed shard and auditable byte accounting."""

    shard: PackedRowShard
    storage_mode: str
    compressed_bytes: int
    tensor_bytes: int
    spool_bytes: int
    backing_path: Path | None
    closed: bool = False


def _packed_tensor_bytes(shard: PackedRowShard) -> int:
    return sum(
        int(getattr(shard, key).numel() * getattr(shard, key).element_size())
        for key in _TENSOR_KEYS
    )


def _raise_reference_error_or_spool_error(
    path: Path,
    spool_error: BaseException,
) -> None:
    """Prefer the legacy eager exception when the input itself is defective."""
    load_packed_row_shard(path)
    raise spool_error


def _is_packed_input_failure(stage: str, error: BaseException) -> bool:
    """Whether retrying the legacy reader is safe and needed for error parity."""
    if stage in {"source_resolve", "source_open", "source_read", "source_stat"}:
        return isinstance(error, (OSError, zstandard.ZstdError))
    if stage in {"safe_open", "safetensors_decode", "construct"}:
        return isinstance(error, (SafetensorError, KeyError))
    return False


@contextmanager
def open_packed_row_shard(
    path: Path,
    *,
    spool_dir: str | Path | None = None,
) -> Iterator[PackedRowShardHandle]:
    """Open a packed shard eagerly or through a temporary disk-backed mmap.

    ``spool_dir=None`` is exactly the legacy eager reader. With a spool
    directory, compressed bytes are streamed into a plain safetensors temp
    file and tensors are exposed through ``safe_open`` without cloning. The
    file and mapping live exactly for the context and are removed on every
    exit. Callers must not retain tensor views beyond the context; the handle's
    root tensors are invalidated at exit to enforce that lifetime for ordinary
    use. If the compressed input is corrupt, the eager reader is invoked only
    on that failure path and its original exception is re-raised verbatim;
    spool-resource failures remain distinct and loud.
    """
    path = Path(path)
    if spool_dir is None:
        shard = load_packed_row_shard(path)
        handle = PackedRowShardHandle(
            shard=shard,
            storage_mode="eager",
            compressed_bytes=path.stat().st_size,
            tensor_bytes=_packed_tensor_bytes(shard),
            spool_bytes=0,
            backing_path=None,
        )
        try:
            yield handle
        finally:
            handle.closed = True
        return

    if not path.name.endswith(COMPRESSED_EXT):
        # Preserve the eager path's exact extension failure.
        load_packed_row_shard(path)
        raise AssertionError("unreachable: eager packed-row extension check returned")
    spool_root = Path(spool_dir)
    if not spool_root.is_dir():
        raise NotADirectoryError(f"packed-row spool directory is absent: {spool_root}")

    stack = ExitStack()
    temp_path: Path | None = None
    stage = "spool_setup"
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.packed-spool-",
            suffix=".safetensors",
            dir=spool_root,
            delete=False,
        ) as output:
            temp_path = Path(output.name)
            stage = "source_resolve"
            resolved = path.resolve()
            stage = "source_open"
            with resolved.open("rb") as source:
                with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                    while True:
                        stage = "source_read"
                        chunk = reader.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        stage = "spool_write"
                        output.write(chunk)
            stage = "spool_flush"
            output.flush()
        stage = "safe_open"
        reader = stack.enter_context(
            safe_open(str(temp_path), framework="pt", device="cpu")
        )
        stage = "safetensors_decode"
        data = {str(key): reader.get_tensor(str(key)) for key in reader.keys()}
        stage = "construct"
        shard = PackedRowShard(**{key: data[key] for key in _TENSOR_KEYS})
        stage = "source_stat"
        compressed_bytes = path.stat().st_size
        stage = "spool_stat"
        spool_bytes = temp_path.stat().st_size
        handle = PackedRowShardHandle(
            shard=shard,
            storage_mode="disk_spool",
            compressed_bytes=compressed_bytes,
            tensor_bytes=_packed_tensor_bytes(shard),
            spool_bytes=spool_bytes,
            backing_path=temp_path,
        )
    except BaseException as spool_error:
        stack.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if _is_packed_input_failure(stage, spool_error):
            _raise_reference_error_or_spool_error(path, spool_error)
            raise AssertionError(
                "unreachable: packed-row error normalization returned"
            ) from spool_error
        raise

    try:
        yield handle
    finally:
        # Drop every caller-visible root tensor before unlinking. This preselection
        # consumer retains no packed views after the context; invalidating the handle
        # makes that lifetime explicit and prevents deleted-but-mapped spool buildup.
        for key in _TENSOR_KEYS:
            tensor = getattr(handle.shard, key)
            setattr(handle.shard, key, t.empty(0, dtype=tensor.dtype))
        del tensor
        data.clear()
        handle.closed = True
        stack.close()
        assert temp_path is not None
        temp_path.unlink(missing_ok=True)


def reconstruct_batch(
    shard: PackedRowShard,
    row_indices: list[int] | range,
    sel_shard: SelectionShard,
    effective_buckets: int,
    device: str = "cpu",
    pad_n_seg: int | None = None,
    pad_F: int | None = None,
) -> ReconstructedBatch:
    """Reconstruct a padded batch from a packed row shard and its selection shard.

    The selection shard must be aligned 1:1 with *shard* (same segment count,
    produced by the precompute pipeline for the same base shard).

    Returns a ReconstructedBatch with selected positive features (with
    discretized bucket values) and negative features (with *effective_buckets*
    as magnitude) appended after positives.
    """
    idx = _build_batch_index(shard, row_indices, pad_n_seg=pad_n_seg)
    B = idx.B
    batch_n_seg = idx.batch_n_seg

    # --- Dimension discovery from selection shard ---
    sel_offsets_t = sel_shard.sel_offsets.long()
    neg_offsets_t = sel_shard.neg_offsets.long()
    sel_counts = sel_offsets_t[idx.global_flat + 1] - sel_offsets_t[idx.global_flat]
    neg_counts = neg_offsets_t[idx.global_flat + 1] - neg_offsets_t[idx.global_flat]
    combined_counts = sel_counts + neg_counts
    batch_f_max = max(int(combined_counts.max().item()) if idx.total_segs > 0 else 0, 1)

    if pad_F is not None:
        assert pad_F >= batch_f_max, f"pad_F={pad_F} < batch_f_max={batch_f_max}"
        batch_f_max = pad_F

    # --- Allocate feature tensors on CPU ---
    enc_fids = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.int64)
    enc_mags = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.int64)
    enc_mask = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.bool)
    enc_pcts = t.zeros(B, batch_n_seg, batch_f_max, dtype=t.float32)

    # --- Vectorized feature fill from selection shard ---
    if idx.total_segs > 0:
        # Positive features
        total_sel = int(sel_counts.sum().item())
        if total_sel > 0:
            sf_starts = sel_offsets_t[idx.global_flat]
            cum_sc = t.cat([t.zeros(1, dtype=t.long), sel_counts.cumsum(0)[:-1]])
            sel_offsets_in = t.arange(total_sel)
            positive_rows = t.repeat_interleave(
                t.stack((cum_sc, sf_starts, idx.b_flat, idx.local_flat)),
                sel_counts,
                dim=1,
            )
            sel_offsets_in = sel_offsets_in - positive_rows[0]
            sel_gather = positive_rows[1] + sel_offsets_in

            b_sel = positive_rows[2]
            local_sel = positive_rows[3]

            enc_fids[b_sel, local_sel, sel_offsets_in] = sel_shard.sel_fids[sel_gather].to(t.int64)
            enc_mags[b_sel, local_sel, sel_offsets_in] = sel_shard.sel_buckets[sel_gather].to(t.int64)
            enc_pcts[b_sel, local_sel, sel_offsets_in] = sel_shard.sel_pcts[sel_gather].float() * (1.0 / 255.0)
            enc_mask[b_sel, local_sel, sel_offsets_in] = True

        # Negative features: appended after positives
        total_neg = int(neg_counts.sum().item())
        if total_neg > 0:
            nf_starts = neg_offsets_t[idx.global_flat]
            cum_nc = t.cat([t.zeros(1, dtype=t.long), neg_counts.cumsum(0)[:-1]])
            neg_offsets_in = t.arange(total_neg) - t.repeat_interleave(cum_nc, neg_counts)
            neg_gather = t.repeat_interleave(nf_starts, neg_counts) + neg_offsets_in

            b_neg = t.repeat_interleave(idx.b_flat, neg_counts)
            local_neg = t.repeat_interleave(idx.local_flat, neg_counts)
            neg_feat_offset = t.repeat_interleave(sel_counts, neg_counts)

            enc_fids[b_neg, local_neg, neg_feat_offset + neg_offsets_in] = sel_shard.neg_fids[neg_gather].to(t.int64)
            enc_mags[b_neg, local_neg, neg_feat_offset + neg_offsets_in] = effective_buckets
            enc_mask[b_neg, local_neg, neg_feat_offset + neg_offsets_in] = True

    # --- Bulk transfer to target device ---
    inputs, targets = idx.inputs, idx.targets
    seg_map, seg_doc_id = idx.seg_map, idx.seg_doc_id
    if device != "cpu":
        inputs = inputs.to(device)
        targets = targets.to(device)
        seg_map = seg_map.to(device)
        seg_doc_id = seg_doc_id.to(device)
        enc_fids = enc_fids.to(device)
        enc_mags = enc_mags.to(device)
        enc_pcts = enc_pcts.to(device)
        enc_mask = enc_mask.to(device)

    return ReconstructedBatch(
        inputs=inputs,
        targets=targets,
        enc_fids=enc_fids,
        enc_mags=enc_mags,
        enc_mask=enc_mask,
        enc_pcts=enc_pcts,
        seg_map=seg_map,
        seg_doc_id=seg_doc_id,
    )


# ---------------------------------------------------------------------------
# Selection shards (pre-selected features for a specific pipeline config)
# ---------------------------------------------------------------------------


@dataclass
class SelectionShard:
    """Pre-selected features for a specific pipeline config, aligned 1:1 with a PackedRowShard.

    Positive features: selected via thresholding/sampling + discretized.
    Negative features: just IDs (bucket = effective_buckets, pct = 0).
    """

    sel_offsets: t.Tensor     # [total_segs + 1] int32 — CSR for positive features
    sel_fids: t.Tensor        # [total_sel] uint16 — selected feature IDs
    sel_buckets: t.Tensor     # [total_sel] uint8 — discretized magnitude buckets
    sel_pcts: t.Tensor        # [total_sel] uint8 — percentile values (0-255)
    neg_offsets: t.Tensor     # [total_segs + 1] int32 — CSR for negative features
    neg_fids: t.Tensor        # [total_neg] uint16 — negative feature IDs

    @property
    def total_segments(self) -> int:
        return len(self.sel_offsets) - 1

    @property
    def total_selected(self) -> int:
        return len(self.sel_fids)

    @property
    def total_negative(self) -> int:
        return len(self.neg_fids)


_SEL_TENSOR_KEYS = [
    "sel_offsets", "sel_fids", "sel_buckets", "sel_pcts",
    "neg_offsets", "neg_fids",
]


def save_selection_shard(
    shard: SelectionShard, path: Path, *, zstd_level: int | None = None,
) -> Path:
    """Save a selection shard to compressed safetensors."""
    tensors = {key: getattr(shard, key) for key in _SEL_TENSOR_KEYS}
    return save_overlay_shard(tensors, path, zstd_level=zstd_level)


def _selection_shard_from_tensors(
    data: Mapping[str, t.Tensor],
) -> SelectionShard:
    return SelectionShard(**{key: data[key] for key in _SEL_TENSOR_KEYS})


def load_selection_shard_bytes(compressed: bytes) -> SelectionShard:
    """Load a selection shard from complete compressed bytes."""
    return _selection_shard_from_tensors(load_overlay_shard_bytes(compressed))


def load_selection_shard(path: Path) -> SelectionShard:
    """Load a selection shard from compressed safetensors."""
    return _selection_shard_from_tensors(load_overlay_shard(path))


# ---------------------------------------------------------------------------
# Selection config hashing
# ---------------------------------------------------------------------------


def compute_selection_hash(
    positive_selection: dict,
    negative_selection: dict | None,
    delta_config: dict | None,
    num_buckets: int,
    seed: int,
    ddp_rank: int = 0,
    feature_mask: dict | None = None,
    feature_clusters: dict | None = None,
    splice_batch_size: object = None,
) -> str:
    """Compute a deterministic SHA-256 hash of the full selection config.

    All inputs are plain dicts/values (not dataclass instances) so this
    can be called from configs without importing the dataclass types.

    ``feature_mask`` is inserted into the payload ONLY when not None, so a
    run without a mask hashes byte-identically to the legacy 6-key payload
    (every existing sel/<hash>/ shard stays valid).

    The optional ``positive_selection["gradient"]`` block (gradient-
    attribution ranking) is canonicalized HERE, centrally, so all four
    ANTI-DESYNC call sites inherit identical semantics with zero per-site
    edits: an absent or null key is REMOVED (the payload stays byte-identical
    to the legacy form — I1); a present block is strictly parsed and replaced
    by its canonical dict, which includes the code-versioned score-reduction
    and split-policy constants so a sel/<hash>/ dir can never be silently
    reused across algorithm revisions.

    Returns:
        64-character hex string.
    """
    positive_selection = dict(positive_selection)
    gradient_raw = positive_selection.pop("gradient", None)
    if gradient_raw is not None:
        positive_selection["gradient"] = canonical_gradient_selection_dict(
            parse_gradient_ranking_spec(gradient_raw)
        )
    payload = {
        "positive_selection": positive_selection,
        "negative_selection": negative_selection,
        "delta_config": delta_config,
        "num_buckets": num_buckets,
        "seed": seed,
        "ddp_rank": ddp_rank,
    }
    if feature_mask is not None:
        # Canonicalize defensively (sorted unique ints, strict {mode,ids}
        # shape mirroring feature_mask.canonical_dict) so the hash is
        # invariant to caller-side ordering/dups. Guarded by `is not None`
        # so a no-mask run's payload/hash stays byte-identical (I1).
        payload["feature_mask"] = {
            "mode": feature_mask["mode"],
            "ids": sorted({int(i) for i in feature_mask["ids"]}),
        }
    if feature_clusters is not None:
        # Static cluster content is already canonicalized by the strict file
        # parser. Re-parse the small descriptor here so all identity sites fail
        # identically on malformed/ad-hoc payloads. The key is conditional:
        # absent clustering preserves every legacy selection hash byte-for-byte.
        payload["feature_clusters"] = FeatureClusterDescriptor.from_dict(
            feature_clusters
        ).canonical_dict()
        if delta_config is not None and delta_config.get("mode") == "splice_counterfactual":
            # This key versions only the non-commuting clustered splice path.
            # Unclustered splice and clustered delta-free payloads retain their
            # exact pre-correction JSON shape and SHA-256 identity.
            payload[CLUSTER_DELTA_SEMANTICS_KEY] = CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
    splice_batching = canonical_splice_selection_batching(
        delta_config,
        splice_batch_size,
    )
    if splice_batching is not None:
        # Conditional for the same compatibility reason as feature_mask and
        # feature_clusters: legacy/disabled batching retains the exact old
        # JSON payload and therefore every existing selection hash.
        payload["splice_selection_batching"] = splice_batching
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
