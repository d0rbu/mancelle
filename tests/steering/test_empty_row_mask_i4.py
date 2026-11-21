"""I4-invariant guard: empty-row NaN guard must not condition on a masked feature.

Tests that the fully-empty-row guard (which prevents NaN in cross-attention) does
not inject a masked feature id into enc_fids when a feature_mask is configured.

Bug: the guard unconditionally set enc_mask[row,0,0]=True over a zeros-initialised
enc_fids buffer. With a whitelist that excludes feature 0 (or a blacklist containing
0), this made the encoder condition on a masked feature after all assert_no_masked
checks had already passed — a silent I4 violation.

Sites covered:
  - Precomputed path: ``precomputed_data_loader`` (driven by 100% label_dropout).
  - Live path: ``encoder_decoder_data_loader_with_state`` (via the inner generator),
    exercising the twin ``_safe_empty_row_fallback`` call at ``dataloader.py``
    line ~1449 directly.
  - B=1 single-row case: precomputed loader with B=1 (spec review empirical gap).

I5 (no-mask byte-identical):
  - When ``feature_mask is None``, the guard produces the same output as before the
    fix (enc_mask[row,0,0]=True) — no change in NaN-guard behaviour.
  - Precomputed path: enc_fids is zeros-initialised, so fallback slot carries fid 0.
  - Live path: enc_fids is filled with real feature data before label_dropout fires,
    so the fallback slot may carry any feature id — the invariant is only that each
    row has at least one True position (NaN-safety preserved).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    encoder_decoder_data_loader_with_state,
    precomputed_data_loader,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import FeaturePipelineConfig, ThresholdSelectionConfig
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small SAE with features 0..N_SAE-1.  Feature 0 is the one the old guard
# would inject — we build masks that make 0 a disallowed id.
N_SAE = 32
_CONTEXT_LEN = 16
_SEGS_PER_ROW = 2
_B = 4


# ---------------------------------------------------------------------------
# Shard builders
# ---------------------------------------------------------------------------


def _make_all_feature0_shard(n_rows: int = _B) -> PackedRowShard:
    """Shard where every feature id is 0 and every pct > 0.

    After selection all active features have fid=0.  A whitelist excluding 0
    (or blacklist containing 0) then filters them all out, leaving enc_mask
    all-False for every segment — the empty-row guard fires.
    """
    T = _CONTEXT_LEN
    total_segs = n_rows * _SEGS_PER_ROW
    feats_per_seg = 2
    total_feats = total_segs * feats_per_seg

    tokens = t.ones(n_rows, T + 1, dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * _SEGS_PER_ROW, _SEGS_PER_ROW, dtype=t.int32)

    seg_size = T // _SEGS_PER_ROW
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)

    for r in range(n_rows):
        for s in range(_SEGS_PER_ROW):
            idx = r * _SEGS_PER_ROW + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == _SEGS_PER_ROW - 1 else (s + 1) * seg_size

    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)

    # ALL feature ids are 0; pct > 0 so topk selection keeps them.
    feature_ids = t.zeros(total_feats, dtype=t.uint16)
    feature_pcts = t.full((total_feats,), 200, dtype=t.uint8)  # high pct → always selected

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

_NUM_BUCKETS = 4


def _make_pipeline(feature_mask: FeatureMask | None = None) -> FeaturePipelineConfig:
    """Threshold pipeline selecting top-3 features per segment (fused path)."""
    boundaries = (
        t.linspace(1 / _NUM_BUCKETS, 1.0 - 1 / _NUM_BUCKETS, _NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE, -1)
        .contiguous()
    )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, 3),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=None,
        effective_buckets=_NUM_BUCKETS,
        feature_mask=feature_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_batches(loader, n: int = 1) -> list:
    batches = []
    for _ in range(n):
        batches.append(next(loader))
    return batches


def _write_shard(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard = _make_all_feature0_shard(n_rows=_B)
    save_packed_row_shard(shard, output_dir / "shard_00000")


def _masked_feature_ids_in_active_positions(
    enc_fids: t.Tensor,
    enc_mask: t.Tensor,
    feature_mask: FeatureMask,
) -> set[int]:
    """Return feature ids in (enc_mask==True) positions that are disallowed."""
    allowed = feature_mask.allowed_universe_mask(N_SAE)  # bool[N_SAE]
    result: set[int] = set()
    for b in range(enc_fids.shape[0]):
        for s in range(enc_fids.shape[1]):
            active = enc_mask[b, s]  # (F,) bool
            if not active.any():
                continue
            active_fids = enc_fids[b, s, active]
            for fid in active_fids.tolist():
                fid_int = int(fid)
                if fid_int >= N_SAE or not allowed[fid_int].item():
                    result.add(fid_int)
    return result


# ===========================================================================
# Test class: Precomputed path — I4 violation under whitelist/blacklist
# ===========================================================================


class TestEmptyRowGuardI4Precomputed:
    """Empty-row guard on the precomputed path must not inject a masked feature.

    Uses 100% label_dropout to guarantee all rows are empty before the guard
    fires, driving the exact codepath that contained the bug.
    """

    def test_whitelist_excluding_0_no_masked_fid_in_active_positions(self, tmp_path):
        """WHITELIST excluding feature 0: empty-row guard must not activate fid 0.

        Pre-fix: guard sets enc_mask[row,0,0]=True over zeros enc_fids → fid 0
        appears in an active position despite not being whitelisted → I4 violated.
        Post-fix: empty rows are left all-False (NaN-safe via cross-attn nan_to_num).
        """
        _write_shard(tmp_path / "packed")
        # Whitelist: only feature ids 5..10 are allowed; 0 is excluded.
        whitelist = FeatureMask("whitelist", tuple(range(5, 11)))
        pipeline = _make_pipeline(feature_mask=whitelist)

        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=_B,
            feature_pipeline=pipeline,
            device="cpu",
            label_dropout=1.0,  # guaranteed all-empty rows → guard fires
        )
        batch = next(loader)
        enc_fids: t.Tensor = batch[2]   # (B, n_seg, F)
        enc_mask: t.Tensor = batch[4]   # (B, n_seg, F)

        bad_fids = _masked_feature_ids_in_active_positions(enc_fids, enc_mask, whitelist)
        assert not bad_fids, (
            f"I4 violated on precomputed path (whitelist): active positions contain "
            f"masked feature ids {bad_fids}. The empty-row guard injected a non-whitelisted fid."
        )

    def test_blacklist_containing_0_no_masked_fid_in_active_positions(self, tmp_path):
        """BLACKLIST containing feature 0: empty-row guard must not activate fid 0.

        Same mechanism: guard injects fid 0 which is explicitly blacklisted.
        """
        _write_shard(tmp_path / "packed")
        blacklist = FeatureMask("blacklist", (0,))  # 0 is disallowed
        pipeline = _make_pipeline(feature_mask=blacklist)

        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=_B,
            feature_pipeline=pipeline,
            device="cpu",
            label_dropout=1.0,
        )
        batch = next(loader)
        enc_fids: t.Tensor = batch[2]
        enc_mask: t.Tensor = batch[4]

        bad_fids = _masked_feature_ids_in_active_positions(enc_fids, enc_mask, blacklist)
        assert not bad_fids, (
            f"I4 violated on precomputed path (blacklist): active positions contain "
            f"masked feature ids {bad_fids}. The empty-row guard injected the blacklisted fid 0."
        )


# ===========================================================================
# Test class: I5 — no-mask path byte-identical
# ===========================================================================


class TestEmptyRowGuardI5NoMask:
    """I5: when feature_mask is None, empty-row guard is byte-identical to pre-fix.

    The guard must still flip enc_mask[row,0,0]=True (over zeros enc_fids, fid==0)
    when no mask is configured — this was the pre-existing no-mask behaviour and
    must not change.
    """

    def test_no_mask_fallback_slot_fid_is_0_and_mask_flipped(self, tmp_path):
        """No mask: empty row gets enc_mask[row,0,0]=True, enc_fids[row,0,0]==0."""
        _write_shard(tmp_path / "packed")
        pipeline = _make_pipeline(feature_mask=None)

        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=_B,
            feature_pipeline=pipeline,
            device="cpu",
            label_dropout=1.0,
        )
        batch = next(loader)
        enc_fids: t.Tensor = batch[2]   # (B, n_seg, F)
        enc_mask: t.Tensor = batch[4]   # (B, n_seg, F)

        # Every row must have at least one True (old NaN-guard invariant preserved).
        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), (
            "I5 broken: no-mask path — some rows have no active features after "
            "100% dropout. Old NaN guard behaviour must be preserved."
        )

        # For every row that was empty (all rows after 100% dropout), the fallback
        # slot must be (0, 0) — fid 0, seg 0, feat 0.
        for b in range(_B):
            assert enc_mask[b, 0, 0].item(), (
                f"I5 broken: row {b} — fallback slot enc_mask[{b},0,0] is False "
                f"(expected True in no-mask path)."
            )
            assert enc_fids[b, 0, 0].item() == 0, (
                f"I5 broken: row {b} — fallback slot enc_fids[{b},0,0]="
                f"{enc_fids[b,0,0].item()} (expected 0 in no-mask path)."
            )


# ===========================================================================
# Test class: NaN safety — fixed empty row produces finite cross-attn output
# ===========================================================================


class TestEmptyRowNaNSafety:
    """After the fix, a fully-empty row (all-False enc_mask) is NaN-safe.

    The cross-attention already uses nan_to_num after SDPA, so an all-False
    mask row produces finite zero output — no explicit fallback fid needed.
    """

    def test_all_false_enc_mask_produces_finite_encoder_output(self):
        """All-False enc_mask through the encoder + cross-attn produces no NaN.

        Validates the NaN-safety claim that justifies removing the fallback
        fid injection for masked paths (the fix relies on this property).
        """
        import os
        os.environ.setdefault("NANOCHAT_ATTN_IMPL", "sdpa")

        from steering.conditional.encoder_decoder.encoder import (
            FeatureEncoder,
            FeatureEncoderConfig,
        )

        cfg = FeatureEncoderConfig(
            n_sae_features=N_SAE,
            num_buckets=_NUM_BUCKETS,
            n_embd=32,
            n_layer=1,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
        encoder = FeatureEncoder(cfg)
        encoder.eval()

        B, n_seg, F = 2, 3, 4
        fids = t.zeros(B, n_seg, F, dtype=t.int64)
        mags = t.zeros(B, n_seg, F, dtype=t.int64)
        # All-False mask: the exact representation used after the fix for empty rows
        mask = t.zeros(B, n_seg, F, dtype=t.bool)

        with t.no_grad():
            enc_out = encoder(fids, mags, mask)

        assert t.isfinite(enc_out.values).all(), (
            "All-False enc_mask through FeatureEncoder produced NaN/Inf. "
            "The fix relies on nan_to_num making this safe — verify encoder.py."
        )
        assert not enc_out.layout.valid_mask.any()


# ===========================================================================
# Live-path harness
# ===========================================================================
#
# Mirrors the minimal infrastructure from tests/steering/test_live_dataloader.py
# (see _SimpleTokenizer / _make_synthetic_data / _make_dataloader there) so the
# live ``_encoder_decoder_data_loader_inner`` generator is exercised directly,
# driving the ``_safe_empty_row_fallback`` call at dataloader.py line ~1449.
#
# Why label_dropout=1.0 forces the guard:
#   After all feature selection the dropout loop zeros enc_mask completely.
#   _safe_empty_row_fallback then sees row_all_empty=True for every row:
#     • feature_mask is not None → returns without touching enc_mask (I4-safe).
#     • feature_mask is None     → flips enc_mask[row,0,0]=True (I5 legacy).
#
# Spy / failure-mode proof:
#   * I4 test: if the OLD pre-fix guard body (enc_mask[row,0,0]=True) were
#     reintroduced, the whitelist/blacklist assertions would catch fid 0 in an
#     active position and fail.
#   * I5 test: if the entire helper call were removed from line ~1449, all rows
#     would remain all-False after label_dropout=1.0 and the assertion
#     "every row has at least one True" would fail.
#   Together they pin BOTH branches of the helper at the live call site.

_LIVE_N_SAE = 200
_LIVE_NUM_BUCKETS = 4
_LIVE_SEGMENT_SIZE: PositiveInt = PositiveInt.parse(20)
_LIVE_N_DOCS = 8
_LIVE_N_SEGS_PER_DOC = 2
_LIVE_N_FEATS_PER_SEG = 6
_LIVE_B = 2
_LIVE_T = 32


class _SimpleLiveTokenizer:
    """Deterministic tokenizer for testing.

    Token 0 = BOS, token n>0 → "t{n:03d} " (5 chars each).
    Identical to _SimpleTokenizer in test_live_dataloader.py.
    """

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        parts = []
        for tid in token_ids:
            if tid == 0:
                parts.append("<bos>")
            else:
                parts.append(f"t{tid:03d} ")
        return "".join(parts)


def _compute_live_token_boundaries(tokenizer, text_tokens, seg_char_starts, seg_char_ends):
    """Compute per-segment token start/end from character offsets."""
    from steering.segmentation import assign_tokens_to_segments

    char_offsets = token_to_char_offsets(tokenizer, text_tokens)
    tok_starts_t = t.tensor(char_offsets[:-1], dtype=t.int64)
    tok_ends_t = t.tensor(char_offsets[1:], dtype=t.int64)
    assignments = assign_tokens_to_segments(
        tok_starts_t,
        tok_ends_t,
        t.tensor(seg_char_starts, dtype=t.int64),
        t.tensor(seg_char_ends, dtype=t.int64),
    )
    assignments = t.cummax(assignments, dim=0).values
    n_segs = len(seg_char_starts)
    token_starts: list[int] = []
    token_ends: list[int] = []
    for s in range(n_segs):
        mask = assignments == s
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            token_starts.append(int(indices[0].item()))
            token_ends.append(int(indices[-1].item()) + 1)
        else:
            prev_end = token_ends[-1] if token_ends else 0
            token_starts.append(prev_end)
            token_ends.append(prev_end)
    return token_starts, token_ends


def _make_live_synthetic_env(tmp_path: Path) -> dict:
    """Build parquet + feature shards + overlays for live dataloader tests.

    Adapted from _make_synthetic_data in test_live_dataloader.py.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    feature_dir = tmp_path / "features" / f"s{_LIVE_SEGMENT_SIZE}"
    feature_dir.mkdir(parents=True)

    tokenizer_checksum = "live_i4_cksum"
    percentile_checksum = "live_i4_pct_cksum"
    tokenizer = _SimpleLiveTokenizer()
    bos_token = tokenizer.get_bos_token_id()

    for pq_num in range(2):
        shard_stem = f"shard_{pq_num:05d}"
        texts = []
        for doc_idx in range(_LIVE_N_DOCS):
            n_words = 8 + (doc_idx % 8) + pq_num * 3
            words = [f"w{pq_num}d{doc_idx}x{w}" for w in range(n_words)]
            texts.append(" ".join(words))

        pq_path = data_dir / f"{shard_stem}.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path), row_group_size=_LIVE_N_DOCS)

        total_segs = _LIVE_N_DOCS * _LIVE_N_SEGS_PER_DOC
        total_feats = total_segs * _LIVE_N_FEATS_PER_SEG

        all_fids = t.zeros(total_feats, dtype=t.int32)
        all_mags = t.zeros(total_feats, dtype=t.float32)
        all_pcts_overlay = t.zeros(total_feats, dtype=t.uint8)
        offsets = t.zeros(total_segs + 1, dtype=t.int64)
        doc_ids = t.zeros(total_segs, dtype=t.int32)
        seg_indices = t.zeros(total_segs, dtype=t.int32)
        seg_positions = t.zeros(total_segs, dtype=t.int32)
        seg_char_starts_t = t.zeros(total_segs, dtype=t.int32)
        seg_char_ends_t = t.zeros(total_segs, dtype=t.int32)
        seg_token_starts_t = t.zeros(total_segs, dtype=t.int32)
        seg_token_ends_t = t.zeros(total_segs, dtype=t.int32)

        feat_offset = 0
        seg_global = 0

        for doc_idx in range(_LIVE_N_DOCS):
            text = texts[doc_idx]
            text_tokens = tokenizer.encode(text, prepend=bos_token)
            char_offsets = token_to_char_offsets(tokenizer, text_tokens)
            n_tokens = len(text_tokens)
            tokens_per_seg = max(1, n_tokens // _LIVE_N_SEGS_PER_DOC)
            doc_seg_char_starts: list[int] = []
            doc_seg_char_ends: list[int] = []

            for seg_idx in range(_LIVE_N_SEGS_PER_DOC):
                doc_ids[seg_global] = doc_idx
                seg_indices[seg_global] = seg_idx
                seg_positions[seg_global] = seg_idx * _LIVE_SEGMENT_SIZE
                offsets[seg_global] = feat_offset

                tok_start = seg_idx * tokens_per_seg
                tok_end = (
                    min((seg_idx + 1) * tokens_per_seg, n_tokens)
                    if seg_idx < _LIVE_N_SEGS_PER_DOC - 1
                    else n_tokens
                )
                char_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
                char_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]

                seg_char_starts_t[seg_global] = char_start
                seg_char_ends_t[seg_global] = char_end
                doc_seg_char_starts.append(int(char_start))
                doc_seg_char_ends.append(int(char_end))

                for f in range(_LIVE_N_FEATS_PER_SEG):
                    fid = (
                        pq_num * _LIVE_N_DOCS * 10
                        + doc_idx * 10
                        + seg_idx * _LIVE_N_FEATS_PER_SEG
                        + f
                    ) % _LIVE_N_SAE
                    all_fids[feat_offset + f] = fid
                    all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx
                    all_pcts_overlay[feat_offset + f] = min(255, max(1, int(255 * (f + 1) / (_LIVE_N_FEATS_PER_SEG + 1))))

                feat_offset += _LIVE_N_FEATS_PER_SEG
                seg_global += 1

            tok_starts, tok_ends = _compute_live_token_boundaries(
                tokenizer, text_tokens, doc_seg_char_starts, doc_seg_char_ends
            )
            for seg_idx in range(_LIVE_N_SEGS_PER_DOC):
                g = doc_idx * _LIVE_N_SEGS_PER_DOC + seg_idx
                seg_token_starts_t[g] = tok_starts[seg_idx]
                seg_token_ends_t[g] = tok_ends[seg_idx]

        offsets[total_segs] = feat_offset

        shard_data = {
            "offsets": offsets,
            "feature_ids": all_fids[:feat_offset],
            "feature_mags": all_mags[:feat_offset],
            "doc_ids": doc_ids,
            "segment_indices": seg_indices,
            "segment_positions": seg_positions,
            "doc_token_counts": t.full((_LIVE_N_DOCS,), 40, dtype=t.int32),
            "segment_char_starts": seg_char_starts_t,
            "segment_char_ends": seg_char_ends_t,
            "n_docs": _LIVE_N_DOCS,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": _LIVE_SEGMENT_SIZE,
        }
        save_features(shard_data, feature_dir / f"{shard_stem}_features")

        tok_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, tokenizer_checksum)
        write_manifest(tok_dir, tokenizer_checksum, OverlayKind.SEG_BOUNDARY)
        save_overlay_shard(
            {
                TensorKey.SEGMENT_TOKEN_STARTS: seg_token_starts_t,
                TensorKey.SEGMENT_TOKEN_ENDS: seg_token_ends_t,
            },
            overlay_shard_path(tok_dir, shard_stem),
        )

        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, percentile_checksum)
        write_manifest(pct_dir, percentile_checksum, OverlayKind.PERCENTILE)
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES: all_pcts_overlay[:feat_offset]},
            overlay_shard_path(pct_dir, shard_stem),
        )

    return {
        "data_dir": str(data_dir),
        "feature_dir": str(feature_dir),
        "tokenizer_checksum": tokenizer_checksum,
        "percentile_checksum": percentile_checksum,
        "tokenizer": tokenizer,
    }


def _make_live_pipeline(
    feature_mask: FeatureMask | None = None,
    n_sae: int = _LIVE_N_SAE,
    num_buckets: int = _LIVE_NUM_BUCKETS,
    threshold: int = 3,
) -> FeaturePipelineConfig:
    """Threshold pipeline for the live dataloader tests."""
    boundaries = (
        t.linspace(1 / num_buckets, 1.0 - 1 / num_buckets, num_buckets - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(n_sae, -1)
        .contiguous()
    )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, threshold),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=None,
        effective_buckets=num_buckets,
        feature_mask=feature_mask,
    )


def _make_live_dataloader(env: dict, *, feature_mask: FeatureMask | None = None, label_dropout: float = 0.0):
    """Build a live encoder_decoder_data_loader_with_state from a synthetic env."""
    from nanochat.dataset import DatasetSplit

    tokenizer = env["tokenizer"]
    feature_index = FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"],
        percentile_checksum=env.get("percentile_checksum"),
    )
    pipeline = _make_live_pipeline(feature_mask=feature_mask)

    return encoder_decoder_data_loader_with_state(
        tokenizer=tokenizer,
        B=_LIVE_B,
        T=_LIVE_T,
        split=DatasetSplit.TRAIN,
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=_LIVE_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=128,
        buffer_size=50,
        data_dir=env["data_dir"],
        prefetch_batches=0,
        label_dropout=label_dropout,
        max_shards=None,
        n_sae_features=None,
    )


def _live_masked_fids_in_active_positions(
    enc_fids: t.Tensor,
    enc_mask: t.Tensor,
    feature_mask: FeatureMask,
    n_sae: int = _LIVE_N_SAE,
) -> set[int]:
    """Return feature ids in enc_mask-True positions that are disallowed by feature_mask."""
    allowed = feature_mask.allowed_universe_mask(n_sae)
    result: set[int] = set()
    for b in range(enc_fids.shape[0]):
        for s in range(enc_fids.shape[1]):
            active = enc_mask[b, s]
            if not active.any():
                continue
            for fid in enc_fids[b, s, active].tolist():
                fid_int = int(fid)
                if fid_int >= n_sae or not allowed[fid_int].item():
                    result.add(fid_int)
    return result


# ===========================================================================
# Test class: Live path — I4 violation under whitelist/blacklist
# ===========================================================================


class TestEmptyRowGuardI4LivePath:
    """Empty-row guard on the LIVE path must not inject a masked feature.

    Uses ``encoder_decoder_data_loader_with_state`` (which delegates to
    ``_encoder_decoder_data_loader_inner``) with label_dropout=1.0 to force
    all rows empty, triggering ``_safe_empty_row_fallback`` at
    dataloader.py line ~1449.

    Spy proof (see module-level comment above the harness constants):
      - I4 tests fail if the old fid-0 injection is reintroduced.
      - I5 test fails if the helper call at line ~1449 is removed.
    """

    @pytest.fixture
    def live_env(self, tmp_path):
        return _make_live_synthetic_env(tmp_path)

    def test_whitelist_excluding_0_no_masked_fid_in_active_positions(self, live_env):
        """WHITELIST excluding fid 0: live guard must leave empty rows all-False (I4).

        label_dropout=1.0 guarantees every row is empty when the guard fires.
        With feature_mask set, the guard must NOT inject fid 0 (which is
        zeros-initialised in enc_fids) into active enc_mask positions.
        """
        whitelist = FeatureMask(FeatureMaskMode.whitelist, tuple(range(5, 11)))
        gen = _make_live_dataloader(live_env, feature_mask=whitelist, label_dropout=1.0)
        _, _, enc_fids, _, enc_mask, *_ = next(gen)

        bad_fids = _live_masked_fids_in_active_positions(enc_fids, enc_mask, whitelist)
        assert not bad_fids, (
            f"I4 violated on LIVE path (whitelist): active positions contain masked "
            f"feature ids {bad_fids}. The empty-row guard injected a non-whitelisted fid "
            f"at dataloader.py line ~1449."
        )

    def test_blacklist_containing_0_no_masked_fid_in_active_positions(self, live_env):
        """BLACKLIST containing fid 0: live guard must leave empty rows all-False (I4).

        Same as above but via blacklist — confirms both FeatureMask modes are covered.
        """
        blacklist = FeatureMask(FeatureMaskMode.blacklist, (0,))
        gen = _make_live_dataloader(live_env, feature_mask=blacklist, label_dropout=1.0)
        _, _, enc_fids, _, enc_mask, *_ = next(gen)

        bad_fids = _live_masked_fids_in_active_positions(enc_fids, enc_mask, blacklist)
        assert not bad_fids, (
            f"I4 violated on LIVE path (blacklist): active positions contain masked "
            f"feature ids {bad_fids}. The empty-row guard injected the blacklisted fid 0 "
            f"at dataloader.py line ~1449."
        )

    def test_i5_no_mask_live_path_nan_guard_active(self, live_env):
        """I5: no-mask live path — empty rows get enc_mask[row,0,0]=True (legacy).

        Unlike the precomputed path, enc_fids in the live path is filled with
        real feature data (not zeros-initialised), so the fallback slot may carry
        any feature id — not necessarily 0.  The invariant is just that every row
        has at least one True position (NaN guard preserved).

        Proof this exercises line ~1449: if _safe_empty_row_fallback were
        removed from that line, label_dropout=1.0 would leave all rows all-False
        and row_has_any.all() would be False → this assertion fails.
        """
        gen = _make_live_dataloader(live_env, feature_mask=None, label_dropout=1.0)
        _, _, enc_fids, _, enc_mask, *_ = next(gen)

        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), (
            "I5 broken on LIVE path: some rows all-False after 100% dropout. "
            "Old NaN-guard behaviour must be preserved when feature_mask is None "
            "(dataloader.py line ~1449 _safe_empty_row_fallback no-mask branch)."
        )

        # The fallback slot is (row, 0, 0).  enc_mask[row,0,0] must be True.
        # enc_fids[row,0,0] may be any feature id — the live buffer is filled with
        # real data, unlike the zeros-initialised precomputed buffer.
        for b in range(_LIVE_B):
            assert enc_mask[b, 0, 0].item(), (
                f"I5 broken on LIVE path: row {b} — fallback slot enc_mask[{b},0,0] "
                f"is False (expected True from NaN-guard flip in no-mask path)."
            )


# ===========================================================================
# Test class: Precomputed path — B=1 single-row case (spec review gap)
# ===========================================================================


class TestEmptyRowGuardB1Precomputed:
    """B=1 edge case: single-row batch through the precomputed loader.

    Spec review noted the B=1 single-empty-row case was only empirically
    verified, not covered in the test file.  These tests close that gap.
    """

    def test_b1_whitelist_no_masked_fid(self, tmp_path):
        """B=1 precomputed: whitelist guard leaves the single empty row all-False."""
        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True)
        shard = _make_all_feature0_shard(n_rows=1)
        save_packed_row_shard(shard, shard_dir / "shard_00000")

        whitelist = FeatureMask(FeatureMaskMode.whitelist, tuple(range(5, 11)))
        pipeline = _make_pipeline(feature_mask=whitelist)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=1,
            feature_pipeline=pipeline,
            device="cpu",
            label_dropout=1.0,
        )
        batch = next(loader)
        enc_fids: t.Tensor = batch[2]
        enc_mask: t.Tensor = batch[4]

        assert enc_fids.shape[0] == 1, f"Expected B=1, got {enc_fids.shape[0]}"
        bad_fids = _masked_feature_ids_in_active_positions(enc_fids, enc_mask, whitelist)
        assert not bad_fids, (
            f"I4 violated for B=1 precomputed path (whitelist): masked fids "
            f"{bad_fids} found in active positions."
        )

    def test_b1_no_mask_fallback_slot(self, tmp_path):
        """B=1 precomputed no-mask: empty row gets enc_mask[0,0,0]=True (I5)."""
        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True)
        shard = _make_all_feature0_shard(n_rows=1)
        save_packed_row_shard(shard, shard_dir / "shard_00000")

        pipeline = _make_pipeline(feature_mask=None)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=1,
            feature_pipeline=pipeline,
            device="cpu",
            label_dropout=1.0,
        )
        batch = next(loader)
        enc_fids: t.Tensor = batch[2]
        enc_mask: t.Tensor = batch[4]

        assert enc_mask[0, 0, 0].item(), (
            "I5 broken for B=1 precomputed: fallback slot enc_mask[0,0,0] is False."
        )
        assert enc_fids[0, 0, 0].item() == 0, (
            f"I5 broken for B=1 precomputed: fallback fid={enc_fids[0,0,0].item()} (expected 0)."
        )
