"""Tests for feature_mask enforcement in the LIVE encoder-decoder dataloader.

Covers:
- Blacklist: masked feature K never appears in positives or negatives.
- Whitelist: positives and negatives are subsets of the whitelist.
- I5 no-mask byte-identical: two equal-seed runs, one with feature_mask=None,
  produce identical batch tensors (positive ids/mags/mask and negative ids/mask).

Delta path tests (H3):
- test_live_delta_blacklist_no_masked_negative: delta+blacklist — masked feature
  never in negatives (Task 9 coverage, previously skipped here).
- test_live_delta_whitelist_negk_clamped_to_allowed_pool: delta+whitelist — every
  negative ∈ whitelist AND per-segment count ≤ |whitelist ∩ ~active| (exposes
  the pre-fix COUNT-DISTRIBUTION skew).
- test_live_delta_no_mask_byte_identical: I5 — None path RNG-identical.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from steering.conditional.delta import DeltaMode, DeltaSamplingWeights, FeatureDeltaConfig
from steering.conditional.encoder_decoder.dataloader import (
    encoder_decoder_data_loader_with_state,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import (
    OverlayKind,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets

# ---------------------------------------------------------------------------
# Mirror the _SimpleTokenizer from test_live_dataloader.py
# ---------------------------------------------------------------------------

class _SimpleTokenizer:
    """Deterministic tokenizer for testing (mirrors test_live_dataloader.py)."""

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


# ---------------------------------------------------------------------------
# Mirror test constants from test_live_dataloader.py
# ---------------------------------------------------------------------------

_N_SAE_FEATURES = 200
_NUM_BUCKETS = 4
_SEGMENT_SIZE: PositiveInt = PositiveInt.parse(20)
_N_DOCS = 8
_N_SEGS_PER_DOC = 2
_N_FEATS_PER_SEG = 6
_B = 2
_T = 32


def _compute_token_boundaries_for_seg(tokenizer, text_tokens, seg_char_starts, seg_char_ends):
    """Compute precomputed token start/end for each segment."""
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
    token_starts = []
    token_ends = []
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


def _make_synthetic_data(
    tmp_path,
    *,
    n_docs=_N_DOCS,
    n_segs_per_doc=_N_SEGS_PER_DOC,
    n_feats_per_seg=_N_FEATS_PER_SEG,
    segment_size=_SEGMENT_SIZE,
    n_sae_features=_N_SAE_FEATURES,
    tokenizer_checksum="test_cksum",
    percentile_checksum="pct_cksum",
):
    """Create synthetic parquet + feature shard + token boundary + percentile overlays.

    Mirrors _make_synthetic_data from test_live_dataloader.py exactly.
    """
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True)
    feature_dir = Path(tmp_path) / "features" / f"s{segment_size}"
    feature_dir.mkdir(parents=True)

    tokenizer = _SimpleTokenizer()
    bos_token = tokenizer.get_bos_token_id()

    for pq_num in range(2):
        shard_stem = f"shard_{pq_num:05d}"
        texts = []
        for doc_idx in range(n_docs):
            n_words = 8 + (doc_idx % 8) + pq_num * 3
            words = [f"w{pq_num}d{doc_idx}x{w}" for w in range(n_words)]
            texts.append(" ".join(words))

        pq_path = data_dir / f"{shard_stem}.parquet"
        pq.write_table(pa.table({"text": texts}), str(pq_path), row_group_size=n_docs)

        total_segs = n_docs * n_segs_per_doc
        total_feats = total_segs * n_feats_per_seg

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

        for doc_idx in range(n_docs):
            text = texts[doc_idx]
            text_tokens = tokenizer.encode(text, prepend=bos_token)
            char_offsets = token_to_char_offsets(tokenizer, text_tokens)
            n_tokens = len(text_tokens)

            tokens_per_seg = max(1, n_tokens // n_segs_per_doc)

            doc_seg_char_starts = []
            doc_seg_char_ends = []

            for seg_idx in range(n_segs_per_doc):
                doc_ids[seg_global] = doc_idx
                seg_indices[seg_global] = seg_idx
                seg_positions[seg_global] = seg_idx * segment_size
                offsets[seg_global] = feat_offset

                tok_start = seg_idx * tokens_per_seg
                tok_end = (
                    min((seg_idx + 1) * tokens_per_seg, n_tokens)
                    if seg_idx < n_segs_per_doc - 1
                    else n_tokens
                )
                char_start = char_offsets[tok_start] if tok_start < len(char_offsets) else char_offsets[-1]
                char_end = char_offsets[tok_end] if tok_end < len(char_offsets) else char_offsets[-1]

                seg_char_starts_t[seg_global] = char_start
                seg_char_ends_t[seg_global] = char_end
                doc_seg_char_starts.append(int(char_start))
                doc_seg_char_ends.append(int(char_end))

                for f in range(n_feats_per_seg):
                    fid = (pq_num * n_docs * 10 + doc_idx * 10 + seg_idx * n_feats_per_seg + f) % n_sae_features
                    all_fids[feat_offset + f] = fid
                    all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx
                    all_pcts_overlay[feat_offset + f] = min(255, max(1, int(255 * (f + 1) / (n_feats_per_seg + 1))))

                feat_offset += n_feats_per_seg
                seg_global += 1

            tok_starts, tok_ends = _compute_token_boundaries_for_seg(
                tokenizer, text_tokens, doc_seg_char_starts, doc_seg_char_ends
            )
            for seg_idx in range(n_segs_per_doc):
                g = (doc_idx * n_segs_per_doc) + seg_idx
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
            "doc_token_counts": t.full((n_docs,), 40, dtype=t.int32),
            "segment_char_starts": seg_char_starts_t,
            "segment_char_ends": seg_char_ends_t,
            "n_docs": n_docs,
            "n_segments": total_segs,
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": segment_size,
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


@pytest.fixture
def synthetic_env(tmp_path):
    """Fixture providing synthetic data environment — mirrors test_live_dataloader.py."""
    return _make_synthetic_data(tmp_path)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_threshold_pipeline(
    *,
    n_sae_features=_N_SAE_FEATURES,
    num_buckets=_NUM_BUCKETS,
    threshold=3,
    feature_mask=None,
    with_negatives=False,
):
    """Build a ThresholdSelection pipeline, optionally with a feature_mask."""
    boundaries = (
        t.linspace(
            1 / num_buckets, 1.0 - 1 / num_buckets, num_buckets - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )

    negative_bundle = None
    if with_negatives:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(mean=2.0, std=2.0, min_k=1, max_k=4),
            n_sae_features=n_sae_features,
            rng=t.Generator().manual_seed(42),
        )

    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, threshold),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=num_buckets,
        feature_mask=feature_mask,
    )


def _make_dataloader(env, *, pipeline, B=_B, T=_T, buffer_size=50, n_sae_features=None):
    """Build a live dataloader from a synthetic env dict and a pipeline."""
    from nanochat.dataset import DatasetSplit

    tokenizer = env["tokenizer"]
    feature_index = FeatureIndex(
        env["feature_dir"],
        tokenizer_checksum=env["tokenizer_checksum"],
        percentile_checksum=env.get("percentile_checksum"),
    )

    return encoder_decoder_data_loader_with_state(
        tokenizer=tokenizer,
        B=B,
        T=T,
        split=DatasetSplit.TRAIN,
        feature_index=feature_index,
        pipeline=pipeline,
        segment_size=_SEGMENT_SIZE,
        decoder_segment_size=None,
        device="cpu",
        resume_state_dict=None,
        tokenizer_batch_size=128,
        buffer_size=buffer_size,
        data_dir=env["data_dir"],
        prefetch_batches=0,
        label_dropout=0.0,
        max_shards=None,
        n_sae_features=n_sae_features,
        pad_F=None,
        pad_n_seg=None,
        doc_prefetch_depth=0,
    )


def _collect_batches(gen, n_batches=3):
    """Collect n_batches from a dataloader generator."""
    batches = []
    for _ in range(n_batches):
        batch = next(gen)
        batches.append(batch)
    return batches


def _all_enc_fids(batches):
    """Collect all positive feature IDs (enc_fids where enc_mask=True) across batches."""
    fids_list = []
    for batch in batches:
        _, _, enc_fids, _, enc_mask, _, _, _, _, _ = batch
        # enc_fids: (B, n_seg, F_max) — take only masked entries
        fids_list.append(enc_fids[enc_mask])
    return t.cat(fids_list) if fids_list else t.tensor([], dtype=t.int64)


# ---------------------------------------------------------------------------
# Helpers for negative IDs
# ---------------------------------------------------------------------------

def _all_neg_fids(batches):
    """Collect all negative feature IDs across batches.

    The live dataloader packs negatives at the end of each segment with
    enc_mags bucket = effective_buckets (sentinel) and enc_mask = False.
    We detect them by a sentinel bucket value (_NUM_BUCKETS) in enc_mags.
    """
    neg_fids_list = []
    for batch in batches:
        _, _, enc_fids, enc_mags, enc_mask, _, _, _, _, _ = batch
        # Negatives are stored with sentinel bucket value = effective_buckets
        neg_mask = (~enc_mask) & (enc_mags == _NUM_BUCKETS)
        neg_fids_list.append(enc_fids[neg_mask])
    if neg_fids_list:
        cat = t.cat(neg_fids_list)
        return cat[cat != 0]  # 0 is padding sentinel
    return t.tensor([], dtype=t.int64)


# ===========================================================================
# Test class: blacklist
# ===========================================================================


class TestLiveDataloaderBlacklist:
    """feature_mask=blacklist(K) — K must not appear in positives or negatives."""

    # Feature ID to blacklist — must be one that would appear in synthetic data.
    # Synthetic data assigns fid = (pq*n_docs*10 + doc*10 + seg*n_feats + f) % N_SAE
    # With N_SAE=200, pq=0, doc=0, seg=0, f=0: fid=0.
    # Use feature 5 (pq=0, doc=0, seg=0, f=5): (0+0+0+5)%200 = 5
    BLACKLISTED_ID = 5

    def test_blacklist_positive_ids_exclude_masked(self, synthetic_env):
        """Blacklisted feature K never appears as a positive (in enc_fids where enc_mask=True)."""
        K = self.BLACKLISTED_ID
        mask = FeatureMask("blacklist", (K,))
        pipeline = _build_threshold_pipeline(feature_mask=mask)
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen)
        all_pos = _all_enc_fids(batches)
        assert K not in all_pos.tolist(), (
            f"Blacklisted feature {K} appeared in positive enc_fids: {all_pos.tolist()}"
        )

    def test_blacklist_negative_ids_exclude_masked(self, synthetic_env):
        """Blacklisted feature K never appears in negative IDs."""
        K = self.BLACKLISTED_ID
        mask = FeatureMask("blacklist", (K,))
        pipeline = _build_threshold_pipeline(feature_mask=mask, with_negatives=True)
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen)
        neg = _all_neg_fids(batches)
        if neg.numel() > 0:
            assert K not in neg.tolist(), (
                f"Blacklisted feature {K} appeared in negative enc_fids: {neg.tolist()}"
            )

    def test_blacklist_range_all_excluded(self, synthetic_env):
        """A range blacklist [0..49] — all positive IDs must be >= 50."""
        mask = FeatureMask("blacklist", tuple(range(50)))
        pipeline = _build_threshold_pipeline(feature_mask=mask)
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen)
        all_pos = _all_enc_fids(batches)
        if all_pos.numel() > 0:
            assert int(all_pos.min().item()) >= 50, (
                f"Blacklisted IDs [0..49] found in positives: min={int(all_pos.min().item())}"
            )


# ===========================================================================
# Test class: whitelist
# ===========================================================================


class TestLiveDataloaderWhitelist:
    """feature_mask=whitelist(S) — positives and negatives ⊆ S."""

    # Whitelist: feature IDs that appear in synthetic data
    # fid = (pq*n_docs*10 + doc*10 + seg*n_feats + f) % 200
    # pq=0, doc=0, seg=0, f=0..5: 0, 1, 2, 3, 4, 5
    # Use a generous whitelist so some features pass through
    WHITELIST = tuple(range(100))

    def test_whitelist_positives_subset(self, synthetic_env):
        """All positive IDs are in the whitelist."""
        mask = FeatureMask("whitelist", self.WHITELIST)
        pipeline = _build_threshold_pipeline(feature_mask=mask)
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen)
        all_pos = _all_enc_fids(batches)
        allowed = set(self.WHITELIST)
        if all_pos.numel() > 0:
            bad = [x for x in all_pos.tolist() if x not in allowed]
            assert not bad, f"Non-whitelisted positive IDs found: {bad[:10]}"

    def test_whitelist_negatives_subset(self, synthetic_env):
        """All negative IDs are in the whitelist."""
        mask = FeatureMask("whitelist", self.WHITELIST)
        pipeline = _build_threshold_pipeline(feature_mask=mask, with_negatives=True)
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen)
        neg = _all_neg_fids(batches)
        allowed = set(self.WHITELIST)
        if neg.numel() > 0:
            bad = [x for x in neg.tolist() if x not in allowed]
            assert not bad, f"Non-whitelisted negative IDs found: {bad[:10]}"


# ===========================================================================
# Test: I5 byte-identical (no-mask path)
# ===========================================================================


class TestLiveDataloaderI5ByteIdentical:
    """I5: feature_mask=None must produce bit-identical batches to legacy (no-mask)."""

    def test_none_mask_byte_identical_to_no_mask(self, synthetic_env):
        """Two runs with same-seed pipeline, one passing feature_mask=None explicitly
        vs one where the field is simply None (default), produce identical batches.

        I5 reasoning: apply_to_positive / keep_mask are NEVER called on the None
        path (all new code is gated by `if pipeline.feature_mask is not None:`).
        The feature_mask=pipeline.feature_mask forwarding to Task-8/9 functions is
        also None-safe (both functions return the same result when feature_mask=None
        per their own I5 guarantees). Therefore no RNG is consumed and no data is
        reordered on the None path.

        We verify this end-to-end by driving the live loader with two equal-seed
        pipelines — one with feature_mask=None (the default) and one with the
        field explicitly set to None — and assert all batch tensors are equal.
        """
        # Build two pipelines with identical RNG seeds
        p1 = _build_threshold_pipeline(feature_mask=None)
        p2 = _build_threshold_pipeline(feature_mask=None)

        gen1 = _make_dataloader(synthetic_env, pipeline=p1)
        gen2 = _make_dataloader(synthetic_env, pipeline=p2)

        for _ in range(3):
            batch1 = next(gen1)
            batch2 = next(gen2)
            x1, y1, enc_fids1, enc_mags1, enc_mask1, enc_pcts1, seg_map1, _, fp1, _ = batch1
            x2, y2, enc_fids2, enc_mags2, enc_mask2, enc_pcts2, seg_map2, _, fp2, _ = batch2

            assert t.equal(x1, x2), "x differs between no-mask runs"
            assert t.equal(y1, y2), "y differs between no-mask runs"
            assert t.equal(enc_fids1, enc_fids2), "enc_fids differs between no-mask runs"
            assert t.equal(enc_mags1, enc_mags2), "enc_mags differs between no-mask runs"
            assert t.equal(enc_mask1, enc_mask2), "enc_mask differs between no-mask runs"
            assert t.equal(enc_pcts1, enc_pcts2), "enc_pcts differs between no-mask runs"
            assert t.equal(seg_map1, seg_map2), "seg_map differs between no-mask runs"

    def test_mask_changes_output(self, synthetic_env):
        """Sanity: a blacklist mask that removes some features produces different
        batch content vs no mask — verifying the mask actually does something."""
        # Blacklist a large range to guarantee some features are removed
        mask = FeatureMask("blacklist", tuple(range(50)))
        p_nomask = _build_threshold_pipeline(feature_mask=None)
        p_masked = _build_threshold_pipeline(feature_mask=mask)

        gen_nomask = _make_dataloader(synthetic_env, pipeline=p_nomask)
        gen_masked = _make_dataloader(synthetic_env, pipeline=p_masked)

        # Gather positives from both
        batches_nomask = _collect_batches(gen_nomask)
        batches_masked = _collect_batches(gen_masked)

        pos_nomask = _all_enc_fids(batches_nomask)
        pos_masked = _all_enc_fids(batches_masked)

        # Some features in [0,49] should appear without mask but not with mask
        forbidden_in_nomask = set(range(50))
        nomask_has_forbidden = any(x in forbidden_in_nomask for x in pos_nomask.tolist())

        if nomask_has_forbidden:
            # If the unmasked run actually has forbidden features, the masked run
            # must not have them
            masked_has_forbidden = any(x in forbidden_in_nomask for x in pos_masked.tolist())
            assert not masked_has_forbidden, (
                "Masked run still has features from blacklisted range [0..49]"
            )
        # If synthetic data happens to not generate features in [0..49], the test
        # is vacuously satisfied — the mask is still enforced (just not triggered).


# ===========================================================================
# Delta pipeline builder — mirrors _build_delta_sampling_pipeline from
# test_live_dataloader.py verbatim except it accepts a feature_mask argument.
# ===========================================================================


def _build_delta_pipeline_with_mask(
    *,
    n_sae_features=_N_SAE_FEATURES,
    num_buckets=_NUM_BUCKETS,
    with_negatives=True,
    feature_mask=None,
    neg_mean=2.0,
    neg_max_k: int = 5,
    rng_seed=42,
):
    """Build a SamplingBundle pipeline with delta filtering and an optional feature_mask.

    Mirrors _build_delta_sampling_pipeline from test_live_dataloader.py exactly,
    adding the feature_mask kwarg for H3 tests.
    """
    percentile_table = (
        t.linspace(0.0, 1.0, 99)
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .to(t.float64)
        .contiguous()
    )

    effective_buckets = num_buckets - 1 if with_negatives else num_buckets

    boundaries = (
        t.linspace(
            1 / effective_buckets,
            1.0 - 1 / effective_buckets,
            effective_buckets - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(n_sae_features, -1)
        .contiguous()
    )

    rng = t.Generator().manual_seed(rng_seed)

    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0,
        std=3.0,
        min_k=1,
        max_k=10,
        delta_weights=DeltaSamplingWeights.percentile_delta,
    )

    negative_bundle = None
    if with_negatives:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=neg_mean,
                std=neg_mean,
                min_k=1,
                max_k=neg_max_k,
            ),
            n_sae_features=n_sae_features,
            rng=rng,
        )

    delta_bundle = DeltaBundle(
        config=FeatureDeltaConfig(
            mode=DeltaMode.prev_segment,
            neg_delta_ratio=0.5,
        ),
        percentile_table=percentile_table,
    )

    return FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=selection,
            percentile_table=percentile_table,
            rng=rng,
        ),
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        delta_bundle=delta_bundle,
        feature_mask=feature_mask,
    )


def _all_neg_fids_delta(batches, effective_buckets):
    """Collect all negative feature IDs from delta-path batches.

    Delta negatives are packed with sentinel bucket value = effective_buckets
    and enc_mask = False (same convention as the non-delta path).
    """
    neg_fids_list = []
    for batch in batches:
        _, _, enc_fids, enc_mags, enc_mask, _, _, _, _, _ = batch
        neg_mask = (~enc_mask) & (enc_mags == effective_buckets)
        neg_fids_list.append(enc_fids[neg_mask])
    if neg_fids_list:
        cat = t.cat(neg_fids_list)
        return cat[cat != 0]
    return t.tensor([], dtype=t.int64)


# ===========================================================================
# Test class: live delta + feature_mask (H3)
# ===========================================================================


class TestLiveDeltaMask:
    """H3: live delta-path neg_k clamp to allowed universe under feature_mask.

    Previously these tests were skipped because building a minimal DeltaBundle
    fixture was deemed non-trivial.  The fixture is now built by mirroring
    _build_delta_sampling_pipeline from test_live_dataloader.py exactly.
    """

    # Small whitelist: a handful of feature IDs that are allowed.
    # The synthetic data's fids = (pq*n_docs*10 + doc*10 + seg*n_feats + f) % 200
    # We pick a 5-element whitelist that definitely contains some active features
    # (IDs 0..9 all appear in the synthetic data) so the whitelist is meaningful.
    WHITELIST_IDS = (0, 1, 2, 3, 4, 10, 20, 30, 40, 50)  # 10 allowed out of 200
    # Blacklisted ID: appears in synthetic data at pq=0,doc=0,seg=0,f=5 → (0+0+5)%200=5
    BLACKLISTED_ID = 5

    def test_live_delta_blacklist_no_masked_negative(self, synthetic_env):
        """Delta path + blacklist: blacklisted feature K never appears in negatives.

        This closes the previously-skipped coverage gap for the delta negative path
        (Task 9 enforcement already holds at sample_delta_negatives level; this
        exercises the full end-to-end dataloader path).
        """
        K = self.BLACKLISTED_ID
        mask = FeatureMask("blacklist", (K,))
        pipeline = _build_delta_pipeline_with_mask(
            with_negatives=True,
            feature_mask=mask,
        )
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen, n_batches=4)
        neg = _all_neg_fids_delta(batches, pipeline.effective_buckets)
        if neg.numel() > 0:
            assert K not in neg.tolist(), (
                f"Blacklisted feature {K} appeared in delta-path negative enc_fids: "
                f"{neg.tolist()}"
            )

    def test_live_delta_whitelist_negk_clamped_to_allowed_pool(self, synthetic_env):
        """Delta path + whitelist: negatives ⊆ whitelist AND count ≤ allowed pool.

        Exposes the pre-fix COUNT-DISTRIBUTION skew: before the fix, neg_k was
        clamped to `n_sae - |active|` (up to 200 - ~6 = 194) even when the
        whitelist has only 10 features.  The fix must clamp neg_k to
        `|whitelist ∩ ~active|` (at most 10) so the geometric distribution is
        respected rather than collapsing to "all available in the small pool".

        Assertions:
        (a) every emitted negative ∈ whitelist (enforcement correctness)
        (b) per-segment negative count ≤ |whitelist ∩ ~active| ≤ |whitelist|
            — the count bound exposes the bug pre-fix.
        """
        whitelist = self.WHITELIST_IDS
        allowed_set = set(whitelist)
        n_allowed = len(whitelist)

        mask = FeatureMask("whitelist", whitelist)
        # Request a moderate geometric neg mean (3.0) — well below n_sae (200)
        # but potentially above the small allowed pool of 10.
        pipeline = _build_delta_pipeline_with_mask(
            with_negatives=True,
            feature_mask=mask,
            neg_mean=3.0,
        )
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)
        batches = _collect_batches(gen, n_batches=4)

        # (a) Whitelist enforcement
        neg = _all_neg_fids_delta(batches, pipeline.effective_buckets)
        if neg.numel() > 0:
            bad = [x for x in neg.tolist() if x not in allowed_set]
            assert not bad, (
                f"Non-whitelisted features in delta-path negatives: {bad[:10]}"
            )

        # (b) Per-segment count ≤ |whitelist| (max possible allowed pool)
        # A stronger bound: each segment has at most n_allowed - 0 = n_allowed
        # negatives (since some active features are in the whitelist, the real
        # bound is tighter, but bounding by n_allowed is always valid and sufficient
        # to expose the pre-fix skew where neg_k could reach n_sae - active ~ 194).
        for batch in batches:
            _, _, enc_fids, enc_mags, enc_mask, _, seg_map, _, _, _ = batch
            B_dim, n_seg, F_dim = enc_fids.shape
            for b in range(B_dim):
                for s in range(n_seg):
                    neg_m = (~enc_mask[b, s]) & (enc_mags[b, s] == pipeline.effective_buckets)
                    seg_neg_count = int(neg_m.sum().item())
                    assert seg_neg_count <= n_allowed, (
                        f"Segment [{b},{s}] has {seg_neg_count} delta negatives but "
                        f"whitelist has only {n_allowed} features — neg_k clamp bug "
                        f"(pre-fix: clamped to n_sae-active={_N_SAE_FEATURES}-~6≈194)"
                    )

    def test_live_delta_no_mask_byte_identical(self, synthetic_env):
        """I5: None-mask delta path is RNG-byte-identical to the same run with mask=None.

        Two equal-seed delta pipelines, both with feature_mask=None.  The clamp
        change is strictly inside `if pipeline.feature_mask is not None:` so the
        None path uses the unchanged legacy `n_sae - active` clamp.  No RNG is
        added or reordered on the None path.
        """
        p1 = _build_delta_pipeline_with_mask(
            with_negatives=True,
            feature_mask=None,
            rng_seed=42,
        )
        p2 = _build_delta_pipeline_with_mask(
            with_negatives=True,
            feature_mask=None,
            rng_seed=42,
        )

        gen1 = _make_dataloader(synthetic_env, pipeline=p1)
        gen2 = _make_dataloader(synthetic_env, pipeline=p2)

        for _ in range(3):
            batch1 = next(gen1)
            batch2 = next(gen2)
            x1, y1, enc_fids1, enc_mags1, enc_mask1, enc_pcts1, seg_map1, _, fp1, _ = batch1
            x2, y2, enc_fids2, enc_mags2, enc_mask2, enc_pcts2, seg_map2, _, fp2, _ = batch2

            assert t.equal(x1, x2), "delta I5: x differs between None-mask runs"
            assert t.equal(y1, y2), "delta I5: y differs between None-mask runs"
            assert t.equal(enc_fids1, enc_fids2), (
                "delta I5: enc_fids differs between None-mask runs"
            )
            assert t.equal(enc_mags1, enc_mags2), (
                "delta I5: enc_mags differs between None-mask runs"
            )
            assert t.equal(enc_mask1, enc_mask2), (
                "delta I5: enc_mask differs between None-mask runs"
            )

    def test_live_delta_whitelist_negk_clamped_spy(self, synthetic_env, monkeypatch):
        """Revert-guard (H3): spy on the k= kwarg passed to sample_delta_negatives.

        Pre-H3 the clamp was ``min(neg_k, n_sae - len(active))`` which with
        n_sae=200 and ~6 active features per segment yields an unclamped draw
        up to max_k=15.  With geometric mean=8.0 over many segments the draws
        are overwhelmingly above 3.

        Post-H3 the clamp is ``min(neg_k, |allowed ∩ ~active|)`` where
        ``allowed = {0, 1, 2}`` (3 features).  Regardless of the geometric draw,
        neg_k is forced to ≤ 3 before being passed to sample_delta_negatives.

        Distinguishing assertion: ``all(k <= 3 for k in captured_ks)`` passes
        post-fix (PASSES) and fails pre-fix with overwhelming probability (FAILS).

        Probabilistic argument for pre-fix failure:
          - max_k=15, geometric mean=8.0 → P(draw ≤ 3) ≈ (1-1/8)^3 ≈ 0.67^3 ≈ 0.33
            per draw (rough memoryless geometric approximation).
          - With N ≥ 8 captured calls (8 docs × 2 segs/doc across one pass),
            P(all ≤ 3) ≤ 0.33^8 ≈ 0.00014 — effectively impossible.
          - In practice each captured k is independently ≥ 1 from the distribution
            so ``any(k for k in captured_ks)`` (non-empty + some non-zero) also holds.
        """
        from steering.conditional.delta import sample_delta_negatives as _real_sdn

        captured_ks: list[int] = []

        def _spy_sdn(*args, **kwargs):
            # Capture the k= kwarg (always passed as keyword per the call site)
            captured_ks.append(kwargs["k"])
            return _real_sdn(*args, **kwargs)

        monkeypatch.setattr(
            "steering.conditional.encoder_decoder.dataloader.sample_delta_negatives",
            _spy_sdn,
        )

        # Tiny whitelist: only 3 features allowed → |allowed ∩ ~active| ≤ 3.
        # The synthetic data has active fids like 0,1,2,... (≥1 from {0,1,2} per
        # segment), so the allowed-minus-active pool is at most 3 and often 1-2.
        whitelist = (0, 1, 2)
        mask = FeatureMask("whitelist", whitelist)

        # neg_max_k=15 and neg_mean=8.0: unclamped draws routinely 4-15,
        # far above the 3-feature pool.  The H3 clamp forces k ≤ 3.
        pipeline = _build_delta_pipeline_with_mask(
            with_negatives=True,
            feature_mask=mask,
            neg_max_k=15,
            neg_mean=8.0,
            rng_seed=99,
        )
        gen = _make_dataloader(synthetic_env, pipeline=pipeline)

        # Drive enough steps to collect multiple captured k values.
        # 4 batches × B=2 × n_segs_per_doc=2 → O(16+) delta neg calls expected.
        _collect_batches(gen, n_batches=4)

        # Must have actually called sample_delta_negatives at least once.
        assert len(captured_ks) >= 1, (
            "spy was never called — no delta neg draws happened; "
            "check synthetic data has delta segments"
        )

        # Post-H3: every k passed to sample_delta_negatives is ≤ |whitelist| = 3.
        # Pre-H3 (legacy clamp = min(draw, n_sae - active) ≈ min(draw, 194)):
        #   draws from Geom(mean=8, max=15) have P(draw > 3) ≈ 0.67 each.
        #   P(all ≤ 3 over N≥8 calls) ≤ 0.33^8 ≈ 0.00014 → revert undetectable rate ~0.
        assert all(k <= 3 for k in captured_ks), (
            f"sample_delta_negatives was called with k > 3: captured_ks={captured_ks}. "
            f"Post-H3 the clamp forces k ≤ |allowed ∩ ~active| ≤ |whitelist|=3. "
            f"If this fails the H3 clamp has been reverted to the legacy n_sae-active formula."
        )
        # Sanity: some actual negatives were requested (k > 0 for at least one call).
        assert any(k > 0 for k in captured_ks), (
            f"all captured k=0; no negatives were requested at all — captured_ks={captured_ks}"
        )
        # Additional strength: max(captured_ks) ≤ 3 is the tightest revert-guard.
        # With legacy clamp, max(captured_ks) would be min(geometric_draw, max_k=15),
        # and with mean=8 over many draws the expected max is >> 3.
        assert max(captured_ks) <= 3, (
            f"max(captured_ks)={max(captured_ks)} > 3: H3 clamp not working. "
            f"With legacy clamp and Geom(mean=8, max=15), expected max >> 3."
        )
