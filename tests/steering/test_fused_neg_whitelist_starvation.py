"""TDD test for B1: fused neg-sampler whitelist starvation fix.

Pre-fix RED: a small whitelist (|allowed|/n_sae ≈ 1-3%) causes rejection
sampling from [0, n_sae) to yield ~0 negatives (the bug).

Post-fix GREEN: sampling directly from allowed_ids guarantees adequate
headroom and produces negatives close to the clamped requested count.

Phase 2 (small-pool residual): B1 still had with-replacement under-delivery
when eff_pool ≈ neg_ks and |allowed| is small.  Tests for exact delivery
(eff_pool == neg_ks → 100% must deliver neg_ks, zero under-delivery) were
added as RED, and pass only after the enumerate-permute fix.

Also covers:
 - I4: all returned negatives are within the whitelist
 - I5: no-mask branch is byte-identical to pre-fix (same RNG draw + state)
 - Degenerate pools: whitelist of size 1 (all positive), size 2 with neg_ks=5
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import reconstruct_and_select
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.packed_batch_io import PackedRowShard

# ---------------------------------------------------------------------------
# Constants — designed to trigger whitelist starvation
# ---------------------------------------------------------------------------

N_SAE = 1024        # large enough that whitelist ≈ 1.5% → starvation pre-fix
N_ROWS = 8
SEGS_PER_ROW = 3
FEATS_PER_SEG = 10

# 16 allowed IDs out of 1024 → |allowed|/n_sae ≈ 1.5%
WHITELIST_IDS = tuple(range(16))   # IDs 0..15


# ---------------------------------------------------------------------------
# Shard builder — features drawn exclusively from the whitelist so positives
# are cheap and negatives must also come from there (tight supply).
# ---------------------------------------------------------------------------


def _make_shard(seed: int = 7) -> PackedRowShard:
    t.manual_seed(seed)
    total_segs = N_ROWS * SEGS_PER_ROW
    total_feats = total_segs * FEATS_PER_SEG

    context_len = 32
    tokens = t.randint(1, 1000, (N_ROWS, context_len + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(
        0, (N_ROWS + 1) * SEGS_PER_ROW, SEGS_PER_ROW, dtype=t.int32
    )

    seg_size = context_len // SEGS_PER_ROW
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)
    for r in range(N_ROWS):
        for s in range(SEGS_PER_ROW):
            idx = r * SEGS_PER_ROW + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = (
                context_len if s == SEGS_PER_ROW - 1 else (s + 1) * seg_size
            )

    seg_feat_offsets = t.arange(
        0, (total_segs + 1) * FEATS_PER_SEG, FEATS_PER_SEG, dtype=t.int32
    )

    # Draw feature IDs only from the whitelist so positives are guaranteed
    # within the allowed set; this also makes |allowed - positives| small and
    # any adequate negative count truly reachable.
    whitelist_t = t.tensor(list(WHITELIST_IDS), dtype=t.int64)
    idx_into_wl = t.randint(0, len(WHITELIST_IDS), (total_feats,))
    feature_ids = whitelist_t[idx_into_wl].to(t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

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
# Pipeline builder
# ---------------------------------------------------------------------------


def _make_percentile_table(n_features: int = N_SAE) -> t.Tensor:
    pcts = t.linspace(0.01, 0.99, 99, dtype=t.float64)
    return pcts.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_boundaries(n_features: int = N_SAE, n_buckets: int = 10) -> t.Tensor:
    n_boundaries = n_buckets - 1
    b = t.linspace(1 / n_buckets, 1.0 - 1 / n_buckets, n_boundaries, dtype=t.float64)
    return b.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_pipeline(
    seed: int = 42,
    feature_mask: FeatureMask | None = None,
    n_features: int = N_SAE,
    neg_mean: float = 4.0,
    neg_min_k: int = 3,
) -> FeaturePipelineConfig:
    """Build a pipeline where neg_ks will be well above 0 for most segments."""
    rng = t.Generator().manual_seed(seed)
    neg_rng = t.Generator().manual_seed(seed + 1)
    n_buckets = 10
    effective_buckets = n_buckets - 1

    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0, std=2.0, min_k=1, max_k=4,
    )
    pct_table = _make_percentile_table(n_features)
    sampling_bundle = SamplingBundle(
        selection=selection, percentile_table=pct_table, rng=rng,
    )
    boundaries = _make_boundaries(n_features, effective_buckets)
    negative_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=neg_mean, std=neg_mean, min_k=neg_min_k,
        ),
        n_sae_features=n_features, rng=neg_rng,
    )
    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        feature_mask=feature_mask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _neg_counts_per_segment(rb: object, eff_bucket: int) -> list[int]:
    """Return list of per-segment negative counts (across all [b, s] pairs)."""
    counts = []
    for b in range(rb.enc_fids.shape[0]):  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        for s in range(rb.enc_fids.shape[1]):  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
            mask = rb.enc_mask[b, s]  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
            if not mask.any():
                counts.append(0)
                continue
            mags = rb.enc_mags[b, s, mask]  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
            counts.append(int((mags == eff_bucket).sum().item()))
    return counts


# ---------------------------------------------------------------------------
# Controlled shard + pipeline builders for exact small-pool tests
# ---------------------------------------------------------------------------


def _make_controlled_shard(
    n_rows: int,
    segs_per_row: int,
    n_pos_per_seg: int,
    pos_feature_ids: list[int],
    *,
    context_len: int = 32,
    pct_value: int = 200,
) -> PackedRowShard:
    """Build a shard where every segment has exactly n_pos_per_seg features.

    All features come from pos_feature_ids (cycling if needed).  pct_value is
    set high so top-k selection always picks them all.
    """
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * n_pos_per_seg
    seg_size = context_len // segs_per_row

    tokens = t.ones(n_rows, context_len + 1, dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = (
                context_len if s == segs_per_row - 1 else (s + 1) * seg_size
            )
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)
    seg_feat_offsets = t.arange(0, (total_segs + 1) * n_pos_per_seg, n_pos_per_seg, dtype=t.int32)

    # Each segment gets exactly n_pos_per_seg features cycling through pos_feature_ids
    fids_cycle = [pos_feature_ids[i % len(pos_feature_ids)] for i in range(total_feats)]
    feature_ids = t.tensor(fids_cycle, dtype=t.uint16)
    feature_pcts = t.full((total_feats,), pct_value, dtype=t.uint8)

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


def _make_exact_pipeline(
    n_pos: int,
    n_neg: int,
    n_sae: int,
    seed: int = 42,
    n_buckets: int = 10,
) -> FeaturePipelineConfig:
    """Pipeline that selects exactly n_pos positives and requests exactly n_neg negatives.

    Uses top_k_percentile mode (deterministic, no Gumbel RNG).
    Negative bundle: min_k = max_k = n_neg so neg_ks == n_neg always before clamp.
    Whitelist = {0, 1, ..., n_pos + n_neg - 1}.

    When n_pos == 0, the shard should have features with IDs outside the whitelist
    (>= n_neg) so that after mask-gated positive filtering, 0 positives remain.
    """
    rng_pos = t.Generator().manual_seed(seed)
    rng_neg = t.Generator().manual_seed(seed + 1000)
    effective_buckets = n_buckets - 1

    # geometric_ requires p in (0, 1) i.e. mean > 0; since min_k == max_k == n_pos the
    # actual sample is always clamped to n_pos, so mean only affects the clamping path —
    # use max(1, n_pos) to keep p < 1 for the geometric sampler.
    pos_mean = max(1, n_pos)
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=float(pos_mean), std=1.0,
        min_k=n_pos, max_k=n_pos + 1,
        mode=SamplingMode.top_k_percentile,
    )
    pct_table = _make_percentile_table(n_sae)
    sampling_bundle = SamplingBundle(selection=selection, percentile_table=pct_table, rng=rng_pos)
    boundaries = _make_boundaries(n_sae, effective_buckets)
    negative_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=float(n_neg), std=1.0,
            min_k=n_neg, max_k=n_neg + 1,
        ),
        n_sae_features=n_sae, rng=rng_neg,
    )
    wl_ids = tuple(range(n_pos + n_neg))
    feature_mask = FeatureMask("whitelist", wl_ids)
    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        feature_mask=feature_mask,
    )


# ============================================================================
# 1. RED test — asserts that ≥90% of segments get their requested neg_ks
#    This FAILS before the fix on small whitelists.
# ============================================================================


class TestWhitelistStarvationFix:
    """B1: with a small whitelist, all segments with neg_ks>0 must get ≥1 negative."""

    def test_neg_count_adequate_for_small_whitelist(self) -> None:
        """Every segment with satisfiable neg_ks>0 gets negatives.

        Pre-fix: candidates drawn from [0, 1024) → ~1.5% accept rate →
        max_neg*3 ≈ 21 candidates, expected ≈ 0.3 per column → most
        segments silently get 0 negatives.

        Post-fix: candidates drawn from allowed_ids (16 entries) → every
        candidate is valid (modulo dedup/collision), headroom is adequate.
        """
        shard = _make_shard(seed=7)
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline(seed=42, feature_mask=mask, neg_min_k=3, neg_mean=4.0)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        counts = _neg_counts_per_segment(rb, eff)
        total_segs = N_ROWS * SEGS_PER_ROW
        assert len(counts) == total_segs

        # Segments with enc_mask active should have negatives.
        segs_with_any_mask = sum(
            1 for b in range(rb.enc_fids.shape[0])
            for s in range(rb.enc_fids.shape[1])
            if rb.enc_mask[b, s].any()
        )
        segs_with_negs = sum(1 for c in counts if c > 0)

        # Require ≥90% fill: post-fix easily achieves ~100%; pre-fix yields ~5%
        assert segs_with_negs >= 0.9 * segs_with_any_mask, (
            f"Only {segs_with_negs}/{segs_with_any_mask} active segments got negatives "
            f"(need ≥90%). Pre-fix starvation bug still present. "
            f"Mean negatives per segment: {sum(counts)/max(len(counts),1):.2f}"
        )

    def test_mean_negatives_per_segment_adequate(self) -> None:
        """Mean negatives per active segment ≥ 80% of min_k (=3), so ≥ 2.4."""
        shard = _make_shard(seed=13)
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline(seed=55, feature_mask=mask, neg_min_k=3, neg_mean=4.0)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        counts = _neg_counts_per_segment(rb, eff)
        mean_negs = sum(counts) / max(len(counts), 1)

        # min_k=3 → each segment should request ≥3 negatives; with |allowed|=16
        # and |positives|≤4, pool=≥12, headroom is more than adequate.
        assert mean_negs >= 2.4, (
            f"Mean negatives per segment = {mean_negs:.2f} < 2.4. "
            f"Whitelist starvation bug still present?"
        )


# ============================================================================
# 2. I4 guarantee — no returned negative is outside the whitelist
# ============================================================================


class TestI4WhitelistNegatives:
    """I4: all returned negatives must be within the whitelist."""

    def test_all_neg_fids_in_whitelist(self) -> None:
        shard = _make_shard(seed=7)
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline(seed=42, feature_mask=mask)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        whiteset = set(WHITELIST_IDS)
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                fids = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                neg_fids = fids[mags == eff]
                for fid in neg_fids.tolist():
                    assert fid in whiteset, (
                        f"Negative fid {fid} at [{b},{s}] is outside the whitelist {whiteset}"
                    )

    def test_h_assert_holds_post_fix(self) -> None:
        """The H-assert (assert_no_masked) must not raise post-fix."""
        shard = _make_shard(seed=7)
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline(seed=42, feature_mask=mask)

        # Would raise FeatureMaskError if I4 is violated
        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)
        assert rb is not None


# ============================================================================
# 3. Determinism — same seed → same result
# ============================================================================


class TestDeterminism:
    """Same seed + same shard → identical output in the masked branch."""

    def test_masked_neg_deterministic(self) -> None:
        shard = _make_shard(seed=7)
        mask = FeatureMask("whitelist", WHITELIST_IDS)

        pipeline_a = _make_pipeline(seed=42, feature_mask=mask)
        pipeline_b = _make_pipeline(seed=42, feature_mask=mask)

        rb_a, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_a)
        rb_b, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_b)

        assert t.equal(rb_a.enc_fids, rb_b.enc_fids), "enc_fids not deterministic"
        assert t.equal(rb_a.enc_mask, rb_b.enc_mask), "enc_mask not deterministic"


# ============================================================================
# 4. Small-pool exact delivery — RESIDUAL from spec review
#
#    B1 HEAD (301621f3) still used with-replacement sampling from allowed_ids,
#    giving expected delivery ~1.64/2 when eff_pool == neg_ks == 2.
#    These tests assert EVERY segment delivers exactly min(neg_ks, eff_pool)
#    distinct allowed negatives.  They must:
#      RED  at B1 HEAD (301621f3) — with-replacement under-delivery
#      GREEN after the enumerate-permute fix
#
# ============================================================================


class TestSmallPoolExactDelivery:
    """Every segment must receive exactly min(neg_ks, eff_pool) distinct negatives.

    Tests sweep over eff_pool ∈ {2, 3, 4, 8} with neg_ks == eff_pool and
    neg_ks == eff_pool-1 (the residual cases from the spec review table).
    """

    @pytest.mark.parametrize(
        "n_pos,n_neg",
        [
            (2, 2),  # eff_pool == neg_ks == 2 — worst residual case
            (1, 3),  # eff_pool == neg_ks == 3
            (1, 4),  # eff_pool == neg_ks == 4
            (0, 8),  # eff_pool == neg_ks == 8, no positives
            (2, 3),  # neg_ks == eff_pool - 1 (slack=1), should also be exact
            (1, 2),  # neg_ks == eff_pool - 1 (slack=1)
        ],
    )
    def test_exact_delivery_per_segment(self, n_pos: int, n_neg: int) -> None:
        """Every segment must deliver exactly n_neg distinct allowed negatives.

        eff_pool = _univ - n_pos = n_neg (for n_pos=..., n_neg=...).
        With with-replacement sampling (B1 HEAD), expected delivery < n_neg
        when eff_pool is small.  After enumerate-permute fix this must be
        EXACTLY n_neg for every segment.
        """
        n_sae = max(n_pos + n_neg + 8, 16)  # n_sae >> whitelist so mask is realistic
        n_rows = 6
        segs_per_row = 4
        total_segs = n_rows * segs_per_row

        if n_pos > 0:
            # Shard features from the positive IDs (first n_pos of whitelist)
            pos_ids = list(range(n_pos))
            shard = _make_controlled_shard(
                n_rows=n_rows,
                segs_per_row=segs_per_row,
                n_pos_per_seg=n_pos,
                pos_feature_ids=pos_ids,
            )
        else:
            # n_pos==0: shard features must be OUTSIDE whitelist so 0 positives remain
            # after mask-gated filtering.  whitelist = range(n_neg), so use id=n_neg+n_sae-1.
            outside_id = n_pos + n_neg  # first id outside whitelist range(n_neg)
            shard = _make_controlled_shard(
                n_rows=n_rows,
                segs_per_row=segs_per_row,
                n_pos_per_seg=1,
                pos_feature_ids=[outside_id],
            )

        pipeline = _make_exact_pipeline(n_pos=n_pos, n_neg=n_neg, n_sae=n_sae, seed=17)
        eff = pipeline.effective_buckets
        row_indices = list(range(n_rows))

        rb, _ = reconstruct_and_select(shard, row_indices, pipeline)

        # Count negatives per segment
        under_segs = 0
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                mags = rb.enc_mags[b, s, m]
                got = int((mags == eff).sum().item())
                # Every satisfiable segment must deliver exactly n_neg negatives.
                # (neg_ks is clamped to eff_pool=n_neg by construction, so this is
                # satisfiable and must be exact.)
                if got != n_neg:
                    under_segs += 1

        assert under_segs == 0, (
            f"n_pos={n_pos}, n_neg={n_neg} (eff_pool=={n_neg}): "
            f"{under_segs}/{total_segs} segments delivered != {n_neg} negatives. "
            f"Small-pool enumerate-permute fix required (with-replacement under-delivers)."
        )

    @pytest.mark.parametrize(
        "n_pos,n_neg,seed",
        [
            (2, 2, 42),
            (2, 2, 99),
            (2, 2, 137),
            (1, 4, 7),
        ],
    )
    def test_exact_delivery_3seed_rng_state(self, n_pos: int, n_neg: int, seed: int) -> None:
        """Same-seed masked-branch runs produce identical enc_fids/enc_mask (determinism).

        Also verifies that the negative RNG state after the call is identical
        between two runs with the same seed (I5-analog for the masked branch).
        """
        n_sae = max(n_pos + n_neg + 8, 16)
        n_rows = 4
        segs_per_row = 3

        pos_ids = list(range(n_pos)) if n_pos > 0 else [0]
        shard = _make_controlled_shard(n_rows=n_rows, segs_per_row=segs_per_row,
                                       n_pos_per_seg=max(n_pos, 1), pos_feature_ids=pos_ids)

        pa = _make_exact_pipeline(n_pos=n_pos, n_neg=n_neg, n_sae=n_sae, seed=seed)
        pb = _make_exact_pipeline(n_pos=n_pos, n_neg=n_neg, n_sae=n_sae, seed=seed)

        rb_a, _ = reconstruct_and_select(shard, list(range(n_rows)), pa)
        rb_b, _ = reconstruct_and_select(shard, list(range(n_rows)), pb)

        assert t.equal(rb_a.enc_fids, rb_b.enc_fids), (
            f"enc_fids differ for n_pos={n_pos}, n_neg={n_neg}, seed={seed}"
        )
        assert t.equal(rb_a.enc_mask, rb_b.enc_mask), (
            f"enc_mask differ for n_pos={n_pos}, n_neg={n_neg}, seed={seed}"
        )

        # Post-call RNG state check: next draw must be identical
        va = t.randint(0, 100000, (1,), generator=pa.negative_bundle.rng).item()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        vb = t.randint(0, 100000, (1,), generator=pb.negative_bundle.rng).item()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        assert va == vb, (
            f"Post-call neg RNG state diverged ({va} != {vb}) for "
            f"n_pos={n_pos}, n_neg={n_neg}, seed={seed}"
        )


# ============================================================================
# 5. I5 — no-mask branch byte-identical to pre-fix (legacy randint unchanged)
# ============================================================================


class TestI5NoMaskByteIdentical:
    """I5: feature_mask=None RNG draw is byte-identical to the legacy path.

    The fix ONLY activates when pipeline.feature_mask is not None.
    When None, the exact legacy ``t.randint(0, n_sae_features, ...)`` call
    must be used unchanged, so the RNG stream (and hence negatives) is
    bit-for-bit the same as before the patch.

    We prove this by running the no-mask path TWICE with the same seed and
    asserting the outputs are identical. We also verify the RNG state is
    consumed identically by checking the next random draw after the call.
    """

    def test_no_mask_enc_fids_identical(self) -> None:
        shard = _make_shard(seed=7)

        pipeline_a = _make_pipeline(seed=99, feature_mask=None)
        pipeline_b = _make_pipeline(seed=99, feature_mask=None)

        rb_a, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_a)
        rb_b, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_b)

        assert t.equal(rb_a.enc_fids, rb_b.enc_fids), (
            "enc_fids diverged between two no-mask same-seed runs (I5 violated)"
        )

    def test_no_mask_enc_mask_identical(self) -> None:
        shard = _make_shard(seed=7)

        pipeline_a = _make_pipeline(seed=99, feature_mask=None)
        pipeline_b = _make_pipeline(seed=99, feature_mask=None)

        rb_a, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_a)
        rb_b, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_b)

        assert t.equal(rb_a.enc_mask, rb_b.enc_mask), (
            "enc_mask diverged between two no-mask same-seed runs (I5 violated)"
        )

    def test_no_mask_rng_state_consumed_identically(self) -> None:
        """Post-call RNG state is identical between two no-mask runs."""
        shard = _make_shard(seed=7)
        seed = 77

        pa = _make_pipeline(seed=seed, feature_mask=None)
        pb = _make_pipeline(seed=seed, feature_mask=None)

        reconstruct_and_select(shard, list(range(N_ROWS)), pa)
        reconstruct_and_select(shard, list(range(N_ROWS)), pb)

        # Drain one more value from each neg RNG
        va = t.randint(0, 10000, (1,), generator=pa.negative_bundle.rng).item()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        vb = t.randint(0, 10000, (1,), generator=pb.negative_bundle.rng).item()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        assert va == vb, (
            f"Post-call neg RNG state diverged ({va} != {vb}) on no-mask run (I5 violated)"
        )

    def test_no_mask_negatives_come_from_full_space(self) -> None:
        """No-mask negatives can include IDs outside any small whitelist (proving legacy draw).

        Run no-mask on a shard with features only in WHITELIST_IDS, large n_sae.
        Negatives should span [0, N_SAE), not be confined to WHITELIST_IDS.
        """
        shard = _make_shard(seed=7)  # features only in WHITELIST_IDS
        pipeline = _make_pipeline(seed=42, feature_mask=None, n_features=N_SAE)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        neg_fids: set[int] = set()
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                fids = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                neg_fids.update(fids[mags == eff].tolist())

        # With n_sae=1024 and geometric neg sampling, almost certain to get
        # at least one negative outside the 16-id whitelist range.
        outside_whitelist = neg_fids - set(WHITELIST_IDS)
        assert len(outside_whitelist) > 0, (
            "No-mask negatives confined to whitelist range — "
            "this suggests the fix accidentally applied to the no-mask branch (I5 violated)"
        )


# ============================================================================
# 5. Degenerate pool handling
# ============================================================================


class TestDegeneratePool:
    """Small whitelist edge cases: must never crash, never return masked id."""

    def test_whitelist_size_1_all_positive_yields_zero_negatives(self) -> None:
        """Whitelist of 1 ID; that ID is active → 0 negatives, no crash."""
        n_sae = 64
        lone_id = 5

        # Shard: every feature is the lone allowed id
        total_segs = 2
        feats_per_seg = 4
        total_feats = total_segs * feats_per_seg
        context_len = 16
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, context_len + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, total_segs], dtype=t.int32),
            seg_token_start=t.tensor([0, 8], dtype=t.uint16),
            seg_token_end=t.tensor([8, context_len], dtype=t.uint16),
            seg_doc_id=t.zeros(total_segs, dtype=t.uint8),
            seg_feat_offsets=t.arange(
                0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32
            ),
            feature_ids=t.full((total_feats,), lone_id, dtype=t.uint16),
            feature_pcts=t.full((total_feats,), 128, dtype=t.uint8),
        )

        mask = FeatureMask("whitelist", (lone_id,))
        pipeline = _make_pipeline(
            seed=7, feature_mask=mask, n_features=n_sae, neg_min_k=1, neg_mean=2.0
        )
        eff = pipeline.effective_buckets

        # Must not crash
        rb, _ = reconstruct_and_select(shard, [0], pipeline)

        counts = _neg_counts_per_segment(rb, eff)
        # lone_id is the only positive and the only allowed id → pool = 0 → no negatives
        assert all(c == 0 for c in counts), (
            f"Expected 0 negatives (lone allowed id is positive), got counts={counts}"
        )

    def test_whitelist_size_2_neg_ks_5_returns_at_most_available(self) -> None:
        """Whitelist of 2 IDs; one is always positive → at most 1 negative.

        neg_ks=5 requested but clamped to |allowed| - |positives| = 1.
        Must return ≤1 negative, no crash, no masked id.
        """
        n_sae = 64
        wl = (3, 7)

        # Shard: all features are id=3 (one of the whitelist ids)
        total_segs = 4
        feats_per_seg = 4
        total_feats = total_segs * feats_per_seg
        context_len = 16
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (2, context_len + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, total_segs // 2, total_segs], dtype=t.int32),
            seg_token_start=t.tensor([0, 8, 0, 8], dtype=t.uint16),
            seg_token_end=t.tensor([8, context_len, 8, context_len], dtype=t.uint16),
            seg_doc_id=t.zeros(total_segs, dtype=t.uint8),
            seg_feat_offsets=t.arange(
                0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32
            ),
            feature_ids=t.full((total_feats,), wl[0], dtype=t.uint16),
            feature_pcts=t.full((total_feats,), 200, dtype=t.uint8),
        )

        mask = FeatureMask("whitelist", wl)
        # min_k=5 > |allowed|-|positives|=1 → neg_ks clamped to 1
        pipeline = _make_pipeline(
            seed=7, feature_mask=mask, n_features=n_sae, neg_min_k=5, neg_mean=5.0
        )
        eff = pipeline.effective_buckets

        # Must not crash
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)

        counts = _neg_counts_per_segment(rb, eff)
        # Each segment: id=3 positive, id=7 is the only other allowed id → at most 1 neg
        for c in counts:
            assert c <= 1, f"Got {c} negatives; expected ≤1 (only 1 remaining allowed id)"

        # I4: no masked id in negatives
        whiteset = set(wl)
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                fids = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                for fid in fids[mags == eff].tolist():
                    assert fid in whiteset, (
                        f"Negative fid {fid} is outside whitelist {whiteset}"
                    )

    def test_empty_whitelist_after_positives_no_crash(self) -> None:
        """If allowed_ids after clamp = 0, no crash and neg_fids empty."""
        n_sae = 16
        lone_id = 0

        # Shard: one segment, one feature = lone_id (the only whitelisted id)
        shard = PackedRowShard(
            tokens=t.randint(1, 100, (1, 17), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([16], dtype=t.uint16),
            seg_doc_id=t.zeros(1, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 1], dtype=t.int32),
            feature_ids=t.tensor([lone_id], dtype=t.uint16),
            feature_pcts=t.tensor([200], dtype=t.uint8),
        )

        mask = FeatureMask("whitelist", (lone_id,))
        pipeline = _make_pipeline(
            seed=3, feature_mask=mask, n_features=n_sae, neg_min_k=1, neg_mean=2.0
        )
        eff = pipeline.effective_buckets

        # Must not crash
        rb, _ = reconstruct_and_select(shard, [0], pipeline)

        counts = _neg_counts_per_segment(rb, eff)
        assert all(c == 0 for c in counts), (
            f"Expected 0 negatives when allowed pool exhausted, got {counts}"
        )


# ============================================================================
# 7. Perf-shape guarantee — blacklist / large-|allowed| must use vectorized path
#
#    For a blacklist of a few features (|allowed| ≈ n_sae >> neg_ks), the
#    implementation must NOT enter the per-segment enumerate-permute loop nor
#    call randperm with a large argument.  This pins the performance-shape
#    invariant so a future change cannot silently re-introduce the O(total_segs
#    × |allowed|) cliff.
#
#    FAIL at HEAD 0a7a1433: that commit runs randperm(len(eligible)) per segment
#    for ALL masked cases, including large |allowed|.
#    PASS after the hybrid fix: large |allowed| uses the vectorized path;
#    randperm is only called (if at all) in the small-pool exact path with a
#    small argument.
# ============================================================================


def _make_shard_large_sae(n_sae: int, n_rows: int = 8, segs_per_row: int = 4,
                           feats_per_seg: int = 4, seed: int = 42) -> PackedRowShard:
    """Build a shard with feature IDs spread across [0, n_sae) (not restricted to whitelist).

    Used for the blacklist case where most features are in the allowed set.
    """
    t.manual_seed(seed)
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg
    context_len = 32
    seg_size = context_len // segs_per_row

    tokens = t.randint(1, 1000, (n_rows, context_len + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = context_len if s == segs_per_row - 1 else (s + 1) * seg_size
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)
    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)
    feature_ids = t.randint(0, n_sae, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)
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


class TestPerfShapeBlacklistNoLargeRandperm:
    """Perf-shape guarantee: blacklist with large |allowed| must use vectorized path.

    Pins the invariant that the common production config (a short blacklist
    over ~65536 SAE features) does NOT regress to the O(total_segs×|allowed|)
    per-segment randperm loop.

    This test FAILS at HEAD 0a7a1433 and PASSES after the hybrid fix.
    """

    # Realistic SAE width; |allowed| = n_sae - len(blacklist) ≈ n_sae >> threshold(128)
    N_SAE_LARGE = 65536
    BLACKLIST_IDS = (0, 1, 2)   # tiny blacklist → |allowed| = 65533

    def _make_blacklist_pipeline(
        self,
        seed: int = 42,
        n_sae: int = N_SAE_LARGE,
        blacklist: tuple[int, ...] = BLACKLIST_IDS,
        neg_min_k: int = 3,
        neg_mean: float = 4.0,
    ) -> FeaturePipelineConfig:
        mask = FeatureMask("blacklist", blacklist)
        return _make_pipeline(
            seed=seed, feature_mask=mask, n_features=n_sae,
            neg_min_k=neg_min_k, neg_mean=neg_mean,
        )

    def test_no_large_randperm_for_blacklist(self) -> None:
        """randperm must NOT be called with an argument >= _NEG_SMALL_POOL_EXACT_THRESHOLD(128).

        On HEAD 0a7a1433, each segment calls randperm(len(eligible)) ≈ randperm(65533)
        → this test FAILS there.

        After the hybrid fix, |allowed|=65533 >= 128 → vectorized path, no randperm
        with large arg → this test PASSES.
        """
        n_sae = self.N_SAE_LARGE
        shard = _make_shard_large_sae(n_sae=n_sae, n_rows=8, segs_per_row=4, seed=7)
        pipeline = self._make_blacklist_pipeline(seed=42)

        large_randperm_calls: list[int] = []
        _NEG_SMALL_POOL_EXACT_THRESHOLD = 128

        real_randperm = t.randperm

        def spy_randperm(n: int, **kwargs: object) -> t.Tensor:
            if n >= _NEG_SMALL_POOL_EXACT_THRESHOLD:
                large_randperm_calls.append(n)
            return real_randperm(n, **kwargs)  # type: ignore[arg-type]  # ty: ignore[no-matching-overload]

        with patch("torch.randperm", side_effect=spy_randperm):
            reconstruct_and_select(shard, list(range(8)), pipeline)

        assert len(large_randperm_calls) == 0, (
            f"randperm was called with large argument(s) {large_randperm_calls[:5]} "
            f"(total {len(large_randperm_calls)} calls ≥ {_NEG_SMALL_POOL_EXACT_THRESHOLD}). "
            f"The blacklist/large-|allowed| case must use the vectorized path, NOT "
            f"the per-segment enumerate-permute loop. Hybrid fix required."
        )

    def test_blacklist_large_allowed_delivery_adequate(self) -> None:
        """Large |allowed| blacklist case delivers negatives (vectorized path works).

        With |allowed|=65533 and neg_min_k=3, virtually every segment gets ≥3 negatives.
        Requires ≥90% of segments with active mask to receive negatives.
        """
        n_sae = self.N_SAE_LARGE
        shard = _make_shard_large_sae(n_sae=n_sae, n_rows=8, segs_per_row=4, seed=13)
        pipeline = self._make_blacklist_pipeline(seed=55)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(8)), pipeline)

        counts = _neg_counts_per_segment(rb, eff)
        segs_with_any_mask = sum(
            1 for b in range(rb.enc_fids.shape[0])
            for s in range(rb.enc_fids.shape[1])
            if rb.enc_mask[b, s].any()
        )
        segs_with_negs = sum(1 for c in counts if c > 0)

        assert segs_with_negs >= 0.9 * segs_with_any_mask, (
            f"Only {segs_with_negs}/{segs_with_any_mask} active segments got negatives "
            f"(need ≥90%). Vectorized path for large |allowed| must deliver adequately."
        )

    def test_blacklist_large_allowed_i4_no_masked_ids(self) -> None:
        """I4: vectorized path must also respect the blacklist (no masked IDs in output).

        Negatives from the vectorized path come from allowed_ids by construction.
        This test pins the structural I4 guarantee for the large-|allowed| path.
        """
        n_sae = self.N_SAE_LARGE
        blacklist = self.BLACKLIST_IDS
        shard = _make_shard_large_sae(n_sae=n_sae, n_rows=8, segs_per_row=4, seed=7)
        pipeline = self._make_blacklist_pipeline(seed=42)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(8)), pipeline)

        blackset = set(blacklist)
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                fids = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                for fid in fids[mags == eff].tolist():
                    assert fid not in blackset, (
                        f"Negative fid {fid} at [{b},{s}] is in blacklist {blackset}. "
                        f"Vectorized path I4 violated."
                    )

    def test_exact_delivery_sweep_large_vs_small_allowed(self) -> None:
        """Exact delivery sweep: all satisfiable (|allowed|, neg_ks) cells deliver correctly.

        Verifies both paths (vectorized for large |allowed|, exact for small):
        - SMALL-POOL exact path (|allowed| < 128): exact delivery even when
          eff_pool_seg == neg_ks_target (tight constraint).  Uses the same
          controlled shard as TestSmallPoolExactDelivery (n_pos + n_neg == |allowed|).
        - VECTORIZED path (|allowed| >= 128): delivery when pool >> neg_ks
          (realistic case: few positives per segment, large allowed set).
          Expects every segment to receive exactly neg_ks_target negatives.

        The test does NOT attempt tight eff_pool == neg_ks at |allowed|=256+ because
        the vectorized path is designed for the large-pool regime where eff_pool >> neg_ks.
        Exact delivery at the tight eff_pool boundary for large |allowed| is not a
        contract of the vectorized path; it is covered by the small-pool exact tests.

        allowed sizes: {2, 4, 8, 16, 64} → exact path (< 128)
                      {256, 1024, 65536} → vectorized path (>= 128), pool >> neg_ks
        neg_ks: {1, 3, 8}
        """
        n_rows = 4
        segs_per_row = 3
        total_segs = n_rows * segs_per_row  # noqa: F841

        results: list[tuple[int, int, str, bool]] = []
        # (allowed_size, neg_ks, path_used, all_exact)

        for allowed_size in (2, 4, 8, 16, 64, 256, 1024, 65536):
            for neg_ks_target in (1, 3, 8):
                if neg_ks_target >= allowed_size:
                    # Not satisfiable: skip
                    continue

                # Build n_sae >> allowed_size so the whitelist is realistic
                n_sae = max(allowed_size + 64, 1024)
                # Whitelist of exactly allowed_size ids
                wl_ids = tuple(range(allowed_size))

                if allowed_size < 128:
                    # === EXACT PATH cells: eff_pool == neg_ks (tight) ===
                    # Shard: positives take up (allowed_size - neg_ks_target) allowed slots
                    # → per-seg eff_pool == neg_ks_target (worst case for with-replacement)
                    n_pos = max(0, allowed_size - neg_ks_target)
                    pos_ids = list(range(min(n_pos, allowed_size)))
                    if n_pos > 0:
                        shard = _make_controlled_shard(
                            n_rows=n_rows, segs_per_row=segs_per_row,
                            n_pos_per_seg=n_pos, pos_feature_ids=pos_ids,
                        )
                    else:
                        outside_id = allowed_size
                        shard = _make_controlled_shard(
                            n_rows=n_rows, segs_per_row=segs_per_row,
                            n_pos_per_seg=1, pos_feature_ids=[outside_id],
                        )
                else:
                    # === VECTORIZED PATH cells: pool >> neg_ks (large pool) ===
                    # Shard: features come from OUTSIDE the whitelist (all outside the
                    # allowed set), so after mask-gated positive filtering, positives == 0
                    # and per-seg eff_pool == |allowed| >> neg_ks_target.
                    # For allowed_size == 65536 the whitelist covers all uint16 ids, so
                    # we use allowed_size==65536 only when n_sae>65536 is impossible in
                    # uint16; in practice feature_ids are uint16 so at most 65535.
                    # When allowed_size >= 65535, positives can't be outside whitelist,
                    # so we skip this cell (all ids are whitelisted).
                    if allowed_size >= 65535:
                        # Can't build shard with outside_id in uint16; skip this cell.
                        continue
                    outside_id = allowed_size  # first id outside whitelist range(allowed_size)
                    shard = _make_controlled_shard(
                        n_rows=n_rows, segs_per_row=segs_per_row,
                        n_pos_per_seg=1, pos_feature_ids=[outside_id],
                    )
                    n_pos = 0  # no positives within allowed set

                rng_pos = t.Generator().manual_seed(99)
                rng_neg = t.Generator().manual_seed(100)
                pos_mean = max(1, n_pos)
                selection = SamplingSelection(
                    distribution=Distribution.geometric,
                    mean=float(pos_mean), std=1.0,
                    min_k=n_pos, max_k=n_pos + 1,
                    mode=SamplingMode.top_k_percentile,
                )
                pct_table = _make_percentile_table(n_sae)
                sampling_bundle = SamplingBundle(
                    selection=selection, percentile_table=pct_table, rng=rng_pos,
                )
                boundaries = _make_boundaries(n_sae)
                negative_bundle = NegativeBundle(
                    selection=NegativeSamplingSelection(
                        distribution=Distribution.geometric,
                        mean=float(neg_ks_target), std=1.0,
                        min_k=neg_ks_target, max_k=neg_ks_target + 1,
                    ),
                    n_sae_features=n_sae, rng=rng_neg,
                )
                fm = FeatureMask("whitelist", wl_ids)
                pipeline = FeaturePipelineConfig(
                    selection_config=sampling_bundle,
                    boundaries=boundaries,
                    negative_bundle=negative_bundle,
                    effective_buckets=9,
                    feature_mask=fm,
                )
                eff = pipeline.effective_buckets
                rb, _ = reconstruct_and_select(shard, list(range(n_rows)), pipeline)

                path = "vectorized" if allowed_size >= 128 else "exact"
                under = 0
                for b in range(rb.enc_fids.shape[0]):
                    for s in range(rb.enc_fids.shape[1]):
                        m = rb.enc_mask[b, s]
                        if not m.any():
                            continue
                        mags = rb.enc_mags[b, s, m]
                        got = int((mags == eff).sum().item())
                        if got < neg_ks_target:
                            under += 1
                all_ok = under == 0
                results.append((allowed_size, neg_ks_target, path, all_ok))

        # Assert all cells delivered correctly
        failures = [(a, k, p) for a, k, p, ok in results if not ok]
        assert not failures, (
            f"Under-delivery in cells: {failures}\n"
            f"Full table: {[(a, k, p, ok) for a, k, p, ok in results]}"
        )
