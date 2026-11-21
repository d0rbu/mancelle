"""Review-finding fixes for fused negative sampling in reconstruct_and_select.

These tests pin the three fused-path fixes from the #637 review:

  #1  Exact delivery, no silent under-delivery.  When neg_ks is clamped to the
      eligible pool (allowed universe minus ALL active features), every segment
      must receive EXACTLY that many distinct negatives — on both the masked
      vectorized path (|allowed| >= 128) and the no-mask path.  A residual
      shortfall is an invariant violation and must raise, never be packed
      silently into neg_valid.

  #2  Flat membership mask.  The collision check between candidate negatives and
      a segment's active features must use a flat [total_segs, n_sae] membership
      lookup (O(n_seg * n_cand)), not a [total_segs, n_cand, active_width] 3-D
      broadcast, and must still exclude EVERY active feature (not just selected
      positives) on dense segments.

  #3  Out-of-range active ids fail loud with a diagnostic message.  An active
      feature id >= n_sae_features (shard/config width mismatch, or an OOB id a
      blacklist keeps) must raise a ValueError that names the offending id — not
      a raw IndexError, and not silent tolerance on the no-mask path.

Pre-fix (PR head ddaab41c):
  * #1 tests RED — the vectorized/no-mask paths under-deliver on tight pools.
  * #3 tests RED — masked path raises a raw IndexError; no-mask path does not
    raise at all.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    _build_active_membership,
    _exact_inactive_negatives,
    reconstruct_and_select,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
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
# Builders
# ---------------------------------------------------------------------------


def _make_boundaries(n_features: int, n_buckets: int = 10) -> t.Tensor:
    n_boundaries = n_buckets - 1
    b = t.linspace(1 / n_buckets, 1.0 - 1 / n_buckets, n_boundaries, dtype=t.float64)
    return b.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_percentile_table(n_features: int) -> t.Tensor:
    pcts = t.linspace(0.01, 0.99, 99, dtype=t.float64)
    return pcts.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_segment_shard(
    per_segment_fids: list[list[int]],
    *,
    context_len: int = 32,
    pct_value: int = 200,
) -> PackedRowShard:
    """One row per segment list; each segment holds exactly its given fids."""
    n_seg = len(per_segment_fids)
    sizes = [len(s) for s in per_segment_fids]
    flat = [fid for seg in per_segment_fids for fid in seg]
    total_feats = len(flat)

    tokens = t.ones(n_seg, context_len + 1, dtype=t.uint16)
    row_seg_offsets = t.arange(0, n_seg + 1, dtype=t.int32)
    seg_token_start = t.zeros(n_seg, dtype=t.uint16)
    seg_token_end = t.full((n_seg,), context_len, dtype=t.uint16)
    seg_doc_id = t.zeros(n_seg, dtype=t.uint8)

    seg_feat_offsets = t.zeros(n_seg + 1, dtype=t.int32)
    for i, sz in enumerate(sizes):
        seg_feat_offsets[i + 1] = seg_feat_offsets[i] + sz

    feature_ids = t.tensor(flat, dtype=t.uint16) if total_feats else t.zeros(0, dtype=t.uint16)
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


def _make_pipeline(
    *,
    n_sae: int,
    neg_min_k: int,
    neg_max_k: int,
    feature_mask: FeatureMask | None,
    pos_max_k: int = 2,
    seed: int = 17,
    table_width: int | None = None,
) -> FeaturePipelineConfig:
    """Build a fused pipeline.

    ``table_width`` sizes the per-feature boundaries/percentile tables used for
    positive selection; when it exceeds ``n_sae`` (the negative universe width)
    the shard may legitimately carry positive ids the negative sampler's
    ``n_sae_features`` does not cover — the misconfiguration finding #3 targets.
    """
    tables_w = table_width if table_width is not None else n_sae
    n_buckets = 10
    effective_buckets = n_buckets - 1
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=float(max(pos_max_k, 1)), std=1.0,
        min_k=1, max_k=pos_max_k,
        mode=SamplingMode.top_k_percentile,
    )
    sampling_bundle = SamplingBundle(
        selection=selection,
        percentile_table=_make_percentile_table(tables_w),
        rng=t.Generator().manual_seed(seed),
    )
    negative_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=float(neg_min_k), std=1.0,
            min_k=neg_min_k, max_k=neg_max_k,
        ),
        n_sae_features=n_sae,
        rng=t.Generator().manual_seed(seed + 1000),
        feature_mask=feature_mask,
    )
    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=_make_boundaries(tables_w, effective_buckets),
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        feature_mask=feature_mask,
    )


def _neg_counts_and_ids(rb, eff_bucket: int) -> list[list[int]]:
    """Per-segment lists of negative feature ids (bucket == eff_bucket)."""
    out: list[list[int]] = []
    for b in range(rb.enc_fids.shape[0]):
        for s in range(rb.enc_fids.shape[1]):
            m = rb.enc_mask[b, s]
            if not m.any():
                out.append([])
                continue
            fids = rb.enc_fids[b, s, m]
            mags = rb.enc_mags[b, s, m]
            out.append(fids[mags == eff_bucket].tolist())
    return out


# ===========================================================================
# #1 — exact delivery on the masked vectorized path (|allowed| >= 128)
# ===========================================================================


class TestVectorizedExactDelivery:
    """Masked vectorized branch must deliver exactly the clamped neg_ks."""

    def test_tight_pool_delivers_exactly_eff_pool(self) -> None:
        """200-id whitelist (vectorized), 150 active/seg, neg_ks clamps to 50.

        Every active segment must deliver EXACTLY 50 distinct negatives, all
        drawn from the 50 eligible ids {150..199}.  Pre-fix the vectorized
        with-replacement draw under-delivers (~28-40) silently.
        """
        n_sae = 256
        allowed = tuple(range(200))          # |allowed| = 200 >= 128 -> vectorized
        active = list(range(150))            # 150 active whitelisted per segment
        eligible = set(range(150, 200))      # the 50 legal negatives
        shard = _make_segment_shard([active, active, active, active])
        mask = FeatureMask(FeatureMaskMode.whitelist, allowed)
        # neg_min_k = 60 > eff_pool(50) -> neg_ks clamps to exactly 50 everywhere.
        pipeline = _make_pipeline(
            n_sae=n_sae, neg_min_k=60, neg_max_k=61, feature_mask=mask,
        )
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, [0, 1, 2, 3], pipeline)

        neg_ids = _neg_counts_and_ids(rb, eff)
        active_segs = [n for n in neg_ids if n]  # segments with any output
        assert active_segs, "expected active segments"
        for negs in active_segs:
            assert len(negs) == 50, (
                f"expected exactly 50 negatives (eff_pool), got {len(negs)}"
            )
            assert len(set(negs)) == 50, "negatives must be distinct"
            assert set(negs) == eligible, (
                "negatives must be exactly the 50 eligible ids {150..199}"
            )


# ===========================================================================
# #1 — exact delivery on the no-mask path
# ===========================================================================


class TestNoMaskExactDelivery:
    """No-mask branch must also deliver exactly the clamped neg_ks."""

    def test_dense_no_mask_delivers_exactly_eff_pool(self) -> None:
        """n_sae=300, 200 active/seg, neg_ks clamps to 100.

        Every active segment must deliver EXACTLY 100 distinct negatives, none
        of them active.  Pre-fix the fixed 3x oversample under-delivers (~53-61).
        """
        n_sae = 300
        active = list(range(200))
        shard = _make_segment_shard([active, active, active])
        pipeline = _make_pipeline(
            n_sae=n_sae, neg_min_k=150, neg_max_k=151, feature_mask=None,
        )
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, [0, 1, 2], pipeline)

        active_set = set(active)
        neg_ids = _neg_counts_and_ids(rb, eff)
        active_segs = [n for n in neg_ids if n]
        assert active_segs, "expected active segments"
        for negs in active_segs:
            assert len(negs) == 100, (
                f"expected exactly 100 negatives (eff_pool), got {len(negs)}"
            )
            assert len(set(negs)) == 100, "negatives must be distinct"
            assert active_set.isdisjoint(negs), "negatives must exclude all active ids"


# ===========================================================================
# #1 — fail-loud postcondition: a residual shortfall must raise
# ===========================================================================


class TestDeliveryShortfallRaises:
    """If delivery falls short of the clamped neg_ks, raise (never silent)."""

    def test_shortfall_after_repair_raises(self) -> None:
        """With the exact repair sabotaged to a no-op, the vectorized path's
        residual under-delivery must trip the fail-loud postcondition.

        This isolates the guard: patch the exact sampler to return nothing so no
        segment can be repaired, then a tight-pool vectorized run must raise
        rather than silently emit fewer negatives than neg_ks.
        """
        n_sae = 256
        allowed = tuple(range(200))
        active = list(range(150))
        shard = _make_segment_shard([active, active])
        mask = FeatureMask(FeatureMaskMode.whitelist, allowed)
        pipeline = _make_pipeline(
            n_sae=n_sae, neg_min_k=60, neg_max_k=61, feature_mask=mask,
        )

        def _empty(*_a: object, **_k: object) -> t.Tensor:
            return t.zeros(0, dtype=t.int64)

        with patch(
            "steering.conditional.encoder_decoder.dataloader._exact_inactive_negatives",
            side_effect=_empty,
        ):
            with pytest.raises(ValueError, match="neg"):
                reconstruct_and_select(shard, [0, 1], pipeline)


# ===========================================================================
# #3 — out-of-range active id fails loud with a diagnostic message
# ===========================================================================


class TestOutOfRangeActiveIdDiagnostic:
    """An active id >= n_sae_features must raise a diagnostic ValueError."""

    def test_blacklist_oob_active_id_raises_valueerror_not_indexerror(self) -> None:
        """Blacklist keeps an OOB id (40) that the neg universe (n_sae=32) omits.

        The per-feature tables are width 64 so positive discretization succeeds;
        only the negative-sampling exclusion path sees the OOB id.

        Pre-fix: raw ``IndexError`` from ``_allowed[active_exclusions]``.
        Post-fix: a ``ValueError`` naming the offending id 40.
        """
        shard = _make_segment_shard([[40]])
        mask = FeatureMask(FeatureMaskMode.blacklist, (0,))
        pipeline = _make_pipeline(
            n_sae=32, neg_min_k=1, neg_max_k=2, feature_mask=mask, table_width=64,
        )
        with pytest.raises(ValueError, match="40"):
            reconstruct_and_select(shard, [0], pipeline)

    def test_no_mask_oob_active_id_raises_valueerror(self) -> None:
        """No-mask path with an OOB active id (40) the neg universe (n_sae=32) omits.

        Pre-fix: silently tolerated (no raise) — the ``==`` broadcast never
        matches an in-range candidate against 40.  Post-fix: a ``ValueError``
        naming the offending id (also makes the flat membership mask safe).
        """
        shard = _make_segment_shard([[40]])
        pipeline = _make_pipeline(
            n_sae=32, neg_min_k=1, neg_max_k=2, feature_mask=None, table_width=64,
        )
        with pytest.raises(ValueError, match="40"):
            reconstruct_and_select(shard, [0], pipeline)


# ===========================================================================
# #2 — dense exclusion correctness (flat membership mask)
# ===========================================================================


class TestFlatMembershipHelper:
    """`_build_active_membership` is a 2-D mask equal to a brute-force reference."""

    def test_shape_is_flat_not_3d(self) -> None:
        all_fids = t.tensor([0, 3, 7, 3, 9], dtype=t.int64)  # seg0={0,3,7}, seg1={3,9}
        sizes = t.tensor([3, 2], dtype=t.long)
        membership = _build_active_membership(all_fids, sizes, total_segs=2, n_sae_features=16)
        assert membership.shape == (2, 16)  # O(n_seg * n_sae), never 3-D
        assert membership.dtype == t.bool

    def test_membership_matches_bruteforce(self) -> None:
        per_seg = [[0, 3, 7], [3, 9], []]
        all_fids = t.tensor([f for seg in per_seg for f in seg], dtype=t.int64)
        sizes = t.tensor([len(s) for s in per_seg], dtype=t.long)
        n_sae = 16
        membership = _build_active_membership(all_fids, sizes, total_segs=3, n_sae_features=n_sae)
        for seg, ids in enumerate(per_seg):
            for fid in range(n_sae):
                assert bool(membership[seg, fid].item()) == (fid in ids), (
                    f"membership[{seg},{fid}] wrong"
                )

    def test_empty_is_all_false(self) -> None:
        membership = _build_active_membership(
            t.zeros(0, dtype=t.int64), t.zeros(2, dtype=t.long), total_segs=2, n_sae_features=8,
        )
        assert membership.shape == (2, 8)
        assert not membership.any()


class TestExactInactiveNegativesHelper:
    """`_exact_inactive_negatives` delivers exactly, disjoint, within allowed."""

    def test_delivers_exactly_k_distinct_inactive(self) -> None:
        n_sae = 20
        active_row = t.zeros(n_sae, dtype=t.bool)
        active_row[[1, 2, 3]] = True
        rng = t.Generator().manual_seed(0)
        got = _exact_inactive_negatives(active_row, None, k=5, rng=rng)
        assert got.numel() == 5
        assert len(set(got.tolist())) == 5, "distinct"
        assert {1, 2, 3}.isdisjoint(got.tolist()), "inactive only"

    def test_respects_allowed_mask(self) -> None:
        n_sae = 20
        active_row = t.zeros(n_sae, dtype=t.bool)
        active_row[0] = True
        allowed = t.zeros(n_sae, dtype=t.bool)
        allowed[[0, 5, 6, 7]] = True  # allowed minus active = {5,6,7}
        rng = t.Generator().manual_seed(0)
        got = _exact_inactive_negatives(active_row, allowed, k=10, rng=rng)
        assert set(got.tolist()) == {5, 6, 7}, "clamped to eligible, allowed-only"

    def test_k_zero_and_empty_pool_return_empty(self) -> None:
        n_sae = 8
        active_row = t.zeros(n_sae, dtype=t.bool)
        rng = t.Generator().manual_seed(0)
        assert _exact_inactive_negatives(active_row, None, k=0, rng=rng).numel() == 0
        active_all = t.ones(n_sae, dtype=t.bool)
        assert _exact_inactive_negatives(active_all, None, k=3, rng=rng).numel() == 0

    def test_deterministic_for_same_seed(self) -> None:
        n_sae = 64
        active_row = t.zeros(n_sae, dtype=t.bool)
        active_row[[10, 20]] = True
        a = _exact_inactive_negatives(active_row, None, 8, t.Generator().manual_seed(3))
        b = _exact_inactive_negatives(active_row, None, 8, t.Generator().manual_seed(3))
        assert t.equal(a, b)


class TestDenseExclusionCorrectness:
    """Negatives exclude EVERY active feature on dense segments (both paths)."""

    def test_masked_dense_negatives_exclude_all_active(self) -> None:
        n_sae = 256
        allowed = tuple(range(200))
        active = list(range(150))
        shard = _make_segment_shard([active, active])
        mask = FeatureMask(FeatureMaskMode.whitelist, allowed)
        pipeline = _make_pipeline(
            n_sae=n_sae, neg_min_k=10, neg_max_k=11, feature_mask=mask,
        )
        eff = pipeline.effective_buckets
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)
        active_set = set(active)
        for negs in _neg_counts_and_ids(rb, eff):
            if not negs:
                continue
            assert active_set.isdisjoint(negs), "negatives must exclude all active ids"
            assert set(negs).issubset(set(range(150, 200)))

    def test_no_mask_dense_negatives_exclude_all_active(self) -> None:
        n_sae = 300
        active = list(range(200))
        shard = _make_segment_shard([active, active])
        pipeline = _make_pipeline(
            n_sae=n_sae, neg_min_k=10, neg_max_k=11, feature_mask=None,
        )
        eff = pipeline.effective_buckets
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)
        active_set = set(active)
        for negs in _neg_counts_and_ids(rb, eff):
            if not negs:
                continue
            assert active_set.isdisjoint(negs), "negatives must exclude all active ids"


# ===========================================================================
# Empty active set (total_feats == 0) with max_neg > 0
#   Segments with zero active features still request negatives; they must be
#   drawn from the full/allowed universe (all-False membership, no active to
#   exclude). Exercises the total_feats==0 guard skip + all-False membership.
# ===========================================================================


class TestEmptyActiveFeatures:
    def test_no_mask_empty_segment_delivers_from_full_universe(self) -> None:
        shard = _make_segment_shard([[]])  # one segment, zero active features
        pipeline = _make_pipeline(n_sae=64, neg_min_k=5, neg_max_k=6, feature_mask=None)
        eff = pipeline.effective_buckets
        rb, _ = reconstruct_and_select(shard, [0], pipeline)
        negs = _neg_counts_and_ids(rb, eff)
        assert [len(x) for x in negs] == [5]
        assert all(0 <= v < 64 for x in negs for v in x)

    def test_masked_all_features_filtered_delivers_from_allowed(self) -> None:
        # Shard features are all outside the whitelist -> 0 active after filter.
        shard = _make_segment_shard([[60, 61, 62]])
        mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(50)))
        pipeline = _make_pipeline(
            n_sae=64, neg_min_k=5, neg_max_k=6, feature_mask=mask, table_width=64,
        )
        eff = pipeline.effective_buckets
        rb, _ = reconstruct_and_select(shard, [0], pipeline)
        negs = _neg_counts_and_ids(rb, eff)
        assert [len(x) for x in negs] == [5]
        assert all(0 <= v < 50 for x in negs for v in x), "negatives must stay in whitelist"

    def test_mixed_empty_and_populated_segments(self) -> None:
        # Two rows: row 0 is an empty segment (no active features), row 1 holds
        # active ids {1,2,3}. Exclusion is PER SEGMENT — the empty segment may
        # legitimately draw any id (including 1/2/3, which are inactive there);
        # only the populated segment must exclude its own active ids.
        shard = _make_segment_shard([[], [1, 2, 3]])
        pipeline = _make_pipeline(n_sae=64, neg_min_k=4, neg_max_k=5, feature_mask=None)
        eff = pipeline.effective_buckets
        rb, _ = reconstruct_and_select(shard, [0, 1], pipeline)
        negs = _neg_counts_and_ids(rb, eff)
        assert all(c >= 4 for c in [len(x) for x in negs])  # both deliver in full
        assert {1, 2, 3}.isdisjoint(negs[1]), (
            f"populated segment (row 1) must exclude its active ids {{1,2,3}}: {negs[1]}"
        )


# ===========================================================================
# Repair-path determinism — the vectorized shortfall repair (randperm) must be
# deterministic for a fixed seed on BOTH masked and no-mask paths.
# ===========================================================================


class TestRepairDeterminism:
    def _run_twice(self, n_sae, active, neg_min_k, mask):
        shard = _make_segment_shard([active, active])
        a = reconstruct_and_select(
            shard, [0, 1],
            _make_pipeline(n_sae=n_sae, neg_min_k=neg_min_k, neg_max_k=neg_min_k + 1,
                           feature_mask=mask, seed=123),
        )[0]
        b = reconstruct_and_select(
            shard, [0, 1],
            _make_pipeline(n_sae=n_sae, neg_min_k=neg_min_k, neg_max_k=neg_min_k + 1,
                           feature_mask=mask, seed=123),
        )[0]
        return a, b

    def test_masked_vectorized_repair_deterministic(self) -> None:
        # |allowed|=200 >= 128 (vectorized), 150 active -> tight pool -> repair fires.
        mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(200)))
        a, b = self._run_twice(256, list(range(150)), 60, mask)
        assert t.equal(a.enc_fids, b.enc_fids)
        assert t.equal(a.enc_mask, b.enc_mask)

    def test_no_mask_repair_deterministic(self) -> None:
        a, b = self._run_twice(300, list(range(200)), 150, None)
        assert t.equal(a.enc_fids, b.enc_fids)
        assert t.equal(a.enc_mask, b.enc_mask)
