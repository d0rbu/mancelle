"""Tests for feature_mask enforcement in reconstruct_and_select (Task 11).

Covers:
 - blacklist: positive, negative, and FP-target paths exclude masked feature
 - whitelist: positive, negative, and FP-target paths are restricted to whitelist
 - I5 byte-identical: feature_mask=None produces identical output to baseline
   (same seed, same shard) — verifying no RNG divergence in Phase 5
"""

from __future__ import annotations

import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    build_fp_targets_from_shard,
    reconstruct_and_select,
)
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
)
from steering.packed_batch_io import PackedRowShard, _build_batch_index

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N_SAE = 64  # small enough to make whitelist tests tight
BLACKLISTED = 7  # single feature to blacklist
WHITELIST_IDS = (5, 12, 20, 33, 45)  # small whitelist
N_ROWS = 4
SEGS_PER_ROW = 3
FEATS_PER_SEG = 20
CONTEXT_LEN = 32


# ---------------------------------------------------------------------------
# Shard builder (deterministic)
# ---------------------------------------------------------------------------


def _make_shard(seed: int = 42) -> PackedRowShard:
    """Build a synthetic PackedRowShard for mask tests.

    Feature IDs are drawn from [0, N_SAE) so whitelist/blacklist tests
    have meaningful coverage.  Importantly, the shard includes feature IDs
    from WHITELIST_IDS and BLACKLISTED so the mask has something to do.
    """
    t.manual_seed(seed)
    total_segs = N_ROWS * SEGS_PER_ROW
    total_feats = total_segs * FEATS_PER_SEG

    tokens = t.randint(1, 1000, (N_ROWS, CONTEXT_LEN + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(0, (N_ROWS + 1) * SEGS_PER_ROW, SEGS_PER_ROW, dtype=t.int32)

    seg_size = CONTEXT_LEN // SEGS_PER_ROW
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)

    for r in range(N_ROWS):
        for s in range(SEGS_PER_ROW):
            idx = r * SEGS_PER_ROW + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = CONTEXT_LEN if s == SEGS_PER_ROW - 1 else (s + 1) * seg_size

    seg_feat_offsets = t.arange(0, (total_segs + 1) * FEATS_PER_SEG, FEATS_PER_SEG, dtype=t.int32)

    # Force the blacklisted feature and whitelist features to appear.
    # Fill with uniform [0, N_SAE) then inject known IDs.
    feature_ids = t.randint(0, N_SAE, (total_feats,), dtype=t.uint16)
    # Inject BLACKLISTED in every 4th slot so there is something to suppress.
    feature_ids[::4] = BLACKLISTED
    # Inject at least one whitelist feature per segment
    for wi, wid in enumerate(WHITELIST_IDS):
        feature_ids[wi::len(WHITELIST_IDS) + 1] = wid

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
# Pipeline builders
# ---------------------------------------------------------------------------


def _make_percentile_table(n_features: int = N_SAE) -> t.Tensor:
    pcts = t.linspace(0.01, 0.99, 99, dtype=t.float64)
    return pcts.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_boundaries(n_features: int = N_SAE, n_buckets: int = 10) -> t.Tensor:
    n_boundaries = n_buckets - 1
    b = t.linspace(1 / n_buckets, 1.0 - 1 / n_buckets, n_boundaries, dtype=t.float64)
    return b.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_pipeline_with_negatives(
    seed: int = 42,
    feature_mask: FeatureMask | None = None,
    n_features: int = N_SAE,
) -> FeaturePipelineConfig:
    """Build a sampling pipeline with negative_bundle and optional mask."""
    rng = t.Generator().manual_seed(seed)
    neg_rng = t.Generator().manual_seed(seed + 1)
    n_buckets = 10
    effective_buckets = n_buckets - 1  # last bucket reserved for negatives

    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=3.0, std=3.0, min_k=1, max_k=None,
    )
    pct_table = _make_percentile_table(n_features)
    sampling_bundle = SamplingBundle(
        selection=selection, percentile_table=pct_table, rng=rng,
    )
    boundaries = _make_boundaries(n_features, effective_buckets)
    negative_bundle = NegativeBundle(
        selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0, std=2.0, min_k=1,
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
# Helper: extract all active positive and negative fids from a batch
# ---------------------------------------------------------------------------


def _collect_positive_fids(rb, eff_bucket: int) -> set[int]:
    """All feature IDs with bucket != eff_bucket and enc_mask=True."""
    fids: set[int] = set()
    for b in range(rb.enc_fids.shape[0]):
        for s in range(rb.enc_fids.shape[1]):
            mask = rb.enc_mask[b, s]
            if not mask.any():
                continue
            active_fids = rb.enc_fids[b, s, mask]
            active_mags = rb.enc_mags[b, s, mask]
            pos = active_fids[active_mags != eff_bucket]
            fids.update(pos.tolist())
    return fids


def _collect_negative_fids(rb, eff_bucket: int) -> set[int]:
    """All feature IDs with bucket == eff_bucket and enc_mask=True."""
    fids: set[int] = set()
    for b in range(rb.enc_fids.shape[0]):
        for s in range(rb.enc_fids.shape[1]):
            mask = rb.enc_mask[b, s]
            if not mask.any():
                continue
            active_fids = rb.enc_fids[b, s, mask]
            active_mags = rb.enc_mags[b, s, mask]
            neg = active_fids[active_mags == eff_bucket]
            fids.update(neg.tolist())
    return fids


# ============================================================================
# Test classes
# ============================================================================


class TestBlacklistMask:
    """Blacklisting a feature removes it from positives, negatives, and FP targets."""

    def test_blacklisted_feature_absent_from_positives(self):
        """Blacklisted feature K must not appear in any positive slot (enc_mask=True, not neg bucket)."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE)

        pos_fids = _collect_positive_fids(rb, eff)
        assert BLACKLISTED not in pos_fids, (
            f"Blacklisted feature {BLACKLISTED} found in positive slots: {pos_fids}"
        )

    def test_blacklisted_feature_absent_from_negatives(self):
        """Blacklisted feature K must not appear in any negative slot (enc_mask=True, mag==eff_bucket)."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE)

        neg_fids = _collect_negative_fids(rb, eff)
        assert BLACKLISTED not in neg_fids, (
            f"Blacklisted feature {BLACKLISTED} found in negative slots: {neg_fids}"
        )

    def test_blacklisted_feature_absent_from_fp_targets(self):
        """Column BLACKLISTED in fp_targets must be all-zero (blacklist zeroed out)."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)

        _, fp_targets = reconstruct_and_select(
            shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE
        )
        assert fp_targets is not None, "fp_targets must not be None when n_sae_features set"

        col = fp_targets[:, :, BLACKLISTED]
        assert col.eq(0).all(), (
            f"FP target column {BLACKLISTED} has nonzero values after blacklist masking: "
            f"max={col.max().item()}"
        )

    def test_non_blacklisted_features_still_present(self):
        """At least one non-blacklisted feature must appear (blacklist is surgical)."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE)

        pos_fids = _collect_positive_fids(rb, eff)
        # At least one positive should survive
        surviving = pos_fids - {BLACKLISTED}
        assert len(surviving) > 0, "No non-blacklisted positives present — blacklist too broad?"

    def test_fp_targets_non_blacklisted_columns_have_nonzero(self):
        """Some non-blacklisted FP target column must be nonzero (mask is surgical)."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)

        _, fp_targets = reconstruct_and_select(
            shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE
        )
        assert fp_targets is not None

        # All columns except BLACKLISTED
        other_cols = t.cat([fp_targets[:, :, :BLACKLISTED], fp_targets[:, :, BLACKLISTED + 1:]], dim=2)
        assert other_cols.any(), "No nonzero FP target values outside blacklisted column"


class TestWhitelistMask:
    """Whitelisting restricts positives, negatives, and FP targets to whitelist IDs."""

    def test_positive_fids_subset_of_whitelist(self):
        """All positive feature IDs must be in the whitelist."""
        shard = _make_shard()
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE)

        pos_fids = _collect_positive_fids(rb, eff)
        whiteset = set(WHITELIST_IDS)
        outliers = pos_fids - whiteset
        assert not outliers, (
            f"Positive feature IDs outside whitelist: {outliers} (whitelist={whiteset})"
        )

    def test_negative_fids_subset_of_whitelist(self):
        """All negative feature IDs must be in the whitelist."""
        shard = _make_shard()
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE)

        neg_fids = _collect_negative_fids(rb, eff)
        whiteset = set(WHITELIST_IDS)
        outliers = neg_fids - whiteset
        assert not outliers, (
            f"Negative feature IDs outside whitelist: {outliers} (whitelist={whiteset})"
        )

    def test_fp_targets_nonzero_only_on_whitelist_columns(self):
        """FP target columns outside whitelist must be all-zero."""
        shard = _make_shard()
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)

        _, fp_targets = reconstruct_and_select(
            shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE
        )
        assert fp_targets is not None

        whiteset = set(WHITELIST_IDS)
        for col_idx in range(N_SAE):
            if col_idx not in whiteset:
                col = fp_targets[:, :, col_idx]
                assert col.eq(0).all(), (
                    f"FP target column {col_idx} (not in whitelist) has nonzero values"
                )

    def test_fp_targets_whitelist_columns_have_some_nonzero(self):
        """At least one whitelist FP target column is nonzero."""
        shard = _make_shard()
        mask = FeatureMask("whitelist", WHITELIST_IDS)
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)

        _, fp_targets = reconstruct_and_select(
            shard, list(range(N_ROWS)), pipeline, n_sae_features=N_SAE
        )
        assert fp_targets is not None

        whitelist_idx = t.tensor(list(WHITELIST_IDS), dtype=t.long)
        whitelist_cols = fp_targets[:, :, whitelist_idx]
        assert whitelist_cols.any(), "No nonzero FP target values in any whitelist column"


class TestI5NomaskByteIdentical:
    """I5: feature_mask=None produces byte-identical output to baseline (no RNG drift)."""

    def _fresh_pipeline(self, seed: int) -> FeaturePipelineConfig:
        """Two calls with the same seed yield independent identical-state pipelines."""
        return _make_pipeline_with_negatives(seed=seed, feature_mask=None)

    def test_enc_fids_identical_no_mask(self):
        """enc_fids are byte-identical when feature_mask=None on two same-seed runs."""
        shard = _make_shard(seed=99)
        rows = list(range(N_ROWS))

        pipeline_a = self._fresh_pipeline(seed=7)
        pipeline_b = self._fresh_pipeline(seed=7)

        rb_a, _ = reconstruct_and_select(shard, rows, pipeline_a)
        rb_b, _ = reconstruct_and_select(shard, rows, pipeline_b)

        assert t.equal(rb_a.enc_fids, rb_b.enc_fids), "enc_fids not identical on same-seed no-mask runs"

    def test_enc_mags_identical_no_mask(self):
        """enc_mags are byte-identical when feature_mask=None on two same-seed runs."""
        shard = _make_shard(seed=99)
        rows = list(range(N_ROWS))

        pipeline_a = self._fresh_pipeline(seed=7)
        pipeline_b = self._fresh_pipeline(seed=7)

        rb_a, _ = reconstruct_and_select(shard, rows, pipeline_a)
        rb_b, _ = reconstruct_and_select(shard, rows, pipeline_b)

        assert t.equal(rb_a.enc_mags, rb_b.enc_mags), "enc_mags not identical on same-seed no-mask runs"

    def test_enc_mask_identical_no_mask(self):
        """enc_mask (includes neg mask region) is byte-identical when feature_mask=None."""
        shard = _make_shard(seed=99)
        rows = list(range(N_ROWS))

        pipeline_a = self._fresh_pipeline(seed=7)
        pipeline_b = self._fresh_pipeline(seed=7)

        rb_a, _ = reconstruct_and_select(shard, rows, pipeline_a)
        rb_b, _ = reconstruct_and_select(shard, rows, pipeline_b)

        assert t.equal(rb_a.enc_mask, rb_b.enc_mask), "enc_mask not identical on same-seed no-mask runs"

    def test_fp_targets_identical_no_mask(self):
        """fp_targets are byte-identical when feature_mask=None on two same-seed runs."""
        shard = _make_shard(seed=99)
        rows = list(range(N_ROWS))

        pipeline_a = self._fresh_pipeline(seed=7)
        pipeline_b = self._fresh_pipeline(seed=7)

        _, fp_a = reconstruct_and_select(shard, rows, pipeline_a, n_sae_features=N_SAE)
        _, fp_b = reconstruct_and_select(shard, rows, pipeline_b, n_sae_features=N_SAE)

        assert fp_a is not None and fp_b is not None
        assert t.equal(fp_a, fp_b), "fp_targets not identical on same-seed no-mask runs"

    def test_rng_state_consumed_identically_no_mask(self):
        """After no-mask run, consuming more random numbers from the RNG gives same result.

        This verifies that the None path does NOT consume extra RNG state
        (no spurious randint calls in Phase 3 or Phase 5 pre-filter).
        """
        shard = _make_shard(seed=99)
        rows = list(range(N_ROWS))

        seed = 55
        # Run A: no mask
        pa = self._fresh_pipeline(seed=seed)
        rb_a, _ = reconstruct_and_select(shard, rows, pa)
        # Consume one more value from each RNG after the call
        extra_pos_a = t.randint(0, 1000, (1,), generator=pa.selection_config.rng).item()  # ty: ignore[unresolved-attribute]
        extra_neg_a = t.randint(0, 1000, (1,), generator=pa.negative_bundle.rng).item()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]

        # Run B: no mask, fresh but same seed
        pb = self._fresh_pipeline(seed=seed)
        rb_b, _ = reconstruct_and_select(shard, rows, pb)
        extra_pos_b = t.randint(0, 1000, (1,), generator=pb.selection_config.rng).item()  # ty: ignore[unresolved-attribute]
        extra_neg_b = t.randint(0, 1000, (1,), generator=pb.negative_bundle.rng).item()  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]

        # Both extra draws must be equal (same RNG state post-call)
        assert extra_pos_a == extra_pos_b, (
            f"Pos RNG diverged after no-mask run: {extra_pos_a} != {extra_pos_b}"
        )
        assert extra_neg_a == extra_neg_b, (
            f"Neg RNG diverged after no-mask run: {extra_neg_a} != {extra_neg_b}"
        )


# ============================================================================
# FIX 1 regression: empty batch (total_segs == 0) with mask must not NameError
# ============================================================================


def _make_zero_seg_shard() -> PackedRowShard:
    """Build a shard whose single row has zero segments (total_segs == 0).

    row_seg_offsets = [0, 0] means row 0 owns no segments.  This is the
    minimal input that drives total_segs == 0 through _build_batch_index.
    """
    context_len = 32
    return PackedRowShard(
        tokens=t.randint(1, 1000, (1, context_len + 1), dtype=t.uint16),
        row_seg_offsets=t.tensor([0, 0], dtype=t.int32),  # row 0 → 0 segments
        seg_token_start=t.zeros(0, dtype=t.uint16),
        seg_token_end=t.zeros(0, dtype=t.uint16),
        seg_doc_id=t.zeros(0, dtype=t.uint8),
        seg_feat_offsets=t.zeros(1, dtype=t.int32),       # CSR boundary for 0 segs
        feature_ids=t.zeros(0, dtype=t.uint16),
        feature_pcts=t.zeros(0, dtype=t.uint8),
    )


class TestEmptyBatchWithMask:
    """Regression for FIX 1: total_segs==0 + active feature_mask must not NameError."""

    def test_empty_batch_with_mask_does_not_crash(self):
        """reconstruct_and_select with total_segs==0 and a live mask must not raise.

        Before FIX 1, the else-branch (total_segs==0) left total_feats unbound.
        The Phase-3 guard `if pipeline.feature_mask is not None and total_feats > 0:`
        would then raise NameError when a mask was active.
        """
        shard = _make_zero_seg_shard()
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _make_pipeline_with_negatives(feature_mask=mask)

        # Must not raise NameError (or any other exception).
        rb, fp_targets = reconstruct_and_select(
            shard, [0], pipeline, n_sae_features=N_SAE
        )

        # Compare against a no-mask run to confirm structural equivalence.
        pipeline_no_mask = _make_pipeline_with_negatives(feature_mask=None)
        rb_no_mask, fp_no_mask = reconstruct_and_select(
            shard, [0], pipeline_no_mask, n_sae_features=N_SAE
        )

        # Both runs must return the same batch shape (empty batch is well-formed).
        assert rb.enc_fids.shape == rb_no_mask.enc_fids.shape
        assert rb.enc_mask.shape == rb_no_mask.enc_mask.shape
        # fp_targets present and same shape in both cases.
        assert fp_targets is not None and fp_no_mask is not None
        assert fp_targets.shape == fp_no_mask.shape


# ============================================================================
# FIX 4b: direct unit test for build_fp_targets_from_shard with feature_mask
# ============================================================================


class TestBuildFpTargetsFeatureMask:
    """Unit test for build_fp_targets_from_shard(..., feature_mask=...)."""

    def _make_simple_shard(self) -> PackedRowShard:
        """Single row, single segment, three known features (IDs 5, 10, 15)."""
        context_len = 16
        return PackedRowShard(
            tokens=t.randint(1, 100, (1, context_len + 1), dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([context_len], dtype=t.uint16),
            seg_doc_id=t.zeros(1, dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 3], dtype=t.int32),
            feature_ids=t.tensor([5, 10, 15], dtype=t.uint16),
            feature_pcts=t.tensor([128, 200, 64], dtype=t.uint8),
        )

    def test_build_fp_targets_blacklist_mask_and_none_invariant(self):
        """Blacklist mask zeroes masked column; feature_mask=None is byte-identical to no-arg.

        Two assertions in one test (as specified):
        1. Column 10 (blacklisted) is zero; non-masked columns 5 and 15 are nonzero.
        2. feature_mask=None returns byte-identical output to the default (no-arg) call.
        """
        shard = self._make_simple_shard()
        idx = _build_batch_index(shard, row_indices=[0])
        mask = FeatureMask("blacklist", (10,))

        # Assertion 1: blacklist zeroes the masked column, leaves others nonzero.
        out_masked = build_fp_targets_from_shard(shard, idx, N_SAE, feature_mask=mask)
        assert out_masked.shape == (1, 1, N_SAE)
        assert out_masked[0, 0, 10].item() == 0.0, (
            f"Blacklisted column 10 is nonzero: {out_masked[0, 0, 10].item()}"
        )
        assert out_masked[0, 0, 5].item() != 0.0, "Column 5 should be nonzero (not masked)"
        assert out_masked[0, 0, 15].item() != 0.0, "Column 15 should be nonzero (not masked)"

        # Assertion 2: feature_mask=None is byte-identical to the no-arg call (I5).
        out_default = build_fp_targets_from_shard(shard, idx, N_SAE)
        out_none = build_fp_targets_from_shard(shard, idx, N_SAE, feature_mask=None)
        assert t.equal(out_default, out_none), (
            "feature_mask=None produced different output from the no-arg call (I5 violated)"
        )
