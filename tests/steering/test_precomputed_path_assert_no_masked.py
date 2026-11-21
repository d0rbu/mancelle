"""B3 defense-in-depth: assert_no_masked on the precomputed-sel load path.

Invariant I4: no masked feature id may appear as a conditioning signal.
The fused path (reconstruct_and_select) already asserts this after selection.
The precomputed path (precomputed_data_loader + reconstruct_batch) must assert
the SAME invariant on conditioning features loaded from a sel/<hash>/ shard.

Without the fix, a mask-mismatched / stale shard silently violates I4 — the
masked feature flows into training/eval with no error.

Test plan
---------
1. Build a synthetic sel/<hash>/ shard (in memory + written to tmp_path) that
   CONTAINS a feature id the configured mask forbids.  Drive precomputed_data_loader
   directly with that shard + a feature_pipeline carrying the mask (bypassing the
   manifest cross-check — the defense-in-depth is what must catch it).

2. PRE-FIX RED: run currently raises no exception; the masked id silently
   appears in enc_fids at the enc_mask-True positions.

3. POST-FIX GREEN: the same run raises FeatureMaskError.

4. Supplementary cases (all green both before and after unless noted):
   a. whitelist case — shard contains id NOT in whitelist → FeatureMaskError
   b. clean shard (no masked id) under a mask → no exception (no false-positive)
   c. no-mask (feature_pipeline.feature_mask is None) → no exception, complete no-op
   d. padding safety: feature id 0 blacklisted; id 0 only appears as padding
      (enc_mask False) → NO FeatureMaskError (assert never inspects padding slots)

Coverage note
-------------
Both the fast path (all rows from a single shard — the common case) and the
slow path (cross-shard boundary → _stack_batch_parts) must be covered by B3's
assertion since both converge at the same post-reconstruct block in
precomputed_data_loader.  The primary tests exercise the fast path (B==1 so
rows stay in one shard).  The slow-path test uses B==2 split across two shards
so _stack_batch_parts is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from errors import FeatureMaskError
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
)
from steering.conditional.feature_pipeline import FeaturePipelineConfig, ThresholdSelectionConfig
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.conditional.thresholding import ThresholdType
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    save_packed_row_shard,
    save_selection_shard,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SAE = 64          # small universe so feature ids are well below uint16 limit
MASKED_FID = 7      # the feature id that masks will forbid
SAFE_FID = 10       # a clean feature id never forbidden in any mask below
_T = 32             # token context length
_N_ROWS = 1         # single row keeps batch logic simple for fast-path tests
_N_SEGS = 2         # two segments per row


# ---------------------------------------------------------------------------
# Helpers: build shards
# ---------------------------------------------------------------------------


def _make_base_shard(fids_per_seg: list[list[int]], n_rows: int = _N_ROWS) -> PackedRowShard:
    """Build a minimal PackedRowShard with exactly the given feature ids per segment.

    fids_per_seg[seg_idx] = list of feature ids for that segment.
    n_rows * n_segs_per_row == len(fids_per_seg).
    """
    n_segs = len(fids_per_seg)
    assert n_segs == n_rows * _N_SEGS, (
        f"n_segs {n_segs} != n_rows {n_rows} * _N_SEGS {_N_SEGS}"
    )
    seg_size = _T // _N_SEGS

    total_feats = sum(len(f) for f in fids_per_seg)

    tokens = t.zeros(n_rows, _T + 1, dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * _N_SEGS, _N_SEGS, dtype=t.int32)

    seg_token_start = t.zeros(n_segs, dtype=t.uint16)
    seg_token_end = t.zeros(n_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(n_segs, dtype=t.uint8)

    for seg_idx in range(n_segs):
        s = seg_idx % _N_SEGS
        seg_token_start[seg_idx] = s * seg_size
        seg_token_end[seg_idx] = _T if s == _N_SEGS - 1 else (s + 1) * seg_size

    # Build CSR offsets from per-seg feature counts
    counts = [len(f) for f in fids_per_seg]
    cumcounts = [0]
    for c in counts:
        cumcounts.append(cumcounts[-1] + c)
    seg_feat_offsets = t.tensor(cumcounts, dtype=t.int32)

    all_fids: list[int] = []
    for fids in fids_per_seg:
        all_fids.extend(fids)

    feature_ids = t.tensor(all_fids, dtype=t.uint16) if all_fids else t.zeros(0, dtype=t.uint16)
    feature_pcts = t.full((total_feats,), 200, dtype=t.uint8)  # non-zero pcts

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


def _make_sel_shard_for_base(base: PackedRowShard) -> SelectionShard:
    """Selection shard that selects ALL features from the base shard (identity)."""
    return SelectionShard(
        sel_offsets=base.seg_feat_offsets.to(t.int32),
        sel_fids=base.feature_ids.clone(),
        sel_buckets=t.ones(base.total_features, dtype=t.uint8),
        sel_pcts=t.full((base.total_features,), 200, dtype=t.uint8),
        neg_offsets=t.zeros(base.total_segments + 1, dtype=t.int32),
        neg_fids=t.tensor([], dtype=t.uint16),
    )


def _write_shard_pair(
    tmp_path: Path,
    fids_per_seg: list[list[int]],
    shard_name: str = "shard_00000",
    n_rows: int = _N_ROWS,
) -> tuple[Path, Path]:
    """Write a base shard + identity selection shard, return (shard_dir, sel_dir)."""
    base_shard = _make_base_shard(fids_per_seg, n_rows=n_rows)
    shard_dir = tmp_path / "packed"
    shard_dir.mkdir(parents=True, exist_ok=True)
    save_packed_row_shard(base_shard, shard_dir / shard_name)
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

    sel_shard = _make_sel_shard_for_base(base_shard)
    sel_dir = tmp_path / "sel"
    sel_dir.mkdir(parents=True, exist_ok=True)
    save_selection_shard(sel_shard, sel_dir / shard_name)

    return shard_dir, sel_dir


def _write_selection_manifest(
    sel_dir: Path,
    shard_dir: Path,
    feature_mask: FeatureMask | None,
) -> None:
    metadata = {
        "packed_rows": {
            "packed_row_fingerprint": packed_row_fingerprint(shard_dir),
        },
    }
    mask_meta = canonical_dict(feature_mask)
    if mask_meta is not None:
        metadata["feature_mask"] = mask_meta
    write_manifest(
        sel_dir,
        "test-selection",
        OverlayKind.PACKED_ROW,
        metadata=metadata,
    )


def _build_pipeline(n_sae: int = N_SAE, feature_mask: FeatureMask | None = None) -> FeaturePipelineConfig:
    """Build a FeaturePipelineConfig with a trivial threshold that keeps all features."""
    # Boundaries: linspace 0→1 for one feature row, expanded to n_sae.
    # With threshold=0 (select everything above 0 percentile) all features pass.
    # We use segment_topk with a high k so positives are accepted freely.
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, n_sae),  # keep all
            precomputed_thresholds=None,
        ),
        boundaries=t.linspace(0.0, 1.0, 3, dtype=t.float64).unsqueeze(0).expand(n_sae, -1).contiguous(),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=feature_mask,
    )


def _run_one_batch(shard_dir: Path, sel_dir: Path, pipeline: FeaturePipelineConfig):
    """Drive precomputed_data_loader for exactly one batch (B=1), return the batch."""
    _write_selection_manifest(sel_dir, shard_dir, pipeline.feature_mask)
    loader = precomputed_data_loader(
        packed_row_dir=shard_dir,
        B=1,
        feature_pipeline=pipeline,
        device="cpu",
        selection_dir=sel_dir,
        effective_buckets=4,
        n_sae_features=None,
    )
    return next(iter(loader))


# ---------------------------------------------------------------------------
# RED test (pre-fix): documents the I4 hole at d538fcb5
#
# This test FAILS pre-fix (the assert is NOT raised — the masked feature
# silently appears in enc_fids at enc_mask-True positions).  After the fix
# is applied the same scenario must raise FeatureMaskError (GREEN).
# ---------------------------------------------------------------------------


class TestPrecomputedPathAssertNoMaskedB3:
    """PRIMARY B3 tests: assert_no_masked defense-in-depth on precomputed-sel path."""

    def test_blacklist_masked_fid_raises(self, tmp_path: Path):
        """Blacklist: shard contains MASKED_FID; precomputed path must raise FeatureMaskError.

        Pre-fix (RED): silently yields batch with MASKED_FID in enc_fids[enc_mask].
        Post-fix (GREEN): FeatureMaskError raised before the batch is yielded.
        """
        # Build shard with MASKED_FID in segment 0
        shard_dir, sel_dir = _write_shard_pair(
            tmp_path,
            fids_per_seg=[[MASKED_FID, SAFE_FID], [SAFE_FID]],
        )
        mask = FeatureMask(FeatureMaskMode.blacklist, (MASKED_FID,))
        pipeline = _build_pipeline(feature_mask=mask)

        with pytest.raises(FeatureMaskError):
            _run_one_batch(shard_dir, sel_dir, pipeline)

    def test_whitelist_excluded_fid_raises(self, tmp_path: Path):
        """Whitelist: shard contains a feature NOT in the whitelist → FeatureMaskError.

        The whitelist only allows SAFE_FID=10.  Shard contains MASKED_FID=7 which
        is not in the whitelist — it must be rejected by the defense-in-depth check.
        """
        shard_dir, sel_dir = _write_shard_pair(
            tmp_path,
            fids_per_seg=[[MASKED_FID, SAFE_FID], [SAFE_FID]],
        )
        # Whitelist only includes SAFE_FID; MASKED_FID is excluded
        mask = FeatureMask(FeatureMaskMode.whitelist, (SAFE_FID,))
        pipeline = _build_pipeline(feature_mask=mask)

        with pytest.raises(FeatureMaskError):
            _run_one_batch(shard_dir, sel_dir, pipeline)

    def test_clean_shard_with_mask_no_raise(self, tmp_path: Path):
        """Clean shard (no masked id) under a mask → no exception (no false-positive)."""
        shard_dir, sel_dir = _write_shard_pair(
            tmp_path,
            fids_per_seg=[[SAFE_FID, SAFE_FID + 1], [SAFE_FID + 2]],
        )
        mask = FeatureMask(FeatureMaskMode.blacklist, (MASKED_FID,))  # MASKED_FID not in shard
        pipeline = _build_pipeline(feature_mask=mask)

        # Must not raise
        batch = _run_one_batch(shard_dir, sel_dir, pipeline)
        enc_fids = batch[2]
        enc_mask = batch[4]
        # Verify the shard active ids are indeed clean
        active_ids = enc_fids[enc_mask].tolist()
        assert MASKED_FID not in active_ids

    def test_no_mask_no_raise(self, tmp_path: Path):
        """I5: feature_mask=None is a complete no-op — no assertion, yields normally."""
        shard_dir, sel_dir = _write_shard_pair(
            tmp_path,
            fids_per_seg=[[MASKED_FID, SAFE_FID], [SAFE_FID]],  # shard has MASKED_FID
        )
        pipeline = _build_pipeline(feature_mask=None)  # No mask configured

        # Must not raise even though shard contains MASKED_FID
        batch = _run_one_batch(shard_dir, sel_dir, pipeline)
        enc_fids = batch[2]
        enc_mask = batch[4]
        # MASKED_FID is present in the output (no-op path)
        active_ids = enc_fids[enc_mask].tolist()
        assert MASKED_FID in active_ids, (
            f"Expected MASKED_FID={MASKED_FID} in no-mask path output; got {active_ids}"
        )


class TestPrecomputedPathPaddingSafetyB3:
    """B3 padding safety: assert never inspects zeros-padded (enc_mask False) slots.

    The B2-class padding trap: if feature id 0 is blacklisted but only occupies
    padding positions (enc_mask=False), the assertion MUST NOT fire.
    """

    def test_padding_fid0_blacklisted_no_false_raise(self, tmp_path: Path):
        """Feature 0 blacklisted; appears ONLY in padding (enc_mask False) → no raise.

        The shard contains features SAFE_FID only (no feature 0 anywhere).
        The enc_fids padding slots are zeros by construction (t.zeros in reconstruct_batch).
        A blacklist {0} must NOT raise because id 0 is not an active conditioning feature.
        """
        # Shard contains only SAFE_FID (never feature 0)
        shard_dir, sel_dir = _write_shard_pair(
            tmp_path,
            fids_per_seg=[[SAFE_FID], [SAFE_FID + 1]],
        )
        # Blacklist feature id 0 — this id only ever appears in padding zeros
        mask = FeatureMask(FeatureMaskMode.blacklist, (0,))
        pipeline = _build_pipeline(feature_mask=mask)

        # Must NOT raise: 0 appears only in padding, never in enc_mask-True positions
        batch = _run_one_batch(shard_dir, sel_dir, pipeline)
        enc_fids = batch[2]
        enc_mask = batch[4]
        # Prove: active ids do not contain 0
        active_ids = enc_fids[enc_mask].tolist()
        assert 0 not in active_ids, (
            f"Expected feature 0 absent from active ids; got active_ids={active_ids}"
        )

    def test_padding_safe_and_real_masked_fid_still_raises(self, tmp_path: Path):
        """Padding safety does NOT suppress a genuine violation.

        Shard has MASKED_FID in an active slot AND feature 0 only in padding.
        The blacklist {MASKED_FID, 0} must raise on the active MASKED_FID,
        not be silenced by the padding-safe check.
        """
        shard_dir, sel_dir = _write_shard_pair(
            tmp_path,
            fids_per_seg=[[MASKED_FID, SAFE_FID], [SAFE_FID]],
        )
        mask = FeatureMask(FeatureMaskMode.blacklist, (MASKED_FID, 0))
        pipeline = _build_pipeline(feature_mask=mask)

        with pytest.raises(FeatureMaskError):
            _run_one_batch(shard_dir, sel_dir, pipeline)


class TestPrecomputedSlowPathB3:
    """B3 covers the slow path (_stack_batch_parts) — cross-shard boundary batch.

    When B==2 and rows come from two different shards, _stack_batch_parts is
    invoked.  The assertion must fire for this path too.
    """

    def test_slow_path_cross_shard_blacklist_raises(self, tmp_path: Path):
        """Cross-shard batch: masked fid in one shard's selection → FeatureMaskError.

        Build TWO shards (each 1 row).  Shard 0 is clean; shard 1 contains MASKED_FID.
        With B=2, both rows end up in pending together → cross-shard → _stack_batch_parts.
        The B3 assertion (which runs after _stack_batch_parts) must catch the violation.
        """
        # Two distinct shard names so they are loaded as separate shards
        base_shard_0 = _make_base_shard([[SAFE_FID], [SAFE_FID + 1]])
        base_shard_1 = _make_base_shard([[MASKED_FID, SAFE_FID], [SAFE_FID]])

        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

        sel_dir = tmp_path / "sel"
        sel_dir.mkdir(parents=True, exist_ok=True)

        for name, base in [("shard_00000", base_shard_0), ("shard_00001", base_shard_1)]:
            save_packed_row_shard(base, shard_dir / name)
            sel = _make_sel_shard_for_base(base)
            save_selection_shard(sel, sel_dir / name)

        mask = FeatureMask(FeatureMaskMode.blacklist, (MASKED_FID,))
        pipeline = _build_pipeline(feature_mask=mask)
        _write_selection_manifest(sel_dir, shard_dir, pipeline.feature_mask)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=4,
            n_sae_features=None,
        )

        with pytest.raises(FeatureMaskError):
            next(iter(loader))

    def test_slow_path_clean_shards_no_raise(self, tmp_path: Path):
        """Cross-shard batch: both shards clean → no false-positive."""
        base_shard_0 = _make_base_shard([[SAFE_FID], [SAFE_FID + 1]])
        base_shard_1 = _make_base_shard([[SAFE_FID + 2], [SAFE_FID + 3]])

        shard_dir = tmp_path / "packed"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))

        sel_dir = tmp_path / "sel"
        sel_dir.mkdir(parents=True, exist_ok=True)

        for name, base in [("shard_00000", base_shard_0), ("shard_00001", base_shard_1)]:
            save_packed_row_shard(base, shard_dir / name)
            sel = _make_sel_shard_for_base(base)
            save_selection_shard(sel, sel_dir / name)

        mask = FeatureMask(FeatureMaskMode.blacklist, (MASKED_FID,))
        pipeline = _build_pipeline(feature_mask=mask)
        _write_selection_manifest(sel_dir, shard_dir, pipeline.feature_mask)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            feature_pipeline=pipeline,
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=4,
            n_sae_features=None,
        )

        batch = next(iter(loader))
        enc_fids = batch[2]
        enc_mask = batch[4]
        active_ids = enc_fids[enc_mask].tolist()
        assert MASKED_FID not in active_ids
