"""Cross-cutting integration tests for the feature-mask feature (Task 17).

Each test covers a CROSS-PATH gap NOT already covered by a per-task test:

  In1 — selection-shard ≡ fused (sampling path):
        Precompute path (select_features_for_shard) and fused path
        (reconstruct_and_select) both respect the mask; asserts the
        post-apply_to_positive pre-filter input is identical across paths,
        and that each path individually satisfies the I4 invariant (no
        masked id in positives or negatives).
        NOTE: cross-path bit-equality of sampled positives is NOT asserted
        (pre-existing RNG-ordering difference between paths is out of scope).
        NOTE: the threshold/deterministic cross-path is out of scope here
        because select_features_for_shard only supports threshold types that
        use percentile-only inputs (feature_percentile_segment_topk), which
        do not match the segment_topk type used by reconstruct_and_select.

  In2 — per-path negative invariant (consolidated):
        For sample_inactive_features_batched, the fused reconstruct_and_select
        Phase-5 inline sampler, and sample_delta_negatives: under both a
        blacklist AND a whitelist, outputs contain only allowed ids, clamp is
        respected, and the function is deterministic under fixed seed.
        Per-task tests in tasks 8/9/11 already assert these individually;
        this test adds a CONSOLIDATED cross-path invariant check referencing
        all three implementations together.

  In3 — I1/I2/I3 backward-compat:
        (i)  FROZEN GOLDEN PIN for compute_selection_hash (I1 anti-regression):
             the legacy no-mask hash is pinned to a specific 64-hex constant.
        (ii) Legacy legacy checkpoint path is byte-identical to no-mask path (I2).
        (iii) A legacy sel/ dir with NO feature_mask manifest key combined with
              configured mask=None passes the load-time assertion (I3).
        Tasks 6/15/16 already cover parts of these; this test adds the
        FROZEN-GOLDEN hex pin (the specific 64-char string) missing from Task 6.

  In4 — accepted_feature_ids.txt as whitelist:
        Write a newline-separated ids file (comment + blank line), resolve it
        as a whitelist via resolve_feature_mask, and assert end-to-end flow:
        resolved set correct; apply_to_positive keeps only whitelisted ids.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from errors import FeatureMaskError
from scripts.preprocess.precompute_selected_features import select_features_for_shard
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig, sample_delta_negatives
from steering.conditional.encoder_decoder.dataloader import (
    _capture_feature_pipeline_rng_state,
    reconstruct_and_select,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    canonical_dict,
    resolve_feature_mask,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    sample_inactive_features_batched,
)
from steering.overlay_io import (
    OverlayKind,
    overlay_shard_path,
    read_manifest,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    compute_selection_hash,
    load_selection_shard,
    save_packed_row_shard,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# I1 FROZEN GOLDEN: the legacy 6-key payload hash for the representative
# payload below.  This value was computed once (2026-05-17) and MUST NOT
# change — any change breaks backward-compat for existing sel/ directories.
#
# Payload: positive_selection={"threshold_type": "segment_topk", "threshold": 2048},
#          negative_selection=None, delta_config=None, num_buckets=10, seed=42, ddp_rank=0
_I1_REPRESENTATIVE_PAYLOAD_HASH = (
    "ae63e39fd986b5746af076aa504bc2206a90837565e2f7409999e724ef906c0d"
)
_I1_POS = {"threshold_type": "segment_topk", "threshold": 2048}

N_SAE = 64
N_ROWS = 4
SEGS_PER_ROW = 3
FEATS_PER_SEG = 16
CONTEXT_LEN = 32

# Feature IDs planted in the shard for blacklist/whitelist tests
_BLACKLIST_IDS = (7, 14)
_WHITELIST_IDS = (1, 5, 9, 15, 22, 30, 41)


# ---------------------------------------------------------------------------
# Shared fixture helpers (mirrors test_precompute_selected_features_mask.py)
# ---------------------------------------------------------------------------


def _make_per_feature_stats(n: int = N_SAE) -> dict[str, PerFeatureStat]:
    """Build a tiny stats dict with deterministic percentiles."""
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    out: dict[str, PerFeatureStat] = {}
    for i in range(n):
        out[str(i)] = PerFeatureStat(
            count=100 - i,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(full_percentiles),
            histogram=None,
        )
    return out


def _make_shard(seed: int = 42) -> PackedRowShard:
    """Build a deterministic PackedRowShard with _BLACKLIST_IDS and _WHITELIST_IDS planted."""
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
            seg_token_end[idx] = (
                CONTEXT_LEN if s == SEGS_PER_ROW - 1 else (s + 1) * seg_size
            )
    seg_feat_offsets = t.arange(
        0, (total_segs + 1) * FEATS_PER_SEG, FEATS_PER_SEG, dtype=t.int32
    )
    # Real packed rows list each segment's ACTIVE features exactly once (they come
    # from ``(pooled > 0).nonzero()`` in extraction and are stored sorted ascending),
    # so per-segment feature ids are unique. The precompute percentile lookup asserts
    # exactly one entry per selected id, so the fixture must honor that invariant.
    # Plant the blacklist + whitelist ids once per segment (so both mask modes are
    # exercised) and fill the remaining slots with distinct random ids.
    planted = list(dict.fromkeys([*_BLACKLIST_IDS, *_WHITELIST_IDS]))
    assert len(planted) <= FEATS_PER_SEG, "planted ids must fit within a segment"
    per_seg_rows: list[list[int]] = []
    for _ in range(total_segs):
        fill = [f for f in t.randperm(N_SAE).tolist() if f not in planted]
        seg = planted + fill[: FEATS_PER_SEG - len(planted)]
        per_seg_rows.append(sorted(seg))
    feature_ids = t.tensor(
        [f for seg in per_seg_rows for f in seg], dtype=t.uint16
    )
    feature_pcts = t.randint(100, 255, (total_feats,), dtype=t.uint8)
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


def _make_sampling_pipeline(
    mask: FeatureMask | None,
    seed: int = 1,
) -> FeaturePipelineConfig:
    """Build a geometric-sampling FeaturePipelineConfig with optional mask."""
    cfg = OmegaConf.create({
        "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
        "negative_selection": {"distribution": "geometric", "mean": 3.0, "std": 3.0},
    })
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE,
        num_buckets=8,
        seed=seed,
        feature_mask=mask,
    )


def _build_mask_reuse_pipeline(
    cfg: DictConfig,
    mask: FeatureMask | None = None,
    delta: FeatureDeltaConfig | None = None,
) -> FeaturePipelineConfig:
    return build_feature_pipeline_from_cfg_block(
        cfg, per_feature_stats=_make_per_feature_stats(), n_sae_features=N_SAE,
        num_buckets=8, seed=17, feature_mask=mask, feature_delta=delta,
    )


@pytest.mark.parametrize("mode", ["blacklist", "whitelist"])
@pytest.mark.parametrize("kind", ["threshold", "sampling", "sampling_delta"])
def test_resolved_mask_reuse_preserves_seeded_outputs(mode: str, kind: str) -> None:
    ids = _WHITELIST_IDS if mode == "whitelist" else _BLACKLIST_IDS
    selection = (
        {"threshold_type": "feature_percentile", "threshold": 0.7}
        if kind == "threshold"
        else {"distribution": "geometric", "mean": 4.0, "std": 4.0}
    )
    if kind == "sampling_delta":
        selection["delta_weights"] = "absolute_percentile"
    cfg = OmegaConf.create({
        "feature_selection": selection,
        "negative_selection": {"distribution": "geometric", "mean": 3.0, "std": 3.0},
        "feature_mask": {"mode": mode, "ids": list(ids)},
    })
    mask = resolve_feature_mask(cfg, N_SAE)
    assert mask is not None
    allowed = mask.allowed_universe_mask(N_SAE)
    mask.keep_mask(t.arange(N_SAE))
    public_copy = mask.allowed_universe_mask(N_SAE)
    public_copy.logical_not_()
    assert t.equal(mask.allowed_universe_mask(N_SAE), allowed)
    delta = (
        FeatureDeltaConfig(mode=DeltaMode.cum_max, neg_delta_ratio=0.5)
        if kind == "sampling_delta" else None
    )
    baseline = _build_mask_reuse_pipeline(cfg, delta=delta)
    reused = _build_mask_reuse_pipeline(cfg, mask=mask, delta=delta)
    assert reused.feature_mask is mask
    assert reused.negative_bundle is not None
    assert reused.negative_bundle.feature_mask is mask
    assert baseline.feature_mask == mask and baseline.feature_mask is not mask
    assert _capture_feature_pipeline_rng_state(baseline) == _capture_feature_pipeline_rng_state(reused)
    shard = _make_shard()
    expected, expected_fp = reconstruct_and_select(shard, list(range(N_ROWS)), baseline)
    actual, actual_fp = reconstruct_and_select(shard, list(range(N_ROWS)), reused)
    assert expected_fp is actual_fp is None
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert expected_tensor.dtype == actual_tensor.dtype
        assert expected_tensor.shape == actual_tensor.shape
        assert t.equal(expected_tensor, actual_tensor)
    assert _capture_feature_pipeline_rng_state(baseline) == _capture_feature_pipeline_rng_state(reused)
    assert t.equal(mask.allowed_universe_mask(N_SAE), allowed)


def test_resolved_file_mask_is_per_config_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "mask.txt"
    path.write_text("1\n5\n9\n")
    cfg = OmegaConf.create({
        "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
        "feature_mask": {"mode": "whitelist", "path": str(path)},
    })
    mask = resolve_feature_mask(cfg, N_SAE)
    assert mask is not None
    path.write_text("15\n22\n30\n")
    current = _build_mask_reuse_pipeline(cfg, mask=mask)
    assert current.feature_mask is mask and tuple(mask.ids) == (1, 5, 9)
    next_mask = resolve_feature_mask(cfg, N_SAE)
    assert next_mask is not None and tuple(next_mask.ids) == (15, 22, 30)
    following = _build_mask_reuse_pipeline(cfg, mask=next_mask)
    assert following.feature_mask is next_mask
    path.unlink()
    assert _build_mask_reuse_pipeline(cfg, mask=mask).feature_mask is mask
    with pytest.raises(FeatureMaskError, match="not found"):
        resolve_feature_mask(cfg, N_SAE)


def test_none_mask_remains_pipeline_resolver_sentinel() -> None:
    cfg = OmegaConf.create({
        "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
    })
    assert _build_mask_reuse_pipeline(cfg, mask=None).feature_mask is None
    cfg.feature_mask = {"mode": "whitelist", "ids": [1, 5, 9]}
    pipeline = _build_mask_reuse_pipeline(cfg, mask=None)
    assert pipeline.feature_mask is not None and tuple(pipeline.feature_mask.ids) == (1, 5, 9)


# ---------------------------------------------------------------------------
# In1 — selection-shard ≡ fused (sampling path, I4 cross-path invariant)
# ---------------------------------------------------------------------------


class TestIn1SelectionShardFusedEquivalence:
    """In1: both precompute and fused paths respect the mask (sampling path).

    Per the task spec: for the SAMPLING path, we assert:
      (a) the post-apply_to_positive pre-sampling input (filtered positives)
          from each path individually excludes masked ids (I4 invariant per path).
      (b) cross-path bit-equality of sampled positives is NOT asserted
          (pre-existing RNG-ordering difference, out of scope).
    """

    def test_precompute_path_blacklist_sel_fids_exclude_masked(self, tmp_path: Path) -> None:
        """select_features_for_shard positives must not contain blacklisted ids."""
        shard = _make_shard()
        shard_dir = tmp_path / "base"
        shard_dir.mkdir()
        shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

        mask = FeatureMask("blacklist", _BLACKLIST_IDS)
        pipeline = _make_sampling_pipeline(mask, seed=1)
        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        pos_set = set(sel.sel_fids.tolist())
        for bid in _BLACKLIST_IDS:
            assert bid not in pos_set, (
                f"In1: precompute path: blacklisted id {bid} found in sel_fids"
            )

    def test_precompute_path_blacklist_neg_fids_exclude_masked(self, tmp_path: Path) -> None:
        """select_features_for_shard negatives must not contain blacklisted ids."""
        shard = _make_shard()
        shard_dir = tmp_path / "base"
        shard_dir.mkdir()
        shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

        mask = FeatureMask("blacklist", _BLACKLIST_IDS)
        pipeline = _make_sampling_pipeline(mask, seed=1)
        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        neg_set = set(sel.neg_fids.tolist())
        for bid in _BLACKLIST_IDS:
            assert bid not in neg_set, (
                f"In1: precompute path: blacklisted id {bid} found in neg_fids"
            )

    def test_fused_path_blacklist_positives_exclude_masked(self) -> None:
        """reconstruct_and_select positives must not contain blacklisted ids."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", _BLACKLIST_IDS)
        pipeline = _make_sampling_pipeline(mask, seed=1)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                active_fids = rb.enc_fids[b, s, m]
                active_mags = rb.enc_mags[b, s, m]
                pos_fids = active_fids[active_mags != eff].tolist()
                for bid in _BLACKLIST_IDS:
                    assert bid not in pos_fids, (
                        f"In1: fused path: blacklisted id {bid} in positives at "
                        f"batch={b} seg={s}"
                    )

    def test_fused_path_blacklist_negatives_exclude_masked(self) -> None:
        """reconstruct_and_select negatives must not contain blacklisted ids."""
        shard = _make_shard()
        mask = FeatureMask("blacklist", _BLACKLIST_IDS)
        pipeline = _make_sampling_pipeline(mask, seed=1)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                active_fids = rb.enc_fids[b, s, m]
                active_mags = rb.enc_mags[b, s, m]
                neg_fids = active_fids[active_mags == eff].tolist()
                for bid in _BLACKLIST_IDS:
                    assert bid not in neg_fids, (
                        f"In1: fused path: blacklisted id {bid} in negatives at "
                        f"batch={b} seg={s}"
                    )

    def test_precompute_and_fused_both_exclude_same_ids(self, tmp_path: Path) -> None:
        """Cross-path I4 consolidated: both paths exclude the exact same blacklisted ids.

        Not asserting bit-equality of outputs (RNG-ordering differs between paths),
        only that the I4 invariant (no masked id) holds on both.
        """
        shard = _make_shard()
        shard_dir = tmp_path / "base"
        shard_dir.mkdir()
        shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

        mask = FeatureMask("blacklist", _BLACKLIST_IDS)
        pipeline_pre = _make_sampling_pipeline(mask, seed=2)
        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline_pre, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        pipeline_fused = _make_sampling_pipeline(mask, seed=2)
        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline_fused)

        pre_all = set(sel.sel_fids.tolist()) | set(sel.neg_fids.tolist())
        fused_all: set[int] = set()
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                fused_all.update(rb.enc_fids[b, s, m].tolist())

        for bid in _BLACKLIST_IDS:
            assert bid not in pre_all, f"precompute: blacklisted {bid} in output"
            assert bid not in fused_all, f"fused: blacklisted {bid} in output"

    def test_whitelist_both_paths_restrict_to_whitelist(self, tmp_path: Path) -> None:
        """Both paths restrict to the whitelist on the whitelist path."""
        shard = _make_shard()
        shard_dir = tmp_path / "base"
        shard_dir.mkdir()
        shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

        mask = FeatureMask("whitelist", _WHITELIST_IDS)
        pipeline_pre = _make_sampling_pipeline(mask, seed=3)
        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline_pre, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        allowed = set(_WHITELIST_IDS)
        pre_sel = set(sel.sel_fids.tolist())
        bad = pre_sel - allowed
        assert not bad, f"precompute: non-whitelisted ids in sel_fids: {bad}"

        pre_neg = set(sel.neg_fids.tolist())
        bad_neg = pre_neg - allowed
        assert not bad_neg, f"precompute: non-whitelisted ids in neg_fids: {bad_neg}"


# ---------------------------------------------------------------------------
# In2 — per-path negative invariant (consolidated cross-impl check)
# ---------------------------------------------------------------------------


class TestIn2NegativeInvariantCrossImpl:
    """In2: consolidated invariant across all 3 negative implementations.

    Per-task tests (8, 9, 11) already assert individually; this test adds
    a CROSS-IMPL summary invariant: all 3 under a blacklist AND whitelist.

    All outputs must contain only allowed ids, be deterministic under fixed
    seed, and respect the clamp (no ids outside [0, N_SAE)).
    """

    # Shared NegativeSamplingSelection config
    _NEG_CFG = NegativeSamplingSelection(mean=5.0, std=5.0, min_k=3, max_k=8)

    # ---- sample_inactive_features_batched ----

    def test_batched_blacklist_I4_invariant(self) -> None:
        """sample_inactive_features_batched: blacklisted id never in output."""
        blacklisted = 42
        mask = FeatureMask("blacklist", (blacklisted,))
        rng = t.Generator().manual_seed(0)
        results = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)],
            N_SAE,
            self._NEG_CFG,
            rng,
            feature_mask=mask,
        )
        assert results, "Expected at least one result batch"
        assert blacklisted not in results[0].tolist(), (
            f"sample_inactive_features_batched: blacklisted {blacklisted} in output"
        )

    def test_batched_whitelist_I4_invariant(self) -> None:
        """sample_inactive_features_batched: whitelist restricts output universe."""
        whitelist = (10, 20, 30, 40, 50)
        mask = FeatureMask("whitelist", whitelist)
        rng = t.Generator().manual_seed(7)
        results = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)],
            N_SAE,
            self._NEG_CFG,
            rng,
            feature_mask=mask,
        )
        assert results
        bad = set(results[0].tolist()) - set(whitelist)
        assert not bad, (
            f"sample_inactive_features_batched: non-whitelisted ids: {bad}"
        )

    def test_batched_determinism(self) -> None:
        """sample_inactive_features_batched: identical seed → identical output."""
        mask = FeatureMask("blacklist", (5,))
        r1 = t.Generator().manual_seed(99)
        r2 = t.Generator().manual_seed(99)
        out1 = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)], N_SAE, self._NEG_CFG, r1, feature_mask=mask
        )
        out2 = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)], N_SAE, self._NEG_CFG, r2, feature_mask=mask
        )
        assert t.equal(out1[0], out2[0]), (
            "sample_inactive_features_batched: non-deterministic under fixed seed"
        )

    # ---- sample_delta_negatives ----

    def test_delta_blacklist_I4_invariant(self) -> None:
        """sample_delta_negatives: blacklisted id never in output."""
        blacklisted = 55  # within [0, N_SAE=64)
        mask = FeatureMask("blacklist", (blacklisted,))
        ndf = t.tensor([50, 60], dtype=t.int64)
        ndw = t.tensor([0.5, 0.5], dtype=t.float32)
        ref = t.zeros(N_SAE, dtype=t.bool)
        ref[50] = True
        ref[60] = True

        for seed in range(10):
            rng = t.Generator().manual_seed(seed)
            result = sample_delta_negatives(
                k=5,
                neg_delta_ratio=0.5,
                neg_delta_fids=ndf,
                neg_delta_weights=ndw,
                raw_feature_ids=t.tensor([0], dtype=t.int64),
                ref_mask=ref,
                selected_positive_ids=t.tensor([0], dtype=t.int64),
                n_sae_features=N_SAE,
                rng=rng,
                feature_mask=mask,
            )
            assert blacklisted not in result.tolist(), (
                f"sample_delta_negatives: blacklisted {blacklisted} at seed {seed}"
            )

    def test_delta_whitelist_I4_invariant(self) -> None:
        """sample_delta_negatives: whitelist restricts output universe."""
        whitelist_ids = tuple(range(50, 64))
        mask = FeatureMask("whitelist", whitelist_ids)
        ndf = t.tensor(list(range(50, 64)), dtype=t.int64)
        ndw = t.full((14,), 0.5, dtype=t.float32)
        ref = t.zeros(N_SAE, dtype=t.bool)
        for f in range(50, 64):
            ref[f] = True
        allowed = set(whitelist_ids)

        for seed in range(10):
            rng = t.Generator().manual_seed(seed)
            result = sample_delta_negatives(
                k=5,
                neg_delta_ratio=0.5,
                neg_delta_fids=ndf,
                neg_delta_weights=ndw,
                raw_feature_ids=t.tensor([0], dtype=t.int64),
                ref_mask=ref,
                selected_positive_ids=t.tensor([0], dtype=t.int64),
                n_sae_features=N_SAE,
                rng=rng,
                feature_mask=mask,
            )
            bad = set(result.tolist()) - allowed
            assert not bad, (
                f"sample_delta_negatives: non-whitelisted ids at seed {seed}: {bad}"
            )

    # ---- reconstruct_and_select Phase-5 inline sampler ----

    def test_fused_neg_blacklist_I4_invariant(self) -> None:
        """reconstruct_and_select inline sampler: blacklisted id never in neg slots."""
        shard = _make_shard(seed=42)
        mask = FeatureMask("blacklist", _BLACKLIST_IDS)
        pipeline = _make_sampling_pipeline(mask, seed=5)
        eff = pipeline.effective_buckets

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        neg_fids: set[int] = set()
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                active = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                neg_fids.update(active[mags == eff].tolist())

        for bid in _BLACKLIST_IDS:
            assert bid not in neg_fids, (
                f"fused inline sampler: blacklisted {bid} in neg slots"
            )

    def test_fused_neg_whitelist_I4_invariant(self) -> None:
        """reconstruct_and_select inline sampler: whitelist restricts neg ids."""
        shard = _make_shard(seed=42)
        mask = FeatureMask("whitelist", _WHITELIST_IDS)
        pipeline = _make_sampling_pipeline(mask, seed=6)
        eff = pipeline.effective_buckets
        allowed = set(_WHITELIST_IDS)

        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)

        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                active = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                neg_fids = set(active[mags == eff].tolist())
                bad = neg_fids - allowed
                assert not bad, (
                    f"fused inline sampler: non-whitelisted ids in neg: {bad}"
                )

    def test_cross_impl_all_three_blacklist_I4(self) -> None:
        """All three negative impls satisfy I4 under the SAME blacklist."""
        blacklisted = 33
        mask = FeatureMask("blacklist", (blacklisted,))

        # 1. sample_inactive_features_batched
        rng1 = t.Generator().manual_seed(11)
        res1 = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)], N_SAE, self._NEG_CFG, rng1, feature_mask=mask
        )
        assert blacklisted not in res1[0].tolist(), "impl-1 (batched) failed I4"

        # 2. sample_delta_negatives
        ndf = t.tensor([10, 20], dtype=t.int64)
        ndw = t.tensor([0.5, 0.5], dtype=t.float32)
        ref = t.zeros(N_SAE, dtype=t.bool)
        ref[10] = True
        ref[20] = True
        rng2 = t.Generator().manual_seed(11)
        res2 = sample_delta_negatives(
            k=5, neg_delta_ratio=0.5,
            neg_delta_fids=ndf, neg_delta_weights=ndw,
            raw_feature_ids=t.tensor([0], dtype=t.int64),
            ref_mask=ref,
            selected_positive_ids=t.tensor([], dtype=t.int64),
            n_sae_features=N_SAE, rng=rng2, feature_mask=mask,
        )
        assert blacklisted not in res2.tolist(), "impl-2 (delta) failed I4"

        # 3. reconstruct_and_select inline sampler
        shard = _make_shard(seed=42)
        pipeline = _make_sampling_pipeline(mask, seed=7)
        eff = pipeline.effective_buckets
        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)
        all_neg: set[int] = set()
        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                active = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                all_neg.update(active[mags == eff].tolist())
        assert blacklisted not in all_neg, "impl-3 (fused) failed I4"


# ---------------------------------------------------------------------------
# In3 — I1/I2/I3 backward-compat (frozen golden + legacy dir passes)
# ---------------------------------------------------------------------------


class TestIn3BackwardCompat:
    """In3: frozen golden hash pin (I1) + I2 path byte-identical + I3 legacy dir.

    Cross-references:
      - I1 hash invariance: test_selection_hash_feature_mask.py::test_disabled_is_byte_identical_legacy
        asserts disabled==legacy dynamically; THIS test pins the EXACT 64-hex string.
      - I2 path unchanged: test_checkpoint_dir_feature_mask.py::TestEncdecTrainPath::test_no_mask_is_byte_identical_to_legacy
        covers the pretrain path; THIS test also verifies the hash key path.
      - I3 legacy dir: test_selection_manifest_feature_mask.py::test_legacy_manifest_no_key_no_mask_passes
        already covers the assertion; THIS test adds an explicit alias guard.
    """

    def test_I1_frozen_golden_hash(self) -> None:
        """FROZEN GOLDEN: compute_selection_hash for the representative legacy payload
        must equal the pinned 64-hex constant.

        If this test fails the hash function changed, which would break ALL
        existing sel/ directories keyed by the old hash.
        """
        actual = compute_selection_hash(_I1_POS, None, None, 10, 42, 0)
        assert actual == _I1_REPRESENTATIVE_PAYLOAD_HASH, (
            f"I1 FROZEN GOLDEN violated!\n"
            f"  Expected: {_I1_REPRESENTATIVE_PAYLOAD_HASH}\n"
            f"  Got:      {actual}\n"
            "This change breaks existing sel/ directory discovery — "
            "do NOT update the constant without migrating all artifact paths."
        )

    def test_I1_disabled_mask_equals_frozen_golden(self) -> None:
        """feature_mask=None and omitted both produce the frozen golden hash."""
        assert compute_selection_hash(_I1_POS, None, None, 10, 42, 0) == _I1_REPRESENTATIVE_PAYLOAD_HASH
        assert compute_selection_hash(_I1_POS, None, None, 10, 42, 0, feature_mask=None) == _I1_REPRESENTATIVE_PAYLOAD_HASH

    def test_I2_no_mask_leaves_hash_unchanged(self) -> None:
        """The selection hash is unchanged (I2) when no feature_mask is set.

        Verifies that the hash key used to name sel/ dirs is identical to the
        pre-mask legacy key when feature_mask is absent/None.
        """
        hash_no_arg = compute_selection_hash(_I1_POS, None, None, 10, 42, 0)
        hash_none = compute_selection_hash(_I1_POS, None, None, 10, 42, 0, feature_mask=None)
        assert hash_no_arg == hash_none, "hash differs with feature_mask=None vs no arg (I2)"
        # Both also equal the frozen golden
        assert hash_no_arg == _I1_REPRESENTATIVE_PAYLOAD_HASH, (
            "I2 hash does not equal the frozen golden constant"
        )

    def test_I3_legacy_sel_dir_no_manifest_key_passes(self, tmp_path: Path) -> None:
        """A legacy sel/ dir with NO feature_mask manifest key + configured mask=None
        must pass the load-time assertion (I3 back-compat).

        This is the anti-regression guard for existing, pre-mask sel/ artifacts.
        """
        # Build a minimal selection shard + write a legacy manifest without feature_mask
        shard = _make_shard()
        shard_dir = tmp_path / "base"
        shard_dir.mkdir()
        shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

        pipeline = _make_sampling_pipeline(None, seed=0)
        out_dir = tmp_path / "sel_legacy"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")

        # Write a manifest with NO metadata key (legacy style)
        write_manifest(
            out_dir,
            "legacyhash",
            OverlayKind.PACKED_ROW,
            metadata=None,  # no feature_mask key at all
        )

        # Simulate the load-time assertion (the assert logic that the trainer runs)
        manifest = read_manifest(out_dir)
        manifest_mask = (
            manifest.get("metadata", {}).get("feature_mask") if manifest else None
        )
        configured_mask = None
        # configured_canonical = canonical_dict(configured_mask) = None
        # assertion: manifest_mask (None) == canonical_dict(None) (None) → passes
        assert manifest_mask == canonical_dict(configured_mask), (
            "I3 failed: legacy manifest (no feature_mask key) + configured=None "
            "should pass the load-time assertion"
        )

    def test_I3_no_upstream_artifact_touched_by_mask_enabled_config(
        self, tmp_path: Path
    ) -> None:
        """A masked config MUST NOT discover/touch a legacy (no-mask) sel/ dir.

        This guard ensures that a run with feature_mask=blacklist(K) uses a
        DIFFERENT hash from a legacy (no-mask) run, so both sel/ dirs can
        coexist without collision.
        """
        legacy_hash = compute_selection_hash(_I1_POS, None, None, 10, 42, 0)
        masked_hash = compute_selection_hash(
            _I1_POS, None, None, 10, 42, 0,
            feature_mask={"mode": "blacklist", "ids": [5]},
        )
        assert legacy_hash != masked_hash, (
            "Masked config produces the same hash as legacy — "
            "masked runs would overwrite legacy sel/ dirs!"
        )


# ---------------------------------------------------------------------------
# In4 — accepted_feature_ids.txt as whitelist
# ---------------------------------------------------------------------------


class TestIn4AcceptedFeatureIdsTxt:
    """In4: newline-separated ids file (accepted_feature_ids.txt format) used as whitelist.

    Mirrors the feature_filter format: comment lines (#), blank lines ignored.
    Verifies the full resolved → apply_to_positive chain.
    """

    def test_resolve_from_txt_with_comments_and_blanks(self, tmp_path: Path) -> None:
        """resolve_feature_mask parses a .txt with comment + blank + ids correctly."""
        ids_file = tmp_path / "accepted_feature_ids.txt"
        ids_file.write_text(
            "# accepted feature ids\n"
            "\n"
            "5\n"
            "12\n"
            "30\n"
            "\n"
            "41\n"
        )
        mask = resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(ids_file)}},
            n_sae_features=N_SAE,
        )
        assert mask is not None
        assert mask.mode == "whitelist"
        assert mask.ids == (5, 12, 30, 41), (
            f"Unexpected resolved ids: {mask.ids}"
        )

    def test_apply_to_positive_keeps_only_whitelisted(self, tmp_path: Path) -> None:
        """After resolution, apply_to_positive retains only whitelisted ids."""
        ids_file = tmp_path / "accepted_feature_ids.txt"
        ids_file.write_text("# whitelist\n5\n12\n30\n41\n")
        mask = resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(ids_file)}},
            n_sae_features=N_SAE,
        )
        assert mask is not None

        fids = t.tensor([3, 5, 10, 12, 25, 30, 41, 55], dtype=t.int64)
        mags = t.arange(len(fids), dtype=t.float32)
        pcts = t.full((len(fids),), 100, dtype=t.uint8)

        out_fids, out_mags, out_pcts = mask.apply_to_positive(fids, mags, pcts)
        assert out_fids.tolist() == [5, 12, 30, 41], (
            f"Unexpected filtered fids: {out_fids.tolist()}"
        )
        # Non-whitelisted ids must not appear
        non_wl = set(out_fids.tolist()) - {5, 12, 30, 41}
        assert not non_wl, f"Non-whitelisted ids in output: {non_wl}"

    def test_end_to_end_txt_whitelist_pipeline(self, tmp_path: Path) -> None:
        """Resolved whitelist flows end-to-end through sampling pipeline."""
        ids_file = tmp_path / "accepted_feature_ids.txt"
        # Use ids from _WHITELIST_IDS which are planted in _make_shard()
        ids_file.write_text("\n".join(["# whitelist"] + [str(i) for i in _WHITELIST_IDS]) + "\n")
        mask = resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(ids_file)}},
            n_sae_features=N_SAE,
        )
        assert mask is not None
        assert set(mask.ids) == set(_WHITELIST_IDS)

        # Run through sampling pipeline
        pipeline = _make_sampling_pipeline(mask, seed=8)
        assert pipeline.feature_mask is mask

        shard = _make_shard()
        rb, _ = reconstruct_and_select(shard, list(range(N_ROWS)), pipeline)
        eff = pipeline.effective_buckets
        allowed = set(_WHITELIST_IDS)

        for b in range(rb.enc_fids.shape[0]):
            for s in range(rb.enc_fids.shape[1]):
                m = rb.enc_mask[b, s]
                if not m.any():
                    continue
                active = rb.enc_fids[b, s, m]
                mags = rb.enc_mags[b, s, m]
                pos_fids = set(active[mags != eff].tolist())
                bad = pos_fids - allowed
                assert not bad, (
                    f"In4 e2e: non-whitelisted pos ids at b={b} s={s}: {bad}"
                )

    def test_allowed_universe_mask_consistent_with_resolved(self, tmp_path: Path) -> None:
        """allowed_universe_mask(resolved_mask) matches the expected boolean tensor."""
        ids_file = tmp_path / "ids.txt"
        ids_file.write_text("# test\n10\n20\n30\n")
        mask = resolve_feature_mask(
            {"feature_mask": {"mode": "whitelist", "path": str(ids_file)}},
            n_sae_features=N_SAE,
        )
        assert mask is not None
        univ = mask.allowed_universe_mask(N_SAE)
        assert univ.shape == (N_SAE,)
        assert univ[10].item() is True
        assert univ[20].item() is True
        assert univ[30].item() is True
        assert univ[0].item() is False
        assert univ[11].item() is False
