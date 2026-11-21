"""Defense-in-depth runtime guard tests (H-assert).

Each guard site:
  1. Fault-injection test: MONKEYPATCHES the selection/sampling helper to inject
     a blacklisted id into the finalized output, then asserts the guard fires
     (raises FeatureMaskError) — proving each guard is actually wired.
  2. Correct-path test: normal masked path does NOT raise.
  3. I5 test: feature_mask=None guard is NEVER entered (assert_no_masked not called).

Sites covered:
  A. scripts/preprocess/precompute_selected_features.select_features_for_shard
     — positive selected_fids (after selection) + negative neg_sampled[0].
  B. steering/conditional/encoder_decoder/dataloader.reconstruct_and_select
     — Phase 4 all_sel_fids (positive flat) + Phase 5 neg_fids[neg_mask].
  C. steering/conditional/encoder_decoder/dataloader (live path)
     — per-segment fids before SegFeatures construction.
  D. steering/conditional/encoder_decoder/self_conditioned.SelfConditionedFeatureProvider.step
     — kept_ids after _apply_selection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch as t

from errors import FeatureMaskError
from steering.conditional.feature_mask import FeatureMask
from steering.refined_types import PositiveInt
from tests.steering._self_conditioned_state import build_test_self_cond_state

# ---------------------------------------------------------------------------
# Import helpers from test_live_dataloader_mask for the Site-C live tests.
# That module lives in the same directory and is not a proper package, so we
# add its directory to sys.path before importing.
# ---------------------------------------------------------------------------
_STEERING_TEST_DIR = os.path.join(os.path.dirname(__file__))
if _STEERING_TEST_DIR not in sys.path:
    sys.path.insert(0, _STEERING_TEST_DIR)

from test_live_dataloader_mask import (  # noqa: E402
    _build_threshold_pipeline as _live_build_threshold_pipeline,
)
from test_live_dataloader_mask import (  # noqa: E402
    _collect_batches as _live_collect_batches,
)
from test_live_dataloader_mask import (  # noqa: E402
    _make_dataloader as _live_make_dataloader,
)
from test_live_dataloader_mask import (  # noqa: E402
    _make_synthetic_data,
)

# ===========================================================================
# Shared constants
# ===========================================================================

N_SAE = 64
BLACKLISTED = 7  # id that must not appear


# ===========================================================================
# Site-C live fixture
# ===========================================================================


@pytest.fixture
def synthetic_env_live(tmp_path):
    """Synthetic env for Site-C live-path tests — reuses _make_synthetic_data."""
    return _make_synthetic_data(tmp_path)


# ===========================================================================
# Helpers shared across sites (mirrors test_fused_reconstruct_select.py)
# ===========================================================================


def _make_boundaries(n_features: int = N_SAE, n_buckets: int = 8) -> t.Tensor:
    n_boundaries = n_buckets - 1
    b = t.linspace(1 / n_buckets, 1.0 - 1 / n_buckets, n_boundaries, dtype=t.float64)
    return b.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_percentile_table(n_features: int = N_SAE) -> t.Tensor:
    pcts = t.linspace(0.01, 0.99, 99, dtype=t.float64)
    return pcts.unsqueeze(0).expand(n_features, -1).contiguous()


def _make_shard(
    n_rows: int = 4,
    segs_per_row: int = 2,
    feats_per_seg: int = 10,
    n_features: int = N_SAE,
    seed: int = 0,
    include_blacklisted: bool = True,
):
    """Build a minimal PackedRowShard.

    When ``include_blacklisted=True`` (default), every segment has BLACKLISTED
    mixed with other (non-masked) features so that ``apply_to_positive`` leaves
    a non-empty set and downstream selection/guard can be exercised.
    When ``include_blacklisted=False``, only safe features are present.
    """
    from steering.packed_batch_io import PackedRowShard
    torch.manual_seed(seed)
    T = 16
    total_segs = n_rows * segs_per_row
    feats_per_seg_actual = feats_per_seg
    total_feats = total_segs * feats_per_seg_actual
    tokens = t.randint(1, 500, (n_rows, T + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)
    seg_size = T // segs_per_row
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)
    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == segs_per_row - 1 else (s + 1) * seg_size
    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg_actual, feats_per_seg_actual, dtype=t.int32)

    if include_blacklisted:
        # Mix: first half BLACKLISTED, second half safe (id=1..feats_per_seg//2)
        half = feats_per_seg_actual // 2
        per_seg = (
            [BLACKLISTED] * half
            + list(range(1, feats_per_seg_actual - half + 1))
        )
        feature_ids_list = per_seg * total_segs
        feature_ids = t.tensor(feature_ids_list, dtype=t.uint16)
    else:
        # All-safe features (id=1..feats_per_seg)
        per_seg = list(range(1, feats_per_seg_actual + 1))
        feature_ids = t.tensor(per_seg * total_segs, dtype=t.uint16)

    feature_pcts = t.full((total_feats,), 200, dtype=t.uint8)
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


def _make_threshold_pipeline(
    n_features: int = N_SAE,
    n_buckets: int = 8,
    with_negatives: bool = False,
    seed: int = 0,
    feature_mask=None,
):
    from steering.conditional.feature_pipeline import (
        FeaturePipelineConfig,
        ThresholdSelectionConfig,
    )
    from steering.conditional.sampling import (
        Distribution,
        NegativeBundle,
        NegativeSamplingSelection,
        ThresholdSelection,
    )
    from steering.conditional.thresholding import ThresholdType

    selection = ThresholdSelection(threshold_type=ThresholdType.segment_topk, threshold=5.0)
    config = ThresholdSelectionConfig(selection=selection, precomputed_thresholds=None)
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric, mean=2.0, std=2.0, min_k=1,
            ),
            n_sae_features=n_features,
            rng=neg_rng,
        )

    return FeaturePipelineConfig(
        selection_config=config,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        feature_mask=feature_mask,
    )


def _make_sampling_pipeline(
    n_features: int = N_SAE,
    n_buckets: int = 8,
    with_negatives: bool = False,
    seed: int = 0,
    feature_mask=None,
):
    from steering.conditional.feature_pipeline import FeaturePipelineConfig
    from steering.conditional.sampling import (
        Distribution,
        NegativeBundle,
        NegativeSamplingSelection,
        SamplingBundle,
        SamplingSelection,
    )

    rng = t.Generator().manual_seed(seed)
    selection = SamplingSelection(
        distribution=Distribution.geometric, mean=3.0, std=3.0, min_k=1, max_k=None,
    )
    pct_table = _make_percentile_table(n_features)
    sampling_bundle = SamplingBundle(selection=selection, percentile_table=pct_table, rng=rng)
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    boundaries = _make_boundaries(n_features, effective_buckets)

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric, mean=2.0, std=2.0, min_k=1,
            ),
            n_sae_features=n_features,
            rng=neg_rng,
        )

    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        feature_mask=feature_mask,
    )


# ===========================================================================
# Site A: precompute_selected_features.select_features_for_shard
# ===========================================================================


def _build_precompute_shard_and_pipeline(tmp_path: Path, feature_mask):
    """Build a minimal shard + pipeline for precompute tests.

    The shard has segments with BLACKLISTED + other features so that
    ``apply_to_positive`` (which runs before the guard) leaves non-empty
    segments when masked. Injection happens AFTER ``apply_to_positive``.
    """
    from omegaconf import OmegaConf

    from steering.conditional.feature_pipeline import (
        build_feature_pipeline_from_cfg_block,
    )
    from steering.packed_batch_io import PackedRowShard, save_packed_row_shard
    from steering.stats_types import PerFeatureStat

    # Each segment has BLACKLISTED + a non-blacklisted feature (id=1)
    # so apply_to_positive leaves fid=1 when masked (non-empty → no early-continue)
    total_segs = 4
    feature_ids_flat = [BLACKLISTED, 1] * total_segs  # 2 features per segment
    feats_per_seg = 2
    seg_feat_offsets = [i * feats_per_seg for i in range(total_segs + 1)]
    T = 8
    shard = PackedRowShard(
        tokens=t.ones(2, T + 1, dtype=t.uint16),
        row_seg_offsets=t.tensor([0, 2, 4], dtype=t.int32),
        seg_token_start=t.zeros(total_segs, dtype=t.uint16),
        seg_token_end=t.full((total_segs,), T, dtype=t.uint16),
        seg_doc_id=t.zeros(total_segs, dtype=t.uint8),
        seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
        feature_ids=t.tensor(feature_ids_flat, dtype=t.uint16),
        feature_pcts=t.full((len(feature_ids_flat),), 200, dtype=t.uint8),
    )
    shard_dir = tmp_path / "base"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    stats = {
        str(i): PerFeatureStat(
            count=100, pct_segments_active=0.5, mean_mag=1.0,
            mean_mag_incl_zeros=0.5, std_mag=0.1, min_mag=0.0,
            max_mag=2.0, median_mag=1.0,
            percentiles=dict(full_percentiles), histogram=None,
        )
        for i in range(N_SAE)
    }
    cfg = OmegaConf.create({
        "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
        "negative_selection": {"distribution": "geometric", "mean": 2.0, "std": 2.0},
    })
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=stats,
        n_sae_features=N_SAE,
        num_buckets=8,
        seed=0,
        feature_mask=feature_mask,
    )
    return shard_path, pipeline


class TestSiteA_PrecomputeSelectedFeatures:
    """Guards in select_features_for_shard fire when injected and are silent on clean paths."""

    def test_positive_guard_fires_on_injected_blacklist(self, tmp_path):
        """Injecting BLACKLISTED into sample_features_batched output must raise FeatureMaskError.

        ``apply_to_positive`` runs BEFORE the guard and legitimately removes BLACKLISTED.
        We simulate a bug where the downstream sampler re-introduces a masked id by
        patching ``sample_features_batched`` (called AFTER apply_to_positive) to inject it.
        """
        from scripts.preprocess.precompute_selected_features import select_features_for_shard
        from steering.conditional.sampling import sample_features_batched

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        shard_path, pipeline = _build_precompute_shard_and_pipeline(tmp_path, mask)

        original_fn = sample_features_batched

        def _inject_blacklisted(all_feature_ids, all_feature_mags, segment_sizes,
                                 percentile_table, config, rng, **kwargs):
            results = original_fn(
                all_feature_ids, all_feature_mags, segment_sizes,
                percentile_table, config, rng, **kwargs,
            )
            # Inject BLACKLISTED into first segment's selected ids
            if results:
                fids, mags = results[0]
                results[0] = (
                    t.cat([fids, t.tensor([BLACKLISTED], dtype=t.int64)]),
                    t.cat([mags, t.zeros(1)]),
                )
            return results

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        with patch(
            "scripts.preprocess.precompute_selected_features.sample_features_batched",
            side_effect=_inject_blacklisted,
        ):
            with pytest.raises(FeatureMaskError):
                select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")

    def test_negative_guard_fires_on_injected_blacklist(self, tmp_path):
        """Injecting BLACKLISTED into sample_inactive_features_batched output must raise.

        Simulates a bug where the negative sampler bypasses the mask and returns a
        blacklisted id. The guard at the neg_sampled finalization point must catch it.
        """
        from scripts.preprocess.precompute_selected_features import select_features_for_shard
        from steering.conditional.sampling import sample_inactive_features_batched

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        shard_path, pipeline = _build_precompute_shard_and_pipeline(tmp_path, mask)

        original_fn = sample_inactive_features_batched

        def _inject_neg_blacklisted(active_ids_list, n_sae, selection, rng, feature_mask=None):
            results = original_fn(active_ids_list, n_sae, selection, rng, feature_mask=feature_mask)
            # Inject BLACKLISTED into the returned negative tensor (list of tensors)
            injected = []
            for r in results:
                injected.append(t.cat([r.to(t.int64), t.tensor([BLACKLISTED], dtype=t.int64)]))
            return injected

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        with patch(
            "scripts.preprocess.precompute_selected_features.sample_inactive_features_batched",
            side_effect=_inject_neg_blacklisted,
        ):
            with pytest.raises(FeatureMaskError):
                select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")

    def test_correct_blacklist_path_does_not_raise(self, tmp_path):
        """Normal blacklist path (BLACKLISTED filtered BEFORE guard) must not raise."""
        from scripts.preprocess.precompute_selected_features import select_features_for_shard

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        shard_path, pipeline = _build_precompute_shard_and_pipeline(tmp_path, mask)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        # Must not raise — the guard should see no blacklisted ids because
        # apply_to_positive + sample_inactive_features_batched already filter them.
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")

    def test_i5_no_mask_guard_not_entered(self, tmp_path):
        """With feature_mask=None, assert_no_masked is NEVER called (I5)."""
        from scripts.preprocess.precompute_selected_features import select_features_for_shard

        shard_path, pipeline = _build_precompute_shard_and_pipeline(tmp_path, None)
        assert pipeline.feature_mask is None

        call_log: list[str] = []

        # Patch FeatureMask.assert_no_masked at the class level; since pipeline.feature_mask
        # is None the patch object should never be invoked.
        original_method = FeatureMask.assert_no_masked

        def _spy(self_mask, feature_ids):
            call_log.append("called")
            return original_method(self_mask, feature_ids)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        with patch.object(FeatureMask, "assert_no_masked", side_effect=_spy):
            select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")

        assert call_log == [], (
            f"assert_no_masked was called {len(call_log)} time(s) on the None-mask path (I5 violated)"
        )


# ===========================================================================
# Site B: reconstruct_and_select — positive + negative guards
# ===========================================================================


class TestSiteB_ReconstructAndSelect:
    """Guards in reconstruct_and_select fire when injected and are silent on clean paths."""

    def test_positive_guard_fires_on_injected_blacklist(self):
        """Inject BLACKLISTED into apply_threshold output; guard must fire.

        The shard has BLACKLISTED mixed with safe features so apply_to_positive
        leaves non-empty segments. Patching apply_threshold (called AFTER the pre-filter)
        to inject BLACKLISTED simulates a downstream selection bug.
        """
        from steering.conditional.encoder_decoder.dataloader import reconstruct_and_select
        from steering.conditional.thresholding import apply_threshold

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        # Use mixed shard so pre-filter leaves non-empty segments
        shard = _make_shard(n_rows=2, segs_per_row=2, feats_per_seg=8, n_features=N_SAE,
                            include_blacklisted=True)
        pipeline = _make_threshold_pipeline(feature_mask=mask, with_negatives=False)

        original_fn = apply_threshold

        def _inject(config, feature_ids, feature_mags, precomputed=None):
            fids, mags = original_fn(config, feature_ids, feature_mags, precomputed)
            # Always inject BLACKLISTED to guarantee the guard fires
            return t.cat([fids, t.tensor([BLACKLISTED], dtype=t.int64)]), t.cat([mags, t.zeros(1)])

        with patch(
            "steering.conditional.encoder_decoder.dataloader.apply_threshold",
            side_effect=_inject,
        ):
            with pytest.raises(FeatureMaskError):
                reconstruct_and_select(shard, list(range(2)), pipeline)

    def test_negative_guard_fires_on_injected_blacklist(self):
        """Prove the negative guard fires: patch assert_no_masked to raise, confirm it's called.

        Strategy: use a shard with only non-blacklisted features so the POSITIVE guard
        (all_sel_fids) passes cleanly (no blacklisted ids there). Then patch
        ``assert_no_masked`` to track distinct calls by the tensor it receives. On the
        first call (positive guard), it succeeds normally. On the second call (negative
        guard), it raises. This proves both guards are wired in sequence.

        Implementation: count calls to assert_no_masked; raise on the 2nd call (which
        is the negative path since positive is first).
        """
        from steering.conditional.encoder_decoder.dataloader import reconstruct_and_select

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        # Safe shard so positive guard sees no blacklisted ids
        shard = _make_shard(n_rows=2, segs_per_row=2, feats_per_seg=8, n_features=N_SAE,
                            include_blacklisted=False)
        pipeline = _make_threshold_pipeline(feature_mask=mask, with_negatives=True)

        call_count = [0]
        original_method = FeatureMask.assert_no_masked

        def _count_and_raise_on_second(self_mask, feature_ids):
            call_count[0] += 1
            if call_count[0] >= 2:
                # 2nd call = negative guard — inject a failure
                raise FeatureMaskError("injected: blacklisted id in negative ids")
            return original_method(self_mask, feature_ids)

        with patch.object(FeatureMask, "assert_no_masked", _count_and_raise_on_second):
            with pytest.raises(FeatureMaskError, match="negative"):
                reconstruct_and_select(shard, list(range(2)), pipeline)

        # Confirm BOTH guards were reached (positive passed, negative raised)
        assert call_count[0] >= 2, (
            f"assert_no_masked called only {call_count[0]} time(s); "
            "expected >=2 (positive guard + negative guard)"
        )

    def test_correct_blacklist_path_does_not_raise(self):
        """Normal blacklist path (correct apply_to_positive + neg universe) must not raise."""
        from steering.conditional.encoder_decoder.dataloader import reconstruct_and_select

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        shard = _make_shard(n_rows=2, segs_per_row=2, feats_per_seg=8, n_features=N_SAE)
        pipeline = _make_threshold_pipeline(feature_mask=mask, with_negatives=True)
        # Must not raise
        reconstruct_and_select(shard, list(range(2)), pipeline)

    def test_i5_no_mask_guard_not_entered(self):
        """With feature_mask=None, assert_no_masked is NEVER called (I5)."""
        from steering.conditional.encoder_decoder.dataloader import reconstruct_and_select

        shard = _make_shard(n_rows=2, segs_per_row=2, feats_per_seg=8, n_features=N_SAE)
        pipeline = _make_threshold_pipeline(feature_mask=None, with_negatives=True)
        assert pipeline.feature_mask is None

        call_log: list[str] = []
        original_method = FeatureMask.assert_no_masked

        def _spy(self_mask, feature_ids):
            call_log.append("called")
            return original_method(self_mask, feature_ids)

        with patch.object(FeatureMask, "assert_no_masked", side_effect=_spy):
            reconstruct_and_select(shard, list(range(2)), pipeline)

        assert call_log == [], (
            f"assert_no_masked called {len(call_log)} time(s) on None-mask path (I5 violated)"
        )


# ===========================================================================
# Site C: live dataloader per-segment guard
# ===========================================================================


class TestSiteC_LiveDataloader:
    """Guard in live dataloader (_document_batches_with_features line 714) fires when
    injection bypasses filtering on the REAL live loader path (no fused shards).

    Path reasoning: _live_make_dataloader calls encoder_decoder_data_loader_with_state
    with prefetch_batches=0 and doc_prefetch_depth=0, which routes through
    _encoder_decoder_data_loader_inner → _document_batches_with_features.  The
    pipeline uses ThresholdSelectionConfig (not SamplingBundle), so use_sampling=False
    and the ThresholdSelectionConfig branch at dataloader.py:571 calls apply_threshold
    for each segment.  After discretization, line 714:
        if pipeline.feature_mask is not None:
            pipeline.feature_mask.assert_no_masked(fids.to(t.int64))
    is the guard exercised here.  reconstruct_and_select is NOT involved — that is
    only called by PrecomputedBatchLoader when packed-row shards are present.
    """

    def test_live_direct_guard_fires_on_injected_blacklist(self, synthetic_env_live):
        """Fault-injection: apply_threshold patched to inject BLACKLISTED into output;
        the live line-714 guard must fire with FeatureMaskError.

        This test is NOT satisfied by reconstruct_and_select: it drives the REAL
        _document_batches_with_features path via the fully-assembled synthetic
        parquet+feature-shard env.  If line 714 were deleted this test would NOT
        raise (the guard is gone) and pytest.raises would fail.
        """
        from steering.conditional.thresholding import apply_threshold

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        # ThresholdSelection pipeline — routes to apply_threshold branch (not sampling)
        pipeline = _live_build_threshold_pipeline(feature_mask=mask)

        original_fn = apply_threshold

        def _inject(config, feature_ids, feature_mags, precomputed=None):
            fids, mags = original_fn(config, feature_ids, feature_mags, precomputed)
            # Force BLACKLISTED into returned fids regardless of selection output.
            # Shapes: fids 1-D int64, mags 1-D float32.
            return (
                t.cat([fids, t.tensor([BLACKLISTED], dtype=t.int64)]),
                t.cat([mags, t.zeros(1, dtype=t.float32)]),
            )

        with patch(
            "steering.conditional.encoder_decoder.dataloader.apply_threshold",
            side_effect=_inject,
        ):
            with pytest.raises(FeatureMaskError):
                gen = _live_make_dataloader(synthetic_env_live, pipeline=pipeline)
                _live_collect_batches(gen, n_batches=2)

    def test_live_direct_correct_blacklist_does_not_raise(self, synthetic_env_live):
        """Companion: same live pipeline + blacklist, NO injection → must NOT raise and
        must produce ≥ 1 batch.  Proves the guard has no false positives on the real
        live path and that the fixture actually exercises _document_batches_with_features
        (line 714 is reachable but sees no blacklisted ids after apply_to_positive).
        """
        mask = FeatureMask("blacklist", (BLACKLISTED,))
        pipeline = _live_build_threshold_pipeline(feature_mask=mask)
        gen = _live_make_dataloader(synthetic_env_live, pipeline=pipeline)
        batches = _live_collect_batches(gen, n_batches=2)
        assert len(batches) >= 1, (
            "live masked pipeline produced zero batches — fixture is broken"
        )

    def test_live_i5_no_mask_guard_not_entered(self, synthetic_env_live):
        """I5 on the LIVE path: feature_mask=None → assert_no_masked NEVER called.

        Drives _document_batches_with_features with feature_mask=None and a spy on
        FeatureMask.assert_no_masked.  Line 714 is strictly gated by
        ``if pipeline.feature_mask is not None:`` so the spy must see 0 calls.
        """
        pipeline = _live_build_threshold_pipeline(feature_mask=None)
        assert pipeline.feature_mask is None

        call_log: list[str] = []
        original_method = FeatureMask.assert_no_masked

        def _spy(self_mask, feature_ids):
            call_log.append("called")
            return original_method(self_mask, feature_ids)

        gen = _live_make_dataloader(synthetic_env_live, pipeline=pipeline)
        with patch.object(FeatureMask, "assert_no_masked", side_effect=_spy):
            _live_collect_batches(gen, n_batches=2)

        assert call_log == [], (
            f"assert_no_masked called {len(call_log)} time(s) on None-mask LIVE path "
            f"(I5 violated — line-714 guard is not properly gated by feature_mask is not None)"
        )


# ===========================================================================
# Site D: SelfConditionedFeatureProvider.step — kept_ids guard
# ===========================================================================


def _make_selfcond_config(feature_mask=None, n_sae: int | None = None):
    from steering.conditional.encoder_decoder.config import (
        FeatureBucketProducer,
        FeaturePredLossMode,
        FeatureSelectionProducer,
    )
    from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
    from steering.conditional.feature_pipeline import ThresholdSelectionConfig
    from steering.conditional.sampling import ThresholdSelection
    from steering.conditional.thresholding import ThresholdType
    from tests.encdec_test_helpers import N_SAE_FEATURES as _N_SAE
    from tests.encdec_test_helpers import NUM_BUCKETS as _NUM_BUCKETS

    if n_sae is None:
        n_sae = _N_SAE  # match the model built by build_model_with_feature_prediction

    selection = ThresholdSelectionConfig(
        selection=ThresholdSelection(threshold_type=ThresholdType.segment_topk, threshold=5.0),
        precomputed_thresholds=None,
    )
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(8),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=selection,
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=_NUM_BUCKETS,  # must match model's encoder embedding table size
        n_sae_features=PositiveInt.parse(n_sae),
        feature_mask=feature_mask,
    )


class TestSiteD_SelfConditionedProvider:
    """Guard in SelfConditionedFeatureProvider.step fires when kept_ids contains a masked id."""

    def test_guard_fires_on_injected_blacklist(self, tmp_path):
        """Patch _apply_selection to inject BLACKLISTED; guard must raise FeatureMaskError."""
        import nanochat.flash_attention as _fa_mod
        from nanochat.flash_attention import FlashAttnImpl
        from steering.conditional.encoder_decoder.self_conditioned import (
            SelfConditionedFeatureProvider,
        )

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        cfg = _make_selfcond_config(feature_mask=mask)

        # We need a model to instantiate the provider; use the test helper
        try:
            from tests.encdec_test_helpers import build_model_with_feature_prediction
        except ImportError:
            pytest.skip("encdec_test_helpers not available")

        torch.manual_seed(0)
        model = build_model_with_feature_prediction()

        provider = SelfConditionedFeatureProvider(model=model, cfg=cfg, device=torch.device("cpu"))

        # Patch _apply_selection to inject BLACKLISTED into kept_ids
        original_apply = provider._apply_selection

        def _inject(feature_ids, scores):
            fids, scores_out = original_apply(feature_ids, scores)
            # Force inject BLACKLISTED even if mask would have blocked it
            return (
                torch.cat([fids, torch.tensor([BLACKLISTED], dtype=fids.dtype, device=fids.device)]),
                torch.cat([scores_out, torch.zeros(1, dtype=scores_out.dtype, device=scores_out.device)]),
            )

        state = build_test_self_cond_state(provider)
        fp_logits = torch.full((1, 8, cfg.n_sae_features), 5.0)

        old_impl = _fa_mod._override_impl
        _fa_mod._override_impl = FlashAttnImpl.SDPA
        try:
            with patch.object(provider, "_apply_selection", side_effect=_inject):
                with pytest.raises(FeatureMaskError):
                    provider.step(state, fp_logits)
        finally:
            _fa_mod._override_impl = old_impl

    def test_correct_blacklist_path_does_not_raise(self):
        """Normal blacklist path in step (FP head strongly predicts BLACKLISTED) must not raise."""
        import nanochat.flash_attention as _fa_mod
        from nanochat.flash_attention import FlashAttnImpl
        from steering.conditional.encoder_decoder.self_conditioned import (
            SelfConditionedFeatureProvider,
        )

        try:
            from tests.encdec_test_helpers import build_model_with_feature_prediction
        except ImportError:
            pytest.skip("encdec_test_helpers not available")

        torch.manual_seed(0)
        model = build_model_with_feature_prediction()

        mask = FeatureMask("blacklist", (BLACKLISTED,))
        cfg = _make_selfcond_config(feature_mask=mask)
        provider = SelfConditionedFeatureProvider(model=model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        # Strongly predict BLACKLISTED — the filter should remove it before the guard
        fp_logits = torch.full((1, 8, cfg.n_sae_features), 1.0)
        fp_logits[0, :, BLACKLISTED] = 100.0

        old_impl = _fa_mod._override_impl
        _fa_mod._override_impl = FlashAttnImpl.SDPA
        try:
            # Must not raise
            provider.step(state, fp_logits)
        finally:
            _fa_mod._override_impl = old_impl

    def test_i5_no_mask_guard_not_entered(self):
        """With feature_mask=None, assert_no_masked is never called from step (I5)."""
        import nanochat.flash_attention as _fa_mod
        from nanochat.flash_attention import FlashAttnImpl
        from steering.conditional.encoder_decoder.self_conditioned import (
            SelfConditionedFeatureProvider,
        )

        try:
            from tests.encdec_test_helpers import build_model_with_feature_prediction
        except ImportError:
            pytest.skip("encdec_test_helpers not available")

        torch.manual_seed(0)
        model = build_model_with_feature_prediction()

        cfg = _make_selfcond_config(feature_mask=None)
        assert cfg.feature_mask is None

        provider = SelfConditionedFeatureProvider(model=model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)
        fp_logits = torch.full((1, 8, cfg.n_sae_features), 1.0)

        call_log: list[str] = []
        original_method = FeatureMask.assert_no_masked

        def _spy(self_mask, feature_ids):
            call_log.append("called")
            return original_method(self_mask, feature_ids)

        old_impl = _fa_mod._override_impl
        _fa_mod._override_impl = FlashAttnImpl.SDPA
        try:
            with patch.object(FeatureMask, "assert_no_masked", side_effect=_spy):
                provider.step(state, fp_logits)
        finally:
            _fa_mod._override_impl = old_impl

        assert call_log == [], (
            f"assert_no_masked called {len(call_log)} time(s) on None-mask path (I5 violated)"
        )
