"""Tests for feature_mask enforcement in select_features_for_shard.

Mirrored fixture approach from TestEndToEndPipelineParity._build_base_shard_and_pipeline
in tests/steering/test_precomputed_batches_integration.py.

Verifies:
- Blacklisted feature K absent from BOTH sel_fids AND neg_fids.
- Output is non-empty (other features still selected/sampled).
- No-mask path produces a valid SelectionShard (sanity / regression).

The test uses seed=1 which is verified (via diagnostic run) to produce K=7
in BOTH sel_fids and neg_fids in the no-mask case, making the blacklist tests
meaningful and reliably failing before the implementation is in place.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.preprocess import precompute_selected_features as precompute_mod
from scripts.preprocess.precompute_selected_features import (
    _lookup_selected_percentiles,
    assert_precompute_supports_pipeline,
    select_features_for_shard,
)
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.overlay_io import overlay_shard_path
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    load_selection_shard,
    save_packed_row_shard,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of SAE features — large enough that K is valid inside the universe
N_SAE_FEATURES = 64
# The blacklisted feature id — present in seg 0 of the base shard (fid=7 for f=7)
# and verified (seed=1) to appear in BOTH sel_fids and neg_fids in the no-mask case.
_K = 7
# Number of segments in the tiny base shard
_N_SEGS = 6
# Features per segment (deterministic: fid = (seg_idx * _N_FEATS_PER_SEG + f) % N_SAE)
_N_FEATS_PER_SEG = 8
# Seed that guarantees K=7 appears in sel_fids AND neg_fids in the no-mask case.
# Verified via diagnostic run before writing these tests.
_SEED_WITH_K = 1


# ---------------------------------------------------------------------------
# Fixture helpers (mirrored from test_build_feature_pipeline_from_cfg_block)
# ---------------------------------------------------------------------------

def _make_per_feature_stats(n: int = N_SAE_FEATURES) -> dict[str, PerFeatureStat]:
    """Build a tiny stats dict with deterministic percentiles (mirrors
    the identical helper in test_build_feature_pipeline_from_cfg_block.py)."""
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


def _build_base_shard(tmp_path: Path) -> tuple[PackedRowShard, Path]:
    """Build a minimal PackedRowShard with deterministic feature_ids including _K.

    Each of the _N_SEGS segments carries _N_FEATS_PER_SEG features whose ids are:
        fid = (seg_idx * _N_FEATS_PER_SEG + f) % N_SAE_FEATURES

    This ensures _K=7 appears in seg_idx=0 at f=7.  Percentiles are set to
    200/255 so the sampling path treats them as high-ranked.

    Returns (shard, shard_path).
    """
    T = 16  # context length (tokens shape: T+1)
    n_rows = 2
    segs_per_row = _N_SEGS // n_rows  # 3

    all_fids: list[int] = []
    all_pcts: list[int] = []
    feat_counts: list[int] = []

    for seg_idx in range(_N_SEGS):
        for f in range(_N_FEATS_PER_SEG):
            fid = (seg_idx * _N_FEATS_PER_SEG + f) % N_SAE_FEATURES
            all_fids.append(fid)
            all_pcts.append(200)  # high percentile → likely sampled
        feat_counts.append(_N_FEATS_PER_SEG)

    seg_feat_offsets = [0]
    for c in feat_counts:
        seg_feat_offsets.append(seg_feat_offsets[-1] + c)

    # row_seg_offsets: 2 rows, each owning segs_per_row segments
    row_seg_offsets = [0, segs_per_row, _N_SEGS]

    # tokens: shape (n_rows, T+1), fill with arbitrary non-zero values
    tokens = t.ones(n_rows, T + 1, dtype=t.uint16)

    shard = PackedRowShard(
        tokens=tokens,
        row_seg_offsets=t.tensor(row_seg_offsets, dtype=t.int32),
        seg_token_start=t.zeros(_N_SEGS, dtype=t.uint16),
        seg_token_end=t.full((_N_SEGS,), T, dtype=t.uint16),
        seg_doc_id=t.zeros(_N_SEGS, dtype=t.uint8),
        seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
        feature_ids=t.tensor(all_fids, dtype=t.uint16),
        feature_pcts=t.tensor(all_pcts, dtype=t.uint8),
    )

    shard_dir = tmp_path / "base"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")
    return shard, shard_path


def _build_pipeline(mask: FeatureMask | None, seed: int = _SEED_WITH_K) -> FeaturePipelineConfig:
    """Build a FeaturePipelineConfig via build_feature_pipeline_from_cfg_block
    with a geometric-sampling feature_selection and negative_selection, using
    the _make_per_feature_stats fixture.

    ``mask`` is passed directly as the ``feature_mask`` keyword arg; when None
    the no-mask (legacy) path is exercised.
    """
    cfg = OmegaConf.create({
        "feature_selection": {
            "distribution": "geometric",
            "mean": 4.0,
            "std": 4.0,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 3.0,
            "std": 3.0,
        },
    })
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=8,
        seed=seed,
        feature_mask=mask,
    )


def _build_percentile_threshold_pipeline(
    *,
    threshold: float = 0.5,
    num_buckets: int = 4,
) -> FeaturePipelineConfig:
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile",
            "threshold": threshold,
        },
        "negative_selection": None,
    })
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=num_buckets,
        seed=0,
        feature_mask=None,
    )


def _build_percentile_segment_topk_pipeline(
    *,
    threshold: float = 1.9,
    num_buckets: int = 4,
) -> FeaturePipelineConfig:
    cfg = OmegaConf.create({
        "feature_selection": {
            "threshold_type": "feature_percentile_segment_topk",
            "threshold": threshold,
        },
        "negative_selection": None,
    })
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=num_buckets,
        seed=0,
        feature_mask=None,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _collect_all_fids(sel: SelectionShard) -> set[int]:
    """Union of all sel_fids and neg_fids across all segments."""
    result: set[int] = set()
    result.update(sel.sel_fids.tolist())
    result.update(sel.neg_fids.tolist())
    return result


def _add_delta_bundle(pipeline: FeaturePipelineConfig) -> FeaturePipelineConfig:
    pipeline.delta_bundle = DeltaBundle(
        config=FeatureDeltaConfig(mode=DeltaMode.prev_segment),
        percentile_table=t.zeros(N_SAE_FEATURES, 99, dtype=t.float64),
    )
    return pipeline


# ---------------------------------------------------------------------------
# Test 0 — feature_delta is rejected before shard skip/output logic
# ---------------------------------------------------------------------------


class TestFeatureDeltaUnsupported:
    """Selection shards cannot encode live document-history delta state."""

    def test_guard_rejects_delta_pipeline(self) -> None:
        pipeline = _add_delta_bundle(_build_pipeline(None))

        with pytest.raises(ValueError, match="does not support feature_delta"):
            assert_precompute_supports_pipeline(pipeline)

    def test_main_rejects_delta_before_selection_hash_and_skip_logic(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()

        pipeline = _add_delta_bundle(_build_pipeline(None))
        stats = SimpleNamespace(
            metadata=SimpleNamespace(n_sae_features=N_SAE_FEATURES),
            per_feature_stats=_make_per_feature_stats(),
        )

        monkeypatch.setattr(precompute_mod, "verify_segment_size", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(precompute_mod, "find_statistics_path", lambda _feature_dir: tmp_path / "stats")
        monkeypatch.setattr(precompute_mod, "load_statistics", lambda _stats_path: stats)
        monkeypatch.setattr(
            precompute_mod,
            "build_feature_pipeline_from_cfg_block",
            lambda *_args, **_kwargs: pipeline,
        )

        def _unexpected_hash(*_args, **_kwargs):
            pytest.fail("selection hash should not be computed for feature_delta precompute")

        monkeypatch.setattr(precompute_mod, "compute_selection_hash", _unexpected_hash)

        cfg = OmegaConf.create({
            "packed_row_dir": str(packed_row_dir),
            "feature_dir": str(feature_dir),
            "segment_size": 128,
            "feature_selection": {
                "distribution": "geometric",
                "mean": 4.0,
                "std": 4.0,
                "delta_weights": "absolute_percentile",
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
            },
            "feature_delta": {
                "mode": "prev_segment",
                "neg_delta_ratio": 1.0,
            },
            "feature_mask": None,
            "seed": 42,
            "num_buckets": 8,
        })

        with pytest.raises(ValueError, match="does not support feature_delta"):
            precompute_mod.main.__wrapped__(cfg)


class TestFrequencyBlacklistProducer:
    def test_main_hashes_resolved_frequency_mask(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        _shard, _shard_path = _build_base_shard(tmp_path)
        packed_row_dir = tmp_path / "base"
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        stats = SimpleNamespace(
            metadata=SimpleNamespace(n_sae_features=N_SAE_FEATURES),
            per_feature_stats=_make_per_feature_stats(),
        )
        seen: dict[str, object] = {}

        def _fake_freq_blacklist(cfg, _feature_dir, _n_sae_features):
            cfg.feature_mask = {
                "mode": "blacklist",
                "ids": [_K],
            }

        def _record_hash(*_args, **kwargs):
            seen["feature_mask"] = kwargs["feature_mask"]
            return "f" * 64

        monkeypatch.setattr(precompute_mod, "verify_segment_size", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(precompute_mod, "find_statistics_path", lambda _feature_dir: tmp_path / "stats")
        monkeypatch.setattr(precompute_mod, "load_statistics", lambda _stats_path: stats)
        monkeypatch.setattr(
            precompute_mod,
            "maybe_apply_frequency_blacklist",
            _fake_freq_blacklist,
        )
        monkeypatch.setattr(precompute_mod, "compute_selection_hash", _record_hash)

        cfg = OmegaConf.create({
            "packed_row_dir": str(packed_row_dir),
            "feature_dir": str(feature_dir),
            "segment_size": 128,
            "feature_selection": {
                "threshold_type": "feature_percentile",
                "threshold": 0.5,
            },
            "negative_selection": None,
            "feature_delta": None,
            "feature_mask": None,
            "seed": 42,
            "num_buckets": 8,
        })

        precompute_mod.main.__wrapped__(cfg)

        assert seen["feature_mask"] == {
            "mode": "blacklist",
            "ids": [_K],
        }


# ---------------------------------------------------------------------------
# Test 1 — blacklist mask: K absent from sel_fids AND neg_fids
#
# These tests use _SEED_WITH_K=1, which (verified empirically) produces K=7
# in BOTH sel_fids and neg_fids when no mask is present.  With the blacklist
# mask, K must be absent from both — these tests fail before implementation.
# ---------------------------------------------------------------------------


class TestFeatureMaskBlacklist:
    """Blacklisted feature K must not appear in any output (positive or negative)."""

    def test_blacklisted_feature_absent_from_sel_fids(self, tmp_path):
        """K is blacklisted → must not appear in sel_fids.

        Uses seed=1, which is verified to include K=7 in sel_fids in the
        no-mask case, so this test fails before the masking is implemented.
        """
        shard, shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        pipeline = _build_pipeline(mask, seed=_SEED_WITH_K)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        assert _K not in set(sel.sel_fids.tolist()), (
            f"Blacklisted feature {_K} found in sel_fids"
        )

    def test_blacklisted_feature_absent_from_neg_fids(self, tmp_path):
        """K is blacklisted → must not appear in neg_fids.

        Uses seed=1, which is verified to include K=7 in neg_fids in the
        no-mask case, so this test fails before the masking is implemented.
        """
        shard, shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        pipeline = _build_pipeline(mask, seed=_SEED_WITH_K)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        assert _K not in set(sel.neg_fids.tolist()), (
            f"Blacklisted feature {_K} found in neg_fids"
        )

    def test_combined_all_fids_exclude_K(self, tmp_path):
        """Union of sel_fids + neg_fids must not contain K.

        This is the stronger combined assertion: K cannot appear in any
        output regardless of which pathway (positive or negative) it came
        from. Uses seed=1 to guarantee K would appear without masking.
        """
        shard, shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        pipeline = _build_pipeline(mask, seed=_SEED_WITH_K)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        combined = _collect_all_fids(sel)
        assert _K not in combined, (
            f"Blacklisted feature {_K} found in combined (sel+neg) fids: {combined}"
        )

    def test_output_is_non_empty(self, tmp_path):
        """With a single-feature blacklist, other features still make it through."""
        shard, shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        pipeline = _build_pipeline(mask, seed=_SEED_WITH_K)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        assert sel.total_selected > 0 or sel.total_negative > 0, (
            "Selection shard is empty — all features were masked out"
        )

    def test_segment_count_matches_base(self, tmp_path):
        """sel_offsets length - 1 must equal total_segments of the base shard."""
        shard, shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        pipeline = _build_pipeline(mask, seed=_SEED_WITH_K)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        assert sel.total_segments == shard.total_segments, (
            f"sel.total_segments={sel.total_segments} != "
            f"shard.total_segments={shard.total_segments}"
        )


# ---------------------------------------------------------------------------
# Test 2 — no-mask path: legacy behavior (K MAY appear, no crash)
#
# This is the regression / sanity test: the no-mask path must still work
# identically to the pre-mask behavior (I1 invariant).
# ---------------------------------------------------------------------------


class TestNoMaskLegacyBehavior:
    """When feature_mask is None, behavior is unchanged (no crash, valid shard)."""

    def test_no_mask_produces_valid_selection_shard(self, tmp_path):
        """No-mask path runs without error and yields a valid SelectionShard."""
        shard, shard_path = _build_base_shard(tmp_path)
        pipeline = _build_pipeline(None, seed=0)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        out_path = overlay_shard_path(out_dir, "shard_00000")

        assert out_path.exists(), "Selection shard file not written"
        sel = load_selection_shard(out_path)
        assert isinstance(sel, SelectionShard)
        assert sel.total_segments == shard.total_segments

    def test_no_mask_pipeline_feature_mask_is_none(self, tmp_path):
        """pipeline.feature_mask must be None when no mask is configured."""
        pipeline = _build_pipeline(None, seed=0)
        assert pipeline.feature_mask is None

    def test_no_mask_k_appears_with_known_seed(self, tmp_path):
        """Sanity: with seed=1 and no mask, K=7 appears in sel+neg (confirms
        the blacklist tests are testing something real).

        If this test fails it means the fixture changed and _SEED_WITH_K needs
        recalibration — not a bug in the mask code.
        """
        shard, shard_path = _build_base_shard(tmp_path)
        pipeline = _build_pipeline(None, seed=_SEED_WITH_K)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        combined = _collect_all_fids(sel)
        # K=7 must appear in the no-mask case with _SEED_WITH_K — this validates
        # that the blacklist tests are exercising a real code path.
        assert _K in combined, (
            f"Fixture invariant broken: K={_K} not found in combined fids with "
            f"seed={_SEED_WITH_K}. Recalibrate _SEED_WITH_K."
        )


class TestPackedPercentileThresholds:
    def test_threshold_precompute_buckets_from_packed_percentiles(self, tmp_path):
        """Packed rows have no raw magnitudes, so buckets come from percentile bytes."""
        shard, shard_path = _build_base_shard(tmp_path)
        pipeline = _build_percentile_threshold_pipeline(threshold=0.5, num_buckets=4)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))

        assert sel.total_selected == int(shard.feature_ids.numel())
        assert set(sel.sel_pcts.tolist()) == {200}
        assert set(sel.sel_buckets.tolist()) == {3}

    def test_segment_topk_preserves_zero_count_no_pass_segments(self, tmp_path) -> None:
        """No-pass percentile rows are a valid reference-path empty selection."""
        shard, _shard_path = _build_base_shard(tmp_path)
        offsets = shard.seg_feat_offsets.tolist()
        pcts = shard.feature_pcts.clone()
        pcts[offsets[0]:offsets[1]] = 0
        pcts[offsets[1]:] = 255
        mixed_shard = PackedRowShard(
            tokens=shard.tokens,
            row_seg_offsets=shard.row_seg_offsets,
            seg_token_start=shard.seg_token_start,
            seg_token_end=shard.seg_token_end,
            seg_doc_id=shard.seg_doc_id,
            seg_feat_offsets=shard.seg_feat_offsets,
            feature_ids=shard.feature_ids,
            feature_pcts=pcts,
        )
        mixed_dir = tmp_path / "mixed"
        mixed_dir.mkdir()
        mixed_path = save_packed_row_shard(mixed_shard, mixed_dir / "shard_00000")
        pipeline = _build_percentile_segment_topk_pipeline(threshold=1.5)
        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)

        out_dir = tmp_path / "sel"
        out_dir.mkdir()
        select_features_for_shard(mixed_path, pipeline, out_dir, "shard_00000")
        sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
        sel_counts = (sel.sel_offsets[1:] - sel.sel_offsets[:-1]).tolist()

        assert sel.total_segments == mixed_shard.total_segments
        assert sel_counts[0] == 0
        assert all(count == 1 for count in sel_counts[1:])

    def test_script_wrapper_is_a_pure_valid_threshold_delegate(
        self, monkeypatch
    ) -> None:
        pipeline = _build_percentile_segment_topk_pipeline(threshold=3.57)
        assert isinstance(pipeline.selection_config, ThresholdSelectionConfig)
        fids = t.tensor([7, 4, 9, 2], dtype=t.int64)
        pcts = t.tensor([0.58, 0.91, 0.57, 0.29], dtype=t.float32)
        expected = t.tensor([4, 7, 9], dtype=t.int64)

        def frozen_delegate(*args) -> t.Tensor:
            assert len(args) == 4
            assert args[0] is fids
            assert args[1] is pcts
            assert args[2] is pipeline.selection_config.selection.threshold_type  # ty: ignore[unresolved-attribute]
            assert args[3] == 3.57
            return expected

        monkeypatch.setattr(
            precompute_mod,
            "select_from_packed_percentiles",
            frozen_delegate,
        )

        actual = precompute_mod._select_from_packed_percentiles(
            fids=fids,
            pcts=pcts,
            selection=pipeline.selection_config,
        )

        assert actual is expected

    def test_lookup_selected_percentiles_rejects_empty_selection(self) -> None:
        with pytest.raises(ValueError, match="requires at least one selected feature"):
            _lookup_selected_percentiles(
                selected_fids=t.tensor([], dtype=t.int64),
                candidate_fids=t.tensor([1, 2, 3], dtype=t.int64),
                candidate_pcts=t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
            )

    def test_lookup_selected_percentiles_rejects_duplicate_candidate_feature(self) -> None:
        with pytest.raises(ValueError, match="selected feature id 1"):
            _lookup_selected_percentiles(
                selected_fids=t.tensor([1], dtype=t.int64),
                candidate_fids=t.tensor([1, 1, 3], dtype=t.int64),
                candidate_pcts=t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
            )

    def test_lookup_selected_percentiles_rejects_missing_selected_feature(self) -> None:
        with pytest.raises(ValueError, match="selected feature id 99"):
            _lookup_selected_percentiles(
                selected_fids=t.tensor([1, 99], dtype=t.int64),
                candidate_fids=t.tensor([1, 2, 3], dtype=t.int64),
                candidate_pcts=t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
            )

    def test_sorted_lookup_exactly_matches_frozen_reference_at_real_scale(
        self, monkeypatch
    ) -> None:
        candidate_fids = t.arange(4096, dtype=t.int64) * 2
        candidate_pcts = t.linspace(
            -0.25,
            1.25,
            4096,
            dtype=t.float64,
        )
        selected_fids = t.tensor(
            [8190, 0, 2048, 74, 74, 6000, 2],
            dtype=t.int64,
        )
        expected = t.tensor(
            [
                float(
                    candidate_pcts[
                        (candidate_fids == feature_id).nonzero(as_tuple=True)[0][0]
                    ].item()
                )
                for feature_id in selected_fids.tolist()
            ],
            dtype=t.float32,
        )
        real_searchsorted = t.searchsorted
        calls = 0

        def recording_searchsorted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_searchsorted(*args, **kwargs)

        monkeypatch.setattr(t, "searchsorted", recording_searchsorted)
        actual = _lookup_selected_percentiles(
            selected_fids=selected_fids,
            candidate_fids=candidate_fids,
            candidate_pcts=candidate_pcts,
        )
        assert calls == 1
        assert actual.device.type == "cpu"
        assert actual.dtype == t.float32
        assert t.equal(actual, expected)

    def test_unsorted_candidates_preserve_reference_fallback(
        self, monkeypatch
    ) -> None:
        def forbidden_searchsorted(*_args, **_kwargs):
            raise AssertionError("unsorted candidates used the sorted fast path")

        monkeypatch.setattr(t, "searchsorted", forbidden_searchsorted)
        actual = _lookup_selected_percentiles(
            selected_fids=t.tensor([5, 1], dtype=t.int64),
            candidate_fids=t.tensor([5, 3, 1], dtype=t.int64),
            candidate_pcts=t.tensor([0.5, 0.3, 0.1], dtype=t.float64),
        )
        assert t.equal(actual, t.tensor([0.5, 0.1], dtype=t.float32))

    def test_unselected_duplicate_candidates_preserve_reference_behavior(self) -> None:
        actual = _lookup_selected_percentiles(
            selected_fids=t.tensor([3], dtype=t.int64),
            candidate_fids=t.tensor([1, 1, 3], dtype=t.int64),
            candidate_pcts=t.tensor([0.1, 0.2, 0.3], dtype=t.float32),
        )
        assert t.equal(actual, t.tensor([0.3], dtype=t.float32))

    def test_empty_candidates_report_first_selected_feature(self) -> None:
        with pytest.raises(ValueError, match="feature id 7 has 0 matching"):
            _lookup_selected_percentiles(
                selected_fids=t.tensor([7, 8], dtype=t.int64),
                candidate_fids=t.tensor([], dtype=t.int64),
                candidate_pcts=t.tensor([], dtype=t.float32),
            )
