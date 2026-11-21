"""Invariant tests for delta conditioning that previously lacked coverage.

Scope:
- build_feature_pipeline cross-validation between feature_delta and
  SamplingSelection.delta_weights.
- compute_doc_deltas invariants (subset, positive-only, pct-range) checked
  directly rather than through equivalence-to-reference.
- Cross-document reset: a fresh compute_doc_deltas call starts from zero ref.
- sample_delta_negatives exclusion semantics (raw, ref, positives).
"""

from __future__ import annotations

import os

import pytest
import torch as t

from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
    compute_doc_deltas,
    sample_delta_negatives,
)
from steering.conditional.feature_pipeline import (
    build_feature_pipeline,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stat() -> PerFeatureStat:
    """Synthesize a PerFeatureStat with monotonic p1..p99 values."""
    percentiles = {f"p{k}": float(k) for k in range(1, 100)}
    return PerFeatureStat(
        count=1000, pct_segments_active=0.5,
        mean_mag=50.0, mean_mag_incl_zeros=25.0,
        std_mag=1.0, min_mag=1.0, max_mag=99.0,
        median_mag=50.0, percentiles=percentiles,
    )


def _stats_dict(n: int) -> dict[str, PerFeatureStat]:
    return {str(i): _make_stat() for i in range(n)}


# ---------------------------------------------------------------------------
# build_feature_pipeline cross-validation
# ---------------------------------------------------------------------------


class TestBuildFeaturePipelineDeltaCrossValidation:
    """Confirms SamplingSelection and feature_delta are validated together."""

    N_SAE = 16

    def _sampling_without_delta_weights(self) -> SamplingSelection:
        return SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=1,
            max_k=8,
            mode=SamplingMode.top_k_percentile,
            delta_weights=None,
        )

    def _sampling_with_delta_weights(self) -> SamplingSelection:
        return SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=1,
            max_k=8,
            mode=SamplingMode.top_k_percentile,
            delta_weights=DeltaSamplingWeights.absolute_percentile,
        )

    def test_feature_delta_without_delta_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="delta_weights"):
            build_feature_pipeline(
                feature_selection=self._sampling_without_delta_weights(),
                negative_selection=None,
                per_feature_stats=_stats_dict(self.N_SAE),
                n_sae_features=self.N_SAE,
                num_buckets=4,
                feature_delta=FeatureDeltaConfig(mode=DeltaMode.prev_segment),
            )

    def test_delta_weights_without_feature_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="delta_weights"):
            build_feature_pipeline(
                feature_selection=self._sampling_with_delta_weights(),
                negative_selection=None,
                per_feature_stats=_stats_dict(self.N_SAE),
                n_sae_features=self.N_SAE,
                num_buckets=4,
                feature_delta=None,
            )

    def test_feature_delta_with_delta_weights_builds(self) -> None:
        pipeline = build_feature_pipeline(
            feature_selection=self._sampling_with_delta_weights(),
            negative_selection=None,
            per_feature_stats=_stats_dict(self.N_SAE),
            n_sae_features=self.N_SAE,
            num_buckets=4,
            feature_delta=FeatureDeltaConfig(mode=DeltaMode.prev_segment),
        )
        assert pipeline.delta_bundle is not None
        assert pipeline.delta_bundle.config.mode is DeltaMode.prev_segment
        assert pipeline.delta_bundle.percentile_table.shape == (self.N_SAE, 99)

    def test_threshold_selection_with_feature_delta_builds(self) -> None:
        """ThresholdSelection does not use delta_weights — should still build."""
        pipeline = build_feature_pipeline(
            feature_selection=ThresholdSelection(
                threshold_type=ThresholdType.feature_percentile, threshold=0.9,
            ),
            negative_selection=None,
            per_feature_stats=_stats_dict(self.N_SAE),
            n_sae_features=self.N_SAE,
            num_buckets=4,
            feature_delta=FeatureDeltaConfig(mode=DeltaMode.cum_max),
        )
        assert pipeline.delta_bundle is not None
        assert pipeline.delta_bundle.config.mode is DeltaMode.cum_max


# ---------------------------------------------------------------------------
# compute_doc_deltas invariants
# ---------------------------------------------------------------------------


class TestComputeDocDeltasInvariants:
    """Properties that must hold regardless of DELTA_DEBUG gating."""

    N_SAE = 1024

    def _pct_table(self) -> t.Tensor:
        return (
            t.linspace(0.0, 1.0, 99)
            .unsqueeze(0)
            .expand(self.N_SAE, -1)
            .to(t.float64)
            .contiguous()
        )

    def test_filtered_fids_are_subset_of_raw(self) -> None:
        """Every filtered fid must come from the same segment's raw fids."""
        seg_fids = [
            t.tensor([1, 3, 5, 7], dtype=t.int64),
            t.tensor([3, 5, 9, 11], dtype=t.int64),
            t.tensor([2, 4, 6], dtype=t.int64),
        ]
        seg_mags = [t.ones(len(f), dtype=t.float32) for f in seg_fids]
        seg_pcts: list[t.Tensor | None] = [
            t.tensor([0.2, 0.4, 0.6, 0.8], dtype=t.float32),
            t.tensor([0.9, 0.1, 0.7, 0.3], dtype=t.float32),
            t.tensor([0.5, 0.5, 0.5], dtype=t.float32),
        ]
        results = compute_doc_deltas(
            seg_fids, seg_mags, seg_pcts, self.N_SAE, DeltaMode.prev_segment,
        )
        for s_idx, (f_fids, _fm, _dv, _nf, _nw, _rm, _fp) in enumerate(results):
            if f_fids.numel() == 0:
                continue
            raw = set(seg_fids[s_idx].tolist())
            filt = set(f_fids.tolist())
            stray = filt - raw
            assert not stray, f"seg {s_idx}: stray fids {stray}"

    def test_all_delta_values_strictly_positive(self) -> None:
        """Delta values on filtered features must all be > 0."""
        seg_fids = [
            t.tensor([0, 1, 2], dtype=t.int64),
            t.tensor([0, 1, 2], dtype=t.int64),
        ]
        seg_mags = [t.ones(3, dtype=t.float32), t.ones(3, dtype=t.float32)]
        seg_pcts: list[t.Tensor | None] = [
            t.tensor([0.5, 0.3, 0.7], dtype=t.float32),
            t.tensor([0.2, 0.6, 0.7], dtype=t.float32),  # mixed up/down
        ]
        results = compute_doc_deltas(
            seg_fids, seg_mags, seg_pcts, self.N_SAE, DeltaMode.prev_segment,
        )
        for s_idx, (_ff, _fm, d_vals, *_rest) in enumerate(results):
            if d_vals is None or d_vals.numel() == 0:
                continue
            assert bool((d_vals > 0).all().item()), (
                f"seg {s_idx}: non-positive delta in filtered set: {d_vals}"
            )

    def test_second_segment_equal_pcts_has_no_positives(self) -> None:
        """Same fids + same pcts across two segs ⇒ second seg has zero positives."""
        fids = t.tensor([3, 7, 11], dtype=t.int64)
        pcts = t.tensor([0.4, 0.5, 0.6], dtype=t.float32)
        mags = t.ones(3, dtype=t.float32)
        results = compute_doc_deltas(
            [fids, fids], [mags, mags], [pcts, pcts], self.N_SAE, DeltaMode.prev_segment,
        )
        assert results[1][0].numel() == 0, (
            "Second seg with identical pcts should have no positive-delta features"
        )

    def test_first_segment_all_features_positive(self) -> None:
        """With zero reference state, every feature of seg 0 has positive delta."""
        fids = t.tensor([2, 5, 9], dtype=t.int64)
        pcts = t.tensor([0.3, 0.6, 0.9], dtype=t.float32)
        mags = t.ones(3, dtype=t.float32)
        results = compute_doc_deltas(
            [fids], [mags], [pcts], self.N_SAE, DeltaMode.prev_segment,
        )
        f_fids, _, d_vals, *_ = results[0]
        assert t.equal(f_fids, fids), "First seg should preserve all fids"
        assert d_vals is not None
        assert bool((d_vals > 0).all().item())

    def test_neg_delta_weights_nonneg(self) -> None:
        """neg_delta_weights must all be >= 0 (they are abs(delta) or ref pcts)."""
        seg_fids = [
            t.tensor([0, 1, 2], dtype=t.int64),
            t.tensor([0, 3], dtype=t.int64),  # dropped 1, 2; new 3
        ]
        seg_mags = [t.ones(3), t.ones(2)]
        seg_pcts: list[t.Tensor | None] = [
            t.tensor([0.5, 0.6, 0.7], dtype=t.float32),
            t.tensor([0.3, 0.8], dtype=t.float32),  # 0 decreased
        ]
        results = compute_doc_deltas(
            seg_fids, seg_mags, seg_pcts, self.N_SAE, DeltaMode.prev_segment,
        )
        _ff, _fm, _dv, neg_fids, neg_weights, _rm, _fp = results[1]
        assert neg_fids.numel() > 0, "Expected at least one negative delta feature"
        assert bool((neg_weights >= 0).all().item())


# ---------------------------------------------------------------------------
# Cross-document reset
# ---------------------------------------------------------------------------


class TestComputeDocDeltasCrossDocReset:
    """Each compute_doc_deltas call allocates a fresh ref buffer.

    This is the invariant the live dataloader relies on — per-doc state must
    not leak across document boundaries.
    """

    N_SAE = 256

    def test_second_doc_starts_fresh(self) -> None:
        """Running doc B should NOT inherit ref state from doc A."""
        fids_a = t.tensor([10, 20, 30], dtype=t.int64)
        pcts_a = t.tensor([0.8, 0.7, 0.6], dtype=t.float32)
        mags_a = t.ones(3)

        fids_b = t.tensor([10, 20, 30], dtype=t.int64)
        pcts_b = t.tensor([0.5, 0.4, 0.3], dtype=t.float32)  # strictly lower than A
        mags_b = t.ones(3)

        # Doc A: warm up — should leave non-zero ref if state were global.
        compute_doc_deltas(
            [fids_a, fids_a], [mags_a, mags_a], [pcts_a, pcts_a],
            self.N_SAE, DeltaMode.cum_max,
        )

        # Doc B: fresh call. Because state is per-call, seg 0 sees all features
        # as positive delta relative to zero — despite doc A's history.
        results = compute_doc_deltas(
            [fids_b], [mags_b], [pcts_b], self.N_SAE, DeltaMode.cum_max,
        )
        f_fids, _fm, d_vals, *_ = results[0]
        assert t.equal(f_fids, fids_b), (
            "Doc B seg 0 should keep all fids — ref carry-over would filter them out"
        )
        assert d_vals is not None
        assert bool((d_vals > 0).all().item())
        # Delta values equal pcts (ref was zero).
        assert t.allclose(d_vals, pcts_b, atol=1e-6)


# ---------------------------------------------------------------------------
# sample_delta_negatives disjointness with raw+ref+positives
# ---------------------------------------------------------------------------


class TestSampleDeltaNegativesExclusions:
    """Exclusion semantics that previously lacked explicit coverage."""

    N_SAE = 200

    def _ref_mask(self, fids: set[int]) -> t.Tensor:
        m = t.zeros(self.N_SAE, dtype=t.bool)
        for f in fids:
            m[f] = True
        return m

    def test_uniform_pool_excludes_raw_and_ref_and_positives(self) -> None:
        """With delta pool empty, uniform picks must avoid raw ∪ ref ∪ selected."""
        raw = t.tensor([0, 1, 2], dtype=t.int64)
        ref = self._ref_mask({0, 1, 2, 50, 51})
        selected_pos = t.tensor([100], dtype=t.int64)
        ref[100] = False  # positives aren't necessarily in ref
        rng = t.Generator().manual_seed(7)
        result = sample_delta_negatives(
            k=20, neg_delta_ratio=0.0,
            neg_delta_fids=t.tensor([], dtype=t.int64),
            neg_delta_weights=t.tensor([], dtype=t.float32),
            raw_feature_ids=raw, ref_mask=ref,
            selected_positive_ids=selected_pos,
            n_sae_features=self.N_SAE,
            rng=rng,
        )
        picked = set(result.tolist())
        forbidden = set(raw.tolist()) | {f for f in range(self.N_SAE) if ref[f]}
        forbidden |= set(selected_pos.tolist())
        assert not (picked & forbidden), f"Overlap with forbidden: {picked & forbidden}"

    def test_delta_pool_drops_selected_positives(self) -> None:
        """A positive id appearing in neg_delta_fids must be filtered out."""
        neg_fids = t.tensor([5, 6, 7, 8], dtype=t.int64)
        neg_weights = t.ones(4, dtype=t.float32)
        selected_pos = t.tensor([5, 8], dtype=t.int64)
        rng = t.Generator().manual_seed(9)
        result = sample_delta_negatives(
            k=4, neg_delta_ratio=1.0,
            neg_delta_fids=neg_fids, neg_delta_weights=neg_weights,
            raw_feature_ids=t.tensor([], dtype=t.int64),
            ref_mask=self._ref_mask({5, 6, 7, 8}),
            selected_positive_ids=selected_pos,
            n_sae_features=self.N_SAE,
            rng=rng,
        )
        picked = set(result.tolist())
        assert 5 not in picked and 8 not in picked, (
            f"Positives leaked into negatives: {picked}"
        )


# ---------------------------------------------------------------------------
# conditional_encdec_delta_4xh100 wrapper config
# ---------------------------------------------------------------------------


class TestConditionalEncdecDelta4xh100Config:
    """The delta-enabled 4xH100 wrapper is the primary entry point for
    real training runs — pin its composition so silent regressions
    can't reach the 4xH100 machine."""

    CONFIG_NAME = "conditional_encdec_delta_4xh100"

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    def test_composes(self, hydra_compose, config_dir: str) -> None:
        cfg = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg is not None

    def test_inherits_4xh100_hardware(self, hydra_compose, config_dir: str) -> None:
        cfg = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg.depth == 20
        assert cfg.device_batch_size == 8
        assert cfg.fp8 is True

    def test_delta_is_enabled(self, hydra_compose, config_dir: str) -> None:
        cfg = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg.feature_delta is not None
        assert cfg.feature_delta.mode == "prev_segment"
        assert cfg.feature_delta.neg_delta_ratio == 1.0

    def test_delta_weights_shapes_positive_pool(
        self, hydra_compose, config_dir: str,
    ) -> None:
        """``percentile_delta`` routes each feature's delta value into
        the positive pool's sampling weights (vs ``absolute_percentile``
        which keeps the standard magnitude-percentile weighting). The
        whole point of delta conditioning here is that positives should
        be the *growing* features, not just the highest-magnitude ones,
        so this config must pin ``percentile_delta``."""
        cfg = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg.feature_selection.delta_weights == "percentile_delta"


# ---------------------------------------------------------------------------
# conditional_encdec_delta_8xh100 wrapper config
# ---------------------------------------------------------------------------


class TestConditionalEncdecDelta8xh100Config:
    """Per-GPU HBM is identical between 4xH100 and 8xH100, so the 8-GPU
    variant is a thin rename that inherits everything from the 4-GPU
    delta config. These assertions pin that inheritance so a future
    edit to the 4-GPU config can't silently diverge the 8-GPU run."""

    CONFIG_NAME = "conditional_encdec_delta_8xh100"

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    def test_composes(self, hydra_compose, config_dir: str) -> None:
        cfg = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg is not None

    def test_inherits_delta_4xh100_body(
        self, hydra_compose, config_dir: str,
    ) -> None:
        cfg = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg.depth == 20
        assert cfg.device_batch_size == 8
        assert cfg.fp8 is True
        assert cfg.feature_delta is not None
        assert cfg.feature_delta.mode == "prev_segment"
        assert cfg.feature_delta.neg_delta_ratio == 1.0
        assert cfg.feature_selection.delta_weights == "percentile_delta"

    def test_run_tag_distinguishes_from_4xh100(
        self, hydra_compose, config_dir: str,
    ) -> None:
        cfg_4 = hydra_compose(config_dir, "conditional_encdec_delta_4xh100")
        cfg_8 = hydra_compose(config_dir, self.CONFIG_NAME)
        assert cfg_8.run != cfg_4.run, (
            "8xH100 must carry its own WandB run tag so results are "
            "separable from the 4xH100 reference run."
        )
        assert "8xh100" in cfg_8.run


# ---------------------------------------------------------------------------
# 4090 benchmark configs with feature_delta disabled
# ---------------------------------------------------------------------------


class TestBench4090DeltaDisabledConfigs:
    """Benchmark configs that disable feature_delta must also clear delta_weights."""

    @pytest.fixture()
    def config_dir(self) -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "training")
        )

    @pytest.mark.parametrize(
        "config_name",
        ["bench_live_4090", "bench_prepacked_4090"],
    )
    def test_delta_weights_cleared_and_pipeline_builds(
        self,
        hydra_compose,
        config_dir: str,
        config_name: str,
    ) -> None:
        cfg = hydra_compose(config_dir, config_name)

        assert cfg.feature_delta is None
        assert cfg.feature_selection.delta_weights is None

        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_stats_dict(64),
            n_sae_features=64,
            num_buckets=cfg.num_buckets,
            feature_delta=None,
        )
        assert pipeline.delta_bundle is None
