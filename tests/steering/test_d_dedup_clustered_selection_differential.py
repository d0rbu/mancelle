"""PR #739 Track D (D-dedup) empirical differential: main's pre-delegation
clustered inline body vs. the A5-delegating replacement.

This module is the MANDATORY empirical differential required by
``docs/plans/2026-07-24-pr739-track-d-dedup-plan.md`` §2, run and recorded
BEFORE any production edit to
``steering/eval/context_cache_build.py::_select_and_bucket_segment``.

Two frozen bodies:

- ``_frozen_legacy_clustered_selection`` -- hand-copied VERBATIM from
  origin/main's ``steering/eval/context_cache_build.py`` (commit
  ``1cb294963dad64d64e1f50d8e1542402bc372d89``, confirmed via
  ``git show origin/main:steering/eval/context_cache_build.py`` byte-identical
  to the working tree before any D-dedup edit -- see the report's
  "MAIN-DRIFT CHECK" section). Only the ``feature_clusters is not None``
  branch is retained (the unclustered ``elif``/``else`` arms are dead when
  this helper is only ever called with a real ``FeatureClusterConsolidation``,
  exactly mirroring how the original guarded that branch).
- ``_delegating_clustered_selection`` -- the exact replacement logic ported
  into production ``_select_and_bucket_segment`` by this PR: computes
  percentiles from RAW (pre-mask) ids via ``magnitude_to_percentile`` +
  ``percentile_to_uint8``, then calls the merged A5
  ``select_clustered_segment`` primitive for masking, consolidation,
  sampling, bucketing, and negative sampling.

Both call the SAME shared ``_validate_buckets_and_dups`` (imported directly
from ``context_cache_build`` -- that helper is untouched by this PR, so
reusing it is faithful, not a new implementation).

Established pattern: mirrors ``_frozen_legacy_sampling_selection`` in
``test_clustered_eval_context_cache.py`` (the PR-1 no-cluster boundary,
frozen the same way for the unclustered path).

Cells A-G below are the plan's required adversarial + replay scenarios.
Every numeric fixture in this file was empirically calibrated against a
throwaway probe script before being pinned here (see the task report); nothing
is hand-derived-and-trusted without having been run.
"""

from __future__ import annotations

import copy
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch as t
from loguru import logger

import steering.conditional.cluster_segment_selection as selection_module
from errors import FeatureClustersError
from steering.conditional.cluster_segment_selection import (
    ClusteredSegmentSelection,
    select_clustered_segment,
)
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline,
)
from steering.conditional.percentile_table import magnitude_to_percentile, percentile_to_uint8
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
    sample_features,
    sample_inactive_features,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval import context_cache_build
from steering.eval.context_cache_build import _validate_buckets_and_dups
from steering.refined_types import PositiveInt
from steering.stats_types import PerFeatureStat

# ===========================================================================
# Step 1: the two bodies under differential.
# ===========================================================================


def _frozen_legacy_clustered_selection(
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    *,
    pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
    num_buckets: int,
    source: str = "frozen-legacy",
) -> tuple[t.Tensor, t.Tensor]:
    """Verbatim clustered branch of origin/main's ``_select_and_bucket_segment``.

    Hand-copied from ``git show origin/main:steering/eval/context_cache_build.py``
    (lines ~227-388, clustered-only sub-path). The unclustered ``elif``/``else``
    arms are omitted because ``feature_clusters`` is always non-None here --
    they are provably dead code on this call path, not a semantic change.
    """
    if raw_fids.numel() != raw_mags.numel():
        raise ValueError(
            f"{source}: raw feature_ids and feature_mags lengths differ "
            f"({raw_fids.numel()} vs {raw_mags.numel()})"
        )
    if raw_fids.numel() and int(raw_fids.max().item()) >= n_sae_features:
        raise ValueError(f"{source}: raw feature id exceeds n_sae_features")
    if raw_fids.numel() and int(raw_fids.min().item()) < 0:
        raise ValueError(f"{source}: raw feature id is negative")

    if not isinstance(pipeline.selection_config, SamplingBundle):
        raise FeatureClustersError(
            "clustered threshold context selection is unsupported; use the audited sampling policy"
        )
    if raw_fids.ndim != 1 or raw_mags.ndim != 1:
        raise FeatureClustersError(
            f"{source}: clustered raw IDs/magnitudes must be one-dimensional"
        )
    if not raw_mags.is_floating_point() or not bool(t.isfinite(raw_mags).all()):
        raise FeatureClustersError(
            f"{source}: clustered raw magnitudes must be finite floating-point values"
        )
    if t.unique(raw_fids.to(t.int64)).numel() != raw_fids.numel():
        raise FeatureClustersError(
            f"{source}: clustered raw segment contains duplicate original feature IDs"
        )

    # R3 fix: apply the feature_mask to positive candidates.
    pos_fids, pos_mags = raw_fids, raw_mags
    if pipeline.feature_mask is not None:
        pos_fids, pos_mags = pipeline.feature_mask.apply_to_positive(raw_fids, raw_mags)

    # Axis A: raw-float percentile (NOT uint8-quantized).
    candidate_pcts = (
        magnitude_to_percentile(
            pos_fids.to(t.int64),
            pos_mags.float(),
            pipeline.selection_config.percentile_table,
        )
        / 100.0
    ).float()
    # Axis C: percentile computed AFTER masking (only survivors are scored).
    candidate_fids, candidate_pcts = feature_clusters.consolidate_segment(
        pos_fids.to(t.int64),
        candidate_pcts,
    )
    # Axis B: t.multinomial-based weighted sampling (via sample_features).
    fids, selected_pcts = sample_features(
        candidate_fids,
        candidate_pcts,
        pipeline.selection_config.percentile_table,
        pipeline.selection_config.selection,
        pipeline.selection_config.rng,
        override_weights=candidate_pcts,
    )
    buckets = percentile_to_bucket(selected_pcts, pipeline.effective_buckets)

    selected_positive_ids = fids.to(t.int64)
    negative_ids: t.Tensor | None = None
    if pipeline.negative_bundle is not None:
        # Axis D: adaptive oversample factor (sample_inactive_features), and
        # unconditionally attempted even when candidate_fids is empty (the
        # empty-segment negative-sampling divergence Cell G probes).
        negative_ids = sample_inactive_features(
            candidate_fids,
            n_sae_features,
            pipeline.negative_bundle.selection,
            pipeline.negative_bundle.rng,
            feature_mask=feature_clusters.effective_negative_mask(),
        )
        if negative_ids.numel() > 0:
            negative_buckets = t.full(
                (negative_ids.numel(),), pipeline.effective_buckets, dtype=t.int64
            )
            fids = t.cat([fids.to(t.int64), negative_ids.to(t.int64)])
            buckets = t.cat([buckets.to(t.int64), negative_buckets])

    if pipeline.feature_mask is not None:
        pipeline.feature_mask.assert_no_masked(fids.to(t.int64))

    if negative_ids is None:
        negative_ids = selected_positive_ids[:0]
    feature_clusters.assert_representatives_only(
        selected_positive_ids, surface=f"{source} positive context-cache selection"
    )
    feature_clusters.assert_representatives_only(
        negative_ids.to(t.int64), surface=f"{source} negative context-cache selection"
    )
    positive_values = {int(fid) for fid in selected_positive_ids.tolist()}
    negative_values = {int(fid) for fid in negative_ids.tolist()}
    if len(positive_values) != int(selected_positive_ids.numel()):
        raise FeatureClustersError(f"{source}: duplicate clustered positive feature IDs")
    if len(negative_values) != int(negative_ids.numel()):
        raise FeatureClustersError(f"{source}: duplicate clustered negative feature IDs")
    overlap = positive_values.intersection(negative_values)
    if overlap:
        raise FeatureClustersError(
            f"{source}: feature ID {min(overlap)} selected as both positive and negative"
        )

    _validate_buckets_and_dups(
        fids, buckets, PositiveInt.parse(num_buckets), source
    )
    return fids.to(t.int64), buckets.to(t.int64)


def _frozen_pre_hoist_delegating_clustered_selection(
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    *,
    pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
    num_buckets: int,
    source: str = "delegating-new",
) -> tuple[t.Tensor, t.Tensor]:
    """The D-dedup replacement: ported verbatim into production by this PR.

    Preserves: FeatureClustersError (not a plain ValueError), the raw-mag
    finiteness pre-check (A5 never receives raw magnitudes), the
    SamplingBundle-only scope boundary (clustered-threshold stays
    unsupported), and the shared id-range checks. Deletes: the local
    duplicate-id pre-check (redundant with A5's own ``_validate_inputs``,
    same FeatureClustersError type) and the ~28L rep-only/dup postcondition
    block (redundant with A5's own ``_assert_result``).

    D-dedup review Finding 2 (owner decision): a MASKED-OUT feature with no
    percentile statistics (an all-zero percentile-table row) is pre-detected
    and dropped BEFORE scoring, with a loud log -- it cannot affect the
    final selection either way, so raising on it would be a false alarm. A
    NON-masked stat-less feature is left alone and still raises (it WOULD
    affect the selection -- a real anomaly).
    """
    if raw_fids.numel() != raw_mags.numel():
        raise ValueError(
            f"{source}: raw feature_ids and feature_mags lengths differ "
            f"({raw_fids.numel()} vs {raw_mags.numel()})"
        )
    if raw_fids.numel() and int(raw_fids.max().item()) >= n_sae_features:
        raise ValueError(f"{source}: raw feature id exceeds n_sae_features")
    if raw_fids.numel() and int(raw_fids.min().item()) < 0:
        raise ValueError(f"{source}: raw feature id is negative")

    if not isinstance(pipeline.selection_config, SamplingBundle):
        raise FeatureClustersError(
            "clustered threshold context selection is unsupported; use the audited sampling policy"
        )
    if raw_fids.ndim != 1 or raw_mags.ndim != 1:
        raise FeatureClustersError(
            f"{source}: clustered raw IDs/magnitudes must be one-dimensional"
        )
    if not raw_mags.is_floating_point() or not bool(t.isfinite(raw_mags).all()):
        raise FeatureClustersError(
            f"{source}: clustered raw magnitudes must be finite floating-point values"
        )

    assert isinstance(pipeline.selection_config, SamplingBundle)
    # Axis A: uint8-quantized percentile (A5's training-currency contract).
    # Axis C: computed on the RAW (pre-mask) ids -- select_clustered_segment
    # owns masking internally, matching precompute_selected_features.py:1244.
    score_ids = raw_fids.to(t.int64)
    score_mags = raw_mags.float()
    if pipeline.feature_mask is not None and score_ids.numel() > 0:
        percentile_table = pipeline.selection_config.percentile_table
        stat_less = (percentile_table[score_ids] == 0).all(dim=1)
        masked_out = ~pipeline.feature_mask.keep_mask(score_ids)
        drop = stat_less & masked_out
        if bool(drop.any()):
            # Textually identical to the pre-hoist body at 75e30cfa. A "frozen"
            # reference that paraphrases the original is not frozen: it silently
            # licenses observable drift in whatever it paraphrased.
            logger.error(
                f"{source}: masked feature(s) {score_ids[drop].tolist()} fired "
                "in the eval extraction corpus but have no percentile "
                "statistics (an all-zero percentile-table row); excluding them "
                "before clustered percentile scoring since they are masked out "
                "regardless -- the extraction and statistics corpora are "
                "independently configured, so this is a legitimate (if "
                "noteworthy) mismatch, not a fatal error"
            )
            keep = ~drop
            score_ids = score_ids[keep]
            score_mags = score_mags[keep]
    percentile_scores = magnitude_to_percentile(
        score_ids,
        score_mags,
        pipeline.selection_config.percentile_table,
    )
    result = select_clustered_segment(
        score_ids,
        percentile_to_uint8(percentile_scores),
        clusters=feature_clusters,
        pipeline=pipeline,
    )
    # Axis B: Gumbel-max weighted sampling / shared _top_k_order (via A5's
    # sample_features_batched). Axis D: fixed oversample factor (via A5's
    # sample_inactive_features_batched). Empty-segment short-circuit: A5
    # returns _empty_result() (zero RNG, zero negatives) for BOTH raw-empty
    # and post-mask-empty inputs, unconditionally -- unlike the legacy body,
    # negatives are never attempted when there are no positive candidates.
    negative_buckets = t.full(
        (result.negative_ids.numel(),), pipeline.effective_buckets, dtype=t.int64
    )
    fids = t.cat([result.positive_ids, result.negative_ids])
    buckets = t.cat([result.positive_buckets, negative_buckets])

    _validate_buckets_and_dups(
        fids, buckets, PositiveInt.parse(num_buckets), source
    )
    return fids.to(t.int64), buckets.to(t.int64)


def _pipeline_rng_states(pipeline: FeaturePipelineConfig) -> tuple[t.Tensor, ...]:
    generators: list[t.Generator] = []
    selection = pipeline.selection_config
    if isinstance(selection, SamplingBundle):
        generators.append(selection.rng)
    if (
        pipeline.negative_bundle is not None
        and all(pipeline.negative_bundle.rng is not rng for rng in generators)
    ):
        generators.append(pipeline.negative_bundle.rng)
    return tuple(generator.get_state().clone() for generator in generators)


def _assert_tensor_byte_equal(
    frozen: t.Tensor,
    production: t.Tensor,
    *,
    surface: str,
) -> None:
    assert frozen.device == production.device, f"G3b hoist changed {surface} device"
    assert frozen.dtype is production.dtype, f"G3b hoist changed {surface} dtype"
    assert frozen.shape == production.shape, f"G3b hoist changed {surface} shape"
    assert (
        frozen.contiguous().numpy().tobytes()
        == production.contiguous().numpy().tobytes()
    ), f"G3b hoist changed {surface} bytes"


@contextmanager
def _captured_logs(sink: list[str]) -> Iterator[None]:
    """Collect fully rendered loguru messages emitted inside the block.

    Records the rendered text, so a change to either the template or the
    interpolated values is caught -- the frozen body interpolates with an
    f-string while production uses loguru's lazy ``{}`` args, and only the
    rendered result is what an operator actually reads.
    """
    handler_id = logger.add(
        lambda message: sink.append(message.record["message"]),
        level="DEBUG",
        format="{message}",
    )
    try:
        yield
    finally:
        logger.remove(handler_id)


def _delegating_clustered_selection(
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    *,
    pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
    num_buckets: int,
    source: str = "delegating-new",
) -> tuple[t.Tensor, t.Tensor]:
    """Call production and prove the G3b hoist preserved the frozen body.

    Every existing Cell A-G invocation passes through this wrapper.  A deep
    copy runs the exact pre-hoist D-dedup body, then the real production
    ``context_cache_build`` call runs on the caller's pipeline.  Outputs,
    exception behavior, and final private RNG states must all match.
    """

    frozen_pipeline = copy.deepcopy(pipeline)
    typed_n_sae_features = PositiveInt.parse(n_sae_features)
    typed_num_buckets = PositiveInt.parse(num_buckets)

    def run_frozen():
        return _frozen_pre_hoist_delegating_clustered_selection(
            raw_fids,
            raw_mags,
            pipeline=frozen_pipeline,
            feature_clusters=feature_clusters,
            n_sae_features=typed_n_sae_features,
            num_buckets=typed_num_buckets,
            source=source,
        )

    def run_production():
        return context_cache_build._select_and_bucket_segment(
            raw_fids,
            raw_mags,
            pipeline=pipeline,
            n_sae_features=typed_n_sae_features,
            num_buckets=typed_num_buckets,
            feature_clusters=feature_clusters,
            source=source,
        )

    frozen_result: tuple[t.Tensor, t.Tensor] | None = None
    production_result: tuple[t.Tensor, t.Tensor] | None = None
    frozen_error: Exception | None = None
    production_error: Exception | None = None
    # Capture logs on BOTH sides. Without this the harness compared only
    # tensors, exceptions and RNG -- so the masked-stat-less diagnostic could
    # (and did) drift between the frozen body and production while every
    # assertion still passed. An operator-facing message is observable
    # behaviour; a proof that the hoist was a pure code motion has to cover it.
    frozen_logs: list[str] = []
    production_logs: list[str] = []
    with _captured_logs(frozen_logs):
        try:
            frozen_result = run_frozen()
        except Exception as exc:
            frozen_error = exc
    with _captured_logs(production_logs):
        try:
            production_result = run_production()
        except Exception as exc:
            production_error = exc

    assert frozen_logs == production_logs, (
        "G3b hoist changed the emitted diagnostics: "
        f"frozen={frozen_logs!r}, production={production_logs!r}"
    )

    assert (frozen_error is None) == (production_error is None), (
        "G3b hoist changed success/failure behavior: "
        f"frozen={frozen_error!r}, production={production_error!r}"
    )
    frozen_rng_states = _pipeline_rng_states(frozen_pipeline)
    production_rng_states = _pipeline_rng_states(pipeline)
    assert len(frozen_rng_states) == len(production_rng_states)
    for index, (frozen_rng, production_rng) in enumerate(
        zip(frozen_rng_states, production_rng_states, strict=True)
    ):
        _assert_tensor_byte_equal(
            frozen_rng,
            production_rng,
            surface=f"final clustered-selection RNG state {index}",
        )
    if production_error is not None:
        assert frozen_error is not None
        assert type(production_error) is type(frozen_error)
        assert str(production_error) == str(frozen_error)
        raise production_error
    assert frozen_result is not None
    assert production_result is not None
    _assert_tensor_byte_equal(
        frozen_result[0],
        production_result[0],
        surface="selected feature IDs",
    )
    _assert_tensor_byte_equal(
        frozen_result[1],
        production_result[1],
        surface="selected buckets",
    )
    return production_result


def _frozen_legacy_magnitude_primitive(
    feature_ids: t.Tensor,
    feature_magnitudes: t.Tensor,
    *,
    percentile_table: t.Tensor,
    clusters: FeatureClusterConsolidation,
    pipeline: FeaturePipelineConfig,
    source: str,
) -> ClusteredSegmentSelection:
    """Adapter used only by the non-vacuousness mutation trap."""

    del percentile_table
    num_buckets = pipeline.effective_buckets + int(
        pipeline.negative_bundle is not None
    )
    ids, buckets = _frozen_legacy_clustered_selection(
        feature_ids,
        feature_magnitudes,
        pipeline=pipeline,
        feature_clusters=clusters,
        n_sae_features=clusters.n_sae_features,
        num_buckets=num_buckets,
        source=source,
    )
    n_positive = int((buckets < pipeline.effective_buckets).sum().item())
    return ClusteredSegmentSelection(
        positive_ids=ids[:n_positive],
        positive_buckets=buckets[:n_positive],
        positive_percentiles_uint8=t.zeros(n_positive, dtype=t.uint8),
        negative_ids=ids[n_positive:],
    )


# ===========================================================================
# Shared fixtures/helpers (scaffolding only).
# ===========================================================================


def _build_clusters(
    tmp_path: Path,
    groups: list[list[int]],
    mask: FeatureMask | None,
    n_sae_features: int,
    *,
    name: str = "clusters.json",
) -> FeatureClusterConsolidation:
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "clusters": groups}), encoding="utf-8")
    resolved = resolve_feature_clusters({"feature_clusters": {"path": str(path)}}, n_sae_features)
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _identity_stats(n_sae_features: int) -> dict[str, PerFeatureStat]:
    """Per-feature stats whose percentile table row is the IDENTITY map
    p_k = k (k=1..99). With this table, magnitude_to_percentile(fid, mag,
    table) == mag for any mag in (1, 99) exactly (verified: for mag=79.8,
    searchsorted lands idx=79, interpolating rows[78]=79/rows[79]=80 gives
    frac=0.8, result=79.8) -- letting fixtures target an EXACT desired
    percentile by setting the raw magnitude to that percentile value
    directly, with no interpolation arithmetic needed at the call site.
    """
    out: dict[str, PerFeatureStat] = {}
    for fid in range(n_sae_features):
        out[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=100.0,
            median_mag=0.5,
            percentiles={f"p{k}": float(k) for k in range(1, 100)},
        )
    return out


def _stats_with_scale(
    n_sae_features: int, scale_for: dict[int, float], default_scale: float = 0.01
):
    out: dict[str, PerFeatureStat] = {}
    for fid in range(n_sae_features):
        scale = scale_for.get(fid, default_scale)
        percentiles = {f"p{p}": p * scale for p in range(1, 100)}
        out[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles=percentiles,
        )
    return out


def _run_both(
    raw_fids: t.Tensor,
    raw_mags: t.Tensor,
    *,
    legacy_pipeline: FeaturePipelineConfig,
    new_pipeline: FeaturePipelineConfig,
    feature_clusters: FeatureClusterConsolidation,
    n_sae_features: int,
    num_buckets: int,
):
    legacy = _frozen_legacy_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=legacy_pipeline,
        feature_clusters=feature_clusters,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        source="legacy",
    )
    new = _delegating_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=new_pipeline,
        feature_clusters=feature_clusters,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        source="new",
    )
    return legacy, new


# ===========================================================================
# Cell A: existing-suite replay (test_clustered_eval_context_cache.py's 5
# clustered fixtures). Expected: mostly PASS (degenerate, single candidate
# post-consolidation) EXCEPT the flagged 3rd test below (>=2 candidates) --
# a finding, NOT proof of equivalence (see module docstring + plan §2).
# ===========================================================================

N_FEATURES = 8


def _pipeline_pair(
    *,
    mask: FeatureMask | None,
    seed: int,
    negatives: bool,
    stats: dict[str, PerFeatureStat] | None = None,
    mean: float = 1.0,
    std: float = 1.0,
    min_k: int = 1,
    max_k: int | None = 2,
    n_sae: int = N_FEATURES,
    num_buckets: int = 5,
) -> tuple[FeaturePipelineConfig, FeaturePipelineConfig]:
    stats = stats if stats is not None else _stats_with_scale(n_sae, {3: 0.02})
    feature_selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=mean,
        std=std,
        min_k=min_k,
        max_k=max_k,
    )
    negative_selection = (
        NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        )
        if negatives
        else None
    )

    def build() -> FeaturePipelineConfig:
        return build_feature_pipeline(
            feature_selection=feature_selection,
            negative_selection=negative_selection,
            per_feature_stats=stats,
            n_sae_features=n_sae,
            num_buckets=num_buckets,
            seed=seed,
            feature_mask=mask,
        )

    return build(), build()


def test_cell_a1_uses_sibling_max_percentile_and_rep_buckets_replay(tmp_path: Path):
    """Cell A1 == test_clustered_selection_uses_sibling_max_percentile_and_rep_buckets.
    Degenerate: cluster [1, 3] both survive (no mask) -> ONE representative
    group post-consolidation -> no stochastic choice. Expect byte-identical.
    """
    clusters = _build_clusters(tmp_path, [[1, 3]], None, N_FEATURES)
    legacy_pipe, new_pipe = _pipeline_pair(mask=None, seed=42, negatives=False)
    legacy, new = _run_both(
        t.tensor([1, 3], dtype=t.int64),
        t.tensor([0.2, 1.6], dtype=t.float32),
        legacy_pipeline=legacy_pipe,
        new_pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=N_FEATURES,
        num_buckets=5,
    )
    assert t.equal(legacy[0], new[0])
    assert t.equal(legacy[1], new[1])
    assert legacy[0].tolist() == [1]
    assert legacy[1].tolist() == [4]


def test_cell_a2_masks_before_representative_choice_and_negatives_replay(tmp_path: Path):
    """Cell A2 == test_clustered_selection_masks_before_representative_choice_and_negatives.
    Degenerate: cluster [1, 3, 5] with mask blacklisting {1, 7} -> survivors
    {3, 5}, single representative (3) -> one positive candidate. Negative
    pool at n_sae_features=8 happens to make the adaptive and fixed
    oversample factors coincide (both compute to 3), so no axis-D
    divergence either. Expect byte-identical.
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 7))
    clusters = _build_clusters(tmp_path, [[1, 3, 5]], mask, N_FEATURES)
    legacy_pipe, new_pipe = _pipeline_pair(mask=mask, seed=42, negatives=True)
    legacy, new = _run_both(
        t.tensor([1, 3, 5], dtype=t.int64),
        t.tensor([0.9, 1.6, 0.4], dtype=t.float32),
        legacy_pipeline=legacy_pipe,
        new_pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=N_FEATURES,
        num_buckets=5,
    )
    assert t.equal(legacy[0], new[0])
    assert t.equal(legacy[1], new[1])
    assert legacy[0][0].item() == 3
    assert legacy[1][0].item() == 3


def test_cell_a3_sequential_rng_stream_replay_reveals_multi_candidate_divergence(tmp_path: Path):
    """Cell A3 == test_clustered_selection_reuses_one_sequential_rng_stream,
    THE flagged non-degenerate existing fixture. Segment 1 ([0, 1, 3, 6])
    consolidates to 3 representative groups ({0}, {1,3}->1, {6}) -- a real
    multi-candidate weighted-mode draw. Empirically DIVERGES: different
    positive winners AND different final RNG state. Confirms the plan's
    flag; this is why "replay the existing suite" is necessary but NOT
    sufficient (the ORIGINAL test itself only checks self-consistency
    across two identically-seeded runs of the SAME implementation, so it
    keeps passing unmodified after delegation even though the underlying
    selected values changed -- see test_clustered_eval_context_cache.py,
    unmodified).
    """
    clusters = _build_clusters(tmp_path, [[1, 3], [2, 4]], None, N_FEATURES)
    legacy_pipe, new_pipe = _pipeline_pair(mask=None, seed=17, negatives=True)
    segments = (
        (
            t.tensor([0, 1, 3, 6], dtype=t.int64),
            t.tensor([0.31, 0.20, 1.60, 0.70], dtype=t.float32),
        ),
        (
            t.tensor([2, 4, 5, 7], dtype=t.int64),
            t.tensor([0.40, 0.80, 0.55, 0.95], dtype=t.float32),
        ),
    )
    legacy_results = []
    new_results = []
    for idx, (ids, mags) in enumerate(segments):
        legacy_results.append(
            _frozen_legacy_clustered_selection(
                ids,
                mags,
                pipeline=legacy_pipe,
                feature_clusters=clusters,
                n_sae_features=N_FEATURES,
                num_buckets=5,
                source=f"legacy-seg{idx}",
            )
        )
        new_results.append(
            _delegating_clustered_selection(
                ids,
                mags,
                pipeline=new_pipe,
                feature_clusters=clusters,
                n_sae_features=N_FEATURES,
                num_buckets=5,
                source=f"new-seg{idx}",
            )
        )
    # Empirically verified divergence (seg0: legacy=[6,7] vs new=[0,5]).
    diverges = any(legacy_results[i][0].tolist() != new_results[i][0].tolist() for i in range(2))
    assert diverges, "expected multi-candidate weighted-mode divergence on this fixture"
    assert not t.equal(
        _pipeline_rng_states(legacy_pipe)[0], _pipeline_rng_states(new_pipe)[0]
    ), "expected final RNG state to diverge too (Gumbel-max vs multinomial consume differently)"


def test_cell_a4_clustered_threshold_still_unsupported_on_both_bodies(tmp_path: Path):
    """Cell A4 == test_clustered_threshold_context_selection_is_explicitly_unsupported.
    The SamplingBundle-only scope boundary is a DELIBERATE, preserved
    restriction on BOTH bodies (not one of the four axes) -- both must
    raise FeatureClustersError identically.
    """
    from steering.conditional.feature_pipeline import build_feature_pipeline as _bfp

    clusters = _build_clusters(tmp_path, [[1, 3]], None, N_FEATURES)
    pipeline = _bfp(
        feature_selection=ThresholdSelection(
            threshold_type=ThresholdType.segment_topk, threshold=1
        ),
        negative_selection=None,
        per_feature_stats=_stats_with_scale(N_FEATURES, {3: 0.02}),
        n_sae_features=N_FEATURES,
        num_buckets=5,
        seed=42,
    )
    raw_fids = t.tensor([1, 3], dtype=t.int64)
    raw_mags = t.tensor([0.2, 1.6], dtype=t.float32)
    with pytest.raises(FeatureClustersError, match="clustered threshold context selection"):
        _frozen_legacy_clustered_selection(
            raw_fids,
            raw_mags,
            pipeline=pipeline,
            feature_clusters=clusters,
            n_sae_features=N_FEATURES,
            num_buckets=5,
        )
    with pytest.raises(FeatureClustersError, match="clustered threshold context selection"):
        _delegating_clustered_selection(
            raw_fids,
            raw_mags,
            pipeline=pipeline,
            feature_clusters=clusters,
            n_sae_features=N_FEATURES,
            num_buckets=5,
        )


# ===========================================================================
# Cell B: near-boundary percentile stress (axis A -- quantization).
# ===========================================================================


def test_cell_b1_bucket_boundary_crossing_from_quantization(tmp_path: Path):
    """A single candidate at raw percentile 79.9% (deterministic
    top_k_percentile mode, k=1, no ties, no cluster interaction --
    isolates axis A alone). Empirically verified:
    percentile_to_uint8(79.9) = round(203.745) = 204;
    uint8_to_percentile(204) = 204/255 = 0.8 EXACTLY (a clean bucket
    boundary at effective_buckets=5). floor(0.799 * 5) = 3 (legacy,
    raw float) vs floor(0.8 * 5) = 4 (new, quantized) -- axis A alone
    flips the OUTPUT BUCKET for the identical selected feature.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="b1.json")
    stats = _identity_stats(n)
    pos_sel = SamplingSelection(
        distribution=Distribution.normal,
        mean=1.0,
        std=0.01,
        min_k=1,
        max_k=2,
        mode=SamplingMode.top_k_percentile,
    )
    legacy_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=1)
    new_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=1)
    raw_fids = t.tensor([5], dtype=t.int64)
    raw_mags = t.tensor([79.9], dtype=t.float32)

    byte = int(percentile_to_uint8(t.tensor([79.9])).item())
    assert byte == 204, "calibration drifted: expected byte 204 for percent 79.9"

    legacy, new = _run_both(
        raw_fids,
        raw_mags,
        legacy_pipeline=legacy_pipe,
        new_pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    assert legacy[0].tolist() == new[0].tolist() == [5], "same feature selected in both"
    assert legacy[1].tolist() == [3], "legacy: floor(0.799*5)=3"
    assert new[1].tolist() == [4], "new: floor(0.8*5)=4 -- axis A alone changed the bucket"


def test_g3b_hoist_differential_is_non_vacuous_against_frozen_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aliasing the hoisted primitive to legacy must trip the proof harness."""

    n = 16
    clusters = _build_clusters(
        tmp_path,
        [[10, 11]],
        None,
        n,
        name="g3b-non-vacuous.json",
    )
    pipeline = build_feature_pipeline(
        SamplingSelection(
            distribution=Distribution.normal,
            mean=1.0,
            std=0.01,
            min_k=1,
            max_k=2,
            mode=SamplingMode.top_k_percentile,
        ),
        None,
        _identity_stats(n),
        n,
        num_buckets=5,
        seed=1,
    )
    monkeypatch.setattr(
        context_cache_build,
        "select_clustered_segment_from_magnitudes",
        _frozen_legacy_magnitude_primitive,
    )

    with pytest.raises(AssertionError, match="G3b hoist changed selected buckets"):
        _delegating_clustered_selection(
            t.tensor([5], dtype=t.int64),
            t.tensor([79.9], dtype=t.float32),
            pipeline=pipeline,
            feature_clusters=clusters,
            n_sae_features=n,
            num_buckets=5,
        )


def test_cell_b2_quantization_induced_tie_flips_top_k_winner(tmp_path: Path):
    """Two candidates at DISTINCT raw percentiles (50.15%, 50.05%) that
    quantize to the SAME uint8 byte (128 both) -- top_k_percentile mode
    (isolating axis A from axis B: _top_k_order is shared code, see Cell
    D1). Legacy has no tie (strictly higher raw score wins
    deterministically, feature 5 always). New has an exact tie post-
    quantization, so the shared _top_k_order's seeded tie-break decides
    the winner. Empirically verified over 60 seeds: 20/60 (33%) divergent
    winners, 0/60 RNG-state diffs (tie-breaking consumes the SAME amount
    of RNG regardless of which candidate wins -- axis A changes WHO wins,
    not how much RNG is used, in this mode).
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="b2.json")
    stats = _identity_stats(n)
    pos_sel = SamplingSelection(
        distribution=Distribution.normal,
        mean=1.0,
        std=0.01,
        min_k=1,
        max_k=2,
        mode=SamplingMode.top_k_percentile,
    )
    raw_fids = t.tensor([5, 6], dtype=t.int64)
    raw_mags = t.tensor([50.15, 50.05], dtype=t.float32)
    byte_x = int(percentile_to_uint8(t.tensor([50.15])).item())
    byte_y = int(percentile_to_uint8(t.tensor([50.05])).item())
    assert byte_x == byte_y == 128, "calibration drifted: expected both to quantize to byte 128"

    divergences = 0
    rng_diffs = 0
    winners = {5: 0, 6: 0}
    n_seeds = 60
    for seed in range(n_seeds):
        legacy_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=seed)
        new_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=seed)
        legacy, new = _run_both(
            raw_fids,
            raw_mags,
            legacy_pipeline=legacy_pipe,
            new_pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=n,
            num_buckets=5,
        )
        assert legacy[0].tolist() == [5], (
            "legacy must deterministically pick the strictly-higher raw score"
        )
        winners[int(new[0][0].item())] += 1
        if legacy[0].tolist() != new[0].tolist():
            divergences += 1
        if not t.equal(
            _pipeline_rng_states(legacy_pipe)[0],
            _pipeline_rng_states(new_pipe)[0],
        ):
            rng_diffs += 1

    assert divergences > 0, "expected at least some seeds to flip the tie toward feature 6"
    assert winners[6] > 0, "new must sometimes pick the quantization-tied feature 6"
    assert rng_diffs == 0, (
        "expected RNG state to be UNAFFECTED by axis A in top_k_percentile mode "
        "(only the tie-break OUTCOME depends on which candidate wins, not the "
        "amount of RNG drawn)"
    )


# ===========================================================================
# Cell C: multi-candidate WEIGHTED-mode stress (axis B -- sampling algorithm).
# >=50 seeds, plain random.Random-style loop (fixed here via a manual seed
# range, matching test_cluster_segment_selection.py's property-loop pattern).
# ===========================================================================


def test_cell_c_weighted_mode_multi_candidate_property_loop(tmp_path: Path):
    """4 distinct, non-boundary candidates (20%/40%/60%/80%), SamplingMode
    .weighted (the dataclass default + every existing production fixture's
    mode), min_k=1 max_k=4 (candidate_count), 50 seeds. Empirically
    verified: 35/50 (70%) divergent selections, 50/50 (100%) RNG-state
    divergence -- t.multinomial and the Gumbel-max trick consume the
    generator differently even for the SAME weights and k.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="c.json")
    stats = _identity_stats(n)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=4,
        mode=SamplingMode.weighted,
    )
    raw_fids = t.tensor([1, 2, 3, 4], dtype=t.int64)
    raw_mags = t.tensor([20.0, 40.0, 60.0, 80.0], dtype=t.float32)

    n_seeds = 50
    divergent_selection = 0
    divergent_rng = 0
    for seed in range(n_seeds):
        legacy_pipe = build_feature_pipeline(
            pos_sel, None, stats, n, num_buckets=5, seed=1000 + seed
        )
        new_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=1000 + seed)
        legacy, new = _run_both(
            raw_fids,
            raw_mags,
            legacy_pipeline=legacy_pipe,
            new_pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=n,
            num_buckets=5,
        )
        if legacy[0].tolist() != new[0].tolist() or legacy[1].tolist() != new[1].tolist():
            divergent_selection += 1
        if not t.equal(
            _pipeline_rng_states(legacy_pipe)[0],
            _pipeline_rng_states(new_pipe)[0],
        ):
            divergent_rng += 1

    assert divergent_selection >= n_seeds * 0.5, (
        f"expected the MAJORITY of {n_seeds} seeds to diverge in weighted mode "
        f"(multinomial vs Gumbel-max), got {divergent_selection}/{n_seeds}"
    )
    assert divergent_rng == n_seeds, (
        f"expected EVERY seed's final RNG state to diverge in weighted mode, "
        f"got {divergent_rng}/{n_seeds}"
    )


# ===========================================================================
# Cell D: top_k_percentile multi-candidate (lower-risk per plan -- shared
# _top_k_order). Two sub-cells: clean percentiles (isolate mode-sharing
# claim) and near-boundary percentiles (axis-A-induced ties, cross-check
# with Cell B2 at a different candidate count).
# ===========================================================================


def test_cell_d1_top_k_percentile_multi_candidate_clean_no_divergence(tmp_path: Path):
    """Same 4-candidate setup as Cell C, but SamplingMode.top_k_percentile.
    Empirically verified: 0/50 (0%) divergence in EITHER selection or RNG
    state -- confirms the plan's "lower risk, shared _top_k_order" claim
    directly: sample_k/sample_k_batched are bit-identical for a single
    segment, and select_top_k_by_percentile{,_batched} both delegate to the
    exact same _top_k_order helper. When there is no quantization-induced
    tie (see Cell D2 / B2 for when there is), this mode is a genuine
    pure-dedup with ZERO observable divergence.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="d1.json")
    stats = _identity_stats(n)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )
    raw_fids = t.tensor([1, 2, 3, 4], dtype=t.int64)
    raw_mags = t.tensor([20.0, 40.0, 60.0, 80.0], dtype=t.float32)

    n_seeds = 50
    divergent_selection = 0
    divergent_rng = 0
    for seed in range(n_seeds):
        legacy_pipe = build_feature_pipeline(
            pos_sel, None, stats, n, num_buckets=5, seed=2000 + seed
        )
        new_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=2000 + seed)
        legacy, new = _run_both(
            raw_fids,
            raw_mags,
            legacy_pipeline=legacy_pipe,
            new_pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=n,
            num_buckets=5,
        )
        if legacy[0].tolist() != new[0].tolist() or legacy[1].tolist() != new[1].tolist():
            divergent_selection += 1
        if not t.equal(
            _pipeline_rng_states(legacy_pipe)[0],
            _pipeline_rng_states(new_pipe)[0],
        ):
            divergent_rng += 1

    assert divergent_selection == 0, (
        f"expected ZERO divergence with clean (non-tying) percentiles in "
        f"top_k_percentile mode, got {divergent_selection}/{n_seeds}"
    )
    assert divergent_rng == 0, f"expected ZERO RNG-state divergence, got {divergent_rng}/{n_seeds}"


def test_cell_d2_top_k_percentile_near_boundary_ties_still_diverge(tmp_path: Path):
    """Same top_k_percentile mode as D1, but with 4 candidates whose raw
    percentiles are deliberately near-boundary (pairwise close enough that
    SOME seeds quantize into ties): 50.15/50.05/70.15/70.05. Cross-checks
    Cell B2's 2-candidate finding at a richer candidate count: axis A can
    still perturb top_k_percentile mode's output via induced ties, even
    though the algorithm (_top_k_order) itself is shared.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="d2.json")
    stats = _identity_stats(n)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )
    raw_fids = t.tensor([1, 2, 3, 4], dtype=t.int64)
    raw_mags = t.tensor([50.15, 50.05, 70.15, 70.05], dtype=t.float32)

    n_seeds = 50
    divergent_selection = 0
    for seed in range(n_seeds):
        legacy_pipe = build_feature_pipeline(
            pos_sel, None, stats, n, num_buckets=5, seed=4000 + seed
        )
        new_pipe = build_feature_pipeline(pos_sel, None, stats, n, num_buckets=5, seed=4000 + seed)
        legacy, new = _run_both(
            raw_fids,
            raw_mags,
            legacy_pipeline=legacy_pipe,
            new_pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=n,
            num_buckets=5,
        )
        if legacy[0].tolist() != new[0].tolist():
            divergent_selection += 1

    assert divergent_selection > 0, (
        "expected near-boundary percentile pairs to produce SOME tie-induced "
        f"divergence in top_k_percentile mode too, got {divergent_selection}/{n_seeds}"
    )


# ===========================================================================
# Cell E: mask-ordering stress (axis C -- percentile computed before vs
# after masking).
# ===========================================================================


def test_cell_e1_post_mask_identity_percentile_call_site_replay(tmp_path: Path, monkeypatch):
    """Replays test_post_mask_identity_clusters_still_use_clustered_
    percentile_semantics: cluster [1, 3], mask blacklists 3
    (is_identity_after_mask). Empirically verified: legacy's
    magnitude_to_percentile is called on [1] only (post-mask survivor);
    new's is called on [1, 3] (the full raw ids -- select_clustered_segment
    masks INSIDE, after scoring). The FINAL selection (fids=[1],
    buckets=[4]) is UNCHANGED for this specific fixture (percentile
    computation is per-feature independent, so scoring the masked sibling
    too does not change feature 1's own score) -- only the internal
    "which ids got scored" observation differs. This is why the plan
    frames axis C's resolution as re-pinning the ``seen`` assertion, not
    the selected-value assertions.
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    clusters = _build_clusters(tmp_path, [[1, 3]], mask, N_FEATURES, name="e1.json")
    assert clusters.is_identity_after_mask
    legacy_pipe, new_pipe = _pipeline_pair(mask=mask, seed=42, negatives=False)

    seen_legacy: list[list[int]] = []
    seen_new: list[list[int]] = []

    def make_fake(seen_list):
        def fake_percentiles(feature_ids, feature_mags, percentile_table):
            seen_list.append(feature_ids.tolist())
            return t.full(feature_ids.shape, 90.0, dtype=t.float64)

        return fake_percentiles

    # sys.modules[__name__] (NOT `import tests.steering.<this file> as _this_module`)
    # -- a self-import under pytest's rootdir-relative import machinery can
    # bind a SEPARATE duplicate module object from the one actually
    # executing this test, silently patching the wrong namespace (the real
    # magnitude_to_percentile keeps running, seen_list stays empty). Empirically
    # caught: an early differential run recorded seen_legacy == [] instead of
    # [[1]] until this was fixed.
    _this_module = sys.modules[__name__]

    raw_fids = t.tensor([1, 3], dtype=t.int64)
    raw_mags = t.tensor([0.2, 1.6], dtype=t.float32)

    monkeypatch.setattr(_this_module, "magnitude_to_percentile", make_fake(seen_legacy))
    legacy = _frozen_legacy_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=legacy_pipe,
        feature_clusters=clusters,
        n_sae_features=N_FEATURES,
        num_buckets=5,
    )
    monkeypatch.setattr(_this_module, "magnitude_to_percentile", make_fake(seen_new))
    monkeypatch.setattr(
        selection_module,
        "magnitude_to_percentile",
        make_fake(seen_new),
    )
    new = _delegating_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=N_FEATURES,
        num_buckets=5,
    )

    assert seen_legacy == [[1]], "legacy scores only the post-mask survivor"
    assert seen_new == [[1, 3], [1, 3]], (
        "frozen and hoisted bodies both score the FULL raw ids before masking"
    )
    assert legacy[0].tolist() == new[0].tolist() == [1]
    assert legacy[1].tolist() == new[1].tolist() == [4]


def _e2_stats_and_clusters(tmp_path: Path, n: int, *, missing_fid: int, mask: FeatureMask | None):
    clusters = _build_clusters(tmp_path, [[1, 2]], mask, n, name=f"e2_{missing_fid}.json")
    stats = {}
    for fid in range(n):
        if fid == missing_fid:
            continue  # NO stats -> all-zero percentile-table row
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles={f"p{p}": float(p) * 0.01 for p in range(1, 100)},
        )
    return clusters, stats


def test_cell_e2a_masked_stat_less_feature_passes_with_loud_log_and_correct_selection(
    tmp_path: Path,
    caplog,
):
    """D-dedup review Finding 2 (owner decision): a masked-out feature with
    NO percentile statistics at all (its magnitude_to_percentile row is
    all-zero because build_percentile_table never saw it) is a legitimate,
    reachable mismatch (the eval extraction corpus and the statistics
    corpus are independently configured). It cannot affect the final
    selection either way (it's masked out), so the delegating body no
    longer raises on it -- it pre-detects the masked+stat-less combination,
    LOGS LOUDLY (logger.error), drops the feature before scoring, and lets
    the build PASS with the correct selection (masked feature excluded,
    the non-masked candidate selected) -- identical to what the
    pre-delegation legacy body always produced for this fixture (it never
    queried the masked feature's row in the first place).
    """
    n = 8
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    clusters, stats = _e2_stats_and_clusters(tmp_path, n, missing_fid=3, mask=mask)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric, mean=1.0, std=1.0, min_k=1, max_k=2
    )
    legacy_pipe = build_feature_pipeline(
        pos_sel, None, stats, n, num_buckets=5, seed=42, feature_mask=mask
    )
    new_pipe = build_feature_pipeline(
        pos_sel, None, stats, n, num_buckets=5, seed=42, feature_mask=mask
    )
    raw_fids = t.tensor([1, 3], dtype=t.int64)  # 3 is masked AND has zero stats
    raw_mags = t.tensor([0.2, 0.5], dtype=t.float32)

    legacy = _frozen_legacy_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=legacy_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    assert legacy[0].tolist() == [1], (
        "legacy does not raise: it never scores the masked zero-stat feature"
    )

    caplog.clear()
    new = _delegating_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    assert new[0].tolist() == legacy[0].tolist() == [1], (
        "new must PASS (not raise) and select the same correct feature as legacy"
    )
    assert new[1].tolist() == legacy[1].tolist()
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert error_records, "expected a loud logger.error for the masked stat-less feature"
    assert "3" in error_records[0].message
    assert "no percentile statistics" in error_records[0].message


def test_cell_e2b_non_masked_stat_less_feature_still_raises(tmp_path: Path):
    """The other half of Finding 2's owner decision: a NON-masked feature
    with no percentile statistics is a genuinely different, real anomaly
    (it WOULD affect the selection, unlike a masked one) -- it must keep
    raising, not be silently dropped.
    """
    n = 8
    mask = FeatureMask(FeatureMaskMode.blacklist, (7,))  # does NOT cover feature 3
    clusters, stats = _e2_stats_and_clusters(tmp_path, n, missing_fid=3, mask=mask)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric, mean=1.0, std=1.0, min_k=1, max_k=2
    )
    new_pipe = build_feature_pipeline(
        pos_sel, None, stats, n, num_buckets=5, seed=42, feature_mask=mask
    )
    raw_fids = t.tensor([1, 3], dtype=t.int64)  # 3 has zero stats but is NOT masked
    raw_mags = t.tensor([0.2, 0.5], dtype=t.float32)

    with pytest.raises(ValueError, match="Percentile statistics are missing"):
        _delegating_clustered_selection(
            raw_fids,
            raw_mags,
            pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=n,
            num_buckets=5,
        )


def test_cell_e2c_ordinary_no_stat_less_case_unchanged(tmp_path: Path):
    """Sanity control: with a mask configured but no stat-less feature
    anywhere in the segment, behavior is unaffected by Finding 2's new
    pre-filter (the stat_less/masked_out intersection is empty, so no
    feature is dropped and no log fires).
    """
    n = 8
    mask = FeatureMask(FeatureMaskMode.blacklist, (3,))
    clusters, stats = _e2_stats_and_clusters(tmp_path, n, missing_fid=99, mask=mask)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric, mean=1.0, std=1.0, min_k=1, max_k=2
    )
    legacy_pipe = build_feature_pipeline(
        pos_sel, None, stats, n, num_buckets=5, seed=42, feature_mask=mask
    )
    new_pipe = build_feature_pipeline(
        pos_sel, None, stats, n, num_buckets=5, seed=42, feature_mask=mask
    )
    raw_fids = t.tensor([1, 3], dtype=t.int64)  # 3 is masked but HAS stats
    raw_mags = t.tensor([0.2, 0.5], dtype=t.float32)

    legacy = _frozen_legacy_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=legacy_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    new = _delegating_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    assert legacy[0].tolist() == new[0].tolist() == [1]
    assert legacy[1].tolist() == new[1].tolist()


# ===========================================================================
# Cell F: negative-sampling pool stress (axis D -- adaptive vs fixed
# oversample factor). n_sae_features=512, restrictive whitelist (pool ~4.7%
# of n_sae_features), >=20 seeds.
# ===========================================================================


def test_cell_f_negative_sampling_small_pool_property_loop(tmp_path: Path):
    """Empirically verified: 20/20 (100%) divergence in BOTH negative_ids
    and final RNG state. With pool_size ~24 against n_sae_features=512
    (~4.7%), legacy's adaptive factor is max(3, 512//24+1)=22 (drawing a
    much larger randint candidate pool per rejection-sampling round); A5's
    factor is the FIXED MIN_OVERSAMPLE_FACTOR=3 regardless of pool
    scarcity -- different-shaped t.randint draws desync the RNG stream
    immediately.
    """
    n_sae = 512
    allowed_ids = list(range(24))
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(allowed_ids))
    clusters = _build_clusters(tmp_path, [[100, 101]], mask, n_sae, name="f.json")
    stats = {}
    for fid in allowed_ids + [100, 101]:
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=0.5,
            percentiles={f"p{p}": float(p) * 0.01 for p in range(1, 100)},
        )
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric, mean=1.0, std=1.0, min_k=1, max_k=2
    )
    neg_sel = NegativeSamplingSelection(
        distribution=Distribution.geometric, mean=5.0, std=5.0, min_k=3, max_k=8
    )
    raw_fids = t.tensor([1], dtype=t.int64)
    raw_mags = t.tensor([0.5], dtype=t.float32)

    n_seeds = 20
    divergent_negatives = 0
    divergent_rng = 0
    for seed in range(n_seeds):
        legacy_pipe = build_feature_pipeline(
            pos_sel, neg_sel, stats, n_sae, num_buckets=5, seed=3000 + seed, feature_mask=mask
        )
        new_pipe = build_feature_pipeline(
            pos_sel, neg_sel, stats, n_sae, num_buckets=5, seed=3000 + seed, feature_mask=mask
        )
        legacy_result = _frozen_legacy_clustered_selection(
            raw_fids,
            raw_mags,
            pipeline=legacy_pipe,
            feature_clusters=clusters,
            n_sae_features=n_sae,
            num_buckets=5,
        )
        new_result = _delegating_clustered_selection(
            raw_fids,
            raw_mags,
            pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=n_sae,
            num_buckets=5,
        )
        legacy_negs = sorted(fid for fid in legacy_result[0].tolist() if fid != 1)
        new_negs = sorted(fid for fid in new_result[0].tolist() if fid != 1)
        if legacy_negs != new_negs:
            divergent_negatives += 1
        if not t.equal(
            _pipeline_rng_states(legacy_pipe)[0],
            _pipeline_rng_states(new_pipe)[0],
        ):
            divergent_rng += 1

    assert divergent_negatives >= n_seeds * 0.5, (
        f"expected the MAJORITY of {n_seeds} small-pool seeds to diverge in "
        f"negative ids, got {divergent_negatives}/{n_seeds}"
    )
    assert divergent_rng == n_seeds, (
        f"expected EVERY seed's negative-sampler RNG state to diverge under "
        f"small-pool conditions, got {divergent_rng}/{n_seeds}"
    )


# ===========================================================================
# Cell G: empty-segment RNG-no-op parity. Do NOT assume from A5's own
# docstring (it documents parity vs sample_features_batched, a DIFFERENT
# historical reference, not context_cache_build's sample_features).
# ===========================================================================


def _empty_pipeline_pair(*, mask, negatives, seed=42, n_sae=16):
    stats = _identity_stats(n_sae)
    pos_sel = SamplingSelection(
        distribution=Distribution.geometric, mean=1.0, std=1.0, min_k=1, max_k=2
    )
    neg_sel = (
        NegativeSamplingSelection(
            distribution=Distribution.geometric, mean=2.0, std=2.0, min_k=1, max_k=3
        )
        if negatives
        else None
    )
    legacy_pipe = build_feature_pipeline(
        pos_sel, neg_sel, stats, n_sae, num_buckets=5, seed=seed, feature_mask=mask
    )
    new_pipe = build_feature_pipeline(
        pos_sel, neg_sel, stats, n_sae, num_buckets=5, seed=seed, feature_mask=mask
    )
    return legacy_pipe, new_pipe


def test_cell_g1_raw_empty_with_negatives_legacy_draws_new_does_not(tmp_path: Path):
    """Raw-empty segment (no feature IDs at all) with negative_bundle
    CONFIGURED. Empirically verified DIVERGENT (a fifth, plan-uncalled-out
    finding, "axis E"): legacy STILL attempts negative sampling (it draws
    from n_sae_features minus an EMPTY active-exclusion set, since nothing
    about a raw-empty POSITIVE segment tells sample_inactive_features to
    skip) -- consumes RNG and returns non-empty negative_ids. A5's
    select_clustered_segment short-circuits on ``feature_ids.numel()==0``
    BEFORE ever attempting negative sampling -- zero RNG, zero negatives,
    unconditionally. This matches training currency: precompute_selected_
    features.py's per-segment loop has its OWN caller-level ``if n_feat ==
    0: continue`` BEFORE it ever calls select_clustered_segment for a
    raw-empty segment, so training never draws negatives for a raw-empty
    segment either -- the legacy eval body's negative draw was a real
    training-parity bug, not a stylistic difference.
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="g1.json")
    legacy_pipe, new_pipe = _empty_pipeline_pair(mask=None, negatives=True, n_sae=n)
    assert legacy_pipe.negative_bundle is not None
    assert new_pipe.negative_bundle is not None
    legacy_state_before = legacy_pipe.negative_bundle.rng.get_state().clone()
    new_state_before = new_pipe.negative_bundle.rng.get_state().clone()

    legacy = _frozen_legacy_clustered_selection(
        t.tensor([], dtype=t.int64),
        t.tensor([], dtype=t.float32),
        pipeline=legacy_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    new = _delegating_clustered_selection(
        t.tensor([], dtype=t.int64),
        t.tensor([], dtype=t.float32),
        pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )

    legacy_rng_changed = not t.equal(
        legacy_pipe.negative_bundle.rng.get_state(), legacy_state_before
    )
    new_rng_changed = not t.equal(new_pipe.negative_bundle.rng.get_state(), new_state_before)

    assert legacy[0].numel() > 0, "legacy draws negatives even for a raw-empty positive segment"
    assert legacy_rng_changed, "legacy consumes RNG for the unconditional negative draw"
    assert new[0].numel() == 0, "new short-circuits to a fully empty result"
    assert not new_rng_changed, "new consumes ZERO RNG for a raw-empty segment"


def test_cell_g2_post_mask_empty_with_negatives_legacy_draws_new_does_not(tmp_path: Path):
    """Same divergence as G1, but the raw segment is NON-empty and becomes
    empty only after masking (every raw id is blacklisted). Empirically
    verified: identical divergence pattern to G1 -- legacy's negative
    sampler uses the (empty) POST-mask candidate set as its exclusion
    list regardless of why it's empty, so it still draws; A5's
    _empty_result() short-circuit fires identically for post-mask-empty
    as it does for raw-empty (same `if feature_ids.numel() == 0` check,
    reached a second time after mask application). Training currency:
    precompute_selected_features.py's clustered branch has no separate
    mask-application step of its own -- a non-empty raw segment that goes
    fully-masked flows straight into select_clustered_segment, which
    returns _empty_result() for it via this exact code path.
    """
    n = 16
    mask = FeatureMask(FeatureMaskMode.blacklist, (1, 2))
    clusters = _build_clusters(tmp_path, [[1, 2]], mask, n, name="g2.json")
    legacy_pipe, new_pipe = _empty_pipeline_pair(mask=mask, negatives=True, n_sae=n)
    assert legacy_pipe.negative_bundle is not None
    assert new_pipe.negative_bundle is not None
    legacy_state_before = legacy_pipe.negative_bundle.rng.get_state().clone()
    new_state_before = new_pipe.negative_bundle.rng.get_state().clone()

    raw_fids = t.tensor([1, 2], dtype=t.int64)  # both blacklisted -> post-mask-empty
    raw_mags = t.tensor([0.3, 0.4], dtype=t.float32)

    legacy = _frozen_legacy_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=legacy_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    new = _delegating_clustered_selection(
        raw_fids,
        raw_mags,
        pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )

    legacy_rng_changed = not t.equal(
        legacy_pipe.negative_bundle.rng.get_state(), legacy_state_before
    )
    new_rng_changed = not t.equal(new_pipe.negative_bundle.rng.get_state(), new_state_before)

    assert legacy[0].numel() > 0, "legacy draws negatives even for a post-mask-empty segment"
    assert legacy_rng_changed
    assert new[0].numel() == 0, "new short-circuits to a fully empty result"
    assert not new_rng_changed


def test_cell_g3_raw_empty_no_negatives_both_are_true_rng_no_ops(tmp_path: Path):
    """Sanity/positive control: with negative_bundle DISABLED, a raw-empty
    segment IS a true RNG no-op on BOTH bodies (confirms the G1/G2
    divergence is specifically about negative sampling when positives are
    empty, not some other artifact of the delegation).
    """
    n = 16
    clusters = _build_clusters(tmp_path, [[10, 11]], None, n, name="g3.json")
    legacy_pipe, new_pipe = _empty_pipeline_pair(mask=None, negatives=False, n_sae=n)
    legacy_state_before = _pipeline_rng_states(legacy_pipe)[0]
    new_state_before = _pipeline_rng_states(new_pipe)[0]

    legacy = _frozen_legacy_clustered_selection(
        t.tensor([], dtype=t.int64),
        t.tensor([], dtype=t.float32),
        pipeline=legacy_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )
    new = _delegating_clustered_selection(
        t.tensor([], dtype=t.int64),
        t.tensor([], dtype=t.float32),
        pipeline=new_pipe,
        feature_clusters=clusters,
        n_sae_features=n,
        num_buckets=5,
    )

    assert legacy[0].numel() == 0 and new[0].numel() == 0
    assert t.equal(_pipeline_rng_states(legacy_pipe)[0], legacy_state_before)
    assert t.equal(_pipeline_rng_states(new_pipe)[0], new_state_before)


def test_cell_h1_unbounded_k_and_whitelist_mask_hoist_parity(tmp_path: Path):
    """Cell H1 — the two legal production config classes cells A-G omit.

    Every other cell pins ``min_k=1`` and a blacklist mask, so the hoist's
    before/after proof did not cover ``min_k=0`` (a selection that may legally
    choose nothing), ``max_k=None`` (unbounded k, where the sampler's candidate
    ceiling comes from the pool rather than the config), or whitelist masks.
    All three are legal on production checkpoints, so leaving them out made the
    parity claim narrower than it read.

    Swept over seeds because the divergence these cells hunt is RNG-ordering
    sensitive: a single sample cannot distinguish "identical" from "identical
    by luck".
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 3, 5))
    clusters = _build_clusters(tmp_path, [[1, 3, 5]], mask, N_FEATURES)
    for seed in range(50):
        legacy_pipe, new_pipe = _pipeline_pair(
            mask=mask,
            seed=seed,
            negatives=True,
            min_k=0,
            max_k=None,
        )
        legacy, new = _run_both(
            t.tensor([1, 3, 5], dtype=t.int64),
            t.tensor([0.9, 1.6, 0.4], dtype=t.float32),
            legacy_pipeline=legacy_pipe,
            new_pipeline=new_pipe,
            feature_clusters=clusters,
            n_sae_features=N_FEATURES,
            num_buckets=5,
        )
        assert t.equal(legacy[0], new[0]), f"ids diverged at seed {seed}"
        assert t.equal(legacy[1], new[1]), f"buckets diverged at seed {seed}"
