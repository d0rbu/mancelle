"""The one ``feature_delta`` configuration clustered selection may accept.

Clustered segment selection used to reject *any* delta config outright. That made
clustered evaluation impossible against every clustered checkpoint that exists,
because all of them are trained with ``delta_weights=percentile_delta`` and a
``splice_counterfactual`` delta — the pipeline's own cross-validation then demands
a non-null ``feature_delta``, so the two guards were mutually unsatisfiable.

One slice is genuinely equivalent to ``feature_delta=null`` here, and only one:

* ``mode=splice_counterfactual`` produces a delta ONLY for segments carrying a
  counterfactual reference. Every other segment gets a ``None`` slot, which the
  training dataloader routes down the ordinary absolute-percentile path
  (``dataloader.py``: "a None slot means no delta for this segment — apply
  absolute-percentile conditioning", plus the ``batch_results[s_idx] ... else
  None`` routing beneath it). Clustered *evaluation* selects over real documents
  with no splice provenance, so no segment can carry such a reference and every
  segment takes that ordinary path.
* ``neg_delta_ratio == 1.0`` keeps the equivalence an identity rather than an
  interpretation: negatives are asked to come entirely from the delta pool, and an
  empty delta pool degenerates wholly to the ordinary pool.

``prev_segment``/``cum_max`` produce a delta for EVERY segment, so they can never
reduce to absolute-percentile conditioning and must stay rejected. A partial
``neg_delta_ratio`` expresses a blend whose eval-time reduction would be a
judgement call, so it stays rejected too.

These tests pin both directions. Widening the accepted slice, or narrowing it back
to "reject all deltas", must fail here.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch as t

from errors import FeatureClustersError
from steering.conditional.cluster_segment_selection import _validate_inputs
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    RawClusteredFeatureSegment,
)

_N_SAE = 8


def _consolidation() -> FeatureClusterConsolidation:
    definition = FeatureClusterDefinition(clusters=((1, 2),))
    resolved = ResolvedFeatureClusters(
        source_path=Path("clusters.json"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=_N_SAE
        ),
    )
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3))
    return FeatureClusterConsolidation.build(resolved, mask)


def _pipeline(delta: FeatureDeltaConfig | None) -> FeaturePipelineConfig:
    """A pipeline whose only property under test is its delta bundle."""
    consolidation = _consolidation()
    return FeaturePipelineConfig(
        # Never read by _validate_inputs; a real value keeps the stub type-exact.
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(
                threshold_type=ThresholdType.feature_percentile, threshold=0.5
            ),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros((_N_SAE, 1), dtype=t.float64),
        negative_bundle=None,
        effective_buckets=1,
        delta_bundle=(
            None
            if delta is None
            else DeltaBundle(
                config=delta,
                percentile_table=t.zeros((_N_SAE, 99), dtype=t.float64),
            )
        ),
        feature_mask=consolidation.feature_mask,
    )


def _validate(delta: FeatureDeltaConfig | None) -> None:
    _validate_inputs(
        t.tensor([1, 3], dtype=t.int64),
        t.tensor([10, 20], dtype=t.uint8),
        clusters=_consolidation(),
        pipeline=_pipeline(delta),
    )


def test_null_delta_is_accepted() -> None:
    """The historical baseline must keep working."""
    _validate(None)


def test_splice_counterfactual_with_full_delta_ratio_is_accepted() -> None:
    """The one equivalent slice: what every real clustered checkpoint carries."""
    _validate(
        FeatureDeltaConfig(
            mode=DeltaMode.splice_counterfactual,
            neg_delta_ratio=1.0,
        )
    )


@pytest.mark.parametrize(
    "mode", [DeltaMode.prev_segment, DeltaMode.cum_max]
)
def test_always_delta_modes_stay_rejected(mode: DeltaMode) -> None:
    """These produce a delta for EVERY segment, so they cannot reduce to null."""
    with pytest.raises(FeatureClustersError, match="produces a delta for"):
        _validate(FeatureDeltaConfig(mode=mode, neg_delta_ratio=1.0))


@pytest.mark.parametrize("ratio", [0.0, 0.5, 0.999])
def test_partial_delta_ratio_stays_rejected(ratio: float) -> None:
    """A blend's eval-time reduction is a judgement, not an identity."""
    with pytest.raises(FeatureClustersError, match="neg_delta_ratio=1.0"):
        _validate(
            FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=ratio,
            )
        )


def test_raw_clustered_segment_carries_no_counterfactual_provenance() -> None:
    """Tripwire for the assumption the accepted delta slice rests on.

    ``_validate_inputs`` accepts ``mode=splice_counterfactual`` because clustered
    evaluation cannot produce a segment carrying a counterfactual reference, so the
    delta is provably empty and dropping it is a no-op. That is a *config-level*
    guard standing in for a *data-level* fact, and the fact holds today only because
    ``RawClusteredFeatureSegment`` structurally has nowhere to put such a reference.

    If a future eval mode adds provenance to this type, the guard would keep
    accepting while a real per-segment delta got silently dropped — and nothing in
    ``_assert_result`` would catch it. Pin the field set so that change fails HERE,
    loudly, with a pointer to the guard that must be revisited.
    """

    assert {field.name for field in fields(RawClusteredFeatureSegment)} == {
        "feature_ids",
        "feature_magnitudes",
        "char_start",
        "char_end",
    }, (
        "RawClusteredFeatureSegment gained or lost a field. If it now carries "
        "splice/counterfactual provenance, revisit the splice_counterfactual "
        "accept-branch in steering/conditional/cluster_segment_selection.py "
        "_validate_inputs: its equivalence-to-null argument no longer holds."
    )
