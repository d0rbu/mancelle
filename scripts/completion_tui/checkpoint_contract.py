"""Fail-loud checkpoint feature semantics for the completion TUI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from errors import FeatureVocabError
from scripts.completion_tui.cluster_view import ClusterView
from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract as SharedCheckpointFeatureContract,
)
from steering.conditional.checkpoint_feature_contract import (
    assert_model_matches_checkpoint_feature_contract,
)
from steering.conditional.checkpoint_feature_contract import (
    resolve_checkpoint_feature_contract as resolve_shared_checkpoint_feature_contract,
)
from steering.conditional.delta import DeltaMode, parse_feature_delta
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
    assert_no_gradient_ranking,
    parse_feature_selection,
)
from steering.conditional.thresholding import ThresholdType
from steering.refined_types import PositiveInt

_PERCENTILE_ONLY_THRESHOLD_TYPES = frozenset(
    {
        ThresholdType.feature_percentile,
        ThresholdType.feature_percentile_segment_topk,
    }
)


@dataclass(frozen=True)
class CheckpointFeatureContract:
    feature_mask: FeatureMask | None
    resolved_clusters: ResolvedFeatureClusters | None
    consolidation: FeatureClusterConsolidation | None
    cluster_view: ClusterView | None

    @property
    def clustered(self) -> bool:
        return self.resolved_clusters is not None


@beartype
def resolve_checkpoint_feature_contract(
    meta: Mapping[str, Any] | None,
    *,
    n_sae_features: PositiveInt,
    meta_path: str,
) -> CheckpointFeatureContract:
    """Resolve mask/clusters from metadata and bind them to statistics width."""
    shared = resolve_shared_checkpoint_feature_contract(
        meta,
        n_sae_features=n_sae_features,
        surface=f"completion_tui checkpoint {meta_path}",
        source_path=meta_path,
        width_source="feature statistics have",
    )
    consolidation = shared.consolidation
    cluster_view = (
        None if consolidation is None else ClusterView.build(consolidation)
    )
    return CheckpointFeatureContract(
        feature_mask=shared.feature_mask,
        resolved_clusters=shared.resolved_clusters,
        consolidation=consolidation,
        cluster_view=cluster_view,
    )


def assert_contract_supports_selection(
    contract: CheckpointFeatureContract,
    user_cfg: Mapping[str, Any],
) -> None:
    """Reject clustered live-selection combinations the TUI cannot reproduce."""
    if not contract.clustered:
        return
    raw_selection = user_cfg.get("feature_selection")
    if raw_selection is None:
        raise ValueError("clustered completion TUI checkpoints require feature_selection metadata")
    raw_feature_delta = user_cfg.get("feature_delta")
    if raw_feature_delta is not None:
        feature_delta = parse_feature_delta(raw_feature_delta)
        assert feature_delta is not None
        if feature_delta.mode is not DeltaMode.splice_counterfactual:
            raise ValueError(
                "clustered completion TUI supports feature_delta only for "
                "mode='splice_counterfactual' with neg_delta_ratio=1.0; got "
                f"mode={feature_delta.mode.value!r}, which produces a delta for "
                "every segment and cannot reduce to ordinary percentile "
                "conditioning at evaluation time"
            )
        if feature_delta.neg_delta_ratio != 1.0:
            raise ValueError(
                "clustered completion TUI supports feature_delta mode="
                "'splice_counterfactual' only with neg_delta_ratio=1.0; got "
                f"neg_delta_ratio={feature_delta.neg_delta_ratio!r}"
            )
    selection = parse_feature_selection(raw_selection)
    assert_no_gradient_ranking(selection, "completion TUI live selection")
    if (
        isinstance(selection, SamplingSelection)
        and selection.delta_weights is not None
        and raw_feature_delta is None
    ):
        raise ValueError(
            "clustered completion TUI checkpoint has feature_selection."
            "delta_weights without feature_delta; refusing this forged "
            "metadata pairing"
        )
    if isinstance(selection, ThresholdSelection) and (
        selection.threshold_type not in _PERCENTILE_ONLY_THRESHOLD_TYPES
    ):
        supported = sorted(kind.value for kind in _PERCENTILE_ONLY_THRESHOLD_TYPES)
        raise ValueError(
            "clustered completion TUI checkpoints do not support threshold type "
            f"{selection.threshold_type.value!r}; supported percentile-only types "
            f"are {supported}"
        )


@beartype
def assert_model_matches_contract(
    feature_vocab: FeatureVocab | None,
    contract: CheckpointFeatureContract,
    *,
    n_sae_features: PositiveInt,
    checkpoint_dir: str,
    loaded_model_n_sae_features: PositiveInt | None = None,
) -> None:
    """Require the model's vocabulary to equal the effective one for the contract.

    The expected vocabulary is ``effective_feature_vocab`` over the checkpoint's
    recorded universe: the consolidation when clustered (representatives keep
    rows, consolidated-away members do not — invariants E1-E7), else the mask,
    else ``None``. A clustered checkpoint written before cluster-aware
    compaction carries a mask-only vocabulary and is rejected exactly, matching
    training-resume strictness.
    """
    if (
        loaded_model_n_sae_features is not None
        and loaded_model_n_sae_features != n_sae_features
    ):
        raise FeatureVocabError(
            f"completion TUI checkpoint {checkpoint_dir} loaded model has "
            f"original n_sae={loaded_model_n_sae_features}, but feature "
            f"statistics have n_sae={n_sae_features}"
        )
    shared = SharedCheckpointFeatureContract(
        feature_mask=contract.feature_mask,
        resolved_clusters=contract.resolved_clusters,
    )
    assert_model_matches_checkpoint_feature_contract(
        feature_vocab,
        shared,
        n_sae_features=n_sae_features,
        checkpoint_dir=checkpoint_dir,
        surface="completion TUI checkpoint",
        missing_metadata_label="feature_mask_canonical metadata",
        missing_metadata_guidance=(
            "Use scripts.analysis.dump_feature_mask to diagnose/recover the mask. "
            "The TUI requires canonical mask metadata in a newly published "
            "checkpoint; do not hand-edit the existing artifact."
        ),
    )
