"""Exact mask, cluster, and feature-vocabulary contract for checkpoints.

Encoder feature IDs remain ORIGINAL SAE IDs at every public boundary, while a
compacted encoder stores only one embedding row per effective feature. A
consumer may therefore use compacted state only after proving that the loaded
state-dict vocabulary equals the vocabulary implied by the checkpoint's own
canonical mask and static-cluster metadata.

This module owns no evaluation policy or filesystem discovery. It reconstructs
only self-contained checkpoint metadata and compares it with the authoritative
vocabulary recovered from the state dict by the model loader.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from errors import FeatureClustersError, FeatureVocabError
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    resolve_feature_clusters_from_checkpoint_meta,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    feature_mask_from_checkpoint_meta,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import PositiveInt


@dataclass(frozen=True)
class CheckpointFeatureContract:
    """Canonical feature universe reconstructed from checkpoint metadata."""

    feature_mask: FeatureMask | None
    resolved_clusters: ResolvedFeatureClusters | None

    @property
    def clustered(self) -> bool:
        return self.resolved_clusters is not None

    @property
    def consolidation(self) -> FeatureClusterConsolidation | None:
        """Rebuild derived mutable tensors so they never become contract authority."""

        if self.resolved_clusters is None:
            return None
        return FeatureClusterConsolidation.build(
            self.resolved_clusters,
            self.feature_mask,
        )

    @property
    def universe(self) -> FeatureClusterConsolidation | FeatureMask | None:
        return self.consolidation if self.clustered else self.feature_mask


@beartype
def resolve_checkpoint_feature_contract(
    meta: Mapping[str, Any] | None,
    *,
    n_sae_features: PositiveInt,
    surface: str,
    source_path: str,
    width_source: str = "the loaded model has",
) -> CheckpointFeatureContract:
    """Reconstruct and width-bind the mask/cluster contract in ``meta``.

    ``source_path`` is provenance for an in-memory cluster definition; this
    function never reads it. Partial or malformed metadata raises through the
    existing strict mask/cluster decoders instead of being normalized.
    """

    feature_mask = feature_mask_from_checkpoint_meta(meta)
    if feature_mask is not None:
        feature_mask.allowed_universe_mask(n_sae_features)

    resolved_clusters = resolve_feature_clusters_from_checkpoint_meta(
        meta,
        surface=surface,
        source_path=source_path,
    )
    if (
        resolved_clusters is not None
        and resolved_clusters.n_sae_features != n_sae_features
    ):
        raise FeatureClustersError(
            f"{surface} cluster descriptor has "
            f"n_sae_features={resolved_clusters.n_sae_features}, but {width_source} "
            f"n_sae_features={n_sae_features}"
        )

    return CheckpointFeatureContract(
        feature_mask=feature_mask,
        resolved_clusters=resolved_clusters,
    )


def assert_feature_vocab_width(
    feature_vocab: FeatureVocab | None,
    n_sae_features: int,
    *,
    name: str,
) -> None:
    """Require a compact vocabulary to use the caller's original SAE width."""

    if (
        feature_vocab is not None
        and feature_vocab.n_sae_features != n_sae_features
    ):
        raise FeatureVocabError(
            f"{name} has feature vocabulary n_sae_features="
            f"{feature_vocab.n_sae_features}, but expected n_sae_features="
            f"{n_sae_features}"
        )


@beartype
def assert_model_matches_checkpoint_feature_contract(
    feature_vocab: FeatureVocab | None,
    contract: CheckpointFeatureContract,
    *,
    n_sae_features: PositiveInt,
    checkpoint_dir: str,
    surface: str,
    missing_metadata_label: str,
    missing_metadata_guidance: str,
) -> None:
    """Require state-dict vocabulary equality with the recorded universe.

    Equality means the complete :class:`FeatureVocab` value: original SAE width
    plus the sorted tuple of every surviving ORIGINAL feature ID. Equal table
    heights or equal fingerprints alone are not accepted substitutes.
    """

    assert_feature_vocab_width(
        feature_vocab,
        n_sae_features,
        name=f"{surface} {checkpoint_dir}",
    )
    if (
        feature_vocab is not None
        and contract.feature_mask is None
        and not contract.clustered
    ):
        raise FeatureVocabError(
            f"{surface} {checkpoint_dir} has a compacted feature vocabulary "
            f"(observed V={feature_vocab.size} of "
            f"n_sae={feature_vocab.n_sae_features}) but lacks "
            f"{missing_metadata_label}. {missing_metadata_guidance}"
        )

    expected = effective_feature_vocab(contract.universe, n_sae_features)
    if feature_vocab == expected:
        return

    observed_size = n_sae_features if feature_vocab is None else feature_vocab.size
    expected_size = n_sae_features if expected is None else expected.size
    observed_n = (
        n_sae_features
        if feature_vocab is None
        else feature_vocab.n_sae_features
    )
    expected_n = n_sae_features if expected is None else expected.n_sae_features
    observed_fingerprint = (
        "full-width" if feature_vocab is None else feature_vocab.fingerprint()
    )
    expected_fingerprint = (
        "full-width" if expected is None else expected.fingerprint()
    )
    raise FeatureVocabError(
        f"{surface} {checkpoint_dir} feature vocabulary does not match exactly "
        "the effective vocabulary derived from its recorded mask/clusters: "
        f"observed V={observed_size} of n_sae={observed_n}, expected "
        f"V={expected_size} of n_sae={expected_n}; observed "
        f"fingerprint={observed_fingerprint}, expected "
        f"fingerprint={expected_fingerprint}. A clustered checkpoint written "
        "before cluster-aware compaction (mask-only vocabulary) must be "
        "retrained or evaluated with the tooling of its own generation."
    )


@beartype
def resolve_and_assert_model_feature_contract(
    model: Any,
    meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
    surface: str,
    missing_metadata_label: str,
    missing_metadata_guidance: str,
) -> CheckpointFeatureContract:
    """Resolve the checkpoint contract and bind it to a loaded encoder-decoder model."""

    encoder = getattr(model, "encoder", None)
    encoder_config = getattr(encoder, "config", None)
    raw_width = getattr(encoder_config, "n_sae_features", None)
    try:
        n_sae_features = PositiveInt.parse(raw_width)
    except (TypeError, ValueError) as exc:
        raise FeatureVocabError(
            f"{surface} {checkpoint_dir} does not expose a valid positive original "
            "SAE width on model.encoder.config.n_sae_features: "
            f"{raw_width!r}"
        ) from exc
    if not hasattr(model, "feature_vocab"):
        raise FeatureVocabError(
            f"{surface} {checkpoint_dir} does not expose model.feature_vocab"
        )

    contract = resolve_checkpoint_feature_contract(
        meta,
        n_sae_features=n_sae_features,
        surface=f"{surface} {checkpoint_dir}",
        source_path=checkpoint_dir,
    )
    assert_model_matches_checkpoint_feature_contract(
        model.feature_vocab,
        contract,
        n_sae_features=n_sae_features,
        checkpoint_dir=checkpoint_dir,
        surface=surface,
        missing_metadata_label=missing_metadata_label,
        missing_metadata_guidance=missing_metadata_guidance,
    )
    assert_consistent = getattr(model, "assert_vocab_consistent", None)
    if not callable(assert_consistent):
        raise FeatureVocabError(
            f"{surface} {checkpoint_dir} does not expose "
            "assert_vocab_consistent()"
        )
    assert_consistent()
    return contract


__all__ = [
    "CheckpointFeatureContract",
    "assert_feature_vocab_width",
    "assert_model_matches_checkpoint_feature_contract",
    "resolve_and_assert_model_feature_contract",
    "resolve_checkpoint_feature_contract",
]
