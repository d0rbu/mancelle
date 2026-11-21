"""AxBench-Base adapter for main's clustered context-cache metadata.

The shipped cache architecture is
:func:`steering.eval.context_cache_io.clustered_context_cache_metadata`.
AxBench-Base adds exactly one field to that established metadata contract:
``feature_contract_fingerprint``.  A separate ``checkpoint_selection_hash``
would be redundant rather than an independent check because
``ClusterFeatureContract`` already fingerprints
``selection.canonical_dict()``, whose payload includes ``selection_hash``.

The checkpoint-seed provenance passed below is reconstructed only from the
already-resolved contract.  It therefore matches the five optional
``checkpoint_*`` keys emitted by the cache producer without introducing a
parallel clustered-cache schema.
"""

from __future__ import annotations

from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.eval.clustered_cache_seed_contract import CheckpointSeedProvenance
from steering.eval.context_cache_io import clustered_context_cache_metadata


def clustered_context_cache_metadata_for_contract(
    contract: ClusterFeatureContract,
) -> dict:
    """Compose main's cache metadata and add the contract fingerprint."""

    checkpoint = contract.checkpoint
    checkpoint_seed_provenance = CheckpointSeedProvenance(
        resolved_tag=checkpoint.resolved_tag,
        step=checkpoint.step,
        model_sha256=checkpoint.model_file.sha256,
        metadata_sha256=checkpoint.metadata_file.sha256,
        preselection_seed=contract.selection.preselection_seed,
    )
    return {
        **clustered_context_cache_metadata(
            feature_mask=contract.feature_mask,
            feature_clusters=contract.consolidation,
            feature_vocab=checkpoint.feature_vocab,
            stats_source_sha256=contract.statistics.file_sha256,
            checkpoint_seed_provenance=checkpoint_seed_provenance,
        ),
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
    }


__all__ = ("clustered_context_cache_metadata_for_contract",)
