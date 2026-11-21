"""Derive the encoder vocabulary from the effective feature universe.

Static feature clusters change which ORIGINAL SAE feature ids can occur after
consolidation: every surviving cluster keeps its smallest allowed member and the
other allowed members are consolidated away.  The encoder therefore needs rows for
the mask-allowed, non-consolidated ids, rather than for the mask universe alone.

This module is deliberately only a bridge between two narrower modules.
``feature_clusters.py`` owns cluster parsing and consolidation but must not know
compact ids; ``feature_vocab.py`` owns compact-id mechanics but must not import the
cluster schema.  Keeping the composition here also makes the mask-only path a
literal delegation to the pre-existing constructor.
"""

from __future__ import annotations

from errors import FeatureClustersError
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_vocab import FeatureVocab


def effective_feature_vocab(
    feature_universe: FeatureClusterConsolidation | FeatureMask | None,
    n_sae_features: int,
) -> FeatureVocab | None:
    """Return the ONLY production vocabulary constructor for cluster-aware models.

    ``feature_universe`` is the single object that decides which original ids get
    embedding rows.  A :class:`FeatureClusterConsolidation` already carries the
    fully resolved mask it was built against (``.feature_mask``), so callers with
    clusters hand over the consolidation ALONE — there is no separate mask
    parameter to drift out of agreement with it.  Callers without clusters hand
    over their ``FeatureMask | None`` exactly as they always did.

    The contract is the effective-vocabulary invariants E1--E7:

    * **E1 -- definition/delegation.** A ``FeatureMask`` or ``None`` literally
      delegates to :meth:`FeatureVocab.from_mask`, preserving its result and
      failure semantics.  A consolidation derives ids exactly from
      ``cluster_allowed_universe``.
    * **E2 -- purity/identity.** The result is a deterministic derived value of the
      resolved mask, canonical cluster content, and original SAE width.  It creates no
      selection hash input and no checkpoint-path leaf.
    * **E3 -- no-cluster identity.** The delegated path adds no behavior, state, or RNG
      draw to an unclustered model.
    * **E4 -- non-emptiness.** A valid mask/consolidation leaves at least one effective
      feature.  The explicit empty check below is a defensive invariant guard.
    * **E5 -- minimum-id preservation.** Cluster representatives are the smallest
      allowed member, so compact row zero denotes the same original id as in the
      mask-only vocabulary; padding semantics are unchanged.
    * **E6 -- row count.** ``V_eff`` equals the mask-allowed row count minus the number
      of consolidated-away members.
    * **E7 -- representative membership.** Every legal representative is present and
      every consolidated-away id is absent, so an away id fails the encoder's normal
      out-of-vocabulary guard.

    The SAE width is still rechecked against a consolidation: the consolidation is
    frozen to the width it was built at, and a disagreeing caller width would
    produce a plausible but wrong row dictionary.  Identity results normalize to
    ``None`` so a non-``None`` vocabulary continues to mean genuine compaction.
    Any other input type raises rather than guessing what universe it describes.
    """
    if feature_universe is None or isinstance(feature_universe, FeatureMask):
        return FeatureVocab.from_mask(feature_universe, n_sae_features)
    if not isinstance(feature_universe, FeatureClusterConsolidation):
        raise TypeError(
            "effective_feature_vocab expects a FeatureClusterConsolidation, a "
            f"FeatureMask, or None; got {type(feature_universe).__name__}"
        )

    if feature_universe.n_sae_features != n_sae_features:
        raise FeatureClustersError(
            "effective feature vocabulary SAE-width mismatch: "
            f"feature_clusters.n_sae_features={feature_universe.n_sae_features}, "
            f"n_sae_features={n_sae_features}"
        )

    ids = tuple(
        int(feature_id)
        for feature_id in feature_universe.cluster_allowed_universe.nonzero().flatten().tolist()
    )
    if not ids:
        raise FeatureClustersError(
            "effective feature vocabulary is empty after applying feature_mask and "
            "feature_clusters; this violates E4 because every surviving cluster must "
            "retain a representative"
        )
    vocab = FeatureVocab(n_sae_features=n_sae_features, original_ids=ids)
    return None if vocab.is_identity else vocab
