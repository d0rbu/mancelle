"""The operator-facing summary of encoder feature-dictionary compaction (issue #695).

Its own module, and deliberately so. This started out inside
``scripts/train/conditional_encdec_train.py`` — a torchrun entrypoint, importable by
nothing — so the two *other* programs that build a compacted encoder
(``conditional_encdec_sft.py``, ``convert_decoder_to_encdec.py``) each hand-rolled the
string instead. The copies drifted, and not cosmetically: only the training one printed
the feature-indexed **parameter saving**, so SFT and conversion operators never saw the
number that decides whether the run fits in VRAM.
"""

from __future__ import annotations

from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_clusters import FeatureClusterConsolidation
from steering.conditional.feature_mask import FeatureMask


def no_compaction_reason(
    feature_mask: FeatureMask | None,
    n_sae_features: int,
    *,
    feature_clusters: FeatureClusterConsolidation | None = None,
) -> str:
    """Why the effective vocabulary normalized to ``None``.

    Identity normalization conflates three distinct configurations: neither a mask nor
    clusters; an all-allowing mask whose clusters (if any) consolidate nothing after it;
    and no mask with inert clusters.  This function keeps those operator-facing reasons
    truthful.  The unclustered branches intentionally retain their exact legacy strings.

    A non-identity cluster consolidation cannot reach this function legitimately because
    it produces a compact vocabulary even when no mask is configured.

    Conflating those cases tells an operator that no mask or clusters are configured
    while they are looking at one in their config. All cases are true statements about
    the physical shape — there is no compaction — but only one is true about the
    configuration that produced it.

    Pure, and the only place the distinction is drawn, so the two callers that resolve a
    mask cannot drift apart on it.
    """
    if feature_clusters is None and feature_mask is None:
        return "no feature_mask is configured"
    if feature_mask is None:
        return (
            "no feature_mask is configured and the configured feature_clusters consolidate nothing"
        )
    reason = (
        f"the configured feature_mask (mode={feature_mask.mode.value}) allows ALL "
        f"{n_sae_features:,} features, so it restricts nothing to compact"
    )
    if feature_clusters is not None:
        reason += ", and the configured feature_clusters consolidate nothing after the mask"
    return reason


def format_compaction_summary(
    model: EncoderDecoderGPT,
    *,
    no_vocab_reason: str,
    vocab_source: str | None = None,
    consolidated_away: int | None = None,
) -> str:
    """One-line summary of the encoder feature-dictionary compaction (issue #695).

    This is the ONLY visibility an operator has into the *realised* compaction ratio.
    It cannot be known offline: it depends on the corpus and the frequency bounds. On the
    real 65k corpus the shipped (1e-5, 0.5) band prunes 3,701 of 65,536 (5.6%).

    A small ratio is NOT a reason to widen the band. Compaction's runtime cost is a single
    gather, so it pays at any ratio, and the band is a SELECTION policy that #695 has no
    business moving to flatter its own numbers.

    Deliberately plain and factual — no threshold, no low-yield warning. There is no
    magic constant anywhere in this design, not even one that changes what is printed.

    **The model is the ONLY source of shape.** The vocabulary and the original dictionary
    width used to be passed in alongside it, and nothing checked that the trio agreed —
    so a caller could hand in a vocabulary the model does not have and get the compaction
    ratio of a model that does not exist. Both are read off ``model`` now
    (``feature_vocab``; ``encoder.config.n_sae_features``), which
    ``EncoderDecoderGPT.__init__`` has already forced into agreement. A disagreeing pair
    is unrepresentable rather than merely unlikely.

    The provenance clauses stay PARAMETERS, because — unlike the shapes — they are facts
    the model provably cannot carry:

    * ``no_vocab_reason`` — why there is *no* vocabulary. REQUIRED, and deliberately has
      no default: ``feature_vocab is None`` is two different worlds (see
      :func:`no_compaction_reason`), and a default could only ever guess at which. The
      old default guessed "no feature_mask is configured", which is a flat lie both to an
      SFT run inheriting a full-width checkpoint *and* to any run whose mask allows
      everything. The mask-resolving callers get this from :func:`no_compaction_reason`;
      SFT — whose encoder shape comes from the CHECKPOINT, not from its own config — says
      what is actually true of it.
    * ``vocab_source`` — where a vocabulary came from. The trainer and the converter
      RESOLVE it from ``cfg.feature_mask`` (so there is nothing to add); SFT INHERITS it
      from the base checkpoint and can neither create nor widen one.
    * ``consolidated_away`` — how many of the omitted rows were removed specifically
      by static feature-cluster consolidation.  The model carries the effective shape,
      not this provenance fact, so the cluster-aware trainer supplies it explicitly.
      Like ``vocab_source``, it is a statement about a VOCABULARY: the uncompacted
      branch returns before either clause and a zero count is suppressed, so
      unclustered and inert-cluster runs log byte-identically to pre-cluster code.

    Requires a constructed ``model``: the parameter saving is computed from the *resolved*
    encoder config and the actual number of value-embedding tables, neither of which is
    knowable from the ratio-shaped config alone. A ``meta`` model is sufficient — nothing
    here reads data.
    """
    enc = model.encoder.config
    feature_vocab = model.feature_vocab
    n_sae_features = enc.n_sae_features

    # The uncompacted branch RETURNS here, exactly as it always has. Both provenance
    # clauses describe properties of a vocabulary; with no vocabulary there are no
    # "dropped rows" for ``consolidated_away`` to partition and nothing for
    # ``vocab_source`` to source, so appending either would change the log line of
    # every uncompacted run (SFT passes ``vocab_source`` unconditionally) — a
    # no-cluster byte-identity (E3) violation, not a cosmetic choice.
    if feature_vocab is None:
        return (
            "Encoder feature dictionary: no compaction "
            f"({n_sae_features:,} rows, the full SAE dictionary) — {no_vocab_reason}."
        )

    kv_dim = enc.n_kv_head * enc.head_dim
    n_ve = len(model.encoder.value_embeds)
    dropped = n_sae_features - feature_vocab.size
    saved = dropped * (enc.n_embd + n_ve * kv_dim)
    before = n_sae_features * (enc.n_embd + n_ve * kv_dim)
    kept_pct = 100.0 * feature_vocab.size / n_sae_features
    summary = (
        f"Encoder feature dictionary: {feature_vocab.size:,}/{n_sae_features:,} rows kept "
        f"({kept_pct:.1f}% of the SAE dictionary). "
        f"Feature-indexed encoder params: {before:,} -> {before - saved:,} "
        f"(saved {saved:,})."
    )
    if vocab_source is not None:
        summary = f"{summary} Vocabulary {vocab_source}."
    # Zero is suppressed, not printed: inert clusters consolidated nothing, and the
    # mask-only sentence above already accounts for every dropped row.
    if consolidated_away is not None and consolidated_away > 0:
        summary = (
            f"{summary} Of the dropped rows, {consolidated_away:,} were consolidated "
            "away by feature_clusters."
        )
    return summary
