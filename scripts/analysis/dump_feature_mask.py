"""Emit a checkpoint's exact ``feature_mask`` as a paste-ready config (issue #695 §4.9.4).

Under encoder feature-dictionary compaction, the vocabulary is a property of the
**checkpoint**, not of the downstream config: evaluating (or SFT-ing) a compacted
checkpoint requires re-specifying the EXACT mask it was trained with, and nothing is
inferred on the operator's behalf. For the default frequency blacklist that means
pointing ``feature_mask.path`` at ``{feature_dir}/blacklist/freq_min*_max*.txt`` — a
**generated** artifact that a ``feature_dir`` move or a statistics recompute destroys.

This tool removes that hostage situation. The checkpoint already carries the truth twice
over — in ``meta.feature_mask_canonical`` and, once compacted, in the load-bearing
``encoder.feature_vocab_ids`` buffer — so the mask is recovered from the checkpoint
**alone** and printed ready to paste. Nothing is silent: the operator explicitly runs
this and explicitly pastes the result.

One qualifier on "alone": when the metadata records no mask, the weights give the allowed
IDS but not the MODE, and mode is load-bearing (``FeatureMask.dirname()`` hashes ``(mode,
ids)``). It is recovered from the ``bl_``/``wl_`` mask leaf on the checkpoint's PATH — an
ANCESTOR of the directory holding ``model_*.pt``, since production nests the gradient and
SFT phases *below* that leaf. A checkpoint copied out of the tree the training run wrote it
into has genuinely lost its mode, and this refuses rather than guess it.

    uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir> [--step N] [--format yaml|txt]

Both sources are cross-checked against each other whenever both exist. If the metadata's
mask does not reproduce the weights' vocabulary, one of the two is wrong and there is no
way to tell which — so this refuses to emit either (``FeatureVocabError``) rather than
hand back a mask that would reinterpret every embedding row.

"No mask" is likewise a CLAIM, and it must be earned rather than assumed. An uncompacted
encoder is **not** evidence that no mask was configured: every pre-#695 checkpoint was
trained with the default frequency blacklist and never compacted, and SFT off such a base
inherits the shape. So the PATH is consulted as a third source. If it carries a
``bl_``/``wl_`` leaf while the metadata records nothing and the encoder is uncompacted, the
three sources CONTRADICT each other, the mask's ids survive in none of them (the leaf is a
one-way hash), and reporting "no feature_mask is configured" would hand the operator a
config whose ``checkpoint_dir`` addresses a different directory entirely. That case raises.
The clean "no mask" exit requires all three to agree.

Streams: the paste-ready artifact goes to **stdout** (redirect it to a file); progress and
provenance go to **stderr** via loguru. Provenance is *also* embedded as ``#`` comments in
the artifact itself, so a redirected dump still records where it came from.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch as t
from loguru import logger

from errors import FeatureClustersError, FeatureMaskError, FeatureVocabError
from nanochat.checkpoint_manager import find_last_step
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FEATURE_CLUSTERS_CANONICAL_KEY,
    FEATURE_CLUSTERS_DESCRIPTOR_KEY,
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    feature_mask_dirname,
)
from steering.conditional.feature_vocab import (
    META_KEY,
    VOCAB_IDS_KEY,
    FeatureVocab,
    feature_vocab_from_state_dict,
)

FORMAT_YAML = "yaml"
FORMAT_TXT = "txt"
FORMATS = (FORMAT_YAML, FORMAT_TXT)
FEATURE_ID_PREVIEW_LIMIT = 8

# The meta key written by the training scripts (see feature_mask.py's
# ``assert_resume_feature_mask_matches``, which reads the same key).
META_MASK_KEY = "feature_mask_canonical"

# Where ``mask`` came from. "meta" = the recorded canonical mask; "vocab" = reconstructed
# from the weights because the meta carried none; "none" = no mask at all.
SOURCE_META = "meta"
SOURCE_VOCAB = "vocab"
SOURCE_NONE = "none"

# A mask leaf exactly as ``FeatureMask.dirname()`` writes it: ``{bl|wl}_{16hex}``, a 64-bit
# sha256 prefix over the canonical ``(mode, ids)``. Anchored on purpose — a directory merely
# *starting* with ``bl_`` (``bl_experiments/``) is not a leaf, and treating it as one would
# turn the honest "no mask" exit into a spurious failure.
MASK_LEAF_RE = re.compile(r"^(?:bl|wl)_[0-9a-f]{16}$")

# A static-cluster leaf carries the full 256-bit descriptor-identity digest.  It is
# deliberately separate from MASK_LEAF_RE: accepting a 16-hex abbreviation here would
# turn an unrelated hand-named directory into evidence that checkpoint semantics were
# thinned.
FC_LEAF_RE = re.compile(r"^fc_[0-9a-f]{64}$")

CLUSTER_EFFECTIVE = "effective"
CLUSTER_PRE_EFFECTIVE = "pre_effective"
CLUSTER_INERT = "inert"


@dataclass(frozen=True)
class DumpedMask:
    """A checkpoint's feature mask, recovered from disk and cross-checked."""

    checkpoint_dir: Path
    step: int
    n_sae_features: int
    mask: FeatureMask | None
    vocab: FeatureVocab | None
    source: str
    feature_clusters: FeatureClusterConsolidation | None = None
    cluster_classification: str | None = None

    @property
    def meta_filename(self) -> str:
        return f"meta_{self.step:06d}.json"


def _read_meta(checkpoint_dir: Path, step: int) -> dict:
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _read_state_dict(checkpoint_dir: Path, step: int) -> dict:
    """Read the weights LAZILY.

    This tool needs exactly three numbers out of the file — ``encoder.feature_vocab_ids``,
    ``encoder.feature_vocab_n_sae``, and ``encoder.feature_embed.weight``'s row count. An
    eager load deserialises every parameter of a ~660M-param checkpoint (GBs of host RAM,
    over ~5-9 MB/s remote mounts) to get them, which rather undercuts a module that
    advertises reading a checkpoint's vocabulary "without loading the weights".

    ``mmap=True`` maps the storages instead of copying them, so only the pages actually
    touched are ever faulted in. Measured on a 256 MB checkpoint: 1.6 MB peak RSS vs
    245.8 MB eager, for byte-identical values. (It requires the zip serialisation format
    ``torch.save`` has produced by default since 1.6 — i.e. every checkpoint this repo
    writes.)
    """
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {model_path}")
    model_data = t.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod. Every sibling
    # loader strips it (nanochat/checkpoint_manager.py's build_model and load_checkpoint;
    # scripts/preprocess/convert_decoder_to_encdec.py) — without it a compile-saved encdec
    # checkpoint carries no bare ``encoder.feature_embed.weight``, so
    # ``feature_vocab_from_state_dict`` rejects it as "is this an encoder-decoder
    # checkpoint?": a wrong answer to a question the operator did not ask. Rebinding the
    # keys copies no storage, so the mmap laziness above survives.
    return {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}


def _mask_from_meta(meta: dict, *, meta_filename: str) -> FeatureMask | None:
    """The mask the training run actually resolved, or ``None`` if it recorded none."""
    raw = meta.get(META_MASK_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise FeatureMaskError(
            f"{meta_filename}: {META_MASK_KEY!r} must be a mapping "
            f'{{"mode": ..., "ids": [...]}}, got {type(raw).__name__}'
        )
    if "mode" not in raw or "ids" not in raw:
        raise FeatureMaskError(
            f"{meta_filename}: {META_MASK_KEY!r} must carry both 'mode' and 'ids', "
            f"got keys {sorted(raw)}"
        )
    try:
        return FeatureMask(mode=FeatureMaskMode(str(raw["mode"])), ids=tuple(raw["ids"]))
    except (FeatureMaskError, ValueError, TypeError) as exc:
        raise FeatureMaskError(f"{meta_filename}: {META_MASK_KEY!r} is invalid: {exc}") from exc


def _clusters_from_meta(
    meta: dict,
    *,
    meta_filename: str,
) -> ResolvedFeatureClusters | None:
    """Strictly reconstruct canonical static clusters without consulting a source file.

    Cluster checkpoint metadata intentionally omits ``source_path`` because the canonical
    definition and descriptor are the semantic record.  ``ResolvedFeatureClusters`` keeps
    a path for provenance, so this supplies a labelled placeholder; consolidation never
    reads it.  Parsing through the public strict constructors, followed by the resolved
    record's descriptor/definition cross-check, prevents this diagnostic tool from
    accepting metadata that training itself would reject.
    """
    has_descriptor = FEATURE_CLUSTERS_DESCRIPTOR_KEY in meta
    has_canonical = FEATURE_CLUSTERS_CANONICAL_KEY in meta
    if not has_descriptor and not has_canonical:
        return None
    if not has_descriptor or not has_canonical:
        missing = (
            FEATURE_CLUSTERS_DESCRIPTOR_KEY
            if not has_descriptor
            else FEATURE_CLUSTERS_CANONICAL_KEY
        )
        present = (
            FEATURE_CLUSTERS_CANONICAL_KEY
            if not has_descriptor
            else FEATURE_CLUSTERS_DESCRIPTOR_KEY
        )
        raise FeatureClustersError(
            f"{meta_filename}: incomplete static feature-cluster metadata: {present!r} "
            f"is present but {missing!r} is missing"
        )
    try:
        descriptor = FeatureClusterDescriptor.from_dict(meta[FEATURE_CLUSTERS_DESCRIPTOR_KEY])
        definition = FeatureClusterDefinition.from_mapping(
            meta[FEATURE_CLUSTERS_CANONICAL_KEY],
            n_sae_features=descriptor.n_sae_features,
        )
        return ResolvedFeatureClusters(
            source_path=Path(f"<reconstructed-from-{meta_filename}>"),
            definition=definition,
            descriptor=descriptor,
        )
    except FeatureClustersError as exc:
        raise FeatureClustersError(
            f"{meta_filename}: invalid static feature-cluster metadata: {exc}"
        ) from exc


def _assert_meta_vocab_agrees(
    meta: dict, vocab: FeatureVocab | None, *, meta_filename: str, checkpoint_dir: Path
) -> None:
    """Cross-check meta's ``feature_vocab`` mirror against the load-bearing buffers.

    The mirror is a convenience fingerprint, not the source of truth (§4.5) — so its
    ABSENCE is fine. Its *disagreement* is not: it means the two halves of the checkpoint
    describe different models.
    """
    recorded = meta.get(META_KEY)
    if recorded is None:
        return
    expected = vocab.metadata_dict() if vocab is not None else None
    if recorded != expected:
        raise FeatureVocabError(
            f"{checkpoint_dir}: {meta_filename} records {META_KEY}={recorded!r} but the "
            f"weights carry {expected!r}. The metadata and the weights describe different "
            "models — the checkpoint is corrupt or was assembled by hand."
        )


def _sample(ids: set[int]) -> str:
    """A bounded, sorted preview of a feature-id set for an error message.

    Display truncation only — the full count is always stated, so nothing is hidden. The
    sets here can hold tens of thousands of ids (V=61,835 on the real 65k dictionary
    under the shipped (1e-5, 0.5) band).
    """
    if not ids:
        return "none"
    shown = sorted(ids)[:FEATURE_ID_PREVIEW_LIMIT]
    body = ", ".join(str(i) for i in shown)
    suffix = (
        f", ... (+{len(ids) - FEATURE_ID_PREVIEW_LIMIT} more)"
        if len(ids) > FEATURE_ID_PREVIEW_LIMIT
        else ""
    )
    return f"{len(ids)} feature(s) [{body}{suffix}]"


def _assert_mask_reproduces_vocab(
    mask: FeatureMask, vocab: FeatureVocab, *, source: str, checkpoint_dir: Path
) -> None:
    """The mask must re-derive the checkpoint's compact table exactly.

    ``FeatureVocab.from_mask`` is the same pure function the trainer used to size the
    tables, so this is a genuine round-trip, not a restatement of what we just parsed.
    """
    derived = FeatureVocab.from_mask(mask, vocab.n_sae_features)
    if derived is None:
        raise FeatureVocabError(
            f"{checkpoint_dir}: the recovered feature_mask (mode={mask.mode.value}, "
            f"{len(mask.ids)} ids) allows ALL {vocab.n_sae_features} features — an "
            "identity universe compacts nothing, so it cannot have produced this "
            f"COMPACTED checkpoint (V={vocab.size} embedding rows)."
        )
    if derived != vocab:
        where = (
            f"{META_MASK_KEY} in the metadata"
            if source == SOURCE_META
            else "the reconstructed mask"
        )
        # Sizes alone are NOT a sufficient diagnostic: the ids can differ while V agrees
        # (a whitelist of {1,4,5} vs a vocab of {1,4,6}), and "allows 3, weights carry 3"
        # reads as a contradiction. Name the actual disagreeing features.
        allowed = set(derived.original_ids)
        present = set(vocab.original_ids)
        raise FeatureVocabError(
            f"{checkpoint_dir}: {where} does not reproduce the checkpoint's compacted "
            f"vocabulary. The mask (mode={mask.mode.value}, {len(mask.ids)} ids) allows "
            f"{derived.size} of {vocab.n_sae_features} features; the weights carry "
            f"{vocab.size} embedding rows ({VOCAB_IDS_KEY}). "
            f"Allowed by the mask but ABSENT from the weights: {_sample(allowed - present)}. "
            f"Present in the weights but EXCLUDED by the mask: {_sample(present - allowed)}. "
            "The metadata and the weights disagree, and there is no way to tell which is "
            "right — so no mask is emitted. Do not guess: recover the mask from the run "
            "that produced this checkpoint."
        )


def _path_component_names(checkpoint_dir: Path) -> set[str]:
    """Every directory NAME on the checkpoint's path — the directory itself and each parent.

    The path is walked both AS GIVEN and fully RESOLVED, so a symlink pointing *into* a mask
    leaf (how operators park checkpoints on scratch disks) still finds the leaf on the far
    side of the link, while a checkpoint COPIED out of its leaf finds nothing — because a
    copy really has lost the record, and inventing a mode for it is the bug. ``Path``
    already normalises a trailing slash, so ``.../bl_x/sft/`` and ``.../bl_x/sft`` walk
    identically.
    """
    names: set[str] = set()
    for base in (checkpoint_dir, checkpoint_dir.resolve()):
        for ancestor in (base, *base.parents):
            if ancestor.name:
                names.add(ancestor.name)
    return names


def _mask_leaves_on_path(checkpoint_dir: Path) -> list[str]:
    """Every ``{bl|wl}_{16hex}`` component on the checkpoint's path.

    Positive evidence that a feature_mask WAS configured, independent of what the metadata
    and the weights say — production writes every masked checkpoint under
    ``append_mask_leaf()``. Used to catch the case where both of the other two sources are
    silent and would otherwise be read as "there was no mask".
    """
    return sorted(n for n in _path_component_names(checkpoint_dir) if MASK_LEAF_RE.match(n))


def _fc_leaves_on_path(checkpoint_dir: Path) -> list[str]:
    """Every ``fc_<64hex>`` component on the given and resolved checkpoint paths."""
    return sorted(n for n in _path_component_names(checkpoint_dir) if FC_LEAF_RE.match(n))


def _vocab_size(vocab: FeatureVocab | None, n_sae_features: int) -> int:
    """Physical row count represented by a normalized optional vocabulary."""
    return n_sae_features if vocab is None else vocab.size


def _classify_clustered_vocab(
    mask: FeatureMask,
    resolved_clusters: ResolvedFeatureClusters,
    checkpoint_vocab: FeatureVocab | None,
    n_sae_features: int,
    *,
    checkpoint_dir: Path,
) -> tuple[FeatureClusterConsolidation, str]:
    """Round-trip a clustered checkpoint against current and pre-migration candidates.

    Effective matching is checked first.  That order is load-bearing for inert clusters:
    when consolidation removes no allowed member, the effective and mask-only candidates
    coincide, and the truthful classification is "configured but inert", not "written by
    old code".
    """
    # Width agreement is checked HERE, before the constructors, so the failure names
    # the checkpoint the way every other diagnostic in this file does. Without it the
    # mismatch would surface as effective_feature_vocab's generic wiring-drift raise,
    # which names neither the checkpoint nor which half of it is wrong. The thin-meta
    # branch (_raise_cluster_evidence_without_mask) carries the same pre-check.
    if resolved_clusters.n_sae_features != n_sae_features:
        raise FeatureVocabError(
            f"{checkpoint_dir}: the feature_clusters descriptor records "
            f"n_sae_features={resolved_clusters.n_sae_features}, but the checkpoint "
            f"weights describe {n_sae_features}. The metadata and the weights describe "
            "different models — the checkpoint is corrupt or was assembled by hand."
        )
    consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)
    effective_vocab = effective_feature_vocab(consolidation, n_sae_features)
    mask_only_vocab = FeatureVocab.from_mask(mask, n_sae_features)

    if checkpoint_vocab == effective_vocab:
        classification = (
            CLUSTER_INERT if consolidation.is_identity_after_mask else CLUSTER_EFFECTIVE
        )
        return consolidation, classification
    if checkpoint_vocab == mask_only_vocab:
        return consolidation, CLUSTER_PRE_EFFECTIVE

    raise FeatureVocabError(
        f"{checkpoint_dir}: clustered checkpoint vocabulary matches neither supported "
        "candidate. The weights carry "
        f"{_vocab_size(checkpoint_vocab, n_sae_features)} rows; the cluster-aware "
        f"effective vocabulary has {_vocab_size(effective_vocab, n_sae_features)} rows; "
        f"the pre-effective-compaction mask-only vocabulary has "
        f"{_vocab_size(mask_only_vocab, n_sae_features)} rows. The feature_mask, "
        "feature_clusters metadata, and weights disagree, so no mask is emitted."
    )


def _raise_cluster_evidence_without_mask(
    checkpoint_dir: Path,
    *,
    n_sae_features: int,
    resolved_clusters: ResolvedFeatureClusters | None,
    mask_leaves: list[str],
    fc_leaves: list[str],
    meta_filename: str,
) -> None:
    """Refuse every thin-meta cluster case before vocab-derived mask reconstruction.

    Effective ids cannot distinguish a member excluded by the mask from one consolidated
    away by its cluster.  Feeding those ids to ``_mask_from_vocab_and_leaf`` would invent
    a strict subset of the original mask universe and either miss the path hash or, under
    a collision, emit the wrong config.  Cluster evidence therefore owns this branch.
    """
    if resolved_clusters is not None:
        descriptor = resolved_clusters.descriptor
        if descriptor.n_sae_features != n_sae_features:
            raise FeatureClustersError(
                f"{checkpoint_dir}: {meta_filename} cluster descriptor records "
                f"n_sae_features={descriptor.n_sae_features}, but the checkpoint weights "
                f"record {n_sae_features}"
            )
        source_counts = (
            f"descriptor content_sha256={descriptor.content_sha256}, "
            f"source_clusters={resolved_clusters.definition.n_clusters}, "
            f"source_members={resolved_clusters.definition.n_members}"
        )
        if mask_leaves:
            raise FeatureMaskError(
                f"{checkpoint_dir}: a feature_mask provably existed because the path "
                f"contains mask leaf/leaves {mask_leaves!r}, but {meta_filename} carries "
                f"no {META_MASK_KEY!r}. Static feature clusters are also recorded "
                f"({source_counts}). A clustered checkpoint's effective vocabulary cannot "
                "distinguish mask-excluded members from consolidated-away members, so the "
                "mask is not uniquely recoverable from the weights. Recover the exact "
                "feature_mask from the training config or wandb run; no mask is emitted."
            )
        consolidation = FeatureClusterConsolidation.build(resolved_clusters, None)
        counts = consolidation.metadata_dict()["counts"]
        raise FeatureMaskError(
            f"{checkpoint_dir}: this is cluster-only encoder compaction: "
            f"{meta_filename} carries static feature_clusters ({source_counts}, "
            f"consolidated_away={counts['consolidated_away']}) but no "
            f"{META_MASK_KEY!r}, and no bl_/wl_ mask leaf appears on the path. There was "
            "no feature_mask to dump. This tool will not invent a whitelist from the "
            "effective vocabulary; configure the same cluster definition from the "
            "training run."
        )

    named_fc = ", ".join(repr(leaf) for leaf in fc_leaves)
    mask_note = f" The path also contains mask leaf/leaves {mask_leaves!r}." if mask_leaves else ""
    raise FeatureMaskError(
        f"{checkpoint_dir}: the path contains static feature-cluster leaf/leaves "
        f"{named_fc}, positive evidence that clusters were configured, but "
        f"{meta_filename} carries neither {FEATURE_CLUSTERS_DESCRIPTOR_KEY!r} nor "
        f"{FEATURE_CLUSTERS_CANONICAL_KEY!r} and no {META_MASK_KEY!r}.{mask_note} "
        "The effective vocabulary cannot distinguish mask-excluded ids from "
        "consolidated-away cluster members, so reconstruction is ambiguous. Recover the "
        "feature_mask and feature_clusters from the training config or wandb run."
    )


def _mask_from_vocab_and_leaf(
    vocab: FeatureVocab, n_sae_features: int, *, checkpoint_dir: Path
) -> FeatureMask:
    """Reconstruct the mask when the meta records none — MODE included.

    The vocabulary recovers the allowed UNIVERSE but NOT the mode, and that is not a
    cosmetic gap: ``FeatureMask.dirname()`` hashes ``(mode, ids)``, so a whitelist of the
    vocab and a blacklist of its complement describe the *identical* universe yet hash to
    DIFFERENT checkpoint leaves (``wl_<hex>`` vs ``bl_<hex>``) and different
    ``compute_selection_hash`` values. The default frequency band always produces a
    BLACKLIST, so emitting a whitelist handed the operator a config that addresses a
    directory holding no checkpoint — the recovery loop did not close for the very case
    this tool exists to serve.

    The mode is not in the weights. It IS on disk: production stores every masked checkpoint
    under ``append_mask_leaf(...)``, so a mask leaf's own NAME records the mode. But that
    leaf is an ANCESTOR, not necessarily the basename — nothing stops there:

    * ``conditional_encdec_train.py`` composes
      ``append_gradient_leaf(append_mask_leaf(.../s{seg}/{fs}, mask), fs)``, so a
      gradient-ranked pretrain lands in ``.../bl_<hex>/grad_{mode}_mf{n}/``;
    * ``conditional_encdec_sft.py`` composes
      ``_build_sft_checkpoint_dir(append_mask_leaf(pretrain_dir, mask), dataset)``, so EVERY
      SFT checkpoint lands in ``.../bl_<hex>/{dataset}_sft_conditioned_checkpoints/`` — and
      an SFT run off a gradient-ranked base (the ``base_checkpoint_dir`` override) puts the
      leaf two levels up.

    So the depth is not fixed, and matching only ``checkpoint_dir.name`` dead-ended with
    "cannot determine the MODE" on precisely the checkpoints #695's guards send operators
    here to recover. Search the whole path instead, and keep the candidate whose dirname a
    component reproduces. That match is not a heuristic: the dirname is a 64-bit hash of
    ``(mode, ids)`` over the ids we just derived from THIS checkpoint's weights, so a hit
    PROVES the mode rather than guessing it. No hit ⇒ raise: the mode is genuinely gone.
    """
    whitelist = FeatureMask(mode=FeatureMaskMode.whitelist, ids=vocab.original_ids)
    allowed = set(vocab.original_ids)
    blacklist = FeatureMask(
        mode=FeatureMaskMode.blacklist,
        ids=tuple(i for i in range(n_sae_features) if i not in allowed),
    )

    names = _path_component_names(checkpoint_dir)
    matched = [c for c in (blacklist, whitelist) if feature_mask_dirname(c) in names]
    if len(matched) == 1:
        return matched[0]

    if matched:
        # Both leaves on one path. They name the two modes of the SAME universe, so they
        # cannot both be what this checkpoint was trained with, and letting ancestor ORDER
        # break the tie would be a guess wearing a proof's clothes. Production writes exactly
        # one mask leaf per path, so this tree was assembled by hand.
        raise FeatureMaskError(
            f"{checkpoint_dir}: AMBIGUOUS — this path carries BOTH candidate mask leaves "
            f"({feature_mask_dirname(blacklist)} and {feature_mask_dirname(whitelist)}), "
            "which are the blacklist and whitelist namings of the one allowed universe "
            f"({vocab.size} of {n_sae_features} features) this checkpoint's weights carry. "
            "They cannot both be the mode it was trained with, and choosing between them "
            "would be a guess — so no config is emitted. Production writes exactly ONE mask "
            "leaf per checkpoint path, so this tree was assembled by hand: point this tool "
            "at the checkpoint under its ORIGINAL leaf."
        )

    raise FeatureMaskError(
        f"{checkpoint_dir}: cannot determine the feature_mask's MODE, so no config can be "
        "emitted that is guaranteed to address this checkpoint.\n"
        f"The weights give the allowed universe ({vocab.size} of {n_sae_features} "
        "features), but the mode is recorded in neither the meta "
        f"({META_MASK_KEY!r} is absent) nor anywhere on this checkpoint's path.\n"
        "A whitelist of the universe and a blacklist of its complement describe the SAME "
        "universe but hash to DIFFERENT checkpoint leaves, so guessing would hand you a "
        "config pointing at a directory with no checkpoint in it. The two possibilities "
        f"are:\n"
        f"  blacklist -> {feature_mask_dirname(blacklist)}\n"
        f"  whitelist -> {feature_mask_dirname(whitelist)}\n"
        "but neither names this directory or ANY of its parents.\n"
        "Production always writes a masked checkpoint under its mask leaf (append_mask_leaf), "
        "with the gradient/SFT phase nested BELOW it — so a path carrying no such leaf means "
        "this checkpoint was copied or renamed out of the tree the training run wrote it "
        "into. Point this tool at it where it was written, and the mode is recovered exactly."
    )


def load_dumped_mask(checkpoint_dir: str | Path, step: int | None = None) -> DumpedMask:
    """Recover a checkpoint's feature mask from disk, cross-checking meta against weights.

    ``step=None`` resolves to the latest checkpoint in the directory.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")
    resolved_step = int(step) if step is not None else int(find_last_step(str(checkpoint_dir)))

    # The weights are the load-bearing record: they alone define the encoder's shape, and
    # ``model_*.pt`` is what makes a step exist at all — so read (and complain about) them
    # first.
    state = _read_state_dict(checkpoint_dir, resolved_step)
    meta = _read_meta(checkpoint_dir, resolved_step)
    meta_filename = f"meta_{resolved_step:06d}.json"

    n_sae_features, vocab = feature_vocab_from_state_dict(state)
    _assert_meta_vocab_agrees(
        meta, vocab, meta_filename=meta_filename, checkpoint_dir=checkpoint_dir
    )

    resolved_clusters = _clusters_from_meta(meta, meta_filename=meta_filename)
    mask = _mask_from_meta(meta, meta_filename=meta_filename)
    source = SOURCE_META
    mask_leaves = _mask_leaves_on_path(checkpoint_dir)
    fc_leaves = _fc_leaves_on_path(checkpoint_dir)
    cluster_evidence = resolved_clusters is not None or bool(fc_leaves)
    if mask is None and cluster_evidence:
        _raise_cluster_evidence_without_mask(
            checkpoint_dir,
            n_sae_features=n_sae_features,
            resolved_clusters=resolved_clusters,
            mask_leaves=mask_leaves,
            fc_leaves=fc_leaves,
            meta_filename=meta_filename,
        )
    if mask is None and vocab is not None:
        mask = _mask_from_vocab_and_leaf(vocab, n_sae_features, checkpoint_dir=checkpoint_dir)
        source = SOURCE_VOCAB
    if mask is None:
        # Reaching here means the meta records no mask AND the encoder is uncompacted (a
        # compacted one either produced a mask above or raised). Two sources are silent —
        # but "silent" is not "there was no mask". The PATH is a third source, and if it
        # carries a mask leaf it says, positively, that one WAS configured.
        assert vocab is None, "a compacted vocab must already have produced a mask above"
        if mask_leaves:
            named = ", ".join(repr(leaf) for leaf in mask_leaves)
            raise FeatureMaskError(
                f"{checkpoint_dir}: INCONSISTENT — this checkpoint's path carries the mask "
                f"leaf {named}, which is how production records that a feature_mask WAS "
                "configured (append_mask_leaf writes every masked checkpoint under a "
                "{bl|wl}_<16hex> directory). But the checkpoint itself records nothing: "
                f"{meta_filename} carries no {META_MASK_KEY!r}, and the encoder is "
                f"UNCOMPACTED ({n_sae_features} rows, one per SAE feature), so the weights "
                "carry no vocabulary either.\n"
                "Neither surviving source holds the mask's IDS — the leaf is a ONE-WAY hash "
                "of (mode, ids), and an uncompacted table has a row for every feature, "
                "masked or not. The mask therefore CANNOT be recovered from this checkpoint, "
                "and no config can be emitted for it.\n"
                "This is NOT 'no feature_mask is configured', which is what this tool used "
                "to report here. That answer was worse than useless: a config with no mask "
                "composes a checkpoint_dir WITHOUT the leaf, so it would have addressed a "
                "DIFFERENT directory from the one you just pointed this tool at.\n"
                "An uncompacted encoder under a mask leaf is a REAL state — pre-#695 runs "
                "were masked but never compacted, and SFT off such a base inherits it — but "
                "those checkpoints record the mask in their metadata. This one does not, so "
                "its metadata is incomplete. Recover the mask from the training config or "
                "the run that wrote this checkpoint."
            )
        return DumpedMask(
            checkpoint_dir=checkpoint_dir,
            step=resolved_step,
            n_sae_features=n_sae_features,
            mask=None,
            vocab=None,
            source=SOURCE_NONE,
        )

    for feature_id in mask.ids:
        if feature_id >= n_sae_features:
            raise FeatureMaskError(
                f"{checkpoint_dir}: {meta_filename} records feature_mask id {feature_id}, "
                f"which is out of range [0, {n_sae_features}) for this checkpoint's SAE. "
                "The metadata and the weights disagree about the dictionary width."
            )

    cluster_state: FeatureClusterConsolidation | None = None
    cluster_classification: str | None = None
    if resolved_clusters is not None:
        cluster_state, cluster_classification = _classify_clustered_vocab(
            mask,
            resolved_clusters,
            vocab,
            n_sae_features,
            checkpoint_dir=checkpoint_dir,
        )
    elif fc_leaves:
        raise FeatureClustersError(
            f"{checkpoint_dir}: the path contains static feature-cluster leaf/leaves "
            f"{fc_leaves!r}, but {meta_filename} carries the feature_mask without the "
            f"required {FEATURE_CLUSTERS_DESCRIPTOR_KEY!r} and "
            f"{FEATURE_CLUSTERS_CANONICAL_KEY!r}. The cluster definition is required to "
            "verify whether the checkpoint vocabulary is effective or mask-only; recover "
            "the unthinned metadata from the training run."
        )
    elif vocab is not None:
        _assert_mask_reproduces_vocab(mask, vocab, source=source, checkpoint_dir=checkpoint_dir)

    return DumpedMask(
        checkpoint_dir=checkpoint_dir,
        step=resolved_step,
        n_sae_features=n_sae_features,
        mask=mask,
        vocab=vocab,
        source=source,
        feature_clusters=cluster_state,
        cluster_classification=cluster_classification,
    )


def _provenance(dumped: DumpedMask, mask: FeatureMask) -> list[str]:
    """The ``#`` header both formats carry, so a redirected dump records its own origin."""
    lines = [
        f"# feature_mask of checkpoint {dumped.checkpoint_dir} @ step {dumped.step}",
        f"# recovered by: uv run python -m scripts.analysis.dump_feature_mask "
        f"{dumped.checkpoint_dir} --step {dumped.step}",
    ]
    if dumped.source == SOURCE_META:
        lines.append(f"# source: {dumped.meta_filename} -> {META_MASK_KEY}")
    else:
        # Describe what was ACTUALLY emitted. This used to hard-code "Emitted as an explicit
        # whitelist of the checkpoint's N vocab ids" — printed directly above a body saying
        # ``mode: blacklist``, and counting the wrong ids into the bargain. The header is the
        # part the operator reads before pasting; it cannot contradict the body about the one
        # field that decides which directory the pasted config addresses.
        lines += [
            f"# source: RECONSTRUCTED from the weights ({VOCAB_IDS_KEY}) — "
            f"{dumped.meta_filename} carries no {META_MASK_KEY}.",
            f"# Emitted as an explicit {mask.mode.value} of {len(mask.ids)} feature id(s). "
            f"The MODE was recovered from the mask leaf {feature_mask_dirname(mask)!r} on "
            "this checkpoint's path; dirname() hashes (mode, ids), so that name PROVES this "
            "config addresses this exact checkpoint rather than merely asserting it.",
        ]
    if dumped.vocab is not None:
        lines.append(
            f"# encoder feature dictionary: COMPACTED to V={dumped.vocab.size} of "
            f"n_sae_features={dumped.n_sae_features} rows; verified to reproduce "
            f"{VOCAB_IDS_KEY}."
        )
    else:
        lines.append(
            f"# encoder feature dictionary: UNCOMPACTED ({dumped.n_sae_features} rows, one "
            "per SAE feature), so this mask is a selection-layer constraint only and could "
            "not be cross-checked against the weights."
        )
    if dumped.feature_clusters is not None:
        cluster_state = dumped.feature_clusters
        counts = cluster_state.metadata_dict()["counts"]
        lines.append(
            "# feature_clusters participated: "
            f"descriptor content_sha256={cluster_state.resolved.descriptor.content_sha256}; "
            f"source_clusters={counts['source_clusters']}; "
            f"source_members={counts['source_members']}; "
            f"surviving_representatives={counts['surviving_cluster_representatives']}; "
            f"consolidated_away={counts['consolidated_away']}."
        )
        if dumped.cluster_classification == CLUSTER_INERT:
            lines.append("# clusters configured but consolidate nothing after the mask.")
        elif dumped.cluster_classification == CLUSTER_PRE_EFFECTIVE:
            lines.append(
                "# checkpoint was written by pre-effective-compaction code; its "
                "mask-only vocabulary retained consolidated-away rows."
            )
        else:
            lines.append("# checkpoint vocabulary is the cluster-aware effective feature universe.")
    return lines


def _render_yaml(dumped: DumpedMask, mask: FeatureMask) -> str:
    ids = ", ".join(str(i) for i in mask.ids)
    lines = [
        *_provenance(dumped, mask),
        "feature_mask:",
        f"  mode: {mask.mode.value}",
        f"  ids: [{ids}]",
    ]
    return "\n".join(lines) + "\n"


def _render_txt(dumped: DumpedMask, mask: FeatureMask) -> str:
    lines = [
        *_provenance(dumped, mask),
        "#",
        "# This file carries IDS ONLY. Save it, then reference it with the mode above:",
        "#   feature_mask:",
        f"#     mode: {mask.mode.value}",
        "#     path: /abs/path/to/this/file.txt",
        *(str(i) for i in mask.ids),
    ]
    return "\n".join(lines) + "\n"


def _render_no_mask(dumped: DumpedMask) -> str:
    """The "genuinely no mask" report — and it is only emitted once that is EARNED.

    All THREE independent sources have to be silent, not two. The metadata records no mask,
    the encoder is uncompacted, and — the check this report used to skip — no ``bl_``/``wl_``
    leaf appears anywhere on the checkpoint's path. A leaf there would PROVE a mask was
    configured, and ``load_dumped_mask`` raises rather than reach here, because "leave
    feature_mask unset" would then send the operator to a directory this checkpoint does not
    live in.
    """
    return (
        "\n".join(
            [
                f"# No feature_mask is configured for checkpoint {dumped.checkpoint_dir} "
                f"@ step {dumped.step}.",
                f"# {dumped.meta_filename} carries no {META_MASK_KEY}, the encoder is "
                f"UNCOMPACTED ({dumped.n_sae_features} rows, one per SAE feature), and no "
                "bl_/wl_ mask leaf appears on the checkpoint's path. All three sources "
                "agree.",
                "# There is nothing to paste: leave feature_mask unset (null).",
            ]
        )
        + "\n"
    )


def render(dumped: DumpedMask, fmt: str = FORMAT_YAML) -> str:
    """The paste-ready artifact. ``yaml`` ⇒ a Hydra block; ``txt`` ⇒ ids for a ``path:``."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {list(FORMATS)}, got {fmt!r}")
    mask = dumped.mask
    if mask is None:
        return _render_no_mask(dumped)
    return _render_yaml(dumped, mask) if fmt == FORMAT_YAML else _render_txt(dumped, mask)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dump_feature_mask",
        description=(
            "Emit a checkpoint's exact feature_mask as a paste-ready config, so evaluating "
            "a compacted checkpoint does not depend on a generated blacklist .txt still "
            "existing on disk."
        ),
    )
    parser.add_argument("checkpoint_dir", type=Path, help="directory holding model_*.pt")
    parser.add_argument(
        "--step", type=int, default=None, help="checkpoint step (default: the latest)"
    )
    parser.add_argument("--format", choices=FORMATS, default=FORMAT_YAML, dest="fmt")
    args = parser.parse_args(argv)

    dumped = load_dumped_mask(args.checkpoint_dir, args.step)
    if dumped.mask is None:
        logger.info(
            f"{dumped.checkpoint_dir} @ step {dumped.step}: no feature_mask configured "
            f"(uncompacted encoder, {dumped.n_sae_features} rows)"
        )
    else:
        vocab_note = (
            f"compacted to V={dumped.vocab.size} of {dumped.n_sae_features}"
            if dumped.vocab is not None
            else f"uncompacted ({dumped.n_sae_features} rows)"
        )
        logger.info(
            f"{dumped.checkpoint_dir} @ step {dumped.step}: feature_mask "
            f"(mode={dumped.mask.mode.value}, {len(dumped.mask.ids)} ids) from "
            f"{'metadata' if dumped.source == SOURCE_META else 'the weights'}; "
            f"encoder {vocab_note}"
        )
        if dumped.feature_clusters is not None:
            counts = dumped.feature_clusters.metadata_dict()["counts"]
            assert dumped.cluster_classification is not None
            classification = {
                CLUSTER_EFFECTIVE: "cluster-aware effective vocabulary",
                CLUSTER_PRE_EFFECTIVE: "written by pre-effective-compaction code",
                CLUSTER_INERT: "clusters configured but consolidate nothing after the mask",
            }[dumped.cluster_classification]
            logger.info(
                f"feature_clusters participated: descriptor content_sha256="
                f"{dumped.feature_clusters.resolved.descriptor.content_sha256}; "
                f"source_clusters={counts['source_clusters']}, "
                f"source_members={counts['source_members']}, "
                f"consolidated_away={counts['consolidated_away']}; {classification}"
            )
    print(render(dumped, args.fmt), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
