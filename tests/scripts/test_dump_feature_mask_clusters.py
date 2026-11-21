"""Cluster-aware recovery and refusal matrix for ``dump_feature_mask``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch as t

from errors import FeatureMaskError, FeatureVocabError
from scripts.analysis.dump_feature_mask import (
    CLUSTER_EFFECTIVE,
    CLUSTER_INERT,
    CLUSTER_PRE_EFFECTIVE,
    load_dumped_mask,
    render,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    append_feature_clusters_leaf,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    append_mask_leaf,
    canonical_dict,
)
from steering.conditional.feature_vocab import FeatureVocab

N_SAE = 8
STEP = 5


def _resolved(clusters: list[list[int]]) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(tuple(tuple(cluster) for cluster in clusters))
    descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=N_SAE,
    )
    return ResolvedFeatureClusters(
        source_path=Path("<cluster-dump-test>"),
        definition=definition,
        descriptor=descriptor,
    )


def _effective(
    mask: FeatureMask | None,
    resolved: ResolvedFeatureClusters,
) -> tuple[FeatureClusterConsolidation, FeatureVocab | None]:
    state = FeatureClusterConsolidation.build(resolved, mask)
    return state, effective_feature_vocab(state, N_SAE)


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    vocab: FeatureVocab | None,
    mask: FeatureMask | None,
    clusters: ResolvedFeatureClusters | None,
    include_mask_meta: bool = True,
    include_cluster_meta: bool = True,
    include_vocab_meta: bool = True,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = N_SAE if vocab is None else vocab.size
    state = {
        "encoder.feature_embed.weight": t.arange(rows * 4, dtype=t.float32).reshape(rows, 4),
        "encoder.magnitude_embed.weight": t.zeros(4, 4),
    }
    if vocab is not None:
        state["encoder.feature_vocab_ids"] = t.tensor(
            vocab.original_ids,
            dtype=t.int64,
        )
        state["encoder.feature_vocab_n_sae"] = t.tensor(N_SAE, dtype=t.int64)
    t.save(state, checkpoint_dir / f"model_{STEP:06d}.pt")

    meta: dict = {"step": STEP, "model_config": {"n_sae_features": N_SAE}}
    if include_mask_meta and mask is not None:
        meta["feature_mask_canonical"] = canonical_dict(mask)
    if include_cluster_meta and clusters is not None:
        meta.update(feature_clusters_checkpoint_metadata(clusters))
    if include_vocab_meta and vocab is not None:
        meta["feature_vocab"] = vocab.metadata_dict()
    (checkpoint_dir / f"meta_{STEP:06d}.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def _production_dir(
    root: Path,
    mask: FeatureMask | None,
    clusters: ResolvedFeatureClusters,
) -> Path:
    masked = append_mask_leaf(str(root), mask)
    return Path(append_feature_clusters_leaf(masked, clusters))


def test_clustered_effective_vocab_emits_mask_with_cluster_provenance(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    resolved = _resolved([[1, 2, 3]])
    state, vocab = _effective(mask, resolved)
    checkpoint_dir = _production_dir(tmp_path, mask, resolved)
    _write_checkpoint(checkpoint_dir, vocab=vocab, mask=mask, clusters=resolved)

    dumped = load_dumped_mask(checkpoint_dir)
    output = render(dumped)

    assert dumped.mask == mask
    assert dumped.feature_clusters is not None
    assert dumped.feature_clusters.resolved.descriptor == state.resolved.descriptor
    assert t.equal(
        dumped.feature_clusters.cluster_allowed_universe,
        state.cluster_allowed_universe,
    )
    assert dumped.cluster_classification == CLUSTER_EFFECTIVE
    assert "feature_clusters participated" in output
    assert resolved.descriptor.content_sha256 in output
    assert "consolidated_away=2" in output
    assert "cluster-aware effective feature universe" in output


def test_clustered_pre_effective_vocab_emits_mask_with_migration_note(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    resolved = _resolved([[1, 2, 3]])
    legacy_vocab = FeatureVocab.from_mask(mask, N_SAE)
    checkpoint_dir = _production_dir(tmp_path, mask, resolved)
    _write_checkpoint(
        checkpoint_dir,
        vocab=legacy_vocab,
        mask=mask,
        clusters=resolved,
    )

    dumped = load_dumped_mask(checkpoint_dir)

    assert dumped.mask == mask
    assert dumped.cluster_classification == CLUSTER_PRE_EFFECTIVE
    assert "written by pre-effective-compaction code" in render(dumped)


def test_inert_clusters_prefer_effective_classification_when_both_match(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 6))
    resolved = _resolved([[1, 2, 3]])
    state, vocab = _effective(mask, resolved)
    assert state.is_identity_after_mask
    checkpoint_dir = _production_dir(tmp_path, mask, resolved)
    _write_checkpoint(checkpoint_dir, vocab=vocab, mask=mask, clusters=resolved)

    dumped = load_dumped_mask(checkpoint_dir)

    assert dumped.cluster_classification == CLUSTER_INERT
    assert "clusters configured but consolidate nothing after the mask" in render(dumped)


def test_clustered_vocab_matching_neither_candidate_refuses_with_all_sizes(
    tmp_path: Path,
):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    resolved = _resolved([[1, 2, 3]])
    wrong = FeatureVocab(N_SAE, (1, 5))
    checkpoint_dir = _production_dir(tmp_path, mask, resolved)
    _write_checkpoint(
        checkpoint_dir,
        vocab=wrong,
        mask=mask,
        clusters=resolved,
        include_vocab_meta=False,
    )

    with pytest.raises(FeatureVocabError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert "weights carry 2 rows" in message
    assert "effective vocabulary has 2 rows" in message
    assert "mask-only vocabulary has 4 rows" in message


def test_thinned_mask_meta_under_mask_leaf_refuses_before_vocab_reconstruction(
    tmp_path: Path,
):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(mask, resolved)
    checkpoint_dir = _production_dir(tmp_path, mask, resolved)
    _write_checkpoint(
        checkpoint_dir,
        vocab=vocab,
        mask=mask,
        clusters=resolved,
        include_mask_meta=False,
    )

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert Path(append_mask_leaf(str(tmp_path), mask)).name in message
    assert "not uniquely recoverable" in message
    assert "training config or wandb" in message


def test_cluster_only_compaction_refuses_to_invent_a_mask(tmp_path: Path):
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(None, resolved)
    checkpoint_dir = _production_dir(tmp_path, None, resolved)
    _write_checkpoint(checkpoint_dir, vocab=vocab, mask=None, clusters=resolved)

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert "cluster-only encoder compaction" in message
    assert resolved.descriptor.content_sha256 in message
    assert "consolidated_away=2" in message
    assert "will not invent a whitelist" in message


def test_fc_leaf_without_cluster_or_mask_meta_is_positive_ambiguous_evidence(
    tmp_path: Path,
):
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(None, resolved)
    checkpoint_dir = _production_dir(tmp_path, None, resolved)
    _write_checkpoint(
        checkpoint_dir,
        vocab=vocab,
        mask=None,
        clusters=resolved,
        include_cluster_meta=False,
    )

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert resolved.descriptor.dirname() in message
    assert "positive evidence" in message
    assert "reconstruction is ambiguous" in message


def test_fc_leaf_is_found_through_a_symlinked_checkpoint_path(tmp_path: Path):
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(None, resolved)
    target = _production_dir(tmp_path / "target", None, resolved)
    _write_checkpoint(
        target,
        vocab=vocab,
        mask=None,
        clusters=resolved,
        include_cluster_meta=False,
    )
    link = tmp_path / "parked_checkpoint"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(FeatureMaskError, match=resolved.descriptor.dirname()):
        load_dumped_mask(link)


def test_copied_out_thin_cluster_meta_still_refuses_without_guessing(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(mask, resolved)
    original = _production_dir(tmp_path / "original", mask, resolved)
    _write_checkpoint(
        original,
        vocab=vocab,
        mask=mask,
        clusters=resolved,
        include_mask_meta=False,
    )
    copied = tmp_path / "copied_out"
    shutil.copytree(original, copied)

    with pytest.raises(FeatureMaskError, match="cluster-only encoder compaction"):
        load_dumped_mask(copied)


def test_mask_and_fc_leaves_may_be_grandparents(tmp_path: Path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(mask, resolved)
    leaf = _production_dir(tmp_path, mask, resolved)
    nested = leaf / "grad_topk" / "sft"
    _write_checkpoint(
        nested,
        vocab=vocab,
        mask=mask,
        clusters=resolved,
        include_mask_meta=False,
    )

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(nested)
    message = str(exc.value)
    assert Path(append_mask_leaf(str(tmp_path), mask)).name in message
    assert resolved.descriptor.dirname() in str(nested.resolve())


def test_ambiguous_dual_mask_leaves_are_all_named_in_cluster_refusal(tmp_path: Path):
    whitelist = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    blacklist = FeatureMask(FeatureMaskMode.blacklist, (0, 4, 5, 7))
    resolved = _resolved([[1, 2, 3]])
    _, vocab = _effective(whitelist, resolved)
    first = Path(append_mask_leaf(str(tmp_path), whitelist))
    second = Path(append_mask_leaf(str(first), blacklist))
    checkpoint_dir = Path(append_feature_clusters_leaf(str(second), resolved))
    _write_checkpoint(
        checkpoint_dir,
        vocab=vocab,
        mask=whitelist,
        clusters=resolved,
        include_mask_meta=False,
    )

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert first.name in message
    assert second.name in message


def test_ambiguous_dual_fc_leaves_without_meta_are_all_named(tmp_path: Path):
    resolved_a = _resolved([[1, 2, 3]])
    resolved_b = _resolved([[4, 5]])
    _, vocab = _effective(None, resolved_a)
    first = Path(append_feature_clusters_leaf(str(tmp_path), resolved_a))
    checkpoint_dir = Path(append_feature_clusters_leaf(str(first), resolved_b))
    _write_checkpoint(
        checkpoint_dir,
        vocab=vocab,
        mask=None,
        clusters=resolved_a,
        include_cluster_meta=False,
    )

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert resolved_a.descriptor.dirname() in message
    assert resolved_b.descriptor.dirname() in message


def test_malformed_fc_like_leaves_are_not_cluster_evidence(tmp_path: Path):
    """Regex boundaries: fc_<16hex> and bl_<64hex> are NOT valid leaves.

    A one-character regex mutation (accepting the mask hash length after `fc_`,
    or the cluster hash length after `bl_`) would turn an ordinary masked
    checkpoint that happens to live under an fc-lookalike directory into
    "cluster evidence" and refuse it. The lookalike leaf below reuses the REAL
    mask leaf's 16-hex digest after `fc_`, the exact near-miss shape.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    masked = Path(append_mask_leaf(str(tmp_path), mask))
    mask_leaf_hex = masked.name.split("_", 1)[1]
    assert len(mask_leaf_hex) == 16
    lookalike = masked / f"fc_{mask_leaf_hex}"
    _write_checkpoint(lookalike, vocab=vocab, mask=mask, clusters=None)

    dumped = load_dumped_mask(lookalike)

    assert dumped.mask == mask
    assert dumped.feature_clusters is None
    assert dumped.cluster_classification is None


def test_a_64_hex_mask_prefix_is_not_a_mask_leaf(tmp_path: Path):
    """The inverse near-miss: bl_<64hex> must not satisfy MASK_LEAF_RE.

    Built as a thin-meta checkpoint (no mask canonical) whose ONLY path evidence
    is the malformed leaf: if the mask regex accepted 64 hex digits, the
    leaf-reconstruction path would engage (and fail differently); the correct
    behavior is the ordinary "cannot determine" refusal with no leaf evidence.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    resolved = _resolved([[1, 2, 3]])
    fake = tmp_path / f"bl_{resolved.descriptor.identity_sha256()}"
    _write_checkpoint(
        fake,
        vocab=vocab,
        mask=mask,
        clusters=None,
        include_mask_meta=False,
    )

    with pytest.raises(FeatureMaskError) as exc:
        load_dumped_mask(fake)
    assert "cannot determine the feature_mask's MODE" in str(exc.value)


def test_descriptor_width_mismatch_names_the_checkpoint(tmp_path: Path):
    """A cluster descriptor recorded at a different SAE width than the weights.

    Must raise the checkpoint_dir-naming corruption error added beside the
    round-trip classifier — not effective_feature_vocab's generic wiring-drift
    message, which names neither the checkpoint nor which half is wrong.
    """
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 6))
    definition = FeatureClusterDefinition(((1, 2, 3),))
    wide_descriptor = FeatureClusterDescriptor.from_definition(
        definition,
        n_sae_features=N_SAE * 2,
    )
    wide_resolved = ResolvedFeatureClusters(
        source_path=Path("<cluster-dump-test>"),
        definition=definition,
        descriptor=wide_descriptor,
    )
    vocab = FeatureVocab.from_mask(mask, N_SAE)
    checkpoint_dir = _production_dir(tmp_path, mask, wide_resolved)
    _write_checkpoint(checkpoint_dir, vocab=vocab, mask=mask, clusters=wide_resolved)

    with pytest.raises(FeatureVocabError) as exc:
        load_dumped_mask(checkpoint_dir)
    message = str(exc.value)
    assert str(checkpoint_dir) in message
    assert f"n_sae_features={N_SAE * 2}" in message
    assert "describe different models" in message
