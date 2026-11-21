"""Tests for the public clustered-evaluation feature-contract facade.

The frozen export list guards import compatibility. Functional tests cover
contract construction and resolution, representative targets, loaded-state
checks, and sidecar/run-marker I/O. Identity and checkpoint helpers have
separate coverage in ``test_cluster_contract_identity.py`` and
``test_cluster_checkpoint_pin.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import scripts.eval.axbench_base_eval as axbench_base_cli
import steering.conditional.splice_counterfactual as splice_counterfactual_module
import steering.eval.axbench_bpb_adv.sample_generation as sample_generation
import steering.eval.cluster_checkpoint_pin as checkpoint_pin_module
import steering.eval.cluster_feature_contract as contract_module
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    build_clustered_selection_validation,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    build_clustered_selection_shard_integrity,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import (
    FEATURE_EMBED_KEY,
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.splice_counterfactual import (
    packed_row_fingerprint,
    resolve_packed_row_fingerprint_strict,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_bpb_adv.cluster_preflight import (
    build_cluster_contract_request,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.cluster_feature_contract import (
    CheckpointPin,
    CheckpointSelectionContract,
    ClusterFeatureContractError,
    ClusterFeatureContractRequest,
    ClusterFeatureTarget,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    assert_cluster_contract_loaded_state,
    assert_loaded_cluster_target,
    build_cluster_feature_contract,
    ensure_feature_contract_run_marker,
    load_feature_contract_sidecar,
    resolve_cluster_feature_contract,
    validate_feature_contract_document,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    compute_selection_hash,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.sae_resolver import ActivationType, SAERelease, SAEWidth, resolve_sae
from steering.stats_io import load_statistics, save_percentile_checksum
from tests.axbench_base._multiprocess_helpers import run_gloo_multiprocess
from tests.axbench_base.fixtures.steering_stats import write_steering_stats

# ============================================================================
# Import-path preservation (plan constraint 5).
# ============================================================================

# Frozen from the oracle's own __all__ (.claude/worktrees/pr739-oracle/
# steering/eval/cluster_feature_contract.py @ f357458b), plus G2-D2's required
# additive ``stat_signature_now`` export. Do not "helpfully" resync this list
# from a live import of this module — the point is an independent, hardcoded
# expectation.
_EXPECTED_ALL = (
    "CONCEPT_PROJECTION_VERSION",
    "FEATURE_CONTRACT_SCHEMA_VERSION",
    "FEATURE_DIFFMEAN_PROJECTION_VERSION",
    "GENERIC_ROOFLINE_SIGNAL_VERSION",
    "MAX_SAFE_SELECTED_FEATURE_WIDTH",
    "PACKED_PERCENTILE_SELECTION_VERSION",
    "QC_TARGET_REDUCTION_VERSION",
    "CheckpointPin",
    "CheckpointSelectionContract",
    "ClusterFeatureContract",
    "ClusterFeatureContractError",
    "ClusterFeatureContractRequest",
    "ClusterFeatureTarget",
    "FeatureDirectoryIdentity",
    "FeatureStatisticsIdentity",
    "SaeIdentity",
    "StableFileDigest",
    "assert_cluster_checkpoint_pin_unchanged",
    "assert_cluster_contract_loaded_state",
    "assert_loaded_cluster_target",
    "build_cluster_feature_contract",
    "ensure_feature_contract_run_marker",
    "load_feature_contract_sidecar",
    "resolve_checkpoint_pin",
    "resolve_cluster_feature_contract",
    "resolve_cluster_feature_contract_ddp",
    "resolve_eval_checkpoint_directory",
    "resolve_selection_manifest_path",
    "stat_signature_now",
    "stream_hash_file_stable",
    "validate_feature_contract_document",
    "write_feature_contract_sidecar",
)


def test_public_surface_matches_oracle_plus_g2_d2() -> None:
    assert tuple(sorted(contract_module.__all__)) == tuple(sorted(_EXPECTED_ALL))


@pytest.mark.parametrize("name", _EXPECTED_ALL)
def test_every_expected_all_name_imports_from_the_facade(name: str) -> None:
    assert hasattr(contract_module, name), (
        f"{name!r} is in the expected __all__ but does not import from "
        "steering.eval.cluster_feature_contract"
    )


def test_all_exported_names_are_actually_importable_via_star_semantics() -> None:
    """``__all__`` entries must be real attributes, not stale strings."""
    missing = [name for name in contract_module.__all__ if not hasattr(contract_module, name)]
    assert missing == []


# ============================================================================
# Trim-by-import: B3-owned code, ported from the oracle's own test file.
# ============================================================================

_A_SHA = "a" * 64
_B_SHA = "b" * 64
_C_SHA = "c" * 64
_D_SHA = "d" * 64


def _clusters(
    source_path: Path,
    *,
    clusters: tuple[tuple[int, ...], ...] = ((1, 2), (4, 5)),
) -> ResolvedFeatureClusters:
    definition = FeatureClusterDefinition(clusters=clusters)
    return ResolvedFeatureClusters(
        source_path=source_path,
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=8),
    )


def _pin(root: Path, mask: FeatureMask, *, model_sha: str = _C_SHA) -> CheckpointPin:
    digest = StableFileDigest(
        path=root / "model_000007.pt",
        sha256=model_sha,
        byte_length=123,
        stat_signature=(0o100644, 1, 2, 123, 4, 5),
    )
    metadata_digest = StableFileDigest(
        path=root / "meta_000007.json",
        sha256=_D_SHA,
        byte_length=45,
        stat_signature=(0o100644, 1, 3, 45, 4, 5),
    )
    return CheckpointPin(
        checkpoint_dir=root,
        resolved_tag="clustered-d12",
        step=7,
        model_file=digest,
        metadata_file=metadata_digest,
        n_sae_features=8,
        feature_vocab=FeatureVocab.from_mask(mask, 8),
        load_bearing_metadata_json="{}",
    )


def _selection() -> CheckpointSelectionContract:
    return CheckpointSelectionContract(
        positive_selection_json=json.dumps(
            {"distribution": "geometric", "mean": 2.0}, sort_keys=True, separators=(",", ":")
        ),
        negative_selection_json=json.dumps(
            {"distribution": "geometric", "mean": 1.0}, sort_keys=True, separators=(",", ":")
        ),
        num_buckets=2,
        effective_positive_buckets=1,
        preselection_seed=42,
        selection_hash=_A_SHA,
        selection_manifest_sha256=_B_SHA,
        selection_shards_sha256=_C_SHA,
        packed_row_fingerprint=_D_SHA,
    )


def _build(
    tmp_path: Path,
    *,
    resolved_clusters: ResolvedFeatureClusters | None = None,
    checkpoint: CheckpointPin | None = None,
    source_mode: str = "statistics_domain",
    source_domain: tuple[int, ...] = (2, 3, 4, 6),
):
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    return build_cluster_feature_contract(
        feature_mask=mask,
        feature_clusters=resolved_clusters or _clusters(tmp_path / "clusters.yaml"),
        statistics=FeatureStatisticsIdentity(
            file_sha256=_A_SHA,
            percentile_table_checksum=_B_SHA,
            n_sae_features=8,
            feature_id_domain=(2, 3, 4, 6),
        ),
        feature_directory=FeatureDirectoryIdentity(
            segment_size=128,
            layer=8,
            activation_type="residual",
            pooling="max_pool",
            width="16k",
            model="gemma-2-2b",
            dataset_phase="base",
            context_independent_features=False,
            ignore_bos_features=True,
        ),
        sae=SaeIdentity(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_8/width_16k/canonical",
            hook_name="blocks.8.hook_resid_post",
            model="gemma-2-2b",
            layer=8,
            activation_type="residual",
            width="16k",
            actual_loaded_width=8,
        ),
        checkpoint=checkpoint or _pin(tmp_path / "checkpoint", mask),
        selection=_selection(),
        concept_source_mode=source_mode,
        concept_source_domain=source_domain,
    )


def test_contract_exposes_representative_target_and_sibling_projection(
    tmp_path: Path,
) -> None:
    contract = _build(tmp_path)

    assert contract.representative_universe == (1, 3, 4, 6)
    assert contract.source_to_representative == ((2, 1), (3, 3), (4, 4), (6, 6))
    assert contract.consolidation.is_identity_after_mask is False
    assert contract.target_for_representative(1) == ClusterFeatureTarget(
        representative_id=1,
        allowed_member_ids=(1, 2),
        projecting_source_ids=(2,),
        feature_contract_fingerprint=contract.feature_contract_fingerprint,
    )

    with pytest.raises(ClusterFeatureContractError, match="not in the representative"):
        contract.target_for_representative(2)


def test_representative_lookup_rejects_bool_and_float_aliases(
    tmp_path: Path,
) -> None:
    """Bool/float aliases of a valid representative remain invalid inputs."""
    contract = _build(tmp_path)
    assert contract.target_for_representative(1).representative_id == 1

    for alias in (True, 1.0):
        with pytest.raises(ClusterFeatureContractError, match="strict int"):
            contract.target_for_representative(alias)  # type: ignore[arg-type]


def test_target_value_object_is_frozen(tmp_path: Path) -> None:
    target = _build(tmp_path).target_for_representative(1)
    with pytest.raises(FrozenInstanceError):
        target.representative_id = 2  # type: ignore[misc]


def test_fingerprint_excludes_paths_but_full_cluster_content_is_semantic(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path)
    moved = _build(
        tmp_path,
        resolved_clusters=_clusters(tmp_path / "moved" / "partition.yaml"),
        checkpoint=replace(first.checkpoint, checkpoint_dir=tmp_path / "moved-checkpoint"),
    )
    assert first.feature_contract_fingerprint == moved.feature_contract_fingerprint
    assert first.semantic_dict() == moved.semantic_dict()
    assert first.provenance_dict() != moved.provenance_dict()

    # Feature 7 is masked. Adding it to a cluster preserves the allowed member
    # mapping and representative universe but must still invalidate identity.
    changed_content = _build(
        tmp_path,
        resolved_clusters=_clusters(tmp_path / "clusters.yaml", clusters=((1, 2, 7), (4, 5))),
    )
    assert changed_content.representative_universe == first.representative_universe
    assert (
        changed_content.representative_to_allowed_members == first.representative_to_allowed_members
    )
    assert changed_content.feature_contract_fingerprint != first.feature_contract_fingerprint


def test_full_checkpoint_sha_is_semantic(tmp_path: Path) -> None:
    first = _build(tmp_path)
    changed_pin = replace(
        first.checkpoint,
        model_file=replace(first.checkpoint.model_file, sha256="e" * 64),
    )
    changed = _build(tmp_path, checkpoint=changed_pin)
    assert changed.feature_contract_fingerprint != first.feature_contract_fingerprint


def test_explicit_source_projects_and_deduplicates_by_representative(
    tmp_path: Path,
) -> None:
    contract = _build(tmp_path, source_mode="explicit_whitelist", source_domain=(2, 4, 6))
    assert contract.source_to_representative == ((2, 1), (4, 4), (6, 6))
    assert contract.target_for_representative(1).projecting_source_ids == (2,)


@pytest.mark.parametrize(
    ("source_domain", "match"),
    [
        ((1,), "missing_statistics"),
        ((7,), "missing_statistics=.*mask_disallowed"),
    ],
)
def test_explicit_source_rejects_missing_or_masked_ids(
    tmp_path: Path,
    source_domain: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ClusterFeatureContractError, match=match):
        _build(tmp_path, source_mode="explicit_whitelist", source_domain=source_domain)


def test_build_raises_when_statistics_domain_concept_projection_is_empty_after_masking(
    tmp_path: Path,
) -> None:
    """Codex fix batch finding 4 (MEDIUM, also lens-3): direct-builder path.

    Reproduces lens-3's exact config: statistics/concept domain (2, 3, 4, 6)
    against a feature mask whitelisting only ids 500..505 -- entirely
    disjoint from that domain. Mask filtering therefore removes every
    statistics-domain source, so ``source_projection`` would be empty; the
    pure constructor must raise rather than return a "successful" contract
    that can never project any source onto any representative.
    ``n_sae_features=512`` is large enough to hold both the domain and the
    disjoint mask range; the direct-builder path bypasses ``_build()``'s
    small (n_sae_features=8) shared fixture entirely since it can't reach
    id 505.
    """
    n_sae_features = 512
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(500, 506)))
    definition = FeatureClusterDefinition(clusters=((500, 501),))
    resolved_clusters = ResolvedFeatureClusters(
        source_path=tmp_path / "clusters.yaml",
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(
            definition, n_sae_features=n_sae_features
        ),
    )
    checkpoint = CheckpointPin(
        checkpoint_dir=tmp_path / "checkpoint",
        resolved_tag="empty-projection-fixture",
        step=7,
        model_file=StableFileDigest(
            path=tmp_path / "checkpoint" / "model_000007.pt",
            sha256=_C_SHA,
            byte_length=123,
            stat_signature=(0o100644, 1, 2, 123, 4, 5),
        ),
        metadata_file=StableFileDigest(
            path=tmp_path / "checkpoint" / "meta_000007.json",
            sha256=_D_SHA,
            byte_length=45,
            stat_signature=(0o100644, 1, 3, 45, 4, 5),
        ),
        n_sae_features=n_sae_features,
        feature_vocab=FeatureVocab.from_mask(mask, n_sae_features),
        load_bearing_metadata_json="{}",
    )
    with pytest.raises(ClusterFeatureContractError, match="concept source projection is empty"):
        build_cluster_feature_contract(
            feature_mask=mask,
            feature_clusters=resolved_clusters,
            statistics=FeatureStatisticsIdentity(
                file_sha256=_A_SHA,
                percentile_table_checksum=_B_SHA,
                n_sae_features=n_sae_features,
                feature_id_domain=(2, 3, 4, 6),
            ),
            feature_directory=FeatureDirectoryIdentity(
                segment_size=128,
                layer=8,
                activation_type="residual",
                pooling="max_pool",
                width="16k",
                model="gemma-2-2b",
                dataset_phase="base",
                context_independent_features=False,
                ignore_bos_features=True,
            ),
            sae=SaeIdentity(
                release="gemma-scope-2b-pt-res",
                sae_id="layer_8/width_16k/canonical",
                hook_name="blocks.8.hook_resid_post",
                model="gemma-2-2b",
                layer=8,
                activation_type="residual",
                width="16k",
                actual_loaded_width=n_sae_features,
            ),
            checkpoint=checkpoint,
            selection=_selection(),
            concept_source_mode="statistics_domain",
            concept_source_domain=(2, 3, 4, 6),
        )


def test_sidecar_document_detects_semantic_mutation(tmp_path: Path) -> None:
    document = _build(tmp_path).document()
    assert validate_feature_contract_document(document) == document

    document["semantic"]["n_sae_features"] = 99
    with pytest.raises(ClusterFeatureContractError, match="fingerprint"):
        validate_feature_contract_document(document)


def test_run_marker_rejects_torn_output_but_accepts_equivalent_moved_provenance(
    tmp_path: Path,
) -> None:
    contract = _build(tmp_path)
    root = tmp_path / "run"
    marker = ensure_feature_contract_run_marker(root, contract)
    assert marker.is_file()

    moved = _build(
        tmp_path,
        resolved_clusters=_clusters(tmp_path / "moved" / "clusters.yaml"),
        checkpoint=replace(contract.checkpoint, checkpoint_dir=tmp_path / "moved-checkpoint"),
    )
    assert moved.feature_contract_fingerprint == contract.feature_contract_fingerprint
    assert ensure_feature_contract_run_marker(root, moved) == marker

    torn = tmp_path / "torn"
    torn.mkdir()
    (torn / "artifact.bin").write_bytes(b"partial")
    with pytest.raises(ClusterFeatureContractError, match="non-empty"):
        ensure_feature_contract_run_marker(torn, contract)


def test_run_marker_accepts_matching_source_contract_before_first_marker(
    tmp_path: Path,
) -> None:
    contract = _build(tmp_path)
    root = tmp_path / "run"
    (root / "source_contract.json").parent.mkdir(parents=True)
    (root / "source_contract.json").write_text(
        json.dumps(
            {"feature_contract_fingerprint": contract.feature_contract_fingerprint}
        ),
        encoding="utf-8",
    )

    marker = ensure_feature_contract_run_marker(root, contract)

    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["feature_contract_fingerprint"] == (
        contract.feature_contract_fingerprint
    )


def test_run_marker_rejects_mismatched_source_contract_before_first_marker(
    tmp_path: Path,
) -> None:
    contract = _build(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    (root / "source_contract.json").write_text(
        json.dumps({"feature_contract_fingerprint": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ClusterFeatureContractError, match="source contract"):
        ensure_feature_contract_run_marker(root, contract)

    assert not (root / "feature_contract.json").exists()


def test_run_marker_never_overwrites_a_different_cluster_contract(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path)
    changed = _build(
        tmp_path,
        resolved_clusters=_clusters(
            tmp_path / "changed-clusters.yaml", clusters=((1, 2, 3), (4, 5))
        ),
    )
    assert first.feature_contract_fingerprint != changed.feature_contract_fingerprint
    root = tmp_path / "run"
    marker = ensure_feature_contract_run_marker(root, first)
    original = marker.read_bytes()

    with pytest.raises(ClusterFeatureContractError, match="expected fingerprint"):
        ensure_feature_contract_run_marker(root, changed)

    assert marker.read_bytes() == original


def test_feature_contract_sidecar_rejects_mutation_while_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapted from the oracle for B2a's fused-hash rewrite (named deviation,
    B2 plan constraint 1 — see this module's own docstring, "Named deviation
    (B3, mechanical)"). The oracle's ``_load_json_mapping`` called
    ``stream_hash_file_stable`` TWICE (before and after an unrelated
    ``read_bytes()``), so the oracle's own version of this test mutated the
    file on the SECOND call. B2a's fused ``_load_json_mapping`` calls
    ``stream_hash_file_stable`` exactly ONCE and then verifies the one
    subsequent ``read_bytes()`` against that single digest — there is no
    "second hash call" left to intercept. The equivalent attack window is now
    BETWEEN the one hash call and the read: mutate the file as a side effect
    of ``stream_hash_file_stable`` itself, immediately before it returns, so
    the digest it hands back describes the OLD bytes while the subsequent
    read observes the NEW ones.
    """
    contract = _build(tmp_path)
    marker = ensure_feature_contract_run_marker(tmp_path / "run", contract)
    real_hash = contract_module.stream_hash_file_stable

    def mutate_then_hash(path: str | Path):
        digest = real_hash(path)
        marker.write_bytes(marker.read_bytes() + b" ")
        return digest

    monkeypatch.setattr(
        "steering.eval.cluster_checkpoint_pin.stream_hash_file_stable",
        mutate_then_hash,
    )

    with pytest.raises(ClusterFeatureContractError, match="changed while"):
        load_feature_contract_sidecar(marker)


def test_loaded_target_must_match_pinned_width_and_vocab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Named port adaptation (#759): unlike build_cluster_feature_contract
    # (a pure constructor that never calls _assert_checkpoint_agreement),
    # assert_cluster_contract_loaded_state DOES call it. _build()'s default
    # checkpoint (_pin) carries a mask-only FeatureVocab.from_mask, but its
    # default clusters ((1, 2), (4, 5)) genuinely consolidate — a real
    # post-#759 checkpoint for this mask/cluster pair must carry the
    # narrower effective_feature_vocab shape instead.
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))
    probe_clusters = _clusters(tmp_path / "clusters.yaml")
    consolidation = FeatureClusterConsolidation.build(probe_clusters, mask)
    effective_vocab = effective_feature_vocab(consolidation, 8)
    checkpoint = replace(_pin(tmp_path / "checkpoint", mask), feature_vocab=effective_vocab)
    contract = _build(tmp_path, checkpoint=checkpoint)
    metadata = {
        "feature_mask_canonical": contract.feature_mask.canonical_dict(),
        "feature_vocab": contract.checkpoint.feature_vocab.metadata_dict(),
        "feature_clusters_descriptor": contract.feature_clusters.descriptor_dict(),
        "feature_clusters_canonical": contract.feature_clusters.canonical_dict(),
    }

    class FakeModel:
        # Mechanical (Codex fix batch): n_sae_features is a constructor
        # parameter, not hardcoded -- see
        # test_loaded_target_rejects_a_mismatched_original_sae_width below,
        # which needs a WRONG width to exercise the previously-untested
        # "original SAE width does not match" raise in
        # assert_cluster_contract_loaded_state.
        def __init__(
            self,
            vocab: FeatureVocab | None,
            *,
            num_buckets: int = 2,
            n_sae_features: int = 8,
        ) -> None:
            self.encoder = type(
                "Encoder",
                (),
                {
                    "config": type(
                        "Config",
                        (),
                        {"n_sae_features": n_sae_features, "num_buckets": num_buckets},
                    )()
                },
            )()
            self.feature_vocab = vocab
            self.checked = False

        def assert_vocab_consistent(self) -> None:
            self.checked = True

    monkeypatch.setattr(
        contract_module,
        "assert_cluster_checkpoint_pin_unchanged",
        lambda _contract: None,
    )
    model = FakeModel(contract.checkpoint.feature_vocab)
    assert_cluster_contract_loaded_state(contract, model, metadata, surface="test target")
    assert_loaded_cluster_target(contract, model, metadata, surface="test target")
    assert model.checked is True

    wrong = FakeModel(FeatureVocab(8, (1, 2, 3)))
    with pytest.raises(ClusterFeatureContractError, match="in-memory FeatureVocab"):
        assert_loaded_cluster_target(contract, wrong, metadata, surface="test target")

    wrong_buckets = FakeModel(contract.checkpoint.feature_vocab, num_buckets=3)
    with pytest.raises(ClusterFeatureContractError, match="bucket count"):
        assert_loaded_cluster_target(contract, wrong_buckets, metadata, surface="test target")

    # Mechanical (Codex fix batch): previously untested raise --
    # assert_cluster_contract_loaded_state's "original SAE width does not
    # match the feature contract" guard (contract.n_sae_features == 8;
    # every other FakeModel in this test carried the matching width, so
    # this specific raise had no direct regression coverage).
    wrong_width = FakeModel(contract.checkpoint.feature_vocab, n_sae_features=7)
    with pytest.raises(ClusterFeatureContractError, match="original SAE width"):
        assert_loaded_cluster_target(contract, wrong_width, metadata, surface="test target")


# ----------------------------------------------------------------------------
# Full-filesystem preflight (cross-layer integration entry point — belongs to
# B3 since resolve_cluster_feature_contract is this module's own function).
# ----------------------------------------------------------------------------


def _clustered_validation_certificate(
    *,
    packed_dir: Path,
    selection_hash: str,
    selection_integrity: dict[str, Any],
    resolved_clusters: ResolvedFeatureClusters,
    segment_count: int,
    max_features_per_segment: int,
) -> dict[str, Any]:
    proof = resolve_packed_row_fingerprint_strict(packed_dir)
    return build_clustered_selection_validation(
        selection_checksum=selection_hash,
        selection_integrity=selection_integrity,
        packed_row_fingerprint=proof.aggregate_sha256,
        packed_content_by_name={
            entry.name.removesuffix(".safetensors.zst"): entry.content_sha256
            for entry in proof.files
        },
        feature_clusters_descriptor=resolved_clusters.descriptor_dict(),
        feature_delta=None,
        effective_buckets=1,
        segment_counts_by_stem={"shard_00000": segment_count},
        max_features_by_stem={
            "shard_00000": max_features_per_segment,
        },
    )


def _write_full_preflight_fixture(tmp_path: Path) -> tuple[ClusterFeatureContractRequest, Any]:
    """Build every file-backed producer artifact consumed by cluster preflight."""

    n_sae_features = 16_384
    mask_ids = tuple(range(6))
    mask = FeatureMask(FeatureMaskMode.whitelist, mask_ids)
    mask_path = tmp_path / "accepted_feature_ids.txt"
    mask_path.write_text("".join(f"{feature_id}\n" for feature_id in mask_ids), encoding="utf-8")

    cluster_path = tmp_path / "feature_clusters.json"
    definition = FeatureClusterDefinition(clusters=((0, 1), (2, 3)))
    cluster_path.write_text(json.dumps(definition.canonical_dict(), sort_keys=True), encoding="utf-8")
    resolved_clusters = ResolvedFeatureClusters(
        source_path=cluster_path.resolve(),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=n_sae_features),
    )
    consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)

    feature_dir = build_feature_dir(
        tmp_path / "features",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )
    feature_dir.mkdir(parents=True)
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags={feature_id: float(feature_id + 1) for feature_id in mask_ids},
        absent=range(len(mask_ids), n_sae_features),
    )
    stats = load_statistics(statistics_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, percentile_checksum)

    packed_dir = feature_dir / "packed_fixture"
    packed_dir.mkdir()
    packed_path = save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[1, 2, 3, 4, 5]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([4], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 4], dtype=t.int32),
            feature_ids=t.tensor([0, 1, 2, 4], dtype=t.uint16),
            feature_pcts=t.tensor([220, 180, 200, 90], dtype=t.uint8),
        ),
        packed_dir / "shard_00000",
    )
    write_manifest(
        packed_dir,
        "packed-fixture",
        OverlayKind.PACKED_ROW,
        metadata={"pct_checksum": percentile_checksum},
        base_shard_paths=[packed_path],
    )

    positive_selection = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    negative_selection = {"distribution": "geometric", "mean": 1.0, "std": 1.0}
    selection_hash = compute_selection_hash(
        positive_selection=positive_selection,
        negative_selection=negative_selection,
        delta_config=None,
        num_buckets=2,
        seed=42,
        ddp_rank=0,
        feature_mask=mask.canonical_dict(),
        feature_clusters=resolved_clusters.descriptor_dict(),
    )
    selection_dir = packed_dir / "sel" / selection_hash
    selection_dir.mkdir(parents=True)
    selection_path = save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 2], dtype=t.int32),
            sel_fids=t.tensor([0, 2], dtype=t.uint16),
            sel_buckets=t.tensor([0, 0], dtype=t.uint8),
            sel_pcts=t.tensor([220, 200], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            neg_fids=t.tensor([4], dtype=t.uint16),
        ),
        selection_dir / "shard_00000",
    )
    cluster_metadata = consolidation.metadata_dict()
    cluster_metadata["status"] = "complete"
    selection_integrity = build_clustered_selection_shard_integrity([selection_path])
    validation = _clustered_validation_certificate(
        packed_dir=packed_dir,
        selection_hash=selection_hash,
        selection_integrity=selection_integrity,
        resolved_clusters=resolved_clusters,
        segment_count=1,
        max_features_per_segment=3,
    )
    write_manifest(
        selection_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata={
            "feature_mask": mask.canonical_dict(),
            "feature_clusters": cluster_metadata,
            "packed_rows": {"packed_row_fingerprint": packed_row_fingerprint(packed_dir)},
            "selection_shards": selection_integrity,
            CLUSTERED_SELECTION_VALIDATION_KEY: validation,
        },
        base_shard_paths=[packed_path],
        cache_dir=packed_dir,
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    # The checkpoint must carry the effective vocabulary after consolidation:
    # features 1 and 3 are folded into representatives 0 and 2.
    feature_vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert feature_vocab is not None
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((feature_vocab.size, 2)),
            VOCAB_IDS_KEY: t.tensor(feature_vocab.original_ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
        },
        checkpoint_dir / "model_000007.pt",
    )
    checkpoint_metadata = {
        "model_config": {"n_layer": 2, "n_embd": 8},
        "encoder_config": {"n_embd": 8, "depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_mask_canonical": mask.canonical_dict(),
        "feature_vocab": feature_vocab.metadata_dict(),
        "feature_clusters_descriptor": resolved_clusters.descriptor_dict(),
        "feature_clusters_canonical": resolved_clusters.canonical_dict(),
        "user_config": {
            "feature_selection": positive_selection,
            "negative_selection": negative_selection,
            "num_buckets": 2,
            "seed": 42,
            "feature_delta": None,
            "feature_prediction": None,
            "feature_dir": str(feature_dir),
            "feature_clusters": {"path": str(cluster_path)},
            "sae": {
                "release": SAERelease.gemmascope.value,
                "model": "gemma-2-2b",
                "activation_type": ActivationType.residual.value,
                "layer": 8,
                "width": SAEWidth.W_16K.value,
                "l0": None,
            },
        },
    }
    (checkpoint_dir / "meta_000007.json").write_text(
        json.dumps(checkpoint_metadata, sort_keys=True), encoding="utf-8"
    )

    resolved_sae = resolve_sae(
        model="gemma-2-2b",
        release=SAERelease.gemmascope,
        activation_type=ActivationType.residual,
        layer=8,
        width=SAEWidth.W_16K,
        l0=None,
    )
    fake_sae = SimpleNamespace(
        W_dec=t.zeros((n_sae_features, 2)),
        cfg=SimpleNamespace(d_in=2, metadata=SimpleNamespace(hook_name=resolved_sae.hook_name)),
    )
    request = ClusterFeatureContractRequest(
        feature_dir=feature_dir,
        feature_statistics_path=statistics_path,
        feature_mask={"mode": "whitelist", "path": str(mask_path)},
        feature_clusters={"path": str(cluster_path)},
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=7,
        checkpoint_tag="fixture-step-7",
    )
    return request, fake_sae


def _cluster_preflight_gloo_worker(
    rank: int,
    world_size: int,
    request: ClusterFeatureContractRequest,
    force: bool,
    fail_peer: bool,
    disagree_peer: bool = False,
    return_none_peer: bool = False,
    fail_rank_zero: bool = False,
    peer_cache_miss: bool = False,
) -> dict[str, Any]:
    del world_size
    counts = {
        "packed_hashes": 0,
        "packed_loads": 0,
        "selection_loads": 0,
        "contract_resolutions": 0,
        "cache_writes": 0,
    }
    original_file_fingerprint = (
        splice_counterfactual_module._file_fingerprint
    )
    original_validating_fingerprint = (
        splice_counterfactual_module._file_fingerprint_with_body_validator
    )
    original_packed_loader = checkpoint_pin_module.load_packed_row_shard_bytes
    original_selection_loader = checkpoint_pin_module.load_selection_shard
    original_contract_resolver = contract_module.resolve_cluster_feature_contract
    original_cache_loader = splice_counterfactual_module._load_file_fingerprint_cache
    original_cache_writer = splice_counterfactual_module.atomic_json_dump

    def _count_fingerprint(*args: Any, **kwargs: Any) -> Any:
        counts["packed_hashes"] += 1
        return original_file_fingerprint(*args, **kwargs)

    def _count_validating_fingerprint(*args: Any, **kwargs: Any) -> Any:
        counts["packed_hashes"] += 1
        return original_validating_fingerprint(*args, **kwargs)

    def _count_packed_load(*args: Any, **kwargs: Any) -> Any:
        counts["packed_loads"] += 1
        return original_packed_loader(*args, **kwargs)

    def _count_selection_load(*args: Any, **kwargs: Any) -> Any:
        counts["selection_loads"] += 1
        return original_selection_loader(*args, **kwargs)

    def _count_contract_resolution(*args: Any, **kwargs: Any) -> Any:
        counts["contract_resolutions"] += 1
        return original_contract_resolver(*args, **kwargs)

    def _peer_missing_cache(*args: Any, **kwargs: Any) -> Any:
        if rank == 1 and kwargs.get("namespace") == "packed_rows":
            return None
        return original_cache_loader(*args, **kwargs)

    def _count_cache_write(*args: Any, **kwargs: Any) -> Any:
        counts["cache_writes"] += 1
        if rank == 1 and peer_cache_miss:
            raise AssertionError("peer attempted to write the shared packed cache")
        return original_cache_writer(*args, **kwargs)

    splice_counterfactual_module.__dict__["_file_fingerprint"] = (
        _count_fingerprint
    )
    splice_counterfactual_module.__dict__[
        "_file_fingerprint_with_body_validator"
    ] = _count_validating_fingerprint
    checkpoint_pin_module.__dict__["load_packed_row_shard_bytes"] = (
        _count_packed_load
    )
    checkpoint_pin_module.__dict__["load_selection_shard"] = (
        _count_selection_load
    )
    contract_module.__dict__["resolve_cluster_feature_contract"] = (
        _count_contract_resolution
    )
    if peer_cache_miss:
        splice_counterfactual_module.__dict__[
            "_load_file_fingerprint_cache"
        ] = _peer_missing_cache
        splice_counterfactual_module.__dict__["atomic_json_dump"] = (
            _count_cache_write
        )
    if disagree_peer and rank == 1:
        counted_contract_resolver = contract_module.resolve_cluster_feature_contract

        def _divergent_contract(*args: Any, **kwargs: Any) -> Any:
            contract = counted_contract_resolver(*args, **kwargs)
            semantic = contract.semantic_dict()
            semantic["ddp_test_divergence"] = True
            semantic_json = contract_module._canonical_json(semantic)
            return replace(
                contract,
                semantic_json=semantic_json,
                feature_contract_fingerprint=(
                    contract_module._sha256_json(semantic)
                ),
            )

        contract_module.__dict__["resolve_cluster_feature_contract"] = (
            _divergent_contract
        )
    elif return_none_peer and rank == 1:
        contract_module.__dict__["resolve_cluster_feature_contract"] = (
            lambda *_args, **_kwargs: None
        )
    local_request = replace(
        request,
        force_cluster_shard_revalidation=force,
        feature_statistics_path=(
            Path(request.feature_statistics_path).with_name("missing.safetensors")
            if (fail_peer and rank == 1) or (fail_rank_zero and rank == 0)
            else request.feature_statistics_path
        ),
    )

    def _sae_loader(resolved: Any) -> Any:
        return SimpleNamespace(
            W_dec=t.zeros((16_384, 2)),
            cfg=SimpleNamespace(
                d_in=2,
                metadata=SimpleNamespace(hook_name=resolved.hook_name),
            ),
        )

    try:
        contract = contract_module.resolve_cluster_feature_contract_ddp(
            local_request,
            sae_loader=_sae_loader,
        )
    except ClusterFeatureContractError as exc:
        return {
            "rank": rank,
            "error": str(exc),
            **counts,
        }
    return {
        "rank": rank,
        "error": None,
        "fingerprint": contract.feature_contract_fingerprint,
        **counts,
    }


def test_full_filesystem_preflight_freezes_every_producer_identity(tmp_path: Path) -> None:
    request, fake_sae = _write_full_preflight_fixture(tmp_path)

    contract = resolve_cluster_feature_contract(request, sae_loader=lambda _resolved: fake_sae)

    assert contract.n_sae_features == 16_384
    assert contract.checkpoint.step == 7
    assert contract.checkpoint.resolved_tag == "fixture-step-7"
    # Post-#759: the persistent vocab is the NARROWER, cluster-consolidated
    # shape (representatives 0, 2 + non-clustered 4, 5), not the mask-only
    # (0..5) shape the oracle's fixture used — see _write_full_preflight_
    # fixture's own "Named port adaptation (#759)" comment.
    assert contract.checkpoint.feature_vocab == FeatureVocab(16_384, (0, 2, 4, 5))
    assert contract.representative_universe == (0, 2, 4, 5)
    assert contract.source_to_representative == ((0, 0), (1, 0), (2, 2), (3, 2), (4, 4), (5, 5))
    assert contract.selection.preselection_seed == 42
    assert contract.selection.num_buckets == 2
    assert contract.selection.effective_positive_buckets == 1
    assert validate_feature_contract_document(contract.document()) == contract.document()

    # Source and target independently rebuilding the same semantic contract is
    # the production BPB three-way comparison, not an in-memory shortcut.
    source_bound = resolve_cluster_feature_contract(
        replace(request, expected_feature_contract_document=contract.document()),
        sae_loader=lambda _resolved: fake_sae,
    )
    assert source_bound.feature_contract_fingerprint == contract.feature_contract_fingerprint

    # Moving equivalent config files is provenance-only and must preserve the
    # semantic fingerprint.
    moved_dir = tmp_path / "moved_config"
    moved_dir.mkdir()
    moved_mask = moved_dir / "mask.txt"
    moved_mask.write_bytes(Path(request.feature_mask["path"]).read_bytes())
    moved_clusters = moved_dir / "clusters.json"
    moved_clusters.write_bytes(Path(request.feature_clusters["path"]).read_bytes())
    moved = resolve_cluster_feature_contract(
        replace(
            request,
            feature_mask={"mode": "whitelist", "path": str(moved_mask)},
            feature_clusters={"path": str(moved_clusters)},
        ),
        sae_loader=lambda _resolved: fake_sae,
    )
    assert moved.feature_contract_fingerprint == contract.feature_contract_fingerprint
    assert moved.provenance_dict() != contract.provenance_dict()

    # Adding only a masked-away member preserves V and representatives, but
    # still changes cluster content and must fail against the checkpoint.
    changed_clusters = moved_dir / "changed_clusters.json"
    changed_clusters.write_text(
        json.dumps({"schema_version": 1, "clusters": [[0, 1, 6], [2, 3]]}), encoding="utf-8"
    )
    with pytest.raises(ClusterFeatureContractError, match="feature cluster mismatch"):
        resolve_cluster_feature_contract(
            replace(request, feature_clusters={"path": str(changed_clusters)}),
            sae_loader=lambda _resolved: fake_sae,
        )


def test_certified_and_forced_full_contract_documents_are_identical(
    tmp_path: Path,
) -> None:
    request, fake_sae = _write_full_preflight_fixture(tmp_path)
    fast = resolve_cluster_feature_contract(
        request,
        sae_loader=lambda _resolved: fake_sae,
    )
    forced = resolve_cluster_feature_contract(
        replace(request, force_cluster_shard_revalidation=True),
        sae_loader=lambda _resolved: fake_sae,
    )
    assert forced.selection == fast.selection
    assert forced.document() == fast.document()
    assert forced.semantic_json == fast.semantic_json
    assert forced.provenance_json == fast.provenance_json
    assert (
        forced.feature_contract_fingerprint
        == fast.feature_contract_fingerprint
    )


def test_one_certified_artifact_resolves_identically_through_every_eval_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_request, fake_sae = _write_full_preflight_fixture(tmp_path)
    direct = resolve_cluster_feature_contract(
        direct_request,
        sae_loader=lambda _resolved: fake_sae,
    )
    assert direct_request.checkpoint_dir is not None
    checkpoint_dir = Path(direct_request.checkpoint_dir)
    source_run_dir = tmp_path / "source-run"
    ensure_feature_contract_run_marker(source_run_dir, direct)

    monkeypatch.setattr(
        checkpoint_pin_module,
        "get_checkpoint_dir",
        lambda _source, *, model_tag: (
            checkpoint_dir
            if model_tag == "fixture-step-7"
            else pytest.fail(f"unexpected checkpoint tag {model_tag!r}")
        ),
    )
    monkeypatch.setattr(
        contract_module,
        "_default_sae_loader",
        lambda _resolved: fake_sae,
    )

    axbench_cfg = AxBenchBaseConfig(
        feature_dir=str(direct_request.feature_dir),
        feature_statistics_path=str(direct_request.feature_statistics_path),
        decoder_only={},
        encdec={
            "source": "fixture-source",
            "model_tag": "fixture-step-7",
            "step": 7,
        },
        device="cpu",
        output_root=str(tmp_path / "axbench-output"),
        conditioning={
            "feature_mask": dict(direct_request.feature_mask),
            "feature_clusters": dict(direct_request.feature_clusters),
        },
    )
    axbench_request = axbench_base_cli._cluster_contract_request(axbench_cfg)
    assert axbench_request is not None
    through_axbench = contract_module.resolve_cluster_feature_contract_ddp(
        axbench_request
    )

    bpb_cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_run_dir),
        output_root=str(tmp_path / "bpb-output"),
        run_tag="shared-certified-artifact",
        feature_dir=str(direct_request.feature_dir),
        feature_statistics_path=str(direct_request.feature_statistics_path),
        feature_mask=dict(direct_request.feature_mask),
        feature_clusters=dict(direct_request.feature_clusters),
        force_cluster_shard_revalidation=True,
        encdec_target={
            "source": "fixture-source",
            "model_tag": "fixture-step-7",
            "step": 7,
        },
    )
    bpb_request = build_cluster_contract_request(bpb_cfg)
    assert bpb_request is not None
    through_bpb_forced = resolve_cluster_feature_contract(bpb_request)

    assert bpb_cfg.run_tag is not None
    paths = AxBenchBpbAdvPaths(
        output_root=bpb_cfg.output_root,
        run_tag=bpb_cfg.run_tag,
    )
    ensure_feature_contract_run_marker(paths.root, direct)
    paths.candidates_dir().mkdir(parents=True)
    paths.candidates_path().touch()
    source_contract_payload = {"source": "same-certified-artifact"}
    paths.source_contract().write_text(
        json.dumps(source_contract_payload),
        encoding="utf-8",
    )
    source_contract = SimpleNamespace(
        feature_contract_fingerprint=direct.feature_contract_fingerprint,
        to_json_dict=lambda: source_contract_payload,
    )
    monkeypatch.setattr(
        sample_generation,
        "validate_stage_0_3_source_run",
        lambda *_args, **_kwargs: SimpleNamespace(contract=source_contract),
    )
    retained_stage_source = object()
    monkeypatch.setattr(
        sample_generation,
        "resolve_clustered_stage_source",
        lambda **_kwargs: retained_stage_source,
    )
    bpb_cfg.force_cluster_shard_revalidation = False
    (
        through_sample_generation,
        through_sample_stage_source,
    ) = sample_generation._resolve_sample_cluster_stage(
        bpb_cfg,
        paths=paths,
    )

    assert through_axbench.document() == direct.document()
    assert through_bpb_forced.document() == direct.document()
    assert through_sample_generation is not None
    assert through_sample_generation.document() == direct.document()
    assert through_sample_stage_source is retained_stage_source


def test_ddp_peer_none_result_is_converted_to_collective_failure_before_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ClusterFeatureContractRequest(
        feature_dir="unused",
        feature_statistics_path="unused",
        feature_mask={},
        feature_clusters={},
    )
    canonical = {"error": None, "fingerprint": "f" * 64}

    monkeypatch.setattr(contract_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(contract_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(contract_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(contract_module.dist, "get_world_size", lambda: 2)

    def broadcast(payload: list[dict[str, str | None]], *, src: int) -> None:
        assert src == 0
        payload[0] = canonical

    def gather(
        gathered: list[dict[str, Any] | None],
        local_status: dict[str, Any],
    ) -> None:
        gathered[0] = {"rank": 0, **canonical}
        gathered[1] = local_status

    monkeypatch.setattr(contract_module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(contract_module.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        contract_module,
        "resolve_cluster_feature_contract",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        ClusterFeatureContractError,
        match="without a local contract",
    ):
        contract_module.resolve_cluster_feature_contract_ddp(request)


def test_ddp_rank_zero_success_returns_local_contract_after_full_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ClusterFeatureContractRequest(
        feature_dir="unused",
        feature_statistics_path="unused",
        feature_mask={},
        feature_clusters={},
    )
    contract = SimpleNamespace(feature_contract_fingerprint="f" * 64)

    monkeypatch.setattr(contract_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(contract_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(contract_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(contract_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        contract_module,
        "resolve_cluster_feature_contract",
        lambda *_args, **_kwargs: contract,
    )

    def broadcast(payload: list[dict[str, str | None]], *, src: int) -> None:
        assert src == 0
        assert payload == [{"error": None, "fingerprint": "f" * 64}]

    def gather(
        gathered: list[dict[str, Any] | None],
        local_status: dict[str, Any],
    ) -> None:
        assert local_status == {
            "rank": 0,
            "error": None,
            "fingerprint": "f" * 64,
        }
        gathered[0] = local_status
        gathered[1] = {**local_status, "rank": 1}

    monkeypatch.setattr(contract_module.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(contract_module.dist, "all_gather_object", gather)

    assert contract_module.resolve_cluster_feature_contract_ddp(request) is contract


def test_two_rank_gloo_cold_and_forced_preflight_have_one_authoritative_reader(
    tmp_path: Path,
) -> None:
    request, _fake_sae = _write_full_preflight_fixture(tmp_path)
    cache_path = (
        Path(request.feature_dir)
        / "packed_fixture"
        / ".packed-row-fingerprint-cache.json"
    )
    cache_path.unlink()

    cold = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        False,
        False,
    )
    assert [item["error"] for item in cold] == [None, None]
    assert cold[0]["fingerprint"] == cold[1]["fingerprint"]
    assert cold[0]["packed_hashes"] == 1
    assert cold[1]["packed_hashes"] == 0
    assert all(item["packed_loads"] == 0 for item in cold)
    assert all(item["selection_loads"] == 0 for item in cold)
    assert cache_path.is_file()

    forced = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        True,
        False,
    )
    assert [item["error"] for item in forced] == [None, None]
    assert forced[0]["fingerprint"] == forced[1]["fingerprint"]
    assert forced[0]["fingerprint"] == cold[0]["fingerprint"]
    assert forced[0]["packed_hashes"] == 1
    assert forced[1]["packed_hashes"] == 0
    assert forced[0]["packed_loads"] == 1
    assert forced[0]["selection_loads"] == 1
    assert forced[1]["packed_loads"] == 0
    assert forced[1]["selection_loads"] == 0


def test_two_rank_gloo_peer_local_failure_is_raised_collectively(
    tmp_path: Path,
) -> None:
    request, _fake_sae = _write_full_preflight_fixture(tmp_path)

    results = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        False,
        True,
    )

    assert all(item["error"] is not None for item in results)
    assert all("rank" in item["error"] for item in results)
    assert all("missing.safetensors" in item["error"] for item in results)


def test_two_rank_gloo_rank_zero_failure_is_canonical_without_peer_resolution(
    tmp_path: Path,
) -> None:
    request, _fake_sae = _write_full_preflight_fixture(tmp_path)

    results = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        False,
        False,
        False,
        False,
        True,
    )

    assert all(item["error"] is not None for item in results)
    assert all("missing.safetensors" in item["error"] for item in results)
    assert all("without a local contract" not in item["error"] for item in results)
    assert results[0]["contract_resolutions"] == 1
    assert results[1]["contract_resolutions"] == 0


def test_two_rank_gloo_peer_missing_cache_hashes_read_only_without_writing(
    tmp_path: Path,
) -> None:
    request, _fake_sae = _write_full_preflight_fixture(tmp_path)
    cache_path = (
        Path(request.feature_dir)
        / "packed_fixture"
        / ".packed-row-fingerprint-cache.json"
    )
    before_cache = cache_path.read_bytes()

    results = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        False,
        False,
        False,
        False,
        False,
        True,
    )

    assert [item["error"] for item in results] == [None, None]
    assert results[0]["fingerprint"] == results[1]["fingerprint"]
    assert results[0]["packed_hashes"] == 0
    assert results[1]["packed_hashes"] == 1
    assert results[1]["cache_writes"] == 0
    assert cache_path.read_bytes() == before_cache


def test_two_rank_gloo_fingerprint_disagreement_is_raised_collectively(
    tmp_path: Path,
) -> None:
    request, _fake_sae = _write_full_preflight_fixture(tmp_path)

    results = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        False,
        False,
        True,
    )

    assert all(item["error"] is not None for item in results)
    assert all("reconstructed different" in item["error"] for item in results)


def test_two_rank_gloo_missing_peer_contract_is_raised_collectively(
    tmp_path: Path,
) -> None:
    request, _fake_sae = _write_full_preflight_fixture(tmp_path)

    results = run_gloo_multiprocess(
        2,
        _cluster_preflight_gloo_worker,
        request,
        False,
        False,
        False,
        True,
    )

    assert all(item["error"] is not None for item in results)
    assert all("without a local contract" in item["error"] for item in results)


def test_loaded_target_rehash_detects_checkpoint_mutation_before_first_forward(
    tmp_path: Path,
) -> None:
    request, fake_sae = _write_full_preflight_fixture(tmp_path)
    contract = resolve_cluster_feature_contract(request, sae_loader=lambda _resolved: fake_sae)
    metadata = json.loads(
        (contract.checkpoint.checkpoint_dir / "meta_000007.json").read_text(encoding="utf-8")
    )

    class _LoadedModel:
        encoder = SimpleNamespace(
            config=SimpleNamespace(
                n_sae_features=contract.n_sae_features,
                num_buckets=contract.selection.num_buckets,
            )
        )
        feature_vocab = contract.checkpoint.feature_vocab

        @staticmethod
        def assert_vocab_consistent() -> None:
            return None

    model = _LoadedModel()
    assert_loaded_cluster_target(contract, model, metadata, surface="fixture target")

    checkpoint_path = contract.checkpoint.model_file.path
    state = t.load(checkpoint_path, map_location="cpu", weights_only=True)
    state[FEATURE_EMBED_KEY][0, 0] = 1.0
    t.save(state, checkpoint_path)

    with pytest.raises(ClusterFeatureContractError, match="changed after preflight"):
        assert_loaded_cluster_target(contract, model, metadata, surface="fixture target")


# ----------------------------------------------------------------------------
# Codex fix batch finding 4 (MEDIUM, also lens-3): empty concept-source
# projection after mask filtering. Covers the resolver and expected-document
# paths (the direct-builder path is
# test_build_raises_when_statistics_domain_concept_projection_is_empty_
# after_masking above). Both paths funnel through the same
# resolve_cluster_feature_contract -> _concept_source_identity ->
# build_cluster_feature_contract chain, so both need a full filesystem
# fixture -- lens-3's disjoint-domain-vs-mask config (concept domain
# (2, 3, 4, 6) against a feature mask entirely outside it -- see
# _write_empty_projection_fixture's own docstring for why the mask range is
# 16000..16005 rather than lens-3's literal 500..505) reproduced via a
# dedicated fixture rather than _write_full_preflight_fixture's
# mask-equals-stats-domain shape, which can never be disjoint by
# construction.
# ----------------------------------------------------------------------------


def _write_empty_projection_fixture(tmp_path: Path) -> tuple[ClusterFeatureContractRequest, Any]:
    """Full filesystem preflight fixture with a feature mask entirely
    disjoint from the statistics/concept domain (2, 3, 4, 6).

    ``n_sae_features`` must stay 16_384 (matching ``SAEWidth.W_16K`` --
    ``_statistics_identity`` cross-checks the feature directory's width
    LABEL against the checkpoint's actual ``n_sae_features``, and 16_384 is
    the only width this repo's ``SAEWidth`` enum defines), so the disjoint
    mask range is pushed up near the top of that width (16000..16005)
    rather than lens-3's literal 500..505 -- same disjoint-from-(2,3,4,6)
    property, compatible with the resolver's width plumbing.
    """

    n_sae_features = 16_384
    mask_ids = tuple(range(16000, 16006))
    mask = FeatureMask(FeatureMaskMode.whitelist, mask_ids)
    mask_path = tmp_path / "accepted_feature_ids.txt"
    mask_path.write_text("".join(f"{feature_id}\n" for feature_id in mask_ids), encoding="utf-8")

    cluster_path = tmp_path / "feature_clusters.json"
    definition = FeatureClusterDefinition(clusters=((16000, 16001),))
    cluster_path.write_text(json.dumps(definition.canonical_dict(), sort_keys=True), encoding="utf-8")
    resolved_clusters = ResolvedFeatureClusters(
        source_path=cluster_path.resolve(),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=n_sae_features),
    )
    consolidation = FeatureClusterConsolidation.build(resolved_clusters, mask)

    feature_dir = build_feature_dir(
        tmp_path / "features",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )
    feature_dir.mkdir(parents=True)
    concept_domain = (2, 3, 4, 6)
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags={feature_id: float(feature_id + 1) for feature_id in concept_domain},
        absent=[feature_id for feature_id in range(n_sae_features) if feature_id not in concept_domain],
    )
    stats = load_statistics(statistics_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, percentile_checksum)

    packed_dir = feature_dir / "packed_fixture"
    packed_dir.mkdir()
    packed_path = save_packed_row_shard(
        PackedRowShard(
            tokens=t.tensor([[1, 2, 3, 4, 5]], dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([4], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, 2], dtype=t.int32),
            # 16002/16003 are mask-allowed but outside the (16000, 16001)
            # cluster entirely -- selection ids must never name a
            # consolidated-away id (16001), which the negative-selection
            # integrity check separately rejects.
            feature_ids=t.tensor([16002, 16003], dtype=t.uint16),
            feature_pcts=t.tensor([220, 180], dtype=t.uint8),
        ),
        packed_dir / "shard_00000",
    )
    write_manifest(
        packed_dir,
        "packed-fixture",
        OverlayKind.PACKED_ROW,
        metadata={"pct_checksum": percentile_checksum},
        base_shard_paths=[packed_path],
    )

    positive_selection = {"distribution": "geometric", "mean": 2.0, "std": 2.0}
    negative_selection = {"distribution": "geometric", "mean": 1.0, "std": 1.0}
    selection_hash = compute_selection_hash(
        positive_selection=positive_selection,
        negative_selection=negative_selection,
        delta_config=None,
        num_buckets=2,
        seed=42,
        ddp_rank=0,
        feature_mask=mask.canonical_dict(),
        feature_clusters=resolved_clusters.descriptor_dict(),
    )
    selection_dir = packed_dir / "sel" / selection_hash
    selection_dir.mkdir(parents=True)
    selection_path = save_selection_shard(
        SelectionShard(
            sel_offsets=t.tensor([0, 1], dtype=t.int32),
            sel_fids=t.tensor([16002], dtype=t.uint16),
            sel_buckets=t.tensor([0], dtype=t.uint8),
            sel_pcts=t.tensor([220], dtype=t.uint8),
            neg_offsets=t.tensor([0, 1], dtype=t.int32),
            neg_fids=t.tensor([16003], dtype=t.uint16),
        ),
        selection_dir / "shard_00000",
    )
    cluster_metadata = consolidation.metadata_dict()
    cluster_metadata["status"] = "complete"
    selection_integrity = build_clustered_selection_shard_integrity([selection_path])
    validation = _clustered_validation_certificate(
        packed_dir=packed_dir,
        selection_hash=selection_hash,
        selection_integrity=selection_integrity,
        resolved_clusters=resolved_clusters,
        segment_count=1,
        max_features_per_segment=2,
    )
    write_manifest(
        selection_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata={
            "feature_mask": mask.canonical_dict(),
            "feature_clusters": cluster_metadata,
            "packed_rows": {"packed_row_fingerprint": packed_row_fingerprint(packed_dir)},
            "selection_shards": selection_integrity,
            CLUSTERED_SELECTION_VALIDATION_KEY: validation,
        },
        base_shard_paths=[packed_path],
        cache_dir=packed_dir,
    )

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    feature_vocab = effective_feature_vocab(consolidation, n_sae_features)
    assert feature_vocab is not None
    t.save(
        {
            FEATURE_EMBED_KEY: t.zeros((feature_vocab.size, 2)),
            VOCAB_IDS_KEY: t.tensor(feature_vocab.original_ids, dtype=t.int64),
            VOCAB_N_SAE_KEY: t.tensor(n_sae_features, dtype=t.int64),
        },
        checkpoint_dir / "model_000007.pt",
    )
    checkpoint_metadata = {
        "model_config": {"n_layer": 2, "n_embd": 8},
        "encoder_config": {"n_embd": 8, "depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_mask_canonical": mask.canonical_dict(),
        "feature_vocab": feature_vocab.metadata_dict(),
        "feature_clusters_descriptor": resolved_clusters.descriptor_dict(),
        "feature_clusters_canonical": resolved_clusters.canonical_dict(),
        "user_config": {
            "feature_selection": positive_selection,
            "negative_selection": negative_selection,
            "num_buckets": 2,
            "seed": 42,
            "feature_delta": None,
            "feature_prediction": None,
            "feature_dir": str(feature_dir),
            "feature_clusters": {"path": str(cluster_path)},
            "sae": {
                "release": SAERelease.gemmascope.value,
                "model": "gemma-2-2b",
                "activation_type": ActivationType.residual.value,
                "layer": 8,
                "width": SAEWidth.W_16K.value,
                "l0": None,
            },
        },
    }
    (checkpoint_dir / "meta_000007.json").write_text(
        json.dumps(checkpoint_metadata, sort_keys=True), encoding="utf-8"
    )

    resolved_sae = resolve_sae(
        model="gemma-2-2b",
        release=SAERelease.gemmascope,
        activation_type=ActivationType.residual,
        layer=8,
        width=SAEWidth.W_16K,
        l0=None,
    )
    fake_sae = SimpleNamespace(
        W_dec=t.zeros((n_sae_features, 2)),
        cfg=SimpleNamespace(d_in=2, metadata=SimpleNamespace(hook_name=resolved_sae.hook_name)),
    )
    request = ClusterFeatureContractRequest(
        feature_dir=feature_dir,
        feature_statistics_path=statistics_path,
        feature_mask={"mode": "whitelist", "path": str(mask_path)},
        feature_clusters={"path": str(cluster_path)},
        checkpoint_dir=checkpoint_dir,
        checkpoint_step=7,
        checkpoint_tag="empty-projection-fixture",
    )
    return request, fake_sae


def test_full_filesystem_preflight_raises_when_concept_projection_is_empty_after_masking(
    tmp_path: Path,
) -> None:
    """Codex fix batch finding 4 (MEDIUM, also lens-3): resolver path.

    Same lens-3 config as the direct-builder regression above, driven
    through the full ``resolve_cluster_feature_contract`` preflight with the
    default concept source (no whitelist path, no expected document ->
    ``_concept_source_identity`` falls back to ``statistics_domain`` mode
    over the statistics file's own domain (2, 3, 4, 6))."""
    request, fake_sae = _write_empty_projection_fixture(tmp_path)
    with pytest.raises(ClusterFeatureContractError, match="concept source projection is empty"):
        resolve_cluster_feature_contract(request, sae_loader=lambda _resolved: fake_sae)


def test_expected_document_path_raises_when_concept_projection_is_empty_after_masking(
    tmp_path: Path,
) -> None:
    """Codex fix batch finding 4 (MEDIUM, also lens-3): expected-document
    path.

    The BPB-style "source contract" branch of ``_concept_source_identity``
    derives ``mode``/``domain`` from an already-resolved expected document
    instead of a whitelist path. Reuses ``_build()``'s small, unrelated
    fixture (default concept source: ``statistics_domain`` over
    ``(2, 3, 4, 6)``) purely to obtain a schema-valid document with that
    mode/domain -- only the ``concept_source`` coordinate is borrowed from it
    (see ``_concept_source_identity``'s own docstring); the empty-projection
    condition is entirely a property of THIS test's own disjoint-mask target
    fixture, unrelated to whatever the small source document's own checkpoint
    looked like."""
    expected_document = _build(tmp_path).document()
    request, fake_sae = _write_empty_projection_fixture(tmp_path)
    request = replace(request, expected_feature_contract_document=expected_document)
    with pytest.raises(ClusterFeatureContractError, match="concept source projection is empty"):
        resolve_cluster_feature_contract(request, sae_loader=lambda _resolved: fake_sae)


def _rehash_document(document: dict[str, Any]) -> dict[str, Any]:
    """Recompute ``feature_contract_fingerprint`` for a (possibly mutated)
    document's ``semantic`` payload, using the same canonicalization
    convention (sorted keys, compact separators, non-ASCII passthrough)
    production's own ``_canonical_json``/``_sha256_json`` use -- reimplemented
    here via plain ``json``/``hashlib`` rather than by calling those private
    helpers (mirrors the golden generator's own ``normalized_sha256``
    precedent: "replicated here via plain json/hashlib, not by calling those
    private helpers, so this is a genuinely separate computation"), so a test
    built on this stays an independent check of "is this document's SHAPE
    rejected", not a self-referential one that would pass even if
    production's own hashing were broken the same way."""
    semantic = document["semantic"]
    fingerprint = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return {**document, "feature_contract_fingerprint": fingerprint}


def test_validate_feature_contract_document_rejects_empty_concept_source_projection(
    tmp_path: Path,
) -> None:
    """Codex pass-2 residual finding 2 (MEDIUM): a THIRD path to the same E4
    empty-projection condition (see ``test_build_raises_when_statistics_
    domain_concept_projection_is_empty_after_masking`` and ``test_full_
    filesystem_preflight_raises_when_concept_projection_is_empty_after_
    masking`` above for the first two) -- a document that never went through
    the live builder at all. A hand-authored ``expected_feature_contract_
    document``, or a sidecar written by an older stack build predating this
    invariant, can carry an empty ``source_to_representative`` while still
    being perfectly schema-valid and correctly hashed against its OWN
    (empty-projection) semantic content.

    Mutating an otherwise-valid document's ``source_to_representative`` to
    empty and RE-HASHING it (via ``_rehash_document``, not production's own
    hasher) proves this is a genuinely independent guard, not a side effect
    of the pre-existing fingerprint check: ``test_sidecar_document_detects_
    semantic_mutation`` above already covers a MISMATCHED hash separately;
    this fixture's fingerprint DOES match its (empty-projection) content, so
    the only thing that can reject it is the new E4 guard itself."""
    document = _build(tmp_path).document()
    document["semantic"]["concept_source"]["source_to_representative"] = []
    corrupted = _rehash_document(document)
    assert corrupted["feature_contract_fingerprint"] != document["feature_contract_fingerprint"]

    with pytest.raises(ClusterFeatureContractError, match="concept source projection is empty"):
        validate_feature_contract_document(corrupted)


def test_load_feature_contract_sidecar_rejects_empty_concept_source_projection(
    tmp_path: Path,
) -> None:
    """Same condition as the in-memory test directly above, through the
    on-disk sidecar-loading path instead: ``load_feature_contract_sidecar``
    fuse-hashes the file and delegates to ``validate_feature_contract_
    document`` (see that function's own module docstring), so this proves
    the guard closes the ON-DISK route too, not only a direct in-memory
    call."""
    document = _build(tmp_path).document()
    document["semantic"]["concept_source"]["source_to_representative"] = []
    corrupted = _rehash_document(document)
    sidecar_path = tmp_path / "corrupted_sidecar.json"
    sidecar_path.write_text(json.dumps(corrupted, indent=2), encoding="utf-8")

    with pytest.raises(ClusterFeatureContractError, match="concept source projection is empty"):
        load_feature_contract_sidecar(sidecar_path)


# ─────────────────────────────────────────────────────────────────────────────
# #931 — the member/source grouping must stay one-pass, not a per-element scan
# ─────────────────────────────────────────────────────────────────────────────


class _ElementAccessCountingTensor(t.Tensor):
    """A ``rep_of`` stand-in that counts every element-wise read.

    ``__getitem__`` is the choke point for per-element access: ``rep_of[i].item()``,
    ``int(rep_of[i])`` and ``float(rep_of[i])`` all route through it, so counting here
    guards the actual property (no element-wise reads) rather than one conversion
    method. ``.tolist()`` is a bulk C-level op and does NOT go through ``__getitem__``,
    which is exactly the distinction under test.
    """

    accesses = 0

    def __getitem__(self, index: Any) -> Any:
        type(self).accesses += 1
        return super().__getitem__(index)


def test_build_performs_no_per_element_rep_of_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``build_cluster_feature_contract`` must never index ``rep_of`` element by element.

    Before #931 the member grouping scanned ``representatives x allowed_ids`` and read
    ``rep_of`` on every inner iteration -- O(|R|*|A|) device round-trips, measured at ~700ns
    each. That is ~268M reads (~190s) at 16k SAE width and ~4.3B (~51min) at 65k, paid once
    per clustered run before any real work happens.

    The invariant that replaced it is size-independent, so it bites at this width-8 fixture
    without needing a realistic SAE: the whole build performs **zero** element-wise reads of
    ``rep_of``, because it is materialized once via ``.tolist()``.

    Counting ``__getitem__`` rather than ``Tensor.item()`` is deliberate. An earlier version of
    this guard counted ``item()`` calls, and a regression written as
    ``int(consolidation.rep_of[feature_id])`` restored the full O(|R|*|A|) scan while passing,
    because ``Tensor.__int__`` never calls the patched ``item()``. Counting the subscript
    closes that bypass for every conversion form.
    """
    real_build = FeatureClusterConsolidation.build

    def spying_build(
        resolved: ResolvedFeatureClusters, feature_mask: FeatureMask | None
    ) -> FeatureClusterConsolidation:
        consolidation = real_build(resolved, feature_mask)
        return replace(
            consolidation,
            rep_of=consolidation.rep_of.as_subclass(_ElementAccessCountingTensor),
        )

    real_grouping = contract_module._group_allowed_ids_by_representative
    grouping_calls: list[tuple[int, ...]] = []

    def spying_grouping(
        allowed_ids: Sequence[int], representative_of_feature_id: Sequence[int]
    ) -> dict[int, list[int]]:
        grouping_calls.append(tuple(allowed_ids))
        return real_grouping(allowed_ids, representative_of_feature_id)

    _ElementAccessCountingTensor.accesses = 0
    monkeypatch.setattr(FeatureClusterConsolidation, "build", spying_build)
    monkeypatch.setattr(
        contract_module, "_group_allowed_ids_by_representative", spying_grouping
    )
    contract = _build(tmp_path)

    assert contract.representative_universe == (1, 3, 4, 6)
    # Proving the helper is one-pass is worthless if production stops calling it: an inlined
    # quadratic scan over the materialized list touches no tensor, so the access counter above
    # stays at zero. Pin that the validated path is the one actually taken.
    assert grouping_calls == [(1, 2, 3, 4, 5, 6)], (
        f"expected exactly one grouping call over allowed_ids, got {grouping_calls!r}; "
        "build_cluster_feature_contract must route through "
        "_group_allowed_ids_by_representative (#931)"
    )
    assert _ElementAccessCountingTensor.accesses == 0, (
        f"build_cluster_feature_contract performed "
        f"{_ElementAccessCountingTensor.accesses} element-wise rep_of read(s); it must "
        "materialize rep_of once with .tolist() instead of indexing it per element (#931)"
    )


class _LookupCountingSequence(Sequence[int]):
    """A ``representative_of_feature_id`` stand-in that counts every lookup."""

    def __init__(self, values: Sequence[int]) -> None:
        self._values = list(values)
        self.lookups = 0

    def __getitem__(self, index: int) -> int:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self.lookups += 1
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


def test_grouping_helper_performs_exactly_one_lookup_per_allowed_id() -> None:
    """The grouping must be ONE pass -- exactly ``len(allowed_ids)`` lookups (#931).

    This is the exact form of the invariant. The tensor-side guard above catches a regression
    that re-reads ``rep_of`` per element, but it is blind to a regression that scans the
    already-materialized Python list quadratically (no tensor access at all, so nothing to
    count). That variant is still O(|R|*|A|) -- ~4.3B iterations at 65k SAE width -- so it needs
    its own guard, and counting lookups into the sequence is the property itself rather than a
    proxy for it.
    """
    representative_of_feature_id = _LookupCountingSequence([0, 1, 1, 3, 4, 4, 6, 7])
    allowed_ids = (1, 2, 3, 4, 5, 6)

    grouped = contract_module._group_allowed_ids_by_representative(
        allowed_ids,
        representative_of_feature_id,
    )

    assert grouped == {1: [1, 2], 3: [3], 4: [4, 5], 6: [6]}
    assert representative_of_feature_id.lookups == len(allowed_ids), (
        f"grouping performed {representative_of_feature_id.lookups} lookups for "
        f"{len(allowed_ids)} allowed ids; it must be a single pass (#931)"
    )


def test_grouping_helper_preserves_allowed_id_order_within_each_group() -> None:
    """Order within a group must follow ``allowed_ids``, which is what keeps the fingerprint stable."""
    grouped = contract_module._group_allowed_ids_by_representative(
        (7, 3, 5, 1),
        [0, 9, 2, 9, 4, 9, 6, 9],
    )

    assert grouped == {9: [7, 3, 5, 1]}


def test_consolidation_rep_of_is_int64() -> None:
    """``rep_of`` is int64, which is what makes #931's ``int(...)`` coercion a provable no-op.

    ``build_cluster_feature_contract`` reads ``rep_of`` via ``.tolist()`` and still wraps each
    value in ``int(...)`` so a hypothetical non-integer tensor would truncate exactly as the old
    ``int(tensor.item())`` did. That defensive branch is unreachable through the public path --
    dropping the ``int(...)`` is an equivalent mutant no test can distinguish -- so this pins the
    property it rests on at its source instead: ``FeatureClusterConsolidation.build`` produces an
    int64 ``rep_of``, for which ``.tolist()`` already yields Python ints.
    """
    definition = FeatureClusterDefinition(clusters=((1, 2), (4, 5)))
    resolved = ResolvedFeatureClusters(
        source_path=Path("/test/clusters.yaml"),
        definition=definition,
        descriptor=FeatureClusterDescriptor.from_definition(definition, n_sae_features=8),
    )
    mask = FeatureMask(FeatureMaskMode.whitelist, (1, 2, 3, 4, 5, 6))

    consolidation = FeatureClusterConsolidation.build(resolved, mask)

    assert consolidation.rep_of.dtype == t.int64
    assert all(type(value) is int for value in consolidation.rep_of.tolist())


# ---------------------------------------------------------------------------
# Splice-overlay tolerance in the checkpoint feature-directory identity check.
#
# `scripts.preprocess.splice_feature_shards` (issue #590) writes its derived
# shards to `<feature_dir>/splice/cut_frac{cut_fraction}/seed{seed}` by
# default (configs/extraction/splice_feature_shards.yaml). A checkpoint
# trained on those shards records that derived directory in
# `user_config.feature_dir`, which `parse_feature_dir` cannot parse: it strips
# only the `context_independent` / `no_bos` leaves, so the trailing three
# overlay components shift `parts[-7:]` onto the wrong window.
#
# The overlay describes how the training corpus was recombined, not which SAE
# feature stack produced it -- which is what this identity check compares.
# ---------------------------------------------------------------------------

_CANONICAL_FEATURE_DIR = (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base"
    "/context_independent/no_bos"
)


@pytest.mark.parametrize("seed_leaf", ["seed42", "seed42_opposite_percentile_pool64"])
def test_strip_splice_overlay_removes_a_trailing_splice_directory(seed_leaf: str) -> None:
    spliced = f"{_CANONICAL_FEATURE_DIR}/splice/cut_frac0.15/{seed_leaf}"
    canonical, overlay = contract_module._strip_splice_overlay(spliced)
    assert canonical == _CANONICAL_FEATURE_DIR
    assert overlay == f"splice/cut_frac0.15/{seed_leaf}"


def test_strip_splice_overlay_leaves_a_canonical_directory_untouched() -> None:
    canonical, overlay = contract_module._strip_splice_overlay(_CANONICAL_FEATURE_DIR)
    assert canonical == _CANONICAL_FEATURE_DIR
    assert overlay is None


@pytest.mark.parametrize(
    "trailing",
    [
        "splice/cut_frac0.15/notseed",  # third component is not seed*
        "splice/frac0.15/seed42",  # second component is not cut_frac*
        "notsplice/cut_frac0.15/seed42",  # first component is not 'splice'
    ],
)
def test_strip_splice_overlay_ignores_a_non_matching_trailing_triple(
    trailing: str,
) -> None:
    """Only the exact documented overlay shape is stripped.

    Anything else stays in the path so `parse_feature_dir` still raises on it,
    rather than this helper silently discarding real path components.
    """

    path = f"{_CANONICAL_FEATURE_DIR}/{trailing}"
    canonical, overlay = contract_module._strip_splice_overlay(path)
    assert canonical == path
    assert overlay is None


def test_strip_splice_overlay_accepts_an_integral_cut_fraction() -> None:
    spliced = f"{_CANONICAL_FEATURE_DIR}/splice/cut_frac1/seed0"
    canonical, overlay = contract_module._strip_splice_overlay(spliced)
    assert canonical == _CANONICAL_FEATURE_DIR
    assert overlay == "splice/cut_frac1/seed0"


def test_spliced_checkpoint_feature_dir_matches_the_canonical_configured_identity() -> None:
    """End-to-end: the assertion this helper exists for now passes.

    Both sides describe the same SAE stack; only the checkpoint side carries
    the splice overlay.
    """

    from steering.feature_paths import parse_feature_dir

    configured = contract_module.FeatureDirectoryIdentity.from_components(
        parse_feature_dir(_CANONICAL_FEATURE_DIR)
    )
    metadata = {
        "user_config": {
            "feature_dir": f"{_CANONICAL_FEATURE_DIR}/splice/cut_frac0.15/seed42"
        }
    }
    contract_module._assert_checkpoint_feature_directory_identity(metadata, configured)


def test_spliced_checkpoint_feature_dir_still_rejects_a_different_stack() -> None:
    """Stripping the overlay must not stop a genuine stack mismatch raising."""

    from steering.feature_paths import parse_feature_dir

    configured = contract_module.FeatureDirectoryIdentity.from_components(
        parse_feature_dir(_CANONICAL_FEATURE_DIR)
    )
    other_stack = _CANONICAL_FEATURE_DIR.replace("/l22/", "/l17/")
    metadata = {
        "user_config": {"feature_dir": f"{other_stack}/splice/cut_frac0.15/seed42"}
    }
    with pytest.raises(
        ClusterFeatureContractError, match="does not match checkpoint training"
    ):
        contract_module._assert_checkpoint_feature_directory_identity(metadata, configured)
