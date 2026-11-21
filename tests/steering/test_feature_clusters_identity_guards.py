from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from errors import FeatureClustersError, RemovedFeatureError
from steering.conditional.feature_clusters import (
    CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY,
    FEATURE_CLUSTERS_CANONICAL_KEY,
    FEATURE_CLUSTERS_DESCRIPTOR_KEY,
    SELECTION_SHARD_INTEGRITY_KEY,
    ClusteredThresholdSelectionSemantics,
    FeatureClusterConsolidation,
    ResolvedFeatureClusters,
    append_feature_clusters_leaf,
    assert_clustered_selection_shard_integrity,
    assert_feature_clusters_selection_metadata,
    assert_resume_clustered_threshold_semantics_match,
    assert_resume_feature_clusters_match,
    build_clustered_selection_shard_integrity,
    extend_clustered_selection_shard_integrity,
    feature_clusters_checkpoint_metadata,
    feature_clusters_dirname,
    reject_feature_clusters_config,
    reject_feature_clusters_metadata,
    resolve_feature_clusters,
    validated_clustered_selection_shard_names,
)
from steering.removed_features import (
    reject_feature_clustering_config,
    reject_feature_clustering_metadata,
)


def _resolved(
    tmp_path: Path,
    clusters: list[list[int]],
    *,
    n_sae_features: int = 16,
    name: str = "clusters.json",
) -> ResolvedFeatureClusters:
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": clusters}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        n_sae_features,
    )
    assert resolved is not None
    return resolved


def _selection_metadata(
    consolidation: FeatureClusterConsolidation,
    *,
    status: str = "complete",
) -> dict:
    cluster_metadata = consolidation.metadata_dict()
    cluster_metadata["status"] = status
    return {"feature_clusters": cluster_metadata}


def _selection_integrity_fixture(tmp_path: Path) -> tuple[Path, dict]:
    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    second = selection_dir / "shard_00001.safetensors.zst"
    first = selection_dir / "shard_00000.safetensors.zst"
    second.write_bytes(b"second-selection-shard")
    first.write_bytes(b"first-selection-shard")
    metadata = {
        SELECTION_SHARD_INTEGRITY_KEY: build_clustered_selection_shard_integrity(
            [second, first]
        )
    }
    return selection_dir, metadata


def test_clustered_selection_integrity_is_ordered_and_exact(tmp_path: Path):
    selection_dir, metadata = _selection_integrity_fixture(tmp_path)
    integrity = metadata[SELECTION_SHARD_INTEGRITY_KEY]

    assert integrity["schema_version"] == 1
    assert [entry["name"] for entry in integrity["shards"]] == [
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
    ]
    assert [entry["byte_length"] for entry in integrity["shards"]] == [
        len(b"first-selection-shard"),
        len(b"second-selection-shard"),
    ]
    assert (
        integrity["shards"][0]["content_sha256"]
        == hashlib.sha256(b"first-selection-shard").hexdigest()
    )
    assert (
        integrity["shards"][1]["content_sha256"]
        == hashlib.sha256(b"second-selection-shard").hexdigest()
    )
    assert_clustered_selection_shard_integrity(
        selection_dir,
        metadata,
        surface="unit selection",
    )


@pytest.mark.parametrize(
    ("path_factory", "match"),
    [
        pytest.param(
            lambda tmp_path: [],
            "requires at least one shard",
            id="empty-artifact-set",
        ),
        pytest.param(
            lambda tmp_path: [
                tmp_path / "duplicate.safetensors.zst",
                tmp_path / "duplicate.safetensors.zst",
            ],
            "duplicate filenames",
            id="duplicate-filename",
        ),
        pytest.param(
            lambda tmp_path: [tmp_path / "wrong-extension.safetensors"],
            "requires .safetensors.zst files",
            id="wrong-extension",
        ),
        pytest.param(
            lambda tmp_path: [tmp_path / "missing.safetensors.zst"],
            "cannot read",
            id="missing-file",
        ),
    ],
)
def test_clustered_selection_integrity_builder_rejects_invalid_artifact_sets(
    tmp_path: Path,
    path_factory,
    match: str,
):
    paths = path_factory(tmp_path)
    for path in paths:
        if path.name != "missing.safetensors.zst":
            path.write_bytes(b"selection")

    with pytest.raises(FeatureClustersError, match=match):
        build_clustered_selection_shard_integrity(paths)


def test_clustered_selection_integrity_builder_wraps_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    shard = tmp_path / "shard_00000.safetensors.zst"
    shard.write_bytes(b"selection")
    original_open = Path.open

    def unreadable(path: Path, *args, **kwargs):
        if path == shard:
            raise OSError("injected read failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable)
    with pytest.raises(FeatureClustersError, match="cannot read") as exc_info:
        build_clustered_selection_shard_integrity([shard])
    assert isinstance(exc_info.value.__cause__, OSError)


def test_clustered_selection_integrity_rejects_symlinked_shard(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.safetensors.zst"
    target.write_bytes(b"selection outside the certified direct path")
    shard = tmp_path / "shard_00000.safetensors.zst"
    shard.symlink_to(target)

    with pytest.raises(FeatureClustersError, match="must be a real regular file"):
        build_clustered_selection_shard_integrity([shard])


def test_clustered_selection_integrity_detects_replacement_between_lstat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "shard_00000.safetensors.zst"
    replacement = tmp_path / "replacement.safetensors.zst"
    shard.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    original_open = Path.open
    replaced = False

    def replace_then_open(path: Path, *args, **kwargs):
        nonlocal replaced
        if path == shard and not replaced:
            replaced = True
            os.replace(replacement, shard)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_then_open)
    with pytest.raises(FeatureClustersError, match="changed before"):
        build_clustered_selection_shard_integrity([shard])


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda metadata: metadata.pop(SELECTION_SHARD_INTEGRITY_KEY),
            "content fingerprint must contain exactly",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY].__setitem__(
                "unexpected", True
            ),
            "must contain exactly",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY].__setitem__(
                "schema_version", True
            ),
            "schema_version must be the strict integer",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY].__setitem__(
                "schema_version", 2
            ),
            "schema_version must be the strict integer",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY].__setitem__(
                "shards", []
            ),
            "shards must be a non-empty list",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY].__setitem__(
                "shards", ()
            ),
            "shards must be a non-empty list",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY][
                "shards"
            ].__setitem__(0, None),
            "entry 0 must contain exactly",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][0].pop(
                "byte_length"
            ),
            "entry 0 must contain exactly",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][
                0
            ].__setitem__("name", "../escaped.safetensors.zst"),
            "invalid direct shard filename",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY][
                "shards"
            ].reverse(),
            "strictly sorted",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY][
                "shards"
            ].__setitem__(
                1,
                copy.deepcopy(metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][0]),
            ),
            "strictly sorted",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][
                0
            ].__setitem__("byte_length", True),
            "non-negative strict int",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][
                0
            ].__setitem__("byte_length", -1),
            "non-negative strict int",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][
                0
            ].__setitem__("content_sha256", "A" * 64),
            "64 lowercase hexadecimal",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][
                0
            ].__setitem__("content_sha256", None),
            "64 lowercase hexadecimal",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY]["shards"][
                0
            ].__setitem__("content_sha256", "0" * 63),
            "64 lowercase hexadecimal",
        ),
        (
            lambda metadata: metadata[SELECTION_SHARD_INTEGRITY_KEY].__setitem__(
                "aggregate_sha256", "0" * 64
            ),
            "aggregate mismatch",
        ),
    ],
)
def test_clustered_selection_integrity_strictly_parses_manifest_metadata(
    tmp_path: Path,
    mutation,
    match: str,
):
    selection_dir, metadata = _selection_integrity_fixture(tmp_path)
    corrupted = copy.deepcopy(metadata)
    mutation(corrupted)

    with pytest.raises(FeatureClustersError, match=match):
        assert_clustered_selection_shard_integrity(
            selection_dir,
            corrupted,
            surface="strict selection",
        )


def test_clustered_selection_integrity_requires_manifest_mapping(tmp_path: Path):
    selection_dir, _metadata = _selection_integrity_fixture(tmp_path)

    with pytest.raises(FeatureClustersError, match="is missing.*metadata"):
        assert_clustered_selection_shard_integrity(
            selection_dir,
            None,
            surface="missing selection",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda selection_dir: (
                selection_dir / "shard_00000.safetensors.zst"
            ).write_bytes(b"other-selection-shard"),
            id="same-length-content-replacement",
        ),
        pytest.param(
            lambda selection_dir: (
                selection_dir / "shard_00002.safetensors.zst"
            ).write_bytes(b"extra-selection-shard"),
            id="extra-shard",
        ),
        pytest.param(
            lambda selection_dir: (
                selection_dir / "shard_00001.safetensors.zst"
            ).unlink(),
            id="missing-shard",
        ),
    ],
)
def test_clustered_selection_integrity_rejects_artifact_set_or_byte_drift(
    tmp_path: Path,
    mutation,
):
    selection_dir, metadata = _selection_integrity_fixture(tmp_path)
    mutation(selection_dir)

    with pytest.raises(FeatureClustersError, match="content fingerprint mismatch"):
        assert_clustered_selection_shard_integrity(
            selection_dir,
            metadata,
            surface="drifted selection",
        )


def test_in_progress_integrity_accepts_untracked_crash_window_file(
    tmp_path: Path,
):
    selection_dir, _metadata = _selection_integrity_fixture(tmp_path)
    first = selection_dir / "shard_00000.safetensors.zst"
    partial_metadata = {
        SELECTION_SHARD_INTEGRITY_KEY: build_clustered_selection_shard_integrity(
            [first]
        )
    }

    names = validated_clustered_selection_shard_names(
        selection_dir,
        partial_metadata,
        surface="interrupted producer",
        allow_untracked=True,
    )

    assert names == frozenset({first.name})
    with pytest.raises(FeatureClustersError, match="content fingerprint mismatch"):
        validated_clustered_selection_shard_names(
            selection_dir,
            partial_metadata,
            surface="completed consumer",
            allow_untracked=False,
        )


@pytest.mark.parametrize("bad", [None, 0, 1, "true", []])
def test_partial_integrity_requires_strict_allow_untracked_bool(
    tmp_path: Path,
    bad,
):
    selection_dir, metadata = _selection_integrity_fixture(tmp_path)
    with pytest.raises(TypeError, match="allow_untracked must be a bool"):
        validated_clustered_selection_shard_names(
            selection_dir,
            metadata,
            surface="strict bool",
            allow_untracked=bad,
        )


@pytest.mark.parametrize("mutation", [b"changed", b""])
def test_in_progress_integrity_never_tolerates_certified_byte_drift(
    tmp_path: Path,
    mutation: bytes,
):
    selection_dir, metadata = _selection_integrity_fixture(tmp_path)
    certified = selection_dir / "shard_00000.safetensors.zst"
    partial_metadata = {
        SELECTION_SHARD_INTEGRITY_KEY: build_clustered_selection_shard_integrity(
            [certified]
        )
    }
    certified.write_bytes(mutation)

    with pytest.raises(FeatureClustersError, match="content fingerprint mismatch"):
        validated_clustered_selection_shard_names(
            selection_dir,
            partial_metadata,
            surface="interrupted producer",
            allow_untracked=True,
        )


def test_incremental_integrity_extension_is_sorted_exact_and_idempotent(
    tmp_path: Path,
):
    selection_dir, _metadata = _selection_integrity_fixture(tmp_path)
    first = selection_dir / "shard_00000.safetensors.zst"
    second = selection_dir / "shard_00001.safetensors.zst"

    progress = extend_clustered_selection_shard_integrity(
        None,
        second,
        surface="progress",
    )
    progress = extend_clustered_selection_shard_integrity(
        progress,
        first,
        surface="progress",
    )
    repeated = extend_clustered_selection_shard_integrity(
        progress,
        first,
        surface="progress",
    )

    assert progress == repeated
    assert progress == build_clustered_selection_shard_integrity([first, second])


def test_incremental_integrity_refuses_to_move_a_certified_identity(
    tmp_path: Path,
):
    shard = tmp_path / "shard_00000.safetensors.zst"
    shard.write_bytes(b"first")
    progress = extend_clustered_selection_shard_integrity(
        None,
        shard,
        surface="progress",
    )
    shard.write_bytes(b"second")

    with pytest.raises(FeatureClustersError, match="refuses to replace"):
        extend_clustered_selection_shard_integrity(
            progress,
            shard,
            surface="progress",
        )


def test_disabled_checkpoint_leaf_is_exact_identity():
    base = "out/conditional/foo"
    assert feature_clusters_dirname(None) == ""
    assert append_feature_clusters_leaf(base, None) is base
    assert append_feature_clusters_leaf(base, None) == base


def test_enabled_checkpoint_leaf_uses_full_descriptor_sha(tmp_path: Path):
    resolved = _resolved(tmp_path, [[1, 2]])
    identity = resolved.descriptor.identity_sha256()
    leaf = feature_clusters_dirname(resolved)
    assert leaf == f"fc_{identity}"
    assert len(leaf) == 3 + 64
    assert append_feature_clusters_leaf("out/base", resolved) == os.path.join(
        "out/base",
        leaf,
    )


def test_checkpoint_leaf_ignores_source_path_but_changes_with_content(tmp_path: Path):
    a = _resolved(tmp_path, [[1, 2]], name="a.json")
    b = _resolved(tmp_path, [[2, 1]], name="b.json")
    c = _resolved(tmp_path, [[1, 3]], name="c.json")
    assert a.source_path != b.source_path
    assert feature_clusters_dirname(a) == feature_clusters_dirname(b)
    assert feature_clusters_dirname(a) != feature_clusters_dirname(c)


def test_checkpoint_leaf_changes_with_original_width(tmp_path: Path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 2]]}),
        encoding="utf-8",
    )
    narrow = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        3,
    )
    wide = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        65_536,
    )
    assert narrow is not None and wide is not None
    assert feature_clusters_dirname(narrow) != feature_clusters_dirname(wide)


def test_disabled_checkpoint_metadata_is_empty_exact_noop():
    assert feature_clusters_checkpoint_metadata(None) == {}


def test_enabled_checkpoint_metadata_is_complete_and_path_independent(tmp_path: Path):
    resolved = _resolved(tmp_path, [[4, 1], [9, 8, 7]])
    metadata = feature_clusters_checkpoint_metadata(resolved)
    assert metadata == {
        FEATURE_CLUSTERS_DESCRIPTOR_KEY: resolved.descriptor_dict(),
        FEATURE_CLUSTERS_CANONICAL_KEY: resolved.canonical_dict(),
    }
    assert str(resolved.source_path) not in json.dumps(metadata)


def test_matching_checkpoint_metadata_resumes(tmp_path: Path):
    configured = _resolved(tmp_path, [[1, 2], [3, 4]])
    assert_resume_feature_clusters_match(
        configured,
        feature_clusters_checkpoint_metadata(configured),
        checkpoint_dir="out/checkpoint",
    )


def test_same_content_at_different_path_resumes(tmp_path: Path):
    configured = _resolved(tmp_path, [[1, 2]], name="configured.json")
    checkpoint_definition = _resolved(tmp_path, [[2, 1]], name="checkpoint.json")
    assert configured.source_path != checkpoint_definition.source_path
    assert_resume_feature_clusters_match(
        configured,
        feature_clusters_checkpoint_metadata(checkpoint_definition),
        checkpoint_dir="out/checkpoint",
    )


@pytest.mark.parametrize("meta", [None, {}, {"step": 10}])
def test_unclustered_run_resumes_unclustered_checkpoint(meta):
    assert_resume_feature_clusters_match(
        None,
        meta,
        checkpoint_dir="out/checkpoint",
    )


@pytest.mark.parametrize(
    "meta",
    [
        {FEATURE_CLUSTERS_DESCRIPTOR_KEY: {"x": 1}},
        {FEATURE_CLUSTERS_CANONICAL_KEY: {"schema_version": 1}},
        {
            FEATURE_CLUSTERS_DESCRIPTOR_KEY: {"x": 1},
            FEATURE_CLUSTERS_CANONICAL_KEY: {"y": 2},
        },
    ],
)
def test_unconfigured_run_rejects_clustered_checkpoint_bidirectionally(meta):
    with pytest.raises(FeatureClustersError, match="was trained with static"):
        assert_resume_feature_clusters_match(
            None,
            meta,
            checkpoint_dir="out/clustered",
        )


def test_unconfigured_run_rejects_user_config_only_cluster_record():
    with pytest.raises(FeatureClustersError, match="trained with static feature clusters"):
        assert_resume_feature_clusters_match(
            None,
            {
                "user_config": {
                    "feature_clusters": {"path": "/producer/clusters.yaml"},
                }
            },
            checkpoint_dir="out/incomplete-cluster-record",
        )


@pytest.mark.parametrize(
    "meta",
    [
        None,
        {},
        {FEATURE_CLUSTERS_DESCRIPTOR_KEY: {"x": 1}},
        {FEATURE_CLUSTERS_CANONICAL_KEY: {"y": 2}},
    ],
)
def test_configured_run_rejects_missing_or_partial_checkpoint_metadata(
    tmp_path: Path,
    meta,
):
    configured = _resolved(tmp_path, [[1, 2]])
    with pytest.raises(FeatureClustersError, match="lacks complete"):
        assert_resume_feature_clusters_match(
            configured,
            meta,
            checkpoint_dir="out/unclustered",
        )


def test_resume_rejects_different_cluster_content_with_same_model_shape(tmp_path: Path):
    configured = _resolved(tmp_path, [[1, 2]], name="configured.json")
    checkpoint = _resolved(tmp_path, [[1, 3]], name="checkpoint.json")
    assert configured.n_sae_features == checkpoint.n_sae_features
    with pytest.raises(FeatureClustersError, match="cluster mismatch"):
        assert_resume_feature_clusters_match(
            configured,
            feature_clusters_checkpoint_metadata(checkpoint),
            checkpoint_dir="out/checkpoint",
        )


def test_resume_rejects_same_content_with_different_original_width(tmp_path: Path):
    path = tmp_path / "clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": [[1, 2]]}),
        encoding="utf-8",
    )
    configured = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        3,
    )
    checkpoint = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        4,
    )
    assert configured is not None and checkpoint is not None
    with pytest.raises(FeatureClustersError, match="cluster mismatch"):
        assert_resume_feature_clusters_match(
            configured,
            feature_clusters_checkpoint_metadata(checkpoint),
            checkpoint_dir="out/checkpoint",
        )


def test_resume_rejects_descriptor_content_disagreement_within_checkpoint(tmp_path: Path):
    a = _resolved(tmp_path, [[1, 2]], name="a.json")
    b = _resolved(tmp_path, [[1, 3]], name="b.json")
    corrupt = {
        FEATURE_CLUSTERS_DESCRIPTOR_KEY: a.descriptor_dict(),
        FEATURE_CLUSTERS_CANONICAL_KEY: b.canonical_dict(),
    }
    with pytest.raises(FeatureClustersError, match="cluster mismatch"):
        assert_resume_feature_clusters_match(
            a,
            corrupt,
            checkpoint_dir="out/corrupt",
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda meta: meta[FEATURE_CLUSTERS_DESCRIPTOR_KEY].__setitem__(
            "schema_version", True
        ),
        lambda meta: meta[FEATURE_CLUSTERS_DESCRIPTOR_KEY].__setitem__(
            "unexpected", 1
        ),
        lambda meta: meta[FEATURE_CLUSTERS_CANONICAL_KEY]["clusters"][0].__setitem__(
            0, True
        ),
        lambda meta: meta[FEATURE_CLUSTERS_CANONICAL_KEY].__setitem__(
            "unexpected", 1
        ),
    ],
)
def test_resume_strictly_rejects_malformed_checkpoint_metadata(
    tmp_path: Path,
    mutate,
):
    configured = _resolved(tmp_path, [[1, 2]])
    corrupt = feature_clusters_checkpoint_metadata(configured)
    mutate(corrupt)
    with pytest.raises(FeatureClustersError, match="invalid static feature-cluster"):
        assert_resume_feature_clusters_match(
            configured,
            corrupt,
            checkpoint_dir="out/corrupt",
        )


# ---------------------------------------------------------------------------
# assert_resume_clustered_threshold_semantics_match (issue: PR #1321 review C1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("meta", [None, {}, {"step": 10}])
def test_clustered_threshold_semantics_resume_rejects_missing_key(meta):
    """A checkpoint with no stamp at all predates versioned semantics (pre-#1321).

    Its ``selection_percentiles`` were computed from post-threshold, boundary-
    relative bucket ranks -- a strictly different currency -- so a missing key
    is rejected exactly like a mismatched one, never silently accepted.
    """
    with pytest.raises(
        FeatureClustersError,
        match=r"checkpoint /ckpt predates versioned clustered-threshold selection semantics",
    ):
        assert_resume_clustered_threshold_semantics_match(meta, surface="checkpoint /ckpt")


def test_clustered_threshold_semantics_resume_rejects_mismatched_value():
    """A wrong (or future) value must be rejected and both values named."""
    with pytest.raises(
        FeatureClustersError,
        match=(
            r"checkpoint /ckpt clustered-threshold selection semantics mismatch: "
            r"checkpoint records 'boundary_relative_rank_v0', expected "
            r"'full_distribution_rank_v1'"
        ),
    ):
        assert_resume_clustered_threshold_semantics_match(
            {CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY: "boundary_relative_rank_v0"},
            surface="checkpoint /ckpt",
        )


def test_clustered_threshold_semantics_resume_accepts_matching_value():
    """The current versioned semantics resumes cleanly (no raise)."""
    result = assert_resume_clustered_threshold_semantics_match(
        {
            CLUSTERED_THRESHOLD_SELECTION_SEMANTICS_META_KEY: (
                ClusteredThresholdSelectionSemantics.FULL_DISTRIBUTION_RANK.value
            )
        },
        surface="checkpoint /ckpt",
    )
    assert result is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda metadata: metadata["feature_clusters"]["counts"].__setitem__(
                "source_clusters", True
            ),
            "source_clusters.*non-negative strict int",
        ),
        (
            lambda metadata: metadata["feature_clusters"]["counts"].__setitem__(
                "unexpected", 1
            ),
            "counts must contain exactly",
        ),
    ],
)
def test_selection_manifest_strictly_parses_derived_counts(
    tmp_path: Path,
    mutate,
    match: str,
):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]]),
        None,
    )
    metadata = {"feature_clusters": configured.metadata_dict() | {"status": "complete"}}
    mutate(metadata)
    with pytest.raises(FeatureClustersError, match=match):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="strict-count manifest",
        )


@pytest.mark.parametrize(
    "cfg",
    [
        {"feature_clusters": {"path": "/tmp/clusters.json"}},
        OmegaConf.create({"feature_clusters": {"path": "/tmp/clusters.json"}}),
        {"feature_clusters": {}},
        {"feature_clusters": False},
    ],
)
def test_unsupported_surface_rejects_every_active_new_config_shape(cfg):
    with pytest.raises(
        FeatureClustersError,
        match="supported only by selected-feature precomputation",
    ):
        reject_feature_clusters_config(cfg, "unit surface")


@pytest.mark.parametrize(
    "cfg",
    [
        None,
        {},
        {"feature_clusters": None},
        OmegaConf.create({"feature_clusters": None}),
    ],
)
def test_unsupported_surface_allows_absent_or_null_new_config(cfg):
    reject_feature_clusters_config(cfg, "unit surface")


def test_unsupported_surface_rejects_non_mapping_config_root():
    with pytest.raises(
        TypeError,
        match=r"feature clusters config must be a mapping, got object",
    ):
        reject_feature_clusters_config(object(), "unit surface")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "meta",
    [
        {FEATURE_CLUSTERS_DESCRIPTOR_KEY: {"x": 1}},
        {FEATURE_CLUSTERS_CANONICAL_KEY: {"y": 2}},
        {"feature_clusters": {"descriptor": {"x": 1}}},
        {"user_config": {"feature_clusters": {"path": "/tmp/x.json"}}},
    ],
)
def test_unsupported_surface_rejects_new_clustered_metadata(meta):
    with pytest.raises(
        FeatureClustersError,
        match="cannot consume a checkpoint/artifact",
    ):
        reject_feature_clusters_metadata(meta, "unit metadata surface")


@pytest.mark.parametrize(
    "meta",
    [
        None,
        [],
        {},
        {FEATURE_CLUSTERS_DESCRIPTOR_KEY: None},
        {FEATURE_CLUSTERS_CANONICAL_KEY: None},
        {"feature_clusters": None},
        {"user_config": {"feature_clusters": None}},
        {"user_config": {}},
    ],
)
def test_unsupported_surface_allows_unclustered_metadata(meta):
    reject_feature_clusters_metadata(meta, "unit metadata surface")


def test_unified_config_guard_preserves_strict_legacy_schema_rejection():
    # The removed legacy key is rejected by mere presence, even null.
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        reject_feature_clustering_config(
            {"feature_clustering": None},
            "legacy surface",
        )
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        reject_feature_clusters_config(
            {"feature_clustering": None},
            "unified surface",
        )
    # The current default is path-only and null means disabled.
    reject_feature_clusters_config(
        {"feature_clusters": None},
        "new surface",
    )


def test_unified_metadata_guard_preserves_typed_legacy_schema_rejection():
    legacy = {"feature_clustering": {"enabled": True}}
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        reject_feature_clustering_metadata(legacy, "legacy metadata surface")
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        reject_feature_clusters_metadata(legacy, "unified metadata surface")


def test_manifest_metadata_contains_full_definition_and_nonidentity_provenance(
    tmp_path: Path,
):
    resolved = _resolved(tmp_path, [[8, 1, 4], [2, 3]])
    consolidation = FeatureClusterConsolidation.build(resolved, None)
    metadata = consolidation.metadata_dict()
    assert metadata["canonical"] == {
        "schema_version": 1,
        "clusters": [[1, 4, 8], [2, 3]],
    }
    assert metadata["descriptor"] == resolved.descriptor_dict()
    assert metadata["source_path"] == str(resolved.source_path)
    # Provenance is inspectable but absent from every identity-bearing object.
    assert "source_path" not in resolved.descriptor_dict()
    assert "source_path" not in resolved.canonical_dict()


def test_selection_manifest_guard_accepts_exact_complete_metadata(tmp_path: Path):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2], [8, 9, 10]]),
        None,
    )
    assert_feature_clusters_selection_metadata(
        configured,
        _selection_metadata(configured),
        surface="unit selection",
    )


def test_selection_manifest_guard_treats_source_path_as_provenance(tmp_path: Path):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]], name="configured.json"),
        None,
    )
    metadata = _selection_metadata(configured)
    metadata["feature_clusters"]["source_path"] = "/different/host/clusters.json"
    assert_feature_clusters_selection_metadata(
        configured,
        metadata,
        surface="moved selection",
    )


def test_selection_manifest_guard_accepts_in_progress_only_for_producer_resume(
    tmp_path: Path,
):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]]),
        None,
    )
    metadata = _selection_metadata(configured, status="in_progress")
    assert_feature_clusters_selection_metadata(
        configured,
        metadata,
        surface="producer resume",
        require_complete=False,
    )
    with pytest.raises(FeatureClustersError, match="status"):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="training consumer",
        )


@pytest.mark.parametrize("metadata", [{}, {"packed_rows": {}}, None])
def test_unclustered_selection_guard_accepts_absent_cluster_metadata(metadata):
    assert_feature_clusters_selection_metadata(
        None,
        metadata,
        surface="ordinary selection",
    )


def test_unclustered_selection_guard_rejects_clustered_artifact(tmp_path: Path):
    clustered = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]]),
        None,
    )
    with pytest.raises(FeatureClustersError, match="cannot consume"):
        assert_feature_clusters_selection_metadata(
            None,
            _selection_metadata(clustered),
            surface="ordinary selection",
        )


@pytest.mark.parametrize("metadata", [None, [], {}, {"feature_clusters": None}])
def test_clustered_selection_guard_requires_mapping_and_cluster_block(
    tmp_path: Path,
    metadata,
):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]]),
        None,
    )
    with pytest.raises(FeatureClustersError, match="metadata|required"):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="clustered selection",
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: raw.pop("counts"), "exactly"),
        (lambda raw: raw.__setitem__("unexpected", 1), "exactly"),
        (lambda raw: raw.__setitem__("source_path", ""), "source_path"),
        (lambda raw: raw.__setitem__("source_path", "   "), "source_path"),
        (lambda raw: raw.__setitem__("source_path", 123), "source_path"),
        (lambda raw: raw.__setitem__("status", "partial"), "status"),
        (lambda raw: raw.__setitem__("counts", {}), "counts must contain exactly"),
    ],
)
def test_clustered_selection_guard_rejects_metadata_shape_and_count_drift(
    tmp_path: Path,
    mutation,
    match: str,
):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2], [4, 5]]),
        None,
    )
    metadata = _selection_metadata(configured)
    mutation(metadata["feature_clusters"])
    with pytest.raises(FeatureClustersError, match=match):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="clustered selection",
        )


def test_clustered_selection_guard_rejects_descriptor_drift(tmp_path: Path):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]], name="configured.json"),
        None,
    )
    other = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 3]], name="other.json"),
        None,
    )
    metadata = _selection_metadata(configured)
    metadata["feature_clusters"]["descriptor"] = other.resolved.descriptor_dict()
    with pytest.raises(FeatureClustersError, match="identity mismatch"):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="clustered selection",
        )


def test_clustered_selection_guard_rejects_canonical_content_drift(tmp_path: Path):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]], name="configured.json"),
        None,
    )
    other = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 3]], name="other.json"),
        None,
    )
    metadata = _selection_metadata(configured)
    metadata["feature_clusters"]["canonical"] = other.resolved.canonical_dict()
    with pytest.raises(FeatureClustersError, match="identity mismatch"):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="clustered selection",
        )


def test_clustered_selection_guard_revalidates_canonical_bounds(tmp_path: Path):
    configured = FeatureClusterConsolidation.build(
        _resolved(tmp_path, [[1, 2]]),
        None,
    )
    metadata = _selection_metadata(configured)
    metadata["feature_clusters"]["canonical"] = {
        "schema_version": 1,
        "clusters": [[1, configured.n_sae_features]],
    }
    with pytest.raises(FeatureClustersError, match="out of range"):
        assert_feature_clusters_selection_metadata(
            configured,
            metadata,
            surface="clustered selection",
        )
