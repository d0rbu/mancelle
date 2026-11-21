from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

from errors import FeatureClustersError
from steering.conditional.feature_clusters import (
    FEATURE_CLUSTERS_SCHEMA_VERSION,
    FEATURE_CLUSTERS_SEMANTICS,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
    canonical_feature_clusters_descriptor,
    resolve_feature_clusters,
)
from steering.packed_batch_io import compute_selection_hash

_MALFORMED_SHA256_DIGESTS = (
    pytest.param("+" + "a" * 63, id="leading-plus"),
    pytest.param("-" + "a" * 63, id="leading-minus"),
    pytest.param(" " + "a" * 63, id="leading-space"),
    pytest.param("a" * 31 + "_" + "a" * 32, id="underscore"),
    pytest.param("0x" + "a" * 62, id="hex-prefix"),
    pytest.param("A" * 64, id="uppercase"),
    pytest.param("١" * 64, id="arabic-indic-digits"),
)


def _write(path: Path, text: str | bytes) -> Path:
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _resolve(path: Path, n_sae_features: int = 65_536) -> ResolvedFeatureClusters:
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        n_sae_features,
    )
    assert resolved is not None
    return resolved


def _json_file(tmp_path: Path, clusters, *, schema_version=1, name="clusters.json") -> Path:
    return _write(
        tmp_path / name,
        json.dumps({"schema_version": schema_version, "clusters": clusters}),
    )


def test_yaml_and_json_logical_content_have_exact_same_canonical_identity(tmp_path: Path):
    yaml_path = _write(
        tmp_path / "clusters.yaml",
        """
        # Source order and comments are provenance only.
        schema_version: 1
        clusters:
          - [9405, 40, 443]
          - [7, 2]
        """,
    )
    json_path = _write(
        tmp_path / "clusters.json",
        '{ "clusters": [[2, 7], [443, 9405, 40]], "schema_version": 1 }',
    )

    yaml_resolved = _resolve(yaml_path)
    json_resolved = _resolve(json_path)

    expected = {
        "schema_version": 1,
        "clusters": [[2, 7], [40, 443, 9405]],
    }
    assert yaml_resolved.canonical_dict() == expected
    assert json_resolved.canonical_dict() == expected
    assert yaml_resolved.definition.canonical_json() == (
        '{"clusters":[[2,7],[40,443,9405]],"schema_version":1}'
    )
    expected_sha = hashlib.sha256(
        yaml_resolved.definition.canonical_json().encode("utf-8")
    ).hexdigest()
    assert yaml_resolved.definition.content_sha256() == expected_sha
    assert yaml_resolved.descriptor == json_resolved.descriptor
    assert yaml_resolved.descriptor.identity_sha256() == (
        json_resolved.descriptor.identity_sha256()
    )
    assert yaml_resolved.source_path != json_resolved.source_path


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
def test_both_yaml_suffixes_are_supported(tmp_path: Path, suffix: str):
    path = _write(
        tmp_path / f"clusters{suffix}",
        "schema_version: 1\nclusters:\n  - [1, 2]\n",
    )
    assert _resolve(path, 3).definition.clusters == ((1, 2),)


def test_member_and_cluster_permutations_do_not_change_identity(tmp_path: Path):
    a = _json_file(tmp_path, [[9, 1, 4], [8, 2]], name="a.json")
    b = _json_file(tmp_path, [[2, 8], [4, 9, 1]], name="b.json")
    ra = _resolve(a, 10)
    rb = _resolve(b, 10)
    assert ra.definition == rb.definition
    assert ra.descriptor == rb.descriptor
    assert ra.descriptor.dirname() == rb.descriptor.dirname()


def test_formatting_comments_and_source_path_do_not_change_identity(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    a = _write(
        tmp_path / "a.yaml",
        "schema_version: 1\nclusters: [[1, 3], [4, 8]]\n",
    )
    b = _write(
        nested / "renamed.yaml",
        """
        # Equivalent content with verbose formatting.
        clusters:
          - - 8
            - 4
          - - 3
            - 1
        schema_version: 1
        """,
    )
    ra = _resolve(a, 9)
    rb = _resolve(b, 9)
    assert ra.descriptor == rb.descriptor
    assert ra.source_path == a.resolve()
    assert rb.source_path == b.resolve()


def test_symlink_path_resolves_to_regular_source_and_same_identity(tmp_path: Path):
    target = _json_file(tmp_path, [[0, 1]], name="target.json")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    direct = _resolve(target, 2)
    linked = _resolve(link, 2)
    assert linked.source_path == target.resolve()
    assert linked.descriptor == direct.descriptor


def test_symlink_loop_is_wrapped_in_typed_path_error(tmp_path: Path):
    loop = tmp_path / "loop.yaml"
    loop.symlink_to(loop.name)
    with pytest.raises(FeatureClustersError, match="not found or inaccessible"):
        _resolve(loop, 2)


def test_same_path_changed_content_changes_content_and_descriptor_hash(tmp_path: Path):
    path = _json_file(tmp_path, [[0, 1]], name="mutable.json")
    before = _resolve(path, 4)
    _json_file(tmp_path, [[0, 2]], name="mutable.json")
    after = _resolve(path, 4)
    assert before.source_path == after.source_path
    assert before.definition.content_sha256() != after.definition.content_sha256()
    assert before.descriptor.identity_sha256() != after.descriptor.identity_sha256()


def test_original_sae_width_is_descriptor_identity_but_not_content_identity(tmp_path: Path):
    path = _json_file(tmp_path, [[1, 2]])
    narrow = _resolve(path, 3)
    wide = _resolve(path, 65_536)
    assert narrow.definition.content_sha256() == wide.definition.content_sha256()
    assert narrow.descriptor.n_sae_features == 3
    assert wide.descriptor.n_sae_features == 65_536
    assert narrow.descriptor != wide.descriptor
    assert narrow.descriptor.identity_sha256() != wide.descriptor.identity_sha256()


def test_descriptor_binds_schema_width_content_and_semantics(tmp_path: Path):
    resolved = _resolve(_json_file(tmp_path, [[1, 2]]), 10)
    assert resolved.descriptor_dict() == {
        "schema_version": FEATURE_CLUSTERS_SCHEMA_VERSION,
        "n_sae_features": 10,
        "content_sha256": resolved.definition.content_sha256(),
        "semantics": FEATURE_CLUSTERS_SEMANTICS,
    }
    assert len(resolved.descriptor.identity_sha256()) == 64
    assert resolved.descriptor.dirname() == (
        f"fc_{resolved.descriptor.identity_sha256()}"
    )


def test_canonical_descriptor_none_is_exact_noop():
    assert canonical_feature_clusters_descriptor(None) is None


def test_canonical_descriptor_returns_new_plain_dict(tmp_path: Path):
    resolved = _resolve(_json_file(tmp_path, [[1, 2]]), 3)
    first = canonical_feature_clusters_descriptor(resolved)
    second = canonical_feature_clusters_descriptor(resolved)
    assert first is not None and second is not None
    assert first == second == resolved.descriptor_dict()
    assert first is not second
    assert all(type(key) is str for key in first)


def test_absent_and_explicit_null_config_both_disable_clusters():
    assert resolve_feature_clusters({}, 10) is None
    assert resolve_feature_clusters({"feature_clusters": None}, 10) is None
    assert (
        resolve_feature_clusters(OmegaConf.create({"feature_clusters": None}), 10)
        is None
    )


def test_non_mapping_config_root_is_rejected() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        resolve_feature_clusters(object(), 10)  # ty: ignore[invalid-argument-type]


def test_omegaconf_path_block_resolves_without_coercing_content(tmp_path: Path):
    path = _json_file(tmp_path, [[0, 2]])
    cfg = OmegaConf.create({"feature_clusters": {"path": str(path)}})
    resolved = resolve_feature_clusters(cfg, 3)
    assert resolved is not None
    assert resolved.definition.clusters == ((0, 2),)


@pytest.mark.parametrize(
    "block,match",
    [
        (True, "path-only mapping"),
        ([], "path-only mapping"),
        ("clusters.json", "path-only mapping"),
        ({}, "exactly one key"),
        ({"path": "x", "enabled": True}, "exactly one key"),
        ({"checksum": "abc"}, "exactly one key"),
        ({"path": None}, "non-empty string"),
        ({"path": True}, "non-empty string"),
        ({"path": 123}, "non-empty string"),
        ({"path": ""}, "non-empty string"),
        ({"path": "   "}, "non-empty string"),
    ],
)
def test_path_only_config_shape_is_strict(block, match: str):
    with pytest.raises(FeatureClustersError, match=match):
        resolve_feature_clusters({"feature_clusters": block}, 10)


def test_missing_path_is_actionable_and_does_not_create_anything(tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FeatureClustersError, match="not found or inaccessible"):
        resolve_feature_clusters(
            {"feature_clusters": {"path": str(missing)}},
            10,
        )
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_directory_path_is_rejected(tmp_path: Path):
    directory = tmp_path / "clusters.yaml"
    directory.mkdir()
    with pytest.raises(FeatureClustersError, match="not a regular file"):
        resolve_feature_clusters(
            {"feature_clusters": {"path": str(directory)}},
            10,
        )


@pytest.mark.parametrize("suffix", [".txt", ".toml", "", ".JSON.backup"])
def test_unsupported_suffix_is_rejected(tmp_path: Path, suffix: str):
    path = _write(tmp_path / f"clusters{suffix}", "{}")
    with pytest.raises(FeatureClustersError, match=r"\.yaml, \.yml, or \.json"):
        _resolve(path, 10)


@pytest.mark.parametrize(
    "text",
    [
        "{",
        '{"schema_version": 1, "clusters": [[1, 2]],}',
        '{"schema_version": 1 "clusters": [[1, 2]]}',
    ],
)
def test_malformed_json_is_wrapped_in_typed_error(tmp_path: Path, text: str):
    path = _write(tmp_path / "bad.json", text)
    with pytest.raises(FeatureClustersError, match="could not parse"):
        _resolve(path, 10)


@pytest.mark.parametrize(
    "text",
    [
        "schema_version: [\n",
        "clusters: [[1, 2]\n",
        "!unsupported_tag value\n",
    ],
)
def test_malformed_or_unsafe_yaml_is_wrapped_in_typed_error(tmp_path: Path, text: str):
    path = _write(tmp_path / "bad.yaml", text)
    with pytest.raises(FeatureClustersError, match="could not parse"):
        _resolve(path, 10)


def test_non_utf8_input_is_wrapped_in_typed_error(tmp_path: Path):
    path = _write(tmp_path / "bad.json", b"\xff\xfe\x00")
    with pytest.raises(FeatureClustersError, match="could not read"):
        _resolve(path, 10)


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
def test_duplicate_top_level_mapping_keys_are_rejected(tmp_path: Path, suffix: str):
    if suffix == "json":
        text = (
            '{"schema_version":1,"clusters":[[1,2]],'
            '"clusters":[[3,4]]}'
        )
    else:
        text = "schema_version: 1\nclusters: [[1, 2]]\nclusters: [[3, 4]]\n"
    path = _write(tmp_path / f"duplicate.{suffix}", text)
    with pytest.raises(FeatureClustersError, match="duplicate mapping key 'clusters'"):
        _resolve(path, 10)


def test_yaml_merge_that_redefines_a_key_is_rejected(tmp_path: Path):
    path = _write(
        tmp_path / "merge.yaml",
        """
        defaults: &defaults
          schema_version: 1
          clusters: [[1, 2]]
        <<: *defaults
        clusters: [[3, 4]]
        """,
    )
    with pytest.raises(FeatureClustersError, match="duplicate mapping key"):
        _resolve(path, 10)


@pytest.mark.parametrize(
    "raw,match",
    [
        (None, "top-level mapping"),
        ([], "top-level mapping"),
        ("text", "top-level mapping"),
        ({}, "exactly schema_version and clusters"),
        ({"schema_version": 1}, "exactly schema_version and clusters"),
        ({"clusters": [[1, 2]]}, "exactly schema_version and clusters"),
        (
            {"schema_version": 1, "clusters": [[1, 2]], "producer": "ising"},
            "extra=",
        ),
        ({"schema_version": 1, "clusters": [[1, 2]], 7: "x"}, "extra="),
    ],
)
def test_top_level_source_shape_is_exact(raw, match: str):
    with pytest.raises(FeatureClustersError, match=match):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize("version", [True, False, 1.0, "1", None, 0, 2, -1])
def test_schema_version_is_strict_integer_one(version):
    raw = {"schema_version": version, "clusters": [[1, 2]]}
    with pytest.raises(FeatureClustersError, match="schema_version"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize("clusters", [None, {}, (), "", [], True, 1])
def test_clusters_must_be_nonempty_json_yaml_style_list(clusters):
    raw = {"schema_version": 1, "clusters": clusters}
    with pytest.raises(FeatureClustersError, match="non-empty list"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize("cluster", [None, {}, (), "1,2", 3, True])
def test_each_cluster_must_be_a_list(cluster):
    raw = {"schema_version": 1, "clusters": [cluster]}
    with pytest.raises(FeatureClustersError, match="must be a list"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize("cluster", [[], [1]])
def test_each_source_cluster_requires_two_members(cluster):
    raw = {"schema_version": 1, "clusters": [cluster]}
    with pytest.raises(FeatureClustersError, match="at least two source IDs"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize(
    "bad",
    [True, False, 1.0, 1.5, "1", None, {}, [], (1,), b"1"],
)
def test_feature_ids_are_strict_ints_never_coerced(bad):
    raw = {"schema_version": 1, "clusters": [[0, bad]]}
    with pytest.raises(FeatureClustersError, match="strict int"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize("cluster", [[1, 1], [2, 3, 2]])
def test_duplicate_members_are_rejected_not_silently_deduplicated(cluster):
    raw = {"schema_version": 1, "clusters": [cluster]}
    with pytest.raises(FeatureClustersError, match="duplicate member"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize(
    "clusters,overlap",
    [
        ([[1, 2], [2, 3]], 2),
        ([[8, 9], [1, 8], [3, 4]], 8),
        ([[4, 7, 9], [0, 2], [7, 8]], 7),
    ],
)
def test_clusters_must_be_pairwise_disjoint(clusters, overlap: int):
    raw = {"schema_version": 1, "clusters": clusters}
    with pytest.raises(FeatureClustersError, match=rf"ID {overlap} appears"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=10)


@pytest.mark.parametrize("bad", [-1, -10, 10, 11, 65_536])
def test_ids_must_be_in_original_sae_range(bad: int):
    n = 10 if bad != 65_536 else 65_536
    raw = {"schema_version": 1, "clusters": [[0, bad]]}
    with pytest.raises(FeatureClustersError, match="negative|out of range"):
        FeatureClusterDefinition.from_mapping(raw, n_sae_features=n)


def test_max_uint16_original_id_is_valid_for_65536_wide_sae():
    raw = {"schema_version": 1, "clusters": [[0, 65_535]]}
    definition = FeatureClusterDefinition.from_mapping(raw, n_sae_features=65_536)
    assert definition.clusters == ((0, 65_535),)


@pytest.mark.parametrize("n", [True, False, 0, -1, 1.0, "10", None])
def test_original_width_is_positive_strict_int(n):
    with pytest.raises(BeartypeCallHintParamViolation, match="n_sae_features"):
        FeatureClusterDefinition.from_mapping(
            {"schema_version": 1, "clusters": [[0, 1]]},
            n_sae_features=n,
        )


def test_direct_definition_constructor_canonicalizes_but_rejects_duplicates():
    definition = FeatureClusterDefinition(clusters=((9, 1), (8, 3, 4)))
    assert definition.clusters == ((1, 9), (3, 4, 8))
    with pytest.raises(FeatureClustersError, match="duplicate member"):
        FeatureClusterDefinition(clusters=((1, 1),))


def test_direct_definition_constructor_rejects_overlap_after_canonicalization():
    with pytest.raises(FeatureClustersError, match="appears in more than one"):
        FeatureClusterDefinition(clusters=((9, 1), (4, 9)))


def test_resolved_object_rejects_descriptor_definition_mismatch(tmp_path: Path):
    a = _resolve(_json_file(tmp_path, [[0, 1]], name="a.json"), 3)
    b = _resolve(_json_file(tmp_path, [[0, 2]], name="b.json"), 3)
    with pytest.raises(FeatureClustersError, match="does not match"):
        ResolvedFeatureClusters(
            source_path=a.source_path,
            definition=a.definition,
            descriptor=b.descriptor,
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"schema_version": True}, "schema"),
        ({"schema_version": 2}, "schema"),
        ({"n_sae_features": True}, "positive strict int"),
        ({"n_sae_features": 0}, "positive strict int"),
        ({"content_sha256": "a" * 63}, "64-character"),
        ({"content_sha256": "A" * 64}, "lowercase"),
        ({"content_sha256": "z" * 64}, "hex"),
        ({"semantics": "future_v2"}, "semantics"),
    ],
)
def test_descriptor_fields_are_strict(overrides, match: str):
    kwargs = {
        "schema_version": 1,
        "n_sae_features": 10,
        "content_sha256": "a" * 64,
        "semantics": FEATURE_CLUSTERS_SEMANTICS,
    }
    kwargs.update(overrides)
    refined_field = next(
        (field for field in ("n_sae_features", "content_sha256") if field in overrides),
        None,
    )
    expected_error = (
        BeartypeCallHintParamViolation if refined_field is not None else FeatureClustersError
    )
    expected_match = refined_field if refined_field is not None else match
    with pytest.raises(expected_error, match=expected_match):
        FeatureClusterDescriptor(**kwargs)


@pytest.mark.parametrize("content_sha256", _MALFORMED_SHA256_DIGESTS)
@pytest.mark.parametrize("construction", ["direct", "from_dict"])
def test_descriptor_rejects_noncanonical_sha256_strings(
    content_sha256: str,
    construction: str,
) -> None:
    descriptor = {
        "schema_version": FEATURE_CLUSTERS_SCHEMA_VERSION,
        "n_sae_features": 10,
        "content_sha256": content_sha256,
        "semantics": FEATURE_CLUSTERS_SEMANTICS,
    }
    expected_error = BeartypeCallHintParamViolation if construction == "direct" else FeatureClustersError
    with pytest.raises(expected_error, match="content_sha256"):
        if construction == "direct":
            FeatureClusterDescriptor(**descriptor)
        else:
            FeatureClusterDescriptor.from_dict(descriptor)


@pytest.mark.parametrize("content_sha256", _MALFORMED_SHA256_DIGESTS)
def test_selection_hash_rejects_noncanonical_cluster_descriptor_before_return(
    content_sha256: str,
) -> None:
    descriptor = {
        "schema_version": FEATURE_CLUSTERS_SCHEMA_VERSION,
        "n_sae_features": 10,
        "content_sha256": content_sha256,
        "semantics": FEATURE_CLUSTERS_SEMANTICS,
    }
    with pytest.raises(FeatureClustersError):
        compute_selection_hash(
            positive_selection={"mode": "threshold"},
            negative_selection=None,
            delta_config=None,
            num_buckets=4,
            seed=42,
            feature_clusters=descriptor,
        )


@pytest.mark.parametrize(
    ("content_sha256", "expected_identity", "expected_selection_hash"),
    [
        (
            "0" * 64,
            "f9809c808f4120d043b086d62546170498d793d94c6357a4be908b562816ffeb",
            "d262611060451ae549227df17231866949208af4f3734678787accbf9c367e04",
        ),
        (
            "a" * 64,
            "3b63f53b6ac83e8191a90024ff06abbc5a6a1ee1e446e358fd12a299f01a1c06",
            "415e59ae66b7555b64af237431e12a713a4d151bb4607806e9222b19859c00e4",
        ),
    ],
)
def test_canonical_sha256_descriptor_strings_preserve_identity(
    content_sha256: str,
    expected_identity: str,
    expected_selection_hash: str,
) -> None:
    descriptor = FeatureClusterDescriptor(
        n_sae_features=10,
        content_sha256=content_sha256,
    )
    assert FeatureClusterDescriptor.from_dict(descriptor.canonical_dict()) == descriptor
    assert descriptor.content_sha256 == content_sha256
    assert descriptor.identity_sha256() == expected_identity
    assert (
        compute_selection_hash(
            positive_selection={"mode": "threshold"},
            negative_selection=None,
            delta_config=None,
            num_buckets=4,
            seed=42,
            feature_clusters=descriptor.canonical_dict(),
        )
        == expected_selection_hash
    )


def test_descriptor_from_dict_requires_exact_shape(tmp_path: Path):
    descriptor = _resolve(_json_file(tmp_path, [[0, 1]]), 2).descriptor
    assert FeatureClusterDescriptor.from_dict(descriptor.canonical_dict()) == descriptor
    for malformed in (
        None,
        {},
        {**descriptor.canonical_dict(), "path": "/tmp/x"},
        {k: v for k, v in descriptor.canonical_dict().items() if k != "semantics"},
    ):
        with pytest.raises(FeatureClustersError, match="must contain exactly"):
            FeatureClusterDescriptor.from_dict(malformed)


def test_large_real_sae_width_file_parses_without_compact_space_confusion(tmp_path: Path):
    path = _json_file(
        tmp_path,
        [[65_535, 40, 9_405], [32_768, 2]],
        name="real_width.json",
    )
    resolved = _resolve(path, 65_536)
    assert resolved.definition.clusters == (
        (2, 32_768),
        (40, 9_405, 65_535),
    )
    assert resolved.n_sae_features == 65_536
