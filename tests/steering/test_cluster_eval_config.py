from __future__ import annotations

import dataclasses

import pytest

from steering.eval.cluster_config import (
    ClusterEvalConfigError,
    axbench_base_config_to_dict,
    axbench_bpb_adv_config_to_dict,
    parse_cluster_eval_paths,
    validate_clustered_method_allowlist,
)


def _cluster_paths() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {"mode": "whitelist", "path": "/tmp/mask.txt"},
        {"path": "/tmp/clusters.yaml"},
    )


def test_path_only_schema_is_paired_and_closed() -> None:
    mask, clusters = _cluster_paths()
    parsed = parse_cluster_eval_paths(
        feature_mask=mask,
        feature_clusters=clusters,
    )
    assert parsed is not None
    assert parsed[0].canonical_dict() == mask
    assert parsed[1].canonical_dict() == clusters

    with pytest.raises(ClusterEvalConfigError, match="together"):
        parse_cluster_eval_paths(feature_mask=mask, feature_clusters=None)
    with pytest.raises(ClusterEvalConfigError, match="exactly"):
        parse_cluster_eval_paths(
            feature_mask={**mask, "ids": [1]},
            feature_clusters=clusters,
        )
    with pytest.raises(ClusterEvalConfigError, match="exactly one key"):
        parse_cluster_eval_paths(
            feature_mask=mask,
            feature_clusters={**clusters, "schema_version": 1},
        )


def test_method_allowlist_rejects_unknown_surface() -> None:
    with pytest.raises(ClusterEvalConfigError, match="unknown.*surface"):
        validate_clustered_method_allowlist(
            ["diffmean_decoder_only"],
            surface="axbench_bpb_ad",
        )


def test_config_to_dict_passthrough_without_new_fields() -> None:
    """On dataclasses lacking the new null fields, both serializers equal plain asdict."""

    @dataclasses.dataclass(frozen=True)
    class _Conditioning:
        feature_mask: dict | None = None

    @dataclasses.dataclass(frozen=True)
    class _LegacyBase:
        conditioning: _Conditioning = dataclasses.field(default_factory=_Conditioning)
        run_tag: str = "t"

    cfg = _LegacyBase()
    assert axbench_base_config_to_dict(cfg) == dataclasses.asdict(cfg)


def test_base_config_to_dict_drops_only_null_feature_clusters() -> None:
    @dataclasses.dataclass(frozen=True)
    class _Conditioning:
        feature_mask: dict | None = None
        feature_clusters: dict | None = None

    @dataclasses.dataclass(frozen=True)
    class _NewBase:
        conditioning: _Conditioning = dataclasses.field(default_factory=_Conditioning)

    payload = axbench_base_config_to_dict(_NewBase())
    assert "feature_clusters" not in payload["conditioning"]
    assert "feature_mask" in payload["conditioning"]

    populated = _NewBase(conditioning=_Conditioning(feature_clusters={"path": "/x"}))
    assert axbench_base_config_to_dict(populated)["conditioning"]["feature_clusters"] == {"path": "/x"}


def test_force_revalidation_false_is_byte_neutral_and_true_is_retained() -> None:
    @dataclasses.dataclass(frozen=True)
    class _Legacy:
        run_tag: str = "legacy"

    @dataclasses.dataclass(frozen=True)
    class _Extended:
        run_tag: str = "legacy"
        force_cluster_shard_revalidation: bool = False

    legacy = dataclasses.asdict(_Legacy())
    for serializer in (
        axbench_base_config_to_dict,
        axbench_bpb_adv_config_to_dict,
    ):
        assert serializer(_Extended()) == legacy
        assert serializer(
            _Extended(force_cluster_shard_revalidation=True)
        ) == {
            **legacy,
            "force_cluster_shard_revalidation": True,
        }


def test_parse_cluster_eval_paths_both_null_returns_none() -> None:
    """The legacy both-disabled case is a plain pass-through, not an error."""

    assert parse_cluster_eval_paths(feature_mask=None, feature_clusters=None) is None


def test_method_allowlist_rejects_missing_allowlist() -> None:
    with pytest.raises(ClusterEvalConfigError, match="requires an explicit method allowlist"):
        validate_clustered_method_allowlist(None, surface="axbench_base")


def test_method_allowlist_rejects_empty_list() -> None:
    with pytest.raises(ClusterEvalConfigError, match="must not be empty"):
        validate_clustered_method_allowlist([], surface="axbench_base")


def test_method_allowlist_rejects_duplicates() -> None:
    with pytest.raises(ClusterEvalConfigError, match="duplicates"):
        validate_clustered_method_allowlist(
            ["control_decoder_only", "control_decoder_only"],
            surface="axbench_base",
        )


def test_method_allowlist_rejects_unknown_method_on_known_surface() -> None:
    with pytest.raises(ClusterEvalConfigError, match="unsupported methods"):
        validate_clustered_method_allowlist(
            ["not_a_real_method"],
            surface="axbench_bpb_adv",
        )
