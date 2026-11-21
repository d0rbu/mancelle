"""Pure contracts for remote raw-feature inventory normalization."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    RemoteShardFamily,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
    resolve_remote_feature_prefix,
)
from steering.remote_training import RemoteObjectInfo
from steering.remote_types import NormalizedRemotePrefix

PREFIX = NormalizedRemotePrefix.parse("features/base")
RAW = RemoteShardFamily.RAW_FEATURES
OVERLAY = RemoteShardFamily.OVERLAY_LEAF


def _info(
    index: int,
    *,
    family: RemoteShardFamily = RAW,
    size: int | None = None,
    digest: str | None = None,
):
    infix = "_features" if family is RAW else ""
    return RemoteObjectInfo(
        f"features/base/shard_{index:05d}{infix}.safetensors.zst",
        index + 1 if size is None else size,
        digest or f"{index + 1:064x}",
    )


def test_listing_permutations_have_numeric_order_and_identical_digest() -> None:
    expected_digest: str | None = None
    for listing in itertools.permutations([_info(2), _info(0), _info(1)]):
        inventory = normalize_remote_feature_inventory(listing, prefix=PREFIX, family=RAW)
        assert [entry.index for entry in inventory] == [0, 1, 2]
        digest = remote_feature_inventory_digest(inventory)
        expected_digest = digest if expected_digest is None else expected_digest
        assert digest == expected_digest


@pytest.mark.parametrize(
    "replacement",
    [
        RemoteObjectInfo("features/base/shard_00001_features.safetensors.zst", 9, "2" * 64),
        RemoteObjectInfo("features/base/shard_00001_features.safetensors.zst", 2, "9" * 64),
        RemoteObjectInfo("features/base/shard_00009_features.safetensors.zst", 2, "2" * 64),
    ],
)
def test_every_identity_field_changes_digest(replacement: RemoteObjectInfo) -> None:
    baseline = normalize_remote_feature_inventory([_info(0), _info(1)], prefix=PREFIX, family=RAW)
    changed = normalize_remote_feature_inventory([_info(0), replacement], prefix=PREFIX, family=RAW)
    assert remote_feature_inventory_digest(changed) != remote_feature_inventory_digest(baseline)


def test_ignores_unrelated_direct_and_nested_objects() -> None:
    inventory = normalize_remote_feature_inventory(
        [
            _info(0),
            RemoteObjectInfo("features/base/stats.json", 1, "a" * 64),
            RemoteObjectInfo(
                "features/base/nested/shard_00001_features.safetensors.zst",
                1,
                "b" * 64,
            ),
        ],
        prefix=PREFIX,
        family=RAW,
    )
    assert [entry.index for entry in inventory] == [0]


@pytest.mark.parametrize(
    "bad",
    [
        RemoteObjectInfo("features/base/shard_0_features.safetensors.zst", 1, "a" * 64),
        RemoteObjectInfo("features/base/not_features.safetensors.zst", 1, "a" * 64),
        RemoteObjectInfo("features/base/shard_00000_features.safetensors.zst", 0, "a" * 64),
        RemoteObjectInfo("features/base/shard_00000_features.safetensors.zst", True, "a" * 64),
        RemoteObjectInfo("features/base/shard_00000_features.safetensors.zst", 1, "A" * 64),
        RemoteObjectInfo("features/base/shard_00000_features.safetensors.zst", 1, "a" * 63),
        RemoteObjectInfo("other/base/shard_00000_features.safetensors.zst", 1, "a" * 64),
    ],
)
def test_rejects_malformed_identity_and_escape(bad: RemoteObjectInfo) -> None:
    with pytest.raises(ValueError):
        normalize_remote_feature_inventory([bad], prefix=PREFIX, family=RAW)


def test_rejects_duplicate_path_and_duplicate_index() -> None:
    with pytest.raises(ValueError, match="duplicate source shard path"):
        normalize_remote_feature_inventory([_info(0), _info(0)], prefix=PREFIX, family=RAW)
    duplicate_index = RemoteObjectInfo(
        "features/base/./shard_00000_features.safetensors.zst", 1, "b" * 64
    )
    with pytest.raises(ValueError):
        normalize_remote_feature_inventory([_info(0), duplicate_index], prefix=PREFIX, family=RAW)


@pytest.mark.parametrize(
    ("family", "accepted", "rejected"),
    [
        (
            RAW,
            "features/base/shard_00007_features.safetensors.zst",
            "features/base/shard_00007.safetensors.zst",
        ),
        (
            OVERLAY,
            "features/base/shard_00007.safetensors.zst",
            "features/base/shard_00007_features.safetensors.zst",
        ),
    ],
)
def test_family_grammar_accepts_only_its_canonical_spelling(
    family: RemoteShardFamily,
    accepted: str,
    rejected: str,
) -> None:
    accepted_info = RemoteObjectInfo(accepted, 1, "a" * 64)
    inventory = normalize_remote_feature_inventory(
        [accepted_info],
        prefix=PREFIX,
        family=family,
    )
    assert inventory == (RemoteFeatureShard(index=7, info=accepted_info),)
    with pytest.raises(ValueError, match=f"malformed direct {family.value}"):
        normalize_remote_feature_inventory(
            [RemoteObjectInfo(rejected, 1, "b" * 64)],
            prefix=PREFIX,
            family=family,
        )


@pytest.mark.parametrize(
    ("family", "name"),
    [
        (RAW, "shard_0000_features.safetensors.zst"),
        (RAW, "shard_00000_feature.safetensors.zst"),
        (RAW, "shard_00000_features.safetensors"),
        (OVERLAY, "shard_0000.safetensors.zst"),
        (OVERLAY, "shard_00000_overlay.safetensors.zst"),
        (OVERLAY, "shard_00000.safetensors"),
    ],
)
def test_family_grammar_rejects_near_miss_names(
    family: RemoteShardFamily,
    name: str,
) -> None:
    with pytest.raises(ValueError, match=f"malformed direct {family.value}"):
        normalize_remote_feature_inventory(
            [RemoteObjectInfo(f"features/base/{name}", 1, "a" * 64)],
            prefix=PREFIX,
            family=family,
        )


def test_prefix_mapping_resolves_symlink_escape(tmp_path: Path) -> None:
    local_root = tmp_path / "features"
    inside = local_root / "s128" / "base"
    inside.mkdir(parents=True)
    assert (
        resolve_remote_feature_prefix(
            inside,
            local_features_root=local_root,
            remote_features_root=NormalizedRemotePrefix.parse("features"),
        )
        == "features/s128/base"
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    escaping = local_root / "escape"
    escaping.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        resolve_remote_feature_prefix(
            escaping,
            local_features_root=local_root,
            remote_features_root=NormalizedRemotePrefix.parse("features"),
        )
