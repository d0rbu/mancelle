"""Pure remote raw-feature inventory normalization shared by narrow runners."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from steering.feature_io import COMPRESSED_EXT
from steering.remote_training import RemoteObjectInfo
from steering.remote_types import NormalizedRemotePrefix


class RemoteShardFamily(StrEnum):
    """Canonical direct-child body grammar for one sealed remote family."""

    RAW_FEATURES = "raw_features"
    OVERLAY_LEAF = "overlay_leaf"


_REMOTE_FEATURE_NAME_RE = {
    RemoteShardFamily.RAW_FEATURES: re.compile(
        rf"^shard_(?P<index>\d{{5}})_features{re.escape(COMPRESSED_EXT)}$"
    ),
    RemoteShardFamily.OVERLAY_LEAF: re.compile(
        rf"^shard_(?P<index>\d{{5}}){re.escape(COMPRESSED_EXT)}$"
    ),
}
_LOWERCASE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RemoteFeatureShard:
    """One numeric raw feature shard and its sealed remote identity."""

    index: int
    info: RemoteObjectInfo

    def identity_row(self) -> list[str | int]:
        return [self.info.path, self.info.byte_length, self.info.xet_hash]


def resolve_remote_feature_prefix(
    path: str | Path,
    *,
    local_features_root: str | Path,
    remote_features_root: NormalizedRemotePrefix,
    surface: str = "feature_dir",
) -> NormalizedRemotePrefix:
    """Map a symlink-resolved local feature directory beneath a local root."""
    resolved_path = Path(path).resolve()
    resolved_root = Path(local_features_root).resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{surface} {resolved_path} escapes remote.local_features_root {resolved_root}"
        ) from exc
    mapped = (
        str(remote_features_root)
        if relative == Path(".")
        else PurePosixPath(str(remote_features_root), *relative.parts).as_posix()
    )
    return NormalizedRemotePrefix.parse(mapped)


def _relative_remote_parts(
    remote_path: str,
    prefix: NormalizedRemotePrefix,
) -> tuple[str, ...]:
    try:
        normalized = str(NormalizedRemotePrefix.parse(remote_path))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bucket returned a non-normalized object path {remote_path!r}") from exc
    try:
        return PurePosixPath(normalized).relative_to(PurePosixPath(str(prefix))).parts
    except ValueError as exc:
        raise ValueError(
            f"bucket listing for {prefix!r} returned object outside the prefix: {remote_path!r}"
        ) from exc


def normalize_remote_feature_inventory(
    listing: Sequence[RemoteObjectInfo],
    *,
    prefix: NormalizedRemotePrefix,
    family: RemoteShardFamily = RemoteShardFamily.RAW_FEATURES,
) -> tuple[RemoteFeatureShard, ...]:
    """Validate one direct-child shard family and return numeric shard order.

    Nested and unrelated direct objects are intentionally ignored. Direct names
    that resemble feature bodies are rejected so a typo cannot silently shrink
    a sealed family.
    """
    if not isinstance(family, RemoteShardFamily):
        raise TypeError(f"family must be a RemoteShardFamily, got {family!r}")
    name_pattern = _REMOTE_FEATURE_NAME_RE[family]
    by_index: dict[int, RemoteFeatureShard] = {}
    seen_paths: set[str] = set()
    for info in listing:
        parts = _relative_remote_parts(info.path, prefix)
        if len(parts) != 1:
            continue
        name = parts[0]
        match = name_pattern.fullmatch(name)
        if match is None:
            if name.startswith("shard_") or "_features" in name:
                raise ValueError(f"malformed direct {family.value} shard-body name: {info.path}")
            continue
        if isinstance(info.byte_length, bool) or not isinstance(info.byte_length, int):
            raise ValueError(
                f"source object {info.path!r} has invalid byte length {info.byte_length!r}"
            )
        if info.byte_length <= 0:
            raise ValueError(
                f"source object {info.path!r} has invalid byte length {info.byte_length!r}"
            )
        if (
            not isinstance(info.xet_hash, str)
            or _LOWERCASE_SHA256_RE.fullmatch(info.xet_hash) is None
        ):
            raise ValueError(f"source object {info.path!r} has invalid Xet hash {info.xet_hash!r}")
        index = int(match.group("index"))
        duplicate_path = info.path in seen_paths
        duplicate_index = index in by_index
        if any((duplicate_path, duplicate_index)):
            raise ValueError(
                "duplicate source shard path or index in bucket listing: "
                f"path={info.path!r}, index={index}, "
                f"duplicate_path={duplicate_path}, duplicate_index={duplicate_index}"
            )
        seen_paths.add(info.path)
        by_index[index] = RemoteFeatureShard(index=index, info=info)
    return tuple(by_index[index] for index in sorted(by_index))


def remote_feature_inventory_digest(
    inventory: Sequence[RemoteFeatureShard],
) -> str:
    """Hash canonical ``[path, byte_length, xet_hash]`` rows."""
    payload = [entry.identity_row() for entry in inventory]
    return hashlib.sha256(
        json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "RemoteFeatureShard",
    "RemoteShardFamily",
    "normalize_remote_feature_inventory",
    "remote_feature_inventory_digest",
    "resolve_remote_feature_prefix",
]
