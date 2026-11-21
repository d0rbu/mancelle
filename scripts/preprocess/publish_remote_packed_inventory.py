"""Build the signed remote packed-row inventory for issue #806 streaming.

``remote.enabled=true`` signs an OFFLOADED family (issue #1463): its bodies live
only in the bucket, so they are streamed back one at a time instead of opened in
place, and the signed inventory is published beside them so the published family
describes itself to pre-selection and remote training.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_json_dump
from steering.packed_rows_offload import build_remote_packed_inventory_from_bucket
from steering.remote_feature_inventory import resolve_remote_feature_prefix
from steering.remote_training import (
    REMOTE_PACKED_INVENTORY_FILENAME,
    HfBucketBackend,
    _join_remote,
    publish_remote_packed_inventory,
    upload_verified_object,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix


def _run_remote(cfg: DictConfig, packed_row_dir: Path) -> dict:
    remote = cfg.remote
    if OmegaConf.select(cfg, "selection_checksum") is not None:
        # The sel/ subtree the cross-check reads is local-only, so honouring the
        # flag here is impossible; dropping it silently is worse than refusing.
        raise ValueError(
            "selection_checksum is not supported with remote.enabled=true: an offloaded "
            "family has no local sel/ subtree to cross-check against"
        )
    for name in ("local_features_root", "remote_packed_root", "staging_dir"):
        if OmegaConf.select(remote, name) is None:
            raise ValueError(f"remote.{name} is required when remote.enabled=true")
    bucket_id = HfBucketId.parse(str(remote.bucket_id))
    packed_prefix = resolve_remote_feature_prefix(
        packed_row_dir,
        local_features_root=Path(str(remote.local_features_root)),
        remote_features_root=NormalizedRemotePrefix.parse(str(remote.remote_packed_root)),
        surface="packed_row_dir",
    )
    backend = HfBucketBackend(bucket_id)
    inventory = build_remote_packed_inventory_from_bucket(
        packed_row_dir,
        backend=backend,
        bucket_id=bucket_id,
        packed_prefix=packed_prefix,
        staging_dir=Path(str(remote.staging_dir)).absolute(),
        retry_timeout_seconds=float(remote.retry_timeout_seconds),
        retry_interval_seconds=float(remote.retry_interval_seconds),
    )
    # Stage, publish, THEN replace the local file: a same-path collision in the
    # bucket must not leave the operator with a rewritten local inventory and a
    # stale remote one.
    inventory_path = packed_row_dir / REMOTE_PACKED_INVENTORY_FILENAME
    staged_path = packed_row_dir / f".{REMOTE_PACKED_INVENTORY_FILENAME}.staged"
    atomic_json_dump(inventory, staged_path)
    # Consumers fetch the inventory from the packed prefix, so an offloaded
    # family is only usable once it is published beside its bodies.
    upload_verified_object(
        backend,
        local_path=staged_path,
        remote_path=_join_remote(str(packed_prefix), REMOTE_PACKED_INVENTORY_FILENAME),
        timeout_seconds=float(remote.retry_timeout_seconds),
        retry_interval_seconds=float(remote.retry_interval_seconds),
    )
    os.replace(staged_path, inventory_path)
    logger.info(f"Published remote packed inventory to {packed_prefix}")
    return inventory


def run(cfg: DictConfig) -> int:
    packed_row_dir = Path(str(cfg.packed_row_dir)).expanduser()
    # `select`, not attribute access: callers have always been able to hand this
    # entrypoint a config carrying only packed_row_dir/selection_checksum.
    if bool(OmegaConf.select(cfg, "remote.enabled", default=False)):
        inventory = _run_remote(cfg, packed_row_dir)
    else:
        selection_checksum = (
            str(cfg.selection_checksum) if cfg.selection_checksum is not None else None
        )
        inventory = publish_remote_packed_inventory(
            packed_row_dir,
            selection_checksum=selection_checksum,
        )
    logger.info(
        "Published local remote packed inventory: dir={} shards={} "
        "packed_row_fingerprint={} aggregate={}",
        packed_row_dir,
        len(inventory["shards"]),
        inventory["packed_row_fingerprint"],
        inventory["aggregate_sha256"],
    )
    return 0


@hydra.main(
    config_path="../../configs/extraction",
    config_name="publish_remote_packed_inventory",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
