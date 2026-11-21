"""Publish the immutable compact HF source required by remote token boundaries."""

from __future__ import annotations

import shutil
import stat
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import hydra
import zstandard
from loguru import logger
from omegaconf import DictConfig

from nanochat.atomic_io import atomic_write_bytes
from steering.feature_io import (
    RegularFileIdentity,
    decode_feature_safetensors_bytes,
    read_feature_compressed_bytes_stable,
)
from steering.overlay_io import BaseShardContentHasher
from steering.remote_training import (
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    upload_verified_object,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.token_boundary_remote import (
    AuthenticatedTokenBoundarySource,
    RemoteTokenBoundaryTransport,
    TokenBoundaryRemoteConfig,
)
from steering.token_boundary_source import (
    TOKEN_BOUNDARY_SOURCE_VECTOR_KEYS,
    TokenBoundarySourceManifest,
    assert_token_boundary_source_tensor_equal,
    compact_token_boundary_payload_from_raw,
    derive_token_boundary_source_prefix,
    load_token_boundary_source_shard_stable,
    save_token_boundary_source_shard,
    serialize_token_boundary_source_manifest,
    token_boundary_source_manifest_path,
    token_boundary_source_shard_path,
)

_RUN_PREFIX = ".token-boundary-source-publisher."


def _resolve_config(cfg: DictConfig) -> TokenBoundaryRemoteConfig:
    raw_prefix = cfg.get("raw_prefix", None)
    if raw_prefix is None:
        raise ValueError("raw_prefix is required")
    staging_raw = cfg.get("staging_dir", None)
    if staging_raw is None:
        raise ValueError("staging_dir is required")
    staging_dir = Path(str(staging_raw)).absolute()
    try:
        observed = staging_dir.lstat()
    except OSError as exc:
        raise ValueError(
            f"staging_dir must be an existing real nonsymlink directory: {staging_dir}"
        ) from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise ValueError(
            f"staging_dir must be an existing real nonsymlink directory: {staging_dir}"
        )
    return TokenBoundaryRemoteConfig(
        bucket_id=HfBucketId.parse(str(cfg.bucket_id)),
        remote_prefix=NormalizedRemotePrefix.parse(str(raw_prefix)),
        staging_dir=staging_dir,
        expected_shard_count=cfg.get("expected_shard_count", None),
        retry_timeout_seconds=cfg.retry_timeout_seconds,
        retry_interval_seconds=cfg.retry_interval_seconds,
    )


def _remove_run_directory(run_dir: Path) -> None:
    if run_dir.parent == run_dir or not run_dir.name.startswith(_RUN_PREFIX):
        raise RuntimeError(f"refusing cleanup outside publisher-owned staging namespace: {run_dir}")
    shutil.rmtree(run_dir)


def run(cfg: DictConfig, *, backend: BucketBackend | None = None) -> int:
    """Build and commit one complete compact family; existing commits authenticate."""
    config = _resolve_config(cfg)
    resolved_backend = backend or HfBucketBackend(config.bucket_id)
    raw_transport = RemoteTokenBoundaryTransport.snapshot(
        config,
        backend=resolved_backend,
    )
    compact_prefix = derive_token_boundary_source_prefix(
        config.remote_prefix,
        raw_transport.inventory_sha256,
    )
    manifest_path = token_boundary_source_manifest_path(compact_prefix)
    if resolved_backend.stat(manifest_path) is not None:
        authenticated = AuthenticatedTokenBoundarySource.snapshot(
            config,
            backend=resolved_backend,
        )
        authenticated.final_inventory_fence()
        logger.info(
            "Compact token-boundary source is already committed and authenticated: "
            f"{compact_prefix}"
        )
        return 0

    run_dir = Path(tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=config.staging_dir))
    base_hasher = BaseShardContentHasher()
    published_compact: list[RemoteObjectInfo] = []
    try:
        with raw_transport:
            for prepared in raw_transport.stream():
                compressed = read_feature_compressed_bytes_stable(
                    prepared.local_path,
                    expected_identity=prepared.file_identity,
                )
                base_hasher.update(prepared.local_path.name, compressed)
                safetensors_bytes = zstandard.ZstdDecompressor().decompress(compressed)
                del compressed

                raw_data = decode_feature_safetensors_bytes(
                    safetensors_bytes,
                    keys=set(TOKEN_BOUNDARY_SOURCE_VECTOR_KEYS),
                )
                del safetensors_bytes
                compact_payload = compact_token_boundary_payload_from_raw(raw_data)
                del raw_data

                candidate_path = run_dir / prepared.local_path.name
                save_token_boundary_source_shard(compact_payload, candidate_path)
                candidate_identity = RegularFileIdentity.capture(candidate_path)
                reloaded = load_token_boundary_source_shard_stable(
                    candidate_path,
                    expected_identity=candidate_identity,
                )
                assert_token_boundary_source_tensor_equal(compact_payload, reloaded)
                del reloaded

                remote_path = token_boundary_source_shard_path(
                    compact_prefix,
                    prepared.index,
                )
                published_compact.append(
                    upload_verified_object(
                        resolved_backend,
                        local_path=candidate_path,
                        remote_path=remote_path,
                        timeout_seconds=config.retry_timeout_seconds,
                        retry_interval_seconds=config.retry_interval_seconds,
                    )
                )
                candidate_path.unlink()
                del compact_payload
                logger.info(
                    f"Published compact token-boundary source shard "
                    f"{prepared.index + 1}/{len(raw_transport.inventory)}"
                )

            compact_config = replace(
                config,
                remote_prefix=compact_prefix,
                expected_shard_count=len(raw_transport.inventory),
            )
            compact_transport = RemoteTokenBoundaryTransport.snapshot(
                compact_config,
                backend=resolved_backend,
            )
            if tuple(entry.info for entry in compact_transport.inventory) != tuple(
                published_compact
            ):
                raise RuntimeError(
                    "compact token-boundary inventory differs from the exact "
                    "validated uploads; refusing manifest commit"
                )
            raw_transport.final_inventory_fence()
            compact_transport.final_inventory_fence()
            manifest = TokenBoundarySourceManifest(
                base_shard_content_hash=base_hasher.hexdigest(),
                raw_inventory_sha256=raw_transport.inventory_sha256,
                schema_version=1,
                shard_count=len(raw_transport.inventory),
                source_inventory_sha256=compact_transport.inventory_sha256,
            )
            local_manifest = run_dir / "manifest.json"
            atomic_write_bytes(
                serialize_token_boundary_source_manifest(manifest),
                local_manifest,
            )
            upload_verified_object(
                resolved_backend,
                local_path=local_manifest,
                remote_path=manifest_path,
                timeout_seconds=config.retry_timeout_seconds,
                retry_interval_seconds=config.retry_interval_seconds,
            )
            local_manifest.unlink()

        authenticated = AuthenticatedTokenBoundarySource.snapshot(
            config,
            backend=resolved_backend,
        )
        authenticated.final_inventory_fence()
        logger.info(f"Committed compact token-boundary source manifest last: {manifest_path}")
        return 0
    finally:
        primary_error = sys.exception()
        try:
            _remove_run_directory(run_dir)
        except FileNotFoundError:
            pass
        except BaseException as cleanup_error:
            if primary_error is not None:
                primary_error.add_note(f"publisher staging cleanup also failed: {cleanup_error!r}")
            else:
                raise


@hydra.main(
    config_path="../../configs/extraction",
    config_name="publish_token_boundary_source",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
