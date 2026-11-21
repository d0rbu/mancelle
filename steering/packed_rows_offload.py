"""Verified publish-then-evict offload for packed-row output shards.

A production packed-row family is far larger than the filesystem that produces
it: the c2048 L22 run writes ~1.5 GiB per output shard over ~1600 shards
(~2.4 TB) against a 1 TB pool quota, so a producer that keeps its own output
exhausts the quota long before it finishes. This module publishes each
completed packed body and its provenance sidecar to the bucket, verifies the
published object's size and Xet identity, records a durable receipt, and only
then removes the local bodies -- the same publish-verify-evict order
``steering/percentile_remote.py`` uses for its staged outputs.

The receipt is what makes that resumable. It holds the FP-target attestation
captured while the body was still local, so an evicted shard stays
resume-complete and still certifies truthfully; the producer reads receipts
exactly where it reads local payloads.

Republication is the recovery path: ``upload_verified_object`` is idempotent for
identical bytes and refuses a same-path collision whose bytes differ, so a
shard whose eviction was interrupted is re-offered to the bucket and proven
identical before it is finally removed.

The published layout mirrors the local one under a caller-supplied root
(``<root>/<packed dir relative to the feature root>/shard_NNNNN.safetensors.zst``
plus ``.../provenance/shard_NNNNN.safetensors.zst``), which is the layout
``steering/remote_preselection.py`` already materializes packed bodies from.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from steering.conditional.encoder_decoder.fp_target_source import (
    FP_TARGET_SOURCE_INTEGRITY_KEY,
    PackedShardAttestation,
    attest_packed_shard,
    validate_fp_target_source_certificate_remote,
)
from steering.gradient_overlay import PROVENANCE_DIRNAME, provenance_dir
from steering.overlay_io import (
    MANIFEST_FILENAME,
    BaseShardContentHasher,
    overlay_shard_path,
    read_manifest,
)
from steering.packed_rows_remote import RemotePackedRowsError
from steering.refined_types import PositiveInt
from steering.remote_feature_inventory import RemoteShardFamily
from steering.remote_training import (
    BucketBackend,
    _fold_file_digests,
    _join_remote,
    _packed_file_fingerprint,
    _packed_file_signature,
    packed_body_digest,
    packed_inventory_entry,
    packed_provenance_total_segments,
    sign_remote_packed_inventory,
    upload_verified_object,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.token_boundary_remote import (
    RemoteTokenBoundaryTransport,
    TokenBoundaryRemoteConfig,
)

OFFLOAD_RECEIPT_DIRNAME = "_offload"
OFFLOAD_RECEIPT_SCHEMA_VERSION = 1


def read_offload_receipts(
    output_dir: Path,
    *,
    identity_digest: str,
) -> dict[int, PackedShardAttestation]:
    """Read the published shards' attestations, keyed by shard index.

    Each ``_offload/shard_NNNNN.json`` receipt carries the FP-target scan taken
    of that shard's packed body just before it was evicted, which is both the
    proof the body was published and the certificate input its bytes can no
    longer supply. Entries that do not match the receipt filename shape are
    ignored (``atomic_json_dump`` stages debris beside its target); anything
    that matches but does not parse fails loud.
    """
    directory = Path(output_dir) / OFFLOAD_RECEIPT_DIRNAME
    if not directory.is_dir():
        return {}
    receipts: dict[int, PackedShardAttestation] = {}
    for path in sorted(directory.glob("shard_[0-9][0-9][0-9][0-9][0-9].json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if (
                raw["schema_version"] != OFFLOAD_RECEIPT_SCHEMA_VERSION
                or raw["identity_digest"] != identity_digest
            ):
                raise RemotePackedRowsError(
                    f"packed-row offload receipt {path} was written for a different schema "
                    f"or source/packing identity: schema={raw['schema_version']!r}, "
                    f"identity={raw['identity_digest']!r}"
                )
            attestation = PackedShardAttestation(**raw["attestation"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise RemotePackedRowsError(f"packed-row offload receipt is unusable: {path}") from exc
        expected_name = overlay_shard_path(directory, path.stem).name
        if attestation.name != expected_name:
            raise RemotePackedRowsError(
                f"packed-row offload receipt {path} attests {attestation.name!r}, "
                f"but its own name identifies {expected_name!r}"
            )
        receipts[int(path.stem.removeprefix("shard_"))] = attestation
    return receipts


def _layout_segment_count(layout: Mapping[str, object]) -> int | None:
    """Segments a packed body's own layout accounts for, or None if unstated."""
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        return None
    for tensor in tensors:
        if isinstance(tensor, Mapping) and tensor.get("name") == "seg_token_start":
            numel = tensor.get("numel")
            return numel if type(numel) is int else None
    return None


def build_remote_packed_inventory_from_bucket(
    packed_row_dir: Path,
    *,
    backend: BucketBackend,
    bucket_id: HfBucketId,
    packed_prefix: NormalizedRemotePrefix,
    staging_dir: Path,
    retry_timeout_seconds: float,
    retry_interval_seconds: float,
) -> dict[str, object]:
    """Sign the inventory of a family whose bodies live only in the bucket.

    ``build_remote_packed_inventory`` opens every packed and provenance body at
    once, which an offloaded family cannot offer -- and materializing one just
    to sign it would re-create the disk problem offload exists to avoid.

    This streams each published body exactly once through the same verified
    one-lookahead transport the producer reads its inputs with, so the download
    of the next body overlaps the hashing of the current one and the pass costs
    bandwidth plus a two-body lookahead per stream rather than the whole family
    (the transport's own bounded admission enforces that). Each body
    is read ONCE: ``packed_body_digest`` folds its size, content digest,
    name-salted fingerprint, and the running family hash in a single pass, and
    both builders share that helper, so a family signed from the bucket is the
    same inventory as the same family signed on disk.
    """
    packed_dir = Path(packed_row_dir)
    manifest_path = packed_dir / MANIFEST_FILENAME
    manifest = read_manifest(packed_dir)
    if manifest is None:
        raise RemotePackedRowsError(f"packed-row manifest is missing or invalid: {manifest_path}")
    metadata = manifest.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    # The recorded family size is what seals the published bodies: without it a
    # partially published family would sign as if it were whole.
    expected_shards = metadata.get("provenance_shards")
    if type(expected_shards) is not int or expected_shards <= 0:
        raise RemotePackedRowsError(
            f"packed-row manifest {manifest_path} has no positive provenance_shards count "
            f"to seal the published family against: {expected_shards!r}"
        )
    # A missing certificate is reported by the validator below, in the wording
    # operators already get from the local path.
    certificate = metadata.get(FP_TARGET_SOURCE_INTEGRITY_KEY, {})
    manifest_before = _packed_file_signature(
        manifest_path,
        surface="cannot capture packed-row manifest identity",
    )

    packed_config = TokenBoundaryRemoteConfig(
        bucket_id=bucket_id,
        remote_prefix=packed_prefix,
        staging_dir=staging_dir,
        expected_shard_count=expected_shards,
        retry_timeout_seconds=retry_timeout_seconds,
        retry_interval_seconds=retry_interval_seconds,
    )
    provenance_config = replace(
        packed_config,
        remote_prefix=NormalizedRemotePrefix.parse(
            _join_remote(str(packed_prefix), PROVENANCE_DIRNAME)
        ),
    )
    # The family grammar ignores nested objects, so each prefix seals exactly
    # its own bodies -- the provenance sidecars and verified-upload claims under
    # the packed prefix are invisible to it.
    packed_transport = RemoteTokenBoundaryTransport.snapshot(
        packed_config,
        backend=backend,
        family=RemoteShardFamily.OVERLAY_LEAF,
    )
    provenance_transport = RemoteTokenBoundaryTransport.snapshot(
        provenance_config,
        backend=backend,
        family=RemoteShardFamily.OVERLAY_LEAF,
    )

    base_hasher = BaseShardContentHasher()
    entries: list[dict[str, object]] = []
    provenance_digests: list[str] = []
    with packed_transport, provenance_transport:
        for packed, provenance in zip(
            packed_transport.stream(),
            provenance_transport.stream(),
            strict=True,
        ):
            # Tripwire for a future transport refactor: both families are sealed
            # to the same contiguous 0..N-1 domain and streamed in position
            # order, which is what makes these two the same shard.
            assert packed.index == provenance.index
            digest = packed_body_digest(packed.local_path, base_hasher=base_hasher)
            *_, layout = digest
            total_segments = packed_provenance_total_segments(provenance.local_path)
            # The local builder anchors the provenance sidecar against its packed
            # body through validate_packed_provenance. Nothing else here would,
            # so anchor it against the packed body's own layout -- already in
            # hand from the read above, and the same invariant the producer
            # enforces when it writes the pair.
            packed_segments = _layout_segment_count(layout)
            if packed_segments != total_segments:
                raise RemotePackedRowsError(
                    f"published provenance for {packed.local_path.name} covers "
                    f"{total_segments} segments, but its packed body holds {packed_segments}"
                )
            entries.append(
                packed_inventory_entry(
                    name=packed.local_path.name,
                    digest=digest,
                    total_segments=total_segments,
                )
            )
            provenance_digests.append(
                _packed_file_fingerprint(
                    provenance.local_path,
                    surface="cannot fingerprint packed provenance",
                )
            )
        packed_transport.final_inventory_fence()
        provenance_transport.final_inventory_fence()

    packed_fold = _fold_file_digests([str(entry["file_fingerprint"]) for entry in entries])
    # The certificate was folded from attestations taken before eviction, so
    # validating it against the freshly streamed inventory re-proves the
    # published bytes against what was certified. (The SAE width it carries is
    # the producer's, pinned there by _validate_packed_manifest; this call
    # re-checks the bodies, not the width.)
    validate_fp_target_source_certificate_remote(
        packed_dir,
        # Same width source the local builder passes: the certificate's own,
        # which `_validate_packed_manifest` already pinned to the configured
        # SAE width when the producer wrote it.
        n_sae_features=cast(int, certificate.get("n_sae_features")),
        inventory={
            "shards": [
                {"name": entry["name"], "byte_length": entry["byte_length"]} for entry in entries
            ],
            "packed_row_fingerprint": packed_fold,
        },
    )
    if (
        _packed_file_signature(
            manifest_path,
            surface="cannot re-check packed-row manifest identity",
        )
        != manifest_before
    ):
        raise RemotePackedRowsError(
            f"packed-row manifest changed while signing {packed_dir} from the bucket"
        )
    return sign_remote_packed_inventory(
        packed_manifest_file_fingerprint=_packed_file_fingerprint(
            manifest_path,
            surface="cannot fingerprint packed-row manifest",
        ),
        shards=entries,
        packed_row_fingerprint=packed_fold,
        provenance_shard_count=len(entries),
        # Name-sorted per-file digests, matching `_fingerprint_files_quiescent`:
        # the stream yields the sealed contiguous family in index order, which
        # is name order for zero-padded shard names.
        provenance_fingerprint=_fold_file_digests(provenance_digests),
        base_shard_content_hash=base_hasher.hexdigest(),
        surface=f"new remote packed inventory for {packed_dir}",
    )


class PackedRowOffloader:
    """Publishes completed packed shards to the bucket and evicts them locally.

    One serialized upload worker: shards are submitted in index order and each
    task waits on its compression future, so publication follows shard order and
    a single in-flight upload never competes with itself. The bounded permit
    count is what actually caps local growth -- the producer blocks in
    ``submit`` once that many shards await publication, which is the
    backpressure that keeps the run inside its quota.
    """

    def __init__(
        self,
        *,
        backend: BucketBackend,
        remote_prefix_by_output_dir: Mapping[Path, NormalizedRemotePrefix],
        identity_digest: str,
        n_sae_features: PositiveInt,
        retry_timeout_seconds: float,
        retry_interval_seconds: float,
        max_pending_shards: int,
    ) -> None:
        if max_pending_shards < 1:
            raise ValueError(f"max_pending_shards must be at least 1, got {max_pending_shards}")
        self._backend = backend
        self._prefixes = {
            Path(output_dir).absolute(): NormalizedRemotePrefix.parse(str(prefix))
            for output_dir, prefix in remote_prefix_by_output_dir.items()
        }
        self._identity_digest = identity_digest
        self._n_sae_features = n_sae_features
        self._retry_timeout_seconds = retry_timeout_seconds
        self._retry_interval_seconds = retry_interval_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="packed-rows-offload",
        )
        self._permits = threading.BoundedSemaphore(max_pending_shards)
        self._futures: list[Future[int]] = []
        # Counter discipline instead of a lock: the single upload worker owns
        # published/evicted/upload_seconds, the producer owns backpressure, and
        # reconcile runs on the producer thread with no worker in flight -- so
        # no counter ever has two concurrent writers.
        self.published_shards = 0
        self.reconciled_shards = 0
        self.evicted_bytes = 0
        self.upload_seconds = 0.0
        self.backpressure_seconds = 0.0

    def __enter__(self) -> PackedRowOffloader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def receipts(self, output_dir: Path) -> dict[int, PackedShardAttestation]:
        """This run's published-shard attestations beside *output_dir*."""
        return read_offload_receipts(output_dir, identity_digest=self._identity_digest)

    def submit(self, output_dir: Path, shard_idx: int, compress_future: Future) -> None:
        """Queue one shard for publication once its compression finishes."""
        for future in self._futures:
            if future.done():
                future.result()  # surface a failed publication as soon as it is observable
        blocked_started = time.monotonic()
        self._permits.acquire()
        self.backpressure_seconds += time.monotonic() - blocked_started
        try:
            future = self._executor.submit(
                self._await_and_offload,
                Path(output_dir),
                shard_idx,
                compress_future,
            )
        except BaseException:
            self._permits.release()
            raise
        future.add_done_callback(lambda _future: self._permits.release())
        self._futures.append(future)

    def drain(self) -> None:
        """Block until every queued publication finished, re-raising failures."""
        pending, self._futures = self._futures, []
        for future in pending:
            future.result()

    def reconcile(self, output_dir: Path, *, upto: int) -> int:
        """Publish and evict shards ``0..upto-1`` that are still local.

        Runs inline before packing resumes: a killed window, a run that predates
        offload, or a finished retained family all leave complete bodies on
        disk, and reclaiming them first is what buys the headroom the rest of
        the run spends.
        """
        output_dir = Path(output_dir)
        reclaimed = 0
        shards = 0
        for shard_idx in range(upto):
            freed = self._offload_shard(output_dir, shard_idx)
            if freed:
                reclaimed += freed
                shards += 1
        self.reconciled_shards += shards
        if shards:
            logger.info(
                f"Offload: reclaimed {reclaimed:,} local byte(s) from {shards} retained "
                f"shard(s) under {output_dir}"
            )
        return reclaimed

    def _await_and_offload(
        self,
        output_dir: Path,
        shard_idx: int,
        compress_future: Future,
    ) -> int:
        # Re-raises a compression failure here as well as at the producer's own
        # join; the producer joins first, so the primary traceback is unchanged.
        compress_future.result()
        return self._offload_shard(output_dir, shard_idx)

    def _offload_shard(self, output_dir: Path, shard_idx: int) -> int:
        """Publish one shard if it is still local, evict it, return bytes freed."""
        stem = f"shard_{shard_idx:05d}"
        packed_path = overlay_shard_path(output_dir, stem)
        provenance_path = overlay_shard_path(provenance_dir(output_dir), stem)
        if packed_path.exists():
            self._publish(output_dir, stem, packed_path, provenance_path)
        elif not provenance_path.exists():
            return 0  # already published and evicted
        # The receipt is written before either unlink, so reaching here with a
        # body still present means its publication is already durable.
        evicted = 0
        for path in (packed_path, provenance_path):
            if path.exists():
                evicted += path.stat().st_size
                path.unlink()
        self.evicted_bytes += evicted
        return evicted

    def _publish(
        self,
        output_dir: Path,
        stem: str,
        packed_path: Path,
        provenance_path: Path,
    ) -> None:
        prefix = self._prefixes[output_dir.absolute()]
        # Certify the body while it is still local: an evicted family can never
        # be rescanned, and the whole-family scan would then certify an empty
        # directory as a complete, passing scan.
        rows, segments, feature_entries = attest_packed_shard(
            packed_path,
            n_sae_features=self._n_sae_features,
        )
        attestation = PackedShardAttestation(
            name=packed_path.name,
            byte_length=packed_path.stat().st_size,
            file_fingerprint=_packed_file_fingerprint(
                packed_path,
                surface=f"cannot fingerprint packed-row body {packed_path}",
            ),
            rows_checked=rows,
            segments_checked=segments,
            feature_entries_checked=feature_entries,
        )
        started = time.monotonic()
        for local_path, remote_path in (
            (packed_path, _join_remote(str(prefix), packed_path.name)),
            (provenance_path, _join_remote(str(prefix), PROVENANCE_DIRNAME, provenance_path.name)),
        ):
            # Verifies the published object's size and Xet identity, and refuses
            # a same-path collision whose bytes differ -- which is also what
            # proves a re-published body is the one already stored.
            upload_verified_object(
                self._backend,
                local_path=local_path,
                remote_path=remote_path,
                timeout_seconds=self._retry_timeout_seconds,
                retry_interval_seconds=self._retry_interval_seconds,
            )
        self.upload_seconds += time.monotonic() - started
        # Receipt BEFORE eviction: a kill in between is repaired by reconcile,
        # while the reverse order could evict the only copy of a shard nothing
        # durably records as published.
        atomic_json_dump(
            {
                "schema_version": OFFLOAD_RECEIPT_SCHEMA_VERSION,
                "identity_digest": self._identity_digest,
                "attestation": asdict(attestation),
            },
            output_dir / OFFLOAD_RECEIPT_DIRNAME / f"{stem}.json",
        )
        self.published_shards += 1
        logger.info(f"Offload: published {stem} to {prefix}")

    def profile_payload(self) -> dict[str, object]:
        return {
            "published_shards": self.published_shards,
            "reconciled_shards": self.reconciled_shards,
            "evicted_bytes": self.evicted_bytes,
            "upload_seconds": self.upload_seconds,
            "backpressure_seconds": self.backpressure_seconds,
        }


__all__ = [
    "OFFLOAD_RECEIPT_DIRNAME",
    "OFFLOAD_RECEIPT_SCHEMA_VERSION",
    "PackedRowOffloader",
    "build_remote_packed_inventory_from_bucket",
    "read_offload_receipts",
]
