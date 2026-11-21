"""Unit contracts for verified publish-then-evict packed-row offload."""

from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest
import torch as t

from steering.conditional.encoder_decoder.fp_target_source import attested_inventory
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.gradient_overlay import provenance_dir
from steering.packed_batch_io import PackedRowShard, save_packed_row_shard
from steering.packed_rows_offload import (
    OFFLOAD_RECEIPT_DIRNAME,
    OFFLOAD_RECEIPT_SCHEMA_VERSION,
    PackedRowOffloader,
    read_offload_receipts,
)
from steering.packed_rows_remote import RemotePackedRowsError
from steering.remote_training import RemoteTrainingError
from steering.remote_types import NormalizedRemotePrefix

IDENTITY_DIGEST = "d" * 64
PREFIX = NormalizedRemotePrefix.parse("packed-out/c32/sall")
N_SAE_FEATURES = 16_384


class _FakeBucket:
    """Filesystem-backed bucket with injectable upload faults and gating."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.uploads: list[str] = []
        self.upload_failures: dict[str, int] = {}
        self.upload_gate: threading.Event | None = None
        self.gated_paths: set[str] = set()

    def _path(self, remote_path: str) -> Path:
        path = self.root / remote_path
        assert path.resolve().is_relative_to(self.root.resolve())
        return path

    @staticmethod
    def _info(path: Path, remote_path: str) -> Any:
        from steering.remote_training import RemoteObjectInfo

        xet = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(remote_path, path.stat().st_size, str(xet.hash))

    def stat(self, remote_path: str) -> Any:
        path = self._path(remote_path)
        return self._info(path, remote_path) if path.is_file() else None

    def list(self, prefix: str) -> Sequence[Any]:
        base = self._path(prefix)
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.uploads.append(remote_path)
        if self.upload_gate is not None and remote_path in self.gated_paths:
            assert self.upload_gate.wait(timeout=30)
        failures = self.upload_failures.get(remote_path, 0)
        if failures:
            self.upload_failures[remote_path] = failures - 1
            raise OSError("injected transient upload failure")
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)


def _name(shard_idx: int) -> str:
    return f"shard_{shard_idx:05d}.safetensors.zst"


def _seed_shard(output_dir: Path, shard_idx: int, *, feature_ids: list[int] | None = None) -> None:
    """Write one shard's packed body and provenance sidecar."""
    ids = [3 + shard_idx, 7 + shard_idx] if feature_ids is None else feature_ids
    provenance_dir(output_dir).mkdir(parents=True, exist_ok=True)
    save_packed_row_shard(
        PackedRowShard(
            tokens=t.full((1, 5), shard_idx + 1, dtype=t.uint16),
            row_seg_offsets=t.tensor([0, 1], dtype=t.int32),
            seg_token_start=t.tensor([0], dtype=t.uint16),
            seg_token_end=t.tensor([4], dtype=t.uint16),
            seg_doc_id=t.tensor([0], dtype=t.uint8),
            seg_feat_offsets=t.tensor([0, len(ids)], dtype=t.int32),
            feature_ids=t.tensor(ids, dtype=t.uint16),
            feature_pcts=t.tensor([10] * len(ids), dtype=t.uint8),
        ),
        output_dir / _name(shard_idx),
        zstd_level=1,
    )
    (provenance_dir(output_dir) / _name(shard_idx)).write_bytes(
        f"provenance-{shard_idx}".encode() * 8
    )


def _offloader(
    bucket: _FakeBucket,
    output_dir: Path,
    *,
    max_pending_shards: int = 4,
) -> PackedRowOffloader:
    return PackedRowOffloader(
        backend=cast(Any, bucket),
        remote_prefix_by_output_dir={output_dir: PREFIX},
        identity_digest=IDENTITY_DIGEST,
        n_sae_features=cast(Any, N_SAE_FEATURES),
        retry_timeout_seconds=5.0,
        retry_interval_seconds=0.01,
        max_pending_shards=max_pending_shards,
    )


def _resolved_future() -> Future:
    future: Future = Future()
    future.set_result((0, 0, 0, None))
    return future


def _receipt_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / OFFLOAD_RECEIPT_DIRNAME / f"shard_{shard_idx:05d}.json"


def _fixture(tmp_path: Path, n_shards: int = 1) -> tuple[Path, Path, _FakeBucket]:
    output_dir = tmp_path / "out"
    bucket_root = tmp_path / "bucket"
    bucket_root.mkdir()
    for shard_idx in range(n_shards):
        _seed_shard(output_dir, shard_idx)
    return output_dir, bucket_root, _FakeBucket(bucket_root)


# --------------------------------------------------------------------------
# Publication, verification, eviction
# --------------------------------------------------------------------------


def test_publish_verifies_then_evicts_and_receipts_the_attestation(tmp_path: Path) -> None:
    output_dir, bucket_root, bucket = _fixture(tmp_path)
    packed_bytes = (output_dir / _name(0)).read_bytes()
    provenance_bytes = (provenance_dir(output_dir) / _name(0)).read_bytes()

    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        offloader.drain()
        assert offloader.published_shards == 1
        assert offloader.evicted_bytes == len(packed_bytes) + len(provenance_bytes)

    # Published bytes are the produced bytes; the local bodies are gone.
    assert (bucket_root / PREFIX / _name(0)).read_bytes() == packed_bytes
    assert (bucket_root / PREFIX / "provenance" / _name(0)).read_bytes() == provenance_bytes
    assert not (output_dir / _name(0)).exists()
    assert not (provenance_dir(output_dir) / _name(0)).exists()

    receipts = read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)
    assert set(receipts) == {0}
    assert receipts[0].name == _name(0)
    assert receipts[0].byte_length == len(packed_bytes)
    assert (receipts[0].rows_checked, receipts[0].segments_checked) == (1, 1)
    assert receipts[0].feature_entries_checked == 2


def test_a_corrupted_upload_is_caught_before_the_local_body_is_evicted(
    tmp_path: Path,
) -> None:
    """The load-bearing guarantee: verify what landed, THEN delete the original."""
    output_dir, bucket_root, _bucket = _fixture(tmp_path)

    class _TruncatingBucket(_FakeBucket):
        def upload(self, local_path: Path, remote_path: str) -> None:
            super().upload(local_path, remote_path)
            target = self._path(remote_path)
            target.write_bytes(target.read_bytes()[:-1])

    bucket = _TruncatingBucket(bucket_root)
    packed_bytes = (output_dir / _name(0)).read_bytes()

    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        with pytest.raises(RemoteTrainingError, match="failed size/Xet verification"):
            offloader.drain()

    assert (output_dir / _name(0)).read_bytes() == packed_bytes
    assert (provenance_dir(output_dir) / _name(0)).is_file()
    assert read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST) == {}


def test_publication_failure_leaves_the_local_bodies_in_place(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    # Outlast the retry budget so publication fails terminally.
    bucket.upload_failures[f"{PREFIX}/{_name(0)}"] = 10_000

    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        with pytest.raises(RemoteTrainingError, match="timed out after .* uploading and verifying"):
            offloader.drain()

    assert (output_dir / _name(0)).is_file()
    assert (provenance_dir(output_dir) / _name(0)).is_file()
    assert read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST) == {}


def test_an_out_of_bounds_feature_id_blocks_publication(tmp_path: Path) -> None:
    """The FP-target scan must still run — publication is its last chance."""
    output_dir, _bucket_root, bucket = _fixture(tmp_path, n_shards=0)
    # uint16 ids above the SAE width are exactly what the certificate rules out.
    _seed_shard(output_dir, 0, feature_ids=[3, N_SAE_FEATURES + 5])

    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        with pytest.raises(Exception, match=r"feature ID 16389 is outside \[0, 16384\)"):
            offloader.drain()

    assert bucket.uploads == []
    assert (output_dir / _name(0)).is_file()
    assert read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST) == {}


def test_compression_failure_propagates_without_publishing(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    failed: Future = Future()
    failed.set_exception(RuntimeError("compression exploded"))

    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, failed)
        with pytest.raises(RuntimeError, match="compression exploded"):
            offloader.drain()
    assert bucket.uploads == []


# --------------------------------------------------------------------------
# Reconcile and interrupted evictions
# --------------------------------------------------------------------------


def test_reconcile_publishes_bodies_left_by_a_pre_offload_run(tmp_path: Path) -> None:
    output_dir, bucket_root, bucket = _fixture(tmp_path, n_shards=3)
    expected = {idx: (output_dir / _name(idx)).read_bytes() for idx in range(3)}
    local_bytes = sum(
        path.stat().st_size
        for idx in range(3)
        for path in (output_dir / _name(idx), provenance_dir(output_dir) / _name(idx))
    )

    with _offloader(bucket, output_dir) as offloader:
        assert offloader.reconcile(output_dir, upto=3) == local_bytes
        payload = offloader.profile_payload()

    assert payload["reconciled_shards"] == 3
    assert payload["published_shards"] == 3
    assert payload["evicted_bytes"] == local_bytes
    assert set(read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)) == {0, 1, 2}
    for shard_idx, body in expected.items():
        assert (bucket_root / PREFIX / _name(shard_idx)).read_bytes() == body
        assert not (output_dir / _name(shard_idx)).exists()


def test_reconcile_is_a_no_op_when_nothing_is_local(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path, n_shards=0)
    output_dir.mkdir(parents=True)
    with _offloader(bucket, output_dir) as offloader:
        assert offloader.reconcile(output_dir, upto=5) == 0
    assert bucket.uploads == []


@pytest.mark.parametrize("evict_provenance", [False, True])
def test_reconcile_finishes_an_eviction_interrupted_after_its_receipt(
    tmp_path: Path,
    evict_provenance: bool,
) -> None:
    output_dir, bucket_root, bucket = _fixture(tmp_path)
    packed_bytes = (output_dir / _name(0)).read_bytes()
    provenance_bytes = (provenance_dir(output_dir) / _name(0)).read_bytes()

    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        offloader.drain()
    receipt_before = _receipt_path(output_dir, 0).read_bytes()

    # Re-create exactly what a kill between the receipt write and the unlinks
    # leaves behind: both bodies, or only the trailing provenance one.
    if not evict_provenance:
        (output_dir / _name(0)).write_bytes(packed_bytes)
    (provenance_dir(output_dir) / _name(0)).write_bytes(provenance_bytes)

    with _offloader(bucket, output_dir) as offloader:
        assert offloader.reconcile(output_dir, upto=1) > 0

    assert not (output_dir / _name(0)).exists()
    assert not (provenance_dir(output_dir) / _name(0)).exists()
    assert _receipt_path(output_dir, 0).read_bytes() == receipt_before
    assert (bucket_root / PREFIX / _name(0)).read_bytes() == packed_bytes


def test_a_body_that_differs_from_the_published_one_is_never_evicted(tmp_path: Path) -> None:
    """Re-publication is the recovery path, so the bucket refuses the mismatch."""
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    with _offloader(bucket, output_dir) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        offloader.drain()

    _seed_shard(output_dir, 0, feature_ids=[11, 13, 17])
    with _offloader(bucket, output_dir) as offloader:
        with pytest.raises(RemoteTrainingError, match="same-path remote object collision"):
            offloader.reconcile(output_dir, upto=1)
    assert (output_dir / _name(0)).is_file()


def test_submit_blocks_once_the_pending_bound_is_reached(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path, n_shards=3)
    gate = threading.Event()
    bucket.upload_gate = gate
    bucket.gated_paths = {f"{PREFIX}/{_name(0)}"}

    with _offloader(bucket, output_dir, max_pending_shards=1) as offloader:
        offloader.submit(output_dir, 0, _resolved_future())
        blocked = threading.Event()

        def _submit_second() -> None:
            offloader.submit(output_dir, 1, _resolved_future())
            blocked.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(_submit_second)
            # The bound is one shard, so the second submit cannot proceed while
            # the first upload is gated — that block is the disk bound.
            assert not blocked.wait(timeout=0.5)
            gate.set()
            pending.result(timeout=30)
        offloader.drain()

    assert set(read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)) == {0, 1}


@pytest.mark.parametrize("max_pending_shards", [0, -1])
def test_offloader_requires_a_positive_pending_bound(
    tmp_path: Path,
    max_pending_shards: int,
) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path, n_shards=0)
    with pytest.raises(ValueError, match="max_pending_shards must be at least 1"):
        _offloader(bucket, output_dir, max_pending_shards=max_pending_shards)


# --------------------------------------------------------------------------
# Receipts
# --------------------------------------------------------------------------


def test_receipt_derived_inventory_equals_the_local_family_inventory(tmp_path: Path) -> None:
    """The evicted family certifies to what a local family would."""
    output_dir, _bucket_root, bucket = _fixture(tmp_path, n_shards=3)
    local = {
        "shards": [
            {"name": path.name, "byte_length": path.stat().st_size}
            for path in sorted(output_dir.glob("*.safetensors.zst"))
        ],
        "packed_row_fingerprint": packed_row_fingerprint(output_dir),
    }

    with _offloader(bucket, output_dir) as offloader:
        offloader.reconcile(output_dir, upto=3)

    receipts = read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)
    assert attested_inventory(list(receipts.values())) == local


def test_read_offload_receipts_ignores_debris_and_a_missing_namespace(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    assert read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST) == {}

    with _offloader(bucket, output_dir) as offloader:
        offloader.reconcile(output_dir, upto=1)
    directory = output_dir / OFFLOAD_RECEIPT_DIRNAME
    (directory / "shard_00000.json.tmp.4242.beef0a").write_text("torn")
    (directory / "notes.txt").write_text("ignored")
    assert set(read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)) == {0}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw | {"schema_version": 2}, "different schema"),
        (lambda raw: raw | {"identity_digest": "e" * 64}, "different schema"),
        (lambda raw: raw | {"attestation": {"name": "x"}}, "unusable"),
        (lambda raw: {k: v for k, v in raw.items() if k != "attestation"}, "unusable"),
        (
            lambda raw: raw | {"attestation": raw["attestation"] | {"name": "shard_00009.zst"}},
            "attests 'shard_00009.zst'",
        ),
    ],
)
def test_read_offload_receipts_rejects_unusable_receipts(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    with _offloader(bucket, output_dir) as offloader:
        offloader.reconcile(output_dir, upto=1)

    path = _receipt_path(output_dir, 0)
    path.write_text(json.dumps(mutate(json.loads(path.read_text()))))
    with pytest.raises(RemotePackedRowsError, match=message):
        read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)


def test_read_offload_receipts_rejects_unparseable_json(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    with _offloader(bucket, output_dir) as offloader:
        offloader.reconcile(output_dir, upto=1)
    _receipt_path(output_dir, 0).write_text("{truncated")
    with pytest.raises(RemotePackedRowsError, match="unusable"):
        read_offload_receipts(output_dir, identity_digest=IDENTITY_DIGEST)


def test_receipt_records_the_current_schema_version(tmp_path: Path) -> None:
    output_dir, _bucket_root, bucket = _fixture(tmp_path)
    with _offloader(bucket, output_dir) as offloader:
        offloader.reconcile(output_dir, upto=1)
    stored = json.loads(_receipt_path(output_dir, 0).read_text())
    assert stored["schema_version"] == OFFLOAD_RECEIPT_SCHEMA_VERSION
    assert stored["identity_digest"] == IDENTITY_DIGEST
