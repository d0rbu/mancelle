"""End-to-end contracts for offloaded (publish-then-evict) packed-row runs.

The load-bearing property is that offload changes only WHERE the produced
bodies live: every published body must be byte-identical to the same run's
local product, and an offloaded prefix must resume and complete exactly like a
retained one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

import scripts.preprocess.precompute_packed_rows as ppr
import scripts.preprocess.publish_remote_packed_inventory as publisher
from errors import PackedRowInvariantError
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import glob_overlay_shards
from steering.packed_rows_offload import (
    OFFLOAD_RECEIPT_DIRNAME,
    build_remote_packed_inventory_from_bucket,
    read_offload_receipts,
)
from steering.packed_rows_remote import RemotePackedRowsError
from steering.remote_feature_inventory import resolve_remote_feature_prefix
from steering.remote_training import (
    REMOTE_PACKED_INVENTORY_FILENAME,
    build_remote_packed_inventory,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from tests.scripts.test_packed_rows_remote import (
    _cfg,
    _remote_output_dir,
    _seeded_remote_run,
)

OFFLOAD_ROOT = "packed-out"


def _offload_cfg(remote_env: dict, *, staging: Path, root: str | None = OFFLOAD_ROOT):
    cfg = _cfg(remote_env, staging=staging, enabled=True)
    cfg.remote.offload.enabled = True
    cfg.remote.offload.remote_packed_root = root
    return cfg


def _split_dirs(feature_dir: Path, env: dict, max_shards: int | None) -> tuple[Path, Path]:
    output_dir = _remote_output_dir(feature_dir, env, max_shards)
    return output_dir, output_dir.parent / f"val_{output_dir.name}"


def _published_prefix(feature_dir: Path, output_dir: Path) -> str:
    return str(
        resolve_remote_feature_prefix(
            output_dir,
            local_features_root=feature_dir,
            remote_features_root=NormalizedRemotePrefix.parse(OFFLOAD_ROOT),
            surface="packed-row output directory",
        )
    )


def _assert_published_matches_reference(
    *,
    bucket_root: Path,
    local_feature_dir: Path,
    remote_feature_dir: Path,
    env: dict,
    max_shards: int | None,
) -> int:
    """Every reference body is published verbatim; nothing is retained locally."""
    published = 0
    for reference_dir, remote_dir in zip(
        _split_dirs(local_feature_dir, env, max_shards),
        _split_dirs(remote_feature_dir, env, max_shards),
        strict=True,
    ):
        prefix = _published_prefix(remote_feature_dir, remote_dir)
        reference_bodies = glob_overlay_shards(reference_dir)
        reference_provenance = glob_overlay_shards(provenance_dir(reference_dir))
        assert reference_bodies and len(reference_bodies) == len(reference_provenance)
        receipts = read_offload_receipts(
            remote_dir,
            identity_digest=_identity_digest_of(receipts_dir=remote_dir / OFFLOAD_RECEIPT_DIRNAME),
        )
        assert sorted(receipts) == list(range(len(reference_bodies)))

        for shard_idx, (body, provenance) in enumerate(
            zip(reference_bodies, reference_provenance, strict=True)
        ):
            expected_body = body.read_bytes()
            expected_provenance = provenance.read_bytes()
            assert (bucket_root / prefix / body.name).read_bytes() == expected_body
            assert (
                bucket_root / prefix / "provenance" / provenance.name
            ).read_bytes() == expected_provenance

            attestation = receipts[shard_idx]
            assert attestation.name == body.name
            assert attestation.byte_length == len(expected_body)
            published += 1

        # Bodies were evicted, while the small local authorities survive.
        assert glob_overlay_shards(remote_dir) == []
        assert glob_overlay_shards(provenance_dir(remote_dir)) == []
        assert (remote_dir / "manifest.json").read_bytes() == (
            reference_dir / "manifest.json"
        ).read_bytes()
        assert sorted(path.name for path in (remote_dir / "_dims").iterdir()) == sorted(
            path.name for path in (reference_dir / "_dims").iterdir()
        )
        for companion in (reference_dir / "_dims").iterdir():
            assert (remote_dir / "_dims" / companion.name).read_bytes() == companion.read_bytes()
    return published


def _identity_digest_of(*, receipts_dir: Path) -> str:
    """Read back the identity digest the run stamped into its own receipts."""
    receipts = sorted(receipts_dir.glob("shard_*.json"))
    assert receipts, f"no offload receipts under {receipts_dir}"
    digests = {json.loads(path.read_text())["identity_digest"] for path in receipts}
    assert len(digests) == 1
    return digests.pop()


def test_offloaded_run_publishes_byte_identical_bodies_and_evicts_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, bucket_root, bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    cfg = _offload_cfg(remote_env, staging=tmp_path / "staging")
    cfg.profile = True
    profile_path = tmp_path / "profile.json"
    cfg.profile_summary_output = str(profile_path)

    assert ppr.run(cfg) == 0

    published = _assert_published_matches_reference(
        bucket_root=bucket_root,
        local_feature_dir=local_feature_dir,
        remote_feature_dir=remote_feature_dir,
        env=env,
        max_shards=None,
    )
    assert published > 0
    assert bucket.uploads

    # Mirrors are deliberately not published: they would only dangle.
    tok_dir = remote_feature_dir / "tok" / env["tok_checksum"][:16]
    mirror_dir = tok_dir / "pct" / env["pct_checksum"][:16] / "c32"
    assert not mirror_dir.exists()

    profile = json.loads(profile_path.read_text())
    offload = profile["remote_transport"]["offload"]
    assert offload["published_shards"] == published
    assert offload["reconciled_shards"] == 0
    assert offload["evicted_bytes"] > 0
    assert offload["upload_seconds"] >= 0.0


def test_offloaded_manifest_counts_the_published_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, _bucket_root, _bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging")) == 0

    for reference_dir, remote_dir in zip(
        _split_dirs(local_feature_dir, env, None),
        _split_dirs(remote_feature_dir, env, None),
        strict=True,
    ):
        metadata = json.loads((remote_dir / "manifest.json").read_text())["metadata"]
        assert metadata["provenance_shards"] == len(glob_overlay_shards(reference_dir))
        assert metadata == json.loads((reference_dir / "manifest.json").read_text())["metadata"]


def test_offloaded_run_reauthenticates_a_completed_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, bucket_root, bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    cfg = _offload_cfg(remote_env, staging=tmp_path / "staging-a")
    assert ppr.run(cfg) == 0
    uploads_after_first_run = list(bucket.uploads)

    # resume=true over a complete, fully evicted family re-validates it from
    # the manifests plus the receipts, without any body coming back.
    rerun = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    rerun.resume = True
    assert ppr.run(rerun) == 0
    assert bucket.uploads == uploads_after_first_run
    assert (
        _assert_published_matches_reference(
            bucket_root=bucket_root,
            local_feature_dir=local_feature_dir,
            remote_feature_dir=remote_feature_dir,
            env=env,
            max_shards=None,
        )
        > 0
    )


def test_interrupted_offloaded_run_resumes_to_the_same_published_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, bucket_root, _bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=4, compress_workers=1)
    )
    cfg = _offload_cfg(remote_env, staging=tmp_path / "staging-a")
    real_write = ppr._write_packing_checkpoint
    interrupted = False

    def _interrupt_after_second_train_checkpoint(context, **kwargs):
        nonlocal interrupted
        real_write(context, **kwargs)
        if context.split == "train" and kwargs["shard_idx"] == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("injected post-checkpoint interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_after_second_train_checkpoint)
    with pytest.raises(RuntimeError, match="post-checkpoint interruption"):
        ppr.run(cfg)

    output_dir, _val_dir = _split_dirs(remote_feature_dir, remote_env, None)
    receipts = read_offload_receipts(
        output_dir,
        identity_digest=_identity_digest_of(receipts_dir=output_dir / OFFLOAD_RECEIPT_DIRNAME),
    )
    # The interrupted window published what it had finished; a shard whose
    # publication the kill cancelled stays local and is resume-complete either
    # way, so the kept prefix covers both.
    assert receipts
    kept = ppr._contiguous_complete_shard_count(output_dir, offloaded=frozenset(receipts))
    assert kept >= len(receipts)

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", real_write)
    resumed = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    resumed.resume = True
    assert ppr.run(resumed) == 0

    assert (
        _assert_published_matches_reference(
            bucket_root=bucket_root,
            local_feature_dir=local_feature_dir,
            remote_feature_dir=remote_feature_dir,
            env=env,
            max_shards=None,
        )
        > 0
    )


def test_enabling_offload_reclaims_bodies_a_retained_run_left_behind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator's upgrade path: resume an in-progress non-offload run."""
    env, _local_env, remote_env, bucket_root, _bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=4, compress_workers=1)
    )
    retained = _cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)
    real_write = ppr._write_packing_checkpoint
    interrupted = False

    def _interrupt_after_second_train_checkpoint(context, **kwargs):
        nonlocal interrupted
        real_write(context, **kwargs)
        if context.split == "train" and kwargs["shard_idx"] == 2 and not interrupted:
            interrupted = True
            raise RuntimeError("injected post-checkpoint interruption")

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", _interrupt_after_second_train_checkpoint)
    with pytest.raises(RuntimeError, match="post-checkpoint interruption"):
        ppr.run(retained)

    output_dir, _val_dir = _split_dirs(remote_feature_dir, remote_env, None)
    retained_bodies = glob_overlay_shards(output_dir)
    assert retained_bodies, "the interrupted retained run should leave local bodies"
    state_path, _lock = ppr._remote_state_paths(output_dir)
    state_before = state_path.read_bytes()

    monkeypatch.setattr(ppr, "_write_packing_checkpoint", real_write)
    resumed = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    resumed.resume = True
    resumed.profile = True
    profile_path = tmp_path / "profile.json"
    resumed.profile_summary_output = str(profile_path)
    # Enabling offload must not invalidate the in-progress resume state.
    assert state_path.read_bytes() == state_before
    assert ppr.run(resumed) == 0

    profile = json.loads(profile_path.read_text())
    assert profile["remote_transport"]["offload"]["reconciled_shards"] == len(retained_bodies)
    assert (
        _assert_published_matches_reference(
            bucket_root=bucket_root,
            local_feature_dir=local_feature_dir,
            remote_feature_dir=remote_feature_dir,
            env=env,
            max_shards=None,
        )
        > 0
    )


def test_offload_reclaims_a_family_a_retained_run_already_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offloading finished artifacts: the certificate written by the local scan
    must still validate against the receipts that replaced its bodies."""
    env, _local_env, remote_env, bucket_root, _bucket, local_feature_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)) == 0
    output_dir, val_dir = _split_dirs(remote_feature_dir, env, None)
    assert glob_overlay_shards(output_dir)
    certificates_before = {
        directory: json.loads((directory / "manifest.json").read_text())["metadata"][
            "fp_target_source_integrity"
        ]
        for directory in (output_dir, val_dir)
    }

    offloaded = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    offloaded.resume = True
    assert ppr.run(offloaded) == 0

    # The bodies moved to the bucket and the untouched certificate still holds.
    for directory in (output_dir, val_dir):
        assert glob_overlay_shards(directory) == []
        assert (
            json.loads((directory / "manifest.json").read_text())["metadata"][
                "fp_target_source_integrity"
            ]
            == certificates_before[directory]
        )
    assert (
        _assert_published_matches_reference(
            bucket_root=bucket_root,
            local_feature_dir=local_feature_dir,
            remote_feature_dir=remote_feature_dir,
            env=env,
            max_shards=None,
        )
        > 0
    )


def test_streamed_inventory_equals_the_local_builder_on_the_same_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1463: signing from the bucket must reproduce the local signature.

    The same family is signed twice — once by the whole-family local builder
    while its bodies are still on disk, then again by the streaming builder
    after offload evicted them. Same artifact, two paths, one inventory.
    """
    env, _local_env, remote_env, bucket_root, bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_cfg(remote_env, staging=tmp_path / "staging-a", enabled=True)) == 0
    output_dir, val_dir = _split_dirs(remote_feature_dir, env, None)
    local_inventories = {
        directory: build_remote_packed_inventory(directory) for directory in (output_dir, val_dir)
    }
    assert all(inventory["shards"] for inventory in local_inventories.values())

    offloaded = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    offloaded.resume = True
    assert ppr.run(offloaded) == 0
    assert glob_overlay_shards(output_dir) == []

    for directory, expected in local_inventories.items():
        streamed = build_remote_packed_inventory_from_bucket(
            directory,
            backend=bucket,
            bucket_id=HfBucketId.parse("owner/test-bucket"),
            packed_prefix=NormalizedRemotePrefix.parse(
                _published_prefix(remote_feature_dir, directory)
            ),
            staging_dir=tmp_path / "inventory-staging",
            retry_timeout_seconds=60.0,
            retry_interval_seconds=0.01,
        )
        assert streamed == expected
    assert bucket_root.is_dir()


def _stream_inventory(
    remote_feature_dir: Path,
    directory: Path,
    bucket,
    staging: Path,
) -> dict:
    return build_remote_packed_inventory_from_bucket(
        directory,
        backend=bucket,
        bucket_id=HfBucketId.parse("owner/test-bucket"),
        packed_prefix=NormalizedRemotePrefix.parse(
            _published_prefix(remote_feature_dir, directory)
        ),
        staging_dir=staging,
        retry_timeout_seconds=60.0,
        retry_interval_seconds=0.01,
    )


def test_streamed_inventory_validates_an_attestation_folded_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production shape: a family produced FRESH with offload on.

    Its certificate is folded from per-shard attestations rather than written by
    the whole-family local scan, and the streaming builder validates the
    inventory against exactly that certificate.
    """
    env, _local_env, remote_env, _bucket_root, bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging")) == 0
    output_dir, val_dir = _split_dirs(remote_feature_dir, env, None)

    for directory in (output_dir, val_dir):
        certificate = json.loads((directory / "manifest.json").read_text())["metadata"][
            "fp_target_source_integrity"
        ]
        inventory = _stream_inventory(
            remote_feature_dir, directory, bucket, tmp_path / f"inv-{directory.name}"
        )
        # The certificate's fold and the streamed fold are the same statement
        # about the same bodies, reached from opposite directions.
        assert inventory["packed_row_fingerprint"] == certificate["packed_row_fingerprint"]
        assert [
            {"name": entry["name"], "byte_length": entry["byte_length"]}
            for entry in inventory["shards"]
        ] == certificate["ordered_shard_inventory"]


def test_streamed_inventory_rejects_a_family_the_manifest_undercounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest's recorded size is what seals the published family."""
    env, _local_env, remote_env, _bucket_root, bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging")) == 0
    output_dir, _val_dir = _split_dirs(remote_feature_dir, env, None)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["provenance_shards"] -= 1
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(Exception, match="exact contiguous family"):
        _stream_inventory(remote_feature_dir, output_dir, bucket, tmp_path / "inv-staging")


def _publisher_cfg(remote_feature_dir: Path, output_dir: Path, staging: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "packed_row_dir": str(output_dir),
            "selection_checksum": None,
            "remote": {
                "enabled": True,
                "bucket_id": "owner/test-bucket",
                "local_features_root": str(remote_feature_dir),
                "remote_packed_root": OFFLOAD_ROOT,
                "staging_dir": str(staging),
                "retry_timeout_seconds": 60.0,
                "retry_interval_seconds": 0.01,
            },
        }
    )


def test_publisher_signs_and_publishes_an_offloaded_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator entrypoint end to end, including a second run over its own output."""
    env, _local_env, remote_env, bucket_root, bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging")) == 0
    output_dir, _val_dir = _split_dirs(remote_feature_dir, env, None)
    monkeypatch.setattr(publisher, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _publisher_cfg(remote_feature_dir, output_dir, tmp_path / "inventory-staging")

    assert publisher.run(cfg) == 0

    prefix = _published_prefix(remote_feature_dir, output_dir)
    local_inventory = json.loads(
        (output_dir / REMOTE_PACKED_INVENTORY_FILENAME).read_text(encoding="utf-8")
    )
    published = bucket_root / prefix / REMOTE_PACKED_INVENTORY_FILENAME
    assert json.loads(published.read_text(encoding="utf-8")) == local_inventory
    assert local_inventory["shards"]

    # Re-running is idempotent, which also proves the family grammar ignores the
    # inventory object now sitting beside the bodies under the same prefix.
    assert publisher.run(cfg) == 0
    assert (
        json.loads(published.read_text(encoding="utf-8"))["aggregate_sha256"]
        == local_inventory["aggregate_sha256"]
    )


@pytest.mark.parametrize("missing", ["local_features_root", "remote_packed_root", "staging_dir"])
def test_publisher_requires_every_remote_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    env, _local_env, remote_env, _bucket_root, _bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    output_dir, _val_dir = _split_dirs(remote_feature_dir, env, None)
    cfg = _publisher_cfg(remote_feature_dir, output_dir, tmp_path / "inventory-staging")
    cfg.remote[missing] = None
    with pytest.raises(ValueError, match=f"remote.{missing} is required"):
        publisher.run(cfg)


def test_streamed_inventory_refuses_a_family_missing_its_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, _bucket_root, bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging")) == 0
    output_dir, _val_dir = _split_dirs(remote_feature_dir, env, None)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"].pop("fp_target_source_integrity")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PackedRowInvariantError, match="requires a completed"):
        build_remote_packed_inventory_from_bucket(
            output_dir,
            backend=bucket,
            bucket_id=HfBucketId.parse("owner/test-bucket"),
            packed_prefix=NormalizedRemotePrefix.parse(
                _published_prefix(remote_feature_dir, output_dir)
            ),
            staging_dir=tmp_path / "inventory-staging",
            retry_timeout_seconds=60.0,
            retry_interval_seconds=0.01,
        )


def test_resuming_an_offloaded_family_without_offload_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, _bucket_root, _bucket, _local_dir, _remote_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging-a")) == 0

    retained = _cfg(remote_env, staging=tmp_path / "staging-b", enabled=True)
    retained.resume = True
    with pytest.raises(RemotePackedRowsError, match="re-enable offload"):
        ppr.run(retained)


def test_fresh_offloaded_run_refuses_to_inherit_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, _bucket_root, _bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging-a")) == 0

    output_dir, val_dir = _split_dirs(remote_feature_dir, env, None)
    for directory in (output_dir, val_dir):
        (directory / "manifest.json").unlink()
        for companion in (directory / "_dims").iterdir():
            companion.unlink()
    # Only the receipts remain: a fresh run must still refuse, because those
    # shards exist in the bucket and a fresh run would contradict them.
    assert ppr._output_has_payload_or_manifest(output_dir)
    fresh = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    with pytest.raises(RemotePackedRowsError, match="fresh remote packed-row output"):
        ppr.run(fresh)


def test_receipts_from_another_identity_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _local_env, remote_env, _bucket_root, _bucket, _local_dir, remote_feature_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    assert ppr.run(_offload_cfg(remote_env, staging=tmp_path / "staging-a")) == 0

    output_dir, _val_dir = _split_dirs(remote_feature_dir, env, None)
    receipt_path = sorted((output_dir / OFFLOAD_RECEIPT_DIRNAME).glob("shard_*.json"))[0]
    payload = json.loads(receipt_path.read_text())
    payload["identity_digest"] = "0" * 64
    receipt_path.write_text(json.dumps(payload))

    rerun = _offload_cfg(remote_env, staging=tmp_path / "staging-b")
    rerun.resume = True
    with pytest.raises(RemotePackedRowsError, match="different schema"):
        ppr.run(rerun)


@pytest.mark.parametrize(
    ("root", "message"),
    [
        (None, "remote.offload.remote_packed_root is required"),
        ("raw", "must be disjoint from the sealed input family"),
        ("raw/deeper", "must be disjoint from the sealed input family"),
    ],
)
def test_offload_configuration_is_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root: str | None,
    message: str,
) -> None:
    _env, _local_env, remote_env, _bucket_root, bucket, _local_dir, _remote_dir = (
        _seeded_remote_run(tmp_path, monkeypatch, n_shards=3)
    )
    cfg = _offload_cfg(remote_env, staging=tmp_path / "staging", root=root)
    with pytest.raises(ValueError, match=message):
        ppr.run(cfg)
    assert bucket.uploads == []
