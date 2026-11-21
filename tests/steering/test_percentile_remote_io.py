"""Narrow shared I/O contracts used by the remote percentile producer."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch as t

import steering.percentile_remote as remote
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import (
    FeatureDirectoryMismatchError,
    RegularFileIdentity,
    read_feature_compressed_bytes_stable,
)
from steering.overlay_io import (
    OverlayKind,
    overlay_content_digest,
    overlay_tensors_content_digest,
    save_overlay_shard,
    write_manifest,
)
from steering.percentile_remote import RemotePercentileError, materialize_overlay_statistics
from steering.stats_io import (
    load_percentile_checksum,
    load_statistics,
    load_tokenizer_checksum,
    save_percentile_checksum,
    save_tokenizer_checksum,
)
from tests.scripts.test_production_manifest_v2 import _make_env


def test_stable_compressed_reader_returns_exact_bytes(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path, n_shards=1)
    shard = next(feature_dir.glob("*_features.safetensors.zst"))
    expected = shard.read_bytes()
    identity = RegularFileIdentity.capture(shard)
    assert read_feature_compressed_bytes_stable(shard, expected_identity=identity) == expected


def test_stable_compressed_reader_rejects_hardlink(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path, n_shards=1)
    shard = next(feature_dir.glob("*_features.safetensors.zst"))
    hardlink = shard.with_name("hardlink_features.safetensors.zst")
    os.link(shard, hardlink)
    identity = RegularFileIdentity.capture(hardlink)
    with pytest.raises(FeatureDirectoryMismatchError, match="being opened"):
        read_feature_compressed_bytes_stable(hardlink, expected_identity=identity)


def test_stable_compressed_reader_rejects_symlink(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path, n_shards=1)
    shard = next(feature_dir.glob("*_features.safetensors.zst"))
    link = shard.with_name("link_features.safetensors.zst")
    link.symlink_to(shard)
    with pytest.raises(FeatureDirectoryMismatchError, match="direct regular file"):
        RegularFileIdentity.capture(link)


def test_overlay_tensor_digest_matches_file_digest(tmp_path: Path) -> None:
    tensors = {
        "z": t.tensor([3, 2, 1], dtype=t.int16),
        "a": t.tensor([[True, False]], dtype=t.bool),
    }
    path = save_overlay_shard(tensors, tmp_path / "payload")
    assert overlay_tensors_content_digest(tensors) == overlay_content_digest(path)


def test_manifest_can_be_deterministic_v2_from_precomputed_hash(tmp_path: Path) -> None:
    output = tmp_path / "overlay"
    base_hash = "a" * 64
    first = write_manifest(
        output,
        "b" * 64,
        OverlayKind.PERCENTILE,
        precomputed_base_hash=base_hash,
        include_created_at=False,
    ).read_bytes()
    second = write_manifest(
        output,
        "b" * 64,
        OverlayKind.PERCENTILE,
        precomputed_base_hash=base_hash,
        include_created_at=False,
    ).read_bytes()
    assert first == second
    manifest = json.loads(first)
    assert manifest["schema_version"] == 2
    assert manifest["base_shard_content_hash"] == base_hash
    assert "created_at" not in manifest


def test_statistics_materialization_preserves_root_tokenizer_metadata(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "source", n_shards=1)
    statistics = load_statistics(stats_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    save_percentile_checksum(stats_path, percentile_checksum)
    save_tokenizer_checksum(stats_path, "overlay-tokenizer")
    overlay_stats = tmp_path / "overlay" / stats_path.name
    overlay_stats.parent.mkdir()
    shutil.copyfile(stats_path, overlay_stats)

    with pytest.raises(RemotePercentileError, match="manifest identity"):
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_stats,
            feature_root_statistics_path=tmp_path / "mismatched" / stats_path.name,
            percentile_checksum=percentile_checksum,
            expected_statistics_object=remote.RemoteObjectInfo(
                path="raw/pct/test/feature_statistics.safetensors",
                byte_length=overlay_stats.stat().st_size + 1,
                xet_hash="0" * 64,
            ),
        )

    missing_root = tmp_path / "missing" / stats_path.name
    materialize_overlay_statistics(
        overlay_statistics_path=overlay_stats,
        feature_root_statistics_path=missing_root,
        percentile_checksum=percentile_checksum,
    )
    assert load_percentile_checksum(missing_root) == percentile_checksum
    assert load_tokenizer_checksum(missing_root) == "overlay-tokenizer"

    existing_root = feature_dir / "existing_statistics.safetensors"
    shutil.copyfile(stats_path, existing_root)
    save_tokenizer_checksum(existing_root, "root-tokenizer")
    materialize_overlay_statistics(
        overlay_statistics_path=overlay_stats,
        feature_root_statistics_path=existing_root,
        percentile_checksum=percentile_checksum,
    )
    assert load_percentile_checksum(existing_root) == percentile_checksum
    assert load_tokenizer_checksum(existing_root) == "root-tokenizer"


def test_statistics_materialization_rejects_missing_overlay_checksum_before_mutation(
    tmp_path: Path,
) -> None:
    _, stats_path = _make_env(tmp_path / "source", n_shards=1)
    statistics = load_statistics(stats_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    overlay_stats = tmp_path / "overlay" / stats_path.name
    overlay_stats.parent.mkdir()
    shutil.copyfile(stats_path, overlay_stats)

    missing_root = tmp_path / "missing" / stats_path.name
    with pytest.raises(RemotePercentileError, match="lacks the contracted"):
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_stats,
            feature_root_statistics_path=missing_root,
            percentile_checksum=percentile_checksum,
        )
    assert not missing_root.exists()

    existing_root = tmp_path / "existing" / stats_path.name
    existing_root.parent.mkdir()
    shutil.copyfile(stats_path, existing_root)
    before = existing_root.read_bytes()
    with pytest.raises(RemotePercentileError, match="lacks the contracted"):
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_stats,
            feature_root_statistics_path=existing_root,
            percentile_checksum=percentile_checksum,
        )
    assert existing_root.read_bytes() == before


def test_statistics_materialization_validates_copy_before_installing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stats_path = _make_env(tmp_path / "source", n_shards=1)
    statistics = load_statistics(stats_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    save_percentile_checksum(stats_path, percentile_checksum)
    overlay_stats = tmp_path / "overlay" / stats_path.name
    overlay_stats.parent.mkdir()
    shutil.copyfile(stats_path, overlay_stats)
    missing_root = tmp_path / "missing" / stats_path.name

    monkeypatch.setattr(
        remote,
        "atomic_write_bytes",
        lambda _payload, _path: (_ for _ in ()).throw(OSError("injected write failure")),
    )
    with pytest.raises(
        RemotePercentileError,
        match="could not be installed",
    ):
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_stats,
            feature_root_statistics_path=missing_root,
            percentile_checksum=percentile_checksum,
        )
    assert not missing_root.exists()


def test_statistics_materialization_installation_races_preserve_external_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stats_path = _make_env(tmp_path / "source", n_shards=1)
    statistics = load_statistics(stats_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    save_percentile_checksum(stats_path, percentile_checksum)
    overlay_stats = tmp_path / "overlay" / stats_path.name
    overlay_stats.parent.mkdir()
    shutil.copyfile(stats_path, overlay_stats)

    link_failure_root = tmp_path / "link-failure" / stats_path.name
    with monkeypatch.context() as scoped:
        scoped.setattr(
            remote.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link failed")),
        )
        with pytest.raises(RemotePercentileError, match="could not be installed"):
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=link_failure_root,
                percentile_checksum=percentile_checksum,
            )
    assert not link_failure_root.exists()
    assert not list(link_failure_root.parent.glob(".*.materialize.*"))

    interrupted_root = tmp_path / "keyboard-interrupt" / stats_path.name
    real_link = remote.os.link

    def interrupt_after_link(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        raise KeyboardInterrupt("interrupted after atomic link")

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", interrupt_after_link)
        with pytest.raises(KeyboardInterrupt, match="interrupted after atomic link"):
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=interrupted_root,
                percentile_checksum=percentile_checksum,
            )
    assert not interrupted_root.exists()
    assert not list(interrupted_root.parent.glob(".*.materialize.*"))

    concurrent_root = tmp_path / "concurrent-root" / stats_path.name

    def publish_concurrent_root(_source, destination, **_kwargs):
        Path(destination).write_bytes(b"must survive")
        raise FileExistsError("concurrent publisher won")

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", publish_concurrent_root)
        with pytest.raises(RemotePercentileError, match="could not be installed"):
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=concurrent_root,
                percentile_checksum=percentile_checksum,
            )
    assert concurrent_root.read_bytes() == b"must survive"

    replaced_root = tmp_path / "replaced-root" / stats_path.name

    def replace_root_after_link(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        replacement = Path(destination).with_name("external-replacement")
        replacement.write_bytes(b"must survive")
        replacement.replace(destination)

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", replace_root_after_link)
        with pytest.raises(RemotePercentileError, match="could not be installed"):
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=replaced_root,
                percentile_checksum=percentile_checksum,
            )
    assert replaced_root.read_bytes() == b"must survive"
    assert not list(replaced_root.parent.glob(".*.materialize.*"))

    replaced_staging_root = tmp_path / "replaced-staging" / stats_path.name

    def replace_staging_before_link(source, _destination, **_kwargs):
        replacement = Path(source).with_name("external-staging-replacement")
        replacement.write_bytes(b"must survive")
        replacement.replace(source)
        raise OSError("staged path replaced")

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", replace_staging_before_link)
        with pytest.raises(RemotePercentileError, match="could not be installed"):
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=replaced_staging_root,
                percentile_checksum=percentile_checksum,
            )
    retained = list(replaced_staging_root.parent.glob(".*.materialize.*"))
    assert len(retained) == 1 and retained[0].read_bytes() == b"must survive"
    assert not replaced_staging_root.exists()

    removed_staging_root = tmp_path / "removed-staging" / stats_path.name

    def remove_staging_after_link(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        Path(source).unlink()

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", remove_staging_after_link)
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_stats,
            feature_root_statistics_path=removed_staging_root,
            percentile_checksum=percentile_checksum,
        )
    assert load_percentile_checksum(removed_staging_root) == percentile_checksum
    assert not list(removed_staging_root.parent.glob(".*.materialize.*"))

    replaced_after_link_root = tmp_path / "replaced-after-link" / stats_path.name

    def replace_staging_after_link(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        replacement = Path(source).with_name("external-after-link")
        replacement.write_bytes(b"must survive")
        replacement.replace(source)

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", replace_staging_after_link)
        materialize_overlay_statistics(
            overlay_statistics_path=overlay_stats,
            feature_root_statistics_path=replaced_after_link_root,
            percentile_checksum=percentile_checksum,
        )
    assert load_percentile_checksum(replaced_after_link_root) == percentile_checksum
    retained_after_link = list(
        replaced_after_link_root.parent.glob(".*.materialize.*")
    )
    assert (
        len(retained_after_link) == 1
        and retained_after_link[0].read_bytes() == b"must survive"
    )


@pytest.mark.parametrize("primary_kind", ["error", "interrupt"])
def test_statistics_materialization_preserves_primary_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_kind: str,
) -> None:
    _, stats_path = _make_env(tmp_path / "source", n_shards=1)
    statistics = load_statistics(stats_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    save_percentile_checksum(stats_path, percentile_checksum)
    overlay_stats = tmp_path / "overlay" / stats_path.name
    overlay_stats.parent.mkdir()
    shutil.copyfile(stats_path, overlay_stats)
    root = tmp_path / primary_kind / stats_path.name
    real_link = remote.os.link
    real_unlink = Path.unlink
    primary: BaseException = (
        OSError("primary link failure")
        if primary_kind == "error"
        else KeyboardInterrupt("primary interruption")
    )

    def fail_after_link(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        raise primary

    def fail_root_unlink(path: Path, *args, **kwargs):
        if path == root:
            raise OSError("root cleanup failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", fail_after_link)
        scoped.setattr(Path, "unlink", fail_root_unlink)
        expected_type = RemotePercentileError if primary_kind == "error" else KeyboardInterrupt
        with pytest.raises(expected_type) as caught:
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=root,
                percentile_checksum=percentile_checksum,
            )
    assert any("root statistics cleanup also failed" in note for note in caught.value.__notes__)
    assert root.exists()
    root.unlink()


def test_statistics_materialization_reports_temporary_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stats_path = _make_env(tmp_path / "source", n_shards=1)
    statistics = load_statistics(stats_path)
    percentile_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    save_percentile_checksum(stats_path, percentile_checksum)
    overlay_stats = tmp_path / "overlay" / stats_path.name
    overlay_stats.parent.mkdir()
    shutil.copyfile(stats_path, overlay_stats)
    real_unlink = Path.unlink

    def fail_staged_unlink(path: Path, *args, **kwargs):
        if ".materialize." in path.name:
            raise OSError("temporary cleanup failure")
        return real_unlink(path, *args, **kwargs)

    successful_root = tmp_path / "successful" / stats_path.name
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", fail_staged_unlink)
        with pytest.raises(OSError, match="temporary cleanup failure"):
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=successful_root,
                percentile_checksum=percentile_checksum,
            )
    retained = next(successful_root.parent.glob(".*.materialize.*"))
    retained.unlink()
    successful_root.unlink()

    failed_root = tmp_path / "failed" / stats_path.name
    with monkeypatch.context() as scoped:
        scoped.setattr(
            remote.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("primary link failure")
            ),
        )
        scoped.setattr(Path, "unlink", fail_staged_unlink)
        with pytest.raises(RemotePercentileError) as caught:
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=failed_root,
                percentile_checksum=percentile_checksum,
            )
    assert any(
        "temporary root statistics cleanup also failed" in note
        for note in caught.value.__notes__
    )
    next(failed_root.parent.glob(".*.materialize.*")).unlink()

    identical = KeyboardInterrupt("same cleanup interruption")
    identical_root = tmp_path / "identical" / stats_path.name

    def raise_identical(*_args, **_kwargs):
        raise identical

    with monkeypatch.context() as scoped:
        scoped.setattr(remote.os, "link", raise_identical)
        scoped.setattr(Path, "unlink", raise_identical)
        with pytest.raises(KeyboardInterrupt) as caught_identical:
            materialize_overlay_statistics(
                overlay_statistics_path=overlay_stats,
                feature_root_statistics_path=identical_root,
                percentile_checksum=percentile_checksum,
            )
    assert caught_identical.value is identical
    assert getattr(identical, "__notes__", ()) == ()
    next(identical_root.parent.glob(".*.materialize.*")).unlink()
