"""Remote-transport integration for statistics read+zstd overlap."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import steering.compute_statistics as statistics_module
import steering.statistics_remote as remote_module
from steering.compute_statistics import compute_statistics
from steering.remote_training import RemoteTrainingError
from steering.statistics_remote import RemoteStatisticsSource
from tests.steering.test_exact_statistics_integration import (
    _assert_safetensors_exact,
    _freeze_volatile_metadata,
)
from tests.steering.test_statistics_remote import _config, _setup_remote


def _pipeline_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("statistics-read-zstd")
    ]


def test_remote_pipeline_matches_local_and_keeps_thread_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    _freeze_volatile_metadata(monkeypatch)
    main_thread = threading.get_ident()

    with monkeypatch.context() as sequential:
        sequential.setattr(
            statistics_module,
            "_should_prefetch_exact_loads",
            lambda **_kwargs: False,
        )
        expected = compute_statistics(
            str(feature_dir),
            str(tmp_path / "local"),
            n_sae_features=4,
            device="cpu",
        )

    prepare_threads: list[int] = []
    decode_threads: list[int] = []
    download_threads: list[int] = []
    real_prepare = statistics_module._prepare_exact_shard_safetensors
    real_decode = statistics_module.decode_feature_safetensors_bytes
    real_download = bucket.download

    def tracked_prepare(**kwargs) -> bytes:
        prepare_threads.append(threading.get_ident())
        return real_prepare(**kwargs)

    def tracked_decode(payload: bytes, *, keys=None) -> dict:
        decode_threads.append(threading.get_ident())
        return real_decode(payload, keys=keys)

    def tracked_download(remote_path: str, local_path: Path) -> None:
        download_threads.append(threading.get_ident())
        real_download(remote_path, local_path)

    monkeypatch.setattr(
        statistics_module,
        "_should_prefetch_exact_loads",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        statistics_module,
        "_prepare_exact_shard_safetensors",
        tracked_prepare,
    )
    monkeypatch.setattr(
        statistics_module,
        "decode_feature_safetensors_bytes",
        tracked_decode,
    )
    monkeypatch.setattr(bucket, "download", tracked_download)

    actual = compute_statistics(
        str(feature_dir),
        str(tmp_path / "remote"),
        n_sae_features=4,
        device="cpu",
        profile_output=str(tmp_path / "profile.json"),
        remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
    )

    assert actual == expected
    _assert_safetensors_exact(
        tmp_path / "remote.safetensors",
        tmp_path / "local.safetensors",
    )
    assert len(prepare_threads) == 3
    assert len(set(prepare_threads)) == 1
    assert main_thread not in prepare_threads
    assert decode_threads == [main_thread] * 3
    assert len(download_threads) == 3
    assert len(set(download_threads)) == 1
    assert set(download_threads).isdisjoint(set(prepare_threads) | {main_thread})
    assert not _pipeline_threads()

    profile = json.loads((tmp_path / "profile.json").read_text())
    assert profile["config"]["read_zstd_prefetch_enabled"] is True
    assert profile["remote_transport"]["staging_high_water_bodies"] <= 2
    assert profile["remote_transport"]["status"] == "complete"
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]


def test_reducer_failure_cancels_remote_before_joining_local_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    monkeypatch.setattr(
        statistics_module,
        "_should_prefetch_exact_loads",
        lambda **_kwargs: True,
    )
    second_download_started = threading.Event()
    cancellation_observed = threading.Event()
    call_order: list[str] = []
    real_download_verified = remote_module.download_verified_object
    real_request_cancel = RemoteStatisticsSource.request_cancel
    real_finish_transport = RemoteStatisticsSource.finish_transport

    def controlled_download(
        backend,
        *,
        expected,
        final_path,
        timeout_seconds,
        retry_interval_seconds,
        cancellation_event,
        verified_publication_callback,
    ):
        if "shard_00000" in expected.path:
            return real_download_verified(
                backend,
                expected=expected,
                final_path=final_path,
                timeout_seconds=timeout_seconds,
                retry_interval_seconds=retry_interval_seconds,
                cancellation_event=cancellation_event,
                verified_publication_callback=verified_publication_callback,
            )
        second_download_started.set()
        assert cancellation_event.wait(timeout=5)
        cancellation_observed.set()
        raise RemoteTrainingError(f"download cancelled: {expected.path}")

    def tracked_request_cancel(source: RemoteStatisticsSource) -> None:
        call_order.append("request_cancel")
        real_request_cancel(source)

    def tracked_finish_transport(source: RemoteStatisticsSource) -> None:
        assert not _pipeline_threads()
        call_order.append("finish_transport")
        real_finish_transport(source)

    def fail_reducer(*_args, **_kwargs) -> None:
        assert second_download_started.wait(timeout=5)
        raise RuntimeError("primary reducer failure")

    monkeypatch.setattr(remote_module, "download_verified_object", controlled_download)
    monkeypatch.setattr(RemoteStatisticsSource, "request_cancel", tracked_request_cancel)
    monkeypatch.setattr(RemoteStatisticsSource, "finish_transport", tracked_finish_transport)
    monkeypatch.setattr(statistics_module, "accumulate_shard", fail_reducer)

    with pytest.raises(RuntimeError, match="primary reducer failure"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            device="cpu",
            profile_output=str(tmp_path / "failure-profile.json"),
            remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
        )

    assert cancellation_observed.is_set()
    assert call_order[0] == "request_cancel"
    assert call_order.index("finish_transport") > 0
    assert not _pipeline_threads()
    assert not (tmp_path / "statistics.safetensors").exists()
    failure_profile = json.loads((tmp_path / "failure-profile.json").read_text())
    assert failure_profile["config"]["read_zstd_prefetch_enabled"] is True
    assert failure_profile["remote_transport"]["status"] == "failed"
    assert failure_profile["remote_transport"]["terminal_error"]["message"] == (
        "primary reducer failure"
    )
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]


def test_request_cancel_is_idempotent_and_does_not_finish_transport(
    tmp_path: Path,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )

    source.request_cancel()
    source.request_cancel()

    assert source._cancel.is_set()
    assert not source._finish_done
    assert source._executor is None
    source.finish_transport()
    assert source._finish_done


def test_pipeline_loader_failure_keeps_release_failure_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    monkeypatch.setattr(
        statistics_module,
        "_should_prefetch_exact_loads",
        lambda **_kwargs: True,
    )

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("primary stable-load failure")

    def fail_release(*_args, **_kwargs):
        raise OSError("secondary release failure")

    monkeypatch.setattr(
        statistics_module,
        "load_feature_safetensors_bytes_stable",
        fail_load,
    )
    monkeypatch.setattr(RemoteStatisticsSource, "release", fail_release)

    with pytest.raises(RuntimeError, match="primary stable-load failure") as caught:
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            device="cpu",
            remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
        )

    assert any(
        "secondary release failure" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not _pipeline_threads()
