"""Correctness and lifecycle tests for exact-pass read+zstd overlap."""

from __future__ import annotations

import gc
import json
import threading
import weakref
from pathlib import Path
from typing import cast

import pytest
import torch as t

import steering.compute_statistics as statistics_module
from steering.compute_statistics import compute_statistics, exact_pass1
from steering.exact_statistics import ExactMagnitudeCounts
from steering.statistics_checkpoint import StatisticsCheckpointStore
from tests.steering.test_exact_statistics_integration import (
    _assert_safetensors_exact,
    _freeze_volatile_metadata,
    _write_shards,
)


class _WeakrefableShard(dict):
    pass


class _PreparedPayload:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload


def _force_pipeline(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(
        statistics_module,
        "_should_prefetch_exact_loads",
        lambda **_kwargs: enabled,
    )


def _run(
    feature_dir: Path,
    output: Path,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_interval_shards: int = 50,
):
    return compute_statistics(
        str(feature_dir),
        str(output),
        n_sae_features=4,
        streaming=True,
        device="cpu",
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval_shards=checkpoint_interval_shards,
    )


def _pipeline_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("statistics-read-zstd")
    ]


class TestPipelineSelection:
    def test_requires_cuda_exact_counts_and_multiple_remaining_shards(self) -> None:
        counts = cast(ExactMagnitudeCounts, object())

        assert statistics_module._should_prefetch_exact_loads(
            resolved_device=t.device("cuda"),
            magnitude_counts=counts,
            remaining_shards=2,
        )
        assert not statistics_module._should_prefetch_exact_loads(
            resolved_device=t.device("cpu"),
            magnitude_counts=counts,
            remaining_shards=3,
        )
        assert not statistics_module._should_prefetch_exact_loads(
            resolved_device=t.device("cuda"),
            magnitude_counts=None,
            remaining_shards=3,
        )
        assert not statistics_module._should_prefetch_exact_loads(
            resolved_device=t.device("cuda"),
            magnitude_counts=counts,
            remaining_shards=1,
        )

    def test_cpu_path_never_constructs_an_executor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        real_load = statistics_module.load_features
        loaded: list[Path] = []

        def tracked_load(path: str | Path) -> dict:
            loaded.append(Path(path))
            return real_load(path)

        monkeypatch.setattr(statistics_module, "load_features", tracked_load)
        monkeypatch.setattr(
            statistics_module,
            "ThreadPoolExecutor",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("CPU path constructed the read+zstd executor")
            ),
        )

        result = exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert result.n_shards == len(paths)
        assert loaded == paths

    def test_skip_detail_path_never_constructs_an_executor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        monkeypatch.setattr(
            statistics_module,
            "ThreadPoolExecutor",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("skip-detail path constructed the read+zstd executor")
            ),
        )
        monkeypatch.setattr(
            statistics_module,
            "gpu_accumulate_shard",
            lambda acc, feature_ids, feature_mags, _chunk_size: (
                statistics_module.accumulate_shard(acc, feature_ids, feature_mags)
            ),
        )

        result = exact_pass1(
            str(tmp_path),
            n_sae_features=4,
            device=t.device("cuda"),
            skip_per_feature_detail=True,
            _runtime_resolved=True,
        )

        assert result.n_shards == len(paths)


class TestPipelineArtifactParity:
    def test_complete_artifact_exactly_matches_sequential_reference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        _freeze_volatile_metadata(monkeypatch)

        with monkeypatch.context() as sequential:
            _force_pipeline(sequential, False)
            expected = _run(feature_dir, tmp_path / "sequential")
        with monkeypatch.context() as pipelined:
            _force_pipeline(pipelined, True)
            actual = _run(feature_dir, tmp_path / "pipelined")

        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "pipelined.safetensors",
            tmp_path / "sequential.safetensors",
        )

    def test_profiled_pipeline_preserves_artifact_and_records_strategy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        _freeze_volatile_metadata(monkeypatch)

        with monkeypatch.context() as sequential:
            _force_pipeline(sequential, False)
            _run(feature_dir, tmp_path / "sequential")
        with monkeypatch.context() as pipelined:
            _force_pipeline(pipelined, True)
            compute_statistics(
                str(feature_dir),
                str(tmp_path / "pipelined"),
                n_sae_features=4,
                streaming=True,
                device="cpu",
                profile_output=str(tmp_path / "profile.json"),
            )

        _assert_safetensors_exact(
            tmp_path / "pipelined.safetensors",
            tmp_path / "sequential.safetensors",
        )
        profile = json.loads((tmp_path / "profile.json").read_text())
        assert profile["schema_version"] == 2
        assert profile["config"]["read_zstd_prefetch_enabled"] is True
        assert all(row["phase_seconds"]["load"] > 0 for row in profile["per_shard"])


class TestPipelineScheduling:
    def test_prepares_on_one_worker_decodes_on_main_and_overlaps_reduction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        _force_pipeline(monkeypatch, True)
        main_thread = threading.get_ident()
        prepare_threads: list[int] = []
        decode_threads: list[int] = []
        preparing = 0
        max_preparing = 0
        lock = threading.Lock()
        second_prepare_started = threading.Event()
        allow_second_prepare = threading.Event()
        real_prepare = statistics_module._prepare_exact_shard_safetensors
        real_decode = statistics_module.decode_feature_safetensors_bytes
        real_accumulate = statistics_module.accumulate_shard
        reductions = 0

        def controlled_prepare(**kwargs) -> bytes:
            nonlocal preparing, max_preparing
            index = kwargs["shard_index"]
            with lock:
                preparing += 1
                max_preparing = max(max_preparing, preparing)
            try:
                prepare_threads.append(threading.get_ident())
                if index == 1:
                    second_prepare_started.set()
                    assert allow_second_prepare.wait(timeout=5)
                return real_prepare(**kwargs)
            finally:
                with lock:
                    preparing -= 1

        def tracked_decode(payload: bytes, *, keys=None) -> dict:
            decode_threads.append(threading.get_ident())
            return real_decode(payload, keys=keys)

        def controlled_accumulate(*args, **kwargs) -> None:
            nonlocal reductions
            if reductions == 0:
                assert second_prepare_started.wait(timeout=5)
                allow_second_prepare.set()
            reductions += 1
            real_accumulate(*args, **kwargs)

        monkeypatch.setattr(
            statistics_module,
            "_prepare_exact_shard_safetensors",
            controlled_prepare,
        )
        monkeypatch.setattr(
            statistics_module,
            "decode_feature_safetensors_bytes",
            tracked_decode,
        )
        monkeypatch.setattr(statistics_module, "accumulate_shard", controlled_accumulate)

        result = exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert result.shard_paths == paths
        assert len(prepare_threads) == len(paths)
        assert set(prepare_threads).isdisjoint({main_thread})
        assert len(set(prepare_threads)) == 1
        assert decode_threads == [main_thread] * len(paths)
        assert max_preparing == 1
        assert reductions == len(paths)
        assert not _pipeline_threads()

    def test_releases_prepared_bytes_and_decoded_shard_before_successor_decode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        _force_pipeline(monkeypatch, True)
        real_prepare = statistics_module._prepare_exact_shard_safetensors
        real_decode = statistics_module.decode_feature_safetensors_bytes
        prepared_refs: list[weakref.ReferenceType[_PreparedPayload]] = []
        decoded_refs: list[list[weakref.ReferenceType[object]]] = []

        def tracked_prepare(**kwargs) -> _PreparedPayload:
            gc.collect()
            assert all(reference() is None for reference in prepared_refs)
            holder = _PreparedPayload(real_prepare(**kwargs))
            prepared_refs[:] = [weakref.ref(holder)]
            return holder

        def tracked_decode(holder: _PreparedPayload, *, keys=None) -> dict:
            gc.collect()
            assert all(
                reference() is None
                for references in decoded_refs
                for reference in references
            )
            data = _WeakrefableShard(real_decode(holder.payload, keys=keys))
            decoded_refs[:] = [
                [
                    weakref.ref(data),
                    weakref.ref(data["feature_ids"]),
                    weakref.ref(data["feature_mags"]),
                ]
            ]
            return data

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

        result = exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        gc.collect()
        assert result.n_shards == len(paths)
        assert all(reference() is None for reference in prepared_refs)
        assert all(
            reference() is None
            for references in decoded_refs
            for reference in references
        )


class TestPipelineCheckpoints:
    def test_does_not_prepare_across_checkpoint_boundary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        _force_pipeline(monkeypatch, True)
        real_prepare = statistics_module._prepare_exact_shard_safetensors
        real_commit = StatisticsCheckpointStore.commit
        events: list[tuple[str, int]] = []
        commit_active = threading.Event()

        def tracked_prepare(**kwargs) -> bytes:
            assert not commit_active.is_set()
            events.append(("prepare", kwargs["shard_index"]))
            return real_prepare(**kwargs)

        def tracked_commit(store, **kwargs):
            processed = kwargs["processed_shards"]
            commit_active.set()
            events.append(("commit_start", processed))
            result = real_commit(store, **kwargs)
            events.append(("commit_end", processed))
            commit_active.clear()
            return result

        monkeypatch.setattr(
            statistics_module,
            "_prepare_exact_shard_safetensors",
            tracked_prepare,
        )
        monkeypatch.setattr(StatisticsCheckpointStore, "commit", tracked_commit)

        _run(
            feature_dir,
            tmp_path / "statistics",
            checkpoint_dir=tmp_path / "checkpoint",
            checkpoint_interval_shards=1,
        )

        assert [event for event in events if event[0] == "prepare"] == [
            ("prepare", index) for index in range(len(paths))
        ]
        for index in range(1, len(paths)):
            assert events.index(("commit_end", index)) < events.index(("prepare", index))

    def test_interrupted_pipeline_resumes_without_committed_body_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        _freeze_volatile_metadata(monkeypatch)

        with monkeypatch.context() as sequential:
            _force_pipeline(sequential, False)
            expected = _run(feature_dir, tmp_path / "expected")

        real_prepare = statistics_module._prepare_exact_shard_safetensors
        first_calls: list[int] = []

        def fail_second(**kwargs) -> bytes:
            index = kwargs["shard_index"]
            first_calls.append(index)
            if index == 1:
                raise OSError("deliberate preparation interruption")
            return real_prepare(**kwargs)

        with monkeypatch.context() as interrupted:
            _force_pipeline(interrupted, True)
            interrupted.setattr(
                statistics_module,
                "_prepare_exact_shard_safetensors",
                fail_second,
            )
            with pytest.raises(OSError, match="deliberate preparation interruption"):
                _run(
                    feature_dir,
                    tmp_path / "interrupted",
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_interval_shards=1,
                )

        assert first_calls == [0, 1]
        resumed_calls: list[int] = []

        def track_resume(**kwargs) -> bytes:
            resumed_calls.append(kwargs["shard_index"])
            return real_prepare(**kwargs)

        with monkeypatch.context() as resumed:
            _force_pipeline(resumed, True)
            resumed.setattr(
                statistics_module,
                "_prepare_exact_shard_safetensors",
                track_resume,
            )
            actual = _run(
                feature_dir,
                tmp_path / "actual",
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_shards=1,
            )

        assert resumed_calls == [1, 2]
        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )


class TestPipelineFailures:
    def test_decode_failure_releases_prepared_payload_and_joins_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _force_pipeline(monkeypatch, True)
        real_prepare = statistics_module._prepare_exact_shard_safetensors
        payload_ref: weakref.ReferenceType[_PreparedPayload] | None = None

        def tracked_prepare(**kwargs) -> _PreparedPayload:
            nonlocal payload_ref
            payload = _PreparedPayload(real_prepare(**kwargs))
            payload_ref = weakref.ref(payload)
            return payload

        def fail_decode(_payload: _PreparedPayload, *, keys=None) -> dict:
            del keys
            raise ValueError("deliberate safetensors decode failure")

        monkeypatch.setattr(
            statistics_module,
            "_prepare_exact_shard_safetensors",
            tracked_prepare,
        )
        monkeypatch.setattr(
            statistics_module,
            "decode_feature_safetensors_bytes",
            fail_decode,
        )

        with pytest.raises(ValueError, match="deliberate safetensors decode failure"):
            exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        gc.collect()
        assert payload_ref is not None
        assert payload_ref() is None
        assert not _pipeline_threads()

    def test_primary_reducer_failure_is_preserved_and_worker_is_joined(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _force_pipeline(monkeypatch, True)
        real_prepare = statistics_module._prepare_exact_shard_safetensors
        second_started = threading.Event()
        release_second = threading.Event()

        def blocked_prepare(**kwargs) -> bytes:
            if kwargs["shard_index"] == 1:
                second_started.set()
                assert release_second.wait(timeout=5)
            return real_prepare(**kwargs)

        def fail_reducer(*_args, **_kwargs) -> None:
            assert second_started.wait(timeout=5)
            release_second.set()
            raise RuntimeError("primary reducer failure")

        monkeypatch.setattr(
            statistics_module,
            "_prepare_exact_shard_safetensors",
            blocked_prepare,
        )
        monkeypatch.setattr(statistics_module, "accumulate_shard", fail_reducer)

        with pytest.raises(RuntimeError, match="primary reducer failure"):
            exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert not _pipeline_threads()

    def test_lookahead_failure_is_secondary_to_reducer_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_shards(tmp_path)
        _force_pipeline(monkeypatch, True)
        second_started = threading.Event()
        fail_second = threading.Event()
        real_prepare = statistics_module._prepare_exact_shard_safetensors

        def controlled_prepare(**kwargs) -> bytes:
            if kwargs["shard_index"] == 1:
                second_started.set()
                assert fail_second.wait(timeout=5)
                raise OSError("secondary lookahead failure")
            return real_prepare(**kwargs)

        def fail_reducer(*_args, **_kwargs) -> None:
            assert second_started.wait(timeout=5)
            fail_second.set()
            raise RuntimeError("primary reducer failure")

        monkeypatch.setattr(
            statistics_module,
            "_prepare_exact_shard_safetensors",
            controlled_prepare,
        )
        monkeypatch.setattr(statistics_module, "accumulate_shard", fail_reducer)

        with pytest.raises(RuntimeError, match="primary reducer failure") as caught:
            exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert any(
            "secondary lookahead failure" in note
            for note in getattr(caught.value, "__notes__", ())
        )
        assert not _pipeline_threads()

    @pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(7)])
    def test_base_exceptions_join_worker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        exception: BaseException,
    ) -> None:
        _write_shards(tmp_path)
        _force_pipeline(monkeypatch, True)
        real_decode = statistics_module.decode_feature_safetensors_bytes
        calls = 0

        def interrupt_decode(payload: bytes, *, keys=None) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise exception
            return real_decode(payload, keys=keys)

        monkeypatch.setattr(
            statistics_module,
            "decode_feature_safetensors_bytes",
            interrupt_decode,
        )

        with pytest.raises(type(exception)):
            exact_pass1(str(tmp_path), n_sae_features=4, device=t.device("cpu"))

        assert not _pipeline_threads()
