"""Exact, crash-safe resume tests for feature-statistics checkpoints."""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import shutil
import signal
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file

import steering.compute_statistics as statistics_module
import steering.exact_statistics as exact_statistics_module
import steering.feature_io as feature_io_module
import steering.statistics_checkpoint as checkpoint_module
from steering.compute_statistics import compute_statistics
from steering.exact_statistics import (
    DEFAULT_COUNT_CHUNK_SIZE,
    POSITIVE_FINITE_F16_COUNT,
    ExactMagnitudeCounts,
    effective_available_memory_bytes,
)
from steering.feature_io import (
    FeatureDirectoryMismatchError,
    RegularFileIdentity,
    load_features,
    load_features_stable,
    read_shard_header,
    read_shard_header_stable,
)
from steering.statistics_checkpoint import (
    StatisticsCheckpointError,
    StatisticsCheckpointStore,
    build_checkpoint_contract,
    capture_shard_inventory,
)
from tests.steering.test_exact_statistics_integration import (
    _assert_safetensors_cuda_close,
    _assert_safetensors_exact,
    _freeze_volatile_metadata,
    _write_shards,
)

_CPU_DEVICE = t.device("cpu")


def _run(
    feature_dir: Path,
    output: Path,
    *,
    checkpoint_dir: Path | None = None,
    checkpoint_interval_shards: int = 1,
    skip_per_feature_detail: bool = False,
):
    return compute_statistics(
        str(feature_dir),
        str(output),
        n_sae_features=4,
        streaming=True,
        device="cpu",
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_interval_shards=checkpoint_interval_shards,
        skip_per_feature_detail=skip_per_feature_detail,
    )


def _run_checkpoint_subprocess(
    feature_dir: Path,
    output: Path,
    checkpoint_dir: Path,
    torch_num_threads: int,
    deterministic_algorithms: bool,
    deterministic_warn_only: bool,
    float32_matmul_precision: str,
) -> None:
    """Reproduce the parent's checkpoint runtime contract in a spawned process."""
    t.set_num_threads(torch_num_threads)
    t.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=deterministic_warn_only,
    )
    t.set_float32_matmul_precision(float32_matmul_precision)
    _run(
        feature_dir,
        output,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_shards=1,
    )


def _manifest(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "manifest.json").read_text())


def _state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / _manifest(checkpoint_dir)["state_filename"]


def _assert_checkpoint_cleaned(checkpoint_dir: Path) -> None:
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["checkpoint.lock"]


class TestCheckpointedPipelineParity:
    def test_fresh_checkpointed_output_exactly_matches_disabled_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        _freeze_volatile_metadata(monkeypatch)

        expected = _run(feature_dir, tmp_path / "expected")
        actual = _run(
            feature_dir,
            tmp_path / "actual",
            checkpoint_dir=tmp_path / "checkpoint",
        )

        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        _assert_checkpoint_cleaned(tmp_path / "checkpoint")

    def test_checkpoint_output_exactly_matches_independent_eager_reference(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        _freeze_volatile_metadata(monkeypatch)
        reference = compute_statistics(
            str(feature_dir),
            str(tmp_path / "eager_reference"),
            n_sae_features=4,
            streaming=False,
            device="cpu",
        )
        checkpointed = _run(
            feature_dir,
            tmp_path / "checkpointed",
            checkpoint_dir=tmp_path / "checkpoint",
        )
        assert checkpointed == reference
        _assert_safetensors_exact(
            tmp_path / "checkpointed.safetensors",
            tmp_path / "eager_reference.safetensors",
        )

    def test_skip_detail_checkpointed_output_exactly_matches_disabled_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        _freeze_volatile_metadata(monkeypatch)

        expected = _run(
            feature_dir,
            tmp_path / "expected",
            skip_per_feature_detail=True,
        )
        actual = _run(
            feature_dir,
            tmp_path / "actual",
            checkpoint_dir=tmp_path / "checkpoint",
            skip_per_feature_detail=True,
        )

        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )

    def test_interrupted_run_resumes_without_rereading_committed_shard_bodies(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        _freeze_volatile_metadata(monkeypatch)
        expected = _run(feature_dir, tmp_path / "expected")

        real_load = statistics_module.load_features_stable
        first_attempt_calls: list[Path] = []

        def fail_second(path, *, expected_identity, keys=None):
            first_attempt_calls.append(Path(path))
            if len(first_attempt_calls) == 2:
                raise OSError("deliberate interruption before shard 2")
            return real_load(path, expected_identity=expected_identity, keys=keys)

        with monkeypatch.context() as attempt:
            attempt.setattr(statistics_module, "load_features_stable", fail_second)
            with pytest.raises(OSError, match="deliberate interruption"):
                _run(
                    feature_dir,
                    tmp_path / "interrupted",
                    checkpoint_dir=checkpoint_dir,
                )

        assert first_attempt_calls == paths[:2]
        assert _manifest(checkpoint_dir)["processed_shards"] == 1
        resumed_calls: list[Path] = []

        def track_resume(path, *, expected_identity, keys=None):
            resumed_calls.append(Path(path))
            return real_load(path, expected_identity=expected_identity, keys=keys)

        with monkeypatch.context() as resumed:
            resumed.setattr(statistics_module, "load_features_stable", track_resume)
            actual = _run(
                feature_dir,
                tmp_path / "actual",
                checkpoint_dir=checkpoint_dir,
            )

        assert resumed_calls == paths[1:]
        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        _assert_checkpoint_cleaned(checkpoint_dir)

    def test_checkpoint_candidate_temporaries_are_released_before_next_shard(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        real_commit = StatisticsCheckpointStore.commit
        real_load = statistics_module.load_features_stable
        sample_refs: list[weakref.ReferenceType[t.Tensor]] = []
        loads = 0

        def tracked_commit(store, **kwargs):
            result = real_commit(store, **kwargs)
            sample_refs.append(weakref.ref(kwargs["tensors"]["mag_subsample"]))
            return result

        def checked_load(path, *, expected_identity, keys=None):
            nonlocal loads
            if loads > 0:
                gc.collect()
                assert all(reference() is None for reference in sample_refs)
            loads += 1
            return real_load(path, expected_identity=expected_identity, keys=keys)

        monkeypatch.setattr(StatisticsCheckpointStore, "commit", tracked_commit)
        monkeypatch.setattr(statistics_module, "load_features_stable", checked_load)
        _run(
            feature_dir,
            tmp_path / "stats",
            checkpoint_dir=tmp_path / "checkpoint",
            checkpoint_interval_shards=1,
        )
        gc.collect()
        assert loads == 3
        assert len(sample_refs) == 3
        assert all(reference() is None for reference in sample_refs)

    def test_completed_pass_survives_output_failure_and_resumes_with_zero_body_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        _freeze_volatile_metadata(monkeypatch)
        expected = _run(feature_dir, tmp_path / "expected")
        real_save = statistics_module.save_statistics

        with monkeypatch.context() as failed:
            failed.setattr(
                statistics_module,
                "save_statistics",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("output failed")),
            )
            with pytest.raises(OSError, match="output failed"):
                _run(
                    feature_dir,
                    tmp_path / "failed",
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_interval_shards=10,
                )

        assert _manifest(checkpoint_dir)["processed_shards"] == len(paths)

        def forbidden_body_read(*args, **kwargs):
            raise AssertionError("completed checkpoint reread a raw shard body")

        with monkeypatch.context() as resumed:
            resumed.setattr(statistics_module, "load_features_stable", forbidden_body_read)
            resumed.setattr(statistics_module, "save_statistics", real_save)
            actual = _run(
                feature_dir,
                tmp_path / "actual",
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_shards=10,
            )

        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )

    def test_output_directory_fsync_failure_retains_completed_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        _freeze_volatile_metadata(monkeypatch)
        expected = _run(feature_dir, tmp_path / "expected")

        with monkeypatch.context() as failed:
            failed.setattr(
                statistics_module,
                "fsync_parent_directory",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    OSError("output directory fsync failed")
                ),
            )
            with pytest.raises(OSError, match="output directory fsync failed"):
                _run(
                    feature_dir,
                    tmp_path / "not_durable",
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_interval_shards=10,
                )

        assert (tmp_path / "not_durable.safetensors").is_file()
        assert _manifest(checkpoint_dir)["processed_shards"] == len(paths)

        def forbidden_body_read(*args, **kwargs):
            raise AssertionError("completed checkpoint reread a raw shard body")

        with monkeypatch.context() as resumed:
            resumed.setattr(statistics_module, "load_features_stable", forbidden_body_read)
            actual = _run(
                feature_dir,
                tmp_path / "actual",
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_shards=10,
            )

        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        _assert_checkpoint_cleaned(checkpoint_dir)

    def test_appended_shard_before_publication_retains_checkpoint_and_no_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        real_stats_domain_shards = statistics_module._stats_domain_shards
        inventory_calls = 0

        def append_before_final_fence(feature_dir_arg, exclude_last_shard):
            nonlocal inventory_calls
            inventory_calls += 1
            if inventory_calls == 2:
                shutil.copyfile(
                    paths[-1],
                    feature_dir / "shard_99999_features.safetensors.zst",
                )
            return real_stats_domain_shards(feature_dir_arg, exclude_last_shard)

        monkeypatch.setattr(
            statistics_module,
            "_stats_domain_shards",
            append_before_final_fence,
        )
        with pytest.raises(
            StatisticsCheckpointError,
            match="inventory changed before output publication",
        ):
            _run(
                feature_dir,
                tmp_path / "must_not_publish",
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_shards=10,
            )

        assert inventory_calls == 2
        assert not (tmp_path / "must_not_publish.safetensors").exists()
        assert _manifest(checkpoint_dir)["processed_shards"] == len(paths)

    def test_replaced_processed_shard_before_publication_retains_checkpoint_and_no_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        real_stats_domain_shards = statistics_module._stats_domain_shards
        inventory_calls = 0

        def replace_before_final_fence(feature_dir_arg, exclude_last_shard):
            nonlocal inventory_calls
            inventory_calls += 1
            if inventory_calls == 2:
                paths[0].write_bytes(paths[0].read_bytes() + b"replacement")
            return real_stats_domain_shards(feature_dir_arg, exclude_last_shard)

        monkeypatch.setattr(
            statistics_module,
            "_stats_domain_shards",
            replace_before_final_fence,
        )
        with pytest.raises(
            StatisticsCheckpointError,
            match="inventory changed before output publication",
        ):
            _run(
                feature_dir,
                tmp_path / "must_not_publish",
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval_shards=10,
            )

        assert inventory_calls == 2
        assert not (tmp_path / "must_not_publish.safetensors").exists()
        assert _manifest(checkpoint_dir)["processed_shards"] == len(paths)

    @pytest.mark.skipif(os.name != "posix", reason="checkpoint contract is POSIX-only")
    def test_real_sigkill_recovers_committed_generation_with_exact_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        seed_paths = _write_shards(feature_dir)
        for index in range(len(seed_paths), 80):
            shutil.copyfile(
                seed_paths[index % len(seed_paths)],
                feature_dir / f"shard_{index:05d}_features.safetensors.zst",
            )
        paths = sorted(feature_dir.glob("*_features.safetensors.zst"))
        checkpoint_dir = tmp_path / "checkpoint"
        _freeze_volatile_metadata(monkeypatch)
        expected = _run(feature_dir, tmp_path / "expected")
        process = multiprocessing.get_context("spawn").Process(
            target=_run_checkpoint_subprocess,
            args=(
                feature_dir,
                tmp_path / "killed",
                checkpoint_dir,
                t.get_num_threads(),
                t.are_deterministic_algorithms_enabled(),
                t.is_deterministic_algorithms_warn_only_enabled(),
                t.get_float32_matmul_precision(),
            ),
        )
        process.start()
        deadline = time.perf_counter() + 30.0
        committed = 0
        try:
            while time.perf_counter() < deadline and process.is_alive():
                manifest_path = checkpoint_dir / "manifest.json"
                if manifest_path.is_file():
                    committed = _manifest(checkpoint_dir)["processed_shards"]
                    if 0 < committed < len(paths):
                        assert process.pid is not None
                        os.kill(process.pid, signal.SIGKILL)
                        break
            process.join(timeout=10.0)
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=10.0)

        assert 0 < committed < len(paths)
        assert process.exitcode == -signal.SIGKILL
        assert _manifest(checkpoint_dir)["processed_shards"] >= committed
        actual = _run(
            feature_dir,
            tmp_path / "actual",
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval_shards=1,
        )
        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        _assert_checkpoint_cleaned(checkpoint_dir)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cuda_checkpointed_output_matches_disabled_within_moment_tolerance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        _freeze_volatile_metadata(monkeypatch)
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "expected"),
            n_sae_features=4,
            streaming=True,
            device="cuda",
            gpu_chunk_size=3,
        )
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "actual"),
            n_sae_features=4,
            streaming=True,
            device="cuda",
            gpu_chunk_size=3,
            checkpoint_dir=str(tmp_path / "checkpoint"),
            checkpoint_interval_shards=1,
        )
        _assert_safetensors_cuda_close(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        _assert_checkpoint_cleaned(tmp_path / "checkpoint")

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cuda_resume_accepts_prechange_cpu_exact_checkpoint_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        _freeze_volatile_metadata(monkeypatch)
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "expected"),
            n_sae_features=4,
            streaming=True,
            device="cuda",
            gpu_chunk_size=3,
        )
        real_add = ExactMagnitudeCounts.add
        real_prepare = statistics_module._prepare_exact_shard_safetensors
        first_attempt_calls: list[Path] = []

        def force_cpu_exact_add(
            reducer: ExactMagnitudeCounts,
            feature_ids: t.Tensor,
            feature_mags: t.Tensor,
            *,
            chunk_size: int = DEFAULT_COUNT_CHUNK_SIZE,
            reduction_device: t.device = _CPU_DEVICE,
        ) -> None:
            del reduction_device
            real_add(
                reducer,
                feature_ids,
                feature_mags,
                chunk_size=chunk_size,
                reduction_device=t.device("cpu"),
            )

        def fail_second(**kwargs):
            first_attempt_calls.append(kwargs["shard_path"])
            if len(first_attempt_calls) == 2:
                raise OSError("interrupt after legacy exact-count checkpoint")
            return real_prepare(**kwargs)

        with monkeypatch.context() as attempt:
            attempt.setattr(ExactMagnitudeCounts, "add", force_cpu_exact_add)
            attempt.setattr(
                statistics_module,
                "_prepare_exact_shard_safetensors",
                fail_second,
            )
            with pytest.raises(OSError, match="legacy exact-count checkpoint"):
                compute_statistics(
                    str(feature_dir),
                    str(tmp_path / "interrupted"),
                    n_sae_features=4,
                    streaming=True,
                    device="cuda",
                    gpu_chunk_size=3,
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_interval_shards=1,
                )

        assert first_attempt_calls == paths[:2]
        assert _manifest(checkpoint_dir)["processed_shards"] == 1
        resumed_calls: list[Path] = []

        def track_resume(**kwargs):
            resumed_calls.append(kwargs["shard_path"])
            return real_prepare(**kwargs)

        with monkeypatch.context() as resumed:
            resumed.setattr(
                statistics_module,
                "_prepare_exact_shard_safetensors",
                track_resume,
            )
            compute_statistics(
                str(feature_dir),
                str(tmp_path / "actual"),
                n_sae_features=4,
                streaming=True,
                device="cuda",
                gpu_chunk_size=3,
                checkpoint_dir=str(checkpoint_dir),
                checkpoint_interval_shards=1,
            )

        assert resumed_calls == paths[1:]
        _assert_safetensors_cuda_close(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        _assert_checkpoint_cleaned(checkpoint_dir)

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cuda_oom_during_shard_does_not_advance_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        real_reduce = exact_statistics_module._cuda_unique_f16_counts
        reductions = 0

        def fail_second_reduction(*args, **kwargs):
            nonlocal reductions
            reductions += 1
            if reductions == 2:
                raise t.cuda.OutOfMemoryError("deliberate CUDA OOM")
            return real_reduce(*args, **kwargs)

        with monkeypatch.context() as failed:
            failed.setattr(
                exact_statistics_module,
                "_cuda_unique_f16_counts",
                fail_second_reduction,
            )
            with pytest.raises(t.cuda.OutOfMemoryError, match="deliberate CUDA OOM"):
                compute_statistics(
                    str(feature_dir),
                    str(tmp_path / "failed"),
                    n_sae_features=4,
                    streaming=True,
                    device="cuda",
                    gpu_chunk_size=3,
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_interval_shards=1,
                )

        assert reductions == 2
        assert _manifest(checkpoint_dir)["processed_shards"] == 1
        assert not (tmp_path / "failed.safetensors").exists()


class TestCheckpointRunIdentity:
    def test_eager_path_does_not_resolve_streaming_runtime(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        monkeypatch.setattr(
            statistics_module,
            "_resolve_exact_runtime",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("streaming runtime resolved")
            ),
        )
        stats = compute_statistics(
            str(feature_dir),
            str(tmp_path / "eager"),
            n_sae_features=4,
            streaming=False,
            device="cuda",
        )
        assert stats.metadata.n_sae_features == 4

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"streaming": False}, "streaming=True"),
            ({"activation_percentile": 0.5}, "activation_percentile=0.0"),
            ({"checkpoint_interval_shards": 0}, "positive int"),
            ({"checkpoint_interval_shards": True}, "positive int"),
            ({"checkpoint_dir": ""}, "must not be empty"),
        ],
    )
    def test_unsupported_modes_fail_before_body_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        kwargs: dict,
        message: str,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        monkeypatch.setattr(
            statistics_module,
            "load_features_stable",
            lambda *args, **kw: (_ for _ in ()).throw(AssertionError("body read")),
        )
        base = {
            "feature_dir": str(feature_dir),
            "output_path": str(tmp_path / "stats"),
            "n_sae_features": 4,
            "checkpoint_dir": str(tmp_path / "checkpoint"),
        }
        base.update(kwargs)
        with pytest.raises(StatisticsCheckpointError, match=message):
            cast(Any, compute_statistics)(**base)
        if base["checkpoint_dir"]:
            assert not (tmp_path / "checkpoint").exists()

    def test_noncanonical_implicit_width_fails_before_body_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        monkeypatch.setattr(
            statistics_module,
            "load_features_stable",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("body read")),
        )
        with pytest.raises(StatisticsCheckpointError, match="resolved before body reads"):
            compute_statistics(
                str(feature_dir),
                str(tmp_path / "stats"),
                checkpoint_dir=str(tmp_path / "checkpoint"),
            )

    def test_checkpoint_directory_cannot_contain_inputs_or_output(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        monkeypatch.setattr(
            statistics_module,
            "load_features_stable",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("body read")),
        )
        with pytest.raises(StatisticsCheckpointError, match="dedicated directory"):
            compute_statistics(
                str(feature_dir),
                str(tmp_path / "stats"),
                n_sae_features=4,
                checkpoint_dir=str(feature_dir),
            )
        checkpoint_dir = tmp_path / "checkpoint"
        with pytest.raises(StatisticsCheckpointError, match="contain output_path"):
            compute_statistics(
                str(feature_dir),
                str(checkpoint_dir / "stats"),
                n_sae_features=4,
                checkpoint_dir=str(checkpoint_dir),
            )

    def test_inventory_replacement_is_rejected_before_resume_body_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        real_load = statistics_module.load_features_stable
        calls = 0

        def fail_second(path, *, expected_identity, keys=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("stop")
            return real_load(path, expected_identity=expected_identity, keys=keys)

        with monkeypatch.context() as attempt:
            attempt.setattr(statistics_module, "load_features_stable", fail_second)
            with pytest.raises(OSError, match="stop"):
                _run(feature_dir, tmp_path / "failed", checkpoint_dir=checkpoint_dir)

        original = paths[0].read_bytes()
        paths[0].write_bytes(original + b"changed")
        monkeypatch.setattr(
            statistics_module,
            "load_features_stable",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("body read")),
        )
        with pytest.raises(StatisticsCheckpointError, match=r"contract\.inventory\[0\]"):
            _run(feature_dir, tmp_path / "resume", checkpoint_dir=checkpoint_dir)

    def test_semantic_configuration_mismatch_is_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        real_load = statistics_module.load_features_stable
        calls = 0

        def fail_second(path, *, expected_identity, keys=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("stop")
            return real_load(path, expected_identity=expected_identity, keys=keys)

        with monkeypatch.context() as attempt:
            attempt.setattr(statistics_module, "load_features_stable", fail_second)
            with pytest.raises(OSError, match="stop"):
                _run(feature_dir, tmp_path / "failed", checkpoint_dir=checkpoint_dir)

        with pytest.raises(StatisticsCheckpointError, match="skip_per_feature_detail"):
            _run(
                feature_dir,
                tmp_path / "resume",
                checkpoint_dir=checkpoint_dir,
                skip_per_feature_detail=True,
            )

    def test_numeric_runtime_mismatch_is_rejected_before_body_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        checkpoint_dir = tmp_path / "checkpoint"
        real_load = statistics_module.load_features_stable
        calls = 0

        def fail_second(path, *, expected_identity, keys=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("stop")
            return real_load(path, expected_identity=expected_identity, keys=keys)

        with monkeypatch.context() as attempt:
            attempt.setattr(statistics_module, "load_features_stable", fail_second)
            with pytest.raises(OSError, match="stop"):
                _run(feature_dir, tmp_path / "failed", checkpoint_dir=checkpoint_dir)

        real_fingerprint = statistics_module.numeric_runtime_fingerprint

        def changed_runtime(**kwargs):
            fingerprint = real_fingerprint(**kwargs)
            fingerprint["machine"] = "different-machine"
            return fingerprint

        monkeypatch.setattr(
            statistics_module,
            "numeric_runtime_fingerprint",
            changed_runtime,
        )
        monkeypatch.setattr(
            statistics_module,
            "load_features_stable",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("body read")),
        )
        with pytest.raises(
            StatisticsCheckpointError,
            match=r"contract\.numeric_runtime\.machine",
        ):
            _run(feature_dir, tmp_path / "resume", checkpoint_dir=checkpoint_dir)

    def test_hydra_defaults_and_entrypoint_forward_checkpoint_options(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config_path = (
            Path(__file__).parents[2]
            / "configs"
            / "extraction"
            / "compute_statistics.yaml"
        )
        cfg = OmegaConf.load(config_path)
        assert cfg.checkpoint_dir is None
        assert cfg.checkpoint_interval_shards == 50
        cfg.feature_dir = str(tmp_path / "features")
        cfg.output_path = str(tmp_path / "stats")
        cfg.device = "cpu"
        cfg.checkpoint_dir = str(tmp_path / "checkpoint")
        cfg.checkpoint_interval_shards = 7
        observed: dict = {}
        monkeypatch.setattr(
            statistics_module,
            "compute_statistics",
            lambda **kwargs: observed.update(kwargs),
        )

        statistics_module.main.__wrapped__(cfg)

        assert observed["checkpoint_dir"] == str(tmp_path / "checkpoint")
        assert observed["checkpoint_interval_shards"] == 7

    @pytest.mark.parametrize("invalid", [True, 0, -1, "5", 1.5])
    def test_hydra_entrypoint_rejects_nonpositive_or_nonstrict_interval(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        invalid: object,
    ) -> None:
        config_path = (
            Path(__file__).parents[2]
            / "configs"
            / "extraction"
            / "compute_statistics.yaml"
        )
        cfg = OmegaConf.load(config_path)
        cfg.feature_dir = str(tmp_path / "features")
        cfg.output_path = str(tmp_path / "stats")
        cfg.checkpoint_interval_shards = invalid
        monkeypatch.setattr(
            statistics_module,
            "compute_statistics",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("compute called")),
        )
        with pytest.raises(StatisticsCheckpointError, match="positive int"):
            statistics_module.main.__wrapped__(cfg)


def _store_fixture(tmp_path: Path):
    shard = tmp_path / "shard.bin"
    shard.write_bytes(b"immutable shard")
    inventory = capture_shard_inventory([shard])
    contract = build_checkpoint_contract(
        inventory=inventory,
        n_sae_features=2,
        exclude_last_shard=False,
        skip_per_feature_detail=True,
        l0_tier=None,
        ignore_bos_features=False,
        sft_mixture_fingerprint_json=None,
        numeric_runtime={"device": "cpu"},
    )
    scalars = {
        "metadata": {
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 4,
            "context_window": 8,
            "layer": 1,
        },
        "total_docs": 0,
        "total_tokens": 0,
        "total_storage": len(b"immutable shard"),
        "subsample_step": 1,
        "magnitude_total_count": None,
    }
    tensors = {
        "a": t.tensor([1, 2, 3], dtype=t.int64),
        "b": t.tensor([0.25, 0.5], dtype=t.float64),
        "c": t.tensor([1.0], dtype=t.float32),
    }
    return contract, scalars, tensors


class TestCheckpointStoreTransactions:
    def test_store_round_trip_is_bit_exact(self, tmp_path: Path) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
            assert state.is_file()
        with StatisticsCheckpointStore(directory) as store:
            restored = store.restore(contract)
            assert restored is not None
            assert restored.processed_shards == 1
            assert restored.scalars == scalars
            assert set(restored.tensors) == set(tensors)
            for name in tensors:
                assert t.equal(restored.tensors[name], tensors[name])

    def test_manifest_failure_preserves_previous_committed_generation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            first = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        original_manifest = (directory / "manifest.json").read_bytes()

        with StatisticsCheckpointStore(directory) as store:
            restored = store.restore(contract)
            assert restored is not None
            different = dict(tensors)
            different["a"] = t.tensor([4, 5, 6], dtype=t.int64)
            monkeypatch.setattr(
                checkpoint_module,
                "atomic_json_dump",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest failed")),
            )
            with pytest.raises(OSError, match="manifest failed"):
                store.commit(
                    contract=contract,
                    processed_shards=2,
                    scalars=scalars,
                    tensors=different,
                )

        assert (directory / "manifest.json").read_bytes() == original_manifest
        assert first.is_file()
        with StatisticsCheckpointStore(directory) as store:
            restored = store.restore(contract)
            assert restored is not None
            assert restored.processed_shards == 1
            assert t.equal(restored.tensors["a"], tensors["a"])
            states = list(directory.glob("state-*.safetensors"))
            assert states == [first]

    def test_disk_admission_rejects_before_state_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        monkeypatch.setattr(
            checkpoint_module.shutil,
            "disk_usage",
            lambda path: SimpleNamespace(free=0),
        )
        monkeypatch.setattr(
            checkpoint_module,
            "atomic_safetensors_save",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("state write")),
        )
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="does not fit"):
                store.commit(
                    contract=contract,
                    processed_shards=1,
                    scalars=scalars,
                    tensors=tensors,
                )
        assert not (directory / "manifest.json").exists()
        assert list(directory.glob("state-*.safetensors")) == []

    def test_memory_admission_rejects_before_state_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        monkeypatch.setattr(
            checkpoint_module,
            "effective_available_memory_bytes",
            lambda: state.stat().st_size - 1,
        )
        monkeypatch.setattr(
            checkpoint_module,
            "safe_open",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("state load")),
        )
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="memory headroom"):
                store.restore(contract)

    def test_corrupt_tensor_payload_fails_loudly(self, tmp_path: Path) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        payload = bytearray(state.read_bytes())
        payload[-1] ^= 0x01
        state.write_bytes(payload)
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="tensor digest mismatch"):
                store.restore(contract)

    def test_corrupt_manifest_scalar_fails_cross_bound_digest(
        self,
        tmp_path: Path,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        manifest = _manifest(directory)
        manifest["scalars"]["total_docs"] = 99
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="scalar digest mismatch"):
                store.restore(contract)

    def test_corrupt_header_fails_loudly(self, tmp_path: Path) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        state.write_bytes(b"truncated")
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="unreadable"):
                store.restore(contract)

    def test_duplicate_manifest_key_fails_loudly(self, tmp_path: Path) -> None:
        directory = tmp_path / "checkpoint"
        directory.mkdir()
        (directory / "manifest.json").write_text('{"schema_version":1,"schema_version":1}')
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="duplicate JSON object key"):
                store.restore({})

    def test_unknown_directory_entry_fails_loudly(self, tmp_path: Path) -> None:
        directory = tmp_path / "checkpoint"
        directory.mkdir()
        (directory / "unrelated.txt").write_text("do not delete me")
        with StatisticsCheckpointStore(directory) as store:
            with pytest.raises(StatisticsCheckpointError, match="unknown entry"):
                store.restore({})
        assert (directory / "unrelated.txt").read_text() == "do not delete me"

    def test_concurrent_store_is_rejected(self, tmp_path: Path) -> None:
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory):
            with pytest.raises(StatisticsCheckpointError, match="already in use"):
                with StatisticsCheckpointStore(directory):
                    raise AssertionError("unreachable")

    def test_state_metadata_cross_binds_manifest(self, tmp_path: Path) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        with safe_open(state, framework="pt") as reader:
            metadata = dict(reader.metadata() or {})
        assert metadata["contract_sha256"] == _manifest(directory)["contract_sha256"]
        assert metadata["state_tensor_sha256"] == _manifest(directory)[
            "state_tensor_sha256"
        ]

    def test_state_load_uses_pread_backend(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        real_open = checkpoint_module.safe_open
        backends: list[str | None] = []

        def tracked_open(*args, **kwargs):
            backends.append(kwargs.get("backend"))
            return real_open(*args, **kwargs)

        monkeypatch.setattr(checkpoint_module, "safe_open", tracked_open)
        with StatisticsCheckpointStore(directory) as store:
            assert store.restore(contract) is not None
        assert backends == ["pread"]

    def test_loaded_state_is_writable_without_changing_committed_file(
        self,
        tmp_path: Path,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        original = state.read_bytes()
        with StatisticsCheckpointStore(directory) as store:
            restored = store.restore(contract)
            assert restored is not None
            restored.tensors["a"][0] = 999
        assert state.read_bytes() == original
        assert load_file(state, device="cpu", backend="pread")["a"][0].item() == 1

    def test_cleanup_retires_manifest_before_deleting_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract, scalars, tensors = _store_fixture(tmp_path)
        directory = tmp_path / "checkpoint"
        real_unlink = Path.unlink
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )

            def fail_state_unlink(path: Path, *args, **kwargs):
                if path == state:
                    raise OSError("state cleanup interrupted")
                return real_unlink(path, *args, **kwargs)

            with monkeypatch.context() as interrupted:
                interrupted.setattr(Path, "unlink", fail_state_unlink)
                with pytest.raises(OSError, match="state cleanup interrupted"):
                    store.cleanup_after_output_commit()

        assert not (directory / "manifest.json").exists()
        assert state.exists()
        with StatisticsCheckpointStore(directory) as store:
            assert store.restore(contract) is None
        assert not state.exists()

    @pytest.mark.very_slow
    def test_full_65k_count_table_round_trip_at_production_size(
        self,
        tmp_path: Path,
    ) -> None:
        n_features = 65_536
        payload_bytes = n_features * POSITIVE_FINITE_F16_COUNT * 8
        if effective_available_memory_bytes() < payload_bytes + 8 * 1024**3:
            pytest.skip("insufficient memory headroom for a full 65k checkpoint")
        if shutil.disk_usage(tmp_path).free < payload_bytes + 1024**3:
            pytest.skip("insufficient disk headroom for a full 65k checkpoint")
        contract, scalars, _small_tensors = _store_fixture(tmp_path)
        tensors = {
            "magnitude_counts": t.zeros(
                (n_features, POSITIVE_FINITE_F16_COUNT),
                dtype=t.int64,
            )
        }
        directory = tmp_path / "checkpoint"
        with StatisticsCheckpointStore(directory) as store:
            state = store.commit(
                contract=contract,
                processed_shards=1,
                scalars=scalars,
                tensors=tensors,
            )
        assert _manifest(directory)["state_tensor_bytes"] == payload_bytes
        assert state.stat().st_size >= payload_bytes
        del tensors
        gc.collect()

        with StatisticsCheckpointStore(directory) as store:
            restored = store.restore(contract)
            assert restored is not None
            counts = restored.tensors["magnitude_counts"]
            assert counts.shape == (n_features, POSITIVE_FINITE_F16_COUNT)
            assert counts.dtype == t.int64
            assert counts[0, 0].item() == 0
            assert counts[-1, -1].item() == 0
            store.cleanup_after_output_commit()
        del counts, restored
        gc.collect()
        _assert_checkpoint_cleaned(directory)


class TestStableFeatureBodyReads:
    def test_stable_loader_exactly_matches_legacy_loader(self, tmp_path: Path) -> None:
        paths = _write_shards(tmp_path)
        expected = load_features(paths[0])
        actual = load_features_stable(
            paths[0],
            expected_identity=RegularFileIdentity.capture(paths[0]),
        )
        assert set(actual) == set(expected)
        for key in expected:
            if isinstance(expected[key], t.Tensor):
                assert actual[key].dtype == expected[key].dtype
                assert actual[key].shape == expected[key].shape
                assert t.equal(actual[key], expected[key])
            else:
                assert actual[key] == expected[key]

    def test_stable_loader_rejects_replacement_before_open(self, tmp_path: Path) -> None:
        paths = _write_shards(tmp_path)
        expected_identity = RegularFileIdentity.capture(paths[0])
        paths[0].write_bytes(paths[0].read_bytes() + b"replacement")
        with pytest.raises(FeatureDirectoryMismatchError, match="changed before stable read"):
            load_features_stable(paths[0], expected_identity=expected_identity)

    def test_stable_loader_rejects_symlink_even_to_original_file(self, tmp_path: Path) -> None:
        paths = _write_shards(tmp_path)
        link = tmp_path / "linked_features.safetensors.zst"
        link.symlink_to(paths[0])
        with pytest.raises(FeatureDirectoryMismatchError, match="direct regular file"):
            RegularFileIdentity.capture(link)

    def test_stable_loader_detects_descriptor_change_during_open(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        expected_identity = RegularFileIdentity.capture(paths[0])
        real_open = feature_io_module.os.open

        def replace_then_open(path, flags, *args):
            target = Path(path)
            replacement = target.with_name("replacement.safetensors.zst")
            replacement.write_bytes(target.read_bytes())
            replacement.replace(target)
            return real_open(path, flags, *args)

        monkeypatch.setattr(feature_io_module.os, "open", replace_then_open)
        with pytest.raises(FeatureDirectoryMismatchError, match="while being opened"):
            load_features_stable(paths[0], expected_identity=expected_identity)

    def test_stable_loader_detects_path_swap_after_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        expected_identity = RegularFileIdentity.capture(paths[0])
        real_capture = RegularFileIdentity.capture
        captures = 0

        def swap_on_final_capture(cls, path):
            nonlocal captures
            captures += 1
            if captures == 2:
                target = Path(path)
                replacement = target.with_name("replacement.safetensors.zst")
                replacement.write_bytes(target.read_bytes())
                replacement.replace(target)
            return real_capture(path)

        monkeypatch.setattr(
            RegularFileIdentity,
            "capture",
            classmethod(swap_on_final_capture),
        )
        with pytest.raises(FeatureDirectoryMismatchError, match="pathname changed"):
            load_features_stable(paths[0], expected_identity=expected_identity)


class TestStableFeatureHeaderReads:
    def test_stable_header_exactly_matches_legacy_header(self, tmp_path: Path) -> None:
        paths = _write_shards(tmp_path)
        expected = read_shard_header(paths[0])
        actual = read_shard_header_stable(
            paths[0],
            expected_identity=RegularFileIdentity.capture(paths[0]),
        )
        assert actual == expected

    def test_stable_header_rejects_replacement_before_open(self, tmp_path: Path) -> None:
        paths = _write_shards(tmp_path)
        expected_identity = RegularFileIdentity.capture(paths[0])
        paths[0].write_bytes(paths[0].read_bytes() + b"replacement")
        with pytest.raises(
            FeatureDirectoryMismatchError,
            match="changed before stable header read",
        ):
            read_shard_header_stable(paths[0], expected_identity=expected_identity)

    def test_stable_header_detects_descriptor_swap_during_open(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _write_shards(tmp_path)
        expected_identity = RegularFileIdentity.capture(paths[0])
        real_open = feature_io_module.os.open

        def replace_then_open(path, flags, *args):
            target = Path(path)
            replacement = target.with_name("replacement.safetensors.zst")
            replacement.write_bytes(target.read_bytes())
            replacement.replace(target)
            return real_open(path, flags, *args)

        monkeypatch.setattr(feature_io_module.os, "open", replace_then_open)
        with pytest.raises(FeatureDirectoryMismatchError, match="while opening"):
            read_shard_header_stable(paths[0], expected_identity=expected_identity)

    def test_checkpoint_pipeline_uses_stable_headers_for_every_inventory_entry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        real_read = statistics_module.read_shard_header_stable
        observed: list[tuple[Path, RegularFileIdentity]] = []

        def tracked(path, *, expected_identity):
            observed.append((Path(path), expected_identity))
            return real_read(path, expected_identity=expected_identity)

        monkeypatch.setattr(statistics_module, "read_shard_header_stable", tracked)
        _run(
            feature_dir,
            tmp_path / "stats",
            checkpoint_dir=tmp_path / "checkpoint",
        )
        assert [path for path, _identity in observed] == paths
        assert [identity for _path, identity in observed] == [
            RegularFileIdentity.capture(path) for path in paths
        ]
