"""Performance-instrumentation tests for feature statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.compute_statistics as statistics_module
import steering.profiling.feature_statistics as statistics_profile_module
from steering.compute_statistics import compute_statistics
from steering.feature_io import load_features
from steering.profiling.feature_statistics import (
    STATISTICS_PROFILE_SCHEMA_VERSION,
    STATISTICS_SHARD_PHASES,
)
from tests.steering.test_exact_statistics_integration import (
    _assert_safetensors_exact,
    _freeze_volatile_metadata,
    _write_shards,
)


def _load_profile(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _expected_shard_counters(paths: list[Path]) -> tuple[int, int]:
    n_activations = 0
    input_bytes = 0
    for path in paths:
        data = load_features(path)
        n_activations += int(data["feature_ids"].numel())
        input_bytes += path.stat().st_size
    return n_activations, input_bytes


class TestStatisticsProfilePipeline:
    def test_profiled_cpu_output_matches_unprofiled_and_reports_every_shard_phase(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        profile_path = tmp_path / "profile.json"
        _freeze_volatile_metadata(monkeypatch)

        expected = compute_statistics(
            str(feature_dir),
            str(tmp_path / "expected"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )
        actual = compute_statistics(
            str(feature_dir),
            str(tmp_path / "actual"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
            profile_output=str(profile_path),
        )

        assert actual == expected
        _assert_safetensors_exact(
            tmp_path / "actual.safetensors",
            tmp_path / "expected.safetensors",
        )
        payload = _load_profile(profile_path)
        assert set(payload) == {
            "schema_version",
            "config",
            "run_regions_seconds",
            "shard_phase_seconds",
            "per_shard",
            "totals",
        }
        assert payload["schema_version"] == STATISTICS_PROFILE_SCHEMA_VERSION
        assert payload["config"]["feature_dir"] == str(feature_dir.resolve())
        assert payload["config"]["requested_device"] == "cpu"
        assert payload["config"]["numeric_runtime"]["device"] == "cpu"
        assert payload["config"]["resolved_n_sae_features"] == 4
        assert payload["config"]["n_input_shards"] == len(paths)
        assert payload["config"]["resumed_from_shard"] == 0
        assert payload["config"]["checkpoint_enabled"] is False
        assert payload["config"]["read_zstd_prefetch_enabled"] is False

        per_shard = payload["per_shard"]
        assert [row["shard_index"] for row in per_shard] == list(range(len(paths)))
        assert [row["shard_name"] for row in per_shard] == [path.name for path in paths]
        for row in per_shard:
            assert set(row) == {
                "shard_index",
                "shard_name",
                "input_bytes",
                "n_activations",
                "n_segments",
                "phase_seconds",
            }
            assert row["input_bytes"] > 0
            assert row["n_activations"] > 0
            assert row["n_segments"] > 0
            assert tuple(row["phase_seconds"]) == STATISTICS_SHARD_PHASES
            assert row["phase_seconds"]["load"] > 0.0
            assert row["phase_seconds"]["validate"] > 0.0
            assert row["phase_seconds"]["exact_counts"] > 0.0
            assert row["phase_seconds"]["moments"] > 0.0
            assert row["phase_seconds"]["segment_stats"] > 0.0
            assert row["phase_seconds"]["bookkeeping"] > 0.0
            assert row["phase_seconds"]["checkpoint"] == 0.0

        expected_activations, expected_bytes = _expected_shard_counters(paths)
        totals = payload["totals"]
        assert totals["n_shards_profiled"] == len(paths)
        assert totals["n_activations"] == expected_activations
        assert totals["input_bytes"] == expected_bytes
        assert totals["wall_seconds"] > 0.0
        assert totals["shards_per_hour"] > 0.0
        assert totals["input_gib_per_second"] > 0.0
        assert totals["million_activations_per_second"] > 0.0

        phase_totals = payload["shard_phase_seconds"]
        assert tuple(phase_totals) == STATISTICS_SHARD_PHASES
        for phase in STATISTICS_SHARD_PHASES:
            assert phase_totals[phase] == pytest.approx(
                sum(row["phase_seconds"][phase] for row in per_shard)
            )

        run_regions = payload["run_regions_seconds"]
        assert run_regions["statistics.preflight"] > 0.0
        assert run_regions["statistics.pass1.wall"] > 0.0
        assert run_regions["statistics.pass1.count_table_allocate"] > 0.0
        assert run_regions["statistics.pass1.finish"] > 0.0
        assert run_regions["statistics.finalize.exact_magnitudes"] > 0.0
        assert run_regions["statistics.finalize.wall"] > 0.0
        assert run_regions["statistics.finalize.wall"] >= run_regions[
            "statistics.finalize.exact_magnitudes"
        ]
        assert run_regions["statistics.output.serialize"] > 0.0

    def test_checkpoint_profile_includes_snapshot_fence_and_cleanup_costs(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        profile_path = tmp_path / "profile.json"

        compute_statistics(
            str(feature_dir),
            str(tmp_path / "stats"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
            checkpoint_dir=str(tmp_path / "checkpoint"),
            checkpoint_interval_shards=1,
            profile_output=str(profile_path),
        )

        payload = _load_profile(profile_path)
        assert payload["config"]["checkpoint_enabled"] is True
        assert payload["config"]["checkpoint_interval_shards"] == 1
        assert len(payload["per_shard"]) == len(paths)
        assert all(
            row["phase_seconds"]["checkpoint"] > 0.0
            for row in payload["per_shard"]
        )
        assert payload["run_regions_seconds"]["statistics.checkpoint.contract"] > 0.0
        assert (
            payload["run_regions_seconds"]["statistics.pass1.checkpoint_restore"]
            > 0.0
        )
        assert payload["run_regions_seconds"]["statistics.output.inventory_fence"] > 0.0
        assert payload["run_regions_seconds"]["statistics.output.checkpoint_cleanup"] > 0.0
        assert sorted(path.name for path in (tmp_path / "checkpoint").iterdir()) == [
            "checkpoint.lock"
        ]

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
    def test_cuda_profile_records_resolved_runtime_and_gpu_phases(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        paths = _write_shards(feature_dir)
        profile_path = tmp_path / "profile.json"

        compute_statistics(
            str(feature_dir),
            str(tmp_path / "stats"),
            n_sae_features=4,
            streaming=True,
            device="cuda",
            gpu_chunk_size=3,
            profile_output=str(profile_path),
        )

        payload = _load_profile(profile_path)
        runtime = payload["config"]["numeric_runtime"]
        assert runtime["device"] == "cuda"
        assert runtime["gpu_chunk_size"] == 3
        assert runtime["gpu_uuid"]
        assert payload["config"]["read_zstd_prefetch_enabled"] is True
        assert len(payload["per_shard"]) == len(paths)
        assert payload["shard_phase_seconds"]["exact_counts"] > 0.0
        assert payload["shard_phase_seconds"]["moments"] > 0.0

    def test_profile_path_cannot_overwrite_statistics_artifact(self, tmp_path: Path) -> None:
        output = tmp_path / "statistics"
        with pytest.raises(ValueError, match="must differ"):
            compute_statistics(
                str(tmp_path / "missing-features"),
                str(output),
                n_sae_features=4,
                profile_output=str(output.with_suffix(".safetensors")),
            )

    def test_profile_rejects_eager_mode_instead_of_emitting_incomplete_phases(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="profiling requires streaming=True"):
            compute_statistics(
                str(tmp_path / "missing-features"),
                str(tmp_path / "statistics"),
                n_sae_features=4,
                streaming=False,
                profile_output=str(tmp_path / "profile.json"),
            )

    def test_failed_computation_does_not_publish_a_misleading_profile(
        self,
        tmp_path: Path,
    ) -> None:
        profile_path = tmp_path / "profile.json"
        with pytest.raises(FileNotFoundError, match="No feature shard files"):
            compute_statistics(
                str(tmp_path / "missing-features"),
                str(tmp_path / "statistics"),
                n_sae_features=4,
                profile_output=str(profile_path),
            )
        assert not profile_path.exists()

    def test_disabled_profile_markers_do_not_read_the_high_resolution_clock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        _write_shards(feature_dir)
        monkeypatch.setattr(
            statistics_profile_module.time,
            "perf_counter_ns",
            lambda: (_ for _ in ()).throw(AssertionError("profiling clock read")),
        )

        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            streaming=True,
            device="cpu",
        )


class TestStatisticsProfileConfig:
    def test_defaults_are_disabled_and_main_forwards_default_or_explicit_path(
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
        assert cfg.profile is False
        assert cfg.profile_summary_output is None
        cfg.feature_dir = str(tmp_path / "features")
        cfg.output_path = str(tmp_path / "stats")
        cfg.device = "cpu"
        observed: list[dict] = []
        monkeypatch.setattr(
            statistics_module,
            "compute_statistics",
            lambda **kwargs: observed.append(kwargs),
        )

        statistics_module.main.__wrapped__(cfg)
        assert observed[-1]["profile_output"] is None

        cfg.profile = True
        statistics_module.main.__wrapped__(cfg)
        assert observed[-1]["profile_output"] == (
            "out/traces/compute_statistics_profile.json"
        )

        cfg.profile_summary_output = str(tmp_path / "custom-profile.json")
        statistics_module.main.__wrapped__(cfg)
        assert observed[-1]["profile_output"] == str(tmp_path / "custom-profile.json")
