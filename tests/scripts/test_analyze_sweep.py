"""Tests for sweep analysis functions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval.analyze_sweep import (
    compute_interaction,
    compute_marginal_effects,
    compute_pareto_front,
    compute_summary_stats,
    filter_valid,
    generate_all,
    is_valid,
    load_results,
    rank_configs,
)


def _make_record(
    *,
    config_index: int = 0,
    val_bpb: float | None = 1.5,
    min_val_bpb: float | None = 1.4,
    total_training_time: float = 30.0,
    peak_memory_mib: float = 8000.0,
    startup_time: float = 1.0,
    error: str | None = None,
    fs_distribution: str = "geometric",
    fs_mean: float = 8.0,
    fs_std: float = 8.0,
    ns_mean: float = 8.0,
    fd_mode: str | None = None,
    fd_neg_delta_ratio: float | None = None,
    fp_weight: float = 0.1,
) -> dict:
    """Create a synthetic sweep result record."""
    return {
        "model_tag": f"sweep_d12/ns_geo_m{ns_mean:g}/fd_{'disabled' if fd_mode is None else fd_mode}/fpw_{fp_weight:g}",
        "config_label": f"{'geo' if fs_distribution == 'geometric' else 'norm'}_m{fs_mean:g}/ns_geo_m{ns_mean:g}/fd_{'disabled' if fd_mode is None else fd_mode}/fpw_{fp_weight:g}",
        "val_bpb": val_bpb,
        "min_val_bpb": min_val_bpb,
        "total_training_time": total_training_time,
        "peak_memory_mib": peak_memory_mib,
        "startup_time": startup_time,
        "error": error,
        "fs_distribution": fs_distribution,
        "fs_mean": fs_mean,
        "fs_std": fs_std,
        "ns_mean": ns_mean,
        "fd_mode": fd_mode,
        "fd_neg_delta_ratio": fd_neg_delta_ratio,
        "fp_weight": fp_weight,
        "config_index": config_index,
        "timestamp": "2026-03-23T14:00:00+00:00",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestLoadResults:
    def test_load_all_records(self, tmp_path: Path) -> None:
        records = [_make_record(config_index=i) for i in range(10)]
        jsonl = tmp_path / "results.jsonl"
        _write_jsonl(jsonl, records)

        loaded = load_results(jsonl)
        assert len(loaded) == 10
        assert loaded[0]["config_index"] == 0
        assert isinstance(loaded[0]["val_bpb"], float)

    def test_load_partial_results(self, tmp_path: Path) -> None:
        records = [_make_record(config_index=i) for i in range(5)]
        jsonl = tmp_path / "results.jsonl"
        _write_jsonl(jsonl, records)

        loaded = load_results(jsonl)
        assert len(loaded) == 5

    def test_load_empty_file(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "results.jsonl"
        jsonl.write_text("")

        loaded = load_results(jsonl)
        assert loaded == []


class TestIsValid:
    def test_valid_record(self) -> None:
        assert is_valid(_make_record()) is True

    def test_error_set(self) -> None:
        assert is_valid(_make_record(error="OOM")) is False

    def test_min_val_bpb_none(self) -> None:
        assert is_valid(_make_record(min_val_bpb=None)) is False

    def test_val_bpb_none(self) -> None:
        assert is_valid(_make_record(val_bpb=None)) is False


class TestRankConfigs:
    def test_top_5(self) -> None:
        records = [_make_record(config_index=i, min_val_bpb=2.0 - i * 0.1) for i in range(20)]
        top5 = rank_configs(records, key="min_val_bpb", n=5, ascending=True)
        assert len(top5) == 5
        # Best (lowest) bpb should be first
        assert top5[0]["min_val_bpb"] < top5[4]["min_val_bpb"]
        assert top5[0]["config_index"] == 19  # 2.0 - 19*0.1 = 0.1

    def test_top_10(self) -> None:
        """Match production usage in print_summary which uses n=10."""
        records = [_make_record(config_index=i, min_val_bpb=2.0 - i * 0.1) for i in range(20)]
        top10 = rank_configs(records, key="min_val_bpb", n=10, ascending=True)
        assert len(top10) == 10
        assert top10[0]["config_index"] == 19

    def test_bottom_n(self) -> None:
        records = [_make_record(config_index=i, min_val_bpb=2.0 - i * 0.1) for i in range(20)]
        bottom5 = rank_configs(records, key="min_val_bpb", n=5, ascending=False)
        assert len(bottom5) == 5
        assert bottom5[0]["min_val_bpb"] > bottom5[4]["min_val_bpb"]


class TestComputeSummaryStats:
    def test_basic_stats(self) -> None:
        records = [_make_record(min_val_bpb=float(i)) for i in range(1, 6)]
        stats = compute_summary_stats(records)
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert "std" in stats

    def test_empty_input(self) -> None:
        stats = compute_summary_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == 0
        assert stats["median"] == 0


class TestComputeMarginalEffects:
    def test_detects_large_effect(self) -> None:
        """fp_weight axis has a huge effect; others are uniform."""
        records = []
        for i, fpw in enumerate([0.0, 0.1, 1.0]):
            # fp_weight=0.0 -> bpb=2.0, 0.1 -> 1.5, 1.0 -> 1.0
            bpb = 2.0 - fpw
            for j in range(10):
                records.append(_make_record(
                    config_index=i * 10 + j,
                    min_val_bpb=bpb + j * 0.001,  # tiny noise
                    fp_weight=fpw,
                    fs_distribution="geometric" if j < 5 else "normal",
                    fs_mean=8.0,
                    ns_mean=8.0,
                ))

        marginals = compute_marginal_effects(records)
        # fp_weight should have the highest eta-squared
        eta_values = {axis: m["eta_squared"] for axis, m in marginals.items()}
        strongest_axis, _ = max(eta_values.items(), key=lambda item: item[1])
        assert strongest_axis == "fp_weight"
        assert eta_values["fp_weight"] > 0.5


class TestComputeInteraction:
    def test_2d_means(self) -> None:
        records = []
        for fs_m in [4.0, 8.0, 16.0]:
            for ns_m in [4.0, 8.0, 16.0]:
                bpb = fs_m / 10 + ns_m / 10
                records.append(_make_record(
                    min_val_bpb=bpb,
                    fs_mean=fs_m,
                    ns_mean=ns_m,
                ))

        interaction = compute_interaction(records, "fs_mean", "ns_mean")
        assert interaction[str(8.0)][str(8.0)] == pytest.approx(1.6)
        assert interaction[str(4.0)][str(16.0)] == pytest.approx(2.0)


class TestComputeParetoFront:
    def test_basic_pareto(self) -> None:
        records = [
            _make_record(config_index=0, min_val_bpb=1.0, total_training_time=10.0),
            _make_record(config_index=1, min_val_bpb=1.5, total_training_time=20.0),
            _make_record(config_index=2, min_val_bpb=0.8, total_training_time=50.0),
            _make_record(config_index=3, min_val_bpb=1.2, total_training_time=40.0),
        ]
        pareto = compute_pareto_front(records)
        pareto_indices = {r["config_index"] for r in pareto}
        assert pareto_indices == {0, 2}

    def test_tie_on_time(self) -> None:
        records = [
            _make_record(config_index=0, min_val_bpb=1.0, total_training_time=10.0),
            _make_record(config_index=1, min_val_bpb=0.8, total_training_time=10.0),
        ]
        pareto = compute_pareto_front(records)
        pareto_indices = {r["config_index"] for r in pareto}
        assert pareto_indices == {1}

    def test_single_point(self) -> None:
        records = [_make_record(config_index=0)]
        pareto = compute_pareto_front(records)
        assert len(pareto) == 1

    def test_all_identical(self) -> None:
        records = [_make_record(config_index=i, min_val_bpb=1.0, total_training_time=10.0) for i in range(3)]
        pareto = compute_pareto_front(records)
        assert len(pareto) == 3


class TestFailedConfigsExcluded:
    def test_filter_excludes_errors(self) -> None:
        records = [
            _make_record(config_index=0),
            _make_record(config_index=1, error="OOM"),
            _make_record(config_index=2, min_val_bpb=None),
        ]
        valid = filter_valid(records)
        assert len(valid) == 1
        assert valid[0]["config_index"] == 0


_PNG_MAGIC = b"\x89PNG"


class TestPlotsGenerateFiles:
    def test_generate_all_creates_pngs(self, tmp_path: Path) -> None:
        # Create a synthetic dataset with variety across axes
        records = []
        idx = 0
        for fs_dist in ["geometric", "normal"]:
            for fs_m in [4.0, 8.0]:
                for ns_m in [4.0, 8.0]:
                    for fd_mode in [None, "prev_segment"]:
                        for fpw in [0.0, 0.1]:
                            records.append(_make_record(
                                config_index=idx,
                                min_val_bpb=1.0 + idx * 0.01,
                                val_bpb=1.0 + idx * 0.01,
                                total_training_time=10.0 + idx,
                                fs_distribution=fs_dist,
                                fs_mean=fs_m,
                                fs_std=fs_m if fs_dist == "geometric" else 1.0,
                                ns_mean=ns_m,
                                fd_mode=fd_mode,
                                fd_neg_delta_ratio=0.1 if fd_mode else None,
                                fp_weight=fpw,
                            ))
                            idx += 1

        output_dir = tmp_path / "plots"
        generate_all(records, output_dir)

        # Check all expected PNGs exist and are valid
        expected_pngs = [
            *[f"marginal_{axis}.png" for axis in ["fs_distribution", "fs_mean", "ns_mean", "fd_mode", "fp_weight"]],
            "heatmap_fs_mean_x_ns_mean.png",
            "heatmap_fd_mode_x_fp_weight.png",
            "heatmap_fs_distribution_x_fd_mode.png",
            "pareto_frontier.png",
        ]
        for name in expected_pngs:
            png_path = output_dir / name
            assert png_path.exists(), f"Missing {name}"
            # Verify valid PNG (non-empty with correct magic bytes)
            data = png_path.read_bytes()
            assert len(data) > 0, f"{name} is empty"
            assert data[:4] == _PNG_MAGIC, f"{name} is not a valid PNG"
