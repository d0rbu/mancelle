"""Tests for the per-feature activation-frequency histogram script.

Covers binning correctness, threshold counting, edge cases (zero segments,
no active features), end-to-end Hydra run, output schema, and determinism.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from matplotlib.container import BarContainer

from tests.steering._freq_blacklist_fixtures import make_stats, write_stats
from viz.plot_feature_activation_frequency import (
    compute_frequency_histogram,
    plot_frequency_histogram,
    threshold_summary,
    write_csv,
)
from viz.plot_feature_activation_frequency import (
    run as script_run,
)


class TestComputeFrequencyHistogram:
    def test_known_counts(self):
        # 5 active features at counts 1, 10, 50, 100, 999 with n_segments=1000:
        # freqs = 0.001, 0.01, 0.05, 0.10, 0.999
        stats = make_stats(
            n_segments=1000,
            counts=dict(enumerate([999, 100, 50, 10, 1])),
            n_sae_features=64,
        )
        freqs, edges, counts = compute_frequency_histogram(stats, n_bins=20)
        assert sorted(freqs.tolist()) == pytest.approx(
            [0.001, 0.01, 0.05, 0.10, 0.999]
        )
        # 20-bin log scale from 1/1000=1e-3 to 1.0; every freq fits
        assert counts.sum() == 5
        assert edges.shape == (21,)
        assert edges[0] == pytest.approx(1e-3)
        assert edges[-1] == pytest.approx(1.0)

    def test_bins_log_spaced(self):
        stats = make_stats(
            n_segments=10_000, counts={0: 1}, n_sae_features=8
        )
        _, edges, _ = compute_frequency_histogram(stats, n_bins=10)
        # log-spaced ⇒ consecutive ratios constant
        ratios = edges[1:] / edges[:-1]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)

    def test_n_segments_one_does_not_crash(self):
        # Degenerate 1-segment corpus: only freq 1.0 is possible; log-bin
        # edges collapse but numpy buckets the lone value without raising.
        stats = make_stats(n_segments=1, counts={3: 1}, n_sae_features=8)
        freqs, edges, counts = compute_frequency_histogram(stats, n_bins=20)
        assert freqs.tolist() == [1.0]
        assert int(counts.sum()) == 1

    def test_n_segments_zero_returns_empty(self):
        stats = make_stats(
            n_segments=0, counts={0: 1}, n_sae_features=8
        )
        freqs, edges, counts = compute_frequency_histogram(stats, n_bins=10)
        assert freqs.size == 0 and edges.size == 0 and counts.size == 0

    def test_no_active_features(self):
        stats = make_stats(
            n_segments=1000, counts={}, n_sae_features=64
        )
        freqs, edges, counts = compute_frequency_histogram(stats, n_bins=10)
        assert freqs.size == 0
        assert edges.shape == (11,) and counts.sum() == 0

    def test_all_features_in_one_high_bin(self):
        # all 4 features at count=1000 (freq=1.0) → all in the last bin
        stats = make_stats(
            n_segments=1000,
            counts=dict.fromkeys(range(4), 1000),
            n_sae_features=16,
        )
        _, edges, counts = compute_frequency_histogram(stats, n_bins=10)
        assert counts[-1] == 4
        assert counts[:-1].sum() == 0

    def test_n_bins_validation(self):
        stats = make_stats(
            n_segments=10, counts={0: 5}, n_sae_features=8
        )
        with pytest.raises(ValueError, match="n_bins"):
            compute_frequency_histogram(stats, n_bins=0)

    def test_bin_count_total_equals_active_features(self):
        # invariant: histogram captures every active feature
        stats = make_stats(
            n_segments=1000,
            counts={i: (i + 1) for i in range(20)},
            n_sae_features=64,
        )
        freqs, _, counts = compute_frequency_histogram(stats, n_bins=30)
        assert counts.sum() == len(freqs) == 20

    def test_deterministic(self):
        stats = make_stats(
            n_segments=100, counts={1: 10, 2: 20, 3: 30}, n_sae_features=16
        )
        a = compute_frequency_histogram(stats, n_bins=15)
        b = compute_frequency_histogram(stats, n_bins=15)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])


class TestThresholdSummary:
    def test_strict_greater(self):
        freqs = np.array([0.0, 0.01, 0.05, 0.5, 1.0])
        out = threshold_summary(freqs, threshold_lines=[0.0, 0.01, 0.1, 1.0])
        # >0.0 → 4 ; >0.01 → 3 ; >0.1 → 2 ; >1.0 → 0
        assert out == [(0.0, 4), (0.01, 3), (0.1, 2), (1.0, 0)]

    def test_empty(self):
        assert threshold_summary(
            np.zeros(0), threshold_lines=[0.01]
        ) == [(0.01, 0)]


class TestPlot:
    def test_renders_without_error(self):
        stats = make_stats(
            n_segments=1000,
            counts={i: (i + 1) * 10 for i in range(5)},
            n_sae_features=16,
        )
        _, edges, counts = compute_frequency_histogram(stats, n_bins=20)
        fig = plot_frequency_histogram(
            stats, edges, counts,
            threshold_lines=[0.01, 0.1], title="test",
        )
        assert fig is not None
        # corpus summary text appears in some text artist
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("n_sae_features" in t for t in texts)
        assert any("pooling" in t for t in texts)

    def test_empty_histogram_renders(self):
        stats = make_stats(
            n_segments=1000, counts={}, n_sae_features=16
        )
        _, edges, counts = compute_frequency_histogram(stats, n_bins=20)
        fig = plot_frequency_histogram(
            stats, edges, counts, threshold_lines=[0.01], title="empty"
        )
        assert fig is not None  # does not crash

    def test_zero_segments_no_bar_drawn(self):
        """n_segments == 0 ⇒ compute returns counts.size == 0; plot must
        skip the bar() call (covers the `if counts.size:` False branch)."""
        stats = make_stats(
            n_segments=0, counts={1: 1}, n_sae_features=16
        )
        _, edges, counts = compute_frequency_histogram(stats, n_bins=20)
        assert counts.size == 0
        fig = plot_frequency_histogram(
            stats, edges, counts, threshold_lines=[0.01], title="zero-seg"
        )
        assert not any(
            isinstance(c, BarContainer) for c in fig.axes[0].containers
        )


class TestWriteCsv:
    def test_schema(self, tmp_path):
        edges = np.array([0.001, 0.01, 0.1, 1.0])
        counts = np.array([5, 10, 2])
        out = tmp_path / "h.csv"
        write_csv(out, edges, counts)
        with out.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["bin_lo", "bin_hi", "n_features"]
        assert len(rows) == 4
        assert int(rows[3][2]) == 2

    def test_bin_edges_are_correct(self, tmp_path):
        # Pin the bin_lo/bin_hi VALUE columns (not just counts): each row's
        # (lo, hi) must be the consecutive edge pair and strictly increasing.
        edges = np.array([0.001, 0.01, 0.1, 1.0])
        counts = np.array([5, 10, 2])
        out = tmp_path / "h.csv"
        write_csv(out, edges, counts)
        with out.open() as f:
            rows = list(csv.reader(f))[1:]
        assert [float(r[0]) for r in rows] == pytest.approx(edges[:-1].tolist())
        assert [float(r[1]) for r in rows] == pytest.approx(edges[1:].tolist())
        for r in rows:
            assert float(r[0]) < float(r[1])


def _cfg_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "configs" / "extraction")


class TestEndToEndScript:
    def test_run_produces_png_and_csv(self, tmp_path):
        write_stats(
            tmp_path,
            n_segments=1000,
            counts={1: 5, 2: 50, 3: 500},
            n_sae_features=16,
        )
        with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
            cfg = compose(
                config_name="plot_feature_activation_frequency",
                overrides=[f"feature_dir={tmp_path}"],
            )
        out = script_run(cfg)
        png = Path(out["png"])
        csvp = Path(out["csv"])
        assert png.is_file() and png.stat().st_size > 1024  # non-trivial PNG
        with png.open("rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"  # PNG magic
        with csvp.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["bin_lo", "bin_hi", "n_features"]
        total = sum(int(r[2]) for r in rows[1:])
        assert total == 3  # 3 active features

    def test_run_custom_threshold_lines(self, tmp_path):
        write_stats(
            tmp_path,
            n_segments=1000,
            counts={1: 5, 2: 500, 3: 999},
            n_sae_features=16,
        )
        with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
            cfg = compose(
                config_name="plot_feature_activation_frequency",
                overrides=[
                    f"feature_dir={tmp_path}",
                    "threshold_lines=[0.4,0.9]",
                    "n_bins=10",
                ],
            )
        out = script_run(cfg)
        assert Path(out["png"]).is_file()

    def test_csv_deterministic_across_runs(self, tmp_path):
        write_stats(
            tmp_path,
            n_segments=100,
            counts={1: 10, 2: 20, 3: 30},
            n_sae_features=16,
        )
        with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
            cfg = compose(
                config_name="plot_feature_activation_frequency",
                overrides=[
                    f"feature_dir={tmp_path}",
                    f"output_dir={tmp_path}/a",
                ],
            )
        a = Path(script_run(cfg)["csv"]).read_bytes()
        with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
            cfg = compose(
                config_name="plot_feature_activation_frequency",
                overrides=[
                    f"feature_dir={tmp_path}",
                    f"output_dir={tmp_path}/b",
                ],
            )
        b = Path(script_run(cfg)["csv"]).read_bytes()
        assert a == b

    def test_missing_stats_file_raises(self, tmp_path):
        with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
            cfg = compose(
                config_name="plot_feature_activation_frequency",
                overrides=[f"feature_dir={tmp_path}"],
            )
        with pytest.raises(FileNotFoundError, match="feature_statistics"):
            script_run(cfg)

    def test_custom_output_dir(self, tmp_path):
        feat_dir = tmp_path / "feat"
        write_stats(
            feat_dir, n_segments=100, counts={1: 50}, n_sae_features=8
        )
        out_dir = tmp_path / "out"
        with initialize_config_dir(config_dir=_cfg_dir(), version_base=None):
            cfg = compose(
                config_name="plot_feature_activation_frequency",
                overrides=[
                    f"feature_dir={feat_dir}",
                    f"output_dir={out_dir}",
                ],
            )
        out = script_run(cfg)
        assert Path(out["png"]).parent == out_dir
        assert (out_dir / "feature_activation_frequency_hist.png").is_file()


def test_compute_matches_pure_python_oracle(tmp_path):
    """Cross-check the implementation against an independent pure-python
    binning oracle (no numpy.histogram)."""
    counts_map = {0: 1, 1: 5, 2: 25, 3: 250, 4: 999}
    n_segments = 1000
    stats = make_stats(
        n_segments=n_segments, counts=counts_map, n_sae_features=16
    )
    n_bins = 12
    freqs, edges, counts = compute_frequency_histogram(stats, n_bins=n_bins)
    expected_freqs = sorted(c / n_segments for c in counts_map.values())
    assert sorted(freqs.tolist()) == pytest.approx(expected_freqs)
    # Independent binning
    expected_counts = [0] * n_bins
    for f in freqs:
        # last bin is right-closed (numpy histogram default)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = (f >= lo and f < hi) or (i == n_bins - 1 and f == hi)
            if in_bin:
                expected_counts[i] += 1
                break
    assert counts.tolist() == expected_counts
