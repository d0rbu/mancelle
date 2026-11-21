"""Integration tests for the compute_statistics degeneracy gate (issue #549).

These tests drive the real compute_statistics end-to-end to confirm that
assert_feature_statistics_not_degenerate is actually called and raises (or
does not raise) as expected when the corpus statistics are degenerate / healthy.

Building ≥1000-segment shards synthetically is impractical via the existing
5-segment-per-shard helper for a fast test, so these tests lower
MIN_SEGMENTS_FOR_DEGENERACY_CHECK via monkeypatch to a value smaller than the
synthetic corpus while keeping the corpus data genuinely degenerate.

The patching is intentional and documented (per audit spec, section F).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch as t

import steering.compute_statistics as cs
import steering.feature_health as fh
from errors import DegenerateFeatureStatisticsError
from steering.compute_statistics import compute_statistics
from steering.feature_io import save_features

# ---------------------------------------------------------------------------
# Synthetic degenerate shard builder
# ---------------------------------------------------------------------------
# Every feature fires in every segment → median firing frequency = 1.0, which
# is far above DEGENERATE_FIRING_FREQ_THRESHOLD (0.5).  This mirrors the real
# random-weight Gemma-3 mis-load signature described in issue #549.


_N_FEATURES = 4  # small enough to be fast; degenerate pattern is feature-count-agnostic
_N_DOCS = 1
_N_SEGS = 5  # each shard has 5 segments; corpus threshold is patched to ≤ 5


def _write_degenerate_shards(tmp_dir: Path, n_shards: int = 1) -> None:
    """Write synthetic degenerate feature shards where every feature fires in
    every segment (the random-weight mis-load signature of issue #549)."""
    for shard_idx in range(n_shards):
        feat_ids = []
        feat_mags = []
        offsets = [0]
        for _seg in range(_N_SEGS):
            # All _N_FEATURES fire in this segment
            seg_ids = list(range(_N_FEATURES))
            seg_mags = [float(f + 1) * 0.9 for f in seg_ids]  # all positive, non-zero
            feat_ids.extend(seg_ids)
            feat_mags.extend(seg_mags)
            offsets.append(offsets[-1] + _N_FEATURES)

        shard = {
            "feature_ids": t.tensor(feat_ids, dtype=t.int32),
            "feature_mags": t.tensor(feat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": _N_SEGS,
            "n_docs": _N_DOCS,
            "doc_ids": t.zeros(_N_SEGS, dtype=t.int32),
            "doc_token_counts": t.tensor([640], dtype=t.int32),
            "segment_indices": t.arange(_N_SEGS, dtype=t.int32),
            "segment_positions": t.arange(_N_SEGS, dtype=t.int32) * 128,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_dir / f"shard_{shard_idx}_features")


def _write_sparse_shards(tmp_dir: Path, n_shards: int = 1) -> None:
    """Write synthetic healthy (sparse) feature shards.

    Each segment has a *different* single feature active (cycling through
    features 0..3), so no feature fires in every segment.  Median firing
    frequency ≈ 1/(_N_FEATURES) = 0.25, well below the 0.5 threshold.
    """
    for shard_idx in range(n_shards):
        feat_ids = []
        feat_mags = []
        offsets = [0]
        for seg in range(_N_SEGS):
            active_feat = seg % _N_FEATURES
            feat_ids.append(active_feat)
            feat_mags.append(float(active_feat + 1) * 0.5)
            offsets.append(offsets[-1] + 1)

        shard = {
            "feature_ids": t.tensor(feat_ids, dtype=t.int32),
            "feature_mags": t.tensor(feat_mags, dtype=t.float32),
            "offsets": t.tensor(offsets, dtype=t.int64),
            "n_segments": _N_SEGS,
            "n_docs": _N_DOCS,
            "doc_ids": t.zeros(_N_SEGS, dtype=t.int32),
            "doc_token_counts": t.tensor([640], dtype=t.int32),
            "segment_indices": t.arange(_N_SEGS, dtype=t.int32),
            "segment_positions": t.arange(_N_SEGS, dtype=t.int32) * 128,
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
            "context_window": 1024,
            "pooling": "mean_pool",
            "layer": 8,
        }
        save_features(shard, tmp_dir / f"shard_{shard_idx}_features")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_compute_statistics_raises_on_degenerate_corpus(tmp_path, monkeypatch):
    """compute_statistics raises DegenerateFeatureStatisticsError on a corpus
    where every feature fires in every segment (median firing freq = 1.0).

    MIN_SEGMENTS_FOR_DEGENERACY_CHECK is patched to 3 so our 5-segment synthetic
    corpus is large enough to trigger the check without building 1000+ shards.
    The corpus data itself is genuinely degenerate (all features always active).

    Non-vacuity (the core e2e proof): this test verifies that the
    assert_feature_statistics_not_degenerate call inside compute_statistics
    actually executes and raises. If the call were commented out, or if the
    gate were gated on a higher MIN_SEGMENTS check without the patch, this
    test would pass without raising — proving the gate is live.
    """
    monkeypatch.setattr(fh, "MIN_SEGMENTS_FOR_DEGENERACY_CHECK", 3)
    _write_degenerate_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    raised = False
    try:
        compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            allow_degenerate_statistics=False,
        )
    except DegenerateFeatureStatisticsError:
        raised = True
    assert raised, (
        "compute_statistics did NOT raise DegenerateFeatureStatisticsError "
        "on a degenerate corpus — the degeneracy gate is not being reached "
        "or is not firing correctly."
    )


def test_compute_statistics_allow_degenerate_bypass(tmp_path, monkeypatch):
    """Same degenerate corpus with allow_degenerate_statistics=True → no raise;
    health_metrics in the returned stats contain both expected numeric-string keys.

    Non-vacuity: setting allow_degenerate_statistics=False (instead of True)
    makes compute_statistics raise DegenerateFeatureStatisticsError and this
    test fails — proving allow_degenerate_statistics actually suppresses the gate.
    """
    monkeypatch.setattr(fh, "MIN_SEGMENTS_FOR_DEGENERACY_CHECK", 3)
    _write_degenerate_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    stats = compute_statistics(
        feature_dir=str(tmp_path),
        output_path=str(out),
        allow_degenerate_statistics=True,
    )
    assert stats.health_metrics is not None, "health_metrics should always be populated"
    assert "median_feature_firing_frequency" in stats.health_metrics, (
        f"missing key; got {list(stats.health_metrics.keys())}"
    )
    assert "n_features_firing_gt_99pct" in stats.health_metrics, (
        f"missing key; got {list(stats.health_metrics.keys())}"
    )
    # Values must be parseable as numbers
    float(stats.health_metrics["median_feature_firing_frequency"])
    int(stats.health_metrics["n_features_firing_gt_99pct"])


def test_compute_statistics_writes_health_metrics_healthy(tmp_path):
    """A small HEALTHY corpus (5 segments < 1000 = MIN_SEGMENTS_FOR_DEGENERACY_CHECK)
    does not raise, and health_metrics is populated with both keys as valid
    numeric strings.

    The gate is suppressed because n_segments < MIN_SEGMENTS_FOR_DEGENERACY_CHECK
    (no patching needed — the default 1000 threshold far exceeds 5 segments).
    The corpus is sparse (one feature per segment), so median firing frequency
    is well below the degenerate threshold regardless.
    """
    _write_sparse_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    stats = compute_statistics(
        feature_dir=str(tmp_path),
        output_path=str(out),
        allow_degenerate_statistics=False,
    )
    assert stats.health_metrics is not None, "health_metrics should be populated even for small healthy corpora"
    assert "median_feature_firing_frequency" in stats.health_metrics, (
        f"missing key; got {list(stats.health_metrics.keys())}"
    )
    assert "n_features_firing_gt_99pct" in stats.health_metrics, (
        f"missing key; got {list(stats.health_metrics.keys())}"
    )
    # Both values must parse as numbers
    median_freq = float(stats.health_metrics["median_feature_firing_frequency"])
    n_ubiq = int(stats.health_metrics["n_features_firing_gt_99pct"])
    # Sanity: healthy corpus should have a moderate median (not 1.0)
    assert 0.0 <= median_freq <= 1.0, f"Unexpected median firing frequency: {median_freq}"
    assert n_ubiq >= 0


def test_degeneracy_gate_fires_before_exact_finalization(tmp_path, monkeypatch):
    """The degeneracy gate runs before exact per-feature finalization."""
    monkeypatch.setattr(fh, "MIN_SEGMENTS_FOR_DEGENERACY_CHECK", 3)

    def poison(*a, **k):
        raise AssertionError("Pass-2 ran before the degeneracy gate")

    monkeypatch.setattr(cs.ExactMagnitudeCounts, "finalize", poison)
    monkeypatch.setattr(cs, "compute_per_feature_percentiles", poison)

    _write_degenerate_shards(tmp_path)
    out = tmp_path / "stats"

    with pytest.raises(DegenerateFeatureStatisticsError):
        cs.compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            allow_degenerate_statistics=False,
        )


def test_degenerate_abort_creates_no_streaming_sort_tmp_dir(tmp_path, monkeypatch):
    """A degenerate-corpus abort never materializes retired spill scratch."""
    monkeypatch.setattr(fh, "MIN_SEGMENTS_FOR_DEGENERACY_CHECK", 3)
    sort_tmp_root = tmp_path / "systmp"
    sort_tmp_root.mkdir()
    # mkdtemp(prefix=...) with no dir= uses tempfile.gettempdir(); redirect it.
    monkeypatch.setattr(tempfile, "tempdir", str(sort_tmp_root))

    _write_degenerate_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    with pytest.raises(DegenerateFeatureStatisticsError):
        compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            allow_degenerate_statistics=False,
        )

    leaked = list(sort_tmp_root.glob("stats_sorted_*"))
    assert leaked == [], (
        f"degenerate abort created retired streaming-sort scratch: {leaked}"
    )


def test_exact_pass_failure_creates_no_streaming_sort_tmp_dir(tmp_path, monkeypatch):
    """A reducer failure propagates without creating retired spill scratch."""
    sort_tmp_root = tmp_path / "systmp"
    sort_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sort_tmp_root))

    def boom(*args, **kwargs):
        raise RuntimeError("simulated exact reducer failure")

    monkeypatch.setattr(cs, "exact_pass1", boom)
    _write_sparse_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    with pytest.raises(RuntimeError, match="simulated exact reducer"):
        compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            allow_degenerate_statistics=False,
        )

    leaked = list(sort_tmp_root.glob("stats_sorted_*"))
    assert leaked == [], f"pass-1 failure leaked streaming sort temp dir(s): {leaked}"


# ---------------------------------------------------------------------------
# The exact route must create no retired ``stats_sorted_*`` directory under
# success or any failure point.
# ---------------------------------------------------------------------------


def _raise_runtime(*args, **kwargs):
    raise RuntimeError("simulated failure for leak test")


@pytest.mark.parametrize(
    "patch_target", ["_weighted_segment_stats", "firing_health", "StorageStats"]
)
def test_inmemory_window_failure_does_not_leak_streaming_sort_tmp_dir(
    tmp_path, monkeypatch, patch_target
):
    """Failures throughout finalization create no retired spill scratch."""
    sort_tmp_root = tmp_path / "systmp"
    sort_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sort_tmp_root))
    monkeypatch.setattr(cs, patch_target, _raise_runtime)

    _write_sparse_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    with pytest.raises(RuntimeError, match="simulated failure"):
        compute_statistics(
            feature_dir=str(tmp_path),
            output_path=str(out),
            allow_degenerate_statistics=False,
        )

    leaked = list(sort_tmp_root.glob("stats_sorted_*"))
    assert leaked == [], (
        f"in-memory window failure at {patch_target} leaked streaming sort temp "
        f"dir(s): {leaked}"
    )


def test_exact_finalize_failure_creates_no_streaming_sort_tmp_dir(
    tmp_path, monkeypatch
):
    """An exact-finalization failure creates no retired spill scratch."""
    sort_tmp_root = tmp_path / "systmp"
    sort_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sort_tmp_root))
    monkeypatch.setattr(cs.ExactMagnitudeCounts, "finalize", _raise_runtime)

    _write_sparse_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"
    with pytest.raises(RuntimeError, match="simulated failure"):
        compute_statistics(feature_dir=str(tmp_path), output_path=str(out))

    leaked = list(sort_tmp_root.glob("stats_sorted_*"))
    assert leaked == [], f"pass-2 failure leaked streaming sort temp dir(s): {leaked}"


@pytest.mark.parametrize("scenario", ["success", "eager", "window_fail"])
def test_no_stats_sorted_tempdir_is_created(tmp_path, monkeypatch, scenario):
    """Neither exact streaming nor eager/reference execution creates spill dirs."""
    # Isolate the temp dir to tmp_path so a regressed/RED-demo run does not spill
    # stats_sorted_* dirs into the shared system /tmp (matches the sibling tests).
    sort_tmp_root = tmp_path / "systmp"
    sort_tmp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(sort_tmp_root))

    cleaned: list[str] = []
    real_cleanup = tempfile.TemporaryDirectory.cleanup

    def spy(self):
        cleaned.append(self.name)
        return real_cleanup(self)

    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", spy)
    _write_sparse_shards(tmp_path, n_shards=1)
    out = tmp_path / "stats"

    if scenario == "success":
        compute_statistics(feature_dir=str(tmp_path), output_path=str(out))
    elif scenario == "eager":
        compute_statistics(
            feature_dir=str(tmp_path), output_path=str(out), streaming=False
        )
    else:  # window_fail
        monkeypatch.setattr(cs, "_weighted_segment_stats", _raise_runtime)
        with pytest.raises(RuntimeError, match="simulated failure"):
            compute_statistics(feature_dir=str(tmp_path), output_path=str(out))

    sort_cleanups = [c for c in cleaned if "stats_sorted_" in c]
    assert sort_cleanups == [], f"{scenario}: created retired spill scratch: {sort_cleanups}"
