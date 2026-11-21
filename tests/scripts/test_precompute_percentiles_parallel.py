"""Tests for multi-GPU parallel precompute_percentiles via RANK/WORLD_SIZE striping.

Tests cover:
1. Two simulated ranks process disjoint shard subsets, output matches single-process
2. Resume works across ranks: pre-existing overlays are skipped
3. Backward compatibility: no RANK/WORLD_SIZE env vars → single-process behavior
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file as sf_save

import scripts.preprocess.precompute_percentiles as producer_module
from scripts.preprocess.precompute_percentiles import run
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
)
from steering.feature_io import (
    TensorKey,
    glob_features,
    load_features,
    save_features,
    shard_stem_from_path,
)
from steering.overlay_io import (
    OverlayKind,
    discover_overlay,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
)
from steering.stats_io import load_statistics, save_statistics

N_SAE = 20  # Small for fast tests


def _make_feature_dir(tmp_path: Path, n_shards: int = 6) -> Path:
    """Create a feature directory with valid statistics and shards.

    Returns the feature directory path. The stats file contains per-feature
    percentile data that build_percentile_table can consume.

    The directory name embeds ``s{segment_size}`` so that
    ``verify_segment_size`` can cross-check the cfg against the on-disk
    convention (segment_size=128 here, matching the shard metadata
    written by this helper).
    """
    feature_dir = tmp_path / "features" / "s128"
    feature_dir.mkdir(parents=True)

    t.manual_seed(42)

    # Build per-feature stat tensors
    fids = t.arange(N_SAE, dtype=t.int32)
    counts = t.full((N_SAE,), 100, dtype=t.int64)
    pct_segments_active = t.full((N_SAE,), 0.5, dtype=t.float32)
    mean_mag = t.rand(N_SAE, dtype=t.float32) + 0.1
    mean_mag_incl_zeros = mean_mag * 0.5
    std_mag = t.rand(N_SAE, dtype=t.float32) * 0.1
    min_mag = t.full((N_SAE,), 0.01, dtype=t.float32)
    max_mag = mean_mag + 1.0
    median_mag = mean_mag.clone()

    # Percentile values: monotonically increasing [N_SAE, 99]
    percentile_values = t.rand(N_SAE, 99, dtype=t.float64).cumsum(dim=1)

    pct_names = ",".join(f"p{k}" for k in range(1, 100))

    tensors = {
        "feature_ids": fids,
        "counts": counts,
        "pct_segments_active": pct_segments_active,
        "mean_mag": mean_mag,
        "mean_mag_incl_zeros": mean_mag_incl_zeros,
        "std_mag": std_mag,
        "min_mag": min_mag,
        "max_mag": max_mag,
        "median_mag": median_mag,
        "percentile_values": percentile_values,
    }
    metadata = {
        "n_sae_features": str(N_SAE),
        "n_shards_processed": str(n_shards),
        "n_docs": str(n_shards),
        "n_segments": str(n_shards),
        "layer": "8",
        "segment_size": "64",
        "context_window": "1024",
        "pooling_mode": "mean_pool",
        "segment_mode": "fixed_tokens",
        "generated_at": "",
        "activation_percentile": "0.0",
        "n_tokens_gpt2": "0",
        "mean_tokens_per_doc": "0",
        "median_tokens_per_doc": "0",
        "n_features_seen": str(N_SAE),
        "mean_active_per_segment": "0",
        "median_active_per_segment": "0",
        "p90_active_per_segment": "0",
        "max_active_per_segment": "0",
        "min_active_per_segment": "0",
        "n_total_sae_features": str(N_SAE),
        "pct_features_active": "0",
        "total_bytes": "0",
        "mean_bytes_per_doc": "0",
        "mean_bytes_per_token": "0",
        "extrap_target_tokens": "0",
        "extrap_estimated_docs": "0",
        "extrap_estimated_storage_gb": "0",
        "extrap_estimated_wall_time_hours": "0",
        "percentile_names": pct_names,
    }

    stats_path = feature_dir / "feature_statistics.safetensors"
    sf_save(tensors, str(stats_path), metadata=metadata)

    # Create feature shards with random data
    for i in range(n_shards):
        n = 15
        shard_fids = t.randint(0, N_SAE, (n,), dtype=t.int32)
        shard_mags = t.rand(n) + 0.1
        data = {
            "offsets": t.tensor([0, n], dtype=t.int64),
            "feature_ids": shard_fids,
            "feature_mags": shard_mags,
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "doc_token_counts": t.tensor([100], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor([64], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 1,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        }
        save_features(data, feature_dir / f"shard_{i:05d}")

    return feature_dir


def _make_cfg(feature_dir: Path, **overrides) -> DictConfig:
    """Create an OmegaConf config for run()."""
    defaults = {
        "feature_dir": str(feature_dir),
        "model": "gemma-2-2b",
        "activation_type": "residual",
        "layer": 8,
        "pooling": "max_pool",
        "segment_size": 128,
        "sae_width": "16k",
        "dataset_phase": None,
        "feature_root": "out/data/features",
        "device": "cpu",
        "resume": False,
        "dry_run": False,
        "force": False,
        "zstd_level": 12,
    }
    defaults.update(overrides)
    return OmegaConf.create(defaults)


def _inventory(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class TestParallelShardStriping:
    """Verify RANK/WORLD_SIZE shard striping produces correct, disjoint output."""

    def test_two_ranks_cover_all_shards(self, tmp_path, monkeypatch):
        """Two simulated ranks process disjoint subsets; union covers all shards."""

        feature_dir = _make_feature_dir(tmp_path / "r0", n_shards=6)
        cfg = _make_cfg(feature_dir)

        # Run rank 0 of 2
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        ret = run(cfg)
        assert ret == 0

        # Run rank 1 of 2
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        cfg_values = OmegaConf.to_container(cfg)
        assert isinstance(cfg_values, dict)
        ret = run(OmegaConf.create({**cfg_values, "resume": True}))
        assert ret == 0

        # All 6 overlay shards should exist
        all_shards = glob_features(feature_dir)
        assert len(all_shards) == 6

        # Find the overlay dir (checksum-based)


        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

        overlay_shards = glob_overlay_shards(out_dir)
        assert len(overlay_shards) == 6

    def test_output_matches_single_process(self, tmp_path, monkeypatch):
        """Parallel output is byte-identical to single-process output."""
        from scripts.preprocess.precompute_percentiles import run


        n_shards = 6

        # Single process run
        feature_dir_single = _make_feature_dir(tmp_path / "single", n_shards=n_shards)
        cfg_single = _make_cfg(feature_dir_single)
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert run(cfg_single) == 0

        # Parallel run (2 ranks)
        feature_dir_par = _make_feature_dir(tmp_path / "parallel", n_shards=n_shards)
        cfg_par = _make_cfg(feature_dir_par)

        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert run(cfg_par) == 0

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "1")
        cfg_values = OmegaConf.to_container(cfg_par)
        assert isinstance(cfg_values, dict)
        assert run(OmegaConf.create({**cfg_values, "resume": True})) == 0

        # Compare overlay shards
        stats = load_statistics(feature_dir_single / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)

        single_dir = overlay_dir(feature_dir_single, OverlayKind.PERCENTILE, checksum)
        par_dir = overlay_dir(feature_dir_par, OverlayKind.PERCENTILE, checksum)

        for shard_path in glob_features(feature_dir_single):
            stem = shard_stem_from_path(shard_path)
            single_ovl = load_overlay_shard(overlay_shard_path(single_dir, stem))
            par_ovl = load_overlay_shard(overlay_shard_path(par_dir, stem))

            single_pcts = single_ovl[TensorKey.FEATURE_PERCENTILES]
            par_pcts = par_ovl[TensorKey.FEATURE_PERCENTILES]
            assert t.equal(single_pcts, par_pcts), f"Mismatch in {stem}"

    def test_ranks_get_disjoint_shards(self, tmp_path, monkeypatch):
        """Rank 0 gets even shards, rank 1 gets odd shards (round-robin)."""
        from scripts.preprocess.precompute_percentiles import run


        feature_dir = _make_feature_dir(tmp_path / "disjoint", n_shards=6)
        cfg = _make_cfg(feature_dir)

        # Run only rank 0 (no resume needed — first run)
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert run(cfg) == 0

        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

        # Rank 0 should have shards 0, 2, 4
        assert overlay_shard_path(out_dir, "shard_00000").exists()
        assert overlay_shard_path(out_dir, "shard_00002").exists()
        assert overlay_shard_path(out_dir, "shard_00004").exists()

        # Rank 0 should NOT have shards 1, 3, 5
        assert not overlay_shard_path(out_dir, "shard_00001").exists()
        assert not overlay_shard_path(out_dir, "shard_00003").exists()
        assert not overlay_shard_path(out_dir, "shard_00005").exists()


class TestResumeAcrossRanks:
    """Resume skips already-processed shards regardless of which rank wrote them."""

    def test_resume_skips_existing_overlays(self, tmp_path, monkeypatch):
        """Pre-existing overlay shards are skipped on resume."""
        from scripts.preprocess.precompute_percentiles import run


        feature_dir = _make_feature_dir(tmp_path / "resume", n_shards=6)
        cfg = _make_cfg(feature_dir, resume=True)

        # First: rank 0 processes shards 0, 2, 4
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert run(cfg) == 0

        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

        # Record mtimes of rank 0's shards
        mtime_0 = overlay_shard_path(out_dir, "shard_00000").stat().st_mtime
        mtime_2 = overlay_shard_path(out_dir, "shard_00002").stat().st_mtime

        # Re-run rank 0 with resume — should skip all its shards
        assert run(cfg) == 0

        # mtimes unchanged (files not rewritten)
        assert overlay_shard_path(out_dir, "shard_00000").stat().st_mtime == mtime_0
        assert overlay_shard_path(out_dir, "shard_00002").stat().st_mtime == mtime_2

    def test_resume_with_different_world_size(self, tmp_path, monkeypatch):
        """Crash-recovery with changed WORLD_SIZE still works via file-exists check."""
        from scripts.preprocess.precompute_percentiles import run


        feature_dir = _make_feature_dir(tmp_path / "ws_change", n_shards=6)

        # First run: single process handles all 6
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        cfg = _make_cfg(feature_dir)
        assert run(cfg) == 0

        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

        # All 6 exist
        assert len(glob_overlay_shards(out_dir)) == 6

        # Re-run with WORLD_SIZE=4, rank 0, resume=true — should skip all
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "0")
        cfg_resume = _make_cfg(feature_dir, resume=True)
        assert run(cfg_resume) == 0

        # Still 6 — nothing new written
        assert len(glob_overlay_shards(out_dir)) == 6


class TestBackwardCompatibility:
    """Without RANK/WORLD_SIZE env vars, behavior is identical to before."""

    def test_no_env_vars_processes_all_shards(self, tmp_path, monkeypatch):
        """Single process with no env vars handles all shards."""
        from scripts.preprocess.precompute_percentiles import run


        # Ensure no stale env vars
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        feature_dir = _make_feature_dir(tmp_path / "compat", n_shards=6)
        cfg = _make_cfg(feature_dir)
        assert run(cfg) == 0

        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)

        assert len(glob_overlay_shards(out_dir)) == 6

    def test_manifest_written_by_default(self, tmp_path, monkeypatch):
        """Manifest is written when no env vars are set (rank defaults to 0)."""
        from scripts.preprocess.precompute_percentiles import run


        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        feature_dir = _make_feature_dir(tmp_path / "manifest", n_shards=2)
        cfg = _make_cfg(feature_dir)
        assert run(cfg) == 0

        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)

        assert discover_overlay(feature_dir, OverlayKind.PERCENTILE, checksum) is not None

    def test_non_zero_rank_skips_manifest(self, tmp_path, monkeypatch):
        """Non-zero ranks do not write the manifest."""
        from scripts.preprocess.precompute_percentiles import run


        feature_dir = _make_feature_dir(tmp_path / "no_manifest", n_shards=4)

        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("WORLD_SIZE", "2")
        monkeypatch.setenv("LOCAL_RANK", "1")
        cfg = _make_cfg(feature_dir)
        assert run(cfg) == 0

        stats = load_statistics(feature_dir / "feature_statistics.safetensors")
        pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
        checksum = compute_percentile_checksum(pct_table)

        # Overlay dir exists (shards were written) but no manifest
        out_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        assert out_dir.is_dir()
        assert not (out_dir / "manifest.json").exists()


def test_percentiles_writes_and_reuses_base_hash_cache(tmp_path, monkeypatch):
    import steering.overlay_io as oio
    from steering.overlay_io import (
        BASE_SHARD_HASH_CACHE_FILENAME,
        compute_base_shard_content_hash,
        read_manifest,
    )

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    feature_dir = _make_feature_dir(tmp_path / "f", n_shards=4)
    cfg = _make_cfg(feature_dir)

    assert run(cfg) == 0
    cache = feature_dir / BASE_SHARD_HASH_CACHE_FILENAME
    assert cache.is_file(), "first run must write the base-hash cache in feature_dir"

    # Byte-identity: stored hash equals the uncached streaming reference.
    stats = load_statistics(feature_dir / "feature_statistics.safetensors")
    pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
    checksum = compute_percentile_checksum(pct_table)
    odir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    manifest = read_manifest(odir)
    assert manifest is not None
    assert manifest["base_shard_content_hash"] == compute_base_shard_content_hash(
        glob_features(feature_dir), cache_dir=None
    )

    # Teeth: a forced re-run hits the cache (0 stream calls).
    calls = {"n": 0}
    orig = oio._stream_base_hash
    monkeypatch.setattr(oio, "_stream_base_hash",
                        lambda p: (calls.__setitem__("n", calls["n"] + 1), orig(p))[1])
    assert run(_make_cfg(feature_dir, force=True)) == 0
    assert calls["n"] == 0, "second run must hit the cache, reading zero base-shard bytes"


def test_incomplete_percentile_overlay_requires_resume(tmp_path, monkeypatch):
    """F1: a valid manifest never turns missing payloads into a complete no-op."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    feature_dir = _make_feature_dir(tmp_path / "incomplete", n_shards=2)
    assert run(_make_cfg(feature_dir)) == 0

    stats = load_statistics(feature_dir / "feature_statistics.safetensors")
    pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
    checksum = compute_percentile_checksum(pct_table)
    output = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    missing = overlay_shard_path(output, "shard_00001")
    missing.unlink()

    before_failure = _inventory(feature_dir)
    with pytest.raises(RuntimeError, match="incomplete"):
        run(_make_cfg(feature_dir))
    assert _inventory(feature_dir) == before_failure
    assert not missing.exists()

    assert run(_make_cfg(feature_dir, resume=True)) == 0
    assert missing.is_file()


def test_percentile_resume_rejects_corrupt_present_payload(tmp_path, monkeypatch):
    """F4: resume validates a present payload instead of blindly skipping it."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    feature_dir = _make_feature_dir(tmp_path / "corrupt", n_shards=1)
    assert run(_make_cfg(feature_dir)) == 0

    stats = load_statistics(feature_dir / "feature_statistics.safetensors")
    pct_table = build_percentile_table(stats.per_feature_stats, N_SAE)
    checksum = compute_percentile_checksum(pct_table)
    payload = overlay_shard_path(
        overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum),
        "shard_00000",
    )
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: t.zeros(1, dtype=t.uint8)}, payload
    )

    before_failure = _inventory(feature_dir)
    with pytest.raises(ValueError, match="numel"):
        run(_make_cfg(feature_dir, resume=True))
    assert _inventory(feature_dir) == before_failure


def test_cpu_run_uses_its_built_table_without_changing_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    feature_dir = _make_feature_dir(tmp_path, n_shards=1)
    data = load_features(glob_features(feature_dir)[0])
    stats = load_statistics(feature_dir / "feature_statistics.safetensors")
    table = build_percentile_table(stats.per_feature_stats, N_SAE)
    original = producer_module.precompute_shard
    observed = []

    def observe(*args, **kwargs):
        observed.append(kwargs.get("_table_is_finite", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(producer_module, "precompute_shard", observe)
    assert producer_module.run(_make_cfg(feature_dir)) == 0
    expected = percentile_to_uint8(magnitude_to_percentile(
        data[TensorKey.FEATURE_IDS], data[TensorKey.FEATURE_MAGS], table,
    ))
    checksum = compute_percentile_checksum(table)
    payload = overlay_shard_path(
        overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum), "shard_00000"
    )
    assert t.equal(load_overlay_shard(payload)["feature_percentiles"], expected)
    assert observed == [True]


@pytest.mark.parametrize("percentile", ["p1", "p50", "p99"])
def test_run_rejects_nan_statistics_before_writing_outputs(tmp_path, monkeypatch, percentile):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    feature_dir = _make_feature_dir(tmp_path, n_shards=1)
    stats_path = feature_dir / "feature_statistics.safetensors"
    stats = load_statistics(stats_path)
    next(iter(stats.per_feature_stats.values())).percentiles[percentile] = float("nan")
    save_statistics(stats, stats_path)
    before = _inventory(feature_dir)

    with pytest.raises(ValueError, match="non-monotonic percentile values"):
        producer_module.run(_make_cfg(feature_dir))

    assert _inventory(feature_dir) == before
