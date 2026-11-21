"""Tests that precompute scripts save their checksums to the statistics file.

These tests would have caught the bug where precompute_percentiles computed a
percentile overlay but never called save_percentile_checksum(), causing
precompute_packed_rows to fail with "No percentile checksum found".

Tests cover:
1. End-to-end: run() saves checksum that matches the overlay directory name
2. Checksum matches overlay directory — overlay dir exists with correct shard count
3. Checksum survives re-read from disk
4. Idempotency: running twice with resume=true produces same checksum
5. Packed rows chain: load_percentile_checksum returns checksum, overlay dir exists
6. Token boundaries parallel: save_tokenizer_checksum is called by run()
7. Checksum NOT written on failure
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import save_features
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    OverlayKind,
    glob_overlay_shards,
    overlay_dir,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from steering.stats_io import (
    _read_metadata,
    load_percentile_checksum,
    load_tokenizer_checksum,
    save_statistics,
)
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)

N_SAE = 200
N_FEATURES_SEEN = 10
SEGMENT_SIZE = 16


# ---------------------------------------------------------------------------
# Helpers — percentile fixture
# ---------------------------------------------------------------------------


def _make_stats_file(feature_dir: Path, n_sae: int = N_SAE) -> Path:
    """Create a minimal but valid statistics file in feature_dir.

    The file is loadable by ``load_statistics()`` and contains per-feature
    percentile data needed by ``build_percentile_table()``.
    """
    per_feature_stats: dict[str, PerFeatureStat] = {}
    for fid in range(N_FEATURES_SEEN):
        pcts = {f"p{k}": float(k) * 0.1 + fid * 0.01 for k in range(1, 100)}
        per_feature_stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.5,
            min_mag=0.1,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=pcts,
        )

    stats = ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="mean_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=64,
            context_window=1024,
            layer=8,
            n_sae_features=n_sae,
            generated_at="test",
            activation_percentile=0.0,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=2,
            n_tokens_gpt2=200,
            mean_tokens_per_doc=100.0,
            median_tokens_per_doc=100,
        ),
        features=SegmentFeatureStats(
            n_segments=10,
            n_features_seen=N_FEATURES_SEEN,
            mean_active_per_segment=5.0,
            median_active_per_segment=5,
            p90_active_per_segment=8,
            max_active_per_segment=10,
            min_active_per_segment=1,
            n_total_sae_features=n_sae,
            pct_features_active=0.5,
            count_percentiles={},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1000,
            mean_bytes_per_doc=500.0,
            mean_bytes_per_token=5.0,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=100.0,
            estimated_wall_time_hours=10.0,
        ),
    )
    return save_statistics(stats, feature_dir / "feature_statistics")


def _make_feature_shard(
    feature_dir: Path, shard_name: str, n_sae: int = N_SAE, seed: int = 42,
) -> Path:
    """Create a minimal feature shard with deterministic data."""
    t.manual_seed(seed)
    n_features = 30
    # Every emitted ID must have a real statistics row. Missing rows are a
    # fail-loud production error, and this fixture tests checksum wiring rather
    # than that independently covered error path.
    fids = t.randint(0, min(n_sae, N_FEATURES_SEEN), (n_features,), dtype=t.int32)
    mags = t.rand(n_features) + 0.1
    offsets = t.tensor([0, 15, n_features], dtype=t.int64)

    data = {
        "offsets": offsets,
        "feature_ids": fids,
        "feature_mags": mags,
        "doc_ids": t.tensor([0, 1], dtype=t.int32),
        "segment_indices": t.tensor([0, 0], dtype=t.int32),
        "segment_positions": t.tensor([0, 0], dtype=t.int32),
        "doc_token_counts": t.tensor([100, 100], dtype=t.int32),
        "segment_char_starts": t.tensor([0, 0], dtype=t.int32),
        "segment_char_ends": t.tensor([64, 64], dtype=t.int32),
        "n_docs": 2,
        "n_segments": 2,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 64,
    }
    return save_features(data, feature_dir / shard_name)


def _make_percentile_env(
    tmp_path: Path, n_shards: int = 2,
) -> tuple[Path, Path]:
    """Create a complete test environment with feature shards + stats file.

    Returns (feature_dir, stats_path).

    The feature directory name embeds ``s{segment_size}`` so that
    ``verify_segment_size`` can cross-check the cfg against the on-disk
    convention (``_parse_segment_size_from_path``); without that
    component the check raises.
    """
    feature_dir = tmp_path / "features" / "s64"
    feature_dir.mkdir(parents=True)

    stats_path = _make_stats_file(feature_dir)

    for i in range(n_shards):
        _make_feature_shard(feature_dir, f"shard_{i:05d}", seed=42 + i)

    return feature_dir, stats_path


def _percentile_cfg(feature_dir: Path, **overrides) -> DictConfig:
    """Create a minimal DictConfig for precompute_percentiles.run().

    ``segment_size`` matches the value embedded in ``_make_feature_shard``
    (64); ``precompute_percentiles.run`` calls
    ``verify_segment_size(feature_dir, expected_segment_size=cfg.segment_size)``
    at the top so the key must be present and consistent.
    """
    cfg = {
        "feature_dir": str(feature_dir),
        "segment_size": 64,
        "device": "cpu",
        "dry_run": False,
        "resume": False,
        "force": False,
        "zstd_level": 12,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# Helpers — token boundary fixture
# ---------------------------------------------------------------------------


class _SimpleTokenizer:
    """Deterministic char-level tokenizer for testing token boundary wiring.

    Token 0 = BOS = "<bos>" (5 chars).
    Token n (n > 0) = f"t{n:03d} " (5 chars each).
    """

    def get_bos_token_id(self):
        return 0

    def get_vocab_size(self):
        return 1000

    def encode(self, text, prepend=None, num_threads=None):
        if isinstance(text, list):
            return [self.encode(t_text, prepend=prepend) for t_text in text]
        words = text.split()
        tokens = [i + 1 for i in range(len(words))]
        if prepend is not None:
            tokens.insert(0, prepend if isinstance(prepend, int) else 0)
        return tokens

    def decode(self, token_ids):
        parts = []
        for tid in token_ids:
            if tid == 0:
                parts.append("<bos>")
            else:
                parts.append(f"t{tid:03d} ")
        return "".join(parts)


def _make_token_boundary_env(
    tmp_path: Path,
) -> tuple[Path, Path, Path, _SimpleTokenizer]:
    """Create a test environment for precompute_token_boundaries.run().

    Returns (feature_dir, stats_path, data_dir, tokenizer).
    """
    tokenizer = _SimpleTokenizer()

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    # ``s{SEGMENT_SIZE}`` component required by ``verify_segment_size``.
    feature_dir = tmp_path / "features" / f"s{SEGMENT_SIZE}"
    feature_dir.mkdir(parents=True)

    n_docs = 2
    n_segs_per_doc = 2
    n_feats_per_seg = 3

    shard_stem = "shard_00000"

    # Create parquet file with text
    texts = []
    for doc_idx in range(n_docs):
        words = [f"word{w}" for w in range(20 + doc_idx * 5)]
        texts.append(" ".join(words))

    pq_path = data_dir / f"{shard_stem}.parquet"
    pq.write_table(pa.table({"text": texts}), str(pq_path))

    # Create feature shard with segment data
    total_segs = n_docs * n_segs_per_doc
    total_feats = total_segs * n_feats_per_seg

    all_fids = t.zeros(total_feats, dtype=t.int32)
    all_mags = t.zeros(total_feats, dtype=t.float32)
    offsets = t.zeros(total_segs + 1, dtype=t.int64)
    doc_ids = t.zeros(total_segs, dtype=t.int32)
    seg_indices = t.zeros(total_segs, dtype=t.int32)
    seg_positions = t.zeros(total_segs, dtype=t.int32)

    feat_offset = 0
    seg_global = 0
    for doc_idx in range(n_docs):
        for seg_idx in range(n_segs_per_doc):
            doc_ids[seg_global] = doc_idx
            seg_indices[seg_global] = seg_idx
            seg_positions[seg_global] = seg_idx * SEGMENT_SIZE
            offsets[seg_global] = feat_offset

            for f in range(n_feats_per_seg):
                fid = (doc_idx * 10 + seg_idx * 5 + f) % N_SAE
                all_fids[feat_offset + f] = fid
                all_mags[feat_offset + f] = 0.1 * (f + 1) + 0.01 * doc_idx

            feat_offset += n_feats_per_seg
            seg_global += 1

    offsets[total_segs] = feat_offset

    # Char boundaries: each word is ~6 chars ("wordN "), use segment_size * 6 per segment
    segment_char_starts = seg_positions * 6
    segment_char_ends = (seg_positions + SEGMENT_SIZE) * 6

    shard_data = {
        "offsets": offsets,
        "feature_ids": all_fids,
        "feature_mags": all_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": t.full((n_docs,), 41, dtype=t.int32),
        "segment_char_starts": segment_char_starts,
        "segment_char_ends": segment_char_ends,
        "n_docs": n_docs,
        "n_segments": total_segs,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": SEGMENT_SIZE,
    }

    save_features(shard_data, feature_dir / f"{shard_stem}_features")

    # Create stats file
    stats_path = _make_stats_file(feature_dir)

    return feature_dir, stats_path, data_dir, tokenizer


# ======================================================================
# 1. End-to-end pipeline wiring test
# ======================================================================


class TestPercentileChecksumWrittenToStatsAfterCompletion:
    """After precompute_percentiles.run() completes, the statistics file
    contains the exact percentile checksum matching the overlay directory."""

    def test_percentile_checksum_written_to_stats_after_completion(self, tmp_path):
        """After run(), load_percentile_checksum() returns the exact checksum
        string that matches the overlay directory name."""
        from scripts.preprocess.precompute_percentiles import run
        from steering.stats_io import load_statistics

        feature_dir, stats_path = _make_percentile_env(tmp_path)
        cfg = _percentile_cfg(feature_dir)

        ret = run(cfg)
        assert ret == 0

        # Load the checksum from the stats file
        checksum = load_percentile_checksum(stats_path)
        assert checksum is not None, (
            "save_percentile_checksum was not called — "
            "this is the exact bug we're testing"
        )

        # Compute expected checksum independently
        stats = load_statistics(stats_path)
        pct_table = build_percentile_table(
            stats.per_feature_stats, stats.metadata.n_sae_features,
        )
        expected_checksum = compute_percentile_checksum(pct_table)

        assert checksum == expected_checksum


# ======================================================================
# 2. Checksum matches overlay directory
# ======================================================================


class TestChecksumMatchesOverlayDirectory:
    """The checksum in the stats file matches the actual overlay directory."""

    def test_checksum_matches_overlay_dir_name(self, tmp_path):
        """Checksum stored in stats file is identical to the overlay directory
        name under pct/. The overlay directory must exist and contain the
        expected number of shard files."""
        from scripts.preprocess.precompute_percentiles import run

        feature_dir, stats_path = _make_percentile_env(tmp_path, n_shards=2)
        cfg = _percentile_cfg(feature_dir)

        ret = run(cfg)
        assert ret == 0

        checksum = load_percentile_checksum(stats_path)
        assert checksum is not None

        # Overlay directory should be pct/<checksum[:16]>/
        pct_root = feature_dir / "pct"
        assert pct_root.is_dir(), "No pct/ directory created"

        overlay_dirs = list(pct_root.iterdir())
        assert len(overlay_dirs) == 1
        assert overlay_dirs[0].name == checksum[:CHECKSUM_PREFIX_LEN]

        # Overlay must have the same number of shards as the feature dir
        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
        overlay_shards = glob_overlay_shards(pct_dir)
        feature_shards = list(
            feature_dir.glob("shard_*_features.safetensors.zst"),
        )
        assert len(overlay_shards) == len(feature_shards), (
            f"{len(overlay_shards)} overlay shards != "
            f"{len(feature_shards)} feature shards"
        )


# ======================================================================
# 3. Checksum survives re-read
# ======================================================================


class TestChecksumSurvivesReRead:
    """Write the checksum, reload the statistics file, verify it's identical."""

    def test_checksum_survives_fresh_disk_read(self, tmp_path):
        """Checksum persists correctly through safetensors serialization."""
        from scripts.preprocess.precompute_percentiles import run

        feature_dir, stats_path = _make_percentile_env(tmp_path)
        cfg = _percentile_cfg(feature_dir)

        ret = run(cfg)
        assert ret == 0

        # First read
        checksum1 = load_percentile_checksum(stats_path)
        assert checksum1 is not None

        # Second read — fresh from disk
        checksum2 = load_percentile_checksum(stats_path)
        assert checksum2 == checksum1

        # Also verify via raw metadata read
        metadata = _read_metadata(stats_path)
        checksum3 = metadata.get("percentile_checksum")
        assert checksum3 == checksum1


# ======================================================================
# 4. Idempotency
# ======================================================================


class TestIdempotency:
    """Running precompute_percentiles twice produces the same checksum."""

    def test_idempotent_with_resume(self, tmp_path):
        """Two runs with resume=true produce identical checksum and intact overlay."""
        from scripts.preprocess.precompute_percentiles import run

        feature_dir, stats_path = _make_percentile_env(tmp_path, n_shards=2)
        cfg = _percentile_cfg(feature_dir)

        # First run
        ret1 = run(cfg)
        assert ret1 == 0
        checksum1 = load_percentile_checksum(stats_path)
        assert checksum1 is not None

        # Second run with resume=true
        cfg2 = _percentile_cfg(feature_dir, resume=True)
        ret2 = run(cfg2)
        assert ret2 == 0
        checksum2 = load_percentile_checksum(stats_path)

        assert checksum2 == checksum1, (
            f"Checksum changed between runs: {checksum1!r} → {checksum2!r}"
        )

        # Overlay should still be intact
        pct_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum1)
        assert pct_dir.is_dir()
        overlay_shards = glob_overlay_shards(pct_dir)
        feature_shards = list(
            feature_dir.glob("shard_*_features.safetensors.zst"),
        )
        assert len(overlay_shards) == len(feature_shards)


# ======================================================================
# 4b. Dry run does NOT write checksum
# ======================================================================


class TestDryRunDoesNotWriteChecksum:
    """dry_run=True must not modify the statistics file."""

    def test_dry_run_does_not_write_percentile_checksum(self, tmp_path):
        """With dry_run=True, load_percentile_checksum() returns None."""
        from scripts.preprocess.precompute_percentiles import run

        feature_dir, stats_path = _make_percentile_env(tmp_path)
        cfg = _percentile_cfg(feature_dir, dry_run=True)

        ret = run(cfg)
        assert ret == 0

        checksum = load_percentile_checksum(stats_path)
        assert checksum is None, (
            f"Checksum {checksum!r} was written during dry_run — "
            "dry_run must not modify the stats file"
        )


# ======================================================================
# 4c. Early-exit "overlay exists" path saves checksum retroactively
# ======================================================================


class TestEarlyExitSavesChecksum:
    """When the overlay already exists from a prior run, re-running the script
    (without resume or force) should still save the checksum to the stats file.
    This covers the retroactive healing path."""

    def test_existing_overlay_retroactively_saves_checksum(self, tmp_path):
        """First run saves overlay + checksum. Wipe checksum from stats, re-run
        — the early-exit path must re-save the checksum."""
        from scripts.preprocess.precompute_percentiles import run
        from steering.stats_io import save_percentile_checksum

        feature_dir, stats_path = _make_percentile_env(tmp_path)
        cfg = _percentile_cfg(feature_dir)

        # First run — produces overlay + saves checksum
        ret = run(cfg)
        assert ret == 0
        original_checksum = load_percentile_checksum(stats_path)
        assert original_checksum is not None

        # Simulate the original bug: wipe the checksum from stats
        save_percentile_checksum(stats_path, "")
        # Overwrite with empty to simulate missing
        from safetensors.torch import load_file, save_file
        tensors = load_file(str(stats_path))
        metadata = _read_metadata(stats_path)
        del metadata["percentile_checksum"]
        save_file(tensors, str(stats_path), metadata=metadata)
        assert load_percentile_checksum(stats_path) is None

        # Re-run — hits "overlay already exists" early exit
        ret2 = run(cfg)
        assert ret2 == 0

        # Checksum must be restored
        restored = load_percentile_checksum(stats_path)
        assert restored == original_checksum, (
            f"Early-exit path did not restore checksum: "
            f"expected {original_checksum!r}, got {restored!r}"
        )


# ======================================================================
# 5. Packed rows can find the checksum
# ======================================================================


class TestPackedRowsCanFindChecksum:
    """After precompute_percentiles, the packed rows pipeline can discover
    the percentile overlay via load_percentile_checksum."""

    def test_packed_rows_checksum_chain(self, tmp_path):
        """Simulate what precompute_packed_rows does: load checksum from stats
        file, then verify the overlay directory at pct/{checksum}/ exists and
        has the right number of shards."""
        from scripts.preprocess.precompute_percentiles import run

        feature_dir, stats_path = _make_percentile_env(tmp_path, n_shards=2)
        cfg = _percentile_cfg(feature_dir)

        ret = run(cfg)
        assert ret == 0

        # This is exactly what precompute_packed_rows.run() does (lines 426-431):
        pct_checksum = load_percentile_checksum(stats_path)
        assert pct_checksum is not None, (
            "load_percentile_checksum returned None — packed rows would fail "
            "with 'No percentile checksum found — run precompute_percentiles first'"
        )

        # Verify the overlay directory exists (same path construction as packed rows)
        pct_dir = (
            feature_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
        )
        assert pct_dir.is_dir(), (
            f"Overlay directory {pct_dir} does not exist — "
            f"FeatureIndex would fail to find precomputed percentiles"
        )

        # Verify overlay shards exist
        overlay_shards = glob_overlay_shards(pct_dir)
        assert len(overlay_shards) > 0, (
            "No overlay shards found in the percentile directory"
        )
        feature_shards = list(
            feature_dir.glob("shard_*_features.safetensors.zst"),
        )
        assert len(overlay_shards) == len(feature_shards)


# ======================================================================
# 6. Token boundaries — tokenizer checksum saved by run()
# ======================================================================


class TestTokenBoundaryChecksumWritten:
    """After precompute_token_boundaries.run() completes, the statistics file
    contains the tokenizer checksum and the overlay directory exists."""

    def test_tokenizer_checksum_written_to_stats_after_completion(self, tmp_path):
        """After run(), load_tokenizer_checksum() returns the correct checksum
        and the tok/{checksum}/seg_boundaries/ directory exists with the right
        shard count."""
        from scripts.preprocess.precompute_token_boundaries import run

        feature_dir, stats_path, data_dir, tokenizer = (
            _make_token_boundary_env(tmp_path)
        )

        expected_checksum = compute_tokenizer_checksum(tokenizer)

        cfg = OmegaConf.create({
            "feature_dir": str(feature_dir),
            "segment_size": SEGMENT_SIZE,
            "data_dir": str(data_dir),
            "dry_run": False,
            "resume": False,
            "force": False,
            "workers": 1,
        })

        with patch(
            "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
            return_value=tokenizer,
        ):
            ret = run(cfg)

        assert ret == 0

        # Verify checksum was saved
        loaded_checksum = load_tokenizer_checksum(stats_path)
        assert loaded_checksum is not None, (
            "save_tokenizer_checksum was not called by "
            "precompute_token_boundaries.run()"
        )
        assert loaded_checksum == expected_checksum

        # Verify overlay directory exists
        tok_root = feature_dir / "tok"
        assert tok_root.is_dir(), "No tok/ directory created"

        overlay_dirs = list(tok_root.iterdir())
        assert len(overlay_dirs) == 1
        assert overlay_dirs[0].name == expected_checksum[:CHECKSUM_PREFIX_LEN]

        # Verify overlay shards exist inside seg_boundaries/
        seg_boundaries_dir = overlay_dirs[0] / "seg_boundaries"
        assert seg_boundaries_dir.is_dir()
        overlay_shards = glob_overlay_shards(seg_boundaries_dir)
        feature_shards = list(
            feature_dir.glob("shard_*_features.safetensors.zst"),
        )
        assert len(overlay_shards) == len(feature_shards)

    def test_tokenizer_checksum_survives_re_read(self, tmp_path):
        """Tokenizer checksum persists through safetensors serialization."""
        from scripts.preprocess.precompute_token_boundaries import run

        feature_dir, stats_path, data_dir, tokenizer = (
            _make_token_boundary_env(tmp_path)
        )

        cfg = OmegaConf.create({
            "feature_dir": str(feature_dir),
            "segment_size": SEGMENT_SIZE,
            "data_dir": str(data_dir),
            "dry_run": False,
            "resume": False,
            "force": False,
            "workers": 1,
        })

        with patch(
            "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
            return_value=tokenizer,
        ):
            ret = run(cfg)

        assert ret == 0

        checksum1 = load_tokenizer_checksum(stats_path)
        assert checksum1 is not None

        # Fresh read from disk
        checksum2 = load_tokenizer_checksum(stats_path)
        assert checksum2 == checksum1

        # Via raw metadata
        metadata = _read_metadata(stats_path)
        assert metadata.get("tokenizer_checksum") == checksum1


# ======================================================================
# 7. Checksum NOT written on failure
# ======================================================================


class TestChecksumNotWrittenOnFailure:
    """If the percentile computation fails partway, the stats file must NOT
    have a percentile checksum pointing to an incomplete overlay."""

    def test_percentile_checksum_not_written_on_shard_read_failure(self, tmp_path):
        """Mock a shard read to raise; verify stats file has no checksum."""
        from scripts.preprocess.precompute_percentiles import run

        feature_dir, stats_path = _make_percentile_env(tmp_path)
        cfg = _percentile_cfg(feature_dir)

        # Verify no checksum before the run
        assert load_percentile_checksum(stats_path) is None

        # Patch load_features within the precompute module to fail
        with patch(
            "scripts.preprocess.precompute_percentiles.load_features",
            side_effect=RuntimeError("Simulated shard read failure"),
        ):
            with pytest.raises(RuntimeError, match="Simulated shard read failure"):
                run(cfg)

        # Stats file must NOT have the checksum
        checksum = load_percentile_checksum(stats_path)
        assert checksum is None, (
            f"Checksum {checksum!r} was written to stats file despite shard "
            f"failure — an incomplete overlay would be pointed to"
        )

    def test_tokenizer_checksum_not_written_on_shard_read_failure(self, tmp_path):
        """Mock a shard read to raise; verify stats file has no tokenizer checksum."""
        from scripts.preprocess.precompute_token_boundaries import run

        feature_dir, stats_path, data_dir, tokenizer = (
            _make_token_boundary_env(tmp_path)
        )

        # Verify no checksum before the run
        assert load_tokenizer_checksum(stats_path) is None

        cfg = OmegaConf.create({
            "feature_dir": str(feature_dir),
            "segment_size": SEGMENT_SIZE,
            "data_dir": str(data_dir),
            "dry_run": False,
            "resume": False,
            "force": False,
            "workers": 1,
        })

        with patch(
            "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
            return_value=tokenizer,
        ), patch(
            "scripts.preprocess.precompute_token_boundaries.load_features",
            side_effect=RuntimeError("Simulated shard read failure"),
        ):
            with pytest.raises(RuntimeError, match="Simulated shard read failure"):
                run(cfg)

        checksum = load_tokenizer_checksum(stats_path)
        assert checksum is None, (
            f"Tokenizer checksum {checksum!r} was written despite shard failure"
        )
