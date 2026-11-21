"""Tests for configurable zstd compression level."""

import pytest
import torch as t
from omegaconf import OmegaConf

from steering.feature_io import ZSTD_LEVEL, load_features, save_features
from steering.overlay_io import load_overlay_shard, save_overlay_shard


def _make_shard_dict(n_segments: int = 3, feats_per_seg: int = 5, n_docs: int = 2) -> dict:
    """Build a minimal valid shard dict for testing."""
    n_features = n_segments * feats_per_seg
    offsets = [i * feats_per_seg for i in range(n_segments + 1)]

    doc_ids = []
    seg_indices = []
    seg_positions = []
    seg_char_starts = []
    seg_char_ends = []
    for d in range(n_docs):
        segs_for_doc = n_segments // n_docs + (1 if d < n_segments % n_docs else 0)
        for s in range(segs_for_doc):
            doc_ids.append(d)
            seg_indices.append(s)
            seg_positions.append(s * 128)
            seg_char_starts.append(s * 512)
            seg_char_ends.append((s + 1) * 512)

    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.randint(0, 16383, (n_features,), dtype=t.int32),
        "feature_mags": t.rand(n_features, dtype=t.float32).abs() + 0.01,
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(seg_positions, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * n_docs, dtype=t.int32),
        "n_docs": n_docs,
        "n_segments": n_segments,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
    }


class TestFeatureIoZstdLevel:
    """save_features with explicit zstd_level produces loadable shards."""

    @pytest.mark.parametrize("level", [1, 12, 19])
    def test_roundtrip_at_level(self, tmp_path, level):
        """Saving at different zstd levels produces identical tensors on load."""
        original = _make_shard_dict()
        out = save_features(original, tmp_path / f"shard_level{level}", zstd_level=level)
        loaded = load_features(out)

        offsets = original["offsets"]
        n_segs = offsets.numel() - 1
        for seg in range(n_segs):
            s, e = offsets[seg].item(), offsets[seg + 1].item()
            orig_sorted = original["feature_ids"][s:e].sort().values
            load_sorted = loaded["feature_ids"][s:e].sort().values
            assert t.equal(orig_sorted, load_sorted), f"feature_ids mismatch at level {level}"

            orig_mags = original["feature_mags"][s:e].sort().values
            load_mags = loaded["feature_mags"][s:e].sort().values
            assert t.allclose(orig_mags, load_mags, atol=1e-3), f"feature_mags mismatch at level {level}"

        assert loaded["n_docs"] == original["n_docs"]
        assert loaded["n_segments"] == original["n_segments"]

    def test_default_uses_module_constant(self, tmp_path):
        """Calling save_features() without zstd_level uses ZSTD_LEVEL (12)."""
        assert ZSTD_LEVEL == 12
        original = _make_shard_dict()
        out = save_features(original, tmp_path / "shard_default")
        loaded = load_features(out)
        assert loaded["n_segments"] == original["n_segments"]

    def test_level_1_larger_than_level_19(self, tmp_path):
        """Lower compression level should produce a larger (or equal) file."""
        data = _make_shard_dict(n_segments=50, feats_per_seg=100, n_docs=5)
        out_1 = save_features(data, tmp_path / "shard_l1", zstd_level=1)
        out_19 = save_features(data, tmp_path / "shard_l19", zstd_level=19)
        assert out_1.stat().st_size >= out_19.stat().st_size


class TestOverlayIoZstdLevel:
    """save_overlay_shard with explicit zstd_level produces loadable shards."""

    @pytest.mark.parametrize("level", [1, 12, 19])
    def test_roundtrip_at_level(self, tmp_path, level):
        """Saving overlay at different zstd levels preserves tensors."""
        tensors = {
            "feature_percentiles": t.randint(0, 255, (100,), dtype=t.uint8),
            "counts": t.tensor([1, 2, 3], dtype=t.int32),
        }
        path = save_overlay_shard(tensors, tmp_path / f"shard_l{level}", zstd_level=level)
        loaded = load_overlay_shard(path)
        for key, tensor in tensors.items():
            assert t.equal(loaded[key], tensor), f"{key} mismatch at level {level}"


class TestHydraConfigsHaveZstdLevel:
    """Each compressed-shard producer should default zstd_level to 12."""

    CONFIG_DIR = "configs/extraction"
    CONFIGS = [
        "extract_features",
        "precompute_percentiles",
        "precompute_token_boundaries",
        "precompute_packed_rows",
    ]

    @pytest.mark.parametrize("config_name", CONFIGS)
    def test_config_has_zstd_level(self, config_name):
        cfg = OmegaConf.load(f"{self.CONFIG_DIR}/{config_name}.yaml")
        assert "zstd_level" in cfg, f"{config_name}.yaml missing zstd_level"
        assert cfg.zstd_level == 12, f"{config_name}.yaml zstd_level != 12"
