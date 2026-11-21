"""Regression tests for the shared ``resolve_feature_dir`` helper.

Verify:

1. Explicit ``cfg.feature_dir`` override returns the literal path.
2. Auto-resolution path mirrors the original three duplicates.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scripts.preprocess._common import resolve_feature_dir


class TestExplicitFeatureDir:
    def test_explicit_feature_dir_returned_as_path(self):
        cfg = OmegaConf.create({"feature_dir": "/tmp/some/feature/dir"})
        result = resolve_feature_dir(cfg)
        assert result == Path("/tmp/some/feature/dir")

    def test_explicit_feature_dir_takes_precedence_over_structured(self):
        cfg = OmegaConf.create({
            "feature_dir": "/tmp/explicit",
            # These would auto-resolve if feature_dir were None — they are
            # ignored when feature_dir is set.
            "feature_root": "/tmp/auto",
            "segment_size": 128,
            "layer": 8,
            "activation_type": "residual",
            "pooling": "mean_pool",
            "sae_width": "16k",
            "model": "google/gemma-2-2b",
            "dataset_phase": "fineweb-edu",
        })
        result = resolve_feature_dir(cfg)
        assert result == Path("/tmp/explicit")


class TestAutoResolutionShape:
    def test_auto_resolution_returns_path(self, tmp_path):
        cfg = OmegaConf.create({
            "feature_dir": None,
            "feature_root": str(tmp_path / "features"),
            "segment_size": 128,
            "layer": 8,
            "activation_type": "residual",
            "pooling": "mean_pool",
            "sae_width": "16k",
            "model": "google/gemma-2-2b",
            "dataset_phase": "base",
        })
        result = resolve_feature_dir(cfg)
        assert isinstance(result, Path)
        # Result must be under the feature_root and include ``s128`` for the
        # ``verify_segment_size`` check + ``l8`` for the layer.
        assert "s128" in str(result)
        assert "l8" in str(result)
        assert result.name == "no_bos"
        assert result.parent.name == "base"
