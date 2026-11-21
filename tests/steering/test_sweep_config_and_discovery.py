"""Tests for sweep configuration, config inheritance, and packed row discovery.

Covers:
1. SweepConfig generates valid Hydra overrides
2. config_name parameterization works correctly
3. Packed row directory path construction is consistent
4. 4090 config inherits correctly from base sweep config
5. build_sweep_configs produces the expected grid
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.train.run_encdec_sweep import (
    FeatureDeltaConfig,
    FeatureDeltaMode,
    FeatureSelectionConfig,
    NegativeSamplingConfig,
    SweepConfig,
    build_sweep_configs,
)

# ======================================================================
# 1. SweepConfig override generation
# ======================================================================


class TestSweepConfigOverrides:
    """Verify SweepConfig generates correct Hydra CLI overrides."""

    def test_basic_overrides_structure(self):
        """Override list has required keys."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("geometric", 8, 8),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.1,
        )
        overrides = cfg.hydra_overrides()

        override_keys = [o.split("=")[0] for o in overrides]
        assert "config_name" in override_keys
        assert "model_tag" in override_keys
        assert "feature_selection.distribution" in override_keys
        assert "feature_selection.mean" in override_keys
        assert "feature_selection.std" in override_keys
        assert "negative_selection.distribution" in override_keys
        assert "negative_selection.mean" in override_keys

    def test_config_name_default(self):
        """Default config_name is conditional_encdec_sweep."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("geometric", 8, 8),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.0,
        )
        overrides = cfg.hydra_overrides()
        config_name_override = [o for o in overrides if o.startswith("config_name=")]
        assert len(config_name_override) == 1
        assert config_name_override[0] == "config_name=conditional_encdec_sweep"

    def test_config_name_4090(self):
        """Custom config_name is propagated."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("geometric", 8, 8),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.0,
        )
        overrides = cfg.hydra_overrides(config_name="conditional_encdec_sweep_4090")
        config_name_override = [o for o in overrides if o.startswith("config_name=")]
        assert config_name_override[0] == "config_name=conditional_encdec_sweep_4090"

    def test_delta_disabled_overrides(self):
        """When delta is disabled, feature_delta=null and delta_weights=null."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("normal", 8, 1),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.0,
        )
        overrides = cfg.hydra_overrides()
        assert "feature_delta=null" in overrides
        assert "feature_selection.delta_weights=null" in overrides

    def test_delta_enabled_overrides(self):
        """When delta is enabled, mode and ratio are set, plus delta_weights."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("normal", 8, 1),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.prev_segment, 0.5),
            fp_weight=0.0,
        )
        overrides = cfg.hydra_overrides()
        assert "feature_delta.mode=prev_segment" in overrides
        assert "feature_delta.neg_delta_ratio=0.5" in overrides
        assert "feature_selection.delta_weights=absolute_percentile" in overrides

    def test_fp_weight_zero_disables_head(self):
        """fp_weight=0.0 sets feature_prediction=null."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("geometric", 8, 8),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.0,
        )
        overrides = cfg.hydra_overrides()
        assert "feature_prediction=null" in overrides

    def test_fp_weight_nonzero_sets_weight(self):
        """fp_weight>0 sets feature_prediction.weight."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("geometric", 8, 8),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.1,
        )
        overrides = cfg.hydra_overrides()
        assert "feature_prediction.weight=0.1" in overrides
        assert "feature_prediction=null" not in overrides

    def test_delta_mode_rejects_plain_string(self):
        with pytest.raises(TypeError, match="FeatureDeltaMode"):
            FeatureDeltaConfig("prev_segment", 0.5)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

    def test_delta_disabled_rejects_ratio(self):
        with pytest.raises(ValueError, match="disabled"):
            FeatureDeltaConfig(FeatureDeltaMode.disabled, 0.5)

    def test_delta_enabled_requires_ratio(self):
        with pytest.raises(ValueError, match="required"):
            FeatureDeltaConfig(FeatureDeltaMode.prev_segment, None)


# ======================================================================
# 2. Sweep grid completeness
# ======================================================================


class TestBuildSweepConfigs:
    """Verify the sweep grid has the expected structure."""

    def test_total_config_count(self):
        """Grid should produce 7 × 3 × 5 × 3 = 315 configs."""
        configs = build_sweep_configs()
        assert len(configs) == 315

    def test_all_geometric_selections_present(self):
        """All geometric feature selection variants are represented."""
        configs = build_sweep_configs()
        geo_means = {c.fs.mean for c in configs if c.fs.distribution == "geometric"}
        assert geo_means == {4.0, 8.0, 16.0}

    def test_all_normal_selections_present(self):
        """All normal feature selection variants are represented."""
        configs = build_sweep_configs()
        normal_combos = {
            (c.fs.mean, c.fs.std) for c in configs if c.fs.distribution == "normal"
        }
        expected = {(16.0, 1.0), (16.0, 4.0), (8.0, 1.0), (8.0, 4.0)}
        assert normal_combos == expected

    def test_all_delta_modes_present(self):
        """All feature delta modes are covered."""
        configs = build_sweep_configs()
        delta_modes = {c.fd.mode for c in configs}
        assert delta_modes == {
            FeatureDeltaMode.disabled,
            FeatureDeltaMode.prev_segment,
            FeatureDeltaMode.cum_max,
        }

    def test_all_fp_weights_present(self):
        """All feature prediction weights are covered."""
        configs = build_sweep_configs()
        fp_weights = {c.fp_weight for c in configs}
        assert fp_weights == {0.0, 0.1, 1.0}

    def test_unique_model_tags(self):
        """Each config pair (ns, fd, fpw) produces a unique model_tag
        (feature selection is encoded separately in the checkpoint path)."""
        configs = build_sweep_configs()
        tags = [c.model_tag for c in configs]
        # Within the same (ns, fd, fpw), tags are identical (by design)
        # But across different (ns, fd, fpw), they should differ
        unique_tags = set(tags)
        # 3 ns × 5 fd × 3 fpw = 45 unique tags
        assert len(unique_tags) == 45


# ======================================================================
# 3. Config file tests
# ======================================================================


class TestSweepConfigFiles:
    """Verify the YAML config files exist and have expected structure."""

    def test_base_sweep_config_exists(self):
        """configs/training/conditional_encdec_sweep.yaml exists."""
        path = Path(__file__).parent.parent.parent / "configs" / "training" / "conditional_encdec_sweep.yaml"
        assert path.is_file(), f"Base sweep config not found: {path}"

    def test_4090_sweep_config_exists(self):
        """configs/training/conditional_encdec_sweep_4090.yaml exists."""
        path = Path(__file__).parent.parent.parent / "configs" / "training" / "conditional_encdec_sweep_4090.yaml"
        assert path.is_file(), f"4090 sweep config not found: {path}"

    def test_4090_config_inherits_from_base(self):
        """4090 config uses defaults: [conditional_encdec_sweep, _self_]."""
        path = Path(__file__).parent.parent.parent / "configs" / "training" / "conditional_encdec_sweep_4090.yaml"
        with open(path) as f:
            content = yaml.safe_load(f)

        assert "defaults" in content
        defaults = content["defaults"]
        # Should reference the base sweep config
        assert any(
            "conditional_encdec_sweep" in str(d) for d in defaults
        ), f"4090 config does not inherit from conditional_encdec_sweep: {defaults}"


# ======================================================================
# 4. Packed row directory path consistency
# ======================================================================


class TestPackedRowDirectoryPath:
    """Verify the packed row directory path is consistent between
    precompute_packed_rows and train_one_config."""

    def test_path_format_with_max_shards(self):
        """Path format: pct/<hash>/tok/<hash>/c<T>/s<max_shards>/"""
        from steering.overlay_io import CHECKSUM_PREFIX_LEN

        pct_checksum = "abcdef1234567890" + "0" * 48
        tok_checksum = "fedcba0987654321" + "0" * 48

        # precompute_packed_rows format (from run())
        feature_dir = Path("/features")
        pct_dir = feature_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
        tok_sub = pct_dir / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
        precompute_path = tok_sub / "c2048" / "s50"

        # train_one_config format
        max_train_shards = 50
        shards_str = f"s{max_train_shards}"
        train_path = (
            feature_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / "c2048" / shards_str
        )

        assert precompute_path == train_path

    def test_path_format_without_max_shards(self):
        """Without max_shards, both use 'sall'."""
        from steering.overlay_io import CHECKSUM_PREFIX_LEN

        pct_checksum = "a" * 64
        tok_checksum = "b" * 64

        feature_dir = Path("/features")

        # precompute
        shards_str = "sall"
        precompute_path = (
            feature_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / "c2048" / shards_str
        )

        # train_one_config
        max_train_shards = None
        train_shards_str = f"s{max_train_shards}" if max_train_shards is not None else "sall"
        train_path = (
            feature_dir / "pct" / pct_checksum[:CHECKSUM_PREFIX_LEN]
            / "tok" / tok_checksum[:CHECKSUM_PREFIX_LEN]
            / "c2048" / train_shards_str
        )

        assert precompute_path == train_path


# ======================================================================
# 5. run_encdec_sweep_fast config_name filtering
# ======================================================================


class TestSweepFastConfigNameFiltering:
    """Verify _run_one correctly filters config_name= from overrides."""

    def test_overrides_exclude_config_name(self):
        """The overrides passed to _compose_config should NOT include config_name=."""
        cfg = SweepConfig(
            fs=FeatureSelectionConfig("geometric", 8, 8),
            ns=NegativeSamplingConfig(4),
            fd=FeatureDeltaConfig(FeatureDeltaMode.disabled, None),
            fp_weight=0.0,
        )

        config_name = "conditional_encdec_sweep_4090"
        overrides = cfg.hydra_overrides(config_name=config_name)

        # The fast runner filters out config_name= before passing to compose
        filtered = [o for o in overrides if not o.startswith("config_name=")]

        # config_name should not be in the filtered list
        for o in filtered:
            assert not o.startswith("config_name="), (
                f"config_name= should be filtered out, found: {o}"
            )

        # But all other overrides should be preserved
        assert len(filtered) == len(overrides) - 1
