"""Tests for feature config naming: _name_from_file_path and parse_feature_configs.

Covers the fix for the config name collision where all feature YAML files
(named ``features.yaml``) produced the same config name ``"features"``,
causing standalone harmfulness eval results to silently overwrite each other.

The fix uses the relative path from ``out/data/common_features/`` as the
config name (e.g. ``safety_sft/f0.9/m0.5``), consistent with directory
discovery naming.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from steering.eval.feature_config import (
    FeatureConfig,
    _name_from_file_path,
    parse_feature_configs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_FEATURES = {42: 3, 108: 7, 256: 1}


def _write_feature_yaml(path: Path, features: dict | None = None) -> Path:
    """Write a features.yaml file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(features or _SAMPLE_FEATURES, f)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================================
# _name_from_file_path unit tests
# ============================================================================


class TestNameFromFilePath:
    """Unit tests for the _name_from_file_path helper."""

    def test_common_features_safety_p90(self) -> None:
        """Safety p90 feature file gets a path-based name."""
        p = Path("out/data/common_features/safety_sft/f0.9/m0.5/features.yaml")
        assert _name_from_file_path(p) == "safety_sft/f0.9/m0.5"

    def test_common_features_capability_p90(self) -> None:
        """Capability p90 feature file gets a distinct path-based name."""
        p = Path(
            "out/data/common_features/capabilities_sft/f0.9/m0.5/features.yaml"
        )
        assert _name_from_file_path(p) == "capabilities_sft/f0.9/m0.5"

    def test_common_features_safety_delta_base(self) -> None:
        """Safety delta-base feature file uses relative path from common_features."""
        p = Path(
            "out/data/common_features/safety_sft/delta_base/dist_geometric/"
            "mean_10.0/std_10.0/min_1/max_40/neg_dist_geometric/mean_10.0/"
            "std_10.0/min_0/max_40/seed_42/r_mean_pct_incl_zeros/m0.5/features.yaml"
        )
        name = _name_from_file_path(p)
        assert name.startswith("safety_sft/delta_base/")
        assert name.endswith("r_mean_pct_incl_zeros/m0.5")

    def test_common_features_capability_delta_base(self) -> None:
        """Capability delta-base gets a different name than safety delta-base."""
        safety = Path(
            "out/data/common_features/safety_sft/delta_base/dist_geometric/"
            "mean_10.0/std_10.0/min_1/max_40/seed_42/features.yaml"
        )
        cap = Path(
            "out/data/common_features/capabilities_sft/delta_base/dist_geometric/"
            "mean_10.0/std_10.0/min_1/max_40/seed_42/features.yaml"
        )
        assert _name_from_file_path(safety) != _name_from_file_path(cap)
        assert _name_from_file_path(safety).startswith("safety_sft/")
        assert _name_from_file_path(cap).startswith("capabilities_sft/")

    def test_common_features_mean_k_standalone(self) -> None:
        """Mean-k standalone feature file gets correct name."""
        p = Path(
            "out/data/common_features/safety_sft/"
            "geometric_m10_min1_max40_d12_seed1234/standalone/features.yaml"
        )
        assert (
            _name_from_file_path(p)
            == "safety_sft/geometric_m10_min1_max40_d12_seed1234/standalone"
        )

    def test_common_features_mean_k_pretrain_ref(self) -> None:
        """Mean-k pretrain-ref feature file gets correct name."""
        p = Path(
            "out/data/common_features/capabilities_sft/"
            "geometric_m10_min1_max40_d12_seed1234/"
            "pretrain_ref_local_l8_1shard/features.yaml"
        )
        name = _name_from_file_path(p)
        assert name == (
            "capabilities_sft/geometric_m10_min1_max40_d12_seed1234/"
            "pretrain_ref_local_l8_1shard"
        )

    def test_outside_common_features_falls_back_to_stem(self, tmp_path: Path) -> None:
        """Files outside common_features use the filename stem (backward compat)."""
        p = tmp_path / "my_custom_features.yaml"
        assert _name_from_file_path(p) == "my_custom_features"

    def test_outside_common_features_named_features(self, tmp_path: Path) -> None:
        """A file named features.yaml outside common_features uses stem 'features'."""
        p = tmp_path / "features.yaml"
        assert _name_from_file_path(p) == "features"

    def test_file_directly_in_common_features_root(self) -> None:
        """A file directly in common_features/ (no subdir) uses stem fallback."""
        p = Path("out/data/common_features/features.yaml")
        # parent_rel would be "." so should fall through to stem
        assert _name_from_file_path(p) == "features"

    def test_custom_filename_in_common_features(self) -> None:
        """A non-features.yaml file in common_features uses parent path."""
        p = Path("out/data/common_features/safety_sft/f0.9/m0.5/custom.yaml")
        # Should still use parent directory path, not the filename
        assert _name_from_file_path(p) == "safety_sft/f0.9/m0.5"

    def test_absolute_path_resolved(self) -> None:
        """Absolute paths are resolved correctly via Path.resolve()."""
        cwd = Path.cwd()
        p = cwd / "out/data/common_features/safety_sft/f0.9/m0.5/features.yaml"
        assert _name_from_file_path(p) == "safety_sft/f0.9/m0.5"


# ============================================================================
# All 8 real feature configs produce distinct names
# ============================================================================


class TestAllConfigsDistinct:
    """Verify that all 8 feature config files from common_features produce
    distinct config names, preventing the collision that motivated the fix."""

    FEATURE_PATHS = [
        "out/data/common_features/safety_sft/f0.9/m0.5/features.yaml",
        "out/data/common_features/safety_sft/geometric_m10_min1_max40_d12_seed1234/standalone/features.yaml",
        "out/data/common_features/safety_sft/geometric_m10_min1_max40_d12_seed1234/pretrain_ref_local_l8_1shard/features.yaml",
        "out/data/common_features/safety_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/r_mean_pct_incl_zeros/m0.5/features.yaml",
        "out/data/common_features/capabilities_sft/f0.9/m0.5/features.yaml",
        "out/data/common_features/capabilities_sft/geometric_m10_min1_max40_d12_seed1234/standalone/features.yaml",
        "out/data/common_features/capabilities_sft/geometric_m10_min1_max40_d12_seed1234/pretrain_ref_local_l8_1shard/features.yaml",
        "out/data/common_features/capabilities_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/r_mean_pct_incl_zeros/m0.5/features.yaml",
    ]

    def test_all_names_unique(self) -> None:
        """All 8 feature file paths produce distinct config names."""
        names = [_name_from_file_path(Path(p)) for p in self.FEATURE_PATHS]
        assert len(set(names)) == len(names), (
            f"Collision detected! Names: {names}"
        )

    def test_safety_and_capability_never_collide(self) -> None:
        """No safety config name matches any capability config name."""
        safety_names = {
            _name_from_file_path(Path(p))
            for p in self.FEATURE_PATHS
            if "safety_sft" in p
        }
        cap_names = {
            _name_from_file_path(Path(p))
            for p in self.FEATURE_PATHS
            if "capabilities_sft" in p
        }
        assert safety_names & cap_names == set()

    def test_no_name_equals_unconditional(self) -> None:
        """No feature file path produces the reserved 'unconditional' name."""
        for p in self.FEATURE_PATHS:
            assert _name_from_file_path(Path(p)) != "unconditional"

    def test_no_name_is_bare_features(self) -> None:
        """No common_features path produces the bare 'features' stem name."""
        for p in self.FEATURE_PATHS:
            assert _name_from_file_path(Path(p)) != "features"


# ============================================================================
# parse_feature_configs integration: single file paths
# ============================================================================


class TestParseFeatureConfigsSingleFile:
    """Tests for parse_feature_configs with individual file paths (format 2),
    verifying the fix produces path-based names."""

    def test_file_in_common_features_gets_path_name(self, tmp_path: Path) -> None:
        """A feature file under common_features gets a path-based name."""
        common = tmp_path / "out" / "data" / "common_features"
        feat_path = common / "safety_sft" / "f0.9" / "m0.5" / "features.yaml"
        _write_feature_yaml(feat_path)

        # Temporarily patch the constant so it points to our tmp dir
        import steering.eval.feature_config as mod

        orig = mod._COMMON_FEATURES_DIR
        mod._COMMON_FEATURES_DIR = str(common)  # ty: ignore[invalid-assignment]
        try:
            configs = parse_feature_configs(str(feat_path))
        finally:
            mod._COMMON_FEATURES_DIR = orig

        assert len(configs) == 1
        assert configs[0].name == "safety_sft/f0.9/m0.5"
        assert configs[0].features == _SAMPLE_FEATURES

    def test_file_outside_common_features_gets_stem_name(
        self, tmp_path: Path
    ) -> None:
        """A feature file outside common_features falls back to stem."""
        feat_path = tmp_path / "my_features.yaml"
        _write_feature_yaml(feat_path)

        configs = parse_feature_configs(str(feat_path))
        assert len(configs) == 1
        assert configs[0].name == "my_features"

    def test_two_files_same_name_different_dirs_get_distinct_names(
        self, tmp_path: Path
    ) -> None:
        """Two features.yaml in different common_features subdirs get distinct names."""
        common = tmp_path / "out" / "data" / "common_features"
        safety = common / "safety_sft" / "f0.9" / "m0.5" / "features.yaml"
        cap = common / "capabilities_sft" / "f0.9" / "m0.5" / "features.yaml"
        _write_feature_yaml(safety)
        _write_feature_yaml(cap, {99: 1})

        import steering.eval.feature_config as mod

        orig = mod._COMMON_FEATURES_DIR
        mod._COMMON_FEATURES_DIR = str(common)  # ty: ignore[invalid-assignment]
        try:
            safety_configs = parse_feature_configs(str(safety))
            cap_configs = parse_feature_configs(str(cap))
        finally:
            mod._COMMON_FEATURES_DIR = orig

        assert safety_configs[0].name != cap_configs[0].name
        assert safety_configs[0].name == "safety_sft/f0.9/m0.5"
        assert cap_configs[0].name == "capabilities_sft/f0.9/m0.5"


# ============================================================================
# parse_feature_configs integration: list of file paths (format 4)
# ============================================================================


class TestParseFeatureConfigsListOfFiles:
    """Tests for parse_feature_configs with a list of file paths (format 4)."""

    def test_list_of_common_features_files_all_distinct(
        self, tmp_path: Path
    ) -> None:
        """A list of features.yaml paths all get unique names."""
        common = tmp_path / "out" / "data" / "common_features"
        paths = []
        for phase in ["safety_sft", "capabilities_sft"]:
            for sub in ["f0.9/m0.5", "k32/seed_42"]:
                p = common / phase / sub / "features.yaml"
                _write_feature_yaml(p, {1: 2})
                paths.append(str(p))

        import steering.eval.feature_config as mod

        orig = mod._COMMON_FEATURES_DIR
        mod._COMMON_FEATURES_DIR = str(common)  # ty: ignore[invalid-assignment]
        try:
            configs = parse_feature_configs(paths)
        finally:
            mod._COMMON_FEATURES_DIR = orig

        assert len(configs) == 4
        names = [c.name for c in configs]
        assert len(set(names)) == 4, f"Collision in list mode: {names}"

    def test_mixed_common_and_external_files(self, tmp_path: Path) -> None:
        """Mixing common_features files with external files works correctly."""
        common = tmp_path / "out" / "data" / "common_features"
        internal = common / "safety_sft" / "f0.5" / "m0.5" / "features.yaml"
        external = tmp_path / "external" / "my_signal.yaml"
        _write_feature_yaml(internal)
        _write_feature_yaml(external, {7: 1})

        import steering.eval.feature_config as mod

        orig = mod._COMMON_FEATURES_DIR
        mod._COMMON_FEATURES_DIR = str(common)  # ty: ignore[invalid-assignment]
        try:
            configs = parse_feature_configs([str(internal), str(external)])
        finally:
            mod._COMMON_FEATURES_DIR = orig

        assert len(configs) == 2
        assert configs[0].name == "safety_sft/f0.5/m0.5"
        assert configs[1].name == "my_signal"


class TestParseFeatureConfigsResolvedNameUniqueness:
    """Resolved format-4 names must be unique before callers write outputs."""

    def test_single_unnamed_inline_config_is_named_custom(self) -> None:
        configs = parse_feature_configs([{"features": {1: 0}}])

        assert [config.name for config in configs] == ["custom"]
        assert [config.features for config in configs] == [{1: 0}]

    def test_multiple_unnamed_inline_configs_collide_and_raise(self) -> None:
        # Both resolve to "custom"; raising is the fix for the silent-overwrite bug.
        with pytest.raises(ValueError, match=r"duplicate resolved name.*custom"):
            parse_feature_configs(
                [
                    {"features": {1: 0}},
                    {"features": {2: 1}},
                ]
            )

    def test_distinct_safe_inline_names_are_unchanged(self) -> None:
        configs = parse_feature_configs(
            [
                {"name": "first", "features": {1: 0}},
                {"name": "second", "features": {2: 1}},
            ]
        )

        assert [config.name for config in configs] == ["first", "second"]
        assert [config.features for config in configs] == [{1: 0}, {2: 1}]

    def test_inline_strength_is_preserved(self) -> None:
        [config] = parse_feature_configs(
            [{"name": "mixed", "strength": 0.25, "features": {1: 0, 2: 1}}]
        )

        assert config.strength == 0.25

    def test_inline_signed_sparse_scores_are_preserved(self) -> None:
        [config] = parse_feature_configs(
            [{"name": "signed", "features": {1: 1, 2: -1}}]
        )

        assert config.features == {1: 1, 2: -1}

    @pytest.mark.parametrize("strength", [-0.01, 1.01, float("nan"), True])
    def test_strength_must_be_a_finite_unit_interval(self, strength: object) -> None:
        with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
            FeatureConfig(name="invalid", features={1: 0}, strength=strength)  # ty: ignore[invalid-argument-type]

    def test_inline_names_colliding_after_sanitization_raise(self) -> None:
        with pytest.raises(ValueError, match="same_name"):
            parse_feature_configs(
                [
                    {"name": "same/name", "features": {1: 0}},
                    {"name": "same name", "features": {2: 1}},
                ]
            )

    def test_equal_external_file_stems_raise_with_offending_name(
        self, tmp_path: Path
    ) -> None:
        first = _write_feature_yaml(tmp_path / "a" / "features_v1.yaml", {1: 0})
        second = _write_feature_yaml(tmp_path / "b" / "features_v1.yaml", {2: 1})

        with pytest.raises(ValueError, match="features_v1"):
            parse_feature_configs([str(first), str(second)])

    def test_external_stem_guard_precedes_output_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = _write_feature_yaml(tmp_path / "a" / "features_v1.yaml", {1: 0})
        second = _write_feature_yaml(tmp_path / "b" / "features_v1.yaml", {2: 1})
        sources = [first, second]
        source_hashes_before = [_sha256(path) for path in sources]

        report_dir = tmp_path / "report"
        report_dir.mkdir()
        existing_output = report_dir / "features_v1.yaml"
        existing_output.write_bytes(b"completed-eval-result\n")
        output_hash_before = _sha256(existing_output)

        writes: list[Path] = []
        unlinks: list[Path] = []

        def record_write(path: Path, *_args, **_kwargs) -> int:
            writes.append(path)
            return 0

        def record_unlink(path: Path, *_args, **_kwargs) -> None:
            unlinks.append(path)

        monkeypatch.setattr(Path, "write_text", record_write)
        monkeypatch.setattr(Path, "unlink", record_unlink)

        def parse_then_emit() -> None:
            configs = parse_feature_configs([str(first), str(second)])
            for config in configs:
                output_path = report_dir / f"{config.name}.yaml"
                if output_path.exists():
                    output_path.unlink()
                output_path.write_text(yaml.safe_dump(config.features))

        with pytest.raises(ValueError, match="features_v1"):
            parse_then_emit()

        assert writes == []
        assert unlinks == []
        assert [_sha256(path) for path in sources] == source_hashes_before
        assert _sha256(existing_output) == output_hash_before


# ============================================================================
# Consistency: single-file naming matches directory discovery naming
# ============================================================================


class TestNamingConsistency:
    """Verify that passing a single file produces the same name as directory
    discovery would for that file. This is the key invariant the fix ensures."""

    def test_single_file_matches_directory_discovery(self, tmp_path: Path) -> None:
        """Passing a file directly produces the same name as discovering it
        via its parent directory."""
        root = tmp_path / "safety_sft"
        for sub in ["f0.5/m0.5", "f0.9/m0.9"]:
            _write_feature_yaml(root / sub / "features.yaml")

        # Directory discovery
        dir_configs = parse_feature_configs(str(root))
        dir_names = {c.name for c in dir_configs}

        # Single file — need to use the parent path logic
        # For files under common_features, _name_from_file_path uses the
        # relative path. For files under an arbitrary directory, directory
        # discovery uses dir_prefix/rel. These should match when the
        # directory is the phase dir (e.g. safety_sft/).
        assert "safety_sft/f0.5/m0.5" in dir_names
        assert "safety_sft/f0.9/m0.9" in dir_names

    def test_directory_discovery_names_are_path_based(self, tmp_path: Path) -> None:
        """Directory discovery never produces bare 'features' as a name."""
        root = tmp_path / "capabilities_sft"
        for sub in ["a/b", "c/d", "e/f"]:
            _write_feature_yaml(root / sub / "features.yaml")

        configs = parse_feature_configs(str(root))
        for c in configs:
            assert c.name != "features", (
                f"Directory discovery produced bare 'features' name: {c.name}"
            )
            assert "capabilities_sft/" in c.name


# ============================================================================
# Output path collision regression test
# ============================================================================


class TestOutputPathCollisionRegression:
    """Regression test: two feature configs that previously collided on
    output path ``features.yaml`` now produce distinct paths."""

    def test_resolve_paths_distinct_for_delta_base_configs(
        self, tmp_path: Path
    ) -> None:
        """Safety and capability delta-base configs resolve to distinct output paths."""
        from unittest.mock import patch

        from scripts.eval.harmfulness_eval import resolve_harmfulness_paths

        base_dir = str(tmp_path)
        os.makedirs(tmp_path / "base_checkpoints" / "d26")

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": base_dir}):
            _, out_safety = resolve_harmfulness_paths(
                source="base",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                config_name="safety_sft/delta_base/k32/seed_42",
            )
            _, out_cap = resolve_harmfulness_paths(
                source="base",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                config_name="capabilities_sft/delta_base/k32/seed_42",
            )

        assert out_safety != out_cap
        assert "safety_sft" in out_safety
        assert "capabilities_sft" in out_cap

    def test_end_to_end_no_collision(self, tmp_path: Path) -> None:
        """Full pipeline: parse two delta-base files → get distinct config names
        → resolve to distinct output paths."""
        common = tmp_path / "out" / "data" / "common_features"
        safety_feat = (
            common / "safety_sft" / "delta_base" / "k32" / "features.yaml"
        )
        cap_feat = (
            common / "capabilities_sft" / "delta_base" / "k32" / "features.yaml"
        )
        _write_feature_yaml(safety_feat, {1: 0, 2: 0})
        _write_feature_yaml(cap_feat, {3: 0, 4: 0})

        import steering.eval.feature_config as mod

        orig = mod._COMMON_FEATURES_DIR
        mod._COMMON_FEATURES_DIR = str(common)  # ty: ignore[invalid-assignment]
        try:
            safety_cfg = parse_feature_configs(str(safety_feat))[0]
            cap_cfg = parse_feature_configs(str(cap_feat))[0]
        finally:
            mod._COMMON_FEATURES_DIR = orig

        # Config names are distinct
        assert safety_cfg.name != cap_cfg.name

        # Features are correct (not swapped)
        assert safety_cfg.features == {1: 0, 2: 0}
        assert cap_cfg.features == {3: 0, 4: 0}

        # Names reflect the directory structure
        assert "safety_sft" in safety_cfg.name
        assert "capabilities_sft" in cap_cfg.name
