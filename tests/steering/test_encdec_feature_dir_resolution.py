"""Unit tests for feature_dir auto-resolution in the encoder-decoder training script.

The training script auto-resolves feature_dir from SAE config params when
feature_dir is None, using resolve_output_dir() from steering.feature_paths.

Run with:
    uv run pytest tests/test_encdec_feature_dir_resolution.py -v
"""

import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from steering.feature_paths import (
    PRODUCER_IGNORE_BOS_FEATURES,
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    resolve_output_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth

# ---------------------------------------------------------------------------
# Helper: simulate the resolution logic from the training script
# ---------------------------------------------------------------------------


def resolve_feature_dir(cfg: DictConfig) -> str:
    """Mirror the auto-resolution logic in conditional_encdec_train.py."""
    if cfg.feature_dir is not None:
        return cfg.feature_dir
    return str(
        resolve_output_dir(
            output_dir=None,
            feature_root=cfg.feature_root,
            segment_size=cfg.segment_size,
            layer=cfg.sae.layer,
            activation_type=ActivationType(cfg.sae.activation_type),
            pooling=PoolingMode(cfg.pooling),
            width=SAEWidth(cfg.sae.width),
            model=cfg.sae.model,
            dataset_phase=cfg.get("dataset_phase", None),
            ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
        )
    )


def _make_cfg(**overrides) -> DictConfig:
    """Build a minimal DictConfig matching the training config structure."""
    base = {
        "feature_dir": None,
        "feature_root": "out/data/features",
        "pooling": "max_pool",
        "dataset_phase": None,
        "segment_size": 128,
        "sae": {
            "activation_type": "residual",
            "layer": 8,
            "width": "16k",
            "model": "gemma-2-2b",
        },
    }
    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(overrides))
    return merged  # ty: ignore[invalid-return-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoResolution:
    """Auto-resolution produces correct path from config params."""

    def test_default_config_produces_expected_path(self):
        cfg = _make_cfg()
        result = resolve_feature_dir(cfg)
        assert (
            result
            == "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/no_bos"
        )

    def test_matches_build_feature_dir_exactly(self):
        cfg = _make_cfg()
        resolved = resolve_feature_dir(cfg)
        built = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
            ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
        )
        assert resolved == str(built)


class TestExplicitOverride:
    """Explicit feature_dir bypasses auto-resolution."""

    def test_explicit_feature_dir_used_as_is(self):
        cfg = _make_cfg(feature_dir="/custom/path/to/features")
        result = resolve_feature_dir(cfg)
        assert result == "/custom/path/to/features"

    def test_explicit_overrides_all_params(self):
        cfg = _make_cfg(
            feature_dir="/override",
            feature_root="ignored",
            pooling="last_token",
            sae={
                "activation_type": "mlp",
                "layer": 99,
                "width": "65k",
                "model": "other",
            },
        )
        result = resolve_feature_dir(cfg)
        assert result == "/override"


class TestDatasetPhase:
    """dataset_phase handling."""

    def test_null_defaults_to_base(self):
        cfg = _make_cfg(dataset_phase=None)
        result = resolve_feature_dir(cfg)
        assert result.endswith("/base/no_bos")

    def test_explicit_base_phase(self):
        cfg = _make_cfg(dataset_phase="base")
        result = resolve_feature_dir(cfg)
        assert result.endswith("/base/no_bos")

    def test_standard_sft_phase(self):
        cfg = _make_cfg(dataset_phase="standard_sft")
        result = resolve_feature_dir(cfg)
        assert result.endswith("/standard_sft/no_bos")

    def test_safety_sft_phase(self):
        cfg = _make_cfg(dataset_phase="safety_sft")
        result = resolve_feature_dir(cfg)
        assert result.endswith("/safety_sft/no_bos")

    def test_capabilities_sft_phase(self):
        cfg = _make_cfg(dataset_phase="capabilities_sft")
        result = resolve_feature_dir(cfg)
        assert result.endswith("/capabilities_sft/no_bos")


class TestAllEnumValues:
    """All ActivationType / PoolingMode / SAEWidth enum values resolve correctly."""

    @pytest.mark.parametrize(
        "act", list(ActivationType), ids=[a.name for a in ActivationType]
    )
    def test_all_activation_types(self, act: ActivationType):
        cfg = _make_cfg(
            sae={
                "activation_type": act.value,
                "layer": 8,
                "width": "16k",
                "model": "gemma-2-2b",
            }
        )
        result = resolve_feature_dir(cfg)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "pool", list(PoolingMode), ids=[p.name for p in PoolingMode]
    )
    def test_all_pooling_modes(self, pool: PoolingMode):
        cfg = _make_cfg(pooling=pool.value)
        result = resolve_feature_dir(cfg)
        assert f"p_{pool.value}" in result

    @pytest.mark.parametrize("width", list(SAEWidth), ids=[w.name for w in SAEWidth])
    def test_all_sae_widths(self, width: SAEWidth):
        cfg = _make_cfg(
            sae={
                "activation_type": "residual",
                "layer": 8,
                "width": width.value,
                "model": "gemma-2-2b",
            }
        )
        result = resolve_feature_dir(cfg)
        assert f"f{width.value}" in result


class TestMissingFeatureRoot:
    """Missing feature_root raises a clear error."""

    def test_missing_feature_root_raises(self):
        cfg = OmegaConf.create(
            {
                "feature_dir": None,
                "pooling": "max_pool",
                "dataset_phase": None,
                "segment_size": 128,
                "sae": {
                    "activation_type": "residual",
                    "layer": 8,
                    "width": "16k",
                    "model": "gemma-2-2b",
                },
            }
        )
        # Slice 14 audit X2: narrowed from bare ``Exception``.
        # ``resolve_feature_dir`` accesses ``cfg.feature_root`` on a
        # ``DictConfig`` that omits the key — OmegaConf raises
        # ``ConfigAttributeError`` (a subclass of ``AttributeError``).
        with pytest.raises(ConfigAttributeError):
            resolve_feature_dir(cfg)
