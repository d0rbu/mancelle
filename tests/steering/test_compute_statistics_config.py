"""Tests for compute_statistics structured feature path config resolution."""

from pathlib import Path

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    resolve_output_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth


class TestAutoConstructFeatureDir:
    """Test that null feature_dir auto-constructs from structured params."""

    def test_default_params_produce_expected_path(self) -> None:
        """Default config values -> expected canonical path."""
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        expected = Path(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        )
        assert result == expected

    def test_different_layer(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=12,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l12/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        )

    def test_different_segment_size(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=64,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s64/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        )

    def test_mlp_activation(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_mlp/p_max_pool/f16k/m_gemma-2-2b/base"
        )

    def test_attention_activation(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.attention,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_att/p_max_pool/f16k/m_gemma-2-2b/base"
        )

    def test_mean_pool(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.mean_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_res/p_mean_pool/f16k/m_gemma-2-2b/base"
        )

    def test_last_token_pooling(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.last_token,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_res/p_last_token/f16k/m_gemma-2-2b/base"
        )

    def test_32k_width(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_32K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_res/p_max_pool/f32k/m_gemma-2-2b/base"
        )

    def test_different_model(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="llama-3-8b",
            dataset_phase=DatasetPhase.BASE,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_llama-3-8b/base"
        )

    def test_sft_phase(self) -> None:
        result = build_feature_dir(
            root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=DatasetPhase.STANDARD_SFT,
        )
        assert result == Path(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/standard_sft"
        )

    def test_fully_custom_combination(self) -> None:
        """Non-default values for every param."""
        result = build_feature_dir(
            root="/tmp/features",
            segment_size=64,
            layer=4,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.mean_pool,
            width=SAEWidth.W_65K,
            model="gpt2-small",
            dataset_phase=DatasetPhase.SAFETY_SFT,
        )
        expected = Path(
            "/tmp/features/s64/l4/a_mlp/p_mean_pool/f65k/m_gpt2-small/safety_sft"
        )
        assert result == expected


class TestExplicitFeatureDirOverride:
    """Test that explicit feature_dir overrides auto-construction."""

    def test_explicit_dir_used_as_is(self) -> None:
        result = resolve_output_dir(
            output_dir="/my/custom/path",
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        assert result == Path("/my/custom/path")

    def test_null_dir_auto_constructs(self) -> None:
        result = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        expected = Path(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        )
        assert result == expected

    def test_explicit_dir_ignores_structured_params(self) -> None:
        """Even with unusual structured params, explicit dir wins."""
        result = resolve_output_dir(
            output_dir="/override",
            feature_root="wrong/root",
            segment_size=999,
            layer=99,
            activation_type=ActivationType.mlp,
            pooling=PoolingMode.mean_pool,
            width=SAEWidth.W_1M,
            model="wrong-model",
            dataset_phase="safety_sft",
        )
        assert result == Path("/override")


class TestInvalidStructuredParams:
    """Test that invalid enum values raise clear errors."""

    def test_invalid_activation_type(self) -> None:
        with pytest.raises(ValueError):
            ActivationType("invalid_act")

    def test_invalid_pooling_mode(self) -> None:
        with pytest.raises(ValueError):
            PoolingMode("invalid_pool")

    def test_invalid_sae_width(self) -> None:
        with pytest.raises(ValueError):
            SAEWidth("999k")

    def test_invalid_dataset_phase(self) -> None:
        with pytest.raises(ValueError):
            DatasetPhase("nonexistent_phase")

    def test_negative_layer(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="layer"):
            build_feature_dir(
                root="out/data/features",
                segment_size=128,
                layer=-1,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.max_pool,
                width=SAEWidth.W_16K,
                model="gemma-2-2b",
                dataset_phase=DatasetPhase.BASE,
            )

    def test_zero_segment_size(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="segment_size"):
            build_feature_dir(
                root="out/data/features",
                segment_size=0,
                layer=8,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.max_pool,
                width=SAEWidth.W_16K,
                model="gemma-2-2b",
                dataset_phase=DatasetPhase.BASE,
            )

    def test_empty_model(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="model"):
            build_feature_dir(
                root="out/data/features",
                segment_size=128,
                layer=8,
                activation_type=ActivationType.residual,
                pooling=PoolingMode.max_pool,
                width=SAEWidth.W_16K,
                model="",
                dataset_phase=DatasetPhase.BASE,
            )


class TestDatasetPhaseResolution:
    """Test that null dataset_phase resolves to 'base'."""

    def test_null_phase_becomes_base(self) -> None:
        result = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=None,
        )
        assert result.name == "base"

    def test_explicit_base_phase(self) -> None:
        result = resolve_output_dir(
            output_dir=None,
            feature_root="out/data/features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase="base",
        )
        assert result.name == "base"
