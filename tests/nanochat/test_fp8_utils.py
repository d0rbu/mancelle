"""Tests for shared FP8 utilities."""

import pytest
import torch
import torch.nn as nn
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from nanochat.fp8_utils import apply_fp8_training, disable_fp8, fp8_module_filter


@pytest.fixture
def simple_model():
    """Small model with various Linear layer sizes for filter testing."""
    model = nn.Sequential(
        nn.Linear(256, 512, bias=False),  # eligible: both dims >= 128, divisible by 16
        nn.Linear(512, 256, bias=False),  # eligible
        nn.Linear(64, 256, bias=False),  # NOT eligible: in_features < 128
        nn.Linear(256, 32, bias=False),  # NOT eligible: out_features < 128
        nn.Linear(
            255, 256, bias=False
        ),  # NOT eligible: in_features not divisible by 16
    )
    return model


class TestFP8ModuleFilter:
    def test_eligible_layers_pass(self, simple_model):
        assert fp8_module_filter(simple_model[0], "0") is True
        assert fp8_module_filter(simple_model[1], "1") is True

    def test_too_small_layers_rejected(self, simple_model):
        assert fp8_module_filter(simple_model[2], "2") is False  # in < 128
        assert fp8_module_filter(simple_model[3], "3") is False  # out < 128

    def test_non_divisible_by_16_rejected(self, simple_model):
        assert fp8_module_filter(simple_model[4], "4") is False

    def test_non_linear_rejected(self):
        assert fp8_module_filter(nn.Embedding(100, 256), "emb") is False
        assert fp8_module_filter(nn.LayerNorm(256), "ln") is False


class TestDisableFP8:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_roundtrip_restores_fp8_modules(self):
        model = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.Linear(256, 256, bias=False),
        ).cuda()

        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            model, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
        assert num_fp8 == 2

        with disable_fp8(model):
            num_fp8_inside = sum(
                1 for m in model.modules() if "Float8" in type(m).__name__
            )
            num_linear = sum(1 for m in model.modules() if type(m).__name__ == "Linear")
            assert num_fp8_inside == 0
            assert num_linear == 2

        num_fp8_after = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
        assert num_fp8_after == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_weight_sharing(self):
        model = nn.Sequential(nn.Linear(256, 256, bias=False)).cuda()
        fp8_config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(
            model, config=fp8_config, module_filter_fn=fp8_module_filter
        )

        layer = model[0]
        assert isinstance(layer, nn.Linear)
        fp8_weight_ptr = layer.weight.data_ptr()

        with disable_fp8(model):
            restored_layer = model[0]
            assert isinstance(restored_layer, nn.Linear)
            assert restored_layer.weight.data_ptr() == fp8_weight_ptr

    def test_noop_without_fp8_modules(self):
        model = nn.Sequential(nn.Linear(256, 256, bias=False))
        with disable_fp8(model):
            num_linear = sum(1 for m in model.modules() if type(m).__name__ == "Linear")
            assert num_linear == 1


class TestApplyFP8Training:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_returns_conversion_counts(self):
        model = nn.Sequential(
            nn.Linear(256, 256, bias=False),  # eligible
            nn.Linear(64, 256, bias=False),  # too small
        ).cuda()

        num_fp8, num_skipped = apply_fp8_training(model, recipe="tensorwise")
        assert num_fp8 == 1
        assert num_skipped == 1
