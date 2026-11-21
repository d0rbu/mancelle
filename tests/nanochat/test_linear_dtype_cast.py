"""Tests for the custom Linear class that casts weights to input dtype (nanochat/gpt.py:53-65)."""

import torch
import pytest

from nanochat.gpt import Linear


class TestLinearDtypeCast:
    """Verify Linear casts weight to match input dtype, master weights stay fp32."""

    def test_bf16_input_bf16_output(self):
        """bf16 input produces bf16 output."""
        layer = Linear(16, 32, bias=False)
        assert layer.weight.dtype == torch.float32, "Master weights must be fp32"
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        y = layer(x)
        assert y.dtype == torch.bfloat16, f"Expected bf16 output, got {y.dtype}"

    def test_fp32_input_fp32_output(self):
        """fp32 input produces fp32 output."""
        layer = Linear(16, 32, bias=False)
        x = torch.randn(2, 16, dtype=torch.float32)
        y = layer(x)
        assert y.dtype == torch.float32, f"Expected fp32 output, got {y.dtype}"

    def test_fp16_input_fp16_output(self):
        """fp16 input produces fp16 output."""
        layer = Linear(16, 32, bias=False)
        x = torch.randn(2, 16, dtype=torch.float16)
        y = layer(x)
        assert y.dtype == torch.float16, f"Expected fp16 output, got {y.dtype}"

    def test_master_weights_unchanged_after_forward(self):
        """Forward pass does not modify the master weight dtype (still fp32)."""
        layer = Linear(16, 32, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        layer(x)
        assert layer.weight.dtype == torch.float32, (
            f"Master weights should remain fp32, got {layer.weight.dtype}"
        )

    def test_bias_cast_matches_input(self):
        """When bias exists, it is also cast to match input dtype."""
        layer = Linear(16, 32, bias=True)
        assert layer.bias.dtype == torch.float32
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        y = layer(x)
        assert y.dtype == torch.bfloat16

    def test_bias_none_handled(self):
        """No bias: forward works without error."""
        layer = Linear(16, 32, bias=False)
        assert layer.bias is None
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        y = layer(x)
        assert y.shape == (2, 32)

    def test_gradient_flows_through_cast(self):
        """Gradients flow through the dtype-casting Linear."""
        layer = Linear(16, 32, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert layer.weight.grad is not None, "Weight should receive gradient"
        assert layer.weight.grad.dtype == torch.float32, (
            f"Weight grad dtype should be fp32, got {layer.weight.grad.dtype}"
        )
        assert x.grad is not None, "Input should receive gradient"

    def test_output_numerically_matches_manual_cast(self):
        """Output matches manually casting weight and doing F.linear."""
        layer = Linear(16, 32, bias=False)
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        y = layer(x)
        y_ref = torch.nn.functional.linear(x, layer.weight.to(torch.bfloat16))
        assert torch.equal(y, y_ref), (
            f"Max diff: {(y - y_ref).abs().max().item():.2e}"
        )

    def test_is_subclass_of_nn_linear(self):
        """Linear is a subclass of nn.Linear (for FP8 convert_to_float8_training)."""
        layer = Linear(16, 32, bias=False)
        assert isinstance(layer, torch.nn.Linear)
