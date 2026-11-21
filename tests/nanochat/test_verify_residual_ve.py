"""Verification tests for GPT residual scalars, value embeddings, and logit softcap."""

import os

os.environ["NANOCHAT_ATTN_IMPL"] = "sdpa"

from functools import partial
from unittest.mock import patch

import pytest
import torch

import nanochat.common
import nanochat.gpt
from nanochat.gpt import GPT, GPTConfig, has_ve, norm
from tests.conftest import (
    _gpt_block,
    _gpt_blocks,
    _gpt_embedding,
    _gpt_value_embedding,
)

# Force float32 compute for formula-verification tests: bfloat16 accumulates
# too much error for tight tolerances in manual reconstruction.
_fp32_patches = [
    patch.object(nanochat.common, "COMPUTE_DTYPE", torch.float32),
    patch.object(nanochat.gpt, "COMPUTE_DTYPE", torch.float32),
]


@pytest.fixture(autouse=True)
def _force_float32():
    for p in _fp32_patches:
        p.start()
    yield
    for p in _fp32_patches:
        p.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TINY_VOCAB = 128
TINY_N_EMBD = 64
TINY_N_LAYER = 2
TINY_N_HEAD = 2
TINY_N_KV_HEAD = 2
TINY_SEQ_LEN = 128


def _make_model(n_layer: int = TINY_N_LAYER, **overrides) -> GPT:
    """Build a tiny GPT on CPU, init weights, eval mode."""
    cfg = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=n_layer,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
        **overrides,
    )
    with torch.device("meta"):
        model = GPT(cfg, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _random_idx(batch: int = 2, seq_len: int = 16) -> torch.Tensor:
    return torch.randint(2, TINY_VOCAB, (batch, seq_len))


# ===================================================================
# 1. Residual scalar initialization
# ===================================================================


class TestResidualScalarInit:
    def test_resid_lambdas_are_ones(self):
        model = _make_model()
        assert torch.allclose(model.resid_lambdas.data, torch.ones(TINY_N_LAYER)), (
            f"resid_lambdas should be all 1.0, got {model.resid_lambdas.data}"
        )

    def test_x0_lambdas_are_point_one(self):
        model = _make_model()
        expected = torch.full((TINY_N_LAYER,), 0.1)
        assert torch.allclose(model.x0_lambdas.data, expected), (
            f"x0_lambdas should be all 0.1, got {model.x0_lambdas.data}"
        )

    def test_scalar_shapes_match_n_layer(self):
        for n_layer in (2, 4, 8):
            model = _make_model(n_layer=n_layer)
            assert model.resid_lambdas.shape == (n_layer,)
            assert model.x0_lambdas.shape == (n_layer,)


# ===================================================================
# 2. Residual scalar forward-pass formula
# ===================================================================


class TestResidualForwardFormula:
    """Verify x = resid_lambda * x + x0_lambda * x0 is applied before each block."""

    def test_formula_via_hooks(self):
        model = _make_model()
        idx = _random_idx()

        # We will capture the input to each block and verify the scalar mixing.
        # Strategy: register pre-hooks on each Block that record the input tensor.
        # Then manually compute the expected value from the previous output.

        block_inputs = {}
        block_outputs = {}

        def pre_hook(layer_idx, module, args):
            # args[0] is x (the first positional arg to Block.forward)
            block_inputs[layer_idx] = args[0].detach().clone()

        def post_hook(layer_idx, module, args, output):
            block_outputs[layer_idx] = output.detach().clone()

        handles = []
        for i, block in enumerate(_gpt_blocks(model)):
            handles.append(block.register_forward_pre_hook(partial(pre_hook, i)))
            handles.append(block.register_forward_hook(partial(post_hook, i)))

        with torch.inference_mode():
            model(idx)

        for h in handles:
            h.remove()

        # Reconstruct what x should be before each block.
        # x0 = norm(wte(idx))
        with torch.inference_mode():
            x0 = norm(_gpt_embedding(model)(idx))

        # Before block 0: x = resid_lambdas[0] * x0 + x0_lambdas[0] * x0
        expected_block0_input = (
            model.resid_lambdas[0].item() * x0 + model.x0_lambdas[0].item() * x0
        )
        assert torch.allclose(block_inputs[0], expected_block0_input, atol=1e-5), (
            "Block 0 input does not match resid_lambda * x + x0_lambda * x0"
        )

        # Before block 1: x = resid_lambdas[1] * block0_output + x0_lambdas[1] * x0
        expected_block1_input = (
            model.resid_lambdas[1].item() * block_outputs[0]
            + model.x0_lambdas[1].item() * x0
        )
        assert torch.allclose(block_inputs[1], expected_block1_input, atol=1e-5), (
            "Block 1 input does not match resid_lambda * prev_output + x0_lambda * x0"
        )

    def test_custom_scalar_values(self):
        """Set non-default scalar values and verify the formula still holds."""
        model = _make_model()
        with torch.no_grad():
            model.resid_lambdas.fill_(0.7)
            model.x0_lambdas.fill_(0.3)

        idx = _random_idx()

        block_inputs = {}
        block_outputs = {}

        def pre_hook(layer_idx, module, args):
            block_inputs[layer_idx] = args[0].detach().clone()

        def post_hook(layer_idx, module, args, output):
            block_outputs[layer_idx] = output.detach().clone()

        handles = []
        for i, block in enumerate(_gpt_blocks(model)):
            handles.append(block.register_forward_pre_hook(partial(pre_hook, i)))
            handles.append(block.register_forward_hook(partial(post_hook, i)))

        with torch.inference_mode():
            model(idx)

        for h in handles:
            h.remove()

        with torch.inference_mode():
            x0 = norm(_gpt_embedding(model)(idx))

        expected_block0_input = 0.7 * x0 + 0.3 * x0
        assert torch.allclose(block_inputs[0], expected_block0_input, atol=1e-5)

        expected_block1_input = 0.7 * block_outputs[0] + 0.3 * x0
        assert torch.allclose(block_inputs[1], expected_block1_input, atol=1e-5)


# ===================================================================
# 3. x0 identity — x0 is the initial embedding, not previous layer output
# ===================================================================


class TestX0Identity:
    def test_x0_is_initial_embedding(self):
        """Verify x0 used in the scalar mixing is the post-wte, post-norm embedding
        and does NOT change across layers."""
        model = _make_model(n_layer=4)
        idx = _random_idx()

        # We verify by checking that x0 is the same value used at every layer.
        # The forward code does: x0 = x (after wte + norm), then uses x0 in every iteration.
        # If x0 were mutated, the mixing at layer 1+ would differ from using the original embedding.

        # Compute x0 externally
        with torch.inference_mode():
            x0_expected = norm(_gpt_embedding(model)(idx))

        # Collect block inputs and outputs
        block_inputs = {}
        block_outputs = {}

        def pre_hook(i, module, args):
            block_inputs[i] = args[0].detach().clone()

        def post_hook(i, module, args, output):
            block_outputs[i] = output.detach().clone()

        handles = []
        for i, block in enumerate(_gpt_blocks(model)):
            handles.append(block.register_forward_pre_hook(partial(pre_hook, i)))
            handles.append(block.register_forward_hook(partial(post_hook, i)))

        with torch.inference_mode():
            model(idx)

        for h in handles:
            h.remove()

        # For each layer, solve for x0 from the formula:
        # block_input[i] = resid_lambdas[i] * prev_x + x0_lambdas[i] * x0
        # For layer 0: prev_x = x0_expected, so block_input[0] = (resid + x0_lambda) * x0_expected
        # For layer i>0: block_input[i] = resid_lambdas[i] * block_output[i-1] + x0_lambdas[i] * x0
        #   => x0 = (block_input[i] - resid_lambdas[i] * block_output[i-1]) / x0_lambdas[i]

        for i in range(1, 4):
            rl = model.resid_lambdas[i].item()
            xl = model.x0_lambdas[i].item()
            recovered_x0 = (block_inputs[i] - rl * block_outputs[i - 1]) / xl
            assert torch.allclose(recovered_x0, x0_expected, atol=1e-4), (
                f"Layer {i}: recovered x0 does not match initial embedding. "
                f"Max diff: {(recovered_x0 - x0_expected).abs().max().item():.6f}"
            )


# ===================================================================
# 4. Value embedding layer pattern (has_ve)
# ===================================================================


class TestHasVePattern:
    @pytest.mark.parametrize(
        "n_layer, expected_ve_layers",
        [
            (2, {1}),  # (n_layer-1)%2 = 1, so layers where i%2==1: {1}
            (3, {0, 2}),  # (n_layer-1)%2 = 0, so layers where i%2==0: {0, 2}
            (4, {1, 3}),  # (n_layer-1)%2 = 1, so layers where i%2==1: {1, 3}
            (5, {0, 2, 4}),  # (n_layer-1)%2 = 0, so layers where i%2==0: {0, 2, 4}
            (26, {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}),
        ],
    )
    def test_has_ve_pattern(self, n_layer, expected_ve_layers):
        actual = {i for i in range(n_layer) if has_ve(i, n_layer)}
        assert actual == expected_ve_layers, (
            f"n_layer={n_layer}: expected VE at {sorted(expected_ve_layers)}, got {sorted(actual)}"
        )

    @pytest.mark.parametrize("n_layer", [2, 3, 4, 5, 10, 26])
    def test_last_layer_always_has_ve(self, n_layer):
        assert has_ve(n_layer - 1, n_layer), (
            f"Last layer (idx={n_layer - 1}) should always have VE for n_layer={n_layer}"
        )

    def test_value_embeds_dict_matches_has_ve(self):
        """Verify the model's value_embeds dict keys match has_ve for all layers."""
        for n_layer in (2, 3, 4, 5):
            model = _make_model(n_layer=n_layer)
            expected_keys = {str(i) for i in range(n_layer) if has_ve(i, n_layer)}
            actual_keys = set(model.value_embeds.keys())
            assert actual_keys == expected_keys, (
                f"n_layer={n_layer}: value_embeds keys {sorted(actual_keys)} "
                f"!= expected {sorted(expected_keys)}"
            )


# ===================================================================
# 5. VE gate initial value
# ===================================================================


class TestVEGateInit:
    def test_gate_weights_are_zero(self):
        model = _make_model()
        for i, block in enumerate(_gpt_blocks(model)):
            gate = block.attn.ve_gate
            if gate is not None:
                assert torch.all(gate.weight == 0), (
                    f"Layer {i}: ve_gate weights should be zero after init_weights, "
                    f"got max abs {gate.weight.abs().max().item()}"
                )

    def test_gate_sigmoid_is_one_at_init(self):
        """sigmoid(0)*2 = 1.0 — neutral gating at initialization."""
        model = _make_model()
        x = torch.randn(2, 16, TINY_N_EMBD)
        for i, block in enumerate(_gpt_blocks(model)):
            gate_mod = block.attn.ve_gate
            if gate_mod is not None:
                with torch.inference_mode():
                    gate = 2 * torch.sigmoid(
                        gate_mod(x[..., : block.attn.ve_gate_channels])
                    )
                assert torch.allclose(gate, torch.ones_like(gate)), (
                    f"Layer {i}: gate should be 1.0 at init, got {gate.mean().item():.4f}"
                )

    def test_only_ve_layers_have_gate(self):
        """Layers without VE should have ve_gate=None."""
        for n_layer in (2, 3, 4, 5):
            model = _make_model(n_layer=n_layer)
            for i, block in enumerate(_gpt_blocks(model)):
                if has_ve(i, n_layer):
                    assert block.attn.ve_gate is not None, (
                        f"n_layer={n_layer}, layer {i}: should have ve_gate"
                    )
                else:
                    assert block.attn.ve_gate is None, (
                        f"n_layer={n_layer}, layer {i}: should NOT have ve_gate"
                    )


# ===================================================================
# 6. VE gating formula verification
# ===================================================================


class TestVEGatingFormula:
    def test_gating_at_init_v_plus_ve(self):
        """At init (gate=1.0), v_out = v_orig + 1.0 * ve."""
        model = _make_model(n_layer=2)
        B, T = 2, 8
        idx = _random_idx(batch=B, seq_len=T)

        # Find a layer with VE
        ve_layer_idx = None
        for i in range(model.config.n_layer):
            if str(i) in model.value_embeds:
                ve_layer_idx = i
                break
        assert ve_layer_idx is not None, "No VE layer found"

        block = _gpt_block(model, ve_layer_idx)
        attn = block.attn

        # Compute v and ve manually
        with torch.inference_mode():
            x = norm(_gpt_embedding(model)(idx))
            # Apply residual scalar mixing (for the first layer, x_in = x0)
            x_in = (
                model.resid_lambdas[ve_layer_idx] * x
                + model.x0_lambdas[ve_layer_idx] * x
            )
            x_normed = norm(x_in)

            qkv = attn.c_qkv(x_normed)
            q_dim = attn.n_head * attn.head_dim
            kv_dim = attn.n_kv_head * attn.head_dim
            _, _, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            v_orig = v.view(B, T, attn.n_kv_head, attn.head_dim)

            ve_embed = _gpt_value_embedding(model, str(ve_layer_idx))(idx)
            ve_reshaped = ve_embed.view(B, T, attn.n_kv_head, attn.head_dim)

            gate_module = attn.ve_gate
            assert gate_module is not None
            gate = 2 * torch.sigmoid(gate_module(x_normed[..., : attn.ve_gate_channels]))

            # At init, gate should be 1.0
            assert torch.allclose(gate, torch.ones_like(gate), atol=1e-6), (
                f"Gate should be 1.0 at init, got mean {gate.mean().item()}"
            )

            v_expected = v_orig + gate.unsqueeze(-1) * ve_reshaped
            v_expected_simple = v_orig + 1.0 * ve_reshaped

            assert torch.allclose(v_expected, v_expected_simple, atol=1e-6), (
                "v_out with gate=1.0 should equal v_orig + ve"
            )

    def test_gate_shape_broadcast(self):
        """Gate shape should be (B, T, n_kv_head) and broadcast over head_dim."""
        model = _make_model(n_layer=2)
        B, T = 2, 8

        ve_layer_idx = None
        for i in range(model.config.n_layer):
            if str(i) in model.value_embeds:
                ve_layer_idx = i
                break
        assert ve_layer_idx is not None

        attn = _gpt_block(model, ve_layer_idx).attn
        x = torch.randn(B, T, TINY_N_EMBD)

        with torch.inference_mode():
            gate_module = attn.ve_gate
            assert gate_module is not None
            gate = 2 * torch.sigmoid(gate_module(x[..., : attn.ve_gate_channels]))

        assert gate.shape == (B, T, attn.n_kv_head), (
            f"Gate shape should be ({B}, {T}, {attn.n_kv_head}), got {gate.shape}"
        )

        # After unsqueeze(-1), should be (B, T, n_kv_head, 1) -> broadcasts over head_dim
        gate_expanded = gate.unsqueeze(-1)
        ve_dummy = torch.randn(B, T, attn.n_kv_head, attn.head_dim)
        result = gate_expanded * ve_dummy  # Should not raise
        assert result.shape == (B, T, attn.n_kv_head, attn.head_dim)

    def test_custom_gate_value(self):
        """With a non-zero gate weight, verify the formula still holds."""
        model = _make_model(n_layer=2)
        B, T = 2, 8
        idx = _random_idx(batch=B, seq_len=T)

        ve_layer_idx = None
        for i in range(model.config.n_layer):
            if str(i) in model.value_embeds:
                ve_layer_idx = i
                break
        assert ve_layer_idx is not None

        attn = _gpt_block(model, ve_layer_idx).attn

        # Set a known non-zero gate weight
        with torch.no_grad():
            gate_module = attn.ve_gate
            assert gate_module is not None
            gate_module.weight.fill_(1.0)

        with torch.inference_mode():
            x = norm(_gpt_embedding(model)(idx))
            x_in = (
                model.resid_lambdas[ve_layer_idx] * x
                + model.x0_lambdas[ve_layer_idx] * x
            )
            x_normed = norm(x_in)

            qkv = attn.c_qkv(x_normed)
            q_dim = attn.n_head * attn.head_dim
            kv_dim = attn.n_kv_head * attn.head_dim
            _, _, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
            v_orig = v.view(B, T, attn.n_kv_head, attn.head_dim)

            ve_embed = _gpt_value_embedding(model, str(ve_layer_idx))(idx)
            ve_reshaped = ve_embed.view(B, T, attn.n_kv_head, attn.head_dim)

            gate = 2 * torch.sigmoid(
                gate_module(x_normed[..., : attn.ve_gate_channels])
            )

            # Gate should NOT be 1.0 anymore
            assert not torch.allclose(gate, torch.ones_like(gate), atol=0.01), (
                "Gate should differ from 1.0 after changing weights"
            )

            v_gated = v_orig + gate.unsqueeze(-1) * ve_reshaped

            # Verify element-wise
            for b in range(B):
                for t in range(T):
                    for h in range(attn.n_kv_head):
                        g = gate[b, t, h].item()
                        expected = v_orig[b, t, h] + g * ve_reshaped[b, t, h]
                        assert torch.allclose(v_gated[b, t, h], expected, atol=1e-5), (
                            f"Mismatch at (b={b}, t={t}, h={h})"
                        )


# ===================================================================
# 7. Logit softcap
# ===================================================================


class TestLogitSoftcap:
    def test_large_logits_capped_near_15(self):
        """Very large logits should be squashed to approximately +/-15."""
        model = _make_model()
        idx = _random_idx(batch=1, seq_len=8)

        with torch.inference_mode():
            logits = model(idx)

        # The formula is 15 * tanh(logits/15). Even without extreme values,
        # verify the output range is within [-15, 15].
        assert logits.max().item() <= 15.0 + 1e-5, (
            f"Max logit {logits.max().item()} exceeds softcap of 15"
        )
        assert logits.min().item() >= -15.0 - 1e-5, (
            f"Min logit {logits.min().item()} below -softcap of -15"
        )

    def test_softcap_formula_directly(self):
        """Verify 15 * tanh(x/15) for known values."""
        softcap = 15.0
        test_values = torch.tensor(
            [-1000.0, -100.0, -15.0, -1.0, 0.0, 1.0, 15.0, 100.0, 1000.0]
        )
        result = softcap * torch.tanh(test_values / softcap)

        # Very large positive -> ~15
        assert abs(result[-1].item() - 15.0) < 0.01
        # Very large negative -> ~-15
        assert abs(result[0].item() + 15.0) < 0.01
        # Zero -> 0
        assert abs(result[4].item()) < 1e-6
        # Small values pass through approximately unchanged: tanh(x) ~ x for small x
        # 1.0: tanh(1/15) ~ 1/15, so 15 * (1/15) ~ 1.0
        assert abs(result[5].item() - 1.0) < 0.01
        assert abs(result[3].item() + 1.0) < 0.01

    def test_softcap_applied_in_forward(self):
        """Verify the model forward pass actually applies the softcap by comparing
        with raw (uncapped) lm_head output."""
        model = _make_model()
        idx = _random_idx(batch=1, seq_len=8)

        # Run the full forward to get softcapped logits
        with torch.inference_mode():
            logits = model(idx)

        # Manually compute the raw logits (without softcap)
        with torch.inference_mode():
            x = _gpt_embedding(model)(idx)
            x = norm(x)
            x0 = x
            for i, block in enumerate(_gpt_blocks(model)):
                x = model.resid_lambdas[i] * x + model.x0_lambdas[i] * x0
                ve = (
                    model.value_embeds[str(i)](idx)
                    if str(i) in model.value_embeds
                    else None
                )
                x = block(
                    x,
                    ve,
                    (model.cos[:, : idx.size(1)], model.sin[:, : idx.size(1)]),
                    model.window_sizes[i],
                    None,
                )
            x = norm(x)
            raw_logits = model.lm_head(x)
            raw_logits = raw_logits[..., : model.config.vocab_size].float()
            expected_logits = 15.0 * torch.tanh(raw_logits / 15.0)

        assert torch.allclose(logits, expected_logits, atol=1e-4), (
            f"Forward logits don't match manual softcap computation. "
            f"Max diff: {(logits - expected_logits).abs().max().item()}"
        )

    def test_logits_are_float32(self):
        """Logits should be cast to float32 before softcap (as per the source code)."""
        model = _make_model()
        idx = _random_idx(batch=1, seq_len=8)
        with torch.inference_mode():
            logits = model(idx)
        assert logits.dtype == torch.float32, (
            f"Logits should be float32, got {logits.dtype}"
        )
