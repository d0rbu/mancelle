"""FP8 correctness tests for encoder-decoder training.

Tests that FP8 conversion works correctly on EncoderDecoderGPT:
- Encoder, decoder, and cross-attention layers all get FP8 conversion where eligible
- Gradients flow through all FP8 parameters
- FP8 and BF16 produce comparable loss/gradients
- disable_fp8 context manager works correctly with nested encoder modules
"""

import copy
import math

import pytest
import torch
import torch.nn as nn

from nanochat.fp8_utils import apply_fp8_training, disable_fp8
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT


# Use dims that are all FP8-eligible (divisible by 16, >= 128)
# to maximize conversion and test coverage
DEC_N_EMBD = 256
DEC_N_LAYER = 4
DEC_N_HEAD = 4
HEAD_DIM = 64
ENC_N_EMBD = 128
ENC_DEPTH = 2
XATTN_FREQ = 2  # cross-attn every 2 decoder layers
N_SAE_FEATURES = 16384
NUM_BUCKETS = 10
VOCAB_SIZE = 256  # divisible by 64 for padding
B = 2
T = 128
SEGMENT_SIZE = 64
N_SEG = T // SEGMENT_SIZE
F_MAX = 8  # features per segment


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP8 tests"
)


@pytest.fixture(scope="module")
def device():
    return "cuda"


@pytest.fixture(scope="module")
def model_and_batch(device):
    """Build tiny encoder-decoder model and synthetic batch."""
    dec_config = GPTConfig(
        sequence_len=T,
        vocab_size=VOCAB_SIZE,
        n_layer=DEC_N_LAYER,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_HEAD,
        n_embd=DEC_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=ENC_N_EMBD, depth=ENC_DEPTH, head_dim=HEAD_DIM)
    xattn_config = CrossAttentionConfig(frequency=XATTN_FREQ)

    with torch.device("meta"):
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
        )
    model.to_empty(device=device)
    model.init_weights()

    # Synthetic batch
    torch.manual_seed(42)
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    y = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    enc_fids = torch.randint(0, N_SAE_FEATURES, (B, N_SEG, F_MAX), device=device)
    enc_mags = torch.randint(0, NUM_BUCKETS, (B, N_SEG, F_MAX), device=device)
    enc_mask = torch.ones(B, N_SEG, F_MAX, dtype=torch.bool, device=device)
    seg_map = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) // SEGMENT_SIZE

    batch = (x, y, enc_fids, enc_mags, enc_mask, seg_map)
    return model, batch


class TestFP8LayerConversion:
    """Test which layers get converted to FP8 and verify breakdown."""

    def test_conversion_counts_by_component(self, model_and_batch):
        model, _ = model_and_batch
        test_model = copy.deepcopy(model)

        num_fp8, num_skipped = apply_fp8_training(test_model, recipe="tensorwise")

        # Count FP8 modules by component
        encoder_fp8 = sum(
            1
            for n, m in test_model.named_modules()
            if "Float8" in type(m).__name__ and n.startswith("encoder.")
        )
        decoder_fp8 = sum(
            1
            for n, m in test_model.named_modules()
            if "Float8" in type(m).__name__ and n.startswith("transformer.")
        )
        xattn_fp8 = sum(
            1
            for n, m in test_model.named_modules()
            if "Float8" in type(m).__name__ and n.startswith("cross_attn_layers.")
        )
        lm_head_fp8 = sum(
            1
            for n, m in test_model.named_modules()
            if "Float8" in type(m).__name__ and n == "lm_head"
        )

        # At least some layers in each component should be converted
        assert encoder_fp8 > 0, (
            f"No encoder layers converted to FP8 (total FP8: {num_fp8})"
        )
        assert decoder_fp8 > 0, (
            f"No decoder layers converted to FP8 (total FP8: {num_fp8})"
        )
        assert xattn_fp8 > 0, (
            f"No cross-attn layers converted to FP8 (total FP8: {num_fp8})"
        )

        # Log breakdown for visibility
        print("\nFP8 conversion breakdown:")
        print(f"  Encoder:        {encoder_fp8}")
        print(f"  Decoder:        {decoder_fp8}")
        print(f"  Cross-attention: {xattn_fp8}")
        print(f"  LM head:        {lm_head_fp8}")
        print(f"  Total FP8:      {num_fp8}, skipped: {num_skipped}")

    def test_encoder_linear_dimensions_logged(self, model_and_batch):
        """Report all encoder Linear layer dimensions for FP8 eligibility analysis."""
        model, _ = model_and_batch
        print("\nEncoder Linear layer dimensions:")
        for name, mod in model.named_modules():
            if name.startswith("encoder.") and isinstance(mod, nn.Linear):
                eligible = (
                    mod.in_features % 16 == 0
                    and mod.out_features % 16 == 0
                    and min(mod.in_features, mod.out_features) >= 128
                )
                status = "ELIGIBLE" if eligible else "SKIPPED"
                print(f"  {name}: ({mod.in_features}, {mod.out_features}) -> {status}")


class TestFP8GradientFlow:
    """Test that gradients flow correctly through FP8 layers."""

    def test_all_params_get_gradients(self, model_and_batch):
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch
        test_model = copy.deepcopy(model)
        apply_fp8_training(test_model, recipe="tensorwise")

        test_model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = test_model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        missing_grad = []
        for name, param in test_model.named_parameters():
            if param.requires_grad and param.grad is None:
                missing_grad.append(name)

        assert len(missing_grad) == 0, (
            f"{len(missing_grad)} parameter(s) got no gradient: {missing_grad}"
        )

    def test_no_nan_gradients(self, model_and_batch):
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch
        test_model = copy.deepcopy(model)
        apply_fp8_training(test_model, recipe="tensorwise")

        test_model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = test_model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        nan_grads = []
        for name, param in test_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)

        assert len(nan_grads) == 0, (
            f"{len(nan_grads)} parameter(s) have NaN gradients: {nan_grads}"
        )

    def test_encoder_grad_norms_reasonable(self, model_and_batch):
        """After one training step, encoder block params should have non-zero,
        finite gradients.

        At exact initialization, c_proj layers are zero-initialized (training
        stability pattern), which zeros out the backward gradient through the
        encoder blocks. One optimizer step breaks this symmetry, after which
        all encoder block parameters should receive non-trivial gradients.
        """
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch
        test_model = copy.deepcopy(model)
        apply_fp8_training(test_model, recipe="tensorwise")

        optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)

        # Step 1: break zero-init symmetry
        test_model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = test_model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()
        optimizer.step()

        # Step 2: now check gradients (c_proj is no longer zero)
        test_model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = test_model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        nonzero_count = 0
        checked_count = 0
        for name, param in test_model.named_parameters():
            if not name.startswith("encoder.blocks."):
                continue
            if param.requires_grad and param.grad is not None:
                checked_count += 1
                grad_norm = param.grad.norm().item()
                assert math.isfinite(grad_norm), (
                    f"{name} has non-finite gradient norm: {grad_norm}"
                )
                if grad_norm > 0:
                    nonzero_count += 1

        assert checked_count > 0, "No encoder block parameters found"
        assert nonzero_count > 0, (
            f"All {checked_count} encoder block parameters have zero gradient norm"
        )


class TestFP8LossConvergence:
    """Test that FP8 training produces comparable loss to BF16."""

    def test_fp8_loss_comparable_to_bf16(self, model_and_batch, device):
        """FP8 and BF16 forward passes should produce loss within tolerance."""
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch

        # BF16 loss
        bf16_model = copy.deepcopy(model)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bf16_loss = bf16_model(x, y, enc_fids, enc_mags, enc_mask, seg_map).item()

        # FP8 loss
        fp8_model = copy.deepcopy(model)
        apply_fp8_training(fp8_model, recipe="tensorwise")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fp8_loss = fp8_model(x, y, enc_fids, enc_mags, enc_mask, seg_map).item()

        # Losses should be close (FP8 rounding introduces small differences)
        rel_diff = abs(fp8_loss - bf16_loss) / bf16_loss
        print(
            f"\nBF16 loss: {bf16_loss:.6f}, FP8 loss: {fp8_loss:.6f}, rel diff: {rel_diff:.4f}"
        )
        assert rel_diff < 0.05, (
            f"FP8 loss ({fp8_loss:.6f}) differs from BF16 ({bf16_loss:.6f}) by {rel_diff:.1%}"
        )

    def test_fp8_multi_step_loss_decreases(self, model_and_batch, device):
        """Loss should decrease over multiple FP8 training steps."""
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch
        test_model = copy.deepcopy(model)
        apply_fp8_training(test_model, recipe="tensorwise")

        optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)
        losses = []

        for step in range(10):
            test_model.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = test_model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"\nFP8 training losses: {[f'{loss:.4f}' for loss in losses]}")
        # Loss should decrease overall (last loss < first loss)
        assert losses[-1] < losses[0], (
            f"FP8 loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


class TestDisableFP8WithEncoderDecoder:
    """Test disable_fp8 context manager with the full encoder-decoder model."""

    def test_disable_restores_correctly(self, model_and_batch):
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch
        test_model = copy.deepcopy(model)
        apply_fp8_training(test_model, recipe="tensorwise")

        num_fp8_before = sum(
            1 for m in test_model.modules() if "Float8" in type(m).__name__
        )
        assert num_fp8_before > 0

        with disable_fp8(test_model):
            num_fp8_inside = sum(
                1 for m in test_model.modules() if "Float8" in type(m).__name__
            )
            assert num_fp8_inside == 0, (
                f"Found {num_fp8_inside} FP8 modules inside disable_fp8"
            )

            # Eval should work in BF16
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = test_model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            assert not torch.isnan(loss), "BF16 eval loss is NaN inside disable_fp8"

        num_fp8_after = sum(
            1 for m in test_model.modules() if "Float8" in type(m).__name__
        )
        assert num_fp8_after == num_fp8_before, (
            f"FP8 modules not restored: before={num_fp8_before}, after={num_fp8_after}"
        )

    def test_disable_eval_matches_bf16_eval(self, model_and_batch):
        """Loss from disable_fp8 eval should match pure BF16 model eval."""
        model, (x, y, enc_fids, enc_mags, enc_mask, seg_map) = model_and_batch

        # Pure BF16
        bf16_model = copy.deepcopy(model)
        bf16_model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bf16_loss = bf16_model(x, y, enc_fids, enc_mags, enc_mask, seg_map).item()

        # FP8 model with disable_fp8 for eval
        fp8_model = copy.deepcopy(model)
        apply_fp8_training(fp8_model, recipe="tensorwise")
        fp8_model.eval()
        with disable_fp8(fp8_model):
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                disabled_loss = fp8_model(
                    x, y, enc_fids, enc_mags, enc_mask, seg_map
                ).item()

        # Should be identical since weights are shared and both run in BF16
        assert bf16_loss == disabled_loss, (
            f"disable_fp8 eval ({disabled_loss:.6f}) != BF16 eval ({bf16_loss:.6f})"
        )


class TestFP8ModuleFilter:
    """Unit tests for fp8_module_filter eligibility logic."""

    def test_non_linear_module_rejected(self):
        """Non-nn.Linear modules must be filtered out (line 25 branch)."""
        from nanochat.fp8_utils import fp8_module_filter

        assert fp8_module_filter(nn.Conv2d(128, 128, 3), "conv") is False
        assert fp8_module_filter(nn.LayerNorm(128), "ln") is False
        assert fp8_module_filter(nn.Embedding(1000, 128), "emb") is False

    def test_small_dimension_rejected(self):
        """Linear layers with any dimension < 128 must be filtered out (line 29 branch)."""
        from nanochat.fp8_utils import fp8_module_filter

        # Exactly at the boundary — 64 < 128, must be rejected
        assert fp8_module_filter(nn.Linear(64, 256), "small_in") is False
        assert fp8_module_filter(nn.Linear(256, 64), "small_out") is False
        # Both dimensions exactly 128 — eligible
        assert fp8_module_filter(nn.Linear(128, 128), "min_eligible") is True

    def test_non_multiple_of_16_rejected(self):
        """Linear layers with dimensions not divisible by 16 must be filtered out."""
        from nanochat.fp8_utils import fp8_module_filter

        assert fp8_module_filter(nn.Linear(130, 256), "not_div16_in") is False
        assert fp8_module_filter(nn.Linear(256, 130), "not_div16_out") is False

    def test_eligible_layer_accepted(self):
        """Large Linear layers with dims divisible by 16 must pass."""
        from nanochat.fp8_utils import fp8_module_filter

        assert fp8_module_filter(nn.Linear(256, 512), "eligible") is True


class TestDisableFP8EdgeCases:
    """Edge-case coverage for disable_fp8 paths not exercised by the full-model tests."""

    def test_no_fp8_layers_is_noop(self):
        """disable_fp8 on a plain BF16 model must yield cleanly without touching anything (lines 71-72)."""
        plain_model = nn.Linear(256, 256)
        modules_before = list(plain_model.named_modules())

        with disable_fp8(plain_model):
            modules_inside = list(plain_model.named_modules())

        modules_after = list(plain_model.named_modules())
        assert modules_before == modules_inside == modules_after, (
            "disable_fp8 on a non-FP8 model must not mutate the module tree"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP8")
    def test_bias_preserved_across_disable_fp8(self):
        """Float8Linear with bias: disable_fp8 must copy bias to the swap Linear (line 84)."""
        device = "cuda"

        # Build a minimal model with a biased Linear large enough for FP8
        class BiasedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(256, 256, bias=True, device=device)

        model = BiasedModel()
        original_bias = model.fc.bias.data.clone()

        apply_fp8_training(model, recipe="tensorwise")
        assert "Float8" in type(model.fc).__name__, "fc should have been converted to Float8Linear"

        with disable_fp8(model):
            # Inside context: module is a plain Linear — bias must still be accessible and correct
            assert hasattr(model.fc, "bias") and model.fc.bias is not None, (
                "Bias missing inside disable_fp8"
            )
            assert torch.equal(model.fc.bias.data, original_bias), (
                "Bias data changed inside disable_fp8"
            )

        # After context: Float8Linear is restored with the original bias
        assert "Float8" in type(model.fc).__name__, "Float8Linear not restored after disable_fp8"
        assert torch.equal(model.fc.bias.data, original_bias), (
            "Bias data changed after disable_fp8 restore"
        )
