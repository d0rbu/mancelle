"""FP8 dimension alignment tests.

Verifies that all matmul dimensions flowing through FP8 Linear layers are
multiples of 16, as required by scaled_mm in the backward pass.

Key dimensions to check:
- Encoder: the single global packed K axis is 32-aligned
- Packed cross-attention: B*T is governed by the decoder's existing alignment

These tests are designed to FAIL if the FP8 alignment padding is missing,
catching dimension misalignment errors during backward passes.
"""

import copy
import math

import pytest
import torch

from nanochat.fp8_utils import apply_fp8_training
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
    build_packed_cross_attention_block_mask,
)
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP8 tests"
)


# ---------------------------------------------------------------------------
# Model dimensions: all static dims are FP8-eligible (divisible by 16, >= 128)
# ---------------------------------------------------------------------------
DEC_N_EMBD = 256
DEC_N_LAYER = 4
DEC_N_HEAD = 4
HEAD_DIM = 64
ENC_N_EMBD = 128
ENC_DEPTH = 2
XATTN_FREQ = 2
N_SAE_FEATURES = 16384
NUM_BUCKETS = 10
VOCAB_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_model(device="cuda"):
    """Build a tiny encoder-decoder model for testing."""
    dec_config = GPTConfig(
        sequence_len=128,
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
    return model


def _build_batch(B, T, n_seg, F, device="cuda"):
    """Build a synthetic batch with specific dimensions.

    Args:
        B: batch size
        T: sequence length
        n_seg: number of segments
        F: features per segment (NOT necessarily a multiple of 16)
        device: device to create tensors on
    """
    segment_size = T // n_seg
    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    y = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    enc_fids = torch.randint(0, N_SAE_FEATURES, (B, n_seg, F), device=device)
    enc_mags = torch.randint(0, NUM_BUCKETS, (B, n_seg, F), device=device)
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool, device=device)
    seg_map = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) // segment_size
    return x, y, enc_fids, enc_mags, enc_mask, seg_map


def _build_batch_variable_segments(B, T, seg_sizes, F, device="cuda"):
    """Build a synthetic batch with variable-sized segments.

    Args:
        B: batch size
        T: sequence length (must equal sum of seg_sizes)
        seg_sizes: list of segment sizes (determines max_seg_size)
        F: features per segment
        device: device to create tensors on
    """
    assert sum(seg_sizes) == T, f"seg_sizes {seg_sizes} don't sum to T={T}"
    n_seg = len(seg_sizes)

    x = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    y = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    enc_fids = torch.randint(0, N_SAE_FEATURES, (B, n_seg, F), device=device)
    enc_mags = torch.randint(0, NUM_BUCKETS, (B, n_seg, F), device=device)
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool, device=device)

    # Build seg_map from seg_sizes
    seg_map_list = []
    for seg_idx, size in enumerate(seg_sizes):
        seg_map_list.extend([seg_idx] * size)
    seg_map = torch.tensor(seg_map_list, device=device).unsqueeze(0).expand(B, -1)

    return x, y, enc_fids, enc_mags, enc_mask, seg_map


# ---------------------------------------------------------------------------
# Test: Cross-attention FP8 with non-16-aligned max_seg_size
# ---------------------------------------------------------------------------


class TestCrossAttentionFP8Alignment:
    """Tests that cross-attention works under FP8 when max_seg_size is not
    a multiple of 16. This is the bug that the previous FP8 fix missed.

    Packed cross-attention keeps the decoder query axis at B*T. The decoder's
    existing sequence alignment therefore supplies the required scaled_mm
    inner dimension even when individual segment lengths are odd.
    """

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda"

    def test_uniform_segments_not_multiple_of_16(self, device):
        """Segment size of 43 → max_seg_size=43 (not % 16 == 0).

        T=128 stays aligned for the decoder, but the cross-attention
        max_seg_size is 43 (128 = 43 + 43 + 42), which would crash FP8
        without the padding fix.
        """
        B, T = 2, 128
        # 3 segments: [43, 43, 42] → max_seg_size = 43 (not %16)
        seg_sizes = [43, 43, 42]
        F = 16  # Already aligned, isolate the cross-attention issue

        model = _build_model(device)
        apply_fp8_training(model, recipe="tensorwise")

        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()  # Would crash here without cross-attention padding

        # Verify gradients exist for cross-attention params
        for name, param in model.named_parameters():
            if "cross_attn" in name and param.requires_grad:
                assert param.grad is not None, f"{name} missing gradient"
                assert torch.isfinite(param.grad).all(), (
                    f"{name} has non-finite gradient"
                )

    def test_variable_segments_odd_max_size(self, device):
        """Variable segments where the largest has 7 tokens → max_seg_size=7.

        This is an extreme case: 7 is prime, so B*n_seg*7 is unlikely to be
        divisible by 16 for typical B and n_seg values.
        """
        B, T = 2, 128
        # Segments: [7, 57, 64] → max_seg_size = 64 (aligned) — too easy.
        # Let's use: [7, 7, 7, ...] to get max_seg_size = 7
        n_seg = T // 7  # 18 segments of 7 tokens
        remainder = T - n_seg * 7  # 128 - 126 = 2
        seg_sizes = [7] * n_seg
        if remainder > 0:
            seg_sizes[-1] += remainder  # last segment: 7+2=9
            # max_seg_size = 9 (not % 16)

        model = _build_model(device)
        # Override sequence_len for the model (we need T=128)
        F = 16

        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        apply_fp8_training(model, recipe="tensorwise")

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_segment_size_3(self, device):
        """Many small segments of size 3 — max_seg_size=3 (not %16).

        B=2, n_seg=42, max_seg_size=3 → B*n_seg*max_seg_size = 252
        252 % 16 = 12 ≠ 0, which would crash FP8 scaled_mm.
        T=128 keeps the decoder aligned; last segment absorbs remainder.
        """
        B = 2
        T = 128
        n_full = 42  # 42 segments of 3 = 126 tokens
        seg_sizes = [3] * n_full + [T - n_full * 3]  # last segment: 2

        model = _build_model(device)
        F = 16

        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        apply_fp8_training(model, recipe="tensorwise")

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_segment_size_1(self, device):
        """Segment size of 1 — minimal non-aligned size.

        max_seg_size=1, B*n_seg*1 likely not divisible by 16.
        """
        B = 2
        T = 128
        seg_sizes = [1] * T  # 128 segments of 1

        model = _build_model(device)
        F = 16

        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        apply_fp8_training(model, recipe="tensorwise")

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# Test: Encoder FP8 with non-16-aligned feature count
# ---------------------------------------------------------------------------


class TestEncoderFP8Alignment:
    """Tests that the encoder handles non-16-aligned feature counts under FP8.

    The encoder gathers every real feature into one global K axis and rounds
    only that axis to a multiple of 32 before any FP8 Linear layer.
    """

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda"

    @pytest.mark.parametrize("F", [1, 3, 7, 10, 13, 15, 17, 31])
    def test_non_aligned_feature_count(self, device, F):
        """Forward+backward with F features (not multiple of 16) should work."""
        B, T = 2, 128
        n_seg = 2
        T // n_seg

        model = _build_model(device)
        apply_fp8_training(model, recipe="tensorwise")

        batch = _build_batch(B, T, n_seg, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        assert torch.isfinite(loss), f"Loss not finite for F={F}: {loss.item()}"

        # Verify encoder params got gradients
        encoder_grads = 0
        for name, param in model.named_parameters():
            if (
                name.startswith("encoder.")
                and param.requires_grad
                and param.grad is not None
            ):
                assert torch.isfinite(param.grad).all(), (
                    f"{name} has non-finite gradient"
                )
                encoder_grads += 1
        assert encoder_grads > 0, "No encoder parameters received gradients"

    def test_encoder_output_dimension_is_globally_packed(self, device):
        """Encoder output has one 32-aligned axis with an exact valid prefix."""
        enc_config = FeatureEncoderConfig(
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            n_embd=ENC_N_EMBD,
            n_layer=ENC_DEPTH,
            n_head=ENC_N_EMBD // HEAD_DIM,
            n_kv_head=ENC_N_EMBD // HEAD_DIM,
            head_dim=HEAD_DIM,
        )
        encoder = FeatureEncoder(enc_config)
        encoder.to(device)
        encoder.init_weights()

        for Nf in [1, 3, 7, 10, 13, 15, 16, 17, 31, 32, 33]:
            B, S = 2, 2
            fids = torch.randint(0, N_SAE_FEATURES, (B, S, Nf), device=device)
            mags = torch.randint(0, NUM_BUCKETS, (B, S, Nf), device=device)
            mask = torch.ones(B, S, Nf, dtype=torch.bool, device=device)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = encoder(fids, mags, mask)

            real_count = B * S * Nf
            assert out.values.shape[0] == 1
            assert out.values.shape[1] % 32 == 0
            assert out.values.shape[2] == ENC_N_EMBD
            assert int(out.layout.valid_mask.sum()) == real_count
            assert out.layout.valid_mask[:real_count].all()
            assert not out.layout.valid_mask[real_count:].any()


# ---------------------------------------------------------------------------
# Test: Combined encoder + cross-attention non-alignment
# ---------------------------------------------------------------------------


class TestCombinedFP8Alignment:
    """Tests with BOTH non-aligned feature count AND non-aligned segment size.

    This is the most realistic failure scenario: real data has arbitrary
    feature counts and segment sizes.
    """

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda"

    @pytest.mark.parametrize(
        "F,seg_sizes",
        [
            (3, [43, 43, 42]),  # F=3 (not %16), max_seg=43 (not %16)
            (7, [7] * 18 + [2]),  # F=7, max_seg=7, both prime
            (1, [1] * 128),  # Minimal: 1 feature, 1-token segments
            (10, [43, 43, 42]),  # F=10, max_seg=43
            (15, [17] * 7 + [9]),  # F=15, max_seg=17, just below/above 16
        ],
    )
    def test_both_non_aligned(self, device, F, seg_sizes):
        """Forward+backward should work even when both F and max_seg_size
        are not multiples of 16. T=128 keeps decoder aligned.
        """
        B = 2
        T = 128
        assert sum(seg_sizes) == T, f"seg_sizes sum to {sum(seg_sizes)}, expected {T}"

        model = _build_model(device)
        apply_fp8_training(model, recipe="tensorwise")

        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
        loss.backward()

        assert torch.isfinite(loss), (
            f"Loss not finite for F={F}, max_seg={max(seg_sizes)}: {loss.item()}"
        )

    def test_fp8_matches_bf16_with_non_aligned_dims(self, device):
        """FP8 and BF16 should produce comparable results even with
        non-aligned dimensions. T=128 for decoder, seg_size=43 for xattn.
        """
        B, T = 2, 128
        seg_sizes = [43, 43, 42]  # max_seg_size=43 (not %16)
        F = 7  # Not %16

        model = _build_model(device)
        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        # BF16 loss
        bf16_model = copy.deepcopy(model)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            bf16_loss = bf16_model(x, y, enc_fids, enc_mags, enc_mask, seg_map).item()

        # FP8 loss
        fp8_model = copy.deepcopy(model)
        apply_fp8_training(fp8_model, recipe="tensorwise")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            fp8_loss = fp8_model(x, y, enc_fids, enc_mags, enc_mask, seg_map).item()

        assert math.isfinite(bf16_loss), f"BF16 loss not finite: {bf16_loss}"
        assert math.isfinite(fp8_loss), f"FP8 loss not finite: {fp8_loss}"

        rel_diff = abs(fp8_loss - bf16_loss) / abs(bf16_loss)
        assert rel_diff < 0.10, (
            f"FP8 loss ({fp8_loss:.6f}) differs too much from BF16 "
            f"({bf16_loss:.6f}), rel diff: {rel_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: Cross-attention unit test (no full model, isolates the layer)
# ---------------------------------------------------------------------------


class TestCrossAttentionLayerFP8:
    """Unit tests for CrossAttentionLayer under FP8.

    Tests the layer in isolation to precisely identify which matmul fails.
    """

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda"

    @pytest.mark.parametrize("max_seg_size", [1, 3, 5, 7, 9, 11, 13, 15, 17])
    def test_fp8_backward_various_segment_sizes(self, device, max_seg_size):
        """FP8 backward pass through c_q and c_proj with non-aligned max_seg_size."""
        dec_n_embd = 256
        enc_n_embd = 128
        n_head = 4
        n_kv_head = 4
        head_dim = 64

        config = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

        layer = CrossAttentionLayer(config).to(device)
        layer.init_weights()
        # Override c_proj to be non-zero for gradient flow
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        apply_fp8_training(layer, recipe="tensorwise")

        # Packed cross-attention projects the original (B, T) query extent.
        # B=8 keeps B*T divisible by 16 while max_seg_size itself stays odd.
        B = 8
        T = max_seg_size * 2  # 2 segments
        n_seg = 2
        F_feat = 16  # Aligned, to isolate the cross-attention issue

        decoder_hidden = torch.randn(
            B, T, dec_n_embd, device=device, requires_grad=True
        )
        encoder_kv = torch.randn(
            B, n_seg, F_feat, enc_n_embd, device=device, requires_grad=True
        )
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device=device)
        seg_map = (
            torch.arange(T, device=device).unsqueeze(0).expand(B, -1) // max_seg_size
        )

        packed = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed.layout)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = layer(
                decoder_hidden,
                packed,
                seg_map,
                packed_block_mask=block_mask,
            )

        loss = out.sum()
        loss.backward()  # Would crash here without padding fix

        assert decoder_hidden.grad is not None, "decoder_hidden missing gradient"
        assert encoder_kv.grad is not None, "encoder_kv missing gradient"
        assert torch.isfinite(decoder_hidden.grad).all(), (
            "decoder_hidden grad has non-finite values"
        )
        assert torch.isfinite(encoder_kv.grad).all(), (
            "encoder_kv grad has non-finite values"
        )

        # Verify output shape
        assert out.shape == (B, T, dec_n_embd)

    def test_fp8_backward_with_non_aligned_F_feat(self, device):
        """FP8 backward through c_kv with non-aligned F_feat.

        This should work because the encoder pads F_feat to %16 before
        cross-attention sees it. But if someone passes unpadded F_feat
        directly, c_kv backward would fail.
        """
        dec_n_embd = 256
        enc_n_embd = 128
        n_head = 4
        n_kv_head = 4
        head_dim = 64

        config = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
        )

        layer = CrossAttentionLayer(config).to(device)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        apply_fp8_training(layer, recipe="tensorwise")

        B = 2
        T = 32  # 2 segments of 16 (aligned)
        n_seg = 2
        max_seg_size = T // n_seg
        F_feat = 16  # Aligned (encoder ensures this)

        decoder_hidden = torch.randn(
            B, T, dec_n_embd, device=device, requires_grad=True
        )
        encoder_kv = torch.randn(
            B, n_seg, F_feat, enc_n_embd, device=device, requires_grad=True
        )
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device=device)
        seg_map = (
            torch.arange(T, device=device).unsqueeze(0).expand(B, -1) // max_seg_size
        )

        packed = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed.layout)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = layer(
                decoder_hidden,
                packed,
                seg_map,
                packed_block_mask=block_mask,
            )

        loss = out.sum()
        loss.backward()

        assert encoder_kv.grad is not None, "encoder_kv missing gradient"
        assert torch.isfinite(encoder_kv.grad).all(), (
            "encoder_kv grad has non-finite values"
        )


# ---------------------------------------------------------------------------
# Test: Correctness after padding (output should not change for real positions)
# ---------------------------------------------------------------------------


class TestPaddingCorrectness:
    """Verify that global packing is invariant to dense input padding."""

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda"

    def test_encoder_padding_preserves_real_features(self, device):
        """Dense padding columns must not change the packed encoder output."""
        enc_config = FeatureEncoderConfig(
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            n_embd=ENC_N_EMBD,
            n_layer=ENC_DEPTH,
            n_head=ENC_N_EMBD // HEAD_DIM,
            n_kv_head=ENC_N_EMBD // HEAD_DIM,
            head_dim=HEAD_DIM,
        )
        encoder = FeatureEncoder(enc_config)
        encoder.to(device)
        encoder.init_weights()

        B, S, Nf = 2, 2, 7
        torch.manual_seed(42)
        fids = torch.randint(0, N_SAE_FEATURES, (B, S, Nf), device=device)
        mags = torch.randint(0, NUM_BUCKETS, (B, S, Nf), device=device)
        mask = torch.ones(B, S, Nf, dtype=torch.bool, device=device)

        # Run with the natural F=7 input.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out7 = encoder(fids, mags, mask)

        # Run with the same real features in a wider dense producer tensor.
        fids_padded = torch.zeros(B, S, 16, dtype=torch.long, device=device)
        mags_padded = torch.zeros(B, S, 16, dtype=torch.long, device=device)
        mask_padded = torch.zeros(B, S, 16, dtype=torch.bool, device=device)
        fids_padded[:, :, :Nf] = fids
        mags_padded[:, :, :Nf] = mags
        mask_padded[:, :, :Nf] = True

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out16 = encoder(fids_padded, mags_padded, mask_padded)

        assert torch.equal(out7.values, out16.values)
        assert torch.equal(out7.layout.valid_mask, out16.layout.valid_mask)
        assert torch.equal(out7.layout.batch_indices, out16.layout.batch_indices)
        assert torch.equal(out7.layout.segment_indices, out16.layout.segment_indices)

    def test_cross_attention_padding_preserves_output(self, device):
        """Cross-attention output should be the same for real positions
        regardless of whether max_seg_size is padded.
        """
        dec_n_embd = 256
        enc_n_embd = 128

        config = CrossAttentionLayerConfig(
            dec_n_embd=dec_n_embd,
            enc_n_embd=enc_n_embd,
            n_head=4,
            n_kv_head=4,
            head_dim=64,
        )
        layer = CrossAttentionLayer(config).to(device)
        layer.init_weights()
        with torch.no_grad():
            layer.c_proj.weight.normal_(std=0.01)

        # Test with non-aligned segment size (7 tokens per segment)
        B, T, n_seg, F_feat = 2, 14, 2, 16
        seg_size = T // n_seg  # 7

        torch.manual_seed(123)
        decoder_hidden = torch.randn(B, T, dec_n_embd, device=device)
        encoder_kv = torch.randn(B, n_seg, F_feat, enc_n_embd, device=device)
        encoder_mask = torch.ones(B, n_seg, F_feat, dtype=torch.bool, device=device)
        seg_map = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) // seg_size

        packed = pack_dense_encoder_output_reference(encoder_kv, encoder_mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed.layout)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = layer(
                decoder_hidden,
                packed,
                seg_map,
                packed_block_mask=block_mask,
            )

        # Output should have correct shape and no NaNs
        assert out.shape == (B, T, dec_n_embd)
        assert not torch.isnan(out).any(), "Cross-attention output has NaN values"
        assert torch.isfinite(out).all(), "Cross-attention output has non-finite values"


# ---------------------------------------------------------------------------
# Test: Multi-step training convergence under FP8 with non-aligned dims
# ---------------------------------------------------------------------------


class TestFP8TrainingWithNonAlignedDims:
    """End-to-end training test: loss should decrease over multiple steps
    even with non-aligned dimensions.
    """

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda"

    def test_loss_decreases_non_aligned_dims(self, device):
        """Multi-step FP8 training with F=7, max_seg_size=43 should converge."""
        B, T = 2, 128
        seg_sizes = [43, 43, 42]  # max_seg_size=43 (not %16)
        F = 7

        model = _build_model(device)
        apply_fp8_training(model, recipe="tensorwise")

        batch = _build_batch_variable_segments(B, T, seg_sizes, F, device)
        x, y, enc_fids, enc_mags, enc_mask, seg_map = batch

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []

        for _ in range(10):
            model.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y, enc_fids, enc_mags, enc_mask, seg_map)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert all(math.isfinite(loss) for loss in losses), (
            f"Non-finite losses: {losses}"
        )
        assert losses[-1] < losses[0], (
            f"Loss did not decrease with non-aligned dims: "
            f"first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
