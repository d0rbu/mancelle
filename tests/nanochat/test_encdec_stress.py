"""Stress tests for the encoder-decoder model under adversarial conditions.

Targets the NaN-gradient fixes for FP8 dtype leaks and all-False SDPA masks.
Every test constructs its own model to avoid cross-test mutation artifacts.
"""

import pytest
import torch
import torch.nn.functional as F

import nanochat.flash_attention as _fa
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Stress tests need the SDPA backend; flex_attention can fail on tiny
    tensors via torch.compile. Save and restore so the override doesn't leak
    into other test files."""
    saved = _fa._override_impl
    _fa._override_impl = FlashAttnImpl.SDPA
    yield
    _fa._override_impl = saved

# ---------------------------------------------------------------------------
# Shared model configuration constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 64
SEQ_LEN = 32
DEC_N_LAYER = 4
DEC_N_HEAD = 2
DEC_N_KV_HEAD = 2
DEC_N_EMBD = 32

ENC_N_EMBD = 32
ENC_DEPTH = 2
ENC_N_HEAD = 2
ENC_N_KV_HEAD = 2
ENC_HEAD_DIM = 16

XATTN_FREQ = 2
XATTN_N_HEAD = 2
XATTN_N_KV_HEAD = 2

N_SAE_FEATURES = 200
NUM_BUCKETS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_dec_config():
    return GPTConfig(
        sequence_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        n_layer=DEC_N_LAYER,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_KV_HEAD,
        n_embd=DEC_N_EMBD,
        window_pattern="L",
    )


def _make_enc_config(depth=ENC_DEPTH):
    return EncoderConfig(
        n_embd=ENC_N_EMBD,
        depth=depth,
        n_head=ENC_N_HEAD,
        n_kv_head=ENC_N_KV_HEAD,
        head_dim=ENC_HEAD_DIM,
    )


def _make_xattn_config(frequency=XATTN_FREQ):
    return CrossAttentionConfig(
        frequency=frequency,
        n_head=XATTN_N_HEAD,
        n_kv_head=XATTN_N_KV_HEAD,
    )


def _build_model(enc_depth=ENC_DEPTH, xattn_freq=XATTN_FREQ):
    """Build a fresh model on CPU with weights initialized."""
    model = EncoderDecoderGPT(
        dec_config=_make_dec_config(),
        enc_config=_make_enc_config(depth=enc_depth),
        xattn_config=_make_xattn_config(frequency=xattn_freq),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        pad_vocab_size_to=VOCAB_SIZE,
    )
    model.init_weights()
    return model


def _make_batch(B, T, n_seg, Nf, device="cpu"):
    """Create a standard batch of inputs for the encoder-decoder model.

    Returns: (idx, targets, feature_ids, magnitudes, mask, seg_map)
    """
    idx = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    targets = torch.randint(0, VOCAB_SIZE, (B, T), device=device)

    feature_ids = torch.randint(0, N_SAE_FEATURES, (B, n_seg, Nf), device=device)
    magnitudes = torch.randint(0, NUM_BUCKETS, (B, n_seg, Nf), device=device)
    mask = torch.ones(B, n_seg, Nf, dtype=torch.bool, device=device)

    # seg_map: evenly divided segments
    seg_size = T // n_seg
    seg_map = torch.arange(T, device=device).unsqueeze(0).expand(B, -1) // seg_size
    seg_map = seg_map.clamp(max=n_seg - 1)

    return idx, targets, feature_ids, magnitudes, mask, seg_map


def _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map):
    """Run forward + backward and return the loss scalar."""
    loss = model(
        idx=idx,
        targets=targets,
        enc_feature_ids=feature_ids,
        enc_magnitudes=magnitudes,
        enc_mask=mask,
        seg_map=seg_map,
    )
    loss.backward()
    return loss


def _assert_no_nan_grads(model):
    """Assert that no parameter gradient contains NaN."""
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


# ===========================================================================
# Class 1: TestExtremeInputValues
# ===========================================================================
class TestExtremeInputValues:
    """Test with extreme feature IDs, magnitude buckets, and feature counts."""

    B = 2
    T = SEQ_LEN
    N_SEG = 2
    NF_DEFAULT = 8

    def test_large_feature_ids_near_boundary(self):
        """Feature IDs at the upper boundary of n_sae_features."""
        model = _build_model()
        idx, targets, _, magnitudes, mask, seg_map = _make_batch(
            self.B, self.T, self.N_SEG, self.NF_DEFAULT
        )
        # Set all feature IDs to the maximum valid index
        feature_ids = torch.full(
            (self.B, self.N_SEG, self.NF_DEFAULT),
            N_SAE_FEATURES - 1,
            dtype=torch.long,
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_magnitude_bucket_zero_everywhere(self):
        """All magnitude buckets set to 0 (minimum)."""
        model = _build_model()
        idx, targets, feature_ids, _, mask, seg_map = _make_batch(
            self.B, self.T, self.N_SEG, self.NF_DEFAULT
        )
        magnitudes = torch.zeros(
            self.B, self.N_SEG, self.NF_DEFAULT, dtype=torch.long
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_magnitude_bucket_max_everywhere(self):
        """All magnitude buckets set to num_buckets-1 (maximum)."""
        model = _build_model()
        idx, targets, feature_ids, _, mask, seg_map = _make_batch(
            self.B, self.T, self.N_SEG, self.NF_DEFAULT
        )
        magnitudes = torch.full(
            (self.B, self.N_SEG, self.NF_DEFAULT),
            NUM_BUCKETS - 1,
            dtype=torch.long,
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_single_feature_per_segment(self):
        """Nf=1: minimal number of features per segment."""
        Nf = 1
        model = _build_model()
        idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
            self.B, self.T, self.N_SEG, Nf
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_many_features_per_segment(self):
        """Nf=100: large number of features per segment."""
        Nf = 100
        model = _build_model()
        idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
            self.B, self.T, self.N_SEG, Nf
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)


# ===========================================================================
# Class 2: TestAdversarialMasks
# ===========================================================================
class TestAdversarialMasks:
    """Test with adversarial encoder mask patterns that stress SDPA masking."""

    B = 2
    T = SEQ_LEN
    N_SEG = 2
    NF = 16

    def _run_with_mask(self, mask):
        """Build model, run forward+backward with given mask, assert no NaN."""
        model = _build_model()
        idx, targets, feature_ids, magnitudes, _, seg_map = _make_batch(
            self.B, self.T, self.N_SEG, self.NF
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_alternating_mask_pattern(self):
        """Alternating True/False within each segment's feature dimension."""
        mask = torch.zeros(self.B, self.N_SEG, self.NF, dtype=torch.bool)
        mask[:, :, ::2] = True  # True at even indices
        self._run_with_mask(mask)

    def test_only_last_position_true(self):
        """Only the last feature position is True in each segment."""
        mask = torch.zeros(self.B, self.N_SEG, self.NF, dtype=torch.bool)
        mask[:, :, -1] = True
        self._run_with_mask(mask)

    def test_only_first_position_true(self):
        """Only the first feature position is True in each segment."""
        mask = torch.zeros(self.B, self.N_SEG, self.NF, dtype=torch.bool)
        mask[:, :, 0] = True
        self._run_with_mask(mask)

    def test_mixed_all_true_all_false_rows(self):
        """Row 0 has all-True mask, row 1 has all-False mask.

        The all-False row is the critical adversarial case: SDPA softmax
        on all-masked keys produces NaN without nan_to_num protection.
        """
        mask = torch.zeros(self.B, self.N_SEG, self.NF, dtype=torch.bool)
        mask[0, :, :] = True  # row 0: all True
        # row 1: all False (adversarial)
        self._run_with_mask(mask)

    def test_different_segment_mask_densities(self):
        """Segment 0 has sparse mask (25%), segment 1 has dense mask (75%)."""
        mask = torch.zeros(self.B, self.N_SEG, self.NF, dtype=torch.bool)
        # Segment 0: 25% density (4 of 16)
        mask[:, 0, :4] = True
        # Segment 1: 75% density (12 of 16)
        mask[:, 1, :12] = True
        self._run_with_mask(mask)


# ===========================================================================
# Class 3: TestMinimalConfigurations
# ===========================================================================
class TestMinimalConfigurations:
    """Test minimal and degenerate model configurations."""

    def test_batch_size_1_seq_len_1(self):
        """Minimum viable forward: B=1, T=1."""
        model = EncoderDecoderGPT(
            dec_config=GPTConfig(
                sequence_len=SEQ_LEN,
                vocab_size=VOCAB_SIZE,
                n_layer=DEC_N_LAYER,
                n_head=DEC_N_HEAD,
                n_kv_head=DEC_N_KV_HEAD,
                n_embd=DEC_N_EMBD,
                window_pattern="L",
            ),
            enc_config=_make_enc_config(),
            xattn_config=_make_xattn_config(),
            n_sae_features=N_SAE_FEATURES,
            num_buckets=NUM_BUCKETS,
            pad_vocab_size_to=VOCAB_SIZE,
        )
        model.init_weights()

        B, T, n_seg, Nf = 1, 1, 1, 4
        idx = torch.randint(0, VOCAB_SIZE, (B, T))
        targets = torch.randint(0, VOCAB_SIZE, (B, T))
        feature_ids = torch.randint(0, N_SAE_FEATURES, (B, n_seg, Nf))
        magnitudes = torch.randint(0, NUM_BUCKETS, (B, n_seg, Nf))
        mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)

        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_single_segment_covering_entire_sequence(self):
        """One segment spans the entire sequence (n_seg=1)."""
        model = _build_model()
        B, T, n_seg, Nf = 2, SEQ_LEN, 1, 8
        idx = torch.randint(0, VOCAB_SIZE, (B, T))
        targets = torch.randint(0, VOCAB_SIZE, (B, T))
        feature_ids = torch.randint(0, N_SAE_FEATURES, (B, n_seg, Nf))
        magnitudes = torch.randint(0, NUM_BUCKETS, (B, n_seg, Nf))
        mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.zeros(B, T, dtype=torch.long)  # all positions map to segment 0

        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_encoder_depth_zero_raises(self):
        """Encoder with depth=0 is invalid and should raise ValueError."""
        with pytest.raises(ValueError, match="at least 1 layer"):
            _build_model(enc_depth=0)

    def test_xattn_frequency_equals_n_layer(self):
        """Cross-attention only at the last decoder layer (frequency = n_layer)."""
        model = _build_model(xattn_freq=DEC_N_LAYER)
        B, T, n_seg, Nf = 2, SEQ_LEN, 2, 8
        idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
            B, T, n_seg, Nf
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)

    def test_zero_initialized_encoder_embeddings(self):
        """Zero out encoder embeddings -- adversarial for RMSNorm (norm of zero)."""
        model = _build_model()
        # Zero out both embedding tables
        with torch.no_grad():
            model.encoder.feature_embed.weight.zero_()
            model.encoder.magnitude_embed.weight.zero_()

        B, T, n_seg, Nf = 2, SEQ_LEN, 2, 8
        idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
            B, T, n_seg, Nf
        )
        loss = _forward_backward(model, idx, targets, feature_ids, magnitudes, mask, seg_map)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        _assert_no_nan_grads(model)


# ===========================================================================
# Class 4: TestGradientAccumulation
# ===========================================================================
class TestGradientAccumulation:
    """Test gradient accumulation across micro-steps."""

    B = 2
    T = SEQ_LEN
    N_SEG = 2
    NF = 8
    MICRO_STEPS = 3

    def test_accumulation_no_nan(self):
        """3 micro-steps of forward+backward without optimizer.step() -- no NaN."""
        model = _build_model()
        model.zero_grad()

        for _ in range(self.MICRO_STEPS):
            idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
                self.B, self.T, self.N_SEG, self.NF
            )
            loss = model(
                idx=idx,
                targets=targets,
                enc_feature_ids=feature_ids,
                enc_magnitudes=magnitudes,
                enc_mask=mask,
                seg_map=seg_map,
            )
            (loss / self.MICRO_STEPS).backward()

        _assert_no_nan_grads(model)

    def test_accumulated_gradient_magnitudes_reasonable(self):
        """After accumulation, gradient magnitudes should not explode (< 100)."""
        model = _build_model()
        model.zero_grad()

        for _ in range(self.MICRO_STEPS):
            idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
                self.B, self.T, self.N_SEG, self.NF
            )
            loss = model(
                idx=idx,
                targets=targets,
                enc_feature_ids=feature_ids,
                enc_magnitudes=magnitudes,
                enc_mask=mask,
                seg_map=seg_map,
            )
            (loss / self.MICRO_STEPS).backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                max_grad = p.grad.abs().max().item()
                assert max_grad < 100, (
                    f"Gradient explosion in {name}: max |grad| = {max_grad}"
                )

    def test_label_dropout_masking_per_microstep(self):
        """Simulate label dropout: mask out different segments each micro-step."""
        model = _build_model()
        model.zero_grad()

        for step in range(self.MICRO_STEPS):
            idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
                self.B, self.T, self.N_SEG, self.NF
            )
            # Mask out one segment per micro-step (rotating)
            seg_to_mask = step % self.N_SEG
            mask[:, seg_to_mask, :] = False

            loss = model(
                idx=idx,
                targets=targets,
                enc_feature_ids=feature_ids,
                enc_magnitudes=magnitudes,
                enc_mask=mask,
                seg_map=seg_map,
            )
            (loss / self.MICRO_STEPS).backward()

        _assert_no_nan_grads(model)


# ===========================================================================
# Class 5: TestMultiStepStability
# ===========================================================================
class TestMultiStepStability:
    """Test training stability over multiple optimizer steps."""

    B = 2
    T = SEQ_LEN
    N_SEG = 2
    NF = 8
    N_STEPS = 20
    LR = 1e-4

    def test_multi_step_loss_finite(self):
        """20 training steps: loss must be finite at every step."""
        model = _build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)

        losses = []
        for _ in range(self.N_STEPS):
            optimizer.zero_grad()
            idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
                self.B, self.T, self.N_SEG, self.NF
            )
            loss = model(
                idx=idx,
                targets=targets,
                enc_feature_ids=feature_ids,
                enc_magnitudes=magnitudes,
                enc_mask=mask,
                seg_map=seg_map,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        for i, l in enumerate(losses):
            assert torch.isfinite(torch.tensor(l)), f"Loss not finite at step {i}: {l}"

    def test_multi_step_no_nan_grads(self):
        """20 training steps: no NaN gradients at any step."""
        model = _build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)

        for step in range(self.N_STEPS):
            optimizer.zero_grad()
            idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
                self.B, self.T, self.N_SEG, self.NF
            )
            loss = model(
                idx=idx,
                targets=targets,
                enc_feature_ids=feature_ids,
                enc_magnitudes=magnitudes,
                enc_mask=mask,
                seg_map=seg_map,
            )
            loss.backward()
            _assert_no_nan_grads(model)
            optimizer.step()

    def test_multi_step_loss_variance_reasonable(self):
        """20 training steps: loss variance should not oscillate wildly.

        With a tiny learning rate and random data, loss should remain
        in a bounded range. We check that the standard deviation of loss
        values is less than 2x the mean (a very generous bound).
        """
        model = _build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)

        losses = []
        for _ in range(self.N_STEPS):
            optimizer.zero_grad()
            idx, targets, feature_ids, magnitudes, mask, seg_map = _make_batch(
                self.B, self.T, self.N_SEG, self.NF
            )
            loss = model(
                idx=idx,
                targets=targets,
                enc_feature_ids=feature_ids,
                enc_magnitudes=magnitudes,
                enc_mask=mask,
                seg_map=seg_map,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        loss_tensor = torch.tensor(losses)
        mean_loss = loss_tensor.mean().item()
        std_loss = loss_tensor.std().item()
        assert std_loss < 2 * abs(mean_loss), (
            f"Loss oscillation too high: mean={mean_loss:.4f}, std={std_loss:.4f}"
        )
