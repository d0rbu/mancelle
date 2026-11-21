"""Tests for previously untested methods in EncoderDecoderGPT.

Covers:
- _compute_window_sizes
- init_buffers
- get_device
- num_scaling_params
"""

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import Block, GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.encoder import EncoderBlock
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.typed_module_list import TypedModuleList
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_SEQ_LEN,
    TINY_VOCAB,
    _make_encdec,
)

# ---------------------------------------------------------------------------
# Force SDPA for every test in this module (FlexAttention has no CPU backward)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_model(window_pattern: str = "L", n_layer: int = TINY_N_LAYER) -> EncoderDecoderGPT:
    """Build a tiny EncoderDecoderGPT with the given window_pattern and n_layer."""
    dec_config = GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=n_layer,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern=window_pattern,
    )
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(frequency=1)
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=xattn_config,
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=TINY_NUM_BUCKETS,
        pad_vocab_size_to=TINY_VOCAB,
    )
    model.init_weights()
    model.eval()
    return model


def _make_forward_inputs(model: EncoderDecoderGPT, B: int = 1, T: int = 8, n_seg: int = 2, F: int = 4):
    """Create minimal valid forward-pass tensors."""
    idx = t.randint(0, model.dec_config.vocab_size, (B, T))
    targets = t.randint(0, model.dec_config.vocab_size, (B, T))
    enc_feature_ids = t.randint(0, TINY_N_SAE_FEATURES, (B, n_seg, F))
    enc_magnitudes = t.randint(0, TINY_NUM_BUCKETS, (B, n_seg, F))
    enc_mask = t.ones(B, n_seg, F, dtype=t.bool)
    seg_size = T // n_seg
    seg_map = t.arange(T).unsqueeze(0).expand(B, -1) // seg_size
    seg_map = seg_map.clamp(max=n_seg - 1)
    return idx, targets, enc_feature_ids, enc_magnitudes, enc_mask, seg_map


# ---------------------------------------------------------------------------
# TestComputeWindowSizes
# ---------------------------------------------------------------------------


class TestComputeWindowSizes:
    def test_all_long_windows(self):
        """Default model with window_pattern='L' creates successfully."""
        model = _make_tiny_model(window_pattern="L")
        assert len(model.window_sizes) == TINY_N_LAYER
        # Every layer should use the long (full) window
        long_window = TINY_SEQ_LEN
        for ws in model.window_sizes:
            assert ws[0] == long_window, (
                f"Expected all windows to be {long_window}, got {ws[0]}"
            )

    def test_mixed_sl_pattern(self):
        """Model with window_pattern='SL' and 4 layers creates successfully."""
        n_layer = 4
        model = _make_tiny_model(window_pattern="SL", n_layer=n_layer)
        assert len(model.window_sizes) == n_layer
        long_window = TINY_SEQ_LEN
        short_window = long_window // 2
        # Pattern "SL" repeats: S, L, S, L — but last layer is always forced to L
        # Layer 0 → 'S', Layer 1 → 'L', Layer 2 → 'S', Layer 3 → 'L' (last forced L)
        expected = [
            (short_window, 0),  # layer 0: S
            (long_window, 0),   # layer 1: L
            (short_window, 0),  # layer 2: S
            (long_window, 0),   # layer 3: L (last layer always L)
        ]
        assert model.window_sizes == expected, (
            f"Window sizes mismatch.\nExpected: {expected}\nGot:      {model.window_sizes}"
        )


# ---------------------------------------------------------------------------
# TestInitBuffers
# ---------------------------------------------------------------------------


class TestInitBuffers:
    def test_init_buffers_preserves_parameters(self):
        """init_buffers() must not change any learned parameter values."""
        model = _make_encdec(device="cpu")

        # Record parameter sums before
        param_sums_before = {
            name: p.sum().item() for name, p in model.named_parameters()
        }

        model.init_buffers()

        # Verify sums are unchanged
        for name, p in model.named_parameters():
            before = param_sums_before[name]
            after = p.sum().item()
            assert before == after, (
                f"Parameter '{name}' changed after init_buffers(): {before} → {after}"
            )

    def test_init_buffers_forward_still_works(self):
        """After calling init_buffers(), a forward pass must not raise."""
        model = _make_encdec(device="cpu")
        model.init_buffers()

        idx, targets, enc_fids, enc_mags, enc_mask, seg_map = _make_forward_inputs(model)
        with t.no_grad():
            loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)

        assert loss.ndim == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"


# ---------------------------------------------------------------------------
# TestGetDevice
# ---------------------------------------------------------------------------


class TestGetDevice:
    def test_returns_cpu_for_cpu_model(self):
        """get_device() returns torch.device('cpu') for a CPU model."""
        model = _make_encdec(device="cpu")
        device = model.get_device()
        assert device == t.device("cpu"), (
            f"Expected torch.device('cpu'), got {device}"
        )


# ---------------------------------------------------------------------------
# TestTypedDecoderTopology
# ---------------------------------------------------------------------------


class TestTypedDecoderTopology:
    def test_preserves_public_module_dict_and_state_dict_layout(self):
        model = _make_tiny_model()

        assert isinstance(model.transformer, t.nn.ModuleDict)
        assert isinstance(model.transformer["h"], t.nn.ModuleList)
        state_keys = set(model.state_dict())
        assert "transformer.wte.weight" in state_keys
        assert any(key.startswith("transformer.h.0.") for key in state_keys)
        assert not any("_decoder_" in key for key in state_keys)

    def test_rejects_non_blocks_at_initialization(self):
        model = _make_tiny_model()
        blocks = model.transformer["h"]
        assert isinstance(blocks, t.nn.ModuleList)

        with pytest.raises(TypeError, match=r"decoder block 0 must be Block"):
            TypedModuleList(
                [t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD)],
                module_type=Block,
                item_label="EncoderDecoderGPT decoder block",
            )

    def test_rejects_non_blocks_at_mutation_boundaries(self):
        model = _make_tiny_model()
        blocks = model.transformer["h"]
        assert isinstance(blocks, t.nn.ModuleList)
        invalid_block = t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD)

        with pytest.raises(TypeError, match=r"decoder block 2 must be Block"):
            blocks.append(invalid_block)
        with pytest.raises(TypeError, match=r"decoder block 0 must be Block"):
            blocks[0] = invalid_block
        with pytest.raises(TypeError, match=r"decoder block 1 must be Block"):
            blocks.insert(1, invalid_block)
        with pytest.raises(TypeError, match=r"decoder block 2 must be Block"):
            blocks.extend([invalid_block])

    def test_slice_preserves_decoder_block_contract(self):
        model = _make_tiny_model()
        blocks = model._decoder_blocks[:1]

        assert blocks.module_type is Block
        with pytest.raises(TypeError, match=r"decoder block 1 must be Block"):
            blocks.append(t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD))

    def test_rejects_typed_list_with_wrong_module_type(self):
        model = _make_tiny_model()
        model.transformer["h"] = TypedModuleList(
            [t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD)],
            module_type=t.nn.Linear,
            item_label="linear",
        )

        with pytest.raises(TypeError, match=r"must be TypedModuleList\[Block\]"):
            _ = model._decoder_blocks

    def test_corrupted_embedding_fails_at_typed_boundary(self):
        model = _make_tiny_model()
        model.transformer["wte"] = t.nn.Linear(
            model.dec_config.n_embd,
            model.dec_config.n_embd,
        )

        with pytest.raises(
            TypeError, match=r"transformer\['wte'\] must be nn\.Embedding"
        ):
            model.get_device()


# ---------------------------------------------------------------------------
# TestTypedEncoderTopology
# ---------------------------------------------------------------------------


class TestTypedEncoderTopology:
    def test_rejects_non_blocks_at_initialization(self):
        with pytest.raises(TypeError, match=r"block 0 must be EncoderBlock"):
            TypedModuleList(
                [t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD)],
                module_type=EncoderBlock,
                item_label="FeatureEncoder block",
            )

    def test_rejects_non_blocks_at_mutation_boundaries(self):
        model = _make_tiny_model()
        blocks: t.nn.ModuleList = model.encoder.blocks
        invalid_block = t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD)

        with pytest.raises(TypeError, match=r"block 2 must be EncoderBlock"):
            blocks.append(invalid_block)
        with pytest.raises(TypeError, match=r"block 0 must be EncoderBlock"):
            blocks[0] = invalid_block
        with pytest.raises(TypeError, match=r"block 1 must be EncoderBlock"):
            blocks.insert(1, invalid_block)
        with pytest.raises(TypeError, match=r"block 2 must be EncoderBlock"):
            blocks.extend([invalid_block])

    def test_slice_preserves_encoder_block_contract(self):
        model = _make_tiny_model()
        blocks = model.encoder.blocks[:1]

        assert blocks.module_type is EncoderBlock
        with pytest.raises(TypeError, match=r"block 1 must be EncoderBlock"):
            blocks.append(t.nn.Linear(TINY_N_EMBD, TINY_N_EMBD))


# ---------------------------------------------------------------------------
# TestNumScalingParams
# ---------------------------------------------------------------------------


class TestNumScalingParams:
    def test_returns_dict_with_expected_keys(self):
        """num_scaling_params() returns a dict with required keys and positive values."""
        model = _make_encdec(device="cpu")
        result = model.num_scaling_params()

        expected_keys = {
            "wte",
            "value_embeds",
            "lm_head",
            "transformer_matrices",
            "scalars",
            "encoder",
            "cross_attn",
            "feature_pred_head",
            "total",
        }
        assert set(result.keys()) == expected_keys, (
            f"Key mismatch.\nExpected: {expected_keys}\nGot:      {set(result.keys())}"
        )
        # Every value must be a non-negative integer
        for key, val in result.items():
            assert isinstance(val, int), f"'{key}' value should be int, got {type(val)}"
            assert val >= 0, f"'{key}' value should be non-negative, got {val}"

        # The main components must be strictly positive
        for key in ("wte", "lm_head", "transformer_matrices", "encoder", "cross_attn", "total"):
            assert result[key] > 0, f"'{key}' should be > 0, got {result[key]}"

    def test_total_matches_actual_params(self):
        """num_scaling_params()['total'] equals the sum of all actual parameters."""
        model = _make_encdec(device="cpu")
        result = model.num_scaling_params()

        actual_total = sum(p.numel() for p in model.parameters())
        assert result["total"] > 0, "Total param count should be positive"
        assert result["total"] <= actual_total, (
            f"Reported total {result['total']} exceeds actual total {actual_total}"
        )
        # The method itself asserts internal consistency; verify the value matches
        assert result["total"] == actual_total, (
            f"num_scaling_params total {result['total']} != actual {actual_total}"
        )
