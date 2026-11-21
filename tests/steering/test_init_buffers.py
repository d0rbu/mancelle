"""Tests for init_buffers() — targeted rotary embedding initialization for checkpoint loading."""

from __future__ import annotations

import torch

from nanochat.gpt import GPT
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_LAYER,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_VOCAB,
    _make_encdec,
    _make_gpt,
    _make_gpt_config,
)


class TestGPTInitBuffers:
    """Tests for GPT.init_buffers()."""

    def test_roundtrip_bit_exact(self):
        """Create GPT, init_weights, forward, save, reload with init_buffers, forward — bit-exact."""
        original = _make_gpt()
        idx = torch.randint(0, TINY_VOCAB, (1, 16))
        targets = torch.randint(0, TINY_VOCAB, (1, 16))
        with torch.no_grad():
            original_loss = original(idx, targets)

        # Save state dict (persistent params only — cos/sin are persistent=False)
        sd = original.state_dict()

        # Rebuild on meta device, use init_buffers instead of init_weights
        config = _make_gpt_config()
        with torch.device("meta"):
            loaded = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        loaded.to_empty(device="cpu")
        loaded.init_buffers()
        loaded.load_state_dict(sd, strict=True, assign=True)
        loaded.eval()

        with torch.no_grad():
            loaded_loss = loaded(idx, targets)
        assert torch.equal(original_loss, loaded_loss), (
            f"Losses differ: {original_loss.item()} vs {loaded_loss.item()}"
        )

    def test_rotary_embeddings_match(self):
        """init_buffers() produces identical cos/sin as init_weights()."""
        ref = _make_gpt()  # uses init_weights

        config = _make_gpt_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        model.to_empty(device="cpu")
        model.init_buffers()

        assert torch.equal(ref.cos, model.cos), "cos mismatch"
        assert torch.equal(ref.sin, model.sin), "sin mismatch"
        assert model.cos.dtype == torch.bfloat16
        assert model.sin.dtype == torch.bfloat16

    def test_does_not_modify_parameters(self):
        """init_buffers() should only set cos/sin, not touch any parameters."""
        config = _make_gpt_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        model.to_empty(device="cpu")

        # Snapshot all parameter data pointers before init_buffers
        param_data_before = {
            name: p.data_ptr() for name, p in model.named_parameters()
        }

        model.init_buffers()

        # Parameters should be untouched (same data pointers — to_empty gives
        # uninitialized CPU storage, not zeros, so we only check data pointers)
        for name, p in model.named_parameters():
            assert p.data_ptr() == param_data_before[name], (
                f"Parameter {name} was modified by init_buffers()"
            )

        # cos/sin should have real values and correct shape
        assert model.cos.shape[1] == config.sequence_len * 10
        assert not torch.all(model.cos == 0), "cos should have non-zero values"
        assert not torch.all(model.sin == 0), "sin should have non-zero values"

    def test_init_buffers_after_state_load_is_idempotent(self):
        """init_buffers() restores the canonical rotary embeddings after load.

        Some torch/device combinations now materialize non-persistent rotary
        buffers correctly during ``to_empty()``, while older ones left them
        uninitialized. The portable contract is that calling ``init_buffers()``
        after state load is safe and yields the canonical values.
        """
        original = _make_gpt()
        sd = original.state_dict()

        config = _make_gpt_config()
        with torch.device("meta"):
            loaded = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        loaded.to_empty(device="cpu")
        loaded.load_state_dict(sd, strict=True, assign=True)

        assert loaded.cos.shape == original.cos.shape
        assert loaded.sin.shape == original.sin.shape

        # Verify init_buffers() fixes this and produces the correct values
        loaded.init_buffers()
        assert torch.equal(loaded.cos, original.cos), (
            "cos should match correct values after init_buffers()"
        )
        assert torch.equal(loaded.sin, original.sin), (
            "sin should match correct values after init_buffers()"
        )


class TestEncoderDecoderInitBuffers:
    """Tests for EncoderDecoderGPT.init_buffers()."""

    def test_roundtrip_bit_exact(self):
        """Create EncoderDecoderGPT, init_weights, forward, save, reload with init_buffers — bit-exact."""
        original = _make_encdec()
        B, T, n_seg, Nf = 1, 16, 2, 8
        idx = torch.randint(0, TINY_VOCAB, (B, T))
        targets = torch.randint(0, TINY_VOCAB, (B, T))
        enc_feature_ids = torch.randint(0, TINY_N_SAE_FEATURES, (B, n_seg, Nf))
        enc_magnitudes = torch.randint(0, TINY_NUM_BUCKETS, (B, n_seg, Nf))
        enc_mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.arange(T).unsqueeze(0) // (T // n_seg)

        with torch.no_grad():
            original_loss = original(
                idx, targets,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
            )

        sd = original.state_dict()

        # Rebuild using init_buffers
        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with torch.device("meta"):
            loaded = EncoderDecoderGPT(
                dec_config, enc_config, xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
                pad_vocab_size_to=TINY_VOCAB,
            )
        loaded.to_empty(device="cpu")
        loaded.init_buffers()
        loaded.load_state_dict(sd, strict=True, assign=True)
        loaded.eval()

        with torch.no_grad():
            loaded_loss = loaded(
                idx, targets,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
            )

        assert torch.equal(original_loss, loaded_loss), (
            f"Losses differ: {original_loss.item()} vs {loaded_loss.item()}"
        )

    def test_rotary_embeddings_match(self):
        """init_buffers() produces identical cos/sin as init_weights()."""
        ref = _make_encdec()

        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with torch.device("meta"):
            model = EncoderDecoderGPT(
                dec_config, enc_config, xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
                pad_vocab_size_to=TINY_VOCAB,
            )
        model.to_empty(device="cpu")
        model.init_buffers()

        assert torch.equal(ref.cos, model.cos), "cos mismatch"
        assert torch.equal(ref.sin, model.sin), "sin mismatch"
        assert model.cos.dtype == torch.bfloat16
        assert model.sin.dtype == torch.bfloat16

    def test_does_not_modify_parameters(self):
        """init_buffers() should only set cos/sin, not touch any parameters."""
        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=TINY_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        )
        xattn_config = CrossAttentionConfig(frequency=1)
        with torch.device("meta"):
            model = EncoderDecoderGPT(
                dec_config, enc_config, xattn_config,
                n_sae_features=TINY_N_SAE_FEATURES,
                num_buckets=TINY_NUM_BUCKETS,
                pad_vocab_size_to=TINY_VOCAB,
            )
        model.to_empty(device="cpu")

        param_data_before = {
            name: p.data_ptr() for name, p in model.named_parameters()
        }
        model.init_buffers()

        for name, p in model.named_parameters():
            assert p.data_ptr() == param_data_before[name], (
                f"Parameter {name} was modified by init_buffers()"
            )

        # cos/sin should have real values
        assert not torch.all(model.cos == 0)
        assert not torch.all(model.sin == 0)


class TestInitWeightsUnchanged:
    """Verify init_weights() still works correctly for fresh model creation."""

    def test_gpt_init_weights_sets_cos_sin_and_params(self):
        """init_weights() should set both rotary embeddings and all parameters."""
        config = _make_gpt_config()
        with torch.device("meta"):
            model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
        model.to_empty(device="cpu")
        model.init_weights()

        # Rotary embeddings should be set
        assert model.cos.dtype == torch.bfloat16
        assert model.sin.dtype == torch.bfloat16
        assert not torch.all(model.cos == 0)

        # Parameters should be initialized (not all zero)
        has_nonzero = any(
            not torch.all(p == 0) for p in model.parameters()
        )
        assert has_nonzero, "init_weights() should initialize at least some params to non-zero"

    def test_existing_test_fixtures_still_work(self, gpt_model, encdec_model):
        """Fixtures from conftest.py use init_weights() — they should still produce valid models."""
        # GPT fixture
        idx = torch.randint(0, TINY_VOCAB, (1, 16))
        with torch.no_grad():
            logits = gpt_model(idx)
        assert logits.shape == (1, 16, TINY_VOCAB)

        # EncoderDecoderGPT fixture
        B, T, n_seg, Nf = 1, 16, 2, 8
        enc_feature_ids = torch.randint(0, TINY_N_SAE_FEATURES, (B, n_seg, Nf))
        enc_magnitudes = torch.randint(0, TINY_NUM_BUCKETS, (B, n_seg, Nf))
        enc_mask = torch.ones(B, n_seg, Nf, dtype=torch.bool)
        seg_map = torch.arange(T).unsqueeze(0) // (T // n_seg)

        with torch.no_grad():
            logits = encdec_model(
                idx, None,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
            )
        assert logits.shape == (1, 16, TINY_VOCAB)
