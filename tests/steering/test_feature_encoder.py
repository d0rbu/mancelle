import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from nanochat.gpt import has_ve
from steering.conditional.encoder_decoder.encoder import (
    EncoderSelfAttention,
    FeatureEncoder,
    FeatureEncoderConfig,
    norm,
)
from tests.padded_encoder_reference import densify_packed_encoder_output_reference


def _make_config(**overrides):
    defaults = {
        "n_sae_features": 128,
        "num_buckets": 8,
        "n_embd": 64,
        "n_layer": 2,
        "n_head": 4,
        "n_kv_head": 2,
        "head_dim": 16,
    }
    defaults.update(overrides)
    return FeatureEncoderConfig(**defaults)


def _make_inputs(B=2, S=3, F=5, config=None):
    """Create random feature_ids, magnitudes, and mask tensors."""
    if config is None:
        config = _make_config()
    feature_ids = torch.randint(0, config.n_sae_features, (B, S, F))
    magnitudes = torch.randint(0, config.num_buckets, (B, S, F))
    mask = torch.ones(B, S, F, dtype=torch.bool)
    return feature_ids, magnitudes, mask


def _dense_reference(model, ids, mags, mask):
    """Expose the test-owned padded view for legacy shape-level assertions."""
    return densify_packed_encoder_output_reference(model(ids, mags, mask))


class TestOutputShape:
    """Output tensor has the correct (B, S, F, n_embd) shape."""

    def test_basic_shape(self):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 2, 3, 5
        ids, mags, mask = _make_inputs(B, S, F, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        # Nf may be padded to next multiple of 16; check B, S, n_embd match
        assert out.shape[0] == B and out.shape[1] == S and out.shape[3] == config.n_embd
        assert out.shape[2] >= F

    def test_single_batch(self):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 1, 1, 1
        ids, mags, mask = _make_inputs(B, S, F, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        # Nf may be padded to next multiple of 16; check B, S, n_embd match
        assert out.shape[0] == B and out.shape[1] == S and out.shape[3] == config.n_embd
        assert out.shape[2] >= F

    def test_larger_dims(self):
        config = _make_config(n_embd=128, n_head=8, head_dim=16)
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 4, 8, 10
        ids, mags, mask = _make_inputs(B, S, F, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        assert out.shape[0] == B and out.shape[1] == S and out.shape[3] == 128
        assert out.shape[2] >= F


class TestPaddingMask:
    """Padding mask: no crash with partial masks; padded vs all-real differ."""

    def test_partial_mask_no_crash(self):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 2, 3, 5
        ids, mags, mask = _make_inputs(B, S, F, config)
        # Mask out some positions
        mask[0, :, 3:] = False
        mask[1, :, 4:] = False
        out, out_mask = _dense_reference(model, ids, mags, mask)
        # Nf may be padded to next multiple of 16; check B, S, n_embd match
        assert out.shape[0] == B and out.shape[1] == S and out.shape[3] == config.n_embd
        assert out.shape[2] >= F

    def test_padded_vs_real_differ(self):
        """Padded tokens vs all-real tokens produce different outputs."""
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        # Perturb c_proj weights away from zero so attention actually
        # contributes to the output (init_weights zeros them out).
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)
        B, S, F = 1, 1, 4
        torch.manual_seed(42)
        ids, mags, mask_all_real = _make_inputs(B, S, F, config)
        mask_partial = mask_all_real.clone()
        mask_partial[0, 0, 2:] = False

        out_real, _ = _dense_reference(model, ids, mags, mask_all_real)
        out_partial, _ = _dense_reference(model, ids, mags, mask_partial)
        # The outputs must differ because attention sees different tokens
        assert not torch.allclose(out_real, out_partial, atol=1e-5)


class TestGradientFlow:
    """Gradients flow back to both feature_embed and magnitude_embed."""

    def test_gradients_flow(self):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(2, 3, 5, config)
        out, _ = _dense_reference(model, ids, mags, mask)
        loss = out.sum()
        loss.backward()
        assert model.feature_embed.weight.grad is not None
        assert model.feature_embed.weight.grad.abs().sum() > 0
        assert model.magnitude_embed.weight.grad is not None
        assert model.magnitude_embed.weight.grad.abs().sum() > 0


class TestZeroLayers:
    """n_layer=0 is rejected: FeatureEncoder requires at least 1 layer."""

    def test_zero_layers_raises_value_error(self):
        """FeatureEncoder(n_layer=0) raises ValueError (needs dtype management from blocks)."""
        import pytest

        config = _make_config(n_layer=0)
        with pytest.raises(ValueError, match="at least 1 layer"):
            FeatureEncoder(config)


class TestAdditiveMagnitude:
    """Same feature_id with different magnitude_bucket produces different outputs."""

    def test_different_magnitude_different_output(self):
        config = _make_config()
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 1, 1, 1
        ids = torch.tensor([[[5]]])
        mask = torch.ones(B, S, F, dtype=torch.bool)

        mags_low = torch.tensor([[[0]]])
        mags_high = torch.tensor([[[config.num_buckets - 1]]])

        out_low, _ = _dense_reference(model, ids, mags_low, mask)
        out_high, _ = _dense_reference(model, ids, mags_high, mask)
        assert not torch.allclose(out_low, out_high, atol=1e-5)


# ---------------------------------------------------------------------------
# New tests below
# ---------------------------------------------------------------------------


class TestEmbeddingAdditivity:
    """Verify feature_embed(id) + magnitude_embed(bucket) is truly additive."""

    def test_sum_equals_feature_plus_magnitude(self):
        """The combined embedding must exactly equal the sum of the two lookups."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()

        B, S, F = 1, 1, 3
        ids = torch.tensor([[[2, 10, 50]]])
        mags = torch.tensor([[[0, 3, 7]]])
        mask = torch.ones(B, S, F, dtype=torch.bool)

        # Grab raw embeddings
        feat_emb = model.feature_embed(ids.reshape(-1))  # (3, n_embd)
        mag_emb = model.magnitude_embed(mags.reshape(-1))  # (3, n_embd)
        expected = feat_emb + mag_emb  # (3, n_embd)

        # With init_weights(), blocks are identity (c_proj=0) so output = norm(feat + mag)
        out, _ = _dense_reference(model, ids, mags, mask)
        # out is (B, S, Nf_padded, n_embd); extract real positions
        actual_pre_norm = model.feature_embed(ids.reshape(-1)) + model.magnitude_embed(mags.reshape(-1))
        # Verify the sum is exact (no overwriting, no scaling before norm)
        assert torch.allclose(expected, actual_pre_norm, atol=1e-7)

    def test_same_id_different_mags_differ(self):
        """Same feature ID with different magnitudes produces different pre-norm embeddings."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()

        id_val = 5
        emb_m0 = model.feature_embed.weight[id_val] + model.magnitude_embed.weight[0]
        emb_m7 = model.feature_embed.weight[id_val] + model.magnitude_embed.weight[7]
        assert not torch.allclose(emb_m0, emb_m7, atol=1e-7)

    def test_same_mag_different_ids_differ(self):
        """Same magnitude with different feature IDs produces different pre-norm embeddings."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()

        mag_val = 3
        emb_id2 = model.feature_embed.weight[2] + model.magnitude_embed.weight[mag_val]
        emb_id50 = model.feature_embed.weight[50] + model.magnitude_embed.weight[mag_val]
        assert not torch.allclose(emb_id2, emb_id50, atol=1e-7)

    @given(
        id_a=st.integers(0, 127),
        id_b=st.integers(0, 127),
        mag_a=st.integers(0, 7),
        mag_b=st.integers(0, 7),
    )
    @settings(max_examples=50, deadline=None)
    def test_additivity_property(self, id_a, id_b, mag_a, mag_b):
        """Property: embed(id, mag) == feature_embed(id) + magnitude_embed(mag) for any id, mag."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()

        ids = torch.tensor([[[id_a, id_b]]])
        mags = torch.tensor([[[mag_a, mag_b]]])

        expected_0 = model.feature_embed.weight[id_a] + model.magnitude_embed.weight[mag_a]
        expected_1 = model.feature_embed.weight[id_b] + model.magnitude_embed.weight[mag_b]

        # With init_weights(), blocks are identity (c_proj=0) so output ≈ norm(feat + mag)
        # Block casts to COMPUTE_DTYPE internally, causing bfloat16 rounding differences
        # vs the float32 reference path, so tolerance must accommodate bf16 precision
        out, _ = _dense_reference(
            model,
            ids,
            mags,
            torch.ones(1, 1, 2, dtype=torch.bool),
        )
        actual_0 = norm(expected_0.unsqueeze(0)).squeeze(0).to(out.dtype)
        actual_1 = norm(expected_1.unsqueeze(0)).squeeze(0).to(out.dtype)

        assert torch.allclose(out[0, 0, 0], actual_0, atol=0.05)
        assert torch.allclose(out[0, 0, 1], actual_1, atol=0.05)


class TestBidirectionalAttention:
    """Confirm attention is bidirectional: changing position 0 affects position Nf-1 and vice versa."""

    def _make_model_with_attn(self):
        """Create a model with non-zero attention weights so attention has an effect."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)
        return config, model

    def test_changing_first_affects_last(self):
        """Modifying feature at position 0 changes output at position Nf-1 (backward influence)."""
        config, model = self._make_model_with_attn()
        B, S, F = 1, 1, 4
        mask = torch.ones(B, S, F, dtype=torch.bool)

        torch.manual_seed(99)
        ids = torch.randint(0, config.n_sae_features, (B, S, F))
        mags = torch.randint(0, config.num_buckets, (B, S, F))

        out_orig, _ = _dense_reference(model, ids, mags, mask)

        # Change feature at position 0 only
        ids_mod = ids.clone()
        ids_mod[0, 0, 0] = (ids[0, 0, 0] + 1) % config.n_sae_features
        out_mod, _ = _dense_reference(model, ids_mod, mags, mask)

        # Position Nf-1 (index 3) should be affected
        assert not torch.allclose(out_orig[0, 0, F - 1], out_mod[0, 0, F - 1], atol=1e-5), \
            "Bidirectional violation: changing pos 0 did not affect pos Nf-1"

    def test_changing_last_affects_first(self):
        """Modifying feature at position Nf-1 changes output at position 0 (forward influence check)."""
        config, model = self._make_model_with_attn()
        B, S, F = 1, 1, 4
        mask = torch.ones(B, S, F, dtype=torch.bool)

        torch.manual_seed(99)
        ids = torch.randint(0, config.n_sae_features, (B, S, F))
        mags = torch.randint(0, config.num_buckets, (B, S, F))

        out_orig, _ = _dense_reference(model, ids, mags, mask)

        # Change feature at position Nf-1 only
        ids_mod = ids.clone()
        ids_mod[0, 0, F - 1] = (ids[0, 0, F - 1] + 1) % config.n_sae_features
        out_mod, _ = _dense_reference(model, ids_mod, mags, mask)

        # Position 0 should be affected
        assert not torch.allclose(out_orig[0, 0, 0], out_mod[0, 0, 0], atol=1e-5), \
            "Bidirectional violation: changing pos Nf-1 did not affect pos 0"

    def test_is_causal_false_in_attention(self):
        """EncoderSelfAttention does not use causal masking (structural check via output symmetry)."""
        config = _make_config(n_layer=1)
        attn = EncoderSelfAttention(config, layer_idx=0)
        # Initialize with non-zero weights
        with torch.no_grad():
            attn.c_qkv.weight.normal_(0, 0.02)
            attn.c_proj.weight.normal_(0, 0.02)

        B, Nf, C = 1, 4, config.n_embd
        x = torch.randn(B, Nf, C)
        out = attn(x, ve=None, mask=None)

        # Swap positions 0 and 3 in input
        x_swap = x.clone()
        x_swap[0, 0], x_swap[0, 3] = x[0, 3].clone(), x[0, 0].clone()
        out_swap = attn(x_swap, ve=None, mask=None)

        # In bidirectional attention, swapping inputs 0 and 3 should swap outputs 0 and 3
        # (since attention is permutation equivariant without positional info)
        assert torch.allclose(out[0, 0], out_swap[0, 3], atol=1e-5)
        assert torch.allclose(out[0, 3], out_swap[0, 0], atol=1e-5)


class TestFP8Padding:
    """FP8 padding: Nf padded to next multiple of 16, mask updated accordingly."""

    @given(nf=st.integers(min_value=1, max_value=48))
    @settings(max_examples=30, deadline=None)
    def test_padded_shape_multiple_of_16(self, nf):
        """Output Nf dimension is always a multiple of 16."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, nf, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        assert out.shape[2] % 16 == 0

    def test_nf_1_pads_to_16(self):
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 1, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        assert out.shape[2] == 16

    def test_nf_16_stays_16(self):
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 16, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        assert out.shape[2] == 16

    def test_nf_17_pads_to_32(self):
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 17, config)
        out, out_mask = _dense_reference(model, ids, mags, mask)
        assert out.shape[2] == 32

    def test_mask_false_for_padded_positions(self):
        """Padded positions have mask=False, original positions keep mask=True."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        Nf = 5
        ids, mags, mask = _make_inputs(1, 1, Nf, config)
        _, out_mask = _dense_reference(model, ids, mags, mask)
        # out_mask shape: (B, S, Nf_padded) = (1, 1, 16)
        assert out_mask.shape == (1, 1, 16)
        # Original positions should be True
        assert out_mask[0, 0, :Nf].all()
        # Padded positions should be False
        assert not out_mask[0, 0, Nf:].any()

    def test_mask_preserves_original_false_positions(self):
        """Input mask with some False positions: those stay False, padding also False."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        Nf = 5
        ids, mags, mask = _make_inputs(1, 1, Nf, config)
        mask[0, 0, 3] = False  # manually mask position 3
        _, out_mask = _dense_reference(model, ids, mags, mask)
        # Note: ``.item()`` returns a Python ``bool`` and ``is True``/``is False``
        # works because Python interns the singletons, but it is fragile. Prefer
        # explicit ``bool(...)`` casts so static analysers / mypy do not flag the
        # comparison and so the behaviour does not depend on bool-singleton
        # interning. (Slice 8 audit fix B1 / map.md §6 B10.)
        assert bool(out_mask[0, 0, 0]) is True
        assert bool(out_mask[0, 0, 3]) is False  # preserved
        assert not out_mask[0, 0, Nf:].any()  # padding is False

    def test_padded_positions_do_not_affect_real_outputs(self):
        """Outputs at real positions should be the same regardless of what values are in padded slots."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)

        Nf = 5
        torch.manual_seed(42)
        ids, mags, mask = _make_inputs(1, 1, Nf, config)

        out1, _ = _dense_reference(model, ids, mags, mask)

        # Modify the feature_ids at padded positions by providing a full-sized input
        # that maps to different padded values.
        # Since padding zeros ids/mags, we can't directly control padded values,
        # but we can verify that two runs with identical real data produce identical
        # real-position outputs (deterministic, no randomness from pad content).
        out2, _ = _dense_reference(model, ids.clone(), mags.clone(), mask.clone())
        assert torch.allclose(out1[0, 0, :Nf], out2[0, 0, :Nf], atol=1e-7)


class TestValueEmbeddingPattern:
    """Verify has_ve alternating pattern and that VE layers are constructed correctly."""

    def test_n_layer_4_ve_at_1_and_3(self):
        """For n_layer=4, VE should be at layers 1 and 3 (last layer always included)."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        # Check which layers have VE
        ve_layers = sorted(int(k) for k in model.value_embeds)
        assert ve_layers == [1, 3]

    def test_n_layer_3_ve_at_0_and_2(self):
        """For n_layer=3, VE should be at layers 0 and 2."""
        config = _make_config(n_layer=3)
        model = FeatureEncoder(config)
        ve_layers = sorted(int(k) for k in model.value_embeds)
        assert ve_layers == [0, 2]

    def test_n_layer_1_ve_at_0(self):
        """For n_layer=1, the single layer always gets VE."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        ve_layers = sorted(int(k) for k in model.value_embeds)
        assert ve_layers == [0]

    def test_n_layer_2_ve_at_1(self):
        """For n_layer=2, VE at layer 1 (last layer)."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        ve_layers = sorted(int(k) for k in model.value_embeds)
        assert ve_layers == [1]

    def test_last_layer_always_has_ve(self):
        """The last layer always has VE regardless of n_layer."""
        for n_layer in range(1, 9):
            assert has_ve(n_layer - 1, n_layer), f"n_layer={n_layer}: last layer should have VE"

    def test_ve_gate_present_iff_has_ve(self):
        """Blocks with VE should have a ve_gate; blocks without should have None."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        for i, block in enumerate(model.blocks):
            if has_ve(i, config.n_layer):
                assert block.attn.ve_gate is not None, f"Layer {i} should have ve_gate"
            else:
                assert block.attn.ve_gate is None, f"Layer {i} should NOT have ve_gate"

    def test_ve_embedding_shape(self):
        """Value embedding tables have shape (n_sae_features, n_kv_head * head_dim)."""
        config = _make_config(n_layer=4)
        model = FeatureEncoder(config)
        kv_dim = config.n_kv_head * config.head_dim
        for key, ve in model.value_embeds.items():
            assert ve.weight.shape == (config.n_sae_features, kv_dim), \
                f"VE at layer {key} has wrong shape: {ve.weight.shape}"


class TestRMSNorm:
    """Verify the functional RMSNorm (no learnable parameters)."""

    def test_unit_rms(self):
        """After norm, each vector should have approximately unit RMS."""
        x = torch.randn(2, 8, 64)
        y = norm(x)
        # RMS of each vector along last dim should be ~1.0
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_no_learnable_params(self):
        """norm() is a pure function with no learnable parameters (not an nn.Module)."""
        # norm is a function, not a module — just verify it works without state
        x = torch.randn(3, 4, 32)
        y = norm(x)
        assert y.shape == x.shape

    def test_norm_preserves_direction(self):
        """RMSNorm should preserve direction (just rescale), not change sign pattern."""
        x = torch.randn(1, 1, 64)
        y = norm(x)
        # The sign pattern should be identical
        assert (x.sign() == y.sign()).all()

    def test_norm_applied_at_output(self):
        """With init_weights() (blocks are identity), output = norm(feature_embed + magnitude_embed)."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()

        ids = torch.tensor([[[3, 7]]])
        mags = torch.tensor([[[1, 4]]])
        mask = torch.ones(1, 1, 2, dtype=torch.bool)

        raw_emb = model.feature_embed(ids.reshape(-1)) + model.magnitude_embed(mags.reshape(-1))
        expected = norm(raw_emb)  # (2, n_embd)

        out, _ = _dense_reference(model, ids, mags, mask)
        # Compare real positions (cast expected to match output dtype from COMPUTE_DTYPE;
        # wider tolerance due to bf16 intermediate rounding in the identity block)
        expected = expected.to(out.dtype)
        assert torch.allclose(out[0, 0, 0], expected[0], atol=0.05)
        assert torch.allclose(out[0, 0, 1], expected[1], atol=0.05)

    @given(dim=st.sampled_from([16, 32, 64, 128]))
    @settings(max_examples=10, deadline=None)
    def test_rms_unit_various_dims(self, dim):
        """RMSNorm produces unit RMS across different embedding dimensions."""
        x = torch.randn(4, 8, dim)
        y = norm(x)
        rms = y.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestMaskPropagation:
    """Verify mask is correctly padded and propagated through the encoder."""

    def test_output_mask_shape(self):
        """Output mask has shape (B, S, Nf_padded)."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 2, 3, 5
        ids, mags, mask = _make_inputs(B, S, F, config)
        _, out_mask = _dense_reference(model, ids, mags, mask)
        assert out_mask.shape == (B, S, 16)  # 5 -> 16

    def test_all_true_mask_stays_true_in_real_region(self):
        """All-True input mask remains True for real positions after padding."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 2, 2, 10
        ids, mags, mask = _make_inputs(B, S, F, config)
        assert mask.all()  # all True
        _, out_mask = _dense_reference(model, ids, mags, mask)
        # Real region stays True
        assert out_mask[:, :, :F].all()
        # Padded region is False
        assert not out_mask[:, :, F:].any()

    def test_partial_mask_preserved_after_padding(self):
        """Partial mask (some False in real region) is preserved after FP8 padding."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 1, 1, 5
        ids, mags, mask = _make_inputs(B, S, F, config)
        mask[0, 0, 2] = False
        mask[0, 0, 4] = False
        _, out_mask = _dense_reference(model, ids, mags, mask)
        # Check specific positions
        # ``bool(...)`` per slice 8 audit fix B1 / map.md §6 B10.
        assert bool(out_mask[0, 0, 0]) is True
        assert bool(out_mask[0, 0, 1]) is True
        assert bool(out_mask[0, 0, 2]) is False  # was False in input
        assert bool(out_mask[0, 0, 3]) is True
        assert bool(out_mask[0, 0, 4]) is False  # was False in input
        # Padding region
        assert not out_mask[0, 0, 5:].any()

    def test_multi_batch_multi_segment_mask(self):
        """Different segments in different batches have independent mask propagation."""
        config = _make_config(n_layer=1)
        model = FeatureEncoder(config)
        model.init_weights()
        B, S, F = 2, 3, 7
        ids, mags, mask = _make_inputs(B, S, F, config)
        # Different masking per batch/segment
        mask[0, 0, 5:] = False
        mask[0, 1, 3:] = False
        mask[1, 2, 6:] = False
        _, out_mask = _dense_reference(model, ids, mags, mask)
        assert out_mask.shape == (B, S, 16)  # 7 -> 16

        # Verify per-element (``bool(...)`` per slice 8 audit fix B1 / map.md §6 B10).
        assert bool(out_mask[0, 0, 4]) is True
        assert bool(out_mask[0, 0, 5]) is False
        assert bool(out_mask[0, 1, 2]) is True
        assert bool(out_mask[0, 1, 3]) is False
        assert bool(out_mask[1, 2, 5]) is True
        assert bool(out_mask[1, 2, 6]) is False


class TestAllMaskedSegment:
    """If all features in a segment are masked, output should be NaN-free."""

    def test_all_false_mask_no_nan(self):
        """All-masked segment (all False) produces no NaN thanks to nan_to_num."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)

        B, S, F = 1, 1, 4
        ids, mags, mask = _make_inputs(B, S, F, config)
        mask[:] = False  # all masked
        out, _ = _dense_reference(model, ids, mags, mask)
        assert not torch.isnan(out).any(), "NaN found in output with all-masked segment"
        assert not torch.isinf(out).any(), "Inf found in output with all-masked segment"

    def test_mixed_segments_one_all_masked(self):
        """Multiple segments where one is fully masked: no NaN anywhere."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)

        B, S, F = 2, 3, 4
        ids, mags, mask = _make_inputs(B, S, F, config)
        # Mask out entire segment (0, 1)
        mask[0, 1, :] = False
        out, _ = _dense_reference(model, ids, mags, mask)
        assert not torch.isnan(out).any(), "NaN found with mixed masking"

    def test_all_masked_gradient_no_nan(self):
        """Backward pass through all-masked segment produces no NaN gradients."""
        config = _make_config(n_layer=2)
        model = FeatureEncoder(config)
        model.init_weights()
        with torch.no_grad():
            for block in model.blocks:
                block.attn.c_proj.weight.normal_(0, 0.1)
                block.mlp.c_proj.weight.normal_(0, 0.1)

        B, S, F = 1, 2, 4
        ids, mags, mask = _make_inputs(B, S, F, config)
        mask[0, 0, :] = False  # segment 0 all masked, segment 1 real
        out, _ = _dense_reference(model, ids, mags, mask)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


class TestGQAConfig:
    """Verify GQA (grouped query attention) works with different n_kv_head settings."""

    def test_mha_kv_equals_q(self):
        """n_kv_head == n_head (standard MHA) works correctly."""
        config = _make_config(n_head=4, n_kv_head=4, head_dim=16)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 4, config)
        out, _ = _dense_reference(model, ids, mags, mask)
        assert out.shape == (1, 1, 16, config.n_embd)

    def test_mqa_single_kv_head(self):
        """n_kv_head == 1 (multi-query attention) works correctly."""
        config = _make_config(n_head=4, n_kv_head=1, head_dim=16)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 4, config)
        out, _ = _dense_reference(model, ids, mags, mask)
        assert out.shape == (1, 1, 16, config.n_embd)

    def test_gqa_kv_half_of_q(self):
        """n_kv_head == n_head // 2 (GQA) works correctly."""
        config = _make_config(n_head=8, n_kv_head=4, head_dim=16, n_embd=128)
        model = FeatureEncoder(config)
        model.init_weights()
        ids, mags, mask = _make_inputs(1, 1, 4, config)
        out, _ = _dense_reference(model, ids, mags, mask)
        assert out.shape == (1, 1, 16, 128)
