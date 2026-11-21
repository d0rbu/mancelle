"""End-to-end integration tests for encoder-decoder LR scaling.

Verifies that all parameter groups (decoder, encoder, cross-attention) receive
correct learning rates and produce gradient updates proportional to their LRs
after a real forward/backward pass.
"""

import pytest
import torch as t
import torch._dynamo

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

# Increase dynamo cache to avoid recompile limit in Muon optimizer
# (fused QKV/KV creates many unique weight shapes across decoder/encoder/xattn)
torch._dynamo.config.cache_size_limit = 32

# ── Small model dimensions ───────────────────────────────────────────────
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
T = 128

# ── Default LR values ───────────────────────────────────────────────────
DEFAULT_UNEMBEDDING_LR = 0.004
DEFAULT_EMBEDDING_LR = 0.2
DEFAULT_MATRIX_LR = 0.02
DEFAULT_SCALAR_LR = 0.5
DEFAULT_ENCODER_EMBEDDING_LR = 0.2
DEFAULT_ENCODER_MATRIX_LR = 0.02
DEFAULT_XATTN_MATRIX_LR = 0.02

DMODEL_LR_SCALE = (DEC_N_EMBD / 768) ** -0.5


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend for all tests in this module.

    FlexAttention does not support backward on CPU. We patch the module-level
    override directly rather than using an env var, because the flash_attention
    module may already be imported by earlier tests in the suite.
    """
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


@pytest.fixture(scope="module")
def model():
    """Build a small EncoderDecoderGPT on CPU and initialise weights."""
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

    m = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    m.init_weights()

    # Perturb ALL c_proj weights away from zero so gradients flow through every
    # path. (init_weights zeros c_proj for residual-friendly init, which blocks
    # gradient flow through attention/MLP branches, value embeddings, and ve_gates)
    with t.no_grad():
        # Decoder blocks
        for block in m._decoder_blocks:
            block.attn.c_proj.weight.normal_(std=0.01)
            block.mlp.c_proj.weight.normal_(std=0.01)
        # Encoder blocks
        for block in m.encoder.blocks:
            block.attn.c_proj.weight.normal_(std=0.01)
            block.mlp.c_proj.weight.normal_(std=0.01)
        # Cross-attention layers
        for layer in m._cross_attention_layer_values():
            layer.c_proj.weight.normal_(std=0.01)

    return m


# ── Helpers ──────────────────────────────────────────────────────────────


def _named_param_ptr_map(model):
    """Map data_ptr -> parameter name."""
    return {p.data_ptr(): name for name, p in model.named_parameters()}


def _groups_for_prefix(optimizer, model, prefix):
    """Return optimizer groups that contain any parameter whose name starts with *prefix*."""
    name_map = _named_param_ptr_map(model)
    result = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            pname = name_map.get(p.data_ptr(), "")
            if pname.startswith(prefix):
                result.append(group)
                break
    return result


def _run_forward_backward(model):
    """Run a single forward+backward pass with synthetic data. Returns (loss, optimizer_ready_model)."""
    B = 2
    idx = t.randint(0, VOCAB_SIZE, (B, T))
    targets = t.randint(0, VOCAB_SIZE, (B, T))

    n_seg = 2
    F_max = 4
    enc_fids = t.randint(0, N_SAE_FEATURES, (B, n_seg, F_max))
    enc_mags = t.randint(0, NUM_BUCKETS, (B, n_seg, F_max))
    enc_mask = t.ones(B, n_seg, F_max, dtype=t.bool)
    seg_map = t.zeros(B, T, dtype=t.long)
    seg_map[:, T // 2 :] = 1  # second half maps to segment 1

    loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
    loss.backward()
    return loss


def _group_label(group, model):
    """Return a human-readable label for an optimizer group."""
    name_map = _named_param_ptr_map(model)
    if group["params"]:
        rep = name_map.get(group["params"][0].data_ptr(), "<unknown>")
    else:
        rep = "<empty>"
    kind = group.get("kind", "?")
    return f"{kind}:{rep}"


# ══════════════════════════════════════════════════════════════════════════
# Integration tests
# ══════════════════════════════════════════════════════════════════════════


class TestLRScalingIntegration:
    """End-to-end tests verifying LR scaling through forward/backward/step."""

    def test_all_groups_receive_gradients(self, model):
        """After a forward+backward pass, every parameter group has non-zero gradients."""
        model.zero_grad(set_to_none=True)

        batch_lr_scale = 2.0
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=0.0,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        _run_forward_backward(model)

        for gi, group in enumerate(opt.param_groups):
            label = _group_label(group, model)
            has_grad = False
            for p in group["params"]:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grad = True
                    break
            assert has_grad, (
                f"Parameter group {gi} ({label}) has no non-zero gradients "
                f"after forward/backward pass"
            )

        model.zero_grad(set_to_none=True)

    def test_initial_lr_includes_batch_scale(self, model):
        """After setup_optimizer with batch_lr_scale=2.0, initial_lr for encoder
        and cross-attention groups must reflect the scaled values."""
        batch_lr_scale = 2.0
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=0.0,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        # Check encoder embedding groups
        enc_embed_groups = _groups_for_prefix(opt, model, "encoder.feature_embed")
        enc_embed_groups += _groups_for_prefix(opt, model, "encoder.magnitude_embed")
        assert len(enc_embed_groups) > 0, "No encoder embedding groups found"
        expected_enc_embed_lr = DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
        for g in enc_embed_groups:
            assert abs(g["initial_lr"] - expected_enc_embed_lr) < 1e-10, (
                f"Encoder embedding initial_lr={g['initial_lr']}, "
                f"expected {expected_enc_embed_lr} "
                f"(base={DEFAULT_ENCODER_EMBEDDING_LR} * batch_scale={batch_lr_scale} * dmodel_scale={DMODEL_LR_SCALE})"
            )

        # Check encoder matrix (Muon) groups
        enc_matrix_groups = _groups_for_prefix(opt, model, "encoder.blocks")
        assert len(enc_matrix_groups) > 0, "No encoder matrix groups found"
        expected_enc_matrix_lr = DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale
        for g in enc_matrix_groups:
            assert abs(g["initial_lr"] - expected_enc_matrix_lr) < 1e-10, (
                f"Encoder matrix initial_lr={g['initial_lr']}, "
                f"expected {expected_enc_matrix_lr} "
                f"(base={DEFAULT_ENCODER_MATRIX_LR} * batch_scale={batch_lr_scale})"
            )

        # Check cross-attention matrix (Muon) groups
        xattn_groups = _groups_for_prefix(opt, model, "cross_attn_layers")
        assert len(xattn_groups) > 0, "No cross-attention groups found"
        expected_xattn_lr = DEFAULT_XATTN_MATRIX_LR * batch_lr_scale
        for g in xattn_groups:
            assert abs(g["initial_lr"] - expected_xattn_lr) < 1e-10, (
                f"Cross-attention initial_lr={g['initial_lr']}, "
                f"expected {expected_xattn_lr} "
                f"(base={DEFAULT_XATTN_MATRIX_LR} * batch_scale={batch_lr_scale})"
            )

        # Check decoder groups also include batch_lr_scale
        dec_lm_head_groups = _groups_for_prefix(opt, model, "lm_head")
        assert len(dec_lm_head_groups) > 0, "No lm_head groups found"
        expected_lm_head_lr = DEFAULT_UNEMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
        for g in dec_lm_head_groups:
            assert abs(g["initial_lr"] - expected_lm_head_lr) < 1e-10, (
                f"lm_head initial_lr={g['initial_lr']}, expected {expected_lm_head_lr}"
            )

    @pytest.mark.parametrize("batch_lr_scale", [1.0, 2.0, 0.5])
    def test_update_magnitude_proportional_to_lr(self, model, batch_lr_scale):
        """After one optimizer step, parameter update magnitudes should be
        ordered consistently with their learning rates.

        Groups with much higher LRs (e.g. embedding_lr vs unembedding_lr)
        should produce larger parameter updates.
        """
        model.zero_grad(set_to_none=True)

        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=0.0,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        _run_forward_backward(model)

        # Record params before step
        params_before = {}
        for gi, group in enumerate(opt.param_groups):
            for pi, p in enumerate(group["params"]):
                params_before[(gi, pi)] = p.data.clone()

        opt.step()

        # Compute mean absolute delta per group
        group_deltas = {}
        for gi, group in enumerate(opt.param_groups):
            deltas = []
            for pi, p in enumerate(group["params"]):
                delta = (p.data - params_before[(gi, pi)]).abs().mean().item()
                deltas.append(delta)
            group_deltas[gi] = sum(deltas) / len(deltas) if deltas else 0.0

        # Every group should have non-zero updates
        for gi, delta in group_deltas.items():
            label = _group_label(opt.param_groups[gi], model)
            assert delta > 0, (
                f"Group {gi} ({label}) had zero parameter update after opt.step()"
            )

        # Verify proportionality: lm_head (unembedding_lr=0.004) should have
        # smaller updates than decoder embeddings (embedding_lr=0.2).
        # Both are AdamW with dmodel_lr_scale, so the ratio is 0.004/0.2 = 0.02.
        lm_head_groups = _groups_for_prefix(opt, model, "lm_head")
        dec_embed_groups = _groups_for_prefix(opt, model, "transformer.wte")
        assert len(lm_head_groups) > 0 and len(dec_embed_groups) > 0

        lm_head_gi = next(
            gi
            for gi, g in enumerate(opt.param_groups)
            if any(
                _named_param_ptr_map(model).get(p.data_ptr(), "").startswith("lm_head")
                for p in g["params"]
            )
        )
        dec_embed_gi = next(
            gi
            for gi, g in enumerate(opt.param_groups)
            if any(
                _named_param_ptr_map(model)
                .get(p.data_ptr(), "")
                .startswith("transformer.wte")
                for p in g["params"]
            )
        )

        # unembedding_lr (0.004) << embedding_lr (0.2), so lm_head delta should
        # generally be smaller than wte delta. This is a loose check since AdamW
        # normalizes by second moment, but the 50x LR ratio should dominate.
        assert group_deltas[lm_head_gi] < group_deltas[dec_embed_gi], (
            f"Expected lm_head updates ({group_deltas[lm_head_gi]:.6e}) "
            f"< wte updates ({group_deltas[dec_embed_gi]:.6e}), "
            f"since unembedding_lr ({DEFAULT_UNEMBEDDING_LR}) << embedding_lr ({DEFAULT_EMBEDDING_LR})"
        )

        # Encoder matrix LR and decoder matrix LR and xattn matrix LR should
        # all be equal (all DEFAULT_*_MATRIX_LR * batch_lr_scale = 0.02 * scale)
        enc_matrix_gis = [
            gi
            for gi, g in enumerate(opt.param_groups)
            if any(
                _named_param_ptr_map(model)
                .get(p.data_ptr(), "")
                .startswith("encoder.blocks")
                for p in g["params"]
            )
        ]
        xattn_gis = [
            gi
            for gi, g in enumerate(opt.param_groups)
            if any(
                _named_param_ptr_map(model)
                .get(p.data_ptr(), "")
                .startswith("cross_attn_layers")
                for p in g["params"]
            )
        ]

        # All these groups have the same LR, so their deltas should be
        # within an order of magnitude of each other (Muon applies different
        # transformations per shape, so we allow generous tolerance)
        all_matrix_gis = enc_matrix_gis + xattn_gis
        if len(all_matrix_gis) >= 2:
            matrix_deltas = [group_deltas[gi] for gi in all_matrix_gis]
            max_delta = max(matrix_deltas)
            min_delta = min(matrix_deltas)
            # Same LR groups should be within 100x of each other
            # (generous tolerance for Muon's orthogonalization effects)
            assert max_delta / max(min_delta, 1e-30) < 100, (
                f"Encoder/xattn matrix groups have same LR but wildly different "
                f"update magnitudes: {matrix_deltas}"
            )

        model.zero_grad(set_to_none=True)

    def test_sft_lr_multiplier_scales_all_groups(self, model):
        """Verify that lr_multiplier applied in SFT scales encoder/xattn groups too.

        The SFT script applies lr_multiplier to all LRs passed to setup_optimizer.
        This test verifies that when we pass scaled LRs, all groups (including
        encoder and cross-attention) have initial_lr that includes the multiplier.
        """
        lr_multiplier = 0.5

        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * lr_multiplier,
            embedding_lr=DEFAULT_EMBEDDING_LR * lr_multiplier,
            scalar_lr=DEFAULT_SCALAR_LR * lr_multiplier,
            matrix_lr=DEFAULT_MATRIX_LR * lr_multiplier,
            weight_decay=0.0,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * lr_multiplier,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * lr_multiplier,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * lr_multiplier,
        )

        name_map = _named_param_ptr_map(model)

        # Check every group's initial_lr matches expectations
        for group in opt.param_groups:
            rep_name = name_map.get(group["params"][0].data_ptr(), "")
            initial_lr = group["initial_lr"]

            if rep_name.startswith("lm_head"):
                expected = DEFAULT_UNEMBEDDING_LR * lr_multiplier * DMODEL_LR_SCALE
                assert abs(initial_lr - expected) < 1e-10, (
                    f"lm_head: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name.startswith("transformer.wte"):
                expected = DEFAULT_EMBEDDING_LR * lr_multiplier * DMODEL_LR_SCALE
                assert abs(initial_lr - expected) < 1e-10, (
                    f"wte: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name.startswith("value_embeds"):
                expected = DEFAULT_EMBEDDING_LR * lr_multiplier * DMODEL_LR_SCALE
                assert abs(initial_lr - expected) < 1e-10, (
                    f"value_embeds: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name == "resid_lambdas":
                expected = DEFAULT_SCALAR_LR * lr_multiplier * 0.01
                assert abs(initial_lr - expected) < 1e-10, (
                    f"resid_lambdas: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name == "x0_lambdas":
                expected = DEFAULT_SCALAR_LR * lr_multiplier
                assert abs(initial_lr - expected) < 1e-10, (
                    f"x0_lambdas: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name.startswith(
                (
                    "encoder.feature_embed",
                    "encoder.magnitude_embed",
                    "encoder.value_embeds",
                )
            ):
                expected = DEFAULT_ENCODER_EMBEDDING_LR * lr_multiplier * DMODEL_LR_SCALE
                assert abs(initial_lr - expected) < 1e-10, (
                    f"{rep_name}: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name.startswith("encoder.blocks"):
                expected = DEFAULT_ENCODER_MATRIX_LR * lr_multiplier
                assert abs(initial_lr - expected) < 1e-10, (
                    f"{rep_name}: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name.startswith("cross_attn_layers"):
                expected = DEFAULT_XATTN_MATRIX_LR * lr_multiplier
                assert abs(initial_lr - expected) < 1e-10, (
                    f"{rep_name}: initial_lr={initial_lr}, expected={expected}"
                )
            elif rep_name.startswith("transformer.h"):
                expected = DEFAULT_MATRIX_LR * lr_multiplier
                assert abs(initial_lr - expected) < 1e-10, (
                    f"{rep_name}: initial_lr={initial_lr}, expected={expected}"
                )
            else:
                pytest.fail(
                    f"Unexpected parameter group with representative '{rep_name}'"
                )

    def test_lr_schedule_multiplier_propagates_via_initial_lr(self, model):
        """Simulate the training loop's LR schedule: group['lr'] = group['initial_lr'] * lrm.

        After applying a schedule multiplier to all groups, encoder/xattn groups
        must also receive the reduced LR (this was broken before the bug fix,
        because initial_lr for those groups did not include batch_lr_scale).
        """
        batch_lr_scale = 2.0
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=0.0,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        # Simulate a warmdown step: lr_multiplier = 0.3
        lrm = 0.3
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        # Verify encoder matrix groups got the schedule applied correctly
        enc_matrix_groups = _groups_for_prefix(opt, model, "encoder.blocks")
        expected_enc_lr = DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale * lrm
        for g in enc_matrix_groups:
            assert abs(g["lr"] - expected_enc_lr) < 1e-10, (
                f"Encoder matrix lr after schedule={g['lr']}, "
                f"expected={expected_enc_lr} "
                f"(base * batch_scale * lrm = {DEFAULT_ENCODER_MATRIX_LR} * {batch_lr_scale} * {lrm})"
            )

        # Verify xattn groups got the schedule applied correctly
        xattn_groups = _groups_for_prefix(opt, model, "cross_attn_layers")
        expected_xattn_lr = DEFAULT_XATTN_MATRIX_LR * batch_lr_scale * lrm
        for g in xattn_groups:
            assert abs(g["lr"] - expected_xattn_lr) < 1e-10, (
                f"Cross-attention lr after schedule={g['lr']}, "
                f"expected={expected_xattn_lr} "
                f"(base * batch_scale * lrm = {DEFAULT_XATTN_MATRIX_LR} * {batch_lr_scale} * {lrm})"
            )

        # Verify decoder matrix groups also got it right
        dec_matrix_groups = _groups_for_prefix(opt, model, "transformer.h")
        expected_dec_lr = DEFAULT_MATRIX_LR * batch_lr_scale * lrm
        for g in dec_matrix_groups:
            assert abs(g["lr"] - expected_dec_lr) < 1e-10, (
                f"Decoder matrix lr after schedule={g['lr']}, "
                f"expected={expected_dec_lr}"
            )
