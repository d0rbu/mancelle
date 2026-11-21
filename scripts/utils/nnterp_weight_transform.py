#!/usr/bin/env python3
"""Experiment: Apply TransformerLens weight transformations to HuggingFace GPT-2.

TransformerLens applies fold_ln, center_writing_weights, center_unembed when loading GPT-2.
These change intermediate representations. The SAE was trained on these transformed activations.

This script:
1. Extracts TransformerLens processed state_dict
2. Converts it back to HuggingFace GPT-2 format
3. Loads into HF model (with LayerNorm set to identity since weights are folded)
4. Verifies activations match TransformerLens at layer 8
5. Benchmarks nnterp tracer.stop() with transformed weights

Usage:
    uv run python -m scripts.utils.nnterp_weight_transform
    uv run python -m scripts.utils.nnterp_weight_transform model_name=gpt2-medium
"""

import time

import hydra
import torch as t
from loguru import logger
from nnterp import StandardizedTransformer
from omegaconf import DictConfig
from transformer_lens import HookedTransformer
from transformers import GPT2LMHeadModel, GPT2Model


def tl_to_hf_state_dict(tl_sd: dict, d_model: int, n_layers: int) -> dict:
    """Convert TransformerLens processed state_dict to HuggingFace GPT-2 format.

    After TransformerLens applies fold_ln, center_writing_weights, etc.,
    the LayerNorm weights are absorbed into attention/MLP weights. We set
    HF LayerNorm to identity (weight=1, bias=0) and use the processed weights.
    """
    hf_sd = {}

    # Embeddings (same shape)
    hf_sd["wte.weight"] = tl_sd["embed.W_E"].clone()
    hf_sd["wpe.weight"] = tl_sd["pos_embed.W_pos"].clone()

    for layer_idx in range(n_layers):
        prefix_tl = f"blocks.{layer_idx}"
        prefix_hf = f"h.{layer_idx}"

        # LayerNorm: set to identity (weights are folded into attn/MLP)
        hf_sd[f"{prefix_hf}.ln_1.weight"] = t.ones(d_model)
        hf_sd[f"{prefix_hf}.ln_1.bias"] = t.zeros(d_model)
        hf_sd[f"{prefix_hf}.ln_2.weight"] = t.ones(d_model)
        hf_sd[f"{prefix_hf}.ln_2.bias"] = t.zeros(d_model)

        # Attention QKV: TL [n_heads, d_model, d_head] -> HF [d_model, 3*d_model]
        W_Q = tl_sd[f"{prefix_tl}.attn.W_Q"]  # [12, 768, 64]
        W_K = tl_sd[f"{prefix_tl}.attn.W_K"]
        W_V = tl_sd[f"{prefix_tl}.attn.W_V"]

        # HF Conv1D: x @ weight, weight is [d_model, n_heads*d_head]
        W_Q_hf = W_Q.permute(1, 0, 2).reshape(d_model, d_model)
        W_K_hf = W_K.permute(1, 0, 2).reshape(d_model, d_model)
        W_V_hf = W_V.permute(1, 0, 2).reshape(d_model, d_model)
        hf_sd[f"{prefix_hf}.attn.c_attn.weight"] = t.cat(
            [W_Q_hf, W_K_hf, W_V_hf], dim=1
        )

        # Attention QKV biases
        b_Q = tl_sd[f"{prefix_tl}.attn.b_Q"].reshape(-1)  # [768]
        b_K = tl_sd[f"{prefix_tl}.attn.b_K"].reshape(-1)
        b_V = tl_sd[f"{prefix_tl}.attn.b_V"].reshape(-1)
        hf_sd[f"{prefix_hf}.attn.c_attn.bias"] = t.cat([b_Q, b_K, b_V])

        # Attention output: TL [n_heads, d_head, d_model] -> HF [d_model, d_model]
        W_O = tl_sd[f"{prefix_tl}.attn.W_O"]  # [12, 64, 768]
        hf_sd[f"{prefix_hf}.attn.c_proj.weight"] = W_O.reshape(d_model, d_model)
        hf_sd[f"{prefix_hf}.attn.c_proj.bias"] = tl_sd[f"{prefix_tl}.attn.b_O"].clone()

        # MLP: same shapes between TL and HF
        hf_sd[f"{prefix_hf}.mlp.c_fc.weight"] = tl_sd[f"{prefix_tl}.mlp.W_in"].clone()
        hf_sd[f"{prefix_hf}.mlp.c_fc.bias"] = tl_sd[f"{prefix_tl}.mlp.b_in"].clone()
        hf_sd[f"{prefix_hf}.mlp.c_proj.weight"] = tl_sd[
            f"{prefix_tl}.mlp.W_out"
        ].clone()
        hf_sd[f"{prefix_hf}.mlp.c_proj.bias"] = tl_sd[f"{prefix_tl}.mlp.b_out"].clone()

    # Final LayerNorm: identity (folded into unembed)
    hf_sd["ln_f.weight"] = t.ones(d_model)
    hf_sd["ln_f.bias"] = t.zeros(d_model)

    return hf_sd


def _cuda_sync() -> None:
    """Synchronize CUDA when benchmarking on a CUDA device; CPU is a no-op."""
    if t.cuda.is_available():
        t.cuda.synchronize()


@hydra.main(
    config_path="../../configs/profiling",
    config_name="weight_transform",
    version_base=None,
)
def main(cfg: DictConfig):
    device = "cuda" if t.cuda.is_available() else "cpu"

    # Load TransformerLens (with all transformations applied)
    logger.info(
        "Loading TransformerLens GPT-2 (with fold_ln, center_writing_weights)..."
    )
    tl_model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    d_model = tl_model.cfg.d_model
    n_layers = tl_model.cfg.n_layers
    vocab_size = tl_model.cfg.d_vocab
    tl_sd = tl_model.state_dict()

    # Convert to HF format
    logger.info("Converting TL state_dict to HF format...")
    hf_sd = tl_to_hf_state_dict(tl_sd, d_model, n_layers)

    # Load into HF model (GPT2Model for activation comparison)
    logger.info("Loading transformed weights into HF GPT2Model...")
    hf_base = GPT2Model.from_pretrained(cfg.model_name)
    # Add 'transformer.' prefix for keys in GPT2Model
    hf_base_sd = dict(hf_sd.items())
    missing, unexpected = hf_base.load_state_dict(hf_base_sd, strict=False)
    if missing:
        logger.info(f"  Missing keys: {missing[:5]}...")
    # Transformers' decorated ``to`` stub loses the normal ``nn.Module``
    # signature. Invoke the inherited in-place implementation explicitly.
    t.nn.Module.to(hf_base, device=device)
    hf_base.eval()

    # Also load into GPT2LMHeadModel (for nnterp which requires lm_head)
    hf_lm = GPT2LMHeadModel.from_pretrained(cfg.model_name)
    hf_lm_sd = {f"transformer.{k}": v for k, v in hf_sd.items()}
    # LM head is tied to wte.weight
    hf_lm_sd["lm_head.weight"] = hf_sd["wte.weight"].clone()
    missing, unexpected = hf_lm.load_state_dict(hf_lm_sd, strict=False)
    if missing:
        logger.info(f"  LM model missing keys: {missing[:5]}...")
    t.nn.Module.to(hf_lm, device=device)
    hf_lm.eval()
    hf_model = hf_base  # use base model for activation comparison

    # Test with same input
    tokens = t.randint(0, vocab_size, (4, 1024), device=device)

    # Get TL activations at layer 8
    logger.info("\nComparing activations at layer 8...")
    with t.no_grad():
        tl_acts = tl_model(tokens, stop_at_layer=8)  # [B, S, 768]

    # Get HF activations at layer 8
    with t.no_grad():
        hf_out = hf_model(tokens, output_hidden_states=True)
        hf_acts = hf_out.hidden_states[8]  # [B, S, 768]

    # Compare
    abs_diff = (tl_acts - hf_acts).abs()
    rel_diff = abs_diff / (tl_acts.abs() + 1e-8)
    cos_sim = t.nn.functional.cosine_similarity(
        tl_acts.flatten().unsqueeze(0),
        hf_acts.flatten().unsqueeze(0),
    )

    logger.info(f"  max_abs_diff:  {abs_diff.max().item():.2e}")
    logger.info(f"  mean_abs_diff: {abs_diff.mean().item():.2e}")
    logger.info(f"  max_rel_diff:  {rel_diff.max().item():.2e}")
    logger.info(f"  cosine_sim:    {cos_sim.item():.8f}")
    logger.info(f"  allclose(1e-4): {t.allclose(tl_acts, hf_acts, atol=1e-4)}")
    logger.info(f"  allclose(1e-3): {t.allclose(tl_acts, hf_acts, atol=1e-3)}")
    logger.info(f"  allclose(1e-2): {t.allclose(tl_acts, hf_acts, atol=1e-2)}")

    if cos_sim.item() < 0.99:
        logger.info("\n  WARNING: Activations don't match well. Aborting benchmark.")
        return

    # ── Now try with nnterp ──
    logger.info("\nLoading nnterp with transformed weights...")

    nn_model = StandardizedTransformer(hf_lm)
    logger.info(f"  nnterp model: {nn_model.num_layers} layers")

    # Verify nnterp activations match
    with t.no_grad():
        with nn_model.trace(tokens) as tracer:
            nn_acts = nn_model.layers_output[7].save()
            tracer.stop()

    nn_diff = (tl_acts - nn_acts).abs()
    nn_cos = t.nn.functional.cosine_similarity(
        tl_acts.flatten().unsqueeze(0),
        nn_acts.flatten().unsqueeze(0),
    )
    logger.info(f"  nnterp vs TL max_abs_diff: {nn_diff.max().item():.2e}")
    logger.info(f"  nnterp vs TL cosine_sim:   {nn_cos.item():.8f}")

    # ── Speed benchmark ──
    logger.info("\n" + "=" * 80)
    logger.info("SPEED BENCHMARK (batch_size=4, seq_len=1024, layer=8)")
    logger.info("=" * 80)

    # Warmup
    for _ in range(5):
        with t.no_grad():
            tl_model(tokens, stop_at_layer=8)
            with nn_model.trace(tokens) as tracer:
                nn_model.layers_output[7].save()
                tracer.stop()

    # TL stop_at_layer
    times = []
    for _ in range(20):
        _cuda_sync()
        s = time.perf_counter()
        with t.no_grad():
            tl_model(tokens, stop_at_layer=8)
        _cuda_sync()
        times.append(time.perf_counter() - s)
    tl_med = sorted(times)[10]

    # nnterp with transformed weights
    times = []
    for _ in range(20):
        _cuda_sync()
        s = time.perf_counter()
        with t.no_grad():
            with nn_model.trace(tokens) as tracer:
                nn_model.layers_output[7].save()
                tracer.stop()
        _cuda_sync()
        times.append(time.perf_counter() - s)
    nn_med = sorted(times)[10]

    # HF raw (full 12 layers)
    times = []
    for _ in range(20):
        _cuda_sync()
        s = time.perf_counter()
        with t.no_grad():
            hf_model(tokens, output_hidden_states=True)
        _cuda_sync()
        times.append(time.perf_counter() - s)
    hf_full_med = sorted(times)[10]

    total_tokens = 4 * 1024
    logger.info(
        f"\n  {'Method':<35} {'Median (ms)':>12} {'Speedup':>10} {'Tokens/s':>12}"
    )
    logger.info("  " + "-" * 75)
    logger.info(
        f"  {'TL stop_at_layer=8':<35} {tl_med * 1000:>10.1f}ms {'1.00x':>10} {total_tokens / (tl_med):>10,.0f}"
    )
    logger.info(
        f"  {'nnterp+TL_weights tracer.stop()':<35} {nn_med * 1000:>10.1f}ms {tl_med / nn_med:>9.2f}x {total_tokens / (nn_med):>10,.0f}"
    )
    logger.info(
        f"  {'HF raw (full 12 layers)':<35} {hf_full_med * 1000:>10.1f}ms {tl_med / hf_full_med:>9.2f}x {total_tokens / (hf_full_med):>10,.0f}"
    )


if __name__ == "__main__":
    main()
