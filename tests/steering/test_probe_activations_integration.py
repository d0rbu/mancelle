"""Integration tests for probe activation backends.

Verifies that nnterp and TransformerLens produce identical activations
at both hook_resid_pre and hook_resid_post hook points. Detects
off-by-one bugs in nnterp layer indexing.

Requires GPU -- skipped on CPU-only machines.
"""

from __future__ import annotations

import pytest
import torch as t
from transformers import GPT2LMHeadModel

from steering.eval.probe_activations import get_probe_activations, trim_hf_model
from steering.extract_features import (
    _get_activations_nnterp,
    _get_activations_tl,
    build_nnterp_model,
    load_sae_and_model,
)
from steering.sae_resolver import ActivationType

pytestmark = pytest.mark.skipif(
    not t.cuda.is_available(), reason="nnterp requires CUDA"
)


class TestLayerIndexing:
    """Verify nnterp and TL produce identical activations at the exact SAE hook point."""

    @pytest.mark.slow
    def test_nnterp_tl_activation_equivalence_resid_post(self):
        """layers_output index must match blocks.{L}.hook_resid_post for GPT-2.

        hook_resid_post of layer L captures the residual stream AFTER block L.
        nnterp layers_output[L] = output of HF layer L = same thing.
        A bug using layers_output[L-1] would capture layer L-1 output instead.
        """
        layer = 8
        hook_name = f"blocks.{layer}.hook_resid_post"
        device = "cuda"

        tl_model, _sae, _tok = load_sae_and_model(
            model_name="gpt2",
            layer=layer,
            device=device,
            sae_release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        nn_model = build_nnterp_model(tl_model, device=device)
        assert nn_model is not None, "nnterp build failed for GPT-2"

        tokens = t.randint(0, 50257, (2, 64), device=device)

        with t.no_grad():
            tl_acts = _get_activations_tl(
                tl_model, tokens, layer, hook_name=hook_name
            )
            nn_acts = _get_activations_nnterp(
                nn_model, tokens, layer, ActivationType.residual, hook_name=hook_name
            )

        assert tl_acts.shape == nn_acts.shape, (
            f"Shape mismatch: TL {tl_acts.shape} vs nnterp {nn_acts.shape}"
        )

        tl_cpu = tl_acts.float().cpu()
        nn_cpu = nn_acts.float().cpu()

        max_diff = (tl_cpu - nn_cpu).abs().max().item()
        cos_sim = t.nn.functional.cosine_similarity(
            tl_cpu.flatten().unsqueeze(0),
            nn_cpu.flatten().unsqueeze(0),
        ).item()

        assert t.allclose(tl_cpu, nn_cpu, atol=1e-3), (
            f"hook_resid_post activations diverge: max_diff={max_diff:.2e}, "
            f"cos_sim={cos_sim:.6f}. "
            f"Likely off-by-one: layers_output[{layer - 1}] != "
            f"blocks.{layer}.hook_resid_post"
        )

    @pytest.mark.slow
    def test_nnterp_tl_activation_equivalence_resid_pre(self):
        """layers_output index must match blocks.{L}.hook_resid_pre for GPT-2.

        hook_resid_pre of layer L captures the residual stream BEFORE block L,
        which equals the output of block L-1. So nnterp layers_output[L-1]
        should match.
        """
        layer = 8
        hook_name = f"blocks.{layer}.hook_resid_pre"
        device = "cuda"

        tl_model, _sae, _tok = load_sae_and_model(
            model_name="gpt2",
            layer=layer,
            device=device,
            sae_release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        nn_model = build_nnterp_model(tl_model, device=device)
        assert nn_model is not None, "nnterp build failed for GPT-2"

        tokens = t.randint(0, 50257, (2, 64), device=device)

        with t.no_grad():
            tl_acts = _get_activations_tl(
                tl_model, tokens, layer, hook_name=hook_name
            )
            nn_acts = _get_activations_nnterp(
                nn_model, tokens, layer, ActivationType.residual, hook_name=hook_name
            )

        assert tl_acts.shape == nn_acts.shape, (
            f"Shape mismatch: TL {tl_acts.shape} vs nnterp {nn_acts.shape}"
        )

        tl_cpu = tl_acts.float().cpu()
        nn_cpu = nn_acts.float().cpu()

        max_diff = (tl_cpu - nn_cpu).abs().max().item()
        cos_sim = t.nn.functional.cosine_similarity(
            tl_cpu.flatten().unsqueeze(0),
            nn_cpu.flatten().unsqueeze(0),
        ).item()

        assert t.allclose(tl_cpu, nn_cpu, atol=1e-3), (
            f"hook_resid_pre activations diverge: max_diff={max_diff:.2e}, "
            f"cos_sim={cos_sim:.6f}."
        )


class TestTrimmedModel:
    """Verify trimmed nnterp model produces same activations as full model."""

    @pytest.mark.slow
    def test_trimmed_nnterp_matches_full_gpt2(self):
        """Trimmed GPT-2 nnterp model gives identical activations at target layer."""
        layer = 8
        device = "cuda"

        tl_model, _sae, _tok = load_sae_and_model(
            model_name="gpt2", layer=layer, device=device,
            sae_release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )

        # Full model
        nn_full = build_nnterp_model(tl_model, device=device)
        assert nn_full is not None

        tokens = t.randint(0, 50257, (2, 64), device=device)
        with t.no_grad():
            full_acts = _get_activations_nnterp(
                nn_full, tokens, layer, ActivationType.residual,
                hook_name=f"blocks.{layer}.hook_resid_post",
            )

        # Trimmed model — rebuild from scratch to avoid mutation issues
        tl_model2, _, _ = load_sae_and_model(
            model_name="gpt2", layer=layer, device=device,
            sae_release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        nn_trimmed = build_nnterp_model(tl_model2, device=device)
        assert nn_trimmed is not None
        trim_hf_model(nn_trimmed.model, layer)

        with t.no_grad():
            trimmed_acts = _get_activations_nnterp(
                nn_trimmed, tokens, layer, ActivationType.residual,
                hook_name=f"blocks.{layer}.hook_resid_post",
            )

        assert t.allclose(full_acts.float().cpu(), trimmed_acts.float().cpu(), atol=1e-5), (
            f"Trimmed model diverges from full model at layer {layer}"
        )

    @pytest.mark.slow
    def test_trimmed_model_parameter_reduction(self):
        """Trimmed model has fewer parameters than full model."""
        full = GPT2LMHeadModel.from_pretrained("gpt2")
        full_params = sum(p.numel() for p in full.parameters())

        trimmed = GPT2LMHeadModel.from_pretrained("gpt2")
        trim_hf_model(trimmed, layer=4)
        trimmed_params = sum(p.numel() for p in trimmed.parameters())

        assert trimmed_params < full_params
        assert trimmed_params / full_params < 0.70


class TestSAEFeatureEquivalence:
    """Verify nnterp and TL produce identical SAE features."""

    @pytest.mark.slow
    def test_top_features_match_gpt2(self):
        """Top-k SAE features must be identical between nnterp and TL activations."""
        layer = 8
        hook_name = f"blocks.{layer}.hook_resid_post"
        device = "cuda"

        tl_model, sae, _tok = load_sae_and_model(
            model_name="gpt2", layer=layer, device=device,
            sae_release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        nn_model = build_nnterp_model(tl_model, device=device)
        assert nn_model is not None

        tokens = t.randint(0, 50257, (1, 32), device=device)

        with t.no_grad():
            tl_acts = _get_activations_tl(tl_model, tokens, layer, hook_name=hook_name)
            nn_acts = _get_activations_nnterp(nn_model, tokens, layer, ActivationType.residual, hook_name=hook_name)

        tl_features = sae.encode(tl_acts.squeeze(0).float())
        nn_features = sae.encode(nn_acts.squeeze(0).float())

        k = 10
        tl_topk = tl_features.topk(k, dim=-1).indices
        nn_topk = nn_features.topk(k, dim=-1).indices

        match_rate = (tl_topk == nn_topk).float().mean().item()
        assert match_rate > 0.95, f"Top-{k} feature match rate is {match_rate:.2%}, expected >95%"


class TestEndToEndProbeEquivalence:
    """Verify get_probe_activations with nnterp and TL produce same probe results."""

    @pytest.mark.slow
    def test_probe_values_match_gpt2(self):
        """Raw probe values from nnterp path must match TL path."""
        from steering.eval.feature_probe import compute_raw_probe

        layer = 8
        hook_name = f"blocks.{layer}.hook_resid_post"
        device = "cuda"

        tl_model, sae, _tok = load_sae_and_model(
            model_name="gpt2", layer=layer, device=device,
            sae_release="gpt2-small-res-jb",
            sae_id=f"blocks.{layer}.hook_resid_pre",
        )
        nn_model = build_nnterp_model(tl_model, device=device)
        assert nn_model is not None

        tokens = t.randint(0, 50257, (1, 32), device=device)
        feature_id = 0

        with t.no_grad():
            tl_act = get_probe_activations(
                tl_model, tokens, layer, hook_name,
                ActivationType.residual, nnterp_model=None,
            ).squeeze(0).float()
            nn_act = get_probe_activations(
                tl_model, tokens, layer, hook_name,
                ActivationType.residual, nnterp_model=nn_model,
            ).squeeze(0).float()

        tl_probe = compute_raw_probe(tl_act, sae.W_enc, sae.b_enc, sae.b_dec, feature_id, apply_b_dec=True)  # ty: ignore[invalid-argument-type]
        nn_probe = compute_raw_probe(nn_act, sae.W_enc, sae.b_enc, sae.b_dec, feature_id, apply_b_dec=True)  # ty: ignore[invalid-argument-type]

        assert t.allclose(tl_probe.cpu(), nn_probe.cpu(), atol=1e-2), (
            f"Probe values diverge: max_diff={(tl_probe - nn_probe).abs().max().item():.4f}"
        )
