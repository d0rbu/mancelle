"""Integration test: real GemmaScope SAE + Gemma model (v1 and v2).

Parametrised over both GemmaScope releases:
- v1: gemma-2-2b + gemmascope residual 16k SAE
- v2: gemma-3-270m + gemmascope_2 residual 16k SAE (l0 tier "medium")

Downloads model weights on first run (~5GB). Skip with:
    uv run pytest tests/test_gemmascope_integration.py -v -k "not slow"

Run with:
    uv run pytest tests/test_gemmascope_integration.py -v
"""

import pytest
import torch as t
from sae_lens import SAE
from transformer_lens import HookedTransformer

from steering.extract_features import load_sae_and_model
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)

# All tests in this file need real model downloads
pytestmark = pytest.mark.slow

# (model_name, release, activation, layer, width, l0) tuples covering both
# GemmaScope releases. Layer 12 is valid for both gemma-2-2b (26 layers) and
# gemma-3-270m (layers {5, 9, 12, 15}).
_GEMMASCOPE_VARIANTS = [
    (
        "gemma-2-2b",
        SAERelease.gemmascope,
        ActivationType.residual,
        12,
        SAEWidth.W_16K,
        None,
    ),
    (
        "gemma-3-270m",
        SAERelease.gemmascope_2,
        ActivationType.residual,
        12,
        SAEWidth.W_16K,
        L0Tier.medium,
    ),
]
_GEMMASCOPE_VARIANT_IDS = ["v1-gemma2-2b", "v2-gemma3-270m"]


@pytest.fixture(
    scope="module",
    params=_GEMMASCOPE_VARIANTS,
    ids=_GEMMASCOPE_VARIANT_IDS,
)
def gemma2_residual(request):
    """Load a Gemma model + residual 16k SAE (cached per-variant in module).

    Parametrised over v1 (gemma-2-2b / gemmascope) and v2 (gemma-3-270m /
    gemmascope_2). Yields ``(model, sae, resolved)``.
    """
    model_name, release, activation, layer, width, l0 = request.param
    resolved = resolve_sae(
        model_name,
        release,
        activation,
        layer,
        width,
        l0,
    )
    model = HookedTransformer.from_pretrained(model_name, device="cpu")
    sae = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cpu",
    )
    return model, sae, resolved


class TestGemmaScopeResidual:
    def test_model_loads(self, gemma2_residual):
        model, _, resolved = gemma2_residual
        # Architecture differs across v1 (gemma-2-2b) and v2 (gemma-3-270m),
        # so assert structural sanity rather than hardcoded dimensions.
        assert model.cfg.n_layers > 0
        assert model.cfg.d_model > 0
        # The resolved hook targets layer 12, which must exist in the model
        # (valid for gemma-2-2b: 26 layers; gemma-3-270m: layers {5,9,12,15}).
        assert model.cfg.n_layers > 12, (
            f"layer 12 out of range for {model.cfg.model_name} "
            f"({model.cfg.n_layers} layers)"
        )
        assert "blocks.12." in resolved.hook_name

    def test_sae_loads(self, gemma2_residual):
        _, sae, _ = gemma2_residual
        assert sae.cfg.d_sae == 16384

    def test_forward_pass_produces_activations(self, gemma2_residual):
        model, sae, resolved = gemma2_residual
        tokens = model.tokenizer.encode("The quick brown fox")
        tokens_t = t.tensor([tokens], dtype=t.long)

        with t.no_grad():
            _, cache = model.run_with_cache(
                tokens_t,
                names_filter=[resolved.hook_name],
            )
            activations = cache[resolved.hook_name]

        assert activations.ndim == 3  # [batch, seq, d_model]
        assert activations.shape[0] == 1
        assert activations.shape[1] == len(tokens)
        assert activations.shape[2] == model.cfg.d_model

    def test_sae_encode_produces_sparse_features(self, gemma2_residual):
        model, sae, resolved = gemma2_residual
        tokens = model.tokenizer.encode("The quick brown fox")
        tokens_t = t.tensor([tokens], dtype=t.long)

        with t.no_grad():
            _, cache = model.run_with_cache(
                tokens_t,
                names_filter=[resolved.hook_name],
            )
            activations = cache[resolved.hook_name]
            sae_out = sae.encode(activations.squeeze(0))

        assert sae_out.shape == (len(tokens), 16384)
        # SAE should produce sparse output (some nonzero, mostly zero)
        nonzero_frac = (sae_out > 0).float().mean().item()
        assert 0.001 < nonzero_frac < 0.5, f"Unexpected sparsity: {nonzero_frac:.4f}"

    @pytest.mark.parametrize(
        "model_name,release,activation,layer,width,l0",
        _GEMMASCOPE_VARIANTS,
        ids=_GEMMASCOPE_VARIANT_IDS,
    )
    def test_load_sae_and_model_structured_path(
        self, model_name, release, activation, layer, width, l0
    ):
        """Verify load_sae_and_model works end-to-end with resolved config."""
        resolved = resolve_sae(
            model_name,
            release,
            activation,
            layer,
            width,
            l0,
        )
        model, sae, tokenizer = load_sae_and_model(
            model_name,
            layer=layer,
            device="cpu",
            resolved=resolved,
        )
        assert model is not None
        assert sae is not None
        assert tokenizer is not None
        assert sae.cfg.d_sae == 16384


class TestGemmaScopeHookNames:
    """Verify hook names produce valid activations for each activation type."""

    def test_residual_hook(self, gemma2_residual):
        model, _, resolved = gemma2_residual
        tokens_t = t.tensor([[1, 2, 3]], dtype=t.long)
        with t.no_grad():
            _, cache = model.run_with_cache(
                tokens_t,
                names_filter=[resolved.hook_name],
            )
        assert resolved.hook_name in cache

    def test_mlp_hook_exists(self, gemma2_residual):
        model, _, _ = gemma2_residual
        hook_name = "blocks.12.hook_mlp_out"
        tokens_t = t.tensor([[1, 2, 3]], dtype=t.long)
        with t.no_grad():
            _, cache = model.run_with_cache(
                tokens_t,
                names_filter=[hook_name],
            )
        assert hook_name in cache
        assert cache[hook_name].ndim == 3

    def test_attention_hook_z_is_4d(self, gemma2_residual):
        model, _, _ = gemma2_residual
        hook_name = "blocks.12.attn.hook_z"
        tokens_t = t.tensor([[1, 2, 3]], dtype=t.long)
        with t.no_grad():
            _, cache = model.run_with_cache(
                tokens_t,
                names_filter=[hook_name],
            )
        assert hook_name in cache
        activations = cache[hook_name]
        # hook_z should be 4D: [batch, seq, n_heads, d_head]
        assert activations.ndim == 4
