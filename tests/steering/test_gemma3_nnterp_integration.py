"""Live integration: download gemma-3-270m + one v2 SAE on CPU.

Skipped by default (very_slow marker). Run manually on a networked host
with HF_TOKEN access to google/gemma-3-270m and the gemma-scope-2 repos:
    uv run pytest tests/steering/test_gemma3_nnterp_integration.py -m very_slow -v
"""

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

from steering.extract_features import (
    _build_nnterp_gemma3,
    _get_activations_nnterp,
)
from steering.models import resolve_tokenizer
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)

pytestmark = pytest.mark.very_slow


@pytest.fixture(scope="module")
def gemma3_270m_nnterp_model():
    # _build_nnterp_gemma3 is keyword-only:
    #   (*, model_name: str, tl_model: HookedTransformer | None,
    #    device: str | torch.device, model_dtype: torch.dtype | None = None)
    return _build_nnterp_gemma3(
        model_name="gemma-3-270m",
        tl_model=None,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )


@pytest.fixture(scope="module")
def gemma3_270m_tokenizer():
    # _build_nnterp_gemma3 wraps an already-instantiated Gemma3ForCausalLM in
    # StandardizedTransformer. nnsight's LanguageModel only auto-loads a
    # tokenizer when given a repo-id string; for a pre-built model object it
    # leaves ``.tokenizer`` as None. The production extraction pipeline obtains
    # the tokenizer via ``resolve_tokenizer`` (see extract_features.py:469),
    # which is the correct, pipeline-faithful path to use here.
    tokenizer = resolve_tokenizer("gemma-3-270m")
    return tokenizer


def test_270m_loads_and_has_expected_layers(gemma3_270m_nnterp_model):
    nnterp_model = gemma3_270m_nnterp_model
    # gemma-3-270m has 18 transformer layers (verified from HF config).
    assert len(nnterp_model.model.layers) == 18


def test_resolved_layer_module_is_gemma3_decoder(gemma3_270m_nnterp_model):
    nnterp_model = gemma3_270m_nnterp_model
    # nnterp/nnsight wraps every submodule in an ``Envoy`` proxy, so
    # ``nnterp_model.model.layers[12]`` is the Envoy, not the raw nn.Module.
    # The underlying torch module is held on ``Envoy._module``.
    envoy = nnterp_model.model.layers[12]
    layer = getattr(envoy, "_module", envoy)
    # transformers re-exports Gemma3DecoderLayer from several module paths
    # (modeling_gemma3 vs the dynamically-built modular path); the loaded
    # instance's class is not guaranteed to be *identical* to the
    # ``modeling_gemma3`` symbol, so a plain isinstance() is brittle. Assert on
    # the MRO class names instead — this still verifies it is the Gemma 3
    # decoder block. The canonical import is kept for traceability/expected
    # name reference.
    assert Gemma3DecoderLayer.__name__ == "Gemma3DecoderLayer"
    assert any(
        cls.__name__ == "Gemma3DecoderLayer" for cls in type(layer).__mro__
    ), f"layer 12 is {type(layer).__name__}, expected a Gemma3DecoderLayer"


def test_residual_extraction_shape_at_layer_12(
    gemma3_270m_nnterp_model, gemma3_270m_tokenizer
):
    nnterp_model = gemma3_270m_nnterp_model
    tokens = gemma3_270m_tokenizer("Hello world", return_tensors="pt")["input_ids"]
    # _get_activations_nnterp signature (positional order differs from the
    # call below, but all args are passed by keyword):
    #   (nnterp_model, tokens, layer,
    #    activation_type=ActivationType.residual, hook_name=None)
    acts = _get_activations_nnterp(
        nnterp_model=nnterp_model,
        tokens=tokens,
        activation_type=ActivationType.residual,
        layer=12,
        hook_name="blocks.12.hook_resid_post",
    )
    assert acts.shape[-1] == 640  # gemma-3-270m hidden_size
    assert torch.isfinite(acts).all()


def test_attention_post_projection_shape(
    gemma3_270m_nnterp_model, gemma3_270m_tokenizer
):
    nnterp_model = gemma3_270m_nnterp_model
    tokens = gemma3_270m_tokenizer("Hello", return_tensors="pt")["input_ids"]
    acts = _get_activations_nnterp(
        nnterp_model=nnterp_model,
        tokens=tokens,
        activation_type=ActivationType.attention,
        layer=12,
        hook_name="blocks.12.hook_attn_out",
    )
    # The Gemma3 self-attention module returns ``(attn_output, attn_weights)``,
    # so the post-projection ``hook_attn_out`` path yields a tuple. This is the
    # documented contract: production callers unwrap ``activations[0]`` (see
    # extract_features.py: `if isinstance(activations, (tuple, list)):
    # activations = activations[0]`). Mirror that here.
    if isinstance(acts, (tuple, list)):
        acts = acts[0]
    # Post-projection: (batch, seq, d_model), NOT (batch, seq, heads, head_dim).
    assert acts.shape[-1] == 640
    assert acts.dim() == 3
    assert torch.isfinite(acts).all()


def test_resolve_sae_for_270m_it_v2():
    # Pure-resolution check (no download): confirms the 270m-it v2 combo
    # resolves to the expected SAELens identifiers.
    resolved = resolve_sae(
        model="gemma-3-270m-it",
        release=SAERelease.gemmascope_2,
        activation_type=ActivationType.residual,
        layer=12,
        width=SAEWidth.W_16K,
        l0=L0Tier.medium,
    )
    assert resolved.release == "gemma-scope-2-270m-it-res"
    assert resolved.sae_id == "layer_12_width_16k_l0_medium"
    assert resolved.hook_name == "blocks.12.hook_resid_post"
