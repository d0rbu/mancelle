"""Real-model integration guards for this PR's two central correctness
properties. Skipped unless CUDA + HF_TOKEN are present; marked very_slow
(run on the extraction box, not CI):

    uv run pytest tests/steering/test_gemma_dtype_offload_integration.py \
        -m very_slow -v

1. _build_nnterp_gemma3 offloads the fp32 TransformerLens companion to CPU
   BEFORE loading the HF Gemma-3 copy (the root-cause 24GB-GPU OOM fix).
2. The gemma-2 nnterp path still verifies — TL stays fp32 so
   _build_nnterp_gemma2's fp32 _verify_logits_match(atol=0.1) passes. This
   is the B1-regression guard: the reverted bf16-TL approach broke it.
"""

import os

import pytest
import torch as t
import transformer_lens as tl

from steering.extract_features import _build_nnterp_gemma3, build_nnterp_model

pytestmark = pytest.mark.very_slow

_NEED_GPU_HF = pytest.mark.skipif(
    not t.cuda.is_available() or not os.environ.get("HF_TOKEN"),
    reason="requires CUDA + HF_TOKEN (gated Gemma repos)",
)


@_NEED_GPU_HF
def test_gemma3_build_offloads_tl_to_cpu():
    """Root-cause OOM fix: after _build_nnterp_gemma3, the TL companion's
    parameters must all be on CPU (offloaded before the HF Gemma-3 load),
    and the returned nnterp model must run a forward pass."""
    tl_model = tl.HookedTransformer.from_pretrained(
        "gemma-3-270m", device="cuda"
    )
    t.cuda.reset_peak_memory_stats()
    nn = _build_nnterp_gemma3(
        model_name="gemma-3-270m",
        tl_model=tl_model,
        device=t.device("cuda"),
        model_dtype=t.bfloat16,
    )
    assert nn is not None
    assert all(p.device.type == "cpu" for p in tl_model.parameters()), (
        "TL companion must be offloaded to CPU by _build_nnterp_gemma3"
    )
    with t.no_grad():
        out = nn(t.tensor([[1, 2, 3, 4, 5]], device="cuda"))
    assert out.logits is not None
    assert out.logits.shape[-1] > 0, "expected a non-empty vocab dim"
    assert bool(t.isfinite(out.logits).all()), (
        "nnterp forward produced non-finite logits"
    )
    # Informational: peak GPU bytes after offload+build (no hard threshold —
    # gemma-3-270m is tiny; the offload assertion above is the real guard).
    print(
        "peak GPU MiB after build:",
        t.cuda.max_memory_allocated() / (1024 * 1024),
    )


@_NEED_GPU_HF
def test_gemma2_nnterp_verify_still_passes():
    """B1-regression guard: gemma-2 nnterp build must still succeed, which
    means _build_nnterp_gemma2's fp32 _verify_logits_match(atol=0.1) passed
    (TL stays fp32 — the reverted bf16-TL approach would fail this gate and
    return None, silently disabling nnterp)."""
    tl_model = tl.HookedTransformer.from_pretrained(
        "gemma-2-2b", device="cuda"
    )
    nn = build_nnterp_model(tl_model, device="cuda", model_dtype=t.bfloat16)
    assert nn is not None, (
        "gemma-2 nnterp build returned None — _verify_logits_match failed "
        "(B1 regression). TL must remain fp32 for the fp32 verify gate."
    )
