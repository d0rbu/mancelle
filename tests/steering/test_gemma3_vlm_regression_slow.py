"""Slow regression tests for gemma-3-4b-pt VLM text-weight loading (issue #549).

Before the fix in :mod:`steering.gemma3_loading`, ``Gemma3ForCausalLM``
silently random-initialised the text decoder for VLM repos (weights stored under
``language_model.model.*``). This caused feature 41629 (app-store / Play-Store
URL activator on GemmaScope-2 layer 22, width 65k) to return activation ≈ 0 even
on canonical URL tokens.

These tests are marked ``@pytest.mark.slow`` and require:
  - CUDA (skip otherwise)
  - Cached HF weights for ``gemma-3-4b-pt`` and the GemmaScope-2 SAE
    (requires a valid ``HF_TOKEN`` with access to the gated repos, plus a
    local HF cache; missing weights will raise ``ModelWeightLoadError`` or
    an HF ``OSError``, not skip)

``sae_lens`` and ``nnterp`` are hard test-suite dependencies (imported at
module level via :mod:`steering.extract_features`) so their absence is a
collection-time error, not a skip.
"""

from __future__ import annotations

import pytest
import torch as t
from sae_lens import SAE

from steering.extract_features import _get_activations_nnterp, build_nnterp_model
from steering.gemma3_loading import load_gemma3_text_decoder  # noqa: F401
from steering.models import resolve_tokenizer
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)

# Canonical URL text used to probe feature 41629. The feature is the app-store /
# Play-Store URL activator on GemmaScope-2 gemma-3-4b-pt layer 22, width 65k
# (verified on Neuronpedia: strongly positive on URL tokens in Play-Store links).
NEURONPEDIA_FEATURE_41629_TEXT = (
    "Download the app at https://play.google.com/store/apps/details?id="
    "com.example.myapp and enjoy."
)


@pytest.mark.slow
def test_feature_41629_fires_on_url_tokens_gemma3_4b():
    """Feature 41629 activates strongly (> 100) on app-store URL tokens.

    PRE-FIX (random weights): the SAE decoder is incoherent because the text
    decoder was randomly initialised. Feature 41629 never exceeds threshold
    on any input, so ``col.max() > 100.0`` FAILS. The peak-position assertion
    can also fail or produce a meaningless result with random activations.

    POST-FIX (correct VLM text weights loaded via
    :mod:`steering.gemma3_loading`): the SAE decodes real interpretable
    features. Feature 41629 fires strongly (≥ threshold > 100) and the peak
    lands on or after the URL region in the tokenised sequence.

    The ``noqa: F401`` import of :func:`steering.gemma3_loading.load_gemma3_text_decoder`
    above proves that the module is importable (catches missing-module errors at
    collection time rather than at runtime).
    """
    if not t.cuda.is_available():
        pytest.skip("needs CUDA")

    model_name, layer, fid = "gemma-3-4b-pt", 22, 41629
    resolved = resolve_sae(
        model=model_name,
        release=SAERelease.gemmascope_2,
        activation_type=ActivationType.residual,
        layer=layer,
        width=SAEWidth.W_65K,
        l0=L0Tier.medium,
    )
    tok = resolve_tokenizer(model_name)
    all_ids = tok(NEURONPEDIA_FEATURE_41629_TEXT, add_special_tokens=True)["input_ids"]
    ids = t.tensor([all_ids], device="cuda")

    # tl_model=None: gemma-3 loads directly from HF via the VLM-aware loader
    # in steering.gemma3_loading (wired through build_nnterp_model internally).
    nnterp = build_nnterp_model(
        tl_model=None, device="cuda", model_dtype=t.bfloat16,
        model_name=model_name,
    )
    sae_raw = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cuda",
    )
    sae = sae_raw[0] if isinstance(sae_raw, tuple) else sae_raw
    sae = sae.to(t.bfloat16)

    with t.no_grad():
        # _get_activations_nnterp returns [batch, seq, d_model]; [0] → [seq, d_model].
        resid = _get_activations_nnterp(
            nnterp, ids, layer,  # ty: ignore[invalid-argument-type]
            activation_type=ActivationType.residual,
            hook_name=resolved.hook_name,
        )[0]
        feats = sae.encode(resid)  # [seq, n_features]

    col = feats[:, fid].float()

    # PRIMARY REGRESSION CHECK: feature fires strongly on the URL.
    # Pre-fix random-weight regime: col.max() ≈ 0 (incoherent decoder).
    assert col.max() > 100.0, (
        f"feature {fid} max={col.max().item():.1f} ≤ 100.0 on URL text; "
        f"expected strong activation on app-store URL tokens — likely the "
        f"pre-fix VLM random-weight regime ({model_name} text decoder not "
        f"loaded correctly; see issue #549 and steering/gemma3_loading.py)"
    )

    # SECONDARY CHECK: the activation peak lands on or after the URL region.
    # Find the approximate start of the URL in the token sequence by searching
    # for the first token that encodes the URL prefix ("https"). A try/except
    # handles edge cases where the tokeniser splits "https" across multiple
    # tokens: in that case url_start_idx defaults to 0 (no positional
    # constraint), keeping the primary assertion as the load-bearing check.
    url_prefix_ids = tok("https", add_special_tokens=False)["input_ids"]
    try:
        url_start_idx = next(
            i for i in range(len(all_ids))
            if all_ids[i] == url_prefix_ids[0]
        )
    except (StopIteration, IndexError):
        url_start_idx = 0

    assert int(col.argmax()) >= url_start_idx - 2, (
        f"feature {fid} peak at token {int(col.argmax())} is more than 2 "
        f"positions before the URL region (starts at token ~{url_start_idx}); "
        f"expected the activation peak within the Play-Store URL fragment"
    )
