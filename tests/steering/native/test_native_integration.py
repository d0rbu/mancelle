"""Layer 4: native vs nnterp vs TL feature-equivalence on REAL models.

Marked `slow`: requires CUDA, HF_TOKEN with gated google/gemma access, and
real GemmaScope SAE weights. Excluded from the default pytest run.

The headline residual gate (`test_native_matches_nnterp_and_tl_residual`) is
hardened (H-R3) to a PRODUCTION-shaped workload: a ragged batch of real
natural-language documents tokenized with the model's own gated-HF tokenizer
(`resolve_tokenizer`), right-padded byte-for-byte the way
`_get_sae_activations_batched` pads in production, with at least one document
long enough to exceed gemma-3's `sliding_window` so the sliding-vs-full
attention boundary is genuinely stressed at scale; parametrized over the
PRODUCTION dtype (bf16) as well as fp32; and instrumented with a peak/resident
VRAM gate that fails if the gc-pinning regression (whole model resident through
the truncated build) returns.
"""

import gc
from typing import Any

import pytest
import torch as t
import torch._dynamo
import torch._inductor.config as _inductor_config
from sae_lens import SAE
from transformers import AutoConfig, AutoModelForCausalLM, Gemma3ForCausalLM

from errors import CompileGateError, ExtractionError
from steering.extract_features import (
    _gate_sparse_features,
    _get_sae_activations_batched,
    _native_compile_gate,
    build_nnterp_model,
    load_sae_and_model,
)
from steering.feature_equivalence import (
    BASELINE_RELATIVE_SLACK,
    compiled_passes_ideal_math_gate,
)
from steering.models import EXTRACTION_MODELS, resolve_tokenizer
from steering.native_collector import _NATIVE_ATTN_IMPLEMENTATION, build_native_model
from steering.sae_resolver import ActivationType, L0Tier, SAERelease, SAEWidth, resolve_sae
from tests.steering.native._hook_reference import hook_reference_residual
from tests.steering.native.conftest_parity import (
    BF16_NATIVE_VS_TL_MAG_ATOL,
    BF16_NATIVE_VS_TL_MAG_RTOL,
    FP32_NATIVE_VS_TL_MAG_ATOL,
    FP32_NATIVE_VS_TL_MAG_RTOL,
    JACCARD_THRESHOLD,
    assert_bit_identical,
    assert_feature_equivalent,
    assert_features_nonempty,
    feature_equivalence_metrics,
    feature_magnitude_metrics,
)

# --------------------------------------------------------------------------- #
# Realistic ragged document corpus for the hardened residual gate.            #
#                                                                             #
# Four representative natural-language documents of DELIBERATELY DIFFERING    #
# lengths (one short sentence → one long multi-paragraph essay).  Tokenized   #
# with the model's real gated-HF tokenizer (`resolve_tokenizer`, padding=     #
# False — exactly how the production worker `_tokenize_and_head_truncate`     #
# encodes a chunk) they yield a RAGGED batch which `_get_sae_activations_     #
# batched` then right-pads with id 0 to the max length and strips per true    #
# length — the verbatim production padding contract this gate exercises end   #
# to end.                                                                     #
#                                                                             #
# `_LONG_DOC` is several distinct paragraphs of coherent English prose (NOT   #
# a spammed/repeated token).  Tokenized with the REAL gemma tokenizers via    #
# `resolve_tokenizer` (`add_special_tokens=True`, padding=False — exactly as  #
# the test below tokenizes) it measures, empirically (verified 2026-05-18     #
# against google/gemma-3-1b-pt and google/gemma-2-2b):                        #
#     gemma-3-1b-pt : 1215 tokens                                             #
#     gemma-2-2b    : 1218 tokens                                             #
# i.e. ~2.4× gemma-3-1b-pt's real published `sliding_window` (512), a ~700-   #
# token margin so a tokenizer-version drift cannot silently drop it back      #
# under the window and re-vacuate the seq>window assertion.  gemma-3-1b-pt@7  #
# keeps layers 0..7, whose `layer_types[:8]` is [sliding×5, full@5,           #
# sliding×2] (gemma-3-1b-pt config: sliding_window_pattern=6 ⇒ index 5 is     #
# the ONLY full_attention layer in the kept slice), so a >sliding_window      #
# sequence runs the per-type sliding mask AND the full-attention mask at      #
# PRODUCTION scale.  The test asserts max(true_lengths) > the loaded model's  #
# real `sliding_window` at runtime (and FAILS LOUDLY, never silently skips,   #
# if a small-window model's corpus is ever too short — see                    #
# `_sliding_window_guard`), so the seq>window guarantee is verified, never    #
# silently assumed.                                                           #
# --------------------------------------------------------------------------- #
_SHORT_DOC = "The cat sat on the mat."

_MED_DOC = (
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose. Chlorophyll in the chloroplasts absorbs sunlight, and the "
    "Calvin cycle then fixes carbon dioxide into sugar."
)

_PARA_DOC = (
    "The history of computing is a story of abstraction. Early mechanical "
    "calculators gave way to electromechanical relays, then vacuum tubes, "
    "transistors, and finally integrated circuits. Each layer hid the "
    "complexity beneath it, letting programmers reason about problems rather "
    "than wiring. Today an engineer composes services without ever touching "
    "a logic gate, yet every abstraction still rests on the physics of "
    "silicon."
)

# A long, multi-paragraph essay of distinct coherent sentences (NOT a single
# token spammed). Empirically measured (2026-05-18) with the REAL gemma
# tokenizers via `resolve_tokenizer`, add_special_tokens=True, padding=False —
# the exact tokenization the test below performs:
#     gemma-3-1b-pt : 1215 tokens   (real sliding_window=512 ⇒ ~2.4× over)
#     gemma-2-2b    : 1218 tokens   (real sliding_window=4096 ⇒ under, by
#                                    design — see _sliding_window_guard)
# The ~700-token margin over gemma-3-1b-pt's 512 window means a tokenizer-
# version drift cannot silently drop it back under the window. Natural-
# language prose throughout so the SAE fires real, non-vacuous features.
_LONG_DOC = (
    "Language models learn statistical regularities from vast text corpora. "
    "During pretraining the network repeatedly predicts the next token given "
    "every previous one, and through this deceptively simple objective it "
    "gradually internalizes syntax, morphology, semantics, factual world "
    "knowledge, discourse structure, and even rudimentary multi-step "
    "reasoning. The optimization landscape is shaped by the data: rare "
    "constructions are learned slowly, frequent idioms quickly, and the "
    "model's eventual competence is a smeared average over everything it has "
    "read. Researchers who want to understand what such a network has "
    "actually learned cannot simply inspect its billions of weights; the "
    "representations are distributed, superposed, and only loosely aligned "
    "with human concepts. "
    "Sparse autoencoders offer one tractable lens. The idea is to take a "
    "dense activation vector from somewhere inside the transformer and "
    "decompose it into a sparse, overcomplete sum of learned feature "
    "directions, each of which ideally corresponds to a single interpretable "
    "concept the model tracks. A single residual-stream position, read this "
    "way, can light up dozens of features at once: one for the grammatical "
    "subject, another for the topic of the sentence, others for sentiment, "
    "register, or the presence of code. Cataloguing which features fire, "
    "where they fire, and how strongly is the daily work of mechanistic "
    "interpretability, and it only yields trustworthy answers when the "
    "activations being decomposed were captured faithfully in the first "
    "place. A subtle bug in the capture path — an off-by-one in the layer "
    "index, a transposed reshape, a pooling step applied before instead of "
    "after a nonlinearity — silently poisons every downstream feature, and "
    "no amount of clever analysis recovers from corrupted inputs. "
    "Attention is the mechanism by which a transformer routes information "
    "between positions. In a causal decoder every position attends only to "
    "earlier ones, never to the future, so any padding appended to the right "
    "of a shorter sequence is provably unable to influence the activations "
    "at genuine token positions — a property that production batching code "
    "leans on heavily when it right-pads a ragged batch to a common length "
    "and then strips each row back to its true length afterwards. Some "
    "layers further restrict attention with a sliding window, allowing each "
    "position to see only a fixed number of recent tokens, while other "
    "layers attend globally across the whole context. Correctly interleaving "
    "these two regimes, in exactly the published pattern, is essential for "
    "long-context fidelity, and it is also the part most likely to be "
    "implemented inconsistently across two different model backends. Crucial "
    "to any honest equivalence test is the observation that when a sequence "
    "is shorter than the sliding window the windowed mask and the full "
    "causal mask are numerically identical: the window simply never bites. "
    "Only a sequence strictly longer than the window forces the two masks "
    "apart, so only a genuinely long document actually exercises that "
    "boundary. A test corpus that stays under the window certifies nothing "
    "about the sliding-versus-full attention path no matter how many times "
    "it is run, which is precisely why the longest document here is sized, "
    "and verified against the real tokenizer, to comfortably exceed it. "
    "Numerical reproducibility across backends is its own deep subtlety. The "
    "same model, run with a different attention kernel, a different "
    "reduction order, or a different floating-point precision, can emit "
    "logits that differ in the last few bits, and those tiny differences are "
    "then amplified as they propagate through the sparse autoencoder into "
    "the reported feature magnitudes. A faithful native collector must "
    "therefore clear two distinct bars. First, when it runs the identical "
    "kernel at the identical dtype as the reference implementation, it must "
    "match that reference bit for bit, with no tolerance at all, because any "
    "divergence under identical conditions can only be a real capture-site, "
    "reshape, or pooling defect rather than benign numerical noise. Second, "
    "when it is compared against a backend that deliberately uses a "
    "different attention implementation — eager versus fused, say — it must "
    "stay within a small, explicitly documented tolerance whose envelope was "
    "measured empirically rather than guessed. Demonstrating both properties "
    "on realistic, ragged, production-scale input, instead of a handful of "
    "hand-picked synthetic token identifiers, is what finally gives an "
    "equivalence gate real teeth: it stresses the exact padding, masking, "
    "long-context, and precision paths that production traffic stresses, and "
    "it fails loudly the moment any of them silently drifts. "
    "It is worth dwelling on why the ragged shape of the batch matters as "
    "much as its length. Real extraction traffic almost never consists of "
    "documents that happen to share a token count; it is a stream of short "
    "queries, medium passages, and the occasional very long article, all "
    "submitted together. The batching layer pads them to a common width, "
    "runs a single forward pass, and then must hand back exactly the "
    "activations belonging to each original document and nothing else. If "
    "the strip step is off by even one position, or if the padding tokens "
    "leak into a pooled statistic, the resulting features look plausible yet "
    "are quietly wrong, and because they are plausible they survive code "
    "review and contaminate every experiment built on top of them. A test "
    "that uses only uniform-length inputs can never surface this class of "
    "defect, so the corpus here deliberately mixes a one-line sentence, a "
    "couple of mid-sized paragraphs, and this deliberately long essay, "
    "guaranteeing the padding and strip arithmetic is exercised with a "
    "genuinely uneven batch rather than a degenerate one. "
    "Finally, consider the asymmetry of the cost. The work of writing a "
    "thorough equivalence test is paid once, by one engineer, over a few "
    "focused hours. The cost of shipping a subtly miscaptured collector is "
    "paid continuously and invisibly: every downstream interpretability "
    "result computed against the faulty activations is wrong, the people "
    "consuming those results make decisions on a false basis, and the "
    "eventual debugging — once someone finally notices the numbers do not "
    "reconcile — sprawls across the entire pipeline because the corruption "
    "entered at the very first step and was faithfully propagated by every "
    "step after it. That asymmetry is the entire justification for spending "
    "real effort on a gate like this one, and for insisting that it stress "
    "the genuine production paths, on realistic input, at production "
    "precision, instead of merely going through the motions on a toy batch "
    "that proves nothing while appearing green."
)

# Differing lengths ⇒ a genuinely ragged batch (production right-pads it).
_REALISTIC_DOCS = [_SHORT_DOC, _MED_DOC, _PARA_DOC, _LONG_DOC]


def _hf_text_config(model_name: str):
    """The model's published HF config (text sub-config if nested).

    The EXACT config the production model loads (`build_native_model` ->
    `EXTRACTION_MODELS[model_name].hf_model_id`); cached locally in the
    slow/GPU env (gated google/gemma access is a slow-test prerequisite).
    Some Gemma-3 repos nest the decoder dims under ``text_config`` (VLM
    repos); flat text configs (e.g. gemma-3-1b-pt -> Gemma3TextConfig,
    gemma-2-2b -> Gemma2Config) expose them directly. Prefer ``text_config``
    when present so the dims/sliding_window are read from the decoder, never
    the vision tower.
    """
    hf_id = EXTRACTION_MODELS[model_name].hf_model_id
    cfg = AutoConfig.from_pretrained(hf_id)
    text_cfg = getattr(cfg, "text_config", None)
    return text_cfg if text_cfg is not None else cfg


# Largest ``sliding_window`` a fast, realistic single-document corpus can be
# expected to exceed. A window at or below this is "small" — the realistic
# corpus MUST exceed it (else the sliding-vs-full boundary is untested and the
# gate is vacuous, so we FAIL CLOSED). A window above this (gemma-2-2b: 4096)
# is impractical to exceed in a fast natural-language doc, so the window
# assertion — and ONLY that assertion — is explicitly, namedly skipped while
# the rest of the gate still runs (gemma-2's full causal attention is
# exercised by ANY input regardless). Sits comfortably above gemma-3-1b-pt's
# real window (512) and the realistic corpus's max (~1215 tok, empirically
# measured) and well below gemma-2-2b's (4096), so the classification is
# unambiguous; config-derived elsewhere (never a hardcoded per-model window).
_PRACTICAL_WINDOW_CEILING = 2048

# Sentinel: model's window is so large that exceeding it in a fast realistic
# doc is impractical ⇒ the seq>window assertion is LEGITIMATELY skipped (only
# that assertion; the full gate still runs). NOT a "no window" / vacuous pass.
_WINDOW_TOO_LARGE_TO_STRESS = object()


def _sliding_window_guard(model_name: str, reachable_len: int):
    """Fail-closed classification of the model's real ``sliding_window``.

    Returns one of exactly THREE outcomes (never a silent self-disable):

    * **int (small window the corpus exceeds)** — the published window is
      ``<= _PRACTICAL_WINDOW_CEILING`` AND ``reachable_len`` (the longest
      tokenized doc) strictly exceeds it (gemma-3-1b-pt: window=512,
      corpus max ~1215). The caller asserts ``max(lengths) > window`` — a
      GENUINE sliding-vs-full attention check at production scale.
    * **raises AssertionError (small window the corpus does NOT exceed)** —
      the window is small (``<= _PRACTICAL_WINDOW_CEILING``) but the corpus
      fails to exceed it. This is the exact vacuity defect a too-short
      ``_LONG_DOC`` (or a tokenizer-version regression) would reintroduce:
      we FAIL LOUDLY here rather than silently skipping, with a message that
      names the shortfall and tells the maintainer to lengthen ``_LONG_DOC``.
    * **``_WINDOW_TOO_LARGE_TO_STRESS`` (impractically large window)** — the
      window is ``> _PRACTICAL_WINDOW_CEILING`` (gemma-2-2b: 4096), too large
      to exceed in a fast realistic single-document corpus. The caller
      EXPLICITLY (named sentinel, not an implicit ``None``) skips ONLY the
      seq>window assertion; gemma-2-2b's full causal attention is exercised
      by any input regardless, and the rest of the gate (realistic ragged
      input, bf16, VRAM, the three oracles) still runs in full.

    The real window is read from the published HF config (never a hardcoded
    512), so this stays correct if Gemma ever republishes a different window.
    A model with no/zero ``sliding_window`` uses full causal attention at
    every layer ⇒ same legitimate explicit skip as the large-window case.
    """
    sw = getattr(_hf_text_config(model_name), "sliding_window", None)
    if not sw or sw <= 0:
        # No windowed layers at all — full causal everywhere; the seq>window
        # assertion is meaningless (not vacuous, not a defect). Explicit skip.
        return _WINDOW_TOO_LARGE_TO_STRESS
    if sw > _PRACTICAL_WINDOW_CEILING:
        # Impractically large (gemma-2-2b: 4096). Explicit, documented skip of
        # ONLY the window assertion — NOT a silent disable.
        return _WINDOW_TOO_LARGE_TO_STRESS
    # Small window (gemma-3-1b-pt: 512). The corpus MUST exceed it or the
    # sliding-vs-full boundary is never stressed and this gate is vacuous —
    # FAIL CLOSED here, never silently skip.
    assert reachable_len > sw, (
        f"corpus too short to stress sliding_window={sw} for {model_name}: "
        f"max seq {reachable_len} <= {sw}; lengthen _LONG_DOC (the realistic "
        f"corpus must strictly exceed a small sliding_window or the "
        f"sliding-vs-full attention boundary is never exercised and this "
        f"gate certifies nothing about it)"
    )
    return sw


def _full_param_bytes(model_name: str, dtype: t.dtype) -> int:
    """Σ parameter bytes of the FULL (untruncated) model at ``dtype``.

    Config-DERIVED (not a magic constant): computes the full model's total
    parameter count from the published config's real dims, then × the dtype's
    element size. This is the denominator the truncated build's resident /
    peak VRAM is compared against, so the threshold tracks the actual model.

    Gemma ties word embeddings (`tie_word_embeddings=True` for both gemma-2
    and gemma-3 — verified), so the lm_head shares storage with
    ``embed_tokens``: it is counted ONCE (as the embedding) and the freed
    lm_head contributes 0 freed bytes; the embedding stays resident in BOTH
    the truncated and the full model. All truncation savings are therefore
    exactly the dropped ``>layer`` decoder layers — the basis of the
    documented resident/full ≈ 0.43–0.52 (correct) vs ≈1.0 (gc-pinned).

    Per decoder layer (Gemma attention + GeGLU MLP; RMSNorm/bias terms are
    negligible and conservatively omitted ⇒ a SLIGHT under-count of full,
    which only makes the resident<0.85·full gate STRICTER, never looser):
        q: H·(n_h·hd)   k: H·(n_kv·hd)   v: H·(n_kv·hd)   o: (n_h·hd)·H
        gate: H·I        up: H·I          down: I·H
    plus the tied embedding ``vocab·H`` counted once for the whole model.
    """
    c = _hf_text_config(model_name)

    def _g(*names: str) -> int:
        for n in names:
            v = getattr(c, n, None)
            if v is not None:
                return int(v)
        raise AssertionError(f"{model_name}: config missing {names}")

    n_layers = _g("num_hidden_layers")
    hidden = _g("hidden_size")
    inter = _g("intermediate_size")
    vocab = _g("vocab_size")
    n_h = _g("num_attention_heads")
    n_kv = _g("num_key_value_heads", "num_attention_heads")
    hd = _g("head_dim")
    per_layer = (
        hidden * (n_h * hd)      # q_proj
        + hidden * (n_kv * hd)   # k_proj
        + hidden * (n_kv * hd)   # v_proj
        + (n_h * hd) * hidden    # o_proj
        + hidden * inter         # gate_proj
        + hidden * inter         # up_proj
        + inter * hidden         # down_proj
    )
    embed = vocab * hidden  # tied embed == lm_head, counted once
    total_params = n_layers * per_layer + embed
    return total_params * t.empty(0, dtype=dtype).element_size()


def _sparse(sae_acts_list):
    ids, mags = [], []
    for a in sae_acts_list:
        pooled = a.max(dim=0).values  # single-segment max-pool proxy
        nz = (pooled > 0).nonzero().flatten()
        ids.append(nz.to(t.int32).cpu())
        mags.append(pooled[nz].float().cpu())
    return ids, mags


@pytest.mark.slow
@pytest.mark.parametrize("model_dtype", [
    # fp32 proves bit-identity at full precision; bf16 is the PRODUCTION dtype
    # (extraction runs bf16) — the gate must certify native ≡ nnterp there too.
    pytest.param(t.float32, id="fp32"),
    pytest.param(t.bfloat16, id="bf16"),
])
@pytest.mark.parametrize("model_name,release,layer,l0", [
    # gemma-2-2b v1: (residual, 16k) valid all layers; layer=6 fine; l0=None canonical.
    ("gemma-2-2b", SAERelease.gemmascope, 6, None),
    # gemma-3-1b-pt v2: _SELECTED_LAYERS["1b"]={7,13,17,22}; layer=7.
    # _res_combos("1b") publishes {small,medium,big} at W_16K for all selected
    # layers; L0Tier.medium is valid.
    ("gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium),
])
def test_native_matches_nnterp_and_tl_residual(
    model_name, release, layer, l0, model_dtype
):
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    resolved = resolve_sae(model=model_name, release=release,
                           activation_type=ActivationType.residual,
                           layer=layer, width=SAEWidth("16k"), l0=l0)
    model, sae, _ = load_sae_and_model(
        model_name, layer, "cuda", resolved=resolved,
    )
    # Thread the PRODUCTION dtype through EVERY arm so native and nnterp run the
    # SAME raw-HF/SDPA model at the SAME precision (⇒ assert_bit_identical must
    # still hold at bf16); the SAE encodes at that dtype too.
    sae = sae.to(model_dtype)

    # --- Realistic production-shaped input -------------------------------- #
    # Tokenize representative natural-language documents of DIFFERING lengths
    # with the model's OWN gated-HF tokenizer — the exact source production
    # uses (`resolve_tokenizer` → AutoTokenizer.from_pretrained(hf_model_id),
    # steering/models.py). `add_special_tokens=True` + `padding=False` mirrors
    # the production worker `_tokenize_and_head_truncate`
    # (`sae_tokenizer(text, padding=False)`), yielding a RAGGED list of int
    # id lists. `_get_sae_activations_batched` then performs the verbatim
    # production padding contract (steering/extract_features.py:1156-1234):
    #   * lengths = [len(ids)…]; max_seq = max(lengths)
    #   * padded = zeros(B, max_seq, long)  → pad id 0, RIGHT-pad
    #     (padded[i, :len(ids)] = ids)
    #   * per-doc strip  sae_acts[i, :lengths[i]]
    # We deliberately do NOT pre-pad here — passing the ragged id lists
    # straight in exercises that production padding/strip path end to end
    # (the gate goes THROUGH _get_sae_activations_batched, never bypassing it).
    tokenizer = resolve_tokenizer(model_name)
    docs = [
        tokenizer(text, padding=False, add_special_tokens=True)["input_ids"]
        for text in _REALISTIC_DOCS
    ]
    lengths = [len(ids) for ids in docs]
    # Production-mirrored invariants: every doc non-empty, lengths genuinely
    # ragged (so the right-pad path is real, not a no-op).
    assert all(n > 0 for n in lengths), f"empty tokenization: {lengths}"
    assert len(set(lengths)) > 1, (
        f"docs must be ragged to exercise right-padding; got {lengths}"
    )
    # seq>sliding_window guarantee VERIFIED, never assumed, and FAIL-CLOSED:
    # `_sliding_window_guard` reads the model's REAL sliding_window from its
    # published HF config (the exact config the production model loads) and
    # returns exactly one of three outcomes — there is NO silent self-disable:
    #   * small window the corpus exceeds (gemma-3-1b-pt: 512, corpus ~1215)
    #     → returns the int; we assert max(lengths) > it here (GENUINE
    #     sliding-vs-full check; @layer=7 the kept layer_types[:8] =
    #     [sliding×5, full@5, sliding×2] ⇒ mixed sliding+full at prod scale);
    #   * small window the corpus does NOT exceed → the guard itself FAILS
    #     LOUDLY (the exact vacuity defect a too-short _LONG_DOC or a
    #     tokenizer-drift would reintroduce — never a silent skip);
    #   * impractically large window (gemma-2-2b: 4096) → the named sentinel
    #     `_WINDOW_TOO_LARGE_TO_STRESS`, an EXPLICIT documented skip of ONLY
    #     this assertion (gemma-2's full causal attention is exercised by any
    #     input regardless; the rest of the gate still runs in full).
    sliding_window = _sliding_window_guard(model_name, max(lengths))
    if sliding_window is _WINDOW_TOO_LARGE_TO_STRESS:
        # gemma-2-2b (window=4096): exceeding it in a fast realistic doc is
        # impractical; full causal attention is already exercised by every
        # input. Explicitly skip ONLY the seq>window assertion — bf16, the
        # VRAM gate, the production padding path and all three oracles below
        # still run unchanged for gemma-2-2b.
        pass
    else:
        # Small-window model (gemma-3-1b-pt: 512): the guard already proved
        # max(lengths) > window (else it raised). Re-assert at the call site
        # so the genuine sliding-vs-full coverage is explicit in the gate.
        assert max(lengths) > sliding_window, (
            f"{model_name}: longest doc {max(lengths)} tok must exceed "
            f"sliding_window {sliding_window} to stress the sliding-vs-full "
            f"attention boundary at production scale (lengths={lengths})"
        )

    # Immediately offload TL to CPU so the GPU has room for native and nnterp
    # models one at a time. gemma-2-2b fp32 ≈10.5 GB; three simultaneous copies
    # would exceed a 24 GB card. Sequential single-model residency avoids OOM.
    model.to("cpu")
    t.cuda.empty_cache()

    # Pass 1 — native model only: native on CUDA, TL on CPU.
    # _get_sae_activations_batched routes to native_model when provided; it does
    # not touch `model` (TL) on the native path — device comes from sae params.
    # Phase-2 PR-1: build_native_model now REQUIRES `layer` (structural
    # truncation depth); pass the SAME `layer` the SAE/extraction uses so the
    # truncated collector serves resid_post(layer) — the rest of this gate
    # (assert_features_nonempty + assert_bit_identical + assert_feature_
    # equivalent) is the unchanged T9 dual-oracle, unweakened.
    #
    # --- Peak/resident VRAM gate for the gc-pinning regression ------------ #
    # The gc-pinning bug class (the original full ModuleList pinned across the
    # in-function gc.collect, so the >layer decoder layers are NEVER reclaimed)
    # is the one not fast-unit-testable — it only manifests as real GPU
    # residency. We instrument the truncated build:
    #
    #   reset_peak_memory_stats() ; build ; capture max_memory_allocated()
    #                                       and the RESIDENT memory_allocated()
    #
    # Arithmetic (config-derived, NOT a magic constant — `_full_param_bytes`
    # computes Σ params from the loaded model's real config dims; Gemma ties
    # word embeddings, so the freed lm_head shares storage with embed_tokens
    # and contributes 0, while the embedding stays resident in BOTH the
    # truncated and full model — all savings are exactly the dropped >layer
    # decoder layers):
    #   resident/full(correct) :  gemma-2-2b@6 ≈ 0.43 ; gemma-3-1b-pt@7 ≈ 0.52
    #   resident/full(REGRESSED, whole model pinned) ≈ 1.0 (≥, can be >1)
    # A FRACTION=0.85 ceiling sits in the wide gap (≥0.33 margin below the
    # correct residency, ≥0.15 above the regression's lower bound), so it
    # PASSES the correct truncated build and FAILS the gc-pinning regression,
    # with margins >> any allocator/CUDA-context slack (single-digit-% of a
    # multi-GB footprint) ⇒ NON-FLAKY.
    #
    # NOTE — why the RESIDENT (current) allocation, not the peak, is the
    # decisive gc-pinning signal: production `build_native_model` moves the
    # FULL fp32 model to CUDA (`hf_model.to(device)`) BEFORE
    # `NativeCollector.truncated` slices/frees, so `max_memory_allocated`
    # transiently hits ~full-fp32 for BOTH a correct and a regressed build —
    # the high-water mark cannot distinguish them (and we must not change
    # production). What differs is memory STILL allocated AFTER the build
    # returns: a correct build's in-function `del _dropped; gc.collect();
    # empty_cache()` reclaims the >layer layers (resident ≈ 0.43–0.52×full); a
    # gc-pinned build keeps them resident (≈ full). So the resident-allocation
    # assertion is the real regression gate; the peak is asserted only against
    # a generous full-model upper sanity bound (we never allocate a MULTIPLE
    # of the model) and reported for the orchestrator. A structural
    # `len(_base.layers) == layer+1` assert is added belt-and-braces: it
    # distinguishes a refactor that simply forgot to slice even if the VRAM
    # probe were ever noisy.
    full_param_bytes = _full_param_bytes(model_name, model_dtype)
    t.cuda.reset_peak_memory_stats()
    native = build_native_model(model_name, "cuda", model_dtype, layer)
    truncated_peak_bytes = t.cuda.max_memory_allocated()
    truncated_resident_bytes = t.cuda.memory_allocated()
    assert native is not None, f"{model_name}: build_native_model returned None"
    # Structural (bug-distinguishing) invariant: exactly layer+1 decoder layers.
    assert len(native._base.layers) == layer + 1, (
        f"{model_name}: truncated _base.layers has "
        f"{len(native._base.layers)} modules, expected {layer + 1} "
        f"(truncation forgot to slice .layers — gc-pinning bug class)"
    )
    # RESIDENT gate (the real gc-pinning regression catcher): after the build's
    # in-function reclaim, only ~layer+1 layers + the (tied) embedding remain.
    assert truncated_resident_bytes < 0.85 * full_param_bytes, (
        f"{model_name} {model_dtype}: truncated build resident "
        f"{truncated_resident_bytes / 1e9:.3f} GB is NOT materially below "
        f"0.85 × full-model params {full_param_bytes / 1e9:.3f} GB — the "
        f">layer decoder layers were not reclaimed (gc-pinning regressed; "
        f"correct residency is ≈0.43–0.52×full)"
    )
    # Peak upper sanity bound: never allocate a MULTIPLE of the full model.
    # The full-fp32 .to(device) transient is inherent (production loads fp32
    # then converts); allow 1.6× full-fp32 for allocator slack + the brief
    # fp32→model_dtype overlap. (Generous on purpose — the resident gate above
    # is the discriminating one; this only trips on a gross double-allocation.)
    full_fp32_bytes = _full_param_bytes(model_name, t.float32)
    assert truncated_peak_bytes < 1.6 * full_fp32_bytes, (
        f"{model_name} {model_dtype}: truncated-build peak "
        f"{truncated_peak_bytes / 1e9:.3f} GB exceeds 1.6 × full-fp32 "
        f"{full_fp32_bytes / 1e9:.3f} GB — a gross over-allocation "
        f"(multiple model copies resident)"
    )
    nat = _get_sae_activations_batched(
        model, sae, docs, layer=layer, native_model=native,
        hook_name=resolved.hook_name, activation_type=ActivationType.residual)
    del native
    t.cuda.empty_cache()
    gc.collect()

    # Pass 2 — nnterp model only: nnterp on CUDA, TL still on CPU.
    # For gemma-2, build_nnterp_model calls _build_nnterp_gemma2 which offloads
    # TL to CPU itself (already there), then loads a fresh HF copy onto CUDA.
    # For gemma-3, build_nnterp_model calls _build_nnterp_gemma3 which loads a
    # completely separate HF copy; TL stays on CPU because we already moved it.
    # SAME model_dtype as native ⇒ native vs nnterp stays BIT-IDENTICAL at
    # bf16 too (both maskless raw-HF/SDPA at the identical precision).
    nnterp = build_nnterp_model(model, device="cuda", model_dtype=model_dtype) \
        if not model_name.startswith("gemma-3") else \
        build_nnterp_model(model, device="cuda", model_dtype=model_dtype,
                           model_name=model_name)
    nn = _get_sae_activations_batched(
        model, sae, docs, layer=layer, nnterp_model=nnterp,
        hook_name=resolved.hook_name, activation_type=ActivationType.residual)
    del nnterp
    t.cuda.empty_cache()
    # gc.collect() is required to break nnsight reference cycles: without it the
    # nnterp model parameters remain GPU-resident despite `del nnterp`, and the
    # subsequent model.to("cuda") fails with OOM on 24 GB cards.
    gc.collect()

    # Pass 3 — TL only: restore TL to CUDA (nnsight cycles freed by gc.collect
    # above), no native/nnterp on GPU. TL must run on CUDA so that native-vs-TL
    # magnitudes are compared within the SAE's fp32 tolerance — CPU fp32 vs CUDA
    # fp32 diverges by up to ~0.09 abs in SAE feature magnitudes due to differing
    # FMA/softmax orderings, which exceeds the 1% rtol oracle tolerance.
    # TL stays fp32 (TransformerLens loads fp32); the documented asymmetric
    # native-vs-TL tolerance already covers the SDPA-vs-eager envelope, and
    # bf16↔fp32 native-vs-TL drift is bounded by the same mag_atol=0.15 window
    # (count+jaccard remain tight, so a wrong site is still caught).
    model.to("cuda")
    tl = _get_sae_activations_batched(
        model, sae, docs, layer=layer,
        hook_name=resolved.hook_name, activation_type=ActivationType.residual)

    a_ids, a_mags = _sparse(nat)
    n_ids, n_mags = _sparse(nn)
    t_ids, t_mags = _sparse(tl)

    # Non-vacuity guard: assert_bit_identical / assert_feature_equivalent PASS
    # when BOTH sides are empty/all-zero, so a regression making both backends
    # emit no features would pass this gate vacuously. Require real features on
    # every side BEFORE the oracle so the comparison certifies something.
    assert_features_nonempty(a_ids, a_mags, label="native")
    assert_features_nonempty(n_ids, n_mags, label="nnterp")
    assert_features_nonempty(t_ids, t_mags, label="TL")

    # native vs nnterp: both run the same raw-HF SDPA model → must be BIT-IDENTICAL.
    # Any divergence here is a real capture-site, reshape, or pooling bug.
    assert_bit_identical(a_ids, a_mags, n_ids, n_mags, label="native-vs-nnterp")

    # native vs TL: count+jaccard stay tight (structural/site bugs caught here);
    # magnitude uses a documented looser tolerance.
    #
    # Root cause: TransformerLens forces eager attention; native/nnterp use SDPA.
    # Gemma-2's attn_logit_softcapping=50.0 makes eager ≠ SDPA in fp32 — a
    # pre-existing TL-vs-HF property, NOT a native bug.
    # gemma-3-1b-pt has no attn_logit_softcap → its residual passes even tight tol.
    #
    # Measured fp32 envelope (2026-05-17, layer=6 gemma-2-2b + layer=7
    # gemma-3-1b-pt, GemmaScope 16k, residual, fp32, on the prior hardcoded toy
    # id batch):
    #   gemma-2-2b  native-vs-TL: max rel err = 0.01474, max abs = 0.08516
    #   gemma-3-1b-pt native-vs-TL: max rel err = 1.12e-5, max abs = 4.88e-4
    #
    # Chosen fp32 tol: mag_rtol=0.04 (2.7× safety on 1.47% max); mag_atol=0.15
    # (covers near-zero features where the 0.085 abs error would blow up rel
    # err). The fp32 native-vs-TL path stays at EXACTLY this tight 0.04/0.15
    # (it is NOT loosened — pass NO mag override for fp32) and is fully matched
    # (matched-fp32 native-vs-TL = 0.00397, well inside it).
    #
    # --- Magnitude envelope: fp32 tight, bf16 data-grounded bf16-ULP-aware --- #
    # The hardened gate's bf16 param surfaced a native-vs-TL MAGNITUDE
    # mismatch the old toy input never fired. Root-caused (2026-05-18, GPU):
    #   * native ≡ nnterp BIT-IDENTICAL at bf16 (assert_bit_identical above
    #     PASSED at bf16; mag_max_abs == 0) — native IS the production backend
    #     and is provably correct; nnterp-vs-TL reproduces the SAME mismatch,
    #     so it is NOT native-specific and NOT a dtype-mishandling bug;
    #   * matched-fp32 native-vs-TL = 0.00397 (no bug at equal precision);
    #   * the divergence is pure bf16 ULP rounding of the HUGE GemmaScope-2
    #     residual features the realistic 1215-token corpus fires (the top
    #     offender, feat 118: native_bf16≈1312 vs TL_fp32≈1328, |Δ|≈16 ≈ 1.5–2
    #     bf16 ULP at magnitude ~1300 — bf16's relative ULP is 2⁻⁷≈0.0078).
    # Because the absolute error is ∝ feature magnitude (∝ bf16 ULP), a
    # RELATIVE tolerance is the physically-correct model. The JumpReLU
    # discontinuity is secondary — only the 1–2 near-threshold features flip,
    # which is the structural jaccard 0.9879 already absorbed by the 0.98
    # bf16 structural gate (mag tol does NOT need to cover a feature-set flip).
    #
    # WHY a higher mag_rtol AND a small mag_atol (not a blanket ≥16 atol):
    # ``t.allclose`` is the ELEMENT-WISE predicate ∀i |aᵢ-bᵢ| ≤ atol+rtol·|bᵢ|,
    # NOT scalar max_abs ≤ atol+rtol·max|b|. Feat 118 ALONE satisfies its bound
    # at the fp32 0.04/0.15 (16 ≤ 0.15+0.04·1328 = 53) — it is the reported
    # global max_abs, but it is NOT what fails allclose. allclose fails because
    # a SMALL-|b| common feature has a bf16-ULP |Δ| exceeding ITS tiny
    # 0.15+0.04·|b| bound (small |b| ⇒ negligible rtol term ⇒ the absolute
    # bf16-ULP/near-threshold floor dominates). So the cure is BOTH: raise
    # mag_rtol so the relative term carries the HUGE features, AND a small,
    # data-grounded mag_atol covering the SMALL-feature absolute floor — but a
    # blanket atol ≥ max_abs (≥16) would make the gate VACUOUS (every feature
    # passes on atol alone, the relative term doing nothing). The atol stays
    # WELL below max_abs so the relative term must do the work for the huge
    # features and a real bug on them is still caught.
    #
    # Measured bf16 native-vs-TL magnitude envelope (gemma-3-1b-pt@7,
    # GemmaScope-2 16k, residual, realistic 1215-tok ragged corpus, 4090,
    # 2026-05-18; instrumented `[envelope]` line below prints mag_max_abs /
    # mag_max_rel EVERY -m slow run so the chosen tol is grounded by a printed
    # measurement, not a one-time claim):
    #   seg 0, 284 common features: max_abs = 16.0, p99(abs) = 4.0,
    #   mean(abs) = 0.3639, max rel err ≈ 16/1328 ≈ 0.0121.
    #
    # --- Structural (jaccard) envelope: fp32 tight, bf16 data-grounded ----- #
    # native ≡ nnterp is the STRONG anchor (assert_bit_identical above — same
    # raw-HF/SDPA model, byte-identical, and it PASSED at bf16 on the 4090
    # 2026-05-18 on the 1215-token realistic batch ⇒ native is correct). The
    # native-vs-TL jaccard is a DELIBERATELY WEAKER oracle: TL forces eager
    # attention and is fp32-only, so at the PRODUCTION bf16 dtype native (bf16)
    # vs TL (fp32) crosses BOTH an attention-kernel and a precision boundary,
    # flipping a few near-threshold SAE features.
    #
    # Measured native-vs-TL structural envelope (gemma-3-1b-pt@7, GemmaScope-2
    # 16k, residual, on the realistic 1215-token ragged corpus; instrumented
    # `[envelope]` line below prints it EVERY -m slow run, so 0.98 is grounded
    # by a printed measurement, not a one-time claim):
    #   fp32 (2026-05-17/18): jaccard = 1.0   (no kernel/precision split for
    #       gemma-3 — no attn_logit_softcap; passes the tight 0.99 default)
    #   bf16 (4090, 2026-05-18): min_jaccard = 0.9895  (the ONLY hardened-gate
    #       failure — 0.9895 < the fp32-sized 0.99 module constant)
    # Chosen bf16 jaccard_threshold = 0.98: ≈0.0095 of safety below the worst
    # observed 0.9895 (mirrors the mag tol's "2.7× safety" data-grounded style),
    # yet still EXTREMELY tight structurally — a wrong capture/layer site gives
    # a grossly different feature SET, jaccard ≪ 0.5, so a real site bug is
    # still caught. fp32 keeps the tight 0.99 default (do NOT pass the override
    # for fp32). count stays at the tight default (rdiff≤0.01) for BOTH dtypes:
    # the failure was jaccard-ONLY; count is NOT loosened (no data shows it
    # needs to be). The bf16 MAGNITUDE tolerance IS data-grounded-loosened too
    # (see the bf16-ULP root cause above and the _BF16_NATIVE_VS_TL_MAG_*
    # constants below) — the SAME asymmetric-oracle relaxation pattern: only
    # the WEAKER bf16 native-vs-TL oracle (kernel + precision boundary) gets
    # the data-grounded looser-but-still-tight gate; fp32 native-vs-TL stays
    # at the tight 0.04/0.15 and the strong native≡nnterp anchor stays
    # bit-identical (assert_bit_identical, no tolerance) at bf16 too.
    #
    # Permanent instrumentation: log the OBSERVED native-vs-TL structural AND
    # magnitude envelope for THIS (model, dtype) param BEFORE asserting, so
    # every slow run prints the numbers that ground the chosen thresholds (not
    # a stale comment). Uses conftest_parity.feature_equivalence_metrics +
    # feature_magnitude_metrics — the SAME per-seg jaccard/count and the SAME
    # per-seg common-feature max_abs/max_rel arithmetic the asserts use (single
    # source of truth: the metrics siblings and the assert call the same
    # _seg_* kernels, so the printed envelope cannot drift from the gate).
    #
    # Single-source the bf16 gates: each of these constants is referenced by
    # BOTH the `[envelope]` print's displayed gate AND the override kwargs
    # below, so the printed "active gate" can never disagree with the gate
    # actually enforced. The prose envelope comments above keep stating the
    # values as documentation; the live values live here.
    _BF16_NATIVE_VS_TL_JACCARD = 0.98  # data-grounded: ~0.0095 below worst-observed 0.9895 (gemma-3-1b-pt@7, 4090, 2026-05-18); structurally still tight (wrong site -> jaccard <<0.5)
    # The fp32 (TIGHT, unchanged) and bf16 (data-grounded, bf16-ULP root-
    # caused) native-vs-TL magnitude tolerances are the SINGLE-SOURCE named
    # conftest_parity constants (BF16/FP32_NATIVE_VS_TL_MAG_RTOL/ATOL) — see
    # their full data-grounded derivation comment there. Imported (not
    # re-literal'd) so this gate and its non-vacuous unit tests cannot drift.
    min_jaccard, max_count_rdiff = feature_equivalence_metrics(a_ids, t_ids)
    mag_max_abs, mag_max_rel = feature_magnitude_metrics(
        a_ids, a_mags, t_ids, t_mags
    )
    _is_bf16 = model_dtype == t.bfloat16
    _mag_rtol = (
        BF16_NATIVE_VS_TL_MAG_RTOL if _is_bf16 else FP32_NATIVE_VS_TL_MAG_RTOL
    )
    _mag_atol = (
        BF16_NATIVE_VS_TL_MAG_ATOL if _is_bf16 else FP32_NATIVE_VS_TL_MAG_ATOL
    )
    print(
        f"[envelope] {model_name} {model_dtype} native-vs-TL: "
        f"min_jaccard={min_jaccard:.6f} max_count_rdiff={max_count_rdiff:.6f} "
        f"mag_max_abs={mag_max_abs:.6g} mag_max_rel={mag_max_rel:.6g} "
        f"(jaccard gate = "
        f"{_BF16_NATIVE_VS_TL_JACCARD if _is_bf16 else 'default 0.99'}, "
        f"count gate = default 0.01, mag gate = "
        f"rtol {_mag_rtol} / atol {_mag_atol})"
    )

    # bf16: data-grounded looser structural jaccard (0.98) AND data-grounded
    # bf16-ULP-aware magnitude rtol/atol (0.05 / 5.0) — the SAME asymmetric-
    # oracle relaxation pattern, applied ONLY to the weaker bf16 native-vs-TL
    # oracle. fp32: NO structural override → the tight 0.99 jaccard / 0.01
    # count module defaults, AND the tight 0.04 / 0.15 magnitude tol (the
    # SAME tight fp32 tol as before — FP32_NATIVE_VS_TL_MAG_*; gemma-3 fp32
    # measured jaccard 1.0, mag matched to 0.00397). count stays tight for
    # bf16 too (failure was jaccard + bf16-ULP magnitude only). The mag tol
    # is ALWAYS passed (it was before too: fp32 used mag_rtol=0.04/
    # mag_atol=0.15) — only its VALUE is dtype-conditional now; the structural
    # jaccard override remains bf16-only.
    structural_kwargs = (
        {"jaccard_threshold": _BF16_NATIVE_VS_TL_JACCARD} if _is_bf16 else {}
    )
    assert_feature_equivalent(
        a_ids, a_mags, t_ids, t_mags, label="native-vs-TL",
        mag_rtol=_mag_rtol, mag_atol=_mag_atol,
        **structural_kwargs,
    )


@pytest.mark.slow
@pytest.mark.parametrize("model_name,release,layer,l0", [
    # gemma-2-2b v1: (mlp, 16k) valid for all layers 0-25 (_GEMMA2_2B_VALID_COMBOS).
    # layer=6 is well within range; l0=None canonical.
    ("gemma-2-2b", SAERelease.gemmascope, 6, None),
    # gemma-3-1b-pt v2: _SELECTED_LAYERS["1b"] = {7,13,17,22}; layer=7.
    # _mlp_combos("1b") publishes {small,medium,big} at W_16K for all selected
    # layers; L0Tier.medium is valid.
    ("gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium),
])
def test_native_matches_nnterp_and_tl_mlp(model_name, release, layer, l0):
    """Native truncated collector is RESIDUAL-ONLY (Phase-2 PR-1).

    Spec §3.2; project decision 2026-05-16. mlp is non-residual, so the
    original Phase-1 "native feature-equivalent to nnterp AND TL for mlp"
    intent is infeasible by design: capture(mlp) raises a typed residual-only
    ExtractionError('native (Phase-2 truncated collector) is residual-only;
    mlp/attention not supported — use backend=auto/nnterp') and the dispatch
    fallback chain degrades exactly as for any other ExtractionError. This is
    NOT a known divergence expected to fail — residual-only-raises is the
    intended, asserted, correct behavior at the native boundary. The heavy
    3-pass nnterp/TL equivalence body is removed: it could never pass for a
    residual-only collector and would only exercise the fallback backends.
    mlp real-model parity is out of native scope (project decision
    2026-05-16), not silently broken.
    """
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    resolved = resolve_sae(model=model_name, release=release,
                           activation_type=ActivationType.mlp,
                           layer=layer, width=SAEWidth("16k"), l0=l0)
    # build_native_model now REQUIRES `layer` (structural truncation depth).
    # The truncated collector is residual-only: a non-residual capture must
    # raise the documented typed ExtractionError at the native boundary (the
    # NEW reality; this is what the xfail expects).
    native = build_native_model(model_name, "cuda", t.float32, layer)
    assert native is not None
    with pytest.raises(ExtractionError, match="residual-only"):
        native.capture(
            t.tensor([[2, 17, 99, 8, 41, 5, 3]], device="cuda"),
            layer,
            hook_name=resolved.hook_name,
            activation_type=ActivationType.mlp,
        )


@pytest.mark.slow
@pytest.mark.parametrize("model_name,release,layer,l0", [
    # gemma-3-1b-pt only: gemma-scope-2 attention uses post-projection
    # hook_attn_out, which nnterp DOES support. v1 (gemma-2-2b) attention uses
    # pre-projection hook_z (nnterp cannot capture it) — tested separately in
    # test_native_matches_tl_v1_hook_z below.
    # _SELECTED_LAYERS["1b"] = {7,13,17,22}; layer=7.
    # _att_combos("1b") publishes {small,medium,big} at W_16K; L0Tier.medium valid.
    ("gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium),
])
def test_native_matches_nnterp_and_tl_v2_attention(model_name, release, layer, l0):
    """Native truncated collector is RESIDUAL-ONLY (Phase-2 PR-1).

    Spec §3.2; project decision 2026-05-16. v2 attention is non-residual, so
    capture(attention) raises a typed residual-only ExtractionError('native
    (Phase-2 truncated collector) is residual-only; mlp/attention not
    supported — use backend=auto/nnterp'); the dispatch fallback chain
    degrades exactly as for any other ExtractionError. This is NOT a known
    divergence expected to fail — residual-only-raises is the intended,
    asserted, correct behavior at the native boundary. (A SEPARATE
    pre-existing downstream bug in the SHARED _get_sae_activations_batched
    4-D hook_z→SAE flatten for reshape_activations='none' gemma-scope-2
    attention SAEs — affecting nnterp/TL too, not native-specific — remains
    tracked for a future phase, but is moot here: the residual-only collector
    errors out first.) gemma-scope-2 attention is out of native scope
    (project decision 2026-05-16).
    """
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    resolved = resolve_sae(model=model_name, release=release,
                           activation_type=ActivationType.attention,
                           layer=layer, width=SAEWidth("16k"), l0=l0)
    # build_native_model now REQUIRES `layer`. The truncated collector is
    # residual-only: a non-residual (attention) capture must raise the
    # documented typed ExtractionError at the native boundary (the NEW
    # reality the xfail expects).
    native = build_native_model(model_name, "cuda", t.float32, layer)
    assert native is not None
    with pytest.raises(ExtractionError, match="residual-only"):
        native.capture(
            t.tensor([[2, 17, 99, 8, 41, 5, 3]], device="cuda"),
            layer,
            hook_name=resolved.hook_name,
            activation_type=ActivationType.attention,
        )


@pytest.mark.slow
@pytest.mark.parametrize("model_name,release,layer,l0", [
    # gemma-2-2b only: GemmaScope v1 attention = pre-projection attn.hook_z.
    # (ActivationType.attention, SAEWidth.W_16K) valid for all layers 0-25
    # per _GEMMA2_2B_VALID_COMBOS (None means all layers); l0=None canonical.
    ("gemma-2-2b", SAERelease.gemmascope, 6, None),
])
def test_native_matches_tl_v1_hook_z(model_name, release, layer, l0):
    """Native truncated collector is RESIDUAL-ONLY (Phase-2 PR-1).

    Spec §3.2; project decision 2026-05-16. v1 pre-projection hook_z is
    attention = non-residual, so capture(attention) raises a typed
    residual-only ExtractionError('native (Phase-2 truncated collector) is
    residual-only; mlp/attention not supported — use backend=auto/nnterp');
    the dispatch fallback chain degrades exactly as for any other
    ExtractionError. This is NOT a known divergence expected to fail —
    residual-only-raises is the intended, asserted, correct behavior at the
    native boundary. (The Phase-1 small native-vs-TL hook_z magnitude drift
    from TL-eager-vs-HF-SDPA + Gemma-2 attn_logit_softcapping=50.0 is moot —
    the residual-only collector errors out first.) v1 pre-projection hook_z
    attention is out of native scope (project decision 2026-05-16).
    """
    if not t.cuda.is_available():
        pytest.skip("CUDA required")
    resolved = resolve_sae(model=model_name, release=release,
                           activation_type=ActivationType.attention,
                           layer=layer, width=SAEWidth("16k"), l0=l0)
    # build_native_model now REQUIRES `layer`. The truncated collector is
    # residual-only: a non-residual (v1 pre-projection hook_z attention)
    # capture must raise the documented typed ExtractionError at the native
    # boundary (the NEW reality the xfail expects).
    native = build_native_model(model_name, "cuda", t.float32, layer)
    assert native is not None
    with pytest.raises(ExtractionError, match="residual-only"):
        native.capture(
            t.tensor([[2, 17, 99, 8, 41, 5, 3]], device="cuda"),
            layer,
            hook_name=resolved.hook_name,
            activation_type=ActivationType.attention,
        )


# =========================================================================== #
# S1 Task 7 — gated torch.compile EMPIRICAL real-model correctness gate.       #
#                                                                             #
# These @pytest.mark.slow arms are THE empirical proof of the whole S1        #
# gated-torch.compile PR.  They drive the REAL production gate                #
# (`steering.extract_features._native_compile_gate`, force=False — the exact  #
# path extraction runs) and the REAL collector arm                            #
# (`NativeCollector.enable_compiled` → `torch.compile(self._base,             #
# backend="inductor", dynamic=True)` + `emulate_precision_casts=True`) on the #
# REAL gemma SAEs/tokenizer and the SAME verbatim `_REALISTIC_DOCS` ragged    #
# corpus the headline residual gate uses.  Nothing here re-implements the     #
# compile, the oracle, or the corpus — it certifies the production code.      #
#                                                                             #
# What each arm proves:                                                       #
#   * `test_compile_gate_oracle_safe[...]` — for {gemma-2-2b@6,               #
#     gemma-3-1b-pt@7}×{fp32,bf16}: the production gate's decision is SAFE     #
#     (PASS ⇒ compiled-active AND independently compiled≈eager within the     #
#     project dtype-aware oracle; or FAIL ⇒ stayed eager, byte-identical to   #
#     the trusted Phase-1 anchor).  HONEST, NOT force-passed: the thin-margin #
#     gemma-3-1b-pt bf16 cell is asserted only to be SAFE; whether compile    #
#     ACTIVATES for it is measured and logged (the T8 honest-reporting        #
#     headline), never assumed.  A `[compile-envelope]` line is printed EVERY #
#     run with the real measured jaccard/count/mag.                           #
#   * `test_compile_gate_teeth_default_inductor_fails` — the teeth regression #
#     guard: gemma-3-1b-pt@7 bf16 with `emulate_precision_casts=False` (the   #
#     known-corrupting default-Inductor regime) — the production gate MUST    #
#     FAIL (stay eager; force=True ⇒ CompileGateError).  Proves the self-     #
#     check genuinely detects the corrupting config; if this ever stops       #
#     failing the gate is vacuous.                                            #
#   * `test_compile_gate_dynamic_shape_recompile_hole` — Minor-3 closure:     #
#     after the gate activates compiled (gemma-2-2b bf16), push ≥2 distinct   #
#     (batch, seq) ragged shapes through the SAME armed+activated collector   #
#     and assert EACH compiled output is feature-equivalent to eager for      #
#     that shape — the single `dynamic=True` graph stays correct across       #
#     shapes with NO un-probed recompile.                                     #
#   * `test_compile_gate_eager_anchor_unchanged` — under all of the above the #
#     eager path is STILL byte-identical (`torch.equal`) to                   #
#     `hook_reference_residual` (PR-1 is the trusted anchor).                  #
#                                                                             #
# Memory: 24 GB 4090.  Mirrors the headline gate's sequential single-model    #
# residency and additionally uses a SMALLER ragged sub-corpus for the heavy   #
# gemma-2-2b fp32 path (`_compile_gate_docs`), since both an eager AND a      #
# compiled inductor graph of gemma-2-2b are resident simultaneously here      #
# (the gate probes eager THEN compiled on the same collector); gemma-3-1b-pt  #
# and all bf16 paths run the full `_REALISTIC_DOCS` corpus.  torch.compile    #
# caches are force-disabled and `torch._dynamo.reset()` is called per         #
# compiled arm so the teeth-guard emulate=False trace can NEVER reuse an      #
# emulate=True cached graph (and vice-versa) — without this isolation the     #
# teeth guard could silently pass on a stale correct graph and the whole      #
# regression proof would be vacuous.                                          #
# =========================================================================== #

# gemma-2-2b fp32 carries BOTH a resident eager copy AND a resident compiled
# inductor graph of the truncated @6 model on the SAME collector (the gate
# probes eager then compiled), plus the SAE — on a 24 GB card the full
# ~1215-token `_LONG_DOC` would OOM that one cell.  Drop `_LONG_DOC` for the
# heavy fp32 path ONLY; the batch stays genuinely RAGGED (short/med/para have
# distinct lengths so the right-pad + per-doc-strip contract is still
# exercised) and the SAE still fires real, non-vacuous features.  bf16 and
# gemma-3-1b-pt run the full corpus (incl. `_LONG_DOC`) — the >sliding_window
# long-context path is preserved for every cell that fits.
_COMPILE_GATE_HEAVY_DOCS = [_SHORT_DOC, _MED_DOC, _PARA_DOC]


def _compile_gate_docs(model_name: str, model_dtype: t.dtype) -> list[str]:
    """The ragged document corpus for a compile-gate cell.

    Full `_REALISTIC_DOCS` (incl. the >sliding_window `_LONG_DOC`) for every
    cell EXCEPT the memory-heaviest gemma-2-2b fp32 (eager + compiled inductor
    graph of the 2B model both resident on one collector + SAE on 24 GB) where
    `_LONG_DOC` is dropped — still genuinely ragged, still non-vacuous, exactly
    the documented memory-aware shape mitigation the headline gate/benchmark
    use, applied to ONLY the cell that needs it.
    """
    if model_name == "gemma-2-2b" and model_dtype == t.float32:
        return _COMPILE_GATE_HEAVY_DOCS
    return _REALISTIC_DOCS


def _tokenize_ragged(model_name: str, docs: list[str]) -> list[list[int]]:
    """Tokenize `docs` with the model's OWN gated-HF tokenizer, verbatim the
    way the headline residual gate (and production `_tokenize_and_head_
    truncate`) does: `add_special_tokens=True`, `padding=False` → a RAGGED
    list of id lists.  Asserts non-empty + genuinely ragged (so the right-pad
    path the gate exercises is real, not a no-op)."""
    tokenizer = resolve_tokenizer(model_name)
    out = [
        tokenizer(text, padding=False, add_special_tokens=True)["input_ids"]
        for text in docs
    ]
    lengths = [len(ids) for ids in out]
    assert all(n > 0 for n in lengths), f"empty tokenization: {lengths}"
    assert len(set(lengths)) > 1, (
        f"docs must be ragged to exercise right-padding; got {lengths}"
    )
    return out


def _right_padded_probe(docs: list[list[int]], device: str) -> t.Tensor:
    """The verbatim production right-pad contract `_get_sae_activations_
    batched` applies (steering/extract_features.py): `padded =
    zeros(B, max_seq, long)`; `padded[i, :len(ids)] = ids` (pad id 0,
    RIGHT-pad).  This is EXACTLY the 2-D `probe_input_ids` shape
    `_native_compile_gate` consumes (it derives true lengths via
    `(row != 0).sum()` and strips per-doc) — so the gate runs through the real
    production padding/strip path here, never a bypass."""
    max_seq = max(len(ids) for ids in docs)
    padded = t.zeros(len(docs), max_seq, dtype=t.long, device=device)
    for i, ids in enumerate(docs):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
    return padded


def _isolate_compile_caches() -> None:
    """Force-disable EVERY torch.compile/inductor persistent cache and reset
    dynamo so a compiled trace in one arm can NEVER be reused by another.

    Critical for the teeth guard: the oracle-safe arm compiles gemma-3-1b-pt@7
    bf16 with `emulate_precision_casts=True`; the teeth guard recompiles the
    SAME architecture with `emulate_precision_casts=False`.  If the persistent
    FX-graph cache served the emulate=True graph to the emulate=False trace the
    teeth guard would silently PASS on a stale-correct graph and the entire
    regression proof would be vacuous.  We also use a FRESH collector per
    compiled arm (distinct `_base` ⇒ distinct dynamo compile id) — this is the
    cache belt to that suspenders.  Test-side only (no production change)."""
    _inductor_config.force_disable_caches = True
    _inductor_config.fx_graph_cache = False
    torch._dynamo.reset()


def _free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _load_sae_only(model_name, layer, l0, release):
    """Resolve+load the real GemmaScope SAE for a compile-gate cell and return
    ONLY the cuda SAE.

    `load_sae_and_model` also loads the full TransformerLens model onto cuda
    (~10 GB fp32 for gemma-2-2b). The compile-gate path NEVER uses TL (it runs
    the native truncated collector + the SAE), so — mirroring the headline
    residual gate's `model.to("cpu"); empty_cache()` memory discipline — the
    unused TL model is offloaded to CPU and freed IMMEDIATELY here, before any
    native/compiled graph is built, so the heavy gemma-2-2b fp32 cell does not
    co-resident TL(~10 GB)+native+compiled on the 24 GB 4090."""
    resolved = resolve_sae(model=model_name, release=release,
                           activation_type=ActivationType.residual,
                           layer=layer, width=SAEWidth("16k"), l0=l0)
    tl_model, sae, _ = load_sae_and_model(
        model_name, layer, "cuda", resolved=resolved,
    )
    tl_model.to("cpu")
    _free(tl_model)
    return resolved, sae


def _full_hf_reference_model(model_name: str, model_dtype: t.dtype):
    """A FRESH, FULL (untruncated) HF causal-LM, loaded byte-for-byte the way
    production `build_native_model` loads it — the correct input for
    `hook_reference_residual`.

    `hook_reference_residual` is the verbatim Phase-1 mechanism: it does
    `hf_model.model.layers[layer]` + a full `hf_model(input_ids=...)` forward
    with a forward hook (so it needs the FULL `Gemma{2,3}ForCausalLM`, whose
    `.model` is the inner decoder — NOT `NativeCollector._base`, which already
    IS that inner decoder and has no `.model`). resid_post(layer) read off a
    forward hook on `layers[layer]` of the FULL model is, by causal-attention
    invariance, identical to the structurally-truncated collector's
    `capture()` (layers > `layer` cannot influence the residual at `layer`) —
    so this full-model hook is exactly the trusted Phase-1 anchor the
    truncated collector's correctness is defined against. Loaded with the
    SAME loader-by-family / `sdpa` / fp32→`model_dtype` / `.eval()` /
    `cuda` recipe as `build_native_model` so the anchor is the production
    eager numerics (no kernel/precision skew vs the collector)."""
    hf_id = EXTRACTION_MODELS[model_name].hf_model_id
    loader = (
        Gemma3ForCausalLM if model_name.startswith("gemma-3")
        else AutoModelForCausalLM
    )
    # widen the HF union so .to(str|device) / overload-wrapped calls
    # type-check — the SAME `hf_model: Any = _loaded` idiom production
    # `build_native_model` uses for this exact HF typing limitation.
    hf: Any = loader.from_pretrained(
        hf_id, torch_dtype=t.float32,
        attn_implementation=_NATIVE_ATTN_IMPLEMENTATION,
    )
    hf = hf.to("cuda").eval()
    if model_dtype != t.float32:
        hf = hf.to(model_dtype)
    return hf


# --------------------------------------------------------------------------- #
# Arm 1 — dynamic+emulate oracle-safe, both cells × {fp32, bf16}.             #
#                                                                             #
# Runs the REAL production gate (`_native_compile_gate`, force=False) on the  #
# realistic ragged corpus, then INDEPENDENTLY re-asserts the gate's decision  #
# is SAFE and re-checks compiled-vs-eager via the project oracle.  The        #
# assertion certifies SAFETY, not a forced speedup:                           #
#   * gate PASS  ⇒ collector ends `_use_compiled True` AND an independent     #
#     `assert_feature_equivalent` (dtype-aware project thresholds) confirms   #
#     the activated compiled path really is feature-equivalent to eager;      #
#   * gate FAIL  ⇒ collector stays `_use_compiled False` AND a re-capture     #
#     with compiled inactive is identical to the eager probe (the rejected    #
#     graph is NOT leaking into the 'safe' fallback).                         #
# Either way the cell is GREEN because the production gate behaved SAFELY.    #
# Only the cells that genuinely pass on this 4090 hard-assert PASS; the       #
# thin-margin gemma-3-1b-pt bf16 cell is asserted SAFE and its real measured  #
# jaccard + activation decision are logged as the T8 honest-reporting input.  #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("model_dtype", [
    pytest.param(t.float32, id="fp32"),
    pytest.param(t.bfloat16, id="bf16"),
])
@pytest.mark.parametrize("model_name,release,layer,l0", [
    ("gemma-2-2b", SAERelease.gemmascope, 6, None),
    ("gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium),
])
def test_compile_gate_oracle_safe(model_name, release, layer, l0, model_dtype):
    """The production compile gate's decision is SAFE on real hardware for
    this (model, dtype) cell under the ideal-math-relative semantics.

    The production gate decides via ``compiled_passes_ideal_math_gate``:
    PASS iff j_compiled_vs_ref >= j_baseline_vs_ref - slack, where the
    reference is fp32-eager (the ideal-math approximation). This test
    INDEPENDENTLY rebuilds an fp32-eager reference, captures bf16-eager
    baseline and compiled features, applies the EXACT same predicate, and
    asserts:
      * gate ACTIVATED compiled (_use_compiled True) — the production
        decision; AND
      * INDEPENDENTLY computed ``passed`` from the predicate is True (the
        gate's decision cannot drift from what an independent re-check
        confirms).
    A `[compile-envelope]` line is printed EVERY run carrying the four
    measurements.
    """
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    _isolate_compile_caches()

    _, sae = _load_sae_only(model_name, layer, l0, release)
    sae = sae.to(model_dtype)

    docs = _tokenize_ragged(
        model_name, _compile_gate_docs(model_name, model_dtype)
    )
    probe = _right_padded_probe(docs, "cuda")
    lengths = [int((row != 0).sum().item()) for row in probe]

    native = build_native_model(model_name, "cuda", model_dtype, layer)
    assert native is not None, f"{model_name}: build_native_model returned None"

    # THE production path: real gate (exactly what extraction runs). The new
    # ideal-math-relative gate builds its own fp32-eager reference internally,
    # captures bf16-eager baseline, captures compiled, applies the predicate,
    # and either ACTIVATES compiled or raises CompileGateError. We do NOT
    # pre-seed anything; the gate owns the decision.
    _native_compile_gate(
        native, sae, probe_input_ids=probe, model_dtype=model_dtype,
    )
    assert native._use_compiled, (
        f"{model_name}@{layer} {model_dtype}: the production gate must "
        f"ACTIVATE on this cell. Under the no-opt-out contract the only "
        f"other outcome would have been CompileGateError; the empirical "
        f"evidence (4090, FineWeb-Edu shard 00388, ideal-math reference) "
        f"says emF dyn beats bf16-eager on this cell."
    )
    gate_activated = native._use_compiled

    # --- INDEPENDENT RE-CHECK ------------------------------------------ #
    # Rebuild the fp32-eager reference (the ideal-math approximation), the
    # bf16-eager baseline, and the compiled features — all on the SAME probe
    # the gate consumed — and apply the SAME predicate the production gate
    # uses. This is the independent measurement that pins the gate's
    # decision (the printed numbers cannot drift from what this re-check
    # verifies).
    hook_name = f"blocks.{native.truncated_layer}.hook_resid_post"

    # baseline (bf16-eager): force eager on the production collector.
    native.set_compiled_active(False)
    baseline_acts = native.capture(
        probe, native.truncated_layer, hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    baseline_ids, baseline_mags = _gate_sparse_features(
        sae, baseline_acts, lengths
    )

    # compiled: force compiled on the production collector, then restore the
    # gate's decision after.
    native.set_compiled_active(True)
    try:
        compiled_acts = native.capture(
            probe, native.truncated_layer, hook_name=hook_name,
            activation_type=ActivationType.residual,
        )
    finally:
        native.set_compiled_active(gate_activated)
    compiled_ids, compiled_mags = _gate_sparse_features(
        sae, compiled_acts, lengths
    )

    # fp32-eager reference: fresh fp32 native + fp32 SAE. Mirrors what the
    # production gate does internally, but built explicitly here so this
    # test's predicate has independent inputs.
    sae.to(t.float32)
    ref_native = build_native_model(model_name, "cuda", t.float32, layer)
    assert ref_native is not None
    ref_acts = ref_native.capture(
        probe, ref_native.truncated_layer, hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    ref_ids, ref_mags = _gate_sparse_features(sae, ref_acts, lengths)
    sae.to(model_dtype)  # restore production dtype
    _free(ref_native, ref_acts)

    # Non-vacuity: every side must fire real features.
    assert_features_nonempty(ref_ids, ref_mags, label="ref")
    assert_features_nonempty(baseline_ids, baseline_mags, label="baseline")
    assert_features_nonempty(compiled_ids, compiled_mags, label="compiled")

    # The EXACT predicate the production gate enforces.
    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=ref_ids,
        baseline_ids=baseline_ids,
        compiled_ids=compiled_ids,
        slack=BASELINE_RELATIVE_SLACK,
    )
    dtype_name = "bf16" if model_dtype == t.bfloat16 else "fp32"
    print(
        f"[compile-envelope] {model_name}@{layer} {dtype_name} "
        f"IDEAL-MATH: "
        f"j_baseline_vs_ref={m['j_baseline_vs_ref']:.6f} "
        f"j_compiled_vs_ref={m['j_compiled_vs_ref']:.6f} "
        f"slack={m['slack']:.6f} margin={m['margin']:.6f} "
        f"-> gate_decision={'COMPILED' if gate_activated else 'EAGER'} "
        f"(independent_predicate_passed={passed})"
    )

    # CORRECTNESS assertion: the gate ACTIVATED, and the INDEPENDENT
    # re-check of the predicate confirms it. The production gate's printed
    # numbers cannot drift from the predicate this test verifies — same
    # inputs, same function, same decision.
    assert passed, (
        f"{model_name}@{layer} {dtype_name}: the production gate ACTIVATED "
        f"compiled but an INDEPENDENT re-check via "
        f"compiled_passes_ideal_math_gate FAILED "
        f"(j_baseline_vs_ref={m['j_baseline_vs_ref']:.6f}, "
        f"j_compiled_vs_ref={m['j_compiled_vs_ref']:.6f}, "
        f"slack={m['slack']:.6f}, margin={m['margin']:.6f}) — the gate "
        f"activated a graph that materially degrades vs ideal-math "
        f"compared to bf16-eager (RELEASE BLOCKER)"
    )
    # And the margin must be non-negative: this is the ACTUAL pass condition.
    assert m["margin"] >= 0, (
        f"{model_name}@{layer} {dtype_name}: gate activated but margin="
        f"{m['margin']:.6f} < 0 — predicate semantics violated"
    )

    _free(native, sae)


# --------------------------------------------------------------------------- #
# Arm 2 — teeth regression guard.                                             #
#                                                                             #
# gemma-3-1b-pt@7 bf16 with `emulate_precision_casts=False` (default Inductor #
# — the KNOWN-corrupting config; S1 investigation: jaccard ≈0.969 < 0.99).    #
# The production gate MUST FAIL: force=False ⇒ collector stays eager          #
# (_use_compiled False); force=True ⇒ CompileGateError.  Proves the self-     #
# check genuinely detects the corrupting regime — if this EVER stops failing  #
# the gate is vacuous.                                                        #
#                                                                             #
# We keep `_native_compile_gate` + `enable_compiled` byte-UNCHANGED. The      #
# faithful way to exercise default-Inductor THROUGH the production gate (whose #
# `enable_compiled()` hard-sets `emulate_precision_casts=True`) is to swap the #
# collector module's `_inductor_config` ALIAS for a shim that swallows the    #
# `emulate_precision_casts=True` write, while the REAL                        #
# `torch._inductor.config.emulate_precision_casts` is explicitly set False    #
# (and restored in `finally`). Inductor's LAZY trace (first compiled call,    #
# inside the gate) then reads the real flag = False ⇒ default Inductor ⇒ the  #
# corrupting regime, with the production gate code path itself untouched.     #
# --------------------------------------------------------------------------- #
class _CorruptingCompiledWrapper:
    """Wraps a real compiled base and adds a large, deterministic perturbation
    to the output ``last_hidden_state``. Simulates a materially-corrupting
    compiled config (analogous to historical default-Inductor at j~0.97
    against ideal-math): compiled output diverges from ideal-math by more
    than bf16-eager does, by more than ``BASELINE_RELATIVE_SLACK``.

    This is the "teeth" mechanism the spec calls for: the gate must detect
    when compiled is materially worse than bf16-eager vs ideal-math. We use
    fault-injection (deterministic, reproducible, isolation-friendly) instead
    of forcing a particular inductor flag combination — the new gate's
    default (emulate=False, no graph-break) is empirically GOOD on 4090, so
    a flag-flip test would no longer reliably reproduce a corrupting regime.
    A fault-injection wrapper that DEFINITIVELY corrupts the compiled output
    is the most direct way to prove the gate has teeth.
    """

    def __init__(self, real_compiled, perturbation: float = 5.0):
        self._real = real_compiled
        self._perturbation = perturbation

    def __call__(self, *args, **kwargs):
        out = self._real(*args, **kwargs)
        # Add a large, broadcast-friendly perturbation. 5.0 is well above
        # bf16-ULP noise but small enough that the SAE still fires real
        # features (so the comparison is non-vacuous).
        out.last_hidden_state = out.last_hidden_state + self._perturbation
        return out


@pytest.mark.slow
def test_compile_gate_teeth_default_inductor_fails():
    """gemma-3-1b-pt@7 bf16: when the compiled path is replaced with a
    materially-corrupting wrapper (simulating the historical default-Inductor
    regime at j~0.97 against ideal-math, ~0.019 worse than bf16-eager — well
    outside slack=0.010), the production gate MUST raise CompileGateError
    AND leave the collector eager (never compiled-but-unverified).

    Why a fault-injection wrapper instead of an inductor flag flip: the new
    gate's default (emulate=False, no graph-break) is empirically GOOD on
    4090 (j=0.990 vs ideal-math, BETTER than bf16-eager). The historical
    "default-Inductor" corrupting regime no longer reliably reproduces via
    any specific flag combination on the current toolchain. A deterministic
    fault-injection wrapper that DEFINITIVELY corrupts compiled output is
    the most direct way to prove the gate has teeth — without it the test
    might silently pass on a regime that stopped corrupting (and the gate
    would be vacuous).

    The teeth still bite: j_compiled_vs_ref drops far below
    j_baseline_vs_ref - slack ⇒ predicate FAIL ⇒ CompileGateError. If this
    EVER stops raising the gate is vacuous.
    """
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    model_name, release, layer, l0 = (
        "gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium
    )
    model_dtype = t.bfloat16

    _, sae = _load_sae_only(model_name, layer, l0, release)
    sae = sae.to(model_dtype)
    docs = _tokenize_ragged(model_name, _REALISTIC_DOCS)
    probe = _right_padded_probe(docs, "cuda")

    _isolate_compile_caches()
    native = build_native_model(model_name, "cuda", model_dtype, layer)
    assert native is not None

    # Arm enable_compiled the normal way (production's call path), then swap
    # _compiled_base for the corrupting wrapper. This is the fault-injection
    # equivalent of the prior "swap inductor config to corrupting regime"
    # mechanism, applied at the compiled-base level instead of the inductor
    # flag level — strictly downstream, equally faithful as a teeth proof.
    native.enable_compiled()
    real_compiled = native._compiled_base
    native._compiled_base = _CorruptingCompiledWrapper(real_compiled, 5.0)

    # Under the no-opt-out contract, a failed predicate MUST raise
    # CompileGateError AND leave the collector eager.
    with pytest.raises(CompileGateError):
        _native_compile_gate(
            native, sae, probe_input_ids=probe, model_dtype=model_dtype,
        )
    assert native._use_compiled is False, (
        "TEETH GUARD VACUOUS or step-4 finally regressed: corrupting "
        "compiled raised CompileGateError but the collector was left "
        "compiled-active — must be eager after the raise"
    )

    # Independently INSTRUMENT the gate's three captures so we can both
    # (a) print the measured envelope for the [compile-envelope] TEETH
    # report, AND (b) assert the regime IS still corrupting via the SAME
    # production predicate (the teeth still bite).
    hook_name = f"blocks.{native.truncated_layer}.hook_resid_post"
    lengths = [int((row != 0).sum().item()) for row in probe]

    # baseline (bf16-eager).
    native.set_compiled_active(False)
    b_acts = native.capture(
        probe, native.truncated_layer, hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    b_ids, b_mags = _gate_sparse_features(sae, b_acts, lengths)

    # compiled (the corrupting wrapper).
    native.set_compiled_active(True)
    try:
        c_acts = native.capture(
            probe, native.truncated_layer, hook_name=hook_name,
            activation_type=ActivationType.residual,
        )
    finally:
        native.set_compiled_active(False)
    c_ids, c_mags = _gate_sparse_features(sae, c_acts, lengths)

    # fp32-eager reference.
    sae.to(t.float32)
    ref_native = build_native_model(model_name, "cuda", t.float32, layer)
    assert ref_native is not None
    r_acts = ref_native.capture(
        probe, ref_native.truncated_layer, hook_name=hook_name,
        activation_type=ActivationType.residual,
    )
    r_ids, r_mags = _gate_sparse_features(sae, r_acts, lengths)
    sae.to(model_dtype)
    _free(ref_native, r_acts)

    passed, m = compiled_passes_ideal_math_gate(
        ref_ids=r_ids,
        baseline_ids=b_ids,
        compiled_ids=c_ids,
        slack=BASELINE_RELATIVE_SLACK,
    )
    print(
        f"[compile-envelope] {model_name}@{layer} bf16 "
        f"TEETH(fault-injected corrupting compiled): "
        f"j_baseline_vs_ref={m['j_baseline_vs_ref']:.6f} "
        f"j_compiled_vs_ref={m['j_compiled_vs_ref']:.6f} "
        f"slack={m['slack']:.6f} margin={m['margin']:.6f} "
        f"-> independent_predicate_passed={passed} (EXPECT FAIL)"
    )

    # Non-vacuity: the regime is GENUINELY corrupting (compiled jaccard vs
    # ref is materially below baseline jaccard vs ref).
    assert not passed, (
        f"TEETH GUARD VACUOUS: the fault-injected corrupting wrapper did "
        f"NOT trigger the gate's fail predicate "
        f"(j_compiled_vs_ref={m['j_compiled_vs_ref']:.6f} >= "
        f"j_baseline_vs_ref-slack={m['j_baseline_vs_ref'] - m['slack']:.6f}); "
        f"the gate is vacuous on this stack (investigate)"
    )
    assert m["margin"] < 0, (
        f"TEETH GUARD: corrupting wrapper margin={m['margin']:.6f} should "
        f"be < 0 (compiled materially worse than baseline)"
    )
    _free(native, sae)


# --------------------------------------------------------------------------- #
# Arm 3 — dynamic-shape recompile-hole proof (Minor-3 closure).              #
#                                                                             #
# Take ONE collector, run the REAL gate so it ACTIVATES compiled on a cell    #
# that genuinely passes (gemma-2-2b bf16), then push ≥2 DISTINCT (batch, seq) #
# ragged shapes representative of production through `capture()` on the SAME  #
# armed+activated collector and assert EACH compiled output is feature-       #
# equivalent (project oracle) to the eager output for that SAME shape.        #
# Proves the single `dynamic=True` graph stays correct across shapes with NO  #
# un-probed recompile (the exact hole Minor-3 raised). Divergence on any      #
# shape ⇒ RELEASE BLOCKER, reported loudly.                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_compile_gate_dynamic_shape_recompile_hole():
    """After the gate activates compiled (gemma-2-2b bf16), ≥2 distinct
    (batch, seq) ragged shapes each stay compiled≈eager — the single
    dynamic=True graph has NO un-probed recompile hole."""
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    _isolate_compile_caches()
    model_name, release, layer, l0 = (
        "gemma-2-2b", SAERelease.gemmascope, 6, None
    )
    model_dtype = t.bfloat16  # gemma-2-2b bf16 genuinely PASSES the gate

    _, sae = _load_sae_only(model_name, layer, l0, release)
    sae = sae.to(model_dtype)

    # Gate probe: the realistic ragged corpus (the production decision input).
    gate_docs = _tokenize_ragged(model_name, _REALISTIC_DOCS)
    gate_probe = _right_padded_probe(gate_docs, "cuda")

    native = build_native_model(model_name, "cuda", model_dtype, layer)
    assert native is not None
    _native_compile_gate(
        native, sae, probe_input_ids=gate_probe, model_dtype=model_dtype,
    )
    # This arm's premise is that the gate ACTIVATED compiled (gemma-2-2b bf16
    # passes on the 4090). If it did not, the recompile-hole question is moot
    # AND we must surface it (do not silently pass a vacuous arm).
    assert native._use_compiled is True, (
        f"{model_name} bf16: the gate did NOT activate compiled, so the "
        f"dynamic-shape recompile-hole cannot be exercised on the activated "
        f"path. gemma-2-2b bf16 is the expected-PASS cell (S1 investigation) "
        f"— investigate before trusting the recompile-hole proof"
    )

    hook_name = f"blocks.{native.truncated_layer}.hook_resid_post"
    mag_rtol = BF16_NATIVE_VS_TL_MAG_RTOL
    mag_atol = BF16_NATIVE_VS_TL_MAG_ATOL

    # >=2 DISTINCT (batch, seq) shapes representative of ragged production:
    #   * (batch=1, short ≈40 tok)        — a single short query;
    #   * (batch>=3, long ≈1.2k padded)   — a ragged multi-doc batch whose max
    #     length is the >sliding_window `_LONG_DOC` (real production scale).
    # NEITHER equals the gate-probe shape in the batch dim, and the short shape
    # is a different seq regime entirely — so each is a genuine NEW (batch,
    # seq) the single dynamic=True graph must serve WITHOUT an un-probed
    # recompile.
    short_doc = _tokenize_ragged(model_name, [_SHORT_DOC, _MED_DOC])[0]
    shape_a = t.tensor([short_doc[:40]], dtype=t.long, device="cuda")
    long_docs = _tokenize_ragged(
        model_name, [_SHORT_DOC, _MED_DOC, _PARA_DOC, _LONG_DOC]
    )
    shape_b = _right_padded_probe(long_docs, "cuda")  # batch=4, seq≈1215

    shapes = {
        f"(B={shape_a.shape[0]},S={shape_a.shape[1]}) short-single": shape_a,
        f"(B={shape_b.shape[0]},S={shape_b.shape[1]}) long-ragged": shape_b,
    }
    assert len({tuple(s.shape) for s in shapes.values()}) >= 2, (
        "recompile-hole proof needs >=2 DISTINCT (batch, seq) shapes"
    )

    for label, ids in shapes.items():
        lens = [int((row != 0).sum().item()) for row in ids]
        # Eager reference for THIS shape (compiled inactive — PR-1 anchor).
        native.set_compiled_active(False)
        e_acts = native.capture(
            ids, native.truncated_layer, hook_name=hook_name,
            activation_type=ActivationType.residual,
        )
        e_ids, e_mags = _gate_sparse_features(sae, e_acts, lens)
        # Compiled output for the SAME shape, on the SAME armed+activated
        # collector (the single dynamic=True graph). Restore the gate's
        # activated state afterwards.
        native.set_compiled_active(True)
        try:
            c_acts = native.capture(
                ids, native.truncated_layer, hook_name=hook_name,
                activation_type=ActivationType.residual,
            )
        finally:
            native.set_compiled_active(True)
        c_ids, c_mags = _gate_sparse_features(sae, c_acts, lens)

        assert_features_nonempty(e_ids, e_mags, label=f"eager {label}")
        assert_features_nonempty(c_ids, c_mags, label=f"compiled {label}")

        mj, mcr = feature_equivalence_metrics(e_ids, c_ids)
        mma, mmr = feature_magnitude_metrics(e_ids, e_mags, c_ids, c_mags)
        print(
            f"[compile-envelope] {model_name}@{layer} bf16 RECOMPILE-HOLE "
            f"{label}: jaccard={mj:.6f} count_rdiff={mcr:.6f} "
            f"mag_max_abs={mma:.6g} mag_max_rel={mmr:.6g} "
            f"vs gate jaccard>={JACCARD_THRESHOLD} "
            f"mag rtol={mag_rtol}/atol={mag_atol} [bf16]"
        )
        # The single dynamic=True graph MUST stay feature-equivalent to eager
        # for this NEW shape — a divergence here is the exact un-probed
        # recompile hole Minor-3 raised (RELEASE BLOCKER).
        assert_feature_equivalent(
            e_ids, e_mags, c_ids, c_mags,
            label=f"compiled-vs-eager {model_name}@{layer} bf16 {label} "
                  f"(dynamic-shape recompile-hole)",
            mag_rtol=mag_rtol, mag_atol=mag_atol,
        )

    _free(native, sae)


# --------------------------------------------------------------------------- #
# Arm 4 — eager anchor unchanged.                                            #
#                                                                             #
# Under ALL the compile machinery the eager `_truncated_forward`/`capture`    #
# MUST still be byte-identical (`torch.equal`) to `hook_reference_residual`   #
# (the verbatim Phase-1 hook). PR-1 eager is the trusted anchor the whole     #
# sub-project's safety rests on; this confirms arming/activating/the gate     #
# never perturbed it. Runs for both cells (the heavier gemma-2-2b on the      #
# memory-aware sub-corpus, gemma-3-1b-pt on the full ragged corpus), at the   #
# production bf16 dtype.                                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("model_name,release,layer,l0", [
    ("gemma-2-2b", SAERelease.gemmascope, 6, None),
    ("gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium),
])
def test_compile_gate_eager_anchor_unchanged(model_name, release, layer, l0):
    """Eager `capture()` == `hook_reference_residual` (torch.equal) even after
    the collector has been armed AND activated compiled by the real gate —
    PR-1 eager remains the trusted byte-identical anchor."""
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    _isolate_compile_caches()
    model_dtype = t.bfloat16  # production dtype

    _, sae = _load_sae_only(model_name, layer, l0, release)
    sae = sae.to(model_dtype)
    docs = _tokenize_ragged(
        model_name, _compile_gate_docs(model_name, model_dtype)
    )
    probe = _right_padded_probe(docs, "cuda")

    native = build_native_model(model_name, "cuda", model_dtype, layer)
    assert native is not None

    # Eager BEFORE any compile machinery == fresh FULL HF model's verbatim
    # Phase-1 hook reference. `hook_reference_residual` needs the FULL
    # `Gemma{2,3}ForCausalLM` (it does `hf_model.model.layers[layer]` + a full
    # forward), NOT `native._base` (which already IS the inner decoder and has
    # no `.model`). resid_post(layer) off a hook on the full model == the
    # truncated collector's capture by causal-attention invariance.
    ref_model = _full_hf_reference_model(model_name, model_dtype)
    ref = hook_reference_residual(ref_model, probe, layer)
    eager_before = native.capture(
        probe, native.truncated_layer,
        hook_name=f"blocks.{native.truncated_layer}.hook_resid_post",
        activation_type=ActivationType.residual,
    )
    assert t.equal(eager_before, ref), (
        f"{model_name}@{layer} bf16: eager capture() != hook_reference "
        f"BEFORE any compile machinery (max_abs="
        f"{(eager_before.float() - ref.float()).abs().max().item()!r}) — "
        f"the PR-1 eager anchor is broken independent of compile"
    )
    _free(ref_model)

    # Run the REAL gate (arms enable_compiled + may activate compiled).
    _native_compile_gate(
        native, sae, probe_input_ids=probe, model_dtype=model_dtype,
    )

    # Eager MUST STILL be byte-identical after arming (and possible
    # activation): force the eager path explicitly and compare to a FRESH HF
    # twin's hook reference (the verbatim Phase-1 mechanism).
    native.set_compiled_active(False)
    eager_after = native.capture(
        probe, native.truncated_layer,
        hook_name=f"blocks.{native.truncated_layer}.hook_resid_post",
        activation_type=ActivationType.residual,
    )
    ref_model2 = _full_hf_reference_model(model_name, model_dtype)
    ref2 = hook_reference_residual(ref_model2, probe, layer)
    assert t.equal(eager_after, ref2), (
        f"{model_name}@{layer} bf16: eager capture() != hook_reference "
        f"AFTER the gate armed/activated compile (max_abs="
        f"{(eager_after.float() - ref2.float()).abs().max().item()!r}) — "
        f"the compile machinery perturbed the trusted PR-1 eager anchor "
        f"(RELEASE BLOCKER)"
    )
    # And eager must be self-consistent before vs after the gate ran.
    assert t.equal(eager_before, eager_after), (
        f"{model_name}@{layer} bf16: eager capture changed across the gate "
        f"run (before vs after differ) — arming/activating compile mutated "
        f"the eager path"
    )
    _free(native, sae, ref_model2)


# --------------------------------------------------------------------------- #
# Arm 5 — GPU memory returns to pre-gate watermark after the gate runs.       #
#                                                                             #
# The ideal-math-relative gate builds a SEPARATE fp32 reference model (~4 GB  #
# for gemma-3-1b) during startup. The transient fp32 model + fp32 SAE MUST be #
# explicitly freed (del + gc.collect + empty_cache) before the gate returns,  #
# so the steady-state VRAM footprint of a gated extraction is unchanged from  #
# the pre-gate watermark (within a small slop band). If the fp32 reference    #
# leaked, a gated gemma-3-1b extraction would silently carry ~4 GB extra      #
# resident — this arm fails LOUDLY if that regression returns.                #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_compile_gate_frees_fp32_reference_memory():
    """After ``_native_compile_gate`` returns, GPU allocated memory is back to
    the pre-gate watermark within a small slop band — the transient fp32
    reference (model + SAE) was freed, not leaked."""
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    model_name, release, layer, l0 = (
        "gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium
    )
    model_dtype = t.bfloat16

    _isolate_compile_caches()
    _, sae = _load_sae_only(model_name, layer, l0, release)
    sae = sae.to(model_dtype)
    docs = _tokenize_ragged(model_name, _REALISTIC_DOCS)
    probe = _right_padded_probe(docs, "cuda")

    native = build_native_model(model_name, "cuda", model_dtype, layer)
    assert native is not None

    # Pre-gate watermark: the production collector + SAE are resident, the
    # gate has NOT yet run. gc + empty_cache so the baseline is the true
    # steady-state allocated bytes.
    gc.collect()
    t.cuda.empty_cache()
    pre_gate_bytes = t.cuda.memory_allocated()

    # Run the production gate (builds + frees the fp32 reference internally).
    _native_compile_gate(
        native, sae, probe_input_ids=probe, model_dtype=model_dtype,
    )
    assert native._use_compiled, "gate must activate (precondition for this arm)"

    # Post-gate watermark: gc + empty_cache so any unreferenced fp32 stack is
    # actually reclaimed, then compare.
    gc.collect()
    t.cuda.empty_cache()
    post_gate_bytes = t.cuda.memory_allocated()

    delta_mb = (post_gate_bytes - pre_gate_bytes) / (1024 * 1024)
    print(
        f"[compile-envelope] {model_name}@{layer} bf16 MEMORY: "
        f"pre_gate={pre_gate_bytes / 1024 / 1024:.1f} MB "
        f"post_gate={post_gate_bytes / 1024 / 1024:.1f} MB "
        f"delta={delta_mb:.1f} MB (fp32 ref ~4 GB transient must be freed)"
    )
    # The gate may legitimately allocate a small amount of steady-state memory
    # (the compiled graph's persistent buffers). 100 MB is a generous slop band
    # that still catches a ~4 GB fp32-reference LEAK by ~40x.
    assert delta_mb < 100.0, (
        f"{model_name}@{layer} bf16: GPU allocated memory grew by "
        f"{delta_mb:.1f} MB across the gate run (pre={pre_gate_bytes} B, "
        f"post={post_gate_bytes} B) — the transient fp32 reference (~4 GB) "
        f"was NOT freed (del + gc.collect + empty_cache regressed); a gated "
        f"extraction would silently carry the extra resident (RELEASE BLOCKER)"
    )

    _free(native, sae)


# =========================================================================== #
# VLM backend-parity + non-degeneracy oracle (issue #549).                    #
#                                                                             #
# gemma-3-4b-pt (and 12b/27b) are VLM repos: weights live under              #
# ``language_model.model.*``, but ``Gemma3ForCausalLM`` expects ``model.*``   #
# and silently random-initialises the text decoder (pre-fix). The fix in      #
# ``steering.gemma3_loading`` remaps the key prefix before loading.           #
#                                                                             #
# This test asserts TWO independent post-fix properties:                      #
#   (a) native ≡ nnterp BIT-IDENTICAL at bf16: both HF-SDPA backends load     #
#       CORRECT text weights → same activations → same SAE features.          #
#   (b) NON-DEGENERACY: feature 41629 (app-store / Play-Store URL activator   #
#       on GemmaScope-2 layer 22, width 65k) fires STRONGLY (> 100) on the   #
#       URL tokens. The pre-fix random-weight regime makes the SAE decoder     #
#       incoherent so this feature never exceeds threshold; additionally the  #
#       random weights produce a DENSE per-token activation signature         #
#       (≥ 50% of features fire per token), which is asserted not to hold.   #
# =========================================================================== #


@pytest.mark.slow
def test_gemma3_4b_vlm_native_nnterp_parity_and_non_degenerate():
    """VLM-repo native ≡ nnterp parity + non-degeneracy oracle (issue #549).

    Requires CUDA + gated google/gemma HF access + cached model weights.
    Skips cleanly without CUDA or without the ``nnterp`` package.

    PRE-FIX (random weights) the test would have FAILED on both:
      - ``t.equal`` would fail: random activations diverge between backends
        because they start from different random states (or the same wrong
        initialisation, but the SAE cannot cancel it consistently).
      - ``native_acts[:, 41629].max() > 100.0`` would fail: the random-weight
        SAE decoder produces incoherent features, none reliably firing strongly
        on any specific input type.
      - ``per_tok_active.max() < n_features * 0.5`` would fail: random SAE
        input → many features exceed the JumpReLU threshold on every token.
    """
    pytest.importorskip("nnterp")
    if not t.cuda.is_available():
        pytest.skip("needs CUDA")

    model_name, layer = "gemma-3-4b-pt", 22
    resolved = resolve_sae(
        model=model_name,
        release=SAERelease.gemmascope_2,
        activation_type=ActivationType.residual,
        layer=layer,
        width=SAEWidth.W_65K,
        l0=L0Tier.medium,
    )

    # URL text: feature 41629 is the app-store URL activator.
    text = (
        "Download the app at https://play.google.com/store/apps/"
        "details?id=com.example.myapp and enjoy."
    )
    tok = resolve_tokenizer(model_name)
    ids = [tok(text, add_special_tokens=True)["input_ids"]]

    # Load SAE directly — avoids loading the full TransformerLens 4B model
    # (~16 GB fp32) which is not needed here: both passes use HF-native
    # backends (native truncated collector + nnterp HF/SDPA). This is the same
    # pattern as _load_sae_only but without the TL overhead for a 4B model.
    sae_raw = SAE.from_pretrained(
        release=resolved.release,
        sae_id=resolved.sae_id,
        device="cuda",
    )
    sae = sae_raw[0] if isinstance(sae_raw, tuple) else sae_raw
    sae = sae.to(t.bfloat16)

    # Pass 1 — native (truncated HF/SDPA, bf16).
    native = build_native_model(model_name, "cuda", t.bfloat16, layer)
    native_acts = _get_sae_activations_batched(
        None, sae, ids, layer,  # type: ignore[arg-type]  # TL unused: native path  # ty: ignore[invalid-argument-type]
        native_model=native,
        hook_name=resolved.hook_name,
        activation_type=ActivationType.residual,
    )[0]
    del native
    t.cuda.empty_cache()
    gc.collect()

    # Pass 2 — nnterp (fresh HF/SDPA, bf16). tl_model=None: gemma-3 loads
    # directly from HF; model_name drives the correct VLM-aware loader.
    nnterp = build_nnterp_model(
        tl_model=None, device="cuda", model_dtype=t.bfloat16,
        model_name=model_name,
    )
    nnterp_acts = _get_sae_activations_batched(
        None, sae, ids, layer,  # type: ignore[arg-type]  # TL unused: nnterp path  # ty: ignore[invalid-argument-type]
        nnterp_model=nnterp,
        hook_name=resolved.hook_name,
        activation_type=ActivationType.residual,
    )[0]
    # Free the 4B nnterp model BEFORE the assertions (mirrors the native side
    # above): nnterp_acts is all the asserts need, and a failed assertion must
    # not leave the model GPU-resident to OOM subsequent tests.
    del nnterp
    t.cuda.empty_cache()
    gc.collect()  # break nnsight ref cycles so the 4B model frees (see headline test :575-580)

    # (a) native ≡ nnterp BIT-IDENTICAL: both correct raw-HF/SDPA — any
    # divergence is a real capture-site, reshape, or weight-loading bug.
    assert t.equal(native_acts, nnterp_acts), (
        f"native vs nnterp NOT bit-identical for {model_name}@{layer} bf16; "
        f"max_abs={(native_acts.float() - nnterp_acts.float()).abs().max().item()!r} "
        f"— mismatch indicates a capture-site or weight-loading bug"
    )

    # (b) NON-DEGENERACY ORACLE (the pre-fix random-weight bug catches):
    # Feature 41629 must fire strongly on the app-store URL tokens.
    assert native_acts[:, 41629].max() > 100.0, (
        f"feature 41629 max activation {native_acts[:, 41629].max().item():.1f} "
        f"≤ 100.0 on URL text — expected strong firing; likely pre-fix "
        f"random-weight regime (VLM text-decoder weight loading bug, issue #549)"
    )
    # Dense-every-token activation is the random-weight SAE signature.
    per_tok_active = (native_acts > 0).sum(-1).float()
    assert per_tok_active.max() < native_acts.shape[-1] * 0.5, (
        f"max per-token active features {per_tok_active.max().item():.0f} "
        f"≥ 50% of {native_acts.shape[-1]} — dense-every-token activation is "
        f"the random-weight signature (pre-fix VLM loading bug, issue #549)"
    )
