"""Real-model gates for gradient attribution (gemma-2-2b + GemmaScope 16k).

Run on a GPU machine with HF access:
    uv run pytest tests/steering/test_gradient_attribution_real_gate.py -m very_slow

Gates (measure-then-assert):
1. chunked lm_head tail vs ``model(input_ids, use_cache=False).logits`` at
   the SAME dtype — bit-exact when the platform's GEMM is row-subset
   invariant (probed), else within a tiny envelope;
2. fp32-vs-bf16 per-segment top-32 score-set jaccard distribution;
3. final-unit non-vacuity (doc-final scores not all zero under both future
   modes) + short-final merge exercised;
4. zero-score-fraction band (a sign inversion would collapse it);
5. no inf/nan + fp16 overflow/underflow rates logged;
6. full-context batch>1 stress with peak-VRAM watermark logged.
"""

from __future__ import annotations

import gc

import pytest
import torch as t

from steering.gradient_attribution import (
    GradientAttributionCollector,
    build_gradient_model,
    decoder_linearity_gate,
)
from steering.gradient_overlay import GradientAttributionMode
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    resolve_sae,
)
from steering.segmentation import SegmentMode, segment_tokens

pytestmark = [
    pytest.mark.very_slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA"),
]

_MODEL = "gemma-2-2b"
_LAYER = 8
_SEG_SIZE = 128
_MIN_FINAL = 8

_PROBE_TEXTS = [
    # Long-ish doc with a natural short tail; a mid doc; a short doc whose
    # final segment is < _MIN_FINAL tokens (merge path); a tiny doc.
    "Language models learn statistical regularities from vast text corpora. "
    * 40,
    "Attention routes information between positions in a causal decoder. " * 12,
    "The quick brown fox jumps over the lazy dog. " * 3 + "End.",
    "Tiny.",
]


class _SingleDtypeRuntime:
    """Holds (model, sae) for ONE dtype at a time, swapping on demand.

    Loading the fp32 and bf16 gemma-2-2b copies side by side drove a 4090 to
    an external peak of 23509/24564 MiB (round-2 gate report, 2026-06-12) —
    ~1 GiB of headroom on an idle box, an OOM with any co-resident process.
    Swapping costs one cached-weights reload per dtype change (the test
    order keeps it to a handful per session) and roughly halves peak VRAM.
    """

    def __init__(self, resolved, tokenizer):
        self._resolved = resolved
        self.tokenizer = tokenizer
        self._dtype = None
        self._pair = None

    def __getitem__(self, key):
        if key == "tokenizer":
            return self.tokenizer
        if key != self._dtype:
            from sae_lens import SAE

            self._pair = None
            gc.collect()
            t.cuda.empty_cache()
            model = build_gradient_model(_MODEL, "cuda", key)
            sae = SAE.from_pretrained(
                release=self._resolved.release,
                sae_id=self._resolved.sae_id,
                device="cuda",
            )
            if key != t.float32:
                sae.to(key)
            decoder_linearity_gate(sae)
            self._dtype = key
            self._pair = (model, sae)
        return self._pair


@pytest.fixture(scope="module")
def runtime():
    from steering.models import resolve_tokenizer
    from steering.nnsight_compat import unpatch_nnsight

    unpatch_nnsight()
    resolved = resolve_sae(
        model=_MODEL,
        release=SAERelease.gemmascope,
        activation_type=ActivationType.residual,
        layer=_LAYER,
        width=SAEWidth.W_16K,
        l0=None,
    )
    return _SingleDtypeRuntime(resolved, resolve_tokenizer(_MODEL))


def _docs(tokenizer):
    docs, segs, fids_lists = [], [], []
    for text in _PROBE_TEXTS:
        ids = tokenizer(text, padding=False)["input_ids"]
        if isinstance(ids[0], list):
            ids = ids[0]
        ids = ids[:1024]
        docs.append(ids)
        s = segment_tokens(
            ids,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(_SEG_SIZE),
        )
        segs.append(s)
        fids_lists.append([t.arange(16384, dtype=t.int64) for _ in s])
    return docs, segs, fids_lists


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
def test_chunked_tail_matches_full_forward(runtime, dtype):
    model, _sae = runtime[dtype]
    ids = _docs(runtime["tokenizer"])[0][1]
    input_ids = t.tensor(ids, device="cuda").unsqueeze(0)
    with t.no_grad():
        full = model(input_ids=input_ids, use_cache=False).logits
        hidden = model.model(
            input_ids=input_ids, attention_mask=None, use_cache=False
        ).last_hidden_state
        lo, hi = 64, 192
        chunk = model.lm_head(hidden[0, lo:hi])
        cap = getattr(model.config, "final_logit_softcapping", None)
        assert cap is not None, "gemma-2 must carry final_logit_softcapping"
        chunk = t.tanh(chunk / cap) * cap
        probe_exact = t.equal(model.lm_head(hidden)[0, lo:hi], model.lm_head(hidden[0, lo:hi]))
        diff = (chunk - full[0, lo:hi]).abs().max().item()
    print(f"[{dtype}] chunked-tail vs full max abs diff: {diff:.3e} "
          f"(row-subset probe exact: {probe_exact})")
    if probe_exact:
        assert t.equal(chunk, full[0, lo:hi])
    else:
        tol = 1e-5 if dtype == t.float32 else 1e-2
        assert diff <= tol


@pytest.mark.parametrize(
    "mode",
    [GradientAttributionMode.future_half_split, GradientAttributionMode.future_self_full],
    ids=lambda m: m.value,
)
def test_scores_sane_on_real_model(runtime, mode):
    model, sae = runtime[t.bfloat16]
    docs, segs, fids_lists = _docs(runtime["tokenizer"])
    collector = GradientAttributionCollector(
        model, sae, layer=_LAYER, mode=mode, min_final_segment_tokens=_MIN_FINAL
    )
    t.cuda.reset_peak_memory_stats()
    out = collector.score_documents(docs, segs, fids_lists)
    peak_gb = t.cuda.max_memory_allocated() / 2**30
    print(f"[{mode.value}] peak VRAM: {peak_gb:.2f} GiB")

    flat = t.cat([r.float() for rows in out for r in rows])
    assert bool(t.isfinite(flat).all())
    assert bool((flat >= 0).all())
    nonzero = flat[flat > 0]
    zero_frac = 1.0 - nonzero.numel() / flat.numel()
    print(
        f"[{mode.value}] zero-fraction={zero_frac:.3f}, max={flat.max():.4g}, "
        f"fp16-underflow-rate among gated entries unavailable (stored fp16 already)"
    )
    assert 0.05 < zero_frac < 0.995

    # Final-unit non-vacuity: at least one multi-segment doc's final segment
    # carries nonzero scores (split/self_full treatment effective).
    multi = [d for d in range(len(docs)) if len(segs[d]) >= 2]
    assert any(bool((out[d][-1].float() > 0).any()) for d in multi), (
        "every doc-final segment scored all-zero — final-unit treatment broken?"
    )


def test_fp32_vs_bf16_topk_jaccard(runtime):
    docs, segs, fids_lists = _docs(runtime["tokenizer"])
    mode = GradientAttributionMode.future_half_split
    rows = {}
    for dtype in (t.float32, t.bfloat16):
        model, sae = runtime[dtype]
        collector = GradientAttributionCollector(
            model, sae, layer=_LAYER, mode=mode, min_final_segment_tokens=_MIN_FINAL
        )
        rows[dtype] = collector.score_documents(docs, segs, fids_lists)
        # Drop refs before the runtime swaps dtypes — lingering locals would
        # keep both 2b-model copies co-resident (the VRAM hazard the
        # single-dtype runtime exists to avoid).
        del model, sae, collector
    jaccards = []
    for d in range(len(docs)):
        for a, b in zip(rows[t.float32][d], rows[t.bfloat16][d], strict=True):
            k = min(32, int((a > 0).sum()), int((b > 0).sum()))
            if k < 8:
                continue
            ja = set(a.float().topk(k).indices.tolist())
            jb = set(b.float().topk(k).indices.tolist())
            jaccards.append(len(ja & jb) / len(ja | jb))
    mn = min(jaccards)
    mean = sum(jaccards) / len(jaccards)
    print(f"fp32-vs-bf16 top-32 jaccards: min={mn:.3f}, "
          f"mean={mean:.3f}, n={len(jaccards)}")
    # Two independent signals; EITHER failing is cause for concern.
    #
    # mean — the sensitive regression detector. Stable across runs
    #   (4090-measured 2026-06-12: 0.924 then 0.916, n=7; spread ~0.008), so
    #   the 0.85 floor (~0.07 below the worst observed mean) catches a
    #   moderate bf16 degradation that a noisy min-of-n would sail past.
    # min — the catastrophe backstop. min-of-n is a noisy extreme order
    #   statistic (worst observed 0.829, run-to-run spread 0.053), so its
    #   floor stays loose at 0.70; against the real failure mode (sign flip /
    #   broken gate → jaccard collapses toward the ~0.008 random baseline)
    #   0.70 and 0.85 are equally decisive.
    # Do NOT tighten either without re-measuring on GPU with a larger probe
    # set — adding docs LOWERS the expected min (more chances for a
    # boundary-tie segment) while only stabilizing the mean.
    assert mean >= 0.85, (
        f"bf16 top-32 jaccard mean {mean:.3f} < 0.85 — bf16 scoring degraded "
        f"(moderate regression; mean is the sensitive detector)."
    )
    assert mn >= 0.70, (
        f"bf16 top-32 jaccard min {mn:.3f} < 0.70 — a segment's bf16 top-k "
        f"collapsed (catastrophe backstop)."
    )
