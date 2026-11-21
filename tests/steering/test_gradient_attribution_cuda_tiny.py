"""CUDA envelopes for gradient attribution on tiny models (no downloads).

Run on a GPU machine:  uv run pytest tests/steering/test_gradient_attribution_cuda_tiny.py -m slow

Measures-then-asserts (the native-collector discipline): CPU-fp32 vs
CUDA-fp32 parity, CUDA bf16-vs-fp32 top-k agreement, an independent
single-backward oracle spot-check on CUDA, batched-vs-single padding
invariance at both dtypes, rerun behavior under CUDA atomics, and all three
modes x merged-unit fixtures on device.
"""

from __future__ import annotations

import copy

import pytest
import torch as t

from steering.gradient_attribution import GradientAttributionCollector
from steering.gradient_overlay import GradientAttributionMode
from tests.reference_gradient_attribution import (
    ALL_MODES,
    TINY_LAYER,
    fixed_segments,
    full_fids,
    make_tiny_sae,
    ref_unit_loss,
    run_grouped_collector_trace,
    run_independent_document_trace,
    seeded_ids,
    tiny_gemma2,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA"),
]

_DOC = 26
_SEG = 8
_MIN_FINAL = 8


def _to_device(model, dtype):
    m = copy.deepcopy(model).to(device="cuda", dtype=dtype)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _collector(model, sae, mode, **kw):
    return GradientAttributionCollector(
        model, sae, layer=TINY_LAYER, mode=mode,
        min_final_segment_tokens=_MIN_FINAL, **kw,
    )


def _score_flat(collector, ids, segs, fids):
    rows = collector.score_documents([ids], [segs], [fids])[0]
    return t.cat([r.float() for r in rows])


@pytest.mark.slow
@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda mode: mode.value)
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
def test_ragged_grouped_vjp_trace_matches_independent_cuda_oracle(dtype, mode):
    """GA-011: real CUDA raw-gradient/reset/final oracle for ragged batches."""
    model = _to_device(tiny_gemma2(True), dtype)
    sae = make_tiny_sae("jumprelu").to(device="cuda", dtype=dtype)
    lengths = [26, 9, 17, 5]
    docs = [seeded_ids(n, seed=200 + n) for n in lengths]
    seg_lists = [fixed_segments(n, _SEG) for n in lengths]
    fid_lists = [[full_fids() for _ in segs] for segs in seg_lists]

    grouped = run_grouped_collector_trace(
        model, sae, docs, seg_lists, fid_lists, mode, _MIN_FINAL
    )
    independent = run_independent_document_trace(
        model, sae, docs, seg_lists, fid_lists, mode, _MIN_FINAL
    )
    assert grouped.unit_trace == independent.unit_trace

    raw_max = 0.0
    final_max = 0.0
    raw_pairs = []
    final_pairs = []
    # Local CUDA calibration (2026-07-21): fp32 raw <= 1.525e-08 and final
    # exactly equal; BF16 raw <= 3.662e-04 and final <= 1.390e-03. Keep a
    # multi-x envelope for kernel/architecture variation while still catching
    # reset, grouping, or accumulation drift.
    raw_rtol, raw_atol = (
        (2e-3, 1e-4) if dtype == t.float32 else (5e-2, 2e-3)
    )
    final_rtol, final_atol = (
        (2e-3, 2e-5) if dtype == t.float32 else (5e-2, 4e-3)
    )
    for doc_idx in range(len(docs)):
        for got, expected in zip(
            grouped.accumulated_grads[doc_idx],
            independent.accumulated_grads[doc_idx],
            strict=True,
        ):
            raw_max = max(raw_max, float((got - expected).abs().max()))
            raw_pairs.append((got, expected))
        for got, expected in zip(
            grouped.final_rows[doc_idx], independent.final_rows[doc_idx], strict=True
        ):
            got_fp32 = got.float()
            expected_fp32 = expected.float()
            final_max = max(
                final_max, float((got_fp32 - expected_fp32).abs().max())
            )
            final_pairs.append((got_fp32, expected_fp32))
    print(
        f"[{dtype}, {mode.value}] grouped-vs-independent raw max={raw_max:.3e}, "
        f"final max={final_max:.3e}"
    )
    assert all(
        t.allclose(got, expected, rtol=raw_rtol, atol=raw_atol)
        for got, expected in raw_pairs
    )
    assert all(
        t.allclose(got, expected, rtol=final_rtol, atol=final_atol)
        for got, expected in final_pairs
    )


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_cpu_vs_cuda_fp32_envelope(mode):
    ids = seeded_ids(_DOC, seed=1)
    segs = fixed_segments(_DOC, _SEG)
    fids = [full_fids() for _ in segs]
    cpu_model = tiny_gemma2(True)
    cpu_sae = make_tiny_sae("jumprelu")
    cuda_model = _to_device(cpu_model, t.float32)
    cuda_sae = make_tiny_sae("jumprelu").to("cuda")

    cpu = _score_flat(_collector(cpu_model, cpu_sae, mode), ids, segs, fids)
    gpu = _score_flat(_collector(cuda_model, cuda_sae, mode), ids, segs, fids)
    diff = (cpu - gpu).abs().max().item()
    print(f"[{mode.value}] CPU-fp32 vs CUDA-fp32 max abs diff: {diff:.3e}")
    assert t.allclose(cpu, gpu, rtol=1e-3, atol=1e-5)


def test_bf16_vs_fp32_topk_agreement():
    """Per-segment top-8 score-set jaccard between CUDA bf16 and CUDA fp32 —
    measured distribution printed, min asserted against a generous floor
    (tighten after first real measurement)."""
    mode = GradientAttributionMode.future_half_split
    ids = seeded_ids(_DOC, seed=2)
    segs = fixed_segments(_DOC, _SEG)
    fids = [full_fids() for _ in segs]
    rows = {}
    for dtype in (t.float32, t.bfloat16):
        model = _to_device(tiny_gemma2(True), dtype)
        sae = make_tiny_sae("jumprelu").to(device="cuda", dtype=dtype)
        rows[dtype] = _collector(model, sae, mode).score_documents(
            [ids], [segs], [fids]
        )[0]
    jaccards = []
    for a, b in zip(rows[t.float32], rows[t.bfloat16], strict=True):
        k = min(8, int((a > 0).sum()), int((b > 0).sum()))
        if k == 0:
            continue
        top_a = set(a.float().topk(k).indices.tolist())
        top_b = set(b.float().topk(k).indices.tolist())
        jaccards.append(len(top_a & top_b) / len(top_a | top_b))
    print(f"bf16-vs-fp32 per-segment top-k jaccards: {jaccards}")
    # 4090-measured 2026-06-10: all 1.0 on the tiny model; keep slack for
    # genuine bf16 reordering on other GPUs.
    assert jaccards and min(jaccards) >= 0.75


@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16], ids=["fp32", "bf16"])
def test_padding_invariance_on_cuda(dtype):
    mode = GradientAttributionMode.future_self_full
    model = _to_device(tiny_gemma2(True), dtype)
    sae = make_tiny_sae("jumprelu").to(device="cuda", dtype=dtype)
    lengths = [26, 9, 17, 5]
    docs = [seeded_ids(n, seed=n) for n in lengths]
    seg_lists = [fixed_segments(n, _SEG) for n in lengths]
    fid_lists = [[full_fids() for _ in segs] for segs in seg_lists]
    collector = _collector(model, sae, mode)
    batched = collector.score_documents(docs, seg_lists, fid_lists)
    max_diff = 0.0
    for d in range(len(docs)):
        single = collector.score_documents(
            [docs[d]], [seg_lists[d]], [fid_lists[d]]
        )[0]
        diff = (
            t.cat([r.float() for r in batched[d]])
            - t.cat([r.float() for r in single])
        ).abs().max().item()
        max_diff = max(max_diff, diff)
    print(f"[{dtype}] batched-vs-single max abs diff: {max_diff:.3e}")
    # Padding must not change real-row scores; bf16 rounding bounds the diff.
    # The collector returns the gradient SENSITIVITY (~0.02 scale; the producer
    # applies the percentile weight separately). 4090-measured bf16: 2026-06-10
    # 9.155e-05, 2026-06-12 2.012e-04 (~5x margin on the worse run). fp32 0.0.
    tol = 1e-5 if dtype == t.float32 else 1e-3
    assert max_diff <= tol


def test_independent_backward_oracle_spotcheck():
    """One segment's suffix gradient computed by a SINGLE autograd.grad
    backward (no chunking) must match the production chunked accumulation
    within a CUDA-fp32 envelope."""
    model = _to_device(tiny_gemma2(True), t.float32)
    sae = make_tiny_sae("standard").to("cuda")
    ids = seeded_ids(24, seed=9)
    segs = fixed_segments(24, _SEG)
    fids = [full_fids() for _ in segs]
    seg_idx, (b, e) = 0, segs[0]

    prod = _collector(model, sae, GradientAttributionMode.future_half_split)
    prod_scores = prod.score_documents([ids], [segs], [fids])[0][seg_idx]

    captured = []

    def _hook(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        h.requires_grad_(True)
        captured.append(h)

    handle = model.model.layers[TINY_LAYER].register_forward_hook(_hook)
    try:
        input_ids = t.tensor(ids, device="cuda").unsqueeze(0)
        hidden = model.model(
            input_ids=input_ids, attention_mask=None, use_cache=False
        ).last_hidden_state
        positions = [j for j in range(len(ids) - 1) if j + 1 >= e]
        loss = ref_unit_loss(model, hidden.cpu().cuda(), ids, positions)
        (g,) = t.autograd.grad(loss, captured[0])
    finally:
        handle.remove()
    g_rows = g[0, b:e].float() @ sae.W_dec.index_select(
        0, full_fids().cuda()
    ).float().T
    with t.no_grad():
        acts = sae.encode(captured[0][0, b:e].detach())
    # The collector returns the gated gradient SENSITIVITY (the producer applies
    # the percentile weight separately); the oracle matches that here.
    gate = acts.index_select(1, full_fids().cuda()) > 0
    oracle = t.relu(-g_rows).masked_fill(~gate, 0.0).amax(0).cpu()
    diff = (prod_scores.float() - oracle.float()).abs().max().item()
    print(f"chunked-accumulation vs single-backward oracle: {diff:.3e}")
    # 4090-measured 3.529e-06 (chunked accumulation vs single backward).
    assert t.allclose(prod_scores.float(), oracle.float(), rtol=1e-3, atol=1e-5)


def test_rerun_stability_under_atomics():
    """CUDA backward uses atomics; reruns may differ in low bits. Measure and
    pin a generous ceiling; assert top-k sets stay identical."""
    mode = GradientAttributionMode.self_
    model = _to_device(tiny_gemma2(True), t.float32)
    sae = make_tiny_sae("jumprelu").to("cuda")
    ids = seeded_ids(_DOC, seed=4)
    segs = fixed_segments(_DOC, _SEG)
    fids = [full_fids() for _ in segs]
    collector = _collector(model, sae, mode)
    a = _score_flat(collector, ids, segs, fids)
    b = _score_flat(collector, ids, segs, fids)
    diff = (a - b).abs().max().item()
    print(f"rerun max abs diff (atomics): {diff:.3e}")
    # 4090-measured 2026-06-10: bit-identical rerun (0.0); 1e-6 still
    # tolerates atomic-reduction noise on larger inputs (100x tighter).
    assert diff <= 1e-6
    k = min(16, int((a > 0).sum()))
    assert set(a.topk(k).indices.tolist()) == set(b.topk(k).indices.tolist())
