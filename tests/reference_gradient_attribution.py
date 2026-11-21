"""Reference gradient-attribution pipeline — simple, readable, unoptimized.

Oracle for the production implementation in
:mod:`steering.gradient_attribution`. Every function prioritizes clarity over
performance: plain Python loops, one fresh forward per backward unit, no
batching, no accumulation tricks. Property tests assert the production
collector matches these references — bit-exactly on CPU fp32 wherever the
spec pins the operation order, with documented tolerances elsewhere.

The spec quantity (pinned here and in production): a segment's attribution
gradient is the ORDERED SUM of per-unit gradients in the schedule order of
``build_backward_schedule``; the per-unit loss is the fp32-CE sum over the
unit's loss positions computed through the chunked lm_head tail
(``lm_head(hidden[lo:hi])`` + tanh softcap iff configured).

Fixtures: tiny randomly-initialized HF Gemma-2 / Gemma-3 models (the real
production architectures — differentiable, causal, softcap-bearing; no
downloads) and real ``sae_lens`` SAEs (standard + jumprelu) built through the
single :func:`make_tiny_sae` helper so a sae_lens API change is a one-place
fix (5.x used a monolithic ``SAEConfig``; the locked 6.x uses per-class
configs — this targets the installed 6.x surface).

Usage:
    uv run pytest tests/reference_gradient_attribution.py -v
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import hypothesis.strategies as st
import pytest
import torch as t
import torch.nn.functional as F
from hypothesis import given, settings
from sae_lens import JumpReLUSAE, JumpReLUSAEConfig, StandardSAE, StandardSAEConfig
from transformers import AutoModelForCausalLM
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from steering.gradient_attribution import (
    GradientAttributionCollector,
    build_backward_schedule,
    decoder_linearity_gate,
    group_units_by_key,
)
from steering.gradient_overlay import GradientAttributionMode

TINY_VOCAB = 96
TINY_HIDDEN = 32
TINY_LAYERS = 4
TINY_LAYER = 2  # the "SAE layer" used by the tests
TINY_D_SAE = 48
TINY_BOS = 2

ALL_MODES = (
    GradientAttributionMode.future_half_split,
    GradientAttributionMode.future_self_full,
    GradientAttributionMode.self_,
)


# ---------------------------------------------------------------------------
# Tiny real-model fixtures (cached; fp32 CPU; frozen)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def tiny_gemma2(softcap: bool = True):
    """Randomly-initialized tiny Gemma-2 (real architecture, no download)."""
    cfg = Gemma2Config(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=256,
        final_logit_softcapping=30.0 if softcap else None,
        bos_token_id=TINY_BOS,
    )
    t.manual_seed(7)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@functools.lru_cache(maxsize=2)
def tiny_gemma3():
    """Randomly-initialized tiny Gemma-3 text model (no final softcap).

    ``sliding_window_pattern=2`` makes ``layer_types`` INTERLEAVE
    sliding/full attention (with a window small enough to bind at test doc
    lengths) — the default uniform construction derives all-sliding layers,
    which never exercises gemma-3's production reality of mixed masks with
    distinct per-type rope (see tests/steering/native/conftest.py's
    ``_build_tiny_gemma3_mixed`` notes).
    """
    cfg = Gemma3TextConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=256,
        bos_token_id=TINY_BOS,
        sliding_window=8,
        sliding_window_pattern=2,  # ty: ignore[unknown-argument]
    )
    assert "full_attention" in set(cfg.layer_types) and (  # ty: ignore[invalid-argument-type]
        "sliding_attention" in set(cfg.layer_types)  # ty: ignore[invalid-argument-type]
    ), f"tiny gemma-3 must interleave attention types, got {cfg.layer_types}"
    t.manual_seed(11)
    model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa")
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_tiny_sae(
    architecture: str = "jumprelu",
    *,
    d_in: int = TINY_HIDDEN,
    d_sae: int = TINY_D_SAE,
    seed: int = 0,
    apply_b_dec_to_input: bool = True,
    device: str = "cpu",
):
    """Build a real sae_lens SAE with random weights — THE single construction
    helper (a sae_lens constructor change is fixed here only).

    Targets the installed sae_lens 6.x per-class configs. Weights are seeded:
    encoder/decoder ~ N(0, 1/sqrt(d)), biases small, and (for jumprelu) a
    strictly positive threshold so the jump genuinely differs from plain ReLU.
    """
    gen = t.Generator().manual_seed(seed)
    sae: StandardSAE | JumpReLUSAE
    if architecture == "standard":
        sae = StandardSAE(
            StandardSAEConfig(
                d_in=d_in,
                d_sae=d_sae,
                dtype="float32",
                device=device,
                apply_b_dec_to_input=apply_b_dec_to_input,
                normalize_activations="none",
            )
        )
    elif architecture == "jumprelu":
        jumprelu = JumpReLUSAE(
            JumpReLUSAEConfig(
                d_in=d_in,
                d_sae=d_sae,
                dtype="float32",
                device=device,
                apply_b_dec_to_input=apply_b_dec_to_input,
                normalize_activations="none",
            )
        )
        with t.no_grad():
            jumprelu.threshold.copy_(t.rand(d_sae, generator=gen) * 0.05 + 0.01)
        sae = jumprelu
    else:
        raise ValueError(f"unsupported tiny SAE architecture: {architecture}")
    with t.no_grad():
        sae.W_enc.copy_(t.randn(d_in, d_sae, generator=gen) / (d_in**0.5))
        sae.W_dec.copy_(t.randn(d_sae, d_in, generator=gen) / (d_sae**0.5))
        sae.b_enc.copy_(t.randn(d_sae, generator=gen) * 0.05)
        sae.b_dec.copy_(t.randn(d_in, generator=gen) * 0.05)
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae


def seeded_ids(n: int, seed: int = 0, bos: bool = True) -> list[int]:
    """A deterministic token-id document of length n (BOS first by default)."""
    gen = t.Generator().manual_seed(seed)
    ids = t.randint(3, TINY_VOCAB, (n,), generator=gen).tolist()
    if bos and n > 0:
        ids[0] = TINY_BOS
    return ids


def fixed_segments(n: int, seg_size: int) -> list[tuple[int, int]]:
    """Reference fixed-size tiling (mirrors steering.segmentation)."""
    segs = []
    start = 0
    while start < n:
        segs.append((start, min(start + seg_size, n)))
        start += seg_size
    return segs


def full_fids(d_sae: int = TINY_D_SAE) -> t.Tensor:
    return t.arange(d_sae, dtype=t.int64)


# ---------------------------------------------------------------------------
# Reference schedule — re-derived from the prose spec by position FILTERING
# ---------------------------------------------------------------------------


def ref_positions_predicting(
    token_lo: int, token_hi: int, doc_len: int
) -> list[int]:
    """Loss positions whose TARGET token falls in ``[token_lo, token_hi)``,
    derived by filtering every valid loss position (independent of the
    production range arithmetic)."""
    return [
        j
        for j in range(doc_len - 1)
        if token_lo <= j + 1 < token_hi
    ]


def ref_final_unit(
    segments: list[tuple[int, int]], min_final: int
) -> tuple[int, list[int]]:
    """(u, unit_seg_indices) per the prose merge rule: merge only when a
    second-to-last segment exists and the final is short; a single-segment
    doc is its own unit. ``min_final >= 2`` (enforced upstream) guarantees a
    1-token final always merges, so every final unit is >= 2 tokens."""
    n_segs = len(segments)
    b_f, e_f = segments[-1]
    if n_segs >= 2 and (e_f - b_f) < min_final:
        return segments[-2][0], [n_segs - 2, n_segs - 1]
    return b_f, [n_segs - 1]


def ref_backward_schedule(
    segments: list[tuple[int, int]],
    doc_len: int,
    mode: GradientAttributionMode,
    min_final: int,
) -> list[dict]:
    """The spec as data, independently re-derived. Returns ``units``; each
    unit is a dict ``{key, positions (list), reads (list of (seg, row_lo,
    row_hi)), reset}``. Every emitted unit has non-empty positions and only
    non-empty read windows — the inputs are assumed valid (doc_len >= 2,
    segment_size >= 2, min_final >= 2), matching production which RAISES
    otherwise (so parity tests for degenerate inputs assert the raise, not
    this function)."""
    e_doc = doc_len
    u, unit_segs = ref_final_unit(segments, min_final)
    first_unit_seg = unit_segs[0]

    if mode is GradientAttributionMode.self_:
        units = []
        for m in range(first_unit_seg):
            b, e = segments[m]
            units.append(
                {
                    "key": ("self", m),
                    "positions": ref_positions_predicting(b, e, doc_len),
                    "reads": [(m, b, e)],
                    "reset": True,
                }
            )
        units.append(
            {
                "key": ("self", first_unit_seg),
                "positions": ref_positions_predicting(u, e_doc, doc_len),
                "reads": [(s, u, e_doc) for s in unit_segs],
                "reset": True,
            }
        )
        return units

    below_read = (
        [(first_unit_seg - 1, *segments[first_unit_seg - 1])]
        if first_unit_seg >= 1
        else []
    )
    if mode is GradientAttributionMode.future_half_split:
        src_lo = u + (e_doc - u) // 2
        read_hi = src_lo
    else:
        src_lo = u
        read_hi = e_doc
    final_reads = [(s, u, read_hi) for s in unit_segs]
    phase1_positions = ref_positions_predicting(u, src_lo, doc_len)
    if phase1_positions and below_read:
        units = [
            {
                "key": ("final", 0),
                "positions": ref_positions_predicting(src_lo, e_doc, doc_len),
                "reads": final_reads,
                "reset": False,
            },
            {
                "key": ("final", 1),
                "positions": phase1_positions,
                "reads": below_read,
                "reset": False,
            },
        ]
    else:
        # Phase 1 has nothing to backward (self_full) or no consumer below:
        # the below-segment read rides on phase 0.
        units = [
            {
                "key": ("final", 0),
                "positions": ref_positions_predicting(src_lo, e_doc, doc_len),
                "reads": final_reads + below_read,
                "reset": False,
            },
        ]
    for m in range(first_unit_seg - 1, 0, -1):
        b, e = segments[m]
        units.append(
            {
                "key": ("chunk", m),
                "positions": ref_positions_predicting(b, e, doc_len),
                "reads": [(m - 1, *segments[m - 1])],
                "reset": False,
            }
        )
    return units


# ---------------------------------------------------------------------------
# Reference gradients and scores — fresh forward per unit, explicit loops
# ---------------------------------------------------------------------------


def ref_capture(model, ids: list[int], layer: int):
    """One unbatched forward; returns (hidden, capt) with capt a leaf root.

    Independent re-implementation of the capture mechanism (not imported
    from production).
    """
    captured = []

    def _hook(_m, _a, output):
        h = output[0] if isinstance(output, tuple) else output
        h.requires_grad_(True)
        captured.append(h)

    handle = model.model.layers[layer].register_forward_hook(_hook)
    try:
        device = next(model.parameters()).device
        input_ids = t.tensor(ids, dtype=t.long, device=device).unsqueeze(0)
        hidden = model.model(
            input_ids=input_ids, attention_mask=None, use_cache=False
        ).last_hidden_state
    finally:
        handle.remove()
    assert len(captured) == 1
    return hidden, captured[0]


def ref_unit_loss(model, hidden: t.Tensor, ids: list[int], positions: list[int]) -> t.Tensor:
    """fp32-CE sum over the unit's loss positions via the chunked lm_head
    tail. Positions must be the contiguous range the schedule prescribes."""
    assert positions == list(range(positions[0], positions[-1] + 1))
    lo, hi = positions[0], positions[-1] + 1
    logits = model.lm_head(hidden[0, lo:hi])
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap is not None:
        logits = t.tanh(logits / cap) * cap
    # On the logits' device: the CUDA gates pass CUDA hidden states (4090
    # report §4.2 — a CPU targets tensor crashed nll_loss there).
    targets = t.tensor(ids[lo + 1 : hi + 1], dtype=t.long, device=logits.device)
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def ref_unit_gradient(model, ids: list[int], layer: int, positions: list[int]) -> t.Tensor:
    """Gradient of one unit's loss at the layer capture — FRESH forward,
    ``torch.autograd.grad`` (no .backward, no accumulation). Returns [T, d];
    an empty position list yields zeros."""
    hidden, capt = ref_capture(model, ids, layer)
    if not positions:
        return t.zeros_like(capt[0])
    loss = ref_unit_loss(model, hidden, ids, positions)
    (g,) = t.autograd.grad(loss, capt)
    return g[0]


def ref_grad_rows_to_scores(
    grad_rows: t.Tensor,
    resid_rows: t.Tensor,
    sae,
    fids: t.Tensor,
    keep_rows: t.Tensor | None = None,
) -> t.Tensor:
    """Variant (i): order-identical SENSITIVITY scoring (gate * relu(-g)
    row-max), fp32. The producer separately multiplies by the activation
    percentile to form the stored pct*sensitivity score."""
    if fids.numel() == 0:
        return t.zeros(0)
    if grad_rows.shape[0] == 0:
        return t.zeros(fids.numel())
    g = grad_rows.to(t.float32) @ sae.W_dec.index_select(0, fids).to(t.float32).T
    with t.no_grad():
        acts = sae.encode(resid_rows.detach())
    gate = acts.index_select(1, fids) > 0
    vals = t.relu(-g).masked_fill(~gate, 0.0)
    if keep_rows is not None:
        vals = vals.masked_fill(~keep_rows.unsqueeze(1), 0.0)
    return vals.amax(dim=0)


def ref_grad_rows_to_scores_via_decode(
    grad_rows: t.Tensor,
    resid_rows: t.Tensor,
    sae,
    fids: t.Tensor,
    keep_rows: t.Tensor | None = None,
) -> t.Tensor:
    """Variant (ii): independent math — per (row, feature) the decoder VJP is
    computed by autograd THROUGH ``sae.decode`` in float64 loops. Tolerance
    comparator (reduction order differs from the matmul)."""
    n_fids = int(fids.numel())
    if n_fids == 0:
        return t.zeros(0, dtype=t.float64)
    n_rows = grad_rows.shape[0]
    if n_rows == 0:
        return t.zeros(n_fids, dtype=t.float64)
    with t.no_grad():
        acts = sae.encode(resid_rows.detach())
    scores = t.zeros(n_fids, dtype=t.float64)
    sae64 = sae.to(t.float64)
    try:
        for r in range(n_rows):
            if keep_rows is not None and not bool(keep_rows[r]):
                continue
            cot = grad_rows[r].to(t.float64)
            z = t.zeros(sae64.W_dec.shape[0], dtype=t.float64, requires_grad=True)
            (vjp,) = t.autograd.grad((sae64.decode(z) * cot).sum(), z)
            for i, fid in enumerate(fids.tolist()):
                if acts[r, fid] > 0:
                    val = max(0.0, -float(vjp[fid]))
                    scores[i] = max(float(scores[i]), val)
    finally:
        sae64.to(t.float32)
    return scores


def ref_score_document(
    model,
    sae,
    ids: list[int],
    segments: list[tuple[int, int]],
    fids_per_seg: list[t.Tensor],
    mode: GradientAttributionMode,
    min_final: int,
    layer: int = TINY_LAYER,
    *,
    bos_token_id: int | None = None,
    ignore_bos: bool = False,
) -> list[t.Tensor]:
    """Reference per-document scores: fresh forward per unit, fold-summed in
    schedule order (future modes) or isolated (self mode), then variant-(i)
    scoring. Returns fp16 rows like production. Inputs are assumed valid
    (every segment scoreable); production raises otherwise."""
    sched = ref_backward_schedule(segments, len(ids), mode, min_final)
    _, capt = ref_capture(model, ids, layer)
    resid = capt.detach()
    ids_t = t.tensor(ids, dtype=t.long)
    acc = t.zeros_like(resid[0])
    out: list[t.Tensor | None] = [None] * len(segments)
    for unit in sched:
        g_unit = ref_unit_gradient(model, ids, layer, unit["positions"])
        if unit["reset"]:
            acc = g_unit
        else:
            acc = acc + g_unit
        for seg_idx, row_lo, row_hi in unit["reads"]:
            keep = None
            if ignore_bos:
                assert bos_token_id is not None
                keep = ids_t[row_lo:row_hi] != int(bos_token_id)
            score = ref_grad_rows_to_scores(
                acc[row_lo:row_hi],
                resid[0, row_lo:row_hi],
                sae,
                fids_per_seg[seg_idx],
                keep_rows=keep,
            )
            out[seg_idx] = score.to(t.float16)
    final: list[t.Tensor] = []
    for row in out:
        assert row is not None
        final.append(row)
    return final


def run_collector(
    model,
    sae,
    docs: list[list[int]],
    doc_segments: list[list[tuple[int, int]]],
    doc_fids: list[list[t.Tensor]],
    mode: GradientAttributionMode,
    min_final: int,
    *,
    layer: int = TINY_LAYER,
    bos_token_id: int | None = None,
    ignore_bos: bool = False,
) -> list[list[t.Tensor]]:
    collector = GradientAttributionCollector(
        model,
        sae,
        layer=layer,
        mode=mode,
        min_final_segment_tokens=min_final,
        bos_token_id=bos_token_id,
        ignore_bos_features=ignore_bos,
    )
    return collector.score_documents(docs, doc_segments, doc_fids)


@dataclass(frozen=True)
class AttributionExecutionTrace:
    """Test-only raw execution witness for GA-011."""

    unit_trace: list[list[tuple]]
    accumulated_grads: list[list[t.Tensor]]
    final_rows: list[list[t.Tensor]]


def _unit_trace_entry(unit) -> tuple:
    return (
        unit.key,
        unit.reset_grad_before,
        tuple((read.seg_idx, read.row_lo, read.row_hi) for read in unit.reads),
    )


class _TracingGradientAttributionCollector(GradientAttributionCollector):
    """Record the grouped collector's raw accumulation at every unit read."""

    def configure_trace(
        self,
        schedules,
        lengths: list[int],
    ) -> list[list[tuple]]:
        unit_trace: list[list[tuple]] = [[] for _ in schedules]
        pending: list[tuple[int, int, object]] = []
        for _key, doc_units in group_units_by_key(schedules):
            for doc_idx, unit in doc_units:
                unit_idx = len(unit_trace[doc_idx])
                unit_trace[doc_idx].append(_unit_trace_entry(unit))
                pending.extend((doc_idx, unit_idx, read) for read in unit.reads)
        self._trace_pending = pending
        self._trace_cursor = 0
        self._trace_lengths = lengths
        self._trace_accumulated: list[list[t.Tensor]] = [
            [] for _ in schedules
        ]
        return unit_trace

    def _score_read(self, capt, capt_grad, input_ids, d, read, fids):
        expected_d, unit_idx, expected_read = self._trace_pending[
            self._trace_cursor
        ]
        self._trace_cursor += 1
        assert d == expected_d and read == expected_read
        assert capt_grad is not None
        observed = self._trace_accumulated[d]
        if len(observed) == unit_idx:
            observed.append(
                capt_grad[d, : self._trace_lengths[d]].detach().float().cpu()
            )
        else:
            assert len(observed) == unit_idx + 1  # another read of this unit
        return super()._score_read(capt, capt_grad, input_ids, d, read, fids)


def run_grouped_collector_trace(
    model,
    sae,
    docs: list[list[int]],
    doc_segments: list[list[tuple[int, int]]],
    doc_fids: list[list[t.Tensor]],
    mode: GradientAttributionMode,
    min_final: int,
    *,
    layer: int = TINY_LAYER,
) -> AttributionExecutionTrace:
    """Execute the real grouped collector while recording its raw fold state."""
    schedules = [
        build_backward_schedule(doc_segments[d], len(docs[d]), mode, min_final)
        for d in range(len(docs))
    ]
    collector = _TracingGradientAttributionCollector(
        model,
        sae,
        layer=layer,
        mode=mode,
        min_final_segment_tokens=min_final,
    )
    unit_trace = collector.configure_trace(schedules, [len(ids) for ids in docs])
    final_rows = collector.score_documents(docs, doc_segments, doc_fids)
    assert collector._trace_cursor == len(collector._trace_pending)
    return AttributionExecutionTrace(
        unit_trace=unit_trace,
        accumulated_grads=collector._trace_accumulated,
        final_rows=final_rows,
    )


def run_independent_document_trace(
    model,
    sae,
    docs: list[list[int]],
    doc_segments: list[list[tuple[int, int]]],
    doc_fids: list[list[t.Tensor]],
    mode: GradientAttributionMode,
    min_final: int,
    *,
    layer: int = TINY_LAYER,
) -> AttributionExecutionTrace:
    """Independent fresh-forward, per-document VJPs with explicit resets."""
    all_unit_traces: list[list[tuple]] = []
    all_accumulated: list[list[t.Tensor]] = []
    all_final_rows: list[list[t.Tensor]] = []
    for ids, segments, fids_per_seg in zip(
        docs, doc_segments, doc_fids, strict=True
    ):
        schedule = ref_backward_schedule(segments, len(ids), mode, min_final)
        _, capt = ref_capture(model, ids, layer)
        resid = capt.detach()
        accumulated: t.Tensor | None = None
        unit_trace: list[tuple] = []
        raw_grads: list[t.Tensor] = []
        rows: list[t.Tensor | None] = [None] * len(segments)
        for unit in schedule:
            g_unit = ref_unit_gradient(model, ids, layer, unit["positions"])
            if accumulated is None or unit["reset"]:
                accumulated = g_unit
            else:
                accumulated = accumulated + g_unit
            unit_trace.append(
                (
                    unit["key"],
                    unit["reset"],
                    tuple(unit["reads"]),
                )
            )
            raw_grads.append(accumulated.detach().float().cpu())
            for seg_idx, row_lo, row_hi in unit["reads"]:
                fids = fids_per_seg[seg_idx].to(
                    device=accumulated.device, dtype=t.int64
                )
                rows[seg_idx] = ref_grad_rows_to_scores(
                    accumulated[row_lo:row_hi],
                    resid[0, row_lo:row_hi],
                    sae,
                    fids,
                ).to(t.float16).cpu()
        final_rows: list[t.Tensor] = []
        for row in rows:
            assert row is not None
            final_rows.append(row)
        all_unit_traces.append(unit_trace)
        all_accumulated.append(raw_grads)
        all_final_rows.append(final_rows)
    return AttributionExecutionTrace(
        unit_trace=all_unit_traces,
        accumulated_grads=all_accumulated,
        final_rows=all_final_rows,
    )


def _norm_production_schedule(units) -> list[dict]:
    return [
        {
            "key": u.key,
            "positions": list(range(u.loss_lo, max(u.loss_lo, u.loss_hi))),
            "reads": [(r.seg_idx, r.row_lo, r.row_hi) for r in u.reads],
            "reset": u.reset_grad_before,
        }
        for u in units
    ]


# ---------------------------------------------------------------------------
# Schedule spec equality + structural properties
# ---------------------------------------------------------------------------

DOC_SHAPES = [
    # (doc_len, seg_size) — all VALID (doc_len >= 2, seg_size >= 2): exact
    # multiple, short tail (merges), single segment, tiny doc, finer tiling.
    (24, 8),
    (17, 8),
    (8, 8),
    (9, 8),
    (5, 8),
    (2, 8),
    (13, 4),
]
# min_final >= 2 (the producer's floor); 2 merges only 1-token finals.
MIN_FINALS = [2, 4, 8]


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("min_final", MIN_FINALS)
@pytest.mark.parametrize("doc_len,seg_size", DOC_SHAPES)
def test_schedule_matches_reference(mode, min_final, doc_len, seg_size):
    segs = fixed_segments(doc_len, seg_size)
    prod_units = build_backward_schedule(segs, doc_len, mode, min_final)
    ref_units = ref_backward_schedule(segs, doc_len, mode, min_final)
    assert _norm_production_schedule(prod_units) == ref_units


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("min_final", MIN_FINALS)
@pytest.mark.parametrize("doc_len,seg_size", DOC_SHAPES)
def test_schedule_structural_invariants(mode, min_final, doc_len, seg_size):
    """Disjoint loss positions; no emitted unit is empty; no read window is
    empty; reads partition the segments EXACTLY (every segment scored — no
    zero placeholders). Future modes cover every prediction from the first
    suffix-serving chunk; self mode covers all of [0, n-2]."""
    segs = fixed_segments(doc_len, seg_size)
    units = ref_backward_schedule(segs, doc_len, mode, min_final)

    for u in units:
        assert u["positions"], f"unit {u['key']} has an empty loss range"
        for r in u["reads"]:
            assert r[2] > r[1], f"unit {u['key']} has an empty read window"

    all_positions: list[int] = []
    for u in units:
        all_positions.extend(u["positions"])
    assert len(all_positions) == len(set(all_positions)), "positions overlap"

    u_start, unit_segs = ref_final_unit(segs, min_final)
    if mode is GradientAttributionMode.self_:
        expected = set(range(doc_len - 1))
    elif unit_segs[0] >= 1:
        # Future modes with consumers below the final unit: every prediction
        # from the first chunk that serves a suffix — targets in [b_1, n)
        # plus the final unit's own window (which begins at u_start <= b_1
        # when merged).
        first_served = min(segs[1][0], u_start)
        expected = set(ref_positions_predicting(first_served, doc_len, doc_len))
    else:
        # The final unit spans the whole doc (single segment, or a merge
        # reaching the doc head): only its SOURCE window is backwarded
        # (phase 1 has no consumer and is not emitted).
        src = (
            u_start + (doc_len - u_start) // 2
            if mode is GradientAttributionMode.future_half_split
            else u_start
        )
        expected = set(ref_positions_predicting(src, doc_len, doc_len))
    assert set(all_positions) == expected

    reads = [r for u in units for r in u["reads"]]
    assert sorted(r[0] for r in reads) == list(range(len(segs)))
    if mode is not GradientAttributionMode.self_ and unit_segs[0] >= 0:
        # All final-unit segments share the unit-wide read window.
        windows = {(r[1], r[2]) for u in units for r in u["reads"] if r[0] in unit_segs}
        assert len(windows) == 1


def test_group_units_preserves_per_doc_order():
    """Merging schedules across ragged docs must keep each doc's own unit
    order (the ordered-sum spec depends on it)."""
    mode = GradientAttributionMode.future_half_split
    docs = [(24, 8), (17, 8), (5, 8), (9, 4)]
    schedules = [
        build_backward_schedule(fixed_segments(n, s), n, mode, 4)
        for n, s in docs
    ]
    grouped = group_units_by_key(schedules)
    seen: dict[int, list[int]] = {d: [] for d in range(len(docs))}
    for _key, doc_units in grouped:
        for d, unit in doc_units:
            seen[d].append(schedules[d].index(unit))
    for d, order in seen.items():
        assert order == sorted(order), f"doc {d} units out of order: {order}"
        assert order == list(range(len(schedules[d])))


# ---------------------------------------------------------------------------
# Production == reference (bit-exact on CPU fp32)
# ---------------------------------------------------------------------------

PARITY_CASES = [
    # (doc_len, seg_size, min_final) — all VALID: normal, merged-final
    # (min_final=2 merges only the 1-token final), bigger merge, single
    # short doc, exact multiple, a 2-token final that does NOT merge.
    (24, 8, 2),
    (26, 8, 2),
    (26, 8, 8),
    (16, 8, 8),
    (5, 8, 8),
    (10, 8, 2),
]


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("arch", ["standard", "jumprelu"])
@pytest.mark.parametrize("doc_len,seg_size,min_final", PARITY_CASES)
def test_production_matches_reference_exact(mode, arch, doc_len, seg_size, min_final):
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae(arch)
    ids = seeded_ids(doc_len, seed=doc_len * 7 + seg_size)
    segs = fixed_segments(doc_len, seg_size)
    fids = [full_fids() for _ in segs]

    prod = run_collector(model, sae, [ids], [segs], [fids], mode, min_final)[0]
    ref = ref_score_document(model, sae, ids, segs, fids, mode, min_final)

    for s, (p, r) in enumerate(zip(prod, ref, strict=True)):
        assert t.equal(p, r), (
            f"segment {s}: production != reference "
            f"(max abs diff {(p.float() - r.float()).abs().max().item():.3e})"
        )


@pytest.mark.parametrize("softcap", [True, False])
def test_parity_softcap_variants(softcap):
    """Softcap on/off both match the reference (and produce different scores
    from each other — the softcap is genuinely applied)."""
    mode = GradientAttributionMode.future_half_split
    sae = make_tiny_sae("jumprelu")
    ids = seeded_ids(20, seed=3)
    segs = fixed_segments(20, 8)
    fids = [full_fids() for _ in segs]
    model = tiny_gemma2(softcap=softcap)
    prod = run_collector(model, sae, [ids], [segs], [fids], mode, 8)[0]
    ref = ref_score_document(model, sae, ids, segs, fids, mode, 8)
    for p, r in zip(prod, ref, strict=True):
        assert t.equal(p, r)


def test_softcap_changes_losses():
    """Non-vacuity at the fp32 loss level: the softcap genuinely enters the
    loss tail. (With tiny random weights |logits| ~ 0.4, the tanh correction
    is ~1e-5 relative — visible in fp32 losses but typically below fp16
    score resolution, so the assertion lives at the loss level.)"""
    ids = seeded_ids(20, seed=3)
    positions = list(range(8, 15))
    losses = {}
    for cap in (True, False):
        model = tiny_gemma2(softcap=cap)
        hidden, _ = ref_capture(model, ids, TINY_LAYER)
        losses[cap] = float(ref_unit_loss(model, hidden, ids, positions))
    assert losses[True] != losses[False], (
        "identical weights with and without final_logit_softcapping produced "
        "identical fp32 losses — the softcap is not being applied (vacuous)"
    )


def _lm_head_row_subset_probe(model) -> bool:
    """Platform probe: is ``lm_head(H)[rows] == lm_head(H[rows])`` bit-exact?

    GEMM M-dimension blocking may change per-row reduction order; on
    platforms where it does, the chunked-tail-vs-full-logits contract is
    tolerance-bounded instead of bit-exact (the production<->oracle contract
    is unaffected: both sides use the same chunked slicing)."""
    ids = seeded_ids(24, seed=99)
    h = model.model(
        input_ids=t.tensor(ids).unsqueeze(0), attention_mask=None, use_cache=False
    ).last_hidden_state
    return t.equal(model.lm_head(h)[0, 4:12], model.lm_head(h[0, 4:12]))


@pytest.mark.parametrize("softcap", [True, False])
def test_chunked_tail_vs_full_forward_logits(softcap):
    """The production loss tail (model.model -> chunked lm_head -> softcap)
    reproduces ``model(input_ids, use_cache=False).logits``: bit-exactly when
    the platform's GEMM is row-subset invariant (probe), else within a 1-2
    ulp envelope (measured and printed)."""
    model = tiny_gemma2(softcap=softcap)
    ids = seeded_ids(24, seed=42)
    input_ids = t.tensor(ids).unsqueeze(0)
    full = model(input_ids=input_ids, use_cache=False).logits
    hidden = model.model(
        input_ids=input_ids, attention_mask=None, use_cache=False
    ).last_hidden_state
    lo, hi = 4, 12
    chunk = model.lm_head(hidden[0, lo:hi])
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap is not None:
        chunk = t.tanh(chunk / cap) * cap
    diff = (chunk - full[0, lo:hi]).abs().max().item()
    print(
        f"chunked-tail vs full logits max abs diff: {diff:.3e} "
        f"(row-subset probe bit-exact: {_lm_head_row_subset_probe(model)})"
    )
    if _lm_head_row_subset_probe(model):
        assert t.equal(chunk, full[0, lo:hi])
    else:
        assert diff <= 1e-6, (
            f"chunked tail diverges from the full forward beyond the GEMM "
            f"blocking envelope: {diff:.3e}"
        )


def test_gemma3_parity():
    """Gemma-3 (no final softcap attribute) through the same contract."""
    model = tiny_gemma3()
    assert getattr(model.config, "final_logit_softcapping", None) is None
    sae = make_tiny_sae("standard")
    ids = seeded_ids(19, seed=5)
    segs = fixed_segments(19, 8)
    fids = [full_fids() for _ in segs]
    for mode in ALL_MODES:
        prod = run_collector(model, sae, [ids], [segs], [fids], mode, 8)[0]
        ref = ref_score_document(model, sae, ids, segs, fids, mode, 8)
        for p, r in zip(prod, ref, strict=True):
            assert t.equal(p, r)


def test_parity_with_subset_fids_and_ignore_bos():
    """Random sorted fid subsets + BOS keep-mask composition."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    ids = seeded_ids(22, seed=9)
    segs = fixed_segments(22, 8)
    gen = t.Generator().manual_seed(13)
    fids = [
        t.sort(t.randperm(TINY_D_SAE, generator=gen)[:17]).values.to(t.int64)
        for _ in segs
    ]
    for mode in ALL_MODES:
        prod = run_collector(
            model, sae, [ids], [segs], [fids], mode, 8,
            bos_token_id=TINY_BOS, ignore_bos=True,
        )[0]
        ref = ref_score_document(
            model, sae, ids, segs, fids, mode, 8,
            bos_token_id=TINY_BOS, ignore_bos=True,
        )
        for p, r in zip(prod, ref, strict=True):
            assert t.equal(p, r)


def test_scores_match_decode_oracle():
    """Variant (ii): independent fp64 autograd-through-decode oracle, with
    tolerance (different reduction order than the matmul path)."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    ids = seeded_ids(17, seed=21)
    segs = fixed_segments(17, 8)
    fids = [full_fids() for _ in segs]
    mode = GradientAttributionMode.future_half_split
    prod = run_collector(model, sae, [ids], [segs], [fids], mode, 8)[0]

    sched = ref_backward_schedule(segs, len(ids), mode, 8)
    _, capt = ref_capture(model, ids, TINY_LAYER)
    resid = capt.detach()
    acc = t.zeros_like(resid[0])
    for unit in sched:
        acc = acc + ref_unit_gradient(model, ids, TINY_LAYER, unit["positions"])
        for seg_idx, row_lo, row_hi in unit["reads"]:
            oracle = ref_grad_rows_to_scores_via_decode(
                acc[row_lo:row_hi], resid[0, row_lo:row_hi], sae, fids[seg_idx]
            )
            assert t.allclose(
                prod[seg_idx].to(t.float64), oracle, rtol=1e-2, atol=1e-4
            ), f"segment {seg_idx} diverges from the decode oracle"


# ---------------------------------------------------------------------------
# Cross-mode identities (reference-free internal consistency)
# ---------------------------------------------------------------------------


def test_future_self_full_final_unit_equals_self_mode():
    """The doc-final unit is the FIRST future-mode unit (nothing accumulated
    before it), so under future_self_full its scores must equal self mode's
    for the same unit — bit-exactly."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    for doc_len, seg_size, min_final in [(24, 8, 0), (26, 8, 8), (8, 8, 8)]:
        ids = seeded_ids(doc_len, seed=doc_len)
        segs = fixed_segments(doc_len, seg_size)
        fids = [full_fids() for _ in segs]
        fut = run_collector(
            model, sae, [ids], [segs], [fids],
            GradientAttributionMode.future_self_full, min_final,
        )[0]
        slf = run_collector(
            model, sae, [ids], [segs], [fids],
            GradientAttributionMode.self_, min_final,
        )[0]
        _, unit_segs = ref_final_unit(segs, min_final)
        for s in unit_segs:
            assert t.equal(fut[s], slf[s]), f"final-unit segment {s} differs"


def test_future_accumulation_equals_fold_of_self_chunks():
    """Reference-free accumulation-order check: a future-mode segment's
    gradient equals the ordered fold of the per-unit gradients (computed in
    isolation) — pinned at the raw gradient level via the reference helpers,
    then end-to-end via production scores."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("standard")
    doc_len, seg_size, min_final = 26, 8, 8
    ids = seeded_ids(doc_len, seed=2)
    segs = fixed_segments(doc_len, seg_size)
    fids = [full_fids() for _ in segs]
    mode = GradientAttributionMode.future_half_split

    sched = ref_backward_schedule(segs, doc_len, mode, min_final)
    _, capt = ref_capture(model, ids, TINY_LAYER)
    resid = capt.detach()
    acc = t.zeros_like(resid[0])
    expected: list[t.Tensor | None] = [None] * len(segs)
    for unit in sched:
        acc = acc + ref_unit_gradient(model, ids, TINY_LAYER, unit["positions"])
        for seg_idx, row_lo, row_hi in unit["reads"]:
            expected[seg_idx] = ref_grad_rows_to_scores(
                acc[row_lo:row_hi], resid[0, row_lo:row_hi], sae, fids[seg_idx]
            ).to(t.float16)
    prod = run_collector(model, sae, [ids], [segs], [fids], mode, min_final)[0]
    for s, (p, e) in enumerate(zip(prod, expected, strict=True)):
        assert e is not None and t.equal(p, e), f"segment {s} fold mismatch"


def test_merge_locality():
    """Turning the merge on must not change any segment outside the merged
    unit — bit-exactly."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    doc_len, seg_size = 26, 8  # final segment length 2
    ids = seeded_ids(doc_len, seed=4)
    segs = fixed_segments(doc_len, seg_size)
    fids = [full_fids() for _ in segs]
    for mode in ALL_MODES:
        unmerged = run_collector(model, sae, [ids], [segs], [fids], mode, 0)[0]
        merged = run_collector(model, sae, [ids], [segs], [fids], mode, 8)[0]
        for s in range(len(segs) - 2):
            assert t.equal(unmerged[s], merged[s]), (
                f"{mode.value}: merge changed non-merged segment {s}"
            )
        assert not t.equal(unmerged[-1], merged[-1]) or not t.equal(
            unmerged[-2], merged[-2]
        ), f"{mode.value}: merge had no effect on the merged unit (vacuous)"


# ---------------------------------------------------------------------------
# Batching: padding + permutation invariance (tolerance contract)
# ---------------------------------------------------------------------------


def _flatten_scores(rows: list[t.Tensor]) -> t.Tensor:
    return t.cat([r.float() for r in rows])


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda mode: mode.value)
def test_grouped_ragged_trace_matches_independent_per_document_vjps(mode):
    """GA-011 CPU oracle: raw folds, resets, and finals across ragged docs."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    lengths = [26, 9, 17, 5]
    docs = [seeded_ids(n, seed=100 + n) for n in lengths]
    seg_lists = [fixed_segments(n, 8) for n in lengths]
    fid_lists = [[full_fids() for _ in segs] for segs in seg_lists]

    grouped = run_grouped_collector_trace(
        model, sae, docs, seg_lists, fid_lists, mode, 8
    )
    independent = run_independent_document_trace(
        model, sae, docs, seg_lists, fid_lists, mode, 8
    )

    key_sets = [{unit[0] for unit in trace} for trace in grouped.unit_trace]
    assert len({frozenset(keys) for keys in key_sets}) > 1  # missing global keys
    assert any(
        len(segs) >= 2 and segs[-1][1] - segs[-1][0] < 8
        for segs in seg_lists
    )  # merged final exists
    if mode is GradientAttributionMode.self_:
        assert any(len(trace) > 1 for trace in grouped.unit_trace)
        assert all(unit[1] for trace in grouped.unit_trace for unit in trace)
    else:
        assert all(not unit[1] for trace in grouped.unit_trace for unit in trace)

    assert grouped.unit_trace == independent.unit_trace
    for doc_idx in range(len(docs)):
        assert len(grouped.accumulated_grads[doc_idx]) == len(
            independent.accumulated_grads[doc_idx]
        )
        for got, expected in zip(
            grouped.accumulated_grads[doc_idx],
            independent.accumulated_grads[doc_idx],
            strict=True,
        ):
            assert t.allclose(got, expected, rtol=1e-3, atol=1e-5)
        for got, expected in zip(
            grouped.final_rows[doc_idx], independent.final_rows[doc_idx], strict=True
        ):
            assert t.equal(got, expected)


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_padding_and_permutation_invariance(mode):
    """A doc's scores in a ragged right-padded batch match the doc scored
    alone. Maskless-causal forward makes this exact in exact arithmetic; in
    fp32 only batched-vs-single kernel reduction-order noise remains (same
    contract as the native collector's documented fp32 envelope). Measured
    and printed before asserting."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    lengths = [26, 9, 17, 5]
    docs = [seeded_ids(n, seed=n) for n in lengths]
    seg_lists = [fixed_segments(n, 8) for n in lengths]
    fid_lists = [[full_fids() for _ in segs] for segs in seg_lists]

    batched = run_collector(model, sae, docs, seg_lists, fid_lists, mode, 8)
    singles = [
        run_collector(model, sae, [docs[d]], [seg_lists[d]], [fid_lists[d]], mode, 8)[0]
        for d in range(len(docs))
    ]
    max_diff = 0.0
    for d in range(len(docs)):
        diff = (
            (_flatten_scores(batched[d]) - _flatten_scores(singles[d]))
            .abs()
            .max()
            .item()
        )
        max_diff = max(max_diff, diff)
    print(f"[{mode.value}] batched-vs-single max abs score diff: {max_diff:.3e}")
    for d in range(len(docs)):
        assert t.allclose(
            _flatten_scores(batched[d]), _flatten_scores(singles[d]),
            rtol=1e-3, atol=1e-5,
        ), f"doc {d} exceeds the batched-vs-single envelope"

    perm = [2, 0, 3, 1]
    permuted = run_collector(
        model, sae,
        [docs[i] for i in perm],
        [seg_lists[i] for i in perm],
        [fid_lists[i] for i in perm],
        mode, 8,
    )
    for pos, d in enumerate(perm):
        assert t.allclose(
            _flatten_scores(permuted[pos]), _flatten_scores(batched[d]),
            rtol=1e-3, atol=1e-5,
        ), f"doc {d} changed under batch permutation"


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------


@settings(deadline=None, max_examples=30)
@given(
    doc_len=st.integers(min_value=2, max_value=40),
    seg_size=st.integers(min_value=2, max_value=10),
    min_final=st.integers(min_value=2, max_value=10),
    mode=st.sampled_from(ALL_MODES),
)
def test_property_schedule_spec(doc_len, seg_size, min_final, mode):
    # Valid-input domain (the producer's guards: doc_len/seg_size/min_final
    # all >= 2). Production == oracle, both returning just the unit list.
    segs = fixed_segments(doc_len, seg_size)
    prod_units = build_backward_schedule(segs, doc_len, mode, min_final)
    ref_units = ref_backward_schedule(segs, doc_len, mode, min_final)
    assert _norm_production_schedule(prod_units) == ref_units


@settings(deadline=None, max_examples=40)
@given(
    doc_len=st.integers(min_value=1, max_value=6),
    seg_size=st.integers(min_value=1, max_value=4),
    min_final=st.integers(min_value=0, max_value=4),
    mode=st.sampled_from(ALL_MODES),
)
def test_property_schedule_raises_or_fully_valid(doc_len, seg_size, min_final, mode):
    """The hard invariant over arbitrary (incl. degenerate) inputs:
    ``build_backward_schedule`` EITHER raises ``GradientAttributionError``
    (structurally unscoreable input) OR returns units that are fully valid —
    no empty loss range, no empty read window, and every segment read EXACTLY
    once. There is no third outcome (no silent zero placeholder, no dropped
    segment)."""
    from errors import GradientAttributionError

    segs = fixed_segments(doc_len, seg_size)
    try:
        units = build_backward_schedule(segs, doc_len, mode, min_final)
    except GradientAttributionError:
        return  # unscoreable input rejected loud — the contract
    read_segs: list[int] = []
    for u in units:
        assert u.loss_hi > u.loss_lo, f"unit {u.key} empty loss range"
        for r in u.reads:
            assert r.row_hi > r.row_lo, f"unit {u.key} empty read window"
            read_segs.append(r.seg_idx)
    assert sorted(read_segs) == list(range(len(segs))), (
        "every segment must be read exactly once (no drop, no silent zero)"
    )


@settings(deadline=None, max_examples=10)
@given(
    doc_len=st.integers(min_value=2, max_value=30),
    seg_size=st.integers(min_value=2, max_value=9),
    seed=st.integers(min_value=0, max_value=10_000),
    mode=st.sampled_from(ALL_MODES),
)
def test_property_production_matches_reference(doc_len, seg_size, seed, mode):
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    ids = seeded_ids(doc_len, seed=seed)
    segs = fixed_segments(doc_len, seg_size)
    fids = [full_fids() for _ in segs]
    prod = run_collector(model, sae, [ids], [segs], [fids], mode, 4)[0]
    ref = ref_score_document(model, sae, ids, segs, fids, mode, 4)
    for p, r in zip(prod, ref, strict=True):
        assert t.equal(p, r)
        assert bool((p.float() >= 0).all()), "scores must be non-negative"


def test_zero_fraction_band():
    """ReLU(-g) should kill a substantial fraction of gated entries — a sign
    inversion would push this toward 0 or 1."""
    model = tiny_gemma2(True)
    sae = make_tiny_sae("standard")
    ids = seeded_ids(32, seed=17)
    segs = fixed_segments(32, 8)
    fids = [full_fids() for _ in segs]
    prod = run_collector(
        model, sae, [ids], [segs], [fids],
        GradientAttributionMode.future_half_split, 8,
    )[0]
    # Only consider features that are gated ON somewhere (score could be > 0).
    _, capt = ref_capture(model, ids, TINY_LAYER)
    flat, gated = [], []
    for s, (b, e) in enumerate(segs[:-1]):  # final segment may be zero by policy
        with t.no_grad():
            acts = sae.encode(capt[0, b:e].detach())
        on = (acts > 0).any(dim=0)
        flat.append(prod[s][on])
        gated.append(on)
    scores = t.cat(flat).float()
    zero_frac = float((scores == 0).float().mean())
    print(f"zero-score fraction among gated features: {zero_frac:.3f}")
    assert 0.05 < zero_frac < 0.95


def test_determinism_rerun_identical():
    model = tiny_gemma2(True)
    sae = make_tiny_sae("jumprelu")
    ids = seeded_ids(26, seed=8)
    segs = fixed_segments(26, 8)
    fids = [full_fids() for _ in segs]
    for mode in ALL_MODES:
        a = run_collector(model, sae, [ids], [segs], [fids], mode, 8)[0]
        b = run_collector(model, sae, [ids], [segs], [fids], mode, 8)[0]
        for x, y in zip(a, b, strict=True):
            assert t.equal(x, y)


# ---------------------------------------------------------------------------
# Decoder-linearity gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["standard", "jumprelu"])
def test_decoder_linearity_gate_passes(arch):
    decoder_linearity_gate(make_tiny_sae(arch))


def test_decoder_linearity_gate_rejects_normalization():
    sae = make_tiny_sae("standard")
    sae.cfg.normalize_activations = "expected_average_only_in"
    with pytest.raises(Exception, match="normalize_activations"):
        decoder_linearity_gate(sae)
