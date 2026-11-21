#!/usr/bin/env python3
"""Benchmark activation collection methods for SAE feature extraction.

Two benchmark families:

GPT-2 (legacy, Phase-1): compares
  1. TransformerLens full forward + hook capture (current ``batched_forward``)
  2. TransformerLens ``stop_at_layer`` (built-in early stopping)
  3. nnterp ``tracer.stop()`` (raw HF + TL-transformed weights)

Gemma (Phase-2, carry-over #2): for each configured ``{name, layer}`` pair
(defaults ``gemma-2-2b@6`` + ``gemma-3-1b-pt@7``) benchmarks the **truncated
native** residual collector (``steering.native_collector.build_native_model``,
residual-only, structural truncation) against the **Phase-1 full-forward-hook**
reference and, when built, nnterp / TransformerLens.

A **pre-timing correctness gate** runs BEFORE any timed iteration of the
native method: the truncated native output MUST be byte-identical
(``torch.equal``, ``max_abs == 0``) to the Phase-1 full-forward-hook reference
on the same input. On mismatch the benchmark for that model is ABORTED
(``BenchmarkGateError``) — a numerically-wrong method is never timed or
reported (no fast-but-wrong).

To keep peak memory ~= one full model (gemma-2-2b OOMs otherwise), each model
is built → gated/measured → freed (``del`` + ``gc.collect()`` +
cuda-guarded ``torch.cuda.empty_cache()``) before the next is built; the small
reference activation is cached on CPU so the baseline model can be freed before
the truncated collector is built.

Usage:
    uv run python scripts/benchmark/benchmark_activation_methods.py
    uv run python scripts/benchmark/benchmark_activation_methods.py batch_size=32
    uv run python scripts/benchmark/benchmark_activation_methods.py smoke=true
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nnterp import StandardizedTransformer
from transformers import AutoModelForCausalLM

from steering.extract_features import (
    ActivationCapture,
    build_nnterp_model,
    load_sae_and_model,
)
from steering.gemma3_loading import load_gemma3_text_decoder
from steering.models import EXTRACTION_MODELS
from steering.native_collector import NativeCollector, build_native_model
from steering.sae_resolver import ActivationType

# Tiny CPU gemma twins for --smoke, built the SAME way as the native test
# suite (same constants/seed/fork_rng) so the smoke gate is exactly the gate
# the unit tests assert. conftest imports pytest at module top; pytest is a
# project dependency importable under `uv run python` (verified) — keeping
# this at module top satisfies the hard "imports at top, no in-function
# imports" rule.
from tests.steering.native.conftest import (
    _build_tiny_gemma2,
    _build_tiny_gemma3,
)

# The TL-style resid_post hook site the truncated native collector serves;
# defensively re-validated by NativeCollector.capture (kept identical to the
# spec §3.2 substring check).
_RESID_POST_HOOK = "blocks.{layer}.hook_resid_post"


class BenchmarkGateError(RuntimeError):
    """Raised when the pre-timing correctness gate fails for a model.

    The truncated native output diverged from the Phase-1 full-forward-hook
    reference (NOT byte-identical). The benchmark for that model is aborted —
    a numerically-wrong method is never timed or reported (spec §5: no
    fast-but-wrong).
    """


def make_dummy_batch(
    batch_size: int, seq_len: int, vocab_size: int, device: str
) -> t.Tensor:
    """Create a batch of random token IDs."""
    return t.randint(0, vocab_size, (batch_size, seq_len), device=device)


def _cuda_sync() -> None:
    """``torch.cuda.synchronize()`` only when CUDA is present (CPU no-op)."""
    if t.cuda.is_available():
        t.cuda.synchronize()


def _free_model(*objs: Any) -> None:
    """Sequential-residency free: drop refs, gc, cuda-guarded empty_cache.

    Re-derived from the Phase-1 "R1 sequential residency" lesson: building
    multiple full models per model-config (baseline-hook + truncated-native +
    nnterp + TL) simultaneously OOMs gemma-2-2b, so each is built → measured →
    freed before the next. The small reference activation is cached on CPU
    (NOT freed here) so the baseline model can be released before the
    truncated collector is built — peak ~= one full model.
    """
    for o in objs:
        del o
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _log_and_free_oom(
    method: str, model_name: str, tokens: t.Tensor, *objs: Any
) -> None:
    """Handle a per-method ``torch.cuda.OutOfMemoryError`` (spike-harness parity).

    Logs a single clear ``[OOM] <method> @ <model> shape=<B>x<S>`` line, then
    frees the offending objects via the same sequential-residency path as
    ``_free_model`` (``del`` + ``gc.collect()`` + cuda-guarded
    ``empty_cache()``). The caller records an ``OOM`` sentinel row and
    CONTINUES — one OOM-ing method/model must NOT crash the whole job or the
    models loop (mirrors the spike harness; spec part 2).
    """
    b, s = int(tokens.shape[0]), int(tokens.shape[1])
    logger.error(f"[OOM] {method} @ {model_name} shape={b}x{s}")
    _free_model(*objs)


def _oom_row(label: str) -> dict:
    """A sentinel result row for a method that OOM'd (rendered, not divided)."""
    return {"label": label, "oom": True}


def benchmark_fn(fn, n_warmup: int, n_iters: int, label: str) -> dict:
    """Benchmark a function with warmup and multiple iterations."""
    # Warmup
    for _ in range(n_warmup):
        fn()
    _cuda_sync()

    times = []
    for _ in range(n_iters):
        _cuda_sync()
        start = time.perf_counter()
        fn()
        _cuda_sync()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_t = t.tensor(times)
    return {
        "label": label,
        "mean_ms": float(times_t.mean() * 1000),
        "std_ms": float(times_t.std() * 1000),
        "min_ms": float(times_t.min() * 1000),
        "max_ms": float(times_t.max() * 1000),
        "median_ms": float(times_t.median() * 1000),
        "n_iters": n_iters,
    }


def run_method_full_forward_hook(model, sae, tokens, layer):
    """Method 1: Full forward pass with hook capture (current approach)."""
    capture = ActivationCapture()
    hook_point = model.blocks[layer].hook_resid_pre
    handle = hook_point.register_forward_hook(capture.hook_fn)
    try:
        with t.no_grad():
            model(tokens)
        activations = capture.activation
        if activations is None:
            raise RuntimeError(f"No activation captured at layer {layer}")
        sae_out = sae.encode(activations.view(-1, activations.shape[-1]))
        return sae_out.view(tokens.shape[0], tokens.shape[1], -1)
    finally:
        handle.remove()


def run_method_stop_at_layer(model, sae, tokens, layer):
    """Method 2: TransformerLens stop_at_layer (skips later layers)."""
    with t.no_grad():
        residual = model(tokens, stop_at_layer=layer)
    sae_out = sae.encode(residual.view(-1, residual.shape[-1]))
    return sae_out.view(tokens.shape[0], tokens.shape[1], -1)


def run_method_nnterp(nnterp_model, sae, input_ids, layer):
    """Method 3: nnterp tracer.stop() with HuggingFace GPT-2."""
    with t.no_grad():
        with nnterp_model.trace(input_ids) as tracer:
            # layers_output[layer-1] = output of block layer-1 = input to block layer
            # This matches hook_resid_pre at `layer`
            activations = nnterp_model.layers_output[layer - 1].save()
            tracer.stop()

    # nnsight .save() returns a plain tensor in recent versions
    acts = activations
    sae_out = sae.encode(acts.view(-1, acts.shape[-1]))
    return sae_out.view(input_ids.shape[0], input_ids.shape[1], -1)


def run_method_nnterp_tl_weights(nnterp_model, sae, input_ids, layer):
    """Method 4: nnterp tracer.stop() with TransformerLens-transformed weights."""
    with t.no_grad():
        with nnterp_model.trace(input_ids) as tracer:
            activations = nnterp_model.layers_output[layer - 1].save()
            tracer.stop()

    acts = activations
    sae_out = sae.encode(acts.view(-1, acts.shape[-1]))
    return sae_out.view(input_ids.shape[0], input_ids.shape[1], -1)


def hook_reference_residual(
    hf_model: Any, input_ids: t.Tensor, layer: int
) -> t.Tensor:
    """Phase-1 full-forward-hook residual capture — the correctness baseline.

    Mirrors the residual hook mechanism of the (now test-only) Phase-1
    ``NativeCollector.capture`` residual path and of
    ``tests/steering/native/_hook_reference.py`` (sans that file's test-only
    non-Tensor ``ExtractionError`` guard, which is unreachable for gemma
    decoder layers — they always emit a tuple whose ``[0]`` is a Tensor;
    byte-identity on valid output is asserted by
    ``test_benchmark_native_gate.py::test_benchmark_hook_reference_is_canonical_phase1_mechanism``):
    a ``register_forward_hook`` on the decoder layer
    ``hf_model.model.layers[layer]`` that unwraps a tuple/list output via
    ``[0]`` and stores ``.detach().clone()``; the FULL forward runs under
    ``torch.no_grad()`` and is MASKLESS; the handle is removed in a
    ``finally:``. Returns ``resid_post(layer)`` ``[B, S, d]``, unstripped —
    exactly what the truncated native collector must reproduce
    byte-identically (spec §4.1; the pre-timing gate's reference).
    """
    module = hf_model.model.layers[layer]
    captured: dict[str, t.Tensor] = {}

    def _fwd_hook(_mod, _inp, out):
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        captured["v"] = tensor.detach().clone()

    handle = module.register_forward_hook(_fwd_hook)
    try:
        with t.no_grad():
            hf_model(input_ids=input_ids, attention_mask=None)
    finally:
        handle.remove()
    return captured["v"]


def run_method_native(
    native_model: NativeCollector,
    sae,
    input_ids: t.Tensor,
    layer: int,
) -> t.Tensor:
    """Truncated native residual collector (structural truncation, no hook).

    Times ``NativeCollector.capture`` for residual on the structurally
    truncated model built by ``build_native_model(model_name, device, dtype,
    layer)`` (residual-only; the Phase-1 full-forward + hook path is removed).
    Returns the raw ``[B, S, d]`` residual when ``sae is None`` (smoke / no
    SAE loaded) or the SAE-encoded feature tensor otherwise — mirroring the
    other ``run_method_*`` SAE-encode tail.
    """
    assert isinstance(native_model, NativeCollector)
    with t.no_grad():
        acts = native_model.capture(
            input_ids,
            layer,
            hook_name=_RESID_POST_HOOK.format(layer=layer),
            activation_type=ActivationType.residual,
        )
    if sae is None:
        return acts
    sae_out = sae.encode(acts.reshape(-1, acts.shape[-1]))
    return sae_out.view(input_ids.shape[0], input_ids.shape[1], -1)


def compare_outputs(
    out_a: t.Tensor, out_b: t.Tensor, label_a: str, label_b: str
) -> dict:
    """Compare two output tensors for numerical equivalence."""
    # Move to CPU for comparison
    a = out_a.float().cpu()
    b = out_b.float().cpu()

    abs_diff = (a - b).abs()
    rel_diff = abs_diff / (a.abs() + 1e-8)

    return {
        "comparison": f"{label_a} vs {label_b}",
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "mean_rel_diff": float(rel_diff.mean()),
        "allclose_atol1e5": bool(t.allclose(a, b, atol=1e-5)),
        "allclose_atol1e4": bool(t.allclose(a, b, atol=1e-4)),
        "allclose_atol1e3": bool(t.allclose(a, b, atol=1e-3)),
        "cosine_sim": float(
            t.nn.functional.cosine_similarity(
                a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
            )
        ),
    }


def _assert_native_matches_reference(
    native_out: t.Tensor,
    reference_out: t.Tensor,
    model_name: str,
) -> None:
    """Pre-timing correctness gate — abort-before-timing, NO fast-but-wrong.

    The truncated native output MUST be byte-identical to the Phase-1
    full-forward-hook reference: same shape, same dtype, ``torch.equal`` /
    ``max_abs == 0`` (spec §4.1). On ANY mismatch this logs a clear error and
    raises ``BenchmarkGateError`` so the caller ABORTS the benchmark for that
    model — a numerically-wrong method is never timed or reported.

    Both tensors are compared on CPU (the reference is cached on CPU for
    sequential residency; ``native_out`` is moved to match) so the gate works
    after the baseline model has been freed.
    """
    a = native_out.detach().cpu()
    b = reference_out.detach().cpu()

    if a.shape != b.shape:
        msg = (
            f"[{model_name}] PRE-TIMING GATE FAILED: truncated native shape "
            f"{tuple(a.shape)} != Phase-1 hook reference {tuple(b.shape)}; "
            f"ABORTING benchmark for {model_name} (no fast-but-wrong)"
        )
        logger.error(msg)
        raise BenchmarkGateError(msg)

    if a.dtype != b.dtype:
        msg = (
            f"[{model_name}] PRE-TIMING GATE FAILED: truncated native dtype "
            f"{a.dtype} != Phase-1 hook reference {b.dtype}; ABORTING "
            f"benchmark for {model_name} (no fast-but-wrong)"
        )
        logger.error(msg)
        raise BenchmarkGateError(msg)

    if not t.equal(a, b):
        max_abs = float((a.float() - b.float()).abs().max())
        msg = (
            f"[{model_name}] PRE-TIMING GATE FAILED: truncated native is NOT "
            f"byte-identical to the Phase-1 full-forward-hook reference "
            f"(max_abs={max_abs:.6g}, require == 0); ABORTING benchmark for "
            f"{model_name} (no fast-but-wrong — never time/report a "
            f"numerically-wrong method)"
        )
        logger.error(msg)
        raise BenchmarkGateError(msg)

    logger.info(
        f"[{model_name}] pre-timing gate PASSED: truncated native == "
        f"Phase-1 full-forward-hook reference (max_abs=0, byte-identical)"
    )


def _build_fresh_hf_gemma(
    model_name: str,
    device: str,
    model_dtype: t.dtype,
    tiny_model: Any = None,
) -> Any:
    """Load a FRESH, UNTRUNCATED Gemma{2,3}ForCausalLM for the hook reference.

    A separate instance from the one ``build_native_model`` mutates in place
    (it slices ``.layers`` / replaces ``.norm``), so the reference forward
    runs the full model. ``tiny_model`` (smoke) bypasses the HF download with
    an already-built tiny model (built the conftest way). Loaded fp32 then
    converted (mirrors ``build_native_model``).
    """
    if tiny_model is not None:
        _loaded = tiny_model
    else:
        hf_id = EXTRACTION_MODELS[model_name].hf_model_id
        if model_name.startswith("gemma-3"):
            _loaded = load_gemma3_text_decoder(
                hf_id, dtype=t.float32, attn_implementation="sdpa"
            )
        else:
            _loaded = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype=t.float32, attn_implementation="sdpa"
            )
    # widen the HF union so .to(str|device) / overload-wrapped calls
    # type-check (verbatim native_collector.py `hf_model: Any = _loaded`
    # idiom — the HF .to() overloads don't accept a bare str).
    m: Any = _loaded
    m = m.to(device).eval()
    if model_dtype != t.float32:
        m = m.to(model_dtype)
    return m


def run_gemma_model_benchmark(
    model_name: str,
    layer: int,
    cfg: DictConfig,
    device: str,
    tiny_factory=None,
) -> dict | None:
    """Sequential-residency benchmark of one gemma model + the pre-timing gate.

    Flow (≤ 1 full model resident at any time — re-derived Phase-1 R1 lesson):

      1. build FRESH untruncated HF gemma → run Phase-1 full-forward-hook
         reference → cache reference activation on **CPU** → FREE the HF model.
      2. build the TRUNCATED native collector (separate fresh model) → run
         ``capture(residual)`` → **PRE-TIMING GATE**: assert byte-identical to
         the cached reference (``_assert_native_matches_reference``); on
         mismatch raise ``BenchmarkGateError`` (abort this model — never time
         a wrong method) → time the native method → FREE.
      3. (real-model, non-smoke, not skip_nnterp) build nnterp → time → FREE;
         build TL → time → FREE. Each built/measured/freed in turn so peak
         ~= one full model (gemma-2-2b OOMs otherwise).

    **Per-method OOM (spike-harness parity, spec part 2):** every timed /
    untimed forward (the full-forward-hook reference, truncated native,
    nnterp, TL) is wrapped in ``try/except torch.cuda.OutOfMemoryError``. On
    OOM the offending method logs ``[OOM] <method> @ <model> shape=BxS``, is
    freed, records an ``OOM`` sentinel row, and the benchmark CONTINUES to the
    next method/model — one OOM never crashes the whole job or the models
    loop. If the REFERENCE forward itself OOMs the model cannot be gated:
    it is recorded ``OOM (baseline)`` (``gate_ok=False``) and the function
    returns that result so the caller continues to the next model (it does
    NOT raise out of ``main``).

    Returns a result dict (per-method medians + speedups vs the Phase-1 hook
    baseline; possibly OOM sentinel rows) or ``None`` if the truncated
    collector could not be built. The pre-timing gate runs in step 2, BEFORE
    any timed iteration (its byte-identity logic is UNCHANGED).
    """
    smoke = tiny_factory is not None
    logger.info(f"--- gemma benchmark: {model_name} @ layer {layer} ---")

    # The native truncated collector returns the raw [B,S,d] residual; the
    # gate compares residuals directly (no SAE needed — byte-identity of the
    # residual transitively gates SAE-feature equivalence, spec §4).
    sae = None

    # ── Step 1: Phase-1 full-forward-hook reference (fresh untruncated) ──────
    ref_hf = _build_fresh_hf_gemma(
        model_name,
        device,
        cfg_dtype(cfg),
        tiny_model=tiny_factory() if smoke else None,
    )
    # Derive the token batch from the reference model's own vocab (smoke: a
    # tiny 2x5 batch; real: cfg.gemma_batch_size x cfg.gemma_seq_len). gemma
    # uses its OWN, smaller shape — NOT the GPT-2 cfg.batch_size/cfg.seq_len:
    # the Phase-1 full-forward-hook reference runs the full forward incl. the
    # 256k-vocab lm_head, so bf16 logits at the GPT-2 16x1024 default are
    # ≈ 8 GB and OOM a 24 GB card (T9-surfaced). gemma_batch_size=8 /
    # gemma_seq_len=512 → bf16 logits ≈ 2.0 GB, fits with margin.
    vocab_size = int(ref_hf.config.vocab_size)
    if smoke:
        tokens = make_dummy_batch(2, 5, vocab_size, device)
    else:
        tokens = make_dummy_batch(
            int(cfg.gemma_batch_size),
            int(cfg.gemma_seq_len),
            vocab_size,
            device,
        )
    # The Phase-1 full-forward-hook reference runs the FULL forward (incl. the
    # 256k-vocab gemma lm_head). If it OOMs this model cannot be gated → record
    # `OOM (baseline)` and return (caller continues to the next model; we do
    # NOT raise out of main — spec part 2).
    try:
        with t.no_grad():
            reference_out = hook_reference_residual(ref_hf, tokens, layer)
        # Cache on CPU so the full model can be freed before the truncated
        # build (sequential residency — peak ~= one full model).
        reference_cpu = reference_out.detach().cpu()
        _free_model(ref_hf, reference_out)
    except t.cuda.OutOfMemoryError:
        _log_and_free_oom(
            "full_forward+hook (baseline)", model_name, tokens, ref_hf
        )
        return {
            "model": model_name,
            "layer": layer,
            "gate_ok": False,
            "methods": [_oom_row("full_forward+hook (baseline)")],
            "total_tokens": int(tokens.shape[0] * tokens.shape[1]),
        }

    # ── Step 2: truncated native — build, GATE (pre-timing), then time ──────
    # PR-1 load/transform split: real-model path loads via build_native_model
    # (the I/O half); smoke bypasses the HF download with a conftest-built
    # tiny twin and calls the genuine pure-transform boundary
    # NativeCollector.truncated directly (no removed _tiny_model seam),
    # mirroring build_native_model's device+fp32→dtype handling.
    if smoke:
        _tiny: Any = tiny_factory()
        _tiny = _tiny.to(device).eval()
        if cfg_dtype(cfg) != t.float32:
            _tiny = _tiny.to(cfg_dtype(cfg))
        native_model = NativeCollector.truncated(
            _tiny, model_name=model_name, layer=layer
        )
    else:
        native_model = build_native_model(
            model_name,
            device,
            cfg_dtype(cfg),
            layer,
        )
    if native_model is None:
        logger.warning(
            f"[{model_name}] native build returned None — skipping"
        )
        _free_model(reference_cpu)
        return None

    result: dict[str, Any] = {
        "model": model_name,
        "layer": layer,
        "gate_ok": True,
        "methods": [],
    }
    total_tokens = int(tokens.shape[0] * tokens.shape[1])

    # The truncated native forward feeds the pre-timing gate. If it OOMs we
    # cannot gate (the gate compares native_out vs the cached reference) →
    # record native as OOM, mark gate_ok=False, SKIP timing native, but still
    # try the (independent) step-3 timed methods. The gate's byte-identity
    # logic is UNCHANGED — it simply has no native output to compare here.
    native_ok = True
    try:
        with t.no_grad():
            native_out = run_method_native(native_model, sae, tokens, layer)
    except t.cuda.OutOfMemoryError:
        _log_and_free_oom(
            "truncated_native", model_name, tokens, native_model
        )
        result["gate_ok"] = False
        result["methods"].append(_oom_row("truncated_native"))
        native_ok = False

    if native_ok:
        # PRE-TIMING CORRECTNESS GATE — raises BenchmarkGateError on any
        # mismatch. MUST run before ANY timed iteration of the native method.
        # (Logic UNCHANGED; not weakened — still byte-identical or abort.)
        _assert_native_matches_reference(
            native_out, reference_cpu, model_name
        )

        # Native timed run — wrapped so an OOM here records an OOM row and
        # continues to step 3 (does NOT crash the job).
        try:
            if device == "cuda":
                t.cuda.reset_peak_memory_stats()
            r_native = benchmark_fn(
                lambda: run_method_native(
                    native_model, sae, tokens, layer
                ),
                cfg.n_warmup,
                cfg.n_iters,
                "truncated_native",
            )
            if device == "cuda":
                r_native["peak_mem_mb"] = (
                    t.cuda.max_memory_allocated() / 1e6
                )
            result["methods"].append(r_native)
        except t.cuda.OutOfMemoryError:
            _log_and_free_oom("truncated_native", model_name, tokens)
            result["methods"].append(_oom_row("truncated_native"))
        finally:
            _free_model(native_model)
    else:
        _free_model(native_model)

    # ── Step 3: Phase-1 full-forward-hook baseline + nnterp + TL (timed) ────
    # Rebuild the full HF model for the *timed* hook baseline (it was freed
    # after the gate to keep peak low); skipped on smoke (CPU-only wiring
    # proof — real perf numbers are T9 on the 4090).
    if not smoke:
        # Timed Phase-1 full-forward-hook baseline. This is the run the T9
        # 24 GB card OOM'd at the GPT-2 16x1024 shape (256k-vocab logits
        # ≈ 8 GB); with the gemma-specific smaller shape it fits, but guard
        # anyway — on OOM record an OOM row and continue (spec part 2).
        base_hf = _build_fresh_hf_gemma(model_name, device, cfg_dtype(cfg))
        try:
            if device == "cuda":
                t.cuda.reset_peak_memory_stats()
            r_hook = benchmark_fn(
                lambda: hook_reference_residual(base_hf, tokens, layer),
                cfg.n_warmup,
                cfg.n_iters,
                "full_forward+hook",
            )
            if device == "cuda":
                r_hook["peak_mem_mb"] = t.cuda.max_memory_allocated() / 1e6
            result["methods"].append(r_hook)
        except t.cuda.OutOfMemoryError:
            _log_and_free_oom(
                "full_forward+hook", model_name, tokens, base_hf
            )
            result["methods"].append(_oom_row("full_forward+hook"))
        finally:
            _free_model(base_hf)

        if not cfg.skip_nnterp:
            nnterp_model = None
            try:
                nnterp_model = build_nnterp_model(
                    None, device=device, model_name=model_name
                )
                if nnterp_model is not None:
                    if device == "cuda":
                        t.cuda.reset_peak_memory_stats()
                    r_nn = benchmark_fn(
                        lambda: _nnterp_resid(nnterp_model, tokens, layer),
                        cfg.n_warmup,
                        cfg.n_iters,
                        "nnterp",
                    )
                    if device == "cuda":
                        r_nn["peak_mem_mb"] = t.cuda.max_memory_allocated() / 1e6
                    result["methods"].append(r_nn)
                    _free_model(nnterp_model)
            except t.cuda.OutOfMemoryError:
                # Distinct from the broad except below: emit the clear [OOM]
                # line + an OOM row, then continue (spec part 2).
                _log_and_free_oom(
                    "nnterp", model_name, tokens, nnterp_model
                )
                result["methods"].append(_oom_row("nnterp"))
            except Exception as e:  # pragma: no cover - real-model only
                logger.warning(f"[{model_name}] nnterp benchmark skipped: {e}")

    result["total_tokens"] = total_tokens
    return result


def _nnterp_resid(nnterp_model, input_ids: t.Tensor, layer: int) -> t.Tensor:
    """nnterp residual at ``layer`` (resid_post) via ``tracer.stop()``."""
    with t.no_grad():
        with nnterp_model.trace(input_ids) as tracer:
            acts = nnterp_model.layers_output[layer].save()
            tracer.stop()
    return acts


def cfg_dtype(cfg: DictConfig) -> t.dtype:
    """Resolve the benchmark model dtype from the hydra cfg (default bf16)."""
    name = str(getattr(cfg, "model_dtype", "bfloat16"))
    return {
        "float32": t.float32,
        "float16": t.float16,
        "bfloat16": t.bfloat16,
    }.get(name, t.bfloat16)


def _print_gemma_report(
    results: list[dict], device: str, smoke: bool
) -> None:
    """Markdown report: median ms / tokens-s / peak-mem / speedup× per model.

    Speedup× is computed vs the Phase-1 full-forward-hook baseline (and the
    table also surfaces nnterp/TL when built). Keeps the existing report
    style (the GPT-2 path's table format).

    A method that OOM'd is recorded as an ``OOM`` sentinel row
    (``{"oom": True}``, no ``median_ms``); it is rendered as ``OOM`` across
    all numeric columns — NEVER divided by (spec part 2). If the baseline
    itself OOM'd there is no valid ``baseline_ms`` so every speedup degrades
    to ``n/a`` (no division by zero / by a missing key).
    """
    print("\n" + "=" * 80)
    print("GEMMA RESULTS (truncated native vs Phase-1 full-forward-hook)")
    print("=" * 80)
    if smoke:
        print(
            "\n  [smoke] tiny CPU models — wiring + pre-timing gate proof "
            "only; real perf numbers are collected on the 4090 (T9)."
        )

    for res in results:
        methods = res["methods"]
        # Baseline = the full_forward+hook row that actually ran (an OOM'd
        # baseline has no median_ms → no valid baseline → speedups = n/a).
        baseline = next(
            (
                m
                for m in methods
                if m["label"] == "full_forward+hook" and not m.get("oom")
            ),
            None,
        )
        baseline_ms = baseline["median_ms"] if baseline else None
        total_tokens = res.get("total_tokens", 0)

        if res.get("gate_ok") is False:
            gate_tag = "  [gate NOT RUN — OOM]"
        elif methods:
            gate_tag = "  [gate PASSED]"
        else:
            gate_tag = ""
        print(f"\n### {res['model']} @ layer {res['layer']}" + gate_tag)
        print()
        print("| Method | Median (ms) | Speedup× | Tokens/sec | Peak MB |")
        print("|--------|------------|----------|------------|---------|")
        for m in methods:
            if m.get("oom"):
                # OOM sentinel: render OOM everywhere, never divide.
                print(
                    f"| {m['label']} | OOM | OOM | OOM | OOM |"
                )
                continue
            spd = (
                f"{baseline_ms / m['median_ms']:.2f}x"
                if baseline_ms
                else "n/a"
            )
            tps = (
                f"{total_tokens / (m['median_ms'] / 1000):,.0f}"
                if total_tokens
                else "n/a"
            )
            peak = (
                f"{m['peak_mem_mb']:.0f}"
                if "peak_mem_mb" in m
                else ("-" if device != "cuda" else "n/a")
            )
            print(
                f"| {m['label']} | {m['median_ms']:.2f} | {spd} | {tps} | {peak} |"
            )


@hydra.main(
    config_path="../../configs/profiling",
    config_name="benchmark_activations",
    version_base=None,
)
def main(cfg: DictConfig):
    # Reduce CUDA caching-allocator fragmentation across this benchmark's
    # sequential-residency build→measure→free cycles (multiple full models
    # built/freed in turn). The T9 OOM error message explicitly recommends
    # expandable_segments:True. setdefault so an externally-set value wins;
    # this is an os.environ statement (NOT an import) so it is allowed in
    # main(), and it MUST run before any CUDA allocation / model build.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    device = cfg.device
    if device == "cuda" and not t.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    smoke = bool(getattr(cfg, "smoke", False))

    logger.info("=" * 80)
    logger.info("ACTIVATION COLLECTION METHOD BENCHMARK")
    logger.info("=" * 80)
    logger.info(
        f"  device={device}, warmup={cfg.n_warmup}, iters={cfg.n_iters}, "
        f"smoke={smoke}"
    )

    # ── Phase-2: gemma truncated-native benchmark (configurable model set) ──
    # cfg.models is a list of {name, layer}; Phase-2 defaults are
    # gemma-2-2b@6 + gemma-3-1b-pt@7 (the Phase-1-gate pairs). The pre-timing
    # gate aborts any model whose truncated output is not byte-identical to
    # the Phase-1 full-forward-hook reference.
    gemma_results: list[dict] = []
    if smoke:
        # Tiny CPU gemma twins built the conftest way (same constants/seed):
        # layer 2 is a real interior truncation for the 4-layer tiny models.
        smoke_models = [
            ("gemma-2-2b", 2, _build_tiny_gemma2),
            ("gemma-3-1b-pt", 2, _build_tiny_gemma3),
        ]
        for name, layer, factory in smoke_models:
            res = run_gemma_model_benchmark(
                name, layer, cfg, device, tiny_factory=factory
            )
            if res is not None:
                gemma_results.append(res)
    else:
        for entry in cfg.models:
            res = run_gemma_model_benchmark(
                str(entry["name"]), int(entry["layer"]), cfg, device
            )
            if res is not None:
                gemma_results.append(res)

    if gemma_results:
        _print_gemma_report(gemma_results, device, smoke)

    # ── GPT-2 path (Phase-1, preserved) — only when explicitly requested ────
    if not bool(getattr(cfg, "run_gpt2", False)) or smoke:
        return

    logger.info("=" * 80)
    logger.info("GPT-2 (Phase-1) ACTIVATION METHOD BENCHMARK")
    logger.info("=" * 80)
    logger.info(
        f"  batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, "
        f"layer={cfg.layer}"
    )

    logger.info("Loading TransformerLens GPT-2 + SAE...")
    layer = int(cfg.layer)
    tl_model, sae, _ = load_sae_and_model(
        model_name="gpt2",
        layer=layer,
        device=device,
        sae_release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
    )
    vocab_size = tl_model.cfg.d_vocab
    logger.info(
        f"  GPT-2: {tl_model.cfg.n_layers} layers, "
        f"d_model={tl_model.cfg.d_model}, vocab={vocab_size}"
    )
    logger.info(f"  SAE: d_sae={sae.cfg.d_sae}")

    nnterp_model = None
    nnterp_tl_model = None
    if not cfg.skip_nnterp:
        logger.info("Loading nnterp StandardizedTransformer (raw HF weights)...")
        try:
            kwargs = {"device_map": device} if device == "cuda" else {}
            nnterp_model = StandardizedTransformer("gpt2", **kwargs)
            logger.info(
                f"  nnterp GPT-2: {nnterp_model.num_layers} layers, "
                f"d_model={nnterp_model.hidden_size}"
            )
        except Exception as e:
            logger.warning(f"Failed to load nnterp: {e}")

        logger.info("Loading nnterp with TransformerLens weight transforms...")
        nnterp_tl_model = build_nnterp_model(tl_model, device=device)
        if nnterp_tl_model is not None:
            logger.info(f"  nnterp+TL: {nnterp_tl_model.num_layers} layers")

    tokens = make_dummy_batch(cfg.batch_size, cfg.seq_len, vocab_size, device)
    logger.info(f"Test batch: {tokens.shape} on {device}")

    print("\n" + "=" * 80)
    print("OUTPUT EQUIVALENCE")
    print("=" * 80)

    with t.no_grad():
        out_hook = run_method_full_forward_hook(tl_model, sae, tokens, cfg.layer)
        out_stop = run_method_stop_at_layer(tl_model, sae, tokens, cfg.layer)

    comp = compare_outputs(out_hook, out_stop, "full_forward+hook", "stop_at_layer")
    print(f"\n  {comp['comparison']}:")
    print(f"    max_abs_diff:  {comp['max_abs_diff']:.2e}")
    print(f"    mean_abs_diff: {comp['mean_abs_diff']:.2e}")
    print(f"    cosine_sim:    {comp['cosine_sim']:.8f}")
    print(f"    allclose(1e-5): {comp['allclose_atol1e5']}")
    print(f"    allclose(1e-4): {comp['allclose_atol1e4']}")

    if nnterp_model is not None:
        try:
            with t.no_grad():
                out_nnterp = run_method_nnterp(nnterp_model, sae, tokens, cfg.layer)

            comp_nn = compare_outputs(
                out_hook, out_nnterp, "full_forward+hook", "nnterp (raw HF)"
            )
            print(f"\n  {comp_nn['comparison']}:")
            print(f"    max_abs_diff:  {comp_nn['max_abs_diff']:.2e}")
            print(f"    cosine_sim:    {comp_nn['cosine_sim']:.8f}")
            print(f"    allclose(1e-3): {comp_nn['allclose_atol1e3']}")
        except Exception as e:
            logger.warning(f"nnterp comparison failed: {e}")
            nnterp_model = None

    comp_tl = None
    if nnterp_tl_model is not None:
        try:
            with t.no_grad():
                out_tl_nn = run_method_nnterp_tl_weights(
                    nnterp_tl_model, sae, tokens, cfg.layer
                )

            comp_tl = compare_outputs(
                out_hook, out_tl_nn, "full_forward+hook", "nnterp+TL_weights"
            )
            print(f"\n  {comp_tl['comparison']}:")
            print(f"    max_abs_diff:  {comp_tl['max_abs_diff']:.2e}")
            print(f"    cosine_sim:    {comp_tl['cosine_sim']:.8f}")
            print(f"    allclose(1e-4): {comp_tl['allclose_atol1e4']}")
            print(f"    allclose(1e-3): {comp_tl['allclose_atol1e3']}")
        except Exception as e:
            logger.warning(f"nnterp+TL comparison failed: {e}")
            nnterp_tl_model = None

    print("\n" + "=" * 80)
    print("SPEED BENCHMARK")
    print("=" * 80)

    if device == "cuda":
        t.cuda.reset_peak_memory_stats()

    results = []

    logger.info("Benchmarking: full_forward + hook...")
    r1 = benchmark_fn(
        lambda: run_method_full_forward_hook(tl_model, sae, tokens, cfg.layer),
        cfg.n_warmup,
        cfg.n_iters,
        "full_forward+hook",
    )
    results.append(r1)
    if device == "cuda":
        r1["peak_mem_mb"] = t.cuda.max_memory_allocated() / 1e6
        t.cuda.reset_peak_memory_stats()

    logger.info("Benchmarking: stop_at_layer...")
    r2 = benchmark_fn(
        lambda: run_method_stop_at_layer(tl_model, sae, tokens, cfg.layer),
        cfg.n_warmup,
        cfg.n_iters,
        "stop_at_layer",
    )
    results.append(r2)
    if device == "cuda":
        r2["peak_mem_mb"] = t.cuda.max_memory_allocated() / 1e6
        t.cuda.reset_peak_memory_stats()

    if nnterp_model is not None:
        logger.info("Benchmarking: nnterp tracer.stop() (raw HF)...")
        try:
            r3 = benchmark_fn(
                lambda: run_method_nnterp(nnterp_model, sae, tokens, cfg.layer),
                cfg.n_warmup,
                cfg.n_iters,
                "nnterp (raw HF)",
            )
            results.append(r3)
            if device == "cuda":
                r3["peak_mem_mb"] = t.cuda.max_memory_allocated() / 1e6
                t.cuda.reset_peak_memory_stats()
        except Exception as e:
            logger.warning(f"nnterp benchmark failed: {e}")

    if nnterp_tl_model is not None:
        logger.info("Benchmarking: nnterp+TL_weights tracer.stop()...")
        try:
            r4 = benchmark_fn(
                lambda: run_method_nnterp_tl_weights(
                    nnterp_tl_model, sae, tokens, cfg.layer
                ),
                cfg.n_warmup,
                cfg.n_iters,
                "nnterp+TL_weights",
            )
            results.append(r4)
            if device == "cuda":
                r4["peak_mem_mb"] = t.cuda.max_memory_allocated() / 1e6
                t.cuda.reset_peak_memory_stats()
        except Exception as e:
            logger.warning(f"nnterp+TL benchmark failed: {e}")

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    baseline_ms = results[0]["median_ms"]
    total_tokens = cfg.batch_size * cfg.seq_len

    print(
        f"\n  {'Method':<28} {'Median (ms)':>12} {'Std (ms)':>10} "
        f"{'Speedup':>10} {'Tokens/s':>12}",
        end="",
    )
    if device == "cuda":
        print(f" {'Peak MB':>10}", end="")
    print()
    print("  " + "-" * 90)

    for r in results:
        speedup = baseline_ms / r["median_ms"]
        tps = total_tokens / (r["median_ms"] / 1000)
        print(
            f"  {r['label']:<28} {r['median_ms']:>10.1f}ms "
            f"{r['std_ms']:>8.1f}ms {speedup:>9.2f}x {tps:>10,.0f}",
            end="",
        )
        if device == "cuda" and "peak_mem_mb" in r:
            print(f" {r['peak_mem_mb']:>8.0f}MB", end="")
        print()

    print("\n" + "=" * 80)
    print("SUMMARY (for PR description)")
    print("=" * 80)
    print()
    print("### Activation Collection Method Comparison")
    print()
    print(
        f"Benchmark: batch_size={cfg.batch_size}, seq_len={cfg.seq_len}, "
        f"layer={cfg.layer}, device={device}"
    )
    print()
    print("| Method | Median (ms) | Speedup | Tokens/sec |")
    print("|--------|------------|---------|------------|")
    for r in results:
        speedup = baseline_ms / r["median_ms"]
        tps = total_tokens / (r["median_ms"] / 1000)
        print(f"| {r['label']} | {r['median_ms']:.1f} | {speedup:.2f}x | {tps:,.0f} |")

    print()
    print("Output equivalence:")
    print(
        f"- stop_at_layer vs full_forward+hook: "
        f"max_abs_diff={comp['max_abs_diff']:.2e}, "
        f"cosine_sim={comp['cosine_sim']:.8f}"
    )
    if nnterp_model is not None:
        print(
            f"- nnterp (raw HF) vs full_forward+hook: "
            f"max_abs_diff={comp_nn['max_abs_diff']:.2e}, "
            f"cosine_sim={comp_nn['cosine_sim']:.8f}"
        )
    if comp_tl is not None:
        print(
            f"- nnterp+TL_weights vs full_forward+hook: "
            f"max_abs_diff={comp_tl['max_abs_diff']:.2e}, "
            f"cosine_sim={comp_tl['cosine_sim']:.8f}"
        )


if __name__ == "__main__":
    main()
