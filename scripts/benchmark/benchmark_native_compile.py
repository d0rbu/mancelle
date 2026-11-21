"""S1 before/after benchmark: eager (PR-1 baseline) vs gated torch.compile.

Drives the REAL production path — `NativeCollector.enable_compiled()` arms the
compiled forward; then `_native_compile_gate` (the ORACLE) checks the compiled
config is no worse than bf16-eager relative to the fp32-eager (ideal-math)
reference, within `BASELINE_RELATIVE_SLACK`.  Only if the gate PASSES does the
collector flip to `_use_compiled=True`; the benchmark then measures both paths
and reports the ORACLE verdict next to the speedup column so a fast-but-oracle-
FAIL cell is structurally impossible to present as a win.

Honest-reporting (spec §3.2): the ORACLE column is determined by the gate, not by
the benchmark.  If the gate FAILS, the compiled row shows "DISABLED(gate-fail)"
and NO speedup is reported.  If the gate PASSES but compiled is SLOWER (speedup <
1.0), the row is flagged with "SLOWER" — a clearly losing result is never silently
buried.

Corpus: the SAME realistic ragged 4-document corpus (`_REALISTIC_DOCS`) as
`tests/steering/native/test_native_integration.py` — mirrored verbatim (same
strings, same tokenization with `resolve_tokenizer`, same right-pad-to-max-len
with id 0) so the benchmark sees exactly the production-shaped workload the slow
gate certifies correctness on.

The construction of the model+SAE+NativeCollector is also mirrored from the
integration test's gate cells (gemma-2-2b@6 / gemma-3-1b-pt@7).

Usage:
    uv run python scripts/benchmark/benchmark_native_compile.py
    uv run python scripts/benchmark/benchmark_native_compile.py smoke=true
    uv run python scripts/benchmark/benchmark_native_compile.py n_iters=30
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
)

# expandable_segments MUST be set before any CUDA allocation (mirrors _common.py).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# This file is a documented direct-script entry point, so Python otherwise puts
# scripts/benchmark (not the repository root) on sys.path. Keep the bootstrap
# before first-party imports; the targeted E402 suppressions encode that contract.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))

from sae_lens import SAE  # noqa: E402

from errors import CompileGateError  # noqa: E402
from steering.extract_features import _native_compile_gate  # noqa: E402
from steering.models import resolve_tokenizer  # noqa: E402
from steering.native_collector import (  # noqa: E402
    NativeCollector,
    build_native_model,
)
from steering.sae_resolver import (  # noqa: E402
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)

# --------------------------------------------------------------------------- #
# Realistic ragged production corpus — VERBATIM from                          #
# tests/steering/native/test_native_integration.py (_REALISTIC_DOCS).        #
# Mirrored here (not imported from tests/) to avoid tests/ import anti-       #
# pattern in production scripts; the benchmark MUST exercise the EXACT same   #
# corpus the slow correctness gate certifies, so the before/after numbers     #
# are on the precise production-shaped, >512-token sliding-window-stressing   #
# workload.  Each constant below is a verbatim mirror of                      #
# tests/steering/native/test_native_integration.py::<same name> (kept in     #
# sync deliberately; do not edit independently).                              #
# --------------------------------------------------------------------------- #
# verbatim mirror of tests/steering/native/test_native_integration.py::_SHORT_DOC (kept in sync deliberately; do not edit independently)
_SHORT_DOC = "The cat sat on the mat."

# verbatim mirror of tests/steering/native/test_native_integration.py::_MED_DOC (kept in sync deliberately; do not edit independently)
_MED_DOC = (
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose. Chlorophyll in the chloroplasts absorbs sunlight, and the "
    "Calvin cycle then fixes carbon dioxide into sugar."
)

# verbatim mirror of tests/steering/native/test_native_integration.py::_PARA_DOC (kept in sync deliberately; do not edit independently)
_PARA_DOC = (
    "The history of computing is a story of abstraction. Early mechanical "
    "calculators gave way to electromechanical relays, then vacuum tubes, "
    "transistors, and finally integrated circuits. Each layer hid the "
    "complexity beneath it, letting programmers reason about problems rather "
    "than wiring. Today an engineer composes services without ever touching "
    "a logic gate, yet every abstraction still rests on the physics of "
    "silicon."
)

# verbatim mirror of tests/steering/native/test_native_integration.py::_LONG_DOC (kept in sync deliberately; do not edit independently)
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
# The same 4-document corpus the slow integration gate uses, verbatim.
_REALISTIC_DOCS = [_SHORT_DOC, _MED_DOC, _PARA_DOC, _LONG_DOC]

# --- Gate cells: (model_name, release, layer, l0) — verbatim from         #
# test_native_integration.py's parametrize decorator.                       #
_CELL_GEMMA2 = ("gemma-2-2b", SAERelease.gemmascope, 6, None)
_CELL_GEMMA3 = ("gemma-3-1b-pt", SAERelease.gemmascope_2, 7, L0Tier.medium)
_CELLS: dict[str, tuple] = {
    "gemma-2-2b": _CELL_GEMMA2,
    "gemma-3-1b-pt": _CELL_GEMMA3,
}

# --------------------------------------------------------------------------- #
# Smoke helpers — tiny CPU Gemma twins + a stub SAE for `smoke=true` and the   #
# CPU smoke test.  These are defined IN THIS SCRIPT (NOT imported from         #
# tests/): a production script must never import from tests/, and the HARD     #
# project rule forbids any in-function/local import.  The factory + constants  #
# below are a verbatim mirror of                                              #
# tests/steering/native/conftest.py::_build_tiny_gemma2/_build_tiny_gemma3     #
# (same constants, same seed, same fork_rng) so a script-built twin is         #
# byte-identical to the one the native unit tests use — kept in sync           #
# deliberately; do not edit independently.                                     #
# --------------------------------------------------------------------------- #
_TINY_VOCAB = 64
_TINY_D_MODEL = 32
_TINY_LAYERS = 4
_TINY_HEADS = 4
_TINY_KV_HEADS = 2
_TINY_HEAD_DIM = 8
_TINY_FFN = 64
_TINY_CTX = 64
_TINY_SEED = 20260516  # verbatim mirror of conftest.TINY_SEED


def _build_tiny_gemma2() -> Gemma2ForCausalLM:
    """Fresh seeded tiny Gemma2 (byte-identical to conftest._build_tiny_gemma2).

    Verbatim mirror of tests/steering/native/conftest.py::_build_tiny_gemma2
    (kept in sync deliberately; do not edit independently).
    """
    with t.random.fork_rng():
        t.manual_seed(_TINY_SEED)
        # Transformers accepts implementation-specific config kwargs through
        # its dynamic constructor surface; keeping that payload explicitly
        # dynamic reflects the third-party API without weakening our model type.
        config_kwargs: dict[str, Any] = {
            "vocab_size": _TINY_VOCAB,
            "hidden_size": _TINY_D_MODEL,
            "intermediate_size": _TINY_FFN,
            "num_hidden_layers": _TINY_LAYERS,
            "num_attention_heads": _TINY_HEADS,
            "num_key_value_heads": _TINY_KV_HEADS,
            "head_dim": _TINY_HEAD_DIM,
            "max_position_embeddings": _TINY_CTX,
            "attn_implementation": "eager",
        }
        model = Gemma2ForCausalLM(Gemma2Config(**config_kwargs))
        t.nn.Module.to(model, dtype=t.float32)
        model.eval()
        return model


def _build_tiny_gemma3() -> Gemma3ForCausalLM:
    """Fresh seeded tiny Gemma3 (byte-identical to conftest._build_tiny_gemma3).

    Verbatim mirror of tests/steering/native/conftest.py::_build_tiny_gemma3
    (kept in sync deliberately; do not edit independently).
    """
    with t.random.fork_rng():
        t.manual_seed(_TINY_SEED)
        config_kwargs: dict[str, Any] = {
            "vocab_size": _TINY_VOCAB,
            "hidden_size": _TINY_D_MODEL,
            "intermediate_size": _TINY_FFN,
            "num_hidden_layers": _TINY_LAYERS,
            "num_attention_heads": _TINY_HEADS,
            "num_key_value_heads": _TINY_KV_HEADS,
            "head_dim": _TINY_HEAD_DIM,
            "max_position_embeddings": _TINY_CTX,
            "attn_implementation": "eager",
        }
        model = Gemma3ForCausalLM(Gemma3TextConfig(**config_kwargs))
        t.nn.Module.to(model, dtype=t.float32)
        model.eval()
        return model


def _smoke_tiny_factories() -> dict[str, Any]:
    """{model_name -> fresh-twin factory} for the smoke path (no tests/ import)."""
    return {
        "gemma-2-2b": _build_tiny_gemma2,
        "gemma-3-1b-pt": _build_tiny_gemma3,
    }


class _TinySAE:
    """Minimal SAE-protocol stub for smoke: linear+ReLU, no real weights.

    Deterministic seeded weights ⇒ reproducible feature activations. Same
    shape contract as the native suite's SAE stubs (`parameters()`, `to()`,
    `encode(x): [*, d] -> [*, f]`).
    """

    def __init__(self, d: int = _TINY_D_MODEL, f: int = 48) -> None:
        with t.random.fork_rng():
            t.manual_seed(42)
            self._w = t.randn(d, f)
            self._b = t.randn(f)

    def parameters(self):
        yield from []

    def to(self, *_args: Any, **_kwargs: Any) -> _TinySAE:
        return self

    def encode(self, x: t.Tensor) -> t.Tensor:
        return t.relu(x.float().matmul(self._w) + self._b)


# Probe input: ragged right-padded (row 1 has 2 trailing pads) — exercises the
# production maskless / right-pad contract at the smoke (CPU) level.
_SMOKE_PROBE_IDS = t.tensor(
    [[3, 9, 1, 7, 2], [5, 5, 8, 0, 0]], dtype=t.long
)


def _dtype_from_str(s: str) -> t.dtype:
    return {"fp32": t.float32, "bf16": t.bfloat16, "float32": t.float32,
            "bfloat16": t.bfloat16}[s]


def _dtype_str(dtype: t.dtype) -> str:
    return {t.float32: "fp32", t.bfloat16: "bf16"}.get(dtype, str(dtype))


def _free() -> None:
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _tokenize_corpus(model_name: str, docs: list[str]) -> list[list[int]]:
    """Tokenize docs using the model's real gated-HF tokenizer.

    Mirrors test_native_integration.py's tokenization: `resolve_tokenizer`,
    `padding=False`, `add_special_tokens=True` — exactly the production worker
    path.
    """
    tokenizer = resolve_tokenizer(model_name)
    return [
        tokenizer(text, padding=False, add_special_tokens=True)["input_ids"]
        for text in docs
    ]


def _pad_batch(
    token_id_lists: list[list[int]], device: str
) -> tuple[t.Tensor, list[int]]:
    """Right-pad to max length with id 0 — verbatim production contract.

    Mirrors _get_sae_activations_batched's padding logic (extract_features.py).
    """
    lengths = [len(ids) for ids in token_id_lists]
    max_seq = max(lengths)
    padded = t.zeros(len(token_id_lists), max_seq, dtype=t.long, device=device)
    for i, ids in enumerate(token_id_lists):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
    return padded, lengths


def _cuda_sync() -> None:
    if t.cuda.is_available():
        t.cuda.synchronize()


def _timed(fn: Any, n_warmup: int, n_iters: int) -> dict:
    """CUDA-synced latency (warmup then N median-based measurements).

    Mirrors the methodology from _common.py::timed: warmup iterations (CUDA
    synced), then n_iters CUDA-synced individual timings.  Returns a stats dict
    with mean_ms / median_ms / std_ms / min_ms / max_ms / n_iters.
    """
    for _ in range(n_warmup):
        fn()
    _cuda_sync()
    times: list[float] = []
    for _ in range(n_iters):
        _cuda_sync()
        s = time.perf_counter()
        fn()
        _cuda_sync()
        times.append(time.perf_counter() - s)
    tt = t.tensor(times)
    return {
        "mean_ms": float(tt.mean() * 1000),
        "median_ms": float(tt.median() * 1000),
        "std_ms": float(tt.std() * 1000),
        "min_ms": float(tt.min() * 1000),
        "max_ms": float(tt.max() * 1000),
        "n_iters": n_iters,
    }


def _measure_path(
    forward_fn: Any,
    padded_ids: t.Tensor,
    n_warmup: int,
    n_iters: int,
) -> dict:
    """Measure latency + peak memory for a single forward path.

    Resets peak memory stats BEFORE the run, so peak_mb reflects this path's
    high-water mark exclusively.  tok/s is computed from the total tokens in the
    padded batch (B * S) over the median latency.
    """
    if t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()

    def _call() -> None:
        with t.no_grad():
            forward_fn(padded_ids)

    stats = _timed(_call, n_warmup, n_iters)
    B, S = padded_ids.shape
    tok_s = int(B * S / (stats["median_ms"] / 1000))
    peak_mb: float | None = None
    if t.cuda.is_available():
        peak_mb = t.cuda.max_memory_allocated() / 1e6
    return {**stats, "tok_per_s": tok_s, "peak_mb": peak_mb}


def _measure_first_call(
    forward_fn: Any, padded_ids: t.Tensor
) -> float:
    """Wall-clock time of the very first forward call (compile / warmup cost)."""
    t0 = time.perf_counter()
    with t.no_grad():
        forward_fn(padded_ids)
    _cuda_sync()
    return time.perf_counter() - t0


def format_row(row: dict) -> str:
    """Format one benchmark row as a human-readable table line.

    HONEST-REPORTING GUARD: if oracle_pass is False, the compiled column shows
    "DISABLED(gate-fail)" and no speedup is printed — structurally impossible to
    present a gate-FAIL cell as a win.  If oracle_pass is True but speedup < 1.0,
    the speedup is printed with a "SLOWER" flag.
    """
    model = row["model"]
    dtype = row["dtype"]
    layer = row["layer"]
    e_ms = row["eager_median_ms"]
    e_ts = row["eager_tok_per_s"]
    e_mb = row.get("eager_peak_mb")

    oracle = "PASS" if row["oracle_pass"] else "FAIL"

    if not row["oracle_pass"]:
        # Gate FAILED — structurally cannot present compiled numbers as a win.
        compiled_str = "DISABLED(gate-fail)"
        speedup_str = "N/A"
        flag = "[ORACLE=FAIL]"
    else:
        c_ms = row["compiled_median_ms"]
        c_ts = row["compiled_tok_per_s"]
        c_mb = row.get("compiled_peak_mb")
        warmup_s = row.get("compiled_warmup_s", float("nan"))
        speedup = row["speedup"]
        compiled_str = (
            f"{c_ms:.1f}ms / {c_ts:,}tok/s / "
            f"{c_mb:.0f}MB" if c_mb is not None else f"{c_ms:.1f}ms / {c_ts:,}tok/s"
        )
        compiled_str += f" / warmup={warmup_s:.1f}s"
        speedup_str = f"{speedup:.3f}x"
        flag = "[SLOWER]" if speedup < 1.0 else ""

    eager_str = (
        f"{e_ms:.1f}ms / {e_ts:,}tok/s / "
        f"{e_mb:.0f}MB" if e_mb is not None else f"{e_ms:.1f}ms / {e_ts:,}tok/s"
    )
    return (
        f"  {model}@{layer}/{dtype} | "
        f"eager={eager_str} | "
        f"compiled={compiled_str} | "
        f"speedup={speedup_str} | "
        f"ORACLE={oracle} {flag}"
    ).rstrip()


def build_row(
    model_name: str,
    layer: int,
    dtype: t.dtype,
    padded_ids: t.Tensor,
    native_model: NativeCollector,
    sae: Any,
    n_warmup: int,
    n_iters: int,
) -> dict:
    """Measure eager and (if gate PASSES) compiled for one (model, layer, dtype) cell.

    PRODUCTION GATE PATH: calls `native_model.enable_compiled()` then
    `_native_compile_gate(native_model, sae, ...)` — the REAL gate including
    `dynamic=True` compile, `emulate_precision_casts=False` (the inductor
    default, empirical winner against the fp32-eager ideal-math reference),
    and the ideal-math-relative gate predicate.  The `oracle_pass` field is
    set from `native_model._use_compiled` AFTER the gate (True iff the gate
    passed and activated the compiled path).

    HONEST-REPORTING GUARD (spec §3.2): if `oracle_pass` is False, NO compiled
    performance numbers are collected, `compiled_median_ms` / `compiled_tok_per_s`
    are set to None, and `speedup` is set to None.  The formatter refuses to
    present these as a win.  If `oracle_pass` is True but speedup < 1.0, the
    `speedup_flag` field is set to "SLOWER" and the formatter flags it loudly.
    """
    dtype_str = _dtype_str(dtype)
    tag = f"{model_name}@{layer}/{dtype_str}"
    logger.info(f"[benchmark_native_compile] {tag}: measuring eager baseline ...")

    # 1. Ensure compile is INACTIVE for the eager baseline.
    native_model.set_compiled_active(False)

    def _eager_forward(ids: t.Tensor) -> t.Tensor:
        return native_model._truncated_forward(ids)

    eager_stats = _measure_path(_eager_forward, padded_ids, n_warmup, n_iters)

    row: dict = {
        "model": model_name,
        "layer": layer,
        "dtype": dtype_str,
        "eager_median_ms": eager_stats["median_ms"],
        "eager_mean_ms": eager_stats["mean_ms"],
        "eager_std_ms": eager_stats["std_ms"],
        "eager_min_ms": eager_stats["min_ms"],
        "eager_max_ms": eager_stats["max_ms"],
        "eager_tok_per_s": eager_stats["tok_per_s"],
        "eager_peak_mb": eager_stats["peak_mb"],
        "eager_n_iters": eager_stats["n_iters"],
    }

    # 2. ARM the compiled path + run the REAL production gate (the ORACLE).
    #    The gate calls enable_compiled() (idempotent), probes eager ≈ compiled
    #    via the project SAE feature-equivalence predicate, and sets
    #    native_model._use_compiled = True iff it PASSES. Under the no-opt-out
    #    contract (post-refactor `a64bcc30`), a FAILED oracle raises
    #    CompileGateError — we catch it here ONLY so the benchmark can record
    #    the cell as ORACLE=FAIL and move on to the next cell instead of
    #    aborting the whole sweep (production extraction still hard-aborts).
    logger.info(f"[benchmark_native_compile] {tag}: running production gate ...")
    try:
        _native_compile_gate(
            native_model,
            sae,
            probe_input_ids=padded_ids,
            model_dtype=dtype,
        )
    except CompileGateError as exc:
        logger.warning(
            f"[benchmark_native_compile] {tag}: gate RAISED CompileGateError "
            f"({exc}); recording as ORACLE=FAIL"
        )
        # Production hard-aborts here; the benchmark recorder records and
        # continues so subsequent cells still get measured.
    oracle_pass = native_model._use_compiled

    row["oracle_pass"] = oracle_pass
    row["oracle_verdict"] = "PASS" if oracle_pass else "FAIL"

    if not oracle_pass:
        # HONEST-REPORTING: gate FAILED — no compiled numbers, no speedup.
        # The guard here is the structural enforcement: we do NOT measure the
        # compiled path if the oracle rejected it; every downstream formatter/
        # assertion can rely on oracle_pass=False ⟹ compiled data absent.
        logger.warning(
            f"[benchmark_native_compile] {tag}: ORACLE=FAIL — compiled path "
            "disabled by gate; NOT measuring compiled performance"
        )
        row["compiled_median_ms"] = None
        row["compiled_mean_ms"] = None
        row["compiled_std_ms"] = None
        row["compiled_min_ms"] = None
        row["compiled_max_ms"] = None
        row["compiled_tok_per_s"] = None
        row["compiled_peak_mb"] = None
        row["compiled_warmup_s"] = None
        row["compiled_n_iters"] = None
        row["speedup"] = None
        row["speedup_flag"] = "ORACLE=FAIL"
        return row

    # 3. Gate PASSED — native_model._use_compiled is True. Measure compiled.
    logger.info(f"[benchmark_native_compile] {tag}: gate PASSED; measuring compiled ...")

    def _compiled_forward(ids: t.Tensor) -> t.Tensor:
        return native_model._truncated_forward(ids)

    # First-call cost (compile + triton kernel compilation).
    warmup_s = _measure_first_call(_compiled_forward, padded_ids)
    compiled_stats = _measure_path(_compiled_forward, padded_ids, n_warmup, n_iters)

    eager_ms = row["eager_median_ms"]
    comp_ms = compiled_stats["median_ms"]
    speedup = eager_ms / comp_ms if comp_ms > 0 else float("inf")

    row["compiled_median_ms"] = comp_ms
    row["compiled_mean_ms"] = compiled_stats["mean_ms"]
    row["compiled_std_ms"] = compiled_stats["std_ms"]
    row["compiled_min_ms"] = compiled_stats["min_ms"]
    row["compiled_max_ms"] = compiled_stats["max_ms"]
    row["compiled_tok_per_s"] = compiled_stats["tok_per_s"]
    row["compiled_peak_mb"] = compiled_stats["peak_mb"]
    row["compiled_warmup_s"] = warmup_s
    row["compiled_n_iters"] = compiled_stats["n_iters"]
    row["speedup"] = speedup
    # HONEST-REPORTING: flag a non-win loudly so it is never silently buried.
    row["speedup_flag"] = "SLOWER" if speedup < 1.0 else "OK"

    return row


def run_real_cell(
    model_name: str,
    layer: int,
    dtype: t.dtype,
    device: str,
    n_warmup: int,
    n_iters: int,
) -> dict:
    """Build real model+SAE+collector, run benchmark, free everything.

    Mirrors the test_native_integration.py construction for the (model, layer,
    dtype) cell:
      - `resolve_sae` with the cell's release / l0 / width
      - `SAE.from_pretrained` via sae_lens
      - `build_native_model` for the truncated NativeCollector
      - realistic ragged corpus tokenized with the model's real tokenizer
      - right-padded the same way as `_get_sae_activations_batched`

    Construction is NOT imported from tests/ — it is mirrored here with a comment
    to avoid the tests/-import-into-production-script anti-pattern.
    """
    if model_name not in _CELLS:
        raise ValueError(
            f"unknown model cell {model_name!r}; known: {list(_CELLS)}"
        )
    name, release, _layer, l0 = _CELLS[model_name]
    if _layer != layer:
        logger.warning(
            f"requested layer {layer} differs from cell default {_layer}; "
            "using requested layer"
        )

    resolved = resolve_sae(
        model=name,
        release=release,
        activation_type=ActivationType.residual,
        layer=layer,
        width=SAEWidth("16k"),
        l0=l0,
    )
    sae_obj = SAE.from_pretrained(
        release=resolved.release, sae_id=resolved.sae_id, device=device
    )
    if isinstance(sae_obj, tuple):
        sae_obj = sae_obj[0]
    sae_obj = sae_obj.to(dtype)

    native_model = build_native_model(model_name, device, dtype, layer)
    assert native_model is not None, f"build_native_model returned None for {model_name}"

    docs = _tokenize_corpus(model_name, _REALISTIC_DOCS)
    padded_ids, _lengths = _pad_batch(docs, device)

    row = build_row(
        model_name, layer, dtype, padded_ids, native_model, sae_obj,
        n_warmup, n_iters,
    )

    del native_model, sae_obj, padded_ids
    _free()
    return row


def run_smoke_cell(
    model_name: str,
    layer: int,
    dtype: t.dtype,
    tiny_factory: Any,
    tiny_sae: Any,
    probe_ids: t.Tensor,
    n_warmup: int,
    n_iters: int,
) -> dict:
    """Smoke path: tiny CPU gemma twins + stub SAE, no GPU/download.

    Used by `--smoke` mode and the smoke test suite.  Accepts pre-built tiny
    models (same pattern as benchmark_activation_methods.py's smoke path).
    `tiny_factory` is a callable returning a fresh seeded HF tiny Gemma model
    (verbatim from conftest._build_tiny_gemma2/_build_tiny_gemma3).
    `tiny_sae` is a SAE-protocol object with `.encode(x)` and `.parameters()`.
    """
    hf_model = tiny_factory()
    native_model = NativeCollector.truncated(
        hf_model, model_name=model_name, layer=layer
    )
    row = build_row(
        model_name, layer, dtype, probe_ids, native_model, tiny_sae,
        n_warmup, n_iters,
    )
    del native_model
    _free()
    return row


@hydra.main(
    config_path="../../configs/profiling",
    config_name="benchmark_native_compile",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint — iterates over configured (model, layer, dtype) cells."""
    device = cfg.get("device", "cuda")
    n_warmup = int(cfg.n_warmup)
    n_iters = int(cfg.n_iters)
    smoke = bool(cfg.get("smoke", False))
    out_dir = Path(cfg.get("out_dir", "."))

    if smoke:
        # Smoke path: tiny CPU twins + stub SAE, no GPU/download. The factories
        # and the stub SAE are module-level helpers defined IN THIS SCRIPT
        # (NOT imported from tests/) — a production script must never import
        # from tests/, and the HARD project rule forbids in-function imports.
        # They are a verbatim mirror of the native suite's conftest twins so
        # the smoke gate is exactly the gate the unit tests assert.
        _tiny_factories = _smoke_tiny_factories()
        _tiny_sae = _TinySAE()
        _probe_ids = _SMOKE_PROBE_IDS
        rows: list[dict] = []
        for model_cfg in cfg.models:
            mn = model_cfg.name
            ly = int(model_cfg.layer)
            for dtype_s in cfg.dtypes:
                dtype = _dtype_from_str(dtype_s)
                if mn not in _tiny_factories:
                    logger.warning(f"smoke: no tiny factory for {mn}, skip")
                    continue
                row = run_smoke_cell(
                    mn, ly, dtype, _tiny_factories[mn], _tiny_sae,
                    _probe_ids, n_warmup, n_iters,
                )
                rows.append(row)
                print(format_row(row))
    else:
        if not t.cuda.is_available():
            logger.error("CUDA required for real benchmark (use smoke=true for CPU)")
            sys.exit(1)
        rows = []
        for model_cfg in cfg.models:
            mn = model_cfg.name
            ly = int(model_cfg.layer)
            for dtype_s in cfg.dtypes:
                dtype = _dtype_from_str(dtype_s)
                logger.info(f"\n{'='*72}\nBENCH {mn}@{ly}/{dtype_s} "
                            f"(warmup={n_warmup} iters={n_iters})\n{'='*72}")
                try:
                    row = run_real_cell(mn, ly, dtype, device, n_warmup, n_iters)
                except Exception as exc:
                    logger.error(f"{mn}@{ly}/{dtype_s}: ERROR: {exc!r}")
                    row = {
                        "model": mn, "layer": ly, "dtype": dtype_s,
                        "error": str(exc),
                        "oracle_pass": False,
                        "oracle_verdict": "ERROR",
                        "speedup": None,
                        "speedup_flag": "ERROR",
                    }
                rows.append(row)
                print(format_row(row))

    # Human table.
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    for row in rows:
        print(format_row(row))
    print("=" * 80)

    # Save raw JSON next to the human table.
    out_json = out_dir / "benchmark_native_compile.json"
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    logger.info(f"[benchmark_native_compile] raw JSON written to {out_json}")


if __name__ == "__main__":
    main()
