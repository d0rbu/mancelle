"""S2 row-packing benchmark: unpacked vs SDPA-packed truncated forward.

The production comparison for S2: packing ragged documents into one sequence
(`capture_packed`) to skip wasted pad-token compute, vs the padded
unpacked forward (`capture`). FlexAttention was evaluated as the packed
attention impl and DROPPED — it lost end-to-end on every GPU tested (RTX 4090
Ada, RTX PRO 6000 Blackwell sm_120, H100 sm_90), eager and compiled, at every
kernel tile, and was less robust to compile; SDPA-packing wins and is simpler
(see git history / the S2 PR for the head-to-head data).

Two synthetic corpora (both ΣL=768 real tokens, so the only variable is
padding) test the pad-fraction sensitivity:
  * realistic (pad≈0.70): lengths [640, 80, 40, 8] — a long doc > gemma-3's
    512 sliding window + short docs, like ragged production traffic.
  * low-pad (pad≈0.00): lengths [192, 192, 192, 192] — uniform; packing has
    almost no pad to save and pays O(ΣL²) attention, so it's neutral/slower
    here (the documented low-pad regression we accept by packing
    unconditionally).

`--compiled` torch.compile's the FULL truncated forward (S1's always-on
production config) for both arms — the production-correct measurement, since
`capture`/`capture_packed` both route through the compiled base when active.

Usage:
    uv run python scripts/benchmark/benchmark_s2_rowpack.py
    uv run python scripts/benchmark/benchmark_s2_rowpack.py --compiled --n-iters 30
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

# Make the repo root importable when run as a script (`uv run python
# scripts/...` puts only the script's own dir on sys.path, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch as t
from loguru import logger

from steering.native_collector import (
    NativeCollector,
    build_native_model,
)
from steering.sae_resolver import ActivationType

# (model_name, layer) — the two production cells.
_CELLS: list[tuple[str, int]] = [("gemma-2-2b", 6), ("gemma-3-1b-pt", 7)]
_DTYPES: list[t.dtype] = [t.float32, t.bfloat16]

# Synthetic corpora (per-doc token-id lists). Both ΣL=768 real tokens, so the
# only difference is the pad-fraction; ids are valid small positive ints.
_REALISTIC_LENS = [640, 80, 40, 8]   # pad_frac = 1 - 768/(4*640) = 0.700
_LOWPAD_LENS = [192, 192, 192, 192]  # pad_frac = 0.0


def _docs_for_lengths(lengths: list[int]) -> list[list[int]]:
    return [[3 + (i * 7 + j) % 59 for j in range(L)] for i, L in enumerate(lengths)]


def _pad_fraction(lengths: list[int]) -> float:
    return 1.0 - sum(lengths) / (len(lengths) * max(lengths))


def _dtype_str(d: t.dtype) -> str:
    return {t.float32: "fp32", t.bfloat16: "bf16"}.get(d, str(d))


def _free() -> None:
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _sync() -> None:
    if t.cuda.is_available():
        t.cuda.synchronize()


def _timed(fn, n_warmup: int, n_iters: int) -> dict:
    """CUDA-synced latency: warmup (discarded) then N measurements."""
    for _ in range(n_warmup):
        fn()
    _sync()
    times: list[float] = []
    for _ in range(n_iters):
        _sync()
        s = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - s)
    tt = t.tensor(times)
    return {
        "median_ms": float(tt.median() * 1000),
        "mean_ms": float(tt.mean() * 1000),
        "std_ms": float(tt.std() * 1000),
        "min_ms": float(tt.min() * 1000),
    }


def _peak_mb() -> float | None:
    return t.cuda.max_memory_allocated() / 1e6 if t.cuda.is_available() else None


def _measure_unpacked(nc: NativeCollector, docs, n_warmup, n_iters) -> dict:
    """Right-pad (id 0) -> maskless capture -> strip: the current prod path."""
    device = nc.device
    lengths = [len(d) for d in docs]
    padded = t.zeros(len(docs), max(lengths), dtype=t.long, device=device)
    for i, ids in enumerate(docs):
        padded[i, : len(ids)] = t.tensor(ids, dtype=t.long, device=device)
    hook = f"blocks.{nc.truncated_layer}.hook_resid_post"

    def _fwd() -> None:
        with t.no_grad():
            nc.capture(padded, nc.truncated_layer, hook_name=hook,
                       activation_type=ActivationType.residual)

    if t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()
    stats = _timed(_fwd, n_warmup, n_iters)
    stats["peak_mb"] = _peak_mb()
    return stats


def _measure_packed(nc: NativeCollector, docs, n_warmup, n_iters) -> dict:
    hook = f"blocks.{nc.truncated_layer}.hook_resid_post"

    def _fwd() -> None:
        nc.capture_packed(docs, nc.truncated_layer, hook_name=hook,
                          activation_type=ActivationType.residual)

    if t.cuda.is_available():
        t.cuda.reset_peak_memory_stats()
    stats = _timed(_fwd, n_warmup, n_iters)
    stats["peak_mb"] = _peak_mb()
    return stats


def _maybe_compile(nc: NativeCollector, compiled: bool) -> None:
    """Arm + activate the S1 compiled forward so capture / capture_packed run
    the FULL truncated forward through torch.compile (S1's always-on prod
    config). The first measured forward's warmup absorbs the compile."""
    if compiled:
        nc.enable_compiled()
        nc.set_compiled_active(True)


def run_cell(model_name: str, layer: int, dtype: t.dtype, device: str,
             n_warmup: int, n_iters: int, compiled: bool) -> list[dict]:
    """Measure unpacked + sdpa-packed for both corpora (compiled if requested)."""
    rows: list[dict] = []
    ds = _dtype_str(dtype)
    nc = build_native_model(model_name, device, dtype, layer)
    assert nc is not None, f"build_native_model None for {model_name}"
    _maybe_compile(nc, compiled)

    for corpus, lens in (("realistic", _REALISTIC_LENS), ("lowpad", _LOWPAD_LENS)):
        docs = _docs_for_lengths(lens)
        unp = _measure_unpacked(nc, docs, n_warmup, n_iters)
        pk = _measure_packed(nc, docs, n_warmup, n_iters)
        row = {
            "model": model_name, "layer": layer, "dtype": ds, "corpus": corpus,
            "compiled": compiled,
            "pad_fraction": round(_pad_fraction(lens), 4),
            "eff_toks": sum(lens),
            "unpacked_ms": round(unp["median_ms"], 3),
            "unpacked_peak_mb": unp["peak_mb"],
            "packed_ms": round(pk["median_ms"], 3),
            "packed_peak_mb": pk["peak_mb"],
            "speedup": round(unp["median_ms"] / pk["median_ms"], 4),
        }
        rows.append(row)
        flag = "" if row["speedup"] >= 1.0 else "  [SLOWER]"
        logger.info(
            f"{model_name}@{layer}/{ds}/{corpus} pad={row['pad_fraction']:.2f} | "
            f"unpacked={row['unpacked_ms']:.1f}ms | "
            f"packed={row['packed_ms']:.1f}ms ({row['speedup']:.2f}x){flag}"
        )
    del nc
    _free()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-iters", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="benchmark_s2_rowpack.json")
    ap.add_argument(
        "--compiled", action="store_true",
        help="torch.compile the FULL truncated forward (S1's always-on prod "
             "config) for both arms. WARNING: each (model, dtype, shape) "
             "triggers an inductor compile in warmup, so it's much slower.",
    )
    args = ap.parse_args()

    if not t.cuda.is_available():
        logger.error("CUDA required (meaningful timing).")
        raise SystemExit(1)

    mode = "COMPILED" if args.compiled else "eager"
    all_rows: list[dict] = []
    for model_name, layer in _CELLS:
        for dtype in _DTYPES:
            logger.info(f"\n{'='*72}\nBENCH {model_name}@{layer}/"
                        f"{_dtype_str(dtype)} mode={mode}\n{'='*72}")
            try:
                all_rows.extend(run_cell(model_name, layer, dtype, args.device,
                                         args.n_warmup, args.n_iters, args.compiled))
            except Exception as exc:
                logger.error(f"{model_name}@{layer}/{_dtype_str(dtype)}: ERROR: {exc!r}")

    print("\n" + "=" * 92)
    print(f"S2 ROW-PACKING [{mode}]: unpacked vs SDPA-packed "
          f"(median ms; speedup vs unpacked)")
    print("=" * 92)
    print(f"  {'cell':<26}{'corpus':<11}{'pad':>5} | {'unpacked':>11} | "
          f"{'packed':>11} | {'speedup':>8} | {'peak MB (unp/pk)':>18}")
    print("-" * 92)
    for r in all_rows:
        flag = "" if r["speedup"] >= 1.0 else " SLOWER"
        peak = (f"{r['unpacked_peak_mb']:.0f}/{r['packed_peak_mb']:.0f}"
                if r["unpacked_peak_mb"] is not None else "n/a")
        print(f"  {r['model']+'@'+str(r['layer'])+'/'+r['dtype']:<26}"
              f"{r['corpus']:<11}{r['pad_fraction']:>5.2f} | "
              f"{r['unpacked_ms']:>10.1f}m | {r['packed_ms']:>10.1f}m | "
              f"{r['speedup']:>7.2f}x{flag} | {peak:>18}")
    print("=" * 92)

    Path(args.out).write_text(json.dumps(all_rows, indent=2, default=str))
    logger.info(f"raw JSON -> {args.out}")


if __name__ == "__main__":
    main()
