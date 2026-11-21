"""Benchmark + correctness check for the producer's batched tokenization (#10).

Proves, on a REAL extraction tokenizer over a REAL parquet shard's documents
(tens of thousands of real docs — not a toy CPU sandbox), that the batched
``_batch_truncate_to_training_window`` used by
``scripts.preprocess.precompute_gradient_attribution`` is:

1. BIT-IDENTICAL to the old per-doc ``_truncate_to_training_window`` loop
   (asserts list equality of (kept_ids, full_n_tokens) for every doc), and
2. quantitatively faster on the CPU tokenize phase that gates the GPU pass
   (one tokenizer call per chunk vs one per doc — fast tokenizers amortize the
   Python->Rust FFI and thread internally).

Tokenization is CPU work, so this needs no GPU — but it MUST run against the
real model's tokenizer and a real corpus shard for the timing to be meaningful.

Usage (on the box, pointing at a real shard parquet for the model)::

    uv run python -m scripts.analysis.benchmark_gradient_tokenization \
        --parquet /path/to/data/<dataset>/shard_00000.parquet \
        --model gemma-2-2b --context-window 1024 --repeats 3

The default ``--sae-max-context`` is large so the truncation limit equals
``--context-window`` (matching a typical run); set both to your run's values
for an exact match. The two implementations are compared on identical inputs.
"""

from __future__ import annotations

import argparse
import time

from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.extract_features import (
    _batch_truncate_to_training_window,
    _truncate_to_training_window,
)
from steering.models import resolve_tokenizer
from steering.parquet_io import read_parquet_texts


def _per_doc(texts, tok, cw, mc):
    return [_truncate_to_training_window(txt, tok, cw, mc) for txt in texts]


def _time(fn, repeats: int) -> tuple[float, list]:
    """Best-of-``repeats`` wall time (seconds) + the last result."""
    best = float("inf")
    result: list = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t0)
    return best, result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True, help="a REAL shard parquet for the model")
    ap.add_argument("--model", default="gemma-2-2b")
    ap.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    ap.add_argument("--sae-max-context", type=int, default=8192)
    ap.add_argument("--chunk-size", type=int, default=2048)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit-docs", type=int, default=0, help="0 = all docs")
    args = ap.parse_args()

    tok = resolve_tokenizer(args.model)
    texts = read_parquet_texts(args.parquet)
    if args.limit_docs:
        texts = texts[: args.limit_docs]
    cw, mc = args.context_window, args.sae_max_context
    print(f"model={args.model}  docs={len(texts):,}  limit=min({cw},{mc})={min(cw, mc)}")

    # Warm up (first tokenizer call pays one-time init; both paths share it).
    _truncate_to_training_window(texts[0] if texts else "", tok, cw, mc)
    _batch_truncate_to_training_window(texts[:1], tok, cw, mc, chunk_size=args.chunk_size)

    per_doc_s, ref = _time(lambda: _per_doc(texts, tok, cw, mc), args.repeats)
    batched_s, got = _time(
        lambda: _batch_truncate_to_training_window(
            texts, tok, cw, mc, chunk_size=args.chunk_size
        ),
        args.repeats,
    )

    # 1. CORRECTNESS: bit-identical outputs (kept_ids + full_n_tokens per doc).
    assert got == ref, "MISMATCH: batched tokenization != per-doc — NOT bit-identical"
    n_trunc = sum(1 for _, n_full in ref if n_full > min(cw, mc))
    print(f"correctness: BIT-IDENTICAL over all {len(texts):,} docs "
          f"({n_trunc:,} truncated to the limit)  ✓")

    # 2. SPEEDUP.
    speedup = per_doc_s / batched_s if batched_s > 0 else float("inf")
    print(f"per-doc loop : {per_doc_s:8.3f}s  ({1e6 * per_doc_s / max(len(texts), 1):.1f} us/doc)")
    print(f"batched      : {batched_s:8.3f}s  ({1e6 * batched_s / max(len(texts), 1):.1f} us/doc)")
    print(f"SPEEDUP      : {speedup:.2f}x  (chunk_size={args.chunk_size})")


if __name__ == "__main__":
    main()
