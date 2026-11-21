"""Measure the real VRAM + speed effect of encoder feature-dictionary compaction (#695).

Builds the QUEUED RUN's model — ``configs/training/
cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000.yaml`` — uncompacted vs compacted, and
reports:

  * parameter counts (feature-indexed, encoder, whole model)
  * PEAK VRAM during a real fwd+bwd+optimizer step (params + grads + Adam state +
    activations), which is the number that actually decides whether a run fits
  * step time

WHAT MATCHES THE QUEUED RUN, AND WHAT DOES NOT
----------------------------------------------
The MODEL SHAPE matches it exactly, and that is asserted — ``tests/scripts/
test_bench_encoder_vocab_compaction.py`` builds this model AND the production model (via
``build_encdec_meta_model_from_training_cfg``, the function the trainer itself calls) from
that YAML and compares configs and every parameter count. A docstring claim cannot drift
away from the config without failing.

This module previously *asserted* the same thing three times while hard-coding an encoder
depth ratio of 0.75 (queued: 1.0), a cross-attention frequency of 4 (queued: 1) and a
decoder vocabulary of 65,536 (queued: 32,768, the nanochat tokenizer's). None of those was
cosmetic: a 9-layer encoder carries SIX feature-indexed tables where the queued 12-layer
encoder carries SEVEN (``feature_embed`` + one value embedding per ``has_ve`` layer), so it
understated the dead weight compaction deletes by ~14%, while the vocab and frequency
errors inflated MODEL TOTAL — the denominator of the saving.

The BATCH SHAPE deliberately does NOT match, because the queued run's does not fit on one
development GPU. Be explicit about it when reading the output:

  * ``--batch`` defaults to 4; the queued run's ``device_batch_size`` is **32**.
  * ``--features`` defaults to 32; the queued run allows up to **96** per segment
    (``max_k: 49`` is exclusive for both positive and negative selection).
  * ``--seq-len`` 1024 and ``--n-seg`` 8 DO match (``max_seq_len: 1024`` /
    ``segment_size: 128``).

So the absolute PEAK VRAM printed here is NOT the queued run's peak — activations scale
with batch. The parameter counts are exact, and the compaction SAVING (the ratio this
exists to measure) is what transfers. Pass ``--batch 32`` on a 96 GB card for the real
figure.

THE VOCABULARY
--------------
``--vocab-size`` defaults to 2,074 — the exact effective vocabulary produced by the
queued run's audited 16,254-ID whitelist plus schema-v1 cluster consolidation. The
production model also has the compact feature-prediction head, and this benchmark feeds
its max-pool BCE objective so parameter, optimizer-state, activation, and step-time
measurements include it.

    uv run python -m scripts.benchmark.bench_encoder_vocab_compaction            # audited V=2,074
    uv run python -m scripts.benchmark.bench_encoder_vocab_compaction --batch 32 # production batch
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
from pathlib import Path

import torch as t

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab

# The run this benchmark exists to size. Every constant below is READ OFF this config; the
# test module composes it and asserts the model built here is shape-identical to the one the
# trainer builds from it.
QUEUED_CONFIG_NAME = "cluster_fp/conditional_encdec_splice_cf_gemma3pt_8xpro6000"

N_SAE = 65536  # sae.width: 65k
REAL_V = 2074  # audited whitelist + schema-v1 cluster consolidation

# --- decoder (queued config + the base_train defaults it inherits) ---------------------
DEC_DEPTH = 12  # depth: 12
DEC_ASPECT_RATIO = 64  # base_train.yaml aspect_ratio
DEC_HEAD_DIM = 128  # base_train.yaml head_dim
DEC_WINDOW_PATTERN = "SSSL"  # base_train.yaml window_pattern
SEQ_LEN = 1024  # max_seq_len: 1024

# The nanochat tokenizer's vocabulary — what conditional_encdec_train.py passes as
# ``base_vocab`` (``tokenizer.get_vocab_size()``); configs/data/tok_train.yaml pins it.
# NOT 65,536: that is the SAE width, and conflating the two doubled wte + lm_head + every
# decoder value embedding.
DEC_VOCAB_SIZE = 32768

# conditional_encdec_train.py's own derivation, reproduced so the two cannot drift.
DEC_N_EMBD = (
    (DEC_DEPTH * DEC_ASPECT_RATIO + DEC_HEAD_DIM - 1) // DEC_HEAD_DIM
) * DEC_HEAD_DIM  # 768
DEC_N_HEAD = DEC_N_EMBD // DEC_HEAD_DIM  # 6

# --- conditioning side ------------------------------------------------------------------
ENC_N_EMBD_RATIO = 1.0  # encoder.n_embd: 1.0 -> 768
ENC_DEPTH_RATIO = 1.0  # encoder.depth: 1.0 -> a FULL-DEPTH 12-layer encoder
ENC_HEAD_DIM = 128  # encoder.head_dim: 128
XATTN_FREQUENCY = 1  # cross_attention.frequency: 1 -> a site at EVERY decoder layer
NUM_BUCKETS = 2  # num_buckets (inherited)

# Batch defaults. SEG_SIZE is the queued run's; BATCH and FEATURES are NOT (see docstring).
SEG_SIZE = 128  # segment_size: 128 -> n_seg = SEQ_LEN // SEG_SIZE = 8
QUEUED_DEVICE_BATCH_SIZE = 32  # what the run actually uses; too big for one dev GPU
QUEUED_MAX_FEATURES = 48 + 48  # both max_k values are exclusive at 49
FEATURE_PRED_CONFIG = FeaturePredictionConfig(
    weight=0.1,
    layer=-1,
    loss_mode=FeaturePredLossMode.MAX_POOL,
    loss_fn=FeaturePredLossFn.BCE,
)


def build_meta_model(vocab: FeatureVocab | None, seq_len: int) -> EncoderDecoderGPT:
    """The queued run's model, on whatever device is active. Construct under ``meta``.

    Split out of :func:`build` so the shape can be compared against the production builder
    without allocating anything — which is what makes the "this IS the queued run's shape"
    claim above a test rather than a comment.
    """
    dec = GPTConfig(
        sequence_len=seq_len,
        vocab_size=DEC_VOCAB_SIZE,
        n_layer=DEC_DEPTH,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_HEAD,
        n_embd=DEC_N_EMBD,
        window_pattern=DEC_WINDOW_PATTERN,
    )
    enc = EncoderConfig(
        n_embd=ENC_N_EMBD_RATIO,
        depth=ENC_DEPTH_RATIO,
        n_head=None,
        n_kv_head=None,
        head_dim=ENC_HEAD_DIM,
    )
    xa = CrossAttentionConfig(frequency=XATTN_FREQUENCY, n_head=None, n_kv_head=None)
    return EncoderDecoderGPT(
        dec,
        enc,
        xa,
        n_sae_features=N_SAE,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=FEATURE_PRED_CONFIG,
        feature_vocab=vocab,
    )


def build(vocab: FeatureVocab | None, device: str, seq_len: int) -> EncoderDecoderGPT:
    with t.device("meta"):
        m = build_meta_model(vocab, seq_len)
    m.to_empty(device=device)
    t.manual_seed(0)
    m.init_weights()
    return m


def make_batch(
    vocab: FeatureVocab | None,
    device: str,
    *,
    B: int,
    T: int,
    n_seg: int,
    F: int,
    vocab_size: int = DEC_VOCAB_SIZE,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """One synthetic training batch.

    Split out of :func:`measure` — which is CUDA-only end to end — so the part that can be
    wrong on any device is reachable from a CPU test. It was: ``measure`` drew token ids from
    a hardcoded ``[0, 65536)`` while ``DEC_VOCAB_SIZE`` is 32,768, so every forward indexed a
    32,768-row ``wte`` out of range and the benchmark could not run at all. 65,536 is the SAE
    width, not the token vocabulary — the exact conflation the comment on ``DEC_VOCAB_SIZE``
    warns about. Nothing caught it because no test called ``measure``.

    ``vocab_size`` is a parameter, not a literal, so a test can pass a small value and prove
    the bound is honoured rather than hoping a random draw happens to exceed it.
    """
    idx = t.randint(0, vocab_size, (B, T), device=device)
    tgt = t.randint(0, vocab_size, (B, T), device=device)
    seg_map = (t.arange(T, device=device) // (T // n_seg)).unsqueeze(0).expand(B, T).contiguous()
    allowed = (
        t.tensor(vocab.original_ids, device=device)
        if vocab is not None
        else t.arange(N_SAE, device=device)
    )
    fids = t.zeros(B, n_seg, F, dtype=t.int64, device=device)
    mags = t.zeros(B, n_seg, F, dtype=t.int64, device=device)
    mask = t.zeros(B, n_seg, F, dtype=t.bool, device=device)
    k = max(1, F // 2)
    for b in range(B):
        for s in range(n_seg):
            fids[b, s, :k] = allowed[t.randint(0, len(allowed), (k,), device=device)]
            mags[b, s, :k] = t.randint(0, 2, (k,), device=device)
            mask[b, s, :k] = True
    return idx, tgt, fids, mags, mask, seg_map


def _host_rss_mib() -> float:
    for line in Path(f"/proc/{os.getpid()}/smaps_rollup").read_text().splitlines():
        if line.startswith("Rss:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("smaps_rollup did not contain an Rss field")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure(
    vocab: FeatureVocab | None, device: str, *, B: int, T: int, n_seg: int, F: int, steps: int
) -> dict:
    # Muon's Newton-Schulz is torch.compiled and specialises on the matrix shapes it
    # sees. Measuring two differently-shaped models in one process otherwise trips
    # FailOnRecompileLimitHit. Reset between runs so each is measured cleanly.
    t._dynamo.reset()
    t._dynamo.config.cache_size_limit = 64
    gc.collect()
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()

    m = build(vocab, device, T)
    t.cuda.synchronize()
    init_peak_allocated = t.cuda.max_memory_allocated() / 2**30
    init_peak_reserved = t.cuda.max_memory_reserved() / 2**30
    m.train()
    opt = m.setup_optimizer()

    counts = m.num_scaling_params()
    enc_cfg = m.encoder.config
    kv = enc_cfg.n_kv_head * enc_cfg.head_dim
    n_ve = len(m.encoder.value_embeds)
    rows = m.encoder.feature_embed.weight.shape[0]
    feat_indexed = rows * (enc_cfg.n_embd + n_ve * kv)

    idx, tgt, fids, mags, mask, seg_map = make_batch(vocab, device, B=B, T=T, n_seg=n_seg, F=F)
    fp_width = m.feature_prediction_width
    assert fp_width is not None
    fp_targets = t.zeros(B, n_seg, fp_width, device=device)
    seg_doc_id = t.arange(B, device=device).unsqueeze(1).expand(B, n_seg).contiguous()

    for _ in range(5):
        result = m(
            idx,
            tgt,
            fids,
            mags,
            mask,
            seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
        loss = result.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    t.cuda.synchronize()
    t.cuda.reset_peak_memory_stats()

    samples_ms: list[float] = []
    for _ in range(steps):
        t.cuda.synchronize()
        started = time.perf_counter()
        result = m(
            idx,
            tgt,
            fids,
            mags,
            mask,
            seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
        loss = result.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        t.cuda.synchronize()
        samples_ms.append((time.perf_counter() - started) * 1e3)

    peak_allocated = t.cuda.max_memory_allocated() / 2**30
    peak_reserved = t.cuda.max_memory_reserved() / 2**30
    out = {
        "rows": rows,
        "feature_indexed": feat_indexed,
        "encoder": counts["encoder"],
        "total": counts["total"],
        "init_peak_allocated_gib": init_peak_allocated,
        "init_peak_reserved_gib": init_peak_reserved,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "host_rss_mib": _host_rss_mib(),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "iqr_ms": _percentile(samples_ms, 0.75) - _percentile(samples_ms, 0.25),
    }
    # The two benchmark arms run sequentially in one process. Release every
    # CUDA-owning local, including ``result`` (which retains the final autograd
    # graph), before emptying the allocator cache; otherwise the compact arm is
    # measured on top of live full-width tensors.
    del (
        m,
        opt,
        result,
        loss,
        idx,
        tgt,
        fids,
        mags,
        mask,
        seg_map,
        fp_targets,
        seg_doc_id,
    )
    gc.collect()
    t.cuda.empty_cache()
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            f"VRAM/speed effect of encoder feature-dictionary compaction at the "
            f"{QUEUED_CONFIG_NAME} model shape. The MODEL shape matches that config "
            f"exactly; the BATCH defaults are downscaled to fit one GPU (see --batch)."
        )
    )
    p.add_argument("--vocab-size", type=int, default=REAL_V)
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help=f"per-device batch. The queued run uses {QUEUED_DEVICE_BATCH_SIZE}; the "
        "default 4 is a downscale so this fits on a development GPU, so the PEAK "
        "VRAM printed is NOT the queued run's absolute peak.",
    )
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--n-seg", type=int, default=SEQ_LEN // SEG_SIZE)
    p.add_argument(
        "--features",
        type=int,
        default=32,
        help=f"feature slots per segment. The queued run allows up to "
        f"{QUEUED_MAX_FEATURES} (max_k 49 exclusive for each branch).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=30,
        help="measured iterations after five warmups (production protocol: 30)",
    )
    a = p.parse_args()

    if not t.cuda.is_available():
        raise SystemExit("CUDA required — this measures VRAM.")
    dev = "cuda"
    enc_depth = int(ENC_DEPTH_RATIO * DEC_DEPTH)
    enc_dim = int(ENC_N_EMBD_RATIO * DEC_N_EMBD)
    print(f"device: {t.cuda.get_device_name(0)}")
    print(f"config: {QUEUED_CONFIG_NAME} (model shape verified against the YAML)")
    print(
        f"shape:  depth={DEC_DEPTH} n_embd={DEC_N_EMBD} vocab={DEC_VOCAB_SIZE:,} | "
        f"enc depth={enc_depth} n_embd={enc_dim} | x-attn every {XATTN_FREQUENCY} "
        f"| n_sae={N_SAE:,}"
    )
    print(f"batch:  B={a.batch} T={a.seq_len} n_seg={a.n_seg} F={a.features}")
    if a.batch != QUEUED_DEVICE_BATCH_SIZE:
        print(
            f"        NOTE: the queued run's device_batch_size is "
            f"{QUEUED_DEVICE_BATCH_SIZE}, so the PEAK VRAM below is NOT its absolute "
            f"peak. The parameter counts are exact; the SAVING is what transfers."
        )
    print()

    g = t.Generator().manual_seed(0)
    ids = tuple(sorted(t.randperm(N_SAE, generator=g)[: a.vocab_size].tolist()))
    vocab = FeatureVocab(n_sae_features=N_SAE, original_ids=ids)

    full = measure(None, dev, B=a.batch, T=a.seq_len, n_seg=a.n_seg, F=a.features, steps=a.steps)
    comp = measure(vocab, dev, B=a.batch, T=a.seq_len, n_seg=a.n_seg, F=a.features, steps=a.steps)

    def row(label, f, c, fmt="{:,}"):
        d = f - c
        pct = 100.0 * d / f if f else 0.0
        print(
            f"  {label:<28} {fmt.format(f):>16} {fmt.format(c):>16} "
            f"{fmt.format(d):>16}  ({pct:5.1f}%)"
        )

    print(f"{'':<30}{'UNCOMPACTED':>16}{'COMPACTED':>17}{'SAVED':>17}")
    print("  " + "-" * 78)
    row("encoder rows", full["rows"], comp["rows"])
    row("feature-indexed params", full["feature_indexed"], comp["feature_indexed"])
    row("encoder params", full["encoder"], comp["encoder"])
    row("MODEL TOTAL params", full["total"], comp["total"])
    print("  " + "-" * 78)
    row(
        "INIT PEAK allocated (GiB)",
        full["init_peak_allocated_gib"],
        comp["init_peak_allocated_gib"],
        "{:.3f}",
    )
    row(
        "INIT PEAK reserved (GiB)",
        full["init_peak_reserved_gib"],
        comp["init_peak_reserved_gib"],
        "{:.3f}",
    )
    row(
        "STEP PEAK allocated (GiB)",
        full["peak_allocated_gib"],
        comp["peak_allocated_gib"],
        "{:.3f}",
    )
    row(
        "STEP PEAK reserved (GiB)",
        full["peak_reserved_gib"],
        comp["peak_reserved_gib"],
        "{:.3f}",
    )
    row("host RSS (MiB)", full["host_rss_mib"], comp["host_rss_mib"], "{:.1f}")
    for label, key in (
        ("step median (ms)", "median_ms"),
        ("step p95 (ms)", "p95_ms"),
        ("step IQR (ms)", "iqr_ms"),
    ):
        row(label, full[key], comp[key], "{:.1f}")
    print()
    print(
        f"  vocabulary: {a.vocab_size:,}/{N_SAE:,} rows kept "
        f"({100 * a.vocab_size / N_SAE:.1f}% of the SAE dictionary)"
    )


if __name__ == "__main__":
    main()
