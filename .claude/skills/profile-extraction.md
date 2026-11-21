---
name: profile-extraction
description: Run SAE feature extraction profiling, parse results, diagnose performance issues
---

# Profile Extraction Skill

When this skill is invoked, follow these steps in order. Execute each step, parse the output, and use it to inform the next step. Provide a complete formatted report at the end.

## Step 1: Verify Environment

Before running anything, confirm the environment is ready:

```bash
# Check CUDA availability
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None; print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB') if torch.cuda.is_available() else None"
```

```bash
# Check that parquet data shards exist
uv run python -c "from nanochat.dataset import list_parquet_files; files = list_parquet_files(); print(f'{len(files)} parquet shards available')"
```

If CUDA is not available, use `--device cpu` in all subsequent commands and note this in the report.
If no parquet shards are available, instruct the user to download them with `uv run python -m nanochat.dataset -n 2`.

## Step 2: Run Torch Profiler (Summary Mode)

Run the standalone profiler script with the tiny model to get a breakdown of where time is spent:

```bash
uv run python scripts/profile_extraction.py --summary-only --n-docs 40 --device cuda --context-window 128
```

For real-model profiling (slower but accurate):

```bash
uv run python scripts/profile_extraction.py --summary-only --n-docs 20 --device cuda --use-real-model
```

### How to parse this output

The output is a `prof.key_averages().table()` with columns: Name, Self CPU %, Self CPU, CPU total %, CPU total, CPU time avg, and optionally CUDA columns. Look for these `record_function` annotations in the output:

| Annotation | What it measures |
|---|---|
| `gpt2_forward` | Single-doc GPT-2 forward pass (unbatched path) |
| `gpt2_forward_batched` | Batched GPT-2 forward pass |
| `sae_encode` | Single-doc SAE encode (unbatched path) |
| `sae_encode_batched` | Batched SAE encode |
| `tokenize` | GPT-2 tokenization of document text |
| `tokenize_bulk` | Bulk tokenization of a row group |
| `masked_mean_pool` | Mean pooling with length mask |
| `sparse_extraction` | Extracting non-zero features and sorting |
| `read_row_group` | Reading parquet row group from disk |
| `save_shard` | Saving output .pt shard file |

Extract the CPU total (or CUDA total if on GPU) for each of these annotations and compute what percentage of total time each consumes.

## Step 3: Run Full Extraction Profiling

Run the Hydra-based extraction on 1 shard with the default config to measure real throughput:

```bash
uv run python -m steering.extract_features command=extract num_shards=1 device=cuda batch_size=64
```

### Key metrics to extract from the output

Parse the log lines for:
- **Total docs processed**: from `Total: X docs`
- **Total tokens**: from `Total: ... X tokens`
- **Total elapsed time**: from `... in X.Xs`
- **ms/doc**: from `X.X ms/doc`
- **us/token**: from `X.X us/token`

Compute derived metrics:
- **Throughput (tokens/sec)** = total_tokens / elapsed_seconds
- **Throughput (docs/sec)** = total_docs / elapsed_seconds
- **Batch utilization** = actual_tokens_in_batch / (batch_size * context_window). If us/token is much higher than expected, batch utilization may be low.

## Step 4: Compare Backends

Run extraction with each backend explicitly to compare TransformerLens vs nnterp:

```bash
# TransformerLens backend
uv run python -m steering.extract_features command=extract num_shards=1 device=cuda backend=transformerlens batch_size=64 output_dir=out/data/features_tl
```

```bash
# nnterp backend (if available)
uv run python -m steering.extract_features command=extract num_shards=1 device=cuda backend=nnterp batch_size=64 output_dir=out/data/features_nnterp
```

If nnterp is not installed or fails, note this and skip the comparison. Also run the dedicated benchmark script for a more controlled comparison:

```bash
uv run python scripts/benchmark_activation_methods.py --batch-size 16 --seq-len 1024 --n-warmup 3 --n-iters 10 --device cuda
```

### How to parse benchmark output

The benchmark script prints a results table with columns: Method, Median (ms), Std (ms), Speedup, Tokens/s, Peak MB. Extract:
- **Speedup of each method** relative to full_forward+hook baseline
- **Tokens/sec** for each method
- **Peak memory** (MB) for each method
- **Output equivalence**: cosine_sim and max_abs_diff between methods

## Step 5: Batch Size Sweep

Run extraction at multiple batch sizes to find the optimal one. Start with powers of 2:

```bash
for bs in 8 16 32 64 128; do
    echo "=== batch_size=$bs ==="
    uv run python -m steering.extract_features command=extract num_shards=1 device=cuda batch_size=$bs output_dir=out/data/features_bs${bs} 2>&1 | tail -3
done
```

If batch_size=128 causes OOM, that is useful data -- record the maximum viable batch size. Parse the ms/doc and us/token from each run to build a batch_size vs throughput table.

## Step 6: Diagnose Issues

Based on the data collected in Steps 2-5, check for each of these issues:

### Decision Tree

```
1. Is GPU utilization low (tokens/sec << 100K on a modern GPU)?
   |
   +-- YES: Is batch_size < 32?
   |   +-- YES --> DIAGNOSIS: "Low GPU utilization - batch too small"
   |   |           FIX: Increase batch_size (try 64, 128)
   |   +-- NO  --> Is tokenization time > 30% of total?
   |       +-- YES --> DIAGNOSIS: "I/O bottleneck - tokenization dominates"
   |       |           FIX: Pre-tokenize documents, or use faster tokenizer
   |       +-- NO  --> Is read_row_group time > 20% of total?
   |           +-- YES --> DIAGNOSIS: "I/O bottleneck - disk read dominates"
   |           |           FIX: Use SSD, prefetch row groups, increase batch_size
   |           +-- NO  --> DIAGNOSIS: "Possible CPU-GPU transfer bottleneck"
   |                       FIX: Profile with torch.profiler chrome trace
   |
   +-- NO: Is the run OOM-ing?
       +-- YES --> DIAGNOSIS: "OOM - batch too large"
       |           FIX: Reduce batch_size, use gradient checkpointing
       +-- NO  --> Is sae_encode > 40% of GPU time?
           +-- YES --> DIAGNOSIS: "SAE encode bottleneck"
           |           FIX: Consider quantizing SAE weights, or batch-encode
           +-- NO  --> DIAGNOSIS: "Pipeline is well-balanced"
                       SUGGESTION: Scale to more shards
```

### Performance Thresholds

| Metric | Good | Needs Improvement | Poor |
|---|---|---|---|
| Throughput (tokens/sec, GPU) | > 100,000 | 50,000 - 100,000 | < 50,000 |
| Throughput (tokens/sec, CPU) | > 5,000 | 1,000 - 5,000 | < 1,000 |
| ms/doc (GPU) | < 50 | 50 - 200 | > 200 |
| ms/doc (CPU) | < 500 | 500 - 2000 | > 2000 |
| Tokenization % of total time | < 10% | 10% - 30% | > 30% |
| SAE encode % of total time | < 30% | 30% - 50% | > 50% |
| Disk I/O % of total time | < 5% | 5% - 20% | > 20% |
| nnterp speedup over TL | > 1.5x | 1.0x - 1.5x | < 1.0x (regression) |
| Batch utilization | > 80% | 50% - 80% | < 50% |
| Peak GPU memory | < 70% VRAM | 70% - 90% VRAM | > 90% VRAM |

## Step 7: Generate Report

After collecting all data, output a formatted report using this template:

```
============================================================
SAE FEATURE EXTRACTION PROFILING REPORT
============================================================

Environment:
  GPU: [device name]
  VRAM: [X.X GB]
  CUDA: [version]
  Parquet shards available: [N]

------------------------------------------------------------
THROUGHPUT
------------------------------------------------------------
  Tokens/sec:  [X,XXX]  [GOOD/NEEDS IMPROVEMENT/POOR]
  Docs/sec:    [X.X]
  ms/doc:      [X.X]
  us/token:    [X.X]

------------------------------------------------------------
TIME BREAKDOWN (% of total)
------------------------------------------------------------
  GPT-2 forward:     [XX.X%]  [bar: ########..]
  SAE encode:        [XX.X%]  [bar: ####......]
  Tokenization:      [XX.X%]  [bar: ##........]
  Sparse extraction: [XX.X%]  [bar: #.........]
  Disk I/O:          [XX.X%]  [bar: #.........]
  Other:             [XX.X%]  [bar: ..........]

------------------------------------------------------------
BOTTLENECK
------------------------------------------------------------
  Primary bottleneck: [GPT-2 forward / SAE encode / Tokenization / I/O]
  Diagnosis: [one-line description]
  Suggested fix: [one-line recommendation]

------------------------------------------------------------
BATCH SIZE SWEEP
------------------------------------------------------------
  batch_size |  ms/doc  |  tokens/sec  |  peak_mem
  -----------|----------|--------------|----------
       8     |  [X.X]   |  [X,XXX]     |  [X MB]
      16     |  [X.X]   |  [X,XXX]     |  [X MB]
      32     |  [X.X]   |  [X,XXX]     |  [X MB]
      64     |  [X.X]   |  [X,XXX]     |  [X MB]
     128     |  [X.X]   |  [X,XXX]     |  [OOM / X MB]

  Recommended batch_size: [N]
  Reason: [best throughput without OOM / diminishing returns after N]

------------------------------------------------------------
BACKEND COMPARISON
------------------------------------------------------------
  Backend           |  Median (ms)  |  Speedup  |  Tokens/sec  |  Peak MB
  ------------------|---------------|-----------|--------------|----------
  full_forward+hook |  [X.X]        |  1.00x    |  [X,XXX]     |  [X]
  stop_at_layer     |  [X.X]        |  [X.XX]x  |  [X,XXX]     |  [X]
  nnterp+TL_weights |  [X.X]        |  [X.XX]x  |  [X,XXX]     |  [X]

  Recommended backend: [nnterp / transformerlens / auto]
  Reason: [nnterp is X.Xx faster with matching outputs / nnterp not available]

  Output equivalence:
    stop_at_layer vs baseline: cosine_sim=[X.XXXX], max_abs_diff=[X.Xe-X]
    nnterp+TL vs baseline:    cosine_sim=[X.XXXX], max_abs_diff=[X.Xe-X]

------------------------------------------------------------
10B TOKEN EXTRAPOLATION
------------------------------------------------------------
  Estimated documents:     [X,XXX,XXX]
  Estimated storage:       [XXX GB]
  Estimated compute time:  [XX GPU-hours]
  Estimated wall time:     [XX hours on 1 GPU / XX hours on 4 GPUs]

------------------------------------------------------------
RECOMMENDATIONS
------------------------------------------------------------
  1. [Primary recommendation based on bottleneck]
  2. [Secondary recommendation]
  3. [Configuration: batch_size=N, backend=X, device=cuda]

============================================================
```

Fill in every field from the data collected. For any step that failed or was skipped, note it in the report as "N/A - [reason]".

## Notes

- All commands assume the working directory is the project root: `/home/d0rb/Documents/Github/nano-gpt-pretrain-steer`
- The Hydra-based extraction uses config at `configs/extract_features.yaml`. Override values on the command line with `key=value` syntax (no `--`).
- The profiler script at `scripts/profile_extraction.py` uses argparse and takes `--flag` style arguments.
- The benchmark script at `scripts/benchmark_activation_methods.py` also uses argparse.
- Output directories used during profiling (out/data/features_tl, out/data/features_nnterp, out/data/features_bsN) can be cleaned up after the report is generated.
- If the user passes additional arguments when invoking this skill (e.g., `--device cpu`, `--batch-size 32`), incorporate those into the commands above.
