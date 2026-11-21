# Precomputed Packed CUDA Staging

This document defines the issue-#842 CUDA staging path reused by issue #868 for
the verified remote packed encoder-decoder training run. It is not a
replacement for `precomputed_data_loader` or issue-#853 rank-striped
compressed-body transport: transport stops at verified compressed bytes, while
this downstream layer provides deep decoded-batch lookahead.

## Scope

The path may run only when all of these are true:

- `remote_training.train_packed.cuda_staging` is present;
- training is using the verified remote packed and remote selection
  materializers;
- a selection directory is present (the fused selection path is forbidden);
- the rank has an explicitly indexed CUDA device;
- batch size, context length, segment count, and selected-feature width are
  static;
- `label_dropout=0.0`, and resumed dataloader state contains no
  `label_dropout_rng_state`.

Local packed training, live training, and validation loaders retain their
existing behavior. Remote packed training without this block still uses the
strict rank-striped body workers and the explicitly closable CPU batch
producer. Decode is therefore already backgrounded on the unstaged route, not
performed by the training thread. Its queue is only `grad_accum_steps` deep,
however: the production B=16/world-size-8 geometry gives four microbatches, one
optimizer step or about 0.7 seconds. The producer cannot enter shard N+1 until
only those four shard-N batches remain, which cannot hide the measured
approximately four-second packed-body decode.

Changing `device_batch_size` at fixed global batch is handled below this layer
by ordered-row re-batching. The staging validator sees only the resulting new
fixed batch shape and stages its tensors without changing contents. The
existing staging prohibition on nonzero `label_dropout` remains unchanged;
CPU/CUDA loader modes that were already valid preserve dropout masks through a
canonical saved-width generator.

## Configuration

The CUDA staging block has exactly two required keys. The shared remote body
retention depth is configured separately under `body_prefetch`:

```yaml
remote_training:
  body_prefetch:
    cache_shards: 24
  train_packed:
    trust_existing_cache: true
    cuda_staging:
      host_queue_bytes: 536870912
      pinned_slots: 3
```

- `host_queue_bytes` is a positive strict integer. It bounds unique CPU tensor
  storage retained in the pageable lookahead queue. A single batch larger than
  the cap fails; the implementation never silently exceeds the cap.
- `pinned_slots` is a positive strict integer. Each slot owns one reusable
  pinned copy of the fixed batch tensor schema. It also bounds the ready GPU
  queue.

These are operational values and remain excluded with the rest of
`remote_training` from the semantic remote-run fingerprint. Adding or removing
the block therefore does not change model/data identity or checkpoint resume
identity. It does change the recorded operational user configuration.

The hardware-agnostic remote-packed preset intentionally leaves
`cuda_staging` absent. The target-specific
`cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_2xh200`
preset enables the values above for its two-H200 run. The issue-#868
`cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100`
target preset enables the same bounded mechanism for the exact production
B=16/BF16 run while retaining the parent run/checkpoint namespace.

For that H100 geometry, the fixed tensor fields retain exactly 2,539,776 bytes
per batch:

| Fields | Retained bytes |
|---|---:|
| `inputs`, `targets`, `seg_map` | 393,216 |
| `enc_fids`, `enc_mags` | 319,488 |
| `enc_mask`, `enc_pcts` | 99,840 |
| `seg_doc_id` | 1,664 |
| compact `fp_targets` (`V=2074`) | 1,725,568 |
| **Total** | **2,539,776** |

The 512 MiB queue admits 211 such batches. A 16,384-row packed body contributes
128 rank-local B=16 batches at world size eight, so the producer can finish the
entire current stripe and enter N+1 before the consumer drains any current
batch. That exposes up to 32 optimizer steps, approximately 22.4 seconds at the
reported 0.7 seconds/step, to hide the measured approximately four-second
decode. The queue still enforces exact retained-storage accounting at runtime;
this arithmetic does not waive the `host_queue_peak_bytes` production gate.

## Dataflow and ownership

```text
spawned rank owner -> shared compressed-body cache
        |
        v
descriptor-pinned all-rank read and decode
        |
        v
precomputed_data_loader(device="cpu")
        |
        v
byte-bounded pageable CPU FIFO
        |
        v
reusable pinned slot -- dedicated cuda:<local-rank> stream
        |
        v
event-complete GPU batch FIFO
        |
        v
consumer stream wait + Tensor.record_stream
        |
        v
unchanged training loop
```

These two CUDA-staging background workers have separate jobs from the spawned
body owner:

1. The host producer advances the ordinary CPU precomputed loader and admits
   complete batches to a FIFO only when their retained storage fits the byte
   cap. This is the deep lookahead that can absorb remote shard-fetch tails.
2. The H2D worker copies one CPU batch synchronously into a leased pinned slot,
   allocates fresh CUDA destinations on a dedicated stream for that rank,
   starts non-blocking copies, records an event, and waits for that event before
   publishing the ready batch.

Memory ownership is intentionally split:

- issue-#853 body workers and the shared cache own signed compressed bytes,
  hidden transport stages, receipts, and their disk admission;
- the staging host producer temporarily owns zstd output, safetensors parsing,
  decoded packed/selection tensors, issue-#826 verification, semantic
  validation, and batch reconstruction while advancing the ordinary loader;
- the pageable FIFO owns only complete reconstructed batch tensor storages and
  is bounded by `host_queue_bytes`;
- `pinned_slots` owns the reusable pinned copies and bounds the ready CUDA FIFO;
- each ready item owns fresh CUDA destinations until consumer-stream
  `record_stream` transfers allocator lifetime responsibility.

There is no decoded-shard cache and no second decoder. Transient decompression
and the current decoded shard are distinct from queue bytes and must be measured
separately in the H100 RSS gate.

The ready message retains the pinned-slot lease through consumer handoff. The
consumer makes its actual current stream wait on the event and calls
`record_stream` on every CUDA tensor before returning the batch to the unchanged
training loop. The slot is then reusable because the H2D read is complete;
the fresh CUDA destinations are not reused.

This handles the two independent allocator-lifetime obligations:

- pinned sources cannot be reused while an asynchronous H2D copy may still
  read them;
- CUDA destinations allocated on the staging stream cannot return to the
  allocator before work recorded on the consumer stream completes.

The implementation never calls process-wide `torch.cuda.synchronize()`.

## Training-step timing contract

The training loop also deliberately has no device-wide synchronize immediately
before its step timer. Both loader routes establish producer-to-consumer
ordering explicitly: the legacy prefetch wrapper queues a CUDA event, and this
staging path makes the consumer stream wait on the transfer event and records
CUDA tensor lifetimes on that stream. A pre-timer device barrier would therefore
add no batch-correctness guarantee and would wait for unrelated work already
enqueued on the dedicated staging stream, serializing H2D prefetch against
training.

The post-optimizer device synchronize remains between `t0` and `t1`. It closes
the timed training step before `dt`, tokens/sec, and MFU are computed, so the
cost the trainer does pay remains visible in those metrics.

## Static batch contract

Every staged item must be the exact 10-tuple from
`precomputed_data_loader(device="cpu")`:

1. `inputs`: `[B, T]`, int64
2. `targets`: `[B, T]`, int64
3. `enc_fids`: `[B, n_seg, F]`, int64
4. `enc_mags`: `[B, n_seg, F]`, int64
5. `enc_mask`: `[B, n_seg, F]`, bool
6. `enc_pcts`: `[B, n_seg, F]`, float32
7. `seg_map`: `[B, T]`, int64
8. `seg_doc_id`: `[B, n_seg]`, int64
9. `fp_targets`: `None` or `[B, n_seg, V]`, float32
10. the string-keyed resumable state dictionary

The first nine fields' tensors must be CPU, contiguous/strided, gradient-free,
and have zero storage offset. Shape, dtype, stride, FP-target presence/width,
and state-key schema are fixed by the first batch and checked on every later
batch. Drift is a hard error. Any tensors nested in the state dictionary must
remain CPU and gradient-free; the state is metadata and is never transferred.

The state dictionary itself is not transferred or rewritten. The exact state
attached to the consumed batch remains the checkpoint cursor, even if the host
worker has materialized later batches. Fresh, legacy-to-staged,
staged-to-legacy, same-shard, cross-shard, cross-epoch, and per-rank resume
parity are therefore defined against the ordinary loader outputs.

## Failure and shutdown behavior

Host completion and exceptions are typed FIFO messages; H2D completion and
exceptions are independently typed FIFO messages. Producer exceptions are
re-raised on the consumer thread with their original type. Transfer or schema
errors fail the run; there is no runtime fallback to the legacy path.
Remote packed and selection decode/semantic-validation failures carry exception
notes with the exact body path and logical epoch; the original exception and
traceback remain primary unless a stronger terminal cleanup failure occurs.

All queue waits poll the shared stop event. Normal completion, early consumer
close, transfer error, and the training loop's `finally` block signal both
workers and join them. The same cancellation state stops the spawned
rank-striped owner before teardown continues. A worker that cannot terminate
within the lifecycle deadline is a hard error rather than a silent thread
leak.

Packed and selection readers never call the backend. They publish a
requester-specific demand, poll the nominal owner's status, then descriptor-pin
the owner-published body. A transport call that does not return during graceful
shutdown is confined to the spawned owner's unique PID/UUID temporary and may
trigger bounded hard process termination. That temporary cannot publish after
the process dies and is swept only after a later startup proves its PID dead.
There is no consumer-side daemon transport boundary or ownership takeover.

On close, the loader logs produced/staged/consumed counts, peak queued host
bytes, producer blocking time, H2D time, and consumer wait time. These are
diagnostic counters, not semantic training metrics.

## Validation

Fast tests cover strict configuration parsing, eligibility, every tensor field,
static-schema drift, byte accounting, byte backpressure, cancellation through
remote lock/download waits, deterministic N+1 decode lead, exact H100 preset
parity, corrupt-body attribution, and remote-run fingerprint parity:

```bash
uv run pytest tests/steering/test_precomputed_cuda_staging.py
```

Single-GPU slow tests add exact legacy-output parity (including state) for
same-shard, cross-shard, cross-epoch, both synthetic DDP ranks, fresh/resume,
FP-disabled, and full-width FP cases. They also exercise event handoff without
a global synchronize, a 64-batch host queue across 1,000 source-slot reuse
cycles, allocator churn, exception propagation, and early close:

```bash
uv run pytest tests/steering/test_precomputed_cuda_staging.py -m slow
```

The explicit two-rank NCCL gate constructs real packed and selection shards
whose rank-striped B=16 batches cross shard boundaries. On each indexed
`cuda:0`/`cuda:1` rank it compares the legacy and staged paths field-for-field,
including FP targets and resume state, fills at least 32 batches of a bounded
64-batch host queue, checks token/feature/magnitude bounds, and then runs BF16
forward/backward through static `torch.compile` at the production
T/n_seg/F/V dimensions:

```bash
RUN_ISSUE842_DDP=1 uv run pytest \
  tests/steering/test_precomputed_cuda_staging_ddp.py -m slow
```

The issue-#868 production gate must be launched by `torchrun` on eight H100s
against immutable packed-row and selection directories. It compares staged
and ordinary batches exactly on every rank through at least three shard/epoch
boundaries, snapshots an exact continuation, resumes from that state, and
then closes the non-exhausted loaders twice to prove cancellation and worker
termination:

```bash
RUN_ISSUE868_H100_DDP=1 \
ISSUE868_PACKED_ROW_DIR=/immutable/path/to/packed/rows \
ISSUE868_SELECTION_DIR=/immutable/path/to/selection \
  uv run torchrun --standalone --nproc_per_node=8 -m pytest -q \
  tests/steering/test_precomputed_cuda_staging_ddp.py \
  -m very_slow -k issue868
```

The H100 target preset is now enabled as an operational configuration, but its
performance benefit remains unproven until the production A/B runs. Benchmark
staged and unstaged arms from the same immutable checkpoint, revision,
artifacts, rank topology, and equivalent isolated cache states. Report
warm-cache and cold-cache results separately, alternate arm order, and collect
at least three runs with twelve post-warmup boundaries per arm. Acceptance
requires:

- zero tensor/state parity mismatches or bound assertions;
- no post-warmup 32-step boundary above two seconds;
- a staged-batch rate that remains above the consumer rate;
- `host_queue_peak_bytes <= host_queue_bytes` on every rank;
- no monotonic host RSS, pinned memory, HBM, thread, descriptor, or temporary
  file growth;
- terminal workers after normal exit, decode failure, and interruption.

Issue #867 remains an independent control-plane polling cost. This issue-#868
change is stacked on PR #870, which supplies the polling fix; both A/B arms
must use that same stacked revision so the staging comparison changes only
issue #868. The gate measures removal of the incremental boundary decode spike.
The projected approximately 710 ms step time is not a current measured result.
