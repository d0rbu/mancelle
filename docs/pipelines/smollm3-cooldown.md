# SmolLM3-3B Pretraining Cooldown

This document is the implementation specification, operator runbook, and evidence
log for continuing the public SmolLM3-3B 200k checkpoint on the repository's
pretraining corpus. It deliberately uses Transformers `Trainer` instead of the
repository's nanochat pretraining stack: the source is a Transformers-native
SmolLM3 model and no SAE conditioning is involved.

## Immutable experiment contract

### Source

| Field | Value |
|---|---|
| Repository | `HuggingFaceTB/SmolLM3-3B-checkpoints` |
| Revision | `8089bf9b71ead724010855dd6fabd1deb90b70ff` |
| Checkpoint | repository root, step 200,000 |
| Source tokens | 471,859,200,000 |
| Context length | 4,096 |
| Tokenizer EOS | 128001 (`<|end_of_text|>`) |
| Parameter load dtype | FP32 |
| Forward/backward autocast | BF16 |
| Attention | FlashAttention 4 |

The source repository publishes model and tokenizer state but no optimizer,
scheduler, RNG, or Trainer state. This is therefore a **weights-only
continuation with a fresh FP32 AdamW optimizer**, not an exact resume of the
upstream optimizer trajectory. The BF16 checkpoint tensors are promoted to
FP32 parameter storage exactly as loaded; no missing optimizer state is
reconstructed or guessed.

The code reuses the repository's strict `SMOLLM3_CONTRACT`, source-config,
source-tokenizer, and loaded-model validators. It must not use
`HFSFTTokenizer`: that adapter intentionally changes the generation EOS to the
chat `<|im_end|>` token (128012), while pretraining requires the immutable
source EOS (128001).

### Data

The approved production input is the 1,400 contiguous derived/spliced parquet
shards at:

```text
/orcd/pool/005/henryac/nano-gpt-pretrain-steer/full3b-clustered-c2048-20260823T194759Z/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/splice/cut_frac0.15/seed42/_parquets
```

Shards 0 through 1,398 are the training source. Shard 1,399 is reserved for
held-out evaluation and preprocessing never reads it into the training
artifacts. Files must be named exactly `shard_00000.parquet` through
`shard_01399.parquet`; gaps, duplicates, extra parquet files, or a different
count fail before output publication.

Preprocessing iterates parquet row groups and their `text` column in the same
order as `steering.parquet_io.read_parquet_texts`. Each document is encoded by
the pinned low-level tokenizer with `add_special_tokens=False`, then EOS 128001
is appended exactly once. Two files are published per input shard:

- `<stem>.ds`: little-endian unsigned 32-bit token IDs.
- `<stem>.ds.index`: little-endian unsigned 64-bit exclusive document-end
  offsets into the token file.

A shard receipt binds the source path/stat identity, tokenizer identity,
document/token/context counts, and SHA-256/byte size of both bodies. Bodies and
the receipt are written atomically, with the receipt last. A root manifest is
written last after validating the smallest deterministic training-shard prefix
whose complete windows cover the approved horizon. This avoids preparing and
storing unused suffix shards while the complete 1,400-shard source inventory
and held-out final shard are still validated. Existing complete byte-identical
shards are reused; incomplete or mismatched output is rejected rather than
overwritten.

The runtime dataset memory-maps one token file and one index file at a time.
Within each source file it enumerates 4,097-token windows with a 4,096-token
stride and drops that file's final incomplete tail. The final target of one
window is therefore the first input of the next, so every available causal
transition is supervised exactly once; windows never join across shard
boundaries. A 4,097-token window becomes 4,096 input tokens and 4,096
next-token labels.

Position IDs start at zero for every context and reset to zero after every EOS
boundary derived from the document index. A target is ignored precisely when
its source token is the first token of a new document; equivalently, labels at
positions whose shifted position ID is zero are `-100`. EOS remains a scored
target. This prevents learning transitions between documents while retaining
all within-document and document-to-EOS predictions.

### Batch and loss topology

The upstream global batch is 576 contexts, or 2,359,296 scheduled tokens per
optimizer update. Each device loads three contexts. The collator concatenates
them into one `[1, 12,288]` row for FlashAttention's variable-length document
path and supplies:

- `input_ids`: three concatenated 4,096-token inputs;
- `position_ids`: independently reset positions;
- `labels`: explicit shifted targets with cross-document boundaries masked;
- `shift_labels`: the same tensor, activating SmolLM3's explicit-label path.

No `attention_mask` is supplied. Passing `[3, 4096]`, an all-ones attention
mask, or omitting `shift_labels` would change attention or label semantics and
is rejected by shape/value assertions and tests.

The required gradient-accumulation count is
`576 / (3 * world_size)`. Supported world sizes are therefore divisors of 192:
1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, or 192. The intended production
64-GPU run uses three micro-steps. A two-GPU canary uses 96.

Transformers counts non-ignored `shift_labels` across the accumulated and
distributed batch. `average_tokens_across_devices=true` therefore produces a
valid-token-weighted loss despite variable document boundaries.

### Horizon and learning rate

The repository target is 524,288,000,000 nominal tokens. Starting at
471,859,200,000 and retaining complete upstream-sized batches gives:

```text
target delta                      52,428,800,000 tokens
22,222 complete updates           52,428,275,712 tokens
final nominal token position     524,287,475,712 tokens
shortfall                                524,288 tokens
22,223 updates would overshoot           1,835,008 tokens
required training contexts             12,799,872
final absolute step label                  222,222
```

`max_steps` and the number of exposed dataset contexts are derived from those
constants; they are not independently configurable.

The standard Transformers linear scheduler runs for 22,222 steps with 2,000
warmup steps and peak learning rate 2e-4. With scheduler stepping after each
optimizer update, the learning rates used for updates 1, 2, 2,000, 2,001,
2,002, and 22,222 are respectively 0, 1e-7, 0.0001999, 0.0002,
0.00019999010978, and approximately 9.89021857e-9; after the final update the
scheduler is at zero. Update 1 still advances Adam moments and step counters.

AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, and fused
Torch kernels. Every unique trainable parameter except the exact tied token
embedding parameter `model.embed_tokens.weight` receives weight decay. Gradient
norm is clipped to 1.0 and a non-finite pre-clip norm aborts before the
optimizer step.

### Checkpoint and resume safety

Trainer writes full model, optimizer, scheduler, RNG, and Trainer state. A
callback validates the expected checkpoint inventory, synchronizes all ranks,
and writes `cooldown_complete.json` atomically and last on rank zero. The marker
contains source/data identity, semantic config digest, world/local-world size,
batch topology, source/final token arithmetic, code and lockfile identity, and
the byte size of every required checkpoint file.

Startup resumes only the latest Trainer checkpoint with a valid marker whose
semantic identity exactly matches the current run. Unmarked `checkpoint-*`
directories are preserved and renamed with a `.partial` suffix so Trainer
cannot select them accidentally. Malformed names, collisions, unexpected
marked identities, missing bodies, or unknown checkpoint layouts fail loudly.
Operational logging/worker/save cadence may change; source, data, target,
attention, precision, optimizer, world topology, batch, seed, code, and lock
identity may not.

### Runtime and evaluation

Training uses ordinary `Trainer` DDP under `torchrun`; there is no custom
training loop, loss function, sampler, optimizer, or scheduler. The only
Trainer overrides are the exact weight-decay name list and finite-gradient-norm
guard. Activation checkpointing is enabled so the fixed B3 microbatch fits
after FP32 Adam state exists; compilation, FSDP, tensor parallelism, FP8, and
TF32 remain disabled.

The launcher runs one `srun` task per node and one `torchrun` process per GPU,
uses the repository's bounded NCCL preflight, the checked-in `.venv`, a static
rendezvous, and `--max_restarts=0`. It performs no installs and forces HF
offline mode after the pinned snapshot and prepared data have been staged.

No in-loop evaluation is enabled. The last parquet shard remains untouched for
an out-of-process LightEval pass; a future exact valid-token-weighted evaluator
may be added only if the initial experiment needs it.

Trainer logs scheduled and supervised tokens/second, steps/second, loss,
learning rate, pre-clip gradient norm, peak allocated memory, absolute nominal
token position, and ETA. MFU is an estimate using
`HFDecoderModelAdapter.estimate_flops()` and `nanochat.common.get_peak_flops()`;
logs state that formula explicitly.

## Test and acceptance matrix

The default CPU suite covers arithmetic, scheduler values, source/config
validation, exact tokenizer behavior, binary encoding/indexing, atomic resume
and corruption handling, context enumeration, position resets, boundary loss
masks, collator shape/content, gradient accumulation, decay groups, finite norm
guard, semantic digest/marker selection, and Hydra/launcher wiring.

Subprocess functional tests prepare multiple real parquet row groups, restart
preprocessing, load the published dataset with multiple workers, train a tiny
SmolLM3 configuration through real `Trainer`, save, interrupt, and resume while
checking exact step/LR/data continuity.

Opt-in slow acceptance tests require the pinned full 3B checkpoint, production
parquets, and CUDA. They validate the exact 3,075,098,624-parameter model and
326 unique parameter names, prove 325/1 decay grouping, compare collator inputs
and model losses against an independent reference on real text, run one BF16
FA4 forward/backward, and execute single- and distributed-Trainer canaries with
full-size 4,096-token contexts. The distributed harness requires its expected
world and local-world sizes explicitly whenever they differ from the original
2/2 gate, preventing a scheduler shortfall from silently weakening acceptance.
The production launch is accepted only after a distributed canary saves a
marked checkpoint and a resume canary consumes its next batch without replay.

## Implementation addenda

Addenda are appended here when implementation evidence changes or narrows the
specification. They must record evidence, not silently rewrite a historical
decision.

- 2026-08-25: implementation started from `origin/main` commit
  `fcb1ac139592496fadb557090d2397deb61e8317`.
- 2026-08-25: the exact pinned tokenizer snapshot was inspected locally. Its
  tokenizer-level IDs are BOS `null`, EOS 128001, and PAD `null`; token 128000
  still resolves to `<|begin_of_text|>` and the model config retains
  `bos_token_id=128000`. Pretraining calls `add_special_tokens=False`, so this
  distinction is asserted without mutating the tokenizer.
- 2026-08-25: preparation was narrowed to the smallest sufficient contiguous
  training-shard prefix. At 52.428B nominal cooldown tokens, uint32 token
  bodies alone are about 210 GB; preparing all unused suffix shards would
  increase storage without changing any training example.
- 2026-08-25: exact supervised-token telemetry was added without a Trainer
  override. It counts document-boundary masks from immutable uint64 indices for
  each logged global context interval; scheduled throughput and MFU remain
  separately labeled.
- 2026-08-25: Engaging job 21249955 completed the no-fallback one-H200 gate on
  the exact pinned checkpoint and production `shard_00000.parquet`: 4 tests
  passed in 624.20 seconds. The gate proved the 3,075,098,624-parameter and
  326-name inventory, 325/1 decay split, `[1, 12288]` FA4 BF16
  forward/backward, exact independent cross-entropy equality, finite gradient
  norm/clipping, and little-endian token/index boundaries. Slurm completed
  with exit code 0 and peak host RSS 25,599,692 KiB.
- 2026-08-25: `tests/steering/smollm3_cooldown_ddp_real.py` was added as a
  torchrun-only two-H200 acceptance harness. It uses 1,152 contexts from one
  receipt-backed real shard, B3/GAS96, the unchanged 22,222-step scheduler,
  marker-last checkpoint 1, and a second invocation that must resume to
  checkpoint 2.
- 2026-08-25: final token-accounting review rejected an initial 4,097-token
  stride before resume acceptance. That stride would have omitted one causal
  target between adjacent contexts. Prepared schema v2 now records an exact
  4,096-token stride, overlaps each prior final target as the next input, and
  tests both the overlap and exact document-boundary supervised-token count.
  The pre-correction checkpoint is retained as diagnostic evidence only and is
  not an accepted cooldown checkpoint.
- 2026-08-25: a schema-v2 two-H200 first step fit at B3, but exact resume
  exposed the real steady-state peak after FP32 Adam moments exist. Both ranks
  OOMed in the 5.87-GiB FP32 causal-loss buffers; allocator defragmentation was
  tested separately and remained insufficient. Activation checkpointing is
  therefore enabled and bound into checkpoint semantic identity. It preserves
  model, data, batch, loss, optimizer, and schedule while recomputing forward
  activations during backward. The non-checkpointed optimizer state is not
  eligible for acceptance or resume.
- 2026-08-25: Engaging job 21254275 reran the one-H200 real-data gate against
  the corrected schema v2 contract and completed with exit code 0. All four
  tests passed in 474.70 seconds using the exact pinned 3,075,098,624-parameter
  checkpoint and a production parquet shard. Its receipt records 55,595
  documents, 41,355,582 tokens, 10,096 contexts, and the required 4,096-token
  stride. The gate again proved the 326-name inventory, 325/1 decay split,
  `[1, 12288]` FA4 BF16 forward/backward, exact independent cross-entropy
  equality, finite gradient clipping, and little-endian token/index bodies.
  Slurm reported peak host RSS 25,267,276 KiB.
- 2026-08-25: two-H200 controls isolated the steady-state memory requirement.
  Schema-v2 job 21255750 completed update 1 without activation checkpointing,
  but its exact-resume job 21256270 OOMed after restoring the FP32 Adam state.
  Job 21257034 repeated the resume with expandable CUDA allocator segments and
  failed at the same pair of 5.87-GiB FP32 causal-loss allocations. This rules
  out allocator fragmentation as a sufficient correction; neither control
  checkpoint is accepted for cooldown training.
- 2026-08-25: the final activation-checkpointed two-H200 acceptance pair passed
  on real 4,096-token contexts at B3/GAS96. Job 21257592 completed update 1 with
  loss 2.4586195945739746, pre-clip gradient norm 0.2886574864387512, learning
  rate 0, 30,611.9 scheduled tokens/s, 30,571.4 supervised tokens/s, and
  estimated MFU 0.3416. Job 21258054 then restored its complete 32-GB
  `checkpoint-1`, completed only update 2 with loss 2.4737582206726074, finite
  gradient norm 0.28467482328414917, and the exact expected learning rate
  1.0000000000000001e-7, and published a complete 35-GB `checkpoint-2` with
  the marker last. The resumed update measured 20,777.8 scheduled tokens/s,
  20,748.8 supervised tokens/s, and estimated MFU 0.2743; this single-update
  canary is a correctness and memory gate, not a steady-state throughput
  benchmark. Slurm completed both jobs with exit code 0. The resume job's
  15:09 wall time included unusually slow shared-filesystem reads and writes of
  the 24.6-GB optimizer state; peak observed device allocation was about
  91.9 GiB per H200 and peak host RSS was 89,122,664 KiB.
- 2026-08-26: PR #1503 generalized only the acceptance harness topology guard,
  retaining 2/2 as its default while requiring explicit expected world and
  local-world sizes for other layouts. A real two-node/one-H200-per-node gate
  then ran from exact PR head `78d518c1ceeec604bab1392e058417da1e203da6`
  with `WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, B3, and GAS96. The bounded NCCL
  preflight passed on `node4803` and `node4900` with the default transport; its
  43-MiB collective medians were 1.081 ms all-reduce, 1.133 ms reduce-scatter,
  and 1.207 ms all-gather. Job 21260604 completed update 1 and marker-last
  `checkpoint-1` with loss 2.4586195945739746, finite gradient norm
  0.2886609435081482, learning rate 0, 26,338.8 scheduled tokens/s, 26,304.0
  supervised tokens/s, and estimated MFU 0.2939. Independent job 21261203
  restored that full checkpoint, retained its step-1 history, completed only
  update 2 with loss 2.4737582206726074, finite gradient norm
  0.28468048572540283, and the exact expected learning rate
  1.0000000000000001e-7, then published marker-last `checkpoint-2`. The resumed
  update measured 30,618.0 scheduled tokens/s, 30,575.3 supervised tokens/s,
  and estimated MFU 0.3417. Both Slurm jobs exited 0; peak observed resumed
  device allocation was about 92.0 GiB per H200. The final marker binds
  `world_size=2`, `local_world_size=1`, both real-data body hashes, the pinned
  source revision, and activation checkpointing under semantic digest
  `72756e83e9a6d94aa3021518acdbfa24aea5ae707399771beefbdd55382ed288`.
