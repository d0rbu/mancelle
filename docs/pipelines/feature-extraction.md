# Feature Extraction Pipeline

Extract SAE (Sparse Autoencoder) features from pre-training data using GemmaScope models. This is the first step before encoder-decoder training.

> **See also:** for the Gemma 3 / gemma-scope-2 (v2) release family, see [`gemma-scope-2.md`](gemma-scope-2.md). All extraction mechanics below (pooling, segmentation, CSR shard format, validation) are shared; only the model family, SAE IDs, and L0 representation differ.

> **Optional post-step:** to recombine the extracted segments into a derived corpus where documents are joined at segment boundaries (for studying cross-document feature distributions) — **without re-running the model**, carrying each segment's features verbatim — splice the feature shards after `compute_statistics`. See [`feature-shard-splice.md`](feature-shard-splice.md). The derived directory is compatible with the core downstream pipeline (statistics → percentiles → token-boundaries → packed-rows → training). Two re-tokenization consumers (`precompute_gradient_attribution` and `validate_features` source cross-ref) support **root-started docs only** and raise on new-started docs; see the compatibility matrix in [`feature-shard-splice.md`](feature-shard-splice.md#overlay--statistics-interaction).

## Overview

```
Parquet shards → Tokenize (Gemma) → Forward pass (TransformerLens + SAE)
    → Segment → Pool → Sparse CSR format → Validate → Safetensors shards
```

**Scripts:** `steering/extract_features.py` (main entry), `steering/compute_statistics.py`, `steering/validate_features.py` (validation)
**Configs:** `configs/extraction/extract_features.yaml`, `configs/extraction/compute_statistics.yaml`, `configs/extraction/validate_features.yaml`

## Quick Start

```bash
# Extract features from 1 shard (default)
python -m steering.extract_features

# Extract from 370 shards on 4 GPUs
torchrun --standalone --nproc_per_node=4 -m steering.extract_features num_shards=370

# Use hardware-specific config
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    --config-name extract_features_4xh200 \
    num_shards=370
```

## End-to-End Pipeline (4xH200)

```bash
bash runs/encdec_4xh200.sh
```

This runs: data download → feature extraction (Phase 1) → statistics computation (Phase 2).

## Detailed Commands

### Feature Extraction

```bash
# Single GPU, basic
python -m steering.extract_features \
    num_shards=10 \
    output_dir=out/data/features

# Multi-GPU with custom config
torchrun --standalone --nproc_per_node=4 \
    -m steering.extract_features \
    num_shards=370 \
    output_dir=out/data/features

# Different SAE width
python -m steering.extract_features \
    sae.num_features=131k \
    layer=12

# Different activation type
python -m steering.extract_features \
    activation_type=mlp

# Extract from SFT data (not base pretraining data)
python -m steering.extract_features \
    dataset_phase=safety_sft \
    num_shards=10

# Resume from a specific shard
python -m steering.extract_features \
    start_shard=50 \
    num_shards=370

# Manual multi-GPU (without torchrun)
RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 python -m steering.extract_features num_shards=300 &
RANK=1 WORLD_SIZE=2 LOCAL_RANK=1 python -m steering.extract_features num_shards=300 &

# Independent jobs targeting the same output_dir: use disjoint exclusive windows
python -m steering.extract_features start_shard=0 num_shards=150 &
python -m steering.extract_features start_shard=150 num_shards=300 &
```

### Statistics Computation

```bash
# Compute corpus-wide feature statistics
python -m steering.compute_statistics \
    feature_dir=out/data/features \
    device=cuda

# Output: feature_dir/feature_statistics.safetensors
```

`compute_statistics` records the shards' gemma-scope-2 `l0_tier` in the
stats file (mirroring shard metadata; `None` for v1) and enforces
cross-shard L0 uniformity via the shared
`steering.feature_io.assert_uniform_l0_tier` rule — a directory whose
shards carry mixed tiers fails fast with `FeatureDirectoryMismatchError`
(the same single rule used by `verify_feature_dir`). This cross-tier check
reads only each shard's safetensors header (`read_shard_metadata`), not
the multi-GB body, so it does not add a per-shard full-decompress to
statistics startup. A corrupt, truncated, or empty shard fails fast with
`CorruptFeatureShardError` (naming the shard) instead of being read in
full or silently treated as a tier-less v1 shard.

Streaming statistics reads each raw shard body exactly once and does not write
sorted-magnitude scratch. Persisted magnitudes are positive finite float16, so
their 31,743 possible bit patterns are a finite ordered domain. The reducer
stores exact int64 multiplicities in a dense
`[n_sae_features, 31_743]` CPU table, then reconstructs the existing linear
percentiles, medians, and `torch.bucketize(edges[1:])` histogram counts from
cumulative integer counts. The existing float64 occurrence/moment reduction,
per-shard p99 sampling rule, output schema, and downstream percentile checksum
are unchanged.

CPU count updates add directly at validated composite keys within each bounded chunk; duplicate keys accumulate exact integer multiplicities without sorting, while CUDA retains its existing reduction.

Each finalization lazily shares one histogram-bin lookup across features (less than 1 MiB of lookup temporaries), using the same unrounded edges, endpoint clamps and lower-bin convention for exact interior-edge ties.

The table uses exactly `n_sae_features * 31_743 * 8` bytes: 15.4995 GiB at
65,536 features. Before allocation the command checks current host headroom and
the tightest finite cgroup-v2 headroom from the process' leaf group through
every ancestor, and fails if the table alone cannot fit. This is a necessary
gate, not a total peak-RSS guarantee: one decompressed shard, reducer chunks,
and CUDA moment transients need additional memory. Wider SAE dictionaries are
accepted only when this exact dense representation fits; the command never
falls back to terabyte-scale magnitude spills.

For resolved CUDA runs that retain per-feature detail and have at least two
shards left, one bounded worker reads and zstd-decompresses the next shard while
the main thread reduces the current shard. The worker returns plain
safetensors bytes; parsing, dtype restoration, delta decoding, validation, and
all PyTorch work remain on the main thread. There is exactly one lookahead
future, never two decoded shards. This adds at most one raw safetensors payload
plus its in-progress compressed input to the transient host-memory envelope.
Lookahead is suppressed whenever the current shard triggers a checkpoint, so a
raw next shard cannot coexist with the count-table snapshot. CPU, skip-detail,
eager, and one-remaining-shard runs retain sequential loading. This execution
choice is automatic and does not enter checkpoint or output identity.

Set `n_sae_features` explicitly for noncanonical directories. A canonical
`.../f16k/...`, `.../f65k/...`, etc. path resolves its declared width before
loading a shard body. For backward-compatible synthetic/library calls only, a
noncanonical path discovers the maximum ID across the complete family before
allocation; that compatibility path reads shard bodies twice. Pass an explicit
width to retain one-read behavior.

`device=cuda` accelerates the unchanged moment accumulators and the bounded
composite-key `unique` reduction used for exact magnitude counts. The dense
15.4995 GiB count table, its sparse integer updates, and finalization remain on
CPU:

```bash
python -m steering.compute_statistics \
    feature_dir=out/data/features/... \
    device=cuda \
    n_sae_features=65536
```

Finalization converts ordinary CPU percentile rows to Python values together;
custom percentile objects, tensor overrides, and active Torch modes retain the
scalar conversion path. The median is computed independently of the requested
percentile tuple, including an empty tuple or one without percentile 50.

Exact-count reduction uses the reducer's fixed 16,000,000-activation chunks;
`gpu_chunk_size` continues to control only the moment accumulators. A 16M exact
chunk reserved about 967 MB (922 MiB) on the measured RTX 4090, but the requirement can
vary with GPU, CUDA, and PyTorch versions. Once the runtime resolves to CUDA,
an exact-count CUDA OOM fails the run instead of silently resizing or falling
back. Use `device=cpu` when that transient does not fit.

The `compute_statistics_gemma3_l22_f65k.yaml` config pins this width for the
65k Gemma-3 pipeline. `skip_per_feature_detail=true` skips the count-table
allocation entirely and retains aggregate count/moment statistics only.

#### Exact-pass checkpoints and resume

Resume is opt-in and applies only to the exact streaming path
(`streaming=true`, `activation_percentile=0.0`). Point `checkpoint_dir` at a
dedicated persistent directory on a local POSIX filesystem that implements
atomic same-directory rename, `fsync`, and `flock`. It must not contain the
feature directory or output path:

```bash
python -m steering.compute_statistics \
    feature_dir=out/data/features/... \
    n_sae_features=65536 \
    checkpoint_dir=/persistent/feature-statistics-checkpoint \
    checkpoint_interval_shards=50
```

The command seals the ordered shard family before the first body read. Header
preflight and full reads are descriptor-bound to the captured regular-file
identity (device, inode, size, mtime, and ctime). A checkpoint also binds the
dictionary width, train/validation split, detail mode, L0/BOS/SFT provenance,
resolved CPU/CUDA numeric runtime, schema version, and algorithm version. A
missing, reordered, replaced, or edited shard—or any semantic/runtime mismatch
on restart—fails loudly rather than mixing two runs.

Each generation is an immutable safetensors snapshot of every exact reducer
state. The writer fsyncs that file, then atomically publishes and fsyncs a
small manifest as the sole commit point. It retains the previous generation
until that manifest is durable, verifies the committed tensor digest and
cross-bound metadata on restore, and holds an exclusive directory lock through
the final statistics-file commit. The final snapshot is retained if output
serialization fails; after a successful atomic output write and parent-directory
fsync, checkpoint state and manifest are removed. Thus a restart replays at
most `checkpoint_interval_shards` completed shards if interruption lands during
the next commit (otherwise at most interval minus one between commits), and
never trusts a torn snapshot or partially published output.

Checkpointing does not enable deterministic CUDA algorithms. CUDA
`scatter_add_` moment reductions may therefore differ slightly between a fresh
run and an interrupted/resumed run, just as two fresh CUDA runs may differ.
The CUDA exact-count reduction still returns exact integer multiplicities and
does not expand this tolerance.
The regression contract permits `rtol=1e-6, atol=1e-6` only for the persisted
per-feature mean, mean-including-zero, and standard-deviation tensors and the
derived corpus magnitude summary. Shard counts, exact float16 magnitude
percentiles/histograms, all other tensors, and all other metadata remain
exact-equality requirements. Use `device=cpu` when exact moment reproducibility
is required.

Immediately before output publication, checkpoint mode lists the statistics
shard domain once more and rechecks every captured file identity. This is a
metadata-only final fence—no shard body is read or hashed—and catches an
addition, removal, or replacement after startup. It cannot make an
uncooperative producer and the statistics writer atomic; callers must still
avoid mutating the directory during publication.

This deliberately stores full snapshots—no delta log or mmap-backed mutable
state. At 65,536 features the count table alone is 15.4995 GiB. A commit needs
roughly one additional generation of free disk while the previous generation
remains valid, so budget a little over 31 GiB peak plus headers and small
reducers. With 1,399 processed shards and the default interval of 50, 28
snapshots write about 434 GiB cumulatively while bounding replay to 50 shards.
Increase the interval to trade more replay for less checkpoint I/O. Restore
also requires enough RAM for the complete state; it uses an owned pread load,
not an mmap that could later change underneath the computation.

Leaving `checkpoint_dir=null` executes the pre-resume path: no inventory
identity restrictions, lock, checkpoint reads, writes, or cleanup. Checkpoint
mode does not support a noncanonical implicit width because that width cannot
be part of the sealed contract before body reads; pass `n_sae_features`
explicitly in that case.

#### Identity-sealed HF streaming input

`remote.enabled=true` is an opt-in transport adapter for the same exact
streaming reducer. It requires `streaming=true`, `activation_percentile=0.0`,
no preloaded shards, an explicit positive `n_sae_features`, and a dedicated
`remote.staging_dir`. Local raw bodies under `feature_dir` are ignored. The
canonical, symlink-resolved path relative to `remote.local_features_root` maps
beneath `remote.remote_features_root` in the configured HF bucket.

Startup lists that prefix once and seals the complete numeric family as ordered
`(path, byte_length, xet_hash)` rows. It must be nonempty and contiguous from
zero; `remote.expected_shard_count=N` additionally requires exactly `0..N-1`.
A null expected count is allowed for bounded experiments but warns prominently.
Last-shard exclusion is applied only after complete-family validation and still
retains a sole shard. The excluded object remains in full-family identity, but
its header is outside the statistics metadata domain.

One transport worker downloads and size/Xet-verifies compressed bytes only.
Once N is ready, it schedules N+1. On eligible CUDA runs, the bounded local
preparation worker then stable-reads and zstd-decompresses N while the main
thread reduces the prior shard; the main thread alone parses safetensors and
creates tensors. At most current and one future compressed body can coexist on
disk, and at most one decompressed lookahead payload exists in RAM. The current
compressed body is removed immediately after successful decompression. This still
transfers every statistics-domain byte. It reduces persistent local storage and
may overlap transfer with computation; it does not reduce network traffic.

Before the worker exists, staging must satisfy:

```text
required = sum(two largest compressed statistics-domain bodies, or all if fewer)
           + 1 GiB
recoverable = current free bytes + recognized retained current/lookahead bytes
require recoverable >= required
```

This covers the runner-owned namespace, not separate HF/Xet client or OS caches;
budget and measure those independently. Staging is locked and carries a
versioned bucket/prefix owner marker. Only canonical raw bodies and proven-dead
download temporaries are cleaned. Unknown files, links, special files,
live/inaccessible-PID temporaries, and identity-replaced paths fail closed and
are preserved. The marker and lock remain for safe reuse.

Shutdown cancels an unstarted lookahead and always joins the sole worker. An HF
client call already in progress cannot be forcibly killed, so interruption may
wait for that call or the configured per-object retry timeout.

Remote checkpoints bind bucket, prefix, full/statistics inventories, excluded
identity, width, split/detail choices, and numeric runtime. Staging location and
retry timing may change on resume, as may null expected-count to the exact
observed count. L0, BOS, and canonical SFT expectations are established
incrementally and persisted with positive progress. A completed checkpoint
restores them without creating staging or downloading a body.

After transport joins and finalization completes, the command relists the full
prefix immediately before local publication and requires exact startup identity.
Append, removal, rename, replacement, or logical-index change leaves any
preexisting output untouched. This is a last-observation fence, not a
transactional snapshot; coordinate remote writers. Statistics mode never
uploads, overwrites, or deletes HF objects, and transport provenance does not
enter the statistics artifact.

The intended 1,400-shard PT L22/f65k production run is gated on a fresh listing
proving exactly 1,400 contiguous objects and operator review of inventory,
capacity, RAM, transfer, checkpoint, staging, and profile paths:

```bash
uv run python -m steering.compute_statistics \
  --config-name compute_statistics \
  feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  output_path=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/feature_statistics.safetensors \
  segment_size=128 n_sae_features=65536 streaming=true \
  activation_percentile=0.0 exclude_last_shard=true device=cuda \
  checkpoint_dir=/persistent/compute-statistics-checkpoint \
  profile=true profile_summary_output=/persistent/compute-statistics-profile.json \
  remote.enabled=true \
  remote.staging_dir=/dedicated/scratch/compute-statistics \
  remote.expected_shard_count=1400
```

Do not launch against an incomplete family; the exact count is a
fail-before-download production gate.

#### Performance baseline sidecar

Set `profile=true` to write per-shard load, validation, exact-count, moment,
segment-statistics, bookkeeping, and checkpoint timings plus run-level setup,
finalization, and publication timings. `profile_summary_output=null` defaults to
`out/traces/compute_statistics_profile.json`. Profiling is opt-in and does not
change the statistics artifact. See
[`profiling.md` § Feature Statistics](../performance/profiling.md#feature-statistics)
for the schema, phase boundaries, and comparison protocol.

## Percentile Overlay Precomputation

`python -m scripts.preprocess.precompute_percentiles` converts immutable base
feature shards into uint8 percentile overlays under
`pct/<feature-statistics-checksum[:16]>/`. Use `resume=true` to skip complete
shards, `force=true` to rebuild them, `device=cpu` for the CPU producer, and
`profile=true profile_summary_output=...` for the JSON sidecar described in
[`docs/performance/profiling.md`](../performance/profiling.md#percentile-precompute-issue-1269).

### Performance options (issue #1269)

The CPU path performs a direct-gather lookup for each active magnitude. It does
not gather an `[N, 99]` row tensor; per-chunk temporaries are bounded to roughly
70 MB at the default 1M-value chunk. This preserves the exact legacy
bisect-right lookup semantics and byte-identical overlay payloads.
The local CPU producer checks its freshly built table once for NaN and reuses
that evidence across chunks; any NaN retains the original queried-entry checks.
Borrowed-table helpers and remote/CUDA paths keep their checked defaults.

### Bounded HF streaming and immutable publication

`remote.enabled=true` selects a single-process remote producer for machines
that cannot retain the raw feature family locally. The logical `feature_dir`
still supplies `feature_statistics.safetensors`, but raw local shard bodies are
ignored. The resolved directory is mapped from `remote.local_features_root` to
`remote.remote_features_root`, and exactly
`remote.expected_shard_count` direct raw objects must form the contiguous
family `0..N-1` with positive sizes and lowercase Xet hashes. The startup
inventory is sealed as ordered `(path, byte_length, xet_hash)` rows and is
relisted unchanged before publication.

One source worker keeps at most one operation ahead. In CUDA mode it downloads,
authenticates, hashes, and zstd-decompresses the next source while the main
thread parses and computes the current shard; parsing, tensor creation, and all
PyTorch/CUDA work remain on the main thread. Completed receipts are authenticated
and hashed but are never decompressed. CPU mode retains synchronous main-thread
decompression. One upload worker publishes an output body and then its immutable
receipt while the next shard computes. Remote mode
still transfers **every raw input byte**: after an incomplete run, receipts
avoid recomputation and re-upload for completed shards, but every source body
is read again because the schema-v2 base hash is defined over the exact ordered
compressed bytes and is not composable from receipts.

Admission happens before a transport worker or remote write:

```text
two largest compressed source bodies
+ remote.max_output_shard_bytes
+ two feature-statistics file generations
+ 1 GiB reserve
```

The dedicated staging directory is locked and bound to the bucket, source and
output prefixes, and deterministic contract digest. Links, special files,
multi-link files, mismatched ownership, and unknown entries fail closed. The
reserve covers atomic-write and small control-artifact overhead; it does not
model OS page cache or Hugging Face/Xet client caches.

CUDA additionally requires an explicit
`remote.max_decompressed_shard_bytes`. Before a source worker or remote write,
the producer compares current host/cgroup headroom with this conservative
lookahead envelope:

```text
3 * remote.max_decompressed_shard_bytes
+ largest compressed source body
+ 1 GiB reserve
```

The three plain-body terms cover the current parsed storage tensors, current
output/parse transients, and exactly one future plain safetensors body. Each zstd
frame is checked against the configured bound before decompression when its
content size is declared, and the decompressor remains capped for frames whose
size is unknown. This execution-only limit is deliberately absent from the
immutable output contract, so a safe resume may raise it without changing bytes.
Shutdown cancels an unstarted source future and always joins the sole worker;
an in-progress backend transfer or zstd call must return before that join completes.

The remote family is published at
`<source-prefix>/pct/<percentile-checksum[:16]>/`. Its timestamp-free
`_remote_contract.json` binds the exact source inventory, percentile table,
segment size, zstd level, explicit CPU/CUDA producer, Torch/CUDA/GPU runtime,
and serialization versions. Each `_receipts/shard_NNNNN.json` binds one sealed
source to the uploaded payload's size, Xet hash, semantic tensor digest, and
value count. CPU and CUDA are distinct immutable producer contracts. Same-device
remote/local output must be exact; CPU and CUDA should not be compared for byte
identity because valid boundary cases can differ by one uint8 bin.

Every canonical object is also authenticated by exactly one uploader-owned
verified-upload claim whose target, byte length, and Xet hash must agree with
the current canonical object. Claim-only state is partial resumable state;
missing, stale, conflicting, duplicate, or malformed claims fail closed. This
prevents a replaced contract, receipt, payload, statistics file, or manifest
from being accepted by the zero-download complete-family path. Nested `tok/`
consumer trees are outside the direct percentile inventory and are ignored.

Completion order is contract, repeated payload then receipt pairs, an
overlay-scoped checksum-bearing `feature_statistics.safetensors`, and finally
the ordinary schema-v2 `manifest.json`. The manifest is the sole completion
marker in the artifact format and existing consumers need no remote-transport
awareness. The HF bucket API cannot atomically commit an output object while
also leasing a separate mutable source prefix: the runner therefore requires
an exact source relist immediately before the manifest upload and diagnoses a
second exact relist immediately afterward. If the source changes concurrently
with that upload, the command returns nonzero and says not to consume the
family, but the immutable manifest may already exist. Production must keep the
source family quiescent for this commit window; a stronger guarantee would
require an immutable/content-addressed source snapshot. Schema-v2 consumers
still reject such a stale family by recomputing its exact compressed-byte base
hash, so manifest presence alone never bypasses source identity validation.
The root statistics file is updated only after manifest commit; the existing
`save_percentile_checksum` no-op preserves bytes when current and retains any
tokenizer checksum or other metadata. A non-dry-run
`scripts.utils.hf_pull category=features` materializes each complete pulled
overlay automatically: it copies the scoped statistics file only if the root
file is absent; otherwise it verifies the table checksum and merges only the
percentile checksum. This pull hook requires schema v2, deterministic producer
bytes for the canonical run contract, receipts, and manifest, the contract's
exact manifest digest, exact local raw and overlay shard-name coverage, every
immutable receipt, the receipt-derived source inventory, each
receipt's exact source and payload size/Xet plus semantic digest, the scoped
statistics object's exact manifest identity, and the manifest's exact
compressed-byte base hash. It also requires exactly one verified-upload claim
for every contract, payload, receipt, statistics, and manifest object, with the
claim body, target, size, and Xet hash agreeing with the local canonical file.
A filtered pull or an old same-named local payload therefore cannot install a
root checksum for missing, stale, or corrupted payloads. Invalid manifests,
path/checksum mismatches, incomplete receipt or payload coverage, or scoped
statistics mismatches abort without mutating an existing root statistics file.
When checksum-addressed generations coexist, the hook selects only the family
whose full table checksum matches the recomputed root statistics; without root
statistics, multiple remote families are rejected before any copy. Ordinary
local percentile overlays have no remote markers and are ignored by this
pull-only materialization hook.

`validate_features` honors `manifest.metadata.producer`. CPU and legacy
overlays are recomputed exactly on CPU. A CUDA overlay is recomputed exactly
only on its matching Torch/CUDA/GPU runtime; otherwise validation uses an
explicit CPU fallback allowing at most one uint8 bin, the known legitimate
cross-device boundary difference. Artifact identity and all non-numerical
checks remain exact in either case.

Resume behavior is deliberately small and immutable:

| Remote state | Result |
|---|---|
| Empty | Publish the contract and process all shards |
| Contract only | `resume=true` continues |
| Payload + matching receipt | Hash the source bytes; skip decode, compute, and upload |
| Payload without receipt | Recompute, download and exact-compare the payload, then publish the receipt |
| Receipt without payload | Fail as corruption |
| Statistics without manifest | Validate/adopt it and finish publication |
| Complete manifest | Relist the source, authenticate every canonical object and claim, repair local statistics, and download no raw bodies |
| Contract/runtime/inventory mismatch | Fail as an immutable collision |

`resume=false` accepts only an empty or valid complete current family; use
`resume=true` for partial state. `force=true`, `device=auto`, and any rank shape
other than `WORLD_SIZE=1 RANK=0 LOCAL_RANK=0` are rejected. Remote `dry_run=true`
performs configuration, inventory, output-name, device, and capacity preflight
without downloading control objects or source bodies, staging creation,
statistics rewrites, or remote writes.

The intended production assertion is explicit rather than inferred from an
older listing. Run the dry-run first and review the printed full digest, bytes,
device/runtime, output state, and admission requirement; a production launch
requires separate authorization:

```bash
# Metadata-only preflight
uv run python -m scripts.preprocess.precompute_percentiles \
  --config-name=precompute_percentiles \
  feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  segment_size=128 device=cuda zstd_level=1 resume=true force=false \
  profile=true profile_summary_output=/persistent/precompute-percentiles-profile.json \
  dry_run=true \
  remote.enabled=true \
  remote.bucket_id=henryac/nano-gpt-pretrain-steer \
  remote.local_features_root=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  remote.remote_features_root=l22_ci_nobos_raw \
  remote.expected_shard_count=1400 \
  remote.max_output_shard_bytes=1000000000 \
  remote.max_decompressed_shard_bytes=4000000000 \
  remote.staging_dir=<persistent-local-staging>

# Separately authorized production launch: only dry_run changes
uv run python -m scripts.preprocess.precompute_percentiles \
  --config-name=precompute_percentiles \
  feature_dir=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  segment_size=128 device=cuda zstd_level=1 resume=true force=false \
  profile=true profile_summary_output=/persistent/precompute-percentiles-profile.json \
  dry_run=false \
  remote.enabled=true \
  remote.bucket_id=henryac/nano-gpt-pretrain-steer \
  remote.local_features_root=out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos \
  remote.remote_features_root=l22_ci_nobos_raw \
  remote.expected_shard_count=1400 \
  remote.max_output_shard_bytes=1000000000 \
  remote.max_decompressed_shard_bytes=4000000000 \
  remote.staging_dir=<persistent-local-staging>
```

Run from the primary repository root, or make `feature_dir` and
`remote.local_features_root` absolute paths resolving to the same PT logical
directory. This is especially important from a worktree whose own `out/` tree
does not contain the 82,241,744-byte statistics artifact. The global
`zstd_level` default remains 12; level 1 above is an explicit production choice
and is part of the immutable remote producer contract even though the output
prefix is checksum-derived. The dry run reports names from the output listing
but cannot inspect control-object contracts because it downloads none. Require
an empty reported output prefix for this fresh level-1 family. If it is nonempty,
stop for explicit contract inspection; never mix a partial level-12 family with
level 1. Lowering `remote.expected_shard_count` against the 1,400-shard prefix
fails exact contiguous-family validation rather than selecting a canary. A real
canary requires a separately published source prefix.

### Profiling

`profile=true` records per-shard load, direct-gather compute, and save phase
timings in the percentile sidecar (peak RSS is reported by
`scripts/benchmark/benchmark_percentiles.py`). The schema is maintained in the
[`Percentile Precompute`](../performance/profiling.md#percentile-precompute-issue-1269)
profiling section.

### Context-window contract

`context_window` is the maximum number of Gemma tokens retained from each
source document before segmentation; the project default is **1024**. It is
recorded in every feature-shard metadata record and must remain identical
through splicing and all downstream overlays.

Context-independent extraction does not mean that Gemma receives the whole
window on every forward. With the default `segment_size: 128`, each forward
contains one 128-token segment (plus an injected BOS when that mode requires
one). Context-dependent extraction can forward a document chunk up to
`context_window`. Dataset splicing never runs Gemma: it re-tokenizes the
source documents with the same `context_window` and `segment_size`, then
recombines already-computed segment features. Its strict metadata check is
what prevents a 1024/2048 splice mismatch.

### Gemma-3 OOM Root-Cause Fix

Extracting from gemma-3-4b-it on a 24 GB GPU would OOM when the fp32
TransformerLens (TL) companion model (~16 GB) was still resident on GPU
while loading the HF Gemma-3 copy. This is fixed at the root cause:
`_build_nnterp_gemma3` in `extract_features.py` now offloads the fp32 TL
model to CPU (`tl_model.to("cpu")` + `torch.cuda.empty_cache()`) **before**
loading the HF Gemma-3 copy — exactly mirroring the existing `_build_nnterp_gemma2`
pattern.

Key points:

- The TL model **always loads in fp32** (TransformerLens default). There is
  no code path that loads TL in bfloat16.
- `model_dtype` (default `bfloat16`) governs only the nnterp/native HF build
  dtype and the `sae.to(model_dtype)` cast — it does **not** affect the TL
  load.
- After the offload the fp32 TL model (~16 GB for a 4B model) resides in host
  RAM. A machine with ~16 GB available host RAM is therefore a prerequisite for
  gemma-3 extraction on a 24 GB GPU.
- **VLM loading (4b/12b/27b).** The 4b/12b/27b checkpoints are
  `Gemma3ForConditionalGeneration` VLM repos whose text weights live under
  `language_model.model.*`; a plain `Gemma3ForCausalLM.from_pretrained` would
  silently random-init them (issue #549). Both the nnterp and native backends
  route through `steering.gemma3_loading.load_gemma3_text_decoder`, which
  remaps those weights and **fails loud** (`ModelWeightLoadError`) on any
  unmaterialised parameter. See
  [`gemma-scope-2.md` § VLM checkpoints](gemma-scope-2.md#vlm-checkpoints-4b12b27b--text-decoder-loading).

## Configuration Reference

### Feature Extraction (`configs/extraction/extract_features.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | gemma-2-2b | TransformerLens model name |
| `activation_type` | residual | Activation site: residual, mlp, attention |
| `layer` | 8 | Transformer layer index |
| `context_window` | 1024 | Source-document context window before segmentation |
| `pooling` | max_pool | How to aggregate activations per segment |
| `segment_size` | 128 | Tokens per segment |
| `segment_mode` | fixed_tokens | `fixed_tokens` or SFT-only `turn_delimited_fixed_tokens` |
| `sae.release` | gemmascope | SAE release family |
| `sae.num_features` | 16k | SAE width: 16k/32k/65k/131k/262k/524k/1m |
| `sae.l0` | null | v1 (`gemmascope`): int average-L0 or `null` (canonical). v2 (`gemmascope_2`): tier string — `small`, `medium`, `medium_seed_1`, or `big` (no null/canonical). The public `sae_resolver.coerce_config_l0` helper coerces the YAML string to `L0Tier` before `resolve_sae`; an out-of-contract value (e.g. a float, a v1 string l0) raises `ValueError`. |
| `num_shards` | 1 | Exclusive upper bound of the shard window; the producer processes `[start_shard, num_shards)` |
| `start_shard` | 0 | First shard index (0-based, inclusive) |
| `gpu_batch_size` | 16 | GPU forward pass batch size; set to `auto` to run the power-of-2 auto-tuner |
| `cpu_batch_size` | 64 | Outer pipeline chunk size; set to `auto` to run the power-of-2 auto-tuner |
| `min_gpu_batch_size` | 1 | Positive lower bound for auto-tuning GPU batch search; must be `<= max_gpu_batch_size` |
| `min_cpu_batch_size` | 1 | Positive lower bound for auto-tuning CPU batch search; must be `<= max_cpu_batch_size` |
| `max_gpu_batch_size` | 512 | Positive upper bound for auto-tuning GPU batch search |
| `max_cpu_batch_size` | 512 | Positive upper bound for auto-tuning CPU batch search |
| `backend` | auto | auto (prefer nnterp), nnterp, transformerlens, or native (home-grown raw-HF, structurally-truncated residual-only fast path; Gemma only, CUDA) |
| `model_dtype` | bfloat16 | Governs the nnterp/native HF model build dtype and `sae.to(model_dtype)` — does **not** affect the TransformerLens model load (TL always loads in fp32). bfloat16 is ~6x faster with <5% magnitude error. |
| `gc_mem_threshold` | 0.9 | GC trigger threshold for VRAM |
| `dataset_phase` | null | Data split: null/base; train SFT standard_sft, safety_sft, capabilities_sft; or val SFT standard_sft_val, safety_sft_val, capabilities_sft_val |
| `data_dir` | null | Parquet input dir (null = nanochat default) |
| `output_dir` | null | Output dir (null = auto-constructed from params) |
| `feature_root` | out/data/features | Root for auto-constructed paths |

All extraction scalars cross one typed runtime-config boundary before model or
artifact work begins. Reusable `phantom-types` domains plus `beartype` enforce
strict integer, positivity, finiteness, unit-interval, non-blank model-name, and
zstd-level requirements. Unary auto-batch bounds use the same typed contract;
only relational search constraints remain as explicit validation branches.

### Hardware-Specific Configs

| Config | Hardware | gpu_batch | cpu_batch |
|--------|----------|-----------|-----------|
| `extract_features.yaml` | Default | 16 | 64 |
| `extract_features_4xh200.yaml` | 4xH200 | 96 | 192 |
| `extract_features_2xb200.yaml` | 2xB200 | 128 | 256 |
| `extract_features_4xb200.yaml` | 4xB200 | 128 | 256 |
| `extract_features_8xpro6000.yaml` | 8xRTX PRO 6000 | 64 | 128 |

## SAE Configuration

### GemmaScope Models

| Model | Sizes Available | Activation Types |
|-------|----------------|-----------------|
| gemma-2-2b | 2B params | residual, mlp, attention |
| gemma-2-9b | 9B params | residual, mlp, attention |
| gemma-2-27b | 27B params | residual, mlp, attention |

### Valid Width/Layer Combinations (gemma-2-2b)

| Activation | Width | Layers |
|-----------|-------|--------|
| residual/mlp/attention | 16k | 0-25 (all 26) |
| residual/mlp/attention | 65k | 0-25 (all 26) |
| residual | 32k/131k/262k/524k/1m | Layer 12 only |

### Pooling Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| `last_token` | Final token's activations only | Fast, limited context |
| `max_pool` | Max across segment tokens | Captures strongest signals (recommended) |
| `mean_pool` | Mean across segment tokens | Balanced representation |

## Turn-delimited segmentation (SFT)

Set `segment_mode=turn_delimited_fixed_tokens` for SFT extraction that must not
place tokens from different conversation turns in the same segment. This mode
requires a non-base SFT `dataset_phase` and the source parquet's
`turn_char_starts` column, produced by the current
`scripts/preprocess/convert_sft_to_parquet.py` converter.

Turn 0 absorbs the tokenizer BOS. For each later turn, extraction starts the
turn at the first Gemma token whose character start is at or after the recorded
boundary; if a token straddles the boundary, it remains in the preceding turn.
Each turn is then split into chunks of at most `segment_size` tokens.

Extraction checks that Gemma decoding matches the source text after removing
the decoded BOS prefix. A mismatch is tolerated only in the final 20 decoded
characters and only when it occurs after every retained turn boundary; tolerated
tail drift is silent. The loud round-trip failure is the only operator signal,
and it fires on every path. One width-preserving normalization is applied to
both sides before comparing: sentencepiece decodes its word-boundary marker
`▁` (U+2581) as a space, so a source text that literally contains that
character (SmolTalk rows discussing BPE do) is compared with the marker mapped
to a space — offsets are unaffected because both are one character wide. No
other character is normalized. Statistics
derive segment lengths from stored positions for both modes.

The fixed-token max/mean pooling path retains its existing reshape reduction.
For a turn-delimited document with two or more segments, extraction builds one
`[n_segments, 2]` offset/length tensor and uses one contiguous segmented
reduction (`max` or `sum`), with exact integer prefix-sum counts for masked
means. Max pooling is exactly equal to the former per-segment loop on the
supported non-negative SAE-activation domain, including the existing zero-fill
behavior for masked BOS rows and the legacy sign bit when a maximum is tied
between `+0.0` and `-0.0` — `segment_reduce` scans each segment in order, so
like `max(dim=0)` it keeps the first tie, and the signed-zero tests pin that
rather than repairing it at runtime.
Mean pooling accumulates float16 and bfloat16 inputs in float32, then restores
the former masked or unmasked rounding point; parity tests require raw-bit
equality for both low-precision dtypes on CPU and bound them to one
output-format ULP on CUDA. Float32 addition order can move results in the low
bits, so its parity tolerance is `(rtol, atol) = (1e-5, 1e-6)`; that bound is
calibrated for segment lengths up to `segment_size`, which is what
`segment_turn_delimited` emits, and the relative error grows with segment
length.
Single-segment calls and arbitrary overlapping, gapped, or reordered spans
retain the direct per-segment reduction, and `last_token` retains its indexed
gather. Context-independent extraction also remains on its existing
single-segment pooling path.

Validation independently re-derives the turn boundaries and segment geometry
from the source parquet.

## Context-Independent Features

By default, each document is passed through the SAE model as a single causal sequence; every token inside segment N attends to all preceding segments 0..N-1 of the same document.

Set `context_independent_features: true` to run each segment through the model as its own independent sequence. Each token's features depend only on tokens within that segment. BOS is automatically prepended to segments that don't already start with BOS (i.e., every SFT segment and every pretraining segment except seg 0); the injected BOS is sliced off before pooling so the stored features describe exactly the real segment tokens.

Enable via CLI:

```bash
python -m steering.extract_features \
    context_independent_features=true \
    num_shards=10
```

Output directory auto-gains a trailing `/context_independent` leaf:

```
out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/context_independent/
```

This guarantees context-dependent and context-independent extractions never collide even with shared config. When `output_dir` is explicit, the leaf is not auto-added — the user is presumed to have encoded intent.

**Semantics vs default:**

| Aspect | default (context-dependent) | context_independent_features |
|--------|--------------|-----------------------------------|
| Forward pass | one per document | one per segment |
| Segment 0 features (pretraining) | identical under causal attention | identical under causal attention |
| Segments 1+ features | see preceding context | see only segment tokens |
| Segment metadata (positions, char boundaries) | identical | identical |
| Output path leaf | — | `/context_independent` |
| Shard metadata | `context_independent_features=false` | `context_independent_features=true` |

Downstream consumers (statistics computation, token boundaries, packed rows, dataloaders) are unchanged.

## Ignore BOS Features

The extraction producer **always** excludes every token whose token ID equals the tokenizer's BOS ID from per-segment SAE pooling. BOS tokens tend to act as **attention sinks** — they accumulate large residual-stream activations that skew per-segment pooled features without encoding meaningful content. Excluding them produces cleaner feature signals, especially for pretraining data where every document's first segment contains a BOS token at position 0.

> **Not configurable.** This was formerly the opt-in `ignore_bos_features` option; it is now frozen **always-on** in the producer and is no longer a config key. A value still passed via config or CLI (`+ignore_bos_features=...`) is logged as an error (`logger.error`) and ignored — extraction proceeds always-on. Extraction therefore requires a tokenizer with a discoverable BOS id; all supported models (gemma-2, gemma-3, nanochat) define one, and a tokenizer without one raises a hard error.

**Keep-mask mechanism.** For each segment the pipeline constructs a boolean mask over the kept token positions:

```
kept_ids   = [   2,  9259, 1902 ]   #  2 == <bos>
keep_mask  = [ False, True,  True ]  #  (kept_ids != 2)
```

Pooling then operates only over positions where `keep_mask` is `True`.

**Per-pooling-mode behavior:**

| Pooling mode | Effect (BOS exclusion, always-on) |
|---|---|
| `max_pool` | Max taken over kept (non-BOS) rows only |
| `mean_pool` | `sum(kept) / count(kept)` — BOS in neither numerator nor denominator |
| `last_token` | Activation of the last non-BOS token |

**All-`<bos>` segment (hard error).** If every token in a segment is a BOS token — for example when `segment_size=1` on pretraining data, or in a degenerate single-`<bos>` document — no positions survive the mask and pooling is undefined. The pipeline raises `BosOnlySegmentError` (a typed `ExtractionError` subclass defined in `errors.py`). This error propagates out of `process_shard` and aborts the extraction run immediately (fail-fast); because a shard is written atomically only after all of its documents are pooled, an aborted run leaves **no partial shard** (resume re-processes the shard). `segment_size=1` is never used in production — all extraction configs use 128 — so in practice this fires only on a genuinely degenerate all-`<bos>` document. These zero-token / zero-segment rejections guarantee full feature coverage, which the FeatureIndex fail-fast contract (see [encoder-decoder-training.md](encoder-decoder-training.md#featureindex-fail-fast-contract)) relies on.

**Segmentation-metadata invariance.** BOS exclusion changes the pooled feature content of `<bos>`-containing segments, so `feature_ids`, `feature_mags`, and `offsets` (the CSR row-pointer = cumsum of per-segment non-zero-feature counts) all legitimately differ from a *legacy* (pre-freeze, flag-off) run. The arrays that remain bit-identical to a legacy run are the tokenization-derived segmentation metadata that downstream token-boundary and packed-row overlays consume: `n_segments`, `n_docs`, `doc_ids`, `segment_indices`, `segment_positions`, `segment_char_starts`, `segment_char_ends`, `doc_token_counts`. Overlays key off that segmentation metadata — not off `offsets` or feature content — so the always-on behavior requires no changes to downstream overlay construction.

**SFT data follows the full-document rule.** Full-sequence SFT extraction starts at the beginning of the rendered text, so the first segment may include a tokenizer BOS. BOS exclusion therefore changes the first SFT segment's pooled features, just as it does for pretraining data — it is **not** a no-op for SFT (only segments that contain no BOS are unaffected).

**Legacy artifacts remain readable.** Artifacts extracted *before* the freeze (flag off, no `/no_bos` leaf, `ignore_bos_features=false`/absent in metadata) keep their meaning: every consumer honors the recorded value, and `validate_features` re-extracts each such shard with its recorded `false` flag. The compute layer's `ignore_bos_features=false` path is retained unchanged and stays byte-identical to legacy behavior — only the *producer's* ability to select it is removed.

**Composition with `context_independent_features`.** The two options fully compose. In context-independent mode the injected per-segment BOS is already sliced off before pooling; with `ignore_bos_features` additionally enabled, the natural seg-0 BOS of pretraining documents is also dropped.

**Output path.** The auto-constructed feature directory gains a trailing `/no_bos` leaf, ordered **after** `/context_independent`. Because BOS exclusion is always-on, every new run uses one of the `ignore_bos_features=true` paths; the `ignore_bos_features=false` paths describe only **legacy** artifacts extracted before the freeze:

| `context_independent_features` | `ignore_bos_features` | `segment_mode` | Path tail |
|---|---|---|---|
| false | false | fixed | `…/{phase}` (legacy only) |
| true | false | fixed | `…/{phase}/context_independent` (legacy only) |
| false | true | fixed | `…/{phase}/no_bos` |
| true | true | fixed | `…/{phase}/context_independent/no_bos` |
| false | false | turn-delimited | `…/{phase}/turn_delimited` (legacy BOS only) |
| true | false | turn-delimited | `…/{phase}/context_independent/turn_delimited` (legacy BOS only) |
| false | true | turn-delimited | `…/{phase}/no_bos/turn_delimited` |
| true | true | turn-delimited | `…/{phase}/context_independent/no_bos/turn_delimited` |

When `output_dir` is explicit, no leaf is auto-appended — the user is presumed to have encoded intent.

**Shard metadata.** Each shard records `ignore_bos_features=true|false`. New runs always record `true`; legacy shards lacking the key load as `false`. `verify_feature_dir` cross-checks the shard metadata (and, when present, the stats-file value) against the path leaf, and `assert_uniform_ignore_bos` rejects a directory that mixes `true` and `false` shards — so a new always-on run cannot be written into a legacy `false` directory.

**Statistics file.** The stats file carries a plain `bool` `ignore_bos_features` metadata field: always written as `"true"` or `"false"` in freshly computed statistics (the uniform value across the directory's shards — `"true"` for any new always-on run, `"false"` only when recomputing stats over a legacy pre-freeze directory), like every other scalar metadata field. A stats file written before this field existed simply lacks the key and loads as `False` via `metadata.get("ignore_bos_features", "false")` — a pre-feature corpus was extracted with ignore_bos off, so `False` is the correct default, not merely convenient. A present-but-non-`true`/`false` value raises `ValueError` (loud). There is no nullable/`None`/"unknown" state and no "written only when not None" behavior. This field does NOT mirror the `l0_tier` nullable model; `l0_tier` stays nullable, `ignore_bos_features` is a plain bool. No stats schema-version bump.

BOS exclusion is automatic — no flag is needed, and none can be set. Passing `+ignore_bos_features=...` only logs an error and is ignored; a bare `ignore_bos_features=...` override is rejected by Hydra (the key is gone from the config schema).

## Head Truncation

Documents longer than `context_window` are head-truncated to align with how the training packing pipeline truncates (`doc[:remaining]`). The nanochat tokenizer determines the character cutoff: it tokenizes the text, decodes the first `context_window` tokens to find the character boundary, then SAE tokens whose content starts within that boundary are kept. This ensures extraction and training see the same prefix of each document.

A hard cap at the SAE model's native context length (`model.cfg.n_ctx`) is applied after the nanochat-based cutoff.

Extraction **refuses to start** when `context_window` exceeds the SAE max context (`resolve_sae_max_context`, read weight-free from the HF config at startup): otherwise every document would be silently truncated to the cap while shard metadata recorded the larger `context_window`, poisoning downstream consumers (token-boundary full-coverage classification, the feature-shard splice). Set `context_window <= sae_max_context`.

**SFT phases need a larger window than the trainer's `max_seq_len`.** The conditioned SFT trainer keeps `max_seq_len` *nanochat* tokens of each conversation and its coverage gate requires the shard to reach that prefix (20-char tolerance), while extraction truncates to `context_window` tokens of the *feature model*. Gemma's tokenizer can need more tokens than nanochat's for the same text (digits are split, long multi-turn rows), so `context_window == max_seq_len` leaves long rows uncovered and the trainer stops with `feature coverage ends N chars before the rendered conversation`. Use `context_window >= 2 x max_seq_len` (production: 2048 for `max_seq_len=1024`, 4096 for a 2048-token base); extraction logs a WARNING for SFT phases below 2048 (#1634).

## Character Offset Decoding (issue #1269)

`token_to_char_offsets()` uses merge-aware progressive prefix decoding whenever
context-dependent token decoding can change the decoded length. The progressive
helper walks back the preceding byte-fallback group after a multi-byte merge; it
does not use the superseded `max(len(prefix), previous_offset)` clamp. Long inputs
first try the O(n) per-token fast path, then the chunked check, and finally the
full progressive path on the first divergent chunk. The Old Italic regression on
`shard_00355/doc_id=15373` is the motivating post-mortem: a 32-token zero-width
run proved that the old chunk-local correction was wrong.

For the nanochat tokenizer, whose `.enc` is a `tiktoken.Encoding`, the same three
paths consume exact per-token byte payloads and one incremental UTF-8
`errors="replace"` decoder. This removes repeated prefix decodes while preserving
the character-offset contract; differential coverage is pinned by
`tests/steering/test_token_offsets_tiktoken_bytes.py`.

## Output Format

### Feature Directory Structure

Auto-constructed path: `{feature_root}/s{seg_size}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}/`

Optional trailing leaves (ordered): `/context_independent` (opt-in), `/no_bos`,
and `/turn_delimited` (when `segment_mode=turn_delimited_fixed_tokens`). BOS
exclusion is always-on, so **every new run carries `/no_bos`**; the leaf is
absent only on legacy pre-freeze artifacts.

Example (legacy pre-freeze base run — no optional leaves; new runs add `/no_bos`):
```
out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/
├── shard_00000_features.safetensors
├── shard_00001_features.safetensors
├── ...
├── shard_00369_features.safetensors
└── feature_statistics.safetensors
```

Derived overlays live in content-addressed subdirectories alongside the
shards: `pct/<hash16>/` (percentiles), `tok/<hash16>/` (token boundaries +
packed rows), and `grad/<hash16>/` (gradient-attribution scores — see
[gradient-attribution-selection.md](gradient-attribution-selection.md)).

Example with both leaves (a context-independent run; `/no_bos` is always present):
```
out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/context_independent/no_bos/
```

### Shard Format (CSR Sparse)

Each shard is a safetensors file with:

| Field | Dtype | Shape | Description |
|-------|-------|-------|-------------|
| `offsets` | int64 (or int32) | (n_segments + 1,) | CSR row pointers |
| `feature_ids` | int32 (or uint16) | (n_total_features,) | Feature indices (ascending-by-id per segment, post-load) |
| `feature_mags` | float16 | (n_total_features,) | Magnitudes (co-sorted with feature_ids); upcast to float32 on read. Stored values must be `< FP16_SAFE_MAX (60000.0)`; `save_features` raises on saturation. |
| `doc_ids` | int32 | (n_segments,) | Document index per segment |
| `segment_indices` | int32 | (n_segments,) | Segment index within document |
| `segment_positions` | int32 | (n_segments,) | Token position of segment start |
| `doc_token_counts` | int32 | (n_docs,) | Tokens per document |
| `segment_char_starts` | int32 | (n_segments,) | Character start boundaries |
| `segment_char_ends` | int32 | (n_segments,) | Character end boundaries |

Dtype optimization: `feature_ids` stored as uint16 when max <= 65535 (else int32), `feature_mags` always stored as float16, `offsets` as int32 when fits (else int64). Auto-upcast on read (uint16 → int32, float16 → float32, int32 → int64).

**Non-empty base-segment invariant.** A base extracted shard with `n_segments > 0` must have **no empty segment**: `offsets[i+1] > offsets[i]` for every `i` (strict). A base-extracted segment that fired zero SAE features is treated as corruption. This is enforced fail-fast at extraction (`EmptyExtractionSegmentError`), in `save_features`, in the shard validators (`validate_shard` / `validate_shard_data` / `assert_csr_invariants`), at `FeatureIndex._load_shard` load time (`CorruptFeatureShardError`), and in the base-reading producers (statistics, percentiles, token boundaries) — issue #522. A whole-shard `n_segments==0` is exempt. The check applies **only** to base extracted shards; downstream selected/derived data (selection shards, packed rows, overlays, clustering, delta, masking, percentile-filtered stats) MAY legitimately be empty and is intentionally exempt. This is related to but distinct from the all-`<bos>` `BosOnlySegmentError` guard (see [Ignore BOS Features](#ignore-bos-features)): that one aborts when *every token* in a segment is `<bos>` (always-on BOS exclusion), whereas this invariant rejects a base segment that ends up with *zero stored features* for any reason.

**Shard metadata** (stored in the safetensors metadata dict, not as tensor fields):

| Key | Values | Notes |
|-----|--------|-------|
| `context_independent_features` | `"true"` / `"false"` | |
| `ignore_bos_features` | `"true"` / `"false"` | Absent in legacy shards — loads as `false` |
| `segment_mode` | `"fixed_tokens"` / `"turn_delimited_fixed_tokens"` | Absent in legacy shards — loads as `fixed_tokens` |
| `l0_tier` | tier string or absent | v2 shards only; absent → `None` for v1 |
| `extraction_backend` | `"nnterp"` / `"native"` / `"transformerlens"` | Actual extraction implementation; absent means unknown legacy provenance. Never `auto`. |

Every new extraction shard records its actual backend, including background saves.
Successful extraction with `backend=auto` records `nnterp`. Normal feature loads
and derived splice shards preserve this field. It does not change tensor payloads,
directory layouts, selection hashes, or statistics schemas.

**Legacy resume remains supported.** The extraction loop skips completed shards
without rewriting them and stamps the backend on newly extracted shards. Before
processing pending shards it checks all existing shard headers in the output
directory, including shards outside the requested range. Unknown legacy provenance
produces a warning with the unknown/total shard count; a conflicting known backend
or malformed value stops the run. Use a separate `output_dir` for a different known
backend. A fully completed run has no new shards to write and remains a no-op.
This preflight is not a lock between independent jobs: concurrent writers to the
same directory must use the same backend.

### Accessing Features

```python
from safetensors.torch import load_file

data = load_file("shard_00000_features.safetensors")
# Get features for segment i
start, end = data["offsets"][i], data["offsets"][i + 1]
seg_ids = data["feature_ids"][start:end]
seg_mags = data["feature_mags"][start:end]
```

## Validation

Validate extracted feature shards for structural correctness, source-data correspondence, and (optionally) activation reproduction.

```bash
# CPU-only structural validation
uv run python -m steering.validate_features feature_dir=out/data/features/s128/l8/...

# With source parquet cross-referencing
uv run python -m steering.validate_features \
    feature_dir=out/data/features/s128/l8/... \
    parquet_dir=data/fineweb-edu/...

# With GPU activation spot-checks
uv run python -m steering.validate_features \
    feature_dir=out/data/features/s128/l8/... \
    parquet_dir=data/fineweb-edu/... \
    model=gemma-2-2b sae_release=gemmascope sae_id=... \
    n_spot_check=10
```

### Validation Layers

| Layer | Function | Requires |
|-------|----------|----------|
| Shard segments | `validate_shard_segments()` — segment tiling, char boundaries, doc grouping | CPU only |
| Source cross-ref | `validate_shard_against_source()` — re-tokenize, verify token counts and char alignment | CPU + parquet |
| Directory | `validate_feature_directory()` — cross-shard metadata consistency, shard/parquet correspondence, statistics aggregates | CPU only |
| Activation | `validate_shard_activations()` — re-run LLM+SAE, compare feature IDs and magnitudes | GPU + model + SAE |

### Validation Config (`configs/extraction/validate_features.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dir` | **required** | Path to feature directory |
| `parquet_dir` | null | Source parquet directory for cross-referencing |
| `resolve_parquet_default` | false | When `true` **and** `parquet_dir` is null, resolves `parquet_dir` to the nanochat default parquet dir (same dir that extraction uses) to enable full source validation without specifying the path explicitly. Default `false` keeps all existing callers byte-identical. |
| `model` | null | Model name (for tokenizer + GPU validation) |
| `n_spot_check` | 5 | 0=CPU-only, 1=first doc, 2=first+last, 3+=length-biased sampling |
| `sae_release` | null | SAE release string (required for GPU spot-checks) |
| `sae_id` | null | SAE id string (required for GPU spot-checks) |
| `backend` | null | Select the shard's recorded `extraction_backend`. All-legacy directories fall back to `auto` with a warning; mixed known/unknown directories require an explicit backend for spot-checks. Explicit `auto`, `nnterp`, or `transformerlens` overrides provenance with a prominent mismatch warning. `auto` means nnterp; both require CUDA. Native replay remains unsupported. Ignored when `n_spot_check=0`. |

**Backend provenance validation (#1637).** Conflicting known backend values or
malformed metadata fail before model loading for GPU spot-checks and also fail
CPU directory validation. Missing metadata is unknown: all-legacy directories and
directories with one consistent known backend plus unknown shards remain valid,
with an incomplete-provenance warning. They cannot be certified backend-uniform.
GPU checks on mixed known/unknown directories require an explicit `backend`; they
never infer the old shards' provenance from the new ones.

For fresh shards, omit `backend` (or set `backend=null`) to replay the recorded
implementation. For intentional cross-backend comparisons, pass, for example,
`backend=transformerlens`; the banner and final summary report recorded,
configured, and effective backend, plus unknown shard counts. A mismatch remains
visible even if the numerical checks pass. Unknown shards' backend matches remain
unverified. Existing Jaccard/count/magnitude thresholds are unchanged; a matching
backend does not prove matching dtype, dependency versions, or batching. Continue
to set `model_dtype` to the extraction dtype.

Native provenance is recorded and accepted by CPU checks, but metadata-selected
native spot-checks fail early because the native collector is layer-bound. Use
`n_spot_check=0` for CPU validation or explicitly request `backend=nnterp` for a
warned cross-backend comparison. There is no automatic backend fallback on build
or forward failure.

### Validation Improvements

- **`l0_tier` cross-shard consistency** — `validate_feature_directory()`'s
  `CONSISTENCY_KEYS` now includes `l0_tier`. A directory whose shards carry
  mixed tiers (v1↔v2 or big↔medium) fails fast. v1 shards are all-None → still
  consistent with each other.
- **Weight-free model branch** — when `model` is set but `n_spot_check=0`,
  `validate_features` loads only a weight-free tokenizer via `resolve_tokenizer`
  plus `resolve_sae_max_context` for the model's max context (HF config loading
  is encapsulated in `steering.models`). A full TL model load is used only for
  models not registered in `EXTRACTION_MODELS` (e.g. gpt2) — it is not needed
  for normal Gemma extraction validation.
- **Non-finite magnitude rejection** — shard magnitude validation now rejects
  `+inf` and `NaN` values in addition to values exceeding `FP16_SAFE_MAX`.

## Statistics Output

`feature_statistics.safetensors` contains:
- **Metadata:** pooling, segment_mode, layer, n_sae_features
- **Corpus stats:** n_docs, n_tokens, mean/median tokens per doc
- **Feature stats:** n_segments, n_features_active, percentiles, histograms
- **Per-feature stats:** count, pct_segments_active, mean/std/min/max/median magnitudes, percentiles
- **Storage stats:** total bytes, bytes per doc/token
- **Extrapolation:** estimated resources for 10B tokens

**Nullable metadata fields** — `l0_tier` is nullable: it carries a concrete value for freshly computed stats and `None` for pre-existing stats files written before the field was introduced (absent key → loads as `None`).

| Field | Fresh stats value | Pre-existing stats value | Type |
|-------|------------------|--------------------------|------|
| `l0_tier` | L0Tier string (v2) or `None` (v1) | `None` | nullable |
| `ignore_bos_features` | `"true"` or `"false"` (uniform across the directory's shards) | `False` (absent key → loads as `False`) | plain bool, always written |

`ignore_bos_features` is a plain bool — always written, never nullable. A pre-existing stats file lacking the key loads as `False`. No stats schema-version bump accompanies the `ignore_bos_features` addition.

## Multi-GPU Distribution

When using torchrun, each rank:
1. Gets assigned `cuda:LOCAL_RANK`
2. Processes shards in a striped pattern (rank 0 gets shards 0, 4, 8, ...; rank 1 gets 1, 5, 9, ...)
3. Writes to the same output directory (shard filenames encode shard index)
4. No inter-GPU communication needed

## Feature Blacklist/Whitelist

The feature blacklist/whitelist (`feature_mask`) has **no effect on extraction**. Base extraction shards store ALL non-zero SAE features unconditionally — they are universal and are never recomputed when a mask is added, changed, or removed (invariant I3). This design keeps the expensive extraction artifacts reusable across any mask configuration.

The mask is configured in `configs/training/conditional_encdec_train.yaml` (or `conditional_encdec_sft.yaml`) and enforced downstream. See [encoder-decoder-training.md](encoder-decoder-training.md) for the full `feature_mask` config schema and enforcement details.

> **The mask is no longer purely a selection-layer concern (issue #695).** Downstream of extraction it now has a *second*, structural role: it sizes the **encoder's** feature-id-indexed embedding tables. That makes the encoder's feature dictionary a **separate axis** from the SAE's — the shards keep one entry per SAE feature (`n_sae_features`, and packed rows still rely on `uint16` ids), while the encoder keeps one embedding row per *allowed* feature (`V`). Extraction, statistics, percentile overlays, token boundaries and packed rows are all untouched by this and remain mask-agnostic (I3).

### Activation-Frequency Blacklist (optional, derived from statistics)

An optional **derived** artifact auto-produces a `feature_mask` blacklist from the **statistics** step's output. It reads `feature_statistics.safetensors` and blacklists every SAE feature whose per-segment activation frequency falls outside the configured kept interval: **KEEP iff `feature_mask_freq_min < freq <= feature_mask_freq_max`**. It does **not** read shards or recompute any extraction artifact (I3); it only reads the statistics file and writes a new sibling under `blacklist/`. Because it reuses the statistics aggregation, **its train/val split is whatever `compute_statistics` used** — the validation shard is excluded by default (`exclude_last_shard: true`). Run statistics computation (Phase 2) before generating it.

- Artifact: `{feature_dir}/blacklist/freq_min{m}_max{M}.txt` (+ `.meta.json`), one feature id per line (`feature_mask.path`-consumable).
- Training auto-generates it when either `feature_mask_freq_min` or `feature_mask_freq_max` is set and no explicit `feature_mask` is configured. KEEP iff `freq_min < freq <= freq_max`; defaults `min=1e-5` (kills dead features and other features too rare to be autointerp-scoreable) and `max=0.5` (kills features active in >50% of segments). Both `null` ⇒ disabled.
- Pre-generate explicitly: `uv run -m scripts.preprocess.precompute_feature_blacklist feature_dir=path/to/features feature_mask_freq_min=1e-5 feature_mask_freq_max=0.5` (config `configs/extraction/precompute_feature_blacklist.yaml`; `force=true` rewrites).
- The generated `.txt` is what a masked training run's `feature_mask.path` points at — and since #695 it therefore also **determines the trained encoder's parameter shape**. Recover it from a checkpoint with `uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>` rather than depending on the file surviving.

See [configuration.md § `feature_mask_freq_min` / `feature_mask_freq_max`](../reference/configuration.md#feature_mask_freq_min--feature_mask_freq_max--auto-activation-frequency-blacklist-producer) for resolution, precedence, no-op/byte-identical rules, and eval reproduction.

### Gradient-attribution scores (selection-layer, derived overlay)

Like the percentile overlay, **gradient-attribution scores** are a derived,
content-addressed overlay (`grad/<hash16>/`, one fp16 score per stored
(segment, feature)) with no effect on extraction. They are produced by
`scripts.preprocess.precompute_gradient_attribution` (full-model forward +
per-segment backward passes over the SAME parquet data, verified per doc
against the base shards) and consumed by the selection precompute to rank
features by their attribution to future predictions — the stored score is
`percentile(activation) × loss-helpfulness sensitivity`, so the producer reads
the percentile overlay too. See
[gradient-attribution-selection.md](gradient-attribution-selection.md).

### Removed feature clustering

The removed clustering-discovery/overlay pipeline is no longer produced or
consumed. Configs and artifacts carrying its legacy singular
`feature_clustering` schema are rejected at load boundaries before they can
affect hashes, caches, training, dataloaders, evals, or reports. This is
distinct from the plural `feature_clusters` static YAML/JSON preselection input
documented in
[encoder-decoder-training.md § Static Feature Clusters](encoder-decoder-training.md#static-feature-clusters-preselection-only).

## Performance Notes

- **bfloat16** is ~6x faster than float32 with <5% magnitude error
- **nnterp backend** is faster than transformerlens for feature extraction
- **native backend** — home-grown raw-HF Gemma-2/Gemma-3 collector, opt-in via `backend=native` (Gemma only, requires CUDA; `auto` still resolves to nnterp/TransformerLens and never selects native). Correctness-equivalent to nnterp/TransformerLens within the validation oracle tolerances (jaccard≥0.99, mags rtol=1e-2). For residual activations native is **bit-identical** to the production nnterp backend (same raw-HF/SDPA model, max_abs=0.0 empirically verified on gemma-2-2b and gemma-3-1b-pt); the validation contract uses a documented looser native-vs-TL magnitude tolerance (`mag_rtol=0.04, mag_atol=0.15`) for Gemma-2, reflecting the pre-existing eager-vs-SDPA numerical drift from `attn_logit_softcapping=50.0` — count (rdiff≤0.01) and jaccard (≥0.99) remain tight so structural/site bugs are still caught.
  - **Truncated fast path:** `backend=native` now runs a *structurally truncated* forward instead of a full forward + hook. At build time the inner Gemma-2/Gemma-3 decoder stack is sliced to layers `0..L` (where `L` is the configured extraction `layer`), the final `.norm` is replaced with `Identity()` (so the truncated model's output IS `resid_post(L)` — the only op the Gemma forward applies between the last decoder layer and its output is `self.norm`), and layers `>L` plus `lm_head` are freed. There is **no forward hook**. The truncated capture is byte-identical (`torch.equal`, max_abs=0.0) to the previous full-forward + forward-hook capture; correctness is gated, not assumed.
  - **Residual-only:** the truncated collector serves the `resid_post(L)` site *only*. Requesting `activation_type` mlp/attention — or a layer other than the one the model was built/truncated for — raises a typed `ExtractionError` (caught by the normal backend fallback chain). For non-residual activations use `backend=auto`/`nnterp`.
  - **Measured speedup** (bf16, RTX 4090, vs the Phase-1 full-forward + forward-hook baseline; *measured, not guaranteed*): gemma-2-2b @ layer 6 ≈ **4.9×** faster (≈179 → 37 ms/forward) with peak GPU memory ≈ **2.2×** lower (≈17.5 → 8.0 GB); gemma-3-1b-pt @ layer 7 ≈ **3.7×** faster (≈75 → 20 ms) with ≈ **2.2×** lower peak memory (≈7.4 → 3.3 GB), also ≈1.4× faster than nnterp. Reproduce with `scripts/benchmark/benchmark_activation_methods.py`, which runs a **pre-timing correctness gate**: it refuses to time or report any method whose output numerically diverges from the full-forward-hook reference (no fast-but-wrong numbers) — that gate verified the truncated path byte-identical (`max_abs=0`) to the Phase-1 reference on both real models before any timing.
  - **Runtime gates:** native `torch.compile` and SDPA row-packing are mandatory self-verifying gates before shard writes. The compile gate compares compiled vs eager against an fp32 reference; the row-pack gate compares packed vs unpacked against an fp32 reference on the same row surface extraction will emit: fixed-token segment rows, context-independent BOS injection/drop when enabled, always-on BOS filtering, and the configured pooling mode. A pass activates the fast path for the run. A fail raises `CompileGateError` or `RowPackGateError` with no silent fallback. Current evidence does not support Gemma 3 cross-document leakage as the root cause of the prior context-independent/no-BOS row-pack rejection; that rejection came from a non-production all-token/BOS-heavy gate shape. FlashAttention-3/4 and FlexAttention row-packing remain out of scope here.
- On NFS/networked storage, initial model loading can take 10-20 minutes
- GPU utilization monitor available via `steering/profiling/gpu_utilization.py`

## Local-Disk Staging

On runs with networked output storage (NFS, object-store-backed mounts), the GPU can sit idle waiting for each shard to flush before the next forward pass begins. The `staging` config group solves this by writing each shard to fast local disk first, then moving it to the final network mount in a background thread — decoupling GPU forward passes from network I/O.

### Configuration keys (`configs/extraction/extract_features.yaml`)

| Key | Default | Description |
|---|---|---|
| `staging.scratch_dir` | `/tmp/nanochat_extract_scratch` | Must be a real local disk (ideally distinct from the network `output_dir`). The loop creates `run_<output-hash>/range_<start>_<num_shards>/rank{r}/{out,in}` below it; writes are atomic and `relocate()` handles cross-device moves to `output_dir`. |
| `staging.output_depth` | `3` | Concurrency of the staged save+move pipeline. `output_depth >= 1` enables staging: up to N shards are saved to local scratch and moved to `output_dir` concurrently. `output_depth = 0` disables staging entirely and writes each shard synchronously and directly to `output_dir` (the pre-staging behavior). |
| `staging.input_depth` | `1` | `>= 1`: N-deep parquet→local-scratch→read prefetch. Each rank stages incoming parquet shards into its own `rank{r}/in` scratch subdirectory (via `shutil.copyfile`); the staged copy is deleted as soon as the texts are loaded into RAM. Up to `input_depth` shards are in-flight concurrently, pipelined ahead of the main GPU loop — each holds its decoded text in host RAM, so raise this only with RAM headroom. `0`: inline read directly from the network parquet (pre-staging behavior). |
| `staging.min_free_gb` | `6.0` | Per-save free-space floor on both `scratch_dir` and the final `output_dir` filesystem. If either falls below this threshold the saver logs periodically and blocks for a bounded interval, then raises if space does not recover. |
| `staging.est_shard_gb` | `2.0` | Estimated compressed shard size in GB. Used only at startup (on local rank 0) for independent scratch/output capacity checks: `output_depth × est_shard_gb × local_world_size + min_free_gb ≤ available_free_gb`. It does not reserve space or affect runtime behavior. |

### Scratch ownership and concurrent range jobs

The extraction window is the half-open interval `[start_shard, num_shards)`, where
`num_shards` is an exclusive upper bound despite its historical name. Staged
scratch is namespaced as:

```
scratch_dir/run_<output-dir-hash>/range_<start_shard>_<num_shards>/rank{r}/{out,in}
```

All ranks in one torchrun launch receive the same range component and retain
separate rank subdirectories. Two independent jobs may safely share a
`staging.scratch_dir` and `output_dir` only when their windows are disjoint (for
example `[0,150)` and `[150,300)`). An identical or overlapping window still
has competing ownership of the same final feature shards and is not made safe by
this scratch isolation. Resuming a job with the exact same `output_dir`,
`start_shard`, and `num_shards` reuses its namespace and sweeps only that window's
stale rank scratch. Capacity checks must cover the aggregate `output_depth ×
est_shard_gb × local_world_size` demand of all simultaneous jobs.

The first range-aware run does not sweep legacy pre-range files left under an
older `run_<output-hash>/rank{r}` layout. This avoids deleting an older launch
that is still finishing; reclaim such an orphan only after confirming no
legacy-layout job is active.

> **Disk-budget note.** The output shards dominate local scratch: peak ≈ `output_depth × shard_size × ranks` (one ~1.8 GB shard per in-flight save+move slot per rank, in `rank{r}/out/`). The input prefetch adds only a small transient term — `input_depth × parquet_size × ranks` (~91 MB parquets in `rank{r}/in/`, each deleted as soon as its texts are read into RAM). For the defaults `output_depth=3`, ~2 GB shards, 4 ranks that is ≈ 24 GB output + ~1 GB input. The startup capacity check (`assert_startup_free_space`) guards on the output term; ensure `scratch_dir` has headroom before a large run.

> **Rename (`save_concurrency` → `staging.output_depth`).** The former top-level `save_concurrency` key that controlled an `AsyncSaver` depth is superseded by `staging.output_depth`. The `StagedSaver` is the sole save codepath; `output_depth=0` is the synchronous direct-save equivalent of the old `save_concurrency=1` path.

> **EXDEV-safe move.** The `relocate()` function in `steering/staging/relocate.py` moves each shard from local scratch to the network `output_dir`. On a cross-filesystem move (which triggers `EXDEV`) it falls back to a copy-then-unlink sequence, preserving atomicity: the shard appears at the destination only after the full copy completes.

### Verifying byte identity after a staged run

> **Destination capacity guard.** The startup capacity estimate is applied independently to scratch and the final output filesystem. Before each staged relocation, the saver applies the same bounded free-space wait to the destination. These checks are an admission guard rather than a reservation, so concurrent writers and filesystem quotas can still trigger the final error guard.

> **Destination failures.** If destination creation, copying, syncing, or publication returns `ENOSPC` or `EDQUOT`, `relocate()` raises `DestinationSpaceError` naming the destination while retaining the scratch source and cleaning any temporary destination file. Other errors keep their original behavior.

`scripts/benchmark/verify_staging_identity.py` is the on-box oracle. Run extraction twice — once with `staging.output_depth=0` (synchronous direct write) and once with staging enabled — then diff the two output directories by decompressed tensors:

```bash
# Direct write (staging disabled)
uv run python -m steering.extract_features \
    --config-name extract_features \
    output_dir=/tmp/feat_direct \
    staging.output_depth=0 \
    num_shards=2

# Staged write
uv run python -m steering.extract_features \
    --config-name extract_features \
    output_dir=/tmp/feat_staged \
    num_shards=2

# Diff by decompressed tensors (exits non-zero on any mismatch)
uv run python -m scripts.benchmark.verify_staging_identity /tmp/feat_direct /tmp/feat_staged
```

The tool loads each shard via `steering.feature_io.load_features`, which decompresses and delta-decodes the stored data, then compares offsets, segmentation metadata, and per-segment feature IDs/magnitudes in **stored order** (feature_io co-sorts feature IDs ascending within each segment, so the stored order is deterministic). It compares the ID and magnitude arrays directly — *not* via independent per-array sorts — so a within-segment ID↔magnitude mispairing is flagged, not hidden. Empty = identical; any mismatch is printed and the process exits with code 1.

> **Input staging is byte-identical.** The `StagedPrefetcher` copies each parquet shard to local scratch via `shutil.copyfile` — a faithful byte-for-byte copy — and then deletes the local copy after the texts are loaded into RAM. The extraction forward pass sees exactly the same texts as an inline read, so the output shards are identical regardless of `staging.input_depth`.

### Batch-log progress and liveness

Extraction emits an INFO line when each shard starts, when its output save
starts, and after extraction completes. During extraction it also emits a
rate-limited heartbeat after every 1,000 completed documents or 60 seconds,
whichever comes first. Each heartbeat has stable `key=value` fields for the
canonical shard name, phase, completed/total documents, fractional progress,
completed segments, and elapsed seconds. These messages are the durable
progress signal for redirected Slurm logs; they are emitted at completed-work
boundaries and therefore do not claim that a long-running model or filesystem
operation is still making progress.

The four tqdm bars (shards, row groups, and document batches) are enabled only
when stderr is a TTY. Redirected logs therefore contain no ANSI progress-bar
control sequences. The cadence is intentionally fixed and has no Hydra knob,
so it cannot change extraction identity or shard bytes. The heartbeat is an
operational liveness aid, not a completion certificate: a run is complete only
after the normal shard result, save/relocation, and final manifest checks pass.

## Token-Boundary Overlay Precomputation

`scripts/preprocess/precompute_token_boundaries` maps each base shard's source-model segment
char boundaries onto nanochat token boundaries and writes them as an immutable overlay under
`tok/<tokenizer_checksum[:16]>/seg_boundaries/`. The overlay tensors depend only on the
tokenization-derived segmentation metadata, never on `feature_ids`/`feature_mags`.

### Performance options (issue #529)

Three optimizations are **unconditional** — each was proven **byte-identical** to the prior
implementation (see `tests/scripts/test_precompute_token_boundaries_optimization.py` and
`tests/steering/test_base_shard_hash_cache.py`), so each is the only codepath with no opt-out
flag. End-to-end on the real 370-shard gemma-3-4b-pt corpus they are byte-identical to the
baseline and cut first-run wall-clock by **~16%** (~5 h):

- **Base-shard hash cache.** The `base_shard_content_hash` is cached in
  `<feature_dir>/.base_shard_hash_cache.json`, stat-validated by each shard's
  `(name, size, mtime_ns, ctime_ns, inode)`. On a hit (base shards unchanged) the per-run hash
  **skips reading every base-shard byte** — turning the 20–30 min manifest pre-pass on a
  1 TB / 370-shard corpus into a stat-only check. The cached value is **exactly** the streaming
  SHA-256, so existing overlays still validate. Any mutation (or a corrupt/old-version cache)
  falls through to a full recompute, and **every cache miss is logged** (`_load_hash_cache`:
  corruption at `WARNING`, stale version at `INFO`, absent cache at `DEBUG`).
- **Worker-pool reuse.** One `spawn`-context worker pool serves the whole run instead of
  respawning + re-pickling the tokenizer per shard (worker init args are detected once and
  constant across shards). Measured **~50 s/shard** saved.
- **Subset load.** Only the segmentation tensors each path consumes are loaded, skipping the
  large `feature_ids`/`feature_mags` upcast + delta-decode; `segment_size` is read from the
  safetensors header only (no body decompress). Measured **~5 s/shard** saved.

| Config key | Default | Effect |
|---|---|---|
| `zstd_level` | `12` | Overlay-save compression level. |
| `workers` | `0` (auto: `cpu_count // 2`) | CPU process workers; each worker is capped to one torch intra-op/inter-op thread and `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. |

### Compact-only HF input

Remote token-boundary runs require a one-time compact source. The configured
`remote.remote_features_root` still maps the logical `feature_dir` to the raw
HF prefix, which is sealed as ordered `(path, byte_length, xet_hash)` rows. Its
digest derives the immutable compact namespace:

```text
<raw-prefix>/_token_boundary_source/v1/<raw-inventory-sha256>/
```

The compact source stores only `offsets`, document/segment/character geometry,
document token counts, and four scalar geometry values. It contains no text and
does not replace `data_dir`: local parquets remain required and must be the exact
family paired with the raw tree. The token-boundary algorithm, tokenizer, worker
pool, overlay bytes, schema-v2 manifest, resume behavior, and downstream readers
are unchanged. Local mode continues to read raw shards exactly as before.

Prepare a complete production source explicitly (this uploads and needs separate
authorization):

```bash
uv run python -m scripts.preprocess.publish_token_boundary_source \
  raw_prefix=<raw-hf-prefix> \
  staging_dir=<existing-dedicated-scratch> \
  expected_shard_count=1400
```

The publisher streams the raw family with one-body lookahead, computes the
existing exact compressed-byte `base_shard_content_hash`, writes deterministic
empty-metadata level-12 compact shards, and verifies each after reload. Sidecars
are immutable and may remain after interruption. Only a strict canonical
`manifest.json`, uploaded after both raw and compact final inventory fences,
commits the family; reruns verify identical sidecars instead of overwriting
them. A one-time source build therefore costs approximately one old raw pass,
while later remote runs receive the compact transfer benefit.

With `remote.enabled=true`, startup authenticates the raw inventory, derived
namespace, commit manifest, exact compact inventory, and paired parquet
preflight before starting compact-body transport. Missing source publication
fails with the publisher command; there is no flag, automatic preparation, or
raw fallback. One verified worker holds the current compact body plus one
lookahead, and disk admission is the two largest compact bodies plus 1 GiB.
Every compact shard is schema-validated through its sealed regular-file
identity. The consumer never downloads a direct raw shard body.

Computed overlays first land in a run-private candidate directory. Existing v2
manifests authenticate against the raw base hash certified by the compact
manifest. Before canonical publication the raw inventory, compact inventory,
and compact manifest identity must all remain unchanged. Fresh and forced
schema-v2 overlay manifests are written last; resume preserves existing
manifest bytes. `force=true` still requires the complete family, partial-range
residue checks are unchanged, and `dry_run=true` computes and validates private
candidates before discarding them.

Example production shape (only after the 1,400-object inventory and parquet
pairing have been independently re-audited):

```bash
uv run python -m scripts.preprocess.precompute_token_boundaries \
  feature_dir=<logical-feature-dir> \
  data_dir=<paired-local-parquets> \
  remote.enabled=true \
  remote.staging_dir=<dedicated-scratch> \
  remote.expected_shard_count=1400 \
  profile=true profile_trace=false
```

### CPU partitioning for concurrent producers (issue #1269)

The token-boundary pool is CPU-bound. An uncapped token pool can consume roughly
30 cores and starve a concurrent percentile producer down to about one runnable
core. Partition the machine explicitly so both producers retain predictable
throughput:

```bash
taskset -c 0-19  uv run python -m scripts.preprocess.precompute_token_boundaries ... workers=20
taskset -c 20-31 uv run python -m scripts.preprocess.precompute_percentiles ...
```

> **Note on the base-shard hash.** `save_features` is not byte-deterministic across
> re-extractions (the safetensors Rust backend randomizes metadata key order per save), so
> `base_shard_content_hash` legitimately changes whenever base shards are re-extracted — the
> cache keys on the *actual on-disk bytes* (via the stat signature) and never returns a stale
> value.

#### Base-shard hash cache: sibling writers (issue #530)

The cache is wired into the overlay writers, not just token boundaries:

- **`precompute_percentiles`** and local-mode **`precompute_packed_rows`** (both train and val phases) hash `glob_features(feature_dir)` and store the cache under **`<feature_dir>/.base_shard_hash_cache.json`** — one shared cache per feature directory. Remote packed-row mode downloads only the selected raw/percentile prefixes, takes the base hash agreed by the verified remote percentile and local token manifests, and does not read or write this cache.
- **`precompute_selected_features`** is different: it hashes the **packed-row** shards (`glob_overlay_shards(packed_row_dir)`) and stores the cache under **`<packed_row_dir>/.base_shard_hash_cache.json`**, not under `feature_dir`.

The extension is **unconditional** (no config toggle, same project stance as #529) and **output-invariant**: the cached digest is always exactly the streaming SHA-256, so no overlay tensor, manifest hash, or packed-row content changes. End-to-end byte-identity was verified by a cross-checkout snapshot (this branch vs pre-#530 main on the same base shards).

### Profiling

Run with `profile=true` to write a JSON sidecar (`profile_summary_output`, default
`out/traces/tok_boundary_profile.json`) with `manifest_pre_pass.{seconds,bytes_read,cache_hit}`,
`worker_pool.{spawn_count,total_init_seconds}`, and per-shard `{load_features, parquet_load,
batch_encode, tokenize, overlay_save}` seconds + bytes. `scripts/benchmark/benchmark_token_boundaries.py`
runs baseline vs optimized variants and emits a Markdown table — point it at the real corpus for
the headline before/after.

## Packed-Row Precomputation

`scripts/preprocess/precompute_packed_rows` (step 5/5) runs the same document iteration +
best-fit packing as the live dataloader, capturing per-row CSR data with ALL extraction
features and writing compressed packed-row shards under
`pct/<pct[:16]>/tok/<tok[:16]>/c{context_len}/s{max_shards}/`. These shards are the **direct
training input** for the encoder-decoder pipeline, so every optimization below is
**byte-identical** to the prior serial implementation.

### Identity-sealed HF raw and percentile input

`remote.enabled=true` replaces raw `shard_NNNNN_features.safetensors.zst` and
percentile `pct/<pct16>/shard_NNNNN.safetensors.zst` reads with two bounded HF
Storage Bucket streams. Statistics, token-boundary leaves, source parquets,
packed/provenance output, FP certificates, mirrors, and profiles remain local
(unless output offload is enabled — see below).
The percentile manifest is fetched once with verified transport from the sealed
bucket listing and validated with the ordinary schema-v2/checksum parser. Local
raw and percentile bodies are deliberately ignored; there is no fallback,
persistent body cache, remote deletion, FUSE mount, or publication
symlink step, and nothing is ever written to the sealed input prefixes.

Startup maps the symlink-resolved logical `feature_dir` from
`remote.local_features_root` to `remote.remote_features_root`, seals the exact
contiguous raw `(path, byte_length, xet_hash)` family and its contiguous
`pct/<pct16>/` subfamily, then checks both against the same-position stems and
selected parquets before downloading a body. Production
must set `remote.expected_shard_count=1400`; null seals the observed contiguous
family and emits a warning. The dedicated staging directory must be disjoint in
both directions from the feature tree, parquet tree, and enabled profile path.
Admission reserves the two largest raw bodies plus the two largest percentile
bodies plus 1 GiB. Steady state holds each stream's current body plus one
lookahead, one decoded/indexed selected shard, and the existing bounded
packed-output compression queue.

`max_shards=N` selects and downloads only the first N parquet-matched bodies
from both streams before the train/validation split. Each download is
size/Xet-verified against its own sealed inventory. The verified remote
percentile manifest and local token manifest must agree on the expected base
content hash; the sealed percentile inventory must cover the raw family
position-for-position, and local token leaf names may only be a subset. The
token overlay is rediscovered against the agreed hash. Final HF re-lists of both
families must exactly match their startup snapshots before any packed manifest
is finalized. Remote mode never writes `.base_shard_hash_cache.json`.

The remote trust boundary is explicit: every downloaded raw and percentile body
is checked for size and Xet identity against its sealed inventory, and each
selected shard is structurally cross-checked for document count, segment size,
overlay numel, and segment counts. No consumer-side check binds the raw
inventory to the base hash agreed by the verified percentile and local token
manifests. That correspondence is an operational trust assumption under the
owner ruling for issue #1437.

Packed and provenance shards are written incrementally to their ordinary
canonical directories. A sibling
`.sN.remote-packing-state.json` (or `.sall...`) authenticates an interrupted
producer's raw and percentile inventory digests, semantic source identity,
selected parquet stems/row counts, and packing settings. Schema-v1 states are
refused with instructions to remove the stale state and both partial
train/validation output directories before starting fresh. `resume=true` keeps
the existing contiguous packed+provenance prefix, replays the deterministic
packing stream, and regenerates the first incomplete suffix. Staging/retry,
compression-worker, and profiling changes do not invalidate that state;
`zstd_level` and packing settings do. `resume_from_shard` is rejected because it
can skip missing prefixes and understate manifest dimensions.
An already-completed resume authenticates the manifests/certificates, republishes
mirrors idempotently, and downloads no shard bodies (the percentile manifest is
still fetched once for source authentication).

Incremental payload visibility is not completion. An output is usable only when
the in-progress state is absent and both the train and validation manifests have
valid completed FP source certificates. Do not start consumers concurrently with
the producer. Validation is finalized first and train last as the practical pair
commit point; a failed finalization keeps the state for an authenticated retry.

#### Output offload: publish, verify, then evict (issue #1462)

A production family is larger than the filesystem that produces it — c2048 emits
~1.5 GiB per output shard over ~1,600 shards (~2.4 TB) against a 1 TB pool
quota — so a producer that retains its own output exhausts the quota mid-run.
`remote.offload.enabled=true` publishes each completed packed body and its
provenance sidecar to `remote.offload.remote_packed_root`, verifies the
published object's size and Xet identity, writes a durable receipt, and only
then removes the local bodies. That order is what makes it safe: a body is never
deleted before the bucket is proven to hold its exact bytes, and a corrupted or
truncated upload fails the run loudly with both local copies intact.

- **Published layout** mirrors the local tree under the configured root
  (`<root>/<packed dir relative to remote.local_features_root>/shard_NNNNN.safetensors.zst`
  plus `.../provenance/shard_NNNNN.safetensors.zst`) — the same layout
  `remote_preselection` already materializes packed bodies from. The root must
  be disjoint from `remote.remote_features_root`; overlap is refused, so output
  can never land inside the sealed input family.
- **Receipts** live in a sibling `_offload/shard_NNNNN.json` namespace and are
  the durable proof of publication: an offloaded shard is resume-complete from
  its receipt alone, and manifest accounting counts the family whether a shard's
  bodies are local or published. Each receipt holds that shard's FP-target
  attestation (see below). Max dims still come from the tiny `_dims/` companion,
  which is never evicted and is still required for an offloaded shard to count
  as kept — so a lost companion makes the packer re-pack and re-publish that
  shard (idempotently) rather than silently understate the manifest.
- **FP-target certification moves to publication time.** The whole-family scan
  opens every local body, which an evicted family cannot offer and which would
  otherwise certify an empty directory as a complete passing scan. Each body is
  instead scanned in the last moment it is local, and the certificate is folded
  from those attestations — byte-identical to the certificate the local scan
  would have written, and re-validated through
  `validate_fp_target_source_certificate_remote`.
- **Backpressure bounds disk.** One serialized upload worker publishes in shard
  order and the producer blocks once `compress_workers + 2` shards await
  publication, so local growth is capped instead of racing the quota.
- **Reconcile on start.** Before packing resumes, any completed shard still on
  disk (from a killed window, from a run that predates offload, or from a
  finished retained family) is published and evicted first — that reclaimed
  headroom is what the rest of the run spends. Enabling offload does **not**
  invalidate an in-progress resume state, so an interrupted retained run can be
  resumed directly into offload mode.
- **Republication is the recovery path.** A shard whose eviction was interrupted
  is simply re-offered to the bucket: `upload_verified_object` is idempotent for
  identical bytes and refuses a same-path collision whose bytes differ, so the
  local body is proven to be the published one before it is finally removed.
- **Mirrors are not published** for an offloaded family: a mirror is a symlink
  to a local body, so it could only dangle. Consume the family from the bucket
  prefix the run logs on startup.

```bash
uv run python -m scripts.preprocess.precompute_packed_rows \
  ... \
  remote.enabled=true \
  remote.offload.enabled=true \
  remote.offload.remote_packed_root=l22_ci_nobos_packed/splice/cut_frac0.15/seed42
```

The profile sidecar reports `remote_transport.offload.{published_shards,
reconciled_shards,evicted_bytes,upload_seconds,backpressure_seconds}`.

#### Signing an offloaded family (issue #1463)

Pre-selection and remote training both read `remote_packed_inventory.json` from
the packed prefix, and the local builder opens every packed and provenance body
at once — which an offloaded family cannot offer, and materializing one just to
sign it would re-create the disk problem offload exists to avoid. So
`publish_remote_packed_inventory` gains a `remote.enabled=true` mode that
streams each published body back exactly once and publishes the signed
inventory beside the bodies:

```bash
uv run python -m scripts.preprocess.publish_remote_packed_inventory \
  packed_row_dir=<canonical packed dir> \
  remote.enabled=true \
  remote.local_features_root=<same root the producer used> \
  remote.remote_packed_root=<same offload root the producer used> \
  remote.staging_dir=<dedicated scratch>
```

Both bodies of each shard arrive through the same verified one-lookahead
transport the producer reads its inputs with, so the next download overlaps the
current body's hashing and the pass costs bandwidth plus a **two-body lookahead
per stream** rather than the whole family. Each body is read once:
`packed_body_digest` folds its size, content digest, name-salted fingerprint and
the running family hash in a single pass. The local and streaming builders share
that helper, the entry shape, and the signing envelope, so signing from the
bucket reproduces the local signature field for field — the gate is exactly
that, on one family signed both ways.

The one field that forces the full pass is `base_shard_content_hash`: it is a
single serial SHA-256 over every packed body in canonical order (not a fold of
per-shard digests), so no receipt can shortcut it. `total_segments`,
`byte_length` and the fingerprints could be, but they arrive free once the bytes
are already streaming.

Fresh-run production spelling (after a separately approved metadata listing
and bounded smoke):

```bash
uv run python -m scripts.preprocess.precompute_packed_rows \
  feature_dir=<production-l22-logical-feature-dir> \
  data_dir=<local-parquet-dir> \
  remote.enabled=true \
  remote.staging_dir=<dedicated-scratch> \
  remote.expected_shard_count=1400 \
  compress_workers=1
```

`profile=true` adds raw inventory count/bytes/digest, joint admission headroom,
transfer and wait times, staging high-water bodies/bytes, processed/decoded body
counts, final-fence status, contiguous kept output counts, and terminal
status/error to the usual packed-row sidecar. The percentile stream exposes the
same transport payload under `remote_transport.percentile`.

### Performance options (issue #531)

Real-corpus profiling (gemma-3-4b-pt l22 f65k, 32-core box) showed the cost is dominated by the
**serial zstd save** of the ~1.5 GB output shards (~74 s/shard at level 12) blocking the loop,
the recurring ~25 s base-shard loads, and memory pressure from the accumulator — **not** the
per-row pack/CSR compute (<1 ms/row). The optimizations target that, all proven byte-identical
(golden `tests/scripts/golden/packed_rows/` + `tests/scripts/test_packed_rows_parallel_compress.py`
+ `tests/steering/test_feature_index_skip_mags.py`), confirmed by reproducing real-corpus
`shard_00000` byte-for-byte against the existing `sall` reference run:

That ranking describes the cited corpus and host, not a permanent assumption. Feature density,
row shape, cache state, and memory pressure can move work into CSR construction or index
materialization; use the profiled subphases below before choosing the next optimization.

- **Parallel shard compress + save.** The packing decision (which docs go in which row) is
  strictly serial — the `SortedDocBuffer` state is order-dependent — but once a shard's rows are
  fixed, assembling + zstd-compressing them is a pure function. A bounded `ThreadPoolExecutor`
  (threads, not processes: zstd + the big `cat`/`stack` release the GIL and the ~1.3 MB/row data
  is shared, so process pickling would swamp the gain) compresses shards off the producer thread;
  each shard writes its own file ⇒ **byte-identical across worker counts** (tested at 1/4/16). A
  `BoundedSemaphore` caps in-flight shards (peak memory ≈ `compress_workers` shards); even
  `compress_workers=1` overlaps the next shard's packing with the current compression.
- **FeatureIndex prefetch.** The precompute index is built with `prefetch=True` so the next base
  shard's ~25 s load + index-build overlaps the current shard's packing.
- **uint16 accumulation.** `_RowCSR` holds each row's `feature_ids` at the final `uint16` dtype
  instead of `int64`, cutting the per-shard accumulator (~109 k feats/row × R rows) ~4× — easing
  the load-vs-accumulate memory pressure. `flush`'s `cat(...).to(uint16)` keeps the bytes
  identical (SAE ids < 65536 ⇒ lossless).
- **Subset load (skip `feature_mags`).** The packing CSR uses `feature_ids` + precomputed
  percentiles, never magnitudes, so the precompute `FeatureIndex(skip_feature_mags=True)`
  subset-loads only the segmentation + `feature_ids` fields, skipping the large `feature_mags`
  upcast + split per shard. The **live dataloader path is unchanged** (default `False`).
- **Encoded percentiles.** Packed preprocessing keeps percentile-overlay bytes encoded through
  its private segment carrier and concatenates them directly into the packed CSR. This is the
  only packed-row path; the deleted uint8→float32→uint8 round trip produced identical artifact
  digests on the bounded benchmark. The live dataloader's float32
  `SegFeatures.percentiles` contract is unchanged. Packed preprocessing also disables bucket
  collection in the shared segment builder: bucket values are needed by live collation but are
  never read by packed CSR construction, so its private carrier has no `buckets` field and
  allocates no per-feature bucket zeros.
- **Base-shard hash cache reuse (local mode).** The train + val manifests reuse one
  `_compute_base_hash(cache_dir=feature_dir)` result (the #529 cache) instead of re-reading +
  SHA-256ing every base shard twice. Remote mode instead uses the base hash agreed by the local
  overlay manifests and never creates this cache.

| Config key | Default | Effect |
|---|---|---|
| `compress_workers` | `0` (auto = `min(cpu//4, 8)`) | Parallel shard compress+save threads. Output byte-identical across all values (incl. 1). |
| `zstd_level` | `12` | Output-shard compression level. Level 9 is ~2.8× faster for ~2% larger shards but changes the bytes (not byte-identical) — opt-in for new runs only. |

> **Local resume.** `resume=true` / `resume_from_shard=N` still work and remain byte-identical, via
> the count-based **replay** (re-packs already-written rows to advance the doc stream — expensive
> on a late resume). Replacing that replay with a checkpoint-seek (persist the doc-stream cursor +
> the `SortedDocBuffer` leftovers, then fast-forward via parquet row-group metadata) is the
> remaining issue-#531 follow-up; it is not yet implemented.

> **Resume-safe manifest dims (issue #539).** Each shard writes a tiny `_dims/shard_NNNNN.json`
> companion (`{max_n_seg, max_seg_size}`) atomically beside it — lock-free (each compress worker
> writes its own) and crash-safe. On resume the inner function seeds the global maxima from the
> kept (contiguous) prefix's companions, so the manifest's `max_n_seg` and raw
> `max_seg_size = max(seg_token_end - seg_token_start)` reflect the true max over
> **existing + new** shards —
> never the post-resume tail only, and an idempotent re-run of a complete dir is never reset to
> `0`. Legacy dirs written before #539 (no `_dims/`), missing companions, or a corrupt companion
> log a warning and fall back to a slow reduce-over-shards. **Limitation:** an explicit
> `resume_from_shard=N` whose prefix shards are **not on this box** ("ignoring local file count")
> cannot recover their dims locally — it warns loudly and the manifest reflects only the new +
> locally-present shards (it may understate); use `resume=true` (needs the local shards) or re-run
> `backfill_packed_row_shapes` once all shards are local for exact dims. The `_dims/` subdir is
> invisible to consumers (`glob_overlay_shards` is non-recursive + `.zst`-only) and the shard bytes
> are unchanged (byte-identity preserved). The raw segment maximum is also
> persisted unchanged in loader resume state. Encoder-decoder cross-attention
> validates it as a semantic upper bound, then separately rounds its internal
> allocation to the next multiple of 16 for FP8; neither producer nor backfill
> pre-aligns this metadata.

> **Byte-identity.** Packed-row shards go through `save_overlay_shard`, which writes empty
> safetensors metadata, so the compressed `.zst` bytes are reproducible (unlike `save_features`)
> — the golden anchor and resume tests compare raw bytes directly.

### Profiling

Run with `profile=true` to write a JSON sidecar (`profile_summary_output`, default
`out/traces/packed_rows_profile.json`) with `manifest_pre_pass.{seconds,bytes_read,cache_hit}`,
run-aggregate `stage_totals.{parquet,lookup,tokenize,seg_to_raw,truncate_assert_append,pack,
csr_build}` seconds, `row_build_detail.{row_to_csr_seconds,row_provenance_seconds}` (the two
components already contained by `csr_build`) plus the nested row-to-CSR segment-build,
boundary, feature-concatenation, and tensor-finalization subphases. Feature concatenation is
itself split into feature-ID/percentile concatenation and count construction. The profile also
records lookup-visible
`base_shard_loads.{count,total_seconds}`, startup scan time,
per-resident-shard FeatureIndex read/decode/overlay/document-index subphases, compression
backpressure/max-in-flight, per-output-shard
`{assemble_seconds,save_seconds,worker_seconds,queue_wait_seconds,save_bytes}`, and totals
(`rows_per_sec`, `wall_seconds`). Remote mode also records per-body transport wait, header, and
activation timings; selected bodies use the same FeatureIndex subphase records as local mode.
Prefetched FeatureIndex work overlaps the main
critical path, so its subphase totals are diagnostic work totals rather than additive wall-time
components.

The correctness-gated benchmark requires both inputs and never writes into the source feature
tree: `scripts/benchmark/benchmark_packed_rows.py feature_dir=... data_dir=...
segment_size=128 sae_width=16k context_len=1024 max_shards=4`. It creates one private input/output
tree per variant, hashes and rebinds only the bounded overlay family once, fences source identity,
samples RSS/PSS/swap, and refuses to publish its report
unless every real packed/provenance/dim payload and timestamp-normalized manifest is exactly
equal across variants. The private bounded hash is intentionally recomputed once per invocation
and its setup time is reported; use repeated fresh CLI processes when you need warm-OS-cache
timing without carrying allocator-reserved pages between variants.


### Extraction source identity

New CLI extractions record `extraction_model_id`, `extraction_sae_release`,
`extraction_sae_id`, `n_sae_features`, and `extraction_documents_sha256` in each
shard header. The model/SAE coordinates come from the resolved loader inputs;
the width comes from the loaded SAE. The versioned document digest binds exact
UTF-8 text, order and document boundaries using length-prefixed bytes, computed
from the same text chunks consumed by extraction (including prefetched input).
Both direct and asynchronous saves preserve these fields through `load_features`.
Writing the added header fields changes newly extracted safetensors bytes and
their compressed shard hashes. Registering `n_sae_features` also exposes a field
that historical shards already contained but older readers ignored. The AxBench
seed-pool canonical digest explicitly excludes that field, preserving existing
sidecar identities without rewriting shard bytes. Other newly added parsed
identity fields remain included in that digest for newly extracted shards.
Existing completed shards retain their original bytes on resume; missing
provenance stays unknown. Fixed self-conditioned ChatCORE preparation requires
these fields and rebuilds legacy inputs whose source identity cannot be proved.
See the [ChatCORE pipeline](self-conditioned-chatcore.md) and
[evaluation runbook](evaluation.md#self-conditioned-chatcore).
