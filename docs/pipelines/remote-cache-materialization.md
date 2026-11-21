# Strict Rank-Striped Remote Body Prefetch

This document defines the issue-#853 remote-body transport and cache contract
used by encoder-decoder training. It replaces the former cache-miss race in
which a training consumer could become a downloader. There is only one remote
body mode now: deterministic owner processes publish completed downloads, and
training processes are cache-only readers.

[Packed-batch CUDA staging](precomputed-packed-prefetch.md) is a separate,
downstream pipeline stage. Body prefetch owns transport and published compressed
bytes only; it never decodes packed or selection bodies.

Remote clustered splice pre-selection is an upstream producer, not this
training cache. Its parent owns compressed packed staging with at most
`active_workers + 1` bodies. Its transport-backed `FeatureIndex` downloads one
authenticated raw/percentile pair on a parsed-cache miss, performs the ordinary
stable decoder/overlay merge, and removes both compressed files immediately.
Shared-store hits download nothing; shared misses execute inside the existing
cross-process materialization lock. Private workers may download independently,
bounded by `selection_source_cache_shards`.

Pre-selection admission adds the dedicated staging filesystem to the existing
host/cgroup, shared-store, and packed-spool checks. It accounts for bounded
packed lookahead, one source pair under shared locking or one per active private
worker, and one output bound per active worker. Rejection occurs before staging
state or remote output is created and never changes the requested topology.
Receipts and verified-upload claims are nested control objects, so the direct
selection-body inventory consumed by this training cache is unchanged.

## Configuration

Every non-null `remote_training` block must contain:

```yaml
remote_training:
  body_prefetch:
    lookahead_shards_per_rank: 2
    cache_shards: 2
```

`lookahead_shards_per_rank` is a strict integer of at least one. There is no
enable switch and no compatibility mode. A remote configuration that omits the
block fails with a migration error explaining that the synchronous demand
downloader was removed.

The values are operational. They are stripped from checkpoint `user_config` and
excluded, with the rest of the remote runtime state, from semantic run
fingerprints. `body_prefetch.cache_shards` controls trailing retention for both
train pre-selected and packed bodies; it defaults to `2` and does not control
lookahead. The same window is used for selection-only and packed streaming.

The default of two gives up to sixteen train shard-pair obligations finalized,
publishing, or in flight on an eight-rank job. Xet high-performance mode
remains a launch-time environment choice. Do not add or use the deprecated
`hf_transfer` path.

## Ownership and consumption

For signed inventory position `i` and DDP world size `W`, the only transport
owner is:

```text
owner(i) = i mod W
```

The index is the position in the authenticated inventory list. It is never a
parsed shard number and never a rank-local row position. A train selection body
and the packed body with the same inventory position have the same owner.

Transport ownership does not change row ownership:

- one spawned downloader process exists per DDP rank;
- only the nominal owner may contact the backend for a body;
- every training rank still reads and decodes every body needed by its rows;
- the existing row-group striping determines which rows each rank yields;
- consumers never download, stage, publish, adopt, or take ownership;
- owner failure is terminal; another rank cannot take over.

Per-body locks coordinate the predetermined owner with readers and eviction.
They do not elect a downloader.

## Startup contract

Before any downloader is spawned (including a multi-node launch):

1. `LOCAL_WORLD_SIZE` and `WORLD_SIZE` must be present, and
   `LOCAL_WORLD_SIZE` must be a positive divisor of the runtime world size.
   All ranks must expose the same shared cache/lock namespace; the sentinel and
   `flock` probes below prove that requirement.
2. Rank 0 acquires an exclusive job-lifetime session lock in the configured
   cache root. Concurrent jobs must use separate cache roots.
3. Rank 0 writes random sentinel bytes and every rank must observe the exact
   published bytes, not merely agree with the other observations.
4. Rank 0 holds a `flock` canary while every other rank proves that it sees the
   same lock namespace.
5. Rank 0 removes the sentinel and creates a random-nonce control tree.
6. Every control and stage directory is validated as a direct directory. Every
   record and body is bounded, schema-checked, and opened without following a
   leaf symlink.
7. Disk admission is completed before process creation.

Every startup phase exchanges rank-local failures before the next collective,
so a rank-0 lock, sentinel, control-tree, admission, or spawn failure makes all
ranks exit symmetrically instead of stranding peers in a barrier.

Startup gathers the exact resume demand from every rank, including
`batch_rows`, cross-shard continuation rows, and the fresh-run shard-zero
demand. Each rank publishes only its own cursor, progress record, and
requester-specific demand records.

## Worker isolation and scheduling

Each rank starts one non-daemon child with the multiprocessing `spawn` context.
The child constructs its own `HfBucketBackend`; no `HfApi`, httpx, Xet, CUDA,
discovery, training, or checkpoint-publication process state is inherited. A
worker performs at most one transfer at a time and never invokes a distributed
collective.

Workers read the union of rank cursors. Each finalizes the nearest
`lookahead_shards_per_rank` obligations in its ownership stripe, including
receipts and pair-ready publication. This makes the final path visible to
shared-filesystem clients before demand. An explicit current demand has
priority over lookahead. Multiple consumers write different requester
directories, while the worker deduplicates transport by signed family identity
and inventory index.

For a paired train position, selection is staged and published before packed.
The pair-ready record appears only after both final bodies and both receipts
exist. Validation selection families use the same owner workers; validation
startup explicitly requests its window.

## Transport and publication

The transport destination is private to one owner PID and UUID. For each body,
the owner:

1. checks cancellation and sticky failures;
2. acquires the existing body lock exclusively;
3. rechecks the final path;
4. downloads through the unique private transport path;
5. trusts the bucket client to return only after a complete transfer;
6. atomically renames the completed transport directly to the final path and
   fsyncs the final directory;
7. writes a nonce-scoped publication receipt;
8. publishes pair-ready only after every component succeeds.

Cancellation after any durable boundary rolls back new final bodies, receipts,
and pair-ready state. A killed process can leave only its uniquely named
transport debris; the next startup removes it only after proving the embedded
PID is dead. A live-PID temporary is a startup error.

There is no canonical hidden stage and no owner-side content verification or
post-download reread. Startup deletes obsolete stages left by older versions.
The HF/Xet client is the transport-integrity boundary; decode remains the
artifact-usability boundary.

## Receipts and reads

A fresh receipt binds:

- schema version and run nonce;
- family identity and inventory aggregate;
- inventory position and exact basename;
- byte length, SHA-256, and optional packed fingerprint from the signed family;
- owner rank and logical generation.

The consumer opens the final body once with
`O_NOFOLLOW|O_NONBLOCK`, requires a direct single-link regular file, and reads
compressed bytes without reopening the pathname. The receipt proves only that
the owner completed publication for this run and generation; it does not claim
that the owner rehashed the downloaded bytes. Strict mode hashes every returned
body, while trusted mode relies on transport completion and decode.

`remote_training.train_packed.trust_existing_cache=true` has only two explicit
effects:

- a body captured and verified before worker startup may retain the existing
  trusted-cache direct-file, single-link, signed-size contract;
- a newly owner-published body skips content hashing after its publication
  receipt is present.

For a preexisting body, the observed descriptor signature is mode, device,
inode, size, `mtime_ns`, `ctime_ns`, and link count. A signature change forces
full verification, but an unchanged signature cannot prove byte identity:
trusted mode may accept same-length bad bytes already named when the descriptor
opens or same-length in-place mutation during the read. Only strict mode
provides per-read content authentication.

Missing, malformed, or stale receipts for fresh bodies fail
immediately. The consumer never repairs them.

## Failure and cancellation

Transport retries never change owners. Retry exhaustion, publication failure,
unexpected worker exit, malformed control state, or worker death creates sticky
rank error state. Consumers poll every rank's status while waiting and report
the family, basename, inventory position, nominal owner, and last worker status
at their bounded timeout.

`RemoteBodyPrefetcher.check_worker_errors()` has two distinct visibility
latencies. Every invocation re-raises an existing sticky failure immediately.
Access to the local worker pipe and owned-process exit code is serialized
between the training thread and remote CPU batch producer: the first check is
immediate, then successful checks occur no more than once per monotonic second
across both threads. A newly published local failure is therefore visible
within one second plus ordinary caller scheduling. A failed local check does
not advance the successful-check time, so the next invocation retries it
immediately.

Cross-rank status records are a separate shared-filesystem check. The first
invocation performs a complete rank-ordered sweep, including the local rank.
After a successful sweep, another starts no sooner than ten monotonic seconds
later. At most one parent thread performs a due sweep; another caller completes
its local checks and returns without waiting for or duplicating peer filesystem
I/O. A peer error can consequently become visible up to ten seconds plus one
sweep duration and ordinary caller scheduling after publication. Missing
records remain permitted. A malformed, inaccessible, stale, symlinked, or
otherwise invalid record still fails loudly and does not advance the successful
sweep time, so the next invocation retries the peer sweep immediately. The
ten-second interval is fixed and is not a Hydra or environment setting.

Neither failure-check cadence changes
`REMOTE_PREFETCH_POLL_SECONDS = 0.05`. Worker commands, cancellation,
body-ready and body-lock waits, startup, teardown, bounded timeout diagnostics,
and `release_session()` terminal polling retain their existing cadence.

The trainer checks worker failure before shard reads, yielded batches,
checkpoint boundaries, validation, and shutdown. The remote CPU batch producer
is explicitly closable and non-daemon. CUDA staging and asynchronous
checkpoint publication keep their own workers but share cancellation and
failure teardown.

Packed or selection decode/semantic-validation errors are annotated downstream
with the exact cache path and logical epoch without changing their type or
traceback. They remain fatal: body transport does not retry a decoder failure,
substitute an empty shard, or publish a fallback.

Teardown order is:

1. stop the outer CPU batch producer or CUDA staging loader;
2. cancel and join the rank-striped body worker;
3. close the checkpoint publisher;
4. after all rank worker status files are terminal, release the cache session
   lock.

A stuck transport may receive bounded hard termination after graceful
cancellation. Teardown failures are attached to an already-active training
exception instead of replacing it.

## Eviction and disk admission

Consumers do not evict. After a rank finishes loading a shard pair it writes
the logical generation. The nominal owner evicts finalized train selection and
packed bodies (as a pair when both are present) only when:

- the minimum progress across all ranks has moved beyond the configured
  trailing `body_prefetch.cache_shards` window;
- no active demand names the position;
- a publication receipt or captured preexisting body exists; and
- the owner obtains the exclusive body lock without racing a reader.

Lookahead bodies are ordinary finalized-body eviction candidates. Epoch wrap
updates logical generations without changing signed body identity. With a
two-shard selection window, generation 200 removes shard indices 0..198;
at generation 370 (epoch 2/shard 0), the previous generation 369 remains and
the newly loaded generation 370 is protected.

The former `2 * cache_shards - 1` claim was not a valid general disk bound. A
paused rank can hold the global minimum while other ranks make every signed
packed body durable. The current admission calculation therefore uses every
signed byte length and conservatively covers:

- all possible finalized packed bodies;
- all accumulating selection finals;
- each owner's largest configured lookahead shard pairs;
- each owner's largest possible active train or validation transport;
- receipts and bounded control records; and
- a one-GiB safety reserve.

Current finals and canonical stages are measured and deducted without double
counting the required peak. Insufficient free space aborts before worker spawn.

## Relationship to other prefetch stages

These mechanisms solve different problems:

1. **Rank-striped body prefetch** moves compressed signed objects from the
   bucket into the shared filesystem cache.
2. **All-rank body consumption** reads under the body lock, applies the
   configured trust policy, decompresses, parses, semantically validates, and
   reconstructs the rows each rank needs.
3. **Closable CPU batch prefetch** backgrounds that work but buffers only
   `grad_accum_steps` complete batches. At B=16/world-size 8 that is four
   microbatches, so next-shard decode begins with only one optimizer step left.
4. **Optional CUDA staging** reuses the same ordinary loader from a
   byte-bounded host producer, allowing decode of N+1 while N remains
   consumable, then moves batches through bounded pageable and pinned queues to
   a rank-local CUDA stream.

The ownership bounds are separate: body prefetch owns compressed cache/stage
bytes; the loader owns transient decompression and the current decoded shard;
`host_queue_bytes` owns complete pageable batches; `pinned_slots` owns pinned
leases and bounds ready CUDA batches. There is no decoded-body cache.

Remote pre-selection producer receipts and verified-upload claim subtrees are
not training bodies. Direct-body discovery ignores those nested objects and
consumes only the timestamp-free completed selection manifest plus its signed
direct selection bodies. Ordinary packed-only and splice-CF producers therefore
share this consumer path without a producer-mode branch.

Issue #867 peer-status polling is also separate. CUDA staging can remove the
incremental decode boundary without removing that control-plane filesystem
cost. The issue-#868 change is stacked on PR #870, which supplies the polling
fix, and both benchmark arms must use that same stacked revision.

Exact resume seeds initial demand from every distinct shard named by pending
`batch_rows`, not only the final cursor shard. Normal lookahead begins at that
cursor and covers continuation rows required when the saved stream is
regrouped to a larger device batch. A migrated first batch can therefore span
several signed shards without bypassing owner assignment, receipts,
cancellation, or cleanup.

A private transport inode transition is an owner-local filesystem event. It is
retried or made terminal; it is not evidence of another rank publishing the
same body.

## Verification

Focused CPU/process gates:

```bash
uv run pytest \
  tests/steering/test_remote_body_prefetch.py \
  tests/steering/test_remote_cache_body_read.py \
  tests/steering/test_remote_training.py \
  tests/steering/test_remote_packed_inventory.py \
  tests/steering/test_feature_clusters_precomputed_loader.py \
  tests/scripts/test_remote_packed_discovery.py \
  tests/scripts/test_conditional_encdec_hydra_contract.py

uv run pytest tests/steering/test_remote_body_prefetch.py -m slow
uvx ty check
uvx ruff check steering scripts tests
```

The process suite covers world sizes 2, 4, and 8; ownership coverage includes
empty, short, uneven, and 584-entry inventories. Permanent semantic tests
compare the strict implementation with local packed/selection artifacts,
including all ten yielded fields, state, RNG, resume, cross-shard rows,
cross-epoch rows, compact/full/disabled FP targets, and clustered selection.

The production acceptance gate must run on the target eight-H100 host with the
signed 584-body inventory and isolated cold cache roots. The corrected
compute-sustaining requirement is 31.56 MB/s useful aggregate body throughput;
acceptance requires at least 40 MB/s, eight distinct owner transfers when eight
eligible bodies are missing, no same-path overlap or non-owner backend calls,
one publication per absent-to-present generation, and exact training/checkpoint
parity. After warmup, require no 32-step decode boundary above two seconds.
Require wall/compute ratio p50 at most 1.05 and p95 at most 1.10 with both arms
on the issue-#867-fixed stack. First-body startup is reported separately.

Run at least three randomized repeats and report network RX, filesystem I/O,
CPU, RSS, Xet buffers, final/stage/temp bytes, HBM, GPU duty, retries, lead
distance, and consumer-wait percentiles. Benchmark Xet launch settings as
separate runs. Local synthetic and small-GPU gates do not substitute for this
production-host acceptance.
