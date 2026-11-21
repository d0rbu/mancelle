# AxBench-Upstream Performance Campaign Specification

Status: OWNER-AUTHORIZED 2026-08-26. Implementation learnings are appended
under **Addenda**; ratified contracts are not silently rewritten.

## 1. Goal

Make the two root-environment AxBench-Upstream producers practical at larger
concept counts without changing their scientific meaning:

- `SelfConditioning` and `SelfConditioningMasked` from
  `steering.eval.axbench_upstream.run_methods`;
- `EncDecDiffMean`, `EncDecLinearProbe`, `EncDecPromptSteering`, and
  `EncDecLoRA` from
  `steering.eval.axbench_upstream.encdec_baselines.run`.

The campaign is profile-first. Pure execution changes require byte-identical
artifacts. Kernel or graph changes are a separate numerical tier and cannot be
adopted without fixed-input deltas, full regression evidence, and owner review
of a judged and roundtrip c2 A/B.

## 2. Scope and non-goals

In scope:

- reproducible profiling and a c2/c500 cost model;
- exact comparison tooling against fresh current-revision serial oracles;
- one production executor for serial and distributed cell execution;
- per-concept fit ownership;
- first-concept tripwire preservation;
- exact crash-safe resume;
- rank-zero-only canonical publication;
- evidence-justified eval cross-attention graph/kernel work.

Out of scope:

- `nanochat/` changes;
- vendored Qwen execution;
- merge, judge, or roundtrip performance;
- new scientific comparison knobs;
- self-conditioned cross-cell batching without a separate design review;
- LoRA batch-semantics changes without separate owner approval;
- paid judge calls without explicit approval.

## 3. Corrected workload arithmetic

One c2 source frame has `2 concepts * 10 inputs * 12 factors = 240` rows.
Rows are not generation cells:

| Producer | Calls per ordinary row | Extra calls | c2 calls | c500 calls |
|---|---:|---:|---:|---:|
| Project | 2 methods | none | 480 | 120,000 |
| Encoder-decoder | 2 main methods | 2 aux methods on factor 1 | 520 | 130,000 |

The accepted encoder-decoder artifact timestamps place its 520-call generation
phase at about 91 minutes on an RTX 4090. The corresponding generation-only
c500 projection is about 380 GPU-hours before profiling-derived corrections.

`self_cond_segment_size` is the decoder sequence length, 1,024. Prefill resets
the continuation counter to zero. With `max_new_tokens=128`, the streaming
loop forwards at most 127 emitted continuation tokens, so the boundary rewind
path is not reached in the c2 workload.

## 4. Current semantic call graph

Both drivers perform strict scalar/path/source/prepared/checkpoint validation,
prepare real Gemma/SAE conditioning, release the extraction stack, load the
target checkpoint, and then generate through:

```text
generate_self_conditioning / generate_encdec_baseline
  -> build_self_conditioning_config
  -> SingleEntryContextProvider
  -> build_engine_self_conditioned
  -> InferenceEngine.generate_stream
  -> TargetAwareEncoderDecoderEvalWrapper
  -> EncoderDecoderGPT
  -> compiled packed Flex cross-attention dispatcher
```

Every cell currently receives a fresh selection RNG, provider, wrapper,
engine, sampling RNG, and KV cache. Residual hooks are removed in `finally`.
LoRA adapters restore the exact original module objects in `finally`.

## 5. Logical identity

### 5.1 Cell identity

The complete immutable identity of one generation cell is:

```text
(driver, method, source_row_position, concept_id, input_id, factor.hex(),
 selection_seed, token_seed)
```

The existing seed derivations remain authoritative and placement-free. No
rank, world size, device index, execution order, or resume location enters a
seed.

### 5.2 Semantic run identity

Resume identity binds at least:

- producer and resume schema/algorithm versions;
- repository commit and root lock digest;
- source Parquet bytes and prepared-manifest identity;
- prepared metadata/training-data bytes, target metadata, and the referenced
  checkpoint body's exact size and SHA-256;
- feature-statistics bytes;
- every scientific configuration value and comparison constant;
- tokenizer/template and method/factor tables;
- Torch/CUDA/backend build and GPU compute architecture.

Rank, world size, local device index, timing, output path, and progress are
operational. World-size changes may reuse completed logical cells only when
all semantic identity fields match.

Heterogeneous numerical runtimes fail loudly rather than mixing resume rows.

## 6. Correctness tiers

### Tier A: pure orchestration

Required equality against a fresh serial oracle:

- byte-identical canonical Parquet files;
- identical schema, order, dtypes, nulls, floats, and generation strings;
- bitwise-identical direction and LoRA tensors with exactly equal metadata;
- canonical candidate direction and LoRA safetensor bytes equal to the
  canonicalized legacy reference bytes;
- identical seeds for every logical cell;
- identical manifest content after removing only a literal tested allowlist
  of operational provenance fields.

Process sharding is not presumed to be Tier A merely because seeds are stable.
It changes each process's compiler history. A real-GPU artifact differential
is therefore a required classification gate.

### Tier B: numerical execution

Any backend, graph, stride, contiguity, prewarm, compile, or CUDA-graph change
that can alter floating-point execution is Tier B. It requires:

- fixed-input numeric deltas against the current kernel;
- bounded graph counts and no silent eager fallback;
- deterministic replay under the same numerical configuration;
- the full CPU and realistic GPU suites;
- fresh c2 A/B artifacts;
- judged and roundtrip metric deltas reported for owner signoff.

No Tier B result is silently reclassified as Tier A because text happens to
match on a small sample.

## 7. Distributed execution contract

The initial supported production topology is one node, one process per visible
GPU, and a shared filesystem. World size one and world size N use one executor.
Torchrun coordinates are resolved through the repository's standard compute
initialization and cleanup.

The canonical ordered cell grid is constructed only after strict input
validation. Cell owner is:

```text
canonical_cell_index % world_size == rank
```

Fit owner is:

```text
concept_id % world_size == rank
```

All ranks must agree on semantic identity and the complete logical grid before
heavy work. Every collective follows an aligned failure exchange so one rank's
ordinary exception cannot leave peers blocked in a later collective.

## 8. Tripwire contract

The current drivers fit all concepts before generation, then execute every
cell of the first concept seen in the source frame before any later-concept
cell. Distributed execution preserves that order at phase granularity:

1. ranks fit their concept stripes;
2. ranks execute only their first-concept cell stripes;
3. rank zero reconstructs the complete per-method first-concept populations;
4. the existing blank and median checks run unchanged;
5. the decision is broadcast;
6. an immutable tripwire-pass receipt is committed;
7. only then may bulk cells be committed.

Bulk resume rows without the matching tripwire receipt are invalid.

## 9. Resume and publication contract

Operational checkpoints are immutable atomic Parquet batches with an exact
schema and semantic fingerprint. Startup validates every existing batch before
skipping any work. Unknown, malformed, corrupt, stale, out-of-grid, or
conflicting duplicate rows fail loudly. Exact duplicate logical rows may be
deduplicated.

World-size changes are supported because ownership is recomputed over missing
logical keys, not encoded into key identity. A crash may lose only an
unpublished in-memory batch.

Every rank proves exact key-set completion, restores source order, and runs the
existing validators, which makes canonical reconstruction itself part of the
aligned failure surface. Rank zero alone writes Parquet atomically, serializes
the manifest, and publishes the manifest last. Peers receive the exact
success/failure result before distributed cleanup.

`force=true` does not delete, overwrite, or reinterpret mismatched resume
evidence. Mismatched evidence requires fresh output paths.

## 10. Comparison harness contract

The harness has two views:

1. strict artifact bytes;
2. semantic diagnostics explaining any byte difference.

It compares Parquet bytes and logical cell values, safetensor canonical bytes,
tensors and metadata, seed ledgers, and manifests under a literal
operational-field allowlist. A candidate safetensor must already have a strict
canonical header. A legacy reference may be canonicalized in memory before the
final byte check; its tensor payload is copied unchanged. It must be falsified
by tests covering cell mutation, row reorder, factor/float drift, dtype/null
drift, training-tensor drift, noncanonical candidate bytes, digest drift, and
an undeclared manifest difference.

## 11. Profiling contract

Measured arms run in isolated fresh processes. Profiler runs are separate from
throughput runs. After warmup, throughput statistics use at least three
samples and report median, p95, range, actual token counts, HBM, utilization,
and idle fraction.

Required measurements:

1. representative per-cell CPU/CUDA breakdown;
2. eager Flex, compiled Flex, and legal test-only reference attention;
3. phase-tagged Dynamo guard/recompile forensics;
4. engine/config/provider construction and controlled reuse;
5. one complete LoRA accumulation window;
6. non-self-conditioned and small-batch throughput ceilings;
7. c2/c500 cost model before and after each accepted change.

Cross-attention observations include shape, dtype, device, stride, storage
offset, contiguity, `requires_grad`, grad mode, packed extent, and phase.

## 12. Performance acceptance

- c2 encoder-decoder wall time improvement is measured end to end; the target
  is at least 5x and the stretch target is 20x.
- No estimate is reported as a measured result.
- Near-linear multi-GPU scaling requires at least two physical GPUs.
- Time-sliced processes may validate orchestration and support a labeled model,
  but cannot establish measured multi-GPU scaling.
- Correctness gates take precedence over every speed target.

## 13. Documentation and quality

Shipping documentation covers torchrun syntax, world-size-one behavior,
resume identity, failure recovery, artifacts, comparison commands, profiling
commands, and limitations. CLI, configuration, and file references remain in
sync. New production modules receive branch-aware coverage and direct tests of
all reachable failure paths. Realistic GPU, Ruff, Ty, Vulture, and diff-quality
gates are required before delivery. The validation scope may be narrowed by an
explicit owner decision recorded below.

## Addenda

### 2026-08-26: create-once initialization must precede strict shared scans

The first real two-rank Gloo producer run exposed a race that in-process tests
could not: one rank's strict store scanner could observe another rank's
create-once identity temporary file. The same race existed for the project fit
identity. The implementation does not weaken unknown-file rejection. Rank zero
now initializes each shared identity, all ranks complete an aligned replay/open
stage, and only then may any rank publish fit or result artifacts.

### 2026-08-26: project fits are operationally serialized

Project dense DiffMean vectors are atomic per-concept fp32 safetensors under a
separate fingerprint-bound resume directory. They are operational checkpoints,
not canonical bundle outputs and not manifest additions. This is required to
make `concept_id % world_size` fit ownership and changed-world-size recovery
real rather than recomputing every fit on every rank.

### 2026-08-26: available scaling evidence boundary

The implementation host exposes one physical RTX 4090. Real two- and
three-process Gloo tests establish orchestration, failure alignment, logical
partitioning, and byte-exact resume with full c2-shaped fake data. They do not
establish CUDA or multi-GPU scaling. Near-linear scaling and the 5x end-to-end
target remain explicitly unmeasured unless a later run uses at least two
physical GPUs.

### 2026-08-26: tokenizer bytes are semantic input

Prompt rendering and generated-text decode both depend on the external
RustBPE artifact under `NANOCHAT_BASE_DIR`. The resume fingerprint therefore
binds `tokenizer/tokenizer.pkl` by exact length and SHA-256 in addition to the
repository source and rendered source inputs. A tokenizer replacement cannot
silently reuse prior fit or generation cells.

The runtime fingerprint also records the exact Python interpreter version.
This was made explicit after the local and H200 profiling environments proved
to share the same Torch/CUDA/data-stack versions but different Python 3.13
patch releases.

### 2026-08-26: targeted validation supersedes the exhaustive default suite

The owner explicitly stopped the repository-wide default pytest run because it
would take disproportionate time. Delivery therefore uses targeted subsets for
every changed or directly exercised surface: strict comparison, immutable cell
storage, work partitioning, both AxBench-Upstream producers, distributed
failure and changed-world-size resume, entry points, cross-attention, static
quality, and realistic GPU behavior. The final report must list the exact test
commands and results and must not describe this evidence as a full-suite pass.
Unrelated failures discovered before interruption are recorded in the execution
plan and remain outside this campaign rather than being silently deselected.

### 2026-08-27: deterministic training-artifact serialization

Fresh serial oracles proved that the legacy Rust safetensors serializer can
permute JSON keys inside `__metadata__` across processes while preserving every
tensor byte and metadata value. The owner approved a narrow contract amendment:
new direction and LoRA artifacts use the repository's existing deterministic
safetensors serializer plus its existing atomic byte writer. The comparator
requires candidate bytes to be canonical, then compares them with an in-memory
canonicalization of the legacy reference. Tensor keys, dtypes, shapes, values,
metadata keys and metadata values remain exact-equality gates. Canonicalization
rebuilds only the JSON header and eight-byte padding; it copies the tensor data
section byte-for-byte.

Each manifest's declared raw artifact SHA-256 must first match that manifest's
own file. Only after every file passes logical/canonical comparison are those
already-verified digest values replaced in the in-memory manifest semantic
view. This permits the expected legacy-raw versus candidate-canonical digest
difference without adding training digests to the operational allowlist or
accepting a stale/forged declaration.

This exception does not relax Parquet bytes, manifest semantics, cell grids,
seeds, or any scientific value. A noncanonical candidate is rejected even when
it is byte-identical to a noncanonical reference.

### 2026-08-27: overlapping current-main refresh

The base advanced from `3821dc643` to `95da9e471` through PRs #1516-#1520.
Unlike the prior refresh, #1518 and #1520 overlap this campaign's exact
AxBench-Upstream surface by adding checkpoint-phase resolution,
depth-relative layer mapping, and the Pythia-2.8B baseline sweep. The
distributed/resume implementation is reconciled with those contracts. The
`3821dc643` oracles remain historical evidence; fresh `95da9e471` clean serial
oracles are required before candidate classification.
