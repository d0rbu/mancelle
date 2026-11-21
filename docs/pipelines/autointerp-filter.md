# Autointerp Intruder Filter

`scripts/preprocess/precompute_autointerp_filter.py` is an offline whitelist producer for `feature_mask`. It scores local extracted SAE features with intruder detection and emits the most interpretable targets.

## Command

```bash
uv run -m scripts.preprocess.precompute_autointerp_filter -- \
  feature_dir=path/to/features \
  output_dir=out/autointerp/my_run \
  token_examples.materialization_batch_size=1
```

Full Gemma 3 4B PT l22 65k context-independent/no-BOS preset:

```bash
uv run -m scripts.preprocess.precompute_autointerp_filter \
  --config-name=precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos
```

For `@hydra.main` entry points, Hydra flags such as `--config-name` must come
before the optional `--` separator. Only `key=value` overrides go after the
separator.

The scorer re-forwards candidate source documents through the SAE and constructs token-level examples. Positive examples highlight target-active tokens, and intruder examples are target-non-activating token windows with Delphi-style false-positive highlights so the judge cannot solve the task from marker presence alone. The default prompt shape is three activating examples plus one intruder, giving four presented options.

The default judge prompt is `judge.prompt_variant=highlight_context_token_first_v1`
with `judge.few_shot_set=sae_realistic_v1`, the best validated pair from the
fixed replay calibration run. Use `judge.prompt_variant=delphi` and
`judge.few_shot_set=delphi` only when you need Delphi-compatible scoring.

Judge scoring is concurrent by default (`judge.max_concurrent=8`): uncached
intruder trials are evaluated through an in-process thread pool while
preserving global target/trial order, parsed-trial cache identity, audit
order, and output schemas -- serial (`max_concurrent=1`) and concurrent runs
over the same targets are byte-identical (see
`tests/steering/autointerp_filter/test_concurrency_parity.py`).
`judge.requests_per_minute` gates judge `complete()` admissions in that
process; `max_concurrent` only caps simultaneous uncached trial evaluations.
Parse retries re-enter `complete()`, and provider transport retries may issue
additional HTTP attempts within one admitted call.

## Single-Process Judge Concurrency

There is no separate process, plan, or coordinator. `judge.max_concurrent`
(default `8`) bounds how many uncached intruder trials are judged at once
through an in-process `ThreadPoolExecutor`; `1` takes a serial fallback loop
with identical semantics. `judge.requests_per_minute` (default `60`)
rate-limits admission of judge `complete()` calls regardless of
`max_concurrent`, and `max_judge_calls` caps total admitted uncached trials
for the whole run (each admitted trial can still make
`judge.parse_max_retries + 1` `complete()` calls before transport retries).

Determinism does not depend on wall-clock completion order.
`_resolve_all_trials` (`steering/eval/autointerp_filter/runner.py`) resolves
every `(target, trial)` globally in one canonical single-threaded admission
pass first: cache hits are filled in place and misses become work items in
strict `(target, trial)` order, so the exact set of admitted trials -- and
where a judge-call budget cap lands -- is fixed *before* any concurrency.
Only judge execution itself runs concurrently. As each successful result is
observed, the calling thread commits it to `intruder_trial_cache.sqlite`
before logging that completion; executor workers never access SQLite.
`_evaluate_trial_wave` then sorts outcomes by their pre-assigned global index
before returning, so result assembly and which trial's error is raised when
several fail remain canonical. A normally drained serial or concurrent wave
caches the same logical set of successful rows, although completion-order
insertion and SQLite's `created_at`/page layout are not file-byte contracts.

One model/SAE/tokenizer loads exactly once per run and is shared across every
target and trial -- there is no per-worker VRAM multiplication.
`scripts/benchmark/measure_autointerp_model_vram.py` measures its resident
VRAM footprint for the production Gemma 3 4B PT l22/f65k stack. The only
cross-run protection is a single `flock`-based `.lockfile` over the whole
`output_dir`, guarding against two runs racing into the same output
directory; there is no plan, no shared on-disk source-index or trial-cache
coordination across processes, and no `/proc/self/fd` lock anchoring.

Token examples for every target are built before any judge call is made (see
Flow below), so the model/SAE are only ever needed during that build phase.
`CachedTokenExampleProvider.release_model()` drops the model/SAE/tokenizer
references and calls `torch.cuda.empty_cache()` once the build phase finishes,
before the judge phase starts -- the judge phase is pure network I/O and does
not touch the provider again. `_ensure_model_loaded` reloads transparently if
the same provider is ever asked to build examples again. Release also clears
the fixed-block startup proof, worker-stream publication state, and
synthetic-probe decoding, so a reloaded model
must repeat the candidate-repeat, serial-shadow, and companion-isolation gates
before another block can commit. This is safe to call even when a caller reuses
one provider across multiple `run()`s.

## Flow

1. Resolve `n_sae_features` from config, `feature_statistics.safetensors`, or the feature-dir width.
2. Resolve candidate IDs, apply an optional input `feature_mask`, build sorted
   singleton targets, then apply `max_targets`.
3. Find candidate source documents for each target and a bounded source-document pool for intruders. Autointerp requires `ignore_bos_features=true` feature shards and rejects legacy/no-BOS-off artifacts. For small target sets the runner can scan shards directly; for full-width runs set `source_index.enabled=true` to build/reuse `source_doc_index.sqlite` plus a manifest pinned to feature/data fingerprints, target IDs, source-domain shards, top-k limits, `context_independent_features`, and `ignore_bos_features`. Source-index builds stream each compressed feature shard through a temporary uncompressed safetensors spool and scan tensor chunks from that file, so peak RAM is bounded by row metadata plus `source_index.chunk_size` rather than the full decompressed shard. The scanner and parent shard merge share one reducer: max-dedup by `(feature_id, stable_key)`, then per-feature top-k by `magnitude DESC, stable_key ASC`. The spool is deleted after each shard; an interrupted process may leave a hidden `.spool.*.safetensors` temp file in the output directory.
The intruder dedup set is scoped per shard rather than globally: feature shard
stems are unique, so cross-shard stable keys never collide, and per-shard
scoping produces byte-identical `intruder_refs` (the reservoir sample is
unchanged) while bounding the set to one shard's intruder candidates instead of
growing with the whole corpus. Duplicate shard stems fail the build loudly.
Set `source_index.scan_workers>1` to scan shards in parallel worker processes;
the parent merges results in shard-index order with at most
`target_count * max_positive_refs_per_feature` survivor rows, plus up to
`scan_workers + 2` pending shard results in the rendezvous window. Output is
byte-identical to the serial build regardless of worker count (see the
`scan_workers` notes in `docs/reference/configuration.md`). Parallel workers
stream shard-spooled and chunk progress events to the parent, so long
source-index builds should show worker scan progress before the parent reaches
the next in-order `SHARD_DONE`; those `SHARD_DONE` lines include separate
`wait=` and `merge=` timings.
Pick `scan_workers` from measurement, not intuition. The source-index build
benchmark prints table hashes plus structured `wait_*`, `merge_*`, `max_spool_*`,
and `estimated_peak_concurrent_spool_*` fields; compare a representative worker
matrix only after the hashes match. More workers can be slower on small or
straggler-heavy subsets because the parent still merges in shard order and every
worker needs decompressed-spool disk headroom.
For repeated full-width runs that differ only in output tag, prompt/judge
settings, or other downstream scoring choices, set
`source_index.shared_cache_dir` to reuse a completed source index across output
directories. The shared-cache filename is derived from the source-index manifest
payload, not from `scan_workers`, so stale semantic contracts are rejected while
worker-count experiments reuse the same SQLite rows.
4. Re-forward those source surfaces through the SAE, build 32-token windows with token-aligned target activations, and cache them in `token_example_cache.sqlite`. Context-independent feature dirs re-forward exactly the extracted segment surface: slice `segment_token_start:end`, inject BOS only when production extraction would, drop injected BOS before token examples, and zero ignored natural BOS positions before active/highlight decisions. Token-example SAE projection is target-feature scoped, so a full 65k run does not project every document through all columns. The generic `token_examples.activation_backend=auto` path uses TransformerLens so prompt examples match the SAELens/Neuronpedia activation space; set `transformerlens` to require the reference path explicitly. TransformerLens token examples load only layers `0..L` for target SAE layer `L`, matching the existing `stop_at_layer=L+1` forward. This is intentionally not a shard-derived approximation: pooled feature shards do not contain token-level activation vectors for `s128/p_max_pool` outputs. The full-corpus Gemma 3 4B PT l22/f65k preset opts into `token_examples.device=cuda`, `token_examples.activation_backend=native`, and `token_examples.model_dtype=bfloat16` for throughput; treat that as the validated fast path rather than the TransformerLens reference path. That bf16 path is tolerance-based, not byte/trial-equivalent to the CPU reference — see [Native bf16 tolerance contract](#native-bf16-tolerance-contract). For gemma-3 4b/12b/27b VLM checkpoints `native` is correct -- it routes through `steering.gemma3_loading.load_gemma3_text_decoder`, which correctly remaps the text weights (see [`gemma-scope-2.md` section VLM checkpoints](gemma-scope-2.md#vlm-checkpoints-4b12b27b--text-decoder-loading)); for all other models set `native` only for an explicitly validated fast-path run.
5. Deduplicate candidate examples by rendered source-window identity, drop any document excluded by `document_blacklist_path` (see below), split activating token windows into `n_quantiles` activation quantiles, and build trials with `n_positive_examples` activating examples plus one target-non-activating intruder.
6. Admit every target's trials globally in one canonical single-threaded pass, in strict (target, trial) order: cache hits are filled in place and misses become a fixed work list *before* any concurrency. Query the judge over that fixed work list -- serially at `judge.max_concurrent=1`, or via a bounded thread pool above it. The calling thread commits each successfully parsed result as soon as it observes that completion, then reassembles all outcomes by pre-assigned global order. Prompt/rendering errors, transport failures, parse exhaustion, and cache-write failures fail the run rather than producing a soft score row; successfully completed sibling trials remain cached for restart.
7. Write up to `audit_sample_limit` per-trial audit rows.
8. Recompute threshold outcomes and `top_k`, then write the final artifacts. The CLI applies `fail_on_empty` to its exit status after finalization.

## Fixed-Block Two-Stream Token-Example Materialization

`token_examples.materialization_batch_size` is a strict positive integer and
defaults to `128`. There is deliberately no `auto` mode. `1` selects the
unchanged single-target implementation and its legacy `token_examples_v6`
cache/manifest identity. Values above `1` select the fixed-block path; the
configured value is a feature-slot/cache block size, not the model batch
dimension.
The initial validated production contract is deliberately narrow:

- native CUDA on Gemma residual-post activations;
- context-independent, `ignore_bos_features=true`, fixed-token shards;
- singleton feature targets; and
- block size exactly `128` and GPU singleton concurrency exactly `2` under the
  runtime fingerprint embedded in the tolerance profile.

Any other real-provider configuration fails before model work with an explicit
instruction to set `materialization_batch_size=1`. An OOM also fails with the
block ID, slot, and true model length; it never retries lower concurrency and
never silently substitutes serial output. This means the generic CPU/`auto` config
must explicitly use `1`, as in the generic command above. The Gemma 3 4B
context-independent/no-BOS preset satisfies the batch contract and therefore
uses the repository default `128`.

### Deterministic blocks and resume behavior

Feature `fid` always maps to `block_id = fid // 128` and `slot = fid % 128`.
Each real slot runs a separate native forward with the exact serial input shape
`[1, L]`, where production `L <= segment_size + 1 = 129`; the optional leading
position is only the explicitly recorded injected BOS. At most two forwards
overlap on two persistent worker-local CUDA streams. Empty slots schedule no
model work. `PreparedBatch.token_tensor()` still assembles a padded CPU tensor
for immutable slot/probe assertions, but that diagnostic tensor is never copied
to CUDA or passed to the model. Finalization writes each true-length result back
to its canonical slot, removes only an explicitly injected BOS, then zeros
natural/interior BOS positions exactly as the serial path does.

The ordered target-plan SHA-256 is part of batch identity. Cache reads are
block-atomic: all requested targets in a block present is a hit; any missing
row causes the entire requested block to be rebuilt, ignoring partial hits.
The rebuilt block is serialized before SQLite work and written with one
`BEGIN IMMEDIATE` transaction. A SQL or process failure cannot publish a
partially new block. Every built row is also structurally validated before the
transaction begins, so a validation failure cannot publish bad rows. Logs are
emitted only after commit. Work runs in canonical
pairs with at most two live blocks: one CPU preparation worker alternates two
named lanes; a separate fixed two-worker executor owns the two CUDA streams;
and the main thread alone owns result consumption in slot order, state mutation,
validation, cache writes, and logging. Worker completion order cannot change
output placement or error precedence. A later block lane may finish first, but
the pair barrier commits the earlier block first. Before that worker exists,
the main thread also
predecodes the two possible synthetic leakage-probe token strings; the tokenizer
is therefore never called concurrently by the main and preparation lanes. CPU
prefetch is excluded from semantic identity and is required to be byte-exact
on/off. Benchmark-only pair timings partition the complete bulk call: each
block's namespace/cache lookup is charged to that block pair and one-time
setup/teardown is charged to the first, excluded pair.

### Runtime validation ownership

Parsed fixed-block, tolerance-profile, source-index request, token-window,
scoring-target, intruder-trial/result, score/report, and prompt-calibration
records store their reusable scalar and immutable-collection domains as nominal
`phantom-types` fields. Their direct constructors and helper entry points accept
ordinary builtin values through `*Like` annotations and enforce those
predicates with `beartype`: strict non-negative/positive/at-least-two integers,
exact binary and zero integers, non-empty strings, finite non-negative numbers,
unit-interval values, lowercase SHA-256 values, complete non-negative integer
and numeric tuples, singleton feature-ID tuples, and string tuples. `bool` is
rejected from every integer or numeric domain. Runtime container annotations
use deterministic full phantom predicates or unsubscripted outer containers,
so these checks do not sample elements or consume Python's global RNG.

Only relationships remain explicit: deterministic feature-to-slot placement,
row/token-slice agreement, token-window length/span parity, score/report count
ordering, tolerance ordering, support count bounds, target-ID upper bounds, CSR
shapes, and persisted artifact identity. Raw cache, prompt-config, manifest, and
feature-shard readers continue translating malformed external data to the
autointerp error hierarchy; invalid values passed directly to parsed runtime
APIs raise `BeartypeCallHintParamViolation` at the named argument.

### Serial anchor and validation gates

`CachedTokenExampleProvider.examples_for_target()` and
`TokenExampleSampler.build_trials()` remain the serial reference APIs. The
runner uses the optional bulk provider extension; third-party/fake providers
without it retain the serial fallback. The batch state machine advances each
target through exactly the serial positive-document order, cap boundary,
dedupe/sort/truncate transition, seeded intruder permutation, overlap filter,
and non-activating cap. Window construction is a shared pure function rather
than a second policy implementation.

Small raw activation drift caused by independent CUDA-stream scheduling is
allowed only inside a profile bound to the exact model/SAE/segment/runtime and
to a calibration-observation digest. Missing digests are legal only in the
explicit calibration harness; normal block execution rejects them. Startup
runs the same slot-addressed singleton block twice (candidate must be
bit-exact), compares representative injected/natural BOS and short/full rows against standalone
serial forwards at both the raw-residual and selected-SAE surfaces, and repeats
one held-fixed probe with changed companion rows to detect shared-state or slot
leakage.
A single-real-row block receives a valid synthetic companion slot; an
otherwise identical companion is explicitly perturbed, and runtime assertions
prove that the probe stayed fixed while some other row changed. The first four
batches, every 128th batch, and the final surfaced work are shadowed against
serial before the current block can commit. Shape, finite-value,
active-support, independently bounded max/p99/mean, and
companion-independence failures raise `AutoInterpBatchValidationError`;
candidate rows are never replaced by serial rows.

Validation against the base implementation uses 100 completed-reference
anchors chosen and frozen before any numeric results: 60 calibration and 40
held out. The serial producer may still be incomplete. Candidate
selection stratifies exact per-segment activation counts from
`feature_statistics.safetensors`; it rejects top-k-capped source-index row
counts as a fake density signal. Candidate
materialization includes every resolved target in each touched block as a
deterministic companion, while only the anchors require standalone reference
rows. Calibration and held-out runs use only their own touched blocks and
leave every opposite-split anchor slot dummy, preventing held-out observation
during profile fitting. The frozen JSON plan is content-hashed and strict-loaded so a growing
reference cache cannot change the held-out set between runs. Calibration
captures every anchor row at both selected-SAE and raw-residual surfaces and
also includes final-window/max metrics; filler startup probes do not widen the
profile. Acceptance requires exact window identity/count/order, active support,
highlights, quantiles, sampled source windows, and UTF-8 bytes from the
production `TokenWindowRenderer` → `render_trial_examples` →
`build_intruder_messages` path. Selected-SAE and raw-residual max/p99/mean
limits are independent. Raw floats may differ within profile, so trial cache
keys may differ even when rendered prompt bytes are exact.

### Document blacklist

Some source documents are fine as corpus text but adversarial for intruder-detection
prompts — for example, a document containing a stray numbered list (`7. Bellwether
constituencies...`) sitting next to a highlighted span can bias an LLM judge into
echoing the embedded number as if it were the requested example index instead of a
real answer, causing repeated parse-exhaustion failures for any target whose trial
happens to draw that document as an intruder (or, symmetrically, as a positive).

Set `document_blacklist_path` to a plain-text file excluding specific documents from
ever being sampled as a positive or intruder candidate, anywhere in the run. Format:
one `<shard_stem>:<doc_id>` entry per line (matching
`SourceDocRef`'s doc-level identity — shard filename plus row index — not a specific
token window or `TokenWindowRef.stable_key()`, which also bakes in the scored
target's feature ids and would need one entry per target ever tested against the
same document); `#` starts a comment, blank lines are skipped. `null` (the default)
disables filtering and preserves the no-blacklist sampling behavior.

Each runner parses the file once and snapshots the resulting unique document-key
set. Semantic identity hashes that canonical parsed set (sorted by shard stem and
document id), rather than the file's raw bytes. Reordering entries, adding comments,
or repeating a line therefore preserves identity, while adding or removing an
excluded document changes `document_blacklist_sha256`. The configured path and this
digest are both persisted in `runner_manifest.json`. The digest is always a lowercase
SHA-256 string: when the path is `null`, it hashes the canonical empty exclusion set
rather than becoming `null` itself.

The filter runs in `TokenExampleSampler.build_trials`, immediately after reading
`examples_for_target()` and before quantile grouping/sampling — downstream of any
`token_example_cache.sqlite` read, so it also takes effect against windows that were
cached before the blacklist existed (no cache-key change, no rebuild required). If
filtering removes every non-activating candidate for a target, or too many positive
quantile buckets, that target degrades gracefully to the existing
`insufficient_examples` outcome rather than a special-cased error. Because both
the path and the digest are persisted in `runner_manifest.json`, editing the
blacklist file in place before a rerun in the same `output_dir` invalidates the
prior compatible manifest instead of silently reusing stale filtering.

## Cache Identity

Serial token-example cache keys include source feature/data fingerprints,
model/SAE/backend/dtype/token-window settings, and the canonical singleton
target identity. Block keys use
`token_examples_v7_ci_block_two_stream_singleton_v1` and additionally bind the
block size, true-length singleton policy, fixed GPU concurrency, block/resume
algorithm, non-vacuous companion probe, ordered target-plan hash, all
source/window caps, validation profile, and runtime numeric fingerprint.
Prefetch scheduling is excluded. Batch size
one conditionally omits the new field from runner manifests and uses the legacy
v6 namespace, preserving compatible pre-batching caches and manifests.

Source-index manifests are fail-closed: if an index exists but its manifest does not exactly match the current feature/data fingerprints, target IDs, shard domain, source-index limits, extraction flags, schema version, and source-index generation `method`, the run aborts instead of silently reusing stale source refs.

Migration: the source-index `method` was bumped `source_doc_index_v1` → `source_doc_index_v2` when the build moved to the vectorized chunk-independent true-reduction scanner. On real (well-formed) feature shards a v1 index has byte-identical rows to v2 — every segment carries each feature at most once, so the legacy scanner's chunk-dependent divergence on duplicate-`(feature, stable_key)` entries cannot arise — but the method is pinned so any pre-existing v1 index is rejected with an "incompatible generation" error rather than reused. Delete the `source_doc_index.sqlite`/`.manifest.json` pair or set `source_index.force_rebuild=true` to rebuild with the current scanner. Token-example and trial caches are keyed on the resulting source refs (unchanged on real shards), so they remain valid across the rebuild.

Trial cache identity includes the canonical target, trial index, intruder
position, quantile, sampling/highlight settings, and each sampled window's
`token_window_v1` stable-key digest. That digest covers shard/doc/source hash,
token bounds, token IDs, target IDs, activations, and active role; it does not
cover `token_strings`. The parsed-result namespace additionally covers judge
model/CoT, prompt/few-shot names, their 16-hex SHA-256-derived content-version
fingerprint, and token-example render context. The rendered final user prompt
is not itself a cache-key field. Consequently, a change to token-string
decoding or renderer behavior must bump an enumerated method/version or render
context; the cache contract must not be described as hashing arbitrary prompt
bytes. As with every truncated or full cryptographic digest, identity also
assumes no relevant hash collision.

Candidate prompt windows are deduped before trial sampling by source position
and token IDs, excluding activation values, so the judge is not shown duplicate
options when two focused forwards produce the same rendered window. Trial
sampling also rejects overlapping same-document token windows across all
presented options. The generic LLM transport cache is bypassed for intruder
prompts, but each successfully parsed trial result stores the model response for
auditability. Transport failures and parse exhaustion abort loudly and are
retried on future runs.

`trial_audit.jsonl.cached` records whether the successful runner invocation read
that trial from its private parsed-result cache. It is execution provenance, not
part of the trial result's semantic identity. A failed invocation can populate
the cache before failing on a later trial, so a subsequent successful invocation
can legitimately change this flag without changing the prompt, response,
prediction, correctness, target score, accepted IDs, or ranking. Literal
audit-byte parity therefore requires identical cache history, not merely
identical cache keys.

Completion-time persistence deliberately permits a higher-global-index success
to be cached before a slower lower-global-index error is known. The outcome sort
still raises the lower canonical error after the fixed wave drains; cache timing
never selects the error. If the process is terminated externally, every row whose
individual SQLite commit completed is durable. In-flight judge calls and futures
that completed but were not yet observed by the calling thread are not claimed
durable; avoiding that residual window would require worker-thread or separate
writer ownership, which this single-threaded cache contract intentionally avoids.

`judge.max_prompt_chars`, `judge.parse_max_retries`,
`judge.requests_per_minute`, and `judge.max_concurrent` are not parsed-result
cache-key fields. The prompt-length limit is enforced only before an uncached
judge call, so an already cached, successfully parsed row may exceed the
current cap. The configured prompt-length cap and parse retries remain
semantic manifest fields. RPM and concurrency are operational provenance;
concurrency is excluded from manifest-compatibility comparison, while RPM
remains exact manifest identity.

## Native bf16 tolerance contract

The full-corpus preset runs token examples with `token_examples.device=cuda`, `activation_backend=native`, and `model_dtype=bfloat16`. This path is **throughput-oriented and tolerance-based — NOT byte- or trial-equivalent to the CPU TransformerLens reference path.** Because token-example caching is keyed on the model/backend/dtype settings, a bf16 cache and an f32/CPU cache never collide; the difference matters only when comparing a bf16 run against the CPU reference.

**The contract is structural, not per-token value-bounded.** bf16 drift on positive windows is *not* smooth rounding: near the SAE JumpReLU threshold a token that is large in f32/CPU can be gated to **exactly `0.0`** in bf16 (a `0 ↔ large` flip, per-token relative error 1.0), alongside a few-percent smooth rounding on the tokens that stay non-zero (measured on `feature:0`: one flip of ≈176 and ~5% smooth drift on the rest). A flip changes a window's `max_activation`, which reshuffles the top-k ranking and the trial identities. No per-element `atol`/`rtol` bounds a flip, so the gate asserts **structure** plus one hard value invariant. Measured consequences against the CPU reference (issue #678 PR A, PR #686):

- **Positive activations**: not bounded per token — flips and ~5% smooth drift both occur. The gate does not assert a per-token activation tolerance on positives.
- **Window identity sets**: mostly but not fully preserved (measured 20/25 positive sets, 24/25 non-activating sets identical; on `feature:0` the flip bumps one window, giving 15/16 matched). The gate requires ≥ 80% of reference windows still matched (`BF16_MIN_MATCHED_FRACTION`, a margin below the measured 0.9375 minimum so a re-measure does not flake).
- **Positive ordering, prompt sampling, and trial identity**: differ (order fraction down to 0.125; prompt windows and trial-cache keys change), because positives are ranked by `(-max_activation, stable_key())` and trial keys embed raw activation floats.
- **Highlights**: mostly exact (24/25 targets; worst-case Jaccard 0.5; gate floor 0.40).
- **Non-activating windows**: activations remain **exactly `0.0`** — a hard invariant enforced by the gate, since non-activating windows are all-zero on both paths; this catches any drift leaking into a non-activating window. Known blind spot: a purely-smooth uniform positive-activation drift that preserves all structure is not caught (a real scale bug also reshuffles `max_activation`s and trips the matched-fraction floor).

This contract is enforced by the test-only `assert_bf16_surface_within_tolerance` helper in `tests/steering/autointerp_filter/_parity_assertions.py` and exercised by the `bf16` gate tests in `tests/steering/autointerp_filter/test_token_example_cpu_cuda_tolerance.py`: a synthetic flip gate, structural-regression rejections, and — when the artifacts are present — the reduced `#660` bf16 cache and the 25-target production-shaped bf16 cache (which carries the real JumpReLU flip). bf16 is accepted as the production default because the issue #678 PR A benchmark (PR #686) measured it ≈11% faster at production window caps and using ≈42% less VRAM (6.86 vs 11.89 GiB) than f32, at the cost of this documented, bounded prompt/trial divergence from the CPU reference. A run that requires CPU-reference byte/trial equivalence must set `token_examples.model_dtype=float32` (still native CUDA) or `token_examples.activation_backend=transformerlens`.

## Concurrency Benchmark

The local proof harness exercises the real runner with synthetic token examples
and a delayed fake judge, so it never calls an external LLM:

```bash
uv run python -m scripts.benchmark.benchmark_autointerp_judge_concurrency \
  --n-targets 8 --n-trials 8 --max-concurrent 8 --delay-s 0.05
```

It asserts normalized scoring/audit artifact equality between serial and
concurrent runs, then prints elapsed time, speedup, calls, and observed peak
concurrency.

Each observed success performs an individual durable cache commit. That work is
normally hidden under network-bound judge latency, but a zero/near-zero-delay
fake judge can make SQLite the throughput ceiling; at that point more judge
threads may be neutral or slower while preserving identical artifacts. Size
`max_concurrent` with a representative judge delay/rate limit, not a zero-cost
fake, and treat durability rather than synthetic speedup as the invariant.

## Delphi Parity

Prompt construction and token highlighting are intentionally byte-compatible with EleutherAI Delphi's `IntruderScorer` prompt and `_prepare_text` renderer. The exact parity check is optional because Delphi is not a hard dependency of this repo:

```bash
DELPHI_REF_PATH=/path/to/delphi \
  uv run pytest tests/steering/autointerp_filter/test_delphi_reference_parity.py -q
```

## Outputs

| File | Purpose |
|---|---|
| `target_scores.jsonl` | One row per feature target with score, outcome, member ID, and optional Neuronpedia description. |
| `autointerp_ranked.yaml` | Concise human-readable ranking sorted by autointerp score descending, with description, feature IDs, score, and non-null errors. |
| `accepted_feature_ids.txt` | Sorted whitelist IDs for downstream `feature_mask`. |
| `manual_feature_filter.yaml` | Commented flat YAML list for manual review, sorted by score descending then feature ID; IDs are quoted and zero-padded to equal width, and partial-trial disclaimers appear after the Neuronpedia link. Uncomment accepted IDs before using it as a whitelist. |
| `manual_feature_filter.meta.json` | Provenance for the manual YAML; kept separate so the YAML remains a flat list. |
| `report.json` | Aggregate counts and output paths. |
| `trial_audit.jsonl` | Bounded detailed machine-readable sanity-check sample of token-level rendered trials and judge responses. |
| `trial_audit.yaml` | Concise human-readable audit sample, including `final_user_prompt_sent_to_judge`, rendered example text, max activations, prediction, correct index, and response. |
| `runner_manifest.json` | Run identity used to reject incompatible reruns in the same output directory; written only after scoring succeeds. |
| `source_doc_index.sqlite` / `.manifest.json` | Optional one-pass source-doc index when `source_index.enabled=true`. |
| `token_example_cache.sqlite` | Token-saving cache of re-forwarded token windows and token-aligned target activations. |
| `intruder_trial_cache.sqlite` | Token-saving cache of parsed trial results. |

Use the whitelist in training or eval configs:

```yaml
feature_mask:
  mode: whitelist
  path: out/autointerp/my_run/accepted_feature_ids.txt
```

The same two files also feed **AxBench-Base Stage 0a concept-candidate
selection**, not only `feature_mask`. Point
`concept_feature_whitelist_path` at either artifact to restrict which
features Stage 0a is allowed to consider as concepts, before the
Neuronpedia description fetch:

```yaml
concept_feature_whitelist_path: out/autointerp/my_run/accepted_feature_ids.txt
# or, after manually uncommenting the desired IDs:
# concept_feature_whitelist_path: out/autointerp/my_run/manual_feature_filter.yaml
```

Use the same SAE/layer feature-statistics domain that produced the whitelist.
Stage 0a raises if even one requested ID has no `per_feature_stats` entry; it
does not warn, drop the ID, and continue.

This is a distinct config surface from `feature_mask` —
`concept_feature_whitelist_path` bounds concept *candidacy* in the eval
pipeline (which features Stage 0a is even allowed to consider as
concepts, before the Neuronpedia description fetch), while `feature_mask`
— including AxBench-Base's own `conditioning.feature_mask` used for
Stage 4 encoder-decoder context-feature selection — bounds which
features *condition* the target model's generation at train/eval time.
See
[`docs/pipelines/axbench-base-eval.md`](axbench-base-eval.md) and
[`docs/reference/configuration.md`](../reference/configuration.md#concept_feature_whitelist_path--axbench-base-stage-0a-concept-whitelist).

For manual review, uncomment IDs in `manual_feature_filter.yaml`, keeping the quoted zero-padded ID strings intact, then point `feature_mask.path` at that YAML only after checking coverage:

```bash
uv run -m scripts.preprocess.compute_manual_filter_coverage -- \
  feature_dir=path/to/features \
  selection_yaml_path=out/autointerp/my_run/manual_feature_filter.yaml
```

An all-commented manual YAML means an empty selection and fast-paths through `feature_statistics.safetensors` without scanning shards, printing `0: n_segments`. Non-empty selections stream the stats-domain shards (`exclude_last_shard=true` by default) and print a selected-feature-count histogram plus nearest-rank `p0,p1,p5,p10,p50,p90,p95,p99,p100`.

The filter never rewrites base feature shards, statistics, percentile overlays, token boundaries, packed rows, or selection shards.

## Concurrency Regression Suite

`scripts/benchmark/benchmark_autointerp_concurrency.py` is a second,
single-process proof harness: it runs the real `AutoInterpFilterRunner` twice
over the SAME deterministic synthetic workload -- once serially
(`max_concurrent=1`) and once concurrently (`max_concurrent=N`) -- against a
fake judge that `time.sleep(delay_seconds)` per call (modelling a
rate-limited network judge). It asserts normalized artifact equality and
prints the observed wall-time speedup. It never calls OpenRouter, and it
replaces the deleted process-sharding benchmark
(`benchmark_autointerp_target_sharding.py`):

```bash
python -m scripts.benchmark.benchmark_autointerp_concurrency \
  --n-targets 12 --n-trials 8 --max-concurrent 8 --delay-seconds 0.05
```

`scripts/benchmark/measure_autointerp_model_vram.py` loads the real
production model/SAE stack once (the `gemma3_4bpt_l22_f65k_ci_nobos` config)
and records its peak/resident VRAM, backing the single-resident-model design:
in-process concurrency shares one loaded model across every target/trial
instead of loading a separate copy per shard worker, as the deleted
process-sharding design required.

```bash
PYTHONPATH=. uv run python -m scripts.benchmark.measure_autointerp_model_vram
```

The focused CPU regression ladder for serial/concurrent byte-identity is:

```bash
uv run pytest -q \
  tests/steering/autointerp_filter/test_serial_golden.py \
  tests/steering/autointerp_filter/test_concurrency_parity.py \
  tests/steering/autointerp_filter/test_concurrency_benchmark.py \
  tests/steering/autointerp_filter/test_artifacts.py \
  tests/steering/autointerp_filter/test_source_index.py \
  tests/steering/autointerp_filter/test_path_safety.py
```

`test_concurrency_parity.py` pins the byte-identity guarantee: serial
(`max_concurrent=1`) and concurrent (`max_concurrent=8` or `16`) runs over the
same targets produce byte-identical artifacts, including under a mid-run
judge-call budget cap, under concurrent trial-cache access, and when several
trials error (the lowest-`global_index` error always wins, matching the
serial run, regardless of wall-clock completion order). Real-data and
real-GPU parity coverage lives in `test_token_example_cpu_cuda_tolerance.py`
and the real-shard tests in `test_source_index.py`, gated behind
`@pytest.mark.requires_feature_data` / `@pytest.mark.slow` and an
artifact-root environment variable; these skip when their prerequisite
artifacts are absent.

Before making a real performance claim, run serial and concurrent
configurations over the same full-size feature/parquet inputs on a quiet box.
Pin deterministic judge responses and identical cache warmth: `cached` is a
persisted audit field, so either hold it equal or explicitly normalize that
documented execution-provenance field. Use the same real model/SAE backend.
Disable Neuronpedia or freeze its descriptions in a common cache/snapshot,
because live description changes alter final artifacts. Compare normalized
final scores, rankings, accepted IDs, manual YAML, reports, and audit
prefixes exactly (excluding only explicitly documented timestamps), then
measure wall time and peak CPU/GPU memory only after parity passes and after
checking that unrelated processes are not contending for the box.

## Source-Index Memory Benchmark

Profile the source-index scanner without judge calls:

```bash
uv run python -m scripts.benchmark.benchmark_autointerp_source_index_memory \
  --shard-path path/to/shard_00000_features.safetensors.zst \
  --n-sae-features 65536 \
  --blacklist-file path/to/blacklist.txt \
  --target-limit 32
```

The benchmark invokes the current production scanner directly and reports
`entries` (CSR nonzeros), `entries_per_second`,
`positive_rows_per_second`, the row/ref hash, and peak RSS. Add
`--assert-exact` on a shard with at most 2,000,000 entries to compare its rows
against an independent entry-by-entry brute-force true-reduction oracle. The
bound keeps the intentionally slow Python oracle out of large real-shard runs.
`--synthetic N --temp-dir path` benchmarks a deterministic production-shaped
shard and provides a seconds-fast exactness regression without touching real
data. In the production scanner's progress events, per-chunk `rows_emitted` counts
worst-filter survivors and `bucket_rows` counts retained state plus
pending rows awaiting a batched merge; batching is a scheduling detail
that tests prove cannot change output.

During a real `source_index.enabled=true` run, the builder logs structured
progress at shard start, after each shard is spooled, after each decoded chunk,
and after the parent merge completes for the shard. These
lines include compressed shard size, uncompressed spool size, decoded entry/row
counts, entry throughput, emitted source-ref rows, retained per-shard bucket
rows, live parent survivor rows, SQLite temp-index size, parent wait/merge
timings in parallel mode, and process RSS. The default test suite includes
spawned-worker byte-identity and worker-progress regressions for
`scan_workers=1` vs `2`; broader worker-count and forced out-of-order checks
stay under the slow marker. The same
progress callback feeds the benchmark JSON fields `spooled_event`,
`chunk_event_count`, and `last_chunk_event`, so a short benchmark run can be
compared directly with the live log when diagnosing slow full-corpus runs.

## Token-Example Materialization Benchmark

Profile the token-example/trial path (issue #678) without judge calls, against a
completed source index:

```bash
uv run python -m scripts.benchmark.benchmark_autointerp_token_examples \
  --output-dir out/benchmarks/issue678/run_tag \
  --report out/benchmarks/issue678/run_tag.json \
  --feature-dir /abs/path/to/features --data-dir /abs/path/to/base_data \
  --source-index-path /abs/path/to/source_doc_index.sqlite \
  --source-index-manifest-path /abs/path/to/source_doc_index.manifest.json \
  --targets 0 \
  --instrumentation off  # or --select-targets 25 (deterministic stratified set)
```

The harness drives the real provider/sampler (token-cache misses enforced; no
LLM judge, no network) and emits per-phase inclusive/exclusive timings
(`source_index_load`, `sqlite_row_load`/`sqlite_apply`, `parquet_text_load`,
`tokenization`, `token_decode`, `model_forward`, `sae_projection`,
`window_selection`, `stable_key_hash`, `token_cache_open`/`token_cache_write`,
`trial_build`), peak RSS and CUDA allocated/reserved peaks, per-target
payload/namespace hashes, and — with `--reference-cache` — token-example,
trial-source, and exact production-rendered-message surface reports from
`steering/eval/autointerp_filter/parity.py`. Instrumentation is installed only
inside the benchmark process and is asserted restorable;
`--instrumentation off` is the headline throughput mode and performs no wrapper
calibration or per-phase CUDA synchronization. `tier1` and `tier2` are
diagnostic modes; tier 2 adds the hot `stable_key`/decode wrappers and reports
the calibrated per-call wrapper overhead. `--targets` names the reference and
parity anchors; `--materialization-targets` may supply a larger deterministic
block-filler superset for the block/two-stream candidate. Production
inputs are treated as read-only (index/manifest stats are asserted unchanged).
Run one path-variant per process so `ru_maxrss` and CUDA peaks stay
attributable, and compare `payload_sha256` across reruns before comparing
timings. Always use a FRESH `--output-dir` per variant. Serial v6 namespaces
retain their legacy identity (where the runner manifest guards cap changes);
block v7 namespaces bind every source/window cap directly. The harness bypasses
runner finalization and refuses any pre-existing output token cache regardless.
Every flat `*_elapsed_s` report field
is an INCLUSIVE phase time — they overlap (`examples_elapsed_s` contains most
other phases) and must not be summed; use `phases[*].exclusive_s` for
non-overlapping breakdowns.

### Measured result: per-row-group source-text read

Profiling with this harness (native f32, preset window caps, completed source
index, token-cache miss) showed `parquet_text_load` dominating cold-run
materialization at ~78% of `examples_elapsed_s`. The cause was `_full_text`
caching whole shard text columns (~240 MiB / ~53k docs each) in a two-entry LRU
that thrashed across the 370-shard corpus. The token-example path needs only a
handful of docs per shard, so it now reads only the parquet **row group**
holding a doc (~1024 rows, ~1/52 the I/O) with a memory-bounded per-row-group
cache (`_load_shard_row_group` / `_shard_row_group_offsets` in
`steering/eval/autointerp_filter/token_examples.py`). This is **byte-identical**
— every doc resolves to the exact same text (the whole-shard byte-identity
oracle lives in `tests/steering/autointerp_filter/_token_example_reference.py`).

End-to-end on the real corpus (baseline vs optimized, same box, 5 stratified
targets, preset caps, native f32, tier 1):

| metric | baseline (LRU-2 whole-shard) | optimized (row-group) |
| --- | ---: | ---: |
| `examples_elapsed_s` | 483.6 | 82.4 (**5.9×**) |
| `parquet_text_load` | 376.6 | 6.6 (**57×**) |
| `source_index_load` (shared) | 33.6 | 31.5 |
| peak RSS | 23.2 GiB | 23.1 GiB |
| parquet share of runtime | 78% | 8% |

All per-target `payload_sha256` values are identical between the baseline and
optimized runs, so the change is a pure speedup with no output difference.
`examples_elapsed_s` varies with GPU/box state across runs; the
`parquet_text_load` reduction (≈377 s → ≈7 s) is the stable, optimization-
attributable figure.
