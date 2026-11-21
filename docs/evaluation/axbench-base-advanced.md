# AxBench-Base Internals

Developer-facing reference for the `steering/eval/axbench_base/` package.
Every stage is walked through end-to-end with file:line references;
every major design decision gets its rationale. Read alongside
[`docs/evaluation/axbench-base.md`](axbench-base.md) (user guide) and
[`docs/pipelines/feature-extraction.md`](../pipelines/feature-extraction.md)
(prerequisite pipeline).

## Background and research question

AxBench ([Wu et al., arXiv:2501.17148](https://arxiv.org/abs/2501.17148))
compared steering methods — SAE-feature-based interventions, diff-of-means
directions, linear probes, LoRA fine-tunes, and prompting — on
instruction-tuned chat models. They found classical baselines
(DiffMean, LoRA) outperform the SAE-based methods on concept relevance
despite the SAEs' interpretability advantage.

AxBench-Base restates the same question for **base** (next-token) language
models. There is no dialog envelope; every completion is an in-context
continuation of a seed prefix. The interesting research axis is whether
our conditional encoder-decoder GPT variant — which injects SAE features
via cross-attention rather than by hand-chosen residual-stream deltas —
can match or exceed the classical steering baselines on a base LM target.

The evaluation concept pool is drawn from GemmaScope (gemma-2-2b residual
stream, 16k features), filtered by corpus count, described via
Neuronpedia, and optionally validated by an LLM judge against the
feature's top activating examples. The scoring rubric mirrors AxBench's
0-2 judge scales for concept relevance and fluency, but is applied to
raw next-token completions rather than chat turns.

## Package layout

```
steering/eval/axbench_base/
|-- __init__.py                    # empty; everything imported from submodules
|-- config.py                      # AxBenchBaseConfig + nested dataclasses + validate_config
|-- config_loader.py               # cfg_to_dataclass: DictConfig -> AxBenchBaseConfig
|-- types.py                       # CandidateConcept, Concept, SeedRow, TargetModelMetadata, Stage, SteeringMode, AxBenchBaseError
|-- paths.py                       # run_tag -> output path resolvers, sanitization
|-- cuda_memory.py                 # gc + CUDA allocator release helper
|-- templates.py                   # _safe_substitute + load_template (literal {{KEY}} replace)
|-- validate.py                    # concept/candidate stage-boundary validators
|-- seeds.py                       # deterministic sampling seed derivation
|-- seed_pools.py                  # Stage 0b: DDP exact-prefix extraction + commit
|-- seed_contract.py               # schema-v2 CSR/manifest identity + strict loading
|-- stage4_prompts.py              # shared owned-BOS prompt construction/keying
|-- stage4_contexts.py             # Stage 3e enumerate/validate exact contexts
|-- concept_list.py                # Stage 0a: prepare_concept_list pre-QC pool
|-- concept_whitelist.py           # optional file-backed Stage 0a candidate whitelist
|-- concept_examples_qc.py         # Stage 0c lazy sub-step: Neuronpedia + LLM judge
|-- negative_prefixes.py           # Stage 0c: prepare_negative_prefixes + _VerifiedNegativePrefix
|-- stage0c_api_generation.py      # optional evidence-grounded OpenRouter continuation source + strict parser
|-- prefix_text.py                 # shared decoded-character bounds + tail window
|-- stage0c_runner.py              # Stage 0c rank orchestration + conditional generator wiring
|-- stage0c_partition.py           # partial checkpoint I/O + deterministic merge
|-- _stage0c_internals.py          # candidate scan/SAE verification/fingerprint internals
|-- synth/continuation_guards.py   # shared envelope, suffix-echo, copy, and phrase-loop primitives
|-- synth/continuation_protocol.py # authenticated Stage 1b/1c request/response protocol
|-- synth_continuation_contract.py # versioned Stage 1b/1c parquet provenance
|-- concept_feature_mapping.py     # Stage 2: canonical + delta_ranked
|-- extraction.py                  # SAE extraction helpers + percentile table + bucket boundaries
|-- coefficient_grids.py           # Stage 4 sweep grid factories
|-- method_registry.py             # Stage 4 method registry (33 entries)
|-- encoder_input.py               # Stage 4 build_mode_aware_encoder_input
|-- generation.py                  # Stage 4: generate_all orchestration
|-- judge.py                       # Stage 5: AxBenchJudgeClient + judge_all
|-- aggregation.py                 # Stage 6: aggregate + leaderboard writer
|-- parquet_writer.py              # ParquetAppendWriter (directory-dataset appender)
|-- dag.py                         # STAGE_DEPENDENCIES + check_stage_dependencies
|-- runtime_state.py               # RuntimeState + validate_env_for_stages
|-- tokenizer_cache.py             # process-local nanochat tokenizer cache for Stage 1
|-- llm_judge_client.py            # LLMJudgeClient base + OpenRouterLLMClient (sync)
|-- openrouter.py                  # OPENROUTER_API_KEY_ENV constant + shared client helpers
|-- model_detection.py             # detect_target_model_metadata
|-- _stats_reader.py               # read_n_sae_features_from_stats (safetensors header)
|-- prompts/                       # template files (.txt)
|   |-- judge_concept_examples.txt
|   |-- judge_concept_relevance.txt
|   |-- judge_fluency.txt
|   |-- stage0c_infer_concept_definition.txt
|   |-- stage0c_continue_with_concept.txt
|   |-- synth_continue_with_concept.txt
|   `-- synth_continue_avoiding_concept.txt
|-- synth/
|   |-- __init__.py                # empty
|   |-- positive_mode.py           # Stage 1b: synthesize_positive_mode
|   |-- positive_checkpoint.py     # Stage 1b strict partial-checkpoint IO
|   |-- negative_mode.py           # Stage 1c: synthesize_negative_mode
|   |-- qc.py                      # exact prefix/output percentile and strict-zero QC
|   |-- openrouter_client.py       # OpenRouterSynthClient (single-prompt wrapper)
|   |-- llm_response_cache.py      # SynthResponseCache (SQLite) + make_cache_key
|   `-- response_parser.py         # extraction-tokenizer loader + legacy raw parser
`-- baselines/
    |-- diffmean.py                # Stage 3a
    |-- linear_probe.py            # Stage 3b
    |-- prompting.py               # Stage 3c
    |-- hooks.py                   # shared residual-stream hook helper
    |-- _capture.py                # shared activation-capture utility
    `-- lora/
        |-- module.py              # hand-rolled LoRA linear
        |-- training.py            # Stage 3d: train_lora_all
        |-- loader.py              # adapter file -> module
        `-- _attach.py             # attach/detach refcount helpers
```

The `__init__.py` files are empty — consumers import from the specific
submodule. This keeps the import graph shallow and makes circular-dep
bugs impossible at the package boundary.

## Data types

All core dataclasses are `@dataclass(frozen=True)`. Two of them have
`__post_init__` guards that enforce invariants at construction time
(`SeedRow` below, `Concept` enforcement lives in `validate_concepts` —
dataclass type annotations are not runtime-checked).

### `AxBenchBaseError`

Single exception class for every user-visible error in the package
(`types.py:10-12`). Every call site raises this rather than
`ValueError` / `RuntimeError`, so callers can `except AxBenchBaseError`
once and handle everything. Private helpers wrap third-party exceptions
(`TypeError`, `ValueError`, `requests.RequestException`) before
re-raising.

### `Concept`

```python
@dataclass(frozen=True)
class Concept:
    concept_id: int
    feature_index: int
    layer: int
    sae_source: str
    description: str
    corpus_count: int
    description_source: str
```

`types.py:100-118`. Serialized to `concepts.json` via `dataclasses.asdict`.
Restored via `Concept(**item)` inside `load_and_validate_concepts`
(`validate.py:65-99`), which catches `TypeError` on missing/unexpected
fields and re-raises as `AxBenchBaseError` so the "concepts.json is
corrupt" case has a single failure mode.

The dataclass has **no** `__post_init__`. Invariant checking
(contiguous `concept_id`, no duplicate `feature_index`,
`feature_index in [0, n_sae_features)`, non-empty description) is
centralized in `validate_concepts` (`validate.py:22-62`) which runs
AFTER deserialization. Rationale: we sometimes need to build partial
concept lists mid-pipeline (e.g., in tests) and having the class itself
refuse unusual constructions would be painful.

`concept_id` is always **dense** (`0, 1, 2, ...`) in immutable Stage 0c
`concepts.json`. Stage 1b writes an ordered filtered subsequence to
`concepts_stage1b.json`; Stage 1c may write a further subsequence to
`concepts_stage1c.json`. Those filtered lists preserve the original IDs and may
therefore be sparse. The last enabled Stage 1 list, not corpus presence, is the
single trainable/scorable population authority for all downstream artifacts.

### `SeedRow`

```python
@dataclass(frozen=True)
class SeedRow:
    shard_idx: int
    doc_idx: int
    source_nanochat_token_count: int
    prefix_source_token_start: int
    prefix_source_token_end: int
    prefix_text: str
    prefix_content_nanochat_tokens: list[int]  # exact BOS-free source slice
    sae_feature_ids: list[int]                 # sorted, unique, max-pooled
    sae_feature_mags: list[float]
    continuation_text: str
    continuation_content_nanochat_tokens: list[int]  # exact M-token slice
```

This is the intentional schema-v2 contract. Stored prefix and continuation
tokens are authoritative nanochat source slices and never include the
pipeline-owned BOS. Character offsets and SAE segment character ranges were
removed because they mixed tokenizer coordinate systems and had no production
consumer.

Construction checks non-negative provenance, non-empty content slices, the
source-count and prefix-bound equations, non-negative token IDs, and basic SAE
CSR structure. The schema-v2 loader adds config-dependent checks: exact `M`,
prefix cap, exact decode-to-text, SAE-width range, sorted unique IDs, and finite
strictly-positive magnitudes. Train/eval metadata and
`seed_pool_manifest.json` bind both files to the current config, tokenizers,
source parquets, extractor, model/SAE tensor hashes, runtime, and batching.

`sae_feature_ids` is sorted ascending; `sae_feature_mags` follows that
order. Feature magnitudes are **max-pooled** across the exact prefix's
independently extracted production segments — if the same feature fires on multiple segments only its
maximum magnitude survives. This matches the "feature set" semantics
the downstream synth filter relies on: Stage 1b classifies a scalar-source
seed as stored-clean when `f_c` is absent from `seed.sae_feature_ids` (or, for
a clustered source, when every allowed member is absent). This stored
classification is only a selection priority; canonical prefix-alone scoring
and the headroom ceiling decide actual eligibility. Duplicate feature IDs
would violate both interpretations.

### `TargetModelMetadata`

```python
@dataclass(frozen=True)
class TargetModelMetadata:
    num_buckets: int | None
    n_sae_features: int | None
    negative_selection: dict | None
    feature_selection: dict | None
    per_feature_stats: dict | None
    device: str
    sae_layer: int
    segment_size: int
```

`types.py:121-168`. Produced once per run by `detect_target_model_metadata`
(`model_detection.py:37-93`). Every encoder-specific field is `None` for
decoder-only targets; `sae_layer` and `segment_size` are ALWAYS populated
(parsed from `cfg.feature_dir`).

Rationale for decoupling `sae_layer` from the target model: there used to
be a hardcoded `_SAE_LAYER = 8` constant in `extraction.py`. Changing
SAEs meant editing source. Now the feature_dir path encodes `l{layer}` via
the standard `{root}/s{seg_size}/l{layer}/...` convention
(`feature_paths.py:71-118`), so swapping to a different layer is a config
change.

Every stage that runs Gemma + SAE takes `sae_layer` as a keyword-only
argument. Most runtime stages get it from cached `TargetModelMetadata`
via `RuntimeState.decoder_only_meta()` / `encdec_meta()`
(`runtime_state.py`, `decoder_only_meta` / `encdec_meta` methods).
Stage 0c is the exception: it constructs a lightweight metadata object
directly from `cfg.feature_dir` so negative-prefix verification does not
briefly load either target model. Synthesis stages take the full
`TargetModelMetadata` and read `.sae_layer` off it at the call site
(`positive_mode.py`, `_run_positive_target_refill_engine`; `negative_mode.py`,
`_synthesize_concept_negatives` /
`_synthesize_all_concept_negatives_parallel`).

## The feature_dir convention

`parse_feature_dir(Path(cfg.feature_dir))` (in `steering/feature_paths.py`)
walks the **last 7** path components and extracts:

- `segment_size` from `s{int}`
- `layer` from `l{int}`
- `activation_type` from `a_{label}` (mapped to `ActivationType` enum)
- `pooling` from `p_{label}` (mapped to `PoolingMode`)
- `width` from `f{label}` (mapped to `SAEWidth`)
- `model` from `m_{name}`
- `dataset_phase` from the last component (`DatasetPhase` enum)

Any leading path prefix is ignored, so both relative (`s128/l8/...`) and
absolute (`/data/features/s128/l8/...`) paths work. Malformed components
raise `ValueError`.

`axbench_base` reads `layer` and `segment_size` from this parse
(`model_detection.py:58-60`) and trusts them as ground truth for
downstream Gemma + SAE calls. `concept_list.py:99` also reads
`layer` here to populate `Concept.layer`. Do **not** hardcode these in
new code; always plumb them through `TargetModelMetadata` or a
parameter.

See `docs/pipelines/feature-extraction.md` for the full feature-extraction
pipeline that produces directories in this layout.

## Stage 0: Preparation

The prep stages produce three artifacts: a concept list (what are we
steering?), two seed pools (what prefixes do we evaluate on?), and a
set of verified negative-mode prefixes (where is the concept already
active?).

### Stage 0a: `prepare_concept_list`

`concept_list.py:83-195`. Builds `candidate_concepts.json`.

Step by step:

1. **Parse feature_dir** to extract `layer` via
   `parse_feature_dir(Path(cfg.feature_dir))["layer"]`
   (`concept_list.py:98-99`). This becomes `CandidateConcept.layer` on
   every emitted candidate row.
2. **Load per-feature stats** via `_load_feature_statistics`
   (`concept_list.py:30-32`). Thin wrapper around `load_statistics` so
   tests can monkey-patch it without mocking the safetensors loader.
3. **Filter by corpus count**: keep features with
   `fstat.count >= cfg.min_corpus_count` (`concept_list.py:109-113`).
   If the filtered list is empty, raise `AxBenchBaseError`.
4. **Fetch Neuronpedia descriptions** via `NeuronpediaClient.get_description`
   (`concept_list.py:120-134`). Failures are logged at DEBUG (not
   WARNING) and counted; individual fetch errors do not halt the stage.
   Descriptions shorter than `cfg.min_description_length` are dropped
   silently.
5. **Shuffle deterministically** with `cfg.concept_list_seed`. This is
   the only ordering step in Stage 0a.
6. **Do not truncate** to `cfg.max_concepts`. Stage 0c needs the full
   pre-QC pool so LLM- or verification-rejected candidates can be
   replaced.
7. **Build `CandidateConcept` rows** with no `concept_id` and no judge
   score. Stage 0c assigns temporary post-LLM-QC IDs for verification
   RNG and final contiguous `Concept.concept_id` values for accepted
   concepts only.
8. **Validate** via `validate_candidate_concepts(...)`. Catches
   invariant violations one last time.
9. **Write `candidate_concepts.json`** using
    `json.dumps(..., ensure_ascii=False)` so Unicode descriptions
    round-trip correctly. Stage 0a does not write `concepts.json`.

**Why deterministic shuffle.** The SAE feature index space is stable but
candidate ordering is not — different `min_corpus_count` thresholds
return different subsets, and Stage 0c consumes the shuffled candidate
pool in order. Pinning the shuffle seed means dev-tier runs can be
diff-compared across config tweaks.

**Why no Stage 0a LLM QC or cap.** Stage 0c usually needs only the first
few candidates that pass both examples QC and negative-prefix
verification. Keeping Stage 0a to filtering + shuffling preserves
resume semantics while avoiding eager LLM calls for the unused tail of
the candidate pool.

**Concept-examples QC rationale.** Stage 0c performs this lazily before
verification. Neuronpedia's human-readable
descriptions are of variable quality — some are auto-generated, some
are written by researchers, some are empty stubs. Feeding a bad
description into Stage 1b's synth prompt ("continue this text with
elements related to '{{DESCRIPTION}}'") produces bad continuations,
which then poison the downstream QC and any trained baselines. The QC
step catches this before the expensive synth calls.

The judge sees the description plus the top-`n_activation_examples` (5)
high-activation examples for the feature, formatted via
`judge_concept_examples.txt`. It rates the alignment on a 0-2 scale
("not a match" / "partial or weak" / "strong"). The prompt (and thus the
rubric) is AxBench's own judge format; see
`concept_examples_qc.py:398-458` for the parser.

`ActivatingExample.text` is decoded via
`tokenizer.convert_tokens_to_string(tokens)`, **not** `"".join(tokens)`,
because Neuronpedia returns raw tokenizer-specific tokens
(SentencePiece `▁`, BPE `Ġ`, byte-fallback `<0x0A>`). Joining them
directly leaks these markers into the judge prompt and corrupts the
comparison; see the Review-fix G note at
`concept_examples_qc.py:17-25`.

**Crash recovery.** Each concept-examples QC call writes to the
`SynthResponseCache` immediately (`llm_response_cache.py:51-56`
commits per-write), so a crash midway through Stage 0c leaves prior
responses queryable on re-run. The cache key is
`sha256("concept_examples_qc" || prompt || f"concept_qc_{desc[:40]}")`
(`concept_examples_qc.py:432-436`), which is stable across runs of the
same config.

When `verification_api_model` is active, the same SQLite cache also stores the
validated/deduplicated raw activation-evidence payload, every definition
attempt, and every continuation attempt. Each write commits before the next
unit of work. Cache identities include the feature/evidence identity,
candidate source coordinates, prefix hash, stable verification seed, content
attempt, selected API model, and full rendered prompt. A rerun therefore
replays completed external calls, while the rank-local Stage 0c checkpoint
controls which candidates must be revisited. A corrupt cached evidence payload
fails validation instead of silently refetching or changing the definition.

### Stage 0b: `prepare_seed_pools`

Produces the schema-v2 `seeds_train.safetensors`,
`seeds_eval.safetensors`, and `seed_pool_manifest.json` commit marker.
Stage 0b is rank-aware and runs inside the same loaded base-model + SAE
lifetime used by Stage 0c.

**Exact raw-source slicing.** Each configured validation parquet is streamed
in deterministic `(validation_shards order, document row)` order. The source
text is encoded by nanochat with no BOS. For raw content length `N`, continuation
length `M`, and prefix cap `C`:

```python
eligible = N >= 2 * M
continuation = source_tokens[N - M : N]
prefix_end = N - M
prefix_start = max(0, prefix_end - C)
prefix = source_tokens[prefix_start:prefix_end]
```

The two stored token lists are those exact source IDs. They are decoded once
to produce `prefix_text` and `continuation_text`; decode/re-encode is never used
to reconstruct tokens. This matters for lossy tokenizer seams, literal
special-token-looking text, and short documents: eligibility is based only on
raw content, never on a synthetic BOS.

**BOS ownership.** Stored content is BOS-free. At a model boundary,
`prepend_owned_bos(tokenizer, content_tokens)` unconditionally creates
`[bos_id, *content_tokens]`. If the first payload token also equals `bos_id`, it
is retained. Ownership is positional and construction-based, not inferred from
token values.

**Independent prefix feature extraction.** Before assigning a candidate index,
Stage 0b tokenizes `prefix_text` through the loaded extraction model exactly as
production extraction does, with context limit
`min(cfg.extraction_max_tokens, model.cfg.n_ctx)`. Empty, over-limit, or
entirely unpoolable prefixes are counted and skipped; `process_shard` is never
allowed to head-truncate a seed. Literal special-token text is preserved, while
the `ignore_bos_features` setting parsed from `feature_dir` masks every Gemma
BOS-token position just as the producer does.

Candidates are materialized in explicit-schema parquet shards and passed to
the unchanged production `steering.extract_features.process_shard` with parsed
layer, activation, pooling, segment size, phase, context-independence, BOS
masking, fixed-token segmentation, resolved hook/L0/backend, and the exact
context window. Stage 0b never reads the original validation feature shard:
those segments describe the source document's head or include context outside
the selected tail prefix and cannot be aligned by an overlap predicate. For
each candidate, feature magnitudes are max-aggregated across its independently
extracted segments and feature IDs are sorted; segment coordinates are not
persisted.

**DDP, determinism, and resume.** Every rank independently computes the same
source hashes, skip counters, candidates, and canonical digest, then must agree
before extraction. Candidate shard identity is independent of rank/world size.
Collective waves assign shard `wave_start + rank`, and gather a success/error
record after each wave so an ordinary Python error cannot desynchronize peers.
Before the full job, ranks hash the actual forward model and SAE tensors,
compare resolved runtime/backend/tokenizer identities, and require identical
digests for a fixed canary batch.

Each committed intermediate feature shard has a sidecar binding candidate
parquet SHA-256, extraction fingerprint, document count, decompressed output
digest, and runtime identity. Data plus sidecar must both validate to resume;
mismatched committed work fails loudly, while uniquely named incomplete
temporaries are ignored. Resume is safe after a world-size change because
shards are not rank-named.

**Final commit.** Rank 0 validates every shard, contiguous unique candidate
indices, exact document token counts, absence of truncation, and one result per
accepted candidate. It shuffles once with `random.Random(seed_pool_seed)` and
splits at `floor(n * seed_pool_train_ratio)`. Both safetensors files are written
atomically with schema-v2 metadata. `seed_pool_manifest.json` is written last
and is the sole commit marker. Loaders require the expected split and current
config expectation and reject missing/torn files, bad hashes, legacy schemas,
and metadata drift with a fresh-run-tag error; there is no ownership migration.

### Stage 0c: `prepare_negative_prefixes`

`negative_prefixes.py:425-491`. Produces `concepts.json`,
`rejected_concepts.json`, and `negative_mode_prefixes.parquet`.

This prep stage scans validation shards for segments where the concept feature
is active, generates several continuations for each candidate prefix, and
locally SAE-scores those continuations. By default the continuation source is
`inclusionai/ling-2.6-flash` through OpenRouter. Setting
`verification_api_model=null` selects the exact schema-8 **base SAE LM** path,
where generation and activation measurement share the SAE's native tokenizer,
context window, and activation distribution. The generation source is the
only branch: candidate discovery, Gemma-token parsing, local SAE percentile
scoring, consistency acceptance, downstream rows, and stop conditions remain
common.

Pipeline per candidate concept:

1. **Lazy examples QC.** If `enable_concept_examples_qc` is true, build
   the shared `ConceptExamplesQC` client with the same wiring as the
   old eager Stage 0a path:
   `OpenRouterLLMClient` wrapped in `SynthResponseCache` at
   `AxBenchBasePaths.from_config(cfg).synth_response_cache()`, the same prompt template,
   and the same Neuronpedia activation-example fetch. If the judge score
   is below `activation_example_judge_min_score`, or the fetch/judge
   fails, record the LLM rejection reason and skip verification.
2. **Stable provisional verification ID.** For verification, assign a
   provisional `concept_id` equal to the candidate's index in the ordered
   Stage 0a pool. Candidate index is stable across rank ownership and world
   size, so single- and multi-rank runs shuffle prefixes and derive generation
   seeds from the same identity. The final dense `concept_id` is assigned only
   after merge, when accepted candidates are sorted and truncated.
3. **Candidate scan.** For each validation shard, load the feature
   shard, iterate segments, and keep those where the concept feature
   `f_c` appears in the segment's feature list
   (`_find_candidate_prefixes`, `negative_prefixes.py:200-255`). For
   each matching segment, slice the parquet document text up to the
   segment's `char_end` as the candidate prefix, clamping to
   `[cfg.min_prefix_chars, cfg.max_prefix_chars]`. Prefixes are
   tokenized with the extraction/Gemma tokenizer
   (`cfg.extraction_tokenizer_id`, with special tokens). Any prefix
   longer than `base_model.cfg.n_ctx - cfg.verification_length` Gemma
   tokens is dropped during scanning, before verification generation.

   `_CandidatePrefixIndex` keeps only fixed run state: decoded validation-shard
   tensors, parquet document texts, the extraction tokenizer, immutable
   configuration, and heartbeat counters. Each feature lookup computes its CSR
   segment matches into a local array and returns a freshly materialized,
   concept-owned candidate list. Ordinary lookup therefore retains no prior
   concept's copied prefix strings, source bridges, or token lists. Clustered
   lookup preserves member order, canonical location deduplication, the shared
   per-shard cap, and final sort while keeping only the active concept's member
   union live. Candidate memory is consequently bounded by the largest active
   concept, in addition to the fixed decoded shards, rather than by all
   previously visited concepts.
4. **Shuffle** the candidate list with
   `random.Random(master_seed + _VERIFY_SEED_OFFSET + concept_id)`
   (`negative_prefixes.py:381-384`) so that different concepts get
   different random subsets even when the candidate pool overlaps.
5. **Verification loop.** For each candidate (until
   `n_verified_prefixes_per_concept_target` are verified), derive the same
   stable verification-seed rows from `master_seed + _VERIFY_SEED_OFFSET +
   concept_id * _CONCEPT_STRIDE + vseed_idx * _VSEED_INCREMENT +
   candidate.doc_idx`. Here `concept_id` is the stable provisional candidate
   index from step 2; `_CONCEPT_STRIDE = 10_000` and `_VSEED_INCREMENT = 100`
   separate the usual candidate and verification streams without making an
   unrestricted mathematical uniqueness claim about arbitrary document
   indices. API and local generation use separate chunk controls so provider
   concurrency does not change the legacy local GPU batch contract.

   a. **Null/local compatibility branch.** When `verification_api_model` is
      explicitly null, this is the exact schema-8 path. In
      `verification_prefill_mode="shared"`, the generator prefills each of the
      `K` candidate prefixes once, uses the Transformers 5.6.2 public
      `Cache.batch_select_indices` method to expand the cache into the
      prefix-major `M = sum(seed_counts)` decode rows, and expands last-token
      logits, attention masks, and prompt lengths with the same index. The
      autoregressive loop, per-row generators, stopping behavior, and result
      order are common to both prefill modes.
      `verification_prefill_mode="repeated"` is the explicit
      reference/rollback algorithm and prefills all `M` rows. Unsupported
      caches fail loudly in shared mode; there is no automatic fallback.

   b. **API branch (default).** `Stage0cApiGenerator` uses
      `inclusionai/ling-2.6-flash` by default and receives the feature ID, never
      `CandidateConcept.description`. It fetches and validates up to
      `4 * n_activation_examples` raw Neuronpedia top-activation rows,
      deduplicates `(decoded_text, peak_char_offset)` in source order, and
      keeps `n_activation_examples`. It recomputes tokenizer alignment and
      renders byte-exact `context_before_peak`, `peak_token_text`, and
      `context_after_peak` fields; it does not insert synthetic markers into
      evidence text. When every independent peak normalizes to the same exact
      text, a trusted local check binds that text into the definition request
      and the returned definition must quote it; this prevents neighboring
      context or capitalization guesses from replacing direct evidence. A
      first user-only OpenRouter prompt derives one concise operational
      definition from this evidence. Each prefix/seed continuation pairs a
      definition-bearing system instruction with a contract-first user prompt
      and deterministic variation key.
      Up to `verification_api_prefix_wave_size` candidate prefixes are proposed
      together, capped by candidate-attempt/pool and zero-hit-bail limits but
      intentionally not by the remaining target. Every proposal is scored;
      only the first target passes in candidate order are retained. The
      independent seed calls are flattened through the existing synth batcher
      and bounded by `synth.max_concurrent` and `synth.requests_per_minute`.

      The candidate index retains a bounded exact source bridge after each
      segment cut, preferring the first sentence end within 64 characters.
      Every attempt uses deterministic nonce-qualified begin/end markers
      chosen to be absent from the prefix, definition, and bridge. The model
      authenticates the final 96 characters of prefix plus bridge and returns
      only post-bridge text; Stage 0c prepends the bridge from local source
      bytes instead of trusting the provider to reconstruct a mid-word cut.
      A hard literal-applicability gate makes an unsteered but realistic
      continuation correct when the existing source entity or action cannot
      directly bear the property; inventing a new actor, analysis, goal,
      parameter regime, scene, analogy, or domain to force activation is
      forbidden.
      The parser ignores outside-envelope chatter but rejects missing nonces,
      repeated begin markers, out-of-order markers, full-prefix echoes, exact
      prefix-suffix echoes of at least 16
      characters, substantial normalized source-phrase echoes anywhere in the
      response, assistant preambles/sign-offs, controls, prompt/meta text,
      definition-phrase copies, foreign-script peak-token copies, clear
      writing-system switches and fragments, fabricated or densely mutated
      Unicode letter forms, phrase loops, unnatural exact-peak
      use, protocol-like content, and special extraction-token IDs. It may
      strip exactly one complete leading
      sentence only when that sentence wholly contains a copied definition
      phrase; the remainder then passes every validation from scratch. It
      retains clean short responses across attempts, concatenates and encodes
      all pieces once, and validates the whole continuation before truncating
      to exactly `verification_length` extraction tokens, so appended garbage
      cannot hide beyond the scored window. The requested end-marker separator
      is one ASCII space. Terminal ASCII framing whitespace is stripped even
      when missing or repeated; the live-observed 95-of-96 left-truncated echo
      and nonce end marker missing its last `]` are also normalized without
      entering scored text; duplicate trailing end-marker chatter is discarded
      after the first authenticated boundary. Missing nonces, duplicate begin markers, and
      substantive echo corruption still fail. Each content failure gets up to
      `max_retries + 1` attempts, with a fixed failure-class correction on the
      next attempt and no reflection of rejected response text. Definition
      exhaustion fails loudly;
      continuation-slot exhaustion yields an empty token row, which the common
      scorer treats as a zero-percentile miss rather than accepting or
      reshaping the batch.

   c. For each continuation from either branch, call
      `compute_fc_percentiles` to get the continuation's feature percentile
      (the prefix percentile is discarded here).
   d. Count `n_passed = sum(p > threshold for p in percentiles)` where
      `threshold = cfg.verification_percentile_threshold * 100`
      (multiplication moves the config from `[0,1]` to the 0-100 scale
      `compute_fc_percentiles` returns).
   e. Accept iff `n_passed >= cfg.min_consistency_passes`. On accept, keep ONLY
      the passing continuations / seeds / percentiles in the
      `_VerifiedNegativePrefix` so downstream Stage 1c can reuse them as
      concept-rich training data.
6. **Candidate accept/reject.** Accept a candidate only if it produced
   at least `cfg.n_verified_prefixes_per_concept_target` verified
   prefixes. Otherwise record the
   `RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES` value in
   `rejected_concepts.json`. LLM-side rejects use the corresponding
   `RejectedConceptReason` enum values for below-threshold scores,
   fetch errors, judge errors, or empty examples.
   **Zero-hit bailout.** Within step 5's per-candidate verification
   loop, the iteration aborts as soon as a candidate has produced zero
   verified prefixes after `cfg.verification_zero_hit_bail_after`
   attempts (default 2000). The candidate is then rejected through the
   same `RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES` path with
   `n_verified_found=0`, identical to the natural-pool-exhaustion case.
   Any single successful verification prior to the threshold permanently
   disables the bailout for that candidate, preserving candidates that
   show even sporadic positive signal. Set the field to `null` to
   restore the legacy "scan every candidate" behavior. The bailout
   fires inside `_verify_concept` *after* the candidate scan and
   shuffle, so it does not change candidate discovery or ownership.

   **Partial-progress checkpointing.** At each candidate-block boundary and
   once more at loop exit, every Stage 0c rank writes
   `stage0c_partial_rank_{rank}.json` plus
   `stage0c_partial_rank_{rank}.parquet`. On rerun with the same `run_tag`, if
   the final artifacts are not all present, every current rank validates its
   checkpoint fingerprint and resumes from its first unvisited owned
   candidate instead of restarting from candidate 0. When `world_size`
   changes, a rank retains only checkpoint state for candidates it still owns;
   candidates whose owner changed are reprocessed by their new rank. The
   partial parquet extends the final prefix schema with `candidate_index`, so
   merge can deterministically remap rows to final concept IDs. On successful
   completion all rank-local files are auto-deleted. On validation failure the
   function raises `AxBenchBaseError` and preserves the partials for explicit
   operator recovery. See the pipelines doc for the full fingerprint field
   list and mid-write crash semantics. With fixed inputs and QC responses,
   same-mode resume is exact both at the same topology and after a topology
   change: stable candidate-derived seeds and canonical merge order make
   reprocessed candidates deterministic.

   After each candidate verification returns or raises, Stage 0c calls
   `Stage0cApiGenerator._release_concept_state()` in API mode, then
   `release_cuda_iteration_memory()` (`gc.collect()` plus
   `torch.cuda.empty_cache()` when CUDA is available). The cross-prefix
   verification path does not run allocator cleanup inside each
   generation chunk; per-seed percentile scoring passes
   `release_memory=False` and relies on the runner-level candidate
   boundary for cleanup. Operational API definitions remain cached across all
   waves for one concept and are released only after its verification ends;
   the tokenizer, client, response cache, and concurrency limiter remain
   run-owned.

   The completion heartbeat runs after this cleanup. On Linux it reports
   `current_rss_kb`, `anonymous_rss_kb`, `swap_kb`,
   `anonymous_total_kb` (`RssAnon + VmSwap`), and `peak_rss_kb` from
   `/proc/self/status`; available fields are also emitted as
   `stage0c.memory.*` profiler counters. Restricted or malformed process
   metrics degrade to best-effort `psutil`/`resource` fields and never abort
   evaluation.

   These are lifetime-only changes. They do not change request concurrency,
   verification quality, candidate order, shuffle/seeds, schemas (API mode
   remains 9; explicit-null local mode remains 8), fingerprints, artifact
   identity, or partial-checkpoint resume compatibility.

   **Batch-shape semantics.** Shared and repeated mode are not an equivalence
   pair. A model may produce different final-prefix logits when the prefill
   batch shape changes, and sampling can amplify that difference. Exactness is
   required across repeated executions, resume boundaries, and distributed
   topologies only within the same mode. Cross-mode token, percentile, or
   accept/reject drift is recorded as validation evidence, not treated as a
   correctness failure. Because it can change verification results,
   `verification_prefill_mode` is part of the Stage 0c fingerprint and schema-7
   artifacts cannot be relabeled as shared-mode results.

7. **Stop condition.** Iterate the Stage 0a candidate pool in order and
   stop as soon as `cfg.max_concepts` candidates have been accepted. The
   accepted concepts are re-numbered to contiguous final
   `concept_id` values `0..max_concepts-1`.
8. **Failure mode.** If the pool is exhausted first, raise
   `AxBenchBaseError` naming the remediation levers:
   lower `verification_percentile_threshold`, raise `min_corpus_count`
   to change the pool, or conclude the base SAE LM is too weak for the
   feature set.
9. **Write artifacts.** Write `concept_examples_qc_report.json` for the
   candidates the LLM judge actually saw, `rejected_concepts.json`,
   final `concepts.json`, and `negative_mode_prefixes.parquet` with the
   fixed 8-column schema
   (`negative_prefixes.py:78-89`). Each row is a
   `_VerifiedNegativePrefix` converted via `dataclasses.asdict`.

**Why verification.** The validation-shard features tell us where the
concept fires on corpus text — but Stage 4's negative-mode eval needs
prefixes whose newly generated text **still activates the feature under the
local SAE**. Default API mode proposes deliberately concept-rich continuations,
but still rejects them through the same local measurement. The explicit-null
local mode instead asks whether the base SAE LM naturally keeps the concept
active. A prefix whose sampled continuations switch topics, or whose API-generated text
does not actually activate the SAE feature, is useless as a
"here's a concept-rich context — steer away from it" test. The N-seed
consistency check tests exactly this.

**Why separate from the seed pool.** Stage 0b's seed pool is
concept-agnostic; every row serves every concept. Stage 0c is
concept-specific — each concept gets its own set of 30 prefixes. We
could have merged the loops but keeping them separate means Stage 0b is
deterministic and cheap (no GPU work) and Stage 0c is where the local SAE
scoring cost—and, with `verification_api_model=null`, base Gemma sampling
cost—lives.

**Artifact compatibility.** `negative_mode_prefixes.parquet` stores only
prefix text and passing verification continuations, so Stage 1c and Stage 4 do
not branch on the generation source. Semantic identity is nevertheless strict:
the null/local path remains checkpoint/completed-artifact schema 8, while API
mode uses schema 9 and conditionally fingerprints the protocol version,
OpenRouter model/provider/rate/concurrency controls, API prefix-wave size,
content retries,
activation-evidence settings, and both prompt hashes. Neither mode can resume
the other's artifacts. Use a new `run_tag` or explicitly restart Stage 0
rather than relabeling rows.

**Why the internal dataclasses.** Earlier drafts passed dicts with
magic string keys (`{"doc_idx": ..., "seg_idx": ...}`). Tests started
reaching into literal keys for correctness, which is fragile. Replacing
with `_PrefixCandidate` and `_VerifiedNegativePrefix` (frozen, typed)
gave us the same wire format via `dataclasses.asdict` at the parquet
write boundary and killed the string-key drift. See the module docstring
at `negative_prefixes.py:33-38`.

#### Multi-GPU Stage 0c (DDP)

Stage 0c is the only DDP-parallel stage (`torchrun --nproc_per_node=N
-m scripts.eval.axbench_base_eval stage=0c …`). Candidates are
modulo-partitioned across ranks (`idx % world_size == rank`). Before shard or
candidate work, ranks collectively compare the ordered candidate-pool hash and
count (and collectively surface any local load error); disagreement fails on
every rank before the block collectives can diverge. Each rank
processes a *block* of its owned indices, then all ranks rendezvous via
one `dist.all_gather_object` to share `(n_accepted, n_visited,
max_accepted_idx, next_block_index)`; the loop terminates once
`sum(n_accepted) >= max_concepts` and every rank has covered every index
`<= global_max_accepted_idx` it owns. `run_merge` then sorts accepts by
`candidate_index`, truncates to `max_concepts`, and filters/sorts over-visit
reject and QC records. With deterministic verification and identical QC
responses/cache state, this makes the JSON artifacts and logical parquet
schema/rows/metadata identical to a `world_size=1` run regardless of block
size (a hard, test-enforced invariant:
`tests/axbench_base/integration/test_stage0c_multi_gpu_byte_identity.py`).
Physical parquet bytes are not an API because PyArrow may vary encoding.

**Adaptive block size.** The effective block size is
`_block_size(cfg, world_size)`:

- `world_size <= 1`: the configured `stage0c_multi_gpu_block_size`
  verbatim (clamped `>= 1`).
- `world_size > 1`: `max(1, min(stage0c_multi_gpu_block_size,
  ceil(max_concepts * 2 / world_size)))`.

The cap stops small-target DDP runs from over-processing: with
`block=16, world_size=4, max_concepts=10`, an uncapped block spans 64
candidates (~6× the target) before the global accept check can fire;
the cap reduces that to ~20, shrinking the over-accept budget
(`effective_B * world_size - 1`) and the worst-case cross-rank NCCL
wait. Standard (50) and full (500) tiers are unaffected (the configured
value is already below the cap).

**Fingerprint / resume interaction.** The checkpoint fingerprint stores
the *raw* `stage0c_multi_gpu_block_size`, NOT the derived effective
value, and excludes `world_size` and `max_concepts`. Changing the configured
block size still invalidates a partial checkpoint; increasing
`max_concepts` continues the same ordered candidate stream, and resuming the
same `run_tag` at a *different* `world_size` does NOT invalidate it
(the effective B is recomputed deterministically and the modulo
partition + current-owner state filtering preserve correctness). Merge requires
a committed JSON checkpoint from every current rank, including ranks with an
empty candidate stripe; a missing rank is treated as an incomplete run rather
than silently omitted.

OpenRouter rate/concurrency controls are process-local. In API-generation
mode, each rank derives definitions and issues continuation calls for its own
candidate stripe, so aggregate request admission and concurrency can reach
`world_size * synth.requests_per_minute` and
`world_size * synth.max_concurrent`. The shared on-disk response cache prevents
paying twice for committed identical prompts, but it is not a global
distributed rate limiter. Configure per-rank values with the provider quota in
mind.

**Process-group timeout.** `nanochat.common.compute_init` preserves
PyTorch's backend default DDP collective timeout (10 min for NCCL, 30
for GLOO) globally — it applies an explicit timeout *only* when
`NCCL_TIMEOUT_SEC` is set, so it is not a global default change. The
axbench-base Stage 0c launcher (`scripts.eval.axbench_base_eval`)
opts itself in by defaulting `NCCL_TIMEOUT_SEC` to `1800` (30 min) for
its long rank-0-only Stage 0a (the serial Neuronpedia fetch runs ~25
min while other ranks are barriered). Operators can override it (e.g.
`NCCL_TIMEOUT_SEC=43200` for a 12h pipeline); setting it affects only
the job where it is set. A non-integer, non-positive, or
`timedelta`-overflowing value is rejected at startup with a clear
error.

## Stage 1: Synthesis

Stage 1 produces two training-corpus parquets: one for positive-mode
training, one for negative-mode training. Each row pairs a prefix with
an `output_text` (the training target) and a `category` /
`content_class` label.

### Stage 1b: `synthesize_positive_mode`

`synth/positive_mode.py`. Produces `synth_training_positive.parquet`,
`stage1b_concept_yield.json`, and `concepts_stage1b.json`.

Pipeline:

1. **Resolve the Stage 1b prefix view** once for the complete train pool.
   For each original `seeds_train` index, retain the exact
   `max_prefix_chars`-bounded tail of decoded `SeedRow.prefix_text`; store
   `None` when the result is below `min_prefix_chars`. Bounds count Python
   Unicode code points. The tuple remains aligned to the immutable Stage 0b
   rows, so shuffle order, cache tags, and provenance keep the original index.
   A zero-eligible pool fails here before the percentile table, any Stage 1b
   SAE forward, or synth-cache/API access. The CLI dispatcher has already
   entered the shared base-model/SAE contexts before calling Stage 1b.
2. **Build the percentile table** from the feature statistics, cached
   via `build_percentile_table_from_stats` (LRU-keyed on the stats
   path).
3. **Resolve the Stage 1b checkpoint.** Reuse the row identity computed from
   the validated train-seed tensors, compute the remaining semantic
   fingerprint, and read the run-local partial pair. With no authoritative
   partial, an existing completed positive parquet is
   selection-contract-validated and cannot be extended at another target. A
   fresh run constructs each source and proves the target through the
   target-independent compatibility stream. A resumed run reconstructs
   deterministic clean/dirty partitions, cursors, buffer, terminal state, and
   outstanding descriptors without calling the prefix scorer for persisted
   candidates. It then scores only unscored full chunks needed to prove a
   higher target feasible. Same-or-higher schema-v2 growth commits the new
   target only after that proof and before provider-request construction;
   decreases fail before prefix-cache creation.
4. **Open a `SynthResponseCache`** for the run
   (`AxBenchBasePaths.from_config(cfg).synth_response_cache()`) and resolve all
   trackers through `_run_positive_target_refill_engine`. Every successful
   provider response is committed immediately. Initial admissions are
   checkpointed before request construction. The cache is context-managed and
   survives a stage abort for an exact warm-cache replay.
5. **Account and filter each concept.** The logical attempt count is always
   `n_positives_per_concept`, the complete-pair target. Discarded-and-replaced
   seeds are internal scheduler work and do not count as drops; only final
   target shortfall does. If the drop fraction is strictly greater than
   `max_concept_drop_fraction`, all rows for that concept are removed.
6. **Prepare the Stage 1b handoff.** Write the non-committed preparing yield
   report, enforcing `max_dropped_concept_fraction` and the non-empty-population
   invariant, then write `concepts_stage1b.json` as an exact ordered,
   non-renumbered subset of immutable Stage 0c `concepts.json`.
7. **Validate and write** the accumulated surviving rich/avoiding rows (in concept
   order) as a single parquet at
   `AxBenchBasePaths.from_config(cfg).synth_training_positive()` with the
   16-column training schema. Publication recomputes every output boundary,
   checks every prefix bound and non-empty output, and validates the adjacency
   identity of each avoiding pair before creating the parquet. The schema
   carries extraction-tokenizer identity, the canonical acceptance record
   `axbench_stage1b_positive_qc_contract`, and the canonical prefix-selection
   record `axbench_stage1b_positive_selection_contract` (headroom math, prefix
   bounds/preparation, prefix-score batch shape, score-span policies, and
   copy-guard version). It also carries
   `axbench_stage1_continuation_contract` with role `stage1b_positive`.
   Stage 2, Stage 3, BPB, and BPB-Adv consumers reject missing, role-swapped,
   malformed, or mismatched contracts before model work.
8. **Commit last.** Replace the preparing yield report with the committed
   report only after the filtered concepts and positive parquet are written.
   Downstream stages treat that final report as the Stage 1b commit marker.

**Nanochat tokenizer lifetime.** Stage 1b and Stage 1c use the
`steering.eval.axbench_base.tokenizer_cache` wrapper for the nanochat
RustBPE tokenizer. The wrapper loads the underlying tokenizer once per
process and shares the object across checkpoint fingerprinting, row
construction, and final boundary validation. This removes repeated
`tokenizer.pkl` reconstruction without changing token IDs, BOS ownership,
`n_prefix_nanochat_tokens`, row ordering, checkpoint fingerprints, or parquet
contracts. The cache is operational only; it is not an artifact identity
dimension. The Gemma/HuggingFace extraction tokenizer remains a separate
cache and must never be substituted here.

Candidate preflight calls `release_cuda_iteration_memory()` after every
concept, including exception paths. During synthesis, serial mode releases at
each terminal concept boundary; parallel mode releases once after the shared
pool, and every abort path releases while abandoning queued work. This is a
pure memory-lifetime step for the SAE QC allocator state; it does not touch LLM
cache keys, retry decisions, row ordering, or parquet values.

The shared synth-QC helpers also release transient Gemma+SAE activation
tensors immediately after each SAE forward. This matters inside one
concept as well as across concepts: when a broad feature has too few
clean seeds, Stage 1b's dirty-prefix fallback can score many prefix
cache-fill chunks before the next explicit cleanup boundary.

Candidate-source and unified target-refill engine details:

1. **Canonical candidate source and initial preflight.** For every
   deterministically shuffled original seed index,
   Stage 1b retains exactly the most recent `max_prefix_chars` characters and
   excludes it if that prepared string is shorter than `min_prefix_chars`.
   The same string is used for scoring, prompting, retry QC, both emitted row
   classes, and `n_prefix_nanochat_tokens`.

   Every candidate that can be selected receives a prefix-alone raw percentile
   from `PrefixPercentileCache` (or its scalar-equivalent path). Let
   `T=accept_threshold`, `C=positive_prefix_headroom_ceiling`,
   `D=100T`, `O=min(100C, 99.5)`, and `S=O-D`. Only scores `<=S` are
   admissible. The config must satisfy `0<T<C<=1`.

   Stored sparse IDs remain a priority hint: a seed is "clean" only when none
   of the target IDs (all allowed members for clustered sources) appears in
   `seed.sae_feature_ids`. Admissible clean seeds retain shuffled order.
   Clean candidates are scored lazily in shuffled order. Only after the clean
   stream is exhausted are all dirty candidates scored, headroom-filtered, and
   stably ranked by ascending percentile with shuffled order as the tie-break.
   Every target initially scores toward 72 admissible candidates with the exact
   historical target-72 batch boundaries. If physical clean/dirty exhaustion
   occurs first, a smaller target can still proceed when it is already
   buffered. Targets above 72 continue through full `qc_batch_size` clean
   chunks; only clean-pool exhaustion shortens a chunk, then the dirty pool is
   scored exactly once. Before provider access, every concept must prove that
   its configured target exists; otherwise Stage 1b raises with complete
   pool/scoring/threshold diagnostics regardless of fallback mode. Surplus
   candidates, source cursors, and physical-exhaustion state remain live so
   replacement and target growth never rescore or reuse a seed.

   Prefix cache misses are scored in deterministic right-padded ragged batches
   of at most `qc_batch_size`. The same limit caps caller-thread post-provider
   continuation QC in Stage 1b and Stage 1c; its effective cap is
   `min(qc_batch_size, max(1, synth.max_concurrent))`. BF16 transformer/SAE
   numerics can depend on batch shape, so the effective cap and stable-chain
   frontier algorithm are semantic Stage 1b identity recorded in the positive
   selection contract. Lower `qc_batch_size` before a fresh transaction if
   either QC surface exceeds GPU memory. An authoritative partial cannot resume
   under a different value, and there is no automatic OOM retry with a
   different shape.

2. **For each admitted seed**, run one complete-pair chain. Rich generation
   permits `cfg.positive_max_attempts_per_seed` total semantic attempts with no
   rejection-sampling backoff:
   a. **Build the authenticated, attempt-specific prompt.**
      `_stage1b_continuation_request` hashes a canonical-JSON identity containing
      the stage, content class, concept/feature IDs, original seed index,
      attempt, bounded-prefix SHA-256, and optional cluster fingerprint.
      Collision-checked 24-hex nonce markers are guaranteed absent from the
      untrusted prefix and description. Rendering checks the trusted template
      for missing placeholders before inserting untrusted values, so literal
      placeholder-shaped source text cannot become prompt syntax. The prompt
      ends at the exact final `min(96, len(prefix))` characters with no trailing
      newline.
   b. **Build the cache key** with
      `make_cache_key(model, prompt,
      f"pos_c{concept_id}_s{seed_idx}_a{attempt}")`
      in both concurrency modes. Including the attempt
      index in the tag is **critical**: retries need fresh responses. Without the
      `_a{attempt}` suffix a retry would hit the failing first
      attempt's cache entry and loop forever.
   c. **Call the synth LLM** through the bounded request-chain pool. Inside
      each call, retryable HTTP failures use the separate
      `cfg.synth.transport_max_retries` ceiling and exponential backoff. If
      those HTTP attempts exhaust, Stage 1b aborts immediately with
      concept/seed/side/semantic-attempt context; it does not charge the
      semantic attempt, discard the seed, or admit a replacement.
   d. **Authenticate and parse the complete provider response.** Require exactly
      one matching begin marker and an ordered end boundary. The audited Stage
      0c normalizations remain exact: the final closing bracket may be absent,
      duplicate end-marker chatter is ignored after the first end, and outside
      chatter is discarded. Immediately after the begin marker, require the
      exact local suffix echo; only the fixed separator set
      (`CRLF CRLF`, `LF LF`, `CRLF`, `LF`, one space, or empty) and a
      one-character left-truncated echo are accepted. Missing, mutated,
      right-truncated, or two-character-left-truncated echoes fail.
   e. **Validate both the full response and stored token window.** Remove the
      authenticated echo and terminal space/tab/CR/LF only; leading continuation
      characters are data and remain byte-exact. Before truncation reject empty
      text, the full retained prefix, residual prefix-suffix overlap of at least
      16 characters, controls, protocol-like markers, substantial normalized
      source copies, and repeated phrase loops. Tokenize the entire continuation
      with `add_special_tokens=False`, reject any special-token ID anywhere
      (including after the future stored window), take exactly
      `cfg.output_length` tokens, decode, terminal-trim, and run the same guards
      again. Protocol failures increment
      `stage1b.continuation_protocol_failures`; special-token failures increment
      `stage1b.special_token_refusals`. Accepted end-marker, outside-chatter,
      suffix-framing, and terminal-separator normalizations increment the
      corresponding `stage1b.continuation_protocol.*` counter with concept,
      attempt, and content-class attributes. Both rejection classes retry
      without entering SAE QC or creating a row.
   f. **Run continuation-only QC** via the scalar
      `output_fc_percentile_for_continuation` reference or its batched plural
      form. With concurrency and `qc_batch_size` both above one, one caller
      callback waits for the lowest stable-index request frontier, scores its
      rich continuations together, then combines the natural continuations of
      rich passes with its LLM-avoiding continuations in one strict-zero batch.
      Provider timing cannot change that frontier. This performs the existing
      joined-tokenization forward and scores only the continuation span. The
      baseline is the exact prefix-alone percentile bound to the selected seed;
      Stage 1b does not recompute or silently replace it at retry time.
      `delta = output_percentile - seed_percentile`.
   g. **Build the row dict** with the 16 columns. If
      `delta >= 100 * cfg.accept_threshold`, the row is accepted (marked
      `qc_accepted=True`) and the chain advances to avoiding QC.
3. **Complete the atomic pair or replace the seed.** Test the natural document
   continuation first; strict-zero success costs no API attempt. Otherwise run
   up to the same total cap of LLM-avoiding attempts in the same chain. Rich or
   avoiding exhaustion discards the entire seed, including an accepted rich
   row, and appends one fresh replacement chain to the global queue tail.
   `positive_zero_hit_bail_after` limits distinct candidates while accepted
   pairs remain zero; outstanding boundary candidates finish, and any success
   lifts the guard and refills all deficits.

   Fallback is not consulted per seed. Only terminal pool/guard shortfall
   dispatches it: `"error"` aborts before publication, while `"warn"` and
   `"best"` retain complete accepted pairs only. Yield accounting always uses
   the configured target as `attempted`, so replacements are not drops and
   only final shortfall contributes to the drop fraction.
4. **Canonicalize, validate, and write.** Sort by input concept order then
   candidate rank, with each rich row immediately followed by its avoiding
   row. Before Arrow applies the existing float32 QC
   schema, the producer checks each full-precision rich row's exact delta
   identity and inclusive decision and checks the strict-zero avoiding-row
   sentinel contract. The Parquet metadata then records a versioned
   `axbench_stage1b_positive_qc_contract` containing both threshold units,
   comparison, percentile currency, and algorithm. Legacy or contradictory
   positive corpora are rejected by Base, BPB, and BPB-Adv consumers. The
   separate selection contract records the target, attempt cap, guard,
   ordering/refill semantics, prefix/headroom protocol, and BF16 batch shape.
   Target 72 uses
   `positive_target_refill_post_provider_qc_v5`; every non-72 target uses
   `positive_target_compatibility_post_provider_qc_v6` and records
   compatibility target 72, the full-chunk post-compatibility rule, and the
   effective post-provider QC batch identity. The third record,
   `axbench_stage1_continuation_contract`, binds the accepted text to the
   authenticated framing, echo, full-response, truncation, and stored-window
   semantics. All three records are required; stale corpora must be regenerated
   under a fresh `run_tag`.

#### Stage 1b checkpoint transaction

`synth/positive_checkpoint.py` owns storage only; candidate, tracker, and
scheduler invariants remain in `positive_mode.py`. Schema version 2 uses
`stage1b_partial.parquet` followed by `stage1b_partial.json`. Parquet appends
three internal `int64` fields (`checkpoint_seed_idx`,
`checkpoint_candidate_rank`, and `checkpoint_acceptance_sequence`) to the
ordinary 16-column training schema. The checkpoint widens the three QC
arithmetic fields (`qc_seed_percentile`, `qc_output_percentile`, and
`qc_delta`) from final-artifact `float32` to internal `float64`; this preserves
their exact arithmetic identity across parquet round trips. It stores only
complete adjacent pairs in global acceptance order. JSON is the commit marker
and records source cursors, buffered candidates, terminal accepted/discarded
descriptors, and outstanding chain rollback descriptors.

Schema version 2 and the parquet/JSON shape remain unchanged. The current
fingerprint algorithm is
`stage1b_growing_target_bulk_seed_rows_identity_resume_v4`; it keeps the
`seed_rows_sha256` key but supplies its value with the bulk tensor identity.
The preceding `stage1b_growing_target_post_provider_qc_resume_v3` fingerprint
continues to bind the effective post-provider cap and stable-frontier
algorithm. An exact pre-batching v2 fingerprint or exact v3 fingerprint
migrates once in memory. Committed v2 rows retain their stored singleton
scores, outstanding descriptors adopt the configured current QC shape, and
the next successful commit writes v4. A current v4 checkpoint rejects any
effective-shape change before prefix scoring or provider access.

The bulk row identity is `stage1b_seed_rows_v2`. It hashes exactly 18 framed
payloads: the identity-version payload, the big-endian unsigned 64-bit row
count, and each complete contiguous tensor in the fixed schema order from
`prefix_text_offsets` through `doc_idx`. Every frame is
`uint64_be(label_bytes) || label || uint64_be(payload_bytes) || payload`.
The schema fixes each tensor name, one-dimensional rank, and dtype; CSR and text
offsets retain empty-list and row boundaries, UTF-8 byte tensors retain exact
text, and float magnitudes retain their persisted IEEE-754 float32 bits. The
implementation requires a little-endian host and feeds CPU-contiguous tensor
buffers to SHA-256 without a bytes copy. It deliberately excludes safetensors
metadata, serialization layout, paths, manifest ordering, and the Stage 0b
producer fingerprint: those do not identify the observed ordered train rows.

Every snapshot proves exact candidate-rank ownership, admitted-seed ownership,
terminal counter derivation, pair/descriptor agreement, contiguous acceptance
sequences, and consistent source exhaustion or zero-hit termination. Persisted
seeds are rebound by original index and revalidated against the current
bounded prefix and clean/dirty partition. Resume never calls
`PrefixPercentileCache.get_many` for buffered, accepted, discarded, or
outstanding descriptors; the next unscored replacement retains the original
cursor and compatibility/full-chunk batch boundary. Expensive token-boundary and QC validation runs once
when each immutable pair is accepted and once when committed rows are restored,
not again for every periodic snapshot. Each snapshot still performs the full
structural transaction checks and atomically rewrites the one checkpoint
parquet required by this contract.

Publication is atomic parquet first, JSON second. A newer even parquet suffix
is truncated to the JSON-committed prefix in memory, while a parquet-only
orphan is ignored. Missing committed rows, a half-pair suffix, malformed JSON,
or any committed-prefix mismatch fails loudly. The exact recursive fingerprint
binds ordered concepts and seeds, statistics bytes, model/SAE/tokenizers,
prompts/protocols, the target-independent compatibility selection stream, and
cluster contracts. Only `n_positives_per_concept` is removed from the semantic
fingerprint. JSON separately records the strict positive
`highest_requested_target`. Operational rate,
concurrency, transport retries, profiling, output path, the tag string, and
the prefix-store toggle are excluded, although the same explicit `run_tag` is
required to select the files. Post-synthesis fallback and population guards
are included because they determine whether and how the checkpointed history
may publish; changing them requires a fresh partial transaction.

Schema-v2 resume accepts the same target or a higher target. Growth restores
state, enlarges tracker targets, and proves the enlarged candidate capacity
before admitting deterministic deficits or publishing the increased target.
If feasibility or that write fails, the prior JSON remains authoritative.
Decreases fail before prefix-cache or provider construction. Schema-v1 is
read-only migration input: the complete
document must be valid, its stored target must be exactly 72, the current
target must be at least 72, and every other exact-type fingerprint field must
match the target-72 Phase 1 fingerprint. Other schema-v1 targets used
target-shaped BF16 prefix-scoring calls and cannot be relabeled as the current
target-72 compatibility stream. Legacy v1/v2/v3 dispatch happens only after
strict JSON parsing and supported schema/algorithm checks. It computes the
exact historical row digest once, validates every legacy field, and normalizes
to schema 2/v4 in memory. The target-72 restriction above remains specific to
schema v1; v2 and v3 retain their established exact migrations. Loading does
not touch disk. The scheduler's normal parquet-first/JSON-second checkpoint
publishes the normalized v4 state before provider requests. A crash before that
commit leaves the old marker authoritative and may repeat the scan; a crash
between writes uses the ordinary ahead-of-marker reconciliation. The next
successful resume reads v4 and performs no legacy scan.
Growth is partial-only and requires reuse of the same explicit `run_tag`; a
completed positive parquet without authoritative JSON is never extended in
place. If final files coexist with an authoritative partial after an
interrupted publication/cleanup boundary, the partial may resume only the
already-committed target; it cannot authorize final-corpus growth.

Stage 0c and Stage 1b share the repository's atomic parquet/byte replacement
primitives but deliberately keep separate transaction validators. Stage 0c
owns independent rows and may truncate any newer suffix to a committed row
count; Stage 1b owns atomic rich/avoiding pairs and must reject a half-pair
suffix. Keeping reconciliation local makes that fail-loud pair boundary
explicit while the physical durability primitives remain shared.

The scheduler checkpoints after initial admission, after each 64 terminal
chains, on every caught `BaseException`, and after completion before final
publication. Outstanding work deliberately loses partial rich/avoid phase and
semantic counters; it restarts at rich attempt zero. Cache-committed responses
replay from SQLite, while a response interrupted before its durable cache
commit may require another provider request. Caught failures preserve every
terminal outcome. Short multi-field scheduler commits defer `SIGINT`, while
provider work, parsing, QC, and I/O remain interruptible. An uncatchable hard
stop can lose at most 64 terminal outcomes since the periodic commit, though
their responses may already be in SQLite. Once the canonical parquet, filtered
concepts, and final `"ok"` yield report are committed, cleanup removes JSON first
and parquet second. Cleanup failure is only a warning because the final
artifacts are authoritative; the warning states whether an authoritative JSON
marker or only a parquet orphan remains.

For the paired avoiding side, the trusted natural Stage 0b continuation stays
outside the LLM text guards and goes directly to strict-zero QC. Only if that
tier fails does the LLM fallback run; its response uses the same
attempt-specific envelope, suffix authentication, full-response validation,
token limiting, and post-decode validation as the rich path before strict-zero
QC. Rich and avoiding prompts, parsers, QC, row builders, and exact boundaries
all receive the identical bounded prefix string.

**Why include paired avoiding rows in the positive-mode parquet.** Stage
2's `_concept_mode_filter` splits the parquet into
`(concept_rich_texts, concept_avoiding_texts)` for delta-ranked mode by
selecting the same `concept_id=cid` on both sides. Positive mode emits a
paired `concept_avoiding` row for each accepted rich row, so Stage 2 never
needs a global-negative sentinel pool.

**Why attempt tag discrimination.** The unified Stage 1b engine includes the
attempt in each rich and avoiding tag. A cache keyed only on `(model, prompt)` would return
the same response on every retry, defeating the retry mechanism. The tag
ensures re-invocations see a fresh LLM call.

**Why percentile delta instead of absolute threshold.** A concept with
a high baseline percentile would be trivially "passed" by any
continuation; a concept with near-zero baseline would be strictly
harder. The delta normalizes — we ask "did the synth move the needle"
rather than "did it exceed an arbitrary threshold". Review-fix 16
pinned this after finding that absolute-threshold QC produced silently
different acceptance rates across concept frequencies.

### Stage 1c: `synthesize_negative_mode`

`synth/negative_mode.py:386-451`. Reads `concepts_stage1b.json` and produces
`synth_training_negative.parquet`, `stage1c_concept_yield.json`, and the final
`concepts_stage1c.json`.

Emits two row types per verified negative-mode prefix:

1. **`concept_rich` rows** — copied verbatim from Stage 0c verification
   continuations (`_copy_concept_rich_rows`, `negative_mode.py:152-195`).
   These rows need no LLM call and no QC — the verification loop
   already confirmed the continuation activates the concept. One row
   per passing verification continuation (3-5 per prefix). The rows
   use `category="positive"` (not `"negative"`) per spec: they are
   the "concept-rich training side of negative mode", i.e., the text
   that the Stage 4 negative-mode generator is asked to steer AWAY
   from.
2. **`concept_avoiding` rows** — synthesized via the OpenRouter LLM
   using `synth_continue_avoiding_concept.txt`
   (`_synthesize_concept_avoiding_row` in `negative_mode.py`),
   with **strict-zero QC** via the scalar `classify_avoid_continuation`
   reference or batched `classify_avoid_continuations` (`synth/qc.py`): the
   continuation is accepted iff the outcome is
   `INACTIVE` — the concept feature does NOT fire above
   `cfg.extraction_threshold` on any token position in the continuation
   (issue #518, mirroring the positive-mode fix in issue #513). An
   empty/whitespace provider continuation is rejected by the authenticated
   parser before the classifier (counted under
   `stage1c.continuation_protocol_failures`);
   a continuation whose tokens merge entirely into the prefix seam is
   rejected by the classifier's `NO_TOKENS` outcome (also counted
   `stage1c.qc_empty_rejects`). Neither is a strict-zero PASS. Resample
   immediately up to `cfg.max_retries` additional content attempts; HTTP
   transport failures separately use `cfg.synth.transport_max_retries` with
   exponential backoff.
   On exhaustion:
   - `"error"`: raise
   - `"warn"`: log + skip
   - `"best"`: **no meaningful definition** for strict-zero QC (there's
     no delta to rank attempts by), falls through to `warn` behavior
     with an explanatory log line

Per-concept loop:

```python
for concept in concepts:
    for prefix_row in concept_prefixes:
        rows.extend(_copy_concept_rich_rows(concept, prefix_row, cfg))
        accepted_row = _synthesize_concept_avoiding_row(...)
        if accepted_row is not None:
            rows.append(accepted_row)
        else:
            # dispatch fallback
            ...
```

In the serial path, Stage 1c releases iteration memory after each concept.
In the bounded parallel path, each caller callback immediately processes up to
`qc_batch_size` ready results without waiting to fill the batch and performs
one ragged strict-zero forward for the successfully parsed continuations.
Provider errors and parse refusals do not consume GPU rows. The helper releases
the batch activation tensors after that forward, and the outer path releases
iteration memory again after the global request-chain pool. These placements
are memory-only and preserve deterministic row assembly. Setting
`qc_batch_size=1` selects the scalar reference callback.

**Yield and population guard.** One verified Stage 0c prefix is one logical
attempt. Its copied rich row(s) and exactly one accepted avoiding row form an
atomic group; an incomplete group contributes no rows and counts as dropped.
The same strict per-concept `>` comparison and aggregate population guard as
Stage 1b apply. Stage 1c writes `concepts_stage1c.json` as an exact ordered
subset of `concepts_stage1b.json`; it never amends either parent file.
`concepts_stage1c.json` is the single population authority for every later
stage and BPB consumer. Extra rows left in the already-committed positive
parquet are ignored by concept ID. A final concept missing either class from a
required corpus is corruption and raises rather than being skipped.

**Schema: 19 columns** (`NEG_TRAINING_SCHEMA` in `negative_mode.py`). The 15 base columns
shared with Stage 1b plus 4 negative-mode extras:

- `prefix_source_shard` (int32)
- `prefix_source_doc` (int32)
- `prefix_source_seg_idx` (int32)
- `qc_zero_activation_verified` (bool) — True only for concept_avoiding
  rows whose `classify_avoid_continuation` outcome was `INACTIVE` (≥1
  continuation token, zero firings above threshold). False for
  concept_rich rows (which never go through that check). Authenticated-parser
  empty outputs count under `stage1c.continuation_protocol_failures`;
  post-parse seam merges with no new tokens count under
  `stage1c.qc_empty_rejects`. Neither is ever written True.

The extras track back to the Stage 0c source segment and document so
diagnostic tooling can correlate training rows with their originating
prefixes.
Stage 1c shares Stage 1b's process-local nanochat tokenizer cache for its
exact owned-BOS output-boundary calculations. Its strict-zero QC and
extraction-tokenizer paths are otherwise unchanged.
The parquet schema also carries `axbench_stage1_continuation_contract` with
role `stage1c_negative`; every Base/BPB/BPB-Adv consumer validates that exact
role and protocol before interpreting rows. BPB-Adv cleaning preserves all
source metadata and revalidates the contract before publishing the filtered
copy.

**Strict-zero QC rationale.** The negative-mode target is
"continuation that does not activate the concept". Percentile delta is
the wrong metric here — a continuation that activates the concept at
the 40th percentile is still activating it. We want the floor, not
the ceiling. `classify_avoid_continuation` checks
`(continuation_fc > extraction_threshold).any()` and returns `ACTIVE`
if any position fires. An accepted (`INACTIVE`) continuation has ≥1
continuation token AND zero firings in the continuation slice.

**Why hash-based gen_seed for concept_avoiding rows.**
`_stable_hash_int` (`negative_mode.py:130-139`) uses sha256 over the
prefix text instead of Python's built-in `hash()`. CPython's hash is
session-randomized via `PYTHONHASHSEED`, which would make the gen_seed
(and thus the provenance trail) differ across runs. The sha256 prefix
is deterministic.

**Why `qc_seed_percentile=NaN` on negative-mode rows.** We're not
gating on prefix percentile here; strict-zero only cares about the
continuation. NaN is the schema signal for "no QC percentile applies".

## Stage 2: Concept feature mapping

`concept_feature_mapping.py:375-437`. Produces `concept_features.safetensors`
and optionally `concept_features_neg.safetensors`.

Writes four tensors + metadata per file. Tensor rows are addressed directly by
the original `concept_id`, so the extent is `max(concept_id) + 1`:

- `feature_ids`: `(max_id + 1, n_rank_store)` int32, `-1` for unused slots
- `percentile_deltas`: `(max_id + 1, n_rank_store)` float32, NaN for
  unused slots
- `canonical_feature_id`: `(max_id + 1,)` int32 — `concept.feature_index`
  for survivors, `-1` for filtered-out rows
- `canonical_rank`: `(max_id + 1,)` int32 — position of the canonical
  feature in the ranked list (sentinel `n_rank_store` if not found)
- `metadata={"mode": "canonical" | "delta_ranked"}`

### Canonical mode

`concept_feature_mapping_canonical`, `concept_feature_mapping.py:166-211`.
Trivial writer. For each concept:

- `i = concept.concept_id`
- `feature_ids[i, 0] = concept.feature_index`
- `feature_ids[i, 1:] = -1`
- `percentile_deltas[i, :] = NaN`
- `canonical_feature_id[i] = concept.feature_index`
- `canonical_rank[i] = 0`

Rows for filtered-out original IDs remain entirely sentinel-valued. No concept
is compacted or renumbered.

No GPU work. Used when we trust Neuronpedia's concept-to-feature mapping
as ground truth (which is the default).

If `cfg.run_negative_mode`, the same file is written twice — once to
`concept_features.safetensors` and once to
`concept_features_neg.safetensors`. Canonical mode doesn't actually
depend on the mode, so the negative file is a byte-identical copy; we
write it for downstream stages that expect both files to exist.

### Delta-ranked mode

`concept_feature_mapping_delta_ranked`, `concept_feature_mapping.py:328-372`.
Runs Gemma + SAE on the synth training corpus and ranks features by the
percentile delta between concept-rich and concept-avoiding rows.

Per concept:

1. Filter the synth rows via `_concept_mode_filter`
   (`concept_feature_mapping.py:100-124`).
   Both positive and negative mode use the same per-concept split:
   concept_rich side is `(cid, concept_rich)` and concept_avoiding side
   is `(cid, concept_avoiding)`.
2. Run `_batched_dense_sae_max_pool` on each side
   (`concept_feature_mapping.py:112-163`). For each text, encode with
   the extraction tokenizer, run the batched SAE extractor at
   `target_model_meta.sae_layer`, and max-pool the
   `(seq_len, n_features)` tensor along the sequence dimension. Texts
   that tokenize to empty yield a zero row. Returns a
   `(n_texts, n_sae_features)` float tensor.
3. Mean-pool both sides across texts to get two per-feature mean
   activation vectors.
4. Convert means to percentiles via `magnitude_to_percentile`
   (`concept_feature_mapping.py:300-302`). The percentile table is
   prebuilt once per run via `build_percentile_table_from_stats` and
   LRU-cached.
5. Compute `delta_pct = pos_pct - neg_pct`.
6. Sort descending, take the top `n_rank_store` feature ids and their
   deltas. Store them in `feature_ids[i]` and `percentile_deltas[i]`.
7. Locate the Neuronpedia-mapped feature (`concept.feature_index`) in
   the ranked list. Record its rank in `canonical_rank[i]`. If it's
   not in the top N, set `canonical_rank[i] = cfg.n_rank_store` as a
   sentinel.

**What this measures.** `canonical_rank` is a diagnostic: "where does
the feature our concept was mapped to actually land on the ranked
delta list produced by our synth corpus?" If Neuronpedia's mapping is
good, the canonical feature will be rank 0 or near it. If it's far
down, either the Neuronpedia description doesn't match what the feature
actually fires on, or the synth LLM failed to generate genuinely
concept-rich text.

**Why mean across texts and max-pool within texts.** Max within a text
captures whether the feature fired anywhere in that text; mean across
texts averages out individual text variance. Alternative choices (mean-
pool within, max across) give worse-behaved rankings — max-within
matches the semantics of the training-time pooling convention while
mean-across smooths over noisy individual generations.

**Callers pass `target_model_meta.sae_layer`** for `sae_layer` so the
SAE runs at the right layer. Do NOT hardcode the layer.

**RuntimeState accessors.** Canonical mode uses
`state.downstream_concepts()` only. Delta-ranked mode uses
`state.downstream_concepts()` +
`state.encdec_meta()` + live `state.base_model() + state.base_sae()`
contexts.

## Stage 3: Baseline training and context enumeration

Five sub-stages produce baseline artifacts under
`baselines/`. The dispatcher in `scripts/eval/axbench_base_eval.py`
(`_run_scalar_baseline_training` at lines 233-295)
(`_run_scalar_baseline_training`) shares the trainer-loop structure
across 3a / 3b / 3d. Training covers exactly the final filtered concept
population. Corpus presence is checked against that population and missing
rows raise instead of silently skipping a concept.

### Stage 3a: DiffMean

`baselines/diffmean.py`. Computes a residual-stream direction per
concept as `mean(concept_rich_acts) - mean(concept_avoiding_acts)` at
`cfg.layer_hooks.diffmean_and_probe_layer`. The helper opens a
residual-stream hook (`baselines/hooks.py`), runs the target model
over the training corpus parquet, captures activations per row via
`baselines/_capture.py`, splits by `content_class`, and writes one
safetensors per `(concept, target, mode)`. Trained for both target
models. Dispatched via `_run_scalar_baseline_training(targets=("decoder_only", "encdec"))`.

**RuntimeState accessors.** `state.downstream_concepts()` + live
`state.decoder_only_model()` + live `state.encdec_model()`.

### Stage 3b: LinearProbe

`baselines/linear_probe.py`. Logistic regression over the same
captured activations Stage 3a uses, targeting the binary
`concept_rich`/`concept_avoiding` label. Writes the unit-normalized
`direction` (the probe's `coef_[0]`), plus the raw `coef` and `intercept`
tensors, per `(concept, target, mode)`.

The Stage 3a/3b safetensors metadata schema is
`{concept_id, mode, target_model, layer, n_pos, n_neg, max_activation}`.
`max_activation` belongs to that exact direction artifact. For its final unit
direction `d`, the trainer computes the largest signed `h_token · d` over all
continuation tokens in that concept's `concept_rich` and `concept_avoiding`
fit rows at the target model's hook layer, in float32 on CPU (no ReLU, absolute
value, probe intercept, or rounding). A zero direction stores exactly `0.0`;
otherwise the value must be finite and positive. The trainer raises rather
than writing a malformed scale. This schema and statistic are unconditional;
there is no configuration knob or feature-statistics fallback.

**RuntimeState accessors.** Same as 3a.

### Direction-owned max-activation steering scale (#1228)

DiffMean and LinearProbe directions are unit-normalized at training time. Their
effective Stage 4, axbench-bpb, and axbench-bpb-adv intervention is
`factor × direction_artifact.max_activation × unit_direction`. The factor grid
and serialized `coefficient_str` remain unchanged. The scale comes **only** from
the direction artifact being applied; `feature_statistics.safetensors` and its
SAE-feature `max_mag` cannot influence residual-direction steering.

For a unit direction `d`, Stage 3 defines `max_activation` as the maximum signed
`h_token · d` over every continuation token in both `concept_rich` and
`concept_avoiding` fit rows at that target model's configured direction hook
layer, computed in float32 on CPU without ReLU, absolute value, probe intercept,
or rounding. A zero direction pairs only with `0.0`; every nonzero direction
requires a finite, strictly positive value. There is no configuration knob and
no fallback to an SAE statistic, `1.0`, or any other constant.

Every scaled stage preflights the exact direction artifacts it will consume and
raises before expensive work if metadata is missing, invalid, negative,
zero-inconsistent, or bound to the wrong concept, mode, target model, or layer.
Artifacts produced before #1228 must be regenerated by re-running Stage 3a/3b
under a fresh run tag. Negative mode applies `-abs(factor) × max_activation`.
AxBench-BPB-Adv counter-steering independently loads and validates the
**counter-direction artifact's own** value; it never borrows the baseline
artifact's value. A magnitude-zero additive control remains the unchanged base
intervention object.

**Resume guard.** The scale is a generation/scoring determinant but is absent
from per-row resume keys. A scaled stage therefore fingerprints the complete
set of consumed `(family, target_model, mode, concept_id, max_activation)`
bindings and writes `dirmaxact:<sha256>` to the existing
`.steering_scale_identity` marker. A changed set/value, scaled durable rows with
no marker, or any legacy `maxact:<...>` marker raises rather than mixing scale
regimes. Runs containing only unscaled method rows are not blocked. Use a fresh
run tag (or clear the run directory) to re-score after changing artifacts.

### Stage 3c: Prompting

`baselines/prompting.py`. Pure data: reads the concept list, writes
`baselines/prompting/{mode}_descriptions.json` with one entry per
concept containing the description the prompting baseline will
inject at Stage 4. One file per mode (NOT per concept).

**RuntimeState accessors.** `state.downstream_concepts()` only. No model
context needed.

### Stage 3d: LoRA

`baselines/lora/training.py`. Trains a rank-`cfg.lora.rank` LoRA
adapter per `(concept, mode)` targeting the decoder-only base model.
`_AttachedAdapterRef` tracks per-layer parent/child references for
clean attach/detach via `baselines/lora/_attach.py`. The loader at
`baselines/lora/loader.py` maps saved tensors back to runtime
modules at Stage 4. Saves to `.pt` extension but uses the
`safetensors` format internally (see
`_save_lora_adapter`, `baselines/lora/training.py:180-199`).

**RuntimeState accessors.** Live `state.decoder_only_model()` only.
Encoder-decoder target is intentionally skipped — no LoRA variant is
trained for encdec.

### Stage 3e: exact Stage 4 context enumeration

`stage4_contexts.py::enumerate_stage4_contexts` is rank-0 and runs after
baseline training. It loads the committed eval seed pool, concepts,
mode-specific synth eligibility, prompting descriptions, enabled methods, and
configured eval-seed caps. Every encoder-decoder prompt is built through the
same `build_stage4_prompt_tokens` function used live in Stage 4:

- ordinary methods: `[owned BOS] + seed content tokens`;
- prompting methods: `[owned BOS] + raw prompting-prefix tokens + seed content`.

Non-prompting methods deduplicate to seed-only token keys. Prompting methods
enumerate every required `(mode, eligible concept, selected eval seed)` prompt.
The stage checks `prompt_token_count + max_new_tokens <= encdec_sequence_len`,
deduplicates by the exact token-context key, and atomically writes
`stage4_contexts.jsonl`, `stage4_contexts_extraction/shard_00000.parquet`, then
`stage4_contexts_manifest.json` last. The manifest pins seed and prompt
identities, enabled-method/config identity, tokenizer checksum, model length,
ordered token keys/counts, and file hashes.

## Stage 4: Generation

`generation.py::generate_all`. Runs the method registry
across the per-concept eval seed pool for the final filtered concept
population. Before Stage 3 training, Stage 4 planning, or Stage 4 generation
can proceed, every final concept must have a complete rich/avoiding class pair
in each required synthesis corpus. Corpus presence is validation evidence,
not a second population-selection mechanism: the Stage 1c filtered concept
list (or the Stage 1b list when negative synthesis is disabled) remains the
sole authority, and missing rows fail loudly rather than silently shrinking
the generation population.

Before constructing `ParquetAppendWriter` or executing the first forward,
Stage 4 resolves every DiffMean/linear-probe coordinate required by the active
method plans and validates the complete artifact identity and configured hook
layer. The direct scalar loader repeats that validation as defense in depth.
Metadata and tensors come from one descriptor-bound byte payload, so replacing
a pathname during validation cannot combine identity from one file version
with a direction from another. This applies to clustered and unclustered
artifacts; clustered files additionally require their expected artifact kind
and cluster provenance.

Each `(method, concept, seed, coefficient)` combination dispatches through
`_run_intervention` (`generation.py`) which either:

1. runs `_generate_with_nanochat_tokens` with the appropriate hook /
   adapter / encoder input active, or
2. runs the context-only generation helper for methods that do not
   install a model hook.

`build_mode_aware_encoder_input` (`encoder_input.py`) constructs the
encdec cross-attention input tensor for every "ours" method, calling
the partition modifiers (`ours_extract_and_add`, etc.) that mutate the set of active
features before they are bucketized. The `inactive_sampler` plumbed
in from `RuntimeState` supplies sampled inactive fids when a
partition modifier needs them (only when the encdec target was
trained with negative feature sampling).

Outputs land as `generations/batch_NNNNNN.parquet` via
`ParquetAppendWriter`. Each row carries
`(method, concept_id, seed_id, coefficient, prefix, output_text,
gen_source, cache_hit)` plus `prompt_token_key`, `prompt_token_count`, and
`seed_pool_fingerprint`.

Every method uses `build_stage4_prompt_tokens`; dispatch receives the complete
model-ready prompt and never prepends or reorders tokens. In particular,
prompting now uses `[BOS] + prompting prefix + seed`, never
`prompting prefix + [BOS] + seed`. Generation enumeration, resume identity,
Stage 3e, and context-cache lookup all share the same exact token list. Existing
generation directories without the strict schema are rejected. Resume also
requires the visible prefix, token key/count, and seed fingerprint to match,
and all method plans are sequence-length-preflighted before any row is written.

### Full-context routing (Stage 4)

SFT-trained encoder-decoders use a cache-backed full-context prompt. Stage 4
dispatch preserves the original prefix tokens and relies on the target bundle's
context provider; target-only prompt conditioning is not supported. Before the
first generation, Stage 4 reconstructs the current Stage 3e contract and
requires the manifest, cache identity/selection hash/model length, and contexts
fingerprint to match. Every expected encoder-decoder token key must resolve to a
non-empty, structurally valid segment tuple.

**RuntimeState accessors.** Live `state.decoder_only_model()` +
`state.encdec_model()` + `state.downstream_concepts()` + `state.seeds_eval()` +
`state.nanochat_tokenizer()` +
`state.decoder_only_meta()` + `state.encdec_meta()` +
`state.inactive_sampler()` (may be `None`).

## Stage 5: Judge

`judge.py::judge_all`. Reads every `generations/batch_NNNNNN.parquet`,
sends each row to the AxBench judge LLM via
`AxBenchJudgeClient` (OpenRouter-backed), parses
`"Rating: [[N]]"` responses for relevance + fluency, and writes
`judge_scores/batch_NNNNNN.parquet` in the original generation-row
order. Stage 5 streams Stage 4 parquet batches in bounded row chunks,
flattens each pending chunk as
`concept0, fluency0, concept1, fluency1, ...`, and scores it through
`AxBenchJudgeClient.score_many()` with up to `cfg.judge.max_concurrent`
workers. Rows are appended only after both scores are known, so parquet
order stays deterministic even when HTTP latency varies. Per-prompt
cache via `judge_cache.sqlite`; concurrent duplicate misses are
deduped by cache key. Invalid responses (malformed or `None`) get
`concept_invalid` / `fluency_invalid=True` so Stage 6 can gate.

**RuntimeState accessors.** `state.downstream_concepts()` +
`state.judge_client()`.
No model context.

## Stage 6: Aggregate

`aggregation.py::aggregate`. Pure CPU math: reads
`judge_scores/*.parquet`, computes per-method per-mode area-under-dose-
response, applies the `invalid_drop_threshold` / `invalid_warning_threshold`
gates, and writes `report.yaml` + `leaderboard.csv` +
`aggregation_report.yaml`.

**RuntimeState accessors.** None. Stage 6 takes only `cfg`.

## Extraction: training parity

When the target was trained with
negative feature sampling, active features are discretized into
`num_buckets - 1` buckets; the final bucket remains reserved for
forced-inactive features exactly as in `build_feature_pipeline`.

**What `extraction_max_tokens` controls.** The hard upper bound on
tokens fed to base Gemma in any extraction forward pass. Passed as
`context_window` to `extract_document_features`, which clamps to
`min(context_window, model.cfg.n_ctx)` internally. Replaces the old
hardcoded `_MAX_CTX_CLAMP = 2048` constant. Swapping to a
longer-context base model (e.g., gemma-2-9b) is now a config-only
change.

**What `extraction_threshold` controls.** The floor for
`classify_avoid_continuation` in strict-zero QC. An activation value
strictly above this counts as "fired"; at or below does not. Default
`0.0` treats any positive activation as firing. Raising this tolerates
small post-activation noise.

## QC deep-dive

Two distinct QC questions:

1. **"Did the concept fire on the continuation at all?"** — tri-state
   (`NO_TOKENS | INACTIVE | ACTIVE`), used by negative-mode strict-zero
   QC (issue #518) and positive-mode avoid tiers (issue #513).
2. **"Did the continuation push the concept's percentile up relative to
   the prefix baseline?"** — scalar delta, used by positive-mode QC.

Both modes tokenize the `(prefix, continuation)` pair exactly **once**
per checked continuation via `classify_avoid_continuation` (issues #513
and #518), combining the token-contribution guard and the strict-zero
activation check into a single tri-state outcome
`NO_TOKENS | INACTIVE | ACTIVE` (see the subsection below).

### `classify_avoid_continuation`

`synth/qc.py`. Tokenizes `(prefix, continuation)` exactly once and
returns a tri-state `AvoidContinuationOutcome`:

- **`NO_TOKENS`** — the continuation contributes no tokens beyond the
  prefix seam (e.g. empty/whitespace, or a string that BPE-merges
  entirely into the prefix); no SAE pass runs. Not a strict-zero PASS:
  rejected and counted under `stage1c.qc_empty_rejects` (negative mode)
  or dropped from the avoid tier (positive mode).
- **`ACTIVE`** — `f_c` fires above `extraction_threshold` at ≥1
  continuation-token position. QC reject.
- **`INACTIVE`** — `f_c` does not fire at any continuation-token
  position. QC accept (strict-zero PASS).

**Distinction from legacy `fc_active_on_continuation`.** The removed
`fc_active_on_continuation` (which returned a plain `bool`) returned
`False` for both `NO_TOKENS` and `INACTIVE`, so callers could not
distinguish the two. The tri-state makes the difference explicit and
enforces correct rejection of degenerate continuations (issue #518).

Records the `synth_qc.fc_active_checks` counter on `ACTIVE`/`INACTIVE`
outcomes (not on `NO_TOKENS`, since the SAE is not run).
After an SAE-backed `ACTIVE`/`INACTIVE` check, the helper releases the
activation tensor and cached CUDA allocator blocks before returning the
tri-state result.

Edge cases:

- Empty/whitespace continuation — Stage 1c's LLM path rejects this in the
  authenticated parser (counted
  `stage1c.continuation_protocol_failures`) before the classifier is called.
  The classifier's `NO_TOKENS` outcome remains reachable for the seam-merge
  case: a non-whitespace parsed string whose BPE tokens all merge into the
  prefix, contributing zero continuation-only positions. That case counts
  under `stage1c.qc_empty_rejects`. Neither is a strict-zero PASS.
- Empty prefix + continuation — raises `AxBenchBaseError` via
  `_tokenize_prefix_and_full` (nothing to analyze).

**Legacy reference.** The original boolean predicate
`fc_active_on_continuation` (removed from production `qc.py` by
issue #518) is preserved verbatim in
`tests/axbench_base/unit/synth_qc_reference.py` as the independent
reference implementation that the Hypothesis equivalence property
`test_classify_equivalent_to_old_two_function_combo` differentially
pins `classify_avoid_continuation` against. It is NOT production code.

### `compute_fc_percentiles` and `output_fc_percentile_for_continuation`

`synth/qc.py`. `compute_fc_percentiles` preserves the diagnostic
`(joined_prefix_percentile, output_percentile)` API. Positive Stage 1b calls
`output_fc_percentile_for_continuation`, which executes the same joined
tokenization/SAE path and returns the exact same output percentile without
reducing the prefix slice. Stage 1b subtracts its separately retained
prefix-alone canonical baseline.

Edge cases:

- Empty prefix → `seed_percentile = 0.0`
- Empty continuation → `output_percentile = 0.0`
- Both empty → raises (tokenization stage)

After the SAE forward, this helper releases the full activation tensor
before returning the scalar percentiles.

### `fc_percentile_on_prefix`

`synth/qc.py`. Max-over-tokens percentile of `f_c` on the
prefix alone, used by Stage 1b's scalar canonical scoring path. Empty text or
empty tokenization is a corpus invariant violation and raises; an inactive
feature in a non-empty prefix scores `0.0`. Oversized prepared model input also
raises rather than being front-clamped. SAE-backed calls release the activation
tensor before returning the scalar percentile.

### `PrefixPercentileCache`

`synth/qc.py`. Stage 1b's seed scorer owns one cache per
`synthesize_positive_mode` call. Keys include the exact BOS-prepended
extraction token IDs, feature ID, base-model identity/context window, SAE identity,
SAE layer, extraction tokenizer ID, and percentile namespace. Hits
return the stored percentile directly; misses are filled in bounded
batches through the same `_run_batched_extractor` path used by the
single-prefix helper. The cache is intentionally in-memory and
stage-scoped, so changing run config or restarting the process cannot
reuse stale QC values. A prefix exceeding `n_ctx` fails loudly; no truncated
input is cached or scored. Each miss-fill batch releases its transient
activation tensors before the next batch starts.

### `_max_percentile`

`synth/qc.py`. The shared helper that converts a 1-D activation
tensor to a percentile. Empty input → `0.0`. Single-element lookup
against `percentile_table` via `magnitude_to_percentile`. Expensive
path — the percentile table load is LRU-cached; the per-call math is
trivial.

### The percentile table

`build_percentile_table_from_stats(stats_path, n_sae_features)`,
`extraction.py:100-112`. LRU-cached lookup builder keyed on
`(stats_path, n_sae_features)`. Loads `per_feature_stats` once and
calls `build_percentile_table` from `steering.conditional.percentile_table`
to produce a `(n_sae_features, 99)` float tensor. Each row holds the
1st through 99th percentile boundaries for one feature's magnitude
distribution. `magnitude_to_percentile` does bisect-right lookups
against these boundaries.

The lookup is implemented as a direct-gather binary search with the same
bisect-right semantics as the reference: ties remain ties, the returned scale
is `[0.5, 99.5]`, and no random jitter is added. The differential percentile
suite pins this behavior and the resulting values exactly.

**Why 99 entries.** The underlying `build_percentile_table` puts the
99 evenly-spaced percentiles (0.5, 1.5, ..., 99.5) — mapping raw
magnitudes to a [0.5, 99.5] percentile scale. The Stage 1b
`accept_threshold` and Stage 0c `verification_percentile_threshold` configs
are fractions in `[0,1]`, while both QC paths receive raw percentile points.
The shared conversion multiplies either fraction by 100 before comparison.
Thus the Stage 1b default `accept_threshold=0.3` means an inclusive 30-point
delta, while Stage 0c's default `verification_percentile_threshold=0.7` means
a strict `> 70`-point absolute gate. The lookup values remain unquantized;
neither path uses the uint8 training-artifact representation.

**No noise hack.** There is explicitly **no** random jitter added to
break ties in the percentile lookup. Ties are resolved by the underlying
`bisect_right` order.

## LLM response cache

`synth/llm_response_cache.py`. SQLite-backed, one file per run. Keyed on
a caller-provided `cache_key` string, which is typically
`make_cache_key(model, prompt, tag)`. The cache lives for the lifetime
of a stage but commits on every successful write, so a crash mid-stage
leaves earlier responses queryable.

### Schema

```sql
CREATE TABLE cache (
  key TEXT PRIMARY KEY,
  response TEXT NOT NULL,
  created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
)
```

`llm_response_cache.py:27-41`. Nothing fancy — UPSERT on `key`, SELECT
on get, COUNT(*) on len, `IN SELECT` on `__contains__`. Per-write commit
gives crash safety at the cost of per-transaction fsync latency.

### `make_cache_key`

`llm_response_cache.py:80-95`. Three components:

```python
payload = f"{model}\n---\n{prompt}\n---\n{tag}"
return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- `model` — OpenRouter model identifier. Disambiguates runs at two
  different models.
- `prompt` — the full user prompt content. Same model on different
  prompts can't collide.
- `tag` — caller-supplied discriminator. Typical format:
  `f"pos_c{concept_id}_s{seed_idx}_a{attempt_idx}"`. The attempt index
  is **critical** — retries need fresh responses, and if `a{attempt}`
  is omitted the retry loop will hit the failing first attempt's cache
  entry forever.

### Known edge case: separator collision

The `\n---\n` separator is a literal. If you jam `"\n---\n"` into any
of `model`, `prompt`, or `tag`, you can construct a collision:
`make_cache_key("m", "p\n---\nq", "tag")` ==
`make_cache_key("m", "p", "q\n---\ntag")`. This is pinned by a test
(`test_llm_response_cache.py`); if you want to harden the key (length-
prefix, JSON encoding) that test will break and tell you loudly so you
can update every downstream call site in one go. For now, no caller
constructs tags from untrusted input, so the collision is academic.

### Storage footprint

Each entry is (key: 64 chars, response: ~500-5000 chars, created_at: 8
bytes). Typical full research run (500 concepts x 72 synth positives x
3 retries + 500 x 30 x 2 negative-synth + visited examples-QC judges) tops out at
~130k entries, ~200 MB of SQLite. Dev tier is < 1 MB.

## LLM-client retry semantics

The OpenRouter clients (`OpenRouterLLMClient` and
`AxBenchJudgeClient`) retry on the following failure modes, bounded by
`cfg.synth.max_retries` / `cfg.judge.parse_max_retries`:

* `requests.RequestException` (network errors, connect/read timeouts).
* HTTP 429 / 5xx responses.
* `ValueError` from `resp.json()` (malformed JSON body).
* `KeyError` / `IndexError` / `TypeError` from extracting the
  message content (response shape mismatch).

Every retry uses exponential backoff (`2 ** attempt` seconds).
Connections are routed through `get_openrouter_session()` in
`openrouter.py`, which sets `Connection: close` to defeat urllib3
stale-keepalive hangs and uses an explicit `timeout=(10, 60)` tuple.

## Synth batch wall-clock ceiling

`complete_synth_requests` enforces a 1800-second batch-level timeout
via `_BATCH_TIMEOUT_SECONDS`. When the timeout fires it cancels every
pending future, shuts down the executor non-blockingly, emits the
`synth.batch_timeout_count` profiler counter, logs a warning naming
completed-vs-total, and raises `AxBenchBaseError`. The 30-minute
ceiling is calibrated to comfortably exceed the worst-case 25-minute
1500-request batch when synth traffic is explicitly limited to 60 RPM, while
the shipped synth ceiling is now 6000 RPM. It still bounds the failure case where a
single stuck socket would otherwise hang the entire stage indefinitely.

## OpenRouter client

`llm_judge_client.py`. `OpenRouterLLMClient` is the synchronous
minimum-viable client used by Stage 0c concept-examples QC and any
other stage-boundary LLM call. Rate-limits to
`cfg.requests_per_minute` by sleeping between calls
(`_rate_limit`, `llm_judge_client.py:179`). Retries on 429 and 5xx
up to `max_retries` times with exponential backoff
(`_backoff`, lines 177-182).
Synth callers pass `cfg.synth.transport_max_retries` to this transport
parameter. Their outer Stage 0c/1 content/QC loops instead use
`cfg.max_retries`, generate an attempt-specific prompt/cache key, and do not
sleep between rejected model responses.

The default Stage 0c API continuation source reuses
`synth/openrouter_client.py::OpenRouterSynthClient` and
`synth/batching.py::complete_synth_requests`, which provide cached responses,
thread-safe sliding-window admission, concurrent ordered batches, transport
retries, and a batch wall-clock ceiling. It replaces the client's model with
`verification_api_model`; the ordinary `synth.model` remains authoritative for
Stages 1b/1c only. `verification_api_prefix_wave_size` bounds candidate
prefixes per API wave, while candidate-attempt/pool and zero-hit-bail limits
may reduce it. Each candidate expands to `n_verification_seeds` calls and
`synth.max_concurrent` bounds the simultaneously executing subset.

### Content validation

After `response.json()["choices"][0]["message"]["content"]` is
extracted, the client explicitly checks:

- The nested key access doesn't raise `KeyError` / `IndexError` /
  `TypeError` → re-raise as `AxBenchBaseError`
  (`llm_judge_client.py:139-144`).
- `content is not None` → otherwise raise (OpenRouter returns null for
  rejected completions; we want to surface that clearly)
  (`llm_judge_client.py:145-148`).
- `isinstance(content, str)` → otherwise raise (future API versions
  might return a dict for structured output; we don't handle that
  silently) (`llm_judge_client.py:149-153`).

Only after all three pass does the successful response get cached.
Failed responses are **never** cached, so the retry budget applies to
genuinely failing calls and cached replays are always known-good.

### JudgeConfig inheritance

`JudgeConfig(LLMClientConfig)` adds three fields on top of the base:
`parse_max_retries`, `invalid_warning_threshold`,
`invalid_drop_threshold` (`config.py:36-42`). These are used by the
Stage 5 judge parser and aggregation. The inherited `max_concurrent`
now drives Stage 5's batched `AxBenchJudgeClient.score_many()` worker
pool, while `requests_per_minute` is enforced by a thread-safe
sliding-window limiter across `complete()` admissions. Transport retries issue
additional HTTP attempts within the already admitted logical call. The
inheritance means synth and judge configs share the
plumbing (rate limit, model id, concurrency) but can diverge on
judge-specific plumbing without duplicating fields.

## Model metadata detection

`model_detection.py:37-93`. Called once per run before any stage that
runs the SAE. Returns a `TargetModelMetadata`.

Encoder-decoder branch: reads
`model.encoder.config.num_buckets` and `.n_sae_features` (the target
encoder knows its own training configuration). Reads
`model._loaded_meta_data["user_config"]["negative_selection"]` and
`["feature_selection"]` for the training-time feature selection
settings. **Raises `AxBenchBaseError`** if `_loaded_meta_data` is
missing — the caller must attach it after `build_encdec_model()`
returns. See the assertion at `model_detection.py:64-72`.

Decoder-only branch: every encoder-specific field is `None`; only
`device`, `sae_layer`, `segment_size` are populated.

Both branches populate `sae_layer` and `segment_size` from
`parse_feature_dir(cfg.feature_dir)`. There is no hardcoded default
for these.

## Paths and run_tag

`paths.py` defines the frozen `AxBenchBasePaths` contract used by every stage.
`output_root=null` resolves to the historical
`get_base_dir() / "eval" / "axbench_base"`; a non-empty configured root is
honored literally. Run outputs resolve under
`effective_output_root / _sanitize(run_tag)`.

The Neuronpedia description cache is deliberately cross-run and lives at
`effective_output_root / "_neuronpedia_cache" / "v1" /
"<identity_sha256>.yaml"`. The full digest is over canonical JSON containing
the exact `model_id` and `sae_source`; the document also validates those
fields when loaded. Same-identity processes serialize read-merge-write with
`flock`, preserve disjoint entries, accept identical overlap, and reject
conflicting overlap. Pending rows are published to memory only after the
temporary file is flushed and fsynced, atomically replaces the target, and
the parent directory is fsynced.
Namespaced documents require exact non-negative integer feature IDs and
non-empty string descriptions; malformed values are never coerced.

After distributed initialization, DDP ranks collectively compare configured
and effective roots. Rank 0 broadcasts autogenerated run tags; explicit tags
must already match exactly. The historical-output shadow preflight is likewise
a rank-0 decision broadcast to every rank before collective artifact work.

### Sanitization

`_sanitize(run_tag)` (`paths.py:19-39`):

1. If `run_tag` contains `/`, `\`, or `..`, return
   `f"traversal_{sha256(run_tag)[:16]}"`. This prevents a caller from
   passing a traversal string like `../../../etc/passwd` and writing
   outside the output directory.
2. Otherwise replace characters outside `[A-Za-z0-9_\-.]` with `_`.
3. If the result is empty or only dots, return `"_"`.

### Run tag autogeneration

The CLI `_run` in `scripts/eval/axbench_base_eval.py` (`_generate_run_tag`
at lines 183-206) generates
`<YYYYMMDD-HHMMSS>-<8char-hex>` in UTC when `cfg.run_tag is None`. The
hex suffix is the first 8 characters of
`sha256(json.dumps(_run_tag_config_to_dict(cfg), sort_keys=True))`, so two
successive runs with the same config content in the same UTC second
share a tag; runs with different serialized identities do not. The null
default root is normalized to the former literal
`"out/eval/axbench_base"` only for this hash, preserving historical default
tags. Explicit Python
callers that build a config manually must set `cfg.run_tag` before
constructing a `RuntimeState` — every accessor that resolves a
run-tag-scoped path raises from `_require_run_tag`.

### Resuming a partial run

Re-run the same command. `check_stage_dependencies` (`dag.py`) walks
the declared DAG, no-ops any stage whose outputs are already on disk,
and otherwise raises a multi-line `AxBenchBaseError` listing every
missing artifact. Stage 4's per-concept dep check
(`dag.py:209-297`) enumerates every
`(concept, target, mode)` **in the final filtered Stage 1 population** so an
operator gets a precise
list of which baseline artifacts to re-train. The
`SynthResponseCache` and `AxBenchJudgeClient`
SQLite caches survive across runs with the same tag, so re-running
expensive stages replays previously-paid LLM calls and GPU forwards
for free.

## Determinism

Per-stage determinism:

| Stage | Deterministic? | Dependencies |
|-------|---------------|--------------|
| 0a | Yes, after Neuronpedia cache warmup | `concept_list_seed` + Neuronpedia cache |
| 0b | Yes, after rank/canary agreement | Exact source hashes and candidate digest; tokenizer/model/SAE/runtime identities; fixed extraction grouping; `seed_pool_seed` |
| 0c | No | Null mode retains per-seed base-SAE-LM sampling and possible hardware/CUDA nondeterminism. API mode depends on external sampled responses. Both API-mode evidence and response bodies are immediately cached, so a warm-cache replay is stable for the same fingerprint, but a cold provider call is a new input. |
| 1b | No | OpenRouter LLM sampling |
| 1c | No | OpenRouter LLM sampling |
| 2 canonical | Yes | No RNG |
| 2 delta_ranked | Yes | Gemma + SAE forward is deterministic under fixed torch seeds on CPU, non-deterministic on CUDA |
| 3e | Yes | Seed fingerprint, prompting-description hashes, method/config fingerprint, tokenizer, and encoder-decoder sequence length |

### Seed partitioning

All RNG flows from `cfg.generation.master_seed` with a set of
offsets:

- `_CONCEPT_SEED_OFFSET = 0xE70001` (Stage 1b)
- `_VERIFY_SEED_OFFSET = 0xE80001` (Stage 0c)
- `_NEG_SYNTH_SEED_OFFSET = 0xE90001` (Stage 1c)

Each offset is a magic number picked to be well-separated from the
others (the specific values are arbitrary but fixed). Within a stage,
per-concept stride is typically `concept_id * 10_000` (Stage 0c) or
`concept_id * 1_000` (Stage 1b) or `concept_id * 1_000_000` (Stage 1c),
and per-vseed stride is `vseed_idx * 100` (Stage 0c). Stage 1b deliberately
retains its historical `concept_id * 1_000 + seed_idx` formula for exact row
provenance compatibility. Replacement candidates can have `seed_idx >= 1000`,
so a Stage 1b `gen_seed` is deterministic provenance but is not globally
unique and must never be used as row identity. The wider Stage 0c/1c strides
retain their existing non-collision assumptions for their bounded coordinate
ranges.

Stage 2 / Stage 4 seed derivation lives in `seeds.py`. The key helper
is `_compute_method_sampling_seed`, which includes `method_name` so each
Stage 4 method has a deterministic, method-specific sampling stream for
the same `(eval_seed_index, concept_id)`.

## Runtime state and lazy loading

`RuntimeState` (`steering/eval/axbench_base/runtime_state.py`, class
starts around line 694)
is the single-instance, main-thread-only object that owns every
heavyweight resource for a single CLI invocation. The CLI constructs
one per run in `_run` and tears it down in a `try/finally`:

```python
state = RuntimeState(cfg=cfg_obj)
try:
    _dispatch_stage(cfg_obj, state)
finally:
    state.close()
```

### Why it exists

Stages need resources that cannot be reconstructed from `cfg` alone:
base Gemma + SAE weights (network + GPU), two nanochat target models
(disk + GPU), `TargetModelMetadata` detected once per run, the OpenRouter
judge client, and the inactive-feature sampler derived
from the encdec's training-time negative-selection config. A pre-CLI
design would either load all of these at startup (one stage pays the
cost for all; OOM on shared GPUs) or re-load them per stage (slow and
redundant). `RuntimeState` lands in the middle: lazy-load on first
access, cache forever, CPU-offload between uses.

### Tiered resources

Four tiers, each with its own lifecycle:

1. **Pure data** (`concepts()`, `seeds_train()`, `seeds_eval()`). Loaded once
   and kept resident, backed by the validated safetensors files Stage 0
   produces. The train cache stores its rows and bulk row digest as one
   snapshot, so `seeds_train()` and `seeds_train_rows_sha256()` share exactly
   one file load. The eval cache remains row-only.

2. **Tokenizers** (`nanochat_tokenizer()`, `extraction_tokenizer()`).
   CPU-cheap, resident once loaded. Neither has a refcount because
   they cannot meaningfully "offload". Stage 1b/1c boundary helpers
   additionally use the process-local `tokenizer_cache.py` wrapper; this is
   distinct from the `RuntimeState` accessor and from the Gemma
   extraction-tokenizer cache.

3. **Heavy models** (`base_model()`, `base_sae()`,
   `decoder_only_model()`, `encdec_model()`). Refcounted Python
   context managers. On first entry the loader is invoked and the
   model is placed on `cfg.device`; on re-entry after a full exit the
   cached instance is moved back from CPU via `.to(self._device)`; on
   final exit (refcount = 0) the model is parked on CPU and
   `torch.cuda.empty_cache()` is called.

4. **Derived metadata, client, inactive sampler** (`decoder_only_meta()`,
   `encdec_meta()`, `judge_client()`, `inactive_sampler()`). The judge client is
   SQLite-backed and lives for the full run; the inactive sampler is a pure
   callable built lazily from the encdec metadata. Stage-owned synthesis caches
   and Stage 4 generation wrappers are constructed at their live call sites.

### Context-manager refcounting

Every heavy-model accessor is a `contextlib.contextmanager`
(`runtime_state.py`, heavy-model context managers around lines
1039-1172). The refcount handles nesting:

```
state.base_model() is entered:  refcount 0 -> 1, load weights, .to(cuda)
  state.base_model() is entered again inside: refcount 1 -> 2, no-op
  inner block exits:                           refcount 2 -> 1, no-op
outer block exits:                             refcount 1 -> 0, .to(cpu)
```

Nested `with` blocks share the single loaded instance and the model
stays live for the entire outer scope. Stage 4 enters the decoder-only and/or
encoder-decoder model contexts selected by its enabled methods; it does not
enter an unused target family.

### CPU-offload strategy

Simple `.to("cpu")` / `.to(self._device)` round-trips between stage
boundaries. No meta-device tricks, no state-dict caching, no
reloading from disk. This costs a few seconds per boundary (the
"tier thrash") but keeps the pattern transparent and mirrors how the
legacy per-script harnesses managed GPU memory.

### Target metadata

`decoder_only_meta()` / `encdec_meta()` briefly enter the
corresponding model context on first call so
`detect_target_model_metadata` can read encoder attributes off a live
model, then cache the resulting frozen dataclass. Later calls return the cached
instance without touching the model at all. Stage 4 constructs each generation
wrapper from the selected already-live target model at its call site.

### Thread safety

Not thread-safe. Every accessor asserts
`threading.current_thread() is threading.main_thread()` and raises
`AxBenchBaseError` otherwise (`runtime_state.py`,
`_check_main_thread` method). Refcount
mutations and lazy-load "check then set" sequences are not atomic,
and the CLI dispatcher runs every stage on the main thread
sequentially. If a future contributor introduces distributed or
parallel stage execution, they must either wrap every refcount
mutation in a lock or move the whole design to an async
context-variable-based model.

### Close protocol

`close()` (`runtime_state.py`, around line 826) is idempotent and
refuses to
run while any heavy-model refcount is non-zero:

```python
if active_refcounts:
    raise AxBenchBaseError(
        "RuntimeState.close() called with active model contexts: "
        f"{active_refcounts}. Exit the with-block(s) before "
        "closing the runtime state."
    )
```

This catches control-flow bugs where a stage escapes its `with` block
via a bare `return` or raised exception without unwinding the
refcount. The `_closed` flag is NOT set when this raises, so callers
can fix the caller, close their contexts, and re-invoke `close()`
successfully. After a successful close every accessor raises from
`_check_not_closed`.

### `validate_env_for_stages`

`runtime_state.py` (`validate_env_for_stages`, around line 143) is a
CLI pre-flight check. Called from
`_run` before `RuntimeState` is constructed, it walks the resolved
stage list, collects the stages that need `OPENROUTER_API_KEY` and
`NEURONPEDIA_API_KEY`, and raises `AxBenchBaseError` naming the
missing env var and every triggering stage. Stage 0a always needs
Neuronpedia for descriptions. Stage 0c needs OpenRouter and
Neuronpedia when examples QC is enabled or
`cfg.verification_api_model` is non-null. This
fails loud before the first 30-second Gemma download, so operators
fix their env, not their patience.

## CLI dispatch (`_run_single_stage`)

`scripts/eval/axbench_base_eval.py` (`_run_single_stage` at lines
298-515). A 14-branch
`if stage_enum is Stage.X` ladder keyed on the `Stage` `StrEnum`. The
caller has already expanded `Stage.ALL` via `_resolve_stage_list` and
run `check_stage_dependencies` against the per-stage dep set, so
this function assumes the prerequisites exist on disk.

### Data-path branches

Stages 0a (`prepare_concept_list`), 0b (`prepare_seed_pools`), and 6
(`aggregate`) take only `cfg` — no runtime state. They dispatch at the top
of the function and return before any model load.

### Runtime-state branches

Every other stage opens one or more `with` blocks against
`state.base_model()`, `state.base_sae()`,
`state.decoder_only_model()`, `state.encdec_model()`, or the engine
variants. The refcount pattern documented above ensures nested
contexts share live models. The branch bodies:

- **Stage 0c (`PREPARE_NEGATIVE_PREFIXES`)** — nests
  `base_model + base_sae` so the verification loop can generate with
  base Gemma while extracting SAE activations from the same model. It
  parses `cfg.feature_dir` for `sae_layer` / `segment_size` and does not
  enter either target-model context.
- **Stage 1b / 1c (`SYNTH_POSITIVE_MODE` / `SYNTH_NEGATIVE_MODE`)**
  — nests `base_model + base_sae` for QC. Stage 1c short-circuits
  when `cfg.run_negative_mode is False`, logging the skip and
  returning without entering either context.
- **Stage 2 (`CONCEPT_FEATURE_MAPPING`)** — dispatches on
  `cfg.concept_feature_mode`. Canonical mode runs
  `concept_feature_mapping_canonical` with zero model contexts.
  Delta-ranked mode nests `base_model + base_sae` and builds the
  positive / optional negative corpus paths before invoking
  `concept_feature_mapping_delta_ranked`.
- **Stages 3a / 3b / 3d
  (`TRAIN_DIFFMEAN` / `TRAIN_LINEAR_PROBE` / `TRAIN_LORA`)** — all
  dispatch through `_run_scalar_baseline_training`
  (`scripts/eval/axbench_base_eval.py`, `_run_scalar_baseline_training`
  at lines 233-295). The helper enters each
  target model context in turn, loops over `[POSITIVE, NEGATIVE]`
  (dropping negative when `run_negative_mode=False`), resolves the
  per-mode corpus path, and invokes the trainer. Stage 3d passes
  `targets=("decoder_only",)` because there is no LoRA variant for
  the encoder-decoder target.
- **Stage 3c (`TRAIN_PROMPTING`)** — no model context. Just writes
  `{positive,negative}_descriptions.json` from the concept list.
- **Stage 4 (`GENERATE`)** — resolves the target families required by the
  enabled method plan, enters only those model contexts, and passes their
  bundles into `generate_all`. Decoder-only-only plans neither load the
  encoder-decoder checkpoint nor require its context cache/inactive sampler;
  encoder-decoder-only plans do not load the decoder-only checkpoint. The
  default all-method plan still holds both targets during generation.
- **Stage 5 (`JUDGE`)** — no model context. Uses
  `state.judge_client()` (SQLite-backed OpenRouter client) and the
  concept list to dispatch `judge_all`.

### The `rng: random.Random` bridge

`_build_inactive_sampler` (`runtime_state.py`, around line 552)
takes
`encdec_meta` + `cfg` and returns a closure with the signature
`(*, rng: random.Random) -> dict[int, int]`. The orchestrator
(`build_mode_aware_encoder_input`) already threads a `random.Random`
through every partition modifier, but the underlying
`sample_inactive_features` expects a `torch.Generator`. The closure
bridges the two by pulling a fresh int64 seed from the incoming
`random.Random` on every call and seeding a fresh `torch.Generator`
with it (`_TORCH_SEED_UPPER_BOUND = (1 << 63) - 1`). This keeps the
orchestrator's RNG API unchanged while producing a deterministic
sequence of inactive samples for a fixed `random.Random` seed. The
`active_feature_ids` argument is passed as an empty int64 tensor
because the orchestrator signature does not forward per-segment
active fids, and the downstream `collision_filter` already handles
the rare case where a sampled inactive fid collides with an active
one. See Phase 7 of the runtime-state PR for the full rationale.

## Memory discipline

The pipeline is designed to run on a single 24 GB consumer GPU with
Gemma-2b + GemmaScope-res-16k + d12 target models. Peak usage is an
**estimated ~13 GB** (calculated from component sizes in the
["Budget breakdown"](#budget-breakdown) section below; not measured
end-to-end) when both target models are live for Stage 4. Stage 0c is
now a feature-extraction-tier stage only: `base_model + base_sae`.

### Per-tier live windows

| Stage    | base_model | base_sae | decoder_only | encdec | Peak |
|----------|------------|----------|--------------|--------|------|
| 0a       |            |          |              |        | 0   |
| 0b       | live       | live     |              |        | gemma+sae |
| 0c       | live       | live     |              |        | gemma+sae |
| 1b       | live       | live     |              |        | gemma+sae |
| 1c       | live       | live     |              |        | gemma+sae |
| 2 canon  |            |          |              |        | 0   |
| 2 delta  | live       | live     |              |        | gemma+sae |
| 3a       |            |          | live (per mode) | live (per mode) | d or e |
| 3b       |            |          | live (per mode) | live (per mode) | d or e |
| 3c       |            |          |              |        | 0   |
| 3d       |            |          | live (per mode) |        | d |
| 4        |            |          | enabled      | enabled | d, e, or d+e |
| 5        |            |          |              |        | 0   |
| 6        |            |          |              |        | 0   |

`d` = decoder-only target weights, `e` = encdec target weights,
`gemma+sae` = base Gemma + SAE together. The CPU-offload pattern
guarantees that every row above is the actual live footprint; the
models not marked "live" are parked on CPU with refcount 0.

### Tier-swap cost

Every stage boundary that transitions from one tier to another pays a
~4 s cost to CPU-offload the previous tier and wake the next tier
via `.to(self._device)`. For 14 stages in a full `stage=all` run that
is ~60 s of thrash total — negligible next to the multi-hour synth /
generation work. We accept this cost in exchange for never holding
Gemma + both target models simultaneously (which would exceed the
~24 GB envelope).

An alternative design would group stages by tier (all Gemma+SAE
stages first, then all target-model stages) so each tier is only woken
once. This was considered and rejected: the synth stages (1b / 1c)
need the base Gemma+SAE for QC, and Stage 2 delta-ranked mode needs
it again, but Stage 2 canonical mode does not — the DAG is irregular
enough that tier-grouping would re-order stages in non-obvious ways
and break the "re-run the same command to resume" contract.

### Budget breakdown {#budget-breakdown}

With `gemma-2-2b` (bf16, ~4 GB) + `gemmascope-res-16k` (~1 GB) +
`d12_base` decoder-only (~4 GB) + `d12_baseline` encdec (~4 GB), the
peak stage is Stage 4
at `d + e = ~8 GB`. Stage 0c is now `gemma + sae = ~5 GB`. Add ~4 GB
of PyTorch allocator overhead + activations + KV cache and the
estimated working envelope is ~13 GB. **This figure is calculated
from component sizes, not measured end-to-end**; real usage may
vary with PyTorch allocator behavior, the batch size you configure,
and whether the inference engine's KV cache is warmed. Use
`device=cpu` for smaller boxes at ~20x runtime cost, or narrow to
`tiers=dev` to reduce per-concept cost if the bottleneck is elapsed
time.

## Testing the benchmark

Tests live under `tests/axbench_base/`:

```
tests/axbench_base/
|-- __init__.py
|-- conftest.py                 # shared fixtures
|-- fixtures/
|   |-- __init__.py
|   |-- build_tiny_shard.py     # tiny parquet + feature shard pair
|   |-- offline_openrouter.py   # fail-closed Session.post transport + diagnostics
|   `-- real_gemma_offline.py   # local Gemma/SAE protocol/QC oracle
|-- unit/                       # mock-based, fast
|   |-- test_stage0c_api_generation.py     # protocol/evidence/cache/parity/adversarial matrix
|   `-- ...                     # one file per module under test
|-- integration/                # real stage functions with tiny resources
|   |-- test_stage0c_api_generation_live.py # @pytest.mark.api, real NP/OpenRouter/tokenizer
|   |-- test_full_pipeline_dev_tier.py      # tier 1 @pytest.mark.slow
|   |-- test_full_pipeline_real_gemma.py    # tier 2 @pytest.mark.very_slow
|   |-- test_issue_1049_real_gpu.py         # 50-concept L22/f65k QC audit
|   |-- test_stage_0_prepare.py
|   |-- test_stage_1_synth.py
|   |-- test_stage_2_concept_map.py
|   |-- test_stage_3_training.py
|   |-- test_stage_4_generation.py
|   |-- test_stage_4_resume.py
|   |-- test_stage_5_judge.py
|   `-- test_stage_6_aggregate.py
`-- property/                   # property-based tests
```

### Three tiers of coverage

- **Unit tests.** Mock-based, sub-second total, one test file per
  module. Every external touchpoint (Neuronpedia, OpenRouter, base
  Gemma + SAE loader, nanochat tokenizer, `InferenceEngine`) is stubbed
  via `monkeypatch.setattr` against a thin `_get_*` or `_build_*`
  wrapper. No network, no GPU.

- **Functional integration.** Run the real stage entrypoints against
  tiny on-disk resources built by `fixtures/build_tiny_shard.py` and
  heavy-model stubs injected at the `runtime_state._load_*` module
  boundary. Covers every stage individually.

- **Live API contract: `@pytest.mark.api`.** The Stage 0c API test fetches
  real Neuronpedia activation examples, uses the real extraction tokenizer,
  submits concurrent OpenRouter continuations for a hostile Unicode prefix
  containing instruction and protocol-like text, and requires both responses
  to parse to the configured token count without boilerplate or marker
  leakage. It is excluded from the default suite because it incurs network
  traffic, provider variability, latency, and cost.

- **Integration tier 1: `@pytest.mark.slow`.** The
  `test_full_pipeline_dev_tier.py` suite loads a real tiny nanochat
  checkpoint, stubs the Gemma+SAE loader with a small in-memory
  stand-in, mocks the OpenRouter wire layer, and runs every stage end
  to end through the CLI's `_run`. Total wall-clock is well under one
  second per assertion. Assertions check real artifact files, real
  `RuntimeState.close()` cleanup, and real `check_stage_dependencies`
  behavior.

- **Integration tier 2: `@pytest.mark.very_slow`.** The
  `test_full_pipeline_real_gemma.py` suite loads the real
  `google/gemma-2-2b` via `transformers` + the real
  `gemma-scope-2b-pt-res-canonical` SAE + a real nanochat `d12_base`
  decoder-only + real nanochat `d12_baseline` encdec, then runs a 5-concept
  dev-tier `stage=all` pipeline with a fail-closed mock at the production
  cached `requests.Session.post` boundary. The local oracle produces and
  validates nonce-framed Stage 0c definitions/continuations, strict-QC-passing
  Stage 1 rich/avoid samples, and parseable Stage 5 ratings. The test injects
  a dummy OpenRouter key only for runtime validation; it does not require a
  provider credential. It remains OpenRouter-offline rather than universally
  network-free because Neuronpedia, HuggingFace, and nanochat cache access are
  real prerequisites. Fails loud with actionable error messages if any real
  prerequisite is missing. Takes
  5-20 minutes on CPU, 1-3 minutes with a GPU. Used as the final
  "does it actually work" gate before merging runtime changes.

- **Issue #1049 real-data gate: `@pytest.mark.very_slow` and
  `@pytest.mark.requires_feature_data`.**
  `test_issue_1049_real_gpu.py` loads the complete 23,953-row
  Gemma-3-4B-PT/GemmaScope-2 L22/f65k clustered Stage 0b train pool and all
  50 concepts. It independently reconstructs the 3,600 legacy-selected
  prefixes, scores them with the production batched CUDA path, runs corrected
  headroom selection over the complete pool, and asserts exact rejection /
  replacement accounting. Ten concepts additionally compare scalar against
  batched prefix scores, cluster against representative-only scores, and the
  new output-only oracle against the retained pair-scoring oracle.

### Running each tier

```bash
# Default: unit + functional tests only.
uv run pytest tests/axbench_base/ -q

# Include tier-1 integration (@pytest.mark.slow).
uv run pytest tests/axbench_base/ -q -m slow

# Tier-2 real-Gemma integration (requires nanochat/HuggingFace/Neuronpedia caches; no provider credential).
uv run pytest tests/axbench_base/integration/test_full_pipeline_real_gemma.py -v -m very_slow

# Issue #1049 full real-data CUDA audit.
AXBENCH_1049_REAL_ROOT=/path/to/nano-gpt-pretrain-steer \
  uv run pytest tests/axbench_base/integration/test_issue_1049_real_gpu.py \
  -v -m 'very_slow and requires_feature_data'

# Live Stage 0c Neuronpedia + OpenRouter protocol test (network + API cost).
uv run pytest tests/axbench_base/integration/test_stage0c_api_generation_live.py -v -m api

# Everything including both integration tiers.
uv run pytest tests/axbench_base/ -q -m 'slow or very_slow or (not slow and not very_slow)'
```

### Monkey-patch conventions

Every module that needs to reach an external system (Neuronpedia, the
base Gemma + SAE, the nanochat tokenizer, an `InferenceEngine`,
`AxBenchJudgeClient`) provides a thin `_get_*` or `_build_*` / `_load_*`
wrapper that tests override via `monkeypatch.setattr`. Grep for
`# Thin wrappers (monkey-patch targets for tests)` to find them. **Do
not** inline the external call; always wrap it.

### The tiny shard fixture

`tests/axbench_base/fixtures/build_tiny_shard.py` builds a paired source
parquet + feature shard. Stage 0b tests patch only source-parquet discovery and
run the real production `process_shard` math against a tiny model/SAE; Stage 0b
does not consume the paired legacy feature shard. The feature shard remains
useful for Stage 0c tests, whose contract intentionally scans source-document
features for candidate negative prefixes.

## Quick reference: where to look

| Question | File |
|---------|-------|
| What does config field X mean? | `config.py` |
| What fields does a SeedRow have? | `types.py` |
| What stages exist and what are their identifiers? | `types.py` (`Stage` StrEnum) |
| Where does file Y get written? | `paths.py` |
| How is concepts.json built? | `concept_list.py` |
| How is the seed pool sliced? | `seed_pools.py` |
| How are negative prefixes verified? | `negative_prefixes.py` |
| Where do Stage 1b paired avoiding rows come from? | `synth/positive_mode.py` |
| How is positive-mode QC done? | `synth/qc.py`, `synth/positive_mode.py` |
| How is negative-mode strict-zero QC done? | `synth/qc.py`, `synth/negative_mode.py` |
| How does the LLM cache work? | `synth/llm_response_cache.py` |
| How is the OpenRouter request retried? | `llm_judge_client.py` |
| How is the target model's metadata detected? | `model_detection.py` |
| How is the SAE layer parsed from feature_dir? | `feature_paths.py`, `model_detection.py` |
| How is Stage 4 feature extraction done? | `extraction.py` |
| How is the concept-feature mapping built? | `concept_feature_mapping.py` |
| How are coefficient grids derived? | `coefficient_grids.py` |
| How are sampling seeds derived? | `seeds.py` |
| How are DiffMean / LinearProbe / Prompting baselines trained? | `baselines/diffmean.py`, `baselines/linear_probe.py`, `baselines/prompting.py` |
| How is LoRA trained and saved? | `baselines/lora/training.py`, `baselines/lora/module.py`, `baselines/lora/loader.py` |
| How is Stage 4 generation dispatched? | `generation.py`, `method_registry.py`, `encoder_input.py` |
| How is the judge run? | `judge.py` |
| How is the leaderboard built? | `aggregation.py` |
| How does the CLI wire stages to resources? | `scripts/eval/axbench_base_eval.py` |
| How does the stage dependency DAG work? | `dag.py` |
| How do heavy models get loaded and offloaded? | `runtime_state.py` |
| How do the per-run SQLite caches live and close? | `synth/llm_response_cache.py`, `judge.py` |
