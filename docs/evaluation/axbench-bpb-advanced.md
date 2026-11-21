# AxBench-BPB: Advanced Reference

Internals, algorithmic detail, and developer reference for
`steering/eval/axbench_bpb/`. Read `docs/evaluation/axbench-bpb.md` first for the
user-facing overview; this document is for implementers, researchers,
and code reviewers.

## Background and research question

AxBench-Base measures steering quality via LLM-judged generation quality:
the steered model generates completions, an LLM judge rates their
concept-relevance and fluency on a 0-2 scale, and the judge scores are
aggregated into a leaderboard. This pipeline is high-signal but slow,
expensive (requires an OpenRouter API key and hundreds of thousands of
LLM-judge calls for the full tier), and non-deterministic (both the
generation temperature and the LLM judge introduce variance).

AxBench-BPB tests a complementary hypothesis: **if a steering intervention
is genuinely effective, it should assign better relative likelihood to
known-good target text than to same-concept avoidance text**. This is
verifiable without any sampling or API calls — the ground-truth synth
corpus produced by axbench-base Stages 1b and 1c is already on disk, and
teacher-forced BPB is a single forward pass per sample. Encoder-decoder
targets always receive full prompt/context conditioning.

The BPB signal is different from, not a substitute for, the judge signal.
BPB captures distributional shift in the target model's log-likelihood
space. Judge scoring captures surface-level concept presence and fluency
in generated text. A method can score well on one and poorly on the other.
Phase 15 of the design includes a deferred Spearman rank-correlation test
(`tests/axbench_bpb/slow/test_leaderboard_vs_axbench_base.py`) to
quantify how well the two leaderboards agree for the `ours_*` family.

**gemma-scope-2 sources.** AxBench-BPB inherits the axbench-base concept
corpus, so when that corpus was built against a gemma-scope-2 (v2) SAE the
published medium-tier Neuronpedia source string omits the tier (e.g.
`22-gemmascope-2-res-16k`) rather than using the v1 4-segment form. If
Neuronpedia does not yet host a requested v2 SAE, the client raises the typed
`NeuronpediaNotPublishedError` (`steering/eval/neuronpedia_client.py`) — catch
that, not a bare 404. The [feature-filter](../pipelines/feature-filter.md)
maps the same 404 to a `not_published` outcome excluded from its error-rate
abort. See
[`docs/pipelines/gemma-scope-2.md`](../pipelines/gemma-scope-2.md).

## Package layout

```
steering/eval/axbench_bpb/
├── types.py              # AxBenchBpbError, Stage, ScoringInputs, WorkUnit,
│                         #   BpbScoreRow, Intervention (re-exports SteeringMode)
├── paths.py              # OutputPaths — write-side path resolvers
├── source_paths.py       # SourcePaths — read-only resolvers anchored at cfg.source_run_dir
│                         #   validate_stage_0_3_artifacts checks all required inputs at startup
├── config.py             # AxBenchBpbConfig + ContextCacheConfig + AggregationConfig
│                         #   + validate_bpb_config
├── sample_loader.py      # parquet rows → ScoringInputs with stable sample_ids
├── tokenization.py       # BOS handling, left-truncation helpers
├── seg_map.py            # build_multi_segment_seg_map for axbench-bpb prefix inputs
├── context_cache.py      # Stage 1: prepare_context_cache orchestrator
├── intervention.py       # build_intervention + apply_intervention context manager
├── vocab_restriction.py  # loaded checkpoint V + ordered membership restriction
├── live_ours_vocab_contract.py
│                         # run-level live-ours score resume/aggregate identity
├── teacher_force.py      # score_sample (single) + score_triple (batched)
├── bpb_scoring.py        # Stage 2: bpb_score orchestrator + work-unit enumeration
├── bpb_cache.py          # rank-local SQLite schema + reconcile_cache
├── bpb_aggregation.py    # Stage 3: compute_per_tuple_rollup,
│                         #   compute_per_concept_rollup, compute_macro_rollup,
│                         #   write_leaderboard_csv
├── parquet_writer.py     # thin wrapper around axbench_base.parquet_writer.ParquetAppendWriter
└── runtime_state.py      # RuntimeState: lazy target models, base Gemma+SAE, tokenizers,
                          #   path resolvers. Refcounted CPU offload.
```

CLI entrypoint: `scripts/eval/axbench_bpb_eval.py` (mirrors
`scripts/eval/axbench_base_eval.py`).

Config files: `configs/eval/axbench_bpb/axbench_bpb_eval.yaml` (base)
and `configs/eval/axbench_bpb/tiers/{dev,standard,full}.yaml`.

## Data types

All types live in `steering/eval/axbench_bpb/types.py`.

**`AxBenchBpbError`** — single user-visible exception. All internal
helpers catch third-party exceptions (`SafetensorError`, `ArrowException`,
`FileNotFoundError`, etc.) and re-raise as `AxBenchBpbError` with a
diagnostic message.

**`Stage`** — `StrEnum` with values `prepare_context_cache`,
`bpb_score`, `bpb_aggregate`, `all`. `Stage.coerce`
raises `AxBenchBpbError` for unrecognized values.

**`ScoringInputs`** (frozen dataclass) — fully preprocessed representation
of one synth row, ready for the scoring kernel:
- `sample_id`: `sha256(concept_id||mode||content_class||prefix_text||output_text).hexdigest()[:16]`
- `prefix_hash`: `sha256(prefix_text).hexdigest()[:16]` — context-cache lookup key
- `prefix_tokens`: nanochat-tokenized, BOS-prepended
- `output_tokens`: nanochat-tokenized, no BOS
- `prefix_context_segments`: tuple of `ContextFeatureSegment` (SAE features per prefix segment)
- `output_byte_weights`: `(n_output,)` int64 tensor — UTF-8 byte length per output token
- `truncated`: `True` if the prefix was left-truncated to fit in context

**`WorkUnit`** (frozen dataclass) — `(method_name, concept_id, sample_id,
coefficient_str)`. The atomic unit of scoring; hash-partitioned across
DDP ranks; primary key of the SQLite done-set index.

**`BpbScoreRow`** (frozen dataclass) — one scored result: all `WorkUnit`
fields plus `mode`, `content_class`, `target`, `bpb_intervened`,
`n_output_tokens`, `n_output_bytes`, `truncated`.

**`Intervention`** (frozen dataclass) — parameterizes one scored
intervention. Use the named factory classmethods (`ours`,
`from_direction`, `lora`, `prompting`) rather than the constructor.
The `kind` field selects the active family; all other fields outside the
active family are `None`. Note: `__hash__` is explicitly disabled because
the `direction` tensor field cannot be hashed reliably.

## Stage walkthroughs

### Stage 1: `prepare_context_cache`

**Purpose**: Pre-extract per-segment SAE features for every unique
prefix text in the synth corpus. The context cache is a hard prerequisite
for Stages 2 and 3 — without it, `sample_loader.py` cannot populate
`ScoringInputs.prefix_context_segments`.

**Process**:

1. Read all `prefix_text` values from `synth_training_positive.parquet`
   and `synth_training_negative.parquet` via `SourcePaths`.
2. Deduplicate by computing `sha256(prefix_text).hexdigest()[:16]`
   (`prefix_hash`). Skip hashes already present in the manifest.
3. For each unique prefix:
   a. Tokenize with the Gemma tokenizer (`extraction_tokenizer_id`).
   b. Chunk into stride-`segment_size` segments (where `segment_size` is
      parsed from `feature_dir` by `parse_feature_dir`).
   c. Run base Gemma + SAE forward; extract per-segment top activations
      using the same `steering.extract_features` helpers used by the
      main extraction pipeline.
   d. Discretize activations into bucket indices via
      `steering.conditional.discretization.discretize_activations` using
      the percentile boundaries from `feature_statistics_path`.
   e. Compute per-segment `char_start, char_end` in BOS-inclusive
      coordinates (matches the extraction pipeline's output format).
   f. Serialize as a tuple of `ContextFeatureSegment` and write atomically
      to `context_cache/{prefix_hash[:2]}/{prefix_hash}.safetensors`.
4. Write `context_cache/manifest.json` once at the end (post-barrier on rank 0 under DDP, or at the end of the loop under single-process).

**Idempotency**: each `.safetensors` file is written via tmp-then-rename.
Re-running skips files already present unless `context_cache.force_rebuild=True`.

**DDP**: every rank participates in extraction. Prefixes are partitioned
round-robin by sorted ``prefix_hash``, so each rank handles approximately
``N/ws`` prefixes (the first ``N mod ws`` ranks get ``ceil(N/ws)``, the
rest get ``floor(N/ws)`` — deterministic + disjoint slices). After the
sharded extraction barrier, rank 0 reads back every per-prefix
safetensors file (sorted by hash for deterministic JSON output) and
writes the manifest via tmp + rename. The post-barrier ordering also
runs the tokenizer compatibility guards (checksum + BOS-length) on
rank 0 only — pre-extraction execution of these guards once deadlocked
NCCL teardown when an AttributeError on an HF tokenizer's BOS-id
attribute propagated through ``destroy_process_group`` while ranks 1+
were still spinning on the extraction barrier (2026-05-13 incident).
The fix narrows the deadlock class because by the time the guards run,
all extraction work is durable on disk.

**Cost**: ~1 base Gemma + SAE forward per prefix; under DDP with world-size ``ws``, each rank performs roughly ``N/ws`` forwards in parallel (vs. ``N`` serial forwards on rank 0 in the legacy design). The ``context_cache.extraction_batch_size`` config field is currently unused at the loop level (one forward per prefix) and is reserved for a future batched-extraction optimization. The cache does not depend on target models — switching target checkpoints does not invalidate it.

### Stage 2: `bpb_score`

**Purpose**: the main method × concept × sample × coefficient sweep.

**Process**:

1. Enforce the run-level live-`ours` vocabulary contract before inspecting
   cache skips (see Caching semantics below).
2. `reconcile_cache` restores the rank-local parquet/SQLite invariant.
3. Load the method registry; apply `enabled_methods` and `disabled_methods` filters.
4. For each method, enumerate the cartesian product of
   `concepts × method.mode.samples × method.coefficient_grid`.
5. Each `WorkUnit` is hash-partitioned: `is_mine(unit, rank, world_size)`.
   Ranks skip units they don't own.
6. For each owned work unit, check the SQLite done-set. Skip if present.
7. Call `build_intervention(method, concept, coefficient, ..., model=...)` to get an
   `Intervention` instance.
8. Collect all samples for this `(method, concept, coefficient)` triple
   and call `score_triple` with `device_batch_size` micro-batching.
9. For each resulting `BpbScoreRow`: write to the current parquet batch
   and insert into SQLite.

**Batching**: within a triple, the encoder-input payload is invariant
across samples. Samples are padded per micro-batch to `(T_max, n_seg_max,
F_max)`. Padding positions contribute zero loss and zero byte weight.

**Intra-triple batching detail**: decoder inputs are padded to `T_max`
within the micro-batch. Encoder inputs are padded to `(n_seg_max, F_max)`.
Per-sample BPB is accumulated separately by masking each sample's
`(loss, byte_weight)` contributions.

### Stage 3: `bpb_aggregate`

**Purpose**: compute the final leaderboard from the raw scored rows.

**Process** (implemented in `bpb_aggregation.py`):

1. **Per-tuple rollup** (`compute_per_tuple_rollup`): read the full
   `bpb_scores/` directory dataset across all ranks and group by
   `(method_name, concept_id, mode, coefficient_str, target)`. Split by
   `content_class` into rich and avoid sides. Compute per-side means
   (`mean_bpb_intervened`, counts).
   Derive:
   - `margin_intervened = mean_bpb_intervened_avoid − mean_bpb_intervened_rich`
   - `primary_margin = margin_intervened` for positive mode;
     `-margin_intervened` for negative mode (sign correction so
     "bigger is better" is uniform)
   - `primary_bpb = mean_bpb_intervened_rich` for positive mode or
     `mean_bpb_intervened_avoid` for negative mode

2. **Per-concept rollup** (`compute_per_concept_rollup`): group by
   `(method_name, concept_id, mode, target)`. Filter: exclude concepts
   where `n_rich < min_rows_per_side` or `n_avoid < min_rows_per_side`.
   Select `best_coef = argmax(primary_margin)`. Compute trapezoidal
   AUC over numeric coefficient grids (NaN for single-coefficient methods).
   Compute `truncation_rate = n_truncated / n_total`.

3. **Macro rollup** (`compute_macro_rollup`): group by
   `(method_name, mode, target)`. Compute mean-of-means over surviving
   concepts. Apply validity gates: warn if `mean_truncation_rate > 0.01`;
   drop from leaderboard if `> 0.05`.

4. **Write outputs**: `per_tuple.parquet`, `per_concept.parquet`,
   `macro.parquet`, `leaderboard.csv` (sorted descending by
   `primary_metric`), `report.yaml`, `aggregation_report.yaml`.

Only rank 0 runs this stage; all other ranks exit after the pre-aggregate
barrier.

## RuntimeState and CPU offload

`RuntimeState` (in `runtime_state.py`) is a lazy resource manager that
holds references to:

- base Gemma model + SAE (needed only by Stage 1)
- decoder-only target model (needed by Stage 2)
- encoder-decoder target model (needed by Stage 2)
- nanochat tokenizer and Gemma tokenizer
- `OutputPaths` and `SourcePaths` instances

Resources are loaded on first access via `@property` with a `None` guard.
Between stage boundaries, when `offload_between_stages=True`, models that
are not needed by the next stage are moved to CPU and de-referenced, so
the GPU memory budget does not accumulate across stages.

The base Gemma + SAE are needed only for Stage 1; they are offloaded
before Stage 2 begins. The target models are needed for Stage 2 and can
be offloaded before Stage 3 aggregation.

### Encdec attribute layout (RuntimeState)

`EncoderDecoderGPT` exposes its decoder `GPTConfig` as
`model.dec_config` and its encoder config as `model.enc_config`;
`model.config` does NOT exist on the encdec class.
`runtime_state.encdec()` reads `model.dec_config.sequence_len` to
populate `TargetModelBundle.max_seq_len`. The decoder-only path
continues to use `model.config.sequence_len` because nanochat's
`GPT` exposes `config` directly.

After loading the encdec, `runtime_state.encdec()` also attaches the
loaded meta dict to `model._loaded_meta_data` so
`detect_target_model_metadata` can read training-time
`negative_selection` / `feature_selection`.

`runtime_state.inactive_sampler()` short-circuits and caches `None` without
inspecting the model vocabulary when `negative_selection` is absent. When it
is present, the accessor reads the loaded checkpoint's persistent allowed-ID
buffer and passes it to the shared AxBench-Base sampler builder. The builder
validates that the buffer is a sorted, unique, non-empty, proper-subset int64
vocabulary matching the statistics/metadata SAE width, copies it once to an
independent contiguous CPU snapshot, draws the configured count, and selects
uniformly without replacement through a random permutation of that snapshot.
This avoids numeric-ID bias from sorted rejection-sampling deduplication. The
closure remains valid after `offload_models()` moves the model or its
persistent buffers. AxBench-Base's own caller omits this optional argument and
therefore retains its exact legacy full-universe behavior.

## Caching semantics

The axbench-bpb caching contract has three layers:

**Context cache (Stage 1 output)**:
- Durable artifact: `context_cache/{ab}/{hash}.safetensors`, written
  atomically via tmp-then-rename.
- Re-runs skip existing files. Force-rebuild via `force_rebuild=True`.
- Index: `context_cache/manifest.json`, written once at the end of Stage 1 (post-barrier on rank 0 under DDP, or at the end of the loop under single-process).
- Consumer: `sample_loader.py` looks up `prefix_hash` in the manifest.
  Missing entry raises `AxBenchBpbError`.

**BPB scores (Stage 2 output)**:
- **Parquet** is the durable artifact: `bpb_scores/rank_{R:03d}/batch_*.parquet`.
  Each batch is an atomic write.
- **SQLite** is the fast-path done-set index:
  `bpb_cache/rank_{R:03d}.sqlite`. One row per scored `WorkUnit`. The
  primary key is `(method_name, concept_id, sample_id, coefficient_str,
  target)`. Connection settings: `synchronous=FULL`, `journal_mode=WAL`.
  Every insert is a transaction + flush.
- **`reconcile_cache`** runs at the top of Stage 2:
  1. Read all parquet batch files in the rank directory.
  2. Read all keys from SQLite.
  3. If parquet has rows whose keys are not in SQLite: reseed SQLite
     from parquet (parquet is ahead of SQLite — a crash between parquet
     write and SQLite insert).
  4. If SQLite has keys with no matching parquet rows: drop those SQLite
     entries (SQLite was ahead of durable write — re-score those units).
  This restores the parquet-is-source-of-truth invariant before scoring
  begins.

- **Live-`ours` semantic identity** is run-level:
  `bpb_scores/.live_ours_vocab_contract` stores
  `axbench-bpb-live-ours-v1:<checkpoint_vocab_fingerprint>`. The affected
  set is derived from the registry as `baseline == ours` and not
  precomputed-bag-backed. Before reconciliation, the guard scans only the
  `method_name` column of durable shards across every rank when the marker is
  absent or stale. Existing affected rows require an exact marker match;
  missing/malformed/mismatched provenance raises with instructions to use a
  fresh `run_tag` or clear the old BPB score/cache directory. If no affected
  rows exist and an affected method is enabled, the expected marker is
  atomically written before its first row. Aggregation requires a well-formed
  marker whenever its already-loaded score table contains an affected method.

## DDP coordination

Two coordination points:

**During `prepare_context_cache`**: every rank loads base Gemma + SAE
and extracts its round-robin slice of prefixes
(``items[ddp_rank::ddp_world_size]`` over sorted prefix hashes). A
first ``dist.barrier()`` ensures every per-rank safetensors file is
durable on disk. Rank 0 then writes the manifest and runs the
tokenizer-compatibility guards. A second ``dist.barrier()`` holds
non-zero ranks until the manifest is visible. Then every rank reads
the manifest independently from a shared filesystem.

**Before `bpb_aggregate`**: all ranks barrier. Rank 0 runs the
aggregation; all other ranks exit.

**No cross-rank communication in the hot loop.** During `bpb_score`, each
rank:
- Reads the per-rank done-set from THIS rank's SQLite cache (read-only,
  no locking needed; no cross-rank reads).
- Enumerates the full work-unit space but filters to self-assigned units.
- Writes only to its own parquet shards and SQLite cache.

**Hash-partitioning**:
```python
def is_mine(unit: WorkUnit, rank: int, world_size: int) -> bool:
    h = int(hashlib.sha256(
        f"{unit.method_name}|{unit.concept_id}|{unit.sample_id}|{unit.coefficient_str}"
        .encode()
    ).hexdigest()[:8], 16)
    return h % world_size == rank
```

Hash-based (not contiguous) distributes heterogeneous coefficient grids
evenly. Resume under different `world_size` is safe but partial: each
rank reads only its own SQLite cache, so units that were scored by a
different rank under the prior partition are re-scored. Outputs remain
correct (deterministic scores; SQLite primary-key idempotency drops
duplicates) but GPU work is wasted in proportion to the partition-overlap
loss. Same-`world_size` resume preserves the full done-set with no waste.

## Parity matrix

The correctness of axbench-bpb rests on parity with existing, well-tested
implementations. Every critical invariant is tested against a reference.

| Axbench-bpb component | Reference implementation | Test file | Tolerance |
|------------------------|--------------------------|-----------|-----------|
| `build_intervention` (`ours_*`) | `axbench_base.intervention_dispatch._dispatch_ours` | `integration/test_axbench_base_dispatch_parity.py` | byte-exact |
| `build_intervention` (diffmean/probe) | axbench-base direction dispatch, including scaled and unscaled regimes | `integration/test_axbench_base_dispatch_parity.py` | exact kind, target model, direction layer, coefficient, and direction tensor |
| Direction hook (`apply_intervention`) | `axbench_base.baselines.hooks` | `functional/test_apply_intervention_contract.py` | byte-exact residual diff |
| LoRA attach/detach | `axbench_base.baselines.lora._attach` | `functional/test_lora_attach_detach.py` | byte-exact `state_dict` roundtrip |
| Seg-map (prefix char-level) | `multiseg_generator.build_multiseg_inputs` + `_build_row_segments_char_level` | `integration/test_training_dataloader_parity.py` | byte-exact |
| `build_segmented_feature_encoder_inputs` | Reused directly (not reimplemented) | Reuse | Reuse |
| Context cache per-prefix features | `steering.extract_features` on same prefix | `functional/test_context_cache_roundtrip.py`, `integration/test_stage0c_context_parity.py` | float tol on magnitudes; byte-exact on fids |
| BPB accumulation | `nanochat.loss_eval.evaluate_bpb` / `encoder_decoder.loss_eval.evaluate_encdec_bpb_conditioned` | `functional/test_score_sample_kernel.py` | 1e-6 fp32; 1e-3 relative bf16 |
| Batched scoring | Training dataloader padded batch shape | `functional/test_batched_scoring_parity.py` | Same as BPB |
| Leaderboard macro stats | Hand-computed from synthetic fixtures | `unit/test_aggregation_math.py` | 1e-9 fp64 |
| Leaderboard directional agreement | AxBench-Base LLM-judge leaderboard on same source | `slow/test_leaderboard_vs_axbench_base.py` | Spearman ρ ≥ 0.3 for `ours_*` (Phase 15, deferred) |

## Intervention dispatch details

### `build_intervention`

Dispatches per method family to produce an `Intervention` instance.
Reads direction/adapter files from `SourcePaths`. All direction loaders,
partition modifiers, and inactive samplers are **imported** from
`steering.eval.axbench_base` — never reimplemented.

DiffMean and linear-probe dispatch uses the AxBench-Base-owned direction
artifact validator for both filesystem paths and source-retained safetensors
bytes. The requested `(family, target_model, mode, concept_id, layer)` must
equal both the path and canonical artifact metadata, and the decoded direction
must be a non-empty finite floating vector. The configured layer is always a
target nanochat/encoder-decoder residual block, not a Gemma SAE layer. The
configuration preflight validates all selected coordinates before model load;
this loader-boundary check prevents internal callers from bypassing it.

DiffMean/LinearProbe calls must also explicitly choose the scale regime. In the
scaled regime, `build_intervention` reads `max_activation` only from the exact
direction artifact being loaded and applies
`factor × max_activation × unit_direction` (negative mode uses
`-abs(factor)`). Stage 3 defines the field as the maximum signed
continuation-token `h_token · d` for unit `d` across both rich and avoiding fit
rows at the target hook layer, computed in float32 on CPU without ReLU or
absolute value. Metadata that is missing, invalid, negative, zero-inconsistent,
or bound to another concept/mode/target/layer raises; there is no configuration
knob, SAE-feature-statistics fallback, or constant fallback. Counter directions
likewise consume their own artifacts' values. The explicit unscaled regime
preserves the pre-#1228 direction and coefficient (no normalization and no
`max_activation` value consumption); the fused loader's independent identity
and tensor validation still runs in both regimes.

`model` is required for every `baseline == "ours"` call and remains optional
for every other baseline. For the seven live non-bag-backed methods, the
ordered concept top-K is restricted to the persistent checkpoint `V` before
`build_mode_aware_encoder_input` applies partition semantics. Order and
duplicates survive, an all-filtered result stays empty, and no lower-ranked
refill exists. Negative inactive IDs are independently sampled from the same
`V` at source. For an uncompacted model the raw list object and historical
sampler call shape pass through exactly.

Bag-backed methods instead keep their existing derive-time checkpoint
fingerprint check and load the bag unchanged. Immediately before either live
or bag-backed output returns, `intervention.py` asserts every mapping key is
inside `V`. That mapping assertion is distinct from both the bag fingerprint
check and `teacher_force.assert_enc_ids_in_vocab`, the final tensor-level
pre-forward backstop.

| Method family | `kind` | Fields populated |
|---------------|--------|-----------------|
| `ours_*_encdec` | `"ours"` | `feature_conditions` (final encoder slot payload from `build_mode_aware_encoder_input`) |
| `diffmean_{target}` | `"diffmean"` | `direction`, `direction_layer`, `direction_coefficient` |
| `linear_probe_{target}` | `"linear_probe"` | `direction`, `direction_layer`, `direction_coefficient` |
| `lora_decoder_only` | `"lora"` | `lora_adapter_path` |
| `prompting_{target}` | `"prompting"` | `prompt_prefix_tokens` |

### `apply_intervention` context manager

```python
@contextlib.contextmanager
def apply_intervention(model, intervention: Intervention):
    if intervention.kind in ("diffmean", "linear_probe"):
        hook = _register_direction_hook(
            model, intervention.direction_layer,
            intervention.direction * intervention.direction_coefficient,
        )
        try:
            yield
        finally:
            hook.remove()
    elif intervention.kind == "lora":
        with _attach_lora_adapter(model, intervention.lora_adapter_path):
            yield
    elif intervention.kind in ("ours", "prompting"):
        yield
    else:
        raise AxBenchBpbError(f"Unknown intervention kind: {intervention.kind!r}")
```

Additionally, if the `Intervention` carries the additive counter-steering add-on
(`counter_direction` / `counter_direction_layer` / `counter_direction_coefficient`,
orthogonal to `kind`), `apply_intervention` registers a residual hook
`counter_direction_coefficient * counter_direction` at `counter_direction_layer`
**in addition to** the per-kind behaviour, via an `ExitStack` so both tear down on
exit. The same hook works on the encoder-decoder decoder (`register_residual_hook`
supports `GPT` and `EncoderDecoderGPT`). This is how additive counter-steering
pushes a specified concept direction on top of any baseline (LoRA adapter / encdec
conditioning / direction hook); `magnitude = 0` ⇒ coefficient 0 ⇒ no-op control.

`_register_direction_hook` and `_attach_lora_adapter` are imported from
`steering.eval.axbench_base.baselines.hooks` and
`steering.eval.axbench_base.baselines.lora._attach` respectively. The
LoRA `_attach_lora_adapter` provides byte-exact attach/detach with state
restoration. The many-cycles-per-process usage pattern in axbench-bpb
(vs. one-shot per-concept in axbench-base) is tested explicitly for
memory leaks and state correctness in `functional/test_lora_attach_detach.py`.

### Sign correction in macro rollup

The conditioned margin `BPB_avoid_interv - BPB_rich_interv` is naturally
"larger is better" for positive-mode methods: the model assigns lower
BPB to concept-rich text than concept-avoiding text. For negative-mode
methods, the desired direction is the reverse, so `primary_margin` is
`-margin_intervened`. This sign correction keeps leaderboard sorting
uniform across modes. Implemented in `compute_per_tuple_rollup`
(`bpb_aggregation.py`).

## Tokenization alignment

Tokenization is the most frequent source of bugs in teacher-forced
scoring. AxBench-BPB is built almost entirely out of existing, tested
primitives:

| Primitive | Location |
|-----------|----------|
| `token_to_char_offsets` | `steering/segmentation.py` |
| `assign_tokens_to_segments` | `steering/segmentation.py` |
| `compute_tokenizer_checksum` | `steering/segmentation.py` |
| `load_tokenizer_checksum` | `steering/stats_io.py` |
| `encode_prefix` | `steering/eval/axbench_bpb/tokenization.py` |
| `build_segmented_eval_plan` | `steering/conditional/encoder_decoder/eval_adapter.py` |
| `build_segmented_feature_encoder_inputs` | `steering/conditional/encoder_decoder/eval_adapter.py` |

Two runtime startup guards enforce alignment invariants before any
scoring:

1. **Tokenizer checksum**: `compute_tokenizer_checksum(nanochat_tok) ==
   load_tokenizer_checksum(feature_dir)`. Raises `AxBenchBpbError` naming
   both checksums if they disagree (feature directory built against a
   different nanochat tokenizer).

2. **BOS length**: `len(gemma_tok.decode([gemma_bos_id])) ==
   len(nanochat_tok.decode([nanochat_bos_id]))`. The BOS-decoded-length
   alignment invariant is relied on throughout the codebase (char offset
   computation assumes BOS decodes to one character-worth of space on
   both sides). Raises `AxBenchBpbError` if violated.

### seg_map tokenizer-mismatch defense

`build_multi_segment_seg_map` (in `seg_map.py`) takes prefix
segments whose `char_end` values come from the Gemma extraction
tokenizer. It re-tokenizes the same prefix text with the nanochat
tokenizer to produce per-token char offsets. When the two tokenizers
produce different char lengths (multi-byte handling, normalization,
whitespace), it delegates to the centralised helper
`steering.segmentation.align_segments_to_decoded_prefix`, which (a)
extends the last segment's `char_end` to cover the full nanochat
decoded length (forward direction) or (b) filters fully-uncovered
segments + clips the last surviving `char_end` (reverse direction).
Each adjustment emits a site-scoped profiler counter so dashboards can
attribute cross-tokenizer events to the responsible consumer:

* `segmentation.char_end_extended.full_sft` — full-sequence SFT data prep
  (`steering/conditional/encoder_decoder/full_sequence_routing.py:build_full_sequence_segment_routing`)
* `segmentation.char_end_extended.target_real_filter` — target-real
  filter (`steering/eval/target_segments.py:filter_segments_to_target`)
* `segmentation.char_end_extended.completion_eval` — completion
  steering eval (`steering/eval/multiseg_generator.py:generate_for_document`
  per-target build)
* `segmentation.char_end_extended.encdec_eval_context` — target-aware
  encdec eval context routing
  (`steering/conditional/encoder_decoder/eval_adapter.py:_align_row_context_segments`,
  called from `_build_conditioning_impl` before encoder packing)
* `segmentation.char_end_extended.bpb_seg_map` — axbench-bpb seg_map
  (`steering/eval/axbench_bpb/seg_map.py:build_multi_segment_seg_map`)

Every event carries `direction=forward|reverse` and `delta=<overall_delta>`
attributes (in events.jsonl; `summary.yaml` shows aggregate counts per
site-scoped name). A delta beyond
`CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS` (canonical 20) raises
`ValueError` in both directions.

The same helper backs the cross-tokenizer defence in:

* `steering/conditional/encoder_decoder/full_sequence_routing.py:build_full_sequence_segment_routing`
  (full-sequence SFT data prep partial-coverage / excess-coverage handling)
* `steering/eval/target_segments.py:filter_segments_to_target`
  (target-real partial-coverage handling)
* `steering/conditional/encoder_decoder/eval_adapter.py:_fill_context_assignments`
  (target-aware encdec eval prefill seg_map)
* `steering/eval/multiseg_generator.py:generate_for_document` per-target
  build (completion-steering encdec inference). The sibling
  `build_multiseg_inputs` entry point currently relies on overlay-
  precomputed token boundaries (`scripts/preprocess/precompute_token_boundaries.py`)
  to absorb cross-tokenizer offsets upstream, so it doesn't invoke the
  helper directly — see the `feat-centralise-segmentation-alignment`
  PR's Commit 4 description for the rationale.

`full_sequence_routing.py`, `target_segments.py`, and `multiseg_generator.py`
consume the Layer-1 char-only primitive
`steering.segmentation.align_char_ranges_to_decoded_prefix` directly (they
operate on parallel `(char_starts, char_ends)` lists rather than
`ContextFeatureSegment` tuples). `seg_map.py` and `eval_adapter.py` consume
the Layer-2 dataclass wrapper
`align_segments_to_decoded_prefix`. Both layers share the same counter,
window, and behaviour.

### Prompting-prefix routing in encdec scoring

When ``intervention.kind == "prompting"`` is scored against an ``encdec``
target and ``inputs.prefix_context_segments`` is non-empty, the
prompt-prefix tokens (``intervention.prompt_prefix_tokens``, prepended to
``inputs.prefix_tokens`` before truncation) need a routing strategy that
preserves the cross-attention seg_map contract:

* Non-zero seg_map values monotonically non-decreasing within each row
  (cross_attention.py:185-196).
* ``padded_idx = seg_map * max_seg_size + pos_in_seg`` unique per row (G9
  invariant in ``_padded_idx_unique_per_row``).

The chosen pattern: prepend an empty masked conditioning slot at encoder
index 0, shift every real segment's index up by 1, and route surviving
prompt-prefix tokens to slot 0. Prompt-prefix tokens then receive zero
contribution from cross-attention (slot 0 is fully masked); inputs-prefix
tokens receive their normal SAE-feature conditioning at the shifted
indices.

This avoids three failure modes:

* Routing prompt-prefix to ``generation_slot`` (non-monotonic; G9
  padded_idx collision; silent scatter corruption).
* Routing prompt-prefix to slot 0 directly without the shift (silently
  double-attends to the first inputs-prefix segment's features).
* Clearing ``prefix_context_segments`` to ``()`` (silently drops the
  inputs-prefix conditioning).

The implementation lives in
``steering.eval.axbench_bpb.teacher_force._build_encdec_seg_map_and_encoder``;
both ``_pad_encoder_inputs`` (batched microbatch path) and ``score_sample``
(per-sample path) call it after running ``project_segments_to_truncated_prefix``.

### Feature-Space Roofline General Validation

Fair feature-DiffMean and fair feature-GCG use sparse concept-level feature
bags. When ``OursGcgConfig.top_k_pos`` / ``top_k_neg`` are ``null``, those
bag sizes resolve to the encoder-decoder checkpoint's training-time
``feature_selection`` / ``negative_selection`` geometric means, then the
DiffMean/GCG code selects that many concept features.

The general-validation roofline is different by design: it is a per-document
oracle, not the fair concept-level bag. For each scored validation document,
``general_val_roofline.derive_real_output_signal`` extracts the document's
actual output-window SAE activations, filters them by ``extraction_threshold``,
discretizes the surviving features, and then keeps the top
``k_pos``/``k_neg`` features in the positive/negative partitions by per-feature
percentile. Those budgets are resolved from the same source as fair
feature-DiffMean/GCG: ``OursGcgConfig.top_k_pos`` / ``top_k_neg`` overrides
when set, otherwise the encoder-decoder checkpoint's training-time
``feature_selection`` / ``negative_selection`` geometric means. Roofline
feature-GCG uses that sparse real-signal bag as its warm start and optimizes
only the generation slot; the preceding prefix/context segments stay fixed to
their real extracted features.

For a validated clustered checkpoint, the same per-document oracle operates in
representative space. It requires `extraction_threshold=0.0`, applies the
resolved feature mask to active original SAE IDs before cluster consolidation,
max-reduces member scores onto each representative, ranks representatives by
descending score with ascending representative ID as the tie-break, and samples
inactive representatives from the mask- and cluster-aware negative universe.
Both the derived output bag and the fixed context-cache bags must contain only
mask-allowed representatives from the checkpoint's exact effective
`FeatureVocab`.

The clustered ranking currency matches training preselection:
`magnitude_to_percentile` produces percent ranks in `[0.5, 99.5]`,
`percentile_to_uint8` quantizes those ranks, and `uint8_to_percentile` decodes
the bytes to normalized `[0, 1]` weights before member-wise `amax` and positive
bucket assignment. These units are intentionally asymmetric;
`normalized_percentile_to_uint8` is not interchangeable here.

There are four owner-approved differences from the frozen PR #739 oracle,
mirroring the merged clustered context-cache behavior. Percentiles are computed
only for active `(feature ID, magnitude)` pairs; this is element-wise equivalent
to full-width computation wherever every queried row has statistics. If an
active feature has an all-zero statistics row and is masked out, it is dropped
before scoring with an error-level log because the extraction and statistics
corpora are independently configured. The same mismatch on a mask-allowed
active feature remains fatal, but is reported as `AxBenchBpbAdvError` so it
stays inside the advanced-eval error taxonomy. The objective's final optimized
bag validation likewise wraps cluster- or mask-contract failures in
`AxBenchBpbAdvError`, preserving the original exception as `__cause__`.

An identity-after-mask cluster contract deliberately enters the exact legacy
ranking and bucketing body, but first applies the same masked active stat-less
drop as a genuinely consolidating contract. It zeroes those disallowed IDs in a
clone before calling the legacy body, so the caller's activation tensor is not
mutated and missing statistics are never queried. This cannot change the
selected bag: the legacy selector is restricted to the checkpoint's
mask-allowed `FeatureVocab`, so a dropped masked ID was never a candidate.
Genuinely consolidating contracts use the quantized normalized-percentile
currency above. Both stamp `signal_policy =
GENERIC_ROOFLINE_SIGNAL_VERSION`. This is safe because the persisted identity
also stamps `feature_contract_fingerprint`, which differs between the two
contracts and prevents cross-contract bag reuse.

### Adversarial BPB Robustness

For the BPB-only adversarial robustness harness, see
[`axbench-bpb-adv.md`](axbench-bpb-adv.md). That eval reuses the AxBench-BPB
teacher-forced scoring contract and Stage 0-3 source artifacts; it does not
consume generated completions or judge scores.

Faster-GCG is wired in but opt-in: per optimizable row it is cheap (~70 s at
full production knobs, GPU-batched candidate evals), but the **default grid
explodes the row count** to weeks-to-months, so the full grid is impractical.
Use the committed `configs/eval/axbench_bpb_adv/axbench_bpb_adv_timing_probe.yaml`
spot-check (`optimized_token_counts=[0, 1, 2]`, `max_steps=2`,
one base method / decoder-only target / fixed context). See the "Faster-GCG Cost
and the Timing-Probe Config" section of `axbench-bpb-adv.md` for the cost model
and grid-reduction levers.
