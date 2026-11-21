# AxBench-BPB Evaluation

An offline, loss-based steering benchmark for base language models.
AxBench-BPB teacher-forces the Stage 1b/1c synth samples through the
target model under each steering intervention and measures per-sample
bits-per-byte (BPB). Encoder-decoder targets are evaluated with full
prompt/context conditioning; AxBench-BPB does not compute unconditioned
encoder-decoder baselines.

## Overview

AxBench-BPB is a deterministic, API-key-free variant of AxBench-Base
that replaces Stages 4-6 (autoregressive generation, LLM judge,
judge-aggregated leaderboard) with teacher-forced BPB scoring on the
synth corpus produced by Stages 1b and 1c. Where AxBench-Base measures
steering quality by asking "does the steered model generate concept-rich
text that an LLM judge rates highly?", AxBench-BPB asks "does the
intervention make the model assign higher likelihood to known-good text?"
The two signals are complementary: BPB is immune to sampling variance
and LLM-judge scoring noise; judge scoring captures fluency and
coherence that BPB cannot.

The primary metric is a conditioned margin:
`BPB(avoid | intervention) - BPB(rich | intervention)`. Positive-mode
methods prefer larger margins; negative-mode methods flip the sign so
that "larger is better" still means the method moved likelihood in the
intended direction. The macro leaderboard ranks methods by
`macro_mean_best_primary_margin`, the mean over concepts of each
concept's selected representative coefficient. The historical column name keeps
`best` for compatibility; the default selector is closest to the scored
feature-DiffMean encoder-decoder baseline.

AxBench-BPB is a read-only consumer of AxBench-Base Stage 0-3
artifacts: it reads the concept list, synth parquets, concept-feature
mappings, and pre-trained baseline directions/adapters from the source
run directory and never writes under it. This makes it safe to re-run
against the same axbench-base run at any time, and to run multiple BPB
evaluations (e.g., at different training checkpoints) against a single
source run.

## Pipeline at a glance

```
  Stage 1                 Stage 2                Stage 3
+-----------------------+  +------------------+  +-------------------+
| prepare_context_cache |  | bpb_score        |  | bpb_aggregate     |
| Collect unique prefix |->| method × concept |->| per-tuple →       |
| texts from synth      |  | × sample × coef  |  | per-concept →     |
| parquets; extract SAE |  | sweep; atomic    |  | macro rollups;    |
| features per segment; |  | parquet + SQLite |  | leaderboard.csv   |
| write hash-keyed      |  | per-rank cache   |  |                   |
| context_cache/        |  |                  |  |                   |
+-----------------------+  +------------------+  +-------------------+
   base Gemma + SAE only     target models         CPU / pure math
```

**Stage 1 — `prepare_context_cache`**: collects every unique `prefix_text`
across `synth_training_positive.parquet` and `synth_training_negative.parquet`,
runs base Gemma + SAE on each prefix to extract per-segment SAE features,
discretizes activations into bucket indices, and writes a per-prefix
safetensors file keyed by `sha256(prefix_text)[:16]`. The manifest
(`context_cache/manifest.json`) records `{prefix_hash: {n_segments, n_features_total}}`
for fast lookup. Under DDP, prefixes are partitioned round-robin by
`sha256(prefix_text)[:16]` across all ranks so every GPU contributes
to extraction; after a barrier ensures every rank's safetensors files
are durable, rank 0 reads them back and writes the manifest, then a
second barrier holds non-zero ranks until downstream stages can see it.

**Stage 2 — `bpb_score`**: the main sweep. Scoring covers the exact final
population in source `concepts_stage1c.json`. Source preflight requires both
`concept_rich` and `concept_avoiding` rows for every final concept in both
modes; scoring repeats that check after sample construction so a torn or
changed source raises instead of silently creating a smaller “scorable
subset.” For each registered method, for each final concept, for each sample
in the method's mode, and for each coefficient in the method's grid, a `WorkUnit` is
constructed and hash-partitioned across DDP ranks. Each rank skips work
units already in its SQLite cache (fast-path done-set lookup), builds the
intervention via `build_intervention`, and calls `score_triple` for
batched teacher-forced scoring. Results are written atomically to
rank-local parquet shards and indexed in a per-rank SQLite database.
When `feature_diffmean_encdec` or `neg_feature_diffmean_encdec` is active, rank
0 first derives one feature-DiffMean conditioning bag per accepted
`(concept, mode)` corpus group, using all accepted source rows for that
concept/mode. Every active group must have at least one `concept_rich` row and
at least one `concept_avoiding` row; one-sided groups cannot define a DiffMean
contrast and raise before scoring rather than being silently skipped. Other
ranks wait at a barrier, then score the same precomputed bags.
Note that the aggregation stage applies a separate `min_rows_per_side`
filter (default 10) on top of the scorable-subset filter; the two layers
are independent (see `aggregation.min_rows_per_side` in Key knobs).

### Compacted-checkpoint vocabulary safety for live `ours` methods

The seven non-bag-backed live `ours` methods are:

- positive: `ours_extract_and_add_encdec`,
  `ours_replace_random_encdec`, `ours_replace_all_encdec`
- negative: `neg_ours_remove_encdec`,
  `neg_ours_add_to_negatives_encdec`,
  `neg_ours_replace_random_negatives_encdec`,
  `neg_ours_replace_all_negatives_encdec`

They have two feature-ID producers. First, the ordered concept top-K comes
from `concept_features.safetensors` or
`concept_features_neg.safetensors`. Second, negative-mode orchestration can
draw inactive features from the checkpoint's training-time negative-selection
distribution. For a compacted encoder-decoder checkpoint, both sources are
restricted to the checkpoint vocabulary `V` read from the loaded model's
persistent `encoder.feature_vocab_ids` buffer:

- concept top-K is filtered **before** the partition modifier runs, so
  replace-random methods remove `len(K_V)` entries, not the unfiltered K;
- inactive features are sampled uniformly without replacement from a captured
  CPU snapshot of `V` at source, rather than sampled from the full SAE universe
  and post-filtered. The configured count is drawn first and clamped to `|V|`;
  a random permutation then selects IDs without numeric-ID bias.

Before that top-K is filtered, every non-sentinel artifact ID is validated
against the loaded target's full SAE range `[0, n_sae_features)`. An invalid
ID raises as a corrupt or mismatched artifact instead of being mistaken for a
valid in-range ID excluded by `V`; only `-1` is accepted as an unused-slot
sentinel. Filtering then preserves artifact order and duplicates. If every
valid stored feature is outside `V`, the filtered list is empty: AxBench-BPB
does not refill from lower-ranked features, pad, raise, or invent a substitute
because the artifact contains no authoritative ranking beyond its stored entries.
Negative orchestration still calls its inactive sampler in this case,
including for replace-all, so the established Python RNG-consumption contract
does not change. Bucket semantics are unchanged.

For uncompacted checkpoints (`model.feature_vocab is None`), this is an exact
no-op. The original top-K Python list is passed through, the inactive sampler
uses its historical no-`feature_mask` call shape, and the
`random.Random`-to-CPU-`torch.Generator` bridge still consumes exactly one
`rng.randint` per sampler call.

There are three independent safety boundaries:

1. Bag-backed GCG/feature-DiffMean methods validate the bag's derive-time
   checkpoint-`V` fingerprint. They are not re-filtered.
2. Every final `ours` `feature_conditions` mapping is checked for membership
   in the loaded checkpoint `V`, catching a corrupt bag or missed producer
   before scoring.
3. Teacher-force scoring retains its tensor-level
   `assert_enc_ids_in_vocab` immediately before model forward.

The loaded checkpoint is authoritative throughout; evaluation configuration,
feature-mask paths, and statistics thresholds are not used to reconstruct
`V`. AxBench-Base compacted Stage 4 support and feature-GCG/g_steer work in
issue #720 are separate from this AxBench-BPB contract.

**Stage 3 — `bpb_aggregate`**: reads the full `bpb_scores/` directory
dataset across all ranks and performs a three-level rollup:
per-(method, concept, mode, coefficient, target) → per-(method, concept,
mode, target) with selected coefficient and AUC → per-(method, mode, target)
macro mean-of-means. The representative coefficient is the one whose
`primary_margin` is closest to the same concept/mode's scored feature-DiffMean
encoder-decoder baseline (`feature_diffmean_encdec` or
`neg_feature_diffmean_encdec`); exact closest-score ties choose the higher
steering score, then smaller `abs(coeff)`, then lexical coefficient.
The macro table is sorted by the primary metric and written to
`aggregation/leaderboard.csv`.

## Prerequisites

Before running any stage, the following artifacts and environment must
be in place:

1. **A completed AxBench-Base run (Stages 0-3 done)**. The path passed
   as `source_run_dir` must contain:
   - `concepts.json`
   - `synth_training_positive.parquet`
   - `synth_training_negative.parquet`
   - `concept_features.safetensors`
   - `concept_features_neg.safetensors`
   - `baselines/` (with DiffMean, LinearProbe, Prompting, LoRA artifacts)

   `validate_bpb_config` checks all of these at CLI startup before any
   heavy resource is loaded, and raises `AxBenchBpbError` naming every
   missing file.

2. **A feature directory** that matches the one used for the source
   axbench-base run. The path must follow the
   `{root}/s{seg_size}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}`
   convention so `parse_feature_dir` can recover `layer`, `segment_size`,
   and SAE identity. Typical path:
   ```
   out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/
   ```

3. **A feature statistics safetensors file** (`feature_statistics.safetensors`
   inside the feature directory). Drives the percentile table used by
   the context-cache extraction and the tokenizer checksum guard.

4. **A target decoder-only nanochat base model**, referenced by
   `decoder_only.{source, model_tag, step}`.

5. **A target encoder-decoder nanochat model**, referenced by
   `encdec.{source, model_tag, step}`.

6. **A GPU**. Stages 1-3 run the base Gemma + SAE (Stage 1) and both
   target models (Stages 2-3). Use `device=cpu` for CI runs at roughly
   20x runtime cost. No API keys are required at any stage.

## Running the full benchmark

The Hydra CLI at `scripts.eval.axbench_bpb_eval` runs all four stages end to
end. A shared `RuntimeState` (see `steering/eval/axbench_bpb/runtime_state.py`)
lazy-loads the heavy resources and CPU-offloads them between stage
boundaries when `offload_between_stages=True` (the default).

```bash
# Dev tier: 20 concepts, ~8 samples/concept/side.
uv run python -m scripts.eval.axbench_bpb_eval \
    tiers=dev \
    source_run_dir=out/eval/axbench_base/<base_run_tag> \
    output_root=out/eval/axbench_bpb \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
    decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
    encdec.source=encdec encdec.model_tag=d13 encdec.step=10000 \
    device=cuda
```

```bash
# Standard tier: 50 concepts, full sample set.
uv run python -m scripts.eval.axbench_bpb_eval \
    tiers=standard \
    source_run_dir=out/eval/axbench_base/<base_run_tag> \
    output_root=out/eval/axbench_bpb \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
    encdec.source=encdec encdec.model_tag=d13 encdec.step=10000
```

```bash
# Full tier: 500 concepts.
uv run python -m scripts.eval.axbench_bpb_eval \
    tiers=full \
    source_run_dir=out/eval/axbench_base/<base_run_tag> \
    output_root=out/eval/axbench_bpb \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
    encdec.source=encdec encdec.model_tag=d13 encdec.step=10000
```

**Tier overrides** are specified via the `tiers=` key and resolved by
Hydra from `configs/eval/axbench_bpb/tiers/{dev,standard,full}.yaml`.
The `dev` tier is the right starting point for iteration and debugging;
it completes in minutes on a single GPU.

**Multi-GPU DDP** launch via torchrun. The CLI calls
``nanochat.common.compute_init`` at startup to wire the distributed
process group, then ``compute_cleanup`` from a finally block on exit
(prevents NCCL state leaks across consecutive runs). Stage 1 runs in
parallel across every rank (see "DDP coordination" in the advanced
reference); Stages 2-4 follow the documented hash-partitioned semantics.

```bash
torchrun --nproc_per_node=8 -m scripts.eval.axbench_bpb_eval -- \
    tiers=full \
    source_run_dir=... output_root=... feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
    encdec.source=encdec encdec.model_tag=d13 encdec.step=10000 \
    device=cuda
```

Work units are hash-partitioned across ranks with no cross-rank
communication in the scoring hot loop; each rank writes only to its own
parquet shards and SQLite cache.

**Recommended NCCL env for production:** Set
``TORCH_NCCL_ASYNC_ERROR_HANDLING=1`` (and optionally
``TORCH_NCCL_TRACE_BUFFER_SIZE=20000``) before launching torchrun. If a
single rank fails between barriers in Stage 1 (e.g. a tokenizer guard
raises on rank 0 after the post-extraction barrier, or rank N OOMs
during extraction), the survivors are stranded at the next barrier
until NCCL's per-collective watchdog timeout fires (default 10
minutes). The async-error handler instead surfaces the failure within
seconds and propagates a clear stack trace from each rank.
``compute_cleanup`` calls ``destroy_process_group().shutdown()`` on
the failing rank but this does NOT broadcast an abort to peers — the
async handler is the operator-side fix.

## Running a subset of stages

`stage=<id>` accepts any member value of the `Stage` `StrEnum` in
`steering/eval/axbench_bpb/types.py`:

```
prepare_context_cache   bpb_score   bpb_aggregate   all
```

Any other value raises `AxBenchBpbError` from `Stage.coerce`.

```bash
# Re-run scoring and aggregation without rebuilding the context cache.
uv run python -m scripts.eval.axbench_bpb_eval \
    stage=bpb_score \
    source_run_dir=... output_root=... run_tag=<existing_tag> \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
    encdec.source=encdec encdec.model_tag=d13 encdec.step=10000

uv run python -m scripts.eval.axbench_bpb_eval \
    stage=bpb_aggregate \
    source_run_dir=... output_root=... run_tag=<existing_tag> \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \
    encdec.source=encdec encdec.model_tag=d13 encdec.step=10000
```

Running `bpb_score` and `bpb_aggregate` in isolation (skipping
`prepare_context_cache`) is the common iteration pattern when you want
to re-aggregate after adjusting `enabled_methods` or
`aggregation.primary_metric`.

## Resuming a crashed run

Re-invoke the same CLI command with the same `run_tag`. The pipeline
will resume safely without duplicate work:

- **Parquet shards are the durable artifact.** Each shard is written
  atomically via a temporary file and rename. A crash mid-write leaves
  the previous shard intact.

- **SQLite is the fast-path index.** The `bpb_cache/rank_{R:03d}.sqlite`
  file records which work units have been scored. On resume,
  `reconcile_cache` runs automatically at the top of `bpb_score`:
  it compares the parquet rows against the SQLite index and resolves
  any discrepancy (parquet-ahead-of-SQLite or SQLite-ahead-of-parquet)
  before scoring begins.

- **Live-`ours` rows have a run-level semantic marker.**
  `bpb_scores/.live_ours_vocab_contract` contains
  `axbench-bpb-live-ours-v1:<checkpoint_vocab_fingerprint>`, where the
  fingerprint is the checkpoint's existing compacted-vocabulary SHA-256 or
  `uncompacted`. The guard runs before `reconcile_cache` or any score-row
  skip. If durable rows for one of the seven live methods already exist, a
  missing, malformed, version-mismatched, or vocabulary-mismatched marker
  raises instead of resuming. Pre-marker affected rows cannot be migrated
  safely because their producing checkpoint vocabulary is unknowable: use a
  fresh `run_tag`, or explicitly clear the old BPB score/cache directory and
  rescore. During concurrent DDP startup, a rank that discovers a peer's newly
  flushed affected shard re-reads the marker after its shard scan and accepts
  only the exact expected identity; it does not reject from a stale pre-scan
  marker snapshot. Aggregation-only runs likewise refuse affected rows without
  a well-formed marker. Unrelated methods and bags are not assigned a broader
  model-weight/config identity by this narrow marker.

- **Context cache files are idempotent.** Each `context_cache/{ab}/{hash}.safetensors`
  is written atomically; re-running Stage 1 skips files that already
  exist on disk.

- **Resume at the same `world_size`** preserves the full done-set: each
  rank reads its own SQLite cache and skips already-scored units.
- **Resume at a different `world_size`** is safe but partial:
  hash-partitioning re-assigns work units, and each rank reads only its
  own SQLite cache (no cross-rank done-set reconstruction). Units that
  were scored by a different rank under the prior partition are
  re-scored. Leaderboard outputs remain correct (deterministic scores),
  but GPU work is wasted in proportion to the partition-overlap loss.
  The matrix at
  `tests/axbench_bpb/integration/test_bpb_resume_world_size_changes.py`
  pins this contract with exact-equality remaining-count assertions.

## Where outputs land

All output goes under `output_root/<run_tag>/`. If `run_tag` is not
specified, the CLI auto-generates `YYYYMMDD-HHMMSS-<8charhash>`. The
run tag is sanitized to prevent directory traversal.

```
output_root/<run_tag>/
├── context_cache/
│   ├── manifest.json
│   └── ab/
│       └── abcdef1234567890.safetensors     (prefix hash → ContextFeatureSegments)
├── bpb_scores/
│   ├── .live_ours_vocab_contract            (live-ours semantics + checkpoint V)
│   ├── rank_000/
│   │   ├── batch_000000.parquet
│   │   └── batch_000001.parquet
│   └── rank_001/
│       └── batch_000000.parquet
├── bpb_cache/
│   ├── rank_000.sqlite
│   └── rank_001.sqlite
└── aggregation/
    ├── per_tuple.parquet                    (per (method, concept, mode, coef, target))
    ├── per_concept.parquet                  (per (method, concept, mode, target))
    ├── macro.parquet                        (per (method, mode, target))
    ├── leaderboard.csv                      (sorted by primary_metric)
    ├── report.yaml                          (run summary + timing + validity)
    └── aggregation_report.yaml              (filtered concepts + dropped methods)
```

Path resolvers for all of the above live in
`steering/eval/axbench_bpb/paths.py` (`OutputPaths` dataclass).

## Configuration reference

All config fields live in `AxBenchBpbConfig` at
`steering/eval/axbench_bpb/config.py`. The base Hydra config is at
`configs/eval/axbench_bpb/axbench_bpb_eval.yaml`. Every default is
enforced at construction; cross-field invariants are checked by
`validate_bpb_config`, which runs before any heavy resource is loaded.

### Required fields

| Field | Type | What it controls |
|-------|------|-----------------|
| `source_run_dir` | `str` | Path to a completed AxBench-Base run. Must contain all Stage 0-3 artifacts. Checked by `validate_stage_0_3_artifacts` at startup. |
| `output_root` | `str` | Parent directory for all BPB run artifacts. Actual run root is `{output_root}/{sanitized_run_tag}`. |
| `feature_dir` | `str` | Path to the precomputed SAE feature shards. Must follow the `s{seg}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}` convention. Must match the feature directory used in the source axbench-base run. |
| `feature_statistics_path` | `str` | Path to `feature_statistics.safetensors`. Drives the percentile table and tokenizer checksum guard. |

### Key knobs

| Config key | Default | Meaning |
|------------|---------|---------|
| `stage` | `"all"` | Which stage(s) to run. One of `prepare_context_cache`, `bpb_score`, `bpb_aggregate`, `all`. |
| `enabled_methods` | `null` | Whitelist of method names to score. `null` means all registered methods. Example: `[diffmean_encdec, diffmean_decoder_only]`. |
| `disabled_methods` | `[]` | Methods to exclude after `enabled_methods` narrows the set. Applied in the method registry filter. |
| `diffmean_and_probe_layer` | `8` | Target nanochat/encoder-decoder residual layer at which DiffMean/probe artifacts were trained and will be applied. It is not the Gemma SAE extraction layer. Every selected direction must carry this exact layer. |
| `filter_unaccepted_samples` | `true` | Drop synth rows with `qc_accepted=False` before scoring. Set to `false` to include all rows regardless of QC status. |
| `device_batch_size` | `16` | Samples per micro-batch within a `(method, concept, coefficient)` triple. Larger values improve GPU utilization; smaller reduces peak memory. |
| `aggregation.min_rows_per_side` | `10` | Minimum number of `concept_rich` (and `concept_avoiding`) rows required per `(method, concept, mode, target)` for that concept to contribute to macro statistics. Concepts below this threshold are recorded in `aggregation_report.yaml`. |
| `aggregation.primary_metric` | `"macro_mean_best_primary_margin"` | Column used to sort `leaderboard.csv`. Any column in the macro table is valid. |
| `feature_diffmean.top_k_pos`, `feature_diffmean.top_k_neg` | `null` | Bag size overrides for the scored feature-DiffMean conditioning baseline. `null` resolves the geometric mean from the encoder-decoder checkpoint metadata; no guessed fallback is used. |
| `offload_between_stages` | `true` | CPU-offload heavy models between stage boundaries. Reduces peak GPU memory at the cost of reload time between stages. |
| `run_tag` | `null` | Unique run identifier. Auto-generated as `YYYYMMDD-HHMMSS-<8charhash>` when `null`. |
| `device` | `"cuda"` | Torch device string. |
| `context_cache.num_buckets` | `2` | Number of magnitude buckets used by the per-feature discretizer. Must match the encdec target's ``encoder.magnitude_embed.weight`` row count — 2 for the canonical d12_s32 / d12_baseline checkpoints in this codebase. Override on the CLI when scoring a model trained with a different bucket count. |
| `context_cache.force_rebuild` | `false` | Delete and rebuild the context cache. Use when the feature directory has changed. |
| `aggregation.invalid_warning_threshold` | `0.01` | Log a warning for methods whose `macro_mean_truncation_rate` exceeds this fraction. |
| `aggregation.invalid_drop_threshold` | `0.05` | Drop methods from the leaderboard whose truncation rate exceeds this fraction. |

Configuration/source preflight derives the enabled DiffMean and linear-probe
coordinates for every scored concept and validates them before distributed
initialization or target-model loading. The shared loader checks the same
family, target, mode, concept, canonical metadata layer, path coordinates, and
finite one-dimensional direction again immediately before hook installation.
Clustered callers pass source-retained bytes; plain BPB uses one
descriptor-bound read. Shape-compatible directions for another target or layer
are rejected by identity rather than being silently accepted.

## Output format

### `leaderboard.csv` columns

| Column | Type | Description |
|--------|------|-------------|
| `method_name` | string | Registry name of the steering method, e.g. `diffmean_encdec`. |
| `mode` | string | `positive` or `negative`. |
| `target` | string | `decoder_only` or `encdec`. |
| `n_concepts_scored` | int | Number of concepts that contributed to this method's macro stats (after `min_rows_per_side` filter). |
| `macro_mean_best_primary_margin` | float | Mean over concepts of each concept's selected-coefficient sign-corrected conditioned margin. The historical column name is retained for compatibility. |
| `macro_mean_best_margin_intervened` | float | Mean over concepts of each concept's raw conditioned margin at the selected coefficient. |
| `macro_mean_dose_auc_margin` | float | Mean over concepts of the coefficient-sweep AUC of the sign-corrected conditioned margin. NaN-safe mean; `NaN` for single-coefficient methods such as LoRA and Prompting. |
| `macro_mean_truncation_rate` | float | Mean fraction of samples that required left-truncation of the prefix to fit in context. Methods above `invalid_drop_threshold` are dropped. |
| `best_coef_mode` | string | Most common best-coefficient value across concepts, as a string (e.g. `"3.0"`, `"None"`). |

### `aggregation/per_tuple.parquet`

One row per `(method_name, concept_id, mode, coefficient_str, target)`.
Contains `n_rich`, `n_avoid`, `mean_bpb_intervened_{rich,avoid}`,
`margin_intervened`, `primary_margin`, and `primary_bpb`.

### `aggregation/per_concept.parquet`

One row per `(method_name, concept_id, mode, target)`.
Contains `best_coef`, `best_primary_margin`, `best_margin_intervened`,
`best_primary_bpb`, `dose_auc_margin`, `n_rich_total`,
`n_avoid_total`, `truncation_rate`.

### `bpb_scores/rank_*/batch_*.parquet` columns

| Column | Type | Description |
|--------|------|-------------|
| `method_name` | string | |
| `concept_id` | int32 | |
| `mode` | string | `positive` or `negative` |
| `content_class` | string | `concept_rich` or `concept_avoiding` |
| `target` | string | `decoder_only` or `encdec` |
| `coefficient_str` | string | Stringified coefficient from the method's grid |
| `sample_id` | string | SHA256-derived 16-char stable identifier |
| `bpb_intervened` | float64 | Teacher-forced BPB under the intervention |
| `n_output_tokens` | int32 | |
| `n_output_bytes` | int32 | UTF-8 byte count of the output text |
| `truncated` | bool | True if the prefix required left-truncation |

## Cost and runtime

The total cost is O(N_concepts × N_methods × N_samples × N_coefs)
forward passes through the target model. With the default 500-concept
full tier, 33 registered methods, ~72 samples per concept, and a typical grid of
5-10 coefficients per method, this is on the order of 10-15 million
forward passes.

Rough estimates for the scoring stage (Stage 3) on a single H100, bf16,
with `device_batch_size=16`:

| Tier | Concepts | Approx. total forwards | Single H100 | 8× DDP |
|------|----------|----------------------|-------------|--------|
| dev | 20 | ~500k | ~3 min | ~25 s |
| standard | 50 | ~1.25M | ~8 min | ~60 s |
| full | 500 | ~12.5M | ~14 h | ~1.7 h |

Stage 1 (`prepare_context_cache`) depends only on the base Gemma + SAE
(not target models) and runs once per source run. On an H100: ~tens of
minutes for 500 concepts. On CPU: hours. The cache is invalidated only
if `context_cache.force_rebuild=True` is passed; switching target
checkpoints does not require rebuilding it.

`bpb_aggregate` is pure pandas/pyarrow math and completes in seconds.

## Troubleshooting

**`AxBenchBpbError: source_run_dir is required`** — `source_run_dir` was
not passed on the CLI. Provide the path to a completed AxBench-Base run:
`source_run_dir=out/eval/axbench_base/<base_run_tag>`.

**`Source run at ... is missing required Stage 0-3 artifacts: ...`** —
the source run did not complete all of Stages 0-3. Re-run axbench-base
for the missing stages before launching axbench-bpb. The error message
names every missing file.

**`BOS decoded-length mismatch`** — the BOS token of the extraction
tokenizer (Gemma) decodes to a different number of characters than the
BOS token of the nanochat tokenizer. This is a tokenizer alignment
invariant the codebase depends on throughout. It should never occur for
the standard `google/gemma-2-2b` + nanochat tokenizer pairing; if it
does, check `extraction_tokenizer_id` and verify the nanochat tokenizer
has not been replaced.

**`Tokenizer checksum mismatch`** — the SHA256 checksum of the nanochat
tokenizer stored in the feature directory does not match the currently
loaded nanochat tokenizer. This means the feature directory was built
against a different tokenizer. Either point `feature_dir` at a directory
built with the current tokenizer, or regenerate the feature directory.
The error names both checksums.

**`zero bytes after masking`** — a sample's output tokens all have
zero-byte weight in the byte-weighted BPB computation. This indicates a
bug in `output_byte_weights` construction or that every output token is
a special token with no UTF-8 representation. Check the synth row's
`output_text` field in the source parquet.

**`non-finite values in output-position loss`** — the teacher-forced
forward pass produced `inf` or `nan` loss values at output positions.
This usually indicates model divergence or a numerical issue with the
intervention (e.g., very large direction coefficient). Check the
`sample_id`, `method_name`, and `coefficient_str` values in the error
message; try reducing the coefficient or disabling the problematic method
with `disabled_methods=[<method>]`.

**`Unknown stage: <value>`** — the `stage=` value is not one of the
valid `Stage` enum members. Valid values are:
`prepare_context_cache`, `bpb_score`, `bpb_aggregate`, `all`.

**`device_batch_size must be >= 1`** — `validate_bpb_config` caught an
invalid batch size. The minimum is 1.

**`aggregation.min_rows_per_side must be >= 1`** — minimum filter
threshold for concept rows must be at least 1. Lower values weaken the
statistical reliability of the macro metrics.

**`Invalid threshold ordering`** — the `aggregation.invalid_warning_threshold`
must be ≤ `aggregation.invalid_drop_threshold`, and both must be in
`[0, 1]`. The default `(0.01, 0.05)` satisfies this.

**Hydra / OmegaConf gotchas** — if constructing `AxBenchBpbConfig`
programmatically from a `DictConfig`, use
`AxBenchBpbConfig(**OmegaConf.to_container(cfg, resolve=True))` so
nested configs (`ContextCacheConfig`, `AggregationConfig`) become real
dataclass instances rather than `DictConfig` proxies.

## Glossary

- **BPB (bits per byte)** — tokenizer-agnostic language model loss metric.
  `BPB = nats_loss / (ln(2) × n_utf8_bytes)`. Normalizing by UTF-8 bytes
  makes BPB comparable across vocabularies. Lower BPB means higher model
  confidence in the text.

- **primary_margin** — the sign-corrected conditioned margin. For
  positive mode this is `BPB(avoid | intervention) - BPB(rich | intervention)`;
  for negative mode the sign is flipped so larger is always better.

- **dose_auc_margin** — trapezoidal integral of `primary_margin` over
  the coefficient grid. `NaN` for single-coefficient methods (LoRA,
  Prompting).

- **macro_mean_best_primary_margin** — the primary leaderboard metric.
  For each concept, take the coefficient that maximizes `primary_margin`;
  then average those best-concept margins across all scored concepts.
  Mean-of-means rather than pooled-over-samples, so each concept
  contributes equally regardless of sample count.

- **Positive mode** — the intervention steers the model toward the target
  concept. Evaluated on `concept_rich` samples (known-good positives).
  `primary_margin` rewards assigning lower BPB to `concept_rich` than
  `concept_avoiding` text.

- **Negative mode** — the intervention steers the model away from the
  target concept. Evaluated on `concept_avoiding` samples (known-good
  avoidances). `primary_margin` rewards assigning lower BPB to
  `concept_avoiding` than `concept_rich` text.

- **concept_rich / concept_avoiding** — sample-level content class.
  `concept_rich` text was synthesized (Stage 1b) or verified (Stage 0c)
  to fire the concept SAE feature. `concept_avoiding` text was synthesized
  in Stage 1c or paired from the natural corpus and strict-QC'd in Stage 1b.

- **WorkUnit** — the atomic unit of scoring: `(method_name, concept_id,
  sample_id, coefficient_str)`. Hash-partitioned across DDP ranks.
  The SQLite `PRIMARY KEY` on these four fields (plus `target`) enforces
  idempotency.

- **context cache** — per-prefix hash-keyed safetensors files storing
  `ContextFeatureSegment` tuples. Built once by Stage 1 against the base
  Gemma + SAE; read by Stages 2 and 3 to populate encoder inputs without
  re-running the SAE forward pass for every scored sample.

- **truncation** — left-prefix truncation applied when
  `n_prefix + n_output > model.dec_config.sequence_len`. Prefix tokens
  are dropped from the leftmost end; output tokens are never truncated.
  Samples that required truncation are flagged `truncated=True` and
  contribute to the `truncation_rate` validity gate.
