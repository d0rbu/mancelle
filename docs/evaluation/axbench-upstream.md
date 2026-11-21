# AxBench Ten-Method Comparison

For suppression on naturally eliciting original chat histories, see
[natural-chat negative AxBench](axbench-negative.md). That path uses a separate
frozen source with fixed neutral sampling and SAE-verified avoiding references,
and supports native DiffMean/LinearProbe and encoder-decoder SelfConditioning.

This pipeline deliberately uses three producer environments and ordinary
Parquet files:

- vendored AxBench's Python 3.12 environment produces a main frame with
  `DiffMean` and `LinearProbe` over the full factor grid and an auxiliary frame
  with `PromptSteering` and `LoRA` at factor 1.0;
- the root project-method environment produces corrected `SelfConditioning`
  and sign-masked `SelfConditioningMasked` over the full factor grid;
- the root encoder-decoder-baseline environment produces `encdec_main` with
  `EncDecDiffMean` and `EncDecLinearProbe` over the full grid and `encdec_aux`
  with `EncDecPromptSteering` and `EncDecLoRA` at factor 1.0;
- a root-level validator merges those five source frames into one ten-method
  frame.

The full factor grid is `0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 5, 6`.
The vendored source and lock live under `vendor/axbench/`; both root producers
use the normal repository lock.

The downstream [AxBench-Adv GEPA evaluation](axbench-adv.md) uses its own local
`ADV_METHOD_FACTORS` table and now covers all six methods. The immutable
five-method c2 bundle remains a read-only parity oracle; the accepted
six-method project/evaluated artifacts live under
`out/axbench/runs/ce941fe47-c2-six-method-20260816/`. AxBench-Adv consumes source
provenance for selection and regenerates every plotted control and candidate.
In particular, its root-method points are not copied from this project Parquet:
the Parquet, project manifest, checkpoint receipt, target model/metadata, and
feature statistics are identity oracles for exact runtime replay.

## Prepare inputs

Create the environment and download the pinned concept-500 inputs, AlpacaEval,
seed datasets, and English Stanza resources:

```bash
uv sync --project vendor/axbench --frozen
uv run --project vendor/axbench --frozen \
  python vendor/axbench/prepare.py
```

The command writes `out/axbench/prepared/prepare_manifest.json` and returns
without downloading when the prepared data is already complete and loadable.
A partial or malformed directory fails with an instruction to rebuild it:

```bash
uv run --project vendor/axbench --frozen \
  python vendor/axbench/prepare.py --force
```

Use `--prepared-dir /path/to/prepared` to place the inputs elsewhere; pass the
same option to the launcher. `--force` removes and recreates only the selected
prepared directory. Before removing a nonempty directory, it refuses
filesystem/repository roots, source-tree paths, symlinked paths, and unmarked
custom directories. Ordinary creation and validation resolve the requested
path, including symlinked parent directories.

## Run the baselines

Use a fresh dump directory for each concept count and process count. AxBench's
own pickle state and rank Parquets implement stage resume; the launcher does
not delete, repair, or skip them. A small `run_manifest.json` prevents a dump
directory from being accidentally reused with different config, input,
concept-count, process-count, or lock settings. It records the repository
commit and dirty-worktree summary when the run is created, plus the source
state used by the latest train, latent, or steering invocation. Later source
commits do not rewrite the creation record or prevent the remaining stages
from resuming in the same dump directory; they update that stage's entry under
`stage_provenance`. Manifests created before stage provenance was added are
upgraded in place with their original commit and an explicit unknown worktree
state.

`--max-concepts` must be positive and cannot exceed the concept rows in the
prepared metadata (500 for the checked-in preparation recipe).

For a two-concept smoke run of the main methods:

```bash
MAIN=vendor/axbench/axbench/sweep/nano_gpt_pretrain_steer/qwen2_5_3b_main.yaml
RUN=out/axbench/runs/c2/main

uv run python -m scripts.eval.axbench_upstream.run_baselines train \
  --config "$MAIN" --dump-dir "$RUN" --max-concepts 2
uv run python -m scripts.eval.axbench_upstream.run_baselines latent \
  --config "$MAIN" --dump-dir "$RUN" --max-concepts 2
uv run python -m scripts.eval.axbench_upstream.run_baselines steering \
  --config "$MAIN" --dump-dir "$RUN" --max-concepts 2
```

The auxiliary config trains LoRA and then emits LoRA and PromptSteering in one
factor-1 Parquet:

```bash
AUX=vendor/axbench/axbench/sweep/nano_gpt_pretrain_steer/qwen2_5_3b_aux.yaml
RUN=out/axbench/runs/c2/aux

uv run python -m scripts.eval.axbench_upstream.run_baselines train \
  --config "$AUX" --dump-dir "$RUN" --max-concepts 2
OPENROUTER_API_KEY=... uv run python -m scripts.eval.axbench_upstream.run_baselines steering \
  --config "$AUX" --dump-dir "$RUN" --max-concepts 2
```

PromptSteering uses AxBench's supported `openai/gpt-5.6-luna` prompt-synthesis
call through OpenRouter and therefore requires `OPENROUTER_API_KEY`; the main
profile makes no provider call. The vendored OpenAI-compatible SDK client uses
`OPENROUTER_BASE_URL` when set, defaulting to `https://openrouter.ai/api/v1`.
The launcher fails before `torchrun` if a selected method or steering dataset
requires that provider and the key is absent. Main steering likewise requires
the completed `inference/latent_data.parquet`; run the latent stage first.
The launcher invokes script files through `torchrun` because the upstream
entry points initialize NCCL even with one process. Pass `--nproc-per-node N`
to use more GPUs, and keep that value fixed for a dump directory.

The baseline outputs are ordinary AxBench files under each dump directory,
including `train/`, `inference/latent_data.parquet`, and
`inference/steering_data.parquet`.

## Run the project methods without baseline generation

Project-only evaluation does not require a Qwen baseline run. Build an
input-only frame directly from the pinned prepared metadata and AlpacaEval
instructions; the checked-in producer defaults to five concepts and ten
prompts per concept across the full factor grid:

```bash
uv run python -m scripts.eval.axbench_upstream.build_project_input
```

The producer loads no generation model and writes no baseline result columns.
It atomically publishes the strict input Parquet and a manifest that records
the validated prepared-manifest digest and the explicit no-model contract. The
consumer requires both files, revalidates their mutual SHA-256 binding and the
prepared source identity, and includes both digests in resume identity. Run
both project methods in the root environment by selecting that source and
disabling the legacy main-frame source:

```bash
uv run python -m scripts.eval.axbench_upstream.run_methods \
  main_path=null \
  project_input_path=out/axbench/runs/c5/project_input/steering_input.parquet \
  project_input_manifest_path=out/axbench/runs/c5/project_input/manifest.json \
  output_path=out/axbench/runs/c5/project_v4/inference/steering_data.parquet \
  manifest_path=out/axbench/runs/c5/project_v4/run_manifest.json \
  max_concepts=5
```

The checked-in methods config retains the accepted legacy two-concept main
frame as its default. Its resume identity and schema-3 run-manifest shape remain
unchanged (the model-free-only keys are omitted):

```bash
uv run python -m scripts.eval.axbench_upstream.run_methods
```

The model-free source intentionally carries raw AlpacaEval prompts rather than
Qwen chat framing. Its project result is therefore not input-equal to an
independently generated Qwen baseline frame and must not be passed to the
strict ten-method merger. The standard checked-in `run_evaluation` entry point
remains a merged-ten-method evaluator; a project-only campaign must use a
dedicated two-method judge config and validate its output as a distinct
artifact rather than relabeling or merging incompatible inputs.

Override the path and size fields with normal Hydra syntax when using another
legacy run directory, for example:

```bash
uv run python -m scripts.eval.axbench_upstream.run_methods \
  main_path=out/axbench/runs/c10/main/inference/steering_data.parquet \
  output_path=out/axbench/runs/c10/project_v4/inference/steering_data.parquet \
  manifest_path=out/axbench/runs/c10/project_v4/run_manifest.json \
  max_concepts=10
```

### Optional natural-language retrieval arm

`NLSteering` uses the same self-conditioned interpolation and full `0..6`
factor grid as `SelfConditioning`, but obtains its dense target from the exact
AxBench concept description rather than fitting positive and same-genre
examples. It is opt-in: the default config, two project methods, ten-method
merge, judge, roundtrip, seeds, and resume identity remain unchanged.

The optional input is one strict schema-1 receipt with exactly
`description_embeddings` and `concept_embeddings` records. Each record binds a
relative NPZ path, raw SHA-256, source-population SHA-256, and the same complete
embedding-space object (`model`, 40-character revision, encoding, and nonempty
library versions). The concept NPZ carries exactly `texts` and `emb`; its
Unicode texts must reproduce the receipt's population hash. The description
NPZ carries exactly `fids` and `emb`. A receipt is admission evidence only when
those producer assertions have been independently audited; shape-correct
digests alone do not qualify a historical source.

For each checkpoint head row, available feature descriptions are unit-normalized
and averaged over that row's exact mask/cluster membership, then normalized
again. For concept query `c` and covered row vector `u_i`, the target is:

```text
s_i = cosine(u_i, c)
m   = lower_median({s_i | i covered})
d_i = (s_i - m) / max_j |s_j - m|       for covered rows
d_i = 0                                  for uncovered rows
```

If the covered centered population is constant, every target is zero. Outputs
are contiguous CPU float32 in checkpoint head-row order and clamped to
`[-1, 1]`. The receipt, source bytes, full mask/cluster identity, exact concept
texts, and the computed target-byte digest enter the NL-only semantic identity.
Every rank revalidates the receipt and source bytes after generation, then
synchronizes success before rank 0 publishes either canonical output. A failed
revalidation leaves completed cells available for a retry with the same inputs.

Run the inherited preset after installing an audited receipt at its configured
path, or override that path explicitly:

```bash
uv run --no-sync python -m scripts.eval.axbench_upstream.run_methods \
  --config-name=methods_nl \
  nl_target_source_manifest_path=/path/to/nl_target_source_manifest.json
```

Both fixed-token and metadata-qualified turn-delimited checkpoints use this
target. Fixed checkpoints retain the established two-method physical cohort
and generate NL cells separately; turn checkpoints retain scalar generation
and their recorded refresh cadence. Disable the arm by returning
`nl_target_source_manifest_path` to null and using a fresh default output root.

The same entry point supports one process per local CUDA device. Under
`torchrun`, `device` must be exactly `cuda`; compute initialization assigns
each rank its local device:

```bash
uv run torchrun --standalone --nproc_per_node=8 \
  -m scripts.eval.axbench_upstream.run_methods -- \
  device=cuda \
  main_path=out/axbench/runs/c500/main/inference/steering_data.parquet \
  output_path=out/axbench/runs/c500/project_v4/inference/steering_data.parquet \
  manifest_path=out/axbench/runs/c500/project_v4/run_manifest.json \
  max_concepts=500
```

The supported production topology is one node, one process per visible GPU,
and a shared filesystem. Both producers first form stable
`(concept_id, input_id)` prompt groups, then assign each first-seen group index
modulo world size. Project groups therefore keep both methods across the full
factor grid on one rank; encoder-decoder groups likewise keep the complete
direction-method factor grid together, with auxiliary cells following their
prompt group. `concept_id % world_size` independently owns each fit. Seeds
contain neither rank nor world size. Direct invocation is the identical
world-size-one executor.

`SelfConditioning` uses the closed policy
`prompt_last_segment_max_pool_logit_rank_diffmean_lerp_v2`. Every unique rendered prompt
is extracted transiently with the checkpoint's nested prompt-conditioning
policy: Gemma-3-4B-PT residual SAE layer 22, context-independent fixed
128-token segments, max pooling, a 2,048-token extraction window, no Gemma BOS
feature pooling, the checkpoint mask and clusters, and independently sampled
active bucket-0 and true-inactive bucket-1 features. Exact nanochat token tuples
are deduplicated in first-occurrence `(concept_id, input_id)` order and reused
across factors. Nothing is written as a prompt cache.

For each concept, the same positive and same-genre `EEEEE` examples produce a
complete dense vector in feature-prediction-head row order:

```text
d = (mean_positive_percentile_rank - mean_negative_percentile_rank) / 100
```

The mask is applied before percentile lookup. Cluster siblings are max-folded
within each document before class averaging. Zero-difference rows remain in the
vector. The completion prediction uses raw FP logits only from decoder
positions routed to the final real prompt segment; the final
`<assistant_start>` position is excluded:

```text
z = MAX_POOL(last_real_prompt_segment_logits).float()
q = 2 * sigmoid(z) - 1
alpha = SelfConditioning_strength / 6
scores = z                                      when alpha == 0
scores = d                                      when alpha == 1
scores = torch.lerp(q, d, alpha)                otherwise
```

Strength 0 ranks `z` directly so distinct finite logits cannot collapse into
sigmoid endpoint ties. Strength 6 uses `d` exactly, and intermediate strengths
retain the historical float32 `q` and `torch.lerp` calculation exactly.
Interpolation happens before feature selection.
The checkpoint's top-level positive count is drawn first and negative count
second from one factor-independent selection seed. Stable highest rows become
active bucket 0; stable lowest remaining rows become inactive bucket 1. Text
sampling retains a separate factor-specific seed.

For a `fixed_tokens` checkpoint, the predicted output slot is used by the first
sampled token and held for the entire at-most-128-token continuation;
`self_cond_segment_size` is the decoder sequence length, so there is no
mid-completion refresh. This is the historical AxBench behavior.

For a `turn_delimited_fixed_tokens` checkpoint, prompt extraction instead uses
the source Gemma segment size from the checkpoint's online-SAE lineage and the
source Gemma context window from its validated statistics artifact. Those
values do not determine generated-token refreshes. The generated assistant uses
the RustBPE cadence recorded by self-conditioning post-training metadata:
`<assistant_start>` is input row 1 of the first generated window, and every
input is forwarded exactly once. After the Kth input row, its outgoing logits
remain the distribution for the next sample, the newly predicted slot is
published, and the next input is the first one conditioned on that slot. There
is no boundary-token replay.

The statistics artifact a turn run is pointed at must be the byte-identical
file conditioned SFT trained against: the online-SAE lineage pins its RAW
sha256, not the re-saveable `content_sha256`, so any rewrite of the file
invalidates it even when the decoded statistics are unchanged. Stamping a
percentile or tokenizer checksum the file does not already carry
(`save_percentile_checksum`) rewrites the safetensors header and makes the run
fail with `Turn-context statistics raw SHA-256 does not
match…`; copy the SFT-time file instead of regenerating it. There is no
relaxed mode here:
`require_exact_statistics_identity=False` — the operator-asserted escape hatch
for pre-`content_sha256` fixed-token checkpoints — is rejected outright on a
turn-delimited checkpoint rather than silently ignored.

That cadence is a precondition, not a default: a turn-delimited conditioned-SFT
checkpoint straight from `scripts/train/conditional_encdec_sft.py` records only
`sft_segment_mode` and no cadence, so setup resolution fails with
`turn-delimited checkpoints require top-level post-training metadata
turn_self_cond_segment_size_rustbpe…`. Such a checkpoint is unusable on every
AxBench surface until self-conditioning post-training has run and recorded the
top-level measured `turn_self_cond_segment_size_rustbpe` (see
[`../pipelines/selfcond-post-training.md`](../pipelines/selfcond-post-training.md)).

The self-conditioned checkpoint AxBench is pointed at is therefore the one that
must carry the whole identity, and post-training re-publishes it: a top-level
`step` written from its own loop state (which is how `model_{step:06d}.pt` is
located), plus `pretrain_conditioning`, `feature_mask_canonical`, both
`feature_clusters_*` blocks, `feature_vocab`, `feature_prediction_width`,
`missing_percentile_policy`, `dataset` and `sft_loss_mask` copied verbatim from
its SFT base. `pretrain_meta_path` does not substitute for them: it supplies
only `user_config.conditioning` and the raw `sae` block.

Turn prompt entries carry their producing `SegmentMode` and exact per-token
segment ownership through preparation into the context provider. Prefill checks
the mode against the checkpoint before accepting the entry, then uses the stored
ownership after removing the terminal assistant marker. Fixed-token entries
cannot be injected into a turn wrapper even if their character ranges happen to
match. Persistent context-cache files remain fixed-token only.

The turn geometry manifest block includes `prompt_statistics_raw_sha256`, copied
from the validated online-SAE lineage, alongside the selection seed and
missing-percentile policy. It records the raw bytes that gated setup; the decoded
statistics content hash remains a separate identity.

Turn-mode dense DiffMean honors the lineage's missing-percentile policy. In
turn mode the percentile table comes from the checkpoint's own SFT-domain
statistics, digest-bound to the online-SAE lineage, while the DiffMean contrast
texts come from the AxBench concept corpus, so a mask-allowed feature that the
SFT mixture never activated has an all-zero percentile row. The fit resolves
those ids through the same `MissingPercentileResolver` the conditioned-SFT
selector uses — read straight off `prompt_selection_context`, never re-derived:
under `missing_percentile_policy=lowest` the uncovered ids are pinned to
percentile rank `0.5`, and under `error` the fit aborts up front with the
policy in force and the complete uncovered population rather than on whichever
id was looked up first. Fixed mode records no lineage and therefore no policy;
its lookups are unchanged and `magnitude_to_percentile` remains their only
guard.

The dense extraction loader and percentile reducers live in
`steering/eval/dense_feature_targets.py`, shared with the offline
[ChatCORE benchmark target builder](../pipelines/self-conditioned-chatcore.md#benchmark-identity-interpolation).
AxBench retains its concept sources and fitting interface; the shared helpers
preserve its document reduction order and FP-head row mapping.

`SelfConditioningMasked` uses the closed policy
`prompt_last_segment_max_pool_logit_rank_diffmean_sign_rank_v3` over the same
prepared prompts and dense vector `d`. Stable DiffMean rank defines
eligibility; pooled model logits perform final selection.

```text
maximum_strength = 6
tau = factor / maximum_strength
real_limit = N * (1 - tau) + 2 * configured_mean * tau
pool_limit = clamp(round(real_limit), 1, N)
```

Python `round` supplies nearest, ties-to-even rounding. The positive limit uses
the checkpoint's positive configured mean and the negative limit uses its
independent negative configured mean; both means must be finite and strictly
positive. `N` is the number of eligible feature-prediction-head rows after the
effective vocabulary and feature mask, never an arithmetic subtraction from
the original SAE width. At factor zero both limits equal `N`, so factor zero
removes the rank cap but retains sign and mask rules. If `2 * mean < 1`, the
maximum-strength limit clamps to one; if `2 * mean > N`, it remains unrestricted
at `N`. Limits are nonincreasing with strength after clamping and can differ by
side.

After feature masking, positive DiffMeans are ranked descending and negative
DiffMeans ascending with stable ties in canonical head-row order. Exact-zero
DiffMeans are excluded. The limit is a deterministic candidate-pool cap, not
the stochastic sampled count. Counts are still drawn positive first and
negative second from the factor-independent selection seed, then clamped to the
actual ranked-pool sizes. Token sampling retains its separate factor-specific
seed. A factor-independent preflight builds the complete post-mask sign pools
immediately after fitting each dense target and fails before target-model load
when a required side is empty. Pool construction advances no RNG state and no
sorting occurs in the generation loop.

Generation is fixed at temperature `1.0` and top-k `50`. These are comparison
constants, not Hydra knobs.

The driver validates the main frame's schema, dtypes, key product, and input
invariants, but deliberately does not apply generation-health checks to the
baseline result columns it discards. The newly generated project frame and the
eventual ten-method merge remain fully strict.

The project run writes
`out/axbench/runs/c2/project_v4/inference/steering_data.parquet` and an ordinary
schema-3 `run_manifest.json` by default. The manifest records the repository and
dirty-worktree summary, vendored commit, target checkpoint, dataset revision,
seed and concept count, fixed temperature/top-k regime, resolved configuration,
root lock digest, input/statistics digests, factor sets, dense DiffMean units
and endpoints, prompt/output conditioning policies, both seed derivations, the
eligible-head-row count, independent configured means, ties-to-even/clamp
contract, and the exact per-factor positive/negative pool-limit table. Pool
construction is deterministic and adds no seed.
For turn-mode checkpoints, both the project-method and the encoder-decoder
baseline manifests additionally write the same flat
`self_conditioning_segment_geometry` record of the prompt segment mode, the
source Gemma SAE model and layer, the prompt segment size and context window in
Gemma tokens, and the generated-assistant cadence in RustBPE tokens. The
marker-anchor and refresh semantics are fixed by the code the manifest header
already pins by commit and lock digest.
The block is absent for fixed mode, whose historical manifest fields and text
remain unchanged. The human-readable `prompt_features` and `completion_slots`
fields are also mode-correct rather than claiming fixed-128 prompt segments or
one immutable completion slot for a turn-mode run.
Dense vectors are serialized only as operational resume artifacts under
`_project_fit_resume_v1/`; they are not canonical comparison outputs or added
to the run manifest. Each vector is an atomic fp32 safetensor bound to the
cell store's semantic fingerprint, and the complete set is sealed by an
immutable SHA-256 receipt. Prepared identity and strict-JSON validity are
checked before any model load. The final manifest is strictly serialized
before canonical publication, and both canonical files use atomic replacement.
Existing output or manifest files fail unless `force=true` is passed.
Even with `force=true`, an identifiable schema-2 threshold-era project manifest
is preserved and rejected with instructions to inspect and manually clear that
exact output or choose a new root; it is never reinterpreted as a rank-era run.

If generation completes but strict final-frame validation fails, the driver
atomically writes the complete schema-shaped frame beside the canonical output
as `steering_data.rejected.parquet`, annotates and re-raises the validation
error, and leaves both canonical artifacts absent or byte-identically intact.
All explicit and derived write paths are checked against every input artifact
before I/O.

Before spending generation budget on the complete sweep, the driver generates
all rows for the first concept present in the input frame and applies both
generation-health tripwires to SelfConditioning and SelfConditioningMasked
independently. The blank-rate check has an
early-only five-blank count floor to avoid deterministic false stops on the
small sample; the median stripped-length check uses the final validator's
eight-character minimum with no count floor. Input rows may be shuffled or
concept-interleaved: the checked concept is executed first, then every result
is restored to its original input-row position before the Parquet is written.

### Root-producer resume and failure recovery

Both root producers create placement-free stores beside `manifest_path`:
`_project_cell_resume_v1/` or `_encdec_cell_resume_v1/`. Rank-local immutable
Parquet batches hold the complete logical cell identity and generation text.
Startup revalidates the semantic fingerprint, complete expected grid, schemas,
all stored keys, duplicates, and the first-concept tripwire receipt before it
skips any work. The fingerprint includes the actual prepared metadata and
training-data bytes and the target `model_<step>.pt` body, not only their
manifests or neighboring metadata. Exact duplicates may be deduplicated;
conflicting duplicates, unknown files, stale identities, corrupt batches,
out-of-grid cells, or bulk cells without the receipt fail loudly.

A normal interruption is restarted with the same semantic inputs and output
paths. World size may change: ownership is recomputed over the same canonical
grid and completed logical cells remain completed. Do not run two independent
launchers concurrently against the same paths. `force=true` permits canonical
Parquet/manifest replacement but never deletes, overwrites, or reinterprets a
mismatched resume store. For a changed semantic configuration, use a fresh
manifest/output root; retain the old root as evidence or remove it manually
only after confirming the exact target.

All heavy phases and every collective boundary use aligned rank-failure
exchange. A rank-local ordinary exception is reported on every rank before any
peer can enter the next collective. Rank zero alone runs the complete
first-concept tripwire, publishes Parquets, and writes the manifest last. Every
rank independently reconstructs source order and validates the same complete
frame before that rank-zero publication stage; this prevents a nonzero rank
from skipping local reconstruction or schema failures and then entering a
later collective.

## Run the encoder-decoder baselines

After vendored main and auxiliary steering complete, fit and run all four
encoder-decoder baseline analogs in the root environment:

```bash
uv run python -m scripts.eval.axbench_upstream.run_encdec_baselines
```

The checked-in `configs/eval/axbench_upstream/encdec_baselines.yaml` is the
two-concept profile. It accepts exactly these top-level fields; missing and
unknown keys are rejected:

| Field | Meaning |
|---|---|
| `main_path` | Validated vendored main steering Parquet used for the full-grid input rows; its producer manifest must be at `<main_path>/../../run_manifest.json` |
| `aux_path` | Validated vendored auxiliary steering Parquet used to recover PromptSteering text; its producer manifest must be at `<aux_path>/../../run_manifest.json` |
| `prepared_dir` | Prepared metadata and positive/same-genre-negative concept-training rows |
| `output_main_path` | `encdec_main` Parquet for EncDecDiffMean and EncDecLinearProbe |
| `output_aux_path` | `encdec_aux` Parquet for EncDecPromptSteering and EncDecLoRA |
| `manifest_path` | Run provenance JSON, published last |
| `train_dir` | Exact resumable per-concept direction and adapter safetensors, bound by the cell-store identity and fit receipt |
| `target_meta_path` | Conditional encoder-decoder checkpoint metadata |
| `feature_statistics_path` | Feature statistics matching that checkpoint |
| `max_concepts` | Positive concept count, matching the prepared data and input frames |
| `device` | Extraction and target-model device |
| `seed` | Base seed for probe/LoRA fitting, self-conditioning selection, and token generation |
| `max_new_tokens` | Positive continuation cap; prompt plus continuation must fit the decoder sequence length |
| `force` | Permit replacing the two canonical Parquets and manifest; never bypasses resume/train-artifact identity checks |

Before loading either model, the driver reads both producer manifests and
requires their target model and revision fields to be internally consistent and
equal across main and auxiliary frames. That identity selects a closed exact
wrapper contract: legacy `Qwen/Qwen2.5-3B-Instruct` or
`HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO`. The SmolLM3 bytes include
the default system prompt rendered under AxBench's pinned 27 August 2026 date.
There is no wrapper inference from frame text: unknown identities, prefix or
suffix drift, original-prompt-tail drift, and blank recovered steering prompts
all fail loudly. Source-manifest hashes are part of resume identity and the
published manifest.

Override fields with normal Hydra syntax, as for the project-method driver:

```bash
uv run python -m scripts.eval.axbench_upstream.run_encdec_baselines \
  main_path=out/axbench/runs/c10/main/inference/steering_data.parquet \
  aux_path=out/axbench/runs/c10/aux/inference/steering_data.parquet \
  output_main_path=out/axbench/runs/c10/encdec_v1/main/inference/steering_data.parquet \
  output_aux_path=out/axbench/runs/c10/encdec_v1/aux/inference/steering_data.parquet \
  manifest_path=out/axbench/runs/c10/encdec_v1/run_manifest.json \
  train_dir=out/axbench/runs/c10/encdec_v1/train \
  max_concepts=10
```

The distributed spelling mirrors the project producer:

```bash
uv run torchrun --standalone --nproc_per_node=8 \
  -m scripts.eval.axbench_upstream.run_encdec_baselines -- \
  device=cuda \
  main_path=out/axbench/runs/c500/main/inference/steering_data.parquet \
  aux_path=out/axbench/runs/c500/aux/inference/steering_data.parquet \
  output_main_path=out/axbench/runs/c500/encdec_v1/main/inference/steering_data.parquet \
  output_aux_path=out/axbench/runs/c500/encdec_v1/aux/inference/steering_data.parquet \
  manifest_path=out/axbench/runs/c500/encdec_v1/run_manifest.json \
  train_dir=out/axbench/runs/c500/encdec_v1/train \
  max_concepts=500
```

When enough GPUs are available, the project and encoder-decoder producers may
also run at the same time. Give the two launchers disjoint visible-device sets
and disjoint output/manifest roots; local ranks are relative to each set. For
example, launch the project command from one terminal with
`CUDA_VISIBLE_DEVICES=0,1,2,3` and `--nproc_per_node=4`, and the encoder-decoder
command from another with `CUDA_VISIBLE_DEVICES=4,5,6,7` and
`--nproc_per_node=4`. Never point concurrent independent launchers at the same
resume root.

The run produces `encdec_main` and `encdec_aux` Parquets at the configured
paths, one exact resumable `train/` artifact directory containing each concept's
DiffMean direction, probe direction, and LoRA adapter, and one manifest. The
manifest records input and lock digests, resolved config, target identity,
generation constants, method factors and formulas, per-concept max
activations, training seeds and metadata, recovered-prompt hashes, the twelve
documented divergences below, and every training artifact's SHA-256. The two
Parquets are atomically published before the manifest; the manifest is the
completion marker. With `force=true`, an existing manifest is deleted at the
publication boundary before either Parquet is written, so a failed forced
rebuild cannot leave a stale completion marker.

Each fit concept has one rank owner. All ranks subsequently reload and validate
the complete artifact set before generation; the immutable fit receipt binds
every filename and SHA-256 to the semantic cell-store fingerprint. A partial
set resumes missing owned concepts. Preexisting artifacts without the matching
resume identity, unexpected files, metadata drift, or a receipt digest mismatch
fail rather than being reused.

After that fit validation, every rank clears its process-local
`torch.compile` state before entering generation. Fit activation capture and
LoRA training exercise the shared compiled packed cross-attention callable with
training shapes and gradient modes; retaining those specializations can exhaust
Dynamo's recompile limit and make autoregressive inference fall back to unfused
eager FlexAttention. The reset is unconditional so every rank enters generation
under the same compiler-state contract whether that rank fitted, resumed, or
owned no concepts. It does not delete Inductor's filesystem cache, alter fit
artifacts, or add a scientific configuration dimension; generation specializes
afresh from the validated fit boundary.

Project ownership is assigned to complete `(concept_id, input_id)` prompt
groups before rank placement. All project cells for a prompt therefore have one
owner: a fresh canonical 24-cell group remains B=24 at both world size one and
world size four instead of being cell-striped into four B=6 fragments. Missing
`SelfConditioning` and `SelfConditioningMasked` rows within that owned group are
decoded as one physical batch only for `fixed_tokens` checkpoints. Every batched
row first performs the original batch-one prefill with a fresh config and
selection RNG under its own dense score policy. This freezes the policy-specific
S0 encoder state, logits, and KV cache before any state is stacked; decode retains
the original method- and factor-specific token RNG per row. A partially completed
fixed-mode resume still batches only the missing cells owned by that rank,
including a scalar singleton remainder. Turn-delimited groups instead generate
each missing cell through the scalar runtime in stable canonical order because a
completion can refresh its conditioning state more than once.

For fixed-token checkpoints, missing EncDecDiffMean and EncDecLinearProbe rows
use the same bounded batch mechanism after the original batch-one prefill under
each row's residual addend. This is an operational optimization, not a new
semantic mode. The stacked decode retains one residual addend and one token RNG
per row.
Stopped rows consume neither further RNG draws nor output tokens; they remain
inert physical filler rows so surviving rows never change batch index. The path
still computes the feature-prediction head on every forwarded token, as the
batch-one non-boundary path does.

The batch path is permitted only in `fixed_tokens` mode and when the continuation
cap cannot reach the next self-conditioning boundary. The checked-in 128-token
cap is below the 1,024-token fixed-mode self-conditioning window, so at most 127
sampled tokens are ever forwarded and S0 remains the active slot for the
complete continuation. A larger cap that could cross the boundary fails before
decode rather than silently dropping FP-buffer state. Turn-delimited generation
always uses the scalar path, including when its recorded cadence is one token;
each prefill carries immutable segment-mode provenance, and the batch API rejects
both a turn-mode active wrapper and any turn-mode row supplied to a fixed wrapper
before batched state or sampling work. PromptSteering, LoRA, singleton resume
remainders in either driver stay on the scalar reference path. If any missing
encoder-decoder cell in an exact `(concept_id, input_id)` prompt group is
selected for a Torch trace, that rank's entire missing prompt group stays scalar
and executes in its original cell order. If any missing project cell in a prompt
group is selected, every missing project cell in that group likewise runs scalar
in canonical order. These rules preserve each selected cell's pre-batching
execution history; unrelated prompt groups remain batched only in fixed mode.
Logical cell identity, seeds, completed-cell resume state, canonical row order,
and output schemas are unchanged; project rank placement now preserves complete
prompt groups, and there is deliberately no batching config knob: checkpoint
segment-mode metadata controls eligibility. Physical
batch size can select a different floating-point kernel, so numerical parity is
gated at the documented backend tolerances rather than by assuming byte-identical
logits across devices or kernel shapes.

The realistic H200 gate compares 16 teacher-forced decode states for the exact
24-row production group: all 12 declared factors for DiffMean and LinearProbe,
in canonical factor-major and method-interleaved order. Each scalar reference
process completes one prefill and decode before the next; the candidate process
performs 24 independent scalar prefills followed immediately by one B=24
decode. At every state the gate requires centered LM-logit relative RMSE
`<= 0.10`, centered correlation `>= 0.995`, top-50 overlap `>= 40/50`, and
full-softmax total variation `<= 0.15`. Feature-prediction logits and every
filled K/V-cache layer use an uncentered comparison, requiring relative RMSE
`<= 0.10` and cosine similarity `>= 0.995`; subtracting the mean there would
incorrectly hide a constant raw-state corruption. The gate also verifies
cell/request identity, source-state immutability, packed encoder row ownership,
and exact forward/comparison counts. These are fixed-input numerical gates.
Sampled strings are not an exact cross-batch oracle because a small accepted
logit delta can change an early categorical draw and therefore the entire
continuation. The strict artifact comparator below remains appropriate for
implementations that promise the same physical numerical path; it is not a
substitute for this B=1/B>1 teacher-forced gate.

The companion project-method H200 gate applies the same 16-state tolerances to
the candidate whole-group W=4 geometry and the legacy cell-striped W=4
geometry. It derives both through the production ownership functions and
requires exact per-rank widths of `[24, 0, 0, 0]` for whole-group ownership and
`[6, 6, 6, 6]` for legacy striping. For this single-prompt gate, the candidate's
sole nonempty group must be the exact canonical B=24 sequence already exercised
by the world-size-one batch, so the gate does not repeat that expensive physical
decode. It separately
executes all four legacy B=6 groups to retain the previous shape oracle. Each
scalar reference process completes one full prefill and decode at a time; each
candidate process performs every independent prefill in its physical group
before its bounded batch decode. The gate covers all 12 factors for
SelfConditioning and ranked-v3 SelfConditioningMasked after fitting the dense
DiffMean from every selected training row. LM logits use the centered gate;
feature-prediction logits and each decoder layer's filled K/V prefix use the
uncentered relative-RMSE/cosine gate. It pins factor-major method-interleaved
cell identity, both ownership geometries, seeds, score-policy type and
independent selection state, source-state and target-parameter immutability, all
768 LM/feature-prediction comparisons, all 18,432 per-layer K/V comparisons,
768 scalar forwards, and 80 batched forwards. It additionally requires exact
candidate/legacy initial-state fingerprints and applies the same tolerances
directly to 384 B=24-versus-B=6 LM/feature-prediction states and 9,216
per-layer newly appended K/V slices.

### Compare, profile, and model campaign cost

The campaign helper is deliberately separate from the production entry points:

```bash
# Tier-A artifact parity, including exact Parquet and canonical training
# safetensor bytes.
uv run python -m scripts.benchmark.benchmark_axbench_upstream compare \
  --kind encdec \
  --reference-manifest out/axbench/oracle/encdec/run_manifest.json \
  --candidate-manifest out/axbench/candidate/encdec/run_manifest.json \
  --output-json out/axbench/candidate/encdec/comparison.json

# Fresh-process profiling. Repeat with --kind project for the other producer.
uv run python -m scripts.benchmark.benchmark_axbench_upstream profile \
  --kind encdec --output-dir out/axbench/profiles/encdec-c2-sample-1 \
  --override main_path=... --override aux_path=... \
  --override output_main_path=... --override output_aux_path=... \
  --override manifest_path=... --override train_dir=...

# One low-overhead complete-run throughput sample. Use fresh output paths and
# repeat in fresh processes at least three times per measured arm.
uv run python -m scripts.benchmark.benchmark_axbench_upstream throughput \
  --kind encdec --output-dir out/axbench/throughput/encdec-c2-sample-1 \
  --override main_path=... --override aux_path=... \
  --override output_main_path=... --override output_aux_path=... \
  --override manifest_path=... --override train_dir=...

# Attribute current-stream CUDA work for named coarse regions without enabling
# selected-cell Torch traces or synchronizing at every region boundary.
uv run python -m scripts.benchmark.benchmark_axbench_upstream throughput \
  --kind project --output-dir out/axbench/throughput/project-c2-device-time-1 \
  --cuda-event-region axbench_upstream.generation.batch_prefills \
  --cuda-event-region axbench_upstream.generation.batch_stream \
  --override main_path=... --override output_path=... \
  --override manifest_path=...

# Isolated production-width packed-attention differential and latency sample.
uv run python -m scripts.benchmark.benchmark_axbench_upstream attention \
  --sequence-length 127 --packed-count 1024 --warmups 5 --iterations 20 \
  --output-json out/axbench/profiles/attention-q127-kv1024.json

# Build explicitly labeled projections from at least three complete-run samples.
uv run python -m scripts.benchmark.benchmark_axbench_upstream cost-model \
  --samples-json out/axbench/profiles/samples.json --workers 8 \
  --output-json out/axbench/profiles/cost-model-w8.json
```

The comparator permits differences only at its literal operational-manifest
path allowlist. It otherwise requires exact manifest semantics, Parquet bytes
and logical values, logical cell grids/seeds, train inventories, safetensor
metadata/tensors, and canonical safetensor bytes. New direction and LoRA
artifacts must already have canonical JSON headers. A legacy reference header
is canonicalized in memory before the final byte comparison because the old
Rust serializer can permute metadata key order across processes; its tensor
payload is copied byte-for-byte, and a noncanonical candidate is rejected.
Each manifest's raw SHA-256 declaration must still match its own file. The
comparator normalizes those two already-verified digest fields only for the
subsequent manifest semantic comparison.
Profiling writes per-rank lightweight region JSONL/YAML/CSV plus CPU/CUDA stack
and Chrome traces for three fixed representative cells (project
SelfConditioning at factor 6, EncDecDiffMean at factor 6, and EncDecLoRA at
factor 1). Each selected-cell `key_averages.json` has a complete
`phase_events` table in addition to its bounded hottest-operation table. The
phase rows split generation into prefill, sampling, stop detection, decoder
forwards, and cache compaction; self-conditioned cells further expose the
speculative/committed boundary forwards, provider selection and bucketing,
segment materialization, encoder invocation, and packed-state append. These
fine-grained markers do not synchronize CUDA: use the trace's CUDA durations,
not marker CPU dispatch time, for device-work attribution. The `throughput`
arm deliberately uses the same normal producer and region recorder but
disables selected-cell Torch traces, the fine-grained markers, and per-region
CUDA synchronization. Use a fresh profile/throughput directory and fresh
producer output paths for every sample; never count a profiler run as a
throughput run or reuse completed cells in a timing sample.

For post-batching attribution, repeat `--cuda-event-region` with exact coarse
region names. The opt-in path records start/end events on that rank's selected
CUDA device and current stream, defers synchronization until all producer work
has finished, and emits one aggregate `cuda_event_timing` event plus matching
YAML/CSV fields. It fails the benchmark when any requested region is absent or
does not have exactly one event pair per observed region invocation. These
durations are inclusive elapsed spans on the recorded stream: nested regions
overlap and must not be summed, and idle gaps while the host submits work or
performs setup between the two event records remain in the span. They are not
summed kernel-active time. Work launched on an auxiliary stream is included
only when that stream rejoins the recorded stream; use a Torch trace when
stream-level attribution matters. Only coarse `profile_region` names are
selectable; the fine-grained trace-only phase markers above are not. With no
`--cuda-event-region`, no CUDA event is allocated and the legacy event, YAML,
and CSV schemas are unchanged. Because the opt-in arm performs one final
device synchronization and allocates timing events, use it to choose a
bottleneck, not as an unbiased throughput sample.

Collect at least three fresh-process throughput samples per measured arm. The
attention subcommand compares the historical and guard-stable V views, eager
and compiled FlexAttention, and a dense SDPA reference on an all-one-segment
shape where that reference is exactly legal; it reports synchronized CUDA/wall
samples, fixed-input deltas, strides, and same-kernel repeat equality. SDPA is
not a production fallback for general segmented packed routing. Cost-model
results are projections, not measurements, and cannot establish multi-GPU
scaling without at least two physical GPUs. Their primary
`generation_seconds` field assumes the campaign's concept-major layout with ten
contiguous prompt groups per concept. Because production also supports
interleaved source rows, every projection additionally reports
`generation_seconds_arbitrary_order_upper_bound` and the corresponding total;
those fields charge a conservative upper bound on first/bulk stage skew over
any supported group order.

### Operating point and shared fit population

These are residual-, adapter-, and prompt-level transplants onto the l22
conditional encoder-decoder, not feature-space methods. Every fit pair is
rendered as a complete user/assistant training conversation. Its decoded text
is extracted through the real Gemma-3-4B-PT l22 SAE path, routed back to the
exact nanochat tokens, and forwarded teacher-forced through the target model
under real conditioning. Layer 9 residual outputs are captured in fp32.

The captured population includes every token position at or after template
prefix length 2: user and assistant content plus `<user_end>`,
`<assistant_start>`, and `<assistant_end>`. DiffMean concatenates those tokens
within each class and is token-weighted. LinearProbe first takes one fp32 mean
per fit row and fits on those row vectors. Rows that exceed the 1,024-token
target window or the 2,048-token Gemma extraction window fail instead of
truncating.

Evaluation uses real extracted prompt features through `PreparedPromptContext`
and pure self-conditioned continuation (`score_policy=None`), then adds the
method delta. EncDecDiffMean and EncDecLinearProbe install their layer-9
residual hook for each scalar prefill and a row-indexed layer-9 residual matrix
for compatible batched decode; EncDecLoRA installs its adapters;
EncDecPromptSteering replaces the prompt. The residual and adapter deltas also
change the decoder states read by the FP head, so they intentionally steer the
self-conditioning loop itself. Generation remains fixed at temperature `1.0`,
top-k `50`, at most 128 new tokens by default, and stop tokens BOS or
`<assistant_end>`.

Compatible project and encoder-decoder baseline cell batches retain the scalar
contract of one independently seeded generator per physical row. Each batched
decode iteration validates the currently active logits together, applies the
canonical sampling arithmetic to each row in stable physical order, and
transfers the sampled token vector to the host once for stop handling. Finished
rows are excluded from both the finite guard and future RNG draws and continue
to occupy their fixed decoder row with the BOS filler. On a non-finite active
row, an error-only scalar replay preserves the legacy first-offending-row
diagnostic and partial decode counter; the successful path does not replay
work.

### Per-method semantics

`EncDecDiffMean` computes token-weighted fp32 class means over the shared
population:

```text
raw = mean(concat(positive token activations))
    - mean(concat(negative token activations))
direction = raw / (norm(raw) + torch.finfo(torch.float32).eps)
max_act = max over every positive and negative fit-row token of (h · direction)
h_new = h + (factor * max_act) * direction
```

The additive epsilon is the vendored unit-norm formula, not `F.normalize`.
`max_act` is signed, fp32, and calibrated on the fit rows rather than a
vendored latent stage. An all-zero direction returns `max_act = 0.0`; a
nonzero direction with a non-finite or non-positive maximum raises instead of
using the vendored silent `50` fallback. Factor 0 keeps the layer-9 hook
installed with coefficient `0.0`, preserving the same exact-float no-op
codepath as every nonzero factor.

`EncDecLinearProbe` uses the project probe convention. Each fit row becomes
one fp32 CPU vector by averaging all of its post-prefix token activations.
Positive row vectors are stacked first with label 1, then negative row vectors
with label 0. The estimator is L1
`sklearn.linear_model.LogisticRegression(l1_ratio=1.0, C=1.0,
max_iter=500, solver="liblinear", random_state=derived_probe_seed,
class_weight=None)`. Its
coefficient is cast to fp32 and normalized as

```text
direction = F.normalize(logistic_regression.coef_[0])
```

The derived probe seed uses the canonical
`"axbench-upstream-encdec-probe-v1"` domain, base seed, and concept ID. The
direction is therefore unit-normalized with `F.normalize`. Steering and
calibration are otherwise identical to EncDecDiffMean: `max_act` is computed
over every token activation from every positive and negative fit row, the
non-positive fail-loud gate is unchanged, and the layer-9 coefficient is
`factor * max_act`.

`EncDecPromptSteering` accepts only the exact vendored Qwen wrapper:

```text
steered_input = PRE + steering_prompt + "\n\nQuestion: " + original_prompt + SUF
PRE = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
SUF = "<|im_end|>\n<|im_start|>assistant\n"
```

The steering prompt must be nonblank and identical across every row of a
concept. The driver strips `PRE` and `SUF`, verifies the exact question tail,
then re-renders the raw text `steering_prompt + "\n\nQuestion: " +
original_prompt` with the target template and prepares real prompt features
for that text. `encdec_aux.steered_input` stores this recovered raw text, not
the Qwen wrapper.

`EncDecLoRA` attaches vendored-shape rsLoRA adapters to `attn.c_proj` at decoder
layers `[2, 5, 7, 9]`:

```text
A shape = (4, d_in)
B shape = (d_out, 4)
delta = (32 / sqrt(4)) * B(A(dropout(x)))
output = base(x) + delta
```

Rank is 4, alpha is 32, rsLoRA scaling is `alpha / sqrt(rank) = 16`, and
dropout is `0.05`. As in locked PEFT 0.13.2, input to the adapter branch is
cast to the fp32 adapter dtype, A/B and the delta are computed in fp32, and the
delta is cast back to the base activation dtype before addition. A uses PEFT's
Kaiming initialization and B starts at zero. Only positive fit rows train the
adapters. Supervised targets obey
`targets[t] = ids[t+1] if mask[t+1] == 1 else -1`, with the final target `-1`.
Training runs 24 epochs, row batches of 6, accumulation 6, and
`AdamW(lr=9e-4, weight_decay=0.0)`; each batch is six real-conditioned B=1
sum-loss forwards divided by its total supervised-token count and then by 6.
Vendored AxBench floors the zero-warmup linear-scheduler horizon and can
silently freeze the learning rate at zero mid-training when row counts are not
divisible by 36. This driver deliberately uses
`24 * max(1, ceil(n_batches / 6))`, so decay spans the real optimizer steps.
The optimizer steps at each accumulation boundary and at every epoch end.
Non-adapter parameters are frozen transactionally, and the caller's training
mode and `requires_grad` flags are restored.

The strength columns are schema values, not interchangeable physical units.
`EncDecDiffMean_strength` and `EncDecLinearProbe_strength` are float64
`factor * max_act` over the full 12-factor grid.
`EncDecPromptSteering_strength` and `EncDecLoRA_strength` are the declared
float64 value `1.0` on their single factor. The four analogs have no
perplexity columns.

Self-conditioning selection seeds are shared with the project methods and are
factor-independent:

```text
selection_seed = first_8_bytes_big_endian(
    sha256("\x1f".join(["axbench-upstream-selfconditioning-selection-v1",
                        str(seed), str(concept_id), str(input_id)]))) % (2**31 - 1)
```

Token-generation seeds are method- and factor-specific:

```text
generation_seed = first_8_bytes_big_endian(
    sha256("\x1f".join([str(seed), method, str(concept_id), str(input_id),
                        float(factor).hex()]))) % (2**31 - 1)
```

Probe and LoRA training use the same separator, digest, first-eight-byte, and
modulus construction over respectively
`["axbench-upstream-encdec-probe-v1", str(seed), str(concept_id)]` and
`["axbench-upstream-encdec-lora-v1", str(seed), str(concept_id)]`. Thus a
factor-0 EncDecDiffMean cell shares the pure self-conditioning selection stream
with its project-method counterpart while retaining a distinct token stream.

### Documented divergences from the vendored baselines

| # | Divergence | Rationale |
|---:|---|---|
| 1 | Target model, template, engine, and environment | Forced by the encoder-decoder architecture |
| 2 | Real-conditioned fits and self-conditioned evaluation | Owner-ratified operating point |
| 3 | B=1 per-cell generation, explicit top-k 50, and per-cell SHA-256 seeds | Determinism; matches the ratified project methods |
| 4 | `max_act` comes from fit rows, not a latent stage | Owner-ratified simplification |
| 5 | Non-positive `max_act` raises; there is no silent `50` | House no-silent-fallbacks rule |
| 6 | Mean and loss accumulation are fp32; vendored accumulation is bf16 | Precision correctness |
| 7 | The probe trains on cached activations | The frozen base makes the math identical; tested |
| 8 | The probe's intercept is fit but used only during training; steering projects onto the normalized coefficient vector alone | Owner-ratified estimator; the vendored GD probe's dead-term note is superseded |
| 9 | The analogs have strength columns and no perplexity columns | Owner-ratified schema |
| 10 | A LoRA batch is six B=1 sum-loss forwards divided by supervised tokens | Exactly equals the batched token mean; tested |
| 11 | Fit-row shuffles are explicitly seeded; vendored code uses framework-global RNG | Determinism; order differs by construction |
| 12 | probe estimator = project row-mean logistic regression (vendored GD probe unlearnable at this activation scale; AUC evidence above) | Owner-ratified after the real-capture estimator comparison |

## Validate and merge

Merge the vendored main and aux outputs, the two encoder-decoder outputs, and
the project output:

```bash
uv run python -m scripts.eval.axbench_upstream.merge_methods \
  --main out/axbench/runs/c2/main/inference/steering_data.parquet \
  --aux out/axbench/runs/c2/aux/inference/steering_data.parquet \
  --encdec-main out/axbench/runs/c2/encdec_v1/main/inference/steering_data.parquet \
  --encdec-aux out/axbench/runs/c2/encdec_v1/aux/inference/steering_data.parquet \
  --project out/axbench/runs/c2/project_v4/inference/steering_data.parquet \
  --output out/axbench/runs/c2/merged_v4/steering_data.parquet \
  --max-concepts 2
```

The merger refuses an existing output before reading any inputs. Pass
`--force` only when an atomic, fully validated replacement is intended.

Before writing, the merger checks exact required columns and dtypes, the full
concept/input/factor key product, input and concept-metadata invariants, and
cross-run prompt/input equality. AxBench legitimately changes `steered_input`
for PromptSteering, and the encoder-decoder prompt analog stores its recovered
raw steered text there, so the field is required to be nonblank in both
aux-like frames but is not compared with main. It also checks method null
masks, finite strengths, positive finite perplexities, non-null declared
generations, and basic quality thresholds: at most 1% whitespace-stripped
blank generations per method (a pure fraction bound with no count floor) and
a median stripped generation length of at least eight characters.
PromptSteering, LoRA, EncDecPromptSteering, and EncDecLoRA cells are null
outside factor 1.0 in the merged frame; every remaining method covers the full
grid. The exact source geometries are 120 rows per concept with 18 main
columns, 10 rows per concept with 18 aux columns, 120 rows per concept with 18
`encdec_main` columns, 10 rows per concept with 18 `encdec_aux` columns, and
120 rows per concept with 18 project columns. The merged frame has 120 rows
per concept and 34 columns. Evaluation preserves those rows and appends 70
ten-method judge fields, producing exactly 104 columns. A project or merged
frame containing legacy `FeatureDiffMean_*` columns fails exact-schema
validation.

Merged artifacts from the former six-method contract no longer pass current
merged-frame validation. Their main, aux, and project source frames remain
valid inputs and can be combined with newly produced `encdec_main` and
`encdec_aux` frames to create a current ten-method merge.

The one supported judged NL product combines the four vendored baselines with
the three project methods. It does not expand the generic ten-method product:

```bash
uv run --no-sync python -m \
  scripts.eval.axbench_upstream.merge_baseline_project_methods \
  --main out/axbench/runs/c2/main/inference/steering_data.parquet \
  --aux out/axbench/runs/c2/aux/inference/steering_data.parquet \
  --project out/axbench/runs/c2/project_nl_v1/inference/steering_data.parquet \
  --output out/axbench/runs/c2/baseline_project_nl/steering_data.parquet \
  --max-concepts 2 \
  --include-nl-steering
```

The opt-in validates all seven method columns, full NL factor coverage, dtypes,
generation health, key product, and exact baseline/project input equality.
Without `--include-nl-steering`, NL columns are rejected as extras.

## Judge and leaderboard outputs

Run the vendored AxBench judge on the merged frame. The input rows must already
be in canonical `(concept_id, input_id, factor)` order. The output directory
must be new or empty; the SQLite cache may be shared across runs but must live
outside the output directory:

```bash
OPENROUTER_API_KEY=... uv run python -m \
  scripts.eval.axbench_upstream.run_evaluation \
  --input out/axbench/runs/c2/merged_v4/steering_data.parquet \
  --output-dir out/axbench/runs/c2/evaluated-v4-live \
  --cache out/axbench/judge-cache.sqlite3
```

The checked-in evaluator uses `openai/gpt-5.6-luna` through OpenRouter. Pass
`--api-base-url` to target another OpenAI-compatible base URL, including the
local server used by the offline acceptance test. The cache key contains the
request schema version, final chat-completions endpoint, model, messages, and
temperature. It has one SHA-256 primary key and ordinary short SQLite
transactions. Concurrent misses can both reach the provider before one stored
response wins; subsequent reads return that winner.

The checked-in evaluator covers all ten method models and appends seven fields
per method, for 70 judge fields total and a 104-column evaluated frame. It
sends requests only for a method's declared factors. In particular, the null
PromptSteering, LoRA, EncDecPromptSteering, and EncDecLoRA padding outside
factor 1.0 causes no provider calls. A response must contain exactly one
finite `Rating: [[number]]` in the inclusive range 0 through 2. Retryable
transport failures, HTTP 429/5xx, null/blank content, and malformed ratings
receive at most four total attempts; other HTTP 4xx and unexpected errors fail
immediately. No trusted price is recorded for this model, so cost reporting is
explicitly unavailable rather than guessed.

Each concept has six full-grid methods with 120 declared cells each and four
factor-1 methods with 10 each. With three judge prompts per declared cell, the
logical request count is `(6 × 120 + 4 × 10) × 3 = 760 × 3 = 2,280` per
concept. That is 4,560 requests for c2, 22,800 for c10, and 1,140,000 for c500.
Paid provider calls require separate authorization after the
offline/current-cache c2 gate.

The command writes AxBench's ordinary `steering.jsonl` and
`steering_data.parquet`, then checks concept, method, factor, rating,
completion, row, and null-mask coverage. Replay the same requests without any
possibility of a paid call by using a fresh output directory and the completed
cache:

```bash
uv run python -m scripts.eval.axbench_upstream.run_evaluation \
  --input out/axbench/runs/c2/merged_v4/steering_data.parquet \
  --output-dir out/axbench/runs/c2/evaluated-v4-cache-only \
  --cache out/axbench/judge-cache.sqlite3 \
  --cache-only
```

Cache-only mode requires the cache to exist, constructs no provider client,
and fails on the first missing or malformed cached response.

Judge the seven-method NL product with its dedicated vendored configuration:

```bash
uv run --no-sync python -m \
  scripts.eval.axbench_upstream.run_baseline_project_evaluation \
  --input out/axbench/runs/c2/baseline_project_nl/steering_data.parquet \
  --output-dir out/axbench/runs/c2/baseline_project_nl_evaluated \
  --cache out/axbench/judge-cache.sqlite3 \
  --cache-only \
  --include-nl-steering
```

This route asserts the exact ordered seven-method config before invoking the
vendored evaluator. The existing six-method route and request/cache keys for
its methods are unchanged.

## Across-concept score histograms

Plot saved judge results without generating text or calling a provider:

```bash
uv run python -m scripts.eval.plot_axbench_histograms \
  pipeline=upstream \
  input_dir=/absolute/path/to/evaluated \
  output_dir=/absolute/path/to/histogram-report \
  '+method_factors={DiffMean:1,LinearProbe:1,LoRA:1,PromptSteering:1,SelfConditioning:5,SelfConditioningMasked:5}'
```

The factors above are an explicit example, not recommended or automatically
selected operating points. Choose the intended factor for each method; that
factor is held fixed across concepts and all three axes. Methods may be a
subset of the supported registry (the ten defaults plus optional `NLSteering`), so historical six-method and project-only
judged files are supported without relabeling them as current merged outputs.
LoRA/PromptSteering and their encoder-decoder analogs only support factor 1.
Hydra needs the leading `+` when adding method names to the default empty map.

The plot has one row per selected method and three columns showing concept
relevance, fluency, and instruction-following, so methods never share an axis.
Each row is labeled with its method and fixed factor. Each observation is one
concept's saved mean over its ten judged prompts.
Every concept has equal weight. The reader checks the raw JSONL scores against
the companion Parquet and checks the saved means against those observations
(`1e-12` absolute/relative tolerance for independently accumulated means).
Selected methods must cover identical concepts and prompts, with input IDs
0–9 per concept. Missing selected scores fail; off-grid null padding is not
counted as zero. A report requires one dataset. Concept relevance always
retains its raw meaning, including on suppression datasets: it is not inverted.

All panels share score limits 0–2, percentage limits 0–100, and 20 equal-width
bins by default (`bins` overrides that count). Bins include their left edge;
the final bin also includes 2. Fractional ratings are preserved. The report is
descriptive: a tight low-scoring distribution is not reliable successful
steering, and a two-concept smoke run cannot establish broad consistency.

Bars are filled with the method color. Every panel annotates the across-concept
mean and sample variance of the plotted per-concept means.

The new or empty output directory must be disjoint from the input directory.
It contains `score_histograms.png`, `concept_scores.csv` (the exact plotted
means and prompt counts), `summary.csv` (mean, sample standard deviation, and
sample variance across concepts; these are blank for one concept), and
`metadata.json`. Metadata records input hashes, plotting revision/source hash,
selected factors, bin edges, counts, percentages, and summaries. It is written
last; its absence means the report is incomplete. Inputs and existing plots
are not modified. The plotting checks do not certify generation provenance or
replace the producer's full artifact validator.

Example output: [masked s32 SFT, first 100 concepts, with factors
DiffMean=0.6, LinearProbe=1.5, LoRA=1, PromptSteering=1,
SelfConditioning=6, and SelfConditioningMasked=6](figures/axbench_masked_s32_c100_score_histograms.png).

For GEPA results, use the [AxBench-Adv mode](axbench-adv.md#across-concept-score-histograms),
which reads recorded factors and emits separate reports for each proposal level.

## Roundtrip metric

The roundtrip metric re-encodes every declared steered generation in the
merged ten-method frame through the concept's own gemma-scope SAE and
checks whether the concept feature reactivates in the text the method
produced: steer with the concept, then measure whether the concept comes
back out. This isolates whether a method's steered text is legibly about
the concept, independent of the judge model used above. Probing is
response-only: only the steered generation text (`{method}_steered_generation`)
is re-encoded, never the prompt. For a declared, non-blank cell the
generation is tokenized with one injected BOS and passed through the probe
SAE; `max_act` is the max, over every **non-BOS** token position, of the
JumpReLU post-activation of the concept's own feature, in fp32, and a cell
**activates** iff `max_act > 0.0`. A blank generation
(`str(g).strip() == ""`) is never probed: it scores `max_act = null`,
`activated = False` — a real miss, not missing data — and still counts in
its cell's declared population, lowering the activation rate rather than
being excluded from it. Off-grid cells (a method's undeclared factor, e.g.
PromptSteering/LoRA outside factor 1.0) get `null` in all three score
columns instead of `False`, so an undeclared cell stays distinguishable
from a declared miss. Texts are deduplicated per concept, in
first-occurrence order, before probing; one dense SAE pass per concept
scores every declared cell for that concept.

The probe identity is fixed and derived from the merged frame's own
`sae_link`/`sae_id` columns rather than passed as a flag: every concept's
`sae_link` (`https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k/<idx>`)
must decode to the same gemma-2-2b layer-20 `gemma-scope-2b-pt-res-canonical`
SAE (SAELens id `saelens_sae_id = layer_20/width_16k/canonical`), and its
`sae_id` column (the int64 Neuronpedia feature index) must equal the feature
index encoded in that link — only the per-concept feature index varies. A
frame naming any other release, a mix of releases, or a `sae_id` that
disagrees with its `sae_link` (including the vendored `sae_id = 0` fallback)
fails before any model load.

Run it on the merged frame:

```bash
uv run python -m scripts.eval.axbench_upstream.run_roundtrip \
  --input out/axbench/runs/c2/merged_v4/steering_data.parquet \
  --output-dir out/axbench/runs/c2/roundtrip-v1
```

For the seven-method baseline-plus-project NL frame, add
`--include-nl-steering`. The sidecar then carries the same three roundtrip
columns for `NLSteering`, records the selected method order in its scoring
metadata, and retains the default ten-method schema when the flag is absent.

`--input` must be named `steering_data.parquet` and pass the same
`validate_steering_frame(source="merged", ...)` check and canonical
`(concept_id, input_id, factor)` row order that `run_evaluation` requires —
the already-evaluated 104-column frame is rejected by the exact-column
check. `--output-dir` must be absent or empty, and path-disjoint from the
input's own directory; there is no `--force` and no resume, so use a fresh
output directory for every attempt. `--device` defaults to `auto` (CUDA if
available, else CPU); pass an explicit `cuda[:N]` or `cpu` to override. Those
two device types are the only supported ones — any other accelerator (`mps`,
`xpu`, …) has no nnterp backend here and is refused loudly rather than scored
on a third, incomparable numeric path.

The sidecar never modifies `steering_data.parquet` and writes only inside
`--output-dir`, so it is safe to run next to (or read from) an
[AxBench-Adv](axbench-adv.md) source directory without disturbing that
pipeline's own source contract.

### Determinism and backend

On CUDA the nnterp backend is required: a failed `build_nnterp_model` raises
rather than silently falling back to the far slower and numerically
divergent TransformerLens path, so every CUDA run stays comparable to every
other CUDA run. On CPU there is no nnterp to build, so TransformerLens is
used intentionally. Either way, the backend actually used (`nnterp` or
`transformerlens`) is recorded in the manifest. Scoring is otherwise
deterministic given identical input bytes, device, backend, and library
versions: fixed canonical row order, fixed method order, first-occurrence
per-concept text deduplication, a fixed batch size (`ROUNDTRIP_BATCH_SIZE =
8`, a module constant, deliberately not a CLI flag), and no RNG.
Cross-backend and cross-device bitwise equality is not promised.

### Prerequisites

- A local GPU for real runs — fp32 gemma-2-2b is roughly 10.5 GB VRAM plus
  the SAE.
- `HF_TOKEN` with access to the gated `google/gemma-2-2b` repo (~10 GB
  download) and `google/gemma-scope-2b-pt-res` (the l20/16k canonical SAE
  parameters).
- A completed merged `steering_data.parquet` (see
  [Validate and merge](#validate-and-merge) above).

### Output artifacts

`roundtrip_data.parquet` — same row count and canonical row order as the
input frame, with five copied key/identity columns plus three columns per
selected method. The ten-method default has 35 columns in
`tuple(METHOD_FACTORS)` order. The explicit NL product has 26 columns in
`DiffMean`, `LinearProbe`, `PromptSteering`, `LoRA`, `SelfConditioning`,
`SelfConditioningMasked`, `NLSteering` order.

| Column | dtype | Semantics |
|---|---|---|
| `concept_id` | int64 | key, copied from input |
| `input_id` | int64 | key, copied from input |
| `factor` | float64 | key, copied from input |
| `sae_link` | string | copied from input (per-concept constant) |
| `sae_id` | int64 | copied from input (already validated against `sae_link`) |
| `{m}_roundtrip_max_act` | float64 | max non-BOS post-activation; `NaN` off-grid and for blanks |
| `{m}_roundtrip_activated` | boolean (nullable) | null off-grid; `False` for blanks; else `max_act > 0.0` |
| `{m}_roundtrip_blank` | boolean (nullable) | null off-grid; else whether the generation was blank |

`roundtrip_summary.parquet` — one row per declared `(method, factor)` pair,
method-major in the same selected order as the sidecar, with ascending factors
within each method. The default's six full 12-factor grids plus four
single-factor grids produce `6 × 12 + 4 × 1 = 76` rows. The NL product's
five full grids plus two single-factor grids produce `5 × 12 + 2 × 1 = 62`
rows:

| Column | dtype | Semantics |
|---|---|---|
| `method` | string | one of the selected default or NL-product methods |
| `factor` | float64 | a factor that method declares |
| `n_declared` | int64 | declared population at this `(method, factor)`: 10 inputs × the run's concept count |
| `n_blank` | int64 | blank generations among them |
| `n_activated` | int64 | activated cells among them (a blank counts as not activated) |
| `activation_rate` | float64 | `n_activated / n_declared`, pooled over concepts |
| `concept_macro_sem` | float64 | sample SEM (`ddof=1`) of per-concept rates; `NaN` under 2 concepts |

There is deliberately no macro-rate column. The declared grid is balanced —
every concept contributes exactly 10 cells at each `(method, factor)`, which
`build_summary` refuses to summarize otherwise — so the mean of the
per-concept rates equals `activation_rate` identically. `concept_macro_sem` is
the part of the per-concept breakdown that pooling really does lose: it says
whether a pooled rate reflects a uniform effect across concepts or a couple of
dominant ones, which (see the GPU suite's separation measurements) is a real
distinction for this metric.

`roundtrip_manifest.json` is written LAST, so its presence on disk is the
completion marker for a run — a failed or interrupted attempt leaves either
no output directory at all, or the two Parquets without a manifest. Fields:

- `schema_version`: `1`.
- `input`: `path`, `sha256`, `n_rows`, `max_concepts`.
- `probe`: `model_name`, `hf_model_id`, `release`, `saelens_sae_id`,
  `hook_name`, `layer`, `activation`, `width`, `n_sae_features`, `l0`,
  `model_dtype`, `sae_dtype`, `backend` (`nnterp` or `transformerlens`),
  `device`, `sae_weights_sha256`, `n_ctx`. `saelens_sae_id` is SAELens' string
  id for the loaded SAE (`layer_20/width_16k/canonical`) — deliberately not
  named `sae_id`, which in the sidecar and the merged frame is the int64
  Neuronpedia FEATURE INDEX.
- `versions`: installed package versions for `sae-lens`, `transformer-lens`,
  `transformers`, and `torch`.
- `scoring`: `batch_size`, `activation_threshold` (`"post_act > 0"`), and
  `blank_policy` (`"count_as_not_activated"`). The explicit NL product also
  records its selected method order in `methods`; the default schema is
  unchanged.
- `accounting`: two keys. `per_method` maps each selected method to an
  `{n_declared, n_blank, n_scored}` object; `n_unique_texts_per_concept_sum`
  is the sum, over concepts, of each concept's own count of unique probed
  texts. Deduplication is per concept, so a text repeated under two concepts
  is encoded, and counted, twice; the number is the probe's real encode
  workload, not a global distinct-text count.

## Disjoint project concept subsets

`run_methods` accepts optional `concept_ids=[50,51,52]` with `max_concepts=100`.
Supply a complete, validated c100 main or model-free project input artifact;
IDs remain global. Only the selected concepts are fitted and generated. Both
project methods retain all twelve factors and ten prompts per concept, with
unchanged global-ID-derived selection and token seeds. NLSteering subsets are
not supported.

`benchmark_axbench_upstream compare --kind project` compares native bundles
with the same subset. `profile --kind project --override 'concept_ids=[50,51,52]'`
traces the smallest selected concept; trace ownership follows the selected
prompt groups.

Prompt preparation replays the **complete input** in canonical order because
checkpoint feature selection consumes shared seeded state. Slicing this
preparation would change selected prompt features. Independent processes must
use disjoint, new output/manifest parent directories. Their native manifests,
cell resume identities, fit identities and fit receipts bind the explicit
subset; existing full-prefix configurations retain their prior identity shape.
Do not pass `force=true` or reuse historical output directories when extending
a campaign. Concatenated outputs need an operational merge receipt referencing
the original native manifests; they are not a single native run.

## Native decoder-only SFT prompt steering

`NativeDecoderPromptSteering` evaluates an explicit native nanochat SFT `.pt`
checkpoint with its trained RustBPE conversation renderer. It does not use an
SAE, predicted features, fitted directions, or a Hugging Face model conversion.
Its only factor is `1.0`: this names one natural-language prompt condition and
is not comparable to an encoder-decoder feature-strength sweep.

The input is the frozen campaign auxiliary `steering_data.parquet`, with ten
AlpacaEval instructions per concept. The producer verifies its supplied SHA-256,
recovers the exact per-concept boosted steering instruction and original question
from `steered_input`, and renders that user text using the native chat tokenizer.
It never synthesizes replacement steering prompts or reruns a baseline. The
original question is preserved for the instruction-following judge.

```bash
uv run python -m scripts.eval.axbench_upstream.run_native_decoder \
  source_path=/path/to/aux/inference/steering_data.parquet \
  source_sha256=<verified-sha256> checkpoint_dir=/path/to/standard_steered_sft_checkpoints \
  step=1003 tokenizer_path=/path/to/tokenizer/tokenizer.pkl \
  output_dir=/fresh/native-output max_concepts=50
```

Set `NANOCHAT_BASE_DIR` to the checkpoint's tokenizer cache root. Keep `output_dir`
disjoint from the frozen input. The full rendered prompt plus the complete
generation budget must fit the checkpoint context; validation and generation use
the same untruncated tokens. Immutable per-cell receipts allow a bounded `max_cells` pilot to
resume into the same production identity. Model, metadata, tokenizer, input,
prompt policy, sampling, resolved cached-attention backend and generation-source
identities bind resume.

The dedicated `scripts.eval.axbench_upstream.run_native_evaluation` Hydra entry
point uses `native_evaluate.yaml`, selects only this method in the unchanged
vendored `LMJudgeEvaluator`, and preserves concept relevance, original-instruction
relevance, fluency and their harmonic mean. Required paths are `input_path`,
`manifest_path`, `output_dir` and `judge_cache`. Completed outputs include raw
judge completions, ratings, provider call accounting and an integrity receipt.
Neither the ten-method merger nor its default method set changes.

## Native decoder-only standalone baselines

`DiffMean`, `LinearProbe`, and `LoRA` can also be fit and generated directly on
the native RustBPE decoder checkpoint. These are standalone baselines, not the
negative additive counter-interventions used by AxBench-Adv. Every method uses
the canonical `original_prompt` without the PromptSteering prefix and fits only
the prepared `train_data.parquet` rows for its own concept. That artifact and
the AlpacaEval generation source are independently hashed; an identical generic
input string in both corpora does not collapse or transfer rows between them.

DiffMean is the token-weighted positive-minus-negative mean at the reviewed
depth-relative decoder block output (layer 9 for D12, layer 11 for D14).
LinearProbe is the normalized coefficient from the canonical L1/liblinear
row-mean probe at that same site. For both methods, a factor means
`factor * positive_fit_max_activation * unit_direction`; it is not a raw
counter-vector magnitude. The fixed standard grid is
`[0, .2, .4, .6, .8, 1, 1.5, 2, 3, 4, 5, 6]`, and factor zero installs no hook.

LoRA is trained separately for every concept from its positive TRAIN
continuations while the base checkpoint stays frozen. Its fixed contract is
rank 4 rsLoRA, alpha 32, dropout 0.05, learning rate 0.0009, 24 epochs, batch
size 6, gradient accumulation 6, and weight decay 0. The upstream Qwen layers
`[7, 14, 21, 28]` map by relative depth to D12 layers `[2, 5, 7, 9]` and D14
layers `[3, 5, 8, 11]`. The shared resolver uses round-half-up,
`floor(target_n_layer * qwen_layer / 36 + 0.5)`; the steering layer is Qwen
layer 28 under the same mapping. Only the reviewed D12/768/1024 and
D14/896/1024 native geometries are admitted; the loaded D14 model must contain
exactly 399,115,836 parameters. This is an architecture mapping, not evidence
that these layers are empirically optimal. Nanochat attention
`c_proj` is the architecture-specific counterpart of the
vendored wrapper's attention output projection. LoRA has only factor 1.0, and
the producer requires actual nonzero adapter updates before publishing a fit.

The current-main native adapter renders the full user conversation with
`render_conversation(max_tokens=None)` and appends `<assistant_start>`. It uses
the same seeded streaming sampler and BOS/assistant-end stops, and rejects
prompt plus generation budgets exceeding the checkpoint context. The identity
records this renderer and the resolved attention backend. Set
`NANOCHAT_BASE_DIR` so `tokenizer_path` is exactly its `tokenizer/tokenizer.pkl`.
Use fresh output roots when changing implementation; prior PromptSteering
artifacts remain valid input carriers and are never rewritten.

On SM12, fitting admits the existing steering-owned native FA4 adapter before
loading the decoder. This avoids the compiled FlexAttention sparse-block/tile
mismatch while preserving causal and sliding-window attention. No backend
override is needed. The fit identity records `full_prefix_attention_backend`
(`native_fa4` on SM12) separately from `attention_backend`, which describes
cached generation (SDPA on SM12), and hashes the shared `sm12_fa4.py` adapter.
Missing native FA4 fails admission; it does not silently select another kernel.

Fit and generation share one entry point and can be run independently:

```bash
uv run python -m scripts.eval.axbench_upstream.run_native_baselines \
  phase=fit methods='[DiffMean,LinearProbe,LoRA]' \
  source_path=/path/to/standard100/steering_data.parquet \
  source_sha256=<verified-sha256> prepared_dir=/path/to/axbench/prepared \
  checkpoint_dir=/path/to/standard_steered_sft_checkpoints step=1003 \
  tokenizer_path=/path/to/tokenizer/tokenizer.pkl output_dir=/fresh/baselines

uv run python -m scripts.eval.axbench_upstream.run_native_baselines \
  phase=generate methods='[DiffMean,LinearProbe,LoRA]' \
  source_path=/path/to/standard100/steering_data.parquet \
  source_sha256=<same-sha256> prepared_dir=/path/to/axbench/prepared \
  checkpoint_dir=/path/to/standard_steered_sft_checkpoints step=1003 \
  tokenizer_path=/path/to/tokenizer/tokenizer.pkl output_dir=/fresh/baselines
```

For D14, select `--config-name=native_baselines_d14` and supply the actual final
SFT `step`, `expected.model_sha256`, `expected.meta_sha256`, and
`expected.tokenizer_sha256`. They intentionally have no defaults. The producer
records those identities in every fit/run identity; D12 retains its historical
defaults and artifact schema.

Defaults require global concept IDs 0 through 99 and ten native test prompts
per concept. `methods` and `concept_lo`/`concept_hi` are explicit, authenticated
work partitions; use separate output roots for independent partitions. Each
method receives its own fit and run manifest, immutable per-cell receipts, and
Parquet. No combined method manifest is synthesized. `max_cells` is a
placement-only interruption/resume bound and does not alter semantic identity.

Judge each completed method independently with native target-only judging:

```bash
uv run python -m scripts.eval.axbench_upstream.run_native_baseline_evaluation \
  method=DiffMean input_path=/path/to/methods/diffmean/steering_data.parquet \
  manifest_path=/path/to/methods/diffmean/run_manifest.json \
  output_dir=/fresh/diffmean-judge judge_cache=/path/to/judge-cache
```

The judge validates the complete method factor grid and producer identity
before any provider request. It does not merge methods or rerun baselines.
