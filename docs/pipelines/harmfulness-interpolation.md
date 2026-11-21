# Harmfulness safety interpolation

Encoder-decoder harmfulness evaluation uses the same two-sided interpolation
policy as ChatCORE and AxBench. `feature_configs` is rejected, including empty
maps; decoder-only evaluation retains its existing behavior. The unsteered
encoder-decoder path is pure-model self-conditioning with real prompt features.

## Fit a held-out target

```bash
NANOCHAT_BASE_DIR=/path/to/runtime .venv/bin/python -m scripts.eval.build_harmfulness_safety_target \
  checkpoint_meta_path=/path/to/meta_001116.json \
  stats_path=/path/to/feature_statistics.safetensors output_dir=/path/to/safety-target
# After reviewing preparation.json and corpus.json, repeat with dry_run=false.
```

The producer uses `allenai/wildguardmix`, `wildguardtrain/train`, at revision
`d29c47f41c8b51348b5c8e8c81c039b3132b66d1`. A positive must have a nonempty response and satisfy either:

- `prompt_harm_label=harmful` and `response_harm_label=unharmful` (safe
  responses to harmful prompts, including refusals).
- `prompt_harm_label=unharmful`, `response_harm_label=unharmful`, and
  `response_refusal_label=compliance` (safe non-refusals of benign requests).

The pinned WildGuard label for a non-refusal is **`compliance`**, not
`not_refusal`. Benign refusals (`response_refusal_label=refusal`) are excluded;
response text is never used to infer refusal. Null annotations are explicitly
unlabelled: they exclude a row when needed by the rule; a null refusal label
is allowed in the harmful-prompt population. Every label field must exist, and
unknown values fail even on rows otherwise excluded. One shared
`safety_fit_selection` function governs both preparation and bundle admission.
These annotations do not guarantee helpfulness. No evaluation
responses or Nemotron SFT data are used. All prompts from BeaverTails-Evaluation
(`a93205d9ba6cc9dc398a34466da3602abddd2d40`) and WildGuardTest are excluded by NFKC,
case folding, and collapsed whitespace before label selection, deduplication,
length filtering, or the cap. Exclusion covers the full test sources, including
prompts outside the selected runtime slice. Blank training prompts are explicitly
audited exclusions. Malformed schemas and unknown labels fail.

The background is the ChatCORE builder's pinned `HuggingFaceTB/smol-smoltalk`
train corpus (`f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`), shuffled by the same
seed-42 adapter and reduced to the first complete exchange with optional system
text. The shared selector excludes exact normalized evaluation and candidate training
messages from this background. Safety uses exact-message matching because
WildGuard includes one-letter inputs such as `i`: treating those as substrings
would exclude essentially every English background document. ChatCORE retains
its existing embedded-input exclusion policy. Defaults remain **128 positives and 1,024 background documents**; smoke runs
may reduce both counts. Positives use natural order after
`numpy.random.default_rng(42).permutation(training_rows)`, retaining original
source row indexes. We do not stratify: this preserves the eligible corpus's
natural mix rather than imposing an unrequested balance. The cap applies after
label, overlap, duplicate, and whole-document length exclusions. After the cap,
the full label/overlap audit continues without tokenizing capped rows.
Bundle admission checks accepted and excluded source-row order against that
permutation, and background rows against their adapter order, as well as
accounting for every visited row exactly once.
Both requested local v2 refits selected **81 harmful-prompt safe responses and
47 benign safe non-refusals** (128 total); their accepted corpora agree exactly.
`subpopulation_counts` records both accepted populations (including zero counts)
in `fit-audit-summary.json` and the bundle's preparation provenance, together
with the exact order policy and seed. Extraction defaults to batch one.

Full prompt-plus-reference conversations are rendered by the actual RustBPE chat
template, excluding its initial BOS in extraction text. Overlength documents
are excluded whole, never truncated. The existing Gemma/SAE extractor max-pools
each document, maps activations to checkpoint-statistics percentiles, applies
the checkpoint mask and representative cluster reduction, averages in float64,
and subtracts the background mean. Dividing that difference by 100 gives the
dense signed target, gathered in exact FP-head row order and stored as float32
`target.safety` alongside int64 `feature_ids` in `targets.safetensors`.

`preparation.json` and `corpus.json` are immutable and written before model
loading. The bundle binds source revisions, accepted response/text hashes,
exclusion counts and inputs, corpus digest, decoder/extraction tokenizers,
checkpoint metadata, source geometry, statistics, mask, clusters, vocabulary,
and source code identity. It uses the shared ChatCORE bundle format and stricter
safety population admission. The fit-rule identity is now
`wildguard_train_safe_responses_v2`; v1 bundles fail with an explicit stale-rule
refit message. Use a new output directory for the refit. A path cannot be reused
with different bytes.
The existing extraction loader does not attest an immutable Gemma weight
revision; this limitation is recorded explicitly in fit provenance.

## Generate without judging

```yaml
source: encdec
feature_configs: null
batch_size: 1
generate_only: true
conditioning:
  context_cache_dir: /path/to/prompt-cache
self_conditioned:
  safety_steering:
    targets: /path/to/safety-target/targets.safetensors
    strength: 2.5
```

Use `python -m scripts.eval.harmfulness_eval` with the exact checkpoint
`model_tag` and `step`. `self_conditioned.safety_steering: null` selects pure
self-conditioning. Strength is finite in `[0,6]`; interpolation weight is
`strength / 6`. Weight zero preserves the shared policy's raw-logit ranking
branch exactly. Other weights interpolate the signed sigmoid FP score toward
the target before positive/negative selection. No sparse map can be mixed in.

Before forwarding, the runtime checks target/checkpoint identity, compact row
order, prompt membership in the excluded evaluation population, and real cache
coverage. Receipts bind the target SHA-256 and strength; changed target bytes
are rejected. Turn checkpoints use the recorded
`turn_self_cond_segment_size_rustbpe`, not the Gemma segment length. Fixed
checkpoints likewise need the authenticated measured cadence/cache contract
used by ChatCORE; an old `self_cond_segment_size` is only an equality assertion. The
assistant-masked and standard checkpoints can share a target only when their
complete fit contracts and original feature row orders agree.

For model-free judging, pass the saved `responses_file`, or use `judge_only=true`
with the original checkpoint selector. Native SFT source chains inspect only
checkpoint metadata to select the response path; explicit files retain their
recorded condition label. Generation batch settings do not constrain judging.
Any configured safety target, strength, or cadence assertion must match the
saved receipt before a judge client is created.

## Build the prompt cache

Use the existing [turn cache pipeline](evaluation.md#turn-delimited-target-real-context-caches)
with `evals=harmfulness`, `num_seeds=1`, `base_seed` equal to evaluation `seed`,
and `harm_num_harmful`/`harm_num_benign` equal to the intended slice. The turn
preset is `build_eval_context_cache_turn`; enumerate, extract the parquet using
the checkpoint SAE/turn source geometry, then convert and validate. Harmfulness
rows are independent prompt prefixes ending in a marker-only assistant turn;
they never contain reference answers. Use extraction `gpu_batch_size=1` and
`cpu_batch_size=1` for a bounded local smoke.

After conversion, validate the persisted cache against the checkpoint and its
original JSONL, parquet, and extracted feature shards (no GPU forward required):

```bash
NANOCHAT_BASE_DIR=/path/to/runtime .venv/bin/python -m scripts.eval.build_eval_context_cache \
  --config-name=build_eval_context_cache_turn stage=validate evals=harmfulness \
  checkpoint_meta_path=/path/to/meta_001116.json \
  stats_path=/path/to/feature_statistics.safetensors \
  contexts_file=/path/to/prompt-cache-inputs/contexts.jsonl \
  context_parquet_dir=/path/to/prompt-cache-inputs/parquet \
  feature_dir=/path/to/prompt-cache-inputs/features \
  output_dir=/path/to/prompt-cache \
  +summary_file=/path/to/prompt-cache-validation.json
```

This replays source geometry, selection, vocabulary, and decoder routing for
every row, including masked checkpoints. A harmfulness-only cache reports
`target_feature_configs: 0` because it has no reference-answer targets. The
standalone `validate_eval_context_cache_turn` entry point shares these checks;
its default `enumerate_evals=true` additionally compares the live prompt
enumeration, so supply the original seed and harmful/benign counts there.

## Progression and campaign 2

Each checkpoint resolves and checks its own feature/source/statistics contract.
A requested encoder-decoder harmfulness progression requires an explicit
`self_conditioned` block. Use `self_conditioned: {safety_steering: null}` for
unsteered feedback; omitting the block fails before scheduling work.
A shared target is valid only for equal contracts. If identity changes, build a
separate target with that checkpoint metadata and select it through the
progression checkpoint override. Checkpoint weight changes alone do not require
a new target: extraction uses the external SAE, while the fit contract binds
its interpretation and vocabulary. See the progression section in
[evaluation](evaluation.md) and [campaign 2 integration](campaign2.md#encoder-decoder-safety-strengths).

A progression override has this shape (phase and step identify the exact cell):

```yaml
self_conditioned:
  safety_steering: {targets: /targets/shared/targets.safetensors, strength: 2.5}
self_conditioned_overrides:
  - phase: safety_sft
    step: 1125
    config:
      stats_path: /statistics/this-stage.safetensors
      conditioning: {context_cache_dir: /cache/this-stage}
      self_conditioned:
        safety_steering: {targets: /targets/this-stage/targets.safetensors, strength: 2.5}
```

Resume checks bind the resolved target bytes, strength, and context-cache
manifest/shard bytes per checkpoint. Existing self-conditioned progression
receipts without a cache digest are rerun once. The declared cache root's leaves
are hashed together; use a dedicated root to avoid invalidation from other
cache builds. Logs and validation summaries do not affect this digest.
Collectors accept the added receipt fields, and progression plots can select
an exact safety target digest and strength without conflating different runs.


## Strengths 0, 3, and 6

`self_conditioned_overrides` selects checkpoint inputs; it does not schedule
multiple strengths for one checkpoint. Collect three runs with a different
**parent directory** for each `results_file` (per-checkpoint receipts live in
its sibling `checkpoints/` directory). This prevents strength runs from sharing
resume files. A checkpoint override can change just `targets` and inherit the
current sweep strength.

The following is the exact local configuration for a **future judged study**
using the existing 8 harmful / 8 benign prompt cache. It calls the paid judge;
it is documented here and was not executed for the fit refinement. For a larger
study, rebuild/validate the prompt cache for the larger slice and change both
counts consistently. On the shared 4090, check compute PIDs before each launch,
wait 60 seconds while busy. The local `s32_v2/gpu_driver.py` helper caps the
allocator at 14 GiB and preserves Hydra module discovery; `timeout` bounds each
collection to 840 seconds. The collector resumes incomplete checkpoint/seed
receipts if a larger study needs several bounded launches.

```bash
export NANOCHAT_BASE_DIR=/home/d0rb/axbench_sft_local/turn_runtime
export NANOCHAT_DTYPE=bfloat16
for strength in 0 3 6; do
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    sleep 60
  done
  PYTHONPATH=. timeout 840 .venv/bin/python \
    /home/d0rb/axbench_sft_local/harm_targets/s32_v2/gpu_driver.py \
    scripts.train.training_progression \
    "/home/d0rb/axbench_sft_local/harm_targets/s32_v2/judged-strength-$strength-gpu.json" \
    mode=collect evals=harmfulness model_family=encoder_decoder \
    model_tag=full3b_clustered_c2048_long_w32_bf16_b14_fp \
    checkpoint_root=conditional_encdec_checkpoints checkpoint_prefix='' \
    'phases=[{phase:s32,subdir:standard_sft_conditioned_checkpoints,steps:[1116]},{phase:s32mask,subdir:standard_steered_sft_conditioned_checkpoints_mask,steps:[1125]}]' \
    num_gpus=1 device_type=cuda dtype=bfloat16 num_seeds=1 base_seed=42 \
    harm_num_harmful=8 harm_num_benign=8 harm_max_new_tokens=64 harm_temperature=0 \
    conditioning.conditioned_configs=null conditioning.target_real_configs=null \
    conditioning.context_cache_dir=/home/d0rb/axbench_sft_local/harm_targets/s32/context/cache \
    stats_path=/home/d0rb/axbench_sft_local/stats/feature_statistics.safetensors \
    "self_conditioned={safety_steering:{targets:/home/d0rb/axbench_sft_local/harm_targets/s32_v2/targets.safetensors,strength:$strength}}" \
    'self_conditioned_overrides=[{phase:s32mask,step:1125,config:{self_conditioned:{safety_steering:{targets:/home/d0rb/axbench_sft_local/harm_targets/s32mask_v2/targets.safetensors}}}}]' \
    "results_file=/home/d0rb/axbench_sft_local/harm_targets/strength_sweep/strength_$strength/results.json"
done
```

The plot CLI now accepts `progression_paths` to combine those separate
`training_progression` JSON collections. It uses the existing metric definitions
and arithmetic mean across judged seeds: harmful ASR, harmful score-1 safe rate,
benign compliance, and overall mean harmfulness score. The x axis names each
phase/checkpoint; each strength gets its own color and line. Receipt disagreements
between seeds, duplicate strength cells, mismatched checkpoint identities/cells,
and missing judged metrics fail loudly. Each input file must use one strength
for all checkpoints. Matching checkpoints must have the same seed set and
`strength_sweep_spec_sha256` receipt: this binds prompt counts, generation
settings, judge model/rubrics, and cache/statistics/target bytes while excluding
only strength and cache/statistics path locations. Resolved completion cadence
and runtime contracts must also agree. Older files remain readable individually,
but must be recollected with the current evaluator before combining strengths.
All strengths for one checkpoint must
use the same target digest; independently validated checkpoints can have
different targets. Strength labels preserve distinct finite values without
rounding them into the same curve. This does not infer a safety effect from generation.

```bash
.venv/bin/python -m scripts.eval.plot_harmfulness_progression \
  model_tag=full3b_clustered_c2048_long_w32_bf16_b14_fp \
  'progression_paths=[/home/d0rb/axbench_sft_local/harm_targets/strength_sweep/strength_0/results.json,/home/d0rb/axbench_sft_local/harm_targets/strength_sweep/strength_3/results.json,/home/d0rb/axbench_sft_local/harm_targets/strength_sweep/strength_6/results.json]' \
  output_dir=/home/d0rb/axbench_sft_local/harm_targets/strength_sweep/figures \
  safety_steering=null
```

Standalone evaluator YAML receipts already support separate strength lines in
all three harmfulness plots when `safety_steering=null`; `progression_paths` is
only needed for the collector's JSON format. Fixture tests render all four
three-strength charts without any judge calls.
