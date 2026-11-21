# Encoder-decoder baseline latent counters

This lane attacks `EncDecPromptSteering`, `EncDecDiffMean`,
`EncDecLinearProbe`, and `EncDecLoRA` on the 400M encoder-decoder SFT target.
It uses the completed encoder-decoder positive producer and **its own judged
factor sweep**. The generic bare baseline names still refer to vendored
models; D12/D14 native decoder results are separate.

The positive direction hook acts at decoder block output **before** cross
attention. The counter uses the project's **post-cross-attention** residual
site. Its coefficient is exactly `residual -= magnitude * saved_direction`;
the positive factor, activation calibration, and model width do not scale it.
Prompt features remain real SAE features, and continuation uses pure
self-conditioning with the checkpoint's own cadence. Every output has at most 128 new
tokens, temperature 1, top-k 50, and the positive producer's per-cell seeds.

## Inputs and qualification

Finish the c50 positive producer and judge its ten-method merged frame first.
Set the config's `positive_root`, source main/aux paths, `judged_input_path`,
`judged_dir`, `feature_statistics_path`, and checkpoint paths to those exact
artifacts. `phase=preflight` validates all 50 producer concepts, the durable
cell store, fit receipt, target weights/metadata, tokenizer, statistics,
prepared inputs, source rows and prompt-steering text, judged generation
columns, raw judge ratings, and factor selection. It also validates the
completed DiffMean and LinearProbe counter-fit manifest and every direction.
This phase requires the trained tokenizer cache through `NANOCHAT_BASE_DIR`.
`COUNTER_FITS` must contain `fit_manifest.json` (the existing campaign uses
`local-encdec-staging/fits/directions`).

```bash
COMMON=(
  "positive_root=$POS" "source_main_path=$MAIN" "source_aux_path=$AUX"
  "prepared_dir=$PREPARED" "target_meta_path=$META1013"
  "feature_statistics_path=$STATS"
  "judged_input_path=$MERGED/steering_data.parquet" "judged_dir=$POS_JUDGED"
  "counter_root=$COUNTER_FITS"
)
uv run python -m scripts.eval.axbench_adv.run_encdec_baseline_counter \
  phase=preflight "${COMMON[@]}"
```

The default `[256,1024,4096,8192,16384,32768,65536]` is a **candidate**
grid from other positives on the same target. Qualify one real GPU input for
all four positives, both counter families, and zero replay without provider
calls:

```bash
uv run python -m scripts.eval.axbench_adv.run_encdec_baseline_counter \
  phase=pilot 'concept_ids=[0]' 'counter_magnitudes=[256]' \
  "output_dir=$RUN/pilot" "${COMMON[@]}"
```

The one-input pilot is a wiring check. To choose the production grid, use
`phase=generate` on several concepts with all seven candidate magnitudes,
then judge and finalize that subset. Inspect concept, fluency, instruction,
harmonic quality, and fitness against the fresh zero for **each** of the
eight positive/counter pairs. Keep the incoherent-collapse endpoint visible;
do not treat a low concept rating with collapsed fluency as a successful
counter. Pin the reviewed grid before starting the disjoint c50 shards.

## Generation, judgment, and merge

Each `phase=generate` output directory has one immutable identity, one
positive-only shared zero per method/concept/input, a fresh magnitude-zero
replay that must match it exactly, and one durable record per nonzero point.
When the current runtime matches the producer runtime, zero is also required
to reproduce the archived positive generation. Across different runtimes the
archived output is not used as a plotted control or claimed equal. The
identity binds model, source and fit hashes, chosen factors, grid, runtime,
concept subset, generation settings, repository commit, and dirty source content.
Resume and merge require the same commit and source state, including untracked
files. Keep generated artifacts under ignored `out/` or outside the repository
so they do not change the source identity. Commit implementation changes before
starting a campaign.
Cell records refer to that identity; the fit inventory is bound once per run.
Completed cell records provide durable progress, with counts reported in logs.
Interrupted atomic-write scratch files do not count as completed cells.
Every shard prepares the full producer prompt population in the original
order before generating its selected concepts: prompt feature sampling shares
RNG state across prompts, so preparing only a subset would change conditioning.

```bash
uv run python -m scripts.eval.axbench_adv.run_encdec_baseline_counter \
  phase=generate 'concept_ids=[0,1,2,3]' "output_dir=$RUN/shard-00" \
  "${COMMON[@]}"
```

Repeat with disjoint ascending `concept_ids` and separate output directories.
Use the **same** other overrides for `phase=judge` and `phase=finalize` in
each directory. Both default to `cache_only=true`; absent cached responses
fail closed. A live provider run additionally requires explicit
`allow_live_provider=true cache_only=false` after authorization. The inner
judge records concept and fluency; finalization uses the existing vendored
AxBench judge for instruction following and verifies its concept/fluency
parity. No GEPA or model fitting is performed in this lane.

```bash
uv run python -m scripts.eval.axbench_adv.run_encdec_baseline_counter \
  phase=judge 'concept_ids=[0,1,2,3]' "output_dir=$RUN/shard-00" \
  "judge_cache=$CACHE" cache_only=true "${COMMON[@]}"
uv run python -m scripts.eval.axbench_adv.run_encdec_baseline_counter \
  phase=finalize 'concept_ids=[0,1,2,3]' "output_dir=$RUN/shard-00" \
  "judge_cache=$CACHE" cache_only=true "${COMMON[@]}"
```

Finalization writes a per-shard native
`manifest.json`, per-point input/judged artifacts, `per_concept.parquet`,
and `aggregate.parquet`. The score tables have all three axes, two- and
three-way harmonic quality, fitness, means, and standard errors, with method,
pressure, magnitude, and population columns for concept-vs-budget plots.

Merge only complete disjoint shards covering c0–49. Merge verifies their
identities, hashes, runtime equality, point population, and exact judged
generation parity before writing the combined plots table and manifest:

```bash
uv run python -m scripts.eval.axbench_adv.run_encdec_baseline_counter \
  phase=merge 'shard_roots=[/path/shard-00,/path/shard-01]' \
  "output_dir=$RUN/merged" "${COMMON[@]}"
```

This document describes the implementation path. A completed c50 attack
result requires the positive producer, judged positive factor sweep, GPU
qualification, grid-quality pilot, all judged shards, and checked merge.
