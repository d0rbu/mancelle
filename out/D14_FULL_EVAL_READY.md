# D14 full native evaluation blocker handoff

Implementation SHA: `acd0a8886b0c8d4e87e0387a6b9d43653ac41a3c`

Checkout base: `5341233bd29501166911b429d053b55042465e03`

Status: the native PromptSteering own-direction counter blocker is implemented
and CPU-qualified. No GPU job, SSH action, provider call, upload, PR creation,
or merge was performed from this checkout. The actual final D14 SFT step and
model/meta/tokenizer hashes do not exist yet while training is ongoing; the D14
counter preset deliberately requires root to supply them.

## Integrated implementation and ancestry

The requested reviewed implementation was integrated, not duplicated:

1. `38869d8112e42696ec57f18477e41b08d39f0aba` became
   `77d9e46d8741427e3ef8496c89feae6732b839ca` (only additive conflicts in
   `docs/evaluation/axbench-adv.md` and `docs/reference/file-reference.md`).
2. `e912044e3b3520478fd507411118d6e1543f2710` became
   `51308b667cd499702b928814670186cb7d808161`.
3. The raw-calibration implementation depends on reviewed ancestor
   `d24153f57cd1dddef73a6eec53b99ef544c3a6d4`; current main did not contain it,
   so it was integrated as `5a3d1240fddc260a8136f660a47f8bb493f5af20`.
4. D14 and standard100-to-Adv50 support is
   `6ca2e92b07748e9bd833a9ce6034476027cfc3aa`, with current-main adaptation
   and formatting in the implementation SHA above.

The stack is based on the checkout's existing D14 standalone/method-Adv contract
`9cd29174d7408a8c4fdd1cb0b9948ac2e2dc716c` and authenticated zero-control
judge replay `1b72cfadd4ab2de92e5609cfdb31d8aa0183ee59`.
`nanochat/` was not modified.

## Contract now enforced

- Legacy 50-source/50-counter operation retains the same complete-source
  behavior and manifest shape. `concept_count: null` adds no semantic scope
  field.
- `native_counter_d14.yaml` requires the actual final step plus exact model,
  metadata, and tokenizer SHA-256 values. It admits only D14/896/1024, layer 11,
  and exactly 399,115,836 loaded parameters through the already-reviewed
  `NativeModelContract`.
- The only new subset is `max_concepts=100, concept_count=50`, meaning the
  canonical prefix 0..49. Arbitrary slicing is rejected.
- The complete authenticated standard100 manifest and Parquet hashes remain in
  run identity, zero-control provenance, fit contracts, point identity, and the
  point's `counter_extension.source_scope`. Fits and cells from a legacy50 or
  different source identity cannot resume into this scope.
- Fit and generation loops cover only concepts 0..49. Each published point
  contains exactly 500 canonical rows and declares 50 concepts to the unchanged
  `load_native_output` / target-only native judge.
- Original PromptSteering generations are never regenerated or rewritten.
  Zero-control aggregation must select concepts 0..49 from the authenticated
  judged standard100 output and retain the hashes recorded in `control.json`.
- DiffMean and LinearProbe remain TRAIN-only canonical AxBench fits with the
  reviewed raw signed-calibration behavior, exact native prompts/seeds, and the
  closed magnitude grid 0.25/0.5/1/2.

## Root commands after final SFT exists

Replace every angle-bracket value with the measured artifact value or absolute
path. Do not reuse the old D12 step/hashes.

### 1. Generate NativeDecoderPromptSteering standard100 once

```bash
NANOCHAT_DTYPE=bfloat16 uv run --no-sync python -m scripts.eval.axbench_upstream.run_native_decoder \
  source_path=/ABS/AXBENCH_AUX100/steering_data.parquet \
  source_sha256=<AXBENCH_AUX100_PARQUET_SHA256> \
  checkpoint_dir=/ABS/D14_FINAL_SFT_CHECKPOINTS \
  step=<ACTUAL_FINAL_SFT_STEP> \
  tokenizer_path=/ABS/TOKENIZER/tokenizer.pkl \
  output_dir=/ABS/D14_NATIVE_PROMPT_STANDARD100 \
  max_concepts=100 \
  max_cells=null
```

Judge that standard100 output (provider work requires separate authorization):

```bash
uv run --no-sync python -m scripts.eval.axbench_upstream.run_native_evaluation \
  input_path=/ABS/D14_NATIVE_PROMPT_STANDARD100/steering_data.parquet \
  manifest_path=/ABS/D14_NATIVE_PROMPT_STANDARD100/run_manifest.json \
  output_dir=/ABS/D14_NATIVE_PROMPT_STANDARD100_JUDGED \
  judge_cache=/ABS/PRIVATE/D14_NATIVE_JUDGE_CACHE.sqlite \
  max_concepts=100 \
  cache_only=false
```

### 2. Fit and generate own-direction counter Adv50 from standard100

```bash
NANOCHAT_DTYPE=bfloat16 uv run --no-sync python -m scripts.eval.axbench_adv.run_native_counter \
  --config-name=native_counter_d14 \
  source_manifest=/ABS/D14_NATIVE_PROMPT_STANDARD100/run_manifest.json \
  source_parquet=/ABS/D14_NATIVE_PROMPT_STANDARD100/steering_data.parquet \
  prepared_dir=/ABS/AXBENCH_PREPARED \
  checkpoint_dir=/ABS/D14_FINAL_SFT_CHECKPOINTS \
  step=<ACTUAL_FINAL_SFT_STEP> \
  tokenizer_path=/ABS/TOKENIZER/tokenizer.pkl \
  expected.model_sha256=<ACTUAL_MODEL_SHA256> \
  expected.meta_sha256=<ACTUAL_META_SHA256> \
  expected.tokenizer_sha256=<ACTUAL_TOKENIZER_SHA256> \
  output_dir=/ABS/D14_NATIVE_COUNTER_ADV50 \
  phase=all
```

The preset already fixes `max_concepts=100`, `concept_count=50`, layer 11,
both counter families, and all four magnitudes.

Judge each of the eight nonzero counter points with the unchanged target-only
native judge:

```bash
for POINT in \
  diffmean/m0p25 diffmean/m0p5 diffmean/m1 diffmean/m2 \
  linear_probe/m0p25 linear_probe/m0p5 linear_probe/m1 linear_probe/m2
do
  uv run --no-sync python -m scripts.eval.axbench_upstream.run_native_evaluation \
    input_path=/ABS/D14_NATIVE_COUNTER_ADV50/points/${POINT}/steering_data.parquet \
    manifest_path=/ABS/D14_NATIVE_COUNTER_ADV50/points/${POINT}/run_manifest.json \
    output_dir=/ABS/D14_NATIVE_COUNTER_ADV50_JUDGED/${POINT} \
    judge_cache=/ABS/PRIVATE/D14_NATIVE_JUDGE_CACHE.sqlite \
    max_concepts=50 \
    cache_only=false
done
```

The magnitude-zero control is not another generation/judge run. Select
concepts 0..49 from `D14_NATIVE_PROMPT_STANDARD100_JUDGED` and require the
standard100 manifest/Parquet hashes and canonical-prefix scope recorded by the
counter's `control.json`.

### 3. Run PromptSteering GEPA Adv50 from the same standard100 source

```bash
NANOCHAT_DTYPE=bfloat16 uv run --no-sync python -m scripts.eval.axbench_adv.run_native_decoder \
  source_manifest=/ABS/D14_NATIVE_PROMPT_STANDARD100/run_manifest.json \
  source_parquet=/ABS/D14_NATIVE_PROMPT_STANDARD100/steering_data.parquet \
  output_dir=/ABS/D14_NATIVE_GEPA_ADV50 \
  mailbox_dir=/ABS/PRIVATE/D14_NATIVE_GEPA_MAILBOX \
  concept_ids='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]' \
  proposal_counts='[1,2,4,8]'
```

The GPU process only exchanges durable mailbox artifacts. Root separately owns
the authorized provider worker.

Judge the five GEPA levels:

```bash
uv run --no-sync python -m scripts.eval.axbench_adv.judge_native_points \
  points_dir=/ABS/D14_NATIVE_GEPA_ADV50/points \
  output_dir=/ABS/D14_NATIVE_GEPA_ADV50_JUDGED \
  judge_cache=/ABS/PRIVATE/D14_NATIVE_JUDGE_CACHE.sqlite \
  max_concepts=100 \
  concept_ids='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]' \
  proposal_counts='[1,2,4,8]'
```

## CPU qualification

- Focused new counter file: `12 passed, 18 warnings in 8.10s`.
- Affected native PromptSteering, native standalone, own-counter, method-Adv,
  direction fitting, final judge, provider, and recovery matrix:
  `314 passed, 2 skipped, 2 deselected, 19 warnings in 32.99s`.
  The two skips are the opt-in authentic recovery-artifact cases.
- Ruff check and format check over all touched Python files: passed.
- `ty` over all touched Python files: passed.
- D14 Hydra composition with representative required step/hash overrides:
  passed; it resolved D14/896, layer 11, standard100 source, prefix50 output.
- `git diff --check`: passed.

No full repository suite was run.

## Remaining root-owned qualification

1. After pretraining and full SFT complete, measure and record the actual final
   step and the model, metadata, and tokenizer SHA-256 values. Confirm metadata
   D14/896/1024 and loaded parameter count 399,115,836.
2. Complete the already-running 4090 qualification in its separate checkout;
   do not treat this CPU work as that result.
3. Run a minimal D14 counter fit/generation GPU slice at layer 11, check finite
   nonzero directions, intervention cleanup, exact resume, and source/point
   hashes before the full eight-point generation.
4. Generate PromptSteering standard100 exactly once, judge it, then use those
   immutable bytes for both counter Adv50 and GEPA Adv50. Verify the zero control
   produces zero new generation calls and correctly selects judged concepts
   0..49 with recorded standard100 provenance.
5. With separate provider authorization, complete GEPA mailbox work and all
   target-only final judging. Reconcile call accounting, caches, point coverage,
   aggregates, and plots.
6. Complete the remaining full D14 ChatCORE and AxBench100 four-method campaign,
   archive receipts, upload only when authorized, and create the PR. GPU
   performance, provider results, uploads, and campaign success remain unclaimed.
