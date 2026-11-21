# AxBench-Base Evaluation Pipeline

Benchmark base (pre-instruction-tuned) language models on concept steering.
AxBench-Base runs 25 `(baseline, target_model)` method combinations across a
curated set of GemmaScope SAE concepts, in both **positive** (inject) and
**negative** (suppress) steering modes, and emits a harmonic-mean
leaderboard keyed on concept relevance and fluency.

The benchmark adapts the scoring rubric from Wu et al. 2025 (AxBench) to a
target that does not take instructions — a nanochat base decoder-only or
conditional encoder-decoder. See `docs/evaluation/axbench-base.md` for the full user
guide and `docs/evaluation/axbench-base-advanced.md` for package internals.

## Overview

```
Stage 0: prepare       candidate_concepts.json + concepts.json + seed pools
                        0a pure data    (CPU-only, Neuronpedia descriptions)
                        0b GPU + data   (exact-prefix production SAE extraction)
                        0c GPU + API    (optional QC LLM + negative-prefix verification)

Stage 1: synth         synth_training_{positive,negative}.parquet
                        1b GPU + API    (rich + paired natural avoiding rows)
                        1c GPU + API    (loads Gemma+SAE; OpenRouter synth LLM)

Stage 2: concept-map   concept_features{,_neg}.safetensors
                        canonical       (CPU-only, trivial writer)
                        delta_ranked    (loads Gemma+SAE over synth corpus)

Stage 3: training      baselines/{diffmean,linear_probe,lora,prompting}/...
                        3a DiffMean     (loads both target models, per-mode)
                        3b LinearProbe  (loads both target models, per-mode)
                        3c Prompting    (CPU-only; concept list only)
                        3d LoRA         (loads decoder-only only; per-mode)
                        3e contexts     (exact Stage 4 encdec prompt enumeration)

Stage 4: generate      generations/batch_NNNNNN.parquet
                        (loads only enabled target families; coefficient sweep)

Stage 5: judge         judge_scores/batch_NNNNNN.parquet
                        (OpenRouter judge LLM, 0-2 relevance + fluency)

Stage 6: aggregate     report.yaml + leaderboard.csv + aggregation_report.yaml
                        (CPU-only; harmonic mean, best-of-grid, invalid-rate gating)
```

### Stage 1 tokenizer ownership

Stage 1b and Stage 1c use the nanochat RustBPE tokenizer for exact
owned-BOS output boundaries, while Gemma remains the extraction tokenizer for
SAE/QC measurements. The Stage 1 boundary helpers share one process-local
nanochat tokenizer instance through
`steering.eval.axbench_base.tokenizer_cache`; this avoids reconstructing
`tokenizer.pkl` for every emitted row. The cache does not alter tokenizer
checksums, token IDs, row schemas, checkpoint identity, or resume behavior.
`RuntimeState.nanochat_tokenizer()` remains a separate per-run accessor for
stages that explicitly request it, and the two tokenizer families must never
be mixed.

`RuntimeState` lazy-loads the heavy resources on first access and
CPU-offloads them between uses (`runtime_state.py` heavy-model
context managers). Peak GPU memory for Gemma-2b + d12 targets is
an estimated ~13 GB (calculated from component sizes, not measured;
see [advanced doc](../evaluation/axbench-base-advanced.md#budget-breakdown))
because the Gemma+SAE tier and the nanochat target tier are never
live simultaneously — each stage enters exactly the contexts it
needs and the refcounted `with` blocks drop every model back to
CPU on exit.

**Scripts:** `scripts/eval/axbench_base_eval.py` (Hydra entry point),
`steering/eval/axbench_base/` (stage implementations).
**Configs:** `configs/eval/axbench_base/axbench_base.yaml` with
`tiers/`, `stages/`, and `methods/` groups.

**External dependencies.** Stage 0c calls OpenRouter when examples QC is
enabled or `verification_api_model` is non-null; Stages 1b, 1c, and 5 also call
OpenRouter in production. Those live provider paths require
`OPENROUTER_API_KEY`. Stage 0a always
fetches Neuronpedia descriptions and requires `NEURONPEDIA_API_KEY`; Stage 0c
also requires it when examples QC or API verification generation is enabled,
because both paths fetch activating examples.

The real-Gemma integration test is the OpenRouter-offline exception: production
clients use the cached thread-local `requests.Session.post` path in
`llm_judge_client.py` and `judge.py`, so the test patches that exact boundary
with the fail-closed `tests/axbench_base/fixtures/offline_openrouter.py`
transport. It also tripwires module-level `requests.post` and OpenRouter
`Session.request`/`get`/`send` before adapter, DNS, socket, or provider access;
non-OpenRouter HTTP traffic remains available for Neuronpedia/HuggingFace
prerequisites.
Its local Gemma/SAE oracle returns only classified, nonce-framed, parser-valid
Stage 0c responses and strict-QC Stage 1 responses. The test injects the dummy
`sk-or-test-offline` value only for runtime credential validation; no OpenRouter
credential is required. This is OpenRouter-offline, not universally
network-free: Neuronpedia, HuggingFace, and nanochat caches remain real
prerequisites. Live provider coverage remains behind `@pytest.mark.api` and
explicit credentials.

Stages 0b, 0c, 1b/1c QC, and 2 (`delta_ranked` mode) run base Gemma + SAE
using `cfg.extraction_tokenizer_id`; Stages 3 and 4 run target-model
forward passes. Stages 0a, 3c, and 6 are CPU-only. Stage 3e briefly
loads the encoder-decoder checkpoint to bind the exact context set to its real
sequence length.
Stage 0c and Stage 1 synth parquets are tokenizer-scoped with
`extraction_tokenizer_id` metadata; Stage 2 validates that metadata
before delta-ranked mapping and writes the tokenizer ID into
delta-ranked concept-feature safetensors. Stage 3 baseline training
also rejects Stage 1 synth parquets missing or mismatching that
metadata, and Stage 4 rejects stale delta-ranked concept-feature files
before dispatching `ours` methods.
The completed Stage 0c parquet additionally records
`neuronpedia_model_id`, the effective `feature_extraction_model_id`, and
`neuronpedia_source`, plus semantic SHA-256 identities for the ordered Stage 0a
candidate pool, `concepts.json`, `rejected_concepts.json`, and the full agreed
Stage 0c fingerprint. The lightweight completed-run resume check validates the
extractor identity, reloads the three JSON inputs/sidecars, and compares their
semantic identities before it skips model loading. JSON whitespace and object
key order do not invalidate an artifact; row/list order and field values do.
Missing or mismatched identity metadata aborts before model loading. Use a new
`run_tag`, or delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c; a
Stage 0c-only automatic rerun could otherwise consume candidate concepts from
the old identity.
Stage 1c validates the completed parquet's extractor-identity fields when
invoked directly, so skipping Stage 0c cannot consume prefixes verified by
another extractor.

## Prerequisites

Before any stage runs you need:

- A feature directory from `docs/pipelines/feature-extraction.md`
  following the `s{seg}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}`
  convention. AxBench-Base parses the path to recover layer, segment size,
  SAE width, and model.
- A `feature_statistics.safetensors` produced by
  `steering.compute_statistics` inside that directory.
- A target decoder-only nanochat base model (see `docs/pipelines/base-training.md`).
- A target encoder-decoder nanochat model (see
  `docs/pipelines/encoder-decoder-training.md`). After loading, attach
  `model._loaded_meta_data = meta_data` — see `docs/evaluation/axbench-base.md#prerequisites`.
- Base Gemma + SAE for activation analysis (loaded through the feature
  extraction helpers).
- `NEURONPEDIA_API_KEY` exported for Stage 0a and Stage 0c whenever examples QC
  or `verification_api_model` is enabled.
- `OPENROUTER_API_KEY` exported for those same Stage 0c API paths and Stages
  1b/1c/5.
- A single CUDA device (24 GB is enough for `gemma-2-2b` + `gemmascope-res-16k`
  at `segment_size=128`).

## Quick Start

The CLI entry point is `scripts.eval.axbench_base_eval`. Three tiers are
predefined under `configs/eval/axbench_base/tiers/`:
`dev` (5 concepts), `standard` (50 concepts), `full` (500 concepts).
Each tier file overrides `max_concepts` only; every other field inherits
from `AxBenchBaseConfig` (`steering/eval/axbench_base/config.py:62-138`).

Every stage is dispatched from the Hydra CLI. A shared
`RuntimeState` owns the base Gemma + SAE, target models, engines,
caches, and the OpenRouter judge client for the lifetime of a single
invocation. A cold encoder-decoder run has one deliberate operator boundary:
run through Stage 3e, extract and convert the exact enumerated contexts, then
run Stages 4-6 with that cache. `stage=all` is convenient for resuming after
the cache exists; on the first pass it fails before generation rather than
silently using an incomplete or stale cache.

```bash
# Dev tier: 5 concepts, stage=all end-to-end smoke test.
uv run python -m scripts.eval.axbench_base_eval \
    tiers=dev \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec       encdec.model_tag=d12_baseline encdec.step=54600
```

```bash
# Standard tier: 50 concepts, overnight run.
uv run python -m scripts.eval.axbench_base_eval \
    tiers=standard \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec       encdec.model_tag=d12_baseline encdec.step=54600
```

```bash
# Full tier: 500 concepts, multi-day run.
uv run python -m scripts.eval.axbench_base_eval \
    tiers=full \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec       encdec.model_tag=d12_baseline encdec.step=54600
```

See [Running the full benchmark](#running-the-full-benchmark) below for
prerequisites, expected runtimes, resume semantics, and how to run only
a subset of stages.

Stage 0c continuation generation uses OpenRouter by default. To select the
shipped model explicitly, pass its model ID:

```bash
uv run python -m scripts.eval.axbench_base_eval \
    stage=0c \
    verification_api_model=inclusionai/ling-2.6-flash \
    verification_api_prefix_wave_size=100 \
    synth.max_concurrent=100 \
    synth.requests_per_minute=6000 \
    <the same feature/model/run overrides used for Stage 0a>
```

The default is `inclusionai/ling-2.6-flash`. Synth traffic defaults to
`max_concurrent=100` and `requests_per_minute=6000` per process. API mode
speculatively proposes up to `verification_api_prefix_wave_size=100` candidate
prefixes per wave, capped by candidate-attempt/pool and zero-hit-bail limits but
not by the remaining accepted-prefix target. With five seeds, a full wave has
500 logical calls and can keep all 100 workers occupied. Every proposal is
locally scored; only the first target passes in candidate order are retained.
The generator independently infers an
operational definition from raw Neuronpedia activation evidence; it never
receives the published Stage 0a description. API responses are strictly framed
and parsed, but accepted continuations still have to pass the local SAE
percentile/consistency gate. Set `verification_api_model=null` for the unchanged
local-generation path. Use a fresh `run_tag` when switching modes because API
artifacts use Stage 0c schema 9 while the null path remains schema 8.

For a clustered target, definition inference runs independently for each
mask-allowed member. The generator cycles the resulting evidence-derived member
descriptions over the global candidate-by-seed proposal stream; it does not ask
the API to collapse them into one representative description. The assignment is
stable across API wave sizes and resume, while cluster-wide local max-pooling
remains the sole acceptance oracle. One allowed member takes the exact ordinary
request and cache-key path.

## Running the full benchmark

**Prerequisites** (same as [Prerequisites](#prerequisites) above).

- A feature directory and `feature_statistics.safetensors`.
- A nanochat decoder-only base checkpoint (`source=base`).
- A nanochat encoder-decoder checkpoint (`source=encdec`).
- `NEURONPEDIA_API_KEY` exported (Stage 0a always requires it; Stage 0c
  requires it when `enable_concept_examples_qc=true` or
  `verification_api_model` is non-null).
- `OPENROUTER_API_KEY` exported (Stages 1b/1c/5 require it; Stage 0c requires
  it for examples QC or non-null `verification_api_model`, and skips it only
  when both are disabled).
- ~13 GB free GPU memory on a single CUDA device (estimate from
  component sizes; see
  [Budget breakdown](../evaluation/axbench-base-advanced.md#budget-breakdown)).
  Use `device=cpu` for small-box runs at ~20x runtime cost.

**Full commands per tier.** Use a fresh, explicit `run_tag`. Run Stages 0a
through 3e first (individually or by running `stage=all` and allowing the
intentional Stage 4 cache preflight to stop the cold run), then follow
[Exact Stage 4 context cache](#exact-stage-4-context-cache) before Stages 4-6.

```bash
# Dev tier: 5 concepts. Run all stages.
uv run python -m scripts.eval.axbench_base_eval \
    tiers=dev \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline encdec.step=54600 \
    device=cuda output_root=out/eval/axbench_base
```

```bash
# Standard tier: 50 concepts. Same flags, different tier.
uv run python -m scripts.eval.axbench_base_eval tiers=standard \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline encdec.step=54600
```

```bash
# Full tier: 500 concepts.
uv run python -m scripts.eval.axbench_base_eval tiers=full \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline encdec.step=54600
```

**Expected wall-clock** is dominated by Stages 1b/1c (OpenRouter at
~60 rpm) and Stage 4 (per-concept coefficient sweep on every enabled target
family; both targets by default). See `docs/evaluation/axbench-base.md#cost-and-runtime` for per-stage
estimates; the headline numbers are: dev tier completes in tens of
minutes, standard tier is an overnight run, full tier is multi-day.
TODO: update with H100 wall-clock after first real run.

**Resume after a crash.** Re-run the same command.
`check_stage_dependencies` (`dag.py`, `check_stage_dependencies`
at line 362) walks the dep DAG for
every stage in the expansion and no-ops any stage whose outputs are
already on disk. The SQLite caches (`synth_response_cache.sqlite` and
`judge_cache.sqlite`) replay any OpenRouter calls that already
returned a good response, so a rerun costs close to zero external
tokens. Stage 4's parquet directory dataset is append-only and
crash-safe per batch.

**Running a single stage.** Pass `stage=<id>` where `<id>` is one of
the `Stage` StrEnum values (`steering/eval/axbench_base/types.py:66-80`):
`0a`, `0b`, `0c`, `1b`, `1c`, `2`, `3a`, `3b`, `3c`, `3d`, `3e`, `4`,
`5`, `6`, or `all`.

```bash
# Re-run just Stage 4 (generation) against an existing run_tag.
uv run python -m scripts.eval.axbench_base_eval tiers=dev \
    stage=4 run_tag=20260414-191100-abcdef01 \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline encdec.step=54600
```

If the requested stage's dep check fails, the error names every missing
path so the operator can re-run the prerequisites first.

### Clustered stage contract

The landed clustered path is deliberately stage-limited. Set paired path-backed
`conditioning.feature_mask` and `conditioning.feature_clusters`, force
`concept_feature_mode=canonical`, and use an explicit method list. Source stages
`0a`, `0b`, `0c`, `1b`, `1c`, `2`, `3a`, and `3b` accept only the six
positive/negative decoder-only DiffMean, linear-probe, and control methods.
Stages `3e` and `4` additionally accept the encoder-decoder controls and seven
ordinary `ours_*` context-editing methods. Every other stage, `stage=all`,
`delta_ranked`, and a missing/duplicate/out-of-allowlist method list fail config
validation before stage work.

Main has no `axbench_base_cluster_source` or
`axbench_base_cluster_stage4` config. The root YAML also does not declare
`conditioning.feature_clusters`, so a CLI run must add it with `+` and preserve
the complete identity-bearing override set across each stage:

```bash
uv run python -m scripts.eval.axbench_base_eval \
    stage=0a run_tag=<fresh-run-tag> \
    enabled_methods='[diffmean_decoder_only,neg_diffmean_decoder_only,linear_probe_decoder_only,neg_linear_probe_decoder_only,control_decoder_only,neg_control_decoder_only]' \
    conditioning.feature_mask='{mode:whitelist,path:/abs/path/accepted_feature_ids.txt}' \
    +conditioning.feature_clusters.path=/abs/path/feature_clusters.yaml \
    concept_feature_mode=canonical \
    <the ordinary feature/model/statistics overrides>
```

This cluster contract is more than an ID filter. Startup reconciles the exact
checkpoint step and persistent effective vocabulary with the mask, canonical
clusters, statistics file, feature-directory/SAE identity, selected-feature
manifest, and concept-source domain. Stage 0a projects and deduplicates the
candidate domain in representative space. Stage 0c keeps the member-description
cycle while using the cluster-wide local max as acceptance. Stages 3a/3b stamp
the feature-contract fingerprint and exact source-corpus digest into direction
artifacts. Reusing a run tag after any of those inputs changes is rejected.

Direction identity is enforced independently of clustering. Stages 3a/3b
stamp the family, target, mode, concept, and target residual training layer;
Stage 4 preflights every selected DiffMean/probe artifact before model forward
or output creation and repeats validation when loading it. The configured
`layer_hooks.diffmean_and_probe_layer` indexes the target nanochat or
encoder-decoder residual stack. It is unrelated to the Gemma SAE site encoded
by `feature_dir` (for example, Gemma layer 22 sources still use target decoder
layer 6 in the shipped Gemma-3 preset).

### Exact Stage 4 context cache

Stage 3e writes the exact encoder-decoder prompt corpus for the current seed
fingerprint, enabled methods, prompting descriptions, eval caps, tokenizer, and
checkpoint sequence length:

```bash
uv run python -m scripts.eval.axbench_base_eval \
    stage=3e run_tag=<fresh-run-tag> \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=... decoder_only.step=... \
    encdec.source=encdec encdec.model_tag=... encdec.step=...
```

The run root now contains `stage4_contexts.jsonl`,
`stage4_contexts_extraction/shard_00000.parquet`, and
`stage4_contexts_manifest.json`. Extract that parquet with the ordinary feature
extractor, mirroring the model, SAE, layer, activation, pooling, segment size,
context-independence, L0, backend/dtype, and context-window settings encoded by
the benchmark's `feature_dir`:

```bash
uv run python -m steering.extract_features \
    data_dir=<run-root>/stage4_contexts_extraction \
    output_dir=<run-root>/stage4_context_features \
    num_shards=1 start_shard=0 \
    model=<model> layer=<layer> activation_type=<activation> \
    pooling=<pooling> segment_size=<segment-size> context_window=<limit> \
    sae.release=<release> sae.num_features=<width> sae.l0=<l0>
```

Convert and validate the extracted shard. `required_contexts_manifest` makes
the converter verify JSONL/parquet ordering and hashes, stamps the contexts
fingerprint into cache metadata, derives the extraction tokenizer's decoded BOS
length, and rejects a conflicting `source_bos_char_len`:

```bash
uv run python -m scripts.eval.build_eval_context_cache \
    stage=convert \
    contexts_file=<run-root>/stage4_contexts.jsonl \
    context_parquet_dir=<run-root>/stage4_contexts_extraction \
    required_contexts_manifest=<run-root>/stage4_contexts_manifest.json \
    extraction_tokenizer_id=<matching-extraction-tokenizer> \
    feature_dir=<run-root>/stage4_context_features \
    output_dir=<context-cache-root> \
    stats_path=<feature-statistics.safetensors> \
    model_max_seq_len=<encdec-sequence-len> \
    n_sae_features=<width> num_buckets=<checkpoint-num-buckets> \
    identity.model_id=<model-id> identity.sae_id=<sae-id> \
    identity.layer=<layer> identity.pooling=<pooling> \
    identity.feature_selection_hash=<checkpoint-selection-hash>

uv run python -m scripts.eval.build_eval_context_cache \
    stage=validate \
    contexts_file=<run-root>/stage4_contexts.jsonl \
    context_parquet_dir=<run-root>/stage4_contexts_extraction \
    required_contexts_manifest=<run-root>/stage4_contexts_manifest.json \
    extraction_tokenizer_id=<matching-extraction-tokenizer> \
    feature_dir=<run-root>/stage4_context_features \
    output_dir=<context-cache-root> \
    stats_path=<feature-statistics.safetensors> \
    model_max_seq_len=<encdec-sequence-len> \
    n_sae_features=<width> num_buckets=<checkpoint-num-buckets> \
    identity.model_id=<model-id> identity.sae_id=<sae-id> \
    identity.layer=<layer> identity.pooling=<pooling> \
    identity.feature_selection_hash=<checkpoint-selection-hash>
```

For a clustered Stage 3e cache, use the shipped
`build_eval_context_cache_clustered_axbench` preset. Supply the same path-backed
mask/clusters, real training-preselection `seed`, exact
`stats_source_sha256`, positive/negative sampling policies, and
`feature_delta=null`. The user-facing positive selection must **omit**
`delta_weights` entirely; the canonical identity resolver injects a null value
only into its internal sampling hash payload to reproduce the producer's
delta-free base-validation identity. A configured `delta_weights` key, a
threshold selection, or a non-null `feature_delta` is rejected before cache
conversion.

```bash
uv run python -m scripts.eval.build_eval_context_cache \
    --config-name build_eval_context_cache_clustered_axbench \
    stage=convert \
    contexts_file=<run-root>/stage4_contexts.jsonl \
    context_parquet_dir=<run-root>/stage4_contexts_extraction \
    required_contexts_manifest=<run-root>/stage4_contexts_manifest.json \
    extraction_tokenizer_id=<matching-extraction-tokenizer> \
    feature_dir=<canonical-stage4-feature-leaf> \
    output_dir=<context-cache-root> \
    stats_path=<feature-statistics.safetensors> \
    stats_source_sha256=<exact-statistics-sha256> \
    feature_mask.path=/abs/path/accepted_feature_ids.txt \
    feature_clusters.path=/abs/path/feature_clusters.yaml \
    seed=<checkpoint-training-preselection-seed> \
    feature_selection='<sampling-policy-without-delta_weights>' \
    negative_selection='<checkpoint-negative-sampling-policy>' \
    n_sae_features=<original-width> num_buckets=<checkpoint-num-buckets> \
    identity.model_id=<model-id> identity.sae_id=<sae-id> \
    identity.layer=<layer> identity.pooling=<pooling> \
    model_max_seq_len=<encdec-sequence-len>
```

Pass that root as `conditioning.context_cache_dir` when running Stage 4. Stage
4 reconstructs the Stage 3e contract, validates cache identity, selection hash,
model length, contexts fingerprint, and every required token key before the
first generation.

### Certified clustered checkpoint preflight

When `conditioning.feature_clusters` is set, startup accepts only a producer-
certified clustered `sel/<hash>/` family. It validates both manifests,
checkpoint/mask/cluster/statistics identity, exact packed/selection stem
coverage, every selection file's compressed SHA-256, and the packed files'
link/target signatures. A signature-valid
`.packed-row-fingerprint-cache.json` proves the packed content aggregate
without reopening packed bodies, so the warm path skips packed hashing and
packed/selection tensor decompression. It still reads the complete compressed
selection inventory on every invocation; for large artifacts that residual
selection I/O is expected.

Cache `missing` and structurally valid `stale` states are rebuilt and atomically
published only after successful validation. Corrupt, truncated, unreadable,
wrong-schema, or internally inconsistent cache data fails before packed body
I/O. To repair it, rerun the same clustered stage with:

```bash
uv run python -m scripts.eval.axbench_base_eval ... \
    force_cluster_shard_revalidation=true
```

Forced mode executes every canonical packed and selection tensor validator,
then replaces and rereads the cache. Under DDP, rank 0 alone performs that full
packed-body pass and publishes; peers independently verify the resulting cache,
selection bytes, manifests, signatures, and contract. The false default is
omitted from run-tag serialization; true is recorded. The flag is rejected for
unclustered runs.

For clustered decoder-only methods, Stage 4 also resolves the configured
checkpoint reference to one numeric step, hashes the exact model and metadata
files, and records that pin in the generation-resume identity. The pathname-
based nanochat load is bracketed by a second digest and `lstat`-signature
check, so an inode swap, in-place metadata change, or content change during
the load fails instead of attaching the earlier pin to different weights. A
null or `"last"` step is therefore committed as its resolved integer.
Replacing either file under the same `source`/`model_tag`/`step` reference
makes the existing resume marker invalid instead of mixing generations from
two checkpoints. Unclustered runs retain the legacy read and collective shape:
they do not call the cluster-marker boundary, and decoder-only Stage 4 does not
probe Stage 3e or clustered-generation manifests.

**Accepted residual — context-cache manifest replacement.**
`save_context_cache` supports `overwrite=True` and installs its manifest through
`tmp.rename(path)`—an atomic replacement, not a create-once install.
`generate_all` only reads that manifest; a separate cache-building tool
produces it out-of-band before the eval run. The provider reads each listed
shard once, checksum-verifies that byte string, and decodes the same byte
string before Stage 4 preflight. The generation marker already binds the
manifest digest observed by `_context_cache_manifest_sha256`, whose shard
digests therefore describe the loaded rows, so a later manifest rename cannot
mix new cache rows into the current run. It can still make the on-disk path
differ from the digest the current run correctly records. Re-reading
immediately before marker installation would only narrow that window: a
non-cooperating writer can rename again afterward.
Closing it requires immutable cache publication (or an equivalent cooperative
write exclusion), not another check. The next run rejects the drift through the
marker cross-check or the provider's shard checksums. The
`initial_contexts_fingerprint` versus `validated_contexts_fingerprint`
cross-check separately guards which Stage-3e contexts were enumerated.

**Skipping negative mode.** Pass `run_negative_mode=false`. This drops
Stage 1c entirely, skips the negative concept-features file in Stage 2,
and narrows the Stage 4 method registry to positive-mode methods only.
Dep checks honor the flag so every downstream stage stops looking for
the negative-mode artifacts.

```bash
uv run python -m scripts.eval.axbench_base_eval tiers=dev \
    run_negative_mode=false \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline encdec.step=54600
```

**Outputs** land under
`<effective_output_root>/<sanitized_run_tag>/`. With `output_root=null`, the
effective root is `$NANOCHAT_BASE_DIR/eval/axbench_base`; a non-empty custom
root is honored literally. See [Outputs](#outputs) below for the
per-file schema, or `docs/evaluation/axbench-base.md#outputs-and-layout` for the
column-level schema.

## Mandatory Overrides

Hydra marks these fields as `???` in `configs/eval/axbench_base/axbench_base.yaml`
so you must set them on every run:

| Field | Purpose |
|-------|---------|
| `feature_dir` | Path to the pre-extracted SAE feature shards. |
| `feature_statistics_path` | Path to `feature_statistics.safetensors` in that dir. |
| `decoder_only.{source,model_tag,step}` | Loader spec for the target base model. |
| `encdec.{source,model_tag,step}` | Loader spec for the target encoder-decoder. |

`device` defaults to `cuda`. `output_root` defaults to `null`, meaning the
historical `$NANOCHAT_BASE_DIR/eval/axbench_base`; set a non-empty string to
relocate all artifacts. `run_tag` is auto-generated
from the current UTC second + an 8-char SHA-256 of the config when left
at `null` (`scripts/eval/axbench_base_eval.py`, `_generate_run_tag` at
lines 183-206).

## Config Groups

`configs/eval/axbench_base/` composes three subgroups:

| Group | Files | Effect |
|-------|-------|--------|
| `tiers/` | `dev.yaml`, `standard.yaml`, `full.yaml` | Sets `max_concepts` (5 / 50 / 500). |
| `stages/` | `all.yaml` | Sets `stage: all`. Add files to this dir to pin individual stages. |
| `methods/` | `all.yaml` | Sets `enabled_methods: null` (all 33 methods in the registry). |

The full 60+ field config reference lives in
`docs/evaluation/axbench-base.md#configuration-reference`. The dataclass
defaults and unary refined domains live in
`steering/eval/axbench_base/config.py`; `beartype` enforces those domains while
Hydra input is converted, and `validate_config` owns the remaining relational
and mode-dependent invariants.

## Stage 0b correctness and resume contract

Stage 0b raw-encodes source text with nanochat and no BOS. For `N` raw tokens,
continuation length `M`, and prefix cap `C`, eligibility is `N >= 2*M`, the
continuation is `[N-M:N]`, and the prefix is
`[max(0, N-M-C):N-M]`. These exact source token IDs are stored; decoded text is
representational only. The schema-v2 seed pool never stores character offsets
or extraction-segment ranges and never reconstructs tokens through text.

Prefix features come from an independent production extraction of that exact
decoded prefix, not from the source document's existing feature shard. The
production extractor's parsed layer/hook/pooling/segment/context/BOS-mask/L0
settings are reused and magnitudes are max-aggregated by feature ID.

The three performance controls are
`seed_extraction_candidate_shard_size=1024`,
`seed_extraction_gpu_batch_size=16`, and
`seed_extraction_cpu_batch_size=64`. All are positive and enter artifact
identity. Every DDP rank independently verifies the canonical source/candidate
digest, actual model/SAE tensor hashes, runtime identity, and fixed canary.
Rank-independent feature shards run in collective waves and carry validation
sidecars, so an interrupted run can resume with a different world size.

`seed_pool_manifest.json` is written last and is the sole commit marker for
both final files. Runtime loading rejects legacy, one-file-only, bad-hash,
wrong-split, wrong-tokenizer/config/model, or malformed seed artifacts. Because
legacy ownership cannot be inferred, the only recovery is a fresh run tag.

During that same validated train load, `seed_contract.py` hashes the row count
and the complete storage of the fixed 16-tensor schema before any tensor is
converted to Python lists. The resulting `stage1b_seed_rows_v2` digest is the
ordered Stage 1b row identity cached beside the reconstructed rows. This binds
CSR/text boundaries and persisted scalar/float bits without depending on
safetensors metadata or file layout. Raw file SHA-256 is unsuitable because
equal tensors can have different serialization metadata; the Stage 0b
`seed_fingerprint` is unsuitable because it identifies producer inputs and the
complete pool rather than the observed ordered train split.

At model boundaries, ordinary prompts are `[owned BOS] + seed content`; a
prompting method uses `[owned BOS] + raw prompting-prefix tokens + seed
content`. Genuine payload tokens equal to the BOS ID are preserved. Stage 1
pair tokenization similarly adds one owned BOS after encoding the combined
prefix+output once, and all Stage 3 trainers require the exact stored output
boundary.

Stage 1b does not consume overlong Stage 0b prefix text verbatim. It resolves
one immutable, original-index-aligned view using the shared
`min_prefix_chars` / `max_prefix_chars` decoded-character bounds, always
retaining the most recent characters. That exact bounded string drives seed
fallback scoring, prompts, parsing, QC, cache identity, rows, and output
boundaries; the committed Stage 0b files and Stage 4 eval seeds are untouched.
All LLM-generated rich and avoiding continuations use deterministic
attempt-specific nonce markers and must repeat the exact final 96 prefix
characters (or the complete shorter prefix) at the authenticated response
boundary. Only the fixed Stage 0c separator variants and a one-character
left-truncated echo are normalized. The parser preserves leading continuation
characters, validates the complete provider response for source copies,
residual suffix overlap, phrase loops, controls, protocol-marker pollution, and
special-token IDs, truncates to `output_length` extraction-tokenizer tokens,
then revalidates the decoded stored window before SAE QC. Failures retry under
a fresh attempt key. Each seed gets
`positive_max_attempts_per_seed` total semantic attempts on the rich side and,
when natural avoiding QC fails, independently on the LLM-avoiding side.
Exhaustion discards the atomic pair and lazily draws the next headroom-safe
candidate; `n_positives_per_concept` is the complete-pair target rather than a
fixed roster. The zero-hit guard limits distinct candidates only until the
first complete pair. Provider transport exhaustion aborts without charging a
semantic attempt or drawing a replacement.
Trusted natural-corpus avoiding text remains strict-zero-QC-only. A final
invariant pass rejects any out-of-bounds or stale-boundary Stage 1b row before
`synth_training_positive.parquet` is published.

At target 72, the positive parquet carries
`positive_target_refill_post_provider_qc_v5`. Non-72 targets carry
`positive_target_compatibility_post_provider_qc_v6`, including compatibility
target 72, the full-chunk post-compatibility batching identifier, and the
effective stable-frontier post-provider QC batch identity; the row schema is
unchanged.
Completed pre-batching v3/v4 selection corpora (and older contracts) and their
downstream products require a fresh auto-generated run tag. An interrupted run
with an explicit reused tag resumes
from `stage1b_partial.parquet` plus the authoritative
`stage1b_partial.json`. Schema-v2 checkpoints admit the same or a higher target
and bind every synthesis-semantic input except the requested target; the JSON
records `highest_requested_target`. Decreases fail before prefix-cache or
provider construction. Post-synthesis fallback and population-drop settings
remain bound,
because they decide whether and how the checkpointed history may publish.
The effective post-provider QC batch cap remains checkpoint identity because
BF16 ragged-batch scoring is not proven shape-invariant. Changing it requires a
fresh partial; an exact pre-batching checkpoint has one explicit in-memory
migration to the configured current shape.
Persisted candidates restore without scorer calls, while outstanding chains
restart from rich attempt zero and reuse any responses already committed to
the durable SQLite cache.

Candidate scoring always begins by filling toward 72 admissible candidates
with the exact historical target-72 scorer calls, including headroom rejection
and dirty fallback. Targets above 72 continue in full clean scorer-batch chunks;
only clean-pool exhaustion produces a partial chunk, then dirty candidates are
scored once. A physically exhausted pool below 72 may still satisfy a smaller
configured target. Surplus candidates stay buffered, so an interrupted `N -> M`
growth does not rescore them and matches a fresh target `M` artifact.

Growth is supported only while JSON remains authoritative and the operator
reuses the same explicit `run_tag`. Restored state scores only the unscored
full chunks needed to prove a higher target feasible before the new target is
committed or any provider request is built. An infeasible growth leaves the
previous target authoritative. A completed positive parquet is validated
before scoring and cannot be extended in place, even if its partial JSON still
exists after interrupted cleanup; that crash-window partial may resume only
the same target. Use a fresh tag for any target change after final publication.
Schema-v1 Phase 1 checkpoints migrate only from stored target 72 after exact
remaining-fingerprint validation. Other v1 targets used target-shaped BF16
prefix-scoring batches and must be finished under the Phase 1 revision or
explicitly restarted. Migration does not rewrite disk until the next
successful checkpoint. Current checkpoints use fingerprint algorithm v4 and
the bulk row digest. Supported v1/v2/v3 checkpoints invoke the exact historical
row serializer once, validate the legacy fingerprint, and normalize to v4 in
memory; malformed or unsupported versions fail before that scan. The existing
initial checkpoint commit writes v4 before provider work. A crash before that
commit leaves the legacy marker untouched and may repeat the scan, while a
second successful resume reads v4 without invoking the legacy serializer.

Stage 1b writes the full checkpoint before provider requests, after each 64
terminal chains, on caught failures (including Ctrl-C), and before final
publication. Writes are atomic parquet first and JSON second, so resume trusts
only the JSON-committed pair prefix and ignores a parquet-only orphan. Caught
failures lose no terminal work; an uncatchable hard stop can lose at most 64
terminal outcomes, whose provider responses may still replay from cache. A
response interrupted before cache commit may be requested again. After the
final `"ok"` yield report, cleanup deletes JSON first and parquet second. If
cleanup warns, JSON remains authoritative when present; parquet is an ignored
orphan only when JSON is absent.

## Outputs

Every run artifact writes under
`{effective_output_root}/{sanitized_run_tag}/`. Path sanitization strips
path separators and unsafe characters; traversal attempts are bounced to
a `traversal_<sha256prefix>` fallback. The cross-run Neuronpedia cache is
identity-addressed outside the run directory.

DDP startup broadcasts rank 0's autogenerated tag and requires exact
configured/effective output-root agreement across ranks. The shared
Neuronpedia writer publishes fetched rows transactionally only after file and
directory fsync, and rejects malformed cache schemas or conflicting
same-feature values.

```
{effective_output_root}/
|-- _neuronpedia_cache/v1/{identity_sha256}.yaml
`-- {run_tag}/
    |-- candidate_concepts.json                # Stage 0a pre-QC filtered + shuffled pool
    |-- concepts.json                          # Stage 0c final verified concepts
    |-- rejected_concepts.json                 # Stage 0c LLM QC + verification failures
    |-- concept_examples_qc_report.json        # Stage 0c (when examples QC enabled)
    |-- synth_response_cache.sqlite            # Stages 0c/1b/1c LLM cache
    |-- seeds_train.safetensors                # Stage 0b
    |-- seeds_eval.safetensors                 # Stage 0b
    |-- seed_pool_manifest.json                # Stage 0b schema-v2 commit marker
    |-- _seed_extraction_v2/                   # resumable rank-independent Stage 0b work
    |-- negative_mode_prefixes.parquet         # Stage 0c
    |-- synth_training_positive.parquet        # Stage 1b, tokenizer + QC/selection/continuation contracts
    |-- stage1b_partial.parquet                 # Stage 1b temporary complete-pair rows
    |-- stage1b_partial.json                    # Stage 1b temporary commit marker
    |-- stage1b_concept_yield.json              # Stage 1b yield/accounting commit marker
    |-- concepts_stage1b.json                   # Stage 1b filtered concept authority
    |-- synth_training_negative.parquet        # Stage 1c, tokenizer + continuation contract
    |-- stage1c_concept_yield.json              # Stage 1c yield/accounting commit marker
    |-- concepts_stage1c.json                   # Stage 1c final concept authority
    |-- concept_features.safetensors           # Stage 2 (positive)
    |-- concept_features_neg.safetensors       # Stage 2 (negative, if run)
    |-- baselines/                              # sparse original IDs from final Stage 1 list
    |   |-- diffmean/{target}/{mode}/concept_NNNN.safetensors
    |   |-- linear_probe/{target}/{mode}/concept_NNNN.safetensors
    |   |-- lora/decoder_only/{mode}/concept_NNNN.pt
    |   `-- prompting/{mode}_descriptions.json
    |-- stage4_contexts.jsonl                  # Stage 3e exact token contexts
    |-- stage4_contexts_extraction/shard_00000.parquet
    |-- stage4_contexts_manifest.json          # Stage 3e commit/identity marker
    |-- generations/                           # Stage 4 directory-dataset
    |   `-- batch_NNNNNN.parquet
    |-- judge_scores/                          # Stage 5 directory-dataset
    |   `-- batch_NNNNNN.parquet
    |-- judge_cache.sqlite                     # Stage 5 SQLite cache
    |-- report.yaml                            # Stage 6
    |-- leaderboard.csv                        # Stage 6
    `-- aggregation_report.yaml                # Stage 6
```

`generations/` and `judge_scores/` are directory-dataset parquet batches
written by `ParquetAppendWriter` (`parquet_writer.py`) for crash-safe
incremental append. See `docs/evaluation/axbench-base.md#outputs-and-layout` for the
full column schemas of every file.

## Method Registry

Stage 4 dispatches through a flat 33-entry registry built at module load
in `steering/eval/axbench_base/method_registry.py` as `ALL_METHODS`
(line 371), built once at import by `_build_all_methods`
(line 354). The breakdown:

| Mode | Baseline | Count | Target models |
|------|----------|-------|---------------|
| positive | `diffmean` | 2 | decoder_only, encdec |
| positive | `linear_probe` | 2 | decoder_only, encdec |
| positive | `lora` | 1 | decoder_only only |
| positive | `prompting` | 2 | decoder_only, encdec |
| positive | `control` | 2 | decoder_only, encdec |
| positive | `ours` | 7 | encdec only |
| negative | `diffmean` | 2 | decoder_only, encdec |
| negative | `linear_probe` | 2 | decoder_only, encdec |
| negative | `lora` | 1 | decoder_only only |
| negative | `prompting` | 2 | decoder_only, encdec |
| negative | `control` | 2 | decoder_only, encdec |
| negative | `ours` | 8 | encdec only |

Totals: 16 positive + 17 negative = **33** (including the 4 no-steering
`control` baselines). Fifteen are
`ours_*_encdec`, `neg_ours_*_encdec`, or Feature-DiffMean methods. Seven
directly exercise the conditional encoder-decoder's context-partition modifiers
(`ours_extract_and_add`,
`ours_replace_random`, `ours_replace_all`, `neg_ours_remove`,
`neg_ours_add_to_negatives`,
`neg_ours_replace_random_negatives`, `neg_ours_replace_all_negatives`). The
remaining eight consume precomputed fair/roofline conditioning bags: four
Feature-GCG methods (`ours_gcg_encdec`, `ours_gcg_roofline_encdec`,
`neg_ours_gcg_encdec`, `neg_ours_gcg_roofline_encdec`) and the analogous four
Feature-DiffMean methods. AxBench-Base does not derive those bags: a direct
Stage 4 run must disable them, and invoking one without a precomputed bag
fails loudly: `_modify_precomputed_conditioning_bag` raises `FeatureGcgError`
(`steering/eval/axbench_base/encoder_input.py:455`). Plain AxBench-BPB derives
and scores the fair Feature-DiffMean bags itself; BPB-Adv additionally derives
and scores the roofline Feature-DiffMean bags and all four Feature-GCG bags.
Three methods (`neg_ours_{add_to_negatives,replace_random_negatives,replace_all_negatives}_encdec`)
set `requires_negative_sampling=True` and are dropped at filter time if
the target encdec was not trained with negative-feature sampling.

Narrow the matrix with `enabled_methods=[...]` or
`disabled_methods=[...]` — `filter_methods`
(`method_registry.py`, `filter_methods` at line 396) applies enable,
then disable, then the
negative-sampling constraint.

## Relationship to AxBench

AxBench (Wu et al. 2025, [arXiv:2501.17148](https://arxiv.org/abs/2501.17148))
evaluates concept steering on **instruction-tuned** models using
dialog-style prompts, where each concept is described to the model via
a chat scaffold. AxBench-Base adapts the same scoring rubric (concept
relevance + fluency on a 0-2 scale) to a **base** target: a plain
next-token continuation model trained on the corpus used for feature
extraction, with no chat template and no instruction-following.

Numbers from AxBench-Base are **not directly comparable** to AxBench
proper. Differences include:

- Target model is a base LM, not an instruction-tuned chat model.
- Prompts are raw-text prefixes drawn from the training distribution,
  not instructions.
- The concept set is filtered for GemmaScope SAE features with
  Neuronpedia descriptions that survive an optional LLM QC judge
  (`concept_examples_qc.py`).
- Negative-mode evaluation seeds are pre-verified prefixes where the
  concept's canonical feature fires above
  `100 * verification_percentile_threshold` raw percentile points
  on at least `min_consistency_passes` of `n_verification_seeds` sampled
  continuations — AxBench has no equivalent prep stage.
- Positive-mode Stage 1b treats `accept_threshold` as the same kind of
  `[0,1]` percentile fraction and accepts an LLM continuation only when its
  raw, unquantized output-minus-seed percentile delta is at least
  `100 * accept_threshold` (30 points at the default `0.3`). The positive
  Parquet records this versioned QC contract, and downstream Base/BPB/BPB-Adv
  consumers fail closed on legacy or mismatched metadata.

The benchmark targets a specific research question: whether conditional
encoder-decoder models can steer base LMs toward or away from
interpretable concepts at least as well as the classical baselines.

## Known Limitations

- **GPU memory peak.** Stages 0c through 4 load either Gemma+SAE or the
  nanochat target models, depending on stage. The `RuntimeState`
  CPU-offload pattern keeps
  peak memory bounded to an estimated ~13 GB with `gemma-2-2b` +
  `gemmascope-res-16k` + d12 decoder-only and encdec targets (this
  figure is calculated from component sizes, not measured end-to-end
  — see
  [Budget breakdown](../evaluation/axbench-base-advanced.md#budget-breakdown)),
  but the feature-extraction tier (Gemma+SAE) and the target-model
  tier are never live simultaneously — stages that need both pay a
  ~4 s tier-swap cost at the stage boundary. Long-running Stage 0c/1b/1c
  loops release Python garbage and cached CUDA allocator blocks at their
  candidate/concept/attempt-wave boundaries. Cross-prefix Stage 0c
  scoring defers per-seed cleanup to the candidate boundary, and the
  shared synth-QC helpers release transient Gemma+SAE activations after
  each SAE forward or dirty-prefix cache-fill chunk. These
  memory-lifetime guards reduce monotonic allocator growth without
  changing artifacts.
  Stage 0c's candidate index keeps decoded validation-shard tensors, parquet
  texts, and the extraction tokenizer as fixed run state, but CSR match arrays
  and fully materialized candidate payloads belong only to the active concept.
  Clustered lookup retains only that concept's member union while preserving
  its existing member order, canonical-location union, shared per-shard cap,
  and final sort. Host candidate memory therefore plateaus at the largest
  active-concept high-water plus fixed shard state instead of accumulating
  across visited concepts. API operational definitions are likewise released
  after all waves for the concept; provider concurrency and request behavior
  are unchanged.
  The after-cleanup completion heartbeat reports available
  `current_rss_kb`, `anonymous_rss_kb`, `swap_kb`,
  `anonymous_total_kb`, and `peak_rss_kb` process metrics and mirrors them as
  `stage0c.memory.*` profiler counters. This ownership change adds no config
  field and does not alter schema 8/9, fingerprints, artifacts, or resume
  compatibility.
- **LoRA is decoder-only.** `AxBenchBasePaths.lora_adapter()` hardcodes the
  `baselines/lora/decoder_only/` subpath. No LoRA variant is trained
  for the encoder-decoder target.
- **Multi-GPU support is limited to Stages 0b and 0c.** Stage 0b
  (`PREPARE_SEED_POOLS`) runs rank-independent candidate shards in collective
  waves; Stage 0c (`PREPARE_NEGATIVE_PREFIXES`) supports DDP
  candidate-partitioned parallelism via torchrun. Every other stage is rank-0-only when
  launched under DDP (the CLI gates non-rank-aware stages to rank 0
  and broadcasts the rank-0 outcome before the next stage). A dependency or
  preflight failure on rank 0 is broadcast as one canonical error so peer ranks
  raise immediately instead of waiting at an unmatched barrier. Launch either rank-aware stage
  via:

      torchrun --standalone --nproc_per_node=N \
          -m scripts.eval.axbench_base_eval stage=0b <other args>

      torchrun --standalone --nproc_per_node=N \
          -m scripts.eval.axbench_base_eval stage=0c <other args>

  Stage 0b fixes candidate shard identity and batch grouping independently of
  rank/world size, verifies model/SAE/canary agreement before full extraction,
  and gathers success/error records after every wave. Valid committed shards
  therefore resume across world-size changes; mismatched data/sidecars fail
  rather than being recomputed under the same identity.

  Tuning knob: `cfg.stage0c_multi_gpu_block_size` (default 16) is the
  *configured* per-rank block size. Under `world_size > 1` the effective
  block size is adaptively capped at
  `max(1, min(stage0c_multi_gpu_block_size, ceil(max_concepts * 2 /
  world_size)))` so small-target runs don't overshoot `max_concepts` by
  an order of magnitude. The over-accept budget at termination is
  `effective_block_size * world_size - 1` (the merge truncates back to
  `cfg.max_concepts` deterministically). The configured value is recorded
  in the Stage 0c fingerprint (changing it invalidates partial
  checkpoints); the *effective* value is derived at run time and
  `world_size` is excluded from the fingerprint, so resuming at a
  different `world_size` is allowed. On a topology change, each rank retains
  only previously committed candidates it still owns under the new modulo
  partition; moved candidates are reprocessed by their new owner. Before any
  rank work, all ranks collectively agree on the candidate-pool content hash
  and count, so filesystem-lagged or concurrently replaced pools fail on every
  rank instead of entering different exhaustion/collective sequences.

  Cost note: when `cfg.enable_concept_examples_qc=True`, QC traffic under
  multi-GPU scales with candidate stripes. With non-null
  `verification_api_model`, each rank also derives definitions and issues
  continuation calls for its stripe. `synth.max_concurrent` and
  `synth.requests_per_minute` are per process, not distributed global limits,
  so aggregate admission can scale linearly with `world_size`.
  Production-grade single-rank QC totals are restored only at
  `world_size=1`. The dev-only
  `concept_examples_qc_target_surviving` cap is supported only at
  `world_size=1`; Stage 0c fails before shard/model work when that cap is set
  under multi-GPU rather than silently applying a different cap per rank.

  **Resume vs. a regenerated candidate pool.** The live per-rank
  checkpoint (`stage0c_partial_rank_{rank}.json` /
  `stage0c_partial_rank_{rank}.parquet` — used at every `world_size`,
  including `1`) folds a content hash of the ordered
  `candidate_concepts.json` pool (`candidate_pool_identity()` in
  `_stage0c_internals.py`) into its checkpoint fingerprint alongside the
  usual config fields. If Stage 0a is re-run under the same pinned
  `run_tag` and produces a different pool — e.g. a changed
  `concept_feature_whitelist_path`, a changed
  `min_corpus_count`, or updated Neuronpedia descriptions — the pool hash
  changes and resume raises `AxBenchBaseError` on the fingerprint mismatch
  instead of silently continuing verification against candidates that no
  longer correspond to the on-disk pool. On successful merge, the agreed pool
  hash is copied into `negative_mode_prefixes.parquet` schema metadata before
  the per-rank files are deleted. The completed parquet also binds the two
  final JSON sidecars and the full Stage 0c fingerprint. A completed resume
  reloads the current JSON artifacts, recomputes all semantic hashes, and
  rejects missing or mismatched metadata rather than reusing a mixed final
  artifact set. Delete the stale Stage 0 artifacts and rerun Stage 0a then
  Stage 0c, or use a new `run_tag`, to proceed.
- **Stage 0c local cross-prefix batching is default-on.**
  In explicit-null local mode, `cfg.verification_cross_prefix_batch_size`
  controls how many candidate prefixes one rank batches inside a concept
  verification chunk. The
  default `4` packs `4 * n_verification_seeds` rows into one padded
  generation loop, then consumes results in the same candidate order as
  the one-candidate path. Larger values such as `8` increase KV-cache memory and
  padding waste for long, uneven prefixes. This knob is included in the exact
  local Stage 0c fingerprint, so changing it invalidates local partial
  checkpoints and requires a fresh Stage 0c run or a new `run_tag`. It is
  dormant in API mode.
  Cross-prefix percentile scoring passes `release_memory=False` so the
  runner-level candidate boundary, not each chunk, owns allocator
  cleanup.
- **Stage 0c API continuation generation is the default and is still
  rejection-sampled.** The shipped model ID selects the existing OpenRouter
  synth client, but only for producing proposed
  verification continuations. `stage0c_api_generation.py` fetches raw
  top-activation evidence, validates/deduplicates it, derives its own
  operational definition without passing the published Neuronpedia
  description. A repeated peak token (preserving exact case, or canonicalizing
  variants that differ only by case) is locally detected and bound into
  the definition request so the model must explain direct evidence rather than
  neighboring context. It then runs independent prefix/seed requests through the
  existing concurrency limiter. A bounded exact source bridge taken from the
  original document repairs mid-word/mid-clause segment cuts. The provider
  authenticates only the final 96 characters of prefix plus bridge; the bridge
  itself is prepended from trusted local bytes. Continuation requests pair a
  dedicated literal-format system instruction with a shorter contract-first
  user prompt. The validated operational definition is repeated as quoted
  system data, but a hard literal-applicability gate keeps local coherence
  authoritative: an incompatible prefix should receive realistic unsteered
  prose instead of an invented actor, analysis, goal, analogy, scene, or
  domain. Definition requests remain user-only. At the default 64-token
  target, 90-130 new words provide extraction-tokenizer headroom without
  requiring a model-added paragraph break. Nonce-delimited output-only
  framing rejects malformed, boilerplate, full-prefix/suffix-echoing,
  substantial normalized source-phrase-echoing, definition-copying, meta,
  repeated-loop, cross-writing-system, and unsupported newly fabricated or
  densely mutated Unicode-letter responses. The prompt asks for one ASCII
  space before the end marker. The parser strips terminal ASCII framing
  whitespace even when missing or repeated and normalizes the live-observed
  95-of-96 left-truncated echo, nonce end marker missing its last `]`, and
  duplicate trailing end-marker chatter, while
  still rejecting missing nonces, duplicate begin markers, and substantive
  echo corruption. Later content attempts receive fixed failure-class guidance
  without reflecting untrusted rejected text. Short clean responses are
  extended in place and re-encoded as one string before exact truncation. The local
  Gemma+SAE scorer remains the
  authority; parse/retry exhaustion is a miss, not an acceptance. Evidence and
  LLM bodies commit to `synth_response_cache.sqlite` as they complete.
  `verification_api_model=null` constructs no generator and retains the exact
  local schema-8 fingerprint. `verification_api_prefix_wave_size` controls the
  API candidate wave independently of local GPU batching; candidate-attempt,
  pool, and zero-hit-bail limits can shrink the effective wave. Measure provider
  latency, retries, and acceptance on the intended feature mix before making a
  performance claim.
- **Shared prefix prefill is the default Stage 0c algorithm.**
  `verification_prefill_mode=shared` prefills the `K` candidate prefixes in a
  chunk once and expands the public Transformers cache plus logits, masks, and
  prompt lengths into the existing `M = K * n_verification_seeds` decode rows.
  `verification_prefill_mode=repeated` is the explicit rollback/reference mode
  and retains the old `M`-row prefill. The common decode loop, seed derivation,
  stopping, scoring, and candidate order are unchanged. The two modes are
  independently exact but cross-mode token equality is not promised because
  model numerics can depend on prefill batch shape. Shared mode requires
  Transformers 5.6.2's public `Cache.batch_select_indices` API; unsupported
  caches fail without fallback.
- **Judge invalid-rate gating.** Methods with a per-combined-mode
  invalid-response rate above `cfg.judge.invalid_drop_threshold`
  (default 5%) are excluded from the leaderboard entirely
  (`aggregation.py`, invalid-rate gating around lines 421-428);
  methods above
  `invalid_warning_threshold` (default 1%) are kept with a warning.
- **GemmaScope + Neuronpedia only.** Concept candidates must have a
  Neuronpedia description tied to a GemmaScope SAE feature. Features
  below `min_corpus_count` in `feature_statistics` are discarded.
- **Optional Stage 0a concept-candidate whitelist.**
  `cfg.concept_feature_whitelist_path` points to
  either `accepted_feature_ids.txt` or `manual_feature_filter.yaml` — the
  same artifacts the [autointerp filter](autointerp-filter.md) produces — and
  restricts Stage 0a candidate features to an explicit set **before** the
  Neuronpedia description fetch, so a curated feature list bounds fetch
  and QC cost instead of just the final selection. This is
  **concept-candidate filtering, distinct from the
  downstream `conditioning.feature_mask`** — the whitelist controls which
  features Stage 0a is even allowed to consider as concepts; `feature_mask`
  separately controls which features condition the encoder-decoder target
  at train/eval time. Every requested ID must be present in the configured
  feature statistics; absence raises before the Neuronpedia client or output
  artifact is created because it can indicate a dead feature or an SAE/layer
  mismatch. After that fail-loud membership gate, filtering order is
  `min_corpus_count`, optional `concept_description_fetch_limit`, THEN the
  Neuronpedia description-quality filter (`min_description_length`) — curated
  features that fall below a threshold are dropped and logged
  (`concept_list.py`, `resolve_concept_feature_whitelist` in
  `concept_whitelist.py`). `candidate_concepts.json` never contains a
  feature outside the whitelist. When the whitelist is active, Stage 0a
  fast-fails with `AxBenchBaseError` if the surviving candidate pool drops
  below `cfg.max_concepts` at any filtering stage, rather than silently
  handing Stage 0c an under-sized pool. See the
  [Troubleshooting](#troubleshooting) table for the exact error text.
- **Non-determinism in Stages 0c, 1b, 1c.** These stages depend on
  sampled LLM output or sampled verification continuations from the
  base SAE LM; reproducing a run byte-for-byte also requires a
  warm `synth_response_cache.sqlite`. See
  `docs/evaluation/axbench-base-advanced.md#determinism`.

## Troubleshooting

| Error | Meaning | Fix |
|-------|---------|-----|
| `AxBenchBaseError: Tokenizer has no BOS token id (get_bos_token_id() returned ...)` | The loaded tokenizer does not implement `get_bos_token_id`. | Use a tokenizer version that exposes BOS; see `steering/eval/axbench_base/tokenizer_utils.py:85`. |
| `AxBenchBaseError: No features with corpus count >= {min_corpus_count}` | The feature statistics file has no features above the threshold. | Lower `min_corpus_count` or use a larger-corpus statistics file. Raised at `concept_list.py:115-116`. |
| `AxBenchBaseError: concept_feature_whitelist_path requested in-range feature IDs that are absent from per_feature_stats: absent_ids=...` | At least one requested ID has no `per_feature_stats` entry: it never fired in the statistics corpus, or the whitelist and statistics refer to different SAE/layer identities. | Investigate the named IDs and confirm the whitelist was produced from the exact configured feature/statistics domain. The run does not drop them or continue. |
| `AxBenchBaseError: No whitelisted features survive filtering: concept_feature_whitelist_path has ... IDs, ... present in feature statistics, and none pass min_corpus_count >= ...` | Every whitelist ID exists in feature statistics, but none meets `min_corpus_count`. | Enlarge the whitelist or lower `min_corpus_count`. |
| `AxBenchBaseError: concept_feature_whitelist_path yields only {N} candidate(s) after min_corpus_count>=... ..., but max_concepts={M} concepts are required` | The whitelist survives filtering but the surviving pool (`N`) is smaller than `cfg.max_concepts` (`M`) — either right after `min_corpus_count` or again after the Neuronpedia description filter. | Enlarge the whitelist, lower `min_corpus_count`/`min_description_length`, or lower `max_concepts`. |
| `AxBenchBaseError: concept_feature_whitelist_path plus concept_description_fetch_limit=... leaves only ... candidate(s) eligible for description fetch, but max_concepts=... concepts are required` | The profiling fetch cap truncates an otherwise adequate whitelist below the required final concept count. | Raise or disable `concept_description_fetch_limit`, enlarge the whitelist, or lower `max_concepts`. The error is raised before the Neuronpedia client is created. |
| `AxBenchBaseError: concept_feature_whitelist_path: manual feature id ... out of range [0, ...)` / `AxBenchBaseError: concept_feature_whitelist_path: ... duplicate ...` / `AxBenchBaseError: concept_feature_whitelist_path ... has unsupported extension ...` / `AxBenchBaseError: concept_feature_whitelist_path ... resolved to an empty selection ...` | The whitelist file failed validation (feature ID outside `[0, n_sae_features)`, a duplicate ID, an extension other than `.txt`/`.yaml`/`.yml`, or zero selected IDs — e.g. an all-commented manual YAML). | Fix the whitelist file: dedupe IDs, keep IDs within range, use a supported extension, and uncomment at least one ID. |
| `AxBenchBaseError: Stage 0c exhausted candidate_concepts.json without accepting ...` | Stage 0c could not find enough candidates that pass lazy examples QC and local-SAE negative-prefix verification, using either the default API continuation source or the explicit-null base-LM source. | Inspect `concept_examples_qc_report.json` and `rejected_concepts.json`; lower `activation_example_judge_min_score` or `verification_percentile_threshold`, adjust the candidate pool, verify the Gemma/SAE/tokenizer identity, and inspect API parse/retry warnings when `verification_api_model` is active. |
| `invalid or legacy Stage 1b QC contract` | `synth_training_positive.parquet` predates the fraction-to-raw-point correction, carries malformed provenance, or (for Base consumers) was generated with a different `accept_threshold`. Its `qc_accepted` flags cannot be trusted under the current semantics. | Preserve the old artifacts for diagnosis and regenerate Stage 1b under a fresh `run_tag`. Do not add metadata to the old file in place. |
| `Stage 1b checkpoint fingerprint mismatch` / target decrease / unsafe v1 migration / malformed, missing, or inconsistent partial rows | The same run tag was reused with a changed synthesis-semantic input, fallback policy, or population-drop threshold; the current target is below `highest_requested_target`; a schema-v1 target is not 72; or JSON commits rows that parquet cannot supply or validate. | Preserve both files for diagnosis. Schema-v2 may resume at the same or a higher target with the same explicit `run_tag`; otherwise restore the exact matching config or deliberately remove both partial files for a fresh run. A parquet-only orphan is safe to overwrite; JSON without parquet is committed data loss. |
| `Stage 1b completed positive corpus target mismatch` | A final positive parquet exists and its current selection-contract target differs from the request. This also applies when `stage1b_partial.json` remains after interrupted final-checkpoint cleanup. | Preserve the completed evidence and use a fresh explicit `run_tag`. A crash-window partial may resume the already-committed target, but completed clustered and unclustered corpora are never overwritten to grow it. Corrupt parquet and same-target selection drift retain their own validator errors. |
| `stage1b/stage1c concept population guard failed` | More concepts exceeded `max_concept_drop_fraction` than `max_dropped_concept_fraction` allows, or no concepts survived. | Inspect the named `stage1*_concept_yield.json` diagnostic and its per-concept reason counts. Changing a Stage 1b fallback/population threshold invalidates its partial fingerprint, so preserve the old files for diagnosis and use a fresh run tag (or explicitly remove both partial files) under the new policy. Do not hand-edit the filtered list. |
| `filtered concepts do not match ... concept-yield report` / `is not an ordered, unique subsequence` | A filtered population file is stale, edited, reordered, renumbered, or paired with a report/parent from another run. | Preserve the artifacts for diagnosis and rerun Stage 1b/1c under a fresh `run_tag`. Stage 0c `concepts.json` is immutable. |
| `negative_mode_prefixes parquet ... missing 'extraction_tokenizer_id' metadata` | The completed Stage 0c prefix artifact was written before tokenizer scoping metadata was added. | Use a new `run_tag`, or delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c for the current tokenizer. |
| `negative_mode_prefixes parquet ... missing 'feature_extraction_model_id' metadata` (or another extractor-identity mismatch) | The completed Stage 0c artifact predates extractor scoping or came from a different model / SAE source. | Use a new `run_tag`, or delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c under the current identity. |
| `negative_mode_prefixes parquet ... missing 'candidate_pool_identity' metadata` / `candidate_pool_identity=... but the current ordered candidate_concepts.json has ...` | The completed Stage 0c output predates pool scoping or Stage 0a regenerated a different ordered candidate pool under the pinned `run_tag`. | Preserve artifacts for diagnosis, then use a new `run_tag`, or delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c. |
| `Stage 0c resume: the completed artifact set is invalid ... Refusing an automatic Stage 0c rerun` | All three canonical outputs exist, but at least one is unreadable, malformed, structurally invalid, or incompatible with the current config. | Preserve the files for diagnosis. Then use a new `run_tag`, or explicitly delete the Stage 0 artifacts and rerun Stage 0a then Stage 0c. |
| `Stage 2: synth training parquet ... missing 'extraction_tokenizer_id' metadata` | The Stage 1 synth artifact was written before tokenizer scoping metadata was added, or was copied from an older run. | Rerun Stage 1b/1c for the current `cfg.extraction_tokenizer_id` before running Stage 2. |
| `Stage 3a DiffMean` / `Stage 3b LinearProbe` / `Stage 3d LoRA: synth training parquet ... missing 'extraction_tokenizer_id' metadata` | Stage 3 was pointed directly at stale Stage 1 synth rows. | Rerun Stage 1b/1c for the current `cfg.extraction_tokenizer_id` before training baselines. |
| `missing 'axbench_stage1b_positive_selection_contract' metadata` / unsupported `positive_prefix_headroom_v2` / `selection contract mismatch` | A positive Stage 1b corpus predates deterministic target/refill, or its target, attempt cap, guard, ordering, prefix/headroom, batch shape, or scoring protocol differs from the current config. Stage 2 and positive-mode Stage 3 refuse to combine those semantics. | Preserve the stale artifact for diagnosis, then regenerate Stage 1b and downstream products under a fresh auto-generated `run_tag`. An interrupted explicit-tag run with no committed positive corpus may reuse its durable response cache. |
| `missing 'axbench_stage1_continuation_contract' metadata` / `Stage 1 continuation contract mismatch` | A Stage 1b/1c corpus predates authenticated continuation parsing, has malformed metadata, has the positive/negative role swapped, or was generated with different envelope/echo/copy/token-window semantics. Base Stage 2/3, BPB, and BPB-Adv refuse to reinterpret its rows. | Preserve the stale artifact for diagnosis and regenerate Stage 1b/1c plus downstream products under a fresh `run_tag`. Never stamp current metadata onto old rows. |
| `concept_features safetensors ... delta_ranked but missing 'extraction_tokenizer_id' metadata` | Stage 4 was pointed at an old or hand-written delta-ranked Stage 2 artifact. | Rerun Stage 2 (`CONCEPT_FEATURE_MAPPING`) with the current `cfg.extraction_tokenizer_id`. |
| `AxBenchBaseError: Missing adapter key: layer_N_... (in {adapter_path})` | A LoRA adapter file was trained with a different `layer_hooks.diffmean_and_probe_layer` or different wrap set. | Retrain the adapter with matching config, or point at the correct adapter. Raised at `baselines/lora/loader.py:148-152`. |
| `AxBenchBaseError: OPENROUTER_API_KEY is not set in the environment, but the following requested stages require it: [...]` | `validate_env_for_stages` caught a missing API key before any heavy-model load. | `export OPENROUTER_API_KEY=sk-or-...` and re-run, OR avoid Stages 1b/1c/5 and set both `enable_concept_examples_qc=false` and `verification_api_model=null` for a local-only Stage 0c. |
| `AxBenchBaseError: NEURONPEDIA_API_KEY is not set in the environment, but the following requested stages require it: [...]` | Runtime preflight found Stage 0a, or Stage 0c with examples QC or API generation, without Neuronpedia credentials. | Export `NEURONPEDIA_API_KEY` and rerun. Disabling both examples QC and API generation removes the Stage 0c requirement, but Stage 0a still requires the key. |
| `AxBenchBaseError: Stage '4' cannot run: ... required artifact(s) missing on disk: [...]` | `check_stage_dependencies` (`dag.py`) walked the Stage 4 per-concept dep table for the final filtered concept artifact and found at least one missing baseline direction / adapter. Corpus presence cannot suppress this check. | Run the prerequisite stages first. The simplest fix is `stage=all` for a full run; otherwise run the stages listed in the error message in order. |
| `AxBenchBaseError: RuntimeState.close() called with active model contexts: {...}` | Internal bug: a stage function escaped its `with state.<model>() as m:` block via a bare `return` or raised without unwinding the refcount. | Open an issue with the full traceback. The state's `close()` refuses to tear down while any refcount is non-zero (`runtime_state.py`, `close()` around line 826) so the operator can see which context leaked. |
| GPU out-of-memory mid-run | Estimated peak memory budget was ~13 GB (component-sum calculation; see [Budget breakdown](../evaluation/axbench-base-advanced.md#budget-breakdown)) and your device is smaller. | Check the device — AxBench-Base expects ~13 GB with Gemma-2b + d12. Either free GPU memory, set `device=cpu` (slower by ~20x), or switch to a smaller tier (`tiers=dev`). |
| `OMP: Error` / OpenRouter rate limit errors during Stage 0c / 1b / 1c / 5 | Synth or judge API throttling. | Lower `cfg.synth.requests_per_minute` (default 6000) and/or `cfg.judge.requests_per_minute` (default 60). The limits are independent and per process. |
| Non-rank-0 ranks crash/abort early under `torchrun` while rank 0 is in a long single-rank stage (e.g. Stage 0a Neuronpedia fetch) | The DDP collective timeout fired on the barriered ranks | The Stage 0c launcher defaults `NCCL_TIMEOUT_SEC=1800` (PyTorch's default is preserved for other DDP jobs — no global change). Raise it further (seconds) for a rank-0-only stage longer than 30 min; setting it affects only this job. Must be a positive integer (and within `timedelta`'s range) or startup fails fast with a clear error. |

## See Also

- `docs/evaluation/axbench-base.md` — full user guide, config reference, output
  schemas, cost and runtime estimates.
- `docs/evaluation/axbench-base-advanced.md` — package internals, stage-by-stage
  deep-dive, LLM response cache, model metadata detection.
- `docs/pipelines/feature-extraction.md` — how to build the
  `feature_dir` this pipeline consumes.
- `docs/pipelines/encoder-decoder-training.md` — how to train the
  encoder-decoder target model.
- `docs/pipelines/base-training.md` — how to train the decoder-only
  target model.
- Wu et al. 2025, "AxBench: Steering LLMs? Even Simple Baselines
  Outperform Sparse Autoencoders"
  ([arXiv:2501.17148](https://arxiv.org/abs/2501.17148)).

## Stage 0c partial-progress checkpointing

At the end of each completed per-rank candidate block, and once more when the
rank loop exits, each Stage 0c rank writes two partial-progress artifacts under
the run output directory:

- `stage0c_partial_rank_{rank}.json` — small bookkeeping (next candidate index,
  accepted/rejected concepts, QC bookkeeping, config fingerprint, extractor
  identity, and candidate-pool identity).
- `stage0c_partial_rank_{rank}.parquet` — verified prefix rows so far,
  with the final `negative_mode_prefixes.parquet` columns plus the temporary
  `candidate_index` used for deterministic cross-rank merge.
  The final parquet carries the extractor and candidate-pool identity directly
  in schema metadata after the per-rank files are merged.

These artifacts let an interrupted Stage 0c (crash, OOM, kill, network issue)
resume from the first unvisited candidate after the last durable block when
re-run with the same `run_tag`, instead of restarting from candidate 0. With
the same topology, the resumed run produces the same JSON bytes and logical
parquet content as an uninterrupted run (physical parquet encoding is not an
API). A changed `world_size` is safe and may reprocess moved candidates. With
fixed inputs and QC responses, stable candidate-derived generation seeds and
canonical merge order preserve exact same-mode output across that topology
change. A changed uncached external-QC response is a changed input, not a
generation-topology guarantee.

Completed-run resume is a separate path. Once the three canonical outputs
exist, `negative_mode_prefixes.parquet` is the durable identity anchor: its
schema metadata pins `extraction_tokenizer_id`, `neuronpedia_model_id`, the
effective `feature_extraction_model_id` (the explicit value, or
`neuronpedia_model_id` when unset), and `neuronpedia_source`. It also pins
semantic SHA-256 identities for the ordered Stage 0a candidate pool,
`concepts.json`, `rejected_concepts.json`, the conditional
`concept_examples_qc_report.json` state, and the full Stage 0c fingerprint.
When QC is enabled, the report is structurally validated and its semantic hash
must match. When QC is disabled, metadata pins a null-report sentinel and a
stale report path is forbidden. Completed resume reloads the JSON artifacts and
compares their current semantic hashes. `None`
from the resume check means only that the canonical set is absent or
incomplete. Once all three outputs exist, missing or mismatched metadata,
unreadable files, wrong schemas, malformed JSON, row-integrity failures, and
current-config incompatibilities all raise before model loading. Completed
state is never silently overwritten: use a new `run_tag`, or explicitly delete
the Stage 0 artifacts and rerun Stage 0a then Stage 0c. This prevents both an
automatic Stage 0c-only rerun from consuming an identity-scoped Stage 0a pool
and any old/new mixture of the completed Stage 0c artifact set.

Each checkpoint is written atomically (parquet first, then JSON; both via tmp
file + rename). The first commit after a topology change may shrink a rank's
retained rows, so it first invalidates the old JSON commit marker; a crash in
that transition safely restarts that rank instead of pairing old JSON with a
smaller new parquet. On successful Stage 0c completion, JSON sidecars and all
row semantics are validated before `negative_mode_prefixes.parquet` is
installed last as the completed-set commit marker. All per-rank partial files
are then deleted because their data is preserved in the canonical artifacts.

### What invalidates a partial checkpoint?

A partial checkpoint is honored only when the current config matches
the checkpoint's config fingerprint. Changing any of these fields
forces an explicit operator decision:

`max_concepts` is deliberately not fingerprinted: after #1069 it is a
stopping target, so increasing it continues the same ordered candidate stream
instead of invalidating already verified candidates.

`stage0c_algorithm_version`, `feature_dir`, `feature_statistics_path`, `validation_shards`,
`neuronpedia_model_id`, `feature_extraction_model_id`, `neuronpedia_source`,
`extraction_tokenizer_id`,
`n_verified_prefixes_per_concept_target`, `n_verification_seeds`,
`min_consistency_passes`, `verification_percentile_threshold`,
`verification_length`, `min_prefix_chars`, `max_prefix_chars`,
`verification_zero_hit_bail_after`,
`verification_candidate_attempt_limit`,
`verification_candidate_scan_limit`, `enable_concept_examples_qc`,
the conditional `concept_examples_qc_config` (including
`n_activation_examples`, the judge provider/model/rate/concurrency settings,
and the Neuronpedia fetch worker count),
`activation_example_judge_min_score`,
`concept_examples_qc_target_surviving`, `generation.master_seed`,
`generation.temperature`, `generation.top_k_sampling`, `qc_batch_size`,
`qc_pipeline`, `verification_prefill_mode`,
`stage0c_multi_gpu_block_size`, and, for explicit-null local mode,
`verification_cross_prefix_batch_size`.

When `verification_api_model` is non-null, the fingerprint additionally has a
conditional API-generation object containing protocol version,
provider/model, requests-per-minute, concurrency, API prefix-wave size,
content-retry budget,
activation-example count, evidence cache/rendering/dedup constants, and hashes
of the definition prompt, continuation user prompt, and continuation system
prompt. For a clustered source it also contains
`cluster_member_description_cycle`: the cycle-policy version, the
member-activation-evidence description source, and the canonical flattened
proposal-slot round-robin assignment. With a null model this object is
absent—not present
with null values—so the legacy 29-field schema-8 fingerprint is byte-identical.
Changing dormant `synth.model` does not invalidate API-mode work because
`verification_api_model` is the authoritative model ID. Changing dormant
`verification_cross_prefix_batch_size` likewise does not invalidate API-mode
work; the conditional API wave size is authoritative there.

The ordered `candidate_concepts.json` content hash is an additional derived
fingerprint field. Changing any candidate row or its order also invalidates the
checkpoint even when every config value above is unchanged.

Issue #713 raises both the Stage 0c algorithm/checkpoint schema and the
per-rank checkpoint schema from 7 to 8. Schema-7 partials, rank checkpoints,
and completed parquets are rejected; there is no migration because old rows
cannot safely be relabeled with shared-prefill semantics. Switching between
`shared` and `repeated` also changes fingerprint identity. Use a new `run_tag`,
or deliberately preserve then remove the stale Stage 0c artifact set before
rerunning. The removed `batched_generation_mode` override fails Hydra
composition instead of silently selecting behavior.

The API-generation algorithm is a conditional schema extension:
`verification_api_model=null` remains schema 8, while the default non-null
model uses schema 9 for partial JSON/parquet and the completed parquet. Readers are passed
the expected schema from the current config and reject cross-mode or unknown
versions. There is deliberately no schema-8-to-9 migration because the stored
continuations came from different generators and the API prompt/evidence
identity cannot be reconstructed from old rows.

`feature_dir` and `feature_statistics_path` are path identities, not recursive
content hashes of the (potentially very large) feature corpus. Resume therefore
relies on the repository-wide immutable-artifact contract: do not overwrite
feature shards, their source parquets, or feature statistics in place. If any
of those inputs is regenerated at the same path, use a new `run_tag` or delete
the Stage 0 partial/final artifacts and rerun from Stage 0a.

### Validation-failure behavior

If the partial checkpoint exists but is incompatible (fingerprint
mismatch, schema-version mismatch, structural corruption), Stage 0c
**raises `AxBenchBaseError`** with a message naming the failing check,
the full paths of the partial files, and instructions to opt into a
fresh restart. The partial files are NOT deleted automatically — this
preserves potentially-expensive verification work and lets the
operator decide whether to recover, copy aside, or discard.

To opt into a fresh restart after a fingerprint change:
- Delete every `stage0c_partial_rank_*.json` and
  `stage0c_partial_rank_*.parquet` file in the run output directory, OR
- Re-run with a different `run_tag` to start a clean run while
  preserving the old partial artifacts.

### Edge case: interrupted before any candidate is accepted

If a durable block has no accepted candidates, its rank checkpoint contains an
empty typed parquet alongside the JSON bookkeeping. Resume treats zero rows as
the exact state implied by zero accepts and continues normally. A JSON commit
marker with a missing parquet is corruption even when there are zero accepts;
Stage 0c preserves the JSON and raises instead of guessing that data was empty.

### Mid-write crash recovery

If a crash interrupts a block commit between the parquet
write and the JSON write, the parquet has rows past the previous
JSON's tally. On resume, the excess rows are truncated to match the
JSON tally, an info-level log line is emitted, and verification
continues normally.
