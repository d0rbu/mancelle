# AxBench-Base Evaluation

A concept-steering benchmark for **base** (pre-instruction-tuned) language
models. AxBench-Base asks: given an SAE feature and its human-readable
description, how well does each steering method push a base LM's generation
toward (or away from) that concept, without breaking fluency?

## What it measures

[Wu et al., "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse
Autoencoders" (arXiv:2501.17148)](https://arxiv.org/abs/2501.17148) evaluates
steering methods on instruction-tuned models using dialog-style prompts.
**AxBench-Base** adapts the same scoring rubric (concept relevance + fluency,
on a 0-2 scale) to a benchmark target that does not take instructions: a
plain next-token continuation model trained on the same corpus used for
encoder-decoder feature extraction.

The target audience is a specific research claim: conditional encoder-decoder
models (our cross-attention-conditioned decoder-only GPT variant) can steer
base LMs toward interpretable concepts at least as well as the classical
steering baselines that AxBench introduced. The evaluation pipeline produces:

- a **concept list** of ~500 SAE features drawn from GemmaScope, each with
  a Neuronpedia-sourced description
- a **training corpus** of concept-rich and concept-avoiding continuations,
  synthesized by an OpenRouter LLM
- **baseline directions / adapters** (DiffMean, LinearProbe, LoRA, Prompting)
  trained per-concept from that corpus
- **generated completions** from every method across a coefficient sweep
- **judge scores** produced by an LLM judge on concept-relevance and fluency
- an aggregated **leaderboard** comparing methods by area-under-the-dose-
  response curve

Two modes are evaluated per concept:

- **Positive mode** — can the method steer a neutral prefix toward the
  target concept? The evaluation seeds are clean prefixes (concept
  feature inactive). A strong method produces continuations that mention
  the concept without becoming incoherent.
- **Negative mode** — can the method steer a concept-rich prefix AWAY
  from the target concept? The evaluation seeds are prefixes where the
  concept is already active (verified by locally SAE-scoring several
  continuations from either the default Stage 0c API generator or the base SAE
  LM selected with `verification_api_model=null`). A strong method redirects
  the continuation into unrelated
  territory while staying fluent.

## Pipeline at a glance

```
          Stage 0                 Stage 1              Stage 2            Stage 3         Stage 4        Stage 5        Stage 6
  +-----------------------+  +----------------+  +------------------+  +----------+  +------------+  +----------+  +-----------+
  | 0a candidate pool     |  | 1b positive    |  | concept -> SAE   |  | baseline |  | generation |  | llm      |  | aggregate |
  | 0b seed pools         |->| 1c negative    |->| feature mapping  |->| training |->| per method |->| judge    |->| leaderbd  |
  | 0c negative prefixes  |  | paired corpora |  | (canon/delta)    |  | 3a-e     |  | + sweep    |  | (api)    |  | (math)    |
  +-----------------------+  +----------------+  +------------------+  +----------+  +------------+  +----------+  +-----------+
  disk + NP + GPU/API opt.    OpenRouter LLM         GPU optional        GPU only       GPU only     OpenRouter      pure math
```

Every stage is wired end to end through the Hydra CLI at
`scripts.eval.axbench_base_eval`. A shared `RuntimeState`
(`steering/eval/axbench_base/runtime_state.py`) lazy-loads the heavy
resources (base Gemma + SAE, target models, engines, SQLite caches,
judge client) and CPU-offloads them between stage boundaries so peak
GPU memory stays bounded. A bare
`uv run python -m scripts.eval.axbench_base_eval tiers=dev ...` kicks off
every stage in order; see
[Running the full benchmark](#running-the-full-benchmark) below.

**Where cost lives.** Stage 0a hits the Neuronpedia API once per candidate
feature to fetch published descriptions. When examples QC is enabled, Stage
0c lazily fetches raw activation examples and judges only the candidates it
actually visits before reaching `max_concepts`; it does not eagerly judge the
entire Stage 0a pool. The `SynthResponseCache` makes those calls and reruns
near-free once warm. Stage 1b runs one OpenRouter request chain per admitted
candidate, with up to `positive_max_attempts_per_seed` semantic attempts on
each generated side and lazy replacement until its complete-pair target is
met. Stage 1c retains up to `cfg.max_retries` immediate content resamples.
Each proposal independently allows `cfg.synth.transport_max_retries`
backoff-governed HTTP retries.
Stage 0b extracts SAE features from every exact seed prefix with the
production Gemma+SAE extractor.
By default Stage 0c uses `inclusionai/ling-2.6-flash` through OpenRouter for
continuation generation: one evidence-grounded definition call per visited
feature plus one call per candidate-prefix/verification-seed slot (including
content retries). Local SAE rejection scoring is unchanged. Set
`verification_api_model=null` to use the base SAE LM locally instead; that
compatibility path preserves the old N-seed generation loop exactly. A
separate API-only prefix wave keeps provider concurrency independent of local
Gemma KV-cache batching. Provider latency, retry rate, and feature hit rate are
workload-dependent, so measure the intended feature/model regime before making
a production wall-clock assumption.

### OpenRouter-offline real-Gemma integration

The real-Gemma integration test is OpenRouter-offline, not universally
network-free. Production clients use the cached thread-local
`requests.Session.post` path; the test patches that exact boundary and also
tripwires module-level `requests.post`, plus OpenRouter `Session.request`,
`Session.get`, and `Session.send`, before any adapter, DNS, socket, or provider
access. Non-OpenRouter HTTP traffic remains available for the separate
Neuronpedia/HuggingFace prerequisites. Every OpenRouter request is classified,
recorded, and fail-closed if it is not one of Stage 0c definition/continuation,
Stage 1 rich/avoid synth, or Stage 5 judge.

The test removes any inherited `OPENROUTER_API_KEY` and injects the clearly
named `sk-or-test-offline` value only to exercise runtime credential
validation and request-header construction. No OpenRouter credential is
required. Stage 0a Neuronpedia descriptions and Stage 0c activation evidence,
plus HuggingFace and nanochat caches, remain separate real prerequisites.
Stage 0c responses are nonce-framed and validated by the production parser;
their generated rows still pass the unchanged local Gemma/SAE percentile and
consistency gate. Live provider coverage remains under `@pytest.mark.api` and
requires explicit credentials.

Stage 2 runs the base Gemma + SAE
forward pass on all training texts when `concept_feature_mode="delta_ranked"`
is set; in the default `canonical` mode it does zero GPU work.

**Where determinism lives.** Stages 0a (post Neuronpedia cache warmup), 0b,
2, and 3e are deterministic given fixed inputs. Stage 0b additionally
checks the candidate stream, model/SAE tensor hashes, and a fixed extraction
canary across ranks before doing full work. Stages 1b, 1c, and 0c are
not — they depend on sampled LLM output (Stages 1b/1c and default API-mode
Stage 0c) or sampled verification continuations from the base SAE LM
(`verification_api_model=null`). Cached API responses make an interrupted
rerun replayable, but do not
make a cold provider response deterministic. See the
"Determinism" section in `docs/evaluation/axbench-base-advanced.md`.

### Clustered subset as landed

AxBench-Base has a cluster-aware subset; it is not a clustered version of the
entire `stage=all` benchmark. Configure `conditioning.feature_mask` and
`conditioning.feature_clusters` together as path-backed mappings, keep
`concept_feature_mode=canonical`, provide an explicit method list, and invoke
one supported stage at a time:

- source construction: `0a`, `0b`, `0c`, `1b`, `1c`, `2`, `3a`, and `3b`,
  using only the positive/negative decoder-only DiffMean, linear-probe, and
  control methods;
- exact generation boundary: `3e` and `4`, using those six methods plus
  `control_encdec`, `neg_control_encdec`, and the seven ordinary
  `ours_*_encdec` context-editing methods in the shared clustered allowlist.

Stages `3c`/`3d` (prompting/LoRA), `5`, `6`, and `all` fail config validation
when clusters are active. Main does not ship the frozen campaign's
`axbench_base_cluster_source` or `axbench_base_cluster_stage4` profiles: start
from `axbench_base.yaml`, add the absent cluster key with
`+conditioning.feature_clusters.path=...`, and preserve every identity-bearing
override across the separately invoked stages. The exact command shape and
allowlists are in the [CLI reference](../cli-reference.md#steering-evals).

The cluster feature contract pins the path-backed mask and canonical cluster
content to the checkpoint, effective vocabulary, statistics, SAE/feature
directory, selected-feature manifest, and concept-source domain. Stage 0a
projects mask-allowed candidates to representatives before deduplication; the
existing Stage 0c member-description cycle and cluster-wide local acceptance
oracle then operate on that fixed target. Stages 3a/3b stamp the contract and
source-corpus digest into their direction artifacts. Stages 3e/4 bind exact
prompt/cache/generation identities and reject drift rather than silently
adopting a different checkpoint or selection.

## Prerequisites

Before running any stage, you need the following artifacts and environment:

1. **A feature directory.** Produced by the feature-extraction pipeline
   documented in [`docs/pipelines/feature-extraction.md`](../pipelines/feature-extraction.md).
   The path must follow the
   `{root}/s{seg_size}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}`
   convention (see `steering/feature_paths.py:71-118`) because `axbench_base`
   parses it to recover `layer`, `segment_size`, `model`, `width`, and
   `phase` at runtime. A typical path for the default SAE:

   ```
   out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/
   ```

2. **A feature statistics safetensors file.** Typically
   `feature_statistics.safetensors` inside the feature directory, produced
   by `steering.compute_statistics`. The file drives concept candidate
   filtering (`corpus_count >= min_corpus_count`), the percentile table
   used by synth QC, and the bucket boundaries used by Stage 4 approach-2
   extraction.

3. **A target decoder-only nanochat base model.** Referenced by
   `cfg.decoder_only = {source, model_tag, step}`. The `RuntimeState`
   loads it on demand via `nanochat.checkpoint_manager.load_model`
   (`runtime_state.py`, `_load_decoder_only_model` around line 392).
   Used in Stages 3a/3b (diffmean + linear probe baseline training,
   decoder-only target), 3d (LoRA training), and 4 (generation dispatch).

4. **A target encoder-decoder nanochat model.** Referenced by
   `cfg.encdec = {source, model_tag, step}`. `RuntimeState._load_encdec_model`
   (`runtime_state.py`, `_load_encdec_model` around line 430)
   attaches the checkpoint metadata to the
   returned module as `model._loaded_meta_data` so downstream
   `detect_target_model_metadata` can read training-time
   `negative_selection` / `feature_selection` without re-opening the
   checkpoint.

5. **A base Gemma model + SAE.** Loaded by
   `RuntimeState._load_base_model` /`_load_base_sae`
   (`runtime_state.py`, `_load_base_model` around line 312 and
   `_load_base_sae` around line 356) via
   `steering.extract_features.load_sae_and_model`. These are the
   analysis pair, not a target. For GemmaScope-res-16k the pairing is
   `google/gemma-2-2b` plus the corresponding SAELens SAE, specified by
   `cfg.neuronpedia_model_id` + `cfg.neuronpedia_source`.

6. **API credentials.** `NEURONPEDIA_API_KEY` is required for Stage 0a and
   for Stage 0c when examples QC is enabled or `verification_api_model` is
   non-null. `OPENROUTER_API_KEY` is required for Stage 0c when either of
   those options is active, Stage 1b,
   Stage 1c, and the Stage 5 judge. The CLI's
   `validate_env_for_stages` (`runtime_state.py`,
   `validate_env_for_stages` around line 143) runs BEFORE
   any heavy-model load so missing credentials surface immediately and
   name every stage in the current run that needs the key. To run the
   Stage 0c without external API access, set
   `enable_concept_examples_qc=false`, `verification_api_model=null`,
   `concept_feature_mode=canonical`,
   and `stage=<id>` to skip Stages 1b/1c/5.

7. **A GPU.** Stages 0b through 4 touch at least one of
   base Gemma + SAE, the decoder-only target, or the encoder-decoder
   target. Peak memory is an estimated ~13 GB for `gemma-2-2b` +
   `gemmascope-res-16k` + d12 decoder-only + d12 encdec under the
   `RuntimeState` CPU-offload pattern (the figure is calculated from
   component sizes, not measured end-to-end; see the advanced
   doc's "Budget breakdown" section); Stage 0c keeps only the
   feature-extraction tier (Gemma+SAE) live, while Stage 4 is the target
   tier peak.
   Use `device=cpu` for CI / small-box runs at ~20x runtime cost.

8. **`NCCL_TIMEOUT_SEC` environment variable** (optional, multi-GPU
   Stages 0b/0c). DDP process-group collective timeout in seconds.
   PyTorch's backend default (~10 min NCCL / 30 min GLOO) is preserved
   for all DDP jobs unless this is set — it is not a global default.
   The Stage 0c launcher (`scripts.eval.axbench_base_eval`) defaults it
   to `1800` (30 min) for its long rank-0-only Stage 0a; setting it
   affects only the job where it is set. Must be a positive integer
   (within `timedelta`'s range); an invalid value fails fast at
   startup. Raise it when a rank-0-only stage runs longer than 30
   minutes under `torchrun`. Also note: `validate_config` requires
   `max_concepts >= 1`.

## Running the full benchmark

The Hydra CLI at `scripts.eval.axbench_base_eval` dispatches every stage from
0a through 6. A cold encoder-decoder run deliberately pauses between Stage 3e
and Stage 4: extract the exact Stage 3e context parquet and convert/validate its
cache before generation. A first `stage=all` invocation can produce 0a-3e and
then fail the cache preflight; after the cache is built, rerunning the same
command resumes at Stage 4. See the
[pipeline runbook](../pipelines/axbench-base-eval.md#exact-stage-4-context-cache)
for exact commands. A shared `RuntimeState` (see
`steering/eval/axbench_base/runtime_state.py`) owns the base Gemma + SAE
pair, both target-model loaders, the
SQLite generation / synth / judge caches, and the OpenRouter judge
client; all four "heavy" models live behind refcounted context
managers that offload to CPU between stage boundaries so peak GPU
memory stays under an estimated ~13 GB with the default
`gemma-2-2b` + d12 stack (component-sum estimate; see
[advanced doc](axbench-base-advanced.md#budget-breakdown)).
Stage 4 enters only the target families that survive
`enabled_methods`/`disabled_methods`; the default all-method plan enters both.

```bash
# Dev tier: 5 concepts. Use a fresh explicit tag; build the Stage 3e cache
# before the first Stage 4 generation as described above.
uv run python -m scripts.eval.axbench_base_eval \
    tiers=dev \
    feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
    feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
    decoder_only.source=base decoder_only.model_tag=d12_base \
    decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline \
    encdec.step=54600 \
    device=cuda output_root=out/eval/axbench_base
```

```bash
# Standard tier: 50 concepts.
uv run python -m scripts.eval.axbench_base_eval tiers=standard \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base \
    decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline \
    encdec.step=54600
```

```bash
# Full tier: 500 concepts.
uv run python -m scripts.eval.axbench_base_eval tiers=full \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base \
    decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline \
    encdec.step=54600
```

### Running a subset of stages

`stage=<id>` accepts any member value of the `Stage` `StrEnum`
(`steering/eval/axbench_base/types.py:66-80`):
`0a`, `0b`, `0c`, `1b`, `1c`, `2`, `3a`, `3b`, `3c`, `3d`, `3e`, `4`,
`5`, `6`, or the pseudo-stage `all`. Every other stage identifier
raises `AxBenchBaseError` from `Stage.coerce`.

```bash
# Only Stage 4 (generation) against an existing run_tag.
uv run python -m scripts.eval.axbench_base_eval tiers=dev stage=4 \
    run_tag=20260414-191100-abcdef01 \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base \
    decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline \
    encdec.step=54600
```

`check_stage_dependencies`
(`steering/eval/axbench_base/dag.py`, `check_stage_dependencies` at
line 362) walks the declared DAG before the stage body runs and
raises `AxBenchBaseError` naming every missing path if a prerequisite
hasn't been produced yet. Stage 4 uses a per-concept dep check (in
the same `dag.py`) that verifies every `(concept, target, mode)`
**in the final filtered Stage 1 population**
baseline direction / adapter already exists.

### Resuming a crashed run

Re-run the same command. The dep check no-ops every stage whose
outputs are already on disk, and the SQLite caches
(`synth_response_cache.sqlite` and `judge_cache.sqlite`) replay prior
OpenRouter responses without
spending tokens. Stage 4's parquet directory dataset is append-only
and crash-safe per batch via `ParquetAppendWriter`.

### Skipping negative mode

Pass `run_negative_mode=false`. This drops Stage 1c entirely, skips the
negative-mode concept-features file in Stage 2, and narrows Stage 4 /
5 / 6 to positive-mode methods only. The Stage 1c handler in
`_run_single_stage` short-circuits with a log line; every downstream
dep check honors the flag so no stage blocks on negative-mode outputs
that were never written.

```bash
uv run python -m scripts.eval.axbench_base_eval tiers=dev \
    run_negative_mode=false \
    feature_dir=... feature_statistics_path=... \
    decoder_only.source=base decoder_only.model_tag=d12_base \
    decoder_only.step=54600 \
    encdec.source=encdec      encdec.model_tag=d12_baseline \
    encdec.step=54600
```

### Where outputs land

`$NANOCHAT_BASE_DIR/eval/axbench_base/<sanitized_run_tag>/` by default.
Set `output_root` to a non-empty string to relocate the complete AxBench-Base
artifact tree; relative custom roots remain relative to the process working
directory. The CLI auto-generates
`run_tag=<YYYYMMDD-HHMMSS>-<8char-hash>`
if none is passed (`scripts/eval/axbench_base_eval.py`, `_generate_run_tag`
at lines 183-206). See
[Outputs and layout](#outputs-and-layout) for the full directory tree.

Migration safeguard: releases before issue #1001 ignored custom `output_root`
for run artifacts. If a config now points at a custom root, that target run is
absent, and the same `run_tag` exists under the historical default, startup
fails before expensive work whenever the tag is already known (including the
explicit-tag migration case). A clustered run with an autogenerated tag must
first resolve the checkpoint identity used to generate that tag. Resume with
`output_root=null`, or explicitly copy and validate the complete run before
retrying. There is no silent fallback or merge.

## Configuration reference

All config fields live in `AxBenchBaseConfig` at
`steering/eval/axbench_base/config.py:62-138`. Nested configs live in the
same file. Every default is enforced at construction; cross-field
invariants are checked by `validate_config` (lines 141-180) which you
should call before launching any stage if you built the config manually.

### Required fields (no default)

| Field | Type | What it controls |
|-------|------|------------------|
| `feature_dir` | `str` | Path to the precomputed SAE feature shards. Must follow the `s{seg}/l{layer}/a_{act}/p_{pool}/f{width}/m_{model}/{phase}` convention so `parse_feature_dir` can extract `layer` + `segment_size`. |
| `feature_statistics_path` | `str` | Path to the `feature_statistics.safetensors` produced by `steering.compute_statistics`. Drives concept filtering and the percentile table. |
| `decoder_only` | `dict[str, Any]` | Loader spec for the target decoder-only base model. Used by Stages 3-4. `gen_source` provenance strings read `source`, `model_tag`, `step` keys off this dict. |
| `encdec` | `dict[str, Any]` | Loader spec for the target encoder-decoder model. Same shape. |
| `device` | `str` | Torch device string, e.g. `"cuda"` or `"cuda:0"`. Stored on `TargetModelMetadata.device` so downstream helpers don't re-read the config. |
| `output_root` | `str \| None` | Parent directory for every AxBench-Base artifact. `None` preserves the historical `$NANOCHAT_BASE_DIR/eval/axbench_base`; a non-empty string is honored literally. Run artifacts live at `{effective_output_root}/{sanitized_run_tag}`. |

### Run bookkeeping

| Field | Default | Notes |
|-------|---------|-------|
| `run_tag` | `None` | Unique identifier for this run. The CLI auto-generates `YYYYMMDD-HHMMSS-<8charhash>` when `None` (`scripts/eval/axbench_base_eval.py`, `_generate_run_tag` at lines 183-206); explicit callers must set it before constructing a `RuntimeState`. Sanitized via `paths._sanitize` to prevent directory escape. Resuming Stage 1b requires reusing the same explicit tag because its partial files are run-local. |
| `stage` | `"all"` | CLI dispatcher input. Accepts any `Stage` `StrEnum` value — `0a`, `0b`, `0c`, `1b`, `1c`, `2`, `3a`, `3b`, `3c`, `3d`, `3e`, `4`, `5`, `6` — or the `all` pseudo-stage that expands to the full ordered stage list. Stage 3e enumerates the exact encoder-decoder prompts required by Stage 4. |
| `force_cluster_shard_revalidation` | `false` | Clustered-only recovery/proof path. False uses a valid stat-certified packed fingerprint cache while still hashing all compressed selection bytes. True runs the canonical full packed/selection validators, then atomically replaces and rereads the cache; under DDP rank 0 alone owns that body pass. Rejected without `conditioning.feature_clusters`; false is omitted from legacy run-tag bytes. |
| `enabled_methods` | `None` | Stage 4 method registry filter. `None` means "all 33 methods in the registry"; a list of method names (e.g. `[ours_extract_and_add_encdec, diffmean_decoder_only]`) restricts to that subset. Passed through `method_registry.filter_methods`. |
| `disabled_methods` | `[]` | Stage 4 method registry exclusion. Applied after `enabled_methods` narrows the set. |

### Feature-directory / SAE

| Field | Default | Notes |
|-------|---------|-------|
| `validation_shards` | `[369]` | Shard indices from which Stage 0b draws seed pools and Stage 0c scans candidate negative prefixes. Must be non-empty. A single shard is usually enough for 500 concepts. |
| `neuronpedia_source` | `"8-gemmascope-res-16k"` | Neuronpedia `sae_source` for description lookups. Matches the default SAE. |
| `neuronpedia_model_id` | `"gemma-2-2b"` | Neuronpedia `model_id`. |
| `feature_extraction_model_id` | `null` | Optional local model id for resolving/loading the feature-extraction model + SAELens SAE. Defaults to `neuronpedia_model_id`. Set this when the Neuronpedia API id differs from the local extraction id, e.g. `neuronpedia_model_id=gemma-3-4b` with `feature_extraction_model_id=gemma-3-4b-pt`. |
| `extraction_max_tokens` | `1024` | Hard upper bound on tokens fed to base Gemma during Stage 2 / Stage 4 extraction. Swapping to a longer-context base model only requires updating this. |
| `extraction_threshold` | `0.0` | Activation threshold for `classify_avoid_continuation` in negative-mode strict-zero QC. Only activations strictly above this value count as "fired". |
| `extraction_tokenizer_id` | `"google/gemma-2-2b"` | HuggingFace ID for the tokenizer used by Stage 0c verification generation tokenization, synth QC, and Stage 2 extraction. Must match the SAE's training tokenizer. |

**v2 (gemma-scope-2) sources.** When the SAE is from the gemma-scope-2 (v2)
release family, published medium-tier `neuronpedia_source` values omit the
tier (e.g. `22-gemmascope-2-res-16k`) instead of the v1 4-segment
`{layer}-gemmascope-{activation}-{width}` form. When Neuronpedia does not yet
host a requested v2 SAE, the client raises the typed
`NeuronpediaNotPublishedError` (`steering/eval/neuronpedia_client.py`) — catch
that specifically rather than a bare 404. The standalone
[feature-filter](../pipelines/feature-filter.md) handles the same condition
on its activation-fetch path by recording a `not_published` outcome that is
excluded from the error-rate abort (exit 0 + warning, not a spurious
failure). See
[`docs/pipelines/gemma-scope-2.md`](../pipelines/gemma-scope-2.md).

**Mass-404 is a source mismatch, not a per-feature gap (Stage 0a fail-loud).**
A *single* 404 is a known per-feature limitation, but a *high* 404 rate almost
always means `neuronpedia_source` / `neuronpedia_model_id` do not match the SAE
being evaluated — e.g. the default 16k `8-gemmascope-res-16k` queried against a
65k SAE, where every feature index `>= 16384` 404s. Left unguarded, Stage 0a
would silently restrict candidacy to the first 16k indices *and* mislabel them
(a `gemma-2-2b` description for index *i* is not `gemma-3-4b`'s feature *i*),
corrupting `concepts.json` and everything downstream (Stage 0c QC, synthesis,
and any BPB-Adv frontier built on the source). Stage 0a therefore opts into the
client's `max_not_published_description_rate` guard, so `prepare_concept_list`
raises `NeuronpediaSourceMismatchError` up front rather than proceeding on a
truncated pool. The description-only load-bearing distinction is deliberate:
decorative callers (autointerp) do not opt in and keep the non-fatal drop
behavior (issue #691). For a gemma-3-4b 65k SAE the matching source is
`neuronpedia_model_id=gemma-3-4b` with `neuronpedia_source=22-gemmascope-2-res-65k`
(and `feature_extraction_model_id=gemma-3-4b-pt` for local loading) — the
ready-made `configs/eval/axbench_base/axbench_base_gemma3_4bpt_l22_f65k.yaml`
sets exactly these, so a Gemma-3 l22 f65k run inherits the correct coordinates
instead of the gemma-2-2b defaults.

Three `validate_config` guards catch the misconfig **before** any stage runs (it
is called at the top of every CLI entrypoint), so all fail loud earlier than the
runtime rate guards. (1) It rejects an `extraction_tokenizer_id` that is not the
extraction model's own gemma-scope-2 tokenizer (catches a gemma-3 model left
with the gemma-2-2b tokenizer). (2) It rejects a run whose extraction model
(`feature_extraction_model_id or neuronpedia_model_id`) does not match the SAE
model encoded in `feature_dir` (the `m_{model}` segment) — the backstop for the
**all-defaults** case, where the model id is the v1 `gemma-2-2b` default that is
absent from the gemma-scope-2 tokenizer map, so the tokenizer check alone would
skip it. (Guard (2) also requires `feature_dir` to be a canonical feature path —
Stage 0a parses the layer, segment size, and SAE model from it — so a malformed
`feature_dir` fails loud here rather than deferring the less-clear failure to
Stage 0a.) (3) It rejects a run whose `neuronpedia_source` **layer or width**
(parsed from e.g. `22-gemmascope-2-res-65k` → layer 22, width 65k) does not
match `feature_dir`'s `l{layer}` / `a_{act}` / `f{width}`. (4) It derives the
canonical Neuronpedia model id for the SAE from `feature_dir`'s model via the
SAELens registry (`resolve_neuronpedia_ids`, which encodes the #667 API/local
split — a `gemma-3-4b-pt` SAE is published under model `gemma-3-4b`) and rejects
a `neuronpedia_model_id` that does not equal it. Guards (3) and (4) are the
**only** ones that catch a *same-model, wrong-layer/width/model* source:
Neuronpedia answers valid HTTP 200s for the wrong site, so no runtime rate guard
(404 / empty / failed) ever fires, yet every concept would be labeled with
another site's descriptions. A `neuronpedia_source` that is not a recognized /
published gemma-scope source also fails loud here (it cannot be checked against
the SAE). The runtime rate guards remain the final backstop for a source that
genuinely 404s / empties / 5xxs. Authentication is diagnosed separately — a
missing or rejected `NEURONPEDIA_API_KEY` raises a clear auth error rather than
tripping the source-mismatch guard.

### Concept list (Stage 0a)

| Field | Default | Notes |
|-------|---------|-------|
| `min_corpus_count` | `10` | Minimum per-feature corpus count from `feature_statistics` to be considered a candidate. Lower = more rare features enter the pool; higher = fewer but more frequent concepts. |
| `min_description_length` | `10` | Minimum character length of the Neuronpedia description. Prunes features with empty or stub descriptions. |
| `concept_list_seed` | `42` | RNG seed for the post-filter shuffle. Determinism depends on this and a warm Neuronpedia cache. |
| `concept_descriptions_cache_only` | `False` | When `True`, Stage 0a requires the exact identity-addressed cache file to exist before work begins, uses only descriptions already in it, and makes no API calls; uncached features in that existing cache are dropped from candidacy. The cache path is `{effective_output_root}/_neuronpedia_cache/v1/<sha256(canonical model_id,sae_source)>.yaml`. Leave `False` for normal fetching. |
| `max_not_published_description_rate` | `0.02` | Stage 0a fail-loud guard (signal 1 of 3). When more than this fraction of the count-filtered candidates return HTTP 404 / not-published from Neuronpedia, `prepare_concept_list` raises `NeuronpediaSourceMismatchError` instead of silently building a truncated, mislabeled concept pool. A high 404 rate means `neuronpedia_source` / `neuronpedia_model_id` do not match the SAE being evaluated (e.g. a 16k source against a 65k SAE); a correctly-configured run sees ~0%, so this threshold is tight. Empty-explanation and transient failures do not count toward it, and `concept_descriptions_cache_only` runs skip the check. `0.0` disables the guard; raise toward `1.0` for an SAE with genuinely sparse Neuronpedia coverage. |
| `max_missing_description_rate` | `0.5` | Fail-loud guard signal 2 of 3. Fires when more than this fraction of candidates have **no usable description at all** — HTTP 404 **plus** HTTP 200 with an *empty* explanation. Catches a wrong source that answers 200-empty (rather than 404) for indices it does not host. A correct source has only a small genuine-empty baseline, so this is deliberately loose. `0.0` disables it. |
| `max_failed_description_rate` | `0.5` | Fail-loud guard signal 3 of 3 (backstop). Fires when more than this fraction of candidates **could not be fetched for any reason** — 404 + empty + *transient* (5xx / rate-limit after retries). A wholly nonexistent or typo'd `neuronpedia_source` answers 5xx (not 404) for every index (the live API returns HTTP 500 for an unknown source), so its drops are all transient and slip the two guards above; this one catches that, and a rate-limit storm that would otherwise silently, nondeterministically truncate the pool. A reachable, correct source fails ~none. `0.0` disables it. |
| `min_source_mismatch_features` | `5` | Absolute floor shared by all three guards above: a guard fires only when its offending count *also* reaches this many features. Stops a small candidate pool (`concept_feature_whitelist_path` / `concept_description_fetch_limit`) or a brief network blip from tripping a rate on a handful of features. |
| `max_concepts` | `500` | Number of final verified concepts Stage 0c must accept into `concepts.json`. Stage 0a writes the full shuffled survivor pool to `candidate_concepts.json` and does not truncate to this value. |
| `enable_concept_examples_qc` | `True` | Whether to run the per-concept LLM judge that validates each description against the feature's top activating examples. Requires `OPENROUTER_API_KEY`. Set `False` to skip the Neuronpedia-activations fetch and save ~1 LLM call per candidate concept. |
| `n_activation_examples` | `5` | How many top-activating examples are fetched per concept and shown to the QC judge. |
| `activation_example_judge_min_score` | `1` | Concepts with QC score strictly below this are dropped. Scale is 0-2: 0=no match, 1=partial, 2=strong. |
| `concept_examples_qc_target_surviving` | `null` | Optional single-rank dev/profiling cap on the number of QC survivors sent to verification. It is rejected when `world_size > 1` because a per-rank cap is not an exact global cap; use `null` for multi-GPU Stage 0c. |
| `concept_examples_qc_judge` | `LLMClientConfig()` | The OpenRouter client config used by the QC judge. It retains the conservative `openai/gpt-5-nano`, 10-concurrent, 60-RPM defaults independently of synth traffic. |

### Seed pools (Stage 0b)

| Field | Default | Notes |
|-------|---------|-------|
| `seed_prefix_max_nanochat_tokens` | `512` | Upper bound on each SeedRow's prefix length. Variable in practice: the actual prefix is `doc_tokens[prefix_end - max_prefix : prefix_end]` where `prefix_end = len(doc_tokens) - seed_continuation_nanochat_tokens`. Defaulted up from the original spec value of 64 (then 256) so base models have more context to latch onto a concept before being asked to continue. Must be >= `seed_continuation_nanochat_tokens`. |
| `seed_continuation_nanochat_tokens` | `256` | How many nanochat tokens of each source document are stored as the natural continuation on every SeedRow. The last M tokens of the document. Stage 1b uses it as the paired concept-avoiding completion after strict QC. Every SeedRow is guaranteed to have exactly this many continuation tokens. |
| `seed_pool_train_ratio` | `0.75` | Train/eval split ratio. `0.75` means 75% of surviving rows go to `seeds_train.safetensors`, 25% to `seeds_eval.safetensors`. Must be strictly between 0 and 1. |
| `seed_pool_seed` | `42` | RNG seed for the shuffle before splitting. |
| `seed_extraction_candidate_shard_size` | `1024` | Number of deterministic candidate rows per resumable Stage 0b extraction shard. Positive and part of seed-artifact identity. |
| `seed_extraction_gpu_batch_size` | `16` | Production `process_shard` GPU batch size. Positive and part of artifact identity because grouping can be numerically observable. |
| `seed_extraction_cpu_batch_size` | `64` | Production `process_shard` CPU staging batch size. Positive and part of artifact identity. |

Stage 0b raw-encodes each source document with the nanochat tokenizer and **no
BOS**. For raw content length `N` and continuation length `M`, a document is
eligible iff `N >= 2*M`; the continuation is `tokens[N-M:N]`,
`prefix_end = N-M`, and
`prefix_start = max(0, prefix_end-seed_prefix_max_nanochat_tokens)`. The
authoritative stored token lists are those exact source slices. Decoding is
used once to obtain the corresponding text, never as an intermediate step for
reconstructing tokens.

The prefix is then tokenized independently by the extraction model exactly as
production feature extraction does. Empty, over-context, or entirely
unpoolable prefixes are skipped before candidate indices are assigned. Stage
0b invokes the ordinary batched `process_shard` extractor with the layer,
activation, pooling, context-independence, BOS masking, hook, L0 tier, and
dataset semantics parsed from `feature_dir`; it does not read or slice the
validation document's pre-existing feature shard. Per-segment activations are
merged by maximum magnitude for each feature ID and sorted by ID.

Stored seeds contain BOS-free source content. A single pipeline-owned
nanochat BOS is prepended only when constructing a target-model input. A
payload token whose numeric ID equals the BOS ID remains payload; ownership is
defined by construction and position, not token-value deduplication.

### Negative prefixes (Stage 0c)

| Field | Default | Notes |
|-------|---------|-------|
| `n_verified_prefixes_per_concept_target` | `30` | Required number of verified negative-mode prefixes for a candidate concept to be accepted. Stage 0c rejects candidates with fewer and keeps pulling from `candidate_concepts.json` until `max_concepts` accepted concepts are found. |
| `n_verification_seeds` | `5` | How many verification continuations to sample per candidate prefix. Each uses a distinct generation seed. |
| `verification_percentile_threshold` | `0.7` | Per-seed percentile fraction the concept feature must exceed for that seed to "pass" (70 raw percentile points by default). Must be finite and in `[0,1]`; it is converted to raw percentile points before comparison. Stage 0c deliberately retains its strict `>` comparison. |
| `min_consistency_passes` | `3` | Minimum number of verification seeds (out of `n_verification_seeds`) that must pass for a candidate to be accepted. Default 3/5 gives a reliability floor of 60%. |
| `verification_length` | `64` | Number of new tokens to sample per verification continuation. |
| `verification_api_model` | `"inclusionai/ling-2.6-flash"` | OpenRouter model for Stage 0c verification continuations. Stage 0c independently derives an operational concept definition from validated raw Neuronpedia activation examples (never the published candidate description). Continuation calls repeat that validated definition as quoted data in a dedicated system instruction and pair it with a contract-first user prompt; definition inference remains user-only. A trusted local source bridge repairs arbitrary segment cuts; the API authenticates only a 96-character suffix and supplies post-bridge text inside a nonce envelope. The parser validates the whole reconstructed response and exactly `verification_length` extraction tokens before the unchanged local SAE rejection gate. Uses `synth.max_concurrent`, `synth.requests_per_minute`, `synth.transport_max_retries`, and the content budget `max_retries`; requires both API keys. Set `null` for the exact legacy schema-8 local-generation path; API artifacts use schema 9. |
| `verification_api_prefix_wave_size` | `100` | Maximum candidate prefixes proposed together in API mode. The wave is capped by the candidate-attempt limit, candidate pool, and zero-hit bailout, but intentionally may exceed the remaining accepted-prefix target to keep provider workers occupied at low hit rates. Every proposal is locally SAE-scored; only the first target passes in candidate order are retained. Each prefix expands to `n_verification_seeds` calls, while `synth.max_concurrent` bounds simultaneous requests. Conditional API fingerprint input and ignored by explicit-null local mode. |
| `min_prefix_chars` | `100` | Shared decoded-text lower bound for Stage 0c candidates and Stage 1b positive seed-prefix views. Values below it are ineligible. If the entire train pool is ineligible, Stage 1b fails before percentile-table construction, any Stage 1b SAE forward, or synth-cache/API access (the CLI dispatcher has already entered the shared base-model/SAE contexts). |
| `max_prefix_chars` | `2048` | Shared decoded-text upper bound for Stage 0c candidates and Stage 1b positive seed-prefix views. Both retain the most recent Python characters (Unicode code points), not the document head; the bound is not bytes or tokenizer tokens. Stage 0c also drops any Gemma-tokenized prefix that would leave fewer than `verification_length` tokens of room in the base SAE LM context window. Stage 1b never rewrites the Stage 0b seed artifact. |
| `verification_zero_hit_bail_after` | `2000` | Per-candidate-concept short-circuit. Abort verification for a candidate when it has produced zero verified prefixes after this many candidate-prefix attempts (treated as a normal `not_enough_verified_prefixes` rejection). Set `null` to disable and scan the full candidate pool. If `verification_candidate_attempt_limit` is smaller, that limit takes effect first. |
| `verification_cross_prefix_batch_size` | `4` | How many candidate prefixes the explicit-null local Gemma path batches inside one verification generation chunk. `1` processes one candidate at a time. Values above `1` keep candidate order and seed formulas unchanged while increasing local GPU KV-cache/padding pressure. Included in the exact schema-8 local fingerprint and dormant in API mode. |
| `verification_prefill_mode` | `"shared"` | Prefill algorithm for each verification chunk. `shared` prefills each candidate prefix once and expands its public Transformers cache into the per-seed decode rows. `repeated` explicitly selects the pre-schema-8 reference algorithm that prefills every repeated seed row. Modes are separately deterministic but may produce different tokens because prefill batch shape is semantically observable. Included in the Stage 0c fingerprint. Requires Transformers 5.6.2 or newer. |
| `stage0c_multi_gpu_block_size` | `16` | Configured per-rank candidate block between DDP rendezvous barriers. Under multi-GPU it is adaptively capped at `max(1, min(stage0c_multi_gpu_block_size, ceil(max_concepts * 2 / world_size)))`. Single-GPU runs use it verbatim (clamped `>= 1`). |

When API generation is enabled, the published description still remains part
of the `Concept` used by Stage 1 and the judges, but it is structurally absent
from the Stage 0c API generator's inputs. That generator receives only the
feature ID, independently fetches raw activation evidence, and derives a
separate operational definition. Both the evidence and LLM responses are
cached immediately. Prefixes are untrusted data and remain byte-identical in
local memory. A bounded exact local source bridge repairs the arbitrary segment
cut; the provider repeats only a 96-character suffix to authenticate context,
and the parser prepends the bridge from local bytes. The default 64-token
request asks for 90-130 words without a model-added paragraph break, while
larger token targets scale that range. A hard literal-applicability gate makes
local coherence authoritative: when the source's existing final-sentence
entity or action cannot directly bear the property, a realistic unsteered
continuation is the correct probe and the unchanged SAE gate rejects it later.
The model may not manufacture applicability by introducing a new actor,
analysis, goal, parameter regime, analogy, scene, or domain. Outside-envelope chatter
is ignored; inside-envelope preambles, sign-offs, malformed framing,
full-prefix/substantial-suffix echoes, substantial normalized source-phrase
echoes anywhere in the response, protocol text, controls, foreign-script
leakage, newly fabricated or densely mutated Unicode letter forms at Latin
boundaries, phrase loops, unnatural recurring-peak use, repeated definition
phrases, and special tokens are rejected and retried. Clean short pieces are
retained, concatenated, and encoded once before exact truncation. The requested
response ends with one ASCII-space separator before the authenticated end
marker. The parser removes terminal ASCII framing whitespace even when it is
missing or repeated, accepts the observed 95-of-96 left-truncated echo, and
normalizes a nonce end marker missing its final `]`; none of those presentation
bytes enter the scored text. Duplicate trailing end-marker chatter is also
discarded after the first authenticated boundary. Missing nonces, duplicate begin markers, and
substantive suffix corruption still fail closed. Each later attempt receives a
fixed correction for the observed failure class; raw rejected output is never
reflected into a prompt. A single
separable leading sentence that copies the definition may be removed; only the
fully revalidated remainder can then enter the scored window. No parser can
prove that arbitrary
natural-language prose is semantically free of every possible form of
"garbage"; the hard contract is exact framing plus explicit syntactic and
known-boilerplate rejection, followed by the local SAE acceptance test.

In Stage 0c API mode, a multi-member target does not receive one synthetic
"cluster description" and does not use any published Neuronpedia description.
The generator independently fetches and validates each mask-allowed member's
raw top-activation evidence, infers one evidence-grounded operational
description per member, and cycles those descriptions over the canonical
flattened `(candidate, verification seed)` proposal stream. The round-robin
offset is global across API waves. The cycle-policy version enters the
conditional Stage 0c fingerprint and resume identity. Per request, the member
ID and evidence identity enter the authenticated envelope and prompt-derived
cache key; the cache tag additionally identifies the global candidate/seed
position, and request attrs record the member ID and global proposal index.
Those per-request identities are not duplicated inside the conditional
API-generation fingerprint subobject. Member assignment is nevertheless
resume-pinned upstream: `representative_to_allowed_members` is part of the
feature-contract semantic payload, and its `feature_contract_fingerprint` is
folded into the full Stage 0c fingerprint. Changing a representative's resolved
members under the same `run_tag` therefore changes the Stage 0c fingerprint and
makes resume fail loudly before generation performs any cache lookup. The
cache-keyed but not resume-pinned surface is the external evidence identity: on
a cold or missing evidence-cache entry, Neuronpedia top-activation evidence can
drift under an unchanged feature contract and can change the inferred member
definition; regardless of whether the resulting definition text happens to
match, the new evidence identity changes the authenticated envelope and
prompt-derived cache key without changing the Stage 0c fingerprint. Every
returned proposal is still accepted only by the cluster-wide local max-pooled
SAE oracle.
This intentionally preserves the distinct behaviors represented by the real
members instead of asking another model to blend them into a single label.

### Positive-mode synthesis (Stage 1b)

| Field | Default | Notes |
|-------|---------|-------|
| `n_positives_per_concept` | `72` | Target number of complete, accepted rich/avoiding pairs per concept. It is not a fixed seed roster: an exhausted seed is discarded atomically and replaced from the deterministic candidate stream. Candidate scoring always starts with the target-72 compatibility stream. A schema-v2 partial may resume at the same or a higher target under the same explicit `run_tag`; decreases and completed-corpus extension are rejected. |
| `output_length` | `64` | Target length of each generated continuation, counted in **extraction-tokenizer tokens** (Gemma tokens, not nanochat tokens). Controls how much the LLM has to generate and how much of the response we keep after re-tokenization. |
| `accept_threshold` | `0.3` | Positive-mode QC fraction. A continuation is accepted iff its raw, unquantized percentile-point delta above the exact prefix-alone baseline is inclusively at least `D = 100 * accept_threshold`; the default therefore requires 30 points. |
| `positive_prefix_headroom_ceiling` | `0.9` | Fractional completion ceiling used during seed admission. Let `O = min(100 * positive_prefix_headroom_ceiling, 99.5)` and `S = O - D`; only exact prefix-alone scores `<= S` are eligible. Defaults give `O=90`, `S=60`. The `99.5` cap is the maximum representable raw firing percentile. |
| `qc_batch_size` | `16` | Maximum ragged-batch size for Stage 0c/pre-provider prefix QC and Stage 1b/1c post-provider continuation QC. Stage 1b records both this prefix-scoring value and the effective post-provider cap `min(qc_batch_size, max(1, synth.max_concurrent))`. Lower it before a fresh transaction to reduce GPU-memory pressure; changing it or the effective post-provider cap cannot resume an authoritative partial, and there is no automatic OOM retry with a different shape. |
| `positive_max_attempts_per_seed` | `8` | Total semantic attempts per Stage 1b seed, applied independently to rich generation and, only when natural avoiding QC fails, LLM-avoiding generation. Natural avoiding QC uses zero API attempts. Protocol/parser/special-token/QC rejection is charged; attempts resample immediately with unchanged attempt-specific prompts and cache keys. |
| `positive_zero_hit_bail_after` | `2000` | Maximum distinct candidates admitted while the concept still has zero complete pairs; `null` disables it. Already outstanding candidates finish, and any success lifts the guard and refills accumulated deficits. |
| `max_retries` | `100` | Additional content proposals after the first for Stage 0c and Stage 1c only. Stage 1b does not consult this compatibility field. |
| `synth.transport_max_retries` | `3` | Additional HTTP attempts after the first for each OpenRouter proposal. Retryable transport failures retain exponential `2**attempt` backoff (1s, 2s, 4s), so one proposal makes at most 4 HTTP requests. This budget is independent of `max_retries`. It is resume-pinned in Stage 0c's API-generation fingerprint because changing it can determine whether a logical proposal succeeds or the content loop advances to a different attempt-specific prompt. |
| `qc_fallback_on_exhaust` | `"error"` | Stage 1b consults fallback only when pool exhaustion or the zero-hit guard makes the target unreachable. `"error"` aborts before publication; `"warn"` and `"best"` retain only complete accepted pairs. Per-seed exhaustion always means discard-and-replace. Stage 1c retains its existing per-prefix behavior. |
| `max_concept_drop_fraction` | `0.5` | Exclude one concept when final `target - accepted_pairs` divided by the target is strictly greater than this threshold. Exactly `0.5` survives. Discarded-and-replaced seeds never count as drops. |
| `max_dropped_concept_fraction` | `0.5` | Fail the Stage 1 stage when its excluded-concept fraction is strictly greater than this threshold. A zero-survivor population always fails. The diagnostic yield report is written with failure status before raising. |

Stage 1b resolves one immutable, index-aligned view of `seeds_train` before
building the percentile table or opening the synth cache. Every eligible view
is the exact tail of `SeedRow.prefix_text` within the shared character bounds;
short entries become ineligible without renumbering the remaining rows.
Original indices still drive shuffle order, cache tags, and `gen_seed`
provenance. The historical Stage 1b formula is intentionally unchanged, so
replacement candidates with indices at or above 1000 can share a `gen_seed`
with another concept; consumers must use the explicit concept/content fields,
not `gen_seed`, as row identity. The same bounded string is used by
dirty-prefix scoring, rich and avoiding prompts/parsers/QC, emitted rows, and
the exact nanochat output boundary. `SeedRow.prefix_text`, its stored token
slice, the Stage 0b fingerprint, and Stage 4 eval seeds remain unchanged.

Stage 1b uses one complete-pair request-chain engine for
`synth.max_concurrent=1` and larger values. Initial targets for every concept
are queued before any replacement. A chain owns one seed through rich retries,
natural avoiding QC, and any LLM-avoiding retries. Same-seed work keeps its
worker slot; terminal replacements receive monotonic chain IDs and enter the
global queue tail, preserving breadth-fair initial scheduling. Stage 1c uses
the chain scheduler without a refill callback and retains its fixed verified
prefixes. Provider completion is the only worker-thread operation:
parsing, Gemma/SAE QC, retry decisions, fallback accounting, and row assembly
stay on the caller thread. Canonical assembly preserves row order, `gen_seed`,
cache tags, and pair filtering independently of response completion order.
With deterministic scoring, published bytes are independent as well. When both
`synth.max_concurrent` and `qc_batch_size` exceed one, the caller selects the
lowest active stable chain indices, waits for that bounded frontier, and reuses
the production ragged extractor: Stage 1b performs at most one rich-percentile
pass plus one combined natural/LLM strict-zero pass per callback, while Stage
1c performs one strict-zero pass. Higher-index provider requests continue in
worker threads while the frontier waits. An effective cap of one retains the
scheduler's scalar callback contract but delegates to the same shared
singleton-shaped QC/application state machine.

BF16 activation results may differ slightly between singleton and batched
shapes, so threshold-adjacent continuations can receive different QC decisions
when the configured effective cap changes. Provider completion timing cannot
change batch membership. The effective cap and stable-frontier algorithm are
recorded in the positive-selection artifact contract and current checkpoint
fingerprint. A pre-batching schema-v2 checkpoint undergoes an exact one-time
in-memory fingerprint migration: accepted singleton-scored rows remain
unchanged, outstanding work adopts the configured current shape, and the next
commit writes the protected current fingerprint.

Candidate scoring is independent of the requested target. Each concept first
scores toward 72 admissible candidates with the literal historical v3 batching
path, including rejection-driven scorer widths and clean-to-dirty behavior. If
the physical pool ends before 72, a smaller requested target may still proceed
when enough candidates were buffered. Targets above 72 continue through full
`qc_batch_size` clean chunks; only clean-pool exhaustion may shorten a chunk,
after which the dirty pool is scored once. Surplus candidates stay buffered.
Consequently target 72 retains the exact current scorer calls and
`positive_target_refill_post_provider_qc_v5` parquet metadata. Other targets
use `positive_target_compatibility_post_provider_qc_v6`, which records
compatibility target 72, the post-compatibility full-chunk rule, and the
effective post-provider QC batch identity. The final row schema is unchanged.

Stage 1b durably records scheduler progress in `stage1b_partial.parquet` and
the authoritative commit marker `stage1b_partial.json`. The parquet contains
only adjacent, complete rich/avoiding pairs in monotonic acceptance order;
discarded, buffered, and in-flight candidates live only as compact JSON
descriptors. A checkpoint is forced after initial admissions, every 64
terminal chains, on every caught failure (including `KeyboardInterrupt`), and
once more before final publication. In-flight chains roll back to rich attempt
zero on resume. Responses already committed to `synth_response_cache.sqlite`
are replayed without a provider call; a request interrupted before its cache
commit may be sent again. Persisted candidates are restored without prefix
rescoring, and final rows are still sorted by concept and candidate rank.

Checkpoint writes replace parquet first and JSON second. JSON therefore
defines the committed pair prefix: an even parquet suffix newer than JSON is
ignored and overwritten later, a parquet-only orphan is ignored, and JSON
without all of its committed parquet rows fails as data loss. Schema-v2 JSON
records `highest_requested_target`. Its fingerprint binds the target-independent
compatibility stream and all other synthesis-semantic inputs, but excludes the
requested target plus rate, concurrency, transport-retry, profiling,
output-root, and run-tag settings.
It also binds `qc_fallback_on_exhaust`, `max_concept_drop_fraction`, and
`max_dropped_concept_fraction`, because those gates decide whether the
checkpointed history may publish and which concept population survives.
Short multi-field scheduler commits defer `SIGINT`; parsing, QC, provider
work, and checkpoint I/O remain interruptible. A caught failure preserves all
terminal chains; `SIGKILL`, host loss, or power loss can lose at most 64
terminal outcomes since the previous periodic checkpoint (their cached
responses can still replay). After the final `"ok"` yield report is committed,
cleanup deletes JSON first and parquet second. A cleanup warning reports which
files remain: JSON is still authoritative if present, while parquet alone is
an ignored orphan.

Resume accepts the same target or a higher one. Growth restores the exact
buffer/cursors, enlarges tracker targets, and scores any unscored full chunks
needed to prove the enlarged target is physically achievable. Only then does
it admit deterministic deficits and commit the increased target before
constructing provider requests. An infeasible target or failed growth
checkpoint write leaves the previous JSON authoritative. A target
decrease fails before prefix-cache or provider construction. Schema-v1 Phase 1
checkpoints migrate only when their stored target is exactly 72 and every
remaining fingerprint field matches the target-72 Phase 1 identity; non-72 v1
files used target-shaped BF16 prefix-scoring calls and fail with
finish-or-restart guidance. Migration is in memory and the old
JSON is not replaced until the next successful parquet-first/JSON-second write.
Schema-v2 v2/v3 checkpoints likewise receive one exact legacy-row validation
and normalize to the current v4 fingerprint in memory. Unsupported or malformed
algorithm values fail before that potentially expensive scan. Loading alone is
read-only: if the process exits before the scheduler's existing initial commit,
the legacy JSON remains authoritative and the scan may repeat on the next run.
Growth requires an authoritative `stage1b_partial.json` and reuse of the same
explicit `run_tag`. A completed positive parquet without that JSON cannot be
extended in place; use a fresh `run_tag`. A remaining authoritative partial
may resume the same target if final publication files also exist after
interrupted cleanup, but it cannot authorize growing already-committed final
evidence.

Every Stage 1b/1c LLM attempt has collision-checked nonce markers derived from
its complete attempt identity. The prompt ends at the exact local prefix
suffix (at most 96 characters), which the response must echo at the
authenticated boundary. Only the fixed Stage 0c separator set and a
one-character left-truncated echo are normalized. After removing that echo,
leading continuation characters remain exact data. Before any SAE QC, the
complete response is checked for residual prefix overlap, controls,
protocol-marker pollution, special-token IDs, source copies, and phrase loops;
after extraction-token truncation, the decoded stored window is checked again.
Every accepted presentation normalization is observable under
`stage1b.continuation_protocol.*` or `stage1c.continuation_protocol.*` with the
concept, attempt, and content class attached; rejection counters remain the
separate `*.continuation_protocol_failures` totals.
A repeated normalized word sequence is rejected when a
non-overlapping phrase of at least 8 words and 48 characters occurs twice.
Source copying is rejected for a shared normalized phrase of at least 7 words
and 32 characters, with an independent 32-alphanumeric-character NFKC fallback
for scripts without whitespace boundaries and punctuation-heavy variants.
Repeated-phrase diagnostics take precedence when both rules match. Content
failures consume the semantic attempt, receive a fresh attempt-specific cache
key, increment
`stage1b.continuation_protocol_failures`, and can never become a `"best"`
fallback row. A provider call that exhausts its internal HTTP attempts instead
aborts Stage 1b immediately: it does not advance the semantic attempt, discard
the seed, or draw a replacement. This applies to `concept_rich` and LLM-generated
`concept_avoiding` rows. Trusted Stage 0b natural continuations are not
LLM-generated and remain governed only by strict-zero SAE QC.

Stage 1b writes a versioned `axbench_stage1b_positive_qc_contract` metadata
record into `synth_training_positive.parquet`. It pins the fractional threshold,
its raw-point equivalent, the inclusive comparison, and the unquantized
percentile currency (`0.0` for strict zero, otherwise `[0.5,99.5]`). The
producer checks every row in full Python precision before Arrow converts the
existing QC columns to float32. Base, BPB, and BPB-Adv consumers reject
positive corpora that lack or contradict this contract. A separate versioned
`axbench_stage1b_positive_selection_contract` pins the pair target, attempt
cap, zero-hit guard, canonical candidate ordering, and lazy atomic replacement
semantics. Target 72 uses
`positive_target_refill_post_provider_qc_v5`; other targets use
`positive_target_compatibility_post_provider_qc_v6` with the
compatibility-stream and post-provider batch-identity fields. The requested
target is intentionally absent from current resumable checkpoint identity; the
effective post-provider cap remains bound until scoring is proven
shape-invariant.
Completed pre-batching v3/v4 corpora (and older contracts) must be regenerated,
along with downstream artifacts, under a fresh auto-generated `run_tag`. An interrupted explicit-tag
run with no committed Stage 1b corpus may reuse that tag and its durable
response cache because prompts and cache-key namespaces are unchanged.
Both positive and negative corpora additionally carry
`axbench_stage1_continuation_contract`, with roles `stage1b_positive` and
`stage1c_negative`. It pins the authenticated framing, echo normalization,
full-response guard, token limit, and stored-window revalidation semantics.
Base, BPB, and BPB-Adv reject missing, malformed, stale, or role-swapped
records rather than reinterpreting old rows.

### LLM client configs

Synth and judge clients share the `LLMClientConfig` base, but `SynthConfig`
overrides the model and both throughput defaults (`config.py`):

| Field | Default | Notes |
|-------|---------|-------|
| `provider` | `"openrouter"` | Only `"openrouter"` is wired today. |
| `model` | synth `"inclusionai/ling-2.6-flash"`; judge `"openai/gpt-5-nano"` | OpenRouter model identifier. `cfg.synth.model` controls Stage 1b/1c; `cfg.judge.model` and the Stage 0c concept-examples QC judge remain independent. A non-null `verification_api_model` explicitly overrides `cfg.synth.model` only for Stage 0c continuation/definition calls. |
| `requests_per_minute` | synth `6000`; judge `60` | Per-process sliding-window admission limit. Non-positive values disable client-side throttling. Stage 0c/1 synth traffic and judge traffic are configured independently. |
| `max_concurrent` | synth `100`; judge `10` | Maximum concurrent OpenRouter calls. API-mode Stage 0c flattens independent prefix/seed calls within its API-specific wave. Stage 1b uses one stage-wide complete-pair request-chain pool at every concurrency value; Stage 1c uses its existing stage-wide fixed-prefix pool above one and serial path at one. Retries within one logical item remain sequential and lazy. Stage 5 uses the judge value. A full default Stage 0c wave is 100 prefixes × 5 seeds = 500 logical calls, with at most 100 simultaneously in flight. Active Stage 0c API generation requires this to be at least 1. |

`JudgeConfig` (`config.py:36-42`) adds three more fields:

| Field | Default | Notes |
|-------|---------|-------|
| `parse_max_retries` | `3` | Stage 5 judge parse-retry budget. 4 total attempts absorbs the ~17% bare-`Rating: N` rate observed against openai/gpt-5-nano plus a safety margin. Was `1` before April 2026; bumped after a 9.6-hour production run surfaced 301/301 parse failures on the concept-examples QC path. |
| `invalid_warning_threshold` | `0.01` | Stage 6 aggregation warns if the fraction of judge-invalid rows per method+mode exceeds this. |
| `invalid_drop_threshold` | `0.05` | Stage 6 aggregation drops a method from the leaderboard entirely if its invalid-rate exceeds this. |

### Concept-feature mapping (Stage 2)

| Field | Default | Notes |
|-------|---------|-------|
| `concept_feature_mode` | `"canonical"` | `"canonical"` writes a trivial mapping where slot 0 holds the Neuronpedia-derived feature index and every other slot is `-1`. `"delta_ranked"` runs Gemma + SAE over the synthesis corpus to rank features by the percentile delta between concept-rich and concept-avoiding rows. Locked-in decision #5. |
| `n_rank_store` | `256` | Width of the ranked slots per concept. In `canonical` mode only slot 0 is populated; in `delta_ranked` mode the top-`n_rank_store` features and their delta percentiles are stored. |
| `run_negative_mode` | `True` | If False, negative-mode stages are skipped and `concept_features_neg.safetensors` is not written. |

### Generation / Layer hooks / LoRA (Stages 3-4)

| Field | Default | Notes |
|-------|---------|-------|
| `generation.max_new_tokens` | `64` | Length of Stage 4 completions. |
| `generation.eval_seeds_per_concept_positive` | `10` | Number of positive-mode eval seeds drawn per concept from `seeds_eval.safetensors` during Stage 4. |
| `generation.eval_seeds_per_concept_negative` | `10` | Same, for negative-mode eval seeds. |
| `generation.master_seed` | `42` | Root seed for all RNG draws in Stages 0c / 1b / 1c / 4. Threaded through every seed offset in `seeds.py`. |
| `generation.temperature` | `0.7` | Sampling temperature for null/local Stage 0c base-SAE-LM verification and Stage 4 generation. It does not control the external provider when `verification_api_model` is set. |
| `generation.top_k_sampling` | `50` | Top-k sampling cutoff for null/local Stage 0c and Stage 4; it does not configure the external API model. |
| `layer_hooks.diffmean_and_probe_layer` | `6` | Target nanochat/encoder-decoder residual-stream layer for DiffMean / LinearProbe / LoRA (Stages 3a/3b/3d). This is not the Gemma SAE extraction layer. DiffMean/probe consumers require the artifact's stamped training layer to equal it. |
| `lora.rank` | `8` | LoRA adapter rank. |
| `lora.alpha` | `16.0` | LoRA scaling coefficient. |
| `lora.dropout` | `0.0` | LoRA adapter dropout. |
| `lora.lr` | `1e-4` | LoRA training learning rate. |
| `lora.steps` | `100` | LoRA training step count per concept. |
| `lora.batch_size` | `8` | LoRA training mini-batch size. |
| `lora.min_training_rows` | `8` | Per-concept minimum training pool. Stage 3d raises for concepts below this threshold. |

DiffMean and linear-probe files have one shared identity contract across
AxBench-Base, plain BPB, and BPB-Adv. Each consumed file must contain canonical
`concept_id`, `mode`, `target_model`, and non-negative decimal `layer` metadata;
its `<family>/<target>/<mode>/concept_NNNN.safetensors` path must agree with
those coordinates. The file must expose exactly one non-empty, finite,
one-dimensional floating `direction` tensor. Cluster-stamped artifacts also
require the expected `artifact_kind`; a legacy unclustered artifact may omit
that key, but a present value must be correct. Stage 4 validates every selected
direction before opening its output writer or generating a token, and validates
again at the direct loader boundary. A layer-10 artifact therefore cannot be
silently installed at configured layer 6. Existing artifacts need no
retraining when they satisfy this contract.

### Configuration contracts

`AxBenchBaseConfig` and its validated nested dataclasses enforce unary value
domains at construction with reusable phantom predicates and `beartype`. This
means invalid Hydra values fail while `cfg_to_dataclass` is building the
configuration, before `validate_config` or any pipeline work. The loader wraps
those constructor violations as `AxBenchBaseError` with the affected field;
direct Python construction raises `BeartypeCallHintParamViolation`.

Constructor-enforced domains include:

- `concept_feature_mode in {"canonical", "delta_ranked"}`
- `qc_fallback_on_exhaust in {"error", "warn", "best"}`
- `validation_shards` is a non-empty list of unique, strict non-negative
  integers
- `n_positives_per_concept` is a positive integer (booleans and
  integer-valued floats are rejected)
- `positive_max_attempts_per_seed` is a positive integer, and
  `positive_zero_hit_bail_after` is `null` or a positive integer (booleans and
  integer-valued floats are rejected)
- `min_prefix_chars` is a strict non-negative integer and `max_prefix_chars` is
  a strict positive integer
- `accept_threshold` and `positive_prefix_headroom_ceiling` are finite,
  non-boolean numbers in `(0, 1]`
- `0 < seed_pool_train_ratio < 1`
- `max_retries >= 0` and `synth.transport_max_retries >= 0`; both must be
  strict integers (booleans are rejected)
- `max_concept_drop_fraction` and `max_dropped_concept_fraction` are finite,
  non-boolean numbers in `[0, 1)`
- `layer_hooks.diffmean_and_probe_layer >= 0`
- `max_not_published_description_rate`, `max_missing_description_rate`, and
  `max_failed_description_rate` each in `[0, 1]`; `min_source_mismatch_features >= 0`
- profiling poll intervals are positive, profiling idle thresholds are finite
  and non-negative, and configured output paths are non-blank

`validate_config` retains the relationships and mode-dependent contracts that
cannot be expressed by a single field type:

- `min_prefix_chars <= max_prefix_chars`
- `0 < accept_threshold < positive_prefix_headroom_ceiling <= 1`; the derived
  headroom band must be representable in the raw `[0, 99.5]` percentile
  currency
- `seed_prefix_max_nanochat_tokens >= seed_continuation_nanochat_tokens`
- API-verification provider and capacity requirements when
  `verification_api_model` is set
- clustered-conditioning stage and method compatibility
- **model/source-identity guards (issue #657):** the `extraction_tokenizer_id`
  must be the extraction model's own gemma-scope-2 tokenizer; the SAE model
  encoded in `feature_dir` (`m_{model}`) must match the extraction model
  (`feature_extraction_model_id or neuronpedia_model_id`); `neuronpedia_source`'s
  layer + activation + width must match `feature_dir`'s `l{layer}` / `a_{act}` /
  `f{width}`; and `neuronpedia_model_id` must equal the canonical Neuronpedia
  model derived from the SAE (`resolve_neuronpedia_ids`). A wrong
  layer/width/model source returns valid 200s and evades every runtime guard.
  See the mass-404 discussion under **Feature-directory / SAE** above for why.
- **bare model ids:** `feature_extraction_model_id` and `neuronpedia_model_id`
  must be **bare** (e.g. `gemma-3-4b-pt`), not org-/path-prefixed — the SAE
  resolver and the Neuronpedia API are keyed by the bare name, so an
  org-prefixed id would pass the bare-compared identity checks yet crash at SAE
  load / mislabel at query time. (`extraction_tokenizer_id` may stay
  org-prefixed; it is a full HF tokenizer id.)

`validate_config(cfg)` runs automatically on every CLI invocation inside
`_run` before the `RuntimeState` is constructed. Explicit Python callers get
unary validation from construction but must still call `validate_config`
before handing the config to any stage function so the relational contracts
run too.

## Outputs and layout

Every run artifact is resolved by the immutable `AxBenchBasePaths` contract.
`output_root=null` uses
`get_base_dir() / "eval" / "axbench_base"`; a non-empty configured root is
used literally. The run root is
`{effective_output_root}/{sanitized_run_tag}`. The sanitizer bounces any
path-traversal-looking
tag to a `traversal_<sha256prefix>` fallback and replaces unsafe
characters with underscores.

The Neuronpedia description cache is cross-run but not cross-identity. It is
stored beside run directories at
`{effective_output_root}/_neuronpedia_cache/v1/<identity_sha256>.yaml`, retains
embedded model/source validation, merges disjoint concurrent writers under an
OS file lock, and rejects conflicting descriptions for the same feature ID.
Pending fetches become visible in memory only after the temporary file is
flushed and fsynced, atomically replaces the target, and the parent directory
is fsynced; malformed feature IDs or descriptions are rejected rather than
coerced.
The former unkeyed `neuronpedia_cache.yaml` is not adopted automatically
because its filename does not prove model/source identity. Cache-only runs
therefore fail fast when the exact new path is absent or contains no validated
descriptions; copy a historical cache only after validating its embedded
`model_id` and `sae_source`.

Under DDP, rank 0 supplies any autogenerated run tag and all ranks
collectively verify the configured and effective output root before artifact
discovery or writes. Rank-dependent Hydra interpolation is rejected rather
than allowing collective stages to split across directories.

Full layout produced by a `stage=all` run:

```
{effective_output_root}/
|-- _neuronpedia_cache/
|   `-- v1/{identity_sha256}.yaml           # Stage 0a, shared by matching identities
`-- {run_tag}/
    |-- candidate_concepts.json                # Stage 0a (pre-QC filtered + shuffled pool)
    |-- concepts.json                          # Stage 0c (final verified concepts)
    |-- rejected_concepts.json                 # Stage 0c (LLM QC + verification failures)
    |-- concept_examples_qc_report.json        # Stage 0c (when examples QC enabled)
    |-- synth_response_cache.sqlite            # Stages 0c/1b/1c (LLM response cache)
    |-- seeds_train.safetensors                # Stage 0b (CSR-packed SeedRows)
    |-- seeds_eval.safetensors                 # Stage 0b
    |-- seed_pool_manifest.json                # Stage 0b sole commit marker
    |-- _seed_extraction_v2/                   # Stage 0b resumable candidates/features
    |-- negative_mode_prefixes.parquet         # Stage 0c (verified prefixes + continuations)
    |-- stage0c_partial_rank_{rank}.json       # Stage 0c rank-local progress (auto-deleted on success)
    |-- stage0c_partial_rank_{rank}.parquet    # Stage 0c rank-local rows (auto-deleted on success)
    |-- synth_training_positive.parquet        # Stage 1b paired rows + three semantic contracts
    |-- stage1b_partial.parquet                 # Stage 1b complete-pair checkpoint rows (temporary)
    |-- stage1b_partial.json                    # Stage 1b checkpoint commit marker (temporary)
    |-- stage1b_concept_yield.json              # Stage 1b counts + population guard
    |-- concepts_stage1b.json                   # Stage 1b ordered filtered population
    |-- synth_training_negative.parquet        # Stage 1c 19 columns + continuation contract
    |-- stage1c_concept_yield.json              # Stage 1c counts + population guard
    |-- concepts_stage1c.json                   # Stage 1c final ordered population
    |-- concept_features.safetensors           # Stage 2 (positive-mode mapping)
    |-- concept_features_neg.safetensors       # Stage 2 (negative-mode mapping, if run)
    |-- baselines/                              # sparse original IDs from final Stage 1 list
    |   |-- diffmean/{target}/{mode}/concept_{id:04d}.safetensors     # Stage 3a
    |   |-- linear_probe/{target}/{mode}/concept_{id:04d}.safetensors # Stage 3b
    |   |-- prompting/{mode}_descriptions.json                        # Stage 3c
    |   `-- lora/decoder_only/{mode}/concept_{id:04d}.pt              # Stage 3d
    |-- stage4_contexts.jsonl                  # Stage 3e exact model-ready contexts
    |-- stage4_contexts_extraction/shard_00000.parquet                # Stage 3e extraction input
    |-- stage4_contexts_manifest.json           # Stage 3e commit/identity manifest
    |-- generations/                           # Stage 4 directory dataset
    |   `-- batch_NNNNNN.parquet
    |-- judge_scores/                          # Stage 5 directory dataset
    |   `-- batch_NNNNNN.parquet
    |-- judge_cache.sqlite                     # Stage 5 SQLite cache
    |-- report.yaml                            # Stage 6 summary
    |-- leaderboard.csv                        # Stage 6 leaderboard
    `-- aggregation_report.yaml                # Stage 6 aggregation details
```

All resolvers are methods on `AxBenchBasePaths` in
`steering/eval/axbench_base/paths.py`: concept, seed, Stage 0c/3e, synth,
concept-feature, baseline, generation, judge, cache, and report paths are
derived from the same immutable `output_root` plus `run_tag` value.

### Key file schemas

**`candidate_concepts.json`**: a JSON array of pre-QC `CandidateConcept`
rows emitted by Stage 0a after corpus-count filtering, description
filtering, and deterministic shuffle. It has no `concept_id` and no
judge score; Stage 0c performs examples QC lazily and assigns final
concept IDs.

**`concepts.json`**: a JSON array of final Stage 0c-accepted `Concept`
objects (`types.py:100-118`).
Fields: `concept_id`, `feature_index`, `layer`, `sae_source`,
`description`, `corpus_count`, `description_source`. `concept_id` is
always dense: `0, 1, 2, ...`, enforced by `validate_concepts`. This file is
immutable Stage 0c provenance and is read as the Stage 1b input only.

**`concepts_stage1b.json` / `concepts_stage1c.json`**: exact ordered
subsequences of their parent concept list. Stage 1b writes the first from
`concepts.json`; Stage 1c reads that file and may write a further-filtered
second list. Original `concept_id` values are never renumbered. When negative
mode is enabled, `concepts_stage1c.json` is the single downstream population
authority; otherwise it is `concepts_stage1b.json`. Stages 2-5, baseline
training, the DAG, AxBench-BPB, and BPB-Adv consume that final authority.
Corpus presence is an integrity check, not another population filter.

**`stage1b_concept_yield.json` / `stage1c_concept_yield.json`**: durable
per-concept attempted/accepted/dropped counts, drop fractions, reason counts,
exclusion decisions, original concept/feature IDs, surviving IDs, aggregate
population loss, configured thresholds, and `status`. A failed aggregate guard
leaves a diagnostic report with `status="failed_population_guard"` but no
committed filtered-list/corpus handoff. After a successful guard, the producer
first writes `status="preparing"`, then publishes the filtered concept list and
parquet, and only then atomically replaces the report with `status="ok"` as the
stage commit marker. A crash during either publication therefore leaves an
uncommitted report even when older same-tag artifacts are still present; the
DAG and runtime loaders reject that torn state.

**`rejected_concepts.json`**: Stage 0c rejection records with
`candidate_index`, `feature_index`, `description`, `reason`,
`n_verified_found`, and `llm_judge_score`. `reason` is serialized as
one of the stable `RejectedConceptReason` enum values from `types.py`.

**`concept_examples_qc_report.json`**: a JSON object with aggregate counts
and a per-feature list. See `QCReport` / `QCFeatureScore` in
`concept_examples_qc.py` (`QCFeatureScore` at line 91, `QCReport`
at line 115). Written by Stage 0c for the candidates the lazy LLM judge
actually evaluated.

**`seeds_train.safetensors` / `seeds_eval.safetensors`**: schema-v2
CSR-packed `SeedRow`s. Each row stores `shard_idx`, `doc_idx`, the raw source
token count and prefix slice bounds, decoded prefix/continuation text, exact
BOS-free nanochat token slices, and sorted unique SAE feature IDs with positive
finite max magnitudes. Character offsets and extraction segment coordinates
are deliberately absent. `seed_pool_manifest.json` is the sole commit marker
for the train/eval pair and pins both file hashes, the seed fingerprint,
source-parquet hashes, tokenizer/model/SAE identities and parameter hashes,
parsed extraction settings, batching, skip counts, and row counts. Runtime
loaders validate the manifest, file hashes and metadata, split, config,
tokenizers, tensors, and every row invariant before returning data. Legacy,
torn, or mismatched artifacts fail with a fresh-run-tag instruction; they are
never heuristically repaired.

The validated train load also computes Stage 1b's ordered-row identity directly
from the 16 contiguous tensors before reconstructing Python rows. Current v4
checkpoints therefore start without reserializing the large CSR lists. A v1,
v2, or v3 partial may take one compatibility scan of the reconstructed rows on
its first successful resume; the scheduler's normal initial checkpoint commit
then publishes v4, and subsequent resumes use the fast tensor identity.

Stage 0b candidate and extracted-feature shards are rank-independent. Every
rank streams the same sources and must agree on the canonical candidate digest
and skip counts. Shards run in collective waves, with sidecars binding the
candidate parquet hash, extraction fingerprint, decompressed output digest,
document count, and runtime/backend identity. A committed shard resumes across
world-size changes only when both data and sidecar validate. The final seed
files are written atomically by rank 0, followed by the manifest last.

**`stage4_contexts.jsonl` / `stage4_contexts_extraction/` /
`stage4_contexts_manifest.json`**: Stage 3e enumerates and deduplicates the
exact model-ready prompts needed by enabled encoder-decoder Stage 4 methods.
Non-prompting methods share seed-only contexts; prompting methods enumerate
every eligible `(mode, concept, eval seed)` prompt. The manifest pins the seed
fingerprint, prompt-description hashes, method/config fingerprint, tokenizer,
model sequence length, ordered token keys/counts, and both file hashes.

**`negative_mode_prefixes.parquet`**: 8 columns
(`negative_prefixes.py:78-89`):
`concept_id`, `prefix_text`, `prefix_source_shard`, `prefix_source_doc`,
`prefix_activating_seg_idx`, `verification_continuations` (list),
`verification_seeds` (list), `verification_percentiles` (list). One row
per verified prefix. The parquet schema metadata includes
`extraction_tokenizer_id`, `neuronpedia_model_id`, the effective
`feature_extraction_model_id`, `neuronpedia_source`, and semantic SHA-256
identities for the ordered Stage 0a pool, `concepts.json`,
`rejected_concepts.json`, the conditional concept-examples QC report state,
and the full Stage 0c fingerprint. When examples QC is enabled, the report is
structurally validated and its semantic hash must match; when disabled, the
metadata pins a null-report sentinel and no stale report may remain. Missing metadata
marks pre-scoping artifacts as stale, and mismatched metadata means Stage 0c
cannot be resumed under the current tokenizer, extractor, candidate pool,
sidecars, prefill/API-generation algorithm, or config. Schema-7 checkpoints
and completed artifacts are not migrated to schema 8. The explicit-null local
mode remains schema 8 byte-for-byte; the default non-null model uses schema 9
and fingerprints the model, protocol, prompt hashes,
client limits, retries, and activation-evidence settings. Completed resume
reloads the JSON artifacts and compares
their semantic hashes before returning the existing output. Once
the three canonical Stage 0c outputs exist, every validation failure—including
tokenizer or extractor mismatch, unreadable data, wrong schemas, malformed
sidecars, and row/config incompatibility—fails before model loading. No
completed artifact is automatically overwritten. Use a new `run_tag`, or
explicitly delete the Stage 0 artifacts and regenerate Stage 0a then Stage 0c.
To reproduce the old generation algorithm in a fresh artifact set, pass
`verification_prefill_mode=repeated`; changing only the mode on an existing
checkpoint is rejected.
The effective feature-extraction model is the explicit override when present
and otherwise falls back to `neuronpedia_model_id`, so those two equivalent
configurations share completed artifacts. Direct Stage 1c runs validate the
four tokenizer/extractor metadata fields before consuming the prefix rows; the
candidate-pool comparison is specific to Stage 0c completed resume.

**`synth_training_positive.parquet`**: 16 columns, schema in
`positive_mode.py`. Contains paired per-concept `concept_rich` and
`concept_avoiding` rows with the same `concept_id`. Downstream Stage 2
uses the symmetric per-concept filter from `concept_feature_mapping.py`:
`concept_id=cid & content_class=concept_rich` is the positive side, and
`concept_id=cid & content_class=concept_avoiding` is the negative side.
The parquet schema metadata includes `extraction_tokenizer_id`; Stage 2
rejects missing or mismatched metadata because parsing and QC depend on
the configured extraction tokenizer. Stage 3 baseline trainers apply the
same check when they are run directly against this parquet.
Only complete, QC-accepted rich/avoiding pairs are published. A best-effort
failed-QC rich row and its avoiding partner are both removed. Stage 1c may
later remove additional concepts from the final list, so this parquet can
contain rows outside the final population; downstream consumers filter by the
authoritative concept list and require every listed concept to be present.

Every Stage 1b row stores the bounded effective prefix described above.
Immediately before publication, Stage 1b revalidates the closed character
bounds, non-empty output, exact recomputed nanochat boundary, and the
same-prefix/same-concept adjacency of every emitted `concept_avoiding` row.
This is a publication guard, not a migration: the committed Stage 0b seed
files remain byte-identical.

`n_prefix_nanochat_tokens` is the exact first model-token index overlapping
`output_text`, including the owned BOS. The prefix and output are concatenated
and encoded once, so a BPE token may cross their character boundary. DiffMean,
LinearProbe, and LoRA recompute this boundary with the same helper and require
exact equality; non-roundtripping text fails instead of using an approximate
`len(encode(prefix))` boundary.

**`synth_training_negative.parquet`**: 19 columns, schema in
`negative_mode.py:74-98`. Same 15 base columns plus
`prefix_source_shard`, `prefix_source_doc`, `prefix_source_seg_idx`,
`qc_zero_activation_verified`. Holds concept_rich rows (copied
verbatim from Stage 0c verification continuations) and concept_avoiding
rows (synthesized and strict-zero-QC'd). Degenerate continuations
(empty, or adding no tokens beyond the prefix seam) are rejected and
retried rather than accepted (issue #518; mirrors the positive-mode fix
from issue #513). Empty LLM responses fail the authenticated parser and count
under `stage1c.continuation_protocol_failures`; a non-empty parsed response
that contributes no tokens at the prefix seam counts under
`stage1c.qc_empty_rejects`. It carries the same `extraction_tokenizer_id`
metadata plus the negative-role Stage 1 continuation contract. Stage 2, Stage
3, BPB, and BPB-Adv reject missing or mismatched metadata before use.
Each verified prefix contributes all of its copied rich rows plus exactly one
accepted avoiding row, or contributes nothing. Stage 1c writes the final
filtered population after applying the same per-concept and aggregate guards.

**`generations/batch_NNNNNN.parquet`**: Stage 4 constructs every prompt in
one shared builder. Ordinary methods use `[owned BOS] + seed content`; prompting
methods use `[owned BOS] + raw prompting-prefix tokens + seed content`.
Generation dispatch receives this complete model-ready list and does not
prepend or reorder it. Each output row records `prompt_token_key`,
`prompt_token_count`, and `seed_pool_fingerprint`. Resume requires those exact
values and the visible prefix to match, and every plan is checked against the
target checkpoint's sequence length before the first row is written.

**`concept_features.safetensors` / `concept_features_neg.safetensors`**:
four tensors + metadata
(`concept_feature_mapping.py:22-28`). The row extent is
`max(surviving concept_id) + 1`, not the survivor count, so original IDs remain
direct tensor indices. Filtered-out IDs below that maximum retain all-sentinel
rows: `feature_ids=-1`, `percentile_deltas=NaN`,
`canonical_feature_id=-1`, and `canonical_rank=n_rank_store`. In `canonical`
mode slot 0 holds the surviving concept's feature; in
`delta_ranked` mode the top-`n_rank_store` features are sorted by
`pos_pct - neg_pct`. Delta-ranked files include
`metadata["extraction_tokenizer_id"]` because the ranking text
extraction is tokenizer-dependent. Stage 4 validates that metadata for
`ours` methods; canonical concept-feature files remain tokenizer
agnostic.

## Cost and runtime

Back-of-envelope estimates for the default 500-concept run with a
typical OpenRouter synth model:

| Stage | API calls / GPU work | Wall-clock |
|-------|---------------------|------------|
| 0a `prepare_concept_list` | Neuronpedia description lookups + filtering + deterministic shuffle | seconds to a few minutes, depending on description-cache warmth |
| 0b `prepare_seed_pools` | Raw parquet streaming + batched production Gemma/SAE extraction | GPU-dependent; candidate shards resume after validation |
| 0c `prepare_negative_prefixes` | lazy examples-QC calls + local SAE verification; default generation uses `inclusionai/ling-2.6-flash` with one definition call per visited feature and one parallel API call per candidate/seed/attempt; `verification_api_model=null` uses the base SAE LM | dominated by provider generation/retries and local SAE verification; benchmark the intended feature/model/provider regime rather than assuming the API proposal source is faster |
| 1b `synthesize_positive_mode` | ~500 x 72 = 36k initial rich calls; each admitted seed can use up to 8 rich calls and, only after natural avoiding failure, 8 LLM-avoiding calls; exhausted seeds draw replacements until target or a terminal guard | provider-capacity dependent; the stage-wide complete-pair request-chain pool can occupy the synth client ceiling of 100 concurrent / 6000 admissions per minute per process while enough work remains |
| 1c `synthesize_negative_mode` | ~500 x 30 = 15k concept-avoiding synth calls | provider-capacity dependent under the same synth ceiling |
| 2 `concept_feature_mapping` (canonical) | zero GPU | < 1 min |
| 2 `concept_feature_mapping` (delta_ranked) | Base Gemma + SAE forward over ~72 + ~216 texts per concept = ~150k forwards | 1-3 h GPU |

Stages 0c, 1b, and 1c release Python garbage and cached CUDA allocator
blocks at their long-running iteration boundaries: after each verified
Stage 0c candidate; after each Stage 1b concept-source preflight and the
unified complete-pair pool; after each serial Stage 1c concept and the
parallel Stage 1c pool. The shared
SAE-QC helpers also release cached allocator blocks after each
Gemma+SAE forward, including Stage 1b's dirty-prefix cache-fill chunks.
These are memory-lifetime guards only; they do not change emitted rows,
cache keys, or retry decisions.

Token cost depends on live OpenRouter pricing and retry behavior. Stage 0c and
Stage 1 synth default to `inclusionai/ling-2.6-flash`; the judge retains
`openai/gpt-5-nano`. Measure cost and wall time on the intended tier rather
than relying on a static estimate.

### Cache role

The `synth_response_cache.sqlite` is the reason re-runs are cheap. Every
concept-examples QC call, every Stage 0c API activation-evidence fetch and
definition/continuation response, every positive-mode synth call, and every
negative-mode synth call hits this SQLite cache keyed on
`sha256(model || prompt || tag)` where `tag` includes the retry attempt
index. Stage 0c's tag also includes its provider `max_tokens` cap, so a changed
request budget cannot replay a response produced under another cap. On a
mid-stage crash, re-launching the same stage replays every
previously-successful call from the cache without spending tokens.
`make_cache_key` lives at `synth/llm_response_cache.py:80-95`.

There is no persistent per-text `extraction_cache/` in the implementation.
Stage 2 runs its SAE max-pool extraction in memory and commits the resulting
`concept_features*.safetensors` files. Stage 3e writes the exact Stage 4
extraction input under `stage4_contexts_extraction/`; that parquet is then
converted into the separately configured context cache before Stage 4.

Stage 1b also has an in-memory prefix-percentile cache for every seed
candidate it actually scores. It is not persisted to disk: it only
deduplicates and batches SAE QC work inside one positive-synthesis stage
invocation, and it is keyed by the exact BOS-prepended extraction tokens plus
model/SAE/layer,
`cfg.extraction_tokenizer_id`, and percentile-stat namespace. Changing
the extraction tokenizer creates cache misses rather than reusing stale
prefix scores. Oversized prepared inputs fail loudly rather than being
clamped under the same key. Each cache-fill batch releases transient Gemma+SAE
activation tensors before the next batch runs, so broad concepts with
few clean seeds do not accumulate dirty-prefix scoring activations.

Delete either disk cache if you want to force a fresh run; both are safe
to remove while the directory is unlocked.

## Troubleshooting

**`No features with corpus count >= {min_corpus_count}`** — no SAE feature
in the statistics file is active on at least `min_corpus_count` documents.
Either the statistics file covers too few documents, or you extracted
against a different corpus than you are statisticking against. Inspect
the distribution of `per_feature_stats[fid].count` in the statistics
file; lower `min_corpus_count` only if you understand that rare features
may have noisy descriptions.

**`Stage 0c exhausted candidate_concepts.json without accepting ...`** —
Stage 0c could not find enough candidates that pass lazy examples QC
and local-SAE negative-prefix verification. In null mode the proposed
continuations come from the base SAE LM; in API mode they come from the
configured OpenRouter model. Inspect
`{run_tag}/concept_examples_qc_report.json` and
`{run_tag}/rejected_concepts.json`: the per-feature list records the
judge's score (0-2) and outcome category (`accepted`, `rejected`,
`fetch_error`, `judge_error`, `empty_examples`). Common fixes: lower
`activation_example_judge_min_score` from 1 to 0, set
`enable_concept_examples_qc=False`, lower
`verification_percentile_threshold`, or check the Neuronpedia
model_id / sae_source pair — a mismatch (e.g. asking for `gemma-2-2b`
when the SAE was trained against `gemma-2-9b`) returns 0 activations
per feature.

**`No features have quality descriptions`** — every Neuronpedia lookup
returned either an error or a description shorter than
`min_description_length`. Check the identity-addressed cache under
`{effective_output_root}/_neuronpedia_cache/v1/` for stored
descriptions; if most are empty, you may be pointing at the wrong SAE
source. If lookups raise, verify network access to `neuronpedia.org`.

**`Stage 1b complete-pair target shortfall`** — every failed candidate was
replaced until the admissible pool was exhausted or the zero-hit guard
terminated the concept. Inspect the per-concept candidate/scoring,
rich/avoid-discard, semantic-attempt, and termination summaries. Then consider
description quality, thresholds, the provider model,
`positive_max_attempts_per_seed`, or deliberately changing
`positive_zero_hit_bail_after`. Under `"warn"` and `"best"`, only complete
pairs survive; a concept whose final target-shortfall fraction is strictly
greater than
`max_concept_drop_fraction` is removed from that stage's explicit filtered
concept list. Stage 1b writes `concepts_stage1b.json`; Stage 1c may filter that
list further and writes the single downstream authority,
`concepts_stage1c.json`. The modes never define independent downstream
populations. Every surviving final concept must have both rich and avoiding
rows in every required corpus; Base, BPB, and BPB-Adv raise on a missing side
instead of silently skipping it.

**`invalid or legacy Stage 1b QC contract`** — a positive synthesis corpus
predates the corrected threshold-unit contract, has malformed metadata, or was
created with an `accept_threshold` different from the current Base config.
Preserve the old run for diagnosis and regenerate Stage 1b under a fresh
`run_tag`; do not relabel the legacy rows in place because their acceptance
decisions were made under different semantics.

**`Stage 1b checkpoint fingerprint mismatch` / target decrease / malformed or
inconsistent partial files** — a synthesis-semantic input differs from the
checkpoint, the requested target is below `highest_requested_target`, an
unsafe schema-v1 migration was attempted, or the JSON-committed parquet prefix
is corrupt or missing. Changing either the pre-provider prefix-scoring batch or
the effective post-provider QC grouping shape requires a fresh partial.
The stage fails before candidate-prefix scoring or provider access. Preserve
both partial files for diagnosis. Either restore the exact matching config and
reuse the same explicit `run_tag`, or explicitly remove both
`stage1b_partial.json` and `stage1b_partial.parquet` to restart Stage 1b. Never
delete only the parquet while retaining JSON: that is committed data loss. A
parquet without JSON is merely an ignored, uncommitted orphan. The
post-synthesis `qc_fallback_on_exhaust`, `max_concept_drop_fraction`, and
`max_dropped_concept_fraction` settings are fingerprinted semantic inputs too.
Changing any of them requires explicitly removing both partial files and
starting a fresh Stage 1b transaction under the new policy.

Schema-v2 same-or-higher target growth is the only target change that does not
require a restart. It requires the same explicit `run_tag` and authoritative
`stage1b_partial.json`; automatic tags normally differ between invocations.
If only a completed positive parquet remains, Stage 1b refuses in-place target
extension before scoring or provider access. Preserve it and use a fresh
explicit `run_tag`.

**`Stage 1b has no eligible positive seed prefixes`** — every Stage 0b train
prefix is shorter than `min_prefix_chars` after applying the exact
`max_prefix_chars` tail window. This fails before percentile-table
construction, any Stage 1b SAE forward, or synth-cache/API access. The CLI
dispatcher has already entered the shared base-model/SAE contexts at this
point. Check that the shared bounds are appropriate for the committed seed
pool; do not edit or regenerate Stage 0b merely to make Stage 1b indices line
up.

**`synth response protocol failure; skipping attempt`** in Stage 1b/1c — the
provider response failed its nonce envelope, exact suffix authentication,
full-response/stored-window validation, or deterministic copy/loop/control
guards. The response is intentionally excluded from SAE QC and from `"best"`
fallback tracking. A later attempt uses a distinct cache key. If all attempts
fail, the ordinary `qc_fallback_on_exhaust` policy applies.

**`OPENROUTER_API_KEY not set in environment`** — the
`OpenRouterLLMClient` raises at construction. Export the key, or
disable every stage that uses the LLM: set
`enable_concept_examples_qc=False`, use `concept_feature_mode="canonical"`,
and skip Stages 1b/1c.

**`SeedRow has empty continuation content tokens`** — a defensive schema-v2
guard. Only documents whose raw, BOS-free nanochat content has at least
`2 * seed_continuation_nanochat_tokens` tokens can become candidates, and the
last `M` source tokens are retained verbatim. Construction and load also check
the source count/slice equations, exact decode, configured cap and `M`, and SAE
CSR invariants.

**`Legacy, incomplete, or incompatible seed artifacts ... use a fresh
AxBench run tag`** — the v2 manifest or one of its two seed files is missing,
torn, stale, or mismatched. Do not copy metadata onto the old files: legacy BOS
ownership and exact source slices cannot be reconstructed. Preserve the old
run read-only and rerun Stage 0b under a new tag.

**`Seed pool is empty after processing ... docs`** — every document in
your validation shard tokenized to fewer than `2 *
seed_continuation_nanochat_tokens` nanochat tokens. Either pick a shard
with longer documents, lower `seed_continuation_nanochat_tokens`, or
pass additional shards via `validation_shards`.

**`seed_prefix_max_nanochat_tokens < seed_continuation_nanochat_tokens`**
— `validate_config` rejects this. The prefix cap must be at least as
large as the continuation length so every SeedRow has prefix tokens.
The default `(512, 256)` already satisfies this.

**`Target encdec model lacks _loaded_meta_data attribute`** — after
calling `build_encdec_model(...)` you forgot
`model._loaded_meta_data = meta_data`. Without it,
`detect_target_model_metadata` cannot read the training
`negative_selection` / `feature_selection` config and refuses to proceed.

**Hydra / OmegaConf gotchas** — the config is a plain dataclass; if you
construct it from a DictConfig, use
`AxBenchBaseConfig(**OmegaConf.to_container(cfg, resolve=True))` so the
nested configs become real dataclass instances rather than DictConfig
proxies. Hydra's lazy resolution can mask missing keys; calling
`validate_config(cfg)` before any stage is the easiest way to fail fast.

## Glossary

- **Base model** — a language model trained with next-token prediction on
  raw corpus data, with no instruction tuning or RLHF. AxBench-Base
  evaluates steering methods on such models; AxBench proper evaluates
  instruction-tuned chat models.
- **Concept** — an SAE feature plus its human-readable description.
  Represented by the `Concept` dataclass. Each concept has a dense
  `concept_id` (0..N-1) and a `feature_index` into the SAE vocabulary.
- **Seed / SeedRow** — a validation-shard document represented as a
  BOS-free exact source prefix slice, a fixed-length exact natural
  continuation slice, source provenance/bounds, and independently extracted
  SAE features active over the prefix. Used as generation context for
  synthesis and eval; the owned BOS is not stored in the row.
- **Prefix** — the tokens fed to the target model before it is asked to
  generate. Variable length, capped by `seed_prefix_max_nanochat_tokens`.
- **Continuation** — the natural document tail after the prefix. Always
  exactly `seed_continuation_nanochat_tokens` tokens long by
  construction. Used as paired natural concept-avoiding text in Stage 1b and
  as tokens-to-predict in Stage 4 approach-2 generation.
- **Positive mode / negative mode** — Stage 4 evaluates steering in two
  directions: push neutral text toward the concept (positive mode) or
  push concept-rich text away from the concept (negative mode). They
  use different seed pools and different training corpora.
- **concept_rich / concept_avoiding** — row-level content class. A
  `concept_rich` row is text that fires the concept feature; a
  `concept_avoiding` row is text that does not. Every synth training
  row has one of these two labels.
- **canonical vs delta_ranked** — Stage 2 concept-to-feature mapping
  mode. `canonical` uses Neuronpedia's mapping as ground truth (trivial
  writer, no GPU). `delta_ranked` runs base Gemma + SAE on the synth
  corpus and sorts features by the percentile delta between concept_rich
  and concept_avoiding rows, with the Neuronpedia feature's position in
  that list stored as `canonical_rank` for diagnostics.
- **Target model** — the model being steered. Two flavors: the decoder-
  only base GPT and the encoder-decoder conditional variant.
- **Base Gemma / base SAE** — the analysis pair, not a target. Used to
  compute SAE activations on arbitrary text during QC, concept-feature
  mapping, and Stage 4 approach-2 extraction.
- **Percentile table** — a `(n_sae_features, 99)` lookup table built
  from the feature statistics by `build_percentile_table`. Used to map
  raw activation magnitudes onto a 0-100 percentile scale so thresholds
  are comparable across features of different scale.
- **gen_source** — a provenance string on synth rows. Formats:
  `openrouter:{model}` for LLM-generated, `local:{source}/{tag}` for
  decoder-only-generated, `corpus:doc_continuation@{shard}/{doc}` for
  Stage 1b's paired natural-continuation rows.
