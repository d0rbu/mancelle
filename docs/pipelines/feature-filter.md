# Feature Filter — User Guide

## Quick start

```bash
export NEURONPEDIA_API_KEY=np-...
export OPENROUTER_API_KEY=sk-or-...

uv run python -m steering.eval.feature_filter.cli \
  model_id=gemma-2-2b \
  sae_source=8-gemmascope-res-16k \
  tokenizer_id=google/gemma-2-2b \
  feature_ids='[42, 100, 1000]' \
  samples_per_bin=3 \
  interp_min_score=1 \
  consistency_min_score=3
```

## Typed configuration domains

The Hydra boundary constructs the feature-filter, judge, and Neuronpedia
dataclasses under `beartype`. Unary requirements are expressed with reusable
`phantom-types` predicates; invalid Hydra values are wrapped in
`ConfigValidationError` naming the top-level or nested block, while direct
library construction raises the corresponding `beartype` violation.

| Config surface | Domain |
|---|---|
| `model_id`, `sae_source` | Non-blank path component without `/`, `\`, or `..` |
| `feature_ids` | `null` or a non-empty list of unique strict non-negative integers |
| `n_sae_features`, concurrency limits, optional `max_judge_calls` | Strict positive integers (`max_judge_calls` may be `null`) |
| `samples_per_bin` | Strict integer in `[1, 5]` |
| `interp_min_score`, `consistency_min_score` | Strict integer in `[0, 5]` |
| `error_abort_fraction` | Finite non-boolean number in `[0, 1]` |
| `shutdown_grace_seconds` | Finite non-negative number |
| `hist_sum_tolerance`, Neuronpedia timeouts/backoffs | Finite positive numbers |
| judge/Neuronpedia retry counts | Strict non-negative integers |

Feature IDs must still be below `n_sae_features`, the Neuronpedia maximum
backoff must still be at least its base backoff, paths must not contain
unresolved Hydra tokens, and output-directory filesystem checks remain explicit
relationship or environment validation.

Sampling reads Neuronpedia's native magnitude bins (5 quintiles from
weakest to peak firing, plus a sentinel top-K bucket that we discard).
For each quintile we return up to `samples_per_bin` entries sorted by
peak activation magnitude. Band 0 entries where `max_value == 0` are
surfaced to the judge as "negative baselines" rather than weak firings.

Outputs land in `./out/feature_filter/${model_id}/${sae_source}/`:
- `feature_scores.jsonl` — per-feature scoring rows
- `filter_report.json` — aggregate report
- `accepted_feature_ids.txt` — the flat list Spec B consumes
- `runner_manifest.json` — run metadata for resume checks
- `np_cache/` — activation payloads, per-feature JSON
- `judge_cache.sqlite` — OpenRouter response cache

## Meta-validation gate (run BEFORE 16k-feature runs)

Before spending OpenRouter credits on a full SAE scan, validate the judge's
signal on 10 hand-picked features:

1. Pick 5 **obviously good** features — ones whose Neuronpedia description
   clearly and consistently describes a single concept (e.g. German articles,
   URLs, punctuation).
2. Pick 5 **known noisy / polysemantic** features — descriptions that
   kind-of-fit at peak activation but break at lower percentiles.
3. Run the pipeline on these 10.

**Block rollout** unless:
- ≥ 4/5 of the "obviously good" features score `consistency >= 3`.
- ≥ 4/5 of the "known noisy" features score `consistency <= 2`.

If the judge fails this check, iterate on the prompt (see
`steering/eval/feature_filter/prompts/judge_feature_interpretability.txt`)
or switch to a stronger judge model (`judge=openai_gpt5_mini`) BEFORE the
full scan.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | error-rate exceeded `error_abort_fraction`, OR concurrent runner conflict |
| 2 | config validation error |
| 3 | auth failure (Neuronpedia or OpenRouter 401) |
| 4 | cost cap exceeded |
| 130 | SIGINT |

## Per-feature outcomes

Each scored feature gets one `outcome`. The error-rate abort (exit 1)
counts only `fetch_error + judge_error + prompt_error`; the others —
including `not_published` — are **excluded** from `error_abort_fraction`.

| Outcome | Meaning | Counts toward error rate? |
|---|---|---|
| `accepted` / `rejected` | judged | no |
| `no_nonzero_activations` | feature never activates | no |
| `fetch_error` | Neuronpedia 401/429/5xx/network/malformed (a genuine failure) | **yes** |
| `judge_error` / `prompt_error` | judge failure | **yes** |
| `not_published` | Neuronpedia returned **HTTP 404** — it does not host this feature/SAE (expected for gemma-scope-2 SAEs Neuronpedia has not ingested) | **no** |

A run where every feature is `not_published` (a v2 SAE Neuronpedia hasn't
ingested) exits **0** with a `WARNING` naming the count and `sae_source`,
instead of the old spurious exit 1. Only HTTP 404 maps to
`not_published`; a true Neuronpedia outage returns 5xx and still counts as
`fetch_error`, so real failures still trip the abort.

## Resuming

Running with the same `output_dir` skips already-scored features. Changing
`model_id` or `sae_source` on resume raises `ResumeMismatchError` — use
a fresh `output_dir` for different SAEs.

**Schema bump (`SCHEMA_VERSION` 2 → 3):** the `not_published` outcome
changed the persisted-artifact schema. Resuming a run whose
`runner_manifest.json`, per-feature JSONL rows, or Neuronpedia activation
cache was written under schema 2 is rejected cleanly with a typed
`SchemaVersionError` / `DataLoadError` — re-run from a fresh `output_dir`
(classification semantics changed, so a resume must re-fetch rather than
silently mix schemas).

## Fixture-staleness watchdog

Run monthly:
```bash
uv run python -m scripts.validation.check_neuronpedia_schema
```
Warns if Neuronpedia's response shape drifts; regenerate fixtures with
`uv run python -m scripts.test.capture_neuronpedia_fixtures`.
