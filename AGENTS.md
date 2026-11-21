# AGENTS.md — nano-gpt-pretrain-steer

You are working on a research repo that pretrains GPT-2 class language models and conditions them on Sparse Autoencoder (SAE) features via an encoder-decoder cross-attention architecture. Built on a vendored copy of [karpathy/nanochat](https://github.com/karpathy/nanochat).

This file is the AI-agent entry point — it points to the docs that have the actual content. **Do not duplicate doc content here. If you find yourself wanting to, update the linked doc instead.**

## Read these first

- [`README.md`](README.md) — project pitch + 1-minute quickstart
- [`docs/cli-reference.md`](docs/cli-reference.md) — every CLI command, grouped by purpose
- [`docs/reference/architecture.md`](docs/reference/architecture.md) — model design, training systems, key choices
- [`docs/reference/configuration.md`](docs/reference/configuration.md) — every Hydra config knob
- [`docs/reference/file-reference.md`](docs/reference/file-reference.md) — file-by-file map
- [`docs/onboarding/glossary.md`](docs/onboarding/glossary.md) — domain vocabulary (SAE, BPB, segment, percentile, overlay, …)
- [`docs/onboarding/workflows.md`](docs/onboarding/workflows.md) — how to do common dev tasks (custom train, eval, profile, debug OOM, …)
- [`docs/pipelines/`](docs/pipelines/) — pipeline deep-dives

## Repo layout

```
steering/    research code (SAE feature extraction + encoder-decoder conditioning)
nanochat/    vendored framework — DO NOT MODIFY (see nanochat/VENDORED.md)
vendor/      vendored projects with their own environments and policies
scripts/     pipeline entry points
configs/     Hydra YAML configs
tasks/       SFT task definitions
tests/       pytest suite (markers: slow, very_slow, benchmark, api, requires_feature_data)
runs/        end-to-end pipeline shell scripts (speedrun.sh, runcpu.sh, ...)
docs/        documentation (reference, pipelines, evaluation, performance, onboarding)
dev/         experiment notes, analysis notebooks
out/         generated artifacts (gitignored)
```

## Critical gotchas

These are the things that will burn you if you don't know them upfront. Each is covered in detail in a linked doc — the bullet here is just the warning.

- **Two tokenizers, never mix.** Gemma (256k vocab) for `steering/` feature extraction; RustBPE (32,768) for `nanochat/` training. Cross-tokenizer alignment uses character-level segment boundaries — see `steering/conditional/encoder_decoder/full_sequence_routing.py`. Full context in [`docs/reference/architecture.md`](docs/reference/architecture.md#tokenizer-warning).
- **`nanochat/` is read-only.** Don't patch it; route fixes via `steering/` adapters or escalate. PRs touching `nanochat/` are routed through `.github/CODEOWNERS`. See [`nanochat/VENDORED.md`](nanochat/VENDORED.md).
- **AxBench uses its own environment and patch policy.** `vendor/axbench/` is a Python 3.12 uv project; run it with `uv --project vendor/axbench` and record reviewed local patches in [`vendor/axbench/VENDORED.md`](vendor/axbench/VENDORED.md).
- **Two Hydra patterns.** `@hydra.main` for single-process scripts; module-load `compose()` for torchrun-launched scripts (which can't use the decorator). The CLI override syntax differs (`--config-name=` vs `config_name=`). See [`docs/cli-reference.md`](docs/cli-reference.md) → "Hydra config selection."
- **`feature_mask` is BOTH a selection-layer AND a structural concern.** Selection layer: it is folded into `compute_selection_hash` via a CONDITIONAL key — never add the key unconditionally or you will orphan every existing `sel/<hash>/` shard (invariant I1). Base shards/stats/pct/tok/packed-rows are NEVER recomputed (I3). **Structural (issue #695):** a resolved mask also *sizes the encoder's feature-id-indexed embedding tables* — they get one row per **allowed** feature (`V`), not one per SAE feature. Ids are remapped at the encoder boundary, so every caller still speaks ORIGINAL SAE ids, but a compacted checkpoint's rows are meaningless without its vocabulary. Compaction is always on whenever a mask resolves; no config knob. See [`docs/pipelines/encoder-decoder-training.md`](docs/pipelines/encoder-decoder-training.md#feature-mask-blacklistwhitelist) and § Encoder Feature-Dictionary Compaction.
- **`feature_mask_freq_min` / `feature_mask_freq_max`** (defaults `1e-5` / `0.5`) are the auto **producer** for `feature_mask`: KEEP iff `freq_min < count/n_segments <= freq_max`. Reads `feature_statistics.safetensors` (`steering/conditional/freq_blacklist.py`). LB inclusive (default `1e-5` kills `count==0` dead features *and* other features too rare to be autointerp-scoreable, i.e. `count <= 1e-5 * n_segments`; pass `min=0` explicitly to kill `count==0` dead features only); UB exclusive (max=0.5 kills features active in >50% of segments). Either bound nullable; both null ⇒ disabled. Source is the stats artifact so the train/val split is whatever `compute_statistics` used (`exclude_last_shard: true` by default). Both-null/empty/explicit-mask ⇒ byte-identical no-op (I1/I2/I5). The bounds never enter any hash — only the resolved id set does. The old single key `feature_mask_freq_threshold` is renamed to these two (no longer recognized). See [`docs/reference/configuration.md`](docs/reference/configuration.md#feature_mask_freq_min--feature_mask_freq_max--auto-activation-frequency-blacklist-producer).
- **Gradient-attribution ranking is offline-only** and folds into `compute_selection_hash` via a CONDITIONAL canonical key (same I1 rule as `feature_mask` — absent ⇒ byte-identical legacy hash). Every live selection path fails fast; the pipeline needs the `grad/<hash>/` overlay + one-time packed provenance. See [`docs/pipelines/gradient-attribution-selection.md`](docs/pipelines/gradient-attribution-selection.md).
- **Imports at the top of every file.** No exceptions, including inside `if __name__ == "__main__":` blocks. Restructure to avoid circular deps.

## Conventions

These conventions apply to project-owned code. Vendored projects retain their
upstream style and use the tooling documented in their `VENDORED.md` files.

- `import torch as t` everywhere
- `from loguru import logger` for all logging (DDP-aware sink in `nanochat/common.py` — rank 0 logs INFO, others log WARN+)
- `uv sync` to install, `uv run` to invoke, `uvx ty check` to type-check (the type checker is `ty`, not pyright)
- CSR sparse format for per-segment features (invariants documented in [`docs/onboarding/glossary.md`](docs/onboarding/glossary.md) § CSR sparse format and [`docs/pipelines/feature-extraction.md`](docs/pipelines/feature-extraction.md) § Shard Format)
- Hydra for all CLI configs; new knobs go in `configs/<area>/*.yaml` with a default

## Where to put new code

| Kind of change | Goes in |
|---|---|
| New SAE pipeline / extraction step | `steering/` |
| New encoder-decoder mechanism | `steering/conditional/encoder_decoder/` |
| New eval | `steering/eval/` + `scripts/<eval>.py` + `configs/eval/<eval>.yaml` |
| New SFT task | `tasks/<name>.py` (subclass `Task` from `tasks/common.py`) |
| New Hydra config | `configs/<area>/<name>.yaml` (with `defaults: [<base>, _self_]` if extending) |
| New end-to-end runner | `runs/<scenario>.sh` |
| Bug fix in `nanochat/` | **Open an issue first** — see `nanochat/VENDORED.md` |
| Reviewed AxBench compatibility fix | `vendor/axbench/` — log it in `vendor/axbench/VENDORED.md` |

## Testing

```bash
uv run pytest tests/                  # fast, default (excludes slow + very_slow)
uv run pytest tests/ -m slow          # GPU-required
uv run pytest tests/ -m benchmark     # perf tests
uvx ty check                          # type check
```

Markers: `slow`, `very_slow`, `benchmark`, `api`, `requires_feature_data`. Configured in `pyproject.toml`.
