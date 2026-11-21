# Artifacts sync — pulling and pushing checkpoints + feature shards

The project's reference checkpoints and feature shards live in a Hugging
Face **storage bucket** at `hf://buckets/<owner>/<bucket>/`. This page
covers how to pull them onto a fresh machine, how to push new artifacts
up, and which logical *categories* are defined.

> **Storage buckets** are HF's S3-like object-storage primitive
> (mutable, non-versioned, optimized for working artifacts like
> checkpoints and intermediate data). They are the right tool for this
> use case, distinct from versioned `model` / `dataset` git repos.

All sync operations go through the `huggingface_hub` Python SDK
(`sync_bucket()`); we never shell out to the `hf` CLI.

## First-time setup

1. Add `HF_TOKEN` to your `.env` (read access for pull; write for push). Get one at <https://huggingface.co/settings/tokens>.
2. Confirm `NANOCHAT_BASE_DIR` is set. If unset, falls back to `nanochat_cache` (relative to repo root), matching `nanochat.common.get_base_dir()`. Note: the project's `.env.example` and `runs/*.sh` set it to `$HOME/.cache/nanochat` instead — that's a docs-level convention, not the code default.
3. The bucket itself is created out-of-band by the maintainer via `hf buckets create <owner>/<name> --private` (one-time setup; not part of any sync flow).

## Pull (operator path)

```bash
# Default: checkpoints (base + sft + encdec). Skips `features` — those are tens-to-hundreds of GB.
bash runs/download_artifacts.sh

# Pull everything including feature shards
bash runs/download_artifacts.sh --all

# Pull a specific subset
bash runs/download_artifacts.sh base_checkpoints features

# Preview without transferring
bash runs/download_artifacts.sh --dry-run

# Show what WOULD be pulled (no sync, no Hydra)
bash runs/download_artifacts.sh --list

# Help
bash runs/download_artifacts.sh --help
```

## Pull selectively (Python entrypoint)

For finer-grained filtering — picking sub-paths inside a category — use the Python entry point directly:

```bash
# Just one category (same as `bash runs/download_artifacts.sh base_checkpoints`)
python -m scripts.utils.hf_pull category=base_checkpoints

# Only one model under that category
python -m scripts.utils.hf_pull category=base_checkpoints \
    +include='["d12/**"]'

# A specific feature shard tree
python -m scripts.utils.hf_pull category=features \
    +include='["s128/l8/a_res/p_max_pool/f16k/**"]'

# Preview a pull
python -m scripts.utils.hf_pull category=features dry_run=true
```

## Push (maintainer)

Pushing requires `HF_TOKEN` with write access to the bucket's owner namespace.

```bash
# Default bulk push (checkpoints only; skips `features` — those are huge)
bash runs/upload_artifacts.sh

# All categories including features
bash runs/upload_artifacts.sh --all

# Specific subset
bash runs/upload_artifacts.sh base_checkpoints features

# Preview
bash runs/upload_artifacts.sh --dry-run

# Show what WOULD be pushed (no sync, no Hydra)
bash runs/upload_artifacts.sh --list

# DESTRUCTIVE — also removes bucket files that don't exist locally
bash runs/upload_artifacts.sh --delete

# Single category, additive
python -m scripts.utils.hf_push category=base_checkpoints

# Single category, preview
python -m scripts.utils.hf_push category=base_checkpoints dry_run=true
```

The `features` category is opt-in on both the upload and download wrappers — pass it explicitly or use `--all`.

## Categories

Defined in [`configs/artifacts/default.yaml`](../../configs/artifacts/default.yaml):

| Category | Local root | Bucket subpath | Default exclude |
|---|---|---|---|
| `base_checkpoints` | `$NANOCHAT_BASE_DIR/base_checkpoints` | `checkpoints/base/` | `**/optim_*.pt` |
| `sft_checkpoints` | `$NANOCHAT_BASE_DIR/sft_checkpoints` | `checkpoints/sft/` | `**/optim_*.pt` |
| `encdec_checkpoints` | `$NANOCHAT_BASE_DIR/conditional_encdec_checkpoints` | `checkpoints/conditional_encdec/` | `**/optim_*.pt` |
| `features` | `out/data/features` | `features/` | (none) |

`default_exclude` filters skip optimizer state by default — those files are huge and rarely needed for downstream eval. To include them, add `no_default_exclude=true` on the CLI.

## Filter semantics

- `+include='["pat1","pat2"]'` — replaces the category's `default_include` (which is usually empty = "all") with these patterns. Glob style; matches `huggingface_hub.sync_bucket`'s `include` parameter.
- `+exclude='["pat3"]'` — appends to the category's `default_exclude`.
- `no_default_include=true` — drop the category's defaults before applying user includes.
- `no_default_exclude=true` — drop the default excludes (e.g., to also pull optimizer state).

## Adding a new category

Edit `configs/artifacts/default.yaml`, add an entry under `categories:`, then it becomes available from the CLI immediately. No code change needed.

## Adding a new bucket / per-machine override

Drop `configs/local/artifacts.yaml` (gitignored via `configs/local/`) overriding `bucket_owner` / `bucket_name`. The Hydra `defaults:` block already wires it in.

## Implementation map

| Where | What |
|---|---|
| `configs/artifacts/default.yaml` | Bucket name + categories |
| `steering/artifacts/config.py` | `ArtifactsConfig`, `CategoryConfig` dataclasses |
| `steering/artifacts/paths.py` | Canonical HF storage-bucket URI construction |
| `steering/artifacts/filters.py` | Default+user include/exclude merge |
| `steering/artifacts/categories.py` | Category lookup → resolved local/remote endpoints |
| `steering/artifacts/sync.py` | `sync_push` / `sync_pull` wrappers around `huggingface_hub.sync_bucket` |
| `scripts/utils/hf_push.py` | Hydra CLI for upload |
| `scripts/utils/hf_pull.py` | Hydra CLI for download |
| `runs/download_artifacts.sh` | Operator one-liner: pull all standard categories |
| `runs/upload_artifacts.sh` | Operator one-liner: push all standard categories |
