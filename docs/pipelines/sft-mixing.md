# SFT-Pretraining Mixing Pipeline

Mix SFT conversation data into pretraining parquet shards so the feature extraction pipeline can process both pretraining and SFT data in a single pass.

## Overview

```
SFT parquets with full-rendered `text` rows ──→ Load text
                                                   ↓
Pretraining shards ──────────────────────→ Merge + shuffle ──→ Mixed shards
                                                                   ↓
                                                           mixing_meta.json
```

**Prerequisites:** SFT parquets must already exist (see `scripts/preprocess/convert_sft_to_parquet.py`).

**Script:** `scripts/preprocess/mix_sft_into_pretraining.py`
**Config:** `configs/data/mix_sft_into_pretraining.yaml`

## Quick Start

```bash
# Mix safety SFT into pretraining data
uv run python -m scripts.preprocess.mix_sft_into_pretraining mixtures='[safety]' model=gemma-2-2b

# Mix multiple SFT mixtures
uv run python -m scripts.preprocess.mix_sft_into_pretraining \
    mixtures='[safety,capabilities]' model=gemma-2-2b

# Custom output name
uv run python -m scripts.preprocess.mix_sft_into_pretraining \
    mixtures='[safety,capabilities]' model=gemma-2-2b output_name=my_mix
```

## How It Works

1. **Load** — Each SFT parquet contributes its pre-rendered `text` rows. These are produced by `scripts/preprocess/convert_sft_to_parquet.py` using the nanochat conversation renderer with the decoder BOS stripped.

2. **Distribute** — Rendered texts are shuffled and distributed uniformly across all pretraining shards (each shard gets `floor(N/S)` or `ceil(N/S)` SFT documents).

3. **Merge** — Each pretraining shard is loaded, SFT texts are appended, and the combined documents are shuffled with a per-shard deterministic seed.

4. **Write** — Output shards have the same names as the input pretraining shards, with a single `text` column. A `mixing_meta.json` records statistics.

Output parquets are format-identical to base pretraining data, so both the dataloader and feature extraction pipeline can consume them without changes.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mixtures` | **required** | List of SFT mixture names, e.g. `[safety]` or `[safety, capabilities]` |
| `model` | `gemma-2-2b` | Model name recorded in `mixing_meta.json` |
| `output_name` | null | Output directory name (null = auto from mixture names, e.g. `capabilities+safety`) |
| `seed` | `0` | Random seed for shuffling |
| `base_data_dir` | null | Pretraining parquet dir (null = `{base_dir}/base_data/`) |
| `sft_data_dir` | null | SFT parquet base dir (null = `{base_dir}/sft_data/`) |
| `output_dir` | null | Output dir (null = `{base_dir}/mixed_data/{output_name}/`) |

## Directory Layout

```
$NANOCHAT_BASE_DIR/
├── base_data/
│   ├── shard_00000.parquet    # pretraining shards (input)
│   ├── shard_00001.parquet
│   └── ...
├── sft_data/
│   ├── safety/                # SFT parquets by mixture (input)
│   │   └── shard_00000.parquet
│   └── capabilities/
│       └── shard_00000.parquet
└── mixed_data/
    └── capabilities+safety/   # output (auto-named)
        ├── shard_00000.parquet
        ├── shard_00001.parquet
        ├── ...
        └── mixing_meta.json
```

## Metadata Output

`mixing_meta.json` contains:

| Field | Description |
|-------|-------------|
| `mixtures` | List of mixture names used |
| `model` | Model name |
| `seed` | Random seed |
| `timestamp` | ISO 8601 UTC timestamp |
| `total_pretraining_docs` | Number of pretraining documents |
| `total_sft_docs` | Number of SFT documents added |
| `total_docs` | Combined total |
| `sft_fraction` | Fraction of SFT documents in the mix |
| `n_shards` | Number of output shards |

## Downstream Usage

Mixed data is for co-training SFT conversations into pretraining. The output
has the same single-`text` format as `base_data`, so extract it as base-phase
input (the explicit `data_dir` selects the mixed directory):

```bash
uv run python -m steering.extract_features \
    dataset_phase=base \
    data_dir=path/to/mixed_data/capabilities+safety/ \
    num_shards=10
```

Do not use features extracted from mixed data for conditioned SFT: mixed
parquets contain no `original_index`, so there is no route back to task indices.
For conditioned-SFT features, follow [SFT Feature Extraction](encoder-decoder-training.md#sft-feature-extraction):
run `convert_sft_to_parquet`, extract features from the *unmixed* SFT parquet,
then run `compute_statistics`.
