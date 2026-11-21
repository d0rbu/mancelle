# Axbench-BPB Test Fixtures

Committed fixtures (~8 MB) used by unit / functional / integration tests.

## Contents

- `mini_axbench_base_run/` — 5 concepts, 4 samples/side, all Stage 0-3
  artifacts, including a committed schema-v2 train/eval seed pair and manifest.
  Its positive synthesis Parquet carries the current versioned Stage 1b QC
  contract and uses raw percentile-point QC values.
- `tiny_encdec/` — 2-layer toy encoder-decoder checkpoint
- `tiny_decoder_only/` — 2-layer toy decoder-only checkpoint
- `tiny_sae/` — 64-feature toy SAE

## Regenerating

Run `uv run python -m tests.axbench_bpb.fixtures.regenerate_fixtures` from
the repo root. This re-runs a minimal axbench-base pipeline in CPU mode
against a toy corpus and writes the outputs back under this directory.

Commit the regenerated files if the axbench-base schemas have changed.
