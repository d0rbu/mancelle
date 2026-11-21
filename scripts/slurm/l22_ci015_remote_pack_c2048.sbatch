#!/usr/bin/env bash
# FULL PRODUCTION launch for the L22 c2048 remote packed-row run (issue #1437):
# all 1400 raw bodies of the l22_ci_nobos_raw/splice/cut_frac0.15/seed42 family
# (max_shards=null), measured ~18 h wall across preempted windows. This is NOT
# a profiling/benchmark harness — profile=true is the pipeline's lightweight
# built-in telemetry sidecar (negligible overhead), kept on so every window
# leaves a diagnosable record.
# Recipe shape (rationale in issue #1437's closing comment):
#   * NO FUSE mount, NO symlink publication into the canonical tree, NO manual
#     resume-state repair: percentile leaves stream through the verified transport and the
#     resume state carries semantic identity only, with checkpoint-seek resume.
#   * Requeues resume automatically: checkpoint-seek re-downloads at most the cursor-boundary
#     bodies; replay remains the automatic fallback for torn checkpoints.
#   * compress_workers=4 (sweep-measured: cw4 saturates ~320 rows/s; cw2 leaves ~5% and
#     38 s/shard of producer backpressure on the table).
#   * OUTPUT OFFLOAD is on (issue #1462): the full c2048 family is ~2.4 TB against a
#     1 TB pool quota, so each completed packed+provenance shard is published to the
#     bucket, verified by size+Xet, receipted, and only then evicted locally. Local
#     growth stays bounded at compress_workers+2 shards. A resume also reconciles
#     whatever an earlier window left on disk before packing continues.
#   * --signal=B:TERM@300: the one-shot SIGTERM handler persists the failure sidecar and
#     drains in-flight zstd (~1-2 min) — TERM is forwarded to the python child.
#   * repo_root: a main checkout at or after the #1437 stack merge (18fd5647f), with its
#     own fresh .venv.
#SBATCH --job-name=l22-ci015-remote-pack-c2048
#SBATCH --partition=mit_preemptable
#SBATCH --account=mit_general
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/orcd/scratch/orcd/007/henryac/nano-gpt-pretrain-steer/logs/l22-ci015-remote-pack-c2048-%j.log

set -Eeuo pipefail

project_root=/orcd/scratch/orcd/007/henryac/nano-gpt-pretrain-steer
repo_root="$project_root/repo"   # main checkout >= 18fd5647f with its own .venv
token_file="$project_root/secrets/hf_token"
feature_dir=/orcd/pool/005/henryac/nano-gpt-pretrain-steer/overlay-runner/packed-feature-view/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos
data_dir=/orcd/pool/005/henryac/nano-gpt-pretrain-steer/overlay-runner/spliced-parquets-cut015-seed42
# Staging on POOL, not scratch: the scratch PROJECT quota is invisible to
# shutil.disk_usage/statvfs, and exhausting it makes Lustre silently drop
# buffered download pages at writeback (full-length wrong-hash bodies that die
# at the sealed-inventory check — see issue #1437's closing comment and #1454).
# Check pool headroom (~6 GB staging) before launch.
staging_dir="/orcd/pool/005/henryac/nano-gpt-pretrain-steer/.l22-ci015-remote-pack-staging-$SLURM_JOB_ID"
profile_path="/orcd/pool/005/henryac/nano-gpt-pretrain-steer/overlay-runner/profiles/packed-remote-ci015-c2048-$SLURM_JOB_ID.json"
job_tmp="$(mktemp -d "/tmp/$USER-l22-pack-$SLURM_JOB_ID-XXXXXX")"
trap 'rm -rf -- "$job_tmp" "$staging_dir"' EXIT INT TERM

export HF_TOKEN="$(<"$token_file")"
export HF_XET_HIGH_PERFORMANCE=1
export HF_HOME="$job_tmp/hf-cache"
export NANOCHAT_BASE_DIR="$job_tmp/nanochat-cache"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ARROW_DEFAULT_MEMORY_POOL=system
export MALLOC_ARENA_MAX=4 OMP_NUM_THREADS=2
mkdir -p "$NANOCHAT_BASE_DIR/tokenizer" "$staging_dir"
cp -f "$project_root/nanochat_cache/tokenizer/tokenizer.pkl" \
      "$project_root/nanochat_cache/tokenizer/token_bytes.pt" "$NANOCHAT_BASE_DIR/tokenizer/"
[[ -s "$feature_dir/feature_statistics.safetensors" ]]
[[ -s "$feature_dir/tok/dc6308e6116cbfe0/seg_boundaries/manifest.json" ]]

cd "$repo_root"
"$repo_root/.venv/bin/python" -u -m scripts.preprocess.precompute_packed_rows \
  "feature_dir=$feature_dir" "data_dir=$data_dir" \
  model=gemma-3-4b-pt layer=22 activation_type=residual pooling=max_pool \
  segment_size=128 sae_width=65k \
  context_len=2048 \
  max_shards=null buffer_size=1000 rows_per_shard=16384 \
  resume=true zstd_level=12 compress_workers=4 \
  profile=true "profile_summary_output=$profile_path" \
  remote.enabled=true remote.bucket_id=henryac/nano-gpt-pretrain-steer \
  "remote.local_features_root=$feature_dir" \
  remote.remote_features_root=l22_ci_nobos_raw/splice/cut_frac0.15/seed42 \
  remote.offload.enabled=true \
  remote.offload.remote_packed_root=l22_ci_nobos_packed/splice/cut_frac0.15/seed42 \
  "remote.staging_dir=$staging_dir" \
  remote.expected_shard_count=1400 \
  remote.retry_timeout_seconds=900 remote.retry_interval_seconds=5 &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM
wait "$child"
