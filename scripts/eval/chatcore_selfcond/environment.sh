#!/usr/bin/env bash
set -euo pipefail
export G=/orcd/data/dylanhm/001/nano-gpt-pretrain-steer
export B=$G/full3b-250k-eval
export S=$G/claude-sft3b-steered-20260905
export TAG=full3b_clustered_c2048_long_w32_bf16_b14_fp
export REPO=$S/chatcore/repo
export UV_PROJECT_ENVIRONMENT=/home/henryac/claude-turnseg-val-20260904/repo/.venv
export HF_HOME=/home/henryac/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 NANOCHAT_DTYPE=bfloat16
cd "$REPO"
test "$(git rev-parse HEAD)" = "$(cat "$S/chatcore/launch/EXPECTED_COMMIT")"

export BASE_SOURCE=$G/full3b-w32fp-prod-20260827/nanochat_cache/conditional_encdec_checkpoints/$TAG/s128/dist_geometric/mean_12.0/std_12.0/wl_b02b21f4c61182eb/fc_129b4f428c5665dea8b72ef8f308d33b40c9d343dd27f8ba93da8a300bbb9149
export BASE_FEATURES=/orcd/pool/005/henryac/nano-gpt-pretrain-steer/full3b-clustered-c2048-20260823T194759Z/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos/splice/cut_frac0.15/seed42
export FIXED_RUNTIME=$B/selfcond_runtime
export TURN_RUNTIME=$S/chatcore/selfcond_runtime
export FIXED_LEAF=$FIXED_RUNTIME/conditional_encdec_checkpoints/$TAG
export TURN_TAG=$TAG/standard_sft_conditioned_checkpoints
export FIXED_CACHE=$B/cache/core_chat_self_conditioned_v4
export TURN_CACHE=$S/chatcore/s32/cache
