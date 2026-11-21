from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch as t

from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import GPTConfig
from nanochat.tokenizer import LogicalTurnSpan, get_tokenizer
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.sft.structured_rollout import (
    STRUCTURED_RENDERER_VERSION,
    STRUCTURED_ROLLOUT_FORMAT,
    STRUCTURED_ROLLOUT_SCHEMA_VERSION,
    StructuredRolloutRow,
    rustbpe_tokenizer_sha256,
    structured_row_digest,
    structured_rows_to_table,
)
from tests.steering.selfcond.test_sft_grpo_checkpoint import _metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_BASE = Path(os.environ.get(
    "NANOCHAT_BASE_DIR", "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/nanochat_cache"
))


@dataclass(frozen=True)
class _Case:
    base_dir: Path
    train_dir: Path
    val_dir: Path
    base_lm_head: t.Tensor


def _aggregate(digests: list[str]) -> str:
    return hashlib.sha256("".join(f"{digest}\n" for digest in digests).encode()).hexdigest()


def _write_artifact(
    path: Path,
    *,
    split: str,
    tokenizer,
    tokenizer_sha256: str,
) -> None:
    path.mkdir(parents=True)
    token_ids = tuple(tokenizer.encode("user question assistant answer with enough tokens"))
    assert len(token_ids) >= 4
    midpoint = max(1, len(token_ids) // 2)
    rows = tuple(
        StructuredRolloutRow(
            target_token_ids=token_ids,
            target_mask=(False,) * midpoint + (True,) * (len(token_ids) - midpoint),
            logical_turns=(
                LogicalTurnSpan("user", 0, midpoint),
                LogicalTurnSpan("assistant", midpoint, len(token_ids)),
            ),
            text=tokenizer.decode(token_ids),
            original_index=index,
        )
        for index in range(4)
    )
    pq.write_table(structured_rows_to_table(rows), path / "shard_00000.parquet")
    row_digest = _aggregate([structured_row_digest(row) for row in rows])
    metadata = {
        "format": STRUCTURED_ROLLOUT_FORMAT,
        "format_version": 1,
        "schema_version": STRUCTURED_ROLLOUT_SCHEMA_VERSION,
        "renderer_version": STRUCTURED_RENDERER_VERSION,
        "tokenizer_sha256": tokenizer_sha256,
        "mixture": "synthetic",
        "split": split,
        "completed_shards": 1,
        "total_shards": 1,
        "total_rows": len(rows),
        "shards": [
            {
                "shard_index": 0,
                "written_row_count": len(rows),
                "row_content_sha256": row_digest,
            }
        ],
        "row_content_sha256": _aggregate([row_digest]),
    }
    (path / "conversion_meta.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )


def _save_base_checkpoint(
    base_dir: Path, *, sequence_len: int = 16, vocab_size: int = 32768,
    head_depth: int = 0, head_block: str = "mlp",
) -> dict[str, t.Tensor]:
    """Shared tiny conditioned checkpoint for actual-trainer integration tests."""
    decoder = GPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    # The encoder deliberately matches the decoder's width so encoder and
    # decoder block shapes coincide. ``muon_step_fused`` / ``adamw_step_fused``
    # are ``@torch.compile(fullgraph=True)`` and recompile once per distinct
    # parameter shape against a default ``recompile_limit`` of 8; a half-width
    # encoder yields 9 distinct Muon shapes and the very first optimizer step
    # dies with ``FailOnRecompileLimitHit``. These dimensions yield 6 Muon and 5
    # AdamW shapes -- the same counts the real 400M conditioned-SFT checkpoint
    # produces -- so this synthetic base stays faithful to production in exactly
    # the property that governs the compile cache.
    encoder = EncoderConfig(
        n_embd=64,
        depth=1,
        n_head=2,
        n_kv_head=2,
        head_dim=32,
    )
    cross_attention = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    feature_prediction = FeaturePredictionConfig(
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
        head_layers=head_depth,
        block=FeaturePredictionBlock(head_block),
    )
    model = EncoderDecoderGPT(
        decoder,
        encoder,
        cross_attention,
        n_sae_features=8,
        num_buckets=2,
        feature_pred_config=feature_prediction,
    )
    model.init_weights()
    # Model a learned SFT head, so a load-time reinitialization is observable.
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("feature_pred_head.residual_blocks."):
                parameter.normal_(std=0.03)
    meta = _metadata()
    meta.update(
        {
            "model_config": asdict(decoder),
            "encoder_config": asdict(encoder),
            "xattn_config": asdict(cross_attention),
            "num_buckets": 2,
            "feature_prediction_config": {
                "block": head_block,
                "head_layers": head_depth,
                "weight": feature_prediction.weight,
                "layer": feature_prediction.layer,
                "loss_mode": feature_prediction.loss_mode.value,
                "loss_fn": feature_prediction.loss_fn.value,
            },
        }
    )
    save_checkpoint(
        str(base_dir),
        1,
        model.state_dict(),
        None,
        meta,
        rank=0,
    )
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@pytest.fixture
def grpo_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Case:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(TOKENIZER_BASE))
    tokenizer = get_tokenizer()
    tokenizer_sha256 = rustbpe_tokenizer_sha256(tokenizer)
    base_dir = tmp_path / "base"
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    _write_artifact(
        train_dir,
        split="train",
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
    )
    _write_artifact(
        val_dir,
        split="val",
        tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_sha256,
    )
    initial = _save_base_checkpoint(base_dir)
    return _Case(
        base_dir=base_dir,
        train_dir=train_dir,
        val_dir=val_dir,
        base_lm_head=initial["lm_head.weight"],
    )


def _run(
    case: _Case,
    output_dir: Path,
    *,
    iterations: int,
    enabled: bool,
    resume: int = -1,
    decay_start: float = 0.8,
    save_every: int = 1,
) -> subprocess.CompletedProcess[str]:
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "scripts.train.conditional_encdec_sft_grpo",
        "--",
        f"base_checkpoint_dir={case.base_dir}",
        "model_step=1",
        f"train_artifact_dir={case.train_dir}",
        f"validation_artifact_dir={case.val_dir}",
        f"output_dir={output_dir}",
        f"resume_from_step={resume}",
        "global_conversations_per_step=1",
        "group_size=2",
        "device_conversation_batch_size=1",
        "trajectory_microbatch_size=1",
        "cursor_prefetch_conversations=0",
        "segment_length=2",
        "min_reward_tokens=2",
        "train_feature_head_only=false",
        f"num_iterations={iterations}",
        "eval_every=0",
        f"save_every={save_every}",
        "device_type=cuda",
    ]
    if enabled:
        # `ntp` is null in the base config, so enabling it means supplying the
        # whole block; Hydra needs `+` to add keys under a null node.
        command += [
            "+ntp.trajectories_per_group=1",
            "+ntp.loss_weight=1.0",
            f"+ntp.decay_start={decay_start}",
            "+ntp.fp_head_lr=null",
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "NANOCHAT_BASE_DIR": str(TOKENIZER_BASE),
            "PYTHONPATH": str(REPO_ROOT),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NANOCHAT_ATTN_IMPL": "sdpa",
        }
    )
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_real_trainer_disabled_and_enabled_resume_and_cross_mode_rejection(
    grpo_case: _Case,
    tmp_path: Path,
) -> None:
    disabled_dir = tmp_path / "disabled"
    disabled = _run(grpo_case, disabled_dir, iterations=2, enabled=False)
    assert disabled.returncode == 0, disabled.stdout + disabled.stderr
    disabled_model, _, _ = load_checkpoint(
        str(disabled_dir),
        2,
        "cpu",
        load_optimizer=False,
        rank=0,
    )
    t.testing.assert_close(
        disabled_model["lm_head.weight"],
        grpo_case.base_lm_head,
        atol=0,
        rtol=0,
    )

    enabled_dir = tmp_path / "enabled"
    enabled = _run(grpo_case, enabled_dir, iterations=2, enabled=True)
    assert enabled.returncode == 0, enabled.stdout + enabled.stderr
    enabled_model, _, _ = load_checkpoint(
        str(enabled_dir),
        2,
        "cpu",
        load_optimizer=False,
        rank=0,
    )
    assert not t.equal(enabled_model["lm_head.weight"], grpo_case.base_lm_head)

    false_to_true = _run(
        grpo_case,
        disabled_dir,
        iterations=2,
        enabled=True,
        resume=1,
    )
    assert false_to_true.returncode != 0
    assert "resume identity mismatch" in false_to_true.stdout + false_to_true.stderr
    true_to_false = _run(
        grpo_case,
        enabled_dir,
        iterations=2,
        enabled=False,
        resume=1,
    )
    assert true_to_false.returncode != 0
    assert "resume identity mismatch" in true_to_false.stdout + true_to_false.stderr


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_real_trainer_decays_the_ntp_weight_through_the_schedule(
    grpo_case: _Case,
    tmp_path: Path,
) -> None:
    """The configured `decay_start` must actually reach the running trainer.

    With `decay_start=0.0` the weight leaves its flat region immediately, so the
    logged per-step values must strictly decrease. They never reach exactly zero:
    decay ends at zero at `progress == 1.0`, which is one step past the last step
    of the run -- a direct consequence of the knob naming when decay BEGINS.
    """
    result = _run(
        grpo_case,
        tmp_path / "annealed",
        iterations=4,
        enabled=True,
        decay_start=0.0,
        save_every=4,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    weights = [
        float(token.removeprefix("w_ntp="))
        for token in (result.stdout + result.stderr).split()
        if token.startswith("w_ntp=")
    ]
    assert len(weights) == 4, weights
    assert weights == sorted(weights, reverse=True) and len(set(weights)) == 4, weights
    assert weights[0] == pytest.approx(1.0)
    assert weights[-1] == pytest.approx(0.25)
