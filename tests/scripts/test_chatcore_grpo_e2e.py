"""CUDA qualification of the actual ChatCORE preset, converter, and trainer."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch as t

from nanochat.checkpoint_manager import load_checkpoint
from scripts.preprocess.convert_sft_to_parquet import main as convert_main
from tests.chatcore_fixtures import _benchmark_dataset, _converter_config, _load_artifact
from tests.chatcore_fixtures import rustbpe_tokenizer as rustbpe_tokenizer
from tests.scripts.test_conditional_encdec_sft_grpo_e2e import _save_base_checkpoint

_ROOT = Path(__file__).resolve().parents[2]


def _run(
    base: Path,
    artifacts: Path,
    output: Path,
    cache: Path,
    *,
    resume: int = -1,
    head_only: bool | None = None,
    expected_failure: str | None = None,
):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        "-m",
        "scripts.train.conditional_encdec_sft_grpo",
        "--",
        "config_name=conditional_encdec_sft_grpo_chatcore",
        f"base_checkpoint_dir={base}",
        "model_step=1",
        f"train_artifact_dir={artifacts}",
        f"validation_artifact_dir={artifacts}_val",
        f"output_dir={output}",
        f"resume_from_step={resume}",
        "global_conversations_per_step=2",
        "group_size=4",
        "device_conversation_batch_size=1",
        "trajectory_microbatch_size=1",
        "kl_beta=0.2",
        "num_iterations=2",
        "eval_every=1",
        "validation_conversations=2",
        "save_every=1",
        "device_type=cuda",
    ]
    if head_only is not None:
        command.append(f"train_feature_head_only={str(head_only).lower()}")
    environment = {
        **os.environ,
        "NANOCHAT_BASE_DIR": str(cache),
        "PYTHONPATH": str(_ROOT),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NANOCHAT_ATTN_IMPL": "sdpa",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    # Kill the whole test-owned process group on timeout, including torchrun's
    # workers and checkpoint re-execs, so failed tests cannot leave GPU jobs.
    with subprocess.Popen(
        command,
        cwd=_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            stdout, _ = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
            pytest.fail(f"ChatCORE GPU trainer exceeded 600 seconds:\n{stdout}")
        if expected_failure is None:
            assert process.returncode == 0, stdout
        else:
            assert process.returncode != 0 and expected_failure in stdout, stdout
    return stdout


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("head_only", [False, True])
def test_chatcore_real_trainer_updates_validates_and_resumes_exactly(
    tmp_path,
    monkeypatch,
    rustbpe_tokenizer,
    head_only,
):
    tokenizer = rustbpe_tokenizer
    cache = tmp_path / "cache"
    tokenizer.save(cache / "tokenizer")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(cache))
    for module in ("tasks.arc", "tasks.mmlu", "tasks.gsm8k"):
        monkeypatch.setattr(f"{module}.load_dataset", _benchmark_dataset)
    monkeypatch.setattr(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer", lambda: tokenizer
    )
    artifacts = tmp_path / "chatcore"
    convert_main.__wrapped__(_converter_config(artifacts, conversations_per_shard=2))
    for split, directory in (("train", artifacts), ("val", Path(f"{artifacts}_val"))):
        artifact = _load_artifact(directory, tokenizer, split)
        assert len(artifact.rows) == 4
        assert sum(sum(row.target_mask) == 2 for row in artifact.rows) == 3
        assert max(len(row.target_token_ids) for row in artifact.rows) < 256

    base = tmp_path / "base"
    with t.random.fork_rng(devices=[]):
        t.manual_seed(42)
        initial = _save_base_checkpoint(
            base, sequence_len=256, vocab_size=tokenizer.get_vocab_size()
        )
    output = tmp_path / "output"
    scope_override = None if head_only else False  # Exercise the inherited default and opt-out.
    log = _run(base, artifacts, output, cache, head_only=scope_override)
    assert "GRPO validation step=1" in log and "GRPO validation step=2" in log
    assert f"train_feature_head_only={str(head_only).lower()}" in log
    assert "trainable_parameters=" in log and "total_parameters=" in log
    trained, optimizer, meta = load_checkpoint(str(output), 2, "cpu", load_optimizer=True, rank=0)
    assert optimizer is not None
    assert any(not t.equal(trained[key], value) for key, value in initial.items())
    assert not t.equal(trained["feature_pred_head.weight"], initial["feature_pred_head.weight"])
    assert t.equal(trained["lm_head.weight"], initial["lm_head.weight"])
    if head_only:
        for name, value in initial.items():
            if not name.startswith("feature_pred_head."):
                assert t.equal(trained[name], value), name
        assert len(optimizer["param_groups"]) == 1
        assert len(optimizer["state"]) == 2
    loop = meta["structured_sft_grpo_v1"]
    for name in ("train_cursor", "validation_cursor"):
        states = loop[name]["rank_states"]
        assert len(states) == 1
        assert states[0]["accepted"] == states[0]["attempted"] == 4
    config = loop["user_config"]
    assert config["segmentation_mode"] == "turn_delimited_fixed_length"
    assert config["min_reward_tokens"] == 1
    assert config["ntp"] is None
    assert config["kl_beta"] == 0.2
    assert config["train_feature_head_only"] is head_only
    if head_only:
        assert loop["resume_identity"]["parameter_manifest"] == [
            "feature_pred_head.weight",
            "feature_pred_head.bias",
        ]

    # A fresh worker resumes the durable step-1 checkpoint and must recreate
    # every model/optimizer tensor and both cursors, not just similar losses.
    resumed_log = _run(base, artifacts, output, cache, resume=1, head_only=scope_override)
    assert "GRPO validation step=2" in resumed_log
    replayed, replay_optimizer, replay_meta = load_checkpoint(
        str(output),
        2,
        "cpu",
        load_optimizer=True,
        rank=0,
    )
    t.testing.assert_close(replayed, trained, rtol=0, atol=0)
    assert replay_optimizer is not None
    assert replay_optimizer.keys() == optimizer.keys() == {"state", "param_groups"}
    assert replay_optimizer["param_groups"] == optimizer["param_groups"]
    t.testing.assert_close(replay_optimizer["state"], optimizer["state"], rtol=0, atol=0)
    for key in ("resume_identity", "train_cursor", "validation_cursor", "completed_optimizer_step"):
        assert replay_meta["structured_sft_grpo_v1"][key] == loop[key]
    # Neither direction may restore incompatible optimizer groups or write a checkpoint.
    _run(
        base,
        artifacts,
        output,
        cache,
        resume=1,
        head_only=not head_only,
        expected_failure="trainable parameter manifest differs",
    )
    after_failure, after_optimizer, after_meta = load_checkpoint(
        str(output),
        2,
        "cpu",
        load_optimizer=True,
        rank=0,
    )
    t.testing.assert_close(after_failure, trained, rtol=0, atol=0)
    assert after_optimizer is not None
    assert after_optimizer["param_groups"] == optimizer["param_groups"]
    t.testing.assert_close(after_optimizer["state"], optimizer["state"], rtol=0, atol=0)
    assert after_meta == replay_meta
