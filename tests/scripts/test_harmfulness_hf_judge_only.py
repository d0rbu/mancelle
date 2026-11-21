"""Model-free HF judging reuses authenticated responses and current judgments."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from scripts.eval import harmfulness_eval
from steering.eval.harmfulness_artifact import (
    build_harmfulness_responses_artifact,
    write_harmfulness_responses_artifact,
)
from steering.eval.harmfulness_generation import (
    HF_BATCHED_SEED_SCHEDULE,
    build_generation_provenance,
)
from steering.eval.harmfulness_judge import ComplianceJudgeResult, JudgeResult


def test_hf_judge_only_reuses_saved_receipts_without_model_or_client(monkeypatch, tmp_path):
    checkpoint = tmp_path / "step_000123"
    checkpoint.mkdir()
    cfg = OmegaConf.load(Path(harmfulness_eval._CONFIG_DIR) / f"{harmfulness_eval._CONFIG_NAME}.yaml")
    cfg.hf_checkpoint_dir = str(checkpoint)
    cfg.judge_only = True
    cfg.num_harmful = cfg.num_benign = 1
    assert cfg.batch_size == 8
    assert cfg.repository_hf_batch_size == 64
    prompts = [
        {"prompt": "harmful prompt", "pool": "harmful", "category": "Violence"},
        {"prompt": "benign prompt", "pool": "benign", "category": "benign"},
    ]
    pairs: list[dict[str, object]] = [{**prompt, "response": "safe response"} for prompt in prompts]
    conditioning = {"conditioning_mode": "unconditioned", "eval_name": "harmfulness_eval"}
    generation = build_generation_provenance(
        condition_name="unconditional",
        model={"format": "repository_hf_sft", "backend": "pythia",
            "architecture": "GPTNeoXForCausalLM", "model_type": "gpt_neox", "step": 123,
            "checkpoint_sha256": "1" * 64, "tokenizer_sha256": "2" * 64},
        device_type="cpu", inference_dtype="bfloat16", feature_conditions=None,
        target_conditioning_mode="none", max_new_tokens=12, temperature=0.7,
        top_k=50, batch_size=4, seed=42, prompt_count=len(pairs),
        seed_schedule=HF_BATCHED_SEED_SCHEDULE,
    )
    response_path = checkpoint / "report/harmfulness_results_responses/unconditional.yaml"
    result_path = checkpoint / "report/harmfulness_results/unconditional.yaml"
    write_harmfulness_responses_artifact(build_harmfulness_responses_artifact(
        responses=pairs, conditioning=conditioning, generation=generation), response_path)
    # Write a current result using deterministic local judgments.
    local_judgments = MagicMock()
    local_judgments.score_all.return_value = [JudgeResult([], 1, "safe")] * len(pairs)
    local_judgments.score_compliance_all.return_value = [ComplianceJudgeResult(5, False, "helpful")]
    harmfulness_eval._judge_and_write(cfg, "unconditional", str(result_path), pairs,
        local_judgments, conditioning=conditioning, generation=generation)
    saved_bytes = response_path.read_bytes(), result_path.read_bytes()

    monkeypatch.setattr(harmfulness_eval, "_compose_cli_config", lambda argv: cfg)
    monkeypatch.setattr(harmfulness_eval, "_load_expected_prompt_pairs", lambda cfg: prompts)
    forbidden = MagicMock(side_effect=AssertionError("model-free reuse must not load weights or construct a client"))
    monkeypatch.setattr(harmfulness_eval, "load_repository_hf_checkpoint", forbidden)
    monkeypatch.setattr(harmfulness_eval, "_load_model_for_harmfulness", forbidden)
    monkeypatch.setattr(harmfulness_eval, "JudgeClient", forbidden)
    current = MagicMock(wraps=harmfulness_eval.current_harmfulness_payload)
    monkeypatch.setattr(harmfulness_eval, "current_harmfulness_payload", current)

    harmfulness_eval.main()

    forbidden.assert_not_called()
    current.assert_called_once_with(str(result_path), model=str(cfg.judge_model), pairs=pairs,
        expected_conditioning=conditioning, expected_generation=generation)
    assert (response_path.read_bytes(), result_path.read_bytes()) == saved_bytes
