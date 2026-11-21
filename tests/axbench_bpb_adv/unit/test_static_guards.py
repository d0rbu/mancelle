from __future__ import annotations

from pathlib import Path


def test_new_eval_does_not_reference_removed_conditioning_or_completion_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "steering" / "eval" / "axbench_bpb_adv"
    paths = [
        *sorted(package_root.glob("*.py")),
        repo_root / "scripts" / "eval" / "axbench_bpb_adv_eval.py",
        repo_root / "configs" / "eval" / "axbench_bpb_adv" / "axbench_bpb_adv_eval.yaml",
    ]
    forbidden = {
        "EMPTY",
        "empty_conditioning",
        "axbench_adv",
        "steering.eval.axbench_adv",
        "generations",
        "judge_scores",
        "_NoopTokenizer",
        "rollout",
        # The neutral-reference apparatus was removed in the class-margin
        # refactor. None of these symbols may reappear anywhere in the bpb-adv
        # package, CLI, or config. The issue-592 aggregate-only strength-family
        # plot intentionally uses "steering_strength" in artifact names, so that
        # substring is no longer a forbidden proxy for the removed runtime path.
        "neutral_reference",
        "NeutralReference",
        "loss_neutral",
        "loss_steered",
        "baseline_x_mode",
        "BaselineXMode",
        "NEUTRAL_REFERENCE_SCHEMA",
        "prepare_neutral_references",
        "PREPARE_NEUTRAL_REFERENCES",
    }
    offenders: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(repo_root)}:{needle}")
    assert offenders == []
