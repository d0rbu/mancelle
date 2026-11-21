# steering/eval/config_validation.py
"""Config validation for post-generate evaluator flags."""

from __future__ import annotations

from steering.eval.completion_steering_eval import EvalStage

_EARLY_STAGES = frozenset(
    {
        EvalStage.SCAN,
        EvalStage.THRESHOLD,
        EvalStage.SELECT,
        EvalStage.GENERATE,
    }
)


def validate_evaluator_flags(
    stage: EvalStage,
    *,
    run_judge: bool,
    run_feature_analysis: bool,
) -> None:
    """Validate that evaluator flags are consistent with the pipeline stage.

    Raises:
        ValueError: If the flags contradict the selected stage.
    """
    if stage in _EARLY_STAGES:
        if run_judge:
            raise ValueError(
                f"stage={stage.value}: run_judge must be false for early pipeline stages"
            )
        if run_feature_analysis:
            raise ValueError(
                f"stage={stage.value}: run_feature_analysis must be false "
                "for early pipeline stages"
            )
    elif stage is EvalStage.JUDGE:
        if not run_judge:
            raise ValueError("stage=judge: run_judge must be true")
        if run_feature_analysis:
            raise ValueError("stage=judge: run_feature_analysis must be false")
    elif stage is EvalStage.FEATURE_ANALYSIS:
        if run_judge:
            raise ValueError("stage=feature_analysis: run_judge must be false")
        if not run_feature_analysis:
            raise ValueError(
                "stage=feature_analysis: run_feature_analysis must be true"
            )
    # stage=ALL: no constraints on flags
