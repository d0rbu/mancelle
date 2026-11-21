"""Replay trial-audit rows against prompt variants for calibration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger

from nanochat.atomic_io import atomic_write_text
from steering.eval.autointerp_filter.artifacts import dump_human_yaml
from steering.eval.autointerp_filter.config import (
    IntruderJudgeConfig,
    validate_intruder_judge_config,
)
from steering.eval.autointerp_filter.intruder import (
    build_intruder_messages_from_texts,
    parse_intruder_response,
)
from steering.eval.autointerp_filter.prompts import resolve_intruder_prompt_spec
from steering.eval.autointerp_filter.types import (
    AutoInterpConfigError,
    IntruderPromptError,
    IntruderResponseParseError,
)
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import LLMJudgeClient, OpenRouterLLMClient
from steering.refined_types import (
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
    UnitInterval,
    UnitIntervalLike,
)


@dataclass(frozen=True, init=False)
class PromptCalibrationConfig:
    trial_audit_path: NonEmptyStr
    output_path: NonEmptyStr
    json_output_path: NonEmptyStr | None
    max_trials: PositiveInt | None
    fail_on_parse_error: bool = True
    progress_interval: NonNegativeInt = NonNegativeInt.parse(10)
    checkpoint_interval: NonNegativeInt | None = NonNegativeInt.parse(10)
    judge: IntruderJudgeConfig = field(default_factory=IntruderJudgeConfig)

    @beartype
    def __init__(
        self,
        trial_audit_path: NonEmptyStrLike,
        output_path: NonEmptyStrLike,
        json_output_path: NonEmptyStrLike | None = None,
        max_trials: PositiveIntLike | None = None,
        fail_on_parse_error: bool = True,
        progress_interval: NonNegativeIntLike = 10,
        checkpoint_interval: NonNegativeIntLike | None = 10,
        judge: IntruderJudgeConfig | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "trial_audit_path",
            NonEmptyStr.parse(trial_audit_path),
        )
        object.__setattr__(self, "output_path", NonEmptyStr.parse(output_path))
        object.__setattr__(
            self,
            "json_output_path",
            None if json_output_path is None else NonEmptyStr.parse(json_output_path),
        )
        object.__setattr__(
            self,
            "max_trials",
            None if max_trials is None else PositiveInt.parse(max_trials),
        )
        object.__setattr__(self, "fail_on_parse_error", fail_on_parse_error)
        object.__setattr__(
            self,
            "progress_interval",
            NonNegativeInt.parse(progress_interval),
        )
        object.__setattr__(
            self,
            "checkpoint_interval",
            None if checkpoint_interval is None else NonNegativeInt.parse(checkpoint_interval),
        )
        object.__setattr__(
            self,
            "judge",
            IntruderJudgeConfig() if judge is None else judge,
        )


@dataclass(frozen=True, init=False)
class PromptCalibrationReport:
    source_trial_audit_path: NonEmptyStr
    output_path: NonEmptyStr
    json_output_path: NonEmptyStr | None
    model: NonEmptyStr
    prompt_variant: NonEmptyStr
    few_shot_set: NonEmptyStr
    prompt_version: NonEmptyStr
    cot: bool
    n_trials: NonNegativeInt
    n_trials_total: NonNegativeInt
    n_correct: NonNegativeInt
    n_parse_errors: NonNegativeInt
    accuracy: UnitInterval
    complete: bool

    @beartype
    def __init__(
        self,
        source_trial_audit_path: NonEmptyStrLike,
        output_path: NonEmptyStrLike,
        json_output_path: NonEmptyStrLike | None,
        model: NonEmptyStrLike,
        prompt_variant: NonEmptyStrLike,
        few_shot_set: NonEmptyStrLike,
        prompt_version: NonEmptyStrLike,
        cot: bool,
        n_trials: NonNegativeIntLike,
        n_trials_total: NonNegativeIntLike,
        n_correct: NonNegativeIntLike,
        n_parse_errors: NonNegativeIntLike,
        accuracy: UnitIntervalLike,
        complete: bool,
    ) -> None:
        object.__setattr__(
            self,
            "source_trial_audit_path",
            NonEmptyStr.parse(source_trial_audit_path),
        )
        object.__setattr__(self, "output_path", NonEmptyStr.parse(output_path))
        object.__setattr__(
            self,
            "json_output_path",
            None if json_output_path is None else NonEmptyStr.parse(json_output_path),
        )
        object.__setattr__(self, "model", NonEmptyStr.parse(model))
        object.__setattr__(
            self,
            "prompt_variant",
            NonEmptyStr.parse(prompt_variant),
        )
        object.__setattr__(
            self,
            "few_shot_set",
            NonEmptyStr.parse(few_shot_set),
        )
        object.__setattr__(
            self,
            "prompt_version",
            NonEmptyStr.parse(prompt_version),
        )
        object.__setattr__(self, "cot", cot)
        object.__setattr__(self, "n_trials", NonNegativeInt.parse(n_trials))
        object.__setattr__(
            self,
            "n_trials_total",
            NonNegativeInt.parse(n_trials_total),
        )
        object.__setattr__(self, "n_correct", NonNegativeInt.parse(n_correct))
        object.__setattr__(
            self,
            "n_parse_errors",
            NonNegativeInt.parse(n_parse_errors),
        )
        object.__setattr__(self, "accuracy", UnitInterval.parse(accuracy))
        object.__setattr__(self, "complete", complete)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.n_trials > self.n_trials_total:
            raise AutoInterpConfigError("prompt calibration n_trials exceeds n_trials_total")
        if self.n_correct > self.n_trials or self.n_parse_errors > self.n_trials:
            raise AutoInterpConfigError("prompt calibration result counts exceed n_trials")


def validate_prompt_calibration_config(cfg: PromptCalibrationConfig) -> None:
    validate_intruder_judge_config(cfg.judge)


def build_prompt_calibration_config(raw: dict[str, Any]) -> PromptCalibrationConfig:
    resolved = dict(raw)
    judge_raw = resolved.get("judge", {})
    if not isinstance(judge_raw, dict):
        raise AutoInterpConfigError("judge must be a mapping")
    try:
        resolved["judge"] = IntruderJudgeConfig(**judge_raw)
        return PromptCalibrationConfig(**resolved)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AutoInterpConfigError(f"prompt calibration config has invalid fields: {exc}") from exc


def run_prompt_calibration_from_config(raw: dict[str, Any]) -> PromptCalibrationReport:
    cfg = build_prompt_calibration_config(raw)
    validate_prompt_calibration_config(cfg)
    llm_cfg = LLMClientConfig(
        provider="openrouter",
        model=cfg.judge.model,
        requests_per_minute=cfg.judge.requests_per_minute,
    )
    return run_prompt_calibration(cfg, OpenRouterLLMClient(llm_cfg))


def run_prompt_calibration(
    cfg: PromptCalibrationConfig,
    client: LLMJudgeClient,
) -> PromptCalibrationReport:
    validate_prompt_calibration_config(cfg)
    rows = _load_trial_audit_rows(Path(cfg.trial_audit_path))
    if cfg.max_trials is not None:
        rows = rows[: cfg.max_trials]
    if not rows:
        raise AutoInterpConfigError("trial_audit_path must contain at least one trial row")
    prompt_spec = resolve_intruder_prompt_spec(
        prompt_variant=cfg.judge.prompt_variant,
        few_shot_set=cfg.judge.few_shot_set,
    )

    result_rows: list[dict[str, Any]] = []
    n_correct = 0
    n_parse_errors = 0
    n_total = len(rows)
    for row_index, row in enumerate(rows):
        result = _score_audit_row(row_index, row, cfg=cfg, client=client)
        result_rows.append(result)
        if result["correct"] is True:
            n_correct += 1
        if result["error_message"] is not None:
            n_parse_errors += 1
        n_done = row_index + 1
        if cfg.progress_interval and (n_done % cfg.progress_interval == 0 or n_done == n_total):
            logger.info(
                "Autointerp prompt calibration progress: "
                f"{n_done}/{n_total} rows, "
                f"accuracy={n_correct / n_done:.3f}, "
                f"parse_errors={n_parse_errors}"
            )
        if cfg.checkpoint_interval and (n_done % cfg.checkpoint_interval == 0 or n_done == n_total):
            _write_calibration_outputs(
                cfg=cfg,
                result_rows=result_rows,
                prompt_version=prompt_spec.version,
                complete=n_done == n_total,
                n_trials_total=n_total,
            )

    report = _write_calibration_outputs(
        cfg=cfg,
        result_rows=result_rows,
        prompt_version=prompt_spec.version,
        complete=True,
        n_trials_total=n_total,
    )
    logger.info(
        "Autointerp prompt calibration complete: "
        f"accuracy={report.accuracy:.3f} "
        f"n_trials={report.n_trials} parse_errors={report.n_parse_errors}"
    )
    return report


def _write_calibration_outputs(
    *,
    cfg: PromptCalibrationConfig,
    result_rows: list[dict[str, Any]],
    prompt_version: str,
    complete: bool,
    n_trials_total: int,
) -> PromptCalibrationReport:
    n_trials = len(result_rows)
    n_correct = sum(1 for row in result_rows if row["correct"] is True)
    n_parse_errors = sum(1 for row in result_rows if row["error_message"] is not None)
    report = PromptCalibrationReport(
        source_trial_audit_path=cfg.trial_audit_path,
        output_path=cfg.output_path,
        json_output_path=cfg.json_output_path,
        model=cfg.judge.model,
        prompt_variant=cfg.judge.prompt_variant,
        few_shot_set=cfg.judge.few_shot_set,
        prompt_version=prompt_version,
        cot=cfg.judge.cot,
        n_trials=n_trials,
        n_trials_total=n_trials_total,
        n_correct=n_correct,
        n_parse_errors=n_parse_errors,
        accuracy=0.0 if n_trials == 0 else n_correct / n_trials,
        complete=complete,
    )
    payload = {
        **asdict(report),
        "rows": result_rows,
    }
    atomic_write_text(dump_human_yaml(payload), Path(cfg.output_path))
    if cfg.json_output_path is not None:
        atomic_write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            Path(cfg.json_output_path),
        )
    return report


def _load_trial_audit_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AutoInterpConfigError(f"trial_audit_path does not exist: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, list):
        raise AutoInterpConfigError("trial_audit_path must contain a YAML list")
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise AutoInterpConfigError(f"trial audit row {row_index} is not a mapping")
        typed_row = cast(dict[str, Any], row)
        _validate_audit_row(row_index, typed_row)
        rows.append(typed_row)
    return rows


def _validate_audit_row(row_index: int, row: dict[str, Any]) -> None:
    examples = row.get("examples")
    if not isinstance(examples, list) or not examples:
        raise AutoInterpConfigError(f"trial audit row {row_index} has no examples")
    for example_index, example in enumerate(examples):
        if not isinstance(example, dict) or not isinstance(example.get("text"), str):
            raise AutoInterpConfigError(
                f"trial audit row {row_index} example {example_index} is missing text"
            )
    correct_index = row.get("correct_index")
    if isinstance(correct_index, bool) or not isinstance(correct_index, int):
        raise AutoInterpConfigError(f"trial audit row {row_index} correct_index must be an int")
    if correct_index < 0 or correct_index >= len(examples):
        raise AutoInterpConfigError(
            f"trial audit row {row_index} correct_index={correct_index} out of range"
        )
    prediction = row.get("prediction")
    if prediction is not None:
        if isinstance(prediction, bool) or not isinstance(prediction, int):
            raise AutoInterpConfigError(
                f"trial audit row {row_index} prediction must be an int when set"
            )
        if prediction < 0 or prediction >= len(examples):
            raise AutoInterpConfigError(
                f"trial audit row {row_index} prediction={prediction} out of range"
            )


def _score_audit_row(
    row_index: int,
    row: dict[str, Any],
    *,
    cfg: PromptCalibrationConfig,
    client: LLMJudgeClient,
) -> dict[str, Any]:
    example_texts = [str(example["text"]) for example in row["examples"]]
    messages = build_intruder_messages_from_texts(
        example_texts,
        cot=cfg.judge.cot,
        prompt_variant=cfg.judge.prompt_variant,
        few_shot_set=cfg.judge.few_shot_set,
    )
    prompt_chars = sum(len(message["content"]) for message in messages)
    if prompt_chars > cfg.judge.max_prompt_chars:
        raise IntruderPromptError(
            f"Intruder prompt {prompt_chars} chars > max_prompt_chars={cfg.judge.max_prompt_chars}"
        )

    prediction: int | None = None
    response: str | None = None
    error_message: str | None = None
    last_error: IntruderResponseParseError | None = None
    for attempt in range(cfg.judge.parse_max_retries + 1):
        response = client.complete(messages, cache_key=None)
        try:
            prediction = parse_intruder_response(response, n_examples=len(example_texts))
            error_message = None
            break
        except IntruderResponseParseError as exc:
            last_error = exc
            error_message = f"attempt {attempt + 1}: {exc}; full_response={response!r}"
            logger.warning(f"Could not parse prompt calibration response: {error_message}")
    if prediction is None and cfg.fail_on_parse_error:
        raise IntruderResponseParseError(
            error_message or "could not parse response"
        ) from last_error

    correct_index = int(row["correct_index"])
    correct = None if prediction is None else prediction == correct_index
    baseline_prediction = row.get("prediction")
    baseline_correct = (
        None if baseline_prediction is None else int(baseline_prediction) == correct_index
    )
    return {
        "row_index": row_index,
        "display_name": row.get("display_name"),
        "feature_ids": row.get("feature_ids"),
        "quantile": row.get("quantile"),
        "correct_index": correct_index,
        "baseline_prediction": baseline_prediction,
        "baseline_correct": baseline_correct,
        "prediction": prediction,
        "correct": correct,
        "response": response,
        "error_message": error_message,
    }
