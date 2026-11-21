"""Delphi-style token-level intruder prompts and judge wrapper."""

from __future__ import annotations

import hashlib
import json
import re

from beartype import beartype
from loguru import logger

from steering.eval.autointerp_filter.config import IntruderJudgeConfig
from steering.eval.autointerp_filter.prompts import (
    DEFAULT_INTRUDER_FEW_SHOT_SET,
    DEFAULT_INTRUDER_PROMPT_VARIANT,
    examples_content,
    resolve_intruder_prompt_spec,
)
from steering.eval.autointerp_filter.token_examples import positive_highlight_indices
from steering.eval.autointerp_filter.types import (
    ExampleRenderer,
    IntruderPromptError,
    IntruderResponseParseError,
    IntruderTrial,
    IntruderTrialResult,
    RenderedExample,
)
from steering.eval.axbench_base.llm_judge_client import LLMJudgeClient
from steering.refined_types import PositiveIntLike

_RESPONSE_RE = re.compile(r"\[RESPONSE\]\s*:\s*(?P<index>-?\d+)")


@beartype
def parse_intruder_response(text: str, *, n_examples: PositiveIntLike) -> int:
    """Parse ``[RESPONSE]: N`` and range-check it."""
    matches = list(_RESPONSE_RE.finditer(text))
    if not matches:
        raise IntruderResponseParseError("intruder judge response missing `[RESPONSE]: <index>`")
    try:
        value = int(matches[-1].group("index"))
    except ValueError:
        raise IntruderResponseParseError(
            "intruder judge response index is not an integer"
        ) from None
    if value < 0 or value >= n_examples:
        raise IntruderResponseParseError(
            f"intruder judge response index {value} out of range [0, {n_examples})"
        )
    return value


def render_trial_examples(
    trial: IntruderTrial,
    renderer: ExampleRenderer,
) -> list[RenderedExample]:
    """Render trial refs with the intruder inserted at ``intruder_index``."""
    positives = [
        renderer.render(
            ref,
            activating=True,
            highlighted_token_indices=positive_highlight_indices(
                ref,
                activation_threshold_ratio=trial.activation_threshold_ratio,
            ),
        )
        for ref in trial.positive_refs
    ]
    intruder = renderer.render(
        trial.intruder_ref,
        activating=False,
        highlighted_token_indices=trial.intruder_highlight_indices,
    )
    return positives[: trial.intruder_index] + [intruder] + positives[trial.intruder_index :]


def build_intruder_messages(
    examples: list[RenderedExample],
    *,
    cot: bool = False,
    prompt_variant: str = DEFAULT_INTRUDER_PROMPT_VARIANT,
    few_shot_set: str = DEFAULT_INTRUDER_FEW_SHOT_SET,
) -> list[dict[str, str]]:
    """Build chat messages for one intruder trial."""
    return build_intruder_messages_from_texts(
        [example.text for example in examples],
        cot=cot,
        prompt_variant=prompt_variant,
        few_shot_set=few_shot_set,
    )


def build_intruder_messages_from_texts(
    example_texts: list[str],
    *,
    cot: bool = False,
    prompt_variant: str = DEFAULT_INTRUDER_PROMPT_VARIANT,
    few_shot_set: str = DEFAULT_INTRUDER_FEW_SHOT_SET,
) -> list[dict[str, str]]:
    """Build chat messages for one intruder trial."""
    spec = resolve_intruder_prompt_spec(
        prompt_variant=prompt_variant,
        few_shot_set=few_shot_set,
    )
    content = examples_content(example_texts)
    few_shot_messages = spec.few_shot_cot_messages if cot else spec.few_shot_messages
    return [
        {"role": "system", "content": spec.system_prompt},
        *(dict(message) for message in few_shot_messages),
        {"role": "user", "content": content},
    ]


def judge_cache_key(config: IntruderJudgeConfig, *, render_context: dict) -> str:
    """Stable identity for parsed trial-result cache entries."""
    payload = {
        "method": "intruder_detection_token_v1",
        "prompt": resolve_intruder_prompt_spec(
            prompt_variant=config.prompt_variant,
            few_shot_set=config.few_shot_set,
        ).cache_payload(),
        "judge": {
            "model": config.model,
            "cot": config.cot,
        },
        "render_context": render_context,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class IntruderDetectionJudge:
    """Synchronous LLM judge for intruder trials."""

    def __init__(self, client: LLMJudgeClient, config: IntruderJudgeConfig) -> None:
        self._client = client
        self._cfg = config

    def evaluate(
        self,
        trial: IntruderTrial,
        *,
        renderer: ExampleRenderer,
    ) -> IntruderTrialResult:
        examples = render_trial_examples(trial, renderer)
        messages = build_intruder_messages(
            examples,
            cot=self._cfg.cot,
            prompt_variant=self._cfg.prompt_variant,
            few_shot_set=self._cfg.few_shot_set,
        )
        prompt_chars = sum(len(m["content"]) for m in messages)
        if prompt_chars > self._cfg.max_prompt_chars:
            raise IntruderPromptError(
                f"Intruder prompt {prompt_chars} chars > "
                f"max_prompt_chars={self._cfg.max_prompt_chars}"
            )

        n_examples = len(examples)
        final_user_prompt = messages[-1]["content"]
        last_error: IntruderResponseParseError | None = None
        parse_failures: list[tuple[int, str, str]] = []
        max_attempts = self._cfg.parse_max_retries + 1
        for attempt in range(max_attempts):
            response = self._client.complete(messages, cache_key=None)
            try:
                prediction = parse_intruder_response(response, n_examples=n_examples)
            except IntruderResponseParseError as exc:
                last_error = exc
                parse_failures.append((attempt + 1, str(exc), response))
                logger.warning(
                    f"Could not parse intruder response "
                    f"(attempt {attempt + 1}/{max_attempts}); "
                    f"{exc}; full_response={response!r}"
                )
                continue
            return IntruderTrialResult(
                trial_key=trial.trial_key,
                prediction=prediction,
                correct_index=trial.intruder_index,
                correct=prediction == trial.intruder_index,
                response=response,
                cached=False,
                error_message=None,
                rendered_examples=tuple(examples),
                final_user_prompt=final_user_prompt,
            )

        failure_details = "; ".join(
            f"attempt {attempt}: {reason}; full_response={response!r}"
            for attempt, reason, response in parse_failures
        )
        raise IntruderResponseParseError(
            f"could not parse intruder response after {max_attempts} attempts: {failure_details}"
        ) from last_error
