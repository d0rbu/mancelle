"""Strict recovery of vendored AxBench prompt-steering text."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import pandas as pd


class BaselineTargetModel(StrEnum):
    """Baseline targets whose exact AxBench chat wrappers are supported."""

    QWEN2_5_3B_INSTRUCT = "Qwen/Qwen2.5-3B-Instruct"
    SMOLLM3_3B_CHECKPOINTS = "HuggingFaceTB/SmolLM3-3B-checkpoints"
    SMOLLM3_3B = "HuggingFaceTB/SmolLM3-3B"
    PYTHIA_2_8B_STANDARD_STEERED_SFT = "Pythia-2.8B standard_steered SFT export"


QWEN_STEERED_PREFIX = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba "
    "Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
)
QWEN_STEERED_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
SMOLLM3_STEERED_PREFIX = (
    "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\n"
    "Today Date: 27 August 2026\nReasoning Mode: /think\n\n## Custom Instructions\n\n"
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an "
    "assistant involves thoroughly exploring questions through a systematic thinking process "
    "before providing the final precise and accurate solutions. This requires engaging in a "
    "comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
    "backtracking, and iteration to develop well-considered thinking process. Please structure "
    "your response into two main sections: Thought and Solution using the specified format: "
    "<think> Thought section </think> Solution section. In the Thought section, detail your "
    "reasoning process in steps. Each step should include detailed considerations such as "
    "analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the "
    "accuracy of the current steps, refining any errors, and revisiting previous steps. In the "
    "Solution section, based on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem correct. The Solution "
    "section should be logical, accurate, and concise and detail necessary steps needed to reach "
    "the conclusion.\n\n<|im_start|>user\n"
)
SMOLLM3_STEERED_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
QUESTION_MARKER = "\n\nQuestion: "
SMOLLM3_NO_THINK_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
_WRAPPER_SENTINEL = "__AXBENCH_ADV_EXACT_PROMPT_CONTENT_7D29A4F1__"


@dataclass(frozen=True)
class PromptWrapper:
    """Exact tokenizer-rendered wrapper selected by baseline provenance."""

    model: BaselineTargetModel
    revision: str | None
    prefix: str
    suffix: str
    error_model_name: str
    source_model_name: str | None = None

    @property
    def detected_model(self) -> str:
        """Return the model identity as recorded in failure/provenance surfaces."""

        model_name = self.model.value if self.source_model_name is None else self.source_model_name
        if self.revision is None:
            return model_name
        return f"{model_name}@{self.revision}"


QWEN_PROMPT_WRAPPER = PromptWrapper(
    model=BaselineTargetModel.QWEN2_5_3B_INSTRUCT,
    revision=None,
    prefix=QWEN_STEERED_PREFIX,
    suffix=QWEN_STEERED_SUFFIX,
    # Preserve every existing Qwen mismatch message byte-for-byte.
    error_model_name="Qwen",
)
SMOLLM3_PROMPT_WRAPPER = PromptWrapper(
    model=BaselineTargetModel.SMOLLM3_3B_CHECKPOINTS,
    revision="it-soup-APO",
    prefix=SMOLLM3_STEERED_PREFIX,
    suffix=SMOLLM3_STEERED_SUFFIX,
    error_model_name="HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO",
)
_PROMPT_WRAPPERS = {
    (wrapper.model.value, wrapper.revision): wrapper
    for wrapper in (QWEN_PROMPT_WRAPPER, SMOLLM3_PROMPT_WRAPPER)
}


def source_manifest_path(frame_path: Path) -> Path:
    """Return the AxBench run manifest adjacent to an inference directory."""

    return Path(frame_path).parent.parent / "run_manifest.json"


def _mapping_value(
    value: object,
    *,
    surface: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{surface} must be a string-keyed object")
    return cast(Mapping[str, Any], value)


def _source_identity_from_manifest(manifest_path: Path) -> tuple[str, str | None]:
    manifest_path = Path(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read AxBench source manifest {manifest_path}: {exc}") from exc
    manifest = _mapping_value(payload, surface=f"AxBench source manifest {manifest_path}")
    config = _mapping_value(
        manifest.get("config"),
        surface=f"AxBench source manifest {manifest_path} config",
    )
    train = _mapping_value(
        config.get("train"),
        surface=f"AxBench source manifest {manifest_path} config.train",
    )
    inference = _mapping_value(
        config.get("inference"),
        surface=f"AxBench source manifest {manifest_path} config.inference",
    )

    model_fields = {
        "target_model": manifest.get("target_model"),
        "config.train.model_name": train.get("model_name"),
        "config.inference.model_name": inference.get("model_name"),
        "config.inference.steering_model_name": inference.get("steering_model_name"),
    }
    if not all(isinstance(value, str) and value for value in model_fields.values()):
        raise ValueError(
            f"AxBench source manifest {manifest_path} has missing or invalid model provenance: "
            f"{model_fields}"
        )
    model_values = {cast(str, value) for value in model_fields.values()}
    if len(model_values) != 1:
        raise ValueError(
            f"AxBench source manifest {manifest_path} has inconsistent model provenance: "
            f"{model_fields}"
        )
    model_name = next(iter(model_values))

    revision_fields = {
        "config.train.model_revision": train.get("model_revision"),
        "config.inference.model_revision": inference.get("model_revision"),
        "config.inference.steering_model_revision": inference.get("steering_model_revision"),
    }
    if not all(
        value is None or isinstance(value, str) and value for value in revision_fields.values()
    ):
        raise ValueError(
            f"AxBench source manifest {manifest_path} has invalid revision provenance: "
            f"{revision_fields}"
        )
    if len(set(revision_fields.values())) != 1:
        raise ValueError(
            f"AxBench source manifest {manifest_path} has inconsistent revision provenance: "
            f"{revision_fields}"
        )
    revision = cast(str | None, next(iter(revision_fields.values())))

    return model_name, revision


def prompt_wrapper_from_manifest(manifest_path: Path) -> PromptWrapper:
    """Resolve one exact statically registered wrapper from a vendored-run manifest."""

    manifest_path = Path(manifest_path)
    model_name, revision = _source_identity_from_manifest(manifest_path)
    wrapper = _PROMPT_WRAPPERS.get((model_name, revision))
    if wrapper is None:
        detected_model = model_name if revision is None else f"{model_name}@{revision}"
        raise ValueError(
            f"AxBench source manifest {manifest_path} names unsupported prompt-wrapper model "
            f"{detected_model}"
        )
    return wrapper


def rendered_prompt_wrapper_from_manifest(
    manifest_path: Path,
    *,
    tokenizer: Any,
    model_type: str,
) -> PromptWrapper:
    """Render the exact runtime wrapper for a supported source host."""

    manifest_path = Path(manifest_path)
    model_name, revision = _source_identity_from_manifest(manifest_path)
    hosts = {
        (BaselineTargetModel.QWEN2_5_3B_INSTRUCT, None, "qwen2"),
        (BaselineTargetModel.SMOLLM3_3B_CHECKPOINTS, "it-soup-APO", "smollm3"),
        (BaselineTargetModel.SMOLLM3_3B, SMOLLM3_NO_THINK_REVISION, "smollm3"),
    }
    if (model_name, revision, model_type) in hosts:
        model = BaselineTargetModel(model_name)
    elif model_type == "gpt_neox" and revision is None and Path(model_name).is_absolute():
        model = BaselineTargetModel.PYTHIA_2_8B_STANDARD_STEERED_SFT
    else:
        detected_model = model_name if revision is None else f"{model_name}@{revision}"
        raise ValueError(
            f"AxBench source manifest {manifest_path} names unsupported rendered "
            f"prompt-wrapper model {detected_model} with model_type={model_type!r}"
        )

    template_kwargs = {"enable_thinking": False} if model is BaselineTargetModel.SMOLLM3_3B else {}
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": _WRAPPER_SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    except Exception as exc:
        raise ValueError(f"could not render prompt wrapper for {model_name}: {exc}") from exc
    if not isinstance(rendered, str) or rendered.count(_WRAPPER_SENTINEL) != 1:
        raise ValueError(
            f"rendered prompt wrapper for {model_name} must contain the sentinel exactly once"
        )
    # AxBench removes one emitted BOS before saving its rendered source prompts.
    bos_token = getattr(tokenizer, "bos_token", None)
    if bos_token and rendered.startswith(bos_token):
        rendered = rendered[len(bos_token) :]
    prefix, suffix = rendered.split(_WRAPPER_SENTINEL)
    if not prefix or not suffix:
        raise ValueError(f"rendered prompt wrapper for {model_name} has an empty prefix or suffix")
    return PromptWrapper(
        model=model,
        revision=revision,
        prefix=prefix,
        suffix=suffix,
        error_model_name=model_name,
        source_model_name=model_name,
    )


def prompt_wrapper_for_frame(frame_path: Path) -> PromptWrapper:
    """Resolve a frame's exact prompt wrapper from its adjacent run manifest."""

    return prompt_wrapper_from_manifest(source_manifest_path(frame_path))


def recover_steered_prompt(
    steered_input: str,
    original_prompt: str,
    *,
    surface: str,
    wrapper: PromptWrapper = QWEN_PROMPT_WRAPPER,
) -> str:
    """Recover raw steering text after validating the provenance-selected wrapper."""

    if not isinstance(steered_input, str):
        raise ValueError(f"{surface}: steered_input must be a string")
    if not steered_input.startswith(wrapper.prefix):
        raise ValueError(
            f"{surface}: steered_input does not have the exact {wrapper.error_model_name} prefix"
        )
    if not steered_input.endswith(wrapper.suffix):
        raise ValueError(
            f"{surface}: steered_input does not have the exact {wrapper.error_model_name} suffix"
        )

    raw_steered_text = steered_input[len(wrapper.prefix) : -len(wrapper.suffix)]
    question_tail = QUESTION_MARKER + original_prompt
    if not raw_steered_text.endswith(question_tail):
        raise ValueError(
            f"{surface}: steered_input does not end with the question marker and original_prompt"
        )
    steering_prompt = raw_steered_text[: -len(question_tail)]
    if not steering_prompt.strip():
        raise ValueError(f"{surface}: steering prompt must be nonempty")
    return raw_steered_text


def recover_concept_steering_prompts(
    aux_frame: pd.DataFrame,
    *,
    max_concepts: int,
    wrapper: PromptWrapper = QWEN_PROMPT_WRAPPER,
) -> dict[int, str]:
    """Recover the single shared steering prompt required for each concept."""

    recovered: dict[int, str] = {}
    for concept_id in range(max_concepts):
        concept_rows = aux_frame.loc[aux_frame["concept_id"].eq(concept_id)]
        if concept_rows.empty:
            raise ValueError(f"concept {concept_id}: no auxiliary rows")

        steering_prompts: set[str] = set()
        for row_index, row in enumerate(concept_rows.itertuples(index=False)):
            input_id = getattr(row, "input_id", row_index)
            raw_steered_text = recover_steered_prompt(
                row.steered_input,
                row.original_prompt,
                surface=f"concept {concept_id} input {input_id}",
                wrapper=wrapper,
            )
            question_tail = QUESTION_MARKER + row.original_prompt
            steering_prompts.add(raw_steered_text[: -len(question_tail)])

        if len(steering_prompts) != 1:
            raise ValueError(
                f"concept {concept_id}: expected one steering prompt, found {len(steering_prompts)}"
            )
        recovered[concept_id] = next(iter(steering_prompts))
    return recovered


__all__ = [
    "BaselineTargetModel",
    "PromptWrapper",
    "QUESTION_MARKER",
    "QWEN_PROMPT_WRAPPER",
    "QWEN_STEERED_PREFIX",
    "QWEN_STEERED_SUFFIX",
    "SMOLLM3_PROMPT_WRAPPER",
    "SMOLLM3_NO_THINK_REVISION",
    "SMOLLM3_STEERED_PREFIX",
    "SMOLLM3_STEERED_SUFFIX",
    "prompt_wrapper_for_frame",
    "prompt_wrapper_from_manifest",
    "recover_concept_steering_prompts",
    "recover_steered_prompt",
    "rendered_prompt_wrapper_from_manifest",
    "source_manifest_path",
]
