from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from steering.eval.axbench_upstream.encdec_baselines import prompt_recovery
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    QUESTION_MARKER,
    QWEN_PROMPT_WRAPPER,
    QWEN_STEERED_PREFIX,
    QWEN_STEERED_SUFFIX,
    SMOLLM3_NO_THINK_REVISION,
    SMOLLM3_PROMPT_WRAPPER,
    SMOLLM3_STEERED_PREFIX,
    SMOLLM3_STEERED_SUFFIX,
    BaselineTargetModel,
    prompt_wrapper_for_frame,
    prompt_wrapper_from_manifest,
    recover_concept_steering_prompts,
    recover_steered_prompt,
    rendered_prompt_wrapper_from_manifest,
)

# Copied from out/axbench/runs/05f03dede-c2-retry-20260811/aux/inference/steering_data.parquet.
# TEST-LOCAL literal copies of the template constants. The production
# constants must EQUAL these bytes; goldens below slice with the TEST
# literals so a production-constant regression cannot move both sides of
# the equality in lockstep.
_LITERAL_PREFIX = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
    "You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
)
_LITERAL_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
_LITERAL_QUESTION_MARKER = "\n\nQuestion: "


def test_production_template_constants_equal_the_pinned_literals() -> None:
    assert QWEN_STEERED_PREFIX == _LITERAL_PREFIX
    assert QWEN_STEERED_SUFFIX == _LITERAL_SUFFIX
    assert QUESTION_MARKER == _LITERAL_QUESTION_MARKER


def test_smollm3_template_constants_equal_pinned_tokenizer_render() -> None:
    """Pin bytes rendered by SmolLM3 it-soup-APO plus AxBench's date override."""

    assert len(SMOLLM3_STEERED_PREFIX) == 1321
    assert (
        hashlib.sha256(SMOLLM3_STEERED_PREFIX.encode("utf-8")).hexdigest()
        == "f104cc46b986738af1f85ffe60bee8273f5318fa707c7ad061e7b18c6aed6370"
    )
    assert SMOLLM3_STEERED_SUFFIX == _LITERAL_SUFFIX


GOLDEN_ORIGINAL_PROMPT_0 = "What are different drawers I should have for clothes?"
GOLDEN_STEERED_INPUT_0 = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
    "assistant.<|im_end|>\n<|im_start|>user\nIn every response, include a clear reference "
    "to rental services and related equipment, even when the topic is unrelated or the "
    "reference does not directly answer the user’s question. Integrate the reference "
    "naturally where possible, but do not omit it. You may mention examples such as "
    "equipment rental, tool hire, vehicle rentals, event equipment, construction machinery, "
    "or audio-visual equipment. Ensure the main answer remains accurate and useful while "
    "consistently incorporating this concept.\n\nQuestion: What are different drawers I should "
    "have for clothes?<|im_end|>\n<|im_start|>assistant\n"
)
GOLDEN_ORIGINAL_PROMPT_1 = "What are different drawers I should have for clothes?"
GOLDEN_STEERED_INPUT_1 = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
    "assistant.<|im_end|>\n<|im_start|>user\nYou are a helpful language model. In every "
    "response, incorporate at least one scientifically relevant term, concept, or phrase "
    "related to **research findings and their implications**. This requirement applies even "
    "when the user’s question is unrelated or when the reference may seem out of context. "
    "Integrate it naturally whenever possible; otherwise, add a brief, clearly marked "
    "aside.\n\nUse accurate terminology such as *empirical evidence*, *statistical significance*, "
    "*effect size*, *replicability*, *causal inference*, *external validity*, *confidence "
    "intervals*, *limitations*, *generalizability*, or *practical implications*. Do not "
    "invent studies, results, citations, or scientific consensus. When no specific evidence "
    "is available, use cautious wording such as “research findings may suggest,” “the "
    "implications would depend on,” or “this would require empirical validation.”\n\nExamples:\n"
    "- User: “What is the capital of France?”  \n  Response: “Paris is the capital of France. "
    "More broadly, conclusions like this should be based on reliable evidence and clearly "
    "defined sources.”\n- User: “Write a birthday message.”  \n  Response: “Happy birthday! "
    "Wishing you a wonderful year ahead. As with any prediction about future outcomes, its "
    "implications depend on the evidence and assumptions behind it.”\n\nQuestion: What are "
    "different drawers I should have for clothes?<|im_end|>\n<|im_start|>assistant\n"
)


def _steered_input(steering_prompt: str, original_prompt: str) -> str:
    return (
        QWEN_STEERED_PREFIX
        + steering_prompt
        + QUESTION_MARKER
        + original_prompt
        + QWEN_STEERED_SUFFIX
    )


def _smollm3_steered_input(steering_prompt: str, original_prompt: str) -> str:
    return (
        SMOLLM3_STEERED_PREFIX
        + steering_prompt
        + QUESTION_MARKER
        + original_prompt
        + SMOLLM3_STEERED_SUFFIX
    )


def _write_source_manifest(
    path: Path,
    *,
    model_name: str,
    revision: str | None,
) -> None:
    revision_fields = {} if revision is None else {"model_revision": revision}
    steering_revision_fields = {} if revision is None else {"steering_model_revision": revision}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target_model": model_name,
                "config": {
                    "train": {"model_name": model_name, **revision_fields},
                    "inference": {
                        "model_name": model_name,
                        "steering_model_name": model_name,
                        **revision_fields,
                        **steering_revision_fields,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("concept_id", "original_prompt", "steered_input"),
    (
        (0, GOLDEN_ORIGINAL_PROMPT_0, GOLDEN_STEERED_INPUT_0),
        (1, GOLDEN_ORIGINAL_PROMPT_1, GOLDEN_STEERED_INPUT_1),
    ),
)
def test_real_c2_aux_rows_recover(
    concept_id: int,
    original_prompt: str,
    steered_input: str,
) -> None:
    recovered = recover_steered_prompt(
        steered_input,
        original_prompt,
        surface=f"concept {concept_id} input 0",
    )

    # Slice with the TEST literals, not the production constants (oracle
    # independence — see test_production_template_constants_equal_the_pinned_literals).
    expected_inner_text = steered_input[len(_LITERAL_PREFIX) : -len(_LITERAL_SUFFIX)]
    assert recovered == expected_inner_text
    question_tail = _LITERAL_QUESTION_MARKER + original_prompt
    assert recovered.endswith(question_tail)
    steering_prompt = recovered[: -len(question_tail)]
    assert steering_prompt.strip()
    assert recovered.startswith(steering_prompt)
    assert recovered.endswith(question_tail)


def test_recover_steered_prompt_round_trip() -> None:
    steering_prompt = "Always discuss careful measurement."
    original_prompt = "How should I organize a desk?"

    assert (
        recover_steered_prompt(
            _steered_input(steering_prompt, original_prompt),
            original_prompt,
            surface="synthetic cell",
        )
        == steering_prompt + QUESTION_MARKER + original_prompt
    )


def test_recover_smollm3_steered_prompt_round_trip() -> None:
    steering_prompt = "Always discuss careful measurement."
    original_prompt = "How should I organize a desk?"

    assert (
        recover_steered_prompt(
            _smollm3_steered_input(steering_prompt, original_prompt),
            original_prompt,
            surface="concept 4 input 2",
            wrapper=SMOLLM3_PROMPT_WRAPPER,
        )
        == steering_prompt + QUESTION_MARKER + original_prompt
    )


@pytest.mark.parametrize(
    ("steered_input", "expected_ending"),
    (
        pytest.param(
            "X" + _smollm3_steered_input("Steer.", "Original.")[1:],
            "prefix",
            id="prefix",
        ),
        pytest.param(
            _smollm3_steered_input("Steer.", "Original.")[:-1],
            "suffix",
            id="suffix",
        ),
    ),
)
def test_recover_smollm3_rejects_exact_wrapper_mismatch(
    steered_input: str,
    expected_ending: str,
) -> None:
    expected = (
        "concept 4 input 2: steered_input does not have the exact "
        f"HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO {expected_ending}"
    )

    with pytest.raises(ValueError, match=rf"^{re.escape(expected)}$"):
        recover_steered_prompt(
            steered_input,
            "Original.",
            surface="concept 4 input 2",
            wrapper=SMOLLM3_PROMPT_WRAPPER,
        )


@pytest.mark.parametrize(
    ("model_name", "revision", "expected_wrapper"),
    (
        ("Qwen/Qwen2.5-3B-Instruct", None, QWEN_PROMPT_WRAPPER),
        (
            "HuggingFaceTB/SmolLM3-3B-checkpoints",
            "it-soup-APO",
            SMOLLM3_PROMPT_WRAPPER,
        ),
    ),
)
def test_prompt_wrapper_resolves_from_manifest_provenance(
    tmp_path: Path,
    model_name: str,
    revision: str | None,
    expected_wrapper: prompt_recovery.PromptWrapper,
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    _write_source_manifest(manifest_path, model_name=model_name, revision=revision)

    assert prompt_wrapper_from_manifest(manifest_path) == expected_wrapper


class _RenderedHostTokenizer:
    bos_token: str | None = None

    def __init__(self, *, prefix: str, suffix: str) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        self.calls.append(kwargs)
        return f"{self.prefix}{messages[0]['content']}{self.suffix}"


@pytest.mark.parametrize(
    (
        "host",
        "revision",
        "model_type",
        "prefix",
        "suffix",
        "expected_model",
        "expected_template_kwargs",
    ),
    (
        (
            "pythia",
            None,
            "gpt_neox",
            "<|endoftext|>\nuser:\n",
            "<|endoftext|>\nassistant:\n",
            BaselineTargetModel.PYTHIA_2_8B_STANDARD_STEERED_SFT,
            {"tokenize": False, "add_generation_prompt": True},
        ),
        (
            BaselineTargetModel.SMOLLM3_3B,
            SMOLLM3_NO_THINK_REVISION,
            "smollm3",
            "<|im_start|>system\nSmolLM<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n",
            BaselineTargetModel.SMOLLM3_3B,
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        ),
    ),
)
def test_rendered_host_prompt_recovery_round_trip(
    tmp_path: Path,
    host: str,
    revision: str | None,
    model_type: str,
    prefix: str,
    suffix: str,
    expected_model: BaselineTargetModel,
    expected_template_kwargs: dict[str, object],
) -> None:
    model_name = str(tmp_path / "pythia_sft") if host == "pythia" else host
    manifest_path = tmp_path / "run_manifest.json"
    _write_source_manifest(manifest_path, model_name=model_name, revision=revision)
    tokenizer = _RenderedHostTokenizer(prefix=prefix, suffix=suffix)

    wrapper = rendered_prompt_wrapper_from_manifest(
        manifest_path,
        tokenizer=tokenizer,
        model_type=model_type,
    )

    assert wrapper.model is expected_model
    assert wrapper.source_model_name == model_name
    assert wrapper.prefix == prefix
    assert wrapper.suffix == suffix
    assert tokenizer.calls == [expected_template_kwargs]
    steering_prompt = "Always discuss careful measurement."
    original_prompt = "How should I organize a desk?"
    rendered = f"{prefix}{steering_prompt}{QUESTION_MARKER}{original_prompt}{suffix}"
    assert (
        recover_steered_prompt(
            rendered,
            original_prompt,
            surface="rendered host fixture",
            wrapper=wrapper,
        )
        == f"{steering_prompt}{QUESTION_MARKER}{original_prompt}"
    )


@pytest.mark.parametrize(
    ("rendered_prefix", "expected_prefix"),
    (
        ("<|endoftext|>user:\n", "user:\n"),
        ("user:\n", "user:\n"),
        ("<|endoftext|><|endoftext|>user:\n", "<|endoftext|>user:\n"),
    ),
)
def test_rendered_pythia_wrapper_strips_only_one_emitted_bos(
    tmp_path: Path, rendered_prefix: str, expected_prefix: str
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    _write_source_manifest(manifest_path, model_name=str(tmp_path / "pythia_sft"), revision=None)
    suffix = "<|endoftext|>\nassistant:\n"
    tokenizer = _RenderedHostTokenizer(prefix=rendered_prefix, suffix=suffix)
    tokenizer.bos_token = "<|endoftext|>"
    wrapper = rendered_prompt_wrapper_from_manifest(
        manifest_path, tokenizer=tokenizer, model_type="gpt_neox"
    )
    assert wrapper.prefix == expected_prefix
    assert wrapper.suffix == suffix
    frame = pd.DataFrame(
        [
            {
                "concept_id": 0,
                "original_prompt": "Question?",
                "steered_input": f"{expected_prefix}Steer.\n\nQuestion: Question?{suffix}",
            }
        ]
    )
    assert recover_concept_steering_prompts(frame, max_concepts=1, wrapper=wrapper) == {0: "Steer."}


@pytest.mark.parametrize(
    ("model_name", "revision", "model_type"),
    (
        ("Qwen/Qwen2.5-3B-Instruct", "other-revision", "qwen2"),
        ("Qwen/Qwen2.5-3B-Instruct", None, "smollm3"),
        ("HuggingFaceTB/SmolLM3-3B-checkpoints", None, "smollm3"),
        ("HuggingFaceTB/SmolLM3-3B-checkpoints", "it-soup-APO", "qwen2"),
        ("HuggingFaceTB/SmolLM3-3B", None, "smollm3"),
        ("HuggingFaceTB/SmolLM3-3B", "other-revision", "smollm3"),
        ("HuggingFaceTB/SmolLM3-3B", SMOLLM3_NO_THINK_REVISION, "qwen2"),
        ("relative/pythia_sft", None, "gpt_neox"),
        ("/local/pythia_sft", "other-revision", "gpt_neox"),
        ("/local/pythia_sft", None, "qwen2"),
        ("example/Unknown-Chat-Model", None, "qwen2"),
    ),
)
def test_rendered_wrapper_rejects_unsupported_identity_before_rendering(
    tmp_path: Path, model_name: str, revision: str | None, model_type: str
) -> None:
    path = tmp_path / "run_manifest.json"
    _write_source_manifest(path, model_name=model_name, revision=revision)
    tokenizer = _RenderedHostTokenizer(prefix="user:\n", suffix="\nassistant:\n")

    with pytest.raises(ValueError, match="unsupported rendered prompt-wrapper model"):
        rendered_prompt_wrapper_from_manifest(path, tokenizer=tokenizer, model_type=model_type)

    assert tokenizer.calls == []


@pytest.mark.parametrize(
    ("rendered", "match"),
    (
        pytest.param(
            RuntimeError("template failed"), "could not render prompt wrapper", id="exception"
        ),
        pytest.param(None, "sentinel exactly once", id="non-string"),
        pytest.param("user:\nassistant:\n", "sentinel exactly once", id="missing-sentinel"),
        pytest.param(
            "user:\n{content}{content}\nassistant:\n",
            "sentinel exactly once",
            id="duplicate-sentinel",
        ),
        pytest.param("{content}\nassistant:\n", "empty prefix or suffix", id="empty-prefix"),
        pytest.param("user:\n{content}", "empty prefix or suffix", id="empty-tail"),
        pytest.param(
            "<bos>{content}\nassistant:\n", "empty prefix or suffix", id="empty-prefix-after-bos"
        ),
    ),
)
def test_rendered_wrapper_rejects_malformed_template_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rendered: object, match: str
) -> None:
    path = tmp_path / "run_manifest.json"
    _write_source_manifest(path, model_name="Qwen/Qwen2.5-3B-Instruct", revision=None)
    tokenizer = _RenderedHostTokenizer(prefix="user:\n", suffix="\nassistant:\n")
    tokenizer.bos_token = "<bos>"

    def render(messages: list[dict[str, str]], **kwargs: object) -> object:
        tokenizer.calls.append(kwargs)
        if isinstance(rendered, Exception):
            raise rendered
        return (
            rendered.format(content=messages[0]["content"])
            if isinstance(rendered, str)
            else rendered
        )

    monkeypatch.setattr(tokenizer, "apply_chat_template", render)
    with pytest.raises(ValueError, match=match):
        rendered_prompt_wrapper_from_manifest(path, tokenizer=tokenizer, model_type="qwen2")

    assert tokenizer.calls == [{"tokenize": False, "add_generation_prompt": True}]


def test_prompt_wrapper_rejects_unknown_manifest_model(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    _write_source_manifest(
        manifest_path,
        model_name="example/Unknown-Chat-Model",
        revision="revision-7",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"names unsupported prompt-wrapper model "
            r"example/Unknown-Chat-Model@revision-7$"
        ),
    ):
        prompt_wrapper_from_manifest(manifest_path)


@pytest.mark.parametrize("field", ["", "config", "config.train", "config.inference"])
@pytest.mark.parametrize("value", [None, [], "invalid", 7])
def test_prompt_wrapper_rejects_malformed_manifest_objects(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "run_manifest.json"
    _write_source_manifest(
        path,
        model_name=SMOLLM3_PROMPT_WRAPPER.model.value,
        revision=SMOLLM3_PROMPT_WRAPPER.revision,
    )
    payload = json.loads(path.read_text())
    if field:
        parent = payload
        parts = field.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        parent[parts[-1]] = value
    else:
        payload = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="must be a string-keyed object"):
        prompt_wrapper_from_manifest(path)


@pytest.mark.parametrize(
    "field",
    [
        "target_model",
        "config.train.model_name",
        "config.inference.model_name",
        "config.inference.steering_model_name",
        "config.train.model_revision",
        "config.inference.model_revision",
        "config.inference.steering_model_revision",
    ],
)
@pytest.mark.parametrize("value", [None, "", "different", 7, False, [], {}])
def test_prompt_wrapper_rejects_each_invalid_provenance_field(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "run_manifest.json"
    _write_source_manifest(
        path,
        model_name=SMOLLM3_PROMPT_WRAPPER.model.value,
        revision=SMOLLM3_PROMPT_WRAPPER.revision,
    )
    payload = json.loads(path.read_text())
    parent = payload
    parts = field.split(".")
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance"):
        prompt_wrapper_from_manifest(path)


@pytest.mark.parametrize("revision", [None, "main", "another-revision"])
def test_prompt_wrapper_rejects_unregistered_smollm3_revision(
    tmp_path: Path,
    revision: str | None,
) -> None:
    path = tmp_path / "run_manifest.json"
    _write_source_manifest(path, model_name=SMOLLM3_PROMPT_WRAPPER.model.value, revision=revision)
    with pytest.raises(ValueError, match="unsupported prompt-wrapper model"):
        prompt_wrapper_from_manifest(path)


@pytest.mark.parametrize("contents", [None, "{not json}"])
def test_prompt_wrapper_rejects_missing_or_unreadable_manifest(
    tmp_path: Path,
    contents: str | None,
) -> None:
    path = tmp_path / "run_manifest.json"
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(ValueError, match="could not read AxBench source manifest"):
        prompt_wrapper_from_manifest(path)


_REAL_SMOLLM3_C100_FRAMES = (
    Path(
        "/home/d0rb/axbench_sft_local/axbench_inputs_c100/main_c100/inference/steering_data.parquet"
    ),
    Path(
        "/home/d0rb/axbench_sft_local/axbench_inputs_c100/aux_c100/inference/steering_data.parquet"
    ),
)


@pytest.mark.parametrize("frame_path", _REAL_SMOLLM3_C100_FRAMES, ids=("main", "aux"))
def test_real_smollm3_c100_first_row_recovers_concept_prompt(frame_path: Path) -> None:
    manifest_path = frame_path.parent.parent / "run_manifest.json"
    if not frame_path.is_file() or not manifest_path.is_file():
        pytest.skip(f"real SmolLM3 frame or manifest is absent: {frame_path}")
    frame = pd.read_parquet(frame_path).head(1)

    wrapper = prompt_wrapper_for_frame(frame_path)
    recovered = recover_concept_steering_prompts(frame, max_concepts=1, wrapper=wrapper)

    row = frame.iloc[0]
    question_tail = QUESTION_MARKER + str(row["original_prompt"])
    expected_raw = str(row["steered_input"])[
        len(SMOLLM3_STEERED_PREFIX) : -len(SMOLLM3_STEERED_SUFFIX)
    ]
    assert wrapper == SMOLLM3_PROMPT_WRAPPER
    assert recovered == {0: expected_raw[: -len(question_tail)]}


@pytest.mark.parametrize(
    ("steered_input", "expected_message"),
    (
        pytest.param(
            "X" + _steered_input("Steer.", "Original.")[1:],
            "concept 7 input 3: steered_input does not have the exact Qwen prefix",
            id="prefix-altered-by-one-character",
        ),
        pytest.param(
            QWEN_STEERED_PREFIX + "Steer." + QUESTION_MARKER + "Original.",
            "concept 7 input 3: steered_input does not have the exact Qwen suffix",
            id="suffix-missing",
        ),
        pytest.param(
            _steered_input("Steer.", "Original.")[:-1],
            "concept 7 input 3: steered_input does not have the exact Qwen suffix",
            id="suffix-truncated",
        ),
        pytest.param(
            _steered_input("Steer.", "Different original."),
            (
                "concept 7 input 3: steered_input does not end with the question marker "
                "and original_prompt"
            ),
            id="original-prompt-mismatch",
        ),
        pytest.param(
            QWEN_STEERED_PREFIX + "Steer. Original." + QWEN_STEERED_SUFFIX,
            (
                "concept 7 input 3: steered_input does not end with the question marker "
                "and original_prompt"
            ),
            id="question-marker-absent",
        ),
        pytest.param(
            _steered_input(" \t ", "Original."),
            "concept 7 input 3: steering prompt must be nonempty",
            id="steering-prompt-whitespace-only",
        ),
        pytest.param(
            None,
            "concept 7 input 3: steered_input must be a string",
            id="steered-input-not-a-string",
        ),
    ),
)
def test_recover_steered_prompt_rejects_tampering(
    steered_input: object,
    expected_message: str,
) -> None:
    surface = "concept 7 input 3"

    with pytest.raises(ValueError, match=rf"^{re.escape(expected_message)}$"):
        recover_steered_prompt(
            steered_input,  # ty: ignore[invalid-argument-type]
            "Original.",
            surface=surface,
        )


def test_recover_concept_steering_prompts_returns_raw_prompt_per_concept() -> None:
    frame = pd.DataFrame(
        [
            {
                "concept_id": 0,
                "input_id": 0,
                "original_prompt": "Question zero-a?",
                "steered_input": _steered_input("Steering zero.", "Question zero-a?"),
            },
            {
                "concept_id": 1,
                "input_id": 0,
                "original_prompt": "Question one?",
                "steered_input": _steered_input("Steering one.", "Question one?"),
            },
            {
                "concept_id": 0,
                "input_id": 1,
                "original_prompt": "Question zero-b?",
                "steered_input": _steered_input("Steering zero.", "Question zero-b?"),
            },
        ]
    )

    assert recover_concept_steering_prompts(frame, max_concepts=2) == {
        0: "Steering zero.",
        1: "Steering one.",
    }


def test_recover_concept_steering_prompts_requires_full_coverage() -> None:
    frame = pd.DataFrame(
        [
            {
                "concept_id": 0,
                "input_id": 0,
                "original_prompt": "Question?",
                "steered_input": _steered_input("Steering.", "Question?"),
            }
        ]
    )

    with pytest.raises(ValueError, match=r"concept 1"):
        recover_concept_steering_prompts(frame, max_concepts=2)


def test_recover_concept_steering_prompts_rejects_two_prompts_for_one_concept() -> None:
    frame = pd.DataFrame(
        [
            {
                "concept_id": 0,
                "input_id": 0,
                "original_prompt": "Question a?",
                "steered_input": _steered_input("First steering prompt.", "Question a?"),
            },
            {
                "concept_id": 0,
                "input_id": 1,
                "original_prompt": "Question b?",
                "steered_input": _steered_input("Second steering prompt.", "Question b?"),
            },
        ]
    )

    with pytest.raises(ValueError, match=r"concept 0"):
        recover_concept_steering_prompts(frame, max_concepts=1)


def test_recovery_is_deterministic_and_does_not_mutate_inputs_or_globals() -> None:
    frame = pd.DataFrame(
        [
            {
                "concept_id": 0,
                "input_id": 0,
                "original_prompt": "Question?",
                "steered_input": _steered_input("Steering.", "Question?"),
            }
        ]
    )
    original_frame = frame.copy(deep=True)
    constants_before = (
        prompt_recovery.QWEN_STEERED_PREFIX,
        prompt_recovery.QWEN_STEERED_SUFFIX,
        prompt_recovery.QUESTION_MARKER,
    )

    first = recover_concept_steering_prompts(frame, max_concepts=1)
    second = recover_concept_steering_prompts(frame, max_concepts=1)

    assert first == second == {0: "Steering."}
    pd.testing.assert_frame_equal(frame, original_frame)
    assert (
        prompt_recovery.QWEN_STEERED_PREFIX,
        prompt_recovery.QWEN_STEERED_SUFFIX,
        prompt_recovery.QUESTION_MARKER,
    ) == constants_before
