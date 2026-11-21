"""Prompt variants and few-shot examples for autointerp intruder detection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from steering.eval.autointerp_filter.types import AutoInterpConfigError


class IntruderPromptVariant(StrEnum):
    delphi = "delphi"
    calibrated_v1 = "calibrated_v1"
    highlight_context_v1 = "highlight_context_v1"
    highlight_context_token_first_v1 = "highlight_context_token_first_v1"


class IntruderFewShotSet(StrEnum):
    delphi = "delphi"
    sae_highlighted_v1 = "sae_highlighted_v1"
    sae_balanced_v1 = "sae_balanced_v1"
    sae_realistic_v1 = "sae_realistic_v1"


DEFAULT_INTRUDER_PROMPT_VARIANT = (
    IntruderPromptVariant.highlight_context_token_first_v1.value
)
DEFAULT_INTRUDER_FEW_SHOT_SET = IntruderFewShotSet.sae_realistic_v1.value


@dataclass(frozen=True)
class IntruderPromptSpec:
    prompt_variant: IntruderPromptVariant
    few_shot_set: IntruderFewShotSet
    version: str
    system_prompt: str
    few_shot_messages: tuple[dict[str, str], ...]
    few_shot_cot_messages: tuple[dict[str, str], ...]

    def cache_payload(self) -> dict[str, str]:
        return {
            "variant": self.prompt_variant.value,
            "few_shot_set": self.few_shot_set.value,
            "version": self.version,
        }


def example_line(index: int, text: str) -> str:
    cleaned = text.replace("\x00", " ")
    return f"Example {index}:{cleaned}"


def examples_content(texts: list[str]) -> str:
    return "Examples:\n\n" + "\n".join(
        example_line(index, text) for index, text in enumerate(texts)
    ) + "\n"


def available_prompt_variants() -> tuple[str, ...]:
    return tuple(variant.value for variant in IntruderPromptVariant)


def available_few_shot_sets() -> tuple[str, ...]:
    return tuple(few_shot.value for few_shot in IntruderFewShotSet)


def resolve_intruder_prompt_spec(
    *,
    prompt_variant: str,
    few_shot_set: str,
) -> IntruderPromptSpec:
    try:
        variant = IntruderPromptVariant(prompt_variant)
    except ValueError as exc:
        raise AutoInterpConfigError(
            f"judge.prompt_variant must be one of {available_prompt_variants()}, "
            f"got {prompt_variant!r}"
        ) from exc
    try:
        few_shots = IntruderFewShotSet(few_shot_set)
    except ValueError as exc:
        raise AutoInterpConfigError(
            f"judge.few_shot_set must be one of {available_few_shot_sets()}, "
            f"got {few_shot_set!r}"
        ) from exc
    system_prompt = _SYSTEM_PROMPTS[variant]
    few_shot_messages = _FEW_SHOT_MESSAGES[few_shots]
    few_shot_cot_messages = _FEW_SHOT_COT_MESSAGES[few_shots]
    return IntruderPromptSpec(
        prompt_variant=variant,
        few_shot_set=few_shots,
        version=_prompt_version(
            variant=variant,
            few_shots=few_shots,
            system_prompt=system_prompt,
            few_shot_messages=few_shot_messages,
            few_shot_cot_messages=few_shot_cot_messages,
        ),
        system_prompt=system_prompt,
        few_shot_messages=few_shot_messages,
        few_shot_cot_messages=few_shot_cot_messages,
    )


def _prompt_version(
    *,
    variant: IntruderPromptVariant,
    few_shots: IntruderFewShotSet,
    system_prompt: str,
    few_shot_messages: tuple[dict[str, str], ...],
    few_shot_cot_messages: tuple[dict[str, str], ...],
) -> str:
    payload = {
        "system_prompt": system_prompt,
        "few_shot_messages": few_shot_messages,
        "few_shot_cot_messages": few_shot_cot_messages,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"{variant.value}:{few_shots.value}:v1:{digest}"


# Prompt text mirrors EleutherAI/delphi's current IntruderScorer prompt
# (Apache-2.0), with this repo supplying local token-window examples.
_DELPHI_SYSTEM_PROMPT = (
    'You are an intelligent and meticulous linguistics researcher doing a "intruder detection" '
    "task.\n\n"
    "You will then be given several text examples, either full sentences or words. Your task is "
    'to determine which examples should be classified as "intruder".\n'
    "Some sentences will have words highlighted with <> tags. Do not overthink.\n\n"
    "There is only ever one intruder in the examples.\n\n"
    "You should write [RESPONSE]: followed by the index of the intruder.\n\n"
)

_CALIBRATED_V1_SYSTEM_PROMPT = (
    "You are evaluating sparse-autoencoder feature examples with an intruder "
    "detection task.\n\n"
    "Each option is a short token window. Most options contain highlighted tokens "
    "that express the same feature behavior. Exactly one option is the intruder: "
    "its highlighted tokens are least consistent with the shared behavior of the "
    "other options.\n\n"
    "The shared behavior may be semantic, lexical, syntactic, formatting-related, "
    "entity-related, language-related, or tokenization-related. The intruder can "
    "also contain highlighted tokens, so do not answer from marker presence, marker "
    "count, or marker position alone.\n\n"
    "Return only [RESPONSE]: followed by the index of the intruder.\n\n"
)

_HIGHLIGHT_CONTEXT_V1_SYSTEM_PROMPT = (
    "You are doing intruder detection for sparse-autoencoder feature examples.\n\n"
    "Each numbered example is a short text window. Text between << and >> is the "
    "highlighted span. Exactly one example is the intruder: it is least consistent "
    "with the shared pattern in the other examples.\n\n"
    "Compare the highlighted span and its nearby context first. If needed, also "
    "use broader topic, style, formatting, entity type, language, or token-fragment "
    "patterns. The intruder may still contain highlights, so do not choose based "
    "only on marker count or marker position.\n\n"
    "Return only [RESPONSE]: followed by the index of the intruder.\n\n"
)

_HIGHLIGHT_CONTEXT_TOKEN_FIRST_V1_SYSTEM_PROMPT = (
    "You are doing intruder detection for sparse-autoencoder feature examples.\n\n"
    "Each numbered example is a short text window. Text between << and >> is the "
    "highlighted span. Exactly one example is the intruder: it is least consistent "
    "with the shared pattern in the other examples.\n\n"
    "Compare the highlighted span and its nearby context first. For short words, "
    "punctuation, whitespace, or token fragments, the exact highlighted text can "
    "be more important than the broader document topic. If needed, also use "
    "topic, style, formatting, entity type, language, or token-fragment patterns. "
    "The intruder may still contain highlights, so do not choose based only on "
    "marker count or marker position.\n\n"
    "Answer with exactly one of the example numbers shown (e.g. 0-3 for four "
    "options). Ignore any other numbers that appear inside the example text "
    "itself -- those are ordinary content, never a valid answer.\n\n"
    "Return only [RESPONSE]: followed by the index of the intruder.\n\n"
)

_DELPHI_EXAMPLE_ONE = """Examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ
Example 1: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the
Example 2: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 3: line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini
"""

_DELPHI_RESPONSE_ONE_COT = (
    "There are 3 examples that are related to American football. The intruder is 2 "
    "because it is about videos and ads."
)
_DELPHI_RESPONSE_ONE = "[RESPONSE]: 2"

_DELPHI_EXAMPLE_TWO = """Examples:

Example 0: enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both
Example 1: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 2: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 3:ĊĊIt has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? Â· Finally, we believe the
"""

_DELPHI_RESPONSE_TWO_COT = (
    "I can see that there are several examples that have the word 'of' before a "
    "capital letter. The intruder is 0 because it does not."
)
_DELPHI_RESPONSE_TWO = "[RESPONSE]: 0"

_DELPHI_EXAMPLE_THREE = """Examples:

Example 0: Climbing
Example 1: running
Example 2: swim
Example 3: eating
Example 4: cycling
"""

_DELPHI_RESPONSE_THREE_COT = (
    "All examples are related to activities, the first 3 and the last one being about "
    "sports and physical activities. Eating is not a sport or physical activity so it "
    "is the intruder."
)
_DELPHI_RESPONSE_THREE = "[RESPONSE]: 3"

# Delphi's default intruder few-shot prompt uses these first three examples,
# none of which have <<>> highlights. Its record-specific intruder samples are
# still highlighted by sample._prepare_text(..., highlighted=True), as are the
# trial examples rendered by this repo.
_DELPHI_FEW_SHOT_MESSAGES = (
    {
        "role": "user",
        "content": _DELPHI_EXAMPLE_ONE,
    },
    {"role": "assistant", "content": _DELPHI_RESPONSE_ONE},
    {
        "role": "user",
        "content": _DELPHI_EXAMPLE_TWO,
    },
    {"role": "assistant", "content": _DELPHI_RESPONSE_TWO},
    {
        "role": "user",
        "content": _DELPHI_EXAMPLE_THREE,
    },
    {"role": "assistant", "content": _DELPHI_RESPONSE_THREE},
)

_DELPHI_FEW_SHOT_COT_MESSAGES = (
    {
        "role": "user",
        "content": _DELPHI_FEW_SHOT_MESSAGES[0]["content"],
    },
    {
        "role": "assistant",
        "content": _DELPHI_RESPONSE_ONE_COT + _DELPHI_RESPONSE_ONE,
    },
    {
        "role": "user",
        "content": _DELPHI_FEW_SHOT_MESSAGES[2]["content"],
    },
    {
        "role": "assistant",
        "content": _DELPHI_RESPONSE_TWO_COT + _DELPHI_RESPONSE_TWO,
    },
    {
        "role": "user",
        "content": _DELPHI_FEW_SHOT_MESSAGES[4]["content"],
    },
    {
        "role": "assistant",
        "content": _DELPHI_RESPONSE_THREE_COT + _DELPHI_RESPONSE_THREE,
    },
)

_SAE_TOPIC_EXAMPLE = """Examples:

Example 0: The compiler lowers the source into <<bytecode>> before the virtual machine runs it.
Example 1: A parser turns the program text into <<tokens>> and syntax nodes.
Example 2: The bakery put fresh <<muffins>> near the front counter before opening.
Example 3: The debugger paused when the process hit a <<breakpoint>> in the loop.
"""
_SAE_TOPIC_RESPONSE_COT = (
    "Examples 0, 1, and 3 are about programming artifacts. Example 2 is about a bakery."
)
_SAE_TOPIC_RESPONSE = "[RESPONSE]: 2"

_SAE_SYNTAX_EXAMPLE = """Examples:

Example 0: The committee voted to carry <<out>> the review next month.
Example 1: She tried to set <<up>> the account before the meeting.
Example 2: The report lists the <<river>> near the eastern border.
Example 3: He asked them to turn <<off>> the lights after class.
"""
_SAE_SYNTAX_RESPONSE_COT = (
    "Examples 0, 1, and 3 highlight particles in phrasal verbs. Example 2 does not."
)
_SAE_SYNTAX_RESPONSE = "[RESPONSE]: 2"

_SAE_FORMATTING_EXAMPLE = """Examples:

Example 0: ## <<Installation>>
Example 1: - <<Usage>>: run the command with the config path.
Example 2: The children watched a <<movie>> after dinner.
Example 3: `<<config.yaml>>` stores the runtime options.
"""
_SAE_FORMATTING_RESPONSE_COT = (
    "Examples 0, 1, and 3 highlight heading, list, or code-like formatting. Example 2 "
    "is ordinary prose."
)
_SAE_FORMATTING_RESPONSE = "[RESPONSE]: 2"

_SAE_TOKENIZATION_EXAMPLE = """Examples:

Example 0: The libraries were designed for inter<<oper>>ability across services.
Example 1: The co<<oper>>ative board approved the proposal.
Example 2: The hikers rested beside the <<lake>> before sunset.
Example 3: Re<<oper>>ation of the plant required a safety review.
"""
_SAE_TOKENIZATION_RESPONSE_COT = (
    "Examples 0, 1, and 3 highlight the same token fragment inside related words. "
    "Example 2 highlights an ordinary whole word."
)
_SAE_TOKENIZATION_RESPONSE = "[RESPONSE]: 2"

_SAE_BALANCED_VERB_EXAMPLE = """Examples:

Example 0: The team agreed to <<review>> the draft before release.
Example 1: The baker set the <<tray>> beside the warm oven.
Example 2: Please <<review>> the error log before merging.
Example 3: The committee will <<review>> the proposal on Friday.
"""
_SAE_BALANCED_VERB_RESPONSE_COT = (
    "Examples 0, 2, and 3 highlight the same verb in similar uses. Example 1 "
    "highlights an unrelated noun."
)
_SAE_BALANCED_VERB_RESPONSE = "[RESPONSE]: 1"

_SAE_BALANCED_CODE_EXAMPLE = """Examples:

Example 0: Run `<<pytest>> tests/unit` before pushing.
Example 1: Open `<<config.yaml>>` and set the path.
Example 2: The option `<<--help>>` prints usage.
Example 3: The child wore a <<blue>> jacket at recess.
"""
_SAE_BALANCED_CODE_RESPONSE_COT = (
    "Examples 0, 1, and 2 highlight code-like text. Example 3 is ordinary prose."
)
_SAE_BALANCED_CODE_RESPONSE = "[RESPONSE]: 3"

_SAE_BALANCED_ARTICLE_EXAMPLE = """Examples:

Example 0: The museum opened <<after>> sunset near the river.
Example 1: She checked <<the>> value before saving the file.
Example 2: They moved <<the>> meeting to Tuesday.
Example 3: Most of <<the>> examples use a definite article before a noun.
"""
_SAE_BALANCED_ARTICLE_RESPONSE_COT = (
    "Examples 1, 2, and 3 highlight the same article. Example 0 highlights a "
    "different function word."
)
_SAE_BALANCED_ARTICLE_RESPONSE = "[RESPONSE]: 0"

_SAE_BALANCED_SUFFIX_EXAMPLE = """Examples:

Example 0: The file was compress<<ed>> overnight.
Example 1: The table was index<<ed>> for faster search.
Example 2: The hikers found <<moss>> under the bridge.
Example 3: The result was normaliz<<ed>> after scaling.
"""
_SAE_BALANCED_SUFFIX_RESPONSE_COT = (
    "Examples 0, 1, and 3 highlight a past-tense suffix fragment. Example 2 "
    "highlights an unrelated whole word."
)
_SAE_BALANCED_SUFFIX_RESPONSE = "[RESPONSE]: 2"

_SAE_REALISTIC_TO_EXAMPLE = """Examples:

Example 0: They agreed <<to>> review the budget tomorrow.
Example 1: The train went <<to>> Denver before noon.
Example 2: She hoped <<to>> finish the article tonight.
Example 3: We need <<to>> update the server logs.
"""
_SAE_REALISTIC_TO_RESPONSE_COT = (
    "Examples 0, 2, and 3 use 'to' before a verb. Example 1 uses 'to' before "
    "a destination."
)
_SAE_REALISTIC_TO_RESPONSE = "[RESPONSE]: 1"

_SAE_REALISTIC_COMMA_EXAMPLE = """Examples:

Example 0: The survey ended in 1998<<,>> after the last interviews.
Example 1: He was born in Boston<<,>> Massachusetts, in 1971.
Example 2: The paper cites Smith<<,>> Jones, and Lee.
Example 3: The recipe needs <<salt>> before baking.
"""
_SAE_REALISTIC_COMMA_RESPONSE_COT = (
    "Examples 0, 1, and 2 highlight commas in ordinary prose. Example 3 "
    "highlights a content word."
)
_SAE_REALISTIC_COMMA_RESPONSE = "[RESPONSE]: 3"

_SAE_REALISTIC_ENTITY_EXAMPLE = """Examples:

Example 0: The committee approved <<the>> proposal after debate.
Example 1: <<Dr. Rivera>> presented the findings at noon.
Example 2: <<Senator Adams>> answered questions after the vote.
Example 3: <<Professor Chen>> revised the syllabus.
"""
_SAE_REALISTIC_ENTITY_RESPONSE_COT = (
    "Examples 1, 2, and 3 highlight titled person names. Example 0 highlights "
    "a common article."
)
_SAE_REALISTIC_ENTITY_RESPONSE = "[RESPONSE]: 0"

_SAE_REALISTIC_SUFFIX_EXAMPLE = """Examples:

Example 0: The file was compress<<ed>> overnight.
Example 1: The table was index<<ed>> for faster search.
Example 2: The hikers found <<moss>> under the bridge.
Example 3: The result was normaliz<<ed>> after scaling.
"""
_SAE_REALISTIC_SUFFIX_RESPONSE_COT = (
    "Examples 0, 1, and 3 highlight a past-tense suffix fragment. Example 2 "
    "highlights an unrelated whole word."
)
_SAE_REALISTIC_SUFFIX_RESPONSE = "[RESPONSE]: 2"

_SAE_HIGHLIGHTED_FEW_SHOT_MESSAGES = (
    {"role": "user", "content": _SAE_TOPIC_EXAMPLE},
    {"role": "assistant", "content": _SAE_TOPIC_RESPONSE},
    {"role": "user", "content": _SAE_SYNTAX_EXAMPLE},
    {"role": "assistant", "content": _SAE_SYNTAX_RESPONSE},
    {"role": "user", "content": _SAE_FORMATTING_EXAMPLE},
    {"role": "assistant", "content": _SAE_FORMATTING_RESPONSE},
    {"role": "user", "content": _SAE_TOKENIZATION_EXAMPLE},
    {"role": "assistant", "content": _SAE_TOKENIZATION_RESPONSE},
)

_SAE_HIGHLIGHTED_FEW_SHOT_COT_MESSAGES = (
    {"role": "user", "content": _SAE_TOPIC_EXAMPLE},
    {"role": "assistant", "content": _SAE_TOPIC_RESPONSE_COT + _SAE_TOPIC_RESPONSE},
    {"role": "user", "content": _SAE_SYNTAX_EXAMPLE},
    {"role": "assistant", "content": _SAE_SYNTAX_RESPONSE_COT + _SAE_SYNTAX_RESPONSE},
    {"role": "user", "content": _SAE_FORMATTING_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_FORMATTING_RESPONSE_COT + _SAE_FORMATTING_RESPONSE,
    },
    {"role": "user", "content": _SAE_TOKENIZATION_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_TOKENIZATION_RESPONSE_COT + _SAE_TOKENIZATION_RESPONSE,
    },
)

_SAE_BALANCED_FEW_SHOT_MESSAGES = (
    {"role": "user", "content": _SAE_BALANCED_VERB_EXAMPLE},
    {"role": "assistant", "content": _SAE_BALANCED_VERB_RESPONSE},
    {"role": "user", "content": _SAE_BALANCED_CODE_EXAMPLE},
    {"role": "assistant", "content": _SAE_BALANCED_CODE_RESPONSE},
    {"role": "user", "content": _SAE_BALANCED_ARTICLE_EXAMPLE},
    {"role": "assistant", "content": _SAE_BALANCED_ARTICLE_RESPONSE},
    {"role": "user", "content": _SAE_BALANCED_SUFFIX_EXAMPLE},
    {"role": "assistant", "content": _SAE_BALANCED_SUFFIX_RESPONSE},
)

_SAE_BALANCED_FEW_SHOT_COT_MESSAGES = (
    {"role": "user", "content": _SAE_BALANCED_VERB_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_BALANCED_VERB_RESPONSE_COT + _SAE_BALANCED_VERB_RESPONSE,
    },
    {"role": "user", "content": _SAE_BALANCED_CODE_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_BALANCED_CODE_RESPONSE_COT + _SAE_BALANCED_CODE_RESPONSE,
    },
    {"role": "user", "content": _SAE_BALANCED_ARTICLE_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_BALANCED_ARTICLE_RESPONSE_COT + _SAE_BALANCED_ARTICLE_RESPONSE,
    },
    {"role": "user", "content": _SAE_BALANCED_SUFFIX_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_BALANCED_SUFFIX_RESPONSE_COT + _SAE_BALANCED_SUFFIX_RESPONSE,
    },
)

_SAE_REALISTIC_FEW_SHOT_MESSAGES = (
    {"role": "user", "content": _SAE_REALISTIC_TO_EXAMPLE},
    {"role": "assistant", "content": _SAE_REALISTIC_TO_RESPONSE},
    {"role": "user", "content": _SAE_REALISTIC_COMMA_EXAMPLE},
    {"role": "assistant", "content": _SAE_REALISTIC_COMMA_RESPONSE},
    {"role": "user", "content": _SAE_REALISTIC_ENTITY_EXAMPLE},
    {"role": "assistant", "content": _SAE_REALISTIC_ENTITY_RESPONSE},
    {"role": "user", "content": _SAE_REALISTIC_SUFFIX_EXAMPLE},
    {"role": "assistant", "content": _SAE_REALISTIC_SUFFIX_RESPONSE},
)

_SAE_REALISTIC_FEW_SHOT_COT_MESSAGES = (
    {"role": "user", "content": _SAE_REALISTIC_TO_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_REALISTIC_TO_RESPONSE_COT + _SAE_REALISTIC_TO_RESPONSE,
    },
    {"role": "user", "content": _SAE_REALISTIC_COMMA_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_REALISTIC_COMMA_RESPONSE_COT + _SAE_REALISTIC_COMMA_RESPONSE,
    },
    {"role": "user", "content": _SAE_REALISTIC_ENTITY_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_REALISTIC_ENTITY_RESPONSE_COT + _SAE_REALISTIC_ENTITY_RESPONSE,
    },
    {"role": "user", "content": _SAE_REALISTIC_SUFFIX_EXAMPLE},
    {
        "role": "assistant",
        "content": _SAE_REALISTIC_SUFFIX_RESPONSE_COT + _SAE_REALISTIC_SUFFIX_RESPONSE,
    },
)

_SYSTEM_PROMPTS = {
    IntruderPromptVariant.delphi: _DELPHI_SYSTEM_PROMPT,
    IntruderPromptVariant.calibrated_v1: _CALIBRATED_V1_SYSTEM_PROMPT,
    IntruderPromptVariant.highlight_context_v1: _HIGHLIGHT_CONTEXT_V1_SYSTEM_PROMPT,
    IntruderPromptVariant.highlight_context_token_first_v1: (
        _HIGHLIGHT_CONTEXT_TOKEN_FIRST_V1_SYSTEM_PROMPT
    ),
}

_FEW_SHOT_MESSAGES = {
    IntruderFewShotSet.delphi: _DELPHI_FEW_SHOT_MESSAGES,
    IntruderFewShotSet.sae_highlighted_v1: _SAE_HIGHLIGHTED_FEW_SHOT_MESSAGES,
    IntruderFewShotSet.sae_balanced_v1: _SAE_BALANCED_FEW_SHOT_MESSAGES,
    IntruderFewShotSet.sae_realistic_v1: _SAE_REALISTIC_FEW_SHOT_MESSAGES,
}

_FEW_SHOT_COT_MESSAGES = {
    IntruderFewShotSet.delphi: _DELPHI_FEW_SHOT_COT_MESSAGES,
    IntruderFewShotSet.sae_highlighted_v1: _SAE_HIGHLIGHTED_FEW_SHOT_COT_MESSAGES,
    IntruderFewShotSet.sae_balanced_v1: _SAE_BALANCED_FEW_SHOT_COT_MESSAGES,
    IntruderFewShotSet.sae_realistic_v1: _SAE_REALISTIC_FEW_SHOT_COT_MESSAGES,
}
