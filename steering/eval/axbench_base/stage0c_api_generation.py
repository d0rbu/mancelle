"""Evidence-grounded OpenRouter generation for AxBench Stage 0c.

The legacy Stage 0c generator samples the local feature-extraction language
model. This module implements the default API alternative selected by
``AxBenchBaseConfig.verification_api_model``; an explicit null selects the
legacy path. Its most important boundary is
structural: generation accepts SAE feature ids, never ``Concept.description``.
The latter is Neuronpedia's published explanation and must not enter these
prompts.  Instead, a concise definition is inferred from independently fetched
top-activating Neuronpedia examples and cached for crash-safe reuse. Clustered
targets infer one such definition for every mask-allowed member and cycle those
member definitions over proposal slots; local acceptance still uses the
cluster-wide max-pooled target in ``_stage0c_internals``.

API text is still rejection sampled.  Parsed continuations flow through the
existing local SAE percentile and consistency checks in
``_stage0c_internals._verified_prefix_from_token_lists``; this module does not
decide whether a prefix passes verification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, cast

from loguru import logger

from steering.eval.axbench_base.cluster_artifacts import clustered_source_enabled
from steering.eval.axbench_base.concept_examples_qc import (
    FetchFn,
    _fetch_activating_examples_raw,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth.batching import (
    SynthCompleter,
    SynthRequest,
    SynthResult,
    complete_synth_requests,
)
from steering.eval.axbench_base.synth.continuation_guards import (
    ResponseEnvelope as _ResponseEnvelope,
)
from steering.eval.axbench_base.synth.continuation_guards import (
    make_response_envelope,
    match_prefix_echo_framing,
    parse_response_envelope,
    prefix_echo_text,
    prefix_suffix_echo_chars,
    validate_no_source_echo_or_repeated_phrase,
)
from steering.eval.axbench_base.synth.llm_response_cache import (
    ResponseCache,
    make_cache_key,
)
from steering.eval.axbench_base.synth.openrouter_client import (
    OpenRouterSynthClient,
)
from steering.eval.axbench_base.synth.response_parser import (
    get_extraction_tokenizer,
)
from steering.eval.axbench_base.templates import (
    load_template,
    render_template_with_untrusted_values,
)
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.feature_filter.types import ActivatingExample
from steering.segmentation import resolve_bos_token_id, special_token_ids

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureTarget

STAGE0C_API_PROTOCOL_VERSION = 23
CLUSTER_MEMBER_DESCRIPTION_CYCLE_VERSION = 1
_EVIDENCE_CACHE_VERSION = 2
_DEFINITION_STRATEGY_VERSION = 3
_EVIDENCE_OVERSAMPLE_FACTOR = 4
_EVIDENCE_WINDOW_CHARS = 240
_DEFINITION_TEMPLATE = "stage0c_infer_concept_definition.txt"
_CONTINUATION_TEMPLATE = "stage0c_continue_with_concept.txt"
_CONTINUATION_SYSTEM_TEMPLATE = "stage0c_continue_system.txt"
_MAX_DEFINITION_CHARS = 500
_MIN_DEFINITION_WORDS = 3
_MAX_DEFINITION_WORDS = 80
_PROVIDER_MIN_MAX_TOKENS = 1024
_PROVIDER_TOKEN_HEADROOM_MULTIPLIER = 8
_MIN_SCRIPT_LETTERS = 24
_DOMINANT_SCRIPT_FRACTION = 0.80
_MIN_DENSE_NEW_LATIN_FORMS = 6
_DENSE_NEW_LATIN_FORM_FRACTION = 0.10
_CONTINUATION_SCRIPT_FRACTION = 0.50
_MIN_UNEXPECTED_SCRIPT_LETTERS = 4
_MIN_UNEXPECTED_CJK_LETTERS = 2
_LEADING_BOILERPLATE_RE = re.compile(
    r"^(?:"
    r"sure(?:\s+thing)?|certainly|of\s+course|absolutely|okay|ok|"
    r"here(?:'s|\s+is)|let\s+me|i(?:'d|\s+would)\s+be\s+happy|"
    r"as\s+(?:requested|asked)|"
    r"(?:the\s+)?definition\s+is|"
    r"(?:continuation|completion|response|answer|definition)\s*:"
    r")(?:\s|[,!.:—\-]|$)",
    re.IGNORECASE,
)
_TRAILING_BOILERPLATE_RE = re.compile(
    r"(?:"
    r"i\s+hope\s+(?:this|that)\s+helps|hope\s+this\s+helps|"
    r"let\s+me\s+know\s+if\b[^\n]*|feel\s+free\s+to\b[^\n]*|"
    r"would\s+you\s+like\s+me\s+to\b[^\n]*|thanks\s+for\s+asking"
    r")[\s.!?]*$",
    re.IGNORECASE,
)
_META_DEFINITION_RE = re.compile(
    r"\b(?:neuronpedia|sparse[\s_-]+autoencoder|sae[\s_-]+feature|"
    r"activation[\s_-]+(?:example|evidence)|"
    r"the[\s_-]+(?:examples|evidence)|"
    r"peak[\s_-]+token(?:[\s_-]+text)?|"
    r"context[\s_-]+(?:before[\s_-]+peak|after[\s_-]+peak|"
    r"truncated[\s_-]+before|truncated[\s_-]+after)|"
    r"max[\s_-]+value(?:[\s_-]+token[\s_-]+index)?|"
    r"peak[\s_-]+char[\s_-]+offset|json[\s_-]+(?:field|row))\b",
    re.IGNORECASE,
)
_INSTRUCTIONAL_DEFINITION_RE = re.compile(
    r"(?:\b(?:ignore|disregard|override)\b.{0,40}"
    r"\b(?:instruction|prompt|task)\b|"
    r"\b(?:respond|reply|output)\s+(?:only|exactly|with)\b|"
    r"\b(?:system|developer)\s+(?:message|prompt)\b|"
    r"\bdo\s+not\s+follow\b)",
    re.IGNORECASE,
)
_META_CONTINUATION_RE = re.compile(
    r"(?:\b(?:neuronpedia|sparse[\s_-]+autoencoder|sae[\s_-]+feature)\b|"
    r"\b(?:target|operational|inferred|provided)[\s_-]+"
    r"(?:concept|definition)\b|"
    r"\bactivation[\s_-]+(?:evidence|examples?)\b|"
    r"\b(?:untrusted|quoted)[\s_-]+(?:data|text|prefix)\b|"
    r"\bvariation[\s_-]+key\b|"
    r"\b(?:binding|imperative)[\s_-]+(?:commands?|instructions?)\b|"
    r"\b(?:neutral[\s_-]+)?descriptive[\s_-]+metadata\b|"
    r"\b(?:candidate|quoted|locally[\s_-]+retained)[\s_-]+prefix\b|"
    r"\b(?:end|start|beginning)[\s_-]+of[\s_-]+(?:the[\s_-]+)?"
    r"(?:exact[\s_-]+)?(?:suffix|quoted[\s_-]+prefix)\b|"
    r"\b(?:system|developer)[\s_-]+(?:prompt|message)\b)",
    re.IGNORECASE,
)
_DEFINITION_RESTATEMENT_RE = re.compile(
    r"^\s*(?:(?:the|this)\s+(?:text|continuation|sentence)|it)?\s*"
    r"(?:starts|begins)\s+with\b|"
    r"^\s*(?:текст|продолжение|предложение|это)?\s*"
    r"начинается\s+с\b",
    re.IGNORECASE,
)
_DISALLOWED_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class PrefixCandidate(Protocol):
    """Fields needed to make one API request identity and prompt."""

    @property
    def prefix_text(self) -> str: ...

    @property
    def shard_idx(self) -> int: ...

    @property
    def doc_idx(self) -> int: ...

    @property
    def seg_idx(self) -> int: ...

    @property
    def source_bridge_text(self) -> str: ...


CompleteRequestsFn = Callable[..., list[SynthResult]]


@dataclass(frozen=True)
class ConceptDefinition:
    """A parsed definition plus the raw-evidence identity it was based on."""

    feature_id: int
    text: str
    evidence_identity: str
    recurring_peak_text: str | None


@dataclass(frozen=True)
class _ContinuationParseContext:
    envelope: _ResponseEnvelope
    request_prefix_text: str
    original_prefix_text: str
    required_source_bridge_text: str


def _template_sha256(name: str) -> str:
    return hashlib.sha256(load_template(name).encode("utf-8")).hexdigest()


def build_continuation_system_prompt(*, concept_definition: str) -> str:
    """Render the continuation-only system instruction from validated data."""
    prompt = render_template_with_untrusted_values(
        _CONTINUATION_SYSTEM_TEMPLATE,
        label="Stage 0c API continuation system",
        CONCEPT_DEFINITION=concept_definition,
    )
    if not prompt.endswith("\n"):
        raise AxBenchBaseError(
            "Stage 0c API continuation system template must end with one newline"
        )
    prompt = prompt[:-1]
    if not prompt or prompt != prompt.strip():
        raise AxBenchBaseError(
            "Stage 0c API continuation system template must contain non-empty "
            "text with no leading or trailing whitespace"
        )
    return prompt


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def api_generation_fingerprint_config(
    cfg: AxBenchBaseConfig,
) -> dict[str, Any] | None:
    """Return conditional Stage 0c API-generation semantic identity.

    ``None`` is intentional: the legacy fingerprint must retain its exact
    schema-8 field set when ``verification_api_model`` is explicitly null.
    """
    if cfg.verification_api_model is None:
        return None
    identity: dict[str, Any] = {
        "protocol_version": STAGE0C_API_PROTOCOL_VERSION,
        "provider": cfg.synth.provider,
        "model": cfg.verification_api_model,
        "requests_per_minute": cfg.synth.requests_per_minute,
        "max_concurrent": cfg.synth.max_concurrent,
        "transport_max_retries": cfg.synth.transport_max_retries,
        "prefix_wave_size": cfg.verification_api_prefix_wave_size,
        "provider_max_tokens": _provider_max_tokens(cfg.verification_length),
        "content_max_retries": cfg.max_retries,
        "n_activation_examples": cfg.n_activation_examples,
        "definition_strategy_version": _DEFINITION_STRATEGY_VERSION,
        "evidence_cache_version": _EVIDENCE_CACHE_VERSION,
        "evidence_oversample_factor": _EVIDENCE_OVERSAMPLE_FACTOR,
        "evidence_window_chars": _EVIDENCE_WINDOW_CHARS,
        "definition_prompt_sha256": _template_sha256(_DEFINITION_TEMPLATE),
        "continuation_prompt_sha256": _template_sha256(_CONTINUATION_TEMPLATE),
        "continuation_system_template_sha256": _template_sha256(_CONTINUATION_SYSTEM_TEMPLATE),
    }
    if clustered_source_enabled(cfg):
        identity["cluster_member_description_cycle"] = {
            "version": CLUSTER_MEMBER_DESCRIPTION_CYCLE_VERSION,
            "description_source": "member_activation_evidence_inference",
            "assignment": "canonical_flat_proposal_slot_round_robin",
        }
    return identity


def _provider_max_tokens(output_length: int) -> int:
    """Return provider-token headroom for an extraction-tokenizer target.

    OpenRouter's ``max_tokens`` uses the provider model's tokenizer and may
    include hidden reasoning, while Stage 0c validates exact length with
    Gemma's tokenizer. Eightfold headroom plus a 1024-token floor prevents
    truncation before the authenticated end marker without weakening the exact
    local length gate; the prompt still bounds visible output length.
    """
    return max(
        _PROVIDER_MIN_MAX_TOKENS,
        output_length * _PROVIDER_TOKEN_HEADROOM_MULTIPLIER,
    )


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AxBenchBaseError(
            f"Stage 0c API identity is not canonical-JSON serializable: {exc}"
        ) from exc


def _semantic_identity(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _script_family(char: str) -> str | None:
    """Map alphabetic Unicode code points to coarse writing-system families."""
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    for script in (
        "LATIN",
        "CYRILLIC",
        "GREEK",
        "ARABIC",
        "HEBREW",
        "DEVANAGARI",
        "BENGALI",
        "GURMUKHI",
        "GUJARATI",
        "ORIYA",
        "TAMIL",
        "TELUGU",
        "KANNADA",
        "MALAYALAM",
        "THAI",
        "LAO",
        "GEORGIAN",
        "ARMENIAN",
        "HANGUL",
    ):
        if script in name:
            return script
    if any(label in name for label in ("CJK", "HIRAGANA", "KATAKANA", "IDEOGRAPH")):
        return "CJK"
    return None


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_family(char)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1
    return counts


def _dominant_script(text: str) -> str | None:
    counts = _script_counts(text)
    total = sum(counts.values())
    if total < _MIN_SCRIPT_LETTERS or not counts:
        return None
    script, count = max(counts.items(), key=lambda item: item[1])
    if count / total < _DOMINANT_SCRIPT_FRACTION:
        return None
    return script


def _validate_script_preservation(prefix_text: str, continuation: str) -> None:
    """Reject dominant switches and unsupported foreign-script fragments."""
    prefix_script = _dominant_script(prefix_text)
    if prefix_script is None:
        return

    continuation_counts = _script_counts(continuation)
    continuation_total = sum(continuation_counts.values())
    continuation_script_count = continuation_counts.get(prefix_script, 0)
    if (
        continuation_total >= _MIN_SCRIPT_LETTERS
        and continuation_script_count / continuation_total < _CONTINUATION_SCRIPT_FRACTION
    ):
        raise AxBenchBaseError(
            f"Stage 0c API continuation changed the dominant writing system from {prefix_script}"
        )

    # A dominant-script ratio alone admitted live-observed fragments such as
    # English template residue in Russian prose and a trailing CJK phrase. The
    # exact local suffix is a better permission boundary than the full prefix:
    # a code or multilingual suffix can legitimately carry its existing
    # scripts forward, while a new script absent from that boundary is model
    # contamination. Small isolated Latin symbols remain allowed.
    local_script_counts = _script_counts(prefix_echo_text(prefix_text))
    for script, count in continuation_counts.items():
        if script == prefix_script:
            continue
        minimum = _MIN_UNEXPECTED_CJK_LETTERS if script == "CJK" else _MIN_UNEXPECTED_SCRIPT_LETTERS
        if count >= minimum and local_script_counts.get(script, 0) < minimum:
            raise AxBenchBaseError(
                "Stage 0c API continuation introduced unsupported "
                f"{script} writing-system text at a {prefix_script} local boundary"
            )


def _validate_unicode_form_preservation(
    prefix_text: str,
    continuation: str,
    recurring_peak_text: str | None,
) -> None:
    """Reject newly fabricated Unicode forms at Latin-script boundaries.

    Live Ling responses occasionally forced an uncertain orthographic concept
    by inventing a decorated ASCII letter, mathematical compatibility glyph,
    or supplementary-plane letter. Those strings can be fluent around the
    artifact and can even activate the target SAE, so script-family checks and
    SAE rejection sampling do not catch them. Keep this gate deliberately
    narrow: it applies only at a confidently Latin source boundary and permits
    every code point already present in trusted local text or in the exact
    recurring peak observed independently in Neuronpedia evidence.
    """
    if _dominant_script(prefix_text) != "LATIN":
        return

    trusted_forms = prefix_text + (recurring_peak_text or "")
    total_letters = sum(unicodedata.category(char).startswith("L") for char in continuation)
    new_non_ascii_latin_letters = 0
    for char in continuation:
        if char in trusted_forms:
            continue
        category = unicodedata.category(char)
        unsupported_combining_mark = category.startswith("M")
        compatibility_letter = (
            category.startswith("L") and unicodedata.normalize("NFKC", char) != char
        )
        supplementary_letter = category.startswith("L") and ord(char) > 0xFFFF
        unassigned_character = category == "Cn"
        script = _script_family(char)
        foreign_letter = category.startswith("L") and script not in {
            None,
            "LATIN",
        }
        if category.startswith("L") and script == "LATIN" and not char.isascii():
            new_non_ascii_latin_letters += 1
        if (
            unsupported_combining_mark
            or compatibility_letter
            or supplementary_letter
            or unassigned_character
            or foreign_letter
        ):
            raise AxBenchBaseError(
                "Stage 0c API continuation introduced an unsupported Unicode "
                "letter form at a Latin local boundary"
            )
    if (
        new_non_ascii_latin_letters >= _MIN_DENSE_NEW_LATIN_FORMS
        and new_non_ascii_latin_letters / max(total_letters, 1) >= _DENSE_NEW_LATIN_FORM_FRACTION
    ):
        raise AxBenchBaseError(
            "Stage 0c API continuation introduced a dense cluster of unsupported "
            "Unicode letter forms at a Latin local boundary"
        )


def _retry_instruction_for_error(error: AxBenchBaseError | None) -> str:
    """Return bounded corrective guidance without reflecting response text."""
    if error is None:
        return (
            "This is the first attempt. Follow the literal envelope, suffix, "
            "single-block, naturalness, and no-copy requirements exactly."
        )
    message = str(error)
    if "assistant boilerplate" in message:
        return (
            "The prior output added assistant chatter. Start with the exact "
            "begin marker and return only the authenticated source text."
        )
    if "exact local prefix suffix" in message:
        return (
            "The prior output copied the final suffix incorrectly. Immediately "
            "after the begin-marker newline, copy every suffix character from "
            "its first character onward, then continue with no separator."
        )
    if "matching begin and end marker" in message or "end marker occurs before" in message:
        return (
            "The prior output used the envelope markers incorrectly. Emit the "
            "exact begin marker once as the first line and the exact end marker "
            "once as the final line, in that order."
        )
    if "paragraph break" in message:
        return (
            "The prior output inserted an early paragraph break. Add no newline "
            "inside the continuation block."
        )
    if "unsupported Unicode letter form" in message:
        return (
            "The prior output fabricated a Unicode letter form. Use only normal "
            "source-language words and characters already appropriate to the "
            "source; do not decorate, mutate, or synthesize glyphs."
        )
    if (
        "copied a substantial source phrase" in message
        or "repeated the locally retained candidate prefix" in message
        or "from the candidate prefix suffix" in message
    ):
        return (
            "The prior output copied source text beyond the mandatory suffix. "
            "Copy only that exact suffix, then move forward with wholly new prose."
        )
    if "multi-word definition" in message or "writing-property definition" in message:
        return (
            "The prior output repeated the property description. Instantiate it "
            "without quoting, defining, naming, or paraphrasing it."
        )
    if "writing system" in message or "writing-system" in message:
        return (
            "The prior output changed writing systems. Use only the source's "
            "dominant language and writing system."
        )
    if "repeated phrase loop" in message:
        return (
            "The prior output looped a phrase. Continue with varied, nonrepeating "
            "sentences and no refrain."
        )
    if "recurring peak text" in message:
        return (
            "The prior output mishandled the required lexical form. Follow the "
            "exact-form rule once in ordinary grammar without quoting or "
            "repetition."
        )
    return (
        "The prior output failed local validation. Re-read every literal-format "
        "and naturalness requirement and produce a fresh corrected continuation."
    )


def _validate_recurring_peak_text(
    prefix_text: str,
    continuation: str,
    recurring_peak_text: str | None,
    *,
    require_occurrence: bool,
) -> None:
    """Require natural same-script anchors and reject foreign-script copies."""
    if recurring_peak_text is None:
        return
    normalized_peak = recurring_peak_text.strip()
    if not normalized_peak:
        return
    escaped_peak = re.escape(normalized_peak)
    if re.search(
        rf"(?<!\w){escaped_peak}(?!\w)(?:[\s,;:—\-]+)"
        rf"{escaped_peak}(?!\w)",
        continuation,
        re.IGNORECASE,
    ):
        raise AxBenchBaseError(
            "Stage 0c API continuation repeated recurring peak text "
            "adjacently instead of using it naturally"
        )
    meta_anchor_noun = (
        r"(?:word|token|sequence|form|term|particle|preposition|use|usage|"
        r"слово|токен|последовательность|форма|термин|частица|предлог|"
        r"употребление)"
    )
    meta_anchor_verb = (
        r"(?:means|functions|serves|sounds|appears|is\s+used|"
        r"означает|функционирует|служит|звучит|появляется|"
        r"используется|является|выступает)"
    )
    quote = r"[\"'“”‘’«»„‟]"
    if re.search(
        rf"{quote}\s*{re.escape(normalized_peak)}\s*{quote}",
        continuation,
        re.IGNORECASE,
    ):
        raise AxBenchBaseError(
            "Stage 0c API continuation quoted recurring peak text instead of using it naturally"
        )
    prefix_script = _dominant_script(prefix_text)
    anchor_counts = _script_counts(normalized_peak)
    if prefix_script is None or not anchor_counts:
        return
    anchor_script, anchor_count = max(anchor_counts.items(), key=lambda item: item[1])
    if anchor_count != sum(anchor_counts.values()):
        return
    peak_occurrences = len(
        re.findall(
            rf"(?<!\w){escaped_peak}(?!\w)",
            continuation,
            re.IGNORECASE,
        )
    )
    if anchor_script == prefix_script:
        if re.search(
            rf"(?:{meta_anchor_noun}\s+{escaped_peak}(?!\w)|"
            rf"(?<!\w){escaped_peak}\s+{meta_anchor_verb})",
            continuation,
            re.IGNORECASE,
        ):
            raise AxBenchBaseError(
                "Stage 0c API continuation discussed recurring peak text "
                "metalinguistically instead of using it naturally"
            )
        if require_occurrence and peak_occurrences == 0:
            raise AxBenchBaseError(
                "Stage 0c API continuation omitted the exact recurring peak "
                "text required at this same-writing-system source boundary"
            )
        if peak_occurrences > 2:
            raise AxBenchBaseError(
                "Stage 0c API continuation overused the exact recurring peak "
                "text instead of using it naturally once or twice"
            )
        return
    if peak_occurrences:
        raise AxBenchBaseError(
            "Stage 0c API continuation copied recurring peak text from a "
            "different writing system instead of using the definition's "
            "source-language equivalent"
        )


def _has_definition_copy(
    continuation: str,
    concept_definition: str | None,
) -> bool:
    if concept_definition is None:
        return False
    definition_words = re.findall(r"[^\W_]+", concept_definition.casefold())
    continuation_words = re.findall(r"[^\W_]+", continuation.casefold())
    if len(definition_words) < 4 or len(continuation_words) < 4:
        return False
    definition_ngrams = {
        tuple(definition_words[index : index + 4])
        for index in range(len(definition_words) - 3)
        if len("".join(definition_words[index : index + 4])) >= 20
    }
    for index in range(len(continuation_words) - 3):
        if tuple(continuation_words[index : index + 4]) in definition_ngrams:
            return True
    return False


def _strip_leading_definition_sentence(
    continuation: str,
    concept_definition: str | None,
) -> str:
    """Remove one complete leading definition-copy sentence, if separable.

    Ling sometimes emits a meta sentence and then a clean, source-like
    continuation. We strip only when the copied four-word phrase is wholly
    inside the first complete sentence. Copies without a sentence boundary,
    copies later in the prose, and repeated copies remain hard failures.
    """
    if not _has_definition_copy(continuation, concept_definition):
        return continuation
    boundary = re.search(r"[.!?](?:[\"'”’\)\]]*)\s+", continuation)
    if boundary is None:
        return continuation
    leading_sentence = continuation[: boundary.end()]
    if not _has_definition_copy(leading_sentence, concept_definition):
        return continuation
    remainder = continuation[boundary.end() :].lstrip()
    if not remainder:
        return continuation
    record_counter("stage0c.api.leading_definition_sentences_stripped", 1)
    separator = "\n" if "\n" in boundary.group(0) else " "
    return separator + remainder


def _validate_no_definition_copy(
    continuation: str,
    concept_definition: str | None,
) -> None:
    if _has_definition_copy(continuation, concept_definition):
        raise AxBenchBaseError(
            "Stage 0c API continuation copied a multi-word definition "
            "phrase instead of instantiating the writing property"
        )


def _make_envelope(
    kind: str,
    identity: object,
    *,
    untrusted_texts: Sequence[str],
) -> _ResponseEnvelope:
    """Build deterministic markers guaranteed absent from untrusted inputs."""
    return make_response_envelope(
        kind,
        identity,
        untrusted_texts=untrusted_texts,
        protocol_version=STAGE0C_API_PROTOCOL_VERSION,
        semantic_identity=_semantic_identity,
        error_subject="Stage 0c API",
        error_type=AxBenchBaseError,
    )


def _parse_envelope(
    response: str,
    envelope: _ResponseEnvelope,
    *,
    tolerate_end_marker_variants: bool = False,
) -> str:
    """Extract one well-ordered envelope, optionally normalizing audited end variants."""
    return parse_response_envelope(
        response,
        envelope,
        tolerate_end_marker_variants=tolerate_end_marker_variants,
        error_subject="Stage 0c API",
        counter_prefix="stage0c.api",
        record_counter_fn=record_counter,
        error_type=AxBenchBaseError,
    )


def _has_leading_boilerplate(text: str) -> bool:
    stripped = text.lstrip()
    return bool(_LEADING_BOILERPLATE_RE.search(stripped) or stripped.startswith("```"))


def _has_trailing_boilerplate(text: str) -> bool:
    return bool(_TRAILING_BOILERPLATE_RE.search(text.rstrip()))


def _validate_continuation_semantics(
    text: str,
    *,
    prefix_text: str,
    concept_definition: str | None,
    recurring_peak_text: str | None,
    scored_window: bool,
    require_recurring_peak: bool = True,
) -> None:
    """Apply the same semantic-artifact gate before and after truncation."""
    if scored_window:
        if _has_leading_boilerplate(text) or _has_trailing_boilerplate(text):
            raise AxBenchBaseError(
                "Stage 0c API scored token window contains assistant boilerplate"
            )
    else:
        if _has_leading_boilerplate(text):
            raise AxBenchBaseError("Stage 0c API continuation begins with assistant boilerplate")
        if _has_trailing_boilerplate(text):
            raise AxBenchBaseError("Stage 0c API continuation ends with assistant boilerplate")

    if _META_CONTINUATION_RE.search(text):
        if scored_window:
            raise AxBenchBaseError(
                "Stage 0c API scored token window describes prompt or interpretation machinery"
            )
        raise AxBenchBaseError(
            "Stage 0c API continuation describes prompt or interpretation "
            "machinery instead of naturally continuing the source text"
        )
    if _DEFINITION_RESTATEMENT_RE.search(text):
        if scored_window:
            raise AxBenchBaseError(
                "Stage 0c API scored token window restates a writing-property definition"
            )
        raise AxBenchBaseError(
            "Stage 0c API continuation restates a writing-property definition "
            "instead of instantiating it"
        )
    subject = (
        "Stage 0c API scored token window" if scored_window else "Stage 0c API continuation"
    )
    validate_no_source_echo_or_repeated_phrase(
        text,
        source_text=prefix_text,
        subject=subject,
    )
    _validate_no_definition_copy(text, concept_definition)
    _validate_script_preservation(prefix_text, text)
    _validate_unicode_form_preservation(
        prefix_text,
        text,
        recurring_peak_text,
    )
    _validate_recurring_peak_text(
        prefix_text,
        text,
        recurring_peak_text,
        require_occurrence=require_recurring_peak,
    )


def _validate_unbridged_boundary(prefix_text: str, continuation: str) -> None:
    """Reject an invented paragraph break across an unfinished source line.

    Production API requests carry an exact source bridge and do not depend on
    this heuristic.  It remains a fail-closed guard for direct parser callers
    and extension requests, where no post-boundary source characters exist.
    """
    if not re.match(r"^[ \t]*\r?\n", continuation):
        return
    stripped_prefix = prefix_text.rstrip(" \t")
    if not stripped_prefix or stripped_prefix.endswith(("\n", "\r")):
        return
    if re.search(r"[.!?…:;\"'”’)\]]$", stripped_prefix):
        return
    raise AxBenchBaseError(
        "Stage 0c API continuation inserted a paragraph break before "
        "finishing the source's current word, sentence, or line"
    )


def parse_definition_response(
    response: str,
    *,
    envelope: _ResponseEnvelope,
) -> str:
    """Parse one concise, non-meta concept definition."""
    raw_definition = _parse_envelope(response, envelope)
    definition = raw_definition.strip()
    if definition != raw_definition:
        record_counter("stage0c.api.normalized_definition_framing", 1)
    if not definition:
        raise AxBenchBaseError("Stage 0c API definition is empty")
    if "\n" in definition or "\r" in definition:
        raise AxBenchBaseError("Stage 0c API definition must be a single line")
    if _DISALLOWED_CONTROL_RE.search(definition):
        raise AxBenchBaseError("Stage 0c API definition contains control characters")
    if len(definition) > _MAX_DEFINITION_CHARS:
        raise AxBenchBaseError(
            f"Stage 0c API definition exceeds {_MAX_DEFINITION_CHARS} characters"
        )
    word_count = len(definition.split())
    if not _MIN_DEFINITION_WORDS <= word_count <= _MAX_DEFINITION_WORDS:
        raise AxBenchBaseError(
            "Stage 0c API definition must contain between "
            f"{_MIN_DEFINITION_WORDS} and {_MAX_DEFINITION_WORDS} words, got "
            f"{word_count}"
        )
    if _has_leading_boilerplate(definition) or _has_trailing_boilerplate(definition):
        raise AxBenchBaseError("Stage 0c API definition contains assistant boilerplate")
    if _META_DEFINITION_RE.search(definition):
        raise AxBenchBaseError(
            "Stage 0c API definition describes the interpretation machinery instead of the concept"
        )
    if _INSTRUCTIONAL_DEFINITION_RE.search(definition):
        raise AxBenchBaseError(
            "Stage 0c API definition contains prompt-injection-like instructions"
        )
    script_counts = _script_counts(definition)
    script_letters = sum(script_counts.values())
    if (
        script_letters >= _MIN_SCRIPT_LETTERS
        and script_counts.get("LATIN", 0) / script_letters < _DOMINANT_SCRIPT_FRACTION
    ):
        raise AxBenchBaseError(
            "Stage 0c API definition must be written in clear English except "
            "for concise quoted foreign-language forms"
        )
    if "AXBENCH_" in definition:
        raise AxBenchBaseError("Stage 0c API definition contains a protocol-like marker")
    return definition


def _parse_continuation_text(
    response: str,
    *,
    envelope: _ResponseEnvelope,
    prefix_text: str,
    concept_definition: str | None = None,
    recurring_peak_text: str | None = None,
    required_source_bridge_text: str = "",
    require_recurring_peak: bool = True,
) -> str:
    """Return one authenticated, semantically clean continuation string.

    Chatter outside the authenticated envelope is deliberately discarded. The
    provider must first repeat a short exact suffix of the prefix plus trusted
    source bridge, proving that it wrote against the actual local boundary. The
    echo is authenticated and removed. The bridge itself never comes from the
    provider: it is prepended from local source data after parsing, making
    arbitrary mid-word and mid-clause candidate cuts unambiguous.
    """
    payload = _parse_envelope(
        response,
        envelope,
        tolerate_end_marker_variants=True,
    )
    # The exact suffix has a known character length, so it both focuses the
    # provider on the real source boundary and defines the parse boundary. A
    # separate split sentinel would be redundant and live Ling responses often
    # placed it after, rather than before, the continuation. The first character
    # after the suffix is continuation data, preserving a required leading
    # space, newline, punctuation mark, or partial-word completion.
    request_context_text = prefix_text + required_source_bridge_text
    framing = match_prefix_echo_framing(payload, request_context_text)
    if framing is None:
        raise AxBenchBaseError(
            "Stage 0c API continuation must repeat the exact local prefix "
            "suffix immediately after the begin-marker newline, append new "
            "continuation data directly at the fixed suffix boundary"
        )
    prefix_echo = framing.exact_echo
    framing_prefix = framing.text
    if framing.used_left_truncation:
        record_counter("stage0c.api.normalized_left_truncated_prefix_echo", 1)
    if framing_prefix not in {"\n" + prefix_echo, "\r\n" + prefix_echo}:
        record_counter("stage0c.api.normalized_prefix_echo_framing", 1)
    framed_continuation = payload[len(framing_prefix) :]
    continuation = framed_continuation.rstrip(" \t\r\n")
    final_separator = framed_continuation[len(continuation) :]
    if final_separator != " ":
        record_counter(
            "stage0c.api.normalized_end_marker_separator",
            1,
            separator_chars=len(final_separator),
        )
    if not continuation or not continuation.strip():
        raise AxBenchBaseError("Stage 0c API continuation is empty")
    # For a substantial prefix, an exact repeat anywhere is almost certainly a
    # failed echo despite the output-only contract. Short strings can naturally
    # recur in prose, so they cannot support this inference.
    if len(request_context_text) >= 32 and request_context_text in continuation:
        raise AxBenchBaseError(
            "Stage 0c API response repeated the locally retained candidate prefix"
        )
    suffix_echo_chars = prefix_suffix_echo_chars(
        request_context_text,
        continuation,
    )
    if suffix_echo_chars:
        raise AxBenchBaseError(
            "Stage 0c API response repeated "
            f"{suffix_echo_chars} character(s) from the candidate prefix suffix"
        )
    if _DISALLOWED_CONTROL_RE.search(continuation):
        raise AxBenchBaseError("Stage 0c API continuation contains disallowed control characters")
    if "AXBENCH_" in continuation:
        raise AxBenchBaseError("Stage 0c API continuation contains a protocol-like marker")
    continuation = _strip_leading_definition_sentence(
        continuation,
        concept_definition,
    )
    _validate_continuation_semantics(
        continuation,
        prefix_text=request_context_text,
        concept_definition=concept_definition,
        recurring_peak_text=recurring_peak_text,
        scored_window=False,
        require_recurring_peak=require_recurring_peak,
    )
    _validate_unbridged_boundary(request_context_text, continuation)
    return required_source_bridge_text + continuation


def _encode_continuation_text(continuation: str, tokenizer: Any) -> list[int]:
    """Encode continuation text and fail on any tokenizer control token."""

    raw_ids = tokenizer.encode(continuation, add_special_tokens=False)
    token_ids = [int(token_id) for token_id in raw_ids]
    bos_id = resolve_bos_token_id(tokenizer)
    polluted = special_token_ids(tokenizer, bos_id).intersection(token_ids)
    if polluted:
        raise AxBenchBaseError(
            "Stage 0c API continuation encodes special token id(s) "
            f"{sorted(polluted)} with add_special_tokens=False"
        )
    return token_ids


def _truncate_and_validate_continuation(
    token_ids: list[int],
    *,
    output_length: int,
    tokenizer: Any,
    prefix_text: str,
    concept_definition: str | None,
    recurring_peak_text: str | None,
    trusted_leading_text: str = "",
) -> list[int]:
    """Return one exact scored window after post-tokenization validation."""
    if len(token_ids) < output_length:
        raise AxBenchBaseError(
            "Stage 0c API continuation is too short: encoded to "
            f"{len(token_ids)} extraction tokens, need at least {output_length}"
        )
    truncated = token_ids[:output_length]
    decoded = tokenizer.decode(truncated)
    semantic_text = decoded
    semantic_prefix_text = prefix_text
    if trusted_leading_text:
        if trusted_leading_text.startswith(decoded):
            semantic_text = ""
            semantic_prefix_text += decoded
        elif decoded.startswith(trusted_leading_text):
            semantic_text = decoded[len(trusted_leading_text) :]
            semantic_prefix_text += trusted_leading_text
        else:
            raise AxBenchBaseError(
                "Stage 0c extraction tokenizer did not preserve the exact local "
                "source bridge at the start of the scored token window"
            )
    _validate_continuation_semantics(
        semantic_text,
        prefix_text=semantic_prefix_text,
        concept_definition=concept_definition,
        recurring_peak_text=recurring_peak_text,
        scored_window=True,
    )
    return truncated


def _validate_examples_payload(
    payload: object,
    *,
    feature_id: int,
    max_examples: int,
) -> list[ActivatingExample]:
    if not isinstance(payload, list) or not payload:
        raise AxBenchBaseError(f"Stage 0c API feature {feature_id} has no activation evidence")
    if len(payload) > max_examples:
        raise AxBenchBaseError(
            f"Stage 0c API feature {feature_id} evidence contains "
            f"{len(payload)} rows, configured maximum is {max_examples}"
        )
    expected_fields = {
        "tokens",
        "text",
        "max_value",
        "max_value_token_index",
        "peak_char_offset",
    }
    examples: list[ActivatingExample] = []
    for row_index, raw_row in enumerate(payload):
        if not isinstance(raw_row, dict) or set(raw_row) != expected_fields:
            raise AxBenchBaseError(
                f"Stage 0c API feature {feature_id} evidence row {row_index} "
                f"must have fields {sorted(expected_fields)!r}"
            )
        typed_row = cast(dict[str, object], raw_row)
        tokens = typed_row["tokens"]
        text = typed_row["text"]
        max_value = typed_row["max_value"]
        max_idx = typed_row["max_value_token_index"]
        peak_offset = typed_row["peak_char_offset"]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(token, str) for token in tokens)
        ):
            raise AxBenchBaseError(
                f"Stage 0c API feature {feature_id} evidence row {row_index} has invalid tokens"
            )
        if not isinstance(text, str) or text == "":
            raise AxBenchBaseError(
                f"Stage 0c API feature {feature_id} evidence row {row_index} "
                "has empty or non-string text"
            )
        if (
            not isinstance(max_value, (int, float))
            or isinstance(max_value, bool)
            or not math.isfinite(float(max_value))
            or float(max_value) <= 0.0
        ):
            raise AxBenchBaseError(
                f"Stage 0c API feature {feature_id} evidence row {row_index} "
                f"has invalid max_value={max_value!r}"
            )
        if (
            not isinstance(max_idx, int)
            or isinstance(max_idx, bool)
            or not 0 <= max_idx < len(tokens)
        ):
            raise AxBenchBaseError(
                f"Stage 0c API feature {feature_id} evidence row {row_index} "
                f"has invalid max_value_token_index={max_idx!r}"
            )
        if (
            not isinstance(peak_offset, int)
            or isinstance(peak_offset, bool)
            or not 0 <= peak_offset <= len(text)
        ):
            raise AxBenchBaseError(
                f"Stage 0c API feature {feature_id} evidence row {row_index} "
                f"has invalid peak_char_offset={peak_offset!r}"
            )
        examples.append(
            ActivatingExample(
                tokens=list(cast(list[str], tokens)),
                text=text,
                max_value=float(max_value),
                max_value_token_index=max_idx,
                peak_char_offset=peak_offset,
            )
        )
    return examples


def _evidence_row(
    example: ActivatingExample,
    *,
    rank: int,
    tokenizer: Any,
) -> dict[str, Any]:
    """Render peak-token evidence without inserting text into the excerpt.

    An earlier marker-insertion format put a synthetic marker immediately
    after the preceding word. A live model then interpreted that artificial
    adjacency as the feature's concept. Keeping before/token/after in separate
    JSON fields makes the annotation boundary explicit and leaves all source
    text byte-exact.
    """
    tokens = example.tokens
    peak_index = example.max_value_token_index
    prefix = tokenizer.convert_tokens_to_string(tokens[:peak_index])
    through_peak = tokenizer.convert_tokens_to_string(tokens[: peak_index + 1])
    if (
        len(prefix) != example.peak_char_offset
        or not through_peak.startswith(prefix)
        or not example.text.startswith(through_peak)
        or len(through_peak) <= len(prefix)
    ):
        raise AxBenchBaseError(
            "Stage 0c API could not align activation evidence peak token "
            f"for rank {rank}: stored_offset={example.peak_char_offset}, "
            f"recomputed_offset={len(prefix)}"
        )
    peak_end = len(through_peak)
    before_start = max(0, example.peak_char_offset - _EVIDENCE_WINDOW_CHARS)
    after_end = min(len(example.text), peak_end + _EVIDENCE_WINDOW_CHARS)
    return {
        "rank": rank,
        "context_before_peak": example.text[before_start : example.peak_char_offset],
        "peak_token_text": example.text[example.peak_char_offset : peak_end],
        "context_after_peak": example.text[peak_end:after_end],
        "context_truncated_before": before_start > 0,
        "context_truncated_after": after_end < len(example.text),
    }


def _evidence_prompt_json(
    examples: Sequence[ActivatingExample],
    *,
    tokenizer: Any,
) -> str:
    return json.dumps(
        [
            _evidence_row(example, rank=rank, tokenizer=tokenizer)
            for rank, example in enumerate(examples, start=1)
        ],
        ensure_ascii=False,
        indent=2,
    )


def _recurring_peak_text(
    examples: Sequence[ActivatingExample],
    *,
    tokenizer: Any,
) -> str | None:
    """Return a shared, whitespace-normalized peak sequence if one exists.

    This is a trusted local observation, not a semantic interpretation. It is
    supplied as a binding anchor to the definition model so a recurring token
    cannot be misread as a neighboring-character feature. Exact recurrence
    preserves case (so repeated ``NASA`` remains ``NASA``). If examples differ
    *only* by case, return their case-folded sequence instead; this covers
    sentence-initial variants such as ``На``/``на`` without inventing a
    capitalization concept. The binding prompt explicitly forbids that
    inference.
    """
    if len(examples) < 2:
        return None
    peak_texts = [
        cast(
            str,
            _evidence_row(example, rank=rank, tokenizer=tokenizer)["peak_token_text"],
        )
        for rank, example in enumerate(examples, start=1)
    ]
    normalized = [" ".join(text.split()) for text in peak_texts]
    if not normalized[0]:
        return None
    if all(text == normalized[0] for text in normalized[1:]):
        return normalized[0]
    casefolded = [text.casefold() for text in normalized]
    if all(text == casefolded[0] for text in casefolded[1:]):
        return casefolded[0]
    return None


def build_definition_prompt(
    *,
    evidence_json: str,
    envelope: _ResponseEnvelope,
    recurring_peak_text: str | None = None,
) -> str:
    peak_constraint = (
        "Every independent peak_token_text value normalizes to the exact "
        "character sequence "
        f"{json.dumps(recurring_peak_text, ensure_ascii=False)}. The definition "
        "must quote that sequence literally and explain its semantic, syntactic, "
        "lexical, or formatting role in context. Include a translation or "
        "same-language equivalent when that helps a writer instantiate the "
        "property in the prefix's language. Do not infer capitalization or "
        "neighboring-letter rules merely from this recurrence."
        if recurring_peak_text is not None
        else (
            "No exact character sequence recurs at every independent peak; "
            "infer conservatively from the full evidence rows."
        )
    )
    return render_template_with_untrusted_values(
        _DEFINITION_TEMPLATE,
        label="Stage 0c API definition",
        BEGIN_MARKER=envelope.begin,
        END_MARKER=envelope.end,
        PEAK_TOKEN_CONSTRAINT=peak_constraint,
        EVIDENCE_JSON=evidence_json,
    )


def build_continuation_prompt(
    *,
    prefix_text: str,
    concept_definition: str,
    variation_key: int,
    output_length: int,
    envelope: _ResponseEnvelope,
    recurring_peak_text: str | None = None,
    source_bridge_text: str = "",
    retry_instruction: str | None = None,
) -> str:
    source_context_text = prefix_text + source_bridge_text
    prefix_echo = prefix_echo_text(source_context_text)
    # Ling reliably clears the 64-token extraction target with this shorter
    # visible-output range. The prior 120-180-word/two-paragraph request caused
    # frequent premature paragraph breaks and unnecessary extension/retry work.
    # Scale larger non-default targets conservatively instead of hard-coding
    # the default benchmark range.
    min_words = max(90, math.ceil(output_length * 1.4))
    max_words = min_words + 40
    prefix_script = _dominant_script(prefix_text)
    peak_counts = _script_counts(recurring_peak_text or "")
    peak_script = max(peak_counts.items(), key=lambda item: item[1])[0] if peak_counts else None
    quoted_peak = json.dumps(recurring_peak_text, ensure_ascii=False)
    if recurring_peak_text is None:
        peak_text_constraint = (
            "No exact lexical form is required. Instantiate the writing "
            "property naturally without discussing its description."
        )
    elif prefix_script is not None and peak_script == prefix_script:
        peak_text_constraint = (
            f"The exact same-writing-system form is {quoted_peak}. Use that "
            "sequence naturally exactly once (twice only if the prose clearly "
            "requires it), but never quote, name, define, or translate it and "
            "never substitute a foreign-language gloss from the property "
            "description. Every occurrence must be grammatically ordinary; "
            "never place two occurrences adjacently."
        )
    elif prefix_script is not None and peak_script is not None:
        peak_text_constraint = (
            f"The exact form {quoted_peak} uses a different writing system "
            "from the source. Do not copy it; naturally instantiate its "
            "source-language equivalent instead."
        )
    else:
        peak_text_constraint = (
            f"Use the exact form {quoted_peak} naturally only when it belongs "
            "in the source language; never quote, name, define, or translate it."
        )
    prompt = render_template_with_untrusted_values(
        _CONTINUATION_TEMPLATE,
        label="Stage 0c API continuation",
        BEGIN_MARKER=envelope.begin,
        END_MARKER=envelope.end,
        PREFIX_TEXT=source_context_text,
        PREFIX_SUFFIX=prefix_echo,
        ECHO_CHARS=len(prefix_echo),
        CONCEPT_DEFINITION=concept_definition,
        PEAK_TEXT_CONSTRAINT=peak_text_constraint,
        RETRY_INSTRUCTION=(
            retry_instruction
            if retry_instruction is not None
            else _retry_instruction_for_error(None)
        ),
        VARIATION_KEY=variation_key,
        MIN_WORDS=min_words,
        MAX_WORDS=max_words,
    )
    if not prompt.endswith("\n"):
        raise AxBenchBaseError(
            "Stage 0c API continuation template must end with one trusted newline"
        )
    # The file's final newline is a source-formatting detail, not part of the
    # quoted bridge. Removing exactly that byte makes the request truly end at
    # the authenticated local boundary, including when the bridge itself ends
    # in a meaningful newline.
    return prompt[:-1]


def _proposal_member_ids(
    feature_id: int,
    cluster_target: ClusterFeatureTarget | None,
) -> tuple[int, ...]:
    """Return the validated, canonical member-description cycle.

    The immutable cluster target has already passed feature-contract
    construction. This API boundary nevertheless reasserts its load-bearing
    fields so a stale or hand-built target cannot silently guide proposals for
    a different representative. A one-member result intentionally takes the
    exact legacy proposal path.
    """

    if cluster_target is None:
        return (feature_id,)
    if cluster_target.representative_id != feature_id:
        raise AxBenchBaseError(
            "Stage 0c API cluster target representative does not match the "
            f"requested feature: {cluster_target.representative_id} != {feature_id}"
        )
    member_ids = cluster_target.allowed_member_ids
    if (
        not isinstance(member_ids, tuple)
        or not member_ids
        or member_ids[0] != feature_id
        or tuple(sorted(set(member_ids))) != member_ids
        or any(type(member_id) is not int or member_id < 0 for member_id in member_ids)
    ):
        raise AxBenchBaseError(
            "Stage 0c API cluster target members must be a non-empty sorted "
            "unique tuple of strict non-negative ints beginning with the representative"
        )
    return member_ids


class Stage0cApiGenerator:
    """Generate shape-stable Stage 0c token rows through OpenRouter.

    Transport, response-cache, tokenizer, and concurrency state live for the
    full Stage 0c run. Evidence-derived operational definitions live only for
    the active concept: they are reused across all of that concept's proposal
    waves, then released by :meth:`_release_concept_state`.
    """

    def __init__(
        self,
        cfg: AxBenchBaseConfig,
        *,
        response_cache: ResponseCache,
        llm_client: SynthCompleter | None = None,
        fetch_examples_fn: FetchFn = _fetch_activating_examples_raw,
        tokenizer: Any | None = None,
        complete_requests_fn: CompleteRequestsFn = complete_synth_requests,
    ) -> None:
        if cfg.verification_api_model is None:
            raise AxBenchBaseError("Stage0cApiGenerator requires verification_api_model to be set")
        self._cfg = cfg
        self._response_cache = response_cache
        self._provider_max_tokens = _provider_max_tokens(
            cfg.verification_length,
        )
        effective_client_cfg = replace(
            cfg.synth,
            model=cfg.verification_api_model,
        )
        self._client = llm_client or OpenRouterSynthClient(
            effective_client_cfg,
            max_retries=cfg.synth.transport_max_retries,
            response_cache=response_cache,
            max_tokens=self._provider_max_tokens,
        )
        self._fetch_examples_fn = fetch_examples_fn
        self._tokenizer = tokenizer
        self._complete_requests_fn = complete_requests_fn
        self._definitions: dict[int, ConceptDefinition] = {}

    def _release_concept_state(self) -> None:
        """Release operational definitions after the active concept ends.

        Candidate features are unique and schema-v1 clusters have disjoint
        membership, so definitions have no cross-concept reuse. Keeping this
        cleanup separate from :meth:`generate_token_rows` is load-bearing:
        one concept may call that method for several API waves and every wave
        must reuse the same validated definitions.
        """
        self._definitions.clear()

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = get_extraction_tokenizer(self._cfg.extraction_tokenizer_id)
        return self._tokenizer

    def _evidence_cache_key(self, feature_id: int) -> str:
        identity = {
            "cache_version": _EVIDENCE_CACHE_VERSION,
            "neuronpedia_model_id": self._cfg.neuronpedia_model_id,
            "neuronpedia_source": self._cfg.neuronpedia_source,
            "feature_id": feature_id,
            "n_activation_examples": self._cfg.n_activation_examples,
            "extraction_tokenizer_id": self._cfg.extraction_tokenizer_id,
        }
        return make_cache_key(
            model="neuronpedia-activation-evidence",
            prompt=_canonical_json(identity),
            tag="stage0c_api_evidence",
        )

    def _load_or_fetch_examples(self, feature_id: int) -> list[ActivatingExample]:
        cache_key = self._evidence_cache_key(feature_id)
        cached = self._response_cache.get(cache_key)
        if cached is not None:
            try:
                payload = json.loads(cached)
            except json.JSONDecodeError as exc:
                raise AxBenchBaseError(
                    "Cached Stage 0c API activation evidence is malformed for "
                    f"feature {feature_id}: {exc}"
                ) from exc
            validated_cached = _validate_examples_payload(
                payload,
                feature_id=feature_id,
                max_examples=self._cfg.n_activation_examples,
            )
            cached_identities = [
                (example.text, example.peak_char_offset) for example in validated_cached
            ]
            if len(set(cached_identities)) != len(cached_identities):
                raise AxBenchBaseError(
                    "Cached Stage 0c API activation evidence contains "
                    f"duplicate source rows for feature {feature_id}"
                )
            return validated_cached

        with profile_region(
            "stage0c.api.fetch_activation_evidence",
            feature_id=feature_id,
        ):
            fetch_limit = self._cfg.n_activation_examples * _EVIDENCE_OVERSAMPLE_FACTOR
            examples = self._fetch_examples_fn(
                self._cfg.neuronpedia_model_id,
                self._cfg.neuronpedia_source,
                feature_id,
                fetch_limit,
                self.tokenizer,
            )
        validated_fetched = _validate_examples_payload(
            [asdict(example) for example in examples],
            feature_id=feature_id,
            max_examples=fetch_limit,
        )
        unique_examples: list[ActivatingExample] = []
        seen: set[tuple[str, int]] = set()
        for example in validated_fetched:
            identity = (example.text, example.peak_char_offset)
            if identity in seen:
                continue
            seen.add(identity)
            unique_examples.append(example)
            if len(unique_examples) == self._cfg.n_activation_examples:
                break
        validated = _validate_examples_payload(
            [asdict(example) for example in unique_examples],
            feature_id=feature_id,
            max_examples=self._cfg.n_activation_examples,
        )
        encoded = _canonical_json([asdict(example) for example in validated])
        self._response_cache.set(cache_key, encoded)
        record_counter("stage0c.api.activation_evidence_fetches", 1)
        return validated

    def _definition_for_feature(self, feature_id: int) -> ConceptDefinition:
        cached_definition = self._definitions.get(feature_id)
        if cached_definition is not None:
            return cached_definition

        examples = self._load_or_fetch_examples(feature_id)
        evidence_json = _evidence_prompt_json(
            examples,
            tokenizer=self.tokenizer,
        )
        evidence_identity = _semantic_identity([asdict(example) for example in examples])
        recurring_peak_text = _recurring_peak_text(
            examples,
            tokenizer=self.tokenizer,
        )
        max_attempts = self._cfg.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            envelope = _make_envelope(
                "DEFINITION",
                {
                    "feature_id": feature_id,
                    "evidence_identity": evidence_identity,
                    "attempt": attempt,
                },
                untrusted_texts=[evidence_json],
            )
            prompt = build_definition_prompt(
                evidence_json=evidence_json,
                envelope=envelope,
                recurring_peak_text=recurring_peak_text,
            )
            cache_key = make_cache_key(
                model=self._cfg.verification_api_model or "",
                prompt=prompt,
                tag=(f"stage0c_definition_f{feature_id}_mt{self._provider_max_tokens}_a{attempt}"),
            )
            try:
                with profile_region(
                    "stage0c.api.infer_definition",
                    feature_id=feature_id,
                    content_attempt=attempt,
                ):
                    response = self._client.complete(prompt, cache_key=cache_key)
                definition_text = parse_definition_response(
                    response,
                    envelope=envelope,
                )
                if recurring_peak_text is not None and recurring_peak_text not in definition_text:
                    raise AxBenchBaseError(
                        "Stage 0c API definition omitted the exact recurring "
                        f"peak text {recurring_peak_text!r}"
                    )
            except AxBenchBaseError as exc:
                last_error = exc
                record_counter(
                    "stage0c.api.definition_failures",
                    1,
                    feature_id=feature_id,
                    content_attempt=attempt,
                )
                logger.warning(
                    "Stage 0c API definition attempt {}/{} failed for feature {}: {}",
                    attempt + 1,
                    max_attempts,
                    feature_id,
                    exc,
                )
                continue
            definition = ConceptDefinition(
                feature_id=feature_id,
                text=definition_text,
                evidence_identity=evidence_identity,
                recurring_peak_text=recurring_peak_text,
            )
            self._definitions[feature_id] = definition
            record_counter("stage0c.api.definition_successes", 1)
            record_counter(
                "stage0c.api.definition_attempts_to_accept",
                attempt + 1,
                feature_id=feature_id,
                outcome="accepted",
            )
            logger.info(
                "Stage 0c API evidence-grounded definition for feature {}: {!r}",
                feature_id,
                definition_text,
            )
            return definition

        record_counter(
            "stage0c.api.definition_attempts_on_exhaustion",
            max_attempts,
            feature_id=feature_id,
            outcome="exhausted",
        )
        raise AxBenchBaseError(
            "Stage 0c API could not derive a valid evidence-grounded definition "
            f"for feature {feature_id} after {max_attempts} attempts: {last_error}"
        ) from last_error

    def generate_token_rows(
        self,
        *,
        feature_id: int,
        candidates: Sequence[PrefixCandidate],
        seed_rows: Sequence[Sequence[int]],
        cluster_target: ClusterFeatureTarget | None = None,
        proposal_sample_offset: int = 0,
    ) -> list[list[list[int]]]:
        """Generate ``[candidate][seed][token]`` rows with stable shape.

        Transport or content-protocol exhaustion for one logical seed becomes
        an empty token list. The existing local verifier scores that seed as a
        miss, preserving the N-seed consistency denominator and retrying other
        candidate prefixes normally.
        """
        if len(candidates) != len(seed_rows):
            raise AxBenchBaseError(
                "Stage 0c API candidates/seed_rows length mismatch: "
                f"{len(candidates)} != {len(seed_rows)}"
            )
        if not candidates:
            return []
        for candidate_index, seeds in enumerate(seed_rows):
            if len(seeds) != self._cfg.n_verification_seeds:
                raise AxBenchBaseError(
                    "Stage 0c API seed row length mismatch for candidate "
                    f"{candidate_index}: got {len(seeds)}, expected "
                    f"n_verification_seeds={self._cfg.n_verification_seeds}"
                )
        if type(proposal_sample_offset) is not int or proposal_sample_offset < 0:
            raise AxBenchBaseError(
                "Stage 0c API proposal_sample_offset must be a strict non-negative int"
            )
        if self._cfg.n_verification_seeds < 1:
            raise AxBenchBaseError("n_verification_seeds must be >= 1")
        if proposal_sample_offset % self._cfg.n_verification_seeds != 0:
            raise AxBenchBaseError(
                "Stage 0c API proposal_sample_offset must begin on a canonical "
                "candidate boundary"
            )
        proposal_member_ids = _proposal_member_ids(feature_id, cluster_target)
        cycle_member_descriptions = len(proposal_member_ids) > 1

        source_bridges: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            source_bridge = getattr(candidate, "source_bridge_text", None)
            if not isinstance(source_bridge, str):
                raise AxBenchBaseError(
                    "Stage 0c API candidate is missing its exact local source "
                    f"bridge at candidate index {candidate_index}"
                )
            source_bridges.append(source_bridge)

        flat_slots = [
            (candidate_index, seed_index, int(seed))
            for candidate_index, seeds in enumerate(seed_rows)
            for seed_index, seed in enumerate(seeds)
        ]
        parsed_by_slot: dict[int, list[int]] = {}
        pending = {
            slot
            for slot, (candidate_index, _seed_index, _seed) in enumerate(flat_slots)
            if source_bridges[candidate_index]
        }
        missing_bridge_candidates = sum(not bridge for bridge in source_bridges)
        if missing_bridge_candidates:
            record_counter(
                "stage0c.api.candidates_without_source_bridge",
                missing_bridge_candidates,
                feature_id=feature_id,
            )
            logger.warning(
                "Stage 0c API skipped {}/{} candidate prefix(es) for feature {} "
                "because the source document ends at the candidate boundary",
                missing_bridge_candidates,
                len(candidates),
                feature_id,
            )
        if not pending:
            flat_rows = [[] for _slot in flat_slots]
            width = self._cfg.n_verification_seeds
            return [flat_rows[start : start + width] for start in range(0, len(flat_rows), width)]

        definitions_by_member = {
            member_id: self._definition_for_feature(member_id)
            for member_id in proposal_member_ids
        }
        definition_by_slot = {
            slot: definitions_by_member[
                proposal_member_ids[
                    (proposal_sample_offset + slot) % len(proposal_member_ids)
                ]
            ]
            for slot in pending
        }
        partial_text_by_slot: dict[int, str] = {}
        retry_instruction_by_slot: dict[int, str] = {}
        max_attempts = self._cfg.max_retries + 1

        for attempt in range(max_attempts):
            requests: list[SynthRequest] = []
            parse_context: dict[int, _ContinuationParseContext] = {}
            for slot in sorted(pending):
                candidate_index, seed_index, seed = flat_slots[slot]
                candidate = candidates[candidate_index]
                definition = definition_by_slot[slot]
                continuation_system_prompt = build_continuation_system_prompt(
                    concept_definition=definition.text,
                )
                continuation_system_prompt_sha256 = _text_sha256(
                    continuation_system_prompt
                )
                partial_text = partial_text_by_slot.get(slot, "")
                request_prefix_text = candidate.prefix_text + partial_text
                required_source_bridge_text = (
                    "" if partial_text else source_bridges[candidate_index]
                )
                envelope_identity: dict[str, Any] = {
                    "feature_id": feature_id,
                    "evidence_identity": definition.evidence_identity,
                    "candidate": {
                        "shard": candidate.shard_idx,
                        "doc": candidate.doc_idx,
                        "segment": candidate.seg_idx,
                        "prefix_identity": _semantic_identity(candidate.prefix_text),
                    },
                    "request_prefix_identity": _semantic_identity(request_prefix_text),
                    "seed": seed,
                    "seed_index": seed_index,
                    "attempt": attempt,
                }
                if cycle_member_descriptions:
                    envelope_identity["description_feature_id"] = definition.feature_id
                    envelope_identity["cluster_member_description_cycle_version"] = (
                        CLUSTER_MEMBER_DESCRIPTION_CYCLE_VERSION
                    )
                envelope = _make_envelope(
                    "CONTINUATION",
                    envelope_identity,
                    untrusted_texts=[
                        request_prefix_text,
                        required_source_bridge_text,
                        definition.text,
                    ],
                )
                prompt = build_continuation_prompt(
                    prefix_text=request_prefix_text,
                    concept_definition=definition.text,
                    variation_key=seed,
                    output_length=self._cfg.verification_length,
                    envelope=envelope,
                    recurring_peak_text=definition.recurring_peak_text,
                    source_bridge_text=required_source_bridge_text,
                    retry_instruction=retry_instruction_by_slot.get(slot),
                )
                request_candidate_index = candidate_index
                if cycle_member_descriptions:
                    global_candidate_index = (
                        proposal_sample_offset // self._cfg.n_verification_seeds
                        + candidate_index
                    )
                    request_candidate_index = global_candidate_index
                    cache_tag = (
                        f"stage0c_continuation_target_f{feature_id}_"
                        f"member_f{definition.feature_id}_"
                        f"c{global_candidate_index}_s{seed_index}_"
                        f"mt{self._provider_max_tokens}_"
                        f"sp{continuation_system_prompt_sha256}_"
                        f"cv{CLUSTER_MEMBER_DESCRIPTION_CYCLE_VERSION}_a{attempt}"
                    )
                else:
                    cache_tag = (
                        f"stage0c_continuation_f{feature_id}_"
                        f"c{candidate_index}_s{seed_index}_"
                        f"mt{self._provider_max_tokens}_"
                        f"sp{continuation_system_prompt_sha256}_"
                        f"a{attempt}"
                    )
                cache_key = make_cache_key(
                    model=self._cfg.verification_api_model or "",
                    prompt=prompt,
                    tag=cache_tag,
                )
                request_attrs = {
                    "feature_id": feature_id,
                    "candidate_index": request_candidate_index,
                    "seed_index": seed_index,
                    "content_attempt": attempt,
                }
                if cycle_member_descriptions:
                    request_attrs.update(
                        {
                            "description_feature_id": definition.feature_id,
                            "proposal_sample_index": proposal_sample_offset + slot,
                            "cluster_member_description_cycle_version": (
                                CLUSTER_MEMBER_DESCRIPTION_CYCLE_VERSION
                            ),
                        }
                    )
                requests.append(
                    SynthRequest(
                        index=slot,
                        prompt=prompt,
                        cache_key=cache_key,
                        system_prompt=continuation_system_prompt,
                        attrs=request_attrs,
                    )
                )
                parse_context[slot] = _ContinuationParseContext(
                    envelope=envelope,
                    request_prefix_text=request_prefix_text,
                    original_prefix_text=candidate.prefix_text,
                    required_source_bridge_text=required_source_bridge_text,
                )

            results = self._complete_requests_fn(
                self._client,
                requests,
                max_concurrent=self._cfg.synth.max_concurrent,
                profile_region_name="stage0c.api.complete_continuation",
                batch_region_name="stage0c.api.complete_continuation_batch",
            )
            result_slots = {result.index for result in results}
            expected_slots = {request.index for request in requests}
            if result_slots != expected_slots or len(results) != len(requests):
                raise AxBenchBaseError(
                    "Stage 0c API completion helper returned an invalid result "
                    f"index set: expected={sorted(expected_slots)}, "
                    f"actual={sorted(result_slots)}"
                )

            for result in results:
                if result.error is not None:
                    retry_instruction_by_slot[result.index] = (
                        "The prior API request produced no usable response. "
                        "Follow the complete contract exactly on this fresh attempt."
                    )
                    record_counter(
                        "stage0c.api.continuation_transport_failures",
                        1,
                        feature_id=feature_id,
                        content_attempt=attempt,
                    )
                    continue
                if result.response_text is None:
                    raise AxBenchBaseError(
                        "Stage 0c API completion result has neither response "
                        f"text nor an error for slot {result.index}"
                    )
                context = parse_context[result.index]
                definition = definition_by_slot[result.index]
                prior_partial_text = partial_text_by_slot.get(result.index, "")
                candidate_index = flat_slots[result.index][0]
                trusted_source_bridge_text = source_bridges[candidate_index]
                try:
                    continuation_piece = _parse_continuation_text(
                        result.response_text,
                        envelope=context.envelope,
                        prefix_text=context.request_prefix_text,
                        concept_definition=definition.text,
                        recurring_peak_text=definition.recurring_peak_text,
                        required_source_bridge_text=(context.required_source_bridge_text),
                        # A short clean response can be extended in later
                        # requests. Individual pieces must remain artifact-
                        # free, but the exact lexical anchor is required once
                        # on the reconstructed continuation below, not once
                        # per provider chunk.
                        require_recurring_peak=False,
                    )
                    combined_continuation = prior_partial_text + continuation_piece
                    if not combined_continuation.startswith(trusted_source_bridge_text):
                        raise AxBenchBaseError(
                            "Stage 0c API continuation lost its locally trusted "
                            "source bridge while extending a short response"
                        )
                    combined_token_ids = _encode_continuation_text(
                        combined_continuation,
                        self.tokenizer,
                    )
                    continuation_is_complete = (
                        len(combined_token_ids) >= self._cfg.verification_length
                    )
                    _validate_continuation_semantics(
                        combined_continuation[len(trusted_source_bridge_text) :],
                        prefix_text=(context.original_prefix_text + trusted_source_bridge_text),
                        concept_definition=definition.text,
                        recurring_peak_text=definition.recurring_peak_text,
                        scored_window=False,
                        require_recurring_peak=continuation_is_complete,
                    )
                    if not continuation_is_complete:
                        partial_text_by_slot[result.index] = combined_continuation
                        retry_instruction_by_slot[result.index] = (
                            "The prior authenticated prose was clean but too short. "
                            "Continue directly from it with new locally coherent "
                            "prose; do not recap or restart it."
                        )
                        record_counter(
                            "stage0c.api.continuation_extensions_requested",
                            1,
                            feature_id=feature_id,
                            content_attempt=attempt,
                            retained_tokens=len(combined_token_ids),
                        )
                        continue
                    parsed = _truncate_and_validate_continuation(
                        combined_token_ids,
                        output_length=self._cfg.verification_length,
                        tokenizer=self.tokenizer,
                        prefix_text=context.original_prefix_text,
                        concept_definition=definition.text,
                        recurring_peak_text=definition.recurring_peak_text,
                        trusted_leading_text=trusted_source_bridge_text,
                    )
                except AxBenchBaseError as exc:
                    retry_instruction_by_slot[result.index] = _retry_instruction_for_error(exc)
                    record_counter(
                        "stage0c.api.continuation_parse_failures",
                        1,
                        feature_id=feature_id,
                        content_attempt=attempt,
                    )
                    logger.warning(
                        "Stage 0c API continuation protocol failure for feature "
                        "{}, slot {}, attempt {}/{}: {}",
                        feature_id,
                        result.index,
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    continue
                parsed_by_slot[result.index] = parsed
                partial_text_by_slot.pop(result.index, None)
                pending.discard(result.index)
                record_counter(
                    "stage0c.api.continuation_attempts_to_accept",
                    attempt + 1,
                    feature_id=feature_id,
                    slot=result.index,
                    outcome="accepted",
                )

            if not pending:
                break

        if pending:
            for slot in sorted(pending):
                record_counter(
                    "stage0c.api.continuation_attempts_on_exhaustion",
                    max_attempts,
                    feature_id=feature_id,
                    slot=slot,
                    outcome="exhausted",
                )
            record_counter(
                "stage0c.api.continuation_exhausted_slots",
                len(pending),
                feature_id=feature_id,
            )
            logger.warning(
                "Stage 0c API exhausted {} content attempt(s) for {}/{} "
                "continuation slot(s) on feature {}; those seeds count as "
                "verification misses",
                max_attempts,
                len(pending),
                len(flat_slots),
                feature_id,
            )

        flat_rows = [parsed_by_slot.get(slot, []) for slot in range(len(flat_slots))]
        width = self._cfg.n_verification_seeds
        return [flat_rows[start : start + width] for start in range(0, len(flat_rows), width)]


__all__ = [
    "STAGE0C_API_PROTOCOL_VERSION",
    "ConceptDefinition",
    "Stage0cApiGenerator",
    "api_generation_fingerprint_config",
    "build_continuation_prompt",
    "build_continuation_system_prompt",
    "build_definition_prompt",
    "parse_definition_response",
]
