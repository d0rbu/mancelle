"""Interpretability + Consistency judge (OpenRouter holistic call)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol

from beartype import beartype
from loguru import logger

from steering.eval.axbench_base.synth.llm_response_cache import make_cache_key
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.feature_filter.types import ActivatingExample, BinSample, PromptTooLongError
from steering.refined_types import NonNegativeIntLike

__all__ = [
    "INTERP_JUDGE_MAX_SCORE",
    "InterpJudge",
    "InterpJudgeConfig",
    "InterpJudgeRating",
    "build_judge_prompt",
    "format_band_blocks",
    "make_interp_judge",
    "marked_snippet",
    "parse_interp_rating",
]

INTERP_JUDGE_MAX_SCORE: int = 5

PROMPTS_DIR = Path(__file__).parent / "prompts"
_TEMPLATE_NAME = "judge_feature_interpretability.txt"


@dataclass(frozen=True)
class InterpJudgeRating:
    interpretability: int
    consistency: int


# Invariants:
# - label must be followed by colon on the SAME line (no newlines between label,
#   colon, and value) — judge emits one "<label>: <value>" pair per line.
# - Value is either ``[[N]]`` (bracketed) or a bare integer/decimal not adjacent
#   to a ``]`` (so we don't greedily grab a partially-bracketed score).
# - Trailing ``(?!\w)`` on the bare alt prevents matches like ``[[3 extra]]``
#   sliding through via the bare branch.
_FIELD_RE = re.compile(
    r"\b(?P<label>interpretability|consistency)[ \t]*:[ \t]*"
    r"(?:\[\[\s*(?P<bracketed>-?\d+(?:\.\d+)?)\s*\]\]|(?<!\[)(?P<bare>-?\d+(?:\.\d+)?)(?!\w)(?!\s*\]))",
    re.IGNORECASE,
)


def _last_int_for(text: str, label: str) -> int | None:
    last: int | None = None
    for m in _FIELD_RE.finditer(text):
        if m.group("label").lower() != label:
            continue
        raw = m.group("bracketed") if m.group("bracketed") is not None else m.group("bare")
        if "." in raw:
            last = None
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 <= value <= INTERP_JUDGE_MAX_SCORE:
            last = value
        else:
            last = None
    return last


def parse_interp_rating(text: str) -> InterpJudgeRating | None:
    interp = _last_int_for(text, "interpretability")
    cons = _last_int_for(text, "consistency")
    if interp is None or cons is None:
        return None
    return InterpJudgeRating(interp, cons)


def marked_snippet(example: ActivatingExample, window_chars: int) -> str:
    full = example.example_text(window_chars)
    peak = example.peak_char_offset
    if window_chars > 0 and len(example.text) > 2 * window_chars:
        start = max(0, peak - window_chars)
        peak_in_window = peak - start
        prefix_len = 3 if start > 0 else 0
        pos = prefix_len + peak_in_window
    else:
        pos = peak
    if pos < 0 or pos >= len(full):
        return full
    token_text = example.tokens[example.max_value_token_index] if example.tokens else ""
    if token_text and full[pos:pos + len(token_text)] == token_text:
        end = pos + len(token_text)
    else:
        end = min(pos + 1, len(full))
    return f"{full[:pos]}<<{full[pos:end]}>>{full[end:]}"


def format_band_blocks(samples: list[BinSample], *, window_chars: int) -> str:
    """Group firing samples by band_label and render one block per band.

    Absent samples (`is_absent=True`) are pulled out of the band ordering
    and rendered in a single trailing "Negative baselines" block.
    """
    if not samples:
        return ""
    non_absent = [s for s in samples if not s.is_absent]
    absent = [s for s in samples if s.is_absent]
    if non_absent:
        indices = [s.bin_index for s in non_absent]
        if indices != sorted(indices, reverse=True):
            raise ValueError(
                f"format_band_blocks expects peak-first ordered samples; "
                f"got bin_index sequence {indices}"
            )
    blocks: list[str] = []
    i = 0
    while i < len(non_absent):
        j = i
        while (j < len(non_absent)
               and non_absent[j].band_label == non_absent[i].band_label):
            j += 1
        group = non_absent[i:j]
        header = f"{non_absent[i].band_label} — {len(group)} examples:"
        lines = [header]
        for n, s in enumerate(group, start=1):
            body = marked_snippet(s.example, window_chars)
            lines.append(f"  [{n}] max_value={s.max_value:.2f}: {body}")
        blocks.append("\n".join(lines))
        i = j
    if absent:
        header = (
            f"Negative baselines (feature did not fire) — "
            f"{len(absent)} examples:"
        )
        lines = [header]
        for n, s in enumerate(absent, start=1):
            body = s.example.example_text(window_chars)
            lines.append(f"  [{n}] max_value=0 (absent): {body}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@cache
def _load_template() -> str:
    path = PROMPTS_DIR / _TEMPLATE_NAME
    return path.read_text(encoding="utf-8")


def build_judge_prompt(description: str, samples: list[BinSample], *, window_chars: int) -> str:
    template = _load_template()
    blocks = format_band_blocks(samples, window_chars=window_chars)
    return template.replace("{DESCRIPTION}", description).replace("{BAND_BLOCKS}", blocks)


@beartype
@dataclass(frozen=True)
class InterpJudgeConfig:
    max_prompt_chars: int = 32_000
    max_examples_in_prompt: int = 25
    example_window_chars: int = 240
    parse_max_retries: NonNegativeIntLike = 3


class InterpJudge(Protocol):
    def __call__(
        self,
        description: str,
        samples: list[BinSample],
        /,
    ) -> InterpJudgeRating | None: ...


def make_interp_judge(llm_client: Any, config: InterpJudgeConfig) -> InterpJudge:
    """Build an interp judge from an LLM client + config."""
    def _judge(description: str, samples: list[BinSample]) -> InterpJudgeRating | None:
        limited = samples[: config.max_examples_in_prompt]
        prompt = build_judge_prompt(description, limited, window_chars=config.example_window_chars)
        if len(prompt) > config.max_prompt_chars:
            raise PromptTooLongError(
                f"Prompt {len(prompt)} chars > max_prompt_chars={config.max_prompt_chars}"
            )
        messages = [{"role": "user", "content": prompt}]
        base_tag = f"interp_{description[:40]}"
        llm_model = getattr(getattr(llm_client, "_cfg", None), "model", "interp_judge")
        max_total = config.parse_max_retries + 1
        for attempt in range(max_total):
            tag = base_tag if attempt == 0 else f"{base_tag}_a{attempt}"
            cache_key = make_cache_key(model=llm_model, prompt=prompt, tag=tag)
            try:
                content = llm_client.complete(messages, cache_key=cache_key)
            except AxBenchBaseError as exc:
                logger.warning(f"LLM transport failure during interp judge: {exc}")
                return None
            rating = parse_interp_rating(content)
            if rating is not None:
                return rating
            logger.warning(
                f"Could not parse interp/consistency (attempt {attempt + 1}/{max_total}); "
                f"response[:200]={content[:200]!r}"
            )
        return None

    return _judge
