"""Prompt template loading and literal-replace substitution.

Templates live under ``steering/eval/axbench_base/prompts/`` as .txt files.
They use double-brace uppercase placeholders (e.g. ``{{PREFIX_TEXT}}``) so
that content containing literal ``{`` / ``}`` characters is safe. The
substitution helper does a single-pass regex replace, not iterative
``str.replace``, so a substituted value that happens to contain a
``{{OTHER_KEY}}`` string cannot become a placeholder for a subsequent
pass. Iterative replacement would be an injection vector when the
substituted-in value is untrusted text (e.g. Stage 1 synth passes
wild validation-shard ``seed.prefix_text`` into ``{{PREFIX_TEXT}}``).
"""

from __future__ import annotations

import functools
import re

from steering.eval.axbench_base.paths import PROMPT_DIR
from steering.eval.axbench_base.types import AxBenchBaseError

# ``{{UPPER_SNAKE_DIGITS}}`` pattern — mirrors the canonical key shape
# used by the prompt templates under ``prompts/``.
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")


def _safe_substitute(template: str, **kwargs) -> str:
    """Replace ``{{UPPER_KEY}}`` placeholders with kwarg values in one pass.

    Single-pass regex substitution: each ``{{KEY}}`` in the template is
    replaced with the matching kwarg value exactly once. A substituted
    value that contains a literal ``{{OTHER_KEY}}`` is NOT re-substituted
    — it passes through as data.

    Unknown placeholders (keys in the template that the caller didn't
    supply) pass through unchanged. Callers that need to guard against
    template drift should call :func:`assert_no_unfilled_placeholders`
    on the result, or otherwise verify the prompt is complete.
    """
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        # Leave unknown placeholders in place so caller's unfilled-
        # placeholder guard can detect them.
        return match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


def assert_no_unfilled_placeholders(prompt: str, *, label: str) -> None:
    """Raise :class:`AxBenchBaseError` if any ``{{UPPER_KEY}}`` placeholder
    remains in the prompt after substitution.

    Templates use ``{{UPPER_SNAKE_DIGITS}}`` placeholder shape; an
    unfilled one indicates the caller forgot a kwarg in
    :func:`_safe_substitute`. Sending such a prompt to OpenRouter wastes
    real money on a garbage continuation, so we refuse to ship one.

    Uses an explicit ``raise`` (NOT a bare ``assert``) so the guard
    survives ``python -O``. Under ``-O``, bare ``assert`` statements
    are stripped from bytecode, which would let an unfilled
    ``{{CONCEPT}}`` template variable reach the OpenRouter API
    silently.

    Match only the strict placeholder pattern ``{{UPPER_SNAKE_DIGITS}}``
    rather than any ``{{``: a substituted-in generation can legitimately
    contain literal ``{{`` (we've seen garbled SAE-steered output like
    ``{{ Vin... [Sentence Fragment End``), and those are not template
    drift — they are valid generation text.
    """
    match = _PLACEHOLDER_RE.search(prompt)
    if match is not None:
        raise AxBenchBaseError(
            f"{label} prompt has unfilled placeholders: "
            f"{prompt[match.start(): match.start() + 60]!r}"
        )


def render_template_with_untrusted_values(
    name: str,
    *,
    label: str,
    **values: object,
) -> str:
    """Render a template whose values may contain placeholder-shaped text.

    Template completeness is checked against trusted sentinels before any
    untrusted value is inserted. Checking the final prompt would mistake a
    literal ``{{UPPER_KEY}}`` in source text or activation evidence for an
    unfilled template variable.
    """
    return render_untrusted_template(
        load_template(name),
        label=label,
        **values,
    )


def render_untrusted_template(
    template: str,
    *,
    label: str,
    **values: object,
) -> str:
    """Render already-loaded template text with the same injection guard."""
    guard_values = {
        key: f"AXBENCH_FILLED_{key}"
        for key in values
    }
    guard_prompt = _safe_substitute(template, **guard_values)
    assert_no_unfilled_placeholders(guard_prompt, label=label)
    return _safe_substitute(template, **values)


@functools.lru_cache(maxsize=16)
def load_template(name: str) -> str:
    """Load a template file by name from ``PROMPT_DIR``, cached with LRU.

    Raises ``AxBenchBaseError`` if the file does not exist, for consistency
    with the rest of the package's error taxonomy. The LRU cache does NOT
    cache this exception — subsequent successful calls after an earlier
    failure still reach disk.
    """
    path = PROMPT_DIR / name
    if not path.is_file():
        raise AxBenchBaseError(
            f"template not found: {name} (looked in {PROMPT_DIR})"
        )
    return path.read_text(encoding="utf-8")
