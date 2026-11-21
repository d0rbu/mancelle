"""Authoritative judge-response parser for AxBench 0/1/2 ratings.

Single source of truth for both Stage 0c (``concept_examples_qc``) and
Stage 5 (``judge_all``). Any new judge-parsing site must import
:func:`parse_axbench_rating` from this module rather than re-inventing
a regex -- two subtly different regexes in two places is how the
original ``Rating: [[N]]``-only bug shipped.

Design
------

The parser accepts the **empirically-observed** set of formats the
judge actually emits (see
``tests/axbench_base/fixtures/judge_response_samples.jsonl``), not
the single format the prompt template asks for. Observed in a
30-response sample against ``openai/gpt-5-nano``:

* ``Rating: [[N]]`` -- canonical (25/30, ~83%)
* ``Rating: N``     -- bare, no brackets (5/30, ~17%)

Plus format-drift tolerances for shapes the live corpus didn't
capture but an LLM can emit at any time:

* ``Rating:[[N]]`` / ``Rating:N``    (no space)
* ``Rating:  [[N]]`` / ``Rating:  N`` (multiple spaces)
* ``Rating:\\tN``                    (tab after colon)
* ``Rating: [[ N ]]``                (spaces inside brackets)
* ``Rating: [N]``                    (single brackets -- mismatched allowed)
* ``Rating: **N**`` / ``Rating: **[[N]]**`` (markdown bold)
* ``Rating: N.``                     (trailing punctuation)
* ``Rating: N`` inside a code block

Rejection contract (returns ``None``):

* empty string, ``None``, ``Rating:`` without a number
* lowercase ``rating:`` (case-sensitive -- a lowercase emission is a
  strong signal the judge went off-script)
* non-rating keywords (``Score:``, ``Grade:``, ...)
* words instead of digits (``Rating: one``)
* decimals (``Rating: 1.5``) -- the rubric is integer-valued
* negative values (``Rating: -1``)
* out-of-range values (``Rating: 3``, etc.)
* responses with no ``Rating:`` keyword at all

Multiple-rating policy
----------------------

When a response contains multiple ``Rating: N`` occurrences (a real
drift risk -- "initially I thought Rating: [[1]] but on reflection
Rating: [[2]]"), the **last** match is authoritative. The prompt
template asks for a concluding rating, so the final mention is the
judge's actual answer.

``None`` return semantics
-------------------------

``None`` means "the response does not contain a well-formed rating".
It is **NOT** the same as "the judge returned 0". Callers MUST
distinguish these two cases -- the historical
``concept_examples_qc._parse_rating`` conflating them is what silently
rejected 16,379 features in a 9.6-hour production run. The standard
pattern is::

    score = parse_axbench_rating(response)
    if score is None:
        # retry, mark invalid, log, etc. -- but DO NOT treat as score=0
        ...
"""

from __future__ import annotations

import re

# Matches:
#
#   Rating: <any mix of whitespace/tabs, asterisks (markdown bold),
#           or opening brackets> (-?\d+) (?!\.\d) (?!\w)
#                                          ^        ^
#                                          |        reject if followed by
#                                          |        a word char: blocks
#                                          |        digit-greedy backtracking
#                                          |        AND ``0x10`` / ``1e2`` /
#                                          |        ``1abc`` non-rubric junk
#                                          reject decimals like "1.5"
#
# The character class ``[\s*\[]*`` after ``Rating:`` absorbs any prefix
# noise (whitespace, ``*`` characters, opening ``[`` brackets) so we
# catch markdown-bold / canonical / bare / single-bracket variants
# with one pattern.
#
# Closing brackets / asterisks are NOT required -- we anchor on the
# digit sequence and let the range check downstream reject nonsense.
# This is deliberate: LLMs frequently drop a closing bracket, and
# silently failing the whole parse because of one missing ``]`` is
# exactly the bug that motivated this rewrite.
#
# **Both negative lookaheads** are required.
#
# Without ``(?!\.\d)``, ``Rating: 1.5`` would silently parse as ``1``.
# Without ``(?!\w)``, two failure classes slip through:
#
# 1. **Decimal backtracking** (verified bug found in code review):
#    ``\d+`` is greedy and backtracks. For ``Rating: 12.5``:
#      - Try ``\d+`` matching ``12``. ``(?!\.\d)`` fails on ``.5``.
#        Backtrack.
#      - Try ``\d+`` matching ``1``. ``(?!\.\d)`` checks next char
#        ``2`` (not ``.``), succeeds. Capture ``1`` -- silently WRONG.
#    With ``(?!\w)``, the backtrack to ``1`` is rejected because
#    the next char ``2`` is a word char.
#
# 2. **Non-decimal numeric literals**: ``Rating: 0x10`` greedy-matches
#    ``0``; ``(?!\.\d)`` accepts (``x`` isn't ``.``); the parser would
#    return ``0`` for hex input. ``Rating: 1e2`` likewise returns ``1``.
#    ``Rating: 1abc`` likewise returns ``1``. With ``(?!\w)``, all of
#    these are rejected because the trailing alphanumeric belongs to
#    a different token.
#
# Acceptable suffixes (not word chars): closing brackets ``]``, period
# ``.`` followed by non-digit / EOF, comma ``,``, end-of-string,
# whitespace.
_RATING_PATTERN = re.compile(r"Rating:[\s*\[]*(-?\d+)(?!\.\d)(?!\w)")


# Maximum valid judge rating on the AxBench 0/1/2 scale.
#
# The AxBench concept-presence and fluency judges both return integers
# in ``{0, 1, 2}``. This named constant replaces the previously
# hardcoded literal ``2`` in:
#
# - :func:`parse_axbench_rating` -- the rating-range validity check.
# - :func:`~steering.eval.axbench_base.aggregation._compute_row_score` --
#   the negative-mode axis flip ``hmean(JUDGE_MAX_SCORE - concept,
#   fluency)`` (high concept presence should map to a low score when
#   the method is supposed to SUPPRESS a concept).
#
# Keeping the constant in :mod:`judge_parser` instead of
# :mod:`aggregation` pins it to the module that owns the input-validity
# contract; any consumer that flips the axis merely imports it.
JUDGE_MAX_SCORE: int = 2


def parse_axbench_rating(response: str | None) -> int | None:
    """Extract the rating score from a judge response.

    Accepts the empirical format variants documented in the module
    docstring. Returns the integer rating on success or ``None`` on
    any failure (bad format, out of range, missing).

    When multiple ``Rating: N`` occurrences are present, the last
    match wins -- the prompt template asks the judge to conclude with
    a rating, so later mentions are the authoritative answer.

    Args:
        response: the judge's full raw response content, or ``None``.

    Returns:
        The rating as an integer in ``[0, JUDGE_MAX_SCORE]`` inclusive,
        or ``None`` if no valid rating could be extracted.
    """
    if response is None:
        return None
    matches = _RATING_PATTERN.findall(response)
    if not matches:
        return None
    try:
        score = int(matches[-1])
    except ValueError:
        # ``_RATING_PATTERN`` only captures ``(-?\d+)`` so ValueError
        # is unreachable in practice; guard defensively against a
        # future regex edit that broadens the capture group.
        return None
    if not (0 <= score <= JUDGE_MAX_SCORE):
        return None
    return score
