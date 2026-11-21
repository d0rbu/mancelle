"""CORE target_real: a zero-character target span skips the example, loudly counted.

A benchmark row whose continuation decodes to zero characters produces an empty
target span. ``filter_segments_to_target`` rejects that (correctly — there are no
segments to pool), but aborting a 22-task eval over one degenerate row is the
wrong trade. The example is skipped and counted; every OTHER rejection stays loud.
"""

from __future__ import annotations

import pytest

from steering.eval.target_conditioning import ContextFeatureSegment
from steering.eval.target_real_runner import (
    core_row_char_spans,
    should_skip_empty_target_core_example,
)
from steering.eval.target_segments import filter_segments_to_target


class _CharTokenizer:
    """One character per token id; decode is just the slice joined."""

    def __init__(self, text: str) -> None:
        self._chars = list(text)

    def decode(self, token_ids) -> str:
        return "".join(self._chars[i] for i in token_ids)


def _row(n_tokens: int):
    return list(range(n_tokens))


def test_normal_row_has_a_positive_span_and_is_not_skipped():
    tok = _CharTokenizer("abcdefgh")
    spans = core_row_char_spans(tok, [_row(8)], [5], [8])
    assert spans == [(5, 8)]
    assert not should_skip_empty_target_core_example(spans)


def test_zero_character_continuation_is_detected_as_empty():
    # Tokens 5..8 all decode to "" -> prefix and full text are the same length.
    tok = _CharTokenizer("abcde")  # only 5 characters exist
    tok._chars = list("abcde") + ["", "", ""]  # 3 characterless tokens
    spans = core_row_char_spans(tok, [_row(8)], [5], [8])
    assert spans == [(5, 5)], spans
    assert should_skip_empty_target_core_example(spans)


def test_whole_example_is_skipped_when_any_row_is_empty():
    # Multiple-choice renders several rows. Scoring a subset would change the
    # answer set, so ONE empty row must condemn the whole example.
    tok = _CharTokenizer("abcde")
    tok._chars = list("abcde") + ["", "", ""]
    good = core_row_char_spans(tok, [_row(8)], [2], [5])
    bad = core_row_char_spans(tok, [_row(8)], [5], [8])
    assert not should_skip_empty_target_core_example(good)
    assert should_skip_empty_target_core_example(good + bad)


def test_reversed_span_also_counts_as_empty():
    assert should_skip_empty_target_core_example([(9, 4)])


def test_filter_segments_to_target_still_raises_on_an_empty_span():
    # The invariant itself is UNCHANGED: the guard lives at the CORE caller, so
    # any other consumer that hands filter_segments_to_target an empty range
    # still fails loud rather than silently pooling nothing.
    seg = ContextFeatureSegment(feature_ids=(1,), buckets=(1,), char_start=0, char_end=10)
    with pytest.raises(ValueError, match="target range must be positive"):
        filter_segments_to_target([seg], target_char_start=2010, target_char_end=2010)
