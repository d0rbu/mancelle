"""Property-level parity for the shared text primitives introduced by #1051."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.prefix_text import window_prefix_text
from steering.eval.axbench_base.synth import continuation_guards
from tests.axbench_base.unit.issue_1051_reference import (
    legacy_repeated_phrase,
    legacy_source_phrase_echo,
    legacy_window_prefix_text,
)


@st.composite
def _text_windows(draw) -> tuple[str, int, int]:
    text = draw(st.text(max_size=512))
    end_char = draw(st.integers(min_value=0, max_value=len(text)))
    max_prefix_chars = draw(st.integers(min_value=0, max_value=640))
    return text, end_char, max_prefix_chars


@given(case=_text_windows())
@settings(max_examples=500, deadline=None)
def test_shared_prefix_window_is_exact_pre_refactor_stage0c_function(
    case: tuple[str, int, int],
) -> None:
    text, end_char, max_prefix_chars = case
    expected = legacy_window_prefix_text(text, end_char, max_prefix_chars)

    assert window_prefix_text(text, end_char, max_prefix_chars) == expected


@given(text=st.text(max_size=240))
@settings(max_examples=500, deadline=None)
def test_shared_repeated_phrase_is_exact_pre_refactor_stage0c_function(
    text: str,
) -> None:
    expected = legacy_repeated_phrase(text)

    assert continuation_guards.repeated_phrase(text) == expected


@given(
    source_text=st.text(max_size=180),
    continuation=st.text(max_size=180),
)
@settings(max_examples=500, deadline=None)
def test_shared_source_echo_is_exact_pre_refactor_stage0c_function(
    source_text: str,
    continuation: str,
) -> None:
    expected = legacy_source_phrase_echo(source_text, continuation)

    assert (
        continuation_guards.source_phrase_echo(source_text, continuation)
        == expected
    )
