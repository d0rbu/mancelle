"""Unit tests for ``tasks.common.render_mc``.

Per Phase 0 map §7.2, ``render_mc`` lacked direct coverage. This pins
its formatting contract for the MC eval prompts.
"""

from __future__ import annotations

from tasks.common import render_mc


def test_basic_4_choice():
    out = render_mc(
        question="What is 2+2?",
        letters=["A", "B", "C", "D"],
        choices=["3", "4", "5", "6"],
    )
    expected = (
        "Multiple Choice question: What is 2+2?\n"
        "- 3=A\n- 4=B\n- 5=C\n- 6=D\n\n"
        "Respond only with the letter of the correct answer."
    )
    assert out == expected


def test_no_space_between_choice_equals_letter():
    """Tokenizer parity: NO whitespace before the answer letter (intentional)."""
    out = render_mc(
        question="Q",
        letters=["A"],
        choices=["x"],
    )
    assert "x=A" in out
    assert "x= A" not in out


def test_5_choice():
    out = render_mc(
        question="Q?",
        letters=["A", "B", "C", "D", "E"],
        choices=["1", "2", "3", "4", "5"],
    )
    for c, letter in zip(
        ["1", "2", "3", "4", "5"], ["A", "B", "C", "D", "E"], strict=True
    ):
        assert f"- {c}={letter}\n" in out


def test_single_choice():
    out = render_mc(question="Q", letters=["A"], choices=["only one"])
    assert out.count("=A") == 1
    assert "only one=A" in out


def test_empty_choices_yields_empty_choice_block():
    out = render_mc(question="Q", letters=[], choices=[])
    # No choices block; question + closing instruction only.
    assert "Multiple Choice question: Q\n" in out
    assert "\nRespond only with the letter of the correct answer." in out
    # Crucially, no orphan "=" line.
    assert "=" not in out.split("\n\nRespond")[0].rstrip("\n").split("\n")[-1]


def test_zip_truncates_at_shorter_iterable():
    """zip stops at the shorter iterable: extra letters are silently dropped."""
    out = render_mc(question="Q", letters=["A", "B", "C"], choices=["x", "y"])
    assert "- x=A\n" in out
    assert "- y=B\n" in out
    assert "=C" not in out
