"""Boundary and diagnostic tests for shared synth continuation guards."""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.synth.continuation_guards import (
    repeated_phrase,
    source_phrase_echo,
    validate_no_source_echo_or_repeated_phrase,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def test_repeated_phrase_rejects_exact_eight_word_48_char_boundary() -> None:
    phrase = "aaaaaaa bbbbbb ccccc ddddd eeeee fffff gggg hhhh"
    assert len(phrase.split()) == 8
    assert len(phrase) == 48

    assert repeated_phrase(f"{phrase}. {phrase}!") == phrase


def test_repeated_phrase_allows_exact_eight_word_47_char_boundary() -> None:
    phrase = "aaaaaa bbbbbb ccccc ddddd eeeee fffff gggg hhhh"
    assert len(phrase.split()) == 8
    assert len(phrase) == 47

    assert repeated_phrase(f"{phrase} {phrase}") is None


def test_repeated_phrase_allows_seven_long_words() -> None:
    phrase = "enormousword " * 7

    assert repeated_phrase(f"{phrase} {phrase}") is None


def test_repeated_phrase_requires_non_overlapping_occurrences() -> None:
    # The qualifying eight-word phrase occurs at starts 0 and 1, but those
    # occurrences overlap. The remaining seven unique words make the overall
    # text long enough for width=8 to be examined without introducing a third,
    # non-overlapping copy.
    repeated = ["longtoken"] * 9
    unique_tail = [f"unique{index}" for index in range(7)]
    text = " ".join(repeated + unique_tail)

    assert repeated_phrase(text) is None


def test_repeated_phrase_normalizes_case_punctuation_and_whitespace() -> None:
    first = "Alpha bravo charlie delta echo foxtrot golf hotel"
    second = "ALPHA, BRAVO; CHARLIE\nDELTA ECHO FOXTROT GOLF HOTEL"

    assert repeated_phrase(f"{first}. Independent bridge. {second}") == first.casefold()


def test_source_echo_rejects_seven_words_at_32_char_boundary() -> None:
    phrase = "aaaa bbbb cccc dddd eeee fff ggg"
    assert len(phrase.split()) == 7
    assert len(phrase) == 32
    assert len("".join(phrase.split())) < 32
    source = f"Source opening. {phrase}. Source ending."
    continuation = f"New material begins, then {phrase} appears."

    assert source_phrase_echo(source, continuation) == phrase


def test_source_echo_allows_six_short_words_below_alphanumeric_fallback() -> None:
    phrase = "small teams check each result now"
    assert len("".join(phrase.split())) < 32
    source = f"Source opening. {phrase}. Source ending."
    continuation = f"New material begins, then {phrase} appears."

    assert source_phrase_echo(source, continuation) is None


def test_source_echo_catches_six_long_words_via_alphanumeric_fallback() -> None:
    phrase = "alphaword bravoword charlieword deltaword echoword foxtrotword"
    source = f"Source opening. {phrase}. Source ending."
    continuation = f"New material begins, then {phrase} appears."

    assert source_phrase_echo(source, continuation) is not None


def test_source_echo_allows_seven_words_below_char_boundary() -> None:
    phrase = "aaaa bbbb cccc dddd eee fff ggg"
    assert len(phrase.split()) == 7
    assert len(phrase) == 31
    assert len("".join(phrase.split())) < 32
    source = f"Source opening. {phrase}. Source ending."
    continuation = f"New material begins, then {phrase} appears."

    assert source_phrase_echo(source, continuation) is None


def test_source_echo_finds_copy_after_clean_opening() -> None:
    phrase = "proposals are being prepared for one plant name"
    source = f"The record says {phrase} before the local boundary."
    continuation = (
        "A clean independent paragraph appears first. "
        f"Later, {phrase}, and the analysis resumes."
    )

    assert source_phrase_echo(source, continuation) == phrase


def test_source_echo_normalizes_case_punctuation_and_whitespace() -> None:
    source = "Sediment moves depending on the direction of incoming waves."
    continuation = (
        "New material. DEPENDING,\non the direction; of incoming waves!"
    )

    assert (
        source_phrase_echo(source, continuation)
        == "depending on the direction of incoming waves"
    )


def test_source_echo_catches_nfkc_alphanumeric_copy_without_word_boundaries() -> None:
    copied = "研究人员正在重建跨物种祖先关系图谱并检查每一条分支以确认共同来源"
    assert len(copied) >= 32
    source = f"本地来源：{copied}。"
    continuation = f"新的材料开始。随后复制：{copied}。"

    assert source_phrase_echo(source, continuation) is not None


def test_source_echo_nfkc_normalizes_compatibility_forms() -> None:
    source = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ１２３４５６"
    continuation = "abcdefghijklmnopqrstuvwxyz123456"

    assert source_phrase_echo(source, continuation) == continuation


def test_source_echo_alphanumeric_fallback_ignores_punctuation() -> None:
    payload = "abcdefghijklmnopqrstuvwxzy123456"
    source = "-".join(payload)
    continuation = f"Before copy: {payload}"

    assert source_phrase_echo(source, continuation) == payload[:32]


def test_validator_reports_repeated_phrase_before_source_echo() -> None:
    phrase = "alpha bravo charlie delta echo foxtrot golf hotel"
    source = f"Source contains {phrase}."
    continuation = f"{phrase}. {phrase}."

    with pytest.raises(
        AxBenchBaseError,
        match=r"^custom subject contains a repeated phrase loop:",
    ) as exc_info:
        validate_no_source_echo_or_repeated_phrase(
            continuation,
            source_text=source,
            subject="custom subject",
        )

    assert "copied a substantial source phrase" not in str(exc_info.value)


def test_validator_reports_source_echo_with_exact_subject() -> None:
    phrase = "alpha bravo charlie delta echo foxtrot golf"
    source = f"Source contains {phrase}."

    with pytest.raises(
        AxBenchBaseError,
        match=r"^Stage 1b concept_rich continuation copied a substantial source phrase:",
    ):
        validate_no_source_echo_or_repeated_phrase(
            f"Independent opening, then {phrase}.",
            source_text=source,
            subject="Stage 1b concept_rich continuation",
        )


def test_validator_allows_clean_continuation() -> None:
    validate_no_source_echo_or_repeated_phrase(
        "A completely independent continuation with no substantial copied sequence.",
        source_text="The source discusses a different subject in unrelated language.",
        subject="clean continuation",
    )
