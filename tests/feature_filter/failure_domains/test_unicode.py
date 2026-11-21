"""Unicode edge cases: CJK, RTL, combining, NFC/NFD, BOM, surrogates."""
import unicodedata

from steering.eval.feature_filter.interp_judge import (
    build_judge_prompt,
    parse_interp_rating,
)
from steering.eval.feature_filter.types import ActivatingExample, BinSample


def _mk(text: str, peak: int = 0) -> BinSample:
    ex = ActivatingExample(
        tokens=list(text), text=text, max_value=1.0,
        max_value_token_index=peak, peak_char_offset=peak,
    )
    return BinSample(
        bin_index=4,
        band_label="band 5/5 (peak firing)",
        is_absent=False,
        max_value=1.0,
        example=ex,
    )


def test_cjk_description_and_snippet() -> None:
    prompt = build_judge_prompt("中文概念", [_mk("这是中文测试")], window_chars=0)
    stripped = prompt.replace("<<", "").replace(">>", "")
    assert "中文概念" in stripped
    assert "这是中文测试" in stripped


def test_rtl_arabic_hebrew() -> None:
    prompt = build_judge_prompt("Hebrew: שלום", [_mk("Arabic: السلام")], window_chars=0)
    assert "שלום" in prompt
    assert "السلام" in prompt


def test_combining_marks_zalgo() -> None:
    zalgo = "H̸̢̧̧̪̪̘̘͍̘͍͎͎e̸̡̢̡l͍̘̘͚l̢̡̪o"
    prompt = build_judge_prompt("desc", [_mk(zalgo)], window_chars=0)
    stripped = prompt.replace("<<", "").replace(">>", "")
    assert zalgo in stripped


def test_emoji_with_zwj() -> None:
    emoji = "👨‍👩‍👧‍👦"  # family emoji with ZWJ
    prompt = build_judge_prompt("family", [_mk(emoji)], window_chars=0)
    stripped = prompt.replace("<<", "").replace(">>", "")
    assert emoji in stripped


def test_nfc_nfd_both_accepted() -> None:
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    build_judge_prompt(nfc, [_mk(nfd)], window_chars=0)  # should not raise


def test_bom_prefixed_response_parsed() -> None:
    response = "\ufeffInterpretability: [[2]]\nConsistency: [[3]]"
    r = parse_interp_rating(response)
    assert r is not None
    assert r.interpretability == 2


def test_surrogate_pair_description() -> None:
    # Astral-plane character (mathematical bold A)
    text = "\U0001d400"
    prompt = build_judge_prompt(text, [_mk(text)], window_chars=0)
    assert text in prompt
