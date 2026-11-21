"""OpenRouter pathologies: empty 200, non-JSON, missing choices, malformed UTF-8, long rant with rating at end."""
from steering.eval.feature_filter.interp_judge import parse_interp_rating


def test_empty_response() -> None:
    assert parse_interp_rating("") is None


def test_non_json_plain_text() -> None:
    assert parse_interp_rating("this is not a rating at all") is None


def test_missing_consistency_field() -> None:
    assert parse_interp_rating("Interpretability: [[3]]") is None


def test_long_rant_with_valid_rating_at_end() -> None:
    rant = (
        "Let me think about this carefully. The feature activates on tokens "
        "related to weather patterns. " * 40
        + "\nInterpretability: [[4]]\nConsistency: [[3]]"
    )
    r = parse_interp_rating(rant)
    assert r is not None
    assert r.interpretability == 4 and r.consistency == 3


def test_malformed_utf8_replacement_char() -> None:
    # Simulate a response that included replacement chars from bad UTF-8 decoding.
    response = "\ufffdInterpretability: [[1]]\nConsistency: [[0]]\ufffd"
    r = parse_interp_rating(response)
    assert r is not None
    assert r.interpretability == 1 and r.consistency == 0
