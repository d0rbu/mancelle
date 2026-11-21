"""Unit tests for tests.conftest.get_tokenizer_or_skip."""

import pytest

from tests.conftest import get_tokenizer_or_skip


def test_returns_tokenizer_on_success(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("tests.conftest.get_tokenizer", lambda: sentinel)
    assert get_tokenizer_or_skip() is sentinel


def test_skips_on_file_not_found(monkeypatch):
    def _raise():
        raise FileNotFoundError("no tokenizer.pkl")

    monkeypatch.setattr("tests.conftest.get_tokenizer", _raise)
    with pytest.raises(pytest.skip.Exception) as ei:
        get_tokenizer_or_skip()
    msg = str(ei.value)
    assert "tokenizer cache unavailable" in msg
    assert "tok_train" in msg


def test_skips_on_value_error(monkeypatch):
    def _raise():
        raise ValueError("corrupt tokenizer dir")

    monkeypatch.setattr("tests.conftest.get_tokenizer", _raise)
    with pytest.raises(pytest.skip.Exception):
        get_tokenizer_or_skip()


def test_does_not_swallow_other_errors(monkeypatch):
    def _raise():
        raise RuntimeError("genuine tokenizer bug")

    monkeypatch.setattr("tests.conftest.get_tokenizer", _raise)
    with pytest.raises(RuntimeError, match="genuine tokenizer bug"):
        get_tokenizer_or_skip()
