"""Error-handling tests for ``InferenceEngine._build_chat_prompt``.

Map SB-13-P1-08:

    The catch tuple was ``(ValueError, RuntimeError, TypeError, AttributeError)``.
    A tokenizer that raises ``KeyError`` on a missing special would
    propagate as ``KeyError`` rather than the documented ``ValueError``.
    The fix adds ``KeyError`` to the tuple.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.eval.inference_engine import InferenceEngine
from tests.conftest import MockTokenizer, _make_gpt

_DEVICE = t.device("cpu")


class _RaisingTokenizer(MockTokenizer):
    """A tokenizer where ``encode_special('<assistant_end>')`` raises an
    arbitrary exception type."""

    def __init__(self, exc_type: type[Exception]):
        super().__init__()
        self._raise_type = exc_type

    def encode_special(self, name: str) -> int:
        if name == "<assistant_end>":
            raise self._raise_type(f"missing {name}")
        return super().encode_special(name)


@pytest.mark.parametrize(
    "exc_type",
    [ValueError, RuntimeError, TypeError, AttributeError, KeyError],
)
def test_build_chat_prompt_raises_value_error_for_each_exception_type(exc_type):
    """All five caught exception types are mapped to a ``ValueError`` with
    the documented message. Previously ``KeyError`` would propagate; the
    fix added it to the tuple."""
    model = _make_gpt(device="cpu")
    tok = _RaisingTokenizer(exc_type)
    engine = InferenceEngine(model, tok, _DEVICE)
    with pytest.raises(ValueError, match="encode_special"):
        engine._build_chat_prompt("hello")


def test_build_chat_prompt_normal_run_succeeds():
    """Sanity: a tokenizer that returns a valid int does not raise."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    engine = InferenceEngine(model, tok, _DEVICE)
    ids, stops = engine._build_chat_prompt("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert isinstance(stops, set)
    # Two stop tokens: bos (=0) and assistant_end (=67 in MockTokenizer).
    assert stops == {0, 67}


def test_build_chat_prompt_returns_value_error_with_chained_exception():
    """The original exception is preserved as a chained cause for
    debugging."""
    model = _make_gpt(device="cpu")
    tok = _RaisingTokenizer(KeyError)
    engine = InferenceEngine(model, tok, _DEVICE)
    with pytest.raises(ValueError) as excinfo:
        engine._build_chat_prompt("hi")
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, KeyError)


class _NoneReturnTokenizer(MockTokenizer):
    """A tokenizer that returns ``None`` from ``encode_special`` (the
    HuggingFaceTokenizer pattern for missing tokens)."""

    def encode_special(self, name: str) -> int | None:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        if name == "<assistant_end>":
            return None  # type: ignore[return-value]
        return super().encode_special(name)


def test_build_chat_prompt_none_return_raises():
    """None return raises a clear ``ValueError`` rather than silently
    disabling the assistant-end stop token."""
    model = _make_gpt(device="cpu")
    tok = _NoneReturnTokenizer()
    engine = InferenceEngine(model, tok, _DEVICE)
    with pytest.raises(ValueError, match="None return silently disables"):
        engine._build_chat_prompt("hi")
