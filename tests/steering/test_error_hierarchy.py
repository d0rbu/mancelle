"""Tests for the custom exception hierarchy."""

import pickle

import pytest

from errors import (
    DataLoadError,
    ExtractionError,
    NanoGPTError,
    ParseError,
    ValidationError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from NanoGPTError."""

    @pytest.mark.parametrize(
        "exc_cls",
        [DataLoadError, ExtractionError, ValidationError, ParseError],
    )
    def test_inherits_from_base(self, exc_cls):
        assert issubclass(exc_cls, NanoGPTError)
        assert issubclass(exc_cls, Exception)

    @pytest.mark.parametrize(
        "exc_cls",
        [NanoGPTError, DataLoadError, ExtractionError, ValidationError, ParseError],
    )
    def test_message_preserved(self, exc_cls):
        msg = "something went wrong"
        exc = exc_cls(msg)
        assert str(exc) == msg

    @pytest.mark.parametrize(
        "exc_cls",
        [NanoGPTError, DataLoadError, ExtractionError, ValidationError, ParseError],
    )
    def test_picklable(self, exc_cls):
        exc = exc_cls("test message")
        restored = pickle.loads(pickle.dumps(exc))
        assert str(restored) == "test message"
        assert type(restored) is exc_cls

    def test_catch_base_catches_all(self):
        for exc_cls in [DataLoadError, ExtractionError, ValidationError, ParseError]:
            with pytest.raises(NanoGPTError):
                raise exc_cls("test")

    def test_catch_specific_does_not_catch_sibling(self):
        with pytest.raises(DataLoadError):
            raise DataLoadError("test")
        with pytest.raises(ParseError):
            raise ParseError("test")
