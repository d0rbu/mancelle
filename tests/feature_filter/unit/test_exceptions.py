"""FeatureFilterError is a NanoGPTError subclass."""
from errors import FeatureFilterError, NanoGPTError
from steering.eval.feature_filter.types import (
    ConcurrentRunnerError,
    ConfigValidationError,
    CostCapExceeded,
    InvalidHistogramError,
    NoNonzeroActivationsError,
    PromptTooLongError,
    ResumeMismatchError,
    SchemaVersionError,
)


def test_feature_filter_error_is_nanogpt_error() -> None:
    assert issubclass(FeatureFilterError, NanoGPTError)


def test_feature_filter_error_carries_message() -> None:
    exc = FeatureFilterError("explicit cause here")
    assert str(exc) == "explicit cause here"


def test_all_feature_filter_subclasses() -> None:
    for cls in (
        ConfigValidationError,
        ConcurrentRunnerError,
        CostCapExceeded,
        InvalidHistogramError,
        NoNonzeroActivationsError,
        PromptTooLongError,
        ResumeMismatchError,
        SchemaVersionError,
    ):
        assert issubclass(cls, FeatureFilterError)
        exc = cls("ctx")
        assert str(exc) == "ctx"
