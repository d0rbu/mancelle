"""Tests for the shared repository error hierarchy."""

import pytest

from errors import FeatureMaskError, NanoGPTError


def test_feature_mask_error_is_nanogpt_error():
    assert issubclass(FeatureMaskError, NanoGPTError)
    with pytest.raises(NanoGPTError):
        raise FeatureMaskError("bad mask")
