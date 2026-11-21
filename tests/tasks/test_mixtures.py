"""Tests for SFT mixture builders."""

import pytest

from tasks.mixtures import SFTMixture


class TestSFTMixture:
    def test_values(self):
        assert SFTMixture.STANDARD == "standard"
        assert SFTMixture.CAPABILITIES == "capabilities"
        assert SFTMixture.SAFETY == "safety"

    def test_from_string(self):
        assert SFTMixture("standard") == SFTMixture.STANDARD

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            SFTMixture("invalid")
