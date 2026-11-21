"""Unit tests for :class:`NeuronpediaConfig`."""

from __future__ import annotations

import dataclasses

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.neuronpedia_config import NeuronpediaConfig


class TestDefaults:
    def test_defaults_match_documented_values(self) -> None:
        cfg = NeuronpediaConfig()
        assert cfg.max_concurrent == 10
        assert cfg.max_retries == 3
        assert cfg.timeout_seconds == 30.0
        assert cfg.backoff_base_seconds == 1.0
        assert cfg.backoff_max_seconds == 60.0
        assert cfg.requests_per_minute is None  # no global rate limit by default

    def test_is_frozen(self) -> None:
        cfg = NeuronpediaConfig()
        # Slice 14 audit X2: narrowed from bare ``Exception``. ``frozen=True``
        # dataclasses raise ``FrozenInstanceError`` (a subclass of
        # ``AttributeError``) on attribute set.
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.max_concurrent = 5  # ty: ignore[invalid-assignment]

    def test_can_override_fields(self) -> None:
        cfg = NeuronpediaConfig(max_concurrent=4, max_retries=5)
        assert cfg.max_concurrent == 4
        assert cfg.max_retries == 5
        # untouched fields keep defaults
        assert cfg.timeout_seconds == 30.0


class TestValidation:
    def test_negative_max_concurrent_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="max_concurrent"):
            NeuronpediaConfig(max_concurrent=0)

    def test_negative_max_retries_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="max_retries"):
            NeuronpediaConfig(max_retries=-1)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="timeout_seconds"):
            NeuronpediaConfig(timeout_seconds=0.0)

    def test_negative_backoff_base_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="backoff_base_seconds"):
            NeuronpediaConfig(backoff_base_seconds=0.0)

    def test_max_lt_base_raises(self) -> None:
        with pytest.raises(ValueError, match="backoff_max_seconds"):
            NeuronpediaConfig(backoff_base_seconds=10.0, backoff_max_seconds=5.0)

    def test_zero_requests_per_minute_raises(self) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="requests_per_minute"):
            NeuronpediaConfig(requests_per_minute=0)

    def test_none_requests_per_minute_allowed(self) -> None:
        cfg = NeuronpediaConfig(requests_per_minute=None)
        assert cfg.requests_per_minute is None
