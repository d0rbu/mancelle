"""Unit tests for the shared signed class-margin helper.

Pins the canonical sign rule (positive -> avoid-rich, negative -> rich-avoid)
that the axbench_bpb / axbench_bpb_adv pipelines all delegate to, so a divergent
edit at any single call site is caught.
"""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.class_margin import margin_sign, signed_class_margin
from steering.eval.axbench_base.types import SteeringMode


def test_margin_sign_positive_is_plus_one() -> None:
    assert margin_sign("positive") == 1.0
    assert margin_sign(SteeringMode.POSITIVE) == 1.0


def test_margin_sign_negative_is_minus_one() -> None:
    assert margin_sign("negative") == -1.0
    assert margin_sign(SteeringMode.NEGATIVE) == -1.0


def test_margin_sign_returns_float() -> None:
    assert isinstance(margin_sign("positive"), float)
    assert isinstance(margin_sign("negative"), float)


@pytest.mark.parametrize("bad_mode", ["", "POSITIVE", "neutral", "pos", "0", "unknown"])
def test_margin_sign_unknown_mode_raises_value_error(bad_mode: str) -> None:
    with pytest.raises(ValueError):
        margin_sign(bad_mode)


def test_signed_class_margin_positive() -> None:
    # positive: avoid - rich
    assert signed_class_margin(mode="positive", rich=1.0, avoid=3.0) == pytest.approx(2.0)


def test_signed_class_margin_negative() -> None:
    # negative: rich - avoid (sign flip)
    assert signed_class_margin(mode="negative", rich=1.0, avoid=3.0) == pytest.approx(-2.0)


def test_signed_class_margin_accepts_enum() -> None:
    assert signed_class_margin(
        mode=SteeringMode.POSITIVE, rich=2.0, avoid=5.0
    ) == pytest.approx(3.0)
    assert signed_class_margin(
        mode=SteeringMode.NEGATIVE, rich=2.0, avoid=5.0
    ) == pytest.approx(-3.0)


@pytest.mark.parametrize("mode", ["positive", "negative"])
@pytest.mark.parametrize("rich", [-2.5, 0.0, 1.0, 7.25])
@pytest.mark.parametrize("avoid", [-1.0, 0.0, 3.0, 9.5])
def test_signed_class_margin_equals_sign_times_avoid_minus_rich(
    mode: str, rich: float, avoid: float
) -> None:
    expected = margin_sign(mode) * (avoid - rich)
    assert signed_class_margin(mode=mode, rich=rich, avoid=avoid) == pytest.approx(
        expected
    )


def test_signed_class_margin_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        signed_class_margin(mode="sideways", rich=1.0, avoid=2.0)
