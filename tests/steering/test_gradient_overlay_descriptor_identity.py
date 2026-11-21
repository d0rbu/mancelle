"""Exact gradient-overlay descriptor identity (audit PR3, GA-009)."""

from __future__ import annotations

import pytest

from errors import GradientAttributionError
from steering.gradient_overlay import (
    assert_descriptor_matches,
    canonical_gradient_descriptor,
)


def _descriptor(*, release: str, l0: str | int | None) -> dict:
    return canonical_gradient_descriptor(
        attribution_mode="future_half_split",
        model="gemma-2-2b" if release == "gemmascope" else "gemma-3-4b-pt",
        model_dtype="float32",
        layer=8 if release == "gemmascope" else 22,
        sae_release=release,
        sae_width="16k" if release == "gemmascope" else "65k",
        sae_l0=l0,
        activation_type="residual",
        ignore_bos_features=True,
        min_final_segment_tokens=8,
        percentile_checksum="a" * 64,
    )


_ARCHIVED_V1_DESCRIPTOR = {
    "attribution_mode": "future_half_split",
    "score_reduction": "pct_relu_neg_max",
    "split_policy_version": 2,
    "min_final_segment_tokens": 8,
    "model": "gemma-2-2b",
    "model_dtype": "float32",
    "layer": 8,
    "sae_release": "gemmascope",
    "sae_width": "16k",
    "sae_l0": None,
    "activation_type": "residual",
    "ignore_bos_features": True,
    "percentile_checksum": "a" * 64,
}
_ARCHIVED_V2_DESCRIPTOR = {
    **_ARCHIVED_V1_DESCRIPTOR,
    "model": "gemma-3-4b-pt",
    "layer": 22,
    "sae_release": "gemmascope_2",
    "sae_width": "65k",
    "sae_l0": "medium",
}


@pytest.fixture(
    params=[
        pytest.param(
            ("gemmascope", None, _ARCHIVED_V1_DESCRIPTOR), id="v1-manifest"
        ),
        pytest.param(
            ("gemmascope_2", "medium", _ARCHIVED_V2_DESCRIPTOR), id="v2-manifest"
        ),
    ]
)
def valid_legacy_descriptor(request) -> tuple[str, str | int | None, dict]:
    """Representative archived GemmaScope-v1 and GemmaScope-v2 manifests."""
    return request.param


def test_valid_v1_v2_descriptors_remain_accepted(valid_legacy_descriptor):
    release, l0, archived = valid_legacy_descriptor
    assert_descriptor_matches(
        _descriptor(release=release, l0=l0),
        archived,
        artifact="archived fixture gradient overlay manifest",
    )


@pytest.mark.parametrize("missing_side", ["expected", "actual"])
def test_missing_key_is_not_equal_to_explicit_none(missing_side):
    expected = _descriptor(release="gemmascope", l0=None)
    actual = dict(expected)
    target = expected if missing_side == "expected" else actual
    del target["sae_l0"]

    with pytest.raises(GradientAttributionError, match="sae_l0"):
        assert_descriptor_matches(expected, actual, artifact="fixture gradient overlay")


def test_explicit_none_matches_explicit_none():
    expected = _descriptor(release="gemmascope", l0=None)
    assert_descriptor_matches(expected, dict(expected), artifact="fixture gradient overlay")


def test_value_difference_is_rejected():
    expected = _descriptor(release="gemmascope_2", l0="medium")
    actual = {**expected, "sae_l0": "big"}
    with pytest.raises(GradientAttributionError, match="sae_l0"):
        assert_descriptor_matches(expected, actual, artifact="fixture gradient overlay")


def test_unexpected_canonical_key_is_rejected():
    expected = _descriptor(release="gemmascope", l0=None)
    actual = {**expected, "unexpected_identity": None}
    with pytest.raises(GradientAttributionError, match="unexpected_identity"):
        assert_descriptor_matches(expected, actual, artifact="fixture gradient overlay")
