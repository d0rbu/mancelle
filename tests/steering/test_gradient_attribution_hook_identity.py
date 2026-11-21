"""Exact loaded-SAE hook identity for gradient attribution (GA-008)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from errors import GradientAttributionError
from steering.gradient_attribution import decoder_linearity_gate
from tests.reference_gradient_attribution import make_tiny_sae

EXPECTED_HOOK = "blocks.2.hook_resid_post"


def _sae_with_hook(
    *, direct: str | None = None, metadata: str | None = None
):
    base = make_tiny_sae("standard")
    cfg = SimpleNamespace(
        architecture="standard",
        normalize_activations="none",
        finetuning_scaling_factor=False,
    )
    if direct is not None:
        cfg.hook_name = direct
    if metadata is not None:
        cfg.metadata = SimpleNamespace(hook_name=metadata)
    base.cfg = cfg
    return base


@pytest.mark.parametrize(
    ("direct", "metadata"),
    [(EXPECTED_HOOK, None), (None, EXPECTED_HOOK), (EXPECTED_HOOK, EXPECTED_HOOK)],
)
def test_exact_resolver_hook_identity_passes(direct, metadata):
    decoder_linearity_gate(
        _sae_with_hook(direct=direct, metadata=metadata),
        expected_hook_name=EXPECTED_HOOK,
    )


@pytest.mark.parametrize(
    ("direct", "metadata", "match"),
    [
        ("blocks.2.hook_resid_pre", None, "expected"),
        ("blocks.3.hook_resid_post", None, "expected"),
        (None, None, "missing"),
        (EXPECTED_HOOK, "blocks.3.hook_resid_post", "conflicting"),
    ],
)
def test_non_exact_or_unproven_hook_identity_is_rejected(direct, metadata, match):
    with pytest.raises(GradientAttributionError, match=match):
        decoder_linearity_gate(
            _sae_with_hook(direct=direct, metadata=metadata),
            expected_hook_name=EXPECTED_HOOK,
        )


@pytest.mark.parametrize(
    "bad_expected",
    ["blocks.2.hook_resid_pre", "hook_resid_post"],
)
def test_expected_hook_must_be_the_configured_layer_resid_post(bad_expected):
    with pytest.raises(GradientAttributionError, match="resolved SAE hook"):
        decoder_linearity_gate(
            _sae_with_hook(direct=EXPECTED_HOOK),
            expected_hook_name=bad_expected,
        )
