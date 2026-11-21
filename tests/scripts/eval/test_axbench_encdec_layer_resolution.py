"""Depth-relative layer resolution for the encoder-decoder baseline analogs."""

from __future__ import annotations

import re

import pytest

from steering.eval.axbench_upstream.encdec_baselines.layer_resolution import (
    VENDORED_LORA_LAYERS,
    VENDORED_QWEN_N_LAYER,
    VENDORED_STEERING_LAYER,
    resolve_vendored_layer,
    resolve_vendored_layers,
)


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def test_vendored_constants_match_the_qwen_sweep_configs() -> None:
    """Pinned against vendor/axbench/.../qwen2_5_3b_{main,aux}.yaml."""

    assert VENDORED_QWEN_N_LAYER == 36
    assert VENDORED_STEERING_LAYER == 28
    assert VENDORED_LORA_LAYERS == (7, 14, 21, 28)


def test_resolve_at_target_n_layer_12_reproduces_original_constants() -> None:
    """The 400M checkpoint's n_layer=12 must give back the pre-refactor 9/(2,5,7,9)."""

    assert resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=12) == 9
    assert resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=12) == (2, 5, 7, 9)


def test_resolve_at_target_n_layer_36_is_the_identity() -> None:
    assert resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=36) == 28
    assert resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=36) == (7, 14, 21, 28)


def test_resolve_for_the_100m_checkpoints_six_layer_decoder() -> None:
    """n_layer=6 must give four distinct, in-range LoRA layers plus an intervention layer."""

    intervention = resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=6)
    lora_layers = resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=6)

    assert intervention == 5
    assert lora_layers == (1, 2, 4, 5)
    assert len(set(lora_layers)) == len(lora_layers)
    assert all(0 <= layer < 6 for layer in lora_layers)
    assert 0 <= intervention < 6


def test_resolve_scales_up_beyond_the_vendored_reference_depth() -> None:
    intervention = resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=24)
    lora_layers = resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=24)

    assert intervention == 19
    assert lora_layers == (5, 9, 14, 19)
    assert len(set(lora_layers)) == len(lora_layers)
    assert all(0 <= layer < 24 for layer in lora_layers)


def test_resolve_layers_fails_loud_on_collision_at_a_pathologically_shallow_depth() -> None:
    """n=3 is the shallowest depth where every value resolves in-range yet collides."""

    assert resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=5) == (1, 2, 3, 4)
    with pytest.raises(
        ValueError,
        match=_exact(
            "target_n_layer=3 is too shallow to give vendored_layers=(7, 14, 21, 28) "
            "distinct fractionally-mapped layers: resolved (1, 1, 2, 2)"
        ),
    ):
        resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=3)


def test_resolve_layer_fails_loud_when_the_result_is_out_of_range() -> None:
    with pytest.raises(
        ValueError,
        match=_exact(
            "resolved layer 1 is out of range for a 1-layer decoder "
            "(vendored_layer=28 of 36)"
        ),
    ):
        resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=1)


@pytest.mark.parametrize("bad_value", [-1, 36, 100])
def test_resolve_layer_rejects_out_of_domain_vendored_layer(bad_value: int) -> None:
    with pytest.raises(ValueError, match="vendored_layer must be an integer in"):
        resolve_vendored_layer(bad_value, target_n_layer=12)


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, True])
def test_resolve_layer_rejects_invalid_target_n_layer(bad_value: object) -> None:
    with pytest.raises(ValueError, match="target_n_layer must be a positive integer"):
        resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=bad_value)  # type: ignore[arg-type]
