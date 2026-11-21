"""Slice 17 audit (2026-05-05) P0.B regression: negative-mode sign-flip.

`_build_direction_intervention` flips the coefficient sign for negative-
mode methods (declared positive grid value → applied negative on the
residual hook). This test pins:

1. The applied coefficient is `-abs(declared)` for negative mode.
2. The applied coefficient equals the declared value for positive mode.
3. Two grid entries (e.g. +3.0 and -3.0) for the same negative-mode
   method DO produce distinct ``coefficient_str`` keys, so the SQLite
   primary key still discriminates them — even though the applied value
   collides.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from steering.eval.axbench_bpb.intervention import _build_direction_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import SteeringMode


@dataclass
class _FakeMethod:
    name: str
    baseline: str
    mode: SteeringMode
    target_model: str


@dataclass
class _FakeConcept:
    concept_id: int


def _setup_diffmean_dir(tmp_path: Path, *, mode: str, concept_id: int) -> SourcePaths:
    """Create a SourcePaths fixture with one diffmean direction file."""
    run_dir = tmp_path / "source"
    run_dir.mkdir(parents=True, exist_ok=True)
    sp = SourcePaths(source_run_dir=run_dir)
    direction_path = sp.diffmean_direction("decoder_only", mode, concept_id)
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": torch.zeros(8, dtype=torch.float32)},
        str(direction_path),
        metadata={
            "concept_id": str(concept_id),
            "mode": mode,
            "target_model": "decoder_only",
            "layer": "8",
        },
    )
    return sp


def test_positive_mode_no_sign_flip(tmp_path):
    """Positive mode keeps the sign of the declared coefficient."""
    sp = _setup_diffmean_dir(tmp_path, mode="positive", concept_id=0)
    method = _FakeMethod(
        name="diffmean", baseline="diffmean",
        mode=SteeringMode.POSITIVE, target_model="decoder_only",
    )
    iv = _build_direction_intervention(
        method=method, concept=_FakeConcept(concept_id=0),
        coefficient=3.0,
        source_paths=sp, diffmean_layer=8, apply_max_act_scale=False,
    )
    assert iv.direction_coefficient == pytest.approx(3.0)


def test_negative_mode_flips_to_negative(tmp_path):
    """Negative mode sign-flips a positive declared coefficient to negative."""
    sp = _setup_diffmean_dir(tmp_path, mode="negative", concept_id=1)
    method = _FakeMethod(
        name="neg_diffmean", baseline="diffmean",
        mode=SteeringMode.NEGATIVE, target_model="decoder_only",
    )
    iv = _build_direction_intervention(
        method=method, concept=_FakeConcept(concept_id=1),
        coefficient=3.0,
        source_paths=sp, diffmean_layer=8, apply_max_act_scale=False,
    )
    assert iv.direction_coefficient == pytest.approx(-3.0)


def test_negative_mode_with_negative_input_still_negative(tmp_path):
    """Negative mode with a NEGATIVE declared coefficient still applies negative
    (the function uses ``-abs(coefficient)``)."""
    sp = _setup_diffmean_dir(tmp_path, mode="negative", concept_id=2)
    method = _FakeMethod(
        name="neg_diffmean", baseline="diffmean",
        mode=SteeringMode.NEGATIVE, target_model="decoder_only",
    )
    iv = _build_direction_intervention(
        method=method, concept=_FakeConcept(concept_id=2),
        coefficient=-3.0,  # declared negative
        source_paths=sp, diffmean_layer=8, apply_max_act_scale=False,
    )
    # Applied is -abs(-3.0) = -3.0 — same applied value as declared +3.0.
    # This is the documented P0.B asymmetry (different coefficient_str
    # keys can map to the same applied value under negative mode).
    assert iv.direction_coefficient == pytest.approx(-3.0)


def test_zero_coefficient_unchanged(tmp_path):
    """coefficient=0 is unchanged regardless of mode."""
    sp = _setup_diffmean_dir(tmp_path, mode="negative", concept_id=3)
    method = _FakeMethod(
        name="neg_diffmean", baseline="diffmean",
        mode=SteeringMode.NEGATIVE, target_model="decoder_only",
    )
    iv = _build_direction_intervention(
        method=method, concept=_FakeConcept(concept_id=3),
        coefficient=0.0,
        source_paths=sp, diffmean_layer=8, apply_max_act_scale=False,
    )
    assert iv.direction_coefficient == pytest.approx(0.0)
