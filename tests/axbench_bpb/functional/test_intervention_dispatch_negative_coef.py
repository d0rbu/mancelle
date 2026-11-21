"""Negative-mode coefficient negation in `_build_direction_intervention`.

axbench-base's `_dispatch_scalar_direction` flips the sign of the
coefficient when ``method.mode == SteeringMode.NEGATIVE``
(`steering/eval/axbench_base/intervention_dispatch.py:420-422`):

    effective_coefficient = float(coefficient)
    if method.mode == SteeringMode.NEGATIVE:
        effective_coefficient = -abs(effective_coefficient)

axbench-bpb's `_build_direction_intervention` must match — otherwise
every `neg_diffmean_*` and `neg_linear_probe_*` method installs a
residual hook that ADDS the direction instead of subtracting,
silently inverting the BPB sign on the leaderboard.

These tests exercise the helper directly and assert on
``Intervention.direction_coefficient``. They use real safetensors
direction files on disk (so `load_file` succeeds) but no model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import SteeringMode


# Helpers copied inline from
# tests/axbench_bpb/integration/test_axbench_base_dispatch_parity.py
# (cross-file test imports are flaky; per project convention we copy
# rather than import).
@dataclass
class _FakeMethod:
    """Minimal stand-in for axbench_base.method_registry.Method."""
    name: str
    baseline: str
    mode: SteeringMode
    target_model: str


@dataclass
class _FakeConcept:
    concept_id: int


def _make_source_paths(tmp_path: Path) -> SourcePaths:
    """Build a SourcePaths rooted at tmp_path with empty dirs."""
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    return SourcePaths(source_run_dir=run_dir)


def _write_diffmean_direction(
    sp: SourcePaths, *, target: str, mode: str, concept_id: int,
) -> Path:
    """Write a real safetensors direction file at the expected location."""
    path = sp.diffmean_direction(target, mode, concept_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": torch.ones(8, dtype=torch.float32)},
        str(path),
        metadata={
            "concept_id": str(concept_id),
            "mode": mode,
            "target_model": target,
            "layer": "4",
        },
    )
    return path


def _write_linear_probe_direction(
    sp: SourcePaths, *, target: str, mode: str, concept_id: int,
) -> Path:
    """Write a real safetensors direction file at the linear_probe location."""
    path = sp.linear_probe_direction(target, mode, concept_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": torch.ones(8, dtype=torch.float32)},
        str(path),
        metadata={
            "concept_id": str(concept_id),
            "mode": mode,
            "target_model": target,
            "layer": "4",
        },
    )
    return path


def test_negative_mode_diffmean_negates_coefficient(tmp_path: Path) -> None:
    """coefficient=3.0 + mode=NEGATIVE => direction_coefficient == -3.0.

    Mirrors `_dispatch_scalar_direction`'s
    `effective_coefficient = -abs(coefficient)` for negative mode.
    """
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="neg_diffmean_decoder_only", baseline="diffmean",
        mode=SteeringMode.NEGATIVE, target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=11)
    _write_diffmean_direction(sp, target="decoder_only", mode="negative", concept_id=11)

    iv = intervention_module._build_direction_intervention(
        method=method, concept=concept, coefficient=3.0,
        source_paths=sp, diffmean_layer=4, apply_max_act_scale=False,
    )

    assert iv.direction_coefficient == -3.0


def test_positive_mode_diffmean_preserves_coefficient(tmp_path: Path) -> None:
    """coefficient=3.0 + mode=POSITIVE => direction_coefficient == 3.0 (unchanged)."""
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="diffmean_decoder_only", baseline="diffmean",
        mode=SteeringMode.POSITIVE, target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=12)
    _write_diffmean_direction(sp, target="decoder_only", mode="positive", concept_id=12)

    iv = intervention_module._build_direction_intervention(
        method=method, concept=concept, coefficient=3.0,
        source_paths=sp, diffmean_layer=4, apply_max_act_scale=False,
    )

    assert iv.direction_coefficient == 3.0


def test_negative_mode_negation_uses_abs(tmp_path: Path) -> None:
    """coefficient=-2.5 + mode=NEGATIVE => direction_coefficient == -2.5.

    The negation rule is `effective = -abs(c)`, NOT plain `effective = -c`.
    With an already-negative input, `-abs(-2.5) == -2.5`. This test pins
    that semantics so a regression to plain negation (-(-2.5) == 2.5)
    would be caught.
    """
    sp = _make_source_paths(tmp_path)
    method = _FakeMethod(
        name="neg_linear_probe_encdec", baseline="linear_probe",
        mode=SteeringMode.NEGATIVE, target_model="encdec",
    )
    concept = _FakeConcept(concept_id=13)
    _write_linear_probe_direction(sp, target="encdec", mode="negative", concept_id=13)

    iv = intervention_module._build_direction_intervention(
        method=method, concept=concept, coefficient=-2.5,
        source_paths=sp, diffmean_layer=4, apply_max_act_scale=False,
    )

    assert iv.direction_coefficient == -2.5
