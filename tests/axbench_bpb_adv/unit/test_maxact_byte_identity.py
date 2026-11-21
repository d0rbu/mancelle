"""Effective-hook oracle for artifact-owned direction scaling (#1228)."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.coefficient_grids import _DIFFMEAN_SCALAR_GRID
from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv.intervention_build import build_steered_intervention

_CONCEPT_ID = 1
_LAYER = 8
_VALUE = 2.0
_ADDITIVE = "additive_direction"


def _write_artifacts(tmp_path, *, mode: str) -> SourcePaths:
    source_paths = SourcePaths(source_run_dir=tmp_path)
    for family in ("diffmean", "linear_probe"):
        path = (
            source_paths.diffmean_direction("decoder_only", mode, _CONCEPT_ID)
            if family == "diffmean"
            else source_paths.linear_probe_direction(
                "decoder_only", mode, _CONCEPT_ID
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"direction": t.tensor([0.0, 7.0])},
            str(path),
            metadata={
                "concept_id": str(_CONCEPT_ID),
                "mode": mode,
                "target_model": "decoder_only",
                "layer": str(_LAYER),
                MAX_ACTIVATION_METADATA_KEY: encode_max_activation(_VALUE),
            },
        )
    return source_paths


def _build(tmp_path, *, method_name: str, coefficient: float, magnitude: float):
    mode = "negative" if method_name.startswith("neg_") else "positive"
    return build_steered_intervention(
        method=get_method_by_name(method_name),
        concept=cast(
            Any, SimpleNamespace(concept_id=_CONCEPT_ID, feature_index=999)
        ),
        coefficient=coefficient,
        source_paths=_write_artifacts(tmp_path, mode=mode),
        diffmean_layer=_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        rng=random.Random(0),
        adversarial_method=_ADDITIVE,
        adversarial_strength=magnitude,
        counter_direction_family="diffmean",
        target="decoder_only",
        apply_max_act_scale=True,
    )


_METHODS = ["diffmean_decoder_only", "neg_diffmean_decoder_only"]
_MAGNITUDES = [0.0, 0.4, 6.0]


@pytest.mark.parametrize("method_name", _METHODS)
@pytest.mark.parametrize("coefficient", _DIFFMEAN_SCALAR_GRID)
@pytest.mark.parametrize("magnitude", _MAGNITUDES)
def test_effective_hook_uses_metadata_value_once(
    tmp_path, method_name, coefficient, magnitude
):
    iv = _build(
        tmp_path,
        method_name=method_name,
        coefficient=coefficient,
        magnitude=magnitude,
    )
    sign = 1.0 if method_name == "diffmean_decoder_only" else -1.0
    assert iv.direction is not None
    assert iv.direction_coefficient == sign * coefficient * _VALUE
    assert iv.direction.norm().item() == pytest.approx(1.0)
    assert (
        iv.direction * cast(float, iv.direction_coefficient)
    ).norm().item() == pytest.approx(coefficient * _VALUE)
    if magnitude == 0.0:
        assert iv.counter_direction is None
        assert iv.counter_direction_coefficient is None
    else:
        assert iv.counter_direction is not None
        assert iv.counter_direction_coefficient == -sign * magnitude * _VALUE
        assert iv.counter_direction.norm().item() == pytest.approx(1.0)
        assert (
            iv.counter_direction * cast(float, iv.counter_direction_coefficient)
        ).norm().item() == pytest.approx(magnitude * _VALUE)
