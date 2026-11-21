"""Per-direction max-activation scaling at the shared BPB consumer (#1228)."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention
from tests.axbench_base.fixtures.steering_stats import write_steering_stats

_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "mini_axbench_base_run"
_LAYER = 8
_CONCEPT_ID = 1
_FEATURE_INDEX = 7
_ARTIFACT_VALUE = 5.0


def _concept() -> SimpleNamespace:
    return SimpleNamespace(concept_id=_CONCEPT_ID, feature_index=_FEATURE_INDEX)


def _write_direction(
    source_paths: SourcePaths,
    method_name: str,
    direction: t.Tensor,
    *,
    max_activation: float | None = _ARTIFACT_VALUE,
    metadata_overrides: dict[str, str] | None = None,
) -> Path:
    method = get_method_by_name(method_name)
    target = str(method.target_model)
    mode = method.mode.value
    path = (
        source_paths.diffmean_direction(target, mode, _CONCEPT_ID)
        if method.baseline.value == "diffmean"
        else source_paths.linear_probe_direction(target, mode, _CONCEPT_ID)
    )
    metadata = {
        "concept_id": str(_CONCEPT_ID),
        "mode": mode,
        "target_model": target,
        "layer": str(_LAYER),
        "n_pos": "2",
        "n_neg": "2",
    }
    if max_activation is not None:
        metadata[MAX_ACTIVATION_METADATA_KEY] = encode_max_activation(max_activation)
    if metadata_overrides:
        metadata.update(metadata_overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"direction": direction.contiguous()}, str(path), metadata=metadata)
    return path


def _build(
    method_name: str,
    *,
    coefficient: float | None,
    source_paths: SourcePaths,
    apply_max_act_scale: bool,
) -> Intervention:
    return build_intervention(
        method=get_method_by_name(method_name),
        concept=_concept(),
        coefficient=coefficient,
        source_paths=source_paths,
        diffmean_layer=_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=apply_max_act_scale,
    )


@pytest.mark.parametrize(
    "method_name", ["diffmean_decoder_only", "linear_probe_decoder_only"]
)
def test_positive_direction_coefficient_scaled(tmp_path, method_name):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, method_name, t.tensor([1.0, 0.0]))
    iv = _build(
        method_name,
        coefficient=4.0,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    assert iv.direction_coefficient == 4.0 * _ARTIFACT_VALUE


@pytest.mark.parametrize(
    "method_name", ["neg_diffmean_decoder_only", "neg_linear_probe_decoder_only"]
)
def test_negative_direction_coefficient_scaled(tmp_path, method_name):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, method_name, t.tensor([1.0, 0.0]))
    iv = _build(
        method_name,
        coefficient=1.5,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    assert iv.direction_coefficient == -abs(1.5) * _ARTIFACT_VALUE


def test_off_mode_is_identity_and_does_not_read_metadata(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        "diffmean_decoder_only",
        t.tensor([3.0, 4.0]),
        max_activation=None,
    )
    iv = _build(
        "diffmean_decoder_only",
        coefficient=4.0,
        source_paths=source_paths,
        apply_max_act_scale=False,
    )
    expected = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=t.tensor([3.0, 4.0]),
        direction_layer=_LAYER,
        coefficient=4.0,
    )
    assert iv == expected
    assert t.equal(cast(t.Tensor, iv.direction), cast(t.Tensor, expected.direction))


def test_missing_field_raises(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        "diffmean_decoder_only",
        t.tensor([1.0, 0.0]),
        max_activation=None,
    )
    with pytest.raises(AxBenchBpbError, match="missing 'max_activation'"):
        _build(
            "diffmean_decoder_only",
            coefficient=1.0,
            source_paths=source_paths,
            apply_max_act_scale=True,
        )


@pytest.mark.parametrize(
    ("field", "wrong", "match"),
    [
        ("concept_id", "999", r"direction artifact identity mismatch.*concept_id"),
        ("mode", "negative", r"direction artifact identity mismatch.*mode"),
        (
            "target_model",
            "encdec",
            r"direction artifact identity mismatch.*target_model",
        ),
        ("layer", "9", r"direction artifact layer mismatch"),
    ],
)
def test_binding_mismatch_matrix(tmp_path, field, wrong, match):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        "diffmean_decoder_only",
        t.tensor([1.0, 0.0]),
        metadata_overrides={field: wrong},
    )
    with pytest.raises(AxBenchBpbError, match=match):
        _build(
            "diffmean_decoder_only",
            coefficient=1.0,
            source_paths=source_paths,
            apply_max_act_scale=True,
        )


def test_zero_direction_zero_value_noop(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        "diffmean_decoder_only",
        t.zeros(2),
        max_activation=0.0,
    )
    iv = _build(
        "diffmean_decoder_only",
        coefficient=3.0,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    assert iv.direction_coefficient == 0.0
    assert t.equal(cast(t.Tensor, iv.direction), t.zeros(2))


def test_zero_value_nonzero_direction_raises(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        "diffmean_decoder_only",
        t.tensor([1.0, 0.0]),
        max_activation=0.0,
    )
    with pytest.raises(
        AxBenchBpbError, match="zero-direction max_activation mismatch"
    ):
        _build(
            "diffmean_decoder_only",
            coefficient=1.0,
            source_paths=source_paths,
            apply_max_act_scale=True,
        )


def test_non_unit_direction_normalized_once(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths, "diffmean_decoder_only", t.tensor([7.0, 0.0])
    )
    iv = _build(
        "diffmean_decoder_only",
        coefficient=2.0,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    assert iv.direction_coefficient == 2.0 * _ARTIFACT_VALUE
    assert t.equal(cast(t.Tensor, iv.direction), t.tensor([1.0, 0.0]))


def test_sae_stats_cannot_influence_scaling(tmp_path):
    """Re-introducing scale_for(concept.feature_index) MUST red this test."""
    source_paths = SourcePaths(source_run_dir=tmp_path / "source")
    _write_direction(
        source_paths, "diffmean_decoder_only", t.tensor([1.0, 0.0])
    )
    sae_value = 97.0
    write_steering_stats(
        tmp_path / "source" / "feature_statistics.safetensors",
        n_sae_features=16,
        max_mags={_FEATURE_INDEX: sae_value},
    )
    iv = _build(
        "diffmean_decoder_only",
        coefficient=2.0,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    assert iv.direction_coefficient == 2.0 * _ARTIFACT_VALUE
    assert iv.direction_coefficient != 2.0 * sae_value


def test_control_branch_unaffected(tmp_path):
    source_paths = SourcePaths(source_run_dir=_MINI)
    scaled = _build(
        "control_decoder_only",
        coefficient=None,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    unscaled = _build(
        "control_decoder_only",
        coefficient=None,
        source_paths=source_paths,
        apply_max_act_scale=False,
    )
    assert scaled == unscaled


def test_lora_branch_unaffected(tmp_path):
    source_paths = SourcePaths(source_run_dir=_MINI)
    scaled = _build(
        "lora_decoder_only",
        coefficient=None,
        source_paths=source_paths,
        apply_max_act_scale=True,
    )
    unscaled = _build(
        "lora_decoder_only",
        coefficient=None,
        source_paths=source_paths,
        apply_max_act_scale=False,
    )
    assert scaled == unscaled


def test_build_intervention_requires_scale_keyword(tmp_path):
    kwargs = {
        "method": get_method_by_name("control_decoder_only"),
        "concept": _concept(),
        "coefficient": None,
        "source_paths": SourcePaths(source_run_dir=tmp_path),
        "diffmean_layer": _LAYER,
        "target_model_meta": SimpleNamespace(),
        "inactive_sampler": None,
        "tokenizer_nanochat": SimpleNamespace(),
        "rng": random.Random(0),
    }
    with pytest.raises(TypeError, match="apply_max_act_scale"):
        cast(Any, build_intervention)(**kwargs)
