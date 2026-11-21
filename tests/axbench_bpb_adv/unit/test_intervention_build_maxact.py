"""Per-direction max-activation scaling of additive counters (#1228)."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save, save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.config import AdversarialMethod
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.intervention_build import (
    AdditiveCounterDirection,
    AdditiveCounterNoOp,
    AdditiveCounterNoOpReason,
    apply_additive_counter,
    attach_additive_counter,
    build_steered_intervention,
    resolve_additive_counter,
)

_LAYER = 8
_CONCEPT_ID = 1
_ADDITIVE = AdversarialMethod.ADDITIVE_DIRECTION.value


def _concept() -> Any:
    return SimpleNamespace(concept_id=_CONCEPT_ID, feature_index=7)


def _metadata(
    *,
    mode: str = "positive",
    target: str = "decoder_only",
    value: float | None = 5.0,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    metadata = {
        "concept_id": str(_CONCEPT_ID),
        "mode": mode,
        "target_model": target,
        "layer": str(_LAYER),
    }
    if value is not None:
        metadata[MAX_ACTIVATION_METADATA_KEY] = encode_max_activation(value)
    if overrides:
        metadata.update(overrides)
    return metadata


def _write_direction(
    source_paths: SourcePaths,
    *,
    family: str,
    mode: str = "positive",
    target: str = "decoder_only",
    value: float | None = 5.0,
    direction: t.Tensor | None = None,
    metadata: dict[str, str] | None = None,
) -> Path:
    path = (
        source_paths.diffmean_direction(target, mode, _CONCEPT_ID)
        if family == "diffmean"
        else source_paths.linear_probe_direction(target, mode, _CONCEPT_ID)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "direction": (
                t.tensor([3.0, 4.0]) if direction is None else direction
            ).contiguous()
        },
        str(path),
        metadata=_metadata(mode=mode, target=target, value=value)
        if metadata is None
        else metadata,
    )
    return path


def _counter(
    base: Intervention,
    *,
    source_paths: SourcePaths,
    method_name: str = "control_decoder_only",
    magnitude: float = 0.4,
    family: str = "diffmean",
    apply_max_act_scale: bool = True,
    source_validation: Any | None = None,
) -> Intervention:
    return apply_additive_counter(
        base,
        adversarial_method=_ADDITIVE,
        adversarial_strength=magnitude,
        method=get_method_by_name(method_name),
        concept=_concept(),
        source_paths=source_paths,
        diffmean_layer=_LAYER,
        target="decoder_only",
        counter_direction_family=family,
        apply_max_act_scale=apply_max_act_scale,
        source_validation=source_validation,
    )


def test_counter_scaled_by_its_own_artifact_value(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, family="diffmean", value=11.0)
    _write_direction(source_paths, family="linear_probe", value=97.0)
    iv = _counter(Intervention.none(), source_paths=source_paths)
    assert iv.counter_direction_coefficient == -0.4 * 11.0
    assert iv.counter_direction is not None
    assert t.allclose(iv.counter_direction, t.tensor([0.6, 0.8]))


def test_counter_value_comes_only_from_its_artifact(tmp_path):
    """The counter seam has NO stats parameter — non-influence is structural.

    The pre-#1228 SAE value reached this path through a SteeringScale provider
    built from cfg.feature_statistics_path; post-#1228 no argument of any
    counter seam can carry a stats file, so a planted stats file is inert BY
    CONSTRUCTION at this level (final-gate review: an earlier version of this
    test planted one anyway, which was vacuous). The live claims here are the
    exact-equality core (the coefficient is magnitude × THIS artifact's value
    and nothing else) plus the cross-family case below; stats non-influence
    with a real config is pinned at the bpb consumer and stage-4 dispatch
    sites.
    """
    source_paths = SourcePaths(source_run_dir=tmp_path)
    artifact_value = 5.0
    _write_direction(source_paths, family="diffmean", value=artifact_value)
    iv = _counter(
        Intervention.none(),
        source_paths=source_paths,
        magnitude=2.0,
    )
    assert iv.counter_direction_coefficient == -2.0 * artifact_value


def test_underflow_counter_coefficient_raises(tmp_path):
    # Nonzero magnitude × positive-but-subnormal scale underflows binary64 to
    # exactly 0.0 — NOT the validated zero-direction pair; must fail loud
    # rather than silently become the magnitude-0 control.
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths, family="diffmean", value=1.401298464324817e-45
    )
    with pytest.raises(
        AxBenchBpbAdvError, match="underflowed to 0.0 from nonzero magnitude"
    ):
        _counter(
            Intervention.none(),
            source_paths=source_paths,
            magnitude=5e-324,
        )


def test_negative_counter_sign_is_positive(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, family="diffmean", mode="negative", value=6.0)
    iv = _counter(
        Intervention.none(),
        source_paths=source_paths,
        method_name="neg_control_decoder_only",
    )
    assert iv.counter_direction_coefficient == 0.4 * 6.0


def test_distinct_base_and_counter_families_use_their_own_values(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, family="linear_probe", value=3.0)
    _write_direction(source_paths, family="diffmean", value=7.0)
    base = build_intervention(
        method=get_method_by_name("linear_probe_decoder_only"),
        concept=_concept(),
        coefficient=2.0,
        source_paths=source_paths,
        diffmean_layer=_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=True,
    )
    iv = _counter(base, source_paths=source_paths, family="diffmean", magnitude=1.0)
    assert iv.direction_coefficient == 2.0 * 3.0
    assert iv.counter_direction_coefficient == -1.0 * 7.0


def test_same_artifact_commensurability(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, family="diffmean", value=5.0)
    base = build_intervention(
        method=get_method_by_name("diffmean_decoder_only"),
        concept=_concept(),
        coefficient=4.0,
        source_paths=source_paths,
        diffmean_layer=_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer_nanochat=SimpleNamespace(),
        rng=random.Random(0),
        apply_max_act_scale=True,
    )
    iv = _counter(base, source_paths=source_paths, magnitude=1.0)
    assert iv.direction_coefficient is not None
    assert iv.counter_direction_coefficient is not None
    assert iv.direction_coefficient + iv.counter_direction_coefficient == (
        4.0 - 1.0
    ) * 5.0


def test_magnitude_zero_returns_base_object_without_artifact_read(tmp_path):
    base = Intervention.none()
    iv = _counter(
        base,
        source_paths=SourcePaths(source_run_dir=tmp_path / "missing"),
        magnitude=0.0,
    )
    assert iv is base


def test_zero_direction_scaled_counter_is_explicit_noop(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        family="diffmean",
        value=0.0,
        direction=t.zeros(2),
    )
    resolution = resolve_additive_counter(
        adversarial_method=_ADDITIVE,
        adversarial_strength=3.0,
        method=get_method_by_name("control_decoder_only"),
        concept=_concept(),
        source_paths=source_paths,
        diffmean_layer=_LAYER,
        target="decoder_only",
        counter_direction_family="diffmean",
        apply_max_act_scale=True,
    )
    assert resolution == AdditiveCounterNoOp(
        AdditiveCounterNoOpReason.ZERO_COEFFICIENT
    )
    base = Intervention.none()
    assert attach_additive_counter(base, resolution) is base


def test_non_additive_resolution_is_explicit_noop(tmp_path):
    resolution = resolve_additive_counter(
        adversarial_method="faster_gcg",
        adversarial_strength=1.0,
        method=get_method_by_name("control_decoder_only"),
        concept=_concept(),
        source_paths=SourcePaths(source_run_dir=tmp_path),
        diffmean_layer=_LAYER,
        target="decoder_only",
        counter_direction_family="diffmean",
        apply_max_act_scale=True,
    )
    assert resolution == AdditiveCounterNoOp(
        AdditiveCounterNoOpReason.NOT_ADDITIVE_DIRECTION
    )


def test_e3_missing_max_activation_raises(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, family="diffmean", value=None)
    with pytest.raises(AxBenchBpbAdvError, match="missing 'max_activation'"):
        _counter(Intervention.none(), source_paths=source_paths)


@pytest.mark.parametrize(
    ("field", "wrong", "match"),
    [
        ("concept_id", "9", r"direction artifact identity mismatch.*concept_id"),
        ("mode", "negative", r"direction artifact identity mismatch.*mode"),
        (
            "target_model",
            "encdec",
            r"direction artifact identity mismatch.*target_model",
        ),
        ("layer", "9", r"direction artifact layer mismatch"),
    ],
)
def test_e7_counter_binding_mismatch_names_field(tmp_path, field, wrong, match):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    metadata = _metadata(overrides={field: wrong})
    _write_direction(source_paths, family="diffmean", metadata=metadata)
    with pytest.raises(AxBenchBpbAdvError, match=match):
        _counter(Intervention.none(), source_paths=source_paths)


def test_e11_scaled_counter_requires_metadata(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        family="diffmean",
        metadata=_metadata(value=None),
    )
    with pytest.raises(AxBenchBpbAdvError, match="missing 'max_activation'"):
        _counter(Intervention.none(), source_paths=source_paths)


def test_certified_bytes_carry_direction_metadata(tmp_path):
    payload = bytes(
        save(
            {"direction": t.tensor([0.0, 2.0])},
            metadata=_metadata(value=9.0),
        )
    )

    class _Validation:
        def direction_artifact_bytes(self, **_kwargs: Any) -> bytes:
            return payload

    iv = _counter(
        Intervention.none(),
        source_paths=SourcePaths(source_run_dir=tmp_path),
        source_validation=_Validation(),
    )
    assert iv.counter_direction_coefficient == -0.4 * 9.0
    assert t.equal(cast(t.Tensor, iv.counter_direction), t.tensor([0.0, 1.0]))


def test_off_mode_preserves_raw_direction_and_coefficient(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(
        source_paths,
        family="diffmean",
        metadata=_metadata(value=None),
    )
    iv = _counter(
        Intervention.none(),
        source_paths=source_paths,
        apply_max_act_scale=False,
    )
    assert iv.counter_direction_coefficient == -0.4
    assert t.equal(cast(t.Tensor, iv.counter_direction), t.tensor([3.0, 4.0]))


def test_build_steered_intervention_threads_scale_mode(tmp_path):
    source_paths = SourcePaths(source_run_dir=tmp_path)
    _write_direction(source_paths, family="diffmean", value=5.0)
    iv = build_steered_intervention(
        method=get_method_by_name("diffmean_decoder_only"),
        concept=_concept(),
        coefficient=4.0,
        source_paths=source_paths,
        diffmean_layer=_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        rng=random.Random(0),
        adversarial_method=_ADDITIVE,
        adversarial_strength=1.0,
        counter_direction_family="diffmean",
        target="decoder_only",
        apply_max_act_scale=True,
    )
    assert iv.direction_coefficient == 4.0 * 5.0
    assert iv.counter_direction_coefficient == -1.0 * 5.0


def test_public_counter_seams_require_scale_keyword(tmp_path):
    common = {
        "adversarial_method": _ADDITIVE,
        "adversarial_strength": 0.0,
        "method": get_method_by_name("control_decoder_only"),
        "concept": _concept(),
        "source_paths": SourcePaths(source_run_dir=tmp_path),
        "diffmean_layer": _LAYER,
        "target": "decoder_only",
        "counter_direction_family": "diffmean",
    }
    with pytest.raises(TypeError, match="apply_max_act_scale"):
        cast(Any, resolve_additive_counter)(**common)
    with pytest.raises(TypeError, match="apply_max_act_scale"):
        cast(Any, apply_additive_counter)(Intervention.none(), **common)


def test_attach_additive_counter_validation() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="bare None"):
        attach_additive_counter(
            Intervention.none(),
            cast(AdditiveCounterDirection | AdditiveCounterNoOp, None),
        )
    with pytest.raises(AxBenchBpbAdvError, match="requires an Intervention"):
        attach_additive_counter(
            cast(Intervention, object()),
            AdditiveCounterNoOp(AdditiveCounterNoOpReason.NOT_ADDITIVE_DIRECTION),
        )
