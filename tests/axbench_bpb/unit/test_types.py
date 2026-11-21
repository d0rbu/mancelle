# tests/axbench_bpb/unit/test_types.py
"""Unit tests for axbench-bpb types."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest
import torch

from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.types import SteeringMode as UpstreamSteeringMode
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    BpbScoreRow,
    Intervention,
    InterventionKind,
    LoraParams,
    ScoringInputs,
    Stage,
    SteeringMode,
    TargetModel,
    WorkUnit,
)
from tests.runtime_attribute_helpers import set_runtime_attribute


def test_intervention_counter_direction_defaults_none() -> None:
    iv = Intervention.lora(adapter_path=Path("/tmp/a.pt"))
    assert iv.counter_direction is None
    assert iv.counter_direction_layer is None
    assert iv.counter_direction_coefficient is None


def test_with_counter_direction_sets_addon_and_preserves_base() -> None:
    """The counter-direction is an ORTHOGONAL add-on: it sets the three counter
    fields and preserves the base intervention's kind + family fields. The base
    instance is unchanged (frozen -> returns a copy)."""
    direction = torch.ones(4)
    base = Intervention.lora(adapter_path=Path("/tmp/a.pt"))
    out = base.with_counter_direction(direction=direction, layer=6, coefficient=-2.0)

    assert out.counter_direction is not None
    assert torch.equal(out.counter_direction, direction)
    assert out.counter_direction_layer == 6
    assert out.counter_direction_coefficient == -2.0
    # base intervention preserved
    assert out.kind is InterventionKind.LORA
    assert out.lora_adapter_path == base.lora_adapter_path
    # original untouched
    assert base.counter_direction is None


def test_with_counter_direction_composes_with_ours_encdec() -> None:
    direction = torch.ones(4)
    base = Intervention.ours(feature_conditions={1: 0})
    out = base.with_counter_direction(direction=direction, layer=6, coefficient=3.0)
    assert out.kind is InterventionKind.OURS
    assert out.feature_conditions == {1: 0}
    assert out.target_model == "encdec"
    assert out.counter_direction_coefficient == 3.0


def test_axbench_bpb_error_is_exception():
    exc = AxBenchBpbError("hello")
    assert isinstance(exc, Exception)
    assert str(exc) == "hello"


def test_axbench_bpb_error_is_catchable_as_exception():
    with pytest.raises(AxBenchBpbError, match="boom"):
        raise AxBenchBpbError("boom")


def test_stage_enum_values():
    assert Stage.PREPARE_CONTEXT_CACHE.value == "prepare_context_cache"
    assert Stage.BPB_SCORE.value == "bpb_score"
    assert Stage.BPB_AGGREGATE.value == "bpb_aggregate"
    assert Stage.ALL.value == "all"


def test_stage_coerce_valid():
    assert Stage.coerce("bpb_score") is Stage.BPB_SCORE
    assert Stage.coerce("all") is Stage.ALL


def test_stage_coerce_invalid():
    with pytest.raises(AxBenchBpbError, match="Unknown stage"):
        Stage.coerce("nonsense")


def test_target_model_coerce_valid() -> None:
    assert TargetModel.coerce("decoder_only") is TargetModel.DECODER_ONLY
    assert TargetModel.coerce(TargetModel.ENCDEC) is TargetModel.ENCDEC


def test_target_model_coerce_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown target model"):
        TargetModel.coerce("bogus")


def test_target_model_preserves_string_boundary_compatibility() -> None:
    mapping: dict[str, str] = {TargetModel.ENCDEC: "value"}
    assert TargetModel.ENCDEC == "encdec"
    assert mapping["encdec"] == "value"
    assert json.dumps({"target": TargetModel.ENCDEC}) == '{"target": "encdec"}'


def test_steering_mode_reexport():
    assert SteeringMode is UpstreamSteeringMode


def test_scoring_inputs_construction():
    si = ScoringInputs(
        sample_id="abc123",
        concept_id=5,
        mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text="the ",
        prefix_hash="def456",
        prefix_tokens=[1, 2],
        output_tokens=[3, 4, 5],
        prefix_context_segments=(),
        n_prefix=2,
        n_output=3,
        output_byte_weights=torch.tensor([1, 2, 1], dtype=torch.int64),
        truncated=False,
    )
    assert si.sample_id == "abc123"
    assert si.mode == SteeringMode.POSITIVE


def test_work_unit_equality_and_hashable():
    a = WorkUnit("m", 1, "s", "0")
    b = WorkUnit("m", 1, "s", "0")
    c = WorkUnit("m", 2, "s", "0")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)


def test_bpb_score_row_construction():
    row = BpbScoreRow(
        method_name="ours_extract_and_add_encdec",
        concept_id=42,
        mode="positive",
        content_class="concept_rich",
        target="encdec",
        coefficient_str="None",
        sample_id="abc123",
        bpb_intervened=1.234,
        n_output_tokens=10,
        n_output_bytes=40,
        truncated=False,
    )
    assert row.bpb_intervened == 1.234


def test_intervention_empty_prompting_construction():
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    assert iv.kind is InterventionKind.PROMPTING
    assert iv.kind == "prompting"
    assert iv.target_model == "encdec"
    assert iv.feature_conditions is None
    assert iv.direction is None
    assert iv.prompt_prefix_tokens == []


def test_intervention_none_construction():
    iv = Intervention.none()
    assert iv.kind is InterventionKind.NONE
    assert iv.kind == "none"
    assert iv.target_model == "decoder_only"
    assert iv.feature_conditions is None
    assert iv.direction is None
    assert iv.prompt_prefix_tokens is None


def test_intervention_ours_construction():
    iv = Intervention.ours(feature_conditions={10: 3, 20: 5})
    assert iv.kind is InterventionKind.OURS
    assert iv.kind == "ours"
    assert iv.target_model == "encdec"
    assert iv.feature_conditions == {10: 3, 20: 5}


def test_intervention_direction_construction():
    d = torch.zeros(16)
    iv = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=d, direction_layer=6, coefficient=1.5,
    )
    assert iv.kind is InterventionKind.DIFFMEAN
    assert iv.kind == "diffmean"
    assert iv.target_model == "decoder_only"
    assert iv.direction_coefficient == 1.5


def test_intervention_lora_construction(tmp_path):
    adapter = tmp_path / "fake_adapter.safetensors"
    iv = Intervention.lora(adapter_path=adapter)
    assert iv.kind is InterventionKind.LORA
    assert iv.kind == "lora"
    assert iv.target_model == "decoder_only"
    assert iv.lora_adapter_path == adapter
    assert iv.direction is None
    assert iv.feature_conditions is None


def test_intervention_prompting_construction():
    tokens = [1, 2, 3]
    iv = Intervention.prompting(target_model="decoder_only", prompt_prefix_tokens=tokens)
    assert iv.kind is InterventionKind.PROMPTING
    assert iv.kind == "prompting"
    assert iv.target_model == "decoder_only"
    assert iv.prompt_prefix_tokens == [1, 2, 3]
    # Factory must defensively copy the caller's list.
    tokens.append(99)
    assert iv.prompt_prefix_tokens == [1, 2, 3]


def test_intervention_from_direction_linear_probe_branch():
    """Covers the second branch of the Literal[diffmean, linear_probe] kind
    constraint; the other branch is in test_intervention_direction_construction."""
    d = torch.ones(8)
    iv = Intervention.from_direction(
        kind="linear_probe", target_model="encdec",
        direction=d, direction_layer=12, coefficient=2.0,
    )
    assert iv.kind is InterventionKind.LINEAR_PROBE
    assert iv.kind == "linear_probe"
    assert iv.target_model == "encdec"
    assert iv.direction_layer == 12
    assert iv.direction_coefficient == 2.0


def test_intervention_is_unhashable():
    """__hash__ is explicitly None — spec-mandated mutable fields
    (dict/list/Tensor) preclude a consistent hash; disabling catches
    accidental use as a dict key / set member early."""
    iv = Intervention.prompting(target_model="encdec", prompt_prefix_tokens=[])
    with pytest.raises(TypeError, match="unhashable"):
        hash(iv)


def test_intervention_rejects_unknown_kind_at_construction():
    with pytest.raises(ValueError, match="Unknown intervention kind"):
        Intervention(kind="bogus", target_model="encdec")


def test_intervention_coerces_valid_string_kind_to_enum():
    iv = Intervention(kind="none", target_model="decoder_only")
    assert iv.kind is InterventionKind.NONE


def test_intervention_coerces_valid_string_target_to_enum() -> None:
    iv = Intervention(kind="none", target_model="decoder_only")
    assert iv.target_model is TargetModel.DECODER_ONLY


def test_intervention_rejects_unknown_target_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown target model"):
        Intervention(kind="none", target_model="bogus")


def test_intervention_from_direction_rejects_non_direction_kind():
    with pytest.raises(ValueError, match="direction interventions"):
        Intervention.from_direction(
            kind=InterventionKind.LORA,
            target_model="decoder_only",
            direction=torch.zeros(4),
            direction_layer=0,
            coefficient=1.0,
        )


# ---------------------------------------------------------------------------
# LoraParams — bundles the co-traveling (cfg, layer) attach pair so the
# all-or-nothing invariant is enforced by construction.
# ---------------------------------------------------------------------------


def test_lora_params_construction_holds_cfg_and_layer():
    cfg = LoraConfig(rank=4, alpha=8.0)
    params = LoraParams(cfg=cfg, layer=6)
    assert params.cfg is cfg
    assert params.layer == 6


def test_lora_params_is_frozen():
    """Frozen like every other axbench-bpb spec dataclass — the attach
    parameters are an immutable bundle, not a mutable bag."""
    params = LoraParams(cfg=LoraConfig(rank=4, alpha=8.0), layer=6)
    with pytest.raises(FrozenInstanceError):
        set_runtime_attribute(params, "layer", 7)


def test_lora_params_requires_cfg():
    """Both-or-neither by construction: you cannot build a LoraParams with
    only ``layer`` — ``cfg`` is a required field, so the half-specified
    state the old loose ``(lora_cfg, lora_layer)`` pair allowed (cfg=None,
    layer set) is now unrepresentable."""
    constructor = cast(Callable[..., LoraParams], LoraParams)
    with pytest.raises(TypeError):
        constructor(layer=6)


def test_lora_params_requires_layer():
    """Symmetric to ``test_lora_params_requires_cfg``: you cannot build a
    LoraParams with only ``cfg`` — ``layer`` is required, so the other
    half-specified state (cfg set, layer=None) is also unrepresentable."""
    constructor = cast(Callable[..., LoraParams], LoraParams)
    with pytest.raises(TypeError):
        constructor(cfg=LoraConfig(rank=4, alpha=8.0))


def test_lora_params_equality_by_value():
    """Frozen dataclass eq is structural — two bundles with the same cfg
    and layer compare equal."""
    cfg = LoraConfig(rank=4, alpha=8.0)
    assert LoraParams(cfg=cfg, layer=6) == LoraParams(cfg=cfg, layer=6)
    assert LoraParams(cfg=cfg, layer=6) != LoraParams(cfg=cfg, layer=7)
