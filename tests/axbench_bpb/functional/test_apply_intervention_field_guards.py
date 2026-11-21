"""Each missing-field guard in ``apply_intervention`` raises AxBenchBpbError.

Five guards (intervention.py):
- diffmean / linear_probe: ``direction is None``
- diffmean / linear_probe: ``direction_layer is None``
- diffmean / linear_probe: ``direction_coefficient is None``
- lora: ``lora_adapter_path is None``
- lora: ``lora`` kwarg missing (the ``LoraParams`` bundle)

The factory classmethods on ``Intervention`` enforce per-family invariants,
so these tests bypass them by calling the dataclass constructor directly
with a deliberately stripped field. The guards in ``apply_intervention``
are the second line of defense — if a future refactor adds a new factory
that drops a field, the corresponding guard still fires here.

The ``direction_missing`` case for kind=diffmean is already covered by
``test_apply_intervention_contract.py:test_apply_direction_missing_direction_raises``.
This file fills in the remaining guards (direction_layer / coefficient
for diffmean, the lora-bundle-missing variant, and adds a parallel
linear_probe direction-missing test to confirm both kinds in the shared
branch fire).

Note: the old "only ``lora_cfg`` supplied" / "only ``lora_layer`` supplied"
disjunction guard collapsed when the co-traveling pair was bundled into the
:class:`LoraParams` dataclass — a half-specified bundle is now unrepresentable
(enforced by construction, covered in ``unit/test_types.py``), so the runtime
guard reduces to the single ``lora is None`` check exercised below.
"""
from __future__ import annotations

import pytest
import torch

from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention, LoraParams

# ---------------------------------------------------------------------------
# diffmean / linear_probe field guards (intervention.py:209-215)
# ---------------------------------------------------------------------------


def test_diffmean_missing_direction_layer_raises():
    """diffmean with direction set but direction_layer=None → AxBenchBpbError."""
    iv = Intervention(
        kind="diffmean", target_model="decoder_only",
        direction=torch.zeros(4),
        direction_layer=None,
        direction_coefficient=1.0,
    )
    with pytest.raises(AxBenchBpbError, match="missing direction_layer"):
        with apply_intervention(model=torch.nn.Linear(4, 4), intervention=iv):
            pass


def test_diffmean_missing_direction_coefficient_raises():
    """diffmean with direction + layer but direction_coefficient=None → AxBenchBpbError."""
    iv = Intervention(
        kind="diffmean", target_model="decoder_only",
        direction=torch.zeros(4),
        direction_layer=4,
        direction_coefficient=None,
    )
    with pytest.raises(AxBenchBpbError, match="missing direction_coefficient"):
        with apply_intervention(model=torch.nn.Linear(4, 4), intervention=iv):
            pass


def test_linear_probe_missing_direction_raises():
    """linear_probe with direction=None → AxBenchBpbError. Parallel of the
    diffmean direction-missing case in ``test_apply_intervention_contract.py``;
    confirms both kinds in the shared ``("diffmean", "linear_probe")`` branch
    exercise the guard."""
    iv = Intervention(
        kind="linear_probe", target_model="encdec",
        direction=None,
        direction_layer=4,
        direction_coefficient=1.0,
    )
    with pytest.raises(AxBenchBpbError, match="missing direction tensor"):
        with apply_intervention(model=torch.nn.Linear(4, 4), intervention=iv):
            pass


def test_linear_probe_missing_direction_layer_raises():
    """linear_probe with direction_layer=None → AxBenchBpbError."""
    iv = Intervention(
        kind="linear_probe", target_model="encdec",
        direction=torch.zeros(4),
        direction_layer=None,
        direction_coefficient=1.0,
    )
    with pytest.raises(AxBenchBpbError, match="missing direction_layer"):
        with apply_intervention(model=torch.nn.Linear(4, 4), intervention=iv):
            pass


def test_linear_probe_missing_direction_coefficient_raises():
    """linear_probe with direction_coefficient=None → AxBenchBpbError."""
    iv = Intervention(
        kind="linear_probe", target_model="encdec",
        direction=torch.zeros(4),
        direction_layer=4,
        direction_coefficient=None,
    )
    with pytest.raises(AxBenchBpbError, match="missing direction_coefficient"):
        with apply_intervention(model=torch.nn.Linear(4, 4), intervention=iv):
            pass


# ---------------------------------------------------------------------------
# lora field guards (intervention.py)
# ---------------------------------------------------------------------------


def test_lora_missing_adapter_path_raises():
    """lora kind with lora_adapter_path=None → AxBenchBpbError. A valid
    ``lora`` bundle is supplied (so the second guard isn't what fires) —
    this proves the ``adapter_path is None`` check is reachable
    independently."""
    iv = Intervention(
        kind="lora", target_model="decoder_only",
        lora_adapter_path=None,
    )
    with pytest.raises(AxBenchBpbError, match="missing lora_adapter_path"):
        with apply_intervention(
            model=torch.nn.Linear(4, 4), intervention=iv,
            lora=LoraParams(cfg=LoraConfig(rank=4, alpha=8.0), layer=4),
        ):
            pass


def test_lora_missing_lora_bundle_kwarg_raises(tmp_path):
    """lora kind with valid adapter_path but ``lora=None`` → AxBenchBpbError.

    The guard fires before any adapter is loaded from disk, so the path
    doesn't need real safetensors content. This is the single check the
    old ``lora_cfg is None or lora_layer is None`` disjunction collapsed
    into once the pair became the all-or-nothing :class:`LoraParams`
    bundle — the half-specified ``(cfg, layer)`` states the disjunction
    guarded against are now unrepresentable by construction
    (see ``unit/test_types.py``)."""
    fake_path = tmp_path / "adapter.safetensors"
    fake_path.write_bytes(b"")  # path exists; guard fires before load
    iv = Intervention.lora(adapter_path=fake_path)
    with pytest.raises(AxBenchBpbError, match=r"requires the lora kwarg"):
        with apply_intervention(
            model=torch.nn.Linear(4, 4), intervention=iv,
            lora=None,
        ):
            pass
