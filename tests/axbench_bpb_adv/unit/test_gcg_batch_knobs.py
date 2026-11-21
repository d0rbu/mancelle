"""Teeth for the feature-GCG batching/split knobs.

Pins four things the review found untested or unenforced:

1. **Bounds** — every knob rejects out-of-range values at construction time
   (refined types for ``OursGcgConfig`` / ``FeatureGcgConfig``),
   including the silent-empty-scored-set maker ``split_ratio >= 1``.
2. **Single-sourced defaults** — the ``8``/``16`` defaults live in ONE constant
   each (``DEFAULT_MAX_FORWARD_BATCH`` / ``DEFAULT_MEAN_G_STEER_BATCH``);
   every config dataclass and signature default reads it, so tuning the
   constant can never leave a copy silently behind.
3. **Chunking arithmetic** — ``mean_g_steer`` genuinely slices its members into
   ``batch_size`` chunks (a non-default value REACHES the loop) and forwards
   ``max_forward_batch`` into the objective construction.
4. **Loud rejects** — ``mean_g_steer(batch_size < 1)`` raises.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.feature_gcg.config import (
    DEFAULT_MAX_FORWARD_BATCH,
    DEFAULT_MAX_GRAD_BATCH,
    DEFAULT_OBJECTIVE_ALPHA,
    FeatureGcgConfig,
)
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.objective import ClassMarginMember
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv import gcg_bags
from steering.eval.axbench_bpb_adv.config import (
    DEFAULT_MEAN_G_STEER_BATCH,
    OursGcgConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_bags import mean_g_steer

# ─────────────────────────────────────────────────────────────────────────────
# 1) Bounds
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [0, -1])
def test_ours_mean_g_steer_batch_bounds(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="mean_g_steer_batch"):
        OursGcgConfig(mean_g_steer_batch=value)


@pytest.mark.parametrize("value", [0, -1])
def test_ours_max_forward_batch_bounds(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_forward_batch"):
        OursGcgConfig(max_forward_batch=value)


@pytest.mark.parametrize("value", [1.0, 0.0, 1.5, -0.5])
def test_split_ratio_must_be_strictly_interior(value: float) -> None:
    """ratio >= 1 makes the held-out (the fair SCORED SET) empty; ratio <= 0
    makes the optimize split empty — both were previously accepted and produced
    silent-zero-results runs."""
    with pytest.raises(BeartypeCallHintParamViolation, match="split_ratio"):
        OursGcgConfig(split_ratio=value)


@pytest.mark.parametrize("value", [0.01, 0.5, 0.99])
def test_split_ratio_interior_values_pass(value: float) -> None:
    assert OursGcgConfig(split_ratio=value).split_ratio == value


@pytest.mark.parametrize("value", [0, -1])
def test_feature_gcg_max_forward_batch_bounds(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_forward_batch"):
        FeatureGcgConfig(max_forward_batch=value)


@pytest.mark.parametrize("value", [0, -1])
def test_ours_max_grad_batch_bounds(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_grad_batch"):
        OursGcgConfig(max_grad_batch=value)


@pytest.mark.parametrize("value", [0, -1])
def test_feature_gcg_max_grad_batch_bounds(value: int) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_grad_batch"):
        FeatureGcgConfig(max_grad_batch=value)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Single-sourced defaults
# ─────────────────────────────────────────────────────────────────────────────


def test_max_forward_batch_default_is_single_sourced() -> None:
    """Every home of the default reads DEFAULT_MAX_FORWARD_BATCH.

    Before this fix the literal ``8`` lived in three places (gradient constant,
    FeatureGcgConfig, OursGcgConfig) plus a private import tunnel; tuning one
    silently left the others behind.
    """
    assert FeatureGcgConfig().max_forward_batch == DEFAULT_MAX_FORWARD_BATCH
    assert OursGcgConfig().max_forward_batch == DEFAULT_MAX_FORWARD_BATCH
    sig = inspect.signature(mean_g_steer)
    assert sig.parameters["max_forward_batch"].default == DEFAULT_MAX_FORWARD_BATCH


def test_max_grad_batch_default_is_single_sourced() -> None:
    """Both homes of the gradient-path default read DEFAULT_MAX_GRAD_BATCH.

    Same failure mode as its forward-path sibling: a literal copied into the
    second config lets one be tuned while the other silently stays behind.
    """
    assert FeatureGcgConfig().max_grad_batch == DEFAULT_MAX_GRAD_BATCH
    assert OursGcgConfig().max_grad_batch == DEFAULT_MAX_GRAD_BATCH
    # It is the GRADIENT path's bound, so it must not silently equal the
    # forward path's: they bound different-sized allocations.
    assert DEFAULT_MAX_GRAD_BATCH <= DEFAULT_MAX_FORWARD_BATCH


def test_mean_g_steer_batch_default_is_single_sourced() -> None:
    assert OursGcgConfig().mean_g_steer_batch == DEFAULT_MEAN_G_STEER_BATCH
    sig = inspect.signature(mean_g_steer)
    assert sig.parameters["batch_size"].default == DEFAULT_MEAN_G_STEER_BATCH


# ─────────────────────────────────────────────────────────────────────────────
# 3) Chunking arithmetic + objective forwarding (non-default values REACH the loop)
# ─────────────────────────────────────────────────────────────────────────────


class _ChunkRecordingObjective:
    """Objective stub recording each losses() chunk size."""

    def __init__(self, chunk_sizes: list[int]) -> None:
        self._chunk_sizes = chunk_sizes

    def losses(self, items: list[Any]) -> list[float]:
        self._chunk_sizes.append(len(items))
        return [0.5] * len(items)


def test_mean_g_steer_chunks_by_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """batch_size=3 over 7 members → losses() sees chunks [3, 3, 1].

    This is the finding-8 teeth: a refactor that drops the ``batch_size``
    threading (reverting to the constant 16) would see ONE chunk of 7 here.
    """
    chunk_sizes: list[int] = []
    seen_mfb: list[int] = []

    def _fake_objective(
        model: Any, *, mode: Any, device: Any, max_forward_batch: int
    ):
        seen_mfb.append(max_forward_batch)
        return _ChunkRecordingObjective(chunk_sizes)

    monkeypatch.setattr(gcg_bags, "class_margin_objective", _fake_objective)
    members = cast(list[ClassMarginMember], [object() for _ in range(7)])
    result = mean_g_steer(
        object(),
        members,
        FeatureConditioning(entries=()),
        mode=SteeringMode.POSITIVE,
        device=t.device("cpu"),
        batch_size=3,
        max_forward_batch=5,
    )
    assert chunk_sizes == [3, 3, 1]
    assert seen_mfb == [5]  # the knob reaches the objective construction
    assert result == pytest.approx(0.5)


def test_mean_g_steer_rejects_non_positive_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gcg_bags,
        "class_margin_objective",
        lambda *a, **k: _ChunkRecordingObjective([]),
    )
    with pytest.raises(AxBenchBpbAdvError, match="batch_size"):
        mean_g_steer(
            object(),
            cast(list[ClassMarginMember], [object()]),
            FeatureConditioning(entries=()),
            mode=SteeringMode.POSITIVE,
            device=t.device("cpu"),
            batch_size=0,
        )


def test_objective_factory_forwards_the_knob_and_late_binds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_objective_factory forwards (mode, device, max_forward_batch) verbatim and
    resolves ``class_margin_objective`` from the module global at CALL time (the
    end-to-end sign-trap monkeypatches it).

    #1245 added ``max_grad_batch``; it is forwarded through the same seam, so a
    distinct value (7 vs 5) catches the two being crossed or one dropped. The
    class-margin ``objective_alpha`` rides the same seam and is given a third
    distinct value (0.25) for the same reason -- it is a SEARCH input, so a
    dropped or crossed alpha would optimize a different objective silently."""
    calls: list[dict[str, Any]] = []

    def _recorder(
        model: Any,
        *,
        mode: Any,
        device: Any,
        max_forward_batch: int,
        max_grad_batch: int,
        objective_alpha: float,
    ):
        calls.append(
            {
                "model": model,
                "mode": mode,
                "device": device,
                "mfb": max_forward_batch,
                "mgb": max_grad_batch,
                "alpha": objective_alpha,
            }
        )
        return "objective-sentinel"

    device = t.device("cpu")
    factory = gcg_bags._objective_factory(SteeringMode.NEGATIVE, device, 5, 7, 0.25)
    # Patch AFTER building the factory: late binding means the patch still wins.
    monkeypatch.setattr(gcg_bags, "class_margin_objective", _recorder)
    assert factory("model-sentinel") == "objective-sentinel"
    assert calls == [
        {
            "model": "model-sentinel",
            "mode": SteeringMode.NEGATIVE,
            "device": device,
            "mfb": 5,
            "mgb": 7,
            "alpha": 0.25,
        }
    ]


def test_objective_factory_defaults_alpha_to_the_plain_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting alpha forwards the plain-margin default, not None or 0.

    Teeth: a factory that forwarded ``None`` would crash deep in the objective,
    and one that forwarded ``0.0`` would silently drop the unsteered class from
    every derive that never asked for an alpha sweep.
    """
    seen: list[float] = []

    def _recorder(model: Any, **kwargs: Any):
        seen.append(kwargs["objective_alpha"])
        return "objective-sentinel"

    factory = gcg_bags._objective_factory(SteeringMode.POSITIVE, t.device("cpu"), 5, 7)
    monkeypatch.setattr(gcg_bags, "class_margin_objective", _recorder)
    factory("model-sentinel")
    assert seen == [DEFAULT_OBJECTIVE_ALPHA]
