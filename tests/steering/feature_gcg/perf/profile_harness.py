"""Profile harness for the feature-space GCG optimization loop.

Instruments the hot regions by *wrapping* (not modifying) the production
callables.  Five **exclusive** regions are recorded:

  - ``losses``               : ``FeatureObjective.losses()`` calls.
  - ``gradients``            : ``FeatureObjective.gradients()`` exclusive of the
                               ``_extract_feature_grads`` call (i.e. the fwd+bwd).
  - ``extract_feature_grads``: the whole ``_extract_feature_grads()`` function
                               (reshapes + per-member ``torch.cat`` + the static
                               ``id_weight`` cat + ``_sanitize_grad``).
  - ``rank_candidates``      : ``FeatureConditioningEditSpace.rank_candidates()``
                               exclusive of ``_add_edits``.
  - ``_add_edits``           : ``FeatureConditioningEditSpace._add_edits()`` (vocab build).

Plus ONE informational sub-metric (NOT part of the exclusive-region sum):

  - ``id_weight_cat_only``   : just the static
                               ``id_weight = torch.cat([feat_w, *ve_weights], dim=1)``
                               line — the specific target of the #12 optimization.
                               Measured by reproducing that exact line (same encoder
                               + ve_keys) inside the patched ``_extract_feature_grads``.

The exclusive regions sum to <= total_ms (the remainder is run_gcg loop overhead:
aggregate_gradient, sample_candidates, model_item, digest, candidate eval).  The
informational ``id_weight_cat_only`` is reported separately under ``"info"`` and
is excluded from the region sum.

Device-aware timing (CRITICAL for GPU correctness)
---------------------------------------------------
CUDA kernel launches are asynchronous: ``time.perf_counter()`` around a region on
a CUDA device captures CPU *dispatch* time, not GPU *execution* time (some regions
sync via ``.item()``, but ``gradients()`` returns before the backward / grad
kernels finish).  ``_Timer`` therefore branches on the device type:

  - **CUDA**: a ``torch.cuda.Event(enable_timing=True)`` pair around the region
    (``start.record()`` / ``end.record()``), with ``torch.cuda.synchronize()``
    before reading ``start.elapsed_time(end)`` (ms).  The events are ordered on
    the default stream, so the nested exclusive math
    (``gradients_excl = gradients_total - extract_feature_grads - id_weight_cat_only``;
    ``rank_candidates_excl = rank_candidates_total - _add_edits``) holds with GPU
    times exactly as it does with wall times.
  - **CPU**: ``time.perf_counter()`` (no CUDA stream to synchronize).

The branch is selected from the ``device`` already threaded through
``profile_optimize`` (``torch.device(device).type == "cuda"``); the CUDA path is
GPU-only and validated on the real-model run, but its *selection* is unit-tested
without a real GPU (``test_timer_cuda_branch_uses_cuda_events``).

Instrumentation strategy
------------------------
- ``_TimedObjective``  wraps any ``FeatureObjective`` (delegation).  Inside
  ``gradients()``, ``_gradient_module._extract_feature_grads`` is temporarily
  patched (restored in ``try/finally``) to time the whole function plus the
  static-cat reproduction.  Production code is never permanently modified.
- ``_TimedEditSpace``  subclasses ``FeatureConditioningEditSpace`` and overrides
  ``rank_candidates`` / ``_add_edits`` to record exclusive timings.
- ``profile_optimize``  assembles the timed objects, calls ``run_gcg`` directly,
  repeats ``repeats`` times, and returns the accumulated breakdown.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch

import steering.conditional.feature_gcg.gradient as _gradient_module
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import (
    FeatureConditioningEditSpace,
    FeatureEdit,
    FeatureProposals,
)
from steering.conditional.feature_gcg.gradient import FeatureGrad
from steering.conditional.feature_gcg.objective import FeatureObjective
from steering.conditional.feature_gcg.optimize import (
    _grad_batch,
    _Group,
    _n_sae_features,
    _penalized_obj_batch,
)
from steering.conditional.feature_gcg.partition import PartitionBounds, partition_bounds_from_model
from steering.refined_types import NonEmptyNonNegativeIntTuple

_T = TypeVar("_T")

# The five exclusive region names produced by profile_optimize.
REGION_NAMES: frozenset[str] = frozenset(
    {"losses", "gradients", "extract_feature_grads", "rank_candidates", "_add_edits"}
)
# Informational sub-metric names (reported but excluded from the region sum).
INFO_NAMES: frozenset[str] = frozenset({"id_weight_cat_only"})


# ---------------------------------------------------------------------------
# Device-aware region timer
# ---------------------------------------------------------------------------


def _resolve_device_type(device: Any) -> str:
    """Return ``"cuda"`` or ``"cpu"`` (etc.) for any device spec.

    ``None`` → ``"cpu"``.  A bare int index (torch convention) → ``"cuda"``.
    Otherwise ``torch.device(device).type``.
    """
    if device is None:
        return "cpu"
    if isinstance(device, int) and not isinstance(device, bool):
        return "cuda"
    return torch.device(device).type


class _Timer:
    """Device-aware elapsed-time measurement in milliseconds.

    On a CUDA device, times via a ``torch.cuda.Event`` pair (capturing async GPU
    kernel time); on CPU, via ``time.perf_counter``.  The branch is chosen from
    ``device_type`` so the selection is unit-testable without a real GPU.
    """

    def __init__(self, device_type: str) -> None:
        self.is_cuda: bool = device_type == "cuda"

    def time(self, fn: Callable[[], _T]) -> tuple[_T, float]:
        """Run ``fn()`` and return ``(result, elapsed_ms)``."""
        if self.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            return result, float(start.elapsed_time(end))
        t0 = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Timing accumulator
# ---------------------------------------------------------------------------


class _Accumulator:
    """Single-threaded accumulator for per-region (and per-info) ms totals."""

    def __init__(self) -> None:
        self._regions: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_ms": 0.0, "calls": 0.0}
        )
        self._info: dict[str, dict[str, float]] = defaultdict(
            lambda: {"total_ms": 0.0, "calls": 0.0}
        )

    def record(self, region: str, elapsed_ms: float) -> None:
        self._regions[region]["total_ms"] += elapsed_ms
        self._regions[region]["calls"] += 1.0

    def record_info(self, name: str, elapsed_ms: float) -> None:
        self._info[name]["total_ms"] += elapsed_ms
        self._info[name]["calls"] += 1.0

    @staticmethod
    def _finish(data: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "total_ms": v["total_ms"],
                "calls": int(v["calls"]),
                "ms_per_call": v["total_ms"] / v["calls"] if v["calls"] > 0 else 0.0,
            }
            for name, v in data.items()
        }

    def regions_dict(self) -> dict[str, dict[str, Any]]:
        return self._finish(self._regions)

    def info_dict(self) -> dict[str, dict[str, Any]]:
        return self._finish(self._info)


# ---------------------------------------------------------------------------
# Timed objective wrapper (delegation — wraps any FeatureObjective)
# ---------------------------------------------------------------------------


class _TimedObjective:
    """Times losses() and gradients() of any FeatureObjective.

    For gradients(), ``_gradient_module._extract_feature_grads`` is temporarily
    patched to time (a) the whole function (the ``extract_feature_grads`` region)
    and (b) a faithful reproduction of just the static ``id_weight`` cat line (the
    informational ``id_weight_cat_only`` metric — the #12 opt target).  The
    exclusive ``gradients`` region subtracts BOTH so the harness's own
    reproduction overhead does not leak into the fwd+bwd estimate.
    """

    def __init__(self, base: FeatureObjective, accum: _Accumulator, timer: _Timer) -> None:
        self._base = base
        self._accum = accum
        self._timer = timer

    def losses(self, items: Sequence[Any]) -> list[float]:
        result, ms = self._timer.time(lambda: self._base.losses(items))
        self._accum.record("losses", ms)
        return result

    def gradients(self, items: Sequence[Any], step: int) -> list[FeatureGrad]:
        original_efg = _gradient_module._extract_feature_grads
        # Mutable holders accumulate across however many times the production
        # gradient path calls _extract_feature_grads (twice for class_margin:
        # once each for the desired + contrast completions).
        cat_total = [0.0]
        efg_total = [0.0]

        def _timed_efg(**kwargs: Any) -> list[FeatureGrad]:
            # (a) Informational: time JUST the static id_weight cat (#12 target),
            #     reproduced from the SAME encoder + ve_keys the real line uses.
            encoder = kwargs["encoder"]
            ve_keys = kwargs["ve_keys"]

            def _cat_only() -> torch.Tensor:
                feat_w = encoder.feature_embed.weight.detach()
                ve_weights = [encoder.value_embeds[k].weight.detach() for k in ve_keys]
                return torch.cat([feat_w, *ve_weights], dim=1)

            _, cat_ms = self._timer.time(_cat_only)
            self._accum.record_info("id_weight_cat_only", cat_ms)
            cat_total[0] += cat_ms

            # (b) The whole _extract_feature_grads function.
            result, efg_ms = self._timer.time(lambda: original_efg(**kwargs))
            self._accum.record("extract_feature_grads", efg_ms)
            efg_total[0] += efg_ms
            return result

        # The wrapper uses **kwargs rather than the production function's explicit
        # keyword-only signature, so expose the intentional test-double boundary.
        gradient_module: Any = _gradient_module
        gradient_module._extract_feature_grads = _timed_efg
        try:
            result, total_ms = self._timer.time(lambda: self._base.gradients(items, step))
        finally:
            gradient_module._extract_feature_grads = original_efg

        # Exclusive gradients = total minus the extract function AND minus the
        # harness's own cat-reproduction overhead (the outer timer saw both).
        self._accum.record("gradients", max(0.0, total_ms - efg_total[0] - cat_total[0]))
        return result


# ---------------------------------------------------------------------------
# Timed edit-space subclass
# ---------------------------------------------------------------------------


class _TimedEditSpace(FeatureConditioningEditSpace):
    """Subclasses FeatureConditioningEditSpace to record exclusive timings.

    - _add_edits      : timed directly (vocab build inner loop).
    - rank_candidates : timed exclusive of _add_edits (gradient matmul + sorting).
    """

    def __init__(
        self,
        *,
        bounds: PartitionBounds,
        allowed_feature_ids: NonEmptyNonNegativeIntTuple,
        warm_start: FeatureConditioning,
        accum: _Accumulator,
        timer: _Timer,
    ) -> None:
        super().__init__(
            bounds=bounds,
            allowed_feature_ids=allowed_feature_ids,
            warm_start=warm_start,
        )
        self._accum = accum
        self._timer = timer
        # _add_edits time within the current rank_candidates call (reset on entry,
        # accumulated in case _add_edits is ever called more than once).
        self._last_add_edits_ms: float = 0.0

    def _add_edits(
        self,
        grad: FeatureGrad,
        current: FeatureConditioning,
        top_k: int,
        magnitude: bool,
        allowed_excluded: Sequence[int],
        n_pos: int,
        n_neg: int,
    ) -> list[FeatureEdit]:
        bound = super()._add_edits  # zero-arg super() resolves here (method body)
        result, ms = self._timer.time(
            lambda: bound(grad, current, top_k, magnitude, allowed_excluded, n_pos, n_neg)
        )
        self._accum.record("_add_edits", ms)
        self._last_add_edits_ms += ms
        return result

    def rank_candidates(
        self, agg_grad: Any, current: FeatureConditioning, cfg: FeatureGcgConfig
    ) -> FeatureProposals:
        bound = super().rank_candidates
        self._last_add_edits_ms = 0.0
        result, total_ms = self._timer.time(lambda: bound(agg_grad, current, cfg))
        # Exclusive: rank_candidates total minus _add_edits time.
        self._accum.record("rank_candidates", max(0.0, total_ms - self._last_add_edits_ms))
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def profile_optimize(
    model: Any,
    members: Sequence[Any],
    warm_start: FeatureConditioning,
    config: FeatureGcgConfig,
    *,
    objective_factory: Callable[[Any], FeatureObjective],
    device: Any = None,
    repeats: int = 3,
) -> dict[str, Any]:
    """Profile the GCG optimize loop and return a per-region timing breakdown.

    Runs ``run_gcg`` (the inner loop of ``optimize_feature_conditioning_universal``)
    with device-aware timed wrappers around the hot regions, ``repeats`` times,
    and accumulates the timing across all runs.

    Parameters
    ----------
    model:
        Loaded encoder-decoder model with ``_loaded_meta_data``,
        ``encoder.feature_embed.weight``, and ``encoder.config.num_buckets``.
    members:
        Optimization members (``FeatureGradMember`` for lm_loss, or
        ``ClassMarginMember`` for class_margin).
    warm_start:
        Initial ``FeatureConditioning`` bag; all repeats start from this bag.
    config:
        GCG hyperparameters (``max_steps``, ``minibatch_size``, etc.).
    objective_factory:
        ``callable(model) -> FeatureObjective``; e.g. ``lm_loss_on_masked_tokens``
        or ``class_margin_objective``.
    device:
        Target device for model forward passes (e.g. ``"cpu"``, ``"cuda:0"``).
        Selects the timer backend (CUDA events vs perf_counter).
    repeats:
        Number of complete optimize runs to accumulate timing over.

    Returns
    -------
    dict with structure::

        {
            "regions": {
                "losses":                {"total_ms", "calls", "ms_per_call"},
                "gradients":             {...},   # fwd+bwd (excl. extract)
                "extract_feature_grads": {...},   # the whole _extract_feature_grads
                "rank_candidates":       {...},   # excl. _add_edits
                "_add_edits":            {...},
            },
            "info": {
                "id_weight_cat_only":    {...},   # the #12-target static cat ONLY
            },
            "n_members": int,
            "n_steps":   int,    # config.max_steps * repeats
            "total_ms":  float,  # wall-clock time across all repeats
        }

    The five regions are *exclusive*:
    ``gradients + extract_feature_grads`` <= the gradients() inclusive time;
    ``rank_candidates + _add_edits`` == rank_candidates() inclusive time.
    Therefore ``sum(regions[*].total_ms) <= total_ms``.  ``id_weight_cat_only`` is
    informational only and is NOT part of that sum.
    """
    accum = _Accumulator()
    timer = _Timer(_resolve_device_type(device))
    member_list = list(members)
    n_members = len(member_list)

    wall_t0 = time.perf_counter()

    for _ in range(repeats):
        bounds = partition_bounds_from_model(model, config)
        base_obj = objective_factory(model)
        timed_obj = _TimedObjective(base_obj, accum, timer)

        n_sae = _n_sae_features(model)
        timed_es = _TimedEditSpace(
            bounds=bounds,
            allowed_feature_ids=NonEmptyNonNegativeIntTuple.parse(
                tuple(range(n_sae))
            ),
            warm_start=warm_start,
            accum=accum,
            timer=timer,
        )

        group = _Group(key_str="profile_run", members=member_list)
        run_gcg(
            edit_space=timed_es,
            groups=[group],
            objective_batch=_penalized_obj_batch(timed_obj, config.reg_lambda),
            gradient_provider_batch=_grad_batch(timed_obj),
            cfg=config,
        )

    total_ms = (time.perf_counter() - wall_t0) * 1000.0

    return {
        "regions": accum.regions_dict(),
        "info": accum.info_dict(),
        "n_members": n_members,
        "n_steps": config.max_steps * repeats,
        "total_ms": total_ms,
    }
