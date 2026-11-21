"""Coefficient grid factories for Stage 4 intervention methods (spec §4.3).

Each factory returns a ``list`` suitable for the outer coefficient sweep
in the generation stage. Different methods expect different grid types:

- Scalar-direction methods (DiffMean, LinearProbe) use a fixed multiplier
  sweep over a single direction vector.
- Single-condition methods (LoRA, Prompting) don't take a coefficient at
  all — the grid is ``[None]`` so the generation loop still runs once.
- SAE-feature methods (ours_canonical, etc.) sweep over the ACTIVE
  buckets of the target model. The set of active buckets depends on
  the model's ``num_buckets`` config and whether negative-feature
  sampling is enabled (if so, the last bucket is reserved for inactive).
"""

from __future__ import annotations

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_base.types import TargetModelMetadata

_DIFFMEAN_SCALAR_GRID: tuple[float, ...] = (
    0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)


def diffmean_scalar_grid(
    target_model_meta: TargetModelMetadata, cfg: object,
) -> list[float]:
    """Default 10-value scalar **factor** grid for DiffMean/LinearProbe steering.

    The factor grid is independent of the target model metadata — the same
    factors are swept across all models. Effective steering is
    ``factor * direction_artifact.max_activation * unit_direction`` (#1228).
    Stage 3 defines that artifact field as the maximum signed ``h_token · d``
    for unit ``d`` over continuation tokens from both fit classes at the target
    hook layer (float32 CPU; no ReLU/absolute value). Application seams fail
    loudly on missing, invalid, negative, zero-inconsistent, or wrongly bound
    metadata. There is no feature-statistics or constant fallback for the SCALE;
    this factory only supplies the comparable factor axis.

    ``cfg.steering_coefficient_grid`` (duck-typed, same pattern as
    :func:`ours_gcg_lambda_grid`) restricts that axis to an explicit subset.
    It exists for adversarial runs whose per-coefficient cost is a full attack
    optimization: the robustness plot keeps ONE representative coefficient per
    family (``_select_representative_coefficient``) and discards the rest, so
    sweeping all ten multiplies attack cost ~10x for output that is thrown away.
    Restricting to a superset of the representatives leaves the selected
    coefficients — and therefore the figure — unchanged.

    ``None``/absent means the full default grid, so no existing run moves.
    """
    del target_model_meta
    override = getattr(cfg, "steering_coefficient_grid", None)
    if override is not None:
        return [float(value) for value in override]
    return list(_DIFFMEAN_SCALAR_GRID)


def single_condition_grid(
    target_model_meta: TargetModelMetadata, cfg: object,
) -> list[None]:
    """Single-condition grid: runs the method exactly once with no coefficient."""
    del target_model_meta, cfg
    return [None]


def active_bucket_grid(
    target_model_meta: TargetModelMetadata, cfg: object,
) -> list[int]:
    """Active-bucket grid for SAE-feature intervention methods.

    Walks 0, 1, ..., n_active-1 where:
    - n_active = num_buckets when negative sampling is disabled
    - n_active = num_buckets - 1 when negative sampling is enabled
      (the last bucket is reserved for inactive features)

    For decoder-only targets (``num_buckets is None``), returns ``[0]``
    as a defensive default. Callers SHOULD NOT ask for this grid on
    decoder-only targets — it's dispatched per-method.
    """
    del cfg
    num_buckets = target_model_meta.num_buckets
    if num_buckets is None:
        return [0]
    has_negative = target_model_meta.negative_selection is not None
    n_active = (num_buckets - 1) if has_negative else num_buckets
    return list(range(n_active))


def ours_gcg_lambda_grid(
    target_model_meta: TargetModelMetadata, cfg: object,
) -> list[float]:
    """Regularisation-lambda grid for the feature-GCG BPB-Adv methods.

    Reads ``cfg.ours_gcg.reg_lambda_grid`` via duck-typing so the function
    works both when ``cfg`` is an ``AxBenchBpbAdvConfig`` (which carries an
    ``OursGcgConfig`` at ``cfg.ours_gcg``) and when it is an
    ``AxBenchBaseConfig`` (which does not — fallback returns ``[0.0]``).

    Duck-typing is intentional: no config attribute is common to every
    coefficient-grid factory, so ``Method.coefficient_grid_fn`` accepts
    ``object`` and each factory narrows only what it consumes. This factory's
    optional ``ours_gcg`` capability is present on ``AxBenchBpbAdvConfig`` and
    absent on ``AxBenchBaseConfig``. A direct import of the BPB-Adv config here
    would create a circular dependency (``axbench_bpb_adv.config`` already
    imports ``method_registry`` which imports this module).
    """
    del target_model_meta
    ours_gcg = getattr(cfg, "ours_gcg", None)
    if ours_gcg is None:
        return [0.0]
    return list(ours_gcg.reg_lambda_grid)


def ours_gcg_alpha_grid(
    target_model_meta: TargetModelMetadata, cfg: object,
) -> list[float]:
    """Class-margin alpha grid for the ``ours_gcg_alpha*`` BPB-Adv methods.

    The sibling of :func:`ours_gcg_lambda_grid`, duck-typed for the same reason
    and with the same ``AxBenchBaseConfig`` fallback. It is a SEPARATE factory
    rather than a parameter of the lambda one because the two grids are
    different axes of the same derive: a method binds to exactly one of them,
    and the coefficient a method reports must be resolvable back to the axis
    that produced its bag (see ``ALPHA_AXIS_GCG_METHODS``).

    The fallback is ``DEFAULT_OBJECTIVE_ALPHA``, not ``0.0``: alpha = 0 is a
    meaningful and very different objective (ignore the unsteered class
    entirely), so defaulting to it would silently redefine the method rather
    than degrade to the plain margin.
    """
    del target_model_meta
    ours_gcg = getattr(cfg, "ours_gcg", None)
    if ours_gcg is None:
        return [DEFAULT_OBJECTIVE_ALPHA]
    return list(ours_gcg.objective_alpha_grid)
