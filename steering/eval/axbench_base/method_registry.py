"""33-method registry for AxBench-Base Stage 4 generation (Task 30).

The AxBench-Base benchmark evaluates 33 (baseline, target_model) combinations:
16 in positive mode (push neutral text TOWARD a concept) and 17 in negative
mode (push concept-rich text AWAY from a concept), including the no-steering
``control`` baseline for both targets. Each entry is a
``Method`` dataclass that pairs a baseline type with a target model and the
coefficient grid factory + partition modifier the orchestrator should
dispatch through at generation time.

Why a dataclass-list registry instead of a class hierarchy:

  * Stage 4 dispatch is data-driven — generation, judging, and
    aggregation all read the registry as a flat list and never
    instantiate or subclass anything.
  * ``cfg.enabled_methods`` / ``cfg.disabled_methods`` are pure-string
    filters; resolving them to ``Method`` instances is a one-liner over
    a list comprehension.
  * Tests can pin counts (14 positive, 15 negative, 3 with
    ``requires_negative_sampling=True``) without depending on any
    runtime model state.

Method enumeration:

  The registry is built from:

    - The 3 positive-mode partition modifiers in
      ``encoder_input._MODIFIER_REGISTRY`` (``ours_extract_and_add``,
      ``ours_replace_random``, ``ours_replace_all``).
    - The 4 dedicated negative-mode partition modifiers
      (``neg_ours_remove``, ``neg_ours_add_to_negatives``,
      ``neg_ours_replace_random_negatives``,
      ``neg_ours_replace_all_negatives``).
    - The 4 feature-GCG Phase-2 ``ours`` methods (2 positive + 2
      negative): ``ours_gcg_encdec``, ``ours_gcg_roofline_encdec``,
      ``neg_ours_gcg_encdec``, ``neg_ours_gcg_roofline_encdec``.
    - The 5 baseline types (``diffmean``, ``linear_probe``, ``lora``,
      ``prompting``, ``control`` — the no-steering base model) crossed
      with the supported target models.
    - The hard constraint that ``ours`` baselines need an encoder
      (encdec only) and ``lora`` only writes to the
      ``baselines/lora/decoder_only/`` path
      (``AxBenchBasePaths.lora_adapter`` is hardcoded to ``decoder_only``).

  Counts:

    Positive (16) =
      diffmean (2 targets) + linear_probe (2) + lora (1) +
      prompting (2) + control (2) + ours (3 partition modifiers +
      2 feature-GCG + 2 feature-DiffMean, encdec only) = 16.

    Negative (17) =
      diffmean (2) + linear_probe (2) + lora (1) + prompting (2) +
      control (2) + ours (4 dedicated negative modifiers +
      2 feature-GCG + 2 feature-DiffMean) = 17.

  The 3 methods with ``requires_negative_sampling=True`` are the ones
  that mutate the encoder's "inactive" partition:
  ``neg_ours_add_to_negatives_encdec``,
  ``neg_ours_replace_random_negatives_encdec``,
  ``neg_ours_replace_all_negatives_encdec``. These need an existing
  inactive partition (or an inactive sampler) supplied by a target
  model that was trained with negative-feature sampling.

  ``method_registry`` tests pin these counts + category invariants and
  the exact 33-method survivor set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from steering.eval.axbench_base.coefficient_grids import (
    active_bucket_grid,
    diffmean_scalar_grid,
    ours_gcg_alpha_grid,
    ours_gcg_lambda_grid,
    single_condition_grid,
)
from steering.eval.axbench_base.encoder_input import PartitionModifier
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
    TargetModelMetadata,
)


class BaselineKind(StrEnum):
    """Which intervention family a :class:`Method` comes from.

    A :class:`StrEnum` so values retain their stable string representation
    across path, parquet, and JSON boundaries while internal dispatch compares
    against named members.
    ``BaselineKind`` lives in this module (not ``types.py``) because
    the enumeration is registry-specific: adding a new baseline kind
    means editing the registry anyway.

    - :attr:`DIFFMEAN`: diff-of-means scalar direction in residual.
    - :attr:`LINEAR_PROBE`: logistic-regression scalar direction.
    - :attr:`LORA`: per-concept LoRA adapter.
    - :attr:`PROMPTING`: text prefix prepended at generation time.
    - :attr:`OURS`: encoder-decoder partition-modifier methods.
    - :attr:`CONTROL`: NO steering — the unmodified base model (decoder) / the
      encoder-decoder with no concept conditioning (encdec). The no-steering
      reference baseline.
    """

    DIFFMEAN = "diffmean"
    LINEAR_PROBE = "linear_probe"
    LORA = "lora"
    PROMPTING = "prompting"
    OURS = "ours"
    CONTROL = "control"


# Type alias for the coefficient grid functions defined in coefficient_grids.py.
# Each grid factory takes target metadata + config and returns a list of
# coefficient values to sweep at generation time.
CoefficientGridFn = Callable[
    [TargetModelMetadata, object],
    list[float] | list[None] | list[int],
]


@dataclass(frozen=True)
class Method:
    """One entry in the AxBench-Base method matrix.

    Fields:
        name: Stable string identifier used by ``cfg.enabled_methods`` /
            ``cfg.disabled_methods``, generation cache keys, and
            judge / leaderboard records. MUST be unique across the
            registry.
        mode: :class:`SteeringMode` direction
            (``SteeringMode.POSITIVE`` — push toward concept — or
            ``SteeringMode.NEGATIVE`` — push away). Serialises to the
            equivalent string in parquet / JSON via :class:`StrEnum`'s
            :class:`str` subclassing.
        baseline: :class:`BaselineKind` member naming the intervention
            family this method comes from.
        target_model: Which target model to apply the method to.
            ``"decoder_only"`` (base GPT) or ``"encdec"``
            (encoder-decoder GPT). ``ours`` methods are always
            ``"encdec"``; ``lora`` methods are always
            ``"decoder_only"``.
        requires_negative_sampling: True for methods whose partition
            modifier mutates the encoder's "inactive" partition. These
            methods require a target model trained with negative-feature
            sampling so the orchestrator's ``inactive_sampler`` has
            something meaningful to draw from. False for everything else.
        coefficient_grid_fn: One of the 3 grid factories from
            ``coefficient_grids``: ``diffmean_scalar_grid`` (10-value
            scalar sweep for direction-vector methods),
            ``single_condition_grid`` (single ``[None]`` for methods
            that don't take a coefficient), or ``active_bucket_grid``
            (sweep over the encoder's active buckets for SAE-feature
            methods).
        partition_modifier_name: :class:`PartitionModifier` member that
            selects the partition modifier for this method, or
            ``None`` for baseline methods that don't take a
            partition-based encoder input. The Stage 4 orchestrator
            looks up the modifier by this field, NOT by ``name``, so a
            single modifier can be shared across methods that differ
            only in their ``name`` and ``mode``.
    """

    name: str
    mode: SteeringMode
    baseline: BaselineKind
    target_model: str
    requires_negative_sampling: bool
    coefficient_grid_fn: CoefficientGridFn
    partition_modifier_name: PartitionModifier | None = field(default=None)


# ---------------------------------------------------------------------------
# Registry construction helpers
# ---------------------------------------------------------------------------


def _baseline_methods() -> list[Method]:
    """Return the 18 (baseline, target, mode) combinations that don't
    use a partition modifier.

    Layout (positive then negative, deterministic order):

        positive: diffmean_decoder_only, diffmean_encdec,
                  linear_probe_decoder_only, linear_probe_encdec,
                  lora_decoder_only,
                  prompting_decoder_only, prompting_encdec,
                  control_decoder_only, control_encdec
        negative: same 9 names with a ``neg_`` prefix.

    18 = 9 positive baselines + 9 negative baselines.
    """
    out: list[Method] = []
    for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
        prefix = "" if mode == SteeringMode.POSITIVE else "neg_"
        # diffmean: scalar direction, both targets, scalar grid.
        for target in ("decoder_only", "encdec"):
            out.append(
                Method(
                    name=f"{prefix}diffmean_{target}",
                    mode=mode,
                    baseline=BaselineKind.DIFFMEAN,
                    target_model=target,
                    requires_negative_sampling=False,
                    coefficient_grid_fn=diffmean_scalar_grid,
                    partition_modifier_name=None,
                ),
            )
        # linear_probe: scalar direction, both targets, scalar grid.
        for target in ("decoder_only", "encdec"):
            out.append(
                Method(
                    name=f"{prefix}linear_probe_{target}",
                    mode=mode,
                    baseline=BaselineKind.LINEAR_PROBE,
                    target_model=target,
                    requires_negative_sampling=False,
                    coefficient_grid_fn=diffmean_scalar_grid,
                    partition_modifier_name=None,
                ),
            )
        # lora: decoder_only ONLY (AxBenchBasePaths.lora_adapter is hardcoded).
        out.append(
            Method(
                name=f"{prefix}lora_decoder_only",
                mode=mode,
                baseline=BaselineKind.LORA,
                target_model="decoder_only",
                requires_negative_sampling=False,
                coefficient_grid_fn=single_condition_grid,
                partition_modifier_name=None,
            ),
        )
        # prompting: applied at prompt-build time, both targets, no coefficient.
        for target in ("decoder_only", "encdec"):
            out.append(
                Method(
                    name=f"{prefix}prompting_{target}",
                    mode=mode,
                    baseline=BaselineKind.PROMPTING,
                    target_model=target,
                    requires_negative_sampling=False,
                    coefficient_grid_fn=single_condition_grid,
                    partition_modifier_name=None,
                ),
            )
        # control: NO steering, both targets, no coefficient. The no-steering
        # reference (decoder: unmodified base model; encdec: prompt-conditioned
        # with no concept injection). Used as a baseline + additive's pure control.
        for target in ("decoder_only", "encdec"):
            out.append(
                Method(
                    name=f"{prefix}control_{target}",
                    mode=mode,
                    baseline=BaselineKind.CONTROL,
                    target_model=target,
                    requires_negative_sampling=False,
                    coefficient_grid_fn=single_condition_grid,
                    partition_modifier_name=None,
                ),
            )
    return out


def _ours_methods() -> list[Method]:
    """Return the 17 ``ours`` methods: 8 positive + 9 negative.

    Positive (8): 3 existing partition-modifier methods + 3 feature-GCG
    Phase-2 methods (fair, fair-alpha, roofline) + fair and roofline
    feature-DiffMean conditioning-bag baselines.

    Negative (9): 4 existing methods + 3 feature-GCG Phase-2 methods
    + fair and roofline negative feature-DiffMean conditioning-bag baselines.
    Of the 9
    negative methods, exactly 3 mutate the inactive partition and therefore
    need negative sampling at generation time (the GCG Phase-2 methods do not
    manipulate the inactive partition; their placeholder modifiers raise before
    any partition mutation).

    The 4 ``ours_gcg*`` fair/roofline entries use ``ours_gcg_lambda_grid`` as
    their coefficient grid (sweeps ``OursGcgConfig.reg_lambda_grid``); the 2
    ``ours_gcg_alpha*`` entries read the SAME fair bags but sweep
    ``objective_alpha_grid`` instead, so alpha is their strength axis.  Their
    partition modifier fns are placeholders that raise ``FeatureGcgError``
    until Task 7 provides the real loader.
    """
    positive_specs: list[tuple[str, PartitionModifier, CoefficientGridFn]] = [
        # (name_without_encdec_suffix, modifier, grid_fn)
        # The method name is exactly the name here — no additional suffix is
        # appended (unlike the existing ours_*_encdec pattern where suffix is
        # "extract_and_add" etc. and the _encdec is appended by the loop).
        # For GCG methods the modifier key already includes _encdec so the method
        # name equals the modifier StrEnum value.
        (
            "ours_extract_and_add_encdec",
            PartitionModifier.OURS_EXTRACT_AND_ADD,
            active_bucket_grid,
        ),
        (
            "ours_replace_random_encdec",
            PartitionModifier.OURS_REPLACE_RANDOM,
            active_bucket_grid,
        ),
        (
            "ours_replace_all_encdec",
            PartitionModifier.OURS_REPLACE_ALL,
            active_bucket_grid,
        ),
        # feature-GCG Phase-2 (Task 1 registers; Task 7 provides real loader)
        (
            "ours_gcg_encdec",
            PartitionModifier.OURS_GCG_ENCDEC,
            ours_gcg_lambda_grid,
        ),
        # Same fair bags, alpha as the strength axis instead of lambda.
        (
            "ours_gcg_alpha_encdec",
            PartitionModifier.OURS_GCG_ALPHA_ENCDEC,
            ours_gcg_alpha_grid,
        ),
        (
            "ours_gcg_roofline_encdec",
            PartitionModifier.OURS_GCG_ROOFLINE_ENCDEC,
            ours_gcg_lambda_grid,
        ),
        (
            "feature_diffmean_encdec",
            PartitionModifier.FEATURE_DIFFMEAN_ENCDEC,
            single_condition_grid,
        ),
        (
            "feature_diffmean_roofline_encdec",
            PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
            single_condition_grid,
        ),
    ]
    negative_specs: list[tuple[str, PartitionModifier, CoefficientGridFn, bool]] = [
        # (method_name, modifier, grid_fn, requires_negative_sampling)
        (
            "neg_ours_remove_encdec",
            PartitionModifier.NEG_OURS_REMOVE,
            single_condition_grid,
            False,
        ),
        (
            "neg_ours_add_to_negatives_encdec",
            PartitionModifier.NEG_OURS_ADD_TO_NEGATIVES,
            single_condition_grid,
            True,
        ),
        (
            "neg_ours_replace_random_negatives_encdec",
            PartitionModifier.NEG_OURS_REPLACE_RANDOM_NEGATIVES,
            single_condition_grid,
            True,
        ),
        (
            "neg_ours_replace_all_negatives_encdec",
            PartitionModifier.NEG_OURS_REPLACE_ALL_NEGATIVES,
            single_condition_grid,
            True,
        ),
        # feature-GCG Phase-2 (Task 1 registers; Task 7 provides real loader)
        (
            "neg_ours_gcg_encdec",
            PartitionModifier.NEG_OURS_GCG_ENCDEC,
            ours_gcg_lambda_grid,
            False,
        ),
        # Same fair bags, alpha as the strength axis instead of lambda.
        (
            "neg_ours_gcg_alpha_encdec",
            PartitionModifier.NEG_OURS_GCG_ALPHA_ENCDEC,
            ours_gcg_alpha_grid,
            # Like its lambda sibling: the GCG methods never mutate the inactive
            # partition, so no negative sampling is needed at generation time.
            False,
        ),
        (
            "neg_ours_gcg_roofline_encdec",
            PartitionModifier.NEG_OURS_GCG_ROOFLINE_ENCDEC,
            ours_gcg_lambda_grid,
            False,
        ),
        (
            "neg_feature_diffmean_encdec",
            PartitionModifier.NEG_FEATURE_DIFFMEAN_ENCDEC,
            single_condition_grid,
            False,
        ),
        (
            "neg_feature_diffmean_roofline_encdec",
            PartitionModifier.NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
            single_condition_grid,
            False,
        ),
    ]
    out: list[Method] = []
    for name, modifier, grid_fn in positive_specs:
        out.append(
            Method(
                name=name,
                mode=SteeringMode.POSITIVE,
                baseline=BaselineKind.OURS,
                target_model="encdec",
                requires_negative_sampling=False,
                coefficient_grid_fn=grid_fn,
                partition_modifier_name=modifier,
            ),
        )
    for name, modifier, grid_fn, needs_neg in negative_specs:
        out.append(
            Method(
                name=name,
                mode=SteeringMode.NEGATIVE,
                baseline=BaselineKind.OURS,
                target_model="encdec",
                requires_negative_sampling=needs_neg,
                coefficient_grid_fn=grid_fn,
                partition_modifier_name=modifier,
            ),
        )
    return out


def _build_all_methods() -> list[Method]:
    """Build the 33-entry registry: 18 baseline methods + 15 ours methods.

    Order is: all positive-mode methods first (16), then all negative-mode
    methods (17). Within each mode, baselines come before ``ours``.

    The 8 precomputed-bag ``ours`` entries are the fair/roofline
    feature-GCG methods plus the fair/roofline Feature-DiffMean methods,
    in both positive and negative mode.
    """
    baselines = _baseline_methods()
    ours = _ours_methods()
    positive = [m for m in baselines if m.mode == SteeringMode.POSITIVE] + [
        m for m in ours if m.mode == SteeringMode.POSITIVE
    ]
    negative = [m for m in baselines if m.mode == SteeringMode.NEGATIVE] + [
        m for m in ours if m.mode == SteeringMode.NEGATIVE
    ]
    return positive + negative


ALL_METHODS: list[Method] = _build_all_methods()


# ---------------------------------------------------------------------------
# Lookup + filter helpers (top-level functions, not class methods)
# ---------------------------------------------------------------------------


def get_method_by_name(name: str) -> Method:
    """Return the unique ``Method`` whose ``name`` matches ``name``.

    Raises ``AxBenchBaseError`` if no method has the given name. The
    registry is small enough (33 entries) that a linear scan is the
    simplest correct implementation; downstream code that needs a
    fast O(1) lookup can build its own dict from ``ALL_METHODS``.
    """
    for m in ALL_METHODS:
        if m.name == name:
            return m
    raise AxBenchBaseError(
        f"unknown method {name!r}. "
        f"Available method names: {sorted(meth.name for meth in ALL_METHODS)}",
    )


def filter_methods(
    all_methods: list[Method],
    *,
    enabled_methods: list[str] | None,
    disabled_methods: list[str],
    target_model_has_negative_sampling: bool,
) -> list[Method]:
    """Apply Stage 4's three method filters in order.

    1. If ``enabled_methods`` is not ``None``, keep only methods whose
       ``name`` appears in that list. An empty list means "keep nothing"
       (explicit narrowing); ``None`` means "no enable filter at all".
    2. Drop any method whose name appears in ``disabled_methods``.
    3. If ``target_model_has_negative_sampling`` is ``False``, drop any
       method with ``requires_negative_sampling=True`` (the orchestrator
       has no inactive partition to feed those modifiers).

    Returns a new list; the input list is never mutated. Method order
    in the input list is preserved.
    """
    enabled_set = set(enabled_methods) if enabled_methods is not None else None
    disabled_set = set(disabled_methods)
    known_names = {m.name for m in all_methods}
    unknown_disabled = sorted(disabled_set - known_names)
    if unknown_disabled:
        raise AxBenchBaseError(
            f"unknown disabled method(s): {unknown_disabled}. "
            f"Available method names: {sorted(known_names)}",
        )
    out: list[Method] = []
    for m in all_methods:
        if enabled_set is not None and m.name not in enabled_set:
            continue
        if m.name in disabled_set:
            continue
        if m.requires_negative_sampling and not target_model_has_negative_sampling:
            continue
        out.append(m)
    return out
