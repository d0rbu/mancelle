"""Explicit, checkpointed policy for uncovered percentile-table rows.

``build_percentile_table`` leaves an all-zero row for every SAE feature the
statistics artifact never saw, and ``magnitude_to_percentile`` refuses to query
one.  That refusal is correct — the same lookup feeds feature-prediction
SUPERVISION targets, so a substituted percentile does not merely rank an unseen
feature conservatively, it teaches the model a specific activation level for a
feature whose activation is simply unknown.

The refusal is also blunt: it fires at the first offending id, mid-run, after
minutes of setup, and tells the operator nothing about how big the hole is.
This module supplies the two things that were missing:

* an UP-FRONT coverage check (:func:`percentile_table_coverage`,
  :func:`check_percentile_coverage`, :func:`enforce_percentile_coverage`) that
  reports the complete distinct uncovered population per table before the first
  batch is built, and
* an EXPLICIT policy (:class:`MissingPercentilePolicy`) recorded in the SFT
  checkpoint, so a pilot run that deliberately tolerates holes can never have
  its ``val/bpb`` silently compared against a production number.

The startup check does NOT replace ``magnitude_to_percentile``'s own raise.
That primitive stays fail-loud: the bitmap is an earlier, more informative
guard, and the primitive is the backstop for anything that slips past it.

Policy semantics
----------------

``error`` (the production default)
    Any uncovered id that will be queried aborts at startup with the complete
    count.  Nothing is substituted anywhere, so every percentile-derived value
    is byte-identical to a run without this module.

``lowest`` (an explicitly non-comparable pilot mode)
    An uncovered id is pinned to the BOTTOM of every statistics-derived
    quantity, using the values the percentile primitive itself already emits
    for a below-p1 magnitude:

    * percentile rank -> ``0.5`` (:data:`LOWEST_PERCENTILE_RANK`)
    * normalized rank / feature-prediction target -> ``0.005``
      (:data:`LOWEST_NORMALIZED_RANK`)
    * magnitude bucket -> ``0`` (:data:`LOWEST_MAGNITUDE_BUCKET`)
    * per-feature percentile thresholds -> ``+inf``
      (:meth:`MissingPercentileResolver.mask_precomputed_thresholds`), so an
      uncovered id ranks below any cutoff above the floor and is not selected
      by a threshold selector.

    One contract, every consumer: unclustered threshold selectors, both
    supported clustered percentile selectors, feature-prediction targets,
    weighted sampling, top-k sampling, feature masks, compact FP layouts and
    binary (single-boundary) bucket spaces all read the same pinned values.
    Weighted sampling can still draw an uncovered id — its weight ``0.5`` is
    positive — which is exactly why the bucket pin exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch as t
from beartype import beartype
from loguru import logger

from steering.conditional.percentile_table import magnitude_to_percentile

# The value ``magnitude_to_percentile`` itself returns for a below-p1 magnitude.
LOWEST_PERCENTILE_RANK = 0.5
# The same value after the ``/100`` normalisation every downstream consumer
# applies before it becomes a sampling weight or a feature-prediction target.
LOWEST_NORMALIZED_RANK = 0.005
# The lowest magnitude bucket in every bucket space, including the degenerate
# single-boundary ("binary") one.
LOWEST_MAGNITUDE_BUCKET = 0

MISSING_PERCENTILE_POLICY_META_KEY = "missing_percentile_policy"

# Number of uncovered ids named in an error/log line. The COUNT is always
# complete; these are only examples to start an investigation from.
_MAX_REPORTED_EXAMPLES = 10


class MissingPercentileError(ValueError):
    """An uncovered percentile row will be queried under the ``error`` policy."""


class MissingPercentilePolicy(Enum):
    """What to do about a feature the percentile statistics never covered."""

    ERROR = "error"
    LOWEST = "lowest"


@beartype
def parse_missing_percentile_policy(value: Any) -> MissingPercentilePolicy:
    """Parse a config value into a policy, naming every accepted spelling."""
    if isinstance(value, MissingPercentilePolicy):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"missing_percentile_policy must be a string, got {value!r}. "
            f"Accepted: {[member.value for member in MissingPercentilePolicy]}."
        )
    normalized = value.strip().lower()
    for member in MissingPercentilePolicy:
        if member.value == normalized:
            return member
    raise ValueError(
        f"Unknown missing_percentile_policy {value!r}. Accepted: "
        f"{[member.value for member in MissingPercentilePolicy]}. "
        "'error' is the production default and aborts at startup when the "
        "percentile statistics do not cover every queried feature; 'lowest' is "
        "an explicitly non-comparable pilot mode."
    )


@beartype
def percentile_table_coverage(percentile_table: t.Tensor) -> t.Tensor:
    """``bool[n_features]`` — ``True`` where the row is NOT the missing row.

    This predicate is deliberately the SAME one
    ``magnitude_to_percentile`` raises on (``row[0] == 0 and row[98] == 0``
    under its monotonicity precondition).  If the two ever drift, the startup
    bitmap would clear ids the primitive then rejects mid-run, which is the
    exact failure this module exists to remove; ``test_missing_percentile.py``
    pins them together.
    """
    if percentile_table.ndim != 2 or percentile_table.shape[1] != 99:
        raise ValueError(
            "percentile_table_coverage requires a [n_features, 99] table, got "
            f"shape {tuple(percentile_table.shape)}"
        )
    first_vals = percentile_table[:, 0]
    last_vals = percentile_table[:, 98]
    return ~((first_vals == 0) & (last_vals == 0))


@beartype
@dataclass(frozen=True)
class PercentileCoverageReport:
    """One table's coverage verdict against one required-id set."""

    table_name: str
    required_set_name: str
    n_required: int
    n_uncovered: int
    examples: tuple[int, ...]

    def describe(self) -> str:
        if self.n_uncovered == 0:
            return (
                f"{self.table_name} covers all {self.n_required} ids queried "
                f"as {self.required_set_name}"
            )
        return (
            f"{self.table_name} is missing percentile statistics for "
            f"{self.n_uncovered} of the {self.n_required} distinct feature ids "
            f"queried as {self.required_set_name} (examples: "
            f"{list(self.examples)})"
        )


@beartype
def check_percentile_coverage(
    *,
    percentile_table: t.Tensor,
    required_ids: t.Tensor,
    table_name: str,
    required_set_name: str,
) -> PercentileCoverageReport:
    """Compare one table's coverage bitmap against one required-id set.

    ``required_ids`` is the set of RAW ids that this particular table will be
    queried with.  Different tables are queried with different sets — the
    feature-prediction table with the pre-selection ids of every segment, the
    sampling table with those ids after feature-mask filtering — so each is
    checked against its own set rather than a shared approximation.
    """
    coverage = percentile_table_coverage(percentile_table)
    distinct = t.unique(required_ids.reshape(-1).long())
    if distinct.numel() and int(distinct.max().item()) >= coverage.shape[0]:
        out_of_range = distinct[distinct >= coverage.shape[0]]
        raise MissingPercentileError(
            f"{required_set_name} contains feature id(s) "
            f"{out_of_range[:_MAX_REPORTED_EXAMPLES].tolist()} outside "
            f"{table_name}'s {coverage.shape[0]} rows; the percentile table was "
            "built for a different SAE width."
        )
    uncovered = distinct[~coverage[distinct]] if distinct.numel() else distinct
    return PercentileCoverageReport(
        table_name=table_name,
        required_set_name=required_set_name,
        n_required=int(distinct.numel()),
        n_uncovered=int(uncovered.numel()),
        examples=tuple(uncovered[:_MAX_REPORTED_EXAMPLES].tolist()),
    )


@beartype
def enforce_percentile_coverage(
    reports: Sequence[PercentileCoverageReport],
    policy: MissingPercentilePolicy,
) -> None:
    """Abort (``error``) or log the non-comparability warning (``lowest``).

    Every report is summarised, not just the first failing one: the operator's
    decision — rebuild statistics, shrink the mixture, or accept a pilot — needs
    the whole picture, and the pre-#1311 behaviour of dying on whichever id
    happened to be looked up first gave them none of it.
    """
    failing = [report for report in reports if report.n_uncovered > 0]
    if not failing:
        logger.info(
            "Percentile coverage OK ("
            + "; ".join(report.describe() for report in reports)
            + f"); missing_percentile_policy={policy.value}"
        )
        return
    detail = "; ".join(report.describe() for report in failing)
    if policy is MissingPercentilePolicy.ERROR:
        raise MissingPercentileError(
            f"Percentile statistics do not cover every queried feature: {detail}. "
            "Percentiles are looked up on the RAW pre-selection feature ids of "
            "each segment — before feature-mask filtering and before cluster "
            "consolidation — so the statistics artifact must cover every ACTIVE "
            "feature in the queried data, not only the allowed representatives. "
            "Rebuild statistics over the full active feature set, or set "
            "missing_percentile_policy=lowest to run an explicitly "
            "NON-COMPARABLE pilot in which uncovered features are pinned to the "
            "bottom of every statistics-derived quantity."
        )
    logger.warning(
        f"missing_percentile_policy=lowest: {detail}. Those features are pinned "
        f"to rank {LOWEST_PERCENTILE_RANK} / normalized "
        f"{LOWEST_NORMALIZED_RANK} / bucket {LOWEST_MAGNITUDE_BUCKET}. This run "
        "is a PILOT: its val/bpb is NOT comparable to a run trained under "
        "missing_percentile_policy=error, and the policy is recorded in every "
        "checkpoint so a resume cannot silently mix the two."
    )


@beartype
@dataclass(frozen=True)
class MissingPercentileResolver:
    """The single runtime contract every percentile consumer reads.

    ``covered`` is the conjunction of every active table's coverage bitmap: a
    feature is treated as known only when EVERY table that will be queried
    covers it, so the clustered and unclustered paths cannot disagree about the
    same id.  It is required for ``lowest`` and unused by ``error`` (which
    substitutes nothing at all and therefore leaves every emitted tensor
    byte-identical to a run without this module).
    """

    policy: MissingPercentilePolicy
    covered: t.Tensor | None = None

    def __post_init__(self) -> None:
        if self.policy is MissingPercentilePolicy.LOWEST and self.covered is None:
            raise ValueError(
                "MissingPercentileResolver(policy=lowest) requires the coverage "
                "bitmap; without it there is nothing to pin."
            )
        if self.covered is not None and self.covered.dtype is not t.bool:
            raise ValueError(
                "MissingPercentileResolver.covered must be a bool bitmap, got "
                f"dtype {self.covered.dtype}"
            )

    @classmethod
    def error(cls) -> MissingPercentileResolver:
        """The production default: substitute nothing, let the primitive raise."""
        return cls(policy=MissingPercentilePolicy.ERROR)

    @property
    def substitutes(self) -> bool:
        return self.policy is MissingPercentilePolicy.LOWEST

    def uncovered_mask(self, feature_ids: t.Tensor) -> t.Tensor:
        """``bool[N]`` — ``True`` where the id has no percentile statistics."""
        if not self.substitutes or feature_ids.numel() == 0:
            return t.zeros(feature_ids.shape, dtype=t.bool, device=feature_ids.device)
        assert self.covered is not None  # enforced in __post_init__
        return ~self.covered.to(feature_ids.device)[feature_ids.long()]

    def ranks(
        self,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
    ) -> t.Tensor:
        """Percentile ranks in ``[0.5, 99.5]``, pinning uncovered ids to the floor.

        Under ``error`` this is exactly ``magnitude_to_percentile`` — same call,
        same raise, no wrapper semantics of its own.
        """
        if not self.substitutes:
            return magnitude_to_percentile(feature_ids, feature_mags, percentile_table)
        uncovered = self.uncovered_mask(feature_ids)
        if not bool(uncovered.any()):
            return magnitude_to_percentile(feature_ids, feature_mags, percentile_table)
        result = t.full(
            feature_ids.shape,
            LOWEST_PERCENTILE_RANK,
            dtype=t.float64,
            device=feature_ids.device,
        )
        known = ~uncovered
        if bool(known.any()):
            result[known] = magnitude_to_percentile(
                feature_ids[known], feature_mags[known], percentile_table
            )
        return result

    def pin_normalized(
        self, feature_ids: t.Tensor, percentiles: t.Tensor
    ) -> t.Tensor:
        """Pin uncovered ids in an already-normalized ``[0, 1]`` percentile array.

        Used by the clustered no-sampling path, whose percentiles come from
        ``compute_percentiles`` over the BOUNDARIES tensor rather than the
        percentile table.  An uncovered feature's boundaries row is all-zero
        too, so without this pin a positive magnitude would interpolate to the
        TOP of the range — the opposite of the contract.
        """
        if not self.substitutes or percentiles.numel() == 0:
            return percentiles
        uncovered = self.uncovered_mask(feature_ids)
        if not bool(uncovered.any()):
            return percentiles
        return t.where(
            uncovered,
            t.full_like(percentiles, LOWEST_NORMALIZED_RANK),
            percentiles,
        )

    def pin_buckets(self, feature_ids: t.Tensor, buckets: t.Tensor) -> t.Tensor:
        """Pin uncovered ids to bucket ``0``.

        Only the UNCLUSTERED path needs this: ``discretize_activations``
        searchsorts a positive magnitude against an all-zero boundaries row and
        returns the TOP bucket.  The clustered path derives its buckets from
        percentiles that ``pin_normalized`` / :meth:`ranks` already pinned, and
        its ids are cluster representatives rather than the queried ids, so
        pinning there would misattribute a covered sibling's percentile.
        """
        if not self.substitutes or buckets.numel() == 0:
            return buckets
        uncovered = self.uncovered_mask(feature_ids)
        if not bool(uncovered.any()):
            return buckets
        return t.where(uncovered, t.full_like(buckets, LOWEST_MAGNITUDE_BUCKET), buckets)

    def mask_precomputed_thresholds(self, thresholds: t.Tensor) -> t.Tensor:
        """Raise uncovered per-feature thresholds to ``+inf``.

        ``build_boundaries_from_stats`` leaves an uncovered feature's threshold
        at ``0.0``, which ADMITS every positive magnitude — the opposite of
        rank ``0.5``.  Lifting it to ``+inf`` makes the unclustered
        ``feature_percentile`` selectors agree with the clustered
        percentile-space ones, where ``0.005`` already fails any cutoff above
        the floor.
        """
        if not self.substitutes:
            return thresholds
        assert self.covered is not None  # enforced in __post_init__
        covered = self.covered.to(thresholds.device)
        if covered.shape[0] != thresholds.shape[0]:
            raise ValueError(
                "MissingPercentileResolver.mask_precomputed_thresholds got "
                f"{thresholds.shape[0]} thresholds for a {covered.shape[0]}-row "
                "coverage bitmap"
            )
        return t.where(covered, thresholds, t.full_like(thresholds, float("inf")))


@beartype
def combined_coverage(bitmaps: Iterable[t.Tensor]) -> t.Tensor:
    """Conjunction of every active table's coverage bitmap."""
    combined: t.Tensor | None = None
    for bitmap in bitmaps:
        combined = bitmap.clone() if combined is None else (combined & bitmap)
    if combined is None:
        raise ValueError("combined_coverage requires at least one coverage bitmap")
    return combined


@beartype
def assert_resume_missing_percentile_policy_matches(
    configured_policy: MissingPercentilePolicy,
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Verify missing-percentile policy identity before resuming an SFT run.

    Checkpoints written before the key existed omit it and are interpreted as
    ``error`` — the only behaviour that existed then — exactly as the loss-mask
    and feature-mask resume guards treat their own historical omissions.  The
    point of pinning it is that a ``lowest`` pilot's ``val/bpb`` must never be
    silently continued as, or compared with, a production number.
    """
    checkpoint_raw = (
        MissingPercentilePolicy.ERROR.value
        if checkpoint_meta is None
        else checkpoint_meta.get(
            MISSING_PERCENTILE_POLICY_META_KEY, MissingPercentilePolicy.ERROR.value
        )
    )
    checkpoint_policy = parse_missing_percentile_policy(checkpoint_raw)
    if checkpoint_policy is configured_policy:
        return
    raise MissingPercentileError(
        f"missing_percentile_policy mismatch on resume from {checkpoint_dir}: "
        f"checkpoint records {checkpoint_policy.value!r}, configured "
        f"{configured_policy.value!r}. The policy changes what every "
        "percentile-derived value means, so min_val_bpb and val/bpb from the "
        "two are not comparable. Resume with the original policy or start a "
        "fresh SFT run."
    )
