"""Sparse action policies and objective math for structured-SFT GRPO.

This module is model-agnostic. It keeps behavior traces sparse and exposes the
same scoring functions to no-grad collection and differentiable replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch as t
import torch.nn.functional as F


class SFTGRPOError(ValueError):
    """An action, reward, or replay contract is invalid."""


class ActionMode(StrEnum):
    """Mutually exclusive structured-SFT feature policies."""

    WEIGHTED_WITHOUT_REPLACEMENT = "weighted_without_replacement"
    BERNOULLI_TOPK_BOTTOMK = "bernoulli_topk_bottomk"


@dataclass(frozen=True)
class SparseActionTrace:
    """One executed live decision in compact feature-head row space."""

    mode: ActionMode
    requested_positive: int
    requested_negative: int
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    behavior_log_terms: tuple[float, ...]

    @property
    def selected_count(self) -> int:
        return len(self.positive_rows) + len(self.negative_rows)


@dataclass(frozen=True)
class ConditioningAction:
    """Original-SAE IDs and fixed two-bucket magnitudes at the encoder boundary."""

    feature_ids: tuple[int, ...]
    magnitudes: tuple[int, ...]
    mask: tuple[bool, ...]


def _validate_logits(logits: t.Tensor) -> t.Tensor:
    if logits.ndim != 1 or logits.numel() == 0:
        raise SFTGRPOError(
            f"action logits must have shape (V,) with V > 0, got {tuple(logits.shape)}"
        )
    logits_f32 = logits.float()
    if not bool(t.isfinite(logits_f32).all()):
        raise SFTGRPOError("action logits must all be finite")
    return logits_f32


def _forward_validated_logits(logits: t.Tensor) -> t.Tensor:
    """Reuse the whole-forward finite guard without another device sync."""
    if logits.ndim != 1 or logits.numel() == 0:
        raise SFTGRPOError(
            f"action logits must have shape (V,) with V > 0, got {tuple(logits.shape)}"
        )
    return logits.float()


def _weighted_log_weights_from_finite_forward(logits: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Transform one or more finite forward rows into stable FP32 log weights."""
    if logits.ndim not in {1, 2} or logits.shape[-1] == 0:
        raise SFTGRPOError(
            f"weighted forward logits must have shape (V,) or (B,V), got {tuple(logits.shape)}"
        )
    logits_f32 = logits.float()
    return F.logsigmoid(logits_f32), F.logsigmoid(-logits_f32)


def _validate_counts(feature_count: int, positive_count: int, negative_count: int) -> None:
    for name, value in (
        ("positive_count", positive_count),
        ("negative_count", negative_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SFTGRPOError(f"{name} must be a non-negative integer")
    if positive_count + negative_count > feature_count:
        raise SFTGRPOError(
            "requested positive+negative count exceeds effective feature vocabulary: "
            f"{positive_count}+{negative_count}>{feature_count}"
        )


def validate_count_support(
    *,
    positive_min: int,
    positive_max_exclusive: int,
    negative_min: int,
    negative_max_exclusive: int,
    feature_count: int,
) -> None:
    """Fail when geometric supports can request zero total or overflow ``V``."""
    for name, lower, upper in (
        ("positive", positive_min, positive_max_exclusive),
        ("negative", negative_min, negative_max_exclusive),
    ):
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (lower, upper)):
            raise SFTGRPOError(f"{name} count bounds must be integers")
        if lower < 0 or upper <= lower:
            raise SFTGRPOError(
                f"{name} count support must satisfy 0 <= min < max_exclusive"
            )
    if positive_min + negative_min == 0:
        raise SFTGRPOError(
            "count configuration can deliberately request zero total features"
        )
    maximum_total = (positive_max_exclusive - 1) + (negative_max_exclusive - 1)
    if maximum_total > feature_count:
        raise SFTGRPOError(
            f"combined count support {maximum_total} exceeds feature vocabulary {feature_count}"
        )


def _draw_log_weighted_row(
    log_weights: t.Tensor,
    available_log_weights: t.Tensor,
    *,
    generator: t.Generator,
) -> tuple[t.Tensor, t.Tensor]:
    denominator = t.logsumexp(available_log_weights, dim=0)
    probabilities = t.exp(available_log_weights - denominator)
    selected = t.multinomial(probabilities, 1, generator=generator).squeeze(0)
    return selected, log_weights[selected] - denominator


@t.no_grad()
def sample_weighted_without_replacement(
    logits: t.Tensor,
    *,
    positive_count: int,
    negative_count: int,
    generator: t.Generator,
) -> SparseActionTrace:
    """Sample ordered positive then negative rows from stable FP32 log weights.

    Selected rows stay as device scalars while later draws are enqueued.  The
    ordered rows and FP32 log terms cross to the host only after the complete
    action has been sampled, avoiding one CUDA synchronization per draw while
    preserving the exact multinomial sequence and Python summation order.
    """
    return _sample_weighted_without_replacement(
        _validate_logits(logits),
        positive_count=positive_count,
        negative_count=negative_count,
        generator=generator,
    )


@t.no_grad()
def _sample_weighted_without_replacement_from_finite_forward(
    logits: t.Tensor,
    *,
    positive_count: int,
    negative_count: int,
    generator: t.Generator,
) -> SparseActionTrace:
    """Sample logits already covered by the whole-forward finite guard."""
    logits_f32 = _forward_validated_logits(logits)
    positive_log_weights, negative_log_weights = _weighted_log_weights_from_finite_forward(
        logits_f32
    )
    return _sample_weighted_without_replacement_from_log_weights(
        positive_log_weights,
        negative_log_weights,
        positive_count=positive_count,
        negative_count=negative_count,
        generator=generator,
    )


def _sample_weighted_without_replacement(
    logits_f32: t.Tensor,
    *,
    positive_count: int,
    negative_count: int,
    generator: t.Generator,
) -> SparseActionTrace:
    positive_log_weights, negative_log_weights = _weighted_log_weights_from_finite_forward(
        logits_f32
    )
    return _sample_weighted_without_replacement_from_log_weights(
        positive_log_weights,
        negative_log_weights,
        positive_count=positive_count,
        negative_count=negative_count,
        generator=generator,
    )


def _sample_weighted_without_replacement_from_log_weights(
    positive_log_weights: t.Tensor,
    negative_log_weights: t.Tensor,
    *,
    positive_count: int,
    negative_count: int,
    generator: t.Generator,
) -> SparseActionTrace:
    if (
        positive_log_weights.ndim != 1
        or negative_log_weights.shape != positive_log_weights.shape
        or positive_log_weights.dtype != t.float32
        or negative_log_weights.dtype != t.float32
    ):
        raise SFTGRPOError("weighted log weights must be aligned FP32 vectors")
    feature_count = positive_log_weights.numel()
    _validate_counts(feature_count, positive_count, negative_count)
    available_positive_log_weights = positive_log_weights.clone()
    available_negative_log_weights = negative_log_weights.clone()
    positive_row_tensors = []
    negative_row_tensors = []
    log_term_tensors = []
    for _ in range(positive_count):
        row, term = _draw_log_weighted_row(
            positive_log_weights,
            available_positive_log_weights,
            generator=generator,
        )
        positive_row_tensors.append(row)
        log_term_tensors.append(term)
        available_positive_log_weights[row] = -t.inf
        available_negative_log_weights[row] = -t.inf
    for _ in range(negative_count):
        row, term = _draw_log_weighted_row(
            negative_log_weights,
            available_negative_log_weights,
            generator=generator,
        )
        negative_row_tensors.append(row)
        log_term_tensors.append(term)
        available_negative_log_weights[row] = -t.inf
    positive_rows = (
        tuple(int(row) for row in t.stack(positive_row_tensors).tolist())
        if positive_row_tensors
        else ()
    )
    negative_rows = (
        tuple(int(row) for row in t.stack(negative_row_tensors).tolist())
        if negative_row_tensors
        else ()
    )
    log_terms = (
        tuple(float(term) for term in t.stack(log_term_tensors).tolist())
        if log_term_tensors
        else ()
    )
    if not all(math.isfinite(term) for term in log_terms):
        raise SFTGRPOError("weighted action has no available row")
    return SparseActionTrace(
        mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
        requested_positive=positive_count,
        requested_negative=negative_count,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        behavior_log_terms=(sum(log_terms),),
    )


def _sample_weighted_without_replacement_batch_from_log_weights(
    positive_log_weights: t.Tensor,
    negative_log_weights: t.Tensor,
    *,
    positive_counts: tuple[int, ...],
    negative_counts: tuple[int, ...],
    generator: t.Generator,
) -> tuple[SparseActionTrace, ...]:
    """Sample a microbatch with one RNG stream and one multinomial per draw.

    This is action-policy version 2. It preserves each row's ordered weighted
    without-replacement distribution, but deliberately changes the mapping
    from stateless seeds to samples so CUDA draws can be batched.
    """
    if (
        positive_log_weights.ndim != 2
        or positive_log_weights.shape[0] == 0
        or positive_log_weights.shape[1] == 0
        or negative_log_weights.shape != positive_log_weights.shape
        or positive_log_weights.dtype != t.float32
        or negative_log_weights.dtype != t.float32
        or len(positive_counts) != positive_log_weights.shape[0]
        or len(negative_counts) != positive_log_weights.shape[0]
    ):
        raise SFTGRPOError(
            "batched weighted log weights and count rows must be aligned FP32 tensors"
        )
    batch_size, feature_count = positive_log_weights.shape
    for positive_count, negative_count in zip(
        positive_counts,
        negative_counts,
        strict=True,
    ):
        _validate_counts(feature_count, positive_count, negative_count)

    available = t.ones_like(positive_log_weights, dtype=t.bool)
    batch_rows = t.arange(batch_size, device=positive_log_weights.device)
    positive_counts_device = t.tensor(
        positive_counts,
        dtype=t.int64,
        device=positive_log_weights.device,
    )
    negative_counts_device = t.tensor(
        negative_counts,
        dtype=t.int64,
        device=positive_log_weights.device,
    )

    def draw_rows(
        log_weights: t.Tensor,
        counts: tuple[int, ...],
        counts_device: t.Tensor,
    ) -> tuple[list[list[int]], list[list[float]]]:
        selected_by_draw: list[t.Tensor] = []
        terms_by_draw: list[t.Tensor] = []
        for draw in range(max(counts, default=0)):
            active = counts_device > draw
            available_log_weights = log_weights.masked_fill(~available, -t.inf)
            # A row can exhaust the vocabulary before another row finishes.
            # Inactive rows still participate in the batched multinomial only
            # to keep one launch per draw, so give them an arbitrary finite
            # distribution without changing their availability state.
            available_log_weights = t.where(
                active.unsqueeze(1),
                available_log_weights,
                t.zeros_like(available_log_weights),
            )
            denominator = t.logsumexp(available_log_weights, dim=1)
            probabilities = t.exp(available_log_weights - denominator.unsqueeze(1))
            selected = t.multinomial(
                probabilities,
                1,
                generator=generator,
            ).squeeze(1)
            terms = log_weights[batch_rows, selected] - denominator
            selected_by_draw.append(selected)
            terms_by_draw.append(terms)
            selected_was_available = available[batch_rows, selected]
            available[batch_rows, selected] = selected_was_available & ~active
        selected_host = (
            t.stack(selected_by_draw).tolist() if selected_by_draw else []
        )
        terms_host = t.stack(terms_by_draw).tolist() if terms_by_draw else []
        return selected_host, terms_host

    positive_rows_by_draw, positive_terms_by_draw = draw_rows(
        positive_log_weights,
        positive_counts,
        positive_counts_device,
    )
    negative_rows_by_draw, negative_terms_by_draw = draw_rows(
        negative_log_weights,
        negative_counts,
        negative_counts_device,
    )

    traces = []
    for row, (positive_count, negative_count) in enumerate(
        zip(positive_counts, negative_counts, strict=True)
    ):
        positive_rows = tuple(
            int(positive_rows_by_draw[draw][row]) for draw in range(positive_count)
        )
        negative_rows = tuple(
            int(negative_rows_by_draw[draw][row]) for draw in range(negative_count)
        )
        log_terms = (
            *(float(positive_terms_by_draw[draw][row]) for draw in range(positive_count)),
            *(float(negative_terms_by_draw[draw][row]) for draw in range(negative_count)),
        )
        if not all(math.isfinite(term) for term in log_terms):
            raise SFTGRPOError("weighted action has no available row")
        traces.append(
            SparseActionTrace(
                mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
                requested_positive=positive_count,
                requested_negative=negative_count,
                positive_rows=positive_rows,
                negative_rows=negative_rows,
                behavior_log_terms=(sum(log_terms),),
            )
        )
    return tuple(traces)


def weighted_ordered_log_likelihood(
    logits: t.Tensor,
    trace: SparseActionTrace,
) -> t.Tensor:
    """Score the exact ordered weighted-without-replacement likelihood."""
    if trace.mode is not ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        raise SFTGRPOError("weighted likelihood requires a weighted action trace")
    return _weighted_ordered_log_likelihood(_validate_logits(logits), trace)


def _weighted_ordered_log_likelihood_from_finite_forward(
    logits: t.Tensor,
    trace: SparseActionTrace,
) -> t.Tensor:
    """Score logits already covered by the whole-forward finite guard."""
    logits_f32 = _forward_validated_logits(logits)
    positive_log_weights, negative_log_weights = _weighted_log_weights_from_finite_forward(
        logits_f32
    )
    return _weighted_ordered_log_likelihood_from_log_weights(
        logits_f32,
        positive_log_weights,
        negative_log_weights,
        trace,
    )


def _weighted_ordered_log_likelihood(
    logits_f32: t.Tensor,
    trace: SparseActionTrace,
) -> t.Tensor:
    positive_log_weights, negative_log_weights = _weighted_log_weights_from_finite_forward(
        logits_f32
    )
    return _weighted_ordered_log_likelihood_from_log_weights(
        logits_f32,
        positive_log_weights,
        negative_log_weights,
        trace,
    )


def _weighted_ordered_log_likelihood_from_log_weights(
    logits_f32: t.Tensor,
    positive_log_weights: t.Tensor,
    negative_log_weights: t.Tensor,
    trace: SparseActionTrace,
) -> t.Tensor:
    if (
        logits_f32.ndim != 1
        or positive_log_weights.shape != logits_f32.shape
        or negative_log_weights.shape != logits_f32.shape
        or logits_f32.dtype != t.float32
        or positive_log_weights.dtype != t.float32
        or negative_log_weights.dtype != t.float32
    ):
        raise SFTGRPOError("weighted logits and log weights must be aligned FP32 vectors")
    _validate_weighted_trace(logits_f32.numel(), trace)
    available = t.ones(logits_f32.numel(), dtype=t.bool, device=logits_f32.device)
    total = logits_f32.sum() * 0.0
    for row in trace.positive_rows:
        total = total + positive_log_weights[row] - t.logsumexp(
            positive_log_weights.masked_fill(~available, -t.inf),
            dim=0,
        )
        available[row] = False
    for row in trace.negative_rows:
        total = total + negative_log_weights[row] - t.logsumexp(
            negative_log_weights.masked_fill(~available, -t.inf),
            dim=0,
        )
        available[row] = False
    return total


def _validate_weighted_trace(feature_count: int, trace: SparseActionTrace) -> None:
    _validate_counts(feature_count, len(trace.positive_rows), len(trace.negative_rows))
    seen_rows: set[int] = set()
    for row in trace.positive_rows:
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < feature_count
            or row in seen_rows
        ):
            raise SFTGRPOError("weighted positive trace contains an invalid or duplicate row")
        seen_rows.add(row)
    for row in trace.negative_rows:
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < feature_count
            or row in seen_rows
        ):
            raise SFTGRPOError(
                "weighted negative trace contains an invalid, duplicate, or positive row"
            )
        seen_rows.add(row)


def _weighted_ordered_log_likelihood_batch_from_log_weights(
    logits_f32: t.Tensor,
    positive_log_weights: t.Tensor,
    negative_log_weights: t.Tensor,
    traces: tuple[SparseActionTrace, ...],
) -> t.Tensor:
    if (
        logits_f32.ndim != 2
        or logits_f32.shape[0] == 0
        or logits_f32.shape[1] == 0
        or positive_log_weights.shape != logits_f32.shape
        or negative_log_weights.shape != logits_f32.shape
        or logits_f32.dtype != t.float32
        or positive_log_weights.dtype != t.float32
        or negative_log_weights.dtype != t.float32
        or len(traces) != logits_f32.shape[0]
    ):
        raise SFTGRPOError(
            "weighted batch logits, log weights, and traces must have aligned FP32 rows"
        )
    feature_count = logits_f32.shape[1]
    for trace in traces:
        _validate_weighted_trace(feature_count, trace)

    batch_size = logits_f32.shape[0]
    totals = logits_f32.sum(dim=1) * 0.0

    def padded_rows(
        rows_by_trace: tuple[tuple[int, ...], ...],
    ) -> tuple[t.Tensor, t.Tensor]:
        maximum = max((len(rows) for rows in rows_by_trace), default=0)
        if maximum == 0:
            return (
                t.empty((batch_size, 0), dtype=t.int64, device=logits_f32.device),
                t.empty((batch_size, 0), dtype=t.bool, device=logits_f32.device),
            )
        padded = [(*rows, *([0] * (maximum - len(rows)))) for rows in rows_by_trace]
        active = [
            (*([True] * len(rows)), *([False] * (maximum - len(rows))))
            for rows in rows_by_trace
        ]
        return (
            t.tensor(padded, dtype=t.int64, device=logits_f32.device),
            t.tensor(active, dtype=t.bool, device=logits_f32.device),
        )

    def selected_membership(rows: t.Tensor, active: t.Tensor) -> t.Tensor:
        membership = t.zeros_like(logits_f32, dtype=t.int64)
        if rows.shape[1]:
            membership.scatter_add_(1, rows, active.to(dtype=t.int64))
        return membership.bool()

    def phase_terms(
        log_weights: t.Tensor,
        rows: t.Tensor,
        active: t.Tensor,
        excluded: t.Tensor,
    ) -> t.Tensor:
        if rows.shape[1] == 0:
            return logits_f32.sum(dim=1) * 0.0
        selected = log_weights.gather(1, rows)
        selected_or_negative_inf = selected.masked_fill(~active, -t.inf)
        suffix_selected_total = t.flip(
            t.logcumsumexp(t.flip(selected_or_negative_inf, dims=(1,)), dim=1),
            dims=(1,),
        )
        never_selected_total = t.logsumexp(
            log_weights.masked_fill(excluded, -t.inf),
            dim=1,
        )
        denominator = t.logaddexp(
            never_selected_total.unsqueeze(1),
            suffix_selected_total,
        )
        safe_selected = t.where(active, selected, t.zeros_like(selected))
        safe_denominator = t.where(active, denominator, t.zeros_like(denominator))
        return t.where(
            active,
            safe_selected - safe_denominator,
            t.zeros_like(selected),
        ).sum(dim=1)

    positive_rows, positive_active = padded_rows(
        tuple(trace.positive_rows for trace in traces)
    )
    negative_rows, negative_active = padded_rows(
        tuple(trace.negative_rows for trace in traces)
    )
    positive_membership = selected_membership(positive_rows, positive_active)
    negative_membership = selected_membership(negative_rows, negative_active)
    totals = totals + phase_terms(
        positive_log_weights,
        positive_rows,
        positive_active,
        positive_membership,
    )
    totals = totals + phase_terms(
        negative_log_weights,
        negative_rows,
        negative_active,
        positive_membership | negative_membership,
    )
    return totals


def _tie_stable_eligible_order(
    logits: t.Tensor,
    eligible: t.Tensor,
    original_feature_ids: t.Tensor,
    *,
    descending: bool,
) -> t.Tensor:
    original_order = t.argsort(original_feature_ids, stable=True)
    candidates = original_order[eligible[original_order]]
    if candidates.numel() == 0:
        return candidates
    score_order = t.argsort(
        logits[candidates],
        descending=descending,
        stable=True,
    )
    return candidates[score_order]


@t.no_grad()
def sample_bernoulli_topk_bottomk(
    logits: t.Tensor,
    *,
    positive_count: int,
    negative_count: int,
    generator: t.Generator,
    original_feature_ids: t.Tensor | None = None,
) -> SparseActionTrace:
    """Classify independently, then select top positives and bottom negatives."""
    logits_f32 = _validate_logits(logits)
    feature_count = logits_f32.numel()
    _validate_counts(feature_count, positive_count, negative_count)
    if original_feature_ids is None:
        original_feature_ids = t.arange(feature_count, device=logits.device)
    if (
        original_feature_ids.ndim != 1
        or original_feature_ids.numel() != feature_count
        or original_feature_ids.dtype != t.int64
        or bool((original_feature_ids < 0).any())
        or t.unique(original_feature_ids).numel() != feature_count
    ):
        raise SFTGRPOError(
            "original_feature_ids must be unique non-negative int64 IDs with shape (V,)"
        )
    draws = t.rand(
        logits_f32.shape,
        dtype=t.float32,
        device=logits.device,
        generator=generator,
    )
    positive_class = draws < t.sigmoid(logits_f32)
    positive_order = _tie_stable_eligible_order(
        logits_f32,
        positive_class,
        original_feature_ids,
        descending=True,
    )
    negative_order = _tie_stable_eligible_order(
        logits_f32,
        ~positive_class,
        original_feature_ids,
        descending=False,
    )
    positive_rows = positive_order[:positive_count]
    negative_rows = negative_order[:negative_count]
    selected_terms = t.cat(
        (
            F.logsigmoid(logits_f32[positive_rows]),
            F.logsigmoid(-logits_f32[negative_rows]),
        )
    )
    return SparseActionTrace(
        mode=ActionMode.BERNOULLI_TOPK_BOTTOMK,
        requested_positive=positive_count,
        requested_negative=negative_count,
        positive_rows=tuple(int(row) for row in positive_rows.tolist()),
        negative_rows=tuple(int(row) for row in negative_rows.tolist()),
        behavior_log_terms=tuple(float(term) for term in selected_terms.tolist()),
    )


def bernoulli_selected_log_terms(
    logits: t.Tensor,
    trace: SparseActionTrace,
) -> t.Tensor:
    """Return the documented selected-feature surrogate terms."""
    if trace.mode is not ActionMode.BERNOULLI_TOPK_BOTTOMK:
        raise SFTGRPOError("Bernoulli surrogate requires a Bernoulli action trace")
    logits_f32 = _validate_logits(logits)
    selected = (*trace.positive_rows, *trace.negative_rows)
    if len(set(selected)) != len(selected) or any(
        not 0 <= row < logits_f32.numel() for row in selected
    ):
        raise SFTGRPOError("Bernoulli trace rows must be in-range and disjoint")
    positive_rows = t.as_tensor(trace.positive_rows, dtype=t.int64, device=logits.device)
    negative_rows = t.as_tensor(trace.negative_rows, dtype=t.int64, device=logits.device)
    return t.cat(
        (
            F.logsigmoid(logits_f32[positive_rows]),
            F.logsigmoid(-logits_f32[negative_rows]),
        )
    )


def replay_ratio(logits: t.Tensor, trace: SparseActionTrace) -> t.Tensor:
    """Return the exact weighted or selected-feature Bernoulli segment ratio."""
    if trace.mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        if len(trace.behavior_log_terms) != 1:
            raise SFTGRPOError("weighted behavior trace must contain one total log term")
        current = weighted_ordered_log_likelihood(logits, trace)
        behavior = current.new_tensor(trace.behavior_log_terms[0])
        return t.exp(current - behavior.detach())
    terms = bernoulli_selected_log_terms(logits, trace)
    if terms.numel() == 0:
        return logits.float().sum() * 0.0 + 1.0
    if len(trace.behavior_log_terms) != terms.numel():
        raise SFTGRPOError(
            "Bernoulli behavior trace term count does not equal selected feature count"
        )
    behavior = terms.new_tensor(trace.behavior_log_terms)
    return t.exp(terms - behavior.detach()).mean()


def behavior_checked_replay_ratio(
    logits: t.Tensor,
    trace: SparseActionTrace,
    *,
    atol: float,
    rtol: float,
) -> t.Tensor:
    """Validate behavior parity and return a ratio from one policy evaluation.

    Replay previously evaluated the selected action likelihood once for the
    fail-loud behavior check and again for the differentiable ratio.  The first
    graph was immediately detached and discarded.  This fused helper retains
    the same likelihood formula and tolerance check while reusing the one
    graph-connected evaluation for the ratio.
    """
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise SFTGRPOError("behavior/replay tolerances must be finite and non-negative")
    if trace.mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        if len(trace.behavior_log_terms) != 1:
            raise SFTGRPOError("weighted behavior trace must contain one total log term")
        replay = weighted_ordered_log_likelihood(logits, trace).reshape(1)
    else:
        replay = bernoulli_selected_log_terms(logits, trace)
    behavior = replay.new_tensor(trace.behavior_log_terms)
    if replay.shape != behavior.shape or not bool(
        t.allclose(replay.detach(), behavior, atol=atol, rtol=rtol)
    ):
        raise SFTGRPOError(
            f"behavior/replay log terms disagree: behavior={behavior.tolist()}, "
            f"replay={replay.detach().tolist()}, atol={atol}, rtol={rtol}"
        )
    if trace.mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        return t.exp(replay[0] - behavior[0].detach())
    if replay.numel() == 0:
        return logits.float().sum() * 0.0 + 1.0
    return t.exp(replay - behavior.detach()).mean()


def _behavior_checked_replay_ratio_from_finite_forward(
    logits: t.Tensor,
    trace: SparseActionTrace,
    *,
    atol: float,
    rtol: float,
) -> t.Tensor:
    """Check a weighted replay row already guarded by its policy forward."""
    if trace.mode is not ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        raise SFTGRPOError("finite-forward replay shortcut requires a weighted action trace")
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise SFTGRPOError("behavior/replay tolerances must be finite and non-negative")
    if len(trace.behavior_log_terms) != 1:
        raise SFTGRPOError("weighted behavior trace must contain one total log term")
    replay = _weighted_ordered_log_likelihood_from_finite_forward(logits, trace).reshape(1)
    behavior = replay.new_tensor(trace.behavior_log_terms)
    if replay.shape != behavior.shape or not bool(
        t.allclose(replay.detach(), behavior, atol=atol, rtol=rtol)
    ):
        raise SFTGRPOError(
            f"behavior/replay log terms disagree: behavior={behavior.tolist()}, "
            f"replay={replay.detach().tolist()}, atol={atol}, rtol={rtol}"
        )
    return t.exp(replay[0] - behavior[0].detach())


def _behavior_checked_replay_ratios_from_finite_log_weights(
    logits: t.Tensor,
    positive_log_weights: t.Tensor,
    negative_log_weights: t.Tensor,
    traces: tuple[SparseActionTrace, ...],
    *,
    atol: float,
    rtol: float,
) -> t.Tensor:
    """Check weighted replay rows whose microbatch log weights are already available."""
    if any(trace.mode is not ActionMode.WEIGHTED_WITHOUT_REPLACEMENT for trace in traces):
        raise SFTGRPOError("finite-log-weight replay shortcut requires weighted action traces")
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise SFTGRPOError("behavior/replay tolerances must be finite and non-negative")
    if any(len(trace.behavior_log_terms) != 1 for trace in traces):
        raise SFTGRPOError("every weighted behavior trace must contain one total log term")
    logits_f32 = logits.float()
    replay = _weighted_ordered_log_likelihood_batch_from_log_weights(
        logits_f32,
        positive_log_weights,
        negative_log_weights,
        traces,
    )
    behavior = replay.new_tensor([trace.behavior_log_terms[0] for trace in traces])
    if replay.shape != behavior.shape or not bool(
        t.allclose(replay.detach(), behavior, atol=atol, rtol=rtol)
    ):
        raise SFTGRPOError(
            f"behavior/replay log terms disagree: behavior={behavior.tolist()}, "
            f"replay={replay.detach().tolist()}, atol={atol}, rtol={rtol}"
        )
    return t.exp(replay - behavior.detach())


def assert_behavior_replay_equal(
    logits: t.Tensor,
    trace: SparseActionTrace,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Fail before update when replay does not reproduce behavior log terms."""
    if atol < 0 or rtol < 0 or not math.isfinite(atol) or not math.isfinite(rtol):
        raise SFTGRPOError("behavior/replay tolerances must be finite and non-negative")
    if trace.mode is ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
        replay = weighted_ordered_log_likelihood(logits, trace).detach().reshape(1)
    else:
        replay = bernoulli_selected_log_terms(logits, trace).detach()
    behavior = replay.new_tensor(trace.behavior_log_terms)
    if replay.shape != behavior.shape or not bool(
        t.allclose(replay, behavior, atol=atol, rtol=rtol)
    ):
        raise SFTGRPOError(
            f"behavior/replay log terms disagree: behavior={behavior.tolist()}, "
            f"replay={replay.tolist()}, atol={atol}, rtol={rtol}"
        )


def trace_to_conditioning(
    trace: SparseActionTrace,
    *,
    compact_to_original: t.Tensor | tuple[int, ...] | None,
) -> ConditioningAction:
    """Translate compact rows only at the encoder boundary and assign buckets."""
    rows = (*trace.positive_rows, *trace.negative_rows)
    if compact_to_original is None:
        feature_ids = rows
    elif isinstance(compact_to_original, tuple):
        if any(isinstance(value, bool) or not isinstance(value, int) for value in compact_to_original):
            raise SFTGRPOError("compact_to_original tuple must contain only integer IDs")
        if any(not 0 <= row < len(compact_to_original) for row in rows):
            raise SFTGRPOError("action trace row is outside compact feature vocabulary")
        feature_ids = tuple(int(compact_to_original[row]) for row in rows)
    else:
        if compact_to_original.ndim != 1 or compact_to_original.dtype != t.int64:
            raise SFTGRPOError("compact_to_original must be an int64 vector")
        if any(not 0 <= row < compact_to_original.numel() for row in rows):
            raise SFTGRPOError("action trace row is outside compact feature vocabulary")
        feature_ids = tuple(int(compact_to_original[row].item()) for row in rows)
    return ConditioningAction(
        feature_ids=tuple(feature_ids),
        magnitudes=(0,) * len(trace.positive_rows) + (1,) * len(trace.negative_rows),
        mask=(True,) * len(rows),
    )


def masked_mean_token_ce(per_token_ce: t.Tensor, reward_mask: t.Tensor) -> t.Tensor:
    """Return the per-row mean CE over valid targets, with shape ``(B,)``.

    The detached reward is exactly its negation, so no-grad collection and the
    differentiable next-token-prediction objective share this one reduction and
    can never drift apart.
    """
    if (
        per_token_ce.shape != reward_mask.shape
        or per_token_ce.ndim != 2
        or reward_mask.dtype != t.bool
    ):
        raise SFTGRPOError("per-token CE and bool reward mask must share shape (B,T)")
    valid = reward_mask.sum(dim=1)
    if bool((valid == 0).any()):
        raise SFTGRPOError("accepted environment reward mask contains no valid target")
    return per_token_ce.float().masked_fill(~reward_mask, 0.0).sum(dim=1) / valid.float()


def conversation_group_advantages(
    rewards: t.Tensor,
    *,
    std_floor: float = 1e-6,
) -> t.Tensor:
    """Normalize FP32 rewards independently along the final group dimension."""
    if rewards.ndim not in {1, 2} or rewards.shape[-1] <= 1:
        raise SFTGRPOError("conversation-local GRPO requires group_size > 1")
    if not math.isfinite(std_floor) or std_floor < 0:
        raise SFTGRPOError("advantage std_floor must be finite and non-negative")
    rewards_f32 = rewards.detach().float()
    if not bool(t.isfinite(rewards_f32).all()):
        raise SFTGRPOError("group rewards must be finite")
    mean = rewards_f32.mean(dim=-1, keepdim=True)
    centered = rewards_f32 - mean
    std = centered.square().mean(dim=-1, keepdim=True).sqrt()
    normalized = centered / t.where(
        std <= std_floor,
        t.ones_like(std),
        std,
    )
    return t.where(std <= std_floor, t.zeros_like(centered), normalized).detach()


def trajectory_mean_ratio(decision_ratios: list[t.Tensor]) -> t.Tensor:
    """Average live decisions, including graph-connected empty actions."""
    if not decision_ratios:
        raise SFTGRPOError("each accepted trajectory must contain a live decision")
    scalars = [ratio.reshape(()) for ratio in decision_ratios]
    return t.stack(scalars).mean()


def policy_gradient_loss(
    trajectory_ratios: t.Tensor,
    advantages: t.Tensor,
) -> t.Tensor:
    """Apply the exact conversation and trajectory mean reduction once."""
    if trajectory_ratios.shape != advantages.shape or trajectory_ratios.ndim != 2:
        raise SFTGRPOError(
            "trajectory ratios and advantages must share shape "
            "(conversations, group_size)"
        )
    if advantages.requires_grad:
        raise SFTGRPOError("advantages must be detached")
    return -(advantages * trajectory_ratios).mean()


def bernoulli_current_to_reference_kl(
    current_logits: t.Tensor,
    reference_logits: t.Tensor,
) -> t.Tensor:
    """Mean component-wise Bernoulli KL(current || reference) in FP32."""
    if current_logits.shape != reference_logits.shape or current_logits.numel() == 0:
        raise SFTGRPOError("current/reference KL logits must share a non-empty shape")
    current = current_logits.float()
    reference = reference_logits.detach().float()
    if not bool(t.isfinite(current).all()) or not bool(t.isfinite(reference).all()):
        raise SFTGRPOError("KL logits must be finite")
    component = (
        t.sigmoid(current) * (current - reference)
        + F.softplus(reference)
        - F.softplus(current)
    )
    return component.mean()


__all__ = [
    "ActionMode",
    "ConditioningAction",
    "SFTGRPOError",
    "SparseActionTrace",
    "assert_behavior_replay_equal",
    "bernoulli_current_to_reference_kl",
    "bernoulli_selected_log_terms",
    "conversation_group_advantages",
    "masked_mean_token_ce",
    "policy_gradient_loss",
    "replay_ratio",
    "behavior_checked_replay_ratio",
    "sample_bernoulli_topk_bottomk",
    "sample_weighted_without_replacement",
    "trace_to_conditioning",
    "trajectory_mean_ratio",
    "validate_count_support",
    "weighted_ordered_log_likelihood",
]
