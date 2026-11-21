"""Aggregation and plotting for AxBench-BPB-Adv class-margin scores.

Mirrors ``steering/eval/axbench_bpb/bpb_aggregation.py``: each score row owns
both completions of a core prefix (``loss_concept_rich`` and
``loss_concept_avoiding``), scored under ONE steered intervention. Aggregation
averages the two losses SEPARATELY across the core prefixes of a concept, then
recomputes the signed class margin ``primary_margin_y``:

* positive mode: ``BPB(avoid) - BPB(rich)``
* negative mode: ``BPB(rich) - BPB(avoid)`` (sign flip)

Higher ``primary_margin_y`` means stronger steering in both modes. Groups are
keyed by the FULL configuration including ``adversarial_strength`` /
``strength_index`` -- strengths are never collapsed. The ``is_control`` (k=0)
group of each non-strength configuration supplies ``baseline_primary_margin``;
when no control group exists the baseline is null and a report flag is raised.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from loguru import logger
from matplotlib.lines import Line2D

from nanochat.atomic_io import atomic_write_parquet, atomic_write_text
from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_bpb.representative_selection import (
    FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE,
    coefficient_abs,
    coefficient_lex,
)
from steering.eval.axbench_bpb_adv.config import ContextSignalMode, ScatterBaselineX
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    GENERAL_VAL_BPB_SCHEMA,
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
)

BPB_BASELINE_LABEL = "k=0 baseline margin"

ROUNDTRIP_PRIMARY_YLABEL = "primary roundtrip y"
ROUNDTRIP_CONTROL_LABEL = "control roundtrip y"

GENERAL_VAL_BPB_YLABEL = (
    "general validation BPB under steering (lower = less collateral damage to "
    "general language modelling)"
)

#: Pinned colours for the encoder-space feature families.
#:
#: Pinned rather than index-assigned so a family keeps its colour across runs
#: with different method populations -- the whole point of these figures is to
#: be read side by side. An index-assigned family also risks landing ON a pinned
#: colour: before ``feature GCG (margin alpha)`` was pinned it took tab10 index 4
#: (#9467bd) in any run without a decoder-only control, rendering it in exactly
#: the ``feature DiffMean`` purple within the same figure.
#:
#: ``feature GCG (margin alpha)`` is deliberately NOT a shade of the ``feature
#: GCG`` red: the two sweep different axes (class-margin alpha vs reg_lambda) and
#: are read as separate curves, so near-identical hues would be a liability
#: rather than a hint of kinship.
_FEATURE_FAMILY_COLOR_OVERRIDES = {
    "feature DiffMean": mcolors.to_rgba("#9467bd"),
    "feature GCG": mcolors.to_rgba("#d62728"),
    "feature GCG (margin alpha)": mcolors.to_rgba("#17becf"),
}


def _require_single_general_val_corpus(rows: list[dict[str, Any]], *, figure: str) -> None:
    """Issue #635: every drawn general_val_bpb in one figure family must share
    ONE corpus signature; raise naming the offenders otherwise."""
    by_corpus: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if row.get("general_val_bpb") is None:
            continue
        corpus = row.get("general_val_corpus")
        # Falsy (`not corpus`) rather than `is None`, matching the joins' and
        # `_single_group_corpus`'s predicate: an empty-string corpus is just as
        # unusable as a null one (issue #635).
        if not corpus:
            raise AxBenchBpbAdvError(
                f"{figure}: row method={row.get('method_name')!r} carries "
                "general_val_bpb without a corpus signature; the aggregates "
                "predate schema v2 — re-run stage=score_general_val + stage=aggregate"
            )
        by_corpus.setdefault(str(corpus), set()).add(
            (str(row.get("method_name")), str(row.get("target_model")))
        )
    if len(by_corpus) > 1:
        detail = "; ".join(
            f"{corpus} <- {sorted(members)}" for corpus, members in sorted(by_corpus.items())
        )
        raise AxBenchBpbAdvError(
            f"{figure} mixes general-val corpus signatures across plotted "
            f"families: {detail}. Every plotted family must be scored on the "
            "same held-out corpus (issue #635); re-score the odd runs out."
        )


PRIMARY_MARGIN_YLABEL = (
    "primary margin (positive: avoid - rich BPB; negative: rich - avoid BPB; "
    "higher = stronger steering)"
)


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise AxBenchBpbAdvError("cannot average an empty value list")
    return sum(materialized) / len(materialized)


def _require_finite_float(value: Any, *, field: str, context: str) -> float:
    if value is None:
        raise AxBenchBpbAdvError(f"{context} has missing {field}")
    number = float(value)
    if not math.isfinite(number):
        raise AxBenchBpbAdvError(f"{context} has non-finite {field}={value!r}")
    return number


def _require_percentile(value: Any, *, field: str, context: str) -> float:
    number = _require_finite_float(value, field=field, context=context)
    if number < 0.0 or number > 100.0:
        raise AxBenchBpbAdvError(f"{context} has {field}={number!r} outside [0, 100]")
    return number


def _signed_margin(*, mode: str, mean_rich: float, mean_avoid: float) -> float:
    # The sign rule lives ONLY in steering.eval.axbench_base.class_margin; this
    # thin wrapper re-raises as the project error type (keeping the aggregation
    # contract) while preserving the ORIGINAL message, so an unexpected ValueError
    # (e.g. a non-numeric loss) is not mislabeled as a steering-mode error.
    try:
        return signed_class_margin(mode=mode, rich=mean_rich, avoid=mean_avoid)
    except ValueError as exc:
        raise AxBenchBpbAdvError(str(exc)) from exc


def _full_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["concept_id"]),
        row["mode"],
        row["method_name"],
        row["target_model"],
        row["coefficient_str"],
        row["adversarial_method"],
        float(row["adversarial_strength"]),
        row["strength_unit"],
        int(row["strength_index"]),
        row["context_signal_mode"],
    )


def _control_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Key identifying the matching is_control group (strength dropped)."""

    return (
        int(row["concept_id"]),
        row["mode"],
        row["method_name"],
        row["target_model"],
        row["coefficient_str"],
        row["adversarial_method"],
        row["context_signal_mode"],
    )


def compute_per_candidate_aggregation(scores: pa.Table) -> pa.Table:
    if not scores.schema.equals(SCORE_SCHEMA):
        raise AxBenchBpbAdvError("expected scores with SCORE_SCHEMA")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in scores.to_pylist():
        grouped.setdefault(_full_key(row), []).append(row)

    # First pass: per-group margins + capture control-group margins.
    aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
    control_margins: dict[tuple[Any, ...], float] = {}
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        (
            concept_id,
            mode,
            method_name,
            target_model,
            coefficient_str,
            adversarial_method,
            adversarial_strength,
            strength_unit,
            strength_index,
            context_signal_mode,
        ) = key
        for row in rows:
            if row["loss_concept_rich"] is None or row["loss_concept_avoiding"] is None:
                raise AxBenchBpbAdvError(
                    f"group concept={concept_id} mode={mode!r} "
                    f"method={method_name!r} target={target_model!r} "
                    f"strength_index={strength_index} is missing a completion side "
                    "(loss_concept_rich / loss_concept_avoiding); refusing to "
                    "aggregate an incomplete class margin"
                )
        # INVARIANT: compute the two side means FIRST, then make exactly ONE
        # _signed_margin call on those means. NEVER average per-pair margins.
        # The margin is linear, so the two paths are numerically identical here
        # and a regression to per-pair averaging cannot be caught by tests; this
        # comment is the only guard. Per-pair averaging would also silently
        # mis-weight uneven cartesian sides under any future non-linear margin.
        mean_rich = _mean(row["loss_concept_rich"] for row in rows)
        mean_avoid = _mean(row["loss_concept_avoiding"] for row in rows)
        primary_margin_y = _signed_margin(
            mode=str(mode), mean_rich=mean_rich, mean_avoid=mean_avoid
        )
        # Group control flag: a per_candidate group is a control iff EVERY score
        # row in it carries the persisted is_control flag (sourced from
        # scoring._is_control_candidate = strength-0 AND prompt unmodified). This
        # reads the persisted score-level flag, NEVER adversarial_strength.
        is_control_group = all(bool(row["is_control"]) for row in rows)
        entry = {
            "concept_id": int(concept_id),
            "mode": str(mode),
            "method_name": str(method_name),
            "target_model": str(target_model),
            "coefficient_str": str(coefficient_str),
            "adversarial_method": str(adversarial_method),
            "adversarial_strength": float(adversarial_strength),
            "strength_unit": str(strength_unit),
            "strength_index": int(strength_index),
            "context_signal_mode": str(context_signal_mode),
            "mean_loss_concept_rich": mean_rich,
            "mean_loss_concept_avoiding": mean_avoid,
            "primary_margin_y": primary_margin_y,
            "baseline_primary_margin": None,
            # Filled by _join_general_val_bpb (in aggregate_bpb_adv_scores) when a
            # general_val table is present; stays null otherwise.
            "general_val_bpb": None,
            "general_val_corpus": None,
            "n_samples": len(rows),
            "truncation_rate": _mean(
                1.0 if (row["rich_truncated"] or row["avoid_truncated"]) else 0.0 for row in rows
            ),
            "n_rich_output_tokens": sum(int(row["n_rich_output_tokens"]) for row in rows),
            "n_avoid_output_tokens": sum(int(row["n_avoid_output_tokens"]) for row in rows),
            "n_rich_output_bytes": sum(int(row["n_rich_output_bytes"]) for row in rows),
            "n_avoid_output_bytes": sum(int(row["n_avoid_output_bytes"]) for row in rows),
            "is_control": is_control_group,
        }
        aggregated[key] = entry
        if is_control_group:
            control_margins[_control_key(rows[0])] = primary_margin_y

    # Second pass: fill baseline_primary_margin from the matching control group.
    for entry in aggregated.values():
        entry["baseline_primary_margin"] = control_margins.get(_control_key(entry))

    out = [aggregated[key] for key in sorted(aggregated)]
    return pa.Table.from_pylist(out, schema=PER_CANDIDATE_AGG_SCHEMA)


def compute_macro_aggregation(per_candidate: pa.Table) -> pa.Table:
    if not per_candidate.schema.equals(PER_CANDIDATE_AGG_SCHEMA):
        raise AxBenchBpbAdvError("expected per-candidate aggregation with PER_CANDIDATE_AGG_SCHEMA")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_candidate.to_pylist():
        key = (
            row["mode"],
            row["method_name"],
            row["target_model"],
            row["coefficient_str"],
            row["adversarial_method"],
            float(row["adversarial_strength"]),
            row["strength_unit"],
            int(row["strength_index"]),
            row["context_signal_mode"],
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        (
            mode,
            method_name,
            target_model,
            coefficient_str,
            adversarial_method,
            adversarial_strength,
            strength_unit,
            strength_index,
            context_signal_mode,
        ) = key
        baseline_values = [
            float(row["baseline_primary_margin"])
            for row in rows
            if row["baseline_primary_margin"] is not None
        ]
        baseline_primary_margin = (
            sum(baseline_values) / len(baseline_values) if baseline_values else None
        )
        # Concept-average the per-candidate general_val_bpb (joined upstream).
        # The intervention is per (target, method, concept, coefficient,
        # adversarial), so this mean over the group's concepts mirrors how
        # primary_margin_y is macro-averaged. Null when no concept had a joined
        # value (feature disabled / unmatched).
        general_val_values = [
            float(row["general_val_bpb"]) for row in rows if row["general_val_bpb"] is not None
        ]
        general_val_bpb = (
            sum(general_val_values) / len(general_val_values) if general_val_values else None
        )
        out.append(
            {
                "mode": str(mode),
                "method_name": str(method_name),
                "target_model": str(target_model),
                "coefficient_str": str(coefficient_str),
                "adversarial_method": str(adversarial_method),
                "adversarial_strength": float(adversarial_strength),
                "strength_unit": str(strength_unit),
                "strength_index": int(strength_index),
                "context_signal_mode": str(context_signal_mode),
                "mean_loss_concept_rich": _mean(row["mean_loss_concept_rich"] for row in rows),
                "mean_loss_concept_avoiding": _mean(
                    row["mean_loss_concept_avoiding"] for row in rows
                ),
                "primary_margin_y": _mean(row["primary_margin_y"] for row in rows),
                "baseline_primary_margin": baseline_primary_margin,
                "general_val_bpb": general_val_bpb,
                "general_val_corpus": _single_group_corpus(rows, group=key),
                "n_concepts": len({int(row["concept_id"]) for row in rows}),
                "n_samples": sum(int(row["n_samples"]) for row in rows),
                "truncation_rate": _mean(row["truncation_rate"] for row in rows),
                # Aggregate the per_candidate is_control across the concepts in
                # this macro group; a true control configuration is control for
                # EVERY concept, so all(...) is the correct macro-level flag.
                "is_control": all(bool(row["is_control"]) for row in rows),
            }
        )
    return pa.Table.from_pylist(out, schema=MACRO_AGG_SCHEMA)


def _roundtrip_primary_y(*, mode: str, percentile: float) -> float:
    if mode == "positive":
        return float(percentile)
    if mode == "negative":
        return 100.0 - float(percentile)
    raise AxBenchBpbAdvError(f"unknown steering mode {mode!r}")


def compute_roundtrip_per_candidate_aggregation(scores: pa.Table) -> pa.Table:
    if not scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA):
        raise AxBenchBpbAdvError("expected scores with ROUNDTRIP_SCORE_SCHEMA")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in scores.to_pylist():
        grouped.setdefault(_full_key(row), []).append(row)

    aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        (
            concept_id,
            mode,
            method_name,
            target_model,
            coefficient_str,
            adversarial_method,
            adversarial_strength,
            strength_unit,
            strength_index,
            context_signal_mode,
        ) = key
        context = (
            f"roundtrip group concept={concept_id} mode={mode!r} "
            f"method={method_name!r} target={target_model!r} "
            f"strength_index={strength_index}"
        )
        for row in rows:
            if row["loss_concept_rich"] is None or row["loss_concept_avoiding"] is None:
                raise AxBenchBpbAdvError(
                    f"{context} is missing BPB baseline "
                    "losses; refusing to build a baseline-loss scatter"
                )
            for field in (
                "loss_concept_rich",
                "loss_concept_avoiding",
                "roundtrip_pre_act",
                "roundtrip_post_act",
                "control_roundtrip_pre_act",
                "control_roundtrip_post_act",
            ):
                _require_finite_float(row[field], field=field, context=context)
            for field in ("roundtrip_percentile", "control_roundtrip_percentile"):
                _require_percentile(row[field], field=field, context=context)
        mean_rich = _mean(row["loss_concept_rich"] for row in rows)
        mean_avoid = _mean(row["loss_concept_avoiding"] for row in rows)
        mean_pre = _mean(row["roundtrip_pre_act"] for row in rows)
        mean_post = _mean(row["roundtrip_post_act"] for row in rows)
        mean_percentile = _mean(row["roundtrip_percentile"] for row in rows)
        primary_y = _roundtrip_primary_y(
            mode=str(mode),
            percentile=mean_percentile,
        )
        mean_control_pre = _mean(row["control_roundtrip_pre_act"] for row in rows)
        mean_control_post = _mean(row["control_roundtrip_post_act"] for row in rows)
        mean_control_percentile = _mean(row["control_roundtrip_percentile"] for row in rows)
        control_y = _roundtrip_primary_y(
            mode=str(mode),
            percentile=mean_control_percentile,
        )
        is_control_group = all(bool(row["is_control"]) for row in rows)
        entry = {
            "concept_id": int(concept_id),
            "mode": str(mode),
            "method_name": str(method_name),
            "target_model": str(target_model),
            "coefficient_str": str(coefficient_str),
            "adversarial_method": str(adversarial_method),
            "adversarial_strength": float(adversarial_strength),
            "strength_unit": str(strength_unit),
            "strength_index": int(strength_index),
            "context_signal_mode": str(context_signal_mode),
            "mean_loss_concept_rich": mean_rich,
            "mean_loss_concept_avoiding": mean_avoid,
            "mean_roundtrip_pre_act": mean_pre,
            "mean_roundtrip_post_act": mean_post,
            "mean_roundtrip_percentile": mean_percentile,
            "primary_roundtrip_y": primary_y,
            "mean_control_roundtrip_pre_act": mean_control_pre,
            "mean_control_roundtrip_post_act": mean_control_post,
            "mean_control_roundtrip_percentile": mean_control_percentile,
            "control_primary_roundtrip_y": control_y,
            # Filled by _join_roundtrip_general_val_bpb when a general_val table is
            # present; stays null otherwise.
            "general_val_bpb": None,
            "general_val_corpus": None,
            "n_samples": len(rows),
            "n_generated_tokens": sum(int(row["n_generated_tokens"]) for row in rows),
            "n_control_generated_tokens": sum(
                int(row["control_n_generated_tokens"]) for row in rows
            ),
            "max_tokens_rate": _mean(
                1.0 if row["stop_reason"] == "max_tokens" else 0.0 for row in rows
            ),
            "control_max_tokens_rate": _mean(
                1.0 if row["control_stop_reason"] == "max_tokens" else 0.0 for row in rows
            ),
            "is_control": is_control_group,
        }
        aggregated[key] = entry

    out = [aggregated[key] for key in sorted(aggregated)]
    return pa.Table.from_pylist(out, schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)


def compute_roundtrip_macro_aggregation(per_candidate: pa.Table) -> pa.Table:
    if not per_candidate.schema.equals(ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA):
        raise AxBenchBpbAdvError(
            "expected per-candidate aggregation with ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA"
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in per_candidate.to_pylist():
        key = (
            row["mode"],
            row["method_name"],
            row["target_model"],
            row["coefficient_str"],
            row["adversarial_method"],
            float(row["adversarial_strength"]),
            row["strength_unit"],
            int(row["strength_index"]),
            row["context_signal_mode"],
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        (
            mode,
            method_name,
            target_model,
            coefficient_str,
            adversarial_method,
            adversarial_strength,
            strength_unit,
            strength_index,
            context_signal_mode,
        ) = key
        out.append(
            {
                "mode": str(mode),
                "method_name": str(method_name),
                "target_model": str(target_model),
                "coefficient_str": str(coefficient_str),
                "adversarial_method": str(adversarial_method),
                "adversarial_strength": float(adversarial_strength),
                "strength_unit": str(strength_unit),
                "strength_index": int(strength_index),
                "context_signal_mode": str(context_signal_mode),
                "mean_loss_concept_rich": _mean(row["mean_loss_concept_rich"] for row in rows),
                "mean_loss_concept_avoiding": _mean(
                    row["mean_loss_concept_avoiding"] for row in rows
                ),
                "mean_roundtrip_pre_act": _mean(row["mean_roundtrip_pre_act"] for row in rows),
                "mean_roundtrip_post_act": _mean(row["mean_roundtrip_post_act"] for row in rows),
                "mean_roundtrip_percentile": _mean(
                    row["mean_roundtrip_percentile"] for row in rows
                ),
                "primary_roundtrip_y": _mean(row["primary_roundtrip_y"] for row in rows),
                "mean_control_roundtrip_pre_act": _mean(
                    row["mean_control_roundtrip_pre_act"] for row in rows
                ),
                "mean_control_roundtrip_post_act": _mean(
                    row["mean_control_roundtrip_post_act"] for row in rows
                ),
                "mean_control_roundtrip_percentile": _mean(
                    row["mean_control_roundtrip_percentile"] for row in rows
                ),
                "control_primary_roundtrip_y": _mean(
                    row["control_primary_roundtrip_y"] for row in rows
                ),
                "general_val_bpb": _mean_optional(row["general_val_bpb"] for row in rows),
                "general_val_corpus": _single_group_corpus(rows, group=key),
                "n_concepts": len({int(row["concept_id"]) for row in rows}),
                "n_samples": sum(int(row["n_samples"]) for row in rows),
                "n_generated_tokens": sum(int(row["n_generated_tokens"]) for row in rows),
                "n_control_generated_tokens": sum(
                    int(row["n_control_generated_tokens"]) for row in rows
                ),
                "max_tokens_rate": _mean(row["max_tokens_rate"] for row in rows),
                "control_max_tokens_rate": _mean(row["control_max_tokens_rate"] for row in rows),
                "is_control": all(bool(row["is_control"]) for row in rows),
            }
        )
    return pa.Table.from_pylist(out, schema=ROUNDTRIP_MACRO_AGG_SCHEMA)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    # Atomic write via the canonical nanochat.atomic_io.atomic_write_text (tmp +
    # fsync + os.replace): the canonical CSV is only ever the COMPLETE, durable
    # file, so a crash/power-loss mid-write never leaves a truncated
    # per_candidate.csv / macro.csv beside the now-atomic parquets.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text("", path)
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(buffer.getvalue(), path)


def _general_val_lookup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Join key shared by per_candidate rows and the general_val table.

    ``adversarial_strength`` (== k for the universal Faster-GCG attack) is part of
    the key: the substituted trigger -- hence ``general_val_bpb`` -- differs per
    ``k``, so two combos that differ only in k must NOT share a general_val
    value. The join lands per-k on the intervention identity."""

    return (
        str(row["target_model"]),
        str(row["method_name"]),
        int(row["concept_id"]),
        str(row["coefficient_str"]),
        str(row["adversarial_method"]),
        float(row["adversarial_strength"]),
    )


def _apply_general_val_join(
    per_candidate: pa.Table, general_val: pa.Table, *, out_schema: pa.Schema
) -> pa.Table:
    """Left-join ``(general_val_bpb, general_val_corpus)`` onto per_candidate rows.

    The shared body of :func:`_join_general_val_bpb` (class-margin) and
    :func:`_join_roundtrip_general_val_bpb` (roundtrip) — the two differ only in
    ``out_schema``. Keeping ONE implementation means the corpus-validation
    invariant ("a general_val_bpb value must carry a corpus signature") can never
    diverge between the two metric families.

    The join key is the intervention identity ``(target_model, method_name,
    concept_id, coefficient_str, adversarial_method, adversarial_strength)`` (see
    ``_general_val_lookup_key``: ``adversarial_strength`` == k is load-bearing
    under the universal-trigger attack). A per_candidate row with no matching
    general_val row keeps ``general_val_bpb`` / ``general_val_corpus`` null.
    """

    if not general_val.schema.equals(GENERAL_VAL_BPB_SCHEMA):
        raise AxBenchBpbAdvError("expected general-val table with GENERAL_VAL_BPB_SCHEMA")
    lookup: dict[tuple[Any, ...], tuple[float | None, str | None]] = {}
    for row in general_val.to_pylist():
        value = row["general_val_bpb"]
        corpus = row["general_val_corpus"]
        if value is not None and not corpus:
            raise AxBenchBpbAdvError(
                "general-val table has a general_val_bpb without a corpus "
                f"signature (intervention_signature="
                f"{row['intervention_signature']!r}); re-run stage=score_general_val"
            )
        lookup[_general_val_lookup_key(row)] = (
            None if value is None else float(value),
            None if value is None else str(corpus),
        )
    rows = per_candidate.to_pylist()
    for row in rows:
        value, corpus = lookup.get(_general_val_lookup_key(row), (None, None))
        row["general_val_bpb"] = value
        row["general_val_corpus"] = corpus
    return pa.Table.from_pylist(rows, schema=out_schema)


def _join_general_val_bpb(per_candidate: pa.Table, general_val: pa.Table) -> pa.Table:
    """Left-join per-intervention general_val_bpb onto class-margin per_candidate rows."""

    return _apply_general_val_join(
        per_candidate, general_val, out_schema=PER_CANDIDATE_AGG_SCHEMA
    )


def _mean_optional(values: Iterable[Any]) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _single_group_corpus(rows: list[dict[str, Any]], *, group: tuple[Any, ...]) -> str | None:
    corpora: set[str] = set()
    for row in rows:
        value = row.get("general_val_bpb")
        corpus = row.get("general_val_corpus")
        # Falsy (`not corpus`) rather than `is None`, matching the joins'
        # predicate: an empty-string corpus is just as unusable for the "null
        # iff general_val_bpb is null" invariant as a null one (issue #635).
        if value is not None and not corpus:
            raise AxBenchBpbAdvError(
                f"macro group {group!r} has general_val_bpb without a corpus "
                "signature; the aggregates predate schema v2 — re-run "
                "stage=score_general_val then stage=aggregate"
            )
        # Collect on the SAME falsy predicate as the guard above: an empty-string
        # corpus is the unusable/missing sentinel, never a distinct signature, so
        # it must not enter the set and trip the >1 mixing raise (issue #635).
        if corpus:
            corpora.add(str(corpus))
    if len(corpora) > 1:
        raise AxBenchBpbAdvError(
            f"macro group {group!r} mixes general-val corpus signatures "
            f"{sorted(corpora)}; every value in one group must come from one corpus"
        )
    return next(iter(corpora)) if corpora else None


def _join_roundtrip_general_val_bpb(per_candidate: pa.Table, general_val: pa.Table) -> pa.Table:
    """Left-join per-intervention general_val_bpb onto roundtrip aggregates."""

    if not per_candidate.schema.equals(ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA):
        raise AxBenchBpbAdvError(
            "expected roundtrip per-candidate aggregation with ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA"
        )
    return _apply_general_val_join(
        per_candidate, general_val, out_schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA
    )


def aggregate_bpb_adv_scores(
    scores_path: Path,
    out_dir: Path,
    *,
    general_val_path: Path | None = None,
    scores_table: pa.Table | None = None,
    general_val_table: pa.Table | None = None,
) -> pa.Table:
    scores = pq.read_table(scores_path) if scores_table is None else scores_table
    if not scores.schema.equals(SCORE_SCHEMA):
        raise AxBenchBpbAdvError(f"schema mismatch in {scores_path}: expected SCORE_SCHEMA")
    if scores.num_rows == 0:
        raise AxBenchBpbAdvError(f"no axbench-bpb-adv scores in {scores_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_candidate = compute_per_candidate_aggregation(scores)
    # Optional: fold general-validation BPB (computed per intervention by the
    # score_general_val stage) into per_candidate BEFORE the macro roll-up, so the
    # macro general_val_bpb is the concept-average over the SAME concepts that
    # contribute the margin.
    general_val = (
        pq.read_table(general_val_path)
        if general_val_table is None
        and general_val_path is not None
        and general_val_path.exists()
        else general_val_table
    )
    if general_val is not None:
        per_candidate = _join_general_val_bpb(per_candidate, general_val)
    macro = compute_macro_aggregation(per_candidate)
    # Atomic + durable (fsync) writes via the canonical helper -- a crash/power
    # loss mid-write never lands a torn parquet at the canonical path.
    atomic_write_parquet(per_candidate, out_dir / "per_candidate.parquet")
    atomic_write_parquet(macro, out_dir / "macro.parquet")
    _write_csv(per_candidate.to_pylist(), out_dir / "per_candidate.csv")
    _write_csv(macro.to_pylist(), out_dir / "macro.csv")
    per_candidate_rows = per_candidate.to_pylist()
    missing_baseline_groups = sum(
        1 for row in per_candidate_rows if row["baseline_primary_margin"] is None
    )
    n_with_general_val = sum(1 for row in per_candidate_rows if row["general_val_bpb"] is not None)
    (out_dir / "aggregation_report.yaml").write_text(
        yaml.safe_dump(
            {
                "n_score_rows": scores.num_rows,
                "n_per_candidate_rows": per_candidate.num_rows,
                "n_macro_rows": macro.num_rows,
                "per_candidate_rows_missing_baseline_primary_margin": (missing_baseline_groups),
                "per_candidate_rows_with_general_val_bpb": n_with_general_val,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return macro


def aggregate_roundtrip_adv_scores(
    scores_path: Path,
    out_dir: Path,
    *,
    general_val_path: Path | None = None,
    scores_table: pa.Table | None = None,
    general_val_table: pa.Table | None = None,
) -> pa.Table:
    scores = pq.read_table(scores_path) if scores_table is None else scores_table
    if not scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA):
        raise AxBenchBpbAdvError(
            f"schema mismatch in {scores_path}: expected ROUNDTRIP_SCORE_SCHEMA"
        )
    if scores.num_rows == 0:
        raise AxBenchBpbAdvError(f"no axbench-bpb-adv roundtrip scores in {scores_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_candidate = compute_roundtrip_per_candidate_aggregation(scores)
    general_val = (
        pq.read_table(general_val_path)
        if general_val_table is None
        and general_val_path is not None
        and general_val_path.exists()
        else general_val_table
    )
    if general_val is not None:
        per_candidate = _join_roundtrip_general_val_bpb(
            per_candidate,
            general_val,
        )
    macro = compute_roundtrip_macro_aggregation(per_candidate)
    atomic_write_parquet(per_candidate, out_dir / "per_candidate.parquet")
    atomic_write_parquet(macro, out_dir / "macro.parquet")
    _write_csv(per_candidate.to_pylist(), out_dir / "per_candidate.csv")
    _write_csv(macro.to_pylist(), out_dir / "macro.csv")
    (out_dir / "aggregation_report.yaml").write_text(
        yaml.safe_dump(
            {
                "n_score_rows": scores.num_rows,
                "n_per_candidate_rows": per_candidate.num_rows,
                "n_macro_rows": macro.num_rows,
                "per_candidate_rows_with_general_val_bpb": sum(
                    1 for row in per_candidate.to_pylist() if row["general_val_bpb"] is not None
                ),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return macro


def _series_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row["mode"]),
        str(row["method_name"]),
        str(row["target_model"]),
        str(row["coefficient_str"]),
        str(row["adversarial_method"]),
        str(row["context_signal_mode"]),
    )


def _target_label(target_model: str) -> str:
    if target_model == "decoder_only":
        return "decoder-only"
    if target_model == "encdec":
        return "encdec"
    return target_model


def _steering_family(method_name: str) -> str:
    method = method_name
    if method.startswith("neg_"):
        method = method.removeprefix("neg_")
    for suffix in ("_decoder_only", "_encdec"):
        if method.endswith(suffix):
            method = method.removesuffix(suffix)
            break
    if method.startswith("feature_diffmean"):
        return "feature DiffMean"
    # BEFORE the generic ours_gcg prefix, which would otherwise swallow it: the
    # alpha methods are a SEPARATE series because their coefficient axis is a
    # different quantity (class-margin alpha, not reg_lambda). Plotting the two
    # under one family would draw a single curve over two incomparable x-axes.
    if method.startswith("ours_gcg_alpha"):
        return "feature GCG (margin alpha)"
    if method.startswith("ours_gcg"):
        return "feature GCG"
    if method.startswith("ours_"):
        return "ours SAE edits"
    if method == "linear_probe":
        return "linear probe"
    return method


def _model_steering_family(row: dict[str, Any]) -> str:
    return (
        f"{_target_label(str(row['target_model']))} / {_steering_family(str(row['method_name']))}"
    )


def _series_marker(method_name: str) -> str:
    """Marker shape for one plotted method series.

    Mode is already split into separate plot panels, so marker shape is reserved
    for the fair/roofline distinction: regular methods are circles, roofline
    oracle methods are stars.
    """

    if "roofline" in method_name:
        return "*"
    return "o"


def _context_linestyle(context_signal_mode: str) -> str:
    # ``not_applicable`` is the additive sweeps' context (no prompt-token signal);
    # it is omitted from the context legend, so it draws SOLID like a real context
    # rather than a dash-dot with no legend key to explain it.
    return {
        ContextSignalMode.FIXED_CONTEXT.value: "-",
        ContextSignalMode.NOT_APPLICABLE.value: "-",
    }.get(context_signal_mode, ":")


def _family_color_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign one colour per plotted family: pinned where declared, else cycled.

    Pinned families are resolved FIRST and their colours are then withheld from
    the cycle, so an index-assigned family can never be drawn in a pinned
    family's colour. Assigning the cycle first (as this did before) let the two
    collide silently: with families sorted, ``feature GCG (margin alpha)`` landed
    on tab10 index 4 -- ``feature DiffMean``'s pinned #9467bd -- in any run
    without a decoder-only control, putting two distinct families on one colour
    in the same figure.
    """
    families = sorted({_model_steering_family(row) for row in rows})
    pinned: dict[str, Any] = {}
    for family in families:
        _target, separator, steering_family = family.partition(" / ")
        if not separator:
            continue
        override = _FEATURE_FAMILY_COLOR_OVERRIDES.get(steering_family)
        if override is not None:
            pinned[family] = override

    cmap = plt.get_cmap("tab10")
    reserved = {mcolors.to_hex(color) for color in pinned.values()}
    available = [
        cmap(i) for i in range(cmap.N) if mcolors.to_hex(cmap(i)) not in reserved
    ]
    # Degenerate only if every cycle colour were pinned; fall back to the full
    # cycle rather than dividing by zero, since a duplicate beats a crash.
    if not available:
        available = [cmap(i) for i in range(cmap.N)]

    color_map: dict[str, Any] = {}
    cycled = 0
    for family in families:
        if family in pinned:
            color_map[family] = pinned[family]
            continue
        color_map[family] = available[cycled % len(available)]
        cycled += 1
    return color_map


def strength_color_for_rank(base_color: Any, *, rank: int, n_ranks: int) -> Any:
    """Return a light-to-dark shade of ``base_color`` for adversarial strength."""

    if n_ranks <= 1:
        progress = 0.0
    else:
        progress = max(0.0, min(1.0, rank / (n_ranks - 1)))
    white_fraction = 0.7 * (1.0 - progress)
    rgb = mcolors.to_rgb(base_color)
    return tuple(white_fraction + (1.0 - white_fraction) * channel for channel in rgb)


def _rank_by_strength_index(rows: list[dict[str, Any]]) -> dict[int, int]:
    return {
        strength_index: rank
        for rank, strength_index in enumerate(sorted({int(row["strength_index"]) for row in rows}))
    }


def _group_rows_by_series(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    by_series: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        by_series.setdefault(_series_key(row), []).append(row)
    return by_series


def _series_baseline_value(
    series_rows: list[dict[str, Any]],
    *,
    baseline_metric: str = "baseline_primary_margin",
) -> float | None:
    values = [
        float(row[baseline_metric]) for row in series_rows if row[baseline_metric] is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _representative_family_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Key for representative-coefficient selection (coefficient EXCLUDED).

    Mirrors ``axbench_bpb``'s per (method, concept, mode, target) grouping over
    which the best coefficient is chosen by ``idxmax``. Here the family spans a
    full coefficient grid AND a full adversarial-strength sweep per coefficient;
    selection groups by everything EXCEPT ``coefficient_str`` and the strength
    axis (``adversarial_strength`` / ``strength_unit`` / ``strength_index``).
    """

    return (
        str(row["mode"]),
        str(row["method_name"]),
        str(row["target_model"]),
        str(row["adversarial_method"]),
        str(row["context_signal_mode"]),
    )


def _is_control_row(row: dict[str, Any]) -> bool:
    """A no-adversary control row, read ONLY from the persisted ``is_control`` flag.

    The flag is authoritative and propagated through both aggregation schemas
    (``PER_CANDIDATE_AGG_SCHEMA`` / ``MACRO_AGG_SCHEMA``): per_candidate
    ``is_control`` = ``all(score is_control)``, macro ``is_control`` =
    ``all(per_candidate is_control)``. It originates from
    ``scoring._is_control_candidate`` (strength-0 AND prompt unmodified), so this
    predicate is unified with that authoritative definition and never re-derives
    control status from ``adversarial_strength``. Indexing (``row["is_control"]``)
    is intentional: a row reaching here without the column is an un-propagated
    path and should surface a KeyError loudly rather than silently mis-classify.
    """

    return bool(row["is_control"])


def _control_baseline_margin(
    coefficient_rows: list[dict[str, Any]],
    *,
    baseline_metric: str = "baseline_primary_margin",
) -> float | None:
    """Best-available control margin for one coefficient family.

    Reads the adversary-agnostic ``baseline_primary_margin`` from the control
    rows (the k=0 group). ``baseline_primary_margin`` is nullable (pyarrow
    default) and may be absent when no control group exists -> returns ``None``,
    and such a coefficient cannot win the argmax.
    """

    values = [
        float(row[baseline_metric])
        for row in coefficient_rows
        if _is_control_row(row) and row.get(baseline_metric) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _select_representative_coefficient(
    rows: list[dict[str, Any]],
    *,
    baseline_metric: str = "baseline_primary_margin",
    baseline_rows: list[dict[str, Any]] | None = None,
    target_scores: dict[tuple[str, str, str], float] | None = None,
) -> dict[tuple[Any, ...], str | None]:
    """Pick the representative ``coefficient_str`` per steering family.

    Per (mode, method_name, target_model, adversarial_method,
    context_signal_mode) choose a single ``coefficient_str``: the coefficient
    whose no-adversary steerability is closest to the scored fair
    encoder-decoder feature-DiffMean bag baseline for the same
    mode/adversary/context tuple. Exact closest-score ties prefer the higher
    steerability, then smaller absolute coefficient, then lexical coefficient.

    ``target_scores`` may be passed pre-computed to skip the linear rescan of
    ``baseline_rows`` when a caller selects repeatedly against the same baseline.
    """

    by_family: dict[tuple[Any, ...], dict[Any, list[dict[str, Any]]]] = {}
    for row in rows:
        family_key = _representative_family_key(row)
        coefficient = row["coefficient_str"]
        by_family.setdefault(family_key, {}).setdefault(coefficient, []).append(row)

    return _select_representative_coefficient_closest_to_feature_diffmean(
        by_family,
        baseline_rows=rows if baseline_rows is None else baseline_rows,
        baseline_metric=baseline_metric,
        target_scores=target_scores,
    )


def _feature_diffmean_target_scores(
    rows: list[dict[str, Any]],
    *,
    baseline_metric: str,
) -> dict[tuple[str, str, str], float]:
    # BPB-Adv counterpart of axbench_bpb.bpb_aggregation.compute_per_concept_rollup's
    # feature_diffmean_targets: the fair feature-DiffMean encdec baseline the
    # representative coefficient is chosen closest to. Kept separate on purpose --
    # here the key carries the adversarial axis (mode, adversarial_method,
    # context_signal_mode) and only no-adversary control rows count. The shared
    # method-name definition lives in FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.
    values: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        mode = str(row["mode"])
        if str(row["target_model"]) != "encdec":
            continue
        if str(row["method_name"]) != FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.get(mode):
            continue
        if not _is_control_row(row):
            continue
        metric_value = row.get(baseline_metric)
        if metric_value is None:
            continue
        key = (
            mode,
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        values.setdefault(key, []).append(float(metric_value))
    return {key: sum(metric_values) / len(metric_values) for key, metric_values in values.items()}


def _missing_representative_target_error(
    *,
    mode: str,
    adversarial_method: str,
    context_signal_mode: str,
    baseline_metric: str,
) -> AxBenchBpbAdvError:
    target_method = FAIR_FEATURE_DIFFMEAN_METHOD_BY_MODE.get(mode)
    return AxBenchBpbAdvError(
        "missing fair feature-DiffMean target row for representative selection: "
        f"mode={mode!r} adversarial_method={adversarial_method!r} "
        f"context_signal_mode={context_signal_mode!r} "
        f"target_method={target_method!r} target_model='encdec' "
        f"metric={baseline_metric!r}. Run scoring with feature-DiffMean "
        "methods enabled."
    )


def _require_representative_target_scores(
    rows: list[dict[str, Any]],
    *,
    target_scores: dict[tuple[str, str, str], float],
    baseline_metric: str,
) -> None:
    """Validate target availability before plot cleanup can delete old output."""

    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        target_key = (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        if target_key in seen:
            continue
        seen.add(target_key)
        if target_scores.get(target_key) is None:
            raise _missing_representative_target_error(
                mode=target_key[0],
                adversarial_method=target_key[1],
                context_signal_mode=target_key[2],
                baseline_metric=baseline_metric,
            )


def _select_representative_coefficient_closest_to_feature_diffmean(
    by_family: dict[tuple[Any, ...], dict[Any, list[dict[str, Any]]]],
    *,
    baseline_rows: list[dict[str, Any]],
    baseline_metric: str,
    target_scores: dict[tuple[str, str, str], float] | None = None,
) -> dict[tuple[Any, ...], str | None]:
    if target_scores is None:
        target_scores = _feature_diffmean_target_scores(
            baseline_rows,
            baseline_metric=baseline_metric,
        )

    selected: dict[tuple[Any, ...], str | None] = {}
    for family_key, by_coefficient in by_family.items():
        mode, _method_name, _target_model, adversarial_method, context_signal_mode = (
            str(value) for value in family_key
        )
        target_key = (mode, adversarial_method, context_signal_mode)
        target_score = target_scores.get(target_key)
        if target_score is None:
            raise _missing_representative_target_error(
                mode=mode,
                adversarial_method=adversarial_method,
                context_signal_mode=context_signal_mode,
                baseline_metric=baseline_metric,
            )

        scored: list[tuple[float, float, float, str, str | None]] = []
        for coefficient, coefficient_rows in by_coefficient.items():
            score = _control_baseline_margin(
                coefficient_rows,
                baseline_metric=baseline_metric,
            )
            if score is None:
                continue
            coefficient_value = None if coefficient is None else str(coefficient)
            scored.append(
                (
                    abs(score - target_score),
                    -score,
                    coefficient_abs(coefficient),
                    coefficient_lex(coefficient),
                    coefficient_value,
                )
            )

        if not scored:
            selected[family_key] = None
            continue
        scored.sort()
        selected[family_key] = scored[0][4]

    return selected


def _filter_rows_to_representative_coefficient(
    rows: list[dict[str, Any]],
    *,
    baseline_metric: str = "baseline_primary_margin",
    baseline_rows: list[dict[str, Any]] | None = None,
    target_scores: dict[tuple[str, str, str], float] | None = None,
) -> list[dict[str, Any]]:
    """Keep only the representative coefficient's rows per steering family.

    Preserves the FULL adversarial-strength sweep of the chosen coefficient (the
    strength axis is never reduced). Families that select ``None`` (no usable
    control margin, or a ``None`` coefficient) are kept wholesale so nothing is
    silently dropped.

    ``target_scores`` may be passed pre-computed to skip re-deriving the fair
    feature-DiffMean baseline when selecting repeatedly against one baseline set.
    """

    representative = _select_representative_coefficient(
        rows,
        baseline_metric=baseline_metric,
        baseline_rows=baseline_rows,
        target_scores=target_scores,
    )
    kept: list[dict[str, Any]] = []
    for row in rows:
        family_key = _representative_family_key(row)
        chosen = representative.get(family_key)
        coefficient = row["coefficient_str"]
        if chosen is None:
            kept.append(row)
            continue
        if coefficient is not None and str(coefficient) == chosen:
            kept.append(row)
    return kept


SCATTER_PLOT_NAME = "steerability_vs_baseline.png"

# Default footer for the representative-coefficient scatter (each family reduced
# to its best coefficient). The per-coefficient scatters override this with a
# fixed-magnitude note (they do NOT pick a representative -- see F14).
_REPRESENTATIVE_SCATTER_FOOTER = (
    "One point per counter-steering strength at each family's representative "
    "coefficient. Color = model/steering family; brightness = counter-steering "
    "strength rank; marker = method variant; line connects a family's "
    "counter-steering sweep. y is the mode-signed steerability margin."
)


def _scatter_baseline_x(
    row: dict[str, Any], *, scatter_baseline_x: ScatterBaselineX | str
) -> float | None:
    """The scatter x-axis baseline for one row, or ``None`` if undefined.

    * ``baseline_loss`` -> the model's NATURAL (unsteered) completion BPB, which
      is **mode-dependent**: positive mode steers TOWARD the concept so the
      natural completion is concept-AVOIDING (``mean_loss_concept_avoiding``);
      negative mode steers AWAY so the natural completion is concept-RICH
      (``mean_loss_concept_rich``). ``None`` if that side's loss is null.
    * ``min`` -> ``min(mean_loss_concept_rich, mean_loss_concept_avoiding)``;
      ``None`` if EITHER side is null (the minimum would be undefined).
    """

    choice = ScatterBaselineX.coerce(scatter_baseline_x)
    if choice is ScatterBaselineX.BASELINE_LOSS:
        mode = str(row["mode"])
        if mode == "positive":
            baseline = row.get("mean_loss_concept_avoiding")
        elif mode == "negative":
            baseline = row.get("mean_loss_concept_rich")
        else:
            raise AxBenchBpbAdvError(f"unknown steering mode {mode!r}")
        return None if baseline is None else float(baseline)
    if choice is ScatterBaselineX.GENERAL_VAL_BPB:
        # The held-out generic-validation BPB under this row's intervention; a
        # concept-agnostic "how broken is the model" axis. None when the
        # general_val pass produced no value for this intervention.
        general_val = row.get("general_val_bpb")
        return None if general_val is None else float(general_val)
    rich = row.get("mean_loss_concept_rich")
    avoid = row.get("mean_loss_concept_avoiding")
    if rich is None or avoid is None:
        return None
    return min(float(rich), float(avoid))


@dataclass(frozen=True)
class _ScatterPoint:
    """One steerability-vs-baseline scatter point at one counter-steering strength.

    A family's points span the COUNTER-STEERING strength sweep (the k=0 control
    plus each adversarial strength) at the family's representative base
    coefficient; ``strength_index`` is the ordering/brightness key.
    """

    mode: str
    method_name: str
    target_model: str
    adversarial_method: str
    context_signal_mode: str
    coefficient_str: str
    adversarial_strength: float
    strength_index: int
    x: float
    y: float

    def family_key(self) -> tuple[str, str, str, str, str]:
        """Method-family identity for connecting a counter-steering sweep.

        ``context_signal_mode`` is part of the point identity. After
        ``recomputed_context`` was removed, each ``adversarial_method`` maps to a
        single context (prompt-token attacks record ``fixed_context``;
        additive-direction rows record ``not_applicable``), and
        ``adversarial_method`` is already in this key, so the context is
        effectively redundant -- it is retained as a behaviour-preserving
        partition that never merges two distinct contexts into one family /
        brightness ranking. ``coefficient_str`` is EXCLUDED because the plot
        reduces each family to a single representative coefficient before
        building points.
        """

        return (
            self.mode,
            self.method_name,
            self.target_model,
            self.adversarial_method,
            self.context_signal_mode,
        )


def _scatter_series_key(point: _ScatterPoint) -> tuple[str, ...]:
    """Trajectory identity for one connected counter-steering sweep.

    The family color/identity (``family_key``) EXCLUDES the base coefficient
    because the plot normally reduces each family to a single representative
    coefficient. But representative selection can keep a multi-coefficient family
    WHOLESALE (no usable control margin -- ``baseline_primary_margin`` is
    nullable), so the connected-line grouping additionally includes
    ``coefficient_str`` (mirroring the strength plot's ``_series_key``). That
    keeps each coefficient its OWN strength sweep instead of one line zigzagging
    across coefficients; in the normal (reduced) case it is identical to
    ``family_key`` since every point shares the one representative coefficient.
    """

    return (*point.family_key(), point.coefficient_str)


def _scatter_trajectories(points: list[_ScatterPoint]) -> list[list[_ScatterPoint]]:
    """Group scatter points into connected counter-steering trajectories.

    One trajectory per series (``_scatter_series_key`` = family + base
    coefficient), each ordered ASCENDING by ``strength_index`` -- which is BOTH
    the connecting-line draw order AND the light->dark brightness ranking.
    Trajectories are returned in a stable (series-key-sorted) order so the plot
    output is deterministic and directly testable. Keeping an unreduced
    multi-coefficient family as SEPARATE per-coefficient sweeps (rather than one
    line zigzagging across coefficients) is the load-bearing invariant.
    """

    by_series: dict[tuple[str, ...], list[_ScatterPoint]] = {}
    for point in points:
        by_series.setdefault(_scatter_series_key(point), []).append(point)
    return [
        sorted(series_points, key=lambda point: point.strength_index)
        for _key, series_points in sorted(by_series.items(), key=lambda item: item[0])
    ]


def _scatter_points(
    rows: list[dict[str, Any]], *, scatter_baseline_x: ScatterBaselineX | str
) -> list[_ScatterPoint]:
    """Build the scatter points for one mode subset.

    PURE point builder: one point per input row (NO control-row filter), so a
    family's full counter-steering-strength sweep contributes a point at every
    strength. The representative-coefficient reduction is applied by the caller
    (``_plot_steerability_vs_baseline_scatter``) BEFORE this is called; here we
    only drop rows with a ``None`` coefficient or a ``None`` baseline x. The y is
    the steerability ``primary_margin_y`` RECOMPUTED via ``_signed_margin`` from
    the row's two side losses (the sign-flip lives only in ``_signed_margin`` and
    is never re-derived here).
    """

    ScatterBaselineX.coerce(scatter_baseline_x)  # validate up front
    points: list[_ScatterPoint] = []
    for row in rows:
        coefficient = row.get("coefficient_str")
        if coefficient is None:
            continue
        x = _scatter_baseline_x(row, scatter_baseline_x=scatter_baseline_x)
        if x is None:
            continue
        rich = row.get("mean_loss_concept_rich")
        avoid = row.get("mean_loss_concept_avoiding")
        if rich is None or avoid is None:
            continue
        mode = str(row["mode"])
        y = _signed_margin(mode=mode, mean_rich=float(rich), mean_avoid=float(avoid))
        points.append(
            _ScatterPoint(
                mode=mode,
                method_name=str(row["method_name"]),
                target_model=str(row["target_model"]),
                adversarial_method=str(row["adversarial_method"]),
                context_signal_mode=str(row["context_signal_mode"]),
                coefficient_str=str(coefficient),
                adversarial_strength=float(row["adversarial_strength"]),
                strength_index=int(row["strength_index"]),
                x=float(x),
                y=float(y),
            )
        )
    return points


def _scatter_xlabel(scatter_baseline_x: ScatterBaselineX | str, *, mode: str) -> str:
    choice = ScatterBaselineX.coerce(scatter_baseline_x)
    if choice is ScatterBaselineX.MIN:
        return "baseline BPB (min of concept-rich / concept-avoiding)"
    if choice is ScatterBaselineX.GENERAL_VAL_BPB:
        return "general validation BPB under steering (higher = more model damage)"
    # ``baseline_loss`` = the natural/unsteered completion, which is
    # concept-avoiding for positive mode and concept-rich for negative mode.
    natural = "concept-avoiding" if mode == "positive" else "concept-rich"
    return f"baseline BPB ({natural})"


def _plot_steerability_vs_baseline_scatter(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    scatter_baseline_x: str,
    mode: str,
    coefficient_selector: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ] = _filter_rows_to_representative_coefficient,
    title_extra: str = "",
    footer_note: str = _REPRESENTATIVE_SCATTER_FOOTER,
) -> Path | None:
    """Scatter steerability (``primary_margin_y``) vs the baseline BPB.

    Per method-family (method, target, adversarial_method, context) the family is
    reduced to its REPRESENTATIVE base coefficient and plotted as its
    COUNTER-STEERING strength sweep: one point per adversarial strength,
    connected light->dark in strength order. Color encodes the model/steering
    family; brightness encodes the counter-steering strength rank; marker
    distinguishes regular methods from roofline oracle methods. Because the
    swept axis is counter-steering strength (not the base coefficient), a
    single-base-config method (LoRA, a no-steering control, encdec-ours) still
    gets a full multi-point curve. Returns the written path, or ``None`` when
    no point survives (all coefficients/baselines null)."""

    # Reduce each family to its representative base coefficient (keeping the full
    # counter-steering sweep of that coefficient), then build one point per
    # strength -- mirroring the strength-curve plots' representative selection.
    # ``coefficient_selector`` defaults to the representative reduction; the
    # per-coefficient scatters pass a fixed-magnitude selector instead.
    selected_rows = coefficient_selector(rows)
    points = _scatter_points(selected_rows, scatter_baseline_x=scatter_baseline_x)
    if not points:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    color_map = _family_color_map(rows)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    try:
        # One connected sweep per series (family + coefficient), ordered by
        # counter-steering strength -- see _scatter_trajectories for the grouping
        # invariant (separate per-coefficient sweeps; ascending-strength order).
        for family_points in _scatter_trajectories(points):
            first = {
                "target_model": family_points[0].target_model,
                "method_name": family_points[0].method_name,
            }
            base_color = color_map[_model_steering_family(first)]
            n_ranks = len(family_points)
            point_colors = [
                strength_color_for_rank(base_color, rank=rank, n_ranks=n_ranks)
                for rank in range(n_ranks)
            ]
            xs = [point.x for point in family_points]
            ys = [point.y for point in family_points]
            # Connect the counter-steering sweep in strength order. Context is a
            # family-key component (one trajectory per context) and linestyle is
            # still keyed on context via _context_linestyle, but both live contexts
            # (fixed_context, not_applicable) now render solid, so in a mixed-context
            # scatter the trajectories are separated by family/color, not by line.
            ax.plot(
                xs,
                ys,
                color=base_color,
                linewidth=1.0,
                alpha=0.5,
                linestyle=_context_linestyle(family_points[0].context_signal_mode),
                zorder=1,
            )
            ax.scatter(
                xs,
                ys,
                marker=_series_marker(family_points[0].method_name),
                s=42,
                c=point_colors,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.95,
                zorder=2,
            )
        ax.set_xlabel(_scatter_xlabel(scatter_baseline_x, mode=mode))
        ax.set_ylabel(PRIMARY_MARGIN_YLABEL)
        ax.set_title(
            f"steerability vs baseline | mode={mode}{title_extra}",
            fontsize="large",
        )
        ax.grid(True, alpha=0.25)
        _add_style_legend(fig, rows=rows, color_map=color_map)
        fig.text(
            0.01,
            0.01,
            footer_note,
            fontsize="x-small",
        )
        fig.tight_layout(rect=(0.0, 0.06, 0.77, 0.96))
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)
    return out_path


def _fixed_coefficient_selector(
    coefficient_str: str,
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Selector that fixes the magnitude: keep ONLY rows at ``coefficient_str``.

    The counterpart to ``_filter_rows_to_representative_coefficient`` (which picks
    the best coefficient per family). Used by the per-coefficient scatters so each
    magnitude gets its own figure, keeping that magnitude's full counter-steering
    strength sweep across every family.
    """

    target = coefficient_str
    return lambda rows: [r for r in rows if str(r["coefficient_str"]) == target]


def _plot_per_coefficient_scatters(
    rows: list[dict[str, Any]],
    by_coeff_dir: Path,
    *,
    scatter_baseline_x: str,
    mode: str,
) -> list[tuple[Path, int]]:
    """Emit one steerability-vs-baseline scatter per distinct ``coefficient_str``.

    For each magnitude (``None`` first, then ascending) draw a scatter that FIXES
    that magnitude across all families (rather than reducing each family to its
    representative coefficient). Returns ``[(png_path, n_points), ...]`` for the
    scatters that produced at least one point, and writes a self-contained
    ``plot_index.csv`` under ``by_coeff_dir``.
    """

    written: list[tuple[Path, int]] = []
    index_rows: list[dict[str, Any]] = []
    # _coefficient_slug is non-injective ("6" and "6.0" both -> "coeff_6"), so a
    # mixed int/float grid could map two DISTINCT coefficient_str values to the
    # SAME directory and silently overwrite a PNG. Detect the collision and FAIL
    # LOUD naming both colliding values rather than emit a wrong figure.
    slug_to_coefficient: dict[str, str] = {}
    for coefficient_str in _distinct_coefficients_sorted(rows):
        slug = _coefficient_slug(coefficient_str)
        prior = slug_to_coefficient.get(slug)
        if prior is not None and prior != coefficient_str:
            raise AxBenchBpbAdvError(
                "per-coefficient scatter slug collision: coefficient_str "
                f"{prior!r} and {coefficient_str!r} both map to the filesystem "
                f"slug coeff_{slug!r}, which would overwrite a single PNG; the "
                "coefficient grid must not mix forms that collapse to one slug"
            )
        slug_to_coefficient[slug] = coefficient_str
        selector = _fixed_coefficient_selector(coefficient_str)
        out_path = by_coeff_dir / f"coeff_{slug}" / SCATTER_PLOT_NAME
        path = _plot_steerability_vs_baseline_scatter(
            rows,
            out_path,
            scatter_baseline_x=scatter_baseline_x,
            mode=mode,
            coefficient_selector=selector,
            title_extra=f", coefficient={coefficient_str}",
            footer_note=(
                "One point per counter-steering strength at a fixed base "
                f"coefficient ({coefficient_str}). Color = model/steering family; "
                "brightness = counter-steering strength rank; marker = method "
                "variant; line connects a family's counter-steering sweep. y is "
                "the mode-signed steerability margin."
            ),
        )
        if path is None:
            continue
        n_points = len(_scatter_points(selector(rows), scatter_baseline_x=scatter_baseline_x))
        written.append((path, n_points))
        index_rows.append(
            {
                "path": str(path.relative_to(by_coeff_dir)),
                "mode": mode,
                "coefficient_str": coefficient_str,
                "plot_name": SCATTER_PLOT_NAME,
                "n_rows": n_points,
            }
        )
    if index_rows:
        _write_csv(index_rows, by_coeff_dir / "plot_index.csv")
    return written


def _write_plot_series_legend(
    rows: list[dict[str, Any]],
    plots_dir: Path,
    *,
    baseline_metric: str = "baseline_primary_margin",
    baseline_column_name: str = "baseline_primary_margin",
    scatter_paths_by_mode: dict[str, str] | None = None,
) -> Path:
    color_map = _family_color_map(rows)
    scatter_paths_by_mode = scatter_paths_by_mode or {}
    legend_rows: list[dict[str, Any]] = []
    by_series = _group_rows_by_series(rows)
    for series_id, (_key, series_rows) in enumerate(
        sorted(by_series.items(), key=lambda item: item[0]),
        start=1,
    ):
        first = series_rows[0]
        family = _model_steering_family(first)
        strengths = [float(row["adversarial_strength"]) for row in series_rows]
        rank_by_strength = _rank_by_strength_index(series_rows)
        n_ranks = len(rank_by_strength)
        scatter_plot_path = scatter_paths_by_mode.get(str(first["mode"]), "")
        legend_rows.append(
            {
                "series_id": f"S{series_id:04d}",
                "model_steering_family": family,
                "color": mcolors.to_hex(color_map[family]),
                "baseline_color": mcolors.to_hex(
                    strength_color_for_rank(color_map[family], rank=0, n_ranks=n_ranks)
                ),
                "max_strength_color": mcolors.to_hex(
                    strength_color_for_rank(
                        color_map[family],
                        rank=max(0, n_ranks - 1),
                        n_ranks=n_ranks,
                    )
                ),
                "mode": str(first["mode"]),
                "marker": _series_marker(str(first["method_name"])),
                "method_name": str(first["method_name"]),
                "target_model": str(first["target_model"]),
                "coefficient_str": str(first["coefficient_str"]),
                "adversarial_method": str(first["adversarial_method"]),
                "strength_unit": str(first["strength_unit"]),
                "context_signal_mode": str(first["context_signal_mode"]),
                "line_style": _context_linestyle(str(first["context_signal_mode"])),
                "n_points": len(series_rows),
                "min_adversarial_strength": min(strengths),
                "max_adversarial_strength": max(strengths),
                baseline_column_name: _series_baseline_value(
                    series_rows,
                    baseline_metric=baseline_metric,
                ),
                # Steerability-vs-baseline scatter PNG for this series' mode
                # subset; lets the legend reference every emitted scatter so the
                # plot bookkeeping and the recursive plot count stay in sync.
                "plot_path": scatter_plot_path,
            }
        )
    out_path = plots_dir / "plot_series_legend.csv"
    _write_csv(legend_rows, out_path)
    return out_path


def _panel_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["adversarial_method"]), str(row["strength_unit"]))


def _panel_title(panel_rows: list[dict[str, Any]]) -> str:
    first = panel_rows[0]
    targets = sorted({_target_label(str(row["target_model"])) for row in panel_rows})
    target_text = ", ".join(targets)
    return f"{first['adversarial_method']} | {first['strength_unit']} | {target_text}"


def _add_style_legend(
    fig: plt.Figure,
    *,
    rows: list[dict[str, Any]],
    color_map: dict[str, Any],
    baseline_label: str = BPB_BASELINE_LABEL,
) -> None:
    family_handles = [
        Line2D([0], [0], color=color, lw=2.5, label=family)
        for family, color in sorted(color_map.items())
    ]
    context_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=_context_linestyle(context),
            lw=2,
            label=context,
        )
        # ``not_applicable`` is the additive rows' (non-prompt-token) context; it
        # carries no meaningful linestyle distinction, so it is omitted from the
        # legend rather than shown as a spurious context mode.
        for context in sorted({str(row["context_signal_mode"]) for row in rows})
        if context != ContextSignalMode.NOT_APPLICABLE
    ]
    shade_handles = [
        Line2D(
            [0],
            [0],
            color=strength_color_for_rank("black", rank=0, n_ranks=2),
            lw=2.5,
            label="control / lowest strength",
        ),
        Line2D(
            [0],
            [0],
            color=strength_color_for_rank("black", rank=1, n_ranks=2),
            lw=2.5,
            label="highest strength",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            lw=1.5,
            label=baseline_label,
        ),
    ]
    handles = family_handles + context_handles + shade_handles
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        fontsize="x-small",
        title="Encoding",
        title_fontsize="small",
        frameon=True,
    )


def _plot_series_grid(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    x_getter: Callable[[dict[str, Any]], Any],
    xlabel: str,
    y_getter: Callable[[dict[str, Any]], Any],
    ylabel: str,
    title: str,
    mode: str,
    draw_baseline_reference: bool = False,
    baseline_metric: str = "baseline_primary_margin",
    baseline_label: str = BPB_BASELINE_LABEL,
) -> Path | None:
    usable = [row for row in rows if x_getter(row) is not None and y_getter(row) is not None]
    if not usable:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_keys = sorted({_panel_key(row) for row in usable})
    ncols = min(2, len(panel_keys))
    nrows = math.ceil(len(panel_keys) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.0 * ncols + 3.0, 4.7 * nrows),
        squeeze=False,
    )
    color_map = _family_color_map(usable)
    by_series = _group_rows_by_series(usable)
    try:
        for ax in axes.flat:
            ax.set_visible(False)
        for panel_index, panel_key in enumerate(panel_keys):
            ax = axes.flat[panel_index]
            ax.set_visible(True)
            panel_rows = [row for row in usable if _panel_key(row) == panel_key]
            for _series_key_value, series_rows in sorted(by_series.items()):
                series_panel_rows = [row for row in series_rows if _panel_key(row) == panel_key]
                if not series_panel_rows:
                    continue
                series_panel_rows.sort(key=lambda row: int(row["strength_index"]))
                first = series_panel_rows[0]
                family = _model_steering_family(first)
                rank_by_strength = _rank_by_strength_index(series_panel_rows)
                n_ranks = len(rank_by_strength)
                point_colors = [
                    strength_color_for_rank(
                        color_map[family],
                        rank=rank_by_strength[int(row["strength_index"])],
                        n_ranks=n_ranks,
                    )
                    for row in series_panel_rows
                ]
                xs = [float(x_getter(row)) for row in series_panel_rows]
                ys = [float(y_getter(row)) for row in series_panel_rows]
                linestyle = _context_linestyle(str(first["context_signal_mode"]))
                for left_idx in range(len(series_panel_rows) - 1):
                    ax.plot(
                        xs[left_idx : left_idx + 2],
                        ys[left_idx : left_idx + 2],
                        linewidth=1.15,
                        color=point_colors[left_idx + 1],
                        linestyle=linestyle,
                        alpha=0.92,
                    )
                ax.scatter(
                    xs,
                    ys,
                    marker=_series_marker(str(first["method_name"])),
                    s=18,
                    c=point_colors,
                    edgecolors="black",
                    linewidths=0.25,
                    alpha=0.95,
                )
                if draw_baseline_reference:
                    baseline = _control_baseline_margin(
                        series_panel_rows,
                        baseline_metric=baseline_metric,
                    )
                    if baseline is not None:
                        ax.axhline(
                            baseline,
                            color=color_map[family],
                            linestyle=":",
                            linewidth=1.0,
                            alpha=0.6,
                        )
            ax.set_title(_panel_title(panel_rows), fontsize="medium")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        fig.suptitle(
            f"{title} | mode={mode}",
            fontsize="large",
        )
        fig.text(
            0.01,
            0.01,
            "Series metadata: ../../plot_series_legend.csv. Color = family; "
            "shade = strength rank; dotted line = no-adversary reference.",
            fontsize="x-small",
        )
        _add_style_legend(
            fig,
            rows=usable,
            color_map=color_map,
            baseline_label=baseline_label,
        )
        fig.tight_layout(rect=(0.0, 0.06, 0.77, 0.95))
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)
    return out_path


def _plot_metric_vs_strength(
    rows: list[dict[str, Any]],
    out_path: Path,
    *,
    metric: str,
    ylabel: str,
    mode: str,
    draw_baseline_reference: bool = False,
    baseline_metric: str = "baseline_primary_margin",
    baseline_label: str = BPB_BASELINE_LABEL,
) -> Path | None:
    return _plot_series_grid(
        rows,
        out_path,
        x_getter=lambda row: row["adversarial_strength"],
        xlabel="adversarial strength",
        y_getter=lambda row: row.get(metric),
        ylabel=ylabel,
        title=ylabel,
        mode=mode,
        draw_baseline_reference=draw_baseline_reference,
        baseline_metric=baseline_metric,
        baseline_label=baseline_label,
    )


def _coefficient_slug(coefficient_str: str) -> str:
    """Filesystem-safe slug for a coefficient_str (numeric digit-safe rules)."""
    if coefficient_str == "None":
        return "none"
    try:
        text = f"{float(coefficient_str):g}"
    except ValueError:
        text = coefficient_str
    return text.replace("-", "neg_").replace(".", "p")


def _coefficient_sort_key(coefficient_str: str) -> tuple[int, float, str]:
    """None first, then numeric ascending, then non-numeric lexicographic."""
    if coefficient_str == "None":
        return (0, 0.0, "")
    try:
        return (1, float(coefficient_str), coefficient_str)
    except ValueError:
        return (2, math.inf, coefficient_str)


def _distinct_coefficients_sorted(rows: list[dict[str, Any]]) -> list[str]:
    distinct = {str(row["coefficient_str"]) for row in rows}
    return sorted(distinct, key=_coefficient_sort_key)


def _plot_subset_dir(plots_dir: Path, *, mode: str) -> Path:
    return plots_dir / mode


def _rows_by_mode(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["mode"]), []).append(row)
    return grouped


def _clear_previous_plot_outputs(plots_dir: Path) -> None:
    if not plots_dir.exists():
        return
    # Recursive: the per-coefficient scatters write nested PNGs AND a nested
    # by_coefficient/plot_index.csv, so a stale prior run must be cleared by
    # walking the whole tree (a non-recursive top-level unlink would leave the
    # by_coefficient artifacts behind and break idempotency).
    for path in plots_dir.rglob("*.png"):
        path.unlink()
    for path in plots_dir.rglob("plot_index.csv"):
        path.unlink()
    legend = plots_dir / "plot_series_legend.csv"
    if legend.exists():
        legend.unlink()
    # Remove directories left empty by the unlinks above (e.g. a
    # by_coefficient/coeff_<slug>/ whose coefficient vanished on a shrinking
    # re-run), bottom-up so a parent is considered only after its children are
    # gone. The plots_dir root itself is preserved. Walk a materialized list of
    # subdirs deepest-first; rmdir only succeeds on (and is restricted to) empty
    # dirs, so a directory still holding artifacts is left untouched.
    subdirs = sorted(
        (p for p in plots_dir.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for directory in subdirs:
        if not any(directory.iterdir()):
            directory.rmdir()


def _write_plot_index(rows: list[dict[str, Any]], path: Path) -> Path:
    _write_csv(rows, path)
    return path


def plot_bpb_adv_summaries(
    macro: pa.Table,
    plots_dir: Path,
    *,
    scatter_baseline_x: str = ScatterBaselineX.BASELINE_LOSS.value,
    plot_methods: list[str] | None = None,
) -> list[Path]:
    # Validate BEFORE clearing so a bad whitelist fails loudly without wiping prior
    # plots. An empty list keeps NO methods (would clear everything and emit
    # nothing); pass None to plot all. The config layer guards this too, but the
    # public function must not silently wipe when called directly.
    if plot_methods is not None and not plot_methods:
        raise AxBenchBpbAdvError(
            "plot_methods is an empty list, which keeps no methods and would clear "
            "all prior plots while emitting nothing; pass None to plot every method "
            "or list the methods to keep"
        )
    all_rows = macro.to_pylist()
    rows = list(all_rows)
    # Plot-stage method whitelist: keep only the named methods (no re-scoring).
    if plot_methods is not None:
        available = {str(row["method_name"]) for row in rows}
        unknown = sorted(set(plot_methods) - available)
        if unknown:
            raise AxBenchBpbAdvError(
                f"plot_methods contains methods absent from the aggregates: "
                f"{unknown}; available methods: {sorted(available)}"
            )
        keep = set(plot_methods)
        rows = [row for row in rows if str(row["method_name"]) in keep]
    # Issue #635: general_val_bpb is drawn on shared axes by the companion line
    # plot and (when selected) the scatters; one corpus per invocation.
    _require_single_general_val_corpus(rows, figure="plot_bpb_adv_summaries")
    # The representative-coefficient baseline (fair feature-DiffMean control
    # scores) depends only on ``all_rows`` -- compute it ONCE and reuse it across
    # every mode and every selection below (primary line, scatter selector, and
    # scatter point count) instead of rescanning all_rows per call.
    representative_target_scores = _feature_diffmean_target_scores(
        all_rows, baseline_metric="baseline_primary_margin"
    )
    _require_representative_target_scores(
        rows,
        target_scores=representative_target_scores,
        baseline_metric="baseline_primary_margin",
    )
    _clear_previous_plot_outputs(plots_dir)
    written: list[Path] = []
    plot_index_rows: list[dict[str, Any]] = []
    # Maps each mode subset to its emitted scatter PNG relpath
    # so the series legend can reference every scatter (keeps the recursive plot
    # count, the plot_index, and the legend in lockstep -- PR#503/#504 lesson).
    scatter_paths_by_mode: dict[str, str] = {}
    # Diagnostics (besides the primary margin plot, which is drawn explicitly).
    diagnostic_specs: list[tuple[str, str, str, bool]] = [
        (
            "mean_loss_concept_rich_vs_strength.png",
            "mean_loss_concept_rich",
            "mean concept-rich BPB",
            False,
        ),
        (
            "mean_loss_concept_avoiding_vs_strength.png",
            "mean_loss_concept_avoiding",
            "mean concept-avoiding BPB",
            False,
        ),
        (
            "baseline_primary_margin_vs_strength.png",
            "baseline_primary_margin",
            "baseline primary margin (k=0 control)",
            False,
        ),
    ]
    for mode, subset_rows in sorted(_rows_by_mode(rows).items()):
        subset_dir = _plot_subset_dir(plots_dir, mode=mode)
        # Primary plot: x = adversarial strength, y = primary_margin_y, with the
        # k=0 control baseline drawn as a dashed reference line. The regular-
        # steering line collapses each method family to its REPRESENTATIVE
        # coefficient (closest to the fair Feature-DiffMean control score) while
        # KEEPING that coefficient's full adversarial-strength sweep -- the
        # diagnostics below still show the complete coefficient grid.
        representative_rows = _filter_rows_to_representative_coefficient(
            subset_rows,
            baseline_rows=all_rows,
            target_scores=representative_target_scores,
        )
        primary_path = _plot_metric_vs_strength(
            representative_rows,
            subset_dir / "primary_margin_vs_strength.png",
            metric="primary_margin_y",
            ylabel=PRIMARY_MARGIN_YLABEL,
            mode=mode,
            draw_baseline_reference=True,
        )
        if primary_path is not None:
            written.append(primary_path)
            plot_index_rows.append(
                {
                    "path": str(primary_path.relative_to(plots_dir)),
                    "mode": mode,
                    "plot_name": "primary_margin_vs_strength.png",
                    "n_rows": len(representative_rows),
                }
            )
        # Companion to the primary plot, SHARING its adversarial-strength x-axis
        # and the same representative-coefficient reduction (one line per family):
        # how much the steering degrades GENERAL language modelling as adversarial
        # pressure ramps. Auto-skips (returns None) when general_val_bpb was not
        # computed -- every y is null, so _plot_series_grid finds no usable rows.
        general_val_path_out = _plot_metric_vs_strength(
            representative_rows,
            subset_dir / "general_val_bpb_vs_strength.png",
            metric="general_val_bpb",
            ylabel=GENERAL_VAL_BPB_YLABEL,
            mode=mode,
            draw_baseline_reference=False,
        )
        if general_val_path_out is not None:
            written.append(general_val_path_out)
            plot_index_rows.append(
                {
                    "path": str(general_val_path_out.relative_to(plots_dir)),
                    "mode": mode,
                    "plot_name": "general_val_bpb_vs_strength.png",
                    "n_rows": len(representative_rows),
                }
            )
        for name, metric, ylabel, draw_baseline in diagnostic_specs:
            path = _plot_metric_vs_strength(
                subset_rows,
                subset_dir / name,
                metric=metric,
                ylabel=ylabel,
                mode=mode,
                draw_baseline_reference=draw_baseline,
            )
            if path is not None:
                written.append(path)
                plot_index_rows.append(
                    {
                        "path": str(path.relative_to(plots_dir)),
                        "mode": mode,
                        "plot_name": name,
                        "n_rows": len(subset_rows),
                    }
                )
        # Steerability-vs-baseline scatter: one point per counter-steering
        # strength at each family's representative base coefficient (y =
        # mode-signed steerability margin; x per scatter_baseline_x). Registered
        # everywhere the line plots are so the recursive plot counter includes it.
        scatter_path = _plot_steerability_vs_baseline_scatter(
            subset_rows,
            subset_dir / SCATTER_PLOT_NAME,
            scatter_baseline_x=scatter_baseline_x,
            mode=mode,
            coefficient_selector=lambda visible_rows: _filter_rows_to_representative_coefficient(
                visible_rows,
                baseline_rows=all_rows,
                target_scores=representative_target_scores,
            ),
        )
        if scatter_path is not None:
            written.append(scatter_path)
            scatter_relpath = str(scatter_path.relative_to(plots_dir))
            scatter_paths_by_mode[str(mode)] = scatter_relpath
            # Count the ACTUALLY-plotted points: representative coefficient per
            # family, swept over counter-steering strength (matches the plot).
            scatter_points = _scatter_points(
                _filter_rows_to_representative_coefficient(
                    subset_rows,
                    baseline_rows=all_rows,
                    target_scores=representative_target_scores,
                ),
                scatter_baseline_x=scatter_baseline_x,
            )
            plot_index_rows.append(
                {
                    "path": scatter_relpath,
                    "mode": mode,
                    "plot_name": SCATTER_PLOT_NAME,
                    "n_rows": len(scatter_points),
                }
            )
        # Per-coefficient steerability scatters: one scatter per distinct
        # coefficient_str under by_coefficient/, fixing the magnitude instead of
        # picking the representative. Self-contained (its own plot_index.csv); the
        # paths join the returned `written` list so the recursive plot counter sees
        # them, but they stay OUT of the top-level plot_index.
        per_coeff = _plot_per_coefficient_scatters(
            subset_rows,
            subset_dir / "by_coefficient",
            scatter_baseline_x=scatter_baseline_x,
            mode=mode,
        )
        for path, _n in per_coeff:
            written.append(path)
        if per_coeff:
            written.append(subset_dir / "by_coefficient" / "plot_index.csv")
        logger.info(
            f"plot_bpb_adv_summaries: mode={mode} emitted {len(per_coeff)} per-coefficient scatters"
        )
    if rows:
        # Legend is written AFTER the loop so it can reference each subset's
        # emitted scatter PNG (the scatter paths are only known once drawn).
        written.append(
            _write_plot_series_legend(
                rows,
                plots_dir,
                scatter_paths_by_mode=scatter_paths_by_mode,
            )
        )
        index_path = _write_plot_index(plot_index_rows, plots_dir / "plot_index.csv")
        written.append(index_path)
    return written


STEERING_STRENGTH_FAMILY_PLOT_NAME = "steering_strength_families.png"
_STRENGTH_OUTLIER_IQR_MULTIPLIER = 1.5

_STEERING_STRENGTH_REQUIRED_COLUMNS = {
    "mode",
    "method_name",
    "target_model",
    "coefficient_str",
    "adversarial_strength",
    "is_control",
    "general_val_bpb",
    "general_val_corpus",
}


@dataclass(frozen=True)
class _StrengthFamilyPoint:
    mode: str
    method_name: str
    target_model: str
    coefficient_str: str
    coefficient_x: float
    metric_y: float
    general_val_bpb: float | None
    general_val_bpb_source: str
    n_source_rows: int

    def family_row(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "method_name": self.method_name,
            "target_model": self.target_model,
        }


def _normalize_strength_coefficient(value: Any) -> tuple[str, float]:
    if value is None or str(value) == "None":
        return "None", 0.0
    text = str(value)
    try:
        coefficient = float(text)
    except ValueError as exc:
        raise AxBenchBpbAdvError(
            f"steering-strength plot requires numeric coefficient_str or None, got {text!r}"
        ) from exc
    if not math.isfinite(coefficient):
        raise AxBenchBpbAdvError(
            f"steering-strength plot requires finite coefficient_str, got {text!r}"
        )
    return text, coefficient


def _require_strength_columns(macro: pa.Table, *, metric_column: str) -> None:
    present = set(macro.schema.names)
    missing = sorted((_STEERING_STRENGTH_REQUIRED_COLUMNS | {metric_column}) - present)
    if missing:
        raise AxBenchBpbAdvError(
            f"steering-strength plot macro is missing required column(s): {missing}"
        )


def _control_rows_for_strength_plot(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        context = f"steering-strength macro row {idx}"
        strength = _require_finite_float(
            row.get("adversarial_strength"),
            field="adversarial_strength",
            context=context,
        )
        if strength != 0.0:
            continue
        if not bool(row.get("is_control")):
            raise AxBenchBpbAdvError(
                "steering-strength plot found adversarial_strength=0.0 with "
                f"is_control=false for method={row.get('method_name')!r}, "
                f"coefficient={row.get('coefficient_str')!r}"
            )
        controls.append(row)
    if not controls:
        raise AxBenchBpbAdvError(
            "steering-strength plot found no no-adversary control rows "
            "(adversarial_strength == 0.0)"
        )
    return controls


def _strength_family_points(
    rows: list[dict[str, Any]],
    *,
    metric_column: str,
) -> list[_StrengthFamilyPoint]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    coefficient_x_by_key: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        coefficient_str, coefficient_x = _normalize_strength_coefficient(row.get("coefficient_str"))
        key = (
            str(row["mode"]),
            str(row["method_name"]),
            str(row["target_model"]),
            coefficient_str,
        )
        grouped.setdefault(key, []).append(row)
        coefficient_x_by_key[key] = coefficient_x

    points: list[_StrengthFamilyPoint] = []
    for key, group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            item[0][2],
            _coefficient_sort_key(item[0][3]),
        ),
    ):
        metric_values = [
            _require_finite_float(
                row.get(metric_column),
                field=metric_column,
                context=(f"steering-strength plot {key[1]} coeff={key[3]} mode={key[0]}"),
            )
            for row in group_rows
        ]
        n_null = sum(1 for row in group_rows if row.get("general_val_bpb") is None)
        if 0 < n_null < len(group_rows):
            raise AxBenchBpbAdvError(
                f"steering-strength plot group method={key[1]!r} coeff={key[3]!r} "
                f"mode={key[0]!r} mixes null and non-null general_val_bpb across "
                f"{len(group_rows)} concatenated rows; a gv-enabled run cannot be "
                "averaged with a gv-less run of the same configuration"
            )
        general_values = [
            _require_finite_float(
                row.get("general_val_bpb"),
                field="general_val_bpb",
                context=(f"steering-strength plot {key[1]} coeff={key[3]} mode={key[0]}"),
            )
            for row in group_rows
            if row.get("general_val_bpb") is not None
        ]
        points.append(
            _StrengthFamilyPoint(
                mode=key[0],
                method_name=key[1],
                target_model=key[2],
                coefficient_str=key[3],
                coefficient_x=coefficient_x_by_key[key],
                metric_y=_mean(metric_values),
                general_val_bpb=_mean(general_values) if general_values else None,
                general_val_bpb_source="measured" if general_values else "missing",
                n_source_rows=len(group_rows),
            )
        )
    return points


def _require_frontier_x_for_every_point(
    points: list[_StrengthFamilyPoint],
) -> None:
    """Every family drawn on the steerability frontier must have a capability
    x-coordinate (``general_val_bpb``).

    The frontier plots steerability against ``general_val_bpb``; a datapoint with
    no ``general_val_bpb`` has no x and cannot be placed. Rather than silently
    omit such a family (which would let a method vanish from the capability axis
    while others are shown — an invalid comparison), this raises and names the
    offenders. Roofline methods are just one case: they now carry a real,
    per-document general_val_bpb (``general_val_roofline``), so a null value means
    the general_val stage did not score them, not that roofline is unimplemented.
    """
    missing = sorted({point.method_name for point in points if point.general_val_bpb is None})
    if missing:
        raise AxBenchBpbAdvError(
            "steering-strength frontier: every plotted family must have a "
            f"general_val_bpb x-coordinate, but {missing} have none. Run "
            "score_general_val for these methods (roofline methods included) or "
            "drop them from the plot — a capability frontier cannot place a "
            "datapoint without a capability x."
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise AxBenchBpbAdvError("cannot compute a percentile of an empty list")
    if q < 0.0 or q > 1.0:
        raise AxBenchBpbAdvError(f"percentile q must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _iqr_bounds(values: list[float]) -> tuple[float, float]:
    if len(values) < 4:
        return (-math.inf, math.inf)
    q1 = _percentile(values, 0.25)
    q3 = _percentile(values, 0.75)
    iqr = q3 - q1
    if iqr <= 0.0:
        return (-math.inf, math.inf)
    pad = _STRENGTH_OUTLIER_IQR_MULTIPLIER * iqr
    return (q1 - pad, q3 + pad)


def _exclude_strength_outliers(
    points: list[_StrengthFamilyPoint],
) -> list[_StrengthFamilyPoint]:
    plottable = [point for point in points if point.general_val_bpb is not None]
    if len(plottable) < 4:
        return points
    general_val_values: list[float] = []
    for point in plottable:
        if point.general_val_bpb is None:
            raise AxBenchBpbAdvError("internal error: plottable point has no general_val_bpb")
        general_val_values.append(float(point.general_val_bpb))
    x_min, x_max = _iqr_bounds(general_val_values)
    y_min, y_max = _iqr_bounds([float(point.metric_y) for point in plottable])

    kept: list[_StrengthFamilyPoint] = []
    for point in points:
        if point.general_val_bpb is None:
            kept.append(point)
            continue
        x = float(point.general_val_bpb)
        y = float(point.metric_y)
        if x_min <= x <= x_max and y_min <= y <= y_max:
            kept.append(point)

    if not any(point.general_val_bpb is not None for point in kept):
        raise AxBenchBpbAdvError(
            "outlier exclusion removed every plottable steering-strength point"
        )
    dropped = len(points) - len(kept)
    if dropped:
        logger.info(
            "steering-strength plot excluded {} outlier point(s) using Tukey "
            "fences: x=[{:.6g}, {:.6g}], y=[{:.6g}, {:.6g}]",
            dropped,
            x_min,
            x_max,
            y_min,
            y_max,
        )
    return kept


def _group_strength_points_by_series(
    points: list[_StrengthFamilyPoint],
) -> dict[tuple[str, str, str], list[_StrengthFamilyPoint]]:
    grouped: dict[tuple[str, str, str], list[_StrengthFamilyPoint]] = {}
    for point in points:
        grouped.setdefault(
            (point.mode, point.method_name, point.target_model),
            [],
        ).append(point)
    for series_points in grouped.values():
        series_points.sort(key=lambda point: _coefficient_sort_key(point.coefficient_str))
    return grouped


def _write_strength_family_legend(
    points: list[_StrengthFamilyPoint],
    plots_dir: Path,
) -> Path:
    color_map = _family_color_map([point.family_row() for point in points])
    legend_rows: list[dict[str, Any]] = []
    for series_id, ((_mode, _method, _target), series_points) in enumerate(
        sorted(_group_strength_points_by_series(points).items(), key=lambda item: item[0]),
        start=1,
    ):
        first = series_points[0]
        family = _model_steering_family(first.family_row())
        coefficients = [point.coefficient_str for point in series_points]
        general_val_sources = sorted(
            {point.general_val_bpb_source for point in series_points}
        )
        legend_rows.append(
            {
                "series_id": f"S{series_id:04d}",
                "model_steering_family": family,
                "color": mcolors.to_hex(color_map[family]),
                "mode": first.mode,
                "marker": _series_marker(first.method_name),
                "method_name": first.method_name,
                "target_model": first.target_model,
                "n_coefficients": len(series_points),
                "coefficient_min": coefficients[0],
                "coefficient_max": coefficients[-1],
                "n_source_rows": sum(point.n_source_rows for point in series_points),
                "has_general_val_bpb": all(
                    point.general_val_bpb is not None for point in series_points
                ),
                "general_val_bpb_source": ";".join(general_val_sources),
                "plot_path": str(Path(first.mode) / STEERING_STRENGTH_FAMILY_PLOT_NAME),
            }
        )
    out_path = plots_dir / "plot_series_legend.csv"
    _write_csv(legend_rows, out_path)
    return out_path


def _plot_strength_family_mode(
    points: list[_StrengthFamilyPoint],
    out_path: Path,
    *,
    metric_label: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    color_map = _family_color_map([point.family_row() for point in points])
    by_series = _group_strength_points_by_series(points)
    ylabel = (
        "BPB class margin (higher = stronger)"
        if metric_label == PRIMARY_MARGIN_YLABEL
        else metric_label
    )

    fig, frontier_ax = plt.subplots(figsize=(7.0, 6.4))
    try:
        for (_mode, _method, _target), series_points in sorted(by_series.items()):
            first = series_points[0]
            family = _model_steering_family(first.family_row())
            color = color_map[family]
            marker = _series_marker(first.method_name)
            frontier_points = [
                point for point in series_points if point.general_val_bpb is not None
            ]
            if frontier_points:
                frontier_x = [
                    float(point.general_val_bpb)
                    for point in frontier_points
                    if point.general_val_bpb is not None
                ]
                frontier_ax.plot(
                    frontier_x,
                    [point.metric_y for point in frontier_points],
                    color=color,
                    marker=marker,
                    linewidth=1.25 if len(frontier_points) > 1 else 0.0,
                    linestyle="-" if len(frontier_points) > 1 else "None",
                    label=family,
                )

        frontier_ax.set_xlabel("general validation BPB")
        frontier_ax.set_ylabel(ylabel)
        frontier_ax.set_title("steerability frontier")
        frontier_ax.grid(True, alpha=0.25)
        # Every plotted family is guaranteed a general_val_bpb x by
        # _require_frontier_x_for_every_point (called before this), so there is no
        # "unavailable"/omitted-family branch here: a missing capability x raises
        # upstream rather than silently dropping a method from the frontier.

        handles, labels = frontier_ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.02),
                ncol=min(3, len(handles)),
                fontsize="small",
            )
        fig.suptitle(f"{points[0].mode} steering-strength families")
        fig.tight_layout(rect=(0.10, 0.20, 0.98, 0.94))
        fig.savefig(out_path, dpi=150)
    finally:
        plt.close(fig)
    return out_path


def plot_steering_strength_families(
    macro: pa.Table,
    plots_dir: Path,
    *,
    metric_column: str,
    metric_label: str,
    plot_methods: list[str] | None = None,
) -> list[Path]:
    """Plot per-method steerability frontier.

    This intentionally uses only no-adversary rows (``adversarial_strength==0``)
    so the x-axis is general validation BPB, not counter-steering pressure.
    Duplicate rows from concatenated runs are averaged per
    (mode, method, target, coefficient).
    """

    _require_strength_columns(macro, metric_column=metric_column)
    if plot_methods is not None and not plot_methods:
        raise AxBenchBpbAdvError(
            "plot_methods is an empty list, which keeps no methods and would clear "
            "all prior plots while emitting nothing; pass None to plot every method "
            "or list the methods to keep"
        )

    rows = macro.to_pylist()
    if plot_methods is not None:
        available = {str(row["method_name"]) for row in rows}
        unknown = sorted(set(plot_methods) - available)
        if unknown:
            raise AxBenchBpbAdvError(
                f"plot_methods contains methods absent from the aggregates: "
                f"{unknown}; available methods: {sorted(available)}"
            )
        keep = set(plot_methods)
        rows = [row for row in rows if str(row["method_name"]) in keep]

    _require_single_general_val_corpus(rows, figure="steering_strength_families")

    control_rows = _control_rows_for_strength_plot(rows)
    points = _strength_family_points(control_rows, metric_column=metric_column)
    _require_frontier_x_for_every_point(points)
    points = _exclude_strength_outliers(points)
    if not points:
        raise AxBenchBpbAdvError("steering-strength plot has no plottable points")

    _clear_previous_plot_outputs(plots_dir)
    written: list[Path] = []
    plot_index_rows: list[dict[str, Any]] = []
    for mode, mode_points in sorted(
        {point.mode: [p for p in points if p.mode == point.mode] for point in points}.items()
    ):
        out_path = _plot_strength_family_mode(
            mode_points,
            _plot_subset_dir(plots_dir, mode=mode) / STEERING_STRENGTH_FAMILY_PLOT_NAME,
            metric_label=metric_label,
        )
        written.append(out_path)
        plot_index_rows.append(
            {
                "path": str(out_path.relative_to(plots_dir)),
                "mode": mode,
                "plot_name": STEERING_STRENGTH_FAMILY_PLOT_NAME,
                "n_rows": len(mode_points),
                "n_source_rows": sum(point.n_source_rows for point in mode_points),
            }
        )

    written.append(_write_strength_family_legend(points, plots_dir))
    index_path = _write_plot_index(plot_index_rows, plots_dir / "plot_index.csv")
    written.append(index_path)
    return written


ROUNDTRIP_SCATTER_PLOT_NAME = "roundtrip_steerability_vs_baseline.png"


def plot_roundtrip_adv_summaries(
    macro: pa.Table,
    plots_dir: Path,
    *,
    scatter_baseline_x: str = ScatterBaselineX.BASELINE_LOSS.value,
    plot_methods: list[str] | None = None,
) -> list[Path]:
    if not macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA):
        raise AxBenchBpbAdvError(
            "expected roundtrip macro aggregation with ROUNDTRIP_MACRO_AGG_SCHEMA"
        )
    ScatterBaselineX.coerce(scatter_baseline_x)
    if plot_methods is not None and not plot_methods:
        raise AxBenchBpbAdvError(
            "plot_methods is an empty list, which keeps no methods and would clear "
            "all prior plots while emitting nothing; pass None to plot every method "
            "or list the methods to keep"
        )
    all_rows = macro.to_pylist()
    rows = list(all_rows)
    if plot_methods is not None:
        available = {str(row["method_name"]) for row in rows}
        unknown = sorted(set(plot_methods) - available)
        if unknown:
            raise AxBenchBpbAdvError(
                f"plot_methods contains methods absent from the aggregates: "
                f"{unknown}; available methods: {sorted(available)}"
            )
        keep = set(plot_methods)
        rows = [row for row in rows if str(row["method_name"]) in keep]
    # Issue #635: general_val_bpb is drawn on shared axes by the companion line
    # plot and (when selected) the scatters; one corpus per invocation.
    _require_single_general_val_corpus(rows, figure="plot_roundtrip_adv_summaries")
    # Fair feature-DiffMean baseline depends only on ``all_rows``; compute once
    # and reuse for every mode's representative selection below.
    roundtrip_target_scores = _feature_diffmean_target_scores(
        all_rows, baseline_metric="primary_roundtrip_y"
    )
    _require_representative_target_scores(
        rows,
        target_scores=roundtrip_target_scores,
        baseline_metric="primary_roundtrip_y",
    )
    _clear_previous_plot_outputs(plots_dir)
    written: list[Path] = []
    plot_index_rows: list[dict[str, Any]] = []
    scatter_paths_by_mode: dict[str, str] = {}
    diagnostic_specs: list[tuple[str, str, str, bool]] = [
        (
            "roundtrip_percentile_mean_vs_strength.png",
            "mean_roundtrip_percentile",
            "mean roundtrip activation percentile",
            False,
        ),
        (
            "roundtrip_post_act_mean_vs_strength.png",
            "mean_roundtrip_post_act",
            "mean roundtrip post-activation",
            False,
        ),
        (
            "roundtrip_pre_act_mean_vs_strength.png",
            "mean_roundtrip_pre_act",
            "mean roundtrip pre-activation",
            False,
        ),
        (
            "control_primary_roundtrip_y_vs_strength.png",
            "control_primary_roundtrip_y",
            ROUNDTRIP_CONTROL_LABEL,
            False,
        ),
    ]
    for mode, subset_rows in sorted(_rows_by_mode(rows).items()):
        subset_dir = _plot_subset_dir(plots_dir, mode=mode)
        representative_rows = _filter_rows_to_representative_coefficient(
            subset_rows,
            baseline_metric="primary_roundtrip_y",
            baseline_rows=all_rows,
            target_scores=roundtrip_target_scores,
        )
        primary_path = _plot_metric_vs_strength(
            representative_rows,
            subset_dir / "primary_roundtrip_y_vs_strength.png",
            metric="primary_roundtrip_y",
            ylabel=ROUNDTRIP_PRIMARY_YLABEL,
            mode=mode,
            draw_baseline_reference=True,
            baseline_metric="primary_roundtrip_y",
            baseline_label="no-adversary primary roundtrip y",
        )
        if primary_path is not None:
            written.append(primary_path)
            plot_index_rows.append(
                {
                    "path": str(primary_path.relative_to(plots_dir)),
                    "mode": mode,
                    "plot_name": "primary_roundtrip_y_vs_strength.png",
                    "n_rows": len(representative_rows),
                }
            )
        for name, metric, ylabel, draw_baseline in diagnostic_specs:
            path = _plot_metric_vs_strength(
                subset_rows,
                subset_dir / name,
                metric=metric,
                ylabel=ylabel,
                mode=mode,
                draw_baseline_reference=draw_baseline,
                baseline_metric="primary_roundtrip_y",
                baseline_label="no-adversary primary roundtrip y",
            )
            if path is not None:
                written.append(path)
                plot_index_rows.append(
                    {
                        "path": str(path.relative_to(plots_dir)),
                        "mode": mode,
                        "plot_name": name,
                        "n_rows": len(subset_rows),
                    }
                )
        scatter_path = _plot_series_grid(
            representative_rows,
            subset_dir / ROUNDTRIP_SCATTER_PLOT_NAME,
            x_getter=lambda row: _scatter_baseline_x(
                row,
                scatter_baseline_x=scatter_baseline_x,
            ),
            xlabel=_scatter_xlabel(scatter_baseline_x, mode=str(mode)),
            y_getter=lambda row: row.get("primary_roundtrip_y"),
            ylabel=ROUNDTRIP_PRIMARY_YLABEL,
            title="roundtrip steerability vs baseline",
            mode=mode,
            draw_baseline_reference=False,
            baseline_metric="primary_roundtrip_y",
            baseline_label="no-adversary primary roundtrip y",
        )
        if scatter_path is not None:
            written.append(scatter_path)
            scatter_relpath = str(scatter_path.relative_to(plots_dir))
            scatter_paths_by_mode[str(mode)] = scatter_relpath
            plot_index_rows.append(
                {
                    "path": scatter_relpath,
                    "mode": mode,
                    "plot_name": ROUNDTRIP_SCATTER_PLOT_NAME,
                    "n_rows": len(representative_rows),
                }
            )
    if rows:
        written.append(
            _write_plot_series_legend(
                rows,
                plots_dir,
                baseline_metric="control_primary_roundtrip_y",
                baseline_column_name="mean_control_primary_roundtrip_y",
                scatter_paths_by_mode=scatter_paths_by_mode,
            )
        )
        index_path = _write_plot_index(plot_index_rows, plots_dir / "plot_index.csv")
        written.append(index_path)
    return written
