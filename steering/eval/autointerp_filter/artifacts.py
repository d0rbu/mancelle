"""Canonical autointerp score finalization and output rendering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import yaml

from nanochat.atomic_io import atomic_write_text
from steering.eval.autointerp_filter.config import AutoInterpFilterConfig
from steering.eval.autointerp_filter.manual_filter import write_manual_feature_filter
from steering.eval.autointerp_filter.types import (
    AutoInterpFilterReport,
    TargetOutcome,
    TargetScore,
)


@dataclass(frozen=True)
class FinalizedScores:
    """Globally finalized scores and accepted feature IDs."""

    ordered_scores: tuple[TargetScore, ...]
    accepted_scores: tuple[TargetScore, ...]
    accepted_feature_ids: tuple[int, ...]


def passes_threshold(score: float, threshold: float | None) -> bool:
    """Return whether a concrete score passes the configured threshold.

    ``threshold`` is ``None`` only when the operator explicitly configures
    ``score_threshold: null`` (the shipped default is 0.6, and
    ``validate_config`` rejects malformed values), which disables score gating
    so acceptance is governed by ``top_k`` alone.
    """
    return threshold is None or score >= threshold


def _resolve_output_path(
    configured_path: str,
    *,
    output_dir: str,
    physical_output_root: Path | None,
) -> Path:
    """Map one configured logical output path onto the physical write root.

    ``physical_output_root`` is an internal combiner safety seam: configured
    paths remain the logical paths recorded in outputs, while writes are routed
    through an already anchored directory (for example ``/proc/self/fd/N``).
    """
    logical = Path(configured_path)
    if physical_output_root is None:
        return logical
    logical_root = Path(output_dir).resolve()
    try:
        relative = logical.resolve().relative_to(logical_root)
    except ValueError as exc:
        raise ValueError(
            f"configured output path {logical} is outside output_dir {logical_root}"
        ) from exc
    return physical_output_root / relative


def finalize_scores(
    scores: list[TargetScore],
    *,
    score_threshold: float | None,
    top_k: int | None,
) -> FinalizedScores:
    """Apply the canonical global threshold, top-k, outcome, and sort rules."""
    eligible = [
        score
        for score in scores
        if score.score is not None and passes_threshold(score.score, score_threshold)
    ]
    eligible.sort(
        key=lambda score: (
            -float(score.score if score.score is not None else -1.0),
            score.feature_ids,
            score.kind,
        )
    )
    if top_k is not None:
        eligible = eligible[:top_k]

    accepted_keys = {score.target_key for score in eligible}
    final_scores = [
        replace(
            score,
            outcome=(
                TargetOutcome.accepted
                if score.target_key in accepted_keys
                else TargetOutcome.rejected
            ),
        )
        if score.score is not None
        else score
        for score in scores
    ]
    ordered = sorted(
        final_scores,
        key=lambda score: (
            score.score is None,
            0.0 if score.score is None else -score.score,
            score.feature_ids,
            score.kind,
        ),
    )
    accepted_feature_ids = sorted(
        {feature_id for score in eligible for feature_id in score.feature_ids}
    )
    return FinalizedScores(
        ordered_scores=tuple(ordered),
        accepted_scores=tuple(eligible),
        accepted_feature_ids=tuple(accepted_feature_ids),
    )


def write_final_outputs(
    cfg: AutoInterpFilterConfig,
    *,
    scores: list[TargetScore],
    audit_rows: list[dict],
    n_targets_total: int | None = None,
    physical_output_root: Path | None = None,
) -> AutoInterpFilterReport:
    """Finalize and atomically write every canonical full-run artifact.

    See :func:`_resolve_output_path` for the ``physical_output_root`` seam.
    """
    output_path = partial(
        _resolve_output_path,
        output_dir=cfg.output_dir,
        physical_output_root=physical_output_root,
    )
    finalized = finalize_scores(
        scores,
        score_threshold=cfg.score_threshold,
        top_k=cfg.top_k,
    )
    ordered = list(finalized.ordered_scores)
    accepted_ids = list(finalized.accepted_feature_ids)

    jsonl = "".join(json.dumps(score_to_json(score), sort_keys=True) + "\n" for score in ordered)
    atomic_write_text(jsonl, output_path(cfg.target_scores_path))
    atomic_write_text(
        "".join(f"{feature_id}\n" for feature_id in accepted_ids),
        output_path(cfg.accepted_features_path),
    )
    limited_audit = audit_rows[: cfg.audit_sample_limit]
    atomic_write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in limited_audit),
        output_path(cfg.trial_audit_path),
    )
    atomic_write_text(
        dump_human_yaml([trial_audit_yaml_row(row) for row in limited_audit]),
        output_path(cfg.trial_audit_yaml_path),
    )
    yaml_payload = {
        "feature_dir": cfg.feature_dir,
        "score_threshold": cfg.score_threshold,
        "top_k": cfg.top_k,
        "accepted_feature_ids": accepted_ids,
        "targets": [score_to_ranked_yaml(score) for score in ordered],
    }
    atomic_write_text(
        dump_human_yaml(yaml_payload),
        output_path(cfg.ranked_yaml_path),
    )
    write_manual_feature_filter(
        scores=ordered,
        output_path=output_path(cfg.manual_feature_filter_path),
        meta_path=output_path(cfg.manual_feature_filter_meta_path),
        feature_dir=cfg.feature_dir,
        target_scores_path=cfg.target_scores_path,
        ranked_yaml_path=cfg.ranked_yaml_path,
        score_threshold=cfg.score_threshold,
        top_k=cfg.top_k,
        neuronpedia_model_id=cfg.neuronpedia.model_id,
        neuronpedia_sae_source=cfg.neuronpedia.sae_source,
        n_requested_trials=cfg.n_trials_per_target,
    )

    report = AutoInterpFilterReport(
        n_targets_total=(len(scores) if n_targets_total is None else n_targets_total),
        n_targets_scored=sum(1 for score in scores if score.score is not None),
        n_targets_accepted=len(finalized.accepted_scores),
        n_features_accepted=len(accepted_ids),
        score_threshold=cfg.score_threshold,
        top_k=cfg.top_k,
        target_scores_path=cfg.target_scores_path,
        ranked_yaml_path=cfg.ranked_yaml_path,
        accepted_features_path=cfg.accepted_features_path,
        manual_feature_filter_path=cfg.manual_feature_filter_path,
        manual_feature_filter_meta_path=cfg.manual_feature_filter_meta_path,
        trial_audit_path=cfg.trial_audit_path,
        trial_audit_yaml_path=cfg.trial_audit_yaml_path,
    )
    atomic_write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        output_path(cfg.report_output_path),
    )
    return report


def score_to_json(score: TargetScore) -> dict:
    """Render one score using the stable persisted JSON schema."""
    payload = asdict(score)
    payload["kind"] = score.kind.value
    payload["outcome"] = score.outcome.value
    payload["feature_ids"] = list(score.feature_ids)
    payload["descriptions"] = {
        str(feature_id): description for feature_id, description in score.descriptions.items()
    }
    return payload


class _HumanReadableDumper(yaml.SafeDumper):
    """Local YAML dumper that keeps multiline prompts readable."""


def _represent_human_string(dumper, value):
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_HumanReadableDumper.add_representer(str, _represent_human_string)


def dump_human_yaml(payload: object) -> str:
    """Render deterministic, human-readable YAML without global side effects."""
    return yaml.dump(
        payload,
        Dumper=_HumanReadableDumper,
        sort_keys=False,
        allow_unicode=False,
    )


def score_to_ranked_yaml(score: TargetScore) -> dict:
    """Render one concise ranked-YAML target row."""
    payload = {
        "description": ranked_description(score),
        "feature_ids": list(score.feature_ids),
        "score": score.score,
    }
    if score.error_message is not None:
        payload["error_message"] = score.error_message
    return payload


def ranked_description(score: TargetScore) -> str | list[dict[str, int | str]] | None:
    """Render singleton or multi-feature description metadata."""
    if len(score.feature_ids) == 1:
        return score.descriptions.get(score.feature_ids[0])
    descriptions: list[dict[str, int | str]] = []
    for feature_id in score.feature_ids:
        description = score.descriptions.get(feature_id)
        if description:
            descriptions.append({"feature_id": feature_id, "description": description})
    return descriptions or None


def trial_audit_yaml_row(row: dict) -> dict:
    """Render the concise human-facing projection of one detailed audit row."""
    return {
        "display_name": row["display_name"],
        "feature_ids": row["feature_ids"],
        "quantile": row["quantile"],
        "prediction": row["prediction"],
        "correct_index": row["correct_index"],
        "response": row["response"],
        "final_user_prompt_sent_to_judge": row["final_user_prompt_sent_to_judge"],
        "examples": [
            {
                "text": example["text"],
                "max_activation": example["max_activation"],
            }
            for example in row["examples"]
        ],
    }
