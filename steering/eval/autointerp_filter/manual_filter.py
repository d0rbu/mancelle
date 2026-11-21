"""Manual review YAML rendering for autointerp feature filters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from nanochat.atomic_io import atomic_write_text
from steering.eval.autointerp_filter.types import SCHEMA_VERSION, TargetScore


@dataclass(frozen=True)
class ManualFeatureFilterMeta:
    feature_dir: str
    target_scores_path: str
    ranked_yaml_path: str
    score_threshold: float | None
    top_k: int | None
    n_targets: int
    neuronpedia_model_id: str | None
    neuronpedia_sae_source: str | None
    generated_at: str
    schema_version: int = SCHEMA_VERSION


def write_manual_feature_filter(
    *,
    scores: list[TargetScore],
    output_path: Path,
    meta_path: Path,
    feature_dir: str,
    target_scores_path: str,
    ranked_yaml_path: str,
    score_threshold: float | None,
    top_k: int | None,
    neuronpedia_model_id: str | None,
    neuronpedia_sae_source: str | None,
    n_requested_trials: int | None = None,
) -> None:
    atomic_write_text(
        render_manual_feature_filter_yaml(
            scores,
            neuronpedia_model_id=neuronpedia_model_id,
            neuronpedia_sae_source=neuronpedia_sae_source,
            n_requested_trials=n_requested_trials,
        ),
        output_path,
    )
    meta = ManualFeatureFilterMeta(
        feature_dir=feature_dir,
        target_scores_path=target_scores_path,
        ranked_yaml_path=ranked_yaml_path,
        score_threshold=score_threshold,
        top_k=top_k,
        n_targets=len(scores),
        neuronpedia_model_id=neuronpedia_model_id,
        neuronpedia_sae_source=neuronpedia_sae_source,
        generated_at=datetime.now(UTC).isoformat(),
    )
    atomic_write_text(
        json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n",
        meta_path,
    )


def render_manual_feature_filter_yaml(
    scores: list[TargetScore],
    *,
    neuronpedia_model_id: str | None,
    neuronpedia_sae_source: str | None,
    n_requested_trials: int | None = None,
) -> str:
    lines: list[str] = []
    feature_id_width = _feature_id_width(scores)
    for score in sorted(scores, key=_manual_score_sort_key):
        feature_id = score.feature_ids[0]
        feature_text = f"{feature_id:0{feature_id_width}d}"
        score_text = "null" if score.score is None else f"{score.score:.3f}"
        url = _neuronpedia_url(
            feature_id,
            model_id=neuronpedia_model_id,
            sae_source=neuronpedia_sae_source,
        )
        trial_note = _manual_trial_note(score, n_requested_trials=n_requested_trials)
        url_suffix = f" {url}" if url is not None else ""
        lines.append(f'# - "{feature_text}"  # score={score_text}{url_suffix}{trial_note}')
    return "\n".join(lines) + ("\n" if lines else "")


def _feature_id_width(scores: list[TargetScore]) -> int:
    max_feature_id = max(
        (feature_id for score in scores for feature_id in score.feature_ids),
        default=0,
    )
    return max(1, len(str(max_feature_id)))


def _manual_trial_note(
    score: TargetScore,
    *,
    n_requested_trials: int | None,
) -> str:
    if (
        score.score is None
        or n_requested_trials is None
        or score.n_trials <= 0
        or score.n_trials >= n_requested_trials
    ):
        return ""
    return f" trials={score.n_trials}/{n_requested_trials} partial"


def _manual_score_sort_key(score: TargetScore) -> tuple[bool, float, int]:
    feature_id = score.feature_ids[0] if score.feature_ids else -1
    if score.score is None:
        return True, 0.0, int(feature_id)
    return False, -float(score.score), int(feature_id)


def _neuronpedia_url(
    feature_id: int,
    *,
    model_id: str | None,
    sae_source: str | None,
) -> str | None:
    if model_id is None or sae_source is None:
        return None
    return f"https://www.neuronpedia.org/{model_id}/{sae_source}/{feature_id}"
