"""Steering signals, exact blend policies, and their fitted-signal cache.

Callers pass ``consolidation.resolved.descriptor.content_sha256`` as the
``cluster_descriptor_sha`` contract component.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

import torch as t
from loguru import logger
from safetensors.torch import load_file, save_file

from scripts.chat_tui.setup import resolve_stats_sha
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
    SelfCondScorePolicy,
    rank_masked_candidate_pools,
)
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.eval.axbench_upstream.methods import (
    MethodSetup,
    self_conditioning_masked_pool_limit,
)
from steering.refined_types import PositiveInt, UnitInterval
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY

STRENGTH_BUDGET = 6.0
BUDGET_EPSILON = 1e-9
FITTER_VERSION = "diffmean_pct_v1"


class SteeringMethod(StrEnum):
    """User-selectable dense steering behavior."""

    OFF = "off"
    INTERPOLATION = "interpolation"
    RESTRICTION = "restriction"


class SignalSource(StrEnum):
    """Producer identity for a dense steering signal."""

    AXBENCH_CONCEPT = "axbench_concept"


@dataclass(frozen=True)
class SteeringSignal:
    """One validated dense target in feature-prediction-head row order."""

    name: str
    target_scores: t.Tensor
    source: SignalSource
    identity: str

    def __post_init__(self) -> None:
        scores = self.target_scores
        if not isinstance(scores, t.Tensor) or scores.ndim != 1:
            raise ValueError("target_scores must be a rank-1 tensor")
        if scores.dtype is not t.float32:
            raise ValueError("target_scores must have dtype float32")
        if not bool(t.isfinite(scores).all()):
            raise ValueError("target_scores must be finite")
        if not bool(((scores >= -1.0) & (scores <= 1.0)).all()):
            raise ValueError("target_scores must lie in [-1, 1]")


@dataclass(frozen=True)
class SteeringSelection:
    """Steering method and its enabled signal-strength pairs."""

    method: SteeringMethod
    enabled: tuple[tuple[SteeringSignal, float], ...]

    def __post_init__(self) -> None:
        seen_identities: set[str] = set()
        for signal, strength in self.enabled:
            if signal.identity in seen_identities:
                raise ValueError(f"duplicate steering signal identity {signal.identity!r}")
            seen_identities.add(signal.identity)
            if (
                isinstance(strength, bool)
                or not isinstance(strength, (int, float))
                or not math.isfinite(strength)
                or not 0.0 <= strength <= STRENGTH_BUDGET
            ):
                raise ValueError(
                    f"strength for steering signal {signal.name!r} must be finite "
                    f"and in [0, {STRENGTH_BUDGET:g}], got {strength!r}"
                )


def _zero_policy(
    *,
    head_width: int,
    negative_selection: NegativeSamplingSelection,
) -> SelfCondScoreInterpolation:
    return SelfCondScoreInterpolation(
        target_scores=t.zeros(head_width, dtype=t.float32),
        negative_selection=negative_selection,
        weight=UnitInterval.parse(0.0),
    )


def build_score_policy(
    selection: SteeringSelection,
    *,
    positive_selection: SamplingSelection,
    negative_selection: NegativeSamplingSelection,
    head_width: int,
    feature_mask: FeatureMask | None = None,
    head_row_original_ids: t.Tensor | None = None,
) -> SelfCondScorePolicy:
    """Build the provider policy for one complete steering selection."""

    for signal, _strength in selection.enabled:
        signal_width = int(signal.target_scores.shape[0])
        if signal_width != head_width:
            raise ValueError(
                f"steering signal {signal.name!r} has target width {signal_width}, "
                f"but the feature-prediction head width is {head_width}"
            )

    if selection.method is SteeringMethod.OFF or not selection.enabled:
        return _zero_policy(
            head_width=head_width,
            negative_selection=negative_selection,
        )

    if selection.method is SteeringMethod.INTERPOLATION:
        total = math.fsum(strength for _signal, strength in selection.enabled)
        if total > STRENGTH_BUDGET + BUDGET_EPSILON:
            entries = ", ".join(
                f"({signal.name!r}, {strength!r})" for signal, strength in selection.enabled
            )
            raise ValueError(
                f"interpolation strength budget exceeded by [{entries}]: total={total!r}, "
                f"budget={STRENGTH_BUDGET!r}"
            )
        if abs(total) <= BUDGET_EPSILON:
            return _zero_policy(
                head_width=head_width,
                negative_selection=negative_selection,
            )
        if abs(total - STRENGTH_BUDGET) <= BUDGET_EPSILON:
            total = STRENGTH_BUDGET

        if len(selection.enabled) == 1:
            signal, _strength = selection.enabled[0]
            # Do not compute float32 (f * d) / f: it is not bit-identical to d.
            target_scores = signal.target_scores.clone()
        else:
            accumulator = t.zeros(head_width, dtype=t.float64)
            for signal, strength in selection.enabled:
                accumulator += signal.target_scores.to(dtype=t.float64) * strength
            target_scores = (accumulator / total).to(dtype=t.float32)

        return SelfCondScoreInterpolation(
            target_scores=target_scores,
            negative_selection=negative_selection,
            weight=UnitInterval.parse(total / STRENGTH_BUDGET),
        )

    original_ids = (
        t.arange(head_width, dtype=t.int64)
        if head_row_original_ids is None
        else head_row_original_ids.to(dtype=t.int64, device="cpu")
    )
    if original_ids.ndim != 1 or int(original_ids.numel()) != head_width:
        raise ValueError(
            "head_row_original_ids must align one-to-one with feature-prediction head rows"
        )
    keep_rows = (
        None
        if feature_mask is None
        else feature_mask.keep_mask(original_ids).to(dtype=t.bool, device="cpu")
    )
    n_candidates = head_width if keep_rows is None else int(keep_rows.sum().item())
    if n_candidates <= 0:
        raise ValueError("chat TUI restriction has no eligible head rows after the feature mask")

    positive_rows = t.ones(head_width, dtype=t.bool)
    negative_rows = t.ones(head_width, dtype=t.bool)
    names: list[str] = []
    for signal, strength in selection.enabled:
        target = signal.target_scores.to(device="cpu")
        signal_positive, signal_negative = rank_masked_candidate_pools(
            target,
            positive_pool_limit=self_conditioning_masked_pool_limit(
                strength,
                n_candidates=n_candidates,
                mean=float(positive_selection.mean),
            ),
            negative_pool_limit=self_conditioning_masked_pool_limit(
                strength,
                n_candidates=n_candidates,
                mean=float(negative_selection.mean),
            ),
            keep_rows=keep_rows,
            positive_min_k=int(positive_selection.min_k),
            negative_min_k=int(negative_selection.min_k),
            owner=f"chat TUI restriction signal {signal.name!r}",
        )
        positive_rows &= signal_positive
        negative_rows &= signal_negative
        names.append(signal.name)

    target_scores = t.zeros(head_width, dtype=t.float32)
    target_scores[positive_rows] = 1.0
    target_scores[negative_rows] = -1.0
    owner = f"chat TUI restriction blend [{', '.join(names)}]"
    rank_masked_candidate_pools(
        target_scores,
        positive_pool_limit=head_width,
        negative_pool_limit=head_width,
        keep_rows=keep_rows,
        positive_min_k=int(positive_selection.min_k),
        negative_min_k=int(negative_selection.min_k),
        owner=owner,
    )
    return SelfCondScoreMasking(
        target_scores=target_scores,
        negative_selection=negative_selection,
        positive_pool_limit=PositiveInt.parse(head_width),
        negative_pool_limit=PositiveInt.parse(head_width),
    )


class SignalStore:
    """One-file-per-concept cache scoped by a fitted-signal contract digest."""

    def __init__(self, cache_dir: Path, contract_digest: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._contract_digest = contract_digest

    def get_or_fit(
        self,
        concept_id: int,
        text_digest: str,
        fitter: Callable[[], t.Tensor],
        *,
        name: str,
        source: SignalSource,
    ) -> SteeringSignal:
        """Load one signal, or fit and atomically publish a cache miss."""

        stem = f"{concept_id}_{text_digest}"
        identity = f"{self._contract_digest}/{stem}"
        path = self._cache_dir / self._contract_digest / f"{stem}.safetensors"
        if path.exists():
            try:
                return SteeringSignal(
                    name=name,
                    target_scores=load_file(str(path))["target_scores"],
                    source=source,
                    identity=identity,
                )
            except Exception as exc:
                logger.warning(
                    "Unreadable steering signal cache {}; refitting: {}",
                    path,
                    exc,
                )

        signal = SteeringSignal(
            name=name,
            target_scores=fitter(),
            source=source,
            identity=identity,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            save_file(
                {"target_scores": signal.target_scores.contiguous()},
                str(temporary_path),
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return signal


def _compute_contract_digest(
    *,
    feature_mask: FeatureMask | None,
    cluster_descriptor_sha: str | None,
    vocab_fingerprint: str | None,
    n_sae_features: int,
    stats_sha256: str,
) -> str:
    """Return the 16-hex fitted-signal contract digest."""

    payload = {
        "mask": feature_mask.canonical_dict() if feature_mask else None,
        "clusters": cluster_descriptor_sha,
        "vocab": vocab_fingerprint or f"full_{n_sae_features}",
        "stats": stats_sha256,
        "fitter": FITTER_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()[:16]


def contract_digest_for_setup(setup: MethodSetup) -> str:
    """Derive the complete fitted-signal cache identity from ``setup``."""

    cluster_descriptor_sha = (
        str(setup.consolidation.resolved.descriptor.content_sha256)
        if setup.consolidation is not None
        else None
    )
    vocab_fingerprint = (
        setup.feature_vocab.fingerprint() if setup.feature_vocab is not None else None
    )
    digest = _compute_contract_digest(
        feature_mask=setup.feature_mask,
        cluster_descriptor_sha=cluster_descriptor_sha,
        vocab_fingerprint=vocab_fingerprint,
        n_sae_features=setup.n_sae_features,
        stats_sha256=resolve_stats_sha(setup),
    )
    if setup.segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        # Turn DiffMean consumes the lineage's missing-percentile policy. Keep
        # fixed cache keys byte-identical; segregate this different fit contract.
        payload = {"base": digest, "lineage": setup.meta[ONLINE_SAE_LINEAGE_META_KEY]}
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return digest
