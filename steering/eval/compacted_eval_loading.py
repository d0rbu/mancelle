"""Validated compacted-checkpoint loading for conditioned eval surfaces.

``conditions_within_vocab`` is a promise about downstream selection, not merely
a way to bypass the loader's deferral.  Keep the allowlist here so only evals
that validate every static target and context-cache row against the loaded
checkpoint vocabulary can declare that stance.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final

from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract,
    resolve_and_assert_model_feature_contract,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    feature_clusters_checkpoint_metadata,
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_mask import canonical_dict
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.prompt_coverage import PromptCoverageSummary

CHAT_EVAL_TURN_SURFACE: Final = "chat_eval_turn"
CHAT_EVAL_SELF_CONDITIONED_SURFACE: Final = "chat_eval_self_conditioned"
CONDITIONED_EVAL_SURFACES: Final[frozenset[str]] = frozenset(
    {"harmfulness_eval", "training_progression", CHAT_EVAL_TURN_SURFACE, CHAT_EVAL_SELF_CONDITIONED_SURFACE}
)

_MISSING_METADATA_LABEL = (
    "feature-universe metadata (feature_mask_canonical or complete feature-cluster metadata)"
)
_MISSING_METADATA_GUIDANCE = (
    "Re-export the checkpoint with the canonical mask/cluster metadata that "
    "defines its encoder feature rows."
)


class EvaluationConditioningMode(StrEnum):
    """Persisted discriminator for conditioned versus decoder-only evals."""

    CONDITIONS_WITHIN_VOCAB = "conditions_within_vocab"
    UNCONDITIONED = "unconditioned"


def _assert_conditioned_eval_surface(eval_name: str) -> None:
    if eval_name not in CONDITIONED_EVAL_SURFACES:
        allowed = ", ".join(sorted(CONDITIONED_EVAL_SURFACES))
        raise ValueError(
            f"conditioned compacted-checkpoint loading is allowed only for "
            f"these within-vocabulary eval surfaces: {allowed}; got "
            f"{eval_name!r}"
        )


def conditioned_eval_conditioning(eval_name: str) -> EvalConditioning:
    """Declare within-vocabulary conditioning for an allowlisted eval surface."""

    _assert_conditioned_eval_surface(eval_name)
    return EvalConditioning.conditions_within_vocab(eval_name)


def assert_conditioned_eval_model(
    model: Any,
    meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
    eval_name: str,
) -> CheckpointFeatureContract | None:
    """Bind a loaded conditioned model to its checkpoint's feature universe.

    Encoder-decoder models delegate the complete equality proof to the shared
    checkpoint contract; malformed, missing, or mismatched metadata therefore
    fails before any eval target is selected.

    A decoder-only model has no encoder feature universe, so there is no
    vocabulary to bind — but it still REJECTS static-cluster metadata, which is
    the behavior these surfaces had before they became cluster-aware. That is
    not vestigial strictness: a checkpoint whose metadata describes clustered
    conditioning has no business reaching a decoder-only build. Either the
    metadata is wrong, or the caller's encoder-decoder branch selection
    mis-routed a conditioned checkpoint — and the second case would run
    UNCONDITIONED generation and report the numbers as conditioned ones, with
    nothing downstream to catch it. Failing here is what keeps "no evaluator
    path silently drops conditioning" true.
    """

    _assert_conditioned_eval_surface(eval_name)
    if not isinstance(model, EncoderDecoderGPT):
        reject_feature_clusters_metadata(meta, f"{eval_name} decoder-only checkpoint")
        return None
    return resolve_and_assert_model_feature_contract(
        model,
        meta,
        checkpoint_dir=checkpoint_dir,
        surface=eval_name,
        missing_metadata_label=_MISSING_METADATA_LABEL,
        missing_metadata_guidance=_MISSING_METADATA_GUIDANCE,
    )


def conditioned_eval_feature_vocab(model: Any) -> FeatureVocab | None:
    """Return the vocabulary that a conditioned eval must select within."""

    if not isinstance(model, EncoderDecoderGPT):
        return None
    return model.feature_vocab


def conditioned_eval_cache_contract_metadata(
    contract: CheckpointFeatureContract,
) -> dict[str, Any]:
    """Return exact clustered-cache metadata without changing legacy caches."""

    if not contract.clustered:
        return {}
    return {
        "feature_mask_canonical": canonical_dict(contract.feature_mask),
        **feature_clusters_checkpoint_metadata(contract.resolved_clusters),
    }


def conditioning_provenance(
    model: Any,
    provider: SafetensorsContextFeatureProvider | None,
    coverage: PromptCoverageSummary | None,
    *,
    eval_name: str,
) -> dict[str, Any]:
    """Describe the conditioning that actually reached one eval run.

    Decoder-only evaluation is legitimately unconditioned and is recorded as
    such.  An encoder-decoder report is emitted only after a concrete provider
    and its exact prompt-coverage summary exist.  The vocabulary fingerprint is
    the checkpoint primitive's established ``metadata_dict`` form; report
    provenance must never introduce a second fingerprint schema.

    Deliberately do not inspect ``provider.expected_feature_vocab`` here.  That
    constructor guard validates the cache, but actual conditioning is proved by
    runtime provider lookups and encoder calls.  Conflating those contracts
    would make a test pass or fail based on a kwarg rather than behavior.
    """

    _assert_conditioned_eval_surface(eval_name)
    if not isinstance(model, EncoderDecoderGPT):
        if provider is not None or coverage is not None:
            raise ValueError(
                f"{eval_name}: decoder-only conditioning provenance requires "
                "provider=None and coverage=None"
            )
        return {
            "conditioning_mode": EvaluationConditioningMode.UNCONDITIONED.value,
            "eval_name": eval_name,
        }

    if provider is None:
        raise ValueError(
            f"{eval_name}: conditions_within_vocab provenance requires a "
            "context-cache provider; generation reached report construction "
            "without one"
        )
    if coverage is None:
        raise ValueError(
            f"{eval_name}: conditions_within_vocab provenance requires an exact "
            "prompt-coverage summary"
        )
    if coverage.surface != eval_name:
        raise ValueError(f"{eval_name}: prompt-coverage summary belongs to {coverage.surface!r}")
    if coverage.cached_row_count != len(provider):
        raise ValueError(
            f"{eval_name}: prompt-coverage cached_row_count changed after "
            f"validation: summary={coverage.cached_row_count}, provider={len(provider)}"
        )

    feature_vocab = conditioned_eval_feature_vocab(model)
    return {
        "conditioning_mode": (EvaluationConditioningMode.CONDITIONS_WITHIN_VOCAB.value),
        "eval_name": eval_name,
        "feature_vocab": (None if feature_vocab is None else feature_vocab.metadata_dict()),
        "context_cache": {
            "manifest_path": str(provider.manifest_path),
            "identity": provider.identity.as_dict(),
            "cached_rows": len(provider),
        },
        "coverage": {
            "expected_prompts": coverage.expected_prompt_count,
            "expected_unique_keys": coverage.expected_unique_keys,
            "extra_cached_rows": coverage.extra_cached_rows,
        },
    }


__all__ = [
    "CHAT_EVAL_SELF_CONDITIONED_SURFACE",
    "CHAT_EVAL_TURN_SURFACE",
    "CONDITIONED_EVAL_SURFACES",
    "EvaluationConditioningMode",
    "assert_conditioned_eval_model",
    "conditioned_eval_cache_contract_metadata",
    "conditioning_provenance",
    "conditioned_eval_conditioning",
    "conditioned_eval_feature_vocab",
]
