"""Fail-closed checkpoint eligibility for structured-SFT GRPO."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from steering.conditional.checkpoint_conditioning import (
    assert_pretrain_conditioning_recoverable,
    checkpoint_user_config,
    read_checkpoint_conditioning,
)
from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract,
    resolve_and_assert_model_feature_contract,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.selfcond_training import (
    resolve_recorded_num_buckets,
)
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_training import GRPOPolicyConfig
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingSelection,
    parse_feature_selection,
    parse_negative_selection,
)


@dataclass(frozen=True)
class GRPOCheckpointEligibility:
    """Resolved immutable policy and effective feature-universe contract."""

    policy: GRPOPolicyConfig
    feature_contract: CheckpointFeatureContract
    capture_layer: int
    cross_attention_layers: tuple[int, ...]
    feature_count: int


def _parse_checkpoint_counts(
    meta: Mapping[str, Any],
) -> tuple[SamplingSelection, NegativeSamplingSelection]:
    lineage = read_checkpoint_conditioning(meta)
    raw_positive = lineage.positive_selection
    if not isinstance(raw_positive, Mapping):
        raise SFTGRPOError(
            "checkpoint lineage does not recover the fixed geometric positive "
            "feature_selection distribution"
        )
    try:
        positive = parse_feature_selection(dict(raw_positive))
    except (TypeError, ValueError) as error:
        raise SFTGRPOError(
            f"checkpoint positive feature_selection is invalid: {error}"
        ) from error
    if not isinstance(positive, SamplingSelection):
        raise SFTGRPOError(
            "structured-SFT GRPO requires checkpoint-recorded sampling selection"
        )
    if not isinstance(lineage.negative_selection, Mapping):
        raise SFTGRPOError(
            "checkpoint lineage does not prove bucket 1 was trained as the negative bucket"
        )
    try:
        negative = parse_negative_selection(dict(lineage.negative_selection))
    except (TypeError, ValueError) as error:
        raise SFTGRPOError(
            f"checkpoint negative_selection is invalid: {error}"
        ) from error
    if negative is None:
        raise SFTGRPOError("checkpoint negative_selection unexpectedly resolved to null")
    return positive, negative


def validate_grpo_checkpoint_eligibility(
    model: Any,
    meta: Mapping[str, Any],
    *,
    checkpoint_dir: str,
    action_mode: ActionMode,
    seed: int,
    action_policy_version: int = 1,
) -> GRPOCheckpointEligibility:
    """Validate all model/lineage facts before data or optional SAE setup."""
    if not isinstance(meta, Mapping):
        raise SFTGRPOError("checkpoint metadata must be a mapping")
    if "encoder_config" not in meta or "xattn_config" not in meta:
        raise SFTGRPOError(
            "structured-SFT GRPO requires a conditioned-SFT-shaped checkpoint"
        )
    feature_head = getattr(model, "feature_pred_head", None)
    feature_config = getattr(model, "feature_pred_config", None)
    if feature_head is None or feature_config is None:
        raise SFTGRPOError("checkpoint must load a feature-prediction head and config")
    metadata_fp = checkpoint_user_config(meta).get("feature_prediction")
    if metadata_fp is None:
        metadata_fp = meta.get("feature_prediction_config")
    if not isinstance(metadata_fp, Mapping):
        raise SFTGRPOError(
            "feature-prediction lineage is not recoverable from checkpoint metadata"
        )
    decoder_config = getattr(model, "dec_config", None)
    decoder_layers = getattr(decoder_config, "n_layer", None)
    capture_layer = getattr(model, "_feature_pred_layer_idx", None)
    if not isinstance(decoder_layers, int) or decoder_layers <= 0:
        raise SFTGRPOError("loaded decoder does not expose a positive layer count")
    if isinstance(capture_layer, bool) or not isinstance(capture_layer, int):
        raise SFTGRPOError("loaded model does not expose an integer FP capture layer")
    if capture_layer != decoder_layers - 1:
        raise SFTGRPOError(
            "feature-prediction capture must be at the final decoder layer: "
            f"capture={capture_layer!r}, final={decoder_layers - 1}"
        )
    if FeaturePredLossMode(feature_config.loss_mode) is not FeaturePredLossMode.MAX_POOL:
        raise SFTGRPOError("feature-prediction pooling must be exactly max_pool")
    if metadata_fp.get("loss_mode") != FeaturePredLossMode.MAX_POOL.value:
        raise SFTGRPOError("checkpoint metadata does not record max_pool FP lineage")
    recorded_layer = metadata_fp.get("layer", -1)
    if recorded_layer not in {-1, decoder_layers - 1}:
        raise SFTGRPOError(
            "checkpoint metadata does not record final-layer FP capture"
        )
    cross_attention = getattr(model, "cross_attn_layers", None)
    if cross_attention is None:
        raise SFTGRPOError("loaded model does not expose cross-attention layers")
    try:
        cross_attention_layers = tuple(sorted(int(key) for key in cross_attention.keys()))
    except (AttributeError, TypeError, ValueError) as error:
        raise SFTGRPOError("loaded cross-attention layer map is invalid") from error
    if not cross_attention_layers or cross_attention_layers[0] > capture_layer:
        raise SFTGRPOError(
            "at least one cross-attention layer must be at or before FP capture"
        )

    assert_pretrain_conditioning_recoverable(
        meta,
        checkpoint_label="structured-SFT GRPO base checkpoint",
    )
    recorded_buckets = resolve_recorded_num_buckets(meta)
    encoder = getattr(model, "encoder", None)
    model_buckets = getattr(getattr(encoder, "config", None), "num_buckets", None)
    if recorded_buckets != 2 or model_buckets != 2:
        raise SFTGRPOError(
            "structured-SFT GRPO requires exactly two buckets in metadata and model; "
            f"got metadata={recorded_buckets!r}, model={model_buckets!r}"
        )
    conditioning = read_checkpoint_conditioning(meta)
    if conditioning.label_dropout is None:
        raise SFTGRPOError("checkpoint does not prove feature-label dropout lineage")
    if conditioning.label_dropout != 0.0:
        raise SFTGRPOError("checkpoint feature-label dropout must be zero")
    if conditioning.has_scheduled_dropout():
        raise SFTGRPOError("checkpoint conditioning_cooldown must not drop feature labels")
    if conditioning.next_segment_conditioning_offset is None:
        raise SFTGRPOError(
            "checkpoint does not prove next-segment feature-prediction offset lineage"
        )
    if conditioning.next_segment_conditioning_offset is not False:
        raise SFTGRPOError("next-segment feature-prediction offset is unsupported")

    positive, negative = _parse_checkpoint_counts(meta)
    feature_count = getattr(feature_head, "out_features", None)
    if not isinstance(feature_count, int) or feature_count <= 0:
        raise SFTGRPOError("feature-prediction action space must be non-empty")
    policy = GRPOPolicyConfig(
        action_mode=ActionMode(action_mode),
        positive_counts=positive,
        negative_counts=negative,
        seed=seed,
        action_policy_version=action_policy_version,
        pool_mode=FeaturePredLossMode.MAX_POOL,
    )
    policy.validate(feature_count=feature_count)

    try:
        feature_contract = resolve_and_assert_model_feature_contract(
            model,
            meta,
            checkpoint_dir=checkpoint_dir,
            surface="structured-SFT GRPO",
            missing_metadata_label="canonical feature-mask/cluster metadata",
            missing_metadata_guidance=(
                "Recover the exact mask and cluster lineage before RL; table height "
                "alone cannot identify compact feature rows."
            ),
        )
    except (TypeError, ValueError) as error:
        raise SFTGRPOError(f"checkpoint feature vocabulary is not exact: {error}") from error
    return GRPOCheckpointEligibility(
        policy=policy,
        feature_contract=feature_contract,
        capture_layer=capture_layer,
        cross_attention_layers=cross_attention_layers,
        feature_count=feature_count,
    )


__all__ = [
    "GRPOCheckpointEligibility",
    "validate_grpo_checkpoint_eligibility",
]
