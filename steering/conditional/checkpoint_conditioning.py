"""Read the conditioning contract a checkpoint was TRAINED under.

Encoder-decoder checkpoints are only interpretable together with the
conditioning scheme their encoder saw during pretraining. Three properties of
that scheme survive only in checkpoint metadata:

``negative_selection``
    When set, pretraining discretised positives into the reduced
    ``effective_buckets = num_buckets - 1`` space and reserved bucket
    ``num_buckets - 1`` as the "this feature is inactive" row (see
    ``steering.conditional.feature_pipeline.build_feature_pipeline`` and the
    negative writer in ``steering/packed_batch_io.py``). Any consumer that
    re-encodes features for such a checkpoint MUST reduce identically or it
    emits positives into the inactive row.

``feature_delta`` / ``feature_selection.delta_weights``
    When set, pretraining ranked candidate positives by a delta signal
    (``percentile_delta`` / ``splice_counterfactual``) rather than by plain
    activation percentile, and derived its negatives from the same delta
    channel. ``assert_conditioning_supported`` below applies the HARDCODED
    policy for what conditioned SFT does about that — it is deliberately not a
    config knob, so behaviour is identical for every operator and every run.

This module lives under ``steering/conditional/`` rather than
``steering/eval/``: it describes a *training-time* contract read by training,
SFT, and eval alike. A training script must not have to import the eval package
to learn how its own checkpoint was conditioned.

Metadata layout
---------------
Pretraining checkpoints record the conditioning under ``meta["user_config"]``.
Conditioned-SFT checkpoints have their own (SFT) ``user_config``, which carries
no pretraining conditioning at all — so an SFT checkpoint used as the base of a
chained SFT run (safety -> capabilities) would otherwise look unconditioned.
SFT therefore re-publishes what it inherited under the top-level
``pretrain_conditioning`` key, and the reader below falls back to it. No
pre-existing artifact carries that key, so every historical checkpoint reads
exactly as it did before this module existed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

from steering.conditional.delta import DeltaMode
from steering.conditional.encoder_decoder.conditioning_cooldown import (
    ConditioningCooldown,
    parse_conditioning_cooldown,
)

# Top-level metadata key under which a conditioned-SFT checkpoint re-publishes
# the pretraining conditioning contract it inherited, so chained SFT runs read
# the same contract the first link read.
PRETRAIN_CONDITIONING_META_KEY = "pretrain_conditioning"

# Hardcoded policy (NOT configurable): the delta modes conditioned SFT may run
# against, and what it does instead. ``splice_counterfactual`` is the only mode
# whose negatives have a defensible SFT analogue — pretraining drew them from a
# spliced counterfactual completion, SFT draws them randomly from the same
# ``negative_selection`` distribution. ``prev_segment`` and ``cum_max`` define
# the positives THEMSELVES as a cross-segment difference; SFT has no analogue at
# all, so they raise.
_DELTA_MODES_SFT_MAY_APPROXIMATE = frozenset({DeltaMode.splice_counterfactual})


def checkpoint_user_config(meta: Mapping | None) -> Mapping:
    """Return the best-effort user-config mapping for checkpoint metadata."""
    if meta is None:
        return {}
    raw_user_config = meta.get("user_config")
    if isinstance(raw_user_config, Mapping):
        return raw_user_config
    return meta


def _pretrain_conditioning_block(meta: Mapping | None) -> Mapping:
    if meta is None:
        return {}
    raw = meta.get(PRETRAIN_CONDITIONING_META_KEY)
    return raw if isinstance(raw, Mapping) else {}


def _read_conditioning_cooldown(raw: Any) -> ConditioningCooldown | None:
    """Read the resolved schedule; configuration alone is not provenance."""
    if raw is None:
        return None
    return ConditioningCooldown.from_metadata(raw)


@dataclass(frozen=True)
class CheckpointConditioning:
    """The pretraining conditioning contract recorded in checkpoint metadata."""

    negative_selection: Any
    feature_delta: Any
    delta_weights: str | None
    positive_selection: Any = None
    label_dropout: Any = None
    next_segment_conditioning_offset: Any = None
    conditioning_cooldown: Mapping[str, Any] | None = None

    def has_scheduled_dropout(self) -> bool:
        """Whether the recorded schedule permits any dropped conditioning."""
        recorded = _read_conditioning_cooldown(self.conditioning_cooldown)
        return (
            recorded is not None and min(recorded.start_fraction, recorded.end_fraction) < 1.0
        )

    def delta_mode(self) -> DeltaMode | None:
        """The pretraining ``feature_delta.mode``, or None when undelta'd.

        Raises rather than guessing when the recorded mode is not a known
        ``DeltaMode`` — an unreadable contract must never be silently treated as
        "no delta", which is precisely the failure this module exists to stop.
        """
        if self.feature_delta is None:
            return None
        if not isinstance(self.feature_delta, Mapping):
            raise ValueError(
                f"checkpoint feature_delta must be a mapping, got "
                f"{type(self.feature_delta).__name__}: {self.feature_delta!r}"
            )
        raw_mode = self.feature_delta.get("mode")
        try:
            return DeltaMode(raw_mode)
        except ValueError:
            supported = sorted(mode.value for mode in DeltaMode)
            raise ValueError(
                f"checkpoint feature_delta.mode={raw_mode!r} is not a known "
                f"delta mode. Supported: {supported}."
            ) from None

    def metadata_dict(self) -> dict[str, Any]:
        """The provenance block a chained SFT checkpoint must re-publish.

        Returns an empty dict when the checkpoint carries no conditioning
        contract, so an unconditioned lineage's saved metadata stays
        byte-identical to what it was before this key existed.
        """
        recorded = {
            "negative_selection": self.negative_selection,
            "feature_delta": self.feature_delta,
            "delta_weights": self.delta_weights,
        }
        if self.positive_selection is not None:
            recorded["positive_selection"] = self.positive_selection
        if self.label_dropout is not None:
            recorded["label_dropout"] = self.label_dropout
        if self.conditioning_cooldown is not None:
            recorded["conditioning_cooldown"] = dict(self.conditioning_cooldown)
        if self.next_segment_conditioning_offset is not None:
            recorded["next_segment_conditioning_offset"] = self.next_segment_conditioning_offset
        if all(value is None for value in recorded.values()):
            return {}
        return {
            PRETRAIN_CONDITIONING_META_KEY: {
                key: (dict(value) if isinstance(value, Mapping) else value)
                for key, value in recorded.items()
            }
        }


def read_checkpoint_conditioning(meta: Mapping | None) -> CheckpointConditioning:
    """Read the training-time conditioning contract out of checkpoint metadata.

    Reads ``meta["user_config"]`` first (pretraining checkpoints), then falls
    back to the top-level ``pretrain_conditioning`` block that conditioned SFT
    re-publishes for its own chained descendants.
    """
    user_config = checkpoint_user_config(meta)
    inherited = _pretrain_conditioning_block(meta)

    negative_selection = user_config.get("negative_selection")
    if negative_selection is None:
        negative_selection = inherited.get("negative_selection")

    positive_selection = user_config.get("feature_selection")
    if positive_selection is None:
        positive_selection = inherited.get("positive_selection")

    feature_delta = user_config.get("feature_delta")
    if feature_delta is None:
        feature_delta = inherited.get("feature_delta")

    feature_selection = user_config.get("feature_selection")
    delta_weights = (
        feature_selection.get("delta_weights") if isinstance(feature_selection, Mapping) else None
    )
    if delta_weights is None:
        delta_weights = inherited.get("delta_weights")

    label_dropout = user_config.get("label_dropout")
    if label_dropout is None:
        label_dropout = inherited.get("label_dropout")

    next_segment_conditioning_offset = user_config.get("next_segment_conditioning_offset")
    if next_segment_conditioning_offset is None:
        next_segment_conditioning_offset = inherited.get("next_segment_conditioning_offset")

    # Resolved absolute endpoints are authoritative when the trainer recorded
    # them. SFT carries that same record across its own user_config boundary.
    direct_cooldown = _read_conditioning_cooldown(
        meta.get("conditioning_cooldown") if meta is not None else None
    )
    inherited_cooldown = _read_conditioning_cooldown(inherited.get("conditioning_cooldown"))
    if (
        direct_cooldown is not None
        and inherited_cooldown is not None
        and direct_cooldown != inherited_cooldown
    ):
        raise ValueError("checkpoint direct and inherited conditioning_cooldown metadata disagree")
    cooldown = direct_cooldown if direct_cooldown is not None else inherited_cooldown
    recorded_user_config = meta.get("user_config") if meta is not None else None
    configured_cooldown = (
        parse_conditioning_cooldown(recorded_user_config.get("conditioning_cooldown"))
        if isinstance(recorded_user_config, Mapping)
        else None
    )
    if configured_cooldown is not None:
        if cooldown is None:
            raise ValueError(
                "checkpoint enables conditioning_cooldown but has no resolved "
                "conditioning_cooldown metadata"
            )
        if cooldown.resolve_config(configured_cooldown) != cooldown:
            raise ValueError(
                "checkpoint conditioning_cooldown config and resolved metadata disagree"
            )

    return CheckpointConditioning(
        negative_selection=negative_selection,
        feature_delta=feature_delta,
        delta_weights=str(delta_weights) if delta_weights is not None else None,
        positive_selection=positive_selection,
        label_dropout=label_dropout,
        next_segment_conditioning_offset=next_segment_conditioning_offset,
        conditioning_cooldown=cooldown.metadata_dict() if cooldown is not None else None,
    )


def checkpoint_negative_selection(meta: Mapping | None) -> Any:
    """Return the checkpoint's training-time negative_selection config, if any."""
    return read_checkpoint_conditioning(meta).negative_selection


def assert_pretrain_conditioning_recoverable(
    meta: Mapping | None,
    *,
    checkpoint_label: str,
) -> None:
    """Reject a chain base whose pretraining conditioning contract is unknowable.

    A conditioned-SFT checkpoint's ``user_config`` describes the SFT run, not the
    pretraining that shaped the encoder — it is recognised by carrying a
    top-level ``encoder_config`` instead of ``user_config.encoder``. Such a
    checkpoint written before ``pretrain_conditioning`` existed reports no
    contract at all, and reading that absence as "unconditioned" is exactly the
    silent corruption this module exists to prevent: SFT would rebuild the full
    ``num_buckets`` space for an encoder trained on ``num_buckets - 1``.
    """
    user_config = checkpoint_user_config(meta)
    is_sft_shaped = meta is not None and "encoder" not in user_config and "encoder_config" in meta
    if not is_sft_shaped or (meta is not None and PRETRAIN_CONDITIONING_META_KEY in meta):
        return
    raise ValueError(
        f"{checkpoint_label} is a conditioned-SFT checkpoint with no "
        f"{PRETRAIN_CONDITIONING_META_KEY!r} metadata, so the conditioning "
        "contract its encoder was PRETRAINED under cannot be recovered. Its "
        "absence must not be read as 'no negative_selection': that would "
        "rebuild the full num_buckets space for an encoder trained on "
        "num_buckets-1 and emit positives into the inactive bucket. Re-run the "
        "first SFT link from the PRETRAIN checkpoint on this version (it now "
        f"records {PRETRAIN_CONDITIONING_META_KEY!r}), then chain from its "
        "output."
    )


def assert_conditioning_supported(
    conditioning: CheckpointConditioning,
    *,
    checkpoint_label: str,
) -> None:
    """Apply the HARDCODED policy for a checkpoint's delta conditioning.

    Not configurable by design: the behaviour must be identical for every
    operator and every run, so there is no key to set and no acknowledgement to
    give.

    Bucket ENCODING and negative EMISSION are both reproduced and are never
    warned about — positives are discretised into the same
    ``effective_buckets = num_buckets - 1`` space pretraining used, and a
    checkpoint trained with ``negative_selection`` gets negatives in the
    reserved inactive row from the same ``sample_inactive_features_batched``
    pretraining used.

    What remains is the DELTA axis, and it splits cleanly:

    ``splice_counterfactual`` -> warn and proceed
        Pretraining derived its negatives from a spliced counterfactual
        completion and ranked its positives by the resulting delta
        (``feature_selection.delta_weights``, e.g. ``percentile_delta``). SFT
        has no counterfactual completion to splice, so it samples negatives
        randomly from the checkpoint's own ``negative_selection`` distribution
        and ranks positives by plain activation percentile (``sample_features``
        never reads ``delta_weights``). Same encoding, different draw.

    ``prev_segment`` / ``cum_max`` -> raise
        These define the positives themselves as a difference against another
        segment of the same document. Conditioned SFT has no analogue for that
        at all, so there is nothing to approximate.
    """
    mode = conditioning.delta_mode()
    if mode is not None and mode not in _DELTA_MODES_SFT_MAY_APPROXIMATE:
        approximable = sorted(m.value for m in _DELTA_MODES_SFT_MAY_APPROXIMATE)
        raise ValueError(
            f"{checkpoint_label} was pretrained with feature_delta.mode="
            f"{mode.value!r}, which conditioned SFT has no analogue for: that "
            "mode defines each positive's magnitude as a difference against "
            "another segment of the same document, and SFT selects features "
            "from a single full-conversation extraction with no such reference "
            "segment. There is no approximation to fall back on, so this run "
            f"cannot proceed. Approximable modes: {approximable}."
        )
    if mode is None and conditioning.delta_weights is None:
        return
    negatives_clause = (
        "(1) negatives — pretraining derived them from a spliced "
        "counterfactual completion, SFT samples them randomly from the "
        "checkpoint's own negative_selection distribution instead; "
        if mode is DeltaMode.splice_counterfactual
        else ""
    )
    logger.warning(
        f"{checkpoint_label} was pretrained with "
        f"feature_delta={conditioning.feature_delta!r} and "
        f"feature_selection.delta_weights={conditioning.delta_weights!r}. "
        "Conditioned SFT reproduces the bucket ENCODING exactly and emits "
        "negatives in the reserved inactive row, but it does NOT reproduce the "
        f"delta channel: {negatives_clause}"
        "(2) positive ranking — pretraining ranked candidates by the delta "
        "signal, SFT ranks them by plain activation percentile "
        "(sample_features never reads delta_weights). The selected feature "
        "distribution therefore differs from pretraining's. Proceeding."
    )
