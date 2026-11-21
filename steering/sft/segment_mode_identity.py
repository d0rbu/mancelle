"""Checkpoint identity for conditioned-SFT feature segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from beartype import beartype

from nanochat.dataset import DatasetSplit
from steering._common.strict_json import canonical_json
from steering.segmentation import SegmentMode, segment_mode_from_metadata

SFT_SEGMENT_MODE_META_KEY = "sft_segment_mode"
ONLINE_SAE_LINEAGE_META_KEY = "structured_sft_online_sae_lineage_v1"
ONLINE_SAE_LINEAGE_VERSION = 1
ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY = "source_segment_mode"
TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY = "turn_self_cond_segment_size_rustbpe"
FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY = "fixed_self_cond_segment_size_rustbpe"
FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY = "fixed_self_cond_cadence_provenance"
FIXED_CADENCE_VERSION = "packed_train_fixed_cadence_v1"
FIXED_CADENCE_SOURCE_WIDTH = 128
FIXED_CADENCE_MAX_ROWS = 4096
SELFCOND_DATA_META_KEY = "selfcond_data"
# Every raw metadata entry the resolvers below (and OnlineSAELineage) read;
# comparing these verbatim binds a reloaded checkpoint to a validated identity.
# The rollout-policy ``selfcond`` block is deliberately absent: no resolver
# reads it since the cadence became a measured top-level value.
SEGMENTATION_IDENTITY_META_KEYS = (
    SFT_SEGMENT_MODE_META_KEY,
    ONLINE_SAE_LINEAGE_META_KEY,
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    SELFCOND_DATA_META_KEY,
    FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY,
)


@beartype
def segmentation_identity_drift(
    loaded_meta: Mapping[str, Any], setup_meta: Mapping[str, Any]
) -> list[str]:
    """Identity keys whose reloaded value differs from the validated setup's.

    Presence plus canonical JSON: ``dict.get`` would equate an absent key with
    null, and raw ``!=`` would equate 37 with 37.0.
    """
    return [
        key
        for key in SEGMENTATION_IDENTITY_META_KEYS
        if (key in loaded_meta, canonical_json(loaded_meta.get(key)))
        != (key in setup_meta, canonical_json(setup_meta.get(key)))
    ]


def _metadata_error(checkpoint_dir: str, detail: str) -> ValueError:
    return ValueError(
        f"checkpoint {checkpoint_dir} has invalid SFT segmentation metadata: {detail}"
    )


def _witness_mode(
    block: Mapping[str, Any],
    *,
    key: str,
    witness: str,
    checkpoint_dir: str,
) -> tuple[str, SegmentMode]:
    """Read one segment-mode witness; an absent key means legacy fixed tokens."""
    try:
        return witness, segment_mode_from_metadata(block, key=key, source=witness)
    except ValueError as exc:
        raise _metadata_error(checkpoint_dir, str(exc)) from exc


@beartype
def resolve_checkpoint_sft_segment_mode(
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> SegmentMode:
    """Resolve every checkpoint witness to one effective SFT segment mode.

    Modern metadata may record the identity at the top level, in the immutable
    online-SAE lineage, and in an SFT self-conditioning data block; disagreement
    between witnesses is corruption rather than a precedence question. A witness
    that is simply absent contributes nothing -- only turn-capable trainers ever
    wrote one -- so a checkpoint with no witness at all is the legacy
    ``fixed_tokens`` identity. A witness that is present but malformed, and a
    carrier block whose shape is corrupt, are still errors.
    """
    if checkpoint_meta is None:
        return SegmentMode.fixed_tokens

    witnesses: list[tuple[str, SegmentMode]] = []
    if SFT_SEGMENT_MODE_META_KEY in checkpoint_meta:
        witnesses.append(
            _witness_mode(
                checkpoint_meta,
                key=SFT_SEGMENT_MODE_META_KEY,
                witness=SFT_SEGMENT_MODE_META_KEY,
                checkpoint_dir=checkpoint_dir,
            )
        )

    if ONLINE_SAE_LINEAGE_META_KEY in checkpoint_meta:
        raw_lineage = checkpoint_meta[ONLINE_SAE_LINEAGE_META_KEY]
        if (
            not isinstance(raw_lineage, Mapping)
            or raw_lineage.get("schema_version") != ONLINE_SAE_LINEAGE_VERSION
        ):
            raise _metadata_error(
                checkpoint_dir,
                f"{ONLINE_SAE_LINEAGE_META_KEY} must be a mapping with "
                f"schema_version={ONLINE_SAE_LINEAGE_VERSION}, got {raw_lineage!r}",
            )
        lineage_witness = f"{ONLINE_SAE_LINEAGE_META_KEY}.{ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY}"
        witnesses.append(
            _witness_mode(
                raw_lineage,
                key=ONLINE_SAE_SOURCE_SEGMENT_MODE_META_KEY,
                witness=lineage_witness,
                checkpoint_dir=checkpoint_dir,
            )
        )

    if SELFCOND_DATA_META_KEY in checkpoint_meta:
        raw_selfcond_data = checkpoint_meta[SELFCOND_DATA_META_KEY]
        if not isinstance(raw_selfcond_data, Mapping):
            raise _metadata_error(
                checkpoint_dir,
                f"selfcond_data must be a mapping, got {type(raw_selfcond_data).__name__}",
            )
        # Conditioned-SFT self-conditioning checkpoints minted before this key
        # existed record no nested mode. Their trainer could only consume
        # fixed-token features, so absence is no witness rather than corruption
        # -- unlike an absent lineage source_segment_mode, which stays an
        # explicit fixed witness because only online-SAE trainers wrote one.
        if SFT_SEGMENT_MODE_META_KEY in raw_selfcond_data:
            witnesses.append(
                _witness_mode(
                    raw_selfcond_data,
                    key=SFT_SEGMENT_MODE_META_KEY,
                    witness=f"selfcond_data.{SFT_SEGMENT_MODE_META_KEY}",
                    checkpoint_dir=checkpoint_dir,
                )
            )

    resolved_modes = {mode for _, mode in witnesses}
    if len(resolved_modes) > 1:
        rendered = ", ".join(f"{name}={mode.value!r}" for name, mode in witnesses)
        raise _metadata_error(
            checkpoint_dir,
            f"conflicting segment-mode witnesses: {rendered}",
        )
    mode = witnesses[0][1] if witnesses else SegmentMode.fixed_tokens
    invalid_key = (
        TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY
        if mode is SegmentMode.fixed_tokens
        else FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY
    )
    if invalid_key in checkpoint_meta:
        raise _metadata_error(
            checkpoint_dir,
            f"{invalid_key} is valid only for "
            + ("turn-delimited SFT checkpoints" if mode is SegmentMode.fixed_tokens else "fixed-token checkpoints"),
        )
    return mode


@beartype
def resolve_turn_self_cond_segment_size_rustbpe(
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> int | None:
    """Read the post-training-MEASURED generated-assistant cadence.

    Fixed-token checkpoints retain their historical caller-owned cadence and
    therefore return ``None``. A turn-delimited checkpoint must carry an
    explicit positive RustBPE cadence, measured from the training artifacts by
    self-conditioning post-training and recorded at the metadata top level
    beside ``sft_segment_mode``; inferring it from source-tokenizer geometry is
    forbidden.
    """
    mode = resolve_checkpoint_sft_segment_mode(
        checkpoint_meta,
        checkpoint_dir=checkpoint_dir,
    )
    meta = checkpoint_meta or {}
    cadence_present = TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY in meta
    if mode is SegmentMode.fixed_tokens:
        return None
    if not cadence_present:
        raise _metadata_error(
            checkpoint_dir,
            "turn-delimited checkpoints require top-level post-training "
            f"metadata {TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY}; it "
            "cannot be inferred from source-tokenizer segment_size",
        )
    raw_cadence = meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY]
    if isinstance(raw_cadence, bool) or not isinstance(raw_cadence, int) or raw_cadence <= 0:
        raise _metadata_error(
            checkpoint_dir,
            f"{TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY} must be a "
            f"positive integer, got {raw_cadence!r}",
        )
    return raw_cadence


@beartype
def resolve_fixed_self_cond_segment_size_rustbpe(
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
    segment_size_assertion: int | None = None,
    loaded_model_sha256: str | None = None,
) -> int | None:
    """Require measured fixed cadence; a CLI assertion is never a fallback.

    Turn checkpoints return None and reject both a fixed key and any fixed
    assertion. Legacy fixed callers need not invoke this reader (AxBench
    continues to use sequence_len).
    """
    mode = resolve_checkpoint_sft_segment_mode(checkpoint_meta, checkpoint_dir=checkpoint_dir)
    if mode is SegmentMode.turn_delimited_fixed_tokens:
        if segment_size_assertion is not None:
            raise _metadata_error(checkpoint_dir, "fixed cadence assertion is invalid on turn checkpoints")
        return None
    raw = (checkpoint_meta or {}).get(FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY)
    if type(raw) is not int or raw <= 0:
        raise _metadata_error(
            checkpoint_dir,
            f"{FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY} must be a recorded positive integer, got {raw!r}",
        )
    provenance = (checkpoint_meta or {}).get(FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY)
    if not isinstance(provenance, Mapping):
        raise _metadata_error(checkpoint_dir, "fixed cadence requires measured provenance")
    if (provenance.get("implementation_version") != FIXED_CADENCE_VERSION
            or provenance.get("split") != DatasetSplit.TRAIN.value
            or type(provenance.get("source_width")) is not int
            or provenance["source_width"] != FIXED_CADENCE_SOURCE_WIDTH):
        raise _metadata_error(checkpoint_dir, "fixed cadence measurement version/TRAIN/source width mismatch")
    measurement = provenance.get("measurement")
    if (not isinstance(measurement, Mapping) or type(measurement.get("cadence")) is not int
            or measurement["cadence"] != raw):
        raise _metadata_error(checkpoint_dir, "recorded fixed cadence contradicts measurement cadence")
    mean = measurement.get("mean")
    if not isinstance(mean, Mapping) or any(
        type(mean.get(key)) is not int or mean[key] <= 0 for key in ("numerator", "denominator")
    ):
        raise _metadata_error(checkpoint_dir, "fixed cadence requires a positive rational mean")
    if raw != round(Fraction(mean["numerator"], mean["denominator"])):
        raise _metadata_error(checkpoint_dir, "recorded fixed cadence contradicts measurement mean")
    if type(measurement.get("windows")) is not int or measurement["windows"] <= 0:
        raise _metadata_error(checkpoint_dir, "fixed cadence requires measured windows")
    if loaded_model_sha256 is not None:
        original = provenance.get("original_checkpoint")
        models = [] if not isinstance(original, list) else [
            row for row in original if isinstance(row, Mapping)
            and isinstance(row.get("path"), str)
            and Path(row["path"]).name.startswith("model_") and Path(row["path"]).suffix == ".pt"
        ]
        if len(models) != 1 or models[0].get("sha256") != loaded_model_sha256:
            raise _metadata_error(checkpoint_dir, "fixed cadence original model digest disagrees with loaded model")
    if segment_size_assertion is not None and (
        type(segment_size_assertion) is not int or segment_size_assertion != raw
    ):
        raise _metadata_error(checkpoint_dir, f"fixed cadence assertion {segment_size_assertion!r} conflicts with measured {raw}")
    return raw


@beartype
def assert_resume_sft_segment_mode_matches(
    configured: SegmentMode,
    checkpoint_meta: Mapping[str, Any] | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Verify feature-segmentation identity before resuming conditioned SFT."""
    checkpoint_mode = resolve_checkpoint_sft_segment_mode(
        checkpoint_meta,
        checkpoint_dir=checkpoint_dir,
    )
    if checkpoint_mode is configured:
        return
    raise ValueError(
        f"{SFT_SEGMENT_MODE_META_KEY} mismatch on resume from {checkpoint_dir}: "
        f"checkpoint records {checkpoint_mode.value!r}, "
        f"configured {configured.value!r}. Resume with features of the recorded "
        "segmentation mode or start a fresh SFT run."
    )
