"""Configuration dataclasses and resolution helpers for the encoder-decoder
conditional training architecture.

The encoder's depth and embedding dimension can be specified as either absolute
values (``int``) or ratios of the decoder's corresponding parameter (``float``).
Resolution functions convert the user-facing config into concrete integers
suitable for model construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

from beartype import beartype

from steering.refined_types import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeaturePredLossMode(StrEnum):
    FULL_SEGMENT = "full_segment"  # mean across all source tokens (per-token loss)
    LAST_TOKEN = "last_token"  # last token of each source segment
    MAX_POOL = "max_pool"  # per-feature max across segment positions, then loss
    BEST_TOKEN = "best_token"  # token with min loss; cannot run at inference


class FeaturePredLossFn(StrEnum):
    BCE = "bce"
    MSE = "mse"


class FeaturePredictionBlock(StrEnum):
    MLP = "mlp"
    TRANSFORMER = "transformer"


class FeatureSelectionProducer(StrEnum):
    """Versioned score producer used to select positive conditioning features.

    ``FULL_PERCENTILE_RANK`` means selection consumes the same full-distribution
    percentile rank that supervises the feature-prediction head. Packed
    pretraining and percentile-based sampling use this producer.

    ``RAW_MAGNITUDE`` means selection consumes raw SAE activation magnitudes.
    A percentile-prediction head can replay only per-feature quantile cutoffs:
    their statistics key maps monotonically into percentile space. Cross-feature
    raw-magnitude ranking has no exact percentile-space inverse and must fail.
    """

    FULL_PERCENTILE_RANK = "full_percentile_rank_v1"
    RAW_MAGNITUDE = "raw_magnitude_v1"


class FeatureBucketProducer(StrEnum):
    """Versioned producer used to bucket selected positive features.

    ``UNIFORM_PERCENTILE`` applies the historical ``floor(score * B)`` rule to
    full-distribution percentile ranks. ``RAW_STATS_BOUNDARIES`` reproduces raw
    magnitude bucketing by translating the persisted statistics-key boundary
    grid into continuous normalized percentile rank.
    """

    UNIFORM_PERCENTILE = "uniform_percentile_v1"
    RAW_STATS_BOUNDARIES = "raw_stats_boundaries_v1"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def validate_feature_prediction_head_layers(value: object) -> int:
    """Validate the total residual depth without coercing metadata or CLI values."""
    try:
        return NonNegativeInt.parse(value)
    except TypeError as exc:
        raise ValueError(
            f"feature prediction head_layers must be a nonnegative integer, got {value!r}"
        ) from exc


@dataclass
class FeaturePredictionConfig:
    """Configuration for the auxiliary feature-prediction head.

    The default ``loss_mode`` is :attr:`FeaturePredLossMode.MAX_POOL`: for
    each segment, take the per-feature max of the head's logits across the
    segment's tokens, then compute the BCE/MSE loss against the next
    segment's target. Max-pool is symmetric across the segment (no
    last-position bias) and runs identically at training and inference.

    Other modes:

    - :attr:`FULL_SEGMENT` — mean of per-token losses across all source
      tokens. Equivalent at inference: mean of fp_logits across positions.
    - :attr:`LAST_TOKEN` — only the last token of each source segment.
    - :attr:`BEST_TOKEN` — argmin over per-token loss against the target.
      **Cannot run at inference** because it needs the next segment's
      target features.
    """

    weight: float = 0.1
    layer: int = -1
    loss_mode: FeaturePredLossMode = FeaturePredLossMode.MAX_POOL
    loss_fn: FeaturePredLossFn = FeaturePredLossFn.BCE
    block: FeaturePredictionBlock = FeaturePredictionBlock.MLP
    head_layers: int = 0

    def __post_init__(self) -> None:
        self.block = FeaturePredictionBlock(self.block)
        self.head_layers = validate_feature_prediction_head_layers(self.head_layers)
        self.loss_mode = FeaturePredLossMode(self.loss_mode)
        self.loss_fn = FeaturePredLossFn(self.loss_fn)


def reject_removed_feature_prediction_keys(config: Mapping) -> None:
    """Reject removed overrides, including Hydra's explicit ``+key`` additions."""
    if "feature_prediction_mlp_layers" in config:
        raise ValueError(
            "feature_prediction_mlp_layers was removed; use feature_prediction_block "
            "and feature_prediction_head_layers. Convert old checkpoints with "
            "steering.tools.migrate_feature_head_metadata."
        )
    raw = config.get("feature_prediction")
    if isinstance(raw, Mapping) and "mlp_layers" in raw:
        raise ValueError(
            "feature_prediction.mlp_layers was removed; use block and head_layers. "
            "Convert old checkpoints with steering.tools.migrate_feature_head_metadata."
        )


def parse_feature_prediction_config(
    raw: Mapping | None, *, checkpoint: bool = False
) -> FeaturePredictionConfig | None:
    """Fresh configs have defaults; checkpoints must explicitly declare architecture."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("feature_prediction metadata must be a mapping")
    reject_removed_feature_prediction_keys({"feature_prediction": raw})
    if checkpoint and not {"block", "head_layers"}.issubset(raw):
        raise ValueError(
            "feature-prediction checkpoint requires explicit block and head_layers; "
            "convert old metadata with steering.tools.migrate_feature_head_metadata"
        )
    return FeaturePredictionConfig(**dict(raw))


def serialize_feature_prediction_config(
    config: FeaturePredictionConfig | None,
) -> dict | None:
    """The single canonical representation used in both checkpoint mirrors."""
    if config is None:
        return None
    return {
        **asdict(config),
        "block": config.block.value,
        "loss_mode": config.loss_mode.value,
        "loss_fn": config.loss_fn.value,
    }


def feature_prediction_config_from_metadata(meta: Mapping) -> FeaturePredictionConfig | None:
    """Read either checkpoint mirror and fail closed if both disagree."""
    user = meta.get("user_config", {})
    if not isinstance(user, Mapping):
        raise ValueError("checkpoint user_config must be a mapping")
    # Historical SFT saved its zero-depth CLI default even for headless models.
    # That unused record describes no head to migrate; fresh CLI/config parsing
    # still rejects the removed option, and head-bearing checkpoints stay strict.
    old_headless_default = (
        user.get("feature_prediction") is None
        and meta.get("feature_prediction_config") is None
        and type(user.get("feature_prediction_mlp_layers")) is int
        and user["feature_prediction_mlp_layers"] == 0
    )
    if not old_headless_default:
        reject_removed_feature_prediction_keys(user)
    mirrors = []
    if "feature_prediction" in user:
        mirrors.append(parse_feature_prediction_config(user["feature_prediction"], checkpoint=True))
    if "feature_prediction_config" in meta:
        mirrors.append(parse_feature_prediction_config(meta["feature_prediction_config"], checkpoint=True))
    if len(mirrors) == 2 and mirrors[0] != mirrors[1]:
        raise ValueError("conflicting feature-prediction metadata mirrors")
    return mirrors[0] if mirrors else None


def validate_feature_prediction_architecture(
    saved: FeaturePredictionConfig | None,
    requested_block: FeaturePredictionBlock,
    requested_depth: int,
    *,
    resuming: bool,
) -> None:
    """Only a fresh linear checkpoint may acquire a different head architecture."""
    block = FeaturePredictionBlock(requested_block)
    depth = validate_feature_prediction_head_layers(requested_depth)
    if saved is None:
        if depth:
            raise ValueError("a headless checkpoint cannot acquire a positive-depth feature head")
        return
    if (saved.block, saved.head_layers) != (block, depth) and (resuming or saved.head_layers):
        raise ValueError(
            f"feature_prediction_block={block.value}, feature_prediction_head_layers={depth} "
            f"does not match checkpoint block={saved.block.value}, head_layers={saved.head_layers}; "
            "only fresh SFT from a linear head can add residual blocks"
        )


def validate_feature_prediction_resume(
    meta: Mapping, requested: FeaturePredictionConfig | None
) -> None:
    saved = feature_prediction_config_from_metadata(meta)
    if (saved is None) != (requested is None):
        raise ValueError("resume feature-prediction head presence differs from checkpoint")
    if requested is not None:
        validate_feature_prediction_architecture(
            saved, requested.block, requested.head_layers, resuming=True
        )


@dataclass
class EncoderConfig:
    """Configuration for the lightweight encoder tower.

    ``n_embd`` and ``depth`` accept *either* an ``int`` (absolute value) or a
    ``float`` (ratio of the decoder's dimension / depth).  Use the
    ``resolve_encoder_dim`` and ``resolve_encoder_depth`` helpers to obtain
    concrete integers.

    When ``n_head`` or ``n_kv_head`` are ``None``, they are derived from the
    resolved embedding dimension and ``head_dim`` (i.e. ``enc_dim // head_dim``),
    following the same scaling law as the decoder.
    """

    n_embd: int | float = (
        0.5  # int → absolute, float → ratio of decoder dim (rounded to 128)
    )
    depth: int | float = 0.5  # int → absolute layers, float → ratio of decoder depth
    n_head: int | None = None  # None → enc_dim // head_dim (same scaling as decoder)
    n_kv_head: int | None = None  # None → same as n_head (MHA)
    head_dim: int = 64
    ve_gate_channels: int = 32


@dataclass
class CrossAttentionConfig:
    """Where and how cross-attention is injected into the decoder.

    When ``n_head`` or ``n_kv_head`` are ``None``, they are derived from
    ``dec_dim // head_dim`` (matching the decoder's head structure).
    """

    frequency: int = 4  # insert cross-attention every N-th decoder layer
    n_head: int | None = None  # None → dec_dim // head_dim (match decoder)
    n_kv_head: int | None = None  # None → same as n_head (MHA)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


@beartype
def resolve_encoder_heads(
    n_head: PositiveInt | None,
    n_kv_head: PositiveInt | None,
    enc_dim: PositiveInt,
    head_dim: PositiveInt,
) -> tuple[PositiveInt, PositiveInt]:
    """Derive encoder head counts from resolved embedding dimension.

    ``@beartype`` owns the four unary positive-integer contracts. The body
    retains only derivation and divisibility relationships among them.

    Parameters
    ----------
    n_head:
        Explicit head count, or ``None`` to derive as ``enc_dim // head_dim``.
    n_kv_head:
        Explicit KV head count, or ``None`` to match ``n_head`` (MHA).
    enc_dim:
        Resolved encoder embedding dimension.
    head_dim:
        Head dimension (shared with decoder).

    Returns
    -------
    (n_head, n_kv_head)
    """
    if n_head is None:
        if enc_dim < head_dim:
            raise ValueError(
                f"Encoder dim ({enc_dim}) must be at least head_dim ({head_dim})"
            )
        n_head = PositiveInt.parse(enc_dim // head_dim)
    if n_kv_head is None:
        n_kv_head = n_head
    if enc_dim % n_head != 0:
        raise ValueError(
            f"Encoder dim ({enc_dim}) must be divisible by n_head ({n_head})"
        )
    if n_head % n_kv_head != 0:
        raise ValueError(
            f"Encoder n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        )
    return n_head, n_kv_head


@beartype
def resolve_encoder_depth(
    depth: NonNegativeInt | NonNegativeFloat,
    decoder_depth: PositiveInt,
) -> NonNegativeInt:
    """Convert a user-facing depth spec into a concrete layer count.

    ``@beartype`` owns the non-negative absolute/ratio domains. The body
    retains the relationship to ``decoder_depth`` and resolves the ratio.

    Parameters
    ----------
    depth:
        * ``int`` — absolute layer count.  Must be in ``[0, 2 * decoder_depth]``.
        * ``float`` — ratio of *decoder_depth*.  Must be in ``[0.0, 2.0]``.
          The result is rounded to the nearest integer.
    decoder_depth:
        Number of transformer layers in the decoder.

    Returns
    -------
    int
        Resolved encoder depth (number of self-attention layers).

    Raises
    ------
    ValueError
        If the value exceeds the allowed maximum relative to decoder depth.
    """
    if isinstance(depth, float):
        if depth > 2.0:
            raise ValueError(f"Encoder depth ratio must be at most 2.0, got {depth}")
        return NonNegativeInt.parse(round(depth * decoder_depth))

    # int path
    if depth > 2 * decoder_depth:
        raise ValueError(
            f"Encoder depth must be at most 2x decoder depth "
            f"({2 * decoder_depth}), got {depth}"
        )
    return NonNegativeInt.parse(depth)


@beartype
def resolve_encoder_dim(
    n_embd: PositiveInt | PositiveFloat,
    decoder_dim: PositiveInt,
) -> PositiveInt:
    """Convert a user-facing embedding-dimension spec into a concrete value.

    ``@beartype`` owns the positive absolute/ratio domains; this function only
    resolves the ratio and returns a refined concrete dimension.

    Parameters
    ----------
    n_embd:
        * ``int`` — absolute dimension.  Must be positive.
        * ``float`` — ratio of *decoder_dim*.  The raw product is rounded to
          the nearest multiple of 128, with a minimum of 128.  Must be
          positive.
    decoder_dim:
        Embedding dimension of the decoder.

    Returns
    -------
    int
        Resolved encoder embedding dimension.

    """
    if isinstance(n_embd, float):
        raw = n_embd * decoder_dim
        # Round to nearest multiple of 128, minimum 128
        resolved = max(128, round(raw / 128) * 128)
        return PositiveInt.parse(resolved)

    # int path
    return PositiveInt.parse(n_embd)
