"""Per-partition count bounds and the L1 count penalty.

Provides:
- PartitionBounds: immutable dataclass holding the positive and negative
  partition count caps (search bounds) plus the bucket layout.
- partition_bounds_from_model: build ``PartitionBounds`` from a loaded model —
  ``num_buckets`` from ``model.encoder.config``, ``has_negative`` from whether
  the checkpoint metadata carries a ``negative_selection``, and the count caps
  straight from the config.
- is_active: bucket predicate — True when bucket belongs to the positive
  (active) partition (i.e. ``bucket < num_buckets - 1``).
- count_penalty: L1 sparsity penalty ``reg_lambda * len(bag)`` — a flat
  per-feature cost that biases the optimizer toward smaller bags.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from steering.conditional.feature_clusters import (
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContractError,
    assert_cluster_contract_loaded_state,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


@dataclass(frozen=True)
class PartitionBounds:
    """Immutable per-partition count caps and bucket layout.

    Both ``max_k_pos`` and ``max_k_neg`` are **exclusive**: the realized
    maximum count for the positive partition is ``max_k_pos - 1``.

    When ``has_negative`` is False, the ``min_k_neg`` and ``max_k_neg`` fields
    are present but carry dummy values and are ignored by the edit space.
    """

    num_buckets: int
    has_negative: bool  # True when negative_selection is present in metadata
    min_k_pos: int
    max_k_pos: int  # EXCLUSIVE: realized max = max_k_pos - 1
    min_k_neg: int
    max_k_neg: int  # EXCLUSIVE: realized max = max_k_neg - 1


def partition_bounds_from_model(
    model: Any,
    cfg: FeatureGcgConfig,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> PartitionBounds:
    """Build :class:`PartitionBounds` from a loaded model and the config.

    The count caps (``min_k`` / ``max_k``, both partitions) come directly from
    *cfg* (validated in :meth:`FeatureGcgConfig.__post_init__`). Only the bucket
    layout is read from the checkpoint:

    * ``num_buckets`` from ``model.encoder.config.num_buckets``;
    * ``has_negative`` from whether
      ``model._loaded_meta_data["user_config"]["negative_selection"]`` is present
      (it determines whether the last bucket is the inactive/negative partition).

    Parameters
    ----------
    model:
        Loaded encoder-decoder model carrying checkpoint metadata and encoder
        configuration.
    cfg:
        Validated Feature-GCG count-bound configuration.
    cluster_contract:
        ``None`` requires the checkpoint to carry no active cluster metadata and
        preserves the legacy rejection path. A supplied value must be a validated
        :class:`~steering.eval.cluster_feature_contract.ClusterFeatureContract`
        whose loaded state agrees with ``model``.

    Raises
    ------
    FeatureGcgError
        If the model lacks ``_loaded_meta_data`` or
        ``encoder.config.num_buckets``, or if contract validation fails.
    """
    meta_data = getattr(model, "_loaded_meta_data", None)
    if meta_data is None:
        raise FeatureGcgError(
            "Model lacks _loaded_meta_data attribute. Attach meta_data to the model "
            "after calling build_encdec_model(): "
            "`model._loaded_meta_data = meta_data`."
        )
    if cluster_contract is None:
        reject_feature_clusters_metadata(
            meta_data,
            "feature_gcg partition checkpoint",
        )
    else:
        if not isinstance(meta_data, Mapping):
            raise FeatureGcgError(
                "feature_gcg partition checkpoint is missing mapping metadata"
            )
        try:
            assert_cluster_contract_loaded_state(
                cluster_contract,
                model,
                meta_data,
                surface="feature_gcg partition checkpoint",
            )
        except ClusterFeatureContractError as exc:
            raise FeatureGcgError(str(exc)) from exc

    user_cfg: dict = meta_data.get("user_config") or {}

    try:
        num_buckets: int = model.encoder.config.num_buckets
    except AttributeError as exc:
        raise FeatureGcgError(
            f"Cannot read num_buckets from model.encoder.config: {exc}. "
            "Is this an encoder-decoder model with a loaded encoder?"
        ) from exc

    has_negative = user_cfg.get("negative_selection") is not None

    return PartitionBounds(
        num_buckets=num_buckets,
        has_negative=has_negative,
        min_k_pos=cfg.min_k_pos,
        max_k_pos=cfg.max_k_pos,
        min_k_neg=cfg.min_k_neg,
        max_k_neg=cfg.max_k_neg,
    )


def is_active(bucket: int, num_buckets: int) -> bool:
    """Return True when *bucket* belongs to the positive (active) partition.

    The last bucket index (``num_buckets - 1``) is reserved for the negative
    (inactive-feature) partition when negative sampling is configured.
    Callers with ``bounds.has_negative=False`` should treat all entries as
    active and not rely on this predicate to decide the partition.

    Parameters
    ----------
    bucket:
        Bucket index of a conditioning entry.
    num_buckets:
        Total number of magnitude discretisation buckets.

    Returns
    -------
    bool
        ``True`` iff ``bucket < num_buckets - 1``.
    """
    return bucket < num_buckets - 1


def count_penalty(bag: FeatureConditioning, reg_lambda: float) -> float:
    """L1 sparsity penalty: ``reg_lambda * len(bag)``.

    A flat per-feature cost on the bag size. The GCG optimiser minimises total
    loss, so this biases proposals toward smaller bags; ``reg_lambda`` scales the
    pressure and ``reg_lambda == 0`` disables it.

    Parameters
    ----------
    bag:
        Current feature conditioning state.
    reg_lambda:
        Non-negative regularisation weight.

    Returns
    -------
    float
        ``reg_lambda * len(bag)``.
    """
    return reg_lambda * len(bag)
