"""Helpers for configuring the feature-GCG BPB-Adv methods (Task 1).

Provides two utilities consumed by the feature-GCG Phase-2 BPB-Adv evaluation:

- ``assert_single_positive_bucket``: validates that a loaded encoder-decoder
  model uses exactly ``num_buckets == 2`` (one positive bucket + one inactive
  bucket), which is the assumption all Phase-2 GCG methods rely on.
- ``resolve_k``: determines the ``(top_k_pos, top_k_neg)`` feature-count pair,
  preferring explicit ``OursGcgConfig`` overrides and otherwise reading the
  checkpoint's sampling-distribution geometric means (``feature_selection.mean``
  for positive, ``negative_selection.mean`` for negative).  Absent or
  unparseable selection metadata RAISES ``FeatureGcgError`` -- ``k`` must come
  from the encoder-decoder's own training selection distribution, never a
  guessed default.

Metadata path for ``resolve_k``
--------------------------------
The live path reads:

    model._loaded_meta_data["user_config"]["feature_selection"]  → pos mean
    model._loaded_meta_data["user_config"]["negative_selection"] → neg mean

Both are parsed via ``steering.conditional.sampling.parse_feature_selection``
and ``parse_negative_selection``.  The geometric mean of the returned
``SamplingSelection`` / ``NegativeSamplingSelection`` is in their ``.mean``
attribute, which is the expected value of the geometric distribution used
during training (e.g. ≈10 for d12_l12 with the default config).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from steering.conditional.feature_clusters import (
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.sampling import (
    SamplingSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContractError,
    assert_cluster_contract_loaded_state,
)

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


def assert_single_positive_bucket(
    model: Any,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[int, int]:
    """Validate that *model* uses exactly two buckets and return their indices.

    The feature-GCG Phase-2 methods assume a binary bucket layout:

      * bucket 0 — the single positive (active) bucket;
      * bucket 1 (``num_buckets - 1``) — the inactive/negative bucket.

    ``is_active(bucket, num_buckets) = bucket < num_buckets - 1`` (from
    ``steering.conditional.feature_gcg.partition``) is True only for
    bucket 0 when ``num_buckets == 2``, which is exactly the single-positive-
    bucket assumption the Phase-2 optimizer relies on.

    Parameters
    ----------
    model:
        A loaded encoder-decoder model.  Must have ``_loaded_meta_data``
        attached (see ``partition_bounds_from_model``) and an
        ``encoder.config.num_buckets`` attribute.
    cluster_contract:
        ``None`` requires the checkpoint to carry no active cluster metadata and
        preserves the legacy rejection path. A supplied value must be a validated
        :class:`~steering.eval.cluster_feature_contract.ClusterFeatureContract`
        whose loaded state agrees with ``model``.

    Returns
    -------
    tuple[int, int]
        ``(positive_bucket, negative_bucket)`` — always ``(0, 1)`` when the
        assertion passes.

    Raises
    ------
    FeatureGcgError
        If ``_loaded_meta_data`` is missing, if ``encoder.config.num_buckets``
        is absent, if ``num_buckets != 2``, or if contract validation fails.
    """
    meta_data = getattr(model, "_loaded_meta_data", None)
    if meta_data is None:
        raise FeatureGcgError(
            "Model lacks _loaded_meta_data attribute. "
            "Attach meta_data to the model after build_encdec_model(): "
            "`model._loaded_meta_data = meta_data`."
        )
    if cluster_contract is None:
        reject_feature_clusters_metadata(
            meta_data,
            "feature_gcg bucket-layout checkpoint",
        )
    else:
        if not isinstance(meta_data, Mapping):
            raise FeatureGcgError(
                "feature_gcg bucket-layout checkpoint is missing mapping metadata"
            )
        cluster_meta = cast(Mapping[str, Any], meta_data)
        try:
            assert_cluster_contract_loaded_state(
                cluster_contract,
                model,
                cluster_meta,
                surface="feature_gcg bucket-layout checkpoint",
            )
        except ClusterFeatureContractError as exc:
            raise FeatureGcgError(str(exc)) from exc

    try:
        num_buckets: int = model.encoder.config.num_buckets
    except AttributeError as exc:
        raise FeatureGcgError(
            f"Cannot read num_buckets from model.encoder.config: {exc}. "
            "Is this an encoder-decoder model with a loaded encoder?"
        ) from exc

    if num_buckets != 2:
        raise FeatureGcgError(
            f"feature-GCG Phase-2 methods require exactly 2 buckets "
            f"(positive_bucket=0, negative_bucket=1), but this checkpoint "
            f"has num_buckets={num_buckets}.  Re-train with num_buckets=2 or "
            "choose a compatible checkpoint."
        )

    positive_bucket = 0
    negative_bucket = num_buckets - 1  # == 1
    return positive_bucket, negative_bucket


def resolve_k(
    model: Any,
    cfg: OursGcgConfig,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[int, int]:
    """Return ``(top_k_pos, top_k_neg)`` for the feature-GCG methods.

    Resolution order for each dimension:

    1. If ``cfg.top_k_pos`` / ``cfg.top_k_neg`` is not ``None``, use it.
    2. Read the checkpoint's ``feature_selection`` / ``negative_selection``
       geometric mean from ``model._loaded_meta_data["user_config"]`` and
       convert to the nearest integer.
    3. RAISE ``FeatureGcgError`` when that metadata is absent or unparseable --
       ``k`` must come from the encoder-decoder's own training selection
       distribution, never a guessed default.

    The metadata key used for the positive selection is:
    ``model._loaded_meta_data["user_config"]["feature_selection"]``
    parsed via ``parse_feature_selection``, returning a ``SamplingSelection``
    whose ``.mean`` attribute is the geometric mean (≈10 for d12_l12).

    The metadata key used for the negative selection is:
    ``model._loaded_meta_data["user_config"]["negative_selection"]``
    parsed via ``parse_negative_selection``, returning a
    ``NegativeSamplingSelection`` with the same ``.mean`` attribute.

    Parameters
    ----------
    model:
        A loaded encoder-decoder model.  If it lacks ``_loaded_meta_data`` (or
        the selection metadata within it), resolution RAISES ``FeatureGcgError``
        for any dimension whose ``top_k_*`` override is ``None``.
    cfg:
        ``OursGcgConfig`` instance.  ``top_k_pos=None`` / ``top_k_neg=None``
        (the defaults) trigger metadata lookup.
    cluster_contract:
        ``None`` requires the checkpoint to carry no active cluster metadata and
        preserves the legacy rejection path. A supplied value must be a validated
        :class:`~steering.eval.cluster_feature_contract.ClusterFeatureContract`
        whose loaded state agrees with ``model``.

    Returns
    -------
    tuple[int, int]
        ``(top_k_pos, top_k_neg)`` as plain Python integers.

    Raises
    ------
    FeatureGcgError
        If required selection metadata is absent or unparseable, or if contract
        validation fails.
    """
    meta_data = getattr(model, "_loaded_meta_data", None)
    if cluster_contract is None:
        reject_feature_clusters_metadata(
            meta_data,
            "feature_gcg top-k checkpoint",
        )
    else:
        if not isinstance(meta_data, Mapping):
            raise FeatureGcgError(
                "feature_gcg top-k checkpoint is missing mapping metadata"
            )
        cluster_meta = cast(Mapping[str, Any], meta_data)
        try:
            assert_cluster_contract_loaded_state(
                cluster_contract,
                model,
                cluster_meta,
                surface="feature_gcg top-k checkpoint",
            )
        except ClusterFeatureContractError as exc:
            raise FeatureGcgError(str(exc)) from exc

    top_k_pos = _resolve_k_single(
        override=cfg.top_k_pos,
        meta_data=meta_data,
        selection_key="feature_selection",
        parse_fn=_parse_pos_mean,
    )
    top_k_neg = _resolve_k_single(
        override=cfg.top_k_neg,
        meta_data=meta_data,
        selection_key="negative_selection",
        parse_fn=_parse_neg_mean,
    )
    return top_k_pos, top_k_neg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_pos_mean(raw: dict) -> float | None:
    """Return the geometric mean from a feature_selection dict, or None on error."""
    try:
        fs = parse_feature_selection(raw)
    except (ValueError, TypeError):
        return None
    if isinstance(fs, SamplingSelection):
        return fs.mean
    return None


def _parse_neg_mean(raw: dict) -> float | None:
    """Return the geometric mean from a negative_selection dict, or None on error."""
    try:
        ns = parse_negative_selection(raw)
    except (ValueError, TypeError):
        return None
    if ns is not None:
        return ns.mean
    return None


def _resolve_k_single(
    *,
    override: int | None,
    meta_data: dict | None,
    selection_key: str,
    parse_fn: Any,
) -> int:
    """Resolve a single top-k dimension.

    Returns *override* when set; otherwise reads the geometric mean from the
    checkpoint's ``selection_key`` metadata. ``k`` MUST come from the encoder-
    decoder's own training selection distribution -- an absent / unparseable
    selection is a real config error (a metadata-less or wrong checkpoint), NOT
    something to paper over with a guessed default. Raise loudly; a caller that
    truly wants a fixed ``k`` sets ``OursGcgConfig.top_k_pos`` / ``top_k_neg``.
    """
    if override is not None:
        return override
    if meta_data is None:
        raise FeatureGcgError(
            f"resolve_k: model has no _loaded_meta_data, so top_k for "
            f"{selection_key!r} cannot be read from the checkpoint's training "
            f"selection distribution. Load a checkpoint carrying its selection "
            f"metadata, or set OursGcgConfig.top_k_pos / top_k_neg explicitly."
        )
    user_cfg: dict = meta_data.get("user_config") or {}
    raw = user_cfg.get(selection_key)
    if raw is None:
        raise FeatureGcgError(
            f"resolve_k: checkpoint user_config has no {selection_key!r}; top_k must "
            f"come from the training selection distribution. Use a checkpoint with the "
            f"selection metadata, or set OursGcgConfig.top_k_pos / top_k_neg explicitly."
        )
    mean = parse_fn(raw)
    if mean is None:
        raise FeatureGcgError(
            f"resolve_k: could not parse a geometric mean from the checkpoint's "
            f"{selection_key!r} metadata; top_k must come from the training selection "
            f"distribution. Set OursGcgConfig.top_k_pos / top_k_neg explicitly."
        )
    return int(mean)
