"""Public API for universal feature-space GCG conditioning optimization.

The supported entry point, :func:`optimize_feature_conditioning_universal`,
accepts one or many prebuilt objective members and optimizes one shared
:class:`~steering.conditional.feature_gcg.feature_conditioning.FeatureConditioning`
bag over them. It parses the checkpoint's partition bounds, builds the feature
edit space, adapts the members for the requested objective, and delegates the
search to :func:`~steering.conditional.feature_gcg.core.run_gcg`.

The base objective is composed with the count penalty (``g + R``), while the
gradient is passed through unchanged: proposal ranking is per feature ID and
count-independent, and the exact objective enforces the count cost.

.. _penalty-composition:

Penalty composition
-------------------
The ``objective_batch`` passed to ``run_gcg`` is::

    g_penalized(items) = base.losses(adapted) + count_penalty(bag, λ)   # λ * len(bag)

where ``adapted`` wraps each ``(member, bag)`` tuple from ``model_item`` into
:class:`_ItemAdapter` ``(member=member, bag=bag)`` so that ``base.losses`` and
``base.gradients`` can access ``.member`` and ``.bag`` via attribute access.

This matches the ``FeatureLmItem`` / ``ClassMarginItem`` duck-typing interface:
both objectives access ``item.member.<field>`` and ``item.bag``.

``gradient_provider_batch = base.gradients`` UNCHANGED — no penalty term.

Allowed feature IDs
-------------------
Feature IDs remain in ORIGINAL SAE-ID space throughout the public contract.
On an uncompacted checkpoint the proposal universe is the full original SAE
width. On an ordinary compacted checkpoint it is the original IDs in the
checkpoint's vocabulary ``V``, obtained from
:func:`~steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids`.
On a sanctioned clustered checkpoint that vocabulary is set-identical to the
validated contract's representative universe. The contract supplies the
proposal universe explicitly as a defensive consistency boundary. The edit
space translates original IDs to compact ``id_weight`` rows only while scoring;
returned and persisted bags retain original IDs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch as t

from errors import FeatureClustersError
from steering.conditional.feature_clusters import (
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.feature_edit_space import FeatureConditioningEditSpace
from steering.conditional.feature_gcg.gradient import (
    FeatureGrad,
    clear_static_weight_cache,
)
from steering.conditional.feature_gcg.objective import FeatureObjective
from steering.conditional.feature_gcg.partition import (
    count_penalty,
    partition_bounds_from_model,
)
from steering.eval.axbench_bpb.vocab_restriction import feature_vocab_allowed_ids
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContractError,
    assert_loaded_cluster_target,
)
from steering.refined_types import NonEmptyNonNegativeIntTuple

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _Group:
    """Minimal group container satisfying :meth:`FeatureConditioningEditSpace.validate`.

    The ``validate`` method reads ``group.key_str`` (non-empty str) and
    ``group.members`` (non-empty sequence); ``concept_id`` and ``mode`` are
    optional (duck-typed via ``getattr``) and surface in progress events when
    present.
    """

    key_str: str
    members: list
    concept_id: Any = None
    mode: Any = None


@dataclass
class _ItemAdapter:
    """Adapts ``(member, bag)`` tuples from ``model_item`` to the ``.member``/``.bag`` protocol.

    Both ``FeatureLmItem`` and ``ClassMarginItem`` access items via ``.member``
    and ``.bag``.  This adapter satisfies that protocol for any ``(member, bag)``
    pair — the concrete types are enforced by the objective, not here.
    """

    member: Any
    bag: FeatureConditioning


def _assert_cluster_contract(
    model: Any,
    cluster_contract: ClusterFeatureContract | None,
) -> None:
    """Require exact cluster metadata when, and only when, validated state exists."""

    metadata = getattr(model, "_loaded_meta_data", None)
    if cluster_contract is None:
        # A clustered checkpoint without validated state still fails here, before
        # partition parsing, edit-space construction, or a model forward.
        reject_feature_clusters_metadata(
            metadata,
            "feature_gcg optimize checkpoint",
        )
        return
    if not isinstance(metadata, Mapping):
        raise FeatureGcgError(
            "feature_gcg optimize checkpoint is missing mapping metadata"
        )
    try:
        assert_loaded_cluster_target(
            cluster_contract,
            model,
            metadata,
            surface="feature_gcg optimize checkpoint",
        )
    except ClusterFeatureContractError as exc:
        raise FeatureGcgError(str(exc)) from exc


def _n_sae_features(model: Any) -> int:
    """The ORIGINAL SAE dictionary width, read from ``model.encoder.config.n_sae_features``.

    This is the ``n_sae`` the optimizer reasons in — the ORIGINAL public width — NOT the
    encoder table's row count. The two COINCIDE only on an UNCOMPACTED encoder; a compacted
    encoder (#695) has ``V < n_sae`` rows, so reading the table height there would report
    ``V`` and silently shrink the candidate universe. Reading ``config.n_sae_features``
    keeps the width correct in both cases (the universal path derives its actual
    proposal universe from :func:`feature_vocab_allowed_ids`, not from this width).
    """
    try:
        return int(model.encoder.config.n_sae_features)
    except AttributeError as exc:
        raise FeatureGcgError(
            f"Cannot read n_sae_features from model.encoder.config.n_sae_features: {exc}. "
            "Is this an encoder-decoder model with a loaded encoder?"
        ) from exc


def _penalized_obj_batch(
    base: FeatureObjective,
    reg_lambda: float,
) -> Callable[[Sequence[Any]], list[float]]:
    """Return an ``objective_batch`` callable that adds ``count_penalty`` to each item's loss.

    Items arriving from ``model_item`` are ``(member, bag)`` tuples; this closure
    wraps them in :class:`_ItemAdapter` before forwarding to the base objective,
    then adds the L1 ``count_penalty(bag, reg_lambda) = reg_lambda * len(bag)`` to
    each scalar.
    """

    def _batch(items: Sequence[Any]) -> list[float]:
        adapted = [_ItemAdapter(member=m, bag=bag) for m, bag in items]
        base_losses = base.losses(adapted)
        return [
            loss + count_penalty(bag, reg_lambda)
            for loss, (_, bag) in zip(base_losses, items, strict=True)
        ]

    return _batch


def _grad_batch(
    base: FeatureObjective,
) -> Callable[[Sequence[Any], int], list[FeatureGrad]]:
    """Return a ``gradient_provider_batch`` callable that adapts ``(member, bag)`` tuples.

    The count penalty is NOT included in the gradient (count-independent
    proposal ranking; the cost is enforced by the exact penalized objective).
    """

    def _batch(items: Sequence[Any], step: int) -> list[FeatureGrad]:
        adapted = [_ItemAdapter(member=m, bag=bag) for m, bag in items]
        return base.gradients(adapted, step)

    return _batch


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def optimize_feature_conditioning_universal(
    model: Any,
    members: Sequence[Any],
    warm_start: FeatureConditioning,
    *,
    objective_factory: Callable[[Any], FeatureObjective],
    config: FeatureGcgConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureConditioning:
    """Optimize one shared feature-conditioning bag over one or many members.

    Returns the GCG-optimized :class:`~steering.conditional.feature_gcg.feature_conditioning.FeatureConditioning`
    bag that minimizes the penalized joint objective
    ``g(members, bag) + count_penalty(bag, reg_lambda)`` simultaneously
    over all members.

    Parameters
    ----------
    model:
        Loaded encoder-decoder model.  Used by ``objective_factory``, by
        :func:`~steering.conditional.feature_gcg.partition.partition_bounds_from_model`,
        and to read ``n_sae_features``.
    members:
        The N optimization units passed to the objective's ``losses`` /
        ``gradients`` methods via :class:`_ItemAdapter` ``(member=m, bag=bag)``.
        Concrete type depends on the objective (e.g.
        :class:`~steering.conditional.feature_gcg.gradient.FeatureGradMember`
        for LM loss,
        :class:`~steering.conditional.feature_gcg.objective.ClassMarginMember`
        for class-margin).
    warm_start:
        Initial :class:`~steering.conditional.feature_gcg.feature_conditioning.FeatureConditioning`
        bag; optimization starts here.
    objective_factory:
        ``callable(model) -> FeatureObjective``; the returned objective's
        ``losses`` / ``gradients`` methods receive :class:`_ItemAdapter` items.
    config:
        GCG hyperparameters.
    cluster_contract:
        ``None`` requires the checkpoint to carry no active cluster metadata and
        preserves the legacy rejection path. A supplied value must be a validated
        :class:`~steering.eval.cluster_feature_contract.ClusterFeatureContract`
        whose loaded state agrees with ``model``.

    Returns
    -------
    FeatureConditioning
        The GCG-optimized bag (the best full-batch bag seen across all steps).

    Raises
    ------
    FeatureGcgError
        If checkpoint and contract state disagree, or if a clustered warm start
        or final bag contains a consolidated-away feature. Representative
        validation errors are wrapped from ``FeatureClustersError`` with the
        original exception preserved as ``__cause__``.
    """
    _assert_cluster_contract(model, cluster_contract)
    # Scope the static-weight cache to this optimize: clear so the next gradient
    # call rebuilds from the current weights.  The ``finally`` below clears it
    # again so detached device tensors cannot outlive this call.
    clear_static_weight_cache()

    bounds = (
        partition_bounds_from_model(model, config)
        if cluster_contract is None
        else partition_bounds_from_model(
            model,
            config,
            cluster_contract=cluster_contract,
        )
    )
    objective = objective_factory(model)

    # Proposal universe (#720): on a COMPACTED checkpoint propose only the ORIGINAL ids in
    # its vocabulary ``V`` (``feature_vocab_allowed_ids`` → the encoder's own sorted
    # ``feature_vocab_ids`` buffer); on an uncompacted one (``allowed is None``) fall back
    # to the full ``range(n_sae)`` — byte-identical to the pre-#720 optimizer. The model's
    # ``feature_vocab`` is handed to the edit space so its ``_id_weight_rows`` can translate
    # those ORIGINAL ids to the compacted ``id_weight`` rows when it scores them (the sole
    # compacted-only op; ``None`` ⇒ that translation is a literal identity).
    allowed = feature_vocab_allowed_ids(model)
    if cluster_contract is None:
        allowed_feature_ids = NonEmptyNonNegativeIntTuple.parse(
            tuple(
                int(feature_id)
                for feature_id in (
                    allowed.tolist()
                    if allowed is not None
                    else range(_n_sae_features(model))
                )
            )
        )
    else:
        if allowed is None:
            raise FeatureGcgError(
                "feature_gcg clustered optimization requires the checkpoint's "
                "persistent FeatureVocab; an uncompacted/identity vocabulary "
                "cannot establish original-ID to compact-row translation"
            )
        representative_ids = t.tensor(
            cluster_contract.representative_universe,
            dtype=t.int64,
        )
        if representative_ids.numel() == 0:
            raise FeatureGcgError(
                "feature_gcg cluster contract has an empty representative universe"
            )

        # The sanctioned build path already proves this set equality:
        # contract representatives == nonzero(cluster_allowed_universe);
        # effective_feature_vocab.original_ids == that same nonzero universe;
        # checkpoint pinning requires FeatureVocab == effective_feature_vocab;
        # loaded-state validation requires model FeatureVocab == the pinned vocab.
        # Selecting representatives and checking ``missing`` are therefore
        # defense-in-depth against a forged/desynchronized contract-model pair;
        # the raise is unreachable for a contract produced by that build path.
        missing = representative_ids[
            ~t.isin(representative_ids, allowed.detach().to("cpu"))
        ]
        if missing.numel() > 0:
            raise FeatureGcgError(
                "feature_gcg representative universe contains IDs outside "
                "the checkpoint FeatureVocab: "
                f"{[int(x) for x in missing[:10].tolist()]}"
            )
        try:
            cluster_contract.consolidation.assert_representatives_only(
                t.tensor(warm_start.feature_ids(), dtype=t.int64),
                surface="feature_gcg warm start",
            )
        except FeatureClustersError as exc:
            raise FeatureGcgError(str(exc)) from exc
        allowed_feature_ids = NonEmptyNonNegativeIntTuple.parse(
            tuple(cluster_contract.representative_universe)
        )
    edit_space = FeatureConditioningEditSpace(
        bounds=bounds,
        allowed_feature_ids=allowed_feature_ids,
        warm_start=warm_start,
        feature_vocab=model.feature_vocab,
    )

    member_list = list(members)
    group = _Group(
        key_str=f"universal_{len(member_list)}m",
        members=member_list,
    )
    try:
        results = run_gcg(
            edit_space=edit_space,
            groups=[group],
            objective_batch=_penalized_obj_batch(objective, config.reg_lambda),
            gradient_provider_batch=_grad_batch(objective),
            cfg=config,
        )

        result = results[0].payload["bag"]
        if cluster_contract is not None:
            try:
                cluster_contract.consolidation.assert_representatives_only(
                    t.tensor(result.feature_ids(), dtype=t.int64),
                    surface="feature_gcg final bag",
                )
            except FeatureClustersError as exc:
                raise FeatureGcgError(str(exc)) from exc
        return result
    finally:
        # ``_static_weights`` materializes detached device tensors that are
        # intentionally shared across every gradient step in THIS optimize.
        # They are not model parameters, so moving the encoder to CPU does not
        # move or release them.  Clear on every exit (including an exception)
        # so a completed GCG group cannot pin CUDA memory while another model
        # is brought on-device for a subsequent DiffMean warm start.
        clear_static_weight_cache()


__all__ = ["optimize_feature_conditioning_universal"]
