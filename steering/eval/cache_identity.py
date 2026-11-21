"""Shared CacheIdentity resolution for eval context caches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.feature_clusters import (
    canonical_feature_clusters_descriptor,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import canonical_dict, resolve_feature_mask
from steering.eval.context_cache_io import (
    CacheIdentity,
    require_identity_field,
    validate_feature_selection_hash,
)
from steering.eval.feature_selection_hash import compute_canonical_feature_selection_hash
from steering.eval.turn_context_contract import (
    TurnContextContract,
    turn_context_contract_hash,
    turn_context_identity_fields,
)
from steering.packed_batch_io import compute_selection_hash
from steering.removed_features import reject_feature_clustering_config
from steering.segmentation import SegmentMode, segment_mode_from_metadata


def resolve_feature_selection_hash(
    identity_cfg: Mapping[Any, Any],
    cond: Mapping[Any, Any] | None,
    *,
    turn_contract: TurnContextContract | None = None,
    checkpoint_contract: CheckpointFeatureContract | None = None,
) -> str:
    """Resolve and validate the effective feature-selection hash.

    Explicit ``identity.feature_selection_hash`` is accepted only as a
    64-character hex string.  When the conditioning block also carries the
    canonical pipeline parameters, the computed hash must match the explicit
    value.  If no explicit hash is present, the canonical parameters are used.
    Missing both is invalid for identity-scoped eval caches.
    """

    if not isinstance(identity_cfg, Mapping):
        raise TypeError("conditioning.identity must be a mapping")
    if cond is not None and not isinstance(cond, Mapping):
        raise TypeError("conditioning must be a mapping or null")

    explicit_raw = identity_cfg.get("feature_selection_hash")
    explicit_hash = (
        None
        if explicit_raw is None
        else validate_feature_selection_hash(
            str(explicit_raw),
            field_name="conditioning.identity.feature_selection_hash",
        )
    )
    if turn_contract is not None:
        canonical_hash = turn_context_contract_hash(turn_contract)
        if explicit_hash is not None and explicit_hash != canonical_hash:
            raise ValueError("explicit turn feature_selection_hash does not match checkpoint contract")
        return canonical_hash
    if segment_mode_from_metadata(cond, source="cache identity") is SegmentMode.turn_delimited_fixed_tokens:
        raise ValueError("turn cache identity requires a complete checkpoint-derived contract")
    canonical_hash = _canonical_hash_from_conditioning(cond, checkpoint_contract=checkpoint_contract)
    if explicit_hash is not None and canonical_hash is not None:
        if explicit_hash != canonical_hash:
            # Codex cross-review fix (Finding 4, MEDIUM): the unclustered
            # and clustered branches of _canonical_hash_from_conditioning
            # hash a genuinely DIFFERENT input set -- the unclustered tail
            # folds stats_source_sha256 into delta_config; the clustered
            # branch omits it (statistics identity is independently pinned
            # by the clustered feature contract + cache manifest instead)
            # and additionally hashes the real conditioning.seed. A single
            # unconditional message naming stats_source_sha256 for BOTH
            # cases was actively wrong for clustered configs.
            clustered = cond is not None and cond.get("feature_clusters") is not None
            canonical_inputs = (
                "conditioning.feature_selection, conditioning.negative_selection, "
                "conditioning.num_buckets, conditioning.seed, conditioning.feature_mask, "
                "and conditioning.feature_clusters (delta_weights retained as null, "
                "statistics identity independently pinned elsewhere -- NOT folded "
                "into this hash)"
                if clustered
                else "conditioning.feature_selection, conditioning.negative_selection, "
                "conditioning.num_buckets, and conditioning.stats_source_sha256"
            )
            raise ValueError(
                "conditioning.identity.feature_selection_hash does not match "
                f"the canonical hash computed from {canonical_inputs} "
                "(including conditional mask, cluster, delta, and "
                "splice-batching semantics): "
                f"explicit={explicit_hash}, canonical={canonical_hash}"
            )
        return explicit_hash
    if explicit_hash is not None:
        return explicit_hash
    if canonical_hash is not None:
        return canonical_hash
    raise ValueError(
        "conditioning.identity.feature_selection_hash is required for eval "
        "context caches. Set a 64-character hex hash, or provide "
        "conditioning.feature_selection, conditioning.num_buckets, and "
        "conditioning.stats_source_sha256 so the canonical hash can be "
        "computed. Old conditioning.positive_selection configs are not "
        "auto-resolved."
    )


def cache_identity_from_config(
    identity_cfg: Mapping[Any, Any] | None,
    cond: Mapping[Any, Any] | None = None,
    *,
    prefix: str = "conditioning.identity",
    turn_contract: TurnContextContract | None = None,
    checkpoint_contract: CheckpointFeatureContract | None = None,
) -> CacheIdentity:
    """Build a validated :class:`CacheIdentity` from config blocks."""

    if turn_contract is not None:
        fields = turn_context_identity_fields(turn_contract)
        for field, expected in fields.items():
            supplied = (identity_cfg or {}).get(field)
            if supplied is not None and supplied != expected:
                raise ValueError(f"{prefix}.{field} conflicts with checkpoint turn contract")
        return CacheIdentity(**fields, feature_selection_hash=resolve_feature_selection_hash(
            identity_cfg or {}, cond, turn_contract=turn_contract,
        ))
    if identity_cfg is None:
        raise ValueError(
            f"{prefix} is required to specify the cache identity "
            "(model_id/sae_id/layer/pooling/feature_selection_hash)"
        )
    if not isinstance(identity_cfg, Mapping):
        raise TypeError(f"{prefix} must resolve to a mapping")
    if cond is not None and not isinstance(cond, Mapping):
        raise TypeError("conditioning must resolve to a mapping or null")
    return CacheIdentity(
        model_id=require_identity_field(identity_cfg, "model_id", str),
        sae_id=require_identity_field(identity_cfg, "sae_id", str),
        layer=require_identity_field(identity_cfg, "layer", int),
        pooling=require_identity_field(identity_cfg, "pooling", str),
        feature_selection_hash=resolve_feature_selection_hash(identity_cfg, cond, checkpoint_contract=checkpoint_contract),
    )


def _canonical_hash_from_conditioning(cond: Mapping[Any, Any] | None, *,
    checkpoint_contract: CheckpointFeatureContract | None = None,
) -> str | None:
    if cond is None:
        return None
    # The removed singular producer marker remains invalid. It must never be
    # confused with the supported plural static-cluster descriptor.
    reject_feature_clustering_config(cond, "eval context cache conditioning")
    fs_sel = cond.get("feature_selection")
    stats_sha = cond.get("stats_source_sha256")
    if fs_sel is None or stats_sha is None:
        return None
    fs_d = _as_dict(fs_sel, "conditioning.feature_selection")
    neg_sel_raw = cond.get("negative_selection")
    neg_sel_d = (
        None if neg_sel_raw is None else _as_dict(neg_sel_raw, "conditioning.negative_selection")
    )
    # S-07-11 P0 fix: thread feature_delta through to the hash so two pipelines
    # that differ only in delta mode/neg_delta_ratio do NOT collide on cache identity.
    # When the conditioning block omits feature_delta, the hash is unchanged
    # (backwards compatibility for existing eval caches).
    feature_clusters_raw = cond.get("feature_clusters")
    feature_delta_raw = cond.get("feature_delta")
    if feature_clusters_raw is not None:
        if "delta_weights" in fs_d:
            raise ValueError(
                "clustered prompt-context cache feature_selection must omit "
                "delta_weights entirely; prompt contexts have no current/reference "
                "pair"
            )
        if feature_delta_raw is not None:
            raise ValueError(
                "clustered prompt-context caches require feature_delta=null; "
                "splice/delta selection is reserved for inline steering"
            )
    feature_delta_d = (
        None
        if feature_delta_raw is None
        else _as_dict(feature_delta_raw, "conditioning.feature_delta")
    )
    num_buckets_raw = cond.get("num_buckets")
    if num_buckets_raw is None:
        raise ValueError(
            "conditioning.num_buckets is required to compute canonical "
            "feature_selection_hash"
        )
    try:
        num_buckets = int(num_buckets_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "conditioning.num_buckets must be a positive integer to compute "
            f"canonical feature_selection_hash; got {num_buckets_raw!r}"
        ) from exc
    if num_buckets <= 0:
        raise ValueError(
            "conditioning.num_buckets must be a positive integer to compute "
            f"canonical feature_selection_hash; got {num_buckets}"
        )
    # Resolve paths through the production parsers before hashing. This keeps
    # cache identity path-independent and makes inline ID/cluster reordering
    # canonical. No mask/clusters remains the exact legacy None/absent hash.
    feature_mask_raw = cond.get("feature_mask")
    feature_mask_d: dict | None = None
    feature_clusters_d: dict | None = None
    if feature_mask_raw is not None or feature_clusters_raw is not None or checkpoint_contract is not None:
        n_sae_raw = cond.get("n_sae_features")
        if n_sae_raw is None:
            raise ValueError(
                "conditioning.n_sae_features is required to resolve "
                "feature_mask/feature_clusters for canonical cache identity"
            )
        try:
            n_sae_features = int(n_sae_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "conditioning.n_sae_features must be a positive integer"
            ) from exc
        if n_sae_features <= 0:
            raise ValueError(
                "conditioning.n_sae_features must be a positive integer"
            )
        if checkpoint_contract is None:
            resolved_mask = resolve_feature_mask(cond, n_sae_features)
            resolved_clusters = resolve_feature_clusters(cond, n_sae_features)
        else:
            resolved_mask = checkpoint_contract.feature_mask
            resolved_clusters = checkpoint_contract.resolved_clusters
            expected_clusters = None if resolved_clusters is None else resolved_clusters.canonical_dict()
            if feature_mask_raw != canonical_dict(resolved_mask) or feature_clusters_raw != expected_clusters:
                raise ValueError("cache config universe conflicts with checkpoint feature contract")
        feature_mask_d = canonical_dict(resolved_mask)
        feature_clusters_d = canonical_feature_clusters_descriptor(
            resolved_clusters
        )
    if feature_clusters_d is not None:
        # D-seed ANTI-DESYNC (site6-eval-context-cache-clustered — see
        # tests/steering/test_selection_hash_cross_site_consistency.py,
        # which enumerates and behaviorally-parity-checks this branch
        # alongside the other 5 production compute_selection_hash call
        # sites). A clustered eval-context cache is inseparable from the
        # compacted checkpoint that consumes it, so its identity must
        # reproduce the checkpoint's REAL training-preselection seed
        # (conditioning.seed) instead of the legacy eval-cache hash's
        # hardcoded seed=0 with the statistics digest folded into
        # delta_config. This mirrors steering/eval/cluster_checkpoint_pin.py's
        # site 5 (_resolve_selection_identity), which pins the same
        # checkpoint-real seed for the analogous checkpoint-preflight
        # identity. feature_mask_d/feature_clusters_d above already support
        # feature_mask being absent for clustered builds (mask-optional
        # tolerance) -- that is preserved unchanged; this branch adds only
        # the seed-pinned hash source.
        #
        # Named deviation (matches the oracle, deliberate, not an
        # oversight): delta_config=None below means stats_source_sha256
        # (resolved above, still required as a presence-gate to reach this
        # branch at all) is NOT folded into this hash -- unlike the
        # unclustered tail below, whose delta_config always carries it.
        # Statistics remain independently pinned by the clustered feature
        # contract and cache manifest instead of by this hash. Track D's
        # scheduled D-contract PR adds the actual checkpoint-bound
        # cross-check (ClusterFeatureContractRequest /
        # resolve_cluster_feature_contract, see
        # steering/eval/cluster_feature_contract.py) that validates
        # statistics/mask/cluster/seed agreement against the target
        # checkpoint directly; until it lands, a stats-only drift for an
        # otherwise-matching clustered config is not caught by this hash
        # alone.
        seed_raw = cond.get("seed")
        if type(seed_raw) is not int or seed_raw < 0:
            raise ValueError(
                "conditioning.seed must be a non-negative integer to "
                "compute the clustered canonical feature_selection_hash "
                "(cache identity is pinned to the real training-"
                f"preselection seed); got {seed_raw!r}"
            )
        # feature_delta=null is already enforced above whenever
        # feature_clusters is configured (see the feature_clusters_raw /
        # feature_delta_raw guard earlier in this function, which raises
        # before this point is ever reached), so delta_config=None here is
        # provably the only reachable value -- no redundant re-check.
        # splice_batch_size is likewise definitionally inert once
        # delta_config=None (compute_selection_hash's
        # canonical_splice_selection_batching short-circuits on a null
        # delta_config); both are passed as explicit None literals purely
        # to keep this call inside the ANTI-DESYNC kwarg-name set (see
        # TestSite6BehavioralParity's seed-dependency sentinel).
        #
        # Codex cross-review fix (Finding 1, HIGH): compute_selection_hash
        # hashes raw KEY PRESENCE, not the parsed value (steering.packed_
        # batch_io.compute_selection_hash; SamplingSelection parsing treats
        # an absent delta_weights key and delta_weights=None identically --
        # steering/conditional/sampling.py -- but the hash payload does
        # not). The packed-row producer's delta-free selection payload
        # (scripts/train/conditional_encdec_train.py's discover_packed_
        # selection, the `base_val_feature_selection_raw` splice_cf
        # original/base-validation branch: `dict(feature_selection_raw);
        # [...]["delta_weights"] = None`) RETAINS the key with a null
        # value; the guard above only forbids feature_clusters_raw
        # configs from setting delta_weights in the SOURCE config (a
        # different, syntactic check) -- it says nothing about what the
        # HASH PAYLOAD must contain. This branch must match the packed-row
        # producer's payload convention exactly (retain the null key), not
        # merely pass through fs_d's raw shape (which happens to omit the
        # key entirely because conditioning.feature_selection's source
        # config never defines it) -- otherwise a clustered eval-cache
        # identity can never agree with the checkpoint's real packed-row
        # selection identity it is supposed to reproduce.
        #
        # Only for sampling-shaped selections ('distribution' present):
        # delta_weights is a SamplingSelection-only concept
        # (steering/conditional/sampling.py's parse_feature_selection is a
        # closed per-arm schema -- a threshold selection dict with a
        # delta_weights key is rejected as an unknown key before ever
        # reaching a SamplingSelection check). Clustered prompt caches are
        # already required to be sampling-shaped by their downstream
        # consumers (e.g. build_eval_context_cache.py's SamplingBundle
        # check); this branch stays parse-agnostic (compute_selection_hash
        # never parses its inputs) but must not inject a sampling-only key
        # into what could conceptually be a threshold-shaped payload.
        positive_selection_d = (
            {**fs_d, "delta_weights": None} if "distribution" in fs_d else dict(fs_d)
        )
        return validate_feature_selection_hash(
            compute_selection_hash(
                positive_selection=positive_selection_d,
                negative_selection=neg_sel_d,
                delta_config=None,
                num_buckets=num_buckets,
                seed=seed_raw,
                ddp_rank=0,
                feature_mask=feature_mask_d,
                feature_clusters=feature_clusters_d,
                splice_batch_size=None,
            ),
            field_name="canonical feature_selection_hash",
        )
    return validate_feature_selection_hash(
        compute_canonical_feature_selection_hash(
            positive_selection=fs_d,
            negative_selection=neg_sel_d,
            num_buckets=num_buckets,
            stats_source_sha256=str(stats_sha),
            feature_delta=feature_delta_d,
            feature_mask=feature_mask_d,
            feature_clusters=feature_clusters_d,
            splice_batch_size=cond.get("selection_splice_batch_size"),
        ),
        field_name="canonical feature_selection_hash",
    )
def _as_dict(value, label: str) -> dict:
    resolved = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if not isinstance(resolved, dict):
        expected = "a dict" if label.endswith("feature_selection") else "a dict or null"
        raise ValueError(f"{label} must be {expected}; got {type(resolved).__name__}")
    return dict(resolved)
