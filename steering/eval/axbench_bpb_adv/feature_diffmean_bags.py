"""Feature-DiffMean conditioning-bag derive for BPB-Adv."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch as t
from loguru import logger

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.vocab_restriction import (
    checkpoint_vocab_fingerprint,
    feature_vocab_allowed_ids,
)
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.feature_diffmean import (
    encdec_dec_sequence_len,
    feature_diffmean_bag,
)
from steering.eval.axbench_bpb_adv.gcg_bags import (
    ClusteredWarmStartCache,
    _assert_prefix_tokens_roundtrip,
    _group_g_steer,
    _group_prefixes_by_concept_mode,
    _members_for_prefixes,
    _prefixed_completions,
    _sae_layer_for_cfg,
)
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths, format_reg_lambda
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    VOCAB_FINGERPRINT_META_KEY,
    write_ours_gcg_bag,
    write_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    FAIR_FEATURE_DIFFMEAN_METHOD_MODES,
    ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES,
    PrecomputedBagFamily,
    active_precomputed_bag_modes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_identity import (
    BAG_DERIVE_SCHEMA_VERSION,
    BAG_DERIVE_SCHEMA_VERSION_KEY,
    committed_bag_for_reuse,
)
from steering.eval.axbench_bpb_adv.prefix_split import (
    split_prefixes_for_concept,
)
from steering.eval.target_conditioning import ContextFeatureSegment

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


def _base_methods_with_representative_targets(
    cfg: AxBenchBpbAdvConfig,
) -> list[str] | None:
    if cfg.base_methods is None:
        return None
    return [*cfg.base_methods, *cfg.representative_target_methods]


def active_fair_feature_diffmean_modes(cfg: AxBenchBpbAdvConfig) -> set[SteeringMode]:
    """Return steering modes whose feature-DiffMean bag method is active."""
    return active_precomputed_bag_modes(
        _base_methods_with_representative_targets(cfg),
        FAIR_FEATURE_DIFFMEAN_METHOD_MODES,
    )


def fair_feature_diffmean_methods_active(cfg: AxBenchBpbAdvConfig) -> bool:
    """True iff a feature-DiffMean conditioning-bag method is active."""
    return bool(active_fair_feature_diffmean_modes(cfg))


def active_roofline_feature_diffmean_modes(
    cfg: AxBenchBpbAdvConfig,
) -> set[SteeringMode]:
    """Return modes whose roofline feature-DiffMean bag method is active."""
    return active_precomputed_bag_modes(
        _base_methods_with_representative_targets(cfg),
        ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES,
    )


def roofline_feature_diffmean_methods_active(cfg: AxBenchBpbAdvConfig) -> bool:
    """True iff a roofline feature-DiffMean conditioning-bag method is active."""
    return bool(active_roofline_feature_diffmean_modes(cfg))


def build_feature_diffmean_bags_fair(
    *,
    prefixes: Sequence[BpbAttackPrefix],
    model: Any,
    base_model: Any,
    sae: Any,
    percentile_table: t.Tensor,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    device: Any,
    tokenizer: Any,
    context_segments_by_prefix_hash: dict[str, tuple[ContextFeatureSegment, ...]],
    cluster_contract: ClusterFeatureContract | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache scored feature-DiffMean conditioning bags.

    This is the scored conditioning-signal-space DiffMean baseline used as the
    representative-coefficient target. It shares the fair feature-GCG held-out
    split so scored rows are not measured on optimize prefixes, but it stops
    after deterministic :func:`feature_diffmean_bag` instead of running GCG.
    """
    torch_device = t.device(device) if not isinstance(device, t.device) else device
    cluster_kwargs = (
        {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    )
    if clustered_warm_starts is not None:
        clustered_warm_starts.assert_matches(
            cluster_contract,
            surface="feature-DiffMean derive",
        )
    positive_bucket, negative_bucket = assert_single_positive_bucket(
        model,
        **cluster_kwargs,
    )
    k_pos, k_neg = resolve_k(model, cfg.ours_gcg, **cluster_kwargs)
    sae_layer = _sae_layer_for_cfg(cfg)
    ours = cfg.ours_gcg
    max_seq_len = encdec_dec_sequence_len(model)
    # A compacted target checkpoint's allowed SAE feature vocabulary V (None on
    # an uncompacted checkpoint) -- restricts feature_diffmean_bag's selection
    # so it can never pick an id that trips the encoder's OOV assert.
    allowed_ids = feature_vocab_allowed_ids(model)

    groups = _group_prefixes_by_concept_mode(prefixes)
    logger.info(
        f"feature-DiffMean derive: {len(groups)} (concept,mode) group(s), "
        f"k_pos={k_pos} k_neg={k_neg} buckets=({positive_bucket},{negative_bucket})"
    )

    for concept_id, mode, group_prefixes in groups:
        optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
            group_prefixes, ratio=ours.split_ratio, seed=ours.split_seed
        )
        if not held_out_prefixes:
            raise AxBenchBpbAdvError(
                "feature-DiffMean derive produced no HELD-OUT prefixes for "
                f"concept_id={concept_id} mode={mode.value} "
                f"(n_prefixes={len(group_prefixes)}, split_ratio={ours.split_ratio}); "
                "the held-out remainder is the scored set."
            )
        for prefix in held_out_prefixes:
            _assert_prefix_tokens_roundtrip(prefix, tokenizer=tokenizer)

        optimize_members = _members_for_prefixes(
            optimize_prefixes,
            device=torch_device,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            allowed_ids=allowed_ids,
            **cluster_kwargs,
        )
        held_out_members = _members_for_prefixes(
            held_out_prefixes,
            device=torch_device,
            context_segments_by_prefix_hash=context_segments_by_prefix_hash,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            allowed_ids=allowed_ids,
            **cluster_kwargs,
        )
        if not optimize_members:
            raise AxBenchBpbAdvError(
                "feature-DiffMean derive produced no optimize members for "
                f"concept_id={concept_id} mode={mode.value} "
                f"(optimize_prefixes={len(optimize_prefixes)}, "
                f"split_ratio={ours.split_ratio}); an empty optimize split cannot "
                "produce a DiffMean bag. Increase split_ratio or provide more prefixes."
            )

        rich_items, avoid_items = _prefixed_completions(
            optimize_prefixes, tokenizer=tokenizer
        )
        if clustered_warm_starts is None:
            # Exact legacy/direct-call path. Clustered production reuses the bag
            # computed during the extractor-only warm-start phase.
            bag = feature_diffmean_bag(
                rich_items,
                avoid_items,
                sae=sae,
                model=base_model,
                sae_layer=sae_layer,
                percentile_table=percentile_table,
                mode=mode,
                k_pos=k_pos,
                k_neg=k_neg,
                positive_bucket=positive_bucket,
                negative_bucket=negative_bucket,
                device=torch_device,
                extraction_batch_size=cfg.feature_diffmean_extraction_batch_size,
                allowed_ids=allowed_ids,
                **cluster_kwargs,
            )
        else:
            bag = clustered_warm_starts.fair_bag(
                concept_id,
                mode,
                contract=cluster_contract,
                surface="feature-DiffMean derive",
            )
        group_g_steer = _group_g_steer(model, ours, mode, torch_device)

        family = PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
        path = paths.ours_gcg_bag(concept_id, mode, 0.0, family)
        vocab_fingerprint = checkpoint_vocab_fingerprint(model)
        identity = {
            "concept_id": str(int(concept_id)),
            "mode": mode.value,
            "mode_tag": family.value,
            "coefficient_str": "None",
            "reg_lambda": format_reg_lambda(0.0),
            "n_optimize_prefixes": str(len(optimize_prefixes)),
            "n_held_out_prefixes": str(len(held_out_prefixes)),
            "n_optimize_members": str(len(optimize_members)),
            "n_held_out_members": str(len(held_out_members)),
            "n_rich_completions": str(len(rich_items)),
            "n_avoid_completions": str(len(avoid_items)),
            "k_pos": str(int(k_pos)),
            "k_neg": str(int(k_neg)),
            "positive_bucket": str(int(positive_bucket)),
            "negative_bucket": str(int(negative_bucket)),
            "extraction_batch_size": str(cfg.feature_diffmean_extraction_batch_size),
            "split_ratio": repr(float(ours.split_ratio)),
            "split_seed": str(int(ours.split_seed)),
        }
        if cluster_contract is not None:
            identity["feature_contract_fingerprint"] = (
                cluster_contract.feature_contract_fingerprint
            )
            identity[BAG_DERIVE_SCHEMA_VERSION_KEY] = BAG_DERIVE_SCHEMA_VERSION
        # Reuse is checked BEFORE the two group_g_steer measurements so a resumed run
        # skips those forwards as well as the write. DiffMean is forward-only and so
        # byte-stable today, but that is a consequence of its shape rather than a
        # guarantee; keeping one reuse rule across every clustered writer means a
        # future change upstream cannot quietly reintroduce #1050 here.
        committed = (
            None
            if cluster_contract is None
            else committed_bag_for_reuse(
                path,
                expected_identity={
                    **identity,
                    VOCAB_FINGERPRINT_META_KEY: vocab_fingerprint,
                },
                expected_feature_contract_fingerprint=(
                    cluster_contract.feature_contract_fingerprint
                ),
            )
        )
        if committed is not None:
            logger.debug(
                "fair feature-DiffMean bag reused: concept_id={} mode={} "
                "n_features={} <- {}",
                concept_id,
                mode.value,
                len(committed.bag),
                path,
            )
        else:
            optimize_g_steer = group_g_steer(optimize_members, bag)
            held_out_g_steer = group_g_steer(held_out_members, bag)
            metadata = {
                **identity,
                "n_features": str(len(bag)),
                "optimize_g_steer": repr(float(optimize_g_steer)),
                "held_out_g_steer": repr(float(held_out_g_steer)),
            }
            write_ours_gcg_bag(
                path, bag, metadata=metadata, vocab_fingerprint=vocab_fingerprint
            )
            logger.info(
                f"feature-DiffMean bag written: concept_id={concept_id} "
                f"mode={mode.value} n_features={len(bag)} "
                f"opt_g_steer={optimize_g_steer:.5f} "
                f"held_g_steer={held_out_g_steer:.5f} -> {path}"
            )

        scored_prefix_path = paths.ours_gcg_scored_prefixes(concept_id, mode, family)
        write_ours_gcg_scored_prefixes(
            scored_prefix_path,
            [str(prefix.core_prefix_id) for prefix in held_out_prefixes],
            concept_id=concept_id,
            mode=mode,
            family=family,
            **(
                {}
                if cluster_contract is None
                else {
                    "feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
        logger.info(
            f"feature-DiffMean scored prefixes: concept_id={concept_id} "
            f"mode={mode.value} held_out={len(held_out_prefixes)} -> "
            f"{scored_prefix_path}"
        )


def build_feature_diffmean_bags_roofline(
    *,
    prefixes: Sequence[BpbAttackPrefix],
    model: Any,
    base_model: Any,
    sae: Any,
    percentile_table: t.Tensor,
    cfg: AxBenchBpbAdvConfig,
    paths: AxBenchBpbAdvPaths,
    device: Any,
    tokenizer: Any,
    context_segments_by_prefix_hash: dict[str, tuple[ContextFeatureSegment, ...]],
    cluster_contract: ClusterFeatureContract | None = None,
    clustered_warm_starts: ClusteredWarmStartCache | None = None,
) -> None:
    """Derive + cache roofline feature-DiffMean bags on fair held-out prefixes.

    This is the in-sample upper bound for conditioning-signal DiffMean: each
    held-out prefix's own rich/avoid completions produce the bag, and that same
    prefix is scored under it. The scored prefix set is exactly the fair held-out
    split so fair-vs-roofline class-margin rows are measured on the same source
    population; only the conditioning signal has oracle access.
    """
    torch_device = t.device(device) if not isinstance(device, t.device) else device
    cluster_kwargs = (
        {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    )
    if clustered_warm_starts is not None:
        clustered_warm_starts.assert_matches(
            cluster_contract,
            surface="roofline feature-DiffMean derive",
        )
    positive_bucket, negative_bucket = assert_single_positive_bucket(
        model,
        **cluster_kwargs,
    )
    k_pos, k_neg = resolve_k(model, cfg.ours_gcg, **cluster_kwargs)
    sae_layer = _sae_layer_for_cfg(cfg)
    ours = cfg.ours_gcg
    max_seq_len = encdec_dec_sequence_len(model)
    # A compacted target checkpoint's allowed SAE feature vocabulary V (None on
    # an uncompacted checkpoint) -- restricts feature_diffmean_bag's selection
    # so it can never pick an id that trips the encoder's OOV assert.
    allowed_ids = feature_vocab_allowed_ids(model)

    groups = _group_prefixes_by_concept_mode(prefixes)
    logger.info(
        f"roofline feature-DiffMean derive: {len(groups)} (concept,mode) group(s), "
        f"split_ratio={ours.split_ratio} k_pos={k_pos} "
        f"k_neg={k_neg} buckets=({positive_bucket},{negative_bucket})"
    )

    for concept_id, mode, group_prefixes in groups:
        optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
            group_prefixes, ratio=ours.split_ratio, seed=ours.split_seed
        )
        if not held_out_prefixes:
            raise AxBenchBpbAdvError(
                "roofline feature-DiffMean derive selected no HELD-OUT prefixes for "
                f"concept_id={concept_id} mode={mode.value} "
                f"(n_prefixes={len(group_prefixes)}, split_ratio={ours.split_ratio}); an empty "
                "selection would silently drop every roofline score downstream."
            )
        logger.info(
            f"roofline feature-DiffMean derive: concept_id={concept_id} "
            f"mode={mode.value} prefixes={len(group_prefixes)} "
            f"optimize={len(optimize_prefixes)} held_out={len(held_out_prefixes)} "
            f"(split_ratio={ours.split_ratio})"
        )

        group_g_steer = _group_g_steer(model, ours, mode, torch_device)
        vocab_fingerprint = checkpoint_vocab_fingerprint(model)
        scored_prefix_ids: list[str] = []
        for prefix in held_out_prefixes:
            _assert_prefix_tokens_roundtrip(prefix, tokenizer=tokenizer)
            rich_items, avoid_items = _prefixed_completions(
                [prefix], tokenizer=tokenizer
            )
            prefix_members = _members_for_prefixes(
                [prefix],
                device=torch_device,
                context_segments_by_prefix_hash=context_segments_by_prefix_hash,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                allowed_ids=allowed_ids,
                **cluster_kwargs,
            )
            if not prefix_members:
                raise AxBenchBpbAdvError(
                    "roofline feature-DiffMean derive produced no class-margin "
                    f"members for concept_id={concept_id} mode={mode.value} "
                    f"prefix_id={prefix.core_prefix_id!r}; an empty result is a "
                    "real data/config inconsistency, not a skippable case."
                )

            if clustered_warm_starts is None:
                # Exact legacy/direct-call path. Clustered production reuses the
                # bag computed during the extractor-only warm-start phase.
                bag = feature_diffmean_bag(
                    rich_items,
                    avoid_items,
                    sae=sae,
                    model=base_model,
                    sae_layer=sae_layer,
                    percentile_table=percentile_table,
                    mode=mode,
                    k_pos=k_pos,
                    k_neg=k_neg,
                    positive_bucket=positive_bucket,
                    negative_bucket=negative_bucket,
                    device=torch_device,
                    extraction_batch_size=cfg.feature_diffmean_extraction_batch_size,
                    allowed_ids=allowed_ids,
                    **cluster_kwargs,
                )
            else:
                bag = clustered_warm_starts.roofline_bag(
                    concept_id,
                    mode,
                    str(prefix.core_prefix_id),
                    contract=cluster_contract,
                    surface="roofline feature-DiffMean derive",
                )
            family = PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN
            path = paths.ours_gcg_bag(
                concept_id,
                mode,
                0.0,
                family,
                prefix_id=prefix.core_prefix_id,
            )
            identity = {
                "concept_id": str(int(concept_id)),
                "mode": mode.value,
                "mode_tag": family.value,
                "prefix_id": str(prefix.core_prefix_id),
                "coefficient_str": "None",
                "reg_lambda": format_reg_lambda(0.0),
                "n_prefix_members": str(len(prefix_members)),
                "n_rich_completions": str(len(rich_items)),
                "n_avoid_completions": str(len(avoid_items)),
                "k_pos": str(int(k_pos)),
                "k_neg": str(int(k_neg)),
                "positive_bucket": str(int(positive_bucket)),
                "negative_bucket": str(int(negative_bucket)),
                "extraction_batch_size": str(cfg.feature_diffmean_extraction_batch_size),
                "split_seed": str(int(ours.split_seed)),
                "split_ratio": repr(float(ours.split_ratio)),
                "scored_split": "held_out",
            }
            if cluster_contract is not None:
                identity["feature_contract_fingerprint"] = (
                    cluster_contract.feature_contract_fingerprint
                )
                identity[BAG_DERIVE_SCHEMA_VERSION_KEY] = BAG_DERIVE_SCHEMA_VERSION
            committed = (
                None
                if cluster_contract is None
                else committed_bag_for_reuse(
                    path,
                    expected_identity={
                        **identity,
                        VOCAB_FINGERPRINT_META_KEY: vocab_fingerprint,
                    },
                    expected_feature_contract_fingerprint=(
                        cluster_contract.feature_contract_fingerprint
                    ),
                )
            )
            if committed is not None:
                logger.debug(
                    "roofline feature-DiffMean bag reused: concept_id={} mode={} "
                    "prefix_id={} n_features={} <- {}",
                    concept_id,
                    mode.value,
                    prefix.core_prefix_id,
                    len(committed.bag),
                    path,
                )
            else:
                in_sample_g_steer = group_g_steer(prefix_members, bag)
                metadata = {
                    **identity,
                    "n_features": str(len(bag)),
                    "in_sample_g_steer": repr(float(in_sample_g_steer)),
                }
                write_ours_gcg_bag(
                    path,
                    bag,
                    metadata=metadata,
                    vocab_fingerprint=vocab_fingerprint,
                )
                logger.info(
                    f"roofline feature-DiffMean bag written: concept_id={concept_id} "
                    f"mode={mode.value} prefix_id={prefix.core_prefix_id} "
                    f"n_features={len(bag)} in_sample_g_steer={in_sample_g_steer:.5f} "
                    f"-> {path}"
                )
            # MUST run on the reuse path too. This append is in the prefix body with
            # no lambda loop beneath it, so the `continue`-on-reuse shape used in
            # gcg_bags.py would silently drop this prefix from the scored set and the
            # scored-prefixes artifact would disagree with the bags on disk.
            scored_prefix_ids.append(str(prefix.core_prefix_id))

        if not scored_prefix_ids:
            raise AxBenchBpbAdvError(
                "roofline feature-DiffMean derive produced no scored prefixes for "
                f"concept_id={concept_id} mode={mode.value}."
            )
        scored_prefix_path = paths.ours_gcg_scored_prefixes(
            concept_id,
            mode,
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        )
        write_ours_gcg_scored_prefixes(
            scored_prefix_path,
            scored_prefix_ids,
            concept_id=concept_id,
            mode=mode,
            family=PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            **(
                {}
                if cluster_contract is None
                else {
                    "feature_contract_fingerprint": (
                        cluster_contract.feature_contract_fingerprint
                    )
                }
            ),
        )
        logger.info(
            f"roofline feature-DiffMean scored prefixes: concept_id={concept_id} "
            f"mode={mode.value} scored={len(scored_prefix_ids)} -> "
            f"{scored_prefix_path}"
        )


__all__ = [
    "active_fair_feature_diffmean_modes",
    "active_roofline_feature_diffmean_modes",
    "build_feature_diffmean_bags_fair",
    "build_feature_diffmean_bags_roofline",
    "fair_feature_diffmean_methods_active",
    "roofline_feature_diffmean_methods_active",
]
