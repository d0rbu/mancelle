"""Shared construction of BPB-Adv clustered preflight requests."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.cluster_feature_contract import ClusterFeatureContractRequest

if TYPE_CHECKING:
    from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig


def build_cluster_contract_request(
    cfg: AxBenchBpbAdvConfig,
) -> ClusterFeatureContractRequest | None:
    """Build BPB's independent config/source/target preflight request."""

    # G2-D12: the pairing check is SYMMETRIC. The oracle returned None whenever
    # clusters were absent without looking at the mask, so a mask-only config
    # silently disabled the preflight instead of failing loud -- the mirror
    # image of the clusters-without-mask case it did reject. validate_config
    # currently rejects the asymmetry upstream, so this was bounded, but a
    # fail-loud helper must not depend on a caller's validation to be correct.
    if cfg.feature_clusters is None:
        if cfg.feature_mask is not None:
            raise AxBenchBpbAdvError(
                "clustered BPB preflight requires feature_clusters with feature_mask"
            )
        return None
    if cfg.feature_mask is None:
        raise AxBenchBpbAdvError(
            "clustered BPB preflight requires feature_mask with feature_clusters"
        )
    encdec = cfg.encdec_target
    source = encdec.get("source")
    model_tag = encdec.get("model_tag")
    if not isinstance(source, str) or not source:
        raise AxBenchBpbAdvError(
            "clustered encdec_target.source must be a non-empty string"
        )
    if not isinstance(model_tag, str) or not model_tag:
        raise AxBenchBpbAdvError(
            "clustered encdec_target.model_tag must be a non-empty string"
        )
    raw_step = encdec.get("step")
    if raw_step in (None, "last"):
        step = None
    elif isinstance(raw_step, Integral) and not isinstance(raw_step, bool):
        step = int(raw_step)
    else:
        raise AxBenchBpbAdvError(
            "clustered encdec_target.step must be a non-negative integer, null, "
            f"or 'last'; got {raw_step!r}"
        )
    if step is not None and step < 0:
        raise AxBenchBpbAdvError("clustered encdec_target.step must be non-negative")
    return ClusterFeatureContractRequest(
        feature_dir=cfg.feature_dir,
        feature_statistics_path=cfg.feature_statistics_path,
        feature_mask=cfg.feature_mask,
        feature_clusters=cfg.feature_clusters,
        checkpoint_source=source,
        checkpoint_model_tag=model_tag,
        checkpoint_step=step,
        checkpoint_tag=model_tag,
        gcg_top_k_pos=cfg.ours_gcg.top_k_pos,
        gcg_top_k_neg=cfg.ours_gcg.top_k_neg,
        force_cluster_shard_revalidation=cfg.force_cluster_shard_revalidation,
        selection_splice_batch_size=cfg.selection_splice_batch_size,
        selection_manifest_path=cfg.selection_manifest_path,
        enabled_methods=(None if cfg.base_methods is None else tuple(cfg.base_methods)),
        expected_feature_contract_path=(
            Path(cfg.source_run_dir) / "feature_contract.json"
        ),
    )


__all__ = ["build_cluster_contract_request"]
