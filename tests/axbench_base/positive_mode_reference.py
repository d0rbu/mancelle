"""Scalar Stage 1b attempt reference retained only for parity tests."""

from __future__ import annotations

from typing import Any

from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.synth import positive_mode
from steering.eval.axbench_base.synth.batching import SynthResult
from steering.eval.axbench_base.synth.continuation_protocol import (
    Stage1ContinuationRequest,
)
from steering.eval.axbench_base.types import Concept
from steering.eval.cluster_feature_contract import ClusterFeatureTarget


def handle_positive_attempt_result(
    result: SynthResult,
    item: Any,
    request: Stage1ContinuationRequest,
    cfg: AxBenchBaseConfig,
    concept: Concept,
    base_model: Any,
    base_sae: Any,
    percentile_table: Any,
    tokenizer: Any,
    gen_source: str,
    *,
    sae_layer: int,
    attempt: int,
    cluster_target: ClusterFeatureTarget | None = None,
) -> None:
    """Run the retired scalar parse, score, and application sequence."""
    if item.done:
        return
    continuation = positive_mode._parse_positive_rich_continuation(
        result,
        item,
        request,
        cfg,
        concept,
        tokenizer,
        attempt=attempt,
    )
    if continuation is None:
        return
    with positive_mode.profile_region(
        "stage1b.compute_qc_percentiles",
        concept_id=concept.concept_id,
        feature_id=concept.feature_index,
        attempt=attempt,
    ):
        out_pct = positive_mode.output_fc_percentile_for_continuation(
            item.selected.prefix_text,
            continuation,
            concept.feature_index,
            base_model,
            base_sae,
            percentile_table,
            sae_layer=sae_layer,
            tokenizer_id=cfg.extraction_tokenizer_id,
            **positive_mode._cluster_target_kwargs(cluster_target),
        )
    positive_mode._apply_positive_rich_result(
        continuation,
        out_pct,
        item,
        cfg,
        concept,
        gen_source,
        attempt=attempt,
    )
