"""Plain AxBench-BPB feature-DiffMean conditioning-bag precompute."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

import torch as t
from loguru import logger

from steering.eval.axbench_base.extraction import build_percentile_table_from_stats
from steering.eval.axbench_bpb.sample_loader import load_raw_synth_rows
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, SteeringMode
from steering.eval.axbench_bpb.vocab_restriction import (
    checkpoint_vocab_fingerprint,
    feature_vocab_allowed_ids,
)
from steering.eval.axbench_bpb_adv.feature_diffmean import (
    PrefixedCompletion,
    feature_diffmean_bag,
)
from steering.eval.axbench_bpb_adv.gcg_bags import (
    assert_sae_matches_encdec,
)
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)
from steering.eval.axbench_bpb_adv.paths import format_reg_lambda
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    write_ours_gcg_bag,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    FAIR_FEATURE_DIFFMEAN_METHOD_MODES,
    PrecomputedBagFamily,
)
from steering.sae_resolver import parse_neuronpedia_source


def feature_diffmean_modes_for_methods(methods: Sequence[Any]) -> set[SteeringMode]:
    """Return the feature-DiffMean steering modes present in *methods*."""

    active_names = {str(method.name) for method in methods}
    return {mode for name, mode in FAIR_FEATURE_DIFFMEAN_METHOD_MODES.items() if name in active_names}


def _rows_by_concept_and_class(
    rows: Sequence[dict[str, Any]],
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    grouped: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        concept_id = int(row["concept_id"])
        content_class = str(row["content_class"])
        grouped.setdefault(concept_id, {}).setdefault(content_class, []).append(row)
    return grouped


def _prefixed_completions_from_rows(
    rows: Sequence[dict[str, Any]],
) -> list[PrefixedCompletion]:
    return [
        PrefixedCompletion(
            prefix_text=str(row["prefix_text"]),
            completion_text=str(row["output_text"]),
        )
        for row in rows
    ]


def _load_synth_rows_for_mode(
    source_paths: SourcePaths,
    mode: SteeringMode,
    *,
    filter_unaccepted: bool,
    concept_ids: Collection[int] | None = None,
) -> list[dict[str, Any]]:
    path = (
        source_paths.synth_training_positive()
        if mode is SteeringMode.POSITIVE
        else source_paths.synth_training_negative()
    )
    return load_raw_synth_rows(
        path,
        filter_unaccepted=filter_unaccepted,
        require_positive_qc_contract=(mode == SteeringMode.POSITIVE),
        expected_continuation_role=(
            "stage1b_positive"
            if mode == SteeringMode.POSITIVE
            else "stage1c_negative"
        ),
        concept_ids=concept_ids,
    )


def validate_feature_diffmean_concept_modes(
    *,
    cfg: Any,
    source_paths: SourcePaths,
    methods: Sequence[Any],
    rows_by_mode: dict[SteeringMode, list[dict[str, Any]]] | None = None,
) -> set[tuple[int, SteeringMode]]:
    """Validate and return concept/mode pairs for plain BPB Feature-DiffMean.

    A feature-DiffMean bag is a contrast between concept-rich and
    concept-avoiding completions. If QC filtering leaves a present concept/mode
    with only one side, the conditioning signal is undefined. Raise here rather
    than silently skipping that group and surfacing a later missing-bag error.
    """

    modes = feature_diffmean_modes_for_methods(methods)
    complete: set[tuple[int, SteeringMode]] = set()
    incomplete: list[str] = []
    for mode in sorted(modes, key=lambda item: item.value):
        rows = (
            rows_by_mode[mode]
            if rows_by_mode is not None
            else _load_synth_rows_for_mode(
                source_paths, mode, filter_unaccepted=cfg.filter_unaccepted_samples
            )
        )
        if not rows:
            raise AxBenchBpbError(
                "plain BPB feature-DiffMean derive has no accepted source rows "
                f"for active mode={mode.value!r}; cannot build conditioning bags."
            )
        for concept_id, by_class in _rows_by_concept_and_class(rows).items():
            n_rich = len(by_class.get("concept_rich", []))
            n_avoid = len(by_class.get("concept_avoiding", []))
            if n_rich and n_avoid:
                complete.add((int(concept_id), mode))
            else:
                incomplete.append(
                    f"concept_id={int(concept_id)} mode={mode.value} "
                    f"concept_rich={n_rich} concept_avoiding={n_avoid}"
                )
    if incomplete:
        details = "; ".join(incomplete[:10])
        if len(incomplete) > 10:
            details += f"; ... ({len(incomplete) - 10} more)"
        raise AxBenchBpbError(
            "plain BPB feature-DiffMean derive requires both content classes for "
            f"every active concept/mode; incomplete groups: {details}"
        )
    if not complete:
        raise AxBenchBpbError(
            "plain BPB feature-DiffMean derive found no complete concept/mode "
            "pairs for the active feature-DiffMean methods."
        )
    return complete


def build_feature_diffmean_bags_plain(
    *,
    cfg: Any,
    output_paths: Any,
    source_paths: SourcePaths,
    methods: Sequence[Any],
    model: Any,
    base_model: Any,
    sae: Any,
    device: Any,
    concept_ids: Collection[int] | None = None,
) -> None:
    """Write feature-DiffMean conditioning bags for active plain-BPB methods.

    Plain BPB has no adversarial held-out prefix axis, so this mirrors existing
    residual baseline semantics: each concept/mode bag is derived from all
    accepted source rows for that concept/mode and then scored on the same BPB
    corpus.
    """

    modes = feature_diffmean_modes_for_methods(methods)
    if not modes:
        return
    # Load each active mode's synth parquet ONCE and reuse it for both validation
    # and bag building -- load_raw_synth_rows is uncached, so re-reading would
    # double this stage's parquet I/O and peak memory on large corpora. Iterate in
    # sorted-by-value order (matching validate_feature_diffmean_concept_modes) so
    # the bag build/log ordering below is deterministic rather than inheriting the
    # hash-randomized iteration order of the ``modes`` set.
    rows_by_mode: dict[SteeringMode, list[dict[str, Any]]] = {
        mode: _load_synth_rows_for_mode(
            source_paths,
            mode,
            filter_unaccepted=cfg.filter_unaccepted_samples,
            concept_ids=concept_ids,
        )
        for mode in sorted(modes, key=lambda item: item.value)
    }
    complete_concept_modes = validate_feature_diffmean_concept_modes(
        cfg=cfg,
        source_paths=source_paths,
        methods=methods,
        rows_by_mode=rows_by_mode,
    )

    try:
        assert_sae_matches_encdec(cfg, model)
    except Exception as exc:
        raise AxBenchBpbError(f"feature-DiffMean SAE identity guard failed: {exc}") from exc

    torch_device = t.device(device) if not isinstance(device, t.device) else device
    positive_bucket, negative_bucket = assert_single_positive_bucket(model)
    k_pos, k_neg = resolve_k(model, cfg.feature_diffmean)
    _release, _act, sae_layer, _width, _l0 = parse_neuronpedia_source(cfg.neuronpedia_source)
    n_sae = int(model.encoder.config.n_sae_features)
    percentile_table = build_percentile_table_from_stats(
        cfg.feature_statistics_path,
        n_sae,
    )
    # A compacted target checkpoint's allowed SAE feature vocabulary V (None on
    # an uncompacted checkpoint) -- restricts feature_diffmean_bag's selection
    # so it can never pick an id that trips the encoder's OOV assert.
    allowed_ids = feature_vocab_allowed_ids(model)

    logger.info(
        "plain BPB feature-DiffMean derive: modes={} k_pos={} k_neg={} buckets=({},{})",
        sorted(mode.value for mode in modes),
        k_pos,
        k_neg,
        positive_bucket,
        negative_bucket,
    )

    for mode, rows in rows_by_mode.items():
        grouped = _rows_by_concept_and_class(rows)
        for concept_id, by_class in sorted(grouped.items()):
            rich_rows = by_class.get("concept_rich", [])
            avoid_rows = by_class.get("concept_avoiding", [])
            if (int(concept_id), mode) not in complete_concept_modes:
                raise AxBenchBpbError(
                    "plain BPB feature-DiffMean derive internal mismatch: "
                    f"concept_id={int(concept_id)} mode={mode.value!r} was "
                    "loaded for building but not validated as complete."
                )
            bag = feature_diffmean_bag(
                _prefixed_completions_from_rows(rich_rows),
                _prefixed_completions_from_rows(avoid_rows),
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
                allowed_ids=allowed_ids,
            )
            path = output_paths.ours_gcg_bag(
                concept_id,
                mode,
                0.0,
                PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
            )
            write_ours_gcg_bag(
                path,
                bag,
                metadata={
                    "concept_id": str(int(concept_id)),
                    "mode": mode.value,
                    "mode_tag": PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN.value,
                    "coefficient_str": "None",
                    "reg_lambda": format_reg_lambda(0.0),
                    "n_features": str(len(bag)),
                    "n_rich_rows": str(len(rich_rows)),
                    "n_avoid_rows": str(len(avoid_rows)),
                    "k_pos": str(int(k_pos)),
                    "k_neg": str(int(k_neg)),
                    "positive_bucket": str(int(positive_bucket)),
                    "negative_bucket": str(int(negative_bucket)),
                },
                vocab_fingerprint=checkpoint_vocab_fingerprint(model),
            )
            logger.info(
                "plain BPB feature-DiffMean bag written: concept_id={} mode={} "
                "n_features={} rich={} avoid={} -> {}",
                concept_id,
                mode.value,
                len(bag),
                len(rich_rows),
                len(avoid_rows),
                path,
            )


__all__ = [
    "build_feature_diffmean_bags_plain",
    "feature_diffmean_modes_for_methods",
    "validate_feature_diffmean_concept_modes",
]
