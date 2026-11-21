"""Real leaf prompt-policy and validation-family agreement gates."""

from __future__ import annotations

from statistics import fmean

import pytest
import torch as t

from scripts.chat_tui.convo import default_val_parquet_dir
from scripts.chat_tui.setup import load_chat_setup, resolve_conditioning_seed
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.encoder_decoder.sft_features import load_sft_features
from steering.conditional.feature_pipeline import append_negative_features
from steering.conditional.percentile_table import magnitude_to_percentile
from steering.conditional.sampling import (
    SamplingBundle,
    sample_features,
    sample_inactive_features_batched,
)
from steering.eval.context_cache_build import build_context_entry_from_extracted_segments
from steering.eval.context_enumeration import context_text_for_extraction
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import resolve_bos_token_id
from tests.test_chat_tui.gpu.conftest import LEAF_META

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")


def _source_bos_char_len(base_model) -> int:
    bos_id = resolve_bos_token_id(base_model.tokenizer)
    if bos_id is None:
        raise ValueError("real extraction tokenizer has no BOS token")
    length = len(base_model.tokenizer.decode([bos_id]))
    if length <= 0:
        raise ValueError("real extraction tokenizer BOS decodes to empty text")
    return length


def _synthetic_raw_segment(setup) -> tuple[t.Tensor, t.Tensor]:
    consolidation = setup.prompt_consolidation
    if consolidation is None:
        raise AssertionError("leaf prompt gate requires the recorded feature clusters")
    table = setup.prompt_pipeline.selection_config.percentile_table
    statistically_covered = (table[:, 0] != 0) | (table[:, -1] != 0)
    allowed = consolidation.allowed_universe & statistically_covered
    representatives = t.nonzero(
        allowed & ~consolidation.consolidated_away,
        as_tuple=False,
    ).flatten()
    siblings = t.nonzero(
        allowed & consolidation.consolidated_away,
        as_tuple=False,
    ).flatten()
    if representatives.numel() < 100 or siblings.numel() < 100:
        raise AssertionError("leaf cluster contract lacks 100 covered representatives and siblings")
    ids = t.cat([representatives[:100], siblings[:100]]).to(t.int64)
    quantile_indices = (t.arange(ids.numel(), dtype=t.int64) % 35) + 55
    magnitudes = table[ids, quantile_indices].to(t.float32)
    assert bool(t.isfinite(magnitudes).all())
    return ids, magnitudes


def _trainer_select_one_segment(
    setup,
    raw_ids: t.Tensor,
    raw_magnitudes: t.Tensor,
) -> dict[int, int]:
    """Mirror the exact public primitives in the live SFT per-segment loop."""

    pipeline = setup.prompt_pipeline
    selection = pipeline.selection_config
    consolidation = setup.prompt_consolidation
    negative = pipeline.negative_bundle
    assert isinstance(selection, SamplingBundle)
    assert consolidation is not None
    assert negative is not None

    full_percentiles = magnitude_to_percentile(
        raw_ids,
        raw_magnitudes,
        selection.percentile_table,
    )
    selected_ids = raw_ids
    selected_magnitudes = raw_magnitudes
    selection_weights = full_percentiles
    if pipeline.feature_mask is not None:
        selected_ids, selected_magnitudes, selection_weights = (
            pipeline.feature_mask.apply_to_positive(
                selected_ids,
                selected_magnitudes,
                selection_weights,
            )
        )
    normalized = (selection_weights / 100.0).to(t.float32)
    representatives, consolidated_percentiles = consolidation.consolidate_segment(
        selected_ids,
        normalized,
    )
    positive_ids, positive_percentiles = sample_features(
        representatives,
        consolidated_percentiles,
        selection.percentile_table,
        selection.selection,
        selection.rng,
        override_weights=consolidated_percentiles,
    )
    buckets = percentile_to_bucket(
        positive_percentiles,
        pipeline.boundaries.shape[1] + 1,
    )
    negative_ids = sample_inactive_features_batched(
        [representatives.to(t.int64)],
        negative.n_sae_features,
        negative.selection,
        negative.rng,
        feature_mask=consolidation.effective_negative_mask(),
    )[0]
    combined_ids, combined_buckets, _percentiles = append_negative_features(
        positive_ids,
        buckets,
        t.zeros_like(buckets, dtype=t.float32),
        negative_ids,
        effective_buckets=pipeline.boundaries.shape[1] + 1,
    )
    return {
        int(feature_id): int(bucket)
        for feature_id, bucket in zip(combined_ids, combined_buckets, strict=True)
    }


def _family_entry(setup, tokenizer, raw_segments, token_ids, source_bos_char_len):
    pipeline = setup.prompt_pipeline
    assert isinstance(pipeline.selection_config, SamplingBundle)
    pipeline.selection_config.rng.manual_seed(resolve_conditioning_seed(setup))
    return build_context_entry_from_extracted_segments(
        token_ids=token_ids,
        segment_feature_ids=tuple(segment.feature_ids for segment in raw_segments),
        segment_feature_magnitudes=tuple(segment.feature_mags for segment in raw_segments),
        segment_char_starts=tuple(segment.char_start for segment in raw_segments),
        segment_char_ends=tuple(segment.char_end for segment in raw_segments),
        pipeline=pipeline,
        n_sae_features=PositiveInt.parse(setup.n_sae_features),
        num_buckets=PositiveInt.parse(2),
        tokenizer=tokenizer,
        source_bos_char_len=NonNegativeInt.parse(source_bos_char_len),
        feature_clusters=setup.prompt_consolidation,
        source="GPU standard-SFT validation family",
    )


def test_leaf_prompt_pipeline_matches_trainer_primitives_on_shared_raw_segment(
    leaf_model_tokenizer,
    shared_extraction_weights,
    stats_path,
) -> None:
    _model, tokenizer = leaf_model_tokenizer
    base_model, _sae = shared_extraction_weights
    tui_setup = load_chat_setup(LEAF_META, stats_path)
    trainer_setup = load_chat_setup(LEAF_META, stats_path)
    raw_ids, raw_magnitudes = _synthetic_raw_segment(tui_setup)
    text = "Synthetic full-size leaf policy comparison segment."
    token_ids = tuple(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id()))
    bos_chars = _source_bos_char_len(base_model)
    extracted_text = context_text_for_extraction(tokenizer, token_ids)

    tui_entry = build_context_entry_from_extracted_segments(
        token_ids=token_ids,
        segment_feature_ids=(raw_ids,),
        segment_feature_magnitudes=(raw_magnitudes,),
        segment_char_starts=(0,),
        segment_char_ends=(bos_chars + len(extracted_text),),
        pipeline=tui_setup.prompt_pipeline,
        n_sae_features=PositiveInt.parse(tui_setup.n_sae_features),
        num_buckets=PositiveInt.parse(2),
        tokenizer=tokenizer,
        source_bos_char_len=NonNegativeInt.parse(bos_chars),
        feature_clusters=tui_setup.prompt_consolidation,
        source="GPU synthetic TUI prompt segment",
    )
    tui_segment = tui_entry.segments[0]
    tui_features = dict(zip(tui_segment.feature_ids, tui_segment.buckets, strict=True))
    trainer_features = _trainer_select_one_segment(
        trainer_setup,
        raw_ids,
        raw_magnitudes,
    )

    assert tui_features == trainer_features


def test_live_extraction_agrees_with_standard_sft_validation_family(
    leaf_model_tokenizer,
    shared_extraction_weights,
    extraction_service_factory,
    stats_path,
    val_feature_family_path,
) -> None:
    if not val_feature_family_path.is_dir():
        pytest.skip(f"required artifact path is absent: {val_feature_family_path}")
    parquet_dir = default_val_parquet_dir()
    if not parquet_dir.is_dir():
        pytest.skip(f"required artifact path is absent: {parquet_dir}")

    _model, tokenizer = leaf_model_tokenizer
    base_model, _sae = shared_extraction_weights
    bos_chars = _source_bos_char_len(base_model)
    store = load_sft_features(str(val_feature_family_path), str(parquet_dir))
    original_indices = sorted(store)[:3]
    if len(original_indices) < 3:
        pytest.fail("standard-SFT validation feature family contains fewer than 3 rows")
    texts = store.load_texts(original_indices)
    conversation_agreements: list[float] = []

    for original_index in original_indices:
        setup = load_chat_setup(LEAF_META, stats_path)
        service = extraction_service_factory(setup, tokenizer)
        text = texts[original_index]
        token_ids = tuple(tokenizer.encode(text, prepend=tokenizer.get_bos_token_id()))
        live = service.prepare(token_ids)
        family = _family_entry(
            setup,
            tokenizer,
            store[original_index],
            token_ids,
            bos_chars,
        )
        n_segments = max(len(live.segments), len(family.segments))
        per_segment: list[float] = []
        for segment_index in range(n_segments):
            live_ids = (
                set(live.segments[segment_index].feature_ids)
                if segment_index < len(live.segments)
                else set()
            )
            family_ids = (
                set(family.segments[segment_index].feature_ids)
                if segment_index < len(family.segments)
                else set()
            )
            union = live_ids | family_ids
            per_segment.append(1.0 if not union else len(live_ids & family_ids) / len(union))
        agreement = fmean(per_segment)
        conversation_agreements.append(agreement)
        print(
            "leaf-prompt agreement "
            f"original_index={original_index} segments_live={len(live.segments)} "
            f"segments_family={len(family.segments)} mean={agreement:.4f} "
            f"min={min(per_segment):.4f}"
        )
        # Bound every segment, not just the mean: a single badly-diverged
        # segment must fail the gate even when the rest agree. The per-segment
        # floor is looser than the mean bound because float16 family storage
        # makes a couple of near-tie selection swaps legitimate (two swaps in
        # a 49-feature segment is already a 0.92 Jaccard).
        worst_index = min(range(len(per_segment)), key=per_segment.__getitem__)
        assert min(per_segment) >= 0.85, (
            f"segment {worst_index} of conversation {original_index} agrees only "
            f"{per_segment[worst_index]:.4f} with the published family"
        )

    overall = fmean(conversation_agreements)
    print(f"leaf-prompt agreement overall_mean={overall:.4f}")
    assert overall >= 0.95
