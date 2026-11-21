"""Shuffled conditioning evaluation pipeline entry point.

Evaluates whether SAE feature conditioning is causally meaningful by comparing
completions generated under two conditioned variants:

  1. Real conditioning — features from the actual target segment
  2. Shuffled — features from a randomly selected donor document

If conditioning is causal, the real condition should produce higher
top-k percentile activations than shuffled.

Three sequential stages:

  Stage 1 (generate) — generate completions under both conditions
  Stage 2 (probe)    — measure top-k percentile activation in base model + SAE
  Stage 3 (aggregate)— compute summary statistics

Usage:
    uv run python scripts/shuffled_conditioning_eval.py source=encdec feature_dir=path/to/features
    uv run python scripts/shuffled_conditioning_eval.py source=encdec stage=probe feature_dir=path/to/features
"""

from __future__ import annotations

import dataclasses
import random as _random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch as t
import yaml
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from nanochat.checkpoint_manager import ModelPhase, load_model
from nanochat.common import autodetect_device_type
from nanochat.dataset import list_parquet_files
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    build_thresholds_from_stats,
    extract_threshold_percentile,
)
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.neuronpedia_client import NeuronpediaClient
from steering.eval.segment_mode_guard import (
    require_fixed_checkpoint_segment_mode,
    require_fixed_feature_dir_segment_mode,
)
from steering.eval.shuffled_conditioning_eval import (
    FeatureSelectionStrategy,
    ShuffledConditioningResult,
    aggregate_results,
    assert_no_feature_clustering,
    compute_topk_percentile_activation,
    deserialize_results,
    generate_for_shuffled_document,
    plot_shuffled_comparison,
    sample_shuffled_segment,
    select_eval_documents,
    select_random_k_features,
    select_top_k_features,
    serialize_results,
)
from steering.extract_features import load_sae_and_model
from steering.feature_io import glob_features, shard_stem_from_path
from steering.feature_paths import parse_feature_dir
from steering.sae_resolver import SAERelease, resolve_sae
from steering.stats_io import find_statistics_path, load_statistics
from steering.tokenizer_protocols import require_bos_tokenizer

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "shuffled_conditioning_eval"


def main() -> None:
    """Run the shuffled conditioning evaluation pipeline."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        cfg = compose(config_name=_CONFIG_NAME, overrides=overrides)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Validate required config ---
    if not cfg.feature_dir:
        logger.error("feature_dir is required. Pass feature_dir=path/to/features")
        sys.exit(1)

    feature_dir = Path(cfg.feature_dir)
    if not feature_dir.is_dir():
        logger.error(f"feature_dir does not exist: {feature_dir}")
        sys.exit(1)

    stage = str(cfg.stage).lower()
    force_recompute = bool(cfg.force_recompute)

    # --- Resolve output directory ---
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = feature_dir / "shuffled_conditioning_eval"
    logger.info(f"Output directory: {output_dir}")

    if force_recompute:
        logger.warning("force_recompute=true — all stages will run regardless of existing outputs")

    # --- Load feature statistics ---
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        logger.error(
            f"feature_statistics.safetensors not found in {feature_dir}. "
            "Run feature extraction with statistics computation first."
        )
        sys.exit(1)

    logger.info(f"Loading feature statistics from {stats_path}...")
    feature_stats = load_statistics(stats_path)
    require_fixed_feature_dir_segment_mode(
        feature_dir,
        surface="shuffled_conditioning_eval",
        statistics_segment_mode=feature_stats.metadata.segment_mode,
    )
    per_feature_stats = feature_stats.per_feature_stats
    n_sae_features = feature_stats.metadata.n_sae_features
    logger.info(f"Loaded statistics: n_sae_features={n_sae_features}")

    # --- Build thresholding and discretization configs ---
    threshold_type = ThresholdType.feature_percentile_segment_topk
    threshold_val = float(cfg.threshold)
    threshold_cfg = ThresholdConfig(threshold_type, threshold_val)
    threshold_pct = extract_threshold_percentile(threshold_cfg)

    # --- Resolve validation feature shard ---
    all_feature_shards = glob_features(str(feature_dir))
    if not all_feature_shards:
        logger.error(f"No feature shards found in {feature_dir}")
        sys.exit(1)
    val_shard_path = all_feature_shards[-1]
    val_shard_stem = shard_stem_from_path(val_shard_path)
    logger.info(f"Validation feature shard: {val_shard_path.name} (stem={val_shard_stem})")

    # --- Stage file paths ---
    completions_path = output_dir / "completions.yaml"
    probe_results_path = output_dir / "probe_results.yaml"
    aggregated_path = output_dir / "aggregated.yaml"

    # --- Stage output tracking ---
    results: list[ShuffledConditioningResult] | None = None

    # -----------------------------------------------------------------------
    # Stage 1 — generate
    # -----------------------------------------------------------------------
    if stage in ("all", "generate"):
        should_skip = stage == "all" and completions_path.exists() and not force_recompute
        if should_skip:
            logger.info("=== Stage 1: Generate (skipped — output exists) ===")
            results = deserialize_results(completions_path)
            logger.info(f"Loaded {len(results)} results from {completions_path}")
        else:
            logger.info("=== Stage 1: Generate ===")

            # Load conditional model
            device_type = autodetect_device_type()
            device = t.device(device_type)
            logger.info(f"Loading model: source={cfg.source}, tag={cfg.model_tag}, step={cfg.step}")
            model, tokenizer, meta = load_model(
                source=cfg.source,
                device=device,
                phase=ModelPhase.EVAL,
                model_tag=cfg.model_tag,
                step=cfg.step,
                eval_conditioning=EvalConditioning.deferred("shuffled_conditioning_eval"),
            )
            # Fail-fast: removed clustered checkpoints would silently produce
            # wrong reactivation results.
            assert_no_feature_clustering(meta)
            if not isinstance(model, EncoderDecoderGPT):
                raise TypeError(
                    "shuffled_conditioning_eval requires an EncoderDecoderGPT "
                    f"checkpoint; got {type(model).__name__}"
                )
            require_fixed_checkpoint_segment_mode(
                meta,
                surface="shuffled_conditioning_eval",
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            eval_tokenizer = require_bos_tokenizer(tokenizer, surface="shuffled_conditioning_eval")
            nextseg_eval = resolve_next_segment_conditioning_offset(cfg, meta)
            logger.info(f"next_segment_conditioning_offset={nextseg_eval}")
            if device.type == "cuda":
                model = model.to(t.bfloat16)

            enc_config = model.encoder.config
            num_buckets = enc_config.num_buckets

            boundaries = build_boundaries_from_stats(
                per_feature_stats,
                n_sae_features,
                num_buckets,
                threshold_percentile=threshold_pct,
            )
            logger.info(f"Built bucket boundaries ({num_buckets} buckets)")

            precomputed_thresholds = build_thresholds_from_stats(
                per_feature_stats, n_sae_features, threshold_type, threshold_val
            )
            logger.info(f"Built {threshold_type.value} thresholds from cached statistics")

            # Build FeatureIndex
            logger.info(f"Scanning feature shards in {feature_dir}...")
            feature_index = FeatureIndex(str(feature_dir))
            logger.info(f"FeatureIndex: {feature_index.n_shards} shards discovered")

            # Load validation parquet text
            all_parquets = list_parquet_files()
            if not all_parquets:
                logger.error("No parquet files found in dataset directory")
                sys.exit(1)
            val_parquet = all_parquets[-1]
            logger.info(f"Loading validation text from {val_parquet}")
            table = pq.read_table(val_parquet, columns=["text"])
            doc_texts: list[str] = table.column("text").to_pylist()
            logger.info(f"Loaded {len(doc_texts)} documents from validation parquet")

            # Load all document segments from the validation shard
            val_docs = feature_index._load_shard(val_shard_stem)
            all_doc_indices = list(val_docs.keys())
            logger.info(f"Loaded {len(all_doc_indices)} documents from val shard")

            # Select evaluation documents (context-window filter applied)
            n_documents = int(cfg.n_documents)
            bos_id = eval_tokenizer.get_bos_token_id()
            max_seq_len = model.dec_config.sequence_len
            selected = select_eval_documents(
                val_docs,
                n_documents,
                doc_texts=doc_texts,
                tokenizer=eval_tokenizer,
                max_seq_len=max_seq_len,
                bos_id=bos_id,
            )
            logger.info(f"Selected {len(selected)} documents for evaluation")

            # Set up Neuronpedia client for feature descriptions
            np_cache_path = str(output_dir / "neuronpedia_cache.yaml")
            np_client = NeuronpediaClient(
                model_id="gemma-2-2b",
                sae_source="8-gemmascope-res-16k",
                cache_path=np_cache_path,
            )

            samples_per_condition = int(cfg.samples_per_condition)
            temperature = float(cfg.temperature)
            top_k_sampling = int(cfg.top_k_sampling) if cfg.top_k_sampling else None
            top_k_features = int(cfg.top_k_features)
            rng = _random.Random(42)

            results = []
            for doc_i, (doc_idx, target_seg_idx) in enumerate(
                tqdm(selected, desc="Generating", unit="doc")
            ):
                if doc_idx >= len(doc_texts):
                    logger.warning(
                        f"doc_idx {doc_idx} out of range "
                        f"(parquet has {len(doc_texts)} docs) — skipping"
                    )
                    continue

                doc_text = doc_texts[doc_idx]
                raw_segments = val_docs[doc_idx]

                # Apply threshold but DON'T discretize — we need raw mags
                # for percentile ranking
                raw_target = raw_segments[target_seg_idx]
                fids, mags = raw_target.feature_ids, raw_target.feature_mags
                if threshold_cfg is not None:
                    fids, mags = apply_threshold(threshold_cfg, fids, mags, precomputed_thresholds)
                if fids.numel() == 0:
                    logger.warning(
                        f"Doc {doc_idx}: target segment has no active features "
                        f"after thresholding — skipping"
                    )
                    continue
                thresholded_target = SegmentFeatures(
                    segment_idx=raw_target.segment_idx,
                    position=raw_target.position,
                    feature_ids=fids,
                    feature_mags=mags,
                    char_start=raw_target.char_start,
                    char_end=raw_target.char_end,
                )

                # Select features from the target segment
                strategy = FeatureSelectionStrategy(str(cfg.feature_selection))
                if strategy is FeatureSelectionStrategy.random_k:
                    top_k_fids = select_random_k_features(
                        thresholded_target,
                        per_feature_stats,
                        k=top_k_features,
                        seed=42 + doc_i,
                    )
                elif strategy is FeatureSelectionStrategy.top_k:
                    top_k_fids = select_top_k_features(
                        thresholded_target, per_feature_stats, k=top_k_features
                    )

                # Fetch Neuronpedia descriptions for these features
                descriptions = np_client.get_descriptions(top_k_fids)
                top_k_descs = [descriptions.get(fid, f"Feature {fid}") for fid in top_k_fids]

                # Sample a shuffled segment from a different document
                try:
                    shuffled_seg, donor_doc_idx = sample_shuffled_segment(
                        target_segment_idx=target_seg_idx,
                        current_doc_idx=doc_idx,
                        feature_index=feature_index,
                        val_shard_stem=val_shard_stem,
                        all_doc_indices=all_doc_indices,
                        boundaries=boundaries,
                        threshold_cfg=threshold_cfg,
                        precomputed_thresholds=precomputed_thresholds,
                        rng=rng,
                    )
                except ValueError as e:
                    logger.warning(f"Doc {doc_idx}: {e} — skipping")
                    continue

                # Generate completions under all 3 conditions
                pre_segment_text, ground_truth_text, samples = generate_for_shuffled_document(
                    model=model,
                    tokenizer=eval_tokenizer,
                    doc_text=doc_text,
                    segments=raw_segments,
                    target_segment_idx=target_seg_idx,
                    shuffled_segment=shuffled_seg,
                    boundaries=boundaries,
                    threshold_cfg=threshold_cfg,
                    precomputed_thresholds=precomputed_thresholds,
                    device=device,
                    samples_per_condition=samples_per_condition,
                    temperature=temperature,
                    top_k=top_k_sampling,
                    base_seed=42 + (len(results) + 1) * 1000,
                    bos_id=bos_id,
                    next_segment_conditioning_offset=nextseg_eval,
                )

                # Build result with 0.0 percentiles (filled in by probe stage)
                real_samples = [(s.text, 0.0) for s in samples if s.condition == "real"]
                shuffled_samples = [(s.text, 0.0) for s in samples if s.condition == "shuffled"]

                result = ShuffledConditioningResult(
                    doc_idx=doc_idx,
                    target_segment_idx=target_seg_idx,
                    n_top_k_features=len(top_k_fids),
                    top_k_feature_ids=top_k_fids,
                    ground_truth_text=ground_truth_text,
                    pre_segment_text=pre_segment_text,
                    shuffled_doc_idx=donor_doc_idx,
                    top_k_feature_descriptions=top_k_descs,
                    real_samples=real_samples,
                    shuffled_samples=shuffled_samples,
                )
                results.append(result)
                logger.info(
                    f"Doc {doc_i + 1}/{len(selected)} (idx={doc_idx}): "
                    f"{len(samples)} samples generated"
                )

            serialize_results(results, completions_path)
            logger.info(f"Stage 1 complete: {len(results)} results saved to {completions_path}")

            # Free conditional model
            logger.info("Freeing conditional model from GPU...")
            del model
            if device.type == "cuda":
                t.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Stage 2 — probe
    # -----------------------------------------------------------------------
    if stage in ("all", "probe"):
        should_skip = stage == "all" and probe_results_path.exists() and not force_recompute
        if should_skip:
            logger.info("=== Stage 2: Probe (skipped — output exists) ===")
            results = deserialize_results(probe_results_path)
            logger.info(f"Loaded {len(results)} probed results from {probe_results_path}")
        else:
            logger.info("=== Stage 2: Probe ===")

            # Load results from generate stage if not in memory
            if results is None:
                if not completions_path.exists():
                    logger.error(
                        f"Completions not found at {completions_path}. Run stage=generate first."
                    )
                    sys.exit(1)
                results = deserialize_results(completions_path)
                logger.info(f"Loaded {len(results)} results from {completions_path}")

            # Free conditional model if still loaded (when running stage=probe alone)
            # (In stage=all flow, it was already freed above)

            # Load base Gemma model + SAE
            device_type = autodetect_device_type()
            device = t.device(device_type)

            dir_info = parse_feature_dir(str(feature_dir))
            layer = dir_info.layer
            activation_type = dir_info.activation_type
            width = dir_info.width
            model_name = dir_info.model

            resolved = resolve_sae(
                model=model_name,
                release=SAERelease.gemmascope,
                activation_type=activation_type,
                layer=layer,
                width=width,
            )
            logger.info(
                f"Resolved SAE: release={resolved.release}, "
                f"id={resolved.sae_id}, hook={resolved.hook_name}"
            )

            base_model, sae, gemma_tokenizer = load_sae_and_model(
                model_name=model_name,
                layer=layer,
                device=str(device),
                resolved=resolved,
            )
            logger.info(f"Loaded base model ({model_name}) and SAE on {device}")

            hook_name = resolved.hook_name

            # Probe all samples
            total_samples = sum(len(r.real_samples) + len(r.shuffled_samples) for r in results)
            pbar = tqdm(total=total_samples, desc="Probing", unit="sample")

            for result in results:
                pre_text = result.pre_segment_text

                def _probe_samples(
                    samples: list[tuple[str, float]],
                    _pre_text: str = pre_text,
                    _top_k_fids: list[int] = result.top_k_feature_ids,
                ) -> list[tuple[str, float]]:
                    updated: list[tuple[str, float]] = []
                    for text, _ in samples:
                        pct = compute_topk_percentile_activation(
                            completion_text=text,
                            pre_segment_text=_pre_text,
                            top_k_feature_ids=_top_k_fids,
                            per_feature_stats=per_feature_stats,
                            base_model=base_model,
                            sae=sae,
                            gemma_tokenizer=gemma_tokenizer,
                            layer=layer,
                            hook_name=hook_name,
                            device=device,
                        )
                        updated.append((text, pct))
                        pbar.update(1)
                    return updated

                result.real_samples = _probe_samples(result.real_samples)
                result.shuffled_samples = _probe_samples(result.shuffled_samples)

            pbar.close()

            serialize_results(results, probe_results_path)
            logger.info(f"Stage 2 complete: probed results saved to {probe_results_path}")

            # Free base model + SAE
            del base_model, sae
            if device.type == "cuda":
                t.cuda.empty_cache()
            logger.info("Freed base model + SAE from GPU")

    # -----------------------------------------------------------------------
    # Stage 3 — aggregate
    # -----------------------------------------------------------------------
    if stage in ("all", "aggregate"):
        logger.info("=== Stage 3: Aggregate ===")

        # Load probed results if not in memory
        if results is None:
            if not probe_results_path.exists():
                logger.error(
                    f"Probe results not found at {probe_results_path}. Run stage=probe first."
                )
                sys.exit(1)
            results = deserialize_results(probe_results_path)
            logger.info(f"Loaded {len(results)} probed results from {probe_results_path}")

        k = int(cfg.top_k_features)
        aggregated = aggregate_results(results, k)

        logger.info(f"Aggregated results (n={aggregated.n_documents}, k={aggregated.k}):")
        logger.info(f"  Real:          {aggregated.real_mean:.4f} +/- {aggregated.real_stderr:.4f}")
        logger.info(
            f"  Shuffled:      {aggregated.shuffled_mean:.4f} +/- {aggregated.shuffled_stderr:.4f}"
        )

        # Save aggregated results
        agg_dict = dataclasses.asdict(aggregated)
        with open(aggregated_path, "w") as f:
            yaml.dump(agg_dict, f, default_flow_style=False)

        # Generate comparison plot
        plots_dir = output_dir / "plots"
        plot_shuffled_comparison(aggregated, output_path=plots_dir / "shuffled_comparison.png")

        logger.info(f"Stage 3 complete: aggregated results saved to {aggregated_path}")


if __name__ == "__main__":
    main()
