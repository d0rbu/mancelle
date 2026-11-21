"""Completion steering evaluation pipeline entry point.

Evaluates whether SAE feature conditioning causally influences encoder-decoder
model completions on real validation data. Runs up to five sequential stages:

  Stage 1 (scan)      — scan validation shard for per-feature activations
  Stage 2 (threshold) — re-scan after applying generation-time thresholds
  Stage 3 (select)    — rank features, fetch Neuronpedia descriptions, pick docs
  Stage 4 (generate)  — generate paired feature-intervention completions
  Stage 5 (judge)     — LLM judge evaluates completion pairs

Usage:
    uv run python scripts/completion_steering_eval.py source=encdec mode=ablation
    uv run python scripts/completion_steering_eval.py source=encdec stage=scan feature_dir=out/data/features/s128/...
"""

from __future__ import annotations

import asyncio
import math
import sys
import time as _time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import pyarrow.parquet as pq
import torch as t
import yaml
from beartype import beartype
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from errors import ValidationError
from nanochat.checkpoint_manager import ModelPhase, load_model
from nanochat.common import autodetect_device_type
from nanochat.dataset import list_parquet_files
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    reject_feature_clusters_config,
    reject_feature_clusters_metadata,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_pipeline import (
    ThresholdSelectionConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.sampling import (
    SamplingBundle,
    assert_no_gradient_ranking,
)
from steering.conditional.thresholding import ThresholdConfig
from steering.eval._stats_helpers import (
    compute_mean_stderr as probe_mean_stderr,
)
from steering.eval._stats_helpers import (
    compute_mean_stderr_or_zero,
)
from steering.eval.completion_judge import (
    CompletionJudgeClient,
    CompletionJudgeItem,
    JudgementScore,
    compute_dose_response,
    plot_ablation_comparison,
    plot_dose_response,
    save_scores_yaml,
)
from steering.eval.completion_steering_eval import (
    ContextFilter,
    EvalMode,
    EvalStage,
    FeatureSelection,
    ScanResultTensors,
    assert_selections_within_mask,
    build_target_segment_maps,
    load_scan_results,
    save_scan_results,
    scan_sampled_features,
    scan_thresholded_features,
    scan_validation_features_vectorized,
    select_from_tensors,
)
from steering.eval.config_validation import validate_evaluator_flags
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.feature_probe import (
    compute_probe_for_completions,
    plot_probe_ablation,
    plot_probe_dose_response,
)
from steering.eval.multiseg_generator import (
    CompletionSample,
    DocumentCompletion,
    GroundTruthFeatureSignals,
    generate_for_document,
)
from steering.eval.neuronpedia_client import NeuronpediaClient
from steering.eval.segment_mode_guard import (
    require_fixed_checkpoint_segment_mode,
    require_fixed_feature_dir_segment_mode,
)
from steering.feature_io import glob_features, shard_stem_from_path
from steering.refined_types import (
    NonNegativeFiniteNumberLike,
    NonNegativeIntLike,
    PositiveIntLike,
)
from steering.stats_io import find_statistics_path, load_statistics
from steering.tokenizer_protocols import require_bos_tokenizer

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "completion_steering_eval"

if TYPE_CHECKING:
    _DocumentCompletionList = list[DocumentCompletion]
    _FeatureSelectionList = list[FeatureSelection]
    _StringList = list[str]
    _TargetSegmentMaps = dict[int, dict[int, int]]
else:
    # Avoid beartype's randomized element sampling for parameterized containers.
    # The domain-bearing elements have already crossed their own typed boundary.
    _DocumentCompletionList = list
    _FeatureSelectionList = list
    _StringList = list
    _TargetSegmentMaps = dict


# ---------------------------------------------------------------------------
# YAML deserialization helpers (avoid duplication across stages)
# ---------------------------------------------------------------------------


def _load_selections_yaml(path: Path) -> list[FeatureSelection]:
    """Load feature selections from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    return [
        FeatureSelection(
            feature_id=s["feature_id"],
            neuronpedia_description=s["neuronpedia_description"],
            n_docs_activated=s["n_docs_activated"],
            selected_docs=s["selected_docs"],
            target_segments={int(k): int(v) for k, v in s.get("target_segments", {}).items()},
        )
        for s in raw
    ]


def _load_completions_yaml(path: Path) -> list[DocumentCompletion]:
    """Load completions from a YAML file.

    Requires the ``ground_truth_signals`` field on every entry.
    Old-format files without this field will raise ``KeyError``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or []
    completions: list[DocumentCompletion] = []
    for c in raw:
        gt_raw = c["ground_truth_signals"]
        # Ensure feature_signals keys are ints (YAML may load them as ints already)
        raw_signals = gt_raw["feature_signals"]
        feature_signals = {int(k): int(v) for k, v in raw_signals.items()}
        foi_mag = gt_raw["feature_of_interest_magnitude"]
        gt_signals = GroundTruthFeatureSignals(
            feature_signals=feature_signals,
            n_features_active=gt_raw["n_features_active"],
            avg_magnitude=float(gt_raw["avg_magnitude"]),
            max_magnitude=int(gt_raw["max_magnitude"]),
            min_magnitude=int(gt_raw["min_magnitude"]),
            feature_of_interest_magnitude=int(foi_mag) if foi_mag is not None else None,
        )
        completions.append(
            DocumentCompletion(
                doc_idx=c["doc_idx"],
                feature_id=c["feature_id"],
                pre_segment_text=c["pre_segment_text"],
                ground_truth_segment=c["ground_truth_segment"],
                target_segment_idx=c["target_segment_idx"],
                samples=[
                    CompletionSample(
                        condition=s["condition"],
                        text=s["text"],
                        seed=s["seed"],
                        bucket_index=s.get("bucket_index"),
                    )
                    for s in c["samples"]
                ],
                ground_truth_signals=gt_signals,
            )
        )
    return completions


def _should_skip_stage(stage: EvalStage, output_path: Path, force_recompute: bool) -> bool:
    """Return True if this stage should be skipped.

    A stage is skipped only when running the full pipeline (stage=ALL),
    its output file already exists, and force_recompute is not set.
    """
    return stage is EvalStage.ALL and output_path.exists() and not force_recompute


@beartype
def _generate_completions(
    selections: _FeatureSelectionList,
    target_segment_maps: _TargetSegmentMaps | None,
    doc_texts: _StringList,
    feature_index,
    val_shard_stem: str,
    model,
    tokenizer,
    mode: EvalMode,
    boundaries: t.Tensor,
    feature_cfg,
    precomputed_thresholds,
    device: t.device,
    num_buckets: PositiveIntLike,
    samples_per_condition: PositiveIntLike,
    temperature: NonNegativeFiniteNumberLike,
    top_k_sampling: PositiveIntLike | None,
    bos_id: NonNegativeIntLike,
    next_segment_conditioning_offset: bool = False,
    feature_mask: FeatureMask | None = None,
) -> _DocumentCompletionList:
    """Iterate over feature selections and generate completions for each document.

    Extracted from Stage 4 of the generate pipeline to allow unit testing.
    Each (feature, doc) pair results in one DocumentCompletion appended to the
    output list.  The inner loop must not be broken out of prematurely — this
    function is the corrected version of the previously-indented bug.

    Args:
        target_segment_maps: ``{feature_id: {doc_idx: max_segment_idx}}``
            from :func:`build_target_segment_maps`.  Used in ablation mode
            to determine which segment to target.  ``None`` or empty is fine
            for insertion mode.
    """
    completions: list[DocumentCompletion] = []

    for feat_i, sel in enumerate(tqdm(selections, desc="Generating", unit="feature")):
        feat_start = _time.monotonic()
        logger.info(
            f"Feature {feat_i + 1}/{len(selections)}: "
            f"id={sel.feature_id}, {len(sel.selected_docs)} docs"
        )
        # Look up pre-built doc_idx -> max_segment_idx map for this feature
        doc_target_map: dict[int, int] = (
            target_segment_maps.get(sel.feature_id, {}) if target_segment_maps else {}
        )

        for doc_i, doc_idx in enumerate(sel.selected_docs):
            if doc_idx >= len(doc_texts):
                raise IndexError(
                    f"doc_idx {doc_idx} out of range (parquet has {len(doc_texts)} docs)"
                )

            doc_text = doc_texts[doc_idx]

            # Load segments for this document (raises KeyError if the doc is
            # absent from the index — corruption; issue #521).
            segments = feature_index.lookup(val_shard_stem, doc_idx)
            if len(segments) < 2:
                logger.warning(f"Doc {doc_idx} has only 1 segment, need >= 2 — skipping")
                continue

            # Determine target segment index
            # sel.target_segments is context-aware (from find_fitting_segment),
            # so it takes precedence over doc_target_map (scan max_segment_idx).
            if doc_idx in sel.target_segments:
                target_seg_idx = sel.target_segments[doc_idx]
            elif mode is EvalMode.ABLATION and doc_idx in doc_target_map:
                target_seg_idx = doc_target_map[doc_idx]
            else:
                # Fallback: use last segment
                target_seg_idx = len(segments) - 1

            # Clamp target to valid range (must be >= 1)
            target_seg_idx = max(1, min(target_seg_idx, len(segments) - 1))

            doc_start = _time.monotonic()
            doc_completion = generate_for_document(
                model=model,
                tokenizer=tokenizer,
                doc_text=doc_text,
                segments=segments,
                target_segment_idx=target_seg_idx,
                feature_id=sel.feature_id,
                mode=mode,
                boundaries=boundaries,
                feature_cfg=feature_cfg,
                precomputed_thresholds=precomputed_thresholds,
                device=device,
                num_buckets=num_buckets,
                doc_idx=doc_idx,
                samples_per_condition=samples_per_condition,
                temperature=temperature,
                top_k=top_k_sampling,
                base_seed=42 + (len(completions) + 1) * 1000,
                bos_id=bos_id,
                next_segment_conditioning_offset=next_segment_conditioning_offset,
                feature_mask=feature_mask,
            )
            if doc_completion is None:
                continue
            completions.append(doc_completion)
            doc_elapsed = _time.monotonic() - doc_start
            n_samples = len(doc_completion.samples)
            logger.info(
                f"  Doc {doc_i + 1}/{len(sel.selected_docs)} "
                f"(idx={doc_idx}): {n_samples} samples in {doc_elapsed:.1f}s"
            )

        feat_elapsed = _time.monotonic() - feat_start
        logger.info(f"Feature {sel.feature_id} done in {feat_elapsed:.1f}s")

    return completions


def _serialize_scores(
    all_scores: dict[int, dict[str, list[JudgementScore]]],
) -> dict:
    """Convert all_scores to a YAML-serializable dict."""
    result: dict = {}
    for fid, cond_scores in all_scores.items():
        result[int(fid)] = {
            cond: [
                {
                    "coherence": s.coherence,
                    "feature_alignment": s.feature_alignment,
                }
                for s in scores
            ]
            for cond, scores in cond_scores.items()
        }
    return result


async def _judge_all_features_async(
    judge_client: CompletionJudgeClient,
    feature_completions: dict[int, list[DocumentCompletion]],
    feature_descriptions: dict[int, str],
    all_scores: dict[int, dict[str, list[JudgementScore]]],
    mode: EvalMode,
    num_buckets: int,
    plots_dir: Path,
    scores_path: Path,
    judge_pbar: tqdm,
) -> None:
    """Run judge stage asynchronously across all features."""
    async with aiohttp.ClientSession() as session:
        for fid, fid_completions in feature_completions.items():
            description = feature_descriptions.get(fid, f"Feature {fid}")
            judge_pbar.set_postfix(feature=fid)

            items: list[CompletionJudgeItem] = []
            item_conditions: list[str] = []
            for comp in fid_completions:
                for sample in comp.samples:
                    items.append(
                        CompletionJudgeItem(
                            context=comp.pre_segment_text,
                            completion=sample.text,
                            feature_description=description,
                            ground_truth=comp.ground_truth_segment,
                        )
                    )
                    item_conditions.append(str(sample.condition))

            results = await judge_client.judge_batch_async(items, session=session)

            condition_scores: dict[str, list[JudgementScore]] = {}
            for score, condition in zip(results, item_conditions, strict=True):
                if score is not None:
                    condition_scores.setdefault(condition, []).append(score)
            judge_pbar.update(len(items))

            all_scores[fid] = condition_scores

            if mode is EvalMode.ABLATION:
                _plot_ablation_for_feature(fid, description, condition_scores, plots_dir)
            else:
                _plot_insertion_for_feature(
                    fid, description, condition_scores, num_buckets, plots_dir
                )

            save_scores_yaml(_serialize_scores(all_scores), scores_path)
            logger.info(f"Feature {fid}: judged {len(items)} samples, saved incremental scores")


def main() -> None:
    """Run the completion steering evaluation pipeline."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        cfg = compose(config_name=_CONFIG_NAME, overrides=overrides)
    reject_feature_clusters_config(cfg, "completion_steering_eval")

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Validate required config ---
    if not cfg.feature_dir:
        logger.error("feature_dir is required. Pass feature_dir=path/to/features")
        sys.exit(1)

    feature_dir = Path(cfg.feature_dir)
    if not feature_dir.is_dir():
        logger.error(f"feature_dir does not exist: {feature_dir}")
        sys.exit(1)

    stage = EvalStage(cfg.stage)
    mode = EvalMode(cfg.mode)

    # --- Validate evaluator flags ---
    run_judge = bool(cfg.run_judge)
    run_feature_analysis = bool(cfg.run_feature_analysis)
    validate_evaluator_flags(stage, run_judge=run_judge, run_feature_analysis=run_feature_analysis)

    # --- Resolve output directory ---
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = feature_dir / "completion_steering_eval"

    force_recompute = bool(cfg.force_recompute)
    if force_recompute:
        logger.warning("force_recompute=true — all stages will run regardless of existing outputs")

    # --- Load model ---
    device_type = autodetect_device_type()
    device = t.device(device_type)
    logger.info(f"Loading model: source={cfg.source}, tag={cfg.model_tag}, step={cfg.step}")
    model, tokenizer, meta = load_model(
        source=cfg.source,
        device=device,
        phase=ModelPhase.EVAL,
        model_tag=cfg.model_tag,
        step=cfg.step,
        eval_conditioning=EvalConditioning.deferred("completion_steering_eval"),
    )
    reject_feature_clusters_metadata(meta, "completion_steering_eval checkpoint")
    if not isinstance(model, EncoderDecoderGPT):
        raise TypeError(
            "completion_steering_eval requires an EncoderDecoderGPT checkpoint; "
            f"got {type(model).__name__}"
        )
    require_fixed_checkpoint_segment_mode(
        meta,
        surface="completion_steering_eval",
    )
    eval_tokenizer = require_bos_tokenizer(tokenizer, surface="completion_steering_eval")
    nextseg_eval = resolve_next_segment_conditioning_offset(cfg, meta)
    logger.info(f"next_segment_conditioning_offset={nextseg_eval}")
    if device.type == "cuda":
        model = model.to(t.bfloat16)

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
        surface="completion_steering_eval",
        statistics_segment_mode=feature_stats.metadata.segment_mode,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    per_feature_stats = feature_stats.per_feature_stats
    n_sae_features = feature_stats.metadata.n_sae_features
    logger.info(f"Loaded statistics: n_sae_features={n_sae_features}")

    # --- Build feature pipeline (delegates parse + build to shared helper) ---
    enc_config = model.encoder.config
    num_buckets = enc_config.num_buckets
    pipeline = build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        use_negative_selection=False,  # eval doesn't use negative sampling
    )
    # Early guard (the central sample_features guard would only fire after
    # the multi-stage shard scans below — minutes in).
    assert_no_gradient_ranking(
        getattr(getattr(pipeline, "selection_config", None), "selection", None),
        "completion steering eval (live selection)",
    )
    # Capture the resolved mask once; threaded through all selection-load sites below.
    feature_mask = pipeline.feature_mask

    # The mask used to VALIDATE materialised/loaded selections (defense-in-depth).
    selection_validation_mask = feature_mask

    boundaries = pipeline.boundaries
    if isinstance(pipeline.selection_config, ThresholdSelectionConfig):
        precomputed_thresholds = pipeline.selection_config.precomputed_thresholds
        feature_cfg: ThresholdConfig | SamplingBundle = ThresholdConfig(
            pipeline.selection_config.selection.threshold_type,
            pipeline.selection_config.selection.threshold,
        )
        logger.info(
            f"Built bucket boundaries ({num_buckets} buckets) and "
            f"{pipeline.selection_config.selection.threshold_type.value} thresholds "
            f"from cached statistics"
        )
    elif isinstance(pipeline.selection_config, SamplingBundle):
        precomputed_thresholds = None
        feature_cfg = pipeline.selection_config
        sampling_sel = pipeline.selection_config.selection
        logger.info(
            f"Built bucket boundaries ({num_buckets} buckets) "
            f"and sampling bundle: {sampling_sel.distribution.value}"
            f"(mean={sampling_sel.mean}, std={sampling_sel.std})"
        )
    else:
        raise TypeError(f"Unsupported selection config: {type(pipeline.selection_config).__name__}")

    # --- Set up FeatureIndex (deferred — only needed for generate/judge stages) ---
    feature_index = None

    # --- Stage variables (populated sequentially when stage="all") ---
    scan_tensors: ScanResultTensors | None = None
    selections = None
    doc_texts: list[str] | None = None

    # --- Resolve validation feature shard (needed by scan, threshold, generate) ---
    all_feature_shards = glob_features(str(feature_dir))
    if not all_feature_shards:
        logger.error(f"No feature shards found in {feature_dir}")
        sys.exit(1)
    val_shard_path = all_feature_shards[-1]
    val_shard_stem = shard_stem_from_path(val_shard_path)
    logger.info(f"Validation feature shard: {val_shard_path.name} (stem={val_shard_stem})")

    # -----------------------------------------------------------------------
    # Stage 1 — scan
    # -----------------------------------------------------------------------
    if stage in (EvalStage.ALL, EvalStage.SCAN):
        scan_output_path = output_dir / "validation_scan.safetensors"

        if _should_skip_stage(stage, scan_output_path, force_recompute):
            logger.info("=== Stage 1: Scan (skipped — output exists) ===")
            scan_tensors = load_scan_results(scan_output_path)
            logger.info(f"Loaded {len(scan_tensors.feature_ids)} features from {scan_output_path}")
        else:
            logger.info("=== Stage 1: Scan ===")

            # Vectorized scan — all tensor operations, no Python loops.
            scan_tensors = scan_validation_features_vectorized(
                val_shard_path,
                per_feature_stats,
            )
            n_features = len(scan_tensors.feature_ids)
            n_acts = len(scan_tensors.doc_idx)
            logger.info(f"Scan complete: {n_features} features, {n_acts} total activations")

            # Save scan results (direct tensor write — instant)
            save_scan_results(scan_tensors, scan_output_path)
            logger.info(f"Scan results saved to {scan_output_path}")

    # -----------------------------------------------------------------------
    # Stage 2 — threshold
    # -----------------------------------------------------------------------
    thresholded_scan_path = output_dir / "validation_scan_thresholded.safetensors"

    if stage in (EvalStage.ALL, EvalStage.THRESHOLD):
        if _should_skip_stage(stage, thresholded_scan_path, force_recompute):
            logger.info("=== Stage 2: Threshold (skipped — output exists) ===")
            scan_tensors = load_scan_results(thresholded_scan_path)
            logger.info(
                f"Loaded {len(scan_tensors.feature_ids)} features from {thresholded_scan_path}"
            )
        else:
            logger.info("=== Stage 2: Feature Selection Scan ===")
            if isinstance(feature_cfg, ThresholdConfig):
                scan_tensors = scan_thresholded_features(
                    val_shard_path,
                    per_feature_stats,
                    feature_cfg,
                    precomputed_thresholds,
                )
            else:
                scan_tensors = scan_sampled_features(
                    val_shard_path,
                    per_feature_stats,
                    feature_cfg,
                )
            n_features = len(scan_tensors.feature_ids)
            n_acts = len(scan_tensors.doc_idx)
            logger.info(
                f"Feature selection scan complete: {n_features} features, "
                f"{n_acts} total activations"
            )
            save_scan_results(scan_tensors, thresholded_scan_path)
            logger.info(f"Feature selection scan saved to {thresholded_scan_path}")

    # -----------------------------------------------------------------------
    # Stage 3 — select
    # -----------------------------------------------------------------------
    if stage in (EvalStage.ALL, EvalStage.SELECT):
        selections_path = output_dir / "feature_selections.yaml"

        if _should_skip_stage(stage, selections_path, force_recompute):
            logger.info("=== Stage 3: Select (skipped — output exists) ===")
            selections = _load_selections_yaml(selections_path)
            assert_selections_within_mask(selections, selection_validation_mask)
            logger.info(f"Loaded {len(selections)} selections from {selections_path}")
        else:
            logger.info("=== Stage 3: Select ===")

            # Load thresholded scan tensors if not in memory
            if scan_tensors is None:
                if not thresholded_scan_path.exists():
                    logger.error(
                        f"Thresholded scan results not found at {thresholded_scan_path}. "
                        "Run stage=threshold first."
                    )
                    sys.exit(1)
                logger.info(f"Loading thresholded scan results from {thresholded_scan_path}")
                scan_tensors = load_scan_results(thresholded_scan_path)
                logger.info(
                    f"Loaded {len(scan_tensors.feature_ids)} features from thresholded scan"
                )

            # Load FeatureIndex (needed for context-length filtering)
            if feature_index is None:
                logger.info(f"Scanning feature shards in {feature_dir}...")
                feature_index = FeatureIndex(str(feature_dir))
                logger.info(f"FeatureIndex: {feature_index.n_shards} shards discovered")

            # Collect all doc indices for insertion mode
            all_doc_indices: set[int] | None = None
            if mode is EvalMode.INSERTION:
                val_docs = feature_index._load_shard(val_shard_stem)
                all_doc_indices = set(val_docs.keys())
                logger.info(f"Insertion mode: {len(all_doc_indices)} total docs in val shard")

            # Load doc texts for context-length filtering during selection
            all_parquets = list_parquet_files()
            if not all_parquets:
                logger.error("No parquet files found in dataset directory")
                sys.exit(1)
            val_parquet = all_parquets[-1]
            logger.info(f"Loading validation text from {val_parquet}")
            table = pq.read_table(val_parquet, columns=["text"])
            doc_texts = table.column("text").to_pylist()
            logger.info(f"Loaded {len(doc_texts)} documents from validation parquet")

            max_seq_len = model.dec_config.sequence_len
            bos_id = eval_tokenizer.get_bos_token_id()

            ctx_f = ContextFilter(
                doc_texts=doc_texts,
                feature_index=feature_index,
                val_shard_stem=val_shard_stem,
                tokenizer=eval_tokenizer,
                max_seq_len=max_seq_len,
                bos_id=bos_id,
            )

            # Randomly select features and sample documents from tensors,
            # excluding any documents whose prefix exceeds the context window.
            selections = select_from_tensors(
                scan_tensors=scan_tensors,
                top_k=cfg.top_k_features,
                docs_per_feature=cfg.docs_per_feature,
                mode=mode,
                all_doc_indices=all_doc_indices,
                ctx_filter=ctx_f,
                feature_mask=feature_mask,
            )
            logger.info(f"Randomly selected {len(selections)} features")

            # Fetch Neuronpedia descriptions for selected features only
            selected_feature_ids = [s.feature_id for s in selections]

            np_cache_path = cfg.neuronpedia.cache_path or str(output_dir / "neuronpedia_cache.yaml")
            np_client = NeuronpediaClient(
                model_id=cfg.neuronpedia.model_id,
                sae_source=cfg.neuronpedia.sae_source,
                cache_path=np_cache_path,
            )

            logger.info(
                f"Fetching Neuronpedia descriptions for {len(selected_feature_ids)} selected features..."
            )
            descriptions = np_client.get_descriptions(selected_feature_ids)
            logger.info(f"Fetched {len(descriptions)} descriptions")

            # Attach descriptions to selections
            selections = [
                FeatureSelection(
                    feature_id=s.feature_id,
                    neuronpedia_description=descriptions.get(
                        s.feature_id, f"Feature {s.feature_id}"
                    ),
                    n_docs_activated=s.n_docs_activated,
                    selected_docs=s.selected_docs,
                    target_segments=s.target_segments,
                )
                for s in selections
            ]
            assert_selections_within_mask(selections, selection_validation_mask)

            # Save selections
            selections_serializable = [
                {
                    "feature_id": s.feature_id,
                    "neuronpedia_description": s.neuronpedia_description,
                    "n_docs_activated": s.n_docs_activated,
                    "selected_docs": s.selected_docs,
                    "target_segments": {int(k): int(v) for k, v in s.target_segments.items()},
                }
                for s in selections
            ]
            with open(selections_path, "w") as f:
                yaml.safe_dump(selections_serializable, f, default_flow_style=False)
            logger.info(f"Feature selections saved to {selections_path}")

    # --- Shared state across stages ---
    completions: list[DocumentCompletion] | None = None

    # -----------------------------------------------------------------------
    # Stage 4 — generate
    # -----------------------------------------------------------------------
    if stage in (EvalStage.ALL, EvalStage.GENERATE):
        completions_path = output_dir / "completions.yaml"

        if _should_skip_stage(stage, completions_path, force_recompute):
            logger.info("=== Stage 4: Generate (skipped — output exists) ===")
            completions = _load_completions_yaml(completions_path)
            logger.info(f"Loaded {len(completions)} completions from {completions_path}")
        else:
            logger.info("=== Stage 4: Generate ===")

            # Lazily build FeatureIndex (only needed for generate/judge)
            if feature_index is None:
                logger.info(f"Scanning feature shards in {feature_dir}...")
                feature_index = FeatureIndex(str(feature_dir))
                logger.info(f"FeatureIndex: {feature_index.n_shards} shards discovered")

            # Load selections if not already in memory
            if selections is None:
                selections_path = output_dir / "feature_selections.yaml"
                if not selections_path.exists():
                    logger.error(
                        f"Feature selections not found at {selections_path}. "
                        "Run stage=select first."
                    )
                    sys.exit(1)
                logger.info(f"Loading feature selections from {selections_path}")
                selections = _load_selections_yaml(selections_path)
                assert_selections_within_mask(selections, selection_validation_mask)
                logger.info(f"Loaded {len(selections)} feature selections")

            # Build target segment maps for ablation mode from thresholded scan
            target_segment_maps: dict[int, dict[int, int]] | None = None
            if mode is EvalMode.ABLATION:
                if not thresholded_scan_path.exists():
                    logger.error(
                        f"Thresholded scan results required for ablation mode but "
                        f"not found at {thresholded_scan_path}. Run stage=threshold first."
                    )
                    sys.exit(1)
                if scan_tensors is None:
                    logger.info(f"Loading thresholded scan from {thresholded_scan_path}")
                    scan_tensors = load_scan_results(thresholded_scan_path)
                selected_fids = [s.feature_id for s in selections]
                target_segment_maps = build_target_segment_maps(scan_tensors, selected_fids)
                logger.info(
                    f"Built target segment maps for {len(target_segment_maps)} features "
                    f"(from {len(scan_tensors.feature_ids)} total)"
                )

            # Load validation parquet text (if not already loaded in Stage 3)
            if doc_texts is None:
                all_parquets = list_parquet_files()
                if not all_parquets:
                    logger.error("No parquet files found in dataset directory")
                    sys.exit(1)
                val_parquet = all_parquets[-1]
                logger.info(f"Loading validation text from {val_parquet}")
                table = pq.read_table(val_parquet, columns=["text"])
                doc_texts = table.column("text").to_pylist()
                logger.info(f"Loaded {len(doc_texts)} documents from validation parquet")

            samples_per_condition = cfg.samples_per_condition
            temperature = cfg.temperature
            top_k_sampling = cfg.top_k_sampling
            bos_id = eval_tokenizer.get_bos_token_id()

            completions = _generate_completions(
                selections=selections,
                target_segment_maps=target_segment_maps,
                doc_texts=doc_texts,
                feature_index=feature_index,
                val_shard_stem=val_shard_stem,
                model=model,
                tokenizer=eval_tokenizer,
                mode=mode,
                boundaries=boundaries,
                feature_cfg=feature_cfg,
                precomputed_thresholds=precomputed_thresholds,
                device=device,
                num_buckets=num_buckets,
                samples_per_condition=samples_per_condition,
                temperature=temperature,
                top_k_sampling=top_k_sampling,
                bos_id=bos_id,
                next_segment_conditioning_offset=nextseg_eval,
                feature_mask=feature_mask,
            )

            # Save completions to YAML
            completions_serializable = []
            for c in completions:
                entry: dict = {
                    "doc_idx": c.doc_idx,
                    "feature_id": c.feature_id,
                    "pre_segment_text": c.pre_segment_text,
                    "ground_truth_segment": c.ground_truth_segment,
                    "target_segment_idx": c.target_segment_idx,
                    "samples": [
                        {
                            "condition": str(s.condition),
                            "text": s.text,
                            "seed": s.seed,
                            "bucket_index": s.bucket_index,
                        }
                        for s in c.samples
                    ],
                }
                gt = c.ground_truth_signals
                if gt is None:
                    raise ValueError(
                        "Completion is missing ground_truth_signals; "
                        "generate_for_document should populate this field."
                    )
                entry["ground_truth_signals"] = {
                    "feature_signals": {int(k): int(v) for k, v in gt.feature_signals.items()},
                    "n_features_active": gt.n_features_active,
                    "avg_magnitude": gt.avg_magnitude,
                    "max_magnitude": gt.max_magnitude,
                    "min_magnitude": gt.min_magnitude,
                    "feature_of_interest_magnitude": gt.feature_of_interest_magnitude,
                }
                completions_serializable.append(entry)
            with open(completions_path, "w") as f:
                yaml.safe_dump(completions_serializable, f, default_flow_style=False)
            logger.info(f"Saved {len(completions)} document completions to {completions_path}")

    # -----------------------------------------------------------------------
    # Stage 5a — feature analysis (probe)
    # -----------------------------------------------------------------------
    if stage is EvalStage.FEATURE_ANALYSIS or (stage is EvalStage.ALL and run_feature_analysis):
        probe_scores_path = output_dir / "probe_scores.yaml"

        if _should_skip_stage(stage, probe_scores_path, force_recompute):
            logger.info("=== Stage 5a: Feature Analysis (skipped — output exists) ===")
            logger.info(f"Probe scores already at {probe_scores_path}")
        else:
            logger.info("=== Stage 5a: Feature Analysis ===")

            # Load completions if not already in memory
            if completions is None:
                completions_path = output_dir / "completions.yaml"
                if not completions_path.exists():
                    logger.error(
                        f"Completions not found at {completions_path}. Run stage=generate first."
                    )
                    sys.exit(1)
                logger.info(f"Loading completions from {completions_path}")
                completions = _load_completions_yaml(completions_path)
                logger.info(f"Loaded {len(completions)} document completions")

            # Free conditional model before loading base model + SAE
            # (model is always loaded at script start, even if not used by this stage)
            logger.info("Freeing conditional model from GPU...")
            del model
            if device.type == "cuda":
                t.cuda.empty_cache()

            use_raw = bool(cfg.feature_analysis.use_raw_activation)
            probe_results = compute_probe_for_completions(
                completions=completions,
                feature_dir=feature_dir,
                use_raw_activation=use_raw,
                device=device,
            )

            # Generate plots
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Load selections for feature descriptions (if not in memory)
            if selections is None:
                selections_path = output_dir / "feature_selections.yaml"
                if not selections_path.exists():
                    logger.error(
                        f"Feature selections not found at {selections_path}. "
                        "Run stage=select first."
                    )
                    sys.exit(1)
                selections = _load_selections_yaml(selections_path)
                assert_selections_within_mask(selections, selection_validation_mask)
            feature_descriptions: dict[int, str] = {
                s.feature_id: s.neuronpedia_description for s in selections
            }

            for fid, presults in probe_results.items():
                desc = feature_descriptions.get(fid, f"Feature {fid}")

                if mode is EvalMode.ABLATION:
                    real_vals = presults.condition_scores.get("real", [])
                    ablated_vals = presults.condition_scores.get("ablated", [])

                    try:
                        r_m, r_se = probe_mean_stderr(real_vals)
                        a_m, a_se = probe_mean_stderr(ablated_vals)
                    except ValidationError:
                        logger.warning(f"Skipping probe plot for feature {fid}: insufficient data")
                        continue

                    plot_probe_ablation(
                        real_mean=r_m,
                        ablated_mean=a_m,
                        feature_id=fid,
                        feature_description=desc,
                        output_path=plots_dir / f"probe_ablation_feature_{fid}.png",
                        real_stderr=r_se,
                        ablated_stderr=a_se,
                    )
                else:
                    # Insertion mode — derive num_buckets from condition keys
                    bucket_keys = sorted(
                        k for k in presults.condition_scores if k.startswith("added_bucket_")
                    )
                    bucket_means: list[float] = []
                    bucket_stderrs: list[float] = []
                    try:
                        for key in bucket_keys:
                            vals = presults.condition_scores[key]
                            m, se = probe_mean_stderr(vals)
                            bucket_means.append(m)
                            bucket_stderrs.append(se)

                    except ValidationError:
                        logger.warning(f"Skipping probe plot for feature {fid}: insufficient data")
                        continue

                    real_vals = presults.condition_scores.get("real", [])
                    real_clean = [v for v in real_vals if not math.isnan(v)]
                    real_mean = sum(real_clean) / len(real_clean) if real_clean else None

                    plot_probe_dose_response(
                        bucket_means=bucket_means,
                        bucket_stderrs=bucket_stderrs,
                        feature_id=fid,
                        feature_description=desc,
                        output_path=plots_dir / f"probe_dose_response_feature_{fid}.png",
                        real_baseline=real_mean,
                    )

            # Save probe scores to YAML
            scores_serializable: dict = {}
            for fid, presults in probe_results.items():
                scores_serializable[int(fid)] = {
                    cond: [{"value": v} for v in vals]
                    for cond, vals in presults.condition_scores.items()
                }
            with open(probe_scores_path, "w") as f:
                yaml.safe_dump(scores_serializable, f, default_flow_style=False)
            logger.info(f"Saved probe scores to {probe_scores_path}")

    # -----------------------------------------------------------------------
    # Stage 5 — judge
    # -----------------------------------------------------------------------
    if stage is EvalStage.JUDGE or (stage is EvalStage.ALL and run_judge):
        scores_path = output_dir / "judge_scores.yaml"

        if _should_skip_stage(stage, scores_path, force_recompute):
            logger.info("=== Stage 5: Judge (skipped — output exists) ===")
            logger.info(f"Judge scores already at {scores_path}")
        else:
            logger.info("=== Stage 5: Judge ===")

            # Load completions if not already in memory
            if completions is None:
                completions_path = output_dir / "completions.yaml"
                if not completions_path.exists():
                    logger.error(
                        f"Completions not found at {completions_path}. Run stage=generate first."
                    )
                    sys.exit(1)
                logger.info(f"Loading completions from {completions_path}")
                completions = _load_completions_yaml(completions_path)
                logger.info(f"Loaded {len(completions)} document completions")

            # Load selections for feature descriptions
            if selections is None:
                selections_path = output_dir / "feature_selections.yaml"
                if not selections_path.exists():
                    logger.error(
                        f"Feature selections not found at {selections_path}. "
                        "Run stage=select first."
                    )
                    sys.exit(1)
                selections = _load_selections_yaml(selections_path)
                assert_selections_within_mask(selections, selection_validation_mask)

            # Build feature_id -> description mapping
            feature_descriptions: dict[int, str] = {
                s.feature_id: s.neuronpedia_description for s in selections
            }

            # Set up judge client
            judge_model = cfg.judge.model
            judge_rpm = cfg.judge.requests_per_minute
            judge_cache_path = str(output_dir / "judge_cache.db")

            judge_concurrency = cfg.judge.concurrency
            judge_client = CompletionJudgeClient(
                model=judge_model,
                cache_path=judge_cache_path,
                requests_per_minute=judge_rpm,
                concurrency=judge_concurrency,
            )

            # Judge all completions and collect scores
            # Group completions by feature_id for analysis
            feature_completions: dict[int, list[DocumentCompletion]] = {}
            for comp in completions:
                feature_completions.setdefault(comp.feature_id, []).append(comp)

            all_scores: dict[
                int, dict[str, list[JudgementScore]]
            ] = {}  # feature_id -> condition -> scores

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Count total samples for progress bar
            total_samples = sum(
                len(s) for comps in feature_completions.values() for c in comps for s in [c.samples]
            )
            judge_pbar = tqdm(total=total_samples, desc="Judging", unit="sample")

            asyncio.run(
                _judge_all_features_async(
                    judge_client=judge_client,
                    feature_completions=feature_completions,
                    feature_descriptions=feature_descriptions,
                    all_scores=all_scores,
                    mode=mode,
                    num_buckets=num_buckets,
                    plots_dir=plots_dir,
                    scores_path=scores_path,
                    judge_pbar=judge_pbar,
                )
            )

            judge_pbar.close()
            logger.info(f"Saved judge scores to {scores_path}")

            judge_client.close()

    logger.info("Completion steering eval pipeline complete.")


def _plot_ablation_for_feature(
    feature_id: int,
    description: str,
    condition_scores: dict[str, list[JudgementScore]],
    plots_dir: Path,
) -> None:
    """Generate ablation comparison plot for a single feature."""
    real_scores = condition_scores.get("real", [])
    ablated_scores = condition_scores.get("ablated", [])

    real_mean, real_stderr = compute_mean_stderr_or_zero(
        [float(score.feature_alignment) for score in real_scores]
    )
    ablated_mean, ablated_stderr = compute_mean_stderr_or_zero(
        [float(score.feature_alignment) for score in ablated_scores]
    )

    plot_ablation_comparison(
        real_mean=real_mean,
        ablated_mean=ablated_mean,
        feature_id=feature_id,
        feature_description=description,
        output_path=plots_dir / f"ablation_feature_{feature_id}.png",
        real_stderr=real_stderr,
        ablated_stderr=ablated_stderr,
    )


def _plot_insertion_for_feature(
    feature_id: int,
    description: str,
    condition_scores: dict[str, list[JudgementScore]],
    num_buckets: int,
    plots_dir: Path,
) -> None:
    """Generate dose-response plot for a single feature."""
    # Collect per-bucket scores
    scores_by_bucket: dict[int, list[JudgementScore]] = {}
    for b in range(num_buckets):
        key = f"added_bucket_{b}"
        scores_by_bucket[b] = condition_scores.get(key, [])

    dose_data = compute_dose_response(scores_by_bucket)

    # Compute real baseline
    real_scores = condition_scores.get("real", [])
    real_mean = (
        sum(s.feature_alignment for s in real_scores) / len(real_scores) if real_scores else None
    )

    plot_dose_response(
        dose_data=dose_data,
        feature_id=feature_id,
        feature_description=description,
        output_path=plots_dir / f"dose_response_feature_{feature_id}.png",
        real_baseline=real_mean,
    )


if __name__ == "__main__":
    main()
