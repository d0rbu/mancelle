"""Launch the completion TUI.

Usage: uv run python -m scripts.completion_tui

Auto-discovers available encoder-decoder checkpoints, matches them to
feature directories and parquet data, and launches the interactive TUI.
If multiple configurations are found, presents a selection menu.
"""

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig

from errors import FeatureClustersError, FeatureMaskError
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.completion_tui.app import (
    CompletionApp,
    _document_evidence_by_segment_id,
)
from scripts.completion_tui.checkpoint_contract import (
    assert_contract_supports_selection,
    assert_model_matches_contract,
    resolve_checkpoint_feature_contract,
)
from scripts.completion_tui.data_loader import DocumentLoader, TUIFeatureSamplingConfig
from scripts.completion_tui.description_service import DescriptionService
from scripts.completion_tui.discovery import (
    TUIConfig,
    _find_all_checkpoint_leaves,
    _find_last_step_safe,
    _load_meta,
    _nested_path,
    discover_configs,
    prompt_selection,
)
from scripts.completion_tui.generation import GenerationManager, LazyGenerationManager
from scripts.completion_tui.live_feature_highlight import (
    LiveFeatureHighlightService,
    resolve_probe_spec,
)
from scripts.completion_tui.state import SessionState
from scripts.completion_tui.transparent import enable_transparent_background
from steering.conditional.discretization import build_boundaries_from_stats
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.conditional.sampling import (
    SamplingSelection,
    ThresholdSelection,
    assert_no_gradient_ranking,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    build_thresholds_from_stats,
    extract_threshold_percentile,
)
from steering.eval.segment_mode_guard import (
    require_fixed_checkpoint_segment_mode,
    require_fixed_feature_dir_segment_mode,
)
from steering.feature_paths import parse_feature_dir
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_neuronpedia_ids,
)
from steering.stats_io import find_statistics_path, load_statistics

DEFAULT_DESCRIPTION_CACHE_PATH = "out/neuronpedia_cache.yaml"


@dataclass(frozen=True)
class LaunchOptions:
    device_type: str | None
    dtype: str
    temperature: float
    top_k: int | None
    max_new_tokens: int
    neuronpedia_model_id: str
    neuronpedia_sae_source: str
    description_cache_path: str


def _cfg_text(cfg: DictConfig, key: str) -> str | None:
    value = cfg.get(key, None)
    if value is None:
        return None
    value_str = str(value)
    return value_str if value_str else None


def _launch_options_from_cfg(cfg: DictConfig) -> LaunchOptions:
    return LaunchOptions(
        device_type=_cfg_text(cfg, "device_type"),
        dtype=str(cfg.dtype),
        temperature=float(cfg.temperature),
        top_k=None if cfg.top_k is None else int(cfg.top_k),
        max_new_tokens=int(cfg.max_new_tokens),
        neuronpedia_model_id=_cfg_text(cfg, "neuronpedia_model_id") or "",
        neuronpedia_sae_source=_cfg_text(cfg, "neuronpedia_sae_source") or "",
        description_cache_path=str(cfg.description_cache_path),
    )


def _resolve_model_dtype(dtype_name: str, device: t.device) -> t.dtype | None:
    if device.type != "cuda":
        return None
    if dtype_name == "bfloat16":
        return t.bfloat16
    if dtype_name == "float16":
        return t.float16
    if dtype_name == "float32":
        return t.float32
    raise ValueError("completion_tui dtype must be one of bfloat16, float16, float32")


def _build_config_from_explicit_overrides(cfg: DictConfig) -> TUIConfig:
    model_tag = _cfg_text(cfg, "model_tag")
    feature_dir = _cfg_text(cfg, "feature_dir")
    parquet_dir = _cfg_text(cfg, "parquet_dir")
    if model_tag is None or feature_dir is None or parquet_dir is None:
        raise ValueError(
            "Explicit completion_tui launch requires model_tag, feature_dir, "
            "and parquet_dir when auto-discovery cannot produce a matching config."
        )

    tag_dir = os.path.join(get_base_dir(), "conditional_encdec_checkpoints", model_tag)
    leaves = _find_all_checkpoint_leaves(tag_dir)
    if not leaves:
        raise ValueError(f"No checkpoint leaves found for model_tag={model_tag!r}")

    requested_step = cfg.get("step", None)
    best: tuple[str, int] | None = None
    for leaf in leaves:
        step = int(requested_step) if requested_step is not None else _find_last_step_safe(leaf)
        if step is None:
            continue
        meta = _load_meta(leaf, step)
        if meta is None:
            continue
        if best is None or step > best[1]:
            best = (leaf, step)
    if best is None:
        raise ValueError(f"No usable checkpoint metadata found for model_tag={model_tag!r}")

    checkpoint_dir, step = best
    meta = _load_meta(checkpoint_dir, step) or {}
    user_cfg = meta.get("user_config", {})
    try:
        parsed = parse_feature_dir(feature_dir)
        segment_size = parsed.segment_size
    except ValueError:
        segment_size = int(user_cfg.get("segment_size", 0))
    num_buckets = int(user_cfg.get("num_buckets", cfg.num_buckets))
    has_negatives = user_cfg.get("negative_selection") is not None
    nested = _nested_path(tag_dir, checkpoint_dir)
    label = f"{model_tag} / {nested}" if nested else model_tag
    return TUIConfig(
        model_tag=model_tag,
        checkpoint_dir=checkpoint_dir,
        nested_path=nested,
        step=step,
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        segment_size=segment_size,
        num_buckets=num_buckets,
        has_negatives=has_negatives,
        label=f"{label} (step {step}, explicit overrides)",
    )


def _select_config_from_cfg(cfg: DictConfig, configs: list[TUIConfig]) -> TUIConfig:
    source = _cfg_text(cfg, "source") or "encdec"
    if source != "encdec":
        raise ValueError("completion_tui only supports source=encdec")

    model_tag = _cfg_text(cfg, "model_tag")
    requested_step = cfg.get("step", None)
    explicit = any(
        value is not None
        for value in (
            model_tag,
            requested_step,
            _cfg_text(cfg, "feature_dir"),
            _cfg_text(cfg, "parquet_dir"),
        )
    )

    selected: TUIConfig
    if explicit:
        candidates = configs
        if model_tag is not None:
            candidates = [c for c in candidates if c.model_tag == model_tag]
        if requested_step is not None:
            step = int(requested_step)
            candidates = [c for c in candidates if c.step == step]
        if candidates:
            selected = max(candidates, key=lambda c: c.step)
        else:
            selected = _build_config_from_explicit_overrides(cfg)
    else:
        if not configs:
            raise ValueError(
                "No encoder-decoder checkpoints found. Ensure checkpoints exist under "
                "<NANOCHAT_BASE_DIR>/conditional_encdec_checkpoints/"
            )
        if len(configs) == 1:
            selected = configs[0]
            print(f"Auto-selected: {selected.label}")
        else:
            selected = prompt_selection(configs)

    feature_dir = _cfg_text(cfg, "feature_dir")
    parquet_dir = _cfg_text(cfg, "parquet_dir")
    if feature_dir is not None or parquet_dir is not None:
        selected = replace(
            selected,
            feature_dir=feature_dir or selected.feature_dir,
            parquet_dir=parquet_dir or selected.parquet_dir,
        )
    return selected


def _resolve_neuronpedia_ids_from_cfg(user_cfg: dict) -> tuple[str, str]:
    """Derive Neuronpedia model_id and sae_source from checkpoint config.

    Explicit ``neuronpedia_model_id`` / ``neuronpedia_sae_source`` fields take
    precedence independently. Missing fields are derived via
    ``resolve_neuronpedia_ids`` from the ``sae`` sub-config when possible.

    Returns empty strings for fields that cannot be resolved.
    """
    # Explicit overrides from the checkpoint/user config.
    model_id = str(user_cfg.get("neuronpedia_model_id", "") or "")
    sae_source = str(user_cfg.get("neuronpedia_sae_source", "") or "")
    if model_id and sae_source:
        return model_id, sae_source

    # Derive from sae config via sae_resolver
    sae_cfg = user_cfg.get("sae")
    if not isinstance(sae_cfg, dict):
        return model_id, sae_source

    try:
        derived_model_id, derived_sae_source = resolve_neuronpedia_ids(
            model=sae_cfg["model"],
            release=SAERelease(sae_cfg["release"]),
            activation_type=ActivationType(sae_cfg["activation_type"]),
            layer=sae_cfg["layer"],
            width=SAEWidth(sae_cfg["width"]),
            l0=coerce_config_l0(
                SAERelease(sae_cfg["release"]), sae_cfg.get("l0")
            ),
        )
    except (KeyError, ValueError):
        return model_id, sae_source
    return model_id or derived_model_id, sae_source or derived_sae_source


def _resolve_neuronpedia_ids(
    options: LaunchOptions,
    user_cfg: dict,
) -> tuple[str, str]:
    derived_model_id, derived_sae_source = _resolve_neuronpedia_ids_from_cfg(user_cfg)
    return (
        options.neuronpedia_model_id or derived_model_id,
        options.neuronpedia_sae_source or derived_sae_source,
    )


def _resolve_description_cache_path(options: LaunchOptions, selected: TUIConfig) -> Path:
    if options.description_cache_path == DEFAULT_DESCRIPTION_CACHE_PATH:
        return Path(selected.feature_dir) / "neuronpedia_cache.yaml"
    return Path(options.description_cache_path)


def _load_and_launch(
    selected: TUIConfig,
    *,
    lazy: bool = False,
    options: LaunchOptions | None = None,
) -> None:
    """Load model, build services, and launch the TUI app."""
    if options is None:
        options = LaunchOptions(
            device_type=None,
            dtype="bfloat16",
            temperature=0.7,
            top_k=50,
            max_new_tokens=256,
            neuronpedia_model_id="",
            neuronpedia_sae_source="",
            description_cache_path=DEFAULT_DESCRIPTION_CACHE_PATH,
        )
    device_type = options.device_type or autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)
    model_dtype = _resolve_model_dtype(options.dtype, device)

    if lazy:
        # Fast path: load only metadata + tokenizer, defer model weights
        meta_path = os.path.join(selected.checkpoint_dir, f"meta_{selected.step:06d}.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        tokenizer = get_tokenizer()
    else:
        meta_path = os.path.join(selected.checkpoint_dir, f"meta_{selected.step:06d}.json")
        logger.info(
            f"Loading model: {selected.model_tag} from {selected.checkpoint_dir} "
            f"(step {selected.step})"
        )
        model, tokenizer, meta = build_encdec_model(
            selected.checkpoint_dir,
            selected.step,
            device,
            ModelPhase.EVAL,
            eval_conditioning=EvalConditioning.conditions_within_vocab("completion_tui"),
        )

    require_fixed_checkpoint_segment_mode(
        meta,
        surface="completion TUI",
        checkpoint_dir=selected.checkpoint_dir,
    )

    user_cfg = meta.get("user_config", {})
    if not isinstance(user_cfg, dict):
        raise ValueError(f"completion_tui checkpoint {meta_path} user_config must be a mapping")

    # Build boundaries from feature statistics
    stats_path = find_statistics_path(selected.feature_dir)
    if stats_path is None:
        print(f"Error: No statistics file found in {selected.feature_dir}")
        sys.exit(1)

    feature_stats = load_statistics(stats_path)
    require_fixed_feature_dir_segment_mode(
        selected.feature_dir,
        surface="completion TUI",
        statistics_segment_mode=feature_stats.metadata.segment_mode,
    )
    n_sae_features = PositiveInt.parse(feature_stats.metadata.n_sae_features)
    contract = resolve_checkpoint_feature_contract(
        meta,
        n_sae_features=n_sae_features,
        meta_path=meta_path,
    )
    assert_contract_supports_selection(contract, user_cfg)
    if not lazy:
        assert_model_matches_contract(
            model.feature_vocab,
            contract,
            n_sae_features=n_sae_features,
            checkpoint_dir=selected.checkpoint_dir,
            loaded_model_n_sae_features=PositiveInt.parse(
                model.encoder.config.n_sae_features
            ),
        )
        if model_dtype is not None:
            model = model.to(dtype=model_dtype)

    # Parse feature_selection and negative_selection from checkpoint config.
    # These determine which features are enabled and how negatives are sampled.
    threshold_pct: float | None = None
    threshold_cfg: ThresholdConfig | None = None
    precomputed_thresholds: t.Tensor | None = None
    feature_selection = None
    negative_selection = None
    percentile_table: t.Tensor | None = None
    corpus_percentile_table = build_percentile_table(
        feature_stats.per_feature_stats, n_sae_features
    )

    feat_sel_cfg = user_cfg.get("feature_selection")
    if feat_sel_cfg:
        selection = parse_feature_selection(feat_sel_cfg)
        assert_no_gradient_ranking(selection, "completion TUI live selection")
        feature_selection = selection
        if isinstance(selection, ThresholdSelection):
            threshold_cfg = ThresholdConfig(selection.threshold_type, selection.threshold)
            threshold_pct = extract_threshold_percentile(threshold_cfg)
            precomputed_thresholds = build_thresholds_from_stats(
                feature_stats.per_feature_stats,
                n_sae_features,
                selection.threshold_type,
                selection.threshold,
            )
        elif isinstance(selection, SamplingSelection):
            percentile_table = corpus_percentile_table

    neg_sel_cfg = user_cfg.get("negative_selection")
    if neg_sel_cfg:
        negative_selection = parse_negative_selection(neg_sel_cfg)

    # Match the training pipeline: when negative_selection is active, positive
    # features use (num_buckets - 1) effective buckets so the last bucket is
    # reserved exclusively for negative suppression signals.
    effective_buckets = selected.num_buckets - 1 if selected.has_negatives else selected.num_buckets
    if effective_buckets <= 1:
        # Binary mode (1 positive bucket + 1 negative bucket): no intra-positive
        # boundaries — all active features get bucket 0.
        boundaries = t.zeros(n_sae_features, 0, dtype=t.float64)
    else:
        boundaries = build_boundaries_from_stats(
            feature_stats.per_feature_stats,
            n_sae_features,
            effective_buckets,
            threshold_percentile=threshold_pct,
        )

    # Build sampling config (used for both document loading and ctrl+r/r in feature editor)
    if contract.clustered:
        sampling_config = TUIFeatureSamplingConfig(
            boundaries=boundaries,
            n_sae_features=n_sae_features,
            feature_selection=feature_selection,
            negative_selection=negative_selection,
            percentile_table=percentile_table,
            precomputed_thresholds=precomputed_thresholds,
            feature_mask=contract.feature_mask,
            consolidation=contract.consolidation,
            effective_buckets=effective_buckets,
        )
    elif contract.feature_mask is not None:
        sampling_config = TUIFeatureSamplingConfig(
            boundaries=boundaries,
            n_sae_features=n_sae_features,
            feature_selection=feature_selection,
            negative_selection=negative_selection,
            percentile_table=percentile_table,
            precomputed_thresholds=precomputed_thresholds,
            feature_mask=contract.feature_mask,
        )
    else:
        sampling_config = TUIFeatureSamplingConfig(
            boundaries=boundaries,
            n_sae_features=n_sae_features,
            feature_selection=feature_selection,
            negative_selection=negative_selection,
            percentile_table=percentile_table,
            precomputed_thresholds=precomputed_thresholds,
        )

    # Build services
    if contract.clustered:
        doc_loader = DocumentLoader(
            selected.feature_dir,
            selected.parquet_dir,
            boundaries=boundaries,
            num_buckets=selected.num_buckets,
            has_negatives=selected.has_negatives,
            threshold_cfg=threshold_cfg,
            precomputed_thresholds=precomputed_thresholds,
            sampling_config=sampling_config,
            percentile_checksum=compute_percentile_checksum(corpus_percentile_table),
            # Clustered conditioning uses percentiles and feature IDs only;
            # retaining per-feature magnitudes for the entire validation shard
            # needlessly doubles the resident CPU footprint.
            skip_feature_mags=True,
            compact_feature_ids=True,
        )
    else:
        doc_loader = DocumentLoader(
            selected.feature_dir,
            selected.parquet_dir,
            boundaries=boundaries,
            num_buckets=selected.num_buckets,
            has_negatives=selected.has_negatives,
            threshold_cfg=threshold_cfg,
            precomputed_thresholds=precomputed_thresholds,
            sampling_config=sampling_config,
        )

    # Description service — derive Neuronpedia identifiers from SAE config,
    # with explicit overrides taking precedence.
    np_model_id, np_sae_source = _resolve_neuronpedia_ids(options, user_cfg)
    description_service: DescriptionService | None = None
    if np_model_id and np_sae_source:
        cache_path = _resolve_description_cache_path(options, selected)
        description_service = DescriptionService(
            model_id=np_model_id,
            sae_source=np_sae_source,
            cache_path=cache_path,
        )

    live_highlight_service: LiveFeatureHighlightService | None = None
    try:
        probe_spec = resolve_probe_spec(user_cfg, selected.feature_dir)
        live_highlight_service = LiveFeatureHighlightService(
            probe_spec,
            corpus_percentile_table,
            device=device,
        )
    except Exception as exc:
        logger.warning("Live feature highlighting unavailable: {}", exc)

    if lazy:

        def _build_manager() -> GenerationManager:
            m, _, lazy_meta = build_encdec_model(
                selected.checkpoint_dir,
                selected.step,
                device,
                ModelPhase.EVAL,
                eval_conditioning=EvalConditioning.conditions_within_vocab("completion_tui"),
            )
            require_fixed_checkpoint_segment_mode(
                lazy_meta,
                surface="completion TUI lazy model load",
                checkpoint_dir=selected.checkpoint_dir,
            )
            lazy_user_cfg = lazy_meta.get("user_config", {})
            if not isinstance(lazy_user_cfg, dict):
                raise ValueError(
                    f"completion_tui lazy checkpoint {meta_path} user_config must be a mapping"
                )
            lazy_contract = resolve_checkpoint_feature_contract(
                lazy_meta,
                n_sae_features=n_sae_features,
                meta_path=meta_path,
            )
            assert_contract_supports_selection(lazy_contract, lazy_user_cfg)
            if lazy_contract.feature_mask != contract.feature_mask:
                raise FeatureMaskError(
                    "completion_tui checkpoint feature_mask_canonical changed "
                    "between launch and lazy model loading"
                )
            launch_clusters = contract.resolved_clusters
            loaded_clusters = lazy_contract.resolved_clusters
            if (launch_clusters is None) != (loaded_clusters is None) or (
                launch_clusters is not None
                and loaded_clusters is not None
                and (
                    launch_clusters.descriptor != loaded_clusters.descriptor
                    or launch_clusters.definition != loaded_clusters.definition
                )
            ):
                raise FeatureClustersError(
                    "completion_tui checkpoint static feature-cluster metadata "
                    "changed between launch and lazy model loading"
                )
            assert_model_matches_contract(
                m.feature_vocab,
                lazy_contract,
                n_sae_features=n_sae_features,
                checkpoint_dir=selected.checkpoint_dir,
                loaded_model_n_sae_features=PositiveInt.parse(
                    m.encoder.config.n_sae_features
                ),
            )
            if model_dtype is not None:
                m = m.to(dtype=model_dtype)
            return GenerationManager(
                model=m,
                tokenizer=tokenizer,
                device=device,
                boundaries=boundaries,
                temperature=options.temperature,
                top_k=options.top_k,
                max_new_tokens=options.max_new_tokens,
                feature_mask=contract.feature_mask,
                consolidation=contract.consolidation,
            )

        generation_manager: GenerationManager | LazyGenerationManager = LazyGenerationManager(
            _build_manager
        )
    else:
        generation_manager = GenerationManager(
            model=model,
            tokenizer=tokenizer,
            device=device,
            boundaries=boundaries,
            temperature=options.temperature,
            top_k=options.top_k,
            max_new_tokens=options.max_new_tokens,
            feature_mask=contract.feature_mask,
            consolidation=contract.consolidation,
        )

    # Build initial state from first document (default to 2 segments)
    doc_ids = doc_loader.available_doc_ids()
    initial_doc = doc_loader.load_document(doc_ids[0], num_segments=2) if doc_ids else None
    initial_segment_evidence = (
        _document_evidence_by_segment_id(initial_doc) if initial_doc is not None else None
    )

    if initial_doc is not None:
        # Truncate text at generation target so marker shows remaining space
        segs = initial_doc.segments
        if segs:
            last = segs[-1]
            cutoff = segs[-2].char_end if len(segs) >= 2 else 0
            if not last.is_generation_target:
                last.is_generation_target = True
            seg_text = initial_doc.text[last.char_start : last.char_end]
            last.target_token_count = max(
                1,
                len(tokenizer.encode(seg_text)) if seg_text else 1,
            )
            last.char_start = cutoff
            last.char_end = cutoff
        text = initial_doc.text[: segs[-1].char_end] if segs else initial_doc.text
        state = SessionState(
            document_text=text,
            segments=initial_doc.segments,
            num_buckets=selected.num_buckets,
            has_negatives=selected.has_negatives,
        )
        total_segments_map = {}
        for did in doc_ids:
            total = doc_loader.total_segments(did)
            if total is not None:
                total_segments_map[did] = total
    else:
        state = SessionState(
            document_text="",
            segments=[],
            num_buckets=selected.num_buckets,
            has_negatives=selected.has_negatives,
        )
        total_segments_map = {}

    # Launch app
    app = CompletionApp(
        state=state,
        tokenizer=tokenizer,
        doc_loader=doc_loader,
        description_service=description_service,
        generation_manager=generation_manager,
        available_doc_ids=doc_ids,
        total_segments_map=total_segments_map,
        sampling_config=sampling_config,
        live_highlight_service=live_highlight_service,
        cluster_view=contract.cluster_view,
        initial_segment_evidence=initial_segment_evidence,
    )
    enable_transparent_background(app)
    app.run()


@hydra.main(config_path="../../configs/serving", config_name="completion_tui", version_base=None)
def main(cfg: DictConfig) -> None:
    options = _launch_options_from_cfg(cfg)
    configs = discover_configs()

    try:
        selected = _select_config_from_cfg(cfg, configs)
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Loading {selected.label}...")
    _load_and_launch(selected, lazy=cfg.lazy, options=options)


if __name__ == "__main__":
    main()
