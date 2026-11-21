"""Reconstruct the training-time ``FeaturePipelineConfig`` from eval ``cfg``.

The encoder-decoder dataloader's fused-inline path consumes a
``FeaturePipelineConfig`` to apply feature selection on every base shard
that lacks a matching selection shard. Selection shards are optional
(operators usually don't have them), so the eval BPB block must rebuild
the same pipeline that training built — using ``cfg.conditioning`` and
the canonical statistics file.

The previous placeholder pipeline (sentinel threshold=0.95, zeros
boundaries) produced garbage feature selection when the fused-inline
fallback fired. Reconstruction here keeps the conditioned BPB path
correct in the common no-selection-shards case.

Pinned invariants (matching ``compute_canonical_feature_selection_hash``):
- ``seed=0``
- ``ddp_rank=0``
- ``feature_delta=None`` (BPB doesn't apply delta conditioning by default)

Stats-file resolution:
1. If ``cfg.bpb.stats_path`` is set, that path wins.
2. Otherwise, derive the canonical feature_dir from ``cfg.conditioning.identity``
   (model_id, sae_id → activation+width via ``_parse_sae_id``, layer, pooling)
   plus ``cfg.bpb.feature_root`` (or default), ``cfg.bpb.segment_size`` (or
   ``cfg.segment_size``), and ``cfg.dataset_phase`` (defaulting to ``base``),
   then call ``find_statistics_path(feature_dir)``.
3. If neither resolves, raise ``ValueError`` with an operator-actionable
   message naming both ``cfg.bpb.stats_path`` and the canonical feature_dir
   layout.
"""

from __future__ import annotations

from pathlib import Path

from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.eval.eval_cache_paths import _parse_sae_id
from steering.eval.segment_mode_guard import require_fixed_feature_dir_segment_mode
from steering.feature_paths import (
    _LABEL_TO_ACTIVATION,
    PRODUCER_IGNORE_BOS_FEATURES,
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
)
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth
from steering.stats_io import find_statistics_path, load_statistics

# Default feature_root matches the extraction conventions
# (configs/extraction/*.yaml all use this).
DEFAULT_FEATURE_ROOT = "out/data/features"


def _resolve_stats_path(cfg) -> Path:
    """Resolve the statistics file path used to build the pipeline.

    Prefers ``cfg.bpb.stats_path`` when set; otherwise derives the canonical
    feature_dir from ``cfg.conditioning.identity`` + ``cfg.bpb.feature_root``
    + segment_size + dataset_phase, then calls
    ``find_statistics_path(feature_dir)``.

    Raises ``ValueError`` with an operator-actionable message if neither
    path resolves to a stats file on disk.
    """
    bpb_cfg = cfg.get("bpb", None)
    explicit_path: str | None = None
    if bpb_cfg is not None:
        explicit_path = bpb_cfg.get("stats_path", None)
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise ValueError(
                f"cfg.bpb.stats_path={explicit_path!r} does not exist. "
                f"Set to the absolute path of a "
                f"feature_statistics.safetensors file produced by the "
                f"extraction pipeline."
            )
        return path

    cond_cfg = cfg.get("conditioning", None)
    if cond_cfg is None:
        raise ValueError(
            "Cannot resolve stats file for BPB conditioning: "
            "cfg.conditioning is required to derive the canonical feature_dir. "
            "Set cfg.bpb.stats_path explicitly, or populate "
            "cfg.conditioning.identity (model_id/sae_id/layer/pooling)."
        )
    identity_cfg = cond_cfg.get("identity", None)
    if identity_cfg is None:
        raise ValueError(
            "Cannot resolve stats file for BPB conditioning: "
            "cfg.conditioning.identity is required to derive the canonical "
            "feature_dir. Set cfg.bpb.stats_path explicitly, or populate "
            "cfg.conditioning.identity.{model_id,sae_id,layer,pooling}."
        )

    model_id = identity_cfg.get("model_id", None)
    sae_id = identity_cfg.get("sae_id", None)
    layer_raw = identity_cfg.get("layer", None)
    pooling_raw = identity_cfg.get("pooling", None)
    if not all([model_id, sae_id, layer_raw is not None, pooling_raw]):
        raise ValueError(
            "Cannot resolve stats file for BPB conditioning: "
            "cfg.conditioning.identity must include model_id, sae_id, layer, "
            "and pooling. Set cfg.bpb.stats_path explicitly to bypass "
            "auto-resolution."
        )

    activation_label, width_int = _parse_sae_id(sae_id)
    activation_type = _LABEL_TO_ACTIVATION[activation_label]
    # Width int → SAEWidth enum lookup
    width: SAEWidth | None = None
    for w_enum, w_int in SAE_WIDTH_TO_NUM_FEATURES.items():
        if w_int == width_int:
            width = w_enum
            break
    if width is None:
        raise ValueError(
            f"Cannot map SAE width {width_int} to SAEWidth enum. Set "
            f"cfg.bpb.stats_path explicitly to bypass auto-resolution."
        )

    pooling = PoolingMode(pooling_raw)

    # segment_size: cfg.bpb.segment_size overrides cfg.segment_size
    bpb_segment_size: int | None = None
    if bpb_cfg is not None:
        bpb_segment_size = bpb_cfg.get("segment_size", None)
    segment_size = bpb_segment_size if bpb_segment_size is not None else cfg.get(
        "segment_size", None
    )
    if segment_size is None:
        raise ValueError(
            "Cannot resolve stats file for BPB conditioning: "
            "neither cfg.bpb.segment_size nor cfg.segment_size is set. "
            "Set one (matching the segment_size of the feature extraction), "
            "or set cfg.bpb.stats_path explicitly."
        )

    feature_root = DEFAULT_FEATURE_ROOT
    if bpb_cfg is not None:
        bpb_feature_root = bpb_cfg.get("feature_root", None)
        if bpb_feature_root is not None:
            feature_root = bpb_feature_root

    dataset_phase_raw = cfg.get("dataset_phase", None)
    dataset_phase = (
        DatasetPhase(dataset_phase_raw) if dataset_phase_raw else DatasetPhase.BASE
    )

    feature_dir = build_feature_dir(
        root=feature_root,
        segment_size=int(segment_size),
        layer=int(layer_raw),
        activation_type=activation_type,
        pooling=pooling,
        width=width,
        model=str(model_id),
        dataset_phase=dataset_phase,
        ignore_bos_features=PRODUCER_IGNORE_BOS_FEATURES,
    )
    found = find_statistics_path(feature_dir)
    if found is None:
        raise ValueError(
            f"Cannot resolve stats file for BPB conditioning: no "
            f"feature_statistics.safetensors found under {feature_dir}. "
            f"Set cfg.bpb.stats_path explicitly to the absolute path of "
            f"the statistics_*.safetensors file, OR ensure the canonical "
            f"feature_dir contains a "
            f"feature_statistics.safetensors produced by the extraction "
            f"pipeline. "
            f"Auto-resolution layout: "
            f"{{cfg.bpb.feature_root or 'out/data/features'}}/"
            f"s{{segment_size}}/l{{layer}}/a_{{act}}/p_{{pool}}/"
            f"f{{width}}/m_{{model}}/{{dataset_phase}}/"
            f"feature_statistics.safetensors"
        )
    return found


def build_feature_pipeline_from_cfg(cfg) -> FeaturePipelineConfig:
    """Reconstruct the training-time ``FeaturePipelineConfig`` from eval cfg.

    Reads:
      - ``cfg.conditioning.feature_selection`` → ``FeatureSelection``
      - ``cfg.conditioning.negative_selection`` (optional) → ``NegativeSamplingSelection``
      - ``cfg.conditioning.num_buckets`` (default 2)
      - ``cfg.conditioning.n_sae_features`` (default 16384)
      - Statistics file via ``_resolve_stats_path`` (cfg.bpb.stats_path
        explicit or auto-resolved from cfg.conditioning.identity).

    Pinned invariants:
      - ``seed=0`` (matches ``compute_canonical_feature_selection_hash``)
      - ``ddp_rank=0``
      - ``feature_delta=None`` (BPB doesn't apply delta conditioning by default)

    Raises:
        ValueError: when the stats file can't be resolved, or
            ``feature_selection`` is missing/null in ``cfg.conditioning``.
        TypeError: when ``cfg.conditioning.feature_selection`` resolves to a
            non-mapping (e.g. string, list). Error originates in the shared
            helper ``build_feature_pipeline_from_cfg_block``.
    """
    cond_cfg = cfg.get("conditioning", None)
    if cond_cfg is None:
        raise ValueError(
            "build_feature_pipeline_from_cfg requires cfg.conditioning to "
            "supply feature_selection / negative_selection / num_buckets / "
            "n_sae_features."
        )
    if cond_cfg.get("feature_selection", None) is None:
        raise ValueError(
            "build_feature_pipeline_from_cfg requires "
            "cfg.conditioning.feature_selection to be set (matching the "
            "training-time pipeline). Add a feature_selection block under "
            "cfg.conditioning, e.g. "
            "{threshold_type: feature_percentile, threshold: 0.95}."
        )

    num_buckets = int(cond_cfg.get("num_buckets", 2))
    n_sae_features = int(cond_cfg.get("n_sae_features", 16384))

    stats_path = _resolve_stats_path(cfg)
    feature_stats = load_statistics(stats_path)
    require_fixed_feature_dir_segment_mode(
        stats_path.parent,
        surface="conditioned BPB evaluation",
        statistics_segment_mode=feature_stats.metadata.segment_mode,
    )
    per_feature_stats = feature_stats.per_feature_stats

    return build_feature_pipeline_from_cfg_block(
        cond_cfg,
        per_feature_stats=per_feature_stats,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        seed=0,
        ddp_rank=0,
        feature_delta=None,
    )
