"""Target model metadata detection (spec §4.3).

Reads training-time configuration off the target model so that downstream
stages (coefficient grids, partition modifiers, SAE extraction) can
adapt to the model's specific feature-selection + bucketing choices.

For encoder-decoder targets, the encoder's config holds ``num_buckets``
and ``n_sae_features`` directly. The training-time ``user_config``
(including ``negative_selection`` and ``feature_selection``) is stored
in ``model._loaded_meta_data`` — the caller of ``build_encdec_model``
is responsible for attaching ``meta_data`` to the returned model:

    model, tokenizer, meta_data = build_encdec_model(...)
    model._loaded_meta_data = meta_data  # REQUIRED before detect_target_model_metadata

Decoder-only targets have no encoder and no training feature-selection
config, so all encoder/selection fields on the returned dataclass are
``None``.

``sae_layer`` and ``segment_size`` are always populated (both branches)
by parsing ``cfg.feature_dir`` via
:func:`steering.feature_paths.parse_feature_dir`. They are the runtime
stand-ins for the previously hardcoded ``_SAE_LAYER = 8`` /
``_MAX_CTX_CLAMP = 2048`` constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from steering.conditional.feature_clusters import (
    assert_resume_feature_clusters_match,
    reject_feature_clusters_metadata,
)
from steering.eval.axbench_base.config import TargetMetadataConfig
from steering.eval.axbench_base.types import AxBenchBaseError, TargetModelMetadata
from steering.feature_paths import parse_feature_dir

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


def detect_target_model_metadata(
    target_model: Any,
    cfg: TargetMetadataConfig,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> TargetModelMetadata:
    """Extract bucket + feature selection semantics from the target model.

    For encoder-decoder: reads ``num_buckets`` and ``n_sae_features`` from
    ``model.encoder.config``; reads ``negative_selection`` and
    ``feature_selection`` from ``model._loaded_meta_data["user_config"]``.

    For decoder-only: returns ``TargetModelMetadata`` with ``None`` for
    all encoder-specific fields and ``device`` from ``cfg.device``.

    Both branches populate ``sae_layer`` / ``segment_size`` from
    ``cfg.feature_dir`` via :func:`parse_feature_dir`.

    Raises:
        AxBenchBaseError: if the target model looks like an encoder-decoder
            (has a ``.encoder`` attribute) but is missing the
            ``_loaded_meta_data`` attribute. The caller must attach this
            after ``build_encdec_model`` returns.
    """
    parsed = parse_feature_dir(Path(cfg.feature_dir))
    sae_layer = int(parsed.layer)
    segment_size = int(parsed.segment_size)
    if hasattr(target_model, "encoder"):
        num_buckets = target_model.encoder.config.num_buckets
        n_sae_features = target_model.encoder.config.n_sae_features
        meta_data = getattr(target_model, "_loaded_meta_data", None)
        if meta_data is None:
            raise AxBenchBaseError(
                "Target encdec model lacks _loaded_meta_data attribute. "
                "After calling build_encdec_model(), attach meta_data to "
                "the model: `model._loaded_meta_data = meta_data` so "
                "detect_target_model_metadata() can read the training "
                "user_config."
            )
        user_cfg = meta_data.get("user_config", {}) or {}
        if cluster_contract is None:
            reject_feature_clusters_metadata(
                meta_data,
                "axbench_base target model metadata",
            )
        else:
            assert_resume_feature_clusters_match(
                cluster_contract.consolidation.resolved,
                meta_data,
                checkpoint_dir="axbench_base target model metadata",
            )
        return TargetModelMetadata(
            num_buckets=num_buckets,
            n_sae_features=n_sae_features,
            negative_selection=user_cfg.get("negative_selection"),
            feature_selection=user_cfg.get("feature_selection"),
            feature_mask=user_cfg.get("feature_mask"),
            per_feature_stats=None,
            device=cfg.device,
            sae_layer=sae_layer,
            segment_size=segment_size,
        )
    return TargetModelMetadata(
        num_buckets=None,
        n_sae_features=None,
        negative_selection=None,
        feature_selection=None,
        feature_mask=None,
        per_feature_stats=None,
        device=cfg.device,
        sae_layer=sae_layer,
        segment_size=segment_size,
    )
