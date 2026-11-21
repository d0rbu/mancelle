"""Checkpoint-bound raw-coordinate contract for target-real turn context caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from steering.conditional.checkpoint_feature_contract import (
    CheckpointFeatureContract,
    resolve_checkpoint_feature_contract,
)
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.sft_grpo_online_sae import OnlineSAELineage
from steering.conditional.encoder_decoder.sft_selection import SegmentSelectionContext
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    feature_clusters_checkpoint_metadata,
)
from steering.conditional.feature_mask import FeatureMask, canonical_dict
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline_from_cfg_block,
    negative_sampling_mask,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import (
    MissingPercentilePolicy,
    MissingPercentileResolver,
    parse_missing_percentile_policy,
    percentile_table_coverage,
)
from steering.conditional.sampling import SamplingBundle, assert_no_gradient_ranking
from steering.feature_paths import PoolingMode
from steering.models import EXTRACTION_MODELS
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum, resolve_bos_token_id
from steering.sft.segment_mode_identity import (
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    resolve_checkpoint_sft_segment_mode,
    resolve_turn_self_cond_segment_size_rustbpe,
)
from steering.stats_io import SAFETENSORS_EXT, load_statistics_and_checksum_from_bytes
from steering.stats_types import ExtractionStatistics

TURN_CONTRACT_VERSION = 1
TURN_COORDINATE_SYSTEM = "source_tokenizer_decoded_with_bos_v1"
TURN_SELECTION_SEMANTICS = "sft_select_conversation_segments_v1"
TURN_ENUMERATION_SEMANTICS = "legacy_chat_target_turns_v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Noncanonical turn contract value: {value!r}")


def canonical_turn_payload(payload: Mapping[str, Any]) -> bytes:
    """Canonical semantic bytes; physical artifact locations never enter these."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=_json_value).encode("utf-8")


@dataclass(frozen=True)
class TurnContextContract:
    """Resolved checkpoint authority, including one authenticated statistics snapshot."""

    semantic: dict[str, Any]
    lineage: OnlineSAELineage
    statistics: ExtractionStatistics
    feature_mask: FeatureMask | None
    consolidation: FeatureClusterConsolidation | None
    feature_vocab: FeatureVocab | None
    model_max_seq_len: int
    provenance: dict[str, Any]

    def build_selection_context(self) -> SegmentSelectionContext:
        """Start one lineage-seeded stream; callers retain it across every shard."""
        return _lineage_selection_context(
            self.lineage, self.statistics, self.feature_mask, self.consolidation,
        )


def _lineage_selection_context(
    lineage: OnlineSAELineage, statistics: ExtractionStatistics, feature_mask: FeatureMask | None,
    consolidation: FeatureClusterConsolidation | None,
) -> SegmentSelectionContext:
    pipeline = build_feature_pipeline_from_cfg_block(
        {"feature_selection": dict(lineage.feature_selection),
         "negative_selection": lineage.negative_selection},
        per_feature_stats=statistics.per_feature_stats,
        n_sae_features=lineage.n_sae_features,
        num_buckets=lineage.num_buckets,
        seed=lineage.selection_seed,
        feature_mask=feature_mask,
    )
    return build_turn_selection_context(
        pipeline=pipeline, missing_percentile_policy=lineage.missing_percentile_policy,
        consolidation=consolidation,
    )


def turn_extraction_tokenizer_id(model: str, configured_id: str | None = None) -> str:
    """Resolve the lineage model through the extraction registry; CLI IDs are assertions."""
    if model not in EXTRACTION_MODELS:
        raise ValueError(f"Unknown checkpoint extraction model: {model!r}")
    hf_id = EXTRACTION_MODELS[model].hf_model_id
    if configured_id is not None and configured_id not in (model, hf_id):
        raise ValueError("extraction_tokenizer_id conflicts with checkpoint SAE model")
    return hf_id


def resolve_turn_context_contract(
    checkpoint_meta: Mapping[str, Any] | str | Path,
    *,
    tokenizer,
    extraction_tokenizer,
    statistics_path: str | Path | None = None,
    checkpoint_contract: CheckpointFeatureContract | None = None,
) -> TurnContextContract:
    """Resolve static target-real conditioning; cadence is optional but validated if present."""
    provenance: dict[str, Any] = {}
    source = "<checkpoint metadata>"
    if isinstance(checkpoint_meta, (str, Path)):
        source = str(checkpoint_meta)
        payload = Path(checkpoint_meta).read_bytes()
        meta = json.loads(payload)
        provenance.update(checkpoint_meta_path=source, checkpoint_metadata_sha256=hashlib.sha256(payload).hexdigest())
    else:
        meta = dict(checkpoint_meta)
    if not isinstance(meta, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    mode = resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=source)
    if mode is not SegmentMode.turn_delimited_fixed_tokens:
        raise ValueError("turn context cache requires a turn-delimited checkpoint")
    if TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY in meta:
        resolve_turn_self_cond_segment_size_rustbpe(meta, checkpoint_dir=source)
    lineage = OnlineSAELineage.from_checkpoint_meta(meta)
    user_config = meta.get("user_config", {})
    # The SFT producer persists raw Hydra defaults in user_config.sae. Only the
    # online-SAE lineage records the inherited extraction coordinates.
    release = SAERelease(lineage.release)
    resolve_sae(model=lineage.model, release=release,
                activation_type=ActivationType(lineage.activation_type), layer=lineage.layer,
                width=SAEWidth(lineage.width), l0=coerce_config_l0(release, lineage.l0))
    buckets = meta.get("num_buckets", user_config.get("num_buckets"))
    if type(buckets) is not int or buckets != lineage.num_buckets:
        raise ValueError("checkpoint num_buckets conflicts with online-SAE lineage")
    for block in (meta, user_config, meta.get("pretrain_conditioning", {})):
        for field in ("num_buckets", "n_sae_features"):
            if block is user_config and field == "num_buckets" and block.get(field) is None:
                continue  # Documented SFT null means inherit the resolved checkpoint value.
            if field in block and (type(block[field]) is not int or block[field] != getattr(lineage, field)):
                raise ValueError(f"checkpoint {field} conflicts with online-SAE lineage")
    model_config = meta.get("model_config", {})
    model_max_seq_len = model_config.get("sequence_len")
    if type(model_max_seq_len) is not int or model_max_seq_len <= 0:
        raise ValueError("checkpoint model_config.sequence_len must be a positive integer")
    feature_contract = resolve_checkpoint_feature_contract(
        meta, n_sae_features=PositiveInt.parse(lineage.n_sae_features),
        surface="turn ChatCORE cache", source_path=source,
    )
    if checkpoint_contract is not None:
        if (
            feature_contract.feature_mask != checkpoint_contract.feature_mask
            or feature_clusters_checkpoint_metadata(feature_contract.resolved_clusters)
            != feature_clusters_checkpoint_metadata(checkpoint_contract.resolved_clusters)
        ):
            raise ValueError("turn mask/consolidation disagrees with loaded checkpoint feature contract")
        # The loaded model's bound contract owns selection and vocabulary. The
        # metadata reconstruction above is an equality assertion, not a second authority.
        feature_contract = checkpoint_contract
    consolidation = feature_contract.consolidation
    vocab = effective_feature_vocab(feature_contract.universe, lineage.n_sae_features)
    vocab_metadata = None if vocab is None else vocab.metadata_dict()
    if meta.get("feature_vocab") != vocab_metadata:
        raise ValueError("checkpoint feature vocabulary conflicts with canonical mask/clusters")
    stats_path = Path(lineage.statistics_path if statistics_path is None else statistics_path)
    statistics = load_turn_statistics_snapshot(stats_path, expected_raw_sha256=lineage.statistics_sha256)
    expected_stats = {
        "segment_mode": mode, "segment_size": lineage.source_segment_size,
        "pooling_mode": PoolingMode.max_pool.value, "layer": lineage.layer,
        "n_sae_features": lineage.n_sae_features,
    }
    for field, expected in expected_stats.items():
        if getattr(statistics.metadata, field) != expected:
            raise ValueError(f"turn statistics {field} conflicts with checkpoint lineage")
    window = statistics.metadata.context_window
    if type(window) is not int or window <= 0:
        raise ValueError("turn statistics context_window must be a positive integer")
    if not statistics.ignore_bos_features:
        raise ValueError("turn statistics require ignore_bos_features=true")
    if isinstance(lineage.l0, str) and statistics.l0_tier != lineage.l0:
        raise ValueError("turn statistics L0 tier conflicts with checkpoint lineage")
    bos_id = resolve_bos_token_id(extraction_tokenizer)
    bos_len = 0 if bos_id is None else len(extraction_tokenizer.decode([bos_id]))
    if bos_len <= 0:
        raise ValueError("extraction tokenizer must have a visible BOS coordinate prefix")
    normalized_lineage = asdict(lineage)
    for field in ("feature_dir", "statistics_path"):
        normalized_lineage.pop(field)
    provenance.update(training_feature_dir=lineage.feature_dir,
                      training_statistics_path=lineage.statistics_path,
                      statistics_path=str(stats_path))
    # Finalize selection defaults before constructing the frozen contract.
    context = _lineage_selection_context(
        lineage, statistics, feature_contract.feature_mask, consolidation,
    )
    if context.sampling_bundle is None:
        raise ValueError("turn context cache requires a sampling selection bundle")
    assert_no_gradient_ranking(context.sampling_bundle.selection, "turn context cache")
    normalized_lineage["feature_selection"] = asdict(context.sampling_bundle.selection)
    normalized_lineage["negative_selection"] = (
        None if context.negative_bundle is None else asdict(context.negative_bundle.selection)
    )
    normalized_lineage["missing_percentile_policy"] = context.missing_percentile.policy.value
    semantic = {
        "contract_version": TURN_CONTRACT_VERSION,
        "source_segment_mode": mode.value,
        "source_segment_size": lineage.source_segment_size,
        "source_context_window": window,
        "segment_char_coordinate_system": TURN_COORDINATE_SYSTEM,
        "source_bos_char_len": bos_len,
        "pooling": PoolingMode.max_pool.value,
        "context_independent_features": True, "ignore_bos_features": True,
        "statistics_raw_sha256": lineage.statistics_sha256,
        "source_lineage": normalized_lineage,
        "feature_mask": canonical_dict(feature_contract.feature_mask),
        **feature_clusters_checkpoint_metadata(feature_contract.resolved_clusters),
        "feature_vocab": vocab_metadata,
        "decoder_tokenizer_checksum": compute_tokenizer_checksum(tokenizer),
        "extraction_tokenizer_checksum": compute_tokenizer_checksum(extraction_tokenizer),
        "enumeration_semantics": TURN_ENUMERATION_SEMANTICS,
        "selection_semantics": TURN_SELECTION_SEMANTICS,
    }
    return TurnContextContract(
        semantic=json.loads(canonical_turn_payload(semantic)),
        lineage=lineage, statistics=statistics, feature_mask=feature_contract.feature_mask,
        consolidation=consolidation,
        feature_vocab=vocab, model_max_seq_len=model_max_seq_len, provenance=provenance,
    )


def turn_context_identity_fields(contract: TurnContextContract) -> dict[str, Any]:
    """Resolve the existing cache identity namespace from the exact SAE coordinates."""
    lineage = contract.lineage
    release = SAERelease(lineage.release)
    resolved = resolve_sae(
        model=lineage.model, release=release, activation_type=ActivationType(lineage.activation_type),
        layer=lineage.layer, width=SAEWidth(lineage.width), l0=coerce_config_l0(release, lineage.l0),
    )
    return {"model_id": lineage.model, "sae_id": resolved.sae_id,
            "layer": lineage.layer, "pooling": contract.semantic["pooling"]}


def turn_context_contract_hash(contract: TurnContextContract) -> str:
    """Hash every semantic field, including the selection seed and raw statistics digest."""
    return hashlib.sha256(canonical_turn_payload(contract.semantic)).hexdigest()


def validate_turn_context_contract(observed: Mapping[str, Any], expected: TurnContextContract) -> None:
    """Require the complete checkpoint-derived semantic contract, without trusting cache authority."""
    if canonical_turn_payload(observed) != canonical_turn_payload(expected.semantic):
        raise ValueError("turn context contract disagrees with checkpoint-derived expectations")


def build_turn_selection_context(
    *,
    pipeline: FeaturePipelineConfig,
    missing_percentile_policy: str,
    consolidation: FeatureClusterConsolidation | None,
) -> SegmentSelectionContext:
    """Expose the exact live conditioned-SFT selector over the caller's RNG."""

    if not isinstance(pipeline.selection_config, SamplingBundle):
        raise ValueError("checkpoint prompt conditioning must use sampling selection")
    percentile_table = pipeline.selection_config.percentile_table
    policy = parse_missing_percentile_policy(missing_percentile_policy)
    missing = (
        MissingPercentileResolver.error()
        if policy is MissingPercentilePolicy.ERROR
        else MissingPercentileResolver(policy, percentile_table_coverage(percentile_table))
    )
    negative_bundle = pipeline.negative_bundle
    if negative_bundle is not None and consolidation is not None:
        negative_bundle = replace(
            negative_bundle,
            feature_mask=negative_sampling_mask(pipeline, consolidation),
        )
    return SegmentSelectionContext(
        threshold_cfg=None,
        precomputed_thresholds=None,
        sampling_bundle=pipeline.selection_config,
        negative_bundle=negative_bundle,
        boundaries=pipeline.boundaries,
        fp_percentile_table=percentile_table,
        cluster_threshold_percentile_table=(
            percentile_table if consolidation is not None else None
        ),
        feature_mask=pipeline.feature_mask,
        consolidation=consolidation,
        missing_percentile=missing,
    )


def load_turn_statistics_snapshot(
    path: Path,
    *,
    expected_raw_sha256: str,
) -> ExtractionStatistics:
    """Digest-check and parse one immutable in-memory statistics snapshot."""

    if path.suffix != SAFETENSORS_EXT:
        raise ValueError(
            f"Unknown statistics file extension: {path.suffix!r} (expected {SAFETENSORS_EXT})"
        )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Cannot read turn-context training statistics at {path}. "
            "For relocated runs, set stats_path=/local/path/to/feature_statistics.safetensors "
            "to the exact training bytes; the checkpoint lineage SHA-256 must match."
        ) from exc
    actual_raw_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_raw_sha256 != expected_raw_sha256:
        raise ValueError(
            "Turn-context statistics raw SHA-256 does not "
            "match the checkpoint's online-SAE lineage: "
            f"expected={expected_raw_sha256!r}, actual={actual_raw_sha256!r}, path={path}. "
            "Set stats_path to the exact checkpoint training statistics bytes."
        )
    statistics, _percentile_checksum, _feature_ids = load_statistics_and_checksum_from_bytes(
        payload
    )
    return statistics
