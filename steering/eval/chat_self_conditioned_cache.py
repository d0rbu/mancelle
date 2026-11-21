"""Fixed prompt-cache authority recovered from the measured checkpoint copy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from steering.conditional.checkpoint_conditioning import read_checkpoint_conditioning
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.self_conditioned import selection_without_delta
from steering.conditional.feature_mask import canonical_dict
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.conditional.sampling import (
    SamplingSelection,
    parse_feature_selection,
    parse_negative_selection,
)
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_cache_io import CacheIdentity, clustered_context_cache_metadata
from steering.eval.turn_context_contract import canonical_turn_payload
from steering.extraction_identity import ExtractionSourceIdentity
from steering.feature_paths import parse_feature_dir
from steering.refined_types import PositiveInt
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAERelease,
    SAEWidth,
    coerce_config_l0,
    resolve_sae,
)
from steering.segmentation import SegmentMode, compute_tokenizer_checksum
from steering.sft.segment_mode_identity import (
    FIXED_CADENCE_SOURCE_WIDTH,
    FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY,
    resolve_fixed_self_cond_segment_size_rustbpe,
)
from steering.stats_io import load_statistics_and_checksum_from_bytes
from steering.stats_types import ExtractionStatistics

FIXED_PROMPT_SELECTION_VERSION = "fixed_prompt_enumeration_stream_v1"


@dataclass(frozen=True)
class FixedChatCacheContract:
    cadence: int
    identity: CacheIdentity
    config: dict[str, Any]
    metadata: dict[str, Any]
    statistics: ExtractionStatistics
    statistics_path: Path
    statistics_sha256: str
    feature_contract: CheckpointFeatureContract
    source: dict[str, Any]

    def pipeline(self):
        return build_feature_pipeline_from_cfg_block(
            self.config, per_feature_stats=self.statistics.per_feature_stats,
            n_sae_features=PositiveInt.parse(self.config["n_sae_features"]),
            num_buckets=PositiveInt.parse(self.config["num_buckets"]),
            seed=self.config["seed"], feature_mask=self.feature_contract.feature_mask,
        )


def resolve_fixed_chat_cache_contract(meta, tokenizer, feature_contract, *, statistics_path=None, segment_size_assertion=None, loaded_model_sha256=None) -> FixedChatCacheContract:
    cadence = resolve_fixed_self_cond_segment_size_rustbpe(meta, checkpoint_dir=str(meta.get("checkpoint_dir", "<checkpoint>")),
        segment_size_assertion=segment_size_assertion, loaded_model_sha256=loaded_model_sha256)
    if cadence is None:
        raise ValueError("fixed cache preparation requires a fixed checkpoint")
    provenance = meta[FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY]
    if provenance.get("tokenizer_checksum") != compute_tokenizer_checksum(tokenizer):
        raise ValueError("fixed measurement decoder tokenizer checksum mismatch")
    inventory = provenance.get("inventories", {}).get("manifests_and_statistics", [])
    stats_rows = [row for row in inventory if "statistics" in Path(row["path"]).name]
    if len(stats_rows) != 1:
        raise ValueError("fixed measurement must pin exactly one training statistics artifact")
    row = stats_rows[0]
    path = Path(statistics_path or row["path"])
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != row["sha256"]:
        raise ValueError("fixed training statistics hash mismatch")
    stats, _, _ = load_statistics_and_checksum_from_bytes(payload)
    user = meta["user_config"]
    directory = Path(user["feature_dir"])
    if directory.parts[-3:-2] == ("splice",):
        directory = directory.parents[2]
    components = parse_feature_dir(directory)
    sae = user["sae"]
    release = SAERelease(sae["release"])
    resolved = resolve_sae(model=sae["model"], release=release,
        activation_type=ActivationType(sae["activation_type"]), layer=sae["layer"],
        width=SAEWidth(sae["width"]), l0=coerce_config_l0(release, sae.get("l0")))
    if (components.segment_mode is not SegmentMode.fixed_tokens or components.segment_size != FIXED_CADENCE_SOURCE_WIDTH
            or components.model != sae["model"] or components.layer != sae["layer"]
            or components.width.value != sae["width"] or components.activation_type.value != sae["activation_type"]
            or not components.context_independent_features or not components.ignore_bos_features):
        raise ValueError("fixed checkpoint SAE/source geometry mismatch; requires independent s128 no_bos features")
    width = meta.get("n_sae_features", stats.metadata.n_sae_features)
    if type(width) is not int or width != SAE_WIDTH_TO_NUM_FEATURES[SAEWidth(sae["width"])]:
        raise ValueError("fixed checkpoint original SAE width conflicts with SAE config")
    if (stats.metadata.segment_mode is not SegmentMode.fixed_tokens or stats.metadata.segment_size != FIXED_CADENCE_SOURCE_WIDTH
            or stats.metadata.layer != sae["layer"] or stats.metadata.n_sae_features != width
            or stats.metadata.pooling_mode != components.pooling.value or not stats.ignore_bos_features):
        raise ValueError("fixed training statistics source geometry mismatch")
    if isinstance(sae.get("l0"), str) and stats.l0_tier != sae["l0"]:
        raise ValueError("fixed training statistics L0 tier conflicts with checkpoint SAE")
    seed = user.get("seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("fixed checkpoint requires its recorded preselection seed")
    conditioning = read_checkpoint_conditioning(meta)
    positive = parse_feature_selection(conditioning.positive_selection)
    if not isinstance(positive, SamplingSelection):
        raise ValueError("fixed prompt preparation requires sampling selection")
    selection = asdict(selection_without_delta(positive))
    selection.pop("delta_weights", None)
    negative = parse_negative_selection(conditioning.negative_selection)
    config = {
        "feature_selection": selection,
        "negative_selection": None if negative is None else asdict(negative),
        "feature_mask": canonical_dict(feature_contract.feature_mask),
        "feature_clusters": None if feature_contract.resolved_clusters is None else feature_contract.resolved_clusters.canonical_dict(),
        "feature_delta": None, "stats_source_sha256": digest,
        "seed": seed, "n_sae_features": width,
        "num_buckets": PositiveInt.parse(meta.get("num_buckets", user.get("num_buckets"))),
    }
    config = json.loads(canonical_turn_payload(config))
    identity = cache_identity_from_config({"model_id": sae["model"], "sae_id": resolved.sae_id,
        "layer": sae["layer"], "pooling": components.pooling.value}, config, checkpoint_contract=feature_contract)
    source = {"segment_mode": SegmentMode.fixed_tokens.value, "segment_size": FIXED_CADENCE_SOURCE_WIDTH,
        "sae": dict(sae), "pooling": components.pooling.value,
        "extraction_identity": ExtractionSourceIdentity(sae["model"], resolved.release, resolved.sae_id).metadata(),
        "context_independent_features": True, "ignore_bos_features": True}
    metadata = {"model_max_seq_len": meta["model_config"]["sequence_len"],
        "prompt_preselection": {"policy": FIXED_PROMPT_SELECTION_VERSION, "seed": seed},
        "source_geometry": source, "decoder_tokenizer_checksum": compute_tokenizer_checksum(tokenizer)}
    consolidation = feature_contract.consolidation
    if consolidation is not None:
        metadata.update(clustered_context_cache_metadata(feature_mask=feature_contract.feature_mask,
            feature_clusters=consolidation, feature_vocab=effective_feature_vocab(consolidation, width),
            stats_source_sha256=digest))
    return FixedChatCacheContract(cadence, identity, config, metadata, stats, path, digest, feature_contract, source)
