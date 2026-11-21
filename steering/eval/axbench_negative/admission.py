"""Weight-free preflight of encoder-decoder prompt-conditioning extraction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformer_lens.loading_from_pretrained import get_pretrained_model_config

from nanochat.checkpoint_manager import _extract_encdec_fields, _iter_ancestor_metas
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossMode,
    feature_prediction_config_from_metadata,
)
from steering.eval.axbench_negative.qc import tokenizer_sha256
from steering.eval.axbench_negative.source import file_sha256
from steering.eval.axbench_upstream.methods import load_method_setup
from steering.eval.context_enumeration import context_text_for_extraction
from steering.models import resolve_tokenizer
from steering.segmentation import SegmentMode, truncate_to_training_window
from steering.sft.turn_spans import rendered_text_and_turn_char_starts


def conditioning_identity(
    setup: Any,
    extraction_tokenizer: Any,
    model_context_window: int,
    *,
    statistics_path: Path,
    pretrain_meta_path: Path | None,
) -> dict[str, Any]:
    """Shared preflight/runtime identity for the target's conditioning contract."""
    window = int(setup.prompt_context_window)
    if type(model_context_window) is not int or not 0 < window <= model_context_window:
        raise ValueError("target conditioning window exceeds its extraction model context")
    return {
        "format": "negative-target-conditioning-admission-v1",
        "sae_model": setup.sae_model,
        "sae_release": setup.resolved_sae.release,
        "sae_id": setup.resolved_sae.sae_id,
        "hook_name": setup.resolved_sae.hook_name,
        "layer": int(setup.sae_layer),
        "n_sae_features": int(setup.n_sae_features),
        "segment_mode": setup.segment_mode.value,
        "segment_size": int(setup.prompt_segment_size),
        "context_window": window,
        "model_context_window": model_context_window,
        "extraction_tokenizer_sha256": tokenizer_sha256(extraction_tokenizer),
        "statistics_sha256": file_sha256(statistics_path),
        "pretrain_meta_sha256": None
        if pretrain_meta_path is None
        else file_sha256(pretrain_meta_path),
    }


@dataclass(frozen=True)
class ConditioningAdmission:
    setup: Any
    tokenizer: Any
    model_context_window: int
    identity: dict[str, Any]

    def prompt_token_count(self, native_ids: Sequence[int], native_tokenizer: Any) -> int:
        """Count the exact production conditioning input; never admit truncation."""
        if self.setup.segment_mode is SegmentMode.fixed_tokens:
            text = context_text_for_extraction(native_tokenizer, native_ids)
        else:
            text, _starts = rendered_text_and_turn_char_starts(native_ids, native_tokenizer)
        _kept, full_count = truncate_to_training_window(
            text,
            self.tokenizer,
            int(self.setup.prompt_context_window),
            self.model_context_window,
        )
        return full_count


def load_conditioning_admission(
    meta_path: Path,
    statistics_path: Path,
    pretrain_meta_path: Path | None = None,
) -> ConditioningAdmission:
    """Load checkpoint policy, statistics, tokenizer and runtime config; no weights."""
    before = (
        file_sha256(meta_path),
        file_sha256(statistics_path),
        None if pretrain_meta_path is None else file_sha256(pretrain_meta_path),
    )
    # Match the actual checkpoint reader before setup, tokenizer/model loading,
    # and any paid screening. Legacy head schemas require explicit conversion.
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_prediction = feature_prediction_config_from_metadata(metadata)
    if feature_prediction is None:
        # Match build_encdec_model's closest/latest ancestor recovery, including
        # validation of both the ancestor's mirrors and the recovered child's.
        for _path, ancestor in _iter_ancestor_metas(str(meta_path.parent)):
            fields = _extract_encdec_fields(ancestor)
            if "feature_prediction" in fields:
                feature_prediction_config_from_metadata(ancestor)
                metadata.setdefault("user_config", {})["feature_prediction"] = fields["feature_prediction"]
                feature_prediction = feature_prediction_config_from_metadata(metadata)
                break
    if feature_prediction is None:
        raise ValueError("negative SelfConditioning requires checkpoint feature_prediction metadata")
    if feature_prediction.loss_mode is not FeaturePredLossMode.MAX_POOL:
        raise ValueError("negative SelfConditioning requires feature_prediction.loss_mode=max_pool")
    setup = load_method_setup(meta_path, statistics_path, pretrain_meta_path=pretrain_meta_path)
    tokenizer = resolve_tokenizer(setup.sae_model)
    # Both extraction loaders expose this TransformerLens config, including
    # the weight-free nnterp wrapper. Its n_ctx can be smaller than the HF
    # architecture maximum (Gemma3-4b: 8192 versus 131072).
    model_context = int(get_pretrained_model_config(setup.sae_model, device="cpu").n_ctx)
    identity = conditioning_identity(
        setup,
        tokenizer,
        model_context,
        statistics_path=statistics_path,
        pretrain_meta_path=pretrain_meta_path,
    )
    after = (
        file_sha256(meta_path),
        identity["statistics_sha256"],
        identity["pretrain_meta_sha256"],
    )
    if before != after:
        raise ValueError("target conditioning metadata/statistics changed during admission")
    return ConditioningAdmission(setup, tokenizer, model_context, identity)
