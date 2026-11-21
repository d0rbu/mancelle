"""Build Campaign 2's cadence-bound view of an ordinary SFT checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from nanochat.tokenizer import get_tokenizer
from scripts.train.conditional_encdec_selfcond import _turn_cadence_renderer
from scripts.train.conditional_encdec_sft import (
    _assert_sft_mixture_fingerprint_matches_task,
    _run_sft_content_alignment_probe,
)
from scripts.utils.campaign2 import inventory, load_strict_json_object, nonempty
from steering.conditional.encoder_decoder.sft_features import (
    assert_sft_source_bos_char_len_matches,
    load_sft_features,
)
from steering.conditional.encoder_decoder.turn_cadence import (
    measure_turn_self_cond_segment_size_rustbpe,
)
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft.segment_mode_identity import (
    TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    resolve_checkpoint_sft_segment_mode,
)
from steering.sft_schema import SFTSplit
from tasks.mixture_names import SFTMixture
from tasks.mixtures import build_task_mixtures, steered_mixture_overrides


@hydra.main(version_base=None, config_path="../../configs/utils", config_name="campaign2_eval_view")
def main(cfg: DictConfig) -> None:
    metadata = nonempty(Path(cfg.metadata)).resolve()
    meta = load_strict_json_object(metadata)
    if resolve_checkpoint_sft_segment_mode(meta, checkpoint_dir=str(metadata.parent)) is not SegmentMode.turn_delimited_fixed_tokens:
        raise ValueError("evaluation view requires a turn-delimited SFT checkpoint")
    if meta.get("sft_loss_mask") != LossMaskMode.COMPLETION_ONLY.value:
        raise ValueError("evaluation view requires completion_only SFT")
    if any(key in meta for key in ("selfcond", "selfcond_data", "grpo")):
        raise ValueError("evaluation view requires ordinary SFT, not post-training")
    output = Path(cfg.output_dir)
    if output.exists():
        raise FileExistsError(f"evaluation view already exists: {output}")
    user = meta["user_config"]
    if user["segment_size"] != 32:
        raise ValueError("Campaign 2 requires source segment_size=32")
    inventory(Path(cfg.parquet_dir), Path(cfg.feature_dir))
    store = load_sft_features(cfg.feature_dir, cfg.parquet_dir)
    if store.segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
        raise ValueError("evaluation cadence requires turn-delimited feature artifacts")
    mixture = SFTMixture(meta["dataset"])
    task, _ = build_task_mixtures(mixture, **steered_mixture_overrides(mixture, cfg.steered_jsonl))
    _assert_sft_mixture_fingerprint_matches_task(
        store=store, task=task, mixture_name=mixture.value, expected_split=SFTSplit.TRAIN,
        feature_dir=cfg.feature_dir, parquet_dir=cfg.parquet_dir, alignment_probe_samples=8,
    )
    tokenizer = get_tokenizer()
    _run_sft_content_alignment_probe(store=store, task=task, tokenizer=tokenizer,
                                    alignment_probe_samples=8, probe_indices=sorted(store))
    assert_sft_source_bos_char_len_matches(cfg.feature_dir, cfg.parquet_dir,
        configured=int(user["feature_source_bos_char_len"]), source=SFTSplit.TRAIN.value)
    measurement = measure_turn_self_cond_segment_size_rustbpe(
        segments_by_document=store, rendered=_turn_cadence_renderer(task, tokenizer), tokenizer=tokenizer,
        segment_size=32, row_capacity=int(meta["model_config"]["sequence_len"]) + 1,
        loss_mask_mode=LossMaskMode.COMPLETION_ONLY, min_loss_tokens=int(user["min_loss_tokens"]),
        feature_source_bos_char_len=int(user["feature_source_bos_char_len"]),
        feature_dir=cfg.feature_dir, parquet_dir=cfg.parquet_dir,
    )
    cadence = int(measurement.cadence)
    if TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY in meta and meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] != cadence:
        raise ValueError("recorded cadence disagrees with current SFT artifacts")
    model = nonempty(metadata.parent / f"model_{meta['step']:06d}.pt")
    meta[TURN_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY] = cadence
    output.mkdir(parents=True)
    (output / model.name).symlink_to(model)
    (output / metadata.name).write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    evidence = {
        "original_metadata": str(metadata), "original_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "feature_dir": cfg.feature_dir, "parquet_dir": cfg.parquet_dir,
        "cadence": cadence, "documents": int(measurement.documents), "windows": int(measurement.windows),
        "mean": str(measurement.mean), "minimum": int(measurement.minimum), "maximum": int(measurement.maximum),
    }
    (output / "campaign2_cadence.json").write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
