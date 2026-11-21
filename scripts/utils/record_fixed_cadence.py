"""Publish a measured fixed-cadence eval leaf with unchanged model weights."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig

from nanochat.tokenizer import RustBPETokenizer
from steering._common.file_inventory import file_inventory, file_signature
from steering.conditional.encoder_decoder.fixed_cadence import (
    FixedCadenceArtifactError,
    measure_fixed_self_cond_segment_size_rustbpe,
)
from steering.sft.segment_mode_identity import (
    FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY,
    FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY,
    resolve_fixed_self_cond_segment_size_rustbpe,
)


def _resolve_published(path: Path, expected: dict[str, Any], model_path: Path, assertion: Any) -> int:
    recorded = json.loads(path.read_text())
    if recorded != expected:
        raise FixedCadenceArtifactError(f"refusing contradictory measurement at {path}")
    linked = path.parent / model_path.name
    if not linked.is_symlink() or linked.resolve() != model_path.resolve():
        raise FixedCadenceArtifactError(f"eval weights must link unchanged original model: {linked}")
    cadence = resolve_fixed_self_cond_segment_size_rustbpe(
        recorded, checkpoint_dir=str(path.parent), segment_size_assertion=assertion,
        loaded_model_sha256=file_inventory([linked])[0]["sha256"],
    )
    if cadence is None:
        raise FixedCadenceArtifactError("published checkpoint did not resolve a fixed cadence")
    return cadence


def run(cfg: DictConfig) -> Path:
    """Measure before publication; existing leaves must agree exactly."""
    meta_path = Path(str(cfg.checkpoint_meta_path)).resolve()
    match = re.fullmatch(r"meta_(\d+)\.json", meta_path.name)
    if match is None:
        raise FixedCadenceArtifactError("checkpoint_meta_path must name meta_<step>.json")
    model_path = meta_path.with_name(f"model_{match[1]}.pt")
    initial = {p: file_signature(p) for p in (meta_path, model_path)}
    original_identity = file_inventory([meta_path, model_path])
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict) or meta.get("step") != int(match[1]):
        raise FixedCadenceArtifactError("checkpoint metadata step disagrees with selected filename")
    tokenizer_dir = Path(str(cfg.tokenizer_dir)).resolve()
    tokenizer_paths = sorted(p for p in tokenizer_dir.rglob("*") if p.is_file())
    if not tokenizer_paths:
        raise FixedCadenceArtifactError("decoder tokenizer directory is empty")
    tokenizer_states = {p: file_signature(p) for p in tokenizer_paths}
    tokenizer_identity = file_inventory(tokenizer_paths)
    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_dir))
    measurement = measure_fixed_self_cond_segment_size_rustbpe(
        checkpoint_meta=meta, feature_dir=str(cfg.feature_dir), parquet_dir=str(cfg.parquet_dir),
        tokenizer=tokenizer, max_rows=cfg.max_rows,
    )
    recorded = meta.get(FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY)
    if FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY in meta and (
        type(recorded) is not int or recorded != measurement.cadence
    ):
        raise FixedCadenceArtifactError("original metadata contains a contradictory fixed cadence")
    provenance = {
        **measurement.provenance,
        "original_checkpoint": original_identity,
        "decoder_tokenizer_inventory": tokenizer_identity,
        "measurement": measurement.metadata_dict(),
    }
    expected = {**meta, FIXED_SELF_COND_SEGMENT_SIZE_RUSTBPE_META_KEY: int(measurement.cadence),
                FIXED_SELF_COND_CADENCE_PROVENANCE_META_KEY: provenance}
    assertion = cfg.get("fixed_segment_size_assertion")
    if any(file_signature(p) != state for p, state in (initial | tokenizer_states).items()):
        raise FixedCadenceArtifactError("checkpoint/model/tokenizer changed during measurement")
    if sorted(p for p in tokenizer_dir.rglob("*") if p.is_file()) != tokenizer_paths:
        raise FixedCadenceArtifactError("tokenizer inventory changed during measurement")
    measurement.assert_artifacts_current()
    output = Path(str(cfg.output_dir)).absolute()
    destination = output / meta_path.name
    if output.exists():
        if not destination.is_file():
            raise FixedCadenceArtifactError(f"refusing incomplete/contradictory existing eval leaf {output}")
        cadence = _resolve_published(destination, expected, model_path, assertion)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
            staged = Path(temporary) / "leaf"
            staged.mkdir()
            (staged / model_path.name).symlink_to(model_path)
            staged_meta = staged / meta_path.name
            with staged_meta.open("x") as stream:
                json.dump(expected, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            cadence = _resolve_published(staged_meta, expected, model_path, assertion)
            # rename fails if a competing nonempty publication already owns the leaf.
            os.rename(staged, output)
    logger.info("Recorded fixed cadence K={} from {} windows in {} rows: {}", cadence,
                measurement.windows, measurement.sampled_rows, destination)
    return destination


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="record_fixed_cadence")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
