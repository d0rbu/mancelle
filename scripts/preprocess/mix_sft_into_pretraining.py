"""Mix SFT conversation data into pretraining parquets.

Reads pre-converted SFT parquets (from convert_sft_to_parquet), loads their
full rendered ``text`` rows, and mixes them uniformly into pretraining data
shards.

Output parquets have a single ``text`` column, identical to base_data format.
Both the pretraining dataloader and feature extraction pipeline can consume them.

Usage:
    uv run python -m scripts.preprocess.mix_sft_into_pretraining mixtures='[safety]' model=gemma-2-2b
    uv run python -m scripts.preprocess.mix_sft_into_pretraining \\
        mixtures='[safety,capabilities]' model=gemma-2-2b output_name=my_mix
"""

import json
import os
import random
from datetime import UTC, datetime
from pathlib import Path

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.common import get_base_dir
from steering.sft_schema import legacy_sft_columns


def discover_shards(data_dir: Path) -> list[Path]:
    """Find all shard_XXXXX.parquet files in a directory, sorted by name."""
    return sorted(data_dir.glob("shard_*.parquet"))


def _guard_output_paths(
    *,
    output_dir: Path,
    base_data_dir: Path,
    pretraining_shards: list[Path],
    sft_dirs: list[Path],
    sft_shards: list[Path],
) -> None:
    """Refuse to overwrite any pretraining or SFT input path."""
    input_paths = [base_data_dir, *pretraining_shards, *sft_dirs, *sft_shards]
    output_paths = [
        output_dir,
        *(output_dir / shard.name for shard in pretraining_shards),
        output_dir / "mixing_meta.json",
    ]
    resolved_inputs = [(path, path.resolve()) for path in input_paths]
    resolved_outputs = [(path, path.resolve()) for path in output_paths]

    for output_path, resolved_output in resolved_outputs:
        for input_path, resolved_input in resolved_inputs:
            same_file = False
            try:
                same_file = os.path.samefile(output_path, input_path)
            except FileNotFoundError:
                pass
            if resolved_output == resolved_input or same_file:
                raise ValueError(
                    f"Refusing to write {output_path}: output aliases an input "
                    f"path {input_path}"
                )


def render_sft_conversations(
    sft_parquet_dirs: list[Path],
) -> list[str]:
    """Load full-rendered SFT parquet text rows from multiple directories.

    Rendering happens in ``convert_sft_to_parquet.py``; this mixer only
    handles full-rendered text rows.

    Returns:
        List of rendered conversation texts.
    """
    texts: list[str] = []
    for sft_dir in sft_parquet_dirs:
        shards = discover_shards(sft_dir)
        if not shards:
            logger.warning(f"No SFT shards found in {sft_dir}")
            continue

        for shard_path in shards:
            table = pq.read_table(shard_path)
            legacy_cols = legacy_sft_columns(table.column_names)
            if legacy_cols or "text" not in table.column_names:
                raise ValueError(
                    f"SFT shard {shard_path} has legacy schema "
                    f"{table.column_names}; expected a full-rendered 'text' column. "
                    "Regenerate it with scripts.preprocess.convert_sft_to_parquet."
                )
            texts.extend(table.column("text").to_pylist())

    return texts


def distribute_uniformly(items: list, n_buckets: int, rng: random.Random) -> list[list]:
    """Shuffle items globally and distribute uniformly across n_buckets.

    Each bucket gets floor(N/n_buckets) or ceil(N/n_buckets) items.
    Deterministic for a given rng state.
    """
    shuffled = list(items)
    rng.shuffle(shuffled)
    buckets: list[list] = [[] for _ in range(n_buckets)]
    for i, item in enumerate(shuffled):
        buckets[i % n_buckets].append(item)
    return buckets


def mix_sft_into_pretraining(
    mixtures: list[str],
    model_name: str,
    base_data_dir: Path,
    sft_data_dir: Path,
    output_dir: Path,
    seed: int = 0,
) -> dict:
    """Mix SFT conversations into pretraining parquets.

    Loads full-rendered SFT text rows, distributes them uniformly across
    pretraining shards, shuffles within each shard, and writes output parquets
    with a single ``text`` column.

    Args:
        mixtures: SFT mixture names (e.g. ``["safety", "capabilities"]``).
        model_name: Model name for metadata (e.g. ``"gemma-2-2b"``).
        base_data_dir: Directory containing pretraining parquet shards.
        sft_data_dir: Base directory containing SFT parquet subdirectories.
        output_dir: Directory to write mixed parquet shards.
        seed: Random seed for shuffling (default 0).

    Returns:
        Metadata dict with mixing statistics.
    """
    # Discover pretraining shards
    pretraining_shards = discover_shards(base_data_dir)
    if not pretraining_shards:
        raise ValueError(f"No pretraining shards found in {base_data_dir}")
    n_shards = len(pretraining_shards)
    logger.info(f"Found {n_shards} pretraining shards in {base_data_dir}")

    # Validate SFT directories exist
    sft_dirs = [sft_data_dir / m for m in mixtures]
    for d in sft_dirs:
        if not d.exists():
            raise ValueError(f"SFT data directory not found: {d}")

    sft_shards = [shard for sft_dir in sft_dirs for shard in discover_shards(sft_dir)]
    _guard_output_paths(
        output_dir=output_dir,
        base_data_dir=base_data_dir,
        pretraining_shards=pretraining_shards,
        sft_dirs=sft_dirs,
        sft_shards=sft_shards,
    )

    # Load all pre-rendered SFT conversations
    logger.info(f"Loading SFT conversations from {mixtures}")
    sft_texts = render_sft_conversations(sft_dirs)
    logger.info(f"Loaded {len(sft_texts)} SFT conversations")

    # Distribute SFT texts uniformly across shards
    rng = random.Random(seed)
    sft_chunks = distribute_uniformly(sft_texts, n_shards, rng)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mix and write each shard
    total_pretraining_docs = 0
    for i, pretraining_shard_path in enumerate(pretraining_shards):
        table = pq.read_table(pretraining_shard_path)
        pretraining_texts = table.column("text").to_pylist()
        total_pretraining_docs += len(pretraining_texts)

        combined = pretraining_texts + sft_chunks[i]

        # Per-shard deterministic shuffle (seed offset avoids correlation)
        shard_rng = random.Random(seed + i + 1)
        shard_rng.shuffle(combined)

        output_path = output_dir / pretraining_shard_path.name
        out_table = pa.table({"text": combined})
        pq.write_table(out_table, output_path)
        logger.info(
            f"Shard {i + 1}/{n_shards}: {len(pretraining_texts)} pretrain + "
            f"{len(sft_chunks[i])} SFT = {len(combined)} docs -> {output_path.name}"
        )

    # Write metadata
    total_sft = len(sft_texts)
    total_docs = total_pretraining_docs + total_sft
    sft_fraction = total_sft / total_docs if total_docs > 0 else 0.0
    metadata = {
        "mixtures": mixtures,
        "model": model_name,
        "seed": seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "total_pretraining_docs": total_pretraining_docs,
        "total_sft_docs": total_sft,
        "total_docs": total_docs,
        "sft_fraction": sft_fraction,
        "n_shards": n_shards,
        "source_shard_names": [p.name for p in pretraining_shards],
    }
    meta_path = output_dir / "mixing_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        f"Mixing complete: {total_pretraining_docs} pretrain + {total_sft} SFT "
        f"= {total_docs} docs ({sft_fraction:.2%} SFT)"
    )
    return metadata


def _default_output_name(mixtures: list[str]) -> str:
    """Generate default output directory name from sorted mixture names."""
    return "+".join(sorted(mixtures))


@hydra.main(
    config_path="../../configs/data",
    config_name="mix_sft_into_pretraining",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """CLI entry point for mixing SFT data into pretraining parquets."""
    raw_mixtures = OmegaConf.to_container(cfg.mixtures, resolve=True)
    if not isinstance(raw_mixtures, list):
        raise ValueError("mixtures must be a list of SFT mixture names")
    mixtures = [str(mixture) for mixture in raw_mixtures]
    model = cfg.model
    seed = cfg.seed

    base_dir = get_base_dir()
    base_data_dir = (
        Path(cfg.base_data_dir) if cfg.base_data_dir else Path(base_dir) / "base_data"
    )
    sft_data_dir = (
        Path(cfg.sft_data_dir) if cfg.sft_data_dir else Path(base_dir) / "sft_data"
    )

    output_name = cfg.output_name or _default_output_name(mixtures)
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir
        else Path(base_dir) / "mixed_data" / output_name
    )

    logger.info(f"Mixing SFT data from {mixtures} into pretraining data")
    logger.info(f"Model: {model}, Seed: {seed}")
    logger.info(f"Output: {output_dir}")

    mix_sft_into_pretraining(
        mixtures=mixtures,
        model_name=model,
        base_data_dir=base_data_dir,
        sft_data_dir=sft_data_dir,
        output_dir=output_dir,
        seed=seed,
    )


if __name__ == "__main__":
    main()
