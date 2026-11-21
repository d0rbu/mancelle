"""Sample random segments with full context for fine-grained interpretability.

Loads extracted feature shards (CSR format), reconstructs document text from
the original parquet source, and writes a YAML file containing:
  - Context: all tokens leading up to the segment within the context window
  - Segment: the target segment text
  - Next segment: one segment further for continuity
  - Features: top-k SAE features ordered by magnitude (descending)

Usage:
    uv run python -m viz.sample_segments
    uv run python -m viz.sample_segments n_samples=50 top_k=100
    uv run python -m viz.sample_segments shard_index=1
"""

import random
from datetime import UTC, datetime
from pathlib import Path

import hydra
import pyarrow.parquet as pq
import torch as t
import yaml
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoTokenizer

from nanochat.dataset import list_parquet_files
from steering.context_defaults import DEFAULT_CONTEXT_WINDOW
from steering.extract_features import ShardData
from steering.feature_io import glob_features, load_features
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, segment_tokens
from steering.tokenizer_protocols import require_text_tokenizer


def load_shard(shard_path: str) -> ShardData:
    """Load a feature shard from disk."""
    data = load_features(shard_path)
    return ShardData.from_dict(data)


def find_eligible_segments(shard: ShardData) -> list[int]:
    """Find segment indices that have at least one segment after them.

    A segment is eligible if it is not the last segment in its document,
    ensuring we can include "one segment further" in the output.

    Returns:
        List of global segment indices (into shard arrays).
    """
    doc_ids = shard.doc_ids
    seg_indices = shard.segment_indices

    # Find max segment index per doc using scatter_reduce
    n_docs = shard.n_docs
    max_seg_per_doc = t.zeros(n_docs, dtype=t.int32)
    max_seg_per_doc.scatter_reduce_(
        0,
        doc_ids.long(),
        seg_indices,
        reduce="amax",
        include_self=False,
    )

    # Eligible = segment_index < max_segment_index for its doc
    max_for_each = max_seg_per_doc[doc_ids.long()]
    eligible_mask = seg_indices < max_for_each
    return eligible_mask.nonzero(as_tuple=True)[0].tolist()


def read_doc_texts(parquet_path: str, doc_ids: set[int]) -> dict[int, str]:
    """Read specific documents from a parquet file by sequential index.

    doc_ids correspond to the sequential order of documents in the parquet
    (doc_id 0 = first row, doc_id 1 = second row, etc.).
    """
    pf = pq.ParquetFile(parquet_path)
    texts: dict[int, str] = {}
    row_offset = 0

    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=["text"])
        rg_size = rg.num_rows

        needed_in_rg = {
            did for did in doc_ids if row_offset <= did < row_offset + rg_size
        }

        if needed_in_rg:
            text_col = rg.column("text").to_pylist()
            for did in needed_in_rg:
                texts[did] = text_col[did - row_offset]

        row_offset += rg_size

        if len(texts) == len(doc_ids):
            break

    return texts


def sample_segments(
    shard_path: str,
    parquet_path: str,
    n_samples: int = 100,
    top_k: int = 50,
    seed: int = 42,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> dict:
    """Sample random segments and build interpretability data.

    For each sampled segment, reconstructs:
    - context: decoded text of all tokens before the segment in the kept window
    - segment: decoded text of the segment itself
    - next_segment: decoded text of the following segment
    - features: top-k (feature_id, magnitude) pairs, descending by magnitude

    Args:
        shard_path: Path to feature shard .safetensors.zst file.
        parquet_path: Path to corresponding parquet file.
        n_samples: Number of segments to sample.
        top_k: Number of top features per segment to include.
        seed: Random seed for reproducibility.
        context_window: Context window size (must match extraction config).

    Returns:
        Dict ready for YAML serialization.
    """
    logger.info(f"Loading shard: {shard_path}")
    shard = load_shard(shard_path)

    logger.info("Finding eligible segments (not last in doc)...")
    eligible = find_eligible_segments(shard)
    logger.info(
        f"Found {len(eligible)} eligible segments out of {shard.n_segments} total"
    )

    rng = random.Random(seed)
    n_samples = min(n_samples, len(eligible))
    sampled_global_indices = sorted(rng.sample(eligible, n_samples))

    needed_doc_ids = {int(shard.doc_ids[i].item()) for i in sampled_global_indices}
    logger.info(f"Need {len(needed_doc_ids)} unique documents for {n_samples} samples")

    logger.info("Loading GPT-2 tokenizer...")
    tokenizer = require_text_tokenizer(
        AutoTokenizer.from_pretrained("gpt2"),
        surface="sample_segments",
    )

    logger.info(f"Reading documents from {parquet_path}...")
    doc_texts = read_doc_texts(parquet_path, needed_doc_ids)

    segment_mode = SegmentMode(shard.segment_mode)
    segment_size = PositiveInt.parse(shard.segment_size)

    samples = []
    for seg_global_idx in sampled_global_indices:
        doc_id = int(shard.doc_ids[seg_global_idx].item())
        seg_idx = int(shard.segment_indices[seg_global_idx].item())
        seg_pos = int(shard.segment_positions[seg_global_idx].item())

        text = doc_texts[doc_id]
        token_ids = tokenizer.encode(text)
        # Head truncation: keep first context_window tokens (matches extraction)
        kept_ids = token_ids[:context_window]

        segments = segment_tokens(
            kept_ids,
            segment_mode,
            segment_size,
        )

        if seg_idx >= len(segments) or seg_idx + 1 >= len(segments):
            logger.warning(
                f"seg_idx={seg_idx} out of range (n_segments={len(segments)}), skipping"
            )
            continue

        seg_start, seg_end = segments[seg_idx]
        next_start, next_end = segments[seg_idx + 1]

        context_text = tokenizer.decode(kept_ids[:seg_start]) if seg_start > 0 else ""
        segment_text = tokenizer.decode(kept_ids[seg_start:seg_end])
        next_segment_text = tokenizer.decode(kept_ids[next_start:next_end])

        # Extract features from CSR (sorted ascending by ID)
        feat_start = int(shard.offsets[seg_global_idx].item())
        feat_end = int(shard.offsets[seg_global_idx + 1].item())
        feat_ids = shard.feature_ids[feat_start:feat_end]
        feat_mags = shard.feature_mags[feat_start:feat_end]
        n_active = len(feat_ids)

        # Sort by magnitude descending to select top-k
        sort_idx = feat_mags.argsort(descending=True)
        sorted_ids = feat_ids[sort_idx][:top_k].tolist()
        sorted_mags = feat_mags[sort_idx][:top_k].tolist()

        top_features = [
            {"id": int(fid), "magnitude": round(float(mag), 4)}
            for fid, mag in zip(sorted_ids, sorted_mags, strict=True)
        ]

        samples.append(
            {
                "global_segment_id": int(seg_global_idx),
                "doc_id": doc_id,
                "segment_index": seg_idx,
                "n_segments_in_doc": len(segments),
                "segment_token_position": seg_pos,
                "context_tokens": seg_start,
                "segment_tokens": seg_end - seg_start,
                "context": context_text,
                "segment": segment_text,
                "next_segment": next_segment_text,
                "n_active_features": n_active,
                "features": top_features,
            }
        )

    output = {
        "metadata": {
            "shard": Path(shard_path).stem,
            "parquet": Path(parquet_path).name,
            "n_samples": len(samples),
            "seed": seed,
            "segment_mode": shard.segment_mode,
            "segment_size": segment_size,
            "context_window": context_window,
            "pooling": shard.pooling,
            "layer": shard.layer,
            "top_k_features": top_k,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "samples": samples,
    }

    return output


def _multiline_str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Use literal block style for multi-line strings in YAML output."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def write_yaml(data: dict, output_path: Path) -> None:
    """Write data to YAML with readable multi-line string formatting."""
    dumper = yaml.Dumper
    dumper.add_representer(str, _multiline_str_representer)

    with open(output_path, "w") as f:
        yaml.dump(
            data,
            f,
            Dumper=dumper,
            default_flow_style=False,
            sort_keys=False,
            width=120,
            allow_unicode=True,
        )


@hydra.main(
    config_path="../configs/extraction",
    config_name="sample_segments",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Sample segments and write interpretability YAML."""
    feature_dir = Path(cfg.feature_dir)
    shard_files = glob_features(feature_dir)
    if not shard_files:
        logger.error(f"No feature shards found in {feature_dir}")
        return

    shard_idx = cfg.shard_index
    if shard_idx >= len(shard_files):
        logger.error(
            f"Shard index {shard_idx} out of range (found {len(shard_files)} shards)"
        )
        return
    shard_path = str(shard_files[shard_idx])

    parquet_paths = list_parquet_files(cfg.data_dir if cfg.data_dir else None)
    if shard_idx >= len(parquet_paths):
        logger.error(f"Shard index {shard_idx} has no corresponding parquet file")
        return
    parquet_path = parquet_paths[shard_idx]

    logger.info(f"Shard: {shard_path}")
    logger.info(f"Parquet: {parquet_path}")

    result = sample_segments(
        shard_path=shard_path,
        parquet_path=parquet_path,
        n_samples=cfg.n_samples,
        top_k=cfg.top_k,
        seed=cfg.seed,
        context_window=cfg.context_window,
    )

    output_path = Path(cfg.output_dir) / cfg.output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(result, output_path)
    logger.info(f"Wrote {len(result['samples'])} samples to {output_path}")


if __name__ == "__main__":
    main()
