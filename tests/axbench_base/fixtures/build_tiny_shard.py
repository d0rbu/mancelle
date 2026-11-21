"""Build a tiny parquet + feature shard pair for seed_pools tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from steering.feature_io import save_features

TINY_TEXTS = [
    "The cat sat on the mat lazily watching the birds outside the window "
    "as the afternoon light drifted through the half-open curtains and "
    "cast long stripes across the wooden floorboards that had been polished "
    "to a soft amber sheen by decades of steady foot traffic and careful care.",
    "Dogs bark loudly at squirrels climbing trees in the backyard near dusk "
    "while the last of the daylight stretches between the fenceposts and the "
    "neighbors' porch lights flicker on one by one, dotting the street with "
    "warm yellow points that mark the slow transition from evening to night.",
    "Python code often uses indentation to delineate blocks of execution paths "
    "and this structural choice is one of the language's most distinctive "
    "design decisions, forcing programmers to adopt a visually consistent "
    "style that tends to make even unfamiliar codebases unusually easy to read.",
    "Quantum entanglement describes correlations between particles separated in space "
    "that cannot be reproduced by any classical theory of local hidden variables, "
    "a fact confirmed repeatedly by experiments beginning with Bell's inequality "
    "tests in the 1970s and continuing through modern loophole-free demonstrations.",
    "The ancient library held scrolls of forgotten knowledge and arcane wisdom "
    "gathered by generations of wandering scholars who braved storms, bandits, "
    "and language barriers to copy rare manuscripts by candlelight and carry the "
    "fragile bundles home wrapped in oiled cloth against the damp night air.",
    "Mountain peaks catch the first rays of dawn before the valleys below see light "
    "and the alpine silence at that hour is broken only by the distant cries of "
    "hawks wheeling against the pink sky as the warming air begins to lift the "
    "mist off the dark lakes tucked between the granite shoulders of the range.",
    "Ocean waves carve slowly through the hardest stone over countless millennia "
    "and the coastlines we take for granted today are merely a single frame in a "
    "vast geological film, one that will eventually render every familiar headland "
    "and cove unrecognizable to anyone who knew them during our brief human era.",
    "Distant galaxies drift apart as the fabric of spacetime itself expands outward "
    "in all directions, carrying their embedded stars and dark matter halos along "
    "for the ride and gradually shifting the light they emit toward longer and "
    "longer wavelengths until only the most patient telescopes can still hear them.",
    "Chess masters study classical openings for years before playing at high level "
    "and the hours spent memorizing lines and analyzing critical positions form "
    "only a fraction of the preparation — the rest is devoted to endgame technique, "
    "tactical puzzles, and the patient cultivation of positional intuition.",
    "café résumé naïve Buddhist texts were written in ancient scripts long ago "
    "and preserved across centuries by copyists working in quiet monastery cells "
    "where the steady scratch of reed pens was the only sound from sunrise until "
    "the small oil lamps were trimmed and snuffed out at the end of the working day.",
]


def build_tiny_shard(out_dir: Path, shard_idx: int = 0) -> tuple[Path, Path]:
    """Build a (parquet, feature shard) pair for one validation shard.

    The parquet has 10 text rows; the feature shard has one segment per
    doc covering the full text range, with 2 active features per segment.

    Returns:
        Tuple of ``(parquet_path, feature_shard_path)`` where
        ``feature_shard_path`` has the ``.safetensors.zst`` extension that
        :func:`steering.feature_io.save_features` forces.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(
        pa.Table.from_pydict({"text": TINY_TEXTS}),
        parquet_path,
    )

    n = len(TINY_TEXTS)
    offsets = t.tensor([i * 2 for i in range(n + 1)], dtype=t.int64)
    feature_ids = t.tensor(
        [fid for i in range(n) for fid in [i % 5, (i % 5) + 10]],
        dtype=t.int32,
    )
    feature_mags = t.tensor(
        [mag for _ in range(n) for mag in [2.0, 1.0]],
        dtype=t.float32,
    )
    doc_ids = t.arange(n, dtype=t.int32)
    seg_indices = t.zeros(n, dtype=t.int32)
    seg_positions = t.zeros(n, dtype=t.int32)
    doc_token_counts = t.full((n,), 20, dtype=t.int32)
    seg_starts = t.zeros(n, dtype=t.int32)
    seg_ends = t.tensor([len(text) for text in TINY_TEXTS], dtype=t.int32)

    data = {
        "offsets": offsets,
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "doc_ids": doc_ids,
        "segment_indices": seg_indices,
        "segment_positions": seg_positions,
        "doc_token_counts": doc_token_counts,
        "segment_char_starts": seg_starts,
        "segment_char_ends": seg_ends,
    }
    feature_shard_path = save_features(
        data, out_dir / f"shard_{shard_idx:05d}_features",
    )
    return parquet_path, feature_shard_path
