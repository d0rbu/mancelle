"""Property-style fuzz: 200 iterations, fixed seed.

Random pipeline configs; assert hash stable across redundant constructions
and mutates whenever any field mutates."""

from __future__ import annotations

import random

from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)


def _random_config(rng: random.Random) -> dict:
    return {
        "positive_selection": {
            "threshold_type": rng.choice([
                "feature_percentile_segment_topk", "segment_topk", "feature_percentile",
            ]),
            "threshold": round(rng.uniform(100.0, 4000.0), 1),
        },
        "negative_selection": (
            None if rng.random() < 0.5 else
            {"distribution": rng.choice(["geometric", "normal"]),
             "mean": round(rng.uniform(1.0, 20.0), 1),
             "std": round(rng.uniform(1.0, 20.0), 1)}
        ),
        "num_buckets": rng.choice([4, 8, 10, 16]),
        "stats_source_sha256": f"{rng.getrandbits(256):064x}",
    }


def test_property_hash_sensitivity():
    rng = random.Random(20260428)
    seen_hashes: set[str] = set()
    for trial in range(200):
        cfg = _random_config(rng)

        h1 = compute_canonical_feature_selection_hash(**cfg)
        h2 = compute_canonical_feature_selection_hash(**cfg)
        assert h1 == h2, f"trial {trial}: redundant constructions produced different hashes"

        for key in ("num_buckets", "stats_source_sha256"):
            mutated = dict(cfg)
            if key == "num_buckets":
                mutated[key] = cfg[key] + 1
            elif key == "stats_source_sha256":
                mutated[key] = "deadbeef" + cfg[key][8:]
            h_mut = compute_canonical_feature_selection_hash(**mutated)
            assert h1 != h_mut, (
                f"trial {trial}: mutating {key} from {cfg[key]!r} to {mutated[key]!r} did not change hash"
            )

        seen_hashes.add(h1)
    assert len(seen_hashes) > 180, (
        f"expected >180 distinct hashes across 200 random configs, got {len(seen_hashes)} (collision suspect)"
    )
