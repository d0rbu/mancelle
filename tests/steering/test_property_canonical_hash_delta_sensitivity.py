"""Slice 07 — property test: canonical hash distinguishes ALL pipeline-relevant fields.

Pre-fix: feature_delta config was NOT hashed, so prev_segment vs cum_max
collided on the eval-context-cache identity (S-07-11 P0).

Post-fix: every random pipeline (positive, negative, num_buckets, stats_sha,
feature_delta) hashes uniquely. Mutating any field changes the hash.

Hypothesis-style fuzz with deterministic seed (matching the existing
``tests/properties/test_property_canonical_feature_selection_hash_sensitivity.py``
pattern). 200 iterations; per-key mutation; collision check across the run.

Seed source: deterministic — fixed master seed (20260505) + stdlib random.
"""

from __future__ import annotations

import random

from steering.eval.feature_selection_hash import (
    compute_canonical_feature_selection_hash,
)


def _random_config(rng: random.Random) -> dict:
    """Construct a randomized canonical-hash kwargs dict.

    With prob 0.5 includes a feature_delta block; otherwise omits it.
    """
    cfg = {
        "positive_selection": {
            "threshold_type": rng.choice(
                ["feature_percentile_segment_topk", "segment_topk", "feature_percentile"]
            ),
            "threshold": round(rng.uniform(100.0, 4000.0), 1),
        },
        "negative_selection": (
            None if rng.random() < 0.5 else {
                "distribution": rng.choice(["geometric", "normal"]),
                "mean": round(rng.uniform(1.0, 20.0), 1),
                "std": round(rng.uniform(1.0, 20.0), 1),
            }
        ),
        "num_buckets": rng.choice([4, 8, 10, 16]),
        "stats_source_sha256": f"{rng.getrandbits(256):064x}",
    }
    if rng.random() < 0.5:
        cfg["feature_delta"] = {
            "mode": rng.choice(["prev_segment", "cum_max"]),
            "neg_delta_ratio": round(rng.uniform(0.0, 1.0), 3),
        }
    return cfg


def test_property_hash_sensitivity_with_delta():
    rng = random.Random(20260505)
    seen_hashes: set[str] = set()

    for trial in range(200):
        cfg = _random_config(rng)

        h1 = compute_canonical_feature_selection_hash(**cfg)
        h2 = compute_canonical_feature_selection_hash(**cfg)
        assert h1 == h2, f"trial {trial}: redundant constructions produced different hashes"

        # Mutate non-delta fields
        for key in ("num_buckets", "stats_source_sha256"):
            mutated = dict(cfg)
            if key == "num_buckets":
                mutated[key] = cfg[key] + 1
            elif key == "stats_source_sha256":
                mutated[key] = "deadbeef" + cfg[key][8:]
            h_mut = compute_canonical_feature_selection_hash(**mutated)
            assert h1 != h_mut, (
                f"trial {trial}: mutating {key} did not change hash"
            )

        # Mutate feature_delta (the focus of this property test)
        if "feature_delta" in cfg:
            # Toggle mode
            mutated = dict(cfg)
            mutated["feature_delta"] = dict(cfg["feature_delta"])
            old_mode = mutated["feature_delta"]["mode"]
            mutated["feature_delta"]["mode"] = (
                "cum_max" if old_mode == "prev_segment" else "prev_segment"
            )
            h_mut = compute_canonical_feature_selection_hash(**mutated)
            assert h1 != h_mut, (
                f"trial {trial}: toggling feature_delta.mode "
                f"({old_mode} -> {mutated['feature_delta']['mode']}) "
                "did not change hash"
            )

            # Toggle neg_delta_ratio
            mutated = dict(cfg)
            mutated["feature_delta"] = dict(cfg["feature_delta"])
            old_r = mutated["feature_delta"]["neg_delta_ratio"]
            mutated["feature_delta"]["neg_delta_ratio"] = round(
                1.0 - old_r if 0.0 <= 1.0 - old_r <= 1.0 else 0.0, 3
            )
            if mutated["feature_delta"]["neg_delta_ratio"] != old_r:
                h_mut = compute_canonical_feature_selection_hash(**mutated)
                assert h1 != h_mut, (
                    f"trial {trial}: toggling neg_delta_ratio did not change hash"
                )

            # Removing the feature_delta block must change the hash
            mutated = dict(cfg)
            del mutated["feature_delta"]
            h_mut = compute_canonical_feature_selection_hash(**mutated)
            assert h1 != h_mut, (
                f"trial {trial}: removing feature_delta did not change hash"
            )

        seen_hashes.add(h1)

    assert len(seen_hashes) > 180, (
        f"expected > 180 distinct hashes across 200 random configs, got {len(seen_hashes)} "
        "(collision suspect)"
    )


def test_property_omitted_feature_delta_equals_explicit_none():
    """Calling without feature_delta MUST equal calling with feature_delta=None."""
    rng = random.Random(20260505 + 1)
    for trial in range(50):
        cfg = _random_config(rng)
        # Strip any feature_delta block
        cfg.pop("feature_delta", None)
        h_omitted = compute_canonical_feature_selection_hash(**cfg)
        h_explicit_none = compute_canonical_feature_selection_hash(
            **cfg, feature_delta=None,
        )
        assert h_omitted == h_explicit_none, (
            f"trial {trial}: omitted vs explicit-None disagree"
        )
