"""Property-style: 200-iteration round-trip fuzz over random
list[SuiteResultVariant]. dump -> load -> dump must be byte-stable."""

from __future__ import annotations

import json
import random

from steering.eval.progression_results import (
    StaticYamlSuiteResult,
    TargetRealLockstepSuiteResult,
    TargetRealPooledSuiteResult,
    dump_suite_to_dict,
    load_suite_from_dict,
)


def _random_variant(rng: random.Random):
    pick = rng.choice([
        "static_yaml", "target_real_lockstep", "target_real_pooled",
    ])
    seeds = {
        str(s): {"score": round(rng.random(), 4)}
        for s in rng.sample(range(1, 100000), rng.randint(1, 4))
    }
    name = f"var_{rng.getrandbits(20):05x}"
    if pick == "static_yaml":
        return StaticYamlSuiteResult(
            name=name,
            seeds=seeds,
        ), name
    if pick == "target_real_lockstep":
        return TargetRealLockstepSuiteResult(
            name=name,
            cache_dir=f"/tmp/{name}",
            seeds=seeds,
        ), name
    return TargetRealPooledSuiteResult(
        name=name,
        cache_dir=f"/tmp/{name}",
        pooling_op=rng.choice(["bucket_max", "bucket_mean"]),
        seeds=seeds,
    ), name


def test_property_round_trip_200_iterations():
    rng = random.Random(20260428)
    for trial in range(200):
        n = rng.randint(1, 8)
        variants_with_names = [_random_variant(rng) for _ in range(n)]
        dumped = [dump_suite_to_dict(v) for v, _ in variants_with_names]
        loaded = [
            load_suite_from_dict(d, name=name)
            for d, (_, name) in zip(dumped, variants_with_names, strict=True)
        ]
        re_dumped = [dump_suite_to_dict(v) for v in loaded]
        assert json.dumps(dumped, sort_keys=True) == json.dumps(re_dumped, sort_keys=True), (
            f"trial {trial}: dump/load/dump round trip not stable"
        )
