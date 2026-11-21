"""Byte/JSON-exact regression gate: main's clustered selection snapshot.

The golden JSON (tests/steering/goldens/cluster_selection_golden.json) is
captured ON MAIN, before the A5 delegation, by running
tests/steering/goldens/gen_cluster_selection_golden.py -- see that module's
docstring for the exact regeneration command and why it must run pre-
delegation. This test replays the IDENTICAL fixture (same module, same
constants) through whatever ``select_features_for_shard`` implementation is
checked out on the current branch and asserts exact equality against the
golden: positive IDs and order, buckets, uint8 percentiles, negative IDs and
order, and the serialized ``torch.Generator.get_state()`` SHA-256 digest
after each of the >= 12 segments. On A5, ``select_features_for_shard``
delegates the clustered arm to ``select_clustered_segment`` -- so this test
is the campaign's first clustered differential proof that the delegation is
byte-identical to main's inline implementation, not just algorithmically
equivalent in the abstract.

This test intentionally does NOT skip when the golden JSON is missing (only
``test_golden_committed`` documents that state, as a hard failure) -- see
.superpowers/sdd/task-a5-report.md for the golden-handoff command the
controller must run before this file's second test can pass.

Review fix batch (lens-2 Important): a third case, ``threshold_with_
negatives``, was added to the generator (threshold selection + an
independently-seeded NegativeBundle, mirroring feature_pipeline.py's
threshold-path RNG convention that neither of the original two cases
exercised). ``test_delegated_selection_matches_main_golden`` below is
parametrized per case name so each is its own independent test node. The
committed golden JSON was regenerated on main's pre-A5-delegation code
(ec6413fe) with the three-case generator; all three cases are
golden-backed and must pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.steering.goldens.gen_cluster_selection_golden import build_and_run

GOLDEN_PATH = Path(__file__).parent / "goldens" / "cluster_selection_golden.json"


def test_golden_committed() -> None:
    assert GOLDEN_PATH.is_file(), (
        f"golden JSON not committed at {GOLDEN_PATH} -- the controller must run "
        "`PYTHONPATH=. uv run python "
        "tests/steering/goldens/gen_cluster_selection_golden.py` ON MAIN "
        "(pre-A5-delegation) and commit the resulting JSON. See "
        ".superpowers/sdd/task-a5-report.md for the full handoff."
    )


@pytest.mark.parametrize("case_name", ["threshold", "sampling", "threshold_with_negatives"])
def test_delegated_selection_matches_main_golden(case_name: str, tmp_path: Path) -> None:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    actual = build_and_run(tmp_path)

    assert actual["schema_version"] == golden["schema_version"]
    assert actual["n_sae_features"] == golden["n_sae_features"]
    assert actual["cluster_groups"] == golden["cluster_groups"]
    assert actual["mask_blacklist_ids"] == golden["mask_blacklist_ids"]

    assert case_name in actual["cases"], (
        f"build_and_run no longer produces case {case_name!r} -- the generator "
        "and this consumer have drifted out of sync"
    )
    assert case_name in golden["cases"], (
        f"golden JSON at {GOLDEN_PATH} is missing case {case_name!r} -- "
        "regenerate it ON MAIN (pre-A5-delegation) via `PYTHONPATH=. uv run "
        "python tests/steering/goldens/gen_cluster_selection_golden.py` and "
        "commit the result; never regenerate it from delegating code."
    )

    actual_case = actual["cases"][case_name]
    golden_case = golden["cases"][case_name]

    assert actual_case["whole_shard"] == golden_case["whole_shard"], (
        f"{case_name}: whole-shard CSR arrays diverge from the main golden"
    )
    assert actual_case["final_rng_state_sha256"] == golden_case["final_rng_state_sha256"], (
        f"{case_name}: final RNG state digest diverges from the main golden"
    )
    assert len(actual_case["per_segment"]) == len(golden_case["per_segment"]), (
        f"{case_name}: segment count diverges from the main golden"
    )
    for actual_seg, golden_seg in zip(
        actual_case["per_segment"], golden_case["per_segment"], strict=True
    ):
        assert actual_seg == golden_seg, (
            f"{case_name} segment {actual_seg.get('segment')}: diverges from the "
            "main golden (ids/buckets/percentiles/negatives/rng digest)"
        )
