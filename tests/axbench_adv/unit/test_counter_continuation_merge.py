"""Publication adds explicit legacy lineage and a complete original-counter grid."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval.axbench_adv.merge_counter_continuations import validate_continuation_inputs
from steering.eval.axbench_adv.counter_resume import QUALIFIED_PARENT, sha256
from steering.eval.axbench_adv.errors import AxBenchAdvError

CHILD = "a" * 40


def _root(tmp_path: Path, name: str, concepts: list[int], *, continuation: bool) -> Path:
    root = tmp_path / name
    root.mkdir()
    doc = {
        "methods": ["SelfConditioning", "SelfConditioningMasked"],
        "pressures": ["linear_probe_counter", "diffmean_counter"],
        "counter_magnitudes": [0.25, 0.5, 1.0, 2.0],
        "max_concepts": 50,
        "concept_ids": concepts,
        "implementation": {
            "repository_git_commit": CHILD if continuation else QUALIFIED_PARENT,
            "package_lock_sha256": "b" * 64,
        },
        "files": {},
    }
    if continuation:
        receipt = {
            "logical_run": "legacy_counter_continuation",
            "snapshot": {"parent_source": QUALIFIED_PARENT},
        }
        path = root / "counter_resume.json"
        path.write_text(json.dumps(receipt))
        doc["counter_resume"] = receipt
        doc["files"]["counter_resume.json"] = sha256(path)
    (root / "manifest.json").write_text(json.dumps(doc))
    return root


def test_valid_original_continuation_grid(tmp_path):
    roots = [
        _root(tmp_path, "old", [0, *range(25, 50)], continuation=False),
        _root(tmp_path, "new", list(range(1, 25)), continuation=True),
    ]
    validate_continuation_inputs(roots, continuation_source=CHILD)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("gap", "publication gap"),
        ("overlap", "repeats"),
        ("source", "lineage"),
        ("receipt", "lineage"),
        ("old_source", "archive implementation"),
    ],
)
def test_publication_refuses_gaps_and_mixed_lineage(tmp_path, mutation, match):
    roots = [
        _root(tmp_path, "old", [0, *range(25, 50)], continuation=False),
        _root(tmp_path, "new", list(range(1, 25)), continuation=True),
    ]
    target = roots[0] if mutation == "old_source" else roots[1]
    path = target / "manifest.json"
    doc = json.loads(path.read_text())
    if mutation == "gap":
        doc["concept_ids"].remove(24)
    if mutation == "overlap":
        doc["concept_ids"].insert(0, 0)
    if mutation in {"source", "old_source"}:
        doc["implementation"]["repository_git_commit"] = "c" * 40
    if mutation == "receipt":
        (target / "counter_resume.json").write_text("changed")
    path.write_text(json.dumps(doc))
    with pytest.raises(AxBenchAdvError, match=match):
        validate_continuation_inputs(roots, continuation_source=CHILD)
