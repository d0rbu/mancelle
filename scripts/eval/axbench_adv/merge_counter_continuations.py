"""Require original-campaign lineage and its exact two-method c50 counter grid."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from scripts.eval.axbench_adv.merge_slices import merge_slices
from steering.eval.axbench_adv.counter_resume import QUALIFIED_PARENT, read_json, sha256
from steering.eval.axbench_adv.errors import AxBenchAdvError

_METHODS = ["SelfConditioning", "SelfConditioningMasked"]
_PRESSURES = ["linear_probe_counter", "diffmean_counter"]


def validate_continuation_inputs(roots: Sequence[Path], *, continuation_source: str) -> None:
    """Add the campaign lineage/coverage gate, retaining native merger validation."""
    if len(continuation_source) != 40 or any(
        c not in "0123456789abcdef" for c in continuation_source
    ):
        raise AxBenchAdvError("continuation_source must be the exact reviewed commit")
    coverage = set()
    lock_hashes = set()
    for root in roots:
        doc = read_json(root / "manifest.json")
        if (
            doc.get("methods") != _METHODS
            or doc.get("pressures") != _PRESSURES
            or doc.get("counter_magnitudes") != [0.25, 0.5, 1.0, 2.0]
            or doc.get("max_concepts") != 50
        ):
            raise AxBenchAdvError("original counter continuation grid differs")
        implementation = doc.get("implementation", {})
        lock_hashes.add(implementation.get("package_lock_sha256"))
        receipt = doc.get("counter_resume")
        if receipt is None:
            if implementation.get("repository_git_commit") != QUALIFIED_PARENT:
                raise AxBenchAdvError("original archive implementation differs")
        else:
            receipt_path = root / "counter_resume.json"
            if (
                implementation.get("repository_git_commit") != continuation_source
                or receipt.get("logical_run") != "legacy_counter_continuation"
                or receipt.get("snapshot", {}).get("parent_source") != QUALIFIED_PARENT
                or doc.get("files", {}).get("counter_resume.json") != sha256(receipt_path)
                or read_json(receipt_path) != receipt
            ):
                raise AxBenchAdvError("counter continuation lineage differs")
        concepts = doc.get("concept_ids", list(range(50)))
        if (
            not isinstance(concepts, list)
            or not concepts
            or any(type(cid) is not int or not 0 <= cid < 50 for cid in concepts)
        ):
            raise AxBenchAdvError("original counter concept grid is malformed")
        concepts = cast(list[int], concepts)
        if concepts != sorted(set(concepts)):
            raise AxBenchAdvError("original counter concept grid is malformed")
        for cid in concepts:
            if cid in coverage:
                raise AxBenchAdvError(f"original counter campaign repeats concept {cid}")
            coverage.add(cid)
    if coverage != set(range(50)):
        raise AxBenchAdvError(
            f"original counter publication gap: missing={sorted(set(range(50)) - coverage)}"
        )
    if len(lock_hashes) != 1 or None in lock_hashes:
        raise AxBenchAdvError("counter runtime package locks differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--continuation-source", required=True)
    args = parser.parse_args()
    validate_continuation_inputs(args.input_root, continuation_source=args.continuation_source)
    print(merge_slices(args.input_root, args.output_root))


if __name__ == "__main__":
    main()
