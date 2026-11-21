"""Seal an operator-created immutable counter snapshot, without claiming completion."""

from __future__ import annotations

import argparse
from pathlib import Path

from steering.eval.axbench_adv.counter_resume import seal_snapshot, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--parent-identity", required=True)
    parser.add_argument("--parent-tokenizer-path", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--concept-lo", type=int, required=True)
    parser.add_argument("--concept-hi", type=int, required=True)
    args = parser.parse_args()
    path = seal_snapshot(
        args.snapshot_root,
        parent_identity=args.parent_identity,
        methods=args.methods,
        concept_ids=list(range(args.concept_lo, args.concept_hi)),
        parent_tokenizer_path=args.parent_tokenizer_path,
    )
    print(f"{sha256(path)}  {path}")


if __name__ == "__main__":
    main()
