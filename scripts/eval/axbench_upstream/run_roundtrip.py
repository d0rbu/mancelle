"""Score one merged multi-method steering Parquet with the roundtrip metric.

Re-encodes every steered generation through the concept's own gemma-scope SAE
and reports whether the concept feature fires on the text the method produced.
See ``steering.eval.axbench_upstream.roundtrip`` for the metric itself.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from steering.eval.axbench_upstream.roundtrip import execute_roundtrip


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--include-nl-steering",
        action="store_true",
        help="score the explicit seven-method baseline-plus-project NL product",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    execute_roundtrip(
        input_path=args.input,
        output_dir=args.output_dir,
        device_arg=args.device,
        include_nl_steering=args.include_nl_steering,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
