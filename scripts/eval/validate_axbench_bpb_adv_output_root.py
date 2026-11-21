"""Validate the permanent current-contract BPB-Adv output inventory."""

from __future__ import annotations

import sys

from steering.eval.axbench_bpb_adv.output_inventory import (
    canonical_output_inventory_json,
    inventory_current_output_root,
)


def main() -> int:
    entries = inventory_current_output_root()
    print(canonical_output_inventory_json(entries), end="")
    return 0 if all(entry.status == "current" for entry in entries) else 1


if __name__ == "__main__":
    sys.exit(main())
