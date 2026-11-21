"""Permanent structural inventory for the BPB-Adv output root."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
)

KNOWN_OUTPUT_ROOT = Path(
    "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/out/eval/axbench_bpb_adv"
)


@dataclass(frozen=True)
class OutputInventoryEntry:
    run_tag: str
    schema_version: int | None
    status: str
    error: str | None = None


def inventory_current_output_root(
    root: Path = KNOWN_OUTPUT_ROOT,
    *,
    enforce_known_root: bool = True,
) -> tuple[OutputInventoryEntry, ...]:
    """Inventory direct-child runs and reject legacy/migration control-plane state."""

    resolved = root.resolve()
    if enforce_known_root and resolved != KNOWN_OUTPUT_ROOT.resolve():
        raise AxBenchBpbAdvError(
            f"current BPB-Adv inventory is restricted to {KNOWN_OUTPUT_ROOT}; got {resolved}"
        )
    if not resolved.is_dir():
        raise AxBenchBpbAdvError(f"BPB-Adv output root is not a directory: {resolved}")

    entries: list[OutputInventoryEntry] = []
    for child in sorted(resolved.iterdir(), key=lambda path: path.name):
        if child.name.startswith("."):
            entries.append(
                OutputInventoryEntry(child.name, None, "migration_control_plane", "hidden entry")
            )
            continue
        if child.is_symlink() or not child.is_dir():
            entries.append(
                OutputInventoryEntry(child.name, None, "invalid_entry", "run must be a directory")
            )
            continue
        contract_path = child / "source_contract.json"
        if not contract_path.is_file():
            entries.append(OutputInventoryEntry(child.name, None, "missing_contract"))
            continue
        payload: Any = None
        try:
            payload = json.loads(contract_path.read_text(encoding="utf-8"))
            schema = payload.get("schema_version") if isinstance(payload, dict) else None
            BpbAdvSourceContract.from_json_dict(payload, path=contract_path)
            if schema not in {
                SOURCE_CONTRACT_SCHEMA_VERSION,
                CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
            }:
                raise AxBenchBpbAdvError(f"unsupported current schema {schema!r}")
        except (OSError, UnicodeError, json.JSONDecodeError, AxBenchBpbAdvError) as exc:
            schema = payload.get("schema_version") if isinstance(payload, dict) else None
            entries.append(OutputInventoryEntry(child.name, schema, "invalid_contract", str(exc)))
            continue
        entries.append(OutputInventoryEntry(child.name, int(schema), "current"))
    return tuple(entries)


def canonical_output_inventory_json(entries: tuple[OutputInventoryEntry, ...]) -> str:
    return json.dumps([asdict(entry) for entry in entries], sort_keys=True, indent=2) + "\n"


__all__ = [
    "KNOWN_OUTPUT_ROOT",
    "OutputInventoryEntry",
    "canonical_output_inventory_json",
    "inventory_current_output_root",
]
