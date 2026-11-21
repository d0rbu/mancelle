from __future__ import annotations

import json

from steering.eval.axbench_bpb_adv.output_inventory import inventory_current_output_root
from steering.eval.axbench_bpb_adv.source_validation import BpbAdvSourceContract


def _write_contract(path, *, schema_version: int = 6) -> None:
    payload = BpbAdvSourceContract(
        schema_version=6,
        seed_fingerprint="a" * 64,
        source_run_dir=None,
    ).to_json_dict()
    if schema_version != 6:
        payload["schema_version"] = schema_version
        payload["contract_hash"] = "b" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inventory_accepts_current_contracts(tmp_path) -> None:
    run = tmp_path / "current"
    run.mkdir()
    _write_contract(run / "source_contract.json")

    entries = inventory_current_output_root(tmp_path, enforce_known_root=False)

    assert [(entry.run_tag, entry.status, entry.schema_version) for entry in entries] == [
        ("current", "current", 6)
    ]


def test_inventory_reports_legacy_missing_and_hidden_entries(tmp_path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    _write_contract(legacy / "source_contract.json", schema_version=5)
    missing = tmp_path / "missing"
    missing.mkdir()
    (tmp_path / ".migration.lock").write_text("", encoding="utf-8")

    entries = inventory_current_output_root(tmp_path, enforce_known_root=False)

    assert [(entry.run_tag, entry.status) for entry in entries] == [
        (".migration.lock", "migration_control_plane"),
        ("legacy", "invalid_contract"),
        ("missing", "missing_contract"),
    ]
