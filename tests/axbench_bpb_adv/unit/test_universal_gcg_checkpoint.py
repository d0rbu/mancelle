from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.conditional.feature_gcg.candidate_search as candidate_search
import steering.eval.axbench_bpb_adv.optimize as optimize_module
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.conditional.feature_gcg.edit_space import GroupResult
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, FasterGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.universal_gcg import GROUP_KEY_COLUMNS, concept_groups
from steering.eval.axbench_bpb_adv.universal_gcg_checkpoint import (
    ResolvedTargetCheckpoint,
    UniversalGcgCheckpointStore,
    assert_target_checkpoints_unchanged,
    build_checkpoint_fingerprint,
    canonical_json_bytes,
    canonical_sha256,
    current_runtime_fingerprint,
    resolve_and_pin_target_checkpoints,
    tree_sha256,
    warn_on_runtime_mismatch,
)


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _attack_rows() -> pa.Table:
    prompt = [0, 97, 98]
    rich = [99]
    avoid = [100]
    return pa.Table.from_pylist(
        [
            {
                "attack_row_id": "row",
                "core_prefix_id": "prefix-000001",
                "concept_rich_sample_id": "rich",
                "concept_avoiding_sample_id": "avoid",
                "concept_id": 1,
                "mode": "positive",
                "method_name": "diffmean_decoder_only",
                "target_model": "decoder_only",
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": 1.0,
                "strength_unit": "optimized_tokens",
                "strength_index": 1,
                "context_signal_mode": "fixed_context",
                "prefix_text": "ab",
                "prefix_hash": "hash",
                "source_prompt_tokens": prompt,
                "source_prompt_token_digest": token_sequence_digest(prompt),
                "concept_rich_output_tokens": rich,
                "concept_avoiding_output_tokens": avoid,
                "concept_rich_output_digest": token_sequence_digest(rich),
                "concept_avoiding_output_digest": token_sequence_digest(avoid),
                "editable_token_start": 2,
                "editable_token_count": 1,
                "original_editable_tokens": [98],
            }
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )


def _trigger_row(*, objective: float = 0.0) -> dict[str, Any]:
    group = concept_groups(_attack_rows().to_pylist())[0]
    return {
        **dict(zip(GROUP_KEY_COLUMNS, group.key, strict=True)),
        "adversarial_strength": 1.0,
        "trigger_tokens": [120],
        "trigger_text": "x",
        "objective_value": objective,
        "max_steps": 1,
        "accepted_steps": 1,
        "visited_sequences": 2,
        "n_pairs": 1,
    }


def _store(tmp_path: Path, *, fingerprint: dict[str, Any] | None = None) -> UniversalGcgCheckpointStore:
    return UniversalGcgCheckpointStore.prepare(
        checkpoint_dir=tmp_path / "checkpoint",
        fingerprint=fingerprint or {"semantic": "v1"},
        expected_groups=concept_groups(_attack_rows().to_pylist()),
        tokenizer=_Tokenizer(),
        max_steps=1,
    )


def _rewrite_manifest(store: UniversalGcgCheckpointStore, payload: Any) -> None:
    store.manifest_path.write_bytes(canonical_json_bytes(payload))


def _replace_shard_with_table(store: UniversalGcgCheckpointStore, table: pa.Table) -> Path:
    original = next(store.groups_dir.glob("*/*.parquet"))
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    key_hash = original.stem.split("_")[1]
    replacement = original.with_name(
        f"group_{key_hash}_{hashlib.sha256(payload).hexdigest()}.parquet"
    )
    original.unlink()
    replacement.write_bytes(payload)
    return replacement


def _table_with_row(table: pa.Table, **updates: Any) -> pa.Table:
    row = table.to_pylist()[0]
    row.update(updates)
    return pa.Table.from_pylist(
        [row], schema=UNIVERSAL_TRIGGERS_SCHEMA
    ).replace_schema_metadata(table.schema.metadata)


def test_store_commits_and_restores_exact_trigger(tmp_path: Path) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    runtime = canonical_sha256({"runtime": "cpu"})

    assert store.commit_trigger(group.key_str, _trigger_row(), runtime)
    assert not store.commit_trigger(group.key_str, _trigger_row(), runtime)

    restored = _store(tmp_path)
    assert restored.load_triggers() == {group.key_str: _trigger_row()}
    assert restored.observed_runtime_hashes == frozenset({runtime})
    shards = list((tmp_path / "checkpoint" / "groups").glob("*/*.parquet"))
    assert len(shards) == 1
    assert shards[0].parent.name == shards[0].name.removeprefix("group_")[:2]


def test_store_accepts_positive_infinity_but_rejects_impossible_objectives(
    tmp_path: Path,
) -> None:
    runtime = canonical_sha256({"runtime": "cpu"})
    group = concept_groups(_attack_rows().to_pylist())[0]
    assert _store(tmp_path / "positive").commit_trigger(
        group.key_str,
        _trigger_row(objective=float("inf")),
        runtime,
    )
    for name, value in (("nan", float("nan")), ("negative_inf", -float("inf"))):
        with pytest.raises(AxBenchBpbAdvError, match="objective_value is impossible"):
            _store(tmp_path / name).commit_trigger(
                group.key_str,
                _trigger_row(objective=value),
                runtime,
            )


def test_manifest_mismatch_names_field_and_requires_fresh_run_tag(tmp_path: Path) -> None:
    _store(tmp_path, fingerprint={"nested": {"value": 1}})
    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"fingerprint\.nested\.value.*expected 2.*observed 1.*fresh run_tag",
    ):
        _store(tmp_path, fingerprint={"nested": {"value": 2}})


def test_noncanonical_manifest_and_orphan_state_fail_loud(tmp_path: Path) -> None:
    store = _store(tmp_path / "noncanonical")
    manifest = store.manifest_path
    parsed = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    with pytest.raises(AxBenchBpbAdvError, match="not canonical JSON.*fresh run_tag"):
        _store(tmp_path / "noncanonical")

    orphan = tmp_path / "orphan" / "checkpoint" / "groups" / "aa"
    orphan.mkdir(parents=True)
    (orphan / "group_bad.parquet").write_bytes(b"bad")
    with pytest.raises(AxBenchBpbAdvError, match="no manifest.*fresh run_tag"):
        _store(tmp_path / "orphan")


def test_stale_known_installer_temporary_is_ignored(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".manifest.json.tmp.123.abcdef12").write_bytes(b"partial")
    store = _store(tmp_path)
    assert store.load_triggers() == {}


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_installer_temporary_name_only_ignores_real_regular_files(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    temp = checkpoint / ".manifest.json.tmp.123.abcdef12"
    if entry_kind == "directory":
        temp.mkdir()
    else:
        target = tmp_path / "target"
        target.write_bytes(b"partial")
        temp.symlink_to(target)
    with pytest.raises(
        AxBenchBpbAdvError,
        match="installer temporary entry is not a real regular file.*fresh run_tag",
    ):
        _store(tmp_path)


def test_group_shard_temporary_directory_is_not_silently_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    shard = next(store.groups_dir.glob("*/*.parquet"))
    (shard.parent / f".{shard.name}.tmp.123.abcdef12").mkdir()
    with pytest.raises(AxBenchBpbAdvError, match="temporary entry.*regular file"):
        _store(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("checkpoint_file", "checkpoint root is not a real directory"),
        ("checkpoint_symlink", "checkpoint root is not a real directory"),
        ("manifest_directory", "manifest changed while reading"),
        ("groups_file", "groups entry is not a real directory"),
        ("unknown_root", "unknown checkpoint entry"),
        ("bad_bucket", "unexpected checkpoint bucket entry"),
        ("bucket_file", "unexpected checkpoint bucket entry"),
    ],
)
def test_checkpoint_tree_rejects_every_unexpected_entry_kind(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    if mutation == "checkpoint_file":
        (tmp_path / "checkpoint").write_bytes(b"not a directory")
    elif mutation == "checkpoint_symlink":
        target = tmp_path / "target"
        target.mkdir()
        (tmp_path / "checkpoint").symlink_to(target, target_is_directory=True)
    else:
        store = _store(tmp_path)
        if mutation == "manifest_directory":
            store.manifest_path.unlink()
            store.manifest_path.mkdir()
        elif mutation == "groups_file":
            store.groups_dir.write_bytes(b"not a directory")
        elif mutation == "unknown_root":
            (store.checkpoint_dir / "notes.txt").write_text("unexpected", encoding="utf-8")
        elif mutation == "bad_bucket":
            (store.groups_dir / "xyz").mkdir(parents=True)
        elif mutation == "bucket_file":
            store.groups_dir.mkdir()
            (store.groups_dir / "aa").write_bytes(b"not a directory")
        else:
            pytest.fail(f"unhandled checkpoint tree mutation {mutation}")
    with pytest.raises(AxBenchBpbAdvError, match=rf"(?s){error}.*fresh run_tag"):
        _store(tmp_path)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("root_list", "manifest root must be an object"),
        ("missing_key", "manifest keys differ"),
        ("extra_key", "manifest keys differ"),
        ("schema_bool", "schema_version has invalid type/value"),
        ("schema_version", "schema_version differs"),
        ("algorithm_version", "algorithm_version differs"),
        ("fingerprint_list", "fingerprint must be an object"),
        ("hash_type", "fingerprint_sha256 is invalid"),
        ("hash_shape", "fingerprint_sha256 is invalid"),
        ("hash_mismatch", "differs from canonical fingerprint"),
    ],
)
def test_manifest_shape_and_type_mutations_fail_loud_with_recovery(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    store = _store(tmp_path)
    payload: Any = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    if mutation == "root_list":
        payload = []
    elif mutation == "missing_key":
        payload.pop("algorithm_version")
    elif mutation == "extra_key":
        payload["extra"] = 1
    elif mutation == "schema_bool":
        payload["schema_version"] = True
    elif mutation == "schema_version":
        payload["schema_version"] = 2
    elif mutation == "algorithm_version":
        payload["algorithm_version"] = "other"
    elif mutation == "fingerprint_list":
        payload["fingerprint"] = []
    elif mutation == "hash_type":
        payload["fingerprint_sha256"] = 3
    elif mutation == "hash_shape":
        payload["fingerprint_sha256"] = "A" * 64
    elif mutation == "hash_mismatch":
        payload["fingerprint_sha256"] = "3" * 64
    else:
        pytest.fail(f"unhandled manifest mutation {mutation}")
    _rewrite_manifest(store, payload)
    with pytest.raises(
        AxBenchBpbAdvError,
        match=rf"{error}.*checkpoint=.*fresh run_tag",
    ):
        _store(tmp_path)


def test_manifest_nonfinite_fingerprint_is_path_qualified_and_fail_loud(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    payload["fingerprint"] = {"bad": float("nan")}
    payload["fingerprint_sha256"] = "3" * 64
    store.manifest_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True),
        encoding="utf-8",
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="manifest.fingerprint is not canonical JSON.*checkpoint=.*fresh run_tag",
    ):
        _store(tmp_path)


@pytest.mark.parametrize("mutation", ["type", "keys", "list_length", "list_value"])
def test_manifest_difference_walker_names_nested_shape_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    fingerprint = {"nested": {"items": [1, 2], "value": 3}}
    store = _store(tmp_path, fingerprint=fingerprint)
    payload = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    observed = payload["fingerprint"]
    if mutation == "type":
        observed["nested"]["value"] = "3"
    elif mutation == "keys":
        observed["nested"]["extra"] = 4
    elif mutation == "list_length":
        observed["nested"]["items"].append(3)
    elif mutation == "list_value":
        observed["nested"]["items"][1] = 9
    else:
        pytest.fail(f"unhandled manifest difference mutation {mutation}")
    payload["fingerprint_sha256"] = canonical_sha256(observed)
    _rewrite_manifest(store, payload)
    with pytest.raises(AxBenchBpbAdvError, match=r"manifest fingerprint\.nested"):
        _store(tmp_path, fingerprint=fingerprint)


def test_manifest_malformed_json_and_invalid_utf8_fail_loud(tmp_path: Path) -> None:
    for name, payload in (("json", b"{"), ("utf8", b"\xff")):
        store = _store(tmp_path / name)
        store.manifest_path.write_bytes(payload)
        with pytest.raises(AxBenchBpbAdvError, match="manifest is malformed JSON"):
            _store(tmp_path / name)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("empty", "exactly one row, observed 0"),
        ("multirow", "exactly one row, observed 2"),
        ("wrong_schema", "logical schema differs"),
        ("missing_metadata", "metadata keys differ"),
        ("extra_metadata", "metadata keys differ"),
        ("schema_metadata", "metadata checkpoint_schema_version differs"),
        ("root_metadata", "metadata manifest_fingerprint_sha256 differs"),
        ("group_metadata", "metadata group_key_str differs"),
        ("runtime_non_ascii", "runtime hash is not ASCII"),
        ("runtime_invalid", "runtime hash is invalid"),
        ("null", "contains null values"),
        ("nested_null", "trigger_tokens contains null values"),
        ("identity", "group identity differs"),
        ("identity_invalid", "invalid group identity"),
        ("strength", "adversarial_strength differs"),
        ("max_steps", "max_steps differs"),
        ("n_pairs", "n_pairs differs"),
        ("accepted_negative", "accepted_steps must be in"),
        ("accepted_large", "accepted_steps must be in"),
        ("visited", "visited_sequences must be >= 1"),
        ("accepted_not_visited", "must be smaller than visited_sequences"),
        ("accepted_not_visited", "accepted_steps must be smaller than"),
        ("trigger_length", "trigger length differs"),
        ("trigger_decode", "trigger tokens cannot be decoded"),
        ("trigger_text", "trigger_text differs"),
        ("objective_nan", "objective_value is impossible"),
        ("objective_negative_inf", "objective_value is impossible"),
    ],
)
def test_every_shard_logical_state_guard_rejects_corruption(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    original = next(store.groups_dir.glob("*/*.parquet"))
    table = pq.read_table(original)
    metadata = dict(table.schema.metadata or {})
    if mutation == "empty":
        table = table.slice(0, 0)
    elif mutation == "multirow":
        table = pa.concat_tables([table, table])
    elif mutation == "wrong_schema":
        table = table.append_column("extra", pa.array([1], type=pa.int32()))
    elif mutation == "missing_metadata":
        metadata.pop(b"group_key_str")
        table = table.replace_schema_metadata(metadata)
    elif mutation == "extra_metadata":
        metadata[b"extra"] = b"1"
        table = table.replace_schema_metadata(metadata)
    elif mutation == "schema_metadata":
        metadata[b"checkpoint_schema_version"] = b"2"
        table = table.replace_schema_metadata(metadata)
    elif mutation == "root_metadata":
        metadata[b"manifest_fingerprint_sha256"] = b"2" * 64
        table = table.replace_schema_metadata(metadata)
    elif mutation == "group_metadata":
        metadata[b"group_key_str"] = b"other"
        table = table.replace_schema_metadata(metadata)
    elif mutation == "runtime_non_ascii":
        metadata[b"runtime_numeric_fingerprint_sha256"] = b"\xff" * 64
        table = table.replace_schema_metadata(metadata)
    elif mutation == "runtime_invalid":
        metadata[b"runtime_numeric_fingerprint_sha256"] = b"not-a-hash"
        table = table.replace_schema_metadata(metadata)
    elif mutation == "null":
        table = _table_with_row(table, trigger_text=None)
    elif mutation == "nested_null":
        table = _table_with_row(table, trigger_tokens=[None])
    elif mutation == "identity":
        table = _table_with_row(table, concept_id=2)
    elif mutation == "identity_invalid":
        table = _table_with_row(table, concept_id=-1)
    elif mutation == "strength":
        table = _table_with_row(table, adversarial_strength=2.0)
    elif mutation == "max_steps":
        table = _table_with_row(table, max_steps=2)
    elif mutation == "n_pairs":
        table = _table_with_row(table, n_pairs=2)
    elif mutation == "accepted_negative":
        table = _table_with_row(table, accepted_steps=-1)
    elif mutation == "accepted_large":
        table = _table_with_row(table, accepted_steps=2)
    elif mutation == "visited":
        table = _table_with_row(table, visited_sequences=0)
    elif mutation == "accepted_not_visited":
        table = _table_with_row(table, accepted_steps=1, visited_sequences=1)
    elif mutation == "trigger_length":
        table = _table_with_row(table, trigger_tokens=[], trigger_text="")
    elif mutation == "trigger_decode":
        table = _table_with_row(table, trigger_tokens=[-1])
    elif mutation == "trigger_text":
        table = _table_with_row(table, trigger_text="wrong")
    elif mutation == "objective_nan":
        table = _table_with_row(table, objective_value=float("nan"))
    elif mutation == "objective_negative_inf":
        table = _table_with_row(table, objective_value=-float("inf"))
    else:
        pytest.fail(f"unhandled shard mutation {mutation}")
    _replace_shard_with_table(store, table)
    with pytest.raises(AxBenchBpbAdvError, match=rf"(?s){error}.*fresh run_tag"):
        _store(tmp_path)


def test_shard_physical_identity_guards_digest_bucket_key_and_parquet(
    tmp_path: Path,
) -> None:
    group = concept_groups(_attack_rows().to_pylist())[0]

    digest_store = _store(tmp_path / "digest")
    digest_store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    digest_shard = next(digest_store.groups_dir.glob("*/*.parquet"))
    digest_shard.write_bytes(digest_shard.read_bytes() + b"changed")
    with pytest.raises(AxBenchBpbAdvError, match="payload digest differs"):
        _store(tmp_path / "digest")

    corrupt_store = _store(tmp_path / "corrupt")
    corrupt_store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    corrupt_shard = next(corrupt_store.groups_dir.glob("*/*.parquet"))
    corrupt_shard.unlink()
    payload = b"not parquet"
    key_hash = hashlib.sha256(group.key_str.encode()).hexdigest()
    corrupt_shard = corrupt_store.groups_dir / key_hash[:2] / (
        f"group_{key_hash}_{hashlib.sha256(payload).hexdigest()}.parquet"
    )
    corrupt_shard.write_bytes(payload)
    with pytest.raises(AxBenchBpbAdvError, match="not valid Parquet"):
        _store(tmp_path / "corrupt")

    bucket_store = _store(tmp_path / "bucket")
    bucket_store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    bucket_shard = next(bucket_store.groups_dir.glob("*/*.parquet"))
    wrong_bucket = "ff" if bucket_shard.parent.name != "ff" else "ee"
    destination = bucket_store.groups_dir / wrong_bucket
    destination.mkdir()
    bucket_shard.rename(destination / bucket_shard.name)
    with pytest.raises(AxBenchBpbAdvError, match="shard bucket differs"):
        _store(tmp_path / "bucket")

    key_store = _store(tmp_path / "key")
    key_store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    key_shard = next(key_store.groups_dir.glob("*/*.parquet"))
    unknown_hash = "0" * 64
    unknown_bucket = key_store.groups_dir / "00"
    unknown_bucket.mkdir(exist_ok=True)
    key_shard.rename(
        unknown_bucket
        / f"group_{unknown_hash}_{key_shard.stem.split('_')[2]}.parquet"
    )
    with pytest.raises(AxBenchBpbAdvError, match="unknown group-key hash"):
        _store(tmp_path / "key")


def test_loaded_trigger_rows_are_defensive_deep_copies(tmp_path: Path) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    loaded = store.load_triggers()
    loaded[group.key_str]["trigger_tokens"][0] = 121
    assert store.load_triggers()[group.key_str]["trigger_tokens"] == [120]


def test_load_triggers_validates_requested_subset_membership(tmp_path: Path) -> None:
    groups = concept_groups(_process_attack_rows().to_pylist())
    store = UniversalGcgCheckpointStore.prepare(
        checkpoint_dir=tmp_path / "checkpoint",
        fingerprint={"semantic": "v1"},
        expected_groups=groups,
        tokenizer=_Tokenizer(),
        max_steps=1,
    )
    first_row = _trigger_row()
    first_row["concept_id"] = 1
    store.commit_trigger(groups[0].key_str, first_row, "1" * 64)
    assert store.load_triggers([groups[0]]) == {groups[0].key_str: first_row}

    unknown_row = deepcopy(groups[0].rows[0])
    unknown_row["concept_id"] = 99
    unknown_row["attack_row_id"] = "unknown"
    unknown_row["core_prefix_id"] = "unknown"
    unknown_row["concept_rich_sample_id"] = "unknown"
    unknown_row["concept_avoiding_sample_id"] = "unknown"
    unknown = concept_groups([unknown_row])[0]
    with pytest.raises(AxBenchBpbAdvError, match="unknown expected groups"):
        store.load_triggers([unknown])


def test_store_rejects_duplicate_expected_group_keys(tmp_path: Path) -> None:
    group = concept_groups(_attack_rows().to_pylist())[0]
    with pytest.raises(AxBenchBpbAdvError, match="expected group keys are not unique"):
        UniversalGcgCheckpointStore(
            checkpoint_dir=tmp_path / "checkpoint",
            fingerprint={"semantic": "v1"},
            expected_groups=[group, group],
            tokenizer=_Tokenizer(),
            max_steps=1,
        )


def test_same_payload_stores_converge_and_divergent_payload_fails_loud(
    tmp_path: Path,
) -> None:
    group = concept_groups(_attack_rows().to_pylist())[0]
    first = _store(tmp_path)
    second = _store(tmp_path)
    assert first.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    assert not second.commit_trigger(group.key_str, _trigger_row(), "1" * 64)

    third = _store(tmp_path)
    changed = _trigger_row()
    changed["objective_value"] = 1.0
    with pytest.raises(AxBenchBpbAdvError, match="differs from committed group"):
        third.commit_trigger(group.key_str, changed, "1" * 64)


@pytest.mark.parametrize("max_steps", [True, -1, 1.5])
def test_store_rejects_invalid_max_steps(tmp_path: Path, max_steps: Any) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="max_steps must be a non-negative integer"):
        UniversalGcgCheckpointStore(
            checkpoint_dir=tmp_path / "checkpoint",
            fingerprint={"semantic": "v1"},
            expected_groups=concept_groups(_attack_rows().to_pylist()),
            tokenizer=_Tokenizer(),
            max_steps=max_steps,
        )


@pytest.mark.parametrize(
    ("key", "runtime", "error"),
    [
        ([], "1" * 64, "group key must be a string"),
        ("known", None, "runtime fingerprint is invalid"),
        ("known", "A" * 64, "runtime fingerprint is invalid"),
    ],
)
def test_commit_rejects_malformed_key_and_runtime_types(
    tmp_path: Path,
    key: Any,
    runtime: Any,
    error: str,
) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    resolved_key = group.key_str if key == "known" else key
    with pytest.raises(AxBenchBpbAdvError, match=error):
        store.commit_trigger(resolved_key, _trigger_row(), runtime)


def test_commit_rejects_unknown_keys_columns_and_arrow_type_errors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    with pytest.raises(AxBenchBpbAdvError, match="unknown or control group"):
        store.commit_trigger("unknown", _trigger_row(), "1" * 64)
    missing = _trigger_row()
    missing.pop("trigger_text")
    with pytest.raises(AxBenchBpbAdvError, match="columns differ"):
        store.commit_trigger(group.key_str, missing, "1" * 64)
    malformed = _trigger_row()
    malformed["concept_id"] = 2**80
    with pytest.raises(AxBenchBpbAdvError, match="does not conform"):
        store.commit_trigger(group.key_str, malformed, "1" * 64)


def test_regular_group_installer_temporary_is_ignored_on_startup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    group = concept_groups(_attack_rows().to_pylist())[0]
    store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    shard = next(store.groups_dir.glob("*/*.parquet"))
    (shard.parent / f".{shard.name}.tmp.123.abcdef12").write_bytes(b"partial")
    assert _store(tmp_path).load_triggers() == {group.key_str: _trigger_row()}


def test_commit_rescan_ignores_temp_and_unrelated_prefix_but_rejects_bad_own_prefix(
    tmp_path: Path,
) -> None:
    group = concept_groups(_attack_rows().to_pylist())[0]
    key_hash = hashlib.sha256(group.key_str.encode("utf-8")).hexdigest()

    accepted = _store(tmp_path / "accepted")
    bucket = accepted.groups_dir / key_hash[:2]
    bucket.mkdir(parents=True)
    unrelated_key = "f" * 64
    if unrelated_key.startswith(key_hash):
        unrelated_key = "e" * 64
    (bucket / f"group_{unrelated_key}_{'0' * 64}.parquet").write_bytes(b"unrelated")
    expected_name = f"group_{key_hash}_{'1' * 64}.parquet"
    (bucket / f".{expected_name}.tmp.123.abcdef12").write_bytes(b"partial")
    assert accepted.commit_trigger(group.key_str, _trigger_row(), "1" * 64)

    rejected = _store(tmp_path / "rejected")
    bad_bucket = rejected.groups_dir / key_hash[:2]
    bad_bucket.mkdir(parents=True)
    (bad_bucket / f"group_{key_hash}_notes").write_bytes(b"bad")
    with pytest.raises(AxBenchBpbAdvError, match="unknown committed checkpoint filename"):
        rejected.commit_trigger(group.key_str, _trigger_row(), "1" * 64)


def test_commit_requires_exactly_one_payload_after_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    group = concept_groups(_attack_rows().to_pylist())[0]
    store = _store(tmp_path)
    key_hash = hashlib.sha256(group.key_str.encode("utf-8")).hexdigest()
    bucket = store.groups_dir / key_hash[:2]

    def _install_nothing(_payload: bytes, _path: Path, *, surface: str) -> bool:
        del surface
        bucket.mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(checkpoint_module, "install_bytes_once", _install_nothing)
    with pytest.raises(AxBenchBpbAdvError, match="exactly one committed payload"):
        store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)


def test_manifest_and_shard_install_failures_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module,
        "install_bytes_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="denied.*fresh run_tag"):
        _store(tmp_path / "manifest")

    monkeypatch.undo()
    store = _store(tmp_path / "shard")
    group = concept_groups(_attack_rows().to_pylist())[0]
    monkeypatch.setattr(
        checkpoint_module,
        "install_bytes_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="denied.*fresh run_tag"):
        store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)


def test_manifest_defensive_read_and_canonicalization_failures_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    store = _store(tmp_path)
    expected = store.manifest_path.read_bytes()
    monkeypatch.setattr(
        checkpoint_module,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="denied.*fresh run_tag"):
        store._validate_manifest(expected)

    monkeypatch.setattr(
        checkpoint_module,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: SimpleNamespace(payload=None),
    )
    with pytest.raises(AxBenchBpbAdvError, match="retained no payload"):
        store._validate_manifest(expected)

    monkeypatch.undo()
    store = _store(tmp_path)
    real_canonical = checkpoint_module.canonical_json_bytes

    def _reject_outer_manifest(value: Any) -> bytes:
        if isinstance(value, dict) and "schema_version" in value:
            raise AxBenchBpbAdvError("forced canonical failure")
        return real_canonical(value)

    monkeypatch.setattr(
        checkpoint_module,
        "canonical_json_bytes",
        _reject_outer_manifest,
    )
    with pytest.raises(AxBenchBpbAdvError, match="forced canonical failure.*fresh run_tag"):
        store._validate_manifest(expected)

    monkeypatch.setattr(checkpoint_module, "canonical_json_bytes", real_canonical)
    with pytest.raises(AxBenchBpbAdvError, match="canonical payload differs"):
        store._validate_manifest(b"deliberately-wrong-expected-payload")


def test_every_checkpoint_enumeration_error_is_path_qualified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_iterdir = Path.iterdir
    checkpoint = tmp_path / "prepare" / "checkpoint"
    checkpoint.mkdir(parents=True)

    def _deny_prepare(path: Path) -> Any:
        if path == checkpoint:
            raise PermissionError("prepare denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _deny_prepare)
    with pytest.raises(AxBenchBpbAdvError, match="cannot enumerate checkpoint root"):
        _store(tmp_path / "prepare")

    monkeypatch.undo()
    root_store = _store(tmp_path / "root")

    def _deny_root(path: Path) -> Any:
        if path == root_store.checkpoint_dir:
            raise PermissionError("root denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _deny_root)
    with pytest.raises(AxBenchBpbAdvError, match="cannot enumerate checkpoint root"):
        root_store._scan_all_shards()

    monkeypatch.undo()
    groups_store = _store(tmp_path / "groups")
    groups_store.groups_dir.mkdir()

    def _deny_groups(path: Path) -> Any:
        if path == groups_store.groups_dir:
            raise PermissionError("groups denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _deny_groups)
    with pytest.raises(AxBenchBpbAdvError, match="cannot enumerate checkpoint groups"):
        groups_store._scan_all_shards()

    monkeypatch.undo()
    bucket_store = _store(tmp_path / "bucket")
    group = concept_groups(_attack_rows().to_pylist())[0]
    bucket_store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    bucket = next(path for path in bucket_store.groups_dir.iterdir())

    def _deny_bucket(path: Path) -> Any:
        if path == bucket:
            raise PermissionError("bucket denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _deny_bucket)
    with pytest.raises(AxBenchBpbAdvError, match="cannot enumerate checkpoint bucket"):
        bucket_store._scan_all_shards()


def test_manifest_and_shard_race_entry_types_fail_loud(tmp_path: Path) -> None:
    manifest_store = _store(tmp_path / "manifest")
    manifest_store.manifest_path.unlink()
    manifest_store.manifest_path.mkdir()
    with pytest.raises(AxBenchBpbAdvError, match="manifest is not a real regular file"):
        manifest_store._scan_all_shards()

    shard_store = _store(tmp_path / "shard")
    group = concept_groups(_attack_rows().to_pylist())[0]
    key_hash = hashlib.sha256(group.key_str.encode("utf-8")).hexdigest()
    bucket = shard_store.groups_dir / key_hash[:2]
    bucket.mkdir(parents=True)
    (bucket / f"group_{key_hash}_{'0' * 64}.parquet").mkdir()
    with pytest.raises(AxBenchBpbAdvError, match="shard is not a real regular file"):
        _store(tmp_path / "shard")


def test_shard_defensive_read_and_serialization_failures_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    group = concept_groups(_attack_rows().to_pylist())[0]
    store = _store(tmp_path)
    store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)
    shard = next(store.groups_dir.glob("*/*.parquet"))
    store._rows.clear()
    monkeypatch.setattr(
        checkpoint_module,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="denied.*fresh run_tag"):
        store._load_shard(shard, expected_bucket=shard.parent.name)

    payload_hash = shard.stem.split("_")[2]
    monkeypatch.setattr(
        checkpoint_module,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: SimpleNamespace(sha256=payload_hash, payload=None),
    )
    with pytest.raises(AxBenchBpbAdvError, match="retained no payload"):
        store._load_shard(shard, expected_bucket=shard.parent.name)

    monkeypatch.undo()
    fresh = _store(tmp_path / "serialize")
    monkeypatch.setattr(
        checkpoint_module.pq,
        "write_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("serialize denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="could not serialize.*fresh run_tag"):
        fresh.commit_trigger(group.key_str, _trigger_row(), "1" * 64)


def test_commit_bucket_enumeration_error_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = concept_groups(_attack_rows().to_pylist())[0]
    store = _store(tmp_path)
    key_hash = hashlib.sha256(group.key_str.encode("utf-8")).hexdigest()
    bucket = store.groups_dir / key_hash[:2]
    original_iterdir = Path.iterdir

    def _deny_bucket(path: Path) -> Any:
        if path == bucket:
            raise PermissionError("commit bucket denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _deny_bucket)
    with pytest.raises(AxBenchBpbAdvError, match="cannot enumerate checkpoint bucket"):
        store.commit_trigger(group.key_str, _trigger_row(), "1" * 64)


def test_expected_group_key_hash_collision_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    groups = concept_groups(_process_attack_rows().to_pylist())[:2]
    original_sha256 = hashlib.sha256
    colliding_inputs = {group.key_str.encode("utf-8") for group in groups}

    def _sha256(value: bytes = b"") -> Any:
        if value in colliding_inputs:
            return SimpleNamespace(hexdigest=lambda: "a" * 64)
        return original_sha256(value)

    monkeypatch.setattr(checkpoint_module.hashlib, "sha256", _sha256)
    with pytest.raises(AxBenchBpbAdvError, match="SHA-256 collision"):
        UniversalGcgCheckpointStore(
            checkpoint_dir=tmp_path / "checkpoint",
            fingerprint={"semantic": "v1"},
            expected_groups=groups,
            tokenizer=_Tokenizer(),
            max_steps=1,
        )


def test_duplicate_and_unknown_committed_shards_fail_loud(tmp_path: Path) -> None:
    runtime = canonical_sha256({"runtime": "cpu"})
    group = concept_groups(_attack_rows().to_pylist())[0]
    store = _store(tmp_path / "duplicate")
    store.commit_trigger(group.key_str, _trigger_row(), runtime)
    original = next(store.groups_dir.glob("*/*.parquet"))
    table = pq.read_table(original)
    metadata = dict(table.schema.metadata or {})
    metadata[b"runtime_numeric_fingerprint_sha256"] = b"3" * 64
    sink = pa.BufferOutputStream()
    pq.write_table(table.replace_schema_metadata(metadata), sink)
    payload = sink.getvalue().to_pybytes()
    parts = original.stem.split("_")
    duplicate = original.with_name(
        f"group_{parts[1]}_{hashlib.sha256(payload).hexdigest()}.parquet"
    )
    duplicate.write_bytes(payload)
    with pytest.raises(AxBenchBpbAdvError, match="multiple committed payloads.*fresh run_tag"):
        _store(tmp_path / "duplicate")

    unknown_store = _store(tmp_path / "unknown")
    bucket = unknown_store.groups_dir / "aa"
    bucket.mkdir(parents=True)
    (bucket / "notes.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(AxBenchBpbAdvError, match="unknown committed checkpoint filename"):
        _store(tmp_path / "unknown")


def test_tree_digest_is_order_stable_and_rejects_symlink(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b").write_bytes(b"two")
    (first / "a").write_bytes(b"one")
    (second / "a").write_bytes(b"one")
    (second / "b").write_bytes(b"two")
    assert tree_sha256(first) == tree_sha256(second)

    (first / "link").symlink_to(first / "a")
    with pytest.raises(AxBenchBpbAdvError, match="symlink or non-regular"):
        tree_sha256(first)


def test_tree_digest_changes_for_bytes_lengths_and_relative_names(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    nested = root / "nested"
    nested.mkdir(parents=True)
    leaf = nested / "leaf"
    leaf.write_bytes(b"one")
    baseline = tree_sha256(root)
    leaf.write_bytes(b"two")
    assert tree_sha256(root) != baseline
    leaf.write_bytes(b"one-longer")
    assert tree_sha256(root) != baseline
    leaf.write_bytes(b"one")
    leaf.rename(nested / "renamed")
    assert tree_sha256(root) != baseline

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(AxBenchBpbAdvError, match="symlink or non-regular"):
        tree_sha256(root)


def test_tree_hashing_wraps_missing_roots_files_and_enumeration_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    with pytest.raises(AxBenchBpbAdvError, match="cannot inspect required tree"):
        tree_sha256(tmp_path / "missing")
    leaf = tmp_path / "leaf"
    leaf.write_bytes(b"file")
    with pytest.raises(AxBenchBpbAdvError, match="root is not a real directory"):
        tree_sha256(leaf)

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(
        checkpoint_module.os,
        "scandir",
        lambda _path: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="cannot enumerate tree directory"):
        tree_sha256(root)

    class _UnreadableEntry:
        name = "entry"
        path = str(root / "entry")

        def stat(self, *, follow_symlinks: bool) -> Any:
            assert not follow_symlinks
            raise PermissionError("denied")

    monkeypatch.setattr(checkpoint_module.os, "scandir", lambda _path: [_UnreadableEntry()])
    with pytest.raises(AxBenchBpbAdvError, match="cannot inspect tree entry"):
        tree_sha256(root)


def test_tree_hashing_detects_file_set_change_and_wraps_descriptor_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"a")
    real_enumerate = checkpoint_module._enumerate_regular_tree
    calls = 0

    def _changing_tree(path: Path) -> list[Path]:
        nonlocal calls
        calls += 1
        result = real_enumerate(path)
        if calls == 2:
            return [*result, Path("added")]
        return result

    monkeypatch.setattr(checkpoint_module, "_enumerate_regular_tree", _changing_tree)
    with pytest.raises(AxBenchBpbAdvError, match="tree file set changed"):
        tree_sha256(root)

    monkeypatch.setattr(checkpoint_module, "_enumerate_regular_tree", real_enumerate)
    monkeypatch.setattr(
        checkpoint_module,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )
    with pytest.raises(AxBenchBpbAdvError, match="could not hash tree file"):
        tree_sha256(root)


def test_single_file_hash_wraps_native_filesystem_failure(tmp_path: Path) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    with pytest.raises(
        AxBenchBpbAdvError,
        match="could not stably hash missing.*fresh run_tag",
    ):
        checkpoint_module._hash_file(tmp_path / "missing", label="missing")


def test_target_checkpoint_resolution_pins_latest_and_detects_postload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    model_path = checkpoint_dir / "model_000007.pt"
    metadata_path = checkpoint_dir / "meta_000007.json"
    model_path.write_bytes(b"model-v1")
    metadata_path.write_text('{"model_config":{}}', encoding="utf-8")
    monkeypatch.setattr(checkpoint_module, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(checkpoint_module, "find_largest_model", lambda _root: "d12")
    monkeypatch.setattr(
        checkpoint_module,
        "get_checkpoint_dir",
        lambda _source, *, model_tag: str(checkpoint_dir),
    )
    monkeypatch.setattr(checkpoint_module, "find_last_step", lambda _root: 7)
    cfg = AxBenchBpbConfig(
        decoder_only={"source": "base", "model_tag": None, "step": None},
    )

    resolved = resolve_and_pin_target_checkpoints(cfg, {"decoder_only"})
    assert cfg.decoder_only["model_tag"] == "d12"
    assert cfg.decoder_only["step"] == 7
    assert resolved["decoder_only"].model_path == model_path
    assert_target_checkpoints_unchanged(resolved)

    model_path.write_bytes(b"model-v2")
    with pytest.raises(AxBenchBpbAdvError, match="target identity changed after runtime setup"):
        assert_target_checkpoints_unchanged(resolved)
    model_path.write_bytes(b"model-v1")
    metadata_path.write_text('{"model_config":{"depth":13}}', encoding="utf-8")
    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"targets\.decoder_only\.metadata_sha256",
    ):
        assert_target_checkpoints_unchanged(resolved)


@pytest.mark.parametrize(
    ("target_names", "target_cfg", "error"),
    [
        ({"other"}, {"source": "base", "model_tag": "d12", "step": 1}, "unsupported"),
        ({"decoder_only"}, {"source": "", "model_tag": "d12", "step": 1}, "source"),
        (
            {"decoder_only"},
            {"source": "base", "model_tag": "", "step": 1},
            "model_tag",
        ),
        (
            {"decoder_only"},
            {"source": "base", "model_tag": "d12", "step": True},
            "non-negative integer",
        ),
        (
            {"decoder_only"},
            {"source": "base", "model_tag": "d12", "step": -1},
            "non-negative integer",
        ),
        (
            {"decoder_only"},
            {"source": "base", "model_tag": "d12", "step": 1.5},
            "non-negative integer",
        ),
    ],
)
def test_target_checkpoint_resolution_rejects_malformed_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_names: set[str],
    target_cfg: dict[str, Any],
    error: str,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module,
        "get_checkpoint_dir",
        lambda _source, *, model_tag: str(tmp_path / str(model_tag)),
    )
    cfg = AxBenchBpbConfig(decoder_only=target_cfg)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        resolve_and_pin_target_checkpoints(cfg, target_names)


def test_explicit_target_checkpoint_identity_never_resolves_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_000003.pt").write_bytes(b"model")
    (checkpoint / "meta_000003.json").write_bytes(b"meta")
    monkeypatch.setattr(
        checkpoint_module,
        "find_largest_model",
        lambda _root: pytest.fail("explicit model tag consulted latest tag"),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "find_last_step",
        lambda _root: pytest.fail("explicit step consulted latest step"),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "get_checkpoint_dir",
        lambda source, *, model_tag: str(checkpoint),
    )
    cfg = AxBenchBpbConfig(
        decoder_only={"source": "base", "model_tag": "d12", "step": 3}
    )
    resolved = resolve_and_pin_target_checkpoints(cfg, {"decoder_only"})
    assert resolved["decoder_only"].model_tag == "d12"
    assert resolved["decoder_only"].step == 3


def test_encdec_target_resolution_pins_nested_leaf_into_encdec_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    checkpoint = tmp_path / "conditional" / "d12" / "s128" / "t1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_000004.pt").write_bytes(b"model")
    (checkpoint / "meta_000004.json").write_bytes(b"meta")
    monkeypatch.setattr(checkpoint_module, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(checkpoint_module, "find_largest_model", lambda root: "d12")
    monkeypatch.setattr(
        checkpoint_module,
        "get_checkpoint_dir",
        lambda source, *, model_tag: str(checkpoint),
    )
    monkeypatch.setattr(checkpoint_module, "find_last_step", lambda root: 4)
    cfg = AxBenchBpbConfig(
        encdec={"source": "encdec", "model_tag": None, "step": None}
    )
    resolved = resolve_and_pin_target_checkpoints(cfg, {"encdec"})
    assert cfg.encdec["model_tag"] == "d12"
    assert cfg.encdec["step"] == 4
    assert resolved["encdec"].checkpoint_dir == checkpoint


def _objective(items: Sequence[tuple[dict[str, Any], Sequence[int]]]) -> list[float]:
    return [0.0 if int(tokens[-1]) == 120 else 1.0 for _row, tokens in items]


def _gradient(
    items: Sequence[tuple[dict[str, Any], Sequence[int]]],
    _step: int,
) -> list[t.Tensor]:
    gradients: list[t.Tensor] = []
    for _row, tokens in items:
        gradient = t.zeros(len(tokens), 2)
        gradient[-1] = t.tensor([1.0, 0.0])
        gradients.append(gradient)
    return gradients


def _build(
    *,
    attack_rows: pa.Table | None = None,
    restored: dict[str, dict[str, Any]] | None = None,
    on_complete: Any = None,
    events: list[dict[str, Any]] | None = None,
    objective_fn: Any = _objective,
    gradient_fn: Any = _gradient,
    group_boundary_callback: Any = None,
) -> tuple[pa.Table, list[dict[str, Any]]]:
    weight = t.zeros(121, 2)
    weight[120] = t.tensor([-1.0, 0.0])
    return build_universal_gcg_candidates(
        attack_rows=_attack_rows() if attack_rows is None else attack_rows,
        tokenizer=_Tokenizer(),
        cfg=FasterGcgConfig(
            max_steps=1,
            candidate_batch_size=1,
            top_k=1,
            distance_weight=0.0,
            temperature=1.0,
        ),
        embedding_weight=weight,
        allowed_token_ids=[120],
        objective=objective_fn,
        gradient_provider_batch=gradient_fn,
        restored_triggers=restored,
        on_trigger_complete=on_complete,
        progress_callback=None if events is None else events.append,
        group_boundary_callback=group_boundary_callback,
    )


def test_all_restored_reconstructs_exactly_without_model_or_pool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, triggers = _build()
    key = concept_groups(_attack_rows().to_pylist())[0].key_str

    def _tripwire(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("restored group must not execute model/search callbacks")

    monkeypatch.setattr(candidate_search, "_materialize_allowed_embedding_pool", _tripwire)
    events: list[dict[str, Any]] = []
    boundaries: list[str] = []
    restored_candidates, restored_triggers = _build(
        restored={key: triggers[0]},
        objective_fn=_tripwire,
        gradient_fn=_tripwire,
        events=events,
        group_boundary_callback=lambda: boundaries.append("boundary"),
    )
    assert restored_candidates.equals(candidates)
    assert restored_triggers == triggers
    assert boundaries == ["boundary"]
    assert events == []


def test_commit_failure_precedes_group_done() -> None:
    events: list[dict[str, Any]] = []

    def _fail_commit(_key: str, _row: dict[str, Any]) -> NoReturn:
        raise RuntimeError("disk failed")

    with pytest.raises(RuntimeError, match="disk failed"):
        _build(on_complete=_fail_commit, events=events)
    assert "group_start" in [event["event"] for event in events]
    assert "group_done" not in [event["event"] for event in events]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_column", "exactly UNIVERSAL_TRIGGERS_SCHEMA columns"),
        ("extra_column", "exactly UNIVERSAL_TRIGGERS_SCHEMA columns"),
        ("null", "must not contain null values"),
        ("nested_null", "trigger_tokens must not contain null values"),
        ("overflow", "does not conform to UNIVERSAL_TRIGGERS_SCHEMA"),
        ("identity", "does not reconstruct exactly"),
        ("strength", "does not reconstruct exactly"),
        ("trigger_length", "replacement length"),
        ("trigger_text", "does not reconstruct exactly"),
        ("max_steps", "does not reconstruct exactly"),
        ("n_pairs", "does not reconstruct exactly"),
        ("accepted_negative", "accepted_steps must be in"),
        ("accepted_large", "accepted_steps must be in"),
        ("visited", "visited_sequences must be >= 1"),
        ("accepted_not_visited", "accepted_steps must be smaller than"),
        ("objective_nan", "impossible objective_value"),
        ("objective_negative_inf", "impossible objective_value"),
    ],
)
def test_builder_rejects_every_malformed_restored_trigger_surface(
    mutation: str,
    error: str,
) -> None:
    _candidates, triggers = _build()
    group = concept_groups(_attack_rows().to_pylist())[0]
    row = deepcopy(triggers[0])
    if mutation == "missing_column":
        row.pop("trigger_text")
    elif mutation == "extra_column":
        row["extra"] = 1
    elif mutation == "null":
        row["trigger_text"] = None
    elif mutation == "nested_null":
        row["trigger_tokens"] = [None]
    elif mutation == "overflow":
        row["concept_id"] = 2**80
    elif mutation == "identity":
        row["concept_id"] = 2
    elif mutation == "strength":
        row["adversarial_strength"] = 2.0
    elif mutation == "trigger_length":
        row["trigger_tokens"] = []
        row["trigger_text"] = ""
    elif mutation == "trigger_text":
        row["trigger_text"] = "wrong"
    elif mutation == "max_steps":
        row["max_steps"] = 2
    elif mutation == "n_pairs":
        row["n_pairs"] = 2
    elif mutation == "accepted_negative":
        row["accepted_steps"] = -1
    elif mutation == "accepted_large":
        row["accepted_steps"] = 2
    elif mutation == "visited":
        row["visited_sequences"] = 0
    elif mutation == "accepted_not_visited":
        row["accepted_steps"] = 1
        row["visited_sequences"] = 1
    elif mutation == "objective_nan":
        row["objective_value"] = float("nan")
    elif mutation == "objective_negative_inf":
        row["objective_value"] = -float("inf")
    else:
        pytest.fail(f"unhandled restored trigger mutation {mutation}")
    with pytest.raises((AxBenchBpbAdvError, ValueError), match=error):
        _build(restored={group.key_str: row})


def test_builder_rejects_unknown_restored_key_before_search() -> None:
    def _tripwire(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("search ran before restored-key membership validation")

    with pytest.raises(AxBenchBpbAdvError, match="do not belong to this build"):
        _build(
            restored={"unknown": _trigger_row()},
            objective_fn=_tripwire,
            gradient_fn=_tripwire,
        )


def test_builder_rejects_duplicate_or_unconsumed_restored_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _candidates, triggers = _build()
    group = concept_groups(_attack_rows().to_pylist())[0]

    def _consume_twice(**kwargs: Any) -> list[GroupResult]:
        restore = kwargs["restore_group_result"]
        assert restore is not None
        validated = kwargs["edit_space"].validate(kwargs["groups"][0])
        assert restore(validated) is not None
        restore(validated)
        pytest.fail("duplicate restored callback did not fail")

    monkeypatch.setattr(optimize_module, "run_gcg", _consume_twice)
    with pytest.raises(AxBenchBpbAdvError, match="consumed more than once"):
        _build(restored={group.key_str: triggers[0]})

    monkeypatch.setattr(optimize_module, "run_gcg", lambda **_kwargs: [])
    with pytest.raises(AxBenchBpbAdvError, match="not consumed exactly once"):
        _build(restored={group.key_str: triggers[0]})


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (object(), "invalid payload"),
        ({"unexpected": True}, "invalid payload"),
        ({"candidates": [], "trigger": "not-a-mapping"}, "must contain one trigger"),
    ],
)
def test_builder_rejects_invalid_finalized_commit_payload(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    error: str,
) -> None:
    def _commit_invalid(**kwargs: Any) -> list[GroupResult]:
        commit = kwargs["commit_group_result"]
        assert commit is not None
        validated = kwargs["edit_space"].validate(kwargs["groups"][0])
        commit(validated, GroupResult(payload=payload))
        pytest.fail("invalid finalized payload did not fail")

    monkeypatch.setattr(optimize_module, "run_gcg", _commit_invalid)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        _build(on_complete=lambda _key, _row: None)


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def test_multi_group_fresh_mixed_and_all_restored_preserve_bytes_order_and_pool_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _process_attack_rows()
    groups = concept_groups(rows.to_pylist())
    original_materialize = candidate_search._materialize_allowed_embedding_pool
    materializations = 0

    def _count_materialization(**kwargs: Any) -> tuple[t.Tensor, t.Tensor]:
        nonlocal materializations
        materializations += 1
        return original_materialize(**kwargs)

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        _count_materialization,
    )
    baseline_candidates, baseline_triggers = _build(attack_rows=rows)
    assert materializations == 1

    restored_edges: dict[str, dict[str, Any]] = {
        str(groups[0].key_str): baseline_triggers[0],
        str(groups[2].key_str): baseline_triggers[2],
    }
    mixed_events: list[dict[str, Any]] = []
    committed: list[str] = []
    mixed_candidates, mixed_triggers = _build(
        attack_rows=rows,
        restored=restored_edges,
        events=mixed_events,
        on_complete=lambda key, _row: committed.append(key),
    )
    assert materializations == 2
    assert _parquet_bytes(mixed_candidates) == _parquet_bytes(baseline_candidates)
    assert mixed_triggers == baseline_triggers
    assert committed == [groups[1].key_str]
    assert {event["group_index"] for event in mixed_events} == {2}

    all_restored: dict[str, dict[str, Any]] = {
        str(group.key_str): trigger
        for group, trigger in zip(groups, baseline_triggers, strict=True)
    }

    def _tripwire(*_args: Any, **_kwargs: Any) -> NoReturn:
        pytest.fail("an all-restored multi-group build reached model/search work")

    all_candidates, all_triggers = _build(
        attack_rows=rows,
        restored=all_restored,
        objective_fn=_tripwire,
        gradient_fn=_tripwire,
    )
    assert materializations == 2
    assert _parquet_bytes(all_candidates) == _parquet_bytes(baseline_candidates)
    assert all_triggers == baseline_triggers


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="not canonical JSON"):
        canonical_json_bytes({"bad": float("nan")})


def test_fingerprint_includes_semantics_and_excludes_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module,
        "_hash_file",
        lambda path, **_kwargs: canonical_sha256({"path": str(path)}),
    )
    monkeypatch.setattr(
        checkpoint_module,
        "tree_sha256",
        lambda path: canonical_sha256({"tree": str(path)}),
    )
    monkeypatch.setattr(checkpoint_module, "get_base_dir", lambda: str(tmp_path / "base"))
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        device_batch_size=4,
        diffmean_and_probe_layer=6,
        lora_adapter={"rank": 2},
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    faster = FasterGcgConfig(max_steps=1)
    groups = concept_groups(_attack_rows().to_pylist())
    target = ResolvedTargetCheckpoint(
        target_name="decoder_only",
        source="base",
        model_tag="d12",
        step=5,
        checkpoint_dir=tmp_path / "checkpoint",
        model_path=tmp_path / "checkpoint" / "model_000005.pt",
        metadata_path=tmp_path / "checkpoint" / "meta_000005.json",
        model_sha256="1" * 64,
        metadata_sha256="2" * 64,
    )

    def _fingerprint(
        current_cfg: AxBenchBpbAdvConfig,
        *,
        current_faster: FasterGcgConfig = faster,
        metadata: Mapping[str, Any] | None = None,
        current_target: ResolvedTargetCheckpoint = target,
        current_groups: Sequence[Any] = groups,
        cluster_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        return build_checkpoint_fingerprint(
            cfg=current_cfg,
            paths=paths,
            faster_cfg=current_faster,
            groups=current_groups,
            resolved_targets={"decoder_only": current_target},
            loaded_metadata={"decoder_only": metadata or {"model_config": {"depth": 12}}},
            cluster_feature_contract_fingerprint=cluster_fingerprint,
        )

    baseline = _fingerprint(cfg)
    assert baseline["algorithm_version"].endswith("checkpoint_v1")
    assert baseline["faster_gcg_config"] == asdict(faster)
    assert baseline["objective"] == {
        "device_batch_size": 4,
        "diffmean_and_probe_layer": 6,
        "next_segment_conditioning_offset": False,
        "lora_adapter": {"rank": 2},
    }
    assert baseline["targets"] == [
        {
            "target_name": "decoder_only",
            "source": "base",
            "model_tag": "d12",
            "step": 5,
            "model_sha256": "1" * 64,
            "metadata_sha256": "2" * 64,
            "effective_loaded_metadata_sha256": canonical_sha256(
                {"model_config": {"depth": 12}}
            ),
        }
    ]
    assert baseline["context_cache_tree_sha256"] == "unused"
    assert baseline["ours_gcg_bags_tree_sha256"] == "unused"
    assert baseline["expected_group_keys"] == {
        "count": 1,
        "sha256": canonical_sha256([groups[0].key_str]),
    }
    operational = deepcopy(cfg)
    operational.log_progress_every = 7
    operational.offload_between_stages = False
    operational.run_tag = "another-logical-path"
    operational.output_root = str(tmp_path / "other-output")
    operational.device = "mps"
    operational.profile = replace(operational.profile, enabled=True)
    operational.plot_methods = ["anything"]
    operational.scatter_baseline_x = "min"
    operational.scores_path = str(tmp_path / "scores.parquet")
    assert _fingerprint(operational) == baseline

    for field_name, value in (
        ("device_batch_size", 8),
        ("diffmean_and_probe_layer", 7),
        ("next_segment_conditioning_offset", True),
        ("lora_adapter", {"rank": 3}),
    ):
        changed_objective = deepcopy(cfg)
        setattr(changed_objective, field_name, value)
        assert _fingerprint(changed_objective) != baseline, field_name

    faster_variants = [
        replace(faster, optimized_token_counts=(0, 2)),
        replace(faster, max_steps=2),
        replace(faster, candidate_batch_size=2),
        replace(faster, top_k=2),
        replace(faster, distance_weight=1.0),
        replace(faster, temperature=0.5),
        replace(faster, master_seed=1),
        replace(faster, universal_minibatch_size=2),
        replace(faster, candidate_eval_device_batch_size=8),
    ]
    for variant in faster_variants:
        assert _fingerprint(cfg, current_faster=variant) != baseline

    assert _fingerprint(cfg, metadata={"model_config": {"depth": 13}}) != baseline
    assert (
        _fingerprint(
            cfg,
            metadata={
                "model_config": {"depth": 12},
                "checkpoint_dir": "/different/machine/path",
            },
        )
        == baseline
    )
    for target_variant in (
        replace(target, source="sft"),
        replace(target, model_tag="d13"),
        replace(target, step=6),
        replace(target, model_sha256="3" * 64),
        replace(target, metadata_sha256="4" * 64),
    ):
        assert _fingerprint(cfg, current_target=target_variant) != baseline
    assert _fingerprint(cfg, cluster_fingerprint="5" * 64) != baseline
    assert _fingerprint(cfg, current_groups=list(reversed(groups))) == baseline
    expanded_groups = concept_groups(_process_attack_rows().to_pylist())
    assert _fingerprint(cfg, current_groups=expanded_groups) != baseline


def test_every_artifact_digest_component_independently_changes_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        feature_statistics_path=str(tmp_path / "feature_statistics.safetensors"),
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    row = _attack_rows().to_pylist()[0]
    row["method_name"] = "ours_gcg_encdec"
    row["target_model"] = "encdec"
    groups = concept_groups([row])
    target = ResolvedTargetCheckpoint(
        target_name="encdec",
        source="encdec",
        model_tag="d12",
        step=5,
        checkpoint_dir=tmp_path / "checkpoint",
        model_path=tmp_path / "checkpoint" / "model_000005.pt",
        metadata_path=tmp_path / "checkpoint" / "meta_000005.json",
        model_sha256="1" * 64,
        metadata_sha256="2" * 64,
    )
    monkeypatch.setattr(checkpoint_module, "get_base_dir", lambda: str(tmp_path / "base"))
    file_values = {
        str(paths.source_contract()): "a" * 64,
        str(paths.attack_rows_path()): "b" * 64,
        str(paths.evaluation_population()): "c" * 64,
        str(Path(cfg.feature_statistics_path)): "d" * 64,
    }
    tree_values = {
        str(tmp_path / "base" / "tokenizer"): "e" * 64,
        str(paths.context_cache_dir()): "f" * 64,
        str(paths.ours_gcg_bags_dir()): "0" * 64,
    }
    monkeypatch.setattr(
        checkpoint_module,
        "_hash_file",
        lambda path, **_kwargs: file_values[str(path)],
    )
    monkeypatch.setattr(
        checkpoint_module,
        "tree_sha256",
        lambda path: tree_values[str(path)],
    )

    def _fingerprint() -> dict[str, Any]:
        return build_checkpoint_fingerprint(
            cfg=cfg,
            paths=paths,
            faster_cfg=FasterGcgConfig(max_steps=1),
            groups=groups,
            resolved_targets={"encdec": target},
            loaded_metadata={"encdec": {"model_config": {"depth": 12}}},
            cluster_feature_contract_fingerprint=None,
        )

    baseline = _fingerprint()
    assert baseline["source_contract_sha256"] == "a" * 64
    assert baseline["attack_rows_sha256"] == "b" * 64
    assert baseline["evaluation_population_sha256"] == "c" * 64
    assert baseline["feature_statistics_sha256"] == "d" * 64
    assert baseline["tokenizer_tree_sha256"] == "e" * 64
    assert baseline["context_cache_tree_sha256"] == "f" * 64
    assert baseline["ours_gcg_bags_tree_sha256"] == "0" * 64

    for mapping in (file_values, tree_values):
        for key, original in list(mapping.items()):
            mapping[key] = "9" * 64
            assert _fingerprint() != baseline, key
            mapping[key] = original


def test_duplicate_expected_fingerprint_groups_fail_before_artifact_use(
    tmp_path: Path,
) -> None:
    group = concept_groups(_attack_rows().to_pylist())[0]
    with pytest.raises(AxBenchBpbAdvError, match="group keys must be unique"):
        build_checkpoint_fingerprint(
            cfg=AxBenchBpbAdvConfig(feature_statistics_path=str(tmp_path / "stats")),
            paths=AxBenchBpbAdvPaths(output_root=str(tmp_path), run_tag="run"),
            faster_cfg=FasterGcgConfig(),
            groups=[group, group],
            resolved_targets={},
            loaded_metadata={},
            cluster_feature_contract_fingerprint=None,
        )


def test_fingerprint_requires_loaded_metadata_for_every_resolved_target(
    tmp_path: Path,
) -> None:
    target = ResolvedTargetCheckpoint(
        target_name="decoder_only",
        source="base",
        model_tag="d12",
        step=1,
        checkpoint_dir=tmp_path,
        model_path=tmp_path / "model",
        metadata_path=tmp_path / "meta",
        model_sha256="1" * 64,
        metadata_sha256="2" * 64,
    )
    with pytest.raises(AxBenchBpbAdvError, match="loaded metadata is missing"):
        build_checkpoint_fingerprint(
            cfg=AxBenchBpbAdvConfig(feature_statistics_path=str(tmp_path / "stats")),
            paths=AxBenchBpbAdvPaths(output_root=str(tmp_path), run_tag="run"),
            faster_cfg=FasterGcgConfig(),
            groups=concept_groups(_attack_rows().to_pylist()),
            resolved_targets={"decoder_only": target},
            loaded_metadata={},
            cluster_feature_contract_fingerprint=None,
        )


def test_runtime_fingerprint_uses_small_cpu_payload_and_cuda_delegate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    cpu_payload, cpu_hash = current_runtime_fingerprint("cpu")
    assert cpu_payload["device_type"] == "cpu"
    assert cpu_hash == canonical_sha256(cpu_payload)
    cuda_payload = {"gpu": "sentinel"}
    monkeypatch.setattr(
        checkpoint_module,
        "runtime_numeric_fingerprint",
        lambda device: cuda_payload if str(device) == "cuda:3" else pytest.fail(str(device)),
    )
    payload, digest = current_runtime_fingerprint("cuda:3")
    assert payload is cuda_payload
    assert digest == canonical_sha256(cuda_payload)


def test_runtime_mismatch_warns_without_manifest_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import steering.eval.axbench_bpb_adv.universal_gcg_checkpoint as checkpoint_module

    group = concept_groups(_attack_rows().to_pylist())[0]
    store = _store(tmp_path)
    stored_runtime = "1" * 64
    current_runtime = "2" * 64
    store.commit_trigger(group.key_str, _trigger_row(), stored_runtime)
    restored = _store(tmp_path)
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    monkeypatch.setattr(
        checkpoint_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )
    warn_on_runtime_mismatch(restored, current_runtime)
    assert len(warnings) == 1
    assert warnings[0][1] == (current_runtime, [stored_runtime])
    assert restored.load_triggers() == {group.key_str: _trigger_row()}
    warnings.clear()
    warn_on_runtime_mismatch(restored, stored_runtime)
    assert warnings == []


def _process_attack_rows() -> pa.Table:
    base = _attack_rows().to_pylist()[0]
    rows: list[dict[str, Any]] = []
    for concept_id in (1, 2, 3):
        row = dict(base)
        row["attack_row_id"] = f"row-{concept_id}"
        row["core_prefix_id"] = f"prefix-{concept_id:06d}"
        row["concept_rich_sample_id"] = f"rich-{concept_id}"
        row["concept_avoiding_sample_id"] = f"avoid-{concept_id}"
        row["concept_id"] = concept_id
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)


def _process_run(
    run_dir_str: str,
    *,
    stall_on_concept_two: bool,
    reject_replayed_concept_one: bool,
) -> None:
    run_dir = Path(run_dir_str)
    rows = _process_attack_rows()
    groups = concept_groups(rows.to_pylist())
    store = UniversalGcgCheckpointStore.prepare(
        checkpoint_dir=run_dir / "checkpoint",
        fingerprint={"process_contract": "v1"},
        expected_groups=groups,
        tokenizer=_Tokenizer(),
        max_steps=1,
    )
    marker = run_dir / "group-two-started"

    def _process_objective(
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
    ) -> list[float]:
        concept_id = int(items[0][0]["concept_id"])
        if reject_replayed_concept_one and concept_id == 1:
            raise RuntimeError("restored concept one was replayed")
        if stall_on_concept_two and concept_id == 2 and not marker.exists():
            marker.write_text("started", encoding="utf-8")
            while True:
                time.sleep(0.05)
        return _objective(items)

    weight = t.zeros(121, 2)
    weight[120] = t.tensor([-1.0, 0.0])

    def _commit(key: str, row: Mapping[str, Any]) -> None:
        store.commit_trigger(key, row, "1" * 64)

    candidates, triggers = build_universal_gcg_candidates(
        attack_rows=rows,
        tokenizer=_Tokenizer(),
        cfg=FasterGcgConfig(
            max_steps=1,
            candidate_batch_size=1,
            top_k=1,
            distance_weight=0.0,
            temperature=1.0,
        ),
        embedding_weight=weight,
        allowed_token_ids=[120],
        objective=_process_objective,
        gradient_provider_batch=_gradient,
        restored_triggers=store.load_triggers(),
        on_trigger_complete=_commit,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(candidates, run_dir / "candidates.parquet")
    pq.write_table(
        pa.Table.from_pylist(triggers, schema=UNIVERSAL_TRIGGERS_SCHEMA),
        run_dir / "universal_triggers.parquet",
    )


def _spawn_process_run(
    run_dir: Path,
    *,
    stall: bool,
    reject_replay: bool,
) -> Any:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_process_run,
        args=(str(run_dir),),
        kwargs={
            "stall_on_concept_two": stall,
            "reject_replayed_concept_one": reject_replay,
        },
    )
    process.start()
    return process


def test_sigkill_after_commit_restores_in_fresh_process_with_byte_parity(
    tmp_path: Path,
) -> None:
    resumed_dir = tmp_path / "resumed"
    interrupted = _spawn_process_run(resumed_dir, stall=True, reject_replay=False)
    deadline = time.monotonic() + 15.0
    marker = resumed_dir / "group-two-started"
    while time.monotonic() < deadline and not marker.exists():
        if interrupted.exitcode is not None:
            break
        time.sleep(0.05)
    assert marker.exists(), f"child exited early with code {interrupted.exitcode}"
    shards = list((resumed_dir / "checkpoint" / "groups").glob("*/*.parquet"))
    assert len(shards) == 1
    if interrupted.pid is None:
        pytest.fail("spawned child has no pid")
    os.kill(interrupted.pid, signal.SIGKILL)
    interrupted.join(timeout=10)
    assert interrupted.exitcode == -signal.SIGKILL

    resumed = _spawn_process_run(resumed_dir, stall=False, reject_replay=True)
    resumed.join(timeout=20)
    assert resumed.exitcode == 0

    control_dir = tmp_path / "control"
    control = _spawn_process_run(control_dir, stall=False, reject_replay=False)
    control.join(timeout=20)
    assert control.exitcode == 0
    for filename in ("candidates.parquet", "universal_triggers.parquet"):
        assert (resumed_dir / filename).read_bytes() == (control_dir / filename).read_bytes()
