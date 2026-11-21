"""Original-message grammar, split integrity and fixed-range source preparation."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_negative.source import (
    ULTRACHAT_TEST_FILE,
    FeatureTarget,
    NegativeSourceError,
    SourceCandidate,
    canonical_bytes,
    file_sha256,
    iter_assistant_prefixes,
    load_source_candidates,
    prepare_ultrachat_source,
    validate_messages,
)
from steering.immutable_io import ImmutableArtifactError


def record(**changes):
    return {
        "source_id": "a",
        "group_id": "family-a",
        "split": "test",
        "source_row": 0,
        "messages": [
            {"role": "system", "content": "  Preserve Ω.\n"},
            {"role": "user", "content": "first?\n "},
            {"role": "assistant", "content": " source answer "},
            {"role": "user", "content": "second?"},
            {"role": "assistant", "content": "future answer"},
        ],
        **changes,
    }


def write_records(path: Path, rows):
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))


def test_cuts_preserve_exact_history_and_omit_future():
    original = record()
    cuts = list(iter_assistant_prefixes(original))
    assert [cut.cut_message_count for cut in cuts] == [2, 4]
    assert list(cuts[0].messages) == original["messages"][:2]
    assert list(cuts[1].messages) == original["messages"][:4]
    assert "future answer" not in json.dumps([cut.to_dict() for cut in cuts])
    assert all(SourceCandidate.from_dict(cut.to_dict()) == cut for cut in cuts)
    serialized = cuts[0].to_dict()
    restored = SourceCandidate.from_dict(serialized)
    serialized["messages"][0]["content"] = "modified deserializer input"
    assert restored == cuts[0]
    assert len({cut.candidate_id for cut in cuts}) == 2
    original["messages"][0]["content"] = "modified caller object"
    assert cuts[0].messages[0]["content"] == "  Preserve Ω.\n"
    cuts[0].messages[0]["content"] = "illegal mutation"
    with pytest.raises(NegativeSourceError, match="digest"):
        cuts[0].to_dict()


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "system only"}],
        [{"role": "assistant", "content": "no user"}],
        [{"role": "user", "content": "   "}],
        [{"role": "user", "content": [{"type": "text", "text": "multipart"}]}],
        [{"role": "user", "content": "x", "name": "hidden"}],
        [{0: "user", "content": "wrong key type"}],
        [{"role": 1, "content": "wrong role type"}],
        [{"role": "user", "content": "x"}, {"role": "tool", "content": "result"}],
        [{"role": "user", "content": "x"}, {"role": "user", "content": "twice"}],
    ],
)
def test_message_grammar_rejects_repairs(messages):
    with pytest.raises(NegativeSourceError):
        validate_messages(messages)


@pytest.mark.parametrize(
    "change",
    [
        {"split": "validation"},
        {"split": "validation", "group_id": "different-family"},
    ],
)
def test_split_leakage_checks_families_and_exact_prefixes(tmp_path, change):
    path = tmp_path / "source.jsonl"
    other = record(source_id="b", **change)
    other["messages"][-1]["content"] = "different future cannot hide prefix leakage"
    write_records(path, [record(), other])
    with pytest.raises(NegativeSourceError, match="crosses splits"):
        load_source_candidates(path)


def test_declared_hash_duplicate_source_and_strict_json(tmp_path):
    path = tmp_path / "source.jsonl"
    write_records(path, [record()])
    assert len(load_source_candidates(path, file_sha256(path))) == 2
    with pytest.raises(NegativeSourceError, match="SHA-256"):
        load_source_candidates(path, "0" * 64)
    write_records(path, [record(), record()])
    with pytest.raises(NegativeSourceError, match="duplicate source_id"):
        load_source_candidates(path)
    for raw in ['{"source_id":"x","source_id":"y"}', '{"source_row":NaN}', "\n"]:
        path.write_text(raw)
        with pytest.raises(NegativeSourceError):
            load_source_candidates(path)


@pytest.mark.parametrize("members", [(1, 0), (1, 1), (), (True,), (2,)])
def test_feature_members_are_canonical(members):
    with pytest.raises(NegativeSourceError):
        FeatureTarget(1, members, "concept")


def test_feature_target_roundtrip():
    target = FeatureTarget(1, (1, 2, 3), "same feature union")
    assert FeatureTarget.from_dict(target.to_dict()) == target


def test_prepare_fixed_first_n_rows_never_refills_rejection(tmp_path):
    snapshot = tmp_path / "snapshot"
    parquet = snapshot / ULTRACHAT_TEST_FILE
    parquet.parent.mkdir(parents=True)
    rows = [
        {"prompt_id": "a", "messages": record()["messages"]},
        {"prompt_id": "b", "messages": [{"role": "user", "content": ""}]},
        {"prompt_id": "c", "messages": [{"role": "user", "content": "not selected"}]},
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    kwargs: dict[str, Any] = {
        "output_dir": tmp_path / "prepared",
        "row_count": 2,
        "snapshot_dir": snapshot,
        "expected_parquet_sha256": file_sha256(parquet),
    }
    manifest = prepare_ultrachat_source(**kwargs)
    assert manifest["selection"] == {"policy": "first_n_original_rows_v1", "start": 0, "stop": 2}
    assert [row["source_row"] for row in manifest["selected_rows"]] == [0, 1]
    assert [row["source_row"] for row in manifest["rejected_rows"]] == [1]
    assert manifest["source"]["rows"] == 1
    assert {
        c.source_row for c in load_source_candidates(kwargs["output_dir"] / "source.jsonl")
    } == {0}
    assert prepare_ultrachat_source(**kwargs) == manifest
    expanded: dict[str, Any] = {**kwargs, "row_count": 3}
    with pytest.raises(ImmutableArtifactError, match="differs"):
        prepare_ultrachat_source(**expanded)
    wrong_hash: dict[str, Any] = {**kwargs, "expected_parquet_sha256": "0" * 64}
    with pytest.raises(NegativeSourceError, match="SHA-256"):
        prepare_ultrachat_source(**wrong_hash)
    overflow: dict[str, Any] = {**kwargs, "row_count": 4}
    with pytest.raises(NegativeSourceError, match="exceeds"):
        prepare_ultrachat_source(**overflow)
