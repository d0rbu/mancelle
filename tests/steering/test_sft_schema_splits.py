"""Typed SFT split, conversion-metadata, and suffix-confinement contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from steering.sft_schema import (
    SFTConversionMeta,
    SFTSplit,
    SFTSplitSelection,
    sft_split_dir,
)


@pytest.mark.parametrize("split", list(SFTSplit))
def test_sft_split_parse_round_trip(split: SFTSplit) -> None:
    assert SFTSplit.parse(split) is split
    assert SFTSplit.parse(split.value) is split


@pytest.mark.parametrize("raw", ["junk", "TRAIN", "both", None, 1])
def test_sft_split_parse_rejects_junk_naming_valid_values(raw: object) -> None:
    with pytest.raises(ValueError, match=r"Invalid SFT split.*'train'.*'val'"):
        SFTSplit.parse(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(SFTSplit.TRAIN, SFTSplitSelection.TRAIN), (SFTSplit.VAL, SFTSplitSelection.VAL)],
)
def test_sft_split_selection_parse_maps_sft_split_explicitly(
    raw: SFTSplit,
    expected: SFTSplitSelection,
) -> None:
    assert SFTSplitSelection.parse(raw) is expected


@pytest.mark.parametrize(
    "raw",
    [SFTSplitSelection.TRAIN, SFTSplitSelection.VAL, SFTSplitSelection.BOTH],
)
def test_sft_split_parse_rejects_sft_split_selection(raw: SFTSplitSelection) -> None:
    with pytest.raises(ValueError, match=r"Invalid SFT split.*'train'.*'val'"):
        SFTSplit.parse(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("train", SFTSplitSelection.TRAIN),
        ("val", SFTSplitSelection.VAL),
        ("both", SFTSplitSelection.BOTH),
    ],
)
def test_sft_split_selection_parse_round_trip(
    raw: str,
    expected: SFTSplitSelection,
) -> None:
    assert SFTSplitSelection.parse(raw) is expected
    assert SFTSplitSelection.parse(expected) is expected


@pytest.mark.parametrize("raw", ["junk", "TRAIN", None, 1])
def test_sft_split_selection_rejects_junk_naming_valid_values(raw: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"Invalid SFT split selection.*'train'.*'val'.*'both'",
    ):
        SFTSplitSelection.parse(raw)


def test_sft_split_selection_membership_and_order() -> None:
    assert SFTSplitSelection.BOTH.splits() == (SFTSplit.TRAIN, SFTSplit.VAL)
    assert SFTSplitSelection.TRAIN.splits() == (SFTSplit.TRAIN,)
    assert SFTSplitSelection.VAL.splits() == (SFTSplit.VAL,)
    assert SFTSplitSelection.TRAIN in SFTSplitSelection.BOTH
    assert SFTSplitSelection.VAL in SFTSplitSelection.BOTH
    assert SFTSplitSelection.VAL not in SFTSplitSelection.TRAIN
    assert SFTSplitSelection.TRAIN not in SFTSplitSelection.VAL


def test_sft_split_dir_both_arms() -> None:
    train_dir = Path("/tmp/sft/safety")
    assert sft_split_dir(train_dir, SFTSplit.TRAIN) == train_dir
    assert sft_split_dir(str(train_dir), SFTSplit.VAL) == Path("/tmp/sft/safety_val")


def _metadata(*, split: object = "val") -> dict[str, object]:
    return {
        "mixture": "safety",
        "split": split,
        "format": "full_nanochat_text",
        "completed_shards": 2,
        "total_shards": 3,
        "total_conversations": 100,
        "future_key": {"kept": True},
    }


def test_conversion_meta_parse_and_serialize_preserve_bytes_and_passthrough() -> None:
    raw = _metadata()
    meta = SFTConversionMeta.parse(raw)

    assert meta.mixture == "safety"
    assert meta.split is SFTSplit.VAL
    assert meta.format == "full_nanochat_text"
    assert meta.completed_shards == 2
    assert meta.total_shards == 3
    assert meta.total_conversations == 100
    assert list(meta.to_json_dict()) == list(raw)
    assert json.dumps(meta.to_json_dict(), indent=2) == json.dumps(raw, indent=2)


def test_conversion_meta_parse_legacy_split_absent_round_trip() -> None:
    raw = _metadata()
    del raw["split"]

    meta = SFTConversionMeta.parse(raw)

    assert meta.split is None
    assert meta.to_json_dict() == raw
    assert "split" not in meta.to_json_dict()


@pytest.mark.parametrize("split", ["bogus", "both", 1, False, []])
def test_conversion_meta_rejects_malformed_split(split: object) -> None:
    with pytest.raises(ValueError, match=r"Invalid SFT split.*'train'.*'val'"):
        SFTConversionMeta.parse(_metadata(split=split))


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        (
            "mixture",
            None,
            "SFT conversion metadata field 'mixture' must be a string, got None.",
        ),
        (
            "format",
            [],
            "SFT conversion metadata field 'format' must be a string, got [].",
        ),
        (
            "completed_shards",
            True,
            "SFT conversion metadata field 'completed_shards' must be an integer, got True.",
        ),
        (
            "total_shards",
            False,
            "SFT conversion metadata field 'total_shards' must be an integer, got False.",
        ),
        (
            "total_conversations",
            "100",
            "SFT conversion metadata field 'total_conversations' must be an integer, got '100'.",
        ),
    ],
)
def test_conversion_meta_rejects_malformed_required_fields(
    field: str,
    value: object,
    expected_message: str,
) -> None:
    raw = _metadata()
    raw[field] = value
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        SFTConversionMeta.parse(raw)


def test_val_suffix_convention_is_confined_to_sft_schema() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "steering" / "sft_schema.py"
    production_paths = sorted((repo_root / "scripts").rglob("*.py")) + sorted(
        (repo_root / "steering").rglob("*.py")
    )
    forbidden_patterns = {
        "endswith": re.compile(r"endswith\(\s*(['\"])_val\1\s*\)"),
        "removesuffix": re.compile(r"removesuffix\(\s*(['\"])_val\1\s*\)"),
        "rstrip": re.compile(r"rstrip\(\s*(['\"])_val\1\s*\)"),
        "lstrip": re.compile(r"lstrip\(\s*(['\"])_val\1\s*\)"),
        "strip": re.compile(r"strip\(\s*(['\"])_val\1\s*\)"),
        "f-string composition": re.compile(
            r"""f(?:"[^"\n]*\{[^}\n]+\}_val"|'[^'\n]*\{[^}\n]+\}_val')"""
        ),
        "concatenation": re.compile(r"""(?:["']_val["']\s*\+|\+\s*["']_val["'])"""),
    }

    violations: list[str] = []
    definition_sites: list[str] = []
    for path in production_paths:
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(repo_root)
        if re.search(r"^def sft_split_dir\(", text, flags=re.MULTILINE):
            definition_sites.append(str(relative))
        if path == schema_path:
            continue
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(f"{relative}: {label}")

    assert violations == []
    assert definition_sites == ["steering/sft_schema.py"]
