"""Independent adversarial review tests for the issue-1673 NL source attachment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from steering.eval.axbench_adv.best_point import load_steering_entries
from steering.eval.axbench_adv.errors import AxBenchAdvError
from tests.axbench_adv.unit import test_nl_source_attachment as fixtures


def test_attachment_rejects_summary_ratings_inconsistent_with_raw_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = fixtures._case(tmp_path, monkeypatch)

    entries = fixtures._entries()
    results = cast(dict[str, Any], entries[0]["results"])
    evaluator = cast(dict[str, Any], results["LMJudgeEvaluator"])
    evaluator["NLSteering"][
        "relevance_concept_ratings"
    ][0] = 1.5
    fixtures._write_jsonl(case["paths"].evaluated_jsonl, entries)

    with pytest.raises(AxBenchAdvError, match="summary.*inconsistent"):
        fixtures._validate(case)


def test_attachment_rejects_duplicate_keys_in_project_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = fixtures._case(tmp_path, monkeypatch)
    path = case["paths"].project_manifest
    original = path.read_text(encoding="utf-8")
    assert original.startswith("{")
    path.write_text(
        '{"methods":{} ,' + original[1:],
        encoding="utf-8",
    )
    # Python's default JSON loader silently keeps the later methods mapping.
    assert "NLSteering" in json.loads(path.read_text(encoding="utf-8"))["methods"]
    with pytest.raises(AxBenchAdvError, match="not valid JSON"):
        fixtures._validate(case)


def test_evaluated_jsonl_binds_each_summary_to_its_named_raw_ratings() -> None:
    entries = fixtures._entries()
    results = cast(dict[str, Any], entries[0]["results"])
    evaluator = cast(dict[str, Any], results["LMJudgeEvaluator"])
    block = cast(dict[str, Any], evaluator["NLSteering"])
    block["raw_relevance_concept_ratings"], block[
        "raw_relevance_instruction_ratings"
    ] = (
        block["raw_relevance_instruction_ratings"],
        block["raw_relevance_concept_ratings"],
    )

    with pytest.raises(
        AxBenchAdvError,
        match=r"NLSteering\.relevance_concept_ratings summary is inconsistent",
    ):
        fixtures.contract._validate_evaluated_jsonl(
            entries,
            max_concepts=100,
            expected_methods=fixtures.JUDGED_METHODS,
        )


@pytest.mark.parametrize("mutation", ["boolean", "out_of_order"])
def test_evaluated_jsonl_rejects_malformed_concept_coordinates(
    tmp_path: Path, mutation: str
) -> None:
    entries = fixtures._entries()
    if mutation == "boolean":
        entries[0]["concept_id"] = True
    else:
        entries[0], entries[1] = entries[1], entries[0]

    path = tmp_path / "evaluated.jsonl"
    fixtures._write_jsonl(path, entries)
    with pytest.raises(AxBenchAdvError, match="each concept exactly once"):
        load_steering_entries(
            path,
            expected_concept_ids=range(100),
        )
