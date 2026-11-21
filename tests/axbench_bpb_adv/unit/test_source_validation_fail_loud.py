"""Fail-loud both-classes-per-prefix source validation for BPB-Adv."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.source_validation import (
    _find_incomplete_class_prefixes,
    validate_stage_0_3_source_run,
)
from tests.axbench_bpb_adv.source_run_builders import (
    both_class_rows,
    one_sided_rows,
    synth_row,
    write_stage_0_3_source_run,
)


def test_find_incomplete_class_prefixes_for_explicit_cleaning() -> None:
    rows = (
        both_class_rows("ok")
        + one_sided_rows("bad-rich", content_class="concept_rich")
        + one_sided_rows("bad-avoid", content_class="concept_avoiding")
    )

    assert _find_incomplete_class_prefixes(rows) == {
        (1, "bad-rich"),
        (1, "bad-avoid"),
    }


def test_validate_source_raises_on_incomplete_prefix(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("ok") + one_sided_rows("bad", content_class="concept_rich"),
        negative_rows=both_class_rows("neg-ok"),
    )

    with pytest.raises(AxBenchBpbAdvError, match="bad"):
        validate_stage_0_3_source_run(root)


def test_validate_source_complete_records_zero_dropped(tmp_path: Path) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("p1") + both_class_rows("p2"),
        negative_rows=both_class_rows("n1"),
    )

    validation = validate_stage_0_3_source_run(root)

    assert validation.n_positive_prefixes == 2
    assert validation.n_negative_prefixes == 1
    assert validation.n_dropped == 0
    assert validation.contract.n_dropped == 0
    assert validation.contract.to_json_dict()["n_dropped"] == 0


def test_validate_source_qc_filter_can_make_prefix_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        positive_rows=[
            synth_row("qc", content_class="concept_rich", qc_accepted=False),
            synth_row("qc", content_class="concept_avoiding"),
        ],
        negative_rows=both_class_rows("neg-ok"),
    )

    with pytest.raises(AxBenchBpbAdvError, match="qc"):
        validate_stage_0_3_source_run(root)

    validation = validate_stage_0_3_source_run(root, filter_unaccepted_samples=False)
    assert validation.n_positive_prefixes == 1
    assert validation.n_dropped == 0


def test_validate_source_filters_out_of_concepts_before_completeness_check(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        concepts=[{"concept_id": 1, "description": "kept"}],
        positive_rows=both_class_rows("ok", concept_id=1)
        + one_sided_rows("foreign", content_class="concept_rich", concept_id=99),
        negative_rows=both_class_rows("neg-ok", concept_id=1),
    )

    validation = validate_stage_0_3_source_run(root)

    assert validation.n_positive_prefixes == 1
    assert validation.n_dropped == 0


def test_validate_source_rejects_authoritative_concept_with_no_rows(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        concepts=[
            {"concept_id": 1, "description": "present"},
            {"concept_id": 2, "description": "absent"},
        ],
        positive_rows=both_class_rows("positive", concept_id=1),
        negative_rows=both_class_rows("negative", concept_id=1),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"authoritative final concept IDs \[2\]",
    ):
        validate_stage_0_3_source_run(root)


def test_stage0c_only_concept_is_invisible_to_bpb_adv_after_stage1c_drop(
    tmp_path: Path,
) -> None:
    """BPB-Adv must read the final list, never immutable Stage 0c provenance."""

    root = tmp_path / "source"
    write_stage_0_3_source_run(
        root,
        concepts=[{"concept_id": 1, "description": "survivor"}],
        positive_rows=(
            both_class_rows("survivor-positive", concept_id=1)
            + both_class_rows("dropped-positive", concept_id=99)
        ),
        negative_rows=both_class_rows("survivor-negative", concept_id=1),
    )
    (root / "concepts.json").write_text(
        json.dumps(
            [
                {"concept_id": 1, "description": "survivor"},
                {"concept_id": 99, "description": "Stage 0c only"},
            ]
        ),
        encoding="utf-8",
    )

    validation = validate_stage_0_3_source_run(root)

    assert validation.concept_ids == frozenset({1})
    assert {int(row["concept_id"]) for row in validation.positive_rows()} == {1}
    assert "dropped-positive" not in validation.context_prefixes()
