"""Tests for filter-derived AxBench-BPB source cleaning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

import steering.eval.axbench_bpb_adv.source_cleaning as source_cleaning
from steering.eval.axbench_bpb.sample_loader import load_raw_synth_rows
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.source_cleaning import clean_axbench_source_run
from steering.eval.axbench_bpb_adv.source_validation import (
    _filter_to_source_concepts,
    _find_incomplete_class_prefixes,
    _read_source_concept_ids,
    validate_stage_0_3_source_run,
)
from tests.axbench_bpb_adv.source_run_builders import (
    both_class_rows,
    one_sided_rows,
    synth_row,
    write_stage_0_3_source_run,
)


def _write_dirty_source(root: Path) -> None:
    concepts = [
        {"concept_id": 1, "description": "one"},
        {"concept_id": 2, "description": "two"},
    ]
    positive_rows = (
        both_class_rows("p-ok", concept_id=1)
        + both_class_rows("p-two", concept_id=2)
        + one_sided_rows("p-rich-only", content_class="concept_rich", concept_id=1)
        + [
            synth_row(
                "p-qc-induced",
                content_class="concept_rich",
                concept_id=1,
                qc_accepted=False,
            ),
            synth_row(
                "p-qc-induced",
                content_class="concept_avoiding",
                concept_id=1,
            ),
        ]
        + both_class_rows("p-out-of-concepts", concept_id=99)
    )
    negative_rows = (
        both_class_rows("n-one", concept_id=1)
        + both_class_rows("n-ok", concept_id=2)
        + one_sided_rows(
            "n-avoid-only",
            content_class="concept_avoiding",
            concept_id=2,
        )
    )
    write_stage_0_3_source_run(
        root,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        concepts=concepts,
    )


def _validator_dropped_keys(path: Path) -> set[tuple[int, str]]:
    source_paths = SourcePaths(path.parent)
    rows = _filter_to_source_concepts(
        load_raw_synth_rows(path, filter_unaccepted=True),
        concept_ids=_read_source_concept_ids(source_paths.concepts_json()),
    )
    return _find_incomplete_class_prefixes(rows)


def test_cleaner_matches_validator_dropped_sets_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    _write_dirty_source(source)

    result = clean_axbench_source_run(source, output_dir=output)

    assert result.validation.n_dropped == 0
    assert result.source_inspection.n_dropped_incomplete_groups == 3
    assert result.positive.dropped_incomplete_groups == (
        {"concept_id": 1, "prefix_text": "p-qc-induced"},
        {"concept_id": 1, "prefix_text": "p-rich-only"},
    )
    assert result.negative.dropped_incomplete_groups == (
        {"concept_id": 2, "prefix_text": "n-avoid-only"},
    )
    assert {
        (entry["concept_id"], entry["prefix_text"])
        for entry in result.positive.dropped_incomplete_groups
    } == _validator_dropped_keys(
        source / "synth_training_positive.parquet",
    )
    assert {
        (entry["concept_id"], entry["prefix_text"])
        for entry in result.negative.dropped_incomplete_groups
    } == _validator_dropped_keys(
        source / "synth_training_negative.parquet",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["source_inspection"]["n_dropped_incomplete_groups"] == 3
    assert manifest["self_verification"]["n_dropped"] == 0
    assert manifest["self_verification"]["contract_hash"] == result.validation.contract_hash
    assert manifest["splits"]["positive"]["qc_rejected_rows"] == 1
    assert manifest["splits"]["positive"]["out_of_source_concept_rows"] == 2
    assert manifest["splits"]["positive"]["incomplete_group_rows"] == 2
    assert manifest["splits"]["negative"]["incomplete_group_rows"] == 1


def test_cleaner_preserves_schema_order_and_non_synth_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    _write_dirty_source(source)

    clean_axbench_source_run(source, output_dir=output)

    source_table = pq.read_table(source / "synth_training_positive.parquet")
    output_table = pq.read_table(output / "synth_training_positive.parquet")
    assert output_table.schema.equals(source_table.schema)
    assert [
        (row["prefix_text"], row["content_class"])
        for row in output_table.to_pylist()
    ] == [
        ("p-ok", "concept_rich"),
        ("p-ok", "concept_avoiding"),
        ("p-two", "concept_rich"),
        ("p-two", "concept_avoiding"),
    ]
    for name in (
        "concepts.json",
        "concepts_stage1c.json",
        "seeds_train.safetensors",
        "seeds_eval.safetensors",
        "seed_pool_manifest.json",
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
        "baselines/prompting/positive_descriptions.json",
    ):
        assert (output / name).read_bytes() == (source / name).read_bytes()
    validation = validate_stage_0_3_source_run(output)
    source_seed_manifest = json.loads(
        (source / "seed_pool_manifest.json").read_text(encoding="utf-8")
    )
    assert validation.n_dropped == 0
    assert validation.contract.seed_fingerprint == source_seed_manifest[
        "seed_fingerprint"
    ]
    assert validation.contract.seed_schema_version == source_seed_manifest[
        "schema_version"
    ]


def test_cleaner_copies_non_synth_artifacts_independently(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    _write_dirty_source(source)

    clean_axbench_source_run(source, output_dir=output)

    original = (source / "concepts_stage1c.json").read_text(encoding="utf-8")
    (output / "concepts_stage1c.json").write_text("[]", encoding="utf-8")

    assert (
        source / "concepts_stage1c.json"
    ).read_text(encoding="utf-8") == original


def test_cleaner_is_idempotent_on_cleaned_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    first = tmp_path / "source_paired"
    second = tmp_path / "source_paired_again"
    _write_dirty_source(source)

    clean_axbench_source_run(source, output_dir=first)
    result = clean_axbench_source_run(first, output_dir=second)

    assert result.source_inspection.n_dropped_incomplete_groups == 0
    for name in ("synth_training_positive.parquet", "synth_training_negative.parquet"):
        first_table = pq.read_table(first / name)
        second_table = pq.read_table(second / name)
        assert second_table.schema.equals(first_table.schema)
        assert second_table.to_pylist() == first_table.to_pylist()


def test_cleaner_can_match_no_filter_unaccepted_configs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    write_stage_0_3_source_run(
        source,
        positive_rows=[
            synth_row(
                "p-qc-kept",
                content_class="concept_rich",
                qc_accepted=False,
            ),
            synth_row("p-qc-kept", content_class="concept_avoiding"),
        ],
        negative_rows=both_class_rows("n-ok"),
    )

    result = clean_axbench_source_run(
        source,
        output_dir=output,
        filter_unaccepted_samples=False,
    )

    assert result.source_inspection.n_dropped_incomplete_groups == 0
    assert result.validation.n_dropped == 0
    positive_rows = pq.read_table(output / "synth_training_positive.parquet").to_pylist()
    assert [row["qc_accepted"] for row in positive_rows] == [False, True]
    assert (
        validate_stage_0_3_source_run(
            output,
            filter_unaccepted_samples=False,
        ).n_dropped
        == 0
    )


def test_cleaner_rejects_unsafe_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_dirty_source(source)
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "marker.txt").write_text("x", encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="must differ"):
        clean_axbench_source_run(source, output_dir=source)
    with pytest.raises(AxBenchBpbAdvError, match="must not be inside"):
        clean_axbench_source_run(source, output_dir=source / "nested")
    with pytest.raises(AxBenchBpbAdvError, match="not empty"):
        clean_axbench_source_run(source, output_dir=nonempty)


def test_cleaner_rejects_split_with_no_complete_pairs_and_leaves_no_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    write_stage_0_3_source_run(
        source,
        positive_rows=(
            one_sided_rows("p-rich-only", content_class="concept_rich")
            + one_sided_rows("p-avoid-only", content_class="concept_avoiding")
        ),
        negative_rows=both_class_rows("n-ok"),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="synth_training_positive.parquet.*no complete class-margin pairs",
    ):
        clean_axbench_source_run(source, output_dir=output)

    assert not output.exists()


def test_cleaner_rejects_final_concept_absent_from_one_cleaned_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    concepts = [
        {"concept_id": 1, "description": "one"},
        {"concept_id": 2, "description": "two"},
    ]
    write_stage_0_3_source_run(
        source,
        concepts=concepts,
        positive_rows=(
            both_class_rows("p-one", concept_id=1)
            + one_sided_rows(
                "p-two-rich-only",
                content_class="concept_rich",
                concept_id=2,
            )
        ),
        negative_rows=(
            both_class_rows("n-one", concept_id=1)
            + both_class_rows("n-two", concept_id=2)
        ),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"positive mode.*no accepted source rows.*\[2\]",
    ):
        clean_axbench_source_run(source, output_dir=output)

    assert not output.exists()


def test_cleaner_failure_does_not_leave_output_that_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    _write_dirty_source(source)
    real_clean = source_cleaning._clean_synth_parquet

    def _fail_negative_once(**kwargs: Any):
        mode = kwargs.get("mode")
        if mode is None:
            mode = kwargs["split"].mode
        if mode == SteeringMode.NEGATIVE:
            raise AxBenchBpbAdvError("simulated clean failure")
        return real_clean(**kwargs)

    monkeypatch.setattr(source_cleaning, "_clean_synth_parquet", _fail_negative_once)
    with pytest.raises(AxBenchBpbAdvError, match="simulated clean failure"):
        clean_axbench_source_run(source, output_dir=output)

    assert not output.exists()
    monkeypatch.setattr(source_cleaning, "_clean_synth_parquet", real_clean)
    result = clean_axbench_source_run(source, output_dir=output)
    assert result.validation.n_dropped == 0


def test_cleaner_internal_self_check_reports_bad_cleaned_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "source_paired"
    _write_dirty_source(source)
    real_clean = source_cleaning._clean_synth_parquet

    def _bad_positive_clean(**kwargs: Any):
        mode = kwargs.get("mode")
        if mode is None:
            mode = kwargs["split"].mode
        if mode != SteeringMode.POSITIVE:
            return real_clean(**kwargs)
        source_path = kwargs.get("source_path")
        output_path = kwargs["output_path"]
        split = kwargs.get("split")
        raw_table = split.table if split is not None else pq.read_table(source_path)
        source_cleaning.atomic_write_parquet(raw_table, output_path)
        return source_cleaning.SourceCleaningSplitResult(
            table_name=output_path.name,
            mode=SteeringMode.POSITIVE.value,
            rows_in=raw_table.num_rows,
            rows_out=raw_table.num_rows,
            qc_rejected_rows=0,
            out_of_source_concept_rows=0,
            incomplete_group_rows=0,
            dropped_incomplete_groups=(),
        )

    monkeypatch.setattr(source_cleaning, "_clean_synth_parquet", _bad_positive_clean)
    with pytest.raises(
        AxBenchBpbAdvError,
        match="internal source-cleaning error: cleaned output still has .* incomplete group",
    ):
        clean_axbench_source_run(source, output_dir=output)

    assert not output.exists()
