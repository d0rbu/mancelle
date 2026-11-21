"""Unit tests for source-side (read-only) path resolvers."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_bpb import source_paths as _source_paths_module
from steering.eval.axbench_bpb.source_paths import (
    SourcePaths,
    validate_stage_0_3_artifacts,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def _positive_row(content_class: str, *, qc_accepted: bool = True) -> dict:
    avoiding = content_class == "concept_avoiding"
    return {
        "concept_id": 3,
        "category": "negative" if avoiding else "positive",
        "content_class": content_class,
        "prefix_text": "prefix",
        "output_text": "output",
        "output_concept": "concept",
        "n_prefix_nanochat_tokens": 2,
        "gen_source": "test",
        "gen_seed": 1,
        "qc_feature_id": 4,
        "qc_seed_percentile": float("nan") if avoiding else 10.0,
        "qc_output_percentile": float("nan") if avoiding else 50.0,
        "qc_delta": float("nan") if avoiding else 40.0,
        "qc_attempts": 1,
        "qc_accepted": qc_accepted,
        "qc_zero_activation_verified": avoiding,
    }


def _materialize_required_source(
    sp: SourcePaths,
    *,
    positive_rows: list[dict] | None = None,
    negative_rows: list[dict] | None = None,
) -> None:
    sp.concepts_json().write_text(
        json.dumps([{"concept_id": 3}]),
        encoding="utf-8",
    )
    positive_schema = schema_with_stage1_continuation_contract(
        schema_with_positive_qc_contract(
            TRAINING_SCHEMA,
            accept_threshold=0.3,
        ),
        role="stage1b_positive",
    )
    pq.write_table(
        pa.Table.from_pylist(
            positive_rows
            if positive_rows is not None
            else [
                _positive_row("concept_rich"),
                _positive_row("concept_avoiding"),
            ],
            schema=positive_schema,
        ),
        sp.synth_training_positive(),
    )
    negative_table = pa.Table.from_pylist(
            negative_rows
            if negative_rows is not None
            else [
                {
                    "concept_id": 3,
                    "content_class": "concept_rich",
                    "qc_accepted": True,
                },
                {
                    "concept_id": 3,
                    "content_class": "concept_avoiding",
                    "qc_accepted": True,
                },
            ],
        )
    negative_table = negative_table.cast(
        schema_with_stage1_continuation_contract(
            negative_table.schema,
            role="stage1c_negative",
        )
    )
    pq.write_table(negative_table, sp.synth_training_negative())
    _touch(sp.seeds_train())
    _touch(sp.seeds_eval())
    _touch(sp.seed_pool_manifest())
    _touch(sp.concept_features())
    _touch(sp.concept_features_neg())
    (sp.source_run_dir / "baselines").mkdir()


def test_source_paths_known_files(tmp_path):
    sp = SourcePaths(source_run_dir=tmp_path)
    assert sp.concepts_json() == tmp_path / "concepts_stage1c.json"
    assert sp.synth_training_positive() == tmp_path / "synth_training_positive.parquet"
    assert sp.synth_training_negative() == tmp_path / "synth_training_negative.parquet"
    assert sp.seeds_train() == tmp_path / "seeds_train.safetensors"
    assert sp.seeds_eval() == tmp_path / "seeds_eval.safetensors"
    assert sp.seed_pool_manifest() == tmp_path / "seed_pool_manifest.json"
    assert sp.concept_features() == tmp_path / "concept_features.safetensors"
    assert sp.concept_features_neg() == tmp_path / "concept_features_neg.safetensors"


def test_source_paths_cluster_artifact_files(tmp_path):
    """The four cluster-eval path builders (config.py's cluster_contract
    plumbing consumers) resolve to the expected filenames under
    source_run_dir. The oracle itself carries no test for these -- see
    docs/plans/2026-07-24-pr739-track-f-plan-part1.md Test design table
    ("oracle has none - closing a real gap")."""
    sp = SourcePaths(source_run_dir=tmp_path)
    assert sp.feature_contract() == tmp_path / "feature_contract.json"
    assert sp.candidate_concepts() == tmp_path / "candidate_concepts.json"
    assert sp.candidate_cluster_provenance() == (
        tmp_path / "candidate_concepts.cluster_provenance.json"
    )
    assert sp.negative_mode_prefixes() == tmp_path / "negative_mode_prefixes.parquet"


def test_cluster_artifact_paths_are_read_only_no_side_effects(tmp_path):
    """Calling the four new path builders must not create the file/dir
    they point at -- source_paths.py's whole-module contract (see its
    docstring) is read-only path resolution, never writes."""
    sp = SourcePaths(source_run_dir=tmp_path)
    sp.feature_contract()
    sp.candidate_concepts()
    sp.candidate_cluster_provenance()
    sp.negative_mode_prefixes()
    assert list(tmp_path.iterdir()) == []


def test_baseline_direction_paths(tmp_path):
    sp = SourcePaths(source_run_dir=tmp_path)
    assert sp.diffmean_direction("encdec", "positive", 42) == (
        tmp_path / "baselines" / "diffmean" / "encdec" / "positive" / "concept_0042.safetensors"
    )
    assert sp.linear_probe_direction("decoder_only", "negative", 7) == (
        tmp_path
        / "baselines"
        / "linear_probe"
        / "decoder_only"
        / "negative"
        / "concept_0007.safetensors"
    )
    assert sp.lora_adapter("positive", 99) == (
        tmp_path / "baselines" / "lora" / "decoder_only" / "positive" / "concept_0099.pt"
    )
    assert sp.prompting_descriptions("negative") == (
        tmp_path / "baselines" / "prompting" / "negative_descriptions.json"
    )


def test_validate_stage_0_3_missing_files(tmp_path):
    sp = SourcePaths(source_run_dir=tmp_path)
    with pytest.raises(AxBenchBpbError, match="missing"):
        validate_stage_0_3_artifacts(sp)


def test_validate_stage_0_3_all_present(tmp_path):
    sp = SourcePaths(source_run_dir=tmp_path)
    _materialize_required_source(sp)

    validate_stage_0_3_artifacts(sp)  # must not raise


def test_validate_stage_0_3_rejects_missing_final_concept_class(tmp_path):
    sp = SourcePaths(source_run_dir=tmp_path)
    _materialize_required_source(
        sp,
        negative_rows=[
            {
                "concept_id": 3,
                "content_class": "concept_rich",
                "qc_accepted": True,
            }
        ],
    )

    with pytest.raises(
        AxBenchBpbError,
        match=r"authoritative filtered concepts.*negative.*concept_avoiding",
    ):
        validate_stage_0_3_artifacts(sp)


def test_validate_stage_0_3_applies_configured_qc_filter(tmp_path):
    sp = SourcePaths(source_run_dir=tmp_path)
    rows = [
        _positive_row("concept_rich", qc_accepted=False),
        _positive_row("concept_avoiding"),
    ]
    _materialize_required_source(sp, positive_rows=rows)

    with pytest.raises(AxBenchBpbError, match="positive.*concept_rich"):
        validate_stage_0_3_artifacts(sp, filter_unaccepted_samples=True)
    validate_stage_0_3_artifacts(sp, filter_unaccepted_samples=False)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"concept_id": True}],
        [{"concept_id": 3}, {"concept_id": 3}],
        [{"concept_id": 4}, {"concept_id": 3}],
    ],
)
def test_validate_stage_0_3_rejects_invalid_final_population_identity(
    tmp_path,
    payload,
):
    sp = SourcePaths(source_run_dir=tmp_path)
    _materialize_required_source(sp)
    sp.concepts_json().write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBpbError, match="final source concepts"):
        validate_stage_0_3_artifacts(sp)


def test_no_write_methods_on_source_paths():
    """Source paths must not offer write helpers — read-only contract."""
    # Verify the module exposes no function with 'write' in its name
    assert not any("write" in name.lower() for name in dir(_source_paths_module))
