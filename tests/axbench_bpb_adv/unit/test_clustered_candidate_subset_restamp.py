"""Regression tests for clustered bounded-candidate subset identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip import (
    _hardlink_copy_run_dir,
    _run_metric,
    candidate_fingerprint,
    sample_candidates,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.eval.axbench_bpb_adv.artifact_identity import (
    PRECOMPUTED_BAG_SET_METADATA_KEY,
    bind_clustered_table,
    load_clustered_candidate_snapshot,
    restamp_candidates_bag_set,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
)
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from tests.axbench_bpb_adv.unit.test_clustered_artifact_identity import (
    _candidate_row,
    _write_dependencies_for_row,
)
from tests.axbench_bpb_adv.unit.test_feature_diffmean import (
    _linear_percentile_table,
    _real_cluster_contract,
)
from tests.scripts.test_benchmark_axbench_bpb_adv_roundtrip import (
    _install_fake_metric_outputs,
)


def _contract() -> ClusterFeatureContract:
    return _real_cluster_contract(
        percentile_table=_linear_percentile_table(4),
        mask=FeatureMask(FeatureMaskMode.whitelist, (0, 1, 2)),
    )


def _write_source_contract(
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    fingerprint = contract.feature_contract_fingerprint
    payload = BpbAdvSourceContract(
        seed_fingerprint="1" * 64,
        schema_version=CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
        source_run_dir="/fixture/clustered-subset-restamp",
        direction_training_layer=6,
        feature_contract_schema_version=contract.schema_version,
        feature_contract_fingerprint=fingerprint,
        candidate_provenance_sha256="2" * 64,
        stage2_feature_contract_fingerprints={
            "concept_features.safetensors": fingerprint,
            "concept_features_neg.safetensors": fingerprint,
        },
        baseline_training_corpus_sha256={
            "positive": "3" * 64,
            "negative": "4" * 64,
        },
    ).to_json_dict()
    paths.source_contract().write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stamped_two_bag_fixture(
    tmp_path: Path,
) -> tuple[
    ClusterFeatureContract,
    AxBenchBpbAdvPaths,
    AxBenchBpbAdvPaths,
    pa.Table,
]:
    contract = _contract()
    source_paths = AxBenchBpbAdvPaths(output_root=tmp_path / "source", run_tag="run")
    _write_source_contract(source_paths, contract)
    rows = [
        _candidate_row(
            candidate_id="candidate-7",
            concept_id=7,
            core_prefix_id="prefix-7",
            target_model="encdec",
        ),
        _candidate_row(
            candidate_id="candidate-8",
            concept_id=8,
            core_prefix_id="prefix-8",
            target_model="encdec",
        ),
    ]
    for index, row in enumerate(rows):
        _write_dependencies_for_row(
            source_paths,
            row,
            bag_bytes=f"bag-{index}".encode(),
            scored_bytes=f"scored-{index}".encode(),
        )
    table = pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)
    stamped = bind_clustered_table(
        table,
        paths=source_paths,
        contract=contract,
        artifact_family="candidates",
    )
    source_paths.candidates_path().parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(stamped, source_paths.candidates_path())

    derived_paths = AxBenchBpbAdvPaths(output_root=tmp_path / "derived", run_tag="run")
    _hardlink_copy_run_dir(source_paths.root, derived_paths.root)
    return contract, source_paths, derived_paths, stamped


def test_clustered_subset_requires_bag_set_restamp_before_snapshot_load(
    tmp_path: Path,
) -> None:
    contract, _source_paths, derived_paths, stamped = _stamped_two_bag_fixture(tmp_path)
    subset = sample_candidates(stamped, sample_size=1, sample_seed=4).table
    assert subset.num_rows == 1
    assert len(set(stamped.column("concept_id").to_pylist())) == 2
    assert len(set(subset.column("concept_id").to_pylist())) == 1
    derived_paths.candidates_path().parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, derived_paths.candidates_path())

    with pytest.raises(AxBenchBpbAdvError, match="identity is absent or stale"):
        load_clustered_candidate_snapshot(
            derived_paths.candidates_path(),
            paths=derived_paths,
            contract=contract,
        )

    restamped = restamp_candidates_bag_set(subset, paths=derived_paths)
    pq.write_table(restamped, derived_paths.candidates_path())
    snapshot = load_clustered_candidate_snapshot(
        derived_paths.candidates_path(),
        paths=derived_paths,
        contract=contract,
    )

    assert snapshot.table.num_rows == subset.num_rows == 1


def test_restamp_candidates_bag_set_requires_reserved_candidate_metadata(
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    table = pa.Table.from_pylist([_candidate_row()], schema=CANDIDATES_SCHEMA)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="requires reserved metadata keys: missing=",
    ):
        restamp_candidates_bag_set(table, paths=paths)


def test_restamp_candidates_bag_set_changes_only_bag_stamp_deterministically(
    tmp_path: Path,
) -> None:
    _contract_value, _source_paths, derived_paths, stamped = _stamped_two_bag_fixture(
        tmp_path
    )
    subset = sample_candidates(stamped, sample_size=1, sample_seed=4).table

    first = restamp_candidates_bag_set(subset, paths=derived_paths)
    second = restamp_candidates_bag_set(subset, paths=derived_paths)
    before = dict(subset.schema.metadata or {})
    after = dict(first.schema.metadata or {})

    assert first.equals(second, check_metadata=True)
    assert first.column_names == subset.column_names
    assert after[PRECOMPUTED_BAG_SET_METADATA_KEY]
    assert after[PRECOMPUTED_BAG_SET_METADATA_KEY] != before[
        PRECOMPUTED_BAG_SET_METADATA_KEY
    ]
    assert {
        key: value
        for key, value in after.items()
        if key != PRECOMPUTED_BAG_SET_METADATA_KEY
    } == {
        key: value
        for key, value in before.items()
        if key != PRECOMPUTED_BAG_SET_METADATA_KEY
    }


def test_sampled_metric_harness_restamps_clustered_subset_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, source_paths, _unused_derived_paths, stamped = _stamped_two_bag_fixture(
        tmp_path
    )
    sample = sample_candidates(stamped, sample_size=1, sample_seed=4)
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    observed_rows: list[int] = []

    def load_derived_snapshot(command: list[str], *, check: bool) -> None:
        assert check is True
        output_root = Path(
            next(
                value.split("=", 1)[1]
                for value in command
                if value.startswith("output_root=")
            )
        )
        derived_paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
        snapshot = load_clustered_candidate_snapshot(
            derived_paths.candidates_path(),
            paths=derived_paths,
            contract=contract,
        )
        observed_rows.append(snapshot.table.num_rows)
        _install_fake_metric_outputs(command)

    monkeypatch.setattr(
        "scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip.subprocess.run",
        load_derived_snapshot,
    )

    result = _run_metric(
        metric="roundtrip",
        args=argparse.Namespace(
            output_root=tmp_path / "source",
            run_tag="run",
            benchmark_dir=benchmark_dir,
            python="python",
        ),
        overrides=(),
        candidate=candidate_fingerprint(source_paths.candidates_path()),
        sample=sample,
    )

    assert observed_rows == [1]
    assert result["candidate_fingerprint"]["rows"] == 1
