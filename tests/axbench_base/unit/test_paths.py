"""Unit tests for axbench_base path resolvers."""

from __future__ import annotations

from steering.eval.axbench_base import paths


def test_cluster_artifacts_paths_resolve_under_run_root(monkeypatch, tmp_path):
    """PR #739 Track C1 companion addition: the 3 path helpers
    ``cluster_artifacts.py`` requires (root commit marker, Stage 0a
    provenance commit marker, Stage 4 generation/resume commit marker)
    resolve under the expected roots with the canonical filenames.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "cluster_artifacts_run"
    output_paths = paths.AxBenchBasePaths(tmp_path / "eval" / "axbench_base", run_tag)
    expected_root = output_paths.run_output_root()

    feature_contract_path = output_paths.feature_contract_json()
    provenance_path = output_paths.candidate_concepts_cluster_provenance_json()
    generation_marker_path = output_paths.generations_feature_contract_json()

    assert feature_contract_path == expected_root / "feature_contract.json"
    assert provenance_path == expected_root / "candidate_concepts.cluster_provenance.json"
    assert generation_marker_path == output_paths.generations_dir() / "feature_contract.json"


def test_stage1b_partial_paths_resolve_under_run_root(tmp_path):
    output_paths = paths.AxBenchBasePaths(tmp_path, "resume-run")
    run_root = output_paths.run_output_root()

    assert output_paths.stage1b_partial_parquet() == (
        run_root / "stage1b_partial.parquet"
    )
    assert output_paths.stage1b_partial_json() == (
        run_root / "stage1b_partial.json"
    )
