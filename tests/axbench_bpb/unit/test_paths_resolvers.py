"""Unit tests for output-side path resolvers."""
from __future__ import annotations


def test_run_output_root_sanitization(tmp_path):
    from steering.eval.axbench_bpb.paths import OutputPaths
    op = OutputPaths(output_root=tmp_path, run_tag="20260423-143022-abcdef01")
    root = op.run_output_root()
    assert root == tmp_path / "20260423-143022-abcdef01"


def test_context_cache_layout(tmp_path):
    from steering.eval.axbench_bpb.paths import OutputPaths
    op = OutputPaths(output_root=tmp_path, run_tag="r1")
    cache_dir = op.context_cache_dir()
    assert cache_dir == tmp_path / "r1" / "context_cache"
    assert op.context_cache_file("abcd1234") == cache_dir / "ab" / "abcd1234.safetensors"
    assert op.context_cache_manifest() == cache_dir / "manifest.json"


def test_bpb_scores_layout(tmp_path):
    from steering.eval.axbench_bpb.paths import OutputPaths
    op = OutputPaths(output_root=tmp_path, run_tag="r1")
    assert op.bpb_scores_dir() == tmp_path / "r1" / "bpb_scores"
    assert op.bpb_scores_rank_dir(2) == tmp_path / "r1" / "bpb_scores" / "rank_002"


def test_bpb_cache_path(tmp_path):
    from steering.eval.axbench_bpb.paths import OutputPaths
    op = OutputPaths(output_root=tmp_path, run_tag="r1")
    assert op.bpb_cache_path(4) == tmp_path / "r1" / "bpb_cache" / "rank_004.sqlite"


def test_aggregation_artifacts(tmp_path):
    from steering.eval.axbench_bpb.paths import OutputPaths
    op = OutputPaths(output_root=tmp_path, run_tag="r1")
    assert op.leaderboard_csv() == tmp_path / "r1" / "aggregation" / "leaderboard.csv"
    assert op.report_yaml() == tmp_path / "r1" / "aggregation" / "report.yaml"


def test_run_tag_sanitization_rejects_traversal(tmp_path):
    from steering.eval.axbench_base.paths import _sanitize
    from steering.eval.axbench_bpb.paths import OutputPaths

    op = OutputPaths(output_root=tmp_path, run_tag="../../etc/passwd")
    root = op.run_output_root()
    # Escape must be prevented: the resolved path must remain under
    # ``tmp_path`` and must not contain a traversal marker.
    assert tmp_path in root.parents
    assert ".." not in root.name
    # And the name must match whatever ``_sanitize`` actually produced for
    # this input (deterministic: ``traversal_<sha256[:16]>``).
    assert root.name == _sanitize("../../etc/passwd")
    assert root.name.startswith("traversal_")
