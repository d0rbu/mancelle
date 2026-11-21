from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.feature_selection_hash import compute_canonical_feature_selection_hash
from steering.overlay_io import OverlayKind
from steering.packed_batch_io import compute_selection_hash

REPO_ROOT = Path(__file__).parents[2]

REMOVED_FEATURE_CLUSTERING_FILES = [
    "configs/extraction/plot_feature_clustering_sweep.yaml",
    "configs/extraction/precompute_feature_clustering.yaml",
    "configs/profiling/benchmark_consolidation.yaml",
    "configs/profiling/benchmark_consolidation_e2e.yaml",
    "docs/pipelines/consolidation-runtime-profiling.md",
    "docs/pipelines/feature-clustering-hyperparameters.md",
    "docs/pipelines/feature-clustering.md",
    "scripts/benchmark/benchmark_consolidation.py",
    "scripts/benchmark/benchmark_consolidation_e2e.py",
    "scripts/benchmark/benchmark_consolidation_phase2_before_after.py",
    "scripts/benchmark/benchmark_consolidation_pr3_before_after.py",
    "scripts/preprocess/precompute_feature_clustering.py",
    "steering/clustering.py",
    "steering/conditional/feature_clustering.py",
    "tests/axbench_base/test_axbench_extraction_clustering.py",
    "tests/profiling/test_consolidation_marker_overhead.py",
    "tests/profiling/test_consolidation_timing_tags.py",
    "tests/scripts/test_benchmark_consolidation_phase2_arms.py",
    "tests/steering/test_bpb_pipeline_feature_clustering.py",
    "tests/steering/test_clustering_context_cache.py",
    "tests/steering/test_clustering_cross_path.py",
    "tests/steering/test_clustering_delta.py",
    "tests/steering/test_clustering_fused.py",
    "tests/steering/test_clustering_magnitude_path.py",
    "tests/steering/test_clustering_method.py",
    "tests/steering/test_clustering_negatives.py",
    "tests/steering/test_clustering_pipeline.py",
    "tests/steering/test_clustering_producer.py",
    "tests/steering/test_clustering_propagation.py",
    "tests/steering/test_clustering_selection_hash.py",
    "tests/steering/test_clustering_sweep.py",
    "tests/steering/test_completion_steering_eval_clustering.py",
    "tests/steering/test_consolidation_batched_equivalence.py",
    "tests/steering/test_consolidation_benchmark.py",
    "tests/steering/test_consolidation_e2e_harness.py",
    "tests/steering/test_consolidation_e2e_real_shard.py",
    "tests/steering/test_consolidation_markers_eval.py",
    "tests/steering/test_consolidation_markers_live.py",
    "tests/steering/test_consolidation_markers_sft.py",
    "tests/steering/test_consolidation_negative_mask_cache_equivalence.py",
    "tests/steering/test_encdec_sft_clustering.py",
    "tests/steering/test_feature_clustering_overlay.py",
    "tests/steering/test_feature_clustering_unit.py",
    "tests/steering/test_feature_config_consolidation.py",
    "tests/steering/test_feature_consolidation.py",
    "tests/steering/test_live_dataloader_clustering.py",
    "tests/steering/test_multiseg_prepare_segments_equivalence.py",
    "tests/steering/test_precompute_feature_clustering.py",
    "tests/steering/test_self_conditioned_clustering.py",
    "viz/clustering_hyperparam_analysis.py",
    "viz/plot_feature_clustering_sweep.py",
]


def test_deleted_feature_clustering_files_do_not_exist() -> None:
    for rel in REMOVED_FEATURE_CLUSTERING_FILES:
        assert not (REPO_ROOT / rel).exists(), rel


def test_removed_file_paths_absent_from_active_code_config_and_docs() -> None:
    allowed_parts = {
        "tests/steering/test_feature_clustering_removed_static.py",
    }
    scan_roots = [
        REPO_ROOT / "configs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "steering",
        REPO_ROOT / "tests",
        REPO_ROOT / "docs" / "reference",
        REPO_ROOT / "docs" / "pipelines",
        REPO_ROOT / "docs" / "onboarding",
        REPO_ROOT / "docs" / "cli-reference.md",
    ]
    offenders: list[str] = []
    for root in scan_roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in allowed_parts:
                continue
            if any(part in path.parts for part in (".venv", "__pycache__")):
                continue
            text = path.read_text(errors="ignore")
            for removed_path in REMOVED_FEATURE_CLUSTERING_FILES:
                if removed_path in text:
                    offenders.append(f"{rel}: {removed_path}")
    assert offenders == []


def test_selection_hash_apis_have_no_feature_clustering_parameter() -> None:
    assert "feature_clustering" not in inspect.signature(compute_selection_hash).parameters
    assert (
        "feature_clustering"
        not in inspect.signature(compute_canonical_feature_selection_hash).parameters
    )


def test_feature_pipeline_config_fields_are_the_surviving_fields() -> None:
    assert [field.name for field in dataclasses.fields(FeaturePipelineConfig)] == [
        "selection_config",
        "boundaries",
        "negative_bundle",
        "effective_buckets",
        "delta_bundle",
        "feature_mask",
    ]


def test_target_model_metadata_has_no_feature_clustering_canonical() -> None:
    names = {field.name for field in dataclasses.fields(TargetModelMetadata)}
    assert "feature_clustering_canonical" not in names


def test_overlay_kind_has_no_clustering_value() -> None:
    assert {kind.value for kind in OverlayKind} == {
        "pct",
        "seg_boundaries",
        "packed_rows",
        "grad",
    }


def test_clustering_dependencies_removed_but_sklearn_kept() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    lock = (REPO_ROOT / "uv.lock").read_text()
    assert "igraph" not in pyproject
    assert "leidenalg" not in pyproject
    assert 'name = "igraph"' not in lock
    assert 'name = "leidenalg"' not in lock
    assert "scikit-learn" in pyproject
    assert 'name = "scikit-learn"' in lock


def test_removed_implementation_symbols_absent_outside_guards_and_tests() -> None:
    forbidden = [
        "FeatureClustering",
        "FeatureConsolidation",
        "ConsolidationContext",
        "feature_clustering_canonical",
        "precompute_feature_clustering",
        "compute_clusters_from_codes",
        "apply_threshold_consolidated",
        "consolidate_magnitude_path",
        "OverlayKind.CLUSTERING",
    ]
    allowed_parts = {
        "steering/removed_features.py",
        "tests/steering/test_feature_clustering_removed_static.py",
        "tests/steering/test_feature_clustering_removed_guards.py",
        "tests/reference_feature_clustering.py",
        "tests/steering/test_feature_clusters_consolidation.py",
        "tests/steering/test_feature_clusters_checkpoint_meta_resolution.py",
        "tests/steering/test_feature_clusters_preselection.py",
        "tests/steering/test_inline_steering_eval.py",
        "tests/steering/test_shuffled_conditioning_eval.py",
        "tests/axbench_base/unit/test_model_detection.py",
        "tests/test_completion_tui/test_checkpoint_contract.py",
    }
    scan_roots = [
        REPO_ROOT / "steering",
        REPO_ROOT / "scripts",
        REPO_ROOT / "configs",
        REPO_ROOT / "tests",
    ]
    offenders: list[str] = []
    for root in scan_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in allowed_parts:
                continue
            if any(part in path.parts for part in (".venv", "__pycache__")):
                continue
            text = path.read_text(errors="ignore")
            for symbol in forbidden:
                if symbol in text:
                    offenders.append(f"{rel}: {symbol}")
    assert offenders == []
