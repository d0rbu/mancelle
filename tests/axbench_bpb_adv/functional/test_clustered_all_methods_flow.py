"""Clustered CPU functional chain for the locked sixteen-method matrix.

The model/SAE-heavy seams have focused real-tiny-model and differential tests.
This test instead drives the shared source -> attack -> candidate -> score ->
aggregate artifact pipeline for every production clustered method in one run,
including conditional table ownership and byte-exact resume.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import pytest
import torch as t

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb_adv.artifact_identity import (
    FEATURE_CONTRACT_METADATA_KEY,
    assert_table_cluster_identity,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    OursGcgConfig,
    Stage,
    validate_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import (
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    ROOFLINE_PRECOMPUTED_BAG_FAMILIES,
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table
from steering.eval.cluster_config import CLUSTERED_AXBENCH_BPB_ADV_METHODS
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.axbench_bpb_adv.artifact_helpers import artifact_identity_metadata
from tests.axbench_bpb_adv.functional.test_cli_stage_chain import (
    _dual_score_pair,
    _TinyTokenizer,
)
from tests.axbench_bpb_adv.source_run_builders import both_class_rows
from tests.axbench_bpb_adv.unit.test_source_validation_clustered_flow import (
    _cluster_source,
)
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)


def _target_metadata() -> dict[str, TargetModelMetadata]:
    common = {
        "n_sae_features": 8,
        "per_feature_stats": None,
        "device": "cpu",
        "sae_layer": 12,
        "segment_size": 4,
    }
    return {
        "decoder_only": TargetModelMetadata(
            num_buckets=None,
            negative_selection=None,
            feature_selection=None,
            feature_mask=None,
            **common,
        ),
        "encdec": TargetModelMetadata(
            num_buckets=2,
            negative_selection={"distribution": "geometric", "mean": 1.0},
            feature_selection={"distribution": "geometric", "mean": 1.0},
            feature_mask={"mode": "whitelist", "feature_ids": [1, 2, 3]},
            **common,
        ),
    }


def _context_segments(candidates: Any) -> dict[str, tuple[ContextFeatureSegment, ...]]:
    return {
        str(row["prefix_hash"]): (
            ContextFeatureSegment(
                feature_ids=(1,),
                buckets=(0,),
                char_start=0,
                char_end=len(str(row["prefix_text"])),
            ),
        )
        for row in candidates.to_pylist()
        if str(row["target_model"]) == "encdec"
    }


def _tracked_artifacts(paths: AxBenchBpbAdvPaths) -> tuple[Path, ...]:
    return (
        paths.source_contract(),
        paths.evaluation_population(),
        paths.attack_rows_path(),
        paths.candidates_path(),
        paths.universal_triggers_path(),
        paths.scores_path(),
        paths.aggregates_dir() / "per_candidate.parquet",
        paths.aggregates_dir() / "macro.parquet",
    )


def _write_stub_precomputed_bag_dependencies(paths: AxBenchBpbAdvPaths) -> None:
    """Materialize the model-free seam's bag inputs for identity validation."""

    rows = pq.read_table(paths.attack_rows_path()).to_pylist()
    for row in rows:
        family = precomputed_bag_method_family(str(row["method_name"]))
        if family is None:
            continue
        reg_lambda = (
            0.0
            if family
            in {
                PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN,
                PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
            }
            else float(row["coefficient_str"])
        )
        prefix_id = (
            str(row["core_prefix_id"]) if family in ROOFLINE_PRECOMPUTED_BAG_FAMILIES else None
        )
        bag_path = paths.ours_gcg_bag(
            int(row["concept_id"]),
            str(row["mode"]),
            reg_lambda,
            family,
            prefix_id=prefix_id,
        )
        bag_path.parent.mkdir(parents=True, exist_ok=True)
        bag_path.write_bytes(
            f"{family.value}:{row['mode']}:{row['concept_id']}:{prefix_id}".encode()
        )
        scored_path = paths.ours_gcg_scored_prefixes(
            int(row["concept_id"]),
            str(row["mode"]),
            family,
        )
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        scored_path.write_bytes(f"{family.value}:{row['mode']}:{row['concept_id']}:scored".encode())


def test_clustered_all_sixteen_methods_route_score_aggregate_and_resume_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    fingerprint = _cluster_source(
        source,
        positive_rows=[
            *both_class_rows("positive-a"),
            *both_class_rows("positive-b"),
        ],
        negative_rows=[
            *both_class_rows("negative-a"),
            *both_class_rows("negative-b"),
        ],
    )
    mask_path = tmp_path / "mask.txt"
    cluster_path = tmp_path / "clusters.yaml"
    mask_path.write_text("1\n2\n3\n", encoding="utf-8")
    cluster_path.write_text(
        "schema_version: 1\nclusters:\n  - [1, 2]\n",
        encoding="utf-8",
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source),
        output_root=str(tmp_path / "out"),
        run_tag="all-sixteen",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "features" / "stats.safetensors"),
        feature_mask={"mode": "whitelist", "path": str(mask_path)},
        feature_clusters={"path": str(cluster_path)},
        base_methods=list(CLUSTERED_AXBENCH_BPB_ADV_METHODS),
        target_models=["decoder_only", "encdec"],
        adversarial_methods=[
            FasterGcgConfig(optimized_token_counts=(0,), max_steps=0),
        ],
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=(0.0,),
            split_ratio=0.5,
            split_seed=0,
            max_steps=0,
        ),
        steering_metric="bpb_class_margin",
        general_val_bpb=GeneralValBpbConfig(enabled=False),
        scatter_baseline_x="baseline_loss",
    )
    validate_config(cfg)
    assert cfg.run_tag is not None
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag=cfg.run_tag)
    paths.root.mkdir(parents=True)
    shutil.copy2(source / "feature_contract.json", paths.root / "feature_contract.json")
    contract: Any = SimpleNamespace(feature_contract_fingerprint=fingerprint)
    tokenizer = _TinyTokenizer()

    monkeypatch.setattr(
        bpb_adv_cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg, _contract=None: (_target_metadata(), tokenizer),
    )
    monkeypatch.setattr(
        model_optimize,
        "require_precomputed_bag_prompt_rows_cover_scored_prefixes",
        lambda rows, **_kwargs: rows,
    )

    def _candidate_stage(**kwargs: Any) -> int:
        return run_model_backed_bpb_candidate_stage(
            **kwargs,
            tokenizer=tokenizer,
            embedding_weight=t.zeros(128, 1),
            test_objective=zero_objective,
            test_gradient_provider_batch=zero_gradient_batch,
        )

    monkeypatch.setattr(
        bpb_adv_cli,
        "run_model_backed_bpb_candidate_stage",
        _candidate_stage,
    )

    def _score_stage(*, candidates: Any, **_kwargs: Any) -> Any:
        return build_bpb_adv_score_table(
            candidates=candidates,
            context_segments_by_prefix_hash=_context_segments(candidates),
            tokenizer=tokenizer,
            token_bytes=t.ones(128, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 64, "encdec": 64},
            score_pair=_dual_score_pair,
        )

    monkeypatch.setattr(
        bpb_adv_cli,
        "run_model_backed_bpb_adv_score_stage",
        _score_stage,
    )

    stages = (
        Stage.VALIDATE_SOURCE,
        Stage.PREPARE_ATTACK_ROWS,
        Stage.BUILD_CANDIDATES,
        Stage.SCORE,
        Stage.SCORE_GENERAL_VAL,
        Stage.AGGREGATE,
    )
    for stage in stages:
        if stage is Stage.BUILD_CANDIDATES:
            _write_stub_precomputed_bag_dependencies(paths)
        bpb_adv_cli._run_stage(
            stage,
            cfg,
            ddp_rank=0,
            ddp_world_size=1,
            cluster_contract=contract,
        )

    attack_rows = pq.read_table(paths.attack_rows_path())
    candidates = pq.read_table(paths.candidates_path())
    scores = pq.read_table(paths.scores_path())
    assert attack_rows.schema.remove_metadata().equals(ATTACK_ROWS_SCHEMA)
    assert candidates.schema.remove_metadata().equals(CANDIDATES_SCHEMA)
    assert scores.schema.remove_metadata().equals(SCORE_SCHEMA)
    expected_methods = set(CLUSTERED_AXBENCH_BPB_ADV_METHODS)
    assert set(attack_rows.column("method_name").to_pylist()) == expected_methods
    assert set(candidates.column("method_name").to_pylist()) == expected_methods
    assert set(scores.column("method_name").to_pylist()) == expected_methods
    # 52, not 50: the two ours_gcg_alpha* methods each contribute one row at
    # their single default coefficient (objective_alpha_grid == (1.0,)), the
    # same shape as the fair GCG pair they share bags with.
    assert scores.num_rows == candidates.num_rows == attack_rows.num_rows == 52
    for table, family in (
        (attack_rows, "attack_rows"),
        (candidates, "candidates"),
        (scores, "class_margin_scores"),
    ):
        identity = artifact_identity_metadata(table)
        assert identity["feature_contract_fingerprint"] == fingerprint
        assert identity["clustered_artifact_family"] == family

    first_bytes = {path: path.read_bytes() for path in _tracked_artifacts(paths)}
    for stage in stages:
        if stage is Stage.BUILD_CANDIDATES:
            _write_stub_precomputed_bag_dependencies(paths)
        bpb_adv_cli._run_stage(
            stage,
            cfg,
            ddp_rank=0,
            ddp_world_size=1,
            cluster_contract=contract,
        )
    assert {path: path.read_bytes() for path in _tracked_artifacts(paths)} == first_bytes

    mutated_attack_rows = pq.read_table(paths.attack_rows_path())
    mutated_metadata = dict(mutated_attack_rows.schema.metadata or {})
    mutated_metadata[FEATURE_CONTRACT_METADATA_KEY] = ("f" * 64).encode("ascii")
    mutated_attack_rows = mutated_attack_rows.replace_schema_metadata(mutated_metadata)
    with pytest.raises(AxBenchBpbAdvError, match="identity is absent or stale"):
        assert_table_cluster_identity(
            mutated_attack_rows,
            paths=paths,
            contract=contract,
            artifact_family="attack_rows",
        )
