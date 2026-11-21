"""Regression contract for issue #1001's AxBench-Base output routing.

These tests intentionally pin the complete path surface, not a representative
sample.  A newly added artifact resolver must be added to the oracle below so
custom-root routing cannot silently regress for one stage while other stages
continue to pass.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError, SteeringMode
from steering.eval.axbench_bpb.source_paths import SourcePaths
from tests.axbench_base import paths as test_paths

_REPO_ROOT = Path(__file__).parents[3]

_LEGACY_FREE_RESOLVERS = frozenset(
    {
        "aggregation_report",
        "candidate_concepts_cluster_provenance_json",
        "candidate_concepts_json",
        "concept_examples_qc_report",
        "concept_features",
        "concept_features_neg",
        "concepts_json",
        "downstream_concepts",
        "diffmean_direction",
        "feature_contract_json",
        "generations_dir",
        "generations_feature_contract_json",
        "judge_cache",
        "judge_scores_dir",
        "leaderboard_csv",
        "linear_probe_direction",
        "lora_adapter",
        "negative_mode_prefixes",
        "prompting_descriptions",
        "profile_dir",
        "rejected_concepts_json",
        "report_yaml",
        "run_output_root",
        "seed_extraction_root",
        "seed_pool_manifest",
        "seeds_eval",
        "seeds_train",
        "stage0c_partial_rank_json",
        "stage0c_partial_rank_parquet",
        "stage1b_concept_yield_report",
        "stage1b_concepts",
        "stage1b_partial_json",
        "stage1b_partial_parquet",
        "stage1c_concept_yield_report",
        "stage1c_concepts",
        "stage4_contexts_jsonl",
        "stage4_contexts_manifest",
        "stage4_contexts_parquet",
        "synth_response_cache",
        "synth_training",
        "synth_training_negative",
        "synth_training_positive",
    }
)


def _config(
    *,
    output_root: str | None,
    run_tag: str | None = "issue-1001",
) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir=("out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"),
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=output_root,
        run_tag=run_tag,
    )


def _complete_relative_oracle(output_paths: paths.AxBenchBasePaths) -> dict[str, Path]:
    """Return every historical artifact resolver and its exact relative path."""
    return {
        "aggregation_report": output_paths.aggregation_report().relative_to(
            output_paths.run_output_root()
        ),
        "candidate_concepts_cluster_provenance_json": (
            output_paths.candidate_concepts_cluster_provenance_json().relative_to(
                output_paths.run_output_root()
            )
        ),
        "candidate_concepts_json": output_paths.candidate_concepts_json().relative_to(
            output_paths.run_output_root()
        ),
        "concept_examples_qc_report": (
            output_paths.concept_examples_qc_report().relative_to(output_paths.run_output_root())
        ),
        "concept_features": output_paths.concept_features().relative_to(
            output_paths.run_output_root()
        ),
        "concept_features_neg": output_paths.concept_features_neg().relative_to(
            output_paths.run_output_root()
        ),
        "concepts_json": output_paths.concepts_json().relative_to(output_paths.run_output_root()),
        "downstream_concepts": output_paths.downstream_concepts(
            run_negative_mode=True
        ).relative_to(output_paths.run_output_root()),
        "diffmean_direction": output_paths.diffmean_direction(
            "decoder_only", SteeringMode.NEGATIVE, 42
        ).relative_to(output_paths.run_output_root()),
        "feature_contract_json": output_paths.feature_contract_json().relative_to(
            output_paths.run_output_root()
        ),
        "generations_dir": output_paths.generations_dir().relative_to(
            output_paths.run_output_root()
        ),
        "generations_feature_contract_json": (
            output_paths.generations_feature_contract_json().relative_to(
                output_paths.run_output_root()
            )
        ),
        "judge_cache": output_paths.judge_cache().relative_to(output_paths.run_output_root()),
        "judge_scores_dir": output_paths.judge_scores_dir().relative_to(
            output_paths.run_output_root()
        ),
        "leaderboard_csv": output_paths.leaderboard_csv().relative_to(
            output_paths.run_output_root()
        ),
        "linear_probe_direction": output_paths.linear_probe_direction(
            "decoder_only", SteeringMode.NEGATIVE, 42
        ).relative_to(output_paths.run_output_root()),
        "lora_adapter": output_paths.lora_adapter(SteeringMode.NEGATIVE, 42).relative_to(
            output_paths.run_output_root()
        ),
        "negative_mode_prefixes": output_paths.negative_mode_prefixes().relative_to(
            output_paths.run_output_root()
        ),
        "prompting_descriptions": output_paths.prompting_descriptions(
            SteeringMode.NEGATIVE
        ).relative_to(output_paths.run_output_root()),
        "profile_dir": output_paths.profile_dir().relative_to(
            output_paths.run_output_root()
        ),
        "rejected_concepts_json": output_paths.rejected_concepts_json().relative_to(
            output_paths.run_output_root()
        ),
        "report_yaml": output_paths.report_yaml().relative_to(output_paths.run_output_root()),
        "seed_extraction_root": output_paths.seed_extraction_root().relative_to(
            output_paths.run_output_root()
        ),
        "seed_pool_manifest": output_paths.seed_pool_manifest().relative_to(
            output_paths.run_output_root()
        ),
        "seeds_eval": output_paths.seeds_eval().relative_to(output_paths.run_output_root()),
        "seeds_train": output_paths.seeds_train().relative_to(output_paths.run_output_root()),
        "stage0c_partial_rank_json": output_paths.stage0c_partial_rank_json(7).relative_to(
            output_paths.run_output_root()
        ),
        "stage0c_partial_rank_parquet": output_paths.stage0c_partial_rank_parquet(7).relative_to(
            output_paths.run_output_root()
        ),
        "stage1b_concept_yield_report": (
            output_paths.stage1b_concept_yield_report().relative_to(
                output_paths.run_output_root()
            )
        ),
        "stage1b_concepts": output_paths.stage1b_concepts().relative_to(
            output_paths.run_output_root()
        ),
        "stage1b_partial_json": output_paths.stage1b_partial_json().relative_to(
            output_paths.run_output_root()
        ),
        "stage1b_partial_parquet": output_paths.stage1b_partial_parquet().relative_to(
            output_paths.run_output_root()
        ),
        "stage1c_concept_yield_report": (
            output_paths.stage1c_concept_yield_report().relative_to(
                output_paths.run_output_root()
            )
        ),
        "stage1c_concepts": output_paths.stage1c_concepts().relative_to(
            output_paths.run_output_root()
        ),
        "stage4_contexts_jsonl": output_paths.stage4_contexts_jsonl().relative_to(
            output_paths.run_output_root()
        ),
        "stage4_contexts_manifest": (
            output_paths.stage4_contexts_manifest().relative_to(output_paths.run_output_root())
        ),
        "stage4_contexts_parquet": output_paths.stage4_contexts_parquet().relative_to(
            output_paths.run_output_root()
        ),
        "synth_response_cache": output_paths.synth_response_cache().relative_to(
            output_paths.run_output_root()
        ),
        "synth_training_negative": (
            output_paths.synth_training_negative().relative_to(output_paths.run_output_root())
        ),
        "synth_training_positive": (
            output_paths.synth_training_positive().relative_to(output_paths.run_output_root())
        ),
    }


_EXPECTED_RELATIVE_PATHS = {
    "aggregation_report": Path("aggregation_report.yaml"),
    "candidate_concepts_cluster_provenance_json": Path(
        "candidate_concepts.cluster_provenance.json"
    ),
    "candidate_concepts_json": Path("candidate_concepts.json"),
    "concept_examples_qc_report": Path("concept_examples_qc_report.json"),
    "concept_features": Path("concept_features.safetensors"),
    "concept_features_neg": Path("concept_features_neg.safetensors"),
    "concepts_json": Path("concepts.json"),
    "downstream_concepts": Path("concepts_stage1c.json"),
    "diffmean_direction": Path("baselines/diffmean/decoder_only/negative/concept_0042.safetensors"),
    "feature_contract_json": Path("feature_contract.json"),
    "generations_dir": Path("generations"),
    "generations_feature_contract_json": Path("generations/feature_contract.json"),
    "judge_cache": Path("judge_cache.sqlite"),
    "judge_scores_dir": Path("judge_scores"),
    "leaderboard_csv": Path("leaderboard.csv"),
    "linear_probe_direction": Path(
        "baselines/linear_probe/decoder_only/negative/concept_0042.safetensors"
    ),
    "lora_adapter": Path("baselines/lora/decoder_only/negative/concept_0042.pt"),
    "negative_mode_prefixes": Path("negative_mode_prefixes.parquet"),
    "prompting_descriptions": Path("baselines/prompting/negative_descriptions.json"),
    "profile_dir": Path("profile"),
    "rejected_concepts_json": Path("rejected_concepts.json"),
    "report_yaml": Path("report.yaml"),
    "seed_extraction_root": Path("_seed_extraction_v2"),
    "seed_pool_manifest": Path("seed_pool_manifest.json"),
    "seeds_eval": Path("seeds_eval.safetensors"),
    "seeds_train": Path("seeds_train.safetensors"),
    "stage0c_partial_rank_json": Path("stage0c_partial_rank_7.json"),
    "stage0c_partial_rank_parquet": Path("stage0c_partial_rank_7.parquet"),
    "stage1b_concept_yield_report": Path("stage1b_concept_yield.json"),
    "stage1b_concepts": Path("concepts_stage1b.json"),
    "stage1b_partial_json": Path("stage1b_partial.json"),
    "stage1b_partial_parquet": Path("stage1b_partial.parquet"),
    "stage1c_concept_yield_report": Path("stage1c_concept_yield.json"),
    "stage1c_concepts": Path("concepts_stage1c.json"),
    "stage4_contexts_jsonl": Path("stage4_contexts.jsonl"),
    "stage4_contexts_manifest": Path("stage4_contexts_manifest.json"),
    "stage4_contexts_parquet": Path("stage4_contexts_extraction/shard_00000.parquet"),
    "synth_response_cache": Path("synth_response_cache.sqlite"),
    "synth_training_negative": Path("synth_training_negative.parquet"),
    "synth_training_positive": Path("synth_training_positive.parquet"),
}


def test_default_root_and_every_relative_path_match_pre_change_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "nanochat-base-poison"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))
    output_paths = paths.AxBenchBasePaths.from_config(
        _config(output_root=None, run_tag="oracle-tag")
    )

    assert output_paths.output_root == base_dir / "eval" / "axbench_base"
    assert output_paths.run_output_root() == (base_dir / "eval" / "axbench_base" / "oracle-tag")
    assert _complete_relative_oracle(output_paths) == _EXPECTED_RELATIVE_PATHS
    assert (
        output_paths.synth_training(SteeringMode.POSITIVE) == output_paths.synth_training_positive()
    )
    assert (
        output_paths.synth_training(SteeringMode.NEGATIVE) == output_paths.synth_training_negative()
    )
    assert (
        output_paths.downstream_concepts(run_negative_mode=False)
        == output_paths.stage1b_concepts()
    )
    assert (
        output_paths.downstream_concepts(run_negative_mode=True)
        == output_paths.stage1c_concepts()
    )


def test_default_paths_constructs_the_exact_historical_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))

    output_paths = test_paths.default_paths("default-helper")

    assert output_paths == paths.AxBenchBasePaths(
        tmp_path / "base/eval/axbench_base",
        "default-helper",
    )


def test_relative_oracle_covers_every_public_path_resolver() -> None:
    public_methods = {
        name
        for name, value in vars(paths.AxBenchBasePaths).items()
        if not name.startswith("_") and (callable(value) or isinstance(value, classmethod))
    }

    assert public_methods == _LEGACY_FREE_RESOLVERS | {
        "from_config",
        "neuronpedia_cache",
    }


def test_custom_root_controls_every_run_artifact_and_ignores_poison_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    poison = tmp_path / "must-not-be-used"
    custom = tmp_path / "literal-custom-root"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(poison))
    output_paths = paths.AxBenchBasePaths.from_config(
        _config(output_root=str(custom), run_tag="custom-tag")
    )

    assert output_paths.output_root == custom
    assert output_paths.run_output_root() == custom / "custom-tag"
    assert _complete_relative_oracle(output_paths) == _EXPECTED_RELATIVE_PATHS
    for relative_path in _complete_relative_oracle(output_paths).values():
        resolved = output_paths.run_output_root() / relative_path
        assert resolved.is_relative_to(custom)
        assert not resolved.is_relative_to(poison)


def test_two_path_contracts_with_same_tag_remain_fully_isolated(
    tmp_path: Path,
) -> None:
    first = paths.AxBenchBasePaths(tmp_path / "first", "shared-tag")
    second = paths.AxBenchBasePaths(tmp_path / "second", "shared-tag")

    assert first.run_output_root() != second.run_output_root()
    assert _complete_relative_oracle(first) == _complete_relative_oracle(second)
    assert first.concepts_json() == tmp_path / "first/shared-tag/concepts.json"
    assert second.concepts_json() == tmp_path / "second/shared-tag/concepts.json"


def test_axbench_bpb_read_only_source_paths_match_base_producer_paths(
    tmp_path: Path,
) -> None:
    producer = paths.AxBenchBasePaths(tmp_path / "root", "shared-run")
    consumer = SourcePaths(producer.run_output_root())

    assert consumer.concepts_json() == producer.stage1c_concepts()
    assert consumer.synth_training_positive() == producer.synth_training_positive()
    assert consumer.synth_training_negative() == producer.synth_training_negative()
    assert consumer.seeds_train() == producer.seeds_train()
    assert consumer.seeds_eval() == producer.seeds_eval()
    assert consumer.seed_pool_manifest() == producer.seed_pool_manifest()
    assert consumer.feature_contract() == producer.feature_contract_json()
    assert consumer.candidate_concepts() == producer.candidate_concepts_json()
    assert (
        consumer.candidate_cluster_provenance()
        == producer.candidate_concepts_cluster_provenance_json()
    )
    assert consumer.negative_mode_prefixes() == producer.negative_mode_prefixes()
    assert consumer.concept_features() == producer.concept_features()
    assert consumer.concept_features_neg() == producer.concept_features_neg()
    assert (
        consumer.diffmean_direction("decoder_only", "negative", 42)
        == producer.diffmean_direction("decoder_only", SteeringMode.NEGATIVE, 42)
    )
    assert (
        consumer.linear_probe_direction("encdec", "positive", 42)
        == producer.linear_probe_direction("encdec", SteeringMode.POSITIVE, 42)
    )
    assert (
        consumer.lora_adapter("negative", 42)
        == producer.lora_adapter(SteeringMode.NEGATIVE, 42)
    )
    assert (
        consumer.prompting_descriptions("positive")
        == producer.prompting_descriptions(SteeringMode.POSITIVE)
    )


def test_path_contract_is_immutable(tmp_path: Path) -> None:
    output_paths = paths.AxBenchBasePaths(tmp_path, "tag")
    with pytest.raises(dataclasses.FrozenInstanceError):
        output_paths.run_tag = "mutated"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("configured", "message"),
    [
        ("", "non-empty string"),
        ("   ", "non-empty string"),
        (42, "null or a non-empty string"),
    ],
)
def test_effective_output_root_rejects_invalid_values(
    configured: object,
    message: str,
) -> None:
    with pytest.raises(AxBenchBaseError, match=message):
        paths.effective_output_root(configured)  # ty: ignore[invalid-argument-type]


def test_from_config_requires_resolved_run_tag(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBaseError, match="run_tag must be set"):
        paths.AxBenchBasePaths.from_config(_config(output_root=str(tmp_path), run_tag=None))


def test_neuronpedia_cache_has_exact_canonical_identity_path(
    tmp_path: Path,
) -> None:
    output_paths = paths.AxBenchBasePaths(tmp_path / "root", "ignored-run-tag")
    cache = output_paths.neuronpedia_cache(
        model_id="gemma-2-2b",
        sae_source="8-gemmascope-res-16k",
    )

    assert cache == (
        tmp_path
        / "root"
        / "_neuronpedia_cache"
        / "v1"
        / "b3795102dca9663486a90cd91059268ff598ffd2e4f0aebdc42328d1b847494d.yaml"
    )
    assert len(cache.stem) == 64
    int(cache.stem, 16)


def test_neuronpedia_cache_is_cross_run_but_identity_isolated(
    tmp_path: Path,
) -> None:
    first_run = paths.AxBenchBasePaths(tmp_path, "run-a")
    second_run = paths.AxBenchBasePaths(tmp_path, "run-b")
    identity_a = {
        "model_id": "gemma-2-2b",
        "sae_source": "8-gemmascope-res-16k",
    }
    identity_b = {
        "model_id": "gemma-3-4b-pt",
        "sae_source": "22-gemmascope-2-res-65k",
    }

    assert first_run.neuronpedia_cache(**identity_a) == (second_run.neuronpedia_cache(**identity_a))
    assert first_run.neuronpedia_cache(**identity_a) != (first_run.neuronpedia_cache(**identity_b))
    assert first_run.neuronpedia_cache(**identity_b).name == (
        "d4ac0602a43996ad421dc978f8bcbd8cd1fe1eb6a06dff788d6f56f706dfce8d.yaml"
    )


def test_shadow_preflight_is_disabled_for_null_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    legacy = tmp_path / "eval/axbench_base/same-tag"
    legacy.mkdir(parents=True)

    paths.assert_no_legacy_output_shadow(_config(output_root=None, run_tag="same-tag"))


def test_shadow_preflight_allows_absent_legacy_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))

    paths.assert_no_legacy_output_shadow(
        _config(output_root=str(tmp_path / "custom"), run_tag="fresh")
    )


def test_shadow_preflight_allows_existing_custom_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    legacy = tmp_path / "base/eval/axbench_base/same-tag"
    custom = tmp_path / "custom/same-tag"
    legacy.mkdir(parents=True)
    custom.mkdir(parents=True)

    paths.assert_no_legacy_output_shadow(
        _config(output_root=str(tmp_path / "custom"), run_tag="same-tag")
    )


def test_shadow_preflight_allows_custom_root_equal_to_legacy_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    legacy_root = tmp_path / "eval/axbench_base"
    (legacy_root / "same-tag").mkdir(parents=True)

    paths.assert_no_legacy_output_shadow(_config(output_root=str(legacy_root), run_tag="same-tag"))


def test_shadow_preflight_rejects_ambiguous_historical_custom_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    custom = tmp_path / "custom"
    legacy_run = base / "eval/axbench_base/same-tag"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base))
    legacy_run.mkdir(parents=True)

    with pytest.raises(
        AxBenchBaseError,
        match=r"Refusing to start a second AxBench-Base artifact tree",
    ) as exc_info:
        paths.assert_no_legacy_output_shadow(_config(output_root=str(custom), run_tag="same-tag"))

    message = str(exc_info.value)
    assert str(custom / "same-tag") in message
    assert str(legacy_run) in message
    assert "output_root=null" in message
    assert "fresh run_tag" in message
    assert not (custom / "same-tag").exists()


def test_production_has_no_calls_to_removed_free_path_resolvers() -> None:
    roots = (
        _REPO_ROOT / "steering/eval/axbench_base",
        _REPO_ROOT / "scripts/eval/axbench_base_eval.py",
        _REPO_ROOT / "scripts/benchmark/profile_axbench_base.py",
        _REPO_ROOT / "scripts/benchmark/profile_stage0c_verification.py",
    )
    violations: list[str] = []
    files = [path for root in roots for path in ([root] if root.is_file() else root.rglob("*.py"))]
    for source_path in files:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "paths"
                and (
                    node.func.attr in _LEGACY_FREE_RESOLVERS
                    or node.func.attr in {"AxBenchBasePaths", "default_paths"}
                )
            ):
                violations.append(
                    f"{source_path.relative_to(_REPO_ROOT)}:{node.lineno}:"
                    f" paths.{node.func.attr}(...)"
                )
    assert violations == []


def test_get_base_dir_is_centralized_in_path_contract() -> None:
    roots = (
        _REPO_ROOT / "steering/eval/axbench_base",
        _REPO_ROOT / "scripts/eval/axbench_base_eval.py",
        _REPO_ROOT / "scripts/benchmark/profile_axbench_base.py",
        _REPO_ROOT / "scripts/benchmark/profile_stage0c_verification.py",
    )
    allowed = Path("steering/eval/axbench_base/paths.py")
    violations: list[str] = []
    files = [path for root in roots for path in ([root] if root.is_file() else root.rglob("*.py"))]
    for source_path in files:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        relative_source = source_path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_base_dir"
                and relative_source != allowed
            ):
                violations.append(f"{relative_source}:{node.lineno}")
    assert violations == []
