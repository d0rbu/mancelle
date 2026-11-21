"""Validates the mini axbench-base fixture parses cleanly via SourcePaths."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import safetensors.torch as st
from safetensors import safe_open

from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
    validate_positive_qc_rows,
)
from steering.eval.axbench_bpb.source_paths import (
    SourcePaths,
    validate_stage_0_3_artifacts,
)

_FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "mini_axbench_base_run"
)


@pytest.fixture
def mini_source_paths() -> SourcePaths:
    return SourcePaths(source_run_dir=_FIXTURE_ROOT)


def test_validate_stage_0_3_artifacts_passes(mini_source_paths):
    """All required Stage 0-3 paths exist; no missing-files error raised."""
    validate_stage_0_3_artifacts(mini_source_paths)


def test_seed_v2_commit_artifacts_exist(mini_source_paths):
    assert mini_source_paths.seeds_train().is_file()
    assert mini_source_paths.seeds_eval().is_file()
    assert mini_source_paths.seed_pool_manifest().is_file()


def test_concepts_json_loads_5_concepts(mini_source_paths):
    concepts = json.loads(mini_source_paths.concepts_json().read_text())
    assert len(concepts) == 5
    assert {c["concept_id"] for c in concepts} == {0, 1, 2, 3, 4}


def test_synth_parquets_have_expected_rows(mini_source_paths):
    pos = pq.read_table(mini_source_paths.synth_training_positive())
    neg = pq.read_table(mini_source_paths.synth_training_negative())
    # 5 concepts × (4 rich + 4 avoid) = 40 rows each
    assert pos.num_rows == 40
    assert neg.num_rows == 40


def test_positive_synth_fixture_has_current_qc_contract(mini_source_paths):
    path = mini_source_paths.synth_training_positive()
    table = pq.read_table(path)
    contract = validate_positive_qc_contract(
        path,
        consumer="mini AxBench-Base fixture test",
        schema=table.schema,
        expected_accept_threshold=0.3,
    )
    assert contract.accept_threshold_fraction == 0.3
    assert contract.accept_threshold_points == 30.0
    validate_positive_qc_rows(
        table.to_pylist(),
        accept_threshold_points=contract.accept_threshold_points,
    )


def test_concept_features_have_canonical_metadata(mini_source_paths):
    for path in (
        mini_source_paths.concept_features(),
        mini_source_paths.concept_features_neg(),
    ):
        with safe_open(str(path), framework="pt") as f:
            metadata = f.metadata() or {}
            assert metadata.get("mode") == "canonical", \
                f"{path}: missing mode=canonical metadata"


def test_baseline_directions_loadable(mini_source_paths):
    """All 20 diffmean + 20 linear_probe direction files load + have 'direction' key."""
    for baseline in ("diffmean", "linear_probe"):
        for target in ("decoder_only", "encdec"):
            for mode in ("positive", "negative"):
                for cid in range(5):
                    if baseline == "diffmean":
                        path = mini_source_paths.diffmean_direction(target, mode, cid)
                    else:
                        path = mini_source_paths.linear_probe_direction(target, mode, cid)
                    tensors = st.load_file(str(path))
                    assert "direction" in tensors, f"missing 'direction' in {path}"
                    assert tensors["direction"].dim() == 1, \
                        f"direction in {path} is not 1-D"
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        metadata = handle.metadata() or {}
                    assert metadata == {
                        "artifact_kind": f"{baseline}_direction_v1",
                        "concept_id": str(cid),
                        "layer": "1",
                        "max_activation": str(2.0 + 0.5 * cid),
                        "mode": mode,
                        "n_neg": "2",
                        "n_pos": "2",
                        "target_model": target,
                    }


def test_lora_adapters_exist(mini_source_paths):
    """10 LoRA adapters under baselines/lora/decoder_only/{mode}/."""
    for mode in ("positive", "negative"):
        for cid in range(5):
            path = mini_source_paths.lora_adapter(mode, cid)
            assert path.exists(), f"missing LoRA adapter: {path}"


def test_prompting_descriptions_have_5_concepts(mini_source_paths):
    for mode in ("positive", "negative"):
        path = mini_source_paths.prompting_descriptions(mode)
        descs = json.loads(path.read_text())
        assert len(descs) == 5
        assert all(str(cid) in descs for cid in range(5))
