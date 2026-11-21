"""Tests for the axbench_base stage dependency DAG.

Covers:
- STAGE_DEPENDENCIES dict structure consistency
- ``check_stage_dependencies`` success / failure paths
- ``list_missing_artifacts`` full-path reporting
- The ``run_negative_mode=False`` skip rule for stage 1c, the training
  stages 3a/3b/3d, and the stage-4 concept_features_neg dependency
- Stage 4 per-concept × per-target × per-mode dep check
- Directory vs file existence checks
- Unknown stage + "all" stage edge cases
- Missing ``run_tag`` guard
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.dag import (
    _ARTIFACT_PATH_METHODS,
    STAGE_DEPENDENCIES,
    _stage_4_missing_artifacts,
    check_stage_dependencies,
    list_missing_artifacts,
)
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import _JUDGE_SCORES_SCHEMA
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Stage,
    SteeringMode,
)
from tests.axbench_base import paths as test_paths


def make_cfg(
    run_tag: str | None = "dag-run",
    run_negative_mode: bool = True,
    feature_statistics_path: str = "stats.safetensors",
    **overrides: Any,
) -> AxBenchBaseConfig:
    """Build a minimal AxBenchBaseConfig for DAG tests.

    Only the mandatory fields are filled in; everything else defaults.
    """
    base = AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path=feature_statistics_path,
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=None,
        run_tag=run_tag,
        run_negative_mode=run_negative_mode,
    )
    if overrides:
        return base.replace(**overrides)
    return base


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AxBenchBaseConfig:
    """A config with NANOCHAT_BASE_DIR pinned to tmp_path so paths resolve there."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    return make_cfg(feature_statistics_path=str(stats_path))


def _touch_file(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("")


def _touch_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_committed_yield_report(
    p: Path,
    *,
    stage: str,
    concepts: list[dict[str, Any]] | None = None,
    surviving_ids: set[int] | None = None,
) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if concepts is None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "stage": stage,
            "status": "ok",
            "dropped_concept_fraction": 0.0,
        }
    else:
        resolved_survivors = (
            {int(concept["concept_id"]) for concept in concepts}
            if surviving_ids is None
            else set(surviving_ids)
        )
        records = [
            {
                "concept_id": int(concept["concept_id"]),
                "feature_index": int(concept["feature_index"]),
                "attempted": 2,
                "accepted": (
                    2 if int(concept["concept_id"]) in resolved_survivors else 0
                ),
                "dropped": (
                    0 if int(concept["concept_id"]) in resolved_survivors else 2
                ),
                "drop_fraction": (
                    0.0 if int(concept["concept_id"]) in resolved_survivors else 1.0
                ),
                "excluded": int(concept["concept_id"]) not in resolved_survivors,
                "exclusion_reason": (
                    None
                    if int(concept["concept_id"]) in resolved_survivors
                    else "drop_fraction_exceeded"
                ),
                "reason_counts": (
                    {}
                    if int(concept["concept_id"]) in resolved_survivors
                    else {"fixture_qc_failure": 2}
                ),
                "emitted_rows": (
                    4 if int(concept["concept_id"]) in resolved_survivors else 0
                ),
            }
            for concept in concepts
        ]
        n_dropped = len(records) - len(resolved_survivors)
        payload = {
            "schema_version": 1,
            "stage": stage,
            "status": "ok",
            "qc_fallback_on_exhaust": "error",
            "max_concept_drop_fraction": 0.5,
            "max_dropped_concept_fraction": 0.5,
            "n_concepts_attempted": len(records),
            "n_concepts_surviving": len(resolved_survivors),
            "n_concepts_dropped": n_dropped,
            "dropped_concept_fraction": n_dropped / len(records),
            "surviving_concept_ids": [
                int(concept["concept_id"])
                for concept in concepts
                if int(concept["concept_id"]) in resolved_survivors
            ],
            "population_guard_failures": [],
            "concepts": records,
        }
    p.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_stage1_yield_reports(
    cfg: AxBenchBaseConfig,
    *,
    include_negative: bool,
) -> None:
    output_paths = _cfg_paths(cfg)
    _write_committed_yield_report(
        output_paths.stage1b_concept_yield_report(),
        stage="stage1b",
    )
    _touch_file(output_paths.stage1b_concepts())
    if include_negative:
        _write_committed_yield_report(
            output_paths.stage1c_concept_yield_report(),
            stage="stage1c",
        )
        _touch_file(output_paths.stage1c_concepts())


def _cfg_paths(cfg: AxBenchBaseConfig) -> paths.AxBenchBasePaths:
    return paths.AxBenchBasePaths.from_config(cfg)


def _tag_paths(run_tag: str) -> paths.AxBenchBasePaths:
    return paths.AxBenchBasePaths(paths.default_output_root(), run_tag)


def _write_stage_rows(
    out_dir: Path,
    *,
    schema: pa.Schema,
    generation_keys: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [{"generation_key": key} for key in generation_keys],
        schema=schema,
    )
    pq.write_table(table, out_dir / "batch_000000.parquet")


def _write_stats_with_n_sae_features(path: Path, n_sae_features: int) -> None:
    """Write a minimal safetensors stats file whose metadata carries
    ``n_sae_features`` so the Stage 4 dep check can read it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )


def _make_concepts_payload(n: int) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": i,
            "feature_index": 100 + i,
            "layer": 8,
            "sae_source": "8-gemmascope-res-16k",
            "description": f"concept description {i}",
            "corpus_count": 50,
            "description_source": "neuronpedia",
        }
        for i in range(n)
    ]


def _write_concepts_json(run_tag: str, n: int) -> list[dict[str, Any]]:
    payload = _make_concepts_payload(n)
    concepts_path = _tag_paths(run_tag).concepts_json()
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_path.write_text(json.dumps(payload), encoding="utf-8")
    _tag_paths(run_tag).stage1b_concepts().write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    _tag_paths(run_tag).stage1c_concepts().write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    _write_committed_yield_report(
        _tag_paths(run_tag).stage1b_concept_yield_report(),
        stage="stage1b",
        concepts=payload,
    )
    _write_committed_yield_report(
        _tag_paths(run_tag).stage1c_concept_yield_report(),
        stage="stage1c",
        concepts=payload,
    )
    return payload


def _write_stage1c_filtered_subset(
    run_tag: str,
    input_concepts: list[dict[str, Any]],
    *,
    surviving_ids: set[int],
) -> list[dict[str, Any]]:
    filtered = [
        concept
        for concept in input_concepts
        if int(concept["concept_id"]) in surviving_ids
    ]
    _tag_paths(run_tag).stage1c_concepts().write_text(
        json.dumps(filtered),
        encoding="utf-8",
    )
    _write_committed_yield_report(
        _tag_paths(run_tag).stage1c_concept_yield_report(),
        stage="stage1c",
        concepts=input_concepts,
        surviving_ids=surviving_ids,
    )
    return filtered


def _touch_stage_4_generic_deps(
    run_tag: str, include_neg_features: bool = True,
) -> None:
    """Satisfy every Stage 4 generic dep (NOT the per-concept training
    artifacts).

    Callers that want the full Stage 4 "all deps present" state must
    also touch the per-concept training artifacts via
    :func:`_touch_stage_4_per_concept_artifacts`.
    """
    _touch_file(_tag_paths(run_tag).concept_features())
    _touch_file(_tag_paths(run_tag).synth_training_positive())
    _touch_file(_tag_paths(run_tag).seeds_eval())
    _touch_file(_tag_paths(run_tag).seed_pool_manifest())
    _touch_file(test_paths.default_paths(run_tag).stage4_contexts_manifest())
    if include_neg_features:
        _touch_file(_tag_paths(run_tag).synth_training_negative())
        _touch_file(_tag_paths(run_tag).concept_features_neg())


def _touch_stage_4_per_concept_artifacts(
    run_tag: str,
    concepts: list[dict[str, Any]],
    run_negative_mode: bool,
) -> None:
    """Write stub files for every per-concept training artifact the
    Stage 4 dep check looks for.
    """
    modes: list[SteeringMode] = [SteeringMode.POSITIVE]
    if run_negative_mode:
        modes.append(SteeringMode.NEGATIVE)

    for mode in modes:
        _touch_file(_tag_paths(run_tag).prompting_descriptions(mode))

    for concept in concepts:
        cid = int(concept["concept_id"])
        for mode in modes:
            for target in ("decoder_only", "encdec"):
                _touch_file(_tag_paths(run_tag).diffmean_direction(target, mode, cid))
                _touch_file(_tag_paths(run_tag).linear_probe_direction(target, mode, cid))
            _touch_file(_tag_paths(run_tag).lora_adapter(mode, cid))


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


def test_every_stage_in_dependencies_is_known_artifact() -> None:
    """Every artifact key in STAGE_DEPENDENCIES must be resolvable via
    _ARTIFACT_PATH_METHODS — otherwise check_stage_dependencies would KeyError
    at runtime for an unrelated cfg.
    """
    for stage, deps in STAGE_DEPENDENCIES.items():
        assert isinstance(deps, list), f"stage {stage} has non-list deps"
        for art in deps:
            assert art in _ARTIFACT_PATH_METHODS, (
                f"stage {stage!r} lists unknown artifact key {art!r}"
            )


def test_stage_dependencies_covers_all_expected_stages() -> None:
    """Sanity check: the DAG declares every stage the pipeline knows
    about. Changing this list is a breaking contract change.
    """
    expected = {
        "0a", "0b", "0c",
        "1b", "1c",
        "2",
        "3a", "3b", "3c", "3d", "3e",
        "4", "5", "6",
    }
    assert set(STAGE_DEPENDENCIES.keys()) == expected


# ---------------------------------------------------------------------------
# check_stage_dependencies — success paths
# ---------------------------------------------------------------------------


def test_stage_0a_has_no_dependencies(cfg: AxBenchBaseConfig) -> None:
    """Stage 0a is the root of the DAG; it must pass with nothing on disk.

    We look up the dep list via ``Stage.PREPARE_CONCEPTS`` so the
    expression type-checks under ``ty``. String-keyed lookups via
    ``STAGE_DEPENDENCIES.get("0a")`` still resolve at runtime thanks
    to StrEnum hash equality — see ``test_stage_dict_lookup_accepts_string``
    in ``test_enums.py`` for that contract.
    """
    check_stage_dependencies(cfg, "0a")  # must not raise
    assert STAGE_DEPENDENCIES[Stage.PREPARE_CONCEPTS] == []


def test_stage_all_is_noop(cfg: AxBenchBaseConfig) -> None:
    """Stage 'all' is validated at each sub-stage's runtime, not ahead."""
    check_stage_dependencies(cfg, "all")  # must not raise even with empty disk


def test_list_missing_artifacts_stage_all_empty(cfg: AxBenchBaseConfig) -> None:
    """``list_missing_artifacts`` also short-circuits on stage "all" —
    the pipeline dispatcher checks each sub-stage individually at its own
    runtime.
    """
    assert list_missing_artifacts(cfg, "all") == []


@pytest.mark.parametrize("training_stage", ["3a", "3b", "3d"])
def test_resume_with_complete_stage_2_proceeds_to_training_stage(
    cfg: AxBenchBaseConfig, training_stage: str,
) -> None:
    """Precondition: all stage 2 outputs exist on disk. Then
    ``check_stage_dependencies(cfg, "3a" | "3b" | "3d")`` must pass,
    demonstrating resume-from-where-we-left-off. Stages 3a/3b/3d all
    train on the synth-training parquet files (positive + negative
    when neg mode is on); Stage 3c is exercised separately because it
    only reads ``concepts.json``.
    """
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _touch_file(_cfg_paths(cfg).synth_training_negative())
    _write_stage1_yield_reports(cfg, include_negative=True)
    check_stage_dependencies(cfg, training_stage)  # no raise


def test_resume_stage_3c_needs_final_filtered_concept_handoff(
    cfg: AxBenchBaseConfig,
) -> None:
    """Prompting reads the final population but not either synth corpus."""
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _write_stage1_yield_reports(cfg, include_negative=True)
    check_stage_dependencies(cfg, "3c")  # no raise


@pytest.mark.parametrize("training_stage", ["3a", "3b", "3d"])
def test_list_missing_artifacts_empty_when_all_training_inputs_present(
    cfg: AxBenchBaseConfig,
    training_stage: str,
) -> None:
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _touch_file(_cfg_paths(cfg).synth_training_negative())
    _write_stage1_yield_reports(cfg, include_negative=True)
    missing = list_missing_artifacts(cfg, training_stage)
    assert missing == []


def test_list_missing_artifacts_empty_for_3c_when_handoff_present(
    cfg: AxBenchBaseConfig,
) -> None:
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _write_stage1_yield_reports(cfg, include_negative=True)
    missing = list_missing_artifacts(cfg, "3c")
    assert missing == []


@pytest.mark.parametrize("training_stage", ["3a", "3b", "3d"])
def test_training_stage_requires_synth_training_negative_when_neg_mode_on(
    cfg: AxBenchBaseConfig, training_stage: str,
) -> None:
    """When ``run_negative_mode=True``, the training stages 3a/3b/3d
    all depend on ``synth_training_negative.parquet``. Omit it and
    verify the error names the missing file.
    """
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    # Intentionally NOT creating synth_training_negative.parquet
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, training_stage)
    assert str(_cfg_paths(cfg).synth_training_negative()) in str(excinfo.value)


@pytest.mark.parametrize("training_stage", ["3a", "3b", "3d"])
def test_training_stage_skips_synth_training_negative_when_neg_mode_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    training_stage: str,
) -> None:
    """When ``run_negative_mode=False``, the training stages drop
    ``synth_training_negative.parquet`` from their dep set so a
    pipeline that never enabled negative mode doesn't wait on a file
    that was never produced.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    cfg = make_cfg(
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
    )
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _write_stage1_yield_reports(cfg, include_negative=False)
    check_stage_dependencies(cfg, training_stage)  # must not raise


# ---------------------------------------------------------------------------
# check_stage_dependencies — failure paths
# ---------------------------------------------------------------------------


def test_stage_4_requires_stage_0_1_2_artifacts(cfg: AxBenchBaseConfig) -> None:
    """Stage 4 has multiple prerequisites. Omit ``concept_features.safetensors``
    and verify the error names the missing file alongside the per-concept
    artifacts.
    """
    assert cfg.run_tag is not None
    # Put every GENERIC Stage 4 dep except concept_features in place and
    # write the concept list so the per-concept enumeration runs.
    concepts = _write_concepts_json(cfg.run_tag, n=2)
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _touch_file(_cfg_paths(cfg).synth_training_negative())
    _touch_file(_cfg_paths(cfg).seeds_eval())
    _touch_file(_cfg_paths(cfg).seed_pool_manifest())
    _touch_file(test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest())
    _touch_file(_cfg_paths(cfg).concept_features_neg())
    # Also stub every per-concept training artifact so the only missing
    # path in the failure is ``concept_features.safetensors``.
    _touch_stage_4_per_concept_artifacts(
        cfg.run_tag, concepts, run_negative_mode=True,
    )
    # Intentionally NOT creating concept_features.safetensors.

    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "4")
    msg = str(excinfo.value)
    assert "concept_features.safetensors" in msg
    assert str(_cfg_paths(cfg).concept_features()) in msg


def test_missing_artifact_raises_clear_error_with_paths(
    cfg: AxBenchBaseConfig,
) -> None:
    """The error must list the FULL on-disk path for each missing artifact."""
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "0b")
    msg = str(excinfo.value)
    assert cfg.run_tag is not None
    assert str(_cfg_paths(cfg).candidate_concepts_json()) in msg
    # Should mention the stage as context
    assert "0b" in msg


def test_missing_multiple_artifacts_all_in_error_message(
    cfg: AxBenchBaseConfig,
) -> None:
    """When several artifacts are missing at once, EVERY missing path
    appears in the single raised message (the operator should not have
    to re-run + crash repeatedly to discover each dep).

    Stage 4 with nothing on disk at all returns early (``concepts.json``
    missing → can't enumerate per-concept files), so we verify the
    GENERIC deps are all present in the failure message.
    """
    assert cfg.run_tag is not None
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "4")
    msg = str(excinfo.value)
    assert str(_cfg_paths(cfg).concepts_json()) in msg
    assert str(_cfg_paths(cfg).concept_features()) in msg
    assert str(_cfg_paths(cfg).concept_features_neg()) in msg
    assert str(_cfg_paths(cfg).seeds_eval()) in msg


def test_list_missing_artifacts_returns_full_paths(
    cfg: AxBenchBaseConfig,
) -> None:
    """Stage 4 with nothing on disk returns exactly the generic dep set
    because ``concepts.json`` is missing and the per-concept enumeration
    is skipped (early-return).
    """
    missing = list_missing_artifacts(cfg, "4")
    assert cfg.run_tag is not None
    # All of stage 4's generic deps are absent (and the per-concept
    # enumeration does NOT run because concepts.json is missing).
    expected_missing = {
        _cfg_paths(cfg).concepts_json(),
        _cfg_paths(cfg).stage1b_concepts(),
        _cfg_paths(cfg).stage1c_concepts(),
        _cfg_paths(cfg).stage1b_concept_yield_report(),
        _cfg_paths(cfg).stage1c_concept_yield_report(),
        _cfg_paths(cfg).synth_training_positive(),
        _cfg_paths(cfg).synth_training_negative(),
        _cfg_paths(cfg).concept_features(),
        _cfg_paths(cfg).concept_features_neg(),
        _cfg_paths(cfg).seeds_eval(),
        _cfg_paths(cfg).seed_pool_manifest(),
        test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest(),
    }
    assert set(missing) == expected_missing
    for p in missing:
        assert p.is_absolute()


# ---------------------------------------------------------------------------
# run_negative_mode skip rule
# ---------------------------------------------------------------------------


def test_stage_1c_skipped_when_neg_mode_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_negative_mode=False → Stage 1c is a NO-OP in the DAG check,
    even with none of its normal prerequisites on disk.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_negative_mode=False)
    check_stage_dependencies(cfg, "1c")  # must not raise


def test_neg_mode_off_does_not_require_concept_features_neg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When run_negative_mode=False, stage 4 drops concept_features_neg
    from its required-artifact set and the per-concept enumeration
    only checks positive-mode training artifacts.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    cfg = make_cfg(
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
    )
    assert cfg.run_tag is not None
    concepts = _write_concepts_json(cfg.run_tag, n=2)
    _touch_file(_cfg_paths(cfg).concept_features())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _touch_file(_cfg_paths(cfg).seeds_eval())
    _touch_file(_cfg_paths(cfg).seed_pool_manifest())
    _touch_file(test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest())
    _touch_stage_4_per_concept_artifacts(
        cfg.run_tag,
        concepts,
        run_negative_mode=False,
    )
    # Intentionally NOT creating concept_features_neg
    check_stage_dependencies(cfg, "4")  # no raise


def test_neg_mode_off_list_missing_omits_neg_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """list_missing_artifacts respects the neg_mode skip rule for stage 4."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    cfg = make_cfg(
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
    )
    assert cfg.run_tag is not None
    concepts = _write_concepts_json(cfg.run_tag, n=2)
    _touch_file(_cfg_paths(cfg).concept_features())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _touch_file(_cfg_paths(cfg).seeds_eval())
    _touch_file(_cfg_paths(cfg).seed_pool_manifest())
    _touch_file(test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest())
    _touch_stage_4_per_concept_artifacts(
        cfg.run_tag,
        concepts,
        run_negative_mode=False,
    )
    missing = list_missing_artifacts(cfg, "4")
    assert missing == []


def test_neg_mode_off_list_missing_stage_1c_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_negative_mode=False)
    assert list_missing_artifacts(cfg, "1c") == []


# ---------------------------------------------------------------------------
# Unknown stage + run_tag guards
# ---------------------------------------------------------------------------


def test_unknown_stage_raises(cfg: AxBenchBaseConfig) -> None:
    with pytest.raises(AxBenchBaseError, match="Unknown stage"):
        check_stage_dependencies(cfg, "7z")


def test_unknown_stage_list_missing_raises(cfg: AxBenchBaseConfig) -> None:
    with pytest.raises(AxBenchBaseError, match="Unknown stage"):
        list_missing_artifacts(cfg, "bogus")


def test_run_tag_not_set_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cfg.run_tag must be set before checking deps; otherwise we can't
    resolve any artifact path.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        check_stage_dependencies(cfg, "0b")


def test_run_tag_not_set_raises_even_for_stage_0a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 0a has no deps, but we still require run_tag to be set —
    the run output root depends on it, and we want crash-early behavior.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        check_stage_dependencies(cfg, "0a")


def test_run_tag_not_set_list_missing_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        list_missing_artifacts(cfg, "0a")


def test_run_tag_not_set_stage_all_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 'all' is a no-op even if run_tag is unset — pipeline will
    set it before actually running anything.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_tag=None)
    check_stage_dependencies(cfg, "all")  # no raise


# ---------------------------------------------------------------------------
# File vs directory existence checks
# ---------------------------------------------------------------------------


def test_file_dep_check_uses_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file-typed artifact key satisfies the check only when it's a
    real regular file, not a directory of the same name.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg()
    assert cfg.run_tag is not None
    # Create candidate_concepts.json as a DIRECTORY, which must NOT satisfy stage 0b
    concepts_path = _cfg_paths(cfg).candidate_concepts_json()
    _touch_dir(concepts_path)
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "0b")
    assert str(concepts_path) in str(excinfo.value)


def test_directory_dep_check_uses_dir_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directory-typed artifact key satisfies the check only when the
    path is a real directory, not an accidentally-created file of the
    same name.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg()
    assert cfg.run_tag is not None
    # Create generations as a REGULAR FILE — that should not count as a dir
    gen_path = _cfg_paths(cfg).generations_dir()
    _touch_file(gen_path)
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "5")
    assert str(gen_path) in str(excinfo.value)

    # Now replace with a real directory — should pass
    gen_path.unlink()
    _touch_dir(gen_path)
    _write_concepts_json(cfg.run_tag, n=1)
    check_stage_dependencies(cfg, "5")  # no raise


def test_stage_6_requires_both_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 6 aggregation needs BOTH ``generations`` and ``judge_scores``
    directories on disk."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg()
    assert cfg.run_tag is not None
    _touch_dir(_cfg_paths(cfg).generations_dir())
    # judge_scores NOT yet created
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "6")
    assert str(_cfg_paths(cfg).judge_scores_dir()) in str(excinfo.value)

    _touch_dir(_cfg_paths(cfg).judge_scores_dir())
    check_stage_dependencies(cfg, "6")  # no raise


def test_stage_6_rejects_incomplete_judge_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing but partial Stage 5 directory does not admit Stage 6."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg()
    assert cfg.run_tag is not None
    _write_stage_rows(
        _cfg_paths(cfg).generations_dir(),
        schema=GENERATIONS_SCHEMA,
        generation_keys=["k0", "k1", "k2"],
    )
    _write_stage_rows(
        _cfg_paths(cfg).judge_scores_dir(),
        schema=_JUDGE_SCORES_SCHEMA,
        generation_keys=["k0", "k1"],
    )

    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, Stage.AGGREGATE)

    message = str(excinfo.value)
    assert "3 generation rows" in message
    assert "2 judge rows" in message
    assert "1 missing" in message
    assert "k2" in message


# ---------------------------------------------------------------------------
# Stage 2 special case: neg features conditional
# ---------------------------------------------------------------------------


def test_stage_1b_depends_only_on_concepts_and_seeds(
    cfg: AxBenchBaseConfig,
) -> None:
    """Stage 1b (positive mode synthesis) depends only on concepts +
    train seeds plus their v2 manifest now that the global-negative pool
    (Stage 1a) is gone.

    The positive avoiding completion is the seed's own natural
    continuation (natural tier) or an LLM fallback, so Stage 1b no
    longer reads a separate ``synth_negatives.parquet`` artifact.
    """
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).seeds_train())
    _touch_file(_cfg_paths(cfg).seed_pool_manifest())
    # No synth_negatives.parquet exists anymore — 1b must still pass.
    check_stage_dependencies(cfg, "1b")  # no raise


def test_stage_1b_requires_concepts_and_seeds(cfg: AxBenchBaseConfig) -> None:
    """With neither concepts.json nor seeds_train present, Stage 1b
    reports both missing prerequisites (and nothing about a now-removed
    global-negative artifact).
    """
    assert cfg.run_tag is not None
    with pytest.raises(AxBenchBaseError) as excinfo:
        check_stage_dependencies(cfg, "1b")
    msg = str(excinfo.value)
    assert str(_cfg_paths(cfg).concepts_json()) in msg
    assert str(_cfg_paths(cfg).seeds_train()) in msg
    assert str(_cfg_paths(cfg).seed_pool_manifest()) in msg


def test_stage_2_ignores_neg_features_question(
    cfg: AxBenchBaseConfig,
) -> None:
    """Stage 2 prerequisites are purely its OWN inputs (concepts +
    synth_training_positive). concept_features_neg is an OUTPUT of
    stage 2 — not a prerequisite — so it must not appear in the deps.
    """
    assert (
        "concept_features_neg.safetensors" not in STAGE_DEPENDENCIES[Stage.CONCEPT_FEATURE_MAPPING]
    )
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _write_stage1_yield_reports(cfg, include_negative=True)
    check_stage_dependencies(cfg, "2")  # no raise


# ---------------------------------------------------------------------------
# Stage enum threading — the tasks-41-48 review fix
# ---------------------------------------------------------------------------


def test_check_stage_dependencies_accepts_stage_enum(
    cfg: AxBenchBaseConfig,
) -> None:
    """``check_stage_dependencies`` accepts a :class:`Stage` member
    directly, not just its bare-string value. The rest of the DAG
    tests intentionally pass strings to pin the backward-compat
    contract; this test pins the enum-first path.
    """
    assert cfg.run_tag is not None
    _touch_file(_cfg_paths(cfg).concepts_json())
    _touch_file(_cfg_paths(cfg).synth_training_positive())
    _write_stage1_yield_reports(cfg, include_negative=True)
    # Passing Stage.CONCEPT_FEATURE_MAPPING instead of "2".
    check_stage_dependencies(cfg, Stage.CONCEPT_FEATURE_MAPPING)  # no raise


def test_list_missing_artifacts_accepts_stage_enum(
    cfg: AxBenchBaseConfig,
) -> None:
    """``list_missing_artifacts`` also accepts :class:`Stage` members.

    With nothing on disk Stage 4 early-returns on the generic dep list
    because ``concepts.json`` is missing and the per-concept enumeration
    can't run without it. The generic dep set is 12 paths, and all twelve
    are absolute.
    """
    assert cfg.run_tag is not None
    missing = list_missing_artifacts(cfg, Stage.GENERATE)
    # Stage 4 has twelve generic artifacts, all of which are absent here.
    assert len(missing) == 12
    for p in missing:
        assert p.is_absolute()


def test_stage_all_enum_is_noop_for_dep_check(
    cfg: AxBenchBaseConfig,
) -> None:
    """:attr:`Stage.ALL` behaves identically to ``"all"``: both resolve
    to the pseudo-stage no-op branch inside the DAG dep check.
    """
    check_stage_dependencies(cfg, Stage.ALL)  # no raise
    assert list_missing_artifacts(cfg, Stage.ALL) == []


def test_neg_mode_off_stage_1c_enum_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``run_negative_mode=False`` skip rule applies whether the
    caller passes ``"1c"`` or ``Stage.SYNTH_NEGATIVE_MODE``.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_cfg(run_negative_mode=False)
    check_stage_dependencies(cfg, Stage.SYNTH_NEGATIVE_MODE)  # no raise


def test_stage_dependencies_dict_uses_enum_keys() -> None:
    """Every key in the public :data:`STAGE_DEPENDENCIES` dict is a
    :class:`Stage` member. :attr:`Stage.ALL` must NOT be a key — it's
    the pseudo-stage expanded by the CLI dispatcher.
    """
    for key in STAGE_DEPENDENCIES:
        assert isinstance(key, Stage), key
    assert Stage.ALL not in STAGE_DEPENDENCIES


# ---------------------------------------------------------------------------
# Stage 4 per-concept × per-target × per-mode dep check (Phase 4)
# ---------------------------------------------------------------------------


class TestStage4PerConceptDepCheck:
    """Cover the per-concept Stage 4 dep-check helper.

    These tests stub the feature statistics file so the helper can read
    ``n_sae_features`` for the concept loader, write a ``concepts.json``
    with a known concept count, and then enumerate the specific
    per-concept artifacts to verify the dep check looks in the right
    places.
    """

    def test_stage_4_missing_when_concepts_json_missing_returns_early(
        self, cfg: AxBenchBaseConfig,
    ) -> None:
        """If ``concepts.json`` is in the generic missing list, the dep
        check must return without enumerating per-concept artifacts —
        we can't iterate concepts we haven't loaded yet.
        """
        assert cfg.run_tag is not None
        # Generic deps all missing; no concepts.json written.
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert _cfg_paths(cfg).concepts_json() in missing
        # No per-concept artifacts should have been checked, so the
        # Missing list contains only the twelve generic deps.
        assert len(missing) == 12
        # And no diffmean/probe/lora paths leaked in.
        for p in missing:
            assert "baselines" not in p.parts

    def test_stage_4_all_deps_satisfied_returns_empty(
        self, cfg: AxBenchBaseConfig,
    ) -> None:
        """When every generic dep + every per-concept × per-target ×
        per-mode artifact is on disk, Stage 4 dep check returns ``[]``.
        """
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=3)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag, concepts, run_negative_mode=True,
        )
        assert list_missing_artifacts(cfg, Stage.GENERATE) == []

    def test_stage_4_missing_one_diffmean_direction_for_one_concept_listed(
        self, cfg: AxBenchBaseConfig,
    ) -> None:
        """Deleting a single DiffMean direction file must make that
        exact path appear in the missing list."""
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=2)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        # Delete one specific DiffMean file
        victim = _cfg_paths(cfg).diffmean_direction(
            "decoder_only",
            SteeringMode.POSITIVE,
            concept_id=1,
        )
        victim.unlink()
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert missing == [victim]

    def test_stage_4_missing_one_linear_probe_direction_for_one_concept_listed(
        self,
        cfg: AxBenchBaseConfig,
    ) -> None:
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=2)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        victim = _cfg_paths(cfg).linear_probe_direction(
            "encdec",
            SteeringMode.NEGATIVE,
            concept_id=0,
        )
        victim.unlink()
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert missing == [victim]

    def test_stage_4_missing_one_lora_adapter_for_one_concept_listed(
        self,
        cfg: AxBenchBaseConfig,
    ) -> None:
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=2)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        victim = _cfg_paths(cfg).lora_adapter(
            SteeringMode.POSITIVE,
            concept_id=1,
        )
        victim.unlink()
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert missing == [victim]

    def test_stage_4_missing_prompting_descriptions_positive_listed(
        self,
        cfg: AxBenchBaseConfig,
    ) -> None:
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=1)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        victim = _cfg_paths(cfg).prompting_descriptions(
            SteeringMode.POSITIVE,
        )
        victim.unlink()
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert victim in missing

    def test_stage_4_missing_prompting_descriptions_negative_when_neg_mode_on_listed(
        self,
        cfg: AxBenchBaseConfig,
    ) -> None:
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=1)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        victim = _cfg_paths(cfg).prompting_descriptions(
            SteeringMode.NEGATIVE,
        )
        victim.unlink()
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert victim in missing

    def test_stage_4_missing_prompting_descriptions_negative_when_neg_mode_off_not_listed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In ``run_negative_mode=False`` the negative prompting file
        is not required, so the dep check must not look for it and
        its absence must not show up in the missing list.
        """
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        stats_path = tmp_path / "stats.safetensors"
        _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
        cfg = make_cfg(
            run_negative_mode=False,
            feature_statistics_path=str(stats_path),
        )
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=1)
        _touch_stage_4_generic_deps(
            cfg.run_tag, include_neg_features=False,
        )
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag, concepts, run_negative_mode=False,
        )
        # Do NOT write the negative prompting file.
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert missing == []
        # Explicit guard: the file we intentionally omitted never got
        # looked at, so it's NOT in the result.
        neg_file = _cfg_paths(cfg).prompting_descriptions(
            SteeringMode.NEGATIVE,
        )
        assert neg_file not in missing

    def test_stage_4_missing_concept_features_and_one_direction_both_listed(
        self,
        cfg: AxBenchBaseConfig,
    ) -> None:
        """When the generic dep set is partially missing AND a
        per-concept file is missing, BOTH appear in the same missing
        list (so the operator sees every break in one shot).
        """
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=2)
        _touch_file(_cfg_paths(cfg).seeds_eval())
        _touch_file(_cfg_paths(cfg).seed_pool_manifest())
        _touch_file(test_paths.default_paths(cfg.run_tag).stage4_contexts_manifest())
        _touch_file(_cfg_paths(cfg).concept_features_neg())
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        # Delete one DiffMean file AND leave concept_features.safetensors
        # absent.
        victim = _cfg_paths(cfg).diffmean_direction(
            "decoder_only",
            SteeringMode.POSITIVE,
            concept_id=0,
        )
        victim.unlink()
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert _cfg_paths(cfg).concept_features() in missing
        assert victim in missing

    def test_stage_4_lora_only_checked_for_decoder_only_not_encdec(
        self, cfg: AxBenchBaseConfig,
    ) -> None:
        """LoRA lives under ``baselines/lora/decoder_only/`` — there is
        NO ``baselines/lora/encdec/`` path the dep check enumerates.
        If the check accidentally grew an encdec LoRA branch every
        test run would fail because that file is never touched by
        :func:`_touch_stage_4_per_concept_artifacts`. The absence of
        a failure here pins the decoder-only-only scope.
        """
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=2)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag, concepts, run_negative_mode=True,
        )
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert missing == []
        # And no path the dep check returned is an encdec LoRA path.
        for p in missing:
            assert "/lora/encdec/" not in str(p)

    def test_stage_4_with_neg_mode_off_no_negative_mode_files_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``run_negative_mode=False`` the dep check must NOT look
        for negative-mode per-concept files. We verify by stubbing only
        the positive-mode ones and expecting an empty missing list.
        """
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        stats_path = tmp_path / "stats.safetensors"
        _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
        cfg = make_cfg(
            run_negative_mode=False,
            feature_statistics_path=str(stats_path),
        )
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=3)
        _touch_stage_4_generic_deps(
            cfg.run_tag,
            include_neg_features=False,
        )
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=False,
        )
        assert list_missing_artifacts(cfg, Stage.GENERATE) == []
        # No negative-mode file for concept 0 was ever created; the
        # dep check must not have looked for it.
        never_created = _cfg_paths(cfg).diffmean_direction(
            "decoder_only",
            SteeringMode.NEGATIVE,
            concept_id=0,
        )
        assert not never_created.exists()
        assert never_created not in list_missing_artifacts(cfg, Stage.GENERATE)

    def test_stage_4_raises_if_run_tag_unset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Stage 4 helper preserves the standard ``run_tag`` guard."""
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        stats_path = tmp_path / "stats.safetensors"
        _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
        cfg = make_cfg(
            run_tag=None,
            feature_statistics_path=str(stats_path),
        )
        with pytest.raises(AxBenchBaseError, match="run_tag"):
            _stage_4_missing_artifacts(cfg)

    @pytest.mark.parametrize("n_concepts", [1, 3, 10])
    def test_stage_4_n_concepts_iterated(
        self,
        cfg: AxBenchBaseConfig,
        n_concepts: int,
    ) -> None:
        """The per-concept dep check scales linearly with the number of
        concepts. We delete one DiffMean file per concept (positive mode,
        decoder_only target) and verify every deletion is surfaced.
        """
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=n_concepts)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        victims: list[Path] = []
        for concept in concepts:
            cid = int(concept["concept_id"])
            victim = _cfg_paths(cfg).diffmean_direction(
                "decoder_only",
                SteeringMode.POSITIVE,
                cid,
            )
            victim.unlink()
            victims.append(victim)
        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert len(missing) == n_concepts
        assert sorted(str(p) for p in missing) == sorted(str(p) for p in victims)

    def test_stage_4_missing_seeds_eval_plus_one_concept_both_listed(
        self,
        cfg: AxBenchBaseConfig,
    ) -> None:
        """Even with ``concepts.json`` present, the generic dep check
        fills ``missing`` first. The per-concept enumeration then
        appends to the same list so the operator sees both layers of
        failure in one pass.
        """
        assert cfg.run_tag is not None
        concepts = _write_concepts_json(cfg.run_tag, n=1)
        _touch_file(_cfg_paths(cfg).concept_features())
        _touch_file(_cfg_paths(cfg).concept_features_neg())
        # Intentionally NOT creating seeds_eval.
        _touch_stage_4_per_concept_artifacts(
            cfg.run_tag,
            concepts,
            run_negative_mode=True,
        )
        victim = _cfg_paths(cfg).lora_adapter(
            SteeringMode.NEGATIVE,
            concept_id=0,
        )
        victim.unlink()

        missing = list_missing_artifacts(cfg, Stage.GENERATE)
        assert _cfg_paths(cfg).seeds_eval() in missing
        assert victim in missing

    def test_stage_4_with_missing_stats_file_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the feature_statistics_path is missing the per-concept
        loader can't determine ``n_sae_features`` and the dep check
        surfaces the AxBenchBaseError from
        :func:`read_n_sae_features_from_stats`.
        """
        monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
        cfg = make_cfg(
            feature_statistics_path=str(tmp_path / "does_not_exist.safetensors"),
        )
        assert cfg.run_tag is not None
        _write_concepts_json(cfg.run_tag, n=1)
        _touch_stage_4_generic_deps(cfg.run_tag, include_neg_features=True)
        with pytest.raises(AxBenchBaseError, match="does not exist"):
            list_missing_artifacts(cfg, Stage.GENERATE)


# ---------------------------------------------------------------------------
# Stage 4 corpus-filtered dep check (issue #508)
# ---------------------------------------------------------------------------


def _write_synth_corpus(path: Path, present_concept_ids: list[int]) -> None:
    rows: list[dict[str, Any]] = []
    for cid in present_concept_ids:
        rows.append({"concept_id": cid, "content_class": "concept_rich"})
        rows.append({"concept_id": cid, "content_class": "concept_avoiding"})
    schema = pa.schema([("concept_id", pa.int32()), ("content_class", pa.string())])
    if path.name == "synth_training_positive.parquet":
        schema = schema_with_positive_qc_contract(
            schema,
            accept_threshold=0.3,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_stage_4_filtered_out_concept_artifacts_not_required(
    cfg: AxBenchBaseConfig,
) -> None:
    """A Stage 1c exclusion, not corpus absence, removes a concept."""
    run_tag = cfg.run_tag
    assert run_tag is not None
    concepts = _write_concepts_json(run_tag, n=2)
    final_concepts = _write_stage1c_filtered_subset(
        run_tag,
        concepts,
        surviving_ids={0},
    )
    _touch_stage_4_generic_deps(run_tag)
    _touch_stage_4_per_concept_artifacts(
        run_tag,
        final_concepts,
        run_negative_mode=True,
    )
    assert _stage_4_missing_artifacts(cfg) == []


def test_stage_4_corpus_absence_cannot_hide_final_concept(
    cfg: AxBenchBaseConfig,
) -> None:
    """Corpus presence is not a second population authority."""
    run_tag = cfg.run_tag
    assert run_tag is not None
    concepts = _write_concepts_json(run_tag, n=2)
    _touch_stage_4_generic_deps(run_tag)
    _write_synth_corpus(_tag_paths(run_tag).synth_training_positive(), [0])
    _write_synth_corpus(_tag_paths(run_tag).synth_training_negative(), [0])
    _touch_stage_4_per_concept_artifacts(
        run_tag,
        [concepts[0]],
        run_negative_mode=True,
    )
    missing = _stage_4_missing_artifacts(cfg)
    assert any("0001" in str(path) for path in missing)


def test_stage_4_final_population_applies_symmetrically_across_modes(
    cfg: AxBenchBaseConfig,
) -> None:
    """Asymmetric corpus presence cannot suppress either mode's artifacts."""
    run_tag = cfg.run_tag
    assert run_tag is not None
    concepts = _write_concepts_json(run_tag, n=2)
    _touch_stage_4_generic_deps(run_tag)
    _write_synth_corpus(_tag_paths(run_tag).synth_training_positive(), [0])
    _write_synth_corpus(_tag_paths(run_tag).synth_training_negative(), [1])
    _touch_stage_4_per_concept_artifacts(
        run_tag,
        concepts,
        run_negative_mode=True,
    )
    missing_positive = _tag_paths(run_tag).diffmean_direction(
        "decoder_only",
        SteeringMode.POSITIVE,
        1,
    )
    missing_negative = _tag_paths(run_tag).diffmean_direction(
        "decoder_only",
        SteeringMode.NEGATIVE,
        0,
    )
    missing_positive.unlink()
    missing_negative.unlink()

    assert set(_stage_4_missing_artifacts(cfg)) == {
        missing_positive,
        missing_negative,
    }


def test_clustered_stage4_requires_only_enabled_method_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safe profile must not demand unsupported encdec scalar/LoRA files."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    run_tag = "clustered-stage4-dag"
    cfg = make_cfg(
        run_tag=run_tag,
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
        conditioning=MappingProxyType(
            {
                "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
                "feature_clusters": {"path": "/clusters.yaml"},
            }
        ),
        enabled_methods=[
            "diffmean_decoder_only",
            "control_encdec",
            "ours_extract_and_add_encdec",
        ],
    )
    concepts = _write_concepts_json(run_tag, n=1)
    _touch_stage_4_generic_deps(run_tag, include_neg_features=False)
    required_direction = _tag_paths(run_tag).diffmean_direction(
        "decoder_only",
        SteeringMode.POSITIVE,
        int(concepts[0]["concept_id"]),
    )

    assert _stage_4_missing_artifacts(cfg) == [required_direction]
    _touch_file(required_direction)
    assert _stage_4_missing_artifacts(cfg) == []
    assert (
        not _tag_paths(run_tag)
        .diffmean_direction(
            "encdec",
            SteeringMode.POSITIVE,
            0,
        )
        .exists()
    )
    assert not _tag_paths(run_tag).lora_adapter(SteeringMode.POSITIVE, 0).exists()


def test_clustered_stage4_does_not_require_disabled_method_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency enumeration must match Stage 4's disabled-method filter."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    run_tag = "clustered-stage4-disabled-dag"
    cfg = make_cfg(
        run_tag=run_tag,
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
        enabled_methods=["diffmean_decoder_only", "control_encdec"],
        disabled_methods=["diffmean_decoder_only"],
    )
    _write_concepts_json(run_tag, n=1)
    _touch_stage_4_generic_deps(run_tag, include_neg_features=False)

    assert _stage_4_missing_artifacts(cfg) == []
    assert (
        not _tag_paths(run_tag)
        .diffmean_direction(
            "decoder_only",
            SteeringMode.POSITIVE,
            0,
        )
        .exists()
    )


def test_clustered_stage4_rejects_enabled_method_outside_closed_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypassing validate_config cannot admit a full-registry-only method."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    run_tag = "clustered-stage4-unsupported-method"
    cfg = make_cfg(
        run_tag=run_tag,
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
        conditioning=MappingProxyType(
            {
                "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
                "feature_clusters": {"path": "/clusters.yaml"},
            }
        ),
        enabled_methods=["diffmean_encdec"],
    )
    _write_concepts_json(run_tag, n=1)
    _touch_stage_4_generic_deps(run_tag, include_neg_features=False)

    with pytest.raises(AxBenchBaseError, match="unsupported methods.*diffmean_encdec"):
        _stage_4_missing_artifacts(cfg)


def test_clustered_stage4_requires_explicit_enabled_methods_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DAG guard remains fail-closed when config validation is bypassed."""

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    stats_path = tmp_path / "stats.safetensors"
    _write_stats_with_n_sae_features(stats_path, n_sae_features=16384)
    run_tag = "clustered-stage4-missing-allowlist"
    cfg = make_cfg(
        run_tag=run_tag,
        run_negative_mode=False,
        feature_statistics_path=str(stats_path),
        conditioning=MappingProxyType(
            {
                "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
                "feature_clusters": {"path": "/clusters.yaml"},
            }
        ),
        enabled_methods=None,
    )
    _write_concepts_json(run_tag, n=1)
    _touch_stage_4_generic_deps(run_tag, include_neg_features=False)

    with pytest.raises(AxBenchBaseError, match="explicit enabled_methods allowlist"):
        _stage_4_missing_artifacts(cfg)
