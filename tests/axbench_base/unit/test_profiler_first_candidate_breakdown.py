"""Tests for the new 'first verified candidate timing breakdown' derived
row in profile-mode summary CSV (PR-1)."""

from __future__ import annotations

import csv

import yaml

from steering.eval.axbench_base.profiler import AxBenchProfiler, profile_region

_PHASE_TO_REGION = {
    "model_load": "stage0c.load_feature_extraction_pair",
    "percentile_table": "stage0c.build_percentile_table",
    "checkpoint_validate": "stage0c.try_load_partial_checkpoint",
    "first_shard_load": "stage0c.candidate_index_load_shard",
    "first_qc_call": "stage0c.examples_qc_filter_one",
    "first_verify_call": "stage0c.verify_concept",
}


def test_summary_csv_contains_first_candidate_breakdown(tmp_path):
    """When profile_region events for the canonical setup phases exist,
    the summary CSV gets a derived 'first_verified_candidate_breakdown'
    section with rows for ALL six phases (model_load, percentile_table,
    checkpoint_validate, first_shard_load, first_qc_call, first_verify_call).

    Beyond row count, this test asserts: each phase's ``region`` column
    matches the canonical region name (catches a phase→region mapping
    regression) and each ``elapsed_s`` parses as a non-negative float
    (catches an empty / null / non-numeric regression).
    """
    import time as _time
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()

    profiler = AxBenchProfiler(
        output_dir=profile_dir,
        metadata={},
    )
    with profiler:
        with profile_region("stage.0c", stage="0c"):
            with profile_region("stage0c.load_feature_extraction_pair"):
                _time.sleep(0.005)
            with profile_region("stage0c.load_candidate_concepts"):
                pass
            with profile_region("stage0c.build_percentile_table"):
                _time.sleep(0.005)
            with profile_region("stage0c.try_load_partial_checkpoint"):
                _time.sleep(0.005)
            with profile_region("stage0c.candidate_index_load_shard", shard=0):
                _time.sleep(0.005)
            with profile_region(
                "stage0c.examples_qc_filter_one",
                candidate_index=0,
                feature_id=0,
            ):
                _time.sleep(0.005)
            with profile_region(
                "stage0c.verify_concept",
                concept_id=0,
                feature_id=0,
            ):
                _time.sleep(0.005)

    summary_csv = profile_dir / "summary.csv"
    assert summary_csv.exists()

    with summary_csv.open() as f:
        rows = list(csv.DictReader(f))

    breakdown = [r for r in rows
                 if r.get("section") == "first_verified_candidate_breakdown"]
    assert len(breakdown) == 6, (
        f"expected exactly 6 breakdown rows after firing all canonical regions; "
        f"got {len(breakdown)}: {breakdown}"
    )
    phases = [r["phase"] for r in breakdown]
    assert phases == list(_PHASE_TO_REGION), (
        f"breakdown phase order must match canonical order; got {phases}"
    )
    for row in breakdown:
        expected_region = _PHASE_TO_REGION[row["phase"]]
        assert row["region"] == expected_region, (
            f"phase {row['phase']!r} maps to region {expected_region!r}; "
            f"got {row['region']!r}"
        )
        elapsed = float(row["elapsed_s"])
        assert 0.0 <= elapsed < 5.0, (
            f"phase {row['phase']!r} elapsed_s={elapsed} unreasonable; "
            f"each region only ran for ~5ms"
        )


def test_summary_yaml_contains_first_candidate_breakdown(tmp_path):
    """The summary YAML payload exposes the breakdown rows under
    'first_verified_candidate_breakdown' for downstream tooling that
    consumes the YAML rather than parsing the CSV.

    Asserts: 6 rows always, region maps match canonical, fired phases
    have positive numeric ``elapsed_s``, unfired phases have None.
    """
    import time as _time
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()

    profiler = AxBenchProfiler(output_dir=profile_dir, metadata={})
    with profiler:
        with profile_region("stage0c.build_percentile_table"):
            _time.sleep(0.005)
        with profile_region("stage0c.candidate_index_load_shard", shard=0):
            _time.sleep(0.005)

    summary_yaml = profile_dir / "summary.yaml"
    payload = yaml.safe_load(summary_yaml.read_text(encoding="utf-8"))

    assert "first_verified_candidate_breakdown" in payload
    bd = payload["first_verified_candidate_breakdown"]
    assert isinstance(bd, list)
    assert len(bd) == 6, (
        f"YAML breakdown must always emit all 6 rows; got {len(bd)}: {bd}"
    )
    fired = {"percentile_table", "first_shard_load"}
    for row in bd:
        expected_region = _PHASE_TO_REGION[row["phase"]]
        assert row["region"] == expected_region
        if row["phase"] in fired:
            elapsed = row["elapsed_s"]
            assert isinstance(elapsed, (int, float))
            assert 0.0 < elapsed < 5.0, (
                f"fired phase {row['phase']!r} elapsed_s={elapsed} should be "
                f"~5ms"
            )
        else:
            assert row["elapsed_s"] is None, (
                f"unfired phase {row['phase']!r} should expose elapsed_s=None; "
                f"got {row['elapsed_s']!r}"
            )


def test_profile_region_synchronize_cuda_false_skips_sync(tmp_path, monkeypatch):
    """When synchronize_cuda=False, profile_region must not call
    _synchronize_cuda_if_available."""
    from unittest.mock import MagicMock

    from steering.eval.axbench_base import profiler as profiler_mod

    sync_mock = MagicMock(name="_synchronize_cuda_if_available")
    monkeypatch.setattr(profiler_mod, "_synchronize_cuda_if_available",
                        sync_mock)

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()

    profiler = AxBenchProfiler(
        output_dir=profile_dir,
        metadata={},
        synchronize_cuda=False,
    )
    with profiler:
        with profile_region("test.no_sync"):
            pass

    assert sync_mock.call_count == 0, (
        f"_synchronize_cuda_if_available was invoked {sync_mock.call_count} "
        f"times despite synchronize_cuda=False"
    )

    summary_yaml = profile_dir / "summary.yaml"
    payload = yaml.safe_load(summary_yaml.read_text(encoding="utf-8"))
    region_names = {r["name"] for r in payload["regions"]}
    assert "test.no_sync" in region_names
