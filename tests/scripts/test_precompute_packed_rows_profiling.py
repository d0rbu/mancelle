"""Profiling is populated correctly and does not change output bytes (issue #531)."""
from pathlib import Path

import torch as t

import scripts.preprocess.precompute_packed_rows as packed_rows_module
from scripts.preprocess._packed_rows_profile import PackedRowsProfile
from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from tests.scripts._packed_rows_fixtures import (
    GOLDEN_SEED,
    SimpleTokenizer,
    _make_feature_index,
    _make_precompute_env,
    _shard_bytes,
)


def _run(env, out: Path, profile=None):
    out.mkdir(parents=True, exist_ok=True)
    return precompute_packed_rows(
        parquet_paths=env["parquet_paths"],
        tokenizer=SimpleTokenizer(),
        feature_index=_make_feature_index(env),
        context_len=32,
        output_dir=out,
        buffer_size=20,
        rows_per_shard=3,
        profile=profile,
    )


def test_profile_does_not_change_bytes(tmp_path):
    t.manual_seed(GOLDEN_SEED)
    env = _make_precompute_env(tmp_path, n_shards=4)
    a = tmp_path / "noprof"
    _run(env, a, profile=None)
    b = tmp_path / "prof"
    _run(env, b, profile=PackedRowsProfile())
    assert _shard_bytes(a) == _shard_bytes(b)


def test_profile_populated(tmp_path):
    t.manual_seed(GOLDEN_SEED)
    env = _make_precompute_env(tmp_path, n_shards=4)
    prof = PackedRowsProfile()
    out = tmp_path / "p"
    total_rows, *_ = _run(env, out, profile=prof)
    n_shards = len(list(out.glob("shard_*.safetensors.zst")))
    assert n_shards > 0
    assert len(prof.per_output_shard) == n_shards
    assert prof.totals["total_rows"] == total_rows
    assert total_rows > 0
    # Per-stage attribution (issue #531 E2): every stage runs in this multi-shard / multi-doc env
    # (parquet read + tokenize per row group; lookup + seg_to_raw + truncate_assert_append per doc;
    # pack + csr_build per row), so EACH must have accrued measurable time. An aggregate-only
    # `sum > 0` check would let deleting any single stage's accumulation line slip past as long as
    # one other stage still accrues — this pins every stage individually.
    assert set(prof.stage_totals) == {
        "parquet", "lookup", "tokenize", "seg_to_raw",
        "truncate_assert_append", "pack", "csr_build",
    }
    for stage, secs in prof.stage_totals.items():
        assert secs > 0.0, f"stage {stage!r} accrued no time ({secs}); its attribution is missing"
    for detail in (
        "row_to_csr_seconds",
        "row_provenance_seconds",
        "build_segments_seconds",
        "segment_boundaries_seconds",
        "feature_concat_seconds",
        "feature_ids_concat_seconds",
        "percentiles_concat_seconds",
        "feature_counts_seconds",
        "row_tensor_finalize_seconds",
    ):
        assert prof.row_build_detail[detail] > 0
    classified_row_build = (
        prof.row_build_detail["row_to_csr_seconds"]
        + prof.row_build_detail["row_provenance_seconds"]
    )
    classified_row_to_csr = sum(
        prof.row_build_detail[key]
        for key in (
            "build_segments_seconds",
            "segment_boundaries_seconds",
            "feature_concat_seconds",
            "row_tensor_finalize_seconds",
        )
    )
    assert prof.stage_totals["csr_build"] >= classified_row_build
    assert prof.stage_totals["csr_build"] - classified_row_build < 0.1
    assert prof.row_build_detail["row_to_csr_seconds"] >= classified_row_to_csr
    assert prof.row_build_detail["row_to_csr_seconds"] - classified_row_to_csr < 0.1
    classified_feature_concat = sum(
        prof.row_build_detail[key]
        for key in (
            "feature_ids_concat_seconds",
            "percentiles_concat_seconds",
            "feature_counts_seconds",
        )
    )
    assert prof.row_build_detail["feature_concat_seconds"] >= classified_feature_concat
    assert prof.row_build_detail["feature_concat_seconds"] - classified_feature_concat < 0.1
    # base-shard loads recorded (one per parquet/feature shard touched)
    assert prof.base_shard_loads["count"] >= 1
    # per-output-shard records carry non-trivial counts
    assert all(s["n_rows"] > 0 for s in prof.per_output_shard)
    assert any(s["save_bytes"] > 0 for s in prof.per_output_shard)
    assert all(s["worker_seconds"] >= s["save_seconds"] for s in prof.per_output_shard)
    assert all(s["queue_wait_seconds"] >= 0 for s in prof.per_output_shard)
    assert prof.compression["max_inflight_shards"] == 1
    assert prof.compression["backpressure_seconds"] >= 0


def test_profile_handles_future_completed_before_callback_registration(
    tmp_path, monkeypatch
):
    """A completed Future calls its callback synchronously during registration.

    Pin the counter/semaphore ordering with an inline executor: every shard is already
    complete when ``add_done_callback`` runs, and multiple flushes prove each permit is
    still released.
    """

    class InlineFuture:
        def __init__(self, result):
            self._result = result

        def add_done_callback(self, callback):
            callback(self)

        def result(self):
            return self._result

    class InlineExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 1

        def submit(self, function, /, *args, **kwargs):
            return InlineFuture(function(*args, **kwargs))

        def shutdown(self):
            return None

    monkeypatch.setattr(packed_rows_module, "ThreadPoolExecutor", InlineExecutor)
    env = _make_precompute_env(tmp_path, n_shards=4)
    prof = PackedRowsProfile()

    _run(env, tmp_path / "inline", profile=prof)

    assert len(prof.per_output_shard) > 1
    assert prof.compression["max_inflight_shards"] == 1
    assert all(shard["queue_wait_seconds"] >= 0 for shard in prof.per_output_shard)
