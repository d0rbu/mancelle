"""Orchestration validation tests for scripts.preprocess.splice_feature_shards.

These exercise the cut_fraction-independent wiring that raises BEFORE recombination:
metadata resolution and fail-fast aggregation (missing parquet, corrupt shard).
The real tokenizer is monkeypatched out so the wiring is tested without a model
download.

Also covers the restored byte-identity checks: healthy-run output existence, and
2-rank striped run (rank-0 + rank-1 union) producing the same feature tensors as a
serial run over all shards.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import torch as t
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from scripts.preprocess._splice_profile import DETAIL_KEYS, PHASE_KEYS
from steering.feature_io import COMPRESSED_EXT, load_features, save_features
from tests.scripts._splice_artifact_compare import assert_splice_dirs_equal
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)
SHARD_TEXTS = {
    0: ["alpha beta", "gamma delta epsilon", "zeta eta"],
    1: ["one two three", "four five", "six seven eight nine"],
    2: ["red green blue", "cyan magenta", "yellow black white"],
}


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    def raw_tokenizer(model, *, load_chat_template):
        assert load_chat_template is False
        return TOK

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", lambda model: 8192)


def _make_base_dir(tmp_path):
    feature_dir = tmp_path / "base"
    parquet_dir = tmp_path / "parquets"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    for idx, texts in SHARD_TEXTS.items():
        shard = make_base_shard(texts, TOK, 2, context_window=2048)
        save_features(shard, feature_dir / f"shard_{idx:05d}")
        atomic_write_parquet(pa.table({"text": texts}), parquet_dir / f"shard_{idx:05d}.parquet")
    return feature_dir, parquet_dir


def _cfg(feature_dir, parquet_dir, output_dir, **overrides):
    base = {
        "feature_dir": str(feature_dir),
        "data_dir": str(parquet_dir),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "model": "gemma-2-2b",
        "segment_size": 2,
        "context_window": 2048,
        "cut_fraction": 0.6,
        "seed": 42,
        "skip_existing": True,
        "write_provenance": True,
        "shard_parallelism": 1,
        "async_feature_save_depth": 0,
        "profile": False,
        "profile_summary_output": "out/traces/splice_feature_shards_profile.json",
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_segment_size_config_must_match_metadata(tmp_path):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="segment_size"):
        sfs.run(_cfg(feature_dir, parquet_dir, out, segment_size=4))


def test_fail_fast_on_missing_parquet(tmp_path):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    (parquet_dir / "shard_00001.parquet").unlink()  # remove one parquet
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no parquet matching"):
        sfs.run(_cfg(feature_dir, parquet_dir, out))


def test_fail_fast_on_corrupt_shard(tmp_path):
    """Corrupt shard (doc-count mismatch) raises AlignmentError → fail-fast RuntimeError.

    Discriminating contrast:
    - Healthy shards (0 and 2) are processed sequentially BEFORE the raise and
      produce output — their derived feature files must exist.
    - Corrupt shard (1) fails with AlignmentError (captured per-shard) and must
      NOT produce any derived feature file.
    - The run() aggregates failures and raises at the end — the "fail-fast
      aggregation" side of the contract.
    """
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    # Overwrite shard 1's parquet with mismatched docs -> alignment failure -> RuntimeError.
    atomic_write_parquet(
        pa.table({"text": ["totally different", "doc count too"]}),  # 2 docs, base has 3
        parquet_dir / "shard_00001.parquet",
    )
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="failed to splice"):
        sfs.run(_cfg(feature_dir, parquet_dir, out))

    # Healthy shards produce output; corrupt shard must not.
    assert (out / f"shard_00000_features{COMPRESSED_EXT}").exists(), "shard 0 must have succeeded"
    assert not (out / f"shard_00001_features{COMPRESSED_EXT}").exists(), "shard 1 must have failed"
    assert (out / f"shard_00002_features{COMPRESSED_EXT}").exists(), "shard 2 must have succeeded"


def test_profile_writes_json_summary(tmp_path):
    """profile=true writes a parseable JSON summary at profile_summary_output whose
    per-shard status serializes as the StrEnum string "ok"."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    profile_path = tmp_path / "traces" / "prof.json"
    sfs.run(
        _cfg(feature_dir, parquet_dir, out, profile=True, profile_summary_output=str(profile_path))
    )

    assert profile_path.exists(), "profile JSON must be written at profile_summary_output"
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    assert set(data) == {"config", "totals", "per_shard"}
    assert data["config"]["shard_parallelism"] == 1
    assert data["totals"]["n_ok"] == 3
    assert data["totals"]["n_failed"] == 0
    assert data["totals"]["n_skipped"] == 0
    assert set(data["totals"]["phase_seconds"]) == set(PHASE_KEYS)
    assert set(data["totals"]["detail_seconds"]) == set(DETAIL_KEYS)
    assert len(data["per_shard"]) == 3
    # SpliceJobStatus is a StrEnum -> serializes as the plain string "ok"/"failed".
    assert all(r["status"] == "ok" for r in data["per_shard"])
    assert all(isinstance(r["status"], str) for r in data["per_shard"])
    assert all(set(r["phase_seconds"]) == set(PHASE_KEYS) for r in data["per_shard"])
    assert all(set(r["detail_seconds"]) == set(DETAIL_KEYS) for r in data["per_shard"])
    assert all(r["phase_seconds"]["total"] > 0.0 for r in data["per_shard"])
    assert data["totals"]["detail_seconds"]["source.truncate_tokenize"] > 0.0
    assert data["totals"]["detail_seconds"]["save.zstd_compress"] > 0.0


def test_profile_records_failed_shard_incrementally(tmp_path):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    atomic_write_parquet(
        pa.table({"text": ["wrong", "doc count"]}),
        parquet_dir / "shard_00001.parquet",
    )
    out = tmp_path / "out"
    profile_path = tmp_path / "traces" / "prof.json"
    with pytest.raises(RuntimeError, match="failed to splice"):
        sfs.run(
            _cfg(
                feature_dir,
                parquet_dir,
                out,
                profile=True,
                profile_summary_output=str(profile_path),
            )
        )

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    failed = [r for r in data["per_shard"] if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["shard_stem"] == "shard_00001"
    assert failed[0]["error_type"] == "AlignmentError"
    assert failed[0]["phase_seconds"]["total"] > 0.0
    assert set(failed[0]["detail_seconds"]) == set(DETAIL_KEYS)


def test_profile_path_rank_suffixed_under_ddp(tmp_path, monkeypatch):
    """Under WORLD_SIZE>1 the profile JSON is rank-suffixed (.rankNNNNN) so ranks
    don't clobber one another."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    profile_path = tmp_path / "traces" / "prof.json"
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    sfs.run(
        _cfg(feature_dir, parquet_dir, out, profile=True, profile_summary_output=str(profile_path))
    )

    rank_path = profile_path.with_name("prof.rank00001.json")
    assert rank_path.exists(), "rank-1 profile JSON must be rank-suffixed"
    assert not profile_path.exists(), "the un-suffixed path must not be written under DDP"
    data = json.loads(rank_path.read_text(encoding="utf-8"))
    assert data["config"]["rank"] == 1 and data["config"]["world_size"] == 2


def test_unpicklable_tokenizer_guard(tmp_path, monkeypatch):
    """shard_parallelism>1 with an unpicklable tokenizer raises a clear ValueError
    BEFORE spawning any worker (the guard pickles the tokenizer up front)."""
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    tok = CharMergeTokenizer(bos_id=500)
    tok_any: Any = tok
    tok_any._unpicklable = lambda: None  # makes the instance unpicklable
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return tok

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    with pytest.raises(ValueError, match="requires a picklable tokenizer"):
        sfs.run(_cfg(feature_dir, parquet_dir, tmp_path / "out", shard_parallelism=2))


def test_rank_striped_byte_identity(tmp_path, monkeypatch):
    """Rank-0 + rank-1 striped runs produce feature tensors identical to a serial run.

    Striping is POSITIONAL over the shard-index-sorted job list — ``sorted[rank::world_size]``
    (a slice with step = world_size), NOT ``shard_index % world_size``. For the
    contiguous 0,1,2 indices here the two happen to coincide, but the mechanism is the
    positional slice:
    - world_size=2, rank=0 -> positions 0,2 in the sorted list → shard_00000 + shard_00002
    - world_size=2, rank=1 -> position 1 in the sorted list    → shard_00001

    After both striped runs write to the same output_dir, the union of their outputs
    must be t.equal to the serial run's output for every shard and every key tensor.
    """
    feature_dir, parquet_dir = _make_base_dir(tmp_path)

    out_serial = tmp_path / "serial"
    sfs.run(_cfg(feature_dir, parquet_dir, out_serial, shard_parallelism=1))

    out_striped = tmp_path / "striped"

    # Rank-0 pass (shards 0 and 2).
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    sfs.run(_cfg(feature_dir, parquet_dir, out_striped, shard_parallelism=1))

    # Rank-1 pass (shard 1); skip_existing=True is safe since rank-0 didn't write shard 1.
    monkeypatch.setenv("RANK", "1")
    sfs.run(_cfg(feature_dir, parquet_dir, out_striped, shard_parallelism=1))

    monkeypatch.delenv("RANK")
    monkeypatch.delenv("WORLD_SIZE")

    _TENSOR_KEYS = ("feature_ids", "feature_mags", "offsets", "doc_ids", "segment_indices")
    for stem in ("shard_00000", "shard_00001", "shard_00002"):
        serial_path = out_serial / f"{stem}_features{COMPRESSED_EXT}"
        striped_path = out_striped / f"{stem}_features{COMPRESSED_EXT}"
        assert serial_path.exists(), f"serial/{stem} output missing"
        assert striped_path.exists(), f"striped/{stem} output missing"
        s = load_features(serial_path)
        r = load_features(striped_path)
        for key in _TENSOR_KEYS:
            assert t.equal(s[key], r[key]), f"striped vs serial mismatch: {stem}/{key}"
        # Byte-level comparison for derived parquet + provenance sidecar (Fix B6).
        serial_pq = out_serial / sfs.PARQUET_DIRNAME / f"{stem}.parquet"
        striped_pq = out_striped / sfs.PARQUET_DIRNAME / f"{stem}.parquet"
        assert serial_pq.exists(), f"serial/{stem} parquet missing"
        assert striped_pq.exists(), f"striped/{stem} parquet missing"
        assert serial_pq.read_bytes() == striped_pq.read_bytes(), f"parquet byte mismatch: {stem}"
        serial_prov = out_serial / sfs.PROVENANCE_DIRNAME / f"{stem}.json"
        striped_prov = out_striped / sfs.PROVENANCE_DIRNAME / f"{stem}.json"
        assert serial_prov.exists(), f"serial/{stem} provenance missing"
        assert striped_prov.exists(), f"striped/{stem} provenance missing"
        assert serial_prov.read_bytes() == striped_prov.read_bytes(), (
            f"provenance byte mismatch: {stem}"
        )


@pytest.mark.parametrize("zstd_threads,zstd_level", [(0, None), (2, 8)])
def test_profile_and_process_pool_artifact_parity(tmp_path, monkeypatch, zstd_threads, zstd_level):
    real_save = sfs.save_features_presorted
    calls = []

    def recording_save(*args, **kwargs):
        calls.append((kwargs["zstd_threads"], kwargs["zstd_level"]))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(sfs, "save_features_presorted", recording_save)
    feature_dir, parquet_dir = _make_base_dir(tmp_path)

    reference = tmp_path / "reference"
    sfs.run(_cfg(
        feature_dir, parquet_dir, reference, profile=False, shard_parallelism=1,
        zstd_threads=zstd_threads, zstd_level=zstd_level,
    ))

    profiled = tmp_path / "profiled"
    sfs.run(
        _cfg(
            feature_dir,
            parquet_dir,
            profiled,
            profile=True,
            shard_parallelism=1,
            zstd_threads=zstd_threads, zstd_level=zstd_level,
            profile_summary_output=str(tmp_path / "serial_profile.json"),
        )
    )
    assert_splice_dirs_equal(reference, profiled)

    parallel = tmp_path / "parallel"
    sfs.run(
        _cfg(
            feature_dir,
            parquet_dir,
            parallel,
            profile=True,
            shard_parallelism=2,
            zstd_threads=zstd_threads, zstd_level=zstd_level,
            profile_summary_output=str(tmp_path / "parallel_profile.json"),
        )
    )
    assert_splice_dirs_equal(reference, parallel)
    # Spawn workers use their own module; these six calls cover both parent save branches.
    assert calls == [(zstd_threads, zstd_level)] * 6


@pytest.mark.parametrize("zstd_threads,zstd_level", [(0, None), (2, 8)])
def test_async_feature_save_artifact_parity_and_profile(tmp_path, zstd_threads, zstd_level):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)

    reference = tmp_path / "reference"
    sfs.run(_cfg(feature_dir, parquet_dir, reference, profile=False, zstd_threads=zstd_threads, zstd_level=zstd_level))

    async_out = tmp_path / "async"
    profile_path = tmp_path / "async_profile.json"
    sfs.run(
        _cfg(
            feature_dir,
            parquet_dir,
            async_out,
            profile=True,
            async_feature_save_depth=1,
            zstd_threads=zstd_threads, zstd_level=zstd_level,
            profile_summary_output=str(profile_path),
        )
    )

    assert_splice_dirs_equal(reference, async_out)
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    assert data["config"]["async_feature_save_depth"] == 1
    assert data["config"]["zstd_threads"] == zstd_threads
    assert data["config"]["zstd_level"] == zstd_level
    assert data["totals"]["n_ok"] == 3
    assert data["totals"]["n_failed"] == 0
    assert all(r["status"] == "ok" for r in data["per_shard"])
    assert all(r["phase_seconds"]["save_features"] > 0.0 for r in data["per_shard"])
    assert data["totals"]["detail_seconds"]["save.atomic_write"] > 0.0


def test_async_feature_save_failure_is_failed_shard_not_success(tmp_path, monkeypatch):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    profile_path = tmp_path / "async_failure_profile.json"
    real_save = sfs.save_features_presorted

    def fail_first_shard(data, path, **kwargs):
        if Path(path).name == "shard_00000":
            raise RuntimeError("async save boom")
        return real_save(data, path, **kwargs)

    monkeypatch.setattr(sfs, "save_features_presorted", fail_first_shard)

    with pytest.raises(RuntimeError, match="failed to splice"):
        sfs.run(
            _cfg(
                feature_dir,
                parquet_dir,
                out,
                profile=True,
                async_feature_save_depth=1,
                profile_summary_output=str(profile_path),
            )
        )

    assert not (out / f"shard_00000_features{COMPRESSED_EXT}").exists()
    assert (out / sfs.PARQUET_DIRNAME / "shard_00000.parquet").exists()
    assert (out / sfs.PROVENANCE_DIRNAME / "shard_00000.json").exists()

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    failed = [r for r in data["per_shard"] if r["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["shard_stem"] == "shard_00000"
    assert failed[0]["error_type"] == "RuntimeError"
    assert "async save boom" in failed[0]["error_message"]
    assert failed[0]["output_feature_bytes"] == 0


def test_async_feature_save_profile_finishes_at_save_completion(tmp_path, monkeypatch):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    profile_path = tmp_path / "async_profile.json"
    first_save_done = threading.Event()
    artificial_next_shard_seconds = 0.5
    real_save = sfs.save_features_presorted

    def fake_splice_feature_shard(*args, **kwargs):
        shard_name = kwargs["shard_name"]
        if shard_name == "shard_00001":
            assert first_save_done.wait(timeout=5.0), "first async save did not complete"
            deadline = time.perf_counter() + artificial_next_shard_seconds
            while time.perf_counter() < deadline:
                pass
        derived = make_base_shard(["alpha beta"], TOK, 2, context_window=2048)
        provenance = {"docs": [{"n_switches": 0}]}
        return derived, ["alpha beta"], provenance

    def recording_save(data, path, **kwargs):
        result = real_save(data, path, **kwargs)
        if Path(path).name == "shard_00000":
            first_save_done.set()
        return result

    monkeypatch.setattr(sfs, "splice_feature_shard", fake_splice_feature_shard)
    monkeypatch.setattr(sfs, "save_features_presorted", recording_save)

    sfs.run(
        _cfg(
            feature_dir,
            parquet_dir,
            out,
            profile=True,
            async_feature_save_depth=1,
            profile_summary_output=str(profile_path),
        )
    )

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    first = next(r for r in data["per_shard"] if r["shard_stem"] == "shard_00000")
    second = next(r for r in data["per_shard"] if r["shard_stem"] == "shard_00001")
    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert second["phase_seconds"]["total"] >= artificial_next_shard_seconds
    # These fixture write phases are disjoint; their fsync latency must not
    # count as time leaked from the next shard's work.
    own_writes = sum(
        first["phase_seconds"][key]
        for key in ("write_parquet", "write_provenance", "save_features")
    )
    assert first["phase_seconds"]["total"] - own_writes < artificial_next_shard_seconds * 0.75
    assert first["cpu_seconds"] < artificial_next_shard_seconds * 0.75


def test_async_feature_save_rejects_process_pool_before_outputs(tmp_path):
    feature_dir, parquet_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="async_feature_save_depth"):
        sfs.run(
            _cfg(
                feature_dir,
                parquet_dir,
                out,
                shard_parallelism=2,
                async_feature_save_depth=1,
            )
        )
    assert not out.exists(), "no output dir must be created for invalid async/process-pool config"
