"""Configuration, scheduling, publication, and resume contracts for pairing mode."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import steering.feature_io as feature_io
from scripts.benchmark import benchmark_splice_feature_shards as benchmark
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import (
    COMPRESSED_EXT,
    glob_features,
    load_features,
    read_shard_metadata,
    save_features,
)
from steering.remote_training import VERIFIED_UPLOAD_CLAIMS_DIRNAME, RemoteTrainingError
from steering.stats_io import save_statistics
from tests.scripts._splice_artifact_compare import assert_splice_dirs_equal
from tests.scripts.test_splice_feature_shards_remote import FakeBucket, _download_output_tree
from tests.steering._splice_pairing_fixtures import (
    PAIRING_TOKENIZER,
    PairingCorpus,
    build_pairing_corpus,
    pairing_run_config,
    snapshot_files,
)


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return PAIRING_TOKENIZER

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", lambda _model: 8192)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)


@pytest.fixture
def corpus(tmp_path: Path) -> PairingCorpus:
    return build_pairing_corpus(tmp_path)


def _remote_config(corpus: PairingCorpus, output: Path):
    return pairing_run_config(
        corpus,
        output,
        remote={
            "enabled": True,
            "bucket_id": "owner/pairing-tests",
            "local_features_root": str(corpus.root / "features"),
            "remote_features_root": "features",
            "staging_dir": str(corpus.root / "staging"),
            "expected_shard_count": 3,
            "retry_timeout_seconds": 1.0,
            "retry_interval_seconds": 0.001,
        },
    )


def test_default_and_explicit_random_have_identical_artifacts_without_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = build_pairing_corpus(tmp_path, statistics=False)
    original = snapshot_files(source.feature_dir)

    def unexpected_statistics(*_args, **_kwargs):
        pytest.fail("random pairing must not load statistics")

    monkeypatch.setattr(sfs, "load_statistics", unexpected_statistics)
    default_cfg = pairing_run_config(source, tmp_path / "default", pairing_mode="random")
    del default_cfg.pairing_mode
    del default_cfg.pairing_candidate_pool_size
    sfs.run(default_cfg)
    sfs.run(
        pairing_run_config(
            source,
            tmp_path / "explicit",
            pairing_mode="random",
            pairing_candidate_pool_size="inactive-and-ignored",
        )
    )
    assert_splice_dirs_equal(tmp_path / "default", tmp_path / "explicit")
    assert snapshot_files(source.feature_dir) == original
    for sidecar in (tmp_path / "explicit" / sfs.PROVENANCE_DIRNAME).glob("*.json"):
        params = json.loads(sidecar.read_text())["params"]
        assert not any(key.startswith("pairing_") for key in params)


@pytest.mark.parametrize("mode", ["random", "opposite_percentile"])
@pytest.mark.parametrize("profiled", [False, True])
def test_local_runner_keeps_legacy_feature_serialization(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch, mode: str, profiled: bool
) -> None:
    def unexpected_canonicalization(*_args, **_kwargs):
        pytest.fail("local splice jobs must retain the default feature serializer")

    monkeypatch.setattr(feature_io, "deterministic_safetensors_bytes", unexpected_canonicalization)
    output = corpus.root / "local"
    sfs.run(pairing_run_config(
        corpus, output, pairing_mode=mode, profile=profiled,
        profile_summary_output=str(corpus.root / "profile.json"),
    ))
    assert len(glob_features(output)) == 3


@pytest.mark.parametrize("mode", [None, "opposite", 1, True, ["random"]])
def test_pairing_mode_rejects_invalid_values_before_outputs(
    corpus: PairingCorpus, mode: object
) -> None:
    output = corpus.root / "invalid"
    with pytest.raises((ValueError, TypeError), match="pairing_mode"):
        sfs.run(pairing_run_config(corpus, output, pairing_mode=mode))
    assert not output.exists()


@pytest.mark.parametrize("pool", [0, 1, -2, True, 8.0, "8", None])
def test_active_pool_requires_an_integer_at_least_two_before_outputs(
    corpus: PairingCorpus, pool: object
) -> None:
    output = corpus.root / "invalid"
    with pytest.raises((ValueError, TypeError), match="pairing_candidate_pool_size"):
        sfs.run(pairing_run_config(corpus, output, pairing_candidate_pool_size=pool))
    assert not output.exists()


def test_opposite_requires_local_statistics_before_outputs(tmp_path: Path) -> None:
    source = build_pairing_corpus(tmp_path, statistics=False)
    output = tmp_path / "missing_stats"
    with pytest.raises((ValueError, FileNotFoundError), match="statistic"):
        sfs.run(pairing_run_config(source, output))
    assert not output.exists()


@pytest.mark.parametrize(
    "field,value",
    [("segment_size", 2), ("context_window", 16), ("layer", 9), ("pooling_mode", "mean_pool")],
)
def test_statistics_metadata_mismatch_fails_before_publication(
    corpus: PairingCorpus, field: str, value: object
) -> None:
    assert corpus.statistics is not None
    changed = replace(
        corpus.statistics,
        metadata=replace(corpus.statistics.metadata, **{field: value}),
    )
    save_statistics(changed, corpus.feature_dir / "feature_statistics.safetensors")
    output = corpus.root / "mismatched_stats"
    with pytest.raises(ValueError, match="statistic"):
        sfs.run(pairing_run_config(corpus, output))
    assert not output.exists()


@pytest.mark.parametrize("statistics_width", [8, 32768])
def test_canonical_feature_width_mismatch_fails_before_publication(
    corpus: PairingCorpus, statistics_width: int
) -> None:
    assert corpus.statistics is not None
    canonical = (
        corpus.root / "s4" / "l8" / "a_res" / "p_max_pool" / "f16k" / "m_gemma-2-2b" / "base"
    )
    canonical.parent.mkdir(parents=True)
    corpus.feature_dir.rename(canonical)
    source = replace(corpus, feature_dir=canonical)
    changed_stats = replace(
        corpus.statistics,
        metadata=replace(corpus.statistics.metadata, n_sae_features=statistics_width),
    )
    save_statistics(changed_stats, canonical / "feature_statistics.safetensors")
    output = corpus.root / "width_mismatch"
    with pytest.raises(ValueError, match="n_sae_features|width"):
        sfs.run(pairing_run_config(source, output))
    assert not output.exists()


@pytest.mark.parametrize("width", [None, 8, 16], ids=["legacy-absent", "matching", "wrong"])
@pytest.mark.parametrize("shard_index", [0, 1], ids=["bootstrap", "later"])
@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_pairing_statistics_check_each_present_source_width(
    corpus: PairingCorpus,
    monkeypatch: pytest.MonkeyPatch,
    width: int | None,
    shard_index: int,
    remote: bool,
) -> None:
    # Explicit-model, noncanonical paths still carry authoritative width headers.
    # IDs 0/1 fit both widths, so ID bounds cannot detect this metadata mismatch.
    stem = f"shard_{shard_index:05d}"
    source_path = corpus.feature_dir / f"{stem}_features{COMPRESSED_EXT}"
    assert "n_sae_features" not in read_shard_metadata(source_path)
    if width is not None:
        data = load_features(source_path)
        data["n_sae_features"] = width
        save_features(data, corpus.feature_dir / stem)
    source_before = snapshot_files(corpus.feature_dir)
    output = corpus.root / "features" / "derived"
    bucket = FakeBucket(corpus.root / "bucket")
    if remote:
        for path in glob_features(corpus.feature_dir):
            bucket.put(path, f"features/base/{path.name}")
        monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
        cfg = _remote_config(corpus, output)
    else:
        cfg = pairing_run_config(corpus, output)

    if width == 16:
        with pytest.raises((ValueError, RuntimeError), match="n_sae_features|failed to splice"):
            sfs.run(cfg)
        assert not (output / f"{stem}_features{COMPRESSED_EXT}").exists()
        assert not any(stem in item.path for item in bucket.list("features/derived"))
    else:
        sfs.run(cfg)
        if remote:
            _download_output_tree(bucket, "features/derived", output)
        assert len(glob_features(output)) == 3
    assert snapshot_files(corpus.feature_dir) == source_before


def test_random_does_not_validate_source_width_against_unused_statistics(
    corpus: PairingCorpus,
) -> None:
    path = corpus.feature_dir / f"shard_00000_features{COMPRESSED_EXT}"
    data = load_features(path)
    data["n_sae_features"] = 16  # The unused statistics have width 8.
    save_features(data, corpus.feature_dir / "shard_00000")
    output = corpus.root / "random_width"
    sfs.run(pairing_run_config(corpus, output, pairing_mode="random"))
    assert len(glob_features(output)) == 3


@pytest.mark.parametrize(
    "execution",
    [{}, {"shard_parallelism": 2}, {"async_feature_save_depth": 2}],
    ids=["serial", "spawn", "async"],
)
def test_later_shard_must_match_the_pairing_statistics(
    corpus: PairingCorpus, execution: dict
) -> None:
    later_shard = corpus.feature_dir / f"shard_00001_features{COMPRESSED_EXT}"
    data = load_features(later_shard)
    data["layer"] = 9
    save_features(data, corpus.feature_dir / "shard_00001")
    output = corpus.root / "later_metadata_mismatch"
    with pytest.raises((ValueError, RuntimeError), match="statistic|failed to splice"):
        sfs.run(pairing_run_config(corpus, output, **execution))
    assert not (output / f"shard_00001_features{COMPRESSED_EXT}").exists()


def test_opposite_default_path_separates_mode_and_pool(corpus: PairingCorpus) -> None:
    sfs.run(pairing_run_config(corpus, None))
    sfs.run(pairing_run_config(corpus, None, pairing_candidate_pool_size=4))
    sfs.run(pairing_run_config(corpus, None, pairing_mode="random"))
    parent = corpus.feature_dir / "splice" / "cut_frac1.0"
    for leaf in ("seed42", "seed42_opposite_percentile_pool8", "seed42_opposite_percentile_pool4"):
        assert len(glob_features(parent / leaf)) == 3


@pytest.mark.parametrize("execution", ["spawn", "async", "striped"])
def test_opposite_execution_modes_have_identical_complete_artifacts(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch, execution: str
) -> None:
    source_before = snapshot_files(corpus.feature_dir)
    parquets_before = snapshot_files(corpus.data_dir)
    serial = corpus.root / "serial"
    candidate = corpus.root / execution
    sfs.run(pairing_run_config(corpus, serial))
    if execution == "spawn":
        sfs.run(pairing_run_config(corpus, candidate, shard_parallelism=2))
    elif execution == "async":
        sfs.run(pairing_run_config(corpus, candidate, async_feature_save_depth=2))
    else:
        monkeypatch.setenv("WORLD_SIZE", "2")
        for rank in (0, 1):
            monkeypatch.setenv("RANK", str(rank))
            sfs.run(pairing_run_config(corpus, candidate))
    assert_splice_dirs_equal(serial, candidate)
    assert snapshot_files(corpus.feature_dir) == source_before
    assert snapshot_files(corpus.data_dir) == parquets_before


def test_opposite_profile_and_resume_preserve_all_completed_bytes(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_statistics = sfs.load_statistics
    reads = []

    def counted_statistics(path):
        reads.append(Path(path))
        return read_statistics(path)

    monkeypatch.setattr(sfs, "load_statistics", counted_statistics)
    output = corpus.root / "profiled"
    profile = corpus.root / "profile.json"
    cfg = pairing_run_config(corpus, output, profile=True, profile_summary_output=str(profile))
    sfs.run(cfg)
    assert reads == [corpus.feature_dir / "feature_statistics.safetensors"]
    before = snapshot_files(output)
    first_profile = json.loads(profile.read_text())
    assert first_profile["totals"]["detail_seconds"]["recombine.opposite_pairing"] > 0
    assert first_profile["config"]["pairing_mode"] == "opposite_percentile"
    assert first_profile["config"]["pairing_candidate_pool_size"] == 8
    sfs.run(cfg)
    assert len(reads) == 2  # One snapshot per run, including the all-skip verification pass.
    assert snapshot_files(output) == before
    assert json.loads(profile.read_text())["totals"]["n_skipped"] == 3


@pytest.mark.parametrize(
    "original_mode,overrides,match",
    [
        ("random", {"pairing_mode": "opposite_percentile"}, "pairing_mode"),
        ("opposite_percentile", {"pairing_mode": "random"}, "pairing_mode"),
        ("opposite_percentile", {"pairing_candidate_pool_size": 4}, "pairing_candidate_pool_size"),
    ],
)
def test_resume_rejects_pairing_mismatch_without_mutation(
    corpus: PairingCorpus, original_mode: str, overrides: dict, match: str
) -> None:
    output = corpus.root / "resume"
    sfs.run(pairing_run_config(corpus, output, pairing_mode=original_mode))
    before = snapshot_files(output)
    with pytest.raises(ValueError, match=match):
        sfs.run(pairing_run_config(corpus, output, **overrides))
    assert snapshot_files(output) == before


@pytest.mark.parametrize(
    "key,wrong",
    [
        ("pairing_algorithm", "future_algorithm"),
        ("pairing_percentile_checksum", "0" * 64),
        ("pairing_mode", None),
        ("pairing_candidate_pool_size", None),
    ],
)
def test_resume_rejects_incompatible_or_incomplete_pairing_sidecars(
    corpus: PairingCorpus, key: str, wrong: object
) -> None:
    output = corpus.root / "corrupt_resume"
    cfg = pairing_run_config(corpus, output)
    sfs.run(cfg)
    sidecar = output / sfs.PROVENANCE_DIRNAME / "shard_00000.json"
    payload = json.loads(sidecar.read_text())
    if wrong is None:
        del payload["params"][key]
    else:
        payload["params"][key] = wrong
    sidecar.write_text(json.dumps(payload))
    before = snapshot_files(output)
    with pytest.raises(ValueError, match=key):
        sfs.run(cfg)
    assert snapshot_files(output) == before


def test_random_resume_rejects_orphaned_active_pairing_fields(corpus: PairingCorpus) -> None:
    output = corpus.root / "orphaned_pairing"
    cfg = pairing_run_config(corpus, output, pairing_mode="random")
    sfs.run(cfg)
    sidecar = output / sfs.PROVENANCE_DIRNAME / "shard_00000.json"
    payload = json.loads(sidecar.read_text())
    payload["params"]["pairing_algorithm"] = "bounded_swap_v1"
    sidecar.write_text(json.dumps(payload))
    before = snapshot_files(output)
    with pytest.raises(ValueError, match="pairing"):
        sfs.run(cfg)
    assert snapshot_files(output) == before


def test_changed_percentile_table_rejects_resume(corpus: PairingCorpus) -> None:
    output = corpus.root / "changed_stats"
    cfg = pairing_run_config(corpus, output)
    sfs.run(cfg)
    before = snapshot_files(output)
    assert corpus.statistics is not None
    changed_stats = replace(
        corpus.statistics,
        per_feature_stats={
            key: replace(stat, percentiles={p: value * 2 for p, value in stat.percentiles.items()})
            for key, stat in corpus.statistics.per_feature_stats.items()
        },
    )
    save_statistics(changed_stats, corpus.feature_dir / "feature_statistics.safetensors")
    with pytest.raises(ValueError, match="pairing_percentile_checksum"):
        sfs.run(cfg)
    assert snapshot_files(output) == before


def test_benchmark_mirrors_statistics_and_runs_pairing_from_isolated_inputs(
    corpus: PairingCorpus,
) -> None:
    source_before = snapshot_files(corpus.feature_dir)
    mirrored_features, mirrored_parquets, stems = benchmark._prepare_inputs(
        source_feature_dir=corpus.feature_dir,
        source_data_dir=corpus.data_dir,
        run_root=corpus.root / "benchmark",
        max_shards=2,
    )
    assert stems == ["shard_00000", "shard_00001"]
    assert (mirrored_features / "feature_statistics.safetensors").read_bytes() == (
        corpus.feature_dir / "feature_statistics.safetensors"
    ).read_bytes()
    output = corpus.root / "benchmark" / "output"
    cfg = pairing_run_config(corpus, output)
    cfg.feature_dir = str(mirrored_features)
    cfg.data_dir = str(mirrored_parquets)
    sfs.run(cfg)
    assert len(glob_features(output)) == 2
    assert snapshot_files(corpus.feature_dir) == source_before


def test_remote_later_shard_metadata_mismatch_never_publishes_bad_triplet(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    later = corpus.feature_dir / f"shard_00001_features{COMPRESSED_EXT}"
    data = load_features(later)
    data["layer"] = 9
    save_features(data, corpus.feature_dir / "shard_00001")
    bucket = FakeBucket(corpus.root / "bucket")
    for path in glob_features(corpus.feature_dir):
        bucket.put(path, f"features/base/{path.name}")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _remote_config(corpus, corpus.root / "features" / "derived")
    with pytest.raises(RuntimeError, match="layer|statistic|failed"):
        sfs.run(cfg)
    assert not any("shard_00001" in item.path for item in bucket.list("features/derived"))


@pytest.mark.parametrize("striped", [False, True])
def test_remote_pairing_matches_local_and_resumes_without_publication(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch, striped: bool
) -> None:
    output = corpus.root / "features" / "derived"
    sfs.run(pairing_run_config(corpus, output))
    bucket = FakeBucket(corpus.root / "bucket")
    for path in glob_features(corpus.feature_dir):
        bucket.put(path, f"features/base/{path.name}")
    original_source = {item.path: item.xet_hash for item in bucket.list("features/base")}
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _remote_config(corpus, output)
    # Remote mode must read controls locally but feature bodies from its sealed bucket.
    for path in glob_features(corpus.feature_dir):
        path.unlink()
    ranks = (0, 1) if striped else (0,)
    monkeypatch.setenv("WORLD_SIZE", str(len(ranks)))
    for rank in ranks:
        monkeypatch.setenv("RANK", str(rank))
        sfs.run(cfg)
    downloaded = corpus.root / "downloaded"
    _download_output_tree(bucket, "features/derived", downloaded)
    assert_splice_dirs_equal(output, downloaded)
    assert {item.path: item.xet_hash for item in bucket.list("features/base")} == original_source
    bucket.upload_order.clear()
    for rank in ranks:
        monkeypatch.setenv("RANK", str(rank))
        sfs.run(cfg)
    assert bucket.upload_order == []
    assert not list((corpus.root / "staging").rglob(f"*{COMPRESSED_EXT}"))
    # Completed remote artifacts must also refuse a new pairing contract.
    cfg.pairing_candidate_pool_size = 4
    with pytest.raises(ValueError, match="pairing_candidate_pool_size"):
        sfs.run(cfg)
    assert bucket.upload_order == []


@pytest.mark.parametrize("mode", ["random", "opposite_percentile"])
@pytest.mark.parametrize("phase", ["parquet", "provenance", "feature"])
@pytest.mark.parametrize("profiled", [False, True], ids=["unprofiled", "profiled"])
def test_full_remote_run_resumes_after_each_canonical_upload_interruption(
    corpus: PairingCorpus,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    phase: str,
    profiled: bool,
) -> None:
    output = corpus.root / "features" / "derived"
    sfs.run(pairing_run_config(corpus, output, pairing_mode=mode))
    source_before = snapshot_files(corpus.feature_dir)
    suffixes = {
        "parquet": "_parquets/shard_00000.parquet",
        "provenance": "_splice_provenance/shard_00000.json",
        "feature": f"shard_00000_features{COMPRESSED_EXT}",
    }

    class InterruptedBucket(FakeBucket):
        interrupted = True

        def upload(self, local_path: Path, remote_path: str) -> None:
            if self.interrupted and remote_path.endswith(suffixes[phase]):
                raise OSError(f"interruption after {phase} claim, before canonical body")
            super().upload(local_path, remote_path)

    bucket = InterruptedBucket(corpus.root / "bucket")
    for path in glob_features(corpus.feature_dir):
        bucket.put(path, f"features/base/{path.name}")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _remote_config(corpus, output)
    cfg.pairing_mode = mode
    cfg.profile = profiled
    cfg.profile_summary_output = str(corpus.root / "remote-profile.json")
    cfg.remote.retry_timeout_seconds = 0.02
    cfg.remote.retry_interval_seconds = 0.001

    with pytest.raises(RemoteTrainingError, match="timed out"):
        sfs.run(cfg)

    staged_output = corpus.root / "staging" / "rank00000" / "output"
    original_triplet = {
        suffix: (staged_output / suffix).read_bytes() for suffix in suffixes.values()
    }
    original_claims = {
        item.path: item
        for item in bucket.list("features/derived")
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME in Path(item.path).parts
    }
    failed_name = Path(suffixes[phase]).name
    assert any(Path(path).parent.name == failed_name for path in original_claims)
    canonical = {
        item.path.removeprefix("features/derived/")
        for item in bucket.list("features/derived")
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME not in Path(item.path).parts
    }
    assert canonical == set(list(suffixes.values())[:list(suffixes).index(phase)])

    # Exercise a complete run restart: startup removes old staged bodies and
    # recomputes them. Retrying only _commit_remote_job would miss this regression.
    bucket.interrupted = False
    sfs.run(cfg)
    for suffix, original_bytes in original_triplet.items():
        assert bucket._path(f"features/derived/{suffix}").read_bytes() == original_bytes
    for path, original_identity in original_claims.items():
        assert bucket.stat(path) == original_identity
    final_claims = [
        item for item in bucket.list("features/derived")
        if VERIFIED_UPLOAD_CLAIMS_DIRNAME in Path(item.path).parts
    ]
    assert len(final_claims) == 9  # One immutable claim per object, no retry variants.
    downloaded = corpus.root / "downloaded"
    _download_output_tree(bucket, "features/derived", downloaded)
    assert_splice_dirs_equal(output, downloaded)
    assert snapshot_files(corpus.feature_dir) == source_before
    assert not list((corpus.root / "staging").rglob(f"*{COMPRESSED_EXT}"))
    if profiled:
        profile = json.loads(Path(cfg.profile_summary_output).read_text())
        assert profile["totals"]["committed"] == 3
        assert profile["totals"]["n_failed"] == 0

    bucket.upload_order.clear()
    sfs.run(cfg)
    assert bucket.upload_order == []
    cfg.seed += 1
    with pytest.raises(ValueError, match="seed"):
        sfs.run(cfg)
    assert bucket.upload_order == []


@pytest.mark.parametrize("mode", ["random", "opposite_percentile"])
def test_completed_legacy_remote_triplets_skip_without_reserialization(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    output = corpus.root / "features" / "derived"
    sfs.run(pairing_run_config(corpus, output, pairing_mode=mode))
    bucket = FakeBucket(corpus.root / "bucket")
    for path in glob_features(corpus.feature_dir):
        bucket.put(path, f"features/base/{path.name}")
    # Publish the ordinary local writer's legacy bytes, including their claims.
    for path in sorted(output.rglob("*")):
        if path.is_file():
            sfs.upload_verified_object(
                bucket, local_path=path,
                remote_path=f"features/derived/{path.relative_to(output).as_posix()}",
                timeout_seconds=1.0, retry_interval_seconds=0.001,
            )
    before = list(bucket.list("features"))
    bucket.upload_order.clear()

    def unexpected_save(*_args, **_kwargs):
        pytest.fail("completed legacy triplets must be verified and skipped")

    monkeypatch.setattr(sfs, "save_features_presorted", unexpected_save)
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _remote_config(corpus, output)
    cfg.pairing_mode = mode
    sfs.run(cfg)
    assert bucket.upload_order == []
    assert list(bucket.list("features")) == before


@pytest.mark.parametrize("mode", ["random", "opposite_percentile"])
def test_remote_restart_preserves_conflicting_legacy_feature_claims(
    corpus: PairingCorpus, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    feature_name = f"shard_00000_features{COMPRESSED_EXT}"

    class InterruptedBucket(FakeBucket):
        interrupted = True

        def upload(self, local_path: Path, remote_path: str) -> None:
            if self.interrupted and remote_path.endswith(feature_name):
                raise OSError("interrupted legacy feature publication")
            super().upload(local_path, remote_path)

    bucket = InterruptedBucket(corpus.root / "bucket")
    for path in glob_features(corpus.feature_dir):
        bucket.put(path, f"features/base/{path.name}")
    monkeypatch.setattr(sfs, "HfBucketBackend", lambda _bucket_id: bucket)
    cfg = _remote_config(corpus, corpus.root / "features" / "derived")
    cfg.pairing_mode = mode
    cfg.remote.retry_timeout_seconds = 0.02
    cfg.remote.retry_interval_seconds = 0.001
    real_save = sfs.save_features_presorted

    def legacy_save(data, path, **kwargs):
        kwargs["deterministic"] = False
        return real_save(data, path, **kwargs)

    # Represent a pre-fix run with an outstanding claim for noncanonical bytes.
    with monkeypatch.context() as legacy:
        legacy.setattr(sfs, "save_features_presorted", legacy_save)
        with pytest.raises(RemoteTrainingError, match="timed out"):
            sfs.run(cfg)
    claim_prefix = f"features/derived/{VERIFIED_UPLOAD_CLAIMS_DIRNAME}/{feature_name}"
    original_claims = list(bucket.list(claim_prefix))
    assert len(original_claims) == 1

    bucket.interrupted = False
    with pytest.raises(RemoteTrainingError, match="collision"):
        sfs.run(cfg)
    assert bucket.stat(original_claims[0].path) == original_claims[0]
    assert len(bucket.list(claim_prefix)) == 2
    assert bucket.stat(f"features/derived/{feature_name}") is None
