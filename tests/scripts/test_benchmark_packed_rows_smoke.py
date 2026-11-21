"""Smoke test for the packed-rows benchmark harness (issue #531).

Stubs ``run`` (whose full real-corpus invocation needs a stats file + percentile/token
overlays at run-computed checksums — exercised by the measure-first real-corpus handoff)
to write a canned sidecar, and verifies the harness's own logic: it runs each variant,
reads each sidecar, and emits a Markdown table with the per-stage + save columns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.preprocess.precompute_packed_rows as ppr
from scripts.benchmark import benchmark_packed_rows
from scripts.preprocess._packed_rows_profile import PackedRowsProfile, ShardSaveTiming
from tests.scripts._packed_rows_fixtures import (
    SEGMENT_SIZE,
    SimpleTokenizer,
    _make_precompute_env,
)


def _write_minimal_benchmark_inputs(tmp_path: Path, *, n_shards: int = 4) -> None:
    feature_dir = tmp_path / "features"
    data_dir = tmp_path / "data"
    feature_dir.mkdir()
    data_dir.mkdir()
    for idx in range(n_shards):
        stem = f"shard_{idx:05d}"
        (feature_dir / f"{stem}_features.safetensors.zst").write_bytes(
            f"feature-{idx}".encode()
        )
        (data_dir / f"{stem}.parquet").write_bytes(f"parquet-{idx}".encode())
    for overlay_dir, checksum, kind in (
        (feature_dir / "pct" / "pctinput", "pctinput-checksum", "pct"),
        (
            feature_dir / "tok" / "tokinput" / "seg_boundaries",
            "tokinput-checksum",
            "seg_boundaries",
        ),
    ):
        overlay_dir.mkdir(parents=True)
        (overlay_dir / "manifest.json").write_text(
            json.dumps({"checksum": checksum, "kind": kind}) + "\n"
        )
        for idx in range(n_shards):
            (overlay_dir / f"shard_{idx:05d}.safetensors.zst").write_bytes(
                f"overlay-{idx}".encode()
            )


def _write_fake_packed_outputs(cfg) -> None:
    output = (
        Path(cfg.feature_dir)
        / "pct"
        / "pctoutput"
        / "tok"
        / "tokoutput"
        / f"c{int(cfg.context_len)}"
        / f"s{int(cfg.max_shards)}"
    )
    for split_dir in (output, output.parent / f"val_{output.name}"):
        (split_dir / "provenance").mkdir(parents=True)
        (split_dir / "_dims").mkdir()
        (split_dir / "shard_00000.safetensors.zst").write_bytes(b"packed")
        (split_dir / "provenance" / "shard_00000.safetensors.zst").write_bytes(
            b"provenance"
        )
        (split_dir / "_dims" / "shard_00000.json").write_text(
            '{"max_n_seg": 1, "max_seg_size": 2}\n'
        )
        (split_dir / "manifest.json").write_text(
            json.dumps({"type": "packed_rows", "created_at": "variant-local"}) + "\n"
        )


def _recursive_file_digests(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _write_fake_profile(cfg) -> None:
    profile = PackedRowsProfile()
    profile.base_shard_loads = {"count": 1, "total_seconds": 0.1}
    profile.add_output_shard(
        ShardSaveTiming(
            shard_idx=0,
            n_rows=1,
            n_segments=1,
            n_features=1,
            save_seconds=0.1,
            save_bytes=1,
        )
    )
    profile.totals = {
        "wall_seconds": 0.2,
        "total_rows": 1,
        "rows_per_sec": 5.0,
        "n_shards_written": 1,
    }
    profile.write(Path(cfg.profile_summary_output))


def test_bench_smoke(tmp_path, monkeypatch):
    _write_minimal_benchmark_inputs(tmp_path, n_shards=5)
    seen_run_configs = []
    seen_feature_dirs = []

    def fake_run(cfg) -> int:
        # run() reads these directly (not via .get); assert the harness supplies them so the
        # smoke catches a missing required key (the real run() failed on a missing context_len).
        assert cfg.feature_dir and cfg.segment_size and cfg.context_len
        assert cfg.sae_width == "16k"
        assert "decoder_segment_size" not in cfg
        for required in (
            "buffer_size",
            "rows_per_shard",
            "zstd_level",
            "resume",
            "resume_from_shard",
            "compress_workers",
            "profile",
            "profile_summary_output",
        ):
            assert required in cfg
        seen_run_configs.append(
            (
                int(cfg.max_shards),
                int(cfg.get("compress_workers", 0)),
                int(cfg.get("zstd_level", 0)),
            )
        )
        seen_feature_dirs.append(Path(cfg.feature_dir))
        prof = PackedRowsProfile(config={"compress_workers": int(cfg.get("compress_workers", 0))})
        prof.stage_totals["csr_build"] = 0.5
        prof.row_build_detail = {
            "row_to_csr_seconds": 0.35,
            "row_provenance_seconds": 0.15,
            "build_segments_seconds": 0.2,
            "segment_boundaries_seconds": 0.05,
            "feature_concat_seconds": 0.07,
            "feature_ids_concat_seconds": 0.02,
            "percentiles_concat_seconds": 0.02,
            "feature_counts_seconds": 0.01,
            "row_tensor_finalize_seconds": 0.03,
        }
        prof.stage_totals["seg_to_raw"] = 0.7
        prof.base_shard_loads = {"count": 1, "total_seconds": 1.2}
        prof.add_output_shard(
            ShardSaveTiming(shard_idx=0, n_rows=3, n_segments=9, n_features=40,
                            assemble_seconds=0.1, save_seconds=2.0, save_bytes=999)
        )
        prof.totals = {"wall_seconds": 4.0, "total_rows": 3, "rows_per_sec": 0.75,
                       "n_shards_written": 1}
        prof.write(Path(cfg.profile_summary_output))
        _write_fake_packed_outputs(cfg)
        return 0

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", fake_run)

    out_md = tmp_path / "bench.md"
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "max_shards": 4,
            "report_md": str(out_md),
            "sidecar_dir": str(tmp_path / "sidecars"),
            "run_dir": str(tmp_path / "run"),
            "variants": [
                {
                    "name": "serial",
                    "overrides": {"compress_workers": 1, "zstd_level": 12},
                },
                {
                    "name": "parallel",
                    "overrides": {"compress_workers": 8, "zstd_level": 9},
                },
            ],
        }
    )
    md = benchmark_packed_rows.run_bench(cfg)

    assert out_md.is_file()
    assert "| variant |" in md
    assert "serial" in md and "parallel" in md
    assert "csr_build_s" in md  # per-stage columns present
    assert "row_to_csr_s" in md and "row_provenance_s" in md
    assert "build_segments_s" in md and "feature_concat_s" in md
    assert "feature_ids_concat_s" in md and "feature_counts_s" in md
    assert "packed_save med/p90" in md
    assert "worker med/p90" in md
    table_lines = [line for line in md.splitlines() if line.startswith("|")]
    assert table_lines[0].count("|") == table_lines[1].count("|")
    assert "rss GiB" in md
    assert "| reference |" in md
    assert "| exact |" in md
    # both sidecars were written + read
    assert (tmp_path / "sidecars" / "serial.json").is_file()
    assert (tmp_path / "sidecars" / "parallel.json").is_file()
    assert seen_run_configs == [(4, 1, 12), (4, 8, 9)]
    assert len(set(seen_feature_dirs)) == 2
    assert all(path.is_relative_to(tmp_path / "run") for path in seen_feature_dirs)
    assert all(
        len(list(path.glob("*_features.safetensors.zst"))) == 4
        for path in seen_feature_dirs
    )
    assert all((path / ".base_shard_hash_cache.json").is_file() for path in seen_feature_dirs)
    metadata = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert metadata["selected_stems"] == [f"shard_{idx:05d}" for idx in range(4)]
    assert len(metadata["bounded_base_shard_content_hash"]) == 64
    assert metadata["bounded_base_hash_seconds"] >= 0
    assert len(metadata["input_stat_inventory_sha256"]) == 64
    for feature_dir in seen_feature_dirs:
        pct_manifest = json.loads(
            (feature_dir / "pct" / "pctinput" / "manifest.json").read_text()
        )
        tok_manifest = json.loads(
            (
                feature_dir
                / "tok"
                / "tokinput"
                / "seg_boundaries"
                / "manifest.json"
            ).read_text()
        )
        assert pct_manifest["base_shard_content_hash"] == metadata[
            "bounded_base_shard_content_hash"
        ]
        assert tok_manifest["base_shard_content_hash"] == metadata[
            "bounded_base_shard_content_hash"
        ]
    assert not list((tmp_path / "features").glob("pct/*/tok/*/c*/s*"))
    serial_artifacts = json.loads(
        (tmp_path / "sidecars" / "serial.artifacts.json").read_text()
    )
    parallel_artifacts = json.loads(
        (tmp_path / "sidecars" / "parallel.artifacts.json").read_text()
    )
    assert serial_artifacts == parallel_artifacts


def test_bench_real_producer_isolates_variants_and_requires_exact_parity(
    tmp_path, monkeypatch
):
    """Regression for the old harness's real-producer failure mode.

    Both variants used to target the same source-tree ``s3`` directory.  The
    second run therefore failed no-clobber (or could overwrite production under
    weaker producer semantics).  Exercise the actual producer so a mocked
    sidecar-only test cannot hide that interaction.
    """
    env = _make_precompute_env(tmp_path, n_shards=3)
    source_feature_dir = Path(env["feature_dir"])
    before = _recursive_file_digests(source_feature_dir)

    monkeypatch.setattr(ppr, "get_tokenizer", lambda: SimpleTokenizer())
    monkeypatch.setattr(ppr, "verify_segment_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ppr,
        "find_statistics_path",
        lambda directory: Path(directory) / "feature_statistics.safetensors",
    )
    monkeypatch.setattr(
        ppr, "load_percentile_checksum", lambda _path: env["pct_checksum"]
    )
    monkeypatch.setattr(
        ppr, "compute_tokenizer_checksum", lambda _tokenizer: env["tok_checksum"]
    )

    run_dir = tmp_path / "bench-run"
    cfg = OmegaConf.create(
        {
            "feature_dir": env["feature_dir"],
            "data_dir": env["data_dir"],
            "segment_size": SEGMENT_SIZE,
            "sae_width": "16k",
            "context_len": 32,
            "buffer_size": 20,
            "rows_per_shard": 3,
            "max_shards": 3,
            "zstd_level": 12,
            "run_dir": str(run_dir),
            "variants": [
                {
                    "name": "serial",
                    "overrides": {"compress_workers": 1},
                },
                {
                    "name": "parallel",
                    "overrides": {"compress_workers": 2},
                },
            ],
        }
    )

    markdown = benchmark_packed_rows.run_bench(cfg)

    assert "serial" in markdown and "parallel" in markdown
    assert _recursive_file_digests(source_feature_dir) == before
    assert not list(source_feature_dir.glob("pct/*/tok/*/c32/s3"))
    reference = json.loads(
        (run_dir / "sidecars" / "serial.artifacts.json").read_text()
    )
    optimized = json.loads(
        (run_dir / "sidecars" / "parallel.artifacts.json").read_text()
    )
    assert reference
    assert reference == optimized
    assert (run_dir / "input_inventory.json").is_file()
    reference_profile = json.loads(
        (run_dir / "sidecars" / "serial.json").read_text()
    )
    optimized_profile = json.loads(
        (run_dir / "sidecars" / "parallel.json").read_text()
    )
    assert reference_profile["manifest_pre_pass"]["cache_hit"] is True
    assert optimized_profile["manifest_pre_pass"]["cache_hit"] is True
    assert "preserve_percentile_uint8" not in reference_profile["config"]
    assert "preserve_percentile_uint8" not in optimized_profile["config"]


def test_bench_fails_closed_on_artifact_divergence(tmp_path, monkeypatch):
    _write_minimal_benchmark_inputs(tmp_path)
    run_dir = tmp_path / "run"
    report = tmp_path / "report.md"

    def divergent_run(cfg) -> int:
        _write_fake_profile(cfg)
        _write_fake_packed_outputs(cfg)
        if int(cfg.compress_workers) == 2:
            output = next(
                Path(cfg.feature_dir).glob(
                    f"pct/*/tok/*/c{int(cfg.context_len)}/s{int(cfg.max_shards)}"
                )
            )
            (output / "shard_00000.safetensors.zst").write_bytes(b"divergent")
        return 0

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", divergent_run)
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "max_shards": 4,
            "run_dir": str(run_dir),
            "report_md": str(report),
            "variants": [
                {"name": "one", "overrides": {"compress_workers": 1}},
                {"name": "two", "overrides": {"compress_workers": 2}},
            ],
        }
    )

    with pytest.raises(AssertionError, match="artifact mismatch.*train/shard_00000"):
        benchmark_packed_rows.run_bench(cfg)
    assert not report.exists(), "a divergent comparison must not publish a report"


def test_bench_fails_closed_if_source_changes_during_run(tmp_path, monkeypatch):
    _write_minimal_benchmark_inputs(tmp_path)
    source_shard = tmp_path / "features" / "shard_00000_features.safetensors.zst"
    report = tmp_path / "report.md"

    def mutating_run(cfg) -> int:
        _write_fake_profile(cfg)
        _write_fake_packed_outputs(cfg)
        source_shard.write_bytes(b"mutated-production-input")
        return 0

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", mutating_run)
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "max_shards": 4,
            "run_dir": str(tmp_path / "run"),
            "report_md": str(report),
        }
    )

    with pytest.raises(AssertionError, match="source input inventory changed"):
        benchmark_packed_rows.run_bench(cfg)
    assert not report.exists()


def test_bench_requires_max_shards(tmp_path, monkeypatch):
    """Guard (issue #531 G-A1): the bench runs run() against the REAL feature tree; an unbounded
    run (max_shards absent) targets sall/ — the production training input — and would overwrite it
    with resume=false. run_bench must refuse without a max_shards bound, BEFORE invoking run().
    Teeth: run_packed is stubbed to fail loudly so that, if the guard were removed, the test would
    surface that wrong error instead of the expected ValueError."""
    def must_not_run(cfg):  # pragma: no cover - only reached if the guard is removed
        raise AssertionError("run() must not be invoked when max_shards is unbounded")

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", must_not_run)
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            # no max_shards → would resolve to the production sall/ dir
            "sidecar_dir": str(tmp_path / "sidecars"),
        }
    )
    with pytest.raises(ValueError, match="max_shards is required"):
        benchmark_packed_rows.run_bench(cfg)


def test_bench_requires_sae_width_before_running(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark_packed_rows,
        "run_packed",
        lambda _cfg: pytest.fail("run_packed must not run without sae_width"),
    )
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "max_shards": 4,
        }
    )

    with pytest.raises(ValueError, match="sae_width is required"):
        benchmark_packed_rows.run_bench(cfg)


def test_bench_rejects_variant_max_shards_bypass_before_any_mutation(
    tmp_path, monkeypatch
):
    """A variant cannot replace a safe cap with null and reach the production ``sall`` leaf."""
    production_shard = (
        tmp_path
        / "features"
        / "pct"
        / "production"
        / "tok"
        / "production"
        / "c1024"
        / "sall"
        / "shard_00000.safetensors.zst"
    )
    production_shard.parent.mkdir(parents=True)
    production_shard.write_bytes(b"production-packed-row-bytes")
    before_sha256 = hashlib.sha256(production_shard.read_bytes()).hexdigest()
    sidecar_dir = tmp_path / "sidecars"
    report_md = tmp_path / "bench.md"
    called = False

    def destructive_run(cfg):
        nonlocal called
        called = True
        production_shard.write_bytes(b"overwritten")
        production_shard.unlink()
        raise AssertionError("run_packed must not be called for an unsafe variant")

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", destructive_run)
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "context_len": 1024,
            "max_shards": 4,
            "sidecar_dir": str(sidecar_dir),
            "report_md": str(report_md),
            "variants": [
                {"name": "safe", "overrides": {"compress_workers": 1}},
                {"name": "unsafe", "overrides": {"max_shards": None}},
            ],
        }
    )

    with pytest.raises(ValueError, match="max_shards"):
        benchmark_packed_rows.run_bench(cfg)

    assert not called
    assert production_shard.is_file()
    assert hashlib.sha256(production_shard.read_bytes()).hexdigest() == before_sha256
    assert not sidecar_dir.exists()
    assert not report_md.exists()


@pytest.mark.parametrize("max_shards", [0, -1])
def test_bench_requires_positive_resolved_max_shards(
    tmp_path, monkeypatch, max_shards
):
    called = False

    def must_not_run(cfg):
        nonlocal called
        called = True
        raise AssertionError("run_packed must not be called without a positive shard cap")

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", must_not_run)
    sidecar_dir = tmp_path / "sidecars"
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "max_shards": max_shards,
            "sidecar_dir": str(sidecar_dir),
        }
    )

    with pytest.raises(ValueError, match="max_shards.*positive"):
        benchmark_packed_rows.run_bench(cfg)

    assert not called
    assert not sidecar_dir.exists()


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape", "nested/name", "", ".hidden"],
)
def test_bench_rejects_unsafe_variant_names_before_mutation(
    tmp_path, monkeypatch, unsafe_name
):
    called = False

    def must_not_run(_cfg):
        nonlocal called
        called = True
        raise AssertionError("unsafe variant must not run")

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", must_not_run)
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "max_shards": 4,
            "run_dir": str(tmp_path / "run"),
            "variants": [{"name": unsafe_name, "overrides": {}}],
        }
    )

    with pytest.raises(ValueError, match="variant name"):
        benchmark_packed_rows.run_bench(cfg)
    assert not called
    assert not (tmp_path / "run").exists()


def test_bench_rejects_run_dir_overlapping_source_before_mutation(tmp_path, monkeypatch):
    _write_minimal_benchmark_inputs(tmp_path)
    source = tmp_path / "features"
    before = _recursive_file_digests(source)
    monkeypatch.setattr(
        benchmark_packed_rows,
        "run_packed",
        lambda _cfg: pytest.fail("overlapping benchmark must not run"),
    )
    cfg = OmegaConf.create(
        {
            "feature_dir": str(source),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "max_shards": 4,
            "run_dir": str(source / "benchmark-output"),
        }
    )

    with pytest.raises(ValueError, match="run_dir.*overlaps source"):
        benchmark_packed_rows.run_bench(cfg)
    assert _recursive_file_digests(source) == before
    assert not (source / "benchmark-output").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_dir", "other/features"),
        ("data_dir", "other/data"),
        ("segment_size", 64),
        ("sae_width", "65k"),
        ("context_len", 2048),
        ("decoder_segment_size", 128),
        ("buffer_size", 8),
        ("rows_per_shard", 1),
        ("resume", True),
        ("resume_from_shard", 0),
        ("profile", False),
        ("profile_summary_output", "other/profile.json"),
    ],
)
def test_bench_rejects_variant_identity_and_output_overrides_before_mutation(
    tmp_path, monkeypatch, field, value
):
    called = False

    def must_not_run(cfg):
        nonlocal called
        called = True
        raise AssertionError("run_packed must not be called for an unsafe variant")

    monkeypatch.setattr(benchmark_packed_rows, "run_packed", must_not_run)
    sidecar_dir = tmp_path / "sidecars"
    cfg = OmegaConf.create(
        {
            "feature_dir": str(tmp_path / "features"),
            "data_dir": str(tmp_path / "data"),
            "segment_size": 128,
            "sae_width": "16k",
            "context_len": 1024,
            "max_shards": 4,
            "sidecar_dir": str(sidecar_dir),
            "report_md": str(tmp_path / "bench.md"),
            "variants": [
                {"name": "unsafe", "overrides": {field: value}},
            ],
        }
    )

    with pytest.raises(ValueError, match=field):
        benchmark_packed_rows.run_bench(cfg)

    assert not called
    assert not sidecar_dir.exists()
