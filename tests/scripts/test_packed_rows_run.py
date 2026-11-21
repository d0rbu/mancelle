"""run()-level integration coverage for precompute_packed_rows (issue #531 H-X1 / F-A1).

Every other packed-rows test drives the inner ``precompute_packed_rows`` directly; the public
``run(cfg)`` orchestration — and the #531 glue that lives ONLY there — had no test. This drives
the real ``run()`` end-to-end on the synthetic env and pins:

  * O6: the base-shard content hash is computed EXACTLY ONCE and the SAME value is threaded into
    BOTH the train and val manifests (no second ~1 TB re-hash, no divergent provenance hash);
  * F-A1: the cross-shard global-max reduction lands non-zero ``max_n_seg`` / ``max_seg_size`` in
    the manifest metadata (the encoder-decoder static-padding shapes);
  * compress_workers auto/explicit resolution + the profile sidecar (manifest_pre_pass, totals,
    per-output-shard) are written and self-consistent.

``run()`` computes ``tok_checksum`` from ``get_tokenizer()`` and loads ``pct_checksum`` from a
statistics file — neither exists in the inner-function fixture. Rather than synthesise a stats
file + re-key overlays, we monkeypatch those PRE-EXISTING, separately-tested helpers (NOT under
#531 review) so ``run()`` reaches the actual #531 glue with the env's checksums.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import scripts.preprocess.precompute_packed_rows as ppr
from errors import PackedRowInvariantError
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.overlay_io import glob_overlay_shards
from tests.encdec_test_helpers import build_threshold_pipeline
from tests.scripts._packed_rows_fixtures import (
    N_SAE,
    SEGMENT_SIZE,
    SimpleTokenizer,
    _make_precompute_env,
)


def _wire_run(monkeypatch, env):
    """Point run()'s tokenizer + checksum lookups at the synthetic env (pre-existing helpers,
    not #531 code), and return spy state capturing the base-hash + manifest calls."""
    monkeypatch.setattr(ppr, "get_tokenizer", lambda: SimpleTokenizer())
    # verify_segment_size cross-checks an s{N} path component (the directory convention) that the
    # inner-function fixture's flat `features/` dir lacks; it is a pre-existing sanity check, not
    # #531 code, so no-op it to reach the glue under test.
    monkeypatch.setattr(ppr, "verify_segment_size", lambda *a, **k: None)
    monkeypatch.setattr(
        ppr, "find_statistics_path", lambda d: Path(d) / "feature_statistics.safetensors"
    )
    monkeypatch.setattr(ppr, "load_percentile_checksum", lambda _p: env["pct_checksum"])
    monkeypatch.setattr(ppr, "compute_tokenizer_checksum", lambda _tok: env["tok_checksum"])

    # Both values are lists (homogeneous) so len() counts calls without an int|list union.
    spy: dict[str, list] = {"hash_calls": [], "manifests": []}
    real_hash = ppr._compute_base_hash
    real_wm = ppr.write_manifest

    def spy_hash(*a, **k):
        spy["hash_calls"].append(None)
        return real_hash(*a, **k)

    def spy_wm(overlay_dir_path, *a, **k):
        spy["manifests"].append((Path(overlay_dir_path), k.get("precomputed_base_hash")))
        return real_wm(overlay_dir_path, *a, **k)

    monkeypatch.setattr(ppr, "_compute_base_hash", spy_hash)
    monkeypatch.setattr(ppr, "write_manifest", spy_wm)
    return spy


def _cfg(env, sidecar: Path) -> DictConfig:
    cfg = OmegaConf.create(
        {
            "feature_dir": env["feature_dir"],
            "sae_width": "16k",
            "data_dir": env["data_dir"],
            "segment_size": SEGMENT_SIZE,
            "context_len": 32,
            "buffer_size": 20,
            "rows_per_shard": 3,
            "max_shards": None,  # → sall output dir; n_shards parquet → train[:-1] / val[-1:]
            "zstd_level": 12,
            "resume": False,
            "resume_from_shard": None,
            "compress_workers": 4,
            "profile": True,
            "profile_summary_output": str(sidecar),
        }
    )
    assert isinstance(cfg, DictConfig)
    return cfg


def _recursive_digest_map(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            payload = f"symlink:{os.readlink(path)}".encode()
        elif path.is_file():
            payload = path.read_bytes()
        else:
            continue
        result[relative] = hashlib.sha256(payload).hexdigest()
    return result


def _packed_output_dirs(env: dict, context_len: int = 32) -> tuple[Path, Path]:
    parent = (
        Path(env["feature_dir"])
        / "pct"
        / env["pct_checksum"][:16]
        / "tok"
        / env["tok_checksum"][:16]
        / f"c{context_len}"
    )
    return parent / "sall", parent / "val_sall"


def _fused_loader_shards(packed_dir: Path) -> set[str]:
    boundaries = (
        t.tensor([0.25, 0.5, 0.75], dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE, -1)
        .contiguous()
    )
    loader = precomputed_data_loader(
        packed_dir,
        B=1,
        feature_pipeline=build_threshold_pipeline(boundaries),
        device="cpu",
    )
    seen = set()
    while True:
        state = next(loader)[-1]
        if state["epoch"] != 1:
            return seen
        seen.add(state["shard"])


def test_run_threads_single_base_hash_into_both_manifests(tmp_path, monkeypatch):
    """O6 + F-A1: run() hashes the base shards ONCE, threads the SAME hash into both manifests,
    and writes non-zero static-padding dims. Teeth: a regression dropping val's
    ``precomputed_base_hash`` records None for val (caught); computing the hash per-manifest makes
    hash_calls == 2 (caught)."""
    env = _make_precompute_env(tmp_path, n_shards=3)
    spy = _wire_run(monkeypatch, env)
    sidecar = tmp_path / "profile.json"

    ret = ppr.run(_cfg(env, sidecar))
    assert ret == 0

    # O6: exactly one base-hash computation, two manifests (train + val), same hash threaded in.
    n_hash = len(spy["hash_calls"])
    assert n_hash == 1, f"base hash computed {n_hash}x, expected once (O6)"
    assert len(spy["manifests"]) == 2, "run() must write a train AND a val manifest"
    (train_dir, train_arg), (val_dir, val_arg) = spy["manifests"]
    assert train_arg is not None and train_arg == val_arg, (
        "both manifests must receive the SAME precomputed base hash (O6 single-hash reuse)"
    )

    train = json.loads((train_dir / "manifest.json").read_text())
    val = json.loads((val_dir / "manifest.json").read_text())
    for m in (train, val):
        assert m["schema_version"] == 2
        assert m["kind"] == "packed_rows"
        assert m["metadata"]["type"] == "packed_rows"
    # The single computed hash actually landed in both manifests (O6 byte-identical provenance).
    assert train["base_shard_content_hash"] == val["base_shard_content_hash"] == train_arg
    # F-A1: the reduced global maxima are written non-zero (static-padding shapes).
    assert train["metadata"]["max_n_seg"] > 0 and train["metadata"]["max_seg_size"] > 0


def test_run_writes_consistent_profile_sidecar(tmp_path, monkeypatch):
    """The run()-level profile path (config, manifest_pre_pass, totals, per_output_shard) was
    unexercised. Pin that the sidecar is written and self-consistent, the RESOLVED compress_workers
    is recorded, and the base-hash pre-pass hashed every base shard."""
    env = _make_precompute_env(tmp_path, n_shards=3)
    _wire_run(monkeypatch, env)
    sidecar = tmp_path / "profile.json"

    assert ppr.run(_cfg(env, sidecar)) == 0
    assert sidecar.is_file(), "profile=true must write the JSON sidecar"
    prof = json.loads(sidecar.read_text())

    assert prof["config"]["compress_workers"] == 4, "sidecar records the RESOLVED compress_workers"
    assert prof["totals"]["total_rows"] > 0
    assert len(prof["per_output_shard"]) > 0
    assert prof["totals"]["n_shards_written"] == len(prof["per_output_shard"])
    assert prof["row_build_detail"]["row_to_csr_seconds"] > 0
    assert prof["row_build_detail"]["row_provenance_seconds"] > 0
    assert prof["stage_totals"]["csr_build"] >= sum(
        prof["row_build_detail"][key]
        for key in ("row_to_csr_seconds", "row_provenance_seconds")
    )
    assert prof["startup"]["feature_index_scan_seconds"] > 0
    load_records = prof["feature_index_loads"]
    assert load_records[0]["phase"] == "scan_validation"
    resident = [record for record in load_records if record["phase"] == "resident_load"]
    assert resident
    required_detail = {
        "compressed_read",
        "zstd_decompress",
        "safetensors_deserialize",
        "metadata_parse",
        "tensor_upcast",
        "feature_id_delta_decode",
        "base_postprocess",
        "percentile_overlay",
        "array_validation",
        "token_boundary_overlay",
        "materialize_total",
        "doc_index",
    }
    assert required_detail <= set(resident[0]["detail_seconds"])
    assert all(resident[0]["detail_seconds"][key] >= 0 for key in required_detail)
    assert resident[0]["compressed_bytes"] > 0
    assert resident[0]["decompressed_bytes"] > resident[0]["compressed_bytes"]
    assert prof["compression"]["max_inflight_shards"] > 0
    # manifest_pre_pass: the base-hash pre-pass ran over every base shard (issue #531 F-A3).
    pre = prof["manifest_pre_pass"]
    assert pre["n_shards_hashed"] == env["n_shards"] and pre["n_shards_hashed"] > 0
    assert pre["cache_hit"] is False  # cold cache on first run
    assert isinstance(prof["resume"], dict) and prof["resume"]["mode"] == "none"


@pytest.mark.parametrize("mode", ["resume", "resume_from_shard"])
@pytest.mark.parametrize("field", ["buffer_size", "rows_per_shard"])
def test_run_refuses_resume_packing_identity_drift_before_writes(
    tmp_path, monkeypatch, mode, field
):
    env = _make_precompute_env(tmp_path, n_shards=3)
    _wire_run(monkeypatch, env)
    cfg = _cfg(env, tmp_path / "profile.json")
    cfg.profile = False
    assert ppr.run(cfg) == 0
    before = _recursive_digest_map(Path(env["feature_dir"]))

    cfg[field] = int(cfg[field]) + 1
    cfg.resume = mode == "resume"
    cfg.resume_from_shard = 1 if mode == "resume_from_shard" else None
    with pytest.raises(PackedRowInvariantError, match=field):
        ppr.run(cfg)

    assert _recursive_digest_map(Path(env["feature_dir"])) == before


def test_run_preflights_val_resume_identity_before_train_writes(tmp_path, monkeypatch):
    env = _make_precompute_env(tmp_path, n_shards=3)
    _wire_run(monkeypatch, env)
    cfg = _cfg(env, tmp_path / "profile.json")
    cfg.profile = False
    cfg.buffer_size = 4
    assert ppr.run(cfg) == 0
    train_dir, val_dir = _packed_output_dirs(env)
    assert list(train_dir.glob("*.safetensors.zst"))
    assert list(val_dir.glob("*.safetensors.zst"))

    val_manifest_path = val_dir / "manifest.json"
    val_manifest = json.loads(val_manifest_path.read_text())
    val_manifest["metadata"]["buffer_size"] = int(cfg.buffer_size) + 1
    val_manifest_path.write_text(json.dumps(val_manifest, indent=2) + "\n")
    before = _recursive_digest_map(Path(env["feature_dir"]))
    cfg.resume = True

    with pytest.raises(PackedRowInvariantError, match="buffer_size"):
        ppr.run(cfg)
    assert _recursive_digest_map(Path(env["feature_dir"])) == before


def test_fresh_rerun_refuses_stale_trailing_payloads_without_mutation(
    tmp_path, monkeypatch
):
    many_root = tmp_path / "many"
    few_root = tmp_path / "few"
    many_root.mkdir()
    few_root.mkdir()
    many_env = _make_precompute_env(many_root, n_shards=3)
    few_env = _make_precompute_env(few_root, n_shards=3)
    _wire_run(monkeypatch, many_env)

    many_cfg = _cfg(many_env, many_root / "profile.json")
    many_cfg.profile = False
    many_cfg.buffer_size = 4
    many_cfg.rows_per_shard = 1
    assert ppr.run(many_cfg) == 0

    few_cfg = _cfg(few_env, few_root / "profile.json")
    few_cfg.profile = False
    few_cfg.buffer_size = 4
    few_cfg.rows_per_shard = 100
    assert ppr.run(few_cfg) == 0

    many_train, _many_val = _packed_output_dirs(many_env)
    few_train, _few_val = _packed_output_dirs(few_env)
    many_names = {path.name for path in glob_overlay_shards(many_train)}
    few_names = {path.name for path in glob_overlay_shards(few_train)}
    assert len(few_names) < len(many_names)
    would_be_stale = many_names - few_names
    assert would_be_stale
    # Diagnostic: both discovery and the fused loader consume every physical
    # file, including the trailing names a shorter unsafe rerun would retain.
    assert _fused_loader_shards(many_train) == many_names
    assert would_be_stale <= _fused_loader_shards(many_train)

    before = _recursive_digest_map(Path(many_env["feature_dir"]))
    many_cfg.rows_per_shard = 100
    with pytest.raises(PackedRowInvariantError, match="resume=true|move/delete"):
        ppr.run(many_cfg)
    assert _recursive_digest_map(Path(many_env["feature_dir"])) == before


def test_fresh_run_preflights_val_payload_before_creating_train_shards(
    tmp_path, monkeypatch
):
    env = _make_precompute_env(tmp_path, n_shards=3)
    _wire_run(monkeypatch, env)
    cfg = _cfg(env, tmp_path / "profile.json")
    cfg.profile = False
    cfg.buffer_size = 4
    train_dir, val_dir = _packed_output_dirs(env)
    val_dir.mkdir(parents=True)
    (val_dir / "shard_00000.safetensors.zst").write_bytes(b"stale val payload")
    before = _recursive_digest_map(Path(env["feature_dir"]))

    with pytest.raises(PackedRowInvariantError, match="resume=true|move/delete"):
        ppr.run(cfg)
    assert _recursive_digest_map(Path(env["feature_dir"])) == before
    assert not train_dir.exists()
