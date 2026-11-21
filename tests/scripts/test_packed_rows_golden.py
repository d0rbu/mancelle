"""Byte-identity golden anchors for precompute_packed_rows.

Regenerates packed-row shards from a fixed, *seeded* synthetic env and asserts the raw
``.zst`` bytes equal the committed golden. ``save_overlay_shard`` writes EMPTY safetensors
metadata, so the compressed bytes are reproducible (unlike ``save_features``). The committed
golden includes the corrected end-of-input buffer drain from issue #548; a pinned pre-fix
producer separately proves that every previously complete shard stays byte-identical.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.preprocess.precompute_packed_rows as candidate_ppr
import steering.overlay_io as overlay_io
from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from steering.conditional.feature_index import FeatureIndex
from tests.scripts._packed_rows_fixtures import (
    GOLDEN_SEED,
    SimpleTokenizer,
    _make_precompute_env,
    _run_precompute,
    _shard_bytes,
)

GOLDEN = Path(__file__).parent / "golden" / "packed_rows"
# Re-pinned 2026-07-22: the previous pin (a0b29e68, the audit PR2 base) loaded a
# producer that predated main's safetensors/IO refactor, so exec'ing it against
# today's shared modules raised instead of running. The pin must track a commit
# whose producer is source-compatible with the CURRENT shared IO surface; when a
# future refactor changes that surface, re-pin here and re-capture the digests
# below in the same commit.
REFERENCE_COMMIT = "eb12d2f8fdec0ab4f7fec91a2cf62e90466bb841"
REFERENCE_PRODUCER_PATH = "scripts/preprocess/precompute_packed_rows.py"


def _load_reference_producer() -> ModuleType:
    """Load the pinned reference producer directly from git object storage."""
    result = subprocess.run(
        ["git", "show", f"{REFERENCE_COMMIT}:{REFERENCE_PRODUCER_PATH}"],
        check=True,
        capture_output=True,
        text=True,
    )
    module = ModuleType("_packed_rows_reference")
    module.__file__ = str(Path.cwd() / REFERENCE_PRODUCER_PATH)
    sys.modules[module.__name__] = module
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    return module


def _clone_precompute_env(source: Path, destination: Path, env: dict) -> dict:
    shutil.copytree(source, destination)
    return {
        **env,
        "data_dir": str(destination / "data"),
        "feature_dir": str(destination / "features"),
        "parquet_paths": [
            str(destination / "data" / Path(path).name)
            for path in env["parquet_paths"]
        ],
    }


def _run_full_producer(
    module: ModuleType, env: dict, *, compress_workers: int, monkeypatch
) -> None:
    monkeypatch.setattr(module, "get_tokenizer", lambda: SimpleTokenizer())
    monkeypatch.setattr(module, "verify_segment_size", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "find_statistics_path",
        lambda directory: Path(directory) / "stats.stub",
    )
    monkeypatch.setattr(
        module, "load_percentile_checksum", lambda _path: env["pct_checksum"]
    )
    monkeypatch.setattr(
        module, "compute_tokenizer_checksum", lambda _tokenizer: env["tok_checksum"]
    )
    monkeypatch.setattr(
        module,
        "_compute_base_hash",
        lambda *_args, **_kwargs: SimpleNamespace(
            hexdigest="ab" * 32,
            bytes_read=0,
            cache_hit=True,
        ),
    )
    cfg = OmegaConf.create(
        {
            "feature_dir": env["feature_dir"],
            "data_dir": env["data_dir"],
            # Required by the producer since the re-pin; matches the config
            # default and the other packed-row test fixtures.
            "sae_width": "16k",
            "segment_size": 8,
            "context_len": 32,
            "buffer_size": 4,
            "rows_per_shard": 3,
            "max_shards": None,
            "zstd_level": 12,
            "resume": False,
            "resume_from_shard": None,
            "compress_workers": compress_workers,
            "profile": False,
            "profile_summary_output": None,
        }
    )
    assert module.run(cfg) == 0


def _producer_artifacts(env: dict) -> dict[str, tuple[str, bytes | str]]:
    feature_dir = Path(env["feature_dir"])
    pct = env["pct_checksum"][:16]
    tok = env["tok_checksum"][:16]
    roots = (
        feature_dir / "pct" / pct / "tok" / tok / "c32" / "sall",
        feature_dir / "pct" / pct / "tok" / tok / "c32" / "val_sall",
        feature_dir / "tok" / tok / "pct" / pct / "c32" / "sall",
        feature_dir / "tok" / tok / "pct" / pct / "c32" / "val_sall",
    )
    artifacts: dict[str, tuple[str, bytes | str]] = {}
    for root in roots:
        assert root.is_dir(), f"producer output missing: {root}"
        for path in sorted(root.rglob("*")):
            # Dot-prefixed files are stat-validated caches (e.g.
            # .packed-row-fingerprint-cache.json, .base_shard_hash_cache.json).
            # They embed size/mtime/inode, so they differ between two runs of the
            # SAME code by design — they are not product output and must stay out
            # of the byte-identity comparison.
            if path.name.startswith("."):
                continue
            relative = path.relative_to(feature_dir).as_posix()
            if path.is_symlink():
                artifacts[relative] = ("symlink", os.readlink(path))
            elif path.is_file():
                artifacts[relative] = ("file", path.read_bytes())
    return artifacts


def _frozen_digest_map(
    artifacts: dict[str, tuple[str, bytes | str]],
) -> dict[str, str]:
    """Compact anchor for provenance, dims, manifests, and mirror targets."""
    result: dict[str, str] = {}
    for path, (kind, payload) in artifacts.items():
        if kind == "symlink":
            result[path] = f"symlink:{payload}"
        elif "/provenance/" in path or "/_dims/" in path or path.endswith(
            "/manifest.json"
        ):
            assert isinstance(payload, bytes)
            result[path] = hashlib.sha256(payload).hexdigest()
    return result


# Captured from REFERENCE_COMMIT with created_at frozen below. The existing
# committed golden remains the packed-shard anchor; this compact map adds the
# provenance, _dims, manifest, and mirror artifacts produced by run().
FROZEN_RUN_ARTIFACTS: dict[str, str] = {
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/_dims/shard_00000.json": "89dde79907710e22af67714b7c7fe6bc0158797398c4f50e93b28c199886d254",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/_dims/shard_00001.json": "89dde79907710e22af67714b7c7fe6bc0158797398c4f50e93b28c199886d254",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/_dims/shard_00002.json": "4cc9cdb4e0f78bbc3d85af3c72b36b6669eb73081c714d8703d98095cc2f6cba",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/_dims/shard_00003.json": "89dde79907710e22af67714b7c7fe6bc0158797398c4f50e93b28c199886d254",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/_dims/shard_00004.json": "4cc9cdb4e0f78bbc3d85af3c72b36b6669eb73081c714d8703d98095cc2f6cba",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/manifest.json": "cdbaa4980624392e830ce6a29199cb330c4d9c4a9e36ffd0ec47043cace4bed8",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/provenance/shard_00000.safetensors.zst": "ef0d4ffa0fd5ea8bd88c1c11ac597675d2088b58786cfc77b1fbbcd7fe275d4a",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/provenance/shard_00001.safetensors.zst": "7fccb73bd3b73ad3a72833af84b9dafa3e8e9b1b76b09e5f68f15bde02a093b3",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/provenance/shard_00002.safetensors.zst": "c1dd7e061c4e7e7c91ef94604895fa08642fa97c7de826ce85ca70a20c5ff03f",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/provenance/shard_00003.safetensors.zst": "07f462240109c3d3633db8afb516bbeebee05519fc98b4688ea747cf5739db57",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/provenance/shard_00004.safetensors.zst": "79a80948358394d0798f38eb2570e92605c366e43264bc8c1bd1fc686f5f272e",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/_dims/shard_00000.json": "89dde79907710e22af67714b7c7fe6bc0158797398c4f50e93b28c199886d254",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/_dims/shard_00001.json": "4cc9cdb4e0f78bbc3d85af3c72b36b6669eb73081c714d8703d98095cc2f6cba",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/manifest.json": "62ebc99904b7345e12c10cb2e093437744daf935522081eb720efd789ba95581",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/provenance/shard_00000.safetensors.zst": "e3cd1ae8288fb01f2305b4b9422f635669826a1f37bc3a0b34b3786cdbcd379d",
    "pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/provenance/shard_00001.safetensors.zst": "bba859e8a0c88da99b939a18f7f9c1f9b59df175ccd0ca66fab734b9c5f5caf0",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/sall/shard_00000.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/shard_00000.safetensors.zst",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/sall/shard_00001.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/shard_00001.safetensors.zst",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/sall/shard_00002.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/shard_00002.safetensors.zst",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/sall/shard_00003.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/shard_00003.safetensors.zst",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/sall/shard_00004.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/sall/shard_00004.safetensors.zst",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/val_sall/shard_00000.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/shard_00000.safetensors.zst",
    "tok/tok_precompute_t/pct/pct_precompute_t/c32/val_sall/shard_00001.safetensors.zst": "symlink:../../../../../../pct/pct_precompute_t/tok/tok_precompute_t/c32/val_sall/shard_00001.safetensors.zst",
}


def _regenerate(tmp_path: Path) -> dict[str, bytes]:
    """Rebuild the golden env deterministically and return the packed-row shard bytes."""
    tmp_path.mkdir(parents=True, exist_ok=True)  # _make_precompute_env mkdir()s subdirs non-recursively
    t.manual_seed(GOLDEN_SEED)
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "packed"
    _run_precompute(env, out, rows_per_shard=3)  # context_len=32, buffer_size=20 (deterministic)
    return _shard_bytes(out)


def test_golden_committed():
    files = sorted(GOLDEN.glob("shard_*.safetensors.zst"))
    assert files, (
        "golden anchor not committed — run the capture step (plan Task 3 Step 3) and force-add "
        "past .gitignore"
    )


def test_regeneration_is_reproducible(tmp_path):
    """The seeded env is self-consistent: two regenerations match byte-for-byte (guards
    against hidden nondeterminism beyond the seeded percentile randint)."""
    a = _regenerate(tmp_path / "a")
    b = _regenerate(tmp_path / "b")
    assert a == b


def test_packed_rows_byte_identical_to_golden(tmp_path):
    regenerated = _regenerate(tmp_path)
    golden = {p.name: p.read_bytes() for p in sorted(GOLDEN.glob("shard_*.safetensors.zst"))}
    if not golden:
        pytest.skip("golden anchor not committed yet")
    assert set(regenerated) == set(golden), "shard set differs from golden"
    for name in sorted(golden):
        assert regenerated[name] == golden[name], f"{name} bytes differ from golden"


def test_production_config_byte_identical_to_golden(tmp_path):
    """A5 (#540): the FULL production composition (O1 parallel compress + O2 prefetch + O4
    skip-mags) reproduces the committed golden bytes. The golden's own _regenerate uses the default
    index at compress_workers=1, so this pins the combined production config directly against the
    committed anchor (rather than only transitively via the per-opt equivalence tests)."""
    golden = {p.name: p.read_bytes() for p in sorted(GOLDEN.glob("shard_*.safetensors.zst"))}
    if not golden:
        pytest.skip("golden anchor not committed yet")
    tmp_path.mkdir(parents=True, exist_ok=True)
    t.manual_seed(GOLDEN_SEED)
    env = _make_precompute_env(tmp_path, n_shards=4)
    out = tmp_path / "prod"
    out.mkdir()
    precompute_packed_rows(
        parquet_paths=env["parquet_paths"],
        tokenizer=SimpleTokenizer(),
        feature_index=FeatureIndex(
            env["feature_dir"], prefetch=True, skip_feature_mags=True,
            percentile_checksum=env["pct_checksum"], tokenizer_checksum=env["tok_checksum"],
        ),
        context_len=32, output_dir=out, buffer_size=20, rows_per_shard=3, compress_workers=4,
    )
    regenerated = _shard_bytes(out)
    assert set(regenerated) == set(golden), "shard set differs from golden under production config"
    for name in sorted(golden):
        assert regenerated[name] == golden[name], f"{name} differs from golden under production config"


def test_tail_drain_preserves_complete_reference_shards(
    tmp_path, monkeypatch
):
    """The fix only extends the old partial tail; prior full shards stay exact."""
    monkeypatch.setattr(
        overlay_io.time,
        "strftime",
        lambda *_args, **_kwargs: "2026-07-21T00:00:00Z",
    )
    source = tmp_path / "source"
    source.mkdir()
    t.manual_seed(GOLDEN_SEED)
    source_env = _make_precompute_env(source, n_shards=3)
    reference_module = _load_reference_producer()

    snapshots = {}
    for label, module, workers in (
        ("reference_a", reference_module, 1),
        ("reference_b", reference_module, 1),
        ("candidate_serial", candidate_ppr, 1),
        ("candidate_parallel", candidate_ppr, 4),
    ):
        env = _clone_precompute_env(source, tmp_path / label, source_env)
        _run_full_producer(
            module, env, compress_workers=workers, monkeypatch=monkeypatch
        )
        snapshots[label] = _producer_artifacts(env)

    reference = snapshots["reference_a"]
    assert snapshots["reference_b"] == reference, "reference fixture is unstable"
    candidate = snapshots["candidate_serial"]
    assert snapshots["candidate_parallel"] == candidate

    # This fixture produced four complete train shards plus one partial shard,
    # and two complete val shards, before the old producer dropped its buffered
    # tail. The corrected producer may extend/append at those tails only.
    full_prefix_lengths = {
        "/c32/sall/": 4,
        "/c32/val_sall/": 2,
    }
    checked = 0
    for path, payload in reference.items():
        shard_match = re.search(r"shard_(\d{5})", path)
        if shard_match is None:
            continue
        for split_marker, prefix_length in full_prefix_lengths.items():
            if split_marker in path and int(shard_match.group(1)) < prefix_length:
                assert candidate[path] == payload, path
                checked += 1
                break
    assert checked > 0
    assert len(candidate) > len(reference)
    assert _frozen_digest_map(reference) == FROZEN_RUN_ARTIFACTS
