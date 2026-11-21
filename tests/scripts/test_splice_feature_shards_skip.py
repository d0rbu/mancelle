"""Verified-skip / partial-state / missing-sidecar raise semantics for the splice
orchestrator (issue #590 / PR #597).

STRICT resume contract for an existing ``output_dir`` (skip_existing=True):

(a) shard + parquet + sidecar all present -> the sidecar's recorded params are
    compared to the current run; identical -> skip (no byte change), ANY mismatch
    -> ValueError naming the differing keys (mixed-provenance output_dir).
(b) shard + parquet present, sidecar missing -> raise ValueError immediately.
    There is no basis to verify the existing outputs belong to the current run
    params; delete them or use a fresh output_dir.
(c) exactly one of shard/parquet present -> ValueError (partial == corrupt state).
"""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import COMPRESSED_EXT, save_features
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
    def raw_tokenizer(_model, *, load_chat_template):
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
        save_features(
            make_base_shard(texts, TOK, 2, context_window=2048), feature_dir / f"shard_{idx:05d}"
        )
        atomic_write_parquet(pa.table({"text": texts}), parquet_dir / f"shard_{idx:05d}.parquet")
    return feature_dir, parquet_dir


def _cfg(feature_dir, parquet_dir, output_dir, **overrides):
    base = {
        "feature_dir": str(feature_dir),
        "data_dir": str(parquet_dir),
        "output_dir": str(output_dir),
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


_RUN_PARAMS = sfs.SpliceRunParams(
    model="gemma-2-2b",
    segment_size=2,
    context_window=2048,
    sae_max_context=8192,
    cut_fraction=0.6,
    seed=42,
    write_provenance=True,
    skip_existing=True,
)


def test_verify_provenance_corrupt_json_raises(tmp_path):
    """A sidecar that is not valid JSON is unreadable/corrupt -> refuse to skip."""
    prov = tmp_path / "shard_00000.json"
    prov.write_text("{ this is not valid json ", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable/corrupt"):
        sfs._verify_matching_provenance(prov, 0, "shard_00000", _RUN_PARAMS)


def test_verify_provenance_schema_v1_raises(tmp_path):
    """A schema-v1 sidecar (pre-#596, no would-have-followed records) must raise
    in the skip path — otherwise it is silently skipped and never upgraded to v2,
    and splice_counterfactual training then rejects it (issue #596 finding [0])."""
    prov = tmp_path / "shard_00000.json"
    prov.write_text(
        json.dumps({"schema_version": 1, "params": {}, "shard_seed": 123}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version=1"):
        sfs._verify_matching_provenance(prov, 0, "shard_00000", _RUN_PARAMS)


def test_verify_provenance_missing_params_block_raises(tmp_path):
    """A syntactically valid sidecar with no 'params' dict is corrupt -> raise."""
    prov = tmp_path / "shard_00000.json"
    prov.write_text(json.dumps({"schema_version": 2, "shard_seed": 123}), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'params' block"):
        sfs._verify_matching_provenance(prov, 0, "shard_00000", _RUN_PARAMS)


def test_verify_provenance_params_not_a_dict_raises(tmp_path):
    """'params' present but not a dict is equally corrupt -> the same raise."""
    prov = tmp_path / "shard_00000.json"
    prov.write_text(
        json.dumps({"schema_version": 2, "params": "oops", "shard_seed": 123}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no 'params' block"):
        sfs._verify_matching_provenance(prov, 0, "shard_00000", _RUN_PARAMS)


def _snapshot(out_dir, *, predicate=lambda p: True):
    return {p: p.read_bytes() for p in out_dir.rglob("*") if p.is_file() and predicate(p)}


def _not_provenance(p) -> bool:
    return sfs.PROVENANCE_DIRNAME not in p.parts


# ---------------------------------------------------------------------------
# (a) all present + params match -> verified skip, zero byte change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zstd_threads,zstd_level", [(0, None), (2, 8), (0, 12)])
def test_verified_skip_match_does_not_rewrite_any_byte(tmp_path, zstd_threads, zstd_level):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    cfg = _cfg(fd, pq_dir, out, zstd_threads=zstd_threads, zstd_level=zstd_level)
    sfs.run(cfg)
    before = _snapshot(out)
    assert before, "first run must write outputs"

    for path in (out / sfs.PROVENANCE_DIRNAME).glob("*.json"):
        params = json.loads(path.read_text())["params"]
        assert params.get("zstd_threads", 0) == zstd_threads
        assert ("zstd_threads" in params) == (zstd_threads != 0)
        assert params.get("zstd_level") == zstd_level
        assert ("zstd_level" in params) == (zstd_level is not None)

    sfs.run(cfg)  # identical params -> verified skip

    after = _snapshot(out)
    assert set(after) == set(before), "verified skip must not add/remove files"
    for path, data in before.items():
        assert after[path] == data, f"verified skip rewrote {path.name}"


@pytest.mark.parametrize("initial,requested", [(0, 2), (2, 0), (2, 1)])
def test_threads_mismatch_rejects_resume_without_rewriting(tmp_path, initial, requested):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out, zstd_threads=initial))
    before = _snapshot(out)
    with pytest.raises(ValueError, match="zstd_threads"):
        sfs.run(_cfg(fd, pq_dir, out, zstd_threads=requested))
    assert _snapshot(out) == before


@pytest.mark.parametrize("invalid", [False, True, 0.0, -1, "0", None])
def test_invalid_recorded_threads_rejects_resume(tmp_path, invalid):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out))
    path = out / sfs.PROVENANCE_DIRNAME / "shard_00000.json"
    provenance = json.loads(path.read_text())
    provenance["params"]["zstd_threads"] = invalid
    path.write_text(json.dumps(provenance))
    before = _snapshot(out)
    with pytest.raises(ValueError, match="provenance zstd_threads"):
        sfs.run(_cfg(fd, pq_dir, out))
    assert _snapshot(out) == before


# ---------------------------------------------------------------------------
# (a) mismatch -> ValueError naming the differing key, zero byte change
# ---------------------------------------------------------------------------


def test_mismatched_params_raise_naming_key(tmp_path):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out, cut_fraction=0.6))
    before = _snapshot(out)

    # Same explicit output_dir, different cut_fraction -> mixed-provenance dir.
    with pytest.raises(ValueError, match="cut_fraction"):
        sfs.run(_cfg(fd, pq_dir, out, cut_fraction=0.3))

    after = _snapshot(out)
    assert set(after) == set(before), "mismatch raise must not add/remove files"
    for path, data in before.items():
        assert after[path] == data, f"mismatch raise rewrote {path.name}"


# ---------------------------------------------------------------------------
# (b) sidecar missing -> raise ValueError, shard bytes untouched
# ---------------------------------------------------------------------------


def test_skip_existing_missing_sidecar_raises(tmp_path):
    """shard+parquet present, sidecar missing -> ValueError with clear message; shard bytes untouched."""
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out))

    shard0 = out / f"shard_00000_features{COMPRESSED_EXT}"
    shard0_bytes = shard0.read_bytes()

    # Delete all sidecars: shard+parquet now exist but sidecar is gone.
    prov_dir = out / sfs.PROVENANCE_DIRNAME
    for p in prov_dir.glob("*.json"):
        p.unlink()

    with pytest.raises(ValueError, match="no provenance sidecar to verify against"):
        sfs.run(_cfg(fd, pq_dir, out))

    # The derived shard bytes must be untouched (no recompute, no rewrite).
    assert shard0.read_bytes() == shard0_bytes, (
        "missing-sidecar raise must not rewrite the derived shard"
    )


# ---------------------------------------------------------------------------
# (c) partial outputs (shard XOR parquet) -> ValueError, no re-run
# ---------------------------------------------------------------------------


def test_partial_outputs_missing_parquet_raise(tmp_path):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out))

    shard0 = out / f"shard_00000_features{COMPRESSED_EXT}"
    shard0_bytes = shard0.read_bytes()
    (out / sfs.PARQUET_DIRNAME / "shard_00000.parquet").unlink()  # shard w/o parquet

    with pytest.raises(ValueError, match="partial"):
        sfs.run(_cfg(fd, pq_dir, out))

    assert shard0.read_bytes() == shard0_bytes, "partial-state raise must not touch the shard"


def test_partial_outputs_missing_shard_raise(tmp_path):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out))

    parquet0 = out / sfs.PARQUET_DIRNAME / "shard_00000.parquet"
    parquet0_bytes = parquet0.read_bytes()
    (out / f"shard_00000_features{COMPRESSED_EXT}").unlink()  # parquet w/o shard

    with pytest.raises(ValueError, match="partial"):
        sfs.run(_cfg(fd, pq_dir, out))

    assert parquet0.read_bytes() == parquet0_bytes, "partial-state raise must not touch the parquet"


@pytest.mark.parametrize("initial,requested", [(None, 8), (8, None), (8, 3), (None, 12), (12, None)])
def test_level_mismatch_rejects_resume_without_rewriting(tmp_path, initial, requested):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    sfs.run(_cfg(fd, pq_dir, out, zstd_level=initial))
    before = _snapshot(out)
    with pytest.raises(ValueError, match="zstd_level"):
        sfs.run(_cfg(fd, pq_dir, out, zstd_level=requested))
    assert _snapshot(out) == before


@pytest.mark.parametrize("recorded", [None, False, True, 0, 23, 8.0, "8"])
def test_recorded_level_validation_and_null_legacy_resume(tmp_path, recorded):
    fd, pq_dir = _make_base_dir(tmp_path)
    out = tmp_path / "out"
    cfg = _cfg(fd, pq_dir, out)
    sfs.run(cfg)
    path = out / sfs.PROVENANCE_DIRNAME / "shard_00000.json"
    provenance = json.loads(path.read_text())
    provenance["params"]["zstd_level"] = recorded
    path.write_text(json.dumps(provenance))
    before = _snapshot(out)
    if recorded is None:
        sfs.run(cfg)
    else:
        with pytest.raises(ValueError, match="provenance zstd_level"):
            sfs.run(cfg)
    assert _snapshot(out) == before
