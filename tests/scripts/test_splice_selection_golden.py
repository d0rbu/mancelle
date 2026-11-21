"""Byte-identity golden for splice_counterfactual preselection (#740).

Pins the SELECTION SHARD bytes (the payload PR-2/PR-3 caching must never change;
save_overlay_shard writes empty metadata ⇒ reproducible across independent builds)
and every manifest field EXCEPT two content-hash fields that hash upstream inputs
which are byte-nondeterministic across builds (save_features HashMap metadata
order + absolute tmp paths in the hashed sidecar — reference_save_features_
nondeterminism). Those two fields are selection INPUTS computed before any caching,
so the cache change cannot alter them; full manifest byte-identity between the
reference and candidate cache paths is asserted by the same-build parity test
(test_splice_preselection_parity.py). See plan addendum 2026-07-19b.
"""

import json
from pathlib import Path

import pytest

from tests.steering._splice_cf_cache_fixtures import build_interleaved_env, run_selection_main

GOLDEN = Path(__file__).parent / "golden" / "splice_selection"
_VOLATILE = ("derived_feature_fingerprint", "sidecar_fingerprint")
_SENTINEL = "<volatile-input-fingerprint>"


def _shards(files: dict[str, bytes]) -> dict[str, bytes]:
    return {n: b for n, b in files.items() if n.endswith(".safetensors.zst")}


def _mask_manifest(raw: bytes) -> dict:
    obj = json.loads(raw)
    sc = obj["metadata"]["splice_counterfactual"]
    for k in _VOLATILE:
        assert isinstance(sc.get(k), str) and len(sc[k]) == 64 and all(
            c in "0123456789abcdef" for c in sc[k]
        ), f"{k} is not a 64-hex digest: {sc.get(k)!r}"
        sc[k] = _SENTINEL
    return obj


def test_golden_committed() -> None:
    shards = sorted(GOLDEN.glob("shard_*.safetensors.zst"))
    assert shards, "shard golden not committed — run the --capture step and force-add"
    assert (GOLDEN / "manifest.masked.json").is_file(), "masked-manifest golden missing"


def test_regeneration_shards_reproducible(tmp_path: Path, monkeypatch) -> None:
    a = _shards(run_selection_main(build_interleaved_env(tmp_path / "a"), monkeypatch=monkeypatch))
    b = _shards(run_selection_main(build_interleaved_env(tmp_path / "b"), monkeypatch=monkeypatch))
    assert set(a) == set(b) and a, "shard set differs / empty across regenerations"
    for name in a:
        assert a[name] == b[name], f"{name} shard bytes nondeterministic across builds"


def test_selection_shards_match_golden(tmp_path: Path, monkeypatch) -> None:
    golden = {p.name: p.read_bytes() for p in sorted(GOLDEN.glob("shard_*.safetensors.zst"))}
    if not golden:
        pytest.skip("shard golden not committed yet")
    produced = _shards(run_selection_main(build_interleaved_env(tmp_path), monkeypatch=monkeypatch))
    assert set(produced) == set(golden), "shard set differs from golden"
    for name, golden_bytes in golden.items():
        assert produced[name] == golden_bytes, f"{name} bytes differ from golden"


def test_manifest_stable_fields_match_golden(tmp_path: Path, monkeypatch) -> None:
    gpath = GOLDEN / "manifest.masked.json"
    if not gpath.is_file():
        pytest.skip("masked-manifest golden not committed yet")
    files = run_selection_main(build_interleaved_env(tmp_path), monkeypatch=monkeypatch)
    produced = _mask_manifest(files["manifest.json"])
    golden = json.loads(gpath.read_text())
    # created_at is frozen by run_selection_main; the two volatile fingerprints
    # are masked to the sentinel; everything else must match byte-for-byte.
    assert json.dumps(produced, sort_keys=True) == json.dumps(golden, sort_keys=True), (
        "manifest stable fields differ from golden"
    )


def test_capacity_variants_match_golden(tmp_path: Path, monkeypatch) -> None:
    """issue #740: retain-block cache capacity must never change output bytes.

    Compares ONLY the selection shard bytes (via ``_shards``) against the
    committed shard golden, at every capacity — this is the core PR-2
    correctness claim. The manifest is deliberately excluded from this
    comparison: manifest identity ACROSS capacities on the SAME build is
    covered by the Task 2.4 same-build parity suite
    (test_splice_preselection_parity.py); comparing it here against the
    golden would spuriously fail on the two upstream-input fingerprint
    fields that are byte-nondeterministic across independent
    ``build_interleaved_env`` builds (see module docstring / plan addendum
    2026-07-19b), which has nothing to do with the cache capacity.
    """
    golden = {p.name: p.read_bytes() for p in sorted(GOLDEN.glob("shard_*.safetensors.zst"))}
    if not golden:
        pytest.skip("shard golden not committed yet")
    for cap in (1, 2, 3, 5):
        env = build_interleaved_env(tmp_path / f"cap{cap}")
        produced = _shards(run_selection_main(
            env, cache_shards=cap, sidecar_shards=cap, monkeypatch=monkeypatch
        ))
        assert set(produced) == set(golden), f"cap {cap}: file set differs"
        for name, golden_bytes in golden.items():
            assert produced[name] == golden_bytes, f"cap {cap}: {name} bytes differ"
