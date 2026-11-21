"""Retain-block integration for splice preselection (issue #740).

Covers the ``_ordered_source_stems`` helper directly (dedup + first-need order
+ bound-check error identity), and the per-packed-shard retain block wired
into ``select_splice_counterfactual_features_for_shard`` / ``main`` via the
``selection_source_cache_shards`` / ``selection_sidecar_cache_shards`` config
keys: capacity 1 (both) must be a structural no-op (never touches
``drop_except``), capacity > 1 must invoke it, the two knobs must gate their
own index independently, and both must reject non-positive ints / bools.
Byte-identity across capacities is the golden suite's job
(``tests/scripts/test_splice_selection_golden.py::test_capacity_variants_match_golden``),
not this file's -- except at UNEQUAL capacities (incl. the shipped default
5 != 16), which is this file's job (see the "unequal-capacity correctness"
section below): the retain pre-seed loop must produce a strict PREFIX of the
capacity-1 reference's own access order even when the two indices' capacities
differ, both for byte-identical output and for first-error identity under
multi-artifact corruption.
"""

import json
import logging
import re
import shutil
from pathlib import Path

import pytest

import scripts.preprocess.precompute_selected_features as psf
from steering.conditional.splice_counterfactual import validate_packed_provenance
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    provenance_dir,
)
from steering.overlay_io import glob_overlay_shards, load_overlay_shard
from steering.packed_batch_io import load_packed_row_shard
from tests.scripts.test_splice_selection_golden import _shards
from tests.steering._splice_cf_cache_fixtures import (
    build_interleaved_env,
    run_selection_main,
    worst_case_stem_alternations,
)


def _spy_calls(monkeypatch, cls: type, method_name: str) -> dict[str, int]:
    """Monkeypatch ``cls.method_name`` to count calls while preserving behavior."""
    counts = {"n": 0}
    real = getattr(cls, method_name)

    def spy(self, *args, **kwargs):
        counts["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cls, method_name, spy)
    return counts


# --- _ordered_source_stems (pure helper, no I/O) ----------------------------


def test_ordered_source_stems_dedups_in_first_need_order() -> None:
    packed_info = psf.PackedProvenanceInfo(
        base_shard_stems=["shard_a", "shard_b", "shard_c"],
        packed_shard_paths=[],
        provenance_shard_paths=[],
        packed_stem_fingerprint="fp1",
        packed_row_fingerprint="fp2",
        provenance_fingerprint="fp3",
    )
    # base_shard_idx 2 and 0 each recur non-adjacently; first-need order dedups
    # to distinct stems in order of first appearance, not sorted/index order.
    ordered_group_keys = [(2, 0), (0, 1), (2, 5), (1, 0), (0, 3)]
    stems = psf._ordered_source_stems(ordered_group_keys, packed_info, "shard_test")
    assert stems == ["shard_c", "shard_a", "shard_b"]


def test_ordered_source_stems_out_of_range_raises() -> None:
    packed_info = psf.PackedProvenanceInfo(
        base_shard_stems=["shard_a"],
        packed_shard_paths=[],
        provenance_shard_paths=[],
        packed_stem_fingerprint="fp1",
        packed_row_fingerprint="fp2",
        provenance_fingerprint="fp3",
    )
    # Same message shape as the in-loop bound check it mirrors (error identity
    # with the reference capacity-1 path — see the docstring in the source).
    with pytest.raises(psf.SpliceProvenanceError, match="base_shard_idx 7 out of range"):
        psf._ordered_source_stems([(7, 0)], packed_info, "shard_test")


# --- retain block wiring (full main() integration) --------------------------


def test_capacity_one_skips_retain(tmp_path: Path, monkeypatch) -> None:
    """At capacity 1 (both indices) the script must NOT touch the cache protocol
    (structural legacy path). Spy on FeatureIndex.drop_except."""
    calls = _spy_calls(monkeypatch, psf.FeatureIndex, "drop_except")
    env = build_interleaved_env(tmp_path)
    run_selection_main(env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch)
    assert calls["n"] == 0


def test_capacity_gt_one_invokes_retain(tmp_path: Path, monkeypatch) -> None:
    calls = _spy_calls(monkeypatch, psf.FeatureIndex, "drop_except")
    env = build_interleaved_env(tmp_path)
    run_selection_main(env, cache_shards=3, sidecar_shards=3, monkeypatch=monkeypatch)
    assert calls["n"] >= 1


def test_feature_only_capacity_gates_independently(tmp_path: Path, monkeypatch) -> None:
    """Feature cache on (3), sidecar cache off (1): only FeatureIndex sees
    drop_except -- the two knobs must gate their own index, not move in
    lockstep."""
    f_calls = _spy_calls(monkeypatch, psf.FeatureIndex, "drop_except")
    s_calls = _spy_calls(monkeypatch, psf.SpliceProvenanceIndex, "drop_except")
    env = build_interleaved_env(tmp_path)
    run_selection_main(env, cache_shards=3, sidecar_shards=1, monkeypatch=monkeypatch)
    assert f_calls["n"] >= 1
    assert s_calls["n"] == 0


def test_sidecar_only_capacity_gates_independently(tmp_path: Path, monkeypatch) -> None:
    """Sidecar cache on (3), feature cache off (1): only SpliceProvenanceIndex
    sees drop_except."""
    f_calls = _spy_calls(monkeypatch, psf.FeatureIndex, "drop_except")
    s_calls = _spy_calls(monkeypatch, psf.SpliceProvenanceIndex, "drop_except")
    env = build_interleaved_env(tmp_path)
    run_selection_main(env, cache_shards=1, sidecar_shards=3, monkeypatch=monkeypatch)
    assert f_calls["n"] == 0
    assert s_calls["n"] >= 1


# --- config validation --------------------------------------------------


def test_invalid_cache_shards_raises(tmp_path: Path, monkeypatch) -> None:
    env = build_interleaved_env(tmp_path)
    with pytest.raises(ValueError):
        run_selection_main(env, cache_shards=0, sidecar_shards=1, monkeypatch=monkeypatch)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0])
def test_invalid_source_cache_shards_raises(tmp_path: Path, monkeypatch, bad) -> None:
    env = build_interleaved_env(tmp_path)
    with pytest.raises(ValueError):
        run_selection_main(env, cache_shards=bad, sidecar_shards=1, monkeypatch=monkeypatch)


@pytest.mark.parametrize("bad", [0, -1, True, 1.0])
def test_invalid_sidecar_cache_shards_raises(tmp_path: Path, monkeypatch, bad) -> None:
    env = build_interleaved_env(tmp_path)
    with pytest.raises(ValueError):
        run_selection_main(env, cache_shards=1, sidecar_shards=bad, monkeypatch=monkeypatch)


# --- unequal-capacity correctness (issue #740 review fix) -------------------
#
# The retain pre-seed loop originally gated each index's ensure_resident by
# ITS OWN capacity independently (`i < feature_index.max_resident_shards` /
# `i < splice_index.max_resident_shards`). At UNEQUAL capacities -- including
# the shipped default source=5/sidecar=16 -- the larger-capacity index kept
# pre-loading stems the smaller one had already stopped at, so a sidecar
# could be pre-touched before its paired feature shard. Output bytes stayed
# identical, but first-error identity broke: a corrupt feature shard of one
# stem plus a corrupt sidecar of a LATER stem could raise a different
# exception than the capacity-1 reference. The fix bounds the pre-seed loop
# by min(both capacities) so the pre-seeded prefix is always a strict prefix
# of the reference's own access order. These tests are the missing coverage
# that would have caught the bug -- every test above only ever sets both
# capacities equal.


def _stems_first_need_for_shard(packed_row_dir: Path, shard_path: Path) -> list[str]:
    """Recompute the exact stems_first_need order _ordered_source_stems would
    produce for one packed shard -- used to pick deterministic corruption
    targets by first-need position rather than guessing stem names."""
    packed_info = validate_packed_provenance(packed_row_dir)
    prov = load_overlay_shard(provenance_dir(packed_row_dir) / shard_path.name)
    packed = load_packed_row_shard(shard_path)
    base_idx = prov[PROVENANCE_BASE_SHARD_IDX_KEY]
    doc_idx = prov[PROVENANCE_DOC_IDX_KEY]
    groups: dict[tuple[int, int], list[int]] = {}
    for seg_idx in range(packed.total_segments):
        key = (int(base_idx[seg_idx].item()), int(doc_idx[seg_idx].item()))
        groups.setdefault(key, []).append(seg_idx)
    ordered_group_keys = [
        key for key, _segs in sorted(groups.items(), key=lambda item: min(item[1]))
    ]
    return psf._ordered_source_stems(ordered_group_keys, packed_info, shard_path.stem)


def _find_shard_with_min_stems(packed_row_dir: Path, min_stems: int) -> list[str]:
    """First-need stems of the first packed shard (by name) referencing
    >= min_stems distinct source stems. Raises if the fixture no longer has
    one (self-documenting guard, matching test_splice_preselection_parity.py's
    own fixture guard test)."""
    for shard_path in sorted(glob_overlay_shards(packed_row_dir)):
        stems = _stems_first_need_for_shard(packed_row_dir, shard_path)
        if len(stems) >= min_stems:
            return stems
    raise AssertionError(
        f"no packed shard in {packed_row_dir} references >= {min_stems} distinct "
        "source stems -- build_interleaved_env's packing parameters must have "
        "changed; this test needs >= 4 to exercise unequal caps (2, 5)."
    )


def _corrupt_feature_shard(feature_dir: Path, stem: str) -> None:
    """Overwrite a source feature shard with bytes that fail zstd decompression
    deterministically (same bytes -> same error, every time it is parsed)."""
    path = feature_dir / f"{stem}_features.safetensors.zst"
    assert path.is_file(), f"expected feature shard at {path}"
    path.write_bytes(b"not-a-valid-zstd-frame-deliberately-corrupted-for-test")


def _corrupt_sidecar(feature_dir: Path, stem: str) -> None:
    """Break a _splice_provenance sidecar's schema_version so SpliceProvenanceIndex
    raises deterministically as soon as it is parsed."""
    path = feature_dir / "_splice_provenance" / f"{stem}.json"
    assert path.is_file(), f"expected sidecar at {path}"
    obj = json.loads(path.read_text())
    obj["schema_version"] = 1
    path.write_text(json.dumps(obj))


def _capture_exception(fn, *args, **kwargs) -> BaseException:
    """Run fn and return the exception it raises (fails the test if it doesn't)."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        return exc
    raise AssertionError(f"{fn!r} did not raise")


def test_unequal_cap_byte_parity_against_reference(tmp_path: Path, monkeypatch) -> None:
    """Byte-identity at the shipped default (5, 16) and at (2, 5) -- both
    UNEQUAL caps -- against the (1, 1) reference, on the SAME build (isolates
    the cache-capacity effect from the cross-build upstream-fingerprint
    nondeterminism documented in plan addendum 2026-07-19b; same pattern as
    Task 2.4's same-build parity harness). The fixture's worst packed shard
    references 4 distinct stems, so min(2, 5) = 2 genuinely leaves a 2-stem
    tail deferred to lazy loading -- this is not a vacuous check.
    """
    env = build_interleaved_env(tmp_path)
    packed_row_dir = Path(env["packed_row_dir"])

    default_produced = _shards(
        run_selection_main(env, cache_shards=5, sidecar_shards=16, monkeypatch=monkeypatch)
    )
    shutil.rmtree(packed_row_dir / "sel")

    unequal_produced = _shards(
        run_selection_main(env, cache_shards=2, sidecar_shards=5, monkeypatch=monkeypatch)
    )
    shutil.rmtree(packed_row_dir / "sel")

    reference = _shards(
        run_selection_main(env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch)
    )

    for label, produced in (("(5,16) default", default_produced), ("(2,5)", unequal_produced)):
        assert set(produced) == set(reference), f"{label}: shard file set differs from (1,1)"
        for name in reference:
            assert produced[name] == reference[name], f"{label}: {name} bytes differ from (1,1)"


def test_two_fault_error_identity_at_unequal_caps(tmp_path: Path, monkeypatch) -> None:
    """Regression test for the review-found bug: with a corrupt FEATURE shard
    on one source stem and a corrupt SIDECAR on a DIFFERENT, LATER (first-need
    order) source stem, an unequal-capacity run must raise the IDENTICAL
    exception (type + message) as the (1, 1) reference.

    Construction (verified by hand against this exact fixture, see
    task-2.3-report.md): the packed shard with 4 distinct first-need stems
    [s0, s1, s2, s3] is run with cache_shards=2 (feature), sidecar_shards=5
    (sidecar). The OLD, buggy per-index-bound pre-seed loop reached s2/s3 on
    the sidecar side (positions 2, 3 < 5) while never reaching them on the
    feature side (positions 2, 3 >= 2) -- so a corrupt sidecar at s3 fired
    during pre-seed, BEFORE the main loop ever reached s2's corrupt feature
    shard, which is what the (1, 1) reference (no pre-seed at all) hits
    first. The fix's min(2, 5) = 2 bound never pre-seeds s2 or s3 on EITHER
    index, so both faulty artifacts are left to the main loop's natural
    order, which reaches s2 (the earlier of the two) first at every
    capacity -- matching the reference. Confirmed empirically: temporarily
    reverting to the old per-index-bound loop makes this test fail (candidate
    raises SpliceProvenanceError on s3's sidecar; reference raises ZstdError
    on s2's feature shard).

    Single build, not two independent build_interleaved_env calls:
    SpliceProvenanceError messages embed the absolute _splice_provenance
    directory path, which would differ across independent builds' tmp_path
    prefixes for reasons unrelated to cache capacity (the same class of
    landmine plan addendum 2026-07-19b already hit for manifest
    fingerprints). One build + resetting sel/ between runs (Task 2.4's
    same-build parity pattern) keeps the comparison exact.
    """
    env = build_interleaved_env(tmp_path)
    packed_row_dir = Path(env["packed_row_dir"])
    feature_dir = Path(env["feature_dir"])

    stems = _find_shard_with_min_stems(packed_row_dir, min_stems=4)
    feature_fault_stem = stems[2]  # position 2: >= feature cap (2), < sidecar cap (5)
    sidecar_fault_stem = stems[3]  # position 3: >= feature cap (2), < sidecar cap (5)
    assert feature_fault_stem != sidecar_fault_stem

    _corrupt_feature_shard(feature_dir, feature_fault_stem)
    _corrupt_sidecar(feature_dir, sidecar_fault_stem)

    candidate_exc = _capture_exception(
        run_selection_main, env, cache_shards=2, sidecar_shards=5, monkeypatch=monkeypatch
    )
    shutil.rmtree(packed_row_dir / "sel")
    reference_exc = _capture_exception(
        run_selection_main, env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch
    )

    assert type(candidate_exc) is type(reference_exc), (
        f"candidate raised {type(candidate_exc).__name__}, reference raised "
        f"{type(reference_exc).__name__}"
    )
    assert str(candidate_exc) == str(reference_exc)


# --- Issue #740 hardening item 5: direct min(caps) prefix-content spy ------


def test_pre_seed_prefix_matches_min_capacity(tmp_path: Path, monkeypatch) -> None:
    """Direct, fast tripwire for the exact bug class fixed in commit
    6df9d127: the retain pre-seed loop must touch EXACTLY
    stems_first_need[:min(caps)] on BOTH indices, in call order -- not more,
    not fewer, not reordered.

    Instruments ``_ordered_source_stems`` (called exactly once per packed
    shard, immediately before the pre-seed loop) to record each shard's
    stems_first_need by shard_name and to mark which shard is "current", then
    spies on FeatureIndex.ensure_resident / SpliceProvenanceIndex.
    ensure_resident to record the exact call-order stem sequence attributed
    to whichever shard is current. At unequal caps (2, 5) on
    build_interleaved_env (whose worst packed shard references 4 distinct
    stems), asserts that EVERY shard's captured ensure_resident sequence on
    BOTH indices equals stems_first_need[:2] exactly, and that at least one
    shard actually has > 2 stems (so the prefix check is not vacuous).
    """
    env = build_interleaved_env(tmp_path)
    cache_shards, sidecar_shards = 2, 5
    min_cap = min(cache_shards, sidecar_shards)

    expected_by_shard: dict[str, list[str]] = {}
    current_shard: list[str] = [""]
    feature_calls_by_shard: dict[str, list[str]] = {}
    sidecar_calls_by_shard: dict[str, list[str]] = {}

    real_ordered_source_stems = psf._ordered_source_stems

    def ordered_spy(ordered_group_keys, packed_info, shard_name):
        stems = real_ordered_source_stems(ordered_group_keys, packed_info, shard_name)
        expected_by_shard[shard_name] = stems
        current_shard[0] = shard_name
        feature_calls_by_shard[shard_name] = []
        sidecar_calls_by_shard[shard_name] = []
        return stems

    real_feature_ensure = psf.FeatureIndex.ensure_resident
    real_sidecar_ensure = psf.SpliceProvenanceIndex.ensure_resident

    def feature_spy(self, stem):
        feature_calls_by_shard[current_shard[0]].append(stem)
        real_feature_ensure(self, stem)

    def sidecar_spy(self, stem):
        sidecar_calls_by_shard[current_shard[0]].append(stem)
        real_sidecar_ensure(self, stem)

    monkeypatch.setattr(psf, "_ordered_source_stems", ordered_spy)
    monkeypatch.setattr(psf.FeatureIndex, "ensure_resident", feature_spy)
    monkeypatch.setattr(psf.SpliceProvenanceIndex, "ensure_resident", sidecar_spy)

    run_selection_main(
        env, cache_shards=cache_shards, sidecar_shards=sidecar_shards, monkeypatch=monkeypatch
    )

    assert expected_by_shard, "the retain block never ran (f_cache_on/s_cache_on both false?)"
    checked_nonvacuous = 0
    for shard_name, stems_first_need in expected_by_shard.items():
        expected_prefix = stems_first_need[:min_cap]
        assert feature_calls_by_shard[shard_name] == expected_prefix, (
            f"{shard_name}: FeatureIndex.ensure_resident calls "
            f"{feature_calls_by_shard[shard_name]} != expected min({cache_shards},"
            f"{sidecar_shards})={min_cap} prefix {expected_prefix} of "
            f"stems_first_need={stems_first_need}"
        )
        assert sidecar_calls_by_shard[shard_name] == expected_prefix, (
            f"{shard_name}: SpliceProvenanceIndex.ensure_resident calls "
            f"{sidecar_calls_by_shard[shard_name]} != expected min({cache_shards},"
            f"{sidecar_shards})={min_cap} prefix {expected_prefix} of "
            f"stems_first_need={stems_first_need}"
        )
        if len(stems_first_need) > min_cap:
            checked_nonvacuous += 1
    assert checked_nonvacuous > 0, (
        "no packed shard in this fixture references > "
        f"{min_cap} distinct stems -- cannot distinguish a correct prefix "
        "from a vacuous full-coverage case"
    )


# --- Issue #740 hardening item 6: script-level over-capacity warning -------


def test_stems_first_need_exceeds_capacity_warns(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """The SCRIPT's own over-capacity warning (select_splice_counterfactual_
    features_for_shard), distinct from the per-index retain() overflow
    warnings already unit-tested in test_feature_index_lru_n.py /
    test_splice_provenance_lru_n.py. At (2, 2) on build_interleaved_env
    (whose worst packed shard references 4 distinct stems > cap 2), this
    must fire with the exact shard name and stem/capacity counts.
    """
    env = build_interleaved_env(tmp_path)
    max_distinct, _max_alt = worst_case_stem_alternations(env["packed_row_dir"])
    assert max_distinct > 2, (
        f"fixture's worst packed shard references only {max_distinct} distinct "
        "stems; need > 2 to exceed cache capacity 2 and trigger the warning"
    )

    with caplog.at_level(logging.WARNING):
        run_selection_main(env, cache_shards=2, sidecar_shards=2, monkeypatch=monkeypatch)

    pattern = re.compile(
        rf"shard_\d+: references {max_distinct} source shards > source cache capacity 2"
    )
    matches = [r.message for r in caplog.records if pattern.search(r.message)]
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert matches, (
        f"expected a warning matching {pattern.pattern!r} from "
        f"select_splice_counterfactual_features_for_shard; WARNING records were: "
        f"{warnings}"
    )


# --- Issue #740 hardening item 9 (diagnostic): warm-vs-cold cache hits -----


def test_cache_hits_positive_at_shared_capacity(tmp_path: Path, monkeypatch) -> None:
    """Diagnostic, not correctness: at cap > 1 on the multi-packed-shard
    fixture, both indices must record cache_stats.hits > 0 -- proof that
    cross-shard reuse is actually happening, not just that output bytes
    happen to match. This is the only test in the hardening pass that would
    catch a silently-defeated optimization (e.g. retain()/ensure_resident()
    becoming permanent no-ops while output stays byte-identical only because
    the reference path never depended on caching in the first place).
    """
    env = build_interleaved_env(tmp_path)
    feature_indices: list[psf.FeatureIndex] = []
    sidecar_indices: list[psf.SpliceProvenanceIndex] = []

    real_feature_init = psf.FeatureIndex.__init__
    real_sidecar_init = psf.SpliceProvenanceIndex.__init__

    def feature_init_spy(self, *args, **kwargs):
        real_feature_init(self, *args, **kwargs)
        feature_indices.append(self)

    def sidecar_init_spy(self, *args, **kwargs):
        real_sidecar_init(self, *args, **kwargs)
        sidecar_indices.append(self)

    monkeypatch.setattr(psf.FeatureIndex, "__init__", feature_init_spy)
    monkeypatch.setattr(psf.SpliceProvenanceIndex, "__init__", sidecar_init_spy)

    run_selection_main(env, cache_shards=5, sidecar_shards=16, monkeypatch=monkeypatch)

    assert len(feature_indices) == 1, f"expected 1 FeatureIndex, got {len(feature_indices)}"
    assert len(sidecar_indices) == 1, (
        f"expected 1 SpliceProvenanceIndex, got {len(sidecar_indices)}"
    )
    assert feature_indices[0].cache_stats.hits > 0, (
        "FeatureIndex saw zero cache hits at cache_shards=5 on the multi-shard "
        "fixture -- caching is silently defeated (or the fixture no longer "
        "reuses stems across packed shards)"
    )
    assert sidecar_indices[0].cache_stats.hits > 0, (
        "SpliceProvenanceIndex saw zero cache hits at sidecar_shards=16 on the "
        "multi-shard fixture -- caching is silently defeated"
    )
