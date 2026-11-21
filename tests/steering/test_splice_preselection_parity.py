"""Byte-identity parity, resume, failure-identity, and immutability tests for
the splice-preselection bounded-cache retain protocol (issue #740).

Every parity/resume/failure test builds exactly ONE
``build_interleaved_env`` and runs BOTH the (1, 1) reference and a candidate
capacity on that SAME build (snapshot -> wipe sel/ -> rerun), never two
independent builds. Two independent builds' manifest.json legitimately
differ in two upstream-input fingerprint fields (plan addendum 2026-07-19b),
and SpliceProvenanceError messages embed absolute paths that would
otherwise differ across independent tmp_path roots for reasons unrelated to
cache capacity (the same landmine addendum 2026-07-19c documents for
test_splice_preselection_retain.py's two-fault regression test).

See tests/steering/test_splice_preselection_retain.py for the UNEQUAL-
capacity retain-block wiring coverage (feature-vs-sidecar independent
gating, and the two-fault first-error-identity regression at (2, 5)) this
file does not repeat -- this file adds the MASK x CLUSTERS x CAPACITY
dimensions to the parity matrix, plus resume, single-fault failure-identity,
and cache-tensor immutability.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
import zstandard
from safetensors.torch import load as safetensors_load

import scripts.preprocess.precompute_selected_features as psf
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.gradient_overlay import PROVENANCE_DOC_IDX_KEY, provenance_dir
from steering.overlay_io import load_overlay_shard, save_overlay_shard
from tests.scripts.test_splice_selection_golden import _shards
from tests.steering._splice_cf_cache_fixtures import (
    WHITELIST_MASK_BELOW_150,
    build_interleaved_env,
    build_multisplice_env,
    run_selection_main,
    splice_doc_stats,
    worst_case_stem_alternations,
    write_cluster_yaml,
)


def test_fixture_exhibits_interleaved_access(tmp_path: Path) -> None:
    """The parity fixture must reproduce the multi-source cache-thrashing pattern;
    otherwise the byte-identity matrix would silently only cover the trivial
    single-stem case that never evicts."""
    env = build_interleaved_env(tmp_path)
    distinct, alternations = worst_case_stem_alternations(env["packed_row_dir"])
    assert distinct >= 3, f"fixture packed shards reference only {distinct} distinct stems"
    assert alternations >= 4, f"fixture shows only {alternations} stem alternations"


def test_fixture_exhibits_zero_splice_docs(tmp_path: Path) -> None:
    """The parity fixture must also reproduce the no-op-splice case issue #740
    asks for: at least one derived doc with ZERO splice positions (a plain
    carried-through source doc, or a loop-doc whose own internal segments are
    all natural adjacency). Previously true only by accident (4 of 16 derived
    docs, never asserted); pin it as a guaranteed, named dimension so a future
    fixture change cannot silently drop this case."""
    env = build_interleaved_env(tmp_path)
    total, zero, _max_splices = splice_doc_stats(env["feature_dir"])
    assert zero >= 1, f"fixture produced {zero}/{total} zero-splice derived docs; need >= 1"


def test_fixture_exhibits_multisplice_docs(tmp_path: Path) -> None:
    """The multi-splice fixture (issue #740 hardening item 1) must reproduce
    >= 2 splice positions within a SINGLE derived doc; otherwise the
    integration parity test below would silently only cover the same <= 1-
    splice-per-doc case build_interleaved_env already exercises (see
    test_fixture_exhibits_interleaved_access / test_fixture_exhibits_zero_
    splice_docs), never genuinely reaching the multi-splice code path in
    resolve_counterfactual_refs / compute_doc_counterfactual_deltas."""
    env = build_multisplice_env(tmp_path)
    _total, _zero, max_splices = splice_doc_stats(env["feature_dir"])
    assert max_splices >= 2, (
        f"multisplice fixture's derived docs never exceed {max_splices} splice "
        "position(s) in one doc -- build_multisplice_env's doc length/"
        "cut_fraction must have changed; see steering/feature_shard_splice.py's "
        "recombine_segments for why longer docs are needed to chain >= 2 "
        "splices into one derived doc."
    )


# --- Shared helpers ----------------------------------------------------------

# Capacity pairs the parity matrix must cover (plan addendum 2026-07-19c):
# two EQUAL pairs plus the shipped default (5, 16) and another UNEQUAL pair
# (2, 5) -- equal-only coverage was the blind spot that let the retain
# pre-seed loop's per-index-bound bug (fixed in Task 2.3) go undetected.
_CAP_PAIRS = [(2, 2), (4, 4), (5, 16), (2, 5)]


def _total_selected(files: dict[str, bytes]) -> int:
    """Sum sel_fids + neg_fids across every selection shard in ``files``.

    A cheap non-degeneracy witness for the mask/cluster parity arms: a
    byte-identical empty-vs-empty comparison would trivially "pass" without
    proving the retain path was exercised against any real selected
    feature. Every parity test below asserts this is > 0.
    """
    dctx = zstandard.ZstdDecompressor()
    total = 0
    for data in _shards(files).values():
        tensors = safetensors_load(dctx.decompress(data))
        total += len(tensors["sel_fids"]) + len(tensors["neg_fids"])
    return total


def _one_build_ref_vs_candidate(
    tmp_path,
    *,
    mask: dict | None = None,
    clusters_path: str | None = None,
    cache_shards: int,
    sidecar_shards: int,
    monkeypatch,
    env_factory=build_interleaved_env,
) -> dict[str, bytes]:
    """Run a candidate capacity then the (1, 1) reference on the SAME build.

    Plan addendum 2026-07-19b item 3: snapshot -> wipe sel/ -> rerun, so both
    arms share identical upstream-input fingerprints (derived_feature_
    fingerprint / sidecar_fingerprint), and the FULL tree (every shard PLUS
    manifest.json) can be compared byte-for-byte -- not just shards, which is
    all a cross-build comparison could ever support.

    ``env_factory`` defaults to build_interleaved_env; pass
    build_multisplice_env to run this SAME reference-vs-candidate comparison
    over the multi-splice-per-doc corpus (issue #740 hardening item 1).
    """
    env = env_factory(tmp_path)
    candidate = run_selection_main(
        env,
        cache_shards=cache_shards,
        sidecar_shards=sidecar_shards,
        mask=mask,
        clusters_path=clusters_path,
        monkeypatch=monkeypatch,
    )
    shutil.rmtree(env["packed_row_dir"] / "sel")
    reference = run_selection_main(
        env,
        cache_shards=1,
        sidecar_shards=1,
        mask=mask,
        clusters_path=clusters_path,
        monkeypatch=monkeypatch,
    )
    assert set(reference) == set(candidate), (
        f"sel file set differs (cache_shards={cache_shards}, "
        f"sidecar_shards={sidecar_shards}, mask={mask is not None}, "
        f"clusters={clusters_path is not None})"
    )
    for name in reference:
        assert reference[name] == candidate[name], (
            f"{name} differs (cache_shards={cache_shards}, "
            f"sidecar_shards={sidecar_shards}, mask={mask is not None}, "
            f"clusters={clusters_path is not None})"
        )
    return reference


# --- Task 2.4: parity matrix (mask x clusters x capacity) -------------------


@pytest.mark.parametrize("cache_shards,sidecar_shards", _CAP_PAIRS)
def test_parity_no_mask(tmp_path, monkeypatch, cache_shards, sidecar_shards) -> None:
    reference = _one_build_ref_vs_candidate(
        tmp_path,
        cache_shards=cache_shards,
        sidecar_shards=sidecar_shards,
        monkeypatch=monkeypatch,
    )
    assert _total_selected(reference) > 0


@pytest.mark.parametrize("cache_shards,sidecar_shards", _CAP_PAIRS)
def test_parity_whitelist_mask(tmp_path, monkeypatch, cache_shards, sidecar_shards) -> None:
    reference = _one_build_ref_vs_candidate(
        tmp_path,
        mask=WHITELIST_MASK_BELOW_150,
        cache_shards=cache_shards,
        sidecar_shards=sidecar_shards,
        monkeypatch=monkeypatch,
    )
    assert _total_selected(reference) > 0


@pytest.mark.parametrize("cache_shards,sidecar_shards", _CAP_PAIRS)
def test_parity_active_clusters(tmp_path, monkeypatch, cache_shards, sidecar_shards) -> None:
    clusters = write_cluster_yaml(tmp_path)
    reference = _one_build_ref_vs_candidate(
        tmp_path,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=clusters,
        cache_shards=cache_shards,
        sidecar_shards=sidecar_shards,
        monkeypatch=monkeypatch,
    )
    assert _total_selected(reference) > 0


@pytest.mark.parametrize("cache_shards,sidecar_shards", _CAP_PAIRS)
def test_parity_identity_after_mask_clusters(
    tmp_path, monkeypatch, cache_shards, sidecar_shards
) -> None:
    clusters = write_cluster_yaml(tmp_path, identity_after_mask=True)
    reference = _one_build_ref_vs_candidate(
        tmp_path,
        mask=WHITELIST_MASK_BELOW_150,
        clusters_path=clusters,
        cache_shards=cache_shards,
        sidecar_shards=sidecar_shards,
        monkeypatch=monkeypatch,
    )
    assert _total_selected(reference) > 0


# --- Issue #740 hardening item 1: multi-splice-per-doc integration ---------


def test_multisplice_parity_matches_reference(tmp_path, monkeypatch) -> None:
    """Byte-identity at an unequal candidate capacity (2, 5) against the
    (1, 1) reference, on the SAME multi-splice build (issue #740 hardening
    item 1). Genuine multi-splice-per-doc docs (>= 2 splice positions in one
    derived doc -- see test_fixture_exhibits_multisplice_docs) exercise
    resolve_counterfactual_refs + the delta path across cache capacities, a
    dimension build_interleaved_env's docs (never > 1 splice/doc) cannot
    reach: every splice position in a multi-splice doc resolves its
    would-have-followed reference via a SEPARATE feature_index.lookup call
    within the SAME doc-processing iteration, so a cache bug that corrupts
    state between successive lookups on the same doc would only surface here.
    """
    reference = _one_build_ref_vs_candidate(
        tmp_path,
        cache_shards=2,
        sidecar_shards=5,
        monkeypatch=monkeypatch,
        env_factory=build_multisplice_env,
    )
    assert _total_selected(reference) > 0


def test_repeated_run_determinism(tmp_path, monkeypatch) -> None:
    """Cross-BUILD shard-bytes determinism.

    manifest.json legitimately differs across independent builds (addendum
    2026-07-19b), so only shards are compared here -- full-tree identity
    across CACHE CAPACITY on the SAME build is what every parametrized test
    above already asserts.
    """
    a = _shards(
        run_selection_main(
            build_interleaved_env(tmp_path / "a"),
            cache_shards=4,
            sidecar_shards=4,
            monkeypatch=monkeypatch,
        )
    )
    b = _shards(
        run_selection_main(
            build_interleaved_env(tmp_path / "b"),
            cache_shards=4,
            sidecar_shards=4,
            monkeypatch=monkeypatch,
        )
    )
    assert set(a) == set(b) and a, "shard set differs / empty across regenerations"
    for name in a:
        assert a[name] == b[name], f"{name} shard bytes nondeterministic across builds"


# --- Task 2.5: resume ---------------------------------------------------------


def test_resume_matches_full_run(tmp_path, monkeypatch) -> None:
    """ONE build: run a full candidate pass, snapshot it, delete all-but-
    the-first sel shard, rerun (the resume path), and compare the resumed
    output to the snapshot.

    Same build => shards AND manifest.json must both be identical (addendum
    2026-07-19b item 4) -- comparing against a SECOND, independently built
    "full" run would spuriously fail on the two upstream-input fingerprint
    fields that addendum documents, for reasons unrelated to resume
    correctness. Uses an unequal candidate cap (2, 5) so the retain block
    is genuinely exercised on both the initial and the resumed pass.
    """
    env = build_interleaved_env(tmp_path)
    full = run_selection_main(env, cache_shards=2, sidecar_shards=5, monkeypatch=monkeypatch)

    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shards = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert len(shards) >= 2, (
        f"fixture produced only {len(shards)} packed sel shard(s); need >= 2 "
        "to exercise a genuine partial resume"
    )
    for p in shards[1:]:
        p.unlink()

    resumed = run_selection_main(env, cache_shards=2, sidecar_shards=5, monkeypatch=monkeypatch)
    assert set(resumed) == set(full)
    for name in full:
        assert resumed[name] == full[name], f"{name} differs after resume"


# --- Issue #740 hardening item 2: resume at a DIFFERENT cache capacity -----
#
# The cache-shard knobs are producer-only and are NOT part of the selection
# hash (precompute_selected_features.main's selection_source_cache_shards /
# selection_sidecar_cache_shards are omitted from compute_selection_hash), so
# an operator resuming an interrupted run may legitimately choose a different
# capacity than the interrupted run used. test_resume_matches_full_run above
# only ever resumes at the SAME capacity it interrupted -- these two tests
# cover both directions of a capacity CHANGE across the interruption.


def test_resume_at_higher_capacity_than_interrupted(tmp_path, monkeypatch) -> None:
    """Interrupt a (1, 1) run, resume at a LARGER (5, 16).

    ``full`` is itself an un-resumed (1, 1) run, so it already IS "a fresh
    uninterrupted (1, 1) run on the same build" -- no separate reference run
    is needed for this direction.
    """
    env = build_interleaved_env(tmp_path)
    full = run_selection_main(env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch)

    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shards = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert len(shards) >= 2, (
        f"fixture produced only {len(shards)} packed sel shard(s); need >= 2 "
        "to exercise a genuine partial resume"
    )
    for p in shards[1:]:
        p.unlink()

    resumed = run_selection_main(env, cache_shards=5, sidecar_shards=16, monkeypatch=monkeypatch)
    assert set(resumed) == set(full)
    for name in full:
        assert resumed[name] == full[name], (
            f"{name} differs after resuming an interrupted (1,1) run at (5,16)"
        )


def test_resume_at_lower_capacity_than_interrupted(tmp_path, monkeypatch) -> None:
    """Reverse direction: interrupt a (5, 16) run, resume at (1, 1).

    Here the initial full pass was NOT itself the (1, 1) reference, so this
    test gets an INDEPENDENT fresh (1, 1) run on the SAME build (snapshot ->
    wipe sel/ -> rerun, the same pattern _one_build_ref_vs_candidate uses)
    rather than assuming the (5, 16) snapshot is an equivalent stand-in for
    it -- a hypothetical bug shared by both the (5,16) pass and a (1,1)
    resume of it would otherwise go undetected by a resumed-vs-full
    comparison alone.
    """
    env = build_interleaved_env(tmp_path)
    run_selection_main(env, cache_shards=5, sidecar_shards=16, monkeypatch=monkeypatch)

    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shards = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert len(shards) >= 2, (
        f"fixture produced only {len(shards)} packed sel shard(s); need >= 2 "
        "to exercise a genuine partial resume"
    )
    for p in shards[1:]:
        p.unlink()

    resumed = run_selection_main(env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch)

    shutil.rmtree(env["packed_row_dir"] / "sel")
    fresh_reference = run_selection_main(
        env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch
    )
    assert set(resumed) == set(fresh_reference)
    for name in fresh_reference:
        assert resumed[name] == fresh_reference[name], (
            f"{name} differs after resuming an interrupted (5,16) run at (1,1)"
        )


# --- Issue #740 hardening item 3: resume with a non-first missing shard ----


def test_resume_with_missing_shard_not_first(tmp_path, monkeypatch) -> None:
    """Resume when the MISSING shard is NOT the first one exercises a
    subtlety test_resume_matches_full_run (which always deletes all-but-the-
    FIRST shard) does not: the skip-path for already-done shards preceding
    the missing one (splice_counterfactual_stats_for_packed_shard) touches
    the shared splice_index (via doc_splices) for those shards BEFORE the
    main loop ever reaches the live (missing) one. Verify this does not
    pollute cache state in a way that changes output: build a corpus with
    >= 3 packed sel shards, run full at (2, 5), delete the MIDDLE shard only
    (leaving one intact on each side), resume at the SAME (2, 5), and compare
    against an independent fresh (1, 1) reference on the same build.
    """
    env = build_interleaved_env(tmp_path, n_shards=7)
    run_selection_main(env, cache_shards=2, sidecar_shards=5, monkeypatch=monkeypatch)

    sel_dir = next((env["packed_row_dir"] / "sel").iterdir())
    shards = sorted(sel_dir.glob("shard_*.safetensors.zst"))
    assert len(shards) >= 3, (
        f"fixture produced only {len(shards)} packed sel shard(s); need >= 3 "
        "so a MIDDLE shard can be deleted while leaving a shard before AND "
        "after it intact"
    )
    middle = shards[len(shards) // 2]
    middle.unlink()

    resumed = run_selection_main(env, cache_shards=2, sidecar_shards=5, monkeypatch=monkeypatch)

    shutil.rmtree(env["packed_row_dir"] / "sel")
    reference = run_selection_main(env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch)

    assert set(resumed) == set(reference)
    for name in reference:
        assert resumed[name] == reference[name], (
            f"{name} differs after resuming with a non-first (middle) shard missing"
        )


# --- Task 2.5: single-fault failure-identity ---------------------------------
#
# Each fault is engineered to surface at a specific, DOCUMENTED pipeline
# stage: STARTUP (validate_packed_provenance / the fingerprint pass, before
# FeatureIndex/SpliceProvenanceIndex are even constructed with their cache
# capacities -- reference and candidate execute the textually identical code
# path) or LOAD-TIME (inside the per-shard retain/group-loop machinery,
# where the fault is still capacity-independent because presence/validity of
# an artifact is shard CONTENT, not a caching fact). See each helper's
# docstring for its specific staging. Task 2.3's test_splice_preselection_
# retain.py already covers the TWO-fault case where UNEQUAL caps could
# reorder which of two simultaneous faults is discovered first -- these
# single-fault tests do not repeat that; with only one thing that can go
# wrong, there is no "which one first" question to reorder.


def _delete_sidecar(feature_dir: str, stem: str) -> None:
    """Delete a _splice_provenance sidecar file entirely.

    LOAD-TIME fault: splice_sidecar_fingerprint (the startup pass) only
    checks that the sidecar DIRECTORY exists and contains >= 1 JSON file --
    it never verifies every referenced stem has one -- so this is not caught
    until SpliceProvenanceIndex._parse_shard's ``path.is_file()`` check,
    reached via either the retain block's ensure_resident (cap > 1, if the
    stem's first-need position is within the pre-seed bound) or the group
    loop's resolve_counterfactual_refs (always, for every group -- cap 1, or
    cap > 1 beyond the pre-seed bound). Both call the identical parse path.
    """
    path = Path(feature_dir) / "_splice_provenance" / f"{stem}.json"
    assert path.is_file(), f"expected sidecar at {path}"
    path.unlink()


def _malform_sidecar(feature_dir: str, stem: str) -> None:
    """Overwrite a sidecar with ``{}`` (present, but missing schema_version).

    LOAD-TIME fault, same staging as ``_delete_sidecar``: splice_sidecar_
    fingerprint only hashes raw bytes at startup (never parses JSON), so a
    syntactically-valid-but-empty sidecar passes startup and is only
    discovered when SpliceProvenanceIndex._parse_shard actually parses it.
    """
    path = Path(feature_dir) / "_splice_provenance" / f"{stem}.json"
    assert path.is_file(), f"expected sidecar at {path}"
    path.write_text("{}")


def _corrupt_feature_shard_bytes(feature_dir: str, stem: str) -> None:
    """Overwrite a source feature shard with bytes that fail zstd
    decompression deterministically.

    LOAD-TIME fault: derived_feature_fingerprint (the startup fingerprint
    pass) only hashes raw shard bytes -- it never decompresses/parses them
    -- so this is not caught there. It is not caught until
    FeatureIndex._load_shard actually decompresses the shard, reached via
    either the retain block's ensure_resident (cap > 1, within the pre-seed
    bound) or the group loop's feature_index.lookup (cap 1, or cap > 1
    beyond the pre-seed bound). Both paths decompress the SAME bytes, so
    both raise identically regardless of which reaches it first.

    Caller MUST NOT pass the alphabetically-first shard stem
    ("shard_00000"): main() calls verify_segment_size(feature_dir, ...)
    very early (before selection_hash/output_dir even exist), which reads
    metadata from glob_features(feature_dir)[0] specifically -- corrupting
    that exact shard fails main() before ANY selection code runs (even
    earlier than validate_packed_provenance), which is a different, less
    interesting "trivially identical" STARTUP case already covered by
    ``_truncate_provenance_length`` below. Use a later stem (e.g.
    "shard_00001") to actually reach the retain/group-loop machinery this
    fault is meant to exercise.
    """
    path = Path(feature_dir) / f"{stem}_features.safetensors.zst"
    assert path.is_file(), f"expected feature shard at {path}"
    path.write_bytes(b"not-a-valid-zstd-frame-deliberately-corrupted-for-test")


def _point_provenance_at_missing_doc(packed_row_dir: Path, packed_stem: str) -> None:
    """Point one packed segment's doc_idx at a doc absent from its source
    feature shard's per-doc index.

    LOAD-TIME fault, but structurally CAPACITY-INDEPENDENT: doc presence is
    shard CONTENT (the same per-doc dict is built regardless of whether the
    shard was pre-seeded via ensure_resident or lazily loaded via lookup).
    The retain block itself never touches this fault (ensure_resident loads
    a whole shard; it never probes individual doc ids) -- only
    feature_index.lookup(derived_stem, doc_id) in the group loop discovers a
    missing doc, unconditionally, for every capacity. Not caught by startup
    validate_packed_provenance, which range-checks base_shard_idx but never
    doc_idx.
    """
    prov_path = provenance_dir(packed_row_dir) / f"{packed_stem}.safetensors.zst"
    prov = load_overlay_shard(prov_path)
    doc_idx = prov[PROVENANCE_DOC_IDX_KEY].clone()
    doc_idx[0] = 999_999  # far beyond this fixture's real doc range (0-2)
    prov[PROVENANCE_DOC_IDX_KEY] = doc_idx
    save_overlay_shard(prov, prov_path)


def _truncate_provenance_length(packed_row_dir: Path, packed_stem: str) -> None:
    """Drop one entry from a packed shard's doc_idx provenance tensor so its
    length no longer matches that packed shard's segment count.

    STARTUP fault: caught by validate_packed_provenance's length-equality
    check, called once in main() BEFORE FeatureIndex/SpliceProvenanceIndex
    are even constructed with their (possibly unequal) cache capacities --
    reference and candidate execute the textually identical code path up to
    the raise. The trivially-identical end of the spectrum.
    """
    prov_path = provenance_dir(packed_row_dir) / f"{packed_stem}.safetensors.zst"
    prov = load_overlay_shard(prov_path)
    prov[PROVENANCE_DOC_IDX_KEY] = prov[PROVENANCE_DOC_IDX_KEY][:-1]
    save_overlay_shard(prov, prov_path)


def _run_and_capture(env, *, cache_shards, sidecar_shards, monkeypatch) -> Exception:
    """Run selection and return the raised exception object.

    Fails the test (rather than returning a sentinel) if run_selection_main
    does not raise -- a corruption helper that silently stopped corrupting
    anything must not be able to pass this test vacuously.
    """
    try:
        run_selection_main(
            env,
            cache_shards=cache_shards,
            sidecar_shards=sidecar_shards,
            monkeypatch=monkeypatch,
        )
    except Exception as exc:  # noqa: BLE001 -- comparing error identity across arbitrary types
        return exc
    raise AssertionError("expected run_selection_main to raise, but it succeeded")


def _fault_identity(
    tmp_path,
    corrupt,
    *,
    cache_shards: int = 2,
    sidecar_shards: int = 5,
    monkeypatch,
) -> Exception:
    """Apply ``corrupt(env)`` to a single build, run the candidate, wipe
    sel/, run the (1, 1) reference on the SAME corrupted build, and assert
    the two fail identically (type + message). Returns the shared exception.

    Single build (not two independent build_interleaved_env calls): several
    of the faults above raise messages embedding absolute paths under
    feature_dir/packed_row_dir, which would differ across independent
    tmp_path roots for reasons unrelated to cache capacity -- see the module
    docstring and plan addendum 2026-07-19c.
    """
    env = build_interleaved_env(tmp_path)
    corrupt(env)
    candidate = _run_and_capture(
        env, cache_shards=cache_shards, sidecar_shards=sidecar_shards, monkeypatch=monkeypatch
    )
    # ignore_errors: a sufficiently-early fault (e.g. verify_segment_size's
    # startup peek at the first shard) can raise before sel/ is ever created.
    shutil.rmtree(env["packed_row_dir"] / "sel", ignore_errors=True)
    reference = _run_and_capture(env, cache_shards=1, sidecar_shards=1, monkeypatch=monkeypatch)
    assert type(candidate) is type(reference) and str(candidate) == str(reference), (
        f"candidate raised {type(candidate).__name__}({candidate}) but the "
        f"(1,1) reference raised {type(reference).__name__}({reference})"
    )
    return reference


def test_missing_sidecar_fails_identically(tmp_path, monkeypatch) -> None:
    exc = _fault_identity(
        tmp_path,
        lambda env: _delete_sidecar(env["feature_dir"], "shard_00000"),
        monkeypatch=monkeypatch,
    )
    assert isinstance(exc, SpliceProvenanceError)
    assert "sidecar file missing" in str(exc)


def test_malformed_sidecar_fails_identically(tmp_path, monkeypatch) -> None:
    exc = _fault_identity(
        tmp_path,
        lambda env: _malform_sidecar(env["feature_dir"], "shard_00000"),
        monkeypatch=monkeypatch,
    )
    assert isinstance(exc, SpliceProvenanceError)
    assert "schema_version=" in str(exc)


def test_missing_derived_doc_fails_identically(tmp_path, monkeypatch) -> None:
    exc = _fault_identity(
        tmp_path,
        lambda env: _point_provenance_at_missing_doc(env["packed_row_dir"], "shard_00000"),
        monkeypatch=monkeypatch,
    )
    assert isinstance(exc, SpliceProvenanceError)
    assert "missing derived doc" in str(exc)


def test_provenance_length_mismatch_fails_identically(tmp_path, monkeypatch) -> None:
    exc = _fault_identity(
        tmp_path,
        lambda env: _truncate_provenance_length(env["packed_row_dir"], "shard_00000"),
        monkeypatch=monkeypatch,
    )
    assert isinstance(exc, SpliceProvenanceError)
    assert "provenance has lengths" in str(exc)


def test_corrupt_feature_shard_fails_identically(tmp_path, monkeypatch) -> None:
    # "shard_00001", NOT "shard_00000": see _corrupt_feature_shard_bytes's
    # docstring -- corrupting the alphabetically-first shard is caught by
    # main()'s early verify_segment_size peek, before any selection or
    # cache-capacity-dependent code runs at all.
    exc = _fault_identity(
        tmp_path,
        lambda env: _corrupt_feature_shard_bytes(env["feature_dir"], "shard_00001"),
        monkeypatch=monkeypatch,
    )
    # NOT a SpliceProvenanceError: this fault fails inside zstd decompression
    # (FeatureIndex._load_shard -> load_features), a different failure
    # surface than the sidecar/provenance faults above. Documents that the
    # retain path and the reference path hit the identical non-splice
    # exception, not merely "some exception or other."
    assert not isinstance(exc, SpliceProvenanceError)


# --- Task 2.5: cache-tensor immutability -------------------------------------


def _segments_digest(docs: Mapping) -> str:
    h = hashlib.sha256()
    for doc in sorted(docs):
        for seg in docs[doc]:
            h.update(seg.feature_ids.numpy().tobytes())
            h.update(seg.feature_mags.numpy().tobytes())
    return h.hexdigest()


def test_resident_tensors_not_mutated(tmp_path, monkeypatch) -> None:
    """A full candidate run must not mutate any cached SegmentFeatures tensor.

    Residency would amplify any accidental in-place write into a
    divergence: capture a content checksum of every stem's per-doc tensors
    the FIRST time it is admitted, then re-load each captured stem fresh from
    disk after the whole run finishes and confirm the digest is unchanged.

    Spies on FeatureIndex._admit (issue #740 hardening item 4), NOT
    ensure_resident: _admit is the single insertion point BOTH the retain
    block's pre-seed path (ensure_resident) and the lazy miss path
    (lookup -> _insert_resident) go through. The previous ensure_resident-only
    spy only ever observed pre-seeded stems -- at cache_shards=2 (below the
    fixture's known worst-case 4 distinct stems per packed shard), some
    stems are necessarily reached only via the lazy path (beyond the
    pre-seed bound), so this asserts every stem the FeatureIndex ever
    discovered was ACTUALLY admitted (and therefore checked) at least once,
    regardless of which path reached it.
    """
    env = build_interleaved_env(tmp_path)
    captured: dict[str, str] = {}
    indices: list[psf.FeatureIndex] = []

    real_admit = psf.FeatureIndex._admit

    def spy(self, stem, docs):
        if not indices or indices[-1] is not self:
            indices.append(self)
        if stem not in captured:
            captured[stem] = _segments_digest(docs)
        real_admit(self, stem, docs)

    monkeypatch.setattr(psf.FeatureIndex, "_admit", spy)
    run_selection_main(env, cache_shards=2, sidecar_shards=2, monkeypatch=monkeypatch)
    assert captured, "spy never observed an admitted shard -- FeatureIndex was never populated"
    assert len({id(idx) for idx in indices}) == 1, (
        "expected exactly one FeatureIndex instance across the whole run, got "
        f"{len({id(idx) for idx in indices})}"
    )
    feature_index = indices[0]
    assert set(captured) == set(feature_index.shard_stems), (
        f"admitted stems {sorted(captured)} != discovered shard stems "
        f"{sorted(feature_index.shard_stems)} -- some stem was never resident "
        "(pre-seeded or lazy) during the run, so its mutation would go "
        "unchecked below"
    )

    # Reconstruct the same storage representation. Comparing a compact uint16
    # cache against the default int64 public representation hashes different
    # byte widths even when every feature ID is numerically unchanged.
    fresh = psf.FeatureIndex(
        env["feature_dir"],
        percentile_checksum=env["pct_checksum"],
        skip_feature_mags=feature_index._skip_feature_mags,
        compact_feature_ids=feature_index._compact_feature_ids,
    )
    for stem, digest in captured.items():
        docs = fresh._load_shard(stem)
        assert _segments_digest(docs) == digest, f"{stem} tensors mutated during the run"
