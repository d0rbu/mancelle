"""Tests for shared clustered-evaluation identity primitives.

Cover digest contents and mutation schedules, file-signature layout,
canonical-JSON determinism and rejected inputs, identity construction and
feature-ID relationships, public exports, and exact schema/algorithm constants.
Additional unary-domain property tests live in
``test_cluster_contract_value_domains.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import steering.eval.cluster_contract_identity as cci
import steering.immutable_io as immutable_io
from errors import FeatureClustersError
from steering.eval.cluster_contract_identity import (
    ClusterFeatureContractError,
    FeatureDirectoryIdentity,
    FeatureStatisticsIdentity,
    SaeIdentity,
    StableFileDigest,
    stream_hash_file_stable,
)
from steering.feature_paths import DatasetPhase, FeatureDirectoryComponents, PoolingMode
from steering.immutable_io import is_sha256_hex
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode

# --------------------------------------------------------------------------
# Version / internal constants — pinned exactly (semantics contracts).
# --------------------------------------------------------------------------


def test_version_constants_are_pinned_exactly() -> None:
    assert cci.FEATURE_CONTRACT_SCHEMA_VERSION == 1
    assert cci.MAX_SAFE_SELECTED_FEATURE_WIDTH == 65_536
    assert cci.CONCEPT_PROJECTION_VERSION == "mask_first_min_allowed_representative_v1"
    assert cci.QC_TARGET_REDUCTION_VERSION == "member_token_amax_own_percentile_v1"
    assert cci.PACKED_PERCENTILE_SELECTION_VERSION == "training_uint8_roundtrip_cluster_amax_v1"
    assert cci.FEATURE_DIFFMEAN_PROJECTION_VERSION == "per_completion_uint8_amax_then_mean_v1"
    assert cci.GENERIC_ROOFLINE_SIGNAL_VERSION == "per_document_cluster_signal_v2"
    # _SHA256_HEX_LENGTH removed: the digest-shape check is now the canonical
    # steering.immutable_io.is_sha256_hex (PR #893 review dedup).
    assert cci._HASH_CHUNK_SIZE == 8 * 1024 * 1024


def test_public_surface_matches_the_b1_boundary() -> None:
    """``__all__`` is the B1 surface plus G2-D2's shared stat-signature API."""

    assert set(cci.__all__) == {
        "CONCEPT_PROJECTION_VERSION",
        "FEATURE_CONTRACT_SCHEMA_VERSION",
        "FEATURE_DIFFMEAN_PROJECTION_VERSION",
        "GENERIC_ROOFLINE_SIGNAL_VERSION",
        "MAX_SAFE_SELECTED_FEATURE_WIDTH",
        "PACKED_PERCENTILE_SELECTION_VERSION",
        "QC_TARGET_REDUCTION_VERSION",
        "ClusterFeatureContractError",
        "FeatureDirectoryIdentity",
        "FeatureStatisticsIdentity",
        "SaeIdentity",
        "StableFileDigest",
        "stat_signature_now",
        "stream_hash_file_stable",
    }


def test_cluster_feature_contract_error_is_a_feature_clusters_error() -> None:
    assert issubclass(ClusterFeatureContractError, FeatureClustersError)
    with pytest.raises(FeatureClustersError):
        raise ClusterFeatureContractError("boom")


# --------------------------------------------------------------------------
# Canonical JSON: determinism, non-ASCII, nesting, compact separators.
# --------------------------------------------------------------------------


def test_canonical_json_is_deterministic_across_key_insertion_order() -> None:
    first = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    second = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert cci._canonical_json(first) == cci._canonical_json(second)


def test_canonical_json_uses_compact_separators_with_no_whitespace() -> None:
    assert cci._canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_canonical_json_preserves_non_ascii_characters_literally() -> None:
    encoded = cci._canonical_json({"name": "café", "lang": "日本語"})
    assert "café" in encoded
    assert "日本語" in encoded
    assert "\\u" not in encoded


def test_canonical_json_sorts_keys_recursively_through_nested_containers() -> None:
    value = {"outer": [{"z": 1, "a": 2}, {"b": 3}]}
    assert cci._canonical_json(value) == '{"outer":[{"a":2,"z":1},{"b":3}]}'


def test_sha256_json_matches_manual_hash_of_canonical_json() -> None:
    value = {"b": 1, "a": 2}
    expected = hashlib.sha256(cci._canonical_json(value).encode("utf-8")).hexdigest()
    assert cci._sha256_json(value) == expected


def test_sha256_json_is_insensitive_to_key_insertion_order() -> None:
    assert cci._sha256_json({"a": 1, "b": 2}) == cci._sha256_json({"b": 2, "a": 1})


def test_sha256_json_changes_when_content_changes() -> None:
    assert cci._sha256_json({"a": 1}) != cci._sha256_json({"a": 2})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("a" * 64, True),
        ("0123456789abcdef" * 4, True),
        ("A" * 64, False),  # uppercase hex rejected
        ("a" * 63, False),  # too short
        ("a" * 65, False),  # too long
        ("g" * 64, False),  # non-hex character
        ("", False),
        (None, False),
        (12345, False),
    ],
    ids=[
        "all-a-lowercase",
        "mixed-lowercase-hex",
        "uppercase-rejected",
        "too-short",
        "too-long",
        "non-hex-char",
        "empty-string",
        "none",
        "int",
    ],
)
def test_is_sha256_accepts_only_lowercase_64char_hex(value: Any, expected: bool) -> None:
    assert is_sha256_hex(value) is expected


# --------------------------------------------------------------------------
# _canonical_json — Codex fix batch finding 2 (collision-safety: non-str
# keys, non-finite floats, allow_nan=False, documented -0.0 policy).
# Hash-stability proof: every well-formed payload already covered by the
# pre-fix tests above produces byte-identical output (those tests are
# unmodified and still green); this section additionally pins one exact
# reference digest, computed independently with the pre-fix formula
# (``json.dumps(value, sort_keys=True, separators=(",", ":"),
# ensure_ascii=False)``, no ``allow_nan`` kwarg), to prove the fix changes
# nothing for well-formed input.
# --------------------------------------------------------------------------


def test_canonical_json_hash_is_unchanged_for_a_representative_well_formed_payload() -> None:
    """Reference digest computed independently with the pre-fix formula
    (plain ``json.dumps(..., sort_keys=True, separators=(",", ":"),
    ensure_ascii=False)``, no domain walk, no ``allow_nan``) — proves the fix
    (recursive key/float validation + ``allow_nan=False``) is a no-op for
    well-formed input.
    """
    value = {
        "model": "gemma-2-2b",
        "layer": 8,
        "ignore_bos_features": True,
        "width": "16k",
        "ids": [3, 1, 2],
    }
    assert (
        cci._canonical_json(value)
        == '{"ids":[3,1,2],"ignore_bos_features":true,"layer":8,"model":"gemma-2-2b","width":"16k"}'
    )
    assert (
        cci._sha256_json(value)
        == "7a24cb0b84a82980288884f6b94ef46ed78c35fecfe7c292481326fdd91e746a"
    )


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({1: "x"}, "$"),
        ({True: "x"}, "$"),
        ({None: "x"}, "$"),
        ({1.5: "x"}, "$"),
        ({"a": {2: "y"}}, "$.a"),
        ({"a": [{"b": {3: "z"}}]}, "$.a[0].b"),
    ],
    ids=["int-key", "bool-key", "none-key", "float-key", "nested-key", "deeply-nested-key"],
)
def test_canonical_json_rejects_non_string_keys_recursively_with_path(
    value: Any, path: str
) -> None:
    """A non-str key is rejected outright rather than silently coerced —
    ``json.dumps`` alone would hash ``{1: "x"}`` and ``{"1": "x"}``
    identically, exactly the collision Codex flagged.
    """
    with pytest.raises(
        ClusterFeatureContractError,
        match=rf"^canonical JSON payload at {re.escape(path)} has a non-string key ",
    ):
        cci._canonical_json(value)


def test_canonical_json_does_not_collide_int_and_string_keys() -> None:
    """Direct proof of the collision Codex named: without the fix,
    ``{1: "x"}`` (rejected) and ``{"1": "x"}`` (accepted) would hash
    identically once ``json.dumps`` coerced the int key to a string.
    """
    assert cci._canonical_json({"1": "x"}) == '{"1":"x"}'
    with pytest.raises(ClusterFeatureContractError, match="non-string key"):
        cci._canonical_json({1: "x"})


@pytest.mark.parametrize(
    ("value", "path"),
    [
        (float("nan"), "$"),
        (float("inf"), "$"),
        (float("-inf"), "$"),
        ([1, float("nan")], "$[1]"),
        ({"a": float("inf")}, "$.a"),
        ({"a": [{"b": float("-inf")}]}, "$.a[0].b"),
    ],
    ids=["nan", "inf", "-inf", "in-list", "in-dict", "deeply-nested"],
)
def test_canonical_json_rejects_non_finite_floats_recursively_with_path(
    value: Any, path: str
) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match=rf"^canonical JSON payload at {re.escape(path)} has a non-finite float ",
    ):
        cci._canonical_json(value)


def test_canonical_json_allow_nan_false_is_defense_in_depth_for_json_dumps_itself() -> None:
    """The recursive walk already rejects every non-finite float, so
    ``json.dumps``'s own ``allow_nan=False`` should be unreachable for any
    input this function accepts — this asserts that invariant directly
    rather than just trusting the walk.
    """
    with pytest.raises(ClusterFeatureContractError):
        cci._canonical_json(float("nan"))
    # If allow_nan defaulted back to True and the walk were ever bypassed,
    # json.dumps would silently emit the non-standard `NaN` token instead of
    # raising — assert that stdlib behavior directly so a future edit that
    # drops allow_nan=False is caught even if the walk itself regresses.
    assert json.dumps(float("nan")) == "NaN"


def test_canonical_json_documents_negative_zero_as_distinct_from_positive_zero() -> None:
    """-0.0 policy (documented on ``_canonical_json``): negative zero is
    *not* normalized. ``json.dumps`` already renders it deterministically as
    the literal ``-0.0``, distinct from ``0.0`` — two payloads differing
    only in float sign-of-zero hash differently, on purpose.
    """
    assert cci._canonical_json(-0.0) == "-0.0"
    assert cci._canonical_json(0.0) == "0.0"
    assert cci._sha256_json({"x": -0.0}) != cci._sha256_json({"x": 0.0})
    # Deterministic across repeated calls (not a one-off artifact of dict
    # ordering or process state).
    assert cci._canonical_json(-0.0) == cci._canonical_json(-0.0)


def test_canonical_json_wraps_json_dumps_typeerror_for_a_leaf_type_the_walk_does_not_reject() -> (
    None
):
    """Defense in depth: a leaf type the recursive walk does not itself
    special-case (only dict keys and float values are inspected) still
    reaches ``json.dumps``, whose raw ``TypeError`` is wrapped rather than
    leaked.
    """
    with pytest.raises(
        ClusterFeatureContractError,
        match=r"^canonical JSON payload contains a value json\.dumps cannot serialize: ",
    ):
        cci._canonical_json({"a": {1, 2, 3}})


# --------------------------------------------------------------------------
# StableFileDigest.__post_init__ validation (raise sites 1-2).
# --------------------------------------------------------------------------


def test_stable_file_digest_accepts_valid_fields(tmp_path: Path) -> None:
    digest = StableFileDigest(
        path=tmp_path / "f.pt",
        sha256="a" * 64,
        byte_length=0,
        stat_signature=(1, 2, 3, 4, 5, 6),
    )
    assert digest.byte_length == 0
    assert digest.sha256 == "a" * 64


def test_stable_file_digest_rejects_non_hex_sha256(tmp_path: Path) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match="^file SHA-256 must be lowercase hexadecimal$",
    ):
        StableFileDigest(
            path=tmp_path / "f.pt",
            sha256="not-a-hash",
            byte_length=0,
            stat_signature=(1, 2, 3, 4, 5, 6),
        )


def test_stable_file_digest_rejects_negative_byte_length(tmp_path: Path) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match="^file byte length must be non-negative$",
    ):
        StableFileDigest(
            path=tmp_path / "f.pt",
            sha256="a" * 64,
            byte_length=-1,
            stat_signature=(1, 2, 3, 4, 5, 6),
        )


# --------------------------------------------------------------------------
# StableFileDigest.__post_init__ — Codex fix batch finding 3 (strict domain
# validation the oracle never had) + finding 1 ripple (stat_signature grew
# from a 5-tuple to a 6-tuple to include st_mode).
# --------------------------------------------------------------------------


def test_stable_file_digest_rejects_a_bare_string_path(tmp_path: Path) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match="^file digest path must be a pathlib.Path$",
    ):
        StableFileDigest(
            path=cast(Any, str(tmp_path / "f.pt")),
            sha256="a" * 64,
            byte_length=0,
            stat_signature=(1, 2, 3, 4, 5, 6),
        )


@pytest.mark.parametrize(
    "byte_length",
    [True, False, 1.5],
    ids=["bool-true", "bool-false", "float"],
)
def test_stable_file_digest_rejects_non_strict_int_byte_length(
    tmp_path: Path, byte_length: Any
) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match="^file byte length must be non-negative$",
    ):
        StableFileDigest(
            path=tmp_path / "f.pt",
            sha256="a" * 64,
            byte_length=byte_length,
            stat_signature=(1, 2, 3, 4, 5, 6),
        )


@pytest.mark.parametrize(
    "stat_signature",
    [
        (1, 2, 3, 4, 5),  # arity 5 — the pre-fix shape, now rejected
        (1, 2, 3, 4, 5, 6, 7),  # arity 7
        (),
        (1, 2, 3, 4, 5, "6"),  # one non-int field
        (1, 2, 3, 4, 5, True),  # bool-as-int field
        [1, 2, 3, 4, 5, 6],  # a list, not a tuple
    ],
    ids=["arity-5", "arity-7", "empty", "non-int-field", "bool-field", "not-a-tuple"],
)
def test_stable_file_digest_rejects_malformed_stat_signature(
    tmp_path: Path, stat_signature: Any
) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match=r"^file stat signature must be a 6-tuple of strict ints",
    ):
        StableFileDigest(
            path=tmp_path / "f.pt",
            sha256="a" * 64,
            byte_length=0,
            stat_signature=stat_signature,
        )


# --------------------------------------------------------------------------
# _stat_signature.
# --------------------------------------------------------------------------


def test_stat_signature_reports_mode_first_and_size_fourth(tmp_path: Path) -> None:
    """Codex finding 1: the stat signature grew by st_mode (now the first
    field) to make a symlink substitution at ``path`` detectable via the
    mode bits, not just size/time.
    """
    target = tmp_path / "f.bin"
    target.write_bytes(b"0123456789")
    signature = cci._stat_signature(target)
    assert len(signature) == 6
    assert all(type(field) is int for field in signature)
    assert stat.S_ISREG(signature[0])
    assert signature[3] == 10


def test_stat_signature_uses_lstat_and_detects_a_symlink_at_the_path(tmp_path: Path) -> None:
    """Using ``lstat`` (not ``stat``) means a symlink substituted at a
    previously-hashed path is itself detected via the mode field, instead of
    being transparently followed to its target.
    """
    target = tmp_path / "f.bin"
    target.write_bytes(b"0123456789")
    real_signature = cci._stat_signature(target)

    link = tmp_path / "link.bin"
    other = tmp_path / "other.bin"
    other.write_bytes(b"0123456789")  # same size as target, different inode
    link.symlink_to(other)

    link_signature = cci._stat_signature(link)
    assert not stat.S_ISREG(link_signature[0])
    assert link_signature != real_signature


# --------------------------------------------------------------------------
# stream_hash_file_stable: happy path + digest stability + raise sites 3-6.
# --------------------------------------------------------------------------


def test_stream_hash_file_stable_matches_manual_sha256_and_captures_stat(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.pt"
    payload = b"x" * 1000
    target.write_bytes(payload)

    digest = stream_hash_file_stable(target)

    assert digest.sha256 == hashlib.sha256(payload).hexdigest()
    assert digest.byte_length == len(payload)
    assert digest.path == target.resolve()
    assert digest.stat_signature == cci._stat_signature(target.resolve())


def test_stream_hash_file_stable_is_content_identical_for_a_moved_file(
    tmp_path: Path,
) -> None:
    """Digest stability across a move: same bytes at a different path hash equal."""
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    payload = b"identical payload bytes"
    first = first_dir / "model.pt"
    second = second_dir / "model.pt"
    first.write_bytes(payload)
    second.write_bytes(payload)

    first_digest = stream_hash_file_stable(first)
    second_digest = stream_hash_file_stable(second)

    assert first_digest.sha256 == second_digest.sha256
    assert first_digest.byte_length == second_digest.byte_length == len(payload)
    assert first_digest.path != second_digest.path


def test_stream_hash_file_stable_differs_for_changed_bytes(tmp_path: Path) -> None:
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    first.write_bytes(b"payload-one")
    second.write_bytes(b"payload-two-longer")

    assert stream_hash_file_stable(first).sha256 != stream_hash_file_stable(second).sha256


def test_stream_hash_file_stable_raises_file_not_found_for_a_missing_path(
    tmp_path: Path,
) -> None:
    """``resolve(strict=True)`` passes a raw FileNotFoundError through unwrapped."""
    missing = tmp_path / "does-not-exist.pt"
    with pytest.raises(FileNotFoundError):
        stream_hash_file_stable(missing)


def test_stream_hash_file_stable_rejects_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "a_directory"
    directory.mkdir()
    resolved = directory.resolve()
    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input is not a regular file: {re.escape(str(resolved))}$",
    ):
        stream_hash_file_stable(directory)


def test_stream_hash_file_stable_wraps_os_error_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    resolved = target.resolve()
    real_open = Path.open

    def faulty_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved:
            raise OSError("simulated read failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", faulty_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^could not hash contract input {re.escape(str(resolved))}$",
    ):
        stream_hash_file_stable(target)


def test_stream_hash_file_stable_detects_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a changed post-read descriptor signature with the exact public error.

    Descriptor-bound hashing observes pre-open lstat, post-open fstat, post-read
    fstat, and final lstat. Perturb the third observation so the post-read fstat
    differs from its post-open baseline, simulating mutation during the read.
    """
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    resolved = target.resolve()
    real_signature = immutable_io._fd_bound_signature
    calls = 0

    def flaky_signature(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
        nonlocal calls
        calls += 1
        signature = real_signature(result)
        if calls == 3:
            # Perturb only the post-read fstat observation — simulates a
            # concurrent writer racing the hash loop through the same fd.
            return (
                signature[0],
                signature[1],
                signature[2],
                signature[3],
                signature[4] + 1,
                signature[5],
            )
        return signature

    monkeypatch.setattr(immutable_io, "_fd_bound_signature", flaky_signature)

    with pytest.raises(
        ClusterFeatureContractError,
        match=(f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$"),
    ):
        stream_hash_file_stable(target)
    assert calls == 3


# --------------------------------------------------------------------------
# stream_hash_file_stable — Codex fix batch finding 5 blind spots, exercised
# through the *full* wired stack (cluster_contract_identity ->
# immutable_io), proving the fix is carried through the delegation, not just
# present in the low-level helper (which has its own, more exhaustive
# coverage in test_immutable_io.py).
# --------------------------------------------------------------------------


def test_stream_hash_file_stable_hashes_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.pt"
    target.write_bytes(b"")
    digest = stream_hash_file_stable(target)
    assert digest.sha256 == hashlib.sha256(b"").hexdigest()
    assert digest.byte_length == 0


def test_stream_hash_file_stable_detects_a_real_equal_size_swap_around_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex finding 1, headline proof, through the full public entry point:
    an equal-size A->B swap staged immediately before ``open()`` and
    restored to fresh A-content immediately after is now detected — this is
    the exact race described in the finding ("an equal-size A->B swap before
    open() followed by B->A restoration...returns sha(B) with A's
    signature") and is the test that *fails* under the oracle's pathname-
    only before/after ``stat()`` pair (see
    ``test_immutable_io.py::test_the_same_swap_is_invisible_to_the_oracles_pathname_only_before_after_check``
    for an explicit empirical comparison against a reimplementation of that
    pre-fix algorithm).
    """
    target = tmp_path / "model.pt"
    a_bytes = b"A" * 4096
    b_bytes = b"B" * 4096
    target.write_bytes(a_bytes)
    b_source = tmp_path / ".staged_b"
    b_source.write_bytes(b_bytes)
    a_restore_source = tmp_path / ".staged_a_restore"
    a_restore_source.write_bytes(a_bytes)
    resolved = target.resolve()

    real_open = Path.open
    swapped = {"done": False}

    def swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            os.replace(b_source, resolved)
            handle = real_open(self, *args, **kwargs)
            os.replace(a_restore_source, resolved)
            return handle
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$",
    ):
        stream_hash_file_stable(target)
    assert swapped["done"]


def test_stream_hash_file_stable_detects_a_symlink_substituted_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symlink-target swap: the path is a regular file when first observed
    (lstat), but gets replaced by a symlink immediately before ``open()`` —
    detected because the descriptor-bound fstat after open reports the
    symlink target's identity, which mismatches the original regular file's.
    """
    target = tmp_path / "model.pt"
    target.write_bytes(b"original content")
    link_target = tmp_path / "elsewhere.pt"
    link_target.write_bytes(b"different content, same-ish size")
    resolved = target.resolve()

    real_open = Path.open
    swapped = {"done": False}

    def symlink_swapping_open(self: Path, *args: Any, **kwargs: Any):
        if self == resolved and not swapped["done"]:
            swapped["done"] = True
            resolved.unlink()
            resolved.symlink_to(link_target)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", symlink_swapping_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$",
    ):
        stream_hash_file_stable(target)
    assert swapped["done"]


def test_stream_hash_file_stable_detects_a_rename_immediately_after_the_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rename-mid-read: the descriptor stays valid and reads the original
    bytes correctly (POSIX unlink/rename semantics don't affect an already-
    open fd), but the final lstat at the original path — now empty — is
    what surfaces the tamper.
    """
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    renamed = tmp_path / "renamed-away.pt"
    resolved = target.resolve()

    real_open = Path.open

    class _RenamingHandle:
        def __init__(self, handle: Any) -> None:
            self._handle = handle
            self._renamed = False

        def read(self, *args: Any, **kwargs: Any) -> bytes:
            chunk = self._handle.read(*args, **kwargs)
            if not chunk and not self._renamed:
                self._renamed = True
                os.replace(resolved, renamed)
            return chunk

        def fileno(self) -> int:
            return self._handle.fileno()

        def __enter__(self) -> _RenamingHandle:
            return self

        def __exit__(self, *exc_info: Any) -> None:
            self._handle.__exit__(*exc_info)

    def renaming_open(self: Path, *args: Any, **kwargs: Any):
        handle = real_open(self, *args, **kwargs)
        if self == resolved:
            return _RenamingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", renaming_open)

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^contract input changed while it was being hashed: {re.escape(str(resolved))}$",
    ):
        stream_hash_file_stable(target)
    assert renamed.read_bytes() == b"payload"


# --------------------------------------------------------------------------
# _assert_digest_stat_unchanged: happy path + raise sites 7-8.
# --------------------------------------------------------------------------


def test_assert_digest_stat_unchanged_accepts_an_untouched_file(tmp_path: Path) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    digest = stream_hash_file_stable(target)
    assert cci._assert_digest_stat_unchanged(digest) is None


def test_assert_digest_stat_unchanged_raises_when_the_file_disappears(
    tmp_path: Path,
) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    digest = stream_hash_file_stable(target)
    target.unlink()

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^pinned contract input disappeared: {re.escape(str(digest.path))}$",
    ):
        cci._assert_digest_stat_unchanged(digest)


def test_assert_digest_stat_unchanged_raises_when_the_file_mutates(tmp_path: Path) -> None:
    target = tmp_path / "model.pt"
    target.write_bytes(b"payload")
    digest = stream_hash_file_stable(target)
    target.write_bytes(b"payload-plus-more-bytes-changes-size-and-mtime")

    with pytest.raises(
        ClusterFeatureContractError,
        match=f"^pinned contract input changed after hashing: {re.escape(str(digest.path))}$",
    ):
        cci._assert_digest_stat_unchanged(digest)


# --------------------------------------------------------------------------
# FeatureDirectoryIdentity.
# --------------------------------------------------------------------------


def test_feature_directory_identity_from_components_maps_enum_values_to_strings() -> None:
    components = FeatureDirectoryComponents(
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        context_independent_features=False,
        ignore_bos_features=True,
    )

    identity = FeatureDirectoryIdentity.from_components(components)

    assert identity == FeatureDirectoryIdentity(
        segment_size=128,
        layer=8,
        activation_type="residual",
        pooling="max_pool",
        width="16k",
        model="gemma-2-2b",
        dataset_phase="base",
        context_independent_features=False,
        ignore_bos_features=True,
    )
    assert identity.canonical_dict() == {
        "segment_size": 128,
        "layer": 8,
        "activation_type": "residual",
        "pooling": "max_pool",
        "width": "16k",
        "model": "gemma-2-2b",
        "dataset_phase": "base",
        "context_independent_features": False,
        "ignore_bos_features": True,
    }


def test_feature_directory_identities_differ_only_by_segment_mode() -> None:
    fixed_components = FeatureDirectoryComponents(
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        context_independent_features=False,
        ignore_bos_features=True,
    )
    turn_components = replace(
        fixed_components,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    assert FeatureDirectoryIdentity.from_components(
        fixed_components
    ) != FeatureDirectoryIdentity.from_components(turn_components)


def test_turn_feature_directory_identity_canonical_dict_carries_mode() -> None:
    components = FeatureDirectoryComponents(
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        context_independent_features=False,
        ignore_bos_features=True,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens,
    )
    assert FeatureDirectoryIdentity.from_components(components).canonical_dict()[
        "segment_mode"
    ] == SegmentMode.turn_delimited_fixed_tokens.value


# --------------------------------------------------------------------------
# FeatureStatisticsIdentity + relational feature-ID tuple validation.
# --------------------------------------------------------------------------


def test_feature_statistics_identity_accepts_valid_fields() -> None:
    identity = FeatureStatisticsIdentity(
        file_sha256="a" * 64,
        percentile_table_checksum="b" * 64,
        n_sae_features=8,
        feature_id_domain=(1, 2, 5),
    )
    assert identity.canonical_dict() == {
        "file_sha256": "a" * 64,
        "percentile_table_checksum": "b" * 64,
        "n_sae_features": 8,
        "feature_id_domain": [1, 2, 5],
    }


def test_feature_statistics_identity_delegates_domain_validation_at_construction() -> None:
    """Wiring check: construction reaches ``_validate_feature_id_tuple`` (raise site 11)."""
    with pytest.raises(
        ClusterFeatureContractError,
        match="^statistics feature-ID domain must be a sorted unique tuple$",
    ):
        FeatureStatisticsIdentity(
            file_sha256="a" * 64,
            percentile_table_checksum="b" * 64,
            n_sae_features=8,
            feature_id_domain=(3, 1, 2),
        )


@pytest.mark.parametrize(
    "values",
    [(2, 1), (1, 1, 2)],
    ids=["unsorted", "duplicate"],
)
def test_validate_feature_id_tuple_rejects_non_sorted_unique_input(
    values: tuple[int, ...],
) -> None:
    with pytest.raises(
        ClusterFeatureContractError,
        match="^x must be a sorted unique tuple$",
    ):
        cci._validate_feature_id_tuple(values, n_sae_features=8, field_name="x")


def test_validate_feature_id_tuple_rejects_id_at_width() -> None:
    with pytest.raises(ClusterFeatureContractError, match=r"^x contains invalid ID 8 for width 8$"):
        cci._validate_feature_id_tuple((0, 8), n_sae_features=8, field_name="x")


def test_sae_identity_round_trips_through_canonical_dict() -> None:
    identity = SaeIdentity(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_8/width_16k/canonical",
        hook_name="blocks.8.hook_resid_post",
        model="gemma-2-2b",
        layer=8,
        activation_type="residual",
        width="16k",
        actual_loaded_width=16384,
    )
    assert identity.canonical_dict() == {
        "release": "gemma-scope-2b-pt-res",
        "sae_id": "layer_8/width_16k/canonical",
        "hook_name": "blocks.8.hook_resid_post",
        "model": "gemma-2-2b",
        "layer": 8,
        "activation_type": "residual",
        "width": "16k",
        "actual_loaded_width": 16384,
    }
