from __future__ import annotations

import dataclasses
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

import steering.immutable_io as immutable_io
from steering.eval.axbench_bpb_adv import source_validation as sv
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.cluster_feature_contract import (
    StableFileDigest,
    stat_signature_now,
    stream_hash_file_stable,
)

_FINGERPRINT = "a" * 64


def _representative_document() -> dict[str, Any]:
    return {
        "semantic": {
            # G2-D6 requires the SAE width so member IDs can be domain-bounded.
            "n_sae_features": 8,
            "feature_clusters": {
                "representative_universe": [1, 3],
                "representative_to_allowed_members": [
                    [1, [1, 2]],
                    [3, [3, 4]],
                ],
            }
        }
    }


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"owner": "string-value"}, "string-value"),
        ({b"owner": b"bytes-value"}, "bytes-value"),
        ({}, None),
        (None, None),
    ],
)
def test_metadata_value_accepts_both_metadata_key_encodings(
    metadata: Mapping[str, str] | Mapping[bytes, bytes] | None,
    expected: str | None,
) -> None:
    assert sv._metadata_value(metadata, "owner") == expected


def test_metadata_value_rejects_non_utf8_bytes() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="is not UTF-8"):
        sv._metadata_value({b"owner": b"\xff"}, "owner")


def test_safetensors_metadata_reads_valid_metadata(tmp_path: Path) -> None:
    path = tmp_path / "artifact.safetensors"
    save_file(
        {"value": t.tensor([1], dtype=t.int64)},
        str(path),
        metadata={"owner": "cluster-a"},
    )

    assert sv._safetensors_metadata(path, required=True) == {"owner": "cluster-a"}


def test_safetensors_metadata_optional_and_required_invalid_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not-safetensors.bin"
    path.write_bytes(b"plain bytes")

    assert sv._safetensors_metadata(path, required=False) is None
    with pytest.raises(AxBenchBpbAdvError, match="not a valid safetensors file"):
        sv._safetensors_metadata(path, required=True)


def test_require_artifact_fingerprint_accepts_exact_match() -> None:
    assert (
        sv._require_artifact_fingerprint(
            surface="Stage 2",
            metadata={"feature_contract_fingerprint": _FINGERPRINT},
            expected=_FINGERPRINT,
        )
        == _FINGERPRINT
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"feature_contract_fingerprint": "b" * 64},
        {},
        None,
    ],
)
def test_require_artifact_fingerprint_rejects_mismatch_or_absence(
    metadata: Mapping[str, str] | None,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="fingerprint mismatch"):
        sv._require_artifact_fingerprint(
            surface="Stage 2",
            metadata=metadata,
            expected=_FINGERPRINT,
        )


def test_representative_universe_accepts_sorted_unique_nonnegative_ids() -> None:
    assert sv._representative_universe_from_document(_representative_document()) == frozenset(
        {1, 3}
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not-a-list",
        [-1],
        [3, 1],
        [1, 1],
        [],
        # F1: wrong CONTAINER type. "not-a-list" above is iterable, so it trips
        # the NEXT condition in the or-chain rather than the isinstance guard --
        # these prove the guard itself is load-bearing.
        None,
        42,
        True,
        {},
        {"1": 1},
    ],
)
def test_representative_universe_rejects_each_malformed_shape(raw: object) -> None:
    document = _representative_document()
    document["semantic"]["feature_clusters"]["representative_universe"] = raw

    with pytest.raises(AxBenchBpbAdvError, match="malformed representative universe"):
        sv._representative_universe_from_document(document)


def test_representative_members_accept_canonical_rows() -> None:
    assert sv._representative_members_from_document(_representative_document()) == {
        1: (1, 2),
        3: (3, 4),
    }


def test_representative_members_rejects_missing_list() -> None:
    document = _representative_document()
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = None

    with pytest.raises(AxBenchBpbAdvError, match="missing representative members"):
        sv._representative_members_from_document(document)


@pytest.mark.parametrize(
    "raw",
    [
        [[1]],
        [["1", [1]]],
        [[1, [1, "2"]]],
        # F1: wrong CONTAINER type at an outer level. The leaf-value cases above
        # cannot prove the isinstance guards are load-bearing, because a string
        # is iterable and accidentally trips the next condition in the or-chain.
        [None],
        [42],
        [True],
        [{}],
        [[1, None]],
        [[1, 42]],
        [[1, "12"]],
    ],
)
def test_representative_members_rejects_malformed_rows(raw: object) -> None:
    document = _representative_document()
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = raw

    with pytest.raises(AxBenchBpbAdvError, match="malformed representative members"):
        sv._representative_members_from_document(document)


@pytest.mark.parametrize(
    "raw",
    [
        [[1, []]],
        [[1, [1, 3, 2]]],
        [[1, [2]]],
        [[1, [1]], [1, [1, 2]]],
    ],
)
def test_representative_members_rejects_inconsistent_rows(raw: object) -> None:
    document = _representative_document()
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = raw

    with pytest.raises(AxBenchBpbAdvError, match="inconsistent representative members"):
        sv._representative_members_from_document(document)


def test_require_representative_concepts_accepts_only_universe_members() -> None:
    sv._require_representative_concepts(
        rows=[{"feature_index": 1}, {"feature_index": 3}],
        representative_universe=frozenset({1, 3}),
        surface="concepts.json",
    )


@pytest.mark.parametrize(
    "row",
    [
        {"feature_index": 2},
        {"feature_index": True},
        {"wrong_key": 1},
        None,
    ],
)
def test_require_representative_concepts_rejects_nonrepresentatives(
    row: Any,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="non-representative feature IDs"):
        sv._require_representative_concepts(
            rows=[row],
            representative_universe=frozenset({1, 3}),
            surface="concepts.json",
        )


def test_snapshot_rejects_top_level_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    link = tmp_path / "artifact.bin"
    link.symlink_to(target)

    with pytest.raises(AxBenchBpbAdvError, match="must not be symbolic links"):
        sv._snapshot_artifact("concepts.json", link)


def test_snapshot_rejects_symlink_inside_directory(tmp_path: Path) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    (root / "linked.bin").symlink_to(target)

    with pytest.raises(AxBenchBpbAdvError, match="must not contain symlinks"):
        sv._snapshot_artifact("baselines/", root)


def test_snapshot_rejects_top_level_nonregular_file(tmp_path: Path) -> None:
    fifo = tmp_path / "artifact.fifo"
    os.mkfifo(fifo)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="is not a regular file or directory",
    ):
        sv._snapshot_artifact("concepts.json", fifo)


@pytest.mark.parametrize(
    "name",
    [
        "concepts.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
    ],
)
def test_snapshot_retains_exact_semantic_source_payload(
    tmp_path: Path,
    name: str,
) -> None:
    path = tmp_path / name
    payload = f"payload-for-{name}".encode()
    path.write_bytes(payload)

    snapshot = sv._snapshot_artifact(name, path)

    assert snapshot.payload == payload
    assert snapshot.byte_size == len(payload)
    assert snapshot.sha256 == snapshot.contract_sha256


def test_snapshot_does_not_retain_unrelated_or_large_source_payloads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seeds_train.safetensors"
    save_file({"seed": t.tensor([1], dtype=t.int64)}, str(path))

    snapshot = sv._snapshot_artifact(path.name, path)

    assert snapshot.payload is None


def test_required_snapshot_payload_rejects_absent_and_unretained_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seed_pool_manifest.json"
    path.write_text('{"files": {}}', encoding="utf-8")
    snapshot = sv._snapshot_artifact(path.name, path)

    with pytest.raises(AxBenchBpbAdvError, match="missing retained payload concepts.json"):
        sv._required_snapshot_payload({}, "concepts.json")
    with pytest.raises(
        AxBenchBpbAdvError,
        match="missing retained payload seed_pool_manifest.json",
    ):
        sv._required_snapshot_payload({path.name: snapshot}, path.name)


@pytest.mark.parametrize("kind", ["fifo", "socket"])
def test_snapshot_rejects_nonregular_directory_child(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    child = root / f"artifact.{kind}"
    if kind == "fifo":
        os.mkfifo(child)
        with pytest.raises(AxBenchBpbAdvError, match="only regular files"):
            sv._snapshot_artifact("baselines/", root)
        return

    os.mknod(child, stat.S_IFSOCK | 0o600)
    assert stat.S_ISSOCK(child.lstat().st_mode)
    with pytest.raises(AxBenchBpbAdvError, match="only regular files"):
        sv._snapshot_artifact("baselines/", root)


@pytest.mark.parametrize("nested", [False, True])
def test_snapshot_rejects_file_mutated_during_stable_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    target = tmp_path / "baselines" / "artifact.bin" if nested else tmp_path / "artifact.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"before")
    real_fstat = immutable_io.os.fstat
    n_fstats = 0

    def mutate_before_second_fstat(fd: int) -> os.stat_result:
        nonlocal n_fstats
        n_fstats += 1
        if n_fstats == 2:
            target.write_bytes(b"mutated-during-hash")
        return real_fstat(fd)

    monkeypatch.setattr(immutable_io.os, "fstat", mutate_before_second_fstat)
    path = target.parent if nested else target
    name = "baselines/" if nested else "concepts.json"

    with pytest.raises(AxBenchBpbAdvError, match="could not stably snapshot"):
        sv._snapshot_artifact(name, path)
    assert n_fstats == 2


def test_snapshot_rejects_file_changed_between_raw_and_contract_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "concepts.json"
    path.write_bytes(b"before")
    real_stream_hash = sv.stream_hash_file_stable

    def hash_then_mutate(candidate: str | Path) -> StableFileDigest:
        digest = real_stream_hash(candidate)
        digest.path.write_bytes(b"changed-after-raw-hash")
        return digest

    monkeypatch.setattr(sv, "stream_hash_file_stable", hash_then_mutate)

    with pytest.raises(AxBenchBpbAdvError, match="artifact changed while"):
        sv._snapshot_artifact("concepts.json", path)


def test_snapshot_rejects_directory_inventory_change_after_retained_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    (root / "existing.bin").write_bytes(b"existing")
    real_stream_hash = sv.stream_hash_file_descriptor_bound

    def retain_then_add_child(
        path: Path,
        **kwargs: Any,
    ) -> immutable_io.StreamedFileDigest:
        digest = real_stream_hash(path, **kwargs)
        (root / "added.bin").write_bytes(b"added")
        return digest

    monkeypatch.setattr(
        sv,
        "stream_hash_file_descriptor_bound",
        retain_then_add_child,
    )

    with pytest.raises(AxBenchBpbAdvError, match="directory changed while"):
        sv._snapshot_artifact("baselines/", root)


def test_snapshot_rejects_directory_file_change_after_retained_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    child = root / "artifact.bin"
    child.write_bytes(b"before")
    real_stream_hash = sv.stream_hash_file_descriptor_bound

    def retain_then_mutate(
        path: Path,
        **kwargs: Any,
    ) -> immutable_io.StreamedFileDigest:
        digest = real_stream_hash(path, **kwargs)
        child.write_bytes(b"changed-after-retained-read")
        return digest

    monkeypatch.setattr(
        sv,
        "stream_hash_file_descriptor_bound",
        retain_then_mutate,
    )

    with pytest.raises(AxBenchBpbAdvError, match="directory file changed while"):
        sv._snapshot_artifact("baselines/", root)


def test_snapshot_brands_directory_recheck_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"content")
    real_signature = sv.stat_signature_now
    n_signatures = 0

    def unavailable_signature(path: Path) -> tuple[int, int, int, int, int, int]:
        nonlocal n_signatures
        n_signatures += 1
        if n_signatures == 3:
            raise FileNotFoundError("deterministic recheck failure")
        return real_signature(path)

    monkeypatch.setattr(sv, "stat_signature_now", unavailable_signature)

    with pytest.raises(AxBenchBpbAdvError, match="directory changed while"):
        sv._snapshot_artifact("baselines/", root)
    assert n_signatures == 3


def test_snapshot_brands_initial_directory_stat_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    monkeypatch.setattr(
        sv,
        "stat_signature_now",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("root vanished")),
    )

    with pytest.raises(AxBenchBpbAdvError, match="stably snapshot clustered source directory"):
        sv._snapshot_artifact("baselines/", root)


def test_snapshot_brands_initial_child_stat_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    child = root / "artifact.bin"
    child.write_bytes(b"content")
    real_signature = sv.stat_signature_now

    def _signature(path: Path) -> tuple[int, int, int, int, int, int]:
        if path == child:
            raise FileNotFoundError("child vanished")
        return real_signature(path)

    monkeypatch.setattr(sv, "stat_signature_now", _signature)
    with pytest.raises(AxBenchBpbAdvError, match="stably snapshot clustered source artifact"):
        sv._snapshot_artifact("baselines/", root)


def test_snapshot_rejects_child_changed_before_descriptor_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"content")
    real_stream = sv.stream_hash_file_descriptor_bound

    def _mismatched_digest(path: Path, **kwargs: Any) -> Any:
        digest = real_stream(path, **kwargs)
        return dataclasses.replace(
            digest,
            stat_signature=(0, 0, 0, 0, 0, 0),
        )

    monkeypatch.setattr(
        sv,
        "stream_hash_file_descriptor_bound",
        _mismatched_digest,
    )
    with pytest.raises(AxBenchBpbAdvError, match="changed before it was retained"):
        sv._snapshot_artifact("baselines/", root)


def test_d2_oracle_five_tuple_false_positive_and_shipped_green(
    tmp_path: Path,
) -> None:
    root = tmp_path / "baselines"
    root.mkdir()
    child = root / "artifact.bin"
    child.write_bytes(b"unchanged")
    digest = stream_hash_file_stable(child)
    signatures = (("artifact.bin", digest.stat_signature),)

    def oracle_five_field_signature(path: Path) -> tuple[int, int, int, int, int]:
        result = path.stat()
        return (
            int(result.st_dev),
            int(result.st_ino),
            int(result.st_size),
            int(result.st_mtime_ns),
            int(result.st_ctime_ns),
        )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="directory file changed while it was being hashed",
    ):
        changed = [
            relative
            for relative, signature in signatures
            if oracle_five_field_signature(root / relative) != signature
        ]
        if changed:
            raise AxBenchBpbAdvError(
                "clustered source directory file changed while it was being hashed: "
                f"{root / changed[0]}"
            )

    snapshot = sv._snapshot_artifact("baselines/", root)
    assert snapshot.byte_size == len(b"unchanged")
    assert snapshot.entry_count == 1


def test_d2_public_stat_signature_arity_matches_stable_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"content")

    assert len(stat_signature_now(path)) == len(stream_hash_file_stable(path).stat_signature)


def test_d3_file_contract_digest_matches_unclustered_hash_route(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seeds_train.safetensors"
    save_file({"seed": t.tensor([1, 2], dtype=t.int64)}, str(path))

    snapshot = sv._snapshot_artifact("seeds_train.safetensors", path)

    assert snapshot.sha256 == stream_hash_file_stable(path).sha256
    assert snapshot.contract_sha256 == sv._hash_required_file(
        "seeds_train.safetensors",
        path,
    )
    assert snapshot.contract_sha256 != snapshot.sha256


def test_d3_directory_contract_digest_preserves_unclustered_path_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "baselines"
    (root / "a").mkdir(parents=True)
    (root / "a" / "b").write_bytes(b"nested")
    (root / "a-c").write_bytes(b"sibling")

    path_order = [child.relative_to(root).as_posix() for child in sorted(root.rglob("*"))]
    snapshot_order = [
        child.relative_to(root).as_posix()
        for child in sorted(
            root.rglob("*"),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
    ]
    snapshot = sv._snapshot_artifact("baselines/", root)

    assert path_order == ["a", "a/b", "a-c"]
    assert snapshot_order == ["a", "a-c", "a/b"]
    assert snapshot.contract_sha256 == sv._hash_directory(root)
    assert snapshot.contract_sha256 == snapshot.sha256


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        # Member outside the SAE domain (n_sae_features=8).
        ([[1, [1, 999]], [3, [3]]], "inconsistent representative members"),
        # Negative member id.
        ([[1, [1]], [3, [3, -1]]], "inconsistent representative members"),
        # Key that is not a representative, and a representative left out.
        ([[1, [1]], [5, [5]]], "do not match the representative universe"),
        # Every representative present but one extra key.
        ([[1, [1]], [3, [3]], [7, [7]]], "do not match the representative universe"),
        # Empty mapping.
        ([], "empty representative-member mapping"),
    ],
)
def test_d6_representative_members_are_domain_and_universe_bounded(
    raw: object, match: str
) -> None:
    """G2-D6: the oracle bounded member rows by shape only.

    It checked strict-int shape, ordering, uniqueness and representative-first,
    but never bounded members by the SAE width, never required the mapping's
    keys to BE the representative universe, and accepted an empty mapping. So a
    self-consistently stamped ``[[1, [1, 999]], [5, [5]]]`` with universe
    ``{1, 3}`` and ``n_sae_features=8`` was admissible -- 999 is outside the SAE
    domain, 5 is not a representative, and 3 is missing entirely. Downstream
    candidate-provenance catches missing/extra representatives, but its own ID
    helper checks only non-negativity, so the out-of-domain member survives it.
    """

    document = _representative_document()
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = raw

    with pytest.raises(AxBenchBpbAdvError, match=match):
        sv._representative_members_from_document(document)


@pytest.mark.parametrize("width", [None, 0, -1, "8", True, 8.0])
def test_d6_representative_members_require_a_valid_sae_width(width: object) -> None:
    """The domain bound is only meaningful if the width itself is trustworthy."""

    document = _representative_document()
    document["semantic"]["n_sae_features"] = width

    with pytest.raises(AxBenchBpbAdvError, match="malformed n_sae_features"):
        sv._representative_members_from_document(document)


def test_d5_snapshot_rechecks_the_declared_path_not_the_resolved_target(
    tmp_path: Path,
) -> None:
    """G2-D5: a symlink substituted at the declared path must not be followed.

    ``stream_hash_file_stable`` canonicalizes via ``resolve(strict=True)``, so
    ``digest.path`` is the RESOLVED target. Rechecking that instead of the
    declared pathname followed a symlink swapped in after the up-front guard and
    never looked at the declared path again. ``stat_signature_now`` uses lstat,
    so rechecking the original surfaces the substitution through its mode field.

    Modelled deterministically: the up-front guard is neutralised for exactly
    one call, so the substituted symlink reaches the hash-and-recheck body --
    the same state a real post-guard swap would produce.
    """

    real = tmp_path / "real.bin"
    real.write_bytes(b"payload")
    declared = tmp_path / "concepts.json"
    declared.symlink_to(real)

    calls = {"n": 0}
    original_is_symlink = Path.is_symlink

    def only_first_call_reports_regular(self: Path) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return original_is_symlink(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "is_symlink", only_first_call_reports_regular)
        with pytest.raises(AxBenchBpbAdvError, match="changed while it was being hashed"):
            sv._snapshot_artifact("concepts.json", declared)


def test_d5_snapshot_still_accepts_an_honest_regular_file(tmp_path: Path) -> None:
    """Guard against over-correction: the D5 change must not reject valid input.

    For an honest regular file, lstat(declared) and the fd-bound stat of the
    resolved target describe the same inode, so the recheck is a no-op.
    """

    path = tmp_path / "concepts.json"
    path.write_bytes(b"{}")

    snapshot = sv._snapshot_artifact("concepts.json", path)

    assert snapshot.byte_size == 2
    assert snapshot.contract_sha256 == sv._hash_required_file("concepts.json", path)


@pytest.mark.parametrize("bad", [None, 42, True, "semantic", []])
def test_f1_non_mapping_semantic_or_clusters_fails_loud(bad: Any) -> None:
    """F1: the two `isinstance(..., Mapping)` ternary guards must be load-bearing.

    Deleting either guard survived the whole suite, because every existing
    fixture supplies a well-typed Mapping at both levels. Without them a
    non-Mapping document degrades from a clean AxBenchBpbAdvError to an
    unhandled AttributeError/TypeError.
    """

    for document in ({"semantic": bad}, {"semantic": {"feature_clusters": bad}}):
        with pytest.raises(AxBenchBpbAdvError, match="malformed representative universe"):
            sv._representative_universe_from_document(document)


def test_f2_require_fingerprint_absent_raises_without_relying_on_assert() -> None:
    """G2-D7: the guard must hold under `python -O`, where asserts are stripped.

    The oracle used a bare `assert actual is not None`, so a caller passing
    `expected=None` got a silent `None` back from a function typed `-> str` and
    named "require". Now an explicit raise.
    """

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint is absent"):
        sv._require_artifact_fingerprint(
            surface="Stage 2", metadata=None, expected=cast(Any, None)
        )


@pytest.mark.parametrize(
    ("member_id", "accepted"),
    [(6, True), (7, True), (8, False), (9, False)],
)
def test_d6_width_bound_is_exclusive_at_the_boundary(
    member_id: int, accepted: bool
) -> None:
    """Pin the EXACT boundary; an off-by-one here survived the whole suite.

    Weakening ``value >= n_sae_features`` to ``value > n_sae_features`` passed
    all 1476 tests, because no case exercised a member id exactly equal to the
    width. With n_sae_features=8 the valid domain is [0, 8), so 7 is the last
    legal id and 8 must be rejected.
    """

    document = _representative_document()
    document["semantic"]["n_sae_features"] = 8
    document["semantic"]["feature_clusters"]["representative_universe"] = [1]
    document["semantic"]["feature_clusters"]["representative_to_allowed_members"] = [
        [1, [1, member_id]]
    ]

    if accepted:
        assert sv._representative_members_from_document(document) == {
            1: (1, member_id)
        }
    else:
        with pytest.raises(
            AxBenchBpbAdvError, match="inconsistent representative members"
        ):
            sv._representative_members_from_document(document)


@pytest.mark.parametrize("bad", [None, 42, True, "semantic", []])
def test_d8_members_traversal_fails_loud_on_a_non_mapping_document(bad: Any) -> None:
    """G2-D8: bare subscripts leaked KeyError/TypeError instead of the domain error.

    The sibling ``_representative_universe_from_document`` already guarded this
    exact nested path defensively; this one did not, so the two disagreed on how
    a malformed document surfaces. Oracle-inherited, byte-identical there.
    """

    for document in ({}, {"semantic": bad}, {"semantic": {"feature_clusters": bad}}):
        with pytest.raises(AxBenchBpbAdvError, match="missing representative members"):
            sv._representative_members_from_document(document)
