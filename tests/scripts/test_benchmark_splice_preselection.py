"""Unit tests for the issue #740 PR-3a benchmark driver
(``scripts.analysis.benchmark_splice_preselection``).

Exercises the symlink-farm subset builder and the
``splice_preselection_locality`` analyzer integration on the SYNTHETIC
``build_interleaved_env`` fixture (never the real corpus -- that is the gated
acceptance suite's job). It does not invoke the multi-hour production
selection command. It exercises the driver's own subset, output-boundary,
exact-tree comparator, analyzer, resource-sampling, and log-parsing contracts;
the selection implementation's process-level byte identity is covered by the
dedicated parity matrix.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
from pathlib import Path

import pytest

from scripts.analysis.benchmark_splice_preselection import (
    BenchmarkArm,
    build_subset,
    compare_selected_shards_to_existing_reference,
    compare_selection_output_trees,
    parse_instrumented_log,
    parse_time_v_output,
    parse_worker_store_arms,
    resolve_benchmark_output_path,
    resolve_existing_reference_dir,
    run_arm,
)
from scripts.analysis.splice_preselection_locality import analyze
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import glob_overlay_shards, read_manifest
from tests.steering._splice_cf_cache_fixtures import build_interleaved_env


def _packed_stems(packed_dir: Path) -> list[str]:
    return sorted(p.name.split(".")[0] for p in glob_overlay_shards(packed_dir))


def test_metrics_output_defaults_below_real_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    assert resolve_benchmark_output_path(scratch, None) == (
        scratch / "metrics.json"
    ).resolve()


def test_metrics_output_accepts_nested_real_path_below_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    requested = scratch / "reports" / "run.json"
    assert resolve_benchmark_output_path(scratch, requested) == requested.resolve()


@pytest.mark.parametrize("relative", ["outside.json", "scratch"])
def test_metrics_output_rejects_path_not_strictly_below_scratch(
    tmp_path: Path,
    relative: str,
) -> None:
    scratch = tmp_path / "scratch"
    requested = tmp_path / relative
    with pytest.raises(ValueError, match="below scratch"):
        resolve_benchmark_output_path(scratch, requested)


def test_metrics_output_rejects_scratch_symlink(tmp_path: Path) -> None:
    target = tmp_path / "real-scratch"
    target.mkdir()
    scratch = tmp_path / "scratch-link"
    scratch.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="scratch must not be a symlink"):
        resolve_benchmark_output_path(scratch, None)


def test_metrics_output_rejects_output_symlink(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite")
    linked = scratch / "metrics.json"
    linked.symlink_to(outside)
    with pytest.raises(ValueError, match="output must not be a symlink"):
        resolve_benchmark_output_path(scratch, linked)
    assert outside.read_text() == "do not overwrite"


def test_existing_reference_resolves_real_disjoint_directory(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    reference = tmp_path / "existing-selection"
    reference.mkdir()
    assert resolve_existing_reference_dir(scratch, reference) == reference.resolve()
    assert resolve_existing_reference_dir(scratch, None) is None


@pytest.mark.parametrize("relative", ["scratch", "scratch/nested"])
def test_existing_reference_must_be_disjoint_from_scratch(
    tmp_path: Path,
    relative: str,
) -> None:
    scratch = tmp_path / "scratch"
    reference = tmp_path / relative
    reference.mkdir(parents=True)
    with pytest.raises(ValueError, match="must be disjoint"):
        resolve_existing_reference_dir(scratch, reference)


def test_existing_reference_rejects_symlink_or_non_directory(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    directory = tmp_path / "real-reference"
    directory.mkdir()
    link = tmp_path / "linked-reference"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        resolve_existing_reference_dir(scratch, link)

    regular = tmp_path / "regular-file"
    regular.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a directory"):
        resolve_existing_reference_dir(scratch, regular)


def test_parse_worker_store_arms_exact_issue_758_matrix() -> None:
    parsed = parse_worker_store_arms(
        "serial=1:private:0,w8c12=8:shared:12,"
        "w12c16=12:shared:16,w16c20=16:shared:20"
    )
    assert parsed == [
        BenchmarkArm("serial", 1, False, 0),
        BenchmarkArm("w8c12", 8, True, 12),
        BenchmarkArm("w12c16", 12, True, 16),
        BenchmarkArm("w16c20", 16, True, 20),
    ]
    assert parsed[0].hydra_overrides() == [
        "selection_num_workers=1",
        "selection_shared_store=false",
        "selection_shared_store_capacity=0",
    ]
    assert parsed[-1].hydra_overrides() == [
        "selection_num_workers=16",
        "selection_shared_store=true",
        "selection_shared_store_capacity=20",
    ]


def test_parse_worker_store_arms_trims_empty_commas_without_reordering() -> None:
    assert parse_worker_store_arms(" first=2:shared:3, ,second=1:private:0, ") == [
        BenchmarkArm("first", 2, True, 3),
        BenchmarkArm("second", 1, False, 0),
    ]


def _write_selection_tree(
    root: Path,
    *,
    created_at: str,
    shard_bytes: bytes = b"exact-selection-bytes",
    metadata_value: int = 7,
) -> None:
    root.mkdir()
    (root / "shard_00000.safetensors.zst").write_bytes(shard_bytes)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": "a" * 64,
                "kind": "packed_row",
                "created_at": created_at,
                "metadata": {"value": metadata_value},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_output_comparator_proves_shard_bytes_and_normalized_manifest(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_selection_tree(reference, created_at="2026-07-21T01:00:00Z")
    _write_selection_tree(candidate, created_at="2026-07-21T02:00:00Z")

    result = compare_selection_output_trees(reference, candidate)

    assert result["ok"] is True
    assert result["manifest_raw_equal"] is False
    assert result["manifest_normalized_equal"] is True
    assert result["manifest_created_at"] == {
        "reference": "2026-07-21T01:00:00Z",
        "candidate": "2026-07-21T02:00:00Z",
    }
    shard = result["file_identities"]["shard_00000.safetensors.zst"]
    assert shard["reference_bytes"] == len(b"exact-selection-bytes")
    assert shard["reference_sha256"] == shard["candidate_sha256"]


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        pytest.param(
            lambda root: (root / "shard_00000.safetensors.zst").write_bytes(
                b"exact-selection-byteX"
            ),
            "byte_mismatches",
            id="same-length-shard-drift",
        ),
        pytest.param(
            lambda root: (root / "extra.safetensors.zst").write_bytes(b"extra"),
            "extra",
            id="extra-file",
        ),
        pytest.param(
            lambda root: (root / "shard_00000.safetensors.zst").unlink(),
            "missing",
            id="missing-file",
        ),
        pytest.param(
            lambda root: (root / "unexpected-empty-directory").mkdir(),
            "extra_directories",
            id="extra-empty-directory",
        ),
    ],
)
def test_output_comparator_rejects_every_file_set_or_byte_difference(
    tmp_path: Path,
    mutation,
    expected_field: str,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_selection_tree(reference, created_at="same")
    _write_selection_tree(candidate, created_at="same")
    mutation(candidate)

    result = compare_selection_output_trees(reference, candidate)

    assert result["ok"] is False
    assert result[expected_field]


def test_output_comparator_rejects_symlink_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_selection_tree(reference, created_at="same")
    _write_selection_tree(candidate, created_at="same")
    candidate_shard = candidate / "shard_00000.safetensors.zst"
    candidate_shard.unlink()
    candidate_shard.symlink_to(reference / candidate_shard.name)

    result = compare_selection_output_trees(reference, candidate)

    assert result["ok"] is False
    assert result["candidate_unsafe_entries"] == [
        "shard_00000.safetensors.zst:symlink"
    ]


def test_output_comparator_rejects_manifest_semantic_drift(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_selection_tree(reference, created_at="same", metadata_value=7)
    _write_selection_tree(candidate, created_at="same", metadata_value=8)

    result = compare_selection_output_trees(reference, candidate)

    assert result["ok"] is False
    assert result["manifest_normalized_equal"] is False
    assert result["byte_mismatches"] == ["manifest.json"]


@pytest.mark.parametrize("bad_manifest", ["[]", "not-json"])
def test_output_comparator_fails_loudly_on_invalid_manifest(
    tmp_path: Path,
    bad_manifest: str,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_selection_tree(reference, created_at="same")
    _write_selection_tree(candidate, created_at="same")
    (candidate / "manifest.json").write_text(bad_manifest, encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        compare_selection_output_trees(reference, candidate)


def test_existing_reference_comparator_checks_only_named_exact_shards(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "existing"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    selected = ["shard_00000", "shard_00002"]
    for stem, payload in zip(selected, (b"first", b"second"), strict=True):
        filename = f"{stem}.safetensors.zst"
        (reference / filename).write_bytes(payload)
        (candidate / filename).write_bytes(payload)
    (reference / "shard_00422.safetensors.zst").write_bytes(b"unselected-extra")

    result = compare_selected_shards_to_existing_reference(
        reference,
        candidate,
        selected,
    )

    assert result["ok"] is True
    assert result["missing_reference"] == []
    assert result["missing_candidate"] == []
    assert result["byte_mismatches"] == []
    assert sorted(result["file_identities"]) == [
        "shard_00000.safetensors.zst",
        "shard_00002.safetensors.zst",
    ]
    assert (
        result["file_identities"]["shard_00000.safetensors.zst"][
            "reference_sha256"
        ]
        == result["file_identities"]["shard_00000.safetensors.zst"][
            "candidate_sha256"
        ]
    )


def test_existing_reference_comparator_reports_missing_and_byte_drift(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "existing"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    (reference / "shard_00000.safetensors.zst").write_bytes(b"reference")
    (candidate / "shard_00000.safetensors.zst").write_bytes(b"candidate")
    (reference / "shard_00001.safetensors.zst").write_bytes(b"only-reference")
    (candidate / "shard_00002.safetensors.zst").write_bytes(b"only-candidate")

    result = compare_selected_shards_to_existing_reference(
        reference,
        candidate,
        ["shard_00000", "shard_00001", "shard_00002"],
    )

    assert result["ok"] is False
    assert result["byte_mismatches"] == ["shard_00000.safetensors.zst"]
    assert result["missing_candidate"] == ["shard_00001.safetensors.zst"]
    assert result["missing_reference"] == ["shard_00002.safetensors.zst"]


@pytest.mark.parametrize(
    "shard_names",
    [[], ["shard_00000", "shard_00000"], ["../shard_00000"]],
)
def test_existing_reference_comparator_rejects_ambiguous_shard_domain(
    tmp_path: Path,
    shard_names: list[str],
) -> None:
    reference = tmp_path / "existing"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    with pytest.raises(ValueError, match="at least one|duplicates|plain shard stems"):
        compare_selected_shards_to_existing_reference(
            reference,
            candidate,
            shard_names,
        )


@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_existing_reference_comparator_rejects_symlinked_shard(
    tmp_path: Path,
    side: str,
) -> None:
    reference = tmp_path / "existing"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    payload = tmp_path / "payload"
    payload.write_bytes(b"same bytes")
    for label, root in (("reference", reference), ("candidate", candidate)):
        path = root / "shard_00000.safetensors.zst"
        if label == side:
            path.symlink_to(payload)
        else:
            path.write_bytes(payload.read_bytes())

    with pytest.raises(ValueError, match="must be a real regular file"):
        compare_selected_shards_to_existing_reference(
            reference,
            candidate,
            ["shard_00000"],
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "no arms"),
        (", ,", "no arms"),
        ("bad name=1:private:0", "must be named"),
        ("missing-equals", "must be named"),
        ("x=1:shared", "expected workers:mode:capacity"),
        ("x=many:shared:2", "non-integer"),
        ("x=1:shared:many", "non-integer"),
        ("x=0:private:0", "workers must be"),
        ("x=1:unknown:0", "mode must be"),
        ("x=2:shared:0", "capacity must be"),
        ("x=2:shared:-1", "capacity must be"),
        ("x=1:private:1", "capacity must be 0"),
        ("x=1:private:0,x=2:shared:3", "duplicate arm name"),
    ],
)
def test_parse_worker_store_arms_rejects_ambiguous_or_unsafe_shapes(
    raw: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_worker_store_arms(raw)


# --- build_subset ------------------------------------------------------------


def test_build_subset_symlinks_only_the_chosen_shards(tmp_path: Path) -> None:
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    all_stems = _packed_stems(packed_dir)
    assert len(all_stems) >= 2, "fixture must build at least 2 packed shards for a real subset"
    chosen = all_stems[:2]
    prod_snapshot_before = {p.name for p in packed_dir.iterdir()}

    subset = build_subset(packed_dir, tmp_path / "scratch_subset", chosen)

    assert subset == tmp_path / "scratch_subset"
    assert _packed_stems(subset) == chosen

    for stem in chosen:
        fn = f"{stem}.safetensors.zst"
        packed_link = subset / fn
        prov_link = provenance_dir(subset) / fn
        assert packed_link.is_symlink(), f"{fn} was copied, not symlinked"
        assert prov_link.is_symlink(), f"provenance/{fn} was copied, not symlinked"
        assert packed_link.resolve() == (packed_dir / fn).resolve()
        assert prov_link.resolve() == (provenance_dir(packed_dir) / fn).resolve()

    # Read-only on the production dir: iterating it after the build shows
    # exactly what was there before.
    assert {p.name for p in packed_dir.iterdir()} == prod_snapshot_before


def test_build_subset_manifest_provenance_shards_matches_subset_count(tmp_path: Path) -> None:
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    original_manifest = read_manifest(packed_dir)
    assert original_manifest is not None
    chosen = _packed_stems(packed_dir)[:2]

    subset = build_subset(packed_dir, tmp_path / "scratch2", chosen)
    produced = json.loads((subset / "manifest.json").read_text())

    assert produced["metadata"]["provenance_shards"] == len(chosen)
    # base_shard_stems is the FULL original source-shard list, unchanged --
    # packed-segment provenance indexes into it by absolute position, so
    # truncating it would resolve a subset shard's provenance to the wrong
    # stem entirely.
    assert (
        produced["metadata"]["base_shard_stems"]
        == original_manifest["metadata"]["base_shard_stems"]
    )
    assert len(original_manifest["metadata"]["base_shard_stems"]) > len(chosen), (
        "test is only meaningful if base_shard_stems is NOT already == the chosen subset"
    )
    # Every other manifest field (checksum, kind, created_at, ...) is
    # preserved verbatim -- the same approach the already-passing gated
    # real-l22 test uses (test_splice_preselection_real_l22.py's
    # _build_symlink_subset).
    for key in original_manifest:
        if key == "metadata":
            continue
        assert produced[key] == original_manifest[key], f"manifest field {key!r} was altered"
    for key in original_manifest["metadata"]:
        if key == "provenance_shards":
            continue
        assert produced["metadata"][key] == original_manifest["metadata"][key], (
            f"metadata field {key!r} was altered"
        )


def test_build_subset_missing_packed_shard_raises(tmp_path: Path) -> None:
    env = build_interleaved_env(tmp_path / "env")
    with pytest.raises(FileNotFoundError):
        build_subset(Path(env["packed_row_dir"]), tmp_path / "scratch3", ["shard_does_not_exist"])


def test_build_subset_rejects_empty_shard_list(tmp_path: Path) -> None:
    env = build_interleaved_env(tmp_path / "env")
    with pytest.raises(ValueError):
        build_subset(Path(env["packed_row_dir"]), tmp_path / "scratch4", [])


def test_build_subset_missing_manifest_raises(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not_a_packed_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        build_subset(empty_dir, tmp_path / "scratch5", ["shard_00000"])


def test_build_subset_refuses_to_overwrite_prior_scratch(tmp_path: Path) -> None:
    """A retry cannot silently mix with or delete a prior benchmark."""
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    all_stems = _packed_stems(packed_dir)
    scratch = tmp_path / "scratch6"

    build_subset(packed_dir, scratch, all_stems[:1])
    before = {path.name for path in scratch.iterdir()}
    with pytest.raises(ValueError, match="absent or empty"):
        build_subset(packed_dir, scratch, all_stems[:2])

    assert {path.name for path in scratch.iterdir()} == before
    assert _packed_stems(scratch) == all_stems[:1]


@pytest.mark.parametrize(
    "names",
    [
        ["shard_00000", "shard_00000"],
        ["../shard_00000"],
        ["nested/shard_00000"],
        [""],
    ],
)
def test_build_subset_rejects_duplicate_or_path_like_shard_names(
    tmp_path: Path,
    names: list[str],
) -> None:
    env = build_interleaved_env(tmp_path / "env")
    with pytest.raises(ValueError, match="duplicates|plain packed-shard stems"):
        build_subset(Path(env["packed_row_dir"]), tmp_path / "scratch_names", names)


def test_build_subset_rejects_source_and_scratch_path_overlap(tmp_path: Path) -> None:
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    stem = _packed_stems(packed_dir)[0]

    for unsafe in (packed_dir, packed_dir / "nested", packed_dir.parent):
        with pytest.raises(ValueError, match="must be disjoint"):
            build_subset(packed_dir, unsafe, [stem])


def test_build_subset_rejects_nonempty_unowned_scratch_without_mutation(
    tmp_path: Path,
) -> None:
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    scratch = tmp_path / "occupied"
    scratch.mkdir()
    marker = scratch / "important.txt"
    marker.write_text("preserve")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_subset(packed_dir, scratch, [_packed_stems(packed_dir)[0]])

    assert marker.read_text() == "preserve"


# --- analyzer integration -----------------------------------------------------


def test_analyzer_on_subset_reports_sane_monotone_load_counts(tmp_path: Path) -> None:
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    all_stems = _packed_stems(packed_dir)
    subset = build_subset(packed_dir, tmp_path / "scratch7", all_stems)

    report = analyze(subset, [1, 2, 4])

    assert report["n_packed_shards"] == len(all_stems)
    assert report["unique_source_stems"] > 0
    loads = report["loads_by_capacity"]
    assert set(loads) == {1, 2, 4}
    # A bigger cache can never need MORE loads than a smaller one replaying
    # the identical access sequence.
    assert loads[1] >= loads[2] >= loads[4]
    # Every capacity's load count is bounded below by the ideal/cold-cache
    # floor (the true number of distinct source stems) and above by the
    # capacity-1 sequential-reload count.
    assert report["unique_source_stems"] <= loads[4] <= loads[1]
    assert len(report["per_shard"]) == len(all_stems)


def test_analyzer_subset_matches_full_dir_when_subset_is_everything(tmp_path: Path) -> None:
    """A subset built from EVERY packed shard must report identical locality
    to analyzing the original dir directly: the symlink farm changes nothing
    about the packed-segment access sequence the analyzer replays.
    """
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    all_stems = _packed_stems(packed_dir)
    subset = build_subset(packed_dir, tmp_path / "scratch8", all_stems)

    caps = [1, 2, 3]
    subset_report = analyze(subset, caps)
    full_report = analyze(packed_dir, caps)

    assert subset_report["loads_by_capacity"] == full_report["loads_by_capacity"]
    assert subset_report["unique_source_stems"] == full_report["unique_source_stems"]
    assert subset_report["per_shard"] == full_report["per_shard"]


def test_analyzer_on_true_subset_has_fewer_or_equal_unique_stems_than_full_dir(
    tmp_path: Path,
) -> None:
    env = build_interleaved_env(tmp_path / "env")
    packed_dir = Path(env["packed_row_dir"])
    all_stems = _packed_stems(packed_dir)
    assert len(all_stems) >= 2
    subset = build_subset(packed_dir, tmp_path / "scratch9", all_stems[:1])

    caps = [1, 2]
    subset_report = analyze(subset, caps)
    full_report = analyze(packed_dir, caps)

    assert subset_report["n_packed_shards"] == 1
    assert full_report["n_packed_shards"] == len(all_stems)
    assert subset_report["unique_source_stems"] <= full_report["unique_source_stems"]


# --- pure log-parsing helpers (run_arm's building blocks) --------------------


def test_parse_instrumented_log_extracts_per_shard_and_totals() -> None:
    text = (
        "2026-07-19 10:00:00 | INFO | precompute_selected_features:main:2104 - "
        "Processing 3 base shards...\n"
        "2026-07-19 10:00:01 | INFO | precompute_selected_features:select:1772 - "
        "  shard_00000: splice_counterfactual 42 segs, 10 sel, 3 neg (1034.8s)\n"
        "2026-07-19 10:00:02 | INFO | precompute_selected_features:select:1772 - "
        "  shard_00001: splice_counterfactual 17 segs, 5 sel, 1 neg (767.8s)\n"
        "2026-07-19 10:00:03 | INFO | precompute_selected_features:main:2186 - "
        "Done: 3 shards in 2504s\n"
        "2026-07-19 10:00:03 | INFO | precompute_selected_features:main:2189 - "
        "Peak RSS this process: 51.2 GB\n"
    )
    parsed = parse_instrumented_log(text)
    assert parsed["per_shard_seconds"] == {"shard_00000": 1034.8, "shard_00001": 767.8}
    assert parsed["total_elapsed_seconds"] == 2504.0
    assert parsed["total_shards_reported"] == 3
    assert parsed["self_reported_peak_rss_gb"] == 51.2


def test_parse_instrumented_log_missing_lines_returns_nones() -> None:
    parsed = parse_instrumented_log("nothing relevant in this log\n")
    assert parsed["per_shard_seconds"] == {}
    assert parsed["total_elapsed_seconds"] is None
    assert parsed["total_shards_reported"] is None
    assert parsed["self_reported_peak_rss_gb"] is None
    assert parsed["resource_profiles"] == []
    assert parsed["resource_profile_errors"] == []


def test_parse_instrumented_log_extracts_exact_resource_profile_json() -> None:
    profile = {
        "schema_version": 1,
        "shard": "shard_00007",
        "packed_segments": 282343,
        "source_doc_groups": 4000,
        "distinct_source_stems": 13,
        "stages_seconds": {
            "packed_and_provenance_load": 1.25,
            "grouping": 2.5,
            "selection": 100.0,
            "flatten_and_save": 3.0,
            "total": 106.75,
        },
        "feature_cache_delta": {"shared_hits": 12, "private_fallbacks": 0},
        "sidecar_cache_delta": {"loads": 13, "bytes_read": 12345},
    }
    text = (
        "2026-07-21 12:00:00 | INFO | worker - SPLICE_RESOURCE_PROFILE "
        + json.dumps(profile, sort_keys=True, separators=(",", ":"))
        + "\n"
    )

    parsed = parse_instrumented_log(text)

    assert parsed["resource_profiles"] == [profile]
    assert parsed["resource_profile_errors"] == []


def test_parse_instrumented_log_retains_valid_profiles_and_reports_bad_line() -> None:
    text = (
        'INFO SPLICE_RESOURCE_PROFILE {"schema_version":1,"shard":"good"}\n'
        'INFO SPLICE_RESOURCE_PROFILE {"schema_version":1,"shard":bad}\n'
        'INFO SPLICE_RESOURCE_PROFILE {"schema_version":1,"shard":"later"}\n'
    )
    parsed = parse_instrumented_log(text)
    assert parsed["resource_profiles"] == [
        {"schema_version": 1, "shard": "good"},
        {"schema_version": 1, "shard": "later"},
    ]
    assert len(parsed["resource_profile_errors"]) == 1
    assert parsed["resource_profile_errors"][0].startswith("line 2:")


def test_parse_time_v_output_extracts_rss_and_wall_clock() -> None:
    text = (
        '\tCommand being timed: "python -m foo"\n'
        "\tUser time (seconds): 12.34\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:17.34\n"
        "\tMaximum resident set size (kbytes): 53687091\n"
    )
    parsed = parse_time_v_output(text)
    assert parsed["peak_rss_kb"] == 53687091
    assert parsed["wall_clock"] == "0:17.34"


def test_parse_time_v_output_missing_fields_returns_none() -> None:
    parsed = parse_time_v_output("no time -v markers here\n")
    assert parsed["peak_rss_kb"] is None
    assert parsed["wall_clock"] is None


@pytest.mark.skipif(shutil.which("/usr/bin/time") is None, reason="GNU /usr/bin/time not available")
def test_run_arm_wraps_a_trivial_command_and_parses_time_v(tmp_path: Path) -> None:
    """A trivial command through the real nice + /usr/bin/time -v wrapper --
    proves run_arm's subprocess plumbing without depending on
    precompute_selected_features at all.
    """
    log_path = tmp_path / "trivial.log"
    metrics = run_arm(["python3", "-c", "print('hello from run_arm')"], log_path)

    assert metrics["returncode"] == 0
    assert metrics["ok"] is True
    assert metrics["peak_rss_kb"] is not None and metrics["peak_rss_kb"] > 0
    assert metrics["wall_clock"] is not None
    assert metrics["instrumented"] == parse_instrumented_log(log_path.read_text())
    assert log_path.is_file()
    assert "hello from run_arm" in log_path.read_text()


@pytest.mark.skipif(shutil.which("/usr/bin/time") is None, reason="GNU /usr/bin/time not available")
def test_run_arm_reports_nonzero_returncode_without_raising(tmp_path: Path) -> None:
    log_path = tmp_path / "failing.log"
    metrics = run_arm(["python3", "-c", "import sys; sys.exit(3)"], log_path)

    assert metrics["returncode"] == 3
    assert metrics["ok"] is False


@pytest.mark.skipif(
    shutil.which("/usr/bin/time") is None or not Path("/proc/self/smaps_rollup").is_file(),
    reason="GNU /usr/bin/time and Linux /proc required",
)
def test_run_arm_records_process_tree_pss_uss_and_raw_samples(tmp_path: Path) -> None:
    log_path = tmp_path / "sampled.log"
    resource_path = tmp_path / "sampled.resources.jsonl"
    metrics = run_arm(
        ["python3", "-c", "import time; time.sleep(0.06)"],
        log_path,
        resource_path=resource_path,
        resource_interval_seconds=0.01,
    )

    assert metrics["returncode"] == 0
    assert metrics["ok"] is True
    assert metrics["resource_path"] == str(resource_path)
    resources = metrics["resources"]
    assert resources["schema_version"] == 1
    assert resources["sample_count"] >= 2
    assert resources["peak_process_count"] >= 1
    assert resources["peak_rss_kB_sum"] > 0
    assert resources["peak_pss_kB_sum"] > 0
    assert resources["peak_uss_kB_sum"] >= 0
    assert resources["peak_pss_kB_sum"] <= resources["peak_rss_kB_sum"]
    assert resource_path.read_text().count("\n") == resources["sample_count"]


def test_run_arm_rejects_nonpositive_resource_interval_before_launch(tmp_path: Path) -> None:
    log_path = tmp_path / "never-created.log"
    with pytest.raises(ValueError, match="resource_interval_seconds"):
        run_arm(
            ["python3", "-c", "raise AssertionError('must not launch')"],
            log_path,
            resource_path=tmp_path / "never-created.jsonl",
            resource_interval_seconds=0,
        )
    assert not log_path.exists()


@pytest.mark.skipif(shutil.which("/usr/bin/time") is None, reason="GNU /usr/bin/time not available")
def test_run_arm_sampling_failure_terminates_the_separate_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_pid: int | None = None

    def fail_sampler(process, _output, **_kwargs):
        nonlocal captured_pid
        captured_pid = process.pid
        assert os.getsid(process.pid) == process.pid
        raise RuntimeError("synthetic sampler failure")

    monkeypatch.setattr(
        "scripts.analysis.benchmark_splice_preselection.sample_process_until_exit",
        fail_sampler,
    )
    with pytest.raises(RuntimeError, match="synthetic sampler failure"):
        run_arm(
            ["python3", "-c", "import time; time.sleep(300)"],
            tmp_path / "failed-sampling.log",
            resource_path=tmp_path / "failed-sampling.jsonl",
            resource_interval_seconds=0.01,
        )

    assert captured_pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(captured_pid, signal.SIGCONT)
