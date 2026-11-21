"""Artifact metrics plus unclustered source-contract schema v6.

Pins the post-#512 and post-#734 source-contract semantics:
- artifact_byte_sizes holds BYTES for every required artifact (files: st_size;
  directories: recursive sum of regular-file st_size).
- artifact_entry_counts holds the recursive entry count for DIRECTORY artifacts only.
- ordinary files and directories retain exact-byte hashes.
- seed safetensors and their manifest use canonical semantic hashes, while the
  seed manifest continues to verify their exact transport bytes.
- schema_version is 6 and the committed seed-v2 pair remains in the scientific
  source identity.
Also asserts the artifact-metric migration invariants, immutable installation,
and the direction-training-layer identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
from steering.eval.axbench_base.seed_contract import sha256_file
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv import source_validation as sv
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import (
    SOURCE_CONTRACT_FILENAME,
    AxBenchBpbAdvPaths,
)
from steering.eval.axbench_bpb_adv.source_validation import (
    SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
    _directory_byte_size_and_entry_count,
    _hash_directory,
    contract_hash,
    validate_stage_0_3_source_run,
    write_source_contract,
)
from steering.safetensors_io import deterministic_safetensors_bytes
from tests.axbench_bpb_adv.source_run_builders import (
    STUB_FILE_ARTIFACTS,
    both_class_rows,
    write_stage_0_3_source_run,
)

_DIRECTION_PAYLOAD = deterministic_safetensors_bytes(
    {"direction": t.tensor([1.0, -1.0], dtype=t.float32)},
    metadata={
        "concept_id": "1",
        "mode": "positive",
        "target_model": "decoder_only",
        "layer": "6",
    },
)
# Known, DISTINCT file byte-lengths so the directory byte-size != its entry-count.
_BASELINES_FILE_BYTES = len(_DIRECTION_PAYLOAD) + 7 + 11
# Recursive descendants of baselines/: 7 directories + 3 files == 10 entries.
#   dirs: diffmean, diffmean/decoder_only, diffmean/decoder_only/positive,
#         prompting, lora, lora/decoder_only, lora/decoder_only/negative
#   files: the 3 written below
_BASELINES_ENTRY_COUNT = 10

# Frozen golden hashes for the deterministic fixture artifacts. They anchor
# _hash_directory / _hash_file against silent algorithm changes: the migration
# test (test_exact_equality_migration_invariants_vs_old_v1) only proves internal
# consistency because it recomputes the reference via the SAME functions, so it
# cannot catch a future change to those functions. The baselines/ tree and the
# b"x" stub files are pure-content (no pyarrow), so these literals are stable
# across environments; update them ONLY if _hash_file / _hash_directory
# intentionally change (which would also change every on-disk contract hash).
_GOLDEN_BASELINES_DIR_HASH = (
    "8bb007f3c875780fd1b7e42161c0fe0866a1169333d9e732237b0a8f27aac16d"
)
_GOLDEN_STUB_FILE_HASH = (  # sha256(b"x")
    "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"
)
def _write_baselines_tree(baselines: Path) -> None:
    """A realistic nested baselines/ tree with files of distinct known lengths."""
    baselines.mkdir()
    p1 = baselines / "diffmean" / "decoder_only" / "positive"
    p1.mkdir(parents=True)
    (p1 / "concept_0001.safetensors").write_bytes(_DIRECTION_PAYLOAD)
    p2 = baselines / "prompting"
    p2.mkdir(parents=True)
    (p2 / "positive_descriptions.json").write_bytes(b"b" * 7)
    p3 = baselines / "lora" / "decoder_only" / "negative"
    p3.mkdir(parents=True)
    (p3 / "concept_0001.pt").write_bytes(b"c" * 11)


def _write_source_run(root: Path) -> None:
    """A valid Stage 0-3 source run whose baselines/ is a realistic nested tree."""
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("abc"),
        negative_rows=both_class_rows("xyz"),
        baselines_builder=_write_baselines_tree,
    )


_FILE_ARTIFACT_NAMES = frozenset(
    {
        "concepts.json",
        "synth_training_positive.parquet",
        "synth_training_negative.parquet",
        "seeds_train.safetensors",
        "seeds_eval.safetensors",
        "seed_pool_manifest.json",
        *STUB_FILE_ARTIFACTS,
    }
)
_ALL_ARTIFACT_NAMES = _FILE_ARTIFACT_NAMES | {"baselines/"}


@pytest.fixture
def source_run(tmp_path: Path) -> Path:
    """A freshly-built, valid Stage 0-3 source run directory."""
    root = tmp_path / "source"
    _write_source_run(root)
    return root


@pytest.fixture
def source_contract(source_run: Path) -> BpbAdvSourceContract:
    """The contract from validating the ``source_run`` fixture once."""
    return validate_stage_0_3_source_run(source_run).contract


def _expected_serialized(contract: BpbAdvSourceContract) -> str:
    return (
        json.dumps(contract.to_json_dict(), sort_keys=True, indent=2, default=str)
        + "\n"
    )


def _seed_fingerprint(source_run: Path) -> str:
    manifest = json.loads(
        (source_run / "seed_pool_manifest.json").read_text(encoding="utf-8")
    )
    return str(manifest["seed_fingerprint"])


def _reconstruct_v1_sizes_and_hashes(
    root: Path,
) -> tuple[dict[str, int], dict[str, str]]:
    """Recompute the pre-#512 (schema v1) ``artifact_sizes`` + ``artifact_hashes``
    from the documented old formulas, to prove the new contract only relabels and
    extends them (it does not recompute or drop any preserved value)."""
    paths = SourcePaths(root)
    old_sizes: dict[str, int] = {}
    old_hashes: dict[str, str] = {}
    for name, path in sv._required_artifacts(paths):
        if path.is_dir():
            old_sizes[name] = sum(1 for _ in path.rglob("*"))
            old_hashes[name] = _hash_directory(path)
        else:
            old_sizes[name] = path.stat().st_size
            old_hashes[name] = sv._hash_file(path)
    return old_sizes, old_hashes


def test_source_contract_filename_constant_is_wired(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    assert SOURCE_CONTRACT_FILENAME == "source_contract.json"
    assert paths.source_contract().name == SOURCE_CONTRACT_FILENAME


def test_write_source_contract_uses_immutable_install(
    source_contract: BpbAdvSourceContract,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bytes, Path, str]] = []

    def fake_install(payload: bytes, final_path: Path, *, surface: str) -> bool:
        calls.append((payload, Path(final_path), surface))
        return True

    monkeypatch.setattr(sv, "install_bytes_once", fake_install)
    out = tmp_path / "deep" / "nested" / "source_contract.json"
    write_source_contract(out, source_contract)

    assert len(calls) == 1
    payload, path, surface = calls[0]
    assert path == out
    assert payload.decode("utf-8") == _expected_serialized(source_contract)
    assert surface == "BPB-Adv source contract"


def test_write_source_contract_bytes_and_parent_dirs(
    source_contract: BpbAdvSourceContract, tmp_path: Path
) -> None:
    out = tmp_path / "a" / "b" / "c" / "source_contract.json"

    returned = write_source_contract(out, source_contract)

    assert returned == out
    assert out.exists()
    assert out.read_text(encoding="utf-8") == _expected_serialized(source_contract)


def test_directory_metrics_empty_dir(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    assert _directory_byte_size_and_entry_count(d) == (0, 0)


def test_directory_metrics_flat_files(tmp_path: Path) -> None:
    d = tmp_path / "flat"
    d.mkdir()
    (d / "a").write_bytes(b"x" * 3)
    (d / "b").write_bytes(b"y" * 4)
    # 2 files, 7 bytes, no subdirs
    assert _directory_byte_size_and_entry_count(d) == (7, 2)


def test_directory_metrics_nested(tmp_path: Path) -> None:
    d = tmp_path / "nested"
    sub = d / "x" / "y"
    sub.mkdir(parents=True)
    (sub / "f").write_bytes(b"z" * 9)
    (d / "g").write_bytes(b"w" * 1)
    # files: f(9) + g(1) = 10 bytes; entries: dirs x, x/y + files f, g = 4
    assert _directory_byte_size_and_entry_count(d) == (10, 4)


def test_directory_metrics_empty_subdir_counts_but_adds_no_bytes(
    tmp_path: Path,
) -> None:
    d = tmp_path / "withsub"
    d.mkdir()
    (d / "f").write_bytes(b"q" * 6)
    before = _directory_byte_size_and_entry_count(d)
    (d / "emptysub").mkdir()
    after = _directory_byte_size_and_entry_count(d)
    assert before == (6, 1)
    assert after == (6, 2)  # entry count +1 (the dir); bytes unchanged


def test_directory_metrics_dir_only_no_files(tmp_path: Path) -> None:
    # A directory tree with only subdirectories (no files) -> 0 bytes, but the
    # nested dirs still count as entries.
    d = tmp_path / "dironly"
    (d / "a" / "b").mkdir(parents=True)
    (d / "c").mkdir()
    # descendant dirs: a, a/b, c == 3 entries; 0 non-dir children -> 0 bytes
    assert _directory_byte_size_and_entry_count(d) == (0, 3)


def test_directory_metrics_match_reference_formulas(tmp_path: Path) -> None:
    d = tmp_path / "ref"
    _write_baselines_tree(d)
    byte_size, entry_count = _directory_byte_size_and_entry_count(d)
    # byte_size == independent sum of every non-dir child's st_size
    assert byte_size == sum(
        c.stat().st_size for c in d.rglob("*") if not c.is_dir()
    )
    # entry_count == the pre-#512 formula exactly
    assert entry_count == sum(1 for _ in d.rglob("*"))
    # and the known absolute values (these differ -> two genuine units)
    assert byte_size == _BASELINES_FILE_BYTES
    assert entry_count == _BASELINES_ENTRY_COUNT
    assert byte_size != entry_count


def test_contract_has_split_fields_and_no_artifact_sizes(
    source_contract: BpbAdvSourceContract,
) -> None:
    assert hasattr(source_contract, "artifact_byte_sizes")
    assert hasattr(source_contract, "artifact_entry_counts")
    assert not hasattr(source_contract, "artifact_sizes")


def test_contract_schema_version_is_6_and_binds_seed_v2_and_layer(
    source_contract: BpbAdvSourceContract,
    source_run: Path,
) -> None:
    assert SOURCE_CONTRACT_SCHEMA_VERSION == 6
    assert source_contract.schema_version == 6
    assert source_contract.to_json_dict()["schema_version"] == 6
    assert source_contract.seed_schema_version == 2
    assert source_contract.seed_fingerprint == _seed_fingerprint(source_run)
    assert source_contract.direction_training_layer == 6


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("schema_version", 1),
        ("token_semantics", "legacy-bos-inclusive"),
        ("seed_fingerprint", "9" * 64),
    ],
)
def test_source_validation_rejects_legacy_or_mismatched_seed_contract(
    source_run: Path,
    field: str,
    stale_value: object,
) -> None:
    manifest_path = source_run / "seed_pool_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = stale_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="seed schema v2|seed schema"):
        validate_stage_0_3_source_run(source_run)


def test_source_contract_dataclass_rejects_schema_v3() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="schema v6"):
        BpbAdvSourceContract(
            seed_fingerprint="1" * 64,
            schema_version=3,
        )


def test_byte_sizes_cover_all_artifacts_entry_counts_dir_only(
    source_contract: BpbAdvSourceContract,
) -> None:
    assert set(source_contract.artifact_byte_sizes) == _ALL_ARTIFACT_NAMES
    assert set(source_contract.artifact_hashes) == _ALL_ARTIFACT_NAMES
    assert set(source_contract.artifact_entry_counts) == {"baselines/"}


def test_file_byte_sizes_equal_stat_st_size(
    source_run: Path, source_contract: BpbAdvSourceContract
) -> None:
    for name in _FILE_ARTIFACT_NAMES:
        assert source_contract.artifact_byte_sizes[name] == (source_run / name).stat().st_size


def test_baselines_byte_size_and_entry_count_known_and_differ(
    source_contract: BpbAdvSourceContract,
) -> None:
    assert source_contract.artifact_byte_sizes["baselines/"] == _BASELINES_FILE_BYTES
    assert source_contract.artifact_entry_counts["baselines/"] == _BASELINES_ENTRY_COUNT
    assert (
        source_contract.artifact_byte_sizes["baselines/"]
        != source_contract.artifact_entry_counts["baselines/"]
    )


def test_artifact_hashes_match_frozen_golden(
    source_contract: BpbAdvSourceContract,
) -> None:
    # Anchors _hash_directory (paths + sizes + content over the nested tree)...
    assert source_contract.artifact_hashes["baselines/"] == _GOLDEN_BASELINES_DIR_HASH
    # ...and _hash_file (plain sha256 of the file bytes) for each b"x" stub.
    for name in STUB_FILE_ARTIFACTS:
        assert source_contract.artifact_hashes[name] == _GOLDEN_STUB_FILE_HASH


def test_stage_4_5_dirs_excluded_from_both_metric_dicts(source_run: Path) -> None:
    (source_run / "generations").mkdir()
    (source_run / "judge_scores").mkdir()
    contract = validate_stage_0_3_source_run(source_run).contract
    assert "generations" not in contract.artifact_byte_sizes
    assert "judge_scores" not in contract.artifact_byte_sizes
    assert "generations" not in contract.artifact_entry_counts
    assert "judge_scores" not in contract.artifact_entry_counts


def test_artifact_metric_invariants_survive_schema_v6(source_run: Path) -> None:
    old_sizes, old_hashes = _reconstruct_v1_sizes_and_hashes(source_run)
    new = validate_stage_0_3_source_run(source_run).contract

    # (1) scientific sources are schema v6 and bind seed v2 plus direction layer
    assert new.schema_version == 6
    assert new.seed_schema_version == 2
    assert new.direction_training_layer == 6
    # (2) every file byte size is byte-identical to the old artifact_sizes
    for name in _FILE_ARTIFACT_NAMES:
        assert new.artifact_byte_sizes[name] == old_sizes[name]
    # (3) the directory entry count == old artifact_sizes['baselines/'] (relocated verbatim)
    assert new.artifact_entry_counts["baselines/"] == old_sizes["baselines/"]
    # (4) the directory byte size is the ONLY genuinely-new datum
    assert new.artifact_byte_sizes["baselines/"] == _BASELINES_FILE_BYTES
    # (5) ordinary artifacts keep their exact-byte hash; only the seed pair and
    # manifest move to canonical semantic hashes.
    semantic_seed_artifacts = {
        "seeds_train.safetensors",
        "seeds_eval.safetensors",
        "seed_pool_manifest.json",
    }
    for name in _ALL_ARTIFACT_NAMES - semantic_seed_artifacts:
        assert new.artifact_hashes[name] == old_hashes[name]


def _reverse_safetensors_metadata_order(path: Path) -> None:
    raw = path.read_bytes()
    header_length = int.from_bytes(raw[:8], byteorder="little")
    header_end = 8 + header_length
    header = json.loads(raw[8:header_end])
    metadata = header["__metadata__"]
    header["__metadata__"] = {
        key: metadata[key] for key in reversed(list(metadata))
    }
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(encoded) <= header_length
    path.write_bytes(
        raw[:8]
        + encoded
        + b" " * (header_length - len(encoded))
        + raw[header_end:]
    )


def test_contract_ignores_safetensors_metadata_key_order(source_run: Path) -> None:
    first = validate_stage_0_3_source_run(source_run).contract
    seed_path = source_run / "seeds_train.safetensors"
    manifest_path = source_run / "seed_pool_manifest.json"
    raw_hash_before = sha256_file(seed_path)

    _reverse_safetensors_metadata_order(seed_path)
    raw_hash_after = sha256_file(seed_path)
    assert raw_hash_after != raw_hash_before

    manifest_bytes = manifest_path.read_bytes()
    assert manifest_bytes.count(raw_hash_before.encode("ascii")) == 1
    manifest_path.write_bytes(
        manifest_bytes.replace(
            raw_hash_before.encode("ascii"),
            raw_hash_after.encode("ascii"),
        )
    )

    second = validate_stage_0_3_source_run(source_run).contract
    assert second.artifact_byte_sizes == first.artifact_byte_sizes
    assert second.artifact_hashes == first.artifact_hashes
    assert contract_hash(second.to_json_dict()) == contract_hash(first.to_json_dict())


def test_metric_sensitivity_content_change_same_length(source_run: Path) -> None:
    first = validate_stage_0_3_source_run(source_run).contract
    target = source_run / "baselines" / "prompting" / "positive_descriptions.json"
    target.write_bytes(b"X" * 7)  # same length, different content
    second = validate_stage_0_3_source_run(source_run).contract
    assert (
        second.artifact_byte_sizes["baselines/"]
        == first.artifact_byte_sizes["baselines/"]
    )
    assert (
        second.artifact_entry_counts["baselines/"]
        == first.artifact_entry_counts["baselines/"]
    )
    assert second.artifact_hashes["baselines/"] != first.artifact_hashes["baselines/"]
    assert contract_hash(second.to_json_dict()) != contract_hash(first.to_json_dict())


def test_metric_sensitivity_append_bytes(source_run: Path) -> None:
    first = validate_stage_0_3_source_run(source_run).contract
    target = source_run / "baselines" / "prompting" / "positive_descriptions.json"
    target.write_bytes(b"b" * 7 + b"more")
    second = validate_stage_0_3_source_run(source_run).contract
    assert (
        second.artifact_byte_sizes["baselines/"]
        > first.artifact_byte_sizes["baselines/"]
    )
    assert (
        second.artifact_entry_counts["baselines/"]
        == first.artifact_entry_counts["baselines/"]
    )
    assert second.artifact_hashes["baselines/"] != first.artifact_hashes["baselines/"]


def test_metric_sensitivity_add_empty_subdir(source_run: Path) -> None:
    first = validate_stage_0_3_source_run(source_run).contract
    (source_run / "baselines" / "prompting" / "newsub").mkdir()
    second = validate_stage_0_3_source_run(source_run).contract
    assert (
        second.artifact_byte_sizes["baselines/"]
        == first.artifact_byte_sizes["baselines/"]
    )
    assert (
        second.artifact_entry_counts["baselines/"]
        == first.artifact_entry_counts["baselines/"] + 1
    )
    assert second.artifact_hashes["baselines/"] != first.artifact_hashes["baselines/"]


def test_metric_sensitivity_add_file(source_run: Path) -> None:
    first = validate_stage_0_3_source_run(source_run).contract
    (source_run / "baselines" / "prompting" / "negative_descriptions.json").write_bytes(
        b"n" * 3
    )
    second = validate_stage_0_3_source_run(source_run).contract
    assert (
        second.artifact_byte_sizes["baselines/"]
        == first.artifact_byte_sizes["baselines/"] + 3
    )
    assert (
        second.artifact_entry_counts["baselines/"]
        == first.artifact_entry_counts["baselines/"] + 1
    )
    assert second.artifact_hashes["baselines/"] != first.artifact_hashes["baselines/"]


def test_validation_is_deterministic(source_run: Path) -> None:
    a = validate_stage_0_3_source_run(source_run).contract
    b = validate_stage_0_3_source_run(source_run).contract
    assert a.artifact_byte_sizes == b.artifact_byte_sizes
    assert a.artifact_entry_counts == b.artifact_entry_counts
    assert a.artifact_hashes == b.artifact_hashes
    assert contract_hash(a.to_json_dict()) == contract_hash(b.to_json_dict())


def test_contract_hash_excludes_itself_and_is_deterministic(
    source_contract: BpbAdvSourceContract,
) -> None:
    payload = source_contract.to_json_dict()
    assert "contract_hash" in payload
    h1 = contract_hash(payload)
    payload_with_bogus = dict(payload)
    payload_with_bogus["contract_hash"] = "deadbeef"
    assert contract_hash(payload_with_bogus) == h1
    assert contract_hash(payload) == h1


def test_schema_version_change_changes_contract_hash(
    source_contract: BpbAdvSourceContract,
) -> None:
    payload = source_contract.to_json_dict()
    bumped = dict(payload)
    bumped["schema_version"] = 999
    assert contract_hash(bumped) != contract_hash(payload)


def test_persisted_contract_has_v6_schema_seed_layer_and_split_fields(
    source_contract: BpbAdvSourceContract,
    source_run: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "out" / "source_contract.json"

    write_source_contract(out, source_contract)
    raw = out.read_text(encoding="utf-8")

    # Exact serialized bytes (pins format across the atomic-write switch).
    assert raw == _expected_serialized(source_contract)
    loaded = json.loads(raw)
    assert loaded["schema_version"] == 6
    assert loaded["direction_training_layer"] == 6
    assert loaded["seed_schema_version"] == 2
    assert loaded["seed_fingerprint"] == _seed_fingerprint(source_run)
    assert "artifact_byte_sizes" in loaded
    assert "artifact_entry_counts" in loaded
    assert "artifact_sizes" not in loaded
    assert loaded["artifact_entry_counts"] == {"baselines/": _BASELINES_ENTRY_COUNT}
    assert loaded["artifact_byte_sizes"]["baselines/"] == _BASELINES_FILE_BYTES


def test_stage_validate_source_writes_v6_contract_through_driver(tmp_path: Path) -> None:
    """Drive the REAL `_stage_validate_source` (not `write_source_contract` directly)
    over a multi-file nested baselines/ tree, and confirm the persisted contract is
    schema v6 with semantic seed/direction identity and correct split metrics.
    Re-running the stage validates the immutable file byte-identically."""
    src = tmp_path / "source"
    _write_source_run(src)
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(src),
        output_root=str(tmp_path / "out"),
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        run_tag="run",
        target_models=["decoder_only"],
        base_methods=["diffmean_decoder_only"],
    )
    contract_path = AxBenchBpbAdvPaths(
        output_root=cfg.output_root, run_tag="run"
    ).source_contract()

    bpb_adv_cli._stage_validate_source(cfg)
    first = contract_path.read_bytes()
    bpb_adv_cli._stage_validate_source(cfg)  # the second (prepare_attack_rows) write
    second = contract_path.read_bytes()

    assert first == second  # dual-write is byte-identical through the driver
    loaded = json.loads(first)
    assert loaded["schema_version"] == 6
    assert loaded["direction_training_layer"] == 6
    assert loaded["seed_schema_version"] == 2
    assert loaded["seed_fingerprint"] == _seed_fingerprint(src)
    assert "artifact_sizes" not in loaded
    assert loaded["artifact_entry_counts"] == {"baselines/": _BASELINES_ENTRY_COUNT}
    assert loaded["artifact_byte_sizes"]["baselines/"] == _BASELINES_FILE_BYTES
    assert set(loaded["artifact_byte_sizes"]) == _ALL_ARTIFACT_NAMES
