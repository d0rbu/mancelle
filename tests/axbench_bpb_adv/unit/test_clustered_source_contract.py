from __future__ import annotations

import hashlib
import inspect
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

from steering._common.strict_json import canonical_json_str_fallback
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
)
from steering.eval.axbench_bpb_adv import source_validation as sv
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
    contract_hash,
    write_source_contract,
)
from steering.eval.cluster_feature_contract import FEATURE_CONTRACT_SCHEMA_VERSION

_SEED_FINGERPRINT = "a" * 64
_FEATURE_FINGERPRINT = "b" * 64
_CANDIDATE_PROVENANCE_SHA256 = "c" * 64
_POSITIVE_CORPUS_SHA256 = "d" * 64
_NEGATIVE_CORPUS_SHA256 = "e" * 64
_STAGE2_NAMES = {
    "concept_features.safetensors",
    "concept_features_neg.safetensors",
}
_CLUSTER_FIELDS = {
    "feature_contract_schema_version",
    "feature_contract_fingerprint",
    "candidate_provenance_sha256",
    "stage2_feature_contract_fingerprints",
    "baseline_training_corpus_sha256",
}
_FROZEN_V6_KEYS = {
    "artifact_byte_sizes",
    "artifact_entry_counts",
    "artifact_hashes",
    "bpb_only",
    "contract_hash",
    "direction_training_layer",
    "n_dropped",
    "schema_version",
    "seed_fingerprint",
    "seed_schema_version",
    "seed_token_semantics",
    "source_eval",
    "source_run_dir",
}


def _clustered_contract(**overrides: Any) -> BpbAdvSourceContract:
    kwargs: dict[str, Any] = {
        "seed_fingerprint": _SEED_FINGERPRINT,
        "schema_version": CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
        "feature_contract_schema_version": FEATURE_CONTRACT_SCHEMA_VERSION,
        "feature_contract_fingerprint": _FEATURE_FINGERPRINT,
        "candidate_provenance_sha256": _CANDIDATE_PROVENANCE_SHA256,
        "stage2_feature_contract_fingerprints": dict.fromkeys(_STAGE2_NAMES, _FEATURE_FINGERPRINT),
        "baseline_training_corpus_sha256": {
            "positive": _POSITIVE_CORPUS_SHA256,
            "negative": _NEGATIVE_CORPUS_SHA256,
        },
        "direction_training_layer": 6,
    }
    kwargs.update(overrides)
    return BpbAdvSourceContract(**kwargs)


def _rendered_contract(contract: BpbAdvSourceContract) -> bytes:
    return (
        json.dumps(
            contract.to_json_dict(),
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def test_schema_versions_do_not_collide() -> None:
    assert SOURCE_CONTRACT_SCHEMA_VERSION == 6
    assert CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION == 7


@pytest.mark.parametrize("schema_version", [3, 4, 5, 8])
def test_source_contract_rejects_schema_outside_v6_v7(
    schema_version: int,
) -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"schema v6.*or v7",
    ):
        BpbAdvSourceContract(
            seed_fingerprint=_SEED_FINGERPRINT,
            schema_version=schema_version,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"seed_schema_version": SEED_SCHEMA_VERSION + 1},
            "require AxBench seed schema v2",
        ),
        (
            {"seed_token_semantics": f"{SEED_TOKEN_SEMANTICS}-stale"},
            "seed token semantics mismatch",
        ),
        (
            {"seed_fingerprint": "A" * 64},
            "seed fingerprint must be SHA-256",
        ),
    ],
)
def test_source_contract_preserves_seed_contract_rejections(
    overrides: dict[str, Any],
    match: str,
) -> None:
    kwargs: dict[str, Any] = {"seed_fingerprint": _SEED_FINGERPRINT}
    kwargs.update(overrides)
    with pytest.raises(AxBenchBpbAdvError, match=match):
        BpbAdvSourceContract(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("source_eval", "other", "source_eval"),
        ("bpb_only", False, "must be true"),
        ("bpb_only", 1, "must be true"),
    ],
)
def test_current_contract_rejects_invalid_fixed_fields(
    field: str,
    value: Any,
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=match):
        BpbAdvSourceContract(
            seed_fingerprint=_SEED_FINGERPRINT,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("feature_contract_schema_version", FEATURE_CONTRACT_SCHEMA_VERSION),
        ("feature_contract_fingerprint", _FEATURE_FINGERPRINT),
        ("candidate_provenance_sha256", _CANDIDATE_PROVENANCE_SHA256),
        (
            "stage2_feature_contract_fingerprints",
            {"concept_features.safetensors": _FEATURE_FINGERPRINT},
        ),
        (
            "baseline_training_corpus_sha256",
            {"positive": _POSITIVE_CORPUS_SHA256},
        ),
    ],
)
def test_unclustered_v6_rejects_each_cluster_field(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="schema v6 cannot carry cluster ownership fields",
    ):
        BpbAdvSourceContract(
            seed_fingerprint=_SEED_FINGERPRINT,
            **{field_name: value},
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"feature_contract_schema_version": None},
            "wrong feature-contract schema version",
        ),
        (
            {
                "feature_contract_schema_version": (
                    FEATURE_CONTRACT_SCHEMA_VERSION + 1
                )
            },
            "wrong feature-contract schema version",
        ),
        (
            {"feature_contract_fingerprint": None},
            "feature_contract_fingerprint must be SHA-256",
        ),
        (
            {"feature_contract_fingerprint": "not-a-sha256"},
            "feature_contract_fingerprint must be SHA-256",
        ),
        (
            {"candidate_provenance_sha256": None},
            "candidate_provenance_sha256 must be SHA-256",
        ),
        (
            {"candidate_provenance_sha256": "f" * 63},
            "candidate_provenance_sha256 must be SHA-256",
        ),
        (
            {"stage2_feature_contract_fingerprints": {}},
            "must bind both Stage 2 feature-contract fingerprints",
        ),
        (
            {
                "stage2_feature_contract_fingerprints": {
                    "concept_features.safetensors": _FEATURE_FINGERPRINT,
                }
            },
            "must bind both Stage 2 feature-contract fingerprints",
        ),
        (
            {
                "stage2_feature_contract_fingerprints": dict.fromkeys(_STAGE2_NAMES, _FEATURE_FINGERPRINT)
                | {"unexpected.safetensors": _FEATURE_FINGERPRINT}
            },
            "must bind both Stage 2 feature-contract fingerprints",
        ),
        (
            {
                "stage2_feature_contract_fingerprints": {
                    "concept_features.safetensors": _FEATURE_FINGERPRINT,
                    "concept_features_neg.safetensors": "f" * 64,
                }
            },
            "Stage 2 fingerprints disagree with the root feature contract",
        ),
        (
            {"baseline_training_corpus_sha256": {}},
            "must bind the positive and negative baseline-training corpora",
        ),
        (
            {
                "baseline_training_corpus_sha256": {
                    "positive": _POSITIVE_CORPUS_SHA256,
                }
            },
            "must bind the positive and negative baseline-training corpora",
        ),
        (
            {
                "baseline_training_corpus_sha256": {
                    "positive": _POSITIVE_CORPUS_SHA256,
                    "negative": _NEGATIVE_CORPUS_SHA256,
                    "other": "f" * 64,
                }
            },
            "must bind the positive and negative baseline-training corpora",
        ),
        (
            {
                "baseline_training_corpus_sha256": {
                    "positive": "not-a-sha256",
                    "negative": _NEGATIVE_CORPUS_SHA256,
                }
            },
            "digest for 'positive' must be SHA-256",
        ),
        (
            {
                "baseline_training_corpus_sha256": {
                    "positive": _POSITIVE_CORPUS_SHA256,
                    "negative": "E" * 64,
                }
            },
            "digest for 'negative' must be SHA-256",
        ),
    ],
)
def test_clustered_v7_rejects_missing_or_malformed_fields(
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match=match):
        _clustered_contract(**overrides)


def test_clustered_v7_to_json_retains_all_cluster_fields() -> None:
    payload = _clustered_contract().to_json_dict()

    assert payload["schema_version"] == 7
    assert payload["direction_training_layer"] == 6
    assert _CLUSTER_FIELDS <= set(payload)
    assert payload["feature_contract_schema_version"] == FEATURE_CONTRACT_SCHEMA_VERSION
    assert payload["feature_contract_fingerprint"] == _FEATURE_FINGERPRINT
    assert payload["candidate_provenance_sha256"] == _CANDIDATE_PROVENANCE_SHA256
    assert set(payload["stage2_feature_contract_fingerprints"]) == _STAGE2_NAMES
    assert payload["baseline_training_corpus_sha256"] == {
        "positive": _POSITIVE_CORPUS_SHA256,
        "negative": _NEGATIVE_CORPUS_SHA256,
    }
    assert payload["contract_hash"] == contract_hash(payload)


def test_unclustered_v6_frozen_byte_identity() -> None:
    contract = BpbAdvSourceContract(
        seed_fingerprint="a" * 64,
        source_run_dir="/src/run",
        artifact_byte_sizes={"concepts.json": 11, "baselines/": 42},
        artifact_entry_counts={"baselines/": 2},
        artifact_hashes={"concepts.json": "b" * 64},
        direction_training_layer=6,
    )

    payload = contract.to_json_dict()
    rendered = (
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
    ).encode("utf-8")

    assert set(payload) == _FROZEN_V6_KEYS
    assert (
        payload["contract_hash"]
        == "a4482de54f07d60cafc929d00bc7ac7ee97e474615a032a9ab600dd191b6ca4d"
    )
    assert (
        hashlib.sha256(rendered).hexdigest()
        == "4042b482a754b5ea9dfaf4093cc2d271dd952f64feb2530827707e04cf26f187"
    )


def test_unclustered_v6_strips_cluster_fields_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hash_inputs: list[dict[str, Any]] = []

    def record_hash(payload: dict[str, Any]) -> str:
        hash_inputs.append(dict(payload))
        return "recorded-after-strip"

    monkeypatch.setattr(sv, "contract_hash", record_hash)

    payload = BpbAdvSourceContract(
        seed_fingerprint=_SEED_FINGERPRINT
    ).to_json_dict()

    assert payload["contract_hash"] == "recorded-after-strip"
    assert len(hash_inputs) == 1
    assert _CLUSTER_FIELDS.isdisjoint(hash_inputs[0])
    assert _CLUSTER_FIELDS.isdisjoint(payload)


def test_unclustered_v6_writer_rejects_changed_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source_contract.json"
    first = BpbAdvSourceContract(
        seed_fingerprint=_SEED_FINGERPRINT,
        source_run_dir="/source/one",
    )
    second = BpbAdvSourceContract(
        seed_fingerprint=_SEED_FINGERPRINT,
        source_run_dir="/source/two",
    )

    write_source_contract(path, first)
    first_bytes = path.read_bytes()
    with pytest.raises(AxBenchBpbAdvError, match="source drift"):
        write_source_contract(path, second)

    assert path.read_bytes() == first_bytes


def test_clustered_v7_writer_identical_replay_is_a_noop(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "source_contract.json"
    contract = _clustered_contract()

    assert write_source_contract(path, contract) == path
    first_bytes = path.read_bytes()
    first_stat = path.stat()
    assert write_source_contract(path, contract) == path
    second_stat = path.stat()

    assert first_bytes == _rendered_contract(contract)
    assert path.read_bytes() == first_bytes
    assert (
        second_stat.st_dev,
        second_stat.st_ino,
        second_stat.st_size,
        second_stat.st_mtime_ns,
        second_stat.st_ctime_ns,
    ) == (
        first_stat.st_dev,
        first_stat.st_ino,
        first_stat.st_size,
        first_stat.st_mtime_ns,
        first_stat.st_ctime_ns,
    )


def test_clustered_v7_writer_rejects_changed_content_with_fresh_run_tag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source_contract.json"
    original = _clustered_contract(source_run_dir="/source/one")
    changed = _clustered_contract(source_run_dir="/source/two")

    write_source_contract(path, original)
    original_bytes = path.read_bytes()

    with pytest.raises(AxBenchBpbAdvError, match="fresh run_tag"):
        write_source_contract(path, changed)

    assert path.read_bytes() == original_bytes


def test_hash_required_file_preserves_all_three_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact"
    calls: list[tuple[str, Path]] = []

    def hash_safetensors(candidate: Path) -> str:
        calls.append(("safetensors", candidate))
        return "safetensors"

    def hash_manifest(candidate: Path) -> str:
        calls.append(("manifest", candidate))
        return "manifest"

    def hash_file(candidate: Path) -> str:
        calls.append(("file", candidate))
        return "file"

    monkeypatch.setattr(sv, "_hash_safetensors_semantic", hash_safetensors)
    monkeypatch.setattr(sv, "_hash_seed_manifest_semantic", hash_manifest)
    monkeypatch.setattr(sv, "_hash_file", hash_file)

    assert sv._hash_required_file("seeds_train.safetensors", path) == "safetensors"
    assert sv._hash_required_file("seeds_eval.safetensors", path) == "safetensors"
    assert sv._hash_required_file("seed_pool_manifest.json", path) == "manifest"
    assert sv._hash_required_file("concepts.json", path) == "file"
    assert calls == [
        ("safetensors", path),
        ("safetensors", path),
        ("manifest", path),
        ("file", path),
    ]


def test_recorded_seed_hash_ignores_safetensors_header_reordering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "seeds_train.safetensors"
    save_file(
        {
            "z_tensor": t.tensor([1, 2], dtype=t.int64),
            "a_tensor": t.tensor([3.5], dtype=t.float32),
        },
        str(path),
        metadata={"z_key": "last", "a_key": "first"},
    )
    raw_before = path.read_bytes()
    recorded_before = sv._hash_required_file("seeds_train.safetensors", path)

    header_length = int.from_bytes(raw_before[:8], byteorder="little")
    header_end = 8 + header_length
    header = json.loads(raw_before[8:header_end])
    metadata = header["__metadata__"]
    header["__metadata__"] = {
        key: metadata[key] for key in reversed(list(metadata))
    }
    reordered_header = {
        key: header[key] for key in reversed(list(header))
    }
    encoded = json.dumps(
        reordered_header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(encoded) <= header_length
    path.write_bytes(
        raw_before[:8]
        + encoded
        + b" " * (header_length - len(encoded))
        + raw_before[header_end:]
    )

    raw_after = path.read_bytes()
    recorded_after = sv._hash_required_file("seeds_train.safetensors", path)

    assert hashlib.sha256(raw_after).digest() != hashlib.sha256(raw_before).digest()
    assert recorded_after == recorded_before


def test_stage2_fingerprints_must_equal_root_not_merely_each_other() -> None:
    """Close a gap a surviving mutation exposed.

    The Codex/lens-B mutation that weakened the Stage-2 check from "both values
    must equal the ROOT fingerprint" to "both values must merely agree with each
    other" survived the ENTIRE suite. The existing malformed fixture pairs one
    correct value with one wrong one, so the two also disagree with each other --
    it cannot tell the two rules apart.

    This case can: both Stage-2 values are equal to each other and both are
    wrong relative to the root fingerprint.
    """

    with pytest.raises(
        AxBenchBpbAdvError,
        match="Stage 2 fingerprints disagree with the root feature contract",
    ):
        _clustered_contract(
            stage2_feature_contract_fingerprints=dict.fromkeys(_STAGE2_NAMES, "f" * 64)
        )


@pytest.mark.parametrize(
    "version",
    [7.0, Decimal(7), Fraction(7), True, "7", None, [7], {7: 1}],
)
def test_clustered_v7_requires_a_strict_int_schema_version(version: Any) -> None:
    """Equality alone lets a non-int serialize to a form this validator rejects.

    ``Decimal(7)`` and ``numpy.int64(7)`` compare equal to 7 and pass an ``==``
    gate, but ``json.dumps(..., default=str)`` writes them as the STRING "7" --
    and string "7" is rejected on read. The write would succeed and the read
    would fail. Unhashable values must also surface as AxBenchBpbAdvError rather
    than a raw TypeError, which is why the membership test uses a tuple.
    """

    with pytest.raises(AxBenchBpbAdvError):
        _clustered_contract(schema_version=version)


@pytest.mark.parametrize("version", [True, 1.0, Decimal(1), "1", None])
def test_clustered_v7_requires_a_strict_int_feature_contract_version(
    version: Any,
) -> None:
    """``True`` compares equal to 1 and would persist as a JSON boolean."""

    with pytest.raises(
        AxBenchBpbAdvError, match="wrong feature-contract schema version"
    ):
        _clustered_contract(feature_contract_schema_version=version)


@pytest.mark.parametrize(
    "field_name", ["stage2_feature_contract_fingerprints", "baseline_training_corpus_sha256"]
)
@pytest.mark.parametrize("bad", [False, 0, [], ""])
def test_unclustered_v6_rejects_falsey_non_mapping_cluster_fields(
    field_name: str, bad: Any
) -> None:
    """Truthiness alone let a malformed non-mapping sit on a v6 contract.

    It could never reach the v6 hash (``to_json_dict`` pops all five), but the
    v6/v7 separation is supposed to be airtight.
    """

    with pytest.raises(
        AxBenchBpbAdvError, match="cannot carry cluster ownership fields"
    ):
        BpbAdvSourceContract(seed_fingerprint=_SEED_FINGERPRINT, **{field_name: bad})


def test_canonical_json_str_fallback_keeps_wide_parameter_annotation() -> None:
    """Pin the annotation itself -- restoring the oracle's narrower one survived.

    main widened ``canonical_json_str_fallback(payload: Any)`` in ae2879cf so
    ``_hash_safetensors_semantic`` can pass a metadata mapping. The oracle still
    has ``dict[str, Any]``. Reverting to the oracle's signature passed all 1225
    tests, so behaviour alone does not guard this named main-drift preservation.
    """

    annotation = inspect.signature(canonical_json_str_fallback).parameters["payload"].annotation
    assert annotation in ("Any", Any), annotation
