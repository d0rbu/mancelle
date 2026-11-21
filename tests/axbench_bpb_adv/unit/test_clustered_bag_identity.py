"""Cluster fingerprints at precomputed-bag artifact leaf readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from safetensors.torch import save_file

from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    FEATURE_CONTRACT_FINGERPRINT_META_KEY,
    read_ours_gcg_bag,
    read_ours_gcg_scored_prefixes,
    write_ours_gcg_bag,
    write_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import PrecomputedBagFamily

_FINGERPRINT = "a" * 64
_OTHER_FINGERPRINT = "b" * 64


def _write_clustered_bag(
    path: Path,
    *,
    fingerprint: str = _FINGERPRINT,
) -> None:
    write_ours_gcg_bag(
        path,
        FeatureConditioning.from_dict({0: 0, 3: 1}),
        metadata={
            "concept_id": "7",
            FEATURE_CONTRACT_FINGERPRINT_META_KEY: fingerprint,
        },
        vocab_fingerprint="vocab-fingerprint",
    )


def _write_clustered_prefixes(
    path: Path,
    *,
    prefix_ids: list[str] | None = None,
    fingerprint: str = _FINGERPRINT,
) -> None:
    write_ours_gcg_scored_prefixes(
        path,
        ["p2", "p1", "p1"] if prefix_ids is None else prefix_ids,
        concept_id=7,
        mode=SteeringMode.POSITIVE,
        family=PrecomputedBagFamily.FAIR_GCG,
        feature_contract_fingerprint=fingerprint,
    )


def _write_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def test_clustered_bag_reader_accepts_exact_feature_contract_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bag.safetensors"
    _write_clustered_bag(path)

    bag, metadata = read_ours_gcg_bag(
        path,
        expected_feature_contract_fingerprint=_FINGERPRINT,
    )

    assert bag == {0: 0, 3: 1}
    assert metadata[FEATURE_CONTRACT_FINGERPRINT_META_KEY] == _FINGERPRINT


def test_unclustered_bag_reader_rejects_cluster_stamp_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bag.safetensors"
    _write_clustered_bag(path)

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"unclustered reader rejects cluster-stamped precomputed bag artifact",
    ):
        read_ours_gcg_bag(path)


def test_unstamped_legacy_bag_reads_without_expectation(tmp_path: Path) -> None:
    path = tmp_path / "legacy-bag.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([0, 3], dtype=t.int64),
            "buckets": t.tensor([0, 1], dtype=t.int64),
        },
        str(path),
    )

    bag, metadata = read_ours_gcg_bag(path)

    assert bag == {0: 0, 3: 1}
    assert metadata == {}


def test_bag_reader_accepts_stringifiable_non_pathlike_path(tmp_path: Path) -> None:
    path = tmp_path / "legacy-bag.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([0], dtype=t.int64),
            "buckets": t.tensor([1], dtype=t.int64),
        },
        str(path),
    )

    class StringifiablePath:
        def __str__(self) -> str:
            return str(path)

    bag, metadata = read_ours_gcg_bag(StringifiablePath())

    assert bag == {0: 1}
    assert metadata == {}


def test_clustered_bag_reader_mismatch_names_expected_actual_and_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bag.safetensors"
    _write_clustered_bag(path)

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        read_ours_gcg_bag(
            path,
            expected_feature_contract_fingerprint=_OTHER_FINGERPRINT,
        )

    message = str(exc_info.value)
    assert "feature-contract fingerprint mismatch" in message
    assert str(path) in message
    assert f"expected={_OTHER_FINGERPRINT}" in message
    assert f"actual='{_FINGERPRINT}'" in message


def test_clustered_bag_reader_rejects_missing_stamp_when_expected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bag.safetensors"
    write_ours_gcg_bag(
        path,
        FeatureConditioning.from_dict({0: 0}),
        metadata={"concept_id": "7"},
        vocab_fingerprint="vocab-fingerprint",
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"feature-contract fingerprint mismatch: expected=.*actual=None",
    ):
        read_ours_gcg_bag(
            path,
            expected_feature_contract_fingerprint=_FINGERPRINT,
        )


def test_scored_prefix_artifact_stamps_and_validates_cluster_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prefixes.json"
    _write_clustered_prefixes(path)

    assert read_ours_gcg_scored_prefixes(
        path,
        expected_feature_contract_fingerprint=_FINGERPRINT,
    ) == ["p1", "p2"]
    payload = json.loads(path.read_text())
    assert payload[FEATURE_CONTRACT_FINGERPRINT_META_KEY] == _FINGERPRINT


def test_scored_prefix_reader_mismatch_names_expected_actual_and_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prefixes.json"
    _write_clustered_prefixes(path)

    with pytest.raises(AxBenchBpbAdvError) as exc_info:
        read_ours_gcg_scored_prefixes(
            path,
            expected_feature_contract_fingerprint=_OTHER_FINGERPRINT,
        )

    message = str(exc_info.value)
    assert "feature-contract fingerprint mismatch" in message
    assert str(path) in message
    assert f"expected={_OTHER_FINGERPRINT}" in message
    assert f"actual='{_FINGERPRINT}'" in message


def test_unclustered_scored_prefix_reader_rejects_cluster_stamp_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prefixes.json"
    _write_clustered_prefixes(path)

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"unclustered reader rejects cluster-stamped scored-prefix artifact",
    ):
        read_ours_gcg_scored_prefixes(path)


def test_clustered_scored_prefix_reader_rejects_missing_stamp_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prefixes.json"
    write_ours_gcg_scored_prefixes(
        path,
        ["p1"],
        concept_id=7,
        mode=SteeringMode.POSITIVE,
        family=PrecomputedBagFamily.FAIR_GCG,
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"feature-contract fingerprint mismatch: expected=.*actual=None",
    ):
        read_ours_gcg_scored_prefixes(
            path,
            expected_feature_contract_fingerprint=_FINGERPRINT,
        )


def test_clustered_scored_prefix_replay_is_noop_and_changed_content_is_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prefixes.json"
    _write_clustered_prefixes(path)
    original = path.read_bytes()
    original_identity = _write_identity(path)

    _write_clustered_prefixes(path, prefix_ids=["p1", "p2"])
    assert path.read_bytes() == original
    assert _write_identity(path) == original_identity

    with pytest.raises(AxBenchBpbAdvError, match="content changed on resume"):
        _write_clustered_prefixes(path, prefix_ids=["p1", "p3"])
    assert path.read_bytes() == original
    assert _write_identity(path) == original_identity

    with pytest.raises(AxBenchBpbAdvError, match="content changed on resume"):
        _write_clustered_prefixes(path, fingerprint=_OTHER_FINGERPRINT)
    assert path.read_bytes() == original
    assert _write_identity(path) == original_identity
