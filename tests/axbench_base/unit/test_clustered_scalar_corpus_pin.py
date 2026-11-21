"""Clustered scalar trainers bind directions to immutable corpus bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch.nn as nn

from steering.eval.axbench_base.baselines import _capture as capture_mod
from steering.eval.axbench_base.baselines import diffmean as diffmean_mod
from steering.eval.axbench_base.baselines import linear_probe as linear_probe_mod
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError, Concept


def _concept() -> Concept:
    return Concept(
        concept_id=7,
        feature_index=7,
        layer=22,
        sae_source="test/sae",
        description="test concept",
        corpus_count=1,
        description_source="test",
    )


def test_cluster_pinned_reader_parses_exact_hashed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "training.parquet"
    expected = pa.table({"feature_id": [2, 5], "score": [0.25, 0.75]})
    pq.write_table(expected, path)

    actual, digest = capture_mod._read_cluster_pinned_training_table(path)

    assert actual.equals(expected)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cluster_pinned_reader_rejects_replacement_after_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training.parquet"
    pq.write_table(pa.table({"feature_id": [2]}), path)
    digest_before = capture_mod.stream_hash_file_stable(path)
    pq.write_table(pa.table({"feature_id": [999, 1000]}), path)
    monkeypatch.setattr(
        capture_mod,
        "stream_hash_file_stable",
        lambda _path: digest_before,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="changed between stable hashing and immutable read",
    ):
        capture_mod._read_cluster_pinned_training_table(path)


def test_cluster_pinned_reader_rejects_byte_length_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training.parquet"
    pq.write_table(pa.table({"feature_id": [2]}), path)
    digest = capture_mod.stream_hash_file_stable(path)
    wrong_length_digest = SimpleNamespace(
        path=digest.path,
        sha256=digest.sha256,
        byte_length=digest.byte_length + 1,
    )
    monkeypatch.setattr(
        capture_mod,
        "stream_hash_file_stable",
        lambda _path: wrong_length_digest,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="changed between stable hashing and immutable read",
    ):
        capture_mod._read_cluster_pinned_training_table(path)


def test_cluster_pinned_reader_wraps_stable_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "training.parquet"
    path.write_bytes(b"unreadable")

    def _fail_stable_read(_path: Path):
        raise OSError("synthetic stable-read failure")

    monkeypatch.setattr(
        capture_mod,
        "stream_hash_file_stable",
        _fail_stable_read,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Could not read a stable clustered synth training corpus",
    ):
        capture_mod._read_cluster_pinned_training_table(path)


def test_cluster_pinned_reader_wraps_parse_failure(tmp_path: Path) -> None:
    path = tmp_path / "training.parquet"
    path.write_bytes(b"not a parquet payload")

    with pytest.raises(
        AxBenchBaseError,
        match="Could not parse clustered synth training corpus",
    ):
        capture_mod._read_cluster_pinned_training_table(path)


@pytest.mark.parametrize(
    ("module", "trainer_name"),
    [
        (diffmean_mod, "train_diffmean_all"),
        (linear_probe_mod, "train_linear_probe_all"),
    ],
)
def test_clustered_trainers_use_pinned_reader_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    trainer_name: str,
) -> None:
    class _PinnedReadReached(Exception):
        pass

    def _pinned_read(path: Path) -> tuple[pa.Table, str]:
        assert path == tmp_path / "training.parquet"
        raise _PinnedReadReached

    def _legacy_read(path: Path) -> pa.Table:
        pytest.fail(f"clustered trainer used legacy path read for {path}")

    monkeypatch.setattr(module, "_read_cluster_pinned_training_table", _pinned_read)
    monkeypatch.setattr(module, "pq", SimpleNamespace(read_table=_legacy_read))
    trainer = getattr(module, trainer_name)

    with pytest.raises(_PinnedReadReached):
        trainer(
            cfg=cast(AxBenchBaseConfig, object()),
            target_model=nn.Identity(),
            target_model_name="decoder_only",
            mode="positive",
            concepts=[_concept()],
            training_corpus_path=tmp_path / "training.parquet",
            feature_contract_fingerprint="a" * 64,
        )


@pytest.mark.parametrize(
    ("module", "trainer_name"),
    [
        (diffmean_mod, "train_diffmean_all"),
        (linear_probe_mod, "train_linear_probe_all"),
    ],
)
def test_unclustered_trainers_keep_exact_legacy_path_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    trainer_name: str,
) -> None:
    class _LegacyReadReached(Exception):
        pass

    def _pinned_read(path: Path) -> tuple[pa.Table, str]:
        pytest.fail(f"unclustered trainer used clustered pinned read for {path}")

    def _legacy_read(path: Path) -> pa.Table:
        assert path == tmp_path / "training.parquet"
        raise _LegacyReadReached

    monkeypatch.setattr(module, "_read_cluster_pinned_training_table", _pinned_read)
    monkeypatch.setattr(module, "pq", SimpleNamespace(read_table=_legacy_read))
    trainer = getattr(module, trainer_name)

    with pytest.raises(_LegacyReadReached):
        trainer(
            cfg=cast(AxBenchBaseConfig, object()),
            target_model=nn.Identity(),
            target_model_name="decoder_only",
            mode="positive",
            concepts=[_concept()],
            training_corpus_path=tmp_path / "training.parquet",
        )
