"""Fail-loud validation guards for ``steering.extract_features.process_shard``."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import steering.extract_features as ef
from errors import ValidationError
from steering.extract_features import PoolingMode
from steering.shard_validation import ValidationResult
from tests.steering.test_extract_features_integration import (
    TinyHookedTransformer,
    TinySAE,
)


class _RecordingSaver:
    def __init__(self) -> None:
        self.submissions: list[tuple[dict, str]] = []

    def submit(self, data: dict, out_stem: str) -> None:
        self.submissions.append((data, out_stem))


def _make_parquet(path: Path) -> str:
    pq.write_table(pa.table({"text": ["validation guard regression"]}), path)
    return str(path)


def _process_shard(
    tmp_path: Path,
    *,
    background_saver: _RecordingSaver | None = None,
) -> ef.ShardResult:
    return ef.process_shard(
        _make_parquet(tmp_path / "shard_00000.parquet"),
        TinyHookedTransformer(),  # ty: ignore[invalid-argument-type]
        TinySAE(),  # ty: ignore[invalid-argument-type]
        layer=1,
        output_dir=str(tmp_path),
        context_window=64,
        pooling=PoolingMode.max_pool,
        segment_size=16,
        gpu_batch_size=2,
        cpu_batch_size=2,
        pipeline_post=False,
        background_saver=background_saver,  # ty: ignore[invalid-argument-type]
    )


def test_invalid_in_memory_shard_raises_before_background_submit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = _RecordingSaver()
    monkeypatch.setattr(
        ef,
        "validate_shard_data",
        lambda _data: ValidationResult(
            valid=False,
            violations=["injected in-memory corruption"],
        ),
    )

    with pytest.raises(
        ValidationError,
        match=r"shard_00000.*injected in-memory corruption",
    ):
        _process_shard(tmp_path, background_saver=saver)

    assert saver.submissions == []
    assert not ef.feature_path(tmp_path, "shard_00000").exists()


def test_invalid_in_memory_shard_raises_before_direct_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_calls: list[tuple[tuple, dict]] = []
    real_save_features = ef.save_features

    def record_save(*args, **kwargs):
        save_calls.append((args, kwargs))
        return real_save_features(*args, **kwargs)

    monkeypatch.setattr(
        ef,
        "validate_shard_data",
        lambda _data: ValidationResult(
            valid=False,
            violations=["injected direct-save corruption"],
        ),
    )
    monkeypatch.setattr(ef, "save_features", record_save)

    with pytest.raises(
        ValidationError,
        match=r"shard_00000.*injected direct-save corruption",
    ):
        _process_shard(tmp_path)

    assert save_calls == []
    assert not ef.feature_path(tmp_path, "shard_00000").exists()


def test_invalid_saved_shard_raises_instead_of_returning_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ef,
        "validate_shard_data",
        lambda _data: ValidationResult(valid=True),
    )
    monkeypatch.setattr(
        ef,
        "validate_shard",
        lambda _path: ValidationResult(
            valid=False,
            violations=["injected post-write corruption"],
        ),
    )

    with pytest.raises(
        ValidationError,
        match=r"shard_00000_features.*injected post-write corruption",
    ):
        _process_shard(tmp_path)
