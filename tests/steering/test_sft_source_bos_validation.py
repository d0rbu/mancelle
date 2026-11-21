"""Artifact-derived source-BOS validation for conditioned SFT."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.distributed as dist

import scripts.preprocess.precompute_token_boundaries as token_boundaries
import steering.conditional.encoder_decoder.sft_features as sft_features
from errors import DataLoadError
from steering._common.ddp import AlignedRankFailure, rank_synchronized_stage
from steering.feature_io import TensorKey, save_features
from steering.parquet_io import detect_source_bos_char_len_from_data


def _write_text_parquet(
    path: Path,
    texts: list[str],
    *,
    row_group_size: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": pa.array(texts, type=pa.string())}),
        path,
        row_group_size=row_group_size,
    )


def _detector_data(
    *,
    doc_ids: list[int],
    char_ends: list[int],
    doc_token_counts: list[int],
    context_window: int = 64,
) -> dict[str, object]:
    return {
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "context_window": context_window,
    }


def _write_feature_family(
    feature_dir: Path,
    parquet_dir: Path,
    stem: str,
    *,
    text: str,
    bos_char_len: int,
) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    save_features(
        {
            "offsets": t.tensor([0, 1], dtype=t.int64),
            "feature_ids": t.tensor([7], dtype=t.int32),
            "feature_mags": t.tensor([1.0], dtype=t.float32),
            "doc_ids": t.tensor([0], dtype=t.int32),
            "segment_indices": t.tensor([0], dtype=t.int32),
            "segment_positions": t.tensor([0], dtype=t.int32),
            "doc_token_counts": t.tensor([4], dtype=t.int32),
            "segment_char_starts": t.tensor([0], dtype=t.int32),
            "segment_char_ends": t.tensor(
                [len(text) + bos_char_len],
                dtype=t.int32,
            ),
            "n_docs": 1,
            "n_segments": 1,
            "layer": 8,
            "context_window": 64,
            "pooling": "max_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        },
        feature_dir / f"{stem}_features",
    )
    _write_text_parquet(parquet_dir / f"{stem}.parquet", [text])


def test_precompute_script_reexports_the_shared_detector() -> None:
    assert (
        token_boundaries._detect_source_bos_char_len_from_data
        is detect_source_bos_char_len_from_data
    )


def test_detector_uses_last_segment_and_python_character_length(tmp_path: Path) -> None:
    parquet = tmp_path / "unicode.parquet"
    _write_text_parquet(parquet, ["é🙂"])

    detected = detect_source_bos_char_len_from_data(
        _detector_data(
            doc_ids=[0, 0],
            char_ends=[3, 7],
            doc_token_counts=[3],
        ),
        parquet,
    )

    assert detected == 5


def test_detector_skips_full_and_negative_candidates(tmp_path: Path) -> None:
    parquet = tmp_path / "candidates.parquet"
    texts = ["full", "negative", "usable"]
    _write_text_parquet(parquet, texts)

    detected = detect_source_bos_char_len_from_data(
        _detector_data(
            doc_ids=[0, 1, 2],
            char_ends=[9, len(texts[1]) - 1, len(texts[2]) + 2],
            doc_token_counts=[64, 4, 4],
        ),
        parquet,
    )

    assert detected == 2


def test_detector_never_looks_past_the_first_row_group(tmp_path: Path) -> None:
    parquet = tmp_path / "row-groups.parquet"
    _write_text_parquet(parquet, ["full", "usable"], row_group_size=1)

    with pytest.raises(ValueError, match="No short document found in the first row group"):
        detect_source_bos_char_len_from_data(
            _detector_data(
                doc_ids=[0, 1],
                char_ends=[9, 11],
                doc_token_counts=[64, 4],
            ),
            parquet,
        )


def test_sft_family_match_uses_first_shard_and_exact_load_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    _write_feature_family(
        feature_dir,
        parquet_dir,
        "shard_00001",
        text="later",
        bos_char_len=9,
    )
    _write_feature_family(
        feature_dir,
        parquet_dir,
        "shard_00000",
        text="first",
        bos_char_len=5,
    )
    observed: list[tuple[Path, set[str] | None]] = []
    real_load = sft_features.load_features

    def recording_load(path, *, keys=None):
        observed.append((Path(path), keys))
        return real_load(path, keys=keys)

    monkeypatch.setattr(sft_features, "load_features", recording_load)

    assert (
        sft_features.assert_sft_source_bos_char_len_matches(
            str(feature_dir),
            str(parquet_dir),
            configured=5,
            source="train",
        )
        == 5
    )
    assert observed == [
        (
            feature_dir / "shard_00000_features.safetensors.zst",
            {
                TensorKey.DOC_IDS,
                TensorKey.SEGMENT_CHAR_ENDS,
                TensorKey.DOC_TOKEN_COUNTS,
            },
        )
    ]


def test_sft_family_mismatch_names_values_role_and_paths(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    _write_feature_family(
        feature_dir,
        parquet_dir,
        "shard_00000",
        text="sample",
        bos_char_len=2,
    )

    with pytest.raises(DataLoadError) as exc_info:
        sft_features.assert_sft_source_bos_char_len_matches(
            str(feature_dir),
            str(parquet_dir),
            configured=5,
            source="validation",
        )

    message = str(exc_info.value)
    assert "validation feature_source_bos_char_len mismatch" in message
    assert "detected=2" in message
    assert "configured=5" in message
    assert str(feature_dir / "shard_00000_features.safetensors.zst") in message
    assert str(parquet_dir / "shard_00000.parquet") in message


def test_sft_family_detector_failure_is_wrapped_with_artifact_context(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    _write_feature_family(
        feature_dir,
        parquet_dir,
        "shard_00000",
        text="sample",
        bos_char_len=5,
    )
    (parquet_dir / "shard_00000.parquet").unlink()

    with pytest.raises(DataLoadError, match="Could not verify conditioned SFT train") as exc_info:
        sft_features.assert_sft_source_bos_char_len_matches(
            str(feature_dir),
            str(parquet_dir),
            configured=5,
            source="train",
        )

    assert exc_info.value.__cause__ is not None


def test_sft_family_without_feature_shards_fails_closed(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()

    with pytest.raises(DataLoadError, match="no feature shard files found"):
        sft_features.assert_sft_source_bos_char_len_matches(
            str(feature_dir),
            str(tmp_path / "parquet"),
            configured=5,
            source="train",
        )


def _gloo_bos_failure_worker(
    rank: int,
    world_size: int,
    init_file: str,
    feature_dir: str,
    parquet_dir: str,
    result_queue,
) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
        )
        try:
            with rank_synchronized_stage(
                "SFT feature segmentation",
                device="cpu",
            ):
                sft_features.assert_sft_source_bos_char_len_matches(
                    feature_dir,
                    parquet_dir,
                    configured=5 if rank == 0 else 0,
                    source="train",
                )
        except AlignedRankFailure as exc:
            result_queue.put((rank, str(exc)))
        else:
            result_queue.put((rank, "survived"))
    except BaseException as exc:  # noqa: BLE001 - report child failure
        result_queue.put((rank, f"{type(exc).__name__}: {exc}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed Gloo is unavailable",
)
def test_one_rank_bos_mismatch_aborts_every_rank(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    _write_feature_family(
        feature_dir,
        parquet_dir,
        "shard_00000",
        text="sample",
        bos_char_len=5,
    )
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo_bos_failure")
    processes = [
        context.Process(
            target=_gloo_bos_failure_worker,
            args=(
                rank,
                2,
                init_file,
                str(feature_dir),
                str(parquet_dir),
                result_queue,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [result_queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)

    assert [process.exitcode for process in processes] == [0, 0]
    messages = dict(results)
    assert set(messages) == {0, 1}
    assert all("SFT feature segmentation" in message for message in messages.values())
    assert all("rank 1" in message for message in messages.values())
    assert all("DataLoadError" in message for message in messages.values())
    assert all("detected=5" in message for message in messages.values())
    assert all("configured=0" in message for message in messages.values())
