"""Conditioned-SFT startup discovery and checkpoint identity wiring."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import OmegaConf

from errors import DataLoadError
from nanochat.common import DummyWandb
from nanochat.flash_attention import FlashAttnImpl
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig
from steering.conditional.encoder_decoder.sft_features import SFTFeatureLoadReport
from steering.feature_io import save_features
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import SFT_SEGMENT_MODE_META_KEY
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.encdec_test_helpers import N_EMBD, N_SAE_FEATURES, NUM_BUCKETS, build_tiny_model
from tests.steering.test_encdec_sft_conditioned_integration import (
    MultiTurnTask,
    SFTMockTokenizer,
    build_synthetic_features,
    build_synthetic_sft_store,
)


@contextmanager
def _loguru_sink() -> Iterator[StringIO]:
    """Capture real loguru emissions (INFO and above) through a sink."""
    sink = StringIO()
    handler_id = sft.logger.add(sink, format="{level} | {message}", level="INFO")
    try:
        yield sink
    finally:
        sft.logger.remove(handler_id)


def _feature_dir(
    tmp_path: Path,
    mode: SegmentMode,
    *,
    phase: DatasetPhase = DatasetPhase.SAFETY_SFT,
) -> str:
    return str(
        build_feature_dir(
            root=tmp_path / "features",
            segment_size=128,
            layer=8,
            activation_type=ActivationType.residual,
            pooling=PoolingMode.max_pool,
            width=SAEWidth.W_16K,
            model="gemma-2-2b",
            dataset_phase=phase,
            segment_mode=mode,
        )
    )


def test_train_val_mismatch_names_both_directories_and_modes() -> None:
    """The train/val disagreement raises first and logs nothing before it.

    ``/features/train`` is deliberately non-canonical: if the path cross-check
    ran before the train/val comparison it would emit its "Non-canonical"
    WARNING, so the empty sink pins the ordering as well as the message.
    """
    with _loguru_sink() as sink:
        with pytest.raises(
            DataLoadError,
            match=(
                r"same segmentation mode as training.*"
                r"/features/train.*fixed_tokens.*"
                r"/features/val/turn_delimited.*turn_delimited_fixed_tokens"
            ),
        ):
            sft._finalize_sft_segment_mode(
                SegmentMode.fixed_tokens,
                SegmentMode.turn_delimited_fixed_tokens,
                train_feature_dir="/features/train",
                val_feature_dir="/features/val/turn_delimited",
                resuming=False,
                checkpoint_meta=None,
                checkpoint_dir="/checkpoints/sft-run",
            )

    assert sink.getvalue() == ""


@pytest.mark.parametrize("mode", list(SegmentMode))
def test_matching_val_store_is_accepted(tmp_path: Path, mode: SegmentMode) -> None:
    assert (
        sft._finalize_sft_segment_mode(
            mode,
            mode,
            train_feature_dir=_feature_dir(tmp_path / "train", mode),
            val_feature_dir=_feature_dir(
                tmp_path / "val",
                mode,
                phase=DatasetPhase.SAFETY_SFT_VAL,
            ),
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )
        is mode
    )


def test_finalize_logs_discovered_mode(tmp_path: Path) -> None:
    feature_dir = _feature_dir(tmp_path, SegmentMode.turn_delimited_fixed_tokens)

    with _loguru_sink() as sink:
        mode = sft._finalize_sft_segment_mode(
            SegmentMode.turn_delimited_fixed_tokens,
            None,
            train_feature_dir=feature_dir,
            val_feature_dir=None,
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )

    assert mode is SegmentMode.turn_delimited_fixed_tokens
    assert sink.getvalue().splitlines() == [
        "INFO | SFT feature segmentation mode: turn_delimited_fixed_tokens"
    ]


def test_non_canonical_feature_dir_is_accepted_with_warning(tmp_path: Path) -> None:
    feature_dir = str(tmp_path / "custom_features")

    with _loguru_sink() as sink:
        mode = sft._finalize_sft_segment_mode(
            SegmentMode.turn_delimited_fixed_tokens,
            None,
            train_feature_dir=feature_dir,
            val_feature_dir=None,
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )

    assert mode is SegmentMode.turn_delimited_fixed_tokens
    lines = sink.getvalue().splitlines()
    warnings = [line for line in lines if line.startswith("WARNING | ")]
    assert len(warnings) == 1
    assert "Non-canonical conditioned-SFT feature directory accepted" in warnings[0]
    assert feature_dir in warnings[0]
    assert "INFO | SFT feature segmentation mode: turn_delimited_fixed_tokens" in lines


@pytest.mark.parametrize(
    ("path_mode", "artifact_mode"),
    [
        (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
        (SegmentMode.turn_delimited_fixed_tokens, SegmentMode.fixed_tokens),
    ],
)
def test_canonical_feature_path_must_match_artifact_mode(
    tmp_path: Path,
    path_mode: SegmentMode,
    artifact_mode: SegmentMode,
) -> None:
    feature_dir = _feature_dir(tmp_path, path_mode)
    with pytest.raises(
        DataLoadError,
        match=(
            rf"directory .*{path_mode.value}.*artifacts record "
            rf"'{artifact_mode.value}'.*re-extract into the canonical output_dir"
        ),
    ):
        sft._finalize_sft_segment_mode(
            artifact_mode,
            None,
            train_feature_dir=feature_dir,
            val_feature_dir=None,
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )


@pytest.mark.parametrize("mode", list(SegmentMode))
def test_canonical_feature_path_accepts_matching_artifact_mode(
    tmp_path: Path,
    mode: SegmentMode,
) -> None:
    feature_dir = _feature_dir(tmp_path, mode)
    assert (
        sft._finalize_sft_segment_mode(
            mode,
            None,
            train_feature_dir=feature_dir,
            val_feature_dir=None,
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )
        is mode
    )


def test_symlinked_canonical_path_is_checked_after_resolution(tmp_path: Path) -> None:
    canonical = Path(
        _feature_dir(tmp_path / "canonical", SegmentMode.turn_delimited_fixed_tokens)
    )
    canonical.mkdir(parents=True)
    alias = tmp_path / "feature_alias"
    alias.symlink_to(canonical, target_is_directory=True)

    with pytest.raises(
        DataLoadError,
        match=r"feature_alias.*turn_delimited_fixed_tokens.*fixed_tokens",
    ):
        sft._finalize_sft_segment_mode(
            SegmentMode.fixed_tokens,
            None,
            train_feature_dir=str(alias),
            val_feature_dir=None,
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )


@pytest.mark.parametrize(
    ("path_mode", "artifact_mode"),
    [
        (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
        (SegmentMode.turn_delimited_fixed_tokens, SegmentMode.fixed_tokens),
    ],
)
def test_validation_feature_path_must_match_artifact_mode(
    tmp_path: Path,
    path_mode: SegmentMode,
    artifact_mode: SegmentMode,
) -> None:
    with pytest.raises(
        DataLoadError,
        match=(
            rf"directory .*{path_mode.value}.*artifacts record "
            rf"'{artifact_mode.value}'.*re-extract into the canonical output_dir"
        ),
    ):
        sft._finalize_sft_segment_mode(
            artifact_mode,
            artifact_mode,
            train_feature_dir=_feature_dir(tmp_path / "train", artifact_mode),
            val_feature_dir=_feature_dir(
                tmp_path / "val",
                path_mode,
                phase=DatasetPhase.SAFETY_SFT_VAL,
            ),
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )


def test_validation_leaf_is_not_checked_without_a_validation_store(
    tmp_path: Path,
) -> None:
    """``eval_every>0`` with an empty val split resolves a dir but loads no store.

    ``main()`` then passes ``val_mode=None`` alongside a real
    ``val_feature_dir``; there is no artifact mode to cross-check the leaf
    against, so the run must proceed silently rather than trip a guard.
    """
    mode = SegmentMode.turn_delimited_fixed_tokens
    with _loguru_sink() as sink:
        resolved = sft._finalize_sft_segment_mode(
            mode,
            None,
            train_feature_dir=_feature_dir(tmp_path / "train", mode),
            val_feature_dir=_feature_dir(
                tmp_path / "val",
                SegmentMode.fixed_tokens,
                phase=DatasetPhase.SAFETY_SFT_VAL,
            ),
            resuming=False,
            checkpoint_meta=None,
            checkpoint_dir="/checkpoints/sft-run",
        )

    assert resolved is mode
    assert sink.getvalue().splitlines() == [
        f"INFO | SFT feature segmentation mode: {mode.value}"
    ]


@pytest.mark.parametrize(
    ("feature_mode", "checkpoint_meta"),
    [
        (SegmentMode.turn_delimited_fixed_tokens, {}),
        (
            SegmentMode.fixed_tokens,
            {SFT_SEGMENT_MODE_META_KEY: SegmentMode.turn_delimited_fixed_tokens.value},
        ),
    ],
)
def test_resume_rejects_fixed_turn_mismatch_in_both_directions(
    tmp_path: Path,
    feature_mode: SegmentMode,
    checkpoint_meta: dict[str, str],
) -> None:
    recorded = checkpoint_meta.get(
        SFT_SEGMENT_MODE_META_KEY,
        SegmentMode.fixed_tokens.value,
    )
    with pytest.raises(
        ValueError,
        match=(
            rf"/checkpoints/sft-run.*{recorded}.*{feature_mode.value}.*"
            r"recorded segmentation mode"
        ),
    ):
        sft._finalize_sft_segment_mode(
            feature_mode,
            None,
            train_feature_dir=_feature_dir(tmp_path, feature_mode),
            val_feature_dir=None,
            resuming=True,
            checkpoint_meta=checkpoint_meta,
            checkpoint_dir="/checkpoints/sft-run",
        )


@dataclass
class _MainHarness:
    train_feature_dir: Path
    val_feature_dir: Path | None
    train_parquet_dir: Path
    val_parquet_dir: Path | None
    saved_meta: list[dict]
    saved_dirs: list[str]
    finalize_calls: list[tuple[tuple, dict]]
    bos_checks: list[tuple[str, str, int, str]]
    load_observations: list[tuple[str, str, bool | None]]
    sink: StringIO


def _write_real_feature_family(
    feature_dir: Path,
    parquet_dir: Path,
    features: dict,
    *,
    mode: SegmentMode,
) -> None:
    offsets = [0]
    feature_ids: list[int] = []
    feature_mags: list[float] = []
    doc_ids: list[int] = []
    segment_indices: list[int] = []
    segment_positions: list[int] = []
    segment_char_starts: list[int] = []
    segment_char_ends: list[int] = []
    doc_token_counts: list[int] = []
    for doc_id, segments in sorted(features.items()):
        doc_token_counts.append(max(segment.char_end for segment in segments))
        for segment in segments:
            feature_ids.extend(segment.feature_ids.tolist())
            feature_mags.extend(segment.feature_mags.tolist())
            offsets.append(len(feature_ids))
            doc_ids.append(doc_id)
            segment_indices.append(segment.segment_idx)
            segment_positions.append(segment.position)
            segment_char_starts.append(segment.char_start)
            segment_char_ends.append(segment.char_end)
    save_features(
        {
            "offsets": t.tensor(offsets, dtype=t.int64),
            "feature_ids": t.tensor(feature_ids, dtype=t.int32),
            "feature_mags": t.tensor(feature_mags, dtype=t.float32),
            "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(segment_indices, dtype=t.int32),
            "segment_positions": t.tensor(segment_positions, dtype=t.int32),
            "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
            "segment_char_starts": t.tensor(segment_char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(segment_char_ends, dtype=t.int32),
            "n_docs": len(features),
            "n_segments": len(doc_ids),
            "layer": 8,
            "context_window": 1024,
            "pooling": "max_pool",
            "segment_size": 128,
            "segment_mode": mode.value,
        },
        feature_dir / "shard_00000_features",
    )
    pq.write_table(
        pa.table(
            {
                "original_index": sorted(features),
                "text": [f"document-{index}" for index in sorted(features)],
            }
        ),
        parquet_dir / "shard_00000.parquet",
    )


@contextmanager
def _patched_main_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_mode: SegmentMode = SegmentMode.turn_delimited_fixed_tokens,
    val_mode: SegmentMode | None = None,
    eval_every: int = 0,
    bos_mismatch: bool = False,
    real_store: bool = False,
    task: Task | None = None,
    n_segments_per_conv: int = 1,
    feature_pred_config: FeaturePredictionConfig | None = None,
    n_embd: int = N_EMBD,
) -> Iterator[_MainHarness]:
    train_feature_dir = Path(_feature_dir(tmp_path / "train", train_mode))
    train_parquet_dir = tmp_path / "train_parquet"
    train_feature_dir.mkdir(parents=True)
    train_parquet_dir.mkdir()
    active_val_mode = train_mode if val_mode is None else val_mode
    val_feature_dir = (
        Path(
            _feature_dir(
                tmp_path / "val",
                active_val_mode,
                phase=DatasetPhase.SAFETY_SFT_VAL,
            )
        )
        if eval_every > 0
        else None
    )
    val_parquet_dir = tmp_path / "val_parquet" if eval_every > 0 else None
    if val_feature_dir is not None and val_parquet_dir is not None:
        val_feature_dir.mkdir(parents=True)
        val_parquet_dir.mkdir()
        (val_feature_dir / "shard_00000_features.safetensors.zst").write_bytes(
            b"startup-presence-only"
        )
        pq.write_table(
            pa.table({"original_index": [0], "text": ["presence-only"]}),
            val_parquet_dir / "shard_00000.parquet",
        )
    for feature_dir in (train_feature_dir, val_feature_dir):
        if feature_dir is not None:
            write_steering_stats(
                feature_dir / "feature_statistics.safetensors",
                n_sae_features=N_SAE_FEATURES,
            )
    tokenizer = SFTMockTokenizer()
    task = MultiTurnTask(n=4) if task is None else task
    val_task = task if eval_every > 0 else None
    features = build_synthetic_features(task, tokenizer, n_segments_per_conv=n_segments_per_conv)
    base_model = build_tiny_model(
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        feature_pred_config=feature_pred_config,
        n_embd=n_embd,
        n_layer=2,
        seq_len=64,
        xattn_frequency=1,
    )
    selection = (
        {"threshold_type": "segment_topk", "threshold": 2}
        if real_store
        else {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 1,
            "max_k": 4,
            "mode": "top_k_percentile",
            "delta_weights": None,
        }
    )
    base_meta = {
        "model_config": asdict(base_model.dec_config),
        "user_config": {
            "encoder": asdict(base_model.enc_config),
            "cross_attention": asdict(base_model.xattn_config),
            "num_buckets": NUM_BUCKETS,
            "feature_selection": selection,
            "negative_selection": None,
            "feature_delta": None,
            "sae": {
                "model": "gemma-2-2b",
                "release": "gemmascope",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
            },
        },
    }
    if feature_pred_config is not None:
        base_meta["user_config"]["feature_prediction"] = sft._serialize_fp_config(feature_pred_config)
    argv = [
        "conditional_encdec_sft.py",
        "config_name=conditional_encdec_sft",
        "model_tag=unit",
        f"base_checkpoint_dir={tmp_path / 'base_checkpoint'}",
        f"feature_dir={train_feature_dir}",
        f"sft_parquet_dir={train_parquet_dir}",
        "device_type=cpu",
        "run=dummy",
        "max_seq_len=64",
        "device_batch_size=2",
        "total_batch_size=128",
        "num_iterations=1",
        "lr_warmup_steps=0",
        f"eval_every={eval_every}",
        "save_every=1",
        "min_loss_tokens=1",
        "alignment_probe_samples=0",
        "feature_source_bos_char_len=0",
        "compile_static=true",
        "dry_run=false",
    ]
    if not real_store:
        argv.extend(
            [
                "feature_selection.mean=2.0",
                "feature_selection.std=2.0",
                "feature_selection.min_k=1",
                "feature_selection.max_k=4",
            ]
        )
    if val_feature_dir is not None and val_parquet_dir is not None:
        argv.extend(
            [
                f"val_feature_dir={val_feature_dir}",
                f"val_sft_parquet_dir={val_parquet_dir}",
            ]
        )
    saved_meta: list[dict] = []
    saved_dirs: list[str] = []
    finalize_calls: list[tuple[tuple, dict]] = []
    bos_checks: list[tuple[str, str, int, str]] = []
    load_observations: list[tuple[str, str, bool | None]] = []
    real_finalize = sft._finalize_sft_segment_mode
    real_compose = sft.compose

    def compose_without_percentile_table(*args, **kwargs):
        cfg = real_compose(*args, **kwargs)
        cfg.feature_selection = OmegaConf.create(selection)
        return cfg

    def finalize_spy(*args, **kwargs):
        finalize_calls.append((args, kwargs))
        return real_finalize(*args, **kwargs)

    def capture_save(
        checkpoint_dir,
        _step,
        _model_data,
        _optimizer_data,
        meta,
        *,
        rank,
    ) -> None:
        del rank
        saved_dirs.append(str(checkpoint_dir))
        saved_meta.append(meta)

    def setup_optimizer(model, _cfg):
        optimizer = t.optim.AdamW(model.parameters(), lr=1e-3)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            group["kind"] = "adamw"
        return optimizer

    def unexpected_generator(**_kwargs):
        raise AssertionError("generator constructed after BOS mismatch")

    def record_bos_check(
        checked_feature_dir: str,
        checked_parquet_dir: str,
        *,
        configured: int,
        source: str,
    ) -> int:
        bos_checks.append(
            (checked_feature_dir, checked_parquet_dir, configured, source)
        )
        if bos_mismatch:
            raise DataLoadError(
                "Conditioned SFT train feature_source_bos_char_len mismatch: "
                "detected=5, configured=0"
            )
        return configured

    monkeypatch.setattr(sys, "argv", argv)
    if real_store:
        monkeypatch.setattr(sft, "compose", compose_without_percentile_table)
    monkeypatch.setattr(sft, "compute_init", lambda _device: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(sft, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        sft,
        "load_checkpoint",
        lambda *_args, **_kwargs: (dict(base_model.state_dict()), None, base_meta),
    )
    monkeypatch.setattr(sft, "create_wandb_run", lambda *_args: DummyWandb())
    monkeypatch.setattr(sft, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(sft, "get_token_bytes", lambda **_: t.ones(tokenizer.vocab_size, dtype=t.int64))
    monkeypatch.setattr(
        sft,
        "build_task_mixtures",
        lambda _dataset: (task, val_task),
    )
    monkeypatch.setattr(sft, "_setup_sft_optimizer", setup_optimizer)
    monkeypatch.setattr(sft.t, "compile", lambda model, **_kwargs: model)
    monkeypatch.setattr(sft.flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(sft, "save_checkpoint", capture_save)
    monkeypatch.setattr(sft, "_finalize_sft_segment_mode", finalize_spy)
    monkeypatch.setattr(
        sft,
        "assert_sft_source_bos_char_len_matches",
        record_bos_check,
    )
    if bos_mismatch:
        monkeypatch.setattr(
            sft,
            "make_sft_data_generator",
            unexpected_generator,
        )

    train_store = build_synthetic_sft_store(
        features,
        mixture="safety",
        segment_mode=train_mode,
    )
    val_store = build_synthetic_sft_store(
        features,
        split=SFTSplit.VAL,
        mixture="safety",
        segment_mode=active_val_mode,
    )
    real_load = sft.load_sft_features

    def resolve_store(feature_dir, parquet_dir, **_kwargs):
        load_observations.append((str(feature_dir), str(parquet_dir), None))
        if str(parquet_dir) == str(train_parquet_dir):
            return train_store
        if val_parquet_dir is not None and str(parquet_dir) == str(val_parquet_dir):
            return val_store
        raise AssertionError(f"unexpected SFT parquet directory: {parquet_dir}")

    def counted_real_load(feature_dir, parquet_dir, **kwargs):
        report = kwargs.pop("report", None) or SFTFeatureLoadReport()
        store = real_load(feature_dir, parquet_dir, report=report, **kwargs)
        load_observations.append(
            (str(feature_dir), str(parquet_dir), report.cache_hit)
        )
        return store

    if real_store:
        assert eval_every == 0
        _write_real_feature_family(
            train_feature_dir,
            train_parquet_dir,
            features,
            mode=train_mode,
        )

    with (
        _loguru_sink() as sink,
        patch.object(
            sft,
            "load_sft_features",
            side_effect=counted_real_load if real_store else resolve_store,
        ),
        patch.object(
            sft,
            "_assert_sft_mixture_fingerprint_matches_task",
            return_value=None,
        ),
    ):
        yield _MainHarness(
            train_feature_dir,
            val_feature_dir,
            train_parquet_dir,
            val_parquet_dir,
            saved_meta,
            saved_dirs,
            finalize_calls,
            bos_checks,
            load_observations,
            sink,
        )


@pytest.mark.parametrize("bos_mismatch", [False, True])
def test_main_calls_behavioral_helper_logs_and_saves_discovered_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bos_mismatch: bool,
) -> None:
    mode = SegmentMode.turn_delimited_fixed_tokens
    with _patched_main_runtime(
        tmp_path,
        monkeypatch,
        train_mode=mode,
        bos_mismatch=bos_mismatch,
    ) as harness:
        if bos_mismatch:
            with pytest.raises(DataLoadError, match=r"detected=5, configured=0"):
                sft.main()
        else:
            sft.main()

    assert len(harness.finalize_calls) == 1
    args, kwargs = harness.finalize_calls[0]
    assert args == (mode, None)
    assert kwargs["resuming"] is False
    assert kwargs["checkpoint_meta"] is None
    assert harness.bos_checks == [
        (
            str(harness.train_feature_dir),
            str(harness.train_parquet_dir),
            0,
            SFTSplit.TRAIN.value,
        )
    ]
    if bos_mismatch:
        assert harness.saved_meta == []
        return
    assert kwargs["checkpoint_dir"] == harness.saved_dirs[0]
    assert (
        "INFO | SFT feature segmentation mode: turn_delimited_fixed_tokens"
        in harness.sink.getvalue().splitlines()
    )
    assert len(harness.saved_meta) == 1
    assert harness.saved_meta[0][SFT_SEGMENT_MODE_META_KEY] == mode.value


def test_main_rejects_train_val_segment_mode_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _patched_main_runtime(
        tmp_path,
        monkeypatch,
        eval_every=1,
        train_mode=SegmentMode.turn_delimited_fixed_tokens,
        val_mode=SegmentMode.fixed_tokens,
    ) as harness:
        with pytest.raises(
            DataLoadError,
            match=(
                rf"same segmentation mode as training.*{harness.train_feature_dir}.*"
                rf"turn_delimited_fixed_tokens.*{harness.val_feature_dir}.*fixed_tokens"
            ),
        ):
            sft.main()

    assert [call[1] for call in harness.load_observations[:2]] == [
        str(harness.train_parquet_dir),
        str(harness.val_parquet_dir),
    ]


def test_main_accepts_matching_train_and_val_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode = SegmentMode.turn_delimited_fixed_tokens
    with _patched_main_runtime(
        tmp_path,
        monkeypatch,
        eval_every=1,
        train_mode=mode,
        val_mode=mode,
    ) as harness:
        sft.main()

    assert len(harness.finalize_calls) == 1
    assert harness.finalize_calls[0][0] == (mode, mode)
    requested_parquet_dirs = [call[1] for call in harness.load_observations]
    assert str(harness.train_parquet_dir) in requested_parquet_dirs
    assert str(harness.val_parquet_dir) in requested_parquet_dirs
    assert len(harness.saved_meta) == 1
    assert harness.saved_meta[0][SFT_SEGMENT_MODE_META_KEY] == mode.value


def test_startup_feature_store_is_reused_by_the_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _patched_main_runtime(
        tmp_path,
        monkeypatch,
        train_mode=SegmentMode.fixed_tokens,
        real_store=True,
    ) as harness:
        sft.main()

    train_reports = [
        cache_hit
        for feature_dir, parquet_dir, cache_hit in harness.load_observations
        if feature_dir == str(harness.train_feature_dir)
        and parquet_dir == str(harness.train_parquet_dir)
    ]
    assert train_reports == [False, True]
