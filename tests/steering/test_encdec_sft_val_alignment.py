"""Validation-split identity and startup artifact contracts for conditioned SFT."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from errors import DataLoadError
from scripts.train.conditional_encdec_sft import (
    SFTState,
    _build_sft_val_loader,
    _resolve_and_require_sft_val_artifacts,
    _resolve_sft_val_parquet_dir,
    _resolve_val_feature_dir,
    _sft_artifact_recovery_instructions,
    _task_steered_jsonl,
    make_sft_data_generator,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.feature_paths import (
    STEERED_JSONL_PLACEHOLDER,
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    parse_feature_dir,
)
from steering.sae_resolver import ActivationType, SAEWidth
from steering.sft.loss_mask import LossMaskMode
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft_schema import SFTSplit
from tasks.common import TaskMixture
from tasks.mixtures import (
    STEERED_MIXTURES,
    SFTMixture,
    build_task_mixtures,
    sft_val_family,
)
from tasks.steered import SteeredConversations, SteeredTaskMixture
from tests.encdec_test_helpers import N_SAE_FEATURES, NUM_BUCKETS
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    build_synthetic_features,
    patch_load_sft_features,
)


def _write_steered_jsonl(tmp_path: Path, rows: int = 3) -> str:
    path = tmp_path / "steered.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt {index}"},
                    {"role": "assistant", "content": f"steered answer {index}"},
                ]
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return str(path)


def _boundaries() -> t.Tensor:
    return (
        t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )


def _val_loader(
    *,
    task,
    tokenizer,
    feature_dir: str = "/val/features",
    parquet_dir: str = "/val/parquet",
    mixture_name: str = "capabilities",
):
    return _build_sft_val_loader(
        val_task=task,
        tokenizer=tokenizer,
        B=2,
        T=64,
        device="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        feature_dir=feature_dir,
        expected_feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1000),
        boundaries=_boundaries(),
        precomputed_thresholds=None,
        sampling_bundle=None,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        alignment_probe_samples=0,
        mixture_name=mixture_name,
    )


def test_val_loader_rejects_feature_dir_other_than_resolved_val_dir() -> None:
    tokenizer = SFTMockTokenizer()
    val_task = SingleTurnTask(n=1)
    with pytest.raises(
        DataLoadError,
        match=r"feature_dir='/train/features', expected_feature_dir='/val/features'",
    ):
        _build_sft_val_loader(
            val_task=val_task,
            tokenizer=tokenizer,
            B=1,
            T=64,
            device="cpu",
            ddp_rank=0,
            ddp_world_size=1,
            feature_dir="/train/features",
            expected_feature_dir="/val/features",
            parquet_dir="/val/parquet",
            threshold_cfg=ThresholdConfig(ThresholdType.segment_topk, 1000),
            boundaries=_boundaries(),
            precomputed_thresholds=None,
            sampling_bundle=None,
            loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
            min_loss_tokens=1,
        )


def test_val_loader_rejects_train_conversion_before_index_coverage() -> None:
    """A train store must fail by identity even when every val index is present."""
    tokenizer = SFTMockTokenizer()
    train_task = SingleTurnTask(n=4, user_chars=11, asst_chars=17, prefix="train")
    val_task = SingleTurnTask(n=4, user_chars=5, asst_chars=9, prefix="val")
    assert train_task[0] != val_task[0]
    assert len(tokenizer.render_conversation(train_task[0])[0]) != len(
        tokenizer.render_conversation(val_task[0])[0]
    )
    train_features = build_synthetic_features(train_task, tokenizer)

    with (
        patch_load_sft_features(
            train_features,
            split=SFTSplit.TRAIN,
            mixture="capabilities",
        ),
        pytest.raises(
            DataLoadError,
            match=r"expected_split='val', conversion_split='train'",
        ),
    ):
        next(_val_loader(task=val_task, tokenizer=tokenizer))


def test_val_loader_metadata_less_nonconventional_override_preserves_dataloaderror(
    tmp_path: Path,
) -> None:
    """A metadata-less override must retain the split-mismatch diagnostics."""
    tokenizer = SFTMockTokenizer()
    val_task = SingleTurnTask(n=1)
    parquet_dir = tmp_path / "operator_validation_override"
    parquet_dir.mkdir()
    (parquet_dir / "shard_00000.parquet").touch()
    features = build_synthetic_features(val_task, tokenizer)
    override = _resolve_sft_val_parquet_dir(str(parquet_dir), "/unused", "safety")

    with (
        patch_load_sft_features(features, split=None, mixture=None),
        pytest.raises(
            DataLoadError,
            match=rf"expected_split='val'.*conversion_split=None.*"
            rf"Feature dir: /val/features.*Parquet dir: {re.escape(str(parquet_dir))}",
        ) as exc_info,
    ):
        next(_val_loader(task=val_task, tokenizer=tokenizer, parquet_dir=override))

    message = str(exc_info.value)
    assert "Feature dir: /val/features" in message
    assert f"Parquet dir: {parquet_dir}" in message
    assert f"mixture={parquet_dir.name} split=val" in message
    assert "dataset_phase=" not in message


def test_val_loader_yields_identity_unique_to_val_features() -> None:
    tokenizer = SFTMockTokenizer()
    train_task = SingleTurnTask(n=4, user_chars=11, asst_chars=17, prefix="train")
    val_task = SingleTurnTask(n=4, user_chars=5, asst_chars=9, prefix="val")
    train_features = build_synthetic_features(train_task, tokenizer)
    raw_val_features = build_synthetic_features(val_task, tokenizer)
    val_features = {
        idx: [replace(segment, feature_ids=segment.feature_ids + 100) for segment in segments]
        for idx, segments in raw_val_features.items()
    }
    train_ids = {
        int(fid)
        for segments in train_features.values()
        for segment in segments
        for fid in segment.feature_ids
    }
    val_only_id = int(val_features[0][0].feature_ids[0])
    assert val_only_id not in train_ids

    with patch_load_sft_features(
        val_features,
        split=SFTSplit.VAL,
        mixture="capabilities",
    ):
        batch = next(_val_loader(task=val_task, tokenizer=tokenizer))

    enc_fids, enc_mask = batch[2], batch[4]
    assert val_only_id in enc_fids[enc_mask].tolist()


def test_empty_val_mixture_requires_no_val_artifacts() -> None:
    train_task = SingleTurnTask(n=1)
    with patch(
        "tasks.mixtures.build_mixture",
        return_value=([train_task], []),
    ):
        train, val = build_task_mixtures(SFTMixture.SAFETY)
    assert len(train) == 1
    assert val is None
    assert _resolve_and_require_sft_val_artifacts(
        has_val=False,
        eval_every=150,
        val_parquet_override=None,
        val_feature_override=None,
        base_dir="/missing/base",
        feature_dir="not/a/feature/path",
        dataset="safety",
        val_dataset="safety",
    ) == (None, None)


def test_missing_val_parquet_fails_at_startup_naming_both_paths(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "missing_parquet"
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (feature_dir / "shard_00000_features.safetensors.zst").touch()

    with pytest.raises(
        DataLoadError,
        match=re.escape(f"Resolved val parquet dir: {parquet_dir}"),
    ) as exc_info:
        _resolve_and_require_sft_val_artifacts(
            has_val=True,
            eval_every=1,
            val_parquet_override=str(parquet_dir),
            val_feature_override=str(feature_dir),
            base_dir=str(tmp_path),
            feature_dir="unused",
            dataset="safety",
            val_dataset="safety",
        )
    assert f"Resolved val feature dir: {feature_dir}" in str(exc_info.value)
    assert "mixture=safety split=val" in str(exc_info.value)
    assert "dataset_phase=safety_sft_val" in str(exc_info.value)


def test_missing_val_features_fails_at_startup_naming_both_paths(tmp_path: Path) -> None:
    parquet_dir = tmp_path / "parquet"
    feature_dir = tmp_path / "missing_features"
    parquet_dir.mkdir()
    (parquet_dir / "shard_00000.parquet").touch()

    with pytest.raises(
        DataLoadError,
        match=re.escape(f"Resolved val feature dir: {feature_dir}"),
    ) as exc_info:
        _resolve_and_require_sft_val_artifacts(
            has_val=True,
            eval_every=1,
            val_parquet_override=str(parquet_dir),
            val_feature_override=str(feature_dir),
            base_dir=str(tmp_path),
            feature_dir="unused",
            dataset="capabilities",
            val_dataset="capabilities",
        )
    assert f"Resolved val parquet dir: {parquet_dir}" in str(exc_info.value)


def test_eval_every_zero_is_explicit_val_artifact_opt_out() -> None:
    assert _resolve_and_require_sft_val_artifacts(
        has_val=True,
        eval_every=0,
        val_parquet_override=None,
        val_feature_override=None,
        base_dir="/missing/base",
        feature_dir="not/a/feature/path",
        dataset="safety",
        val_dataset="safety",
    ) == (None, None)


def test_val_parquet_resolver_default_and_override() -> None:
    assert _resolve_sft_val_parquet_dir(None, "/base", "safety") == "/base/sft_data/safety_val"
    assert _resolve_sft_val_parquet_dir("/override/val", "/base", "safety") == "/override/val"


@pytest.mark.parametrize(
    "dataset", [SFTMixture.STANDARD_STEERED, SFTMixture.STANDARD_STEERED_BALANCED]
)
def test_steered_val_artifacts_resolve_onto_the_standard_family(
    tmp_path: Path, dataset: SFTMixture
) -> None:
    """A steered run validates on standard's parquet, features and recovery."""
    val_dataset = sft_val_family(dataset).value
    assert val_dataset == "standard"

    train_feature_dir = build_feature_dir(
        root=tmp_path / "root",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="google/gemma-2-2b",
        dataset_phase=DatasetPhase.STANDARD_STEERED_SFT
        if dataset is SFTMixture.STANDARD_STEERED
        else DatasetPhase.STANDARD_STEERED_BALANCED_SFT,
    )
    expected_parquet = Path(str(tmp_path)) / "sft_data" / "standard_val"
    expected_features = Path(
        _resolve_val_feature_dir(None, str(train_feature_dir), dataset.value, val_dataset)
    )
    assert parse_feature_dir(str(expected_features)).dataset_phase is (
        DatasetPhase.STANDARD_SFT_VAL
    )

    with pytest.raises(
        DataLoadError,
        match=re.escape(f"Resolved val parquet dir: {expected_parquet}"),
    ) as exc_info:
        _resolve_and_require_sft_val_artifacts(
            has_val=True,
            eval_every=1,
            val_parquet_override=None,
            val_feature_override=None,
            base_dir=str(tmp_path),
            feature_dir=str(train_feature_dir),
            dataset=dataset.value,
            val_dataset=val_dataset,
        )
    message = str(exc_info.value)
    assert f"Resolved val parquet dir: {expected_parquet}" in message
    assert f"Resolved val feature dir: {expected_features}" in message
    # The recovery commands name the VAL FAMILY, never the steered mixture.
    assert "mixture=standard split=val" in message
    assert "dataset_phase=standard_sft_val" in message
    assert dataset.value not in message.split("Recovery commands:")[1]


@pytest.mark.parametrize(
    "dataset", [SFTMixture.STANDARD_STEERED, SFTMixture.STANDARD_STEERED_BALANCED]
)
def test_val_loader_accepts_the_family_fingerprint_and_rejects_the_steered_one(
    dataset: SFTMixture,
) -> None:
    """A steered run's val artifacts are fingerprinted ``standard``.

    The trainer passes ``mixture_name=val_dataset`` at its val loader call
    site; this is the behaviour that argument buys. Reverting it to
    ``str(cfg.dataset)`` names the steered mixture, whose fingerprint the
    standard val conversion never carries — so the gate must reject it.
    """
    tokenizer = SFTMockTokenizer()
    val_task = SingleTurnTask(n=4, user_chars=5, asst_chars=9, prefix="val")
    val_family = sft_val_family(dataset).value
    assert val_family == "standard" != dataset.value
    family_fingerprint = compute_mixture_fingerprint(
        val_task, mixture_name=val_family, split=SFTSplit.VAL
    )
    features = build_synthetic_features(val_task, tokenizer)

    with patch_load_sft_features(
        features,
        split=SFTSplit.VAL,
        mixture=val_family,
        parquet_fingerprint=family_fingerprint,
        feature_fingerprint=family_fingerprint,
        enforce_fingerprints=True,
    ):
        assert (
            next(_val_loader(task=val_task, tokenizer=tokenizer, mixture_name=val_family))
            is not None
        )

    with (
        patch_load_sft_features(
            features,
            split=SFTSplit.VAL,
            mixture=val_family,
            parquet_fingerprint=family_fingerprint,
            feature_fingerprint=family_fingerprint,
            enforce_fingerprints=True,
        ),
        pytest.raises(DataLoadError, match="fingerprint mismatch"),
    ):
        next(_val_loader(task=val_task, tokenizer=tokenizer, mixture_name=dataset.value))


@pytest.mark.parametrize("mixture", sorted(STEERED_MIXTURES))
def test_steered_recovery_command_carries_the_required_override(
    tmp_path: Path, mixture: SFTMixture
) -> None:
    """A steered TRAIN recovery command must be paste-able.

    Without ``steered_jsonl=`` the printed conversion dies immediately with
    "requires steered_jsonl=<path>", which is the second failure an operator
    should never have to discover.
    """
    steered_path = _write_steered_jsonl(tmp_path)
    message = _sft_artifact_recovery_instructions(
        mixture.value, split=SFTSplit.TRAIN, steered_jsonl=steered_path
    )
    assert f"mixture={mixture.value} split=train steered_jsonl={steered_path}" in message
    assert f"dataset_phase={mixture.value}_sft" in message

    # With no path in hand the emitter still names the required override.
    placeholder = _sft_artifact_recovery_instructions(mixture.value, split=SFTSplit.TRAIN)
    assert f"steered_jsonl={STEERED_JSONL_PLACEHOLDER}" in placeholder


@pytest.mark.parametrize("mixture", ["standard", "safety", "capabilities"])
def test_non_steered_recovery_command_never_mentions_steered_jsonl(mixture: str) -> None:
    for split in SFTSplit:
        message = _sft_artifact_recovery_instructions(mixture, split=split)
        assert f"mixture={mixture} split={split.value}\n" in message
        assert "steered_jsonl" not in message


def test_task_steered_jsonl_recovers_the_file_a_steered_mixture_was_built_from(
    tmp_path: Path,
) -> None:
    """The emitters name the operator's OWN path, recovered from the mixture."""
    steered_path = _write_steered_jsonl(tmp_path)
    steered = SteeredConversations(steered_path)
    base = SingleTurnTask(n=6)
    assert _task_steered_jsonl(SteeredTaskMixture([base], steered)) == steered_path
    assert _task_steered_jsonl(TaskMixture([base, steered])) is None
    assert _task_steered_jsonl(base) is None


def test_trainer_source_derives_every_val_identity_from_the_family() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "train"
        / "conditional_encdec_sft.py"
    ).read_text()
    normalized = re.sub(r"\s+", "", source)
    assert "val_dataset=sft_val_family(dataset_mixture).value" in normalized
    assert normalized.count("val_dataset=val_dataset") == 1
    # The val loader is fingerprinted against the family; the train generator
    # keeps this run's own dataset name.
    assert "mixture_name=val_dataset" in normalized
    assert "mixture_name=str(cfg.dataset),expected_split=SFTSplit.TRAIN" in normalized


def test_generator_rejects_raw_expected_split_string() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="expected_split"):
        make_sft_data_generator(
            task=SingleTurnTask(n=1),
            tokenizer=SFTMockTokenizer(),
            B=1,
            T=8,
            device="cpu",
            sft_state=SFTState(),
            expected_split="val",  # ty: ignore[invalid-argument-type]
        )


def test_val_feature_resolver_round_trips_through_builder(tmp_path: Path) -> None:
    train_dir = build_feature_dir(
        root=tmp_path / "root",
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="google/gemma-2-2b",
        dataset_phase=DatasetPhase.SAFETY_SFT,
        context_independent_features=True,
        ignore_bos_features=True,
    )
    val_dir = _resolve_val_feature_dir(None, str(train_dir), "safety", "safety")
    parsed = parse_feature_dir(val_dir)
    assert parsed.dataset_phase is DatasetPhase.SAFETY_SFT_VAL
    assert parsed.context_independent_features is True
    assert parsed.ignore_bos_features is True
    assert parsed.model == "google/gemma-2-2b"
    assert val_dir == str(train_dir).replace("safety_sft", "safety_sft_val")


def test_val_feature_resolver_rejects_unparseable_train_path() -> None:
    with pytest.raises(DataLoadError, match="Cannot derive val_feature_dir"):
        _resolve_val_feature_dir(None, "/not/a/canonical/path", "safety", "safety")


def test_val_feature_resolver_rejects_wrong_train_phase(tmp_path: Path) -> None:
    feature_dir = build_feature_dir(
        root=tmp_path,
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.CAPABILITIES_SFT,
    )
    with pytest.raises(DataLoadError, match="expected 'safety_sft'"):
        _resolve_val_feature_dir(None, str(feature_dir), "safety", "safety")
