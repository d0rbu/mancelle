"""Correctness and lifecycle tests for issue-#842 packed CUDA staging."""

from __future__ import annotations

import copy
import gc
import threading
import time
import weakref
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

import steering.conditional.encoder_decoder.precomputed_cuda_staging as staging_module
from steering.conditional.encoder_decoder.dataloader import (
    ClosableBatchPrefetcher,
    precomputed_data_loader,
)
from steering.conditional.encoder_decoder.precomputed_cuda_staging import (
    PrecomputedBatch,
    PrecomputedCudaStagingError,
    PrecomputedCudaStagingLoader,
    _allocate_slot,
    _assert_same_schema,
    _ByteBoundedQueue,
    _Data,
    _stage_batch,
    _StagingStopped,
    retained_tensor_bytes,
    validate_cpu_precomputed_batch,
    validate_precomputed_cuda_staging_eligibility,
)
from steering.refined_types import PositiveInt
from steering.remote_training import (
    RemoteMaterializationCancelled,
    RemoteTrainingConfig,
    RemoteTrainingError,
    RemoteTrainPackedConfig,
    build_remote_run_fingerprint,
    validate_remote_training_eligibility,
)
from tests.steering.test_precomputed_dataloader_ddp import (
    _make_pipeline,
    _write_shards_with_selection,
)

requires_cuda = pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA not available")
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "training"
_EXTRACTION_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "extraction"


@pytest.fixture
def fake_cuda_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"record": 0, "synchronize": 0, "wait": 0, "stream_sync": 0}

    class FakeEvent:
        def __init__(self, blocking: bool = False) -> None:
            assert blocking is False

        def record(self, stream: object) -> None:
            assert isinstance(stream, FakeStream)
            calls["record"] += 1

        def synchronize(self) -> None:
            calls["synchronize"] += 1

    class FakeStream:
        def wait_event(self, event: object) -> None:
            assert isinstance(event, FakeEvent)
            calls["wait"] += 1

        def synchronize(self) -> None:
            calls["stream_sync"] += 1

    monkeypatch.setattr(t.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(t.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(t.cuda, "Stream", lambda **_kwargs: FakeStream())
    monkeypatch.setattr(t.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(t.cuda, "Event", FakeEvent)
    monkeypatch.setattr(t.cuda, "current_stream", lambda _device: FakeStream())
    monkeypatch.setattr(
        staging_module,
        "_allocate_slot",
        lambda schema: tuple(None for _ in schema.tensors),
    )
    monkeypatch.setattr(
        staging_module,
        "_stage_batch",
        lambda _slot, batch, *, device: batch,
    )
    return calls


def _cpu_batch(
    *,
    batch_size: int = 2,
    context_len: int = 8,
    n_seg: int = 3,
    pad_f: int = 4,
    fp_width: int | None = 5,
    sentinel: int = 0,
    state_overrides: dict[str, Any] | None = None,
) -> PrecomputedBatch:
    base = sentinel * 10_000
    inputs = (
        t.arange(batch_size * context_len, dtype=t.int64).reshape(batch_size, context_len).add(base)
    )
    targets = inputs.add(1)
    enc_fids = (
        t.arange(batch_size * n_seg * pad_f, dtype=t.int64)
        .reshape(batch_size, n_seg, pad_f)
        .remainder(2_074)
    )
    enc_mags = t.full_like(enc_fids, sentinel % 2)
    enc_mask = t.ones(batch_size, n_seg, pad_f, dtype=t.bool)
    enc_pcts = (
        t.arange(batch_size * n_seg * pad_f, dtype=t.float32)
        .reshape(batch_size, n_seg, pad_f)
        .div(100)
    )
    seg_map = (
        t.arange(context_len, dtype=t.int64).remainder(n_seg).expand(batch_size, -1).contiguous()
    )
    seg_doc_id = t.full((batch_size, n_seg), sentinel, dtype=t.int64)
    fp_targets = (
        None
        if fp_width is None
        else t.arange(
            batch_size * n_seg * fp_width,
            dtype=t.float32,
        ).reshape(batch_size, n_seg, fp_width)
    )
    state: dict[str, Any] = {
        "batch_idx": sentinel,
        "shard": f"shard_{sentinel:05d}.safetensors.zst",
        "shard_batch_offset": sentinel,
        "epoch": 1,
        "max_seg_size": context_len,
        "pad_F": pad_f,
        "ddp_world_size": 1,
        "batch_rows": [[1, f"shard_{sentinel:05d}", sentinel]],
        "feature_pipeline_rng_state": {"selection": t.tensor([sentinel], dtype=t.int64)},
    }
    if state_overrides:
        state.update(state_overrides)
    return (
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        enc_pcts,
        seg_map,
        seg_doc_id,
        fp_targets,
        state,
    )


def _valid_eligibility(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "device": t.device("cuda:0"),
        "label_dropout": 0.0,
        "selection_dir": object(),
        "remote_materializer": object(),
        "packed_materializer": object(),
        "resume_state_dict": None,
        "expected_batch_size": 16,
        "expected_context_len": 1024,
        "expected_pad_f": 96,
        "expected_n_seg": 13,
    }
    values.update(overrides)
    return values


class TestRemoteCudaStagingConfig:
    def test_cache_depth_belongs_to_shared_body_prefetch_config(self) -> None:
        parsed = RemoteTrainPackedConfig.parse({})

        assert parsed.trust_existing_cache is False
        assert parsed.cuda_staging is None

    def test_full_config_parses_strict_values(self) -> None:
        parsed = RemoteTrainPackedConfig.parse(
            {
                "trust_existing_cache": True,
                "cuda_staging": {
                    "host_queue_bytes": 512 * 1024 * 1024,
                    "pinned_slots": 3,
                },
            }
        )

        assert parsed.cuda_staging is not None
        assert parsed.cuda_staging.host_queue_bytes == 512 * 1024 * 1024
        assert parsed.cuda_staging.pinned_slots == 3

    def test_staging_is_operational_and_does_not_change_run_fingerprint(self) -> None:
        contracts = {
            "train": {"identity": "train"},
            "derived_val": {"identity": "derived"},
            "base_val": {"identity": "base"},
        }
        legacy = build_remote_run_fingerprint(
            resolved_training_config={
                "depth": 12,
                "remote_training": {
                    "body_prefetch": {
                        "lookahead_shards_per_rank": 2,
                        "cache_shards": 24,
                    },
                },
            },
            world_size=2,
            selection_contracts=contracts,
        )
        staged = build_remote_run_fingerprint(
            resolved_training_config={
                "depth": 12,
                "remote_training": {
                    "body_prefetch": {
                        "lookahead_shards_per_rank": 2,
                        "cache_shards": 24,
                    },
                    "train_packed": {
                        "cuda_staging": {
                            "host_queue_bytes": 512 * 1024 * 1024,
                            "pinned_slots": 3,
                        },
                    },
                },
            },
            world_size=2,
            selection_contracts=contracts,
        )

        assert staged == legacy

    def test_top_level_remote_eligibility_rejects_dropout_before_startup(self) -> None:
        remote = RemoteTrainingConfig.parse(
            {
                "bucket_id": "owner/bucket",
                "body_prefetch": {"lookahead_shards_per_rank": 2},
                "selection_prefix": "features",
                "checkpoint_prefix": "checkpoints",
                "selection_wait_timeout_seconds": 1,
                "checkpoint_retry_timeout_seconds": 1,
                "retry_interval_seconds": 0.1,
                "resume_latest": False,
                "train_packed": {
                    "cuda_staging": {
                        "host_queue_bytes": 1,
                        "pinned_slots": 1,
                    },
                },
            }
        )
        assert remote is not None
        cfg = {
            "use_precomputed_batches": True,
            "feature_clusters": {"path": "clusters.yaml"},
            "feature_delta": {"mode": "splice_counterfactual"},
            "max_seq_len": 1024,
            "model_tag": "test",
            "resume_from_step": -1,
            "label_dropout": 0.1,
        }

        with pytest.raises(RemoteTrainingError, match="label_dropout=0.0"):
            validate_remote_training_eligibility(cfg, remote)

    def test_remote_packed_hydra_preset_accepts_explicit_opt_in(self) -> None:
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR),
            version_base=None,
        ):
            cfg = compose(
                config_name=(
                    "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed"
                ),
                overrides=[
                    "+remote_training.train_packed.cuda_staging="
                    "{host_queue_bytes:536870912,pinned_slots:3}"
                ],
            )
        raw = OmegaConf.to_container(cfg.remote_training, resolve=True)
        parsed = RemoteTrainingConfig.parse(raw)

        assert parsed is not None
        assert parsed.train_packed is not None
        assert parsed.train_packed.cuda_staging is not None
        assert parsed.train_packed.cuda_staging.host_queue_bytes == 536_870_912
        assert parsed.train_packed.cuda_staging.pinned_slots == 3

    def test_shipped_remote_packed_preset_keeps_staging_disabled(self) -> None:
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR),
            version_base=None,
        ):
            cfg = compose(
                config_name=(
                    "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed"
                ),
            )
        raw = OmegaConf.to_container(cfg.remote_training, resolve=True)
        parsed = RemoteTrainingConfig.parse(raw)

        assert parsed is not None
        assert parsed.train_packed is not None
        assert parsed.train_packed.cuda_staging is None

    def test_two_h200_preset_enables_measured_staging(self) -> None:
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR),
            version_base=None,
        ):
            cfg = compose(
                config_name=(
                    "cluster_fp/"
                    "conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_2xh200"
                ),
            )
        raw = OmegaConf.to_container(cfg.remote_training, resolve=True)
        parsed = RemoteTrainingConfig.parse(raw)

        assert cfg.feature_dir == (
            "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
            "m_gemma-3-4b-pt/base/context_independent/no_bos/"
            "splice/cut_frac0.15/seed42"
        )
        assert cfg.data_dir is None
        assert cfg.device_batch_size == 32
        assert cfg.total_batch_size == 524_288
        assert cfg.fp8 is False
        assert parsed is not None
        assert parsed.train_packed is not None
        assert parsed.body_prefetch.cache_shards == 24
        assert parsed.train_packed.trust_existing_cache is True
        assert parsed.train_packed.cuda_staging is not None
        assert parsed.train_packed.cuda_staging.host_queue_bytes == 536_870_912
        assert parsed.train_packed.cuda_staging.pinned_slots == 3

    def test_eight_h100_preset_is_the_existing_run_with_only_staging_added(
        self,
    ) -> None:
        feature_dir = (
            "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
            "m_gemma-3-4b-pt/base/context_independent/no_bos/"
            "splice/cut_frac0.15/seed42"
        )
        with initialize_config_dir(
            config_dir=str(_CONFIG_DIR),
            version_base=None,
        ):
            old_command = compose(
                config_name=(
                    "cluster_fp/conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed"
                ),
                overrides=[
                    f"feature_dir={feature_dir}",
                    "device_batch_size=16",
                    "total_batch_size=524288",
                    "fp8=false",
                    "steering_eval_every=0",
                ],
            )
            target = compose(
                config_name=(
                    "cluster_fp/"
                    "conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed_8xh100"
                ),
            )

        old_resolved = OmegaConf.to_container(old_command, resolve=True)
        target_resolved = OmegaConf.to_container(target, resolve=True)
        assert isinstance(old_resolved, dict)
        assert isinstance(target_resolved, dict)
        old_semantics = copy.deepcopy(old_resolved)
        target_semantics = copy.deepcopy(target_resolved)
        target_staging = target_semantics["remote_training"]["train_packed"].pop("cuda_staging")

        assert target_staging == {
            "host_queue_bytes": 536_870_912,
            "pinned_slots": 3,
        }
        assert target_semantics == old_semantics
        assert target.run == old_command.run
        assert target.model_tag == old_command.model_tag
        assert target.feature_dir == old_command.feature_dir
        assert target.feature_mask == old_command.feature_mask
        assert target.feature_clusters == old_command.feature_clusters
        assert target.feature_selection == old_command.feature_selection
        assert target.negative_selection == old_command.negative_selection
        assert target.feature_delta == old_command.feature_delta

        contracts = {
            "train": {"identity": "train"},
            "derived_val": {"identity": "derived"},
            "base_val": {"identity": "base"},
        }
        assert build_remote_run_fingerprint(
            resolved_training_config=cast(Mapping[str, Any], target_resolved),
            world_size=8,
            selection_contracts=contracts,
        ) == build_remote_run_fingerprint(
            resolved_training_config=cast(Mapping[str, Any], old_resolved),
            world_size=8,
            selection_contracts=contracts,
        )

    def test_eight_h100_queue_crosses_the_full_rank_local_shard_stripe(self) -> None:
        batch_size = 16
        context_len = 1_024
        n_seg = 13
        pad_f = 96
        fp_width = 2_074
        field_bytes = {
            "inputs": batch_size * context_len * 8,
            "targets": batch_size * context_len * 8,
            "enc_fids": batch_size * n_seg * pad_f * 8,
            "enc_mags": batch_size * n_seg * pad_f * 8,
            "enc_mask": batch_size * n_seg * pad_f,
            "enc_pcts": batch_size * n_seg * pad_f * 4,
            "seg_map": batch_size * context_len * 8,
            "seg_doc_id": batch_size * n_seg * 8,
            "fp_targets": batch_size * n_seg * fp_width * 4,
        }
        retained_batch_bytes = sum(field_bytes.values())
        host_queue_bytes = 512 * 1024 * 1024
        rank_batches_per_shard = 16_384 // 8 // batch_size
        grad_accum_steps = 524_288 // (batch_size * context_len * 8)

        assert retained_batch_bytes == 2_539_776
        assert host_queue_bytes // retained_batch_bytes == 211
        assert rank_batches_per_shard == 128
        assert host_queue_bytes >= rank_batches_per_shard * retained_batch_bytes
        assert grad_accum_steps == 4
        assert rank_batches_per_shard // grad_accum_steps == 32
        assert 32 * 0.7 >= 4.0

    def test_c1024_h200_eval_context_extraction_uses_safe_gemma_window(self) -> None:
        with initialize_config_dir(
            config_dir=str(_EXTRACTION_CONFIG_DIR),
            version_base=None,
        ):
            cfg = compose(
                config_name=(
                    "extract_eval_context_features_"
                    "gemma3_4bpt_l22_f65k_ci_nobos_c1024_h200"
                ),
            )

        assert cfg.model == "gemma-3-4b-pt"
        assert cfg.context_window == 2048
        assert cfg.segment_size == 128
        assert cfg.context_independent_features is True
        assert cfg.gpu_batch_size == 32
        assert cfg.cpu_batch_size == 64

    @pytest.mark.parametrize(
        "raw",
        [
            {"cache_shards": 2, "cuda_staging": None},
            {"cache_shards": 2, "cuda_staging": {}},
            {
                "cache_shards": 2,
                "cuda_staging": {"host_queue_bytes": 1},
            },
            {
                "cache_shards": 2,
                "cuda_staging": {"pinned_slots": 1},
            },
            {
                "cache_shards": 2,
                "cuda_staging": {
                    "host_queue_bytes": 1,
                    "pinned_slots": 1,
                    "unknown": 1,
                },
            },
            {
                "cache_shards": 2,
                "cuda_staging": {
                    "host_queue_bytes": True,
                    "pinned_slots": 1,
                },
            },
            {
                "cache_shards": 2,
                "cuda_staging": {
                    "host_queue_bytes": 1,
                    "pinned_slots": False,
                },
            },
            {
                "cache_shards": 2,
                "cuda_staging": {
                    "host_queue_bytes": 0,
                    "pinned_slots": 1,
                },
            },
            {
                "cache_shards": 2,
                "cuda_staging": {
                    "host_queue_bytes": 1,
                    "pinned_slots": 0,
                },
            },
        ],
    )
    def test_invalid_or_partial_config_fails_loudly(self, raw: object) -> None:
        with pytest.raises(RemoteTrainingError, match="train_packed|cuda_staging"):
            RemoteTrainPackedConfig.parse(raw)


class TestStagingEligibility:
    def test_exact_remote_cuda_route_is_accepted_without_touching_cuda(self) -> None:
        assert validate_precomputed_cuda_staging_eligibility(
            **_valid_eligibility(device="cuda:7")
        ) == t.device("cuda:7")

    @pytest.mark.parametrize("device", ["cpu", t.device("cpu"), "cuda"])
    def test_device_must_be_explicitly_indexed_cuda(self, device: object) -> None:
        with pytest.raises(
            PrecomputedCudaStagingError,
            match="explicitly indexed CUDA",
        ):
            validate_precomputed_cuda_staging_eligibility(**_valid_eligibility(device=device))

    @pytest.mark.parametrize("label_dropout", [0.01, 1.0, None])
    def test_label_dropout_is_forbidden(self, label_dropout: object) -> None:
        with pytest.raises(PrecomputedCudaStagingError, match="label_dropout=0.0"):
            validate_precomputed_cuda_staging_eligibility(
                **_valid_eligibility(label_dropout=label_dropout)
            )

    @pytest.mark.parametrize(
        "missing",
        ["selection_dir", "remote_materializer", "packed_materializer"],
    )
    def test_every_verified_remote_input_is_required(self, missing: str) -> None:
        with pytest.raises(
            PrecomputedCudaStagingError,
            match="verified remote packed and selection",
        ):
            validate_precomputed_cuda_staging_eligibility(**_valid_eligibility(**{missing: None}))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("expected_batch_size", True),
            ("expected_batch_size", 0),
            ("expected_context_len", 0),
            ("expected_context_len", 1024.0),
            ("expected_pad_f", None),
            ("expected_pad_f", -1),
            ("expected_n_seg", None),
            ("expected_n_seg", 0),
        ],
    )
    def test_static_dimensions_are_positive_strict_ints(
        self,
        field: str,
        value: object,
    ) -> None:
        with pytest.raises(PrecomputedCudaStagingError, match="positive static"):
            validate_precomputed_cuda_staging_eligibility(**_valid_eligibility(**{field: value}))

    def test_resume_dropout_rng_metadata_is_refused(self) -> None:
        with pytest.raises(
            PrecomputedCudaStagingError,
            match="label_dropout_rng_state",
        ):
            validate_precomputed_cuda_staging_eligibility(
                **_valid_eligibility(
                    resume_state_dict={"label_dropout_rng_state": t.get_rng_state()}
                )
            )


class TestCpuBatchContract:
    @pytest.mark.parametrize(
        ("batch", "message"),
        [
            (list(_cpu_batch()), "exact 10-tuple"),
            (_cpu_batch()[:9], "exact 10-tuple"),
            ((*_cpu_batch()[:9], []), "string-keyed dict"),
            ((*_cpu_batch()[:9], {1: "not a string key"}), "string-keyed dict"),
        ],
    )
    def test_outer_tuple_and_state_container_are_exact(
        self,
        batch: object,
        message: str,
    ) -> None:
        with pytest.raises(PrecomputedCudaStagingError, match=message):
            validate_cpu_precomputed_batch(
                batch,
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    @pytest.mark.parametrize("fp_width", [None, 1, 2_074])
    def test_exact_batch_contract_and_schema(self, fp_width: int | None) -> None:
        batch = _cpu_batch(fp_width=fp_width)

        validated, schema = validate_cpu_precomputed_batch(
            batch,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        assert validated == batch
        assert len(schema.tensors) == 9
        if fp_width is None:
            assert schema.tensors[8] is None
        else:
            assert schema.tensors[8] is not None
        assert schema.state_keys == frozenset(batch[9])

    @pytest.mark.parametrize(
        ("index", "replacement", "message"),
        [
            (0, t.zeros(2, 7, dtype=t.int64), "inputs.*shape"),
            (0, t.zeros(2, 8, dtype=t.int32), "inputs.*dtype"),
            (1, None, "targets.*tensor"),
            (2, t.zeros(2, 3, 5, dtype=t.int64), "enc_fids.*shape"),
            (3, t.zeros(2, 3, 4, dtype=t.float32), "enc_mags.*dtype"),
            (4, t.zeros(2, 3, 4, dtype=t.uint8), "enc_mask.*dtype"),
            (5, t.zeros(2, 3, 4, dtype=t.float64), "enc_pcts.*dtype"),
            (6, t.zeros(2, 7, dtype=t.int64), "seg_map.*shape"),
            (7, t.zeros(2, 8, dtype=t.int64), "seg_doc_id.*shape"),
            (8, t.zeros(2, 3, 5, dtype=t.bool), "fp_targets.*dtype"),
        ],
    )
    def test_every_tensor_field_is_checked(
        self,
        index: int,
        replacement: object,
        message: str,
    ) -> None:
        values = list(_cpu_batch())
        values[index] = replacement

        with pytest.raises(PrecomputedCudaStagingError, match=message):
            validate_cpu_precomputed_batch(
                tuple(values),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_nonzero_storage_offset_is_refused(self) -> None:
        values = list(_cpu_batch())
        values[0] = t.zeros(2, 9, dtype=t.int64)[:, 1:]

        with pytest.raises(PrecomputedCudaStagingError, match="storage_offset=0"):
            validate_cpu_precomputed_batch(
                tuple(values),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_noncontiguous_tensor_is_refused(self) -> None:
        values = list(_cpu_batch())
        values[0] = t.zeros(8, 2, dtype=t.int64).transpose(0, 1)

        with pytest.raises(PrecomputedCudaStagingError, match="contiguous"):
            validate_cpu_precomputed_batch(
                tuple(values),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    @pytest.mark.parametrize(
        ("index", "replacement", "message"),
        [
            (
                0,
                t.empty(2, 8, dtype=t.int64, device="meta"),
                "inputs must be on CPU",
            ),
            (
                0,
                t.sparse_coo_tensor(
                    t.empty(2, 0, dtype=t.int64),
                    t.empty(0, dtype=t.int64),
                    size=(2, 8),
                ),
                "contiguous, strided",
            ),
            (
                5,
                t.zeros(2, 3, 4, dtype=t.float32, requires_grad=True),
                "gradient-free",
            ),
            (
                8,
                t.zeros(2, 3, dtype=t.float32),
                "fp_targets must be None or",
            ),
            (
                8,
                t.zeros(2, 3, 0, dtype=t.float32),
                "positive FP-target width",
            ),
        ],
    )
    def test_device_layout_gradient_and_fp_rank_are_checked(
        self,
        index: int,
        replacement: object,
        message: str,
    ) -> None:
        values = list(_cpu_batch())
        values[index] = replacement

        with pytest.raises(PrecomputedCudaStagingError, match=message):
            validate_cpu_precomputed_batch(
                tuple(values),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_zero_segment_dimension_is_refused_before_other_fields(self) -> None:
        values = list(_cpu_batch())
        values[2] = t.empty(2, 0, 4, dtype=t.int64)

        with pytest.raises(PrecomputedCudaStagingError, match="positive static n_seg"):
            validate_cpu_precomputed_batch(
                tuple(values),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_encoder_feature_rank_is_checked_before_dimension_lookup(self) -> None:
        values = list(_cpu_batch())
        values[2] = t.empty(2, 12, dtype=t.int64)

        with pytest.raises(PrecomputedCudaStagingError, match="enc_fids must have shape"):
            validate_cpu_precomputed_batch(
                tuple(values),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    @pytest.mark.parametrize(
        "state_tensor",
        [
            t.empty(1, device="meta"),
            t.zeros(1, requires_grad=True),
        ],
    )
    def test_nested_state_tensors_must_be_safe_cpu_metadata(
        self,
        state_tensor: t.Tensor,
    ) -> None:
        batch = _cpu_batch(
            state_overrides={"nested": {"tensor": state_tensor}},
        )

        with pytest.raises(
            PrecomputedCudaStagingError,
            match="state tensors must be on CPU and gradient-free",
        ):
            validate_cpu_precomputed_batch(
                batch,
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_label_dropout_state_is_refused_per_batch(self) -> None:
        batch = _cpu_batch(state_overrides={"label_dropout_rng_state": t.get_rng_state()})

        with pytest.raises(
            PrecomputedCudaStagingError,
            match="label_dropout_rng_state",
        ):
            validate_cpu_precomputed_batch(
                batch,
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_manifest_segment_dimension_is_checked_on_first_batch(self) -> None:
        with pytest.raises(
            PrecomputedCudaStagingError,
            match="manifest max_n_seg",
        ):
            validate_cpu_precomputed_batch(
                _cpu_batch(n_seg=3),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
                expected_n_seg=4,
            )

    def test_schema_rejects_shape_dtype_fp_and_state_key_drift(self) -> None:
        _, expected = validate_cpu_precomputed_batch(
            _cpu_batch(),
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        variants = [
            _cpu_batch(n_seg=4),
            _cpu_batch(fp_width=None),
            _cpu_batch(state_overrides={"new_key": "drift"}),
        ]

        for variant in variants:
            _, actual = validate_cpu_precomputed_batch(
                variant,
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )
            with pytest.raises(PrecomputedCudaStagingError, match="static CPU batch schema"):
                _assert_same_schema(expected, actual)


class TestByteAccountingAndBackpressure:
    def test_retained_bytes_deduplicate_views_and_include_state_tensors(self) -> None:
        root = t.zeros(32, dtype=t.float32)
        value = {
            "a": root[:16],
            "b": root[16:],
            "state": t.zeros(7, dtype=t.uint8),
        }

        assert retained_tensor_bytes(value) == (
            root.untyped_storage().nbytes() + value["state"].untyped_storage().nbytes()
        )

    def test_one_batch_larger_than_cap_is_rejected(self) -> None:
        stop = threading.Event()
        bounded = _ByteBoundedQueue(9, stop)

        with pytest.raises(PrecomputedCudaStagingError, match="exceeds"):
            bounded.put_data(object(), 10)

    def test_second_producer_blocks_until_bytes_are_released(self) -> None:
        stop = threading.Event()
        bounded = _ByteBoundedQueue(10, stop)
        bounded.put_data("first", 10)
        entered = threading.Event()
        finished = threading.Event()

        def put_second() -> None:
            entered.set()
            bounded.put_data("second", 10)
            finished.set()

        producer = threading.Thread(target=put_second)
        producer.start()
        assert entered.wait(timeout=1)
        assert not finished.wait(timeout=0.05)
        first = bounded.get()
        assert getattr(first, "batch", None) == "first"
        assert finished.wait(timeout=1)
        producer.join(timeout=1)
        assert not producer.is_alive()
        assert bounded.current_bytes == 10
        assert bounded.peak_bytes == 10

    @pytest.mark.parametrize("capacity", [10, 11])
    def test_exactly_one_batch_or_one_byte_more_still_blocks_the_second(
        self,
        capacity: int,
    ) -> None:
        stop = threading.Event()
        bounded = _ByteBoundedQueue(capacity, stop)
        bounded.put_data("first", 10)
        entered = threading.Event()
        finished = threading.Event()

        def put_second() -> None:
            entered.set()
            bounded.put_data("second", 10)
            finished.set()

        producer = threading.Thread(target=put_second)
        producer.start()
        assert entered.wait(timeout=1)
        assert not finished.is_set()
        assert bounded.current_bytes == 10
        assert bounded.peak_bytes == 10

        assert getattr(bounded.get(), "batch", None) == "first"
        assert finished.wait(timeout=1)
        producer.join(timeout=1)
        assert not producer.is_alive()
        assert bounded.current_bytes == 10
        assert bounded.peak_bytes == 10

    def test_stop_unblocks_a_full_queue_waiter(self) -> None:
        stop = threading.Event()
        bounded = _ByteBoundedQueue(10, stop)
        bounded.put_data("first", 10)
        stopped = threading.Event()

        def blocked_put() -> None:
            with pytest.raises(_StagingStopped):
                bounded.put_data("second", 10)
            stopped.set()

        producer = threading.Thread(target=blocked_put)
        producer.start()
        time.sleep(0.02)
        stop.set()
        bounded.stop()
        assert stopped.wait(timeout=1)
        producer.join(timeout=1)

    def test_control_message_is_dropped_after_stop(self) -> None:
        stop = threading.Event()
        bounded = _ByteBoundedQueue(10, stop)
        stop.set()

        bounded.put_control(staging_module._End())

        with pytest.raises(_StagingStopped):
            bounded.get()

    def test_ready_queue_retries_after_full(self) -> None:
        stop = threading.Event()

        class FullOnce:
            calls = 0
            message: object | None = None

            def put(self, message: object, *, timeout: float) -> None:
                assert timeout == 0.1
                self.calls += 1
                if self.calls == 1:
                    raise staging_module.queue.Full
                self.message = message

        ready = FullOnce()
        runtime = SimpleNamespace(stop_event=stop, ready_queue=ready)
        message = staging_module._End()

        staging_module._put_ready(cast(Any, runtime), message)

        assert ready.calls == 2
        assert ready.message is message

    def test_ready_queue_cancellation_raises_stopped(self) -> None:
        stop = threading.Event()

        class CancelWhileFull:
            def put(self, _message: object, *, timeout: float) -> None:
                assert timeout == 0.1
                stop.set()
                raise staging_module.queue.Full

        runtime = SimpleNamespace(
            stop_event=stop,
            ready_queue=CancelWhileFull(),
        )

        with pytest.raises(_StagingStopped):
            staging_module._put_ready(
                cast(Any, runtime),
                staging_module._End(),
            )


class TestPinnedSlotAndTransferContract:
    @pytest.mark.parametrize("fp_width", [None, 5])
    def test_slot_allocation_replays_schema_as_pinned_cpu_strides(
        self,
        fp_width: int | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, schema = validate_cpu_precomputed_batch(
            _cpu_batch(fp_width=fp_width),
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        calls: list[dict[str, Any]] = []

        def fake_empty_strided(
            shape: tuple[int, ...],
            stride: tuple[int, ...],
            **kwargs: Any,
        ) -> object:
            calls.append({"shape": shape, "stride": stride, **kwargs})
            return object()

        monkeypatch.setattr(t, "empty_strided", fake_empty_strided)

        slot = _allocate_slot(schema)

        expected_specs = [spec for spec in schema.tensors if spec is not None]
        assert len(calls) == len(expected_specs)
        assert (slot[8] is None) is (fp_width is None)
        for call, spec in zip(calls, expected_specs, strict=True):
            assert call == {
                "shape": spec.shape,
                "stride": spec.stride,
                "dtype": spec.dtype,
                "device": "cpu",
                "pin_memory": True,
            }

    @pytest.mark.parametrize("fp_width", [None, 5])
    def test_transfer_is_sync_into_pinned_then_async_to_fresh_cuda(
        self,
        fp_width: int | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch(fp_width=fp_width)
        pinned_copies: list[tuple[object, bool]] = []
        destination_copies: list[tuple[object, bool]] = []
        allocations: list[dict[str, Any]] = []

        class FakePinned:
            def __init__(self, source: t.Tensor) -> None:
                self.shape = source.shape
                self.dtype = source.dtype
                self._stride = source.stride()

            def stride(self) -> tuple[int, ...]:
                return self._stride

            def copy_(self, source: object, *, non_blocking: bool) -> None:
                pinned_copies.append((source, non_blocking))

        class FakeDestination:
            def copy_(self, source: object, *, non_blocking: bool) -> None:
                destination_copies.append((source, non_blocking))

        def fake_empty_strided(
            shape: tuple[int, ...],
            stride: tuple[int, ...],
            **kwargs: Any,
        ) -> FakeDestination:
            allocations.append({"shape": shape, "stride": stride, **kwargs})
            return FakeDestination()

        slot = cast(
            Any,
            tuple(None if source is None else FakePinned(source) for source in batch[:9]),
        )
        monkeypatch.setattr(t, "empty_strided", fake_empty_strided)

        staged = _stage_batch(slot, batch, device=t.device("cuda:7"))

        tensor_count = 8 if fp_width is None else 9
        assert len(pinned_copies) == tensor_count
        assert len(destination_copies) == tensor_count
        assert len(allocations) == tensor_count
        assert (staged[8] is None) is (fp_width is None)
        assert staged[9] is batch[9]
        assert all(non_blocking is False for _, non_blocking in pinned_copies)
        assert all(non_blocking is True for _, non_blocking in destination_copies)
        assert all(allocation["device"] == t.device("cuda:7") for allocation in allocations)

    @pytest.mark.parametrize("pinned_is_none", [False, True])
    def test_transfer_rejects_tensor_presence_drift(
        self,
        pinned_is_none: bool,
    ) -> None:
        batch_values = list(_cpu_batch())
        slot_values: list[object | None] = [None] * 9
        if pinned_is_none:
            assert batch_values[0] is not None
        else:
            slot_values[0] = object()
            batch_values[0] = None

        with pytest.raises(PrecomputedCudaStagingError, match="FP-target presence"):
            _stage_batch(
                cast(Any, tuple(slot_values)),
                cast(PrecomputedBatch, tuple(batch_values)),
                device=t.device("cuda:0"),
            )


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, t.Tensor):
        assert isinstance(actual, t.Tensor)
        assert actual.device == expected.device
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
        assert t.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        actual_dict = cast(dict[Any, Any], actual)
        expected_dict = cast(dict[Any, Any], expected)
        assert actual_dict.keys() == expected_dict.keys()
        for key in expected_dict:
            _assert_nested_equal(actual_dict[key], expected_dict[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_cpu_visible_bits_equal(
    actual: t.Tensor,
    expected: t.Tensor,
    *,
    label: str,
) -> None:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    actual_bytes = actual_cpu.contiguous().reshape(-1).view(t.uint8)
    expected_bytes = expected_cpu.contiguous().reshape(-1).view(t.uint8)
    mismatch = actual_bytes != expected_bytes
    if bool(mismatch.any()):
        first_byte_offsets = mismatch.nonzero(as_tuple=True)[0][:16].tolist()
        first_element = first_byte_offsets[0] // actual_cpu.element_size()
        actual_value = actual_cpu.reshape(-1)[first_element].item()
        expected_value = expected_cpu.reshape(-1)[first_element].item()
        raise AssertionError(
            f"{label} differs after CPU transfer: "
            f"byte_mismatches={int(mismatch.sum())}, "
            f"first_byte_offsets={first_byte_offsets}, "
            f"first_element={first_element}, "
            f"actual={actual_value!r}, expected={expected_value!r}"
        )


def _assert_cuda_batch_equal(
    actual: PrecomputedBatch,
    expected: PrecomputedBatch,
    *,
    device: t.device,
) -> None:
    field_names = (
        "inputs",
        "targets",
        "enc_fids",
        "enc_mags",
        "enc_mask",
        "enc_pcts",
        "seg_map",
        "seg_doc_id",
        "fp_targets",
    )
    for field_name, actual_tensor, expected_tensor in zip(
        field_names,
        actual[:9],
        expected[:9],
        strict=True,
    ):
        if expected_tensor is None:
            assert actual_tensor is None
            continue
        assert actual_tensor is not None
        assert actual_tensor.device == device
        assert actual_tensor.dtype == expected_tensor.dtype
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.stride() == expected_tensor.stride()
        assert actual_tensor.storage_offset() == expected_tensor.storage_offset()
        _assert_cpu_visible_bits_equal(
            actual_tensor,
            expected_tensor,
            label=field_name,
        )
    _assert_nested_equal(actual[9], expected[9])


def test_cuda_batch_equality_compares_cpu_visible_bytes() -> None:
    expected = _cpu_batch()
    actual = copy.deepcopy(expected)

    _assert_cuda_batch_equal(
        actual,
        expected,
        device=t.device("cpu"),
    )


def test_cpu_visible_bit_equality_accepts_scalar_tensors() -> None:
    _assert_cpu_visible_bits_equal(
        t.tensor(0.27964586019515991),
        t.tensor(0.27964586019515991),
        label="loss",
    )


def test_cuda_batch_equality_reports_exact_field_and_offset() -> None:
    expected = _cpu_batch()
    actual = copy.deepcopy(expected)
    actual[3][0, 0, 0] = 1

    with pytest.raises(
        AssertionError,
        match=r"enc_mags differs.*first_element=0.*actual=1, expected=0",
    ):
        _assert_cuda_batch_equal(
            actual,
            expected,
            device=t.device("cpu"),
        )


class _ParityConsumer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(32_768, 16)
        self.feature_embed = nn.Embedding(2_074, 16)
        self.magnitude_embed = nn.Embedding(2, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, batch: PrecomputedBatch) -> t.Tensor:
        (
            inputs,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            enc_pcts,
            seg_map,
            _seg_doc_id,
            fp_targets,
            _state,
        ) = batch
        feature_hidden = self.feature_embed(enc_fids)
        feature_hidden = feature_hidden + self.magnitude_embed(enc_mags)
        weights = enc_mask.unsqueeze(-1) * (enc_pcts.unsqueeze(-1) + 1)
        segment_hidden = (feature_hidden * weights).sum(2)
        segment_hidden = segment_hidden / weights.sum(2).clamp_min(1)
        token_hidden = self.token_embed(inputs)
        token_hidden = token_hidden + segment_hidden.gather(
            1,
            seg_map.unsqueeze(-1).expand(-1, -1, segment_hidden.shape[-1]),
        )
        prediction = self.output(token_hidden).squeeze(-1)
        loss = (prediction - targets.to(prediction.dtype).div(100)).square().mean()
        if fp_targets is not None:
            loss = loss + fp_targets.mean() * 1e-7
        return loss


class TestWorkerPipelineWithoutHardware:
    @pytest.mark.parametrize(
        (
            "device",
            "pinned_slots",
            "host_queue_bytes",
            "error_type",
            "message",
        ),
        [
            ("cpu", 1, 1, PrecomputedCudaStagingError, "explicitly indexed CUDA"),
            ("cuda", 1, 1, PrecomputedCudaStagingError, "explicitly indexed CUDA"),
            ("cuda:0", 0, 1, BeartypeCallHintParamViolation, "pinned_slots"),
            ("cuda:0", False, 1, BeartypeCallHintParamViolation, "pinned_slots"),
            ("cuda:0", 1, 0, BeartypeCallHintParamViolation, "host_queue_bytes"),
            ("cuda:0", 1, True, BeartypeCallHintParamViolation, "host_queue_bytes"),
        ],
    )
    def test_constructor_rejects_invalid_operational_values_before_threads(
        self,
        device: str,
        pinned_slots: object,
        host_queue_bytes: object,
        error_type: type[Exception],
        message: str,
    ) -> None:
        with pytest.raises(error_type, match=message):
            PrecomputedCudaStagingLoader(
                iter([_cpu_batch()]),
                device=device,
                host_queue_bytes=cast(Any, host_queue_bytes),
                pinned_slots=cast(Any, pinned_slots),
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

    def test_workers_exit_when_cancellation_precedes_entry(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        stop = threading.Event()
        stop.set()

        staging_module._producer_main(
            cast(
                Any,
                SimpleNamespace(
                    stop_event=stop,
                ),
            )
        )
        staging_module._h2d_main(
            cast(
                Any,
                SimpleNamespace(
                    device=t.device("cuda:0"),
                    stop_event=stop,
                ),
            )
        )

        assert fake_cuda_runtime["record"] == 0

    def test_consumer_retries_empty_ready_queue_while_workers_are_alive(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()

        def delayed() -> Iterator[PrecomputedBatch]:
            time.sleep(0.15)
            yield batch

        loader = PrecomputedCudaStagingLoader(
            delayed(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        try:
            _assert_nested_equal(next(loader), batch)
        finally:
            loader.close()

        assert fake_cuda_runtime["wait"] == 1

    def test_staging_enters_next_shard_before_any_current_batch_is_consumed(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        current_batches = [_cpu_batch(sentinel=index) for index in range(8)]
        next_batch = _cpu_batch(sentinel=8)
        entered_next_decode = threading.Event()

        def two_shards() -> Iterator[PrecomputedBatch]:
            yield from current_batches
            entered_next_decode.set()
            yield next_batch

        batch_bytes = retained_tensor_bytes(current_batches[0])
        loader = PrecomputedCudaStagingLoader(
            two_shards(),
            device="cuda:0",
            host_queue_bytes=batch_bytes * 10,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        try:
            assert entered_next_decode.wait(timeout=1)
            assert loader.stats().consumed_batches == 0
            assert loader.stats().produced_batches >= len(current_batches)
        finally:
            loader.close()

        assert loader.is_terminal()
        assert fake_cuda_runtime["record"] >= 1

    def test_four_batch_closable_route_enters_next_shard_with_one_step_left(
        self,
    ) -> None:
        prefetch_batches = 4
        fifth_requested = threading.Event()
        entered_next_decode = threading.Event()

        def two_shards() -> Iterator[int]:
            for index in range(8):
                if index == prefetch_batches:
                    fifth_requested.set()
                yield index
            entered_next_decode.set()
            yield 8

        loader = ClosableBatchPrefetcher(
            two_shards(),
            prefetch_batches=PositiveInt.parse(prefetch_batches),
            stop_callback=lambda: None,
            failure_check=lambda: None,
        )
        consumed: list[int] = []
        try:
            assert fifth_requested.wait(timeout=1)
            assert loader._queue.qsize() == prefetch_batches
            assert not entered_next_decode.is_set()

            consumed.extend(next(loader) for _ in range(prefetch_batches))
            assert entered_next_decode.wait(timeout=1)
            assert consumed == [0, 1, 2, 3]
            assert loader._queue.qsize() == prefetch_batches
            assert prefetch_batches == 524_288 // (16 * 1_024 * 8)
        finally:
            loader.close()

        assert not loader._thread.is_alive()

    @pytest.mark.parametrize("rank", [0, 1])
    @pytest.mark.parametrize("n_sae_features", [None, 16_384])
    def test_real_packed_loader_fifo_parity_across_shards_and_epochs(
        self,
        fake_cuda_runtime: dict[str, int],
        tmp_path: Path,
        rank: int,
        n_sae_features: int | None,
    ) -> None:
        packed_dir = tmp_path / "packed"
        selection_dir = tmp_path / "selection"
        _write_shards_with_selection(
            packed_dir,
            selection_dir,
            n_shards=2,
            n_rows=5,
            context_len=32,
            segs_per_row=3,
        )
        reference = precomputed_data_loader(
            packed_dir,
            B=2,
            feature_pipeline=_make_pipeline(16_384),
            device="cpu",
            selection_dir=selection_dir,
            n_sae_features=n_sae_features,
            ddp_rank=rank,
            ddp_world_size=2,
        )
        inner = precomputed_data_loader(
            packed_dir,
            B=2,
            feature_pipeline=_make_pipeline(16_384),
            device="cpu",
            selection_dir=selection_dir,
            n_sae_features=n_sae_features,
            ddp_rank=rank,
            ddp_world_size=2,
        )
        staged = PrecomputedCudaStagingLoader(
            inner,
            device="cuda:0",
            host_queue_bytes=32 * 1024 * 1024,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=32,
            expected_pad_f=5,
        )
        states: list[dict[str, Any]] = []
        try:
            for _ in range(6):
                expected_batch = next(reference)
                actual_batch = next(staged)
                _assert_nested_equal(actual_batch, expected_batch)
                states.append(actual_batch[9])
        finally:
            staged.close()
            reference.close()

        assert any(state["epoch"] > 1 for state in states)
        stats = staged.stats()
        assert stats.consumed_batches == 6
        assert 6 <= stats.staged_batches <= 8
        assert fake_cuda_runtime["record"] == stats.staged_batches

    def test_finite_fifo_handoff_and_metrics(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batches = [_cpu_batch(sentinel=index) for index in range(40)]
        batch_bytes = retained_tensor_bytes(batches[0])
        loader = PrecomputedCudaStagingLoader(
            iter(batches),
            device="cuda:0",
            host_queue_bytes=batch_bytes * 2,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        actual = list(loader)

        for actual_batch, expected_batch in zip(actual, batches, strict=True):
            _assert_nested_equal(actual_batch, expected_batch)
        stats = loader.stats()
        assert stats.produced_batches == 40
        assert stats.staged_batches == 40
        assert stats.consumed_batches == 40
        assert stats.host_queue_peak_bytes <= batch_bytes * 2
        assert fake_cuda_runtime == {
            "record": 40,
            "synchronize": 40,
            "wait": 40,
            "stream_sync": 0,
        }
        assert all(not thread.is_alive() for thread in loader._runtime.threads)

    def test_producer_error_is_rethrown_and_workers_join(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()

        def failing() -> Iterator[PrecomputedBatch]:
            yield batch
            raise LookupError("fake producer failure")

        loader = PrecomputedCudaStagingLoader(
            failing(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch) * 2,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        _assert_nested_equal(next(loader), batch)
        with pytest.raises(LookupError, match="fake producer failure"):
            next(loader)
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["synchronize"] == 1

    def test_noted_decode_failure_remains_primary_and_cancels_all_workers(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        cancel_event = threading.Event()
        remote_prefetch_terminal = threading.Event()
        inner_closed = threading.Event()
        failure = LookupError("synthetic packed decode failure")
        failure.add_note(
            "remote packed-body decode/validation context: "
            "path=/cache/shard_00017.safetensors.zst, logical_epoch=3"
        )

        def remote_prefetch_worker() -> None:
            cancel_event.wait()
            remote_prefetch_terminal.set()

        remote_worker = threading.Thread(
            target=remote_prefetch_worker,
            name="synthetic-remote-prefetch",
        )
        remote_worker.start()

        class DecodeFails(Iterator[PrecomputedBatch]):
            def __next__(self) -> PrecomputedBatch:
                raise failure

            def close(self) -> None:
                inner_closed.set()

        loader = PrecomputedCudaStagingLoader(
            DecodeFails(),
            device="cuda:0",
            host_queue_bytes=1,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
            remote_read_cancel_event=cancel_event,
        )

        with pytest.raises(LookupError, match="synthetic packed decode failure") as error:
            next(loader)

        remote_worker.join(timeout=1)
        assert error.value is failure
        assert any(
            "path=/cache/shard_00017.safetensors.zst, logical_epoch=3" in note
            for note in getattr(error.value, "__notes__", [])
        )
        assert cancel_event.is_set()
        assert inner_closed.is_set()
        assert remote_prefetch_terminal.is_set()
        assert not remote_worker.is_alive()
        assert loader.is_terminal()
        loader.close()
        loader.close()

    def test_schema_drift_is_rethrown_and_workers_join(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        first = _cpu_batch()
        drifted = _cpu_batch(n_seg=4)
        loader = PrecomputedCudaStagingLoader(
            iter([first, drifted]),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(first) * 3,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        _assert_nested_equal(next(loader), first)
        with pytest.raises(PrecomputedCudaStagingError, match="static CPU batch schema"):
            next(loader)
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["synchronize"] == 1

    def test_oversized_first_batch_fails_before_staging(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch) - 1,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(PrecomputedCudaStagingError, match="exceeds"):
            next(loader)
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0

    def test_mid_transfer_failure_orders_stream_before_slot_release(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()

        def fail_transfer(
            _slot: object,
            _batch: object,
            *,
            device: object,
        ) -> PrecomputedBatch:
            raise RuntimeError(f"fake H2D failure on {device}")

        monkeypatch.setattr(staging_module, "_stage_batch", fail_transfer)
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(RuntimeError, match="fake H2D failure"):
            next(loader)
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["stream_sync"] == 1

    def test_failure_delivery_tolerates_concurrent_cancellation(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()

        def fail_transfer(
            _slot: object,
            _batch: object,
            *,
            device: object,
        ) -> PrecomputedBatch:
            raise RuntimeError(f"undeliverable H2D failure on {device}")

        def cancel_instead_of_delivering(
            runtime: object,
            _message: object,
        ) -> None:
            runtime = cast(Any, runtime)
            runtime.stop_event.set()
            runtime.host_queue.stop()
            raise _StagingStopped

        monkeypatch.setattr(staging_module, "_stage_batch", fail_transfer)
        monkeypatch.setattr(staging_module, "_put_ready", cancel_instead_of_delivering)
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(StopIteration):
            next(loader)

        assert loader._runtime.closed is True
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["stream_sync"] == 1

    def test_duplicate_free_slot_on_failure_is_contained(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()

        def fail_transfer(
            _slot: object,
            _batch: object,
            *,
            device: object,
        ) -> PrecomputedBatch:
            raise RuntimeError(f"duplicate-slot H2D failure on {device}")

        monkeypatch.setattr(staging_module, "_get_free_slot", lambda _runtime: 0)
        monkeypatch.setattr(staging_module, "_stage_batch", fail_transfer)
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(RuntimeError, match="duplicate-slot H2D failure"):
            next(loader)

        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["stream_sync"] == 1

    def test_stream_synchronization_failure_replaces_transfer_error_with_note(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()

        class SyncFails:
            def synchronize(self) -> None:
                raise OSError("staging stream synchronization failed")

        def fail_transfer(
            _slot: object,
            _batch: object,
            *,
            device: object,
        ) -> PrecomputedBatch:
            raise RuntimeError(f"original transfer failure on {device}")

        monkeypatch.setattr(t.cuda, "Stream", lambda **_kwargs: SyncFails())
        monkeypatch.setattr(staging_module, "_stage_batch", fail_transfer)
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(
            OSError,
            match="staging stream synchronization failed",
        ) as error:
            next(loader)

        assert any(
            "original transfer failure" in note for note in getattr(error.value, "__notes__", [])
        )
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0

    def test_consumer_records_every_cuda_tensor_on_its_actual_stream(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()
        recorded_streams: list[object] = []

        class FakeDevice:
            type = "cuda"

        class FakeCudaTensor:
            device = FakeDevice()

            def record_stream(self, stream: object) -> None:
                recorded_streams.append(stream)

        fake_tensors = [FakeCudaTensor() for _ in range(9)]
        monkeypatch.setattr(
            staging_module,
            "retained_tensor_bytes",
            lambda _batch: 1,
        )
        monkeypatch.setattr(
            staging_module,
            "_walk_tensors",
            lambda value: iter(()) if isinstance(value, dict) else iter(fake_tensors),
        )
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=1,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        next(loader)
        loader.close()

        assert len(recorded_streams) == 9
        assert len({id(stream) for stream in recorded_streams}) == 1
        assert fake_cuda_runtime["wait"] == 1

    def test_consumer_record_stream_failure_closes_workers(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()

        class FakeDevice:
            type = "cuda"

        class RecordFails:
            device = FakeDevice()

            def record_stream(self, _stream: object) -> None:
                raise RuntimeError("record_stream failed")

        monkeypatch.setattr(
            staging_module,
            "retained_tensor_bytes",
            lambda _batch: 1,
        )
        monkeypatch.setattr(
            staging_module,
            "_walk_tensors",
            lambda value: iter(()) if isinstance(value, dict) else iter([RecordFails()]),
        )
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device="cuda:0",
            host_queue_bytes=1,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(RuntimeError, match="record_stream failed"):
            next(loader)

        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["wait"] == 1

    def test_consumer_close_failure_is_chained_from_record_stream_failure(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()

        class FakeDevice:
            type = "cuda"

        class RecordFails:
            device = FakeDevice()

            def record_stream(self, _stream: object) -> None:
                raise RuntimeError("record_stream root failure")

        class CloseFails(Iterator[PrecomputedBatch]):
            yielded = False

            def __next__(self) -> PrecomputedBatch:
                if self.yielded:
                    raise StopIteration
                self.yielded = True
                return batch

            def close(self) -> None:
                raise OSError("close after record_stream failed")

        monkeypatch.setattr(
            staging_module,
            "retained_tensor_bytes",
            lambda _batch: 1,
        )
        monkeypatch.setattr(
            staging_module,
            "_walk_tensors",
            lambda value: iter(()) if isinstance(value, dict) else iter([RecordFails()]),
        )
        loader = PrecomputedCudaStagingLoader(
            CloseFails(),
            device="cuda:0",
            host_queue_bytes=1,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(OSError, match="close after record_stream failed") as error:
            next(loader)

        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "record_stream root failure"
        assert loader._runtime.closed is True
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["wait"] == 1

    def test_malformed_internal_data_message_fails_loudly(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def malformed_h2d(runtime: object) -> None:
            runtime = cast(Any, runtime)
            runtime.ready_queue.put(_Data(object()))

        monkeypatch.setattr(staging_module, "_h2d_main", malformed_h2d)
        loader = PrecomputedCudaStagingLoader(
            iter([_cpu_batch()]),
            device="cuda:0",
            host_queue_bytes=1_000_000,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(PrecomputedCudaStagingError, match="malformed"):
            next(loader)

        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0

    def test_inner_close_failure_is_rethrown_after_workers_stop(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()

        class CloseFails(Iterator[PrecomputedBatch]):
            yielded = False

            def __next__(self) -> PrecomputedBatch:
                if self.yielded:
                    raise StopIteration
                self.yielded = True
                return batch

            def close(self) -> None:
                raise RuntimeError("inner close failed")

        loader = PrecomputedCudaStagingLoader(
            CloseFails(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        next(loader)

        with pytest.raises(RuntimeError, match="inner close failed"):
            loader.close()

        assert loader._runtime.closed is True
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 1

    def test_inner_close_failure_is_chained_from_producer_failure(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()

        class ProducerAndCloseFail(Iterator[PrecomputedBatch]):
            yielded = False

            def __next__(self) -> PrecomputedBatch:
                if self.yielded:
                    raise LookupError("producer failed")
                self.yielded = True
                return batch

            def close(self) -> None:
                raise RuntimeError("inner close also failed")

        loader = PrecomputedCudaStagingLoader(
            ProducerAndCloseFail(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        next(loader)

        with pytest.raises(RuntimeError, match="inner close also failed") as error:
            next(loader)

        assert isinstance(error.value.__cause__, LookupError)
        assert str(error.value.__cause__) == "producer failed"
        assert loader._runtime.closed is True
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 1

    def test_stopped_workers_without_message_end_iteration_cleanly(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def stop_without_message(runtime: object) -> None:
            runtime = cast(Any, runtime)
            runtime.stop_event.set()
            runtime.host_queue.stop()

        monkeypatch.setattr(staging_module, "_h2d_main", stop_without_message)
        monkeypatch.setattr(staging_module, "_producer_main", stop_without_message)
        loader = PrecomputedCudaStagingLoader(
            iter([_cpu_batch()]),
            device="cuda:0",
            host_queue_bytes=1_000_000,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        with pytest.raises(StopIteration):
            next(loader)

        assert loader._runtime.closed is True
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0

    def test_partial_thread_start_failure_stops_started_worker(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_start = threading.Thread.start
        started: list[threading.Thread] = []

        def fail_second_start(thread: threading.Thread) -> None:
            if started:
                raise RuntimeError("second thread failed to start")
            started.append(thread)
            original_start(thread)

        monkeypatch.setattr(threading.Thread, "start", fail_second_start)

        with pytest.raises(RuntimeError, match="second thread failed to start"):
            PrecomputedCudaStagingLoader(
                iter([_cpu_batch()]),
                device="cuda:0",
                host_queue_bytes=1_000_000,
                pinned_slots=1,
                expected_batch_size=2,
                expected_context_len=8,
                expected_pad_f=4,
            )

        assert len(started) == 1
        assert not started[0].is_alive()
        assert fake_cuda_runtime["record"] == 0

    def test_early_close_is_idempotent(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()

        def endless() -> Iterator[PrecomputedBatch]:
            while True:
                yield copy.deepcopy(batch)

        loader = PrecomputedCudaStagingLoader(
            endless(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch) * 2,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        next(loader)
        loader.close()
        loader.close()

        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        with pytest.raises(StopIteration):
            next(loader)
        assert fake_cuda_runtime["wait"] == 1

    def test_close_signals_remote_read_cancellation_before_join(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        cancel_event = threading.Event()
        entered = threading.Event()
        inner_closed = threading.Event()

        class CancellableRemoteRead(Iterator[PrecomputedBatch]):
            def __next__(self) -> PrecomputedBatch:
                entered.set()
                if not cancel_event.wait(timeout=5):
                    raise AssertionError("staging close did not cancel the remote read")
                raise RemoteMaterializationCancelled("synthetic remote read cancelled")

            def close(self) -> None:
                inner_closed.set()

        loader = PrecomputedCudaStagingLoader(
            CancellableRemoteRead(),
            device="cuda:0",
            host_queue_bytes=1_000_000,
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
            remote_read_cancel_event=cancel_event,
        )
        assert loader._runtime.stop_event is cancel_event
        assert entered.wait(timeout=1)

        started = time.monotonic()
        loader.close()

        assert time.monotonic() - started < 1
        assert cancel_event.is_set()
        assert inner_closed.is_set()
        assert loader._runtime.closed is True
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0

    def test_unreferenced_loader_finalizer_stops_both_workers(
        self,
        fake_cuda_runtime: dict[str, int],
    ) -> None:
        batch = _cpu_batch()

        def endless() -> Iterator[PrecomputedBatch]:
            while True:
                yield copy.deepcopy(batch)

        loader = PrecomputedCudaStagingLoader(
            endless(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch) * 2,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        next(loader)
        runtime = loader._runtime
        loader_ref = weakref.ref(loader)

        del loader
        gc.collect()

        assert loader_ref() is None
        assert runtime.closed is True
        assert all(not thread.is_alive() for thread in runtime.threads)
        assert fake_cuda_runtime["wait"] == 1

    def test_finalizer_logs_timeout_without_hiding_retryable_runtime(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()
        entered = threading.Event()
        release = threading.Event()
        logged: list[str] = []

        def blocked() -> Iterator[PrecomputedBatch]:
            entered.set()
            release.wait()
            yield batch

        monkeypatch.setattr(staging_module, "_JOIN_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(
            staging_module.logger,
            "error",
            lambda message: logged.append(str(message)),
        )
        loader = PrecomputedCudaStagingLoader(
            blocked(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        assert entered.wait(timeout=1)

        staging_module._finalize_runtime(loader._runtime)

        assert logged and "did not stop" in logged[0]
        assert loader._runtime.closed is False
        release.set()
        monkeypatch.setattr(staging_module, "_JOIN_TIMEOUT_SECONDS", 1.0)
        loader.close()
        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0

    def test_finalizer_logs_unexpected_cleanup_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        logged: list[str] = []

        def fail_stop(_runtime: object, *, raise_on_timeout: bool) -> None:
            assert raise_on_timeout is False
            raise RuntimeError("finalizer cleanup failed")

        monkeypatch.setattr(staging_module, "_stop_runtime", fail_stop)
        monkeypatch.setattr(
            staging_module.logger,
            "exception",
            lambda message: logged.append(str(message)),
        )

        staging_module._finalize_runtime(cast(Any, object()))

        assert logged == ["precomputed CUDA staging finalizer failed"]

    def test_join_timeout_is_loud_and_close_can_be_retried(
        self,
        fake_cuda_runtime: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _cpu_batch()
        entered = threading.Event()
        release = threading.Event()

        def blocked() -> Iterator[PrecomputedBatch]:
            entered.set()
            release.wait()
            yield batch

        monkeypatch.setattr(staging_module, "_JOIN_TIMEOUT_SECONDS", 0.01)
        loader = PrecomputedCudaStagingLoader(
            blocked(),
            device="cuda:0",
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        assert entered.wait(timeout=1)

        with pytest.raises(PrecomputedCudaStagingError, match="did not stop"):
            loader.close()
        release.set()
        monkeypatch.setattr(staging_module, "_JOIN_TIMEOUT_SECONDS", 1.0)
        loader.close()

        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        assert fake_cuda_runtime["record"] == 0


@requires_cuda
@pytest.mark.slow
class TestCudaStagingIntegration:
    @pytest.mark.parametrize("fp_width", [None, 5, 2_074])
    def test_exact_parity_for_every_tensor_and_state(
        self,
        fp_width: int | None,
    ) -> None:
        device = t.device("cuda", t.cuda.current_device())
        expected = [_cpu_batch(fp_width=fp_width, sentinel=index) for index in range(12)]
        cap = max(retained_tensor_bytes(batch) for batch in expected) * 3
        loader = PrecomputedCudaStagingLoader(
            iter(expected),
            device=device,
            host_queue_bytes=cap,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        try:
            actual = list(loader)
        finally:
            loader.close()

        assert len(actual) == len(expected)
        for actual_batch, expected_batch in zip(actual, expected, strict=True):
            _assert_cuda_batch_equal(actual_batch, expected_batch, device=device)
        stats = loader.stats()
        assert stats.produced_batches == len(expected)
        assert stats.staged_batches == len(expected)
        assert stats.consumed_batches == len(expected)
        assert stats.host_queue_peak_bytes <= cap

    def test_no_process_wide_cuda_synchronize_is_used(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        device = t.device("cuda", t.cuda.current_device())
        batch = _cpu_batch()
        loader = PrecomputedCudaStagingLoader(
            iter([batch]),
            device=device,
            host_queue_bytes=retained_tensor_bytes(batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        def forbidden_global_synchronize(*args: object, **kwargs: object) -> None:
            raise AssertionError("global torch.cuda.synchronize is forbidden")

        monkeypatch.setattr(t.cuda, "synchronize", forbidden_global_synchronize)
        try:
            actual = next(loader)
            _assert_cuda_batch_equal(actual, batch, device=device)
            with pytest.raises(StopIteration):
                next(loader)
        finally:
            loader.close()

    def test_loss_gradients_and_optimizer_step_equal_legacy_transfer(self) -> None:
        device = t.device("cuda", t.cuda.current_device())
        cpu_batch = _cpu_batch(fp_width=2_074)
        legacy_batch = cast(
            PrecomputedBatch,
            (
                *[None if tensor is None else tensor.to(device) for tensor in cpu_batch[:9]],
                copy.deepcopy(cpu_batch[9]),
            ),
        )
        staged_loader = PrecomputedCudaStagingLoader(
            iter([copy.deepcopy(cpu_batch)]),
            device=device,
            host_queue_bytes=retained_tensor_bytes(cpu_batch),
            pinned_slots=1,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        staged_batch = next(staged_loader)
        staged_loader.close()
        deterministic_was_enabled = t.are_deterministic_algorithms_enabled()
        deterministic_warn_only = t.is_deterministic_algorithms_warn_only_enabled()
        t.use_deterministic_algorithms(True)
        try:
            # CUDA embedding/gather backward otherwise uses nondeterministic
            # accumulation. On H200, even legacy-vs-legacy repeats differed by
            # one or two float32 ULPs. Select deterministic kernels so this
            # remains an exact staging parity test rather than an atomic-order
            # comparison; every assertion below is still byte-for-byte.
            t.manual_seed(842)
            legacy_model = _ParityConsumer().to(device)
            staged_model = copy.deepcopy(legacy_model)
            legacy_optimizer = t.optim.SGD(legacy_model.parameters(), lr=1e-3)
            staged_optimizer = t.optim.SGD(staged_model.parameters(), lr=1e-3)

            legacy_loss = legacy_model(legacy_batch)
            staged_loss = staged_model(staged_batch)
            _assert_cpu_visible_bits_equal(
                staged_loss,
                legacy_loss,
                label="loss",
            )
            legacy_loss.backward()
            staged_loss.backward()
            for (
                (legacy_name, legacy_parameter),
                (staged_name, staged_parameter),
            ) in zip(
                legacy_model.named_parameters(),
                staged_model.named_parameters(),
                strict=True,
            ):
                assert staged_name == legacy_name
                assert legacy_parameter.grad is not None
                assert staged_parameter.grad is not None
                assert staged_parameter.grad.device == legacy_parameter.grad.device
                assert staged_parameter.grad.dtype == legacy_parameter.grad.dtype
                assert staged_parameter.grad.shape == legacy_parameter.grad.shape
                assert staged_parameter.grad.stride() == legacy_parameter.grad.stride()
                _assert_cpu_visible_bits_equal(
                    staged_parameter.grad,
                    legacy_parameter.grad,
                    label=f"{staged_name}.grad",
                )
            legacy_optimizer.step()
            staged_optimizer.step()
            for (
                (legacy_name, legacy_parameter),
                (staged_name, staged_parameter),
            ) in zip(
                legacy_model.named_parameters(),
                staged_model.named_parameters(),
                strict=True,
            ):
                assert staged_name == legacy_name
                assert staged_parameter.device == legacy_parameter.device
                assert staged_parameter.dtype == legacy_parameter.dtype
                assert staged_parameter.shape == legacy_parameter.shape
                assert staged_parameter.stride() == legacy_parameter.stride()
                _assert_cpu_visible_bits_equal(
                    staged_parameter,
                    legacy_parameter,
                    label=staged_name,
                )
        finally:
            t.use_deterministic_algorithms(
                deterministic_was_enabled,
                warn_only=deterministic_warn_only,
            )

    def test_thousands_of_slot_wraps_survive_source_poison_and_allocator_churn(
        self,
    ) -> None:
        device = t.device("cuda", t.cuda.current_device())
        total = 1_000
        sources: dict[int, PrecomputedBatch] = {}

        def batches() -> Iterator[PrecomputedBatch]:
            for index in range(total):
                batch = _cpu_batch(
                    batch_size=1,
                    context_len=4,
                    n_seg=2,
                    pad_f=2,
                    fp_width=3,
                    sentinel=index,
                )
                sources[index] = batch
                yield batch

        prototype = _cpu_batch(
            batch_size=1,
            context_len=4,
            n_seg=2,
            pad_f=2,
            fp_width=3,
        )
        batch_bytes = retained_tensor_bytes(prototype)
        host_queue_batches = 64
        loader = PrecomputedCudaStagingLoader(
            batches(),
            device=device,
            host_queue_bytes=batch_bytes * host_queue_batches,
            pinned_slots=2,
            expected_batch_size=1,
            expected_context_len=4,
            expected_pad_f=2,
        )
        try:
            deep_queue_target = batch_bytes * 32
            deadline = time.monotonic() + 10
            while (
                loader.stats().host_queue_peak_bytes < deep_queue_target
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            assert loader.stats().host_queue_peak_bytes >= deep_queue_target
            for index, actual in enumerate(loader):
                assert actual[0][0, 0].item() == index * 10_000
                assert actual[7][0, 0].item() == index
                assert int(actual[2].max().item()) < 2_074
                assert int(actual[3].max().item()) < 2
                source = sources.pop(index)
                for source_tensor in source[:9]:
                    if source_tensor is not None:
                        source_tensor.fill_(123)
                assert actual[0][0, 0].item() == index * 10_000
                assert actual[7][0, 0].item() == index
                _ = [t.empty(257 + offset, dtype=t.uint8, device=device) for offset in range(7)]
        finally:
            loader.close()

        assert loader.stats().consumed_batches == total
        assert loader.stats().host_queue_peak_bytes <= (batch_bytes * host_queue_batches)

    def test_producer_exception_reaches_consumer_and_workers_close(self) -> None:
        device = t.device("cuda", t.cuda.current_device())
        batch = _cpu_batch()

        def failing() -> Iterator[PrecomputedBatch]:
            yield batch
            raise ValueError("producer exploded")

        loader = PrecomputedCudaStagingLoader(
            failing(),
            device=device,
            host_queue_bytes=retained_tensor_bytes(batch) * 2,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )

        _assert_cuda_batch_equal(next(loader), batch, device=device)
        with pytest.raises(ValueError, match="producer exploded"):
            next(loader)
        assert all(not thread.is_alive() for thread in loader._runtime.threads)

    def test_early_close_is_idempotent_and_releases_workers(self) -> None:
        device = t.device("cuda", t.cuda.current_device())
        batch = _cpu_batch()

        def endless() -> Iterator[PrecomputedBatch]:
            while True:
                yield copy.deepcopy(batch)

        loader = PrecomputedCudaStagingLoader(
            endless(),
            device=device,
            host_queue_bytes=retained_tensor_bytes(batch) * 2,
            pinned_slots=2,
            expected_batch_size=2,
            expected_context_len=8,
            expected_pad_f=4,
        )
        next(loader)
        loader.close()
        loader.close()

        assert all(not thread.is_alive() for thread in loader._runtime.threads)
        with pytest.raises(StopIteration):
            next(loader)


@requires_cuda
@pytest.mark.slow
class TestPackedLoaderLegacyParity:
    @pytest.mark.parametrize(
        (
            "n_shards",
            "rows_per_shard",
            "batch_size",
            "rank",
            "world_size",
            "n_batches",
        ),
        [
            (1, 8, 4, 0, 1, 4),
            (2, 5, 4, 0, 1, 5),
            (2, 5, 2, 0, 2, 5),
            (2, 5, 2, 1, 2, 5),
        ],
    )
    @pytest.mark.parametrize("n_sae_features", [None, 16_384])
    def test_same_shard_cross_shard_cross_epoch_and_rank_parity(
        self,
        tmp_path,
        n_shards: int,
        rows_per_shard: int,
        batch_size: int,
        rank: int,
        world_size: int,
        n_batches: int,
        n_sae_features: int | None,
    ) -> None:
        packed_dir = tmp_path / "packed"
        selection_dir = tmp_path / "selection"
        _write_shards_with_selection(
            packed_dir,
            selection_dir,
            n_shards=n_shards,
            n_rows=rows_per_shard,
            context_len=32,
            segs_per_row=3,
        )
        device = t.device("cuda", t.cuda.current_device())
        reference = precomputed_data_loader(
            packed_dir,
            B=batch_size,
            feature_pipeline=_make_pipeline(16_384),
            device=str(device),
            selection_dir=selection_dir,
            n_sae_features=n_sae_features,
            ddp_rank=rank,
            ddp_world_size=world_size,
        )
        cpu_inner = precomputed_data_loader(
            packed_dir,
            B=batch_size,
            feature_pipeline=_make_pipeline(16_384),
            device="cpu",
            selection_dir=selection_dir,
            n_sae_features=n_sae_features,
            ddp_rank=rank,
            ddp_world_size=world_size,
        )
        staged = PrecomputedCudaStagingLoader(
            cpu_inner,
            device=device,
            host_queue_bytes=32 * 1024 * 1024,
            pinned_slots=2,
            expected_batch_size=batch_size,
            expected_context_len=32,
            expected_pad_f=5,
        )
        try:
            for _ in range(n_batches):
                expected_batch = next(reference)
                actual_batch = next(staged)
                _assert_cuda_batch_equal(actual_batch, expected_batch, device=device)
        finally:
            staged.close()
            reference.close()

    def test_fresh_and_resume_batches_match_legacy_exactly(self, tmp_path) -> None:
        packed_dir = tmp_path / "packed"
        selection_dir = tmp_path / "selection"
        _write_shards_with_selection(
            packed_dir,
            selection_dir,
            n_shards=2,
            n_rows=5,
            context_len=32,
            segs_per_row=3,
        )
        initial = precomputed_data_loader(
            packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(16_384),
            device="cpu",
            selection_dir=selection_dir,
        )
        resume_state = next(initial)[9]
        initial.close()
        device = t.device("cuda", t.cuda.current_device())
        reference = precomputed_data_loader(
            packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(16_384),
            device=str(device),
            selection_dir=selection_dir,
            resume_state_dict=resume_state,
        )
        cpu_inner = precomputed_data_loader(
            packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(16_384),
            device="cpu",
            selection_dir=selection_dir,
            resume_state_dict=resume_state,
        )
        staged = PrecomputedCudaStagingLoader(
            cpu_inner,
            device=device,
            host_queue_bytes=8 * 1024 * 1024,
            pinned_slots=2,
            expected_batch_size=4,
            expected_context_len=32,
            expected_pad_f=5,
        )
        try:
            for _ in range(5):
                _assert_cuda_batch_equal(
                    next(staged),
                    next(reference),
                    device=device,
                )
        finally:
            staged.close()
            reference.close()
