"""Native-checkpoint prepare/write integration for topology migration."""

from __future__ import annotations

import copy
import json
import math
import shutil
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_json_dump, atomic_torch_save
from nanochat.gpt import OptimizerKind
from scripts.train.conditional_encdec_train import (
    _checkpoint_user_config,
    build_encdec_meta_model_from_training_cfg,
)
from steering.conditional.encoder_decoder.topology_resume import (
    PackedRowInventory,
    RowReference,
)
from steering.conditional.feature_vocab import (
    META_KEY as FEATURE_VOCAB_META_KEY,
)
from steering.conditional.feature_vocab import (
    VOCAB_IDS_KEY,
    VOCAB_N_SAE_KEY,
    FeatureVocab,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    SelectionShard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.tools.migrate_encdec_train_state import (
    OptimizerLayout,
    _adam_is_replicated,
    _assert_files_unchanged,
    _build_optimizer_layout,
    _canonical_groups_at_checkpoint,
    _file_identity,
    _safe_mmap_load,
    _setup_optimizer,
    _validate_model_contract,
    _validate_optimizer_shards,
    execute_migration,
    prepare_migration,
)
from tests.encdec_test_helpers import _make_test_shard

TRAINING_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")
SOURCE_WORLD = 2
DESTINATION_WORLD = 3
STEP = 4
BASE_VOCAB = 32
N_SAE_FEATURES = 16
CONTEXT_LEN = 8
DEVICE_BATCH_SIZE = 1
FEATURE_VOCAB = FeatureVocab(
    n_sae_features=N_SAE_FEATURES,
    original_ids=(0, 2, 4, 6, 8, 10, 12, 14),
)


def _tiny_config(base: DictConfig, *, world_size: int, identity: str) -> DictConfig:
    cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(base, resolve=True)))
    cfg.run = identity
    cfg.model_tag = identity
    cfg.device_type = "cuda"
    cfg.depth = 2
    cfg.aspect_ratio = 8
    cfg.head_dim = 8
    cfg.window_pattern = "L"
    cfg.max_seq_len = CONTEXT_LEN
    cfg.encoder.n_embd = 16
    cfg.encoder.depth = 1
    cfg.encoder.n_head = None
    cfg.encoder.n_kv_head = None
    cfg.encoder.head_dim = 8
    cfg.cross_attention.frequency = 1
    cfg.cross_attention.n_head = None
    cfg.cross_attention.n_kv_head = None
    cfg.device_batch_size = DEVICE_BATCH_SIZE
    cfg.total_batch_size = world_size * DEVICE_BATCH_SIZE * CONTEXT_LEN
    cfg.num_iterations = 10
    cfg.target_param_data_ratio = 10.0
    cfg.fp8 = False
    cfg.label_dropout = 0.0
    cfg.use_precomputed_batches = True
    cfg.feature_mask = {"mode": "whitelist", "ids": list(FEATURE_VOCAB.original_ids)}
    cfg.feature_mask_freq_min = None
    cfg.feature_mask_freq_max = None
    cfg.feature_clusters = None
    cfg.feature_delta = None
    cfg.feature_prediction = None
    cfg.initialize_encoder_embeddings_from_sae = None
    cfg.remote_training = None
    cfg.resume_from_step = -1
    return cfg


def _model_state(model: Any) -> dict[str, t.Tensor]:
    state = {
        name: t.zeros(tuple(tensor.shape), dtype=tensor.dtype)
        for name, tensor in model.state_dict().items()
    }
    for name, tensor in state.items():
        if (
            name == "transformer.wte.weight"
            or name.startswith(("value_embeds.", "encoder.value_embeds."))
            or name in {
                "encoder.feature_embed.weight",
                "encoder.magnitude_embed.weight",
            }
        ):
            state[name] = tensor.to(t.bfloat16)
    state[VOCAB_IDS_KEY] = t.tensor(FEATURE_VOCAB.original_ids, dtype=t.int64)
    state[VOCAB_N_SAE_KEY] = t.tensor(N_SAE_FEATURES, dtype=t.int64)
    return state


def _logical_tensor(
    shape: tuple[int, ...],
    offset: int,
    *,
    dtype: t.dtype = t.float32,
) -> t.Tensor:
    values = t.arange(math.prod(shape), dtype=t.float32).reshape(shape)
    return values.add_(offset).to(dtype)


def _source_optimizer_shards(
    layout: OptimizerLayout,
    groups: list[dict[str, Any]],
    *,
    world_size: int,
) -> tuple[dict[str, Any], ...]:
    shards = [{"state": {}, "param_groups": copy.deepcopy(groups)} for _rank in range(world_size)]
    for group in layout.groups:
        if group.kind == OptimizerKind.ADAMW.value:
            for param_id in group.param_ids:
                parameter = layout.parameters[param_id]
                exp_avg = _logical_tensor(
                    parameter.shape,
                    100 * param_id,
                    dtype=parameter.dtype,
                )
                exp_avg_sq = _logical_tensor(
                    parameter.shape,
                    100 * param_id + 50,
                    dtype=parameter.dtype,
                )
                replicated = _adam_is_replicated(parameter, world_size)
                for rank, shard in enumerate(shards):
                    if replicated:
                        rank_avg = exp_avg.clone()
                        rank_sq = exp_avg_sq.clone()
                    else:
                        rank_rows = parameter.shape[0] // world_size
                        start = rank * rank_rows
                        rank_avg = exp_avg[start : start + rank_rows].clone()
                        rank_sq = exp_avg_sq[start : start + rank_rows].clone()
                    shard["state"][param_id] = {
                        "step": STEP,
                        "exp_avg": rank_avg,
                        "exp_avg_sq": rank_sq,
                    }
            continue

        first_id = group.param_ids[0]
        parameter = layout.parameters[first_id]
        chunk = math.ceil(len(group.param_ids) / world_size)
        second_tail = (
            (parameter.shape[-2], 1)
            if parameter.shape[-2] >= parameter.shape[-1]
            else (1, parameter.shape[-1])
        )
        logical_momentum = _logical_tensor(
            (len(group.param_ids), *parameter.shape),
            10_000 + first_id,
            dtype=parameter.dtype,
        )
        logical_second = _logical_tensor(
            (len(group.param_ids), *second_tail),
            20_000 + first_id,
            dtype=parameter.dtype,
        )
        for rank, shard in enumerate(shards):
            start = rank * chunk
            owned = min(chunk, max(0, len(group.param_ids) - start))
            momentum = t.zeros((chunk, *parameter.shape), dtype=parameter.dtype)
            second = t.zeros((chunk, *second_tail), dtype=parameter.dtype)
            if owned:
                momentum[:owned].copy_(logical_momentum[start : start + owned])
                second[:owned].copy_(logical_second[start : start + owned])
            shard["state"][first_id] = {
                "momentum_buffer": momentum,
                "second_momentum_buffer": second,
            }
    return tuple(shards)


def _write_precomputed_artifacts(tmp_path: Path) -> tuple[Path, Path, PackedRowInventory]:
    packed_dir = tmp_path / "packed"
    selection_dir = packed_dir / "sel" / "selection-e2e"
    packed_dir.mkdir()
    selection_dir.mkdir(parents=True)
    packed_paths: list[Path] = []
    for index in range(2):
        packed_shard = _make_test_shard(
            n_rows=6,
            context_len=CONTEXT_LEN,
            segs_per_row=2,
            feats_per_seg=2,
            n_features=N_SAE_FEATURES,
            seed=index + 1,
        )
        stem = f"shard_{index:05d}"
        packed_paths.append(save_packed_row_shard(packed_shard, packed_dir / stem))
        offsets = t.zeros(packed_shard.total_segments + 1, dtype=t.int32)
        save_selection_shard(
            SelectionShard(
                sel_offsets=offsets,
                sel_fids=t.empty(0, dtype=t.uint16),
                sel_buckets=t.empty(0, dtype=t.uint8),
                sel_pcts=t.empty(0, dtype=t.uint8),
                neg_offsets=offsets.clone(),
                neg_fids=t.empty(0, dtype=t.uint16),
            ),
            selection_dir / stem,
        )
    write_manifest(
        selection_dir,
        "selection-e2e",
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
            }
        },
        base_shard_paths=packed_paths,
    )
    return packed_dir, selection_dir, PackedRowInventory.from_directory(packed_dir)


def _source_dataloader_envelope(inventory: PackedRowInventory) -> dict[str, Any]:
    rank_states: list[dict[str, Any]] = []
    boundary = RowReference(1, 0, 0)
    for rank in range(SOURCE_WORLD):
        first = inventory.first_owned_at_or_after(
            boundary,
            rank=rank,
            world_size=SOURCE_WORLD,
        )
        stream = inventory.consecutive_owned(
            first,
            rank=rank,
            world_size=SOURCE_WORLD,
            count=5,
        )
        pending = stream[-1]
        rank_states.append(
            {
                "batch_idx": 4,
                "shard": inventory.shards[pending.shard_index].name,
                "shard_batch_offset": 1,
                "epoch": pending.epoch,
                "max_seg_size": CONTEXT_LEN,
                "pad_F": 4,
                "ddp_world_size": SOURCE_WORLD,
                "batch_rows": [list(inventory.to_tuple(pending))],
                "feature_pipeline_rng_state": [],
            }
        )
    return {"ddp_world_size": SOURCE_WORLD, "rank_states": rank_states}


@pytest.mark.parametrize("head", [None, {"block": "mlp", "head_layers": 2}, {"block": "transformer", "head_layers": 2}])
def test_prepare_and_execute_native_checkpoint_end_to_end(
    tmp_path: Path,
    hydra_compose,
    head,
) -> None:
    base = hydra_compose(TRAINING_CONFIG_DIR, "conditional_encdec_train")
    source_cfg = _tiny_config(base, world_size=SOURCE_WORLD, identity="tiny_w2")
    destination_cfg = _tiny_config(
        base,
        world_size=DESTINATION_WORLD,
        identity="tiny_w3",
    )
    source_cfg.feature_prediction = head
    destination_cfg.feature_prediction = head
    source_dir = tmp_path / "source"
    destination_dir = tmp_path / "destination"
    source_dir.mkdir()
    source_dir.chmod(0o750)
    packed_dir, selection_dir, inventory = _write_precomputed_artifacts(tmp_path)

    model = build_encdec_meta_model_from_training_cfg(
        source_cfg,
        base_vocab=BASE_VOCAB,
        n_sae_features=N_SAE_FEATURES,
        feature_vocab=FEATURE_VOCAB,
    )
    model_state = _model_state(model)
    _validate_model_contract(model, model, model_state)
    for wrong_dtype_name, wrong_dtype in (
        ("encoder.feature_embed.weight", t.float32),
        ("lm_head.weight", t.bfloat16),
    ):
        wrong_dtype_state = dict(model_state)
        wrong_dtype_state[wrong_dtype_name] = wrong_dtype_state[wrong_dtype_name].to(
            wrong_dtype
        )
        with pytest.raises(ValueError, match="shape/dtype"):
            _validate_model_contract(model, model, wrong_dtype_state)
    model_path = source_dir / f"model_{STEP:06d}.pt"
    atomic_torch_save(model_state, model_path)
    source_optimizer = _setup_optimizer(
        model,
        source_cfg,
        total_batch_size=int(source_cfg.total_batch_size),
        base_vocab=BASE_VOCAB,
        n_sae_features=N_SAE_FEATURES,
        feature_vocab=FEATURE_VOCAB,
    )
    layout = _build_optimizer_layout(model, model_state, source_optimizer)
    source_groups = _canonical_groups_at_checkpoint(
        source_optimizer,
        source_cfg,
        step=STEP,
    )
    source_shards = _source_optimizer_shards(
        layout,
        source_groups,
        world_size=SOURCE_WORLD,
    )
    _validate_optimizer_shards(
        source_shards,
        layout=layout,
        canonical_groups=source_groups,
        step=STEP,
    )
    for rank, shard in enumerate(source_shards):
        atomic_torch_save(shard, source_dir / f"optim_{STEP:06d}_rank{rank}.pt")

    atomic_json_dump(
        {
            "step": STEP,
            "val_bpb": 1.0,
            "model_config": asdict(model.dec_config),
            "user_config": _checkpoint_user_config(source_cfg),
            "device_batch_size": DEVICE_BATCH_SIZE,
            "max_seq_len": CONTEXT_LEN,
            "dataloader_state_dict": _source_dataloader_envelope(inventory),
            "loop_state": {"smooth_train_loss": 1.0},
            FEATURE_VOCAB_META_KEY: FEATURE_VOCAB.metadata_dict(),
        },
        source_dir / f"meta_{STEP:06d}.json",
    )
    tool_cfg = OmegaConf.create(
        {
            "source_checkpoint_dir": str(source_dir),
            "source_step": STEP,
            "packed_row_dir": str(packed_dir),
            "selection_dir": str(selection_dir),
            "destination_dir": str(destination_dir),
            "destination_world_size": DESTINATION_WORLD,
            "dry_run": True,
            "max_replay_rows": None,
            "destination_training": destination_cfg,
        }
    )

    guarded_cfg = cast(
        DictConfig,
        OmegaConf.create(OmegaConf.to_container(tool_cfg, resolve=True)),
    )
    guarded_cfg.max_replay_rows = 1
    with pytest.raises(ValueError, match="exceeding max_replay_rows=1"):
        prepare_migration(guarded_cfg)
    assert not destination_dir.exists()

    noncanonical_selection = tmp_path / "moved-selection"
    shutil.copytree(selection_dir, noncanonical_selection)
    noncanonical_cfg = cast(
        DictConfig,
        OmegaConf.create(OmegaConf.to_container(tool_cfg, resolve=True)),
    )
    noncanonical_cfg.selection_dir = str(noncanonical_selection)
    with pytest.raises(ValueError, match="canonical.*sel"):
        prepare_migration(noncanonical_cfg)

    selection_manifest_path = selection_dir / "manifest.json"
    valid_selection_manifest = _safe_json(selection_manifest_path)
    invalid_selection_manifest = copy.deepcopy(valid_selection_manifest)
    invalid_selection_manifest["checksum"] = "wrong-selection"
    atomic_json_dump(invalid_selection_manifest, selection_manifest_path)
    with pytest.raises(ValueError, match="checksum does not match"):
        prepare_migration(tool_cfg)
    atomic_json_dump(valid_selection_manifest, selection_manifest_path)

    meta_path = source_dir / f"meta_{STEP:06d}.json"
    valid_metadata = _safe_json(meta_path)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata["device_batch_size"] = DEVICE_BATCH_SIZE + 1
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="device_batch_size disagrees"):
        prepare_migration(tool_cfg)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata["max_seq_len"] = CONTEXT_LEN + 1
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="max_seq_len disagrees"):
        prepare_migration(tool_cfg)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata["step"] = float(STEP)
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="metadata step"):
        prepare_migration(tool_cfg)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata["model_config"]["window_pattern"] = "S"
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="model_config disagrees"):
        prepare_migration(tool_cfg)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata[FEATURE_VOCAB_META_KEY]["sha256"] = "0" * 64
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="feature vocabulary disagrees"):
        prepare_migration(tool_cfg)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata["loop_state"] = None
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="mapping loop_state"):
        prepare_migration(tool_cfg)
    invalid_metadata = copy.deepcopy(valid_metadata)
    invalid_metadata["dataloader_state_dict"]["rank_states"][0]["batch_idx"] = STEP + 1
    invalid_metadata["dataloader_state_dict"]["rank_states"][1]["batch_idx"] = STEP + 1
    atomic_json_dump(invalid_metadata, meta_path)
    with pytest.raises(ValueError, match="batch_idx disagrees"):
        prepare_migration(tool_cfg)
    atomic_json_dump(valid_metadata, meta_path)

    prepared = prepare_migration(tool_cfg)
    assert not destination_dir.exists()
    assert prepared.plan.step == STEP
    assert prepared.plan.grad_accum_steps == 1
    assert prepared.plan.lr_scale_ratio == pytest.approx(math.sqrt(3 / 2))
    assert prepared.plan.dataloader.replay_rows == 2

    published = execute_migration(prepared)
    assert published == destination_dir
    assert stat.S_IMODE(destination_dir.stat().st_mode) == 0o750
    assert {path.name for path in destination_dir.iterdir()} == {
        f"model_{STEP:06d}.pt",
        f"meta_{STEP:06d}.json",
        f"optim_{STEP:06d}_rank0.pt",
        f"optim_{STEP:06d}_rank1.pt",
        f"optim_{STEP:06d}_rank2.pt",
        "topology_migration_manifest.json",
    }
    assert (destination_dir / model_path.name).read_bytes() == model_path.read_bytes()

    target_shards = tuple(
        _safe_mmap_load(destination_dir / f"optim_{STEP:06d}_rank{rank}.pt")
        for rank in range(DESTINATION_WORLD)
    )
    _validate_optimizer_shards(
        target_shards,
        layout=prepared.optimizer_layout,
        canonical_groups=prepared.target_param_groups,
        step=STEP,
    )
    metadata = _safe_json(destination_dir / f"meta_{STEP:06d}.json")
    assert metadata["step"] == STEP
    assert metadata["user_config"]["resume_from_step"] == STEP
    assert metadata["user_config"]["total_batch_size"] == 3 * CONTEXT_LEN
    assert metadata["dataloader_state_dict"]["ddp_world_size"] == DESTINATION_WORLD
    assert metadata["topology_migration"]["dataloader_state"]["skipped_rows"] == 0
    assert metadata[FEATURE_VOCAB_META_KEY] == FEATURE_VOCAB.metadata_dict()
    if head is None:
        assert "feature_prediction_config" not in metadata
    else:
        assert metadata["feature_prediction_config"] == metadata["user_config"]["feature_prediction"]
        assert metadata["feature_prediction_config"]["block"] == head["block"]
        assert metadata["feature_prediction_config"]["head_layers"] == head["head_layers"]

    with pytest.raises(FileExistsError, match="destination already exists"):
        execute_migration(prepared)


def _safe_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def test_checkpoint_source_identity_fence_detects_in_place_change(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"before")
    identities = {source: _file_identity(source)}

    source.write_bytes(b"after-with-different-size")

    with pytest.raises(ValueError, match="source files changed"):
        _assert_files_unchanged(identities)
