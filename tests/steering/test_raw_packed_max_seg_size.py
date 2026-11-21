"""Issue #723: raw packed-row ``max_seg_size`` end-to-end contracts.

Packed-row producers persist the largest observed segment length, not an
FP8-aligned allocation size. These tests keep that raw value intact across the
producer/backfill, manifest, precomputed-loader state, eager model, optimizer,
and resume boundaries.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.preprocess.backfill_packed_row_shapes import backfill_packed_row_shapes
from scripts.preprocess.precompute_packed_rows import _compute_shard_max_dims
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    save_packed_row_shard,
    save_selection_shard,
)
from tests.encdec_test_helpers import build_threshold_pipeline

RAW_MAX_SEG_SIZE = 17
ALIGNED_MAX_SEG_SIZE = 32
CONTEXT_LEN = 2 * RAW_MAX_SEG_SIZE
BATCH_SIZE = 2
N_SAE_FEATURES = 64
NUM_BUCKETS = 4
VOCAB_SIZE = 128


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    """Use the supported CPU backward path for the real-model parity test."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _packed_shard() -> PackedRowShard:
    """Two rows, each with two exactly-17-token segments."""
    generator = t.Generator().manual_seed(723)
    n_rows = BATCH_SIZE
    n_segments = n_rows * 2
    features_per_segment = 3
    n_features = n_segments * features_per_segment
    return PackedRowShard(
        tokens=t.randint(
            1,
            VOCAB_SIZE,
            (n_rows, CONTEXT_LEN + 1),
            dtype=t.uint16,
            generator=generator,
        ),
        row_seg_offsets=t.tensor([0, 2, 4], dtype=t.int32),
        seg_token_start=t.tensor([0, 17, 0, 17], dtype=t.uint16),
        seg_token_end=t.tensor([17, 34, 17, 34], dtype=t.uint16),
        seg_doc_id=t.zeros(n_segments, dtype=t.uint8),
        seg_feat_offsets=t.arange(
            0,
            n_features + 1,
            features_per_segment,
            dtype=t.int32,
        ),
        feature_ids=t.arange(n_features, dtype=t.int64).to(t.uint16),
        feature_pcts=t.arange(1, n_features + 1, dtype=t.int64).to(t.uint8),
    )


def _selection_shard(base: PackedRowShard) -> SelectionShard:
    """Select every synthetic feature with a valid non-padding bucket."""
    return SelectionShard(
        sel_offsets=base.seg_feat_offsets.to(t.int32),
        sel_fids=base.feature_ids.clone(),
        sel_buckets=t.ones(base.total_features, dtype=t.uint8),
        sel_pcts=base.feature_pcts.clone(),
        neg_offsets=t.zeros(base.total_segments + 1, dtype=t.int32),
        neg_fids=t.empty(0, dtype=t.uint16),
    )


def _write_precomputed_artifacts(root: Path) -> tuple[Path, Path]:
    packed_dir = root / "packed"
    selection_dir = packed_dir / "sel" / "issue723"
    packed_dir.mkdir(parents=True)
    selection_dir.mkdir(parents=True)

    base = _packed_shard()
    save_packed_row_shard(base, packed_dir / "shard_00000")
    save_selection_shard(_selection_shard(base), selection_dir / "shard_00000")
    (packed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": "issue723-packed",
                "kind": "packed_rows",
                "metadata": {
                    "type": "packed_rows",
                    "context_len": CONTEXT_LEN,
                    "max_n_seg": 2,
                    "max_seg_size": RAW_MAX_SEG_SIZE,
                },
            }
        )
    )
    write_manifest(
        selection_dir,
        "issue723-selection",
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
            },
            "feature_delta": None,
            "feature_mask": None,
        },
    )
    return packed_dir, selection_dir


def _pipeline():
    boundaries = (
        t.linspace(
            1 / NUM_BUCKETS,
            1.0 - 1 / NUM_BUCKETS,
            NUM_BUCKETS - 1,
            dtype=t.float64,
        )
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries, num_buckets=NUM_BUCKETS)


def _loader(
    packed_dir: Path,
    selection_dir: Path,
    *,
    resume_state_dict: dict[str, Any] | None = None,
):
    return precomputed_data_loader(
        packed_row_dir=packed_dir,
        B=BATCH_SIZE,
        feature_pipeline=_pipeline(),
        device="cpu",
        selection_dir=selection_dir,
        effective_buckets=NUM_BUCKETS,
        n_sae_features=N_SAE_FEATURES,
        resume_state_dict=resume_state_dict,
    )


def _model() -> EncoderDecoderGPT:
    t.manual_seed(723)
    model = EncoderDecoderGPT(
        GPTConfig(
            sequence_len=CONTEXT_LEN,
            vocab_size=VOCAB_SIZE,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=64,
            window_pattern="L",
        ),
        EncoderConfig(
            n_embd=32,
            depth=1,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        ),
        CrossAttentionConfig(frequency=1, n_head=2, n_kv_head=2),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    model.init_weights()
    return model


def _loss(
    model: EncoderDecoderGPT,
    batch: tuple,
    *,
    static_max_seg_size: int | None,
) -> t.Tensor:
    inputs, targets, enc_fids, enc_mags, enc_mask, _, seg_map, _, _, _ = batch
    result = model(
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        static_max_seg_size=static_max_seg_size,
    )
    assert isinstance(result, t.Tensor)
    return result


def _parameter_grads(model: EncoderDecoderGPT) -> dict[str, t.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _assert_optimizer_states_exact(
    left: t.optim.Optimizer,
    right: t.optim.Optimizer,
) -> None:
    left_states = list(left.state.values())
    right_states = list(right.state.values())
    assert len(left_states) == len(right_states)
    for left_state, right_state in zip(left_states, right_states, strict=True):
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            left_value = left_state[key]
            right_value = right_state[key]
            if isinstance(left_value, t.Tensor):
                assert isinstance(right_value, t.Tensor)
                t.testing.assert_close(
                    left_value,
                    right_value,
                    atol=0.0,
                    rtol=0.0,
                )
            else:
                assert left_value == right_value


def test_producer_and_backfill_persist_raw_unaligned_maximum(tmp_path: Path) -> None:
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    base = _packed_shard()
    save_packed_row_shard(base, packed_dir / "shard_00000")
    (packed_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": "issue723-packed",
                "kind": "packed_rows",
                "metadata": {
                    "type": "packed_rows",
                    "context_len": CONTEXT_LEN,
                    "buffer_size": 1,
                    "rows_per_shard": 1,
                    "pct_checksum": "issue723-pct",
                    "tok_checksum": "issue723-tok",
                },
            },
            indent=2,
        )
        + "\n"
    )

    assert _compute_shard_max_dims(base) == (2, RAW_MAX_SEG_SIZE)
    shapes = backfill_packed_row_shapes(packed_dir)
    manifest = json.loads((packed_dir / "manifest.json").read_text())

    assert shapes == {"max_n_seg": 2, "max_seg_size": RAW_MAX_SEG_SIZE}
    assert manifest["metadata"]["max_seg_size"] == RAW_MAX_SEG_SIZE
    assert manifest["metadata"]["max_seg_size"] % 16 != 0


def test_fingerprinting_ignores_unrelated_manifest_metadata(tmp_path: Path) -> None:
    packed_dir, _ = _write_precomputed_artifacts(tmp_path)
    expected = packed_row_fingerprint(packed_dir)
    manifest_path = packed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["future_extension"] = {"version": 1}
    manifest_path.write_text(json.dumps(manifest))

    assert packed_row_fingerprint(packed_dir) == expected


def test_loader_preserves_raw_value_in_state_and_resume_identity(tmp_path: Path) -> None:
    packed_dir, selection_dir = _write_precomputed_artifacts(tmp_path)
    batch = next(_loader(packed_dir, selection_dir))
    saved_state = batch[-1]

    assert saved_state["max_seg_size"] == RAW_MAX_SEG_SIZE
    resumed = next(
        _loader(
            packed_dir,
            selection_dir,
            resume_state_dict=saved_state,
        )
    )
    assert resumed[-1]["max_seg_size"] == RAW_MAX_SEG_SIZE

    manifest_path = packed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["metadata"]["max_seg_size"] = RAW_MAX_SEG_SIZE + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"max_seg_size mismatch.*17.*18"):
        next(
            _loader(
                packed_dir,
                selection_dir,
                resume_state_dict=saved_state,
            )
        )


def test_raw_dynamic_and_aligned_eager_training_steps_are_bit_exact(
    tmp_path: Path,
) -> None:
    packed_dir, selection_dir = _write_precomputed_artifacts(tmp_path)
    batch = next(_loader(packed_dir, selection_dir))
    raw_bound = batch[-1]["max_seg_size"]
    assert raw_bound == RAW_MAX_SEG_SIZE

    reference = _model()
    dynamic_model = copy.deepcopy(reference)
    raw_model = copy.deepcopy(reference)
    aligned_model = copy.deepcopy(reference)
    dynamic_optimizer = t.optim.AdamW(dynamic_model.parameters(), lr=1e-3)
    raw_optimizer = t.optim.AdamW(raw_model.parameters(), lr=1e-3)
    aligned_optimizer = t.optim.AdamW(aligned_model.parameters(), lr=1e-3)

    losses: list[t.Tensor] = []
    grads: list[dict[str, t.Tensor]] = []
    for model, optimizer, bound in (
        (dynamic_model, dynamic_optimizer, None),
        (raw_model, raw_optimizer, raw_bound),
        (aligned_model, aligned_optimizer, ALIGNED_MAX_SEG_SIZE),
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, batch, static_max_seg_size=bound)
        loss.backward()
        losses.append(loss.detach().clone())
        grads.append(_parameter_grads(model))
        optimizer.step()

    for candidate in losses[1:]:
        t.testing.assert_close(losses[0], candidate, atol=0.0, rtol=0.0)
    assert grads[0].keys() == grads[1].keys() == grads[2].keys()
    for name in grads[0]:
        t.testing.assert_close(grads[0][name], grads[1][name], atol=0.0, rtol=0.0)
        t.testing.assert_close(grads[0][name], grads[2][name], atol=0.0, rtol=0.0)

    dynamic_params = dict(dynamic_model.named_parameters())
    raw_params = dict(raw_model.named_parameters())
    aligned_params = dict(aligned_model.named_parameters())
    assert dynamic_params.keys() == raw_params.keys() == aligned_params.keys()
    for name, dynamic_param in dynamic_params.items():
        t.testing.assert_close(
            dynamic_param,
            raw_params[name],
            atol=0.0,
            rtol=0.0,
        )
        t.testing.assert_close(
            dynamic_param,
            aligned_params[name],
            atol=0.0,
            rtol=0.0,
        )

    _assert_optimizer_states_exact(dynamic_optimizer, raw_optimizer)
    _assert_optimizer_states_exact(dynamic_optimizer, aligned_optimizer)
