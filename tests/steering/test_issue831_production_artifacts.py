"""Issue #831 full production-body descriptor and decoder acceptance gate.

This gate uses one complete canonical c1024 packed shard (roughly 774 MB
compressed, 16,384 rows) and its real clustered splice-counterfactual
selection shard. It is intentionally excluded from ordinary CI.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch as t

import steering.remote_training as remote_module
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    parse_clustered_selection_shard_inventory,
)
from steering.packed_batch_io import (
    PackedRowShard,
    ReconstructedBatch,
    SelectionShard,
    load_packed_row_shard,
    load_packed_row_shard_bytes,
    load_selection_shard,
    load_selection_shard_bytes,
    reconstruct_batch,
)
from steering.remote_training import parse_remote_packed_inventory

pytestmark = [
    pytest.mark.very_slow,
    pytest.mark.requires_feature_data,
]

_CANONICAL_ROOT = Path("/home/d0rb/Documents/Github/nano-gpt-pretrain-steer")
_ASSET_ROOT = Path(os.environ.get("ISSUE831_ASSET_ROOT", str(_CANONICAL_ROOT)))
_PACKED_DIR = _ASSET_ROOT / (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos/"
    "splice/cut_frac0.15/seed42/pct/623c372d3f121b18/"
    "tok/dc6308e6116cbfe0/c1024/sall"
)
_SELECTION_HASH = "e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3"
_GIB = 1024**3


def _require(path: Path, purpose: str) -> Path:
    if not path.is_file():
        pytest.skip(f"missing issue-#831 real {purpose}: {path}")
    return path


def _available_ram_gib() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values["MemAvailable"] / _GIB


def _tensor_contract(value: t.Tensor) -> dict[str, Any]:
    raw = value.detach().cpu().contiguous().reshape(-1).view(t.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": tuple(value.shape),
        "stride": value.stride(),
        "storage_offset": value.storage_offset(),
        "numel": value.numel(),
        "logical_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _packed_contract(shard: PackedRowShard) -> dict[str, Any]:
    return {
        field: _tensor_contract(getattr(shard, field))
        for field in (
            "tokens",
            "row_seg_offsets",
            "seg_token_start",
            "seg_token_end",
            "seg_doc_id",
            "seg_feat_offsets",
            "feature_ids",
            "feature_pcts",
        )
    }


def _selection_contract(shard: SelectionShard) -> dict[str, Any]:
    return {
        field: _tensor_contract(getattr(shard, field))
        for field in (
            "sel_offsets",
            "sel_fids",
            "sel_buckets",
            "sel_pcts",
            "neg_offsets",
            "neg_fids",
        )
    }


def _batch_contract(batch: ReconstructedBatch) -> dict[str, Any]:
    return {field: _tensor_contract(getattr(batch, field)) for field in ReconstructedBatch._fields}


def test_complete_production_packed_and_selection_bodies_match_legacy_path_decoder() -> None:
    available_gib = _available_ram_gib()
    if available_gib < 12:
        pytest.skip(
            "issue-#831 full-body differential needs >=12 GiB available RAM; "
            f"have {available_gib:.1f} GiB"
        )
    inventory_path = _require(
        _PACKED_DIR / "remote_packed_inventory.json",
        "packed inventory",
    )
    raw_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory = parse_remote_packed_inventory(
        raw_inventory,
        surface="issue-#831 real packed inventory",
    )
    packed_entry = inventory["shards"][0]
    packed_path = _require(
        _PACKED_DIR / packed_entry["name"],
        "packed body",
    )
    assert packed_entry["byte_length"] > 700_000_000
    assert packed_path.stat().st_size == packed_entry["byte_length"]

    selection_dir = _PACKED_DIR / "sel" / _SELECTION_HASH
    selection_manifest_path = _require(
        selection_dir / "manifest.json",
        "selection manifest",
    )
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selection_inventory = parse_clustered_selection_shard_inventory(
        selection_manifest["metadata"][SELECTION_SHARD_INTEGRITY_KEY],
        surface="issue-#831 real selection inventory",
    )
    selection_by_name = {entry["name"]: entry for entry in selection_inventory["shards"]}
    selection_entry = selection_by_name[packed_entry["name"]]
    selection_path = _require(
        selection_dir / selection_entry["name"],
        "selection body",
    )
    assert selection_entry["byte_length"] > 5_000_000
    assert selection_path.stat().st_size == selection_entry["byte_length"]

    # Establish the pre-#831 path-loader oracle first, including a real
    # reconstruction through aligned packed/selection CSR data.
    path_packed = load_packed_row_shard(packed_path)
    path_selection = load_selection_shard(selection_path)
    assert path_packed.tokens.shape == (16_384, 1_025)
    assert path_packed.total_segments == packed_entry["total_segments"]
    assert path_packed.total_segments > 100_000
    assert path_packed.total_features > 10_000_000
    assert path_selection.total_segments == path_packed.total_segments
    path_packed_contract = _packed_contract(path_packed)
    path_selection_contract = _selection_contract(path_selection)
    path_batch_contract = _batch_contract(
        reconstruct_batch(
            path_packed,
            [0, 1],
            path_selection,
            effective_buckets=16,
        )
    )
    del path_packed, path_selection
    gc.collect()

    # Exercise the exact production descriptor -> immutable bytes -> decoder
    # path. The helper hashes the same bytes returned to the byte decoder.
    packed_compressed = remote_module._read_cache_body_bytes(
        packed_path,
        entry=packed_entry,
        verify_content=True,
    )
    assert len(packed_compressed) == packed_entry["byte_length"]
    assert hashlib.sha256(packed_compressed).hexdigest() == packed_entry["content_sha256"]
    bytes_packed = load_packed_row_shard_bytes(packed_compressed)
    del packed_compressed

    selection_compressed = remote_module._read_cache_body_bytes(
        selection_path,
        entry=selection_entry,
        verify_content=True,
    )
    assert len(selection_compressed) == selection_entry["byte_length"]
    assert hashlib.sha256(selection_compressed).hexdigest() == selection_entry["content_sha256"]
    bytes_selection = load_selection_shard_bytes(selection_compressed)
    del selection_compressed

    assert _packed_contract(bytes_packed) == path_packed_contract
    assert _selection_contract(bytes_selection) == path_selection_contract
    assert (
        _batch_contract(
            reconstruct_batch(
                bytes_packed,
                [0, 1],
                bytes_selection,
                effective_buckets=16,
            )
        )
        == path_batch_contract
    )
