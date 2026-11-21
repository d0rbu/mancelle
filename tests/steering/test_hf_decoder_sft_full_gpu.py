"""Opt-in full-model, full-mixture, one-H200 decoder-SFT acceptance gate.

Run only after both immutable model snapshots and every standard-mixture
dataset are present in the offline caches. The gate intentionally interrupts
after one committed update and restores for a second. The output root must be a
new, explicit scratch path with enough space for one portable model and two FP32
model/optimizer restart checkpoints::

    CUDA_VISIBLE_DEVICES=0 \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    RUN_HF_DECODER_SFT_FULL_GPU=1 \
    HF_DECODER_SFT_DATA_MANIFEST=/shared/data/standard-sft-manifest.json \
    HF_DECODER_SFT_FULL_OUTPUT_ROOT=/shared/results/hf-sft-one-h200-<run-id> \
    uv run pytest -m very_slow \
      tests/steering/test_hf_decoder_sft_full_gpu.py -q -s

There is deliberately no tiny-model, synthetic-data, non-H200, online-download,
or temporary-filesystem fallback.  Once explicitly enabled, a missing
production prerequisite is a test failure.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors import safe_open

import scripts.train.chat_sft as chat_sft
from steering.sft.hf_contract import HF_SFT_FORMAT_VERSION

pytestmark = [
    pytest.mark.very_slow,
    pytest.mark.requires_feature_data,
    pytest.mark.skipif(
        os.environ.get("RUN_HF_DECODER_SFT_FULL_GPU") != "1",
        reason=(
            "set RUN_HF_DECODER_SFT_FULL_GPU=1 for the full one-H200 "
            "decoder-SFT acceptance gate"
        ),
    ),
]


_SOURCES = {
    "pythia": {
        "config_name": "chat_sft_pythia_2_8b",
        "model_id": "EleutherAI/pythia-2.8b",
        "revision": "2a259cdd96a4beb1cdf467512e3904197345f6a9",
        "architecture": "GPTNeoXForCausalLM",
        "parameter_count": 2_775_208_960,
        "tied_embeddings": False,
    },
    "smollm3": {
        "config_name": "chat_sft_smollm3_stage1_step_40000",
        "model_id": "HuggingFaceTB/SmolLM3-3B-checkpoints",
        "revision": "524d818535ee98e56a4f0d621c84dd5c8fb8d3d4",
        "architecture": "SmolLM3ForCausalLM",
        "parameter_count": 3_075_098_624,
        "tied_embeddings": True,
    },
}


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    assert value is not None and value.strip(), f"{name} must be a nonempty path"
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized_parameter_count(checkpoint: Path) -> int:
    count = 0
    names: set[str] = set()
    model_files = sorted(checkpoint.glob("model*.safetensors"))
    assert model_files, f"portable export contains no safetensors: {checkpoint}"
    for path in model_files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                assert name not in names, f"duplicate serialized parameter {name}"
                names.add(name)
                shape = handle.get_slice(name).get_shape()
                count += math.prod(shape)
    return count


def _assert_one_h200_offline() -> None:
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert t.cuda.is_available(), "the full decoder-SFT gate requires CUDA"
    assert t.cuda.device_count() == 1, (
        "run this gate with exactly one visible H200; distributed topologies use "
        "the separate torchrun acceptance launcher"
    )
    device_name = t.cuda.get_device_name(0)
    assert "H200" in device_name, f"expected an H200, got {device_name!r}"
    assert t.cuda.get_device_properties(0).total_memory >= 135 * 1024**3


@pytest.mark.parametrize("family", ["pythia", "smollm3"])
def test_full_model_real_standard_mixture_update_restart_export_and_profile(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupt after update one, restore, finish update two, and retain evidence."""
    _assert_one_h200_offline()
    source = _SOURCES[family]
    manifest = Path(_required_environment("HF_DECODER_SFT_DATA_MANIFEST")).resolve()
    assert manifest.is_file(), f"missing immutable data manifest: {manifest}"
    manifest_payload = json.loads(manifest.read_text())
    assert isinstance(manifest_payload, dict | list) and manifest_payload, (
        "the immutable data manifest must be a nonempty JSON object or array"
    )
    manifest_sha256 = _sha256_file(manifest)
    base_output = Path(
        _required_environment("HF_DECODER_SFT_FULL_OUTPUT_ROOT")
    ).resolve()
    output = base_output / family
    profile_output = output / "profile"
    assert not output.exists(), f"acceptance output must be new: {output}"
    output.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output.parent).free
    assert free_bytes >= 100 * 1024**3, (
        f"full {family} gate requires at least 100 GiB free before starting; "
        f"found {free_bytes / 1024**3:.1f} GiB at {output.parent}"
    )

    common_args = [
        f"config_name={source['config_name']}",
        "run=dummy",
        "num_iterations=2",
        "device_batch_size=1",
        "max_seq_len=1024",
        "total_batch_size=1024",
        "eval_every=0",
        "save_every=0",
        "dry_run=false",
        "profile_timing=true",
        "profile_trace_steps=[1]",
        "profile_trace_ranks=[0]",
        f"profile_output_dir={profile_output}",
        "huggingface.attention_implementation=sdpa",
        "huggingface.distributed_implementation=ddp_fused_adamw",
        "huggingface.torch_compile=false",
        "huggingface.restart_save_every=1",
        f"huggingface.data_snapshot_sha256={manifest_sha256}",
        f"huggingface.output_dir={output}",
        "huggingface.validate_export=true",
    ]

    class _IntentionalInterruption(RuntimeError):
        pass

    real_restart_save = chat_sft.save_hf_restart_distributed

    def save_then_interrupt(**kwargs: Any) -> Path:
        marker = real_restart_save(**kwargs)
        if kwargs["completed_step"] == 1:
            raise _IntentionalInterruption("restart step one is durably complete")
        return marker

    monkeypatch.setattr(chat_sft, "save_hf_restart_distributed", save_then_interrupt)
    with pytest.raises(_IntentionalInterruption, match="durably complete"):
        chat_sft.main([*common_args, "huggingface.restart_from_step=-1"])
    step_one_marker = output / "restart" / "complete_000001.json"
    assert step_one_marker.is_file() and step_one_marker.stat().st_size > 0

    # Release the interrupted invocation's full-size CUDA object graph before
    # exercising the production restore path in the same pytest process.
    monkeypatch.setattr(
        chat_sft,
        "save_hf_restart_distributed",
        real_restart_save,
    )
    gc.collect()
    t.cuda.empty_cache()
    chat_sft.main([*common_args, "huggingface.restart_from_step=1"])

    portable = output / "step_000002"
    restart = output / "restart"
    metadata = json.loads((portable / "sft_metadata.json").read_text())
    marker = json.loads((restart / "complete_000002.json").read_text())
    restart_meta = json.loads((restart / "meta_000002.json").read_text())
    first_marker = json.loads(step_one_marker.read_text())

    assert metadata["format_version"] == HF_SFT_FORMAT_VERSION
    assert metadata["completed_optimizer_step"] == 2
    assert metadata["source"] == {
        "model_id": source["model_id"],
        "revision": source["revision"],
        "architecture": source["architecture"],
        "model_type": "gpt_neox" if family == "pythia" else "smollm3",
    }
    assert metadata["output"]["architecture"] == source["architecture"]
    assert metadata["batch"] == {
        "world_size": 1,
        "device_batch_size": 1,
        "max_seq_len": 1024,
        "total_batch_size": 1024,
        "gradient_accumulation_steps": 1,
    }
    assert metadata["optimizer"]["name"] == "AdamW(fused=True)"
    assert metadata["optimizer"]["distributed_implementation"] == "ddp_fused_adamw"
    assert metadata["attention_implementation"] == "sdpa"
    assert metadata["tied_embeddings"] is source["tied_embeddings"]
    assert metadata["mixture_fingerprints"]["train"]["len_mixture"] > 500_000
    assert metadata["mixture_fingerprints"]["validation"]["len_mixture"] > 5_000
    assert (
        metadata["user_config"]["huggingface"]["data_snapshot_sha256"]
        == manifest_sha256
    )
    assert _serialized_parameter_count(portable) == source["parameter_count"]

    assert marker["completed_optimizer_step"] == 2
    assert marker["world_size"] == marker["local_world_size"] == 1
    assert marker["distributed_implementation"] == "ddp_fused_adamw"
    assert marker["semantic_identity_sha256"] == restart_meta[
        "semantic_identity_sha256"
    ]
    assert first_marker["semantic_identity_sha256"] == marker[
        "semantic_identity_sha256"
    ]
    expected_restart_files = {
        "model_000002.pt",
        "meta_000002.json",
        "optim_000002_rank0.pt",
    }
    assert set(marker["files"]) == expected_restart_files
    for name, size in marker["files"].items():
        path = restart / name
        assert path.is_file()
        assert path.stat().st_size == size > 0
    assert (restart / "model_000002.pt").stat().st_size > 10 * 1024**3
    assert (restart / "optim_000002_rank0.pt").stat().st_size > 10 * 1024**3

    timing_path = profile_output / "chat_sft_timing_rank0.csv"
    trace_path = profile_output / "chat_sft_trace_rank0_step1.json"
    assert timing_path.is_file() and timing_path.stat().st_size > 0
    assert trace_path.is_file() and trace_path.stat().st_size > 0
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_names = {
        event.get("name")
        for event in trace_payload.get("traceEvents", [])
        if isinstance(event, dict)
    }
    assert {
        "chat_sft/whole_step",
        "chat_sft/batch_construction",
        "chat_sft/forward_backward",
        "chat_sft/optimizer_step",
    } <= trace_names
    with timing_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name = {row["name"]: row for row in rows}
    assert set(by_name) == {
        "chat_sft/batch_construction",
        "chat_sft/forward_backward",
        "chat_sft/optimizer_step",
        "chat_sft/paid_step",
        "chat_sft/whole_step",
    }
    for row in rows:
        assert int(row["elapsed_ns"]) > 0
        assert int(row["rank"]) == 0
        assert int(row["step"]) == 2
