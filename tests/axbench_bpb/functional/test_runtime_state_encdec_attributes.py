"""W4 functional tests: BPB RuntimeState.encdec must:
  - read max_seq_len from model.dec_config.sequence_len (NOT model.config)
  - attach the loaded meta dict as model._loaded_meta_data so
    detect_target_model_metadata can read training-time configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch

import steering.eval.axbench_bpb.runtime_state as rs_module

# Use the same imports as test_runtime_state_offload.py to stay
# consistent with the existing fixture pattern.
from errors import FeatureClustersError
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.cluster_feature_contract import ClusterFeatureContract


@dataclass
class _DecCfg:
    sequence_len: int = 4096


class _EncdecModelLike(torch.nn.Module):
    """Mimics EncoderDecoderGPT: dec_config / enc_config but no .config."""

    def __init__(self) -> None:
        super().__init__()
        self.dec_config = _DecCfg(sequence_len=4096)
        self.enc_config = object()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))


class _DecoderModelLike(torch.nn.Module):
    """Mimics nanochat GPT: has model.config.sequence_len."""

    class _Cfg:
        sequence_len = 2048

    def __init__(self) -> None:
        super().__init__()
        self.config = _DecoderModelLike._Cfg()
        self.weight = torch.nn.Parameter(torch.randn(2, 2))


class _StubTokenizer:
    def get_vocab_size(self) -> int:
        return 65536


def _fake_load_model(*, source, device, phase, model_tag, step, eval_conditioning=None):
    meta = {"source": source, "model_tag": model_tag, "step": step}
    if source == "encdec":
        return _EncdecModelLike(), _StubTokenizer(), meta
    return _DecoderModelLike(), _StubTokenizer(), meta


def _build_min_cfg() -> AxBenchBpbConfig:
    """Build the minimum AxBenchBpbConfig using the pattern from
    test_runtime_state_offload.py."""
    return AxBenchBpbConfig(
        source_run_dir="/tmp/source_run",
        output_root="/tmp/output_root",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        device="cpu",
        decoder_only={"source": "decoder_only", "model_tag": "d12", "step": 0},
        encdec={"source": "encdec", "model_tag": "encdec", "step": 0},
    )


@pytest.fixture
def stub_load(monkeypatch):
    monkeypatch.setattr(rs_module, "load_model", _fake_load_model)


def test_encdec_runtime_state_assigns_loaded_meta_data(stub_load):
    cfg = _build_min_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    bundle = rs.encdec()
    assert hasattr(bundle.model, "_loaded_meta_data"), (
        "encdec model is missing _loaded_meta_data; downstream "
        "detect_target_model_metadata cannot read training-time config"
    )
    meta = bundle.model._loaded_meta_data
    assert meta["source"] == "encdec"


def test_encdec_runtime_state_uses_dec_config_sequence_len(stub_load):
    cfg = _build_min_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    bundle = rs.encdec()
    assert bundle.max_seq_len == 4096


def test_encdec_runtime_state_rejects_turn_checkpoint_before_caching(monkeypatch):
    def _turn_load(*, source, device, phase, model_tag, step, eval_conditioning=None):
        del source, device, phase, model_tag, step, eval_conditioning
        return (
            _EncdecModelLike(),
            _StubTokenizer(),
            {"sft_segment_mode": "turn_delimited_fixed_tokens"},
        )

    monkeypatch.setattr(rs_module, "load_model", _turn_load)
    runtime = RuntimeState(cfg=_build_min_cfg(), ddp_rank=0, ddp_world_size=1)

    with pytest.raises(
        ValueError,
        match=r"axbench_bpb does not support checkpoint .* with segment mode "
        r"'turn_delimited_fixed_tokens'",
    ):
        runtime.encdec()

    assert runtime._encdec is None
    assert runtime._encdec_metadata is None


def test_decoder_only_runtime_state_uses_config_sequence_len(stub_load):
    cfg = _build_min_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    bundle = rs.decoder_only()
    assert bundle.max_seq_len == 2048


# ---------------------------------------------------------------------------
# cluster_contract=None backward-compat regression (PR #739 Track F1b, Test
# Design's first bullet): reject_feature_clusters_metadata must still fire
# UNCONDITIONALLY at both of runtime_state.py's legacy call sites when
# checkpoint metadata carries feature_clusters descriptors -- byte-identical
# to pre-F1b main. No test previously pinned this negative case explicitly.
# ---------------------------------------------------------------------------


def _fake_load_model_with_feature_clusters(
    *, source, device, phase, model_tag, step, eval_conditioning=None,
):
    meta = {
        "source": source, "model_tag": model_tag, "step": step,
        "feature_clusters": {"schema_version": 1, "clusters": [[0, 1]]},
    }
    if source == "encdec":
        return _EncdecModelLike(), _StubTokenizer(), meta
    return _DecoderModelLike(), _StubTokenizer(), meta


def test_encdec_rejects_feature_clusters_metadata_when_cluster_contract_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rs_module, "load_model", _fake_load_model_with_feature_clusters,
    )
    cfg = _build_min_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    assert rs.cluster_contract is None

    with pytest.raises(FeatureClustersError, match="cannot consume a checkpoint"):
        rs.encdec()


def test_decoder_only_rejects_feature_clusters_metadata_unconditionally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """decoder_only() has no clustered branch at all (F1b does not touch
    this method) -- the reject must fire regardless of cluster_contract,
    since there is no accept fork to gate it."""
    monkeypatch.setattr(
        rs_module, "load_model", _fake_load_model_with_feature_clusters,
    )
    cfg = _build_min_cfg()
    rs = RuntimeState(
        cfg=cfg,
        ddp_rank=0,
        ddp_world_size=1,
        cluster_contract=cast(ClusterFeatureContract, object()),
    )

    with pytest.raises(FeatureClustersError, match="cannot consume a checkpoint"):
        rs.decoder_only()


def test_encdec_accepts_feature_clusters_metadata_when_cluster_contract_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The accept fork's other half: with a cluster_contract present,
    encdec() must route to assert_loaded_cluster_target instead of the
    unconditional reject -- proven by observing the reject path does NOT
    fire (assert_loaded_cluster_target is stubbed to a no-op spy)."""
    monkeypatch.setattr(
        rs_module, "load_model", _fake_load_model_with_feature_clusters,
    )
    calls: list[tuple] = []
    monkeypatch.setattr(
        rs_module,
        "assert_loaded_cluster_target",
        lambda contract, model, meta, *, surface: calls.append(
            (contract, model, meta, surface)
        ),
    )
    cfg = _build_min_cfg()
    contract = cast(ClusterFeatureContract, object())
    rs = RuntimeState(
        cfg=cfg,
        ddp_rank=0,
        ddp_world_size=1,
        cluster_contract=contract,
    )

    bundle = rs.encdec()  # must not raise FeatureClustersError

    assert len(calls) == 1
    assert calls[0][0] is contract
    assert calls[0][3] == "axbench_bpb encdec checkpoint"
    assert bundle.cluster_contract is contract
