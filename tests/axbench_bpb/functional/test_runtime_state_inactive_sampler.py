"""Functional tests for ``RuntimeState.inactive_sampler()`` lazy accessor.

Mirrors the axbench-base sentinel pattern: ``inactive_sampler()`` builds
the closure once on first call (consulting ``encdec_meta.negative_selection``)
and caches the result thereafter — including the ``None`` branch, so that
encdec targets without negative-feature sampling do not re-enter the
builder on every call.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch as t

from steering.eval.axbench_base import runtime_state as base_runtime_state_module
from steering.eval.axbench_base.types import AxBenchBaseError, TargetModelMetadata
from steering.eval.axbench_bpb import runtime_state as rs_module
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle

# Reuses the fixture builder from test_teacher_force_encdec_inputs.py rather
# than re-implementing FeatureClusterConsolidation.build/FeatureMask/
# FeatureVocab wiring a second time (owner lens: duplicate-helper hunt).
from tests.axbench_bpb.unit.test_teacher_force_encdec_inputs import (
    _cluster_contract,
)


@dataclass
class _FakeEncdecModel:
    """Minimal stand-in for the encdec model — the `inactive_sampler` accessor
    only forwards this opaquely to the patched ``detect_target_model_metadata``."""

    feature_vocab: object | None = None
    encoder: object | None = None

    def to(self, _device: str) -> _FakeEncdecModel:
        return self


def _make_cfg() -> AxBenchBpbConfig:
    return AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
    )


def _make_meta(*, negative_selection: dict | None) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=4 if negative_selection is not None else None,
        n_sae_features=128,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _stub_encdec_bundle(
    *,
    allowed_feature_ids: t.Tensor | None = None,
) -> TargetModelBundle:
    """Build a TargetModelBundle wrapping a fake encdec model.

    `inactive_sampler` only reads `bundle.model` (which it forwards to
    `detect_target_model_metadata`); the rest of the bundle is opaque.
    """
    model = _FakeEncdecModel()
    if allowed_feature_ids is not None:
        model.feature_vocab = object()
        model.encoder = SimpleNamespace(feature_vocab_ids=allowed_feature_ids)
    return TargetModelBundle(
        model=model,
        tokenizer=object(),
        max_seq_len=64,
        target_name="encdec",
    )


def test_inactive_sampler_is_none_when_encdec_meta_has_none(monkeypatch):
    """Encdec targets without negative_selection: inactive_sampler() → None,
    cached so subsequent calls do not re-invoke the builder."""
    cfg = _make_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    # Patch the encdec() accessor so we don't hit `nanochat.checkpoint_manager.load_model`.
    bundle = _stub_encdec_bundle()
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: bundle)

    # Patch detect_target_model_metadata at the runtime_state import site:
    # the accessor calls it once per uncached invocation of inactive_sampler().
    detect_calls: list[tuple] = []

    def _fake_detect(target_model, cfg_arg):
        detect_calls.append((target_model, cfg_arg))
        return _make_meta(negative_selection=None)

    monkeypatch.setattr(
        rs_module, "detect_target_model_metadata", _fake_detect,
    )

    # The builder must NEVER be invoked on the None-branch — patch it to record
    # call counts so we catch any accidental invocation.
    builder_calls: list[tuple] = []

    def _spy_builder(*, encdec_meta, cfg):
        builder_calls.append((encdec_meta, cfg))
        return lambda *, rng: {}

    monkeypatch.setattr(rs_module, "_build_inactive_sampler", _spy_builder)
    monkeypatch.setattr(
        rs_module,
        "feature_vocab_allowed_ids",
        lambda _model: pytest.fail(
            "negative_selection=None must not inspect checkpoint vocabulary"
        ),
    )

    # First call → None.
    sampler1 = rs.inactive_sampler()
    assert sampler1 is None
    assert len(builder_calls) == 0
    assert len(detect_calls) == 1

    # Second call: cached. detect_target_model_metadata MUST NOT run again,
    # nor must the builder.
    sampler2 = rs.inactive_sampler()
    assert sampler2 is None
    assert len(builder_calls) == 0
    assert len(detect_calls) == 1


@pytest.mark.parametrize(
    "allowed_feature_ids",
    [None, t.tensor([1, 3, 7], dtype=t.int64)],
)
def test_inactive_sampler_built_once_when_negative_selection_set(
    monkeypatch,
    allowed_feature_ids,
):
    """Encdec targets with negative_selection: inactive_sampler() builds the
    closure on first call and returns the same closure on subsequent calls."""
    cfg = _make_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)

    bundle = _stub_encdec_bundle(allowed_feature_ids=allowed_feature_ids)
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: bundle)

    detect_calls: list[tuple] = []
    negative_selection = {
        "distribution": "uniform", "min_k": 1, "max_k": 8, "mean": 4, "std": 1,
    }

    def _fake_detect(target_model, cfg_arg):
        detect_calls.append((target_model, cfg_arg))
        return _make_meta(negative_selection=negative_selection)

    monkeypatch.setattr(
        rs_module, "detect_target_model_metadata", _fake_detect,
    )

    builder_calls: list[tuple] = []
    def sentinel_closure(*, rng):
        del rng
        return {99: 3}

    def _spy_builder(*, encdec_meta, cfg, allowed_feature_ids):
        builder_calls.append((encdec_meta, cfg, allowed_feature_ids))
        return sentinel_closure

    monkeypatch.setattr(rs_module, "_build_inactive_sampler", _spy_builder)

    # First call: builder runs once.
    sampler1 = rs.inactive_sampler()
    assert sampler1 is sentinel_closure
    assert len(builder_calls) == 1
    assert builder_calls[0][0].negative_selection == negative_selection
    assert builder_calls[0][1] is cfg
    assert builder_calls[0][2] is allowed_feature_ids
    assert len(detect_calls) == 1

    # Second call: cached. Builder is not re-invoked.
    sampler2 = rs.inactive_sampler()
    assert sampler2 is sentinel_closure
    assert len(builder_calls) == 1
    assert len(detect_calls) == 1


def test_validation_failure_leaves_accessor_retryable(monkeypatch):
    cfg = _make_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    allowed = t.tensor([1, 3], dtype=t.int64)
    bundle = _stub_encdec_bundle(allowed_feature_ids=allowed)
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: bundle)
    monkeypatch.setattr(
        rs_module,
        "detect_target_model_metadata",
        lambda *_args, **_kwargs: _make_meta(
            negative_selection={
                "distribution": "geometric",
                "mean": 3.0,
            }
        ),
    )
    calls = 0
    def sentinel(*, rng):
        del rng
        return {1: 3}

    def flaky_builder(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise AxBenchBaseError("invalid checkpoint vocabulary")
        return sentinel

    monkeypatch.setattr(rs_module, "_build_inactive_sampler", flaky_builder)

    with pytest.raises(AxBenchBaseError, match="invalid checkpoint vocabulary"):
        rs.inactive_sampler()
    assert rs._inactive_sampler_loaded is False
    assert rs._inactive_sampler is None
    assert rs.inactive_sampler() is sentinel
    assert calls == 2


def test_sampler_snapshot_survives_model_offload(monkeypatch):
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        offload_between_stages=True,
    )
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    allowed = t.tensor([1, 3, 7], dtype=t.int64)
    bundle = _stub_encdec_bundle(allowed_feature_ids=allowed)
    rs._encdec = bundle
    monkeypatch.setattr(
        rs_module,
        "detect_target_model_metadata",
        lambda *_args, **_kwargs: _make_meta(
            negative_selection={
                "distribution": "normal",
                "mean": 3.0,
                "std": 0.01,
            }
        ),
    )
    monkeypatch.setattr(
        base_runtime_state_module,
        "read_n_sae_features_from_stats",
        lambda _path: 128,
    )

    sampler = rs.inactive_sampler()
    assert sampler is not None
    before = sampler(rng=random.Random(9))
    assert bundle.model.encoder is not None
    bundle.model.encoder.feature_vocab_ids = t.tensor([99], dtype=t.int64)
    rs.offload_models()
    after = sampler(rng=random.Random(9))

    assert after == before
    assert set(after).issubset({1, 3, 7})


def test_cluster_contract_none_does_not_trigger_the_stub_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward-compat sentinel: the default ``cluster_contract=None``
    must reach the ordinary (unstubbed) code path exactly as before F1b."""
    cfg = _make_cfg()
    rs = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    assert rs.cluster_contract is None
    bundle = _stub_encdec_bundle()
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: bundle)
    monkeypatch.setattr(
        rs_module,
        "detect_target_model_metadata",
        lambda *_args, **_kwargs: _make_meta(negative_selection=None),
    )

    assert rs.inactive_sampler() is None


def test_cluster_contract_reaches_representative_universe_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The un-stubbed path threads the contract and samples reps, not members."""

    cfg = _make_cfg()
    contract = _cluster_contract()
    contract.representative_universe = tuple(
        int(value)
        for value in t.nonzero(
            contract.consolidation.cluster_allowed_universe,
            as_tuple=False,
        )
        .squeeze(1)
        .tolist()
    )
    rs = RuntimeState(
        cfg=cfg,
        ddp_rank=0,
        ddp_world_size=1,
        cluster_contract=contract,
    )
    bundle = _stub_encdec_bundle(
        allowed_feature_ids=t.tensor([0, 1, 2], dtype=t.int64),
    )
    monkeypatch.setattr(RuntimeState, "encdec", lambda self: bundle)
    detect_calls: list[tuple[object, object, dict[str, object]]] = []

    def _fake_detect(
        target_model: object,
        cfg_arg: object,
        **kwargs: object,
    ) -> TargetModelMetadata:
        detect_calls.append((target_model, cfg_arg, kwargs))
        return _make_meta(
            negative_selection={
                "distribution": "uniform",
                "min_k": 1,
                "max_k": 1,
            },
        )

    def sentinel(*, rng):
        del rng
        return {0: 3}
    builder_calls: list[t.Tensor] = []

    def _spy_builder(
        *,
        encdec_meta: TargetModelMetadata,
        cfg: AxBenchBpbConfig,
        allowed_feature_ids: t.Tensor,
    ):
        assert encdec_meta.negative_selection is not None
        assert cfg is rs.cfg
        builder_calls.append(allowed_feature_ids)
        return sentinel

    monkeypatch.setattr(rs_module, "detect_target_model_metadata", _fake_detect)
    monkeypatch.setattr(rs_module, "_build_inactive_sampler", _spy_builder)
    monkeypatch.setattr(
        rs_module,
        "feature_vocab_allowed_ids",
        lambda _model: pytest.fail(
            "clustered sampling must use representative_universe"
        ),
    )

    assert rs.inactive_sampler() is sentinel
    assert detect_calls == [
        (bundle.model, cfg, {"cluster_contract": contract}),
    ]
    assert len(builder_calls) == 1
    assert builder_calls[0].dtype is t.int64
    assert builder_calls[0].tolist() == [0, 2]
