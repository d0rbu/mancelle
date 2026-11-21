# tests/axbench_bpb/unit/test_runtime_state_target_device.py
"""Unit + functional tests for RuntimeState.target_device().

The helper promotes the ambiguous ``"cuda"`` device string to
``"cuda:<current_device>"`` ONLY when ``cfg.device == "cuda"`` AND
``ddp_world_size > 1``. All other inputs are returned verbatim.

Commit 1432fcc8 added the helper to fix per-rank device pinning under
torchrun + cfg.device="cuda". Pre-fix, third-party loaders silently
resolved bare "cuda" to cuda:0 on every rank, causing
"tensors on different devices" crashes on the first forward pass.

Test pyramid:
- Unit: enumerate every branch of the promotion logic
- Functional: verify decoder_only / encdec / base_gemma_and_sae all
  call target_device() (not cfg.device directly)
- Integration (CUDA-required, skipped on CPU): per-rank promotion
  under a real torchrun-emulating spawn produces distinct device
  strings per rank
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

import steering.eval.axbench_bpb.runtime_state as runtime_state_module
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import (
    RuntimeState,
    _build_cluster_pre_forward_guard,
)
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from steering.sae_resolver import L0Tier, resolve_sae


def _make_rs(*, device: str, ddp_world_size: int = 1) -> RuntimeState:
    """Build a RuntimeState with only the device + ws set; paths are
    placeholders. target_device() consults only ``cfg.device`` and
    ``self.ddp_world_size``."""
    cfg = AxBenchBpbConfig(
        source_run_dir="/tmp/src",
        output_root="/tmp/out",
        feature_dir="/tmp/feat",
        feature_statistics_path="/tmp/stats.safetensors",
        device=device,
    )
    return RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=ddp_world_size)


def _install_cluster_contract(
    state: RuntimeState, contract: object
) -> ClusterFeatureContract:
    typed_contract = cast(ClusterFeatureContract, contract)
    state.cluster_contract = typed_contract
    return typed_contract


# ---------- Unit: clustered pre-forward guard (PR #739 Track F1b) ----------


def test_clustered_pre_forward_guard_rehashes_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract: Any = object()
    metadata = {"pin": "metadata"}
    model = SimpleNamespace(_loaded_meta_data=metadata)
    calls: list[tuple[object, object, object, str]] = []

    def _assert_loaded(
        actual_contract: object,
        actual_model: object,
        actual_metadata: object,
        *,
        surface: str,
    ) -> None:
        calls.append((actual_contract, actual_model, actual_metadata, surface))

    monkeypatch.setattr(
        runtime_state_module,
        "assert_loaded_cluster_target",
        _assert_loaded,
    )
    guard = _build_cluster_pre_forward_guard(
        contract,
        model,
    )
    bundle = TargetModelBundle(
        model=model,
        tokenizer=object(),
        max_seq_len=8,
        target_name="encdec",
        cluster_contract=contract,
        cluster_pre_forward_guard=guard,
    )

    bundle.ensure_ready_for_forward()
    bundle.ensure_ready_for_forward()

    assert calls == [
        (
            contract,
            model,
            metadata,
            "axbench_bpb encdec checkpoint immediately before first forward",
        )
    ]


def test_clustered_pre_forward_guard_reads_live_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {"pin": "original"}
    replacement = {"pin": "replacement"}
    model = SimpleNamespace(_loaded_meta_data=original)
    seen: list[object] = []
    monkeypatch.setattr(
        runtime_state_module,
        "assert_loaded_cluster_target",
        lambda _contract, _model, metadata, *, surface: seen.append(metadata),
    )

    guard = _build_cluster_pre_forward_guard(
        cast(ClusterFeatureContract, object()), model
    )
    model._loaded_meta_data = replacement
    guard()

    assert seen == [replacement]


def test_clustered_pre_forward_guard_rejects_lost_live_metadata() -> None:
    model = SimpleNamespace(_loaded_meta_data={"pin": "metadata"})
    guard = _build_cluster_pre_forward_guard(
        cast(ClusterFeatureContract, object()), model
    )
    model._loaded_meta_data = None

    with pytest.raises(AxBenchBpbError, match="lost.*metadata.*before first forward"):
        guard()


def test_clustered_bundle_without_pre_forward_guard_fails_loudly() -> None:
    bundle = TargetModelBundle(
        model=object(),
        tokenizer=object(),
        max_seq_len=8,
        target_name="encdec",
        cluster_contract=cast(ClusterFeatureContract, object()),
    )

    with pytest.raises(AxBenchBpbError, match="missing.*pre-forward"):
        bundle.ensure_ready_for_forward()


def test_build_cluster_pre_forward_guard_raises_when_metadata_missing_up_front() -> None:
    """Extends the oracle's 4 cases: the guard-builder itself (not just the
    returned closure) must reject a model that never received
    ``_loaded_meta_data`` at all -- the sentinel used by
    ``RuntimeState.encdec()`` immediately after loading."""
    model = SimpleNamespace()  # no _loaded_meta_data attribute at all

    with pytest.raises(AxBenchBpbError, match="missing loaded checkpoint metadata"):
        _build_cluster_pre_forward_guard(cast(ClusterFeatureContract, object()), model)


# ---------- Unit: pure branch coverage of target_device() ----------


def test_target_device_returns_cpu_verbatim_single_process():
    """Single-process + cpu: helper is a no-op."""
    rs = _make_rs(device="cpu", ddp_world_size=1)
    assert rs.target_device() == "cpu"


def test_target_device_returns_cpu_verbatim_under_ddp():
    """DDP + cpu: helper is a no-op (cpu does not need rank pinning)."""
    rs = _make_rs(device="cpu", ddp_world_size=4)
    assert rs.target_device() == "cpu"


def test_target_device_returns_mps_verbatim():
    """mps is single-device by construction; helper is a no-op."""
    rs = _make_rs(device="mps", ddp_world_size=1)
    assert rs.target_device() == "mps"


def test_target_device_returns_explicit_cuda_pin_verbatim_single_process():
    """Explicit cuda:0 pin under single-process: returned verbatim."""
    rs = _make_rs(device="cuda:0", ddp_world_size=1)
    assert rs.target_device() == "cuda:0"


def test_target_device_returns_explicit_cuda_pin_verbatim_under_ddp():
    """Explicit cuda:3 pin under DDP: returned verbatim (operator chose
    this pin intentionally; do not override)."""
    rs = _make_rs(device="cuda:3", ddp_world_size=4)
    assert rs.target_device() == "cuda:3"


def test_target_device_returns_bare_cuda_verbatim_single_process():
    """Single-process + bare ``cuda``: helper short-circuits to the
    cfg value (no promotion needed because there's no rank ambiguity).
    """
    rs = _make_rs(device="cuda", ddp_world_size=1)
    assert rs.target_device() == "cuda"


def test_target_device_promotes_bare_cuda_under_ddp():
    """DDP + bare ``cuda``: helper promotes to ``cuda:<current_device>``.

    We patch ``torch.cuda.current_device`` so the test does not require
    CUDA. The behavior under test is purely the promotion logic.
    """
    rs = _make_rs(device="cuda", ddp_world_size=4)
    with patch.object(torch.cuda, "current_device", return_value=2):
        assert rs.target_device() == "cuda:2"


def test_target_device_promotes_with_each_local_rank():
    """Sanity: distinct local-rank values produce distinct promoted
    device strings. Catches a hypothetical hardcoded ``cuda:0``."""
    rs = _make_rs(device="cuda", ddp_world_size=8)
    seen = []
    for local_rank in (0, 1, 4, 7):
        with patch.object(torch.cuda, "current_device", return_value=local_rank):
            seen.append(rs.target_device())
    assert seen == ["cuda:0", "cuda:1", "cuda:4", "cuda:7"]


# ---------- Functional: bundle accessors thread target_device() through ----------


def test_decoder_only_loader_uses_target_device(monkeypatch):
    """``decoder_only()`` constructs its bundle by calling
    ``load_model(device=self.target_device(), ...)``. Pin the wiring
    so a refactor that re-introduces ``cfg.device`` is caught.
    """
    rs = _make_rs(device="cuda", ddp_world_size=4)

    captured: dict = {}

    def _fake_load_model(*, source, device, phase, model_tag, step, eval_conditioning=None):
        captured["device"] = device
        model = MagicMock()
        model.config = MagicMock(sequence_len=8)
        return model, MagicMock(), {}

    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.load_model", _fake_load_model,
    )
    # Stub cuda.current_device so target_device() yields cuda:2.
    with patch.object(torch.cuda, "current_device", return_value=2):
        rs.cfg.decoder_only = {
            "source": "base", "model_tag": "d13", "step": 1,
        }
        rs.decoder_only()

    assert captured["device"] == "cuda:2", (
        f"decoder_only() must pass target_device() to load_model; "
        f"got device={captured['device']!r}"
    )


def test_encdec_loader_uses_target_device(monkeypatch):
    """Same wiring guard for ``encdec()`` — the encdec bundle's model
    must land on the per-rank device, not on ``cuda:0`` by default."""
    rs = _make_rs(device="cuda", ddp_world_size=4)
    captured: dict = {}

    def _fake_load_model(*, source, device, phase, model_tag, step, eval_conditioning=None):
        captured["device"] = device
        model = MagicMock()
        model.dec_config = MagicMock(sequence_len=8)
        return model, MagicMock(), {}

    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.load_model", _fake_load_model,
    )
    with patch.object(torch.cuda, "current_device", return_value=3):
        rs.cfg.encdec = {"source": "encdec", "model_tag": "d13", "step": 1}
        rs.encdec()

    assert captured["device"] == "cuda:3"


def test_base_gemma_and_sae_loader_uses_target_device(monkeypatch):
    """The Stage 1 SAE+Gemma loader must pin to the per-rank GPU too —
    otherwise rank N's extraction tries to forward through a model
    sharded across cuda:0 + cuda:N. Dedicated identity tests live below."""
    rs = _make_rs(device="cuda", ddp_world_size=4)
    captured: dict = {}

    def _fake_parse(_):
        return "release", "act", 8, "16k", None

    def _fake_resolve(*, model, release, activation_type, layer, width, l0):
        return MagicMock()

    def _fake_load_sae(*, model_name, layer, device, resolved):
        captured["device"] = device
        return (
            MagicMock(),
            SimpleNamespace(cfg=SimpleNamespace(d_sae=64)),
            MagicMock(),
        )

    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.parse_neuronpedia_source",
        _fake_parse,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.resolve_sae", _fake_resolve,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.load_sae_and_model",
        _fake_load_sae,
    )
    with patch.object(torch.cuda, "current_device", return_value=1):
        rs.base_gemma_and_sae()

    assert captured["device"] == "cuda:1"


# ---------- Unit: clustered base_gemma_and_sae SAE-identity checks (PR #739 Track F1b) ----------


def test_unclustered_runtime_preserves_parsed_l0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unclustered and clustered runs both preserve the source-encoded tier.

    The v1 grammar cannot encode an L0, so v1 remains unaffected; see
    ``test_unclustered_v1_runtime_still_passes_none_l0_to_resolver`` for the
    byte-identity pin across every v1 source shape.
    """
    rs = _make_rs(device="cpu")
    assert rs.cluster_contract is None
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    real_l0 = L0Tier.medium
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, real_l0),
    )

    def _resolve(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(release="r", sae_id="s", hook_name="h")

    monkeypatch.setattr(runtime_state_module, "resolve_sae", _resolve)
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: (
            "model",
            SimpleNamespace(cfg=SimpleNamespace(d_sae=64)),
            "tok",
        ),
    )

    rs.base_gemma_and_sae()

    assert observed["l0"] is real_l0


@pytest.mark.parametrize(
    "source",
    [
        "8-gemmascope-res-16k",
        "20-gemmascope-res-65k",
        "12-gemmascope-att-16k",
        "5-gemmascope-mlp-65k",
    ],
)
def test_unclustered_v1_runtime_still_passes_none_l0_to_resolver(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    """Every v1 grammar shape remains on the canonical ``l0=None`` path."""
    rs = _make_rs(device="cpu")
    rs.cfg.neuronpedia_source = source
    observed_l0: list[object] = []

    def _resolve(**kwargs: Any):
        observed_l0.append(kwargs["l0"])
        return resolve_sae(**kwargs)

    monkeypatch.setattr(runtime_state_module, "resolve_sae", _resolve)
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: (
            "model",
            SimpleNamespace(cfg=SimpleNamespace(d_sae=64)),
            "tok",
        ),
    )

    rs.base_gemma_and_sae()

    assert observed_l0 == [None]


def _install_unclustered_identity_load(
    monkeypatch: pytest.MonkeyPatch,
    rs: RuntimeState,
    sae: object,
) -> tuple[SimpleNamespace, list[dict[str, object]]]:
    """Install the common unclustered post-load identity-test doubles."""
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    release = SimpleNamespace(value="gemmascope")
    resolved = SimpleNamespace(
        release="resolved-release",
        sae_id="layer_8/width_16k/canonical",
        hook_name="blocks.8.hook_resid_post",
    )
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (release, activation, 8, width, None),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_sae",
        lambda **_kwargs: resolved,
    )
    load_calls: list[dict[str, object]] = []

    def _load(**kwargs: object) -> tuple[str, object, str]:
        load_calls.append(kwargs)
        return "model", sae, "tok"

    monkeypatch.setattr(runtime_state_module, "load_sae_and_model", _load)
    assert rs.cluster_contract is None
    return resolved, load_calls


def _assert_extraction_resources_uncached(rs: RuntimeState) -> None:
    """Assert that a rejected post-load SAE did not populate any cache."""
    assert rs._base_gemma is None
    assert rs._base_sae is None
    assert rs._base_gemma_tail_offloaded_after is None


def test_unclustered_runtime_sae_rejects_wrong_tier_same_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1: a loaded SAE's reported sae_id must match the resolver output."""
    rs = _make_rs(device="cpu")
    wrong_tier_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=64,
            metadata={"sae_id": "layer_8/width_16k/average_l0_37"},
        ),
    )
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, wrong_tier_sae,
    )

    with pytest.raises(AxBenchBpbError, match="sae_id"):
        rs.base_gemma_and_sae()

    assert len(load_calls) == 1
    _assert_extraction_resources_uncached(rs)


def test_unclustered_runtime_sae_rejects_mismatched_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2: a loaded SAE's reported release must match the resolver output."""
    rs = _make_rs(device="cpu")
    wrong_release_sae = SimpleNamespace(
        cfg=SimpleNamespace(d_sae=64, release="wrong-release"),
    )
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, wrong_release_sae,
    )

    with pytest.raises(AxBenchBpbError, match="cfg.release"):
        rs.base_gemma_and_sae()

    assert len(load_calls) == 1
    _assert_extraction_resources_uncached(rs)


def test_unclustered_runtime_sae_rejects_bare_model_name_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R3: FU-D normalization still rejects genuinely different bare ids."""
    rs = _make_rs(device="cpu")
    wrong_model_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=64,
            metadata={"model_name": "google/gemma-2-9b"},
        ),
    )
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, wrong_model_sae,
    )

    with pytest.raises(AxBenchBpbError, match="model_name"):
        rs.base_gemma_and_sae()

    assert len(load_calls) == 1
    _assert_extraction_resources_uncached(rs)


def test_unclustered_runtime_sae_rejects_mismatched_neuronpedia_id_without_sae_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R4: Neuronpedia identity independently catches a substituted SAE."""
    rs = _make_rs(device="cpu")
    wrong_neuronpedia_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=64,
            metadata={
                "neuronpedia_id": "gemma-2-2b/8-gemmascope-res-16k-WRONG",
            },
        ),
    )
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, wrong_neuronpedia_sae,
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: ("gemma-2-2b", "8-gemmascope-res-16k"),
    )

    with pytest.raises(AxBenchBpbError, match="neuronpedia_id"):
        rs.base_gemma_and_sae()

    assert len(load_calls) == 1
    _assert_extraction_resources_uncached(rs)


def test_unclustered_runtime_sae_accepts_fully_populated_matching_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U1: all matching reported identity fields are accepted and cached."""
    rs = _make_rs(device="cpu")
    matching_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=64,
            release="resolved-release",
            model_name="gemma-2-2b",
            metadata={
                "model_name": "gemma-2-2b",
                "sae_id": "layer_8/width_16k/canonical",
                "neuronpedia_id": "gemma-2-2b/8-gemmascope-res-16k",
            },
        ),
    )
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, matching_sae,
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: ("gemma-2-2b", "8-gemmascope-res-16k"),
    )

    assert rs.base_gemma_and_sae() == ("model", matching_sae)
    assert len(load_calls) == 1
    assert rs._base_gemma == "model"
    assert rs._base_sae is matching_sae


def test_unclustered_runtime_sae_without_identity_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2/D7: a loader reporting no identity metadata remains tolerated."""
    rs = _make_rs(device="cpu")
    sae = object()
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, sae,
    )

    assert rs.base_gemma_and_sae() == ("model", sae)
    assert len(load_calls) == 1
    assert rs._base_gemma == "model"
    assert rs._base_sae is sae


def test_unclustered_runtime_sae_accepts_unpublished_neuronpedia_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U3: an unpublished resolver combination cannot validate this field."""
    rs = _make_rs(device="cpu")
    sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=64,
            metadata={
                "neuronpedia_id": "gemma-2-2b/8-gemmascope-res-16k",
            },
        ),
    )
    _resolved, load_calls = _install_unclustered_identity_load(
        monkeypatch, rs, sae,
    )

    def _unpublished(**_kwargs: object) -> tuple[str, str]:
        raise ValueError("not published")

    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        _unpublished,
    )

    assert rs.base_gemma_and_sae() == ("model", sae)
    assert len(load_calls) == 1
    assert rs._base_gemma == "model"
    assert rs._base_sae is sae


def test_clustered_runtime_shape_check_precedes_identity_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U4: clustered shape validation still wins when identity also fails."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    l0 = SimpleNamespace(value="big")
    resolved = SimpleNamespace(
        release="resolved-release",
        sae_id="layer_8_width_16k_l0_big",
        hook_name="blocks.8.hook_resid_post",
    )
    contract = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type=activation.value,
            layer=8,
            width=width.value,
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (SimpleNamespace(value="gemmascope_2"), activation, 8, width, l0),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_sae",
        lambda **_kwargs: resolved,
    )
    wrong_shape_and_identity_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=16,
            hook_name=resolved.hook_name,
            release="wrong-release",
        ),
        W_dec=torch.zeros((16, 3)),
    )
    load_calls: list[dict[str, object]] = []

    def _load(**kwargs: object) -> tuple[str, object, str]:
        load_calls.append(kwargs)
        return "model", wrong_shape_and_identity_sae, "tok"

    monkeypatch.setattr(runtime_state_module, "load_sae_and_model", _load)

    with pytest.raises(
        AxBenchBpbError,
        match="does not match the cluster feature contract",
    ):
        rs.base_gemma_and_sae()

    assert len(load_calls) == 1
    _assert_extraction_resources_uncached(rs)


def test_clustered_runtime_sae_preserves_l0_and_validates_exact_loaded_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rs = _make_rs(device="cpu")
    rs.cfg.neuronpedia_model_id = "gemma-3-4b-pt"
    rs.cfg.neuronpedia_source = "22-gemmascope-2-res-16k-big"
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    l0 = object()
    resolved = SimpleNamespace(
        release="release-big",
        sae_id="layer_22_width_16k_l0_big",
        hook_name="blocks.22.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model="gemma-3-4b-pt",
            activation_type="residual",
            layer=22,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.parse_neuronpedia_source",
        lambda _source: (object(), activation, 22, width, l0),
    )

    def _resolve(**kwargs):
        observed.update(kwargs)
        return resolved

    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.resolve_sae",
        _resolve,
    )
    sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=8,
            metadata=SimpleNamespace(hook_name=resolved.hook_name),
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.load_sae_and_model",
        lambda **_kwargs: ("model", sae, "tokenizer"),
    )

    assert rs.base_gemma_and_sae() == ("model", sae)
    assert observed["l0"] is l0


def test_clustered_runtime_uses_local_model_id_not_neuronpedia_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rs = _make_rs(device="cpu")
    rs.cfg.neuronpedia_model_id = "gemma-3-4b"
    rs.cfg.feature_extraction_model_id = "gemma-3-4b-pt"
    rs.cfg.neuronpedia_source = "22-gemmascope-2-res-16k-big"
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release-big",
        sae_id="layer_22_width_16k_l0_big",
        hook_name="blocks.22.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model="gemma-3-4b-pt",
            activation_type="residual",
            layer=22,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 22, width, object()),
    )

    def _resolve(**kwargs: object) -> object:
        calls["resolve_model"] = kwargs["model"]
        return resolved

    monkeypatch.setattr(runtime_state_module, "resolve_sae", _resolve)
    sae = SimpleNamespace(
        cfg=SimpleNamespace(d_sae=8, hook_name=resolved.hook_name),
        W_dec=torch.zeros((8, 3)),
    )

    def _load(**kwargs: object) -> tuple[str, object, str]:
        calls["load_model"] = kwargs["model_name"]
        return "model", sae, "tokenizer"

    monkeypatch.setattr(runtime_state_module, "load_sae_and_model", _load)

    assert rs.base_gemma_and_sae() == ("model", sae)
    assert calls == {
        "resolve_model": "gemma-3-4b-pt",
        "load_model": "gemma-3-4b-pt",
    }


def test_clustered_runtime_sae_rejects_same_shape_different_resolved_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release="release",
            sae_id="expected-sae-id",
            hook_name="blocks.8.hook_resid_post",
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.resolve_sae",
        lambda **_kwargs: SimpleNamespace(
            release="release",
            sae_id="different-sae-id",
            hook_name="blocks.8.hook_resid_post",
        ),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.runtime_state.load_sae_and_model",
        lambda **_kwargs: pytest.fail("identity mismatch must fail before loading"),
    )

    with pytest.raises(AxBenchBpbError, match="identity does not match"):
        rs.base_gemma_and_sae()


def test_clustered_runtime_sae_rejects_mismatched_hook_name_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extends test_clustered_runtime_sae_rejects_same_shape_different_
    resolved_id with a DIFFERENT mismatched field: release/sae_id agree,
    only hook_name differs. Same pre-load rejection must fire -- proving
    the identity tuple genuinely compares hook_name, not just release/
    sae_id."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release="release",
            sae_id="sae-id",
            hook_name="blocks.8.hook_resid_post",
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_sae",
        lambda **_kwargs: SimpleNamespace(
            release="release",
            sae_id="sae-id",
            hook_name="blocks.8.hook_resid_DIFFERENT",  # only this differs
        ),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: pytest.fail("hook_name mismatch must fail before loading"),
    )

    with pytest.raises(AxBenchBpbError, match="identity does not match"):
        rs.base_gemma_and_sae()


def test_clustered_runtime_sae_rejects_mismatched_d_sae_after_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pre-load identity tuple (release/sae_id/hook_name/layer/width)
    can agree while the ACTUALLY LOADED SAE object still disagrees on
    dictionary width -- e.g. a stale/corrupt SAELens cache returning the
    wrong-shaped tensors under a matching name. Proves the d_sae check
    fires AFTER load, independent of the pre-load identity-tuple check."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.8.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    # d_sae=16 != contract.n_sae_features=8 -- pre-load identity matched, so
    # this can ONLY be caught by the post-load width check.
    mismatched_sae = SimpleNamespace(
        cfg=SimpleNamespace(d_sae=16, hook_name=resolved.hook_name),
        W_dec=torch.zeros((16, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", mismatched_sae, "tok"),
    )

    with pytest.raises(AxBenchBpbError, match="does not match the cluster"):
        rs.base_gemma_and_sae()


def test_clustered_runtime_sae_rejects_mismatched_w_dec_shape_after_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same post-load-only proof as the d_sae case, but for the SEPARATE
    ``W_dec.shape[0]`` arm of the OR-chain: ``sae.cfg.d_sae`` itself
    reports the correct width while the actual decoder tensor's row count
    disagrees -- a plausible desync between a SAE object's declared config
    and its real parameter shapes."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.8.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    desynced_sae = SimpleNamespace(
        cfg=SimpleNamespace(d_sae=8, hook_name=resolved.hook_name),
        W_dec=torch.zeros((16, 3)),  # disagrees with cfg.d_sae=8
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", desynced_sae, "tok"),
    )

    with pytest.raises(AxBenchBpbError, match="does not match the cluster"):
        rs.base_gemma_and_sae()


# ---------- Codex F1b review (HIGH bug fix): loaded-object tier/identity ----------
# checks beyond shape (d_sae/hook_name/W_dec.shape[0] alone cannot distinguish
# GemmaScope-2 L0 tiers -- big/medium/small share release, layer, width,
# hook_name, and dictionary width; only sae_id/neuronpedia_id differ). Mirrors
# cluster_checkpoint_pin.py's _resolve_and_validate_sae preflight.


def test_clustered_runtime_sae_rejects_wrong_tier_same_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex's empirically-reproduced HIGH bug: the contract + resolver pin
    the GemmaScope-2 ``big`` tier, but ``load_sae_and_model`` silently
    returns a ``medium``-tier SAE double sharing the SAME release/layer/
    width/hook_name/d_sae/W_dec.shape (sae_resolver.py's
    _resolve_gemmascope_2: ``sae_id = f"layer_{layer}_width_{width.value}_
    l0_{l0_tier.value}"`` -- tiers differ ONLY in sae_id/neuronpedia_id).
    Before the fix, every shape-only check passed and the wrong-tier SAE
    was silently accepted and cached. Must now raise -- and the spy on
    load_sae_and_model confirms the raise fires strictly AFTER the load
    actually ran, not before (this is a post-load-only gap; nothing about
    the request itself was wrong)."""
    rs = _make_rs(device="cpu")
    rs.cfg.neuronpedia_model_id = "gemma-3-4b-pt"
    rs.cfg.neuronpedia_source = "22-gemmascope-2-res-16k-big"
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    l0 = SimpleNamespace(value="big")
    resolved_big = SimpleNamespace(
        release="gemma-scope-2-4b-pt-res-16k",
        sae_id="layer_22_width_16k_l0_big",
        hook_name="blocks.22.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=22,
            width="16k",
            release=resolved_big.release,
            sae_id=resolved_big.sae_id,
            hook_name=resolved_big.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 22, width, l0),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved_big)
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: ("gemma-3-4b-pt", "22-gemmascope-2-res-16k-big"),
    )
    # Same release/hook/d_sae/W_dec.shape as `big` -- only the tier
    # fingerprint in cfg.metadata contradicts it.
    medium_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release=resolved_big.release,
            hook_name=resolved_big.hook_name,
            d_sae=8,
            metadata={
                "sae_id": "layer_22_width_16k_l0_medium",
                "neuronpedia_id": "gemma-3-4b-pt/22-gemmascope-2-res-16k",
            },
        ),
        W_dec=torch.zeros((8, 3)),
    )
    load_calls: list[dict] = []

    def _fake_load(**kwargs: object) -> tuple[str, object, str]:
        load_calls.append(kwargs)
        return "model", medium_sae, "tok"

    monkeypatch.setattr(runtime_state_module, "load_sae_and_model", _fake_load)

    with pytest.raises(AxBenchBpbError, match="sae_id"):
        rs.base_gemma_and_sae()

    assert len(load_calls) == 1, "the raise must fire AFTER load_sae_and_model actually ran"
    assert rs._base_gemma is None, "the wrong-tier SAE must not be cached"


def test_clustered_runtime_sae_rejects_mismatched_release_after_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """release is checked independently of the sae_id/neuronpedia_id tier
    fields -- a loaded SAE reporting the WRONG release (but a matching
    hook/d_sae/W_dec.shape, and no sae_id/neuronpedia_id metadata to
    otherwise catch it) must still be rejected."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.8.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    wrong_release_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release="a-completely-different-release",
            hook_name=resolved.hook_name,
            d_sae=8,
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", wrong_release_sae, "tok"),
    )

    with pytest.raises(AxBenchBpbError, match="cfg.release"):
        rs.base_gemma_and_sae()


@pytest.mark.parametrize(
    ("loaded_model_name", "configured_model_name"),
    [
        pytest.param(
            "google/gemma-3-4b-pt",
            "gemma-3-4b-pt",
            id="real-artifact-org-prefixed-loaded-vs-bare-configured",
        ),
        pytest.param(
            "gemma-3-4b-pt",
            "google/gemma-3-4b-pt",
            id="bare-loaded-vs-org-prefixed-configured",
        ),
        pytest.param(
            "gemma-3-4b-pt",
            "gemma-3-4b-pt",
            id="both-bare",
        ),
    ],
)
def test_clustered_runtime_sae_accepts_bare_normalized_model_name(
    monkeypatch: pytest.MonkeyPatch,
    loaded_model_name: str,
    configured_model_name: str,
) -> None:
    """Equivalent bare model ids are accepted in every supported shape.

    The org-prefixed fixture is the literal ``model_name`` from
    ``google/gemma-scope-2-4b-pt`` at
    ``resid_post/layer_22_width_16k_l0_medium/config.json``. SAELens 6.42.0
    migrates that value into ``cfg.metadata.model_name``.
    """
    rs = _make_rs(device="cpu")
    rs.cfg.neuronpedia_model_id = configured_model_name
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.22.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=configured_model_name,
            activation_type="residual",
            layer=22,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 22, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    loaded_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release=resolved.release,
            hook_name=resolved.hook_name,
            d_sae=8,
            metadata=SimpleNamespace(model_name=loaded_model_name),
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", loaded_sae, "tok"),
    )

    assert rs.base_gemma_and_sae() == ("model", loaded_sae)


def test_clustered_runtime_sae_rejects_different_bare_normalized_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalizing org prefixes must not weaken the model identity check."""
    rs = _make_rs(device="cpu")
    rs.cfg.neuronpedia_model_id = "gemma-3-4b-pt"
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.22.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=22,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 22, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    loaded_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release=resolved.release,
            hook_name=resolved.hook_name,
            d_sae=8,
            metadata=SimpleNamespace(model_name="google/gemma-3-12b-pt"),
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", loaded_sae, "tok"),
    )

    with pytest.raises(AxBenchBpbError, match="model_name"):
        rs.base_gemma_and_sae()


def test_clustered_runtime_sae_rejects_mismatched_model_name_after_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """model_name is checked independently too -- a loaded SAE whose own
    cfg.model_name disagrees with the resolved extraction model id must be
    rejected, even though the pre-load configured/contract identity tuple
    (which compares the REQUEST, not what was actually returned) agreed."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.8.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    wrong_model_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release=resolved.release,
            hook_name=resolved.hook_name,
            d_sae=8,
            model_name="a-totally-different-model",
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", wrong_model_sae, "tok"),
    )

    with pytest.raises(AxBenchBpbError, match="model_name"):
        rs.base_gemma_and_sae()


def test_clustered_runtime_sae_rejects_mismatched_neuronpedia_id_when_sae_id_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolates the neuronpedia_id arm specifically: cfg.metadata carries
    NO sae_id key at all (so the sae_id check -- which runs first -- is a
    no-op and cannot be what catches this), only a neuronpedia_id that
    disagrees with what resolve_neuronpedia_ids computes for the resolved
    identity."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.8.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: ("gemma-2-2b", "8-gemmascope-res-16k"),
    )
    contradicting_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release=resolved.release,
            hook_name=resolved.hook_name,
            d_sae=8,
            metadata={"neuronpedia_id": "gemma-2-2b/8-gemmascope-res-16k-WRONG"},
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", contradicting_sae, "tok"),
    )

    with pytest.raises(AxBenchBpbError, match="neuronpedia_id"):
        rs.base_gemma_and_sae()


def test_clustered_runtime_sae_accepts_fully_populated_matching_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive case for all four new checks at once: release, model_name,
    sae_id, AND neuronpedia_id are all populated on the loaded SAE and all
    agree with the resolved identity -- must NOT raise (proves the new
    checks have no false positives when metadata is fully, correctly
    populated, not just absent)."""
    rs = _make_rs(device="cpu")
    activation = SimpleNamespace(value="residual")
    width = SimpleNamespace(value="16k")
    resolved = SimpleNamespace(
        release="release", sae_id="sae-id", hook_name="blocks.8.hook_resid_post",
    )
    contract: object = SimpleNamespace(
        n_sae_features=8,
        sae=SimpleNamespace(
            model=rs.cfg.neuronpedia_model_id,
            activation_type="residual",
            layer=8,
            width="16k",
            release=resolved.release,
            sae_id=resolved.sae_id,
            hook_name=resolved.hook_name,
        ),
    )
    _install_cluster_contract(rs, contract)
    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (object(), activation, 8, width, object()),
    )
    monkeypatch.setattr(runtime_state_module, "resolve_sae", lambda **_kwargs: resolved)
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: ("gemma-2-2b", "8-gemmascope-res-16k"),
    )
    matching_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release=resolved.release,
            hook_name=resolved.hook_name,
            model_name=rs.cfg.neuronpedia_model_id,
            d_sae=8,
            metadata={
                "sae_id": resolved.sae_id,
                "neuronpedia_id": "gemma-2-2b/8-gemmascope-res-16k",
            },
        ),
        W_dec=torch.zeros((8, 3)),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        lambda **_kwargs: ("model", matching_sae, "tok"),
    )

    assert rs.base_gemma_and_sae() == ("model", matching_sae)


# ---------- Integration: real CUDA path (skipped without CUDA) ----------


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires 2+ CUDA devices",
)
def test_target_device_returns_current_cuda_device_after_set_device():
    """End-to-end: after torch.cuda.set_device(1), target_device()
    returns 'cuda:1' for an axbench-bpb RuntimeState that sees DDP.
    Mirrors what compute_init does under torchrun."""
    rs = _make_rs(device="cuda", ddp_world_size=2)
    torch.cuda.set_device(1)
    assert rs.target_device() == "cuda:1"
    # Reset to 0 to avoid leaking state to the next test.
    torch.cuda.set_device(0)
