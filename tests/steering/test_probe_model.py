"""Exactness and lifetime tests for the shared probe-model loader."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from steering.eval.probe_activations import get_probe_activations
from steering.eval.probe_model import (
    ProbeModel,
    _restore_transformerlens_probe_device,
    load_probe_model,
)
from steering.extract_features import load_sae_and_model
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
    resolve_sae,
)


class _FakeSAE(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.ones(1))
        self.cfg = SimpleNamespace(apply_b_dec_to_input=True)


class _RecordingBlock(t.nn.Linear):
    def __init__(self, index: int, events: list[str]) -> None:
        super().__init__(4, 4)
        self.index = index
        self.events = events

    def to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        self.events.append(f"block_{self.index}_to_{device}")
        return super().to(*args, **kwargs)


class _FakeProbeTransformer(t.nn.Module):
    def __init__(self, n_layers: int, events: list[str]) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(n_ctx=128, device="cpu")
        self.blocks = t.nn.ModuleList(
            [_RecordingBlock(index, events) for index in range(n_layers)]
        )
        self.unembed = t.nn.Linear(4, 4)
        self.ln_final = t.nn.LayerNorm(4)
        self.tokenizer = MagicMock()


def _resolved(layer: int) -> ResolvedSAE:
    return ResolvedSAE(
        release="test-release",
        sae_id="test-sae",
        hook_name=f"blocks.{layer}.hook_resid_post",
    )


def _trim_like_probe_model(
    model: HookedTransformer,
    sae: _FakeSAE,
    *,
    layer: int,
) -> ProbeModel:
    del model.blocks[layer + 1 :]
    del model.unembed
    del model.ln_final
    model.eval()
    return ProbeModel(
        model=model,
        sae=sae,
        tokenizer=model.tokenizer,
        hook_name=f"blocks.{layer}.hook_resid_post",
        layer=layer,
    )


def _tiny_transformer_pair() -> tuple[HookedTransformer, HookedTransformer]:
    cfg = HookedTransformerConfig(
        n_layers=3,
        d_model=16,
        n_ctx=16,
        d_head=4,
        n_heads=4,
        d_mlp=32,
        d_vocab=64,
        act_fn="relu",
        normalization_type="LN",
        device="cpu",
        seed=123,
    )
    reference = HookedTransformer(cfg)
    candidate = HookedTransformer(cfg)
    candidate.load_state_dict(reference.state_dict())
    return reference, candidate


def test_probe_activations_are_exact_after_tail_offload() -> None:
    """Early-stopped probe activations are exactly invariant to tail offload.

    SCOPE, stated honestly (cross-review finding): this runs CPU-only, so the
    offload moves an already-CPU tail to CPU. Replacing the offload helper with
    a no-op therefore still passes THIS test -- what it establishes is that the
    early-stopped activations are bit-identical across the two loader arms, not
    that migration happened. The no-op is caught by
    ``test_load_probe_model_offloads_then_deletes_tail_and_can_move`` below, whose
    ``events == ["block_2_to_cpu", "block_3_to_cpu", "load_sae"]`` assertion pins
    both that the tail moves and that it moves BEFORE the SAE load; the two tests
    are load-bearing together.

    CUDA activation exactness, real device migration, ``empty_cache()`` and any
    peak-VRAM reduction are NOT established here and are GPU-gated -- they route
    to the campaign's H3 acceptance run.
    """
    layer = 1
    reference_model, offloaded_model = _tiny_transformer_pair()
    model_loads = iter((reference_model, offloaded_model))

    with (
        patch(
            "steering.extract_features.HookedTransformer.from_pretrained",
            side_effect=lambda *_args, **_kwargs: next(model_loads),
        ),
        patch(
            "steering.extract_features.SAE.from_pretrained",
            side_effect=lambda **_kwargs: _FakeSAE(),
        ),
    ):
        model_without, sae_without, _ = load_sae_and_model(
            model_name="tiny-probe",
            layer=layer,
            device="cpu",
            resolved=_resolved(layer),
            offload_transformer_tail=False,
        )
        model_with, sae_with, _ = load_sae_and_model(
            model_name="tiny-probe",
            layer=layer,
            device="cpu",
            resolved=_resolved(layer),
            offload_transformer_tail=True,
        )

    assert isinstance(sae_without, _FakeSAE)
    assert isinstance(sae_with, _FakeSAE)
    probe_without = _trim_like_probe_model(model_without, sae_without, layer=layer)
    probe_with = _trim_like_probe_model(model_with, sae_with, layer=layer)
    assert probe_without.model is not None
    assert probe_with.model is not None
    inputs = (
        t.tensor([[1]], dtype=t.long),
        t.tensor([[1, 2, 3, 4, 5]], dtype=t.long),
        t.tensor([[5, 4, 3, 2], [6, 7, 8, 9]], dtype=t.long),
    )

    with (
        patch.object(
            probe_without.model,
            "run_with_cache",
            wraps=probe_without.model.run_with_cache,
        ) as run_without,
        patch.object(
            probe_with.model,
            "run_with_cache",
            wraps=probe_with.model.run_with_cache,
        ) as run_with,
    ):
        for tokens in inputs:
            without = get_probe_activations(
                probe_without.model,
                tokens,
                layer=probe_without.layer,
                hook_name=probe_without.hook_name,
                activation_type=probe_without.activation_type,
            )
            with_offload = get_probe_activations(
                probe_with.model,
                tokens,
                layer=probe_with.layer,
                hook_name=probe_with.hook_name,
                activation_type=probe_with.activation_type,
            )

            assert t.equal(without, with_offload)

    assert run_without.call_count == len(inputs)
    assert run_with.call_count == len(inputs)
    for activation_call in (*run_without.call_args_list, *run_with.call_args_list):
        assert activation_call.kwargs["stop_at_layer"] == layer + 1


@patch("steering.eval.probe_model.resolve_sae")
@patch("steering.extract_features.SAE.from_pretrained")
@patch("steering.extract_features.HookedTransformer.from_pretrained")
def test_load_probe_model_offloads_then_deletes_tail_and_can_move(
    mock_from_pretrained,
    mock_sae_from_pretrained,
    mock_resolve_sae,
) -> None:
    events: list[str] = []
    model = _FakeProbeTransformer(n_layers=4, events=events)
    sae = _FakeSAE()
    mock_from_pretrained.return_value = model
    mock_sae_from_pretrained.side_effect = lambda **_kwargs: (
        events.append("load_sae") or sae
    )
    mock_resolve_sae.return_value = _resolved(layer=1)

    probe = load_probe_model(
        model_name="tiny-probe",
        layer=1,
        activation_type="residual",
        sae_release="gemmascope",
        sae_width="16k",
        device="cpu",
    )

    assert events == ["block_2_to_cpu", "block_3_to_cpu", "load_sae"]
    assert probe.model is not None
    assert len(probe.model.blocks) == 2
    assert not hasattr(probe.model, "unembed")
    assert not hasattr(probe.model, "ln_final")
    assert probe.to(t.device("cpu")) is probe
    assert probe.device.type == "cpu"
    assert next(probe.sae.parameters()).device.type == "cpu"
    assert all(not parameter.requires_grad for parameter in probe.model.parameters())
    assert all(not parameter.requires_grad for parameter in probe.sae.parameters())


@patch("steering.eval.probe_model.build_nnterp_model")
@patch("steering.eval.probe_model.resolve_sae")
@patch("steering.extract_features.SAE.from_pretrained")
@patch("steering.extract_features.HookedTransformer.from_pretrained")
def test_load_probe_model_can_release_transformerlens_after_nnterp_build(
    mock_from_pretrained,
    mock_sae_from_pretrained,
    mock_resolve_sae,
    mock_build_nnterp,
) -> None:
    class _FakeNnterp(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = t.nn.Module()
            self.model.model = t.nn.Module()
            self.model.model.layers = t.nn.ModuleList(
                [t.nn.Linear(4, 4) for _ in range(4)]
            )

    model = _FakeProbeTransformer(n_layers=4, events=[])
    mock_from_pretrained.return_value = model
    mock_sae_from_pretrained.return_value = _FakeSAE()
    mock_resolve_sae.return_value = _resolved(layer=1)
    nnterp = _FakeNnterp()
    mock_build_nnterp.return_value = nnterp

    probe = load_probe_model(
        model_name="tiny-probe",
        layer=1,
        activation_type="residual",
        sae_release="gemmascope",
        sae_width="16k",
        device="cuda",
        keep_transformerlens_model=False,
    )

    assert probe.model is None
    assert probe.nnterp_model is nnterp
    assert probe.context_length == 128
    assert len(nnterp.model.model.layers) == 2
    assert probe.device.type == "cpu"
    mock_build_nnterp.assert_called_once_with(model, device="cuda")


def test_probe_model_prefers_nnterp_device_when_transformerlens_is_offloaded() -> None:
    """The activation backend and SAE must share the device used by probing."""

    class _NnterpBackend(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter(
                "weight",
                t.nn.Parameter(t.ones(1, device="meta")),
            )

    transformer_lens_model = t.nn.Linear(1, 1)
    nnterp_model = _NnterpBackend()
    probe = ProbeModel(
        model=transformer_lens_model,
        sae=_FakeSAE(),
        tokenizer=MagicMock(),
        hook_name="blocks.0.hook_resid_post",
        layer=0,
        nnterp_model=nnterp_model,
    )

    assert probe.device.type == "meta"


def test_probe_model_to_moves_only_active_nnterp_backend() -> None:
    events: list[str] = []
    transformer_lens_model = _FakeProbeTransformer(n_layers=2, events=events)
    nnterp_model = t.nn.Linear(4, 4)
    probe = ProbeModel(
        model=transformer_lens_model,
        sae=_FakeSAE(),
        tokenizer=MagicMock(),
        hook_name="blocks.0.hook_resid_post",
        layer=0,
        nnterp_model=nnterp_model,
    )

    probe.to("cpu")

    assert events == []
    assert next(nnterp_model.parameters()).device.type == "cpu"


def test_transformerlens_fallback_restores_only_executed_blocks() -> None:
    events: list[str] = []
    model = _FakeProbeTransformer(n_layers=4, events=events)

    _restore_transformerlens_probe_device(model, layer=1, device="cpu")

    assert events == ["block_0_to_cpu", "block_1_to_cpu"]
    assert model.cfg.device == "cpu"


@patch("steering.eval.probe_model.resolve_sae")
@patch("steering.extract_features.SAE.from_pretrained")
@patch("steering.extract_features.HookedTransformer.from_pretrained")
def test_load_probe_model_accepts_last_layer_with_empty_tail(
    mock_from_pretrained,
    mock_sae_from_pretrained,
    mock_resolve_sae,
) -> None:
    events: list[str] = []
    model = _FakeProbeTransformer(n_layers=3, events=events)
    mock_from_pretrained.return_value = model
    mock_sae_from_pretrained.side_effect = lambda **_kwargs: (
        events.append("load_sae") or _FakeSAE()
    )
    mock_resolve_sae.return_value = _resolved(layer=2)

    probe = load_probe_model(
        model_name="tiny-probe",
        layer=2,
        activation_type="residual",
        sae_release="gemmascope",
        sae_width="16k",
        device="cpu",
    )

    assert events == ["load_sae"]
    assert probe.model is not None
    assert len(probe.model.blocks) == 3


@patch("steering.eval.probe_model.resolve_sae")
@patch("steering.extract_features.SAE.from_pretrained")
@patch("steering.extract_features.HookedTransformer.from_pretrained")
def test_load_probe_model_rejects_model_without_blocks_before_sae_load(
    mock_from_pretrained,
    mock_sae_from_pretrained,
    mock_resolve_sae,
) -> None:
    model = t.nn.Linear(4, 4)
    mock_from_pretrained.return_value = model
    mock_resolve_sae.return_value = _resolved(layer=0)

    with pytest.raises(ValueError, match="no blocks collection"):
        load_probe_model(
            model_name="tiny-probe",
            layer=0,
            activation_type="residual",
            sae_release="gemmascope",
            sae_width="16k",
            device="cpu",
        )

    mock_sae_from_pretrained.assert_not_called()


@patch("steering.eval.probe_model.resolve_sae")
@patch("steering.extract_features.SAE.from_pretrained")
@patch("steering.extract_features.HookedTransformer.from_pretrained")
def test_load_probe_model_resolves_sae_with_bare_model_id(
    mock_from_pretrained,
    mock_sae_from_pretrained,
    mock_resolve_sae,
) -> None:
    """Org-prefixed probe ids must be normalized ONLY at the SAE-registry
    boundary; weight loading keeps the caller's id verbatim.

    The BPB-Adv roundtrip validator pins probe.model_name to
    feature_extraction_model_id, which production presets set org-prefixed
    (``google/gemma-3-4b-pt``), while the gemma-scope-2 registry is keyed on
    bare ids — without the normalization the loader raises "Unsupported
    model 'google/gemma-3-4b-pt'".
    """
    events: list[str] = []
    model = _FakeProbeTransformer(n_layers=4, events=events)
    mock_from_pretrained.return_value = model
    mock_sae_from_pretrained.return_value = _FakeSAE()
    mock_resolve_sae.return_value = _resolved(layer=1)

    load_probe_model(
        model_name="google/gemma-3-4b-pt",
        layer=1,
        activation_type="residual",
        sae_release="gemmascope_2",
        sae_width="65k",
        device="cpu",
        l0=L0Tier.medium,
    )

    assert mock_resolve_sae.call_args.kwargs["model"] == "gemma-3-4b-pt"
    assert mock_from_pretrained.call_args.args[0] == "google/gemma-3-4b-pt"


def test_gemmascope_2_registry_accepts_bare_id_and_rejects_prefixed() -> None:
    """The real resolver contract this normalization exists for: bare ids
    resolve, org-prefixed ids raise. If the registry ever learns to accept
    prefixed ids this pin flags the normalization as removable."""
    resolved = resolve_sae(
        model="gemma-3-4b-pt",
        release=SAERelease("gemmascope_2"),
        activation_type=ActivationType("residual"),
        layer=22,
        width=SAEWidth("65k"),
        l0=L0Tier("medium"),
    )
    assert "layer_22" in resolved.sae_id

    with pytest.raises(ValueError, match="Unsupported model 'google/gemma-3-4b-pt'"):
        resolve_sae(
            model="google/gemma-3-4b-pt",
            release=SAERelease("gemmascope_2"),
            activation_type=ActivationType("residual"),
            layer=22,
            width=SAEWidth("65k"),
            l0=L0Tier("medium"),
        )
