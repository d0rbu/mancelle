"""LoRA attach/detach lifecycle tests (spec §15).

The 7 named gates that must hold for LoRA scoring. Each test exercises
``apply_intervention(model, Intervention.lora(adapter_path=...),
lora=LoraParams(cfg=..., layer=...))`` against the tiny LoRA adapter
fixture built by Task I.1.
"""
from __future__ import annotations

import gc
from pathlib import Path

import pytest
import torch

from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.types import Intervention, LoraParams

# Match the constants from regenerate_fixtures.py
TINY_LORA_LAYER = 1
TINY_LORA_RANK = 4
TINY_LORA_ALPHA = 8.0


def _adapter_path() -> Path:
    """Path to the tiny LoRA adapter fixture (Task I.1)."""
    here = Path(__file__).resolve()
    return (
        here.parent.parent / "fixtures"
        / "tiny_decoder_only" / "lora_adapter_concept_0000.pt"
    )


def _adapter_path_b() -> Path:
    """Path to the second tiny LoRA adapter fixture with non-zero
    ``lora_B`` (different params from concept_0000)."""
    here = Path(__file__).resolve()
    return (
        here.parent.parent / "fixtures"
        / "tiny_decoder_only" / "lora_adapter_concept_0001.pt"
    )


def _lora_cfg() -> LoraConfig:
    return LoraConfig(rank=TINY_LORA_RANK, alpha=TINY_LORA_ALPHA)


def _lora_params() -> LoraParams:
    return LoraParams(cfg=_lora_cfg(), layer=TINY_LORA_LAYER)


def _capture_state(model) -> dict[str, torch.Tensor]:
    """Snapshot model state_dict for byte-equal comparison."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _states_byte_equal(a: dict, b: dict) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def test_byte_exact_detach_roundtrip(tiny_decoder_only):
    """100 attach/detach cycles → state_dict byte-equal to start."""
    model, _cfg = tiny_decoder_only
    iv = Intervention.lora(adapter_path=_adapter_path())
    before = _capture_state(model)
    for _ in range(100):
        with apply_intervention(
            model, iv, lora=_lora_params(),
        ):
            pass
    after = _capture_state(model)
    assert _states_byte_equal(before, after), \
        "model state drifted after 100 attach/detach cycles"


def test_exception_during_forward_detach(tiny_decoder_only):
    """Raising inside the with-block must still tear down the adapter."""
    model, _cfg = tiny_decoder_only
    iv = Intervention.lora(adapter_path=_adapter_path())
    before = _capture_state(model)

    class _Synthetic(RuntimeError):
        pass

    with pytest.raises(_Synthetic):
        with apply_intervention(
            model, iv, lora=_lora_params(),
        ):
            raise _Synthetic("synthetic mid-forward failure")

    after = _capture_state(model)
    assert _states_byte_equal(before, after), \
        "model state drifted after exception inside with-block"


def test_concept_a_then_concept_b_interleaving(tiny_decoder_only):
    """Adapter A → forward → detach → adapter B → forward → detach.

    Uses two distinct LoRA adapters (concept_0000 has zero lora_B,
    concept_0001 has non-zero lora_B). Verifies:
      - each detach restores byte-equal state to baseline
      - the two adapters produce DIFFERENT losses (proves they really
        are different, not the same fixture loaded twice)
    """
    model, _cfg = tiny_decoder_only
    iv_a = Intervention.lora(adapter_path=_adapter_path())
    iv_b = Intervention.lora(adapter_path=_adapter_path_b())
    x = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))

    before = _capture_state(model)
    with apply_intervention(
        model, iv_a, lora=_lora_params(),
    ):
        with torch.no_grad():
            loss_a = model(x, targets=targets, loss_reduction="none")
    mid = _capture_state(model)
    with apply_intervention(
        model, iv_b, lora=_lora_params(),
    ):
        with torch.no_grad():
            loss_b = model(x, targets=targets, loss_reduction="none")
    after = _capture_state(model)

    assert _states_byte_equal(before, mid), "state drifted after first detach"
    assert _states_byte_equal(before, after), "state drifted after second detach"
    # Two distinct adapters MUST produce different losses — otherwise
    # one of the attaches silently no-op'd (cross-attach contamination
    # or an adapter that contributes 0 to the forward).
    assert not torch.equal(loss_a, loss_b), (
        "concept_0000 and concept_0001 produced byte-equal losses — "
        "either the adapters are accidentally identical or one attach "
        "silently no-op'd"
    )


def test_detach_idempotency(tiny_decoder_only, monkeypatch):
    """detach_fn is called exactly once per with-block exit."""
    model, _cfg = tiny_decoder_only
    iv = Intervention.lora(adapter_path=_adapter_path())

    detach_calls = {"count": 0}
    real_attach = intervention_module.attach_lora_from_disk

    def _counting_attach(model, adapter_path, lora_cfg, *, layer):
        adapters, real_detach = real_attach(
            model, adapter_path, lora_cfg, layer=layer,
        )
        def _counted():
            detach_calls["count"] += 1
            real_detach()
        return adapters, _counted

    monkeypatch.setattr(
        intervention_module, "attach_lora_from_disk", _counting_attach,
    )

    with apply_intervention(
        model, iv, lora=_lora_params(),
    ):
        pass

    assert detach_calls["count"] == 1, \
        f"detach_fn called {detach_calls['count']} times, expected 1"


def test_mixed_ours_lora_ours_interleaving(tiny_decoder_only, tiny_encdec):
    """ours → lora → ours sequence: ours BPB unchanged across LoRA boundary.

    The ours runs use the encdec model (Intervention.ours requires
    target_model='encdec'); the LoRA run uses decoder_only. We verify
    the ours-mode forward output is identical pre- and post-LoRA on a
    different model, since the LoRA attach mutates only the decoder_only
    bundle's parameters."""
    encdec_model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    decoder_model, _ = tiny_decoder_only
    encdec_x = torch.randint(0, 128, (1, 8))
    encdec_targets = torch.randint(0, 128, (1, 8))

    iv_ours = Intervention.ours(feature_conditions={})
    iv_lora = Intervention.lora(adapter_path=_adapter_path())

    # Run 1: ours (no LoRA)
    with apply_intervention(encdec_model, iv_ours):
        with torch.no_grad():
            loss_ours_1 = encdec_model(
                encdec_x, encdec_targets,
                enc_feature_ids=torch.zeros(1, 1, 1, dtype=torch.int64),
                enc_magnitudes=torch.zeros(1, 1, 1, dtype=torch.int64),
                enc_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
                seg_map=torch.zeros(1, encdec_x.shape[-1], dtype=torch.long),
                loss_reduction="none",
            )

    # Run 2: lora on decoder_only
    with apply_intervention(
        decoder_model, iv_lora,
        lora=_lora_params(),
    ):
        pass  # we don't compare; just exercise the attach/detach

    # Run 3: ours again — should match Run 1 byte-exactly
    with apply_intervention(encdec_model, iv_ours):
        with torch.no_grad():
            loss_ours_2 = encdec_model(
                encdec_x, encdec_targets,
                enc_feature_ids=torch.zeros(1, 1, 1, dtype=torch.int64),
                enc_magnitudes=torch.zeros(1, 1, 1, dtype=torch.int64),
                enc_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
                seg_map=torch.zeros(1, encdec_x.shape[-1], dtype=torch.long),
                loss_reduction="none",
            )

    assert torch.equal(loss_ours_1, loss_ours_2), \
        "ours-mode loss diverged after LoRA attach/detach on a different model"


def test_memory_leak_500_iterations(tiny_decoder_only):
    """500 attach/detach cycles → zero LoRAAdapter instances reachable.

    The strict signal is the LoRAAdapter-instance count after the loop:
    if any adapter survives gc.collect() it's a real leak. The
    ``object count delta < N`` heuristic from the previous version is
    non-diagnostic (might pass while leaking, might fail flakily for
    unrelated reasons) and has been removed.
    """
    model, _cfg = tiny_decoder_only
    iv = Intervention.lora(adapter_path=_adapter_path())
    iter_layer = model.transformer.h[TINY_LORA_LAYER]

    for _ in range(500):
        with apply_intervention(
            model, iv, lora=_lora_params(),
        ):
            pass

    # 1) No LoRAAdapter modules should remain attached on the layer
    for name, sub in iter_layer.named_modules():
        assert not isinstance(sub, LoRAAdapter), \
            f"residual LoRAAdapter at {name} after 500 detach cycles"

    # 2) STRICT: no LoRAAdapter instances reachable anywhere after gc.
    # This catches references held in closures, frames, or any attribute
    # we might add later — without depending on a fragile object-count
    # threshold.
    gc.collect()
    residual_loras = [obj for obj in gc.get_objects() if isinstance(obj, LoRAAdapter)]
    assert len(residual_loras) == 0, (
        f"{len(residual_loras)} LoRAAdapter instances still reachable "
        f"after 500 attach/detach cycles — memory leak"
    )


def test_counter_direction_hook_installed_and_removed(tiny_decoder_only):
    """The additive counter-direction add-on installs a residual hook on the
    decoder layer during apply_intervention and removes it on exit."""
    model, cfg = tiny_decoder_only
    layer = TINY_LORA_LAYER
    iv = Intervention.none().with_counter_direction(
        direction=torch.ones(cfg.n_embd), layer=layer, coefficient=2.0,
    )
    block = model.transformer.h[layer]
    assert len(block._forward_hooks) == 0
    with apply_intervention(model, iv):
        assert len(block._forward_hooks) == 1  # counter hook installed
    assert len(block._forward_hooks) == 0  # removed on exit


def test_counter_direction_shifts_forward_and_zero_is_noop(tiny_decoder_only):
    model, cfg = tiny_decoder_only
    layer = TINY_LORA_LAYER
    direction = torch.ones(cfg.n_embd)
    x = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))

    def _loss(iv):
        with apply_intervention(model, iv):
            with torch.no_grad():
                return model(x, targets=targets, loss_reduction="none")

    base = _loss(Intervention.none())
    counter = _loss(
        Intervention.none().with_counter_direction(
            direction=direction, layer=layer, coefficient=2.0
        )
    )
    zero = _loss(
        Intervention.none().with_counter_direction(
            direction=direction, layer=layer, coefficient=0.0
        )
    )
    assert not torch.equal(base, counter)  # the counter hook fires
    assert torch.equal(base, zero)  # magnitude 0 -> documented no-op


def test_counter_direction_composes_with_lora_and_tears_down(tiny_decoder_only):
    """LoRA adapter (module swap) AND counter hook (forward hook) coexist in one
    forward, and BOTH are torn down (state byte-equal, no residual hook)."""
    model, cfg = tiny_decoder_only
    layer = TINY_LORA_LAYER
    iv = Intervention.lora(adapter_path=_adapter_path()).with_counter_direction(
        direction=torch.ones(cfg.n_embd), layer=layer, coefficient=1.5,
    )
    before = _capture_state(model)
    block = model.transformer.h[layer]
    with apply_intervention(model, iv, lora=_lora_params()):
        assert len(block._forward_hooks) == 1  # counter hook alongside the lora swap
    after = _capture_state(model)
    assert len(block._forward_hooks) == 0
    assert _states_byte_equal(before, after)  # lora detached + hook removed


def test_additive_diffmean_two_hooks_equals_legacy_folded_coefficient(
    tiny_decoder_only,
):
    """GOLDEN equivalence: the new additive composition for a diffmean baseline
    countered along its OWN diffmean direction (steering hook at +base + counter
    hook at -magnitude) is numerically equivalent to the legacy single hook at
    (base - magnitude). Two same-direction residual hooks superpose; they agree
    up to float non-associativity (``O+3d-1.25d`` vs ``O+1.75d`` differ only by
    rounding ~1e-5 amplified through the net), NOT a steering-relevant amount."""
    model, cfg = tiny_decoder_only
    layer = TINY_LORA_LAYER
    direction = torch.randn(cfg.n_embd)
    base_coef, magnitude = 3.0, 1.25
    x = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))

    composed = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=direction, direction_layer=layer, coefficient=base_coef,
    ).with_counter_direction(direction=direction, layer=layer, coefficient=-magnitude)
    folded = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=direction, direction_layer=layer, coefficient=base_coef - magnitude,
    )

    def _loss(iv):
        with apply_intervention(model, iv):
            with torch.no_grad():
                return model(x, targets=targets, loss_reduction="none")

    assert torch.allclose(_loss(composed), _loss(folded), rtol=1e-3, atol=1e-4)


def test_additive_diffmean_counter_cancels_to_no_steering(tiny_decoder_only):
    """The cancellation invariant: a diffmean baseline at coefficient α countered
    along the SAME diffmean direction at magnitude α nets to ZERO residual delta
    -> exactly the unmodified model (`Intervention.none()`). This is what
    `control_decoder_only` at magnitude 0 reduces to, so the cancelled diffmean
    datapoint equals the decoder control's magnitude-0 point."""
    model, cfg = tiny_decoder_only
    layer = TINY_LORA_LAYER
    direction = torch.randn(cfg.n_embd)
    alpha = 2.5
    x = torch.randint(0, 128, (1, 8))
    targets = torch.randint(0, 128, (1, 8))

    cancelled = Intervention.from_direction(
        kind="diffmean", target_model="decoder_only",
        direction=direction, direction_layer=layer, coefficient=alpha,
    ).with_counter_direction(direction=direction, layer=layer, coefficient=-alpha)

    def _loss(iv):
        with apply_intervention(model, iv):
            with torch.no_grad():
                return model(x, targets=targets, loss_reduction="none")

    assert torch.allclose(
        _loss(cancelled), _loss(Intervention.none()), rtol=1e-3, atol=1e-4
    )


def test_counter_direction_on_encdec_ours(tiny_encdec):
    """The same add-on attaches a residual hook on the EncoderDecoderGPT decoder
    block while the encdec conditioning (ours) is held."""
    encdec_model, dec_cfg, _enc_cfg, _xattn_cfg = tiny_encdec
    layer = 0
    iv = Intervention.ours(feature_conditions={}).with_counter_direction(
        direction=torch.ones(dec_cfg.n_embd), layer=layer, coefficient=2.0,
    )
    block = encdec_model.transformer.h[layer]
    assert len(block._forward_hooks) == 0
    with apply_intervention(encdec_model, iv):
        assert len(block._forward_hooks) == 1  # counter hook on the encdec decoder
    assert len(block._forward_hooks) == 0


def test_concurrent_access_guard(tiny_decoder_only):
    """Double-attach without a preceding detach raises AxBenchBaseError.

    Trying to wrap a LoRAAdapter (itself an nn.Module, not an nn.Linear
    subclass) with another LoRAAdapter raises AxBenchBaseError. The outer
    apply_intervention's finally clause still fires, restoring state to
    byte-equal baseline."""
    model, _cfg = tiny_decoder_only
    iv = Intervention.lora(adapter_path=_adapter_path())
    before = _capture_state(model)

    with apply_intervention(
        model, iv, lora=_lora_params(),
    ):
        # Inner attach on an already-LoRA-adapted model raises AxBenchBaseError:
        # the sublayers are now LoRAAdapter instances, not nn.Linear.
        with pytest.raises(AxBenchBaseError):
            with apply_intervention(
                model, iv, lora=_lora_params(),
            ):
                pass
        # outer should still see attached adapters here (outer detach not yet called)

    # outer apply_intervention exited cleanly → detach_fn restored originals
    after = _capture_state(model)
    assert _states_byte_equal(before, after), \
        "nested attach/detach left state non-equal to baseline"
