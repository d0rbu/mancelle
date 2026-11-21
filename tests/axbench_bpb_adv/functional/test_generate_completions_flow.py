"""Functional test for decoder-only completion sampling under an intervention.

Drives ``sample_generation.generate_completions_for_datapoint`` on the REAL tiny
2-layer decoder-only GPT (CPU) -- the unit tests are model-free, so this is the
place the actual ``apply_intervention`` -> ``InferenceEngine.generate_stream``
hook path runs for the single-datapoint generation primitive. The end-to-end
orchestration (``run_generate_samples_core`` reading the run dir + writing the
per-family YAMLs) is covered separately in ``test_generate_samples_flow.py``.

Two cases:

* a control (``Intervention.none()``): generation returns ``n_completions``
  strings AND the same ``base_seed`` yields byte-identical lists across two calls
  (sampling determinism), and
* a REAL diffmean intervention built via the shared
  ``intervention_build.build_steered_intervention`` on the ``mini_axbench_base_run``
  fixture (real direction safetensors): the residual hook actually fires, the call
  returns ``n_completions`` strings, and it is likewise deterministic.

The tiny model + the mini fixture are loaded LOCALLY here (no cross-root
conftest dependency, per the conftest caveat).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.intervention_build import build_steered_intervention
from steering.eval.axbench_bpb_adv.sample_generation import (
    generate_completions_for_datapoint,
)

_TINY_CKPT = (
    Path(__file__).resolve().parents[2]
    / "axbench_bpb"
    / "fixtures"
    / "tiny_decoder_only"
    / "ckpt.pt"
)
_MINI_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "axbench_bpb"
    / "fixtures"
    / "mini_axbench_base_run"
)
# Real tiny LoRA adapter fixture (concept_0001 has NON-ZERO lora_B, so it
# genuinely contributes to the forward -- unlike concept_0000's zero lora_B).
# Loaded the SAME way scoring/surrogate load it: Intervention.lora(adapter_path)
# + LoraParams(cfg, layer) threaded into apply_intervention. The cfg constants
# mirror tests/axbench_bpb/functional/test_lora_attach_detach.py (rank 4,
# alpha 8.0, layer 1 -- a valid block index for the 2-layer tiny model).
_LORA_ADAPTER = (
    Path(__file__).resolve().parents[2]
    / "axbench_bpb"
    / "fixtures"
    / "tiny_decoder_only"
    / "lora_adapter_concept_0001.pt"
)
_LORA_RANK = 4
_LORA_ALPHA = 8.0
_LORA_LAYER = 1
# diffmean_and_probe layer the mini direction files are validated against (the
# fixture writes no layer metadata, so any value passes; 8 matches the sibling
# intervention tests). Must be a valid block index for the residual hook -- the
# tiny model has 2 layers, so the hook is installed at layer 1 here.
_DIFFMEAN_LAYER = 1
_COUNTER_FAMILY = "diffmean"
_CONCEPT_ID = 1
_CORE_PREFIX_ID = "core_anchor"
_COEFFICIENT_STR = "4.0"
_PROMPT_TOKENS = [0, 5, 6, 7]


class _TinyTokenizer:
    """Minimal tokenizer for the tiny model (vocab 128, BOS = 0)."""

    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 128

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _tiny_model() -> GPT:
    state = t.load(_TINY_CKPT, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**state["config"])
    with t.device("meta"):
        model = GPT(cfg, pad_vocab_size_to=state["pad_vocab_size_to"])
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.eval()
    return model


def _concept() -> Any:
    # The builders only read concept.concept_id (to locate the per-concept
    # direction file); a minimal stand-in suffices (mirrors the sibling tests).
    return SimpleNamespace(concept_id=_CONCEPT_ID)


def _diffmean_intervention(*, magnitude: float) -> Intervention:
    """Build the SAME steered (+ optional counter) intervention the scorer scores.

    One call to ``build_steered_intervention`` (base + additive counter), seeded
    with the scorer's per-row seed so generation is byte-identical to scoring.
    """

    method = get_method_by_name("diffmean_decoder_only")
    rng = random.Random(
        hashlib.sha256(
            f"diffmean_decoder_only|{_CONCEPT_ID}|{_CORE_PREFIX_ID}|{_COEFFICIENT_STR}".encode()
        ).digest()
    )
    return build_steered_intervention(
        method=method,
        concept=_concept(),
        coefficient=_coefficient_from_str(_COEFFICIENT_STR),
        source_paths=SourcePaths(source_run_dir=_MINI_FIXTURE),
        diffmean_layer=_DIFFMEAN_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=_TinyTokenizer(),
        rng=rng,
        adversarial_method="additive_direction",
        adversarial_strength=magnitude,
        counter_direction_family=_COUNTER_FAMILY,
        target="decoder_only",
        apply_max_act_scale=False,
    )


def _generate(model: GPT, intervention: Intervention, *, base_seed: int) -> list[str]:
    return generate_completions_for_datapoint(
        model=model,
        tokenizer=_TinyTokenizer(),
        device=t.device("cpu"),
        intervention=intervention,
        prompt_tokens=list(_PROMPT_TOKENS),
        n_completions=3,
        temperature=0.8,
        top_k=None,
        max_new_tokens=6,
        base_seed=base_seed,
    )


def _generate_with(
    model: GPT,
    intervention: Intervention,
    *,
    base_seed: int,
    temperature: float,
    lora: LoraParams | None = None,
) -> list[str]:
    """``_generate`` with a caller-chosen temperature (+ optional LoRA bundle).

    The fixed-temperature ``_generate`` covers the existing sampling-determinism
    cases; this variant lets the Gap-1 test pin ``temperature=0`` (greedy /
    argmax -> seed-independent) and the Gap-2 test thread a real ``LoraParams``.
    """

    return generate_completions_for_datapoint(
        model=model,
        tokenizer=_TinyTokenizer(),
        device=t.device("cpu"),
        intervention=intervention,
        prompt_tokens=list(_PROMPT_TOKENS),
        n_completions=3,
        temperature=temperature,
        top_k=None,
        max_new_tokens=6,
        base_seed=base_seed,
        lora=lora,
    )


def test_control_generation_returns_n_completions_and_is_deterministic() -> None:
    model = _tiny_model()
    intervention = Intervention.none()

    first = _generate(model, intervention, base_seed=1234)
    second = _generate(model, intervention, base_seed=1234)

    # n_completions strings returned.
    assert len(first) == 3
    assert all(isinstance(completion, str) for completion in first)
    # Same base_seed -> byte-identical completion lists (sampling determinism).
    assert first == second
    # A different base_seed yields a different draw (the seeds genuinely drive the
    # RNG, not a constant).
    third = _generate(model, intervention, base_seed=9999)
    assert third != first


def test_diffmean_intervention_runs_hook_path_and_is_deterministic() -> None:
    model = _tiny_model()
    # magnitude 0.0 -> pure baseline diffmean steering (a real residual hook, no
    # counter); this exercises the apply_intervention diffmean hook path.
    intervention = _diffmean_intervention(magnitude=0.0)
    assert intervention.kind == "diffmean"
    assert intervention.direction is not None

    first = _generate(model, intervention, base_seed=2024)
    second = _generate(model, intervention, base_seed=2024)

    assert len(first) == 3
    assert all(isinstance(completion, str) for completion in first)
    assert first == second


def test_diffmean_with_additive_counter_runs_and_is_deterministic() -> None:
    model = _tiny_model()
    # magnitude > 0 -> the additive counter-direction hook is attached ON TOP of
    # the baseline steering; both residual hooks must install + tear down cleanly.
    intervention = _diffmean_intervention(magnitude=0.4)
    assert intervention.counter_direction is not None
    assert intervention.counter_direction_coefficient is not None

    first = _generate(model, intervention, base_seed=77)
    second = _generate(model, intervention, base_seed=77)

    assert len(first) == 3
    assert first == second


def test_steering_hook_reaches_the_sampler_logits() -> None:
    # Proof the diffmean residual hook output actually reaches the model's
    # next-token logits (and is NOT silently dropped) -- the precondition for the
    # sampled completions to reflect the steering. Asserting on the logit delta
    # rather than on sampled-token inequality keeps this non-fragile: at a tiny
    # model's modest logit perturbation, most temperature-0.8 draws still coincide
    # token-for-token, so a sampled-text inequality assertion would be seed-luck.
    # The completion-level determinism is covered by the tests above; here we pin
    # the steering's *effect on the distribution* the sampler draws from.
    model = _tiny_model()
    intervention = _diffmean_intervention(magnitude=0.0)
    idx = t.tensor([_PROMPT_TOKENS], dtype=t.long)
    with t.inference_mode():
        base_logits = model(idx)[:, -1, :].clone()
        with apply_intervention(model, intervention):
            steered_logits = model(idx)[:, -1, :].clone()
    # The steering moves the final-position logits (non-trivially) and reorders
    # the argmax -- the hook output is live, not dropped.
    assert float((base_logits - steered_logits).norm()) > 1e-4
    assert int(base_logits.argmax()) != int(steered_logits.argmax())


def test_public_generate_applies_intervention_end_to_end() -> None:
    # GAP 1: prove the PUBLIC generate path actually installs the intervention.
    # The logit test above asserts on a direct model(idx) forward, so a
    # regression that dropped the `with apply_intervention(...)` wrapper INSIDE
    # generate_completions_for_datapoint would leave it green. Here we call the
    # public entry point TWICE with the SAME prompt/seed/sampling but two
    # different interventions and require the completion lists to DIFFER.
    #
    # Mechanism: temperature=0 -> InferenceEngine.generate_stream takes the
    # greedy/argmax path (sample_next_token(..., temperature=0.0) returns
    # torch.argmax; rng stays None, so the draw is seed-independent and
    # deterministic). The sibling logit test proves the diffmean hook at
    # coefficient 4.0 flips the final-token argmax, so under greedy decoding the
    # very first generated token must already diverge -- a robust, NON-fragile
    # difference (no temperature-luck). Verified to have teeth: temporarily
    # removing the `with apply_intervention(...)` wrapper makes both lists equal,
    # turning the assert below RED.
    model = _tiny_model()
    control = Intervention.none()
    # magnitude 0.0 -> pure baseline diffmean steering at coefficient 4.0 (the
    # exact intervention whose argmax flip the logit test pins).
    steered = _diffmean_intervention(magnitude=0.0)

    control_greedy = _generate_with(model, control, base_seed=4242, temperature=0.0)
    steered_greedy = _generate_with(model, steered, base_seed=4242, temperature=0.0)

    assert len(control_greedy) == 3
    assert len(steered_greedy) == 3
    # Greedy is seed-independent: every completion in a list is identical (this
    # also confirms temperature=0 truly takes the argmax path, not sampling).
    assert len(set(control_greedy)) == 1
    assert len(set(steered_greedy)) == 1
    # The intervention reaches generation: control vs strong steering diverge.
    # (RED if generate_completions_for_datapoint skips apply_intervention.)
    assert control_greedy != steered_greedy


def test_lora_path_generates_and_is_deterministic() -> None:
    # GAP 2: exercise the lora codepath END-TO-END (real adapter, not a mock).
    # Builds a lora-kind Intervention from the real tiny adapter fixture exactly
    # the way scoring.py / surrogate.py do (Intervention.lora(adapter_path=...)
    # + LoraParams(cfg, layer) passed as `lora=` into apply_intervention) and
    # runs it through the public generate path. concept_0001 has non-zero lora_B
    # so the adapter genuinely contributes to the forward. We assert it returns
    # n_completions strings and is deterministic (same base_seed -> identical).
    # Chosen the real end-to-end path over a threading-assertion because the
    # adapter loads cleanly on the tiny model (matches the lora attach/detach
    # functional tests), giving a stronger guarantee than a monkeypatch.
    model = _tiny_model()
    lora_intervention = Intervention.lora(adapter_path=_LORA_ADAPTER)
    assert lora_intervention.kind == "lora"
    lora_params = LoraParams(
        cfg=LoraConfig(rank=_LORA_RANK, alpha=_LORA_ALPHA), layer=_LORA_LAYER
    )

    first = _generate_with(
        model, lora_intervention, base_seed=555, temperature=0.8, lora=lora_params
    )
    second = _generate_with(
        model, lora_intervention, base_seed=555, temperature=0.8, lora=lora_params
    )

    assert len(first) == 3
    assert all(isinstance(completion, str) for completion in first)
    # Same base_seed -> byte-identical completions (the lora attach is
    # deterministic and the seeds drive the sampler reproducibly).
    assert first == second


def test_steered_generation_is_seed_sensitive() -> None:
    # GAP 4: under a REAL (steered) intervention, a different base_seed yields a
    # different completion draw -- the seeds genuinely drive the RNG even while
    # the steering hook fires (the control-intervention seed-sensitivity is
    # covered above; this pins it for the steered path). Uses temperature 0.8 so
    # the sampler (not argmax) is exercised; magnitude 0.0 keeps it a single
    # diffmean steering hook at coefficient 4.0.
    model = _tiny_model()
    steered = _diffmean_intervention(magnitude=0.0)

    first = _generate_with(model, steered, base_seed=2024, temperature=0.8)
    different_seed = _generate_with(model, steered, base_seed=9999, temperature=0.8)

    assert len(first) == 3
    assert first != different_seed


def test_encdec_intervention_fails_loud() -> None:
    # Encoder-decoder sampling is a deliberate follow-up; an encdec-targeted
    # intervention must be rejected up front (decoder-only-first), never silently
    # routed through the decoder-only path.
    model = _tiny_model()
    encdec_intervention = Intervention.ours(feature_conditions={})
    assert encdec_intervention.target_model == "encdec"
    with pytest.raises(AxBenchBpbAdvError, match="encoder-decoder"):
        _generate(model, encdec_intervention, base_seed=1)
