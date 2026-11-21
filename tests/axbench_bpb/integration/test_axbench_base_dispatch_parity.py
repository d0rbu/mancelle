"""Parity: build_intervention(ours_*) == axbench-base Stage 4 dispatch.

Task 5.2 delivers the dispatcher scaffold; Task 13.1 wires in a real
mini-fixture for end-to-end parity. This file holds both:
  - a placeholder SKIPPED test that will be populated in Task 13.1
  - unit tests that exercise dispatch routing via monkeypatch so the
    scaffold itself is covered now.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from random import Random
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.intervention_dispatch import _dispatch_scalar_direction
from steering.eval.axbench_base.method_registry import (
    BaselineKind,
    _build_all_methods,
    get_method_by_name,
)
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention, SteeringMode
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths


@dataclass
class _FakeMethod:
    """Minimal stand-in for axbench_base.method_registry.Method."""
    name: str
    baseline: str
    mode: SteeringMode
    target_model: str


@dataclass
class _FakeConcept:
    concept_id: int


def _make_source_paths(tmp_path: Path) -> SourcePaths:
    """Build a SourcePaths rooted at tmp_path with empty dirs."""
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    return SourcePaths(source_run_dir=run_dir)


@pytest.mark.xfail(
    strict=True,
    run=False,
    reason=(
        "Task 13.1 placeholder: requires mini axbench-base fixture. Once a "
        "fixture is wired in, replace `xfail(run=False)` with the real test "
        "body. Tracked in audit findings.md slice-17 X2 §2.1 placeholder "
        "test conversion."
    ),
)
def test_ours_extract_and_add_parity_matches_axbench_base():
    """Populated in Task 13.1 when a mini fixture is wired in."""


def test_direction_intervention_numeric_parity_with_axbench_base(tmp_path: Path) -> None:
    """Numeric parity (allclose) between axbench-bpb's
    ``_build_direction_intervention`` and axbench-base's
    ``_dispatch_scalar_direction`` for a canonical method-concept-mode
    tuple (diffmean / decoder_only / NEGATIVE).

    Both code paths:
      1) load the same safetensors direction file
      2) apply the same `-abs(c)` rule for negative mode

    They must produce byte-identical (direction tensor) + scalar-equal
    (coefficient) values; otherwise the residual hook in axbench-bpb
    diverges from axbench-base and the leaderboard becomes non-comparable.
    """
    sp = _make_source_paths(tmp_path)
    target = "decoder_only"
    mode = SteeringMode.NEGATIVE
    concept_id = 7
    coefficient = 3.0

    # Write a real safetensors direction file at the expected path
    direction_truth = torch.randn(8, dtype=torch.float32)
    direction_path = sp.diffmean_direction(target, mode.value, concept_id)
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": direction_truth},
        str(direction_path),
        metadata={
            "concept_id": str(concept_id),
            "mode": mode.value,
            "target_model": target,
            "layer": "8",
        },
    )

    # axbench-bpb path (under test)
    method = _FakeMethod(
        name="neg_diffmean_decoder_only", baseline="diffmean",
        mode=mode, target_model=target,
    )
    iv = intervention_module._build_direction_intervention(
        method=method, concept=_FakeConcept(concept_id=concept_id),
        coefficient=coefficient, source_paths=sp, diffmean_layer=8,
        apply_max_act_scale=False,
    )

    # axbench-base reference: replicate _dispatch_scalar_direction's
    # coefficient-negation + safetensors load (intervention_dispatch.py:420-422).
    expected_direction = direction_truth
    expected_coefficient = -abs(float(coefficient))  # negative mode

    assert iv.direction is not None
    assert torch.allclose(iv.direction, expected_direction, atol=1e-6, rtol=0.0), (
        "axbench-bpb's _build_direction_intervention loaded a direction "
        "tensor that does not match axbench-base's _dispatch_scalar_direction"
    )
    assert iv.direction_coefficient == expected_coefficient, (
        f"coefficient mismatch: axbench-bpb={iv.direction_coefficient} "
        f"vs axbench-base={expected_coefficient}"
    )
    assert iv.direction_layer == 8


def test_scaled_direction_parity_between_bpb_and_axbench_base(tmp_path: Path) -> None:
    """One artifact yields exactly the same scaled coefficient and unit vector."""
    source_paths = _make_source_paths(tmp_path)
    concept_id = 7
    layer = 8
    coefficient = 2.0
    stored_value = 5.5
    direction_path = source_paths.diffmean_direction(
        "decoder_only", SteeringMode.POSITIVE.value, concept_id
    )
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": torch.tensor([3.0, 4.0])},
        str(direction_path),
        metadata={
            "concept_id": str(concept_id),
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": str(layer),
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(stored_value),
        },
    )
    method = get_method_by_name("diffmean_decoder_only")
    concept = _FakeConcept(concept_id=concept_id)
    bpb = intervention_module._build_direction_intervention(
        method=method,
        concept=concept,
        coefficient=coefficient,
        source_paths=source_paths,
        diffmean_layer=layer,
        apply_max_act_scale=True,
    )

    def _load_same_artifact(*_args: Any, **_kwargs: Any):
        with safe_open(str(direction_path), framework="pt", device="cpu") as handle:
            return handle.get_tensor("direction"), handle.metadata() or {}

    hook = MagicMock()
    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            side_effect=_load_same_artifact,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=hook,
        ) as hook_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=GenerationResult(
                text="ok", token_count=1, stop_reason="bos_token"
            ),
        ),
    ):
        _dispatch_scalar_direction(
            method=method,
            concept=cast(Any, concept),
            coefficient=coefficient,
            prefix_tokens=[1],
            model=MagicMock(),
            tokenizer=MagicMock(),
            device="cpu",
            target_model_name="decoder_only",
            cfg=cast(Any, SimpleNamespace(
                output_root=str(tmp_path),
                run_tag="parity",
                layer_hooks=SimpleNamespace(diffmean_and_probe_layer=layer),
                generation=SimpleNamespace(),
            )),
            context_provider=None,
            sampling_seed=0,
            apply_max_act_scale=True,
            cluster_contract=None,
        )

    assert bpb.direction_coefficient == coefficient * stored_value
    reference = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=hook_spy.call_args.kwargs["direction"],
        direction_layer=hook_spy.call_args.kwargs["layer"],
        coefficient=hook_spy.call_args.kwargs["coefficient"],
    )
    assert reference.direction_coefficient == bpb.direction_coefficient
    assert bpb.direction is not None
    assert reference.direction is not None
    assert torch.equal(reference.direction, bpb.direction)
    assert reference.kind == bpb.kind
    assert reference.target_model == bpb.target_model
    assert reference.direction_layer == bpb.direction_layer


def test_dispatch_ours_calls_ours_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline='ours' -> _build_ours_intervention is invoked."""
    called = {"ours": 0}

    def _fake(**kwargs: Any) -> Intervention:
        called["ours"] += 1
        return Intervention.ours(feature_conditions={42: 1})

    monkeypatch.setattr(intervention_module, "_build_ours_intervention", _fake)

    method = _FakeMethod(
        name="ours_extract_and_add_encdec", baseline="ours",
        mode=SteeringMode.POSITIVE, target_model="encdec",
    )
    result = build_intervention(
        method=method, concept=_FakeConcept(concept_id=1),
        coefficient=0, source_paths=_make_source_paths(tmp_path),
        diffmean_layer=6, target_model_meta=None, inactive_sampler=None,
        tokenizer_nanochat=None, rng=random.Random(0),
        apply_max_act_scale=False,
        model=SimpleNamespace(feature_vocab=None),
    )

    assert called["ours"] == 1
    assert result.kind == "ours"
    assert result.feature_conditions == {42: 1}


def test_dispatch_diffmean_calls_direction_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline='diffmean' -> _build_direction_intervention, baseline propagates."""
    called: dict[str, Any] = {"direction": 0, "captured_baseline": None}

    def _fake(**kwargs: Any) -> Intervention:
        called["direction"] += 1
        called["captured_baseline"] = kwargs["method"].baseline
        return Intervention.from_direction(
            kind="diffmean", target_model="decoder_only",
            direction=torch.zeros(4), direction_layer=6, coefficient=1.0,
        )

    monkeypatch.setattr(intervention_module, "_build_direction_intervention", _fake)

    method = _FakeMethod(
        name="diffmean_decoder_only", baseline="diffmean",
        mode=SteeringMode.POSITIVE, target_model="decoder_only",
    )
    result = build_intervention(
        method=method, concept=_FakeConcept(concept_id=2),
        coefficient=1.0, source_paths=_make_source_paths(tmp_path),
        diffmean_layer=6, target_model_meta=None, inactive_sampler=None,
        tokenizer_nanochat=None, rng=random.Random(0),
        apply_max_act_scale=False,
    )

    assert called["direction"] == 1
    assert called["captured_baseline"] == "diffmean"
    assert result.kind == "diffmean"


def test_dispatch_linear_probe_calls_direction_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline='linear_probe' -> same _build_direction_intervention helper."""
    called = {"direction": 0}

    def _fake(**kwargs: Any) -> Intervention:
        called["direction"] += 1
        return Intervention.from_direction(
            kind="linear_probe", target_model="encdec",
            direction=torch.zeros(4), direction_layer=12, coefficient=2.0,
        )

    monkeypatch.setattr(intervention_module, "_build_direction_intervention", _fake)

    method = _FakeMethod(
        name="linear_probe_encdec", baseline="linear_probe",
        mode=SteeringMode.POSITIVE, target_model="encdec",
    )
    result = build_intervention(
        method=method, concept=_FakeConcept(concept_id=3),
        coefficient=2.0, source_paths=_make_source_paths(tmp_path),
        diffmean_layer=12, target_model_meta=None, inactive_sampler=None,
        tokenizer_nanochat=None, rng=random.Random(0),
        apply_max_act_scale=False,
    )

    assert called["direction"] == 1
    assert result.kind == "linear_probe"


def test_dispatch_lora_calls_lora_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline='lora' -> _build_lora_intervention."""
    called = {"lora": 0}
    fake_path = tmp_path / "adapter.safetensors"

    def _fake(**kwargs: Any) -> Intervention:
        called["lora"] += 1
        return Intervention.lora(adapter_path=fake_path)

    monkeypatch.setattr(intervention_module, "_build_lora_intervention", _fake)

    method = _FakeMethod(
        name="lora_decoder_only", baseline="lora",
        mode=SteeringMode.POSITIVE, target_model="decoder_only",
    )
    result = build_intervention(
        method=method, concept=_FakeConcept(concept_id=4),
        coefficient=None, source_paths=_make_source_paths(tmp_path),
        diffmean_layer=6, target_model_meta=None, inactive_sampler=None,
        tokenizer_nanochat=None, rng=random.Random(0),
        apply_max_act_scale=False,
    )

    assert called["lora"] == 1
    assert result.kind == "lora"


def test_dispatch_prompting_calls_prompting_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """baseline='prompting' -> _build_prompting_intervention with tokenizer."""
    called: dict[str, Any] = {"prompting": 0, "seen_tokenizer": None}

    def _fake(**kwargs: Any) -> Intervention:
        called["prompting"] += 1
        called["seen_tokenizer"] = kwargs["tokenizer"]
        return Intervention.prompting(
            target_model=kwargs["target_model"], prompt_prefix_tokens=[1, 2, 3],
        )

    monkeypatch.setattr(intervention_module, "_build_prompting_intervention", _fake)

    sentinel_tok = object()
    method = _FakeMethod(
        name="prompting_encdec", baseline="prompting",
        mode=SteeringMode.POSITIVE, target_model="encdec",
    )
    result = build_intervention(
        method=method, concept=_FakeConcept(concept_id=5),
        coefficient=None, source_paths=_make_source_paths(tmp_path),
        diffmean_layer=6, target_model_meta=None, inactive_sampler=None,
        tokenizer_nanochat=sentinel_tok, rng=random.Random(0),
        apply_max_act_scale=False,
    )

    assert called["prompting"] == 1
    assert called["seen_tokenizer"] is sentinel_tok
    assert result.kind == "prompting"


def test_dispatch_unknown_baseline_raises(tmp_path: Path) -> None:
    """Unknown baseline string surfaces a loud AxBenchBpbError."""
    method = _FakeMethod(
        name="futurebaseline_x", baseline="futurebaseline",
        mode=SteeringMode.POSITIVE, target_model="decoder_only",
    )
    with pytest.raises(AxBenchBpbError, match="Unknown baseline"):
        build_intervention(
            method=method, concept=_FakeConcept(concept_id=6),
            coefficient=None, source_paths=_make_source_paths(tmp_path),
            diffmean_layer=6, target_model_meta=None, inactive_sampler=None,
            tokenizer_nanochat=None, rng=random.Random(0),
            apply_max_act_scale=False,
        )


def test_direction_helper_missing_file_raises(tmp_path: Path) -> None:
    """Direct test of _build_direction_intervention when the file is missing."""
    method = _FakeMethod(
        name="diffmean_encdec", baseline="diffmean",
        mode=SteeringMode.POSITIVE, target_model="encdec",
    )
    with pytest.raises(AxBenchBpbError, match="diffmean direction file missing"):
        intervention_module._build_direction_intervention(
            method=method, concept=_FakeConcept(concept_id=7),
            coefficient=1.0, source_paths=_make_source_paths(tmp_path),
            diffmean_layer=6,
            apply_max_act_scale=False,
        )


def test_direction_helper_linear_probe_missing_file_raises(tmp_path: Path) -> None:
    """Parallel of test_direction_helper_missing_file_raises for the
    linear_probe branch — distinct path (linear_probe_direction) and error
    string (`linear_probe direction file missing`)."""
    method = _FakeMethod(
        name="linear_probe_encdec", baseline="linear_probe",
        mode=SteeringMode.POSITIVE, target_model="encdec",
    )
    with pytest.raises(AxBenchBpbError, match="linear_probe direction file missing"):
        intervention_module._build_direction_intervention(
            method=method, concept=_FakeConcept(concept_id=7),
            coefficient=1.0, source_paths=_make_source_paths(tmp_path),
            diffmean_layer=6,
            apply_max_act_scale=False,
        )


def test_lora_helper_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="lora adapter file missing"):
        intervention_module._build_lora_intervention(
            concept=_FakeConcept(concept_id=8), mode="positive",
            source_paths=_make_source_paths(tmp_path),
        )


def test_prompting_helper_missing_descriptions_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="prompting descriptions file missing"):
        intervention_module._build_prompting_intervention(
            concept=_FakeConcept(concept_id=9), mode="positive",
            source_paths=_make_source_paths(tmp_path),
            target_model="encdec", tokenizer=None,
        )


def test_prompting_helper_missing_concept_description_raises(tmp_path: Path) -> None:
    sp = _make_source_paths(tmp_path)
    desc_path = sp.prompting_descriptions("positive")
    desc_path.parent.mkdir(parents=True, exist_ok=True)
    desc_path.write_text(json.dumps({"42": "some prompt"}))

    class _FakeTok:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

    with pytest.raises(AxBenchBpbError, match="no prompting description for concept"):
        intervention_module._build_prompting_intervention(
            concept=_FakeConcept(concept_id=9), mode="positive",
            source_paths=sp, target_model="encdec", tokenizer=_FakeTok(),
        )


def test_prompting_helper_applies_template_render(tmp_path: Path) -> None:
    """The template wrapper (e.g., 'Related to {desc}:\\n\\n') must be
    rendered BEFORE tokenization — parity with axbench-base's Stage 4
    dispatch (`_prompting_prefix_text`)."""
    sp = _make_source_paths(tmp_path)
    desc_path = sp.prompting_descriptions("positive")
    desc_path.parent.mkdir(parents=True, exist_ok=True)
    desc_path.write_text(json.dumps({"99": "cats"}))

    captured: dict[str, str] = {}

    class _CapturingTok:
        def encode(self, text: str) -> list[int]:
            captured["text"] = text
            return [1, 2, 3]

    intervention_module._build_prompting_intervention(
        concept=_FakeConcept(concept_id=99), mode="positive",
        source_paths=sp, target_model="encdec", tokenizer=_CapturingTok(),
    )
    # Raw description stored in JSON was "cats"; the template must wrap it.
    assert captured["text"] == "Related to cats:\n\n"


def test_prompting_helper_negative_mode_template(tmp_path: Path) -> None:
    """Negative mode uses 'Not related to' prefix, not 'Related to'."""
    sp = _make_source_paths(tmp_path)
    desc_path = sp.prompting_descriptions("negative")
    desc_path.parent.mkdir(parents=True, exist_ok=True)
    desc_path.write_text(json.dumps({"7": "dogs"}))

    captured: dict[str, str] = {}

    class _CapturingTok:
        def encode(self, text: str) -> list[int]:
            captured["text"] = text
            return [1, 2, 3]

    intervention_module._build_prompting_intervention(
        concept=_FakeConcept(concept_id=7), mode="negative",
        source_paths=sp, target_model="decoder_only", tokenizer=_CapturingTok(),
    )
    assert captured["text"] == "Not related to dogs:\n\n"


# -----------------------------------------------------------------------------
# Task I.15: 27-method shape-correct dispatch parity (mini fixture).
# -----------------------------------------------------------------------------


def test_parity_all_methods(tmp_path: Path) -> None:
    """For each method x concept_id 0 in the mini fixture, build_intervention
    must produce a structurally-correct Intervention payload. Verifies:

    - ours methods -> kind="ours", target_model="encdec", feature_conditions
      populated (possibly empty)
    - diffmean / linear_probe -> direction loaded, direction_layer set,
      direction_coefficient is the supplied float
    - lora -> adapter_path exists on disk
    - prompting -> prompt_prefix_tokens populated

    This is a SHAPE check; full byte-exact parity vs axbench-base's own
    Stage 4 dispatch is deferred (would require running both pipelines
    on the same source -- see deferred-agent-prompt §2 for that test).
    """
    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures" / "mini_axbench_base_run"
    )
    sp = SourcePaths(source_run_dir=fixture)

    # Bag-backed methods require adv_paths to locate
    # their precomputed conditioning bags. Task-10 made a MISSING bag raise (the
    # score-stage prefix filter guarantees only prefixes-with-bags reach the
    # builder), so we WRITE a real bag for each bag-backed family here at the
    # (concept_id=0, mode, reg_lambda=0.0, family) key the parity loop resolves.
    # Those methods then load a real bag and satisfy the "ours -> kind='ours',
    # feature_conditions populated" shape assertion below, keeping them IN the
    # parity sweep. (The grid for a GCG method with cfg=None resolves to [0.0];
    # feature-DiffMean also uses reg_lambda=0.0; roofline passes no
    # core_prefix_id => prefix_id=None.) Non-bag-backed methods ignore adv_paths.
    adv_paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="parity")

    def _write_precomputed_bag(path: Path, feature_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "feature_ids": torch.tensor([feature_id], dtype=torch.int64),
                "buckets": torch.tensor([0], dtype=torch.int64),
            },
            str(path),
        )

    for _family, _mode, _fid in (
        ("fair", SteeringMode.POSITIVE, 11),
        ("fair", SteeringMode.NEGATIVE, 12),
        ("roofline", SteeringMode.POSITIVE, 13),
        ("roofline", SteeringMode.NEGATIVE, 14),
        ("feature_diffmean", SteeringMode.POSITIVE, 15),
        ("feature_diffmean", SteeringMode.NEGATIVE, 16),
        ("feature_diffmean_roofline", SteeringMode.POSITIVE, 17),
        (
            "feature_diffmean_roofline",
            SteeringMode.NEGATIVE,
            18,
        ),
    ):
        _write_precomputed_bag(
            adv_paths.ours_gcg_bag(0, _mode, 0.0, _family), _fid
        )

    # Minimal target_model_meta stand-ins (matching what build_intervention reads).
    # _build_ours_intervention reads target_model_meta to build inactive_sampler;
    # for this test, num_buckets=4 matches the fixture's encoder config.
    @dataclass(frozen=True)
    class _Meta:
        negative_selection: Any = None
        num_buckets: int | None = 4

    target_metas: dict[str, Any] = {
        "decoder_only": _Meta(num_buckets=None),
        "encdec": _Meta(num_buckets=4),
    }

    class _DummyTokenizer:
        def encode(self, text: str) -> list[int]:
            return [ord(c) % 128 for c in text]

    methods = _build_all_methods()
    assert len(methods) >= 20, f"got {len(methods)} methods, expected ~27"

    rng = Random(0)
    for method in methods:
        # Pick a representative coefficient -- first from grid, or None
        try:
            coef_grid = list(method.coefficient_grid_fn(
                target_metas[method.target_model],
                None,
            ))
        except Exception:
            coef_grid = [None]
        coef = coef_grid[0] if coef_grid else None

        try:
            iv = build_intervention(
                method=method,
                concept=_FakeConcept(concept_id=0),
                coefficient=coef,
                source_paths=sp,
                diffmean_layer=1,
                target_model_meta=target_metas[method.target_model],
                inactive_sampler=None,
                tokenizer_nanochat=_DummyTokenizer(),
                rng=rng,
                apply_max_act_scale=False,
                adv_paths=adv_paths,
                # Legacy (uncompacted) model stand-in: this is a general SHAPE
                # parity check across every method, not a V-restriction test --
                # a legacy checkpoint's back-compat rule (no vocab_fingerprint in
                # an unstamped bag's meta -> loads) keeps every bag-backed method
                # in the sweep exactly as before checkpoint-V stamping existed.
                model=SimpleNamespace(feature_vocab=None),
            )
        except Exception as exc:
            raise AssertionError(
                f"build_intervention failed for {method.name}: {exc}"
            ) from exc

        if method.baseline == BaselineKind.OURS:
            assert iv.kind == "ours", f"{method.name}: expected kind='ours', got {iv.kind!r}"
            assert iv.target_model == "encdec", (
                f"{method.name}: expected target_model='encdec', got {iv.target_model!r}"
            )
            assert iv.feature_conditions is not None, (
                f"{method.name}: feature_conditions is None"
            )
        elif method.baseline == BaselineKind.DIFFMEAN:
            assert iv.kind == "diffmean", (
                f"{method.name}: expected kind='diffmean', got {iv.kind!r}"
            )
            assert iv.direction is not None, f"{method.name}: direction is None"
            assert iv.direction_layer == 1, (
                f"{method.name}: expected direction_layer=1, got {iv.direction_layer}"
            )
        elif method.baseline == BaselineKind.LINEAR_PROBE:
            assert iv.kind == "linear_probe", (
                f"{method.name}: expected kind='linear_probe', got {iv.kind!r}"
            )
            assert iv.direction is not None, f"{method.name}: direction is None"
        elif method.baseline == BaselineKind.LORA:
            assert iv.kind == "lora", (
                f"{method.name}: expected kind='lora', got {iv.kind!r}"
            )
            assert iv.target_model == "decoder_only", (
                f"{method.name}: expected target_model='decoder_only', got {iv.target_model!r}"
            )
            assert iv.lora_adapter_path is not None, (
                f"{method.name}: lora_adapter_path is None"
            )
            assert iv.lora_adapter_path.exists(), (
                f"{method.name}: LoRA path doesn't exist: {iv.lora_adapter_path}"
            )
        elif method.baseline == BaselineKind.PROMPTING:
            assert iv.kind == "prompting", (
                f"{method.name}: expected kind='prompting', got {iv.kind!r}"
            )
            assert iv.prompt_prefix_tokens is not None, (
                f"{method.name}: prompt_prefix_tokens is None"
            )
            assert len(iv.prompt_prefix_tokens) > 0, (
                f"{method.name}: prompt_prefix_tokens is empty"
            )
        elif method.baseline == BaselineKind.CONTROL:
            # No steering: decoder-only -> kind NONE (no hook); encdec -> kind
            # OURS with EMPTY concept conditioning (prompt-slot conditioning only).
            if method.target_model == "encdec":
                assert iv.kind == "ours", (
                    f"{method.name}: expected kind='ours', got {iv.kind!r}"
                )
                assert iv.feature_conditions == {}, (
                    f"{method.name}: control encdec must have empty feature_conditions"
                )
            else:
                assert iv.kind == "none", (
                    f"{method.name}: expected kind='none', got {iv.kind!r}"
                )
                assert iv.target_model == "decoder_only", (
                    f"{method.name}: expected target_model='decoder_only', got {iv.target_model!r}"
                )
        else:
            pytest.fail(f"unexpected baseline kind: {method.baseline}")
