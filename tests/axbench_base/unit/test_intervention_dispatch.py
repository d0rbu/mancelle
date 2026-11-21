"""Unit tests for Stage 4 ``_run_intervention`` dispatch (Task 37).

Covers :func:`run_intervention` in
``steering.eval.axbench_base.intervention_dispatch``, which dispatches a
per-method intervention (load artifact, install hooks / adapters /
prompting, build encoder input, generate, clean up) and returns a
:class:`GenerationResult`.

Every test mocks at the boundary of the dispatch's cooperators — the
safetensors loader, the residual hook helper, the LoRA attach helper,
the encoder-input builder, and the two generation helpers — so the
tests exercise dispatch logic, not the cooperators themselves.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch as t
import torch.nn as nn
from safetensors.torch import save_file

from errors import FeatureClustersError
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.artifact_metadata import (
    EXTRACTION_TOKENIZER_METADATA_KEY,
)
from steering.eval.axbench_base.coefficient_grids import (
    diffmean_scalar_grid,
    single_condition_grid,
)
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
    LayerHooksConfig,
    LoraConfig,
)
from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    DirectionScaleRequest,
    encode_max_activation,
    preflight_direction_scales,
)
from steering.eval.axbench_base.encoder_input import PartitionModifier
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.intervention_dispatch import (
    _dispatch_lora,
    _dispatch_prompting,
    _load_scalar_direction_artifact,
    _load_top_k_concept_fids,
    _sampling_seed,
    run_intervention,
)
from steering.eval.axbench_base.method_registry import BaselineKind, Method
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
    TargetBundle,
    TargetModelMetadata,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.steering_stats import write_steering_stats

# ---------------------------------------------------------------------------
# Stubs and fixtures
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Small tokenizer stub with deterministic encode/decode semantics.

    ``encode(text)`` returns ``[ord(c) % 256 for c in text]``.
    ``decode(ids)`` is the inverse shape: ``"tok:<id>,<id>,..."``.
    """

    def __init__(self) -> None:
        self.encode_calls: list[dict[str, Any]] = []

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        self.encode_calls.append({"text": text, "prepend": prepend})
        out = [ord(c) % 256 for c in text]
        if prepend is not None:
            out.insert(0, prepend)
        return out

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        return "tok:" + ",".join(str(i) for i in ids)


class _FakeModel(nn.Module):
    """Tiny ``nn.Module`` used as the ``target_model`` argument.

    Has a single parameter so ``next(model.parameters()).device`` works
    (the real residual hook helper uses this to move the direction to
    the model device — we bypass it via patching in most tests, but the
    sentinel-state test inspects the model directly).
    """

    def __init__(self) -> None:
        super().__init__()
        self.probe_param = nn.Parameter(t.zeros(1))


def _make_cfg(
    *,
    output_root: str | None = None,
    feature_statistics_path: str = "stats.safetensors",
) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir="out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        feature_statistics_path=feature_statistics_path,
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=output_root,
        run_tag="test-run",
        generation=GenerationConfig(
            max_new_tokens=5,
            temperature=0.7,
            top_k_sampling=50,
            master_seed=42,
        ),
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=3),
        lora=LoraConfig(),
    )


def _make_meta(
    *,
    num_buckets: int | None = 8,
    negative_selection: dict | None = None,
) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=num_buckets,
        n_sae_features=16384,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


class _EmptyContextProvider:
    def get(self, token_ids: Sequence[int]) -> tuple[()]:
        del token_ids
        return ()


def _target_bundle(
    name: str,
    model: nn.Module,
    tokenizer: _StubTokenizer,
) -> TargetBundle:
    return TargetBundle(
        name=name,
        model=model,
        meta=_make_meta(),
        tokenizer=tokenizer,
        context_provider=_EmptyContextProvider() if name == "encdec" else None,
    )


def _make_concept(concept_id: int = 7) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=1234,
        layer=8,
        sae_source="gemma-2-2b/16k",
        description="a concept about wizards",
        corpus_count=100,
        description_source="test",
    )


def _method_diffmean(
    target: str = "decoder_only",
    mode: SteeringMode = SteeringMode.POSITIVE,
) -> Method:
    return Method(
        name=f"{'neg_' if mode == SteeringMode.NEGATIVE else ''}diffmean_{target}",
        mode=mode,
        baseline=BaselineKind.DIFFMEAN,
        target_model=target,
        requires_negative_sampling=False,
        coefficient_grid_fn=diffmean_scalar_grid,
        partition_modifier_name=None,
    )


def _method_linear_probe(
    target: str = "decoder_only",
    mode: SteeringMode = SteeringMode.POSITIVE,
) -> Method:
    return Method(
        name=(f"{'neg_' if mode == SteeringMode.NEGATIVE else ''}linear_probe_{target}"),
        mode=mode,
        baseline=BaselineKind.LINEAR_PROBE,
        target_model=target,
        requires_negative_sampling=False,
        coefficient_grid_fn=diffmean_scalar_grid,
        partition_modifier_name=None,
    )


def _method_lora(mode: SteeringMode = SteeringMode.POSITIVE) -> Method:
    return Method(
        name=f"{'neg_' if mode == SteeringMode.NEGATIVE else ''}lora_decoder_only",
        mode=mode,
        baseline=BaselineKind.LORA,
        target_model="decoder_only",
        requires_negative_sampling=False,
        coefficient_grid_fn=single_condition_grid,
        partition_modifier_name=None,
    )


def _method_prompting(
    target: str = "decoder_only",
    mode: SteeringMode = SteeringMode.POSITIVE,
) -> Method:
    return Method(
        name=f"{'neg_' if mode == SteeringMode.NEGATIVE else ''}prompting_{target}",
        mode=mode,
        baseline=BaselineKind.PROMPTING,
        target_model=target,
        requires_negative_sampling=False,
        coefficient_grid_fn=single_condition_grid,
        partition_modifier_name=None,
    )


def _method_ours(
    suffix: str = "extract_and_add",
    mode: SteeringMode = SteeringMode.POSITIVE,
    modifier_name: PartitionModifier | None = None,
) -> Method:
    if modifier_name is None:
        modifier_name = PartitionModifier(
            f"{'neg_' if mode == SteeringMode.NEGATIVE else ''}ours_{suffix}"
            if mode == SteeringMode.NEGATIVE
            else f"ours_{suffix}"
        )
    return Method(
        name=(f"{'neg_' if mode == SteeringMode.NEGATIVE else ''}ours_{suffix}_encdec"),
        mode=mode,
        baseline=BaselineKind.OURS,
        target_model="encdec",
        requires_negative_sampling=False,
        coefficient_grid_fn=single_condition_grid,
        partition_modifier_name=modifier_name,
    )


@pytest.fixture
def fake_model() -> _FakeModel:
    return _FakeModel()


@pytest.fixture
def stub_tokenizer() -> _StubTokenizer:
    return _StubTokenizer()


def _clustered_control_bundle(
    model: nn.Module,
    tokenizer: _StubTokenizer,
    *,
    guard: Callable[[], None] | None,
) -> TargetBundle:
    return TargetBundle(
        name="encdec",
        model=model,
        meta=_make_meta(),
        tokenizer=tokenizer,
        context_provider=_EmptyContextProvider(),
        cluster_contract=cast(Any, object()),
        cluster_pre_forward_guard=guard,
    )


def _control_encdec_method() -> Method:
    return Method(
        name="control_encdec",
        mode=SteeringMode.POSITIVE,
        baseline=BaselineKind.CONTROL,
        target_model="encdec",
        requires_negative_sampling=False,
        coefficient_grid_fn=single_condition_grid,
        partition_modifier_name=None,
    )


def test_clustered_encdec_guard_runs_immediately_before_first_forward(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    events: list[str] = []
    expected = GenerationResult(text="ok", token_count=1, stop_reason="bos_token")

    def _generate(**_kwargs: Any) -> GenerationResult:
        events.append("forward")
        return expected

    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        side_effect=_generate,
    ):
        result = run_intervention(
            method=_control_encdec_method(),
            concept=_make_concept(),
            eval_seed=0,
            coefficient=None,
            prefix_tokens=[1],
            target=_clustered_control_bundle(
                fake_model,
                stub_tokenizer,
                guard=lambda: events.append("guard"),
            ),
            cfg=_make_cfg(),
            concept_features_path=Path("unused"),
            apply_max_act_scale=False,
        )

    assert result == expected
    assert events == ["guard", "forward"]


def test_run_intervention_requires_scale_keyword(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    with pytest.raises(TypeError, match="apply_max_act_scale"):
        cast(Any, run_intervention)(
            method=_control_encdec_method(),
            concept=_make_concept(),
            eval_seed=0,
            coefficient=None,
            prefix_tokens=[1],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=_make_cfg(),
            concept_features_path=Path("unused"),
        )


def test_clustered_scalar_guard_is_after_setup_and_immediately_before_forward(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    events: list[str] = []
    expected = GenerationResult(text="ok", token_count=1, stop_reason="bos_token")
    handle = SimpleNamespace(remove=lambda: events.append("remove_hook"))
    target = TargetBundle(
        name="encdec",
        model=fake_model,
        meta=_make_meta(),
        tokenizer=stub_tokenizer,
        context_provider=_EmptyContextProvider(),
        cluster_contract=cast(Any, object()),
        cluster_pre_forward_guard=lambda: events.append("guard"),
    )

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            side_effect=lambda *_args, **_kwargs: events.append("load_direction")
            or (t.ones(3), {}),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            side_effect=lambda **_kwargs: events.append("install_hook") or handle,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            side_effect=lambda **_kwargs: events.append("forward") or expected,
        ),
    ):
        result = run_intervention(
            method=_method_diffmean(target="encdec", mode=SteeringMode.POSITIVE),
            concept=_make_concept(),
            eval_seed=0,
            coefficient=1.0,
            prefix_tokens=[1],
            target=target,
            cfg=_make_cfg(),
            concept_features_path=Path("unused"),
            apply_max_act_scale=False,
            cluster_source_corpus_sha256="a" * 64,
        )

    assert result == expected
    assert events == [
        "load_direction",
        "install_hook",
        "guard",
        "forward",
        "remove_hook",
    ]


def test_clustered_ours_guard_follows_bag_assertion_and_precedes_forward(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    events: list[str] = []
    expected = GenerationResult(text="ok", token_count=1, stop_reason="bos_token")

    class _RecordingConsolidation:
        def assert_representatives_only(self, ids: t.Tensor, *, surface: str) -> None:
            assert ids.tolist() == [5]
            assert surface.endswith("encoder input")
            events.append("assert_representatives")

    target = TargetBundle(
        name="encdec",
        model=fake_model,
        meta=_make_meta(),
        tokenizer=stub_tokenizer,
        context_provider=_EmptyContextProvider(),
        cluster_contract=cast(
            Any,
            SimpleNamespace(consolidation=_RecordingConsolidation()),
        ),
        cluster_pre_forward_guard=lambda: events.append("guard"),
    )

    def _load_bag(*_args: Any, **kwargs: Any) -> list[int]:
        assert kwargs["expected_representative_id"] == 1234
        events.append("load_bag")
        return [5]

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_top_k_concept_fids",
            side_effect=_load_bag,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.build_mode_aware_encoder_input",
            side_effect=lambda **_kwargs: events.append("build_encoder_input") or {5: 1},
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            side_effect=lambda **_kwargs: events.append("forward") or expected,
        ),
    ):
        result = run_intervention(
            method=_method_ours(
                suffix="extract_and_add",
                mode=SteeringMode.POSITIVE,
            ),
            concept=_make_concept(),
            eval_seed=0,
            coefficient=1,
            prefix_tokens=[1],
            target=target,
            cfg=_make_cfg(),
            concept_features_path=Path("unused"),
            apply_max_act_scale=False,
        )

    assert result == expected
    assert events == [
        "load_bag",
        "build_encoder_input",
        "assert_representatives",
        "guard",
        "forward",
    ]


def test_clustered_encdec_missing_guard_fails_before_forward(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        side_effect=AssertionError("forward must not start"),
    ) as generate:
        with pytest.raises(AxBenchBaseError, match="missing.*pre-forward"):
            run_intervention(
                method=_control_encdec_method(),
                concept=_make_concept(),
                eval_seed=0,
                coefficient=None,
                prefix_tokens=[1],
                target=_clustered_control_bundle(
                    fake_model,
                    stub_tokenizer,
                    guard=None,
                ),
                cfg=_make_cfg(),
                concept_features_path=Path("unused"),
                apply_max_act_scale=False,
            )
    generate.assert_not_called()


def test_clustered_decoder_only_bundle_skips_encdec_pre_forward_guard(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    guard_calls: list[str] = []
    target = TargetBundle(
        name="decoder_only",
        model=fake_model,
        meta=_make_meta(),
        tokenizer=stub_tokenizer,
        context_provider=None,
        cluster_contract=cast(Any, object()),
        cluster_pre_forward_guard=lambda: guard_calls.append("guard"),
    )

    target.ensure_ready_for_forward()

    assert guard_calls == []


def test_unclustered_dispatch_does_not_enter_cluster_guard_path(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = GenerationResult(text="ok", token_count=1, stop_reason="bos_token")
    target = TargetBundle(
        name="encdec",
        model=fake_model,
        meta=_make_meta(),
        tokenizer=stub_tokenizer,
        context_provider=_EmptyContextProvider(),
    )

    def _unexpected_guard(_self: TargetBundle) -> None:
        raise AssertionError("legacy dispatch must not call the cluster guard")

    monkeypatch.setattr(TargetBundle, "ensure_ready_for_forward", _unexpected_guard)
    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        return_value=expected,
    ):
        result = run_intervention(
            method=_control_encdec_method(),
            concept=_make_concept(),
            eval_seed=0,
            coefficient=None,
            prefix_tokens=[1],
            target=target,
            cfg=_make_cfg(),
            concept_features_path=Path("unused"),
            apply_max_act_scale=False,
        )

    assert result == expected


# ---------------------------------------------------------------------------
# 1. DiffMean decoder-only installs hook + removes on exit
# ---------------------------------------------------------------------------


def test_diffmean_decoder_only_installs_hook(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_diffmean(target="decoder_only", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    direction = t.tensor([1.0, 2.0, 3.0])
    expected = GenerationResult(text="hello", token_count=4, stop_reason="bos_token")
    handle = MagicMock()

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(direction, {}),
        ) as load_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=handle,
        ) as hook_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ) as gen_spy,
    ):
        result = run_intervention(
            method=method,
            concept=concept,
            eval_seed=2,
            coefficient=0.5,
            prefix_tokens=[10, 20],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    assert result == expected
    assert load_spy.call_count == 1
    assert hook_spy.call_count == 1
    hook_kwargs = hook_spy.call_args.kwargs
    # Called with the loaded direction.
    assert t.equal(hook_kwargs["direction"], direction)
    # Coefficient forwarded as positive float for positive mode.
    assert hook_kwargs["coefficient"] == pytest.approx(0.5)
    # Layer comes from cfg.
    assert hook_kwargs["layer"] == cfg.layer_hooks.diffmean_and_probe_layer
    # Generation helper called exactly once.
    assert gen_spy.call_count == 1
    # Handle removed exactly once on clean path.
    assert handle.remove.call_count == 1


# ---------------------------------------------------------------------------
# 3. DiffMean negative mode negates coefficient
# ---------------------------------------------------------------------------


def test_diffmean_negative_mode_negates_coefficient(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_diffmean(target="decoder_only", mode=SteeringMode.NEGATIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    direction = t.tensor([1.0, 2.0, 3.0])
    handle = MagicMock()
    expected = GenerationResult(text="x", token_count=1, stop_reason="bos_token")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(direction, {}),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=handle,
        ) as hook_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=1,
            coefficient=1.5,
            prefix_tokens=[1],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    # Negative mode → coefficient is negated (subtract direction).
    assert hook_spy.call_args.kwargs["coefficient"] == pytest.approx(-1.5)
    assert handle.remove.call_count == 1


# ---------------------------------------------------------------------------
# 3b. Per-direction max-activation scaling (#1228 Seam A)
# ---------------------------------------------------------------------------


def _direction_metadata(
    *, mode: SteeringMode, value: float = 7.0, **overrides: str
) -> dict[str, str]:
    metadata = {
        "concept_id": "7",
        "mode": mode.value,
        "target_model": "decoder_only",
        "layer": "3",
        MAX_ACTIVATION_METADATA_KEY: encode_max_activation(value),
    }
    metadata.update(overrides)
    return metadata


def _run_diffmean_capturing_coefficient(
    *,
    mode: SteeringMode,
    coefficient: float,
    apply_max_act_scale: bool,
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    metadata: dict[str, str] | None = None,
) -> tuple[float, t.Tensor]:
    method = _method_diffmean(target="decoder_only", mode=mode)
    cfg = _make_cfg()
    concept = _make_concept()  # feature_index == 1234
    direction = t.tensor([1.0, 2.0, 3.0])
    expected = GenerationResult(text="x", token_count=1, stop_reason="bos_token")
    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(
                direction,
                _direction_metadata(mode=mode) if metadata is None else metadata,
            ),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=MagicMock(),
        ) as hook_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=2,
            coefficient=coefficient,
            prefix_tokens=[10, 20],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=apply_max_act_scale,
        )
    kwargs = hook_spy.call_args.kwargs
    return kwargs["coefficient"], kwargs["direction"]


def test_diffmean_positive_scales_coefficient_by_artifact_value(
    fake_model: _FakeModel, stub_tokenizer: _StubTokenizer
) -> None:
    coef, direction = _run_diffmean_capturing_coefficient(
        mode=SteeringMode.POSITIVE,
        coefficient=0.5,
        apply_max_act_scale=True,
        fake_model=fake_model,
        stub_tokenizer=stub_tokenizer,
    )
    assert coef == 0.5 * 7.0
    assert t.allclose(direction, t.tensor([1.0, 2.0, 3.0]) / t.sqrt(t.tensor(14.0)))


def test_sae_stats_cannot_influence_stage4_dispatch_scaling(
    tmp_path: Path,
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    stats_path = tmp_path / "feature_statistics.safetensors"
    artifact_value = 7.0
    sae_value = 97.0
    cfg = _make_cfg(
        output_root=str(tmp_path / "out"),
        feature_statistics_path=str(stats_path),
    )
    method = _method_diffmean()
    concept = _make_concept()
    artifact_path = paths.AxBenchBasePaths.from_config(cfg).diffmean_direction(
        "decoder_only", method.mode, concept.concept_id
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.tensor([1.0, 0.0, 0.0])},
        str(artifact_path),
        metadata=_direction_metadata(mode=method.mode, value=artifact_value),
    )
    write_steering_stats(
        stats_path,
        n_sae_features=2048,
        max_mags={concept.feature_index: sae_value},
    )
    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=MagicMock(),
        ) as hook_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=GenerationResult(
                text="unused", token_count=0, stop_reason="bos_token"
            ),
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=2.0,
            prefix_tokens=[1],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=True,
        )
    applied = hook_spy.call_args.kwargs["coefficient"]
    assert applied == 2.0 * artifact_value
    assert applied != 2.0 * sae_value


def test_stage4_rejects_direction_regenerated_after_preflight(
    tmp_path: Path,
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    cfg = _make_cfg(output_root=str(tmp_path / "out"))
    method = _method_diffmean()
    concept = _make_concept()
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    artifact_path = output_paths.diffmean_direction(
        "decoder_only", method.mode, concept.concept_id
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(value: float) -> None:
        save_file(
            {"direction": t.tensor([1.0, 0.0, 0.0])},
            str(artifact_path),
            metadata=_direction_metadata(mode=method.mode, value=value),
        )

    _write(7.0)
    preflight = preflight_direction_scales(
        [
            DirectionScaleRequest(
                family="diffmean",
                target_model="decoder_only",
                mode="positive",
                concept_id=concept.concept_id,
                path=artifact_path,
            )
        ],
        expected_layer=cfg.layer_hooks.diffmean_and_probe_layer,
        stage_label="Stage 4 test",
    )
    _write(11.0)

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=MagicMock(),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=GenerationResult(
                text="unused", token_count=0, stop_reason="bos_token"
            ),
        ),
        pytest.raises(
            AxBenchBaseError,
            match="stage-4 direction-scale mid-run regeneration hazard.*artifact key",
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=1.0,
            prefix_tokens=[1],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=True,
            expected_max_activation=preflight.values[
                ("diffmean", "decoder_only", "positive", concept.concept_id)
            ],
        )


def test_diffmean_negative_scales_abs_coefficient_by_artifact_value(
    fake_model: _FakeModel, stub_tokenizer: _StubTokenizer
) -> None:
    coef, _ = _run_diffmean_capturing_coefficient(
        mode=SteeringMode.NEGATIVE,
        coefficient=1.5,
        apply_max_act_scale=True,
        fake_model=fake_model,
        stub_tokenizer=stub_tokenizer,
    )
    # negative mode: -abs(1.5) * 7.0
    assert coef == pytest.approx(-1.5 * 7.0)


def test_diffmean_off_mode_is_identity(
    fake_model: _FakeModel, stub_tokenizer: _StubTokenizer
) -> None:
    coef, direction = _run_diffmean_capturing_coefficient(
        mode=SteeringMode.POSITIVE,
        coefficient=0.5,
        apply_max_act_scale=False,
        fake_model=fake_model,
        stub_tokenizer=stub_tokenizer,
        metadata={},
    )
    assert coef == pytest.approx(0.5)
    assert t.equal(direction, t.tensor([1.0, 2.0, 3.0]))


def test_diffmean_scaled_missing_field_raises(
    fake_model: _FakeModel, stub_tokenizer: _StubTokenizer
) -> None:
    with pytest.raises(AxBenchBaseError, match="missing 'max_activation'"):
        _run_diffmean_capturing_coefficient(
            mode=SteeringMode.POSITIVE,
            coefficient=0.5,
            apply_max_act_scale=True,
            fake_model=fake_model,
            stub_tokenizer=stub_tokenizer,
            metadata={
                "concept_id": "7",
                "mode": "positive",
                "target_model": "decoder_only",
                "layer": "3",
            },
        )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("concept_id", "8"),
        ("mode", "negative"),
        ("target_model", "encdec"),
        ("layer", "6"),
    ],
)
def test_diffmean_scaled_binding_mismatch_names_field(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    field: str,
    wrong: str,
) -> None:
    metadata = _direction_metadata(mode=SteeringMode.POSITIVE)
    metadata[field] = wrong
    with pytest.raises(
        AxBenchBaseError, match=rf"direction metadata mismatch.*{field}"
    ):
        _run_diffmean_capturing_coefficient(
            mode=SteeringMode.POSITIVE,
            coefficient=0.5,
            apply_max_act_scale=True,
            fake_model=fake_model,
            stub_tokenizer=stub_tokenizer,
            metadata=metadata,
        )


def _write_clustered_scalar_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_kind: str = "diffmean_direction_v1",
) -> tuple[Path, Path, Method, Concept, t.Tensor]:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    method = _method_diffmean()
    concept = _make_concept()
    corpus_path = test_paths.default_paths("test-run").synth_training_positive()
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_bytes(b"exact clustered scalar corpus")
    corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    direction_path = test_paths.default_paths("test-run").diffmean_direction(
        "decoder_only",
        SteeringMode.POSITIVE,
        concept.concept_id,
    )
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    direction = t.tensor([1.0, 2.0, 3.0])
    save_file(
        {"direction": direction},
        str(direction_path),
        metadata={
            "feature_contract_fingerprint": "a" * 64,
            "source_corpus_sha256": corpus_sha256,
            "artifact_kind": artifact_kind,
            "concept_id": str(concept.concept_id),
            "mode": SteeringMode.POSITIVE.value,
            "target_model": "decoder_only",
            "layer": "3",
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(7.0),
        },
    )
    return direction_path, corpus_path, method, concept, direction


def test_clustered_scalar_artifact_requires_exact_corpus_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_path, corpus_path, method, concept, direction = _write_clustered_scalar_fixture(
        tmp_path, monkeypatch
    )
    contract = cast(Any, _ConceptFeatureContract())

    loaded, metadata = _load_scalar_direction_artifact(
        direction_path,
        method=method,
        concept=concept,
        target_model_name="decoder_only",
        layer=3,
        output_paths=test_paths.default_paths("test-run"),
        cluster_contract=contract,
    )
    assert t.equal(loaded, direction)
    assert metadata[MAX_ACTIVATION_METADATA_KEY] == encode_max_activation(7.0)

    corpus_path.write_bytes(b"mutated clustered scalar corpus")
    with pytest.raises(AxBenchBaseError, match="identity mismatch"):
        _load_scalar_direction_artifact(
            direction_path,
            method=method,
            concept=concept,
            target_model_name="decoder_only",
            layer=3,
            output_paths=test_paths.default_paths("test-run"),
            cluster_contract=contract,
        )


def test_scalar_artifact_cluster_ownership_is_bidirectional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_path, _corpus_path, method, concept, _direction = _write_clustered_scalar_fixture(
        tmp_path,
        monkeypatch,
    )

    with pytest.raises(AxBenchBaseError, match="rejects cluster-stamped"):
        _load_scalar_direction_artifact(
            direction_path,
            method=method,
            concept=concept,
            target_model_name="decoder_only",
            layer=3,
            output_paths=test_paths.default_paths("test-run"),
            cluster_contract=None,
        )
    direction_path, _corpus_path, method, concept, _direction = _write_clustered_scalar_fixture(
        tmp_path,
        monkeypatch,
        artifact_kind="linear_probe_direction_v1",
    )
    with pytest.raises(AxBenchBaseError, match="artifact_kind"):
        _load_scalar_direction_artifact(
            direction_path,
            method=method,
            concept=concept,
            target_model_name="decoder_only",
            layer=3,
            output_paths=test_paths.default_paths("test-run"),
            cluster_contract=cast(Any, _ConceptFeatureContract()),
        )


def test_clustered_scalar_artifact_rejects_malformed_source_corpus_sha256(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_path, _corpus_path, method, concept, _direction = _write_clustered_scalar_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(AxBenchBaseError, match="lowercase SHA-256"):
        _load_scalar_direction_artifact(
            direction_path,
            method=method,
            concept=concept,
            target_model_name="decoder_only",
            layer=3,
            output_paths=test_paths.default_paths("test-run"),
            cluster_contract=cast(Any, _ConceptFeatureContract()),
            source_corpus_sha256="g" * 64,
        )


def test_clustered_scalar_artifact_wraps_source_corpus_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_path, _corpus_path, method, concept, _direction = _write_clustered_scalar_fixture(
        tmp_path, monkeypatch
    )

    with patch(
        "steering.eval.axbench_base.intervention_dispatch.stream_hash_file_stable",
        side_effect=OSError("mutation during read"),
    ):
        with pytest.raises(
            AxBenchBaseError,
            match="could not verify.*source corpus",
        ):
            _load_scalar_direction_artifact(
                direction_path,
                method=method,
                concept=concept,
                target_model_name="decoder_only",
                layer=3,
                output_paths=test_paths.default_paths("test-run"),
                cluster_contract=cast(Any, _ConceptFeatureContract()),
            )


# ---------------------------------------------------------------------------
# 4. LinearProbe decoder-only installs hook
# ---------------------------------------------------------------------------


def test_linear_probe_decoder_only_installs_hook(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_linear_probe(target="decoder_only", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    direction = t.tensor([0.5, 0.25])
    handle = MagicMock()
    expected = GenerationResult(text="lp", token_count=2, stop_reason="bos_token")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(direction, {}),
        ) as load_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=handle,
        ) as hook_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ) as gen_spy,
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=3,
            coefficient=2.0,
            prefix_tokens=[1, 2],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    # Path resolver was hit once.
    assert load_spy.call_count == 1
    assert "linear_probe" in str(load_spy.call_args.args[0])
    # Hook was installed with the loaded direction, positive coefficient, and
    # the configured layer.
    hook_kwargs = hook_spy.call_args.kwargs
    assert t.equal(hook_kwargs["direction"], direction)
    assert hook_kwargs["coefficient"] == pytest.approx(2.0)
    assert hook_kwargs["layer"] == cfg.layer_hooks.diffmean_and_probe_layer
    assert gen_spy.call_count == 1
    assert handle.remove.call_count == 1


# ---------------------------------------------------------------------------
# 5. LoRA attaches then detaches
# ---------------------------------------------------------------------------


def test_lora_attaches_then_detaches(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_lora(mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    detach_fn = MagicMock()
    expected = GenerationResult(text="lora", token_count=4, stop_reason="bos_token")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.attach_lora_from_disk",
            return_value=([], detach_fn),
        ) as attach_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ) as gen_spy,
    ):
        result = run_intervention(
            method=method,
            concept=concept,
            eval_seed=4,
            coefficient=None,
            prefix_tokens=[1, 2, 3],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    assert result == expected
    assert attach_spy.call_count == 1
    attach_kwargs = attach_spy.call_args.kwargs
    assert attach_kwargs["lora_cfg"] is cfg.lora
    assert attach_kwargs["layer"] == cfg.layer_hooks.diffmean_and_probe_layer
    assert gen_spy.call_count == 1
    assert detach_fn.call_count == 1


def test_clustered_lora_guard_is_after_attach_and_immediately_before_forward(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    events: list[str] = []
    expected = GenerationResult(text="lora", token_count=1, stop_reason="bos_token")

    def _attach(**_kwargs: Any) -> tuple[list[Any], Callable[[], None]]:
        events.append("attach")
        return [], lambda: events.append("detach")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.attach_lora_from_disk",
            side_effect=_attach,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            side_effect=lambda **_kwargs: events.append("forward") or expected,
        ),
    ):
        result = _dispatch_lora(
            method=_method_lora(),
            concept=_make_concept(),
            prefix_tokens=[1],
            model=fake_model,
            tokenizer=stub_tokenizer,
            device="cpu",
            cfg=_make_cfg(),
            context_provider=None,
            sampling_seed=0,
            pre_forward_guard=lambda: events.append("guard"),
        )

    assert result == expected
    assert events == ["attach", "guard", "forward", "detach"]


# ---------------------------------------------------------------------------
# 6. Prompting dispatch consumes an already constructed prompt
# ---------------------------------------------------------------------------


def test_prompting_positive_forwards_model_ready_prompt_unchanged(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_prompting(target="decoder_only", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept(concept_id=11)
    expected = GenerationResult(text="p", token_count=1, stop_reason="bos_token")

    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        return_value=expected,
    ) as gen_spy:
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=5,
            coefficient=None,
            prefix_tokens=[50, 51, 52],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    assert gen_spy.call_count == 1
    passed_tokens = gen_spy.call_args.kwargs["nanochat_tokens"]
    assert passed_tokens == [50, 51, 52]
    assert stub_tokenizer.encode_calls == []


def test_clustered_prompting_guard_runs_immediately_before_first_forward(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    events: list[str] = []
    expected = GenerationResult(text="p", token_count=1, stop_reason="bos_token")

    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        side_effect=lambda **_kwargs: events.append("forward") or expected,
    ):
        result = _dispatch_prompting(
            prefix_tokens=[50],
            model=fake_model,
            tokenizer=stub_tokenizer,
            device="cpu",
            cfg=_make_cfg(),
            context_provider=_EmptyContextProvider(),
            sampling_seed=0,
            pre_forward_guard=lambda: events.append("guard"),
        )

    assert result == expected
    assert events == ["guard", "forward"]


# ---------------------------------------------------------------------------
# 7. Prompting negative prepends "Not related to"
# ---------------------------------------------------------------------------


def test_prompting_negative_forwards_model_ready_prompt_unchanged(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_prompting(target="decoder_only", mode=SteeringMode.NEGATIVE)
    cfg = _make_cfg()
    concept = _make_concept(concept_id=13)
    expected = GenerationResult(text="p", token_count=1, stop_reason="bos_token")

    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        return_value=expected,
    ) as gen_spy:
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=6,
            coefficient=None,
            prefix_tokens=[60, 61],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    assert gen_spy.call_args.kwargs["nanochat_tokens"] == [60, 61]
    assert stub_tokenizer.encode_calls == []


# ---------------------------------------------------------------------------
# 8. Ours methods call build_mode_aware_encoder_input
# ---------------------------------------------------------------------------


def test_ours_methods_call_build_mode_aware_encoder_input(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    tmp_path: Path,
) -> None:
    method = _method_ours(suffix="extract_and_add", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept(concept_id=0)
    expected = GenerationResult(text="o", token_count=1, stop_reason="bos_token")

    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([1234], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.build_mode_aware_encoder_input",
            return_value={1: 2, 2: 3},
        ) as builder_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ) as gen_spy,
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=7,
            coefficient=1,
            prefix_tokens=[70, 71, 72],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=cf_path,
            apply_max_act_scale=False,
        )

    assert builder_spy.call_count == 1
    kwargs = builder_spy.call_args.kwargs
    assert kwargs["method"] is method
    assert kwargs["top_k_concept_fids"] == [10, 20]
    assert kwargs["coefficient"] == 1
    # The merged dict was forwarded to the generation helper.
    assert gen_spy.call_count == 1
    gen_kwargs = gen_spy.call_args.kwargs
    assert gen_kwargs["encoder_input"] == {1: 2, 2: 3}
    assert gen_kwargs["nanochat_tokens"] == [70, 71, 72]


def test_ours_delta_ranked_concept_features_validate_tokenizer_metadata(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    tmp_path: Path,
) -> None:
    method = _method_ours(suffix="extract_and_add", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg().replace(extraction_tokenizer_id="custom/current-tokenizer")
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([1234], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "delta_ranked",
            EXTRACTION_TOKENIZER_METADATA_KEY: "custom/stale-tokenizer",
        },
    )

    with pytest.raises(
        AxBenchBaseError,
        match="custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        run_intervention(
            method=method,
            concept=_make_concept(concept_id=0),
            eval_seed=7,
            coefficient=1,
            prefix_tokens=[70, 71, 72],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=cf_path,
            apply_max_act_scale=False,
        )


# ---------------------------------------------------------------------------
# 10. Unknown baseline raises AxBenchBaseError
# ---------------------------------------------------------------------------


def test_unknown_baseline_raises(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    # The test exercises the dispatcher's "unknown baseline" raise, so
    # we deliberately smuggle a bogus string into a field typed as
    # :class:`BaselineKind`. A :func:`typing.cast` keeps ``ty`` happy.
    method = Method(
        name="bogus_method",
        mode=SteeringMode.POSITIVE,
        baseline=cast(BaselineKind, "bogus"),
        target_model="decoder_only",
        requires_negative_sampling=False,
        coefficient_grid_fn=single_condition_grid,
        partition_modifier_name=None,
    )
    cfg = _make_cfg()
    concept = _make_concept()

    with pytest.raises(AxBenchBaseError, match="Unknown method baseline"):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=None,
            prefix_tokens=[1],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )


# ---------------------------------------------------------------------------
# 11. Hook removed on generation exception
# ---------------------------------------------------------------------------


def test_hook_removed_on_generation_exception(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_diffmean(target="decoder_only", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    direction = t.tensor([1.0, 2.0])
    handle = MagicMock()

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(direction, {}),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=handle,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            side_effect=RuntimeError("boom"),
        ),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            run_intervention(
                method=method,
                concept=concept,
                eval_seed=0,
                coefficient=1.0,
                prefix_tokens=[1],
                target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
                cfg=cfg,
                concept_features_path=Path("irrelevant"),
                apply_max_act_scale=False,
            )

    # Even though generation raised, the hook handle was removed.
    assert handle.remove.call_count == 1


# ---------------------------------------------------------------------------
# 12. LoRA detached on generation exception
# ---------------------------------------------------------------------------


def test_lora_detached_on_generation_exception(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    method = _method_lora(mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    detach_fn = MagicMock()

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.attach_lora_from_disk",
            return_value=([], detach_fn),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            side_effect=RuntimeError("lora boom"),
        ),
    ):
        with pytest.raises(RuntimeError, match="lora boom"):
            run_intervention(
                method=method,
                concept=concept,
                eval_seed=0,
                coefficient=None,
                prefix_tokens=[1],
                target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
                cfg=cfg,
                concept_features_path=Path("irrelevant"),
                apply_max_act_scale=False,
            )

    assert detach_fn.call_count == 1


# ---------------------------------------------------------------------------
# 13. Sampling seed determinism
# ---------------------------------------------------------------------------


def test_sampling_seed_deterministic() -> None:
    s1 = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=42)
    s2 = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=42)
    assert s1 == s2


# ---------------------------------------------------------------------------
# 14. Different methods yield different sampling seeds
# ---------------------------------------------------------------------------


def test_sampling_seed_different_methods_differ() -> None:
    a = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=42)
    b = _sampling_seed("linear_probe", concept_id=1, eval_seed=2, master_seed=42)
    assert a != b


def test_sampling_seed_different_concepts_differ() -> None:
    """Defensive: concept_id is also mixed into the seed."""
    a = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=42)
    b = _sampling_seed("diffmean", concept_id=99, eval_seed=2, master_seed=42)
    assert a != b


def test_sampling_seed_different_eval_seeds_differ() -> None:
    a = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=42)
    b = _sampling_seed("diffmean", concept_id=1, eval_seed=3, master_seed=42)
    assert a != b


def test_sampling_seed_different_master_seeds_differ() -> None:
    a = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=42)
    b = _sampling_seed("diffmean", concept_id=1, eval_seed=2, master_seed=43)
    assert a != b


def test_sampling_seed_fits_int31() -> None:
    s = _sampling_seed("diffmean", concept_id=10_000, eval_seed=9999, master_seed=1)
    assert 0 <= s < (1 << 31)


# ---------------------------------------------------------------------------
# 15. Dispatch preserves target model state
# ---------------------------------------------------------------------------


def test_dispatch_preserves_target_model_state(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    """After a diffmean dispatch completes, the model should have no
    residual forward-hooks on its transformer blocks.

    (We mock the hook-install side-effect in every other test; here we
    still mock the hook but additionally verify no real hooks accumulate
    on the model — an important regression guard for the finally block.)
    """
    method = _method_diffmean(target="decoder_only", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    direction = t.tensor([1.0, 2.0])
    handle = MagicMock()
    expected = GenerationResult(text="x", token_count=1, stop_reason="bos_token")

    # Snapshot: the fake model starts with no forward hooks.
    def _all_forward_hooks(m: nn.Module) -> int:
        return sum(len(mod._forward_hooks) for mod in m.modules())

    before = _all_forward_hooks(fake_model)

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(direction, {}),
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=handle,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=1.0,
            prefix_tokens=[1],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    after = _all_forward_hooks(fake_model)
    assert after == before
    # Handle removed once.
    assert handle.remove.call_count == 1


# ---------------------------------------------------------------------------
# Additional coverage: ours_replace_all, negative ours method, inactive sampler
# ---------------------------------------------------------------------------


def test_ours_negative_method_passes_inactive_sampler(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    tmp_path: Path,
) -> None:
    method = _method_ours(
        suffix="remove",
        mode=SteeringMode.NEGATIVE,
        modifier_name=PartitionModifier.NEG_OURS_REMOVE,
    )
    cfg = _make_cfg()
    concept = _make_concept(concept_id=0)

    cf_path = tmp_path / "concept_features_neg.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[77, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([1234], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    expected = GenerationResult(text="n", token_count=1, stop_reason="bos_token")
    sampler = MagicMock(name="inactive_sampler")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.build_mode_aware_encoder_input",
            return_value={},
        ) as builder_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=None,
            prefix_tokens=[1],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=cf_path,
            apply_max_act_scale=False,
            inactive_sampler=sampler,
        )

    # Sampler was forwarded into the encoder-input builder.
    kwargs = builder_spy.call_args.kwargs
    assert kwargs["inactive_sampler"] is sampler
    assert kwargs["top_k_concept_fids"] == [77]


def test_top_k_concept_fids_filters_sentinel_negative_ones(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    tmp_path: Path,
) -> None:
    """``-1`` is the sentinel for unused slots — must not appear in the
    top-K list forwarded to the modifier."""
    method = _method_ours(suffix="extract_and_add", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept(concept_id=0)

    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[99, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([1234], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    expected = GenerationResult(text="x", token_count=1, stop_reason="bos_token")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.build_mode_aware_encoder_input",
            return_value={},
        ) as builder_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=1,
            prefix_tokens=[1],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=cf_path,
            apply_max_act_scale=False,
        )

    assert builder_spy.call_args.kwargs["top_k_concept_fids"] == [99]


def test_prompting_dispatch_never_reads_description_artifacts(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    """Description lookup belongs to the shared Stage 4 prompt builder."""
    method = _method_prompting(target="decoder_only", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept(concept_id=99)
    with patch(
        "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
        return_value=GenerationResult(text="", token_count=0, stop_reason="bos_token"),
    ) as gen_spy:
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=None,
            prefix_tokens=[1],
            target=_target_bundle("decoder_only", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )
    assert gen_spy.call_args.kwargs["nanochat_tokens"] == [1]


def test_ours_encoder_input_forwarded_nonempty(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
    tmp_path: Path,
) -> None:
    """Smoke test: the ``encoder_input`` dict produced by the builder is
    threaded through to ``generate_stream``."""
    method = _method_ours(suffix="replace_all", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept(concept_id=0)

    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[5, 6, 7]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([1234], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    encoder_out = {5: 1, 6: 1, 7: 1}
    expected = GenerationResult(text="abc", token_count=3, stop_reason="bos_token")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch.build_mode_aware_encoder_input",
            return_value=encoder_out,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ) as gen_spy,
    ):
        result = run_intervention(
            method=method,
            concept=concept,
            eval_seed=1,
            coefficient=1,
            prefix_tokens=[100],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=cf_path,
            apply_max_act_scale=False,
        )

    assert result == expected
    assert gen_spy.call_args.kwargs["encoder_input"] == encoder_out


def test_diffmean_encdec_uses_encdec_path(
    fake_model: _FakeModel,
    stub_tokenizer: _StubTokenizer,
) -> None:
    """``target_model_name="encdec"`` should be forwarded into the
    diffmean path resolver."""
    method = _method_diffmean(target="encdec", mode=SteeringMode.POSITIVE)
    cfg = _make_cfg()
    concept = _make_concept()
    direction = t.tensor([1.0, 2.0])
    handle = MagicMock()
    expected = GenerationResult(text="x", token_count=1, stop_reason="bos_token")

    with (
        patch(
            "steering.eval.axbench_base.intervention_dispatch._load_scalar_direction_artifact",
            return_value=(direction, {}),
        ) as load_spy,
        patch(
            "steering.eval.axbench_base.intervention_dispatch.register_residual_hook",
            return_value=handle,
        ),
        patch(
            "steering.eval.axbench_base.intervention_dispatch._generate_with_nanochat_tokens",
            return_value=expected,
        ),
    ):
        run_intervention(
            method=method,
            concept=concept,
            eval_seed=0,
            coefficient=0.1,
            prefix_tokens=[1],
            target=_target_bundle("encdec", fake_model, stub_tokenizer),
            cfg=cfg,
            concept_features_path=Path("irrelevant"),
            apply_max_act_scale=False,
        )

    # Path loaded must contain "encdec" as the target model segment.
    assert "encdec" in str(load_spy.call_args.args[0])


# ---------------------------------------------------------------------------
# concept_features safetensors metadata validation
# ---------------------------------------------------------------------------


def test_load_top_k_concept_fids_validates_metadata_mode_present(
    tmp_path: Path,
) -> None:
    """``_load_top_k_concept_fids`` must validate the safetensors
    ``metadata={"mode": ...}`` annotation written by Stage 2. Missing
    metadata → ``AxBenchBaseError``. Without this check, a corrupted
    or hand-crafted file would silently load and Stage 4 would use
    the wrong feature rankings.

    Defense surfaced by Phase 2 cross-cutting Stream 1 (data-flow audit).
    """
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        # NOTE: deliberately no metadata kwarg.
    )
    with pytest.raises(AxBenchBaseError, match="metadata mode=None"):
        _load_top_k_concept_fids(cf_path, concept_id=0)


def test_load_top_k_concept_fids_validates_metadata_mode_value(
    tmp_path: Path,
) -> None:
    """An unknown ``mode`` (neither ``"canonical"`` nor
    ``"delta_ranked"``) raises with a clear list of valid options."""
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "fancy_new_mode"},
    )
    with pytest.raises(
        AxBenchBaseError,
        match="mode='fancy_new_mode'.*canonical.*delta_ranked",
    ):
        _load_top_k_concept_fids(cf_path, concept_id=0)


def test_load_top_k_concept_fids_accepts_canonical_metadata(
    tmp_path: Path,
) -> None:
    """Happy path: ``mode="canonical"`` metadata yields the top-K fids."""
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )
    fids = _load_top_k_concept_fids(cf_path, concept_id=0)
    assert fids == [10, 20]  # -1 sentinel filtered


class _ConceptFeatureClusterProjection:
    @staticmethod
    def assert_representatives_only(feature_ids: t.Tensor, *, surface: str) -> None:
        del surface
        invalid = [fid for fid in feature_ids.tolist() if fid not in {10, 20}]
        if invalid:
            raise FeatureClustersError(
                f"concept feature test contains non-representative feature ID {invalid[0]}"
            )


class _ConceptFeatureContract:
    feature_contract_fingerprint = "a" * 64
    consolidation = _ConceptFeatureClusterProjection()


@pytest.mark.parametrize("artifact_fingerprint", [None, "b" * 64])
def test_clustered_concept_features_require_exact_fingerprint(
    tmp_path: Path,
    artifact_fingerprint: str | None,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    metadata = {"mode": "canonical"}
    if artifact_fingerprint is not None:
        metadata["feature_contract_fingerprint"] = artifact_fingerprint
    save_file(
        {"feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32)},
        str(cf_path),
        metadata=metadata,
    )

    with pytest.raises(AxBenchBaseError, match="do not match.*feature contract"):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            cluster_contract=cast(Any, _ConceptFeatureContract()),
        )


def test_stale_contract_concept_features_are_rejected(
    tmp_path: Path,
) -> None:
    """An artifact produced under contract A cannot be loaded under contract B."""

    cf_path = tmp_path / "concept_features_contract_a.safetensors"
    save_file(
        {"feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32)},
        str(cf_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": "a" * 64,
        },
    )
    contract_b = SimpleNamespace(
        feature_contract_fingerprint="b" * 64,
        consolidation=_ConceptFeatureClusterProjection(),
    )

    with pytest.raises(AxBenchBaseError, match="do not match.*feature contract"):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            cluster_contract=cast(Any, contract_b),
        )


def test_clustered_concept_features_accept_reps_and_unclustered_rejects_stamp(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, -1, -1]], dtype=t.int32),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": "a" * 64,
        },
    )

    assert _load_top_k_concept_fids(
        cf_path,
        concept_id=0,
        cluster_contract=cast(Any, _ConceptFeatureContract()),
        expected_representative_id=10,
    ) == [10]
    with pytest.raises(AxBenchBaseError, match="rejects cluster-stamped"):
        _load_top_k_concept_fids(cf_path, concept_id=0)


def test_clustered_concept_features_require_canonical_feature_identity(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {"feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32)},
        str(cf_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": "a" * 64,
        },
    )

    with pytest.raises(AxBenchBaseError, match="missing canonical_feature_id"):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            cluster_contract=cast(Any, _ConceptFeatureContract()),
            expected_representative_id=10,
        )


def test_clustered_concept_features_reject_swapped_canonical_row(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "canonical_feature_id": t.tensor([999], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": "a" * 64,
        },
    )

    with pytest.raises(
        AxBenchBaseError,
        match="canonical_feature_id.*999.*Concept.feature_index.*10",
    ):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            cluster_contract=cast(Any, _ConceptFeatureContract()),
            expected_representative_id=10,
        )


def test_clustered_concept_features_reject_nonrepresentative_row(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 30, -1]], dtype=t.int32),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": "a" * 64,
        },
    )

    with pytest.raises(FeatureClustersError, match="non-representative feature ID 30"):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            cluster_contract=cast(Any, _ConceptFeatureContract()),
            expected_representative_id=10,
        )


def test_clustered_concept_features_reject_extra_representative(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": "a" * 64,
        },
    )

    with pytest.raises(AxBenchBaseError, match=r"expected exactly \[10\]"):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            cluster_contract=cast(Any, _ConceptFeatureContract()),
            expected_representative_id=10,
        )


def test_load_top_k_concept_fids_rejects_negative_non_sentinel(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, -2, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    with pytest.raises(AxBenchBaseError, match="only -1 is a valid.*sentinel"):
        _load_top_k_concept_fids(cf_path, concept_id=0)


def test_load_top_k_concept_fids_accepts_delta_ranked_metadata(
    tmp_path: Path,
) -> None:
    """Happy path: ``mode="delta_ranked"`` metadata also accepted."""
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[42, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([42], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "delta_ranked"},
    )
    fids = _load_top_k_concept_fids(cf_path, concept_id=0)
    assert fids == [42]


def test_load_top_k_concept_fids_rejects_delta_ranked_missing_tokenizer_metadata(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[42, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([42], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "delta_ranked"},
    )
    with pytest.raises(
        AxBenchBaseError,
        match="delta_ranked but missing 'extraction_tokenizer_id' metadata",
    ):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            expected_tokenizer_id="custom/current-tokenizer",
        )


def test_load_top_k_concept_fids_rejects_delta_ranked_mismatched_tokenizer_metadata(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[42, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([42], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "delta_ranked",
            EXTRACTION_TOKENIZER_METADATA_KEY: "custom/stale-tokenizer",
        },
    )
    with pytest.raises(
        AxBenchBaseError,
        match="custom/stale-tokenizer.*custom/current-tokenizer",
    ):
        _load_top_k_concept_fids(
            cf_path,
            concept_id=0,
            expected_tokenizer_id="custom/current-tokenizer",
        )


def test_load_top_k_concept_fids_accepts_delta_ranked_matching_tokenizer_metadata(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[42, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([42], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={
            "mode": "delta_ranked",
            EXTRACTION_TOKENIZER_METADATA_KEY: "custom/current-tokenizer",
        },
    )
    fids = _load_top_k_concept_fids(
        cf_path,
        concept_id=0,
        expected_tokenizer_id="custom/current-tokenizer",
    )
    assert fids == [42]


def test_load_top_k_concept_fids_keeps_canonical_tokenizer_agnostic(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 20, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )
    assert _load_top_k_concept_fids(
        cf_path,
        concept_id=0,
        expected_tokenizer_id="custom/current-tokenizer",
    ) == [10, 20]


def test_load_top_k_concept_fids_indexes_requested_concept_row(
    tmp_path: Path,
) -> None:
    """Stage 4 must load the row for ``concept_id``, not row 0 for every concept."""
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor(
                [
                    [10, 11, -1],
                    [20, 21, -1],
                    [30, -1, -1],
                ],
                dtype=t.int32,
            ),
            "percentile_deltas": t.zeros(3, 3),
            "canonical_feature_id": t.tensor([10, 20, 30], dtype=t.int32),
            "canonical_rank": t.tensor([0, 0, 0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    assert _load_top_k_concept_fids(cf_path, concept_id=1) == [20, 21]


def test_load_top_k_concept_fids_rejects_out_of_bounds_concept_id(
    tmp_path: Path,
) -> None:
    cf_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[10, 11, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor([10], dtype=t.int32),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(cf_path),
        metadata={"mode": "canonical"},
    )

    with pytest.raises(AxBenchBaseError, match="out of bounds"):
        _load_top_k_concept_fids(cf_path, concept_id=3)
