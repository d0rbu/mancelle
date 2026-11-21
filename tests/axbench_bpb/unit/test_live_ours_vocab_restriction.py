from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.encoder_input import build_mode_aware_encoder_input
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept, TargetModelMetadata
from steering.eval.axbench_bpb import intervention as intervention_module
from steering.eval.axbench_bpb.intervention import build_intervention
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import AxBenchBpbError, Intervention

LIVE_METHODS = (
    "ours_extract_and_add_encdec",
    "ours_replace_random_encdec",
    "ours_replace_all_encdec",
    "neg_ours_remove_encdec",
    "neg_ours_add_to_negatives_encdec",
    "neg_ours_replace_random_negatives_encdec",
    "neg_ours_replace_all_negatives_encdec",
)
RAW_TOP_K = [9, 3, 11, 3, 14]
ALLOWED = [1, 3, 7, 11, 12]
FILTERED_TOP_K = [3, 11, 3]
SAMPLED_INACTIVE = {1: 4, 3: 4, 7: 4, 12: 4}


def _concept() -> Concept:
    return Concept(
        concept_id=0,
        feature_index=3,
        layer=8,
        sae_source="test",
        description="test concept",
        corpus_count=10,
        description_source="test",
    )


def _meta(*, negative_selection: dict[str, Any] | None) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=5,
        n_sae_features=16,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _model(
    allowed: list[int] | None,
    *,
    device: str = "cpu",
) -> SimpleNamespace:
    if allowed is None:
        return SimpleNamespace(feature_vocab=None)
    return SimpleNamespace(
        feature_vocab=object(),
        encoder=SimpleNamespace(
            feature_vocab_ids=t.tensor(
                allowed,
                dtype=t.int64,
                device=device,
            )
        ),
    )


def _write_concept_artifacts(root: Path, top_k: list[int]) -> SourcePaths:
    tensor = t.tensor([top_k or [-1]], dtype=t.int64)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("concept_features.safetensors", "concept_features_neg.safetensors"):
        save_file(
            {"feature_ids": tensor},
            str(root / name),
            metadata={"mode": "canonical"},
        )
    return SourcePaths(source_run_dir=root)


def _build(
    *,
    method_name: str,
    source_paths: SourcePaths,
    model: Any,
    rng: random.Random | None = None,
    inactive_sampler: Any | None = None,
    target_model_meta: Any | None = None,
) -> Intervention:
    return build_intervention(
        method=get_method_by_name(method_name),
        concept=_concept(),
        coefficient=2,
        source_paths=source_paths,
        diffmean_layer=8,
        target_model_meta=(
            target_model_meta
            if target_model_meta is not None
            else _meta(negative_selection={"distribution": "geometric", "mean": 3.0})
        ),
        inactive_sampler=inactive_sampler,
        tokenizer_nanochat=None,
        rng=rng or random.Random(0),
        model=model,
        apply_max_act_scale=False,
    )


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [
        ("ours_extract_and_add_encdec", {3: 2, 11: 2}),
        ("ours_replace_random_encdec", {3: 2, 11: 2}),
        ("ours_replace_all_encdec", {3: 2, 11: 2}),
        ("neg_ours_remove_encdec", SAMPLED_INACTIVE),
        (
            "neg_ours_add_to_negatives_encdec",
            {1: 4, 3: 4, 7: 4, 12: 4, 11: 4},
        ),
        ("neg_ours_replace_random_negatives_encdec", {3: 4, 11: 4}),
        ("neg_ours_replace_all_negatives_encdec", {3: 4, 11: 4}),
    ],
)
def test_all_live_methods_restrict_both_producers_before_conditioning(
    tmp_path: Path,
    method_name: str,
    expected: dict[int, int],
) -> None:
    calls = 0

    def sampler(*, rng: random.Random) -> dict[int, int]:
        nonlocal calls
        del rng
        calls += 1
        return dict(SAMPLED_INACTIVE)

    intervention = _build(
        method_name=method_name,
        source_paths=_write_concept_artifacts(tmp_path, RAW_TOP_K),
        model=_model(ALLOWED),
        inactive_sampler=sampler,
    )

    assert intervention.feature_conditions == expected
    assert set(intervention.feature_conditions or {}).issubset(ALLOWED)
    assert 9 not in (intervention.feature_conditions or {})
    assert 14 not in (intervention.feature_conditions or {})
    assert calls == (1 if method_name.startswith("neg_") else 0)


def test_filtered_order_and_duplicates_reach_modifier_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[int]] = []

    def spy_builder(**kwargs: Any) -> dict[int, int]:
        seen.append(kwargs["top_k_concept_fids"])
        return {3: 2, 11: 2}

    monkeypatch.setattr(
        intervention_module,
        "build_mode_aware_encoder_input",
        spy_builder,
    )
    _build(
        method_name="ours_replace_random_encdec",
        source_paths=_write_concept_artifacts(tmp_path, RAW_TOP_K),
        model=_model(ALLOWED),
    )
    assert seen == [FILTERED_TOP_K]


@pytest.mark.parametrize("method_name", LIVE_METHODS)
def test_all_filtered_out_and_empty_artifact_are_legal(
    tmp_path: Path,
    method_name: str,
) -> None:
    sampler_calls = 0

    def sampler(*, rng: random.Random) -> dict[int, int]:
        nonlocal sampler_calls
        del rng
        sampler_calls += 1
        return {1: 4, 7: 4}

    top_k = [] if method_name.endswith("all_encdec") else [9, 14]
    intervention = _build(
        method_name=method_name,
        source_paths=_write_concept_artifacts(tmp_path, top_k),
        model=_model([1, 7]),
        inactive_sampler=sampler,
    )
    conditions = intervention.feature_conditions or {}
    if method_name.startswith("ours_") or method_name.endswith("replace_all_negatives_encdec"):
        assert conditions == {}
    else:
        assert conditions == {1: 4, 7: 4}
    assert sampler_calls == (1 if method_name.startswith("neg_") else 0)


@pytest.mark.parametrize("method_name", LIVE_METHODS)
def test_uncompacted_path_matches_raw_legacy_reference_and_rng_state(
    tmp_path: Path,
    method_name: str,
) -> None:
    source_paths = _write_concept_artifacts(tmp_path, RAW_TOP_K)
    method = get_method_by_name(method_name)
    meta = _meta(negative_selection={"distribution": "geometric", "mean": 3.0})

    def sampler(*, rng: random.Random) -> dict[int, int]:
        del rng
        return dict(SAMPLED_INACTIVE)

    actual_rng = random.Random(2026)
    expected_rng = random.Random(2026)
    actual = _build(
        method_name=method_name,
        source_paths=source_paths,
        model=_model(None),
        rng=actual_rng,
        inactive_sampler=sampler,
    )
    expected = build_mode_aware_encoder_input(
        method=method,
        active={},
        inactive={},
        top_k_concept_fids=RAW_TOP_K,
        coefficient=2,
        target_model_meta=meta,
        inactive_sampler=sampler,
        rng=expected_rng,
    )

    assert actual.feature_conditions == expected
    assert list((actual.feature_conditions or {}).items()) == list(expected.items())
    assert actual_rng.getstate() == expected_rng.getstate()


def test_uncompacted_path_passes_original_top_k_list_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = [5, 2, 5]
    monkeypatch.setattr(
        intervention_module,
        "_load_top_k_concept_fids",
        lambda *_args, **_kwargs: raw,
    )
    monkeypatch.setattr(
        intervention_module,
        "restrict_to_vocab",
        lambda *_args, **_kwargs: pytest.fail("legacy path must not call restrict_to_vocab"),
    )

    def spy_builder(**kwargs: Any) -> dict[int, int]:
        assert kwargs["top_k_concept_fids"] is raw
        return {5: 2, 2: 2}

    monkeypatch.setattr(
        intervention_module,
        "build_mode_aware_encoder_input",
        spy_builder,
    )
    _build(
        method_name="ours_extract_and_add_encdec",
        source_paths=SourcePaths(source_run_dir=tmp_path),
        model=_model(None),
    )


def test_malicious_sampler_oov_fails_at_intervention_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(AxBenchBpbError, match="producer invariant violation"):
        _build(
            method_name="neg_ours_remove_encdec",
            source_paths=_write_concept_artifacts(tmp_path, [3]),
            model=_model([3, 11]),
            inactive_sampler=lambda *, rng: {99: 4},
        )


def test_ours_requires_model_before_artifact_access(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="Every ours path requires"):
        _build(
            method_name="ours_extract_and_add_encdec",
            source_paths=SourcePaths(source_run_dir=tmp_path),
            model=None,
        )


@pytest.mark.parametrize(
    "method_name",
    [
        "control_encdec",
        "diffmean_encdec",
        "linear_probe_encdec",
        "lora_decoder_only",
        "prompting_encdec",
    ],
)
def test_non_ours_dispatch_keeps_model_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    monkeypatch.setattr(
        intervention_module,
        "_build_direction_intervention",
        lambda **_kwargs: Intervention.none(),
    )
    monkeypatch.setattr(
        intervention_module,
        "_build_lora_intervention",
        lambda **_kwargs: Intervention.none(),
    )
    monkeypatch.setattr(
        intervention_module,
        "_build_prompting_intervention",
        lambda **_kwargs: Intervention.none(),
    )
    build_intervention(
        method=get_method_by_name(method_name),
        concept=_concept(),
        coefficient=None,
        source_paths=SourcePaths(source_run_dir=tmp_path),
        diffmean_layer=8,
        target_model_meta=None,
        inactive_sampler=None,
        tokenizer_nanochat=None,
        rng=random.Random(0),
        model=None,
        apply_max_act_scale=False,
    )


def test_one_element_v_and_already_inside_v(tmp_path: Path) -> None:
    one = _build(
        method_name="ours_extract_and_add_encdec",
        source_paths=_write_concept_artifacts(tmp_path / "one", [7, 8]),
        model=_model([7]),
    )
    inside = _build(
        method_name="ours_extract_and_add_encdec",
        source_paths=_write_concept_artifacts(tmp_path / "inside", [3, 11, 3]),
        model=_model([3, 11]),
    )
    assert one.feature_conditions == {7: 2}
    assert inside.feature_conditions == {3: 2, 11: 2}


def test_compacted_path_rejects_out_of_sae_range_before_vocab_filter(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        AxBenchBpbError,
        match=r"concept feature artifact.*outside.*\[0, 16\).*16",
    ):
        _build(
            method_name="ours_extract_and_add_encdec",
            source_paths=_write_concept_artifacts(tmp_path, [16]),
            model=_model([3, 11]),
        )


@pytest.mark.parametrize(
    ("n_sae_features", "match"),
    [(None, "must be an integer"), (True, "must be an integer"), (0, "must be positive")],
)
def test_compacted_path_requires_valid_sae_width_before_vocab_filter(
    tmp_path: Path,
    n_sae_features: Any,
    match: str,
) -> None:
    with pytest.raises(AxBenchBpbError, match=match):
        _build(
            method_name="ours_extract_and_add_encdec",
            source_paths=_write_concept_artifacts(tmp_path, [3]),
            model=_model([3, 11]),
            target_model_meta=SimpleNamespace(n_sae_features=n_sae_features),
        )


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_compacted_candidate_tensor_is_constructed_on_allowed_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = intervention_module.restrict_to_vocab
    seen_devices: list[t.device] = []

    def spy(ids: t.Tensor, allowed: t.Tensor | None):
        seen_devices.append(ids.device)
        return original(ids, allowed)

    monkeypatch.setattr(intervention_module, "restrict_to_vocab", spy)
    _build(
        method_name="ours_extract_and_add_encdec",
        source_paths=_write_concept_artifacts(tmp_path, RAW_TOP_K),
        model=_model(ALLOWED, device="cuda"),
    )
    assert seen_devices
    assert all(device.type == "cuda" for device in seen_devices)
