"""Post-load SAE identity checks for the AxBench-Base runtime."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch as t

from steering.eval.axbench_base import runtime_state as runtime_state_module
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    ResolvedSAE,
    SAERelease,
    SAEWidth,
)


def _make_runtime_state(
    make_valid_axbench_cfg: Any,
    *,
    release: SAERelease = SAERelease.gemmascope_2,
) -> RuntimeState:
    if release is SAERelease.gemmascope_2:
        cfg = make_valid_axbench_cfg(
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_source="22-gemmascope-2-res-16k",
            device="cpu",
        )
    else:
        cfg = make_valid_axbench_cfg(
            neuronpedia_model_id="gemma-2-2b",
            feature_extraction_model_id=None,
            neuronpedia_source="8-gemmascope-res-16k",
            device="cpu",
        )
    return RuntimeState(cfg=cfg)


def _install_identity_load(
    monkeypatch: pytest.MonkeyPatch,
    sae: object,
    *,
    release: SAERelease = SAERelease.gemmascope_2,
) -> tuple[ResolvedSAE, list[dict[str, Any]], MagicMock]:
    if release is SAERelease.gemmascope_2:
        layer = 22
        l0: L0Tier | None = L0Tier.medium
        resolved = ResolvedSAE(
            release="gemma-scope-2-4b-pt-res",
            sae_id="layer_22_width_16k_l0_medium",
            hook_name="blocks.22.hook_resid_post",
        )
        expected_neuronpedia = (
            "gemma-3-4b",
            "22-gemmascope-2-res-16k",
        )
    else:
        layer = 8
        l0 = None
        resolved = ResolvedSAE(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_8/width_16k/canonical",
            hook_name="blocks.8.hook_resid_post",
        )
        expected_neuronpedia = ("gemma-2-2b", "8-gemmascope-res-16k")

    model = SimpleNamespace()
    load_calls: list[dict[str, Any]] = []
    build_nnterp_model = MagicMock(return_value=None)

    monkeypatch.setattr(
        runtime_state_module,
        "parse_neuronpedia_source",
        lambda _source: (
            release,
            ActivationType.residual,
            layer,
            SAEWidth.W_16K,
            l0,
        ),
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_sae",
        lambda **_kwargs: resolved,
    )
    monkeypatch.setattr(
        runtime_state_module,
        "resolve_neuronpedia_ids",
        lambda **_kwargs: expected_neuronpedia,
    )

    def _load_sae_and_model(**kwargs: Any) -> tuple[object, object, object]:
        load_calls.append(kwargs)
        return model, sae, object()

    monkeypatch.setattr(
        runtime_state_module,
        "load_sae_and_model",
        _load_sae_and_model,
    )
    monkeypatch.setattr(
        runtime_state_module,
        "build_nnterp_model",
        build_nnterp_model,
    )
    return resolved, load_calls, build_nnterp_model


def _assert_extraction_resources_uncached(state: RuntimeState) -> None:
    assert state._feat_extraction_loaded is False
    assert state._base_model is None
    assert state._base_sae is None


def _assert_wrapped_value_error(
    exc_info: pytest.ExceptionInfo[AxBenchBaseError],
    match: str,
) -> None:
    assert "Failed to load feature-extraction model" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert re.search(match, str(exc_info.value.__cause__))


def test_unclustered_runtime_sae_rejects_wrong_tier_via_sae_id(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loaded SAE's reported sae_id must match the resolved L0 tier."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    wrong_tier_sae = SimpleNamespace(
        cfg=SimpleNamespace(sae_id="layer_22_width_16k_l0_small"),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        wrong_tier_sae,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        state._ensure_feature_extraction_loaded()

    _assert_wrapped_value_error(exc_info, "sae_id.*wrong L0 tier")
    assert len(load_calls) == 1
    build_nnterp_model.assert_not_called()
    _assert_extraction_resources_uncached(state)


def test_unclustered_runtime_sae_rejects_mismatched_release(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loaded SAE's reported release must match the resolver output."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    wrong_release_sae = SimpleNamespace(
        cfg=SimpleNamespace(release="wrong-release"),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        wrong_release_sae,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        state._ensure_feature_extraction_loaded()

    _assert_wrapped_value_error(exc_info, "cfg.release.*does not match")
    assert len(load_calls) == 1
    build_nnterp_model.assert_not_called()
    _assert_extraction_resources_uncached(state)


def test_unclustered_runtime_sae_rejects_bare_model_name_mismatch(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare normalization must still reject genuinely different model ids."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    wrong_model_sae = SimpleNamespace(
        cfg=SimpleNamespace(model_name="google/gemma-3-12b-pt"),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        wrong_model_sae,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        state._ensure_feature_extraction_loaded()

    _assert_wrapped_value_error(exc_info, "model_name.*does not match")
    assert len(load_calls) == 1
    build_nnterp_model.assert_not_called()
    _assert_extraction_resources_uncached(state)


def test_unclustered_runtime_sae_accepts_org_prefix_only_model_name_difference(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAELens' forced ``google/`` prefix is not an identity mismatch."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    sae = SimpleNamespace(
        cfg=SimpleNamespace(
            model_name="google/gemma-3-4b-pt",
            sae_id="layer_22_width_16k_l0_medium",
        ),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        sae,
    )

    state._ensure_feature_extraction_loaded()

    assert len(load_calls) == 1
    build_nnterp_model.assert_called_once()
    assert state._feat_extraction_loaded is True
    assert state._base_sae is sae


def test_unclustered_runtime_sae_rejects_mismatched_neuronpedia_id(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neuronpedia identity independently catches a substituted SAE."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    wrong_neuronpedia_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            metadata={
                "neuronpedia_id": (
                    "gemma-3-4b/22-gemmascope-2-res-16k-WRONG"
                ),
            },
        ),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        wrong_neuronpedia_sae,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        state._ensure_feature_extraction_loaded()

    _assert_wrapped_value_error(exc_info, "neuronpedia_id.*does not match")
    assert len(load_calls) == 1
    build_nnterp_model.assert_not_called()
    _assert_extraction_resources_uncached(state)


def test_unclustered_v2_without_tier_identity_fails_closed(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v2 SAE with only non-tier identity fields cannot enter runtime state."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release="gemma-scope-2-4b-pt-res",
            model_name="google/gemma-3-4b-pt",
            metadata={"model_name": "google/gemma-3-4b-pt"},
        ),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        sae,
    )

    with pytest.raises(AxBenchBaseError) as exc_info:
        state._ensure_feature_extraction_loaded()

    _assert_wrapped_value_error(exc_info, "cannot prove.*L0 tier")
    assert len(load_calls) == 1
    build_nnterp_model.assert_not_called()
    _assert_extraction_resources_uncached(state)


def test_unclustered_runtime_sae_accepts_fully_populated_matching_identity(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All matching reported identity fields are accepted and cached."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    matching_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            release="gemma-scope-2-4b-pt-res",
            model_name="google/gemma-3-4b-pt",
            sae_id="layer_22_width_16k_l0_medium",
            metadata={
                "model_name": "google/gemma-3-4b-pt",
                "sae_id": "layer_22_width_16k_l0_medium",
                "neuronpedia_id": "gemma-3-4b/22-gemmascope-2-res-16k",
            },
        ),
    )
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        matching_sae,
    )

    state._ensure_feature_extraction_loaded()

    assert len(load_calls) == 1
    build_nnterp_model.assert_called_once()
    assert state._feat_extraction_loaded is True
    assert state._base_sae is matching_sae


def test_unclustered_v1_without_identity_is_accepted(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v1 SAE reporting no identity metadata remains tolerated."""
    state = _make_runtime_state(
        make_valid_axbench_cfg,
        release=SAERelease.gemmascope,
    )
    sae = SimpleNamespace(cfg=SimpleNamespace())
    _resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        sae,
        release=SAERelease.gemmascope,
    )

    state._ensure_feature_extraction_loaded()

    assert len(load_calls) == 1
    build_nnterp_model.assert_called_once()
    assert state._feat_extraction_loaded is True
    assert state._base_sae is sae


def test_clustered_shape_error_precedes_identity_error_and_caching(
    make_valid_axbench_cfg: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cluster shape/hook validation retains precedence over identity checks."""
    state = _make_runtime_state(make_valid_axbench_cfg)
    wrong_shape_and_identity_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            d_sae=16383,
            hook_name="blocks.22.hook_resid_post",
            sae_id="layer_22_width_16k_l0_small",
        ),
        W_dec=t.empty((16383, 1)),
    )
    resolved, load_calls, build_nnterp_model = _install_identity_load(
        monkeypatch,
        wrong_shape_and_identity_sae,
    )
    contract: Any = SimpleNamespace(
        n_sae_features=16384,
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
    state.cluster_contract = contract

    with pytest.raises(AxBenchBaseError) as exc_info:
        state._ensure_feature_extraction_loaded()

    _assert_wrapped_value_error(
        exc_info,
        "does not match the cluster feature contract",
    )
    assert "sae_id" not in str(exc_info.value.__cause__)
    assert len(load_calls) == 1
    build_nnterp_model.assert_not_called()
    _assert_extraction_resources_uncached(state)
