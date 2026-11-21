from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.sampling import (
    parse_negative_selection,
    sample_k,
)
from steering.eval.axbench_base import runtime_state as runtime_state_module
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import AxBenchBaseError, TargetModelMetadata

N_SAE = 16
NEGATIVE_SELECTION = {
    "distribution": "geometric",
    "mean": 4.0,
    "min_k": 0,
    "max_k": 8,
}


def _meta(
    *,
    n_sae_features: int | None = N_SAE,
    negative_selection: dict[str, Any] = NEGATIVE_SELECTION,
) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=5,
        n_sae_features=n_sae_features,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


@pytest.fixture(autouse=True)
def _stats_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_state_module,
        "read_n_sae_features_from_stats",
        lambda _path: N_SAE,
    )


def _cfg() -> AxBenchBaseConfig:
    return cast(
        AxBenchBaseConfig,
        SimpleNamespace(feature_statistics_path="/unused/stats.safetensors"),
    )


def _direct_uniform_oracle(
    *,
    allowed: t.Tensor,
    seed: int,
    negative_selection: dict[str, Any] = NEGATIVE_SELECTION,
) -> tuple[dict[int, int], object]:
    python_rng = random.Random(seed)
    torch_seed = python_rng.randint(
        0,
        runtime_state_module._TORCH_SEED_UPPER_BOUND,
    )
    torch_rng = t.Generator(device="cpu")
    torch_rng.manual_seed(torch_seed)
    selection = parse_negative_selection(negative_selection)
    assert selection is not None
    k = sample_k(
        selection.distribution,
        selection.mean,
        selection.std,
        torch_rng,
        min_k=selection.min_k,
        max_k=selection.max_k,
    )
    k = min(k, allowed.numel())
    if k <= 0:
        sampled = allowed[:0]
    else:
        permutation = t.randperm(allowed.numel(), generator=torch_rng)
        sampled = allowed[permutation[:k]]
    return {int(fid): 4 for fid in sampled.tolist()}, python_rng.getstate()


@st.composite
def _valid_sampler_cases(
    draw: Any,
) -> tuple[t.Tensor, dict[str, Any], int]:
    allowed = sorted(
        draw(
            st.sets(
                st.integers(min_value=0, max_value=N_SAE - 1),
                min_size=1,
                max_size=N_SAE - 1,
            )
        )
    )
    min_k = draw(st.integers(min_value=0, max_value=4))
    max_k = draw(
        st.one_of(
            st.none(),
            st.integers(min_value=min_k + 1, max_value=N_SAE + 4),
        )
    )
    mean = draw(st.integers(min_value=1, max_value=10))
    distribution = draw(st.sampled_from(["geometric", "normal"]))
    selection: dict[str, Any] = {
        "distribution": distribution,
        "mean": float(mean),
        "min_k": min_k,
        "max_k": max_k,
    }
    if distribution == "normal":
        selection["std"] = draw(
            st.floats(
                min_value=0.1,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
    seed = draw(st.integers(min_value=0, max_value=2**63 - 1))
    return t.tensor(allowed, dtype=t.int64), selection, seed


@settings(max_examples=40, deadline=None)
@given(case=_valid_sampler_cases())
def test_masked_branch_property_matches_uniform_without_replacement_oracle(
    case: tuple[t.Tensor, dict[str, Any], int],
) -> None:
    allowed, selection, seed = case
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(negative_selection=selection),
        cfg=_cfg(),
        allowed_feature_ids=allowed,
    )
    rng = random.Random(seed)

    actual = sampler(rng=rng)
    expected, expected_state = _direct_uniform_oracle(
        allowed=allowed,
        seed=seed,
        negative_selection=selection,
    )

    assert actual == expected
    assert rng.getstate() == expected_state
    assert set(actual).issubset(set(allowed.tolist()))


@pytest.mark.parametrize("seed", list(range(12)))
def test_masked_branch_matches_uniform_without_replacement_oracle_and_rng_bridge(
    seed: int,
) -> None:
    allowed = t.tensor([1, 3, 4, 9, 12], dtype=t.int64)
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(),
        cfg=_cfg(),
        allowed_feature_ids=allowed,
    )
    rng = random.Random(seed)

    actual = sampler(rng=rng)
    expected, expected_state = _direct_uniform_oracle(
        allowed=allowed,
        seed=seed,
    )

    assert actual == expected
    assert rng.getstate() == expected_state
    assert set(actual).issubset(set(allowed.tolist()))
    assert set(actual.values()).issubset({4})


def test_one_id_v_never_returns_another_id_and_k_clamps() -> None:
    selection = {
        "distribution": "normal",
        "mean": 100.0,
        "std": 0.001,
    }
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(negative_selection=selection),
        cfg=_cfg(),
        allowed_feature_ids=t.tensor([7], dtype=t.int64),
    )
    assert sampler(rng=random.Random(0)) == {7: 4}


def test_empty_draw_returns_empty_dict() -> None:
    selection = {
        "distribution": "normal",
        "mean": 0.0,
        "std": 0.001,
        "min_k": 0,
    }
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(negative_selection=selection),
        cfg=_cfg(),
        allowed_feature_ids=t.tensor([2, 5], dtype=t.int64),
    )
    assert sampler(rng=random.Random(0)) == {}


def test_masked_sampler_is_deterministic_and_varies_across_a_seed_matrix() -> None:
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(),
        cfg=_cfg(),
        allowed_feature_ids=t.tensor([1, 3, 4, 9, 12], dtype=t.int64),
    )
    outputs = [tuple(sampler(rng=random.Random(seed)).items()) for seed in range(20)]
    assert sampler(rng=random.Random(123)) == sampler(rng=random.Random(123))
    assert len(set(outputs)) > 1


def test_builder_captures_independent_cpu_snapshot() -> None:
    allowed = t.tensor([1, 5, 11], dtype=t.int64)
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(),
        cfg=_cfg(),
        allowed_feature_ids=allowed,
    )
    before = sampler(rng=random.Random(123))
    allowed[:] = t.tensor([2, 6, 12], dtype=t.int64)
    after = sampler(rng=random.Random(123))
    assert after == before


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_cuda_allowed_vocabulary_matches_cpu_uniform_oracle() -> None:
    allowed_cpu = t.tensor([1, 5, 11], dtype=t.int64)
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(),
        cfg=_cfg(),
        allowed_feature_ids=allowed_cpu.cuda(),
    )
    rng = random.Random(17)

    actual = sampler(rng=rng)
    expected, expected_state = _direct_uniform_oracle(
        allowed=allowed_cpu,
        seed=17,
    )

    assert actual == expected
    assert rng.getstate() == expected_state


def test_legacy_branch_keeps_historical_sampler_call_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def historical_sampler(
        *,
        active_feature_ids: t.Tensor,
        n_sae_features: int,
        config: Any,
        rng: t.Generator,
    ) -> t.Tensor:
        del config
        calls.append((active_feature_ids.numel(), n_sae_features))
        return t.tensor([2, 8], dtype=t.int64)

    monkeypatch.setattr(
        runtime_state_module,
        "sample_inactive_features",
        historical_sampler,
    )
    sampler = runtime_state_module._build_inactive_sampler(
        encdec_meta=_meta(),
        cfg=_cfg(),
        allowed_feature_ids=None,
    )
    assert sampler(rng=random.Random(4)) == {2: 4, 8: 4}
    assert calls == [(0, N_SAE)]


@pytest.mark.parametrize(
    ("allowed", "match"),
    [
        (t.tensor([1, 2], dtype=t.int32), "dtype torch.int64"),
        (t.tensor([[1, 2]], dtype=t.int64), "one-dimensional"),
        (t.tensor([], dtype=t.int64), "non-empty"),
        (t.tensor([1, 1], dtype=t.int64), "strictly increasing"),
        (t.tensor([3, 2], dtype=t.int64), "strictly increasing"),
        (t.tensor([-1, 2], dtype=t.int64), "non-negative"),
        (t.tensor([1, N_SAE], dtype=t.int64), "less than"),
        (t.arange(N_SAE, dtype=t.int64), "proper subset"),
    ],
)
def test_invalid_allowed_vocabulary_fails_at_construction(
    allowed: t.Tensor,
    match: str,
) -> None:
    with pytest.raises(AxBenchBaseError, match=match):
        runtime_state_module._build_inactive_sampler(
            encdec_meta=_meta(),
            cfg=_cfg(),
            allowed_feature_ids=allowed,
        )


def test_missing_metadata_width_fails_at_construction() -> None:
    with pytest.raises(AxBenchBaseError, match="n_sae_features is None"):
        runtime_state_module._build_inactive_sampler(
            encdec_meta=_meta(n_sae_features=None),
            cfg=_cfg(),
            allowed_feature_ids=t.tensor([1, 2], dtype=t.int64),
        )


def test_metadata_statistics_width_mismatch_fails_at_construction() -> None:
    with pytest.raises(AxBenchBaseError, match="does not match"):
        runtime_state_module._build_inactive_sampler(
            encdec_meta=_meta(n_sae_features=N_SAE + 1),
            cfg=_cfg(),
            allowed_feature_ids=t.tensor([1, 2], dtype=t.int64),
        )
