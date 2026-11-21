from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch as t

from steering.eval.autointerp_filter.parity import (
    FP32_NATIVE_VS_TL_ATOL as _FP32_NATIVE_VS_TL_ATOL,
)
from steering.eval.autointerp_filter.parity import (
    compare_token_example_surface as _compare_token_example_surface,
)
from steering.eval.autointerp_filter.parity import (
    compare_trial_surfaces as _compare_trial_surfaces,
)
from steering.eval.autointerp_filter.parity import (
    load_token_example_set_by_target as _load_token_example_set_by_target,
)
from steering.eval.autointerp_filter.parity import (
    tensor_tolerance_report as _tensor_tolerance_report,
)
from steering.eval.autointerp_filter.token_examples import (
    _selected_sae_activations,
)
from steering.eval.autointerp_filter.types import TokenExampleSet, TokenWindowRef
from tests.steering.autointerp_filter._parity_assertions import (
    LATENT_NATIVE_VS_TL_ATOL as _LATENT_NATIVE_VS_TL_ATOL,
)
from tests.steering.autointerp_filter._parity_assertions import (
    SAE_NATIVE_VS_TL_ATOL as _SAE_NATIVE_VS_TL_ATOL,
)
from tests.steering.autointerp_filter._parity_assertions import (
    assert_bf16_surface_within_tolerance as _assert_bf16_surface_within_tolerance,
)
from tests.steering.autointerp_filter._parity_assertions import (
    assert_tensor_report_within as _assert_tensor_report_within,
)
from tests.steering.autointerp_filter._parity_assertions import (
    assert_token_example_surface_within_tolerance as _assert_token_example_surface_within_tolerance,
)
from tests.steering.autointerp_filter._parity_assertions import (
    load_single_token_example_set as _load_single_token_example_set,
)


class _FakeSae(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.W_enc = t.nn.Parameter(
            t.tensor(
                [
                    [0.60, -0.20, 0.10],
                    [0.10, 0.50, -0.30],
                    [-0.40, 0.20, 0.70],
                ],
                dtype=t.float32,
            ),
            requires_grad=False,
        )
        self.b_enc = t.nn.Parameter(
            t.tensor([0.05, -0.10, 0.20], dtype=t.float32),
            requires_grad=False,
        )
        self.threshold = None

    def process_sae_in(self, acts: t.Tensor) -> t.Tensor:
        return acts

    def activation_fn(self, hidden_pre: t.Tensor) -> t.Tensor:
        return t.relu(hidden_pre)


def _ref(
    *,
    doc_id: int = 1,
    token_start: int = 0,
    activations: tuple[float, ...] = (0.0, 10.0, 0.0, 2.0),
    max_activation: float | None = None,
    token_ids: tuple[int, ...] = (101, 102, 103, 104),
) -> TokenWindowRef:
    token_end = token_start + len(token_ids)
    if max_activation is None:
        max_activation = max(activations)
    return TokenWindowRef(
        shard_stem="shard_00000_features",
        shard_index=0,
        doc_id=doc_id,
        source_text_hash=f"hash-{doc_id}",
        token_start=token_start,
        token_end=token_end,
        token_ids=token_ids,
        token_strings=tuple(f"tok{idx}" for idx in token_ids),
        target_feature_ids=(0,),
        activations=activations,
        max_activation=float(max_activation),
        quantile=None,
        active=max_activation > 0,
    )


def _issue660_benchmark_root(repo_root: Path) -> Path:
    override = os.environ.get("ISSUE660_BENCHMARK_ROOT")
    if override:
        return Path(override)
    candidates = [repo_root / "out" / "benchmarks" / "issue660"]
    parts = repo_root.parts
    if ".codex" in parts:
        codex_index = parts.index(".codex")
        candidates.append(Path(*parts[:codex_index]) / "out" / "benchmarks" / "issue660")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _trial_fixture_examples(*, activation_delta: float = 0.0) -> TokenExampleSet:
    base_maxes = (12.0, 11.0, 10.0, 9.0, 8.0, 7.0)
    positives = tuple(
        _ref(
            doc_id=idx,
            token_start=idx * 10,
            activations=(0.0, max_value + activation_delta, 0.0, 1.0),
        )
        for idx, max_value in enumerate(base_maxes, start=1)
    )
    non_activating = tuple(
        _ref(
            doc_id=100 + idx,
            token_start=idx * 10,
            activations=(0.0, 0.0, 0.0, 0.0),
        )
        for idx in range(6)
    )
    return TokenExampleSet(positives=positives, non_activating=non_activating)


def test_latent_and_sae_activation_tolerance_accepts_small_cuda_drift() -> None:
    cpu_latents = t.tensor(
        [
            [0.20, -0.10, 0.40],
            [1.00, 0.30, -0.20],
            [-0.50, 0.80, 0.10],
        ],
        dtype=t.float32,
    )
    cuda_latents = cpu_latents + t.tensor(
        [
            [0.004, -0.002, 0.003],
            [-0.005, 0.004, -0.001],
            [0.002, 0.006, -0.004],
        ],
        dtype=t.float32,
    )
    latent_report = _tensor_tolerance_report(
        cpu_latents.flatten().tolist(),
        cuda_latents.flatten().tolist(),
        atol=_LATENT_NATIVE_VS_TL_ATOL,
    )
    _assert_tensor_report_within(latent_report, atol=_LATENT_NATIVE_VS_TL_ATOL)

    sae = _FakeSae()
    cpu_sae = _selected_sae_activations(
        sae=sae,
        acts=cpu_latents,
        feature_ids=(0, 1, 2),
    )
    cuda_sae = _selected_sae_activations(
        sae=sae,
        acts=cuda_latents,
        feature_ids=(0, 1, 2),
    )
    sae_report = _tensor_tolerance_report(
        cpu_sae.flatten().tolist(),
        cuda_sae.flatten().tolist(),
        atol=_SAE_NATIVE_VS_TL_ATOL,
    )
    _assert_tensor_report_within(sae_report, atol=_SAE_NATIVE_VS_TL_ATOL)


def test_latent_and_sae_activation_tolerance_rejects_large_cuda_drift() -> None:
    cpu_latents = t.tensor([[0.20, -0.10, 0.40]], dtype=t.float32)
    cuda_latents = t.tensor([[0.20, -0.10, 0.80]], dtype=t.float32)

    latent_report = _tensor_tolerance_report(
        cpu_latents.flatten().tolist(),
        cuda_latents.flatten().tolist(),
        atol=_LATENT_NATIVE_VS_TL_ATOL,
    )

    with pytest.raises(AssertionError):
        _assert_tensor_report_within(latent_report, atol=_LATENT_NATIVE_VS_TL_ATOL)


def test_token_example_cpu_cuda_tolerance_allows_small_fp32_drift() -> None:
    cpu = TokenExampleSet(
        positives=(_ref(activations=(0.0, 2392.3022, 0.0, 0.0)),),
        non_activating=(_ref(doc_id=2, activations=(0.0, 0.0, 0.0, 0.0)),),
    )
    cuda = TokenExampleSet(
        positives=(_ref(activations=(0.0, 2392.3030, 0.0, 0.0)),),
        non_activating=(_ref(doc_id=2, activations=(0.0, 0.0, 0.0, 0.0)),),
    )

    _assert_token_example_surface_within_tolerance(
        _compare_token_example_surface(cpu, cuda),
    )


def test_token_example_cpu_cuda_tolerance_reports_window_reordering() -> None:
    cpu = TokenExampleSet(
        positives=(_ref(doc_id=1), _ref(doc_id=2)),
        non_activating=(),
    )
    cuda = TokenExampleSet(
        positives=(_ref(doc_id=2), _ref(doc_id=1)),
        non_activating=(),
    )

    report = _compare_token_example_surface(cpu, cuda)

    assert report.positives.identity_set_equal
    assert report.positives.same_identity_order_count == 0
    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)


def test_token_example_cpu_cuda_tolerance_rejects_large_activation_drift() -> None:
    cpu = TokenExampleSet(
        positives=(_ref(activations=(0.0, 2392.3022, 0.0, 0.0)),),
        non_activating=(),
    )
    cuda = TokenExampleSet(
        positives=(_ref(activations=(0.0, 2392.60, 0.0, 0.0)),),
        non_activating=(),
    )

    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(
            _compare_token_example_surface(cpu, cuda),
        )


def test_highlight_surface_rejects_threshold_crossing_inside_activation_tolerance() -> None:
    cpu = TokenExampleSet(
        positives=(_ref(activations=(0.0, 10.0, 2.95, 0.0)),),
        non_activating=(),
    )
    cuda = TokenExampleSet(
        positives=(_ref(activations=(0.0, 10.0, 3.05, 0.0)),),
        non_activating=(),
    )

    report = _compare_token_example_surface(cpu, cuda)

    assert report.positives.activation.within_abs_tolerance
    assert report.positives.highlight_exact_count == 0
    assert report.positives.min_highlight_jaccard == 0.5
    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)


def test_trial_surface_report_separates_prompt_stability_from_trial_key_exactness() -> None:
    cpu = _trial_fixture_examples()
    cuda = _trial_fixture_examples(activation_delta=0.001)

    report = _compare_trial_surfaces(
        cpu,
        cuda,
        n_positive_examples=2,
        n_trials=4,
        n_quantiles=2,
    )

    assert report.reference_count == report.candidate_count == 4
    assert report.same_prompt_surface_count == 4
    assert report.same_intruder_window_count == 4
    assert report.same_trial_key_count == 0
    assert report.min_positive_window_jaccard == 1.0


@pytest.mark.requires_feature_data
def test_issue660_direct_cpu_tl_vs_native_cuda_f32_artifacts_match_full_surface_gate() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_root = _issue660_benchmark_root(repo_root)
    cpu_path = benchmark_root / "token_examples_one_target_reduced" / "token_example_cache.sqlite"
    cuda_path = (
        benchmark_root
        / "token_examples_one_target_reduced_native_cuda_f32"
        / "token_example_cache.sqlite"
    )
    if not cpu_path.exists() or not cuda_path.exists():
        pytest.skip("issue660 direct CPU/CUDA f32 token-example artifacts are not present")

    report = _compare_token_example_surface(
        _load_single_token_example_set(cpu_path),
        _load_single_token_example_set(cuda_path),
    )

    _assert_token_example_surface_within_tolerance(report)
    assert report.positives.same_stable_key_order_count < report.positives.reference_count


@pytest.mark.requires_feature_data
def test_issue660_source_index_cpu_tl_vs_native_cuda_f32_artifacts_match_tolerant_surfaces() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_root = _issue660_benchmark_root(repo_root)
    cpu_path = (
        benchmark_root
        / "token_examples_feature0_source_index_tl_cpu_pr675_20260710"
        / "token_example_cache.sqlite"
    )
    cuda_path = (
        benchmark_root
        / "token_examples_feature0_source_index_native_cuda_f32_pr675_20260710"
        / "token_example_cache.sqlite"
    )
    if not cpu_path.exists() or not cuda_path.exists():
        pytest.skip("issue660 source-index CPU/CUDA f32 token-example artifacts are not present")

    cpu = _load_single_token_example_set(cpu_path)
    cuda = _load_single_token_example_set(cuda_path)
    report = _compare_token_example_surface(cpu, cuda)

    _assert_token_example_surface_within_tolerance(
        report,
        positive_min_order_fraction=0.60,
    )
    assert report.positives.same_identity_order_count < report.positives.reference_count
    assert report.non_activating.same_identity_order_count == (
        report.non_activating.reference_count
    )

    trial_report = _compare_trial_surfaces(
        cpu,
        cuda,
        n_positive_examples=2,
        n_trials=8,
    )
    assert trial_report.reference_count == trial_report.candidate_count == 8
    assert trial_report.same_intruder_window_count == 8
    assert trial_report.same_prompt_surface_count < 8
    assert trial_report.same_trial_key_count == 0
    assert trial_report.mean_positive_window_jaccard > 0.0


@pytest.mark.requires_feature_data
def test_issue660_native_bf16_artifact_does_not_satisfy_f32_surface_gate() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_root = _issue660_benchmark_root(repo_root)
    cpu_path = benchmark_root / "token_examples_one_target_reduced" / "token_example_cache.sqlite"
    cuda_path = (
        benchmark_root
        / "token_examples_one_target_reduced_native_cuda"
        / "token_example_cache.sqlite"
    )
    if not cpu_path.exists() or not cuda_path.exists():
        pytest.skip("issue660 CPU/CUDA bf16 token-example artifacts are not present")

    report = _compare_token_example_surface(
        _load_single_token_example_set(cpu_path),
        _load_single_token_example_set(cuda_path),
    )

    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)


def test_bf16_gate_accepts_jumprelu_flip_in_matched_window() -> None:
    # bf16 drift near the SAE JumpReLU threshold gates a token from large to
    # EXACTLY 0.0 (a flip, per-token relative error 1.0) that no per-token
    # tolerance bounds. The window identity, matched set, and highlights are
    # unchanged, so the STRUCTURAL bf16 gate accepts it while the strict f32
    # gate (which bounds per-token activation drift) rejects it.
    cpu = TokenExampleSet(
        positives=(_ref(doc_id=1, activations=(0.0, 200.0, 50.0, 0.0)),),
        non_activating=(_ref(doc_id=2, activations=(0.0, 0.0, 0.0, 0.0)),),
    )
    bf16 = TokenExampleSet(
        positives=(_ref(doc_id=1, activations=(0.0, 200.0, 0.0, 0.0)),),  # 50.0 -> 0.0
        non_activating=(_ref(doc_id=2, activations=(0.0, 0.0, 0.0, 0.0)),),
    )
    report = _compare_token_example_surface(cpu, bf16)
    assert report.positives.activation.max_abs == 50.0  # the flip; unbounded by atol
    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)  # strict f32 gate
    _assert_bf16_surface_within_tolerance(cpu, bf16)  # structural bf16 gate


def test_bf16_gate_tolerates_bounded_identity_and_order_divergence() -> None:
    # 10 CPU positive windows; bf16 keeps 9 (with drift) and swaps 1 boundary
    # window for a different source doc -- exactly the hard-cap boundary flip the
    # bf16 contract accepts (matched 9/10 >= 0.80 floor). The f32 gate rejects it.
    cpu = TokenExampleSet(
        positives=tuple(
            _ref(doc_id=i, activations=(0.0, float(100 * i), 0.0, 0.0))
            for i in range(1, 11)
        ),
        non_activating=(),
    )
    bf16 = TokenExampleSet(
        positives=tuple(
            _ref(doc_id=i, activations=(0.0, float(104 * i), 0.0, 0.0))
            for i in range(1, 10)
        )
        + (_ref(doc_id=99, activations=(0.0, 950.0, 0.0, 0.0)),),
        non_activating=(),
    )
    report = _compare_token_example_surface(cpu, bf16)
    assert not report.positives.identity_set_equal
    assert report.positives.matched_window_count == 9
    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)  # strict f32 gate
    _assert_bf16_surface_within_tolerance(cpu, bf16)  # structural bf16 gate


def test_bf16_gate_rejects_structural_regressions() -> None:
    base = (_ref(doc_id=1, activations=(0.0, 1000.0, 0.0, 0.0)),)
    # (a) a non-activating window drifted off exactly zero violates the hard
    # bf16 invariant (non-activating windows are all-zero on both paths).
    with pytest.raises(AssertionError):
        _assert_bf16_surface_within_tolerance(
            TokenExampleSet(
                positives=base,
                non_activating=(_ref(doc_id=2, activations=(0.0, 0.0, 0.0, 0.0)),),
            ),
            TokenExampleSet(
                positives=base,
                non_activating=(_ref(doc_id=2, activations=(0.0, 0.01, 0.0, 0.0)),),
            ),
        )
    # (b) too many windows lost (matched fraction below the 0.80 floor): only
    # 1/5 windows survive -- a structural break the gate must reject.
    cpu_many = TokenExampleSet(
        positives=tuple(
            _ref(doc_id=i, activations=(0.0, float(100 * i), 0.0, 0.0))
            for i in range(1, 6)
        ),
        non_activating=(),
    )
    bf16_few = TokenExampleSet(
        positives=(_ref(doc_id=1, activations=(0.0, 104.0, 0.0, 0.0)),)
        + tuple(
            _ref(doc_id=90 + i, activations=(0.0, float(100 * i), 0.0, 0.0))
            for i in range(1, 5)
        ),
        non_activating=(),
    )
    with pytest.raises(AssertionError):
        _assert_bf16_surface_within_tolerance(cpu_many, bf16_few)


@pytest.mark.requires_feature_data
def test_issue660_native_bf16_artifact_satisfies_bf16_surface_gate() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_root = _issue660_benchmark_root(repo_root)
    cpu_path = benchmark_root / "token_examples_one_target_reduced" / "token_example_cache.sqlite"
    cuda_path = (
        benchmark_root
        / "token_examples_one_target_reduced_native_cuda"
        / "token_example_cache.sqlite"
    )
    if not cpu_path.exists() or not cuda_path.exists():
        pytest.skip("issue660 CPU/CUDA bf16 token-example artifacts are not present")

    cpu = _load_single_token_example_set(cpu_path)
    bf16 = _load_single_token_example_set(cuda_path)
    report = _compare_token_example_surface(cpu, bf16)

    # The reduced artifact fails the strict f32 gate but satisfies the bf16
    # envelope, and the accepted activation drift is far above the f32 atol.
    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)
    _assert_bf16_surface_within_tolerance(cpu, bf16)
    assert report.positives.activation.max_abs > _FP32_NATIVE_VS_TL_ATOL


def _issue678_benchmark_root() -> Path:
    override = os.environ.get("ISSUE678_BENCHMARK_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "out" / "benchmarks" / "issue678"


@pytest.mark.requires_feature_data
def test_issue678_m25_native_bf16_satisfies_bf16_surface_gate() -> None:
    # The 25-target production-shaped run exercises the corpus behavior the bf16
    # floors are derived from: feature:0 has a JumpReLU flip (activation ~176 ->
    # 0.0) that bumps one window out of the top-16 (matched 15/16, identity set
    # NOT equal), so the strict f32 gate fails but the structural bf16 gate holds.
    root = _issue678_benchmark_root()
    tl_path = root / "m25_tl_cpu" / "token_example_cache.sqlite"
    bf16_path = root / "m25_native_bf16" / "token_example_cache.sqlite"
    if not tl_path.exists() or not bf16_path.exists():
        pytest.skip("issue678 m25 CPU/bf16 token-example caches are not present")

    cpu = _load_token_example_set_by_target(tl_path, "feature:0")
    bf16 = _load_token_example_set_by_target(bf16_path, "feature:0")
    report = _compare_token_example_surface(cpu, bf16)

    assert not report.positives.identity_set_equal  # a window was bumped by the flip
    assert report.positives.activation.max_abs > 100.0  # the JumpReLU flip magnitude
    with pytest.raises(AssertionError):
        _assert_token_example_surface_within_tolerance(report)
    _assert_bf16_surface_within_tolerance(cpu, bf16)
