"""Real-GPU coverage for the upstream AxBench encoder-decoder baselines.

The unit and driver suites use tiny models and stubs.  This module exercises
the corresponding production seams with the l22 checkpoint, Gemma-3-4B-PT
and its real SAE, the prepared c2 training data, and the accepted c2 auxiliary
Parquet.  Runtime is bounded deliberately: direction fitting uses the first
eight rows from each class, and the short LoRA fit uses the first eight
positive rows for four epochs.  The full driver consumes every selected row
and uses the production 24-epoch LoRA schedule; the project numerical gate
likewise fits its dense direction from every selected training row.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch as t

from nanochat.tokenizer import get_tokenizer
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS
from steering.conditional.encoder_decoder.packed_encoder import (
    stack_packed_encoder_outputs,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SamplingBundle,
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
)
from steering.eval.axbench_base.baselines.hooks import (
    _get_block,
    register_batched_residual_hook,
)
from steering.eval.axbench_upstream.data import ConceptTrainingPairs, select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines import run as encdec_driver
from steering.eval.axbench_upstream.encdec_baselines.capture import (
    FitRowConditioning,
    build_fit_row_conditioning,
    capture_post_prefix_activations,
)
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.generation import (
    DirectionGenerationRequest,
    generate_encdec_baseline,
    steering_delta,
)
from steering.eval.axbench_upstream.encdec_baselines.layer_resolution import (
    VENDORED_LORA_LAYERS,
    VENDORED_STEERING_LAYER,
    resolve_vendored_layer,
    resolve_vendored_layers,
)
from steering.eval.axbench_upstream.encdec_baselines.lora import (
    LoRATrainingRow,
    RsLoRAAdapter,
    train_encdec_lora,
)
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    QUESTION_MARKER,
    QWEN_STEERED_PREFIX,
    QWEN_STEERED_SUFFIX,
    recover_concept_steering_prompts,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    render_fit_row,
    template_prefix_length,
)
from steering.eval.axbench_upstream.methods import (
    SELF_CONDITIONING_MASKED_POLICY,
    MethodSetup,
    PreparedPromptContext,
    SelfConditioningGenerationRequest,
    build_self_conditioning_config,
    fit_dense_diffmean,
    load_extraction_stack,
    load_method_setup,
    load_target_model,
    prepare_prompt_context,
)
from steering.eval.axbench_upstream.prompt_prep import RenderedPromptPlan
from steering.eval.axbench_upstream.run_methods import (
    _project_score_policy,
    derive_generation_seed,
    derive_selection_seed,
)
from steering.eval.axbench_upstream.work_grid import (
    ENCDEC_MAIN_METHODS,
    PROJECT_METHODS,
    build_encdec_cell_grid,
    build_project_cell_grid,
    owned_cells,
    owned_project_cells,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.inference_engine import (
    InferenceEngine,
    SelfConditionedPrefill,
    build_engine_self_conditioned,
    stack_prefilled_kv_caches,
)
from steering.eval.target_conditioning import (
    SingleEntryContextProvider,
    validate_feature_dict,
)

DEFAULT_MAIN_REPO_ROOT = Path("/home/d0rb/Documents/Github/nano-gpt-pretrain-steer")
MAIN_REPO_ROOT_ENV = "NANO_GPT_PRETRAIN_STEER_MAIN_REPO"


def _main_repo_root() -> Path:
    """Resolve real artifacts outside this worktree without copying them."""

    configured_root = os.environ.get(MAIN_REPO_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    nanochat_base = os.environ.get("NANOCHAT_BASE_DIR")
    if nanochat_base:
        resolved_base = Path(nanochat_base).expanduser().resolve()
        if resolved_base.name == "nanochat_cache":
            return resolved_base.parent
    return DEFAULT_MAIN_REPO_ROOT


MAIN_REPO_ROOT = _main_repo_root()
NANOCHAT_BASE_DIR = (
    Path(os.environ.get("NANOCHAT_BASE_DIR", MAIN_REPO_ROOT / "nanochat_cache"))
    .expanduser()
    .resolve()
)

# These are the real paths declared by configs/eval/axbench_upstream/methods.yaml.
TARGET_META_PATH = NANOCHAT_BASE_DIR / (
    "conditional_encdec_checkpoints/l22_main_h100_8rank_db16_v1/s128/"
    "dist_geometric/mean_12.0/std_12.0/wl_9172c540310ba842/"
    "fc_5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039/"
    "meta_054600.json"
)
FEATURE_STATISTICS_PATH = MAIN_REPO_ROOT / (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/"
    "context_independent/no_bos/feature_statistics.safetensors"
)
PREPARED_DIR = MAIN_REPO_ROOT / "out/axbench/prepared"
AUX_PARQUET_PATH = MAIN_REPO_ROOT / (
    "out/axbench/runs/05f03dede-c2-retry-20260811/aux/inference/steering_data.parquet"
)

DEVICE = "cuda"
BASE_SEED = 42
PREFIX_LENGTH = 2
HIDDEN_SIZE = 768
FIT_ROWS_PER_CLASS = 8
MAX_NEW_TOKENS = 128


def _uncentered_relative_rmse_and_cosine(
    actual: t.Tensor,
    expected: t.Tensor,
) -> tuple[float, float]:
    """Compare absolute tensor state without erasing a constant corruption."""
    left = actual.detach().to(device="cpu", dtype=t.float64).flatten()
    right = expected.detach().to(device="cpu", dtype=t.float64).flatten()
    assert left.shape == right.shape
    assert bool(t.isfinite(left).all().item())
    assert bool(t.isfinite(right).all().item())
    tiny = t.finfo(t.float64).tiny
    relative_rmse = float(
        (left - right).square().mean().sqrt() / right.square().mean().sqrt().clamp_min(tiny)
    )
    cosine = (
        1.0
        if t.equal(left, right)
        else float(t.dot(left, right) / (left.norm() * right.norm()).clamp_min(tiny))
    )
    return relative_rmse, cosine


def test_uncentered_state_gate_rejects_large_additive_corruption() -> None:
    expected = t.linspace(-3.0, 5.0, 257, dtype=t.float64)
    exact_rmse, exact_cosine = _uncentered_relative_rmse_and_cosine(expected, expected)
    zero_rmse, zero_cosine = _uncentered_relative_rmse_and_cosine(
        t.zeros_like(expected),
        t.zeros_like(expected),
    )
    shifted_rmse, shifted_cosine = _uncentered_relative_rmse_and_cosine(
        expected + 1_000.0,
        expected,
    )

    assert exact_rmse == 0.0
    assert exact_cosine == pytest.approx(1.0)
    assert zero_rmse == 0.0
    assert zero_cosine == 1.0
    assert shifted_rmse > 0.10
    assert shifted_cosine < 0.995


@dataclass(frozen=True)
class _GemmaStack:
    setup: MethodSetup
    model: Any
    sae: Any
    tokenizer: Any
    training: ConceptTrainingPairs
    aux_frame: pd.DataFrame
    allocated_mib: float


@dataclass(frozen=True)
class _TargetStack:
    model: Any
    tokenizer: Any
    allocated_mib: float


@dataclass(frozen=True)
class _RealCase:
    positive_rows: tuple[LoRATrainingRow, ...]
    negative_rows: tuple[LoRATrainingRow, ...]
    eval_prompt: str
    prepared_prompt: PreparedPromptContext


@dataclass(frozen=True)
class _CapturedFits:
    positive: tuple[t.Tensor, ...]
    negative: tuple[t.Tensor, ...]
    diffmean: FittedDirection
    probe: FittedDirection


@dataclass
class _TeacherForcedMetrics:
    """Every scalar-versus-batch value required by the full-size gates."""

    lm_relative_rmse: list[float] = field(default_factory=list)
    lm_correlation: list[float] = field(default_factory=list)
    lm_top50_overlap: list[int] = field(default_factory=list)
    lm_total_variation: list[float] = field(default_factory=list)
    fp_relative_rmse: list[float] = field(default_factory=list)
    fp_cosine: list[float] = field(default_factory=list)
    kv_relative_rmse: list[float] = field(default_factory=list)
    kv_cosine: list[float] = field(default_factory=list)

    @staticmethod
    def _centered_similarity(actual: t.Tensor, expected: t.Tensor) -> tuple[float, float]:
        left = actual.detach().to(device="cpu", dtype=t.float64).flatten()
        right = expected.detach().to(device="cpu", dtype=t.float64).flatten()
        assert left.shape == right.shape
        assert bool(t.isfinite(left).all().item())
        assert bool(t.isfinite(right).all().item())
        left = left - left.mean()
        right = right - right.mean()
        denominator = right.square().mean().sqrt().clamp_min(t.finfo(t.float64).tiny)
        relative_rmse = float((left - right).square().mean().sqrt() / denominator)
        correlation = float(
            t.dot(left, right) / (left.norm() * right.norm()).clamp_min(t.finfo(t.float64).tiny)
        )
        return relative_rmse, correlation

    def record_lm(self, actual: t.Tensor, expected: t.Tensor) -> None:
        relative_rmse, correlation = self._centered_similarity(actual, expected)
        self.lm_relative_rmse.append(relative_rmse)
        self.lm_correlation.append(correlation)
        actual_vector = actual.detach().to(device="cpu", dtype=t.float64).flatten()
        expected_vector = expected.detach().to(device="cpu", dtype=t.float64).flatten()
        actual_top50 = set(t.topk(actual_vector, 50).indices.tolist())
        expected_top50 = set(t.topk(expected_vector, 50).indices.tolist())
        self.lm_top50_overlap.append(len(actual_top50 & expected_top50))
        actual_distribution = t.softmax(actual_vector, dim=0)
        expected_distribution = t.softmax(expected_vector, dim=0)
        self.lm_total_variation.append(
            float(0.5 * (actual_distribution - expected_distribution).abs().sum())
        )

    def record_fp(self, actual: t.Tensor, expected: t.Tensor) -> None:
        relative_rmse, cosine = _uncentered_relative_rmse_and_cosine(actual, expected)
        self.fp_relative_rmse.append(relative_rmse)
        self.fp_cosine.append(cosine)

    def record_kv(self, actual: t.Tensor, expected: t.Tensor) -> None:
        """Record K/V state separately for every decoder layer."""

        assert actual.size(0) == expected.size(0)
        for layer in range(actual.size(0)):
            relative_rmse, cosine = _uncentered_relative_rmse_and_cosine(
                actual[layer], expected[layer]
            )
            self.kv_relative_rmse.append(relative_rmse)
            self.kv_cosine.append(cosine)

    def assert_acceptance(self, *, label: str, rows: int, steps: int, layers: int) -> None:
        comparisons = rows * steps
        assert len(self.lm_relative_rmse) == comparisons
        assert len(self.lm_correlation) == comparisons
        assert len(self.lm_top50_overlap) == comparisons
        assert len(self.lm_total_variation) == comparisons
        assert len(self.fp_relative_rmse) == comparisons
        assert len(self.fp_cosine) == comparisons
        assert len(self.kv_relative_rmse) == 2 * comparisons * layers
        assert len(self.kv_cosine) == 2 * comparisons * layers
        print(
            f"{label}: "
            f"lm_rel_rmse_max={max(self.lm_relative_rmse):.6f} "
            f"lm_corr_min={min(self.lm_correlation):.6f} "
            f"lm_top50_overlap_min={min(self.lm_top50_overlap)} "
            f"lm_tv_max={max(self.lm_total_variation):.6f} "
            f"fp_rel_rmse_max={max(self.fp_relative_rmse):.6f} "
            f"fp_cosine_min={min(self.fp_cosine):.6f} "
            f"kv_rel_rmse_max={max(self.kv_relative_rmse):.6f} "
            f"kv_cosine_min={min(self.kv_cosine):.6f}"
        )
        assert max(self.lm_relative_rmse) <= 0.10
        assert min(self.lm_correlation) >= 0.995
        assert min(self.lm_top50_overlap) >= 40
        assert max(self.lm_total_variation) <= 0.15
        assert max(self.fp_relative_rmse) <= 0.10
        assert min(self.fp_cosine) >= 0.995
        assert max(self.kv_relative_rmse) <= 0.10
        assert min(self.kv_cosine) >= 0.995


def test_teacher_forced_metrics_preserve_absolute_fp_and_per_layer_kv_state() -> None:
    expected = t.linspace(-3.0, 5.0, 257, dtype=t.float64)
    shifted = expected + 1_000.0
    metrics = _TeacherForcedMetrics()

    metrics.record_fp(shifted, expected)
    metrics.record_kv(t.stack((expected, shifted)), t.stack((expected, expected)))

    assert len(metrics.fp_relative_rmse) == len(metrics.fp_cosine) == 1
    assert metrics.fp_relative_rmse[0] > 0.10
    assert metrics.fp_cosine[0] < 0.995
    assert len(metrics.kv_relative_rmse) == len(metrics.kv_cosine) == 2
    assert metrics.kv_relative_rmse[0] == 0.0
    assert metrics.kv_cosine[0] == 1.0
    assert metrics.kv_relative_rmse[1] > 0.10
    assert metrics.kv_cosine[1] < 0.995


def _require_real_assets() -> None:
    if not t.cuda.is_available():
        pytest.skip(
            "torch.cuda.is_available() is False: the realistic AxBench encdec "
            "suite requires the real Gemma and l22 models on CUDA"
        )
    for label, path, expected_kind in (
        ("l22 checkpoint metadata", TARGET_META_PATH, "file"),
        ("feature statistics", FEATURE_STATISTICS_PATH, "file"),
        ("prepared AxBench directory", PREPARED_DIR, "dir"),
        ("c2 auxiliary Parquet", AUX_PARQUET_PATH, "file"),
    ):
        exists = path.is_file() if expected_kind == "file" else path.is_dir()
        if not exists:
            pytest.skip(f"missing real asset ({label}): {path}")


def _missing_local_hf_weights(exc: BaseException) -> bool:
    """Recognize cache-miss/download failures without hiding model defects."""

    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    combined = "\n".join(messages).lower()
    cache_miss_markers = (
        "localentrynotfounderror",
        "cannot find the requested files in the local cache",
        "couldn't connect to",
        "could not connect to",
        "offline mode",
        "local_files_only",
    )
    return any(marker in combined for marker in cache_miss_markers)


def _allocated_mib() -> float:
    t.cuda.synchronize()
    return t.cuda.memory_allocated() / (1024**2)


@pytest.fixture(scope="module")
def real_gemma_stack() -> Iterator[_GemmaStack]:
    """Load the real method setup plus Gemma extraction stack exactly once."""

    _require_real_assets()
    # This historical checkpoint predates content_sha256 and records the raw
    # safetensors-file digest, whose metadata ordering was nondeterministic.
    # The exact statistics path and both observed digests are pinned by the
    # campaign specification; make the required operator exception explicit.
    setup = load_method_setup(
        TARGET_META_PATH,
        FEATURE_STATISTICS_PATH,
        require_exact_statistics_identity=False,
    )
    try:
        model, sae = load_extraction_stack(setup, DEVICE)
    except BaseException as exc:
        if _missing_local_hf_weights(exc):
            pytest.skip(
                "Gemma-3-4B-PT/SAE weights are not loadable from the local Hugging "
                f"Face cache and a download failed: {type(exc).__name__}: {exc}"
            )
        raise

    tokenizer = get_tokenizer()
    training_by_concept = select_training_pairs(PREPARED_DIR, max_concepts=1)
    training = training_by_concept[0]
    aux_frame = pd.read_parquet(AUX_PARQUET_PATH)
    assert setup.sae_model == "gemma-3-4b-pt"
    assert setup.sae_layer == 22
    assert setup.n_sae_features == 65_536
    assert len(training.positive) >= FIT_ROWS_PER_CLASS
    assert len(training.negative) >= FIT_ROWS_PER_CLASS
    assert not aux_frame.empty
    assert {"concept_id", "input_id", "original_prompt", "steered_input"} <= set(aux_frame.columns)
    sae_parameters = list(sae.parameters())
    assert sae_parameters
    assert {parameter.device.type for parameter in sae_parameters} == {"cuda"}
    assert {parameter.dtype for parameter in sae_parameters} == {t.bfloat16}
    nnterp_module = getattr(model.nnterp_model, "model", model.nnterp_model)
    nnterp_parameters = list(nnterp_module.parameters())
    assert nnterp_parameters
    assert {parameter.device.type for parameter in nnterp_parameters} == {"cuda"}
    assert {parameter.dtype for parameter in nnterp_parameters} == {t.bfloat16}
    allocated_mib = _allocated_mib()
    print(f"real encdec GPU fixture: Gemma+SAE allocated={allocated_mib:.1f} MiB")

    loaded = _GemmaStack(
        setup=setup,
        model=model,
        sae=sae,
        tokenizer=tokenizer,
        training=training,
        aux_frame=aux_frame,
        allocated_mib=allocated_mib,
    )
    yield loaded

    if hasattr(model, "nnterp_model"):
        del model.nnterp_model
    del loaded, model, sae
    gc.collect()
    t.cuda.empty_cache()


@pytest.fixture(scope="module")
def real_target_stack(real_gemma_stack: _GemmaStack) -> Iterator[_TargetStack]:
    """Load the real l22 target once while the measured Gemma stack remains resident."""

    model, tokenizer = load_target_model(real_gemma_stack.setup, DEVICE)
    model.eval()
    parameters = list(model.parameters())
    assert parameters
    assert {parameter.device.type for parameter in parameters} == {"cuda"}
    assert all(bool(t.isfinite(parameter.detach()).all().item()) for parameter in parameters)
    assert int(model.dec_config.sequence_len) == _sequence_len(real_gemma_stack.setup)
    allocated_mib = _allocated_mib()
    incremental_mib = allocated_mib - real_gemma_stack.allocated_mib
    print(
        "real encdec GPU fixture: Gemma+SAE+target "
        f"allocated={allocated_mib:.1f} MiB (target increment={incremental_mib:.1f} MiB)"
    )
    loaded = _TargetStack(model=model, tokenizer=tokenizer, allocated_mib=allocated_mib)
    yield loaded

    del loaded, model
    gc.collect()
    t.cuda.empty_cache()


def _sequence_len(setup: MethodSetup) -> int:
    model_config = setup.meta.get("model_config")
    sequence_len = model_config.get("sequence_len") if isinstance(model_config, dict) else None
    assert isinstance(sequence_len, int) and not isinstance(sequence_len, bool)
    assert sequence_len > 0
    return sequence_len


def _n_layer(setup: MethodSetup) -> int:
    model_config = setup.meta.get("model_config")
    n_layer = model_config.get("n_layer") if isinstance(model_config, dict) else None
    assert isinstance(n_layer, int) and not isinstance(n_layer, bool)
    assert n_layer > 0
    return n_layer


def _intervention_layer(setup: MethodSetup) -> int:
    return resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=_n_layer(setup))


def _lora_layers(setup: MethodSetup) -> tuple[int, ...]:
    return resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=_n_layer(setup))


def _build_row(
    stack: _GemmaStack,
    pair: tuple[str, str],
    *,
    class_name: str,
    row_index: int,
) -> LoRATrainingRow:
    surface = f"real concept 0 {class_name} row {row_index}"
    rendered = render_fit_row(
        stack.tokenizer,
        input_text=pair[0],
        output_text=pair[1],
        sequence_len=_sequence_len(stack.setup),
        surface=surface,
    )
    conditioning = build_fit_row_conditioning(
        stack.setup,
        stack.model,
        stack.sae,
        stack.tokenizer,
        token_ids=rendered.ids,
        surface=surface,
    )
    assert rendered.ids == conditioning.token_ids
    return LoRATrainingRow(rendered=rendered, conditioning=conditioning)


@pytest.fixture(scope="module")
def real_case(real_gemma_stack: _GemmaStack) -> _RealCase:
    """Extract the bounded fit rows and one real eval prompt once."""

    positive_rows = tuple(
        _build_row(real_gemma_stack, pair, class_name="positive", row_index=row_index)
        for row_index, pair in enumerate(real_gemma_stack.training.positive[:FIT_ROWS_PER_CLASS])
    )
    negative_rows = tuple(
        _build_row(real_gemma_stack, pair, class_name="negative", row_index=row_index)
        for row_index, pair in enumerate(real_gemma_stack.training.negative[:FIT_ROWS_PER_CLASS])
    )
    eval_rows = real_gemma_stack.aux_frame.loc[
        real_gemma_stack.aux_frame["concept_id"].eq(0)
        & real_gemma_stack.aux_frame["input_id"].eq(0)
    ]
    assert len(eval_rows) == 1
    eval_prompt = str(eval_rows.iloc[0]["original_prompt"])
    rendered_prompt = render_prompt_for_chat_completion(real_gemma_stack.tokenizer, eval_prompt)
    prepared_prompt = prepare_prompt_context(
        real_gemma_stack.setup,
        real_gemma_stack.model,
        real_gemma_stack.sae,
        real_gemma_stack.tokenizer,
        token_ids=rendered_prompt.token_ids,
        assistant_end_id=rendered_prompt.assistant_end_id,
    )
    assert tuple(rendered_prompt.token_ids) == prepared_prompt.token_ids
    assert prepared_prompt.segments
    return _RealCase(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        eval_prompt=eval_prompt,
        prepared_prompt=prepared_prompt,
    )


@pytest.fixture(scope="module")
def captured_fits(
    real_gemma_stack: _GemmaStack,
    real_target_stack: _TargetStack,
    real_case: _RealCase,
) -> _CapturedFits:
    """Capture each bounded row once and share both deterministic fits."""

    intervention_layer = _intervention_layer(real_gemma_stack.setup)
    positive = tuple(
        capture_post_prefix_activations(
            real_target_stack.model,
            row.conditioning,
            layer=intervention_layer,
            prefix_length=PREFIX_LENGTH,
        )
        for row in real_case.positive_rows
    )
    negative = tuple(
        capture_post_prefix_activations(
            real_target_stack.model,
            row.conditioning,
            layer=intervention_layer,
            prefix_length=PREFIX_LENGTH,
        )
        for row in real_case.negative_rows
    )
    diffmean = fit_diffmean_direction(
        positive,
        negative,
        intervention_layer=intervention_layer,
        surface="real concept 0 EncDecDiffMean",
    )
    probe = fit_probe_direction(
        positive,
        negative,
        intervention_layer=intervention_layer,
        seed=BASE_SEED,
        concept_id=0,
        surface="real concept 0 EncDecLinearProbe",
    )
    return _CapturedFits(
        positive=positive,
        negative=negative,
        diffmean=diffmean,
        probe=probe,
    )


def _assert_conditioning_contract(
    conditioning: FitRowConditioning,
    setup: MethodSetup,
) -> None:
    assert conditioning.segments
    assert len(conditioning.segment_assignments) == len(conditioning.token_ids)
    assignments = conditioning.segment_assignments
    assert assignments == tuple(sorted(assignments))
    assert set(assignments) == set(range(len(conditioning.segments)))
    assert assignments[0] == 0
    assert assignments[-1] == len(conditioning.segments) - 1
    for segment_index, segment in enumerate(conditioning.segments):
        assert segment.feature_ids
        assert len(segment.feature_ids) == len(segment.buckets)
        selected = {
            int(feature_id): int(bucket)
            for feature_id, bucket in zip(
                segment.feature_ids,
                segment.buckets,
                strict=True,
            )
        }
        assert len(selected) == len(segment.feature_ids)
        validate_feature_dict(
            selected,
            setup.n_sae_features,
            2,
            name=f"real fit row segment {segment_index}",
            feature_vocab=setup.feature_vocab,
        )


def _assert_direction(fitted: FittedDirection) -> float:
    assert fitted.direction.shape == (HIDDEN_SIZE,)
    assert fitted.direction.dtype == t.float32
    assert fitted.direction.device.type == "cpu"
    assert bool(t.isfinite(fitted.direction).all().item())
    assert math.isfinite(fitted.max_activation)
    assert fitted.max_activation > 0.0
    return float(fitted.direction.norm().item())


def _generate(
    gemma: _GemmaStack,
    target: _TargetStack,
    prepared: PreparedPromptContext,
    *,
    seed: int,
) -> str:
    assert prepared.token_ids
    assert prepared.segments
    assert len(prepared.token_ids) + MAX_NEW_TOKENS <= int(target.model.dec_config.sequence_len)
    return generate_encdec_baseline(
        gemma.setup,
        target.model,
        target.tokenizer,
        prepared,
        selection_seed=derive_selection_seed(BASE_SEED, 0, 0),
        seed=seed,
        max_new_tokens=MAX_NEW_TOKENS,
    )


@pytest.mark.slow
def test_real_fit_row_conditioning_contract(
    real_gemma_stack: _GemmaStack,
    real_case: _RealCase,
) -> None:
    """The first real concept-0 positive row obeys every conditioning contract."""

    first = real_case.positive_rows[0]
    assert first.rendered.ids == first.conditioning.token_ids
    assert len(first.rendered.ids) <= _sequence_len(real_gemma_stack.setup)
    assert template_prefix_length(real_gemma_stack.tokenizer) == PREFIX_LENGTH
    _assert_conditioning_contract(first.conditioning, real_gemma_stack.setup)


@pytest.mark.slow
def test_real_capture_and_fits_concept0(
    real_gemma_stack: _GemmaStack,
    real_case: _RealCase,
    captured_fits: _CapturedFits,
) -> None:
    """Eight rows per class give finite real captures and deterministic fits."""

    assert len(captured_fits.positive) == FIT_ROWS_PER_CLASS
    assert len(captured_fits.negative) == FIT_ROWS_PER_CLASS
    for rows, activations in (
        (real_case.positive_rows, captured_fits.positive),
        (real_case.negative_rows, captured_fits.negative),
    ):
        for row, activation in zip(rows, activations, strict=True):
            assert activation.shape == (len(row.rendered.ids) - PREFIX_LENGTH, HIDDEN_SIZE)
            assert activation.dtype == t.float32
            assert activation.device.type == "cuda"
            assert bool(t.isfinite(activation).all().item())

    diffmean_norm = _assert_direction(captured_fits.diffmean)
    probe_norm = _assert_direction(captured_fits.probe)
    assert abs(diffmean_norm - 1.0) <= 1e-6
    assert abs(probe_norm - 1.0) <= 1e-6
    assert captured_fits.probe.max_activation > 0.0

    intervention_layer = _intervention_layer(real_gemma_stack.setup)
    diffmean_refit = fit_diffmean_direction(
        captured_fits.positive,
        captured_fits.negative,
        intervention_layer=intervention_layer,
        surface="real concept 0 EncDecDiffMean deterministic refit",
    )
    probe_refit = fit_probe_direction(
        captured_fits.positive,
        captured_fits.negative,
        intervention_layer=intervention_layer,
        seed=BASE_SEED,
        concept_id=0,
        surface="real concept 0 EncDecLinearProbe deterministic refit",
    )
    assert (
        diffmean_refit.direction.contiguous().numpy().tobytes()
        == captured_fits.diffmean.direction.contiguous().numpy().tobytes()
    )
    assert (
        probe_refit.direction.contiguous().numpy().tobytes()
        == captured_fits.probe.direction.contiguous().numpy().tobytes()
    )
    assert diffmean_refit.max_activation == captured_fits.diffmean.max_activation
    assert probe_refit.max_activation == captured_fits.probe.max_activation


@pytest.mark.slow
def test_real_generation_factor0_equals_control(
    real_gemma_stack: _GemmaStack,
    real_target_stack: _TargetStack,
    real_case: _RealCase,
    captured_fits: _CapturedFits,
) -> None:
    """A zero hook is exact, while the real factor-6 intervention changes text."""

    intervention_layer = _intervention_layer(real_gemma_stack.setup)
    seed = derive_generation_seed(BASE_SEED, "EncDecDiffMean", 0, 0, 0.0)
    control = _generate(
        real_gemma_stack,
        real_target_stack,
        real_case.prepared_prompt,
        seed=seed,
    )
    with steering_delta(
        real_target_stack.model,
        direction=captured_fits.diffmean.direction,
        coefficient=0.0,
        layer=intervention_layer,
    ):
        factor_zero = _generate(
            real_gemma_stack,
            real_target_stack,
            real_case.prepared_prompt,
            seed=seed,
        )
    assert factor_zero == control

    coefficient = 6.0 * captured_fits.diffmean.max_activation
    assert math.isfinite(coefficient) and coefficient > 0.0
    with steering_delta(
        real_target_stack.model,
        direction=captured_fits.diffmean.direction,
        coefficient=coefficient,
        layer=intervention_layer,
    ):
        factor_six = _generate(
            real_gemma_stack,
            real_target_stack,
            real_case.prepared_prompt,
            seed=seed,
        )
    assert factor_six != control, (
        "factor-6 real generation unexpectedly equaled control: "
        f"control_chars={len(control)}, factor6_chars={len(factor_six)}"
    )
    assert factor_six.strip(), (
        f"factor-6 generation was blank: control_chars={len(control)}, factor6_chars=0"
    )


@pytest.mark.slow
@t.inference_mode()
def test_real_b24_direction_decode_matches_scalar_numerically(
    monkeypatch: pytest.MonkeyPatch,
    real_gemma_stack: _GemmaStack,
    real_target_stack: _TargetStack,
    real_case: _RealCase,
    captured_fits: _CapturedFits,
) -> None:
    """Gate the production B=24 teacher-forced LM/FP/KV state versus B=1."""

    intervention_layer = _intervention_layer(real_gemma_stack.setup)
    device = next(real_target_stack.model.parameters()).device
    device_index = t.cuda.current_device() if device.index is None else device.index
    device_name = t.cuda.get_device_name(device_index)
    assert "H200" in device_name, (
        f"production B=24 numerical gate requires H200, got {device_name!r}"
    )

    source_frame = pd.DataFrame(
        [{"concept_id": 0, "input_id": 0, "factor": factor} for factor in FULL_FACTORS]
    )
    source_rows = tuple(source_frame.itertuples(index=False))
    direction_cells = tuple(
        cell
        for cell in build_encdec_cell_grid(source_frame, seed=BASE_SEED)
        if cell.method in ENCDEC_MAIN_METHODS
    )
    expected_identities = tuple(
        (source_row_position, method, float(factor).hex())
        for source_row_position, factor in enumerate(FULL_FACTORS)
        for method in ENCDEC_MAIN_METHODS
    )
    assert (
        tuple((cell.source_row_position, cell.method, cell.factor_hex) for cell in direction_cells)
        == expected_identities
    )
    assert len(direction_cells) == 2 * len(FULL_FACTORS) == 24
    assert {(cell.concept_id, cell.input_id) for cell in direction_cells} == {(0, 0)}

    fits = {
        "EncDecDiffMean": captured_fits.diffmean,
        "EncDecLinearProbe": captured_fits.probe,
    }
    captured_request_groups: list[tuple[DirectionGenerationRequest, ...]] = []

    def capture_requests(
        setup: MethodSetup,
        model: Any,
        tokenizer: Any,
        prepared: PreparedPromptContext,
        request_rows: Sequence[DirectionGenerationRequest],
        *,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        assert setup is real_gemma_stack.setup
        assert model is real_target_stack.model
        assert tokenizer is real_target_stack.tokenizer
        assert prepared is real_case.prepared_prompt
        assert max_new_tokens == MAX_NEW_TOKENS
        captured = tuple(request_rows)
        assert all(isinstance(request, DirectionGenerationRequest) for request in captured)
        captured_request_groups.append(captured)
        return tuple("captured" for _request in captured)

    prompt_tokens = real_case.prepared_prompt.token_ids
    prompt_plan = RenderedPromptPlan(
        rows=source_rows,
        rendered_tokens_by_pair={(0, 0): prompt_tokens},
        rendered_prompt_by_pair={
            (0, 0): (real_case.eval_prompt, real_case.prepared_prompt.assistant_end_id)
        },
        prompt_by_tokens={
            prompt_tokens: (real_case.eval_prompt, real_case.prepared_prompt.assistant_end_id)
        },
    )
    monkeypatch.setattr(encdec_driver, "generate_encdec_direction_batch", capture_requests)
    captured_generations = encdec_driver._generate_encdec_direction_cells(
        direction_cells,
        rows=source_rows,
        prompt_plan=prompt_plan,
        prepared_by_tokens={prompt_tokens: real_case.prepared_prompt},
        fitted={0: {"diffmean": captured_fits.diffmean, "probe": captured_fits.probe}},
        intervention_layer=intervention_layer,
        setup=real_gemma_stack.setup,
        model=real_target_stack.model,
        tokenizer=real_target_stack.tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert captured_generations == ("captured",) * len(direction_cells)
    assert len(captured_request_groups) == 1
    requests = captured_request_groups[0]
    assert len(requests) == len(direction_cells)
    expected_selection_seed = derive_selection_seed(BASE_SEED, 0, 0)
    assert {request.selection_seed for request in requests} == {expected_selection_seed}
    assert len({request.token_seed for request in requests}) == len(requests)
    for cell, request in zip(direction_cells, requests, strict=True):
        source_row: Any = source_rows[cell.source_row_position]
        factor = float(source_row.factor)
        fit = fits[cell.method]
        assert (int(source_row.concept_id), int(source_row.input_id), factor.hex()) == (
            cell.concept_id,
            cell.input_id,
            cell.factor_hex,
        )
        assert request.direction is fit.direction
        assert request.coefficient == factor * fit.max_activation
        assert request.layer == intervention_layer
        assert request.selection_seed == cell.selection_seed == expected_selection_seed
        assert (
            request.token_seed
            == cell.token_seed
            == derive_generation_seed(
                BASE_SEED,
                cell.method,
                cell.concept_id,
                cell.input_id,
                factor,
            )
        )
    direction_snapshots = {
        method: (fit.direction.clone(), fit.max_activation) for method, fit in fits.items()
    }

    def prepare_rows(
        row_indices: tuple[int, ...],
    ) -> list[tuple[InferenceEngine, SelfConditionedPrefill]]:
        prepared_rows = []
        selection_rng_start_states: list[t.Tensor] = []
        for row in row_indices:
            request = requests[row]
            config = build_self_conditioning_config(
                real_gemma_stack.setup,
                real_target_stack.model,
                None,
                selection_seed=request.selection_seed,
            )
            selection: Any = config.selection
            assert selection.rng.initial_seed() == request.selection_seed
            selection_rng_start_states.append(selection.rng.get_state().clone())
            provider = SingleEntryContextProvider(
                real_case.prepared_prompt.token_ids,
                real_case.prepared_prompt.segments,
            )
            engine = build_engine_self_conditioned(
                real_target_stack.model,
                real_target_stack.tokenizer,
                next(real_target_stack.model.parameters()).device,
                config,
                context_provider=provider,
            )
            with steering_delta(
                real_target_stack.model,
                direction=request.direction,
                coefficient=request.coefficient,
                layer=request.layer,
            ):
                prefill = engine.prepare_self_conditioned_prefill(
                    list(real_case.prepared_prompt.token_ids),
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            prepared_rows.append((engine, prefill))
        assert all(
            t.equal(state, selection_rng_start_states[0])
            for state in selection_rng_start_states[1:]
        )
        return prepared_rows

    def assert_independent_prefills(
        prepared_rows: Sequence[tuple[InferenceEngine, SelfConditionedPrefill]],
        row_indices: tuple[int, ...],
    ) -> None:
        assert len(prepared_rows) == len(row_indices)
        runtimes = [engine._model._self_cond for engine, _prefill in prepared_rows]
        assert all(runtime is not None for runtime in runtimes)
        assert len({id(engine) for engine, _prefill in prepared_rows}) == len(row_indices)
        assert len({id(engine._model) for engine, _prefill in prepared_rows}) == len(row_indices)
        assert len({id(runtime) for runtime in runtimes}) == len(row_indices)
        assert len({id(runtime.config) for runtime in runtimes}) == len(row_indices)
        assert len({id(runtime.provider) for runtime in runtimes}) == len(row_indices)
        selection_rngs = [runtime.config.selection.rng for runtime in runtimes]
        assert len({id(rng) for rng in selection_rngs}) == len(row_indices)
        assert {rng.initial_seed() for rng in selection_rngs} == {expected_selection_seed}
        assert len({id(prefill) for _engine, prefill in prepared_rows}) == len(row_indices)
        assert len({id(prefill.kv_cache) for _engine, prefill in prepared_rows}) == len(row_indices)
        assert len(
            {prefill.kv_cache.k_cache.data_ptr() for _engine, prefill in prepared_rows}
        ) == len(row_indices)
        assert len(
            {prefill.kv_cache.v_cache.data_ptr() for _engine, prefill in prepared_rows}
        ) == len(row_indices)
        assert len(
            {
                prefill.runtime.state.encoder_out.values.data_ptr()
                for _engine, prefill in prepared_rows
            }
        ) == len(row_indices)
        assert all(prefill.kv_cache.batch_size == 1 for _engine, prefill in prepared_rows)
        assert all(
            prefill.runtime.state.encoder_out.layout.batch_size == 1
            for _engine, prefill in prepared_rows
        )
        assert all(
            prefill.runtime.tokens_in_seg == 0 and not prefill.runtime.fp_buffer
            for _engine, prefill in prepared_rows
        )

    decode_steps = 16
    forced_rows = tuple(
        tuple(prompt_tokens[(step + row * 7) % len(prompt_tokens)] for step in range(decode_steps))
        for row in range(len(requests))
    )
    scalar_forwards = 0

    def capture_scalar_rows(
        row_indices: tuple[int, ...],
    ) -> tuple[
        dict[int, list[t.Tensor]],
        dict[int, list[t.Tensor]],
        dict[int, list[t.Tensor]],
        dict[int, list[t.Tensor]],
    ]:
        nonlocal scalar_forwards
        scalar_logits = {row: [] for row in row_indices}
        scalar_fp_logits = {row: [] for row in row_indices}
        scalar_k_caches = {row: [] for row in row_indices}
        scalar_v_caches = {row: [] for row in row_indices}
        for row in row_indices:
            request = requests[row]
            # Scalar production completes one entire cell before constructing
            # the next. Preserve that compiler history instead of front-loading
            # all prefills as the batched candidate does.
            engine, prefill = prepare_rows((row,))[0]
            with steering_delta(
                real_target_stack.model,
                direction=request.direction,
                coefficient=request.coefficient,
                layer=request.layer,
            ):
                for token in forced_rows[row]:
                    logits = engine._model(
                        t.tensor([[token]], dtype=t.long, device=device),
                        kv_cache=prefill.kv_cache,
                    )
                    scalar_forwards += 1
                    runtime = engine._model.self_cond_runtime_snapshot()
                    scalar_logits[row].append(logits.detach().cpu())
                    scalar_fp_logits[row].append(runtime.fp_buffer[-1].detach().cpu())
                    position = int(prefill.kv_cache.get_pos())
                    scalar_k_caches[row].append(
                        prefill.kv_cache.k_cache[:, :, :position].detach().cpu()
                    )
                    scalar_v_caches[row].append(
                        prefill.kv_cache.v_cache[:, :, :position].detach().cpu()
                    )
        return (
            scalar_logits,
            scalar_fp_logits,
            scalar_k_caches,
            scalar_v_caches,
        )

    batch_groups = (("w1-b24", tuple(range(len(requests)))),)
    assert [len(indices) for _label, indices in batch_groups] == [24]
    metrics = _TeacherForcedMetrics()
    batched_forwards = 0
    for _label, row_indices in batch_groups:
        # Each reference and candidate models a separate fresh production
        # process owning exactly these rows. The reference interleaves each
        # scalar prefill/decode; the candidate performs all scalar prefills and
        # then the one bounded batched decode.
        t.compiler.reset()
        (
            scalar_logits,
            scalar_fp_logits,
            scalar_k_caches,
            scalar_v_caches,
        ) = capture_scalar_rows(row_indices)
        t.compiler.reset()
        selected_rows = prepare_rows(row_indices)
        assert_independent_prefills(selected_rows, row_indices)
        selected_prefills = tuple(prefill for _engine, prefill in selected_rows)
        assert len(selected_prefills) == len(row_indices)
        source_cache_snapshots = tuple(
            (
                int(prefill.kv_cache.get_pos()),
                prefill.logits.detach().cpu().clone(),
                prefill.kv_cache.k_cache[:, :, : int(prefill.kv_cache.get_pos())]
                .detach()
                .cpu()
                .clone(),
                prefill.kv_cache.v_cache[:, :, : int(prefill.kv_cache.get_pos())]
                .detach()
                .cpu()
                .clone(),
            )
            for prefill in selected_prefills
        )
        source_encoder_snapshots = tuple(
            (
                prefill.runtime.state.encoder_out.values.detach().cpu().clone(),
                prefill.runtime.state.encoder_out.layout.flat_indices.detach().cpu().clone(),
                prefill.runtime.state.encoder_out.layout.valid_mask.detach().cpu().clone(),
                prefill.runtime.state.encoder_out.layout.batch_indices.detach().cpu().clone(),
                prefill.runtime.state.encoder_out.layout.segment_indices.detach().cpu().clone(),
            )
            for prefill in selected_prefills
        )
        source_position = source_cache_snapshots[0][0]
        assert {snapshot[0] for snapshot in source_cache_snapshots} == {source_position}
        batch_cache = stack_prefilled_kv_caches(
            tuple(prefill.kv_cache for prefill in selected_prefills)
        )
        batch_encoder = stack_packed_encoder_outputs(
            tuple(prefill.runtime.state.encoder_out for prefill in selected_prefills)
        )
        seg_map = t.tensor(
            [[prefill.runtime.state.target_slot_idx] for prefill in selected_prefills],
            dtype=t.long,
            device=device,
        )
        assert batch_cache.batch_size == len(row_indices)
        assert int(batch_cache.get_pos()) == source_position
        assert batch_encoder.layout.batch_size == len(row_indices)
        assert batch_encoder.values.data_ptr() not in {
            prefill.runtime.state.encoder_out.values.data_ptr() for prefill in selected_prefills
        }
        assert tuple(seg_map.shape) == (len(row_indices), 1)
        assert seg_map.flatten().tolist() == [
            prefill.runtime.state.target_slot_idx for prefill in selected_prefills
        ]
        for row, (prefill, cache_snapshot) in enumerate(
            zip(selected_prefills, source_cache_snapshots, strict=True)
        ):
            position, _logits, expected_k, expected_v = cache_snapshot
            assert t.equal(batch_cache.k_cache[:, row : row + 1, :position].cpu(), expected_k)
            assert t.equal(batch_cache.v_cache[:, row : row + 1, :position].cpu(), expected_v)
            source_encoder = prefill.runtime.state.encoder_out
            source_layout = source_encoder.layout
            source_valid = source_layout.valid_mask
            batch_layout = batch_encoder.layout
            batch_valid = batch_layout.valid_mask & batch_layout.batch_indices.eq(row)
            assert int(batch_valid.sum().item()) == int(source_valid.sum().item())
            assert t.equal(
                batch_encoder.values[0, batch_valid],
                source_encoder.values[0, source_valid],
            )
            assert t.equal(
                batch_layout.segment_indices[batch_valid],
                source_layout.segment_indices[source_valid],
            )
            assert t.equal(
                batch_layout.flat_indices[batch_valid] % batch_layout.feature_count,
                source_layout.flat_indices[source_valid] % source_layout.feature_count,
            )
        handle = register_batched_residual_hook(
            real_target_stack.model,
            intervention_layer,
            t.stack([requests[row].direction for row in row_indices]),
            [requests[row].coefficient for row in row_indices],
        )
        try:
            for step in range(decode_steps):
                emitted = t.tensor(
                    [[forced_rows[row][step]] for row in row_indices],
                    dtype=t.long,
                    device=device,
                )
                batched_logits, batched_fp_logits = real_target_stack.model(
                    emitted,
                    None,
                    seg_map=seg_map,
                    kv_cache=batch_cache,
                    encoder_out=batch_encoder,
                    encoder_strict=True,
                    return_fp_logits=True,
                )
                batched_forwards += 1
                position = int(batch_cache.get_pos())
                expected_position = source_position + step + 1
                assert position == expected_position
                assert batch_cache.cache_seqlens.tolist() == [expected_position] * len(row_indices)
                assert batched_logits.size(0) == len(row_indices)
                assert batched_fp_logits.size(0) == len(row_indices)
                for physical_row, logical_row in enumerate(row_indices):
                    actual_lm = batched_logits[physical_row : physical_row + 1]
                    expected_lm = scalar_logits[logical_row][step]
                    metrics.record_lm(actual_lm, expected_lm)
                    metrics.record_fp(
                        batched_fp_logits[physical_row : physical_row + 1],
                        scalar_fp_logits[logical_row][step],
                    )
                    for actual_cache, expected_cache in (
                        (
                            batch_cache.k_cache[:, physical_row : physical_row + 1, :position],
                            scalar_k_caches[logical_row][step],
                        ),
                        (
                            batch_cache.v_cache[:, physical_row : physical_row + 1, :position],
                            scalar_v_caches[logical_row][step],
                        ),
                    ):
                        metrics.record_kv(actual_cache, expected_cache)
        finally:
            handle.remove()

    for prefill, cache_snapshot, encoder_snapshot in zip(
        selected_prefills,
        source_cache_snapshots,
        source_encoder_snapshots,
        strict=True,
    ):
        position, logits_snapshot, k_snapshot, v_snapshot = cache_snapshot
        assert int(prefill.kv_cache.get_pos()) == position
        assert t.equal(prefill.logits.cpu(), logits_snapshot)
        assert t.equal(prefill.kv_cache.k_cache[:, :, :position].cpu(), k_snapshot)
        assert t.equal(prefill.kv_cache.v_cache[:, :, :position].cpu(), v_snapshot)
        encoder_out = prefill.runtime.state.encoder_out
        layout = encoder_out.layout
        values, flat_indices, valid_mask, batch_indices, segment_indices = encoder_snapshot
        assert t.equal(encoder_out.values.cpu(), values)
        assert t.equal(layout.flat_indices.cpu(), flat_indices)
        assert t.equal(layout.valid_mask.cpu(), valid_mask)
        assert t.equal(layout.batch_indices.cpu(), batch_indices)
        assert t.equal(layout.segment_indices.cpu(), segment_indices)
        assert prefill.runtime.tokens_in_seg == 0
        assert not prefill.runtime.fp_buffer
    for method, fit in fits.items():
        direction_snapshot, max_activation_snapshot = direction_snapshots[method]
        assert t.equal(fit.direction, direction_snapshot)
        assert fit.max_activation == max_activation_snapshot

    expected_states = decode_steps * sum(len(indices) for _label, indices in batch_groups)
    assert expected_states == 384
    assert scalar_forwards == expected_states
    assert batched_forwards == decode_steps == 16
    layers = int(real_target_stack.model.dec_config.n_layer)
    assert expected_states * 2 * layers == 9_216
    metrics.assert_acceptance(
        label="real H200 B=24 batched direction numerical gate",
        rows=sum(len(indices) for _label, indices in batch_groups),
        steps=decode_steps,
        layers=layers,
    )


@pytest.mark.slow
@t.inference_mode()
def test_real_project_b24_and_w4_b6_decode_matches_scalar_numerically_on_h200(
    real_gemma_stack: _GemmaStack,
    real_target_stack: _TargetStack,
    real_case: _RealCase,
) -> None:
    """Gate candidate W4/B24 and every legacy W4/B6 project geometry."""

    device = next(real_target_stack.model.parameters()).device
    properties = t.cuda.get_device_properties(device)
    assert "H200" in properties.name
    assert properties.total_memory >= 135 * 1024**3
    assert real_gemma_stack.setup.sae_model == "gemma-3-4b-pt"
    assert real_gemma_stack.setup.sae_layer == 22
    assert real_gemma_stack.setup.n_sae_features == 65_536
    assert _n_layer(real_gemma_stack.setup) == 12
    layers = int(real_target_stack.model.dec_config.n_layer)
    assert layers == 12
    assert HIDDEN_SIZE == int(real_target_stack.model.dec_config.n_embd)

    positive_texts = tuple(
        input_text + output_text for input_text, output_text in real_gemma_stack.training.positive
    )
    negative_texts = tuple(
        input_text + output_text for input_text, output_text in real_gemma_stack.training.negative
    )
    dense_diffmean = fit_dense_diffmean(
        real_gemma_stack.setup,
        real_gemma_stack.model,
        real_gemma_stack.sae,
        positive_texts,
        negative_texts,
        concept_id=0,
        extraction_batch_size=8,
    )
    feature_vocab = real_gemma_stack.setup.feature_vocab
    assert feature_vocab is not None
    assert feature_vocab.n_sae_features == real_gemma_stack.setup.n_sae_features == 65_536
    assert feature_vocab.fingerprint() == (
        "8c183369ce4a8ff1e5f153a749e9eb71f9def65283d0a22b4d33bf5ff2d6f16c"
    )
    assert real_target_stack.model.feature_vocab == feature_vocab
    assert (
        real_target_stack.model.feature_vocab_size
        == real_target_stack.model.feature_prediction_width
        == feature_vocab.size
        == 2_074
    )
    allowed_ids = real_gemma_stack.setup.allowed_ids
    assert allowed_ids is not None
    assert t.equal(allowed_ids, feature_vocab.new_to_old_tensor(device="cpu"))
    assert dense_diffmean.shape == (int(feature_vocab.size),)
    assert dense_diffmean.dtype == t.float32
    assert dense_diffmean.device.type == "cpu"
    assert dense_diffmean.is_contiguous()
    assert bool(t.isfinite(dense_diffmean).all().item())
    dense_diffmean_snapshot = dense_diffmean.clone()

    frame = pd.DataFrame(
        {
            "concept_id": [0] * len(FULL_FACTORS),
            "input_id": [0] * len(FULL_FACTORS),
            "factor": list(FULL_FACTORS),
        }
    )
    cells = build_project_cell_grid(frame, seed=BASE_SEED)
    expected_identity = tuple(
        (row, method, float(factor).hex())
        for row, factor in enumerate(FULL_FACTORS)
        for method in PROJECT_METHODS
    )
    assert (
        tuple((cell.source_row_position, cell.method, cell.factor_hex) for cell in cells)
        == expected_identity
    )
    assert len(cells) == len(FULL_FACTORS) * len(PROJECT_METHODS) == 24
    assert all(cell.driver == "project" for cell in cells)
    assert all((cell.concept_id, cell.input_id) == (0, 0) for cell in cells)

    requests = tuple(
        SelfConditioningGenerationRequest(
            policy=_project_score_policy(
                cell,
                setup=real_gemma_stack.setup,
                diffmean=dense_diffmean,
                factor=float.fromhex(cell.factor_hex),
            ),
            selection_seed=cell.selection_seed,
            token_seed=cell.token_seed,
        )
        for cell in cells
    )
    expected_selection_seed = derive_selection_seed(BASE_SEED, 0, 0)
    assert {request.selection_seed for request in requests} == {expected_selection_seed}
    assert tuple(request.token_seed for request in requests) == tuple(
        derive_generation_seed(BASE_SEED, method, 0, 0, factor)
        for factor in FULL_FACTORS
        for method in PROJECT_METHODS
    )
    assert len({request.token_seed for request in requests}) == len(requests)
    assert SELF_CONDITIONING_MASKED_POLICY.endswith("sign_rank_v3")
    for cell, request in zip(cells, requests, strict=True):
        expected_type = (
            SelfCondScoreInterpolation
            if cell.method == "SelfConditioning"
            else SelfCondScoreMasking
        )
        assert type(request.policy) is expected_type
        assert t.equal(request.policy.target_scores, dense_diffmean_snapshot)
        assert request.policy.target_scores.data_ptr() != dense_diffmean.data_ptr()
    assert len({request.policy.target_scores.data_ptr() for request in requests}) == len(requests)

    index_by_cell = {cell: index for index, cell in enumerate(cells)}
    legacy_w4_groups = tuple(
        tuple(index_by_cell[cell] for cell in owned_cells(cells, rank=rank, world_size=4))
        for rank in range(4)
    )
    candidate_w4_groups = tuple(
        tuple(index_by_cell[cell] for cell in owned_project_cells(cells, rank=rank, world_size=4))
        for rank in range(4)
    )
    canonical_group = tuple(range(len(cells)))
    candidate_w1_group = tuple(
        index_by_cell[cell] for cell in owned_project_cells(cells, rank=0, world_size=1)
    )
    assert [len(indices) for indices in legacy_w4_groups] == [6, 6, 6, 6]
    assert [len(indices) for indices in candidate_w4_groups] == [24, 0, 0, 0]
    assert candidate_w1_group == canonical_group
    assert candidate_w4_groups == (canonical_group, (), (), ())
    assert candidate_w4_groups[0] == candidate_w1_group
    assert sorted(index for indices in legacy_w4_groups for index in indices) == list(
        canonical_group
    )
    assert sorted(index for indices in candidate_w4_groups for index in indices) == list(
        canonical_group
    )
    batch_groups = (
        # Candidate W=1 and W=4 enter the exact same ordered B=24 physical
        # decode, so execute it once while retaining all four legacy B=6 runs.
        ("w1-b24-and-candidate-w4-rank0-b24", candidate_w1_group),
        *((f"legacy-w4-rank{rank}-b6", indices) for rank, indices in enumerate(legacy_w4_groups)),
    )
    assert [len(indices) for _label, indices in batch_groups] == [24, 6, 6, 6, 6]
    assert sum((len(indices) for indices in legacy_w4_groups), start=0) == len(cells)

    def prepare_rows(
        row_indices: tuple[int, ...],
    ) -> tuple[tuple[InferenceEngine, SelfConditionedPrefill, Any], ...]:
        rows = []
        selection_rngs: list[t.Generator] = []
        selection_rng_start_states: list[t.Tensor] = []
        for row in row_indices:
            request = requests[row]
            config = build_self_conditioning_config(
                real_gemma_stack.setup,
                real_target_stack.model,
                request.policy,
                selection_seed=request.selection_seed,
            )
            assert config.score_policy is request.policy
            assert isinstance(config.selection, SamplingBundle)
            assert config.selection.rng.initial_seed() == expected_selection_seed
            selection_rngs.append(config.selection.rng)
            selection_rng_start_states.append(config.selection.rng.get_state().clone())
            provider = SingleEntryContextProvider(
                real_case.prepared_prompt.token_ids,
                real_case.prepared_prompt.segments,
            )
            engine = build_engine_self_conditioned(
                real_target_stack.model,
                real_target_stack.tokenizer,
                device,
                config,
                context_provider=provider,
            )
            prefill = engine.prepare_self_conditioned_prefill(
                list(real_case.prepared_prompt.token_ids),
                max_new_tokens=MAX_NEW_TOKENS,
            )
            rows.append((engine, prefill, config))
        assert len(rows) == len(row_indices)
        assert len({id(engine) for engine, _prefill, _config in rows}) == len(row_indices)
        assert len({id(engine._model) for engine, _prefill, _config in rows}) == len(row_indices)
        assert len({id(config) for _engine, _prefill, config in rows}) == len(row_indices)
        assert len({id(rng) for rng in selection_rngs}) == len(row_indices)
        assert all(
            t.equal(state, selection_rng_start_states[0])
            for state in selection_rng_start_states[1:]
        )
        for (_engine, _prefill, config), row in zip(rows, row_indices, strict=True):
            assert config.score_policy is requests[row].policy
        return tuple(rows)

    prompt_tokens = real_case.prepared_prompt.token_ids
    decode_steps = 16
    forced_rows = tuple(
        tuple(prompt_tokens[(step + row * 7) % len(prompt_tokens)] for step in range(decode_steps))
        for row in range(len(requests))
    )
    parameter_versions = tuple(
        (parameter.data_ptr(), parameter._version)
        for parameter in real_target_stack.model.parameters()
    )
    scalar_forwards = 0

    def capture_scalar_rows(
        row_indices: tuple[int, ...],
    ) -> tuple[
        dict[int, list[t.Tensor]],
        dict[int, list[t.Tensor]],
        dict[int, list[t.Tensor]],
        dict[int, list[t.Tensor]],
    ]:
        nonlocal scalar_forwards
        scalar_logits = {row: [] for row in row_indices}
        scalar_fp_logits = {row: [] for row in row_indices}
        scalar_k_caches = {row: [] for row in row_indices}
        scalar_v_caches = {row: [] for row in row_indices}
        for row in row_indices:
            engine, prefill, _config = prepare_rows((row,))[0]
            initial_position = int(prefill.kv_cache.get_pos())
            target_slot = prefill.runtime.state.target_slot_idx
            assert prefill.runtime.tokens_in_seg == 0
            assert not prefill.runtime.fp_buffer
            assert prefill.segment_size > decode_steps
            for step, token in enumerate(forced_rows[row]):
                logits = engine._model(
                    t.tensor([[token]], dtype=t.long, device=device),
                    kv_cache=prefill.kv_cache,
                )
                scalar_forwards += 1
                runtime = engine._model.self_cond_runtime_snapshot()
                position = int(prefill.kv_cache.get_pos())
                assert runtime.tokens_in_seg == step + 1
                assert len(runtime.fp_buffer) == step + 1
                assert runtime.state.target_slot_idx == target_slot
                assert position == initial_position + step + 1
                scalar_logits[row].append(logits.detach().cpu())
                scalar_fp_logits[row].append(runtime.fp_buffer[-1].detach().cpu())
                scalar_k_caches[row].append(
                    prefill.kv_cache.k_cache[:, :, :position].detach().cpu()
                )
                scalar_v_caches[row].append(
                    prefill.kv_cache.v_cache[:, :, :position].detach().cpu()
                )
        return scalar_logits, scalar_fp_logits, scalar_k_caches, scalar_v_caches

    def source_tensors(prefill: SelfConditionedPrefill) -> tuple[t.Tensor, ...]:
        encoder_out = prefill.runtime.state.encoder_out
        position = int(prefill.kv_cache.get_pos())
        return (
            prefill.logits.detach().cpu().clone(),
            prefill.kv_cache.k_cache[:, :, :position].detach().cpu().clone(),
            prefill.kv_cache.v_cache[:, :, :position].detach().cpu().clone(),
            prefill.kv_cache.cache_seqlens.detach().cpu().clone(),
            encoder_out.values.detach().cpu().clone(),
            encoder_out.layout.flat_indices.detach().cpu().clone(),
            encoder_out.layout.valid_mask.detach().cpu().clone(),
            encoder_out.layout.batch_indices.detach().cpu().clone(),
            encoder_out.layout.segment_indices.detach().cpu().clone(),
        )

    metrics = _TeacherForcedMetrics()
    cross_shape_metrics = _TeacherForcedMetrics()
    candidate_lm: dict[tuple[int, int], t.Tensor] = {}
    candidate_fp: dict[tuple[int, int], t.Tensor] = {}
    candidate_k_token: dict[tuple[int, int], t.Tensor] = {}
    candidate_v_token: dict[tuple[int, int], t.Tensor] = {}
    candidate_source_fingerprints: dict[
        int,
        tuple[tuple[tuple[int, ...], str, str], ...],
    ] = {}
    legacy_cross_shape_keys: set[tuple[int, int]] = set()
    legacy_source_rows: set[int] = set()

    def tensor_fingerprint(value: t.Tensor) -> tuple[tuple[int, ...], str, str]:
        cpu = value.detach().cpu().contiguous()
        digest = hashlib.sha256(cpu.view(t.uint8).numpy().tobytes()).hexdigest()
        return tuple(cpu.shape), str(cpu.dtype), digest

    batched_forwards = 0
    for group_index, (_label, row_indices) in enumerate(batch_groups):
        # Model each reference and candidate as a fresh producer process with
        # the exact physical group owned by that topology.
        t.compiler.reset()
        (
            scalar_logits,
            scalar_fp_logits,
            scalar_k_caches,
            scalar_v_caches,
        ) = capture_scalar_rows(row_indices)
        t.compiler.reset()
        batch_rows = prepare_rows(row_indices)
        prefills = tuple(prefill for _engine, prefill, _config in batch_rows)
        assert len(prefills) == len(row_indices)
        initial_position = int(prefills[0].kv_cache.get_pos())
        assert {int(prefill.kv_cache.get_pos()) for prefill in prefills} == {initial_position}
        assert all(prefill.runtime.tokens_in_seg == 0 for prefill in prefills)
        assert all(not prefill.runtime.fp_buffer for prefill in prefills)
        assert all(prefill.segment_size > decode_steps for prefill in prefills)
        source_snapshots = tuple(source_tensors(prefill) for prefill in prefills)
        for logical_row, snapshot in zip(row_indices, source_snapshots, strict=True):
            fingerprints = tuple(tensor_fingerprint(value) for value in snapshot)
            if group_index == 0:
                assert logical_row not in candidate_source_fingerprints
                candidate_source_fingerprints[logical_row] = fingerprints
            else:
                assert candidate_source_fingerprints[logical_row] == fingerprints
                assert logical_row not in legacy_source_rows
                legacy_source_rows.add(logical_row)
        for prefill in prefills:
            assert bool(prefill.kv_cache.k_cache[:, :, initial_position:].eq(0).all().item())
            assert bool(prefill.kv_cache.v_cache[:, :, initial_position:].eq(0).all().item())

        batch_cache = stack_prefilled_kv_caches(tuple(prefill.kv_cache for prefill in prefills))
        batch_encoder = stack_packed_encoder_outputs(
            tuple(prefill.runtime.state.encoder_out for prefill in prefills)
        )
        assert batch_cache.batch_size == len(row_indices)
        assert int(batch_cache.get_pos()) == initial_position
        assert batch_cache.cache_seqlens.tolist() == [initial_position] * len(row_indices)
        assert batch_encoder.layout.batch_size == len(row_indices)
        assert (
            batch_encoder.layout.n_segments
            == prefills[0].runtime.state.encoder_out.layout.n_segments
        )
        assert set(
            batch_encoder.layout.batch_indices[batch_encoder.layout.valid_mask].tolist()
        ) == set(range(len(row_indices)))
        seg_map = t.tensor(
            [[prefill.runtime.state.target_slot_idx] for prefill in prefills],
            dtype=t.long,
            device=device,
        )
        assert tuple(seg_map.shape) == (len(row_indices), 1)
        for physical_row, prefill in enumerate(prefills):
            assert t.equal(
                batch_cache.k_cache[:, physical_row : physical_row + 1, :initial_position].cpu(),
                prefill.kv_cache.k_cache[:, :, :initial_position].cpu(),
            )
            assert t.equal(
                batch_cache.v_cache[:, physical_row : physical_row + 1, :initial_position].cpu(),
                prefill.kv_cache.v_cache[:, :, :initial_position].cpu(),
            )
            source_encoder = prefill.runtime.state.encoder_out
            source_layout = source_encoder.layout
            source_valid = source_layout.valid_mask
            batch_layout = batch_encoder.layout
            batch_valid = batch_layout.valid_mask & batch_layout.batch_indices.eq(physical_row)
            assert int(batch_valid.sum().item()) == int(source_valid.sum().item())
            assert t.equal(
                batch_encoder.values[0, batch_valid],
                source_encoder.values[0, source_valid],
            )
            assert t.equal(
                batch_layout.segment_indices[batch_valid],
                source_layout.segment_indices[source_valid],
            )
            assert t.equal(
                batch_layout.flat_indices[batch_valid] % batch_layout.feature_count,
                source_layout.flat_indices[source_valid] % source_layout.feature_count,
            )

        for step in range(decode_steps):
            emitted = t.tensor(
                [[forced_rows[row][step]] for row in row_indices],
                dtype=t.long,
                device=device,
            )
            batched_logits, batched_fp_logits = real_target_stack.model(
                emitted,
                None,
                seg_map=seg_map,
                kv_cache=batch_cache,
                encoder_out=batch_encoder,
                encoder_strict=True,
                return_fp_logits=True,
            )
            batched_forwards += 1
            position = int(batch_cache.get_pos())
            expected_position = initial_position + step + 1
            assert position == expected_position
            assert batch_cache.cache_seqlens.tolist() == [expected_position] * len(row_indices)
            assert batched_logits.size(0) == len(row_indices)
            assert batched_fp_logits.size(0) == len(row_indices)
            for physical_row, logical_row in enumerate(row_indices):
                key = (logical_row, step)
                metrics.record_lm(
                    batched_logits[physical_row : physical_row + 1],
                    scalar_logits[logical_row][step],
                )
                metrics.record_fp(
                    batched_fp_logits[physical_row : physical_row + 1],
                    scalar_fp_logits[logical_row][step],
                )
                metrics.record_kv(
                    batch_cache.k_cache[:, physical_row : physical_row + 1, :position],
                    scalar_k_caches[logical_row][step],
                )
                metrics.record_kv(
                    batch_cache.v_cache[:, physical_row : physical_row + 1, :position],
                    scalar_v_caches[logical_row][step],
                )
                current_lm = batched_logits[physical_row : physical_row + 1]
                current_fp = batched_fp_logits[physical_row : physical_row + 1]
                current_k_token = batch_cache.k_cache[
                    :, physical_row : physical_row + 1, position - 1 : position
                ]
                current_v_token = batch_cache.v_cache[
                    :, physical_row : physical_row + 1, position - 1 : position
                ]
                if group_index == 0:
                    assert key not in candidate_lm
                    candidate_lm[key] = current_lm.detach().cpu().clone()
                    candidate_fp[key] = current_fp.detach().cpu().clone()
                    candidate_k_token[key] = current_k_token.detach().cpu().clone()
                    candidate_v_token[key] = current_v_token.detach().cpu().clone()
                else:
                    cross_shape_metrics.record_lm(current_lm, candidate_lm[key])
                    cross_shape_metrics.record_fp(current_fp, candidate_fp[key])
                    cross_shape_metrics.record_kv(current_k_token, candidate_k_token[key])
                    cross_shape_metrics.record_kv(current_v_token, candidate_v_token[key])
                    assert key not in legacy_cross_shape_keys
                    legacy_cross_shape_keys.add(key)

        for prefill, expected in zip(prefills, source_snapshots, strict=True):
            assert prefill.runtime.tokens_in_seg == 0
            assert not prefill.runtime.fp_buffer
            assert int(prefill.kv_cache.get_pos()) == initial_position
            for actual, snapshot in zip(source_tensors(prefill), expected, strict=True):
                assert t.equal(actual, snapshot)
            assert bool(prefill.kv_cache.k_cache[:, :, initial_position:].eq(0).all().item())
            assert bool(prefill.kv_cache.v_cache[:, :, initial_position:].eq(0).all().item())

    expected_states = decode_steps * sum(len(indices) for _label, indices in batch_groups)
    assert expected_states == 768
    assert scalar_forwards == expected_states
    assert batched_forwards == len(batch_groups) * decode_steps == 80
    assert expected_states * 2 * layers == 18_432
    expected_cross_shape_keys = {
        (row, step) for row in range(len(requests)) for step in range(decode_steps)
    }
    assert set(candidate_lm) == expected_cross_shape_keys
    assert set(candidate_fp) == expected_cross_shape_keys
    assert set(candidate_k_token) == expected_cross_shape_keys
    assert set(candidate_v_token) == expected_cross_shape_keys
    assert legacy_cross_shape_keys == expected_cross_shape_keys
    assert set(candidate_source_fingerprints) == set(range(len(requests)))
    assert legacy_source_rows == set(range(len(requests)))
    metrics.assert_acceptance(
        label="real project H200 candidate W4/B24 plus four legacy W4/B6 groups",
        rows=sum(len(indices) for _label, indices in batch_groups),
        steps=decode_steps,
        layers=layers,
    )
    cross_shape_metrics.assert_acceptance(
        label="real project H200 direct candidate B24 versus legacy B6",
        rows=len(requests),
        steps=decode_steps,
        layers=layers,
    )
    assert t.equal(dense_diffmean, dense_diffmean_snapshot)
    assert (
        tuple(
            (parameter.data_ptr(), parameter._version)
            for parameter in real_target_stack.model.parameters()
        )
        == parameter_versions
    )
    for request in requests:
        assert t.equal(request.policy.target_scores, dense_diffmean_snapshot)


@pytest.mark.slow
def test_real_lora_short_training_and_generation(
    monkeypatch: pytest.MonkeyPatch,
    real_gemma_stack: _GemmaStack,
    real_target_stack: _TargetStack,
    real_case: _RealCase,
) -> None:
    """Four real LoRA epochs trend down, stay finite, and detach transactionally."""

    lora_layers = _lora_layers(real_gemma_stack.setup)
    recorded_losses: list[t.Tensor] = []
    original_forward = real_target_stack.model.forward

    def recording_forward(
        idx: t.Tensor, targets: t.Tensor | None = None, **kwargs: Any
    ) -> t.Tensor:
        loss = original_forward(idx, targets, **kwargs)
        assert loss.ndim == 0
        assert loss.device.type == "cuda"
        assert loss.requires_grad
        assert isinstance(targets, t.Tensor)
        supervised_tokens = targets.ne(-1).sum()
        assert int(supervised_tokens.item()) > 0
        normalized_loss = loss / supervised_tokens
        recorded_losses.append(normalized_loss.detach().to(dtype=t.float32, device="cpu"))
        return loss

    original_projections = {
        layer: _get_block(real_target_stack.model, layer).attn.c_proj for layer in lora_layers
    }
    # Scope the recording wrapper to the TRAINING call only: the same model
    # generates later in this test, and generation's return_fp_logits prefill
    # returns a (logits, fp_logits) tuple the loss-shaped wrapper must never see.
    with monkeypatch.context() as training_patch:
        training_patch.setattr(real_target_stack.model, "forward", recording_forward)
        model_device = next(real_target_stack.model.parameters()).device
        cpu_rng_before = t.get_rng_state().clone()
        cuda_rng_before = t.cuda.get_rng_state(model_device).clone()
        state = train_encdec_lora(
            real_target_stack.model,
            real_case.positive_rows,
            layers=lora_layers,
            base_seed=BASE_SEED,
            concept_id=0,
            surface="real concept 0 EncDecLoRA short fit",
            epochs=4,
        )
        assert t.equal(t.get_rng_state(), cpu_rng_before)
        assert t.equal(t.cuda.get_rng_state(model_device), cuda_rng_before)
    # monkeypatch.undo() re-binds the inherited method as an INSTANCE
    # attribute; drop it so the module-scoped model leaves no shadow over
    # the class-level forward for later tests.
    real_target_stack.model.__dict__.pop("forward", None)
    assert len(recorded_losses) == 4 * FIT_ROWS_PER_CLASS
    assert all(bool(t.isfinite(loss).item()) for loss in recorded_losses)
    quartile = len(recorded_losses) // 4
    first_quartile_mean = float(t.stack(recorded_losses[:quartile]).mean())
    last_quartile_mean = float(t.stack(recorded_losses[-quartile:]).mean())
    assert last_quartile_mean < first_quartile_mean
    assert len(state) == 2 * len(lora_layers)
    for key, value in state.items():
        assert key.endswith(("_A", "_B"))
        assert value.dtype == t.float32
        assert value.device.type == "cpu"
        assert bool(t.isfinite(value).all().item())
    assert any(bool(state[key].ne(0).any().item()) for key in state if key.endswith("_B"))
    for layer, original in original_projections.items():
        assert _get_block(real_target_stack.model, layer).attn.c_proj is original

    seed = derive_generation_seed(BASE_SEED, "EncDecLoRA", 0, 0, 1.0)
    control_before = _generate(
        real_gemma_stack,
        real_target_stack,
        real_case.prepared_prompt,
        seed=seed,
    )
    with steering_delta(real_target_stack.model, lora_state=state, lora_layers=lora_layers):
        assert all(
            isinstance(_get_block(real_target_stack.model, layer).attn.c_proj, RsLoRAAdapter)
            for layer in lora_layers
        )
        adapted = _generate(
            real_gemma_stack,
            real_target_stack,
            real_case.prepared_prompt,
            seed=seed,
        )
    assert adapted != control_before, (
        "two-epoch real LoRA did not change factor-1 generation: "
        f"control_chars={len(control_before)}, adapted_chars={len(adapted)}"
    )
    for layer, original in original_projections.items():
        assert _get_block(real_target_stack.model, layer).attn.c_proj is original
    control_after = _generate(
        real_gemma_stack,
        real_target_stack,
        real_case.prepared_prompt,
        seed=seed,
    )
    assert control_after == control_before


@pytest.mark.slow
def test_real_prompt_recovery_roundtrip(real_gemma_stack: _GemmaStack) -> None:
    """Both real c2 prompts recover exactly and fit the l22 generation budget."""

    recovered = recover_concept_steering_prompts(
        real_gemma_stack.aux_frame,
        max_concepts=2,
    )
    assert set(recovered) == {0, 1}
    assert all(prompt.strip() for prompt in recovered.values())
    sequence_len = _sequence_len(real_gemma_stack.setup)
    for concept_id in (0, 1):
        rows = real_gemma_stack.aux_frame.loc[
            real_gemma_stack.aux_frame["concept_id"].eq(concept_id)
            & real_gemma_stack.aux_frame["input_id"].eq(0)
        ]
        assert len(rows) == 1
        original_prompt = str(rows.iloc[0]["original_prompt"])
        raw_steered_prompt = recovered[concept_id] + QUESTION_MARKER + original_prompt
        reconstructed = QWEN_STEERED_PREFIX + raw_steered_prompt + QWEN_STEERED_SUFFIX
        assert reconstructed == str(rows.iloc[0]["steered_input"])
        assert raw_steered_prompt.endswith(original_prompt)
        rendered = render_prompt_for_chat_completion(
            real_gemma_stack.tokenizer,
            raw_steered_prompt,
        )
        assert rendered.token_ids
        assert isinstance(rendered.assistant_end_id, int)
        assert len(rendered.token_ids) + MAX_NEW_TOKENS <= sequence_len, (
            f"concept {concept_id} steered prompt uses {len(rendered.token_ids)} tokens; "
            f"budget {MAX_NEW_TOKENS} exceeds sequence_len={sequence_len}"
        )
