"""Functional contracts for the AxBench encoder-decoder baseline driver."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from scripts.benchmark import benchmark_axbench_upstream as benchmark
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, INPUT_COLUMNS, METHOD_FACTORS
from scripts.eval.axbench_upstream.run_baselines import AXBENCH_UPSTREAM_COMMIT
from steering.conditional.delta import DeltaSamplingWeights
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
)
from steering.eval.axbench_upstream.data import ConceptTrainingPairs
from steering.eval.axbench_upstream.encdec_baselines import run as driver
from steering.eval.axbench_upstream.encdec_baselines.capture import FitRowConditioning
from steering.eval.axbench_upstream.encdec_baselines.directions import FittedDirection
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    QUESTION_MARKER,
    QWEN_STEERED_PREFIX,
    QWEN_STEERED_SUFFIX,
    SMOLLM3_STEERED_PREFIX,
    SMOLLM3_STEERED_SUFFIX,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import RenderedFitRow
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.conftest import (
    TINY_HEAD_DIM,
    TINY_N_EMBD,
    TINY_N_HEAD,
    TINY_N_KV_HEAD,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_SEQ_LEN,
    TINY_VOCAB,
    MockTokenizer,
)
from tests.encdec_test_helpers import (
    EXPECTED_TURN_GEOMETRY,
    TURN_FIELD_DEFAULTS,
    turn_prompt_selection_context,
)
from tests.encdec_test_helpers import method_setup_double as _dispatch_setup

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def _main_frame(max_concepts: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for concept_id in range(max_concepts):
        for input_id in range(10):
            for factor in FULL_FACTORS:
                rows.append(
                    {
                        "dataset_name": "AlpacaEval",
                        "concept_id": concept_id,
                        "input_concept": f"concept-{concept_id}",
                        "input_id": input_id,
                        "factor": factor,
                        "original_prompt": f"prompt {concept_id}/{input_id}",
                        "steered_input": f"main steered {concept_id}/{input_id}",
                        "input": f"input {concept_id}/{input_id}",
                        "suppress_original": f"suppress original {concept_id}/{input_id}",
                        "suppress_rewrite": f"suppress rewrite {concept_id}/{input_id}",
                        "steered_prompt": f"steered prompt {concept_id}/{input_id}",
                        "defense": "none",
                        "sae_link": f"https://example.test/{concept_id}",
                        "sae_id": 100 + concept_id,
                        "DiffMean_steered_generation": "healthy DiffMean generation",
                        "DiffMean_strength": factor,
                        "LinearProbe_steered_generation": "healthy LinearProbe generation",
                        "LinearProbe_strength": factor,
                    }
                )
    frame = pd.DataFrame(rows)
    for column in ("concept_id", "input_id", "sae_id"):
        frame[column] = frame[column].astype("int64")
    for column in ("factor", "DiffMean_strength", "LinearProbe_strength"):
        frame[column] = frame[column].astype("float64")
    return frame.loc[
        :,
        [
            *INPUT_COLUMNS,
            "DiffMean_steered_generation",
            "DiffMean_strength",
            "LinearProbe_steered_generation",
            "LinearProbe_strength",
        ],
    ]


def _aux_frame(main: pd.DataFrame) -> pd.DataFrame:
    aux = main.loc[main["factor"].eq(1.0), list(INPUT_COLUMNS)].copy()
    aux["steered_input"] = pd.Series(
        [
            QWEN_STEERED_PREFIX
            + f"steer-{int(row.concept_id)}"
            + QUESTION_MARKER
            + str(row.original_prompt)
            + QWEN_STEERED_SUFFIX
            for row in aux.itertuples(index=False)
        ],
        index=aux.index,
        dtype="object",
    )
    aux["PromptSteering_steered_generation"] = "healthy prompt generation"
    aux["PromptSteering_perplexity"] = pd.Series([1.0] * len(aux), index=aux.index, dtype="float64")
    aux["LoRA_steered_generation"] = "healthy LoRA generation"
    aux["LoRA_perplexity"] = pd.Series([1.0] * len(aux), index=aux.index, dtype="float64")
    return aux.loc[
        :,
        [
            *INPUT_COLUMNS,
            "PromptSteering_steered_generation",
            "PromptSteering_perplexity",
            "LoRA_steered_generation",
            "LoRA_perplexity",
        ],
    ]


def _training(max_concepts: int = 2) -> dict[int, ConceptTrainingPairs]:
    return {
        concept_id: ConceptTrainingPairs(
            concept_id=concept_id,
            concept=f"concept-{concept_id}",
            genre="text",
            positive=((f"positive input {concept_id}", f"positive output {concept_id}"),),
            negative=((f"negative input {concept_id}", f"negative output {concept_id}"),),
        )
        for concept_id in range(max_concepts)
    }


def _write_inputs(tmp_path: Path, *, max_concepts: int = 2) -> dict[str, Any]:
    prepared_dir = tmp_path / "prepared"
    (prepared_dir / "generate").mkdir(parents=True)
    (prepared_dir / "prepare_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": {"revision": "dataset-revision"},
                "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    (prepared_dir / "generate" / "metadata.jsonl").write_text("{}\n", encoding="utf-8")
    (prepared_dir / "generate" / "train_data.parquet").write_bytes(b"training")
    main = _main_frame(max_concepts)
    main_path = tmp_path / "main" / "inference" / "steering_data.parquet"
    aux_path = tmp_path / "aux" / "inference" / "steering_data.parquet"
    main_path.parent.mkdir(parents=True)
    aux_path.parent.mkdir(parents=True)
    main.to_parquet(main_path, index=False)
    _aux_frame(main).to_parquet(aux_path, index=False)
    source_manifest = {
        "target_model": "Qwen/Qwen2.5-3B-Instruct",
        "config": {
            "train": {"model_name": "Qwen/Qwen2.5-3B-Instruct"},
            "inference": {
                "model_name": "Qwen/Qwen2.5-3B-Instruct",
                "steering_model_name": "Qwen/Qwen2.5-3B-Instruct",
            },
        },
    }
    for path in (main_path, aux_path):
        (path.parent.parent / "run_manifest.json").write_text(
            json.dumps(source_manifest),
            encoding="utf-8",
        )
    target_meta_path = tmp_path / "meta_000007.json"
    target_meta_path.write_text('{"step": 7}', encoding="utf-8")
    (tmp_path / "model_000007.pt").write_bytes(b"target-model")
    feature_statistics_path = tmp_path / "feature_statistics.safetensors"
    feature_statistics_path.write_bytes(b"statistics")
    root = tmp_path / "encdec_v1"
    return {
        "main_path": main_path,
        "aux_path": aux_path,
        "prepared_dir": prepared_dir,
        "output_main_path": root / "main" / "inference" / "steering_data.parquet",
        "output_aux_path": root / "aux" / "inference" / "steering_data.parquet",
        "manifest_path": root / "run_manifest.json",
        "train_dir": root / "train",
        "target_meta_path": target_meta_path,
        "feature_statistics_path": feature_statistics_path,
        "max_concepts": max_concepts,
        "device": "cpu",
        "seed": 42,
        "max_new_tokens": 8,
        "force": False,
        "pretrain_meta_path": None,
        "require_exact_statistics_identity": True,
    }


def _fit_conditioning(token_ids: tuple[int, ...]) -> FitRowConditioning:
    return FitRowConditioning(
        token_ids=token_ids,
        segments=(ContextFeatureSegment((1, 2), (0, 1), 0, len(token_ids)),),
        segment_assignments=(0,) * len(token_ids),
    )


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    *,
    collapsed_method: str | None = None,
) -> dict[str, Any]:
    events: list[str] = []
    batches: list[tuple[tuple[str, ...], int, int, int]] = []
    setup = SimpleNamespace(
        target_meta_path=Path(kwargs["target_meta_path"]).resolve(),
        meta={
            "step": 7,
            # n_layer=12 reproduces the vendored-fraction mapping's original
            # hand-picked constants exactly (intervention layer 9, LoRA
            # layers (2, 5, 7, 9)) -- see layer_resolution.py.
            "model_config": {"sequence_len": 64, "n_layer": 12},
            "user_config": {"model_tag": "tiny-test-target"},
        },
        n_sae_features=8,
        feature_vocab=None,
        **TURN_FIELD_DEFAULTS,
    )
    model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=64, n_embd=3))

    def render(_tokenizer: Any, prompt: str) -> Any:
        steered = prompt.startswith("steer-")
        original = prompt.split(QUESTION_MARKER, 1)[1] if steered else prompt
        concept_id, input_id = (int(value) for value in original.removeprefix("prompt ").split("/"))
        family = 90 if steered else 10
        return SimpleNamespace(
            token_ids=[1, family + concept_id, 30 + input_id, 2],
            assistant_end_id=9,
        )

    def render_fit(
        _tokenizer: Any,
        *,
        input_text: str,
        output_text: str,
        sequence_len: int,
        surface: str,
    ) -> RenderedFitRow:
        del output_text, sequence_len
        concept_id = int(input_text.rsplit(" ", 1)[1])
        polarity = 1 if "positive" in surface else 2
        return RenderedFitRow(
            ids=(1, 20 + concept_id, polarity, 9),
            mask=(0, 0, 1, 1),
            prompt_token_count=2,
        )

    def prepare(
        _setup: Any,
        _extraction_model: Any,
        _sae: Any,
        _tokenizer: Any,
        *,
        token_ids: tuple[int, ...],
        assistant_end_id: int,
    ) -> driver.PreparedPromptContext:
        events.append("prepare")
        return driver.PreparedPromptContext(
            token_ids=tuple(token_ids),
            assistant_end_id=assistant_end_id,
            segments=(ContextFeatureSegment((1, 2), (0, 1), 0, 4),),
        )

    def build_conditioning(
        _setup: Any,
        _extraction_model: Any,
        _sae: Any,
        _tokenizer: Any,
        *,
        token_ids: tuple[int, ...],
        surface: str,
    ) -> FitRowConditioning:
        events.append(f"condition:{surface}")
        return _fit_conditioning(tuple(token_ids))

    def capture(
        _model: Any,
        row: FitRowConditioning,
        *,
        layer: int,
        prefix_length: int,
    ) -> t.Tensor:
        assert layer == 9  # resolved intervention layer for n_layer=12
        assert prefix_length == 2
        concept_id = row.token_ids[1] - 20
        polarity = row.token_ids[2]
        events.append(f"capture:{concept_id}:{polarity}")
        return t.tensor(
            [[float(concept_id + polarity), float(polarity), 1.0]],
            dtype=t.float32,
        )

    def fit_diffmean(
        _positive: Any,
        _negative: Any,
        *,
        intervention_layer: int,
        surface: str,
    ) -> FittedDirection:
        assert intervention_layer == 9
        concept_id = int(surface.split()[1])
        events.append(f"fit-diffmean:{concept_id}")
        return FittedDirection(
            direction=t.tensor([1.0 + concept_id, 0.0, 0.0]),
            max_activation=2.0 + concept_id,
            metadata={"method": "diffmean", "intervention_layer": intervention_layer},
        )

    def fit_probe(
        _positive: Any,
        _negative: Any,
        *,
        intervention_layer: int,
        seed: int,
        concept_id: int,
        surface: str,
    ) -> FittedDirection:
        assert intervention_layer == 9
        assert seed == kwargs["seed"]
        assert str(concept_id) in surface
        events.append(f"fit-probe:{concept_id}")
        return FittedDirection(
            direction=t.tensor([10.0 + concept_id, 0.0, 0.0]),
            max_activation=3.0 + concept_id,
            metadata={"method": "probe", "intervention_layer": intervention_layer},
        )

    def save_direction(path: Path, fitted: FittedDirection) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"direction:{fitted.max_activation!r}".encode())

    def load_direction(path: Path) -> FittedDirection:
        maximum = float(path.read_bytes().decode().split(":", 1)[1])
        concept_id = int(path.name.split("_")[1])
        probe = "linear_probe" in path.name
        events.append(f"load-direction:{'probe' if probe else 'diffmean'}:{concept_id}")
        return FittedDirection(
            direction=t.tensor([10.0 + concept_id if probe else 1.0 + concept_id, 0.0, 0.0]),
            max_activation=maximum,
            metadata={
                "method": "probe" if probe else "diffmean",
                "intervention_layer": 9,
            },
        )

    def train_lora(
        _model: Any,
        rows: Any,
        *,
        layers: Any,
        base_seed: int,
        concept_id: int,
        surface: str,
    ) -> dict[str, t.Tensor]:
        assert list(layers) == [2, 5, 7, 9]
        assert base_seed == kwargs["seed"]
        assert len(rows) == 1
        assert str(concept_id) in surface
        events.append(f"fit-lora:{concept_id}")
        return {"concept": t.tensor([float(concept_id)])}

    def save_lora(path: Path, state: dict[str, t.Tensor], metadata: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "concept": float(state["concept"].item()),
                    "metadata": {key: str(value) for key, value in metadata.items()},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def load_lora(path: Path) -> tuple[dict[str, t.Tensor], dict[str, str]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        events.append(f"load-lora:{int(path.name.split('_')[1])}")
        return {"concept": t.tensor([payload["concept"]])}, payload["metadata"]

    active: list[tuple[str, float]] = []

    @contextmanager
    def delta(
        _model: Any,
        *,
        direction: t.Tensor | None = None,
        coefficient: float | None = None,
        layer: int | None = None,
        lora_state: dict[str, t.Tensor] | None = None,
        lora_layers: Any = None,
    ) -> Iterator[None]:
        if lora_state is not None:
            assert list(lora_layers) == [2, 5, 7, 9]
            active.append(("EncDecLoRA", float(lora_state["concept"].item())))
        else:
            assert direction is not None and coefficient is not None
            assert layer == 9
            method = "EncDecLinearProbe" if float(direction[0]) >= 10.0 else "EncDecDiffMean"
            active.append((method, coefficient))
        try:
            yield
        finally:
            active.pop()

    def generate(
        _setup: Any,
        _model: Any,
        _tokenizer: Any,
        prepared: driver.PreparedPromptContext,
        *,
        selection_seed: int,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        assert max_new_tokens == kwargs["max_new_tokens"]
        steered = prepared.token_ids[1] >= 90
        concept_id = prepared.token_ids[1] - (90 if steered else 10)
        input_id = prepared.token_ids[2] - 30
        if active:
            method, value = active[-1]
            if method == "EncDecDiffMean":
                factor = value / (2.0 + concept_id)
            elif method == "EncDecLinearProbe":
                factor = value / (3.0 + concept_id)
            else:
                factor = 1.0
            factor = min(FULL_FACTORS, key=lambda candidate: abs(candidate - factor))
        else:
            assert steered
            method = "EncDecPromptSteering"
            factor = 1.0
        events.append(f"generate:{method}:c{concept_id}")
        if method == collapsed_method:
            return " "
        return f"{method}|c{concept_id}|i{input_id}|f{factor}|sel{selection_seed}|tok{seed}"

    def generate_batch(
        _setup: Any,
        _model: Any,
        _tokenizer: Any,
        prepared: driver.PreparedPromptContext,
        requests: Any,
        *,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        assert max_new_tokens == kwargs["max_new_tokens"]
        concept_id = prepared.token_ids[1] - 10
        input_id = prepared.token_ids[2] - 30
        requests = tuple(requests)
        batches.append(
            (
                tuple(
                    "EncDecLinearProbe" if float(request.direction[0]) >= 10.0 else "EncDecDiffMean"
                    for request in requests
                ),
                concept_id,
                input_id,
                len(requests),
            )
        )
        generated: list[str] = []
        for request in requests:
            method = (
                "EncDecLinearProbe" if float(request.direction[0]) >= 10.0 else "EncDecDiffMean"
            )
            scale = (3.0 if method == "EncDecLinearProbe" else 2.0) + concept_id
            factor = min(
                FULL_FACTORS,
                key=lambda candidate: abs(candidate - request.coefficient / scale),
            )
            events.append(f"generate:{method}:c{concept_id}")
            if method == collapsed_method:
                generated.append(" ")
            else:
                generated.append(
                    f"{method}|c{concept_id}|i{input_id}|f{factor}|"
                    f"sel{request.selection_seed}|tok{request.token_seed}"
                )
        return tuple(generated)

    def tokenizer() -> object:
        events.append("get-tokenizer")
        return object()

    def template_prefix(_tokenizer: object) -> int:
        events.append("template-prefix")
        return 2

    def load_setup(*_args: Any, **_kwargs: Any) -> Any:
        events.append("load-setup")
        return setup

    def load_extraction(*_args: Any, **_kwargs: Any) -> tuple[object, object]:
        events.append("load-extraction")
        return object(), object()

    def load_target(*_args: Any, **_kwargs: Any) -> tuple[Any, object]:
        events.append("load-target")
        return model, object()

    def reset_compiler() -> None:
        events.append("compiler-reset")

    monkeypatch.setattr(driver, "get_tokenizer", tokenizer)
    monkeypatch.setattr(
        driver,
        "_tokenizer_artifact_identity",
        lambda: {"name": "tokenizer.pkl", "byte_length": 9, "sha256": "a" * 64},
    )
    monkeypatch.setattr(driver, "render_prompt_for_chat_completion", render)
    monkeypatch.setattr(driver, "render_fit_row", render_fit)
    monkeypatch.setattr(
        driver, "select_training_pairs", lambda *_a, **_k: _training(kwargs["max_concepts"])
    )
    monkeypatch.setattr(driver, "load_method_setup", load_setup)
    monkeypatch.setattr(driver, "load_extraction_stack", load_extraction)
    monkeypatch.setattr(driver, "prepare_prompt_context", prepare)
    monkeypatch.setattr(driver, "build_fit_row_conditioning", build_conditioning)
    monkeypatch.setattr(driver, "load_target_model", load_target)
    monkeypatch.setattr(driver.t.compiler, "reset", reset_compiler)
    monkeypatch.setattr(driver, "template_prefix_length", template_prefix)
    monkeypatch.setattr(driver, "assert_prepared_prompt_matches", lambda *_a, **_k: None)
    monkeypatch.setattr(driver, "capture_post_prefix_activations", capture)
    monkeypatch.setattr(driver, "fit_diffmean_direction", fit_diffmean)
    monkeypatch.setattr(driver, "fit_probe_direction", fit_probe)
    monkeypatch.setattr(driver, "save_direction", save_direction)
    monkeypatch.setattr(driver, "load_direction", load_direction)
    monkeypatch.setattr(driver, "train_encdec_lora", train_lora)
    monkeypatch.setattr(driver, "save_lora", save_lora)
    monkeypatch.setattr(driver, "load_lora", load_lora)
    monkeypatch.setattr(driver, "validate_lora_state", lambda *_a, **_k: None)
    monkeypatch.setattr(driver, "steering_delta", delta)
    monkeypatch.setattr(driver, "generate_encdec_baseline", generate)
    monkeypatch.setattr(driver, "generate_encdec_direction_batch", generate_batch)
    return {"batches": batches, "events": events, "setup": setup}


def _expected_generation(method: str, row: Any, *, base_seed: int) -> str:
    concept_id = int(row.concept_id)
    input_id = int(row.input_id)
    factor = float(row.factor)
    selection_seed = driver.derive_selection_seed(base_seed, concept_id, input_id)
    token_seed = driver.derive_generation_seed(
        base_seed,
        method,
        concept_id,
        input_id,
        factor,
    )
    return f"{method}|c{concept_id}|i{input_id}|f{factor}|sel{selection_seed}|tok{token_seed}"


def test_turn_mode_encdec_manifest_records_segment_geometry_and_scalar_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn-mode encdec run must publish the same geometry block as project."""

    # kills: dropping the _turn_self_conditioning_segment_geometry splice from
    # the encdec manifest (two checkpoints with different cadences would then
    # compare as identical provenance), and leaving divergence note 3
    # advertising a bounded B>1 decode that the turn batch gate makes
    # unreachable.
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)
    setup = observed["setup"]
    setup.segment_mode = driver.SegmentMode.turn_delimited_fixed_tokens
    setup.sae_model = "gemma-3-4b-pt"
    setup.sae_layer = 22
    setup.prompt_segment_size = 17
    setup.prompt_context_window = 333
    setup.turn_self_cond_segment_size_rustbpe = 37
    setup.prompt_statistics_raw_sha256 = "a" * 64
    setup.prompt_selection_context = turn_prompt_selection_context()

    driver.run_encdec_baselines(**kwargs)

    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["self_conditioning_segment_geometry"] == EXPECTED_TURN_GEOMETRY
    # The batch gate makes every turn cell scalar, so the note must say so.
    assert observed["batches"] == []
    note = manifest["divergence_notes"]["3_per_cell_generation_and_sha256_seeds"]
    assert "compatible direction cells use bounded B>1" not in note
    assert "every cell stays scalar because turn-delimited" in note
    assert "the bounded B>1 decode never runs" in note


def test_fixed_mode_encdec_manifest_is_unchanged_by_the_turn_geometry_splice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed-mode manifests must not gain the turn geometry key at all."""

    # kills: emitting a null/absent self_conditioning_segment_geometry key for
    # fixed checkpoints. The fixed divergence note is pinned verbatim by
    # test_driver_golden_frames_and_manifest_shape, so it is not restated here.
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)

    driver.run_encdec_baselines(**kwargs)

    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    assert "self_conditioning_segment_geometry" not in manifest


def test_driver_golden_frames_and_manifest_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)

    encdec_main, encdec_aux = driver.run_encdec_baselines(**kwargs)

    assert encdec_main.shape == (240, 18)
    assert encdec_aux.shape == (20, 18)
    assert tuple(encdec_main.columns) == (*INPUT_COLUMNS, *driver.ENCDEC_MAIN_COLUMNS)
    assert tuple(encdec_aux.columns) == (*INPUT_COLUMNS, *driver.ENCDEC_AUX_COLUMNS)
    for row in encdec_main.itertuples(index=False):
        assert row.EncDecDiffMean_steered_generation == _expected_generation(
            "EncDecDiffMean",
            row,
            base_seed=kwargs["seed"],
        )
        assert row.EncDecLinearProbe_steered_generation == _expected_generation(
            "EncDecLinearProbe",
            row,
            base_seed=kwargs["seed"],
        )
        assert row.EncDecDiffMean_strength == row.factor * (2.0 + row.concept_id)
        assert row.EncDecLinearProbe_strength == row.factor * (3.0 + row.concept_id)
    for row in encdec_aux.itertuples(index=False):
        assert row.steered_input == f"steer-{row.concept_id}{QUESTION_MARKER}{row.original_prompt}"
        assert row.EncDecPromptSteering_steered_generation == _expected_generation(
            "EncDecPromptSteering",
            row,
            base_seed=kwargs["seed"],
        )
        assert row.EncDecLoRA_steered_generation == _expected_generation(
            "EncDecLoRA",
            row,
            base_seed=kwargs["seed"],
        )
        assert row.EncDecPromptSteering_strength == 1.0
        assert row.EncDecLoRA_strength == 1.0
    assert tuple(column for column in encdec_aux if column.endswith("_steered_generation")) == (
        "EncDecPromptSteering_steered_generation",
        "EncDecLoRA_steered_generation",
    )
    assert len(observed["batches"]) == 20
    assert all(batch_size == 24 for _methods, _concept, _input, batch_size in observed["batches"])
    assert {
        (concept_id, input_id)
        for _methods, concept_id, input_id, _batch_size in observed["batches"]
    } == {(concept_id, input_id) for concept_id in range(2) for input_id in range(10)}
    assert all(
        methods == ("EncDecDiffMean", "EncDecLinearProbe") * 12
        for methods, _concept, _input, _batch_size in observed["batches"]
    )

    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "repository_git_commit",
        "repository_git_worktree",
        "vendored_axbench_upstream_commit",
        "target",
        "dataset_revision",
        "prepared_manifest",
        "configuration",
        "layer_resolution",
        "generation",
        "package_lock_sha256",
        "inputs",
        "method_order",
        "methods",
        "divergence_notes",
        "train_artifacts",
    }
    assert set(manifest) == expected_top
    expected_types = {
        "schema_version": int,
        "repository_git_commit": str,
        "repository_git_worktree": dict,
        "vendored_axbench_upstream_commit": str,
        "target": dict,
        "dataset_revision": str,
        "prepared_manifest": dict,
        "configuration": dict,
        "layer_resolution": dict,
        "generation": dict,
        "package_lock_sha256": str,
        "inputs": dict,
        "method_order": list,
        "methods": dict,
        "divergence_notes": dict,
        "train_artifacts": dict,
    }
    for key, expected_type in expected_types.items():
        assert type(manifest[key]) is expected_type
    assert manifest["schema_version"] == 1
    assert manifest["method_order"] == list(driver.METHOD_ORDER)
    assert manifest["configuration"] == {
        "main_path": str(Path(kwargs["main_path"]).resolve()),
        "aux_path": str(Path(kwargs["aux_path"]).resolve()),
        "prepared_dir": str(Path(kwargs["prepared_dir"]).resolve()),
        "output_main_path": str(Path(kwargs["output_main_path"]).resolve()),
        "output_aux_path": str(Path(kwargs["output_aux_path"]).resolve()),
        "manifest_path": str(Path(kwargs["manifest_path"]).resolve()),
        "train_dir": str(Path(kwargs["train_dir"]).resolve()),
        "target_meta_path": str(Path(kwargs["target_meta_path"]).resolve()),
        "feature_statistics_path": str(Path(kwargs["feature_statistics_path"]).resolve()),
        "max_concepts": 2,
        "device": "cpu",
        "seed": 42,
        "max_new_tokens": 8,
        "force": False,
        "pretrain_meta_path": None,
        "require_exact_statistics_identity": True,
    }
    assert manifest["layer_resolution"] == {
        "vendored_qwen_n_layer": 36,
        "vendored_steering_layer": 28,
        "vendored_lora_layers": [7, 14, 21, 28],
        "target_n_layer": 12,
        "formula": "round(target_n_layer * vendored_layer / vendored_qwen_n_layer)",
        "resolved_intervention_layer": 9,
        "resolved_lora_layers": [2, 5, 7, 9],
    }
    assert manifest["inputs"] == {
        "main_steering_parquet_sha256": hashlib.sha256(
            Path(kwargs["main_path"]).read_bytes()
        ).hexdigest(),
        "main_source_manifest_sha256": hashlib.sha256(
            (Path(kwargs["main_path"]).parent.parent / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "aux_steering_parquet_sha256": hashlib.sha256(
            Path(kwargs["aux_path"]).read_bytes()
        ).hexdigest(),
        "aux_source_manifest_sha256": hashlib.sha256(
            (Path(kwargs["aux_path"]).parent.parent / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "feature_statistics_sha256": hashlib.sha256(
            Path(kwargs["feature_statistics_path"]).read_bytes()
        ).hexdigest(),
    }
    assert set(manifest["methods"]) == {
        "EncDecDiffMean",
        "EncDecLinearProbe",
        "EncDecPromptSteering",
        "EncDecLoRA",
    }
    assert len(manifest["divergence_notes"]) == 12
    assert all(isinstance(value, str) for value in manifest["divergence_notes"].values())
    assert manifest["divergence_notes"]["3_per_cell_generation_and_sha256_seeds"] == (
        "independent B=1 prefills; compatible direction cells use bounded B>1 decode; "
        "PromptSteering, LoRA, and singleton resume remainders stay scalar; top_k=50 with "
        "per-cell SHA-256 seeds"
    )
    selection_seed_contract = {
        "domain": "axbench-upstream-selfconditioning-selection-v1",
        "fields": "domain,base_seed,concept_id,input_id",
        "join": "\\x1f",
        "digest": "sha256 first 8 bytes big-endian",
        "modulus": "2**31 - 1",
    }
    token_seed_contract = {
        "fields": "base_seed,method,concept_id,input_id,float_factor_hex",
        "join": "\\x1f",
        "digest": "sha256 first 8 bytes big-endian",
        "modulus": "2**31 - 1",
    }
    probe_seed_contract = {
        "domain": "axbench-upstream-encdec-probe-v1",
        "fields": "domain,base_seed,concept_id",
        "join": "\\x1f",
        "digest": "sha256 first 8 bytes big-endian",
        "modulus": "2**31 - 1",
    }
    for method in manifest["methods"].values():
        assert method["selection_seed"] == selection_seed_contract
        assert method["token_seed"] == token_seed_contract
    for method in ("EncDecDiffMean", "EncDecLinearProbe"):
        assert manifest["methods"][method]["intervention_layer"] == 9
    assert manifest["methods"]["EncDecDiffMean"]["formula"] == (
        "direction = (token_mean_pos - token_mean_neg) / (norm + f32_eps); "
        "steer h += factor*max_act*direction at the resolved intervention_layer"
    )
    probe_method = manifest["methods"]["EncDecLinearProbe"]
    assert probe_method["formula"] == (
        "direction = F.normalize(L1 liblinear_logreg_coef(row_mean_activations)); "
        "steer identical to diffmean"
    )
    assert probe_method["probe_seed"] == probe_seed_contract
    assert probe_method["estimator"] == {
        "solver": "liblinear",
        "l1_ratio": "1.0 (L1)",
        "C": 1.0,
        "max_iter": 500,
        "random_state_domain": "axbench-upstream-encdec-probe-v1",
        "class_weight": None,
    }
    assert manifest["methods"]["EncDecPromptSteering"]["question_marker"] == QUESTION_MARKER
    assert manifest["methods"]["EncDecPromptSteering"]["source_prompt_wrapper"] == {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": None,
    }
    assert manifest["methods"]["EncDecLoRA"]["scaling_formula"] == ("scaling = alpha / sqrt(rank)")
    assert manifest["methods"]["EncDecLoRA"]["forward_formula"] == (
        "output = base(x) + (B @ A @ dropout(x_fp32)) * scaling, "
        "cast back to the activation dtype (PEFT 0.13.2 dtype boundary)"
    )
    assert manifest["methods"]["EncDecLoRA"]["training_seed"] == {
        "domain": "axbench-upstream-encdec-lora-v1",
        "fields": "domain,base_seed,concept_id",
        "join": "\\x1f",
        "digest": "sha256 first 8 bytes big-endian",
        "modulus": "2**31 - 1",
        "consumers": (
            "CPU default generator + active accelerator generator (dropout; both RNG states "
            "restored) + isolated CPU shuffle generator"
        ),
    }
    assert manifest["divergence_notes"]["8_probe_intercept_semantics"] == (
        "the L1 estimator fits an intercept used only during training; "
        "steering projects onto the normalized coefficient vector alone"
    )
    for method, expected in (("EncDecDiffMean", 2.0), ("EncDecLinearProbe", 3.0)):
        encoded = manifest["methods"][method]["max_activation_by_concept"]["0"]
        assert isinstance(encoded, str)
        assert float(encoded) == expected
        assert repr(float(encoded)) == encoded
    lora = manifest["methods"]["EncDecLoRA"]["training_by_concept"]["0"]
    for key in (
        "rank",
        "alpha",
        "scaling",
        "dropout",
        "layers",
        "seed",
        "n_rows",
        "optimizer_steps",
        "scheduler_horizon",
    ):
        assert key in lora
    assert isinstance(lora["rank"], int)
    assert isinstance(lora["alpha"], float)
    assert isinstance(lora["layers"], list)
    assert lora["optimizer_steps"] == driver.LORA_EPOCHS
    assert lora["scheduler_horizon"] == driver.LORA_EPOCHS
    assert manifest["methods"]["EncDecLoRA"]["layers"] == [2, 5, 7, 9]

    method_keys = {
        "EncDecDiffMean": "diffmean",
        "EncDecLinearProbe": "probe",
        "EncDecLoRA": "lora",
    }
    artifact_map: dict[int, dict[str, dict[str, str]]] = {}
    for entry in manifest["train_artifacts"]["files"]:
        artifact_map.setdefault(entry["concept_id"], {})[method_keys[entry["method"]]] = {
            "filename": entry["path"],
            "sha256": entry["sha256"],
        }
    expected_artifact_map: dict[int, dict[str, dict[str, str]]] = {}
    for concept_id in range(kwargs["max_concepts"]):
        expected_artifact_map[concept_id] = {}
        for key, suffix in (
            ("diffmean", "diffmean"),
            ("probe", "linear_probe"),
            ("lora", "lora"),
        ):
            filename = f"concept_{concept_id:05d}_{suffix}.safetensors"
            artifact = Path(kwargs["train_dir"]) / filename
            expected_artifact_map[concept_id][key] = {
                "filename": filename,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
    assert artifact_map == expected_artifact_map


def test_driver_recovers_smollm3_prompts_from_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    model_name = "HuggingFaceTB/SmolLM3-3B-checkpoints"
    revision = "it-soup-APO"
    source_manifest = {
        "target_model": model_name,
        "config": {
            "train": {"model_name": model_name, "model_revision": revision},
            "inference": {
                "model_name": model_name,
                "model_revision": revision,
                "steering_model_name": model_name,
                "steering_model_revision": revision,
            },
        },
    }
    for key in ("main_path", "aux_path"):
        path = Path(kwargs[key])
        (path.parent.parent / "run_manifest.json").write_text(
            json.dumps(source_manifest),
            encoding="utf-8",
        )
    aux_path = Path(kwargs["aux_path"])
    aux = pd.read_parquet(aux_path)
    aux["steered_input"] = pd.Series(
        [
            SMOLLM3_STEERED_PREFIX
            + "steer-0"
            + QUESTION_MARKER
            + str(original_prompt)
            + SMOLLM3_STEERED_SUFFIX
            for original_prompt in aux["original_prompt"]
        ],
        index=aux.index,
        dtype="object",
    )
    aux.to_parquet(aux_path, index=False)
    _install_fakes(monkeypatch, kwargs)

    _encdec_main, encdec_aux = driver.run_encdec_baselines(**kwargs)

    assert set(encdec_aux["steered_input"]) == {
        f"steer-0{QUESTION_MARKER}prompt 0/{input_id}" for input_id in range(10)
    }
    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["methods"]["EncDecPromptSteering"]["source_prompt_wrapper"] == {
        "model": model_name,
        "revision": revision,
    }


@pytest.mark.parametrize("source_key", ["main_path", "aux_path"])
@pytest.mark.parametrize("failure", ["missing", "invalid", "different-model"])
def test_source_provenance_rejection_precedes_models_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_key: str,
    failure: str,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    state = _install_fakes(monkeypatch, kwargs)
    source = Path(kwargs[source_key]).parent.parent / "run_manifest.json"
    if failure == "missing":
        source.unlink()
    elif failure == "invalid":
        source.write_text("{}")
    else:
        payload = json.loads(source.read_text())
        model = "HuggingFaceTB/SmolLM3-3B-checkpoints"
        payload["target_model"] = model
        payload["config"]["train"].update(model_name=model, model_revision="it-soup-APO")
        payload["config"]["inference"].update(
            model_name=model,
            model_revision="it-soup-APO",
            steering_model_name=model,
            steering_model_revision="it-soup-APO",
        )
        source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source manifest|different prompt-wrapper provenance"):
        driver.run_encdec_baselines(**kwargs)
    assert state["events"] == []
    for key in ("output_main_path", "output_aux_path", "manifest_path", "train_dir"):
        assert not Path(kwargs[key]).exists()
    assert not (Path(kwargs["manifest_path"]).parent / "_encdec_cell_resume_v1").exists()


@pytest.mark.parametrize("source_key", ["main_path", "aux_path"])
@pytest.mark.parametrize("output_key", ["output_main_path", "output_aux_path", "manifest_path"])
def test_source_manifest_symlink_cannot_be_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_key: str,
    output_key: str,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    state = _install_fakes(monkeypatch, kwargs)
    source = Path(kwargs[source_key]).parent.parent / "run_manifest.json"
    output = Path(kwargs[output_key])
    original = source.read_bytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(original)
    source.unlink()
    source.symlink_to(output)
    kwargs["force"] = True
    with pytest.raises(driver.ImmutableArtifactError, match="regular file"):
        driver.run_encdec_baselines(**kwargs)
    assert not any(event.startswith(("load-", "generate:", "fit-")) for event in state["events"])
    assert source.read_bytes() == original


@pytest.mark.parametrize("source_key", ["main_path", "aux_path"])
def test_resume_rejects_changed_source_manifest_without_replacing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_key: str,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    state = _install_fakes(monkeypatch, kwargs)
    driver.run_encdec_baselines(**kwargs)
    outputs = {
        Path(kwargs[key]): Path(kwargs[key]).read_bytes()
        for key in ("output_main_path", "output_aux_path", "manifest_path")
    }
    source = Path(kwargs[source_key]).parent.parent / "run_manifest.json"
    source.write_bytes(source.read_bytes() + b"\n")
    kwargs["force"] = True
    state["events"].clear()
    with pytest.raises(driver.CellStoreError, match="differs from the current result"):
        driver.run_encdec_baselines(**kwargs)
    assert not any(
        event.startswith(("load-extraction", "load-target", "generate:", "fit-"))
        for event in state["events"]
    )
    assert all(path.read_bytes() == original for path, original in outputs.items())


@pytest.mark.parametrize(
    ("selector_index", "expected_method", "expected_factor"),
    (
        pytest.param(0, "EncDecDiffMean", 6.0, id="factor-6-diffmean"),
        pytest.param(1, "EncDecLoRA", 1.0, id="factor-1-lora"),
    ),
)
def test_default_trace_selector_keeps_entire_prompt_group_scalar_in_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector_index: int,
    expected_method: str,
    expected_factor: float,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    observed = _install_fakes(monkeypatch, kwargs)
    cells = driver.build_encdec_cell_grid(pd.read_parquet(kwargs["main_path"]), seed=kwargs["seed"])
    selector = benchmark._default_trace_selectors("encdec")[selector_index]
    assert (selector.method, selector.concept_id, selector.input_id, selector.factor_hex) == (
        expected_method,
        0,
        0,
        float(expected_factor).hex(),
    )
    assert sum(selector.matches(cell) for cell in cells) == 1

    dispatches: list[tuple[str, tuple[driver.CellKey, ...]]] = []
    scalar_impl = driver._generate_encdec_cell
    batch_impl = driver._generate_encdec_direction_cells

    def scalar(cell: driver.CellKey, **call_kwargs: Any) -> str:
        dispatches.append(("scalar", (cell,)))
        return scalar_impl(cell, **call_kwargs)

    def batch(group: Any, **call_kwargs: Any) -> tuple[str, ...]:
        grouped = tuple(group)
        dispatches.append(("batch", grouped))
        return batch_impl(grouped, **call_kwargs)

    monkeypatch.setattr(driver, "_generate_encdec_cell", scalar)
    monkeypatch.setattr(driver, "_generate_encdec_direction_cells", batch)
    monkeypatch.setattr(driver, "cell_trace_requested", selector.matches)

    driver.run_encdec_baselines(**kwargs)

    def dispatches_for(input_id: int) -> list[tuple[str, tuple[driver.CellKey, ...]]]:
        return [
            dispatch
            for dispatch in dispatches
            if (dispatch[1][0].concept_id, dispatch[1][0].input_id) == (0, input_id)
        ]

    selected_pair_cells = tuple(
        cell for cell in cells if (cell.concept_id, cell.input_id) == (0, 0)
    )
    assert dispatches_for(0) == [("scalar", (cell,)) for cell in selected_pair_cells]
    for input_id in range(1, 10):
        pair_cells = tuple(
            cell for cell in cells if (cell.concept_id, cell.input_id) == (0, input_id)
        )
        direction_cells = tuple(
            cell for cell in pair_cells if cell.method in driver.ENCDEC_MAIN_METHODS
        )
        auxiliary_cells = tuple(
            cell for cell in pair_cells if cell.method in driver.ENCDEC_AUX_METHODS
        )
        assert dispatches_for(input_id) == [
            ("batch", direction_cells),
            *(("scalar", (cell,)) for cell in auxiliary_cells),
        ]

    assert [
        (concept_id, input_id, batch_size)
        for _methods, concept_id, input_id, batch_size in observed["batches"]
    ] == [(0, input_id, 24) for input_id in range(1, 10)]
    assert sum(event.startswith("generate:") for event in observed["events"]) == 260


def test_missing_resume_subset_batches_only_still_missing_direction_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 0, "input_id": 0, "factor": 0.0},
            {"concept_id": 0, "input_id": 0, "factor": 1.0},
        ]
    )
    cells = driver.build_encdec_cell_grid(frame, seed=42)
    missing = (
        next(
            cell
            for cell in cells
            if cell.method == "EncDecDiffMean" and cell.factor_hex == (0.0).hex()
        ),
        next(
            cell
            for cell in cells
            if cell.method == "EncDecLinearProbe" and cell.factor_hex == (1.0).hex()
        ),
        next(cell for cell in cells if cell.method == "EncDecPromptSteering"),
    )

    class Store:
        def __init__(self) -> None:
            self.appended: list[tuple[driver.CellKey, str]] = []

        def missing(self, _cells: Any) -> tuple[driver.CellKey, ...]:
            return missing

        def append(self, cell: driver.CellKey, generation: str) -> None:
            self.appended.append((cell, generation))

    store = Store()
    observed_batches: list[tuple[driver.CellKey, ...]] = []
    observed_scalars: list[driver.CellKey] = []

    def batch(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        grouped = tuple(group)
        observed_batches.append(grouped)
        return tuple(f"batch-{index}" for index in range(len(grouped)))

    def scalar(cell: driver.CellKey, **_kwargs: Any) -> str:
        observed_scalars.append(cell)
        return "scalar"

    monkeypatch.setattr(driver, "_generate_encdec_direction_cells", batch)
    monkeypatch.setattr(driver, "_generate_encdec_cell", scalar)
    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    driver._append_missing_encdec_cells(
        cast(Any, store),
        cells,
        rows=tuple(frame.itertuples(index=False)),
        prompt_plan=cast(Any, None),
        steered_plan=cast(Any, None),
        prepared_by_tokens={},
        steered_prepared_by_tokens={},
        fitted={},
        lora_states={},
        intervention_layer=9,
        lora_layers=(2, 5, 7, 9),
        setup=cast(Any, _dispatch_setup()),
        model=cast(Any, None),
        tokenizer=cast(Any, None),
        max_new_tokens=8,
    )

    assert observed_batches == [missing[:2]]
    assert observed_scalars == [missing[2]]
    assert store.appended == [
        (missing[0], "batch-0"),
        (missing[1], "batch-1"),
        (missing[2], "scalar"),
    ]

    monkeypatch.setattr(
        driver,
        "_generate_encdec_direction_cells",
        lambda _group, **_kwargs: ("too-short",),
    )
    with pytest.raises(RuntimeError, match="wrong number of generations"):
        driver._append_missing_encdec_cells(
            cast(Any, Store()),
            cells,
            rows=tuple(frame.itertuples(index=False)),
            prompt_plan=cast(Any, None),
            steered_plan=cast(Any, None),
            prepared_by_tokens={},
            steered_prepared_by_tokens={},
            fitted={},
            lora_states={},
            intervention_layer=9,
            lora_layers=(2, 5, 7, 9),
            setup=cast(Any, _dispatch_setup()),
            model=cast(Any, None),
            tokenizer=cast(Any, None),
            max_new_tokens=8,
        )


@pytest.mark.parametrize(
    ("segment_mode", "expected_dispatch"),
    (
        pytest.param(driver.SegmentMode.fixed_tokens, "batch", id="fixed-batches"),
        pytest.param(
            driver.SegmentMode.turn_delimited_fixed_tokens,
            "scalar",
            id="turn-falls-back-to-scalar",
        ),
    ),
)
def test_encdec_direction_batch_dispatch_is_limited_to_fixed_token_mode(
    monkeypatch: pytest.MonkeyPatch,
    segment_mode: driver.SegmentMode,
    expected_dispatch: str,
) -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 0, "input_id": 0, "factor": 0.0},
            {"concept_id": 0, "input_id": 0, "factor": 1.0},
        ]
    )
    cells = driver.build_encdec_cell_grid(frame, seed=42)
    missing = (
        next(
            cell
            for cell in cells
            if cell.method == "EncDecDiffMean" and cell.factor_hex == (0.0).hex()
        ),
        next(
            cell
            for cell in cells
            if cell.method == "EncDecLinearProbe" and cell.factor_hex == (1.0).hex()
        ),
    )
    dispatches: list[tuple[str, tuple[driver.CellKey, ...]]] = []

    class Store:
        def __init__(self) -> None:
            self.appended: list[tuple[driver.CellKey, str]] = []

        def missing(self, _cells: Any) -> tuple[driver.CellKey, ...]:
            return missing

        def append(self, cell: driver.CellKey, generation: str) -> None:
            self.appended.append((cell, generation))

    def scalar(cell: driver.CellKey, **_kwargs: Any) -> str:
        dispatches.append(("scalar", (cell,)))
        return f"generated:{cell.selection_seed}:{cell.token_seed}"

    def batch(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        grouped = tuple(group)
        dispatches.append(("batch", grouped))
        return tuple(f"generated:{cell.selection_seed}:{cell.token_seed}" for cell in grouped)

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(driver, "_generate_encdec_cell", scalar)
    monkeypatch.setattr(driver, "_generate_encdec_direction_cells", batch)
    store = Store()

    driver._append_missing_encdec_cells(
        cast(Any, store),
        cells,
        rows=tuple(frame.itertuples(index=False)),
        prompt_plan=cast(Any, None),
        steered_plan=cast(Any, None),
        prepared_by_tokens={},
        steered_prepared_by_tokens={},
        fitted={},
        lora_states={},
        intervention_layer=9,
        lora_layers=(2, 5, 7, 9),
        setup=cast(Any, _dispatch_setup(segment_mode=segment_mode)),
        model=cast(Any, None),
        tokenizer=cast(Any, None),
        max_new_tokens=8,
    )

    expected_dispatches = (
        [("batch", missing)]
        if expected_dispatch == "batch"
        else [("scalar", (cell,)) for cell in missing]
    )
    assert dispatches == expected_dispatches
    assert store.appended == [
        (cell, f"generated:{cell.selection_seed}:{cell.token_seed}") for cell in missing
    ]


def test_direction_cell_batch_rejects_every_incompatible_group() -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 0, "input_id": 0, "factor": 0.0},
            {"concept_id": 0, "input_id": 0, "factor": 1.0},
            {"concept_id": 0, "input_id": 1, "factor": 0.0},
        ]
    )
    rows = tuple(frame.itertuples(index=False))
    cells = driver.build_encdec_cell_grid(frame, seed=42)

    def cell(method: str, source_row_position: int) -> driver.CellKey:
        return next(
            candidate
            for candidate in cells
            if candidate.method == method and candidate.source_row_position == source_row_position
        )

    first = cell("EncDecDiffMean", 0)
    same_pair = cell("EncDecLinearProbe", 1)
    nondirection = cell("EncDecLoRA", 1)
    other_pair = cell("EncDecDiffMean", 2)
    prompt_plan = SimpleNamespace(
        rendered_tokens_by_pair={(0, 0): (10,), (0, 1): (11,)},
    )
    fit = SimpleNamespace(direction=t.ones(2), max_activation=1.0)
    common = {
        "prompt_plan": prompt_plan,
        "prepared_by_tokens": {(10,): object(), (11,): object()},
        "fitted": {0: {"diffmean": fit, "probe": fit}},
        "intervention_layer": 9,
        "setup": None,
        "model": None,
        "tokenizer": None,
        "max_new_tokens": 8,
    }

    with pytest.raises(ValueError, match="at least two cells"):
        driver._generate_encdec_direction_cells((first,), rows=rows, **common)
    with pytest.raises(ValueError, match="non-direction cell"):
        driver._generate_encdec_direction_cells(
            (nondirection, first),
            rows=rows,
            **common,
        )
    with pytest.raises(ValueError, match="share one concept/input pair"):
        driver._generate_encdec_direction_cells(
            (first, other_pair),
            rows=rows,
            **common,
        )

    corrupted = list(rows)
    corrupted[first.source_row_position] = SimpleNamespace(
        concept_id=0,
        input_id=0,
        factor=99.0,
    )
    with pytest.raises(RuntimeError, match="no longer matches its source row"):
        driver._generate_encdec_direction_cells(
            (first, same_pair),
            rows=tuple(corrupted),
            **common,
        )


def test_fresh_single_rank_run_reloads_every_published_fit_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    observed = _install_fakes(monkeypatch, kwargs)

    driver.run_encdec_baselines(**kwargs)

    assert [
        event for event in observed["events"] if event.startswith(("load-direction:", "load-lora:"))
    ] == [
        "load-direction:diffmean:0",
        "load-direction:probe:0",
        "load-lora:0",
    ]


def test_compiler_reset_is_once_after_fit_validation_and_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    observed = _install_fakes(monkeypatch, kwargs)

    driver.run_encdec_baselines(**kwargs)

    events = observed["events"]
    reset_index = events.index("compiler-reset")
    fit_load_indices = [
        index
        for index, event in enumerate(events)
        if event.startswith(("load-direction:", "load-lora:"))
    ]
    generation_indices = [
        index for index, event in enumerate(events) if event.startswith("generate:")
    ]
    assert events.count("compiler-reset") == 1
    assert fit_load_indices
    assert generation_indices
    assert max(fit_load_indices) < reset_index < min(generation_indices)


def test_compiler_reset_helper_rearms_an_existing_compiled_callable() -> None:
    compile_count = 0

    def backend(graph: t.fx.GraphModule, _inputs: list[t.Tensor]) -> Any:
        nonlocal compile_count
        compile_count += 1
        return graph.forward

    def add_one(value: t.Tensor) -> t.Tensor:
        return value + 1

    compiled = t.compile(add_one, backend=backend)
    try:
        expected = t.tensor([2.0])
        assert t.equal(compiled(t.ones(1)), expected)
        assert t.equal(compiled(t.ones(1)), expected)
        assert compile_count == 1

        driver._reset_compiler_state_before_generation()

        assert t.equal(compiled(t.ones(1)), expected)
        assert compile_count == 2
    finally:
        t.compiler.reset()


def test_prefitted_resume_still_resets_before_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    observed = _install_fakes(monkeypatch, kwargs)
    driver.run_encdec_baselines(**kwargs)

    events = observed["events"]
    fit_events_before = [
        event
        for event in events
        if event.startswith(("capture:", "fit-diffmean:", "fit-probe:", "fit-lora:"))
    ]
    generation_count_before = sum(event.startswith("generate:") for event in events)
    reset_count_before = events.count("compiler-reset")

    Path(kwargs["output_main_path"]).unlink()
    Path(kwargs["output_aux_path"]).unlink()
    Path(kwargs["manifest_path"]).unlink()
    resume_root = Path(kwargs["manifest_path"]).parent / "_encdec_cell_resume_v1"
    for batch_path in (resume_root / "rank_00000").glob("batch_*.parquet"):
        batch_path.unlink()
    (resume_root / "tripwire.json").unlink()

    driver.run_encdec_baselines(**kwargs)

    fit_events_after = [
        event
        for event in events
        if event.startswith(("capture:", "fit-diffmean:", "fit-probe:", "fit-lora:"))
    ]
    assert fit_events_after == fit_events_before
    assert events.count("compiler-reset") == reset_count_before + 1
    second_reset_index = len(events) - 1 - events[::-1].index("compiler-reset")
    second_generation_indices = [
        index
        for index, event in enumerate(events)
        if index > second_reset_index and event.startswith("generate:")
    ]
    assert len(second_generation_indices) == generation_count_before
    assert second_reset_index < min(second_generation_indices)


def test_compiler_reset_failure_prevents_every_generation_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    observed = _install_fakes(monkeypatch, kwargs)

    def fail_reset() -> None:
        observed["events"].append("compiler-reset-failed")
        raise RuntimeError("deliberate compiler reset failure")

    monkeypatch.setattr(driver.t.compiler, "reset", fail_reset)

    with pytest.raises(RuntimeError, match="deliberate compiler reset failure"):
        driver.run_encdec_baselines(**kwargs)

    assert observed["events"].count("compiler-reset-failed") == 1
    assert not [event for event in observed["events"] if event.startswith("generate:")]
    assert not Path(kwargs["output_main_path"]).exists()
    assert not Path(kwargs["output_aux_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


def test_invalid_reloaded_fit_cannot_install_receipt_or_begin_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    observed = _install_fakes(monkeypatch, kwargs)
    real_load = driver.load_direction

    def corrupt_width(path: Path) -> FittedDirection:
        loaded = real_load(path)
        if "diffmean" not in path.name:
            return loaded
        return FittedDirection(
            direction=t.ones(2, dtype=t.float32),
            max_activation=loaded.max_activation,
            metadata=loaded.metadata,
        )

    monkeypatch.setattr(driver, "load_direction", corrupt_width)

    with pytest.raises(driver.CellStoreError, match="violates the target-model contract"):
        driver.run_encdec_baselines(**kwargs)

    resume_root = Path(kwargs["manifest_path"]).parent / "_encdec_cell_resume_v1"
    assert not (resume_root / "fit_artifacts.json").exists()
    assert not [event for event in observed["events"] if event.startswith("generate:")]
    assert "compiler-reset" not in observed["events"]


@pytest.mark.parametrize("failing_loader", ["direction", "lora"])
def test_fit_artifact_loader_failures_are_labeled_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_loader: str,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    for path, _method, _concept_id in driver._expected_artifacts(train_dir, max_concepts=1):
        path.write_bytes(b"placeholder")
    fitted = FittedDirection(
        direction=t.ones(3, dtype=t.float32),
        max_activation=1.0,
        metadata={"method": "diffmean", "intervention_layer": 9},
    )

    def load_direction(path: Path) -> FittedDirection:
        if failing_loader == "direction":
            raise OSError("direction read failed")
        return FittedDirection(
            direction=fitted.direction,
            max_activation=fitted.max_activation,
            metadata={
                **fitted.metadata,
                "method": "probe" if "linear_probe" in path.name else "diffmean",
            },
        )

    def load_lora(_path: Path) -> tuple[dict[str, t.Tensor], dict[str, str]]:
        raise OSError("LoRA read failed")

    monkeypatch.setattr(driver, "load_direction", load_direction)
    monkeypatch.setattr(driver, "load_lora", load_lora)
    expected = (
        "could not load encoder-decoder direction artifacts"
        if failing_loader == "direction"
        else "could not load valid LoRA artifact"
    )
    with pytest.raises(driver.CellStoreError, match=expected):
        driver._load_fit_artifacts(
            train_dir=train_dir,
            max_concepts=1,
            seed=42,
            intervention_layer=9,
            lora_layers=(2, 5, 7, 9),
            fit_rows={0: {"positive": [cast(Any, object())]}},
            model=SimpleNamespace(dec_config=SimpleNamespace(n_embd=3)),
        )


def test_direction_collection_validator_rejects_missing_and_mislabeled_fits() -> None:
    valid = {
        key: FittedDirection(
            direction=t.ones(3, dtype=t.float32),
            max_activation=1.0,
            metadata={"method": key, "intervention_layer": 9},
        )
        for key in ("diffmean", "probe")
    }
    driver._validate_fitted_direction_layers(
        valid,
        concept_id=0,
        intervention_layer=9,
        expected_width=3,
    )

    with pytest.raises(driver.CellStoreError, match="direction set differs"):
        driver._validate_fitted_direction_layers(
            {"diffmean": valid["diffmean"]},
            concept_id=0,
            intervention_layer=9,
            expected_width=3,
        )
    mislabeled = {
        **valid,
        "probe": FittedDirection(
            direction=valid["probe"].direction,
            max_activation=1.0,
            metadata={"method": "wrong", "intervention_layer": 9},
        ),
    }
    with pytest.raises(driver.CellStoreError, match="metadata has method='wrong'"):
        driver._validate_fitted_direction_layers(
            mislabeled,
            concept_id=0,
            intervention_layer=9,
            expected_width=3,
        )


def test_encdec_fit_phase_has_one_exact_owned_concept_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    observed: list[tuple[str, dict[str, Any]]] = []

    @contextmanager
    def record_region(name: str, **attrs: Any) -> Iterator[None]:
        observed.append((name, attrs))
        yield

    monkeypatch.setattr(driver, "profile_region", record_region)

    driver.run_encdec_baselines(**kwargs)

    assert [item for item in observed if item[0] == "axbench_upstream.encdec.concept_fits"] == [
        ("axbench_upstream.encdec.concept_fits", {"owned_concepts": 2})
    ]
    assert [
        item for item in observed if item[0] == "axbench_upstream.encdec.compiler_phase_reset"
    ] == [("axbench_upstream.encdec.compiler_phase_reset", {})]


def test_interleaved_main_restores_both_frames_to_main_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    main = pd.read_parquet(kwargs["main_path"])
    shuffled = main.sample(frac=1.0, random_state=17).reset_index(drop=True)
    shuffled.to_parquet(kwargs["main_path"], index=False)
    _install_fakes(monkeypatch, kwargs)

    encdec_main, encdec_aux = driver.run_encdec_baselines(**kwargs)

    pd.testing.assert_frame_equal(
        encdec_main.loc[:, list(INPUT_COLUMNS)],
        shuffled.loc[:, list(INPUT_COLUMNS)],
    )
    expected_aux = shuffled.loc[shuffled["factor"].eq(1.0), list(INPUT_COLUMNS)].copy()
    expected_aux["steered_input"] = pd.Series(
        [
            f"steer-{int(row.concept_id)}{QUESTION_MARKER}{row.original_prompt}"
            for row in expected_aux.itertuples(index=False)
        ],
        index=expected_aux.index,
        dtype="object",
    )
    pd.testing.assert_frame_equal(
        encdec_aux.loc[:, list(INPUT_COLUMNS)],
        expected_aux,
    )


def test_template_prefix_is_checked_before_any_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    state = _install_fakes(monkeypatch, kwargs)

    driver.run_encdec_baselines(**kwargs)

    events = state["events"]
    assert events.index("get-tokenizer") < events.index("template-prefix")
    assert events.index("template-prefix") < events.index("load-setup")
    assert events.index("template-prefix") < events.index("load-extraction")
    assert events.index("template-prefix") < events.index("load-target")


def test_invalid_checkpoint_geometry_precedes_resume_store_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)

    def reject_setup(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("invalid turn checkpoint cadence")

    monkeypatch.setattr(driver, "load_method_setup", reject_setup)
    monkeypatch.setattr(
        driver,
        "CellResultStore",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid checkpoint geometry reached resume-store construction"
        ),
    )

    with pytest.raises(ValueError, match="invalid turn checkpoint cadence"):
        driver.run_encdec_baselines(**kwargs)

    assert not (Path(kwargs["manifest_path"]).parent / "_encdec_cell_resume_v1").exists()


def test_target_sequence_len_rejects_nonpositive_metadata_value() -> None:
    setup = SimpleNamespace(meta={"model_config": {"sequence_len": 0}})

    with pytest.raises(
        ValueError,
        match=_exact("target metadata model_config.sequence_len must be a positive integer"),
    ):
        driver._target_sequence_len(cast(Any, setup))


def test_prepared_context_rejects_duplicate_segment_feature_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = SimpleNamespace(n_sae_features=8, feature_vocab=None)
    model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=64))
    prepared = driver.PreparedPromptContext(
        token_ids=(1, 2),
        assistant_end_id=9,
        segments=(ContextFeatureSegment((1, 1), (0, 1), 0, 2),),
    )
    monkeypatch.setattr(driver, "assert_prepared_prompt_matches", lambda *_args: None)

    with pytest.raises(
        ValueError,
        match=_exact("prepared prompt segment 0 contains duplicate feature IDs"),
    ):
        driver._validate_prepared_context(
            cast(Any, setup),
            model,
            object(),
            prepared,
            "prompt",
            max_new_tokens=8,
        )


def test_driver_rejects_wrong_template_prefix_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    monkeypatch.setattr(driver, "template_prefix_length", lambda _tokenizer: 3)

    with pytest.raises(
        ValueError,
        match=_exact("AxBench encoder-decoder baseline template prefix length must equal 2, got 3"),
    ):
        driver.run_encdec_baselines(**kwargs)


def test_driver_rejects_loaded_target_sequence_length_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    mismatched_model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=63))
    monkeypatch.setattr(
        driver,
        "load_target_model",
        lambda *_args, **_kwargs: (mismatched_model, object()),
    )

    with pytest.raises(
        ValueError,
        match=_exact("loaded target sequence length differs from target metadata: 63 != 64"),
    ):
        driver.run_encdec_baselines(**kwargs)


@pytest.mark.parametrize("source", ["main", "aux"])
def test_structural_input_validation_precedes_any_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    source_path = Path(kwargs[f"{source}_path"])
    frame = pd.read_parquet(source_path).drop(columns=["input"])
    frame.to_parquet(source_path, index=False)
    monkeypatch.setattr(
        driver,
        "load_method_setup",
        lambda *_a, **_k: pytest.fail("invalid frame reached model setup"),
    )

    with pytest.raises(ValueError, match=f"{source} frame columns"):
        driver.run_encdec_baselines(**kwargs)


def test_prepared_manifest_rejection_precedes_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    manifest_path = Path(kwargs["prepared_dir"]) / "prepare_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": {"revision": "dataset-revision"},
                "axbench_upstream_commit": "wrong",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        driver,
        "load_method_setup",
        lambda *_a, **_k: pytest.fail("invalid manifest reached model setup"),
    )

    with pytest.raises(ValueError, match="vendored snapshot"):
        driver.run_encdec_baselines(**kwargs)


def test_concept_name_mismatch_precedes_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    training = _training()
    training[0] = ConceptTrainingPairs(
        concept_id=0,
        concept="wrong-concept",
        genre="text",
        positive=training[0].positive,
        negative=training[0].negative,
    )
    monkeypatch.setattr(driver, "select_training_pairs", lambda *_a, **_k: training)
    monkeypatch.setattr(
        driver,
        "load_method_setup",
        lambda *_a, **_k: pytest.fail("concept mismatch reached model setup"),
    )

    with pytest.raises(ValueError, match="concept_id=0.*wrong-concept.*concept-0"):
        driver.run_encdec_baselines(**kwargs)


def test_steered_prompt_recovery_failure_preserves_cell_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    aux = pd.read_parquet(kwargs["aux_path"])
    selected = aux["concept_id"].eq(0) & aux["input_id"].eq(3)
    aux.loc[selected, "steered_input"] = "tampered"
    aux.to_parquet(kwargs["aux_path"], index=False)
    monkeypatch.setattr(driver, "select_training_pairs", lambda *_a, **_k: _training())

    with pytest.raises(
        ValueError,
        match="concept 0 input 3: steered_input does not have the exact Qwen prefix",
    ):
        driver.run_encdec_baselines(**kwargs)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("max_concepts", True, "max_concepts must be a positive integer"),
        ("device", "", "device must be a nonempty string"),
        ("seed", -1, "seed must be a nonnegative integer"),
        ("seed", 2**63, "seed must be a nonnegative integer in \\[0, 2\\^63 - 1\\]"),
        ("max_new_tokens", 0, "max_new_tokens must be a positive integer"),
        ("force", 1, "force must be a boolean"),
    ],
)
def test_scalar_config_closure_precedes_artifact_io(
    tmp_path: Path,
    key: str,
    value: object,
    match: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs[key] = value
    with pytest.raises(ValueError, match=match):
        driver.run_encdec_baselines(**kwargs)


_READ_PATH_NAMES = (
    "main_path",
    "main_source_manifest",
    "aux_path",
    "aux_source_manifest",
    "prepared_dir",
    "prepared_manifest",
    "prepared_metadata",
    "prepared_training_data",
    "target_meta_path",
    "feature_statistics_path",
)
_WRITE_PATH_NAMES = (
    "output_main_path",
    "output_aux_path",
    "rejected main output path",
    "rejected aux output path",
    "manifest_path",
    "train_dir",
)


def _resolved_path_maps(kwargs: dict[str, Any]) -> tuple[dict[str, Path], dict[str, Path]]:
    prepared_dir = Path(kwargs["prepared_dir"])
    output_main = Path(kwargs["output_main_path"])
    output_aux = Path(kwargs["output_aux_path"])
    reads = {
        "main_path": Path(kwargs["main_path"]),
        "main_source_manifest": Path(kwargs["main_path"]).parent.parent / "run_manifest.json",
        "aux_path": Path(kwargs["aux_path"]),
        "aux_source_manifest": Path(kwargs["aux_path"]).parent.parent / "run_manifest.json",
        "prepared_dir": prepared_dir,
        "prepared_manifest": prepared_dir / "prepare_manifest.json",
        "prepared_metadata": prepared_dir / "generate" / "metadata.jsonl",
        "prepared_training_data": prepared_dir / "generate" / "train_data.parquet",
        "target_meta_path": Path(kwargs["target_meta_path"]),
        "feature_statistics_path": Path(kwargs["feature_statistics_path"]),
    }
    writes = {
        "output_main_path": output_main,
        "output_aux_path": output_aux,
        "rejected main output path": output_main.with_suffix(".rejected.parquet"),
        "rejected aux output path": output_aux.with_suffix(".rejected.parquet"),
        "manifest_path": Path(kwargs["manifest_path"]),
        "train_dir": Path(kwargs["train_dir"]),
    }
    return reads, writes


@pytest.mark.parametrize(
    ("write_name", "read_name"),
    [(write_name, read_name) for write_name in _WRITE_PATH_NAMES for read_name in _READ_PATH_NAMES],
)
def test_every_write_read_path_alias_pair_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_name: str,
    read_name: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    reads, writes = _resolved_path_maps(kwargs)
    read_value = reads[read_name]
    if write_name == "rejected main output path":
        main_output = Path(kwargs["output_main_path"])
        aux_output = Path(kwargs["output_aux_path"])
        monkeypatch.setattr(
            driver,
            "_rejected_output_path",
            lambda output: (
                read_value if output == main_output else aux_output.with_suffix(".rejected.parquet")
            ),
        )
    elif write_name == "rejected aux output path":
        main_output = Path(kwargs["output_main_path"])
        aux_output = Path(kwargs["output_aux_path"])
        monkeypatch.setattr(
            driver,
            "_rejected_output_path",
            lambda output: (
                read_value if output == aux_output else main_output.with_suffix(".rejected.parquet")
            ),
        )
    else:
        kwargs[write_name] = read_value
    with pytest.raises(ValueError, match="must not alias"):
        driver.run_encdec_baselines(**kwargs)
    del writes


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        (_WRITE_PATH_NAMES[first], _WRITE_PATH_NAMES[second])
        for first in range(len(_WRITE_PATH_NAMES))
        for second in range(first + 1, len(_WRITE_PATH_NAMES))
    ],
)
def test_every_write_write_path_alias_pair_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_name: str,
    second_name: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _reads, writes = _resolved_path_maps(kwargs)
    first_value = writes[first_name]
    canonical_keys = {
        "output_main_path",
        "output_aux_path",
        "manifest_path",
        "train_dir",
    }
    rejected_names = {"rejected main output path", "rejected aux output path"}
    if first_name in canonical_keys and second_name in canonical_keys:
        kwargs[second_name] = first_value
    else:
        collision = tmp_path / "shared-write-collision"
        if first_name in canonical_keys:
            kwargs[first_name] = collision
        if second_name in canonical_keys:
            kwargs[second_name] = collision
        main_output = Path(kwargs["output_main_path"])

        def rejected(output: Path) -> Path:
            name = (
                "rejected main output path" if output == main_output else "rejected aux output path"
            )
            return (
                collision
                if name in {first_name, second_name}
                else output.with_suffix(".rejected.parquet")
            )

        assert rejected_names & {first_name, second_name}
        monkeypatch.setattr(driver, "_rejected_output_path", rejected)
    with pytest.raises(ValueError, match="must not alias"):
        driver.run_encdec_baselines(**kwargs)


@pytest.mark.parametrize("collision", ["output_inside_train", "train_is_output_parent"])
def test_train_dir_containment_aliases_raise(tmp_path: Path, collision: str) -> None:
    kwargs = _write_inputs(tmp_path)
    train_dir = Path(kwargs["train_dir"])
    if collision == "output_inside_train":
        kwargs["output_main_path"] = train_dir / "concept_00000_diffmean.safetensors"
    else:
        kwargs["train_dir"] = Path(kwargs["output_main_path"]).parent

    with pytest.raises(ValueError, match="train_dir must not contain or be contained"):
        driver.run_encdec_baselines(**kwargs)


def test_train_dir_may_share_output_parent_without_containment(tmp_path: Path) -> None:
    kwargs = _write_inputs(tmp_path)
    output_main = Path(kwargs["output_main_path"]).resolve()
    kwargs["train_dir"] = output_main.parent / "train"

    rejected_main, rejected_aux = driver._validate_artifact_paths(
        main_path=Path(kwargs["main_path"]).resolve(),
        aux_path=Path(kwargs["aux_path"]).resolve(),
        prepared_dir=Path(kwargs["prepared_dir"]).resolve(),
        output_main_path=output_main,
        output_aux_path=Path(kwargs["output_aux_path"]).resolve(),
        manifest_path=Path(kwargs["manifest_path"]).resolve(),
        train_dir=Path(kwargs["train_dir"]).resolve(),
        target_meta_path=Path(kwargs["target_meta_path"]).resolve(),
        feature_statistics_path=Path(kwargs["feature_statistics_path"]).resolve(),
    )

    assert rejected_main == output_main.with_suffix(".rejected.parquet")
    assert rejected_aux == Path(kwargs["output_aux_path"]).resolve().with_suffix(
        ".rejected.parquet"
    )


@pytest.mark.parametrize("gated_key", ["output_main_path", "output_aux_path", "manifest_path"])
def test_force_semantics_cover_each_gated_path(tmp_path: Path, gated_key: str) -> None:
    kwargs = _write_inputs(tmp_path)
    gated_path = Path(kwargs[gated_key])
    gated_path.parent.mkdir(parents=True, exist_ok=True)
    gated_path.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="pass force=true"):
        driver.run_encdec_baselines(**kwargs)
    assert gated_path.read_bytes() == b"keep"


def test_train_dir_must_be_absent_or_empty(tmp_path: Path) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["force"] = True
    train_dir = Path(kwargs["train_dir"])
    train_dir.mkdir(parents=True)
    (train_dir / "old.safetensors").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="train output directory must be empty"):
        driver.run_encdec_baselines(**kwargs)


@pytest.mark.parametrize("gated_key", ["output_main_path", "output_aux_path", "manifest_path"])
def test_force_replaces_each_gated_path_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gated_key: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["force"] = True
    gated_path = Path(kwargs[gated_key])
    gated_path.parent.mkdir(parents=True, exist_ok=True)
    gated_path.write_bytes(b"old")
    _install_fakes(monkeypatch, kwargs)

    driver.run_encdec_baselines(**kwargs)

    assert gated_path.read_bytes() != b"old"


def test_lora_optimizer_steps_and_scheduler_horizon_use_real_step_count() -> None:
    metadata = driver._lora_training_metadata(
        layers=(2, 5, 7, 9), base_seed=42, concept_id=3, n_rows=43
    )

    assert metadata["layers"] == [2, 5, 7, 9]
    assert metadata["optimizer_steps"] == driver.LORA_EPOCHS * 2
    assert metadata["scheduler_horizon"] == driver.LORA_EPOCHS * 2


def test_driver_declared_grid_is_encdec_diffmean_method_factors() -> None:
    assert driver.METHOD_FACTORS["EncDecDiffMean"] is METHOD_FACTORS["EncDecDiffMean"]


def _generations(count: int, blanks: int, *, length: int = 40) -> list[str]:
    return [""] * blanks + ["x" * length] * (count - blanks)


def test_all_four_first_concept_tripwires_keep_blank_floor_and_median_check() -> None:
    driver._assert_first_concept_not_collapsed(
        {
            method: _generations(120 if method in driver.ENCDEC_MAIN_METHODS else 10, 4)
            for method in (*driver.ENCDEC_MAIN_METHODS, *driver.ENCDEC_AUX_METHODS)
        },
        0,
    )
    for method in (*driver.ENCDEC_MAIN_METHODS, *driver.ENCDEC_AUX_METHODS):
        count = 120 if method in driver.ENCDEC_MAIN_METHODS else 10
        with pytest.raises(ValueError, match=f"encoder-decoder baseline method {method!r}"):
            driver._assert_first_concept_not_collapsed(
                {method: _generations(count, 5)},
                0,
                label="encoder-decoder baseline method",
            )
        with pytest.raises(ValueError, match=r"median stripped length is 1"):
            driver._assert_first_concept_not_collapsed({method: ["x"] * count}, 0)


def test_first_concept_collapse_prevents_every_bulk_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs, collapsed_method="EncDecLoRA")

    with pytest.raises(ValueError, match="EncDecLoRA.*concept_id=0.*collapse detected"):
        driver.run_encdec_baselines(**kwargs)

    generation_events = [event for event in observed["events"] if event.startswith("generate:")]
    assert generation_events
    assert all(event.endswith("c0") for event in generation_events)
    assert not Path(kwargs["output_main_path"]).exists()
    assert not Path(kwargs["output_aux_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


@pytest.mark.parametrize(
    "identity_input",
    [
        "model_000007.pt",
        "prepared/generate/metadata.jsonl",
        "prepared/generate/train_data.parquet",
    ],
)
def test_resume_identity_rejects_mutated_scientific_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_input: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    driver.run_encdec_baselines(**kwargs)
    path = tmp_path / identity_input
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    kwargs["force"] = True

    with pytest.raises(driver.CellStoreError, match="differs from the current result"):
        driver.run_encdec_baselines(**kwargs)


@pytest.mark.parametrize(
    ("collapsed_method", "source", "output_key"),
    [
        ("EncDecDiffMean", "encdec_main", "output_main_path"),
        ("EncDecPromptSteering", "encdec_aux", "output_aux_path"),
    ],
)
def test_invalid_frame_writes_its_own_rejected_parquet_and_note(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collapsed_method: str,
    source: str,
    output_key: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs, collapsed_method=collapsed_method)
    monkeypatch.setattr(driver, "_assert_first_concept_not_collapsed", lambda *_a, **_k: None)
    rejected_path = Path(kwargs[output_key]).with_suffix(".rejected.parquet")

    with pytest.raises(ValueError, match="blank generations") as caught:
        driver.run_encdec_baselines(**kwargs)

    assert rejected_path.is_file()
    assert any(
        f"complete rejected {source} frame written" in note for note in caught.value.__notes__
    )
    assert not Path(kwargs["manifest_path"]).exists()


def test_failure_after_first_parquet_leaves_no_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    real_write = driver.atomic_write_parquet
    calls = 0

    def fail_second(table: Any, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second publication failure")
        real_write(table, path)

    monkeypatch.setattr(driver, "atomic_write_parquet", fail_second)
    with pytest.raises(OSError, match="second publication"):
        driver.run_encdec_baselines(**kwargs)
    assert Path(kwargs["output_main_path"]).is_file()
    assert not Path(kwargs["output_aux_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


def test_force_removes_stale_manifest_before_second_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["force"] = True
    output_main = Path(kwargs["output_main_path"])
    output_aux = Path(kwargs["output_aux_path"])
    manifest_path = Path(kwargs["manifest_path"])
    for path, payload in (
        (output_main, b"old-main"),
        (output_aux, b"old-aux"),
        (manifest_path, b"old-complete-manifest"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _install_fakes(monkeypatch, kwargs)
    real_write = driver.atomic_write_parquet
    calls = 0

    def fail_second(table: Any, path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated forced aux publication failure")
        real_write(table, path)

    monkeypatch.setattr(driver, "atomic_write_parquet", fail_second)

    with pytest.raises(OSError, match="forced aux publication failure"):
        driver.run_encdec_baselines(**kwargs)

    assert not manifest_path.exists()
    assert output_main.read_bytes() != b"old-main"
    assert output_aux.read_bytes() == b"old-aux"


def test_dictconfig_forwards_exact_closed_key_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    raw = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        driver,
        "run_encdec_baselines",
        lambda **received: captured.update(received) or (pd.DataFrame(), pd.DataFrame()),
    )

    driver.run_encdec_baselines_from_dictconfig(OmegaConf.create(raw))

    assert captured == kwargs
    for key in (
        "main_path",
        "aux_path",
        "prepared_dir",
        "output_main_path",
        "output_aux_path",
        "manifest_path",
        "train_dir",
        "target_meta_path",
        "feature_statistics_path",
    ):
        assert isinstance(captured[key], Path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda config: config.pop("force"), r"missing=\['force'\]"),
        (lambda config: config.update(top_k=50), r"unknown=\['top_k'\]"),
        (
            lambda config: config.update(extraction_batch_size=8),
            r"unknown=\['extraction_batch_size'\]",
        ),
    ],
)
def test_dictconfig_rejects_missing_and_unknown_keys(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    raw = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    mutation(raw)
    with pytest.raises(ValueError, match=match):
        driver.run_encdec_baselines_from_dictconfig(OmegaConf.create(raw))


def test_dictconfig_requires_string_key_mapping() -> None:
    with pytest.raises(ValueError, match="config must be a mapping"):
        driver.run_encdec_baselines_from_dictconfig(
            cast(Any, OmegaConf.create(["not", "a", "mapping"]))
        )
    with pytest.raises(ValueError, match="config keys must be strings"):
        driver.run_encdec_baselines_from_dictconfig(cast(Any, OmegaConf.create({1: "value"})))


def test_default_hydra_config_matches_c2_contract() -> None:
    config_dir = _REPO_ROOT / "configs" / "eval" / "axbench_upstream"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="encdec_baselines")
    assert cfg.main_path.endswith(
        "05f03dede-c2-retry-20260811/main/inference/steering_data.parquet"
    )
    assert cfg.aux_path.endswith("05f03dede-c2-retry-20260811/aux/inference/steering_data.parquet")
    assert (
        cfg.output_main_path == "out/axbench/runs/c2/encdec_v1/main/inference/steering_data.parquet"
    )
    assert (
        cfg.output_aux_path == "out/axbench/runs/c2/encdec_v1/aux/inference/steering_data.parquet"
    )
    assert cfg.max_concepts == 2
    assert cfg.device == "cuda"
    assert cfg.seed == 42
    assert cfg.max_new_tokens == 128
    assert cfg.force is False


def test_hydra_shim_forwards_to_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.eval.axbench_upstream import run_encdec_baselines as shim

    cfg = OmegaConf.create({"sentinel": "config"})
    captured: list[Any] = []
    monkeypatch.setattr(shim, "run_encdec_baselines_from_dictconfig", captured.append)

    shim.main.__wrapped__(cfg)

    assert captured == [cfg]


# Deep enough that the vendored-fraction layer resolution (see
# layer_resolution.py) gives EncDecDiffMean/EncDecLoRA distinct, in-range
# layers; the shared TINY_N_LAYER=2 decoder is too shallow for that.
_DRIVER_FP_N_LAYER = 6


def _build_driver_fp_encdec() -> EncoderDecoderGPT:
    model = EncoderDecoderGPT(
        dec_config=GPTConfig(
            sequence_len=TINY_SEQ_LEN,
            vocab_size=TINY_VOCAB,
            n_layer=_DRIVER_FP_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            n_embd=TINY_N_EMBD,
            window_pattern="L",
        ),
        enc_config=EncoderConfig(
            n_embd=TINY_N_EMBD,
            depth=_DRIVER_FP_N_LAYER,
            n_head=TINY_N_HEAD,
            n_kv_head=TINY_N_KV_HEAD,
            head_dim=TINY_HEAD_DIM,
        ),
        xattn_config=CrossAttentionConfig(frequency=1),
        n_sae_features=PositiveInt.parse(TINY_N_SAE_FEATURES),
        num_buckets=PositiveInt.parse(TINY_NUM_BUCKETS),
        pad_vocab_size_to=TINY_VOCAB,
        feature_pred_config=FeaturePredictionConfig(
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
        ),
    )
    model.init_weights()
    model.eval()
    return model


def _tiny_method_setup(kwargs: dict[str, Any]) -> Any:
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        delta_weights=DeltaSamplingWeights.percentile_delta,
        min_k=1,
        max_k=2,
        mode=SamplingMode.top_k_percentile,
    )
    negative = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=1.0,
        std=1.0,
        min_k=0,
        max_k=2,
    )
    return _dispatch_setup(
        target_meta_path=Path(kwargs["target_meta_path"]).resolve(),
        meta={
            "step": 7,
            "model_config": {"sequence_len": TINY_SEQ_LEN, "n_layer": _DRIVER_FP_N_LAYER},
            "user_config": {"model_tag": "tiny-fp-encdec"},
        },
        n_sae_features=TINY_N_SAE_FEATURES,
        feature_vocab=None,
        feature_mask=None,
        consolidation=None,
        percentile_table=t.zeros((TINY_N_SAE_FEATURES, 99), dtype=t.float64),
        output_selection=selection,
        output_negative_selection=negative,
    )


def test_tiny_encdec_real_fit_and_generation_factor_zero_matches_no_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise real capture, fits, LoRA, deltas, and self-conditioned decode.

    Only extraction-facing functions are replaced with deterministic real-
    feature stand-ins; the target is a tiny FP-head encoder-decoder using the
    shared conftest dimensions.
    """

    t.manual_seed(123)
    kwargs = _write_inputs(tmp_path, max_concepts=1)
    kwargs["max_new_tokens"] = 1
    main = pd.read_parquet(kwargs["main_path"])
    prompts = {input_id: chr(ord("a") + input_id) for input_id in range(10)}
    main["original_prompt"] = [prompts[int(value)] for value in main["input_id"]]
    main.to_parquet(kwargs["main_path"], index=False)
    aux = _aux_frame(main)
    aux["steered_input"] = [
        QWEN_STEERED_PREFIX + "s" + QUESTION_MARKER + str(row.original_prompt) + QWEN_STEERED_SUFFIX
        for row in aux.itertuples(index=False)
    ]
    aux.to_parquet(kwargs["aux_path"], index=False)

    tokenizer = MockTokenizer()
    model = _build_driver_fp_encdec()
    setup = _tiny_method_setup(kwargs)
    training = {
        0: ConceptTrainingPairs(
            concept_id=0,
            concept="concept-0",
            genre="text",
            positive=(("p", "y"),),
            negative=(("n", "z"),),
        )
    }

    def prepare(
        _setup: Any,
        _extraction_model: Any,
        _sae: Any,
        active_tokenizer: MockTokenizer,
        *,
        token_ids: tuple[int, ...],
        assistant_end_id: int,
    ) -> driver.PreparedPromptContext:
        exact = tuple(token_ids)
        decoded = active_tokenizer.decode(list(exact))
        return driver.PreparedPromptContext(
            token_ids=exact,
            assistant_end_id=assistant_end_id,
            segments=(ContextFeatureSegment((2, 5), (0, 1), 0, len(decoded)),),
        )

    def condition(
        _setup: Any,
        _extraction_model: Any,
        _sae: Any,
        _tokenizer: Any,
        *,
        token_ids: tuple[int, ...],
        surface: str,
    ) -> FitRowConditioning:
        del surface
        exact = tuple(token_ids)
        return FitRowConditioning(
            token_ids=exact,
            segments=(ContextFeatureSegment((2, 5), (0, 1), 0, len(exact)),),
            segment_assignments=(0,) * len(exact),
        )

    real_validate = driver.validate_steering_frame

    def validate_intake_only(frame: pd.DataFrame, **options: Any) -> None:
        if options.get("check_method_values") is False:
            real_validate(frame, **options)

    monkeypatch.setattr(_fa_mod, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(driver, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        driver,
        "_tokenizer_artifact_identity",
        lambda: {"name": "tokenizer.pkl", "byte_length": 1, "sha256": "a" * 64},
    )
    monkeypatch.setattr(driver, "select_training_pairs", lambda *_a, **_k: training)
    monkeypatch.setattr(driver, "load_method_setup", lambda *_a, **_k: setup)
    monkeypatch.setattr(driver, "load_extraction_stack", lambda *_a, **_k: (object(), object()))
    monkeypatch.setattr(driver, "prepare_prompt_context", prepare)
    monkeypatch.setattr(driver, "build_fit_row_conditioning", condition)
    monkeypatch.setattr(driver, "load_target_model", lambda *_a, **_k: (model, tokenizer))
    monkeypatch.setattr(driver, "_assert_first_concept_not_collapsed", lambda *_a, **_k: None)
    monkeypatch.setattr(driver, "validate_steering_frame", validate_intake_only)

    encdec_main, encdec_aux = driver.run_encdec_baselines(**kwargs)

    assert encdec_main.shape == (120, 18)
    assert encdec_aux.shape == (10, 18)
    factor_zero = encdec_main.loc[
        encdec_main["input_id"].eq(0) & encdec_main["factor"].eq(0.0)
    ].iloc[0]
    rendered = render_prompt_for_chat_completion(tokenizer, prompts[0])
    prepared = prepare(
        setup,
        object(),
        object(),
        tokenizer,
        token_ids=tuple(rendered.token_ids),
        assistant_end_id=rendered.assistant_end_id,
    )
    no_hook = driver.generate_encdec_baseline(
        setup,
        model,
        tokenizer,
        prepared,
        selection_seed=driver.derive_selection_seed(kwargs["seed"], 0, 0),
        seed=driver.derive_generation_seed(
            kwargs["seed"],
            "EncDecDiffMean",
            0,
            0,
            0.0,
        ),
        max_new_tokens=kwargs["max_new_tokens"],
    )
    assert factor_zero.EncDecDiffMean_steered_generation == no_hook
