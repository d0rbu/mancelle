"""Standalone DiffMean, LinearProbe, and LoRA for a native RustBPE decoder.

These are ordinary positive steering baselines.  They render the canonical
AxBench training pairs with the checkpoint's chat tokenizer, fit only on the
TRAIN split, and generate from the original AlpacaEval user prompts.  The
AxBench-Adv native counter path is intentionally not reused: its saved vectors
are calibrated for raw negative attacks, whereas the direction methods here
apply ``factor * max_activation * unit_direction``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import pyarrow as pa
import torch as t
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, INPUT_COLUMNS
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.native_mailbox import digest_json, read_json, write_json
from steering.eval.axbench_base.baselines.hooks import _get_block, register_residual_hook
from steering.eval.axbench_upstream.data import ConceptTrainingPairs, select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
    load_direction,
    save_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.layer_resolution import (
    VENDORED_LORA_LAYERS,
    VENDORED_STEERING_LAYER,
    resolve_vendored_layer,
    resolve_vendored_layers,
)
from steering.eval.axbench_upstream.encdec_baselines.lora import (
    LORA_ALPHA,
    LORA_BATCH_ROWS,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_GRAD_ACCUM,
    LORA_LR,
    LORA_RANK,
    LORA_SEED_DOMAIN,
    LORA_WEIGHT_DECAY,
    _derive_lora_seed,
    apply_lora_state,
    attach_encdec_lora,
    detach_encdec_lora,
    load_lora,
    save_lora,
    validate_lora_state,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    RenderedFitRow,
    render_fit_row,
    targets_from_mask,
    template_prefix_length,
)
from steering.eval.axbench_upstream.native_contract import sha256
from steering.eval.axbench_upstream.native_decoder import (
    NativeDecoderAdapter,
    canonical_bytes,
    checkpoint_identity,
    native_runtime_settings,
)
from steering.immutable_io import install_bytes_once

FORMAT = "axbench-native-decoder-baseline-v1"
METHODS = ("DiffMean", "LinearProbe", "LoRA")
DIRECTION_METHODS = ("DiffMean", "LinearProbe")
METHOD_FACTORS: Mapping[str, tuple[float, ...]] = {
    "DiffMean": FULL_FACTORS,
    "LinearProbe": FULL_FACTORS,
    "LoRA": (1.0,),
}
METHOD_KIND = {"DiffMean": "diffmean", "LinearProbe": "linear_probe", "LoRA": "lora"}
CAPTURE_SITE = "decoder_block_output"
CAPTURE_PREFIX_LENGTH = 2
INTERVENTION_LAYER = 9
LORA_LAYERS = (2, 5, 7, 9)
_SEED_DOMAIN = "axbench-native-decoder-standalone-v1"
_SEED_MODULUS = 2**31 - 1


@dataclass(frozen=True)
class NativeModelContract:
    """One reviewed native decoder geometry and its depth-relative intervention sites."""

    n_layer: int
    n_embd: int
    sequence_len: int
    intervention_layer: int
    lora_layers: tuple[int, ...]
    parameter_count: int | None


def _reviewed_model_contract(
    n_layer: int, n_embd: int, *, parameter_count: int | None = None
) -> NativeModelContract:
    return NativeModelContract(
        n_layer=n_layer,
        n_embd=n_embd,
        sequence_len=1024,
        intervention_layer=resolve_vendored_layer(
            VENDORED_STEERING_LAYER, target_n_layer=n_layer
        ),
        lora_layers=resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=n_layer),
        parameter_count=parameter_count,
    )


LEGACY_MODEL_CONTRACT = _reviewed_model_contract(12, 768)
D14_MODEL_CONTRACT = _reviewed_model_contract(14, 896, parameter_count=399_115_836)
REVIEWED_MODEL_CONTRACTS = {
    (contract.n_layer, contract.n_embd, contract.sequence_len): contract
    for contract in (LEGACY_MODEL_CONTRACT, D14_MODEL_CONTRACT)
}


def resolve_native_model_contract(geometry: Mapping[str, int]) -> NativeModelContract:
    """Resolve only an explicitly reviewed native checkpoint geometry."""

    key = tuple(geometry.get(field) for field in ("n_layer", "n_embd", "sequence_len"))
    contract = REVIEWED_MODEL_CONTRACTS.get(cast(tuple[int, int, int], key))
    if contract is None:
        raise AxBenchAdvError(
            "native standalone checkpoint geometry is not reviewed: "
            f"n_layer={key[0]!r}, n_embd={key[1]!r}, sequence_len={key[2]!r}"
        )
    return contract


class NativeBaselineAdapter(Protocol):
    model: Any
    tokenizer: Any

    def generate_user_text(
        self,
        user_text: str,
        *,
        seed: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str: ...


def _strict_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AxBenchAdvError(f"{label} must be a string-keyed mapping")
    return dict(value)


def _metadata_geometry(checkpoint_dir: Path, step: int) -> dict[str, int]:
    path = checkpoint_dir / f"meta_{step:06d}.json"
    try:
        document = json.loads(path.read_text())
        config = document["model_config"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AxBenchAdvError(f"invalid native checkpoint metadata: {path}") from exc
    result: dict[str, int] = {}
    for key in ("n_layer", "n_embd", "sequence_len"):
        value = config.get(key)
        if type(value) is not int or value <= 0:
            raise AxBenchAdvError(f"native checkpoint model_config.{key} is invalid")
        result[key] = value
    return result


def _runtime_identity(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, int]]:
    if type(cfg.step) is not int:
        raise AxBenchAdvError("native standalone step must be an integer")
    checkpoint = checkpoint_identity(
        Path(str(cfg.checkpoint_dir)), int(cfg.step), Path(str(cfg.tokenizer_path))
    )
    geometry = _metadata_geometry(Path(str(cfg.checkpoint_dir)), int(cfg.step))
    expected = _strict_mapping(
        OmegaConf.to_container(cfg.expected, resolve=True), label="expected native identity"
    )
    if set(expected) != {
        "step", "model_sha256", "meta_sha256", "tokenizer_sha256", "n_layer", "n_embd"
    } or any(type(expected[key]) is not int for key in ("step", "n_layer", "n_embd")):
        raise AxBenchAdvError("expected native identity has invalid fields or types")
    observed = {
        "step": checkpoint["step"],
        "model_sha256": checkpoint["model_sha256"],
        "meta_sha256": checkpoint["meta_sha256"],
        "tokenizer_sha256": checkpoint["tokenizer_sha256"],
        "n_layer": geometry["n_layer"],
        "n_embd": geometry["n_embd"],
    }
    if observed != expected:
        raise AxBenchAdvError(f"native runtime checkpoint/tokenizer identity differs: {observed!r}")
    contract = resolve_native_model_contract(geometry)
    if type(cfg.layer) is not int or cfg.layer != contract.intervention_layer:
        raise AxBenchAdvError(
            "native standalone direction layer differs from the reviewed geometry mapping"
        )
    if any(type(value) is not int for value in cfg.lora_layers) or tuple(cfg.lora_layers) != contract.lora_layers:
        raise AxBenchAdvError(
            "native standalone LoRA layers differ from the reviewed geometry mapping"
        )
    return checkpoint, geometry


def _selected_methods(cfg: DictConfig) -> tuple[str, ...]:
    methods = tuple(str(value) for value in cfg.methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in METHODS for method in methods)
    ):
        raise AxBenchAdvError(f"methods must be a unique nonempty subset of {list(METHODS)}")
    return methods


def _selected_concepts(cfg: DictConfig) -> tuple[int, ...]:
    lo, hi = cfg.concept_lo, cfg.concept_hi
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= int(cfg.max_concepts):
        raise AxBenchAdvError(
            "concept_lo/concept_hi must select a nonempty range within max_concepts"
        )
    if int(cfg.max_concepts) != 100:
        raise AxBenchAdvError("the production native baseline source must declare standard100")
    return tuple(range(lo, hi))


def _source_frame(cfg: DictConfig) -> tuple[pd.DataFrame, str]:
    source_path = Path(str(cfg.source_path)).resolve()
    digest = sha256(source_path)
    if digest != str(cfg.source_sha256):
        raise AxBenchAdvError("frozen standard100 source SHA-256 mismatch")
    frame = _baseline_source_frame(pd.read_parquet(source_path), max_concepts=int(cfg.max_concepts))
    if sha256(source_path) != digest:
        raise AxBenchAdvError("frozen standard100 source changed while loading")
    return frame, digest


def _baseline_source_frame(frame: pd.DataFrame, *, max_concepts: int) -> pd.DataFrame:
    """Validate the completed standard grid without consuming its boosted prompt.

    The completed native PromptSteering artifact may retain the vendored model's
    full system/user rendering in ``steered_input``. Standalone baselines must
    neither parse nor regenerate that condition: their sole generation input is
    the byte-preserved ``original_prompt``.
    """

    if type(max_concepts) is not int or max_concepts != 100:
        raise AxBenchAdvError("native standalone source must declare standard100")
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise AxBenchAdvError(f"native standalone source is missing {sorted(missing)}")
    result = frame.loc[:, list(INPUT_COLUMNS)].copy().reset_index(drop=True)
    expected_keys = [
        (concept_id, input_id) for concept_id in range(max_concepts) for input_id in range(10)
    ]
    actual_keys = list(result[["concept_id", "input_id"]].itertuples(index=False, name=None))
    if (
        len(result) != max_concepts * 10
        or actual_keys != expected_keys
        or result.duplicated(["concept_id", "input_id"]).any()
        or not result["factor"].eq(1.0).all()
        or not result["dataset_name"].eq("AlpacaEval").all()
    ):
        raise AxBenchAdvError(
            "native standalone source must be the ordered standard100 factor-1 grid"
        )
    for column in ("input_concept", "original_prompt"):
        if (
            not result[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise AxBenchAdvError(f"native standalone source {column} must be nonblank text")
    if any(group["input_concept"].nunique() != 1 for _, group in result.groupby("concept_id")):
        raise AxBenchAdvError("native standalone source concept labels differ within a concept")
    if "raw_prompt" in frame and not frame["raw_prompt"].reset_index(drop=True).equals(
        result["original_prompt"]
    ):
        raise AxBenchAdvError("native standalone source raw_prompt differs from original_prompt")
    return result


def _assert_train_only_population(
    source: pd.DataFrame, training: Mapping[int, ConceptTrainingPairs]
) -> None:
    """Fence fitting to the prepared train-data population.

    A generic input string can legitimately occur in both independently
    prepared TRAIN and AlpacaEval test corpora. Population separation is by
    authenticated artifact and selection path, not by global text uniqueness.
    """

    expected_ids = set(range(100))
    if set(training) != expected_ids:
        raise AxBenchAdvError("native standalone TRAIN population must cover global IDs 0..99")
    source_concepts = source.groupby("concept_id", sort=False)["input_concept"].first()
    for concept_id, examples in training.items():
        if examples.concept_id != concept_id or source_concepts.get(concept_id) != examples.concept:
            raise AxBenchAdvError(
                f"native standalone TRAIN/source concept differs for global ID {concept_id}"
            )
        if not examples.positive or not examples.negative:
            raise AxBenchAdvError(
                f"native standalone TRAIN population is empty for global ID {concept_id}"
            )
        for pairs in (examples.positive, examples.negative):
            if any(
                not isinstance(input_text, str)
                or not isinstance(output_text, str)
                or not input_text.strip()
                or not output_text.strip()
                for input_text, output_text in pairs
            ):
                raise AxBenchAdvError(
                    f"native standalone TRAIN row is invalid for global ID {concept_id}"
                )


def _method_frame(source: pd.DataFrame, *, method: str, concepts: Sequence[int]) -> pd.DataFrame:
    selected = source.loc[source["concept_id"].isin(concepts), list(INPUT_COLUMNS)].copy()
    rows: list[dict[str, Any]] = []
    for record in selected.to_dict(orient="records"):
        for factor in METHOD_FACTORS[method]:
            row = dict(record)
            row["factor"] = factor
            # Carrier fields remain for AxBench schema compatibility.  Native
            # generation below consumes only this exact original user prompt.
            row["raw_prompt"] = row["original_prompt"]
            row["native_user_prompt"] = row["original_prompt"]
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["concept_id", "input_id", "factor"], kind="stable").reset_index(
        drop=True
    )
    expected = [
        (concept_id, input_id, factor)
        for concept_id in concepts
        for input_id in range(10)
        for factor in METHOD_FACTORS[method]
    ]
    actual = list(frame[["concept_id", "input_id", "factor"]].itertuples(index=False, name=None))
    if actual != expected or not frame["native_user_prompt"].equals(frame["original_prompt"]):
        raise AxBenchAdvError("native standalone frame is not the canonical original-prompt grid")
    return frame


def derive_cell_seed(
    base_seed: int, method: str, concept_id: int, input_id: int, factor: float
) -> int:
    if method not in METHODS:
        raise ValueError(f"unknown native standalone method {method!r}")
    fields = (
        _SEED_DOMAIN,
        str(base_seed),
        method,
        str(concept_id),
        str(input_id),
        float(factor).hex(),
    )
    return (
        int.from_bytes(hashlib.sha256("\x1f".join(fields).encode()).digest()[:8], "big")
        % _SEED_MODULUS
    )


def capture_native_post_prefix_activations(
    model: nn.Module,
    rendered: RenderedFitRow,
    *,
    layer: int,
    prefix_length: int,
) -> t.Tensor:
    """Capture one native decoder block output over the fit population."""

    model_config = cast(Any, model).config
    if type(layer) is not int or not 0 <= layer < int(model_config.n_layer):
        raise AxBenchAdvError("native capture layer is outside the decoder")
    if type(prefix_length) is not int or not 0 <= prefix_length < len(rendered.ids):
        raise AxBenchAdvError("native capture prefix length is outside the rendered fit row")
    device = next(model.parameters()).device
    idx = t.tensor([rendered.ids], dtype=t.int64, device=device)
    captured: list[t.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: t.Tensor) -> None:
        if not isinstance(output, t.Tensor) or output.ndim != 3:
            raise AxBenchAdvError("native decoder block output must have shape (B,T,C)")
        captured.append(output)

    handle = _get_block(model, layer).register_forward_hook(hook)
    try:
        with t.no_grad():
            model(idx)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise AxBenchAdvError(f"native fit capture expected one block output, got {len(captured)}")
    result = captured[0][0, prefix_length:, :].detach().to(dtype=t.float32, device="cpu")
    if result.ndim != 2 or result.shape[0] == 0:
        raise AxBenchAdvError("native fit capture produced no post-prefix activations")
    return result


def _capture_class(
    adapter: NativeBaselineAdapter,
    pairs: Sequence[tuple[str, str]],
    *,
    concept_id: int,
    class_name: str,
    layer: int,
) -> tuple[list[RenderedFitRow], list[t.Tensor]]:
    rendered_rows: list[RenderedFitRow] = []
    activations: list[t.Tensor] = []
    for row_index, (input_text, output_text) in enumerate(pairs):
        surface = f"native standalone concept {concept_id} {class_name} row {row_index}"
        rendered = render_fit_row(
            adapter.tokenizer,
            input_text=input_text,
            output_text=output_text,
            sequence_len=int(adapter.model.config.sequence_len),
            surface=surface,
        )
        rendered_rows.append(rendered)
        activations.append(
            capture_native_post_prefix_activations(
                adapter.model,
                rendered,
                layer=layer,
                prefix_length=CAPTURE_PREFIX_LENGTH,
            )
        )
    return rendered_rows, activations


def _render_rows(
    adapter: NativeBaselineAdapter,
    pairs: Sequence[tuple[str, str]],
    *,
    concept_id: int,
) -> tuple[RenderedFitRow, ...]:
    return tuple(
        render_fit_row(
            adapter.tokenizer,
            input_text=input_text,
            output_text=output_text,
            sequence_len=int(adapter.model.config.sequence_len),
            surface=f"native standalone concept {concept_id} LoRA TRAIN row {row_index}",
        )
        for row_index, (input_text, output_text) in enumerate(pairs)
    )


def _lora_metadata(
    *,
    contract: Mapping[str, Any],
    seed: int,
    concept_id: int,
    n_rows: int,
    lora_layers: Sequence[int] = LORA_LAYERS,
) -> dict[str, Any]:
    n_batches = math.ceil(n_rows / LORA_BATCH_ROWS)
    optimizer_steps = LORA_EPOCHS * math.ceil(n_batches / LORA_GRAD_ACCUM)
    return {
        "format": FORMAT,
        "contract_sha256": digest_json(contract),
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "scaling": LORA_ALPHA / math.sqrt(LORA_RANK),
        "dropout": LORA_DROPOUT,
        "layers": list(lora_layers),
        "seed": _derive_lora_seed(seed, concept_id),
        "seed_domain": LORA_SEED_DOMAIN,
        "n_rows": n_rows,
        "epochs": LORA_EPOCHS,
        "batch_rows": LORA_BATCH_ROWS,
        "gradient_accumulation_batches": LORA_GRAD_ACCUM,
        "learning_rate": LORA_LR,
        "weight_decay": LORA_WEIGHT_DECAY,
        "optimizer_steps": optimizer_steps,
        "scheduler_horizon": optimizer_steps,
        "population": "positive TRAIN rows only",
    }


def train_native_lora(
    model: nn.Module,
    rows: Sequence[RenderedFitRow],
    *,
    base_seed: int,
    concept_id: int,
    epochs: int = LORA_EPOCHS,
    layers: Sequence[int] = LORA_LAYERS,
) -> dict[str, t.Tensor]:
    """Train canonical rsLoRA on positive TRAIN completions with a frozen base."""

    if not rows or type(epochs) is not int or epochs <= 0:
        raise ValueError("native LoRA requires nonempty rows and positive integer epochs")
    derived_seed = _derive_lora_seed(base_seed, concept_id)
    device = next(model.parameters()).device
    cpu_rng = t.get_rng_state()
    device_rng = t.cuda.get_rng_state(device) if device.type == "cuda" else None
    entry_training = model.training
    base_parameters = list(model.named_parameters())
    entry_requires_grad = {
        id(parameter): parameter.requires_grad for _, parameter in base_parameters
    }
    refs = []
    succeeded = False
    try:
        t.random.default_generator.manual_seed(derived_seed)
        if device.type == "cuda":
            with t.cuda.device(device):
                t.cuda.manual_seed(derived_seed)
        shuffle = t.Generator(device="cpu").manual_seed(derived_seed)
        refs = attach_encdec_lora(model, layers)
        for _name, parameter in base_parameters:
            parameter.requires_grad_(False)
        model.train()
        parameters = [
            parameter for ref in refs for parameter in (ref.adapter.lora_A, ref.adapter.lora_B)
        ]
        optimizer = t.optim.AdamW(parameters, lr=LORA_LR, weight_decay=LORA_WEIGHT_DECAY)
        n_batches = math.ceil(len(rows) / LORA_BATCH_ROWS)
        horizon = epochs * max(1, math.ceil(n_batches / LORA_GRAD_ACCUM))
        scheduler = t.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: max(0.0, 1.0 - step / horizon)
        )
        prepared: list[tuple[t.Tensor, t.Tensor, int]] = []
        for row in rows:
            idx = t.tensor([row.ids], dtype=t.int64, device=device)
            target_values = targets_from_mask(row.ids, row.mask)
            targets = t.tensor([target_values], dtype=t.int64, device=device)
            prepared.append((idx, targets, sum(value != -1 for value in target_values)))
        optimizer.zero_grad()
        for _epoch in range(epochs):
            order = t.randperm(len(rows), generator=shuffle).tolist()
            for step, start in enumerate(range(0, len(rows), LORA_BATCH_ROWS)):
                loss_sum: t.Tensor | None = None
                supervised = 0
                for row_index in order[start : start + LORA_BATCH_ROWS]:
                    idx, targets, count = prepared[row_index]
                    row_loss = cast(Any, model)(idx, targets, loss_reduction="sum")
                    if not isinstance(row_loss, t.Tensor):
                        raise TypeError("native LoRA loss forward must return a tensor")
                    loss_sum = row_loss if loss_sum is None else loss_sum + row_loss
                    supervised += count
                if loss_sum is None or supervised <= 0:
                    raise RuntimeError("native LoRA batch has no supervised tokens")
                loss = loss_sum / supervised / LORA_GRAD_ACCUM
                if not bool(t.isfinite(loss).item()):
                    raise ValueError("native LoRA loss is non-finite")
                loss.backward()
                if (step + 1) % LORA_GRAD_ACCUM == 0 or (step + 1) == n_batches:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
        state = {
            key: value
            for ref in refs
            for key, value in (
                (
                    f"layer_{ref.layer}_attn_c_proj_lora_A",
                    ref.adapter.lora_A.detach().cpu().contiguous().clone(),
                ),
                (
                    f"layer_{ref.layer}_attn_c_proj_lora_B",
                    ref.adapter.lora_B.detach().cpu().contiguous().clone(),
                ),
            )
        }
        if not all(bool(t.isfinite(value).all()) for value in state.values()):
            raise ValueError("native LoRA produced non-finite adapter weights")
        if not any(bool((value != 0).any()) for key, value in state.items() if key.endswith("_B")):
            raise RuntimeError("native LoRA training left every B matrix unchanged at zero")
        succeeded = True
        return state
    finally:
        t.set_rng_state(cpu_rng)
        if device_rng is not None:
            t.cuda.set_rng_state(device_rng, device=device)
        if refs:
            detach_encdec_lora(model, refs)
        for _name, parameter in base_parameters:
            parameter.requires_grad_(entry_requires_grad[id(parameter)])
        model.train(entry_training)
        leaked = [name for name, parameter in base_parameters if parameter.grad is not None]
        for _name, parameter in base_parameters:
            parameter.grad = None
        if leaked and succeeded:
            raise RuntimeError("frozen native LoRA base retained gradients: " + ", ".join(leaked))


@contextmanager
def scoped_native_lora(
    model: nn.Module, state: dict[str, t.Tensor], *, layers: Sequence[int] = LORA_LAYERS
) -> Iterator[None]:
    validate_lora_state(model, state, layers=layers)
    refs = attach_encdec_lora(model, layers)
    try:
        apply_lora_state(refs, state)
        model.eval()
        yield
    finally:
        detach_encdec_lora(model, refs)


@contextmanager
def scoped_native_direction(
    model: nn.Module, *, layer: int, direction: t.Tensor, coefficient: float
) -> Iterator[None]:
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise AxBenchAdvError(
            "native standalone direction coefficient must be finite and nonnegative"
        )
    if coefficient == 0.0:
        yield
        return
    handle = register_residual_hook(model, layer, direction, coefficient)
    try:
        yield
    finally:
        handle.remove()


def _fit_contract(
    *,
    identity_sha256: str,
    method: str,
    examples: ConceptTrainingPairs,
    geometry: Mapping[str, int],
) -> dict[str, Any]:
    model_contract = resolve_native_model_contract(geometry)
    return {
        "format": FORMAT,
        "identity_sha256": identity_sha256,
        "method": method,
        "concept_id": examples.concept_id,
        "concept": examples.concept,
        "genre": examples.genre,
        "training_split": "TRAIN",
        "n_positive_rows": len(examples.positive),
        "n_negative_rows": len(examples.negative),
        "capture_site": CAPTURE_SITE if method in DIRECTION_METHODS else None,
        "intervention_layer": (
            model_contract.intervention_layer if method in DIRECTION_METHODS else None
        ),
        "lora_layers": list(model_contract.lora_layers) if method == "LoRA" else None,
        "n_layer": geometry["n_layer"],
        "n_embd": geometry["n_embd"],
        "rendering": "trained RustBPE full user/assistant conversation",
        "population": (
            "all post-template-prefix positive and same-genre negative TRAIN tokens"
            if method in DIRECTION_METHODS
            else "positive TRAIN completion targets only"
        ),
    }


def _artifact_path(output: Path, method: str, concept_id: int) -> Path:
    return output / "methods" / METHOD_KIND[method] / "fits" / f"c{concept_id:05d}.safetensors"


def _validate_direction(
    fitted: FittedDirection, *, method: str, contract: Mapping[str, Any], width: int
) -> None:
    expected_method = "diffmean" if method == "DiffMean" else "probe"
    if (
        fitted.direction.shape != (width,)
        or fitted.direction.dtype is not t.float32
        or fitted.direction.device.type != "cpu"
        or not fitted.direction.is_contiguous()
        or not bool(t.isfinite(fitted.direction).all())
        or not math.isfinite(fitted.max_activation)
        or fitted.max_activation < 0.0
        or fitted.metadata.get("method") != expected_method
        or fitted.metadata.get("native_contract") != dict(contract)
    ):
        raise AxBenchAdvError(f"native standalone {method} artifact violates its fit contract")


def _fit_methods(
    *,
    adapter: NativeBaselineAdapter,
    training: Mapping[int, ConceptTrainingPairs],
    source: pd.DataFrame,
    output: Path,
    identity: Mapping[str, Any],
    geometry: Mapping[str, int],
    methods: Sequence[str],
    concepts: Sequence[int],
    seed: int,
) -> None:
    identity_sha = digest_json(identity)
    model_contract = resolve_native_model_contract(geometry)
    source_concepts = source.groupby("concept_id", sort=False)["input_concept"].first()
    for concept_id in concepts:
        examples = training[concept_id]
        if source_concepts.get(concept_id) != examples.concept:
            raise AxBenchAdvError(f"fit/source concept differs for global concept_id={concept_id}")
        direction_methods = [method for method in methods if method in DIRECTION_METHODS]
        missing_directions = [
            method
            for method in direction_methods
            if not _artifact_path(output, method, concept_id).is_file()
        ]
        positive_rows: tuple[RenderedFitRow, ...] | None = None
        if missing_directions:
            pos_rows, positive = _capture_class(
                adapter,
                examples.positive,
                concept_id=concept_id,
                class_name="positive",
                layer=model_contract.intervention_layer,
            )
            _neg_rows, negative = _capture_class(
                adapter,
                examples.negative,
                concept_id=concept_id,
                class_name="negative",
                layer=model_contract.intervention_layer,
            )
            positive_rows = tuple(pos_rows)
            for method in missing_directions:
                contract = _fit_contract(
                    identity_sha256=identity_sha,
                    method=method,
                    examples=examples,
                    geometry=geometry,
                )
                raw = (
                    fit_diffmean_direction(
                        positive,
                        negative,
                        intervention_layer=model_contract.intervention_layer,
                        surface=f"native standalone c{concept_id} DiffMean",
                    )
                    if method == "DiffMean"
                    else fit_probe_direction(
                        positive,
                        negative,
                        intervention_layer=model_contract.intervention_layer,
                        seed=seed,
                        concept_id=concept_id,
                        surface=f"native standalone c{concept_id} LinearProbe",
                    )
                )
                fitted = replace(raw, metadata={**raw.metadata, "native_contract": contract})
                _validate_direction(
                    fitted, method=method, contract=contract, width=geometry["n_embd"]
                )
                save_direction(_artifact_path(output, method, concept_id), fitted)
        for method in direction_methods:
            contract = _fit_contract(
                identity_sha256=identity_sha, method=method, examples=examples, geometry=geometry
            )
            fitted = load_direction(_artifact_path(output, method, concept_id))
            _validate_direction(fitted, method=method, contract=contract, width=geometry["n_embd"])
        if "LoRA" in methods:
            path = _artifact_path(output, "LoRA", concept_id)
            contract = _fit_contract(
                identity_sha256=identity_sha, method="LoRA", examples=examples, geometry=geometry
            )
            metadata = _lora_metadata(
                contract=contract,
                seed=seed,
                concept_id=concept_id,
                n_rows=len(examples.positive),
                lora_layers=model_contract.lora_layers,
            )
            if not path.is_file():
                rows = positive_rows or _render_rows(
                    adapter, examples.positive, concept_id=concept_id
                )
                state = train_native_lora(
                    adapter.model,
                    rows,
                    base_seed=seed,
                    concept_id=concept_id,
                    layers=model_contract.lora_layers,
                )
                save_lora(path, state, metadata)
            state, stored = load_lora(path)
            validate_lora_state(adapter.model, state, layers=model_contract.lora_layers)
            if stored != {key: str(value) for key, value in metadata.items()}:
                raise AxBenchAdvError(f"native standalone LoRA metadata differs for c{concept_id}")
        logger.info(
            "Native standalone fits complete concept={} methods={}", concept_id, list(methods)
        )


def _publish_fit_manifests(
    *, output: Path, identity: Mapping[str, Any], methods: Sequence[str], concepts: Sequence[int]
) -> None:
    identity_sha = digest_json(identity)
    for method in methods:
        files = []
        expected: set[Path] = set()
        for concept_id in concepts:
            path = _artifact_path(output, method, concept_id)
            expected.add(path)
            if not path.is_file():
                raise AxBenchAdvError(f"native standalone fit artifact is absent: {path}")
            files.append({"concept_id": concept_id, "path": str(path), "sha256": sha256(path)})
        fit_dir = output / "methods" / METHOD_KIND[method] / "fits"
        observed = set(fit_dir.glob("c*.safetensors")) if fit_dir.exists() else set()
        if observed != expected:
            raise AxBenchAdvError(f"native standalone {method} fit inventory differs")
        manifest = {
            "schema_version": 1,
            "format": FORMAT,
            "complete": True,
            "method": method,
            "identity_sha256": identity_sha,
            "concept_ids": list(concepts),
            "artifacts": files,
        }
        install_bytes_once(
            canonical_bytes(manifest),
            output / "methods" / METHOD_KIND[method] / "fit_manifest.json",
            surface=f"native standalone {method} fit manifest",
        )


def _load_fits(
    *,
    adapter: NativeBaselineAdapter,
    output: Path,
    identity: Mapping[str, Any],
    geometry: Mapping[str, int],
    training: Mapping[int, ConceptTrainingPairs],
    method: str,
    concepts: Sequence[int],
    seed: int,
) -> dict[int, FittedDirection | dict[str, t.Tensor]]:
    model_contract = resolve_native_model_contract(geometry)
    fit_manifest_path = output / "methods" / METHOD_KIND[method] / "fit_manifest.json"
    if not fit_manifest_path.is_file():
        raise AxBenchAdvError(f"generate requires completed {method} fit_manifest.json")
    manifest = read_json(fit_manifest_path)
    if (
        manifest.get("complete") is not True
        or manifest.get("method") != method
        or manifest.get("identity_sha256") != digest_json(identity)
        or manifest.get("concept_ids") != list(concepts)
    ):
        raise AxBenchAdvError(f"native standalone {method} fit manifest identity differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise AxBenchAdvError(f"native standalone {method} fit inventory is invalid")
    artifact_ids = [item.get("concept_id") for item in artifacts if isinstance(item, dict)]
    if len(artifact_ids) != len(artifacts) or artifact_ids != list(concepts):
        raise AxBenchAdvError(f"native standalone {method} fit inventory differs")
    loaded: dict[int, FittedDirection | dict[str, t.Tensor]] = {}
    for concept_id in concepts:
        path = _artifact_path(output, method, concept_id)
        entry = next((item for item in artifacts if item["concept_id"] == concept_id), None)
        if not isinstance(entry, dict) or sha256(path) != entry.get("sha256"):
            raise AxBenchAdvError(f"native standalone {method} fit hash differs for c{concept_id}")
        contract = _fit_contract(
            identity_sha256=digest_json(identity),
            method=method,
            examples=training[concept_id],
            geometry=geometry,
        )
        if method in DIRECTION_METHODS:
            fitted = load_direction(path)
            _validate_direction(fitted, method=method, contract=contract, width=geometry["n_embd"])
            loaded[concept_id] = fitted
        else:
            state, stored = load_lora(path)
            validate_lora_state(adapter.model, state, layers=model_contract.lora_layers)
            expected = _lora_metadata(
                contract=contract,
                seed=seed,
                concept_id=concept_id,
                n_rows=len(training[concept_id].positive),
                lora_layers=model_contract.lora_layers,
            )
            if stored != {key: str(value) for key, value in expected.items()}:
                raise AxBenchAdvError(f"native standalone LoRA metadata differs for c{concept_id}")
            loaded[concept_id] = state
    return loaded


def native_baseline_cell_filename(concept_id: int, input_id: int, factor: float) -> str:
    """Return the immutable standalone-cell name for one method coordinate."""

    factor_index = METHOD_FACTORS["DiffMean"].index(factor) if factor in FULL_FACTORS else 0
    return f"c{concept_id:05d}_i{input_id:05d}_f{factor_index:02d}.json"


def _generate_method(
    *,
    adapter: NativeBaselineAdapter,
    revalidate_inputs: Callable[[], None],
    source: pd.DataFrame,
    output: Path,
    identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    method: str,
    concepts: Sequence[int],
    fits: Mapping[int, FittedDirection | dict[str, t.Tensor]],
    seed: int,
    generation: Mapping[str, Any],
    max_new_cells: int | None,
) -> tuple[bool, int]:
    model_contract = resolve_native_model_contract(cast(Mapping[str, int], identity["geometry"]))
    root = output / "methods" / METHOD_KIND[method]
    cells = root / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    frame = _method_frame(source, method=method, concepts=concepts)
    records = frame.to_dict(orient="records")
    expected_names = {
        native_baseline_cell_filename(
            int(row["concept_id"]), int(row["input_id"]), float(row["factor"])
        )
        for row in records
    }
    observed_names = {path.name for path in cells.glob("*.json") if path.is_file()}
    if observed_names - expected_names:
        raise AxBenchAdvError(f"native standalone {method} has unexpected cells")
    method_identity = {
        "run_identity_sha256": digest_json(identity),
        "method": method,
        "factors": list(METHOD_FACTORS[method]),
        "concept_ids": list(concepts),
        "prompt": "canonical original user prompt without boosted steering instruction",
        "operation": (
            "residual += factor * fit_max_activation * saved_fp32_unit_direction"
            if method in DIRECTION_METHODS
            else "trained rsLoRA adapters active at factor 1.0"
        ),
    }
    method_sha = digest_json(method_identity)
    results: list[str] = []
    generated = 0
    for row in records:
        concept_id = int(row["concept_id"])
        input_id = int(row["input_id"])
        factor = float(row["factor"])
        artifact_path = _artifact_path(output, method, concept_id)
        cell = {
            "concept_id": concept_id,
            "input_id": input_id,
            "factor_hex": factor.hex(),
            "seed": derive_cell_seed(seed, method, concept_id, input_id, factor),
            "fit_sha256": sha256(artifact_path),
            "prompt_sha256": hashlib.sha256(str(row["original_prompt"]).encode()).hexdigest(),
        }
        path = cells / native_baseline_cell_filename(concept_id, input_id, factor)
        if path.exists():
            record = read_json(path)
            if (
                set(record) != {"method_identity_sha256", "cell", "generation"}
                or record.get("method_identity_sha256") != method_sha
                or record.get("cell") != cell
                or not isinstance(record.get("generation"), str)
            ):
                raise AxBenchAdvError(f"invalid/conflicting native standalone cell: {path}")
            results.append(record["generation"])
            continue
        if max_new_cells is not None and generated >= max_new_cells:
            write_json(
                root / "progress.json",
                {"complete": False, "completed_cells": len(results), "total_cells": len(frame)},
            )
            return False, generated
        fitted = fits[concept_id]
        if method in DIRECTION_METHODS:
            assert isinstance(fitted, FittedDirection)
            scope = scoped_native_direction(
                adapter.model,
                layer=model_contract.intervention_layer,
                direction=fitted.direction,
                coefficient=factor * fitted.max_activation,
            )
        else:
            lora_state = cast(dict[str, t.Tensor], fitted)
            scope = scoped_native_lora(
                adapter.model, lora_state, layers=model_contract.lora_layers
            )
        with scope:
            text = adapter.generate_user_text(
                str(row["original_prompt"]), seed=cell["seed"], **generation
            )
        install_bytes_once(
            canonical_bytes(
                {"method_identity_sha256": method_sha, "cell": cell, "generation": text}
            ),
            path,
            surface=f"native standalone {method} generation cell",
        )
        results.append(text)
        generated += 1
        write_json(
            root / "progress.json",
            {"complete": False, "completed_cells": len(results), "total_cells": len(frame)},
        )
    if (
        len(results) != len(frame)
        or {path.name for path in cells.glob("*.json") if path.is_file()} != expected_names
    ):
        raise AxBenchAdvError(f"native standalone {method} cell coverage is incomplete")
    frame[f"{method}_steered_generation"] = results
    output_path = root / "steering_data.parquet"
    atomic_write_parquet(pa.Table.from_pandas(frame, preserve_index=False), output_path)
    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "method": method,
        "factors": list(METHOD_FACTORS[method]),
        "complete": True,
        "identity_sha256": digest_json(identity),
        "method_identity": method_identity,
        "checkpoint": dict(checkpoint),
        "source": {
            "max_concepts": 100,
            "concept_ids": list(concepts),
            "input_path": identity["inputs"]["source"]["path"],
            "input_sha256": identity["inputs"]["source"]["sha256"],
        },
        "generation": {"seed": seed, **dict(generation)},
        "identity": {
            "path": str(output / "identity.json"),
            "sha256": sha256(output / "identity.json"),
        },
        "fit_manifest": {
            "path": str(root / "fit_manifest.json"),
            "sha256": sha256(root / "fit_manifest.json"),
        },
        "policy": {
            "prompt": "original_prompt",
            "boosted_prompt_used": False,
            "training_split": "TRAIN",
            "layer": model_contract.intervention_layer if method in DIRECTION_METHODS else None,
            "lora_layers": list(model_contract.lora_layers) if method == "LoRA" else None,
        },
        "output": {"path": str(output_path), "sha256": sha256(output_path), "rows": len(frame)},
    }
    revalidate_inputs()
    install_bytes_once(
        canonical_bytes(manifest),
        root / "run_manifest.json",
        surface=f"native standalone {method} manifest",
    )
    write_json(
        root / "progress.json",
        {"complete": True, "completed_cells": len(results), "total_cells": len(results)},
    )
    return True, generated


def load_native_baseline_output(
    manifest_path: Path, output_path: Path | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = read_json(Path(manifest_path))
    method = manifest.get("method")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("format") != FORMAT
        or method not in METHODS
        or manifest.get("factors") != list(METHOD_FACTORS[cast(str, method)])
        or manifest.get("complete") is not True
    ):
        raise ValueError("expected a completed native standalone baseline manifest")
    identity = manifest.get("identity")
    fit_manifest = manifest.get("fit_manifest")
    method_identity = manifest.get("method_identity")

    def authenticated_file(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        raw_path, expected_sha = record.get("path"), record.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            return False
        path = Path(raw_path)
        return path.is_file() and sha256(path) == expected_sha

    if (
        not isinstance(identity, dict)
        or not isinstance(fit_manifest, dict)
        or not isinstance(method_identity, dict)
        or not authenticated_file(identity)
        or digest_json(read_json(Path(identity["path"]))) != manifest.get("identity_sha256")
        or not authenticated_file(fit_manifest)
        or method_identity.get("run_identity_sha256") != manifest.get("identity_sha256")
        or method_identity.get("method") != method
        or method_identity.get("factors") != list(METHOD_FACTORS[cast(str, method)])
    ):
        raise ValueError("native standalone manifest identity/fit authentication differs")
    path = Path(manifest["output"]["path"]) if output_path is None else Path(output_path)
    if sha256(path) != manifest["output"]["sha256"]:
        raise ValueError("native standalone output SHA-256 mismatch")
    frame = pd.read_parquet(path)
    if sha256(path) != manifest["output"]["sha256"]:
        raise ValueError("native standalone output changed while loading")
    concepts = manifest["source"].get("concept_ids")
    if (
        not isinstance(concepts, list)
        or not concepts
        or any(type(concept_id) is not int for concept_id in concepts)
        or concepts != list(range(concepts[0], concepts[-1] + 1))
        or method_identity.get("concept_ids") != concepts
    ):
        raise ValueError("native standalone manifest concept_ids are invalid")
    expected_keys = [
        (concept_id, input_id, factor)
        for concept_id in concepts
        for input_id in range(10)
        for factor in METHOD_FACTORS[cast(str, method)]
    ]
    actual_keys = list(
        frame[["concept_id", "input_id", "factor"]].itertuples(index=False, name=None)
    )
    column = f"{method}_steered_generation"
    if (
        actual_keys != expected_keys
        or len(frame) != manifest["output"]["rows"]
        or column not in frame
        or not frame[column].map(lambda value: isinstance(value, str)).all()
        or not frame["native_user_prompt"].equals(frame["original_prompt"])
    ):
        raise ValueError("native standalone output row/generation/original-prompt contract differs")
    return manifest, frame


def run_native_baselines(cfg: DictConfig) -> dict[str, Any]:
    """Fit and/or generate explicitly selected standalone native baselines."""

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise AxBenchAdvError("native standalone config must be a mapping")
    phase = str(cfg.phase)
    if phase not in {"fit", "generate", "all"}:
        raise AxBenchAdvError("phase must be fit, generate, or all")
    methods = _selected_methods(cfg)
    concepts = _selected_concepts(cfg)
    if type(cfg.seed) is not int:
        raise AxBenchAdvError("seed must be an integer")
    max_cells = cfg.max_cells
    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise AxBenchAdvError("max_cells must be null or a positive integer")
    source, source_sha = _source_frame(cfg)
    checkpoint, geometry = _runtime_identity(cfg)
    model_contract = resolve_native_model_contract(geometry)
    prepared_dir = Path(str(cfg.prepared_dir)).resolve()
    prepared_metadata_path = prepared_dir / "generate" / "metadata.jsonl"
    prepared_training_path = prepared_dir / "generate" / "train_data.parquet"
    prepared_metadata_sha = sha256(prepared_metadata_path)
    prepared_training_sha = sha256(prepared_training_path)
    training = select_training_pairs(prepared_dir, max_concepts=int(cfg.max_concepts))
    _assert_train_only_population(source, training)
    if (
        sha256(Path(str(cfg.source_path))) != source_sha
        or sha256(prepared_metadata_path) != prepared_metadata_sha
        or sha256(prepared_training_path) != prepared_training_sha
    ):
        raise AxBenchAdvError("native standalone input changed while loading training rows")
    output = Path(str(cfg.output_dir)).resolve()
    input_paths = (
        Path(str(cfg.source_path)).resolve(),
        prepared_dir,
        Path(str(cfg.checkpoint_dir)).resolve(),
        Path(str(cfg.tokenizer_path)).resolve(),
    )
    if any(
        output == path or output.is_relative_to(path) or path.is_relative_to(output)
        for path in input_paths
    ):
        raise AxBenchAdvError("native standalone output must be disjoint from every input")
    root = Path(__file__).resolve().parents[3]
    implementation_files = (
        "steering/eval/axbench_upstream/native_baselines.py",
        "steering/eval/axbench_upstream/native_decoder.py",
        "steering/conditional/encoder_decoder/sm12_fa4.py",
        "steering/eval/axbench_upstream/native_contract.py",
        "steering/eval/axbench_upstream/data.py",
        "steering/eval/axbench_upstream/encdec_baselines/directions.py",
        "steering/eval/axbench_upstream/encdec_baselines/lora.py",
        "steering/eval/axbench_upstream/encdec_baselines/rendering.py",
        "steering/eval/axbench_base/baselines/hooks.py",
        "steering/eval/inference_engine.py",
        "steering/eval/chat_generation.py",
        "steering/sft/loss_mask.py",
        "steering/safetensors_io.py",
        "steering/eval/axbench_base/direction_scale.py",
        "nanochat/checkpoint_manager.py",
        "nanochat/common.py",
        "nanochat/flash_attention.py",
        "nanochat/gpt.py",
        "nanochat/engine.py",
        "nanochat/tokenizer.py",
        "nanochat/sampling.py",
        "uv.lock",
    )
    generation = {
        "max_new_tokens": int(cfg.max_new_tokens),
        "temperature": float(cfg.temperature),
        "top_k": int(cfg.top_k),
    }
    identity = {
        "schema_version": 1,
        "format": FORMAT,
        "inputs": {
            "source": {"path": str(Path(str(cfg.source_path)).resolve()), "sha256": source_sha},
            "prepared_metadata": {
                "path": str(prepared_metadata_path),
                "sha256": prepared_metadata_sha,
            },
            "prepared_training_data": {
                "path": str(prepared_training_path),
                "sha256": prepared_training_sha,
            },
            "checkpoint": checkpoint,
        },
        "configuration": {
            "methods": list(methods),
            "concept_ids": list(concepts),
            "max_concepts": 100,
            "direction_factors": list(FULL_FACTORS),
            "lora_factors": [1.0],
            "layer": model_contract.intervention_layer,
            "capture_site": CAPTURE_SITE,
            "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
            "lora_layers": list(model_contract.lora_layers),
            "seed": int(cfg.seed),
            "generation": generation,
            "prompt": "original_prompt only",
            "renderer": "RustBPE.render_conversation(max_tokens=None) + assistant_start",
            "training_split": "TRAIN only",
            "direction_scale": "factor * fit-row max_activation * unit direction",
        },
        "geometry": geometry,
        "runtime": {
            "device": str(cfg.device),
            **native_runtime_settings(full_prefix_device=t.device(str(cfg.device))),
            "cuda_version": t.version.cuda,
        },
        "implementation_sha256": {name: sha256(root / name) for name in implementation_files},
    }
    identity_path = output / "identity.json"
    if output.exists() and not identity_path.is_file() and any(
        not (path.is_file() and path.name.startswith(".identity.json.tmp."))
        for path in output.iterdir()
    ):
        raise AxBenchAdvError("native standalone refuses nonempty output without identity.json")
    output.mkdir(parents=True, exist_ok=True)
    install_bytes_once(
        canonical_bytes(identity), identity_path, surface="native standalone identity"
    )
    identity_file_sha = sha256(identity_path)
    if identity_file_sha != hashlib.sha256(canonical_bytes(identity)).hexdigest():
        raise AxBenchAdvError("native standalone identity bytes are not canonical")

    adapter: NativeDecoderAdapter | None = None

    def get_adapter() -> NativeDecoderAdapter:
        nonlocal adapter
        if adapter is None:
            adapter = NativeDecoderAdapter(
                Path(str(cfg.checkpoint_dir)),
                int(cfg.step),
                str(cfg.device),
                Path(str(cfg.tokenizer_path)),
            )
            if any(
                adapter.identity.get(key) != checkpoint[key]
                for key in ("step", "model_sha256", "meta_sha256", "tokenizer_sha256")
            ):
                raise AxBenchAdvError("native standalone runtime changed while loading the model")
            if template_prefix_length(adapter.tokenizer) != CAPTURE_PREFIX_LENGTH:
                raise AxBenchAdvError(
                    "trained RustBPE template prefix is not the reviewed two tokens"
                )
            if (
                int(adapter.model.config.n_layer) != geometry["n_layer"]
                or int(adapter.model.config.n_embd) != geometry["n_embd"]
                or int(adapter.model.config.sequence_len) != geometry["sequence_len"]
            ):
                raise AxBenchAdvError("loaded native model geometry differs from metadata")
            if model_contract.parameter_count is not None and sum(
                parameter.numel() for parameter in adapter.model.parameters()
            ) != model_contract.parameter_count:
                raise AxBenchAdvError("loaded native model parameter count differs")
        return adapter

    def revalidate_inputs() -> None:
        if sha256(identity_path) != identity_file_sha:
            raise AxBenchAdvError("native standalone identity changed during the run")
        current_source, current_source_sha = _source_frame(cfg)
        if current_source_sha != source_sha or not current_source.equals(source):
            raise AxBenchAdvError("native standalone source changed during the run")
        current_checkpoint, current_geometry = _runtime_identity(cfg)
        if current_checkpoint != checkpoint or current_geometry != geometry:
            raise AxBenchAdvError("native standalone checkpoint changed during the run")
        if sha256(prepared_metadata_path) != prepared_metadata_sha:
            raise AxBenchAdvError("native standalone prepared_metadata changed during the run")
        if sha256(prepared_training_path) != prepared_training_sha:
            raise AxBenchAdvError("native standalone prepared_training_data changed during the run")

    if phase in {"fit", "all"}:
        _fit_methods(
            adapter=get_adapter(),
            training=training,
            source=source,
            output=output,
            identity=identity,
            geometry=geometry,
            methods=methods,
            concepts=concepts,
            seed=int(cfg.seed),
        )
        revalidate_inputs()
        _publish_fit_manifests(output=output, identity=identity, methods=methods, concepts=concepts)
    if phase == "fit":
        status = {
            "phase": "fit_complete",
            "identity_sha256": digest_json(identity),
            "methods": list(methods),
        }
        write_json(output / "status.json", status)
        return status

    remaining = max_cells
    completed: list[str] = []
    for method in methods:
        fits = _load_fits(
            adapter=get_adapter(),
            output=output,
            identity=identity,
            geometry=geometry,
            training=training,
            method=method,
            concepts=concepts,
            seed=int(cfg.seed),
        )
        done, generated = _generate_method(
            adapter=get_adapter(),
            revalidate_inputs=revalidate_inputs,
            source=source,
            output=output,
            identity=identity,
            checkpoint=checkpoint,
            method=method,
            concepts=concepts,
            fits=fits,
            seed=int(cfg.seed),
            generation=generation,
            max_new_cells=remaining,
        )
        if done:
            completed.append(method)
            revalidate_inputs()
        if remaining is not None:
            remaining -= generated
        if not done:
            status = {
                "phase": "generation_incomplete",
                "identity_sha256": digest_json(identity),
                "completed_methods": completed,
            }
            write_json(output / "status.json", status)
            return status
    revalidate_inputs()
    status = {
        "phase": "generation_complete_pending_native_judges",
        "identity_sha256": digest_json(identity),
        "completed_methods": completed,
        "method_manifests": {
            method: str(output / "methods" / METHOD_KIND[method] / "run_manifest.json")
            for method in completed
        },
    }
    write_json(output / "status.json", status)
    return status


__all__ = (
    "CAPTURE_PREFIX_LENGTH",
    "CAPTURE_SITE",
    "DIRECTION_METHODS",
    "FORMAT",
    "D14_MODEL_CONTRACT",
    "INTERVENTION_LAYER",
    "LEGACY_MODEL_CONTRACT",
    "LORA_LAYERS",
    "METHOD_FACTORS",
    "METHODS",
    "derive_cell_seed",
    "load_native_baseline_output",
    "native_baseline_cell_filename",
    "resolve_native_model_contract",
    "run_native_baselines",
    "scoped_native_direction",
    "scoped_native_lora",
    "train_native_lora",
)
