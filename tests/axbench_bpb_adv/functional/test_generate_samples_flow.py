"""End-to-end test for the per-datapoint sampled-completions orchestrator.

Builds a tiny FINISHED bpb-adv run dir on disk (``candidates/candidates.parquet``
+ ``aggregates/{macro,per_candidate}.parquet``) for two decoder-only families
(``diffmean_decoder_only`` + ``control_decoder_only``), one concept, additive
counter-steering strengths 0..1 at one representative coefficient, then drives the
ORCHESTRATION CORE (``run_generate_samples_core``) with the REAL tiny 2-layer
decoder-only GPT (CPU) and the ``mini_axbench_base_run`` source (real diffmean
direction safetensors).

Asserts one YAML per family under
``samples/<mode>/<method>__additive_direction.yaml``, the nested
schema (family metadata -> concepts -> prefixes -> datapoints -> completions with
``n_completions`` strings each, metrics present), and that an encdec target is
rejected loudly.

The model + the mini fixture are loaded LOCALLY here (no cross-root conftest
dependency, per the conftest caveat), mirroring ``test_generate_completions_flow``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

import steering.eval.axbench_bpb_adv.sample_generation as sg
from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.direction_scale import encode_max_activation
from steering.eval.axbench_bpb.types import AxBenchBpbError, LoraParams, SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import (
    core_prefix_id as _real_core_prefix_id,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.sample_generation import (
    GenerationConfig,
    run_generate_samples_core,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
)

_FIXTURES = Path(__file__).resolve().parents[2] / "axbench_bpb" / "fixtures"
_TINY_CKPT = _FIXTURES / "tiny_decoder_only" / "ckpt.pt"
_MINI_FIXTURE = _FIXTURES / "mini_axbench_base_run"

# Both families share concept_id=1 ("dogs" in the mini fixture's concepts.json),
# the representative coefficient 4.0, and the additive counter sweep {0.0, 1.0}.
_CONCEPT_ID = 1
_CORE_PREFIX_ID = "cp_a"
_COEFFICIENT_STR = "4.0"
_PROMPT_TOKENS = [0, 5, 6, 7]
_DIFFMEAN_LAYER = 1  # valid block index for the 2-layer tiny model
_VOCAB = 128

# (method_name, mode) for the two decoder-only families under test.
_FAMILIES = (
    ("diffmean_decoder_only", "positive"),
    ("control_decoder_only", "positive"),
)


def _feature_diffmean_method(mode: str) -> str:
    if mode == "negative":
        return "neg_feature_diffmean_encdec"
    return "feature_diffmean_encdec"


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return _VOCAB

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


def _candidate_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
) -> dict[str, Any]:
    candidate_id = f"{method_name}|{mode}|s{strength_index}"
    return {
        "candidate_id": candidate_id,
        "attack_row_id": candidate_id,
        "core_prefix_id": _CORE_PREFIX_ID,
        "concept_rich_sample_id": "rich",
        "concept_avoiding_sample_id": "avoid",
        "concept_id": _CONCEPT_ID,
        "mode": mode,
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "prefix_text": "dogs are great",
        "prefix_hash": "ph",
        "source_prompt_tokens": list(_PROMPT_TOKENS),
        "source_prompt_token_digest": "srcdig",
        "attacked_prompt_text": "dogs are great",
        "attacked_prompt_tokens": list(_PROMPT_TOKENS),
        "attacked_prompt_token_digest": "attdig",
        "concept_rich_output_tokens": [11, 12],
        "concept_avoiding_output_tokens": [21, 22],
        "concept_rich_output_digest": "rdig",
        "concept_avoiding_output_digest": "adig",
        "editable_token_start": 0,
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": 0.0,
        "candidate_metadata_json": '{"candidate":"no_adversary_control"}',
    }


def _candidates_table() -> pa.Table:
    rows: list[dict[str, Any]] = []
    for method_name, mode in _FAMILIES:
        for strength_index, strength in ((0, 0.0), (1, 1.0)):
            rows.append(
                _candidate_row(
                    method_name=method_name,
                    mode=mode,
                    strength_index=strength_index,
                    adversarial_strength=strength,
                )
            )
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def _per_candidate_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
    is_control: bool,
) -> dict[str, Any]:
    return {
        "concept_id": _CONCEPT_ID,
        "mode": mode,
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0 - 0.1 * strength_index,
        # k=0 control margin: present on the control rows so representative
        # selection can pick the 4.0 coefficient.
        "baseline_primary_margin": 0.9 if is_control else None,
        "general_val_bpb": None,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 2,
        "n_avoid_output_tokens": 2,
        "n_rich_output_bytes": 2,
        "n_avoid_output_bytes": 2,
        "is_control": is_control,
    }


def _per_candidate_table() -> pa.Table:
    rows: list[dict[str, Any]] = []
    for method_name, mode in _FAMILIES:
        rows.append(
            _per_candidate_row(
                method_name=method_name,
                mode=mode,
                strength_index=0,
                adversarial_strength=0.0,
                is_control=True,
            )
        )
        rows.append(
            _per_candidate_row(
                method_name=method_name,
                mode=mode,
                strength_index=1,
                adversarial_strength=1.0,
                is_control=False,
            )
        )
    return pa.Table.from_pylist(rows, schema=PER_CANDIDATE_AGG_SCHEMA)


def _macro_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
    is_control: bool,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0 - 0.1 * strength_index,
        "baseline_primary_margin": 0.9 if is_control else None,
        "general_val_bpb": None,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": is_control,
    }


def _feature_diffmean_macro_target(
    *,
    mode: str,
    adversarial_method: str,
    context_signal_mode: str,
    strength_unit: str,
    score: float = 0.9,
) -> dict[str, Any]:
    row = _macro_row(
        method_name=_feature_diffmean_method(mode),
        mode=mode,
        strength_index=0,
        adversarial_strength=0.0,
        is_control=True,
    )
    row["target_model"] = "encdec"
    row["coefficient_str"] = "None"
    row["adversarial_method"] = adversarial_method
    row["context_signal_mode"] = context_signal_mode
    row["strength_unit"] = strength_unit
    row["primary_margin_y"] = score
    row["baseline_primary_margin"] = score
    return row


def _with_feature_diffmean_macro_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                str(row["mode"]),
                str(row["adversarial_method"]),
                str(row["context_signal_mode"]),
                str(row["strength_unit"]),
            )
            for row in rows
        }
    )
    return [
        *rows,
        *[
            _feature_diffmean_macro_target(
                mode=mode,
                adversarial_method=adversarial_method,
                context_signal_mode=context_signal_mode,
                strength_unit=strength_unit,
            )
            for mode, adversarial_method, context_signal_mode, strength_unit in keys
        ],
    ]


def _macro_table() -> pa.Table:
    rows: list[dict[str, Any]] = []
    for method_name, mode in _FAMILIES:
        rows.append(
            _macro_row(
                method_name=method_name,
                mode=mode,
                strength_index=0,
                adversarial_strength=0.0,
                is_control=True,
            )
        )
        rows.append(
            _macro_row(
                method_name=method_name,
                mode=mode,
                strength_index=1,
                adversarial_strength=1.0,
                is_control=False,
            )
        )
    rows = _with_feature_diffmean_macro_targets(rows)
    return pa.Table.from_pylist(rows, schema=MACRO_AGG_SCHEMA)


def _roundtrip_per_candidate_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
    is_control: bool,
) -> dict[str, Any]:
    return {
        "concept_id": _CONCEPT_ID,
        "mode": mode,
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "mean_roundtrip_pre_act": 8.0,
        "mean_roundtrip_post_act": 4.0,
        "mean_roundtrip_percentile": 80.0 - 10.0 * strength_index,
        "primary_roundtrip_y": 80.0 - 10.0 * strength_index,
        "mean_control_roundtrip_pre_act": 8.0,
        "mean_control_roundtrip_post_act": 4.0,
        "mean_control_roundtrip_percentile": 80.0,
        "control_primary_roundtrip_y": 80.0,
        "general_val_bpb": None,
        "n_samples": 1,
        "n_generated_tokens": 3,
        "n_control_generated_tokens": 3,
        "max_tokens_rate": 0.0,
        "control_max_tokens_rate": 0.0,
        "is_control": is_control,
    }


def _roundtrip_per_candidate_table() -> pa.Table:
    rows: list[dict[str, Any]] = []
    for method_name, mode in _FAMILIES:
        rows.append(
            _roundtrip_per_candidate_row(
                method_name=method_name,
                mode=mode,
                strength_index=0,
                adversarial_strength=0.0,
                is_control=True,
            )
        )
        rows.append(
            _roundtrip_per_candidate_row(
                method_name=method_name,
                mode=mode,
                strength_index=1,
                adversarial_strength=1.0,
                is_control=False,
            )
        )
    return pa.Table.from_pylist(rows, schema=ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)


def _roundtrip_macro_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
    is_control: bool,
) -> dict[str, Any]:
    row = _roundtrip_per_candidate_row(
        method_name=method_name,
        mode=mode,
        strength_index=strength_index,
        adversarial_strength=adversarial_strength,
        is_control=is_control,
    )
    row.pop("concept_id")
    row["n_concepts"] = 1
    return row


def _feature_diffmean_roundtrip_macro_target(
    *,
    mode: str,
    adversarial_method: str,
    context_signal_mode: str,
    strength_unit: str,
    score: float = 80.0,
) -> dict[str, Any]:
    row = _roundtrip_macro_row(
        method_name=_feature_diffmean_method(mode),
        mode=mode,
        strength_index=0,
        adversarial_strength=0.0,
        is_control=True,
    )
    row["target_model"] = "encdec"
    row["coefficient_str"] = "None"
    row["adversarial_method"] = adversarial_method
    row["context_signal_mode"] = context_signal_mode
    row["strength_unit"] = strength_unit
    row["primary_roundtrip_y"] = score
    return row


def _with_feature_diffmean_roundtrip_macro_targets(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                str(row["mode"]),
                str(row["adversarial_method"]),
                str(row["context_signal_mode"]),
                str(row["strength_unit"]),
            )
            for row in rows
        }
    )
    return [
        *rows,
        *[
            _feature_diffmean_roundtrip_macro_target(
                mode=mode,
                adversarial_method=adversarial_method,
                context_signal_mode=context_signal_mode,
                strength_unit=strength_unit,
            )
            for mode, adversarial_method, context_signal_mode, strength_unit in keys
        ],
    ]


def _roundtrip_macro_table() -> pa.Table:
    rows: list[dict[str, Any]] = []
    for method_name, mode in _FAMILIES:
        rows.append(
            _roundtrip_macro_row(
                method_name=method_name,
                mode=mode,
                strength_index=0,
                adversarial_strength=0.0,
                is_control=True,
            )
        )
        rows.append(
            _roundtrip_macro_row(
                method_name=method_name,
                mode=mode,
                strength_index=1,
                adversarial_strength=1.0,
                is_control=False,
            )
        )
    rows = _with_feature_diffmean_roundtrip_macro_targets(rows)
    return pa.Table.from_pylist(rows, schema=ROUNDTRIP_MACRO_AGG_SCHEMA)


def _write_finished_run(output_root: Path) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(_candidates_table(), paths.candidates_path())
    pq.write_table(_per_candidate_table(), paths.aggregates_dir() / "per_candidate.parquet")
    pq.write_table(_macro_table(), paths.aggregates_dir() / "macro.parquet")
    return paths


def _write_finished_roundtrip_run(output_root: Path) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.roundtrip_aggregates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(_candidates_table(), paths.candidates_path())
    pq.write_table(
        _roundtrip_per_candidate_table(),
        paths.roundtrip_aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        _roundtrip_macro_table(),
        paths.roundtrip_aggregates_dir() / "macro.parquet",
    )
    return paths


def _with_coefficient(table: pa.Table, coefficient_str: str) -> pa.Table:
    rows = table.to_pylist()
    for row in rows:
        row["coefficient_str"] = coefficient_str
    return pa.Table.from_pylist(rows, schema=table.schema)


def _write_mixed_metric_run(
    output_root: Path,
    *,
    stale_legacy_coefficient: str = _COEFFICIENT_STR,
    stale_roundtrip_coefficient: str = _COEFFICIENT_STR,
) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    paths.roundtrip_aggregates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(_candidates_table(), paths.candidates_path())
    pq.write_table(
        _with_coefficient(_per_candidate_table(), stale_legacy_coefficient),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        _with_coefficient(_macro_table(), stale_legacy_coefficient),
        paths.aggregates_dir() / "macro.parquet",
    )
    pq.write_table(
        _with_coefficient(
            _roundtrip_per_candidate_table(),
            stale_roundtrip_coefficient,
        ),
        paths.roundtrip_aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        _with_coefficient(_roundtrip_macro_table(), stale_roundtrip_coefficient),
        paths.roundtrip_aggregates_dir() / "macro.parquet",
    )
    return paths


def _generation_config(**overrides: Any) -> GenerationConfig:
    defaults: dict[str, Any] = {
        "n_concepts": 5,
        "n_prefixes_per_concept": 1,
        "n_completions": 3,
        "temperature": 0.8,
        "top_k": None,
        "max_new_tokens": 6,
        "base_seed": 0,
        "methods": None,
    }
    defaults.update(overrides)
    return GenerationConfig(**defaults)


def _run_core(
    paths: AxBenchBpbAdvPaths,
    *,
    target_model: str = "decoder_only",
    steering_metric: str = "bpb_class_margin",
    generation: GenerationConfig | None = None,
    lora: LoraParams | None = None,
) -> None:
    run_generate_samples_core(
        paths=paths,
        source_run_dir=_MINI_FIXTURE,
        model=_tiny_model(),
        tokenizer=_TinyTokenizer(),
        device=t.device("cpu"),
        target_model=target_model,
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=_DIFFMEAN_LAYER,
        counter_direction_family="diffmean",
        scatter_baseline_x="baseline_loss",
        steering_metric=steering_metric,
        generation=generation if generation is not None else _generation_config(),
        lora=lora,
        apply_max_act_scale=False,
    )


def _run_core_scaled(
    paths: AxBenchBpbAdvPaths,
    *,
    source_run_dir: Path,
    generation: GenerationConfig | None = None,
) -> None:
    run_generate_samples_core(
        paths=paths,
        source_run_dir=source_run_dir,
        model=_tiny_model(),
        tokenizer=_TinyTokenizer(),
        device=t.device("cpu"),
        target_model="decoder_only",
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=_DIFFMEAN_LAYER,
        counter_direction_family="diffmean",
        scatter_baseline_x="baseline_loss",
        steering_metric="bpb_class_margin",
        generation=generation if generation is not None else _generation_config(),
        lora=None,
        apply_max_act_scale=True,
    )


def _source_with_direction_maxact(tmp_path: Path, value: float) -> Path:
    source_run_dir = tmp_path / "source_run"
    shutil.copytree(_MINI_FIXTURE, source_run_dir)
    direction_path = (
        source_run_dir
        / "baselines"
        / "diffmean"
        / "decoder_only"
        / "positive"
        / "concept_0001.safetensors"
    )
    with safe_open(str(direction_path), framework="pt", device="cpu") as handle:
        direction = handle.get_tensor("direction")
    save_file(
        {"direction": direction.contiguous()},
        str(direction_path),
        metadata={
            "concept_id": "1",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": str(_DIFFMEAN_LAYER),
            "n_pos": "2",
            "n_neg": "2",
            "max_activation": encode_max_activation(value),
        },
    )
    return source_run_dir


def test_generate_samples_core_preflights_dead_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A standalone generate_samples run with an invalid zero-valued nonzero
    # direction artifact must FAIL-FAST *before* generating.
    #
    # TEETH: patch the family-generation entry to raise a sentinel. With the
    # pre-flight in place, the artifact contract raises BEFORE
    # the family loop is reached, so the sentinel never fires. If the pre-flight
    # call were removed, generation would start and hit the sentinel (a
    # RuntimeError), so `pytest.raises(AxBenchBaseError, ...)` would FAIL --
    # distinguishing the pre-flight from the identical deep consumer error a
    # real model would otherwise produce.
    paths = _write_finished_run(tmp_path / "out")
    # concept_id=1 -> feature_index 1 in the mini fixture; make feature 1 dead.
    source_run_dir = _source_with_direction_maxact(tmp_path, 0.0)

    def _boom(*_args: Any, **_kwargs: Any):
        raise RuntimeError("generation started before the max_act pre-flight")

    monkeypatch.setattr(sg, "_generate_family_completions", _boom)

    with pytest.raises(
        AxBenchBpbAdvError,
        match="zero-direction max_activation mismatch",
    ):
        _run_core_scaled(paths, source_run_dir=source_run_dir)
    # Fail-fast: it raised BEFORE any per-family YAML was written.
    assert not list(paths.samples_dir().rglob("*.yaml"))


def test_generate_samples_core_with_live_scale_writes_yaml(tmp_path: Path) -> None:
    # The preflight passes through when the feature is alive; generation proceeds.
    paths = _write_finished_run(tmp_path / "out")
    source_run_dir = _source_with_direction_maxact(tmp_path, 44.0)
    _run_core_scaled(paths, source_run_dir=source_run_dir)
    written = sorted(p.name for p in (paths.samples_dir() / "positive").glob("*.yaml"))
    assert written == [
        "control_decoder_only__additive_direction.yaml",
        "diffmean_decoder_only__additive_direction.yaml",
    ]


def test_generate_samples_preflights_only_filtered_consumers(tmp_path: Path) -> None:
    paths = _write_finished_run(tmp_path / "out")
    rows = _candidates_table().to_pylist()
    absent_encdec = _candidate_row(
        method_name="diffmean_encdec",
        mode="positive",
        strength_index=0,
        adversarial_strength=0.0,
    )
    absent_encdec["candidate_id"] = "absent-encdec-direction"
    absent_encdec["attack_row_id"] = "absent-encdec-direction"
    absent_encdec["target_model"] = "encdec"
    rows.append(absent_encdec)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    source_run_dir = _source_with_direction_maxact(tmp_path, 44.0)
    absent_path = (
        source_run_dir
        / "baselines"
        / "diffmean"
        / "encdec"
        / "positive"
        / "concept_0001.safetensors"
    )
    absent_path.unlink()

    _run_core_scaled(
        paths,
        source_run_dir=source_run_dir,
        generation=_generation_config(methods=["diffmean_decoder_only"]),
    )

    written = sorted(
        p.name for p in (paths.samples_dir() / "positive").glob("*.yaml")
    )
    assert written == ["diffmean_decoder_only__additive_direction.yaml"]


def test_generate_samples_skips_unselected_concept_artifacts(
    tmp_path: Path,
) -> None:
    """Concept/prefix selection also narrows the preflight (second F2 conjunct).

    The sibling test above pins the family/method filtering dimension; this one
    pins the selection dimension INSIDE a kept family: with ``n_concepts=1``,
    deterministic ascending selection keeps concept 1 and drops concept 2, so
    concept 2's direction artifact must never be preflighted — deleting it must
    not fail the run. Surfaced by the final-gate mutation sweep: reverting the
    selection narrowing (membership check forced True) survived the sibling
    test because its counterexample row is excluded by the family whitelist
    before selection ever runs.
    """
    paths = _write_finished_run(tmp_path / "out")
    rows = _candidates_table().to_pylist()
    unselected = _candidate_row(
        method_name="diffmean_decoder_only",
        mode="positive",
        strength_index=0,
        adversarial_strength=0.0,
    )
    unselected["candidate_id"] = "unselected-concept-direction"
    unselected["attack_row_id"] = "unselected-concept-direction"
    unselected["concept_id"] = 2
    rows.append(unselected)
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    source_run_dir = _source_with_direction_maxact(tmp_path, 44.0)
    unselected_path = (
        source_run_dir
        / "baselines"
        / "diffmean"
        / "decoder_only"
        / "positive"
        / "concept_0002.safetensors"
    )
    unselected_path.unlink()

    _run_core_scaled(
        paths,
        source_run_dir=source_run_dir,
        generation=_generation_config(
            n_concepts=1, methods=["diffmean_decoder_only"]
        ),
    )

    written = sorted(
        p.name for p in (paths.samples_dir() / "positive").glob("*.yaml")
    )
    assert written == ["diffmean_decoder_only__additive_direction.yaml"]


def test_generate_samples_core_writes_one_yaml_per_family(tmp_path: Path) -> None:
    paths = _write_finished_run(tmp_path / "out")
    _run_core(paths)

    samples_dir = paths.samples_dir()
    # One YAML per family at samples/positive/<method>__additive_direction.yaml.
    mode_dir = samples_dir / "positive"
    written = sorted(p.name for p in mode_dir.glob("*.yaml"))
    assert written == [
        "control_decoder_only__additive_direction.yaml",
        "diffmean_decoder_only__additive_direction.yaml",
    ]

    for method_name, _mode in _FAMILIES:
        doc = yaml.safe_load(
            (mode_dir / f"{method_name}__additive_direction.yaml").read_text()
        )
        # Family metadata.
        assert doc["method_name"] == method_name
        assert doc["mode"] == "positive"
        assert doc["target_model"] == "decoder_only"
        assert doc["adversarial_method"] == "additive_direction"
        assert doc["representative_coefficient"] == _COEFFICIENT_STR
        assert doc["scatter_baseline_x"] == "baseline_loss"
        assert doc["strength_unit"] == "counter_direction_magnitude"
        assert doc["sampling"]["n_completions"] == 3
        # Nested concept -> prefix -> datapoints -> completions.
        assert len(doc["concepts"]) == 1
        concept_doc = doc["concepts"][0]
        assert concept_doc["concept_id"] == _CONCEPT_ID
        assert concept_doc["concept_label"] == "dogs"
        prefix_doc = concept_doc["prefixes"][0]
        assert prefix_doc["core_prefix_id"] == _CORE_PREFIX_ID
        datapoints = prefix_doc["datapoints"]
        # Sorted distinct strength_index {0, 1}.
        assert [dp["strength_index"] for dp in datapoints] == [0, 1]
        # Metrics present (not silently null).
        assert datapoints[0]["is_control"] is True
        assert datapoints[1]["is_control"] is False
        assert datapoints[0]["adversarial_strength"] == 0.0
        assert datapoints[1]["adversarial_strength"] == 1.0
        assert all(dp["primary_margin_y"] is not None for dp in datapoints)
        # Exactly n_completions completion strings per datapoint.
        for dp in datapoints:
            assert len(dp["completions"]) == 3
            assert all(isinstance(c, str) for c in dp["completions"])


def test_generate_samples_core_reads_roundtrip_aggregates(tmp_path: Path) -> None:
    paths = _write_finished_roundtrip_run(tmp_path / "out")
    _run_core(
        paths,
        steering_metric="roundtrip",
        generation=_generation_config(
            n_completions=1,
            methods=["diffmean_decoder_only"],
        ),
    )

    doc = yaml.safe_load(
        (
            paths.samples_dir()
            / "positive"
            / "diffmean_decoder_only__additive_direction.yaml"
        ).read_text()
    )
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    assert [dp["strength_index"] for dp in datapoints] == [0, 1]
    assert all("primary_roundtrip_y" in dp for dp in datapoints)
    assert all("primary_margin_y" not in dp for dp in datapoints)
    assert datapoints[0]["primary_roundtrip_y"] == 80.0
    assert len(datapoints[0]["completions"]) == 1


def test_generate_samples_core_mixed_artifacts_roundtrip_metric_wins(
    tmp_path: Path,
) -> None:
    paths = _write_mixed_metric_run(
        tmp_path / "out",
        stale_legacy_coefficient="8.0",
        stale_roundtrip_coefficient=_COEFFICIENT_STR,
    )
    _run_core(
        paths,
        steering_metric="roundtrip",
        generation=_generation_config(
            n_completions=1,
            methods=["diffmean_decoder_only"],
        ),
    )

    doc = yaml.safe_load(
        (
            paths.samples_dir()
            / "positive"
            / "diffmean_decoder_only__additive_direction.yaml"
        ).read_text()
    )
    assert doc["representative_coefficient"] == _COEFFICIENT_STR
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    assert all("primary_roundtrip_y" in dp for dp in datapoints)
    assert all("primary_margin_y" not in dp for dp in datapoints)
    assert datapoints[0]["primary_roundtrip_y"] == 80.0


def test_generate_samples_core_mixed_artifacts_class_margin_metric_wins(
    tmp_path: Path,
) -> None:
    paths = _write_mixed_metric_run(
        tmp_path / "out",
        stale_legacy_coefficient=_COEFFICIENT_STR,
        stale_roundtrip_coefficient="8.0",
    )
    _run_core(
        paths,
        steering_metric="bpb_class_margin",
        generation=_generation_config(
            n_completions=1,
            methods=["diffmean_decoder_only"],
        ),
    )

    doc = yaml.safe_load(
        (
            paths.samples_dir()
            / "positive"
            / "diffmean_decoder_only__additive_direction.yaml"
        ).read_text()
    )
    assert doc["representative_coefficient"] == _COEFFICIENT_STR
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    assert all("primary_margin_y" in dp for dp in datapoints)
    assert all("primary_roundtrip_y" not in dp for dp in datapoints)
    assert datapoints[0]["primary_margin_y"] == 1.0


# ---------------------------------------------------------------------------
# F1 (regression): a finished run carrying BOTH a positive AND a negative family
# must generate a YAML for EACH. core_prefix_id is MODE-SPECIFIC, so a global
# (mode-agnostic) concept/prefix selection hands the negative family a positive
# family's prefix id -> representative_attack_for_datapoint raises (mode mismatch) and
# the whole run aborts. The fix derives selection + the strength axis PER FAMILY
# from the mode-correct candidate rows. This fixture uses the REAL mode-specific
# attack_rows.core_prefix_id so the positive/negative prefix ids genuinely differ.
# ---------------------------------------------------------------------------

# Both families steer concept_id=1 ("dogs") at coefficient 4.0 over the additive
# counter sweep {0.0, 1.0}.
_MM_PREFIX_TEXT = "dogs are great"
# (method_name, mode) for the positive + negative families under test.
_MM_FAMILIES = (
    ("diffmean_decoder_only", "positive"),
    ("neg_diffmean_decoder_only", "negative"),
)


def _mm_core_prefix_id(mode: str) -> str:
    # The REAL mode-specific id: positive and negative differ (sha256 over
    # concept|mode|prefix), exactly as the producer emits them.
    return _real_core_prefix_id(
        concept_id=_CONCEPT_ID,
        mode=SteeringMode(mode),
        prefix_text=_MM_PREFIX_TEXT,
    )


def _mm_candidate_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
) -> dict[str, Any]:
    row = _candidate_row(
        method_name=method_name,
        mode=mode,
        strength_index=strength_index,
        adversarial_strength=adversarial_strength,
    )
    # Mode-correct core_prefix_id + a humanised prefix text.
    row["core_prefix_id"] = _mm_core_prefix_id(mode)
    row["prefix_text"] = _MM_PREFIX_TEXT
    return row


def _mm_per_candidate_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
    is_control: bool,
) -> dict[str, Any]:
    row = _per_candidate_row(
        method_name=method_name,
        mode=mode,
        strength_index=strength_index,
        adversarial_strength=adversarial_strength,
        is_control=is_control,
    )
    return row


def _mm_macro_row(
    *,
    method_name: str,
    mode: str,
    strength_index: int,
    adversarial_strength: float,
    is_control: bool,
) -> dict[str, Any]:
    row = _macro_row(
        method_name=method_name,
        mode=mode,
        strength_index=strength_index,
        adversarial_strength=adversarial_strength,
        is_control=is_control,
    )
    return row


def _write_multimode_finished_run(output_root: Path) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    per_candidate: list[dict[str, Any]] = []
    macro: list[dict[str, Any]] = []
    for method_name, mode in _MM_FAMILIES:
        for strength_index, strength in ((0, 0.0), (1, 1.0)):
            is_control = strength_index == 0
            candidates.append(
                _mm_candidate_row(
                    method_name=method_name,
                    mode=mode,
                    strength_index=strength_index,
                    adversarial_strength=strength,
                )
            )
            per_candidate.append(
                _mm_per_candidate_row(
                    method_name=method_name,
                    mode=mode,
                    strength_index=strength_index,
                    adversarial_strength=strength,
                    is_control=is_control,
                )
            )
            macro.append(
                _mm_macro_row(
                    method_name=method_name,
                    mode=mode,
                    strength_index=strength_index,
                    adversarial_strength=strength,
                    is_control=is_control,
                )
            )
    pq.write_table(
        pa.Table.from_pylist(candidates, schema=CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    pq.write_table(
        pa.Table.from_pylist(per_candidate, schema=PER_CANDIDATE_AGG_SCHEMA),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _with_feature_diffmean_macro_targets(macro),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    return paths


def test_generate_samples_core_emits_yaml_for_both_modes(tmp_path: Path) -> None:
    # F1: with BOTH a positive and a negative family present (mode-specific
    # core_prefix_ids), a YAML must be written for EACH mode. Before the fix, the
    # mode-agnostic global selection handed the negative family the positive
    # family's prefix id and representative_attack_for_datapoint raised, aborting the
    # whole run with NO negative YAML (and, in the default registry that emits both
    # families, no positive one either).
    paths = _write_multimode_finished_run(tmp_path / "out")
    _run_core(paths)

    samples_dir = paths.samples_dir()
    positive_yaml = (
        samples_dir
        / "positive"
        / "diffmean_decoder_only__additive_direction.yaml"
    )
    negative_yaml = (
        samples_dir
        / "negative"
        / "neg_diffmean_decoder_only__additive_direction.yaml"
    )
    assert positive_yaml.exists()
    assert negative_yaml.exists()

    for out_path, method_name, mode in (
        (positive_yaml, "diffmean_decoder_only", "positive"),
        (negative_yaml, "neg_diffmean_decoder_only", "negative"),
    ):
        doc = yaml.safe_load(out_path.read_text())
        assert doc["method_name"] == method_name
        assert doc["mode"] == mode
        assert doc["representative_coefficient"] == _COEFFICIENT_STR
        prefix_doc = doc["concepts"][0]["prefixes"][0]
        # The selected prefix id is the MODE-CORRECT one for this family.
        assert prefix_doc["core_prefix_id"] == _mm_core_prefix_id(mode)
        datapoints = prefix_doc["datapoints"]
        assert [dp["strength_index"] for dp in datapoints] == [0, 1]
        assert datapoints[0]["adversarial_strength"] == 0.0
        assert datapoints[1]["adversarial_strength"] == 1.0
        for dp in datapoints:
            assert len(dp["completions"]) == 3
            assert all(isinstance(c, str) for c in dp["completions"])


def test_generate_samples_core_methods_whitelist(tmp_path: Path) -> None:
    # The optional methods whitelist restricts which families are emitted.
    paths = _write_finished_run(tmp_path / "out")
    _run_core(paths, generation=_generation_config(methods=["diffmean_decoder_only"]))

    mode_dir = paths.samples_dir() / "positive"
    written = sorted(p.name for p in mode_dir.glob("*.yaml"))
    assert written == ["diffmean_decoder_only__additive_direction.yaml"]


def test_generate_samples_core_rejects_encdec_target(tmp_path: Path) -> None:
    # Encoder-decoder sampling is a deliberate follow-up; an encdec target must be
    # rejected loudly, never silently routed through the decoder-only path.
    paths = _write_finished_run(tmp_path / "out")
    with pytest.raises(AxBenchBpbAdvError, match="encoder-decoder"):
        _run_core(paths, target_model="encdec")


# ---------------------------------------------------------------------------
# M-4: the orchestrator threads LoraParams so a lora_decoder_only family
# generates WITH its adapter (mirroring the scorer's lora threading).
# ---------------------------------------------------------------------------

# The mini fixture ships a REAL tiny LoRA adapter for concept_1 at
# baselines/lora/decoder_only/positive/concept_0001.pt (rank 4, layer 1 -- a
# valid block index for the 2-layer tiny model, non-zero lora_B so it genuinely
# contributes to the forward). The lora method's coefficient grid is a single
# [None]; `_build_lora_intervention` ignores the coefficient, so the family's
# representative coefficient is the string "None".
_LORA_COEFFICIENT_STR = "None"
_LORA_RANK = 4
_LORA_ALPHA = 8.0
_LORA_LAYER = 1


def _lora_candidate_row(*, strength_index: int) -> dict[str, Any]:
    candidate_id = f"lora_decoder_only|positive|s{strength_index}"
    return {
        "candidate_id": candidate_id,
        "attack_row_id": candidate_id,
        "core_prefix_id": _CORE_PREFIX_ID,
        "concept_rich_sample_id": "rich",
        "concept_avoiding_sample_id": "avoid",
        "concept_id": _CONCEPT_ID,
        "mode": "positive",
        "method_name": "lora_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": _LORA_COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "prefix_text": "dogs are great",
        "prefix_hash": "ph",
        "source_prompt_tokens": list(_PROMPT_TOKENS),
        "source_prompt_token_digest": "srcdig",
        "attacked_prompt_text": "dogs are great",
        "attacked_prompt_tokens": list(_PROMPT_TOKENS),
        "attacked_prompt_token_digest": "attdig",
        "concept_rich_output_tokens": [11, 12],
        "concept_avoiding_output_tokens": [21, 22],
        "concept_rich_output_digest": "rdig",
        "concept_avoiding_output_digest": "adig",
        "editable_token_start": 0,
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": 0.0,
        "candidate_metadata_json": '{"candidate":"no_adversary_control"}',
    }


def _lora_per_candidate_row(*, strength_index: int, is_control: bool) -> dict[str, Any]:
    return {
        "concept_id": _CONCEPT_ID,
        "mode": "positive",
        "method_name": "lora_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": _LORA_COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0,
        "baseline_primary_margin": 0.9 if is_control else None,
        "general_val_bpb": None,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 2,
        "n_avoid_output_tokens": 2,
        "n_rich_output_bytes": 2,
        "n_avoid_output_bytes": 2,
        "is_control": is_control,
    }


def _lora_macro_row(*, strength_index: int, is_control: bool) -> dict[str, Any]:
    return {
        "mode": "positive",
        "method_name": "lora_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": _LORA_COEFFICIENT_STR,
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": strength_index,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0,
        "baseline_primary_margin": 0.9 if is_control else None,
        "general_val_bpb": None,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": is_control,
    }


def _write_lora_finished_run(output_root: Path) -> AxBenchBpbAdvPaths:
    """A finished run dir with a single ``lora_decoder_only`` family.

    Strength_index 0 is the k=0 control (magnitude 0.0 -> pure lora, no additive
    counter hook), so the lora-kind intervention is exercised in isolation: any
    completion proves the adapter attached.
    """

    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [_lora_candidate_row(strength_index=0)], schema=CANDIDATES_SCHEMA
        ),
        paths.candidates_path(),
    )
    pq.write_table(
        pa.Table.from_pylist(
            [_lora_per_candidate_row(strength_index=0, is_control=True)],
            schema=PER_CANDIDATE_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _with_feature_diffmean_macro_targets(
                [_lora_macro_row(strength_index=0, is_control=True)]
            ),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    return paths


def test_generate_samples_core_threads_lora_so_adapter_generates(
    tmp_path: Path,
) -> None:
    # M-4: a lora_decoder_only family must generate WITH its adapter -- the SAME
    # intervention the scorer scored. The orchestrator builds a lora-kind
    # Intervention (Intervention.lora(adapter_path=...)) whose apply_intervention
    # REQUIRES the LoraParams bundle; threading `lora=` end-to-end is what lets
    # the adapter attach. Has teeth: dropping the `lora=` thread (so lora is None)
    # makes apply_intervention raise "apply_intervention(lora) requires the lora
    # kwarg (LoraParams)" -- the assert below would turn RED.
    paths = _write_lora_finished_run(tmp_path / "out")
    lora = LoraParams(
        cfg=LoraConfig(rank=_LORA_RANK, alpha=_LORA_ALPHA), layer=_LORA_LAYER
    )
    _run_core(paths, lora=lora)

    out_path = (
        paths.samples_dir()
        / "positive"
        / "lora_decoder_only__additive_direction.yaml"
    )
    assert out_path.exists()
    doc = yaml.safe_load(out_path.read_text())
    assert doc["method_name"] == "lora_decoder_only"
    assert doc["representative_coefficient"] == _LORA_COEFFICIENT_STR
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    # The single control datapoint generated n_completions strings WITH the
    # adapter attached (proof the lora bundle threaded all the way down).
    assert [dp["strength_index"] for dp in datapoints] == [0]
    completions = datapoints[0]["completions"]
    assert len(completions) == 3
    assert all(isinstance(c, str) for c in completions)


def test_generate_samples_core_lora_family_without_params_fails_loud(
    tmp_path: Path,
) -> None:
    # The negative: WITHOUT the threaded LoraParams the lora-kind intervention
    # cannot attach -- apply_intervention fails loud (AxBenchBpbError, from the
    # axbench_bpb layer) rather than silently generating an un-adapted (wrong)
    # completion. This pins that the threading is load-bearing (and that a None
    # lora is not silently swallowed).
    paths = _write_lora_finished_run(tmp_path / "out")
    with pytest.raises(AxBenchBpbError, match="requires the lora kwarg"):
        _run_core(paths, lora=None)


# ---------------------------------------------------------------------------
# M-2: a family with a representative coefficient but an EMPTY per_candidate
# strength slice is warned + skipped (symmetric with the no-rep-coeff skip),
# never written as a degenerate empty-datapoint YAML.
# ---------------------------------------------------------------------------

# The macro selects coefficient "4.0" (its control margin is present) but the
# CANDIDATES table carries this family's rows ONLY at "8.0" -- so the candidate
# slice at the representative "4.0" is empty (the strength axis is now sourced
# from candidates, not per_candidate, so the skip keys on the candidates).
_REP_COEFFICIENT = "4.0"
_ORPHAN_COEFFICIENT = "8.0"


def _write_empty_strength_slice_run(output_root: Path) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)

    # candidates ONLY at the orphan "8.0": no candidate at the representative
    # "4.0", so _family_candidates(... "4.0") == [] -> empty candidate slice.
    candidate_orphan = _candidate_row(
        method_name="diffmean_decoder_only",
        mode="positive",
        strength_index=0,
        adversarial_strength=0.0,
    )
    candidate_orphan["coefficient_str"] = _ORPHAN_COEFFICIENT
    pq.write_table(
        pa.Table.from_pylist([candidate_orphan], schema=CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )

    # macro at "4.0" (control margin present -> "4.0" is selected as the family's
    # representative coefficient).
    macro_row = _macro_row(
        method_name="diffmean_decoder_only",
        mode="positive",
        strength_index=0,
        adversarial_strength=0.0,
        is_control=True,
    )
    pq.write_table(
        pa.Table.from_pylist(
            _with_feature_diffmean_macro_targets([macro_row]),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )

    # per_candidate also at the orphan "8.0" (irrelevant to the skip now: the
    # candidate slice gates it). Present so the table is non-empty / schema-valid.
    per_candidate_orphan = _per_candidate_row(
        method_name="diffmean_decoder_only",
        mode="positive",
        strength_index=0,
        adversarial_strength=0.0,
        is_control=True,
    )
    per_candidate_orphan["coefficient_str"] = _ORPHAN_COEFFICIENT
    pq.write_table(
        pa.Table.from_pylist([per_candidate_orphan], schema=PER_CANDIDATE_AGG_SCHEMA),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    return paths


def test_generate_samples_core_warns_and_skips_empty_strength_slice(
    tmp_path: Path, caplog: Any
) -> None:
    # M-2: the family HAS a representative coefficient ("4.0") but its CANDIDATE
    # slice at that coefficient is EMPTY (candidates only have "8.0" rows). The
    # orchestrator must logger.warning + SKIP -- symmetric with the
    # no-representative-coefficient skip -- and write NO YAML (no degenerate
    # empty-datapoint doc). Teeth: without the skip this falls through to
    # _family_strength_axis / build_family_yaml_doc with no candidate rows.
    paths = _write_empty_strength_slice_run(tmp_path / "out")
    with caplog.at_level(logging.WARNING):
        _run_core(paths)

    # Nothing written for the skipped family.
    samples_dir = paths.samples_dir()
    written = sorted(p for p in samples_dir.rglob("*.yaml")) if samples_dir.exists() else []
    assert written == []
    # The skip is LOUD: a warning naming the empty candidate slice + the family.
    text = caplog.text
    assert "no matching candidate rows" in text
    assert "diffmean_decoder_only" in text
    assert _REP_COEFFICIENT in text


# ---------------------------------------------------------------------------
# GAP A.2: a family whose control rows give NO usable control margin (all
# baseline_primary_margin=None) selects representative_coefficient=None ->
# run_generate_samples_core must WARN + skip it (no YAML), while a paired GOOD
# family (real control margin) IS written. Pins the None-representative skip arm
# of the orchestrator (sample_generation ~:910).
# ---------------------------------------------------------------------------

# The good family keeps the default diffmean_decoder_only identity; the bad one
# is a distinct method/mode so it forms its own representative-family key.
_GOOD_METHOD = "diffmean_decoder_only"
_BAD_METHOD = "neg_diffmean_decoder_only"


def _none_margin_macro_row(
    *, method_name: str, mode: str, strength_index: int, is_control: bool
) -> dict[str, Any]:
    # Like _macro_row but with NO control margin (baseline_primary_margin=None on
    # the control rows too) -> _select_representative_coefficient picks None.
    row = _macro_row(
        method_name=method_name,
        mode=mode,
        strength_index=strength_index,
        adversarial_strength=0.0 if strength_index == 0 else 1.0,
        is_control=is_control,
    )
    row["baseline_primary_margin"] = None
    return row


def _write_mixed_margin_run(output_root: Path) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    per_candidate: list[dict[str, Any]] = []
    macro: list[dict[str, Any]] = []
    # GOOD family: positive diffmean with a real control margin (representative
    # coefficient resolves to 4.0). BAD family: a negative family whose control
    # rows carry NO margin (representative coefficient resolves to None).
    for strength_index, strength in ((0, 0.0), (1, 1.0)):
        is_control = strength_index == 0
        candidates.append(
            _candidate_row(
                method_name=_GOOD_METHOD,
                mode="positive",
                strength_index=strength_index,
                adversarial_strength=strength,
            )
        )
        per_candidate.append(
            _per_candidate_row(
                method_name=_GOOD_METHOD,
                mode="positive",
                strength_index=strength_index,
                adversarial_strength=strength,
                is_control=is_control,
            )
        )
        macro.append(
            _macro_row(
                method_name=_GOOD_METHOD,
                mode="positive",
                strength_index=strength_index,
                adversarial_strength=strength,
                is_control=is_control,
            )
        )
        # BAD family: a NEGATIVE-mode candidate (mode-correct core_prefix_id) whose
        # macro control margin is None -> no representative coefficient.
        bad_candidate = _candidate_row(
            method_name=_BAD_METHOD,
            mode="negative",
            strength_index=strength_index,
            adversarial_strength=strength,
        )
        bad_candidate["core_prefix_id"] = _real_core_prefix_id(
            concept_id=_CONCEPT_ID,
            mode=SteeringMode.NEGATIVE,
            prefix_text="dogs are great",
        )
        candidates.append(bad_candidate)
        macro.append(
            _none_margin_macro_row(
                method_name=_BAD_METHOD,
                mode="negative",
                strength_index=strength_index,
                is_control=is_control,
            )
        )

    pq.write_table(
        pa.Table.from_pylist(candidates, schema=CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    pq.write_table(
        pa.Table.from_pylist(per_candidate, schema=PER_CANDIDATE_AGG_SCHEMA),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _with_feature_diffmean_macro_targets(macro),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    return paths


def test_generate_samples_core_skips_none_representative_writes_good(
    tmp_path: Path, caplog: Any
) -> None:
    paths = _write_mixed_margin_run(tmp_path / "out")
    with caplog.at_level(logging.WARNING):
        _run_core(paths)

    samples_dir = paths.samples_dir()
    written = sorted(p.name for p in samples_dir.rglob("*.yaml"))
    # ONLY the good family was written; the None-representative family was skipped.
    assert written == [f"{_GOOD_METHOD}__additive_direction.yaml"]
    good_yaml = (
        samples_dir / "positive" / f"{_GOOD_METHOD}__additive_direction.yaml"
    )
    assert good_yaml.exists()

    # The skip is LOUD and names the bad family + the no-control-margin reason.
    text = caplog.text
    assert "no representative coefficient" in text
    assert _BAD_METHOD in text


# ---------------------------------------------------------------------------
# GAP A.6: _validate_table fails loud on schema drift. A candidates parquet with
# a dropped/renamed column must abort run_generate_samples_core with the
# schema-mismatch error (sample_generation ~:714), never silently proceed on a
# wrong-shaped table.
# ---------------------------------------------------------------------------


def test_generate_samples_core_rejects_schema_drift(tmp_path: Path) -> None:
    paths = _write_finished_run(tmp_path / "out")
    # Rewrite candidates.parquet with a DROPPED column ("concept_id" renamed) so
    # its schema no longer equals CANDIDATES_SCHEMA.
    candidates = pq.read_table(paths.candidates_path())
    drifted = candidates.rename_columns(
        [
            "concept_id_renamed" if name == "concept_id" else name
            for name in candidates.schema.names
        ]
    )
    pq.write_table(drifted, paths.candidates_path())

    with pytest.raises(AxBenchBpbAdvError, match="unexpected schema"):
        _run_core(paths)


# ---------------------------------------------------------------------------
# GAP A.7 (BEHAVIORAL): a faster_gcg family whose candidate rows carry DISTINCT
# attacked_prompt_tokens per strength_index. For prompt-token attacks the BASE
# intervention is constant (apply_additive_counter is a no-op for faster_gcg) and
# the ATTACKED PROMPT varies per strength -- the regime no test currently
# exercises. Because the sampling seeds (base_seed + i) are REUSED across
# strengths and the base intervention is identical, the ONLY thing that can make
# the completions differ across strengths is the per-strength prompt. So distinct
# completions across strengths is direct evidence the generation ran per-strength
# from the per-strength prompt.
# ---------------------------------------------------------------------------

_GCG_METHOD = "diffmean_decoder_only"
# Two clearly-distinct prompts (different token ids) so the model's continuations
# diverge under the (identical) base intervention + (identical) seeds.
_GCG_PROMPT_S0 = [0, 5, 6, 7]
_GCG_PROMPT_S1 = [0, 40, 41, 42]


def _gcg_candidate_row(
    *, strength_index: int, attacked_prompt_tokens: list[int]
) -> dict[str, Any]:
    row = _candidate_row(
        method_name=_GCG_METHOD,
        mode="positive",
        strength_index=strength_index,
        adversarial_strength=float(strength_index),
    )
    # faster_gcg / optimized_tokens / fixed_context, distinct attacked prompt.
    row["adversarial_method"] = "faster_gcg"
    row["strength_unit"] = "optimized_tokens"
    row["context_signal_mode"] = "fixed_context"
    row["attacked_prompt_tokens"] = list(attacked_prompt_tokens)
    row["attacked_prompt_text"] = "".join(
        chr(token_id) for token_id in attacked_prompt_tokens if token_id != 0
    )
    return row


def _gcg_per_candidate_row(*, strength_index: int, is_control: bool) -> dict[str, Any]:
    row = _per_candidate_row(
        method_name=_GCG_METHOD,
        mode="positive",
        strength_index=strength_index,
        adversarial_strength=float(strength_index),
        is_control=is_control,
    )
    row["adversarial_method"] = "faster_gcg"
    row["strength_unit"] = "optimized_tokens"
    row["context_signal_mode"] = "fixed_context"
    return row


def _gcg_macro_row(*, strength_index: int, is_control: bool) -> dict[str, Any]:
    row = _macro_row(
        method_name=_GCG_METHOD,
        mode="positive",
        strength_index=strength_index,
        adversarial_strength=float(strength_index),
        is_control=is_control,
    )
    row["adversarial_method"] = "faster_gcg"
    row["strength_unit"] = "optimized_tokens"
    row["context_signal_mode"] = "fixed_context"
    return row


def _write_gcg_per_strength_prompt_run(output_root: Path) -> AxBenchBpbAdvPaths:
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)

    candidates = [
        _gcg_candidate_row(strength_index=0, attacked_prompt_tokens=_GCG_PROMPT_S0),
        _gcg_candidate_row(strength_index=1, attacked_prompt_tokens=_GCG_PROMPT_S1),
    ]
    per_candidate = [
        _gcg_per_candidate_row(strength_index=0, is_control=True),
        _gcg_per_candidate_row(strength_index=1, is_control=False),
    ]
    macro = [
        _gcg_macro_row(strength_index=0, is_control=True),
        _gcg_macro_row(strength_index=1, is_control=False),
    ]
    pq.write_table(
        pa.Table.from_pylist(candidates, schema=CANDIDATES_SCHEMA),
        paths.candidates_path(),
    )
    pq.write_table(
        pa.Table.from_pylist(per_candidate, schema=PER_CANDIDATE_AGG_SCHEMA),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _with_feature_diffmean_macro_targets(macro),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )
    return paths


def test_generate_samples_core_faster_gcg_prompt_varies_per_strength(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_gcg_per_strength_prompt_run(tmp_path / "out")

    # Spy that WRAPS the real generation (calls through to the real tiny model so
    # the YAML is produced from genuine completions) while recording the
    # prompt_tokens threaded in per (strength) call. This asserts the precise
    # behavioral claim -- the generation ran per strength FROM the per-strength
    # attacked prompt -- without depending on the tiny 2-layer model's greedy
    # output being prompt-sensitive (it collapses to a repeated token regardless).
    real_generate = sg.generate_completions_for_datapoint
    seen_prompts: list[list[int]] = []

    def _spy_generate(**kwargs: Any) -> list[str]:
        seen_prompts.append(list(kwargs["prompt_tokens"]))
        return real_generate(**kwargs)

    monkeypatch.setattr(sg, "generate_completions_for_datapoint", _spy_generate)

    _run_core(
        paths,
        generation=_generation_config(n_completions=2, max_new_tokens=6),
    )

    out_path = (
        paths.samples_dir()
        / "positive"
        / f"{_GCG_METHOD}__faster_gcg.yaml"
    )
    assert out_path.exists()
    doc = yaml.safe_load(out_path.read_text())
    assert doc["adversarial_method"] == "faster_gcg"
    assert doc["strength_unit"] == "optimized_tokens"
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    assert [dp["strength_index"] for dp in datapoints] == [0, 1]
    # Each datapoint still got its n_completions real strings.
    for dp in datapoints:
        assert len(dp["completions"]) == 2
        assert all(isinstance(c, str) for c in dp["completions"])
    # TEETH: generation ran once per strength, and EACH ran from that strength's
    # OWN attacked prompt. For a prompt-token attack the base intervention is
    # constant and the prompt is what varies per strength; if the orchestrator had
    # reused one prompt for every strength, both calls would carry _GCG_PROMPT_S0
    # and this ordered-equality check would fail.
    assert seen_prompts == [_GCG_PROMPT_S0, _GCG_PROMPT_S1]
    # Issue #543: each datapoint SURFACES its per-strength attacked prompt + edit +
    # objective, and the displayed prompt equals the one generation ran from.
    for dp, expected_tokens in zip(datapoints, (_GCG_PROMPT_S0, _GCG_PROMPT_S1), strict=True):
        assert "attacked_prompt_text" in dp
        assert "replacement_text" in dp
        assert "objective_value" in dp
        expected_text = "".join(chr(tid) for tid in expected_tokens if tid != 0)
        assert dp["attacked_prompt_text"] == expected_text


# ---------------------------------------------------------------------------
# Issue #543: multi-candidate-per-datapoint selection (the regime no synthetic
# test covered before -- two GCG candidates of ONE (concept, prefix, strength)
# carry DIFFERENT optimized prompts + objectives; the STRONGEST (lowest g) must
# be the one generated-from AND surfaced).
# ---------------------------------------------------------------------------


def _gcg_candidate_row_full(
    *,
    strength_index: int,
    candidate_id: str,
    attacked_prompt_tokens: list[int],
    objective_value: float | None,
) -> dict[str, Any]:
    row = _gcg_candidate_row(
        strength_index=strength_index, attacked_prompt_tokens=attacked_prompt_tokens
    )
    row["candidate_id"] = candidate_id
    row["attack_row_id"] = candidate_id
    row["objective_value"] = objective_value
    return row


def test_generate_samples_core_faster_gcg_multi_candidate_selects_min_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    weak = [0, 5, 6, 7]
    strong = [0, 40, 41, 42]
    candidates = [
        _gcg_candidate_row(strength_index=0, attacked_prompt_tokens=[0, 5, 6, 7]),
        # strength 1 has TWO candidates; the stronger attack (lower g) carries `strong`
        # and the LARGER candidate_id, so a min-candidate_id selector would mis-pick.
        _gcg_candidate_row_full(
            strength_index=1, candidate_id="aaa_weak",
            attacked_prompt_tokens=weak, objective_value=-0.1,
        ),
        _gcg_candidate_row_full(
            strength_index=1, candidate_id="zzz_strong",
            attacked_prompt_tokens=strong, objective_value=-0.9,
        ),
    ]
    per_candidate = [
        _gcg_per_candidate_row(strength_index=0, is_control=True),
        _gcg_per_candidate_row(strength_index=1, is_control=False),
    ]
    macro = [
        _gcg_macro_row(strength_index=0, is_control=True),
        _gcg_macro_row(strength_index=1, is_control=False),
    ]
    pq.write_table(
        pa.Table.from_pylist(candidates, schema=CANDIDATES_SCHEMA), paths.candidates_path()
    )
    pq.write_table(
        pa.Table.from_pylist(per_candidate, schema=PER_CANDIDATE_AGG_SCHEMA),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _with_feature_diffmean_macro_targets(macro),
            schema=MACRO_AGG_SCHEMA,
        ),
        paths.aggregates_dir() / "macro.parquet",
    )

    real_generate = sg.generate_completions_for_datapoint
    seen_prompts: list[list[int]] = []

    def _spy_generate(**kwargs: Any) -> list[str]:
        seen_prompts.append(list(kwargs["prompt_tokens"]))
        return real_generate(**kwargs)

    monkeypatch.setattr(sg, "generate_completions_for_datapoint", _spy_generate)
    _run_core(paths, generation=_generation_config(n_completions=1, max_new_tokens=4))

    out_path = (
        paths.samples_dir() / "positive" / f"{_GCG_METHOD}__faster_gcg.yaml"
    )
    doc = yaml.safe_load(out_path.read_text())
    dp1 = doc["concepts"][0]["prefixes"][0]["datapoints"][1]
    # TEETH: the strongest attack (g=-0.9, `strong` tokens) is generated-from AND
    # surfaced -- proving objective_value (not candidate_id) drives the choice.
    assert seen_prompts[1] == strong
    assert dp1["objective_value"] == -0.9
    assert dp1["attacked_prompt_text"] == "".join(chr(tid) for tid in strong if tid != 0)


_GOLDEN_ADDITIVE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "golden"
    / "issue_543_additive_diffmean_decoder_only__additive_direction.yaml"
)


def test_generate_samples_core_additive_yaml_byte_identical_to_pre_543(
    tmp_path: Path,
) -> None:
    # EXACT-EQUALITY against the existing (#538) implementation: the additive
    # samples YAML must not move a single byte -- issue #543 only ADDS prompt-token
    # fields to faster_gcg/gepa families.
    paths = _write_finished_run(tmp_path / "out")
    _run_core(paths, generation=_generation_config(n_completions=3, max_new_tokens=6))
    produced = (
        paths.samples_dir()
        / "positive"
        / "diffmean_decoder_only__additive_direction.yaml"
    ).read_bytes()
    assert produced == _GOLDEN_ADDITIVE.read_bytes(), (
        "additive samples YAML drifted from the pre-#543 golden -- additive output "
        "must stay byte-identical (issue #543 only adds prompt-token fields)"
    )
