"""INTEGRATION: real GCG on d12_base -- shared-prefix determinism and tolerance.

Two real-model guards on the Faster-GCG optimizer (both ``slow``, GPU):

* ``test_real_gcg_batched_optimizer_deterministic_and_bounded`` -- the candidate
  search now ranks by the shared-prefix loss-only objective,
  run under ``torch.no_grad()`` so the B-sample forward retains no autograd graph).
  That deliberately drops byte-identity to the grad path: on compiled flex-attention
  the ``no_grad`` reduction/stride path diverges (~1e-1) and the batch shape adds
  ~5e-2 bf16 rounding -- the accepted cost of a tractable search. So this test pins
  the two properties that DO still hold: (1) the optimizer is run-to-run
  DETERMINISTIC (same cfg+seed -> byte-identical full row), and (2) the shared
  objective stays within a bounded tolerance of both the test-only full-sequence
  oracle and grad-path reference at a fixed token sequence (no trajectory
  sensitivity).
* ``test_opt_byte_identical_under_real_diffmean_residual_hook`` -- OPT #1 (cached
  embedding hoist) and OPT #2 (loss-only == gradient-path loss) remain BYTE-IDENTICAL
  on the SINGLE-prompt primitives (``teacher_forced_bpb_loss`` / ``_gradient``, which
  batching left untouched -- the encdec path still loops them), under a real diffmean
  residual hook.

Marked ``slow`` (real-model, GPU): deselected by the default ``-m 'not slow ...'``.
Falls back to skip (not fail) when d12_base / CUDA are unavailable.
"""

from __future__ import annotations

import gc
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import torch as t

import steering.conditional.feature_gcg.core as gcg_core
import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
from nanochat.checkpoint_manager import load_model
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.edit_primitives import EditableSpan
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention, SteeringMode
from steering.eval.axbench_bpb.vocab_restriction import feature_vocab_allowed_ids
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.model_optimize import BpbObjectiveCallbacks
from steering.eval.axbench_bpb_adv.optimize import (
    build_universal_gcg_candidates,
    objective_gradient,
    objective_value,
)
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import _build_token_bytes
from steering.eval.axbench_bpb_adv.surrogate import (
    bpb_surrogate_embedding_weight,
    teacher_forced_bpb_gradient,
    teacher_forced_bpb_gradient_batch,
    teacher_forced_bpb_loss_shared_prefix_kv_batch,
)
from steering.eval.axbench_bpb_adv.token_filter import build_allowed_token_ids
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.axbench_bpb_adv.faster_gcg_oracle import build_faster_gcg_candidates
from tests.axbench_bpb_adv.full_sequence_reference import (
    teacher_forced_bpb_loss,
    teacher_forced_bpb_loss_full_sequence_reference,
)
from tests.axbench_bpb_adv.functional._d12_base_harness import (
    D12_DIR,
    D12_STEP,
    D12_TAG,
    load_d12_base,
    make_objective,
    single_pair_attack_rows,
)
from tests.axbench_bpb_adv.legacy_surrogate_oracle import (
    legacy_teacher_forced_bpb_gradient,
    legacy_teacher_forced_bpb_gradient_batch,
)


class _PreOptObjective:
    """Faithful PRE-OPTIMIZATION reference for ``BpbObjectiveCallbacks``.

    Mirrors the original ``_evaluate``: BOTH ``objective()`` and
    ``gradient_provider()`` run the full grad-enabled
    ``teacher_forced_bpb_gradient`` forward + embedding VJP
    with NO ``embedding_weight`` kwarg (so the embedding matrix is rebuilt every
    call), cached by (attack_row_id, token-digest) as a single full tuple.
    """

    def __init__(self, bundle: TargetModelBundle, token_bytes: t.Tensor, device: t.device):
        self._bundle = bundle
        self._token_bytes = token_bytes
        self._device = device
        self._cache: dict[tuple[str, str], tuple[float, float, t.Tensor, t.Tensor]] = {}

    def _evaluate(self, row: dict[str, Any], tokens: Sequence[int]):
        digest = "|".join(str(int(x)) for x in tokens)
        key = (str(row["attack_row_id"]), digest)
        if key in self._cache:
            return self._cache[key]
        prompt_tokens = [int(x) for x in tokens]
        rich = teacher_forced_bpb_gradient(
            target_bundle=self._bundle,
            prompt_tokens=prompt_tokens,
            output_tokens=[int(x) for x in row["concept_rich_output_tokens"]],
            prefix_context_segments=(),
            output_token_bytes=self._token_bytes,
            intervention=Intervention.none(),
            device=self._device,
        )
        avoid = teacher_forced_bpb_gradient(
            target_bundle=self._bundle,
            prompt_tokens=prompt_tokens,
            output_tokens=[int(x) for x in row["concept_avoiding_output_tokens"]],
            prefix_context_segments=(),
            output_token_bytes=self._token_bytes,
            intervention=Intervention.none(),
            device=self._device,
        )
        if SteeringMode(str(row["mode"])) == SteeringMode.POSITIVE:
            result = (rich.loss, avoid.loss, rich.token_gradient, avoid.token_gradient)
        else:
            result = (avoid.loss, rich.loss, avoid.token_gradient, rich.token_gradient)
        self._cache[key] = result
        return result

    def objective(self, row: dict[str, Any], tokens: Sequence[int]) -> float:
        ld, lc, _gd, _gc = self._evaluate(row, tokens)
        return objective_value(loss_desired=ld, loss_contrast=lc)

    def gradient_provider(self, row: dict[str, Any], tokens: Sequence[int], _step: int) -> t.Tensor:
        _ld, _lc, gd, gc = self._evaluate(row, tokens)
        return objective_gradient(grad_loss_desired=gd, grad_loss_contrast=gc)


def _multi_attack_rows(
    prompt_tokens: list[int],
    editable_start: int,
    editable_count: int,
    *,
    n_rows: int,
) -> pa.Table:
    base = single_pair_attack_rows(prompt_tokens, editable_start, editable_count).to_pylist()[0]
    rows: list[dict[str, Any]] = []
    for idx in range(n_rows):
        row = dict(base)
        row["attack_row_id"] = f"integ-row-{idx}"
        row["core_prefix_id"] = f"coreprefix{idx + 1:06d}"
        row["concept_rich_sample_id"] = f"rich-{idx}"
        row["concept_avoiding_sample_id"] = f"avoid-{idx}"
        row["prefix_text"] = f"the quick brown fox {idx}"
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)


def _cfg() -> FasterGcgConfig:
    return FasterGcgConfig(
        optimized_token_counts=(0, 2),
        max_steps=2,
        candidate_batch_size=8,
        top_k=16,
        distance_weight=10.0,
        temperature=0.1,
        master_seed=0,
    )


# Upper bound on shared-prefix drift from both reference objectives at a fixed token
# sequence on d12_base. The loss-only path runs under torch.no_grad() (the
# compiled-flex-attention no_grad reduction diverges ~1e-1, see surrogate.py #510)
# and at batch-of-1 adds no padding, so the bound is dominated by the no_grad term.
# Generous to stay robust across GPUs/driver versions; the search-determinism check
# is the exact guard.
_BATCHED_VS_GRAD_TOL = 0.2
_ENCDEC_D12_TAG = "d12_baseline"
_ENCDEC_D12_STEP = 54600
_ENCDEC_D12_DIR = (
    Path.home()
    / ".cache"
    / "nanochat"
    / "conditional_encdec_checkpoints"
    / _ENCDEC_D12_TAG
    / "s128"
    / "dist_geometric"
    / "mean_10.0"
    / "std_10.0"
)


def _full_sequence_reference_objective(
    *,
    bundle: TargetModelBundle,
    token_bytes: t.Tensor,
    row: dict[str, Any],
    prompt_tokens: list[int],
    device: t.device,
) -> float:
    losses = teacher_forced_bpb_loss_full_sequence_reference(
        target_bundle=bundle,
        prompts=[prompt_tokens, prompt_tokens],
        outputs=[
            [int(value) for value in row["concept_rich_output_tokens"]],
            [int(value) for value in row["concept_avoiding_output_tokens"]],
        ],
        prefix_context_segments_list=[(), ()],
        output_token_bytes=token_bytes,
        intervention=Intervention.none(),
        device=device,
        device_batch_size=2,
    )
    desired, contrast = (
        (losses[0], losses[1])
        if SteeringMode(str(row["mode"])) == SteeringMode.POSITIVE
        else (losses[1], losses[0])
    )
    return objective_value(loss_desired=desired, loss_contrast=contrast)


@pytest.mark.slow
def test_real_universal_gcg_batched_stage_callbacks_on_multi_pair_group() -> None:
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    device = t.device("cuda")
    token_bytes = _build_token_bytes(bundle.tokenizer)
    prompt_tokens = [
        int(x) for x in bundle.tokenizer.encode("the quick brown fox jumps over")
    ][:8]
    attack_rows = _multi_attack_rows(
        prompt_tokens,
        editable_start=len(prompt_tokens) - 2,
        editable_count=2,
        n_rows=2,
    )
    cfg = FasterGcgConfig(
        optimized_token_counts=(2,),
        max_steps=1,
        candidate_batch_size=4,
        top_k=8,
        distance_weight=10.0,
        temperature=0.1,
        master_seed=0,
        universal_minibatch_size=2,
    )
    embedding_weight = bpb_surrogate_embedding_weight(bundle.model)
    allowed = [
        int(tid)
        for tid in build_allowed_token_ids(bundle.tokenizer)
        if 0 <= int(tid) < int(embedding_weight.shape[0])
    ]

    def _run_universal() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        objective = make_objective(bundle, token_bytes, device)
        boundary_calls = 0

        def _boundary() -> None:
            nonlocal boundary_calls
            boundary_calls += 1
            objective.clear_eval_cache()

        candidates, triggers = build_universal_gcg_candidates(
            attack_rows=attack_rows,
            tokenizer=bundle.tokenizer,
            cfg=cfg,
            embedding_weight=objective.embedding_weight(),
            allowed_token_ids=allowed,
            objective=objective.objective_batch,
            gradient_provider_batch=objective.gradient_provider_batch,
            group_boundary_callback=_boundary,
        )
        assert boundary_calls == 1
        rows = candidates.to_pylist()
        assert len(rows) == 2
        assert len(triggers) == 1
        trigger = triggers[0]
        assert trigger["n_pairs"] == 2
        assert trigger["accepted_steps"] <= cfg.max_steps
        assert trigger["visited_sequences"] >= trigger["accepted_steps"] + 1
        assert math.isfinite(float(trigger["objective_value"]))
        assert {tuple(row["replacement_tokens"]) for row in rows} == {
            tuple(trigger["trigger_tokens"])
        }
        assert all(
            row["objective_value"] is not None
            and math.isfinite(float(row["objective_value"]))
            for row in rows
        )
        return rows, triggers

    first_rows, first_triggers = _run_universal()
    second_rows, second_triggers = _run_universal()
    assert first_rows == second_rows
    assert first_triggers == second_triggers

    objective = make_objective(bundle, token_bytes, device)
    items = [(row, prompt_tokens) for row in attack_rows.to_pylist()]
    batched_values = objective.objective_batch(items)
    pre_opt = _PreOptObjective(bundle, token_bytes, device)
    reference_values = [
        pre_opt.objective(row, prompt_tokens) for row, _tokens in items
    ]
    full_sequence_values = [
        _full_sequence_reference_objective(
            bundle=bundle,
            token_bytes=token_bytes,
            row=row,
            prompt_tokens=prompt_tokens,
            device=device,
        )
        for row, _tokens in items
    ]
    assert len(batched_values) == len(reference_values) == 2
    for shared, grad_reference, full_reference in zip(
        batched_values,
        reference_values,
        full_sequence_values,
        strict=True,
    ):
        assert all(math.isfinite(value) for value in (shared, grad_reference, full_reference))
        assert abs(shared - full_reference) <= _BATCHED_VS_GRAD_TOL
        assert abs(shared - grad_reference) <= _BATCHED_VS_GRAD_TOL


@pytest.mark.slow
def test_real_d12_small_fixture_has_exact_full_reference_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one stronger-than-required full-vs-cached trajectory regression.

    The production contract intentionally permits legacy-oracle trajectory
    divergence and treats shared-prefix scoring as its own search algorithm.
    This small deterministic fixture currently agrees exactly, so retain that
    useful sensitivity check without generalizing it into a deployment gate.
    Floating-point objective values remain covered separately by the drift gate.
    """

    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    device = t.device("cuda")
    token_bytes = _build_token_bytes(bundle.tokenizer)
    prompt_tokens = [
        int(value)
        for value in bundle.tokenizer.encode("the quick brown fox jumps over")
    ][:8]
    attack_rows = _multi_attack_rows(
        prompt_tokens,
        editable_start=len(prompt_tokens) - 2,
        editable_count=2,
        n_rows=2,
    )
    cfg = FasterGcgConfig(
        optimized_token_counts=(2,),
        max_steps=2,
        candidate_batch_size=8,
        top_k=16,
        distance_weight=10.0,
        temperature=0.1,
        master_seed=1227,
        universal_minibatch_size=2,
    )
    embedding_weight = bpb_surrogate_embedding_weight(bundle.model)
    allowed = [
        int(token_id)
        for token_id in build_allowed_token_ids(bundle.tokenizer)
        if 0 <= int(token_id) < int(embedding_weight.shape[0])
    ]

    active_candidate_trace: list[dict[str, Any]] = []
    original_objectives = gcg_core._objectives_of_candidates

    def _recording_objectives(
        edit_space: Any,
        members: Sequence[Any],
        candidate_targets: Sequence[Any],
        objective_batch: Any,
    ) -> list[float]:
        values = original_objectives(
            edit_space,
            members,
            candidate_targets,
            objective_batch,
        )
        active_candidate_trace.append(
            {
                "member_ids": [str(member[0]["attack_row_id"]) for member in members],
                "candidates": [tuple(int(token) for token in target) for target in candidate_targets],
                "best_index": min(range(len(values)), key=lambda index: values[index]),
            }
        )
        return values

    monkeypatch.setattr(gcg_core, "_objectives_of_candidates", _recording_objectives)

    def _without_float_fields(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                key: value
                for key, value in row.items()
                if key not in {"objective_value", "best_value"}
            }
            for row in rows
        ]

    def _run() -> dict[str, Any]:
        active_candidate_trace.clear()
        objective = make_objective(bundle, token_bytes, device)
        events: list[dict[str, Any]] = []
        candidates, triggers = build_universal_gcg_candidates(
            attack_rows=attack_rows,
            tokenizer=bundle.tokenizer,
            cfg=cfg,
            embedding_weight=objective.embedding_weight(),
            allowed_token_ids=allowed,
            objective=objective.objective_batch,
            gradient_provider_batch=objective.gradient_provider_batch,
            progress_callback=events.append,
            group_boundary_callback=objective.clear_eval_cache,
        )
        return {
            "candidate_trace": list(active_candidate_trace),
            "candidate_rows": _without_float_fields(candidates.to_pylist()),
            "trigger_rows": _without_float_fields(triggers),
            "events": _without_float_fields(events),
        }

    def _full_reference_adapter(**kwargs: Any) -> list[float]:
        kwargs.pop("source_prompts")
        kwargs.pop("editable_spans")
        return teacher_forced_bpb_loss_full_sequence_reference(**kwargs)

    with monkeypatch.context() as reference_patch:
        reference_patch.setattr(
            model_optimize,
            "teacher_forced_bpb_loss_shared_prefix_kv_batch",
            _full_reference_adapter,
        )
        reference = _run()
    shared_prefix = _run()

    assert shared_prefix == reference


@pytest.mark.slow
def test_real_d12_encdec_shared_prefix_drift_for_every_production_k() -> None:
    """Gate cached SDPA against full-sequence attention on production encdec."""

    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    checkpoint = _ENCDEC_D12_DIR / f"model_{_ENCDEC_D12_STEP:06d}.pt"
    if not checkpoint.exists():
        pytest.skip(f"requires encoder-decoder checkpoint {checkpoint}")

    os.environ.setdefault("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
    model, tokenizer, _meta = load_model(
        source="encdec",
        device="cuda",
        phase="eval",
        model_tag=_ENCDEC_D12_TAG,
        step=_ENCDEC_D12_STEP,
        eval_conditioning=EvalConditioning.conditions_within_vocab("axbench_bpb"),
    )
    assert isinstance(model, EncoderDecoderGPT)
    model.eval()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=model.dec_config.sequence_len,
        target_name="encdec",
    )
    base_tokens = [
        int(token_id)
        for token_id in tokenizer.encode(
            "the quick brown fox jumps over the lazy dog " * 12
        )
    ]
    assert len(base_tokens) >= 40
    source = base_tokens[:40]
    outputs = [
        [int(token_id) for token_id in tokenizer.encode(" rich concept completion")][:3],
        [int(token_id) for token_id in tokenizer.encode(" avoiding response")][:2],
    ]
    token_bytes = _build_token_bytes(tokenizer)

    for k in (1, 2, 4, 8, 16, 32):
        prompts = [source, [*source[:-k], *reversed(source[-k:])]]
        segments = [
            (
                ContextFeatureSegment(
                    feature_ids=(1,),
                    buckets=(1,),
                    char_start=0,
                    char_end=len(tokenizer.decode(prompt)),
                ),
            )
            for prompt in prompts
        ]
        common = {
            "target_bundle": bundle,
            "prompts": prompts,
            "outputs": outputs,
            "prefix_context_segments_list": segments,
            "output_token_bytes": token_bytes,
            "intervention": Intervention.ours(feature_conditions={1: 1}),
            "device": t.device("cuda"),
            "device_batch_size": 2,
        }
        full = teacher_forced_bpb_loss_full_sequence_reference(
            **common  # ty: ignore[invalid-argument-type]
        )
        shared = teacher_forced_bpb_loss_shared_prefix_kv_batch(
            source_prompts=[source, source],
            editable_spans=[EditableSpan(40 - k, k), EditableSpan(40 - k, k)],
            **common,  # ty: ignore[invalid-argument-type]
        )
        assert all(math.isfinite(value) for value in [*full, *shared])
        assert max(
            abs(full_value - shared_value)
            for full_value, shared_value in zip(full, shared, strict=True)
        ) <= _BATCHED_VS_GRAD_TOL


@pytest.mark.slow
def test_real_gcg_batched_optimizer_deterministic_and_bounded() -> None:
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    device = t.device("cuda")
    token_bytes = _build_token_bytes(bundle.tokenizer)

    # A short real prompt; Faster-GCG edits the final two positions.
    prompt_tokens = bundle.tokenizer.encode("the quick brown fox jumps over")
    prompt_tokens = [int(x) for x in prompt_tokens][:8]
    attack_rows = single_pair_attack_rows(
        prompt_tokens,
        editable_start=len(prompt_tokens) - 2,
        editable_count=2,
    )
    row0 = attack_rows.to_pylist()[0]

    cfg = _cfg()
    embedding_weight = bpb_surrogate_embedding_weight(bundle.model)
    allowed = [
        int(tid)
        for tid in build_allowed_token_ids(bundle.tokenizer)
        if 0 <= int(tid) < int(embedding_weight.shape[0])
    ]

    def _run_optimizer() -> dict[str, Any]:
        # Fresh callbacks each run (empty eval cache) so a pass is a real recompute,
        # not a cache replay -- determinism must come from the model, not the cache.
        obj: BpbObjectiveCallbacks = make_objective(bundle, token_bytes, device)
        rows = build_faster_gcg_candidates(
            attack_rows=attack_rows,
            tokenizer=bundle.tokenizer,
            cfg=cfg,
            embedding_weight=obj.embedding_weight(),
            allowed_token_ids=allowed,
            objective=obj.objective,
            gradient_provider=lambda row, tokens, step: obj.gradient_provider_batch(
                [(row, tokens)], step
            )[0],
        ).to_pylist()
        assert len(rows) == 1
        return rows[0]

    # (1) Run-to-run DETERMINISM: same cfg+seed -> byte-identical full row. This is
    # the exact guard the byte-identity-vs-grad assertion used to provide; it still
    # holds because the bf16/no_grad forward is deterministic for fixed shapes+seed.
    first = _run_optimizer()
    second = _run_optimizer()
    assert first == second
    # Non-degenerate: the search produced a finite objective on a real forward + VJP.
    assert first["objective_value"] is not None
    assert math.isfinite(float(first["objective_value"]))

    # (2) The batched loss-only objective stays within a bounded tolerance of the
    # grad-path reference at a FIXED token sequence (the starting prompt) -- no
    # optimizer-trajectory sensitivity, so a ranking flip cannot mask a real drift.
    batched = make_objective(bundle, token_bytes, device).objective(row0, prompt_tokens)
    grad_ref = _PreOptObjective(bundle, token_bytes, device).objective(row0, prompt_tokens)
    full_ref = _full_sequence_reference_objective(
        bundle=bundle,
        token_bytes=token_bytes,
        row=row0,
        prompt_tokens=prompt_tokens,
        device=device,
    )
    assert math.isfinite(batched) and math.isfinite(grad_ref) and math.isfinite(full_ref)
    assert abs(batched - full_ref) <= _BATCHED_VS_GRAD_TOL
    assert abs(batched - grad_ref) <= _BATCHED_VS_GRAD_TOL


@pytest.mark.slow
def test_opt_byte_identical_under_real_diffmean_residual_hook() -> None:
    """Closes the residual-hook coverage gap: OPT#1 (cached vs rebuilt embedding)
    and OPT#2 (loss-only vs gradient-path loss) must be byte-identical even when
    a REAL diffmean steering intervention installs a residual-stream forward hook
    (not just the no-op `none()` / `ours()` interventions the other tests use).

    Driven at the surrogate level on the real d12_base model so the residual hook
    runs through both forwards. A non-degeneracy check confirms the hook is
    actually active (it shifts the loss vs `none()`), so this is not a vacuous
    no-op-intervention test.
    """
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    device = t.device("cuda")
    token_bytes = _build_token_bytes(bundle.tokenizer)
    prompt_tokens = [int(x) for x in bundle.tokenizer.encode("the quick brown fox jumps over")][:8]
    rich_output = [264, 412, 99]

    n_embd = int(bundle.model.config.n_embd)
    direction = t.randn(n_embd, generator=t.Generator().manual_seed(1234)).to(device)
    intervention = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=6,
        coefficient=0.5,
    )
    cached_emb = bpb_surrogate_embedding_weight(bundle.model)
    # Reference: rebuild the embedding every call (no kwarg) under the residual hook.
    grad_rebuild = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=rich_output,
        prefix_context_segments=(),
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
    )
    # OPT#1: pass the cached static embedding -> must be byte-identical.
    grad_cached = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=rich_output,
        embedding_weight=cached_emb,
        prefix_context_segments=(),
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
    )
    assert grad_cached.loss == grad_rebuild.loss
    assert t.equal(grad_cached.token_gradient, grad_rebuild.token_gradient)
    assert grad_cached.embedding_weight is cached_emb

    # OPT#2: loss-only forward -> loss byte-identical to the gradient path, under the hook.
    loss_only = teacher_forced_bpb_loss(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=rich_output,
        prefix_context_segments=(),
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
    )
    assert loss_only == grad_rebuild.loss

    # Non-degeneracy: the residual hook genuinely changed the forward (vs none()),
    # so the byte-identity above is exercising the hooked path, not a no-op.
    grad_none = teacher_forced_bpb_gradient(
        target_bundle=bundle,
        prompt_tokens=list(prompt_tokens),
        output_tokens=rich_output,
        prefix_context_segments=(),
        output_token_bytes=token_bytes,
        intervention=Intervention.none(),
        device=device,
    )
    assert grad_rebuild.loss != grad_none.loss


_D12_ENCDEC_ROOT = (
    Path.home() / ".cache" / "nanochat" / "conditional_encdec_checkpoints"
)
_D12_ENCDEC_TAG = "d12_baseline"
_D12_ENCDEC_STEP = 54_600


def _production_tokens(tokenizer: Any, text: str, count: int) -> list[int]:
    encoded = [int(token_id) for token_id in tokenizer.encode((text + " ") * 512)]
    if len(encoded) < count:
        raise AssertionError(f"fixture text encoded to only {len(encoded)} tokens")
    return encoded[:count]


def _release_cuda_bundle(bundle: TargetModelBundle) -> None:
    bundle.model.to("cpu")
    gc.collect()
    t.cuda.empty_cache()


@pytest.mark.slow
def test_real_d12_vjp_matches_legacy_backward_exactly_and_preserves_state() -> None:
    """Exact old/new CUDA gate at production lengths, including mixed batching."""

    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    device = t.device("cuda")
    old_bundle = load_d12_base()
    prompt = _production_tokens(old_bundle.tokenizer, "the quick brown fox", 432)
    output = _production_tokens(old_bundle.tokenizer, "jumps over the lazy dog", 25)
    samples = [
        (prompt, output),
        (prompt[:389], output[:19]),
        (prompt[:417], output[:23]),
    ]
    token_bytes = _build_token_bytes(old_bundle.tokenizer)
    n_embd = int(old_bundle.model.config.n_embd)
    direction = t.randn(
        n_embd,
        generator=t.Generator().manual_seed(1226),
    ).to(device)
    intervention = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=6,
        coefficient=0.5,
    )
    old_cache = bpb_surrogate_embedding_weight(old_bundle.model)
    old_bundle.model.train(True)
    t.cuda.synchronize()
    old_single = legacy_teacher_forced_bpb_gradient(
        target_bundle=old_bundle,
        prompt_tokens=prompt,
        output_tokens=output,
        prefix_context_segments=(),
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=old_cache,
    )
    old_batch = legacy_teacher_forced_bpb_gradient_batch(
        target_bundle=old_bundle,
        prompts=[sample[0] for sample in samples],
        outputs=[sample[1] for sample in samples],
        prefix_context_segments_list=[() for _ in samples],
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=old_cache,
        device_batch_size=2,
    )
    t.cuda.synchronize()
    assert old_bundle.model.training is True
    assert all(parameter.grad is None for parameter in old_bundle.model.parameters())
    _release_cuda_bundle(old_bundle)
    del old_bundle

    new_bundle = load_d12_base()
    new_cache = bpb_surrogate_embedding_weight(new_bundle.model)
    new_bundle.model.train(True)
    original_flags = tuple(
        parameter.requires_grad for parameter in new_bundle.model.parameters()
    )
    saw_frozen_forward = False

    def _assert_frozen(module: t.nn.Module, _inputs: tuple[Any, ...]) -> None:
        nonlocal saw_frozen_forward
        saw_frozen_forward = True
        assert all(not parameter.requires_grad for parameter in module.parameters())

    hook = new_bundle.model.register_forward_pre_hook(_assert_frozen)
    try:
        t.cuda.synchronize()
        new_single = teacher_forced_bpb_gradient(
            target_bundle=new_bundle,
            prompt_tokens=prompt,
            output_tokens=output,
            prefix_context_segments=(),
            output_token_bytes=token_bytes,
            intervention=intervention,
            device=device,
            embedding_weight=new_cache,
        )
        new_batch = teacher_forced_bpb_gradient_batch(
            target_bundle=new_bundle,
            prompts=[sample[0] for sample in samples],
            outputs=[sample[1] for sample in samples],
            prefix_context_segments_list=[() for _ in samples],
            output_token_bytes=token_bytes,
            intervention=intervention,
            device=device,
            embedding_weight=new_cache,
            device_batch_size=PositiveInt.parse(2),
        )
        t.cuda.synchronize()
    finally:
        hook.remove()

    assert saw_frozen_forward
    assert new_bundle.model.training is True
    assert tuple(
        parameter.requires_grad for parameter in new_bundle.model.parameters()
    ) == original_flags
    assert all(parameter.grad is None for parameter in new_bundle.model.parameters())
    assert new_single.loss == old_single.loss
    assert t.equal(new_single.token_gradient, old_single.token_gradient)
    assert new_single.token_gradient.abs().sum() > 0
    assert new_single.embedding_weight is new_cache
    assert old_single.embedding_weight is old_cache
    assert t.equal(new_cache, old_cache)
    for new_result, old_result in zip(new_batch, old_batch, strict=True):
        assert new_result.loss == old_result.loss
        assert t.equal(new_result.token_gradient, old_result.token_gradient)
        assert new_result.token_gradient.abs().sum() > 0
        assert new_result.embedding_weight is new_cache


def _load_d12_encdec() -> TargetModelBundle:
    os.environ.setdefault("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
    model, tokenizer, _meta = load_model(
        source="encdec",
        device="cuda",
        phase="eval",
        model_tag=_D12_ENCDEC_TAG,
        step=_D12_ENCDEC_STEP,
        eval_conditioning=EvalConditioning.conditions_within_vocab(
            "issue_1226_vjp_gate"
        ),
    )
    model.eval()
    model_any: Any = model
    return TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=int(model_any.dec_config.sequence_len),
        target_name="encdec",
    )


@pytest.mark.slow
def test_real_d12_encdec_vjp_matches_legacy_backward_exactly() -> None:
    """Exact old/new CUDA gate for the real encoder-decoder ``ours`` path."""

    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    checkpoint = (
        _D12_ENCDEC_ROOT
        / _D12_ENCDEC_TAG
        / "s128"
        / "dist_geometric"
        / "mean_10.0"
        / "std_10.0"
        / f"model_{_D12_ENCDEC_STEP:06d}.pt"
    )
    if not checkpoint.exists():
        pytest.skip("requires d12_baseline encoder-decoder checkpoint")

    device = t.device("cuda")
    old_bundle = _load_d12_encdec()
    prompt = _production_tokens(old_bundle.tokenizer, "conditioned model context", 192)
    output = _production_tokens(old_bundle.tokenizer, "desired continuation", 25)
    decoded_prompt = old_bundle.tokenizer.decode(prompt)
    allowed_ids = feature_vocab_allowed_ids(old_bundle.model)
    feature_id = 7 if allowed_ids is None else int(allowed_ids[0].item())
    segments = (
        ContextFeatureSegment(
            feature_ids=(feature_id,),
            buckets=(1,),
            char_start=0,
            char_end=len(decoded_prompt),
        ),
    )
    intervention = Intervention.ours(feature_conditions={feature_id: 1})
    token_bytes = _build_token_bytes(old_bundle.tokenizer)
    old_cache = bpb_surrogate_embedding_weight(old_bundle.model)
    t.cuda.synchronize()
    old = legacy_teacher_forced_bpb_gradient(
        target_bundle=old_bundle,
        prompt_tokens=prompt,
        output_tokens=output,
        prefix_context_segments=segments,
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=old_cache,
    )
    t.cuda.synchronize()
    _release_cuda_bundle(old_bundle)
    del old_bundle

    new_bundle = _load_d12_encdec()
    new_cache = bpb_surrogate_embedding_weight(new_bundle.model)
    original_training = new_bundle.model.training
    original_flags = tuple(
        parameter.requires_grad for parameter in new_bundle.model.parameters()
    )
    saw_frozen_forward = False

    def _assert_frozen(module: t.nn.Module, _inputs: tuple[Any, ...]) -> None:
        nonlocal saw_frozen_forward
        saw_frozen_forward = True
        assert all(not parameter.requires_grad for parameter in module.parameters())

    hook = new_bundle.model.register_forward_pre_hook(_assert_frozen)
    try:
        t.cuda.synchronize()
        new = teacher_forced_bpb_gradient(
            target_bundle=new_bundle,
            prompt_tokens=prompt,
            output_tokens=output,
            prefix_context_segments=segments,
            output_token_bytes=token_bytes,
            intervention=intervention,
            device=device,
            embedding_weight=new_cache,
        )
        t.cuda.synchronize()
    finally:
        hook.remove()

    assert saw_frozen_forward
    assert new_bundle.model.training is original_training
    assert tuple(
        parameter.requires_grad for parameter in new_bundle.model.parameters()
    ) == original_flags
    assert all(parameter.grad is None for parameter in new_bundle.model.parameters())
    assert new.loss == old.loss
    assert t.equal(new.token_gradient, old.token_gradient)
    assert new.token_gradient.abs().sum() > 0
    assert new.embedding_weight is new_cache
    assert old.embedding_weight is old_cache
    assert t.equal(new_cache, old_cache)


# Lower bound on the batched-vs-single prompt-token GRADIENT cosine on d12_base.
# The batched forward + VJP changes the matmul/attention batch shape, so bf16 rounding
# perturbs both the forward loss and the backprop -- the SAME accepted cost as the
# batched loss/ranking path. The cosine bound is the STRUCTURAL guard: a sign flip,
# a wrong per-sample slice, or a leaked cross-sample gradient collapses it to far
# below 0.9, whereas pure bf16 batch-shape noise leaves it ~0.999 (observed ~0.9989
# on the most-padded short sample at device_batch_size=2). 0.99 tolerates the bf16
# noise while still failing loudly on any structural error. Exact (byte-identical)
# correctness of the ALGORITHM is pinned separately by test_surrogate_gradient_batch.
_BATCHED_GRAD_COS_MIN = 0.99
_BATCHED_GRAD_LOSS_TOL = 0.2


@pytest.mark.slow
def test_real_batched_gradient_matches_single_within_tolerance() -> None:
    """The batched per-sample gradient must track the trusted single-prompt path on
    the REAL d12_base model: each sample's loss within tolerance and its
    token_gradient cosine-aligned (a wrong per-sample VJP extraction, a leaked
    cross-sample gradient, or a sign error would collapse the cosine). Driven under a
    REAL diffmean residual hook so the hooked backward is exercised, with samples of
    MIXED length so the per-sample padding offset matters. The exact (byte-identical)
    counterpart on a non-bf16 model is test_surrogate_gradient_batch.py."""
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    device = t.device("cuda")
    token_bytes = _build_token_bytes(bundle.tokenizer)
    cached_emb = bpb_surrogate_embedding_weight(bundle.model)

    base = [int(x) for x in bundle.tokenizer.encode("the quick brown fox jumps over")]
    # Mixed-length prompts + outputs (rich/avoid-like), all real token ids.
    samples: list[tuple[list[int], list[int]]] = [
        (base[:8], [264, 412, 99]),
        (base[:5], [171, 333]),
        (base[:7], [12, 88, 901, 5]),
        (base[:4], [42]),
    ]
    n_embd = int(bundle.model.config.n_embd)
    direction = t.randn(n_embd, generator=t.Generator().manual_seed(7)).to(device)
    intervention = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=6,
        coefficient=0.5,
    )

    batched = teacher_forced_bpb_gradient_batch(
        target_bundle=bundle,
        prompts=[p for p, _ in samples],
        outputs=[o for _, o in samples],
        prefix_context_segments_list=[() for _ in samples],
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=cached_emb,
        # Force chunking so the multi-microbatch path is real.
        device_batch_size=PositiveInt.parse(2),
    )
    singles = [
        teacher_forced_bpb_gradient(
            target_bundle=bundle,
            prompt_tokens=list(prompt),
            output_tokens=list(output),
            prefix_context_segments=(),
            output_token_bytes=token_bytes,
            intervention=intervention,
            device=device,
            embedding_weight=cached_emb,
        )
        for prompt, output in samples
    ]

    assert len(batched) == len(singles)
    for got, want in zip(batched, singles, strict=True):
        assert abs(got.loss - want.loss) <= _BATCHED_GRAD_LOSS_TOL
        cos = t.nn.functional.cosine_similarity(
            got.token_gradient.flatten(), want.token_gradient.flatten(), dim=0
        )
        assert float(cos) >= _BATCHED_GRAD_COS_MIN
        # The shared static embedding is passed verbatim (OPT#1), not rebuilt.
        assert got.embedding_weight is cached_emb

    # Non-degeneracy: the gradients are real (non-zero), so the cosine bound is not
    # vacuously satisfied by two zero vectors.
    assert all(s.token_gradient.abs().sum() > 0 for s in singles)

    # B=1 has the SAME (1, seq) shape as the single path (no padding), so it is far
    # tighter than the padded multi-sample batch.
    [one] = teacher_forced_bpb_gradient_batch(
        target_bundle=bundle,
        prompts=[samples[0][0]],
        outputs=[samples[0][1]],
        prefix_context_segments_list=[()],
        output_token_bytes=token_bytes,
        intervention=intervention,
        device=device,
        embedding_weight=cached_emb,
    )
    assert abs(one.loss - singles[0].loss) <= 1.0e-2
    one_cos = t.nn.functional.cosine_similarity(
        one.token_gradient.flatten(), singles[0].token_gradient.flatten(), dim=0
    )
    assert float(one_cos) >= 0.9999


@pytest.mark.slow
def test_real_d12_allowed_embedding_cache_matches_raw_ranking_exactly() -> None:
    """Issue #712: production-shape cache hits equal fresh gathers bit-for-bit."""

    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    weight = bpb_surrogate_embedding_weight(bundle.model)
    allowed = tuple(
        int(token_id)
        for token_id in build_allowed_token_ids(bundle.tokenizer)
        if 0 <= int(token_id) < int(weight.shape[0])
    )
    assert weight.shape == (32_768, 5_376)
    assert weight.dtype == t.float32
    assert weight.device.type == "cpu"
    assert not weight.requires_grad
    assert len(allowed) == 32_734

    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=allowed,
    )
    generator = t.Generator().manual_seed(712)
    for k in (1, 4, 32):
        gradient = t.randn(k + 1, weight.shape[1], generator=generator)
        current_tokens = (0, *allowed[:k])
        raw = rank_replacements_from_gradients(
            embedding_gradient=gradient,
            embedding_weight=weight,
            editable_positions=range(1, k + 1),
            allowed_token_ids=allowed,
            current_tokens=current_tokens,
            top_k=32,
            distance_weight=10.0,
        )
        cached = rank_replacements_from_gradients(
            embedding_gradient=gradient,
            embedding_weight=weight,
            editable_positions=range(1, k + 1),
            allowed_token_ids=cache.allowed_token_ids,
            current_tokens=current_tokens,
            top_k=32,
            distance_weight=10.0,
            allowed_embedding_cache=cache,
        )
        cached_again = rank_replacements_from_gradients(
            embedding_gradient=gradient,
            embedding_weight=weight,
            editable_positions=range(1, k + 1),
            allowed_token_ids=cache.allowed_token_ids,
            current_tokens=current_tokens,
            top_k=32,
            distance_weight=10.0,
            allowed_embedding_cache=cache,
        )
        assert raw == cached == cached_again
