"""Shared CPU-vs-CUDA token-example/trial surface comparison logic (#679/#678).

Extracted verbatim from tests/steering/autointerp_filter/
test_token_example_cpu_cuda_tolerance.py so the parity gates and the
benchmark harness share one implementation. Pure comparison/reporting:
importing this module must never mutate provider or cache state.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch as t

from steering.eval.autointerp_filter.config import (
    AutoInterpFilterConfig,
    TokenExamplesConfig,
)
from steering.eval.autointerp_filter.intruder import (
    build_intruder_messages,
    render_trial_examples,
)
from steering.eval.autointerp_filter.prompts import (
    DEFAULT_INTRUDER_FEW_SHOT_SET,
    DEFAULT_INTRUDER_PROMPT_VARIANT,
)
from steering.eval.autointerp_filter.renderers import TokenWindowRenderer
from steering.eval.autointerp_filter.token_examples import (
    TokenExampleSampler,
    positive_highlight_indices,
)
from steering.eval.autointerp_filter.types import (
    AutoInterpCacheError,
    IntruderTrial,
    ScoringTarget,
    TargetKind,
    TokenExampleSet,
    TokenWindowRef,
)
from steering.refined_types import PositiveInt, UnitInterval

FP32_NATIVE_VS_TL_ATOL = 0.15
HIGHLIGHT_THRESHOLD_RATIO = 0.3


@dataclass(frozen=True)
class TensorToleranceReport:
    reference_values: int
    candidate_values: int
    support_mismatches: int
    max_abs: float
    p99_abs: float
    mean_abs: float
    within_abs_tolerance: bool

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TokenGroupSurfaceReport:
    reference_count: int
    candidate_count: int
    identity_set_equal: bool
    matched_window_count: int
    same_identity_order_count: int
    same_stable_key_order_count: int
    activation: TensorToleranceReport
    max_activation: TensorToleranceReport
    quantile_exact_count: int
    highlight_exact_count: int
    min_highlight_jaccard: float

    @property
    def same_identity_order_fraction(self) -> float:
        if self.reference_count == 0:
            return 1.0
        return self.same_identity_order_count / self.reference_count

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["same_identity_order_fraction"] = self.same_identity_order_fraction
        return payload


@dataclass(frozen=True)
class TokenExampleSurfaceReport:
    positives: TokenGroupSurfaceReport
    non_activating: TokenGroupSurfaceReport

    def to_json(self) -> dict:
        return {
            "positives": self.positives.to_json(),
            "non_activating": self.non_activating.to_json(),
        }


@dataclass(frozen=True)
class TrialSurfaceReport:
    reference_count: int
    candidate_count: int
    same_prompt_surface_count: int
    same_trial_key_count: int
    same_intruder_window_count: int
    min_positive_window_jaccard: float
    mean_positive_window_jaccard: float

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RenderedTrialSurfaceReport:
    """Exact production renderer/message comparison for a trial sequence."""

    reference_count: int
    candidate_count: int
    exact_rendered_examples_count: int
    exact_message_bytes_count: int
    reference_messages_sha256: str
    candidate_messages_sha256: str

    @property
    def all_rendered_examples_exact(self) -> bool:
        return (
            self.reference_count == self.candidate_count
            and self.exact_rendered_examples_count == self.reference_count
        )

    @property
    def all_message_bytes_exact(self) -> bool:
        return (
            self.reference_count == self.candidate_count
            and self.exact_message_bytes_count == self.reference_count
            and self.reference_messages_sha256 == self.candidate_messages_sha256
        )

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["all_rendered_examples_exact"] = self.all_rendered_examples_exact
        payload["all_message_bytes_exact"] = self.all_message_bytes_exact
        return payload


class StaticTokenExampleProvider:
    def __init__(self, examples: TokenExampleSet) -> None:
        self._examples = examples

    def prepare(self, targets: list[ScoringTarget]) -> None:
        _ = targets

    def cache_context(self) -> dict[str, Any]:
        return {}

    def examples_for_target(self, target: ScoringTarget) -> TokenExampleSet:
        _ = target
        return self._examples


def source_window_payload(ref: TokenWindowRef) -> dict[str, Any]:
    payload = ref.to_json()
    payload.pop("activations")
    payload.pop("max_activation")
    payload.pop("quantile")
    return payload


def source_window_key(ref: TokenWindowRef) -> str:
    return json.dumps(source_window_payload(ref), sort_keys=True, separators=(",", ":"))


def refs_by_identity(refs: tuple[TokenWindowRef, ...]) -> dict[str, TokenWindowRef]:
    by_identity: dict[str, TokenWindowRef] = {}
    for ref in refs:
        key = source_window_key(ref)
        if key in by_identity:
            raise AutoInterpCacheError(f"duplicate token-window identity in compared set: {key}")
        by_identity[key] = ref
    return by_identity


def tensor_tolerance_report(
    reference_values: list[float],
    candidate_values: list[float],
    *,
    atol: float,
) -> TensorToleranceReport:
    if len(reference_values) != len(candidate_values):
        return TensorToleranceReport(
            reference_values=len(reference_values),
            candidate_values=len(candidate_values),
            support_mismatches=max(len(reference_values), len(candidate_values)),
            max_abs=float("inf"),
            p99_abs=float("inf"),
            mean_abs=float("inf"),
            within_abs_tolerance=False,
        )
    if not reference_values:
        return TensorToleranceReport(
            reference_values=0,
            candidate_values=0,
            support_mismatches=0,
            max_abs=0.0,
            p99_abs=0.0,
            mean_abs=0.0,
            within_abs_tolerance=True,
        )
    reference = t.tensor(reference_values, dtype=t.float32)
    candidate = t.tensor(candidate_values, dtype=t.float32)
    diff = (candidate - reference).abs()
    max_abs = float(diff.max().item())
    return TensorToleranceReport(
        reference_values=len(reference_values),
        candidate_values=len(candidate_values),
        support_mismatches=int(((reference > 0.0) != (candidate > 0.0)).sum().item()),
        max_abs=max_abs,
        p99_abs=float(t.quantile(diff, 0.99).item()),
        mean_abs=float(diff.mean().item()),
        within_abs_tolerance=max_abs <= atol,
    )


def jaccard(left: tuple[Any, ...], right: tuple[Any, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def compare_token_group_surface(
    reference_refs: tuple[TokenWindowRef, ...],
    candidate_refs: tuple[TokenWindowRef, ...],
    *,
    atol: float,
    highlight_threshold_ratio: float = HIGHLIGHT_THRESHOLD_RATIO,
) -> TokenGroupSurfaceReport:
    reference_by_id = refs_by_identity(reference_refs)
    candidate_by_id = refs_by_identity(candidate_refs)
    reference_keys = [source_window_key(ref) for ref in reference_refs]
    candidate_keys = [source_window_key(ref) for ref in candidate_refs]
    common_keys = sorted(set(reference_by_id) & set(candidate_by_id))

    reference_activations: list[float] = []
    candidate_activations: list[float] = []
    reference_maxes: list[float] = []
    candidate_maxes: list[float] = []
    highlight_exact_count = 0
    quantile_exact_count = 0
    highlight_jaccards: list[float] = []
    for key in common_keys:
        reference = reference_by_id[key]
        candidate = candidate_by_id[key]
        reference_activations.extend(reference.activations)
        candidate_activations.extend(candidate.activations)
        reference_maxes.append(reference.max_activation)
        candidate_maxes.append(candidate.max_activation)
        if reference.quantile == candidate.quantile:
            quantile_exact_count += 1
        reference_highlights = positive_highlight_indices(
            reference,
            activation_threshold_ratio=highlight_threshold_ratio,
        )
        candidate_highlights = positive_highlight_indices(
            candidate,
            activation_threshold_ratio=highlight_threshold_ratio,
        )
        if reference_highlights == candidate_highlights:
            highlight_exact_count += 1
        highlight_jaccards.append(jaccard(reference_highlights, candidate_highlights))

    return TokenGroupSurfaceReport(
        reference_count=len(reference_refs),
        candidate_count=len(candidate_refs),
        identity_set_equal=set(reference_by_id) == set(candidate_by_id),
        matched_window_count=len(common_keys),
        same_identity_order_count=sum(
            reference_key == candidate_key
            for reference_key, candidate_key in zip(reference_keys, candidate_keys, strict=False)
        ),
        same_stable_key_order_count=sum(
            reference.stable_key() == candidate.stable_key()
            for reference, candidate in zip(reference_refs, candidate_refs, strict=False)
        ),
        activation=tensor_tolerance_report(
            reference_activations,
            candidate_activations,
            atol=atol,
        ),
        max_activation=tensor_tolerance_report(
            reference_maxes,
            candidate_maxes,
            atol=atol,
        ),
        quantile_exact_count=quantile_exact_count,
        highlight_exact_count=highlight_exact_count,
        min_highlight_jaccard=min(highlight_jaccards, default=1.0),
    )


def compare_token_example_surface(
    reference: TokenExampleSet,
    candidate: TokenExampleSet,
    *,
    atol: float = FP32_NATIVE_VS_TL_ATOL,
    highlight_threshold_ratio: float = HIGHLIGHT_THRESHOLD_RATIO,
) -> TokenExampleSurfaceReport:
    return TokenExampleSurfaceReport(
        positives=compare_token_group_surface(
            reference.positives,
            candidate.positives,
            atol=atol,
            highlight_threshold_ratio=highlight_threshold_ratio,
        ),
        non_activating=compare_token_group_surface(
            reference.non_activating,
            candidate.non_activating,
            atol=atol,
            highlight_threshold_ratio=highlight_threshold_ratio,
        ),
    )


def sqlite_ro_uri(path: Path) -> str:
    """Read-only sqlite URI with proper escaping.

    Raw f-string URIs truncate at '?'/'#' and percent-decode '%XX' in path
    components, silently opening a DIFFERENT file; Path.as_uri() percent-encodes
    the path portion so sqlite's URI parser round-trips it exactly.
    """
    return f"{path.resolve().as_uri()}?mode=ro"


def load_token_example_set_by_target(path: Path, target_key: str) -> TokenExampleSet:
    with closing(sqlite3.connect(sqlite_ro_uri(path), uri=True)) as conn:
        rows = conn.execute(
            "SELECT namespace, payload FROM token_example_sets WHERE target_key = ?",
            (target_key,),
        ).fetchall()
    if not rows:
        raise AutoInterpCacheError(f"{path} has no token_example_sets row for {target_key!r}")
    namespaces = sorted({row[0] for row in rows})
    if len(namespaces) > 1:
        raise AutoInterpCacheError(
            f"{path} has {len(namespaces)} namespaces for {target_key!r}: {namespaces}"
        )
    if len(rows) > 1:
        raise AutoInterpCacheError(
            f"{path} has {len(rows)} rows for {target_key!r} in one namespace; "
            "a well-formed cache has a (namespace, target_key) primary key"
        )
    return TokenExampleSet.from_json(json.loads(rows[0][1]))


def _trial_target() -> ScoringTarget:
    return ScoringTarget(
        kind=TargetKind.feature,
        feature_ids=(0,),
        target_key="feature:0",
        display_name="feature:0",
    )


def build_trial_surfaces(
    examples: TokenExampleSet,
    *,
    n_positive_examples: int = 2,
    n_trials: int = 8,
    n_quantiles: int = 5,
    target: ScoringTarget | None = None,
    seed: int = 42,
    example_ctx_len: int = 32,
    activation_threshold_ratio: float = 0.3,
) -> list[tuple[str, tuple[str, ...], str, int, int | None, tuple[int, ...]]]:
    trials = _build_trials(
        examples,
        n_positive_examples=n_positive_examples,
        n_trials=n_trials,
        n_quantiles=n_quantiles,
        target=target,
        seed=seed,
        example_ctx_len=example_ctx_len,
        activation_threshold_ratio=activation_threshold_ratio,
    )
    return [
        (
            trial.trial_key,
            tuple(source_window_key(ref) for ref in trial.positive_refs),
            source_window_key(trial.intruder_ref),
            trial.intruder_index,
            trial.quantile,
            trial.intruder_highlight_indices,
        )
        for trial in trials
    ]


def _build_trials(
    examples: TokenExampleSet,
    *,
    n_positive_examples: int,
    n_trials: int,
    n_quantiles: int,
    target: ScoringTarget | None,
    seed: int,
    example_ctx_len: int,
    activation_threshold_ratio: float,
) -> list[IntruderTrial]:
    cfg = AutoInterpFilterConfig(
        feature_dir="unused",
        output_dir="unused",
        n_positive_examples=PositiveInt.parse(n_positive_examples),
        n_trials_per_target=PositiveInt.parse(n_trials),
        n_quantiles=PositiveInt.parse(n_quantiles),
        seed=seed,
        token_examples=TokenExamplesConfig(
            example_ctx_len=PositiveInt.parse(example_ctx_len),
            activation_threshold_ratio=UnitInterval.parse(activation_threshold_ratio),
        ),
    )
    sampler = TokenExampleSampler(cfg, provider=StaticTokenExampleProvider(examples))
    return sampler.build_trials(target if target is not None else _trial_target())


def _rendered_trial_payloads(
    examples: TokenExampleSet,
    *,
    n_positive_examples: int,
    n_trials: int,
    n_quantiles: int,
    target: ScoringTarget | None,
    seed: int,
    example_ctx_len: int,
    activation_threshold_ratio: float,
    cot: bool,
    prompt_variant: str,
    few_shot_set: str,
) -> list[tuple[tuple[str, ...], bytes]]:
    renderer = TokenWindowRenderer()
    payloads: list[tuple[tuple[str, ...], bytes]] = []
    for trial in _build_trials(
        examples,
        n_positive_examples=n_positive_examples,
        n_trials=n_trials,
        n_quantiles=n_quantiles,
        target=target,
        seed=seed,
        example_ctx_len=example_ctx_len,
        activation_threshold_ratio=activation_threshold_ratio,
    ):
        rendered = render_trial_examples(trial, renderer)
        messages = build_intruder_messages(
            rendered,
            cot=cot,
            prompt_variant=prompt_variant,
            few_shot_set=few_shot_set,
        )
        message_bytes = json.dumps(
            messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        payloads.append((tuple(example.text for example in rendered), message_bytes))
    return payloads


def compare_rendered_trial_surfaces(
    reference: TokenExampleSet,
    candidate: TokenExampleSet,
    *,
    n_positive_examples: int = 2,
    n_trials: int = 8,
    n_quantiles: int = 5,
    target: ScoringTarget | None = None,
    seed: int = 42,
    example_ctx_len: int = 32,
    activation_threshold_ratio: float = 0.3,
    cot: bool = False,
    prompt_variant: str = DEFAULT_INTRUDER_PROMPT_VARIANT,
    few_shot_set: str = DEFAULT_INTRUDER_FEW_SHOT_SET,
) -> RenderedTrialSurfaceReport:
    """Compare exact renderer output and serialized production chat messages."""
    common = {
        "n_positive_examples": n_positive_examples,
        "n_trials": n_trials,
        "n_quantiles": n_quantiles,
        "target": target,
        "seed": seed,
        "example_ctx_len": example_ctx_len,
        "activation_threshold_ratio": activation_threshold_ratio,
        "cot": cot,
        "prompt_variant": prompt_variant,
        "few_shot_set": few_shot_set,
    }
    reference_payloads = _rendered_trial_payloads(reference, **common)
    candidate_payloads = _rendered_trial_payloads(candidate, **common)
    exact_rendered = sum(
        reference_row[0] == candidate_row[0]
        for reference_row, candidate_row in zip(
            reference_payloads,
            candidate_payloads,
            strict=False,
        )
    )
    exact_messages = sum(
        reference_row[1] == candidate_row[1]
        for reference_row, candidate_row in zip(
            reference_payloads,
            candidate_payloads,
            strict=False,
        )
    )

    def aggregate_sha256(payloads: list[tuple[tuple[str, ...], bytes]]) -> str:
        digest = hashlib.sha256()
        for _rendered, message_bytes in payloads:
            digest.update(len(message_bytes).to_bytes(8, "big"))
            digest.update(message_bytes)
        return digest.hexdigest()

    return RenderedTrialSurfaceReport(
        reference_count=len(reference_payloads),
        candidate_count=len(candidate_payloads),
        exact_rendered_examples_count=exact_rendered,
        exact_message_bytes_count=exact_messages,
        reference_messages_sha256=aggregate_sha256(reference_payloads),
        candidate_messages_sha256=aggregate_sha256(candidate_payloads),
    )


def compare_trial_surfaces(
    reference: TokenExampleSet,
    candidate: TokenExampleSet,
    *,
    n_positive_examples: int = 2,
    n_trials: int = 8,
    n_quantiles: int = 5,
    target: ScoringTarget | None = None,
    seed: int = 42,
    example_ctx_len: int = 32,
    activation_threshold_ratio: float = 0.3,
) -> TrialSurfaceReport:
    reference_trials = build_trial_surfaces(
        reference,
        n_positive_examples=n_positive_examples,
        n_trials=n_trials,
        n_quantiles=n_quantiles,
        target=target,
        seed=seed,
        example_ctx_len=example_ctx_len,
        activation_threshold_ratio=activation_threshold_ratio,
    )
    candidate_trials = build_trial_surfaces(
        candidate,
        n_positive_examples=n_positive_examples,
        n_trials=n_trials,
        n_quantiles=n_quantiles,
        target=target,
        seed=seed,
        example_ctx_len=example_ctx_len,
        activation_threshold_ratio=activation_threshold_ratio,
    )
    prompt_surface_matches = 0
    trial_key_matches = 0
    intruder_window_matches = 0
    positive_jaccards: list[float] = []
    for reference_trial, candidate_trial in zip(
        reference_trials,
        candidate_trials,
        strict=False,
    ):
        reference_key, reference_positives, reference_intruder, *reference_rest = reference_trial
        candidate_key, candidate_positives, candidate_intruder, *candidate_rest = candidate_trial
        if reference_key == candidate_key:
            trial_key_matches += 1
        if reference_intruder == candidate_intruder:
            intruder_window_matches += 1
        if (
            reference_positives == candidate_positives
            and reference_intruder == candidate_intruder
            and reference_rest == candidate_rest
        ):
            prompt_surface_matches += 1
        positive_jaccards.append(
            jaccard(reference_positives, candidate_positives),
        )

    if positive_jaccards:
        mean_positive_jaccard = sum(positive_jaccards) / len(positive_jaccards)
        min_positive_jaccard = min(positive_jaccards)
    else:
        mean_positive_jaccard = 1.0
        min_positive_jaccard = 1.0
    return TrialSurfaceReport(
        reference_count=len(reference_trials),
        candidate_count=len(candidate_trials),
        same_prompt_surface_count=prompt_surface_matches,
        same_trial_key_count=trial_key_matches,
        same_intruder_window_count=intruder_window_matches,
        min_positive_window_jaccard=min_positive_jaccard,
        mean_positive_window_jaccard=mean_positive_jaccard,
    )
