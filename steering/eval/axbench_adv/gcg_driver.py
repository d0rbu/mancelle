"""Root-owned universal appended-suffix GCG driver for AxBench-Adv."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch as t

from nanochat.atomic_io import atomic_write_text
from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    rank_replacements_from_gradients,
    replacement_pool_exhausted,
    sample_candidate_sequences,
)
from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.edit_primitives import (
    TokenReplacement,
    token_sequence_digest,
)
from steering.conditional.feature_gcg.edit_space import GroupResult, ValidatedGroup
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix
from steering.eval.axbench_adv.worker_client import GcgFilteredCandidate, GcgPrepared
from steering.eval.axbench_upstream.data import ConceptTrainingPairs
from steering.eval.axbench_upstream.seeds import derive_selection_seed

_SEPARATOR = "\n\n"


class GcgRuntime(Protocol):
    def prepare_gcg(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str,
    ) -> GcgPrepared: ...

    def gcg_filter(
        self, candidate_token_ids: Sequence[Sequence[int]]
    ) -> tuple[GcgFilteredCandidate, ...]: ...

    def gcg_score(
        self,
        candidate_token_ids: Sequence[Sequence[int]],
        member_indices: Sequence[int],
        *,
        candidate_eval_device_batch_size: int,
    ) -> list[float]: ...

    def gcg_gradient(
        self, token_ids: Sequence[int], member_indices: Sequence[int]
    ) -> list[t.Tensor]: ...

    def evaluate(self, suffix: str, input_ids: Sequence[int]) -> Sequence[RawEvaluation]: ...


@dataclass(frozen=True)
class _Group:
    key_str: str
    members: tuple[int, ...]
    seed: tuple[int, ...]


@dataclass(frozen=True)
class _SearchPayload:
    token_ids: tuple[int, ...]
    suffix: str
    objective: float
    accepted_steps: int
    visited_count: int


class AppendSuffixEditSpace:
    """Small token edit-space adapter over the shared GCG optimizer."""

    def __init__(self, runtime: GcgRuntime, prepared: GcgPrepared):
        self.runtime = runtime
        self.prepared = prepared
        self.sentinel_id = self._sentinel_id(prepared)
        self.cache = AllowedEmbeddingCache(
            embedding_weight=prepared.embedding_weight,
            allowed_token_ids=prepared.allowed_token_ids,
        )

    @staticmethod
    def _sentinel_id(prepared: GcgPrepared) -> int:
        value = prepared.identity.get("rank_sentinel_id")
        if type(value) is not int or not 0 <= value < prepared.embedding_weight.shape[0]:
            raise AxBenchAdvError("GCG prepare identity lacks an in-vocab rank sentinel")
        return value

    def validate(self, group: _Group) -> ValidatedGroup:
        if not group.members or len(group.members) != self.prepared.member_count:
            raise AxBenchAdvError("GCG group does not cover the prepared population")
        if len(group.seed) == 0:
            raise AxBenchAdvError("positive-k GCG group requires a seed")
        return ValidatedGroup(group.key_str, list(group.members), group)

    def is_trivial(self, validated: ValidatedGroup) -> bool:
        del validated
        return False

    def finalize_trivial(self, validated: ValidatedGroup) -> GroupResult:
        del validated
        raise AxBenchAdvError("positive-k GCG groups are never trivial")

    def init_target(self, validated: ValidatedGroup) -> list[int]:
        return list(validated.payload.seed)

    def _padded(self, target: Sequence[int]) -> list[int]:
        return [self.sentinel_id, *(int(token_id) for token_id in target)]

    def digest(self, target: list[int]) -> str:
        return token_sequence_digest(self._padded(target))

    def aggregate_gradient(self, per_member: Iterable[tuple[t.Tensor, int]]) -> t.Tensor:
        gradients = [gradient for gradient, _member in per_member]
        if not gradients:
            raise AxBenchAdvError("GCG gradient minibatch must not be empty")
        shape = gradients[0].shape
        if any(gradient.shape != shape for gradient in gradients):
            raise AxBenchAdvError("GCG member gradients disagree in shape")
        return t.stack(gradients, dim=0).mean(dim=0)

    def rank_candidates(
        self, agg_grad: t.Tensor, current: list[int], cfg: Any
    ) -> dict[int, tuple[TokenReplacement, ...]]:
        padded_gradient = t.cat(
            [t.zeros((1, agg_grad.shape[1]), dtype=agg_grad.dtype), agg_grad], dim=0
        )
        return rank_replacements_from_gradients(
            embedding_gradient=padded_gradient,
            embedding_weight=self.prepared.embedding_weight,
            editable_positions=range(1, len(current) + 1),
            allowed_token_ids=self.prepared.allowed_token_ids,
            current_tokens=self._padded(current),
            top_k=cfg.top_k,
            distance_weight=cfg.distance_weight,
            allowed_embedding_cache=self.cache,
        )

    def sample_candidates(
        self,
        proposals: Mapping[int, Sequence[TokenReplacement]],
        current: list[int],
        *,
        batch_size: int,
        temperature: float,
        rng: random.Random,
        visited: set[str],
    ) -> list[list[int]]:
        sampled = sample_candidate_sequences(
            current_tokens=self._padded(current),
            replacements_by_position=proposals,
            batch_size=batch_size,
            temperature=temperature,
            rng=rng,
            visited_digests=visited,
        )
        candidates = [list(candidate.prompt_tokens[1:]) for candidate in sampled]
        filtered = self.runtime.gcg_filter(candidates)
        accepted: list[list[int]] = []
        for sampled_ids, item in zip(candidates, filtered, strict=True):
            if not item.valid:
                continue
            if tuple(sampled_ids) != item.token_ids or not item.text:
                raise AxBenchAdvError("GCG filter changed a valid sampled candidate")
            try:
                canonical = canonicalize_suffix(
                    item.text,
                    forbidden_tokens=self.prepared.forbidden_control_tokens,
                )
            except AxBenchAdvError:
                continue
            if canonical != item.text:
                raise AxBenchAdvError("GCG filter changed a valid sampled candidate")
            accepted.append(list(item.token_ids))
        return accepted

    def full_batch_pool_exhausted(
        self,
        proposals: Mapping[int, Sequence[TokenReplacement]],
        current: list[int],
        visited: AbstractSet[str],
    ) -> bool:
        return replacement_pool_exhausted(
            current_tokens=self._padded(current),
            replacements_by_position=proposals,
            visited_digests=visited,
        )

    def model_item(self, member: int, target: list[int]) -> tuple[int, tuple[int, ...]]:
        return member, tuple(target)

    def group_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        del validated
        return {}

    def step_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        del validated
        return {}

    def finalize(
        self,
        validated: ValidatedGroup,
        target: list[int],
        *,
        objective: float,
        accepted_steps: int,
        visited_count: int,
    ) -> GroupResult:
        del validated
        filtered = self.runtime.gcg_filter([target])
        if (
            len(filtered) != 1
            or not filtered[0].valid
            or filtered[0].token_ids != tuple(target)
            or len(filtered[0].token_ids) != len(self.prepared.seed_token_ids)
            or not filtered[0].text
            or canonicalize_suffix(
                filtered[0].text,
                forbidden_tokens=self.prepared.forbidden_control_tokens,
            )
            != filtered[0].text
        ):
            raise AxBenchAdvError("retained GCG suffix no longer passes canonical filtering")
        return GroupResult(
            _SearchPayload(
                tuple(target), filtered[0].text, objective, accepted_steps, visited_count
            )
        )


def _training_rows(
    training: ConceptTrainingPairs,
    *,
    master_seed: int,
    evaluation_prompts: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    positive = training.positive
    negative = training.negative
    if not positive or len(positive) != len(negative):
        raise AxBenchAdvError("GCG requires equal nonempty positive and negative populations")
    held_out = set(evaluation_prompts)
    if any(prompt in held_out for prompt, _response in (*positive, *negative)):
        raise AxBenchAdvError("GCG training inputs overlap final evaluation prompts")
    rows: list[dict[str, Any]] = []
    for pair_id, (positive_row, negative_row) in enumerate(zip(positive, negative, strict=True)):
        rows.append(
            {
                "pair_id": pair_id,
                "positive": {
                    "sample_id": derive_selection_seed(
                        master_seed, training.concept_id, 10 + 2 * pair_id
                    ),
                    "prompt": positive_row[0],
                    "response": positive_row[1],
                },
                "negative": {
                    "sample_id": derive_selection_seed(
                        master_seed, training.concept_id, 11 + 2 * pair_id
                    ),
                    "prompt": negative_row[0],
                    "response": negative_row[1],
                },
            }
        )
    return rows, training_pair_identity(training)


def training_pair_identity(training: ConceptTrainingPairs) -> str:
    """Hash exact ordered positive/negative pair content, independent of search knobs."""

    document = {
        "concept_id": training.concept_id,
        "concept": training.concept,
        "genre": training.genre,
        "pairs": [
            {"positive": list(positive), "negative": list(negative)}
            for positive, negative in zip(training.positive, training.negative, strict=True)
        ],
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _score_callback(
    runtime: GcgRuntime,
    items: Sequence[tuple[int, tuple[int, ...]]],
    *,
    device_batch_size: int,
) -> list[float]:
    if not items:
        return []
    candidates: list[tuple[int, ...]] = []
    member_groups: list[list[int]] = []
    for member, tokens in items:
        if not candidates or tokens != candidates[-1]:
            candidates.append(tokens)
            member_groups.append([])
        member_groups[-1].append(member)
    reference = member_groups[0]
    if any(members != reference for members in member_groups):
        raise AxBenchAdvError("GCG objective items are not candidate-major rectangular data")
    return runtime.gcg_score(
        candidates,
        reference,
        candidate_eval_device_batch_size=device_batch_size,
    )


def _gradient_callback(
    runtime: GcgRuntime,
    items: Sequence[tuple[int, tuple[int, ...]]],
) -> list[t.Tensor]:
    if not items:
        raise AxBenchAdvError("GCG gradient batch must not be empty")
    tokens = items[0][1]
    if any(item_tokens != tokens for _member, item_tokens in items):
        raise AxBenchAdvError("GCG gradient batch must share one current suffix")
    return runtime.gcg_gradient(tokens, [member for member, _tokens in items])


def execute_gcg_cell(
    *,
    config: AxBenchAdvGcgConfig,
    runtime: GcgRuntime,
    method: str,
    concept_id: int,
    concept: str,
    factor: float,
    training_pairs: ConceptTrainingPairs,
    evaluation_prompts: Sequence[str],
    evaluation_input_ids: Sequence[int],
    output_dir: Path,
    totals: dict[str, int],
) -> dict[int, Sequence[tuple[int, RawEvaluation]]]:
    """Search every configured positive k and run only retained final suffixes."""

    search_cfg = config.search
    rows, training_identity = _training_rows(
        training_pairs,
        master_seed=int(search_cfg.master_seed),
        evaluation_prompts=evaluation_prompts,
    )
    results: dict[int, Sequence[tuple[int, RawEvaluation]]] = {}
    for k in search_cfg.optimized_token_counts:
        cell_dir = Path(output_dir) / "gcg" / method / f"c{concept_id}" / f"k{k}"
        if k == 0:
            record = {
                "schema_version": 1,
                "objective": config.objective.value,
                "effective_config": config.effective_identity(),
                "training_pair_identity": training_identity,
                "method": method,
                "concept_id": concept_id,
                "concept": concept,
                "factor": factor,
                "token_count": 0,
                "suffix_token_ids": [],
                "suffix": "",
                "diagnostics": {"control": True, "accepted_steps": 0, "visited_sequences": 0},
            }
            atomic_write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", cell_dir / "search.json"
            )
            continue

        prepared = runtime.prepare_gcg(
            objective=config.objective.value,
            alpha=config.alpha,
            rows=rows,
            suffix_token_count=int(k),
            device_batch_size=config.device_batch_size,
            separator=_SEPARATOR,
        )
        if prepared.member_count != len(rows):
            raise AxBenchAdvError("GCG prepared population differs from training pairs")
        if len(prepared.seed_token_ids) != k:
            raise AxBenchAdvError("GCG prepared seed token count differs from requested k")
        space = AppendSuffixEditSpace(runtime, prepared)
        group = _Group(
            key_str=f"{method}|c{concept_id}|k{k}|{config.objective.value}",
            members=tuple(range(len(rows))),
            seed=prepared.seed_token_ids,
        )
        device_batch_size = (
            config.device_batch_size
            if search_cfg.candidate_eval_device_batch_size is None
            else int(search_cfg.candidate_eval_device_batch_size)
        )
        [group_result] = run_gcg(
            edit_space=space,
            groups=[group],
            objective_batch=lambda items, batch_size=device_batch_size: _score_callback(
                runtime, items, device_batch_size=batch_size
            ),
            gradient_provider_batch=lambda items, _step: _gradient_callback(runtime, items),
            cfg=search_cfg,
            full_batch_pool_exhausted=space.full_batch_pool_exhausted,
        )
        payload = group_result.payload
        if not isinstance(payload, _SearchPayload):
            raise AxBenchAdvError("GCG optimizer returned an invalid result payload")
        evaluations = runtime.evaluate(payload.suffix, evaluation_input_ids)
        if len(evaluations) != len(evaluation_input_ids):
            raise AxBenchAdvError("GCG final evaluation changed input cardinality")
        paired = list(zip(evaluation_input_ids, evaluations, strict=True))
        results[int(k)] = paired
        totals["actual_generation_calls"] += len(evaluations)
        totals["actual_evaluator_calls"] += len(evaluations)
        totals["paid_judge_cache_misses"] += sum(
            evaluation.paid_judge_cache_misses for evaluation in evaluations
        )
        record = {
            "schema_version": 1,
            "objective": config.objective.value,
            "effective_config": config.effective_identity(),
            "training_pair_identity": training_identity,
            "model_tokenizer_identity": dict(prepared.identity),
            "layouts_sha256": prepared.layouts_sha256,
            "direction_calibration": prepared.direction_calibration,
            "method": method,
            "concept_id": concept_id,
            "concept": concept,
            "factor": factor,
            "token_count": int(k),
            "suffix_token_ids": list(payload.token_ids),
            "suffix": payload.suffix,
            "objective_value": payload.objective,
            "diagnostics": {
                "control": False,
                "accepted_steps": payload.accepted_steps,
                "visited_sequences": payload.visited_count,
            },
        }
        atomic_write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            cell_dir / "search.json",
        )
    return results


__all__ = [
    "AppendSuffixEditSpace",
    "GcgRuntime",
    "execute_gcg_cell",
    "training_pair_identity",
]
