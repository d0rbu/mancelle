"""Fixed neutral draws followed by separate, bounded avoidance synthesis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from steering.eval.axbench_base.synth.batching import SynthRequest, complete_synth_requests
from steering.eval.axbench_negative.api import (
    AVOIDANCE_INSTRUCTION,
    ApiResponse,
    ChatApiClient,
    ConversationQualityJudge,
    QualityAssessment,
    avoiding_messages,
    canonical_json,
    digest,
    strict_object,
)
from steering.eval.axbench_negative.artifacts import (
    SOURCE_FORMAT,
    NegativeReferenceGroup,
    NegativeSourceBank,
    load_negative_source,
    pair_key,
    publish_negative_source,
    read_record,
    sample_path,
    write_record,
)
from steering.eval.axbench_negative.qc import ChatAssessment, ChatScorer
from steering.eval.axbench_negative.source import (
    FeatureTarget,
    SourceCandidate,
    value_sha256,
)


@dataclass(frozen=True)
class ProducerConfig:
    n_verification_seeds: int = 5
    min_consistency_passes: int = 3
    max_concurrent: int = 8
    avoid_max_attempts: int = 3
    max_candidates_per_feature: int | None = 30

    def __post_init__(self) -> None:
        for name in ("n_verification_seeds", "min_consistency_passes", "max_concurrent",
                     "avoid_max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_consistency_passes > self.n_verification_seeds:
            raise ValueError("min_consistency_passes cannot exceed n_verification_seeds")
        if self.max_candidates_per_feature is not None and (
            type(self.max_candidates_per_feature) is not int or self.max_candidates_per_feature < 1
        ):
            raise ValueError("max_candidates_per_feature must be positive or null")


def neutral_sample_id(candidate: SourceCandidate, api: ChatApiClient, slot: int) -> str:
    """A draw depends on its original conversation and slot, never its features."""
    return digest({"format": "negative-neutral-draw-v1", "candidate_id": candidate.candidate_id,
                   "messages_sha256": candidate.messages_sha256,
                   "sampler": api.identity, "slot": slot})


class _NeutralBatchClient:
    """Adapt Base's bounded scheduler while persisting every returned slot immediately."""

    def __init__(self, api: ChatApiClient, root: Path) -> None:
        self.api, self.root = api, root

    def complete(self, prompt: str, *, cache_key: str | None = None,
                 system_prompt: str | None = None) -> str:
        if cache_key is None or system_prompt is not None:
            raise ValueError("neutral batch requests require only an immutable call key")
        data = strict_object(prompt)
        messages = data.pop("messages")
        if value_sha256(messages) != data["messages_sha256"]:
            raise ValueError("neutral request messages differ from their source identity")
        response = self.api.complete(messages, cache_key=cache_key)
        value = {**data, "sample_id": cache_key, "response": response.to_dict(),
                 "requested_sampling": self.api.requested_sampling(cache_key)}
        write_record(self.root, sample_path(cache_key), value)
        return canonical_json(value)


def _assessment(
    root: Path, candidate: SourceCandidate, target: FeatureTarget, sample: Mapping[str, Any],
    scorer: ChatScorer,
) -> tuple[str, dict[str, Any]]:
    sample_id = sample["sample_id"]
    path = f"assessments/{pair_key(candidate.candidate_id, target.feature_id)}_{sample_id}.json"
    existing = read_record(root, path)
    if existing is not None:
        return path, existing
    response = ApiResponse.from_dict(sample["response"])
    value: dict[str, Any] = {"sample_id": sample_id, "candidate_id": candidate.candidate_id,
                             "feature_id": target.feature_id, "hit": False,
                             "assessment": None, "error": None}
    if response.status in {"malformed", "transport_error"}:
        value.update(status="unknown", error=f"API outcome: {response.status}")
    elif response.status in {"empty", "truncated"}:
        # A provider-invalid draw is a terminal nonhit, never measured inactivity.
        value.update(status="invalid", error=f"API outcome: {response.status}")
        value.update(diagnostic_assessment=None, diagnostic_error=None)
        if response.content is not None:
            try:
                value["diagnostic_assessment"] = scorer.assess_completion(
                    candidate, response.content, target,
                ).to_dict()
            except Exception as exc:
                value["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    else:
        try:
            assert response.content is not None
            measured = scorer.assess_completion(candidate, response.content, target)
            value.update(status="assessed", assessment=measured.to_dict(),
                         hit=measured.status == "active")
        except Exception as exc:
            value.update(status="unknown", error=f"{type(exc).__name__}: {exc}")
    return path, write_record(root, path, value)


def _quality(
    judge: ConversationQualityJudge, candidate: SourceCandidate, response: ApiResponse, key: str,
) -> QualityAssessment:
    assert response.content is not None
    result = judge.assess(candidate.messages, response.content, cache_key=key)
    if response.refusal:
        return QualityAssessment(False, True, result.coherent, result.substantive,
                                 result.task_fulfilled, "provider marked refusal; " + result.reason,
                                 result.status, result.api_response, result.requested_sampling)
    return result


def _avoid(
    root: Path, candidate: SourceCandidate, target: FeatureTarget, scorer: ChatScorer,
    api: ChatApiClient, judge: ConversationQualityJudge, config: ProducerConfig,
) -> dict[str, Any]:
    pair = pair_key(candidate.candidate_id, target.feature_id)
    path = f"avoidance/{pair}.json"
    existing = read_record(root, path)
    if existing is not None:
        return existing
    attempts: list[dict[str, Any]] = []
    accepted = None
    for index in range(config.avoid_max_attempts):
        key = digest({"format": "negative-avoiding-attempt-v1", "pair": pair,
                      "attempt": index, "api": api.identity,
                      "template_sha256": digest(AVOIDANCE_INSTRUCTION),
                      "description": target.description})
        attempt_path = f"avoidance_attempts/{pair}_{index}.json"
        attempt = read_record(root, attempt_path)
        if attempt is None:
            response = api.complete(avoiding_messages(candidate.messages, target.description),
                                    cache_key=key)
            attempt = {"index": index, "cache_key": key, "response": response.to_dict(),
                       "requested_sampling": api.requested_sampling(key),
                       "assessment": None, "quality": None, "status": "rejected", "error": None}
            if response.status in {"transport_error", "malformed"}:
                attempt.update(status="unknown", error=f"API outcome: {response.status}")
            elif response.status == "complete":
                try:
                    assert response.content is not None
                    measured = scorer.assess_completion(candidate, response.content, target)
                    attempt["assessment"] = measured.to_dict()
                    if measured.status == "inactive" and measured.max_activation == 0:
                        quality = _quality(judge, candidate, response, f"avoid-quality:{key}")
                        attempt["quality"] = quality.to_dict()
                        if quality.status == "unavailable":
                            attempt["status"] = "unknown"
                        elif quality.useful:
                            attempt["status"] = "accepted"
                except Exception as exc:
                    attempt.update(status="unknown", error=f"{type(exc).__name__}: {exc}")
            write_record(root, attempt_path, attempt)
        attempts.append(attempt)
        if attempt["status"] == "accepted":
            accepted = index
            break
        if attempt["status"] == "unknown":
            break
    return write_record(root, path, {"candidate_id": candidate.candidate_id,
                                    "feature_id": target.feature_id, "attempts": attempts,
                                    "accepted_attempt": accepted})


def build_negative_source(
    *, candidates: Sequence[SourceCandidate], targets: Sequence[FeatureTarget], scorer: ChatScorer,
    neutral_api: ChatApiClient, avoiding_api: ChatApiClient, quality_judge: ConversationQualityJudge,
    config: ProducerConfig, output_root: Path, semantic_identity: Mapping[str, Any],
) -> NegativeSourceBank:
    """Build/replay a frozen source. UNKNOWN records are never silently remeasured."""
    root = Path(output_root).expanduser().resolve()
    # Canonical target order allows independent concepts to share exactly the same draws.
    targets = tuple(sorted(targets, key=lambda target: target.feature_id))
    source = {"candidates": [candidate.to_dict() for candidate in candidates],
              "targets": [target.to_dict() for target in targets]}
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("duplicate source candidates")
    if not targets or len({target.feature_id for target in targets}) != len(targets):
        raise ValueError("targets must be nonempty and unique")
    family_splits: dict[str, str] = {}
    text_splits: dict[str, str] = {}
    for candidate in candidates:
        if family_splits.setdefault(candidate.group_id, candidate.split) != candidate.split:
            raise ValueError("source family appears in multiple splits")
        if text_splits.setdefault(candidate.messages_sha256, candidate.split) != candidate.split:
            raise ValueError("exact source conversation appears in multiple splits")
    reserved = {"format", "producer", "scorer", "neutral_api", "avoiding_api", "quality_judge",
                "avoidance_template_sha256", "source_population_sha256"}
    if set(semantic_identity).intersection(reserved - {"scorer"}):
        raise ValueError("semantic identity overlaps producer-owned fields")
    if "scorer" in semantic_identity and semantic_identity["scorer"] != scorer.identity:
        raise ValueError("caller scorer identity differs from actual scorer")
    identity = {**dict(semantic_identity), "format": SOURCE_FORMAT,
                "producer": asdict(config), "scorer": scorer.identity,
                "neutral_api": neutral_api.identity, "avoiding_api": avoiding_api.identity,
                "quality_judge": quality_judge.identity,
                "avoidance_template_sha256": digest(AVOIDANCE_INSTRUCTION),
                "source_population_sha256": digest(source)}
    write_record(root, "contract.json", {"identity": identity, "identity_sha256": digest(identity)})
    write_record(root, "candidates.json", source)
    if read_record(root, "manifest.json") is not None:
        bank = load_negative_source(root)
        if bank.identity != identity:
            raise ValueError("completed source identity differs from the requested build")
        return bank

    selected: dict[int, list[SourceCandidate]] = {target.feature_id: [] for target in targets}
    prefix_unknown = 0
    for candidate in candidates:
        path = f"mining/{candidate.candidate_id}.json"
        record: dict[str, Any] | None = read_record(root, path)
        if record is None:
            try:
                measured = scorer.assess_prefix(candidate, targets)
                if set(measured) != set(selected):
                    raise ValueError("prefix scorer returned an incomplete target population")
                record = {"candidate_id": candidate.candidate_id, "status": "assessed",
                          "assessments": {str(fid): value.to_dict() for fid, value in measured.items()}}
            except Exception as exc:
                record = {"candidate_id": candidate.candidate_id, "status": "unknown",
                          "assessments": {}, "error": f"{type(exc).__name__}: {exc}"}
            write_record(root, path, record)
        if record["status"] == "unknown":
            prefix_unknown += 1
            continue
        for target in targets:
            assessment = ChatAssessment.from_dict(record["assessments"][str(target.feature_id)])
            rows = selected[target.feature_id]
            if assessment.status == "active" and (
                config.max_candidates_per_feature is None or len(rows) < config.max_candidates_per_feature
            ):
                rows.append(candidate)

    selected_ids = {candidate.candidate_id for rows in selected.values() for candidate in rows}
    requests: list[SynthRequest] = []
    for candidate in candidates:
        if candidate.candidate_id not in selected_ids:
            continue
        for slot in range(config.n_verification_seeds):
            sample_id = neutral_sample_id(candidate, neutral_api, slot)
            if read_record(root, sample_path(sample_id)) is None:
                requests.append(SynthRequest(
                    index=len(requests), cache_key=sample_id,
                    prompt=canonical_json({"candidate_id": candidate.candidate_id,
                                           "messages_sha256": candidate.messages_sha256,
                                           "slot": slot, "messages": list(candidate.messages)})))
    complete_synth_requests(_NeutralBatchClient(neutral_api, root), requests,
                            max_concurrent=config.max_concurrent,
                            profile_region_name="negative.neutral.request",
                            batch_region_name="negative.neutral.batch")

    groups: list[NegativeReferenceGroup] = []
    counts = {"candidates": len(candidates), "targets": len(targets), "prefix_unknown": prefix_unknown,
              "screened_pairs": sum(map(len, selected.values())), "neutral_admitted_pairs": 0,
              "unknown_pairs": 0, "rich_quality_rejected_pairs": 0, "avoidance_rejected_pairs": 0,
              "neutral_samples": len(selected_ids) * config.n_verification_seeds, "final_groups": 0}
    for target in targets:
        for candidate in selected[target.feature_id]:
            pair = pair_key(candidate.candidate_id, target.feature_id)
            samples: list[dict[str, Any]] = []
            outcomes: list[dict[str, Any]] = []
            paths: list[str] = []
            for slot in range(config.n_verification_seeds):
                sample = read_record(root, sample_path(neutral_sample_id(candidate, neutral_api, slot)))
                if sample is None:
                    raise ValueError("fixed neutral slot was not persisted")
                samples.append(sample)
                path, outcome = _assessment(root, candidate, target, sample, scorer)
                paths.append(path)
                outcomes.append(outcome)
            hits = sum(int(outcome["hit"]) for outcome in outcomes)
            unknown = sum(outcome["status"] == "unknown" for outcome in outcomes)
            admitted = unknown == 0 and hits >= config.min_consistency_passes
            write_record(root, f"decisions/{pair}.json", {
                "candidate_id": candidate.candidate_id, "feature_id": target.feature_id,
                "k": config.n_verification_seeds, "m": config.min_consistency_passes,
                "sample_ids": [sample["sample_id"] for sample in samples], "assessment_paths": paths,
                "hits": hits, "unknown": unknown,
                "status": "admitted" if admitted else "incomplete" if unknown else "rejected"})
            counts["unknown_pairs"] += int(unknown > 0)
            if not admitted:
                continue
            counts["neutral_admitted_pairs"] += 1
            rich_ids: list[str] = []
            rich_responses: list[str] = []
            for sample, outcome in zip(samples, outcomes, strict=True):
                if not outcome["hit"]:
                    continue
                sample_id = sample["sample_id"]
                path = f"quality/{pair}_{sample_id}.json"
                record = read_record(root, path)
                response = ApiResponse.from_dict(sample["response"])
                if record is None:
                    record = write_record(root, path, {
                        "candidate_id": candidate.candidate_id, "feature_id": target.feature_id,
                        "sample_id": sample_id,
                        "quality": _quality(quality_judge, candidate, response,
                                             f"rich-quality:{sample_id}").to_dict()})
                if QualityAssessment.from_dict(record["quality"]).useful:
                    assert response.content is not None
                    rich_ids.append(sample_id)
                    rich_responses.append(response.content)
            if not rich_ids:
                counts["rich_quality_rejected_pairs"] += 1
                continue
            avoidance = _avoid(root, candidate, target, scorer, avoiding_api, quality_judge, config)
            accepted = avoidance["accepted_attempt"]
            if accepted is None:
                counts["avoidance_rejected_pairs"] += 1
                continue
            avoiding = ApiResponse.from_dict(avoidance["attempts"][accepted]["response"])
            assert avoiding.content is not None
            groups.append(NegativeReferenceGroup(
                candidate.candidate_id, target.feature_id, target.member_ids, target.description,
                candidate.source_id, candidate.group_id, candidate.split, candidate.source_row,
                candidate.cut_message_count, candidate.messages, candidate.messages_sha256,
                tuple(rich_ids), tuple(rich_responses), avoiding.content))
    counts["final_groups"] = len(groups)
    return publish_negative_source(root, identity=identity, groups=groups, counts=counts)
