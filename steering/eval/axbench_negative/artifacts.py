"""Immutable evidence records and manifest-last negative reference banks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from steering.eval.axbench_negative.api import (
    ApiResponse,
    QualityAssessment,
    canonical_json,
    digest,
    strict_object,
)
from steering.eval.axbench_negative.qc import ChatAssessment
from steering.eval.axbench_negative.source import FeatureTarget, SourceCandidate, value_sha256
from steering.immutable_io import install_bytes_once, stream_hash_file_descriptor_bound

SOURCE_FORMAT = "axbench-negative-source-v1"


def write_record(root: Path, relative: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Install a strict JSON record once, rejecting conflicting replays."""
    path = _safe_path(root, relative)
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    install_bytes_once(payload, path, surface="negative-source evidence")
    return dict(value)


def read_record(root: Path, relative: str) -> dict[str, Any] | None:
    path = _safe_path(root, relative)
    if not path.exists():
        return None
    receipt = stream_hash_file_descriptor_bound(path, retain_payload=True)
    if not isinstance(receipt.payload, bytes):
        raise ValueError("missing retained evidence bytes")
    return strict_object(receipt.payload.decode("utf-8"))


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or "\\" in relative:
        raise ValueError("invalid evidence relative path")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("evidence path escapes its source root")
    return path


def sample_path(sample_id: str) -> str:
    if len(sample_id) != 64 or any(c not in "0123456789abcdef" for c in sample_id):
        raise ValueError("sample_id must be a SHA-256 identity")
    return f"samples/{sample_id[:2]}/{sample_id}.json"


def pair_key(candidate_id: str, feature_id: int) -> str:
    return digest({"candidate_id": candidate_id, "feature_id": feature_id})


@dataclass(frozen=True)
class NegativeReferenceGroup:
    candidate_id: str
    feature_id: int
    member_ids: tuple[int, ...]
    description: str
    source_id: str
    group_id: str
    split: str
    source_row: int
    cut_message_count: int
    messages: tuple[dict[str, str], ...]
    messages_sha256: str
    rich_sample_ids: tuple[str, ...]
    rich_responses: tuple[str, ...]
    avoiding_response: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NegativeReferenceGroup:
        """Decode reference fields; the bank validator checks their evidence bindings."""
        data = dict(value)
        for field in ("member_ids", "messages", "rich_sample_ids", "rich_responses"):
            if not isinstance(data.get(field), (list, tuple)):
                raise ValueError(f"reference group {field} must be a sequence")
            data[field] = tuple(data[field])
        result = cls(**data)
        if not isinstance(result.candidate_id, str):
            raise ValueError("invalid reference candidate ID")
        if type(result.feature_id) is not int or result.feature_id < 0:
            raise ValueError("invalid reference feature ID")
        return result


@dataclass(frozen=True)
class NegativeSourceBank:
    root: Path
    manifest_sha256: str
    identity: dict[str, Any]
    groups: tuple[NegativeReferenceGroup, ...]
    counts: dict[str, int]

    @property
    def references(self) -> tuple[NegativeReferenceGroup, ...]:
        return self.groups


def _require_record(records: Mapping[str, dict[str, Any]], path: str) -> dict[str, Any]:
    if path not in records:
        raise ValueError(f"source manifest lacks required evidence {path}")
    return records[path]


def _validate_all_evidence(
    records: Mapping[str, dict[str, Any]], identity: Mapping[str, Any],
    counts: Mapping[str, int],
) -> tuple[NegativeReferenceGroup, ...]:
    """Validate the entire population once, including rejected pairs and references."""
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("invalid negative-source counts")
    references = _require_record(records, "references.json")
    groups = tuple(NegativeReferenceGroup.from_dict(value) for value in references["groups"])
    group_map = {(group.candidate_id, group.feature_id): group for group in groups}
    if len(group_map) != len(groups):
        raise ValueError("duplicate published reference group")
    if _require_record(records, "contract.json") != {
        "identity": identity, "identity_sha256": digest(identity),
    }:
        raise ValueError("source contract differs from manifest identity")
    source = _require_record(records, "candidates.json")
    if digest(source) != identity["source_population_sha256"]:
        raise ValueError("source population differs from semantic identity")
    config = identity["producer"]
    k, m, cap = (config[key] for key in (
        "n_verification_seeds", "min_consistency_passes", "max_candidates_per_feature"))
    if type(k) is not int or type(m) is not int or not 1 <= m <= k:
        raise ValueError("invalid fixed neutral denominator")
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError("invalid source candidate cap")
    maximum = config["avoid_max_attempts"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("invalid avoidance attempt limit")
    candidate_rows = {row["candidate_id"]: SourceCandidate.from_dict(row).to_dict()
                      for row in source["candidates"]}
    if len(candidate_rows) != len(source["candidates"]):
        raise ValueError("duplicate source candidate IDs")
    targets = {row["feature_id"]: FeatureTarget.from_dict(row).to_dict()
               for row in source["targets"]}
    if not targets or len(targets) != len(source["targets"]):
        raise ValueError("source targets must be nonempty and unique")
    selected: dict[int, list[dict[str, Any]]] = {fid: [] for fid in targets}
    expected_files = {"contract.json", "candidates.json", "references.json"}
    prefix_unknown = 0
    text_splits: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    for candidate in candidate_rows.values():
        split = candidate["split"]
        if family_splits.setdefault(candidate["group_id"], split) != split:
            raise ValueError("source family appears in multiple splits")
        if text_splits.setdefault(candidate["messages_sha256"], split) != split:
            raise ValueError("exact source conversation appears in multiple splits")
        path = f"mining/{candidate['candidate_id']}.json"
        expected_files.add(path)
        mining = _require_record(records, path)
        if mining["candidate_id"] != candidate["candidate_id"]:
            raise ValueError("prefix mining candidate mismatch")
        if mining["status"] == "unknown":
            if mining["assessments"]:
                raise ValueError("unknown prefix has contradictory measurements")
            prefix_unknown += 1
            continue
        if mining["status"] != "assessed" or set(mining["assessments"]) != {str(fid) for fid in targets}:
            raise ValueError("prefix mining lacks complete target assessments")
        for fid in targets:
            assessment = ChatAssessment.from_dict(mining["assessments"][str(fid)])
            if assessment.status == "active" and (cap is None or len(selected[fid]) < cap):
                selected[fid].append(candidate)
    observed = {"candidates": len(source["candidates"]), "targets": len(targets),
                "prefix_unknown": prefix_unknown, "screened_pairs": sum(map(len, selected.values())),
                "neutral_admitted_pairs": 0, "unknown_pairs": 0,
                "rich_quality_rejected_pairs": 0, "avoidance_rejected_pairs": 0,
                "neutral_samples": 0, "final_groups": len(groups)}
    used_samples: set[str] = set()
    expected_groups: set[tuple[str, int]] = set()
    for fid, candidates in selected.items():
        for candidate in candidates:
            pair = pair_key(candidate["candidate_id"], fid)
            dpath = f"decisions/{pair}.json"
            expected_files.add(dpath)
            decision = _require_record(records, dpath)
            sample_ids, assessment_paths = [], []
            hits, unknown = 0, 0
            rich: dict[str, ApiResponse] = {}
            for slot in range(k):
                sid = digest({"format": "negative-neutral-draw-v1",
                              "candidate_id": candidate["candidate_id"],
                              "messages_sha256": candidate["messages_sha256"],
                              "sampler": identity["neutral_api"], "slot": slot})
                sample_ids.append(sid)
                used_samples.add(sid)
                spath = sample_path(sid)
                apath = f"assessments/{pair}_{sid}.json"
                assessment_paths.append(apath)
                expected_files.update((spath, apath))
                sample = _require_record(records, spath)
                if (sample["sample_id"] != sid or sample["slot"] != slot
                        or sample["candidate_id"] != candidate["candidate_id"]
                        or sample["messages_sha256"] != candidate["messages_sha256"]):
                    raise ValueError("neutral sample identity mismatch")
                sampling = dict(identity["neutral_api"]["sampling"])
                if "seed" in sampling:
                    sampling["seed"] = int(digest({"master_seed": sampling["seed"],
                                                   "call_key": sid})[:16], 16) % (2 ** 31)
                if sample["requested_sampling"] != sampling:
                    raise ValueError("neutral requested sampling mismatch")
                response = ApiResponse.from_dict(sample["response"])
                if response.requested_model != identity["neutral_api"]["model"]:
                    raise ValueError("neutral requested model mismatch")
                outcome = _require_record(records, apath)
                if (outcome["sample_id"] != sid or outcome["feature_id"] != fid
                        or outcome["candidate_id"] != candidate["candidate_id"]):
                    raise ValueError("feature assessment identity mismatch")
                measured, hit = outcome["assessment"], False
                if response.status in {"malformed", "transport_error"}:
                    status = "unknown"
                elif response.status in {"empty", "truncated"}:
                    status = "invalid"
                elif measured is not None:
                    status = "assessed"
                    hit = ChatAssessment.from_dict(measured).status == "active"
                else:
                    status = "unknown"
                if (outcome["status"] != status or type(outcome["hit"]) is not bool
                        or outcome["hit"] != hit or (status != "assessed" and measured is not None)):
                    raise ValueError("neutral status/hit contradicts its observation")
                diagnostic = outcome.get("diagnostic_assessment")
                if diagnostic is not None:
                    if status != "invalid":
                        raise ValueError("only terminal invalid draws have separate diagnostic scoring")
                    ChatAssessment.from_dict(diagnostic)
                hits += int(hit)
                unknown += int(status == "unknown")
                if hit:
                    rich[sid] = response
            admitted = unknown == 0 and hits >= m
            expected_decision = {"candidate_id": candidate["candidate_id"], "feature_id": fid,
                                 "k": k, "m": m, "sample_ids": sample_ids,
                                 "assessment_paths": assessment_paths, "hits": hits, "unknown": unknown,
                                 "status": "admitted" if admitted else "incomplete" if unknown else "rejected"}
            if canonical_json(decision) != canonical_json(expected_decision):
                raise ValueError("neutral decision contradicts its fixed-k population")
            observed["unknown_pairs"] += int(unknown > 0)
            if not admitted:
                continue
            observed["neutral_admitted_pairs"] += 1
            useful: dict[str, str] = {}
            for sid, response in rich.items():
                qpath = f"quality/{pair}_{sid}.json"
                expected_files.add(qpath)
                quality_record = _require_record(records, qpath)
                if (quality_record["sample_id"] != sid or quality_record["feature_id"] != fid
                        or quality_record["candidate_id"] != candidate["candidate_id"]):
                    raise ValueError("rich quality identity mismatch")
                if QualityAssessment.from_dict(quality_record["quality"]).useful:
                    if response.refusal or response.content is None:
                        raise ValueError("a provider refusal cannot be a useful reference")
                    useful[sid] = response.content
            if not useful:
                observed["rich_quality_rejected_pairs"] += 1
                continue
            vpath = f"avoidance/{pair}.json"
            expected_files.add(vpath)
            avoidance = _require_record(records, vpath)
            if avoidance["candidate_id"] != candidate["candidate_id"] or avoidance["feature_id"] != fid:
                raise ValueError("avoidance identity mismatch")
            raw_attempts = avoidance["attempts"]
            if (not isinstance(raw_attempts, list) or not 1 <= len(raw_attempts) <= maximum
                    or any(not isinstance(value, dict) for value in raw_attempts)):
                raise ValueError("invalid avoidance attempt population")
            attempts = cast(list[dict[str, Any]], raw_attempts)
            accepted = None
            for index, attempt in enumerate(attempts):
                tpath = f"avoidance_attempts/{pair}_{index}.json"
                expected_files.add(tpath)
                if _require_record(records, tpath) != attempt or attempt["index"] != index:
                    raise ValueError("avoidance attempt evidence mismatch")
                response = ApiResponse.from_dict(attempt["response"])
                if response.requested_model != identity["avoiding_api"]["model"]:
                    raise ValueError("avoidance requested model mismatch")
                call_key = digest({"format": "negative-avoiding-attempt-v1", "pair": pair,
                                   "attempt": index, "api": identity["avoiding_api"],
                                   "template_sha256": identity["avoidance_template_sha256"],
                                   "description": targets[fid]["description"]})
                sampling = dict(identity["avoiding_api"]["sampling"])
                if "seed" in sampling:
                    sampling["seed"] = int(digest({"master_seed": sampling["seed"],
                                                   "call_key": call_key})[:16], 16) % (2 ** 31)
                if attempt["cache_key"] != call_key or attempt["requested_sampling"] != sampling:
                    raise ValueError("avoidance call identity mismatch")
                status = attempt["status"]
                if status not in {"accepted", "rejected", "unknown"}:
                    raise ValueError("invalid avoidance status")
                measured = None if attempt["assessment"] is None else ChatAssessment.from_dict(attempt["assessment"])
                quality = None if attempt["quality"] is None else QualityAssessment.from_dict(attempt["quality"])
                passing = (response.status == "complete" and not response.refusal
                           and measured is not None and measured.status == "inactive"
                           and measured.max_activation == 0 and quality is not None and quality.useful)
                if (status == "accepted") != passing:
                    raise ValueError("avoidance admission contradicts strict-zero/quality evidence")
                is_unknown = (response.status in {"malformed", "transport_error"}
                              or (response.status == "complete" and measured is None)
                              or (measured is not None and measured.status == "inactive"
                                  and measured.max_activation == 0 and quality is None)
                              or (quality is not None and quality.status == "unavailable"))
                if (status == "unknown") != is_unknown:
                    raise ValueError("avoidance unknown status contradicts response/QC evidence")
                if response.status != "complete" and (measured is not None or quality is not None):
                    raise ValueError("invalid avoidance response claims a measurement")
                if quality is not None and (measured is None or measured.status != "inactive"
                                            or measured.max_activation != 0):
                    raise ValueError("avoidance quality judgment precedes strict-zero QC")
                if status in {"accepted", "unknown"} and index != len(attempts) - 1:
                    raise ValueError("avoidance retried a terminal outcome")
                if status == "accepted":
                    accepted = index
            recorded_accepted = avoidance["accepted_attempt"]
            if ((recorded_accepted is not None and type(recorded_accepted) is not int)
                    or recorded_accepted != accepted):
                raise ValueError("avoidance accepted index mismatch")
            if attempts[-1]["status"] == "rejected" and len(attempts) != maximum:
                raise ValueError("avoidance stopped before its declared attempt limit")
            if accepted is None:
                observed["avoidance_rejected_pairs"] += 1
                continue
            key = (candidate["candidate_id"], fid)
            expected_groups.add(key)
            group = group_map.get(key)
            if group is None:
                raise ValueError("accepted references missing from source population")
            expected_reference = {
                **candidate, **targets[fid], "rich_sample_ids": list(useful),
                "rich_responses": list(useful.values()),
                "avoiding_response": attempts[accepted]["response"]["content"],
            }
            # Compare JSON encodings so booleans cannot masquerade as integer metadata.
            if canonical_json(group.to_dict()) != canonical_json(expected_reference):
                raise ValueError("published reference differs from its validated source and responses")
    observed["neutral_samples"] = len(used_samples)
    if observed != counts:
        raise ValueError("source yield counts contradict evidence")
    if expected_groups != set(group_map) or expected_files != set(records):
        raise ValueError("source contains missing or unrelated stage evidence")
    return groups


def publish_negative_source(
    root: Path, *, identity: Mapping[str, Any], groups: Sequence[NegativeReferenceGroup],
    counts: Mapping[str, int],
) -> NegativeSourceBank:
    root = Path(root).resolve()
    write_record(root, "references.json", {"groups": [group.to_dict() for group in groups]})
    files: dict[str, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        receipt = stream_hash_file_descriptor_bound(_safe_path(root, relative), retain_payload=True)
        if not isinstance(receipt.payload, bytes):
            raise ValueError("missing retained source bytes")
        records[relative] = strict_object(receipt.payload.decode("utf-8"))
        files[relative] = {"sha256": receipt.sha256, "bytes": receipt.byte_length}
    validated_groups = _validate_all_evidence(records, identity, counts)
    manifest = {
        "format": SOURCE_FORMAT, "source_kind": "natural_chat_negative", "schema_version": 1,
        "identity": dict(identity), "identity_sha256": digest(dict(identity)),
        "files": files, "groups": len(validated_groups), "counts": dict(counts),
        "model_generations_present": True,
    }
    write_record(root, "manifest.json", manifest)
    return NegativeSourceBank(root, value_sha256(manifest), dict(identity), validated_groups, dict(counts))


def load_negative_source(root: Path) -> NegativeSourceBank:
    root = Path(root).expanduser().resolve()
    manifest = read_record(root, "manifest.json")
    if manifest is None:
        raise ValueError("negative source has no completed manifest")
    if manifest.get("format") != SOURCE_FORMAT or manifest.get("source_kind") != "natural_chat_negative" or manifest.get("schema_version") != 1:
        raise ValueError("unsupported negative-source contract")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or digest(identity) != manifest.get("identity_sha256"):
        raise ValueError("negative-source identity digest mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("invalid negative-source file manifest")
    records: dict[str, dict[str, Any]] = {}
    for relative, expected in files.items():
        path = _safe_path(root, relative)
        receipt = stream_hash_file_descriptor_bound(path, retain_payload=True)
        if receipt.sha256 != expected["sha256"] or receipt.byte_length != expected["bytes"]:
            raise ValueError(f"negative-source file digest mismatch: {relative}")
        if not isinstance(receipt.payload, bytes):
            raise ValueError("missing retained source bytes")
        records[relative] = strict_object(receipt.payload.decode("utf-8"))
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("invalid negative-source counts")
    groups = _validate_all_evidence(records, identity, counts)
    if len(groups) != manifest.get("groups"):
        raise ValueError("negative-source group count mismatch")
    receipt = stream_hash_file_descriptor_bound(root / "manifest.json", retain_payload=True)
    if receipt.payload != (canonical_json(manifest) + "\n").encode("utf-8"):
        raise ValueError("negative-source manifest changed during load")
    return NegativeSourceBank(root, receipt.sha256, identity, groups, counts)
