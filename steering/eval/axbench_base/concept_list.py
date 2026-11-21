"""Stage 0a: build the pre-QC ``candidate_concepts.json`` pool.

Schema note: this artifact now contains pre-LLM-QC rows with
``feature_index``, ``description``, ``corpus_count``, ``layer``,
``sae_source``, and ``description_source``. It intentionally has no
``concept_id`` or judge score; Stage 0c performs examples QC lazily and
assigns final concept IDs. Runs with pre-refactor artifacts must re-run
Stage 0a.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.cluster_artifacts import (
    CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_VERSION,
    install_clustered_bytes_once,
    load_cluster_candidate_identity,
)
from steering.eval.axbench_base.concept_examples_qc import (
    ConceptExamplesQC,
    make_llm_concept_examples_judge,
)
from steering.eval.axbench_base.concept_whitelist import (
    resolve_concept_feature_whitelist,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.synth.response_parser import get_extraction_tokenizer
from steering.eval.axbench_base.types import AxBenchBaseError, CandidateConcept
from steering.eval.axbench_base.validate import validate_candidate_concepts
from steering.eval.cluster_feature_contract import stream_hash_file_stable
from steering.eval.neuronpedia_client import NeuronpediaClient
from steering.feature_paths import parse_feature_dir
from steering.stats_io import load_statistics, load_statistics_and_checksum_from_bytes
from steering.stats_types import ExtractionStatistics

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


def _raise_clustered_underfill(
    cfg: AxBenchBaseConfig,
    *,
    stage: str,
    counts: dict[str, int],
) -> None:
    if counts[stage] >= cfg.max_concepts:
        return
    rendered = ", ".join(f"{name}={value}" for name, value in counts.items())
    raise AxBenchBaseError(
        "Clustered Stage 0a cannot supply the requested concept population: "
        f"underfilled_at={stage}, max_concepts={cfg.max_concepts}, {rendered}. "
        "Enlarge the independent concept whitelist/conditioning mask, lower "
        "min_corpus_count or min_description_length, raise or disable "
        "concept_description_fetch_limit, or lower max_concepts. Refusing to "
        "create downstream API clients or commit source artifacts."
    )


def _prepare_clustered_concept_list(
    cfg: AxBenchBaseConfig,
    contract: ClusterFeatureContract,
) -> Path:
    """Cluster-only Stage 0a branch; the legacy branch below stays unchanged."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before calling prepare_concept_list")
    output = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    provenance_path = paths.AxBenchBasePaths.from_config(
        cfg
    ).candidate_concepts_cluster_provenance_json()
    if output.exists() or provenance_path.exists():
        # The provenance sidecar is the commit marker for the candidate bytes.
        # Validate the complete pair before statistics loading or API creation;
        # a partial/stale pair is never repaired or overwritten in place.
        identity = load_cluster_candidate_identity(
            cfg,
            candidate_path=output,
            expected_fingerprint=contract.feature_contract_fingerprint,
        )
        if identity is None:
            raise AxBenchBaseError(
                "Clustered Stage 0a candidate artifacts were read through an "
                "unclustered configuration boundary"
            )
        # Named BUG-FIX deviation from the oracle (review finding 3, HIGH):
        # the oracle's resume path stops at validating the committed
        # artifact's INTERNAL consistency (fingerprint + sha256 agree) and
        # never re-applies the CURRENT cfg.max_concepts against it. An
        # artifact committed under run_tag=X, max_concepts=2 (2
        # representatives) would then be silently reused by a later run
        # under the SAME run_tag=X with max_concepts=3 -- even though a
        # FRESH run with max_concepts=3 would raise the underfill error
        # below. Resume and fresh-run must agree: re-check the committed
        # pool's actual size against the CURRENT cfg.max_concepts, raising
        # the identical underfill error (same helper, same message shape)
        # a fresh run would. A resume at the SAME or a LOWER max_concepts
        # than the one the artifact was created under still succeeds.
        committed_candidate_count = len(json.loads(output.read_text()))
        _raise_clustered_underfill(
            cfg,
            stage="described_survivors",
            counts={"described_survivors": committed_candidate_count},
        )
        logger.info(
            "Reusing immutable clustered Stage 0a candidates at {}",
            output,
        )
        return output
    if Path(cfg.feature_statistics_path).expanduser().resolve() != Path(
        contract.provenance_dict()["feature_statistics_path"]
    ):
        raise AxBenchBaseError(
            "Clustered Stage 0a feature statistics path differs from preflight"
        )

    statistics_path = Path(cfg.feature_statistics_path).expanduser().resolve()
    expected_statistics_sha256 = contract.statistics.file_sha256
    statistics_digest_before = stream_hash_file_stable(statistics_path)
    if statistics_digest_before.sha256 != expected_statistics_sha256:
        raise AxBenchBaseError(
            "Clustered Stage 0a feature statistics bytes changed after preflight: "
            f"expected_sha256={expected_statistics_sha256}, "
            f"actual_sha256={statistics_digest_before.sha256}"
        )

    # Fused hash+read (see _load_feature_statistics's own docstring for the
    # named BUG-FIX deviation this replaces): the separate post-load
    # stream_hash_file_stable re-check the oracle performed here is now
    # redundant and removed -- its job is subsumed by the fused read's own
    # verification of the exact bytes that get parsed.
    with profile_region("stage0a.load_feature_statistics"):
        stats = _load_feature_statistics(
            cfg.feature_statistics_path,
            expected_sha256=expected_statistics_sha256,
        )
    per_feature = stats.per_feature_stats
    if stats.metadata.n_sae_features != contract.n_sae_features:
        raise AxBenchBaseError(
            "Clustered Stage 0a statistics width changed after preflight: "
            f"{stats.metadata.n_sae_features} != {contract.n_sae_features}"
        )

    source_projection = dict(contract.source_to_representative)
    raw_sources = contract.concept_source_domain
    mask_allowed_sources = tuple(
        source_id for source_id in raw_sources if source_id in source_projection
    )
    unique_representatives = tuple(
        dict.fromkeys(source_projection[source_id] for source_id in mask_allowed_sources)
    )
    outcomes: dict[int, dict[str, Any]] = {}
    members_by_rep = dict(contract.representative_to_allowed_members)
    projecting_by_rep = dict(contract.representative_to_projecting_sources)
    for representative in unique_representatives:
        outcomes[representative] = {
            "representative_id": representative,
            "allowed_member_ids": list(members_by_rep[representative]),
            "projecting_source_ids": list(projecting_by_rep[representative]),
            "statistics_source_id": representative,
            "outcome": "projected",
        }

    representatives_with_stats: list[int] = []
    for representative in unique_representatives:
        if str(representative) not in per_feature:
            outcomes[representative]["outcome"] = "missing_representative_statistics"
            continue
        representatives_with_stats.append(representative)

    count_survivors: list[int] = []
    for representative in representatives_with_stats:
        if int(per_feature[str(representative)].count) < cfg.min_corpus_count:
            outcomes[representative]["outcome"] = "below_min_corpus_count"
            continue
        outcomes[representative]["outcome"] = "count_survivor"
        count_survivors.append(representative)

    counts = {
        "raw_source_ids": len(raw_sources),
        "mask_allowed_source_ids": len(mask_allowed_sources),
        "unique_representatives": len(unique_representatives),
        "representatives_with_statistics": len(representatives_with_stats),
        "count_survivors": len(count_survivors),
    }
    for key in (
        "mask_allowed_source_ids",
        "unique_representatives",
        "representatives_with_statistics",
        "count_survivors",
    ):
        _raise_clustered_underfill(cfg, stage=key, counts=counts)

    fetch_survivors = list(count_survivors)
    if (
        cfg.concept_description_fetch_limit is not None
        and len(fetch_survivors) > cfg.concept_description_fetch_limit
    ):
        rng = random.Random(cfg.concept_list_seed)
        rng.shuffle(fetch_survivors)
        excluded = fetch_survivors[cfg.concept_description_fetch_limit :]
        fetch_survivors = fetch_survivors[: cfg.concept_description_fetch_limit]
        for representative in excluded:
            outcomes[representative]["outcome"] = "excluded_by_description_fetch_limit"
    counts["description_fetch_survivors"] = len(fetch_survivors)
    _raise_clustered_underfill(
        cfg,
        stage="description_fetch_survivors",
        counts=counts,
    )

    # All ID/domain/count failures above happen before the API client exists.
    with profile_region("stage0a.build_neuronpedia_client"):
        client = _get_neuronpedia_client(cfg)
    with profile_region(
        "stage0a.fetch_neuronpedia_descriptions",
        rows=len(fetch_survivors),
    ):
        fid_to_desc = client.get_descriptions(
            fetch_survivors,
            cache_only=cfg.concept_descriptions_cache_only,
            max_not_published_rate=cfg.max_not_published_description_rate,
            max_missing_rate=cfg.max_missing_description_rate,
            max_failed_rate=cfg.max_failed_description_rate,
            min_mismatch_features=cfg.min_source_mismatch_features,
        )

    described: list[tuple[int, str]] = []
    for representative in fetch_survivors:
        raw_description = fid_to_desc.get(representative)
        if raw_description is None:
            outcomes[representative]["outcome"] = "description_unavailable"
            continue
        description = raw_description.strip()
        if len(description) < cfg.min_description_length:
            outcomes[representative]["outcome"] = "description_below_quality_threshold"
            continue
        outcomes[representative]["outcome"] = "candidate_emitted"
        described.append((representative, description))

    counts["described_survivors"] = len(described)
    _raise_clustered_underfill(cfg, stage="described_survivors", counts=counts)
    for name, value in counts.items():
        logger.info("Clustered Stage 0a {}={}", name, value)
        record_counter(f"stage0a.clustered.{name}", value)

    rng = random.Random(cfg.concept_list_seed)
    rng.shuffle(described)
    concepts = [
        CandidateConcept(
            feature_index=representative,
            layer=contract.feature_directory.layer,
            sae_source=cfg.neuronpedia_source,
            description=description,
            corpus_count=int(per_feature[str(representative)].count),
            description_source="neuronpedia",
        )
        for representative, description in described
    ]
    validate_candidate_concepts(concepts, n_sae_features=contract.n_sae_features)

    candidate_payload = [asdict(concept) for concept in concepts]
    candidate_bytes = json.dumps(candidate_payload, indent=2).encode("utf-8")
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    provenance = {
        "schema_version": CLUSTER_CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "candidate_concepts_sha256": candidate_sha256,
        "concept_projection_version": contract.semantic_dict()["algorithms"][
            "concept_projection"
        ],
        "counts": counts,
        "representatives": [outcomes[representative] for representative in unique_representatives],
    }
    provenance_bytes = json.dumps(provenance, indent=2).encode("utf-8")

    # Re-check after the potentially long description fetch so another writer
    # cannot be silently replaced. Exact concurrent/replayed content is a
    # no-op; any different committed pair requires a fresh run tag.
    if output.exists() or provenance_path.exists():
        identity = load_cluster_candidate_identity(
            cfg,
            candidate_path=output,
            expected_fingerprint=contract.feature_contract_fingerprint,
        )
        if identity is None:
            raise AxBenchBaseError(
                "Clustered Stage 0a candidate artifacts were read through an "
                "unclustered configuration boundary"
            )
        if (
            identity.candidate_concepts_sha256 != candidate_sha256
            or identity.candidate_provenance_sha256
            != hashlib.sha256(provenance_bytes).hexdigest()
        ):
            raise AxBenchBaseError(
                "Clustered Stage 0a candidate content changed under an existing "
                "run_tag; use a fresh run_tag instead of overwriting evidence"
            )
    else:
        install_clustered_bytes_once(
            candidate_bytes,
            output,
            surface="Stage 0a candidate concepts",
        )
        install_clustered_bytes_once(
            provenance_bytes,
            provenance_path,
            surface="Stage 0a candidate provenance commit marker",
        )
    logger.info(
        "Wrote {} clustered representative candidates with feature contract {}",
        len(concepts),
        contract.feature_contract_fingerprint,
    )
    return output


def _load_feature_statistics(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ExtractionStatistics:
    """Thin wrapper around ``load_statistics`` so tests can monkey-patch.

    ``expected_sha256`` (clustered Stage 0a only) switches this from the
    plain path-based ``load_statistics`` (legacy branch, unchanged, no
    hash verification at all) to a FUSED read: ``path``'s bytes are read
    exactly once, hash-verified against ``expected_sha256``, and THOSE
    SAME verified bytes are then parsed via
    ``load_statistics_and_checksum_from_bytes`` -- so the returned stats
    are provably built from the bytes that were verified. Named BUG-FIX
    deviation from the oracle (review finding, mirrors C1's Finding-1(b) /
    ``cluster_checkpoint_pin._load_json_mapping``'s fused design): the
    oracle bracketed this same call with two independent
    ``stream_hash_file_stable`` calls (before/after) that only ever
    compared the PATH's digest to a fixed expected value -- since
    before==expected AND after==expected implies before==after by
    transitivity, that bracket is no stronger than a before/after
    snapshot compare and never verified the bytes this function's own
    (previously bare) ``load_statistics`` call actually parsed. A
    concurrent writer round-tripping the file A->B->A entirely inside the
    old load window passed both outer checks while the parsed stats
    silently held B's transient content. Fusing the hash check onto the
    exact bytes about to be parsed closes that gap.
    """
    if expected_sha256 is None:
        return load_statistics(Path(path))
    payload = Path(path).read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise AxBenchBaseError(
            "Clustered Stage 0a feature statistics bytes changed while loading: "
            f"expected_sha256={expected_sha256}, "
            f"actual_sha256={actual_sha256}"
        )
    stats, _percentile_checksum, _feature_ids = load_statistics_and_checksum_from_bytes(payload)
    return stats


def _get_neuronpedia_client(cfg: AxBenchBaseConfig) -> NeuronpediaClient:
    """Thin factory so tests can monkey-patch.

    Forwards ``cfg.neuronpedia`` so per-call retries / concurrency are
    tunable from the top-level config without poking the constructor."""
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    cache_path = output_paths.neuronpedia_cache(
        model_id=cfg.neuronpedia_model_id,
        sae_source=cfg.neuronpedia_source,
    )
    if cfg.concept_descriptions_cache_only and not cache_path.is_file():
        raise AxBenchBaseError(
            "concept_descriptions_cache_only=true requires an existing "
            f"identity-addressed Neuronpedia cache at {cache_path} for "
            f"(model_id={cfg.neuronpedia_model_id!r}, "
            f"sae_source={cfg.neuronpedia_source!r}). Refusing to continue "
            "with an empty cache; validate and copy a matching historical "
            "cache to this exact path or disable cache-only mode."
        )
    client = NeuronpediaClient(
        model_id=cfg.neuronpedia_model_id,
        sae_source=cfg.neuronpedia_source,
        cache_path=cache_path,
        config=cfg.neuronpedia,
    )
    if cfg.concept_descriptions_cache_only and client.cache_size() == 0:
        raise AxBenchBaseError(
            "concept_descriptions_cache_only=true requires at least one "
            f"identity-validated description in {cache_path}; the canonical "
            "cache is blank, empty, or legacy/unverifiable. Populate and "
            "validate this exact identity-addressed cache or disable "
            "cache-only mode."
        )
    return client


def _build_concept_examples_qc_client(
    cfg: AxBenchBaseConfig, response_cache: Any = None,
) -> ConceptExamplesQC:
    """Build the default production QC client. Tests monkey-patch this.

    The extraction tokenizer is required so that Neuronpedia token lists
    are detokenized via ``tokenizer.convert_tokens_to_string``, not via
    ``"".join``. The API-key precondition on ``OpenRouterLLMClient`` is
    re-raised with a clearer error that points at the config toggle.

    ``response_cache`` is forwarded to the underlying
    :class:`OpenRouterLLMClient` so concept-examples judge calls (which
    fire ~once per candidate concept) persist across crashes.

    ``max_workers`` is sourced from ``cfg.neuronpedia.max_concurrent`` so
    the activation-example fetch parallelism stays consistent with the
    rest of the Neuronpedia call sites (description prefetch, etc.) and
    is tunable from one place.
    """
    tokenizer = get_extraction_tokenizer(cfg.extraction_tokenizer_id)
    try:
        llm = OpenRouterLLMClient(
            cfg.concept_examples_qc_judge, response_cache=response_cache,
        )
    except AxBenchBaseError as exc:
        raise AxBenchBaseError(
            f"Cannot construct concept examples QC judge: {exc}. "
            "Set OPENROUTER_API_KEY in the environment, or set "
            "cfg.enable_concept_examples_qc=False to disable the QC step."
        ) from exc
    judge = make_llm_concept_examples_judge(
        llm,
        max_examples_in_prompt=cfg.n_activation_examples,
        model=cfg.concept_examples_qc_judge.model,
    )
    return ConceptExamplesQC(
        model_id=cfg.neuronpedia_model_id,
        sae_source=cfg.neuronpedia_source,
        n_examples=cfg.n_activation_examples,
        judge=judge,
        tokenizer=tokenizer,
        min_score=cfg.activation_example_judge_min_score,
        max_workers=cfg.neuronpedia.max_concurrent,
    )


def prepare_concept_list(
    cfg: AxBenchBaseConfig,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Path:
    """Stage 0a: filter feature stats by corpus count, fetch Neuronpedia
    descriptions, shuffle deterministically, and write the pre-QC
    ``candidate_concepts.json`` pool.

    Stage 0c consumes this full candidate pool, runs examples QC lazily,
    verifies candidates in order, and writes the final ``concepts.json``
    with exactly ``cfg.max_concepts`` accepted concepts. Stage 0a
    intentionally does not truncate to ``cfg.max_concepts``, does not
    call the LLM judge, and does not write ``concepts.json``.

    The extraction ``layer`` stored in each ``CandidateConcept`` is
    derived from ``cfg.feature_dir`` via ``parse_feature_dir`` rather
    than hardcoded.

    Raises ``AxBenchBaseError`` if no features survive either filter.
    """
    if cluster_contract is not None:
        return _prepare_clustered_concept_list(cfg, cluster_contract)

    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling prepare_concept_list"
        )

    parsed_feature_dir = parse_feature_dir(Path(cfg.feature_dir))
    extraction_layer = int(parsed_feature_dir.layer)

    # The SAE model <-> extraction model consistency check (feature_dir's
    # ``m_{model}`` segment must match feature_extraction_model_id /
    # neuronpedia_model_id, issue #657) lives in validate_config so both
    # model-identity guards sit together and fire before any stage runs.

    with profile_region("stage0a.load_feature_statistics"):
        stats = _load_feature_statistics(cfg.feature_statistics_path)
    per_feature = stats.per_feature_stats
    n_sae_features = stats.metadata.n_sae_features
    record_counter("stage0a.per_feature_entries", len(per_feature))
    logger.info(
        f"Loaded feature statistics: n_sae_features={n_sae_features}, "
        f"per_feature_entries={len(per_feature)}"
    )

    whitelist = resolve_concept_feature_whitelist(cfg, n_sae_features=n_sae_features)
    whitelist_set = set(whitelist) if whitelist is not None else None
    if whitelist is not None:
        assert whitelist_set is not None
        stats_feature_ids = {int(fid) for fid in per_feature}
        absent_ids = [fid for fid in whitelist if fid not in stats_feature_ids]
        if absent_ids:
            raise AxBenchBaseError(
                "concept_feature_whitelist_path requested in-range feature IDs "
                "that are absent from per_feature_stats: "
                f"absent_ids={absent_ids}, "
                f"feature_statistics_path={cfg.feature_statistics_path!r}. "
                "An absent ID may be a dead feature that never activated in the "
                "statistics corpus, or the whitelist may target the wrong SAE or "
                "layer. Refusing to create a Neuronpedia client or write the "
                "candidate-concepts artifact."
            )
        n_in_stats = len(whitelist)
        logger.info(
            f"concept_feature_whitelist_path active: {len(whitelist)} IDs, "
            f"{n_in_stats} present in feature statistics"
        )
        record_counter("stage0a.whitelist_size", len(whitelist))
        record_counter("stage0a.whitelist_present_in_stats", n_in_stats)

    with profile_region("stage0a.filter_candidates", rows=len(per_feature)):
        candidates: list[int] = [
            int(fid)
            for fid, fstat in per_feature.items()
            if fstat.count >= cfg.min_corpus_count
            and (whitelist_set is None or int(fid) in whitelist_set)
        ]
    if not candidates:
        if whitelist is not None:
            raise AxBenchBaseError(
                "No whitelisted features survive filtering: "
                f"concept_feature_whitelist_path has {len(whitelist)} IDs, "
                f"{n_in_stats} present in feature statistics, "
                f"and none pass min_corpus_count >= {cfg.min_corpus_count}"
            )
        raise AxBenchBaseError(
            f"No features with corpus count >= {cfg.min_corpus_count}"
        )
    logger.info(f"{len(candidates)} features pass min_corpus_count filter")
    record_counter("stage0a.candidates_after_count_filter", len(candidates))
    if whitelist is not None:
        logger.info(
            "concept_feature_whitelist_path: "
            f"{n_in_stats - len(candidates)} whitelisted IDs "
            f"dropped by min_corpus_count={cfg.min_corpus_count}"
        )
        if len(candidates) < cfg.max_concepts:
            raise AxBenchBaseError(
                f"concept_feature_whitelist_path yields only {len(candidates)} "
                f"candidate(s) "
                f"after min_corpus_count>={cfg.min_corpus_count} "
                f"(whitelist={len(whitelist)}, present-in-stats={n_in_stats}), but "
                f"max_concepts={cfg.max_concepts} concepts are required. Enlarge the "
                f"whitelist, lower min_corpus_count, or lower max_concepts."
            )
    if (
        cfg.concept_description_fetch_limit is not None
        and len(candidates) > cfg.concept_description_fetch_limit
    ):
        rng = random.Random(cfg.concept_list_seed)
        rng.shuffle(candidates)
        candidates = candidates[: cfg.concept_description_fetch_limit]
        logger.info(
            f"Applied concept_description_fetch_limit="
            f"{cfg.concept_description_fetch_limit}; "
            f"fetching descriptions for {len(candidates)} candidates"
        )
        record_counter(
            "stage0a.candidates_after_description_fetch_limit", len(candidates),
        )
    if whitelist is not None and len(candidates) < cfg.max_concepts:
        raise AxBenchBaseError(
            "concept_feature_whitelist_path plus "
            f"concept_description_fetch_limit="
            f"{cfg.concept_description_fetch_limit} leaves only "
            f"{len(candidates)} candidate(s) eligible for description fetch, "
            f"but max_concepts={cfg.max_concepts} concepts are required. "
            "Raise or disable concept_description_fetch_limit, enlarge the "
            "whitelist, or lower max_concepts. Refusing to create a "
            "Neuronpedia client or write the candidate-concepts artifact."
        )

    with profile_region("stage0a.build_neuronpedia_client"):
        client = _get_neuronpedia_client(cfg)
    with profile_region(
        "stage0a.fetch_neuronpedia_descriptions",
        rows=len(candidates),
    ):
        fid_to_desc = client.get_descriptions(
            candidates,
            cache_only=cfg.concept_descriptions_cache_only,
            max_not_published_rate=cfg.max_not_published_description_rate,
            max_missing_rate=cfg.max_missing_description_rate,
            max_failed_rate=cfg.max_failed_description_rate,
            min_mismatch_features=cfg.min_source_mismatch_features,
        )
    n_client_errors = len(candidates) - len(fid_to_desc)
    with profile_region("stage0a.filter_descriptions", rows=len(candidates)):
        described: list[tuple[int, str]] = []
        for fid in candidates:
            raw = fid_to_desc.get(fid)
            if raw is None:
                continue
            desc = raw.strip()
            if len(desc) < cfg.min_description_length:
                continue
            described.append((fid, desc))

    if not described:
        raise AxBenchBaseError(
            f"No features have quality descriptions (min_description_length="
            f"{cfg.min_description_length}, client errors={n_client_errors})"
        )
    logger.info(f"{len(described)} features have valid descriptions")
    record_counter("stage0a.described_candidates", len(described))

    if whitelist is not None:
        logger.info(
            f"concept_feature_whitelist_path: {len(candidates) - len(described)} "
            f"whitelisted IDs dropped during Neuronpedia description fetch/filter"
        )

    if whitelist is not None and len(described) < cfg.max_concepts:
        raise AxBenchBaseError(
            f"concept_feature_whitelist_path yields only {len(described)} "
            f"candidate(s) with "
            f"quality descriptions (min_description_length={cfg.min_description_length}), "
            f"but max_concepts={cfg.max_concepts} concepts are required. Enlarge the "
            f"whitelist, relax the description/count filters, or lower max_concepts."
        )

    rng = random.Random(cfg.concept_list_seed)
    rng.shuffle(described)

    concepts: list[CandidateConcept] = [
        CandidateConcept(
            feature_index=fid,
            layer=extraction_layer,
            sae_source=cfg.neuronpedia_source,
            description=desc,
            corpus_count=int(per_feature[str(fid)].count),
            description_source="neuronpedia",
        )
        for fid, desc in described
    ]

    # Post-build sanity check: catches any structural issue before downstream
    # stages read the file.
    with profile_region("stage0a.validate_candidate_concepts", rows=len(concepts)):
        validate_candidate_concepts(concepts, n_sae_features=n_sae_features)

    out = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    out.parent.mkdir(parents=True, exist_ok=True)
    with profile_region("stage0a.write_candidate_concepts", rows=len(concepts)):
        out.write_text(
            json.dumps([asdict(c) for c in concepts], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    logger.info(f"Wrote {len(concepts)} candidate concepts to {out}")
    record_counter("stage0a.candidate_concepts_written", len(concepts))
    return out
