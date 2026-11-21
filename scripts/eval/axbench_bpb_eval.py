"""Axbench-BPB: offline BPB-based steering eval.

Runs against a completed axbench-base run dir (read-only); writes outputs
to a separate tree. Mirrors scripts/axbench_base_eval.py shape.

Usage (from repo root):

    uv run python -m scripts.eval.axbench_bpb_eval \\
        tiers=dev \\
        source_run_dir=out/eval/axbench_base/<run_tag> \\
        output_root=out/eval/axbench_bpb \\
        feature_dir=... feature_statistics_path=... \\
        decoder_only.source=base decoder_only.model_tag=d13 decoder_only.step=10000 \\
        encdec.source=encdec      encdec.model_tag=d13      encdec.step=10000 \\
        device=cuda

Stage dispatch: ``stage=all`` (default) runs all stages sequentially;
otherwise specify one of: ``prepare_context_cache``, ``bpb_score``,
``bpb_aggregate``. Each stage runs against a completed
axbench-base run dir (read-only); writes outputs to a separate
output_root tree.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.common import compute_cleanup, compute_init
from steering.conditional.discretization import build_boundaries_from_stats
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.method_registry import (
    _build_all_methods,
    filter_methods,
)
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    Concept,
)
from steering.eval.axbench_bpb.bpb_aggregation import (
    compute_macro_rollup,
    compute_per_concept_rollup,
    compute_per_tuple_rollup,
    write_leaderboard_csv,
)
from steering.eval.axbench_bpb.bpb_scoring import run_bpb_score_stage
from steering.eval.axbench_bpb.config import (
    AggregationConfig,
    AxBenchBpbConfig,
    ContextCacheConfig,
    FeatureDiffMeanConfig,
    validate_bpb_config,
)
from steering.eval.axbench_bpb.context_cache import (
    build_context_cache,
    load_context_segments,
    load_manifest,
    write_context_cache_manifest,
)
from steering.eval.axbench_bpb.feature_diffmean_bags import (
    build_feature_diffmean_bags_plain,
    feature_diffmean_modes_for_methods,
)
from steering.eval.axbench_bpb.live_ours_vocab_contract import (
    assert_aggregate_live_ours_vocab_contract,
)
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.sample_loader import (
    collect_unique_prefixes,
    load_raw_synth_rows,
)
from steering.eval.axbench_bpb.scoring_inputs import parquet_row_to_scoring_input
from steering.eval.axbench_bpb.tokenization import (
    assert_bos_lengths_match,
    assert_tokenizer_checksum_matches,
)
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    LoraParams,
    Stage,
    SteeringMode,
)
from steering.eval.segment_mode_guard import require_fixed_feature_dir_segment_mode
from steering.sae_resolver import parse_neuronpedia_source
from steering.stats_io import load_statistics


def _maybe_barrier(world_size: int) -> None:
    """Block until all ranks reach this point. No-op for single-rank runs.

    Required between stages that write shared state and stages that
    read it: rank 0 must not start aggregation while other ranks are
    still flushing parquet shards.
    """
    if world_size <= 1:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.barrier()


def _propagate_rank0_derive(error: Exception | None, *, world_size: int) -> None:
    """Fail loud on ALL ranks when a rank-0-only derive raises.

    The feature-DiffMean bag derive runs on rank 0 only, but the stages that
    follow it are collective. If rank 0 raised inside that region, ranks 1+ would
    otherwise sail past the derive barrier and then DEADLOCK at the next
    collective in ``run_bpb_score_stage`` while rank 0's traceback unwinds (the
    rank-0-raise-before-barrier hang documented on
    :func:`_stage_prepare_context_cache`). Rank 0 therefore captures its error,
    always reaches the barrier, then broadcasts the outcome here so every rank
    aborts together: rank 0 re-raises the original error; ranks 1+ raise a
    summary pointing at rank 0's log.
    """
    payload: list[bool] = [error is not None]
    if world_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise AxBenchBpbError(
                "distributed feature-DiffMean derive requires initialized torch.distributed"
            )
        dist.broadcast_object_list(payload, src=0)
    if error is not None:
        raise error
    if payload[0]:
        raise AxBenchBpbError(
            "rank 0 failed to derive feature-DiffMean conditioning bags; all "
            "ranks aborting (see the rank 0 log for the original error)"
        )


def _device_type_from_cfg(device: str) -> str:
    """Extract the bare device type from ``cfg.device`` for ``compute_init``.

    ``nanochat.common.compute_init`` requires one of ``"cuda" | "cpu" | "mps"``
    (the :class:`DeviceType` ``StrEnum``). Operators routinely pass pinned
    forms like ``"cuda:0"`` on the CLI; this helper strips the optional
    ``":<index>"`` suffix and lowercases so callers can pass the raw cfg
    value through. Mirrors ``scripts/eval/axbench_base_eval._device_type_from_cfg``.

    Raises ``AxBenchBpbError`` for unsupported device strings (e.g. ``"xpu"``)
    so the operator gets a clear domain error instead of an opaque
    ``AssertionError`` deep inside ``compute_init``.
    """
    head = str(device).split(":", 1)[0].strip().lower()
    if head not in ("cuda", "cpu", "mps"):
        raise AxBenchBpbError(
            f"Unsupported cfg.device {device!r}: expected 'cuda', 'cuda:N', 'cpu', or 'mps'."
        )
    return head


def _generate_run_tag(cfg: AxBenchBpbConfig) -> str:
    """Deterministic per-config run tag: ``YYYYMMDD-HHMMSS-<8 hex>``.

    The 8-char hex suffix is a SHA-256 digest of the canonicalized config
    JSON, so two ranks under torchrun (or two re-invocations within the
    same UTC second) produce IDENTICAL tags as long as the config is
    identical. Two runs with different cfgs produce different suffixes.
    Two runs with identical cfgs in different UTC seconds produce
    different timestamps but identical suffixes.

    The previous time-ns-derived hash diverged per-process under
    torchrun, causing each rank to write to its own output tree; the
    parallel-Stage-1 design then silently broke because the manifest
    was visible only on rank 0's tree. Mirrors
    ``scripts/eval/axbench_base_eval._generate_run_tag``.

    Uses ``datetime.now(tz=UTC)`` (not local time) so multi-node
    deployments with mixed timezones still produce identical
    timestamps for the same UTC instant.
    """
    now = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S")
    cfg_payload = dataclasses.asdict(cfg)
    # This optional API/local-model split was added after the legacy run-tag
    # contract. Disabled means absent, not a new explicit null determinant.
    # Keeping the key out preserves the exact historical suffix for every
    # ordinary run while an explicit override still receives a distinct tag.
    if cfg_payload.get("feature_extraction_model_id") is None:
        cfg_payload.pop("feature_extraction_model_id", None)
    cfg_bytes = json.dumps(cfg_payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(cfg_bytes).hexdigest()[:8]
    return f"{now}-{digest}"


def _read_all_parquets(root: Path) -> pa.Table:
    """Read every *.parquet under root (recursively) into a single Table.

    Used by the bpb_aggregate stage to combine rank-local score shards
    into a single table for the aggregation chain.
    """
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise AxBenchBpbError(f"No parquet shards found under {root}")
    tables = [pq.read_table(p) for p in paths]
    return pa.concat_tables(tables, promote_options="default")


def _cfg_to_dataclass(cfg: DictConfig) -> AxBenchBpbConfig:
    """Coerce a Hydra DictConfig to the typed AxBenchBpbConfig dataclass.

    Nested configs (context_cache, aggregation) are lifted out first so
    their own dataclasses can be instantiated cleanly. Any remaining keys
    that don't correspond to ``AxBenchBpbConfig`` fields raise an
    ``AxBenchBpbError`` listing the offending name(s) and the valid
    keys — surfacing typos like ``devicee: cuda`` clearly instead of
    via the opaque ``TypeError`` produced by the dataclass constructor.
    """
    raw = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(raw, dict)
    resolved: dict[str, Any] = {str(k): v for k, v in raw.items()}
    # Strip hydra-internal config block if present
    resolved.pop("hydra", None)

    def nested_mapping(name: str) -> dict[str, Any]:
        value = resolved.pop(name, {})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise AxBenchBpbError(f"{name} must be a config object")
        return {str(key): item for key, item in value.items()}

    cc_dict = nested_mapping("context_cache")
    agg_dict = nested_mapping("aggregation")
    feature_diffmean_dict = nested_mapping("feature_diffmean")

    # Validate: every remaining key must correspond to an AxBenchBpbConfig
    # field. Surfaces typos (e.g. "devicee") as a clear domain error.
    valid_fields = {f.name for f in dataclasses.fields(AxBenchBpbConfig)}
    extras = set(resolved.keys()) - valid_fields
    if extras:
        raise AxBenchBpbError(
            f"Unknown axbench-bpb config keys: {sorted(extras)}. "
            f"Valid keys: {sorted(valid_fields)} (plus 'context_cache', "
            f"'aggregation', 'feature_diffmean', 'hydra')."
        )

    try:
        context_cache = ContextCacheConfig(**cc_dict)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbError(f"invalid context_cache config: {exc}") from exc
    try:
        aggregation = AggregationConfig(**agg_dict)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbError(f"invalid aggregation config: {exc}") from exc
    try:
        feature_diffmean = FeatureDiffMeanConfig(**feature_diffmean_dict)
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbError(f"invalid feature_diffmean config: {exc}") from exc
    try:
        return AxBenchBpbConfig(
            context_cache=context_cache,
            aggregation=aggregation,
            feature_diffmean=feature_diffmean,
            **resolved,
        )
    except (BeartypeCallHintParamViolation, TypeError) as exc:
        raise AxBenchBpbError(f"invalid axbench-bpb config: {exc}") from exc


def _build_samples_by_concept_mode(
    runtime_state: RuntimeState,
    context_segments_by_hash: dict[str, tuple],
    token_bytes: torch.Tensor,
    tokenizer: Any,
    max_seq_len: int,
) -> dict:
    """Read both synth parquets, convert each row to ScoringInputs,
    grouped by (concept_id, mode)."""
    sp = runtime_state.source_paths()
    concept_ids = runtime_state.source_concept_ids()
    pos_rows = load_raw_synth_rows(
        sp.synth_training_positive(),
        filter_unaccepted=runtime_state.cfg.filter_unaccepted_samples,
        require_positive_qc_contract=True,
        expected_continuation_role="stage1b_positive",
        concept_ids=concept_ids,
    )
    neg_rows = load_raw_synth_rows(
        sp.synth_training_negative(),
        filter_unaccepted=runtime_state.cfg.filter_unaccepted_samples,
        expected_continuation_role="stage1c_negative",
        concept_ids=concept_ids,
    )
    out: dict[tuple, list] = {}
    for mode, rows in [
        (SteeringMode.POSITIVE, pos_rows),
        (SteeringMode.NEGATIVE, neg_rows),
    ]:
        for row in rows:
            si = parquet_row_to_scoring_input(
                row,
                mode=mode,
                context_segments_by_hash=context_segments_by_hash,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                token_bytes=token_bytes,
            )
            key = (si.concept_id, mode)
            out.setdefault(key, []).append(si)
    return out


def _require_complete_scoring_population(
    concepts: list[Concept],
    samples_by_concept_mode: dict,
) -> None:
    """Prove the final authority has both score sides in both modes.

    Source preflight performs the same check before any model load. This
    second boundary protects direct stage callers and catches a source changed
    between preflight and scoring. Missing rows are corruption, never a second
    population filter.
    """

    required_classes = (CONCEPT_RICH_LABEL, CONCEPT_AVOIDING_LABEL)
    required_class_set = set(required_classes)
    incomplete: dict[str, dict[int, list[str]]] = {}
    for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
        by_concept: dict[int, list[str]] = {}
        for concept in concepts:
            samples = samples_by_concept_mode.get(
                (concept.concept_id, mode),
                (),
            )
            observed = {
                sample.content_class
                for sample in samples
                if sample.content_class in required_class_set
            }
            missing = [
                content_class for content_class in required_classes if content_class not in observed
            ]
            if missing:
                by_concept[concept.concept_id] = missing
        if by_concept:
            incomplete[mode.value] = by_concept
    if incomplete:
        raise AxBenchBpbError(
            "bpb_score: final concepts_stage1c population lacks required "
            f"scoring classes: {incomplete}; the final concept list is "
            "authoritative and missing source rows cannot be skipped"
        )


def _build_coefficient_grids(
    methods: list,
    target_model_metas: dict,
    cfg: Any,
) -> dict[str, list]:
    """Each method exposes coefficient_grid_fn(meta, cfg); call it to get the list.

    The grid function takes a TargetModelMetadata and AxBenchBaseConfig.
    Our cfg (AxBenchBpbConfig) is structurally compatible for the fields
    the grid functions actually access. Pass it through; type-check
    suppressed at the call site.
    """
    return {
        m.name: list(
            m.coefficient_grid_fn(
                target_model_metas[m.target_model],
                cfg,
            )
        )
        for m in methods
    }


def _build_token_bytes(tokenizer: Any) -> torch.Tensor:
    """Build a (vocab_size,) int64 tensor of byte counts per token id.

    Decodes each token id via tokenizer.decode and uses len(...).encode("utf-8")
    as the byte count. Tokens that fail to decode (typically special / pad
    tokens not appearing in real samples) get byte count 0.
    """
    vocab_size = tokenizer.get_vocab_size()
    weights = torch.zeros(vocab_size, dtype=torch.int64)
    for tid in range(vocab_size):
        try:
            text = tokenizer.decode([tid])
            weights[tid] = len(text.encode("utf-8"))
        except Exception:
            pass
    return weights


def _load_context_segments_by_hash(runtime_state: RuntimeState) -> dict[str, tuple]:
    """Read the context-cache manifest + per-prefix safetensors → mapping."""
    manifest = load_manifest(runtime_state.output_paths())
    out: dict[str, tuple] = {}
    for prefix_hash in manifest:
        cache_file = runtime_state.output_paths().context_cache_file(prefix_hash)
        out[prefix_hash] = load_context_segments(cache_file)
    return out


def _stage_prepare_context_cache(runtime_state: RuntimeState) -> None:
    """Stage 0: build the per-prefix SAE context cache.

    All ranks participate in extraction. Prefix work is sharded
    round-robin by sorted ``prefix_hash`` inside ``build_context_cache``.

    The tokenizer compatibility guards (slice-17 P1.B) run AFTER the
    extraction barrier on rank 0 only. Pre-extraction execution
    deadlocked rank 0 in a 2026-05-13 run: ``assert_bos_lengths_match``
    called ``tokenizer_extraction.get_bos_token_id()`` on an HF
    ``GemmaTokenizer`` (which exposes ``bos_token_id`` as an attribute
    only), raising ``AttributeError``. Under DDP the exception
    propagated into ``compute_cleanup`` → ``dist.destroy_process_group``,
    which then hung on the NCCL teardown because ranks 1-3 were still
    spinning on their post-extraction barrier — the traceback never
    flushed to stderr. The guard itself is fixed (now uses
    ``resolve_bos_token_id``), but moving it post-barrier eliminates
    this class of deadlock: if any future guard raises, the barriers
    have already drained and cleanup can proceed cleanly.

    Sequence:
        1. All ranks: load Gemma+SAE + boundaries + sharded extraction
        2. Barrier (all extraction durable)
        3. Rank 0: write manifest + run tokenizer guards
        4. Barrier (ranks 1+ see manifest before downstream stages)

    See docs/evaluation/axbench-bpb.md:76 for the broader stage contract.
    """
    rank = runtime_state.ddp_rank
    ws = runtime_state.ddp_world_size

    logger.info(f"[stage1 rank {rank}/{ws}] entering stage")
    # Load the small statistics artifact and reject unsupported turn geometry
    # before loading Gemma/SAE, creating the cache directory, or entering any
    # extraction collective. Every rank executes this deterministic preflight.
    stats = load_statistics(runtime_state.cfg.feature_statistics_path)
    require_fixed_feature_dir_segment_mode(
        runtime_state.cfg.feature_dir,
        surface="AxBench-BPB context-cache extraction",
        statistics_segment_mode=stats.metadata.segment_mode,
    )
    # All ranks load Gemma + SAE so they can extract their shard. This
    # is the parallel-Stage-1 design: previously ranks 1+ idled on a
    # barrier for the full duration of Stage 1, wasting GPUs 1..N.
    # Now each rank does ~1/world_size of the work.
    logger.info(f"[stage1 rank {rank}] loading base Gemma + SAE")
    base_model, base_sae = runtime_state.base_gemma_and_sae()
    logger.info(f"[stage1 rank {rank}] base Gemma + SAE loaded")
    # Resolve sae_layer + per-feature bucket boundaries (cheap, in-memory).
    # All ranks do this so they can run extract_prefix_features locally.
    logger.info(f"[stage1 rank {rank}] resolving sae_layer + boundaries")
    _release, _act, sae_layer, _width, _tier = parse_neuronpedia_source(
        runtime_state.cfg.neuronpedia_source
    )
    boundaries = build_boundaries_from_stats(
        stats.per_feature_stats,
        n_sae_features=base_sae.cfg.d_sae,
        num_buckets=runtime_state.cfg.context_cache.num_buckets,
    )
    logger.info(f"[stage1 rank {rank}] boundaries ready")

    # Sharded extraction: each rank handles every world_size-th prefix.
    logger.info(f"[stage1 rank {rank}] starting sharded extraction")
    build_context_cache(
        cfg=runtime_state.cfg,
        output_paths=runtime_state.output_paths(),
        source_positive=runtime_state.source_paths().synth_training_positive(),
        source_negative=runtime_state.source_paths().synth_training_negative(),
        base_model=base_model,
        base_sae=base_sae,
        boundaries=boundaries,
        segment_size=runtime_state.cfg.context_cache.segment_size,
        sae_layer=sae_layer,
        ddp_rank=rank,
        ddp_world_size=ws,
        concept_ids=runtime_state.source_concept_ids(),
    )
    logger.info(f"[stage1 rank {rank}] sharded extraction done, entering barrier 1")

    # Synchronize: all per-rank safetensors files must be durable before
    # rank 0 reads them back to write the manifest.
    _maybe_barrier(ws)
    logger.info(f"[stage1 rank {rank}] passed barrier 1")

    if rank == 0:
        # Under DDP, build_context_cache returns None and skips the manifest
        # write; rank 0 writes it here after the barrier above ensures all
        # ranks' safetensors files exist. For single-process runs (world
        # size 1) build_context_cache already wrote the manifest — we
        # rebuild it here to keep this branch invariant under world_size.
        logger.info("[stage1 rank 0] collecting unique prefixes for manifest")
        unique = collect_unique_prefixes(
            runtime_state.source_paths().synth_training_positive(),
            runtime_state.source_paths().synth_training_negative(),
            filter_unaccepted=runtime_state.cfg.filter_unaccepted_samples,
            concept_ids=runtime_state.source_concept_ids(),
        )
        logger.info(f"[stage1 rank 0] writing manifest ({len(unique)} prefixes)")
        write_context_cache_manifest(
            unique_prefixes=unique,
            output_paths=runtime_state.output_paths(),
        )
        logger.info("[stage1 rank 0] manifest written")

        # Tokenizer compatibility guards (slice-17 P1.B). Run AFTER the
        # barrier — see docstring for the deadlock that motivated this.
        logger.info("[stage1 rank 0] loading decoder_only for tokenizer guards")
        decoder_bundle = runtime_state.decoder_only()
        logger.info("[stage1 rank 0] decoder_only loaded; running checksum guard")
        assert_tokenizer_checksum_matches(
            tokenizer=decoder_bundle.tokenizer,
            feature_dir=runtime_state.cfg.feature_dir,
        )
        logger.info("[stage1 rank 0] loading extraction tokenizer for BOS guard")
        extraction_tokenizer = runtime_state.extraction_tokenizer()
        logger.info("[stage1 rank 0] checksum guard passed; running BOS guard")
        assert_bos_lengths_match(
            tokenizer_extraction=extraction_tokenizer,
            tokenizer_nanochat=decoder_bundle.tokenizer,
        )
        logger.info("[stage1 rank 0] BOS guard passed")

    logger.info(f"[stage1 rank {rank}] entering barrier 2")
    # Second barrier: ranks 1+ must wait for the manifest before bpb_score
    # reads it.
    _maybe_barrier(ws)
    logger.info(f"[stage1 rank {rank}] passed barrier 2; stage complete")


def _stage_bpb_score(runtime_state: RuntimeState) -> None:
    """Stage 3: score every sample × method × coefficient under both targets."""
    decoder_bundle = runtime_state.decoder_only()
    encdec_bundle = runtime_state.encdec()
    context_segments_by_hash = _load_context_segments_by_hash(runtime_state)
    token_bytes = _build_token_bytes(decoder_bundle.tokenizer)

    # Detect target metadata (used by build_intervention for ours_* methods)
    decoder_meta = detect_target_model_metadata(
        decoder_bundle.model,
        runtime_state.cfg,
    )
    encdec_meta = detect_target_model_metadata(
        encdec_bundle.model,
        runtime_state.cfg,
    )
    target_model_metas = {
        "decoder_only": decoder_meta,
        "encdec": encdec_meta,
    }

    # Methods + filters
    all_methods = _build_all_methods()
    methods = filter_methods(
        all_methods,
        enabled_methods=runtime_state.cfg.enabled_methods,
        disabled_methods=runtime_state.cfg.disabled_methods,
        target_model_has_negative_sampling=(encdec_meta.negative_selection is not None),
    )

    concepts: list[Concept] = runtime_state.source_concepts()

    samples_by_concept_mode = _build_samples_by_concept_mode(
        runtime_state,
        context_segments_by_hash,
        token_bytes,
        decoder_bundle.tokenizer,
        decoder_bundle.max_seq_len,
    )
    _require_complete_scoring_population(concepts, samples_by_concept_mode)
    coefficient_grids = _build_coefficient_grids(
        methods,
        target_model_metas,
        runtime_state.cfg,
    )

    feature_diffmean_modes = feature_diffmean_modes_for_methods(methods)
    precomputed_bag_paths = None
    if feature_diffmean_modes:
        precomputed_bag_paths = runtime_state.output_paths()
        # Rank 0 derives the bags; capture any error so rank 0 ALWAYS reaches the
        # barrier below (ranks 1+ are already blocked on it). The failure is then
        # broadcast so every rank aborts loudly together instead of ranks 1+
        # deadlocking at the next collective in run_bpb_score_stage.
        derive_error: Exception | None = None
        if runtime_state.ddp_rank == 0:
            try:
                base_model, base_sae = runtime_state.base_gemma_and_sae()
                build_feature_diffmean_bags_plain(
                    cfg=runtime_state.cfg,
                    output_paths=precomputed_bag_paths,
                    source_paths=runtime_state.source_paths(),
                    methods=methods,
                    model=encdec_bundle.model,
                    base_model=base_model,
                    sae=base_sae,
                    device=torch.device(runtime_state.target_device()),
                    concept_ids=runtime_state.source_concept_ids(),
                )
            except Exception as exc:  # noqa: BLE001 -- reach barrier, then re-raise
                derive_error = exc
        _maybe_barrier(runtime_state.ddp_world_size)
        _propagate_rank0_derive(derive_error, world_size=runtime_state.ddp_world_size)

    lora_params = (
        LoraParams(
            cfg=LoraConfig(**runtime_state.cfg.lora),
            layer=runtime_state.cfg.diffmean_and_probe_layer,
        )
        if runtime_state.cfg.lora
        else None
    )

    run_bpb_score_stage(
        cfg=runtime_state.cfg,
        output_paths=runtime_state.output_paths(),
        source_paths=runtime_state.source_paths(),
        methods=methods,
        concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
        decoder_only_bundle=decoder_bundle,
        encdec_bundle=encdec_bundle,
        target_model_metas=target_model_metas,
        tokenizer_nanochat=decoder_bundle.tokenizer,
        device=torch.device(runtime_state.target_device()),
        ddp_rank=runtime_state.ddp_rank,
        ddp_world_size=runtime_state.ddp_world_size,
        diffmean_layer=runtime_state.cfg.diffmean_and_probe_layer,
        inactive_sampler=runtime_state.inactive_sampler(),
        precomputed_bag_paths=precomputed_bag_paths,
        lora=lora_params,
    )
    if runtime_state.cfg.offload_between_stages:
        runtime_state.offload_models()
    # All ranks must finish flushing rank-local parquet shards before
    # rank 0 transitions to BPB_AGGREGATE and reads the shared output
    # directory. Without this barrier, rank 0 can produce a partial
    # leaderboard built from a subset of shards.
    _maybe_barrier(runtime_state.ddp_world_size)


def _stage_bpb_aggregate(runtime_state: RuntimeState) -> None:
    """Stage 4: roll up conditioned BPB scores and emit leaderboard CSV."""
    # Aggregation runs only on rank 0 to avoid duplicate writes
    if runtime_state.ddp_rank != 0:
        return

    output_paths = runtime_state.output_paths()
    scores_tbl = _read_all_parquets(output_paths.bpb_scores_dir())
    assert_aggregate_live_ours_vocab_contract(
        scores_table=scores_tbl,
        scores_dir=output_paths.bpb_scores_dir(),
    )

    per_tuple = compute_per_tuple_rollup(scores_tbl)
    per_concept = compute_per_concept_rollup(
        per_tuple,
        min_rows_per_side=runtime_state.cfg.aggregation.min_rows_per_side,
    )
    macro, validity_report = compute_macro_rollup(
        per_concept,
        invalid_warning_threshold=runtime_state.cfg.aggregation.invalid_warning_threshold,
        invalid_drop_threshold=runtime_state.cfg.aggregation.invalid_drop_threshold,
    )

    aggregate_dir = output_paths.aggregation_dir()
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(per_tuple, aggregate_dir / "per_tuple.parquet")
    pq.write_table(per_concept, aggregate_dir / "per_concept.parquet")
    pq.write_table(macro, aggregate_dir / "macro.parquet")
    write_leaderboard_csv(
        macro,
        output_paths.leaderboard_csv(),
        primary_metric=runtime_state.cfg.aggregation.primary_metric,
    )

    # Write validity report for diagnostics
    (aggregate_dir / "validity_report.json").write_text(json.dumps(validity_report, indent=2))
    logger.info(
        f"axbench-bpb: leaderboard written to {output_paths.leaderboard_csv()} "
        f"({macro.num_rows} methods)"
    )


_STAGE_HANDLERS: dict[Stage, Callable[[RuntimeState], None]] = {
    Stage.PREPARE_CONTEXT_CACHE: _stage_prepare_context_cache,
    Stage.BPB_SCORE: _stage_bpb_score,
    Stage.BPB_AGGREGATE: _stage_bpb_aggregate,
}


def _run_stage(stage: Stage, runtime_state: RuntimeState) -> None:
    """Dispatch a single stage."""
    logger.info(f"→ Stage: {stage.value}")
    handler = _STAGE_HANDLERS.get(stage)
    if handler is None:
        raise NotImplementedError(f"Unknown stage: {stage}")
    handler(runtime_state)


def _resolve_stages_to_run(stage: Stage) -> list[Stage]:
    """Expand ``stage`` into the explicit ordered stage list.

    ``Stage.ALL`` → every Stage in canonical order; any other Stage member
    → a single-element list. Hoisted out of ``main`` so the Stage.ALL
    expansion is unit-testable without driving the whole CLI.
    """
    if stage == Stage.ALL:
        return [
            Stage.PREPARE_CONTEXT_CACHE,
            Stage.BPB_SCORE,
            Stage.BPB_AGGREGATE,
        ]
    return [stage]


@hydra.main(
    config_path="../../configs/eval/axbench_bpb",
    config_name="axbench_bpb_eval",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Process config FIRST — surface typos, missing artifacts, and run_tag
    # generation BEFORE any distributed initialisation. This ordering
    # mirrors scripts/eval/axbench_base_eval._run and ensures any failure
    # in cfg coercion / validation / run_tag generation cannot leak the
    # process group (it hasn't been initialised yet).
    cfg_obj = _cfg_to_dataclass(cfg)
    validate_bpb_config(cfg_obj)
    if cfg_obj.run_tag is None:
        # Deterministic over cfg → all ranks under torchrun resolve to
        # the same tag and write to the same output tree. Pre-fix this
        # used time.time_ns() which diverged per-process and silently
        # broke parallel Stage 1 (manifest written by rank 0 to a tree
        # ranks 1+ never look at).
        cfg_obj.run_tag = _generate_run_tag(cfg_obj)

    # Now initialise the distributed process group when launched under
    # torchrun. compute_init is a no-op for single-process runs (returns
    # ddp_rank=0, ddp_world_size=1) and only calls dist.init_process_group
    # when the torchrun env vars (RANK/WORLD_SIZE/LOCAL_RANK) are set +
    # device_type=cuda. Without this, dist.is_initialized() below would
    # always return False even under torchrun, silently collapsing all
    # ranks to world_size=1.
    #
    # Use _device_type_from_cfg to strip ``:N`` from pinned forms like
    # ``"cuda:0"`` — compute_init's DeviceType StrEnum only accepts the
    # bare type.
    #
    # compute_init runs INSIDE the try/finally so compute_cleanup fires
    # even if compute_init partially succeeds (e.g., init_process_group
    # succeeds but the post-init dist.barrier() raises). compute_cleanup
    # short-circuits when dist.is_initialized() is False, so calling it
    # on a never-initialized PG is a safe no-op.
    try:
        compute_init(device_type=_device_type_from_cfg(cfg_obj.device))
        if dist.is_available() and dist.is_initialized():
            ddp_rank = dist.get_rank()
            ddp_world_size = dist.get_world_size()
        else:
            ddp_rank, ddp_world_size = 0, 1

        if ddp_rank == 0:
            logger.info(f"axbench-bpb run_tag={cfg_obj.run_tag}")
            logger.info(f"source_run_dir={cfg_obj.source_run_dir}")
            logger.info(f"output_root={cfg_obj.output_root}")
            logger.info(f"stage={cfg_obj.stage}, world_size={ddp_world_size}")

        runtime_state = RuntimeState(
            cfg=cfg_obj,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

        stages_to_run = _resolve_stages_to_run(Stage.coerce(cfg_obj.stage))
        for s in stages_to_run:
            _run_stage(s, runtime_state)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
